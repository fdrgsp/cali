"""Pinned CASCADE catalogue, atomic downloads, and verified model manifests."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import shlex
import stat
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

import yaml

if TYPE_CHECKING:
    from collections.abc import Callable

CATALOGUE_REVISION = "c6978d5ff33edad8792c76e040412f0636913092"
CATALOGUE_SHA256 = "8a1545a44d5b8feec4513e119992eaa81a0b371a37dba9d56afa1addb8dc74b9"
CATALOGUE_URL = (
    "https://raw.githubusercontent.com/PTRRupprecht/CascadeTorch/"
    + CATALOGUE_REVISION
    + "/Pretrained_models/available_models_CascadeTorch.yaml"
)
_MAX_DOWNLOAD_BYTES = 256 * 1024 * 1024
_MAX_EXTRACTED_BYTES = 512 * 1024 * 1024
_WEIGHT_PATTERN = re.compile(r"Model_NoiseLevel_(\d+)_Ensemble_(\d+)\.pth")


class CascadeModelError(ValueError):
    """CASCADE model selection or provenance cannot be verified."""


class CascadeModelNotFound(CascadeModelError):
    """The selected model or pinned catalogue is unavailable locally."""


class CascadeDownloadError(CascadeModelNotFound):
    """A model download failed without publishing an incomplete cache entry."""


class CascadeDownloadCancelled(CascadeDownloadError):
    """An explicitly cancelled download left no incomplete model cache entry."""


def _check_download_cancelled(cancel_requested: Callable[[], bool] | None) -> None:
    if cancel_requested is not None and cancel_requested():
        raise CascadeDownloadCancelled("CASCADE model download cancelled.")


@dataclass(frozen=True)
class CatalogueEntry:
    """A pinned download link with a name-derived rate hint, never a default."""

    name: str
    url: str
    info: str
    sampling_rate_hint: float | None


@dataclass(frozen=True)
class CascadeModel:
    """Verified model identity and authoritative inference metadata."""

    name: str
    directory: Path
    catalogue_revision: str
    manifest_sha256: str
    sampling_rate: float
    smoothing: float
    causal_kernel: bool
    window_size: int
    before_fraction: float
    noise_levels: tuple[int, ...]
    ensemble_size: int
    weight_files: tuple[str, ...]
    config_sha256: str

    @property
    def valid_start(self) -> int:
        """Return the upstream predictor's leading padding in retained frames."""
        return int(self.before_fraction * self.window_size)

    @property
    def trailing_padding(self) -> int:
        """Return the upstream predictor's trailing padding in retained frames."""
        return int((1 - self.before_fraction) * self.window_size)

    @property
    def minimum_frames(self) -> int:
        """Require a complete receptive field and at least one unpadded sample."""
        return max(self.window_size, self.valid_start + self.trailing_padding + 1)

    def valid_interval(self, retained_count: int) -> tuple[int, int]:
        """Resolve the half-open valid interval for a retained trace."""
        if retained_count < self.minimum_frames:
            raise CascadeModelError(
                f"{self.name} requires at least {self.minimum_frames} retained frames; "
                f"received {retained_count}."
            )
        return self.valid_start, retained_count - self.trailing_padding


def cascade_model_dir(model_dir: str | Path | None = None) -> Path:
    """Resolve an explicit cache override without creating directories on import."""
    if model_dir is not None:
        return Path(model_dir).expanduser()
    override = os.environ.get("CALI_CASCADE_MODELS")
    return (
        Path(override).expanduser()
        if override
        else Path.home() / ".cali" / "cascade_models"
    )


def _model_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]*", name):
        raise CascadeModelError(
            "A CASCADE model name must not contain paths or whitespace."
        )
    return name


def _download_command(name: str, root: Path) -> str:
    return (
        f"cali cascade-download {shlex.quote(name)} "
        f"--model-dir {shlex.quote(str(root))}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch_url(
    url: str,
    target: Path,
    limit: int,
    *,
    cancel_requested: Callable[[], bool] | None = None,
) -> None:
    if urlsplit(url).scheme != "https":
        raise CascadeDownloadError("CASCADE downloads require an HTTPS URL.")
    _check_download_cancelled(cancel_requested)
    try:
        deadline = time.monotonic() + 120
        with urlopen(url, timeout=30) as response, target.open("wb") as output:
            if urlsplit(response.geturl()).scheme != "https":
                raise CascadeDownloadError(
                    "The model download redirected away from HTTPS."
                )
            total = 0
            while True:
                _check_download_cancelled(cancel_requested)
                block = response.read(1024 * 1024)
                _check_download_cancelled(cancel_requested)
                if not block:
                    break
                if time.monotonic() > deadline:
                    raise CascadeDownloadError("CASCADE download exceeded 120 seconds.")
                total += len(block)
                if total > limit:
                    raise CascadeDownloadError(
                        "CASCADE download exceeds its size limit."
                    )
                output.write(block)
    except (OSError, URLError) as error:
        _check_download_cancelled(cancel_requested)
        raise CascadeDownloadError(f"CASCADE download failed: {error}") from error


def get_cascade_catalogue(
    model_dir: str | Path | None = None,
    *,
    allow_download: bool = False,
    allow_bundled: bool = False,
    cancel_requested: Callable[[], bool] | None = None,
) -> tuple[CatalogueEntry, ...]:
    """Read pinned metadata, optionally using the offline bundled model list.

    The bundled fallback creates no cache and downloads no model weights. Existing
    cache entries remain authoritative and must pass the same checksum check.
    """
    _check_download_cancelled(cancel_requested)
    root = cascade_model_dir(model_dir)
    path = root / f"catalogue-{CATALOGUE_REVISION}.yaml"
    source = "Cached"
    if not path.is_file():
        if allow_bundled and not allow_download:
            path = Path(__file__).parent / "resources" / "cascade_catalogue.yaml"
            source = "Bundled"
        elif not allow_download:
            raise CascadeModelNotFound(
                "Pinned CASCADE catalogue is not cached. Fetch a model with "
                "cali cascade-download <name>."
            )
        else:
            root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=".catalogue-", dir=root
            ) as temporary:
                staged = Path(temporary) / "catalogue.yaml"
                _fetch_url(
                    CATALOGUE_URL,
                    staged,
                    1024 * 1024,
                    cancel_requested=cancel_requested,
                )
                if _sha256(staged) != CATALOGUE_SHA256:
                    raise CascadeModelError(
                        "Downloaded catalogue differs from the pinned checksum."
                    )
                _check_download_cancelled(cancel_requested)
                os.replace(staged, path)
    if path.stat().st_size > 1024 * 1024 or _sha256(path) != CATALOGUE_SHA256:
        raise CascadeModelError(
            f"{source} CASCADE catalogue differs from the pinned checksum."
        )
    try:
        data = yaml.safe_load(path.read_bytes())
    except yaml.YAMLError as error:
        raise CascadeModelError("Pinned CASCADE catalogue is invalid YAML.") from error
    if not isinstance(data, dict) or not data:
        raise CascadeModelError("Pinned CASCADE catalogue must contain model entries.")
    entries = []
    if any(not isinstance(name, str) for name in data):
        raise CascadeModelError("Invalid CASCADE catalogue model name.")
    for name, value in sorted(data.items()):
        if not isinstance(name, str) or not isinstance(value, dict):
            raise CascadeModelError("Invalid CASCADE catalogue entry.")
        _model_name(name)
        link = value.get("Link")
        if not isinstance(link, str) or urlsplit(link).scheme != "https":
            raise CascadeModelError(f"{name} has no HTTPS model download link.")
        rate_match = re.search(r"_(\d+(?:\.\d+)?)Hz(?:_|$)", name)
        entries.append(
            CatalogueEntry(
                name,
                link,
                str(value.get("Info", "")),
                float(rate_match[1]) if rate_match else None,
            )
        )
    return tuple(entries)


def cascade_model_rate_matches(frame_rate: float, model_rate: float) -> bool:
    """Apply the inference tolerance relative to the authoritative model rate."""
    return (
        math.isfinite(frame_rate)
        and frame_rate > 0
        and math.isfinite(model_rate)
        and model_rate > 0
        and abs(frame_rate - model_rate) / model_rate <= 0.01
    )


def compatible_cascade_models(
    catalogue: tuple[CatalogueEntry, ...], frame_rate: float
) -> tuple[CatalogueEntry, ...]:
    """Filter name-derived rate hints without ever choosing the nearest model."""
    if not math.isfinite(frame_rate) or frame_rate <= 0:
        raise CascadeModelError("A finite positive acquisition rate is required.")
    return tuple(
        entry
        for entry in catalogue
        if entry.sampling_rate_hint is not None
        and cascade_model_rate_matches(frame_rate, entry.sampling_rate_hint)
    )


def _number(cfg: dict[str, Any], key: str, *, allow_zero: bool = False) -> float:
    value = cfg.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CascadeModelError(f"Model config {key} must be numeric.")
    number = float(value)
    if not math.isfinite(number) or (number < 0 if allow_zero else number <= 0):
        raise CascadeModelError(f"Model config {key} must be finite and positive.")
    return number


def _integer(cfg: dict[str, Any], key: str) -> int:
    value = _number(cfg, key)
    if not value.is_integer():
        raise CascadeModelError(f"Model config {key} must be a whole number.")
    return int(value)


def _validated_config(folder: Path, name: str) -> dict[str, Any]:
    path = folder / "config.yaml"
    if path.stat().st_size > 1024 * 1024:
        raise CascadeModelError("Model configuration exceeds its size limit.")
    try:
        cfg = yaml.safe_load(path.read_bytes())
    except yaml.YAMLError as error:
        raise CascadeModelError("Model configuration is invalid YAML.") from error
    if not isinstance(cfg, dict) or cfg.get("model_name") != name:
        raise CascadeModelError(
            "Model configuration does not match the requested model."
        )
    _number(cfg, "sampling_rate")
    _number(cfg, "smoothing", allow_zero=True)
    window = _integer(cfg, "windowsize")
    before = _number(cfg, "before_frac")
    if not 0 < before < 1 or int(before * window) < 1 or int((1 - before) * window) < 1:
        raise CascadeModelError(
            "Model config must leave nonzero leading and trailing padding."
        )
    cfg["ensemble_size"] = _integer(cfg, "ensemble_size")
    noise = cfg.get("noise_levels")
    if (
        not isinstance(noise, list)
        or not noise
        or any(type(value) is not int or value <= 0 for value in noise)
        or len(set(noise)) != len(noise)
    ):
        raise CascadeModelError(
            "Model noise levels must be distinct positive integers."
        )
    if cfg.get("causal_kernel") not in (0, 1, False, True):
        raise CascadeModelError("Model config causal_kernel must be 0 or 1.")
    if len(noise) * cfg["ensemble_size"] > 4999:
        raise CascadeModelError("Model config requests too many ensemble weights.")
    return cfg


def _manifest_body(folder: Path, name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    weights = sorted(folder.glob("*.pth"))
    expected = {
        (noise, ensemble)
        for noise in cfg["noise_levels"]
        for ensemble in range(cfg["ensemble_size"])
    }
    found = set()
    for path in weights:
        match = _WEIGHT_PATTERN.fullmatch(path.name)
        if not match or path.is_symlink():
            raise CascadeModelError(f"Unexpected model weight file: {path.name}.")
        pair = (int(match[1]), int(match[2]))
        if pair in found:
            raise CascadeModelError("Duplicate model noise/ensemble weight.")
        found.add(pair)
    if found != expected:
        raise CascadeModelError(
            "Model weights do not cover the configured noise/ensemble pairs."
        )
    files = []
    for path in [folder / "config.yaml", *weights]:
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise CascadeModelError("Model files must be nonempty regular files.")
        files.append(
            {"path": path.name, "size": path.stat().st_size, "sha256": _sha256(path)}
        )
    return {
        "schema_version": 1,
        "model_name": name,
        "catalogue_revision": CATALOGUE_REVISION,
        "catalogue_sha256": CATALOGUE_SHA256,
        "files": files,
    }


def _manifest_hash(body: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_cascade_model(
    name: str,
    model_dir: str | Path | None = None,
    *,
    expected_manifest: str | None = None,
) -> CascadeModel:
    """Verify every cached config/weight before exposing a model for inference."""
    name = _model_name(name)
    root = cascade_model_dir(model_dir)
    folder = root / name
    command = _download_command(name, root)
    if (
        not folder.is_dir()
        or not (folder / "config.yaml").is_file()
        or not (folder / "manifest.json").is_file()
    ):
        raise CascadeModelNotFound(
            f"Verified CASCADE model {name} is missing. Run: {command}"
        )
    try:
        if (folder / "config.yaml").is_symlink() or (
            folder / "manifest.json"
        ).is_symlink():
            raise CascadeModelError("Model metadata must be regular files.")
        if (folder / "manifest.json").stat().st_size > 1024 * 1024:
            raise CascadeModelError("Model manifest exceeds its size limit.")
        recorded = json.loads((folder / "manifest.json").read_text())
        cfg = _validated_config(folder, name)
        body = _manifest_body(folder, name, cfg)
        digest = _manifest_hash(body)
        if recorded != {**body, "manifest_sha256": digest}:
            raise CascadeModelError(
                "Cached CASCADE model differs from its recorded manifest."
            )
        if expected_manifest is not None and digest != expected_manifest.lower():
            raise CascadeModelError(
                "CASCADE model does not match the required manifest checksum."
            )
    except (OSError, json.JSONDecodeError) as error:
        raise CascadeModelError(f"Cannot verify model {name}: {error}") from error
    return CascadeModel(
        name,
        folder,
        CATALOGUE_REVISION,
        digest,
        float(cfg["sampling_rate"]),
        float(cfg["smoothing"]),
        bool(cfg["causal_kernel"]),
        int(cfg["windowsize"]),
        float(cfg["before_frac"]),
        tuple(cfg["noise_levels"]),
        int(cfg["ensemble_size"]),
        tuple(file["path"] for file in body["files"][1:]),
        body["files"][0]["sha256"],
    )


def _extract_model(
    archive: Path,
    folder: Path,
    name: str,
    *,
    cancel_requested: Callable[[], bool] | None = None,
) -> None:
    _check_download_cancelled(cancel_requested)
    with ZipFile(archive) as zipped:
        members = zipped.infolist()
        if (
            len(members) > 5000
            or sum(info.file_size for info in members) > _MAX_EXTRACTED_BYTES
        ):
            raise CascadeDownloadError("Model archive exceeds its extraction limits.")
        written = set()
        for info in members:
            _check_download_cancelled(cancel_requested)
            # ZipInfo normalizes Windows separators and truncates NUL bytes.
            # Validate the archive's original name before trusting that result.
            raw_name = info.orig_filename
            path = PurePosixPath(raw_name)
            if (
                raw_name != info.filename
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in raw_name
                or ":" in raw_name
                or stat.S_ISLNK(info.external_attr >> 16)
            ):
                raise CascadeDownloadError(
                    "Model archive contains an unsafe file path."
                )
            if info.is_dir():
                continue
            if path.name != "config.yaml" and path.suffix != ".pth":
                continue
            if len(path.parts) > 1 and path.parts != (name, path.name):
                raise CascadeDownloadError(
                    "Model archive has an unexpected model directory."
                )
            if path.name.casefold() in written:
                raise CascadeDownloadError(
                    "Model archive contains duplicate inference files."
                )
            written.add(path.name.casefold())
            with zipped.open(info) as source, (folder / path.name).open("wb") as output:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    _check_download_cancelled(cancel_requested)
                    output.write(block)


def download_cascade_model(
    name: str,
    model_dir: str | Path | None = None,
    *,
    expected_manifest: str | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> CascadeModel:
    """Download and atomically publish a verified model, or reuse its offline cache.

    Cancellation is checked between network reads, archive entries and before
    publishing. An in-flight network read retains the existing 30-second timeout.
    """
    name = _model_name(name)
    _check_download_cancelled(cancel_requested)
    root = cascade_model_dir(model_dir)
    target = root / name
    if target.exists():
        return load_cascade_model(name, root, expected_manifest=expected_manifest)
    try:
        catalogue = get_cascade_catalogue(
            root, allow_download=True, cancel_requested=cancel_requested
        )
    except CascadeDownloadCancelled:
        raise
    except CascadeDownloadError as error:
        raise CascadeDownloadError(
            f"{error} Retry with: {_download_command(name, root)}"
        ) from error
    entry = next((value for value in catalogue if value.name == name), None)
    if entry is None:
        raise CascadeModelError(f"Model {name} is not in the pinned CASCADE catalogue.")
    from cali.logger import cali_logger

    cali_logger.info(f"📦 Downloading CASCADE model {name}...")
    with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=root) as temporary:
        staging = Path(temporary)
        archive = staging / "model.zip"
        folder = staging / name
        folder.mkdir()
        try:
            _fetch_url(
                entry.url,
                archive,
                _MAX_DOWNLOAD_BYTES,
                cancel_requested=cancel_requested,
            )
        except CascadeDownloadCancelled:
            raise
        except CascadeDownloadError as error:
            raise CascadeDownloadError(
                f"{error} Retry with: {_download_command(name, root)}"
            ) from error
        try:
            _extract_model(archive, folder, name, cancel_requested=cancel_requested)
            cfg = _validated_config(folder, name)
            body = _manifest_body(folder, name, cfg)
            digest = _manifest_hash(body)
            (folder / "manifest.json").write_text(
                json.dumps({**body, "manifest_sha256": digest}, indent=2) + "\n"
            )
            verified = load_cascade_model(
                name, staging, expected_manifest=expected_manifest
            )
            _check_download_cancelled(cancel_requested)
            try:
                folder.rename(target)
            except OSError as error:
                if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
                existing = load_cascade_model(
                    name, root, expected_manifest=verified.manifest_sha256
                )
                return existing
        except (OSError, BadZipFile) as error:
            raise CascadeDownloadError(
                f"Cannot unpack/verify CASCADE model {name}: {error}"
            ) from error
    return load_cascade_model(name, root, expected_manifest=expected_manifest)
