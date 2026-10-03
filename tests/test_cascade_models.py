"""Offline tests for model selection, provenance, downloads, and CLI dispatch."""

from __future__ import annotations

import hashlib
import io
import json
import stat
from typing import TYPE_CHECKING, Any
from urllib.error import URLError
from zipfile import ZipFile, ZipInfo

import pytest
import yaml

from cali import _cascade_models as models
from cali.__main__ import main

if TYPE_CHECKING:
    from pathlib import Path

NAME = "Global_EXC_10Hz_smoothing200ms"


def _config(**changes: Any) -> bytes:
    return yaml.safe_dump(
        {
            "model_name": NAME,
            "sampling_rate": 10,
            "smoothing": 0.2,
            "causal_kernel": 0,
            "windowsize": 64,
            "before_frac": 0.5,
            "noise_levels": [2, 3],
            "ensemble_size": 2,
            **changes,
        }
    ).encode()


def _archive(
    config: bytes | None = None, extra: tuple[str | ZipInfo, bytes] | None = None
) -> bytes:
    stream = io.BytesIO()
    with ZipFile(stream, "w") as archive:
        archive.writestr(f"{NAME}/config.yaml", config or _config())
        for noise in (2, 3):
            for ensemble in (0, 1):
                archive.writestr(
                    f"{NAME}/Model_NoiseLevel_{noise}_Ensemble_{ensemble}.pth",
                    f"test weight {noise}:{ensemble}".encode(),
                )
        if extra:
            archive.writestr(*extra)
    return stream.getvalue()


@pytest.fixture
def fake_download(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    catalogue = yaml.safe_dump(
        {
            NAME: {"Link": "https://example.test/model.zip", "Info": "test"},
            "Other_30Hz_test": {"Link": "https://example.test/other.zip"},
            "Unknown_rate": {"Link": "https://example.test/unknown.zip"},
        }
    ).encode()
    state: dict[str, Any] = {"catalogue": catalogue, "archive": _archive(), "calls": []}
    monkeypatch.setattr(
        models, "CATALOGUE_SHA256", hashlib.sha256(catalogue).hexdigest()
    )

    def fetch(url: str, target: Path, limit: int) -> None:
        state["calls"].append(url)
        data = state["catalogue"] if url == models.CATALOGUE_URL else state["archive"]
        assert len(data) <= limit
        target.write_bytes(data)

    monkeypatch.setattr(models, "_fetch_url", fetch)
    return state


def test_missing_model_is_offline_and_actionable(
    tmp_path: Path, fake_download: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cache with spaces"
    monkeypatch.setenv("CALI_CASCADE_MODELS", str(root))
    assert models.cascade_model_dir() == root
    assert models.cascade_model_dir(tmp_path) == tmp_path
    with pytest.raises(
        models.CascadeModelNotFound, match="cali cascade-download"
    ) as err:
        models.load_cascade_model(NAME)
    assert f"--model-dir '{root}'" in str(err.value)
    with pytest.raises(models.CascadeModelNotFound, match="catalogue"):
        models.get_cascade_catalogue(root)
    assert not root.exists()
    assert fake_download["calls"] == []


def test_atomic_download_manifest_and_offline_reuse(
    tmp_path: Path, fake_download: dict[str, Any]
) -> None:
    model = models.download_cascade_model(NAME, tmp_path)
    assert model.sampling_rate == 10
    assert model.smoothing == 0.2
    assert model.noise_levels == (2, 3)
    assert len(model.weight_files) == 4
    assert model.catalogue_revision == models.CATALOGUE_REVISION
    assert model.minimum_frames == 65
    assert model.valid_interval(65) == (32, 33)
    assert model.valid_interval(100) == (32, 68)
    with pytest.raises(models.CascadeModelError, match="65 retained"):
        model.valid_interval(64)
    manifest = json.loads((model.directory / "manifest.json").read_text())
    assert manifest["files"][0]["path"] == "config.yaml"
    assert manifest["files"][0]["sha256"] == model.config_sha256
    assert [entry["path"] for entry in manifest["files"][1:]] == list(
        model.weight_files
    )
    assert manifest["manifest_sha256"] == model.manifest_sha256
    assert models.download_cascade_model(NAME, tmp_path) == model
    assert len(fake_download["calls"]) == 2
    assert not list(tmp_path.glob(".*"))
    with pytest.raises(models.CascadeModelError, match="required manifest"):
        models.load_cascade_model(NAME, tmp_path, expected_manifest="0" * 64)


def test_rate_choices_never_choose_nearest(
    tmp_path: Path, fake_download: dict[str, Any]
) -> None:
    catalogue = models.get_cascade_catalogue(tmp_path, allow_download=True)
    assert [
        entry.name for entry in models.compatible_cascade_models(catalogue, 10)
    ] == [NAME]
    assert models.compatible_cascade_models(catalogue, 12) == ()
    assert models.compatible_cascade_models(catalogue, 10.05)[0].name == NAME
    assert models.compatible_cascade_models(catalogue, 10.2) == ()
    with pytest.raises(models.CascadeModelError, match="positive acquisition"):
        models.compatible_cascade_models(catalogue, float("nan"))
    with pytest.raises(models.CascadeModelError, match="not in the pinned"):
        models.download_cascade_model("Other_12Hz_test", tmp_path)
    assert len(fake_download["calls"]) == 1


@pytest.mark.parametrize("name", ["../model", "/absolute", "bad name", "", ".hidden"])
def test_invalid_selection_never_downloads(
    name: str, tmp_path: Path, fake_download: dict[str, Any]
) -> None:
    with pytest.raises(models.CascadeModelError, match="paths or whitespace"):
        models.download_cascade_model(name, tmp_path)
    assert fake_download["calls"] == []


@pytest.mark.parametrize("damage", ["weight", "config", "missing", "extra", "manifest"])
def test_every_load_verifies_model_and_refuses_overwrite(
    damage: str, tmp_path: Path, fake_download: dict[str, Any]
) -> None:
    model = models.download_cascade_model(NAME, tmp_path)
    weight = model.directory / model.weight_files[0]
    if damage == "weight":
        weight.write_bytes(b"modified")
    elif damage == "config":
        (model.directory / "config.yaml").write_bytes(_config(smoothing=0.3))
    elif damage == "missing":
        weight.unlink()
    elif damage == "extra":
        (model.directory / "Model_NoiseLevel_4_Ensemble_0.pth").write_bytes(b"extra")
    else:
        (model.directory / "manifest.json").write_text("{}")
    for action in (models.load_cascade_model, models.download_cascade_model):
        with pytest.raises(models.CascadeModelError):
            action(NAME, tmp_path)
    assert len(fake_download["calls"]) == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"sampling_rate": 0},
        {"sampling_rate": float("nan")},
        {"smoothing": -1},
        {"windowsize": 2.5},
        {"before_frac": 1},
        {"noise_levels": [2, 2]},
        {"noise_levels": [True]},
        {"ensemble_size": 1},
        {"ensemble_size": 5000},
        {"causal_kernel": "false"},
        {"model_name": "wrong"},
    ],
)
def test_invalid_config_never_publishes(
    changes: dict[str, Any], tmp_path: Path, fake_download: dict[str, Any]
) -> None:
    fake_download["archive"] = _archive(_config(**changes))
    with pytest.raises(models.CascadeModelError):
        models.download_cascade_model(NAME, tmp_path)
    assert not (tmp_path / NAME).exists()
    assert not list(tmp_path.glob(".*"))


@pytest.mark.parametrize(
    "path",
    [
        "../escape",
        "/absolute",
        "foo\\escape",
        "C:escape",
        "wrong/config.yaml",
        "config.yaml",
    ],
)
def test_unsafe_or_duplicate_archive_never_publishes(
    path: str, tmp_path: Path, fake_download: dict[str, Any]
) -> None:
    fake_download["archive"] = _archive(extra=(path, b"bad"))
    with pytest.raises(models.CascadeDownloadError):
        models.download_cascade_model(NAME, tmp_path)
    assert not (tmp_path / NAME).exists()
    assert not list(tmp_path.glob(".*"))


def test_symlink_archive_rejected(
    tmp_path: Path, fake_download: dict[str, Any]
) -> None:
    link = ZipInfo("link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    fake_download["archive"] = _archive(extra=(link, b"/elsewhere"))
    with pytest.raises(models.CascadeDownloadError, match="unsafe"):
        models.download_cascade_model(NAME, tmp_path)
    assert not list(tmp_path.glob(".*"))


def test_bad_zip_and_extraction_limit(
    tmp_path: Path, fake_download: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_download["archive"] = b"not a zip"
    with pytest.raises(models.CascadeDownloadError, match="unpack/verify"):
        models.download_cascade_model(NAME, tmp_path)
    fake_download["archive"] = _archive()
    monkeypatch.setattr(models, "_MAX_EXTRACTED_BYTES", 1)
    with pytest.raises(models.CascadeDownloadError, match="extraction limits"):
        models.download_cascade_model(NAME, tmp_path)
    assert not (tmp_path / NAME).exists()
    assert not list(tmp_path.glob(".*"))


def test_catalogue_checksum_failure_cleanup(
    tmp_path: Path, fake_download: dict[str, Any]
) -> None:
    fake_download["catalogue"] = b"changed catalogue"
    root = tmp_path / "cache"
    with pytest.raises(models.CascadeModelError, match="pinned checksum"):
        models.download_cascade_model(NAME, root)
    assert list(root.iterdir()) == []


def test_required_manifest_failure_cleanup(
    tmp_path: Path, fake_download: dict[str, Any]
) -> None:
    with pytest.raises(models.CascadeModelError, match="required manifest"):
        models.download_cascade_model(NAME, tmp_path, expected_manifest="0" * 64)
    assert not (tmp_path / NAME).exists()
    assert not list(tmp_path.glob(".*"))


@pytest.mark.parametrize("error", [KeyboardInterrupt, models.CascadeDownloadError])
def test_interrupted_download_cleanup(
    error: type[BaseException],
    tmp_path: Path,
    fake_download: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models.get_cascade_catalogue(tmp_path, allow_download=True)

    def fail(url: str, target: Path, limit: int) -> None:
        target.write_bytes(b"partial")
        raise error("interrupted")

    monkeypatch.setattr(models, "_fetch_url", fail)
    with pytest.raises(error):
        models.download_cascade_model(NAME, tmp_path)
    assert not (tmp_path / NAME).exists()
    assert not list(tmp_path.glob(".*"))


def test_cli_reports_manifest_and_failure_exit(
    tmp_path: Path, fake_download: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    main(["cascade-download", NAME, "--model-dir", str(tmp_path)])
    output = capsys.readouterr().out
    assert f"Model: {NAME}" in output
    assert "Sampling rate: 10 Hz" in output
    digest = models.load_cascade_model(NAME, tmp_path).manifest_sha256
    assert f"Manifest SHA-256: {digest}" in output
    main(
        [
            "cascade-download",
            NAME,
            "--model-dir",
            str(tmp_path),
            "--expected-manifest",
            digest,
        ]
    )
    assert len(fake_download["calls"]) == 2
    with pytest.raises(SystemExit) as err:
        main(["cascade-download", "Missing", "--model-dir", str(tmp_path)])
    assert err.value.code == 1
    assert "not in the pinned" in capsys.readouterr().err
    with pytest.raises(SystemExit) as err:
        main(["cascade-download"])
    assert err.value.code == 2


def test_fetch_requires_https(tmp_path: Path) -> None:
    with pytest.raises(models.CascadeDownloadError, match="HTTPS"):
        models._fetch_url("http://example.test/model", tmp_path / "bad", 100)


@pytest.mark.parametrize("redirect", [False, True])
def test_fetch_bounds_stream_and_redirects(
    redirect: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Response(io.BytesIO):
        def geturl(self) -> str:
            return "http://bad.test" if redirect else "https://good.test"

    def open_url(url: str, timeout: int) -> Response:
        assert timeout == 30
        return Response(b"too many bytes")

    monkeypatch.setattr(models, "urlopen", open_url)
    with pytest.raises(models.CascadeDownloadError, match=r"HTTPS|size limit"):
        models._fetch_url("https://good.test", tmp_path / "target", 1)


def test_offline_cli_reports_actionable_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def offline(url: str, timeout: int) -> None:
        raise URLError("offline")

    monkeypatch.setattr(models, "urlopen", offline)
    with pytest.raises(models.CascadeModelNotFound, match="cali cascade-download"):
        models.download_cascade_model(NAME, tmp_path)
    with pytest.raises(SystemExit) as err:
        main(["cascade-download", NAME, "--model-dir", str(tmp_path)])
    assert err.value.code == 1
    assert "offline" in capsys.readouterr().err
    assert not (tmp_path / NAME).exists()
    assert not list(tmp_path.glob(".*"))
