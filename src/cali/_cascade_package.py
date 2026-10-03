"""Lazy access to the immutable, separately licensed CASCADE dependency."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

CASCADE_PACKAGE_REVISION = "95ea3cac2d5719583f4a9eb03d68c78bcbc20d08"
CASCADE_PACKAGE_URL = (
    "git+https://github.com/fdrgsp/CascadeTorch@" + CASCADE_PACKAGE_REVISION
)
_SOURCE_SHA256 = {
    "__init__.py": "5715bf0c3e855e571f4b56c15b360e02960f41624c8b1b3d7b875a7162bdafb0",
    "cascade.py": "c246857e0f5285eca147a301c987274469fccb71da7d656e25054542e62c961e",
    "checks.py": "2a7561d1c76bf61b4613fed5bba730615d066b94d1faeefbdfee13f52df75c06",
    "config.py": "c4ffdd1057493dcb4d458d7bc708ce0fe34b1337fe97f3d4aeb53be452af829e",
    "utils.py": "123e628696595ffdd35fc0d9eb797a6d13af3cd78d1f612102c1be0ed26167fb",
    "utils_discrete_spikes.py": (
        "7d99baa43b704173cdf129582019b45c34f35fb557c49ce196f0690a6f2bbcfe"
    ),
}
_INSTALL_MESSAGE = "Install cali[cascade] to use CASCADE spike inference"


@dataclass(frozen=True)
class CascadePackage:
    """Verified upstream APIs and exact source provenance for a backend call."""

    cascade: ModuleType
    config: ModuleType
    utils: ModuleType
    torch: ModuleType
    package_version: str
    package_revision: str
    source_manifest_sha256: str


def load_cascade_package() -> CascadePackage:
    """Load optional modules only on use, verifying the pinned source contents.

    Wheels retain identical source provenance even without ``direct_url.json``.
    Normalize Git's Windows line endings before hashing; no model code is copied
    into cali. An unavailable or modified installation never selects OASIS instead.
    """
    try:
        package_version = version("CascadeTorch")
        package = import_module("cascade2p")
    except (ImportError, PackageNotFoundError) as error:
        raise ImportError(_INSTALL_MESSAGE) from error
    if package_version != "2.0" or not package.__file__:
        raise ImportError(f"{_INSTALL_MESSAGE}; the pinned package is required")
    folder = Path(package.__file__).parent
    try:
        for name, expected_hash in _SOURCE_SHA256.items():
            contents = (folder / name).read_bytes().replace(b"\r\n", b"\n")
            if hashlib.sha256(contents).hexdigest() != expected_hash:
                raise ImportError(
                    f"{_INSTALL_MESSAGE}; {name} differs from pinned commit "
                    f"{CASCADE_PACKAGE_REVISION}"
                )
    except OSError as error:
        raise ImportError(
            f"{_INSTALL_MESSAGE}; the installed package is incomplete"
        ) from error
    try:
        cascade = import_module("cascade2p.cascade")
        config = import_module("cascade2p.config")
        utils = import_module("cascade2p.utils")
        torch = import_module("torch")
    except ImportError as error:
        raise ImportError(
            f"{_INSTALL_MESSAGE}; a required dependency is unavailable"
        ) from error
    for module, names in (
        (cascade, ("predict",)),
        (config, ("read_config",)),
        (utils, ("calculate_noise_levels", "define_model")),
    ):
        if any(not callable(getattr(module, name, None)) for name in names):
            raise ImportError(
                f"{_INSTALL_MESSAGE}; an upstream inference API is missing"
            )
    manifest = json.dumps(_SOURCE_SHA256, sort_keys=True, separators=(",", ":"))
    return CascadePackage(
        cascade,
        config,
        utils,
        torch,
        package_version,
        CASCADE_PACKAGE_REVISION,
        hashlib.sha256(manifest.encode()).hexdigest(),
    )
