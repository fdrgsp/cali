"""Tests for auto_match_files and auto_match_files_grouped in cali.gui._util."""

from pathlib import Path

import pytest

from cali.gui._util import auto_match_files, auto_match_files_grouped

# ---------------------------------------------------------------------------
# auto_match_files (1:1 matching)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filenames, targets, expected_keys",
    [
        # exact stem match
        (
            ["A1_0000.tif", "A1_0001.tif"],
            ["A1_0000", "A1_0001"],
            {"A1_0000": "A1_0000.tif", "A1_0001": "A1_0001.tif"},
        ),
        # target as substring
        (
            ["A1_0000_labels.tif", "experiment_B5_0001.tif"],
            ["A1_0000", "B5_0001"],
            {"A1_0000": "A1_0000_labels.tif", "B5_0001": "experiment_B5_0001.tif"},
        ),
        # word boundary allows correct match
        (
            ["file_A1_data.tif"],
            ["A1"],
            {"A1": "file_A1_data.tif"},
        ),
        # longest target matched first
        (
            ["A1_0000_labels.tif"],
            ["A1", "A1_0000"],
            {"A1_0000": "A1_0000_labels.tif"},
        ),
    ],
    ids=["exact", "substring", "boundary-ok", "longest-first"],
)
def test_auto_match_files_matches(
    tmp_path: Path,
    filenames: list[str],
    targets: list[str],
    expected_keys: dict[str, str],
) -> None:
    files = [tmp_path / f for f in filenames]
    result = auto_match_files(files, targets)
    expected = {k: tmp_path / v for k, v in expected_keys.items()}
    assert result == expected


@pytest.mark.parametrize(
    "filenames, targets",
    [
        # no match at all
        (["image_001.tif"], ["A1_0000"]),
        # word boundary prevents false match (A1 should NOT match A10_0000)
        (["A10_0000.tif"], ["A1"]),
        # ambiguous: target matches two unclaimed files
        (["A1_0000_v1.tif", "A1_0000_v2.tif"], ["A1_0000"]),
    ],
    ids=["no-match", "boundary-blocks", "ambiguous"],
)
def test_auto_match_files_no_match(
    tmp_path: Path, filenames: list[str], targets: list[str]
) -> None:
    files = [tmp_path / f for f in filenames]
    assert auto_match_files(files, targets) == {}


def test_auto_match_files_each_file_used_once(tmp_path: Path) -> None:
    """A1_0000 (longer) claims the file; A1 gets nothing."""
    f = tmp_path / "A1_0000.tif"
    result = auto_match_files([f], ["A1_0000", "A1"])
    assert result == {"A1_0000": f}
    assert "A1" not in result


@pytest.mark.parametrize(
    "files, targets",
    [([], ["A1"]), ([Path("A1.tif")], []), ([], [])],
    ids=["no-files", "no-targets", "both-empty"],
)
def test_auto_match_files_empty_inputs(files: list[Path], targets: list[str]) -> None:
    assert auto_match_files(files, targets) == {}


# ---------------------------------------------------------------------------
# auto_match_files_grouped (many:1 matching)
# ---------------------------------------------------------------------------


def test_auto_match_grouped_multiple_files_per_target(tmp_path: Path) -> None:
    files = [
        tmp_path / "A1_fov0.tif",
        tmp_path / "A1_fov1.tif",
        tmp_path / "B2_fov0.tif",
    ]
    result = auto_match_files_grouped(files, ["A1", "B2"])
    assert set(result["A1"]) == {files[0], files[1]}
    assert result["B2"] == [files[2]]


def test_auto_match_grouped_longest_target_wins(tmp_path: Path) -> None:
    f = tmp_path / "A1_0000_data.tif"
    result = auto_match_files_grouped([f], ["A1", "A1_0000"])
    assert "A1_0000" in result
    assert "A1" not in result


def test_auto_match_grouped_word_boundary(tmp_path: Path) -> None:
    files = [tmp_path / "A10_0000.tif"]
    result = auto_match_files_grouped(files, ["A1", "A10"])
    assert "A1" not in result
    assert result["A10"] == [files[0]]


@pytest.mark.parametrize(
    "filenames, targets",
    [
        # no match
        (["random_image.tif"], ["A1", "B2"]),
        # tie: both targets same length
        (["AB_CD_data.tif"], ["AB", "CD"]),
    ],
    ids=["no-match", "tie-skipped"],
)
def test_auto_match_grouped_no_match(
    tmp_path: Path, filenames: list[str], targets: list[str]
) -> None:
    files = [tmp_path / f for f in filenames]
    assert auto_match_files_grouped(files, targets) == {}


@pytest.mark.parametrize(
    "files, targets",
    [([], ["A1"]), ([Path("A1.tif")], []), ([], [])],
    ids=["no-files", "no-targets", "both-empty"],
)
def test_auto_match_grouped_empty_inputs(files: list[Path], targets: list[str]) -> None:
    assert auto_match_files_grouped(files, targets) == {}
