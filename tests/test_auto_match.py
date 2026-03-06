"""Tests for auto_match_files and auto_match_files_grouped in cali.gui._util."""

from pathlib import Path

from cali.gui._util import auto_match_files, auto_match_files_grouped


class TestAutoMatchFiles:
    """Tests for the 1:1 auto_match_files function."""

    def test_exact_stem_match(self, tmp_path: Path) -> None:
        files = [tmp_path / "A1_0000.tif", tmp_path / "A1_0001.tif"]
        targets = ["A1_0000", "A1_0001"]
        result = auto_match_files(files, targets)
        assert result == {"A1_0000": files[0], "A1_0001": files[1]}

    def test_target_as_substring(self, tmp_path: Path) -> None:
        files = [tmp_path / "A1_0000_labels.tif", tmp_path / "experiment_B5_0001.tif"]
        targets = ["A1_0000", "B5_0001"]
        result = auto_match_files(files, targets)
        assert result == {"A1_0000": files[0], "B5_0001": files[1]}

    def test_no_match(self, tmp_path: Path) -> None:
        files = [tmp_path / "image_001.tif"]
        targets = ["A1_0000"]
        result = auto_match_files(files, targets)
        assert result == {}

    def test_word_boundary_prevents_false_match(self, tmp_path: Path) -> None:
        """A1 should NOT match A10_0000."""
        files = [tmp_path / "A10_0000.tif"]
        targets = ["A1"]
        result = auto_match_files(files, targets)
        assert result == {}

    def test_word_boundary_allows_correct_match(self, tmp_path: Path) -> None:
        """A1 should match file_A1_data.tif (bounded by non-alnum)."""
        files = [tmp_path / "file_A1_data.tif"]
        targets = ["A1"]
        result = auto_match_files(files, targets)
        assert result == {"A1": files[0]}

    def test_longest_target_matched_first(self, tmp_path: Path) -> None:
        """A1_0000 should be matched before A1 when both are targets."""
        f = tmp_path / "A1_0000_labels.tif"
        files = [f]
        targets = ["A1", "A1_0000"]
        result = auto_match_files(files, targets)
        # A1_0000 is longer, should claim the file
        assert result == {"A1_0000": f}

    def test_ambiguous_match_skipped(self, tmp_path: Path) -> None:
        """If a target matches multiple unclaimed files, skip it."""
        files = [tmp_path / "A1_0000_v1.tif", tmp_path / "A1_0000_v2.tif"]
        targets = ["A1_0000"]
        result = auto_match_files(files, targets)
        assert result == {}

    def test_each_file_used_at_most_once(self, tmp_path: Path) -> None:
        f = tmp_path / "A1_0000.tif"
        files = [f]
        # Both targets could match, but A1_0000 is longer and claims the file
        targets = ["A1_0000", "A1"]
        result = auto_match_files(files, targets)
        assert result == {"A1_0000": f}
        assert "A1" not in result

    def test_empty_inputs(self, tmp_path: Path) -> None:
        assert auto_match_files([], ["A1"]) == {}
        assert auto_match_files([tmp_path / "A1.tif"], []) == {}
        assert auto_match_files([], []) == {}


class TestAutoMatchFilesGrouped:
    """Tests for the many:1 auto_match_files_grouped function."""

    def test_multiple_files_per_target(self, tmp_path: Path) -> None:
        files = [
            tmp_path / "A1_fov0.tif",
            tmp_path / "A1_fov1.tif",
            tmp_path / "B2_fov0.tif",
        ]
        targets = ["A1", "B2"]
        result = auto_match_files_grouped(files, targets)
        assert set(result["A1"]) == {files[0], files[1]}
        assert result["B2"] == [files[2]]

    def test_longest_target_wins(self, tmp_path: Path) -> None:
        """File containing A1_0000 should match A1_0000, not A1."""
        f = tmp_path / "A1_0000_data.tif"
        files = [f]
        targets = ["A1", "A1_0000"]
        result = auto_match_files_grouped(files, targets)
        assert "A1_0000" in result
        assert "A1" not in result

    def test_word_boundary(self, tmp_path: Path) -> None:
        files = [tmp_path / "A10_0000.tif"]
        targets = ["A1", "A10"]
        result = auto_match_files_grouped(files, targets)
        assert "A1" not in result
        assert result["A10"] == [files[0]]

    def test_no_match(self, tmp_path: Path) -> None:
        files = [tmp_path / "random_image.tif"]
        targets = ["A1", "B2"]
        result = auto_match_files_grouped(files, targets)
        assert result == {}

    def test_tie_skipped(self, tmp_path: Path) -> None:
        """If two targets of the same length match, skip that file."""
        # Both "AB" and "CD" are length 2
        files = [tmp_path / "AB_CD_data.tif"]
        targets = ["AB", "CD"]
        result = auto_match_files_grouped(files, targets)
        assert result == {}

    def test_empty_inputs(self, tmp_path: Path) -> None:
        assert auto_match_files_grouped([], ["A1"]) == {}
        assert auto_match_files_grouped([Path("A1.tif")], []) == {}
        assert auto_match_files_grouped([], []) == {}
