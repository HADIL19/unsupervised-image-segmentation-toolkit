import numpy as np
import pytest

from imgseg.common.io_utils import ensure_dir, largest_component, read_rgb


def test_read_rgb_missing_file_raises_clear_error(tmp_path):
    missing = tmp_path / "nope.png"
    with pytest.raises(FileNotFoundError, match="Could not read image"):
        read_rgb(str(missing))


def test_largest_component_picks_the_bigger_blob():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[1:3, 1:3] = 255       # small blob, area 4
    mask[10:18, 10:18] = 255   # large blob, area 64
    result = largest_component(mask)
    assert result[1:3, 1:3].sum() == 0
    assert result[10:18, 10:18].sum() == 255 * 8 * 8


def test_largest_component_empty_mask_returns_empty():
    mask = np.zeros((10, 10), dtype=np.uint8)
    result = largest_component(mask)
    assert result.sum() == 0
    assert result.shape == mask.shape


def test_ensure_dir_creates_and_returns_path(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    returned = ensure_dir(str(target))
    assert returned == str(target)
    assert target.is_dir()
