"""Unit tests for src/aerial_detector/data.py."""

import numpy as np
import pytest

from aerial_detector.data import filter_and_remap, obb_to_hbb


class TestObbToHbb:
    def test_axis_aligned_rectangle(self):
        """An axis-aligned rectangle should convert to itself."""
        points = np.array(
            [
                [0.2, 0.3],
                [0.6, 0.3],
                [0.6, 0.7],
                [0.2, 0.7],
            ]
        )
        cx, cy, w, h = obb_to_hbb(points)
        assert cx == pytest.approx(0.4)
        assert cy == pytest.approx(0.5)
        assert w == pytest.approx(0.4)
        assert h == pytest.approx(0.4)

    def test_rotated_square_bbox_larger(self):
        """A 45°-rotated square should have an AABB larger than the original side."""
        # Square of side 0.2 centered at (0.5, 0.5), rotated 45°
        s = 0.1  # half-diagonal
        points = np.array(
            [
                [0.5, 0.5 - s],  # top
                [0.5 + s, 0.5],  # right
                [0.5, 0.5 + s],  # bottom
                [0.5 - s, 0.5],  # left
            ]
        )
        cx, cy, w, h = obb_to_hbb(points)
        assert cx == pytest.approx(0.5)
        assert cy == pytest.approx(0.5)
        assert w == pytest.approx(0.2)  # diagonal of the square
        assert h == pytest.approx(0.2)


class TestFilterAndRemap:
    def test_keeps_and_remaps_target_classes(self):
        p = np.zeros((4, 2))
        objects = [(0, p), (5, p), (13, p), (18, p)]
        mapping = {0: 0, 13: 1, 18: 2}
        result = filter_and_remap(objects, mapping)
        assert [cid for cid, _ in result] == [0, 1, 2]

    def test_empty_input(self):
        assert filter_and_remap([], {0: 0}) == []

    def test_no_target_classes(self):
        p = np.zeros((4, 2))
        assert filter_and_remap([(5, p), (7, p)], {0: 0}) == []
