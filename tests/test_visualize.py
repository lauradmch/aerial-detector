"""Unit tests for src/aerial_detector/visualize.py."""
import numpy as np
import pytest

from aerial_detector.visualize import draw_yolo_labels


@pytest.fixture
def blank_image():
    """A 640x640 black BGR image."""
    return np.zeros((640, 640, 3), dtype=np.uint8)


class TestDrawYoloLabels:
    def test_no_crash_on_missing_label(self, blank_image, tmp_path):
        """Missing label file should be a no-op, not raise."""
        img_before = blank_image.copy()
        draw_yolo_labels(blank_image, tmp_path / "does_not_exist.txt",
                         (0, 255, 0), "GT")
        assert np.array_equal(blank_image, img_before)

    def test_draws_box_when_label_present(self, blank_image, tmp_path):
        """A single label line should modify the image (non-zero pixels)."""
        label = tmp_path / "sample.txt"
        # class 0 (airplane), centered, 20% × 20% of the image
        label.write_text("0 0.5 0.5 0.2 0.2\n")
        
        draw_yolo_labels(blank_image, label, (0, 255, 0), "GT")
        
        # Image was all-zero; if a rectangle was drawn, some pixels are now non-zero
        assert blank_image.sum() > 0

    def test_handles_multiple_objects(self, blank_image, tmp_path):
        """Multiple lines in the label file should not raise."""
        label = tmp_path / "multi.txt"
        label.write_text("0 0.2 0.2 0.1 0.1\n1 0.5 0.5 0.15 0.15\n2 0.8 0.8 0.1 0.1\n")
        draw_yolo_labels(blank_image, label, (0, 0, 255), "GT")
        assert blank_image.sum() > 0