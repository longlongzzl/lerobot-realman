from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from rm75_app.perception.tabletop_roi import crop_polygon_roi, normalize_five_point_polygon


class TabletopRoiTest(unittest.TestCase):
    def test_accepts_five_ordered_points(self):
        points = normalize_five_point_polygon(
            [[0.1, 0.2], [0.8, 0.1], [0.95, 0.6], [0.6, 0.9], [0.15, 0.8]]
        )
        self.assertEqual(len(points), 5)

    def test_rejects_crossed_polygon(self):
        with self.assertRaisesRegex(ValueError, "交叉"):
            normalize_five_point_polygon(
                [[0.1, 0.1], [0.9, 0.9], [0.9, 0.1], [0.1, 0.9], [0.5, 0.95]]
            )

    def test_crops_polygon_bounding_box_and_reports_offset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            Image.new("RGB", (100, 80), (255, 255, 255)).save(source)
            result = crop_polygon_roi(
                source,
                [[0.2, 0.2], [0.8, 0.2], [0.9, 0.5], [0.7, 0.8], [0.2, 0.7]],
                root / "crop.jpg",
            )
            self.assertEqual(result["offset_x"], 20)
            self.assertEqual(result["offset_y"], 16)
            self.assertTrue(Path(result["image_path"]).is_file())
            self.assertLess(result["crop_width"], 100)


if __name__ == "__main__":
    unittest.main()
