from __future__ import annotations

import unittest

from rm75_app.perception.remote_vlm_provider import _extract_json_object, inventory_to_sam3_items, normalize_inventory


class RemoteVLMProviderTest(unittest.TestCase):
    def test_extracts_fenced_json(self):
        result = _extract_json_object('```json\n{"objects": []}\n```')
        self.assertEqual(result, {"objects": []})

    def test_normalizes_qwen_thousand_coordinate_boxes(self):
        result = normalize_inventory(
            {
                "objects": [
                    {
                        "temporary_id": "obj_01",
                        "noun_phrase": "orange carrot",
                        "aliases": ["carrot"],
                        "bbox_2d": [100, 200, 600, 800],
                        "confidence": 0.9,
                    }
                ]
            }
        )
        self.assertEqual(result["objects"][0]["bbox_normalized"], [0.1, 0.2, 0.6, 0.8])

    def test_builds_sam3_text_box_item(self):
        inventory = normalize_inventory(
            {"objects": [{"noun_phrase": "cup", "bbox_normalized": [0.1, 0.2, 0.6, 0.8]}]}
        )
        item = inventory_to_sam3_items(inventory, 640, 480)[0]
        self.assertEqual(item["mode"], "text_box")
        self.assertEqual(item["box"], [64.0, 96.0, 384.0, 384.0])

    def test_maps_cropped_inventory_back_to_full_frame(self):
        inventory = normalize_inventory(
            {"objects": [{"noun_phrase": "carrot", "bbox_normalized": [0.1, 0.2, 0.6, 0.8]}]}
        )
        item = inventory_to_sam3_items(inventory, 300, 200, offset_x=120, offset_y=80)[0]
        self.assertEqual(item["box"], [150.0, 120.0, 300.0, 240.0])


if __name__ == "__main__":
    unittest.main()
