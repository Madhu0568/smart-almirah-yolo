import tempfile
import unittest
from pathlib import Path

import app as app_module
from app import app, init_db, record_detections


class SmartAlmirahAPITests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        app.config["TESTING"] = True
        app.config["DATABASE_PATH"] = str(Path(self.temp_dir.name) / "test.db")
        app.config["VOICE_ENABLED"] = False
        init_db()
        self.client = app.test_client()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dashboard_is_served(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Smart Almirah", response.data)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_inventory_starts_empty(self):
        response = self.client.get("/api/inventory")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["total_count"], 0)

    def test_detection_requires_image_or_running_camera(self):
        response = self.client.post("/api/detect")
        self.assertEqual(response.status_code, 400)
        self.assertIn("Choose an image", response.get_json()["error"])

    def test_stop_camera_releases_capture_device(self):
        class FakeCapture:
            def __init__(self):
                self.released = False

            def release(self):
                self.released = True

        fake_capture = FakeCapture()
        app_module._camera_capture = fake_capture
        app_module._camera_thread = None
        app_module._camera_started_at = "2026-11-21T00:00:00+00:00"
        app_module._camera_running.set()

        app_module.stop_camera_resources()

        self.assertTrue(fake_capture.released)
        self.assertFalse(app_module._camera_running.is_set())
        self.assertIsNone(app_module._camera_capture)
        self.assertIsNone(app_module._camera_thread)
        self.assertIsNone(app_module._camera_started_at)

    def test_config_accepts_valid_values(self):
        response = self.client.put(
            "/api/config",
            json={"confidence_threshold": 0.75, "image_size": 768, "voice_enabled": False},
        )
        self.assertEqual(response.status_code, 200)
        config = response.get_json()["config"]
        self.assertEqual(config["confidence_threshold"], 0.75)
        self.assertEqual(config["image_size"], 768)
        self.assertFalse(config["voice_enabled"])

    def test_config_rejects_invalid_image_size(self):
        response = self.client.put("/api/config", json={"image_size": 750})
        self.assertEqual(response.status_code, 400)

    def test_config_rejects_out_of_range_threshold(self):
        response = self.client.put(
            "/api/config",
            json={"confidence_threshold": 1.5},
        )
        self.assertEqual(response.status_code, 400)

    def test_config_rejects_non_boolean_voice_setting(self):
        response = self.client.put(
            "/api/config",
            json={"voice_enabled": "yes"},
        )
        self.assertEqual(response.status_code, 400)

    def test_detection_updates_persistent_inventory_and_log(self):
        record_detections([
            {
                "item_id": "detection-one",
                "category": "shirt",
                "color": "blue",
                "confidence": 0.82,
                "bounding_box": {"x": 10, "y": 20, "width": 30, "height": 40},
                "inference_time_ms": 15.5,
                "detected_at": "2026-11-20T10:00:00+00:00",
            },
            {
                "item_id": "detection-two",
                "category": "shirt",
                "color": "blue",
                "confidence": 0.91,
                "bounding_box": {"x": 11, "y": 21, "width": 31, "height": 41},
                "inference_time_ms": 16.5,
                "detected_at": "2026-11-20T10:01:00+00:00",
            },
        ])

        inventory = self.client.get("/api/inventory").get_json()
        self.assertEqual(inventory["total_items"], 1)
        self.assertEqual(inventory["total_count"], 2)
        self.assertEqual(inventory["items"][0]["confidence"], 0.91)

        log = self.client.get("/api/detection-log").get_json()
        self.assertEqual(log["total"], 2)
        self.assertEqual(len(log["entries"]), 2)

        stats = self.client.get("/api/stats").get_json()
        self.assertEqual(stats["total_detections"], 2)
        self.assertEqual(stats["unique_items"], 1)

    def test_inventory_item_can_be_removed(self):
        record_detections([
            {
                "item_id": "detection-delete",
                "category": "jacket",
                "color": "black",
                "confidence": 0.88,
                "bounding_box": {"x": 0, "y": 0, "width": 10, "height": 10},
                "inference_time_ms": 20.0,
                "detected_at": "2026-11-20T10:00:00+00:00",
            }
        ])
        item_id = self.client.get("/api/inventory").get_json()["items"][0]["item_id"]
        response = self.client.delete("/api/inventory/" + item_id)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get("/api/inventory").get_json()["total_count"], 0)

    def test_missing_inventory_item_returns_404(self):
        response = self.client.delete("/api/inventory/not-a-real-id")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
