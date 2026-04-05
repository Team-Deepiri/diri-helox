"""DeviceManager uses deepiri_gpu_utils.torch_device when installed."""
from __future__ import annotations

import unittest
from unittest.mock import patch

try:
    import torch  # noqa: F401
except ImportError:
    torch = None  # type: ignore[misc, assignment]


@unittest.skipUnless(torch is not None, "torch required (poetry install / requirements.txt)")
class TestDeviceManagerGpuUtils(unittest.TestCase):
    @patch("core.device_manager.resolve_torch_device")
    def test_auto_uses_resolve_torch_device(self, mock_resolve) -> None:
        from core.device_manager import DeviceManager
        from deepiri_gpu_utils.torch_device import DeviceDecision

        mock_resolve.return_value = DeviceDecision(
            device="cpu", notes=["detect backend=cpu"], torch_available=True
        )
        dm = DeviceManager()
        self.assertEqual(dm.get_device().type, "cpu")
        mock_resolve.assert_called_once_with("auto")

    def test_force_cpu(self) -> None:
        from core.device_manager import DeviceManager

        dm = DeviceManager(force_device="cpu")
        self.assertEqual(dm.get_device().type, "cpu")


if __name__ == "__main__":
    unittest.main()
