"""HeloX runtime device selection uses deepiri_gpu_utils.torch_device."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch  # noqa: F401
except ImportError:
    torch = None  # type: ignore[misc, assignment]


@unittest.skipUnless(torch is not None, "torch required (poetry install / requirements.txt)")
class TestDeviceManagerGpuUtils(unittest.TestCase):
    @patch("core.gpu_utils.resolve_torch_device")
    def test_detect_device_maps_cpu_and_supported_accelerators(self, mock_resolve) -> None:
        from core.gpu_utils import detect_device

        for device_name in ("cpu", "cuda", "mps"):
            with self.subTest(device=device_name):
                mock_resolve.reset_mock()
                mock_resolve.return_value = SimpleNamespace(
                    device=device_name,
                    notes=[f"detect backend={device_name}"],
                    torch_available=True,
                )

                self.assertEqual(detect_device().type, device_name)
                mock_resolve.assert_called_once_with("auto")

    @patch("core.device_manager.resolve_torch_device")
    def test_auto_uses_resolve_torch_device(self, mock_resolve) -> None:
        from core.device_manager import DeviceManager

        mock_resolve.return_value = SimpleNamespace(
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
