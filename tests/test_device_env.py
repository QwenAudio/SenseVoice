import unittest

from utils.device_env import resolve_sensevoice_device


class ResolveSensevoiceDeviceTest(unittest.TestCase):
    def test_explicit_cpu(self):
        self.assertEqual(resolve_sensevoice_device("cpu", cuda_available=True), "cpu")

    def test_explicit_cuda(self):
        self.assertEqual(
            resolve_sensevoice_device("cuda:1", cuda_available=False), "cuda:1"
        )

    def test_auto_prefers_cuda_when_present(self):
        self.assertEqual(
            resolve_sensevoice_device("auto", cuda_available=True), "cuda:0"
        )

    def test_auto_falls_back_to_cpu(self):
        self.assertEqual(
            resolve_sensevoice_device("auto", cuda_available=False), "cpu"
        )


if __name__ == "__main__":
    unittest.main()
