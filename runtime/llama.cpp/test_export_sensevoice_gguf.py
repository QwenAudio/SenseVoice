import json
import tempfile
import unittest
from pathlib import Path

import export_sensevoice_gguf


class AudioCppModelSpecTest(unittest.TestCase):
    def write_spec(self, spec):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "model-spec.json"
        path.write_text(json.dumps(spec), encoding="utf-8")
        return path

    def test_loads_schema_v1_sense_asr_spec_as_compact_json(self):
        path = self.write_spec(
            {"family": "sense_asr", "schema_version": 1, "name": "SenseVoice"}
        )

        spec_json = export_sensevoice_gguf.load_audio_cpp_model_spec(path)

        self.assertEqual(
            spec_json,
            '{"family":"sense_asr","schema_version":1,"name":"SenseVoice"}',
        )

    def test_rejects_non_sense_asr_family(self):
        path = self.write_spec({"family": "other", "schema_version": 1})

        with self.assertRaisesRegex(ValueError, "family must be sense_asr"):
            export_sensevoice_gguf.load_audio_cpp_model_spec(path)

    def test_rejects_non_v1_schema(self):
        path = self.write_spec({"family": "sense_asr", "schema_version": 2})

        with self.assertRaisesRegex(ValueError, "schema version 1"):
            export_sensevoice_gguf.load_audio_cpp_model_spec(path)


if __name__ == "__main__":
    unittest.main()
