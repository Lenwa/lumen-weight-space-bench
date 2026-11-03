"""CPU-only checks of the public reproduction entry points."""
import json
from pathlib import Path
import tempfile
import unittest


class QuickstartTests(unittest.TestCase):
    def test_real_three_checkpoint_pipeline_and_existing_output_refusal(self):
        from quickstart import run_demo
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'demo'
            result = run_demo(out)
            self.assertTrue(result['noop_matches'])
            self.assertTrue((out / 'edited/weight-edit-receipt.json').is_file())
            receipt = json.loads((out / 'edited/weight-edit-receipt.json').read_text())
            self.assertEqual(len(receipt['edited_tensors']), 1)
            with self.assertRaises(FileExistsError):
                run_demo(out)


if __name__ == '__main__':
    unittest.main()
