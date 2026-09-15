"""Python import caches must not invalidate a recorded-data inventory."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH=Path(__file__).resolve().parents[1]/"results/public_archive_certificate/verify.py"
spec=importlib.util.spec_from_file_location("archive_inventory",PATH)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class InventoryTests(unittest.TestCase):
    def test_import_cache_is_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/"data.csv").write_text("x\n1\n")
            with patch.object(module,"HERE",root):
                expected=module.hashes()
                (root/"__pycache__").mkdir()
                (root/"__pycache__/verify.cpython-312.pyc").write_bytes(b"cache")
                self.assertEqual(module.hashes(),expected)

    def test_unlisted_data_and_changed_evidence_remain_visible(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);data=root/"data.csv";data.write_text("x\n1\n")
            with patch.object(module,"HERE",root):
                expected=module.hashes();data.write_text("x\n2\n")
                self.assertNotEqual(module.hashes()["data.csv"],expected["data.csv"])
                (root/"extra.csv").write_text("x\n3\n")
                self.assertIn("extra.csv",module.hashes())

if __name__=="__main__":unittest.main()
