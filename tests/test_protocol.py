"""Uji regresi ringan untuk protokol development dan intervensi."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from build_crops import balance_and_split
from common import load_config
from dataset import validate_development_manifest
from intervensi import acak_petak


class TestDevelopmentProtocol(unittest.TestCase):
    def test_manifest_terkunci(self):
        for config in ("configs/config_pio_dev.yaml",
                       "configs/config_pio_eq_dev.yaml"):
            result = validate_development_manifest(load_config(config))
            self.assertEqual(result["counts"]["train"],
                             {"alive": 179, "dead": 58, "total": 237})
            self.assertEqual(result["counts"]["val"],
                             {"alive": 115, "dead": 40, "total": 155})

    def test_split_dua_arah_tidak_bocor(self):
        rows = []
        for label, name in ((0, "alive"), (1, "dead")):
            for base in range(5):
                rows.append({"label": label, "label_name": name,
                             "base_image": f"{label}-{base}",
                             "source_split": "train"})
        cfg = {"seed": 42, "crops": {"max_alive_per_dead": 3,
               "split_by_base_image": True,
               "split_names": ["train", "val"], "split_ratio": [0.6, 0.4]}}
        out = balance_and_split(rows, cfg)
        seen = {}
        for row in out:
            old = seen.setdefault(row["base_image"], row["split"])
            self.assertEqual(old, row["split"])
        self.assertEqual({r["split"] for r in out}, {"train", "val"})

    def test_chick_ditolak_dari_development(self):
        cfg = load_config("configs/config_pio_dev.yaml")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.csv"
            fields = ["path", "label", "label_name", "split", "source_split",
                      "base_image", "src_file", "bbox", "conf", "origin", "domain"]
            rows = [
                ["alive/a.jpg", 0, "alive", "train", "pio", "pio-a", "a.jpg",
                 "[0,0,1,1]", 1, "pio_gt", "cctv"],
                ["dead/d.jpg", 1, "dead", "val", "test", "ayam (1)",
                 "ayam (1).jpg", "[0,0,1,1]", 1, "coco_gt", "closeup"],
            ]
            with open(manifest, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f); writer.writerow(fields); writer.writerows(rows)
            import hashlib
            sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
            lock = root / "lock.json"
            lock.write_text(json.dumps({"manifest_sha256": sha,
                "base_image_to_split": {"pio-a": "train", "ayam (1)": "val"}}))
            cfg["crops"]["manifest"] = str(manifest)
            cfg["crops"]["out_dir"] = str(root)
            cfg["protocol"]["split_lock"] = str(lock)
            cfg["protocol"].pop("expected_counts", None)
            with self.assertRaisesRegex(ValueError, "chick"):
                validate_development_manifest(cfg)


    def test_development_result_tidak_punya_test(self):
        run = Path("outputs/runs_pio_dev/ce__hier_addone__s42/result.json")
        if not run.exists():
            self.skipTest("smoke output belum ada")
        data = json.loads(run.read_text(encoding="utf-8"))
        self.assertTrue(data["development_only"])
        self.assertNotIn("test", data)
        self.assertNotIn("test_tuned", data)

    def test_validation_view_tetap(self):
        config = load_config("configs/config_pio_dev.yaml")
        from dataset import TwoViewDataset, read_manifest
        rows = read_manifest(config, "val")[:3]
        ds = TwoViewDataset(rows, config, seed=123, policy={"ops": []})
        a = ds[0]
        b = ds[0]
        self.assertTrue(np.array_equal(a[0].numpy(), b[0].numpy()))
        self.assertTrue(np.array_equal(a[1].numpy(), b[1].numpy()))



class TestIntervention(unittest.TestCase):
    def test_deterministik_tanpa_strip_atau_fixed_point(self):
        image = np.arange(19 * 23 * 3, dtype=np.uint16).reshape(19, 23, 3)
        first, meta = acak_petak(image, "crop-1", seed=17,
                                  return_metadata=True)
        second = acak_petak(image, "crop-1", seed=17)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.shape, image.shape)
        self.assertEqual(meta["fixed_points"], 0)
        self.assertEqual(meta["coverage_pixels"], 19 * 23)
        self.assertFalse(np.array_equal(first, image))
        # Baris/kolom terakhir ikut tersentuh; implementasi lama meninggalkannya.
        self.assertFalse(np.array_equal(first[-1], image[-1]))
        self.assertFalse(np.array_equal(first[:, -1], image[:, -1]))

class TestFixedBenchmark(unittest.TestCase):
    def test_loader_memisahkan_tiga_label(self):
        from eval_fixed_chick import load_all_rows
        rows = load_all_rows(Path("data/label_chick/lembar_label.csv"),
                             Path("data/label_chick/crops"))
        self.assertEqual(sum(r["y"] == 1 for r in rows), 22)
        self.assertEqual(sum(r["y"] == 0 for r in rows), 921)
        self.assertEqual(sum(r["y"] == -1 for r in rows), 272)

if __name__ == "__main__":
    unittest.main()
