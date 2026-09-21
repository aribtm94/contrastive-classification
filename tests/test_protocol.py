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

    def test_manifest_dev2_terkunci(self):
        """
        Susunan dua-domain-mati ikut dikunci, seperti susunan lama.

        Tanpa ini hanya 237/155 yang terjaga, dan lengan dev2 bisa berubah
        jumlahnya tanpa satu pun uji gagal - padahal registry-nya sudah beku
        dan seluruh angka di domain_mati_kedua.md bergantung pada komposisi
        ini (300 hidup PIO, 98 mati Roboflow, 200 mati archive_4).
        """
        for config in ("configs/config_pio_dev2.yaml",
                       "configs/config_pio_dev2_eq.yaml"):
            result = validate_development_manifest(load_config(config))
            self.assertEqual(result["counts"]["train"],
                             {"alive": 192, "dead": 184, "total": 376}, config)
            self.assertEqual(result["counts"]["val"],
                             {"alive": 108, "dead": 114, "total": 222}, config)

    def test_dev2_punya_dua_sumber_mati(self):
        """
        Yang membedakan lengan ini dari yang lama HANYA ada di manifest,
        bukan di config: kelas mati wajib berasal dari dua origin. Kalau
        archive_4 hilang dari manifest, jumlahnya masih bisa kebetulan cocok
        lewat jalur lain, jadi sumbernya diperiksa terpisah.
        """
        import csv as _csv
        from collections import Counter as _Counter
        manifest = Path("data/crops_pio_dev2/manifest.csv")
        if not manifest.exists():
            self.skipTest("manifest dev2 belum dibangun")
        per_kelas = {}
        for r in _csv.DictReader(open(manifest, encoding="utf-8")):
            per_kelas.setdefault(r["label_name"], _Counter())[r["origin"]] += 1
        self.assertEqual(dict(per_kelas["dead"]),
                         {"coco_gt": 98, "archive4_rgb": 200})
        self.assertEqual(dict(per_kelas["alive"]), {"pio_gt": 300})

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


    def _manifest_dua_domain(self, root: Path):
        """Manifest kecil: hidup PIO + mati dari DUA domain (archive_4)."""
        manifest = root / "manifest.csv"
        fields = ["path", "label", "label_name", "split", "source_split",
                  "base_image", "src_file", "bbox", "conf", "origin", "domain"]
        rows = [
            ["alive/a1.jpg", 0, "alive", "train", "pio", "pio-a1", "a1.jpg",
             "[0,0,1,1]", 1, "pio_gt", "cctv"],
            ["alive/a2.jpg", 0, "alive", "val", "pio", "pio-a2", "a2.jpg",
             "[0,0,1,1]", 1, "pio_gt", "cctv"],
            ["dead/d1.jpg", 1, "dead", "train", "train", "rf-d1", "d1.jpg",
             "[0,0,1,1]", 1, "coco_gt", "closeup"],
            ["dead/d2.jpg", 1, "dead", "val", "archive4", "archive4_mati_rgb_1",
             "mati_rgb_1.jpg", "[0,0,640,480]", 1,
             "archive4_rgb", "closeup_kaggle"],
        ]
        with open(manifest, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            writer.writerows(rows)
        import hashlib
        sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
        lock = root / "lock.json"
        lock.write_text(json.dumps({
            "manifest_sha256": sha,
            "base_image_to_split": {"archive4_mati_rgb_1": "val",
                                    "pio-a1": "train", "pio-a2": "val",
                                    "rf-d1": "train"}}))
        return manifest, lock

    def _cfg_dua_domain(self, root: Path, manifest: Path, lock: Path):
        cfg = load_config("configs/config_pio_dev.yaml")
        cfg["crops"]["manifest"] = str(manifest)
        cfg["crops"]["out_dir"] = str(root)
        cfg["protocol"]["split_lock"] = str(lock)
        cfg["protocol"].pop("expected_counts", None)
        cfg["protocol"].pop("allowed_provenance", None)
        return cfg

    def test_archive4_ditolak_tanpa_allowed_provenance(self):
        """Perilaku LAMA harus tetap: sumber tak terdaftar ditolak.

        Ini yang menjaga config_pio_dev.yaml. Kalau uji ini lolos padahal
        allowed_provenance tidak diisi, berarti penjaga provenance-nya sudah
        lumpuh dan sumber apa pun bisa menyelinap masuk diam-diam.
        """
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, lock = self._manifest_dua_domain(root)
            cfg = self._cfg_dua_domain(root, manifest, lock)
            with self.assertRaisesRegex(ValueError, "provenance tidak cocok"):
                validate_development_manifest(cfg)

    def test_archive4_diterima_jika_didaftarkan(self):
        """Dengan allowed_provenance, domain mati kedua boleh masuk."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, lock = self._manifest_dua_domain(root)
            cfg = self._cfg_dua_domain(root, manifest, lock)
            cfg["protocol"]["allowed_provenance"] = {
                "alive": [["pio_gt", "cctv"]],
                "dead": [["coco_gt", "closeup"],
                         ["archive4_rgb", "closeup_kaggle"]]}
            hasil = validate_development_manifest(cfg)
            self.assertEqual(hasil["provenance"]["archive4_rgb/closeup_kaggle"], 1)
            self.assertEqual(hasil["provenance"]["coco_gt/closeup"], 1)
            # Sumber hidup TETAP satu domain - itu memang belum berubah.
            self.assertEqual(hasil["provenance"]["pio_gt/cctv"], 2)

    def test_hidup_tetap_tidak_boleh_dari_archive4(self):
        """Mendaftarkan archive_4 untuk mati tidak membukanya untuk hidup."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.csv"
            fields = ["path", "label", "label_name", "split", "source_split",
                      "base_image", "src_file", "bbox", "conf", "origin", "domain"]
            rows = [
                ["alive/a1.jpg", 0, "alive", "train", "archive4", "a4-sehat",
                 "sehat_rgb_1.jpg", "[0,0,640,480]", 1,
                 "archive4_rgb", "closeup_kaggle"],
                ["alive/a2.jpg", 0, "alive", "val", "pio", "pio-a2", "a2.jpg",
                 "[0,0,1,1]", 1, "pio_gt", "cctv"],
                ["dead/d1.jpg", 1, "dead", "train", "train", "rf-d1", "d1.jpg",
                 "[0,0,1,1]", 1, "coco_gt", "closeup"],
                ["dead/d2.jpg", 1, "dead", "val", "train", "rf-d2", "d2.jpg",
                 "[0,0,1,1]", 1, "coco_gt", "closeup"],
            ]
            with open(manifest, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(fields)
                writer.writerows(rows)
            import hashlib
            sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
            lock = root / "lock.json"
            lock.write_text(json.dumps({"manifest_sha256": sha,
                "base_image_to_split": {}}))
            cfg = self._cfg_dua_domain(root, manifest, lock)
            cfg["protocol"]["allowed_provenance"] = {
                "alive": [["pio_gt", "cctv"]],
                "dead": [["coco_gt", "closeup"],
                         ["archive4_rgb", "closeup_kaggle"]]}
            with self.assertRaisesRegex(ValueError, "provenance tidak cocok"):
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

class TestManifestSdnet(unittest.TestCase):
    """SDNET2018 dipakai sebagai uji kewarasan pipeline, jadi manifestnya
    harus tunduk pada aturan anti-bocor yang sama dengan manifest ayam."""

    MANIFEST = Path("data/crops_sdnet/manifest.csv")

    def _baca(self):
        if not self.MANIFEST.exists():
            self.skipTest("manifest SDNET belum dibangun")
        with open(self.MANIFEST, encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_foto_sumber_tidak_bocor_antar_split(self):
        # Satu foto beton menyumbang sampai 249 ubin. Kalau satu foto muncul
        # di train sekaligus di test, skornya palsu - persis kesalahan yang
        # sudah pernah terjadi pada data ayam.
        milik = {}
        for r in self._baca():
            lama = milik.setdefault(r["base_image"], r["split"])
            self.assertEqual(lama, r["split"],
                             f"foto {r['base_image']} ada di dua split")

    def test_domain_holdout_tidak_masuk_latih(self):
        # wall dan pavement tidak boleh muncul di train/val/test in-domain.
        for r in self._baca():
            if r["domain"] in ("wall", "pavement"):
                self.assertTrue(r["split"].startswith("test_"),
                                f"{r['path']} domain hold-out di {r['split']}")
            else:
                self.assertIn(r["split"], ("train", "val", "test"))

    def test_kedua_label_ada_di_setiap_split(self):
        # Kalau satu split kehilangan satu kelas, AUC-nya tidak terdefinisi
        # dan seluruh laporan lintas domain jadi tidak bisa dibaca.
        per_split = {}
        for r in self._baca():
            per_split.setdefault(r["split"], set()).add(r["label"])
        for split, label in per_split.items():
            self.assertEqual(label, {"0", "1"}, f"split {split} tidak lengkap")

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
