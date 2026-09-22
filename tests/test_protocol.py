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

    def test_manifest_hen_terkunci(self):
        """Empat config HEN ikut dikunci, seperti 237/155 dan dev2.

        Angka di bawah TIDAK dikarang: dibaca dari jalan pertama, dibekukan ke
        protocol.expected_counts, lalu jalan kedua melaporkan "split lock:
        cocok" untuk keempatnya. Tanpa tes ini komposisi lengan bisa bergeser
        tanpa satu pun uji gagal, dan seluruh angka lengan HEN bergantung
        padanya.

        Dua susunan, dan selisihnya yang jadi pengukuran: lengan campur 120/83
        crop dari 40 foto, lengan semua 1149/851.
        """
        harap = {
            "configs/config_hen_campur.yaml": (
                {"alive": 76, "dead": 44, "total": 120},
                {"alive": 50, "dead": 33, "total": 83}),
            "configs/config_hen_campur_eq.yaml": (
                {"alive": 76, "dead": 44, "total": 120},
                {"alive": 50, "dead": 33, "total": 83}),
            "configs/config_hen_semua.yaml": (
                {"alive": 852, "dead": 297, "total": 1149},
                {"alive": 648, "dead": 203, "total": 851}),
            "configs/config_hen_semua_eq.yaml": (
                {"alive": 852, "dead": 297, "total": 1149},
                {"alive": 648, "dead": 203, "total": 851}),
        }
        for config, (tr, va) in harap.items():
            if not Path(config).exists():
                self.skipTest(f"{config} belum ada")
            hasil = validate_development_manifest(load_config(config))
            self.assertEqual(hasil["counts"]["train"], tr, config)
            self.assertEqual(hasil["counts"]["val"], va, config)

    def test_hen_domain_sama_antar_kelas(self):
        """Inti susunan HEN: `domain` WAJIB identik antara hidup dan mati.

        Kalau kelas mati dan hidup datang dari domain berbeda, label = domain
        dan angka berapa pun jadi tak berarti - itu yang terjadi pada lengan
        PIO. Di sini kedua kelas dari kandang yang sama, dan tes ini yang
        menjaganya tetap begitu.

        Origin juga diperiksa TUNGGAL: coco_gt terbukti himpunan bagian HEN v2
        (36/36 foto punya kembaran, 13 pasang MAE 0.000), jadi ia dilipat jadi
        satu origin hen_gt. Kalau muncul origin kedua, klaim "satu sumber mati"
        di laporan berhenti benar - dan tes ini yang memberitahu.
        """
        from collections import Counter as _Counter
        for nama in ("campur", "campur_eq", "semua", "semua_eq"):
            manifest = Path(f"data/crops_hen_{nama}/manifest.csv")
            if not manifest.exists():
                self.skipTest(f"manifest hen_{nama} belum dibangun")
            with open(manifest, encoding="utf-8") as f:
                baris = list(csv.DictReader(f))
            per_kelas = {}
            for r in baris:
                per_kelas.setdefault(r["label_name"], _Counter())[r["domain"]] += 1
            # label_name di manifest "hidup"/"mati" - kunci "alive"/"dead"
            # hanya dipakai dict counts (dataset.py:527). Dua kosakata di dua
            # lapis; jangan disamakan.
            self.assertEqual(sorted(per_kelas), ["hidup", "mati"], nama)
            self.assertEqual(set(per_kelas["hidup"]), {"hen_kandang"}, nama)
            self.assertEqual(set(per_kelas["mati"]), {"hen_kandang"}, nama)
            origin = {r["origin"] for r in baris}
            self.assertEqual(origin, {"hen_gt"},
                             f"hen_{nama}: origin harus TUNGGAL, dapat {origin}")

    def test_grup_hen_menangkap_kembar_lintas_split(self):
        """Regresi temuan utama lengan ini, dan satu-satunya tes di berkas ini
        yang menguji FUNGSINYA langsung, bukan hasilnya.

        Dua kebocoran berbeda harus tertutup sekaligus:
            (a) stem sama, isi beda    -> tetap satu foto (cabang stem)
            (b) stem BEDA, isi kembar  -> tetap satu foto (cabang korelasi)
        Cabang (b) yang penting: terukur 465 pasangan kembar ber-stem berbeda,
        melibatkan 177 frame. Pada jalan pertama cabang itu MEMANG sempat jadi
        kode mati karena pemotongan per-stem; kalau ia mati lagi, kebocorannya
        kembali tanpa ada yang gagal.

        Sekaligus dijaga bahwa frame tak berkaitan TIDAK tergabung - tanpa
        pemeriksaan itu, fungsi yang menggabungkan semua frame jadi satu foto
        akan lulus tes ini.
        """
        import importlib
        try:
            bch = importlib.import_module("build_crops_hen")
        except Exception as e:                       # pragma: no cover
            self.skipTest(f"build_crops_hen tak bisa diimpor: {e}")

        def sidik(v):
            a = np.asarray(v, dtype=np.float32).ravel()
            a = a - a.mean()
            n = float(np.linalg.norm(a))
            return a / n if n else a

        rng = np.random.default_rng(0)
        pola_a = rng.normal(size=1024)
        pola_b = rng.normal(size=1024)
        kembar = pola_a + rng.normal(scale=0.02, size=1024)

        frames = [
            {"stem": "s1", "split_zip": "train", "sidik": sidik(pola_a)},
            {"stem": "s2", "split_zip": "valid", "sidik": sidik(kembar)},
            {"stem": "s1", "split_zip": "test",  "sidik": sidik(pola_b)},
        ]
        # Kalau data ujinya sendiri tidak sah, tesnya bisa "lulus" tanpa
        # menguji apa pun. Jadi korelasinya diperiksa lebih dulu.
        r_kembar = float(frames[0]["sidik"] @ frames[1]["sidik"])
        self.assertGreaterEqual(
            r_kembar, 0.97, f"data uji tidak sah: korelasi kembar {r_kembar:.4f}")

        id_foto = bch.kelompokkan_foto(frames, ambang=0.97)
        self.assertEqual(id_foto[0], id_foto[1],
                         "kembar lintas-stem TIDAK tergabung: cabang korelasi mati")
        self.assertEqual(id_foto[0], id_foto[2],
                         "stem sama TIDAK tergabung: cabang stem mati")

        lain = [{"stem": "z1", "split_zip": "train", "sidik": sidik(pola_a)},
                {"stem": "z2", "split_zip": "train", "sidik": sidik(pola_b)}]
        r_lain = float(lain[0]["sidik"] @ lain[1]["sidik"])
        self.assertLess(r_lain, 0.97, f"data uji tidak sah: {r_lain:.4f}")
        id_lain = bch.kelompokkan_foto(lain, ambang=0.97)
        self.assertNotEqual(id_lain[0], id_lain[1],
                            "frame tak berkaitan ikut tergabung: over-merge total")

    def test_foto_hen_tidak_bocor_antar_split(self):
        """Satu id foto tak pernah ada di dua split.

        Diuji di dua tempat: fungsinya DAN manifest jadinya, karena keduanya
        bisa gagal sendiri-sendiri - fungsinya benar tapi pemakaiannya salah,
        atau sebaliknya.
        """
        import importlib
        import random as _random
        try:
            bch = importlib.import_module("build_crops_hen")
        except Exception as e:                       # pragma: no cover
            self.skipTest(f"build_crops_hen tak bisa diimpor: {e}")

        # Jumlah crop per foto sengaja sangat tidak rata (1..79 seperti data
        # asli): itu yang membuat pembagian per FOTO bisa jauh dari 60/40, dan
        # itulah sebab bagi_per_foto menghitung rasio atas CROP.
        foto_ke_jumlah = {f"hen_f{i:04d}": (1 + (i * 7) % 79) for i in range(40)}
        petakan = bch.bagi_per_foto(foto_ke_jumlah, ["train", "val"],
                                    [0.6, 0.4], _random.Random(42))
        self.assertEqual(set(petakan), set(foto_ke_jumlah))
        self.assertEqual(set(petakan.values()), {"train", "val"}, "satu split kosong")
        n_val = sum(foto_ke_jumlah[f] for f, sp in petakan.items() if sp == "val")
        total = sum(foto_ke_jumlah.values())
        self.assertAlmostEqual(n_val / total, 0.4, delta=0.12,
                               msg=f"porsi crop val {n_val}/{total} jauh dari 0.4")

        for nama in ("campur", "campur_eq", "semua", "semua_eq"):
            manifest = Path(f"data/crops_hen_{nama}/manifest.csv")
            if not manifest.exists():
                continue
            with open(manifest, encoding="utf-8") as f:
                milik = {}
                for r in csv.DictReader(f):
                    lama = milik.setdefault(r["base_image"], r["split"])
                    self.assertEqual(
                        lama, r["split"],
                        f"hen_{nama}: foto {r['base_image']} ada di dua split")

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

    def test_hen_ditolak_tanpa_allowed_alive_source(self):
        """Penjaga atas suntingan dataset.py: alive_source non-pio WAJIB
        didaftarkan eksplisit di protocol.allowed_alive_source.

        Sebelum suntingan itu dataset.py memaku satu string ("pio") dan lengan
        HEN tidak mungkin jalan. Sesudahnya bawaannya {"pio"}, jadi empat
        config lama berperilaku byte-identik. Yang tidak boleh terjadi:
        seseorang menggeser alive_source tanpa mendaftarkannya dan
        pemeriksaannya lewat diam-diam. Kalau tes ini gagal, palangnya sudah
        turun sendiri.
        """
        for config in ("configs/config_hen_campur.yaml",
                       "configs/config_hen_semua.yaml"):
            if not Path(config).exists():
                self.skipTest(f"{config} belum ada")
            cfg = load_config(config)
            self.assertIn(cfg["crops"].get("alive_source"),
                          ("hen_campuran", "hen_semua"))
            # Buang daftar izinnya -> bawaan {"pio"} berlaku -> wajib TOLAK.
            cfg["protocol"].pop("allowed_alive_source", None)
            with self.assertRaises(ValueError, msg=config) as ctx:
                validate_development_manifest(cfg)
            self.assertIn("allowed_alive_source", str(ctx.exception))

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

class TestManifestKolektor(unittest.TestCase):
    """KolektorSDD2 satu-satunya lengan yang menjalankan pipeline DUA TAHAP
    penuh (detektor dilatih dari nol -> kotaknya memotong crop -> classifier
    menilai crop itu), jadi manifestnya tunduk pada aturan anti-bocor yang
    sama, ditambah satu yang khas dataset ini: split test paper tetap utuh."""

    MANIFEST = Path("data/crops_kolektor_gt/manifest.csv")

    def _baca(self):
        if not self.MANIFEST.exists():
            self.skipTest("manifest Kolektor belum dibangun")
        with open(self.MANIFEST, encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_gambar_tidak_bocor_antar_split(self):
        # Satu gambar Kolektor menyumbang crop cacat DAN crop normal. Itu
        # justru sebabnya build_crops.balance_and_split melempar ValueError
        # ("satu base_image punya dua label") dan build_crops_kolektor.py
        # punya bagi_per_gambar() sendiri - jadi penjaganya harus di sini.
        milik = {}
        for r in self._baca():
            lama = milik.setdefault(r["base_image"], r["split"])
            self.assertEqual(lama, r["split"],
                             f"gambar {r['base_image']} ada di dua split")

    def test_test_split_utuh_sesuai_paper(self):
        """Split test paper (110 cacat / 894 normal) tidak boleh disentuh.

        Yang diperiksa: tiap baris ber-source_split "test" HARUS ber-split
        "test" dan sebaliknya - tak ada gambar test menyelinap ke train/val,
        tak ada gambar train mengaku test. Jumlah CROP-nya tidak dipaku karena
        bergantung pada normal_per_cacat dan --sumber; yang dipaku pemisahannya.
        """
        baris = self._baca()
        for r in baris:
            if r["source_split"] == "test":
                self.assertEqual(r["split"], "test", r["base_image"])
            else:
                self.assertIn(r["split"], ("train", "val"), r["base_image"])
        g_test = {r["base_image"] for r in baris if r["source_split"] == "test"}
        g_lain = {r["base_image"] for r in baris if r["source_split"] != "test"}
        self.assertEqual(g_test & g_lain, set(),
                         "gambar yang sama ada di split paper test DAN train")

    def test_orphan_copy_dibuang(self):
        """Berkas "10301 (copy).png" tak punya pasangan _GT dan dibuang di
        staging dengan hitungan tercetak. Kalau ia muncul di manifest, tata
        letaknya dibangun versi lain dan hitungan 110/894 serta 246/2085 yang
        dilaporkan tidak lagi berlaku."""
        for r in self._baca():
            self.assertNotIn("copy", r["base_image"].lower(),
                             f"berkas yatim ikut masuk: {r['base_image']}")
            self.assertNotIn("_GT", r["src_file"],
                             f"mask ikut jadi gambar: {r['src_file']}")

    def test_domain_sama_antar_kelas(self):
        """Seperti lengan HEN: domain identik antar kelas, supaya label tidak
        bisa dibaca dari domainnya."""
        per_kelas = {}
        for r in self._baca():
            per_kelas.setdefault(r["label_name"], set()).add(r["domain"])
        self.assertEqual(sorted(per_kelas), ["cacat", "normal"])
        for nama, dom in per_kelas.items():
            self.assertEqual(dom, {"kolektor_permukaan"}, nama)


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
