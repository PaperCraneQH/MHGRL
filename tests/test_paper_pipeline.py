"""Exercise the paper configuration through the training model factory."""
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.chemberta_cache import smiles_hash
from models import build_model
from models.paper_mhgrl import PaperMHGRL
from training.losses import make_loss_fn


def write_embeddings(path, model_cfg, texts, hidden, rng):
    with h5py.File(path, "w") as cache:
        cache.attrs["model_name"] = model_cfg["model_name"]
        cache.attrs["pool"] = model_cfg["pool"]
        cache.attrs["hidden"] = hidden
        for value in texts:
            group = cache.create_group(smiles_hash(value))
            group.create_dataset("emb", data=rng.normal(size=hidden).astype(np.float16))


class PaperPipelineTest(unittest.TestCase):
    def test_training_configs(self):
        paths = sorted((ROOT / "configs").rglob("*.yaml"))
        self.assertEqual([p.relative_to(ROOT / "configs").as_posix() for p in paths], [
            "base.yaml", "holdout.yaml", "random.yaml",
            "scaffold.yaml", "seq_identity.yaml",
        ])
        for path in paths:
            with self.subTest(split=path.stem):
                cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertEqual(cfg["fusion"]["type"], "paper_mhgrl")
                self.assertEqual(
                    tuple(cfg["fusion"][name] for name in ("hidden_dim", "n_layers", "n_heads", "k")),
                    (256, 3, 4, 5),
                )
                train = cfg["training"]
                self.assertEqual(train["batch_size_train"], 64)
                self.assertEqual(train["epochs"], 200)
                self.assertEqual(train["early_stop_patience"], 30)
                self.assertEqual(train["seed_sequence"], [43, 44, 46, 50, 58])
                self.assertEqual(train["result_std_ddof"], 1)
                self.assertEqual(train["loss"]["type"], "mse_only")
                self.assertFalse(train["ema"]["enabled"])
                self.assertEqual(train["scheduler"]["name"], "none")

    def test_cached_embeddings_and_contact_features(self):
        torch.set_num_threads(2)
        torch.manual_seed(43)
        rng = np.random.default_rng(43)
        cfg = yaml.safe_load((ROOT / "configs/random.yaml").read_text(encoding="utf-8"))
        codes, sequences, smiles = ["test1", "test2"], ["ACDE", "FGHI"], ["CC", "CCC"]
        with tempfile.TemporaryDirectory() as work:
            work = Path(work)
            features, scaler = work / "features.npz", work / "scaler.npz"
            width = cfg["plitext"]["n_shells"] * cfg["plitext"]["n_pairs"]
            np.savez(
                features, codes=np.asarray(codes),
                feats=rng.poisson(3, (len(codes), width)).astype(np.float32),
                partition_family="paper_residue_min_0_30p5A", feature_layout="shell_major",
            )
            np.savez(scaler, mean=np.ones(width), std=np.full(width, 2.0))
            cfg["plitext"].update(features_path=str(features), scaler_path=str(scaler))
            for branch, inputs, hidden in (("ligand", smiles, 384), ("protein", sequences, 1024)):
                cache = work / f"{branch}.h5"
                write_embeddings(cache, cfg[branch], inputs, hidden, rng)
                cfg[branch]["cache_path"] = str(cache)

            net = build_model(cfg, "ChemBERTa").train()
            self.assertIsInstance(net.fusion, PaperMHGRL)
            self.assertFalse(net.lig.training)
            self.assertFalse(net.prot.training)
            result = net(None, sequences, smiles, pdb_codes=codes)
            self.assertEqual(result.shape, (2,))
            self.assertTrue(torch.isfinite(result).all())
            target = torch.tensor([5.2, 6.1])
            criterion = make_loss_fn(cfg["training"]["loss"])
            loss = criterion(result, target)
            self.assertTrue(torch.allclose(loss, (result - target).square().mean()))
            loss.backward()
            self.assertIsNotNone(net.fusion.layers[0].q_proj.weight.grad)
            self.assertIsNotNone(net.plitext.encoder.lstm.weight_ih_l0.grad)


if __name__ == "__main__":
    unittest.main()
