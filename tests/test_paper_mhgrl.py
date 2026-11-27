import importlib.util
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "paper_mhgrl.py"
SPEC = importlib.util.spec_from_file_location("paper_mhgrl", MODEL_PATH)
model_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model_module)
PaperGraphLayer = model_module.PaperGraphLayer
PaperMHGRL = model_module.PaperMHGRL
PaperShellEncoder = model_module.PaperShellEncoder


class PaperMHGRLTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(43)

    def make_model(self, mode="dynamic"):
        return PaperMHGRL(
            dim=8, n_shells=5, n_layers=3, n_heads=2, k=2, dropout=0,
            lig_dim=3, prot_dim=7, head_dims=(12,), topology_mode=mode,
        )

    def inputs(self):
        return torch.randn(2, 5, 8), torch.randn(2, 3), torch.randn(2, 7)

    def test_bilstm_and_prediction_train_with_mse(self):
        model = self.make_model()
        shell_encoder = PaperShellEncoder(dim=8, n_features=6)
        shell_features = torch.randn(2, 5, 6)
        states = shell_encoder(shell_features)
        torch.testing.assert_close(states, shell_encoder.lstm(shell_features)[0])
        _, ligand, protein = self.inputs()
        prediction = model(states, ligand, protein)
        self.assertEqual(prediction.shape, (2,))
        self.assertEqual(model.head[0].in_features, (5 + 2) * 8)
        F.mse_loss(prediction, torch.tensor([3.0, 5.0])).backward()
        for param in (shell_encoder.lstm.weight_ih_l0, model.protein_proj.weight,
                      model.ligand_proj.weight, model.layers[0].q_proj.weight):
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())
            self.assertGreater(param.grad.abs().sum().item(), 0)

    def test_directed_neighbors_and_initial_graph(self):
        model = self.make_model().eval()
        shell, ligand, protein = self.inputs()
        input_states = torch.cat((shell, model.protein_proj(protein)[:, None],
                                  model.ligand_proj(ligand)[:, None]), dim=1)
        neighbors = model.layers[0].build_topology(input_states)
        self.assertEqual(neighbors.shape, (2, 7, 2))
        self.assertTrue((neighbors < 5).all())
        self.assertTrue((neighbors[:, :5] != torch.arange(5)[None, :, None]).all())
        adjacency = model.initial_adjacency()
        self.assertEqual(adjacency.shape, (7, 7))
        self.assertFalse(adjacency.diag().any())
        self.assertFalse(adjacency[5:, 5:].any())
        self.assertTrue(adjacency[:5, 5:].all() and adjacency[5:, :5].all())
        original = model.encode(shell, ligand, protein)
        altered = model.encode(shell, -ligand, -protein)
        torch.testing.assert_close(original[:, :5], altered[:, :5])
        self.assertFalse(torch.allclose(original[:, 5:], altered[:, 5:]))

    def test_attention_matches_reference_calculation(self):
        layer = PaperGraphLayer(dim=4, n_heads=2, n_shells=4, k=2,
                                dropout=0, ffn_mult=2).double()
        with torch.no_grad():
            layer.attn_norm.weight.copy_(torch.tensor([0.7, 1.3, 0.4, 1.7]))
            layer.attn_norm.bias.copy_(torch.tensor([0.5, -0.2, 0.3, -0.7]))
        states = torch.randn(2, 6, 4, dtype=torch.float64)
        normalized = layer.attn_norm(states)
        query = layer.q_proj(normalized).reshape(2, 6, 2, 2)
        key = layer.k_proj(normalized).reshape(2, 6, 2, 2)
        value = layer.v_proj(normalized).reshape(2, 6, 2, 2)
        selected = layer.build_topology(states)
        combined = []
        for b in range(2):
            nodes = []
            for i in range(6):
                candidates = [j for j in range(4) if j != i]
                neighbors = sorted(
                    candidates,
                    key=lambda j: -F.cosine_similarity(normalized[b, i], normalized[b, j], dim=0),
                )[:2]
                self.assertEqual(neighbors, selected[b, i].tolist())
                heads = []
                for h in range(2):
                    logits = torch.stack([query[b, i, h] @ key[b, j, h] for j in neighbors])
                    weights = (logits / math.sqrt(2)).softmax(dim=0)
                    heads.append(sum(w * value[b, j, h] for w, j in zip(weights, neighbors)))
                nodes.append(torch.cat(heads))
            combined.append(torch.stack(nodes))
        intermediate = states + layer.out_proj(torch.stack(combined))
        expected = intermediate + layer.ffn(layer.ffn_norm(intermediate))
        torch.testing.assert_close(layer(states), expected)

    def test_topology_ablation_modes(self):
        for mode, expected in (("dynamic", [1, 1, 1]), ("first_layer", [1, 0, 0]),
                               ("predefined", [1, 0, 0])):
            with self.subTest(mode=mode):
                model = self.make_model(mode)
                with patch.object(model.layers[0], "select_neighbors",
                                  wraps=model.layers[0].select_neighbors) as first, \
                     patch.object(model.layers[1], "select_neighbors",
                                  wraps=model.layers[1].select_neighbors) as second, \
                     patch.object(model.layers[2], "select_neighbors",
                                  wraps=model.layers[2].select_neighbors) as third:
                    model(*self.inputs())
                self.assertEqual([first.call_count, second.call_count, third.call_count], expected)

        layer = self.make_model("predefined").layers[0]
        states = torch.randn(2, 7, 8)
        physical = layer.build_topology(states, predefined=True)
        expected_shell_keys = torch.tensor([[0, 1], [1, 0], [2, 1], [3, 2], [4, 3]])
        torch.testing.assert_close(physical[:, :5], expected_shell_keys.expand(2, -1, -1))
        torch.testing.assert_close(physical[:, 5:], layer.build_topology(states)[:, 5:])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
