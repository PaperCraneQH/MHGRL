import csv
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from preprocess import build_paper_shell_features as paper


def _atom(serial, name, residue, sequence, x, element, chain="A"):
    return (f"ATOM  {serial:5d} {name:4} {residue:3} {chain}{sequence:4d}    "
            f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}"
            f"          {element:>2}\n")


class PaperShellFeaturesTest(unittest.TestCase):
    def setUp(self):
        self.shape = (60, len(paper.RESIDUE_TYPES), len(paper.LIGAND_TYPES))

    def test_each_residue_has_one_shell_and_counts_ligand_atoms(self):
        residues = [("ALA", np.array([[1.2, 0., 0.], [1.3, 0., 0.]]))]
        # Two atoms are in the assigned shell; the farther atom does not
        # create a second occurrence of the residue in a different shell.
        types = ["C", "Cl", "O"]
        ligand = np.array([[0., 0., 0.], [2.4, 0., 0.], [0., 2., 0.]])
        counts = paper.residue_shell_contacts(residues, types, ligand).reshape(self.shape)
        alanine = paper.RESIDUE_INDEX["ALA"]
        self.assertEqual(counts[1, alanine, paper.LIGAND_INDEX["C"]], 1)
        self.assertEqual(counts[1, alanine, paper.LIGAND_INDEX["HAL"]], 1)
        self.assertEqual(counts.sum(), 2)

    def test_left_closed_right_open_boundaries_and_unknown_types(self):
        positions = (0.0, 1.0, 1.5, 30.0, 30.5)
        residues = [
            ("MSE", np.array([[value, 0., 0.]])) for value in positions
        ]
        counts = paper.residue_shell_contacts(
            residues, ["Zn"], np.array([[0., 0., 0.]])).reshape(self.shape)
        np.testing.assert_array_equal(np.flatnonzero(counts.sum(axis=(1, 2))),
                                      [0, 1, 2, 59])
        unknowns = counts[:, paper.RESIDUE_INDEX["OTH"], paper.LIGAND_INDEX["DU"]]
        np.testing.assert_array_equal(unknowns[unknowns > 0], np.ones(4))

    def test_build_from_manifest_with_spaces(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "input data"
            folder.mkdir()
            protein, ligand = folder / "protein structure.pdb", folder / "ligand structure.pdb"
            protein.write_text(
                _atom(1, " CA ", "ALA", 1, 1.1, "C") +
                _atom(2, " CA ", "ALA", 1, 1.2, "C", chain="B"), encoding="ascii")
            ligand.write_text(_atom(1, " C  ", "LIG", 1, 0.0, "C"), encoding="ascii")
            manifest = folder / "input paths.tsv"
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t")
                writer.writerow(("PDB_code", "protein", "ligand"))
                writer.writerow(("1ABC", protein.name, ligand.name))
            out = folder / "output data.npz"
            script = Path(paper.__file__).resolve()
            run = subprocess.run(
                [sys.executable, str(script), "--inputs", str(manifest),
                 "--out", str(out), "--nproc", "2"],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            with np.load(out, allow_pickle=False) as features:
                self.assertEqual(features["codes"].tolist(), ["1abc"])
                self.assertEqual(features["feats"].shape, (1, 60 * 168))
                self.assertEqual(features["partition_family"].item(), paper.PARTITION_FAMILY)
                self.assertEqual(features["shell_upper_edges_angstrom"][-1], 30.5)
                counts = features["feats"][0].reshape(self.shape)
                self.assertEqual(counts[1, paper.RESIDUE_INDEX["ALA"],
                                        paper.LIGAND_INDEX["C"]], 2)
                self.assertEqual(counts.sum(), 2)

    def test_build_from_pdbbind_index(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            source_root = folder / "PDBbind root"
            complex_dir = source_root / "2019" / "1abc"
            complex_dir.mkdir(parents=True)
            (complex_dir / "1abc_protein.pdb").write_text(
                _atom(1, " CA ", "ALA", 1, 1.1, "C"), encoding="ascii")
            (complex_dir / "1abc_ligand.pdb").write_text(
                _atom(1, " C  ", "LIG", 1, 0.0, "C"), encoding="ascii")
            index = folder / "PDBbind index.csv"
            index.write_text("PDB_code,file\n1ABC,2019\n", encoding="ascii")
            out = folder / "output.npz"
            script = Path(paper.__file__).resolve()
            run = subprocess.run(
                [sys.executable, str(script), "--pdbbind_pl_root", str(source_root),
                 "--pdbbind_index", str(index), "--index_csvs", str(index),
                 "--staging", str(folder / "staging"), "--out", str(out), "--nproc", "1"],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            with np.load(out, allow_pickle=False) as features:
                self.assertEqual(features["codes"].tolist(), ["1abc"])
                self.assertEqual(features["feats"].sum(), 1)


if __name__ == "__main__":
    unittest.main()
