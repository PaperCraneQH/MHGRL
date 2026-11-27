"""PDBbind dataset for PLM-based training."""
from __future__ import annotations

import os
from typing import Iterable, List, Optional

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


_ATOM_LIST = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
    'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
    'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn',
    'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown',
]
_DEGREE_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
_NUMH_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
_IMVAL_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def _one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set {allowable_set}")
    return [x == s for s in allowable_set]


def _one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def _atom_features(atom):
    return np.array(
        _one_of_k_encoding_unk(atom.GetSymbol(), _ATOM_LIST)
        + _one_of_k_encoding(atom.GetDegree(), _DEGREE_LIST)
        + _one_of_k_encoding_unk(atom.GetTotalNumHs(), _NUMH_LIST)
        + _one_of_k_encoding_unk(atom.GetImplicitValence(), _IMVAL_LIST)
        + [atom.GetIsAromatic()],
        dtype=np.float32,
    )


def smiles_to_graph(smiles: str):
    """Build a GraphDTA-style ligand graph with normalized atom features.

    Returns (atom count, features [N, 78], edge index [2, E]).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feat = _atom_features(atom)
        s = feat.sum()
        features.append(feat / s if s > 0 else feat)

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edges.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])

    if len(edges) == 0:
        edge_index = [[0], [0]]
    else:
        edge_index = list(zip(*edges))
        edge_index = [list(edge_index[0]), list(edge_index[1])]
    return c_size, features, edge_index


class PDBbindESMDataset(InMemoryDataset):
    """Cache ligand graphs, pK labels, and raw sequences and SMILES.

    Cache files are located at ``<root>/<subdir>/<dataset>.pt``. If omitted,
    ``subdir`` is inferred from the dataset name. Existing
    ``<root>/processed/<dataset>.pt`` files can still be read.
    """

    SPLIT_NAMES = ('base_nocasf', 'base', 'random', 'scaffold',
                   'seq_identity', 'holdout', 'gems_5fold', 'paper')
    PAPER_DATASETS = ('train_paper', 'valid_paper', 'test2013', 'test2016', 'test2019')

    @staticmethod
    def infer_subdir(dataset_name: str) -> str:
        """Infer the split subdirectory from the .pt name. Returns '' if unknown (fallback to root)."""
        if dataset_name in PDBbindESMDataset.PAPER_DATASETS:
            return 'paper'
        if dataset_name in ('CASF-2016', 'CSAR-HiQ'):
            return 'external'
        for prefix in ('train_', 'valid_', 'test_'):
            if dataset_name.startswith(prefix):
                rest = dataset_name[len(prefix):]
                # Match longer split names first to distinguish base_nocasf from base.
                for s in sorted(PDBbindESMDataset.SPLIT_NAMES, key=len, reverse=True):
                    if rest == s or rest.startswith(s + '_'):
                        return s
        return ''

    def __init__(self, root: str = 'data_processed_esm', dataset: str = 'unnamed',
                 subdir: Optional[str] = None,
                 xd: Optional[List[str]] = None,
                 xt: Optional[List[str]] = None,
                 y: Optional[Iterable[float]] = None,
                 xp: Optional[List[str]] = None,
                 transform=None, pre_transform=None):
        # PyG reads processed_dir during parent initialization.
        self._subdir = subdir if subdir is not None else self.infer_subdir(dataset)
        self.dataset = dataset

        legacy_path = os.path.join(root, 'processed', dataset + '.pt')

        super().__init__(root, transform, pre_transform)

        new_path = self.processed_paths[0]
        if os.path.isfile(new_path):
            print(f'Pre-processed data found: {new_path}, loading ...', flush=True)
            self.data, self.slices = torch.load(new_path)
        elif os.path.isfile(legacy_path):
            print(f'[migrate] using legacy flat cache {legacy_path}.\n'
                  f'    Consider running: python create_data_PDBbind_esm.py --migrate\n'
                  f'    to move it to {new_path}.', flush=True)
            self.data, self.slices = torch.load(legacy_path)
        else:
            print(f'Pre-processed data {new_path} not found, doing pre-processing...', flush=True)
            assert xd is not None and xt is not None and y is not None, \
                'xd / xt / y must be provided to build a new dataset'
            self.process(xd, xt, y, xp=xp)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        if self._subdir:
            return os.path.join(self.root, self._subdir)
        return self.root

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd: List[str], xt: List[str], y: Iterable[float],
                xp: Optional[List[str]] = None):
        y = list(y)
        assert len(xd) == len(xt) == len(y), \
            f'len mismatch: xd={len(xd)} xt={len(xt)} y={len(y)}'
        if xp is not None:
            assert len(xp) == len(xd), \
                f'len mismatch: xp={len(xp)} xd={len(xd)}'

        data_list: List[Data] = []
        skipped = 0
        for i in tqdm(range(len(xd)), desc=f'building {self.dataset}'):
            smi = xd[i]
            seq = xt[i]
            label = y[i]
            g = smiles_to_graph(smi)
            if g is None:
                skipped += 1
                continue
            c_size, features, edge_index = g
            d = Data(
                x=torch.tensor(np.asarray(features), dtype=torch.float32),
                edge_index=torch.tensor(np.asarray(edge_index), dtype=torch.long),
                y=torch.tensor([float(label)], dtype=torch.float32),
            )
            d.c_size = torch.tensor([c_size], dtype=torch.long)
            d.seq = str(seq) if seq is not None else ''
            d.smiles = str(smi) if smi is not None else ''
            # Structure feature stores use the lowercase PDB code as their key.
            d.pdb_code = (str(xp[i]).lower() if xp is not None and xp[i] is not None else '')
            data_list.append(d)

        if skipped:
            print(f'[{self.dataset}] skipped {skipped} samples (RDKit failed to parse SMILES)', flush=True)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        print(f'[{self.dataset}] graph build done, n={len(data_list)}, saving ...', flush=True)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
