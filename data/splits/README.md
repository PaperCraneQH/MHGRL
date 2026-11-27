# Dataset splits

The CSV files in this directory contain one column, `PDB_code`, and record
split membership only. They do not include structures, sequences, SMILES or
binding affinities.

| Path | Membership |
| --- | --- |
| `cold_start/base/` | Five training and validation partitions for external evaluation |
| `cold_start/random/` | Five random train, validation and test partitions |
| `cold_start/scaffold/` | Five ligand-scaffold train, validation and test partitions |
| `cold_start/seq_identity/` | Five 30% protein-sequence-identity train, validation and test partitions |
| `cold_start/holdout/` | Fixed temporal split using pre-2018 training, 2018 validation and 2019 test complexes |
| `CASF-2016.csv` | 285 external test complexes |
| `CSAR-HiQ_dedup.csv` | 80 external test complexes after removing training-set overlaps and one candidate without usable SMILES |

To run preprocessing, supply your own authorized copies of the raw structures
and place the following metadata files locally in this directory.

- `PL-2020R1.csv` supplies `PDB_code` and the structure folder in `file`.
- `PL-2020R1_seq_smiles.csv` and `CSAR-HiQ_seq_smiles.csv` supply
  `PDB_code`, `sequence`, `smiles`, and `-logKd/Ki`. Preprocessing joins
  these tables to the split lists by `PDB_code`.
