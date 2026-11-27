"""Build residue-assigned radial shell contact features."""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
from scipy.spatial.distance import cdist

if __package__:
    from .prepare_inputs import DataRoots, build_pdbbind_year_map, convert_lig, find_protein_ligand
else:
    from prepare_inputs import DataRoots, build_pdbbind_year_map, convert_lig, find_protein_ligand


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "features_residue" / "paper"
PARTITION_FAMILY = "paper_residue_min_0_30p5A"
N_SHELLS = 60
UPPER_EDGES = 1.0 + np.arange(N_SHELLS) * 0.5
RESIDUE_TYPES = (
    "GLY", "ALA", "VAL", "LEU", "ILE", "PRO", "PHE", "TYR", "TRP", "SER",
    "THR", "CYS", "MET", "ASN", "GLN", "ASP", "GLU", "LYS", "ARG", "HIS", "OTH",
)
LIGAND_TYPES = ("H", "C", "O", "N", "P", "S", "HAL", "DU")
HALOGENS = frozenset(("F", "CL", "BR", "I", "AT", "TS"))
RESIDUE_INDEX = {name: i for i, name in enumerate(RESIDUE_TYPES)}
LIGAND_INDEX = {name: i for i, name in enumerate(LIGAND_TYPES)}
N_PAIRS = len(RESIDUE_TYPES) * len(LIGAND_TYPES)


def _element(line: str) -> str:
    element = line[76:78].strip().upper()
    if element:
        return element
    atom_name = line[12:16]
    letters = "".join(c for c in atom_name.upper() if c.isalpha())
    if atom_name.startswith(" ") or atom_name[:1].isdigit():
        return letters[:1]
    return letters[:2] if letters[:2] in HALOGENS else letters[:1]


def _atom_records(path: Path):
    """Read the first PDB model, using unlabelled or primary alternate sites."""
    with path.open(encoding="ascii", errors="replace") as stream:
        for line in stream:
            record = line[:6].strip()
            if record == "ENDMDL":
                break
            if record not in {"ATOM", "HETATM"} or line[16:17] not in {" ", "A"}:
                continue
            try:
                coord = [float(line[i:i+8]) for i in (30, 38, 46)]
            except ValueError as exc:
                raise ValueError(f"invalid PDB coordinates in {path}") from exc
            if not np.isfinite(coord).all():
                raise ValueError(f"non-finite PDB coordinates in {path}")
            yield line, coord


def _read_residues(path: Path) -> list[tuple[str, np.ndarray]]:
    residues = {}
    for line, coord in _atom_records(path):
        name = line[17:20].strip().upper()
        key = (line[21:27], name)  # chain, residue number, insertion code
        residues.setdefault(key, []).append(coord)
    if not residues:
        raise ValueError(f"no protein residues in {path}")
    return [(key[1], np.asarray(coords, dtype=np.float64))
            for key, coords in residues.items()]


def _read_ligand(path: Path) -> tuple[list[str], np.ndarray]:
    records = [(_element(line), coord) for line, coord in _atom_records(path)]
    if not records:
        raise ValueError(f"no ligand atoms in {path}")
    elements, coords = zip(*records)
    return list(elements), np.asarray(coords, dtype=np.float64)


def residue_shell_contacts(
    residues: list[tuple[str, np.ndarray]],
    ligand_elements: list[str],
    ligand_xyz: np.ndarray,
) -> np.ndarray:
    """Count residue–ligand atom pairs within each residue's assigned shell.

    Distances are in angstroms. The minimum over all ligand atoms assigns a
    residue to one shell. A ligand atom contributes one count there when its
    minimum distance to the residue is inside that shell's radial interval.
    """
    ligand_xyz = np.asarray(ligand_xyz, dtype=np.float64)
    if (not ligand_elements or ligand_xyz.shape != (len(ligand_elements), 3)
            or not np.isfinite(ligand_xyz).all()):
        raise ValueError("ligand atom types and finite coordinates must align")
    channels = np.asarray([
        LIGAND_INDEX.get("HAL" if elem in HALOGENS else "H" if elem == "D" else elem,
                         LIGAND_INDEX["DU"])
        for elem in (str(element).upper() for element in ligand_elements)
    ], dtype=np.intp)
    counts = np.zeros((N_SHELLS, len(RESIDUE_TYPES), len(LIGAND_TYPES)),
                      dtype=np.float32)
    for name, xyz in residues:
        distances = cdist(ligand_xyz, xyz).min(axis=1)
        shell = np.searchsorted(UPPER_EDGES, distances.min(), side="right")
        if shell == N_SHELLS:
            continue
        residue = RESIDUE_INDEX.get(name.upper(), RESIDUE_INDEX["OTH"])
        # Every pair distance is at least the residue's minimum, so the lower
        # bound of the assigned shell is already satisfied.
        contacts = channels[distances < UPPER_EDGES[shell]]
        counts[shell, residue] += np.bincount(contacts, minlength=len(LIGAND_TYPES))
    return counts.reshape(-1)


def _worker(row: tuple[str, Path, Path]):
    code, protein, ligand = row
    try:
        elements, ligand_xyz = _read_ligand(ligand)
        return code, residue_shell_contacts(_read_residues(protein), elements, ligand_xyz)
    except (OSError, ValueError) as exc:
        return code, str(exc)


def _validated_code(raw: str) -> str:
    code = raw.strip().lower()
    if not re.fullmatch(r"[a-z0-9]{4}", code):
        raise ValueError(f"invalid PDB code {raw!r}")
    return code


def read_input_manifest(path: Path) -> list[tuple[str, Path, Path]]:
    """Read PDB code / protein PDB / ligand PDB from a three-column TSV."""
    def resolve(raw: str) -> Path:
        item = Path(raw)
        if item.is_absolute():
            return item
        local = path.parent / item
        if local.is_file():
            return local
        # Legacy manifests may record paths relative to the repository root.
        rooted = ROOT / item
        return rooted if rooted.is_file() else local

    rows = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for line, fields in enumerate(csv.reader(stream, delimiter="\t"), 1):
            if not fields or not "".join(fields).strip():
                continue
            if len(fields) == 1:
                fields = fields[0].split()
            if len(fields) != 3:
                raise ValueError(f"{path}:{line}: expected PDB code and two tab-separated paths")
            if not rows and fields[0].strip().lower() in {"pdb_code", "code"}:
                continue
            code = _validated_code(fields[0])
            if code in rows:
                raise ValueError(f"{path}:{line}: duplicate PDB code {code}")
            rows[code] = (code, resolve(fields[1]), resolve(fields[2]))
    return list(rows.values())


def _collect_codes(index_csvs: list[Path], codes_file: Path | None) -> list[str]:
    if codes_file:
        return sorted({_validated_code(line) for line in codes_file.read_text(
            encoding="utf-8-sig").splitlines() if line.strip()})
    codes = set()
    for source in index_csvs:
        with source.open(newline="", encoding="utf-8-sig") as stream:
            rows = csv.DictReader(stream)
            if "PDB_code" not in (rows.fieldnames or []):
                raise ValueError(f"{source}: missing PDB_code column")
            codes.update(_validated_code(row["PDB_code"]) for row in rows if row["PDB_code"])
    return sorted(codes)


def _prepare_rows(codes: list[str], roots: DataRoots, staging: Path):
    rows, failed = [], []
    year_map = build_pdbbind_year_map(roots.pdbbind_index)
    ligand_dir = staging / "ligands"
    ligand_dir.mkdir(parents=True, exist_ok=True)
    for code in codes:
        protein, ligand, _ = find_protein_ligand(code, year_map, roots)
        if protein is None:
            failed.append((code, "protein or ligand file not found"))
            continue
        if ligand is None or not ligand.is_file():
            ligand = protein.with_name(f"{code}_ligand.pdb")
            if not ligand.is_file():
                failed.append((code, "protein or ligand file not found"))
                continue
        if ligand.suffix.lower() == ".pdb":
            rows.append((code, protein.resolve(), ligand.resolve()))
            continue
        lig_pdb = ligand_dir / f"{code}_ligand.pdb"
        # Convert in a temporary directory to avoid leaving a partial cached file.
        if not lig_pdb.is_file():
            with TemporaryDirectory(dir=ligand_dir) as tmp:
                temp_pdb = Path(tmp) / lig_pdb.name
                ok, _ = convert_lig(ligand, temp_pdb)
                if not ok:
                    failed.append((code, "ligand conversion failed"))
                    continue
                os.replace(temp_pdb, lig_pdb)
        rows.append((code, protein.resolve(), lig_pdb.resolve()))
    with (staging / "inputs.tsv").open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream, delimiter="\t").writerows(rows)
    return rows, failed


def build(rows: list[tuple[str, Path, Path]], out: Path, *, nproc: int = 1,
          failed: list[tuple[str, str]] | None = None) -> None:
    codes, arrays = [], []
    failures = list(failed or ())

    def collect(results):
        for code, value in results:
            if isinstance(value, str):
                failures.append((code, value))
            else:
                codes.append(code)
                arrays.append(value)

    if nproc == 1:
        collect(map(_worker, rows))
    else:
        with mp.Pool(processes=nproc) as pool:
            collect(pool.imap(_worker, rows, chunksize=16))
    out.parent.mkdir(parents=True, exist_ok=True)
    failure_path = out.with_suffix(".failures.tsv")
    with failure_path.open("w", newline="", encoding="utf-8") as stream:
        report = csv.writer(stream, delimiter="\t")
        report.writerow(("PDB_code", "reason"))
        report.writerows(failures)
    if not arrays:
        raise ValueError(f"no complexes could be processed; see {failure_path}")

    with NamedTemporaryFile(dir=out.parent, suffix=".npz", delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        np.savez_compressed(
            temp_path, codes=np.asarray(codes), feats=np.stack(arrays),
            partition_family=np.asarray(PARTITION_FAMILY),
            feature_layout=np.asarray("shell_major"),
            n_shells=np.asarray(N_SHELLS), n_pairs=np.asarray(N_PAIRS),
            max_radius_angstrom=np.asarray(UPPER_EDGES[-1]),
            shell_lower_edges_angstrom=np.concatenate(([0.0], UPPER_EDGES[:-1])),
            shell_upper_edges_angstrom=UPPER_EDGES,
            shell_boundary_rule=np.asarray("left_closed_right_open"),
            residue_types=np.asarray(RESIDUE_TYPES),
            ligand_atom_types=np.asarray(LIGAND_TYPES),
            contact_rule=np.asarray("residue_minimum_then_pair_distance_inside_assigned_shell"),
        )
        os.replace(temp_path, out)
    finally:
        temp_path.unlink(missing_ok=True)
    print(f"Wrote {len(codes)} complexes to {out} ({len(failures)} failed).")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--inputs", type=Path, help="three-column PDB code / protein PDB / ligand PDB TSV")
    source.add_argument("--index_csvs", "--index-csvs", nargs="+", type=Path)
    source.add_argument("--codes_file", "--codes-file", type=Path)
    parser.add_argument("--pdbbind_pl_root", "--pdbbind-pl-root", default=os.environ.get("PDBBIND_PL_ROOT"))
    parser.add_argument("--pdbbind_index", "--pdbbind-index", default=os.environ.get("PDBBIND_INDEX"))
    parser.add_argument("--casf_root", "--casf-root", default=os.environ.get("CASF_ROOT"))
    parser.add_argument("--csar_root", "--csar-root", default=os.environ.get("CSAR_ROOT"))
    parser.add_argument("--staging", type=Path, default=OUT_ROOT / "staging")
    parser.add_argument("--out", type=Path, default=OUT_ROOT / "residue_N60_paper.npz")
    parser.add_argument("--nproc", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--limit", type=int, help="process only the first N complexes")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.nproc < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--nproc and --limit must be positive")
    if args.out.suffix.lower() != ".npz":
        parser.error("--out must end in .npz")
    if args.out.exists() and not args.overwrite:
        parser.error(f"{args.out} already exists; use --overwrite to replace it")
    try:
        if args.inputs:
            rows, failed = read_input_manifest(args.inputs)[:args.limit], []
        else:
            codes = _collect_codes(args.index_csvs or [], args.codes_file)[:args.limit]
            roots = DataRoots(casf_root=args.casf_root, pdbbind_pl_root=args.pdbbind_pl_root,
                              csar_root=args.csar_root, pdbbind_index=args.pdbbind_index)
            rows, failed = _prepare_rows(codes, roots, args.staging)
        build(rows, args.out, nproc=args.nproc, failed=failed)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
