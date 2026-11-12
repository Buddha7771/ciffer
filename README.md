# ciffer

[![PyPI](https://img.shields.io/pypi/v/ciffer)](https://pypi.org/project/ciffer/) [![Tests](https://github.com/Buddha7771/ciffer/actions/workflows/test.yml/badge.svg)](https://github.com/Buddha7771/ciffer/actions/workflows/test.yml)

Biomolecular structures as NumPy tables, ready for ML pipelines.

Every structure is three linked tables (atoms, residues, and chains) whose columns are NumPy arrays. Built on [gemmi](https://github.com/project-gemmi/gemmi).

- **Read and write in one line.** mmCIF and PDB, with models, altlocs, and assemblies chosen as you read.
- **Navigate like PyMOL.** Select atoms, hop to their residues and chains, and extract what you keep.
- **Handle columns like pandas.** Every column is a NumPy array, and your own columns go in with `[]`.
- **Save once, load as is.** `to_bytes()` keeps every column, bond, and metadata entry.

## Installation

```sh
pip install ciffer
```

ciffer needs Python 3.11 or newer.

## Usage

### A first look

```python
import ciffer

st = ciffer.fetch("1stp", assembly=True)  # from RCSB, as the biological tetramer

st.atoms.name[:4]  # ['N', 'CA', 'C', 'O']
st.atoms.xyz.shape  # (4004, 3)
len(st.atoms.bond)  # 3280

st.residues.name[:4]  # ['ASP', 'PRO', 'SER', 'LYS']
st.residues.label_seq_id[:4]  # [1, 2, 3, 4]

st.chains.label_asym_id[:3]  # ['A1', 'B1', 'C1']
st.chains.entity_type[:3]  # ['polymer', 'non-polymer', 'water']

st.metadata["resolution"]  # 2.6
st.metadata["release_date"]  # '1992-10-15'

st  # prints every column, edge, and metadata key
```

<details>
<summary>Output of <code>st</code></summary>

```text
<Structure atoms=4004 residues=976 chains=12>
  atoms
    name       <U3
    element    <U1
    altloc     <U1
    occupancy  float64
    b_factor   float64
    charge     int64
    xyz        float64 (3,)
    edges
      bond  3280 edges
        order     int64
        aromatic  bool
      struct_conn  0 edges
        type   <U1
        order  int64
  residues
    name          <U3
    label_seq_id  int64
    auth_seq_id   <U3
    ins_code      <U1
    comp_type     <U17
  chains
    label_asym_id  <U2
    auth_asym_id   <U2
    entity_id      <U1
    entity_type    <U11
    polymer_type   <U14
    description    <U32
  metadata
    id            '1STP'
    title         'STRUCTURAL ORIGINS OF HIGH-AFFINITY BIOTIN BINDING TO STREPTAVIDIN'
    keywords      'BIOTIN BINDING PROTEIN'
    deposit_date  '1992-03-12'
    methods       ['X-RAY DIFFRACTION']
    authors       ['Weber, P.C.', 'Salemme, F.R.']
    resolution    2.6
    release_date  '1992-10-15'
    model         1
    models        [1]
    assembly      '1'
    assemblies    ['1']
```

</details>

### Read and write

```python
ensemble = ciffer.fetch("1l2y", directory="data", model="all")  # cached in data/
len(ensemble)  # 38
ciffer.write(ensemble[0], "trp_cage.pdb.gz")
ciffer.read("trp_cage.pdb.gz").atoms.xyz.shape  # (304, 3)
```

`read` takes `.cif`, `.mmcif`, `.pdb`, and `.ent`, gzipped or not. `fetch` also takes extended ids such as `pdb_00001stp`. Both accept `model`, `altloc`, and `assembly`.

### Select and navigate

Select rows with a mask, hop between levels, and `extract()` what you keep:

```python
import numpy as np

biotin = st.residues[st.residues.name == "BTN"].atoms
protein = st.chains[st.chains.entity_type == "polymer"].atoms
distance = np.linalg.norm(protein.xyz[:, None] - biotin.xyz, axis=-1).min(axis=1)
pocket = protein[distance < 5].residues
pocket.name  # ['ASN', 'LEU', 'SER', 'TYR', 'SER', 'VAL', ...]
pocket.chains.auth_asym_id  # ['A1', 'A2', 'A3', 'A4']
ciffer.write(pocket.extract(), "pocket.cif")
```

### Label and edit

Add your own column with `[]`, select with it, and edit through the whole level:

```python
st.atoms["pocket"] = np.isin(st.atoms.indices, pocket.atoms.indices)
ca = st.atoms[st.atoms.name == "CA"]
len(ca[ca["pocket"]])  # 72
noise = np.random.default_rng(0).normal(scale=0.5, size=(len(pocket.atoms), 3))
st.atoms.xyz[pocket.atoms.indices] += noise  # writes into the structure
# A selection such as `ca` holds copies, so `ca.xyz += 1` raises a TypeError.
```

### Save and load

Store a structure as bytes in any database, such as SQLite, and load it back:

```python
import sqlite3

db = sqlite3.connect("structures.db")
db.execute("CREATE TABLE IF NOT EXISTS structures (id TEXT PRIMARY KEY, data BLOB)")
db.execute("REPLACE INTO structures VALUES (?, ?)", ("1stp", st.to_bytes()))
db.commit()

(data,) = db.execute("SELECT data FROM structures WHERE id = '1stp'").fetchone()
st = ciffer.Structure.from_bytes(data)
int(st.atoms["pocket"].sum())  # 616
len(st.atoms.bond)  # 3280
st.metadata["resolution"]  # 2.6
```

Your column, the bonds, and the metadata all come back.
