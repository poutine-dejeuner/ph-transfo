import warnings
from pathlib import Path

import deepchem as dc
from rdkit import Chem, RDLogger

warnings.filterwarnings("ignore", module="deepchem")
RDLogger.DisableLog("rdApp.*")

# featurizer = MolGraphConvFeaturizer(use_edges=True)
featurizer = 'Raw'
dataset_dc = dc.molnet.load_qm9(featurizer=featurizer, randomize=True)
tasks, dataset, transformers = dataset_dc
train, valid, test = dataset

x,y,w,ids = train.X, train.y, train.w, train.ids
conf=x[0].GetConformer().GetPositions()
print(conf)
print("X shape: ", x.shape)
print("y shape: ", y.shape)
print("w shape: ", w.shape)
print("ids shape: ", ids.shape)

print(x[0])
print(y[0])
print(w[0])
print(ids[0])

# Extract and show SMILES from ID
print("\n=== SMILES Sources ===")
sample_0_id = ids[0]

# Method 1: Parse the ID as SMILES and canonicalize
mol = Chem.MolFromSmiles(sample_0_id)
if mol:
    smiles_canonical = Chem.MolToSmiles(mol)
    print(f"1. Canonical SMILES: {smiles_canonical}")
else:
    print(f"1. Could not parse ID as SMILES: {sample_0_id}")

# Access 3D atomic coordinates from original SDF file
print("\n=== 3D Atomic Coordinates ===")

sdf_path = Path(__file__).parent.parent.parent / "data/deepchem/qm9/gdb9.sdf"
try:
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)

    # Find the molecule matching sample 0 ID
    for idx, mol_3d in enumerate(suppl):
        if mol_3d is None:
            continue
        # Check if this molecule matches our sample (rough matching by atom count)
        if mol_3d.GetNumAtoms() == mol.GetNumAtoms():
            print(f"Found matching 3D structure at SDF index {idx}")

            # Method 2: Extract SMILES from SDF structure
            smiles_from_3d = Chem.MolToSmiles(mol_3d)
            print(f"2. SMILES from 3D SDF: {smiles_from_3d}")

            # Check SDF properties
            if mol_3d.HasProp('_Name'):
                print(f"3. SDF _Name: {mol_3d.GetProp('_Name')}")

            conf = mol_3d.GetConformer()
            positions = conf.GetPositions()  # Returns Nx3 array
            print(f"\nAtomic coordinates shape: {positions.shape}")
            print("Sample atom coordinates:")
            for atom_idx in range(positions.shape[0]):
                atom = mol_3d.GetAtomWithIdx(atom_idx)
                print(f"  Atom {atom_idx} ({atom.GetSymbol()}): {positions[atom_idx]}")
            break
except FileNotFoundError:
    print(f"SDF file not found at {sdf_path}")
except Exception as e:
    print(f"Error reading SDF: {e}")
