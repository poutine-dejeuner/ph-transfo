
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from reg_transfo.datamodules.molecule_data import MoleculeConformerDataset


def compute_stats():
    # Initialize dataset
    print("Initializing dataset...")
    # Using the default path from the class, or you can specify if needed
    try:
        dataset = MoleculeConformerDataset()
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        return

    n_samples = len(dataset)
    print(f"Dataset size: {n_samples}")

    energies = []
    atom_counts = []

    print("Collecting energies and atom counts...")
    # Iterate through the dataset
    # We use a loop instead of dataloader to avoid collation overhead if we just want energies
    # But getitem does some processing.
    # Dataset __getitem__ returns: {'features': ..., 'energy': float}

    for i in tqdm(range(n_samples)):
        try:
            item = dataset[i]
            energies.append(item['energy'])
            atom_counts.append(item['features'].shape[0])
        except Exception as e:
            print(f"Error reading item {i}: {e}")
            continue

    energies = np.array(energies)
    atom_counts = np.array(atom_counts)

    # Compute statistics
    mean_energy = np.mean(energies)
    std_energy = np.std(energies)
    min_energy = np.min(energies)
    max_energy = np.max(energies)
    median_energy = np.median(energies)

    print("\n--- CREMP Dataset Statistics ---")
    print(f"Count: {len(energies)}")
    print(f"Mean Energy: {mean_energy:.4f}")
    print(f"Std Energy: {std_energy:.4f}")
    print(f"Min Energy: {min_energy:.4f}")
    print(f"Max Energy: {max_energy:.4f}")
    print(f"Median Energy: {median_energy:.4f}")

    # Atom statistics
    mean_atoms = np.mean(atom_counts)
    std_atoms = np.std(atom_counts)
    min_atoms = np.min(atom_counts)
    max_atoms = np.max(atom_counts)
    median_atoms = np.median(atom_counts)

    print("\n--- Atom Count Statistics ---")
    print(f"Mean Atoms: {mean_atoms:.2f}")
    print(f"Std Atoms: {std_atoms:.2f}")
    print(f"Min Atoms: {min_atoms}")
    print(f"Max Atoms: {max_atoms}")
    print(f"Median Atoms: {median_atoms}")

    # Plot histogram for energies
    plt.figure(figsize=(10, 6))
    plt.hist(energies, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Histogram of Molecule Energies (CREMP)')
    plt.xlabel('Energy')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    # Add stats box
    textstr = '\n'.join((
        f'Mean: {mean_energy:.2f}',
        f'Std: {std_energy:.2f}',
        f'Min: {min_energy:.2f}',
        f'Max: {max_energy:.2f}'))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.95, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    output_path = 'cremp_energy_histogram.png'
    plt.savefig(output_path)
    print(f"\nEnergy histogram saved to {output_path}")

    # Plot histogram for atoms
    plt.figure(figsize=(10, 6))
    plt.hist(atom_counts, bins=range(int(min_atoms), int(max_atoms) + 2), alpha=0.7, color='green', edgecolor='black', align='left')
    plt.title('Histogram of Atom Counts (CREMP)')
    plt.xlabel('Number of Atoms')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    # Add stats box for atoms
    textstr_atoms = '\n'.join((
        f'Mean: {mean_atoms:.2f}',
        f'Std: {std_atoms:.2f}',
        f'Min: {min_atoms}',
        f'Max: {max_atoms}'))
    plt.gca().text(0.95, 0.95, textstr_atoms, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    output_path_atoms = 'cremp_atom_count_histogram.png'
    plt.savefig(output_path_atoms)
    print(f"Atom count histogram saved to {output_path_atoms}")

if __name__ == "__main__":
    compute_stats()
