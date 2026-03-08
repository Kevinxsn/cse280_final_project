import msprime
import numpy as np
from tqdm import tqdm


'''
	1.	choose parameters (n, \alpha, \theta)
	2.	simulate ancestry under exponential growth
	3.	add mutations
	4.	extract the SNP matrix
	5.	save each replicate as training examples
'''




def simulate_dataset(
    num_replicates=1000,
    n_choices=(20, 50, 100),
    alpha_range=(0.0, 0.05),
    theta_range=(5.0, 50.0),
    rho=10.0,
    sequence_length=50_000,
    Ne=10_000,
    seed=12345,
):
    """
    Generate many SNP matrices and their labels for supervised learning.

    Returns
    -------
    dataset : list of dict
        Each entry contains:
        - 'X': SNP matrix
        - 'positions': SNP positions
        - 'alpha': target label
        - 'theta'
        - 'n_samples'
        - 'metadata'
    """
    rng = np.random.default_rng(seed)
    dataset = []

    for i in tqdm(range(num_replicates)):
        n_samples = int(rng.choice(n_choices))
        alpha = float(rng.uniform(*alpha_range))
        theta = float(rng.uniform(*theta_range))
        sim_seed = int(rng.integers(1, 2**31 - 1))

        G, positions, meta = simulate_snp_matrix(
            n_samples=n_samples,
            alpha=alpha,
            theta=theta,
            rho=rho,
            sequence_length=sequence_length,
            Ne=Ne,
            random_seed=sim_seed,
        )

        dataset.append({
            "X": G,
            "positions": positions,
            "alpha": alpha,
            "theta": theta,
            "n_samples": n_samples,
            "metadata": meta,
        })

    return dataset


dataset = simulate_dataset(
    num_replicates=20000,
    n_choices=(20, 50),
    alpha_range=(0.0, 0.05),
    theta_range=(5.0, 20.0),
    rho=10.0,
    sequence_length=20_000,
    Ne=10_000,
    seed=1,
)

print("Number of replicates:", len(dataset))
print("Example keys:", dataset[0].keys())
print("Example SNP matrix shape:", dataset[0]["X"].shape)
print("Example alpha:", dataset[0]["alpha"])