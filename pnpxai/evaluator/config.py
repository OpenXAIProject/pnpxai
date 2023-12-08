default_evaluator_kwargs = dict(
    MuFidelity=dict(
        n_perturbations=200,
        noise_scale=0.2,
        batch_size=32,
        grid_size=9,
        baseline=0.0
    ),
    Sensitivity=dict(
        num_iter=10,
        epsilon=0.2,
    ),
    Complexity=dict(
        n_bins=10,
    ),
)
