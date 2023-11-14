default_evaluator_kwargs = dict(
    Infidelity=dict(
        num_repeat=200,
        noise_scale=0.2,
        batch_size=32,
    ),
    Sensitivity=dict(
        num_iter=10,
        epsilon=0.2,
    )
)
