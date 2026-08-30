import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import typing

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Two-pot Bayesian root allocation

    This notebook develops the first, passive-sampling version of the model.
    A plant grows irreversible roots in two patches. Each patch is either low
    mean or high mean, and the plant uses noisy samples to infer its type.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Patch types and resource samples

    Each patch has a fixed but unobserved type $z_i\in\{L,H\}$, for
    $i\in\{1,2\}$. A low-mean patch has resource mean $\mu_L$ and a
    high-mean patch has resource mean $\mu_H$, where $\mu_H>\mu_L$.
    Before observing a patch, the plant assigns the prior probabilities

    $$
    P(z_i=H)=q_0,\qquad P(z_i=L)=1-q_0.
    $$

    The plant assumes that the two possible resource distributions are known:

    $$
    f_L(r)=\frac{1}{\sqrt{2\pi\sigma^2}}
    \exp\left(-\frac{(r-\mu_L)^2}{2\sigma^2}\right),
    $$

    $$
    f_H(r)=\frac{1}{\sqrt{2\pi\sigma^2}}
    \exp\left(-\frac{(r-\mu_H)^2}{2\sigma^2}\right).
    $$

    Thus, resource samples are Normally distributed with common known variance
    $\sigma^2$. The plant does not infer $f_L$ or $f_H$ in this model; it
    infers which of them is the source distribution of each patch.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Irreversible root allocation

    At time $t$, total root mass is

    $$
    B_t=B_0+t\delta,
    $$

    where $B_0$ is initial root mass and $\delta$ is the new root mass produced
    in each time step. Root masses in the two patches are

    $$
    \mathbf b_t=(b_{1,t},b_{2,t}),\qquad b_{1,t}+b_{2,t}=B_t,
    $$

    with proportions $p_{i,t}=b_{i,t}/B_t$. The action at time $t$ is the
    allocation of the new mass,

    $$
    \mathbf d_t=(d_{1,t},d_{2,t}),\qquad d_{1,t}+d_{2,t}=\delta,
    $$

    and root mass changes irreversibly:

    $$
    b_{i,t+1}=b_{i,t}+d_{i,t}.
    $$

    The normalized reward is

    $$
    R_t=p_{1,t}r_{1,t}+p_{2,t}r_{2,t}.
    $$

    The model is evaluated over a finite horizon $T$, with objective

    $$
    \max_{\pi}\;\mathbb E^\pi\left[\sum_{t=0}^{T}R_t\right].
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bayesian inference

    Let $\mathcal H_{i,t}=\{r_{i,0},\ldots,r_{i,t-1}\}$ be the samples from
    patch $i$ before time $t$. The plant's belief that patch $i$ is high mean is

    $$
    q_{i,t}=P(z_i=H\mid\mathcal H_{i,t}).
    $$

    After observing $r_{i,t}$, Bayes' rule gives

    $$
    q_{i,t+1}=
    \frac{q_{i,t}f_H(r_{i,t})}
    {q_{i,t}f_H(r_{i,t})+(1-q_{i,t})f_L(r_{i,t})}.
    $$

    It is convenient to define the log posterior odds

    $$
    y_{i,t}=\log\frac{q_{i,t}}{1-q_{i,t}},
    \qquad
    y_{i,0}=\log\frac{q_0}{1-q_0}.
    $$

    Dividing the high-type and low-type posterior probabilities gives

    $$
    y_{i,t+1}=y_{i,t}+\log\frac{f_H(r_{i,t})}{f_L(r_{i,t})}.
    $$

    For the two Normal densities above,

    $$
    \log\frac{f_H(r_{i,t})}{f_L(r_{i,t})}
    =\frac{\mu_H-\mu_L}{\sigma^2}
    \left(r_{i,t}-\frac{\mu_H+\mu_L}{2}\right),
    $$

    and therefore

    $$
    y_{i,t+1}=y_{i,t}+
    \frac{\mu_H-\mu_L}{\sigma^2}
    \left(r_{i,t}-\frac{\mu_H+\mu_L}{2}\right).
    $$

    The posterior probability is recovered as

    $$
    q_{i,t}=\frac{1}{1+e^{-y_{i,t}}},
    $$

    and the expected resource rate of patch $i$ is

    $$
    \hat r_{i,t}=q_{i,t}\mu_H+(1-q_{i,t})\mu_L.
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Allocation policy and limiting case

    At each time step, the plant samples both patches, updates its beliefs, and
    allocates the new mass. The greedy Bayesian policy is

    $$
    d_{1,t}=
    \begin{cases}
    \delta, & \hat r_{1,t+1}>\hat r_{2,t+1},\\
    0, & \hat r_{1,t+1}<\hat r_{2,t+1},\\
    \delta/2, & \hat r_{1,t+1}=\hat r_{2,t+1},
    \end{cases}
    \qquad
    d_{2,t}=\delta-d_{1,t}.
    $$

    In this passive-sampling baseline, each occupied patch provides one sample
    per time step regardless of its root mass. Allocation therefore affects
    reward but not the rate of learning.

    For finite $T$, early noisy samples can lead to growth in the wrong patch,
    and this growth cannot be moved later. In the limit $T\to\infty$, repeated
    samples identify the true patch type: $q_{i,t}\to1$ for a high-mean patch
    and $q_{i,t}\to0$ for a low-mean patch.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    controls = mo.md(
        r"""
        ## Simulation controls

        {horizon}

        {initial_mass}

        {growth}

        {low_mean}

        {high_mean}

        {noise}

        {prior}

        {seed}
        """
    ).batch(
        horizon=mo.ui.slider(
            start=5, stop=300, step=5, value=100,
            label="Time steps $T$", show_value=True,
        ),
        initial_mass=mo.ui.slider(
            start=2, stop=100, step=2, value=20,
            label="Initial root mass $B_0$", show_value=True,
        ),
        growth=mo.ui.slider(
            start=0.1, stop=10.0, step=0.1, value=1.0,
            label="New root mass per step $\\delta$", show_value=True,
        ),
        low_mean=mo.ui.slider(
            start=0.5, stop=9.0, step=0.5, value=2.0,
            label="Low mean $\\mu_L$", show_value=True,
        ),
        high_mean=mo.ui.slider(
            start=1.0, stop=10.0, step=0.5, value=5.0,
            label="High mean $\\mu_H$", show_value=True,
        ),
        noise=mo.ui.slider(
            start=0.1, stop=5.0, step=0.1, value=1.5,
            label="Sample standard deviation $\\sigma$", show_value=True,
        ),
        prior=mo.ui.slider(
            start=0.05, stop=0.95, step=0.05, value=0.5,
            label="Prior probability $q_0$", show_value=True,
        ),
        seed=mo.ui.slider(
            start=0, stop=100, step=1, value=1,
            label="Random seed", show_value=True,
        ),
    )
    controls
    return (controls,)


@app.cell
def _(controls, np):
    from typing import cast
    values = cast(dict[str, float], controls.value)
    T = int(values["horizon"])
    B0 = float(values["initial_mass"])
    delta = float(values["growth"])
    mu_L = float(values["low_mean"])
    mu_H = max(float(values["high_mean"]), mu_L + 0.1)
    sigma = float(values["noise"])
    q0 = float(values["prior"])
    seed = int(values["seed"])

    rng = np.random.default_rng(seed)
    standard_noise = rng.normal(size=(T, 2))

    def simulate(true_means):
        b = np.array([B0 / 2, B0 / 2], dtype=float)
        y = np.full(2, np.log(q0 / (1 - q0)), dtype=float)

        posterior_high = np.empty((T + 1, 2))
        allocation = np.empty((T + 1, 2))
        rewards = np.empty(T)

        posterior_high[0] = q0
        allocation[0] = b / b.sum()

        for t in range(T):
            r = true_means + sigma * standard_noise[t]
            p = b / b.sum()
            rewards[t] = np.dot(p, r)

            y += (mu_H - mu_L) / sigma**2 * (
                r - (mu_H + mu_L) / 2
            )
            q = 1 / (1 + np.exp(-y))
            posterior_high[t+1] = q

            expected_rate = q * mu_H + (1 - q) * mu_L
            d = np.zeros(2)
            if expected_rate[0] > expected_rate[1]:
                d[0] = delta
            elif expected_rate[1] > expected_rate[0]:
                d[1] = delta
            else:
                d[:] = delta / 2

            b = b + d
            allocation[t+1] = b / b.sum()

        return {
            "posterior_high": posterior_high,
            "allocation": allocation,
            "mean_reward": rewards.mean(),
        }

    scenarios = {
        "low-low": np.array([mu_L, mu_L]),
        "high-high": np.array([mu_H, mu_H]),
        "low-high": np.array([mu_L, mu_H]),
    }
    results = {name: simulate(means) for name, means in scenarios.items()}
    return T, mu_H, results, scenarios, sigma


@app.cell(hide_code=True)
def _(T, mo):
    mo.md(
        f"""
        ## Results

        Each column below is one possible two-pot environment. The upper row is
        the posterior probability that each pot is high mean. Dashed horizontal
        lines show the true hidden type: $0$ for low mean and $1$ for high mean.
        The lower row shows the root-mass proportions after each allocation
        decision. The horizon in this run is $T={T}$.
        """
    )
    return


@app.cell
def _(T, mu_H, plt, results, scenarios, sigma):
    figure, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    time = range(T + 1)

    for column, (name, true_means) in enumerate(scenarios.items()):
        posterior_axis = axes[0, column]
        allocation_axis = axes[1, column]
        result = results[name]

        for i, color in enumerate(["C0", "C1"]):
            true_type = float(true_means[i] == mu_H)
            posterior_axis.plot(
                time,
                result["posterior_high"][:, i],
                color=color,
                label=f"pot {i + 1}",
            )
            posterior_axis.axhline(
                true_type, color=color, linestyle="--", alpha=0.55
            )
            allocation_axis.plot(
                time,
                result["allocation"][:, i],
                color=color,
                label=f"pot {i + 1}",
            )

        posterior_axis.set_title(
            f"{name}: means ({true_means[0]:.1f}, {true_means[1]:.1f})"
        )
        posterior_axis.set_ylim(-0.05, 1.05)
        posterior_axis.set_ylabel(r"$P(z_i=H\mid H_{i,t})$")
        posterior_axis.legend(loc="best")

        allocation_axis.set_ylim(-0.05, 1.05)
        allocation_axis.set_xlabel("time step")
        allocation_axis.set_ylabel("root-mass proportion")
        allocation_axis.legend(loc="best")

    figure.suptitle("Bayesian learning and irreversible root allocation, noise variance = {:.2f}".format(sigma), y=1.02)
    figure.tight_layout()
    figure
    return


@app.cell(hide_code=True)
def _(mo, results):
    rows = "\n".join(
        f"| {name} | {result['mean_reward']:.3f} |"
        for name, result in results.items()
    )
    mo.md(
        "## Mean normalized reward\n\n"
        "| scenario | mean reward over the simulated horizon |\n"
        "|---|---:|\n"
        f"{rows}"
    )
    return


if __name__ == "__main__":
    app.run()
