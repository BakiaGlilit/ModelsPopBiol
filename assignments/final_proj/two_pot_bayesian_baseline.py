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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Root mass as information

    The previous model assumed that every occupied patch produced one equally
    precise sample at every time step. Root allocation therefore changed reward,
    but not learning. Here, root mass also changes the precision of the resource
    signal.

    Let $s_{i,t,k}$ be the resource value encountered by the $k$ th effective
    sampling unit in patch $i$. Conditional on the fixed patch type $z_i$,

    $$
    s_{i,t,k}\mid z_i=L\sim\mathcal N(\mu_L,\sigma^2),
    \qquad
    s_{i,t,k}\mid z_i=H\sim\mathcal N(\mu_H,\sigma^2).
    $$

    We interpret $b_{i,t}$, the root mass at the $i$ th patch in time $t$, as the effective number of independent sampling
    units in patch $i$. The plant observes their average resource rate,

    $$
    x_{i,t}=\frac{1}{b_{i,t}}\sum_{k=1}^{b_{i,t}}u_{i,t,k}.
    $$

    Hence,

    $$
    (x_{i,t}\mid z_i=L,b_{i,t})
    \sim\mathcal N\left(\mu_L,\frac{\sigma^2}{b_{i,t}}\right),
    $$

    $$
    (x_{i,t}\mid z_i=H,b_{i,t})
    \sim\mathcal N\left(\mu_H,\frac{\sigma^2}{b_{i,t}}\right).
    $$

    More root mass therefore gives a more precise estimate of the patch mean.
    The interpretation is an approximation when root mass is continuous:
    $b_{i,t}$ measures effective sampling effort rather than a literal count of
    roots.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Bayesian update with mass-dependent precision

    As before, let

    $$
    q_{i,t}=P(z_i=H\mid\mathcal H_{i,t}),
    \qquad
    y_{i,t}=\log\frac{q_{i,t}}{1-q_{i,t}}.
    $$

    For a given root mass $b$ (in a patch), the two densities of the observed average are

    $$
    f_H^{(b)}(x)=
    \sqrt{\frac{b}{2\pi\sigma^2}}
    \exp\left(-\frac{b(x-\mu_H)^2}{2\sigma^2}\right),
    $$

    $$
    f_L^{(b)}(x)=
    \sqrt{\frac{b}{2\pi\sigma^2}}
    \exp\left(-\frac{b(x-\mu_L)^2}{2\sigma^2}\right).
    $$

    Bayes' rule is still

    $$
    q_{i,t+1}=
    \frac{q_{i,t}f_H^{(b)}(x_{i,t})}
    {q_{i,t}f_H^{(b)}(x_{i,t})+
    (1-q_{i,t})f_L^{(b)}(x_{i,t})}.
    $$

    To obtain the log-odds update, take the ratio of the two Normal densities:

    $$
    \log \frac{f_H^{(b)}(x)}{f_L^{(b)}(x)}
    =-\frac{b(x-\mu_H)^2}{2\sigma^2}
    +\frac{b(x-\mu_L)^2}{2\sigma^2}.
    $$

    Expanding the two squares gives

    $$
    \log\frac{f_H^{(b)}(x)}{f_L^{(b)}(x)}
    =\frac{b(\mu_H-\mu_L)}{\sigma^2}
    \left(x-\frac{\mu_H+\mu_L}{2}\right).
    $$

    Therefore,

    $$
    y_{i,t+1}=y_{i,t}+
    \frac{b_{i,t}(\mu_H-\mu_L)}{\sigma^2}
    \left(x_{i,t}-\frac{\mu_H+\mu_L}{2}\right).
    $$

    The key difference from the previous model is that the evidence
    supplied by a sample is proportional to current root mass $b_{i,t}$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Finite-horizon allocation

    At time $t$, existing roots produce $\mathbf x_t$, the plant updates its
    posterior $\mathbf y_{t+1}$, and then allocates $\mathbf d_t$. The allocation
    changes both future reward and the precision of the next observation:

    $$
    \mathbf b_{t+1}=\mathbf b_t+\mathbf d_t,
    \qquad
    \operatorname{Var}(x_{i,t+1}\mid z_i,b_{i,t+1})
    =\frac{\sigma^2}{b_{i,t+1}}.
    $$

    We assume that the plant knows the total number of time steps, $T$, and the current time step, $t$. Thus, it knows how many time steps remain.

    Let $J_t(\mathbf b_t,\mathbf y_{t+1})$ denote the largest expected total normalized reward that the plant can obtain from time $t+1$ until time $T$, given its current root allocation $\mathbf b_t$ and current beliefs $\mathbf y_{t+1}$.

    The exact finite-horizon value function is

    $$
    J_t(\mathbf b_t,\mathbf y_{t+1})=
    \max_{\substack{\mathbf d_t\geq0\\d_{1,t}+d_{2,t}=\delta}}
    \mathbb E\left[
    R_{t+1}+J_{t+1}(\mathbf b_t+\mathbf d_t,\mathbf y_{t+2})
    \right],
    $$

    with terminal value $J_T=0$. The expectation is over the unknown patch
    types and future resource signals. There is no simple closed-form solution,
    because both allocation and belief are continuous state variables.

    The simulation compares two policies. The greedy policy allocates all new
    mass to the patch with the larger posterior expected mean. The two-step
    rollout policy evaluates a finite set of possible allocations by Monte Carlo
    simulation of the next two time steps, including the information obtained
    after the first future observation. It is an approximation to the value
    function above and can allocate to an uncertain patch when this has future
    information value.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mass_controls = mo.md(
        r"""
        ## Simulation controls

        {horizon}

        {initial_mass}

        {growth}

        {low_mean}

        {high_mean}

        {noise}

        {prior}

        {policy}

        {action_grid}

        {rollouts}

        {seed}
        """
    ).batch(
        horizon=mo.ui.slider(
            start=5, stop=150, step=5, value=60,
            label="Time steps $T$", show_value=True,
        ),
        initial_mass=mo.ui.slider(
            start=2, stop=80, step=2, value=12,
            label="Initial root mass $B_0$", show_value=True,
        ),
        growth=mo.ui.slider(
            start=0.5, stop=8.0, step=0.5, value=2.0,
            label="New root mass per step $\\delta$", show_value=True,
        ),
        low_mean=mo.ui.slider(
            start=0.5, stop=8.0, step=0.5, value=2.0,
            label="Low mean $\\mu_L$", show_value=True,
        ),
        high_mean=mo.ui.slider(
            start=1.0, stop=10.0, step=0.5, value=5.0,
            label="High mean $\\mu_H$", show_value=True,
        ),
        noise=mo.ui.slider(
            start=0.5, stop=8.0, step=0.5, value=4.0,
            label="Per-unit resource SD $\\sigma$", show_value=True,
        ),
        prior=mo.ui.slider(
            start=0.05, stop=0.95, step=0.05, value=0.5,
            label="Prior probability $q_0$", show_value=True,
        ),
        policy=mo.ui.dropdown(
            options=["greedy", "two-step rollout"], value="two-step rollout",
            label="Allocation policy",
        ),
        action_grid=mo.ui.slider(
            start=3, stop=15, step=2, value=7,
            label="Candidate allocations in rollout", show_value=True,
        ),
        rollouts=mo.ui.slider(
            start=20, stop=300, step=20, value=100,
            label="Monte Carlo rollouts per candidate", show_value=True,
        ),
        seed=mo.ui.slider(
            start=0, stop=100, step=1, value=3,
            label="Random seed", show_value=True,
        ),
    )
    mass_controls
    return (mass_controls,)


@app.cell
def _(mass_controls, np):
    from typing import cast

    mass_values = cast(dict[str, object], mass_controls.value)
    mass_T = int(cast(float, mass_values["horizon"]))
    mass_B0 = float(cast(float, mass_values["initial_mass"]))
    mass_delta = float(cast(float, mass_values["growth"]))
    mass_mu_L = float(cast(float, mass_values["low_mean"]))
    mass_mu_H = max(
        float(cast(float, mass_values["high_mean"])), mass_mu_L + 0.1
    )
    mass_sigma = float(cast(float, mass_values["noise"]))
    mass_q0 = float(cast(float, mass_values["prior"]))
    mass_policy = cast(str, mass_values["policy"])
    mass_n_actions = int(cast(float, mass_values["action_grid"]))
    mass_n_rollouts = int(cast(float, mass_values["rollouts"]))
    mass_seed = int(cast(float, mass_values["seed"]))

    mass_midpoint = (mass_mu_H + mass_mu_L) / 2
    mass_difference = mass_mu_H - mass_mu_L

    def mass_sigmoid(log_odds):
        clipped = np.clip(log_odds, -700, 700)
        return 1 / (1 + np.exp(-clipped))

    def mass_update(log_odds, observation, root_mass):
        return log_odds + root_mass * mass_difference / mass_sigma**2 * (
            observation - mass_midpoint
        )

    def mass_greedy_action(probability_high):
        expected_rate = (
            probability_high * mass_mu_H
            + (1 - probability_high) * mass_mu_L
        )
        action = np.zeros(2)
        if expected_rate[0] > expected_rate[1]:
            action[0] = mass_delta
        elif expected_rate[1] > expected_rate[0]:
            action[1] = mass_delta
        else:
            action[:] = mass_delta / 2
        return action

    def mass_rollout_action(root_mass, log_odds, planning_rng):
        """Choose d_t by a two-step Monte Carlo look-ahead."""
        probability_high = mass_sigmoid(log_odds)
        candidate_first_actions = np.linspace(
            0, mass_delta, mass_n_actions
        )
        candidate_values = []

        for first_to_pot_1 in candidate_first_actions:
            first_action = np.array(
                [first_to_pot_1, mass_delta - first_to_pot_1]
            )
            next_mass = root_mass + first_action
            total_value = 0.0

            for _ in range(mass_n_rollouts):
                sampled_high = planning_rng.random(2) < probability_high
                sampled_means = np.where(
                    sampled_high, mass_mu_H, mass_mu_L
                )

                next_signal = planning_rng.normal(
                    sampled_means, mass_sigma / np.sqrt(next_mass)
                )
                next_reward = np.dot(
                    next_mass / next_mass.sum(), next_signal
                )
                next_log_odds = mass_update(
                    log_odds, next_signal, next_mass
                )
                next_probability = mass_sigmoid(next_log_odds)

                second_action = mass_greedy_action(next_probability)
                second_mass = next_mass + second_action
                second_signal = planning_rng.normal(
                    sampled_means, mass_sigma / np.sqrt(second_mass)
                )
                second_reward = np.dot(
                    second_mass / second_mass.sum(), second_signal
                )
                total_value += next_reward + second_reward

            candidate_values.append(total_value / mass_n_rollouts)

        best_index = int(np.argmax(candidate_values))
        best_to_pot_1 = candidate_first_actions[best_index]
        return np.array([best_to_pot_1, mass_delta - best_to_pot_1])

    standard_noise = np.random.default_rng(mass_seed).normal(
        size=(mass_T, 2)
    )

    def mass_simulate(true_means, scenario_index):
        root_mass = np.array([mass_B0 / 2, mass_B0 / 2], dtype=float)
        log_odds = np.full(
            2, np.log(mass_q0 / (1 - mass_q0)), dtype=float
        )

        posterior_high = np.empty((mass_T + 1, 2))
        allocation = np.empty((mass_T + 1, 2))
        observation_sd = np.empty((mass_T + 1, 2))
        rewards = np.empty(mass_T)

        posterior_high[0] = mass_q0
        allocation[0] = root_mass / root_mass.sum()
        observation_sd[0] = mass_sigma / np.sqrt(root_mass)

        planning_rng = np.random.default_rng(
            mass_seed + 1000 + scenario_index
        )

        for time_index in range(mass_T):
            signal = true_means + (
                mass_sigma / np.sqrt(root_mass)
            ) * standard_noise[time_index]
            rewards[time_index] = np.dot(
                root_mass / root_mass.sum(), signal
            )

            log_odds = mass_update(log_odds, signal, root_mass)
            probability_high = mass_sigmoid(log_odds)
            posterior_high[time_index + 1] = probability_high

            if mass_policy == "greedy":
                action = mass_greedy_action(probability_high)
            else:
                action = mass_rollout_action(
                    root_mass, log_odds, planning_rng
                )

            root_mass = root_mass + action
            allocation[time_index + 1] = root_mass / root_mass.sum()
            observation_sd[time_index + 1] = mass_sigma / np.sqrt(root_mass)

        return {
            "posterior_high": posterior_high,
            "allocation": allocation,
            "observation_sd": observation_sd,
            "mean_reward": rewards.mean(),
        }

    mass_scenarios = {
        "low-low": np.array([mass_mu_L, mass_mu_L]),
        "high-high": np.array([mass_mu_H, mass_mu_H]),
        "low-high": np.array([mass_mu_L, mass_mu_H]),
    }
    mass_results = {
        name: mass_simulate(true_means, index)
        for index, (name, true_means) in enumerate(mass_scenarios.items())
    }
    return mass_T, mass_mu_H, mass_policy, mass_results, mass_scenarios


@app.cell(hide_code=True)
def _(mass_T, mass_policy, mo):
    mo.md(
        f"""
        ## Results

        The upper row shows posterior probabilities of a high-mean patch. The
        middle row shows irreversible root-mass proportions. The lower row shows
        the standard deviation of the next observation, $\\sigma/\\sqrt{{b_{{i,t}}}}$.
        The current policy is **{mass_policy}** and the horizon is $T={mass_T}$.
        Dashed lines in the upper row show the true hidden types.
        """
    )
    return


@app.cell
def _(mass_T, mass_mu_H, mass_results, mass_scenarios, plt):
    mass_figure, mass_axes = plt.subplots(
        3, 3, figsize=(14, 10), sharex=True
    )
    mass_time = range(mass_T + 1)

    for column, (name, true_means) in enumerate(mass_scenarios.items()):
        result = mass_results[name]
        posterior_axis = mass_axes[0, column]
        allocation_axis = mass_axes[1, column]
        precision_axis = mass_axes[2, column]

        for pot_index, color in enumerate(["C0", "C1"]):
            true_type = float(true_means[pot_index] == mass_mu_H)
            posterior_axis.plot(
                mass_time,
                result["posterior_high"][:, pot_index],
                color=color,
                label=f"pot {pot_index + 1}",
            )
            posterior_axis.axhline(
                true_type, color=color, linestyle="--", alpha=0.55
            )
            allocation_axis.plot(
                mass_time,
                result["allocation"][:, pot_index],
                color=color,
                label=f"pot {pot_index + 1}",
            )
            precision_axis.plot(
                mass_time,
                result["observation_sd"][:, pot_index],
                color=color,
                label=f"pot {pot_index + 1}",
            )

        posterior_axis.set_title(
            f"{name}: means ({true_means[0]:.1f}, {true_means[1]:.1f})"
        )
        posterior_axis.set_ylim(-0.05, 1.05)
        posterior_axis.set_ylabel(r"$P(z_i=H\mid\mathcal H_{i,t})$")
        posterior_axis.legend(loc="best")

        allocation_axis.set_ylim(-0.05, 1.05)
        allocation_axis.set_ylabel("root-mass proportion")
        allocation_axis.legend(loc="best")

        precision_axis.set_xlabel("time step")
        precision_axis.set_ylabel("next-observation SD")
        precision_axis.legend(loc="best")

    mass_figure.suptitle(
        "Root mass changes both allocation and information", y=1.01
    )
    mass_figure.tight_layout()
    mass_figure
    return


@app.cell(hide_code=True)
def _(mass_results, mo):
    mass_rows = "\n".join(
        f"| {name} | {result['mean_reward']:.3f} |"
        for name, result in mass_results.items()
    )
    mo.md(
        "## Mean normalized reward\n\n"
        "| scenario | mean reward over the simulated horizon |\n"
        "|---|---:|\n"
        f"{mass_rows}"
    )
    return


if __name__ == "__main__":
    app.run()
