import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import typing
    import pandas as pd
    import itertools
    import seaborn as sns


    return mo, np, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plant foraging and prior
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Two-pot Bayesian root allocation
    We aim to model the two-pot expeerimental setting using bayesian inference tools, in order to have a plant foraging model of plants.
    In the two-pots expereiment setting, at an early root development stage, the root system is split into 2, and is reallocataed into two seperate pots.
    The procces enable to check the plant decision between the two pots, while experimenting with diffrent regimes.

    First, we will introduce a basic model, with 2 types of resource level (low, high), and the prior is the probability for a patch to be from the "high" patch. Root allocation is treated as irreversible, and the samples from the patches are noisy.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Patch types and resource samples

    Each patch has a fixed but unobserved type $z_i\in\{L,H\}$, for
    $i\in\{1,2\}$, with resource mean of $\mu_L, \mu_H$ respectively, where $\mu_H>\mu_L$.
    Before observing a patch (sampling), the plant assigns the prior probabilities

    $$
    P(z_i=H)=q_0,\qquad P(z_i=L)=1-q_0.
    $$

    The noise is assumed to be gaussian, independent of the mean, and known. Hence, the resource distributions are:

    $$
    f_L(r)=\frac{1}{\sqrt{2\pi\sigma^2}}
    \exp\left(-\frac{(r-\mu_L)^2}{2\sigma^2}\right),
    $$

    $$
    f_H(r)=\frac{1}{\sqrt{2\pi\sigma^2}}
    \exp\left(-\frac{(r-\mu_H)^2}{2\sigma^2}\right).
    $$

    The plant does not infer $f_L$ or $f_H$ in this model; it
    infers which of them is the source distribution of each patch.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Irreversible root allocation

    It is asumed to have a constant root growth of $\delta$ in each time step, independtaly of the root mass. It is unrealistic assumption, since investment in the root versus the shoot is not a constant parameter.

    At time $t$, total root mass is $B_t=B_0+t\delta$, where $B_0$ is initial root mass and $\delta$ is the new root mass produced in each time step. Root masses in the two patches are

    $$
    \mathbf b_t=(b_{1,t},b_{2,t}),\qquad b_{1,t}+b_{2,t}=B_t,
    $$

    with proportions $p_{i,t}=b_{i,t}/B_t$. The action at time $t$ is the
    allocation of the new mass,

    $$
    \mathbf d_t=(d_{1,t},d_{2,t}),\qquad d_{1,t}+d_{2,t}=\delta.
    $$

    The root mass can only increase, hence each mass change is irreversible:

    $$
    b_{i,t+1}=b_{i,t}+d_{i,t}.
    $$

    In each time step the root sample from the distribution of its patch. The sample is the resouce that the plant get. We would note this reward at time $t$ and patch $i$ as $r_{i,t}$.
    The total reward at time $t$ is

    $$
    R_t=p_{1,t}r_{1,t}+p_{2,t}r_{2,t}.
    $$

    The model is evaluated over a $T$ steps, with objective

    $$
    \max_{\pi}\;\mathbb E^\pi\left[\sum_{t=0}^{T}R_t\right].
    $$
    wheares $\pi$ is the Policy: the set of rule that determines the next step under the current state.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Bayesian inference

    Let $\mathcal H_{i,t}=\{r_{i,0},\ldots,r_{i,t-1}\}$ be the vector of samples from
    patch $i$ before time $t$. The plant's prior for patch $i$ being "high" is the probability $q_{i,t}=P(z_i=H\mid\mathcal H_{i,t})$.

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
    \todo

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
    ### Allocation policy and limiting case

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
        ### Base simulation controls

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
    return B0, T, delta, mu_H, mu_L, q0, seed, sigma, standard_noise


@app.cell
def _(np, standard_noise):
    def simulate(true_means, T,B0, delta, mu_L, mu_H, sigma, q0, seed):
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

        sim_parameters = {"T": T, "B0": B0, "delta": delta, "mu_L": mu_L, "mu_H": mu_H, "sigma": sigma, "q0": q0, "seed": seed}
        return {
            "posterior_high": posterior_high,
            "allocation": allocation,
            "mean_reward": rewards.mean(),
            "sim_parameter" : sim_parameters
        }

    return (simulate,)


@app.cell
def _(B0, T, delta, mu_H, mu_L, np, q0, seed, sigma, simulate):
    scenarios = {
        "low-low": np.array([mu_L, mu_L]),
        "high-high": np.array([mu_H, mu_H]),
        "low-high": np.array([mu_L, mu_H]),
    }
    results = {name: simulate(means, T,B0, delta, mu_L, mu_H, sigma, q0, seed) for name, means in scenarios.items()}
    return results, scenarios


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For an easier comparison of diffrent parameter, the simuation results will convert into data frame:
    """)
    return


@app.cell
def _(pd):
    # `simulations` is the dictionary shown above
    # (with "low-low", "high-high", "low-high", and "sim_parameter" keys)
    def convert_sim_to_df(results):
        rows = []
        for condition, simulation_results in results.items():
            params = simulation_results["sim_parameter"]

            simulation_outputs = {
                name: value
                for name, value in simulation_results.items()
                if name != "sim_parameter"
            }

            mean_map = {
                "low": params["mu_L"],
                "high": params["mu_H"],
            }

            pot_1_type, pot_2_type = condition.split("-")

            row = {
                "pot_1_mean": mean_map[pot_1_type],
                "pot_2_mean": mean_map[pot_2_type],
                **params,
                **simulation_outputs,
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        result_columns = [
            column for column in df.columns
            if column not in {"pot_1_mean", "pot_2_mean", *params.keys()}
        ]

        df = df[["pot_1_mean", "pot_2_mean", *params.keys(), *result_columns]]
        return df

    return (convert_sim_to_df,)


@app.cell
def _(convert_sim_to_df, results):
    df_chap1 = convert_sim_to_df(results)
    df_chap1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since there is no depenency of the sampling rate in the mass, is enough to check on a hight T value, since the model behaviour in the first part of this simulation is the same as checking lower T value.
    """)
    return


@app.cell
def _(convert_sim_to_df, pd, scenarios, simulate, true_means):
    _mean_pairs = [(2,5), (1, 5), (1, 2), (2,3)]
    _stds = [1, 1.5, 3, ]
    _T, _B0, _delta = 100, 1, 1 
    _seed = 1 
    _q0 = 0.5
    _num_simulations = len(_mean_pairs) * len(_stds)*3
    _df_lists = [None] * _num_simulations
    _i = 0
    for mean_pair in _mean_pairs:
        for std in _stds: 
            _mu_L, _mu_H = mean_pair
            _results = {name: simulate(true_means, _T,_B0, _delta, _mu_L, _mu_H, std, _q0, _seed) for name, means in scenarios.items()}
            _df = convert_sim_to_df(_results)
            _df_lists[_i]= _df
            _i+=   1


    res_df = pd.concat(_df_lists, ignore_index=True)        
    res_df   
    return (res_df,)


@app.cell
def _(pd, res_df):
    trajectory_columns = ["posterior_high", "allocation"]

    # Keep every parameter/identifier column from the original dataframe.
    identifier_columns = [
        column
        for column in res_df.columns
        if column not in trajectory_columns + ["mean_reward"]
    ]

    _rows = []

    for simulation_id, simulation in res_df.iterrows():
        identifiers = simulation[identifier_columns].to_dict()

        for result_name in trajectory_columns:
            result_array = simulation[result_name]

            for _t in range(result_array.shape[0]):
                for _pot_index in range(2):
                    _rows.append(
                        {
                            **identifiers,
                            "result": result_name,
                            "t": _t,
                            "pot": _pot_index + 1,
                            "pot_mean": simulation[f"pot_{_pot_index + 1}_mean"],
                            "value": result_array[_t, _pot_index],
                            "simulation_id": simulation_id,
                        }
                    )

    res_df_long = pd.DataFrame(_rows)

    res_df_long
    return (res_df_long,)


@app.cell
def _(res_df_long):
    res_df_long
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    we will analyze seperately the low-low and high-high scenarios, and the low-high scenarios.
    """)
    return


@app.cell
def _(res_df_long):
    df_low_high_long = res_df_long.loc[res_df_long["pot_1_mean"]<res_df_long["pot_2_mean"]]
    return


@app.cell
def _(plt, res_df_long, sns):
    _g = sns.relplot(data=res_df_long.loc[res_df_long["result"]=="posterior_high"].loc[res_df_long["pot"]==1], x="t", y="value", hue="sigma", row="mu_L", col="mu_H")

    for _axis in _g.axes.flat:
        _axis.axhline(0.5, color="black", linestyle="--", alpha=0.6)

    _g.set_axis_labels(
        "Time step",
        "Posterior probability that pot 1 is high",
    )

    _g.set(ylim=(-0.05, 1.05))
    _g.set_titles(row_template=r"$\mu_L={row_name}$", col_template=r"$\mu_H={col_name}$")

    _g.figure.suptitle(
        "Learning about pot 1 across mean pairs and noise levels",
        y=1.02,
    )
    _g.figure.tight_layout()
    plt.show
    return


@app.cell
def _(plt, res_df_long, sns):
    _g = sns.relplot(data=res_df_long.loc[res_df_long["result"]=="posterior_high"].loc[res_df_long["pot"]==2], x="t", y="value", hue="sigma", row="mu_L", col="mu_H")

    for _axis in _g.axes.flat:
        _axis.axhline(0.5, color="black", linestyle="--", alpha=0.6)

    _g.set_axis_labels(
        "Time step",
        "Posterior probability that pot 1 is high",
    )

    _g.set(ylim=(-0.05, 1.05))
    _g.set_titles(row_template=r"$\mu_L={row_name}$", col_template=r"$\mu_H={col_name}$")

    _g.figure.suptitle(
        "Learning about pot 1 across mean pairs and noise levels",
        y=1.02,
    )
    _g.figure.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since there is mass in both pots, and there is yet no relation between the resouce level in one pot to the second, the result show that the converges of the posterior probability to be from high type in pot $i$ depend on the mean in the pot and the std.
    """)
    return


@app.cell(hide_code=True)
def _(T, mo):
    mo.md(f"""
    ## Results

    Each column below is one possible two-pot environment. The upper row is
    the posterior probability that each pot is high mean. Dashed horizontal
    lines show the true hidden type: $0$ for low mean and $1$ for high mean.
    The lower row shows the root-mass proportions after each allocation
    decision. The horizon in this run is $T={T}$.
    """)
    return


app._unparsable_cell(
    r"""
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
                color=color,   `                         label=f"pot {i + 1}",
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
    """,
    name="_"
)


@app.cell
def _():
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
