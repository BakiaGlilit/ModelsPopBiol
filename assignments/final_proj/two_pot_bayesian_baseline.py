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
    return B0, T, cast, delta, mu_H, mu_L, q0, seed, sigma, standard_noise


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
def _(cast, mass_controls, np):
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

    _standard_noise = np.random.default_rng(mass_seed).normal(
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
            ) * _standard_noise[time_index]
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
    mo.md(f"""
    ## Results

    The upper row shows posterior probabilities of a high-mean patch. The
    middle row shows irreversible root-mass proportions. The lower row shows
    the standard deviation of the next observation, $\\sigma/\\sqrt{{b_{{i,t}}}}$.
    The current policy is **{mass_policy}** and the horizon is $T={mass_T}$.
    Dashed lines in the upper row show the true hidden types.
    """)
    return


@app.cell
def _(mass_T, mass_mu_H, mass_results, mass_scenarios, plt):
    _mass_figure, _mass_axes = plt.subplots(
        3, 3, figsize=(14, 10), sharex=True
    )
    mass_time = range(mass_T + 1)

    for _column, (_name, _true_means) in enumerate(mass_scenarios.items()):
        _result = mass_results[_name]
        _posterior_axis = _mass_axes[0, _column]
        _allocation_axis = _mass_axes[1, _column]
        _precision_axis = _mass_axes[2, _column]

        for pot_index, _color in enumerate(["C0", "C1"]):
            _true_type = float(_true_means[pot_index] == mass_mu_H)
            _posterior_axis.plot(
                mass_time,
                _result["posterior_high"][:, pot_index],
                color=_color,
                label=f"pot {pot_index + 1}",
            )
            _posterior_axis.axhline(
                _true_type, color=_color, linestyle="--", alpha=0.55
            )
            _allocation_axis.plot(
                mass_time,
                _result["allocation"][:, pot_index],
                color=_color,
                label=f"pot {pot_index + 1}",
            )
            _precision_axis.plot(
                mass_time,
                _result["observation_sd"][:, pot_index],
                color=_color,
                label=f"pot {pot_index + 1}",
            )

        _posterior_axis.set_title(
            f"{_name}: means ({_true_means[0]:.1f}, {_true_means[1]:.1f})"
        )
        _posterior_axis.set_ylim(-0.05, 1.05)
        _posterior_axis.set_ylabel(r"$P(z_i=H\mid H_{i,t})$")
        _posterior_axis.legend(loc="best")

        _allocation_axis.set_ylim(-0.05, 1.05)
        _allocation_axis.set_ylabel("root-mass proportion")
        _allocation_axis.legend(loc="best")

        _precision_axis.set_xlabel("time step")
        _precision_axis.set_ylabel("next-observation SD")
        _precision_axis.legend(loc="best")

    _mass_figure.suptitle(
        "Root mass changes both allocation and information", y=1.01
    )
    _mass_figure.tight_layout()
    _mass_figure
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


@app.cell
def _():
    import typing as unit_typing

    return (unit_typing,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Information-dependent sampling with discrete root units

    The passive-sampling baseline gave every occupied pot an equally precise
    signal, regardless of allocation. Here, more roots in a pot produce a more
    precise signal. This creates a trade-off between current expected reward and
    learning about a pot for later decisions.

    We use a root sampling unit of mass $\eta$. Let $n_{i,t}$ be the integer
    number of units in pot $i$, and let $G$ be the integer number of new units
    added per time step. Physical mass can be recovered as

    $$
    b_{i,t}=\eta n_{i,t},\qquad B_t=\eta N_t,
    \qquad \delta=\eta G.
    $$

    The simulation works directly in units. Consequently, every action is
    biologically feasible and the possible allocations of the new growth are

    $$
    \mathcal A=\{(g,G-g):g=0,1,\ldots,G\}.
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Observation and posterior update

    Each root unit produces one independent resource encounter. The observed
    signal $x_{i,t}$ is the average across the $n_{i,t}$ units in pot $i$.
    Conditional on the hidden patch type,

    $$
    (x_{i,t}\mid z_i=L,n_{i,t})
    \sim\mathcal N\left(\mu_L,\frac{\sigma^2}{n_{i,t}}\right),
    $$

    $$
    (x_{i,t}\mid z_i=H,n_{i,t})
    \sim\mathcal N\left(\mu_H,\frac{\sigma^2}{n_{i,t}}\right).
    $$

    Let $q_{i,t}=P(z_i=H\mid\mathcal H_{i,t})$. After observing $x_{i,t}$,
    the plant updates its belief by Bayes' rule. The code uses equivalent log
    posterior odds, $y_{i,t}=\log(q_{i,t}/(1-q_{i,t}))$, because they give the
    stable update

    $$
    y_{i,t+1}=y_{i,t}+
    \frac{n_{i,t}(\mu_H-\mu_L)}{\sigma^2}
    \left(x_{i,t}-\frac{\mu_H+\mu_L}{2}\right).
    $$

    Posterior probabilities are recovered by

    $$
    q_{i,t}=\frac{1}{1+e^{-y_{i,t}}}.
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Greedy and rollout policies

    After processing signals at time $t$, the decision state is

    $$
    S_t=(\mathbf n_t,\mathbf q_{t+1}).
    $$

    The posterior expected resource rate in pot $i$ is

    $$
    \hat\mu_{i,t}=q_{i,t+1}\mu_H+(1-q_{i,t+1})\mu_L.
    $$

    The greedy policy maximizes only expected reward at the next time step. Its
    expected reward is linear in $g_{1,t}$, so it allocates all $G$ new units to
    the pot with the larger $\hat\mu_{i,t}$. In a tie, the code uses the most
    even feasible split.

    The full model is a finite-horizon belief-state MDP. Its optimal value
    function satisfies

    $$
    J_t(S_t)=\max_{\mathbf g_t\in\mathcal A}
    \mathbb E\left[
    R_{t+1}+J_{t+1}(S_{t+1})
    \mid S_t,\mathbf g_t
    \right].
    $$

    The two-step rollout is an approximation to this equation. For every
    feasible current action, it simulates the next two rewards and then uses
    the greedy policy for the second action:

    $$
    Q_t^{(2)}(S_t,\mathbf g_t)=
    \mathbb E\left[
    R_{t+1}+R_{t+2}
    \mid S_t,\mathbf g_t,
    \mathbf g_{t+1}=\pi_{t+1}^{\mathrm G}(S_{t+1})
    \right].
    $$

    The rollout action is the feasible action with the largest simulated value.
    """)
    return


@app.cell
def _(mo):
    unit_horizon = mo.ui.slider(
        start=10, stop=150, step=5, value=60,
        label="Time steps $T$", show_value=True,
    )
    unit_eta = mo.ui.slider(
        start=0.1, stop=5.0, step=0.1, value=1.0,
        label="Mass of one root unit $\\eta$", show_value=True,
    )
    unit_initial_units = mo.ui.slider(
        start=2, stop=80, step=2, value=8,
        label="Initial root units $N_0$", show_value=True,
    )
    unit_growth_units = mo.ui.slider(
        start=1, stop=12, step=1, value=2,
        label="New root units per step $G$", show_value=True,
    )
    unit_low_mean = mo.ui.slider(
        start=0.0, stop=8.0, step=0.5, value=2.0,
        label="Low mean $\\mu_L$", show_value=True,
    )
    unit_high_mean = mo.ui.slider(
        start=1.0, stop=12.0, step=0.5, value=5.0,
        label="High mean $\\mu_H$", show_value=True,
    )
    unit_noise = mo.ui.slider(
        start=0.5, stop=10.0, step=0.5, value=5.0,
        label="Single-unit signal SD $\\sigma$", show_value=True,
    )
    unit_prior = mo.ui.slider(
        start=0.05, stop=0.95, step=0.05, value=0.5,
        label="Prior probability $q_0$", show_value=True,
    )
    unit_scenario = mo.ui.dropdown(
        options=["low-low", "high-high", "low-high"],
        value="low-high", label="True patch types",
    )
    unit_planning_rollouts = mo.ui.slider(
        start=20, stop=200, step=20, value=80,
        label="Planning simulations per feasible action $M$", show_value=True,
    )
    unit_repetitions = mo.ui.slider(
        start=5, stop=50, step=5, value=20,
        label="Repeated simulations for comparison", show_value=True,
    )
    unit_seed = mo.ui.slider(
        start=0, stop=100, step=1, value=4,
        label="Random seed", show_value=True,
    )
    unit_controls = mo.md(
        f"""
        {unit_horizon}

        {unit_eta}

        {unit_initial_units}

        {unit_growth_units}

        {unit_low_mean}

        {unit_high_mean}

        {unit_noise}

        {unit_prior}

        {unit_scenario}

        {unit_planning_rollouts}

        {unit_repetitions}

        {unit_seed}
        """
    ).batch(
        horizon=unit_horizon,
        eta=unit_eta,
        initial_units=unit_initial_units,
        growth_units=unit_growth_units,
        low_mean=unit_low_mean,
        high_mean=unit_high_mean,
        noise=unit_noise,
        prior=unit_prior,
        scenario=unit_scenario,
        planning_rollouts=unit_planning_rollouts,
        repetitions=unit_repetitions,
        seed=unit_seed,
    )
    unit_controls
    return (unit_controls,)


@app.cell
def _(np, unit_controls, unit_typing):
    unit_values = unit_controls.value
    unit_T = int(unit_typing.cast(float, unit_values["horizon"]))
    unit_eta_value = float(unit_typing.cast(float, unit_values["eta"]))
    unit_N0 = int(unit_typing.cast(float, unit_values["initial_units"]))
    unit_G = int(unit_typing.cast(float, unit_values["growth_units"]))
    unit_mu_L = float(unit_typing.cast(float, unit_values["low_mean"]))
    unit_mu_H = max(
        float(unit_typing.cast(float, unit_values["high_mean"])), unit_mu_L + 0.1
    )
    unit_sigma = float(unit_typing.cast(float, unit_values["noise"]))
    unit_q0 = float(unit_typing.cast(float, unit_values["prior"]))
    unit_M = int(unit_typing.cast(float, unit_values["planning_rollouts"]))
    unit_n_repetitions = int(unit_typing.cast(float, unit_values["repetitions"]))
    unit_seed_value = int(unit_typing.cast(float, unit_values["seed"]))
    unit_scenario_name = unit_typing.cast(str, unit_values["scenario"])
    unit_true_mean_options = {
        "low-low": np.array([unit_mu_L, unit_mu_L]),
        "high-high": np.array([unit_mu_H, unit_mu_H]),
        "low-high": np.array([unit_mu_L, unit_mu_H]),
    }
    unit_true_means = unit_true_mean_options[unit_scenario_name]
    unit_action_set = np.array(
        [(units_to_pot_1, unit_G - units_to_pot_1)
        for units_to_pot_1 in range(unit_G + 1)],
        dtype=int,
    )
    unit_B0 = unit_eta_value * unit_N0
    unit_delta = unit_eta_value * unit_G
    return (
        unit_B0,
        unit_G,
        unit_M,
        unit_N0,
        unit_T,
        unit_action_set,
        unit_delta,
        unit_eta_value,
        unit_mu_H,
        unit_mu_L,
        unit_n_repetitions,
        unit_q0,
        unit_scenario_name,
        unit_seed_value,
        unit_sigma,
        unit_true_means,
    )


@app.cell(hide_code=True)
def _(mo, unit_B0, unit_G, unit_delta, unit_eta_value):
    mo.md(f"""
    The current physical interpretation is $\\eta={unit_eta_value:.2f}$ mass
    units per root sampling unit, $B_0={unit_B0:.2f}$ initial mass, and
    $\\delta={unit_delta:.2f}$ new mass per step. The plant chooses among
    exactly **{unit_G + 1}** feasible allocations at each decision time.
    """)
    return


@app.cell
def _(np, unit_G, unit_mu_H, unit_mu_L, unit_sigma):
    unit_midpoint = (unit_mu_H + unit_mu_L) / 2
    unit_mean_gap = unit_mu_H - unit_mu_L

    def unit_sigmoid(log_odds):
        return 1 / (1 + np.exp(-np.clip(log_odds, -700, 700)))

    def unit_update_log_odds(log_odds, signal, root_units):
        return log_odds + root_units * unit_mean_gap / unit_sigma**2 * (
            signal - unit_midpoint
        )

    def unit_posterior_means(probability_high):
        return probability_high * unit_mu_H + (1 - probability_high) * unit_mu_L

    def unit_greedy_action(probability_high):
        expected_means = unit_posterior_means(probability_high)
        if expected_means[0] > expected_means[1]:
            return np.array([unit_G, 0], dtype=int)
        if expected_means[1] > expected_means[0]:
            return np.array([0, unit_G], dtype=int)
        return np.array([unit_G // 2, unit_G - unit_G // 2], dtype=int)

    return unit_greedy_action, unit_sigmoid, unit_update_log_odds


@app.cell
def _(
    np,
    unit_M,
    unit_action_set,
    unit_greedy_action,
    unit_mu_H,
    unit_mu_L,
    unit_sigma,
    unit_sigmoid,
    unit_update_log_odds,
):
    def unit_rollout_action(root_units, log_odds, planning_rng):
        """Evaluate every feasible first action, then act greedily once."""
        probability_high = unit_sigmoid(log_odds)
        action_values = np.empty(len(unit_action_set))

        for action_index, first_action in enumerate(unit_action_set):
            first_units = root_units + first_action
            simulated_total_reward = 0.0

            for _ in range(unit_M):
                sampled_high_types = planning_rng.random(2) < probability_high
                sampled_means = np.where(
                    sampled_high_types, unit_mu_H, unit_mu_L
                )
                first_signal = planning_rng.normal(
                    sampled_means, unit_sigma / np.sqrt(first_units)
                )
                first_reward = np.dot(
                    first_units / first_units.sum(), first_signal
                )
                first_log_odds = unit_update_log_odds(
                    log_odds, first_signal, first_units
                )

                second_probability = unit_sigmoid(first_log_odds)
                second_action = unit_greedy_action(second_probability)
                second_units = first_units + second_action
                second_signal = planning_rng.normal(
                    sampled_means, unit_sigma / np.sqrt(second_units)
                )
                second_reward = np.dot(
                    second_units / second_units.sum(), second_signal
                )
                simulated_total_reward += first_reward + second_reward

            action_values[action_index] = simulated_total_reward / unit_M

        return unit_action_set[int(np.argmax(action_values))].copy()

    return (unit_rollout_action,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Matched policy simulations

    Both policies begin from the same state and receive the same standardized
    environmental noise realization. Thus their difference is due to the policy
    and the resulting irreversible allocation, rather than unrelated random
    draws. Index $0$ in the posterior and allocation plots is the initial state
    before the first observation.
    """)
    return


@app.cell
def _(
    np,
    unit_N0,
    unit_T,
    unit_greedy_action,
    unit_q0,
    unit_sigma,
    unit_true_means,
    unit_update_log_odds,
):
    def unit_simulate_greedy(standard_noise):
        root_units = np.array([unit_N0 // 2, unit_N0 // 2], dtype=int)
        log_odds = np.full(2, np.log(unit_q0 / (1 - unit_q0)))
        posterior = np.empty((unit_T + 1, 2))
        allocation = np.empty((unit_T + 1, 2))
        observation_sd = np.empty((unit_T + 1, 2))
        rewards = np.empty(unit_T)
        posterior[0] = unit_q0
        allocation[0] = root_units / root_units.sum()
        observation_sd[0] = unit_sigma / np.sqrt(root_units)

        for time_index in range(unit_T):
            signal = unit_true_means + (
                unit_sigma / np.sqrt(root_units)
            ) * standard_noise[time_index]
            rewards[time_index] = np.dot(
                root_units / root_units.sum(), signal
            )
            log_odds = unit_update_log_odds(log_odds, signal, root_units)
            probability_high = 1 / (1 + np.exp(-np.clip(log_odds, -700, 700)))
            posterior[time_index + 1] = probability_high
            action = unit_greedy_action(probability_high)
            root_units = root_units + action
            allocation[time_index + 1] = root_units / root_units.sum()
            observation_sd[time_index + 1] = unit_sigma / np.sqrt(root_units)

        return {
            "posterior": posterior,
            "allocation": allocation,
            "observation_sd": observation_sd,
            "rewards": rewards,
            "total_reward": rewards.sum(),
        }

    return (unit_simulate_greedy,)


@app.cell
def _(
    np,
    unit_N0,
    unit_T,
    unit_q0,
    unit_rollout_action,
    unit_seed_value,
    unit_sigma,
    unit_true_means,
    unit_update_log_odds,
):
    def unit_simulate_rollout(standard_noise, planning_seed):
        root_units = np.array([unit_N0 // 2, unit_N0 // 2], dtype=int)
        log_odds = np.full(2, np.log(unit_q0 / (1 - unit_q0)))
        posterior = np.empty((unit_T + 1, 2))
        allocation = np.empty((unit_T + 1, 2))
        observation_sd = np.empty((unit_T + 1, 2))
        rewards = np.empty(unit_T)
        planning_rng = np.random.default_rng(planning_seed + unit_seed_value)
        posterior[0] = unit_q0
        allocation[0] = root_units / root_units.sum()
        observation_sd[0] = unit_sigma / np.sqrt(root_units)

        for time_index in range(unit_T):
            signal = unit_true_means + (
                unit_sigma / np.sqrt(root_units)
            ) * standard_noise[time_index]
            rewards[time_index] = np.dot(
                root_units / root_units.sum(), signal
            )
            log_odds = unit_update_log_odds(log_odds, signal, root_units)
            probability_high = 1 / (1 + np.exp(-np.clip(log_odds, -700, 700)))
            posterior[time_index + 1] = probability_high
            action = unit_rollout_action(root_units, log_odds, planning_rng)
            root_units = root_units + action
            allocation[time_index + 1] = root_units / root_units.sum()
            observation_sd[time_index + 1] = unit_sigma / np.sqrt(root_units)

        return {
            "posterior": posterior,
            "allocation": allocation,
            "observation_sd": observation_sd,
            "rewards": rewards,
            "total_reward": rewards.sum(),
        }

    return (unit_simulate_rollout,)


@app.cell
def _(
    np,
    unit_T,
    unit_seed_value,
    unit_simulate_greedy,
    unit_simulate_rollout,
):
    unit_standard_noise = np.random.default_rng(unit_seed_value).normal(
        size=(unit_T, 2)
    )
    unit_greedy_result = unit_simulate_greedy(unit_standard_noise)
    unit_rollout_result = unit_simulate_rollout(
        unit_standard_noise, planning_seed=10_000
    )
    return unit_greedy_result, unit_rollout_result


@app.cell
def _(
    plt,
    unit_T,
    unit_greedy_result,
    unit_rollout_result,
    unit_scenario_name,
    unit_true_means,
):
    unit_plot_time = range(unit_T + 1)
    unit_policy_results = [
        ("Greedy", unit_greedy_result),
        ("Two-step rollout", unit_rollout_result),
    ]
    unit_trajectory_figure, unit_trajectory_axes = plt.subplots(
        3, 2, figsize=(13, 10), sharex="col"
    )

    for column_index, (policy_name, policy_result) in enumerate(unit_policy_results):
        unit_trajectory_axes[0, column_index].plot(
            unit_plot_time, policy_result["posterior"][:, 0], label="pot 1"
        )
        unit_trajectory_axes[0, column_index].plot(
            unit_plot_time, policy_result["posterior"][:, 1], label="pot 2"
        )
        unit_trajectory_axes[0, column_index].set_title(policy_name)
        unit_trajectory_axes[0, column_index].set_ylim(-0.05, 1.05)
        unit_trajectory_axes[0, column_index].set_ylabel(r"$P(z_i=H\mid H)$")
        unit_trajectory_axes[0, column_index].legend()

        unit_trajectory_axes[1, column_index].plot(
            unit_plot_time, policy_result["allocation"][:, 0], label="pot 1"
        )
        unit_trajectory_axes[1, column_index].plot(
            unit_plot_time, policy_result["allocation"][:, 1], label="pot 2"
        )
        unit_trajectory_axes[1, column_index].set_ylim(-0.05, 1.05)
        unit_trajectory_axes[1, column_index].set_ylabel("root-unit proportion")
        unit_trajectory_axes[1, column_index].legend()

        unit_trajectory_axes[2, column_index].plot(
            unit_plot_time, policy_result["observation_sd"][:, 0], label="pot 1"
        )
        unit_trajectory_axes[2, column_index].plot(
            unit_plot_time, policy_result["observation_sd"][:, 1], label="pot 2"
        )
        unit_trajectory_axes[2, column_index].set_xlabel("time step")
        unit_trajectory_axes[2, column_index].set_ylabel("next-signal SD")
        unit_trajectory_axes[2, column_index].legend()

    unit_trajectory_figure.suptitle(
        "Discrete root allocation: "
        f"{unit_scenario_name} true means "
        f"({unit_true_means[0]:.1f}, {unit_true_means[1]:.1f})"
    )
    unit_trajectory_figure.tight_layout()
    unit_trajectory_figure
    return


@app.cell
def _(plt, unit_T, unit_greedy_result, unit_rollout_result):
    unit_reward_time = range(1, unit_T + 1)
    unit_reward_figure, unit_reward_axis = plt.subplots(figsize=(8, 4))
    unit_reward_axis.plot(
        unit_reward_time,
        unit_greedy_result["rewards"].cumsum(),
        label="greedy",
    )
    unit_reward_axis.plot(
        unit_reward_time,
        unit_rollout_result["rewards"].cumsum(),
        label="two-step rollout",
    )
    unit_reward_axis.set_xlabel("time step")
    unit_reward_axis.set_ylabel("cumulative normalized reward")
    unit_reward_axis.set_title("One matched environmental realization")
    unit_reward_axis.legend()
    unit_reward_figure.tight_layout()
    unit_reward_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Repeated-simulation comparison

    A single trajectory can favor either policy by chance. The following
    calculation repeats the selected true-patch scenario. In each repetition,
    greedy and rollout receive the same standardized environmental noise.
    Therefore the reported difference estimates

    $$
    \mathbb E\left[
    \sum_{t=1}^{T}R_t^{\mathrm{roll}}
    -\sum_{t=1}^{T}R_t^{\mathrm G}
    \right].
    $$

    A positive estimate means that the two-step rollout obtained a larger
    average normalized reward for the selected parameters. It is not guaranteed
    to be positive for every realization, nor does two-step rollout necessarily
    equal the exact optimal MDP policy.
    """)
    return


@app.cell
def _(
    np,
    unit_T,
    unit_n_repetitions,
    unit_seed_value,
    unit_simulate_greedy,
    unit_simulate_rollout,
):
    unit_greedy_totals = np.empty(unit_n_repetitions)
    unit_rollout_totals = np.empty(unit_n_repetitions)
    unit_evaluation_rng = np.random.default_rng(unit_seed_value + 50_000)

    for repetition_index in range(unit_n_repetitions):
        repetition_noise = unit_evaluation_rng.normal(size=(unit_T, 2))
        greedy_repetition = unit_simulate_greedy(repetition_noise)
        rollout_repetition = unit_simulate_rollout(
            repetition_noise, planning_seed=100_000 + repetition_index
        )
        unit_greedy_totals[repetition_index] = greedy_repetition["total_reward"]
        unit_rollout_totals[repetition_index] = rollout_repetition["total_reward"]

    unit_reward_differences = unit_rollout_totals - unit_greedy_totals
    return unit_greedy_totals, unit_reward_differences, unit_rollout_totals


@app.cell(hide_code=True)
def _(
    mo,
    np,
    unit_greedy_totals,
    unit_reward_differences,
    unit_rollout_totals,
):
    unit_difference_mean = unit_reward_differences.mean()
    unit_difference_se = unit_reward_differences.std(ddof=1) / np.sqrt(
        len(unit_reward_differences)
    )
    unit_rollout_win_fraction = np.mean(unit_reward_differences > 0)
    mo.md(
        f"""
        **Repeated-simulation comparison**

        - Mean cumulative reward, greedy: `{unit_greedy_totals.mean():.2f}`
        - Mean cumulative reward, two-step rollout: `{unit_rollout_totals.mean():.2f}`
        - Mean difference, rollout minus greedy: `{unit_difference_mean:.2f}`
          (Monte Carlo standard error `{unit_difference_se:.2f}`)
        - Fraction of realizations where rollout was larger:
          `{unit_rollout_win_fraction:.2%}`
        """
    )
    return


@app.cell
def _(plt, unit_greedy_totals, unit_rollout_totals):
    unit_summary_figure, unit_summary_axis = plt.subplots(figsize=(7, 4))
    unit_summary_axis.boxplot(
        [unit_greedy_totals, unit_rollout_totals],
        tick_labels=["greedy", "two-step rollout"],
    )
    unit_summary_axis.set_ylabel("cumulative normalized reward")
    unit_summary_axis.set_title("Distribution across repeated simulations")
    unit_summary_figure.tight_layout()
    unit_summary_figure
    return


if __name__ == "__main__":
    app.run()
