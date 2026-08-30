import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import marimo as mo

    return mo, np, plt


@app.cell
def _(mo):
    # Interactive parameter sliders
    T_slider = mo.ui.slider(10, 500, value=100, step=10, label="Time steps (T)")
    B0_slider = mo.ui.slider(5.0, 50.0, value=20.0, step=1.0, label="Initial biomass (B0)")
    delta_slider = mo.ui.slider(0.1, 5.0, value=1.0, step=0.1, label="Growth rate (δ)")
    mu_L_slider = mo.ui.slider(0.5, 5.0, value=2.0, step=0.1, label="Low patch mean (μL)")
    mu_H_slider = mo.ui.slider(0.5, 10.0, value=5.0, step=0.1, label="High patch mean (μH)")
    sigma_slider = mo.ui.slider(0.1, 3.0, value=1.5, step=0.1, label="Noise std (σ)")
    q0_slider = mo.ui.slider(0.1, 0.9, value=0.5, step=0.05, label="Prior P(high) (q0)")

    mo.vstack([
        T_slider, B0_slider, delta_slider,
        mu_L_slider, mu_H_slider, sigma_slider, q0_slider
    ])

    return (
        B0_slider,
        T_slider,
        delta_slider,
        mu_H_slider,
        mu_L_slider,
        q0_slider,
        sigma_slider,
    )


@app.cell
def _(
    B0_slider,
    T_slider,
    delta_slider,
    mu_H_slider,
    mu_L_slider,
    np,
    q0_slider,
    sigma_slider,
):
    # Extract slider values
    T = T_slider.value
    B0 = B0_slider.value
    delta = delta_slider.value
    mu_L = mu_L_slider.value
    mu_H = mu_H_slider.value
    sigma = sigma_slider.value
    q0 = q0_slider.value

    # Seed for reproducibility
    rng = np.random.default_rng(1)

    # True, fixed patch types: each patch is independently low or high
    is_high = rng.random(2) < q0
    mu_true = np.where(is_high, mu_H, mu_L)

    # Initial root masses and beliefs
    b = np.array([B0 / 2, B0 / 2])
    y = np.full(2, np.log(q0 / (1 - q0)))
    return T, b, delta, is_high, mu_H, mu_L, mu_true, rng, sigma, y


@app.cell
def _(T, b, delta, mu_H, mu_L, mu_true, np, rng, sigma, y):
    samples = []
    posterior_high = []
    allocation = []
    rewards = []
    _b=b.copy()
    _y=y.copy()
    for t in range(T):
        # Existing roots sample both patches
        r = rng.normal(mu_true, sigma)
        p = _b / _b.sum()
        rewards.append(np.dot(p, r))
        samples.append(r)

        # Bayesian update
        _y += (mu_H - mu_L) / sigma**2 * (
            r - (mu_H + mu_L) / 2
        )
        q = 1 / (1 + np.exp(-_y))
        posterior_high.append(q)

        # Greedy allocation of new, irreversible root growth
        expected_rate = q * mu_H + (1 - q) * mu_L
        d = np.zeros(2)
        if expected_rate[0] > expected_rate[1]:
            d[0] = delta
        elif expected_rate[1] > expected_rate[0]:
            d[1] = delta
        else:
            d[:] = delta / 2

        _b = _b + d
        allocation.append(_b / _b.sum())

    samples = np.array(samples)
    posterior_high = np.array(posterior_high)
    allocation = np.array(allocation)

    return allocation, posterior_high, rewards


@app.cell
def _(is_high, plt, posterior_high):
    # Plot 1: Posterior belief evolution
    fig1, ax1 = plt.subplots(figsize=(10, 5))

    for i in range(2):
        ax1.plot(posterior_high[:, i], label=f"patch {i + 1}", linewidth=2)
        ax1.axhline(float(is_high[i]), color=f"C{i}", ls="--", alpha=0.5, linewidth=1.5)

    ax1.set_xlabel("time step")
    ax1.set_ylabel(r"$P(z_i=H\mid\mathcal{H}_{i,t})$")
    ax1.set_title("Posterior Belief Evolution")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend()
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    fig1
    return


@app.cell
def _(allocation, plt):
    # Plot 2: Root mass allocation
    fig2, ax2 = plt.subplots(figsize=(10, 5))

    ax2.plot(allocation[:, 0], label="patch 1", linewidth=2)
    ax2.plot(allocation[:, 1], label="patch 2", linewidth=2)
    ax2.set_xlabel("time step")
    ax2.set_ylabel("root-mass proportion")
    ax2.set_title("Root Mass Allocation Over Time")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    fig2
    return


@app.cell
def _(allocation, is_high, mo, np, rewards):
    # Summary statistics
    mo.md(f"""
    ## Simulation Summary
    - **True patch types:** Patch 1 = {"High" if is_high[0] else "Low"}, Patch 2 = {"High" if is_high[1] else "Low"}
    - **Mean reward per step:** {np.mean(rewards):.3f}
    - **Final root allocation:** Patch 1: {allocation[-1, 0]:.2%}, Patch 2: {allocation[-1, 1]:.2%}
    """)
    return


if __name__ == "__main__":
    app.run()
