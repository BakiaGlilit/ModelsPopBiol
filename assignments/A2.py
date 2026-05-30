import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from numba import jit
    import sympy as sp

    return jit, mo, np, plt, sns


@app.cell
def _(sns):
    red, blue, green = sns.color_palette('Set1', 3) ###
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Assignment 2: Discrete-time deterministic models
    ## [Models in Population Biology](https://modelspopbiol.yoavram.com/)
    ## Yoav Ram
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # General instructions

    1. When instructed to implement a function, use the given function names and parameters lists; failure to do so may cause test functions to fail during grading.
    1. When instructed to generate a plot, make sure that the plot is clear, that axes are propely labeled, and that the notebook is saved with the plot inline, so that the grader can see the plot without running the code. Make sure that you re-generate the plot if you changed the code!
    1. Cells that begin with `###` and lines that end with `###` should not be removed or modified, they are used for automatic grading.
    1. Note that the last cell in the notebook says __end of assignment__; if you are missing anything please download the origianl file from the course website.
    1. This exercise doesn't put much emphasis on efficieny or runtime. But, your code should still run within a reasonable time (a few minutes) and you should use idioms learned in class, e.g. array opreations, wherever possible.
    1. Questions regarding the exercises should be posted to the course forum. You can also visit the Office Hours, but please do not email the course staff with questions about the exercise.
    1. Intructions for submitting the exercise are on the course website.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ex 1: SIS model

    In this exercise we'll model the spread of an infectious disease that spreads through contact with an infected individual.
    Infected individuals remain infected for some time and then become susceptible again (rather than recovering).

    Here, $S$ stands for *susceptible* and $I$ stands for *infected*. The total population size is $N=S+I$.

    Susceptible individuals meet $c$ individuals every day, of which $I/N$ are infected. When meeting an infected individuals, they become infected with probability $b$.
    Thus, on average $\beta S I/N$ susceptible individuals become infected every day, where $\beta=b \cdot c$ is the transmission rate.
    On average, $\gamma I$ infected individuals recover every day, hence $\gamma$ is the recovery rate.

    Therefore, we can write the model as

    $$
    S_{t+1} = S_t - \beta S_t \frac{I_t}{N} + \gamma I_t
    $$
    $$
    I_{t+1} = I_t + \beta S_t \frac{I_t}{N} - \gamma I_t
    $$

    Say that you start with a population of 1000 people, of which only 10 are infected (the rest are susceptible).
    That means your "initial state" is $S=990, I=10$, i.e. 990 are susceptible and 10 are infected.
    """)
    return


@app.cell
def _():
    ###
    SI0 = 990, 10

    β = 1.1
    γ = 0.5
    return SI0, β, γ


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Implement a function called `step_SIS(SI, β, γ)`** that given the current state `SI=(S, I)` and the parameters $\gamma$ and $\beta$, generates the next state.

    Note: you should make sure that $S$ and $I$ don't go below 0 or above $N$.
    """)
    return


@app.cell
def _(SI0, β, γ):
    def step_SIS(SI, β, γ): ###
        N =sum(SI)
        St= SI[0]
        It= SI[1]
        St1 = St - β * St * (It / N) + γ * It
        It1 = It + β * St * (It / N) - γ * It
        if St1 < 0: 
            St1, It1 = 0, It1
        if It1 < 0: 
            St1, It1 = St1, 0 
        if It1 > N: 
             St1, It1 = St1, N 
        if St1 > N: 
            St1, It1 = N, It1 
        return [St1, It1]

    step_SIS(SI0, β, γ) ###
    return (step_SIS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Implement a function called `simulation_SIS(SI0, β, γ, days)`** that given an initial state `SI0=(S0, I0)`, parameters $\gamma$ and $\beta$, and the number of days $days$ to run the simulation, simulates the dynamics and returns a vector `SI` in which the value at index `t, j` gives state `j` at day `t` (`j` being 0 for $S$ and 1 for $I$).

    Note that you should call `step_SIS` from `simulation_SIS`.

    Think: What is the type of the returned value? How many dimensions does it have?
    """)
    return


@app.cell
def _(np, step_SIS):
    def simulation_SIS(SI0, β, γ, days): ###
        _init_array = np.zeros((days, 2))
        _init_array[0] = SI0
        for t in range(1, days):
            _init_array[t] = np.array(step_SIS(_init_array[t-1], β, γ))
        return _init_array

    return (simulation_SIS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Run and plot** the dynamics for 90 days.
    """)
    return


@app.cell
def _(SI0, simulation_SIS, β, γ):
    sim_SIS = simulation_SIS(SI0, β, γ, days=90)
    return (sim_SIS,)


@app.cell
def _(plt, sim_SIS):
    plt.plot(sim_SIS[:, 0], color='blue', label='S')
    plt.plot(sim_SIS[:, 1], color='red',  label='I')
    plt.xlabel('Day')
    plt.ylabel('Count')
    plt.title('SIS model')
    plt.legend()
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For every $\beta, \gamma$ combination there is an expected equilibrium number of infected individuals $I^*$.

    **Plot $I^*$ as a function of $\beta$**.

    **Add a vertical line** for $\beta=\gamma$: epidemiological theory suggest that $R_0=\beta/\gamma$ is the reproductive number of an infectious disease. When $R_0<1$, the disease will die without infecting much of the population, whereas when $R_0>0$ the disease will become an epidemic, or even a pandemic, and will infect a significant fraction of the population.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    at equilibrium, the number of infected individuals does not change, meaning $I_{t+1} = I_t$, etc, $\Delta I = 0$.

    $$0 = \beta S^* \frac{I^*}{N} - \gamma I^*$$
    take $I^*$ out:
    $$0 = I^* \left( \frac{\beta S^*}{N} - \gamma \right)$$
    we got trivial solution is $I^* = 0$, and $\frac{\beta (S^*)}{N} = \gamma$, whereas $S^* = N - I^*$.

    then $\frac{\beta (N - I^*)}{N} = \gamma$, and solving for $I^*$:

    $$I^* = N\left(1 - \frac{\gamma}{\beta}\right)$$

    there is biological meaning to this only for $I^* > 0$, i.e when $\frac{\gamma}{\beta}\le 1  \to \gamma \le \beta$.
    """)
    return


@app.cell
def find_I_star(SI0, simulation_SIS):
    def find_I_star(β, γ, days): ###
        SI = simulation_SIS(SI0, β, γ, days)
        return SI[-1, 1]   

    return (find_I_star,)


@app.cell
def _(find_I_star, np, plt, γ):
    _days = 900
    betas = np.linspace(0.01, 2.0, _days)
    I_stars = [find_I_star(β, γ, days=_days) for β in betas]
    plt.plot(betas, I_stars, color='red', label='$I^*$')
    plt.axvline(x=γ, color='black', linestyle='--', label=f'$\\beta = \\gamma = {γ}$  ($R_0 = 1$)')
    plt.xlabel('$\\beta$')
    plt.ylabel('$I^*$')
    plt.title('Endemic equilibrium $I^*$ as a function of $\\beta$')
    plt.legend()
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SEIS model

    An possible extension is the SEIS model, in which suscptibles (S) become "exposed" (E) during contact with infected (I), exposed then become infected after an incubation time of $\Delta$ days, and then infected recover to become susceptible again.

    When $\Delta=0$, we get the SIS model again.

    **Implement this model and plot the dynamics for a range of $\Delta$ values.**

    **UPDATE**: note that previously the plot I had here was incorrect.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    number of new exposures at time $t$: $F_t = \beta S_t \frac{I_t}{N}$

    $S_{t+1}=S_t - F_t + \gamma I_t$

    $E_{t+1}=E_t + F_t - F_{t-\Delta}$

    $I_{t+1}=I_t + F_{t-\Delta} - \gamma I_t$

    $F_{t-\Delta}$: number of individuals that became exposed $\Delta$ days ago and are now becoming infected.
    sanity check: for  $\Delta=0$ we get $F_{t-\Delta}=F_t$, therefore:

    $E_{t+1}=E_t+F_t-F_t=E_t$.

    If $E_0=0$, then $E_t=0$ for all $t$, and the model reduces to

    $S_{t+1}=S_t - \beta S_t \frac{I_t}{N} + \gamma I_t$

    $I_{t+1}=I_t + \beta S_t \frac{I_t}{N} - \gamma I_t,$


    which is the SIS model.
    """)
    return


@app.cell
def _(np):
    def step_SEIS(SEI, β, γ, Δ, t=None): ###
        """
        since we have dependency in time, we should have the values in SEI for the last Δ days. 
        """
        if t is None:
            t = len(SEI)-1
        St, Et, It = SEI[t]
        N = St + Et + It
        Ft = β * St * It / N

        StD, EtD, ItD = SEI[t-Δ] if t - Δ >= 0 else (St, Et, It)

        if t >= Δ:
            StD, EtD, ItD = SEI[t-Δ]
            FtD = β * StD * ItD / N
        else:
            FtD = 0


        S_next = St - Ft + γ * It
        E_next = Et + Ft - FtD
        I_next = It + FtD - γ * It
        if len(SEI) > t+1:
            SEI[t+1] = [S_next, E_next, I_next]
        else: 
            SEI.append([S_next, E_next, I_next])
        return SEI

    def simulation_SEIS(SEI0, β, γ, Δ, days): ###
        SEI = np.zeros((days, 3))
        SEI[0] = SEI0
        for t in range(days - 1):
            SEI = step_SEIS(SEI, β, γ, Δ, t=t)
        return SEI

    return (simulation_SEIS,)


@app.cell
def _(plt, simulation_SEIS, β, γ):
    SEI0 = (990, 0, 10)  # S, E, I — initially no exposed
    deltas = [0, 2, 5, 10, 25]
    days = 90
    _fig, _axes = plt.subplots(len(deltas), 1, figsize=(8, 12), sharex=True)
    for _ax, Δ in zip(_axes, deltas):
        sim = simulation_SEIS(SEI0, β, γ, Δ, days)
        _ax.plot(sim[:, 0], color='blue', label='S')
        _ax.plot(sim[:, 1], color='green', label='E')
        _ax.plot(sim[:, 2], color='red', label='I')
        _ax.set_ylabel('Count')
        _ax.set_title(f'Δ = {Δ}')
        _ax.legend(loc='right')
    _axes[-1].set_xlabel('Day')
    _fig.suptitle(f'SEIS, β = {β}, γ = {γ}, days = {days}', y=1.01)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    as a a sanity check, we can see that in the case of $\Delta=0$ we get the sane result as the SIS model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ex 2: Logistic model

    The discrete-time logistc model is given by:

    $$
    N_{t+1} = r N_t \left(1 - N_t\right)
    $$

    This model is notoriously strange for some values of $r$.

    A similar model is the Ricker model, which have a somewhat nicer behaviour:

    $$
    N_{t+1} = N_t e^{r \left(1 - N_t\right)}
    $$

    **Implement both models and plot their dynamics for a set of $r$ values**.
    """)
    return


@app.cell
def _(jit, np):
    @jit ###
    def logistic(N0, r, tmax): ###
        N = np.zeros(tmax)
        N[0] = N0
        for t in range(tmax - 1):
            N[t+1] = r * N[t] * (1 - N[t])
        return N

    return (logistic,)


@app.cell
def _(jit, np):
    @jit ###
    def ricker(N0, r, tmax): ###
        N = np.zeros(tmax)
        N[0] = N0
        for t in range(tmax - 1):
            N[t+1] = N[t] * np.exp(r * (1 - N[t]))
        return N

    return (ricker,)


@app.cell
def _(logistic, plt, ricker):
    rs = [0.1, 1, 2, 3, 4] ###
    N0 = 0.1
    fig, axes = plt.subplots(len(rs), 2, figsize=(8, 8), sharex=True, sharey=False)
    for i, r in enumerate(rs):
        N = logistic(N0, r, 100)
        axes[i, 0].plot(N, '-')
        N = ricker(N0, r, 100)
        axes[i, 1].plot(N, '-')
        axes[i, 0].set_title('Logistic, r={}'.format(r))
        axes[i, 1].set_title('Ricker, r={}'.format(r))
        axes[i, 0].set_ylabel('N')
    axes[-1, 0].set_xlabel('t')
    axes[-1, 1].set_xlabel('t')
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A bifurcation plot shows how the equilbrlium values of a model change when one of its parameters change.

    The Bifurcation plot of the logistic model, which shows $N^*$ as a function of $r$, is very well known, so we will reproduce it here.
    For every value of $r$, it shows the values reached by the model after it ran for many steps.
    If the model reaches a stable equilibrium, there will be a single value; otherwise there could be several values if the model reaches a stable cycle, of very many if the model reaches an unstable cycle or becomes chaotic (!!!).

    **Plot a *bifurcation plot* for both models:**
    - choose a set of $r$ values
    - for each $r_i$ value, run the model for $n$ , to get $N_1, \ldots, N_n$
    - plot the last $m<n$ values as a function of $r_i$: $(r_i, N_{n-m}), \dots (r_i, N_{n})$.
    """)
    return


@app.cell
def bifurcation(np, plt):
    def bifurcation(model, npts=200, rs_d = 0.01, rs_u = 4.0, tmax=1000, lp=200): ###
        """"
        rs_d: lower bound of r values
        rs_u: upper bound of r values
        npts: number of r values to simulate
        tmax: total number of time steps to simulate for each r
        lp: number of last points to plot for each r (to show the attractor)
        """
        N0 = 0.1
        rs = np.linspace(rs_d, rs_u, npts)
    
        for r in rs:
            N = model(N0, r, tmax)
            plt.plot(
                [r] * lp,       # same r value repeated lp times
                N[-lp:],        # last lp values of N
                ',k',          # pixel marker, black
                alpha=0.1
            )
        plt.xlabel('r')
        plt.ylabel('$N^*$')

    return (bifurcation,)


@app.cell
def _(bifurcation, logistic, plt):
    bifurcation(logistic)
    plt.title('Logistic model')
    plt.gcf()
    return


@app.cell
def _(bifurcation, plt, ricker):
    bifurcation(ricker)
    plt.title('Ricker model')
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bonus: Interactive exploration of the logistic map

    The static plots above show only 5 values of $r$. The really interesting behavior — period-doubling and the route to chaos — happens at very specific values, so to *see* the transition it's much more illuminating to drag a slider through the range and watch the dynamics update.

    **Build an interactive plot using marimo widgets:**

    1. In the first cell below, create an `mo.ui.slider` for $r$ over the range $[0.1, 4.0]$ with a small step (e.g. `0.01`). Assign it to `r_ui` and display it.
    1. In the second cell, use `r_ui.value` to call `logistic(0.1, r_ui.value, 100)` and plot $N_t$ vs $t$.

    Drag the slider through the range. Around what value of $r$ does the fixed point become unstable? Where does a 2-cycle appear? When does chaos set in?

    See the slider examples in `notebooks/predator-prey.py`.
    """)
    return


@app.cell
def _():
    # your code here: use r_ui.value to run logistic(0.1, r_ui.value, 100) and plot
    return


@app.cell
def _(mo):
    r_ui = mo.ui.slider(0.1, 4.0, step=0.01, value=0.5, label='r')
    r_ui
    return (r_ui,)


@app.cell
def _(logistic, mo, plt, r_ui):
    _N = logistic(0.1, r_ui.value, 100)

    _fig, _ax = plt.subplots()
    _ax.plot(_N, '-o', markersize=3)
    _ax.set_xlabel('t')
    _ax.set_ylabel('$N_t$')
    _ax.set_title(f'Logistic model, r = {r_ui.value:.2f}')
    _ax.set_ylim(0, 1) 
    mo.mpl.interactive(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    __end of assignment__
    """)
    return


if __name__ == "__main__":
    app.run()
