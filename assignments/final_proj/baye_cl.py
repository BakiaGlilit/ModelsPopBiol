import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np 
    import pandas as pd 
    import scipy as sp
    import matplotlib.pyplot as plt
    import sympy as sym

    return mo, np, plt, sp, sym


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Recap
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Distributions
    ### Normal distribution
    A random variable $X$ is said to have a Normal (or Gaussian) distribution with mean $\mu$ and variance $\sigma^2$, written $X\sim N(\mu,\sigma^2)$, if its density is
    \[
        f(x)=\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}},\qquad -\infty<x<\infty.
    \]
    Its mean and variance are $\mathbb{E}[X]=\mu,\quad\operatorname{Var}(X)=\sigma^2$. When $\mu=0$ and $\sigma^2=1$, $Z\sim N(0,1)$ is called the standard Normal distribution. Linear combinations of independent Normal variables are also Normally distributed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _(mo):
    mu_norm = mo.ui.slider(-10.0, 10.0, step=0.1, value=0.0, label="μ")
    sigma_norm = mo.ui.slider(0.1, 10.0, step=0.1, value=1.0, label="σ")
    mo.hstack([mu_norm, sigma_norm])
    return mu_norm, sigma_norm


@app.cell
def _(mu_norm, np, plt, sigma_norm, sp):
    _x = np.linspace(mu_norm.value - 4*sigma_norm.value, mu_norm.value + 4*sigma_norm.value, 500)
    _fig, _ax = plt.subplots(figsize=(6, 3))
    _ax.plot(_x, sp.stats.norm.pdf(_x, mu_norm.value, sigma_norm.value), 'b-')
    _ax.set_title(f"Normal Distribution N({mu_norm.value:.1f}, {sigma_norm.value:.1f}^2)")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### chi-squared distribution with degree of freedome
    A random variable $X$ is said to have a chi-squared distribution with
    $\nu$ degrees of freedom, written $X\sim\chi^2_\nu$, if it can be expressed as
    \[
        X = Z_1^2+\cdots+Z_\nu^2,
    \]
    where $Z_1,\ldots,Z_\nu$ are independent standard Normal random variables, $Z_i\sim N(0,1)$. Its density is
    \[
        f(x)=\frac{1}{2^{\nu/2}\Gamma(\nu/2)}x^{\nu/2-1}e^{-x/2},\qquad x>0.
    \]
    Its mean and variance are $\mathbb{E}[X]=\nu,\quad\operatorname{Var}(X)=2\nu$.
    For a Normal sample,
    \[
        \frac{(n-1)S^2}{\sigma^2}\sim \chi^2_{n-1}.
    \]
    """)
    return


@app.cell
def _(mo):
    df_chi = mo.ui.slider(1, 30, step=1, value=5, label="ν (df)")
    df_chi
    return (df_chi,)


@app.cell
def _(df_chi, np, plt, sp):
    _x = np.linspace(0.01, sp.stats.chi2.ppf(0.999, df_chi.value), 500)
    _fig, _ax = plt.subplots(figsize=(6, 3))
    _ax.plot(_x, sp.stats.chi2.pdf(_x, df_chi.value), 'r-')
    _ax.set_title(f"Chi-Squared Distribution χ²({df_chi.value})")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### F distribution
    A random variable $F$ has an $F$ distribution with $\nu_1$ and $\nu_2$ degrees of freedom, written $F\sim F_{\nu_1,\nu_2}$, if
    \[
        F=\frac{U_1/\nu_1}{U_2/\nu_2},
    \]
    where $U_1\sim\chi^2_{\nu_1},\quad U_2\sim\chi^2_{\nu_2}$, and $U_1$ and $U_2$ are independent. Its density is
    \[
        f(x)=\frac{\Gamma\!\left((\nu_1+\nu_2)/2\right)}{\Gamma\!\left(\nu_1/2\right)\Gamma\!\left(\nu_2/2\right)}\left(\frac{\nu_1}{\nu_2}\right)^{\nu_1/2}\frac{x^{\nu_1/2-1}}{\left(1+\frac{\nu_1}{\nu_2}x\right)^{(\nu_1+\nu_2)/2}},\qquad x>0.
    \]
    It arises naturally in inference as the ratio of two independent sample variances, $\frac{S_1^2/\sigma_1^2}{S_2^2/\sigma_2^2}\sim F_{n_1-1,n_2-1}$.
    """)
    return


@app.cell
def _(mo):
    df1_f = mo.ui.slider(1, 50, step=1, value=5, label="ν1")
    df2_f = mo.ui.slider(1, 50, step=1, value=10, label="ν2")
    mo.vstack([df1_f, df2_f])
    return df1_f, df2_f


@app.cell
def _(df1_f, df2_f, np, plt, sp):
    _x_max = max(sp.stats.f.ppf(0.99, df1_f.value, df2_f.value), sp.stats.chi2.ppf(0.99, df1_f.value))
    _x = np.linspace(0.01, _x_max, 500)

    _fig, _ax = plt.subplots(figsize=(6, 3.5))
    _ax.plot(_x, sp.stats.f.pdf(_x, df1_f.value, df2_f.value), 'g-', lw=2, label=f"F({df1_f.value}, {df2_f.value})")
    _ax.plot(_x, sp.stats.chi2.pdf(_x, df1_f.value), 'r--', alpha=0.7, label=f"χ²({df1_f.value}) [U1]")
    _ax.plot(_x, sp.stats.chi2.pdf(_x, df2_f.value), 'b--', alpha=0.7, label=f"χ²({df2_f.value}) [U2]")

    _ax.set_title(f"F Distribution vs. Underlying Chi-Squared Distributions")
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Student's t distribution
    A random variable $T$ has a Student's $t$ distribution with $\nu$ degrees of freedom, written $T\sim t_\nu$,
    if
    \[
        T=\frac{Z}{\sqrt{U/\nu}},
    \]
    where $Z\sim N(0,1),\quad U\sim\chi^2_\nu$,
    and $Z$ and $U$ are independent. Its density is
    \[
        f(t)=\frac{\Gamma\!\left((\nu+1)/2\right)}{\sqrt{\nu\pi}\,\Gamma\!\left(\nu/2\right)}\left(1+\frac{t^2}{\nu}\right)^{- (\nu+1)/2},\qquad -\infty<t<\infty.
    \]
    The $t$ distribution is symmetric around zero but has heavier tails than the standard Normal distribution. As $\nu\to\infty$, $t_\nu \longrightarrow N(0,1)$.
    """)
    return


@app.cell
def _(mo):
    df_t = mo.ui.slider(1, 50, step=1, value=5, label="ν (df)")
    df_t
    return (df_t,)


@app.cell
def _(df_t, np, plt, sp):
    _x = np.linspace(-5, 5, 500)
    _fig, _ax = plt.subplots(figsize=(6, 3))
    _ax.plot(_x, sp.stats.t.pdf(_x, df_t.value), 'm-', label=f"t ({df_t.value})")
    _ax.plot(_x, sp.stats.norm.pdf(_x, 0, 1), 'k--', alpha=0.5, label="N(0,1)")
    _ax.set_title(f"Student's t Distribution t({df_t.value})")
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Bayesian reminder
    Let $\theta$ denote an unknown parameter and let $y=(y_1,\ldots,y_n)$ denote the observed sample. The prior distribution is $p(\theta)$, and the likelihood is $p(y\mid\theta)$. From Bayes' rule we get the posterior distribution
    \[
    p(\theta\mid y)=\frac{p(y\mid\theta)p(\theta)}{p(y)}\propto p(y\mid\theta)p(\theta).
    \]
    Thus, Bayesian learning may be viewed as posterior $\propto$ likelihood $\times$ prior:
    \[
        \text{prior belief} \quad\xrightarrow{\;\text{data / likelihood}\;}\quad \text{posterior belief}.
    \]

    For a model with several unknown parameters, $\theta=(\theta_1,\ldots,\theta_k)$, Bayes' rule gives a joint posterior
    $p(\theta_1,\ldots,\theta_k\mid y)$. If inference is required only for one parameter, the remaining parameters are treated as nuisance parameters and are marginalized out. In particular, for
    \[
        Y_i\mid\mu,\sigma^2 \sim N(\mu,\sigma^2),
    \]
    with both $\mu$ and $\sigma^2$ unknown, Bayesian inference begins with the joint likelihood $p(y\mid\mu,\sigma^2)$ and a joint prior $p(\mu,\sigma^2)$. The resulting joint posterior $p(\mu,\sigma^2\mid y)$ contains all information about the two parameters after observing the sample. Marginal inference for the mean is obtained by integrating out the variance,
    \[
        p(\mu\mid y) = \int_0^\infty p(\mu,\sigma^2\mid y)\,d\sigma^2,
    \]
    while inference for the variance is obtained by integrating out $\mu$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Introduction
    TODO
    * why assuming prior in plants and animal is reasonable
    * why choosing normal prior
    * limited number of samples
    *
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For simplification, we will start from bayesian learning of 2 distributions with shared variance, and then procceed to learning of 2 normal distributions with diffrent $\mu$ and $\sigma^2$ values, what is known as the Behrens Fisher problem.
    i will rely on chapter 17 in the book Introduction to Bayesian Statistics, Third Edition, by William M. Bolstad, James M. Curran (
    https://onlinelibrary.wiley.com/doi/book/10.1002/9781118593165). The full book is aviable in https://sci-hub.sidesgame.com/10.1002/9781118593165.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Two Normal populations with unknown means and a common unknown variance
    In this section, i will rely on chapter 17 in the book
    >Introduction to Bayesian Statistics, Third Edition, by William M. Bolstad, James M. Curran (
    https://onlinelibrary.wiley.com/doi/book/10.1002/9781118593165).

    The full book is aviable in https://sci-hub.sidesgame.com/10.1002/9781118593165.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Roadmap: from two patch samples to the difference in means
    Picture a root system sampling nutrient concentration from two soil patches. What matters for foraging is not $\mu_1$ or $\sigma^2$ on their own, but the *difference* $\mu_d=\mu_1-\mu_2$: is patch 1 richer than patch 2, and by how much? As in the Bayesian reminder above, $\sigma^2$ here is a nuisance parameter, common to both patches, that must be marginalized out before we can make a statement about $\mu_d$ alone.

    The rest of this section follows a fixed route toward that goal: (1) write the joint likelihood of $y_1,y_2$ given $\mu_1,\mu_2,\sigma^2$; (2) place conjugate priors on $\mu_1,\mu_2,\sigma^2$ and update them to a joint posterior; (3) read off the conditional posterior of $\mu_d$ given $\sigma^2$; and (4) marginalize $\sigma^2$ out using Theorem 17.1, arriving at a Student-$t$ posterior for $\mu_d$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Problem setting
    Suppose that we have two independent random samples $\quad y_1=(y_{11},\ldots,y_{1n_1}), \quad y_2=(y_{21},\ldots,y_{2n_2})$,
    where
    \[
        Y_{1i}\mid\mu_1,\sigma^2 \sim N(\mu_1,\sigma^2), \qquad Y_{2j}\mid\mu_2,\sigma^2 \sim N(\mu_2,\sigma^2).
    \]
    The two populations have different unknown means $\mu_1$ and $\mu_2$, but share the same unknown variance $\sigma^2$.
    We want to infer the parameters $\mu_1, \mu_2, \sigma^2$ based on the samples.

    Biologically, $y_{1i}$ and $y_{2j}$ are successive nutrient-concentration readings a root draws from patch 1 and patch 2 of a split-root system; the shared $\sigma^2$ encodes the simplifying assumption that the two patches are equally variable, differing only in mean richness.

    The samples are independent, hence their joint likelihood is the product of the two individual likelihoods:
    \[
    p(y_1,y_2\mid\mu_1,\mu_2,\sigma^2) = p(y_1\mid\mu_1,\sigma^2) p(y_2\mid\mu_2,\sigma^2).
    \]

    Define
    \[
    SS_1=\sum_{i=1}^{n_1}(y_{1i}-\bar y_1)^2, \qquad SS_2=\sum_{j=1}^{n_2}(y_{2j}-\bar y_2)^2,
    \]
    and the pooled sum of squares
    \[
        SS_p=SS_1+SS_2.
    \]

    Using
    \[
        \sum_{i=1}^{n_1}(y_{1i}-\mu_1)^2=SS_1+n_1(\bar y_1-\mu_1)^2,\qquad \sum_{j=1}^{n_2}(y_{2j}-\mu_2)^2=SS_2+n_2(\bar y_2-\mu_2)^2
    \]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Observation likelihood
    For one observation $y_{ji}$,
    \[
    f(y_{1i}\mid\mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left\{-\frac{(y_i-\mu)^2}{2\sigma^2}\right\}.
    \]
    Since the observations are independent, the likelihood is the product of their densities:
    \[
    L(\mu_j,\sigma^2) =\prod_{i=1}^{n_{j}} f(y_{ji}\mid\mu_1,\sigma^2) \propto (\sigma^2)^{-{n_{j}}/2} \exp\left\{ -\frac{1}{2\sigma^2} \sum_{i=1}^{n_{j}}(y_{ji}-\mu_{j})^2 \right\}.
    \]
    Hence, using the defenition of $SS_p$, the joint likelihood for the observations can be written as
    \[
    \begin{aligned}
    p(y_1,y_2\mid\mu_1,\mu_2,\sigma^2)&\propto
    \frac{1}{(\sigma^2)^{\frac{n_1}{2}}} \exp\left\{-\frac{n_1(\bar y_1-\mu_1)^2}{2\sigma^2}\right\}
    \times\frac{1}{(\sigma^2)^{\frac{n_2}{2}}}\exp\left\{-\frac{n_2(\bar y_2-\mu_2)^2}{2\sigma^2}\right\}\times\exp\left\{-\frac{SS_p}{2\sigma^2}\right\}.
    \end{aligned}
    \]
    Equivalently, separating the variance component,
    \[
    \begin{aligned}
    p(y_1,y_2\mid\mu_1,\mu_2,\sigma^2)&\propto\frac{1}{(\sigma^2)^{1/2}}\exp\left\{-\frac{n_1(\bar y_1-\mu_1)^2}{2\sigma^2}\right\}
    & \times \frac{1}{(\sigma^2)^{1/2}} \exp\left\{-\frac{n_2(\bar y_2-\mu_2)^2}{2\sigma^2}\right\}
    & \times\frac{1}{(\sigma^2)^{(n_1+n_2-2)/2}}\exp\left\{-\frac{SS_p}{2\sigma^2}\right\}.
    \end{aligned}
    \]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As step (2) of the roadmap, we first need a conjugate prior for $\sigma^2$ itself. Define $\sigma^2=\frac{S}{W}$.
    Recall that if $W\sim\chi^2_\kappa$, then
    \[
    f_W(w)= \frac{1}{2^{\kappa/2}\Gamma(\frac{\kappa}{2})} w^{\frac{\kappa}{2}-1}e^{-w/2}, \qquad w>0.
    \]
    We then say that $\quad \sigma^2\sim S\times\operatorname{Inv}\chi^2_\kappa,$
    whose density has kernel
    \[
    g(\sigma^2) \propto (\sigma^2)^{-\kappa/2-1} \exp\left\{-\frac{S}{2\sigma^2}\right\}.
    \]
    This has the same functional form in $\sigma^2$ as the normal likelihood. Consequently, multiplying the prior by the likelihood produces another scaled inverse chi-squared density:
    \[
    \underbrace{ (\sigma^2)^{-\kappa/2-1} e^{-S/(2\sigma^2)} }_{\text{prior}} \; \underbrace{ (\sigma^2)^{-n/2} e^{-SS/(2\sigma^2)} }_{\text{likelihood}} \propto \underbrace{ (\sigma^2)^{-(\kappa+n)/2-1} e^{-(S+SS)/(2\sigma^2)} }_{\text{posterior}}.
    \]
    The scaled inverse chi-squared distribution is a conjugate prior for the normal variance, with the simple updates $\kappa'=\kappa+n, \qquad S'=S+SS.$


    Thus, conditional on $\sigma^2$, the likelihood has the form of two independent Normal densities for $\mu_1$ and $\mu_2$, together with an inverse chi-squared component for $\sigma^2$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Theorem 17.1
    The last building block we need is a bridge from Normal and chi-squared variables to the Student-$t$ distribution — exactly what step (4) of the roadmap requires to marginalize $\sigma^2$ out of the posterior of $\mu_d$.

    **Theorem**: If $z$ and $w$ are independent random variables having the $\operatorname{normal}(0,1^2)$ distribution and the chi-squared distribution with $\kappa$ degrees of freedom respectively, then

    \[
    u = \frac{z}{\sqrt{\dfrac{w}{\kappa}}}
    \]

    will have the Student's $t$ distribution with $\kappa$ degrees of freedom.

    We will not prove it, but rather see a small simulation for it, to convince ourself.
    """)
    return


@app.cell
def _(mo):
    n_sim = mo.ui.slider(1000, 200000, step=1000, value=50000, label="N (repetitions)")
    nu_t_sim = mo.ui.slider(1, 30, step=1, value=5, label="ν (degrees of freedom)")
    mo.vstack([n_sim, nu_t_sim])
    return n_sim, nu_t_sim


@app.cell
def _(n_sim, np, nu_t_sim, plt, sp):
    _Z = np.random.normal(0, 1, n_sim.value)
    _U = np.random.chisquare(nu_t_sim.value, n_sim.value)
    _T = _Z / np.sqrt(_U / nu_t_sim.value)
    _x = np.linspace(-5, 5, 500)

    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.hist(_Z, bins=100, density=True, color="blue", alpha=0.3, label="Z ~ N(0,1)", histtype="step")
    _ax.hist(_U, bins=100, density=True, color="red", alpha=0.3, label=f"U ~ χ²({nu_t_sim.value})", histtype="step")
    _ax.hist(_T, bins=100, density=True, color="green", alpha=0.5, label="Simulated T", range=(-5, 5))
    _ax.plot(_x, sp.stats.t.pdf(_x, df=nu_t_sim.value), "k-", linewidth=2, label=f"Theoretical t({nu_t_sim.value})")
    _ax.set_xlim(-5, 15)
    _ax.legend()
    _ax.set_title(f"Student's t Construction via Simulation, N={n_sim.value}, nu={nu_t_sim.value} ")
    _fig
    return


@app.cell
def _(mo):
    n_bayes_s2 = mo.ui.slider(5, 200, step=5, value=25, label="Sample size (n)")
    kappa_bayes_s2 = mo.ui.slider(1, 20, step=1, value=2, label="Prior degrees of freedom (κ)")
    S_bayes_s2 = mo.ui.slider(1.0, 100.0, step=1.0, value=20.0, label="Prior scale (S)")
    K_bayes_s2 = mo.ui.slider(5, 50, step=5, value=20, label="Draws (K)")
    sigma2_true_bayes = mo.ui.slider(0.5, 10.0, step=0.5, value=4.0, label="True σ²")

    mo.hstack([
        mo.hstack([n_bayes_s2, sigma2_true_bayes, K_bayes_s2]),
        mo.hstack([kappa_bayes_s2, S_bayes_s2])
    ])
    return (
        K_bayes_s2,
        S_bayes_s2,
        kappa_bayes_s2,
        n_bayes_s2,
        sigma2_true_bayes,
    )


@app.cell
def _(
    K_bayes_s2,
    S_bayes_s2,
    kappa_bayes_s2,
    n_bayes_s2,
    np,
    plt,
    sigma2_true_bayes,
    sp,
):
    _mu = 0.0
    _sigma2_true = sigma2_true_bayes.value
    _n = n_bayes_s2.value
    _kappa = kappa_bayes_s2.value
    _S = S_bayes_s2.value
    _K = K_bayes_s2.value

    _rng = np.random.default_rng(7)
    _y = _rng.normal(loc=_mu, scale=np.sqrt(_sigma2_true), size=_n)
    _SS = np.sum((_y - _mu) ** 2)

    _kappa_post = _kappa + _n
    _S_post = _S + _SS

    _prior_W = _rng.chisquare(df=_kappa, size=_K)
    _prior_sigma2 = _S / _prior_W

    _post_W = _rng.chisquare(df=_kappa_post, size=_K)
    _post_sigma2 = _S_post / _post_W

    _sigma2_prior_est = _S / _kappa
    _sigma2_post_est = _S_post / _kappa_post

    _x = np.linspace(-8, 8, 1000)
    _true_pdf = sp.stats.norm.pdf(_x, loc=_mu, scale=np.sqrt(_sigma2_true))
    _prior_pdf = sp.stats.norm.pdf(_x, loc=_mu, scale=np.sqrt(_sigma2_prior_est))
    _post_pdf = sp.stats.norm.pdf(_x, loc=_mu, scale=np.sqrt(_sigma2_post_est))

    _fig, _ax = plt.subplots(figsize=(9, 5))

    for _s2 in _prior_sigma2:
        _pdf = sp.stats.norm.pdf(_x, loc=_mu, scale=np.sqrt(_s2))
        _ax.plot(_x, _pdf, color="C0", alpha=0.08)

    for _s2 in _post_sigma2:
        _pdf = sp.stats.norm.pdf(_x, loc=_mu, scale=np.sqrt(_s2))
        _ax.plot(_x, _pdf, color="C2", alpha=0.18)

    _ax.plot(_x, _prior_pdf, color="C0", linestyle=":", linewidth=3, label=rf"Prior: $\sigma^2=S/\kappa={_sigma2_prior_est:.2f}$")
    _ax.plot(_x, _true_pdf, color="C1", linewidth=3, label=rf"True population: $N({_mu}, {_sigma2_true})$, $\sigma^2={_sigma2_true}$")
    _ax.plot(_x, _post_pdf, color="C2", linestyle="--", linewidth=3, label=rf"Posterior: $\hat{{\sigma}}_B^2={_sigma2_post_est:.2f}$")
    _ax.scatter(_y, np.zeros_like(_y), color="black", marker="|", s=180, label=rf"Observed sample ($n={_n}$)")

    _ax.set_xlabel(r"$y$")
    _ax.set_ylabel("Density")
    _ax.set_title(rf"Bayesian learning of $\sigma^2$ ($n={_n}$, $K={_K}$ prior/posterior draws)")
    _ax.legend()
    _ax.grid(alpha=0.25)

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Finding the Approximate Posterior when the Joint Conjugate Prior Is Used for All Parameters

    The joint posterior is proportional to the joint prior multiplied by the joint
    likelihood:
    \[
    g_{\mu_1,\mu_2,\sigma^2}
    (\mu_1,\mu_2,\sigma^2\mid \mathbf y_1,\mathbf y_2)
    \propto
    g_{\mu_1,\mu_2,\sigma^2}(\mu_1,\mu_2,\sigma^2)
    \,f(\mathbf y_1,\mathbf y_2\mid\mu_1,\mu_2,\sigma^2).
    \]

    Using the conditional normal priors for $\mu_1$ and $\mu_2$, the inverse
    chi-squared prior for $\sigma^2$, and the corresponding likelihood factors,
    we obtain
    \[
    \begin{align}
    g_{\mu_1,\mu_2,\sigma^2}(\mu_1,\mu_2,\sigma^2\mid \mathbf y_1,\mathbf y_2)&\propto\frac{1}{(\sigma^2)^{1/2}}\exp\left\{-\frac{n_{10}}{2\sigma^2}(\mu_1-m_1)^2\right\}
    \nonumber\\&\quad\times\frac{1}{(\sigma^2)^{1/2}}\exp\left\{-\frac{n_{20}}{2\sigma^2}(\mu_2-m_2)^2\right\}
    \nonumber\\&\quad\times\frac{1}{(\sigma^2)^{\frac{\kappa}{2}+1}}\exp\left\{-\frac{S}{2\sigma^2}\right\}
    \nonumber\\&\quad\times\frac{1}{(\sigma^2)^{1/2}}\exp\left\{-\frac{n_1}{2\sigma^2}(\bar y_1-\mu_1)^2\right\}
    \nonumber\\&\quad\times\frac{1}{(\sigma^2)^{1/2}}\exp\left\{-\frac{n_2}{2\sigma^2}(\bar y_2-\mu_2)^2\right\}
    \nonumber\\&\quad\times\frac{1}{(\sigma^2)^{\frac{(n_1+n_2)}{2}-1}}\exp\left\{-\frac{SS_p}{2\sigma^2}\right\}
    \end{align}
    \]
    """)
    return


@app.cell
def _(sym):
    mu1_sym, mu2_sym = sym.symbols("mu_1 mu_2")
    m1_sym, m2_sym = sym.symbols("m_1 m_2")
    ybar1_sym, ybar2_sym = sym.symbols(r"\bar{y}_1 \bar{y}_2")
    n1_sym, n2_sym = sym.symbols("n_1 n_2", positive=True)
    n10_sym, n20_sym = sym.symbols("n_{10} n_{20}", positive=True)
    kappa_sym, S_sym, SSp_sym = sym.symbols(r"\kappa S SS_p", positive=True)

    mu1_part_sym = n10_sym * (mu1_sym - m1_sym)**2 + n1_sym * (mu1_sym - ybar1_sym)**2
    n1_post_sym = n1_sym + n10_sym
    m1_post_sym = (n1_sym * ybar1_sym + n10_sym * m1_sym) / n1_post_sym
    mu1_remainder_sym = n1_sym * n10_sym / n1_post_sym * (ybar1_sym - m1_sym)**2
    mu1_completed_sym = n1_post_sym * (mu1_sym - m1_post_sym)**2 + mu1_remainder_sym
    mu1_check_sym = sym.simplify(mu1_part_sym - mu1_completed_sym)

    mu2_part_sym = n20_sym * (mu2_sym - m2_sym)**2 + n2_sym * (mu2_sym - ybar2_sym)**2
    n2_post_sym = n2_sym + n20_sym
    m2_post_sym = (n2_sym * ybar2_sym + n20_sym * m2_sym) / n2_post_sym
    mu2_remainder_sym = n2_sym * n20_sym / n2_post_sym * (ybar2_sym - m2_sym)**2
    mu2_completed_sym = n2_post_sym * (mu2_sym - m2_post_sym)**2 + mu2_remainder_sym
    mu2_check_sym = sym.simplify(mu2_part_sym - mu2_completed_sym)

    kappa_post_sym = kappa_sym + n1_sym + n2_sym - 2
    S_post_sym = S_sym + SSp_sym
    return (
        S_post_sym,
        kappa_post_sym,
        m1_post_sym,
        m2_post_sym,
        mu1_check_sym,
        mu1_completed_sym,
        mu1_part_sym,
        mu1_remainder_sym,
        mu2_check_sym,
        mu2_completed_sym,
        mu2_part_sym,
        mu2_remainder_sym,
        n1_post_sym,
        n2_post_sym,
    )


@app.cell(hide_code=True)
def _(
    S_post_sym,
    kappa_post_sym,
    m1_post_sym,
    m2_post_sym,
    mo,
    mu1_check_sym,
    mu1_completed_sym,
    mu1_part_sym,
    mu1_remainder_sym,
    mu2_check_sym,
    mu2_completed_sym,
    mu2_part_sym,
    mu2_remainder_sym,
    n1_post_sym,
    n2_post_sym,
    sym,
):
    mo.md(f"""
    ### Conjugate posterior update
    Completing the square in $\mu_1$,
    \[
    {sym.latex(mu1_part_sym)}={sym.latex(mu1_completed_sym)}
    \]
    with posterior parameters
    \[
    n_1' = {sym.latex(n1_post_sym)}, \qquad m_1' = {sym.latex(m1_post_sym)},
    \]
    and a remainder that does not depend on $\mu_1$ (SymPy confirms the identity: difference $={sym.latex(mu1_check_sym)}$).

    By the identical argument for $\mu_2$,
    \[
    n_2' = {sym.latex(n2_post_sym)}, \qquad m_2' = {sym.latex(m2_post_sym)}.
    \]

    The variance factor matches a scaled inverse chi-squared kernel directly, giving $\kappa' = {sym.latex(kappa_post_sym)}$ and $S' = {sym.latex(S_post_sym)}$. Altogether, the conjugate posterior is
    \[
    \mu_1 \mid \sigma^2, y \sim N(m_1', \sigma^2/n_1'), \qquad \mu_2 \mid \sigma^2, y \sim N(m_2', \sigma^2/n_2'), \qquad \sigma^2 \mid y \sim S' \cdot \mathrm{{Inv}}\chi^2_{{\kappa'}}.
    \]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So, overall, after we did the derivation using sympy, the terms are grouped into three conjugate pairs, of the prior and likelihood terms for $\mu_1, \mu_2, \sigma^2$.

    Combining each pair gives the approximate posterior
    \[
    \begin{align}
    g_{\mu_1,\mu_2,\sigma^2}(\mu_1,\mu_2,\sigma^2\mid \mathbf y_1,\mathbf y_2)
    &\propto\frac{1}{(\sigma^2)^{1/2}}\exp\left\{-\frac{n_1'}{2\sigma^2}(\mu_1-m_1')^2\right\}
    \nonumber\\&\quad\times\frac{1}{(\sigma^2)^{1/2}}\exp\left\{-\frac{n_2'}{2\sigma^2}(\mu_2-m_2')^2\right\}
    \nonumber\\&\quad\times\frac{1}{(\sigma^2)^{\frac{\kappa'}{2}+1}}\exp\left\{-\frac{S'}{2\sigma^2}\right\}.
    \end{align}
    \]

    With the updated constants
    \[
    \kappa'=\kappa+n_1+n_2-2, \qquad
    n_1'=n_1+n_{10}, \qquad n_2'=n_2+n_{20}, \qquad S'=S+SS_p \qquad
    m_1'=\frac{n_1\bar y_1+n_{10}m_1}{n_1+n_{10}}, \qquad m_2'=\frac{n_2\bar y_2+n_{20}m_2}{n_2+n_{20}}
    \]
    """)
    return


@app.cell
def _(m1_post_sym, m2_post_sym, n1_post_sym, n2_post_sym, sym):
    mud_sym = sym.symbols(r"\mu_d")
    sigma2_sym = sym.symbols(r"\sigma^2", positive=True)
    md_post_sym = sym.simplify(m1_post_sym - m2_post_sym)
    vard_post_sym = sym.simplify(sigma2_sym / n1_post_sym + sigma2_sym / n2_post_sym)
    vard_post_factored_sym = sym.factor(vard_post_sym)
    precision_d_sym = sym.simplify(1 / (1 / n1_post_sym+ 1 / n2_post_sym))
    precision_d_factored_sym = sym.factor(precision_d_sym)
    return (
        md_post_sym,
        mud_sym,
        precision_d_factored_sym,
        sigma2_sym,
        vard_post_factored_sym,
    )


@app.cell(hide_code=True)
def _(md_post_sym, mo, precision_d_factored_sym, sym, vard_post_factored_sym):
    mo.md(rf"""
    ### Posterior of $\mu_d = \mu_1 - \mu_2$ 

    Define $\mu_d = \mu_1 - \mu_2.$
    Since $\mu_1$ and $\mu_2$ are conditionally independent Normal random variables, their difference is also Normal.
    The posterior mean is
    \[
    m_d'=m_1' - m_2'={sym.latex(md_post_sym)}.
    \]
    The posterior variance is
    \[
    \operatorname{{Var}}(\mu_d \mid \sigma^2,y)={sym.latex(vard_post_factored_sym)}.
    \]
    The inverse variance factor is
    \[
    \frac{{1}}{{\frac{{1}}{{n_1'}}+\frac{{1}}{{n_2'}}}}={sym.latex(precision_d_factored_sym)}.
    \]
    Therefore,
    \[
    \mu_d \mid \sigma^2,y \sim N\left( m_d', \sigma^2 \left[ \frac{{1}}{{n_1'}}+\frac{{1}}{{n_2'}} \right] \right).
    \]
    """)
    return


@app.cell
def _(mud_sym, sigma2_sym, sym):
    md_post_display_sym = sym.symbols("m_d'")
    n1_post_display_sym, n2_post_display_sym = sym.symbols("n_1' n_2'", positive=True)
    kappa_post_display_sym = sym.symbols(r"\kappa'", positive=True)
    S_post_display_sym = sym.symbols("S'", positive=True)
    precision_d_display_sym = sym.simplify(1 / (1 / n1_post_display_sym+1 / n2_post_display_sym))
    normal_d_kernel_sym = (sigma2_sym ** (-sym.Rational(1, 2))* sym.exp(-precision_d_display_sym* (mud_sym - md_post_display_sym) ** 2/ (2 * sigma2_sym)))
    sigma2_kernel_sym = (sigma2_sym ** (-kappa_post_display_sym / 2 - 1)* sym.exp( -S_post_display_sym / (2 * sigma2_sym)))
    joint_power_sym = sym.simplify(-sym.Rational(1, 2)- kappa_post_display_sym / 2- 1)
    joint_exponent_terms_sym = (precision_d_display_sym* (mud_sym - md_post_display_sym) ** 2+ S_post_display_sym)
    return (
        S_post_display_sym,
        joint_exponent_terms_sym,
        joint_power_sym,
        kappa_post_display_sym,
        md_post_display_sym,
        n1_post_display_sym,
        n2_post_display_sym,
        precision_d_display_sym,
    )


@app.cell(hide_code=True)
def _(
    joint_exponent_terms_sym,
    joint_power_sym,
    mo,
    precision_d_display_sym,
    sym,
):
    mo.md(rf"""
    ### Joint posterior of \(\mu_d\) and \(\sigma^2\)

    Earlier, we found the parameter for the distribution of $\mu_d = \mu_1 - \mu_2$, and found that its precision is ${sym.latex(precision_d_display_sym)} \frac{{1}}{{\sigma^2}}$.  Hence:
    \[
    p(\mu_d \mid \sigma^2,y)\propto\frac{{1}}{{(\sigma^2)^{{1/2}}}}\exp\left\{{-\frac{{1}}{{2\sigma^2}}\left({sym.latex(precision_d_display_sym)}\right)(\mu_d-m_d')^2\right\}}.
    \]

    In addition, we found

    \[
    p(\sigma^2 \mid y)\propto\frac{{1}}{{(\sigma^2)^{{\kappa'/2+1}}}}\exp\left\{{-\frac{{S'}}{{2\sigma^2}}\right\}}.
    \]

    Using the product rule for conditional distributions,

    \[
    p(\mu_d,\sigma^2 \mid y)=p(\mu_d \mid \sigma^2,y)\,p(\sigma^2 \mid y).
    \]

    Thus,

    \[
    p(\mu_d,\sigma^2 \mid y) \propto \frac{{1}}{{(\sigma^2)^{{1/2}}}} \exp\left\{{ -\frac{{1}}{{2\sigma^2}} \left( \frac{{n_1'n_2'}}{{n_1'+n_2'}} \right) (\mu_d-m_d')^2 \right\}} \frac{{1}}{{(\sigma^2)^{{\kappa'/2+1}}}} \exp\left\{{ -\frac{{S'}}{{2\sigma^2}} \right\}}.
    \]

    Combining only the powers of $\sigma^2$: $-\frac{{1}}{{2}}-\frac{{\kappa'}}{{2}}-1={sym.latex(joint_power_sym)}$,
    and summing the exponent values: 
    \[
    {sym.latex(joint_exponent_terms_sym)}.
    \]

    So the joint posterior is: 

    \[
    {{p(\mu_d,\sigma^2 \mid y)\propto\frac{{1}}{{(\sigma^2)^{{(\kappa'+1)/2+1}}}}\exp\left\{{-\frac{{1}}{{2\sigma^2}}\left[\frac{{n_1'n_2'}}{{n_1'+n_2'}}(\mu_d-m_d')^2+S'\right]\right\}}}}
    \]
    """)
    return


@app.cell
def _(
    S_post_display_sym,
    kappa_post_display_sym,
    md_post_display_sym,
    mud_sym,
    n1_post_display_sym,
    n2_post_display_sym,
    sigma2_sym,
    sym,
):
    # Variance factor of mu_d conditional on sigma^2
    variance_factor_d_sym = sym.simplify(1 / n1_post_display_sym+ 1 / n2_post_display_sym)
    # Z ~ N(0,1)
    z_d_sym = sym.simplify((mud_sym - md_post_display_sym)/ sym.sqrt( sigma2_sym * variance_factor_d_sym))
    # From sigma^2 ~ S' * Inv-chi^2_kappa', W = S' / sigma^2 ~ chi^2_kappa'
    w_d_sym = sym.simplify(S_post_display_sym / sigma2_sym)
    # Theorem 17.1: t = Z / sqrt(W / kappa')
    t_from_theorem_sym = sym.simplify(z_d_sym/ sym.sqrt( w_d_sym / kappa_post_display_sym))
    # Bayesian variance estimate
    sigmaB2_display_sym = sym.simplify(S_post_display_sym/ kappa_post_display_sym)
    # Desired final form
    t_final_sym = sym.simplify((mud_sym - md_post_display_sym)/ sym.sqrt(sigmaB2_display_sym* variance_factor_d_sym))
    # Verify that the theorem gives the desired expression
    t_check_sym = sym.simplify(t_from_theorem_sym - t_final_sym)
    return t_check_sym, t_from_theorem_sym, w_d_sym, z_d_sym


@app.cell
def _(mo, sym, t_check_sym, t_from_theorem_sym, w_d_sym, z_d_sym):
    mo.md(
        rf"""
    ### Marginal posterior of $\mu_d$ using Theorem 17.1

    We have already obtained the conditional posterior

    \[
    \mu_d\mid\sigma^2,y \sim N\left( m_d', \sigma^2 \left[ \frac{{1}}{{n_1'}}+\frac{{1}}{{n_2'}} \right] \right).
    \]

    To apply Theorem 17.1, first standardize this Normal random variable. Define

    \[
    Z = \frac{{ \mu_d-m_d' }}{{ \sqrt{{ \sigma^2\left(\frac{{1}}{{n_1'}}+\frac{{1}}{{n_2'}}\right)}}}}.
    \]

    Using the quantities developed by SymPy,
    \[
    Z = {sym.latex(z_d_sym)},
    \]

    and therefore $Z\sim N(0,1)$. From the posterior distribution of the common variance,

    \[
    \sigma^2\mid y \sim S'\,\mathrm{{Inv}}\text{{-}}\chi^2_{{\kappa'}},
    \]

    the definition of the scaled inverse chi-squared distribution gives $W=\frac{{S'}}{{\sigma^2}}\sim \chi^2_{{\kappa'}}$, and SymPy represents this transformation as $W={sym.latex(w_d_sym)}$.

    Thus we have the two random variables required by Theorem 17.1:

    \[
    Z\sim N(0,1), \qquad W\sim\chi^2_{{\kappa'}}.
    \]

    By Theorem 17.1,

    \[
    t=\frac{{Z}}{{\sqrt{{W/\kappa'}}}}\sim t_{{\kappa'}}.
    \]

    Substituting the expressions for $\mu_d$ and $W$,

    \[
    t = \frac{{ \displaystyle \frac{{ \mu_d-m_d' }}{{ \sqrt{{ \sigma^2 \left( \frac{{1}}{{n_1'}}+\frac{{1}}{{n_2'}} \right) }} }} }}{{\displaystyle \sqrt{{ \frac{{S'}}{{\kappa'\sigma^2}}}}}}.
    \]

    and SymPy simplifies this expression to

    \[
    t={sym.latex(t_from_theorem_sym)}.
    \]

    The factor $\sigma^2$ cancels. Since $\hat{{\sigma}}_B^2=\frac{{S'}}{{\kappa'}}$, the result can be written as
    \[
    \boxed{{t=\frac{{\mu_d-m_d'}}{{\hat{{\sigma}}_B\sqrt{{\frac{{1}}{{n_1'}}+\frac{{1}}{{n_2'}}}}}}}}
    \]

    and therefore $t\sim t_{{\kappa'}}$.
    Finally, SymPy verifies that the expression obtained directly from Theorem 17.1 and the final standardized expression are identical:

    \[
    {sym.latex(t_check_sym)}=0.
    \]

    Thus, although the conditional posterior of $\mu_d$ given $\sigma^2$ is Normal, accounting for the uncertainty in the unknown common variance produces a Student-$t$ marginal posterior for the difference of means.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A shared variance leaves no room for risk
    Under a common $\sigma^2$, $\mu_d$ is the only feature that distinguishes the two patches: the posterior derived above never involves the variance ratio $\rho=\sigma_1^2/\sigma_2^2$, because $\rho=1$ was imposed from the very start. A root equipped with only this model has no basis for treating a "high-variance" patch differently from a "low-variance" one — variability itself carries no information, and allocation could only ever track the difference in mean nutrient concentration.

    Yet risk-sensitive foraging is defined precisely by sensitivity to variance: the Energy Budget Rule predicts a preference for the high-variance patch when below an energetic requirement, and for the low-variance patch when above it, a pattern observed experimentally in plants (Dener et al. 2016). For that preference to emerge as a *Bayesian* consequence rather than a hard-wired rule, the two patches must be allowed genuinely different variances $\sigma_1^2\neq\sigma_2^2$. Giving up the shared-variance assumption is therefore not a mathematical nicety but the minimal change needed to let variance itself become information — and it is exactly this relaxation, together with its extra nuisance parameter $\rho$, that defines the Behrens–Fisher problem taken up next.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Behrens-Fisher problem
    In the biological setting, the diffrent patches do not neccerly share variance. Moreover, when checking for risk sensitivity (Dener et al 2016), the goal is to check the allocation to  high variance patch vs low variance patch under diffrent mean nutrient values.
    Hence, we would be intersting in learning 2 normal distributions with unknown means and variances $N(\mu_1,\sigma_1^2), N(\mu_2,\sigma_2^2)$.
    We will based our approach for  bayesian inference of this setting on the paper
    > MORTON B. BROWN, The two-means problem—a secondarily Bayes approach, Biometrika, Volume 54, Issue 1-2, June 1967, Pages 85–91, https://doi.org/10.1093/biomet/54.1-2.85

    and have a prior assumptions about the relative variability
    \[
    \rho=\frac{\sigma_1^2}{\sigma_2^2}
    \]

    The assumption of relative sensing of the variability is reasonable in animals, in which webber law (TODO) have been confirmed experimently. There is yet no such experiments in plants, so it might be less accurate for our root model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Model and primary parameter

    Let $y_1=(y_{11},\ldots,y_{1n_1}), \qquad y_2=(y_{21},\ldots,y_{2n_2})$ be two independent samples. The observations are modeled as

    \[
    Y_{1j}\mid\mu_1,\sigma_1^2\overset{\mathrm{iid}}{\sim}N(\mu_1,\sigma_1^2),\qquad j=1,\ldots,n_1, \qquad Y_{2j}\mid\mu_2,\sigma_2^2\overset{\mathrm{iid}}{\sim}N(\mu_2,\sigma_2^2),\qquad j=1,\ldots,n_2.
    \]
    The two populations have unknown means $\mu_1$ and $\mu_2$, and unknown, potentially unequal variances $\sigma_1^2$ and $\sigma_2^2$. Define the sample means and variances:

    \[
    \bar y_1=\frac{1}{n_1}\sum_{j=1}^{n_1}y_{1j},\qquad\bar y_2=\frac{1}{n_2}\sum_{j=1}^{n_2}y_{2j},\qquad s_1^2=\frac{1}{f_1}\sum_{j=1}^{n_1}(y_{1j}-\bar y_1)^2,\qquad  s_2^2=\frac{1}{f_2}\sum_{j=1}^{n_2}(y_{2j}-\bar y_2)^2,
    \]
    where $f_1=n_1-1, \qquad f_2=n_2-1$ are the degrees of freedom of the two samples. The primary parameter is the difference between the population means $\delta=\mu_1-\mu_2$.  The variances are secondary parameters: they determine the uncertainty in inference about $\delta$, but are not themselves the primary quantity of interest.
    """)
    return


@app.cell
def _(sym):
    bf_mu1_sym, bf_mu2_sym = sym.symbols(r"\mu_1 \mu_2",real=True,)
    bf_sigma1_2_sym, bf_sigma2_2_sym = sym.symbols(r"\sigma_1^2 \sigma_2^2", positive=True,)
    bf_n1_sym, bf_n2_sym = sym.symbols(r"n_1 n_2",positive=True,integer=True,)
    bf_f1_sym, bf_f2_sym = bf_n1_sym - 1, bf_n2_sym - 1
    bf_ybar1_sym, bf_ybar2_sym = sym.symbols(r"\bar{y}_1 \bar{y}_2",real=True,)
    bf_s1_2_sym, bf_s2_2_sym = sym.symbols(r"s_1^2 s_2^2",positive=True,)
    bf_delta_sym = sym.simplify(bf_mu1_sym - bf_mu2_sym)
    return (
        bf_delta_sym,
        bf_f1_sym,
        bf_f2_sym,
        bf_n1_sym,
        bf_n2_sym,
        bf_s1_2_sym,
        bf_s2_2_sym,
        bf_sigma1_2_sym,
        bf_sigma2_2_sym,
        bf_ybar1_sym,
        bf_ybar2_sym,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Observed and population variance shares

    The uncertainty in the observed difference of sample means, $\bar y_1-\bar y_2,$
    is determined by the two estimated sampling variances, $\frac{s_1^2}{n_1}$ and $\frac{s_2^2}{n_2}.$

    Define the observed share of this uncertainty contributed by the first sample as

    \[
    c=\frac{s_1^2/n_1}{s_1^2/n_1+s_2^2/n_2}.
    \]

    Since $s_1^2,s_2^2>0$, $0<c<1$, the corresponding population quantity is

    \[
    \gamma=\frac{\sigma_1^2/n_1}{\sigma_1^2/n_1+\sigma_2^2/n_2}.
    \]
    The parameter $\gamma$ is a in terms of the population variance ratio $\rho=\frac{\sigma_1^2}{\sigma_2^2}$:
    \[
    \gamma(\rho)=\frac{\rho n_2}{\rho n_2+n_1}
    \]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ploting the relation between the matmatical varaibles
    """)
    return


@app.cell
def _(np, plt):
    _rho = np.linspace(0.1, 20.0, 500)  # Variance Ratio: sigma1^2 / sigma2^2
    # Under equal sampling, gamma = rho / (rho + 1)
    _gamma = _rho / (_rho + 1.0)
    # Highlight specific biological variance ratios (e.g., Equal, 2x, 5x, 0.2x)
    _key_ratios = [0.2, 0.5, 1.0, 2.0, 5.0]
    _key_shares = [_r / (_r + 1.0) for _r in _key_ratios]

    plt.figure(figsize=(8, 4.5))
    plt.plot(
        _rho,
        _gamma,
        color="navy",
        lw=2.5,
        label=r"Equal Sampling Share: $\gamma = \frac{\rho}{\rho + 1}$",
    )

    # Annotate key biological contrast points
    for _r, _g in zip(_key_ratios, _key_shares):
        plt.scatter(_r, _g, color="crimson", zorder=5)
        # Fixed string formatting with raw string 'r' and clean LaTeX syntax
        _label_text = rf"$\rho = {_r:.1f}, \quad \gamma = {_g:.2f}$"
        plt.annotate(
            _label_text, 
            (_r, _g),
            textcoords="offset points",
            xytext=(5, -10),
            fontsize=9,
        )
    plt.text(
        8.5,
        0.45,
        r"$\gamma = \frac{\rho}{\rho + 1}$",
        fontsize=12,
        color="k",
        bbox=dict(
            facecolor="aliceblue",
            edgecolor="navy",
            alpha=0.8,
        ),)
    plt.title(
        r"Population Variance Share ($\gamma$) as a Function of Patch Variance Contrast ($\rho$)"
    )
    plt.xlabel(
        r"Patch Variance Ratio $\rho = \sigma_1^2 / \sigma_2^2$ ($\sigma_H^2 / \sigma_L^2$)"
    )
    plt.ylabel(r"Population Variance Share $\gamma$")
    plt.xlim(0, 20)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.show()
    return


@app.cell
def _(np, plt):
    # Equal sample sizes across patches (n1 = n2 = n)
    # Log-spaced range for rho spanning from 0.05 to 20
    _rho = np.logspace(-1.3, 1.3, 500)

    # Hyperbolic share: gamma = rho / (rho + 1)
    _gamma = _rho / (_rho + 1.0)

    plt.figure(figsize=(8.5, 4.8))

    # 1. Sigmoid S-curve on log scale
    plt.plot(
        _rho,
        _gamma,
        color="k",
        lw=2.5,
        label=r"True Share: $\gamma = \frac{\rho}{\rho + 1}$",
    )

    # 2. Point of Equality (rho = 1.0, gamma = 0.5)
    _rho_eq, _gamma_eq = 1.0, 0.5
    plt.scatter(_rho_eq, _gamma_eq, color="crimson", s=70, zorder=5)

    plt.annotate(
        r"Point of Equality" + "\n" + r"$(\rho = 1.0, \gamma = 0.5)$",
        (_rho_eq, _gamma_eq),
        textcoords="offset points",
        xytext=(15, -15),
        fontsize=9,
        color="crimson",
        weight="bold",
    )

    # 3. Asymptotic Bounds at gamma = 0 and gamma = 1
    plt.axhline(1.0, color="gray", linestyle=":", alpha=0.6)
    plt.axhline(0.0, color="gray", linestyle=":", alpha=0.6)

    # 4. Mathematical Formula Box
    plt.text(
        0.12,
        0.80,
        r"$\gamma = \frac{\rho}{\rho + 1}$",
        fontsize=12,
        color="k",
        bbox=dict(
            facecolor="white",
            edgecolor="k",
            alpha=0.8,
        ),)

    plt.xscale("log")  # Set X-axis to Logarithmic
    plt.title(
        r"Symmetric Saturation of Variance Share ($\gamma$) across Patch Variance Contrast ($\rho$)"
    )
    plt.xlabel(
        r"Patch Variance Ratio $\rho = \sigma_1^2/\sigma_2^2$ (Log Scale)"
    )
    plt.ylabel(r"Population Variance Share $\gamma$")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, which="both", alpha=0.25)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### shoham say stuff
    its really intersting that we get the sigmoid shape.
    its biologicaly reasonable to assume log scale sensing, from webber law.
    and the realtion of gamma and rho is derivied matmaticaly and its a relation that represent the information we get in a relation to the real information in the environment.
    so maybe risk averse and risk prone can arise from this alone?
    TODO i should check on that ::lucide:alarm-clock-check::
    """)
    return


@app.cell
def _(
    bf_n1_sym,
    bf_n2_sym,
    bf_s1_2_sym,
    bf_s2_2_sym,
    bf_sigma1_2_sym,
    bf_sigma2_2_sym,
    sym,
):
    bf_c_sym = sym.simplify((bf_s1_2_sym / bf_n1_sym)/ (bf_s1_2_sym / bf_n1_sym+ bf_s2_2_sym / bf_n2_sym))
    bf_gamma_sym = sym.simplify((bf_sigma1_2_sym / bf_n1_sym)/ (bf_sigma1_2_sym / bf_n1_sym+ bf_sigma2_2_sym / bf_n2_sym))
    bf_rho_sym = sym.symbols(r"\rho",positive=True,)
    bf_gamma_from_rho_sym = sym.simplify((bf_rho_sym * bf_n2_sym)/ (bf_rho_sym * bf_n2_sym+ bf_n1_sym))
    bf_rho_from_gamma_sym = sym.solve(sym.Eq(bf_gamma_sym,bf_gamma_from_rho_sym,),bf_rho_sym,)[0]
    return bf_c_sym, bf_gamma_from_rho_sym, bf_gamma_sym, bf_rho_sym


@app.cell
def _(bf_c_share_sym, bf_f1_sym, bf_f2_sym, bf_gamma_share_sym, sym):
    bf_D_sym, bf_W_sym = sym.symbols(r"D W",positive=True,)
    bf_Z_sym, bf_U_sym = sym.symbols(r"Z U",real=True,)
    bf_U1_sym = sym.simplify(bf_f1_sym * bf_c_share_sym* bf_D_sym/ (bf_gamma_share_sym* bf_W_sym))
    bf_U2_sym = sym.simplify(bf_f2_sym* (1 - bf_c_share_sym)* bf_D_sym/ ((1 - bf_gamma_share_sym)* bf_W_sym))
    bf_U_from_components_sym = sym.simplify(bf_U1_sym + bf_U2_sym)
    bf_phi_numerator_sym = sym.simplify(bf_U_from_components_sym* bf_W_sym/ bf_D_sym)
    bf_df_total_rigorous_sym = sym.simplify(bf_f1_sym + bf_f2_sym)
    bf_phi2_rigorous_sym = sym.simplify(bf_phi_numerator_sym/ bf_df_total_rigorous_sym)
    bf_phi_rigorous_sym = sym.sqrt(bf_phi2_rigorous_sym)
    bf_D_over_W_sym = sym.simplify(bf_U_sym/ bf_phi_numerator_sym)
    bf_W_over_D_sym = sym.simplify(1 / bf_D_over_W_sym)
    bf_t_from_Z_U_sym = sym.simplify(bf_Z_sym/ sym.sqrt(bf_U_sym/ bf_df_total_rigorous_sym))
    bf_v_from_Z_U_sym = sym.simplify(bf_Z_sym* sym.sqrt(bf_W_over_D_sym))
    bf_v_from_t_sym = sym.simplify(bf_phi_rigorous_sym* bf_t_from_Z_U_sym)
    return (
        bf_U1_sym,
        bf_U2_sym,
        bf_U_from_components_sym,
        bf_W_over_D_sym,
        bf_phi2_rigorous_sym,
        bf_phi_rigorous_sym,
        bf_t_from_Z_U_sym,
        bf_v_from_Z_U_sym,
        bf_v_from_t_sym,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Conditional distribution of the standardized difference

    Define the observed and population variances of the difference of sample means by

    \[
    D = \frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}, \qquad W = \frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}.
    \]

    Thus, $D$ is the observed squared standard error of $\bar y_1-\bar y_2$, whereas $W$ is its unknown population variance. Because the samples are Normal,

    \[
    Z=\frac{(\bar y_1-\bar y_2)-(\mu_1-\mu_2)}{\sqrt{W}}\sim N(0,1).
    \]

    The standardized difference introduced earlier can therefore be written as $v$, the Behrens--Fisher statistic:
    \[
    v = \frac{(\bar y_1-\bar y_2)-(\mu_1-\mu_2) }{\sqrt{D} } = Z\sqrt{\frac{W}{D}}.
    \]

    To derive the ratio $W/D$, define the two independent chi-squared quantities

    \[
    U_1 = \frac{f_1s_1^2}{\sigma_1^2}, \qquad U_2 = \frac{f_2s_2^2}{\sigma_2^2},  \Rightarrow U_1\sim \chi^2_{f_1}, \qquad U_2\sim\chi^2_{f_2}, \qquad U=U_1+U_2\sim\chi^2_{f_1+f_2}.
    \]
    """)
    return


@app.cell
def _(
    bf_U1_sym,
    bf_U2_sym,
    bf_U_from_components_sym,
    bf_W_over_D_sym,
    bf_phi2_rigorous_sym,
    bf_phi_rigorous_sym,
    mo,
    sym,
):
    mo.md(
        r"""
    Using the definitions of $c$ and $\gamma$, SymPy gives  
    \[
    U_1 ="""+ sym.latex(bf_U1_sym)+ r""", \qquad U_2="""+ sym.latex(bf_U2_sym)+ r"""\Longrightarrow  U=U_1+U_2="""   + sym.latex(bf_U_from_components_sym) + r"""
    \]

    We would define $\phi(c, \gamma)^2$ as the factor 
    \[
    U=(f_1+f_2)\phi(c,\gamma)^2\frac{D}{W}
    \]

    Since $f_1+f_2=n_1+n_2-2$, 

    \[
    \phi(c,\gamma)^2="""+ sym.latex(bf_phi2_rigorous_sym)+ r""" \Longrightarrow \phi(c,\gamma)="""   + sym.latex(bf_phi_rigorous_sym)+ r"""
    \]

    Solving this identity for the ratio of the two variance terms gives
    \[
    \frac{W}{D}="""+ sym.latex(bf_W_over_D_sym)+ r"""
    \]

    """
    )
    return


@app.cell
def _(bf_t_from_Z_U_sym, bf_v_from_Z_U_sym, bf_v_from_t_sym, mo, sym):
    mo.md(
        r"""
    Substituting this expression into $v=Z\sqrt{W/D}$ gives
    \[
    v="""+ sym.latex(bf_v_from_Z_U_sym)+ r"""
    \]
    Define
    \[
    T="""+ sym.latex(bf_t_from_Z_U_sym)+ r"""
    \]

    Since $Z\sim N(0,1)$, $U\sim\chi^2_{f_1+f_2}$, and these variables are
    independent, the Student's $t$ result developed above gives $T\sim t_{f_1+f_2}$. Therefore,

    \[
    v="""+ sym.latex(bf_v_from_t_sym)   + r"""\Longrightarrow v\mid c,\gamma\sim\phi(c,\gamma)t_{f_1+f_2}.
    \]

    $\phi(c,\gamma)$ is the **conditional scale factor** of the Behrens--Fisher statistic."""
    )
    return


@app.cell
def _(np, plt, sp):
    # Degrees of freedom matching root sample sizes

    _n1, _n2 = 10, 10

    _f1, _f2 = _n1 - 1, _n2 - 1

    _f_total = _f1 + _f2

    _N = 100_000  # Repetitions


    # Draw independent Chi-Squared random variables

    _rng = np.random.default_rng(42)

    _U1_draws = _rng.chisquare(df=_f1, size=_N)

    _U2_draws = _rng.chisquare(df=_f2, size=_N)


    # Sum of Chi-Squared variables: U = U1 + U2

    _U_draws = _U1_draws + _U2_draws


    # Theoretical densities

    _u_grid = np.linspace(0, 40, 500)

    _pdf_U1 = sp.stats.chi2.pdf(_u_grid, df=_f1)

    _pdf_U2 = sp.stats.chi2.pdf(_u_grid, df=_f2)

    _pdf_U = sp.stats.chi2.pdf(_u_grid, df=_f_total)


    plt.figure(figsize=(9, 4.5))


    # Histograms

    plt.hist(_U1_draws, bins=80, density=True, alpha=0.3, color="crimson", label=rf"Sample 1: $U_1 \sim \chi^2_{{{_f1}}}$")

    plt.hist(_U2_draws, bins=80, density=True, alpha=0.3, color="teal", label=rf"Sample 2: $U_2 \sim \chi^2_{{{_f2}}}$")

    plt.hist(_U_draws, bins=80, density=True, alpha=0.4, color="navy", label=rf"Combined: $U = U_1 + U_2 \sim \chi^2_{{{_f_total}}}$")


    # Overlay PDF curves

    plt.plot(_u_grid, _pdf_U1, color="crimson", lw=2, linestyle="--")

    plt.plot(_u_grid, _pdf_U2, color="teal", lw=2, linestyle="--")

    plt.plot(_u_grid, _pdf_U, color="navy", lw=2.5, label=rf"Theory $\chi^2_{{{_f_total}}}$ PDF")


    plt.title(rf"Additive Property of Variance Noise: $U_1 + U_2 = U \sim \chi^2_{{{_f_total}}}$")

    plt.xlabel(r"Value ($u$)")

    plt.ylabel("Probability Density")

    plt.legend(fontsize=9)

    plt.grid(alpha=0.2)

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plotting the realtion between the the parameter to the observed one
    """)
    return


@app.cell
def _(np, plt):
    # Grid of observed share (c) vs true population share (gamma)
    _c_vals = np.linspace(0.01, 0.99, 300)
    _gamma_vals = np.linspace(0.01, 0.99, 300)
    _C, _G = np.meshgrid(_c_vals, _gamma_vals)

    # Degrees of freedom (e.g., f1 = f2 = 9 for n1 = n2 = 10)
    _f1, _f2 = 9, 9

    # Scale factor formula: (f1+f2)*phi^2 = (f1*c)/gamma + (f2*(1-c))/(1-gamma)
    _phi2 = (_f1 * _C / _G + _f2 * (1.0 - _C) / (1.0 - _G)) / (_f1 + _f2)
    _phi = np.sqrt(_phi2)

    plt.figure(figsize=(8.5, 6.5))

    # 1. Filled Color Map using the same color scale as the ratio plot
    _levels = np.linspace(0.7, 2.0, 50)
    _cf = plt.contourf(_C, _G, _phi, levels=_levels, cmap="RdYlBu_r", extend="both")
    _cbar = plt.colorbar(_cf)
    _cbar.set_label(r"Scale Factor Penalty $\phi(c, \gamma)$", fontsize=10)

    # 2. Faint Overlay Contour Lines for Precision
    _line_levels = [0.8, 0.9, 0.95, 1.0, 1.1, 1.3, 1.5, 2.0]
    _cs = plt.contour(_C, _G, _phi, levels=_line_levels, colors="black", linewidths=0.6, alpha=0.7)
    plt.clabel(_cs, inline=True, fontsize=8, fmt=r"$\phi=%.2f$")

    # 3. Line of Agreement (c = gamma where phi = 1.0)
    plt.plot([0, 1], [0, 1], color="black", linestyle="--", lw=2, label=r"Agreement Line ($c = \gamma \Rightarrow \phi = 1$)")

    # 4. Highlight Central Equal Share Point (0.5, 0.5)
    plt.scatter(0.5, 0.5, color="black", s=70, zorder=5)

    plt.title(rf"Direct Share Space: Observed ($c$) vs. True ($\gamma$) Scale Factor $\phi(c,\gamma)$ ($f_1={_f1}, f_2={_f2}$)")
    plt.xlabel(r"Observed Variance Share $c$ (from Root Sample)")
    plt.ylabel(r"True Population Variance Share $\gamma$ (Parameter)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    plt.grid(True, alpha=0.15, color="k")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    for sanity check, note that for $f_1=f_2$ we get the same graph as in the paper:
    ![alt](public\brown_fig_dim.png)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### shoham writing stuff
    it may be intersting to compare the relationship between c and rho even under linear assumptions between c and gamma since we saw previously the sigmoid relation between c to rho.

    the reason the relation bertween c and gamma is not neccerly linear is that c is gamma with added noise: its the differnce between the population variance to the real variance (sigma vs S and stuff)
    """)
    return


@app.cell
def _(np, plt):
    # Log-spaced range for observed sample variance ratio rho_hat = s1^2 / s2^2
    _rho_hat = np.logspace(-1.3, 1.3, 500)

    # Under equal sample sizes (n1 = n2 = n), c relates to rho_hat via c = rho_hat / (rho_hat + 1)
    _c_share = _rho_hat / (_rho_hat + 1.0)

    plt.figure(figsize=(8.5, 4.8))

    # 1. Sigmoid S-curve mapping observed sample variance ratio to observed share
    plt.plot(
        _rho_hat,
        _c_share,
        color="darkgreen",
        lw=2.5,
        label=r"Sample Mapping: $c = \frac{\hat{\rho}}{\hat{\rho} + 1}$",
    )

    # 2. Point of Sample Variance Equality (rho_hat = 1.0, c = 0.5)
    _rho_eq, _c_eq = 1.0, 0.5
    plt.scatter(_rho_eq, _c_eq, color="crimson", s=70, zorder=5)

    plt.annotate(
        r"Equal Sample Variances" + "\n" + r"$(\hat{\rho} = 1.0, c = 0.5)$",
        (_rho_eq, _c_eq),
        textcoords="offset points",
        xytext=(15, -15),
        fontsize=9,
        color="crimson",
        weight="bold",
    )

    # 3. Asymptotic Bounds at c = 0 and c = 1
    plt.axhline(1.0, color="gray", linestyle=":", alpha=0.6)
    plt.axhline(0.0, color="gray", linestyle=":", alpha=0.6)

    # 4. Mathematical Formula Box
    plt.text(
        0.12,
        0.80,
        r"$c = \frac{\hat{\rho}}{\hat{\rho} + 1}, \quad \text{where } \hat{\rho} = \frac{s_1^2}{s_2^2}$",
        fontsize=14,
        color="darkgreen",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="honeydew",
            edgecolor="darkgreen",
            alpha=0.8,
        ),
    )

    plt.xscale("log")  # Set X-axis to Logarithmic Scale
    plt.title(
        r"Transformation of Sample Variance Ratio ($\hat{\rho}$) to Observed Share ($c$)"
    )
    plt.xlabel(
        r"Sample Variance Ratio $\hat{\rho} = s_1^2 / s_2^2$ (Log Scale)"
    )
    plt.ylabel(r"Observed Sample Variance Share $c$")
    plt.ylim(-0.0, 1.0)
    plt.grid(True, which="both", alpha=0.25)
    plt.show()
    return


@app.cell
def _(
    bf_f1_sym,
    bf_f2_sym,
    bf_gamma_from_rho_sym,
    bf_gamma_sym,
    bf_phi2_sym,
    bf_rho_sym,
    sym,
):
    # Substitute gamma = (rho * n2) / (rho * n2 + n1) into phi^2(c, gamma)
    bf_phi2_from_rho_sym = sym.simplify(bf_phi2_sym.subs(bf_gamma_sym, bf_gamma_from_rho_sym))
    # Express phi(c, rho) analytically
    bf_phi_from_rho_sym = sym.sqrt(bf_phi2_from_rho_sym)
    # Special Case: Equal Sample Sizes (n1 = n2 => f1 = f2)
    # Replaces gamma with rho / (rho + 1)
    bf_gamma_equal_sampling = bf_rho_sym / (bf_rho_sym + 1)
    bf_phi2_equal_sampling_sym = sym.simplify(bf_phi2_sym.subs([(bf_f2_sym, bf_f1_sym), (bf_gamma_sym, bf_gamma_equal_sampling)]))
    bf_phi_equal_sampling_sym = sym.sqrt(bf_phi2_equal_sampling_sym)
    return (
        bf_phi2_equal_sampling_sym,
        bf_phi2_from_rho_sym,
        bf_phi_equal_sampling_sym,
        bf_phi_from_rho_sym,
    )


@app.cell
def _(
    bf_phi2_equal_sampling_sym,
    bf_phi2_from_rho_sym,
    bf_phi_equal_sampling_sym,
    bf_phi_from_rho_sym,
    mo,
    sym,
):
    mo.md(rf"""
    ### Transforming the Scale Factor to Physical Patch Contrast ($\rho$)
    While the scale factor $\phi(c, \gamma)$ is naturally expressed using the variance share $\gamma \in (0,1)$[cite: 1], plant foraging decisions operate directly on physical soil patch variance ratios $\rho = \frac{{\sigma_1^2}}{{\sigma_2^2}} \in (0, \infty)$[cite: 1].
    Using the parameter mapping $\gamma = \frac{{\rho n_2}}{{\rho n_2 + n_1}}$, SymPy substitutes this relationship directly into $\phi(c, \gamma)^2$ to yield the scale factor as a function of observed share $c$ and physical patch contrast $\rho$[cite: 1]:
    $$
    \phi(c, \rho)^2 = {sym.latex(bf_phi2_from_rho_sym)}
    $$
    Taking the square root gives:
    $$
    \phi(c, \rho) = {sym.latex(bf_phi_from_rho_sym)}
    $$

    #### Equal Sampling Simplification ($n_1 = n_2$)
    Under equal sample allocation, $f_1 = f_2$ and $\gamma = \frac{{\rho}}{{\rho + 1}}$[cite: 1]. SymPy simplifies the scale factor squared down to[cite: 1]:
    $$
    \phi(c, \rho)^2 = {sym.latex(bf_phi2_equal_sampling_sym)}
    $$
    And the explicit scale factor becomes:
    $$
    \phi(c, \rho) = {sym.latex(bf_phi_equal_sampling_sym)}
    $$
    We can now map this scale factor directly into the $2\text{{D}}$ phase space of observed variance ratio ($\hat{{\rho}}$) vs. true variance ratio ($\rho$)[cite: 1].
    """)
    return


@app.cell
def _(np, plt):
    # Log-spaced grids for sample ratio (rho_hat) and true ratio (rho)
    _rho_hat_vals = np.logspace(-1.3, 1.3, 300)
    _rho_vals = np.logspace(-1.3, 1.3, 300)
    _RHO_HAT, _RHO = np.meshgrid(_rho_hat_vals, _rho_vals)

    # Convert ratios to variance shares under equal sampling (n1 = n2)
    _C = _RHO_HAT / (_RHO_HAT + 1.0)
    _G = _RHO / (_RHO + 1.0)

    # Scale factor phi formula: (f1=f2) => phi = sqrt( 0.5 * (c/gamma + (1-c)/(1-gamma)) )
    _phi2 = 0.5 * (_C / _G + (1.0 - _C) / (1.0 - _G))
    _phi = np.sqrt(_phi2)

    plt.figure(figsize=(8.5, 6.5))

    # 1. Filled Color Map for Scale Factor Phi
    _levels = np.linspace(0.7, 2.0, 50)
    _cf = plt.contourf(_RHO_HAT, _RHO, _phi, levels=_levels, cmap="RdYlBu_r", extend="both")
    _cbar = plt.colorbar(_cf)
    _cbar.set_label(r"Scale Factor Penalty $\phi(\hat{\rho}, \rho)$", fontsize=10)

    # 2. Overlay Faint Line Contours for Specific Values
    _line_levels = [0.8, 0.9, 1.0, 1.1, 1.3, 1.5, 1.8, 2.0]
    _cs = plt.contour(_RHO_HAT, _RHO, _phi, levels=_line_levels, colors="k", linewidths=0.6, alpha=0.7)
    plt.clabel(_cs, inline=False, fontsize=8, fmt=r"$\phi=%.2f$")

    # 3. Agreement Line (rho_hat = rho)
    plt.plot(
        [0.05, 20],
        [0.05, 20],
        color="black",
        linestyle="--",
        lw=2,
        label=r"Agreement Line ($\hat{\rho} = \rho \Rightarrow \phi = 1$)",
    )

    # 4. Point of Complete Equality (1.0, 1.0)
    plt.scatter(1.0, 1.0, color="black", s=70, zorder=5)

    # Log-Log Scale
    plt.xscale("log")
    plt.yscale("log")

    plt.title(r"2D Color Phase Space: Scale Factor $\phi(\hat{\rho}, \rho)$ Uncertainty Landscape")
    plt.xlabel(r"Observed Sample Variance Ratio $\hat{\rho} = s_1^2 / s_2^2$ (Data on X-Axis)")
    plt.ylabel(r"True Population Variance Ratio $\rho = \sigma_1^2 / \sigma_2^2$ (Parameter on Y-Axis)")
    plt.xlim(0.05, 20)
    plt.ylim(0.05, 20)
    plt.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    plt.grid(True, which="both", alpha=0.15, color="k")
    plt.show()
    return


@app.cell
def _(np, plt):
    # Grid of observed share c in (0, 1) and true variance ratio rho on log scale
    _c_vals = np.linspace(0.01, 0.99, 300)
    _rho_vals = np.logspace(-1.3, 1.3, 300)
    _C, _RHO = np.meshgrid(_c_vals, _rho_vals)

    # Degrees of freedom (equal sample sizes, e.g., f1 = f2 = 9)
    _f1, _f2 = 9, 9

    # Convert true population ratio (rho) to population variance share (gamma)
    _G = _RHO / (_RHO + 1.0)

    # Scale factor phi formula: (f1+f2)*phi^2 = (f1*c)/gamma + (f2*(1-c))/(1-gamma)
    _phi2 = (_f1 * _C / _G + _f2 * (1.0 - _C) / (1.0 - _G)) / (_f1 + _f2)
    _phi = np.sqrt(_phi2)

    plt.figure(figsize=(8.5, 6.5))

    # 1. Filled Color Map for Scale Factor Phi
    _levels = np.linspace(0.7, 2.0, 50)
    _cf = plt.contourf(_C, _RHO, _phi, levels=_levels, cmap="RdYlBu_r", extend="both")
    _cbar = plt.colorbar(_cf)
    _cbar.set_label(r"Scale Factor Penalty $\phi(c, \rho)$", fontsize=10)

    # 2. Overlay Faint Line Contours
    _line_levels = [0.8, 0.9, 0.95, 1.0, 1.1, 1.3, 1.5, 1.8, 2.0]
    _cs = plt.contour(_C, _RHO, _phi, levels=_line_levels, colors="k", linewidths=0.6, alpha=0.7)
    plt.clabel(_cs, inline=False, fontsize=8, fmt=r"$\phi=%.2f$")

    # 3. Agreement Curve: c = rho / (rho + 1) where phi = 1.0
    _rho_curve = np.logspace(-1.3, 1.3, 200)
    _c_agreement = _rho_curve / (_rho_curve + 1.0)
    plt.plot(_c_agreement, _rho_curve, color="black", linestyle="--", lw=2, label=r"Agreement Curve ($c = \frac{\rho}{\rho+1} implies \phi = 1$)")

    # 4. Highlight Central Point of Complete Equality (c = 0.5, rho = 1.0)
    plt.scatter(0.5, 1.0, color="black", s=70, zorder=5)

    # Y-axis is Logarithmic for true variance ratio rho; X-axis is Linear for share c
    plt.yscale("log")

    plt.title(r"2D Hybrid Phase Space: Observed Share ($c$) vs. Population Contrast ($\rho$)")
    plt.xlabel(r"Observed Variance Share $c = \frac{s_1^2/n_1}{s_1^2/n_1 + s_2^2/n_2}$ (Linear Scale)")
    plt.ylabel(r"True Population Variance Ratio $\rho = \sigma_1^2 / \sigma_2^2$ (Log Scale)")
    plt.xlim(0, 1)
    plt.ylim(0.05, 20)
    plt.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    plt.grid(True, which="both", alpha=0.15, color="k")
    plt.show()
    return


@app.cell
def _(np, plt, sp):
    _c_grid = np.linspace(0.01, 0.99, 500)
    _f1, _f2 = 9, 9 # Assuming n1=10, n2=10
    plt.figure(figsize=(8, 4))
    for _gamma_true in [0.2, 0.5, 0.8]:
        _z_stat = (_c_grid / (1.0 - _c_grid)) * ((1.0 - _gamma_true) / _gamma_true)
        _dz_dc = (1.0 - _gamma_true) / (_gamma_true * (1.0 - _c_grid)**2)
        _likelihood = sp.stats.f.pdf(_z_stat, dfn=_f1, dfd=_f2) * _dz_dc
        plt.plot(_c_grid, _likelihood, lw=2, label=rf"True $\gamma = {_gamma_true}$")
    plt.title(r"Likelihood of Observed Share $c$ Given True Share $\gamma$ ($n_1=n_2=10$)")
    plt.xlabel(r"Observed Variance Share $c$")
    plt.ylabel(r"Likelihood $p(c \mid \gamma)$")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    return


@app.cell
def _(
    bf_c_sym,
    bf_delta_sym,
    bf_f1_sym,
    bf_f2_sym,
    bf_gamma_sym,
    bf_n1_sym,
    bf_n2_sym,
    bf_s1_2_sym,
    bf_s2_2_sym,
    bf_ybar1_sym,
    bf_ybar2_sym,
    sym,
):
    bf_v_sym = sym.simplify(((bf_ybar1_sym - bf_ybar2_sym)- bf_delta_sym)/ sym.sqrt(bf_s1_2_sym / bf_n1_sym+ bf_s2_2_sym / bf_n2_sym))
    bf_df_total_sym = sym.simplify(bf_f1_sym + bf_f2_sym)
    bf_phi2_sym = sym.simplify((bf_f1_sym * bf_c_sym / bf_gamma_sym + bf_f2_sym * (1 - bf_c_sym)/ (1 - bf_gamma_sym))/ bf_df_total_sym)
    bf_phi_sym = sym.sqrt(bf_phi2_sym)
    return (bf_phi2_sym,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Likelihood of the observed variance share
    To obtain a likelihood for $\gamma$, hold the unknown population share $\gamma$ fixed and regard the observed share $c$ as random.
    For Normal samples,
    \[
    U_1=\frac{f_1s_1^2}{\sigma_1^2}\sim\chi^2_{f_1},\qquad U_2=\frac{f_2s_2^2}{\sigma_2^2}\sim\chi^2_{f_2}.
    \]
    Since $U_1$ and $U_2$ are independent, by defenition of the $F$ distribution,
    \[
    z(c,\gamma)\equiv \frac{U_1/f_1}{U_2/f_2}=\frac{s_1^2/\sigma_1^2}{s_2^2/\sigma_2^2}\sim F_{f_1,f_2}.
    \]
    """)
    return


@app.cell
def _(bf_F_statistic_derivative_sym, bf_F_statistic_sym, mo, sym):
    mo.md(rf"""
    Substituting the definitions of $c$ and $\gamma$, SymPy gives
    \[
    z(c,\gamma) = {sym.latex(bf_F_statistic_sym)}
    \]

    Using $p(c) dc = f_F(z) dz$, we get that $p(c) = f_F(z) \cdot \frac{{\partial z}}{{\partial c}}$. 
    Using sympy: 
    \[
    \frac{{\partial z}}{{\partial c}} = {sym.latex(bf_F_statistic_derivative_sym)}
    \]
    Because $0<c<1$ and $0<\gamma<1$, the derivative is positive. Since $p(c\mid\gamma)=f_{{F_{{f_1,f_2}}}}\cdot z(c,\gamma) \cdot  \frac{{\partial z}}{{\partial c}}$, the likelihood for the unknown population variance share $\gamma$, conditional on the observed share $c$ is:
    \[
    p(c\mid\gamma)=f_{{F_{{f_1,f_2}}}}\left({sym.latex(bf_F_statistic_sym)}\right)\left({sym.latex(bf_F_statistic_derivative_sym)}\right).
    \]

    where $f_{{F_{{f_1,f_2}}}}$ is the density of an $F_{{f_1,_2}}$ random variable.
    """)
    return


@app.cell
def _(sym):
    bf_c_share_sym, bf_gamma_share_sym = sym.symbols(r"c \gamma",positive=True,)
    bf_F_statistic_sym = sym.cancel((bf_c_share_sym/ (1 - bf_c_share_sym))* ((1 - bf_gamma_share_sym)/ bf_gamma_share_sym))
    bf_F_statistic_derivative_sym = sym.simplify(sym.diff(bf_F_statistic_sym,bf_c_share_sym,))
    return (
        bf_F_statistic_derivative_sym,
        bf_F_statistic_sym,
        bf_c_share_sym,
        bf_gamma_share_sym,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Prior for relative patch variability

    The physical quantity for which a prior is specified is
    \[
    \rho=\frac{\sigma_1^2}{\sigma_2^2},
    \]
    not $\gamma$. In a biological interpretation, this prior represents the distribution of relative patch variabilities that has characterized the environment over evolutionary or developmental time.
    Following Brown, use
    \[
    \rho\sim kF_{k_1,k_2}.
    \]
    The scale $k$ locates the prior on the variance-ratio scale, while $k_1$ and $k_2$ control its concentration. For example, a prior centered near $\rho=1$ expresses an environment in which the two patches are usually similarly variable.

    Because
    \[
    \rho=\frac{n_1\gamma}{n_2(1-\gamma)}, \qquad \left|\frac{d\rho}{d\gamma}\right| =\frac{n_1}{n_2(1-\gamma)^2},
    \]
    this prior induces a density $\pi_\gamma(\gamma)$ on $0<\gamma<1$. The transformation to $\gamma$ is mathematical only: the ecological interpretation remains a prior on $\rho$.

    The formal reference prior $\pi_\rho(\rho)\propto1/\rho$ induces
    \[
    \pi_\gamma(\gamma)\propto\frac{1}{\gamma(1-\gamma)}.
    \]
    It is useful as a reference case, but the proper \(kF\) prior is more suitable for the numerical plant model because it permits explicit, biologically meaningful assumptions about plausible variance contrasts.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Posterior uncertainty about relative patch variability

    Bayes' rule combines the likelihood of the observed sample-variance share with the induced prior:
    \[
    p(\gamma\mid c)=\frac{p(c\mid\gamma)\pi_\gamma(\gamma)}{\int_0^1p(c\mid u)\pi_\gamma(u)\,du}.
    \]

    This posterior answers a specific question: given the observed variation in the two root-sampling histories, which population variance contrasts remain plausible? It does not yet tell the plant where to allocate roots. Its role is to carry uncertainty about patch variability into inference about the mean resource contrast $\delta$.
    """)
    return


@app.cell
def _(np, sp):
    def bf_variance_share_posterior(c, n1, n2, k=1.0, k1=10.0, k2=10.0, grid_size=2000):
        f1, f2 = n1 - 1, n2 - 1
        eps = 1e-6
        gamma_grid = np.linspace(eps, 1.0 - eps, grid_size)
        rho_grid = n1 * gamma_grid / (n2 * (1.0 - gamma_grid))
        drho_dgamma = n1 / (n2 * (1.0 - gamma_grid) ** 2)
        prior_rho = sp.stats.f.pdf(rho_grid / k, dfn=k1, dfd=k2) / k
        prior_gamma = prior_rho * drho_dgamma
        z = c / (1.0 - c) * (1.0 - gamma_grid) / gamma_grid
        dz_dc = (1.0 - gamma_grid) / (gamma_grid * (1.0 - c) ** 2)
        likelihood = sp.stats.f.pdf(z, dfn=f1, dfd=f2) * dz_dc
        posterior_kernel = likelihood * prior_gamma
        posterior = posterior_kernel / sp.integrate.trapezoid(posterior_kernel, gamma_grid)
        return gamma_grid, prior_gamma, likelihood, posterior

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Secondary prior for the relative variance

    The primary parameter is the difference between the means, $\delta=\mu_1-\mu_2$. Brown's approach does not require separate prior assumptions about the two population variances. Instead, it specifies a prior only for their ratio,

    \[
    \rho=\frac{\sigma_1^2}{\sigma_2^2}.
    \]

    This ratio determines the population variance share $\gamma$. Therefore, a prior density $\pi_\rho(\rho)$ induces a prior density $\pi_\gamma(\gamma)$.

    A flexible family used by Brown is $\rho\sim kF_{k_1,k_2}$, where $k$ determines the expected scale of the variance ratio, and $k_1,k_2$ determine how concentrated the prior is. The reference case is

    \[
    \pi_\rho(\rho)\propto\frac{1}{\rho}.
    \]

    This assigns equal prior weight to equal multiplicative changes in the variance ratio. We derive the corresponding prior for $\gamma$ below.
    """)
    return


@app.cell
def _(bf_gamma_share_sym, bf_n1_sym, bf_n2_sym, sym):
    bf_rho_share_sym = sym.symbols( r"\rho", positive=True,)
    bf_rho_from_gamma_share_sym = sym.solve(sym.Eq(bf_gamma_share_sym,(bf_rho_share_sym * bf_n2_sym)/ (bf_rho_share_sym * bf_n2_sym+ bf_n1_sym),),bf_rho_share_sym,)[0]
    bf_rho_gamma_jacobian_sym = sym.simplify(sym.diff(bf_rho_from_gamma_share_sym,bf_gamma_share_sym,))
    bf_pi_rho_sym = sym.Function(r"\pi_\rho")
    bf_pi_gamma_sym = sym.simplify(bf_pi_rho_sym(bf_rho_from_gamma_share_sym)* bf_rho_gamma_jacobian_sym)
    bf_reference_pi_rho_sym = 1 / bf_rho_share_sym
    bf_reference_pi_gamma_sym = sym.simplify(bf_reference_pi_rho_sym.subs(bf_rho_share_sym,bf_rho_from_gamma_share_sym,)* bf_rho_gamma_jacobian_sym)
    return (
        bf_pi_gamma_sym,
        bf_reference_pi_gamma_sym,
        bf_rho_from_gamma_share_sym,
        bf_rho_gamma_jacobian_sym,
    )


@app.cell
def _(
    bf_pi_gamma_sym,
    bf_reference_pi_gamma_sym,
    bf_rho_from_gamma_share_sym,
    bf_rho_gamma_jacobian_sym,
    mo,
    sym,
):
    mo.md(
        r"""
    The inverse transformation from $\gamma$ to the variance ratio is

    \[
    \rho="""+ sym.latex(bf_rho_from_gamma_share_sym)+ r"""
    \]

    and SymPy gives the Jacobian

    \[
    \left| \frac{d\rho}{d\gamma}\right|="""+ sym.latex(bf_rho_gamma_jacobian_sym)+ r"""
    \]

    Thus, the change-of-variables formula gives the induced prior density
    \[
    \pi_\gamma(\gamma)="""+ sym.latex(bf_pi_gamma_sym)+ r"""
    \]

    For the reference prior

    \[
    \pi_\rho(\rho)\propto\frac{1}{\rho},
    \]

    SymPy gives

    \[
    \pi_\gamma(\gamma) \propto """+ sym.latex(bf_reference_pi_gamma_sym)+ r"""
    \]

    This reference prior is improper because it diverges as $\gamma$ approaches zero or one. An informative prior, such as $\rho\sim kF_{k_1,k_2}$, can instead concentrate prior weight on biologically plausible values of the relative variance.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Posterior distribution of the population variance share
    The likelihood derived in the preceding step is $p(c\mid\gamma)$.
    The secondary prior specifies the density $\pi_\gamma(\gamma)$. Therefore, Bayes' rule gives the posterior density of the unknown population variance share:
    \[
    p(\gamma\mid c) \propto p(c\mid\gamma)\pi_\gamma(\gamma).
    \]
    This posterior distribution combines the observed relative variability of the two samples with the prior assumptions about theirpopulation variance ratio.
    """)
    return


@app.cell
def _(
    bf_F_statistic_derivative_sym,
    bf_F_statistic_sym,
    bf_gamma_share_sym,
    bf_pi_gamma_sym,
    sym,
):
    bf_f_F_sym = sym.Function(r"f_F")
    bf_likelihood_c_given_gamma_sym = sym.Mul(bf_f_F_sym(bf_F_statistic_sym),bf_F_statistic_derivative_sym, evaluate=False,)
    bf_posterior_gamma_kernel_sym = sym.Mul(bf_likelihood_c_given_gamma_sym,bf_pi_gamma_sym,evaluate=False,)
    bf_gamma_integration_sym = sym.symbols(r"u",positive=True,)
    bf_posterior_gamma_normalizer_sym = sym.Integral(bf_posterior_gamma_kernel_sym.subs(bf_gamma_share_sym, bf_gamma_integration_sym,),(bf_gamma_integration_sym,0,1,),)
    bf_posterior_gamma_sym = sym.Mul(bf_posterior_gamma_kernel_sym,1 / bf_posterior_gamma_normalizer_sym,evaluate=False,)
    return (
        bf_likelihood_c_given_gamma_sym,
        bf_posterior_gamma_kernel_sym,
        bf_posterior_gamma_normalizer_sym,
        bf_posterior_gamma_sym,
    )


@app.cell
def _(
    bf_likelihood_c_given_gamma_sym,
    bf_posterior_gamma_kernel_sym,
    bf_posterior_gamma_normalizer_sym,
    bf_posterior_gamma_sym,
    mo,
    sym,
):
    mo.md(
        r"""
    Let $f_F$ denote the density of an $F_{f_1,f_2}$ random variable. The likelihood obtained in Step 4 is

    \[
    p(c\mid\gamma)="""    + sym.latex(bf_likelihood_c_given_gamma_sym) + r"""
    \]

    Multiplying this likelihood by the prior gives the unnormalized posterior
    density,

    \[
    p(\gamma\mid c) \propto"""+ sym.latex(bf_posterior_gamma_kernel_sym)+ r"""
    \]

    The normalizing constant is

    \[
    """  + sym.latex(bf_posterior_gamma_normalizer_sym)  + r"""
    \]

    Therefore, the posterior density is

    \[
    p(\gamma\mid c)="""    + sym.latex(bf_posterior_gamma_sym)    + r"""
    \]

    This integral usually has no simple closed form. In the numerical part ofthe notebook, it will be evaluated on a grid of values $0<\gamma<1$ and normalized numerically.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Posterior mixture for the standardized difference

    In Step 3, we found that conditional on $\gamma$,

    \[
    v\mid c,\gamma \sim \phi(c,\gamma)t_{f_1+f_2}.
    \]

    The scale factor $\phi(c,\gamma)$ remains uncertain because $\gamma$ is unknown. We therefore average the conditional distribution of $v$ over the posterior distribution $p(\gamma\mid c)$.

    This produces a posterior mixture of scaled Student's $t$ distributions. In general, this mixture is not itself a Student's $t$ distribution.
    """)
    return


@app.cell
def _(bf_c_share_sym, bf_f1_sym, bf_f2_sym, bf_gamma_share_sym, sym):
    bf_v_display_sym = sym.symbols(r"v",real=True,)
    bf_phi2_conditional_sym = sym.simplify((bf_f1_sym* bf_c_share_sym/ bf_gamma_share_sym+ bf_f2_sym* (1 - bf_c_share_sym)/ (1 - bf_gamma_share_sym))/ (bf_f1_sym + bf_f2_sym))
    bf_phi_conditional_sym = sym.sqrt(bf_phi2_conditional_sym)
    bf_inverse_phi_conditional_sym = sym.simplify(1 / bf_phi_conditional_sym)
    bf_t_argument_sym = sym.simplify(bf_v_display_sym/ bf_phi_conditional_sym)
    return (
        bf_inverse_phi_conditional_sym,
        bf_phi2_conditional_sym,
        bf_phi_conditional_sym,
        bf_t_argument_sym,
    )


@app.cell
def _(
    bf_inverse_phi_conditional_sym,
    bf_phi2_conditional_sym,
    bf_phi_conditional_sym,
    bf_t_argument_sym,
    mo,
    sym,
):
    mo.md(
        r"""
    SymPy gives the conditional scale factor

    \[
    \phi(c,\gamma)^2="""+ sym.latex(bf_phi2_conditional_sym)
       + r"""
    \]

    and hence

    \[
    \phi(c,\gamma)="""+ sym.latex(bf_phi_conditional_sym)+ r"""
    \]

    If $f_{t_{f_1+f_2}}$ denotes the density of a Student's $t$ random variable with $f_1+f_2$ degrees of freedom, then the conditional densityof $v$ is

    \[
    p(v\mid c,\gamma)="""+ sym.latex(bf_inverse_phi_conditional_sym)+ r"""f_{t_{f_1+f_2}}\left("""+ sym.latex(bf_t_argument_sym)+ r"""\right).
    \]

    Finally, marginalizing over the posterior distribution of $\gamma$ gives

    \[
    p(v\mid c)=\int_0^1"""+ sym.latex(bf_inverse_phi_conditional_sym)+ r"""f_{t_{f_1+f_2}}\left("""+ sym.latex(bf_t_argument_sym)+ r"""\right)p(\gamma\mid c)\,d\gamma.
    \]

    This is the secondary Bayesian posterior distribution of the standardized difference between the two means.
    """
    )
    return


@app.cell
def _(np, plt, sp):
    # Setup numerical grid for gamma in (0, 1)

    _eps = 1e-5

    _gamma_grid = np.linspace(_eps, 1.0 - _eps, 500)


    # Example parameters

    _n1, _n2 = 5, 25  # Unbalanced sample sizes (e.g., f1=4, f2=24)[cite: 1]

    _f1, _f2 = _n1 - 1, _n2 - 1

    _c = 0.3  # Observed sample variance share[cite: 1]


    # Secondary Prior on rho = sigma1^2 / sigma2^2 ~ k * F(k1, k2)[cite: 1]

    _k, _k1, _k2 = 1.0, 10, 10



    # Transformation: rho = (n1 * gamma) / (n2 * (1 - gamma))[cite: 1]

    _rho_grid = (_n1 * _gamma_grid) / (_n2 * (1.0 - _gamma_grid))

    _drho_dgamma = (_n1 * _n2) / ((_n2 * (1.0 - _gamma_grid)) ** 2)



    # Prior pi(gamma) via change of variables[cite: 1]

    _pi_gamma = sp.stats.f.pdf(_rho_grid / _k, dfn=_k1, dfd=_k2) * (_drho_dgamma / _k)


    # Likelihood p(c | gamma)[cite: 1]

    _z_stat = (_c / (1.0 - _c)) * ((1.0 - _gamma_grid) / _gamma_grid)

    _dz_dc = (1.0 - _gamma_grid) / (_gamma_grid * (1.0 - _c) ** 2)

    _likelihood_c = sp.stats.f.pdf(_z_stat, dfn=_f1, dfd=_f2) * _dz_dc


    # Unnormalized & Normalized Posterior p(gamma | c)[cite: 1]

    _unnorm_post_gamma = _likelihood_c * _pi_gamma

    _norm_const = sp.integrate.trapezoid(_unnorm_post_gamma, _gamma_grid)

    _post_gamma = _unnorm_post_gamma / _norm_const


    # Plot 1: Prior vs Likelihood vs Posterior on Variance Share gamma

    plt.figure(figsize=(8, 4))

    plt.plot(_gamma_grid, _pi_gamma / sp.integrate.trapezoid(_pi_gamma, _gamma_grid), 'k:', label=r"Prior $\pi(\gamma)$")

    plt.plot(_gamma_grid, _likelihood_c / sp.integrate.trapezoid(_likelihood_c, _gamma_grid), 'r--', label=r"Likelihood $p(c \mid \gamma)$")

    plt.plot(_gamma_grid, _post_gamma, 'b-', lw=2, label=r"Posterior $p(\gamma \mid c)$")

    plt.title(r"Bayesian Update of Variance Share $\gamma$ ($c=0.3, n_1=5, n_2=25$)")

    plt.xlabel(r"Population Variance Share $\gamma$")

    plt.ylabel("Density")

    plt.legend()

    plt.grid(alpha=0.3)

    plt.show()
    return


if __name__ == "__main__":
    app.run()
