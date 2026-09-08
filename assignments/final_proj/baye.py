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
    # Some generals terms
    ### Nuisance parameter

    ### Jeffreys' Priors
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Some defenitions
    #### normal distribution
    #### chi-squared distribution with degree of freedome
    #### Student's t distribution with degree of freedome
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Chi-squared distribution**:
    A random variable $X$ is said to have a chi-squared distribution with
    $\nu$ degrees of freedom, written
    \[
        X\sim\chi^2_\nu,
    \]
    if it can be expressed as
    \[
        X = Z_1^2+\cdots+Z_\nu^2,
    \]
    where $Z_1,\ldots,Z_\nu$ are independent standard Normal random variables,
    $Z_i\sim N(0,1)$. Its density is
    \[
        f(x)
        =
        \frac{1}{2^{\nu/2}\Gamma(\nu/2)}
        x^{\nu/2-1}e^{-x/2},
        \qquad x>0.
    \]
    Its mean and variance are
    \[
        \mathbb{E}[X]=\nu,
        \qquad
        \operatorname{Var}(X)=2\nu.
    \]
    For a Normal sample,
    \[
        \frac{(n-1)S^2}{\sigma^2}
        \sim \chi^2_{n-1}.
    \]
    **Student's $t$ distribution:**
    A random variable $T$ has a Student's $t$ distribution with $\nu$ degrees
    of freedom, written
    \[
        T\sim t_\nu,
    \]
    if
    \[
        T
        =
        \frac{Z}{\sqrt{U/\nu}},
    \]
    where
    \[
        Z\sim N(0,1),
        \qquad
        U\sim\chi^2_\nu,
    \]
    and $Z$ and $U$ are independent. Its density is
    \[
        f(t)
        =
        \frac{\Gamma\!\left((\nu+1)/2\right)}
             {\sqrt{\nu\pi}\,
              \Gamma\!\left(\nu/2\right)}
        \left(
            1+\frac{t^2}{\nu}
        \right)^{-(\nu+1)/2},
        \qquad -\infty<t<\infty.
    \]
    The $t$ distribution is symmetric around zero but has heavier tails than the
    standard Normal distribution. As $\nu\to\infty$,
    \[
        t_\nu \longrightarrow N(0,1).
    \]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### learning normal distribution
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    we will used the derivation of learning normal distribution with unknown $\sigma^2$ and $\mu$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    posterior $\propto$ likelihood $\times$ prior
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Introduction
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
    ### Bayesian reminder
    Let $\theta$ denote an unknown parameter and let $y=(y_1,\ldots,y_n)$ denote the observed sample. The prior distribution
    is $p(\theta)$, and the likelihood is $p(y\mid\theta)$.
    From Bayes' rule we get the posterior distribution
    \[
    p(\theta\mid y)=\frac{p(y\mid\theta)p(\theta)}{p(y)}\propto
        p(y\mid\theta)p(\theta).
    \]
    Thus, Bayesian learning may be viewed as
    \[
        \text{prior belief}
        \quad\xrightarrow{\;\text{data / likelihood}\;}\quad
        \text{posterior belief}.
    \]
    For a model with several unknown parameters,
    $\theta=(\theta_1,\ldots,\theta_k)$, Bayes' rule gives a joint posterior
    $p(\theta_1,\ldots,\theta_k\mid y)$. If inference is required only for one
    parameter, the remaining parameters are treated as nuisance parameters and
    are marginalized out. In particular, for
    \[
        Y_i\mid\mu,\sigma^2 \sim N(\mu,\sigma^2),
    \]
    with both $\mu$ and $\sigma^2$ unknown, Bayesian inference begins with the
    joint likelihood $p(y\mid\mu,\sigma^2)$ and a joint prior
    $p(\mu,\sigma^2)$. The resulting joint posterior
    $p(\mu,\sigma^2\mid y)$ contains all information about the two parameters
    after observing the sample. Marginal inference for the mean is obtained by
    integrating out the variance,
    \[
        p(\mu\mid y)
        =
        \int_0^\infty
        p(\mu,\sigma^2\mid y)\,d\sigma^2,
    \]
    while inference for the variance is obtained by integrating out $\mu$.
    For the Normal model, suitable conjugate or Jeffreys priors lead to posterior
    distributions that can be expressed using the inverse--chi-squared and
    Student's $t$ families.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sh
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Theorem**: If $z$ and $w$ are independent random variables having the
    $\operatorname{normal}(0,1^2)$ distribution and the chi-squared distribution
    with $\kappa$ degrees of freedom respectively, then

    \[
    u = \frac{z}{\sqrt{\dfrac{w}{\kappa}}}
    \]

    will have the Student's $t$ distribution with $\kappa$ degrees of freedom.

    In words, a normal random variable with mean $0$ and variance $1$ divided
    by the square root of an independent chi-squared random variable over its
    degrees of freedom will have the Student's $t$ distribution.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    we will not prove it, but rather see a small simulation for it, to convince ourself:
    """)
    return


@app.cell
def _(np, plt, t):
    N = 100_000       # number of repetitions
    nu = 5            # degrees of freedom

    # Independent draws
    Z = np.random.normal(0, 1, N)
    U = np.random.chisquare(nu, N)

    # Theorem 17.1
    T = Z / np.sqrt(U / nu)

    # Compare simulation with theoretical Student-t density
    x = np.linspace(-5, 5, 500)

    plt.hist(Z, bins=100, density=True, alpha=0.5,
             label="z", fill=False)
    plt.hist(U, bins=100, density=True, alpha=0.5,
             label="u", fill=False)
    plt.hist(T, bins=100, density=True, alpha=0.5,
             label="Simulation")
    plt.plot(x, t.pdf(x, df=nu), linewidth=2,
             label=f"Student t, df={nu}")

    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Two Normal populations with unknown means and a common unknown variance
    Suppose that we have two independent random samples
    \[
        y_1=(y_{11},\ldots,y_{1n_1}), \qquad y_2=(y_{21},\ldots,y_{2n_2}),
    \]
    where
    \[
        Y_{1i}\mid\mu_1,\sigma^2 \sim N(\mu_1,\sigma^2), \qquad Y_{2j}\mid\mu_2,\sigma^2 \sim N(\mu_2,\sigma^2).
    \]
    The two populations have different unknown means $\mu_1$ and $\mu_2$, but share the same unknown variance $\sigma^2$.
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
        \sum_{i=1}^{n_1}(y_{1i}-\mu_1)^2=SS_1+n_1(\bar y_1-\mu_1)^2,
    \qquad
        \sum_{j=1}^{n_2}(y_{2j}-\mu_2)^2=SS_2+n_2(\bar y_2-\mu_2)^2
    \]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    the joint likelihood can be written as
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
    Thus, conditional on $\sigma^2$, the likelihood has the form of two independent Normal densities for $\mu_1$ and $\mu_2$, together with an inverse chi-squared component for $\sigma^2$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Finding the Posterior when a Joint Conjugate Prior for $\mu$
    and $\sigma^2$ Is Used
    In Chapter 11 we found that the conjugate prior for $\mu$, the mean of a normal
    observation with known variance $\sigma^2$, is the normal(m; s2) prior distribution.
    In Chapter 15 we found that the conjugate prior for $\sigma^2$, the variance of a
    normal observation with known mean $\mu$, is S times an inverse chi-squared
    with $\kappa$ degrees of freedom. We might think that the joint conjugate prior for
    both parameters $\mu$ and $\sigma^2$ of a normal observation would be the product of
    the independent conjugate priors for each parameter. However, this is not the
    case. The product of independent conjugate priors is a perfectly acceptable
    prior, but it is not jointly conjugate. If we used that prior4, then there is no
    exact formula for the posterior that can be found by simple updating rules.
    Instead, the posterior would have to be found numerically. Later, in Chapter
    20, we will see how we can draw random samples from this posterior using
    the computational Bayesian approach to inference. In this section, we will
    see what form the actual joint conjugate prior takes, and how to do inference
    when we use it.
    The
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why use an inverse chi-squared prior for $\sigma^2$?
    #### Normal likelihood
    Suppose
    \[
    Y_1,\ldots,Y_n\mid\mu,\sigma^2
    \overset{\mathrm{iid}}{\sim}N(\mu,\sigma^2).
    \]
    For one observation,
    \[
    f(y_i\mid\mu,\sigma^2)
    =\frac{1}{\sqrt{2\pi\sigma^2}}
    \exp\left\{-\frac{(y_i-\mu)^2}{2\sigma^2}\right\}.
    \]
    Since the observations are independent, the likelihood is the product of
    their densities:
    \[
    L(\mu,\sigma^2)
    =\prod_{i=1}^n f(y_i\mid\mu,\sigma^2)
    \propto
    (\sigma^2)^{-n/2}
    \exp\left\{
    -\frac{1}{2\sigma^2}
    \sum_{i=1}^n(y_i-\mu)^2
    \right\}.
    \]

    We get that for a normal sample, the likelihood as a function of the unknown variance has
    the kernel
    \[
    L(\sigma^2)
    \propto
    (\sigma^2)^{-n/2}
    \exp\left\{-\frac{SS}{2\sigma^2}\right\}.
    \]
    Recall that if $W\sim\chi^2_\kappa$, then
    \[
    f_W(w)=
    \frac{1}{2^{\kappa/2}\Gamma(\frac{\kappa}{2})}
    w^{\frac{\kappa}{2}-1}e^{-w/2},
    \qquad w>0.
    \]
    Now define
    \[
    W=\frac{S}{\sigma^2},
    \qquad\text{or equivalently}\qquad
    \sigma^2=\frac{S}{W}.
    \]
    We then say that
    \[
    \sigma^2\sim S\times\operatorname{Inv}\chi^2_\kappa,
    \]
    whose density has kernel
    \[
    g(\sigma^2)
    \propto
    (\sigma^2)^{-\kappa/2-1}
    \exp\left\{-\frac{S}{2\sigma^2}\right\}.
    \]
    This has the same functional form in $\sigma^2$ as the normal likelihood.
    Consequently, multiplying the prior by the likelihood produces another
    scaled inverse chi-squared density:
    \[
    \underbrace{
    (\sigma^2)^{-\kappa/2-1}
    e^{-S/(2\sigma^2)}
    }_{\text{prior}}
    \;
    \underbrace{
    (\sigma^2)^{-n/2}
    e^{-SS/(2\sigma^2)}
    }_{\text{likelihood}}
    \propto
    \underbrace{
    (\sigma^2)^{-(\kappa+n)/2-1}
    e^{-(S+SS)/(2\sigma^2)}
    }_{\text{posterior}}.
    \]
    Thus the scaled inverse chi-squared distribution is a conjugate prior for the
    normal variance, with the simple updates
    \[
    \kappa'=\kappa+n,
    \qquad
    S'=S+SS.
    \]
    """)
    return


@app.cell
def _(np, plt, sp):
    # Bayesian learning of sigma^2
    # Single figure: prior -> sample -> posterior

    # True population
    _mu = 0.0
    _sigma2_true = 4.0

    # Sample size
    _n = 25
    # Prior: sigma^2 ~ S * Inv-chi^2_kappa
    _kappa = 2
    _S = 20
    # Number of possible populations drawn
    _K = 20
    _rng = np.random.default_rng(7)

    # Generate data
    _y = _rng.normal(
        loc=_mu,
        scale=np.sqrt(_sigma2_true),
        size=_n)

    _SS = np.sum((_y - _mu) ** 2)

    # Posterior parameters
    _kappa_post = _kappa + _n
    _S_post = _S + _SS

    # Draw possible sigma^2 values
    # Prior draws
    _prior_W = _rng.chisquare(
        df=_kappa,
        size=_K)
    _prior_sigma2 = _S / _prior_W

    # Posterior draws
    _post_W = _rng.chisquare(
        df=_kappa_post,
        size=_K)
    _post_sigma2 = _S_post / _post_W

    # Representative prior and posterior variances
    _sigma2_prior_est = _S / _kappa
    _sigma2_post_est = _S_post / _kappa_post

    # Normal densities
    _x = np.linspace(-8, 8, 1000)
    _true_pdf = sp.stats.norm.pdf(
        _x,
        loc=_mu,
        scale=np.sqrt(_sigma2_true))
    _prior_pdf = sp.stats.norm.pdf(
        _x,
        loc=_mu,
        scale=np.sqrt(_sigma2_prior_est))
    _post_pdf = sp.stats.norm.pdf(
        _x,
        loc=_mu,
        scale=np.sqrt(_sigma2_post_est))


    plt.figure(figsize=(9, 6))

    for _s2 in _prior_sigma2: # Faint Normal populations sampled from PRIOR
        _pdf = sp.stats.norm.pdf(
            _x,
            loc=_mu,
            scale=np.sqrt(_s2))
        plt.plot(
            _x,
            _pdf,
            alpha=0.06)

    # Faint Normal populations sampled from POSTERIOR
    for _s2 in _post_sigma2:
        _pdf = sp.stats.norm.pdf(
            _x,
            loc=_mu,
            scale=np.sqrt(_s2))
        plt.plot(
            _x,
            _pdf,
            alpha=0.18)

    plt.plot(
        _x,
        _prior_pdf,
        linestyle=":",
        linewidth=3,
        label=(
            rf"Prior: "
            rf"$\sigma^2=S/\kappa={_sigma2_prior_est:.2f}$"
        ))

    plt.plot(
        _x,
        _true_pdf,
        linewidth=3,
        label=(
            rf"True population: "
            rf"$N({_mu},{_sigma2_true})$, "
            rf"$\sigma^2={_sigma2_true}$"
        )
    )

    plt.plot(
        _x,
        _post_pdf,
        linestyle="--",
        linewidth=3,
        label=(
            rf"Posterior: "
            rf"$\hat{{\sigma}}_B^2={_sigma2_post_est:.2f}$"
        )
    )

    plt.scatter(
        _y,
        np.zeros_like(_y),
        marker="|",
        s=180,
        label=rf"Observed sample ($n={_n}$)")

    plt.xlabel(r"$y$")
    plt.ylabel("Density")

    plt.title(
        rf"Bayesian learning of $\sigma^2$ "
        rf"($n={_n}$, $K={_K}$ prior/posterior draws)")

    plt.legend()
    plt.grid(alpha=0.25)

    plt.show()
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
def _(mo, sym):
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

    mo.md(
        f"""
    ### Conjugate posterior update
    For $\mu_1$:

    \[
    {sym.latex(mu1_part_sym)}
    =
    {sym.latex(mu1_completed_sym)}
    \]

    where

    \[
    n_1' = {sym.latex(n1_post_sym)}, \qquad 
    m_1' = {sym.latex(m1_post_sym)}.
    \]

    The remainder is 
    $${sym.latex(mu1_remainder_sym)}$$
    and does not depend on $\mu_1$. The symbolic check gives ${sym.latex(mu1_check_sym)}$

    For $\mu_2$:

    \[
    {sym.latex(mu2_part_sym)}
    =
    {sym.latex(mu2_completed_sym)}
    \]

    where

    \[
    n_2' = {sym.latex(n2_post_sym)}, \qquad 
    m_2' = {sym.latex(m2_post_sym)}.
    \]

    Again, the remainder is 
    $${sym.latex(mu2_remainder_sym)}$$ 
    and does not depend on $\mu_2$, the symbolic check gives ${sym.latex(mu2_check_sym)}$. 

    For $\sigma^2$:

    \[
    \kappa' = {sym.latex(kappa_post_sym)},\qquad
    S' = {sym.latex(S_post_sym)}.
    \]

    Therefore the three conjugate components are
    \[
    \mu_1 \mid \sigma^2, y
    \sim N(m_1', \sigma^2/n_1'),
    \]

    \[
    \mu_2 \mid \sigma^2, y
    \sim
    N(m_2', \sigma^2/n_2'),
    \]

    and

    \[
    \sigma^2 \mid y
    \sim S' \cdot \mathrm{{Inv}}\chi^2_{{\kappa'}}.
    \]
    """
    )
    return m1_post_sym, m2_post_sym, n1_post_sym, n2_post_sym


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So, overall, after we did the derivation using sympy, th terms are grouped into three conjugate pairs, of the prior and likelihood terms for:
    * $\mu_1$
    * $\mu_2$
    * $\sigma^2$

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

    md_post_sym = sym.simplify(
        m1_post_sym - m2_post_sym
    )

    vard_post_sym = sym.simplify(
        sigma2_sym / n1_post_sym
        + sigma2_sym / n2_post_sym
    )

    vard_post_factored_sym = sym.factor(
        vard_post_sym
    )

    precision_d_sym = sym.simplify(
        1 / (
            1 / n1_post_sym
            + 1 / n2_post_sym
        )
    )

    precision_d_factored_sym = sym.factor(
        precision_d_sym
    )
    return (
        md_post_sym,
        mud_sym,
        precision_d_factored_sym,
        sigma2_sym,
        vard_post_factored_sym,
    )


@app.cell
def _(md_post_sym, mo, precision_d_factored_sym, sym, vard_post_factored_sym):
    mo.md(
        rf"""
    ### Posterior of the difference of means

    Define

    \[
    \mu_d = \mu_1 - \mu_2.
    \]

    Since $\mu_1$ and $\mu_2$ are conditionally independent Normal random variables, their difference is also Normal.

    The posterior mean is

    \[
    m_d'
    =
    m_1' - m_2'
    =
    {sym.latex(md_post_sym)}.
    \]

    The posterior variance is

    \[
    \operatorname{{Var}}(\mu_d \mid \sigma^2,y)
    =
    {sym.latex(vard_post_factored_sym)}.
    \]

    The inverse variance factor is

    \[
    \frac{{1}}{{\frac{{1}}{{n_1'}}
    +\frac{{1}}{{n_2'}}}}={sym.latex(precision_d_factored_sym)}.
    \]

    Therefore,

    \[
    \mu_d \mid \sigma^2,y
    \sim
    N\left(
    m_d',
    \sigma^2
    \left[
    \frac{{1}}{{n_1'}}+\frac{{1}}{{n_2'}}
    \right]
    \right).
    \]
    """
    )
    return


@app.cell
def _(mud_sym, sigma2_sym, sym):
    md_post_display_sym = sym.symbols("m_d'")
    n1_post_display_sym, n2_post_display_sym = sym.symbols(
        "n_1' n_2'", positive=True
    )
    kappa_post_display_sym = sym.symbols(r"\kappa'", positive=True)
    S_post_display_sym = sym.symbols("S'", positive=True)

    precision_d_display_sym = sym.simplify(
        1 / (
            1 / n1_post_display_sym
            +
            1 / n2_post_display_sym
        )
    )

    normal_d_kernel_sym = (
        sigma2_sym ** (-sym.Rational(1, 2))
        * sym.exp(
            -precision_d_display_sym
            * (mud_sym - md_post_display_sym) ** 2
            / (2 * sigma2_sym)
        )
    )

    sigma2_kernel_sym = (
        sigma2_sym ** (-kappa_post_display_sym / 2 - 1)
        * sym.exp(
            -S_post_display_sym / (2 * sigma2_sym)
        )
    )

    joint_power_sym = sym.simplify(
        -sym.Rational(1, 2)
        - kappa_post_display_sym / 2
        - 1
    )

    joint_exponent_terms_sym = (
        precision_d_display_sym
        * (mud_sym - md_post_display_sym) ** 2
        + S_post_display_sym
    )
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


@app.cell
def _(
    joint_exponent_terms_sym,
    joint_power_sym,
    mo,
    precision_d_display_sym,
    sym,
):
    mo.md(
        rf"""
    ### Joint posterior of \(\mu_d\) and \(\sigma^2\)

    Earlier, we found the parameter for the distribution of $\mu_d = \mu_1 - \mu_2$, and found that its precision is ${sym.latex(precision_d_display_sym)} \frac{{1}}{{\sigma^2}}$.  Hence:
    \[
    p(\mu_d \mid \sigma^2,y)
    \propto
    \frac{{1}}{{(\sigma^2)^{{1/2}}}}
    \exp\left\{{
    -\frac{{1}}{{2\sigma^2}}
    \left(
    {sym.latex(precision_d_display_sym)}
    \right)
    (\mu_d-m_d')^2
    \right\}}.
    \]

    In addition, we found

    \[
    p(\sigma^2 \mid y)
    \propto
    \frac{{1}}{{(\sigma^2)^{{\kappa'/2+1}}}}
    \exp\left\{{
    -\frac{{S'}}{{2\sigma^2}}
    \right\}}.
    \]

    Using the product rule for conditional distributions,

    \[
    p(\mu_d,\sigma^2 \mid y)
    =
    p(\mu_d \mid \sigma^2,y)\,
    p(\sigma^2 \mid y).
    \]

    Thus,

    \[
    p(\mu_d,\sigma^2 \mid y)
    \propto
    \frac{{1}}{{(\sigma^2)^{{1/2}}}}
    \exp\left\{{
    -\frac{{1}}{{2\sigma^2}}
    \left(
    \frac{{n_1'n_2'}}{{n_1'+n_2'}}
    \right)
    (\mu_d-m_d')^2
    \right\}}
    \frac{{1}}{{(\sigma^2)^{{\kappa'/2+1}}}}
    \exp\left\{{
    -\frac{{S'}}{{2\sigma^2}}
    \right\}}.
    \]

    Combining only the powers of \(\sigma^2\): $-\frac{{1}}{{2}}-\frac{{\kappa'}}{{2}}-1={sym.latex(joint_power_sym)}$,
    and summing the exponent values: 
    \[
    {sym.latex(joint_exponent_terms_sym)}.
    \]

    So the joint posterior is: 

    \[
    {{
    p(\mu_d,\sigma^2 \mid y)
    \propto
    \frac{{1}}{{(\sigma^2)^{{(\kappa'+1)/2+1}}}}
    \exp\left\{{
    -\frac{{1}}{{2\sigma^2}}
    \left[
    \frac{{n_1'n_2'}}{{n_1'+n_2'}}
    (\mu_d-m_d')^2
    +
    S'
    \right]
    \right\}}
    }}
    \]
    """
    )
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
    variance_factor_d_sym = sym.simplify(
        1 / n1_post_display_sym
        + 1 / n2_post_display_sym
    )

    # Z ~ N(0,1)
    z_d_sym = sym.simplify(
        (mud_sym - md_post_display_sym)
        / sym.sqrt(
            sigma2_sym * variance_factor_d_sym
        )
    )

    # From sigma^2 ~ S' * Inv-chi^2_kappa',
    # W = S' / sigma^2 ~ chi^2_kappa'
    w_d_sym = sym.simplify(
        S_post_display_sym / sigma2_sym
    )

    # Theorem 17.1:
    # t = Z / sqrt(W / kappa')
    t_from_theorem_sym = sym.simplify(
        z_d_sym
        / sym.sqrt(
            w_d_sym / kappa_post_display_sym
        )
    )

    # Bayesian variance estimate
    sigmaB2_display_sym = sym.simplify(
        S_post_display_sym
        / kappa_post_display_sym
    )

    # Desired final form
    t_final_sym = sym.simplify(
        (mud_sym - md_post_display_sym)
        / sym.sqrt(
            sigmaB2_display_sym
            * variance_factor_d_sym
        )
    )

    # Verify that the theorem gives the desired expression
    t_check_sym = sym.simplify(
        t_from_theorem_sym - t_final_sym
    )
    return t_check_sym, t_from_theorem_sym, w_d_sym, z_d_sym


@app.cell
def _(mo, sym, t_check_sym, t_from_theorem_sym, w_d_sym, z_d_sym):
    mo.md(
        r"""
    ### Marginal posterior of \(\mu_d\) using Theorem 17.1

    We have already obtained the conditional posterior

    \[
    \mu_d\mid\sigma^2,y \sim N\left( m_d', \sigma^2 \left[ \frac{1}{n_1'}+\frac{1}{n_2'} \right] \right).
    \]

    To apply Theorem 17.1, first standardize this Normal random variable. Define

    \[
    Z = \frac{ \mu_d-m_d' }{ \sqrt{ \sigma^2\left(\frac{1}{n_1'}+\frac{1}{n_2'}\right)}}.
    \]

    Using the quantities developed by SymPy,
    \[
    Z = """    + sym.latex(z_d_sym)    + r""",
    \]

    and therefore $Z\sim N(0,1)$. From the posterior distribution of the common variance,

    \[
    \sigma^2\mid y
    \sim
    S'\,\mathrm{Inv}\text{-}\chi^2_{\kappa'},
    \]

    the definition of the scaled inverse chi-squared distribution gives $W=\frac{S'}{\sigma^2}\sim \chi^2_{\kappa'}$, and  SymPy represents this transformation as $W="""+ sym.latex(w_d_sym)+ r"""$. 

    Thus we have the two random variables required by Theorem 17.1:

    \[
    Z\sim N(0,1), \qquad W\sim\chi^2_{\kappa'}.
    \]

    By Theorem 17.1,

    \[
    t
    =
    \frac{Z}{\sqrt{W/\kappa'}}
    \sim
    t_{\kappa'}.
    \]

    Substituting the expressions for \(Z\) and \(W\),

    \[
    t = \frac{ \displaystyle \frac{ \mu_d-m_d' }{ \sqrt{ \sigma^2 \left( \frac{1}{n_1'}+\frac{1}{n_2'} \right) } } }{
    \displaystyle
    \sqrt{ \frac{S'}{\kappa'\sigma^2}}}.
    \]

    and SymPy simplifies this expression to

    \[
    t
    =
    """
        + sym.latex(t_from_theorem_sym)
        + r""".
    \]

    The factor \(\sigma^2\) cancels. Since $\hat{\sigma}_B^2=\frac{S'}{\kappa'}$, the result can be written as
    \[
    \boxed{
    t
    =
    \frac{
    \mu_d-m_d'
    }{
    \hat{\sigma}_B
    \sqrt{
    \frac{1}{n_1'}+\frac{1}{n_2'}
    }
    }
    }
    \]

    and therefore ${t\sim t_{\kappa'}}$. 
    Finally, SymPy verifies that the expression obtained directly from Theorem 17.1 and the final standardized expression are identical:

    \[
    """
        + sym.latex(t_check_sym)
        + r"""
    =0.
    \]

    Thus, although the conditional posterior of \(\mu_d\) given \(\sigma^2\)
    is Normal, accounting for the uncertainty in the unknown common variance
    produces a Student-\(t\) marginal posterior for the difference of means.
    """
    )
    return


if __name__ == "__main__":
    app.run()
