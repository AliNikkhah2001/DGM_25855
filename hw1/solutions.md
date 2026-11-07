# Deep Generative Models — Homework 1

Solutions for Problems 1, 2, 3, and 6, then the extra Problems 4 and 5.
Notation follows the PDF. \(\log\) in entropy and KL is the same base throughout a given identity; where the problem asks for nats, the base is \(e\). Differential entropy is \(h(p)=-\int p\log p\). A Gaussian law is defined by its moment generating function, including singular covariances. Jointly Gaussian random vectors are independent if and only if they are uncorrelated.

Figure 1 (read off the arrowheads, and consistent with the edges \(E\to G\) and \(C\to F\) named in 3.3(b)):

- \(B\to E\), \(B\to G\), \(E\to G\)
- \(A\to D\), \(A\to C\), \(D\to F\), \(C\to F\)

Figure 2 degrees:

| unit | degree |
|---|---|
| \(x_1,\lambda_1\) | \(m^0=2\) |
| \(x_2,\lambda_2\) | \(m^0=4\) |
| \(x_3,\lambda_3\) | \(m^0=1\) |
| \(x_4,\lambda_4\) | \(m^0=3\) |
| \(h_1,h_4,h_6\) | \(m^1=1\) |
| \(h_2,h_3,h_5\) | \(m^1=3\) |

The autoregressive order is increasing degree: \(x_3,x_1,x_4,x_2\).

Two phrases in the PDF point at themselves (“apply it to (b)” in 1.4, “compare with (b)” in 3.3, “change (b)” in 5.3). Each is answered by the identity that makes the surrounding sentence true, and the numerical check against the other part is recorded.

---

## Problem 1

### 1.1. Law of \(y=Ax+b\)

Let \(x\sim\mathcal N(\mu,\Sigma)\). For every \(s\),

$$
\mathbb E[e^{s^\top y}]
=\mathbb E[e^{s^\top(Ax+b)}]
=e^{s^\top b}\,\mathbb E[e^{(A^\top s)^\top x}]
=e^{s^\top b}\exp\big((A^\top s)^\top\mu+\tfrac12(A^\top s)^\top\Sigma A^\top s\big)
=\exp\big(s^\top(A\mu+b)+\tfrac12 s^\top(A\Sigma A^\top)s\big).
$$

Hence \(y\sim\mathcal N(A\mu+b,\,A\Sigma A^\top)\), possibly degenerate.

### 1.1(a). Scalar combinations

Let \(a,b\in\mathbb R\) and let \((x,y)\) be jointly Gaussian, with cross-covariance \(C=\mathrm{Cov}(x,y)\). Then

$$
z=ax+by\sim\mathcal N\big(a\mu_x+b\mu_y,\; a^2\Sigma_x+b^2\Sigma_y+ab\,C+ab\,C^\top\big),
$$

because \(\mathrm{Cov}(ax,by)=ab\,C\).

- If \(x\perp y\), then \(C=0\). If \(\Sigma_x,\Sigma_y\succ 0\) and \((a,b)\ne(0,0)\), the covariance \(a^2\Sigma_x+b^2\Sigma_y\) is positive definite, so not singular.
- If \(y=x\), then \(z=(a+b)x\) and the covariance is \((a+b)^2\Sigma_x\). It is singular whenever \(a+b=0\) (then \(z\) is a.s. constant) and also whenever \(\Sigma_x\) itself is singular.

In dimension \(d=1\), with \(\sigma_x,\sigma_y>0\) and correlation \(\rho=C/(\sigma_x\sigma_y)\),

$$
\mathrm{Var}(z)=a^2\sigma_x^2+b^2\sigma_y^2+2ab\rho\,\sigma_x\sigma_y.
$$

The right-hand side is a nonnegative quadratic form in \((a,b)\) for every \((a,b)\), so its matrix has determinant \(\sigma_x^2\sigma_y^2(1-\rho^2)\ge 0\). Thus \(|\rho|\le 1\). Equality \(|\rho|=1\) holds if and only if that matrix is singular, if and only if \(y-\mu_y\) is an affine function of \(x-\mu_x\) almost surely (\(\rho=+1\) for positive slope, \(\rho=-1\) for negative slope). This is Cauchy–Schwarz on the centered variables.

### 1.1(b). Rank-deficient image

\(x\sim\mathcal N(0,I_d)\), \(A\in\mathbb R^{m\times d}\), \(m>d\), full column rank. Then \(\mathrm{Cov}(Ax)=AA^\top\) has rank \(d<m\), so the law of \(Ax\) is supported on the column space of \(A\). It is not absolutely continuous on \(\mathbb R^m\) and has no density there.

Thin QR: \(A=QR\) with \(Q\in\mathbb R^{m\times d}\), \(Q^\top Q=I_d\), and \(R\in\mathbb R^{d\times d}\) invertible. Then \(Rx\sim\mathcal N(0,RR^\top)\) with \(RR^\top\succ 0\), and \(Ax=Qz\) with that nondegenerate \(z\in\mathbb R^d\).

Two computations of the second moment:

$$
\mathbb E[{\Vert}Ax\Vert^2]=\mathbb E[x^\top A^\top Ax]=\mathrm{tr}(A^\top A)={\Vert}A\Vert_F^2,
$$

and, since \(Q\) has orthonormal columns, \({\Vert}Ax\Vert^2={\Vert}z\Vert^2\) and \(\mathbb E[{\Vert}z\Vert^2]=\mathrm{tr}(RR^\top)={\Vert}R\Vert_F^2={\Vert}A\Vert_F^2\).

### 1.2. Product of Gaussian densities

For \(\Sigma\succ 0\),

$$
\mathcal N(x\mid\mu,\Sigma)=(2\pi)^{-d/2}(\det\Sigma)^{-1/2}\exp\big(-\tfrac12(x-\mu)^\top\Sigma^{-1}(x-\mu)\big).
$$

The sum of the two precision quadratics has precision \(A^{-1}+B^{-1}\) and linear term \(A^{-1}a+B^{-1}b\). Therefore

$$
C=(A^{-1}+B^{-1})^{-1},\qquad c=C(A^{-1}a+B^{-1}b).
$$

The parallel-sum identities \(A^{-1}+B^{-1}=B^{-1}(A+B)A^{-1}\) give the forms that use only \((A+B)^{-1}\):

$$
C=A(A+B)^{-1}B=B(A+B)^{-1}A,
$$
$$
c=B(A+B)^{-1}a+A(A+B)^{-1}b.
$$

They agree with the precision form because \(CA^{-1}a=B(A+B)^{-1}a\) and \(CB^{-1}b=A(A+B)^{-1}b\).

Integrating the product in \(x\) produces \(Z\), since \(\mathcal N(x\mid c,C)\) integrates to 1. If \(U\sim\mathcal N(a,A)\) and \(V\sim\mathcal N(b,B)\) are independent, that integral is the density of \(U-V\) at \(0\), and \(U-V\sim\mathcal N(a-b,A+B)\). Hence

$$
Z=\mathcal N(a\mid b,\,A+B)=\mathcal N(b\mid a,\,A+B).
$$

(The identity \(\mathcal N(x\mid a,A)\mathcal N(x\mid b,B)=\mathcal N(a\mid b,A+B)\,\mathcal N(x\mid c,C)\) was also checked numerically in one dimension: the ratio of the two sides is constant and equal to this \(Z\).)

### 1.2(a). Gaussian posterior, marginal, MLE, MAP

Likelihood in \(\theta\): \(\bar x\mid\theta\sim\mathcal N(\theta,s^2/n)\), and \(\mathcal N(\bar x\mid\theta,s^2/n)=\mathcal N(\theta\mid\bar x,s^2/n)\). Combine with the prior \(\mathcal N(m_0,s_0^2)\) by 1.2:

$$
\frac1{s_n^2}=\frac1{s_0^2}+\frac n{s^2},\qquad
s_n^2=\frac{s_0^2 s^2}{s^2+n s_0^2},
$$
$$
m_n=s_n^2\Big(\frac{m_0}{s_0^2}+\frac{n\bar x}{s^2}\Big)
=\frac{s^2}{s^2+n s_0^2}\,m_0+\frac{n s_0^2}{s^2+n s_0^2}\,\bar x.
$$

So \(\theta\mid x_{1:n}\sim\mathcal N(m_n,s_n^2)\). The posterior mean is the stated mixture of the prior mean and the MLE.

Marginal of the sample: \(\mathbb E[x_i]=m_0\), \(\mathrm{Var}(x_i)=s^2+s_0^2\), and \(\mathrm{Cov}(x_i,x_j)=s_0^2\) for \(i\ne j\). Thus

$$
(x_1,\ldots,x_n)\sim\mathcal N\big(m_0\mathbf 1,\; s^2 I+s_0^2\mathbf 1\mathbf 1^\top\big).
$$

The likelihood \(\prod_i\mathcal N(x_i\mid\theta,s^2)\) is maximized at the MLE \(\hat\theta_{\mathrm{MLE}}=\bar x\). The MAP is the posterior mode \(m_n\), since the posterior is Gaussian.

- As \(n\to\infty\), the weight on \(m_0\) tends to 0, the weight on \(\bar x\) tends to 1, and \(s_n^2\sim s^2/n\to 0\). The posterior collapses at the MLE, and the MAP converges to the MLE.
- As \(s_0\to\infty\), \(s_n^2\to s^2/n\) and \(m_n\to\bar x\). A flat prior makes the posterior \(\mathcal N(\bar x,s^2/n)\) and the MAP equal to the MLE.

### 1.3. Conditional law, without block inversion

Set \(K=\Sigma_{yx}\Sigma_{xx}^{-1}\), which exists because \(\Sigma_{xx}\succ 0\). The residual

$$
r=y-\mu_y-K(x-\mu_x)
$$

is uncorrelated with \(x\): \(\mathrm{Cov}(r,x)=\Sigma_{yx}-K\Sigma_{xx}=0\). The pair \((r,x)\) is jointly Gaussian, so \(r\perp x\). Therefore

$$
y\mid x=x_0\sim\mathcal N\big(\mu_y+K(x_0-\mu_x),\;\mathrm{Cov}(r)\big),
$$
$$
\mathrm{Cov}(r)=\Sigma_{yy}-\Sigma_{yx}\Sigma_{xx}^{-1}\Sigma_{xy}.
$$

The conditional covariance does not depend on \(x_0\). The subtracted term is positive semidefinite, so \(\mathrm{Cov}(y\mid x)\preceq\Sigma_{yy}\). Equality holds if and only if \(\Sigma_{yx}=0\). For jointly Gaussian vectors that is exactly independence of \(x\) and \(y\).

### 1.3(a). Linear-Gaussian model and the two posterior forms

Write \(y=Ax+b+\varepsilon\) with \(\varepsilon\sim\mathcal N(0,S)\) independent of \(x\). Then \((x,\varepsilon)\) is jointly Gaussian, and so is \((x,y)\):

$$
\begin{pmatrix}x\\ y\end{pmatrix}
\sim\mathcal N\Bigg(
\begin{pmatrix}\mu\\ A\mu+b\end{pmatrix},
\begin{pmatrix}\Sigma & \Sigma A^\top\\ A\Sigma & S+A\Sigma A^\top\end{pmatrix}
\Bigg).
$$

The marginal is \(y\sim\mathcal N(A\mu+b,\,S+A\Sigma A^\top)\).

Conditioning \(x\) on \(y\) by 1.3 gives the form that inverts only \(R:=S+A\Sigma A^\top\):

$$
\mathrm{Cov}(x\mid y)=\Sigma-\Sigma A^\top R^{-1}A\Sigma,
$$
$$
\mathbb E[x\mid y]=\mu+\Sigma A^\top R^{-1}\big(y-(A\mu+b)\big).
$$

The information form comes from the prior precision plus the likelihood precision:

$$
\Sigma_{x\mid y}=(\Sigma^{-1}+A^\top S^{-1}A)^{-1},
$$
$$
\mathbb E[x\mid y]=\Sigma_{x\mid y}\big(\Sigma^{-1}\mu+A^\top S^{-1}(y-b)\big).
$$

Woodbury’s identity is exactly

$$
(\Sigma^{-1}+A^\top S^{-1}A)^{-1}=\Sigma-\Sigma A^\top R^{-1}A\Sigma,
$$

so the covariances agree. For the means, \(\Sigma_{x\mid y}\Sigma^{-1}=I-\Sigma_{x\mid y}A^\top S^{-1}A\), hence the information-form mean equals

$$
\mu+\Sigma_{x\mid y}A^\top S^{-1}\big(y-(A\mu+b)\big).
$$

The remaining identity \(\Sigma_{x\mid y}A^\top S^{-1}=\Sigma A^\top R^{-1}\) follows by multiplying on the left by \(\Sigma^{-1}+A^\top S^{-1}A\):

\begin{align*}
(\Sigma^{-1}+A^\top S^{-1}A)\Sigma A^\top R^{-1}
&=A^\top(I+S^{-1}A\Sigma A^\top)R^{-1}
=A^\top S^{-1}(S+A\Sigma A^\top)R^{-1}
=A^\top S^{-1}.
\end{align*}

The matrix in front of the innovation \(y-(A\mu+b)\) is \(\mathrm{Cov}(x,y)\mathrm{Cov}(y)^{-1}\). It is the linear regression coefficient of \(x\) on \(y\): it converts the residual of the observation into the correction of the prior mean. In the information form the same map is \(\Sigma_{x\mid y}A^\top S^{-1}\).

### 1.4. Mixture moments

Let \(K\) be a discrete component with \(\mathbb P(K=k)=\pi_k\), and \(x\mid K=k\sim\mathcal N(\mu_k,\Sigma_k)\). Then

$$
\mathbb E[x]=\sum_k\pi_k\mu_k=:\mu,
$$
$$
\mathbb E[xx^\top]=\sum_k\pi_k(\Sigma_k+\mu_k\mu_k^\top).
$$

Therefore

$$
\mathrm{Cov}(x)
=\underbrace{\sum_k\pi_k\Sigma_k}_{\text{within}}
+\underbrace{\sum_k\pi_k(\mu_k-\mu)(\mu_k-\mu)^\top}_{\text{between}}.
$$

### 1.4(a). Two-component mixture on the line

Mean: \(\tfrac12(-a)+\tfrac12 a=0\).

Variance: within contribution \(1\), between contribution \(a^2\), so \(\mathrm{Var}(x)=1+a^2\).

For \(Z\sim\mathcal N(a,1)\), \(\mathbb E[Z^4]=a^4+6a^2+3\). The same value holds at \(-a\), so \(\mathbb E_p[x^4]=a^4+6a^2+3\). Excess kurtosis:

$$
\frac{\mathbb E[x^4]}{\mathrm{Var}(x)^2}-3
=\frac{a^4+6a^2+3}{(1+a^2)^2}-3
=\frac{-2a^4}{(1+a^2)^2}<0
\quad(a>0).
$$

A nondegenerate Gaussian has excess kurtosis \(0\), so \(p\) is not Gaussian. Equivalently, its mgf is \(\cosh(at)\,e^{t^2/2}\), which is not the mgf of a Gaussian.

Modes. The unnormalized density \(h(x)=e^{-(x+a)^2/2}+e^{-(x-a)^2/2}\) has \(h'(x)<0\) for all \(x>a\) and \(h'(x)>0\) for all \(x<-a\), so every critical point lies in \((-a,a)\). There,

$$
h'(x)=0\iff \frac{a+x}{a-x}=e^{2ax}.
$$

The function \(k(x)=\log(a+x)-\log(a-x)-2ax\) satisfies \(k(0)=0\) and

$$
k'(x)=2a\frac{x^2-(a^2-1)}{a^2-x^2}.
$$

Also \(h''(0)=2(a^2-1)e^{-a^2/2}\).

- If \(a<1\), then \(k'(x)>0\) on \((0,a)\), so \(x=0\) is the only root. And \(h''(0)<0\), so it is a maximum. One mode.
- If \(a=1\), still only one root. Near \(0\), \(h''(x)=-2x^2+O(x^4)\) after expanding the two Gaussian factors, so \(x=0\) is a strict local maximum. One mode.
- If \(a>1\), then \(k'\) changes from negative to positive at \(\sqrt{a^2-1}\), and \(k(a^-)=+\infty\). So \(k\) has exactly one positive root. Together with \(x=0\) and the negative root, there are three critical points. Now \(h''(0)>0\), so \(x=0\) is a local minimum and the other two critical points are local maxima. Two modes.

Thus \(p\) has two modes if and only if \(a>1\). This threshold was checked on a grid: one peak for \(a\in\{0.5,1\}\), two peaks for \(a\in\{1.01,2\}\).

### 1.4(b). Gaussian KL, the Pythagorean identity, maximum entropy

**(i)** For \(p_i=\mathcal N(\mu_i,\Sigma_i)\), \(\Sigma_i\succ 0\),

$$
\mathrm{KL}(p_0{\Vert}p_1)
=\frac12\Big[
\mathrm{tr}(\Sigma_1^{-1}\Sigma_0)
+(\mu_0-\mu_1)^\top\Sigma_1^{-1}(\mu_0-\mu_1)
-d
+\log\frac{\det\Sigma_1}{\det\Sigma_0}
\Big].
$$

Proof: \(-\mathbb E_{p_0}\log p_1\) contributes \(\tfrac12\log\det\Sigma_1+\tfrac12\mathbb E[(x-\mu_1)^\top\Sigma_1^{-1}(x-\mu_1)]\), and

$$
\mathbb E_{p_0}[(x-\mu_1)^\top\Sigma_1^{-1}(x-\mu_1)]
=\mathrm{tr}(\Sigma_1^{-1}\Sigma_0)+(\mu_0-\mu_1)^\top\Sigma_1^{-1}(\mu_0-\mu_1).
$$

Subtracting \(h(p_0)=\tfrac d2\log(2\pi e)+\tfrac12\log\det\Sigma_0\), and cancelling \(\tfrac d2\log(2\pi)\), leaves the display above. The same formula is the one in the [Book of Statistical Proofs](https://statproofbook.github.io/P/mvn-kl.html).

**(ii)** Let \(g=\mathcal N(\mu_p,\Sigma_p)\) and let \(q\) be any Gaussian. The function \(\log(g/q)\) is a polynomial of degree at most 2. Any such polynomial has the same expectation under \(p\) and under \(g\), because \(p\) and \(g\) share their moments of order \(0,1,2\). Therefore

$$
\mathrm{KL}(p{\Vert}q)-\mathrm{KL}(p{\Vert}g)
=\mathbb E_p\log\frac gq
=\mathbb E_g\log\frac gq
=\mathrm{KL}(g{\Vert}q),
$$

which is the identity

$$
\mathrm{KL}(p{\Vert}q)=\mathrm{KL}(p{\Vert}g)+\mathrm{KL}(g{\Vert}q).
$$

Since \(\mathrm{KL}(g{\Vert}q)\ge 0\), with equality if and only if \(q=g\), the unique minimizer of \(\mathrm{KL}(p{\Vert}q)\) over Gaussians \(q\) is \(q^\star=g=\mathcal N(\mu_p,\Sigma_p)\). The minimum value is

$$
\mathrm{KL}(p{\Vert}g)=h(g)-h(p),
$$

because \(\mathbb E_p\log g=\mathbb E_g\log g=-h(g)\). Applying (i) to the pair \((g,q)\) writes the excess \(\mathrm{KL}(g{\Vert}q)\) in closed form; that is the content of “apply it to (b)”.

In particular \(h(p)\le h(g)\), with equality if and only if \(p=g\). Among densities with a fixed mean and a fixed positive definite covariance, the Gaussian has the largest differential entropy.

Applied to the mixture in (a): the moment-matched Gaussian is \(\mathcal N(0,1+a^2)\), and the minimal KL to a Gaussian is \(h(\mathcal N(0,1+a^2))-h(p)>0\).

### 1.5. Autoregressive factorization of a nondegenerate Gaussian

Partition \(\Sigma\) so that the last coordinate is \(x_d\). By 1.3, with \(x_{<d}\) in the role of the conditioning variable,

$$
w_d=\Sigma_{<d}^{-1}\Sigma_{<d,d},\qquad
\sigma_d^2=\Sigma_{dd}-\Sigma_{d,<d}\Sigma_{<d}^{-1}\Sigma_{<d,d},
$$

and \(x_d\mid x_{<d}\sim\mathcal N(w_d^\top x_{<d},\sigma_d^2)\), with the \(d=1\) factor equal to \(\mathcal N(0,\Sigma_{11})\). The conditional law of a Gaussian is unique, so \((w_d,\sigma_d^2)\) is unique, and so is the factorization \(p(x)=\prod_d\mathcal N(x_d\mid w_d^\top x_{<d},\sigma_d^2)\).

The conditional variance is the Schur complement of the leading principal submatrix \(\Sigma_{1:d-1}\). The determinant factorization of a Schur complement gives \(\sigma_d^2=\det\Sigma_{1:d}/\det\Sigma_{1:d-1}\), with the convention \(\det\Sigma_{1:0}=1\). Therefore

$$
\det\Sigma=\prod_{d=1}^D\sigma_d^2.
$$

A strictly lower-triangular \(W\) has \(D(D-1)/2\) entries and there are \(D\) positive conditional variances, hence \(D(D+1)/2\) free parameters, the same number as a positive definite covariance. Every \(\Sigma\succ 0\) arises exactly once.

Conversely, \(x_d=w_d^\top x_{<d}+\varepsilon_d\) with independent \(\varepsilon_d\sim\mathcal N(0,\sigma_d^2)\) is \(x=(I-W)^{-1}\varepsilon\). The matrix \(I-W\) is unit lower triangular, hence invertible, and \(\varepsilon\sim\mathcal N(0,\mathrm{diag}(\sigma_d^2))\) is nondegenerate. Thus

$$
x\sim\mathcal N\big(0,\;(I-W)^{-1}\Delta(I-W)^{-\top}\big),\qquad\Delta=\mathrm{diag}(\sigma_d^2)\succ 0.
$$

### 1.5(a). AR(1) covariance

The matrix \(\Sigma_{ij}=\dfrac{\sigma^2}{1-\rho^2}\rho^{|i-j|}\) is the covariance of the stationary Gaussian AR(1)

$$
x_1\sim\mathcal N\Big(0,\frac{\sigma^2}{1-\rho^2}\Big),\qquad
x_d=\rho x_{d-1}+\varepsilon_d,\quad\varepsilon_d\sim\mathcal N(0,\sigma^2)\ \text{i.i.d.}
$$

Read the autoregression off \(\Sigma\) itself. For \(d=1\), \(\sigma_1^2=\sigma^2/(1-\rho^2)\). For \(d\ge 2\), the candidate \(w_d=\rho\,e_{d-1}\) satisfies the normal equations: the column of \(\Sigma_{1:d-1}\) indexed by \(d-1\), multiplied by \(\rho\), equals \((\Sigma_{1d},\ldots,\Sigma_{d-1,d})^\top\), because

$$
\rho\cdot\frac{\sigma^2}{1-\rho^2}\rho^{|(j)-(d-1)|}
=\frac{\sigma^2}{1-\rho^2}\rho^{|d-j|}.
$$

The conditional variance collapses:

$$
\sigma_d^2=\frac{\sigma^2}{1-\rho^2}-\rho\cdot\frac{\sigma^2}{1-\rho^2}\rho=\sigma^2.
$$

So \(w_d\) has a single nonzero entry, equal to \(\rho\), in the coordinate of \(x_{d-1}\). This was checked on a \(6\times 6\) Toeplitz matrix with \((\rho,\sigma^2)=(0.5,1.3)\): the regression on the first five coordinates was \((0,0,0,0,0.5)\) and the conditional variance was \(1.3\).

### 1.5(b). Hub and leaves

\(\mathrm{Var}(x_1)=1\), \(\mathrm{Cov}(x_1,x_j)=\beta\), \(\mathrm{Var}(x_j)=\beta^2+s^2\), and \(\mathrm{Cov}(x_j,x_k)=\beta^2\) for distinct leaves. In blocks,

$$
\Sigma=\begin{pmatrix}1 & \beta\mathbf 1^\top\\ \beta\mathbf 1 & s^2 I+\beta^2\mathbf 1\mathbf 1^\top\end{pmatrix}.
$$

**Order \((1,\ldots,D)\).** Here \(\sigma_1^2=1\). For each leaf, \(x_j=\beta x_1+\varepsilon_j\) with \(\varepsilon_j\) independent of \((x_1,x_2,\ldots,x_{j-1})\), so

$$
w_j=\beta\,e_1,\qquad\sigma_j^2=s^2.
$$

There are \(D-1\) nonzero weights, and \(\prod_d\sigma_d^2=(s^2)^{D-1}\).

**Order \((D,\ldots,1)\).** Position \(m+1\) (\(0\le m\le D-2\)) is a leaf conditioned on \(m\) other leaves. The leaf block of size \(m\) has covariance \(s^2I+\beta^2\mathbf 1\mathbf 1^\top\), and the cross-covariance with a new leaf is \(\beta^2\mathbf 1\). The constant vector

$$
c=\frac{\beta^2}{s^2+m\beta^2}
$$

solves the normal equations, so every one of the \(m\) previous leaves receives weight \(c\), and

$$
\sigma^2
=\beta^2+s^2-\frac{m\beta^4}{s^2+m\beta^2}
=\frac{s^2\big(s^2+(m+1)\beta^2\big)}{s^2+m\beta^2}.
$$

For \(m=0\) this is the marginal variance \(\beta^2+s^2\).

The hub is last, conditioned on all \(D-1\) leaves. The same Sherman–Morrison inverse gives the constant weight

$$
\frac{\beta}{s^2+(D-1)\beta^2}
$$

on every leaf, and conditional variance

$$
\sigma_{\mathrm{hub}}^2=\frac{s^2}{s^2+(D-1)\beta^2}.
$$

(The same variance is the Gaussian posterior variance of \(x_1\) after \(D-1\) observations of precision \(\beta^2/s^2\).)

Nonzero weights in this order: \(\sum_{m=1}^{D-2}m+(D-1)=D(D-1)/2\). Every one of those weights is nonzero because \(\beta\ne 0\).

The product of conditional variances equals \(\det\Sigma\) in every order. Multiplying the leaf formulas telescopes to \((s^2)^{D-2}(s^2+(D-1)\beta^2)\), and the hub factor cancels the second piece, leaving \((s^2)^{D-1}\). Same product as in the hub-first order. A direct determinant check for \((\beta,s^2,D)=(0.7,0.4,6)\) gave \(\det\Sigma=(s^2)^{5}\).

**Sparsity cannot be transplanted.** For a fixed ordering the pair \((W,\Delta)\) is unique (1.5). In the leaf-first ordering, the row in position \(m+1\) has \(m\) nonzero entries. For \(D\ge 3\) and \(m\ge 2\) that is already more than one, whereas every row of the hub-first factorization has at most one nonzero weight, and the whole matrix has only \(D-1\) nonzeros rather than \(D(D-1)/2\). A model restricted to the hub-first sparsity therefore cannot reproduce this law in the reversed order. (For \(D=2\) both orders have a single nonzero weight, so the obstruction starts at \(D\ge 3\).)

### 1.5(c). Heteroscedastic conditional

No. Under a jointly nondegenerate Gaussian, 1.3 says that \(\mathrm{Cov}(x_2\mid x_1)\) does not depend on the value of \(x_1\). Here that conditional variance is \(e^{ux_1}\), which is nonconstant because \(u\ne 0\).

(As a second witness: \(\mathbb E[x_2^4]=3e^{2u^2}\) and \((\mathbb E[x_2^2])^2=e^{u^2}\), so the excess kurtosis is \(3(e^{u^2}-1)>0\).)

---

## Problem 2

### 2.1. Monte Carlo

If \(\mathbb E_p|f|<\infty\), then \(\mathbb E[f_N]=\mathbb E_p[f]\), so \(f_N\) is unbiased. If \(\mathrm{Var}_p(f)=\sigma_f^2<\infty\), independence gives \(\mathrm{Var}(f_N)=\sigma_f^2/N\).

Chebyshev: for every \(\varepsilon>0\),

$$
\mathbb P\big(|f_N-\mathbb E_p f|\ge\varepsilon\big)\le\frac{\sigma_f^2}{N\varepsilon^2}.
$$

The right-hand side tends to \(0\), which is the weak law, with rate \(O(1/N)\) in probability deviation and \(O(N^{-1/2})\) in root-mean-square error. The sample size needed for deviation \(\varepsilon\) and probability \(\delta\) is \(N\ge\sigma_f^2/(\delta\varepsilon^2)\).

### 2.1(a). Scaled chi-squared

Let \(Z=(X-1)/2\sim\mathcal N(0,1)\). Then \(f(X)=4Z^2\), so \(\mathbb E[f]=4\) and \(\mathrm{Var}(f)=16\mathrm{Var}(Z^2)=16(3-1)=32\).

Each \((X_n-1)^2/4\sim\chi^2_1\), hence \(f_N=(4/N)\chi^2_N\).

The event \(|f_N-4|\ge 1/2\) is \(|\chi^2_N-N|\ge N/8\), i.e. a relative deviation \(t=1/8\).

**(i) Chebyshev.** \(\mathrm{Var}(f_N)=32/N\), so the probability is at most \(32/(N\cdot 1/4)=128/N\). The bound is at most \(0.05\) once \(N\ge 128/0.05=2560\). This \(N\) is guaranteed.

**(ii) CLT.** Use \(\sqrt{32/N}\) as the standard deviation and the normal quantile \(1.96\):

$$
\frac{1/2}{\sqrt{32/N}}\ge 1.96
\iff N\ge 32\cdot(3.92)^2=491.72\ldots
$$

So \(N\ge 492\). This is an approximation, not a finite-sample guarantee. The chi-squared law is skewed, so the normal tail is not an upper bound.

**(iii) Chernoff.** The mgf of \(\chi^2_N\) is \((1-2\lambda)^{-N/2}\) for \(\lambda<1/2\). Optimizing the Markov bound on each tail produces, for \(t\in(0,1)\),

\begin{align*}
\mathbb P(\chi^2_N\ge N(1+t))&\le\big((1+t)e^{-t}\big)^{N/2}=e^{-NI_+(t)},\\
\mathbb P(\chi^2_N\le N(1-t))&\le\big((1-t)e^{t}\big)^{N/2}=e^{-NI_-(t)},
\end{align*}

with exponents

$$
I_+(t)=\tfrac12\big(t-\log(1+t)\big),\qquad
I_-(t)=\tfrac12\big(-\log(1-t)-t\big).
$$

At \(t=1/8\), \(I_+\approx 0.0036085\) and \(I_-\approx 0.0042657\). The sum of the two tails drops to \(0.05\) or below at \(N=950\) (the bound equals \(0.04983\) at \(950\) and \(0.05002\) at \(949\)). Nearest \(50\) is \(950\). This \(N\) is guaranteed, because it comes from the mgf.

Chebyshev and Chernoff are theorems about this finite \(N\); the CLT is only an asymptotic shape. Chebyshev uses nothing beyond the variance, so it is loose (\(2560\)). Chernoff uses the whole mgf and the correct sub-exponential rate, so it is tighter (\(950\)) but still a bound, not the true quantile. The CLT suggests about \(492\), and it is the most optimistic of the three precisely because it is not guaranteed.

### 2.1(b). Importance sampling

$$
\mathbb E_q\Big[f(x)\frac{p(x)}{q(x)}\Big]=\int x^2 p(x)\,dx=1
$$

whenever \(s>0\), because \(q>0\) everywhere and \(x^2p\) is integrable. So \(\hat I_N\) is unbiased for every \(s>0\), and \(\mathrm{Var}(\hat I_N)=N^{-1}\mathrm{Var}_q(fp/q)\).

The second moment is finite if and only if \(s>1/\sqrt 2\). Indeed

$$
\frac{p(x)^2}{q(x)}=s(2\pi)^{-1/2}\exp\big(-x^2(1-\tfrac1{2s^2})\big),
$$

and \(\int x^4 p^2/q\) converges precisely when the quadratic rate \(1-1/(2s^2)\) is positive. Evaluating the Gaussian moment \(\int x^4 e^{-\beta x^2}\,dx=(3/4)\sqrt{\pi/\beta^5}\) with \(\beta=(2s^2-1)/(2s^2)\) gives

$$
\mathbb E_q\Big[\Big(x^2\frac pq\Big)^2\Big]=\frac{3s^6}{(2s^2-1)^{5/2}},\qquad
\mathrm{Var}_q\Big(x^2\frac pq\Big)=\frac{3s^6}{(2s^2-1)^{5/2}}-1.
$$

(The second-moment formula agrees with a grid quadrature at \(s=\sqrt 3\) to \(10^{-12}\).)

Minimize \(v(u)=3u^3/(2u-1)^{5/2}\) for \(u=s^2>1/2\). The logarithmic derivative \(3/u-5/(2u-1)\) vanishes at \(u=3\), and \(v\to\infty\) at both endpoints \(u\downarrow 1/2\) and \(u\to\infty\), so this critical point is the minimum. Thus the best Gaussian scale is \(s=\sqrt 3\), and the minimal one-sample variance is

$$
\frac{81}{25\sqrt 5}-1=\frac{81\sqrt 5-125}{125}\approx 0.44897.
$$

Ordinary Monte Carlo (\(s=1\)) has variance \(\mathrm{Var}(x^2)=2\).

Over all proposals, \(f\ge 0\), so the zero-variance proposal is \(q^\star(x)\propto f(x)p(x)=x^2 p(x)\). It is already normalized because \(\mathbb E_p[x^2]=1\):

$$
q^\star(x)=x^2\cdot(2\pi)^{-1/2}e^{-x^2/2}.
$$

Then \(f p/q^\star=1\) for \(x\ne 0\), so the importance-sampling estimator has variance \(0\). (Using it requires the unknown integral as a normalizing constant; that is why the optimum is not a practical proposal.)

### 2.2. Perplexity

\(\widehat{\mathrm{PPL}}_N=\exp(\bar\ell_N)\) with \(\bar\ell_N=N^{-1}\sum_{n\le N}\ell(x_n)\). Jensen’s inequality for the convex map \(\exp\) gives

$$
\mathbb E[\widehat{\mathrm{PPL}}_N]\ge\exp(\mathbb E[\bar\ell_N])=\exp(\mathbb E[\ell])=\mathrm{PPL},
$$

for every finite \(N\). Equality holds if and only if \(\bar\ell_N\) is almost surely constant, i.e. \(\ell(x)\) is \(p_{\mathrm{data}}\)-a.s. constant. In particular \(\widehat{\mathrm{PPL}}_N\) is not unbiased unless \(\ell\) is a.s. constant.

If \(\mathbb E|\ell|<\infty\), the weak law gives \(\bar\ell_N\to\mathbb E[\ell]\) in probability, and continuity of \(\exp\) gives \(\widehat{\mathrm{PPL}}_N\to\mathrm{PPL}\) in probability. The estimator is consistent. The strong law upgrades this to almost-sure convergence.

### 2.2(a). Bias expansion

Assume \(\mathbb E|\ell|^3<\infty\). Write \(\delta_N=\bar\ell_N-\mathbb E[\ell]\), so \(\mathbb E\delta_N=0\), \(\mathbb E\delta_N^2=\mathrm{Var}(\ell)/N\), and \(\mathbb E\delta_N^3=O(1/N^2)\). Then

$$
\mathbb E[e^{\delta_N}]=1+\tfrac12\mathbb E[\delta_N^2]+O(1/N^2)=1+\frac{\mathrm{Var}(\ell)}{2N}+o(N^{-1}).
$$

Hence \(c=\mathrm{Var}(\ell)/2\).

### 2.2(b). Gaussian closed form

$$
\ell(x)=\tfrac12\log(2\pi\sigma^2)+\frac{x^2}{2\sigma^2},\qquad
\mathbb E[\ell]=\tfrac12\log(2\pi\sigma^2)+\tfrac r2,
$$

so \(\mathrm{PPL}=\sqrt{2\pi\sigma^2}\,e^{r/2}\). With \(\sum_n x_n^2=s^2\chi^2_N\),

$$
\widehat{\mathrm{PPL}}_N=\sqrt{2\pi\sigma^2}\,\exp\big(\tfrac{r}{2N}\chi^2_N\big).
$$

The chi-squared mgf at \(\lambda=r/(2N)\) exists if and only if \(r/N<1\), i.e. \(N>r\), and then

$$
\mathbb E[\widehat{\mathrm{PPL}}_N]=\sqrt{2\pi\sigma^2}\,(1-r/N)^{-N/2}
=\mathrm{PPL}\cdot e^{-r/2}(1-r/N)^{-N/2}.
$$

For \(N\le r\) the expectation is infinite.

Expand \(\log(1-r/N)=-r/N-r^2/(2N^2)+O(N^{-3})\):

$$
(1-r/N)^{-N/2}=e^{r/2}\big(1+\tfrac{r^2}{4N}+O(N^{-2})\big).
$$

So \(c=r^2/4\). Independently, \(\mathrm{Var}(x^2)=2s^4\) and \(\mathrm{Var}(\ell)=r^2/2\), hence \(\mathrm{Var}(\ell)/2=r^2/4\). This matches 2.2(a). A Monte Carlo check at \((r,N)=(2,8)\) reproduced the closed form.

If \(r>1\), the model variance is smaller than the data variance. Then \(\mathbb E[\widehat{\mathrm{PPL}}_N]=\infty\) for every \(N\le r\): rare large data points make \(-\log p_\theta(x)\) so heavy-tailed that the exponential moment does not exist. The asymptotic bias coefficient \(r^2/4\) is also large. Jensen’s inequality still holds, in the extended sense \(\infty>\mathrm{PPL}\). Consistency is only visible once \(N>r\).

### 2.3. Jensen–Shannon

\begin{align*}
\mathrm{JS}(p\|q)
&=\tfrac12\sum p\log\frac pm+\tfrac12\sum q\log\frac qm\\
&=-\tfrac12 H(p)-\tfrac12 H(q)-\sum m\log m\\
&=H(m)-\tfrac12 H(p)-\tfrac12 H(q),
\end{align*}

where \(m=(p+q)/2\) and \(H\) is the Shannon entropy in the same base.

Let \(Z\sim\mathrm{Bernoulli}(1/2)\) and let \(X\mid Z=0\sim p\), \(X\mid Z=1\sim q\). Then \(X\sim m\) and

$$
I(Z;X)=H(X)-H(X\mid Z)=H(m)-\tfrac12 H(p)-\tfrac12 H(q)=\mathrm{JS}(p{\Vert}q).
$$

In nats, \(0\le I(Z;X)\le H(Z)=\log 2\). Thus \(0\le\mathrm{JS}\le\log 2\). The lower bound is attained if and only if \(Z\perp X\), i.e. \(p=q\). The upper bound is attained if and only if \(H(Z\mid X)=0\), i.e. \(Z\) is a function of \(X\), i.e. \(p\) and \(q\) are mutually singular.

This is the standard mutual-information representation; the bound \(\log 2\) in nats, or \(1\) in bits, is the one on [Wikipedia](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence).

### 2.3(a). Pointwise comparison, and an infinite KL

Since \(m\ge p/2\), \(\log(p/m)\le\log 2\) on \(\{p>0\}\), so \(\mathrm{KL}(p{\Vert}m)\le\log 2\). The same bound holds for \(q\), and averaging recovers \(\mathrm{JS}\le\log 2\). Equality holds simultaneously when \(m=p/2\) \(p\)-a.s. and \(m=q/2\) \(q\)-a.s., i.e. on disjoint supports.

JS remains finite when KL is infinite. If \(p=\delta_0\) and \(q=\delta_1\) on \(\{0,1\}\), then \(q(0)=0\) while \(p(0)=1\), so \(\mathrm{KL}(p{\Vert}q)=\infty\), but the supports are disjoint and \(\mathrm{JS}(p{\Vert}q)=\log 2\), the maximum.

The quoted bound \(0\le\mathrm{JS}\le 1\) is tight for the base-\(2\) logarithm (the maximum is one bit). In nats the supremum is \(\ln 2\), and the upper bound \(1\) is valid but not sharp.

### 2.4. Bernoulli KL near the boundary

\begin{align*}
\mathrm{KL}(\mathrm{Ber}(a)\|\mathrm{Ber}(c))
&=a\log\frac ac+(1-a)\log\frac{1-a}{1-c}\\
&=-a\log c+\big(a\log a+(1-a)\log(1-a)\big)-(1-a)\log(1-c).
\end{align*}

As \(c\to 0^+\), \(-\log(1-c)=c+O(c^2)\), so

$$
\mathrm{KL}(\mathrm{Ber}(a)\Vert\mathrm{Ber}(c))
=-a\log c-H(\mathrm{Ber}(a))+O(c).
$$

The leading term is \(-a\log c\to+\infty\). The constant term after it is \(-H(\mathrm{Ber}(a))\).

In the other direction, \(c\log c\to 0\) and \((1-c)\log((1-c)/(1-a))\to-\log(1-a)\), so

$$
\mathrm{KL}(\mathrm{Ber}(c)\Vert\mathrm{Ber}(a))\to-\log(1-a)\in(0,\infty).
$$

KL is not symmetric: charging the point that the other distribution is abandoning costs an arbitrarily large KL in one direction, and only a finite KL in the other. The \(c\log c\) contribution of a vanishing atom disappears.

### 2.4(a). No triangle inequality

KL does not satisfy the triangle inequality. For \(1>a>b>0\),

\begin{align*}
\mathrm{KL}(\mathrm{Ber}(a)\|\mathrm{Ber}(c))&=-a\log c+O(1),\\
\mathrm{KL}(\mathrm{Ber}(b)\|\mathrm{Ber}(c))&=-b\log c+O(1),
\end{align*}

while \(\mathrm{KL}(\mathrm{Ber}(a)\Vert\mathrm{Ber}(b))\) stays finite. Their difference grows like \(-(a-b)\log c\to+\infty\). So for every sufficiently small \(c>0\),

$$
\mathrm{KL}(\mathrm{Ber}(a)\Vert\mathrm{Ber}(c))
>\mathrm{KL}(\mathrm{Ber}(a)\Vert\mathrm{Ber}(b))+\mathrm{KL}(\mathrm{Ber}(b)\Vert\mathrm{Ber}(c)).
$$

Numerical triple, in nats: \((a,b,c)=(1/2,\,1/5,\,1/100)\),

$$
\mathrm{KL}(\mathrm{Ber}(1/2)\Vert\mathrm{Ber}(1/100))\approx 1.6145,
$$
$$
\mathrm{KL}(\mathrm{Ber}(1/2)\Vert\mathrm{Ber}(1/5))+\mathrm{KL}(\mathrm{Ber}(1/5)\Vert\mathrm{Ber}(1/100))\approx 0.2231+0.4287=0.6518.
$$

---

## Problem 3

Context length at position \(i\) is \(\ell_i=\min(m,i-1)\). A conditional table on \(K\) symbols has \(K-1\) free entries per context.

### 3.1. Inhomogeneous and homogeneous Markov models

$$
N_m^{\mathrm{nh}}
=\sum_{i=1}^n(K-1)K^{\min(m,i-1)}
=(K^m-1)+(n-m)(K-1)K^m.
$$

The sum splits into the growing prefix \(i\le m\), which contributes \(K^m-1\), and the \(n-m\) full windows of length \(m\).

Checks:

- \(m=0\): \(N=n(K-1)\). Each coordinate has its own marginal, and there is no context.
- \(m=n-1\): \(N=K^n-1\). An inhomogeneous model of order \(n-1\) is an arbitrary positive pmf, whose simplex has dimension \(K^n-1\).

For the homogeneous subfamily, the prefix \(x_{1:m}\) has an arbitrary pmf (\(K^m-1\) parameters) and all later positions share one kernel (\(K^m(K-1)\) parameters):

$$
N_m^{\mathrm{h}}=K^m-1+(K-1)K^m=K^{m+1}-1.
$$

Same checks: \(m=0\) gives one shared marginal, dimension \(K-1\); \(m=n-1\) gives \(K^n-1\), because only one position uses the kernel, so tying is vacuous.

Difference:

$$
N_m^{\mathrm{nh}}-N_m^{\mathrm{h}}=(n-m-1)(K-1)K^m.
$$

If \(n=m+1\), the difference is \(0\). There is only one index \(i>m\), so the homogeneous and inhomogeneous families are the same set, both equal to the full simplex. An autoregression of order \(n-1\) on \(n\) symbols is just an arbitrary joint.

### 3.1(a). Mixture transition model

$$
c_{ab}=\lambda t_1(a)+(1-\lambda)t_2(b).
$$

The cross difference vanishes:

$$
c_{00}-c_{01}-c_{10}+c_{11}=0,
$$

because both \(c_{00}-c_{01}\) and \(c_{10}-c_{11}\) equal \((1-\lambda)(t_2(0)-t_2(1))\). So the image lies in that hyperplane.

Conversely, every \(c\in[0,1]^4\) on this hyperplane is attained. Additivity gives a decomposition \(c_{ab}=A_a+B_b\). The normalized choice

$$
B_b=\min_{a\in\{0,1\}}c_{ab},\qquad A_a=c_{ab}-B_b
$$

is independent of the column used to read \(A\), satisfies \(A\ge 0\), \(B\ge 0\), \(\min A=0\), and \(\max A+\max B=\max c\le 1\). Set \(\lambda=\max A\), \(\alpha=A/\lambda\) when \(\lambda>0\), and \(\beta=B/(1-\lambda)\) when \(\lambda<1\). Then \(\alpha,\beta\in[0,1]^2\) and \(c_{ab}=\lambda\alpha_a+(1-\lambda)\beta_b\).

(The construction succeeded on all \(13339\) random cube points drawn on the hyperplane. The linear span of random images has rank \(3\), matching one constraint in \(\mathbb R^4\).)

The image is a full-dimensional subset of a 3-dimensional affine subspace, intersected with the cube, and it has nonempty relative interior (a neighborhood of \((1/2,1/2,1/2,1/2)\)). Its dimension is \(3\).

A free conditional table has dimension \(4\). The five nominal parameters \((\lambda,t_1(0),t_1(1),t_2(0),t_2(1))\) therefore overcount: the fiber of a generic table has dimension \(5-3=2\) (the split of mass between the two components, and a rescaling of \(\lambda\) against the size of \(t_1-t_2\)). The model cannot represent a conditional with a nonzero interaction, for example \(c=(1,0,0,1)\).

### 3.2. KL projection onto autoregressive tables

For \(q(x)=\prod_i q_i(x_i\mid c_i)\) and \(q^\star(x)=\prod_i p(x_i\mid c_i)\),

\begin{align*}
\mathrm{KL}(p\|q)
&=\mathrm{KL}(p\|q^\star)+\sum_{i=1}^n\mathbb E_p\Big[\log\frac{p(x_i\mid c_i)}{q_i(x_i\mid c_i)}\Big]\\
&=\mathrm{KL}(p\|q^\star)+\sum_{i=1}^n\mathbb E\big[\mathrm{KL}\big(p(\cdot\mid c_i)\,\|\,q_i(\cdot\mid c_i)\big)\big].
\end{align*}

Each inner KL is nonnegative. Hence \(\mathrm{KL}(p{\Vert}q)\ge\mathrm{KL}(p{\Vert}q^\star)\), with equality if and only if \(q_i(\cdot\mid c)=p(\cdot\mid c)\) for every \(i\) and every context (all contexts have positive probability because \(p>0\)). The unique closest member of \(Q_m^{\mathrm{nh}}\) is \(q^\star\).

### 3.2(a). The gap as conditional mutual information

Chain rule: \(\log p(x)=\sum_i\log p(x_i\mid x_{<i})\) and \(\log q^\star(x)=\sum_i\log p(x_i\mid c_i)\). For \(i\le m+1\) the window \(c_i\) is the entire past, so those terms cancel. For \(i\ge m+2\),

$$
\log\frac{p(x_i\mid x_{<i})}{p(x_i\mid c_i)}
$$

is the integrand of \(I(X_i;X_{1:i-m-1}\mid X_{i-m:i-1})\). Therefore

$$
\mathrm{KL}(p{\Vert}q^\star)=\sum_{i=m+2}^{n}I\big(X_i;\,X_{1:i-m-1}\mid X_{i-m:i-1}\big).
$$

It is zero if and only if every such mutual information vanishes, i.e. \(p\) itself factors through windows of length \(m\). For \(m=0\),

$$
\mathrm{KL}(p{\Vert}q^\star)=\sum_{i=2}^n I(X_i;X_{1:i-1})=\sum_i H(X_i)-H(X),
$$

the total correlation of \(p\).

### 3.2(b). Maximum likelihood tables

The average log-likelihood of \(N\) i.i.d. sequences is \(\sum_i\sum_{c,k}\hat p_i(c,k)\log q_i(k\mid c)\), where \(\hat p_i(c,k)\) is the empirical frequency of context \(c\) and symbol \(k\) at position \(i\). For each \((i,c)\) with a positive count, the Lagrangian of \(\sum_k N_i(c,k)\log q_k\) subject to \(\sum_k q_k=1\) has solution

$$
\hat q_i(k\mid c)=\frac{N_i(c,k)}{N_i(c)}.
$$

These are the conditionals of the empirical distribution \(\hat p\). Maximizing the likelihood is the same as minimizing \(\mathrm{KL}(\hat p{\Vert}q)\), whose unique minimizer on tables that are seen in the data is the \(q^\star\) of 3.2 built from \(\hat p\). Unseen contexts are unidentified.

### 3.2(c). Best homogeneous model

\(\log q=\log\pi(x_{1:m})+\sum_{i>m}\log T(x_i\mid c_i)\). The prefix expectation \(\mathbb E\log\pi(X_{1:m})\) is maximized uniquely by the true marginal \(\pi^\star=p_{1:m}\). The shared kernel maximizes

$$
\sum_{i=m+1}^n\sum_{c,k}p_i(c,k)\log T(k\mid c),
$$

so

$$
T^\star(k\mid c)=\frac{\sum_{i>m}p_i(c,k)}{\sum_{i>m}p_i(c)}
=\sum_{i=m+1}^n w_i(c)\,p_i(k\mid c),
$$

with weights \(w_i(c)=p_i(c)/\sum_{j>m}p_j(c)\), the relative frequency of context \(c\) at time \(i\).

The prefix of \(q^\star\) is already \(p_{1:m}\), so it contributes nothing extra. The excess KL is the cost of tying the kernels:

$$
\mathrm{KL}(p{\Vert}q_h^\star)-\mathrm{KL}(p{\Vert}q^\star)
=\sum_{i=m+1}^n\sum_c p_i(c)\,\mathrm{KL}\big(p_i(\cdot\mid c)\,\Vert\,T^\star(\cdot\mid c)\big).
$$

It vanishes if and only if \(p_i(\cdot\mid c)\) does not depend on \(i\), for every context that occurs after time \(m\). That is homogeneity of order \(m\).

### 3.3. The graph in Figure 1

Parents: \(A,B\) are roots; \(\mathrm{pa}(C)=\{A\}\), \(\mathrm{pa}(D)=\{A\}\), \(\mathrm{pa}(E)=\{B\}\), \(\mathrm{pa}(F)=\{C,D\}\), \(\mathrm{pa}(G)=\{B,E\}\).

$$
p(x)=p(x_A)p(x_B)p(x_C\mid x_A)p(x_D\mid x_A)p(x_E\mid x_B)p(x_F\mid x_C,x_D)p(x_G\mid x_B,x_E).
$$

Binary conditional probability tables contribute \(2^{|\mathrm{pa}(v)|}\) free parameters: \(1+1+2+2+2+4+4=16\).

**Normalization on any finite DAG.** Induct along a topological order \(v_1,\ldots,v_n\). The empty product sums to 1. Given that the marginal of the first \(t-1\) variables is a pmf,

$$
\sum_{x_{v_t}}p(x_{v_t}\mid x_{\mathrm{pa}(v_t)})=1
$$

for every parent configuration, and the parents are among the previous variables. Multiplying and summing over \(x_{v_t}\) leaves the previous marginal. After \(n\) steps the product sums to 1. Acyclicity is what supplies the topological order.

### 3.3(a). Valid orderings

A valid ordering is a topological ordering.

On \(\{B,E,G\}\) the constraints are \(B\) before \(E\), \(B\) before \(G\), and \(E\) before \(G\). The only order is \(B,E,G\).

On \(\{A,C,D,F\}\) the constraints are \(A\) before \(C\) and \(D\), and both \(C\) and \(D\) before \(F\). So \(A\) is first, \(F\) is last, and \(\{C,D\}\) is free: \(2\) orders.

Interleaving a block of \(3\) with a block of \(4\) can be done in \(\binom{7}{3}\) ways. Total:

$$
\binom{7}{3}\cdot 1\cdot 2=70.
$$

Out of \(7!=5040\) permutations, the probability is \(70/5040=1/72\).

Exhaustive enumeration of the \(5040\) permutations reproduces \(70\).

### 3.3(b). Rooted forest

For a rooted tree in which every node has at most one parent and edges point away from the root, the number of orders in which every node precedes its descendants is \(n!/\prod_v h_v\), where \(h_v\) is the subtree size at \(v\).

Proof by induction. The root occupies the first position. If its child-subtrees have sizes \(h_1,\ldots,h_k\) and \(T_i\) valid orders, the interleaving count is

$$
\frac{(n-1)!}{h_1!\cdots h_k!}\prod_i T_i.
$$

The inductive hypothesis \(T_i=h_i!/\prod_{v\in\text{subtree }i}h_v\) cancels the factorials and leaves \(n!/\prod_v h_v\), using \(h_{\mathrm{root}}=n\). A forest is an interleaving of its trees, and the multinomial coefficient extends the same formula to the whole vertex set: still \(n!/\prod_v h_v\).

Delete \(C\to F\) and \(E\to G\). The forest is

- \(B\) with leaf children \(E,G\): sizes \(h_B=3\), \(h_E=h_G=1\);
- \(A\) with child \(C\) and child \(D\to F\): sizes \(h_A=4\), \(h_D=2\), \(h_C=h_F=1\).

$$
\prod_v h_v=4\cdot 3\cdot 2=24,\qquad \frac{7!}{24}=210.
$$

Enumeration gives \(210\).

Restoring an edge keeps exactly those forest orders that also put the new parent before the child.

- Restoring \(E\to G\): \(E\) and \(G\) are symmetric leaves, so swapping their labels is a bijection of the forest orders. Exactly half survive: fraction \(1/2\), count \(105\).
- Restoring \(C\to F\): inside \(A\)'s tree the three equally likely orders of \(\{C,D,F\}\) with \(D\) before \(F\) are \(CDF\), \(DCF\), \(DFC\), and the first two have \(C\) before \(F\). Fraction \(2/3\), count \(140\). The left tree does not change the relative order inside the right tree.
- Restoring both: the two events use disjoint vertex sets and are functions of the internal orders of the two trees, which are chosen independently before the interleaving. Fraction \(\tfrac12\cdot\tfrac23=\tfrac13\), count \(70\).

Enumeration gives \(105\), \(140\), and \(70\). The both-edges count \(70\) is exactly the direct count in 3.3(a). That is the comparison the last sentence asks for: the fraction computation, using only the forest count, reproduces part (a).

### 3.3(c). A directed cycle does not normalize

\begin{align*}
q(0,0)&=(1-\theta_0)(1-\varphi_0),&
q(1,0)&=\theta_0(1-\varphi_1),\\
q(0,1)&=(1-\theta_1)\varphi_0,&
q(1,1)&=\theta_1\varphi_1.
\end{align*}

Summing and cancelling leaves

$$
Z=1+(\theta_0-\theta_1)(\varphi_0-\varphi_1).
$$

(The four-term expansion was checked against the closed form at \((\theta,\varphi)=((0.2,0.8),(0.3,0.9))\): both equal \(1.36\).)

So \(Z=1\) if and only if \(\theta_0=\theta_1\) or \(\varphi_0=\varphi_1\), i.e. at least one of the two conditionals ignores its parent. Otherwise the product of the two cyclic conditionals is not a pmf.

Maximum likelihood needs \(\sum_x\log(q_\theta(x)/Z(\theta))\). Dropping \(\log Z(\theta)\) is not the log-likelihood as soon as \(Z\) depends on \(\theta\): the score misses \(-\nabla\log Z\), and the critical-point equations are the moment-matching equations of a different objective. The parameters remain coupled through the normalizer, which a DAG factorization would have avoided.

---

## Problem 6

### 6.1. Bayes classifier for class-conditional Gaussians

$$
p(y=1\mid x)=\sigma(a(x)),\qquad
a(x)=\log\frac{\pi}{1-\pi}+\log\mathcal N(x\mid\mu_1,\Sigma_1)-\log\mathcal N(x\mid\mu_0,\Sigma_0).
$$

Expanding the quadratic forms,

\begin{align*}
a(x)&=\log\frac{\pi}{1-\pi}-\tfrac12\log\frac{\det\Sigma_1}{\det\Sigma_0}\\
&\quad+\tfrac12(x-\mu_0)^\top\Sigma_0^{-1}(x-\mu_0)-\tfrac12(x-\mu_1)^\top\Sigma_1^{-1}(x-\mu_1).
\end{align*}

**Shared covariance** \(\Sigma_0=\Sigma_1=\Sigma\). The pure quadratic terms cancel and

$$
a(x)=w^\top x+b,\qquad
w=\Sigma^{-1}\Delta,\qquad
b=-w^\top\bar\mu+\log\frac{\pi}{1-\pi},
$$

with \(\Delta=\mu_1-\mu_0\) and \(\bar\mu=\tfrac12(\mu_0+\mu_1)\). The logit is affine. (Checked against `scipy` log-densities in dimension \(3\).)

**Unequal covariances.**

$$
a(x)=x^\top Q x+L^\top x+c,
$$
$$
Q=\tfrac12(\Sigma_0^{-1}-\Sigma_1^{-1}),\qquad
L=\Sigma_1^{-1}\mu_1-\Sigma_0^{-1}\mu_0,
$$
$$
c=\log\frac{\pi}{1-\pi}-\tfrac12\log\frac{\det\Sigma_1}{\det\Sigma_0}
-\tfrac12\mu_1^\top\Sigma_1^{-1}\mu_1+\tfrac12\mu_0^\top\Sigma_0^{-1}\mu_0.
$$

This is the quadratic discriminant.

### 6.1(a). Parameter counts and the fiber

Shared-covariance generative model: one parameter \(\pi\in(0,1)\), two means in \(\mathbb R^D\), and a covariance in the positive definite cone of dimension \(D(D+1)/2\). Total

$$
1+2D+\frac{D(D+1)}{2}.
$$

Logistic regression: \((w,b)\in\mathbb R^{D+1}\), so \(D+1\) parameters.

Fix \((w,b)\). Choose \(\Sigma\succ 0\) freely and \(\bar\mu\in\mathbb R^D\) freely. Then \(\Delta=\Sigma w\) is determined, the two means are \(\bar\mu\pm\Delta/2\), and

$$
\log\frac{\pi}{1-\pi}=b+w^\top\bar\mu
$$

determines \(\pi\in(0,1)\). Different \((\Sigma,\bar\mu)\) give different generative parameters, and every generative parameter mapping to this \((w,b)\) arises this way. The fiber is diffeomorphic to \(\{\Sigma\succ 0\}\times\mathbb R^D\) and has dimension

$$
\frac{D(D+1)}{2}+D=\frac{D(D+3)}{2}.
$$

This equals the difference of the two parameter counts, so the map onto \((w,b)\) is a submersion: every logistic parameter is realized.

### 6.2. Marginal classifier on the observed coordinates

The marginal of a shared-covariance Gaussian is Gaussian with the corresponding block:

$$
x_V\mid y=c\sim\mathcal N(\mu_{c,V},\Sigma_{VV}).
$$

Part 6.1, applied in the coordinates \(V\), gives

$$
p(y=1\mid x_V)=\sigma(w_V^\top x_V+b_V),
$$
$$
w_V=\Sigma_{VV}^{-1}\Delta_V,\qquad
b_V=-\Delta_V^\top\Sigma_{VV}^{-1}\bar\mu_V+\log\frac{\pi}{1-\pi}.
$$

The matrix \(S_{V\mid M}\) is the conditional covariance of \(x_V\) given \(x_M\). It is not the covariance of this marginal. The Bayes rule for the masked problem inverts \(\Sigma_{VV}\), not \(S_{V\mid M}\).

### 6.2(a). Mean imputation is a different coefficient

The full-data classifier has \(w=\Sigma^{-1}\Delta\) and \(b=-w^\top\bar\mu+\log(\pi/(1-\pi))\). Substituting \(x_M\leftarrow\bar\mu_M\) produces

\begin{align*}
w^\top\tilde x+b
&=w_V^\top x_V+w_M^\top\bar\mu_M-w_V^\top\bar\mu_V-w_M^\top\bar\mu_M+\log\frac{\pi}{1-\pi}\\
&=w_V^\top(x_V-\bar\mu_V)+\log\frac{\pi}{1-\pi}.
\end{align*}

The block formula for \(\Sigma^{-1}\), with Schur complement \(S_{V\mid M}\), is

$$
(\Sigma^{-1})_{VV}=S_{V\mid M}^{-1},\qquad
(\Sigma^{-1})_{VM}=-S_{V\mid M}^{-1}\Sigma_{VM}\Sigma_{MM}^{-1}.
$$

Hence the coefficient actually applied to \(x_V\) is

$$
w_V^{\mathrm{plug}}=S_{V\mid M}^{-1}\big(\Delta_V-\Sigma_{VM}\Sigma_{MM}^{-1}\Delta_M\big).
$$

The Bayes coefficient from 6.2 is \(w_V^{\mathrm{Bayes}}=\Sigma_{VV}^{-1}\Delta_V\). Their difference is

$$
w_V^{\mathrm{plug}}-w_V^{\mathrm{Bayes}}
=S_{V\mid M}^{-1}\big(\Delta_V-\Sigma_{VM}\Sigma_{MM}^{-1}\Delta_M\big)-\Sigma_{VV}^{-1}\Delta_V.
$$

Because the imputed logit is exactly \(w_V^{\mathrm{plug}}\cdot(x_V-\bar\mu_V)+\mathrm{logit}(\pi)\), the two classifiers agree for every \(x_V\) if and only if the coefficient vectors agree. Using \(\Sigma_{VV}-S_{V\mid M}=\Sigma_{VM}\Sigma_{MM}^{-1}\Sigma_{MV}\), that equality rearranges to

$$
\Sigma_{VM}\Sigma_{MM}^{-1}\big(\Delta_M-\Sigma_{MV}\Sigma_{VV}^{-1}\Delta_V\big)=0.
$$

Sufficient conditions: \(\Sigma_{VM}=0\) (the masked block is uncorrelated with the observed block), or \(\Delta_M=\Sigma_{MV}\Sigma_{VV}^{-1}\Delta_V\) (the class-mean gap on \(M\) is the linear regression, in the \(\Sigma\) geometry, of the gap on \(V\)).

Interpretation: filling the mask by the midpoint \(\bar\mu_M\) and reusing the full discriminant pretends that those coordinates were observed. It therefore uses the smaller conditional covariance \(S_{V\mid M}\) rather than the marginal covariance \(\Sigma_{VV}\), and it corrects \(\Delta_V\) by the regression of \(\Delta_M\). That is not the posterior that integrates the missing coordinates out, unless the displayed orthogonality holds.

### 6.3. Shared-covariance MLE

The complete log-likelihood separates into a Bernoulli part in \(\pi\) and a Gaussian part.

$$
\hat\pi=\frac{N_1}{N}.
$$

For fixed \(\Sigma\), the quadratic \((x-\mu_c)^\top\Sigma^{-1}(x-\mu_c)\) is minimized in \(\mu_c\) at the class sample mean

$$
\hat\mu_c=\frac1{N_c}\sum_{y_n=c}x_n,
$$

assuming \(N_c\ge 1\). Let \(S_c=\sum_{y_n=c}(x_n-\hat\mu_c)(x_n-\hat\mu_c)^\top\) and let \(\hat\Sigma=N^{-1}(S_0+S_1)\) be the pooled scatter, assumed positive definite. The profile objective is

$$
L(\Sigma)=-\frac N2\log\det\Sigma-\frac N2\mathrm{tr}(\Sigma^{-1}\hat\Sigma).
$$

Let \(\lambda_i>0\) be the eigenvalues of \(\Sigma^{-1/2}\hat\Sigma\Sigma^{-1/2}\). Then \(\mathrm{tr}(\Sigma^{-1}\hat\Sigma)=\sum\lambda_i\) and \(\log\det\Sigma=\log\det\hat\Sigma-\sum\log\lambda_i\), so

$$
L(\hat\Sigma)-L(\Sigma)=\frac N2\sum_{i=1}^D(\lambda_i-1-\log\lambda_i).
$$

The scalar function \(\lambda-1-\log\lambda\) has derivative \(1-1/\lambda\) and second derivative \(1/\lambda^2>0\), so its minimum on \((0,\infty)\) is \(0\), attained only at \(\lambda=1\). Therefore \(L(\hat\Sigma)\ge L(\Sigma)\), with equality if and only if every \(\lambda_i=1\), if and only if \(\Sigma=\hat\Sigma\).

The MLE is \(\hat\pi=N_1/N\), the two class sample means, and \(\hat\Sigma=N^{-1}(S_0+S_1)\). It is the unique maximizer.

### 6.3(a). Logistic NLL, convexity, separation

With \(z_n=w^\top x_n+b\) and \(\varphi_n=(x_n,1)\),

$$
L(w,b)=\sum_{n=1}^N\big(\log(1+e^{z_n})-y_n z_n\big).
$$

$$
\nabla_w L=\sum_n(\sigma(z_n)-y_n)x_n,\qquad
\partial_b L=\sum_n(\sigma(z_n)-y_n),
$$

or \(\nabla L=\sum_n(\sigma(z_n)-y_n)\varphi_n\). Differentiating again, \(\sigma'(z)=\sigma(z)(1-\sigma(z))\), so

$$
\nabla^2 L=\sum_n\sigma(z_n)(1-\sigma(z_n))\,\varphi_n\varphi_n^\top\succeq 0.
$$

The negative log-likelihood is convex.

Suppose some \((w_0,b_0)\) strictly separates the sample. Along \(\theta_t=t(w_0,b_0)\), \(t\to\infty\), every class-1 logit tends to \(+\infty\) and every class-0 logit tends to \(-\infty\). Each summand tends to \(0\), so \(L(\theta_t)\to 0\). But for every finite \((w,b)\), \(\sigma(z)\in(0,1)\) and every summand is strictly positive, so \(L>0\). The infimum \(0\) is not attained. There is no minimizer.

Every minimizing sequence is unbounded and its margin tends to infinity: \(y_n^\star z_n\to+\infty\) for all \(n\), where \(y_n^\star=2y_n-1\). In particular \(t(w_0,b_0)\) with \(t\to+\infty\) is a minimizing sequence. This is the classical separation obstruction; a short discussion is on [Cross Validated](https://stats.stackexchange.com/questions/273247/theory-logistic-regression-mle-does-not-exist-when-data-are-perfectly-separable).

By contrast, the generative MLE of 6.3 exists and is unique as soon as \(\hat\Sigma\succ 0\) and both classes are nonempty, whether or not the sample is linearly separable. A positive pooled covariance keeps \((w,b)=(\hat\Sigma^{-1}\hat\Delta,\;\ldots)\) finite.

### 6.3(b). Penalized logistic regression

Add \(\tfrac\lambda2{\Vert}w\Vert^2\) with \(\lambda>0\), leaving \(b\) unpenalized. The Hessian becomes

$$
\sum_n\sigma_n(1-\sigma_n)\varphi_n\varphi_n^\top+\lambda\,\mathrm{diag}(I_D,0).
$$

For a nonzero direction \((v,c)\): if \(v\ne 0\), the penalty contributes \(\lambda{\Vert}v\Vert^2>0\); if \(v=0\) and \(c\ne 0\), the quadratic form is \(c^2\sum_n\sigma_n(1-\sigma_n)>0\), because \(\sigma(1-\sigma)>0\) at every finite logit and \(N\ge 1\). The Hessian is positive definite everywhere, so the penalized objective is strictly convex and has at most one minimizer.

It is coercive when both classes are nonempty, which separation already requires. The unpenalized \(L\) is nonnegative, so the objective is at least \(\tfrac\lambda2{\Vert}w\Vert^2\) and tends to infinity as \({\Vert}w\Vert\to\infty\). If \(w\) stays bounded and \(b\to+\infty\), every class-0 term \(\log(1+e^{z_n})\to\infty\). If \(b\to-\infty\), every class-1 term \(\log(1+e^{-z_n})\to\infty\). Thus the objective tends to infinity at infinity, a minimizer exists, and by strict convexity it is unique. Separation does not remove it: the penalty stops \({\Vert}w\Vert\) from running off to infinity.

---

## Problem 4

### 4.1. Parameter counts

FVSBN: one bias per coordinate, and \(w_d\in\mathbb R^{d-1}\) for \(d\ge 2\).

$$
D+\sum_{d=2}^D(d-1)=D+\frac{D(D-1)}{2}=\frac{D(D+1)}{2}.
$$

NADE: \(W\) has \(D_h(D-1)\) entries, \(c\) has \(D_h\), each of the \(D-1\) conditionals \(d\ge 2\) has a vector \(v_d\) and a bias, and \(b_1\) is one more bias.

$$
D_h(D-1)+D_h+(D-1)D_h+D=D_h(2D-1)+D.
$$

NADE is smaller when

$$
D_h(2D-1)<\frac{D(D-1)}{2},\qquad
D_h<\frac{D(D-1)}{2(2D-1)}.
$$

For \(D=784\), the threshold is \(306936/1567\approx 195.875\). The largest such integer is \(D_h=195\). At \(D_h=195\), NADE has \(1371\) fewer parameters; at \(D_h=196\) it has more.

### 4.1(a). Scalar multiplications in \(\log p(x)\)

Count products inside the affine maps, not evaluations of \(\sigma\), \(\log\), or \(\phi\).

- FVSBN: \(w_d^\top x_{<d}\) costs \(d-1\) products. Total \(D(D-1)/2\). At \(D=784\): \(306936\).
- NADE, recomputing each \(a_d\): building \(a_d\) costs \((d-1)D_h\) products and \(v_d^\top h_d\) costs \(D_h\) more, for each \(d\ge 2\). Total

$$
D_h\frac{D(D-1)}{2}+D_h(D-1)=D_h(D-1)\frac{D+2}{2}.
$$

At \(D=784\), \(D_h=500\): \(153468000+391500=153859500\).

- NADE, recursion \(a_{d+1}=a_d+W_{:,d}x_d\): one \(D_h\)-vector product for each of the \(D-1\) columns, plus \((D-1)D_h\) products in the output dots. Total \(2(D-1)D_h\). At the same size: \(783000\).

### 4.1(b). Sampling stays sequential

To sample \(x_d\) one needs the conditional, whose input \(a_d\) depends on \(x_1,\ldots,x_{d-1}\). Those bits are not known until the previous draws have been made, so the \(D\) draws form a chain. The recursion removes repeated work; it does not remove that chain.

For a fixed \(x\), every bit is already known. Each \(a_d=\sum_{k<d}W_{:,k}x_k\) is a function of known numbers, and the \(D\) conditionals may be evaluated independently, in parallel. The recursion is only an arithmetic optimization.

### 4.2. NADE gradients

The Bernoulli negative log-likelihood in the logit \(z\) has derivative \(\sigma(z)-x\). With \(\delta_d=\sigma(z_d)-x_d\),

$$
\frac{\partial L}{\partial z_d}=\delta_d.
$$

Consequences, using \(z_1=b_1\) and \(z_d=v_d^\top h_d+b_d\) for \(d\ge 2\):

\begin{align*}
\frac{\partial L}{\partial b_d}&=\delta_d &&(1\le d\le D),\\
\frac{\partial L}{\partial v_d}&=\delta_d h_d &&(d\ge 2),\\
\frac{\partial L}{\partial h_d}&=\delta_d v_d &&(d\ge 2),\\
\frac{\partial L}{\partial a_d}&=\phi'(a_d)\odot(\delta_d v_d) &&(d\ge 2).
\end{align*}

The bias \(b_d\) and the vector \(v_d\) touch only conditional \(d\). The shared parameters touch every later conditional: \(a_d=c+\sum_{k<d}W_{:,k}x_k\), so

$$
\frac{\partial L}{\partial c}=\sum_{d=2}^D\frac{\partial L}{\partial a_d},\qquad
\frac{\partial L}{\partial W_{:,k}}=x_k\sum_{d=k+1}^D\frac{\partial L}{\partial a_d}.
$$

### 4.2(a). One backward accumulation, MAP, tanh symmetry

Write \(g_d=\partial L/\partial a_d\). After one forward pass (the recursive one, cost \(O(DD_h)\)) the vectors \(g_d\) are known. Set \(S_{D-1}=g_D\) and \(S_k=g_{k+1}+S_{k+1}\) while \(k\) runs from \(D-2\) down to \(1\). Then \(\partial L/\partial W_{:,k}=x_k S_k\). The accumulation is another \(O(DD_h)\) pass, the same order as one forward pass, rather than a separate \(O(D^2 D_h)\) sum for each column.

With \(N\) observations and prior \(\theta\sim\mathcal N(0,\tau^2 I)\), the negative log posterior, up to an additive constant, is

$$
J(\theta)=\sum_{n=1}^N L(x^{(n)};\theta)+\frac{1}{2\tau^2}\Vert\theta\Vert^2.
$$

Its gradient is the sum of the data gradients plus \(\theta/\tau^2\). That additive term is weight decay with coefficient \(1/\tau^2\).

Symmetries of \(\phi=\tanh\), which is odd. For each hidden unit \(j\) and each sign \(\varepsilon_j\in\{\pm 1\}\), replace

$$
c_j\leftarrow\varepsilon_j c_j,\qquad W_{j,:}\leftarrow\varepsilon_j W_{j,:},\qquad (v_d)_j\leftarrow\varepsilon_j(v_d)_j\ \ (d\ge 2).
$$

Then \(a_{d,j}\) changes sign, \(\tanh(a_{d,j})\) changes sign, and \((v_d)_j h_{d,j}\) is unchanged. Every logit, hence \(p_\theta\), is unchanged. Doing this independently for each unit gives the group \(\{\pm 1\}^{D_h}\).

Any common permutation of the coordinates of \(c\), of the rows of \(W\), and of every \(v_d\) also leaves every \(v_d^\top h_d\) unchanged. Together with the sign flips this is the hyperoctahedral action, of order \(D_h!\,2^{D_h}\).

These \(D_h!\,2^{D_h}\) parameter vectors all define the same \(p_\theta\). For a generic \(\theta\) (all hidden rows nonzero, and no two rows equal up to a sign) they are distinct. That order is therefore a lower bound on the number of parameter vectors in a generic fiber. There is no continuous analogue for \(\tanh\): a scale factor does not pass through \(\tanh(au)\).

### 4.3. What an FVSBN can represent

For a positive pmf the logits \(\ell_d(x_{<d})=\log\frac{p(x_d=1\mid x_{<d})}{p(x_d=0\mid x_{<d})}\) may be arbitrary real functions of the past, and they determine the pmf. An FVSBN forces each logit to be affine,

$$
\ell_1=b_1,\qquad \ell_d(x_{<d})=b_d+w_d^\top x_{<d}\ (d\ge 2).
$$

So \(\mathcal F_D\) is exactly the set of positive pmfs whose autoregressive logits, in the fixed order, are affine.

The map from parameters to pmfs is injective. An affine function on the cube determines its coefficients: \(b=\ell(0)\) and \(w_i=\ell(e_i)-\ell(0)\). Different parameters give different logits and therefore different pmfs.

The parameter count is \(\sum_{d=1}^D d=D(D+1)/2\), and the parameter domain \(\mathbb R^{D(D+1)/2}\) is mapped diffeomorphically onto \(\mathcal F_D\) by the coordinatewise sigmoid. Hence \(\dim\mathcal F_D=D(D+1)/2\).

\(\dim\mathcal P_D=2^D-1\). Equality holds for \(D=1\) (\(1=1\)) and \(D=2\) (\(3=3\)). For \(D=3\), \(6<7\), and \(2^D-1\) grows faster than \(D(D+1)/2\) afterwards. Only \(D=1,2\).

### 4.3(a). The parity distribution is not an FVSBN

There are four odd strings and four even strings, and \(4\cdot(\theta/4)+4\cdot((1-\theta)/4)=1\), so \(p\) is a pmf. By symmetry of the three bits it is enough to rule out the order \((x_1,x_2,x_3)\); every other order has the same interaction in its last logit.

For each fixed \((x_1,x_2)\) the two completions have joints \(\theta/4\) and \((1-\theta)/4\), so the marginal of \((x_1,x_2)\) is uniform on four points, equal to \(1/4\). The conditional probability \(p(x_3=1\mid x_1,x_2)\) equals \(\theta\) when \(x_1\oplus x_2=0\) and equals \(1-\theta\) when \(x_1\oplus x_2=1\). With \(L=\log(\theta/(1-\theta))=\log(1/9)\),

$$
\ell(0,0)=\ell(1,1)=L,\qquad \ell(1,0)=\ell(0,1)=-L.
$$

An affine function would satisfy \(\ell(1,1)-\ell(1,0)-\ell(0,1)+\ell(0,0)=0\). The actual second-order interaction is

$$
\ell(1,1)-\ell(1,0)-\ell(0,1)+\ell(0,0)=4L=4\log\frac{1}{9}=-4\log 9\ne 0.
$$

No FVSBN represents \(p\), in any order.

### 4.3(b). Parity with a ReLU NADE of width \(D-1\)

Yes. Let \(L=\log(\theta/(1-\theta))\) and \(D_h=D-1\). Set \(b_1=0\), and \(v_d=0\), \(b_d=0\) for \(2\le d\le D-1\). Then \(p(x_d=1\mid x_{<d})=\sigma(0)=1/2\) for every \(d<D\), so the first \(D-1\) bits are i.i.d. uniform, regardless of \(W\) and \(c\).

For the last conditional, let \(s=\sum_{k=1}^{D-1}x_k\) and use one hidden unit per threshold:

$$
c_j=1-j\ (j=1,\ldots,D-1),\qquad W=\mathbf 1_{D_h\times(D-1)}.
$$

Then \(a_{D,j}=s-(j-1)\) and \(h_j=\mathrm{ReLU}(s-(j-1))\). On integers \(s\in\{0,\ldots,D-1\}\),

$$
\mathbf 1_{s\ge k}=h_k-h_{k+1}\ (k<D-1),\qquad \mathbf 1_{s\ge D-1}=h_{D-1}.
$$

Any function of the integer \(s\) is therefore linear in \(h\): \(g(s)=g(0)+\sum_{k=1}^{D-1}(g(k)-g(k-1))\mathbf 1_{s\ge k}\). Take \(g(s)=L(-1)^s\), so \(g(0)=L\) and \(g(k)-g(k-1)=2L(-1)^k\). Collecting coefficients of the \(h_j\) gives

$$
b_D=L,\qquad v_{D,1}=-2L,\qquad v_{D,j}=4L(-1)^j\ \text{for }j=2,\ldots,D-1.
$$

Then \(z_D=L\) on even popcount and \(z_D=-L\) on odd popcount, so

$$
p(x_D=1\mid x_{<D})=\begin{cases}\theta & \text{if }x_1\oplus\cdots\oplus x_{D-1}=0,\\ 1-\theta & \text{if }x_1\oplus\cdots\oplus x_{D-1}=1.\end{cases}
$$

In either case the conditional probability that the total parity is odd equals \(\theta\). The construction was checked by evaluating every binary input for every \(D\) from \(2\) through \(8\).

### 4.4. Masks for Figure 2

$$
[M^W]_{ki}=\mathbf 1\{m^1(k)\ge m^0(i)\},\qquad
[M^V]_{dk}=\mathbf 1\{m^0(d)>m^1(k)\}.
$$

Columns of \(M^W\) are \((x_1,x_2,x_3,x_4)\) with degrees \((2,4,1,3)\). Rows are \(h_1,\ldots,h_6\):

$$
M^W=\begin{pmatrix}
0&0&1&0\\
1&0&1&1\\
1&0&1&1\\
0&0&1&0\\
1&0&1&1\\
0&0&1&0
\end{pmatrix},\qquad
\#\{1\}=12.
$$

Rows of \(M^V\) are \(\lambda_1,\ldots,\lambda_4\):

$$
M^V=\begin{pmatrix}
1&0&0&1&0&1\\
1&1&1&1&1&1\\
0&0&0&0&0&0\\
1&0&0&1&0&1
\end{pmatrix},\qquad
\#\{1\}=12.
$$

Paths: hidden unit \(k\) sees inputs of degree at most \(m^1(k)\), and output \(d\) sees hidden units of degree strictly less than \(m^0(d)\). So \(\lambda_d\) can depend only on inputs of degree \(<m^0(d)\). Concretely, in the order \(x_3,x_1,x_4,x_2\):

| conditional | earlier variables the ordering allows | variables with a path |
|---|---|---|
| \(\lambda_3\) (\(x_3\)) | none | none |
| \(\lambda_1\) (\(x_1\)) | \(x_3\) | \(x_3\) |
| \(\lambda_4\) (\(x_4\)) | \(x_3,x_1\) | \(x_3\) only |
| \(\lambda_2\) (\(x_2\)) | \(x_3,x_1,x_4\) | \(x_3,x_1,x_4\) |

The network is not as expressive as a general autoregressive model for this order. The missing dependence is \(x_1\to\lambda_4\). A hidden unit that sees \(x_1\) needs degree at least \(2\), but \(\lambda_4\) accepts only degrees strictly less than \(3\). The only integers in between would be degree \(2\), and Figure 2 has no hidden unit of degree \(2\). The units of degree \(3\) see \(x_1\) and are forbidden from writing to \(\lambda_4\); the units of degree \(1\) write to \(\lambda_4\) and cannot see \(x_1\).

**Smallest width that drops nothing the order allows.** Label positions \(1,\ldots,D\). The dependence of position \(j\) on position \(r<j\) has a path if and only if some hidden degree lies in \(\{r,r+1,\ldots,j-1\}\). The consecutive pair \(r\to r+1\) forces a hidden unit of degree exactly \(r\). These \(D-1\) requirements are disjoint, so \(H\ge D-1\). One unit of each degree \(1,2,\ldots,D-1\) meets every interval, so \(H=D-1\) is sufficient. This is the connectivity statement. A single hidden layer of that width still cannot match an arbitrary conditional probability table; what it does not lose is any dependence the ordering itself permits.

### 4.4(a). Any depth, and the non-strict output rule

Use the same rule between successive hidden layers: a unit may read a previous unit only when its own degree is at least the previous unit’s degree. By induction on depth, a unit of degree \(m\) depends only on inputs of degree at most \(m\). The output \(\lambda_d\) reads only units of degree \(<m^0(d)\), hence only inputs of degree \(<m^0(d)\). It does not depend on \(x_d\) or on any later variable.

Therefore each factor \(\mathrm{Bernoulli}(x_d\mid\lambda_d)\) is a conditional probability given the earlier variables in the degree order. The product of an autoregressive factorization is a pmf: summing the variables from last to first, each Bernoulli sums to 1. So \(p\) is normalized.

If the output rule is weakened to \(\mathbf 1\{m^0(d)\ge m^1(k)\}\), a unit of degree \(m^0(d)\) may connect to \(\lambda_d\), and that unit may depend on the input of degree \(m^0(d)\), which is \(x_d\) itself. Then \(\lambda_d\) depends on \(x_d\), the factor is not a conditional distribution in the remaining variables, and \(\sum_{x_d}\mathrm{Bernoulli}(x_d\mid\lambda_d(x_d))\) need not be 1. In Figure 2 this happens for \(\lambda_3\) (through \(h_1,h_4,h_6\), which see \(x_3\)) and for \(\lambda_4\) (through \(h_2,h_3,h_5\), which see \(x_4\)).

### 4.4(b). Random hidden degrees

Degrees are i.i.d. uniform on \(\{1,\ldots,D-1\}\), and the input degrees are \(\{1,\ldots,D\}\).

For an input of degree \(r\), \(\mathbb P(m^1\ge r)=(D-r)/(D-1)\). Summing over \(r\) gives \(\sum_{r=1}^D(D-r)/(D-1)=D/2\). With \(H\) independent rows,

$$
\mathbb E[\#\{M^W=1\}]=\frac{HD}{2}.
$$

For an output of degree \(r\), \(\mathbb P(m^1<r)=(r-1)/(D-1)\), and the same sum is \(D/2\). Thus \(\mathbb E[\#\{M^V=1\}]=HD/2\) as well.

An allowed dependence \(r\to j\), with \(j>r\), is connected when at least one hidden degree falls in an interval of \(j-r\) integers:

$$
\mathbb P(\text{connected})=1-\Big(1-\frac{j-r}{D-1}\Big)^H.
$$

All allowed dependences are connected simultaneously if and only if every degree in \(\{1,\ldots,D-1\}\) appears at least once. Necessity: if degree \(r\) is missing, the consecutive pair \(r\to r+1\) has an empty interval. Sufficiency: the interval \(\{r,\ldots,j-1\}\) contains \(r\). This is the coupon collector on \(M=D-1\) coupons. By inclusion-exclusion, for \(H\ge D-1\),

$$
\mathbb P(\text{all connected})
=\sum_{i=0}^{D-1}(-1)^{D-1-i}\binom{D-1}{i}\Big(\frac{i}{D-1}\Big)^H,
$$

and the probability is \(0\) for \(H<D-1\). The expected number of hidden units needed until every dependence is connected is

$$
\mathbb E[T]=(D-1)\sum_{k=1}^{D-1}\frac1k.
$$

For \(D=784\), \(M=783\) and \(\mathbb E[T]=783\,H_{783}\approx 5669.69\).

---

## Problem 5

The Gaussian negative log-density with variance \(e^s\) is

$$
\ell(x;\mu,s)=\tfrac12\log(2\pi)+\tfrac s2+\frac{(x-\mu)^2}{2e^s}.
$$

### 5.1. Gradients

$$
\delta^\mu:=\frac{\partial\ell}{\partial\mu}=(\mu-x)e^{-s},\qquad
\delta^s:=\frac{\partial\ell}{\partial s}=\tfrac12-\frac{(x-\mu)^2}{2e^s}.
$$

Since \(L=\sum_d\ell(x_d;\mu_d,s_d)\), with \(\mu_1=b_1\), \(s_1=e_1\), and \(\mu_d=v_d^\top h_d+b_d\), \(s_d=u_d^\top h_d+e_d\) for \(d\ge 2\),

\begin{align*}
\frac{\partial L}{\partial b_d}&=\delta^\mu_d, &
\frac{\partial L}{\partial e_d}&=\delta^s_d,\\
\frac{\partial L}{\partial v_d}&=\delta^\mu_d\,h_d\ (d\ge 2), &
\frac{\partial L}{\partial u_d}&=\delta^s_d\,h_d\ (d\ge 2),\\
\frac{\partial L}{\partial h_d}&=\delta^\mu_d v_d+\delta^s_d u_d, &
\frac{\partial L}{\partial a_d}&=\phi'(a_d)\odot\frac{\partial L}{\partial h_d}.
\end{align*}

The shared parameters accumulate exactly as in 4.2, with this new \(g_d=\partial L/\partial a_d\):

$$
\frac{\partial L}{\partial c}=\sum_{d\ge 2}g_d,\qquad
\frac{\partial L}{\partial W_{:,k}}=x_k\sum_{d>k}g_d.
$$

### 5.1(a). Constant Gaussian fit

The stationarity condition \(\sum_n(\mu-x_n)e^{-s}=0\) forces \(\mu=\bar x\). Then \(\sum_n\delta^s=0\) forces

$$
e^{\hat s}=\frac1N\sum_n(x_n-\bar x)^2.
$$

If the sample variance is positive, this is the only critical point. It is the global minimum: for any \((\mu,s)\),

$$
\sum_n\ell(x_n;\mu,s)\ge \tfrac N2\log(2\pi)+\tfrac N2 s+\frac{S_N}{2e^s},
$$

where \(S_N=\sum(x_n-\bar x)^2\le\sum(x_n-\mu)^2\), and the right-hand side, as a function of \(t=e^s\), is minimized at \(t=S_N/N\) with second-derivative \(S_N/t^3>0\). As \(|\mu|\to\infty\) or \(s\to\pm\infty\) the loss tends to infinity whenever \(S_N>0\). If every \(x_n\) equals the same value, then \(s\to-\infty\) with \(\mu\) equal to that value sends the loss to \(-\infty\), so there is no minimizer; for \(x_n\) drawn from a nondegenerate Gaussian this event has probability \(0\).

At the critical point, \(\sum(x_n-\bar x)^2/\sigma_0^2\sim\chi^2_{N-1}\), so

$$
\mathbb E[e^{\hat s}]=\sigma_0^2\frac{N-1}{N}.
$$

### 5.2. Hessian in \((\mu,s)\)

$$
\nabla^2\ell=\begin{pmatrix}
e^{-s} & (x-\mu)e^{-s}\\
(x-\mu)e^{-s} & \tfrac12(x-\mu)^2 e^{-s}
\end{pmatrix}.
$$

Its determinant equals \(-\tfrac12(x-\mu)^2 e^{-2s}\le 0\), and it is strictly negative whenever \(x\ne\mu\). The Hessian is then indefinite. The loss is not jointly convex in \((\mu,s)\), and not jointly concave either.

### 5.2(a). Natural parameters

Set \(\eta_1=\mu e^{-s}\) and \(\eta_2=-\tfrac12 e^{-s}\) (so \(\eta_2<0\) and \(e^{-s}=-2\eta_2\)). Completing the exponential-family representation,

$$
\mathcal N(x\mid\mu,e^s)=\frac{1}{\sqrt{2\pi}}\exp(\eta_1 x+\eta_2 x^2-A(\eta)),
$$
$$
A(\eta)=-\tfrac12\log(-2\eta_2)-\frac{\eta_1^2}{4\eta_2}.
$$

Hence \(\ell=\tfrac12\log(2\pi)-\eta_1 x-\eta_2 x^2+A(\eta)\). The terms in \(x\) are linear in \(\eta\). The Hessian of \(A\) is positive definite on \(\{\eta_2<0\}\):

$$
\partial_{\eta_1\eta_1}A=\frac{1}{-2\eta_2}>0,\qquad
\det\nabla^2 A=-\frac{1}{4\eta_2^3}>0.
$$

So \(A\), and therefore \(\ell\), is strictly convex in \(\eta\).

### 5.2(b). Separate convexity, and the network parameters

For fixed \(s\), \(\partial^2\ell/\partial\mu^2=e^{-s}>0\): strictly convex in \(\mu\). For fixed \(\mu\), \(\partial^2\ell/\partial s^2=\tfrac12(x-\mu)^2 e^{-s}\ge 0\): convex in \(s\), strictly if \(x\ne\mu\).

If \((u_d,e_d,h_d)\) are held fixed, \(\mu_d\) is affine in \((v_d,b_d)\). A convex function of an affine map is convex, so the NLL is convex in \((v_d,b_d)\), jointly over \(d\). Likewise, with everything else fixed, \(s_d\) is affine in \((u_d,e_d)\) and the NLL is convex in those variance parameters.

It is not jointly convex in \((v_d,u_d)\). If \(h_d\ne 0\), the map \((v_d,u_d)\mapsto(\mu_d,s_d)\) is linear and surjective onto \(\mathbb R^2\). Joint convexity would imply joint convexity in \((\mu,s)\), which 5.2 forbids. Explicitly, \(h_d=1\), \(b_d=e_d=0\), \(x=0\), \(\mu=v=1\), \(s=u=0\): the Hessian in \((\mu,s)\) is \(\begin{pmatrix}1&-1\\-1&1/2\end{pmatrix}\), with determinant \(-1/2<0\).

### 5.3. One dimension, training NLL

**One point.** Set \(\mu=x\) and let \(s\to-\infty\). Then \(\ell=s/2+\tfrac12\log(2\pi)\to-\infty\). The NLL is not bounded below: a Gaussian spike on the single observation has infinite likelihood.

**\(N\ge 2\) distinct points.** Let \(S=\sum_n(x_n-\bar x)^2>0\). The inequality in 5.1(a) gives, for every \((\mu,s)\),

$$
\sum_{n=1}^N\ell(x_n;\mu,s)\ge\frac N2\log\Big(2\pi e\frac S N\Big),
$$

with equality at \(\mu=\bar x\) and \(e^s=S/N\). This is the greatest lower bound, and it is attained. The average NLL is bounded below by \(\tfrac12\log(2\pi e S/N)\).

### 5.3(a). Two-dimensional conditional, no floor

No, the training NLL is not bounded below. The first coordinates are distinct, and the hidden layer is wide enough to interpolate arbitrary values of \((\mu_2,s_2)\) at those finitely many inputs. Set \(\mu_2(x_{n1})=x_{n2}\) and send \(s_2(x_{n1})\to-\infty\) at each training input, keeping \(s_1\) fixed. Every conditional term \((x_{n2}-\mu_2)^2 e^{-s_2}\) vanishes while \(s_2/2\to-\infty\), so the sum of NLLs tends to \(-\infty\).

A discrete model cannot do the analogous thing. For any pmf \(q\) and any data distribution \(p_{\mathrm{data}}\),

$$
-\mathbb E_{p_{\mathrm{data}}}\log q=H(p_{\mathrm{data}})+\mathrm{KL}(p_{\mathrm{data}}{\Vert}q)\ge H(p_{\mathrm{data}}).
$$

The average NLL cannot undercut the entropy of the data distribution. Continuous training data are a sum of atoms. No density is absolutely continuous with respect to that atomic measure in the reverse direction that would produce a finite KL lower bound, and a density value at a point can be arbitrarily large. There is no entropy floor under the empirical measure.

### 5.3(b). Variance floor

(The PDF says “change (b)”; the quantity that changes is the unboundedness proved in 5.3(a).) With \(s_d\ge s_{\min}\) for every conditional,

$$
\ell(x_d;\mu_d,s_d)\ge\tfrac12\log(2\pi)+\tfrac{s_d}2\ge\tfrac12\big(\log(2\pi)+s_{\min}\big),
$$

because the squared residual is nonnegative and \(s/2\) is increasing. Summing over the \(D\) coordinates,

$$
-\log p_\theta(x)\ge\frac D2\big(\log(2\pi)+s_{\min}\big)
$$

for every \(x\) and every parameter obeying the floor. The training NLL of 5.3(a) is therefore bounded below, by \(N\) times this constant. The floor is what creates the bound: without it, \(s_2\to-\infty\) was available at the training inputs.

### 5.4(a). Mixture responsibilities

Let \(\varphi_c=\mathcal N(x\mid\mu_c,e^{s_c})\) and \(Z=\sum_j\pi_j\varphi_j\), so \(L=-\log Z\) and \(\gamma_c=\pi_c\varphi_c/Z\).

Softmax Jacobian: \(\partial\pi_j/\partial\ell_c=\pi_j(\mathbf 1_{j=c}-\pi_c)\). Hence

$$
\frac{\partial Z}{\partial\ell_c}=\pi_c\varphi_c-\pi_c Z,\qquad
\frac{\partial L}{\partial\ell_c}=\pi_c-\gamma_c.
$$

Summing over components, \(\sum_c\partial L/\partial\ell_c=\sum\pi_c-\sum\gamma_c=0\). This is the redundancy of the softmax: adding a constant to every logit does not change \(\pi\).

Differentiating through one component only,

$$
\frac{\partial\log\varphi_c}{\partial\mu_c}=(x-\mu_c)e^{-s_c}=-\delta^\mu_c,\qquad
\frac{\partial\log\varphi_c}{\partial s_c}=-\delta^s_c,
$$

and therefore

$$
\frac{\partial L}{\partial\mu_c}=\gamma_c\,\delta^\mu_c,\qquad
\frac{\partial L}{\partial s_c}=\gamma_c\,\delta^s_c.
$$

Compared with 5.1, each Gaussian gradient is multiplied by the responsibility of that component. The logit gradient \(\pi_c-\gamma_c\) has no single-Gaussian counterpart.

Since \(\ell_c=a_c^\top h+\alpha_c\), \(\mu_c=v_c^\top h+b_c\), and \(s_c=u_c^\top h+e_c\),

$$
\frac{\partial L}{\partial h}
=\sum_{c=1}^C\Big((\pi_c-\gamma_c)a_c+\gamma_c\delta^\mu_c\,v_c+\gamma_c\delta^s_c\,u_c\Big).
$$

With a single component, \(\gamma=1\) and \(\pi\) is absent, and this reduces to \(\delta^\mu v+\delta^s u\).

### 5.4(b). Mixture likelihood is unbounded

No upper bound, once \(C\ge 2\) and the \(N\ge 2\) points are distinct. Put \(\mu_1=x_1\), send \(s_1\to-\infty\), and keep \(\pi_1=\pi_2=1/2\), \(\mu_2\) arbitrary, and \(s_2\) fixed. The factor at \(x_1\) tends to infinity because \(\varphi_1(x_1)\to\infty\). At every \(x_n\ne x_1\), \(\varphi_1(x_n)\to 0\), but \(\pi_2\varphi_2(x_n)\) stays equal to a positive constant. The product of the \(N\) mixture densities tends to infinity. One component collapses onto a single observation; another component keeps the remaining observations at a finite density. For \(C=1\) this escape is impossible, by 5.3.

### 5.4(c). The mixture NLL is not convex

Take \(C=2\), one observation \(x=0\), \(\pi=(1/2,1/2)\), \(s_1=s_2=0\), and free means. Up to an additive constant,

$$
L(\mu_1,\mu_2)=-\log\big(e^{-\mu_1^2/2}+e^{-\mu_2^2/2}\big).
$$

At the endpoints \((0,3)\) and \((3,0)\),

$$
L=-\log(1+e^{-4.5})\approx -0.0111.
$$

At the midpoint \((1.5,1.5)\),

$$
L=-\log\big(2e^{-1.125}\big)=-\log 2+1.125\approx 0.4319,
$$

which is larger than the average of the endpoint values. The function lies above the chord, so it is not convex.

Swapping the two components' means leaves \((\pi,s)\) unchanged in this example and does not change the mixture. A generic pair with \(\mu_1\ne\mu_2\) therefore has \(2=2!\) parameter vectors, related by relabeling, that define the same distribution. For a mixture with \(C\) labeled components the same relabeling symmetry has order \(C!\).

---

## What was checked against the PDF, numerically, and in the literature

- The mgf calculation, the product identity \(Z=\mathcal N(a\mid b,A+B)\), Woodbury’s identity for the two posterior covariances, the AR(1) regression, and the hub determinant were re-derived and, where a number is involved, evaluated.
- Excess kurtosis \(-2a^4/(1+a^2)^2\), the mode threshold \(a=1\), the chi-squared Chernoff sample size \(950\), the importance-sampling second moment, the closed-form perplexity, the Bernoulli triangle counterexample, the \(70/210/105/140\) ordering counts, the parity NADE on every input for \(D\le 8\), the MADE masks (12 and 12 ones), and the rank-\(3\) image of the mixture transition model were computed directly.
- Gaussian KL: [Book of Statistical Proofs, mvn-kl](https://statproofbook.github.io/P/mvn-kl.html). Jensen–Shannon range and the mutual-information form: [Wikipedia](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence). Separation for logistic regression: [Cross Validated](https://stats.stackexchange.com/questions/273247/theory-logistic-regression-mle-does-not-exist-when-data-are-perfectly-separable). The strict output mask is the autoregressive constraint of Germain, Gregor, Murray, and Larochelle, MADE (ICML 2015). The zero-variance importance-sampling proposal and the Gaussian maximum-entropy property are the standard theorems (nonnegative integrand; Cover and Thomas).
