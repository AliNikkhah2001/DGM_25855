# Homework 2 study map

Deep Generative Models, course 25732-1, Dr. Sajjad Amini. Written sheet due Wednesday, Aban 27, 23:59. Simulation is two notebooks, 100 points each.

This order is topic by topic. Each block teaches one idea, then uses it in a proof, then in code. Do not start the flow notebook before the change-of-variables lecture, and do not start problem 2 before the ELBO identity in problem 1 is comfortable.

The slides used below are the instructor’s decks on [dgm-sut-amini.github.io](https://dgm-sut-amini.github.io/), copied into [`lectures/`](../lectures/). The foundation deck matches the Fall 2026 Telegram copy slide for slide (46 slides, same section titles).

Slide numbers below are the numbers printed on the slides (`k / N`), not the PDF page index.

## What the lectures already contain

| Lecture | Local file | Slides that this homework uses |
|---|---|---|
| L0 Logistics | `DGM_Presentation-0_Logistics.pdf` | 5 (prerequisites), 12 (books) |
| L2 Foundation | `DGM_Presentation-2_Foundation.pdf` | 12–35 density, expectation, Monte Carlo; 36–42 KL |
| L4 Variational autoencoder | `DGM_Presentation-4_Variational_Autoencoders.pdf` | 12–40 model, importance sampling, ELBO; 56–64 reparameterization; 77–84 reconstruction minus KL |
| L5 Normalizing flows | `DGM_Presentation-5_Flow_Models.pdf` | 3–34 change of variables and Jacobians; 40–51 NICE; 52–64 RealNVP; 65–73 MAF |

## What you have to read outside the VAE and flow decks

These four pieces of the written sheet are not sections of Lectures 4 or 5.

| Gap | Where to read it |
|---|---|
| Conditional ELBO (problem 1) | Same proofs as Lecture 4, with the condition \(u\) written into every density. Optional paper: Sohn, Lee, and Yan, Learning Structured Output Representation using Deep Conditional Generative Models, NeurIPS 2015. |
| Two-layer hierarchical ELBO (problem 2) | Tomczak, *Deep Generative Modeling*, §5.5.2, especially §5.5.2.1 (two-level VAEs). Murphy, *Probabilistic Machine Learning: Advanced Topics* (2023), chapter 21. |
| Variational dequantization (problem 4.2–4.4) | Tomczak §4.1.3.3. Ho, Chen, Srinivas, Duan, and Abbeel, Flow++: Improving Flow-Based Generative Models with Variational Dequantization and Architecture Design, ICML 2019, the variational-dequantization section. |
| Glow invertible 1×1 convolution (problem 5) | Kingma and Dhariwal, Glow: Generative Flow with Invertible 1×1 Convolutions, NeurIPS 2018, the invertible 1×1 convolution section. The determinant facts are the triangular-matrix facts from linear algebra. Lecture 5 stops at a fixed channel permutation inside RealNVP. |

Lectures 8 (diffusion), 9 (inverse problems), and 10 (evaluation) are later in the course. Diffusion is a hierarchical latent model, but it is not the model in problem 2. Leave it until this homework is done.

## Prerequisites

The logistics slide (L0, slide 5) lists three prerequisites and says the course is not an introduction to probability or to deep learning.

| You need to be able to | Where it is | First used in |
|---|---|---|
| Joint, conditional, and marginal densities; Bayes; the chain rule | L2 slides 12–29. MIT RES.6-012, the playlist linked on the course site. | Every problem |
| Expectation of a function, and Jensen’s inequality for a concave function: \(\log \mathbb E[X] \ge \mathbb E[\log X]\) | L2 slides 30–35 for expectation. Jensen is stated on Lecture 4, slide 34. | Theory 1.1, 4.3 |
| KL is nonnegative, zero if and only if the two densities agree almost everywhere, and not symmetric | L2 slides 36–39. Homework 1, problem 2. | Theory 1.1, 1.2, 2, 4.4 |
| Monte Carlo: replace an expectation by a sample average | L2 slides 34–35. Homework 1, problem 2. | Theory 1.4; VAE II.3; Flow II.3 |
| An affine image of a Gaussian is Gaussian. In one dimension, \(z=\mu+\sigma\varepsilon\) with \(\varepsilon\sim\mathcal N(0,1)\) | Homework 1, problem 1. | Theory 1.3, 1.4; VAE I.2 |
| Closed-form KL between univariate or diagonal Gaussians | Homework 1, the Gaussian KL derivation. Problem 1.3 asks you to redo it for two diagonal covariances, then specialize the prior to \(\mathcal N(0,I)\). | Theory 1.3; VAE I.2b |
| Determinant of a triangular matrix, and \(\det(AB)=\det(A)\det(B)\) | Any linear-algebra text. Lecture 5 uses the consequence for Jacobians. | Theory 4.1, 5 |
| Autoregressive conditionals and MADE masks | Homework 1, the autoregressive problems, and Lecture 3. | Flow notebook part III (MAF). Lecture 5, slides 65–73. |
| A small conv net in PyTorch: `nn.Module`, `binary_cross_entropy_with_logits`, autograd, `DataLoader`, `torch.no_grad` | The notebooks scaffold everything around the TODOs. | All code |

Books named on L0, slide 12, in the order they help this homework:

1. Jakub M. Tomczak, *Deep Generative Modeling*, Springer. Chapter 4 is flows. Chapter 5 is latent-variable models. The section numbers below refer to this book.
2. Kevin P. Murphy, *Probabilistic Machine Learning: Advanced Topics*, MIT Press, 2023. Chapter 10 is variational inference (ELBO, tighter bounds). Chapter 11.2 and 11.5 are Monte Carlo and importance sampling. Chapter 21 is variational autoencoders, including posterior collapse and hierarchical VAEs. Chapter 23 is normalizing flows.
3. Christopher M. Bishop and Hugh Bishop, *Deep Learning: Foundations and Concepts*, Springer, 2023. Use it when a derivation in Tomczak or the slides feels too compressed. It is the third book on the logistics slide, not the first place to read for a numbered section of this homework.

Online courses named on the same slide: Stefano Ermon, Stanford CS236; Pieter Abbeel, Peter Chen, Jonathan Ho, and Aravind Srinivas, Berkeley CS294-158 (spring 2020). Use the VAE and flow lectures there if a slide is hard to read. The course site also lists MIT RES.6-012 for probability.

## Order of work

| Step | Block | Do this | Opens |
|---|---|---|---|
| 1 | Probability | L2 slides 12–42, and Homework 1 problems 1 and 2 as a refresher | Every written proof |
| 2 | Unconditional VAE | Lecture 4, slides 12–40, then 56–64, then 77–84 | Theory 1 |
| 3 | Conditional ELBO | Write theory 1.1, 1.2, 1.3, 1.4 | VAE notebook |
| 4 | VAE code | Notebook 1, parts I, II, then III | — |
| 5 | Two latents | Tomczak §5.5.2.1, then theory problem 2 | — |
| 6 | Change of variables | Lecture 5, slides 3–34, then 40–64 | Theory 4.1, 3, 5 |
| 7 | Coupling and flow priors | Write 4.1, then 3.1 and 3.2, then 5 | Flow notebook |
| 8 | Discrete pixels | Tomczak §4.1.3.3 and Flow++, then theory 4.2, 4.3, 4.4 | — |
| 9 | Flow code | Notebook 2, part III, then II, then I | — |

Part I of the flow notebook is 60 points and needs MVTec. It is last because it is the longest experiment, not because it is the hardest idea. The idea is the RealNVP coupling from step 6.

## Block 1. The ELBO, then the conditional VAE

Read Lecture 4 in this order.

| Slides | What the lecture is doing | Homework |
|---|---|---|
| 3–11 | A Gaussian mixture as a latent-variable model. One discrete cluster index explains the iris data. | The picture behind VAE part II: a mixture prior can put mass on several modes, a single Gaussian cannot. |
| 12–16 | Continuous latent \(z\), joint \(p_\theta(x,z)=p_\theta(x\mid z)p_\theta(z)\), generation by sampling \(z\) then \(x\). | The model in theory problem 1, before the condition \(u\) is added. |
| 17–29 | \(\log p_\theta(x)\) is an expectation over the posterior. The posterior is intractable. Importance sampling is unbiased and can have huge variance. | Why a bound is used instead of the likelihood. |
| 30–40 | Jensen gives the ELBO. The gap equals \(\mathrm{KL}(q(z)\|p_\theta(z\mid x))\). The bound is tight if and only if \(q\) is the true posterior. | Theory 1.1, both proofs. Write \(u\) next to \(x\) in every density. |
| 41–55 | Choosing a variational family and tightening the bound. | The equality case in 1.1, and the collapse argument in 1.2. |
| 56–64 | The score-function gradient is high variance. The reparameterization \(z=\mu+\sigma\varepsilon\), \(\varepsilon\sim\mathcal N(0,I)\), moves the parameter into a deterministic map. | Theory 1.4. VAE notebook I.2a and the reconstruction gradient. |
| 65–76 | Stochastic variational inference, then amortization: one network predicts \((\mu,\sigma)\) for every \(x\). | Why the notebook has an encoder rather than a free \(q\) per image. |
| 77–84 | \(\mathrm{ELBO}=\mathbb E[\log p_\theta(x\mid z)]-\mathrm{KL}(q_\phi(z\mid x)\|p_\theta(z))\). | Theory 1.2, the reconstruction / regularization split. VAE notebook I.4. |

Theory problem 1, after those slides:

1. **1.1.** Repeat slide 34 (Jensen) and slides 38–39 (KL identity) with \(p(x,z\mid u)\) and \(q(z\mid x,u)\). Equality holds exactly when the KL is zero, which is when the two densities agree almost everywhere. That fact is L2, slide 37.
2. **1.2.** Expand the joint as on slide 78. The first term is the reconstruction. The second is a KL against the conditional prior \(p(z\mid u)\), not against a posterior. If \(q(z\mid x,u)=p(z\mid u)\), that KL is zero and the gap that remains is \(\log\mathbb E[p(x\mid z,u)]-\mathbb E[\log p(x\mid z,u)]\), which is Jensen again. It is zero only when \(p(x\mid z,u)\) does not depend on \(z\). If the decoder ignores \(z\), the true posterior equals the prior. Matching \(q\) to that prior then makes both the KL and the gap vanish. That is posterior collapse: the latent is unused. Murphy chapter 21 and Tomczak §5.3.6 discuss the same failure.
3. **1.3.** Write the KL integral for two diagonal Gaussians and integrate coordinatewise. The \(\mathcal N(0,I)\) prior is the special case \(\mu_p=0\), \(\sigma_p=1\), which is the formula in VAE notebook I.2. The lecture does not print this integral. Homework 1’s Gaussian KL is the template.
4. **1.4.** Slides 61–63. State \(z=\mu_q+\sigma_q\odot\varepsilon\), check the mean and covariance, and differentiate the Monte Carlo reconstruction through that map. The KL is held in closed form, so it is not inside the Monte Carlo estimate.

## Block 2. VAE notebook, while the ELBO is fresh

File: `hw2/simulation/VAE.ipynb`. Fashion-MNIST, then a 2-D synthetic mixture, then MNIST with missing pixels.

| Part | Points | Implement | Uses |
|---|---|---|---|
| I.2a | 4 | `reparameterize` | Theory 1.4, Lecture 4 slides 61–63 |
| I.2b | 4 | `gaussian_kl` for \(q=\mathcal N(\mu,\mathrm{diag}(\sigma^2))\) against \(\mathcal N(0,I)\) | Theory 1.3 |
| I.3 | 4 | `encode`, `decode`, `forward` → logits, \(\mu\), log-variance | Lecture 4 slides 72–76 |
| I.4 | 7 | `loss_parts`: Bernoulli BCE-with-logits, summed over pixels, plus \(\beta=1\) times the KL, averaged over the batch. Nats per image. | Theory 1.2. Do not put a sigmoid on the logits inside the loss. |
| I.5 | 3 | `evaluate` over a whole loader, no gradient step | — |
| I.6 | 8 | One optimizer step, gradient clip at norm 5, AMP only when `USE_AMP` is true | — |
| I.7 | 5 | Samples from the prior \(\mathcal N(0,I)\), then sigmoid | Lecture 4, “Task 1: Generation”, slides 14–16 |
| I.8 | 5 | Interpolation in latent space | Lecture 4, slides 85–87, attribute directions, as the geometric idea |
| I.9 | write-up | Training curves and the fixed-\(z\) grid | Say what improved and what is still blurry |
| II.2 | 5 | Log-density of a Gaussian mixture | Lecture 4, slides 3–11; Tomczak §5.4.1.4 |
| II.3 | 4 | Monte Carlo estimate of a KL | L2 slides 34–35 |
| II.6 | 3 | Off-mode diagnostic | A single Gaussian prior spreads mass between the modes |
| II.7 | 3 | Discussion | Tie the picture to the KL term in theory 1.2 |
| III.2 | 9 | Observation masks | — |
| III.3 | 5 | Corruption schedule | — |
| III.4 | 7 | Train the matched models | — |
| III.6 | 6 | Fill only the missing pixels | Keep every observed pixel |
| III.7 | 9 | MSE on missing pixels | — |
| III.8 | 9 | Which model wins at 50% random masks and at a 40%-width square, and what clean reconstruction does | The question on the sheet: does training the encoder on incomplete images help restoration |

Part I is 40 points, part II is 15, part III is 45.

## Block 3. Hierarchical VAE

Theory problem 2. There is no matching notebook section.

Read Tomczak §5.5.2 through §5.5.2.1 before writing. The proof pattern is Lecture 4, slides 38–39, applied to the pair \((z_1,z_2)\).

The model is \(p(x,z_1,z_2)=p(z_2)\,p(z_1\mid z_2)\,p(x\mid z_1)\), with \(x\) independent of \(z_2\) given \(z_1\). The variational family factorizes as \(q(z_2\mid x)\,q(z_1\mid z_2,x)\).

1. **2.1.** Start from \(\mathrm{KL}(q(z_1,z_2\mid x)\|p(z_1,z_2\mid x))\) and rearrange, as on slide 38. The ELBO splits into a reconstruction \(\mathbb E[\log p(x\mid z_1)]\), a KL on \(z_2\), and an expected KL on \(z_1\mid z_2\). The inner KL is averaged under \(q(z_2\mid x)\) because \(z_2\) is random under the approximate posterior when the inner conditional is scored.
2. **2.2.** Compare with the ordinary one-layer ELBO on the marginal \(q(z_1\mid x)\). The difference is an expected KL, so the hierarchical ELBO is at most the marginal ELBO. Equality holds when \(q(z_2\mid z_1,x)=p(z_2\mid z_1)\). For a fixed marginal \(q(z_1\mid x)\), that conditional is the one that maximizes the hierarchical ELBO. The marginal prior \(p(z_1)\) may be intractable. An auxiliary \(r(z_2\mid z_1)\) produces a lower bound on \(\log p(z_1)\), and the bound is tight when \(r\) equals the true conditional \(p(z_2\mid z_1)\). That is the same “KL gap” argument as slide 39.

## Block 4. Change of variables, coupling, flow priors, Glow

Read Lecture 5 in this order.

| Slides | What the lecture is doing | Homework |
|---|---|---|
| 3–17 | A smooth bijection pushes a base density forward. Probability of an event is preserved. The density picks up the absolute Jacobian determinant. | Theory 3.1 and 4.1 |
| 18–34 | A flow is a stack of bijections. The log-determinant of a composition is the sum of the log-determinants. | Why each layer must have a tractable determinant |
| 35–39 | Maximum likelihood is exactly that log-density. No ELBO. | The contrast with the VAE |
| 40–51 | NICE: additive coupling. Half the coordinates are unchanged. The Jacobian is triangular with ones on the diagonal, so the log-determinant is zero. A separate rescaling layer carries the volume change. | The simpler cousin of 4.1 |
| 52–64 | RealNVP: affine coupling \(y_B=h_B\odot\exp(s(h_A))+t(h_A))\). The log-determinant is the sum of the scale outputs. The neural nets \(s\) and \(t\) never enter the determinant, because they do not affect the triangular structure. | Theory 4.1, word for word. Flow notebook parts I and II. |
| 65–73 | MAF: an autoregressive transformer with a MADE-style conditioner. Likelihood is parallel. Sampling is sequential. | Flow notebook part III. Homework 1 MADE masks. |
| 74–80 | Inverse autoregressive flow: the opposite cost. Fast sampling, slow likelihood. | Not asked. Useful so you do not implement an IAF where the notebook wants a MAF. |

Then write, in this order:

1. **4.1.** Give the inverse of the affine coupling (solve for \(h_B\) using \(\exp(-s)\)). The log-density formula on the sheet has a minus sign in front of \(\sum_j s_j(y_A)\) because the forward map’s log-determinant is \(+\sum s_j\), and the density of \(y\) is the base density at the preimage times \(\lvert\det J_{f^{-1}}\rvert\).
2. **3.1.** Put the Lecture 5 change-of-variables formula into the KL of Lecture 4, slide 78. The prior is no longer \(\mathcal N(0,I)\) in \(z\). It is the pushforward of \(\mathcal N(0,I)\) through \(f_\psi\).
3. **3.2.** Define the decoder on \(u\) by \(\tilde p(x\mid u)=p(x\mid f(u))\). The marginal likelihood is the same integral. The variational density transforms by the same Jacobian, so the two KLs are equal and the two ELBOs are equal when \(q\) and \(r\) are exact images of each other. Then restrict both families to diagonal Gaussians and use \(f(u_1,u_2)=(u_1,u_2+u_1^2)\) with \(q(z)=\mathcal N(0,I)\). The induced \(r(u)\) is not a diagonal Gaussian, so the restricted optimization problems differ even though the generative models match. Tomczak §5.4.1.8 is the flow-prior section.
4. **5.** Glow replaces RealNVP’s fixed channel permutation by \(Y_{:,i,j}=W X_{:,i,j}\). Invertible if and only if \(\det W\neq 0\), with inverse \(W^{-1}\) at each pixel. The full Jacobian is block-diagonal across the \(HW\) sites, each block contributing \(\det W\), so \(\log\lvert\det J\rvert=HW\log\lvert\det W\rvert\). The LU form \(W=PL(U+\mathrm{diag}(s))\) makes \(\log\lvert\det W\rvert=\sum_i\log\lvert s_i\rvert\), because a permutation has determinant \(\pm 1\), a unit-triangular matrix has determinant 1, and a diagonal matrix has determinant \(\prod s_i\).

## Block 5. From a density to a probability of a pixel

Theory 4.2–4.4. Read Tomczak §4.1.3.3 and the variational-dequantization section of Flow++ after 4.1 is written.

1. **4.2.** The continuous model assigns a density to \(y\in\mathbb R^D\). An integer image \(x\) owns the bin \(x+[0,1)^D\). The probability of \(x\) is the integral of the density over that bin. Those bins partition \(\mathbb R^D\) up to boundaries, so the masses sum to 1. The number \(p_\theta(x)\) is a density, not that mass.
2. **4.3.** A dequantization distribution \(q(u\mid x)\) supported on the unit bin gives a lower bound by importance sampling and Jensen, the same move as Lecture 4, slides 34–35. The bound equals \(\mathbb E[\log p_\theta(x+u)]+H(q(\cdot\mid x))\). The likelihood term wants \(x+u\) in a high-density region. The entropy term stops \(q\) from collapsing to a point.
3. **4.4.** The gap is \(\mathrm{KL}(q(u\mid x)\|r_\theta(u\mid x))\), where \(r\) is the model density renormalized inside the bin. The bound is tight exactly when \(q=r\). Uniform dequantization has density 1 on the bin, so the bound becomes the integral of \(\log p_\theta(x+u)\), and it is tight only when \(p_\theta(x+u)\) is constant on the bin. A conditional flow \(u=g_\phi(\varepsilon;x)\) can match a non-uniform \(r\), which is why it can be tighter. Substitute the change-of-variables formula from Lecture 5 into the objective in 4.3.

## Block 6. Flow notebook

File: `hw2/simulation/Flow.ipynb`. Do part III, then II, then I.

| Part | Points | Implement | Uses |
|---|---|---|---|
| III.1 | 6 | Fit one MAF to the source Lab chroma and one to the target, then transfer | Lecture 5, slides 65–73. Easiest flow in the notebook. |
| III.2 | 4 | Why global palette transfer is easier than conditional colorization, and why the colors can still be semantically wrong | A 2-D density does not know what object a pixel belongs to |
| II.1 | 8 | RGB ↔ YUV, and the split into luminance and subsampled chroma | Conditional model: \(p(\mathrm{chroma}\mid\mathrm{gray})\) |
| II.2 | 10 | The conditional flow step | Lecture 5, slides 52–55 and 71–73 |
| II.3 | 4 | Mean negative log-likelihood per chroma dimension | Lecture 5, slides 35–39 |
| II.4 | 8 | Why held-out NLL is not visual quality; what temperature 0.65 versus 1.1 does; why luminance is only approximately preserved after clipping | — |
| I.1 | 10 | Synthetic defects on brick patches. Healthy images only. | Spatial coupling, Lecture 5 slides 52–55, on features |
| I.2 | 8 | Anomaly score from the flow likelihood | Low likelihood under a healthy-only model |
| I.3 | 7 | Why localization works here, and what is still artificial | The defect is the only thing the training distribution never saw |
| I.4 | 10 | Real MVTec categories, frozen ResNet-18 features, PCA | Same model, harder images |
| I.5 | 10 | Healthy-score statistics | — |
| I.6 | 4 | Fuse the maps | — |
| I.7 | 11 | Best fused pixel AUROC in your run; when the fused map beats the shallow map; one visible limitation of likelihood-based detection | A high-likelihood defect, or a low-likelihood healthy texture, is the limitation to look for |

Part I is 60 points, part II is 30, part III is 10. The notebook says to keep seeds fixed, to run the checks before the long training, and to set `FAST_DEV_RUN=False` for the final run.

## Papers, in the order you will meet them

| When | Paper |
|---|---|
| Before theory 1 | D. P. Kingma and M. Welling, Auto-Encoding Variational Bayes, ICLR 2014. The ELBO and the reparameterization sections. The lecture is enough if those two sections feel familiar. |
| Optional for the condition \(u\) | K. Sohn, H. Lee, and X. Yan, Learning Structured Output Representation using Deep Conditional Generative Models, NeurIPS 2015. |
| With theory 1.2, collapse | Kingma and Welling, An Introduction to Variational Autoencoders, Foundations and Trends in Machine Learning, 2019. The sections on the ELBO, the reparameterization trick, and posterior collapse. |
| With theory 2 | Tomczak §5.5.2. C. K. Sønderby, T. Raiko, L. Maaløe, S. K. Sønderby, and O. Winther, Ladder Variational Autoencoders, NeurIPS 2016, if you want the original two-level construction. |
| With theory 3 | Tomczak §5.4.1.8. D. J. Rezende and S. Mohamed, Variational Inference with Normalizing Flows, ICML 2015, for the Jacobian term inside a variational posterior. The homework puts the flow in the prior. |
| With Lecture 5 | L. Dinh, D. Krueger, and Y. Bengio, NICE: Non-linear Independent Components Estimation, 2015. L. Dinh, J. Sohl-Dickstein, and S. Bengio, Density Estimation Using Real NVP, ICLR 2017. G. Papamakarios, T. Pavlakou, and I. Murray, Masked Autoregressive Flow for Density Estimation, NeurIPS 2017. These three are the references printed on Lecture 5, slide 84. |
| With theory 5 | D. P. Kingma and P. Dhariwal, Glow, NeurIPS 2018. |
| With theory 4.2–4.4 | J. Ho, X. Chen, A. Srinivas, Y. Duan, and P. Abbeel, Flow++, ICML 2019. |

## Slides and sources that were downloaded

From [https://dgm-sut-amini.github.io/](https://dgm-sut-amini.github.io/), saved in [`lectures/`](../lectures/):

| Site entry | Saved file | On the site |
|---|---|---|
| 0. Logistics | `DGM_Presentation-0_Logistics.pdf` | Published |
| 1. Introduction | `DGM_Presentation-1_Introduction.pdf` | Published. Not required for this homework. |
| 2. Foundation | `DGM_Presentation-2_Foundation.pdf` | Published. Required. |
| 3. Deep autoregressive models | `DGM_Presentation-3_Deep_Autoregressive_Models.pdf` | Published. Needed only for the MAF reminder. |
| 4. Variational autoencoders | `DGM_Presentation-4_Variational_Autoencoders.pdf` | Published. Required. |
| 5. Flow models | `DGM_Presentation-5_Flow_Models.pdf` | Published. Required. |
| 6. GANs | `DGM_Presentation-6_GAN.pdf` | Published. Later in the course. |
| 7. Energy-based models | `DGM_Presentation-7_Energy_Based_Models.pdf` | Published. Later in the course. |
| 8. Diffusion | `DGM_Presentation-8_Diffusion_Models.pdf` | The page marks this unavailable, and the filename in the page source 404s. The file with a space, `DGM_Presentation-8_Diffusion Models.pdf`, is on the server and was saved. Later in the course. |
| 9. Inverse problems | — | Marked unavailable. The URL 404s. |
| 10. Evaluation methods | — | Marked unavailable. The URL 404s. |

The course page also links these. They were not copied into the archive except where a file download succeeded:

- R. Salakhutdinov, Learning Deep Generative Models, Annual Review of Statistics and Its Application, 2015. [PDF](https://www.cs.cmu.edu/~rsalakhu/papers/annrev.pdf). A survey. Not a section-by-section source for this homework.
- Y. Song, Generative Modeling by Estimating Gradients of the Data Distribution, 2021. Score-based models, later in the course.
- Y. Song and D. P. Kingma, How to Train Your Energy-Based Models, 2021.
- A. Grover and S. Ermon, Tutorial on Deep Generative Models, IJCAI 2018.
- CVPR 2018 GAN tutorial, UAI 2017 tutorial by S. Mohamed and D. Rezende, and I. Goodfellow’s NeurIPS 2016 GAN tutorial.
- MIT RES.6-012 probability playlist, MIT 18.650 statistics playlist, Stanford CS229.

The Telegram copies of logistics, foundation, and the autoregressive lecture match this series. No separate VAE or flow deck was in that folder. Lecture 4 and Lecture 5 on this site are the VAE and flow lectures.
