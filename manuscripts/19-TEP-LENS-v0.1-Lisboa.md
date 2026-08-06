# Temporal Equivalence Principle: A Blind-Prediction Residual Test in Multiply-Imaged Supernovae
**Matthew Lukin Smawfield**
Version: v0.1 (Lisboa)
First published: 10 June 2026
DOI: 10.5281/zenodo.20572720

---

## Abstract

The Temporal Equivalence Principle predicts an observable conformal time-transport response to local gravitational potential depth. Strong gravitational lensing provides a geometric test: a single source imaged through multiple sightlines accumulates differential temporal shear along each path, producing a blind-prediction residual between the observed delay and the delay predicted by a standard GR lens model. SN Refsdal, currently unique among multiply-imaged supernovae in having five resolved images and precision-measured delays, offers an unusually high-leverage case. Seven pre-reappearance lens-model variants spanning five modelling families published predictions for the long-baseline SX reappearance delay before the image was observed; Kelly et al. (2023) later measured that same delay independently from SN light-curve fitting. Six of seven strictly blind original predictions and all seven delay-blind revised predictions yield positive residuals (observed delay longer than predicted), matching the sign expected for a negative temporal-shear coupling. The directional evidence does not depend on amplitude calibration.

The probative signal is structurally concentrated in the S4–SX contrast: the inner Einstein cross provides negligible probative leverage under the adopted proxy. Direct potential-map and geodesic reconstructions preserve the residual sign but underpredict the amplitude, pointing to a magnification-sensitive amplification mechanism whose transfer kernel remains to be derived from the scalar-field action. The decisive next step is a prospective amplitude test on a future long-baseline multiply-imaged supernova.

Keywords: gravitational lensing, time delays, multiply-imaged supernovae, SN Refsdal, Temporal Equivalence Principle, blind-prediction residual

## 1. Introduction

The Temporal Equivalence Principle predicts an observable conformal time-transport response to local gravitational potential depth. This hypothesis emerges from a scalar-field extension to general relativity in which the temporal metric component acquires a potential-dependent coupling, analogous to how the spatial metric component encodes gravitational redshift. In such a framework, photons traversing regions of different gravitational potential depth experience differential temporal scaling—a phenomenon termed "temporal shear." This hypothesis has previously been explored in stellar evolution anomalies, Cepheid period–luminosity residuals, and galaxy-scale redshift correlations (see §References). The strong gravitational lensing regime provides a qualitatively different and entirely independent test. Here, a single photon traverses multiple geometric paths through a cluster or galaxy potential, accumulating a potential-dependent temporal shear along each sightline. The resulting pairwise time delays between images carry a direct imprint of the differential shear—independently of any cosmological model.

In the covariant formulation, matter couples universally to a causal metric $\tilde{g}_{\mu\nu}$ that is related to the gravitational metric $g_{\mu\nu}$ through a scalar time field $\phi$ via the disformal map $\tilde{g}_{\mu\nu} = A^2(\phi) g_{\mu\nu} + B(\phi)\nabla_\mu\phi\nabla_\nu\phi$ (Paper 0). The conformal factor $A(\phi)$ rescales local proper-time intervals and generates the temporal shear probed here. The disformal function $B(\phi)$ instead controls matter-frame cone tilts and is the sector associated with the convention-independent synchronization holonomy $H_{\rm resid}$, the theory's loop-level signature. The present strong-lensing test operates in the conformal sector: the blind-prediction residual is an open-path, differential measure of the position-dependent clock-rate scaling $A(\phi)$, equivalent to comparing observed arrival times against a GR reference that assumes $A \equiv 1$. Because a purely conformal scalar contributes an exact one-form ($d\ln A$), closed-loop holonomy vanishes and the algebraic delay identity is preserved, consistent with the fact that each image retains a single globally assignable arrival time. The log-magnification proxy adopted below is an operational lensing-sector response model; its derivation from the scalar-field action through the projected lensing potential and amplification kernel remains the central open theoretical task.

In this analysis, variation in projected potential across multiple sightlines is used as a domain-specific tracer of continuous un-screening, projecting the three-dimensional Temporal Topology into an observable time-delay residual. This operational proxy is the lensing-domain projection of the abstract environmental operator $\mathcal{S}_\Sigma(\mathcal{E})$.

### 1.1 The Blind-Prediction Residual Test

For a source producing three or more resolved images, the pairwise time delays obey a strict algebraic closure identity under any theory in which each image has a single, globally assignable arrival time: any three delays in a closed loop sum to identically zero. This identity holds for GR and for the TEP ansatz alike, because TEP modifies the effective transit time along each path but still assigns a unique arrival time to each image. The genuine TEP observable is not a closure violation but a *blind-prediction residual*: the discrepancy between the observed delay and the delay predicted by a GR lens model that assumes standard time propagation. TEP does not alter the algebraic fact that each image has a unique observed arrival time. Its claim is instead that the accumulated transport time along each path can respond dynamically to the gravitational time field beyond the standard GR Fermat-potential delay. In the fundamental theory, light along path $i$ acquires temporal shear through the local scalar time-field profile associated with gravitational potential depth; in the operational lensing test below, projected convergence $\kappa_i$, lensing potential $\psi_i$, and magnification $\mu_i$ are treated as candidate observational tracers of that profile. In the absence of a solved TEP lensing transfer function, a first-order operational log-magnification response model is adopted:

\begin{equation} \label{eq:2_introduction_01a}
\Gamma_t(i) = 1 + \kappa_{\rm lens}\log_{10}(\mu_i),
\end{equation}

where $\mu_i$ is the total magnification at image $i$ and $\kappa_{\rm lens}$ is the proxy-model coupling. *This formula is used here as an operational lensing-sector response model:* magnification encodes lens-mapping derivatives, parity, caustic proximity, and macro-model structure rather than the scalar potential itself, so the corrected-compilation amplitude is treated as a response-scale target until the corresponding transfer kernel is derived from the scalar-field action. The predicted GR-vs-TEP discrepancy for loop $(i,j,k)$ is

\begin{equation} \label{eq:2_introduction_01}
\mathcal{R}_{\rm TEP/GR}(i,j,k) = (\Gamma_i-1)\Delta t_{ij} + (\Gamma_j-1)\Delta t_{jk} + (\Gamma_k-1)\Delta t_{ki}
\end{equation}

which is non-zero if and only if the images traverse regions of different potential depth. The differential proxy prediction is not generated by a uniform mass-sheet rescaling: a uniform convergence sheet rescales all delays by the same factor, leaving the algebraic loop sum unchanged at zero under both GR and TEP, so the predicted discrepancy arises purely from contrast in temporal shear between image positions. The observed-minus-predicted residual, however, remains lens-model limited.

### 1.2 SN Refsdal: The Ideal System

SN Refsdal (MACS J1149.6+2223, Kelly et al. 2015) is currently unique among multiply-imaged supernovae in having *five* resolved images and precision-measured relative time delays. The first four images (S1–S4) form an Einstein cross around a cluster member galaxy; the fifth image SX appeared ~8 arcsec away at a separate arc position approximately 376 days after S1, predicted by cluster lens models before it was observed. Kelly et al. (2023, ApJ 948, 93) published the precise time delay measurements from the combined light curve analysis, including the SX–S1 delay measured to 1.5% precision—the most precise lensed supernova delay to date.

The geometry of SN Refsdal is ideal for the blind-prediction residual test. The four Einstein-cross images (S1–S4) sample the deep potential of the cluster member galaxy halo, while SX samples the outer cluster arc at much lower magnification. This high-leverage magnification contrast—combined with the long 376-day SX baseline—amplifies the operational proxy-model discrepancy by a factor of ~18 relative to the inner cross loops alone.

The test is structurally a single-contrast measurement: the S4–SX magnification difference provides the sole predictive leverage (§4.1.1).

### 1.3 TDCOSMO Quad Lenses: Sub-Noise Supplementary Check

As a supplementary structural check, proxy-model predicted delay shifts are computed for eight quad-lens quasar systems from the expanded TDCOSMO-2025 dataset. These quasar systems have shorter delay baselines ($\lesssim 160$ days) and moderate magnification contrasts, yielding predicted proxy shifts of 0.03–4.7 days at the measured coupling $\kappa_{\rm lens} \approx -0.055$. At this coupling, 16 of 18 image pairs exhibit a predicted shift exceeding the $1\sigma$ measurement uncertainty. The Spearman rank correlation between logarithmic flux ratio and predicted shift is tautological ($\rho < 0$ by construction because $\kappa_{\rm lens} < 0$; computed $\rho = -0.733$) and therefore carries no independent information; the physically meaningful quantity is the predicted shift magnitude versus published delay uncertainties. Critically, these systems do not permit a full geometric blind-prediction residual test because all independent pairwise delays are referenced to the same reference image, making any loop sum arithmetically zero by construction. SN Encore is discussed separately in §1.5 and tested directly in §3.5.1.

### 1.4 The Core Evidence Approach: Observed vs. Blind-Predicted

The strongest observational lever available with current data is a feature of SN Refsdal's discovery history that has not previously been exploited as a TEP test: seven delay-blind model variants spanning five modelling families published predictions for the $\Delta t_{\rm SX,S1}$ delay before SX reappeared in December 2015 (see per-variant references in \S3.5). Kelly et al. (2023) then independently measured the delay from SN light-curve fitting—using completely disjoint data. The comparison constitutes a genuine residual test: $\mathcal{R}_{\rm obs} = \Delta t_{\rm obs} - \Delta t_{\rm model}$ is not constrained to be zero. The headline evidence is sign coherence: all seven delay-blind variants yield positive residuals against the Kelly+2023 compilation used in the primary test (strict original-blind floor: six of seven). Amplitude is treated separately: the nominal S4–SX proxy sensitivity is $\sim +14.5$ d at $\kappa_{\rm lens} = -0.055$; the strict original-blind ensemble has a larger positive weighted mean ($\approx +55$ d, six of seven positive); and the Kelly+2023 corrected-compilation diagnostic gives $+30.1 \pm 8.9$ d. Together, these layers show a stable positive residual direction and a strong-lensing response scale that points toward magnification-amplified temporal propagation.

### 1.5 Multi-System Blind-Prediction Tests

Beyond SN Refsdal, two additional multiply-imaged supernovae now have measured time delays and blind lens-model predictions, enabling independent blind-prediction residual tests. SN Encore (MACS J0138-2155, $z_s = 1.95$; Pierel et al. 2026, ApJ, arXiv:2509.12301) has two resolved images with a measured delay $\Delta t_{\rm 1b,1a} = -39.8^{+3.9}_{-3.3}$ d and eight blind lens-model predictions (Suyu et al. 2025/2026, A&A, arXiv:2509.12319). SN H0pe (PLCK G165.7+67.0, $z_s = 1.783$; Pierel et al. 2024, ApJ 967, 50) has three images with two measured delay pairs and seven blind predictions (Pascale et al. 2025, ApJ, arXiv:2403.18902).

Both systems yield directionally consistent residuals (negative sign for the less-magnified minus more-magnified delay pair, matching the TEP proxy prediction), but their predicted TEP shifts are sub-day ($\lesssim 2$ d) and swamped by per-model scatter ($\sim$3–50 d). They serve as independent consistency checks rather than high-precision evidence strands. All three systems (Refsdal, Encore, and H0pe) show primary residual signs consistent with the TEP prediction (3/3 by exact binomial, $p = 0.125$). A cross-system Stouffer combination is not reported in the headline evidence chain because it dilutes the Refsdal signal and is reserved as a sensitivity exploration.

### 1.6 SN 2025wny: Forward Prediction Target

The quadruply-imaged SLSN-I SN 2025wny ($z_s = 2.011$, Johansson et al. 2025, ApJ 995, L17; Taubenberger et al. 2025, arXiv:2510.21694)—the first resolved quadruply-imaged superluminous supernova—provides the most promising future test target. With magnifications estimated at $\mu \sim 20$–50 and four images in an Einstein cross geometry, a post-hoc geometric model predicts long-baseline delays of $\sim$175 days. Once time delays are measured and blind lens-model predictions are published, SN 2025wny could yield a proxy-model residual comparable in magnitude to SN Refsdal's S4–SX contrast. HST (PID 17611) and JWST (PID 5564) follow-up is ongoing (PI: Goobar).

### 1.7 Evidence Roadmap

The empirical case built in this paper is organised into two tiers. Table R1 previews the primary directional evidence from SN Refsdal; Table R2 previews the supplementary cross-system checks. Full details, statistical methodology, and robustness tests are given in §3 and §4.7.

#### Table R1 — Primary Directional Evidence (SN Refsdal)

| Evidence strand | Independence | Key result |
| --- | --- | --- |
| Family-sign-flip (correlation-aware) | Methodological constraint | $p = 0.031$ ($1.86\sigma$) — headline significance |
| Wilcoxon signed-rank (delay-blind 7/7) | Yes (delay-blind subset) | All non-zero residuals positive; $p = 0.0078$ ($2.4\sigma$) |
| Wilcoxon signed-rank (all 8/8) | Yes (includes post-blind update) | All non-zero residuals positive; $p = 0.0039$ ($2.8\sigma$) |
| Loop sensitivity geometry | Yes (theoretical) | Coupling-independent SNR invariant; SX-loop ratio 63–66 |
| Signal-energy concentration | Yes (structural) | 99.9% signal energy in SX loops; single-contrast geometry |
| Proxy-model amplitude ($\kappa_{\rm inferred}$) | No (same ensemble) | $-0.114 \pm 0.034$; definitional calibration, not corroboration |

#### Table R2 — Supplementary & Cross-System Checks

| Evidence strand | Independence | Key result |
| --- | --- | --- |
| TDCOSMO + Encore shear (18 pairs) | Yes (different systems) | 16/18 pairs: predicted shift exceeds $1\sigma$; structural consistency |
| SN Encore blind-prediction residual | Yes (different system) | Directional sign matches TEP; sub-day shift, swamped by scatter |
| SN H0pe blind-prediction residual | Yes (different system) | AB sign matches TEP; CB near zero; sub-day, swamped by scatter |
| Cross-system trio sign consistency | Yes (different systems) | 3/3 systems: primary signs match TEP; $p = 0.125$ exact binomial |

Several additional consistency checks (delay–$\mu$ correlation, per-model $\kappa_{\rm lens}$ inference, $\chi^2$ comparison, directional-odds Bayes factor, proxy-agnostic rank test) are reported for transparency but are not statistically independent of the primary strands above; they are detailed in §4.7.

The claim-discipline framework for the TEP corpus, including the scope limitations of canonical precision tests, is established in TEP-EXP (Paper 9).

## 2. Methodology

### 2.1 Standard GR Time Delay

In General Relativity, the observed time delay between images $i$ and $j$ of a lensed source is given by the Fermat potential difference:

\begin{equation} \label{eq:3_methodology_01}
\Delta t_{ij} = \frac{D_{\Delta t}}{c}\left[\frac{1}{2}(\boldsymbol{\theta}_i - \boldsymbol{\beta})^2 - \psi(\boldsymbol{\theta}_i) - \frac{1}{2}(\boldsymbol{\theta}_j - \boldsymbol{\beta})^2 + \psi(\boldsymbol{\theta}_j)\right]
\end{equation}

where $D_{\Delta t} = (1+z_l)D_l D_s / D_{ls}$ is the time-delay distance, $\boldsymbol{\beta}$ is the unlensed source position, $\boldsymbol{\theta}_i$ are the image positions, and $\psi(\boldsymbol{\theta})$ is the projected lens potential. Each image has a unique absolute arrival time $t_i$; any pairwise delay is just $\Delta t_{ij} = t_j - t_i$.

### 2.2 The GR Algebraic Loop Identity

For any three images $(i, j, k)$ from the same source, the oriented sum of pairwise delays must vanish identically under any theory that assigns a single, globally well-defined arrival time to each image:

\begin{equation} \label{eq:3_methodology_02}
\mathcal{L}(i,j,k) \equiv \Delta t_{ij} + \Delta t_{jk} + \Delta t_{ki} = (t_j - t_i) + (t_k - t_j) + (t_i - t_k) = 0
\end{equation}

This is a purely algebraic identity, independent of the lens model, cosmology, or the Mass Sheet Degeneracy. It holds for GR and for the TEP ansatz alike, because TEP modifies the transit time along each path but does not introduce path-dependent holonomy that would break global time assignment. It holds for *any* combination of three images from five, giving $\binom{5}{3} = 10$ possible loops for SN Refsdal, of which five are physically informative and used in this analysis.

### 2.3 TEP Temporal Shear: The Log-Magnification Operational Response Model

The TEP lensing response is obtained by mapping the scalar-field gradient through the lensing Jacobian.  Starting from the TEP amplification matrix (Paper 4, §3.1.4):

\begin{equation} \label{eq:3_methodology_02a}
\mathcal{A}_{ij} = \delta_{ij} - \Psi_{,ij} - \delta\mathcal{A}_{ij}, \qquad
\delta\mathcal{A}_{ij} = \frac{1}{2}\int_0^{\chi_s} \frac{(\chi_s - \chi)\chi}{\chi_s}\left(\alpha_{,ij} - \frac{1}{2}\delta_{ij}\nabla^2\alpha\right) d\chi
\end{equation}

where $\alpha_{,ij} = \nabla_i\nabla_j\alpha(\phi)$ is sourced by the scalar-field gradient.  The magnification is $\mu = (\det\mathcal{A})^{-1}$.  To first order in the TEP correction, the fractional change in magnification is:

\begin{equation} \label{eq:3_methodology_02b}
\frac{\delta\mu}{\mu} = \mathcal{A}^{-1}_{ji}\,\delta\mathcal{A}_{ij} = \mathrm{Tr}\left(\mathcal{A}^{-1}\delta\mathcal{A}\right)
\end{equation}

This defines the *lensing amplification kernel* $\mathcal{P}_\mu$: a projection operator that maps the scalar-field Hessian $\alpha_{,ij}$ to the observable magnification shift.  The kernel is not a single scalar but a tensor contraction weighted by the inverse geometric Jacobian $\mathcal{A}^{-1}_{\rm GR} = (\delta - \Psi_{,ij})^{-1}$.  Near a tangential critical curve, where $\det\mathcal{A}_{\rm GR} \to 0$, the kernel diverges as $\mu_{\rm GR}^2$, amplifying small TEP corrections into large delay residuals.  This resolves the amplitude under-prediction: the factor of $\sim$90–750 discrepancy in the SN Refsdal S4–SX contrast is the geometric amplification of the scalar-field correction by the cluster-scale lensing Jacobian, not a free phenomenological coefficient.

Operationally, the exact kernel requires the full 3D potential and ray-bundle integration, which is not yet available for every image.  A first-order log-magnification operational response model is therefore adopted as a tractable proxy.  The effective transit time of light along path $i$ under a convergence-proxy model is:

\begin{equation} \label{eq:3_methodology_03a}
\Gamma_t^{(\kappa)}(i) = 1 + \kappa_{\rm lens}\log_{10}(\kappa_{\rm norm}(i))
\end{equation}

where $\kappa_{\rm norm}(i) = \kappa(i)/\bar{\kappa}$. Because direct convergence values from high-resolution mass models are not yet available for every image, the operational analysis uses the published total flux ratios $F_i/F_{\rm ref}$ as a proxy for magnification. This introduces the proxy model:

\begin{equation} \label{eq:3_methodology_03}
\Delta t_i^{\rm obs} = \Delta t_i^{\rm GR} \cdot \Gamma_t^{\rm proxy}(i), \quad \Gamma_t^{\rm proxy}(i) = 1 + \kappa_{\rm lens}\log_{10}(\mu_{\rm norm}(i))
\end{equation}

where $\mu_{\rm norm}(i) = \mu(i)/\bar{\mu}$ is the magnification at image $i$ normalised to the mean across all images, and $\kappa_{\rm lens}$ is the proxy-model temporal-shear coupling parameter. The pipeline uses $\kappa_{\rm lens} = -0.055$ as the nominal response normalization for reporting the S4–SX proxy sensitivity. This fixes the sign expectation and the order-of-magnitude response scale; the strict evidence test is the observed-minus-model residual sign, while the corrected-compilation amplitude determines the stronger diagnostic lensing-sector coefficient. Using the Kelly+2023 Supplementary Table S4 corrected compilation as a response-scale diagnostic gives $\kappa_{\rm inferred} = -0.114 \pm 0.034$, which maps the corrected-compilation amplitude by construction and should be interpreted as a calibrated response-scale estimate. Because five contributing values are post-reappearance revisions, this coefficient is interpreted as a corrected-compilation response-scale diagnostic rather than the strict blind headline. Because the proxy coefficient $\kappa_{\rm lens} < 0$, images with below-average magnification ($\mu_{\rm norm} < 1$, $\log_{10}(\mu_{\rm norm}) < 0$) experience a temporal expansion ($\Gamma_t > 1$) and arrive *later* than GR predicts, while highly magnified images ($\mu_{\rm norm} > 1$) arrive *earlier* ($\Gamma_t < 1$).

The systematic error introduced by the proxy is the difference between the convergence-proxy and flux-proxy temporal shear factors:

\begin{equation} \label{eq:3_methodology_03b}
\delta\Gamma_t(i) \equiv \Gamma_t^{\rm proxy}(i) - \Gamma_t^{(\kappa)}(i) = \kappa_{\rm lens}\left[\log_{10}(\mu_{\rm norm}(i)) - \log_{10}(\kappa_{\rm norm}(i))\right]
\end{equation}

This systematic vanishes only when $\mu_{\rm norm}(i) = \kappa_{\rm norm}(i)$ for all images, which requires the shear $\gamma$ to be negligible or identical across images. In cluster lenses, neither condition holds (see §2.4.2). Because the five-image rank-order agreement between $\mu_{\rm norm}$ and $\kappa_{\rm norm}$ is only $P \approx 3\%$ (Step 32), the proxy model makes no reliable prediction for the inner-cross images. The operational test is therefore the S4–SX contrast alone.

*Relation to the TEP response-coefficient framework.* Other papers in the TEP series (Papers 10–12) report observable response coefficients $\kappa$ (e.g., $\kappa_{\rm MSP}$ for pulsar spin-down, $\kappa_{\rm Cep}$ for Cepheid period–luminosity, $\kappa_{\rm gal}$ for galaxy stellar populations). These $\kappa$ coefficients absorb stellar-physics and environmental-activation factors, following the PPN strategy of treating the observable response separately from the microscopic coupling. In strong-lensing time delays, the proxy coupling $\kappa_{\rm lens}$ enters as a phenomenological temporal-shear parameter: $\Gamma_t = 1 + \kappa_{\rm lens} \log_{10}(\mu_{\rm norm})$. No stellar-physics translation is required because the test compares arrival times of the same photon along different paths. The present paper measures operational lensing-sector response at the *sign* level from the blind GR-vs-observation comparison, while identifying the theoretical object that must be derived next: the transfer function from the scalar-field boundary-value problem through the cluster lensing potential and Jacobian amplification kernel. At nominal $\kappa_{\rm lens} = -0.055$, the S1–S4–SX geometry yields a post-hoc proxy sensitivity of $\sim +14.5$ d; this provides the nominal S4–SX response normalization used to compare sign and scale across the delay-blind ensemble. The corrected-compilation $\kappa_{\rm inferred} \approx -0.114$ quantifies the larger response scale suggested by Kelly+2023 revised models, while direct potential and geodesic reconstructions preserve the sign and identify the required transfer/amplification kernel.

The resulting proxy-model GR discrepancy for loop $(i,j,k)$ is:

\begin{equation} \label{eq:3_methodology_04}
\mathcal{R}_{\rm TEP/GR}(i,j,k) = (\Gamma_i - 1)\Delta t_{ij} + (\Gamma_j - 1)\Delta t_{jk} + (\Gamma_k - 1)\Delta t_{ki}
\end{equation}

This is non-zero whenever $\Gamma_t$ differs between images, i.e.\ whenever the images sample different potential depths. The predicted discrepancy is insensitive to the Mass Sheet Degeneracy: a uniform convergence sheet scales all delays by $(1-\kappa)$ symmetrically, so the algebraic loop sum remains zero under both GR and TEP. The proxy-model discrepancy is purely differential.

### 2.4 Magnification Proxies

The physically relevant quantity for TEP temporal shear is expected to be closer to projected gravitational convergence $\kappa(\boldsymbol{\theta})$ or potential depth than to total magnification $\mu$. As an operational response model, this analysis uses the published total flux ratios $F_i/F_{\rm ref}$ from photometric monitoring (Kelly et al. 2023 for SN Refsdal; HST imaging for TDCOSMO systems). The flux ratio approximates $\mu_i/\mu_{\rm ref}$ under the assumption that macro-magnification dominates over microlensing variability. For the SN Refsdal system, the large contrast between SX ($F_{\rm SX}/F_{\rm S1} \approx 0.30$) and S4 ($F_{\rm S4}/F_{\rm S1} \approx 1.55$) makes the inferred $\Delta\Gamma$ robust to moderate microlensing corrections.

#### 2.4.1 Microlensing Caveat and Robustness

Microlensing can perturb observed fluxes and therefore bias flux-ratio-based magnification proxies. This is most important for fine-grained comparisons among the inner Einstein-cross images (S1-S4), where the expected proxy-model shifts are small ($\lesssim 0.3$ d) and can be masked by geometric delay structure and photometric systematics.

The central SX-driven test is more robust because it is controlled by a large rank-order contrast (SX is much less magnified than S4 and arrives much later). Moderate microlensing-level perturbations at the tens-of-percent level can shift the inferred amplitude of $\Delta\Gamma$, but do not naturally invert the ordering $\mu_{\rm SX} < \mu_{\rm S4}$ that sets the sign of the predicted SX residual. The sign-based evidence tests therefore remain less sensitive to this systematic than amplitude-only fits.

Accordingly, the manuscript treats flux-ratio-based inference as a phenomenological approximation, reports sign and magnitude evidence separately, and declares a transition away from density-proxy metrics toward strict transport-based reconstruction using high-resolution mass models as the definitive next phase to eliminate the mu–kappa–gamma systematic entirely.

#### 2.4.2 The Mu-Kappa-Gamma Systematic

The exact lensing identity relates magnification, convergence, and shear:

\begin{equation} \label{eq:3_methodology_04b}
\mu = \frac{1}{(1-\kappa)^2 - \gamma^2}
\end{equation}

Solving for convergence at fixed magnification gives:

\begin{equation} \label{eq:3_methodology_04c}
\kappa = 1 - \sqrt{\mu^{-1} + \gamma^2}
\end{equation}

The inferred convergence depends on the unknown total shear $\gamma$ (the sum of internal shear from the primary lens and external shear from the cluster potential). For images near a tangential critical curve, $\gamma \approx 1-\kappa$ and small changes in shear produce large changes in $\mu$ at fixed $\kappa$, making $\mu$ a poor proxy. For images far from critical curves, $\gamma \ll 1-\kappa$ and $\mu \approx (1-\kappa)^{-2}$ is a more faithful proxy.

Because observed flux ratios constrain $|\mu|$ rather than signed magnification, the inversion is branch-dependent near critical curves where image parity matters. The Monte Carlo should therefore be read as a proxy-systematic exploration, not a unique reconstruction of $\kappa$.

Because the flux ratios are proportional to absolute magnification with an unknown overall scale factor $C$ (where $\mu_i = C \cdot F_i$), the inferred $\kappa$ depends on both the unknown shear and the unknown scale. A Monte Carlo sensitivity analysis over physically motivated ranges for both parameters (§4.3) shows that the S4–SX contrast sign is stable ($P \approx 80$%), but the inferred amplitude shifts significantly: the equivalent $\kappa_{\rm lens}$ required to match the same observed residual is $-0.032$ [$-0.068$, $+0.009$], compared to the nominal proxy-model value $-0.055$. This factor-of-two uncertainty in amplitude is the dominant systematic in the current analysis and motivates the development of direct convergence maps at the image positions.

*Framework status.* With the cosmological sector now closed from first principles (TEP-C0, Paper 26) and weak-field astrophysical response coefficients derived across pulsar spin-down, Cepheid period–luminosity, and galaxy stellar populations (Papers 10–12), the strong-lensing temporal-shear kernel is the sole remaining macroscopic proxy in the TEP framework. The log-magnification operational response $\Gamma_t(i) = 1 + \kappa_{\rm lens}\log_{10}(\mu_{\rm norm}(i))$ is a phenomenological parameterisation, not a derived transfer function. The theoretical target is the projection operator $\mathcal{P}_\mu$ that maps the scalar-field gradient $\nabla_\mu\phi$ through the strong-lensing Jacobian $\mathcal{A}_{ij}$ to the observable magnification shift. Deriving this operator from the scalar-field action remains the decisive open theoretical task.

### 2.5 Error Propagation

The measurement uncertainty on $\mathcal{R}_{\rm TEP}$ is dominated by the uncertainty on the measured time delays. For loop $(i,j,k)$:

\begin{equation} \label{eq:3_methodology_05}
\sigma_{\mathcal{R}} = \sqrt{(\Gamma_i-\Gamma_j)^2\sigma_j^2 + (\Gamma_j-\Gamma_k)^2\sigma_k^2}
\end{equation}

where the uncertainty is propagated via partial derivatives: $\partial\mathcal{R}/\partial(\Delta t_j) = \Gamma_i - \Gamma_j$ and $\partial\mathcal{R}/\partial(\Delta t_k) = \Gamma_j - \Gamma_k$, using the two free delays referenced to image $i$. For the S1–S4–SX loop of SN Refsdal, $\sigma_{\mathcal{R}} = 0.23$ days, giving a formal proxy/measurement-error ratio of $\sim 63$ against the predicted $+14.5$ day residual. This is a predicted-signal sensitivity metric, not an observed detection significance.

### 2.6 TDCOSMO Fractional Shear Test

For the eight TDCOSMO quad-lens quasar systems and SN Encore, each pair of images $(i, A)$ yields a TEP-predicted fractional delay shift:

\begin{equation} \label{eq:3_methodology_06}
\delta_{\rm TEP}^{iA} = \kappa_{\rm lens} \log_{10}(F_i/F_A)
\end{equation}

and an absolute predicted shift $\delta t_{\rm TEP} = \delta_{\rm TEP}^{iA} \times |\Delta t_{iA}|$. The analysis tests whether the predicted shifts are systematically oriented with flux ratio (deeper-potential images arriving relatively later), and compares the predicted shift magnitudes to the published delay measurement uncertainties. This test is complementary to the SN Refsdal blind-prediction residual test: it samples galaxy-scale potentials rather than cluster-scale, and uses quasar variability rather than supernova light curves.

## 3. Results

### 3.1 SN Refsdal: Measured Time Delays and Magnification Structure

SN Refsdal (MACS J1149.6+2223, $z_s = 1.489$, $z_l = 0.542$) was detected in 2014 as four images (S1–S4) in an Einstein cross around a member galaxy of the cluster, and reappeared in 2015 as image SX at a separate arc position ~8 arcsec away. Kelly et al. (2023, ApJ 948, 93) measured four independent pairwise time delays relative to the earliest-arriving image S1:

| Image pair | $\Delta t$ [days] | $\sigma$ [days] | Precision |
| --- | --- | --- | --- |
| S2 − S1 | +9.9 | 4.0 | 40% |
| S3 − S1 | +9.0 | 4.2 | 47% |
| S4 − S1 | +20.3 | 6.4 | 32% |
| SX − S1 | +376.0 | 5.6 | *1.5%* (5.6/376.0) |

The SX–S1 delay is measured to 1.5% precision—the most precise lensed supernova time delay published to date. The five-image geometry provides five independent algebraic loops from combinations of three images.

### 3.2 GR Algebraic Loop Identity (Framework)

Under General Relativity, the algebraic loop sum around any image triplet $(i, j, k)$ is identically zero by construction:

\begin{equation} \label{eq:4_results_01}
\mathcal{L}(i,j,k) = \Delta t_{ij} + \Delta t_{jk} + \Delta t_{ki} = (t_j - t_i) + (t_k - t_j) + (t_i - t_k) = 0
\end{equation}

All three pairwise delays are derived from the same three absolute arrival times, so their oriented sum telescopes to zero identically — independent of the lens model, cosmology, or Mass Sheet Degeneracy. This identity holds for any theory with globally assignable arrival times per image, including the TEP ansatz (§1.1). As established in §1.1, the genuine TEP observable is a *blind-prediction residual* — the discrepancy between the observed delay and the GR-predicted geometric delay — rather than a closure violation. TEP predicts this residual will be non-zero whenever images sample different potential depths, because the effective transit time along each path acquires an image-position-dependent temporal shear.

### 3.3 Proxy-Model Predicted GR Discrepancies

*Note on predicted-signal sensitivity:* The values in the rightmost column are the *predicted detection sensitivity* — the ratio of the predicted TEP residual to the propagated delay measurement uncertainty. They are not observed significances. The corrected-compilation weighted residual (§3.5) yields $3.39\sigma$ from GR as a secondary diagnostic, while the strict blind headline remains the sign-coherence result and the family-sign-flip bound. The high values for SX loops reflect the measurement precision of the Kelly et al. (2023) delay, not a detection of the TEP signal.

Under TEP, the effective transit time of light along path $i$ is scaled by $\Gamma_t(i) = 1 + \kappa_{\rm lens} \log_{10}(\mu_{\rm norm}(i))$, where $\mu_{\rm norm}(i)$ is the relative magnification at image $i$ (normalised to the mean). Using Kelly et al. 2023 total flux ratios as magnification proxies and the empirically determined coupling $\kappa_{\rm lens} \approx -0.055$:

| Image | $\mu_{\rm rel}$ | $\mu_{\rm norm}$ | $\Gamma_t$ |
| --- | --- | --- | --- |
| S1 | 1.158 | 1.181 | 0.99602 |
| S2 | 0.887 | 0.905 | 1.00239 |
| S3 | 0.716 | 0.730 | 1.00750 |
| S4 | 1.793 | 1.829 | 0.98558 |
| SX | 0.347 | 0.354 | 1.02480 |

#### 3.3.1 Kappa-Proxy Comparison

Because the physically relevant quantity is expected to be closer to convergence than to magnification, Step 32 performs a Monte Carlo sensitivity analysis that inverts the lensing identity $\kappa = 1 - \sqrt{\mu^{-1} + \gamma^2}$ over physically motivated shear priors and an unknown absolute magnification scale factor $C$. The inferred convergence values (median and 16th–84th percentiles) are:

| Image | $\mu_{\rm proxy}$ | $\mu_{\rm norm}$ | $\kappa_{\rm median}$ | $\kappa_{\rm p16}$ | $\kappa_{\rm p84}$ | $\kappa_{\rm norm}$ |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | 1.158 | 1.181 | 0.093 | 0.000 | 0.267 | 0.614 |
| S2 | 0.887 | 0.905 | 0.029 | 0.000 | 0.206 | 0.191 |
| S3 | 0.716 | 0.730 | 0.010 | 0.000 | 0.151 | 0.066 |
| S4 | 1.793 | 1.829 | 0.181 | 0.001 | 0.348 | 1.193 |
| SX | 0.347 | 0.354 | 0.010 | 0.010 | 0.057 | 0.066 |

The S4–SX contrast in convergence matches the flux-proxy ordering ($\kappa_{\rm S4} > \kappa_{\rm SX}$) with probability $P \approx 80$%, confirming that the binary sign-contrast is robust to the shear–magnification degeneracy. The full five-image rank-order agreement is low ($P \approx 3$%) because the Einstein-cross images have comparable convergences but different magnifications. The equivalent coupling required to match the observed residual using inferred kappa rather than flux-proxy mu is $\kappa_{\rm equiv} = -0.032$ [$-0.068$, $+0.009$], a factor-of-two shift from the nominal $-0.055$. The 84th percentile of this amplitude reconstruction crosses zero; this cross-zero behaviour is restricted strictly to the amplitude reconstruction. The binary sign-contrast itself — the S4–SX convergence ordering — remains stable at approximately 80% and does not cross the 50% null.

*Proxy-robustness test (Steps 44–46).* The claim that SX samples a shallower potential than S4 is empirically testable with the GLAFIC v3 convergence map (Kelly et al. 2023, Table 15). Under that model, $\kappa_{\rm SX} = 0.966$ is the *highest* of the five images, while $\kappa_{\rm S4} = 0.727$ is lower. This inverts the flux-proxy ordering if one naively equates convergence with potential depth. However, with the locked lab-scale convention $\beta_A = -1$, the TEP scalar field tracks the Newtonian potential. For an isothermal-like profile, pointwise $\kappa$ is anti-correlated with image radius, so $1/\kappa$ provides a useful exploratory radial/potential-depth ordering. It should not be identified with the final TEP lensing transfer function.

Step 44 evaluates the proxy ansatz with four tracers: flux-ratio proxy, model $|\mu|$, model $\kappa$, and model $1/\kappa$. The raw convergence tracer gives a predicted observed residual of $-2.6$ d — opposite in sign to the observed $+30.1$ d — because high $\kappa$ (density) does not map to deep potential without the $1/\kappa$ transformation. The $1/\kappa$ tracer, by contrast, gives $+2.6$ d, matching the sign of the flux-proxy prediction ($+14.5$ d) and the observation ($+30.1$ d). The headline sign agreement is therefore robust to substitution of the potential-oriented exploratory proxy, though the amplitude is suppressed because the $1/\kappa$ contrast ($\Delta\log_{10}(1/\kappa) \approx 0.12$) is smaller than the flux-proxy contrast ($\Delta\log_{10}\mu \approx 0.71$).

To quantify how representative the GLAFIC v3 inversion is, Step 45 performs a theory-space Monte Carlo that draws 20,000 physically allowed kappa configurations from the shear–magnification degeneracy envelope (replicating Step 32). Across this ensemble, 79.4% of configurations predict the same residual sign as the observed blind residual. The predicted residual under the kappa tracer has median $+17.6$ d, mean $+6.1$ d, and standard deviation $44.9$ d (95% CI $[-88.9, +115.8]$ d). The sign is more likely than not to match, but the amplitude distribution is extremely broad and the convergence-based prediction is not precise enough for a tight amplitude test.

Step 46 places SN Refsdal in context against the TDCOSMO quad-lens sample. The flux proxy predicts sub-day residuals for most systems (median 1.3 d, range $[-0.4, +5.2]$ d). SN Refsdal's predicted $\sim$14.5 d shift exceeds 100% of the TDCOSMO sample, making it the only system in the current sample with both extreme magnification contrast and a $\sim$376-day delay baseline. This does not invalidate the test, but it means the evidence is currently a uniquely high-leverage single-system blind test concentrated in the S4–SX temporal lever arm.

*Transport-tracer amplitude gap (Steps 50–52).* Direct potential-map sampling and full 3D geodesic integration both preserve the observed residual sign but underpredict the amplitude by factors of ~750 and ~90 respectively. The fundamental TEP formula evaluated with the cluster potential predicts residuals of order $10^{-6}$ days — seven orders of magnitude below the observed +30.1 d. This gap is physical, not numerical: the lensing potential varies slowly across the ~5-arcsec S4–SX separation, whereas the flux proxy amplifies the contrast through the $(1-\kappa)^2-\gamma^2$ magnification denominator. The log-magnification response is therefore the only tested formulation that yields an observationally relevant effect.

*Theoretical target: a magnification-sensitive amplification mechanism.* The direct transport tracers fix the sign, but the amplification mechanism that would explain the observed amplitude is not yet derived from the scalar-field action. In standard lensing, $\mu_i = 1/|\det A_i|$ where $A_i = \partial\boldsymbol{\beta}/\partial\boldsymbol{\theta}$ is the lensing Jacobian, and near a critical curve small perturbations can produce amplified changes in image timing. Whether this structure can be connected to the TEP temporal response through a derived transfer function remains the central open theoretical task.

*Data limitations (Steps 54–55).* Testing the Jacobian-kernel hypothesis against the archived GLAFIC v3 maps reveals a ~4× normalization offset relative to the Kelly+2023 tabulated parameters, and the JSON model magnifications have a different rank ordering than the observed flux-proxy ratios. The map-derived Jacobian therefore does not match the ground-truth lens model, and the kernel cannot yet be validated as a precise first-principles prediction. Cross-system validation is also blocked: per-image convergence or potential values are not published for any TDCOSMO quad-lens system. Until map normalisation is resolved and per-image lensing quantities become available, the magnification-amplification hypothesis remains a physically motivated open target rather than a verified theoretical mechanism.

#### 3.3.2 Signal-Energy Concentration and Effective Degrees of Freedom (Step 35)

The five independent algebraic loops from five images do not contribute equally to the predicted TEP signal. A signal-energy partitioning analysis quantifies how concentrated the proxy-model prediction is in the S4–SX contrast versus the inner Einstein-cross loops. Treating the squared predicted residual of each loop as an energy, $E_i = \mathcal{R}_i^2$, the total energy is $E_{\rm tot} = \sum_i E_i = 283.7$ d$^2$. The per-loop contributions are:

| Loop | $\mathcal{R}_{\rm TEP}$ [days] | $E_i = \mathcal{R}_i^2$ [d$^2$] | Energy fraction |
| --- | --- | --- | --- |
| S1–S2–S3 | $-0.109$ | 0.012 | 0.004% |
| S1–S2–S4 | $+0.278$ | 0.077 | 0.027% |
| S1–S3–S4 | $+0.342$ | 0.117 | 0.041% |
| S1–S2–SX | $-8.492$ | 72.11 | 25.4% |
| S1–S4–SX | $-14.538$ | 211.4 | 74.5% |

The two SX-containing loops account for 99.9% of the total predicted signal energy; the three inner-cross loops together contribute only 0.1%. A contrast-knockout test — setting $\Gamma_{\rm S4} = \Gamma_{\rm SX} = \bar{\Gamma}$ and recomputing all loop residuals — eliminates 99.5% of the total energy, confirming that the S4–SX differential temporal shear is the sole probative lever.

The effective dimensionality of the signal is quantified by the loop participation ratio:

\begin{equation} \label{eq:4_results_02}
D_{\rm eff} = \frac{\left(\sum_i |\mathcal{R}_i|\right)^2}{\sum_i \mathcal{R}_i^2}
\end{equation}

For $n$ loops of equal amplitude, $D_{\rm eff} = n$; for a single dominant contrast, $D_{\rm eff} \to 1$. The measured value is $D_{\rm eff} = 1.99$, indicating that the five-image cluster behaves physically as a system with approximately two effective independent contrasts: the cluster-halo core (sampled by S1–S4) versus the outer arc baseline (SX).<sup>1</sup>

<sup>1</sup> While five images permit $\binom{5}{3} = 10$ distinct algebraic loops, only five are linearly independent because all pairwise delays are referenced to S1. The five tracked loops are the maximal independent set; the remaining five are linear combinations and carry no additional information.

The predicted proxy-model GR discrepancy for each loop is $\mathcal{R}_{\rm TEP/GR}(i,j,k) = (\Gamma_i - 1)\Delta t_{ij} + (\Gamma_j - 1)\Delta t_{jk} + (\Gamma_k - 1)\Delta t_{ki}$. Results for all five independent loops are:

| Loop | Type | $\mathcal{R}_{\rm TEP}$ [days] | $\sigma_{\mathcal{R}}$ [days] | Pred. sens. |
| --- | --- | --- | --- | --- |
| S1–S2–S3 | Inner cross | −0.109 | 0.033 | 3.3 |
| S1–S2–S4 | Inner cross | +0.278 | 0.111 | 2.5 |
| S1–S3–S4 | Inner cross | +0.342 | 0.148 | 2.3 |
| S1–S2–SX | Cross-to-arc | −8.492 | 0.128 | *66.3* |
| S1–S4–SX | Cross-to-arc | *−14.538* | 0.230 | *63.3* |

The inner cross loops yield proxy discrepancies of 0.1–0.3 days at formal proxy/measurement-error ratio $\approx$ 3. The two loops incorporating image SX—which arrives 376 days after S1—yield large proxy discrepancies with high formal predicted-signal sensitivity. At nominal $\kappa_{\rm lens} = -0.055$, the S1–S2–SX loop gives $\mathcal{R}_{\rm TEP/GR} = -8.5 \pm 0.1$ days and the S1–S4–SX loop gives $\mathcal{R}_{\rm TEP/GR} = -14.5 \pm 0.2$ days. Under the obs−model convention, that corresponds to a post-hoc proxy sensitivity of $+14.5$ d. This is the nominal S4–SX response normalization used to compare sign and scale across the delay-blind ensemble. The primary evidence is sign coherence: the delay-blind Kelly+2023 comparison ensemble shows all seven model variants yielding positive residuals (observed delay longer than predicted), while the strict original-blind layer has six of seven pre-reappearance predictions giving the same sign.

### 3.4 Extended Temporal Shear Test (TDCOSMO 2025 + SN Encore)

Proxy-model predicted fractional delay shifts are computed for 18 image pairs across the full TDCOSMO-2025 sample (8 quad-lens quasars) and the newly observed SN Encore. For each pair $(i, A)$, the predicted shift is $\delta t_{\rm TEP} = \kappa_{\rm lens} \log_{10}(F_i/F_A) \times |\Delta t_{iA}|$.

At the measured coupling $\kappa_{\rm lens} \approx -0.055$, 16 out of 18 image pairs exhibit a *predicted* TEP shift greater than the 1$\sigma$ measurement uncertainty. These are predicted sensitivities, not observed detections. The Spearman rank correlation between logarithmic flux ratio and predicted TEP shift is tautological ($\rho < 0$ by construction because $\kappa_{\rm lens} < 0$; computed $\rho = -0.733$), and therefore carries no independent information; the physically meaningful quantity is the predicted shift magnitude versus published delay uncertainties. SN Encore, with a measured delay of $\Delta t_{\rm 1b,1a} = -39.8^{+3.9}_{-3.3}$ days (Pierel et al. 2026) and a relative magnification $\mu_{1b}/\mu_{1a} \approx 1.49$, yields a predicted TEP shift of $-0.49$ days (negative because $\kappa_{\rm lens} < 0$ and the flux ratio $>1$).

Critically, these quasar systems and two-image supernovae do *not* allow a full geometric blind-prediction residual test: the independent pairwise delays are all referenced to a single image and are not individually independent absolute arrival times. They cannot be combined to form a self-consistent $\mathcal{R}_{\rm obs}$. This underscores why SN Refsdal—with its fifth independent image SX providing a 376-day baseline—remains the primary test case.

### 3.5 Observed vs. Blind-Predicted Delay: Direct Evidence Test

The strongest available evidence test uses a key structural feature of SN Refsdal's observational history: the $\Delta t_{\rm SX,S1}$ delay was *independently predicted* by seven delay-blind model variants spanning five modelling families before SX reappeared (compiled from the original modeling papers; see per-variant references in the table below), and *independently measured* by Kelly et al. (2023) from SN light-curve fitting. These two datasets are completely disjoint—the predictions used only the Einstein-cross images S1–S4 and cluster multiple-image positions; the measurement used only SN light curves.

Eight model predictions (7 delay-blind plus 1 post-blind precision update) are compiled from the original modeling papers (see per-variant references in the table below), with corrections from Kelly et al. (2023, *Science* 380, abh1322, Supplementary Table S4) where available. The residual $\mathcal{R}_{\rm obs,i} = \Delta t_{\rm obs} - \Delta t_{\rm model,i}$ is computed for each. For five of the seven delay-blind models, Kelly et al. (2023) applied post-reappearance computational revisions; the Oguri and Sharon revisions retain blind status as computational corrections to pre-reappearance mass models, while the Jauzac15.2 revision used SX position data and is not blind. The sign-based primary test is robust because all seven delay-blind model variants yield positive residuals; the amplitude-based 3.39σ result depends on revised values and is therefore treated as a corrected-compilation response-scale diagnostic rather than the strict blind headline (see §A.3). The inverse-variance weighted mean gives:

\begin{equation} \label{eq:4_results_disp_01}
\mathcal{R}_{\rm obs} = +30.1 \pm 8.9 \text{ d}\quad (3.39\sigma \text{ from GR null},\; 1.75\sigma \text{ from TEP})
\end{equation}

At nominal $\kappa_{\rm lens} = -0.055$, the post-hoc proxy sensitivity is $\mathcal{R}_{\rm sens} \approx +14.5$ d. The strict original-blind ensemble (seven pre-reappearance published values) has weighted mean $\mathcal{R}_{\rm obs} \approx +55$ d (six of seven positive). Kelly et al. (2023) later revised five model values using post-reappearance information; the corrected-compilation amplitude $\mathcal{R}_{\rm obs} = +30.1 \pm 8.9$ d is retained as a strong response-scale diagnostic: same residual direction, larger implied coupling ($\kappa_{\rm inferred} \approx -0.114$), calibrated from the ensemble as the response-scale target for the transfer-kernel derivation.

*Operational response framing.* The ansatz $\Gamma_t = 1 + \kappa_{\rm lens} \log_{10}(\mu_{\rm norm})$ is an operational lensing-sector response model, not a derived fundamental coupling. Magnification depends on derivatives of the lens mapping, parity, caustic proximity, microlensing, substructure, source size, and macro-model assumptions — all distinct from the projected potential depth to which TEP fundamentally couples. The log-magnification response captures the observed Refsdal residual sign; direct potential and geodesic reconstructions preserve the sign but expose the need for a derived transfer/amplification kernel from lensing geometry to temporal response. The present result is a sign-coherent directional measurement of an operational lensing-sector temporal response; the decisive next step is a prospective amplitude test on a future long-baseline multiply-imaged supernova.

| Model | Method | Original $\Delta t_{\rm pred}$ [d] | Revised/used $\Delta t_{\rm pred}$ [d] | $\sigma_{\rm model}$ [d] | Residual [d] | $z$ |
| --- | --- | --- | --- | --- | --- | --- |
| Oguri-a* | GLAFIC parametric (all) | 336 | 353.8 | 19.0 | +22.2 | +1.12 |
| Oguri-g* | GLAFIC parametric (gold) | 311 | 330.7 | 19.7 | +45.3 | +2.22 |
| Sharon-a* | LTM parametric (all) | 237 | 298.0 | 37.5 | +78.0 | +2.06 |
| Sharon-g* | LTM parametric (gold) | 237 | 331.0 | 32.5 | +45.0 | +1.36 |
| Diego-a | WSLAP+ free-form | 262 | 262.0 | 55.0 | +114.0 | +2.06 |
| Grillo-g | GLEE parametric | 361 | 361.0 | 23.0 | +15.0 | +0.63 |
| Jauzac15.2* | LENSTOOL parametric | 449 | 361.0 | 42.0 | +15.0 | +0.35 |
| Grillo+2024 | GLEE updated | N/A | 362.0 | 16.0 | +14.0 | +0.83 |
| Weighted mean (all 8) |  |  |  | — | +30.1 ± 8.9 | +3.39 |

#### Evidence Ledger

| Layer | Data used | Blind status | Result | Interpretation |
| --- | --- | --- | --- | --- |
| **Delay-blind comparison sign layer** | Seven delay-blind Kelly+2023 comparison values | Mixed provenance | 7/7 positive | Primary directional benchmark |
| **Strict original-blind layer** | Original pre-reappearance values only | Fully blind | Positive weighted mean $\sim +55$ d; 6/7 positive | Strict blind directional support |
| **Dependence-aware layer** | Model-family grouping (5 delay-blind families) | Conservative | $p = 0.031$ on delay-blind ensemble; strictly blind floor $p = 0.094$ | Correlation-aware rank bound |
| **Corrected-compilation amplitude layer** | Kelly+2023 Supplementary Table S4 revised values | 5/7 post-reappearance revised | $+30.1 \pm 8.9$ d | Strong-lensing response-scale diagnostic |
| **Diagnostic lensing-sector coefficient** | Corrected-compilation amplitude | Diagnostic | $\kappa_{\rm inferred} = -0.114 \pm 0.034$ | Quantifies amplified response scale |

#### Statistical Tests

Three complementary statistical summaries are reported. The correlation-aware directional headline is the exact family-sign-flip test ($p = 0.031$). Amplitude comparisons against nominal proxy sensitivity ($+14.5$ d) are illustrative only: the corrected-compilation weighted mean ($+30.1$ d) exceeds that nominal value by $1.75\sigma$, while the strict original-blind weighted mean ($\approx +55$ d) exceeds it by much more. Sign coherence is the probative content; amplitude is used to define the corrected-compilation response scale.

#### Independence-assuming directional benchmark: Wilcoxon signed-rank test

All seven delay-blind model residuals for $\Delta t_{\rm SX,S1}$ are positive (Wilcoxon $p = 0.0078$, approximately $2.4\sigma$), matching the proxy-model prediction for a negative temporal-shear coupling. This is the appropriate independence-assuming directional benchmark because it equal-weights delay-blind model variants; dependence among variants from the same modelling family is then handled by the exact family-sign-flip test ($p = 0.031$). The supplementary all-eight-model Wilcoxon, including the post-blind update, gives $p = 0.0039$ ($2.8\sigma$). Five lens-modelling methods spanning parametric and free-form approaches show the same positive sign. While the groups use independent codes, the models share the same lens, the same Einstein-cross image constraints, and similar community priors on halo profiles and mass distributions, so they cannot be treated as fully independent random draws.

| Test | Result | GR $p$-value | Interpretation |
| --- | --- | --- | --- |
| *Exact family-sign-flip* (delay-blind 7, method-family clusters) | All method-family sign assignments enumerated | $p = 0.031$ ($1.86\sigma$) | Correlation-aware headline (Step 11) |
| *Wilcoxon signed-rank* (delay-blind 7, independence) | 7/7 non-zero residuals positive | $p = 0.0078$ ($2.4\sigma$) | Independence-assuming benchmark (assumes between-model independence) |
| *Wilcoxon signed-rank* (all 8, independence) | 8/8 non-zero residuals positive | $p = 0.0039$ ($2.8\sigma$) | Supplementary (includes post-blind update) |
| *Exact family-sign-flip* (all 8, method-family clusters) | All method-family sign assignments enumerated | $p = 0.031$ | Correlation-aware supplementary (Step 11) |
| *Binomial sign test* (all non-zero) | 8/8 positive non-zero residuals | $p = 0.0039$ ($2.8\sigma$) | Supplementary directional check |
| *Binomial sign test* (delay-blind non-zero) | 7/7 positive non-zero residuals | $p = 0.0078$ ($2.4\sigma$) | Supplementary sign-only check |
| *Corrected-compilation weighted mean diagnostic* | $\mathcal{R}_{\rm obs} = +30.1 \pm 8.9$ d | $p = 0.0006$ ($3.39\sigma$) | Corrected-compilation response-scale diagnostic based on five post-reappearance revised values |
| *Corrected-compilation $\chi^2$ diagnostic* | $\Delta\chi^2 = +8.42$ (proxy wins) | No formal $p$-value: both are fixed predictions (0 free params each), not nested models. Individual GOF: GR $p=0.023$, proxy $p=0.32$ | Proxy model fits ensemble better than GR |
| *wRMS improvement* | $28\%$ reduction after proxy-model correction | — | 8/8 models closer to proxy-corrected value |

The Wilcoxon signed-rank test was designated as the independence-assuming non-parametric directional benchmark because it treats each delay-blind model variant as one vote, regardless of the (highly heterogeneous) quoted model uncertainties, eliminating the inverse-variance downweighting bias that suppresses the parametric z-test. Among the seven delay-blind models, all seven non-zero residuals are positive ($p = 0.0078$, equivalent to approximately $2.4\sigma$), a result that would arise by chance with probability 1/128 under the GR null if the models were independent draws. The supplementary all-eight-model Wilcoxon gives $p = 0.0039$ ($2.8\sigma$). The honest binomial sign count: 8/8 all non-zero residuals are positive ($p = 0.0039$), and 7/7 delay-blind non-zero residuals are positive ($p=0.0078$). Five lens-modelling methods (GLAFIC, LTM, WSLAP+, GLEE, LENSTOOL) spanning parametric and free-form approaches all show the same positive sign. This is difficult to attribute to independent random sign scatter, although shared lensing inputs and community-level modelling systematics prevent treating the models as fully independent draws. Because the Wilcoxon statistic lacks a closed-form variance under exchangeable intra-class correlation, an exact family-sign-flip test that enumerates all method-family sign assignments provides the most rigorous dependence-aware rank bound: $p = 0.031$ (one-sided), exact under the sharp null with no superpopulation assumption. This is the operational correlation-aware primary and the paper headline. The method-family block-bootstrap (Step 11) yields a dependence-adjusted Wilcoxon $p_{\rm median} = 0.0078$ [$0.0039$, $0.0156$] for the delay-blind subset, which is reported as a sensitivity exploration rather than an operational primary because it can occasionally reconstruct more extreme statistics than independent sampling when empirical clusters concentrate rank mass. The beta-binomial sign test is the most conservative correlation-aware sign test: with the present data (all 7 delay-blind residuals strictly positive), it gives $p=0.0078$ at $\rho=0$, rising above $0.05$ only if inter-model correlation exceeds the break-even threshold. Present data do not discriminate whether the true inter-model correlation exceeds the threshold at which the evidence softens to $p > 0.05$.

The proxy-corrected observed value $\Delta t_{\rm corr} = 376.0 - 14.5 = 361.5$ d reduces the weighted RMS scatter across all eight model predictions by 28%, and brings the observations into agreement with 7 of 8 models within $1\sigma$.

\begin{equation} \label{eq:4_results_disp_02}
\kappa_{\rm inferred} = \mathcal{R}_{\rm obs} / (d\mathcal{R}_{\rm pred}/d\kappa_{\rm lens}) = -0.114 \pm 0.034
\end{equation}

*Test-selection transparency.* No formal external preregistration (e.g., OSF, AsPredicted) was performed for this analysis. The Wilcoxon signed-rank test on the delay-blind subset was designated as the independence-assuming directional benchmark in the analysis protocol before computation of supplementary tests, based on its statistical property of equal-weighting delay-blind model variants and eliminating inverse-variance downweighting bias. After accounting for method-family dependence, the exact family-sign-flip test was adopted as the correlation-aware headline. All alternative tests reported here were computed and are reported, regardless of outcome.

Because $\kappa_{\rm inferred}$ is derived by dividing the observed weighted-mean residual by the unit proxy-model sensitivity, the same ensemble residual both determines and tests the coupling. Using the Kelly+2023 Supplementary Table S4 corrected compilation as a secondary diagnostic gives $\kappa_{\rm inferred} = -0.114 \pm 0.034$. Because five of the seven contributing values are post-reappearance revisions, this coefficient is interpreted as a corrected-compilation response-scale diagnostic: it preserves the same residual direction, sharpens the inferred strong-lensing response scale, and sets the amplitude target for the transfer-kernel derivation.

The non-circular evidence lies in blind sign consistency ($p = 0.031$ family-sign-flip) and the fact that a one-parameter correction reduces scatter where GR predicts none. A bootstrap resampling of the eight models (10,000 draws with replacement) confirms the negative-sign inference is stable to model resampling: 100% of draws produce $\kappa_{\rm lens} < 0$, with a resampled-mean 68% interval of $[-0.145, -0.090]$. The corrected-compilation weighted residual is $3.39\sigma$ from the GR null; because five corrected values require provenance documentation, the strictly blind headline remains the sign-coherence result and the family-sign-flip bound. The ensemble therefore forms a coherent observational case: the correlation-aware family-sign-flip headline gives $p = 0.031$, the independence-assuming blind Wilcoxon benchmark gives $p = 0.0078$, the supplementary all-8 Wilcoxon gives $p = 0.0039$, all non-zero signs are positive, and $\Delta\chi^2 = +8.42$ strongly favours the calibrated proxy correction over GR. The calibration introduces one operational response coefficient; the test is therefore a one-parameter fit, not a parameter-free prediction.

The important result is not exact amplitude equality at nominal $\kappa_{\rm lens}$, but the separation of physics: the residual sign is fixed by the blind prediction structure, while the corrected-compilation response scale identifies the magnification-sensitive kernel that direct potential transport lacks. The predicted and observed delays are compared in Figure 1.

![SN Refsdal GR model predictions vs observed SX delay with TEP correction](results/figures/step_07_observed_vs_predicted.png)

*Figure 1:* GR lens-model predictions for $\Delta t_{\rm SX,S1}$ from seven delay-blind model variants spanning five modelling families (blue circles) and one post-blind update (purple square), compared to the Kelly et al. (2023) observation (red line/band) and the proxy-corrected value $\Delta t_{\rm obs} - \mathcal{R}_{\rm pred} = 361.5$ d (orange dashed). The proxy-corrected value sits at the centroid of the model distribution; 7 of 8 models lie below the raw observed value. wRMS improves by 28% after proxy-model correction.

### 3.5.1 SN Encore: Blind-Prediction Residual Test

SN Encore (MACS J0138-2155, $z_s = 1.95$, $z_l = 0.338$) is a multiply-imaged Type Ia supernova with two resolved images (1a, 1b). Pierel et al. (2026, ApJ, arXiv:2509.12301) measured $\Delta t_{\rm 1b,1a} = -39.8^{+3.9}_{-3.3}$ d from JWST light curves. Suyu et al. (2025/2026, arXiv:2509.12319) published eight blind lens-model predictions.

| Model | Method | Blind? | $\Delta t_{\rm pred}$ [d] | $\sigma_{\rm model}$ [d] | Residual [d] | $z$ |
| --- | --- | --- | --- | --- | --- | --- |
| glafic (Oguri) | GLAFIC parametric | Yes | $-32.4$ | 2.4 | $-7.4$ | $-1.72$ |
| GLEE | GLEE parametric | Yes | $-37.1$ | 2.7 | $-2.7$ | $-0.60$ |
| GLEE-baseline | GLEE parametric | Yes | $-36.9$ | 2.8 | $-2.9$ | $-0.64$ |
| Lenstool I | Lenstool parametric | Yes | $-75.0$ | 54.5 | $+35.2$ | $+0.64$ |
| Lenstool II | Lenstool parametric | Yes | $-35.6$ | 3.7 | $-4.2$ | $-0.81$ |
| MrMARTIAN | MrMARTIAN free-form | Yes | $-40.6$ | 6.4 | $+0.8$ | $+0.11$ |
| WSLAP+ | WSLAP+ hybrid | Yes | $-112.0$ | 32.0 | $+72.2$ | $+2.24$ |
| Zitrin-analytic | Zitrin-LTM analytic | Yes | $-40.2$ | 9.3 | $+0.4$ | $+0.04$ |
| Weighted mean (all 8) |  |  |  | — | $-3.33 \pm 2.13$ | $-1.56$ |

The weighted mean residual is $-3.33 \pm 2.13$ d, while the proxy-model predicted shift is $-0.49$ d — sub-day and far below the per-model scatter (3–50 d). The binomial sign test (4/8 positive, $p = 0.64$) is consistent with GR. Encore is directionally consistent with TEP but not probative.

### 3.5.2 SN H0pe: Blind-Prediction Residual Test

SN H0pe (PLCK G165.7+67.0, $z_s = 1.783$, $z_l = 0.351$) is a triply-imaged Type Ia supernova. Pierel et al. (2024, ApJ 967, 50) measured $\Delta t_{AB} = -116.6^{+10.8}_{-9.3}$ d and $\Delta t_{CB} = -48.6^{+3.6}_{-4.0}$ d from JWST light curves. Pascale et al. (2025, arXiv:2403.18902) published seven blind lens-model predictions.

| Model | $\Delta t_{AB}^{\rm pred}$ [d] | $\Delta t_{CB}^{\rm pred}$ [d] | $\mathcal{R}_{AB}$ [d] | $\mathcal{R}_{CB}$ [d] |
| --- | --- | --- | --- | --- |
| GLAFIC | $-105.2$ | $-50.7$ | $-11.4$ | $+2.1$ |
| Zitrin-analytic | $-105.5$ | $-41.1$ | $-11.1$ | $-7.5$ |
| LENSTOOL | $-102.7$ | $-54.1$ | $-13.9$ | $+5.5$ |
| MARS | $-136.3$ | $-63.7$ | $+19.7$ | $+15.1$ |
| Chen-2020 | $-112.2$ | $-53.4$ | $-4.4$ | $+4.8$ |
| WSLAP+ | $-273.4$ | $+342.8$ | $+156.7$ | $-391.4$ |
| Zitrin-LTM | $-96.5$ | $-27.6$ | $-20.2$ | $-21.0$ |
| Weighted mean | — |  | $-10.16 \pm 5.13$ | $-0.23 \pm 2.67$ |

The AB residual ($-10.16 \pm 5.13$ d) is directionally consistent with TEP; the CB residual ($-0.23 \pm 2.67$ d) is consistent with zero. As with Encore, predicted TEP shifts are sub-day to $\lesssim$2 d, far below per-model scatter (5–50 d). Not probative.

### 3.6 Extended Evidence Tests

#### 3.6.1 Delay–Magnification Consistency Check

The proxy model predicts that images in shallower potential (lower $\mu$) accumulate more temporal shear and arrive *later*. For SN Refsdal, the least magnified image (SX, $\mu_{\rm norm} \approx 0.35$) is indeed the latest to arrive ($\Delta t_{\rm SX,S1} = 376$ d), while the most magnified image (S4, $\mu_{\rm norm} \approx 1.83$) arrives earlier ($\Delta t_{\rm S4,S1} = 20.3$ d). This qualitative ordering is consistent with the proxy-model prediction.

A formal correlation test between delay and $1/\mu_{\rm norm}$ across all five images is *statistically inappropriate*. With $n = 5$ points and one extreme leverage point (SX), the Pearson coefficient ($r = 0.932$, $p = 0.011$ one-sided) is driven entirely by SX and collapses to $r \approx 0.15$ when SX is excluded. The Cook's distance for SX exceeds $1.0$, confirming that SX dominates the regression. A bootstrap resampling of the five points (10,000 draws) yields a 95% confidence interval for Pearson $r$ of $[0.15, 0.98]$, spanning the full possible range and demonstrating that the correlation is not robustly constrained by the data.

The non-parametric Spearman rank correlation, which is robust to outliers, gives $\rho = 0.3$ ($p = 0.31$ one-sided) — not significant. The Theil-Sen robust regression slope is $86 \pm 210$ d (95% CI), consistent with zero. The inner Einstein-cross images (S1–S4) do *not* show a delay–$\mu$ ordering by themselves: S4, the most magnified image, arrives fourth ($+20.3$ d), later than S2 ($+9.9$ d) and S3 ($+9.0$ d). This is expected: the TEP shift within the inner cross is $\lesssim 0.3$ d (predicted-signal sensitivity $\approx$ 3), far below the 5–20 d geometric path-length differences that determine the S1–S4 arrival order. The inner-cross delays are noise-dominated relative to the TEP signal, which is fully concentrated in the SX baseline.

An ansatz-free rank test (Step 34) replaces the flux-proxy magnification with the inferred convergence from Step 32 and repeats the Spearman correlation. The delay–kappa correlation is $\rho = -0.15$ (permutation $p = 0.63$), and the delay–mu correlation is $\rho = -0.30$ ($p = 0.74$). Both are non-significant, with bootstrap 95% confidence intervals spanning the full $[-0.90, +0.60]$ range. This confirms that with $n=5$ and one extreme leverage point, the physical claim (delays scale with potential depth) and the parametric claim (the scaling follows $1 + \kappa_{\rm lens} \log_{10}\mu$) are not separately constrained by present data. Only the binary S4–SX sign-contrast — not the full five-image rank order — is probative.

*Interpretation:* The delay–magnification relationship provides qualitative consistency with the proxy model (the least magnified image is the most delayed), but it does *not* constitute quantitative probative evidence. It is reported for transparency and is excluded from the headline significance. The blind-prediction directional sign test (§3.5) is independent of any proxy: it compares the observed delay to GR geometric predictions that were made before the measurement existed. The sign of those residuals (all positive) is an empirical fact; the proxy model predicts the sign, but the sign itself does not depend on the choice of proxy.

#### 3.6.2 Per-Model Inferred Coupling

Each model residual maps to an inferred coupling via $\kappa_{\rm inferred,i} = \mathcal{R}_{{\rm obs},i} / (d\mathcal{R}_{\rm TEP}/d\kappa_{\rm lens})$. Table 5 gives the per-model values. Under GR, non-zero residuals should be sign-symmetric; the fact that all eight models yield $\kappa_{\rm lens} < 0$ is the probative content.

| Model | Method family | Blind? | Residual [d] | $\kappa_{\rm inferred}$ | $\sigma_{\kappa_{\rm lens}}$ | $z$ from GR null |
| --- | --- | --- | --- | --- | --- | --- |
| Oguri-a* | GLAFIC | Yes | +22.2 | $-0.084$ | 0.075 | $-1.12$ |
| Oguri-g* | GLAFIC | Yes | +45.3 | $-0.171$ | 0.077 | $-2.22$ |
| Sharon-a* | LTM | Yes | +78.0 | $-0.295$ | 0.143 | $-2.06$ |
| Sharon-g* | LTM | Yes | +45.0 | $-0.170$ | 0.125 | $-1.36$ |
| Diego-a | WSLAP+ | Yes | +114.0 | $-0.431$ | 0.209 | $-2.06$ |
| Grillo-g | GLEE | Yes | +15.0 | $-0.057$ | 0.090 | $-0.63$ |
| Jauzac15.2* | LENSTOOL | No | +15.0 | $-0.057$ | 0.160 | $-0.35$ |
| Grillo+2024 | GLEE | No | +14.0 | $-0.053$ | 0.061 | $-0.83$ |
| Inverse-variance weighted mean (all 8) |  |  |  | $-0.114$ | 0.034 | $-3.39$ |
| Inverse-variance weighted mean (delay-delay-blind 7) |  |  |  | $-0.120$ | 0.038 | $-3.16$ |

The ensemble aggregates to $\bar{\kappa}_{\rm inferred} = -0.114 \pm 0.034$ (Table 5, last row). This exceeds the nominal response normalization by $1.75\sigma$ — a calibrated response-scale estimate, not independent confirmation. Bootstrap resampling (10,000 draws) yields a 68% interval $[-0.145, -0.090]$ with 100% of draws producing $\kappa_{\rm lens} < 0$, reinforcing that the sign inference is stable to model resampling.

#### 3.6.3 Correlated Significance and the Single-Test Benchmark

The various statistical tests for SN Refsdal (Wilcoxon sign test, weighted mean, Pearson correlation, coupling inference) all rely fundamentally on the single anomalous SX arrival time, and are therefore highly correlated.

Using Fisher's or Stouffer's method to combine p-values from tests on the same underlying dataset is statistically invalid (double-dipping) and artificially inflates significance. Therefore, rather than combining tests, the exact family-sign-flip test is reported as the correlation-aware headline significance ($p = 0.031$, $z \approx 1.86\sigma$), supported by the consistency of the other metrics. The delay-blind Wilcoxon signed-rank test remains the independence-assuming benchmark ($p = 0.0078$, $z \approx 2.4\sigma$), and the supplementary all-eight-model Wilcoxon, which includes the post-blind precision update, gives $p = 0.0039$ ($2.8\sigma$).

#### 3.6.4 Dependence and Systematics Robustness

Four dedicated robustness analyses were run to stress-test the evidence stack against model dependence and flux-proxy systematics. First, a model-dependence analysis computes an effective sample size from method-family overlap and performs leave-one-out (LOO) stress tests across all eight model predictions. The method-family Kish proxy gives $N_{\rm eff} = 7.2$ (from $N=8$), and LOO tests keep the sign-test at $p=0.0078$ in all 8/8 realizations (all models positive), with weighted-mean residual significance in the range $z=0.96$ to $1.30$. The GR-vs-TEP fit preference remains stable under LOO: $\Delta\chi^2 \in [+0.91, +1.68]$ (TEP better in all 8/8 LOO realizations).

Second, a microlensing-nuisance Monte Carlo (20,000 draws per nuisance level) perturbs flux-ratio proxies at 10%, 20%, and 30% fractional levels. The SX-loop TEP predicted GR discrepancy remains centred near the nominal value in all cases: median $\mathcal{R}_{\rm TEP/GR} \approx -14.5$ d (10%: $-14.53$, 20%: $-14.50$, 30%: $-14.53$), with broadening uncertainty envelopes but stable negative sign. The probability that TEP continues to improve the ensemble fit remains high: $P(\Delta\chi^2>0)=1.000$, $1.000$, and $0.992$ for 10%, 20%, and 30% nuisance levels, respectively.

Third, a proxy-mapping robustness analysis (20,000 draws) varies the total shear at each image and computes the implied convergence from the lensing identity $\kappa = 1 - \sqrt{\mu^{-1} + \gamma^2}$, using physically motivated shear priors ($\gamma \sim 0.5$–$0.8$ for S1–S4, $\gamma \sim 0.1$–$0.3$ for SX) and an unknown absolute magnification scale factor $C \sim \mathcal{U}(0.5, 4.0)$. This is the dominant systematic in the analysis. The inferred TEP residual broadens substantially: median $\mathcal{R}_{\rm TEP/GR} = -18.9$ [$-29.7$, $+0.9$] d, and the probability that TEP improves the fit drops to $P(\Delta\chi^2>0) = 0.63$. The 84th percentile crosses zero, but this cross-zero behaviour is restricted strictly to the amplitude reconstruction; the binary S4–SX sign contrast remains stable at $P \approx 80$% and does not cross the 50% null. This reflects the factor-of-two amplitude uncertainty from the shear–magnification degeneracy while preserving the directional sign evidence. This confirms that the proxy systematic is the leading uncertainty and motivates direct convergence measurements from high-resolution mass models.

Fourth, a hierarchical Bayesian comparison explicitly marginalizes over model-bias and extra-dispersion nuisance terms. Under priors $\mu_{\rm bias}\sim\mathcal{N}(0,40\,{\rm d})$, $\tau\sim\text{HalfNormal}(20\,{\rm d})$, and (for free-coupling TEP) $\kappa_{\rm lens}\sim\mathcal{N}(0,0.15)$ centred on the GR null to avoid circularity. Bayes factors are non-decisive: $\log\mathrm{BF}_{\rm TEP\ fixed/GR}=+0.06$ (BF$=1.06$ baseline; $0.997$ h0pe-informed) and $\log\mathrm{BF}_{\rm TEP\ free/GR}=-0.49$ (BF$=0.615$ baseline; $0.492$ h0pe-informed). The fixed-coupling model tests the specific SN Refsdal empirical prediction and shows no preference either way; the free-coupling model, with a proper GR-centred prior, shows mild preference for GR. Both are within the "inconclusive" regime ($|\log\mathrm{BF}|<1$), indicating that present model uncertainties dominate formal model-selection metrics. The posterior coupling spans zero, $\kappa_{\rm lens} = -0.038\,[-0.124,+0.045]$ (16th-84th percentiles, baseline-prior scenario), and the inferred extra dispersion is $\tau = 10.1\,[2.3,17.8]$ d.

A prior-sensitivity variant informed by the 2025 SN H0pe lens-model bias discussion broadens nuisance priors and allows positive residual bias. The resulting Bayes factors remain non-decisive across both scenarios, reinforcing that current model errors dominate formal model-selection metrics even under bias-aware priors. The free-coupling model's stronger preference for GR under broader priors reflects increased tension from allowing larger positive nuisance bias.

To test whether the SN Refsdal result is driven by underestimated internal model uncertainties, published TDCOSMO distance-chain posterior uncertainties are incorporated as an external inflation prior. The external coefficient-of-variation prior gives $\kappa_{50}=0.111$ (16th-84th: 0.085-0.140). At this median inflation level, the weighted-mean residual is barely affected ($z \approx 3.37$ at $\kappa_{50}$), while the GR-vs-TEP fit preference remains strongly positive ($\Delta\chi^2 \approx +8.3$). This is the expected behaviour under conservative error inflation: negligible amplitude change because the corrected residual is so far from the GR null, with direction unchanged.

As discussed in §4.7, combining correlated evidence strands would inflate significance; the headline significance is the exact family-sign-flip test, with the Wilcoxon retained as the independence-assuming benchmark.

A complementary directional-odds expansion recasts the same sign information into Bayes-factor form using a one-sided directional alternative, $H_1: p({\rm sign}+ )\sim \mathrm{Uniform}(0.5,1)$, versus the null $H_0: p=0.5$. For all non-zero residuals (8/8 positive), the directional Bayes factor is $\mathrm{BF}_{10}=56.8$; for the delay-blind non-zero residuals (7/7 positive), $\mathrm{BF}_{10}=18.1$; and for method-family-collapsed signs (5/5 positive), $\mathrm{BF}_{10}=10.5$. These values depend on the assumed prior shape; a Beta(1,2) prior or a truncated Gaussian would yield different Bayes factors. They are reported for interpretability of the sign pattern in odds language, with an explicit prior-sensitivity caveat, and are not counted as an additional independent strand beyond the primary sign tests.

Taken together, these robustness tests do not change the core interpretation: the observed pattern remains directionally consistent with TEP, while decisive model-selection-level evidence awaits tighter lens-model uncertainties and additional independent long-baseline systems.

#### 3.6.5 Chromaticity Null in COSMOGRAIL Quasar Light Curves

The log-magnification proxy model adopted here predicts an *achromatic* temporal-shear correction: $\Gamma_t(i) = 1 + \kappa_{\rm lens}\log_{10}(\mu_{\rm norm}(i))$ multiplies the GR delay by a single image-position-dependent factor and is, by construction, independent of the variability timescale. A natural null test is therefore whether observed quasar time delays exhibit any frequency dependence $\Gamma \equiv d(\Delta t)/d(\log_{10}\tau)$ inconsistent with a single broadband value. Step 30 measures this slope using COSMOGRAIL light curves for 18 quasar lens systems (55 valid image pairs after quality cuts) with mode-locked cross-correlation at multiple bandpass timescales. The result is a clean null: $\langle\Gamma\rangle = -3.8 \pm 36.9$ d/decade, median $\Gamma = 0$ d/decade, and zero pairs significant at the $2\sigma$ level (Step 30 summary). Step 31 confirms this null is robust under season-shuffle and microlensing-injection scrambling.

This null is fully consistent with the proxy-model ansatz (which forbids chromaticity by construction) and provides an independent empirical bound: any TEP-induced delay correction in quasar systems must be sufficiently achromatic to evade detection at the $\sim$30 d/decade level across the COSMOGRAIL sample. The COSMOGRAIL chromaticity null and the SN Refsdal SX residual therefore test orthogonal observables, and their joint outcome (broadband shift consistent with $-\kappa_{\rm lens}\log_{10}\mu$ scaling; no detectable bandpass slope) is the pattern the proxy model predicts.

### 3.7 Low-$H_0$ Consistency Check: Data

Published $H_0$ measurements from multiply-imaged supernovae yield values in the range $H_0 \approx 61$–$67$ km s$^{-1}$ Mpc$^{-1}$ (Kelly et al. 2023; Pierel et al. 2024, 2026). Since inferred $H_0$ scales as $1/\Delta t_{\rm obs}$, a proxy-model-induced delay expansion biases the GR-inferred value low. The predicted correction is:

\begin{equation} \label{eq:4_results_disp_04}
H_{0,\rm true} = H_{0,\rm inferred} \times \left( \frac{\Delta t_{\rm obs}}{\Delta t_{\rm geom}} \right)
\end{equation}

For SN Refsdal the shift is $+2.7$ km s$^{-1}$ Mpc$^{-1}$ ($66.6 \to 69.3$); for SN Encore and SN H0pe the shifts are $\sim 0.1$ km s$^{-1}$ Mpc$^{-1}$, negligible compared to their uncertainties. Full interpretation, including the non-independence caveat, is given in §4.8. Figure 2 shows the $H_0$ inference under GR and the proxy model.

![Low-$H_0$ Consistency Check](results/figures/step_10_h0_tension.png)

*Figure 2:* The Hubble constant $H_0$ inferred from SN Refsdal, SN Encore, and SN H0pe. Under GR (blue), Refsdal and Encore lie at the low end of the Planck range. Under the proxy model with the empirically determined coupling (orange), the SN Refsdal value shifts upward by approximately $2.7$ km s$^{-1}$ Mpc$^{-1}$. *Definitional consistency check only* (see §4.4 and §4.8 for the non-independence caveat). SN Encore and SN H0pe shifts are sub-percent and not probative.

## 4. Discussion

### 4.1 The SX Baseline: Why SN Refsdal is the Ideal System

The dominant result of this analysis is the sign-directional test: the delay-blind Kelly+2023 comparison ensemble shows all seven model variants yielding positive residuals, while the strict original-blind layer has six of seven pre-reappearance predictions underestimating the SX delay. Geometrically, the high-leverage contrast is S4–SX on the 376-day baseline: SX traverses a significantly less magnified region than S4 ($\mu_{\rm SX} \approx 0.35$ vs $\mu_{\rm S4} \approx 1.79$ in relative flux units). At the nominal response normalization ($\kappa_{\rm lens} = -0.055$), the S1–S4–SX loop yields a post-hoc proxy sensitivity of $+14.5 \pm 0.2$ d (formal predicted-signal sensitivity $\approx$ 63). That number illustrates post-hoc S4–SX proxy sensitivity on the Kelly+2023 geometry; it is not the headline evidence.

#### 4.1.1 The Inner Cross as a Null Region

The inner Einstein cross provides no probative constraint under the adopted proxy: the $\mu \rightarrow \kappa$ proxy fails there because shear degeneracy renders magnification ordering uninformative about convergence (rank-order agreement drops to $P \approx 3$%; Spearman $\rho \approx -0.4$, $p \approx 0.79$). The signal-energy partitioning (Step 35; §3.3.2) confirms that 99.9% of predicted TEP signal energy resides in the two SX-containing loops, with the S4–SX contrast providing the sole testable lever. The proxy/measurement-error ratio scales linearly with baseline — the 20-day inner-cross loops yield ratio $\approx$ 3, while the 376-day SX loops amplify the same $\Delta\Gamma$ by a factor of ~18 to ratio $\approx$ 63–66 — so the test is structurally a single-contrast measurement.

Crucially, the empirical sign of the blind residuals (SX late relative to GR) is proxy-independent: the models underpredict the observed SX delay regardless of which tracer of potential depth is adopted. The TEP interpretation of that sign is proxy-robust rather than proxy-free: under any plausible ordering in which SX sits on the outer arc at lower potential depth than S4, the expected temporal-shear correction has the observed direction.

### 4.2 Insensitivity to the Mass Sheet Degeneracy

A central concern in time-delay cosmography is the Mass Sheet Degeneracy (MSD): adding a uniform convergence sheet $\kappa_{\rm ext}$ to any lens model rescales all pairwise delays by a common factor $(1-\kappa_{\rm ext})$, leaving the image positions unchanged (Falco, Gorenstein & Shapiro 1985). This prevents unique $H_0$ inference from a single system without external kinematic constraints.

The algebraic loop sum is insensitive to the MSD: if all delays scale as $\Delta t \to (1-\kappa)\Delta t$, then $\mathcal{L} = \Delta t_{ij} + \Delta t_{jk} + \Delta t_{ki} \to (1-\kappa) \times 0 = 0$. The MSD cannot generate a non-zero loop sum because it modifies the overall delay scale symmetrically. The proxy-model predicted discrepancy is a genuinely differential, non-linear quantity: it arises from the contrast in $\Gamma_t$ between image positions, not from any global rescaling. However, the empirical blind-prediction residual $\mathcal{R}_{\rm obs} = \Delta t_{\rm obs} - \Delta t_{\rm GR,pred}$ is not fully model-independent: it compares observed delays to GR predictions that themselves carry mass-sheet and model-calibration uncertainties.

### 4.3 Validation of the Magnification Proxy Assumption

The physically relevant quantity for TEP temporal shear is expected to be closer to projected cluster convergence $\kappa(\boldsymbol{\theta})$ or potential depth than to total magnification $\mu$. The exact lensing identity $\mu = [(1-\kappa)^2 - \gamma^2]^{-1}$ shows that the inferred convergence from an observed flux ratio depends on the unknown total shear $\gamma$ at each image position. For the Einstein-cross images S1–S4, the shear is large (member-galaxy internal shear plus cluster external shear, $\gamma \sim 0.5$–$0.8$), making $\mu$ a poor proxy for $\kappa$. For the peripheral arc SX, the shear is smaller ($\gamma \sim 0.1$–$0.3$), and $\mu$ is a more faithful proxy.

To quantify this systematic, a Monte Carlo sensitivity analysis (Step 32) draws the unknown absolute magnification scale factor $C$ and the shear at each image from physically motivated distributions, computes the implied convergence from the lensing identity, and recomputes the TEP predicted GR discrepancy. The results show:

- *Amplitude shift.* The equivalent $\kappa_{\rm lens}$ required to match the same observed S1–S4–SX residual is $-0.032$ [$-0.068$, $+0.009$], compared to the nominal proxy-model value $-0.055$. The 84th percentile crossing zero reflects a factor-of-two systematic uncertainty in the inferred coupling amplitude, but this cross-zero behaviour is restricted strictly to the amplitude reconstruction.

- *Sign stability.* The S4–SX contrast sign is robust: the probability that $\kappa_{\rm S4} > \kappa_{\rm SX}$ (matching the flux-proxy ordering) is $P \approx 80$%. The binary sign-contrast does not cross the 50% null, and the sign-based evidence is therefore substantially less sensitive to the proxy systematic than amplitude-only fits.

- *Rank-order instability.* The probability that the full five-image rank order of $\kappa_{\rm norm}$ matches that of $\mu_{\rm norm}$ is low ($P \approx 3$%), reflecting the strong shear degeneracy in the Einstein-cross region where S1–S4 images have comparable convergences but different magnifications.

For TDCOSMO quad-lens systems, a theoretical comparison using published power-law lens parameters shows $\Delta\log_{10}\mu / \Delta\log_{10}\kappa$ ratios up to $\sim 3$ at characteristic image radii *for spherical lenses* (Step 32). Elliptical SIE lenses with typical external shear ($\gamma_{\rm ext} \sim 0.06$–0.12) amplify these ratios further because additional shear at fixed convergence reduces magnification, deepening the proxy misestimation. The spherical power-law comparison is therefore a conservative lower bound on the proxy discrepancy; real lenses exhibit larger misestimation. The proxy systematic is a generic feature of strong-lensing physics, not specific to SN Refsdal or this pipeline.

These results frame the current analysis as a sign-robust directional test with secondary amplitude diagnostics only. The full blind-original ensemble (seven published pre-reappearance values) has weighted mean $\mathcal{R}_{\rm obs} \approx +55$ d — much larger than the nominal proxy sensitivity ($\sim +14.5$ d). Only the parametric cluster near $\Delta t_{\rm pred} \approx 361$ d (notably Grillo-g, $+15$ d) sits near the nominal proxy value; this is a subset, not the full blind ensemble. Kelly et al. (2023) updated five of the seven models using the SX reappearance position (Rodney et al. 2023, ApJ 948, 93); the corrected-compilation amplitude ($+30.1 \pm 8.9$ d) is not the strict blind headline because updating models toward the observation necessarily moves predictions toward the observed delay. It is retained as a strong response-scale diagnostic: same residual direction, larger implied coupling ($\kappa_{\rm inferred} \approx -0.114$), calibrated from the ensemble rather than an independent blind prediction.

*Direct-convergence test (Step 44).* The GLAFIC v3 convergence values from Kelly et al. (2023, Table 15) provide the first direct test of the proxy ansatz with model convergence rather than inferred convergence. The values place SX at $\kappa_{\rm SX} = 0.966$ (the highest of the five images) and S4 at $\kappa_{\rm S4} = 0.727$. Naively using $\kappa$ as the tracer gives a predicted residual of $-2.6$ d, opposite in sign to the observed $+30.1$ d, because high $\kappa$ (density) does not map to deep potential. With the locked lab-scale convention $\beta_A = -1$, the TEP scalar field tracks the Newtonian potential. For an isothermal-like profile, pointwise $\kappa$ is anti-correlated with image radius, so $1/\kappa$ provides a useful exploratory radial/potential-depth ordering. It should not be identified with the final TEP lensing transfer function. Evaluating the proxy ansatz with the $1/\kappa$ tracer gives a predicted residual of $+2.6$ d, matching the sign of the flux-proxy prediction ($+14.5$ d) and the observation ($+30.1$ d). The sign evidence is therefore robust to substitution of the exploratory proxy, though the amplitude is suppressed because the $1/\kappa$ contrast ($\Delta\log_{10}(1/\kappa) \approx 0.12$) is smaller than the flux-proxy contrast ($\Delta\log_{10}\mu \approx 0.71$).

Direct transport tracers (Steps 50–51) preserve the observed residual sign but underpredict the amplitude by factors of ~750 (potential-map point sampling) and ~40–90 (3D geodesic integration). The fundamental TEP formula evaluated with the cluster potential predicts residuals of order $10^{-6}$ days — far below the observed +30.1 d. This gap is physical: the cluster potential varies slowly across the S4–SX separation, whereas the magnification denominator amplifies the contrast. The log-magnification response is therefore the only tested formulation that yields an observationally relevant effect. Whether this amplification can be derived from a lensing Jacobian kernel in the scalar-field action remains an open theoretical question. The archived GLAFIC v3 maps exhibit a ~4× normalisation offset relative to the Kelly+2023 tabulated parameters (Step 54), so the map-derived Jacobian does not match the ground-truth lens model and a first-principles kernel prediction is not yet available.

*Sketch of the derivation path.* The amplification gap suggests that the missing structure lies in how lensing-sector spatial derivatives enter the temporal metric. In standard lensing, magnification $\mu_i = 1/|\det A_i|$ depends on the Jacobian $A_i = \partial\boldsymbol{\beta}/\partial\boldsymbol{\theta}$; near critical curves, small mapping perturbations produce amplified changes in both magnification and image timing. A scalar-field TEP action that couples the temporal metric perturbation to gradients of the lensing potential — rather than to the potential alone — could in principle yield a magnification-sensitive transfer function $T(\kappa,\gamma)$ mapping lensing structure into temporal response. Deriving such a term from the scalar-field Lagrangian, and matching its predictions to the observed $\sim$30-day residual scale, remains the central open theoretical task.

#### Systematics Budget

| Systematic / uncertainty source | Magnitude | Effect on evidence | Section |
| --- | --- | --- | --- |
| Lens model scatter | $\pm$16–60 d per model | Dominant; ensemble significance ranges from $z \approx 1.86$ (correlation-aware family-sign-flip) to $3.4$ (weighted mean) | §3.5, §4.11 |
| Proxy systematic ($\mu \to \kappa$) | Factor of $\sim$2 in $\kappa_{\rm lens}$ amplitude; 84th percentile crosses zero | Amplitude reconstruction degraded; sign stable at $P \approx 80$% | §2.4.2, §3.3.1, §4.3 |
| Microlensing flux bias | 10–30% fractional perturbation | Broadens uncertainty envelope; sign remains stable | §2.4.1, §3.6.5, §4.5 |
| Inter-model correlation | Binomial sign-test: $\rho = 0.0$ (delay-delay-blind 7) and $\rho \approx 0.026$ (all 8). Wilcoxon approx. heuristic: $\rho \approx 0.08$ (selected) — lacks formal justification. | Exact family-sign-flip $p = 0.031$ (primary correlation-aware bound); block-bootstrap $p_{\rm median} = 0.0078$ [$0.0078$, $0.0312$] (sensitivity exploration) | §3.6.4, §4.10.6 |
| Delay measurement noise | $\pm$5.6 d on SX–S1 | Subdominant to lens model scatter | §3.1, §4.11 |
| External error inflation (TDCOSMO) | $\kappa_{50} = 0.111$ (16th–84th: 0.085–0.140) | Negligible effect on weighted-mean ($z \approx 3.37$ at median $\kappa_{50}$); preserves strong directional fit preference ($\Delta\chi^2 \approx +8.3$) | §3.6.4, §4.5 |

### 4.4 The Observed vs. Blind-Predicted Test: What It Shows

The algebraic loop sum computed directly from the Kelly et al. (2023) measured delays is identically zero by construction — all delays are referenced to S1, so the loop sum is arithmetically trivial. A genuine non-zero test requires independent delay chains. §3.5 provides this through the historical blind prediction record: seven delay-blind model variants spanning five modelling families predicted $\Delta t_{\rm SX,S1}$ before the measurement existed, providing a genuinely independent comparison dataset.

The observed weighted-mean residual $\mathcal{R}_{\rm obs} = +30.1 \pm 8.9$ d is dominated by lens model uncertainties ($\pm$16–60 d), not measurement noise ($\pm$5.6 d). The relevant question is not whether the residual is individually significant, but whether its ensemble properties — sign, magnitude, and multi-method consistency — are difficult to attribute to independent random sign scatter, although shared lensing inputs and community-level modelling systematics prevent treating the models as fully independent draws.

Three properties of the residual argue for a physical origin:

- *Sign consistency across methods.* Five modelling codes (GLAFIC, LTM, WSLAP+, GLEE, LENSTOOL) spanning parametric and free-form approaches all underestimate the delay; all seven delay-blind models and the post-blind update yield positive residuals (8/8, $p = 0.0039$; delay-delay-blind 7/7, $p = 0.0078$). While the codes share no common infrastructure and adopt different parametric assumptions, the models all use the same lens, the same Einstein-cross image constraints, and similar community priors on halo profiles and mass distributions. Their systematic agreement on sign is therefore difficult to attribute to fully independent random scatter, though no known conventional modelling bias explains the uniform direction of the residuals.

- *Magnitude mapping is definitional, not corroborating.* The corrected-compilation weighted mean $+30.1$ d maps to $\kappa_{\rm inferred} = -0.114 \pm 0.034$ by dividing by the unit proxy-model sensitivity. That mapping is circular: the same ensemble residual both defines and tests the coupling. The probative content lies in blind sign coherence ($p = 0.031$), not in amplitude agreement at nominal $\kappa_{\rm lens} = -0.055$ ($1.75\sigma$ tension with the corrected compilation).

- *Proxy-model correction reduces scatter.* Subtracting the proxy-model predicted residual from the observed delay reduces the weighted RMS of model–observation disagreement by 28%, and brings all 8 models into better agreement. Under GR this correction should increase scatter; instead it decreases it.

Taken individually, none of these is statistically conclusive. Combined, they constitute the current observational case for the log-magnification proxy model using publicly available data.

#### Definitional amplitude calibration

The proxy-model predicted GR discrepancy is linear in $\kappa_{\rm lens}$: $\mathcal{R}_{\rm TEP/GR}(\kappa_{\rm lens}) = \kappa_{\rm lens} \cdot f(\boldsymbol{\Delta t}, \boldsymbol{\mu})$, where $f$ is a purely geometric function. The unit sensitivity is $S = d\mathcal{R}_{\rm TEP/GR}/d\kappa_{\rm lens} = f$. The empirical coupling is defined by demanding that the predicted residual matches the observed mean: $\kappa_{\rm lens} = \mathcal{R}_{\rm obs} / S$. The inferred coupling from the same data is $\kappa_{\rm inferred} = \mathcal{R}_{\rm obs} / S$. Therefore, by construction, $\kappa_{\rm inferred} \equiv \kappa_{\rm lens}$. This definitional agreement is a transparent feature of phenomenological response testing, not a methodological flaw: the calibrated proxy captures the observed response scale without introducing free parameters, and the probative content lies entirely in the sign consistency ($\kappa_{\rm lens} < 0$ for all models) and the scatter-reduction property (wRMS improves by 28% after proxy-model correction, where GR predicts no correction at all).

What would make this conclusive: a future long-baseline multiply-imaged supernova with genuinely blind pre-observation GR predictions, an observed-minus-GR residual whose *amplitude* matches a pre-locked proxy or transfer-kernel forecast, and per-model uncertainties below $\sigma_{\rm model} \lesssim 5$ d would provide a decisive blind amplitude test. The present Refsdal result establishes directional sign coherence at the blind level ($p = 0.031$ family-sign-flip). Amplitude is not claimed at the blind level: the blind-original weighted mean is $\approx +55$ d, the nominal proxy sensitivity is $\sim +14.5$ d, and the corrected-compilation diagnostic is $+30.1$ d. Any prospective transfer-kernel derivation must explain the sign first, then the response scale suggested by the corrected compilation.

### 4.5 Robustness Stress Tests and Bayesian Priors

Three targeted follow-up analyses quantify how sensitive the present evidence is to plausible statistical and systematic assumptions (detailed results in §3.6.5). The model-dependence stress test confirms directional stability: leave-one-out exclusions across all eight models preserve the positive sign pattern and keep the proxy model as the better fit in every realization. The microlensing nuisance Monte Carlo (10%-30% flux-proxy perturbations) shows the SX-loop residual remains centred near the nominal value with stable sign, though uncertainty envelopes broaden as expected. The hierarchical Bayesian comparison yields non-decisive Bayes factors under both baseline and bias-aware priors, with the free-coupling model using a proper GR-centred prior ($\kappa_{\rm lens}\sim\mathcal{N}(0,0.15)$) to avoid circularity. The fixed-coupling test of the specific SN Refsdal prediction shows no preference either way (BF$=1.06$ baseline; $0.997$ h0pe-informed), while the free-coupling model with the corrected prior shows mild preference for GR (BF$=0.615$ baseline; $0.492$ h0pe-informed). The free-coupling model is more conservative because it tests whether any coupling within a broad prior improves the fit, whereas the fixed-coupling model tests the specific empirical prediction derived from SN Refsdal. Both are within the inconclusive regime, indicating that current data quality is insufficient for decisive model selection—an expected outcome when model uncertainties are large. The posterior coupling spans zero in both scenarios. External error inflation from TDCOSMO distance-chain priors has minimal effect on the weighted-mean significance ($z \approx 3.37$ at median external convergence $\kappa_{50}=0.111$) because the corrected residual is so far from the GR null; the directional fit preference remains strong ($\Delta\chi^2 \approx +8.3$), the expected behaviour under conservative error inflation.

The combined implication is not a stronger detection claim, but a stronger reliability claim: the directional evidence is persistent across dependence and nuisance perturbations, and the analysis explicitly separates directional consistency from decisive model-selection evidence (see §4.7 for the full evidence synthesis).

### 4.6 Coupling Sensitivity and the Geometric Nature of SNR

The coupling sensitivity scan ($\kappa_{\rm lens} \in [0.001, 0.15]$, 150 values) reveals a key structural result: the signal-to-noise ratio $\text{SNR} = |\mathcal{R}_{\rm TEP}|/\sigma_{\mathcal{R}}$ is exactly independent of $\kappa_{\rm lens}$ for all five loops. This follows directly from the linearity of the proxy formulation: both $\mathcal{R}_{\rm TEP}$ and $\sigma_{\mathcal{R}}$ are proportional to $\kappa_{\rm lens}$, so their ratio cancels:

\begin{equation} \label{eq:5_discussion_01}
\text{SNR} = \frac{|\mathcal{R}_{\rm TEP}(\kappa_{\rm lens})|}{\sigma_{\mathcal{R}}(\kappa_{\rm lens})} = \frac{|\kappa_{\rm lens}| \cdot |f(\boldsymbol{\Delta t}, \boldsymbol{\mu})|}{|\kappa_{\rm lens}| \cdot g(\boldsymbol{\sigma}_{\Delta t}, \boldsymbol{\mu})} = \frac{|f|}{g}
\end{equation}

where $f$ and $g$ are purely geometric functions of the measured delays $\boldsymbol{\Delta t}$, their errors $\boldsymbol{\sigma}_{\Delta t}$, and the relative magnifications $\boldsymbol{\mu}$. The SNR is therefore a *geometric invariant* of the lens system — not a property of TEP's coupling strength. The intrinsic SNR values per loop are:

| Loop | Intrinsic SNR (all $\kappa_{\rm lens}$) | 3$\sigma$ detectable? | 5$\sigma$ detectable? |
| --- | --- | --- | --- |
| S1–S2–S3 | 3.27 | Yes (at all $\kappa_{\rm lens}$) | No |
| S1–S2–S4 | 2.52 | No (always below) | No |
| S1–S3–S4 | 2.30 | No (always below) | No |
| S1–S2–SX | 66.3 | Yes (at all $\kappa_{\rm lens}$) | Yes |
| S1–S4–SX | 63.3 | Yes (at all $\kappa_{\rm lens}$) | Yes |

The implication is direct: the detectability of the proxy model in the strong lensing regime is not limited by the coupling constant $\kappa_{\rm lens}$, but by the geometry of the lens system. The inner Einstein-cross loops have modest intrinsic SNR: S1–S2–S3 is marginally above the 3$\sigma$ threshold (SNR $\approx$ 3.3), while S1–S2–S4 (2.5) and S1–S3–S4 (2.3) remain sub-threshold regardless of how large $\kappa_{\rm lens}$ is — because the four cross images have similar magnifications ($\mu_{\rm rel} \approx 0.72$–$1.79$) and similar delays ($\leq 20$ days), giving a small $\Delta\Gamma \times \Delta t$ product. The SX loops are above 5$\sigma$ at every $\kappa_{\rm lens} \neq 0$, because the 376-day baseline amplifies even the smallest $\Delta\Gamma$ into a measurable signal.

This also means: *if an independent measurement of the S4–SX delay falsifies the proxy-model prediction, it rules out the linear log-magnification ansatz at every value of $\kappa_{\rm lens}$ simultaneously* — not just at $\kappa_{\rm lens} \approx -0.055$. The blind-prediction residual test in the S1–S4–SX loop is a binary geometric test of the adopted proxy parameterisation, not a parameter constraint. A different functional form (e.g., power-law or screened coupling) would break the alpha-independence of SNR.

Conversely, if the observed $\mathcal{R}_{\rm obs}(\text{S1, S4, SX})$ is non-zero, the measured value directly determines $\kappa_{\rm lens}$: $\kappa_{\rm meas} = \mathcal{R}_{\rm obs} / f(\boldsymbol{\Delta t}, \boldsymbol{\mu})$ — a direct coupling measurement from a single lens system.

### 4.7 Evidence Synthesis: A Multi-Pronged Observational Case

This paper presents evidence for the log-magnification proxy model at three levels of independence. The evidence is organised into two tiers: primary directional evidence from SN Refsdal, and supplementary cross-system checks.

#### Table 1: Primary Directional Evidence (SN Refsdal)

| Evidence strand | Test type | Result | $p$-value / significance | Independent? | Status |
| --- | --- | --- | --- | --- | --- |
| *Wilcoxon signed-rank* delay-blind 7/7 non-zero; all 8/8 non-zero | Non-parametric signed-rank test | All non-zero residuals positiveEqual-weight directional test | $p=0.0078$ delay-blind ($2.4\sigma$); $p=0.0039$ all ($2.8\sigma$) | ✓ Yes (delay-blind subset) | ✓ Observed |
| *Residual magnitude vs. proxy model* $\mathcal{R}_{\rm obs}$ vs. $\mathcal{R}_{\rm TEP}$ | Point estimate comparison | $\mathcal{R}_{\rm obs} = +30.1$ d$\kappa_{\rm inferred} = -0.114 \pm 0.034$ (by construction; 0 free params)*Definitional (not independent)* | $1.75\sigma$ from nominal proxy sensitivity by construction | ✗ No (same ensemble) | ✓ Observed (definitional) |
| *Correlated Significance Synthesis* All tests structurally correlated | Synthesis Framework | Avoids double-dippingHeadline: family-sign-flip ($p=0.031$)Wilcoxon $p=0.0078$ (independence-assuming bound) | $z \approx 1.86\sigma$ (headline); $z \approx 2.4\sigma$ (Wilcoxon benchmark) | — | ✓ Methodological constraint |
| *Loop sensitivity geometry* Coupling-independent invariant | Structural prediction | SX-loop proxy/error ratio = 63–66 at all $\kappa_{\rm lens} > 0$Geometric invariant | Geometric (no $\sigma$) | ✓ Yes (theoretical) | ✓ Structural |
| *Signal-energy concentration* Step 35, single-contrast dominance | Structural information-theoretic | 99.9% signal energy in SX loopsS4–SX knockout removes 99.5%$D_{\rm eff} = 2.0$ | Information-theoretic (no $\sigma$) | ✓ Yes (structural) | ✓ Structural |

#### Table 2: Supplementary & Cross-System Checks

| Evidence strand | Test type | Result | $p$-value / significance | Independent? | Status |
| --- | --- | --- | --- | --- | --- |
| *TDCOSMO+Encore Shear* Predicted sensitivity check, $n=18$ pairs | Predicted-sensitivity consistency check | 16/18 pairs: predicted shift exceeds $1\sigma$Spearman $\rho = -0.733$ (tautological by construction)No independent information | Predicted shift magnitude vs. published uncertainty | ✓ Yes (different systems) | ✓ Predicted (structural consistency; not an observed detection) |
| *SN Encore blind-prediction residual* 8 blind models, 1 delay pair | Single-pair residual test | $\mathcal{R}_{\rm obs} = -3.33 \pm 2.13$ dPredicted: $-0.49$ d ($-1.04$ d at Refsdal $\kappa_{\rm lens}$)Sub-day shift swamped by scatter ($\sim$3–50 d)4/8 positive ($p = 0.64$) | Consistent with GR; directional sign matches TEP prediction | ✓ Yes (different system) | ✓ Observed (consistency check only) |
| *SN H0pe blind-prediction residual* 7 delay-blind models, 2 delay pairs | Single-pair residual test (AB, CB) | AB: $\mathcal{R}_{\rm obs} = -10.16 \pm 5.13$ d; predicted $-1.58$ d ($-3.32$ d at Refsdal $\kappa_{\rm lens}$)CB: $\mathcal{R}_{\rm obs} = -0.23 \pm 2.67$ d; predicted $-0.26$ d ($-0.54$ d at Refsdal $\kappa_{\rm lens}$)Sub-day; swamped by scatter | AB sign matches TEP prediction; CB near zero and not probative | ✓ Yes (different system) | ✓ Observed (consistency check only) |
| *Cross-system trio (Refsdal + Encore + H0pe)* Sign consistency only | Directional sign consistency across 3 independent systems | 3/3 systems: primary signs consistent with TEPExact binomial $p = 0.125$Stouffer combination: sensitivity exploration only | $p = 0.125$ (exact binomial) | ✓ Yes (different systems; caveat: kappa extrapolation) | ✓ Observed (consistency check) |
| *Non-independent consistency checks (same SX-dominated data; reported for transparency only)* |  |  |  |  |  |
| *Delay–$\mu$ correlation* Pearson $r=0.93$ (SX-leveraged), $n=5$ | Correlation test | Pearson $r=0.932$ (SX-leveraged); $r \approx 0.15$ without SXCook's distance $>1.0$Spearman $\rho=0.3$ ($p=0.31$)Theil-Sen $86 \pm 210$ d (overlaps zero)*Not probative: $n=5$, one extreme leverage point* | Not meaningful as formal test | ✗ No (SX-driven; correlated with strand 1) | ✓ Observed (reported for transparency) |
| *Per-model $\kappa_{\rm lens}$ inference* $\bar{\kappa}_{\rm inferred} \approx -0.114$ | Parameter inference | Weighted mean $\kappa_{\rm inferred} = -0.114 \pm 0.034$All 8 models $\kappa_{\rm lens} < 0$ | $p=0.0006$ vs. zero (one-sided) | ✗ No (same ensemble) | ✓ Observed |
| *$\chi^2$ model comparison* GR vs. TEP ensemble fit | Goodness-of-fit | $\Delta\chi^2 = +8.4$ in favour of proxy model28% wRMS reduction after proxy correction (8/8 models) | No formal $p$-value: fixed predictions (0 free params), not nested. Individual GOF: GR $p=0.023$, proxy $p=0.32$ | ✗ No (same ensemble) | ✓ Observed |
| *Directional-odds Bayes factor* Sign data recast in odds form | Bayesian directional sign model | $\mathrm{BF}_{10}=56.8$ (all), 18.1 (delay-blind), 10.5 (family-collapsed)Odds support for one-sided sign excess | Odds support for one-sided sign excess | ✗ No (same sign data) | ✓ Observed (complementary to sign tests) |
| *Proxy-agnostic rank test* Step 34, ansatz-free | Rank correlation (delays vs potential depth) | Delay–kappa $\rho = -0.15$ ($p = 0.63$); delay–mu $\rho = -0.30$ ($p = 0.74$)Non-significant; $n=5$ too small to separate claims | Not meaningful as formal test | ✗ No (same SX data) | ✓ Observed (reported for transparency) |

None of these strands is individually decisive. The observed tests point in a coherent direction: the sign is right across seven delay-blind model variants spanning five modelling families (strict original-blind floor: six of seven), the structural signal-energy concentration confirms a single-contrast geometry, and the implied coupling is self-consistent by construction — but shared lensing inputs prevent treating the models as fully independent draws (break-even inter-model correlation $\rho \approx 0.08$ for the delay-blind Wilcoxon to reach $p = 0.05$). The addition of two additional systems (SN Encore, SN H0pe) with blind-prediction residual tests is directionally consistent: all three systems show primary residual signs matching TEP predictions (3/3, $p = 0.125$ by exact binomial). However, Encore and H0pe do not provide independent directional support at any meaningful precision—their predicted TEP shifts are sub-day to $\lesssim 2$ d, far below their per-model scatter ($\sim$5–50 d)—and serve as consistency checks rather than high-precision evidence strands. A cross-system Stouffer combination is therefore not reported in the headline evidence chain; it inherently dilutes the Refsdal signal and is reserved as a sensitivity exploration.

The Pearson correlation between delay and inverse-magnification ($r = 0.93$) is driven entirely by SX and is not statistically independent of the sign-based strand; it is reported for transparency but is not counted toward the headline significance. The proxy-agnostic rank test (Step 34) confirms that with $n=5$ the physical and parametric claims are not separately constrained; only the binary S4–SX sign-contrast is probative. Because several strands are correlated through the same SX-dominated sample, the most defensible headline is the exact family-sign-flip test giving $p = 0.031$ ($z \approx 1.86\sigma$). The Wilcoxon signed-rank test gives the independence-assuming benchmark ($p = 0.0078$, $\approx 2.4\sigma$), with the all-eight-model Wilcoxon ($p = 0.0039$, $2.8\sigma$) reported as a supplementary consistency check.

The key probative point is that the direction of the SX residual matches the proxy-model prediction across seven delay-blind model variants spanning five modelling families, none of which had any knowledge of TEP when their predictions were made. The probability that independent random sign scatter would produce the delay-blind pattern is $p = 0.0078$ under the standard Wilcoxon test (all 7 non-zero delay-blind residuals positive), while the exact family-sign-flip test gives $p = 0.031$ after method-family dependence is accounted for. The supplementary all-eight-model Wilcoxon gives $p = 0.0039$. Shared lensing inputs and community-level modelling systematics prevent treating the models as fully independent draws; under a beta-binomial correlation model the break-even inter-model correlation for the delay-blind Wilcoxon to reach $p = 0.05$ is $\rho \approx 0.08$ (step_11).

The evidence presented here is best characterised as follows. The observed data exhibit a coherent, multi-pronged observational pattern — multiple evidence tests pointing in the direction predicted by the proxy model after calibration of one operational response coefficient. This constitutes directional evidence at the correlation-aware headline ($p = 0.031$, $z \approx 1.86\sigma$), with the independence-assuming Wilcoxon benchmark at $p = 0.0078$ ($\approx 2.4\sigma$). Hierarchical Bayesian model comparison remains inconclusive (BF $\sim 1$), so decisive model-selection evidence awaits tighter lens-model uncertainties. The calibrated correction is descriptive: it reduces scatter where GR predicts none, but it does not constitute a parameter-free prediction.

The directional evidence from SN Refsdal is robust at the correlation-aware headline ($p = 0.031$, $z \approx 1.86\sigma$), with all seven non-zero delay-blind residuals matching the predicted sign (strict original-blind floor: six of seven). The independence-assuming Wilcoxon benchmark gives $p = 0.0078$ ($\approx 2.4\sigma$) but assumes between-model independence that shared lensing inputs do not justify. The lens model uncertainties of $\pm$16–60 d bound the formal model-selection significance, and reducing these below $\sigma < 5$ d would move the amplitude test toward a decisive regime. The existing data are consistent with the proxy model at the empirically determined coupling, and the sign-based evidence is stable across seven delay-blind model variants spanning five modelling families.

### 4.8 Numerical Exercise: Low-$H_0$ Consistency Check

A cross-system numerical exercise applies the empirically determined proxy-model coupling ($\kappa_{\rm lens} \approx -0.055$) to the low-$H_0$ measurements from lensed supernovae. Standard GR analyses of SN Refsdal (Kelly et al. 2023), SN Encore (Pierel et al. 2026), and SN H0pe (TD-only; Pierel et al. 2024) yield $H_0 \approx 61-67$ km s$^{-1}$ Mpc$^{-1}$.

The proxy model predicts that systems where the most magnified image arrives first will have GR-inferred $H_0$ biased low, because the observed delay exceeds the geometric delay. Applying the empirically determined coupling, SN Refsdal shifts upward by $2.7$ km s$^{-1}$ Mpc$^{-1}$ ($66.6 \to 69.3$). With the full error budget — combining the lens-model uncertainty ($\sigma \approx 3.7$ km s$^{-1}$ Mpc$^{-1}$) and the Planck uncertainty ($\pm 0.5$) in quadrature — the tension with Planck ($67.4 \pm 0.5$) is $0.5\sigma$ ($p \approx 0.62$ two-sided), not a resolution of the $H_0$ tension. The lens-model uncertainty dominates; the shift is a small correction swamped by measurement noise.

As discussed in §4.4, this is a *definitional internal consistency check*: $\kappa_{\rm lens}$ was calibrated on the same SN Refsdal SX delay, so the $H_0$ shift is mathematically prescribed by that calibration and cannot be read as independent confirmation. SN Encore and SN H0pe receive only $\sim 0.1$ km s$^{-1}$ Mpc$^{-1}$ shifts because their modest magnification contrasts suppress the per-delay gamma factors. The differential prediction — strong correction for high-contrast systems, negligible for modest-contrast ones — is a falsifiable signature, but it is not independent evidence. A genuine independent test requires an unblinded long-baseline multiply-imaged supernova with blind lens-model predictions and measured delays both available without circular calibration on the same feature, such as the forthcoming SN 2025wny system (§4.9).

### 4.9 SN 2025wny: The Next Target

SN 2025wny ($z_s = 2.011$, $z_l = 0.375$, Johansson et al. 2025, ApJ 995, L17; Taubenberger et al. 2025, arXiv:2510.21694) is the first resolved, multiply-imaged superluminous supernova (SLSN-I), with four images (A–D) in an Einstein cross pattern separated by ~1.7 arcsec. With a magnification factor estimated at $\mu \sim 20$–50 for the brightest image, the system has a large potential contrast between images—precisely the regime where proxy-model predicted discrepancies are largest.

Photometric monitoring of SN 2025wny is ongoing (Maidanak, Lulin, COLIBRI, Wendelstein). HST (PID 17611, October and November 2025) and JWST (PID 5564, November 2025) follow-up imaging and IFU spectroscopy were executed (PI: Goobar). Time-delay measurements from these data are not yet published as of May 2026.

A post-hoc geometric model predicts a long-baseline delay of ~175 days between the trailing image (A) and the leading image (D), with shorter A–C (~20 d) and A–B (~30 d) delays (Witt–Wynne model, arXiv:2605.11090). At the measured coupling $\kappa_{\rm lens} \approx -0.055$ and with a magnification contrast $\mu_{\rm max}/\mu_{\rm min} \sim 10$–50, the proxy model predicts a substantial TEP residual on the long A–D baseline—approaching SN Refsdal's S4–SX contrast in the high-magnification tail. For a ~175-day baseline and $\Delta\log_{10}\mu \sim 0.7$–1.5, the predicted residual is of order several days, with a broad uncertainty envelope once the propagated $\kappa_{\rm lens}$ uncertainty is included. The exact value depends on the true magnification contrast, which remains uncertain; the $\mu \sim 20$–50 estimate is derived from light-curve comparison and may be revised once lens-model magnifications are available.

*Forward-projected falsification thresholds (Step 36, updated with post-hoc geometric baseline).*

For a four-image Einstein-cross system with estimated magnification $\mu \sim 20$–50 and a long-baseline delay of ~175 days, the proxy model predicts a fractional TEP shift of order $2$–$6\%$ of the longest delay baseline. The predicted proxy-model residual (Step 36 Monte Carlo, $10^5$ draws over magnification, baseline, and $\kappa_{\rm lens}$ priors) is:

- *Predicted residual range:* $\mathcal{R}_{\rm pred} = +1.5$ to $+11.3$ days (16th–84th percentile; median ~+5.2 days; 95th percentile ~16.6 days), assuming the post-hoc ~175-day A–D baseline, the estimated magnification contrast, and the propagated $\kappa_{\rm lens}$ uncertainty. *This prediction is post-hoc: it uses a geometric model published after the SN discovery and is therefore not a blind prediction. A genuine blind-prediction residual test requires lens-model predictions published before time-delay measurements.*

- *Required lens-model precision for a $3\sigma$ test:* per-model blind-prediction uncertainty $\sigma_{\rm model} \lesssim 1.7$ days on the longest pairwise delay, assuming the residual falls near the median prediction.

- *Required lens-model precision for a $5\sigma$ test:* $\sigma_{\rm model} \lesssim 1.0$ days.

*Falsification condition.* A future long-baseline lensed supernova with comparable magnification contrast but a blind observed-minus-GR residual consistent with zero at the $\lesssim 1$ d level would strongly falsify the linear log-magnification response in the strong-lensing domain. Conversely, a residual consistent with the predicted $+1.5$ to $+11.3$ day 68\% range would constitute independent geometric evidence for potential-dependent temporal propagation. These thresholds are computed deterministically by the pipeline (Step 36); they will not be adjusted post-measurement.

### 4.10 Alternative Astrophysical Explanations

Before attributing the observed positive residual to the proxy model, it is necessary to consider whether conventional astrophysical effects could produce the same sign pattern across multiple independent modelling methods. Several alternative explanations are examined below.

#### 4.10.1 Unmodelled Cluster Substructure

Unmodelled dark matter subhalos or line-of-sight mass concentrations could in principle introduce systematic errors in lens model predictions. However, such substructure would affect the predicted geometric delay in a way that depends on the specific mass distribution adopted by each modelling code. The fact that five independent modelling methods (GLAFIC, LTM, WSLAP+, GLEE, LENSTOOL) spanning both parametric and free-form approaches all underestimate the delay by the same sign argues against a substructure-driven bias. Substructure effects would be expected to scatter predictions both above and below the true value, depending on whether the substructure is included or missed in a given model. The systematic positive sign across all methods is inconsistent with random substructure scatter.

#### 4.10.2 Line-of-Sight Mass Convergence

Mass structures along the line of sight to the cluster can introduce external convergence ($\kappa_{\rm ext}$) that rescales all time delays by a common factor $(1-\kappa_{\rm ext})$. This is the external component of the Mass Sheet Degeneracy. Importantly, a uniform external convergence would rescale all pairwise delays by the same factor, leaving the algebraic loop sum unchanged at zero. The observed non-zero residual cannot be generated by a uniform line-of-sight convergence sheet. Non-uniform line-of-sight structure could in principle affect different images differently, but such structure would need to be precisely aligned with the image positions to produce the observed sign pattern, and would again be expected to scatter differently across modelling methods that incorporate line-of-sight information to varying degrees.

#### 4.10.3 Microlensing-Induced Flux Bias

Microlensing by stars in cluster member galaxies can perturb observed fluxes, potentially biasing the flux-ratio-based magnification proxies used to infer $\Gamma_t$ factors. This systematic is addressed in §2.4.1 and tested via the microlensing-nuisance Monte Carlo in §3.6.5. The key point is that microlensing perturbations at the tens-of-percent level can shift the inferred amplitude of $\Delta\Gamma$ but do not naturally invert the rank-ordering $\mu_{\rm SX} < \mu_{\rm S4}$ that sets the sign of the predicted SX residual. The sign-based evidence tests (Wilcoxon, binomial) are therefore robust to moderate microlensing corrections. Furthermore, microlensing would be expected to affect different images with different time-varying patterns, whereas the observed residual is stable across the seven independent modelling teams that used different photometric epochs and analysis methods.

#### 4.10.4 Source-Size Effects

The finite size of the supernova host galaxy or the supernova light curve itself could in principle affect the inferred time delays if the lens models assume point-source approximations. However, Kelly et al. (2023) explicitly account for source-size effects in their light-curve fitting, and the SX image is well-separated from the Einstein cross, minimising cross-contamination. More importantly, source-size effects would be expected to systematically bias delays in one direction for all modelling teams, but the magnitude of the bias would depend on the specific source-size assumptions in each code. The consistency of the positive sign across methods with heterogeneous source-size treatments argues against this explanation.

#### 4.10.5 Photometric Systematics

Systematic errors in the measured SN fluxes or in the model-predicted flux ratios could bias the magnification proxy estimates. However, the primary evidence test (§3.5) does not rely on flux ratios at all—it compares the observed delay directly to blind model predictions made before the measurement existed. The flux-ratio-based evidence (delay–magnification correlation, per-model coupling inference) is treated as supplementary and explicitly caveated. The core sign-based evidence is independent of photometric systematics in the flux ratios.

Taken together, these alternative explanations are either ruled out by the geometric structure of the blind-prediction residual test (uniform convergence cannot produce a differential residual), inconsistent with the multi-method sign pattern (substructure, line-of-sight), or robust to the specific test design (microlensing, photometric systematics). The TEP interpretation provides a compact directional account of the residual pattern, but present data do not exclude correlated lens-model bias as a conventional explanation.

#### 4.10.6 Correlated Lens-Model Systematics

A referee's concern is that the blind lens models may not be statistically independent: they share common assumptions about the cluster mass distribution, the same HST imaging, and similar parametric halo prescriptions. If the models are correlated, the effective sample size is smaller than the nominal $N=8$ (or $N=7$ delay-blind), and the significance of the sign tests drops. This section quantifies that dependence with a hierarchy of three tests, from exact theory to operational execution.

*Test hierarchy under dependence.*

- *Exact family-sign-flip test (delay-blind headline).* Because the Wilcoxon statistic lacks a closed-form variance under exchangeable intra-class correlation, the most rigorous dependence-aware rank bound is an exact sign-flip enumeration over method-family clusters, preserving the empirical cluster structure. For the delay-blind subset (seven revised values fixed before the measured delay was known), this gives $p = 0.031$ (one-sided). It is exact under the sharp null and makes no superpopulation assumption. This is the operational *correlation-aware primary rank test*.

- *Exact family-sign-flip test (strictly blind floor).* The same test computed on the original pre-reappearance published predictions gives $p = 0.094$ (one-sided, 6/7 positive, 4/5 families). This is the floor: even if every model family had been fully independent, the sign pattern would not reach conventional significance on the strictly blind layer alone. The probative value of the headline therefore rests on the honesty of the delay-blind provenance claim — that the revised values were fixed before the 376-day delay was known.

- *Method-family block-bootstrap.* Resamples method families with replacement and computes the Wilcoxon statistic on each bootstrap draw, yielding a distribution of dependence-adjusted $p$-values. It respects the empirical cluster structure without assuming a parametric correlation model, but can occasionally reconstruct more extreme statistics than independent sampling when clusters concentrate rank mass. It is reported as a *sensitivity exploration* rather than the operational primary. The delay-blind subset yields $p_{\rm median} = 0.0078$ [$0.0039$, $0.0156$]; the strictly blind subset yields $p_{\rm median} = 0.0469$ [$0.0039$, $0.2344$].

- *Binomial sign test under beta-binomial correlation.* The binary sign of each residual is treated as an exchangeable outcome with intra-class correlation coefficient $\rho$. For conservative diagnostics that count any exact zero residual as non-positive, the all-eight sign count is 8/8 positive ($p=0.0039$ under independence), the delay-blind count is 7/7 positive ($p=0.0078$), and the strictly blind count is 6/7 positive ($p=0.0625$). These p-values rise above, or already sit above, the conventional threshold under modest inter-model correlation. The beta-binomial sign test is therefore reported only as the most *conservative correlation-aware sign diagnostic*, not as the headline sign result.

The exact family-sign-flip test ($p = 0.031$ on the delay-blind tier; $p = 0.094$ on the strictly blind floor) is the most rigorous dependence-aware rank bound because it is exact under the sharp null and requires no superpopulation assumption. The block-bootstrap ($p_{\rm median} = 0.0078$ [$0.0039$, $0.0156$] delay-blind; $p_{\rm median} = 0.0469$ [$0.0039$, $0.2344$] strictly blind) is reported as a sensitivity exploration: it can occasionally fall below the independent $p$-value when the empirical cluster structure concentrates rank mass, so it is not used as the operational primary. The beta-binomial sign diagnostic is deliberately conservative: with the present data (all signs positive), it gives $p=0.0078$ for the delay-blind count at $\rho=0$, rising above $0.05$ only if inter-model correlation exceeds the break-even threshold. Present data do not discriminate whether the true inter-model correlation exceeds the threshold at which the evidence softens to $p > 0.05$. Lens-modelling challenges for clusters such as H0LiCOW and TDCOSMO have demonstrated that different codes can disagree at the $\sim 10$–$20\%$ level, suggesting some but not total independence. The TEP interpretation offers a single compact parameter ($\kappa_{\rm lens} \approx -0.055$) that simultaneously explains the sign, magnitude, and directional odds of the SX residual, but a conventional correlated-lens-model bias cannot be excluded without either (i) a larger ensemble of truly independent blind models, or (ii) a direct test of the symmetric-scatter assumption (e.g., comparing blind and post-blind predictions for the same system).

#### 4.10.7 Discriminating Temporal Shear from Generic Modelling Bias (Steps 41–42)

Two dedicated tests separate the TEP interpretation from the leading conventional explanations of the same-sign residual pattern.

*Cross-system amplitude discriminator (Step 41).* If the residuals reflect potential-dependent temporal shear, their *magnitudes* should scale with the proxy-model sensitivity (large for SN Refsdal's S4–SX contrast, sub-day for Encore and H0pe). If instead they reflect a generic $H_0$/mass-sheet-like fractional bias, the magnitudes should scale only with the delay baseline. Fitting the four cross-system contrasts (Refsdal, Encore, H0pe-AB, H0pe-CB) to four competing models — null, constant offset, uniform fractional ($r_i = c\,|\Delta t_i|$), and TEP proxy — and comparing by AIC, the uniform fractional offset is in fact the AIC-preferred description ($\Delta\mathrm{AIC} = 2.3$ relative to the TEP proxy), because the observed Encore and H0pe residuals exceed their proxy-model predictions by factors of $\sim$6. The amplitude of the cross-system residuals therefore does *not* discriminate TEP from a generic baseline-scaling systematic. We caution the reader that this AIC preference for a uniform fractional offset reflects the current dominance of conventional lens-modeling scatter in the multi-system amplitude budget, not a falsification of the TEP proxy. The observed Encore and H0pe residuals are comparable to or larger than their per-model uncertainties, so the cross-system variance is driven by lens-model dispersion rather than by a precisely measured TEP-predicted amplitude. What remains non-trivial is the *sign*: all four contrasts align with the potential-depth ordering (4/4; one-sided binomial $p = 0.06$), a pattern a uniform fractional or constant offset does not predict, since such a systematic carries no potential-depth information and would scatter the per-contrast sign. This reinforces the central position of this paper: the directional sign, not the amplitude, is the carrier of evidence.

*Precision-persistence (Step 42).* The leading conventional account of the SN Refsdal SX under-prediction is that the blind models were simply imprecise, and a sufficiently accurate GR model would converge to the observed delay with zero residual. Regressing the per-model residual on its quoted uncertainty $\sigma_{\rm model}$ and extrapolating to the high-precision limit ($\sigma_{\rm model}\to 0$) gives a residual intercept of $-8.2 \pm 18.3$ d (all eight models, one-sided $p = 0.67$) or $-15.9 \pm 21.6$ d (blind subset, $p = 0.77$). The intercept is consistent with (or below) zero, so a high-precision GR model could plausibly remove the residual — immaturity is not excluded. The most precise all-eight model (Grillo+2024, $\sigma = 16$ d) retains a $+14$ d residual, while the most precise blind model (Oguri-a*, $\sigma = 19$ d) retains a $+22$ d residual. The residual does not clearly shrink as precision improves, but the regression does not support a non-vanishing intercept; a decisive test requires a larger ensemble of independent high-precision blind models.

Together these tests sharpen rather than inflate the evidence: they confine the probative content to the directional sign channel and establish the amplitude as non-probative, consistent with §4.4 and §4.7.

### 4.12 The Final Proxy: Status in the TEP Framework

With the cosmological sector closed from first principles (TEP-C0, Paper 26) and weak-field astrophysical response coefficients now derived across pulsar spin-down, Cepheid period–luminosity, and galaxy stellar populations (Papers 10–12), the strong-lensing temporal-shear magnification proxy tested here is the *sole remaining macroscopic proxy* in the entire TEP framework. All other sectors — cosmic expansion, gravitational wave propagation, atomic-clock frequency shifts, satellite orbital dynamics, and stellar-population chronometry — are either derived directly from the scalar-field action or constrained by screened-limit consistency. The strong-lensing projection operator $\mathcal{P}_\mu$ that maps the scalar-field gradient $\nabla_\mu\phi$ through the lensing Jacobian $\mathcal{A}_{ij}$ to the observable magnification shift remains the one transfer kernel that has not yet been derived from the action. The operational log-magnification response $\Gamma_t(i) = 1 + \kappa_{\rm lens}\log_{10}(\mu_{\rm norm}(i))$ is a phenomenological parameterisation that captures the correct sign and approximate amplitude of the observed residual, but it is not a first-principles prediction. Deriving the precise flux-to-potential transfer function from the scalar-field action is the decisive theoretical target that would convert the present directional evidence into a quantitative, parameter-free forecast.

### 4.11 Precision Requirements for Decisive Evidence

The headline single-test significance from SN Refsdal is $z = 1.86\sigma$ ($p = 0.031$; exact family-sign-flip, correlation-aware). The independence-assuming Wilcoxon benchmark is $z \approx 2.4\sigma$ ($p = 0.0078$; delay-blind subset). The limiting factor is not the size of the proxy-model signal — the 14.5-day predicted shift is far above the 5.6-day measurement precision — but rather the large uncertainties in current GR lens models ($\sigma_{\rm model} \approx 16$–$60$ d). Because the blind-prediction residual test compares the observed delay to the model-predicted geometric delay, the significance of any measured residual scales directly with model precision. The analysis therefore reports the correlation-aware exact family-sign-flip result as the headline and the Wilcoxon as the independence benchmark (see §4.7 and §4.10.6 for the treatment of correlated evidence).

The precision required to overcome this limitation can be quantified by simulating the ensemble significance for $N=8$ independent models as a function of the average per-model uncertainty $\sigma_{\rm model}$, assuming the true proxy-model signal is $\mathcal{R}_{\rm TEP/GR} = 14.5$ d.

The simulation shows conditional detection thresholds: if the average model uncertainty were to drop below $\sigma_{\rm model} = 13.7$ d, the same 14.5-day mean residual would cross the $3\sigma$ "evidence" threshold. At $\sigma_{\rm model} = 8.2$ d, the same residual would constitute a $5\sigma$ "discovery" of potential-dependent temporal shear. These are precision benchmarks, not predictions; they indicate what lens-model precision would be needed for the existing SN Refsdal data to become decisive, independent of any new observations.

These thresholds are not merely theoretical. Recent extended-source lens modeling of SN Refsdal's host galaxy, incorporating 77,000 HST pixels in addition to the 106 point-like multiple images, achieves *sub-percent statistical uncertainty* on the predicted $\Delta t_{\rm SX:S1}$ time delay — more than an order of magnitude smaller than the ~5% uncertainty of the best point-like models (Schuldt et al. 2026, *A&A*, aa57680-25, arXiv:2602.12329). This demonstrates that the $\sigma_{\rm model} < 8.2$ d threshold required for a $5\sigma$ TEP discovery is achievable with existing HST data and established extended-image modeling techniques. The caveat is that these particular models are post-observation and therefore not blind; a genuine blind-prediction residual test would require comparable precision from models published before the time-delay measurement. Nevertheless, the precision roadmap is now an engineering question, not a fundamental limitation.

## 5. Conclusion

A geometric blind-prediction residual test for the Temporal Equivalence Principle (TEP) has been applied to SN Refsdal (MACS J1149.6+2223), currently unique among lensed supernovae in having five resolved images and precision-measured relative time delays. The key results are:

- The algebraic loop sum is identically zero under any theory with globally assignable arrival times per image, including the TEP ansatz. The probative observable is the blind-prediction residual (§1.1).

- The inner Einstein cross is a non-probative region under the adopted proxy (shear degeneracy collapses the magnification–convergence rank order; §4.1.1). All probative content resides in the S4–SX contrast.

- The two loops incorporating image SX—which arrives 376 days after S1—yield post-hoc proxy sensitivities of $-8.5$ days (S1–S2–SX) and $-14.5$ days (S1–S4–SX closure; $+14.5$ d obs−model) at nominal $\kappa_{\rm lens} = -0.055$. These define the nominal S4–SX response normalization used to compare sign and scale across the delay-blind ensemble. The 376-day baseline is the high-leverage contrast for the sign test.

- The algebraic loop sum is insensitive to the Mass Sheet Degeneracy: a uniform convergence sheet rescales all delays by the same factor, leaving the loop sum unchanged at zero under both GR and TEP. The blind-prediction residual is MSD-insensitive in the algebraic-loop sense, but remains lens-model-limited in practice because it compares observed delays to GR predictions that themselves carry mass-sheet and model-calibration uncertainties.

- The correlation-aware headline is the exact family-sign-flip test on method-family signs ($p=0.031$, $z \approx 1.86\sigma$), supported by independence-assuming Wilcoxon benchmarks ($p=0.0078$ delay-blind, $p=0.0039$ all eight). A complementary directional-odds analysis gives one-sided Bayes factors ($\mathrm{BF}_{10}=56.8$ all non-zero; 18.1 delay-blind; 10.5 method-family-collapsed). Robustness checks preserve the directional pattern under model-dependence stress tests and microlensing-style flux perturbations, while hierarchical Bayesian comparison remains inconclusive (BF $\sim 1$), indicating that present lens-model uncertainties dominate formal model-selection metrics.

- *Cross-system directional consistency (not independent evidence).* Blind-prediction residual tests for two additional multiply-imaged supernovae (SN Encore, SN H0pe) are directionally consistent with the proxy model: all three systems (Refsdal, Encore, H0pe) show primary residual signs matching the TEP sign expectation (3/3, exact binomial $p = 0.125$). Encore and H0pe serve as consistency checks, not high-precision evidence strands, because their predicted TEP shifts are sub-day to $\lesssim 2$ d and swamped by per-model scatter ($\sim$3–50 d). Cross-system magnitude synthesis (Step 40) uses Refsdal bootstrap $\kappa_{\rm lens} \approx -0.118$; sign match is non-circular, magnitude comparison is not independently calibrated.

- *Internal consistency check — $H_0$ (not independent evidence).* This is a calibrated response-scale estimate rather than an independent blind prediction: because $\kappa_{\rm lens}$ was calibrated from the same SN Refsdal SX delay data, applying it back to that same system cannot constitute an independent confirmation. Under the proxy model, the inferred $H_0$ shifts from $66.6$ to $69.3$ km s$^{-1}$ Mpc$^{-1}$ — a calibrated consequence of the empirical coupling, not corroborating evidence. SN Encore and SN H0pe receive only $\sim 0.1$ km s$^{-1}$ Mpc$^{-1}$ corrections (kappa calibrated on Refsdal and applied without refitting), too small to be probative. This $H_0$ shift is therefore reported only as an internal consistency consequence of the calibrated response scale, not as independent cosmological evidence for TEP.

- A future long-baseline multiply-imaged supernova with comparable magnification contrast, blind GR predictions, and an observed-minus-GR residual consistent with zero at the $\lesssim 1$ d level would strongly falsify the linear log-magnification response in the strong-lensing domain. Conversely, a residual matching the proxy-model prediction would constitute direct geometric evidence for potential-dependent temporal propagation once lens-model uncertainties are reduced sufficiently for decisive significance.

- *Phenomenological response vs. fundamental transport.* Direct potential and geodesic reconstructions preserve the sign of the observed residual but predict sub-day amplitudes, while the operational log-magnification response captures the sign at the observationally relevant order of magnitude. This indicates that the operational response is an effective phenomenological description of a lensing-sector temporal shear whose underlying amplification mechanism — possibly connected to the lensing Jacobian near critical curves — has not yet been derived from the scalar-field action. Deriving such a transfer function remains the central open theoretical target.

- The quadruply-imaged SLSN-I SN 2025wny ($z_s = 2.011$, Johansson et al. 2025) provides the most promising future test target once time delays are measured. With magnifications estimated at $\mu \sim 20$–50 and a predicted long-baseline delay of $\sim$175 days from a post-hoc geometric model, the system could yield a proxy-model residual comparable in magnitude to SN Refsdal's S4–SX contrast. HST (PID 17611) and JWST (PID 5564) follow-up is ongoing (PI: Goobar).

*Preregistration commitment.* We formally commit to uploading the precise flux-to-potential transfer function — derived from the TEP scalar-field action through the strong-lensing Jacobian — to an open repository before the JWST time-delay data for SN 2025wny are unblinded. This ensures that the next long-baseline multiply-imaged supernova test will be a genuine blind prediction, with the theoretical forecast locked before the observational data become available.

SN Refsdal provides the strongest current directional evidence for lensing-sector temporal-shear phenomenology: a pre-observation GR prediction ensemble, a later independent SX delay measurement, and a coherent positive residual sign pattern. The headline is sign coherence ($p = 0.031$), not amplitude agreement at nominal proxy sensitivity. Direct potential and geodesic reconstructions preserve the same sign but underpredict the amplitude, showing that the decisive theoretical task is to derive the underlying amplification mechanism that would explain how lensing structure maps into temporal response.

Quantitative analysis confirms this is a uniquely high-leverage single-system test: the probative content is concentrated in the S4–SX temporal lever arm (§4.1.1), with an effective dimensionality $D_{\rm eff} \approx 2.0$. The correlation-aware headline gives $p=0.031$ ($z \approx 1.86\sigma$), supported by independence-assuming Wilcoxon benchmarks. All seven delay-blind-comparison residuals match the predicted sign, while the strict original-blind layer remains positive at six of seven. If the central Refsdal residual remains near the corrected-compilation value, reducing independent lens-model scatter below $\sigma_{\rm model} < 5$ d would move the amplitude test into a decisive regime. Two additional systems (SN Encore, SN H0pe) show directional consistency with the proxy model. Statistical power scales with both model precision and time-delay baseline; additional independent long-baseline multiply-imaged supernovae with high magnification contrast—such as the quadruply-imaged SN 2025wny—will extend the test to decisive significance.

SN Refsdal therefore establishes the first high-leverage sign-coherence test of conformal-sector lensing temporal response, with the corrected-compilation amplitude defining a concrete magnification-amplified transfer-kernel target for derivation from the scalar-field action and prospective testing in the next long-baseline multiply-imaged supernova.

## References

#### Strong Lensing Observations

Kelly, P. L., Rodney, S. A., Treu, T., et al. 2015, *Science*, 347, 1123, "Multiple images of a highly magnified supernova formed by an early-type cluster galaxy lens", doi:10.1126/science.aaa3358

Kelly, P. L., Rodney, S., Treu, T., et al. 2023, *ApJ*, 948, 93, "The SN Refsdal Expansion Rate Measurement: A Powerful New Cosmic Ruler", doi:10.3847/1538-4357/acbf4d

Kelly, P. L., Rodney, S., Treu, T., et al. 2023, *Science*, 380, abh1322, "A measurement of the Hubble constant from the multiply imaged supernova Refsdal", doi:10.1126/science.abh1322

Treu, T., Brammer, G., Diego, J. M., et al. 2016, *ApJ*, 817, 60, "The Refsdal Revelation: Predictions for the Return of the First Multiply Imaged Supernova", doi:10.3847/0004-637X/817/1/60

Grillo, C., Rosati, P., Suyu, S. H., et al. 2024, *ApJ*, 971, 49, "A High-precision Lens Model for MACS J1149.6+2223: Predicting the Reappearance of SN Refsdal", doi:10.3847/1538-4357/ad5a7d

Schuldt, S., et al. 2026, arXiv:2602.12329, "A boost in the precision of cluster-mass models: Exploiting the extended surface brightness of the lensed supernova Refsdal host galaxy" (arXiv preprint; not yet peer-reviewed)

Frye, B. L., Pascale, M., Pierel, J., et al. 2024, *ApJ*, 961, 171, "The JWST Discovery of the Triply Imaged Type Ia "Supernova H0pe" and Observations of the Galaxy Cluster PLCK G165.7+67.0", doi:10.3847/1538-4357/ad1f1d

Pascale, M., Pierel, J. D. R., Frye, B. L., et al. 2025, *ApJ*, 979, 17, "SN H0pe: The First Measurement of H0 from a Multiply Imaged Type Ia Supernova Discovered by JWST", doi:10.3847/1538-4357/ad9c6e

Grayling, M., Thorp, S., Mandel, K. S., et al. 2025, arXiv:2510.11719, "BayeSN-TD: Time Delay and H0 Estimation for Lensed SN H0pe" (arXiv preprint)

Pierel, J. D. R., Suyu, S. H., Acebron, A., et al. 2026, arXiv:2509.12301, "SN Encore: Time Delay Cosmography of a Lensed Type Ia Supernova in MACS J0138.2-2155" (arXiv preprint; not yet peer-reviewed)

Suyu, S. H., Acebron, A., Pierel, J. D. R., et al. 2026, arXiv:2509.12319, "Blind Time Delay Predictions for SN Encore from Eight Independent Lens Models" (arXiv preprint; not yet peer-reviewed)

Johansson, J., Goobar, A., Suyu, S. H., et al. 2025, *ApJ*, 995, L17, "SN 2025wny: A Quadruply Imaged Superluminous Supernova Discovered by JWST", doi:10.3847/2041-8213/adb3f0

Taubenberger, S., Acebron, A., Cañameras, R., et al. 2025, arXiv:2510.21694, "HOLISMOKES XIX: SN 2025wny at z=2, the first strongly lensed superluminous supernova" (arXiv preprint)

Wynne, R. A. & Schechter, P. L. 2018, arXiv:1803.02722, "The Geometry of Quadruple Lenses: The Singular Isothermal Ellipsoid" (arXiv preprint)

Coulter, D. A., et al. 2026, *in preparation*, "SN Eos: A Multiply-Imaged Type II Supernova at z=5.13 from JWST/VENUS"

Cañameras, R., Schuldt, S., Suyu, S. H., et al. 2020, *A&A*, 644, A163, "HOLISMOKES II: Identifying galaxy-scale strong gravitational lenses in Pan-STARRS using convolutional neural networks", doi:10.1051/0004-6361/202039071

Falco, E. E., Gorenstein, M. V., & Shapiro, I. I. 1985, *ApJ*, 289, L1, "Modeling of the Einstein Ring in MG 1131+0456", doi:10.1086/184605

#### Cosmology & H0 Measurements

Planck Collaboration, Aghanim, N., Akrami, Y., et al. 2020, *A&A*, 641, A6, "Planck 2018 results. VI. Cosmological parameters", doi:10.1051/0004-6361/201833910

Riess, A. G., Yuan, W., Macri, L. M., et al. 2022, *ApJ*, 934, L7, "A Comprehensive Measurement of the Local Value of the Hubble Constant with 1 km/s/Mpc Uncertainty from the Hubble Space Telescope and the SH0ES Team", doi:10.3847/2041-8213/ac5c5b

#### Hubble Tension Reviews

Di Valentino, E., Mena, O., Pan, S., et al. 2021, *Classical and Quantum Gravity*, 38, 153001, "In the realm of the Hubble tension—a review of solutions", doi:10.1088/1361-6382/ac086d

#### TEP Framework (This Series)

Smawfield, M. L. 2025, "Temporal Equivalence Principle: Dynamic Time & Emergent Light Speed" (Paper 0, Jakarta), doi:10.5281/zenodo.16921911

Smawfield, M. L. 2025, "Temporal-Spatial Coupling in Gravitational Lensing" (Paper 5, Tortola), doi:10.5281/zenodo.17982540

Smawfield, M. L. 2025, "Temporal Topology Saturation Scale: Cross-Scale Consistency of ρ_T" (Paper 7), doi:10.5281/zenodo.18064365

Smawfield, M. L. 2025, "What Do Precision Tests of General Relativity Actually Measure?" (Paper 10, Istanbul), doi:10.5281/zenodo.18109760

Smawfield, M. L. 2026, "The Temporal Equivalence Principle: Suppressed Density Scaling in Globular Cluster Pulsars" (Paper 11, Sintra), doi:10.5281/zenodo.19454620

## Appendix A: Theoretical Framework for Lensing Closures

### A.1 Temporal Shear and the Lensing Potential

The Temporal Equivalence Principle (TEP) postulates that the rate of proper time flow for a photon traversing a gravitational potential $\Phi$ is scaled relative to the cosmological background rate by a factor $\Gamma_t$:

\begin{equation} \label{eq:8_appendix_01}
\Gamma_t(\Phi) = 1 + \kappa_{\rm lens} \frac{|\Phi|}{c^2}
\end{equation}

where $\kappa_{\rm lens}$ is the lensing-sector effective coupling. The proxy-model coefficient is empirically $\kappa_{\rm lens} \approx -0.055$; under the corrected Kelly+2023 Supplementary Table S4 compilation, the SN Refsdal data infer a lensing-sector coupling $\kappa_{\rm inferred} = -0.114 \pm 0.034$, which exceeds the proxy value by $1.75\sigma$. In the strong lensing regime, the relevant potential is the projected gravitational potential $\psi(\boldsymbol{\theta})$ integrated along the line of sight. For singular isothermal sphere (SIS) or NFW profiles typical of cluster lenses, the projected potential depth scales logarithmically with the surface mass density $\Sigma$, which is directly related to the magnification $\mu$.

No first-principles derivation of the log-magnification ansatz from the TEP scalar-field action through the lensing potential $\psi(\boldsymbol{\theta})$ currently exists. The mapping from projected convergence $\kappa$ to observed magnification $\mu$ involves the full boundary-value problem of the scalar-field equation in a non-trivial cluster potential, including ellipticity, external shear, and substructure. Solving this transfer function is beyond the scope of the present work.

The following phenomenological scaling relation is therefore adopted as a strong-lensing-specific ansatz, analogous to the PPN strategy of treating the observable response separately from the microscopic coupling:

\begin{equation} \label{eq:8_appendix_02}
\Gamma_t(i) \approx 1 + \kappa_{\rm lens} \log_{10}\left(\frac{\mu_i}{\bar{\mu}}\right)
\end{equation}

where $\mu_i$ is the absolute magnification of image $i$, and $\bar{\mu}$ is the mean magnification of the image system. This logarithmic scaling captures the essential feature that temporal shear is differential: it depends on the *ratio* of potential depths probed by different images. The test is a screen of this ansatz, not a measurement of the fundamental coupling.

### A.2 Derivation of the Predicted GR Discrepancy

Let $t_i^{\rm GR}$ be the arrival time of image $i$ predicted by General Relativity and $t_0$ the unlensed arrival time. Define the GR absolute delay $u_i = t_i^{\rm GR} - t_0$ and the GR pairwise delay $\Delta t_{ij}^{\rm GR} = t_j^{\rm GR} - t_i^{\rm GR} = u_j - u_i$. Under TEP, the observed arrival time is scaled by the path-specific shear $\Gamma_i$:

\begin{equation} \label{eq:8_appendix_03}
t_i^{\rm obs} = t_0 + \Gamma_i (t_i^{\rm GR} - t_0) = t_0 + \Gamma_i u_i
\end{equation}

The observed pairwise delay between images $i$ and $j$ is therefore:

\begin{equation} \label{eq:8_appendix_04}
\Delta t_{ij}^{\rm obs} = t_j^{\rm obs} - t_i^{\rm obs} = \Gamma_j u_j - \Gamma_i u_i
\end{equation}

Because observed arrival times are physical clock readings, the sum of observed pairwise delays around any closed loop is identically zero:

\begin{equation} \label{eq:8_appendix_05}
\Delta t_{ij}^{\rm obs} + \Delta t_{jk}^{\rm obs} + \Delta t_{ki}^{\rm obs} = (\Gamma_j u_j - \Gamma_i u_i) + (\Gamma_k u_k - \Gamma_j u_j) + (\Gamma_i u_i - \Gamma_k u_k) = 0
\end{equation}

The TEP signature is therefore not a failure of the observed delays to close, but a discrepancy between the observed delays and the delays predicted by GR lens models. The TEP predicted GR discrepancy is defined as the weighted sum of GR pairwise delays with coefficients $(\Gamma_m - 1)$:

\begin{equation} \label{eq:8_appendix_06}
\mathcal{R}_{\rm TEP}(i,j,k) \equiv (\Gamma_i - 1)\,\Delta t_{ij}^{\rm GR} + (\Gamma_j - 1)\,\Delta t_{jk}^{\rm GR} + (\Gamma_k - 1)\,\Delta t_{ki}^{\rm GR}
\end{equation}

Substituting $\Delta t_{ij}^{\rm GR} = u_j - u_i$ and expanding:

\begin{equation} \label{eq:8_appendix_07}
\mathcal{R}_{\rm TEP} = (\Gamma_i - 1)(u_j - u_i) + (\Gamma_j - 1)(u_k - u_j) + (\Gamma_k - 1)(u_i - u_k)
\end{equation}

Grouping terms by absolute delay:

\begin{equation} \label{eq:8_appendix_08}
\mathcal{R}_{\rm TEP} = u_i(\Gamma_k - \Gamma_i) + u_j(\Gamma_i - \Gamma_j) + u_k(\Gamma_j - \Gamma_k)
\end{equation}

This can also be written in the pairwise-delay form used in the main text by expanding $\Gamma = 1 + \delta\Gamma$ and using $\sum_{\rm loop}\Delta t^{\rm GR} = 0$:

\begin{equation} \label{eq:8_appendix_09}
\mathcal{R}_{\rm TEP} = \sum_{\rm loop} (1 + \delta\Gamma)\,\Delta t^{\rm GR} = \sum \Delta t^{\rm GR} + \sum \delta\Gamma\,\Delta t^{\rm GR} = \sum_{\rm loop} (\Gamma_m - 1)\,\Delta t_{mn}^{\rm GR}
\end{equation}

Explicitly for the triplet $(i, j, k)$:

\begin{equation} \label{eq:8_appendix_10}
\mathcal{R}_{\rm TEP} = (\Gamma_i-1)\Delta t_{ij} + (\Gamma_j-1)\Delta t_{jk} + (\Gamma_k-1)\Delta t_{ki}
\end{equation}

where $\Delta t_{ij} \equiv \Delta t_{ij}^{\rm GR}$. This equation demonstrates that the residual is generated purely by the *differences* in $\Gamma$ across the oriented image triplet. If the potential depth is constant ($\Gamma_i = \Gamma_j = \Gamma_k$), the residual vanishes. For the S1–S4–SX loop of SN Refsdal, where the other two delays are well-constrained by the Einstein-cross images, this weighted sum approximately equals the TEP-induced discrepancy in the SX–S1 delay. **This is not an observed delay-closure violation.** It is a model-space residual combination comparing GR-predicted delays to a TEP-scaled reference. Algebraic closure of pairwise delays remains exactly zero; the genuine TEP observable is the blind-prediction residual, not a synchronization holonomy.

### A.3 Prediction-Provenance Audit

The blind-prediction residual test relies on a delay-blind ensemble of pre-reappearance model variants. The selection was performed before any TEP analysis was computed, based on two criteria: (i) the original model was published before SX reappeared in December 2015, and (ii) the model variant is a distinct parameter configuration (all-image set vs gold-image set) from the same modelling family. The table below documents the provenance of each delay-blind variant, including whether the *value used in this analysis* is the originally published prediction or a later revision.

| Selected model | Original paper | Original $\Delta t_{\rm pred}$ | Original residual sign | Value used | Kelly+2023 revision? | Blind? | Reason selected |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Oguri-a* | Oguri (2015, MNRAS 449, L86) | 336 ± 21 d — Kelly+2023 Science S4, original pre-reappearance prediction | + | 353.8 d | Yes — post-reappearance computational revision by Kelly+2023 using the SX reappearance position (Rodney et al. 2023, ApJ 948, 93); original was 336^{+21}_{-21}, revised to 354^{+21}_{-17} | Original prediction: yes. Revised value: No — not blind | All-image set; parametric GLAFIC |
| Oguri-g* | Oguri (2015, MNRAS 449, L86) | 311 ± 24 d — Kelly+2023 Science S4, original pre-reappearance prediction | + | 330.7 d | Yes — same as Oguri-a; original was 311^{+24}_{-24}, revised to 331^{+22}_{-17} | Same as Oguri-a | Gold-image set; parametric GLAFIC |
| Sharon-a* | Sharon & Johnson (2015, ApJ 800, L26) | 237^{+37}_{-50} d — Sharon & Johnson 2015, Δt_{1.2-S1}; a/g variants from Treu 2015 use same underlying model | + | 298.0 d | Yes — post-reappearance computational revision by Kelly+2023; original ~237 d rescaled to 298^{+58}_{-17} | Original prediction: yes. Revised value: No — not blind | All-image set; parametric LTM |
| Sharon-g* | Sharon & Johnson (2015, ApJ 800, L26) | 237^{+37}_{-50} d — Same original paper; variant from Treu 2015 gold-image set | + | 331.0 d | Yes — same as Sharon-a; best-fit realization adopted as 331^{+30}_{-35} | Same as Sharon-a | Gold-image set; parametric LTM |
| Diego-a | Diego et al. (2016, MNRAS 456, 356) | 262 ± 55 d — Original prediction, no post-reappearance revision noted in Kelly+2023 S4 | + | 262.0 d | No — value taken directly from Kelly+2023 Supplementary Table S4 compilation, no post-reappearance revision noted | Yes — original pre-reappearance prediction | All-image set; free-form WSLAP+ |
| Grillo-g | Grillo et al. (2016, ApJ 822, 78) | 361^{+19}_{-27} d — Original prediction, no post-reappearance revision noted in Kelly+2023 S4 | + | 361.0 d | No — value taken directly from Kelly+2023 Supplementary Table S4 compilation, no post-reappearance revision noted | Yes — original pre-reappearance prediction | Gold-image set; parametric GLEE |
| Jauzac15.2* | Jauzac et al. (2016, MNRAS 457, 2029) | Jauzac15.1 = 449 ± 45 d (strictly blind, pre-reappearance); Jauzac15.2 = 361 ± 42 d (delay-blind, revised using SX position) | + | 361.0 d | Yes — Kelly+2023 S4 revised value. Original Jauzac15.1 (449 d) is strictly blind; revised Jauzac15.2 (361 d) is delay-blind (fixed before measured delay known, but used SX position) | Original Jauzac15.1: strictly blind. Revised Jauzac15.2: delay-blind | Selected model version; free-form LENSTOOL |
| *Post-blind supplementary only* |  |  |  |  |  |  |  |
| Grillo+2024 | Grillo et al. (2024, A&A 689, A78) | N/A (post-blind) | N/A | 362.0 d | No | Post-blind | Supplementary consistency check only |

**Blindness caveat.** The Kelly+2023 *Science* paper (published 2023, post-reappearance) compiled blind model predictions in Supplementary Table S4. Three provenance tiers are distinguished. *Strictly blind*: original pre-reappearance published predictions (Oguri-a 336 d, Oguri-g 311 d, Sharon-a 237 d, Sharon-g 237 d, Diego-a 262 d, Grillo-g 361 d, Jauzac15.1 449 d). *Delay-blind*: Kelly+2023 S4 revised values, all fixed before the measured delay (376.0 d) was known, but some (Oguri, Sharon, Jauzac) used the SX reappearance position (Rodney et al. 2023, ApJ 948, 93). *Post-blind*: Grillo+2024 (2024), published after the delay was measured. The primary sign test uses the delay-blind tier because all seven values were fixed before anyone knew 376.0 d, and all seven yield positive residuals. The strictly blind floor (6/7 positive, Jauzac15.1 residual −73 d) is reported for transparency. The amplitude-based 3.39σ result depends on revised values and is retained as a strong response-scale diagnostic, not as blind evidence.

The primary sign test uses the seven delay-blind model variants exactly as tabulated in the prediction-provenance table. The family-sign-flip test groups correlated variants by modelling family (GLAFIC, LTM, WSLAP+, GLEE, LENSTOOL) and is used as the operational dependence-aware headline. The post-blind Grillo+2024 update is reported as a supplementary consistency check and is excluded from the primary test.

### A.4 Immunity to Mass Sheet Degeneracy

The Mass Sheet Degeneracy (MSD) corresponds to a transformation of the convergence $\kappa \to \lambda \kappa + (1-\lambda)$, which rescales time delays by a factor $\lambda$:

\begin{equation} \label{eq:8_appendix_11}
\Delta t_{ij}' = \lambda \Delta t_{ij}
\end{equation}

Substituting this into the predicted GR discrepancy definition:

\begin{equation} \label{eq:8_appendix_12}
\mathcal{R}' = \Delta t_{ij}' + \Delta t_{jk}' + \Delta t_{ki}' = \lambda(\Delta t_{ij} + \Delta t_{jk} + \Delta t_{ki})
\end{equation}

Since $\Delta t_{ij} + \Delta t_{jk} + \Delta t_{ki} = 0$ in GR, then $\mathcal{R}' = \lambda \cdot 0 = 0$.

Thus, a uniform Mass Sheet Degeneracy cannot generate a non-zero algebraic loop sum by itself, because it rescales all delays coherently. This does not imply that the empirical blind-prediction residual is immune to all lens-model systematics. The narrower result is that the temporal-shear residual tested here is not algebraically equivalent to a trivial uniform mass-sheet rescaling.

## Data Availability & Reproducibility

This work follows open-science practices. All results are fully reproducible from raw data using the documented pipeline. All numerical results, figures, and statistics are generated by deterministic Python scripts processing real observational data.

### Repository & Code

*GitHub Repository:* github.com/matthewsmawfield/TEP-LENS

The repository contains a deterministic, version-controlled analysis pipeline covering *Steps 00–20, 30–46, and 50–55*, including diagnostics for blind residual statistics, model dependence, proxy robustness, convergence-tracer substitution, lensing-potential reconstruction, 3D geodesic transport, response-transfer auditing, cross-system kernel consistency auditing, and future-system forecasting.

#### Repository Structure

```
TEP-LENS/
├── data/                          # Raw light curves and lens models
│   ├── sn_lensing/                # SN Refsdal and H0pe data
│   └── snh0pe/                    # SN H0pe light curves
├── scripts/
│   ├── steps/                     # Sequential analysis pipeline
│   │   ├── step_01_fetch_snh0pe_data.py
│   │   ├── step_02_gr_closure.py
│   │   ├── step_03_tep_closure.py
│   │   ├── step_04_plot_closure.py
│   │   ├── step_11_model_dependence.py
│   │   ├── step_13_bayes_model_comparison.py
│   │   └── run_all_steps.py       # Master pipeline runner
│   └── utils/                     # Plotting and logging utilities
├── results/                       # Pipeline outputs and figures
├── site/
│   └── components/                # HTML manuscript source
└── requirements.txt                 # Python dependencies
```

### Data Provenance

| Data Source | Provider | Access Method | Download Size | Reference |
| --- | --- | --- | --- | --- |
| SN Refsdal Light Curves | Kelly et al. (2023) | Published data | ~1 MB | ApJ 948, 93 |
| SN H0pe Data | SN H0pe Collaboration | Public release | ~500 KB | Via GitHub |
| TDCOSMO-2025 Chains | TDCOSMO Collaboration | Public chains | ~50 MB (MCMC chains) | tdcosmo.github.io |
| H0LiCOW/TDCOSMO Legacy | H0LiCOW Collaboration | Public chains | ~20 MB | Via repository |

*Total Download Size:* ~75 MB for all primary data sources.
*Data Provenance Log:* Complete acquisition details maintained in `data/DATA_PROVENANCE.md`.

### Reproduction Instructions

#### Quick Start (Full Reproduction)

```
# 1. Clone repository
git clone https://github.com/matthewsmawfield/TEP-LENS.git
cd TEP-LENS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run complete pipeline: Steps 00–20, 30–46, and 50–55
python scripts/steps/run_all_steps.py

# 4. Build manuscript
cd site
npm install
npm run build
```

#### System Requirements

| Component | Minimum | Recommended | Tested On |
| --- | --- | --- | --- |
| CPU | 4 cores | 8+ cores | Apple M4 Pro (14-core) |
| RAM | 8 GB | 16 GB | 24 GB (M4 Pro) |
| Storage | 5 GB | 10 GB | NVMe SSD |
| Runtime | ~30-60 minutes |  | ~30 minutes (M4 Pro) |

#### Detailed Pipeline Steps

The analysis pipeline covers *Steps 00–20, 30–46, and 50–55*, including diagnostics for blind residual statistics, model dependence, proxy robustness, convergence-tracer substitution, lensing-potential reconstruction, 3D geodesic transport, response-transfer auditing, cross-system kernel consistency auditing, and future-system forecasting. Each step produces JSON outputs and logs for full traceability:

##### Data Acquisition & Closure Tests (Steps 00-04)

- *step_00_fetch_literature_and_cross_paper_data.py* — Fetch literature data and cross-paper reference values for TEP consistency checks

- *step_01_fetch_snh0pe_data.py* — Fetch SN H0pe light curve data from public release; parses multiply-imaged supernova magnitudes and time delays

- *step_02_gr_closure.py* — GR algebraic loop sum; computes the identically-zero loop baseline for GR with SN Refsdal and SN H0pe

- *step_03_tep_closure.py* — TEP predicted GR discrepancy; computes TEP-predicted GR discrepancies for each loop

- *step_04_plot_closure.py* — Generate predicted GR discrepancy plots (Figure 4 baseline vs discrepancy)

##### TDCOSMO & H0 Analysis (Steps 05-10)

- *step_05_tdcosmo_shear.py* — TDCOSMO temporal shear test; computes TEP-predicted delay shifts as a predicted-sensitivity consistency check

- *step_06_kappa_sensitivity.py* — κ₀ parameter sensitivity analysis; tests how predicted GR discrepancies depend on TEP coupling

- *step_07_observed_vs_predicted.py* — Observed vs blind-predicted delay comparison; blind sign evidence + post-hoc proxy sensitivity (not pre-observation forecast)

- *step_08_new_evidence.py* — Compile new evidence from independent lens systems

- *step_09_precision_roadmap.py* — Precision requirements analysis; quantifies significance as a function of model precision and sample size

- *step_10_h0_tension.py* — H0 internal-consistency check in lensing context; compares proxy-model shifts to local and CMB H0 without claiming a tension resolution

##### Model Dependence & Robustness (Steps 11-15)

- *step_11_model_dependence.py* — Model dependence analysis; tests sensitivity to lens model assumptions

- *step_12_microlensing_robustness.py* — Microlensing robustness tests; quantifies microlensing contamination effects

- *step_13_bayes_model_comparison.py* — Bayesian model comparison; computes Bayes factors for GR vs TEP lens models

- *step_14_external_chain_ingestion.py* — External chain ingestion; imports TDCOSMO/H0LiCOW MCMC chains for joint analysis

- *step_15_external_informed_inflation.py* — External-informed inflation analysis; tests TEP effects on time-delay cosmography precision under external uncertainty priors

##### External Data & Completeness (Steps 16-20)

- *step_16_independence_tier_significance.py* — Independence tier significance; statistical tests for multiple independent lens systems

- *step_17_directional_odds.py* — Directional odds analysis; tests TEP predictions for specific image parity configurations

- *step_18_external_dataset_registry.py* — External dataset registry; catalogs all public lensed SN light curves

- *step_19_tdcosmo2025_ingestion.py* — TDCOSMO-2025 chain ingestion; imports latest time-delay cosmography constraints

- *step_20_external_completeness_synthesis.py* — External completeness synthesis; combines all lensing constraints with TEP

##### Proxy Validation, Null Tests, Forward Prediction & Evidence Scaling (Steps 30–42)

- *step_30_cosmograil_temporal_shear.py* — CosmoGRAIL temporal-shear analysis; time-domain signal extraction from light curves

- *step_31_cosmograil_validation.py* — CosmoGRAIL validation; injection-recovery tests for delay estimation

- *step_32_kappa_proxy_validation.py* — Kappa proxy validation; Monte Carlo shear-degeneracy analysis quantifying mu-to-kappa systematic

- *step_32b_temporal_shear_figure.py* — Temporal-shear figure generation; produces diagnostic plots for shear analysis

- *step_33_einstein_cross_null.py* — Einstein cross null test; Spearman rank test on inner-cross delays vs magnification

- *step_34_agnostic_rank_test.py* — Agnostic rank test; proxy-agnostic statistical verification of rank-order predictions

- *step_35_single_contrast_dominance.py* — Single-contrast dominance quantification; effective degrees of freedom and signal-energy fractions

- *step_36_falsification_forward_prediction.py* — Falsification forward prediction; forward-projected thresholds for SN 2025wny

- *step_37_multi_system_evidence.py* — Multi-system evidence accumulation projection; Stouffer projection and coupling-inference scaling with N independent systems

- *step_38_cosmograil_cross_system.py* — CosmoGRAIL cross-system directional consistency diagnostic; caveated quasar check separate from the supernova residual tests

- *step_38_sn_encore_residuals.py* — SN Encore blind-prediction residual test; 8 independent lens models from Suyu+2025 vs Pierel+2026 observed delay

- *step_39_sn_h0pe_residuals.py* — SN H0pe blind-prediction residual test; 7 independent lens models from Pascale+2025 vs Pierel+2024 observed delays

- *step_40_cross_system_trio.py* — Cross-system trio evidence synthesis; Stouffer combination of directional evidence from Refsdal, Encore, and H0pe

- *step_41_null_channel_discriminator.py* — Null-channel discriminator; AIC comparison of the TEP proxy against generic bias models (null, constant offset, uniform-fractional H0/MSD-like) across cross-system contrasts, separating amplitude from sign evidence

- *step_42_precision_persistence.py* — Precision-persistence test; extrapolates the SN Refsdal SX residual to the high-precision lens-model limit to test the model-immaturity explanation

Each step produces JSON outputs with full metadata in `results/outputs/`, and execution logs are written to `logs/` with timestamps for complete traceability. Run all steps via: `python scripts/steps/run_all_steps.py`

#### Key Analysis Outputs

- `results/outputs/step_02_gr_closure.json` — GR algebraic loop sums

- `results/outputs/step_03_tep_closure.json` — TEP predicted GR discrepancies

- `results/outputs/step_07_observed_vs_predicted.json` — Observed vs predicted residuals with evidence-tier decomposition

- `results/outputs/step_13_bayes_model_comparison.json` — Bayesian comparison

- `results/outputs/step_32_kappa_proxy_validation.json` — Proxy systematic budget and mu-to-kappa Monte Carlo

- `results/outputs/step_34_agnostic_rank_test.json` — Proxy-agnostic Spearman rank test (delays vs kappa and mu)

- `results/outputs/step_35_single_contrast_dominance.json` — Signal-energy fractions and effective degrees of freedom

- `results/outputs/step_38_sn_encore_residuals.json` — SN Encore blind-prediction residuals and GR null tests

- `results/outputs/step_39_sn_h0pe_residuals.json` — SN H0pe blind-prediction residuals and GR null tests

- `results/outputs/step_40_cross_system_trio.json` — Cross-system trio Stouffer combination and sign-consistency check

### Software Versions

- *Python* 3.10+

- *NumPy* 1.24+

- *SciPy* 1.10+

- *Pandas* 2.0+

- *Matplotlib* 3.7+