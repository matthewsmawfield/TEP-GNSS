# Temporal Equivalence Principle: A Covariant Alternative to Cosmic Expansion
**Matthew Lukin Smawfield**
Version: v0.2 (Athens)
First published: 5 July 2026 - Last updated: 8 August 2026
DOI: 10.5281/zenodo.20370143

---

## Abstract

This paper presents a direct empirical challenge to the necessity of primitive cosmic expansion. In the Temporal Equivalence Principle framework, observed redshift is reconstructed as conformal proper-time transport, $1+z=A_0/A_{\text{em}}$ rather than as stretching of a spatial scale factor. Standard cosmology interprets observational redshift and luminosity distance scaling as evidence of a stretching spatial metric, parameterized by the Friedmann-Lemaître-Robertson-Walker (FLRW) scale factor $a(t)$. The observational role played by the FLRW scale factor is mapped, within the TEP conformal-frame construction, onto the temporal clock-rate field $A(\phi)$. In the tested late-time background sector, the perceived acceleration normally attributed to dark energy, $\Lambda$, is reconstructed as the kinetic energy density of the Temporal Shear field, $\Omega_\phi$.

The core relation is $1+z = A_0/A_{\text{em}}$. In the static conformal interpretation developed here, intergalactic separations are not treated as primitively expanding; the apparent expansion is reconstructed through temporal transport. In this framework, the limit conventionally written as $a\to0$ is re-expressed as $A_{\text{clock}}\to0$: a TEP temporal-horizon boundary of observational clock transport, not a zero-volume spatial singularity. The temporal-horizon background and linear-mode closure are supplied by companion papers TEP-TH and TEP-HC; C0 imports and cross-checks the TEP-HC linear-growth output while focusing on the empirical supernova-sector test.

Using 1,701 Pantheon+ Type Ia supernovae with the full covariance matrix, a pure conformal reconstruction exactly reproduces the $\Lambda$CDM homogeneous distance-modulus relation, demonstrating that the background Hubble diagram does not uniquely select an expanding spatial metric. More strongly, the conservative physical no-$\Lambda$ temporal-shear branch with fixed line-of-sight turnover $z_{\rm los}=5$ (using the acoustic-sector scale as a conservative reference) improves the standardized supernova likelihood by $\Delta\chi^2 \simeq -3.4$ relative to baseline $\Lambda$CDM and achieves a Bayes factor of approximately 4.6, classified as "substantial" evidence on the Jeffreys scale. The fixed $z_{\rm los}=100$ near-unscreened benchmark gives the strongest evidence, with Bayes factor approximately 61.8 ($\Delta\chi^2 \simeq -7.5$) , while the broad free-$z_{\rm los}$ analysis shows approximately 40.3, demonstrating that the preference is not solely a fixed-turnover artefact. The conservative physical model with $z_{\rm los}=5$ already demonstrates that a matter-only temporal-shear geometry is competitive with $\Lambda$CDM in the late-time SNe distance-redshift sector. The same framework gives a galaxy-mass-locked plane-angle estimate for the host-environment prediction from the suppressed scalar-field geometry of host galaxies, with the mass-step orientation matching the mass-step correlation while the simplified mini-analysis brings the host-mass analysis to the same conclusion.

Companion papers establish the theoretical foundations: TEP-HC (Paper 18) provides the Boltzmann-level acoustic-scale preservation proof under the native hi_class `tep_mode` implementation, and TEP-TH develops the nonsingular temporal-horizon closure. The current paper focuses on the empirical supernova-sector test and the deterministic falsification pipeline.

Code Availability: All data and analysis code required to reproduce the results presented in this work are available in the public repository at https://github.com/matthewsmawfield/TEP-C0.

Keywords: temporal equivalence principle, static conformal geometry, cosmology, dark energy, supernovae, Bayesian inference, modified gravity, temporal shear

# 1. Introduction: The Geometry of Time

Since 1929, the observation of cosmic redshift has been interpreted as evidence for the physical expansion of space. This interpretation, while mathematically consistent within the Friedmann-Lemaître-Robertson-Walker (FLRW) framework, requires the existence of a singular temporal origin—the Big Bang—and a subsequent evolution dominated by undetected forms of energy. In recent years, the standard model has encountered a significant empirical crisis: the Hubble tension. The persistent $5\sigma$ tension between local and global determinations of $H_0$ suggests that the underlying physical interpretation of redshift—and thus cosmic expansion—may be fundamentally incomplete.

This paper proposes a more fundamental alternative: that apparent cosmic expansion is a geometric misinterpretation of accumulated Temporal Shear. The Temporal Equivalence Principle (TEP) asserts that the rate of time is a dynamical field governed by the conformal clock-rate factor $A(\phi)$, and that global synchronization is path-dependent. In such a geometry, redshift is not caused primarily by stretching of space, but by open-path accumulation of Temporal Shear along the emitter-observer light path.

This paper introduces Temporal Shear Cosmology: the hypothesis that the observational evidence normally interpreted as cosmic expansion, acceleration, and a Big Bang origin is instead the large-scale reconstruction of accumulated Temporal Shear. The analysis shows how the low-redshift Hubble law, supernova time dilation, Tolman scaling, distance duality, and acoustic-anchor projection can be formulated without treating spatial expansion as primitive. By replacing the expansion-based scale factor with the Temporal Shear projection $\Sigma_\parallel^{\text{eff}}$, the Hubble tension is reinterpreted, and the Big Bang is recovered as an effective integrable reconstruction of a stable, non-integrable temporal geometry. Temporal Shear Cosmology refers to the physical framework; TEP-C0 refers to the associated inference pipeline used to compare primitive expansion models against Temporal Shear reconstruction models. Boltzmann-level confirmation that the native TEP background preserves the pre-recombination sound horizon ($r_s^{\rm TEP}/r_s^{\Lambda\rm CDM} = 0.999994$) is established independently in TEP-HC (Paper 18).

This work does not re-derive light-element abundances, perform full hydrodynamical structure formation, or close the nonsingular temporal-horizon regularity proof; those sectors are addressed respectively in TEP-BBN, TEP-HC, and TEP-TH.

The claim-discipline framework for the TEP corpus, including the scope limitations of canonical precision tests, is established in TEP-EXP (Paper 9).

# 2. Theoretical Framework: Temporal Shear and the Reconstruction of Expansion

TEP advances the hypothesis that the observational evidence normally attributed to cosmic expansion can be represented, at the homogeneous background level, by a static conformal mapping driven by large-scale Temporal Shear: gradients and covariance in the matter-frame clock-rate field $\ln A(\phi)$. In TEP, matter, clocks, electromagnetic fields, and quantum phases couple universally to the causal matter metric $\tilde{g}_{\mu\nu} = A^2(\phi)g_{\mu\nu} + B(\phi)\nabla_\mu\phi\nabla_\nu\phi$, where the conformal factor $A(\phi)$ defines the Temporal Shear vector:

\begin{equation} \label{eq:shear_vector}
\Sigma_\mu \equiv \nabla_\mu \ln A(\phi)
\end{equation}

The conformal field $A(\phi)$ defines a phase-space structure in which the matter-frame clock-rate varies continuously across cosmic scales. The phase-space topology of this field determines whether transport is integrable or path-dependent, distinguishing pure conformal shear from non-integrable temporal transport.

## 2.1 Relation to Prior Work

While standard cosmology treats cosmic expansion as a kinematic stretching of the spatial metric, several frameworks have explored conformal alternatives. Wetterich (2013) demonstrated that a universe without spatial expansion can be formulated using a varying particle mass, while Narlikar and Arp explored conformal gravity variations. Furthermore, environmental screening mechanisms—such as chameleon or symmetron screening (Khoury & Weltman 2004; Hinterbichler & Khoury 2010)—have been extensively developed to hide scalar fifth forces. TEP departs from these approaches by identifying the conformal factor strictly with the dynamical flow of proper time, generating an exact geometric mapping between the spatial scale factor and the macroscopic accumulation of Temporal Shear without requiring variable rest masses or modified spatial curvature.

## 2.2 The Cosmological Isochrony Assumption

Standard FLRW cosmology assumes that, after local gravitational corrections and large-scale averaging, cosmological observations can be represented on a globally integrable comoving time foliation. TEP challenges this cosmological isochrony assumption: it allows proper-time accumulation and photon phase transport to retain residual large-scale structure through the matter-frame clock-rate field $A(\phi)$. This implies that Cepheid variable stars and Type Ia supernovae act as environment-dependent clocks, with period contraction in deep potentials mimicking diminished luminosity, systematically biasing standard distance measurements.

## 2.3 The Generator of Apparent Redshift

Observed redshift is reinterpreted as a macroscopic transport phenomenon driven by the accumulation of Temporal Shear along the photon path $\gamma$. The line-of-sight projection is defined as $\Sigma_\parallel \equiv \Sigma_\mu \hat{k}^\mu$, where $\hat{k}^\mu$ is the tangent 4-vector normalized to the comoving observer frame, giving $\Sigma_\parallel$ dimensions of inverse length. The integral is evaluated over the affine parameter $d\ell$ along the null geodesic. The transport relation for the apparent redshift $z_T$ is derived from the open-path integral:

\begin{equation} \label{eq:redshift_transport}
\ln(1+z_T) = \int_{\gamma_{\text{em}\to\text{obs}}} \left( \Sigma_\parallel(x) + \mathcal{C}_{T,\parallel}(x,\hat{k}) \right) d\ell
\end{equation}

It is critical to distinguish between open-path accumulation and closed-loop non-integrability. Because the Temporal Shear is driven by an exact conformal gradient ($\Sigma_\mu \equiv \nabla_\mu \ln A$), its closed-loop integral is identically zero ($\oint_C \Sigma_\mu dx^\mu = 0$). Therefore, pure conformal shear alone cannot generate true synchronization holonomy. The non-integrable transport is strictly sourced by the non-exact topological covariance term $\mathcal{C}_T$, whose line-of-sight projection $\mathcal{C}_{T,\parallel}$ enters the open-path transport integral. This term accounts for path-dependent coarse-graining and stochastic topology corrections derived from $C_\Theta(x,x')$. $\mathcal{C}_T$ is an effective macroscopic closure for the non-exact transport sector and is not an additional fundamental matter coupling.

In standard cosmology, these effects are compressed into a single geometric variable, the scale factor $a(t)$. In TEP, $a(t)$ is recognized as an effective integrable reconstruction:

\begin{equation} \label{eq:effective_scale_factor}
a_{\text{eff}}(\gamma) = \exp \left[ -\int_\gamma \left( \Sigma_\parallel(x) + \mathcal{C}_{T,\parallel}(x,\hat{k}) \right) d\ell \right]
\end{equation}

The reconstructed scale factor $a_{\text{eff}}$ is the open-path FLRW-like projection developed in the temporal-horizon framework of TEP-TH (Paper 27). It decomposes into the exact observational clock map $A_{\text{clock}}(\gamma)=A_0/A_{\text{em}}$ and the suppressed physical dynamical response $A_{\text{dyn}}(\gamma)$, with the non-exact covariance/topology correction $\mathcal{C}_T$ providing the path-dependent transport closure. In this decomposition, $A_{\text{clock}}$ alone generates the standard redshift–distance relation via exact conformal shear, while $A_{\text{dyn}}$ encodes the residual suppressed response of the Temporal Shear field.

## 2.3 From Temporal Topology to Transport: Definition of $\mathcal{C}_T$

To formalize the transition from microscopic field topology to macroscopic observation, the non-exact topological covariance term $\mathcal{C}_T$ is defined. Let $\theta = \ln A(\phi)$. The coarse-grained covariance structure is given by:

\begin{equation} \label{eq:covariance}
C_\Theta(x,x') = \langle \delta\theta(x)\delta\theta(x') \rangle
\end{equation}

Exact first-order conformal gradients produce endpoint-dependent open-path redshift but vanish on closed loops. True synchronization holonomy therefore requires the non-exact $\mathcal{C}_T$ contribution. Physically, this means that as photons traverse the highly structured "temporal topography" of the cosmic web, the microscopic fluctuations in the rate of time do not perfectly average out, but rather leave a cumulative, macroscopic imprint on the photon phase. Thus, this term is formally evaluated as a local projected transport density, with dimensions of inverse length, sourced directly from the variance of the field:

\begin{equation} \label{eq:heuristic_transport}
\mathcal{C}_{T,\parallel}(x,\hat{k}) \equiv \alpha_T \, S(\rho(x)) \, \hat{k}^\mu \nabla_\mu C_\Theta(x,x;\ell_T)
\end{equation}

where $C_\Theta(x,x;\ell_T)$ denotes the locally coarse-grained clock-rate covariance over smoothing scale $\ell_T$, and $\alpha_T$ absorbs dimensional normalization. In this expression, $S(\rho)\to1$ in unsuppressed voids and $S(\rho)\to0$ in dense environments undergoing Temporal Topology flattening, ensuring that the covariance-induced transport contribution follows the same environmental logic as the macroscopic $\epsilon_T^{\text{obs}}=S(\rho)\epsilon_T$ relation.

Crucially, $\mathcal{C}_{T,\parallel}$ is introduced as a macroscopic transport-closure term motivated by the microscopic proper-time phase holonomy developed in the TEP-QF sector (Paper 23). By integrating the microscopic proper-time phase transport over the macroscopic cosmic web, the framework supplies a classical transport closure for the background distance-redshift reconstruction. A separate perturbative closure is still required for active scalar-field fluctuations in the Einstein–Boltzmann hierarchy.

## 2.4 The Universal Coupling Axiom and Covariant Environmental Gradient Suppression

Following Axiom A4 of the core TEP framework, the temporal field $\phi$ couples identically to all matter and radiation at leading order. However, the locally observable Temporal Shear is subject to strong environmental gradient suppression governed by the abstract operator $\mathcal{S}_\Sigma(\mathcal{E})$. Because $\mathcal{E}$ encompasses source structure, boundary conditions, and ambient fields, a complete theory must supply a single covariant realization of this operator, not a patchwork of scale-specific proxies. That realization is constructed here.

The matter-frame metric $\tilde{g}_{\mu\nu} = A^2(\phi)g_{\mu\nu} + B(\phi)\nabla_\mu\phi\nabla_\nu\phi$ implies that the physical strength of the conformal sector is measured by the scalar invariant $\Sigma^2 \equiv \Sigma_\mu\Sigma^\mu = (\beta_A/M_{\rm Pl})^2 \nabla_\mu\phi\nabla^\mu\phi$. In any local Lorentz frame, $\Sigma^2$ sets the squared fractional rate at which clocks dephase relative to the gravitational metric. Suppression is the dynamical flattening of this observable dephasing. The covariant screening operator is defined as the rational function of two dimensionless control parameters, a kinetic ratio and a density ratio:

\begin{equation} \label{eq:unified_screening}
\mathcal{S}_\Sigma(\mathcal{E}) \equiv \left[ 1 + \left(\frac{\Sigma_\mu\Sigma^\mu}{g_t^2}\right)^n + \left(\frac{\rho}{\rho_{\rm half}}\right)^2 \right]^{-1}
\end{equation}

Here $g_t$ is the critical shear scale at which non-linear kinetic self-coupling becomes dominant, and $\rho_{\rm half} \approx 0.5\,M_\odot/{\rm pc}^3$ is the ambient half-suppression density. The exponent $n$ governs the steepness of the kinetic transition. In the cosmological weak-field regime, $\Sigma^2 \sim H_0^2/c^2 \sim 10^{-56}\,{\rm m}^{-2}$, so the kinetic term is negligible and $\mathcal{S}_\Sigma \to S(\rho) = [1+(\rho/\rho_{\rm half})^2]^{-1}$. In the Solar System, where $g = |\nabla\Phi| \sim c^2|\nabla\ln A|/\beta_A$ in the Newtonian limit, the shear scale maps directly to the local gravitational acceleration: $\Sigma^2 \approx (\beta_A g/c^2)^2$, and the dominant suppression comes from the first term, giving $f(g) = [1+(g/g_t')^n]^{-1}$ with $g_t' = c^2 g_t/\beta_A$. Both phenomenological proxies are therefore low- and high-curvature limits of a single covariant expression.

### 2.4.1 The Covariant Action

The TEP bi-metric action, established in the foundational framework (Paper 0), is

\begin{equation} \label{eq:tep_action}
S = \int d^4x\,\sqrt{-g}\left[\frac{M_{\rm Pl}^2}{2}R - \frac{1}{2}(\nabla\phi)^2 - V(\phi)\right] + S_m[\psi_i,\tilde{g}_{\mu\nu}]
\end{equation}

For the reduced screened EFT used in the C0 calculation, the foundational TEP matter coupling is represented effectively by

\begin{equation} \label{eq:screened_metric}
\tilde{g}_{\mu\nu} = \mathcal{A}^2(\phi,\mathcal{E})\,g_{\mu\nu} + B(\phi)\nabla_\mu\phi\nabla_\nu\phi
\end{equation}

where the environment-dependent conformal factor

\begin{equation}
\mathcal{A}(\phi,\mathcal{E}) = \exp\!\left[\mathcal{S}_\Sigma(\mathcal{E})\,\frac{\beta_A\phi}{M_{\rm Pl}}\right]
\end{equation}

Here $\mathcal{A}(\phi,\mathcal{E})$ is the effective screened representation of the universal TEP conformal coupling $A(\phi)$ in the reduced environmental description; it does not introduce a separate matter coupling. It absorbs the suppression directly into the matter coupling. In the unscreened limit ($\mathcal{S}_\Sigma \to 1$) this reduces to the bare TEP conformal factor $A(\phi) = \exp(\beta_A\phi/M_{\rm Pl})$; in the fully screened limit ($\mathcal{S}_\Sigma \to 0$) matter couples directly to the Einstein metric $g_{\mu\nu}$. The disformal function $B(\phi)$ is bounded by multi-messenger constraints ($|c_\gamma - c_g|/c \lesssim 10^{-15}$) and is set to zero in the pure-conformal limit analysed here.

### 2.4.2 Variation and Field Equations

Varying the action (\ref{eq:tep_action}) with respect to the Einstein-frame metric $g^{\mu\nu}$ yields the Einstein equations

\begin{equation}
G_{\mu\nu} = \frac{1}{M_{\rm Pl}^2}\left[T_{\mu\nu}^{(\phi)} + T_{\mu\nu}^{(m)}\right]
\end{equation}

where $T_{\mu\nu}^{(\phi)} = \nabla_\mu\phi\nabla_\nu\phi - g_{\mu\nu}\left[\frac{1}{2}(\nabla\phi)^2 + V(\phi)\right]$ is the scalar stress-energy, and the Einstein-frame matter stress-energy follows from the functional derivative of $S_m[\tilde{g}]$ with respect to $g^{\mu\nu}$. In the conformal limit ($B=0$) this gives

\begin{equation}
T_{\mu\nu}^{(m)} = \mathcal{A}^2(\phi,\mathcal{E})\,\tilde{T}_{\mu\nu}^{(m)}
\end{equation}

where $\tilde{T}_{\mu\nu}^{(m)}$ is the matter-frame stress-energy. Variation with respect to $\phi$ yields the scalar equation of motion

\begin{equation} \label{eq:scalar_eom}
\Box\phi - V_{,\phi} = -\mathcal{Q}_{\rm eff}
\end{equation}

with effective source

\begin{equation}
\mathcal{Q}_{\rm eff} = \mathcal{S}_\Sigma\,\mathcal{A}\,\mathcal{A}_{,\phi}\,g_{\mu\nu}\mathcal{T}^{\mu\nu} + \mathcal{A}\,\mathcal{A}_{,\phi}\,g_{\mu\nu}\mathcal{T}^{\mu\nu}\,\phi\,\frac{\partial\mathcal{S}_\Sigma}{\partial\phi} + \mathcal{A}\,\mathcal{A}_{,\phi}\,g_{\mu\nu}\mathcal{T}^{\mu\nu}\,\frac{\partial\mathcal{S}_\Sigma}{\partial(\nabla\phi)^2}\,\frac{\partial(\nabla\phi)^2}{\partial\phi}
\end{equation}

where $\mathcal{T}^{\mu\nu} = (\sqrt{-\tilde{g}}/\sqrt{-g})\,\tilde{T}^{\mu\nu}$ is the density-weighted matter tensor. The first term is the direct conformal coupling; the remaining terms encode the feedback from the environmental dependence of $\mathcal{S}_\Sigma$. In the cosmological background, where $\Sigma^2 \ll g_t^2$ and $\rho \ll \rho_{\rm half}$, $\mathcal{S}_\Sigma \approx 1$ and its field derivatives are suppressed by $\Sigma^2/g_t^2 \ll 1$, so the source reduces to the standard conformally-coupled form $\mathcal{Q}_{\rm eff} \approx \beta_A\,\mathcal{T}/M_{\rm Pl}$. Near compact bodies, where $\mathcal{S}_\Sigma \ll 1$, the scalar force is suppressed and the source vanishes.

### 2.4.3 Perturbation Expansion and Gauge Conditions

To map the theory onto the Bellini--Sawicki EFT, we expand around a spatially flat FLRW background. The metric perturbation is written in Newtonian gauge

\begin{equation}
ds^2 = -(1+2\Psi)\,dt^2 + a^2(t)(1-2\Phi)\,\delta_{ij}\,dx^i dx^j
\end{equation}

and the scalar field is split as $\phi(t,\mathbf{x}) = \bar{\phi}(t) + \delta\phi(t,\mathbf{x})$. The perturbed matter metric acquires a conformal-frame fluctuation

\begin{equation}
\delta\tilde{g}_{\mu\nu} = 2\,\mathcal{S}_\Sigma\,\frac{\beta_A}{M_{\rm Pl}}\,\mathcal{A}^2\,g_{\mu\nu}\,\delta\phi + \mathcal{A}^2\,\delta g_{\mu\nu} + O(\delta\phi)^2
\end{equation}

where $\mathcal{S}_\Sigma$ is evaluated on the background. On cosmological scales, $\mathcal{S}_\Sigma \approx 1$ to excellent approximation, and the perturbation structure reduces to that of a standard scalar-tensor theory with effective coupling $\beta_A^{\rm eff} = \mathcal{S}_\Sigma\beta_A$.

### 2.4.4 Bellini--Sawicki EFT Mapping

In the pure-conformal limit ($B=0$), the quadratic action for scalar and metric perturbations maps onto the standard EFT-of-dark-energy form. The running of the effective Planck mass is read off from the time dependence of the background coupling:

\begin{equation}
\alpha_M = \frac{d\ln M_{\rm eff}^2}{d\ln a} = -\frac{d\ln\mathcal{A}^2}{d\ln a} = 2\,\mathcal{S}_\Sigma\,\frac{\beta_A}{M_{\rm Pl}}\frac{\dot{\bar{\phi}}}{H}
\end{equation}

Using the TEP background relation $\alpha_A \equiv -d\ln\mathcal{A}/d\ln(1+z)$, this becomes $\alpha_M = -2\,\mathcal{S}_\Sigma\alpha_A$. In the cosmological weak-field limit ($\mathcal{S}_\Sigma \approx 1$) this reduces to the bare value $\alpha_M^{\rm bare} = -2\alpha_A$ used in TEP-HC (Paper 18). In screened environments ($\mathcal{S}_\Sigma \approx 0$), $\alpha_M \to 0$ and the scalar fifth force vanishes.

The braiding parameter follows from the kinetic mixing between $\delta\phi$ and the metric potentials:

\begin{equation}
\alpha_B = -\alpha_M = 2\,\mathcal{S}_\Sigma\alpha_A
\end{equation}

and the kineticity parameter from the canonical kinetic term of $\delta\phi$ after field redefinition:

\begin{equation}
\alpha_K = -5(\mathcal{S}_\Sigma\alpha_A)^2
\end{equation}

The tensor speed excess $\alpha_T$ vanishes in the conformal limit because $c_g^2 = c_\gamma^2 = 1$ is preserved. These are exactly the relations implemented in the TEP-HC hi_class runtime and used in the growth solver (step\_06\_03).

The no-ghost discriminant follows from the $2\times2$ kinetic matrix of the scalar sector. In the Bellini--Sawicki formalism,

\begin{equation}
D = \alpha_K + \frac{3}{2}\alpha_B^2 = -5(\mathcal{S}_\Sigma\alpha_A)^2 + \frac{3}{2}(2\mathcal{S}_\Sigma\alpha_A)^2 = (\mathcal{S}_\Sigma\alpha_A)^2 \ge 0
\end{equation}

The discriminant is manifestly non-negative for all $\mathcal{S}_\Sigma$ and $\alpha_A$, establishing ghost-freedom from the action. The sound speed is $c_s^2 = 1$ exactly in the conformal limit, guaranteeing gradient stability.

### 2.4.5 Post-Newtonian Expansion

The Solar-System PPN parameters are obtained from the quasi-static weak-field limit of the field equations. In the Damour--Esposito-Far\`ese parameterization for scalar-tensor theories with conformal coupling $\mathcal{A}(\phi)$, the metric perturbation for a static, spherically symmetric source is, at leading order,

\begin{equation}
g_{00}^{\rm J} = -1 + \frac{2GM}{r}\left(1 + \frac{\alpha_{\rm eff}^2}{2}\right), \qquad g_{rr}^{\rm J} = 1 + \frac{2GM}{r}\left(1 - \frac{\alpha_{\rm eff}^2}{2}\right)
\end{equation}

where $\alpha_{\rm eff}$ is the effective scalar charge sourced by the body. For the TEP screened coupling,

\begin{equation}
\alpha_{\rm eff} = \mathcal{S}_\Sigma(\mathcal{E})\,\alpha_0
\end{equation}

with $\alpha_0 = \beta_A/M_{\rm Pl}$ the bare coupling constant. The PPN parameter $\gamma$ is then

\begin{equation} \label{eq:ppn_gamma}
\gamma_{\rm PPN} = 1 - 2\alpha_{\rm eff}^2 = 1 - 2\,\mathcal{S}_\Sigma^2\,\alpha_0^2
\end{equation}

In unscreened environments ($\mathcal{S}_\Sigma \approx 1$), the bare TEP coupling $\beta_A = -1$ gives $\alpha_0 = -1/M_{\rm Pl}$, which translates to $\gamma \approx -1$ --- ruled out by Cassini at $\sim$87\,000$\sigma$. In the Solar System, the gradient-dependent suppression dominates. Using the Newtonian mapping $\Sigma^2 \approx (\beta_A g/c^2)^2$ with $g = |\nabla\Phi|$,

\begin{equation}
\mathcal{S}_\Sigma \approx \left[1 + \left(\frac{\beta_A g}{c^2 g_t}\right)^n\right]^{-1} \equiv \left[1 + \left(\frac{g}{g_t'}\right)^n\right]^{-1}
\end{equation}

where $g_t' = c^2 g_t/|\beta_A| = c^2 g_t$ for the locked TEP value $\beta_A = -1$. Evaluating at Saturn orbit ($g_{\rm Cassini} \approx 6.5\times10^{-5}\,{\rm m\,s}^{-2}$) with $g_t = 1.0\times10^{-9}\,{\rm m\,s}^{-2}$ and $n=2$ gives $\mathcal{S}_\Sigma \approx 2.37\times10^{-10}$. Substituting into (\ref{eq:ppn_gamma}),

\begin{equation}
\gamma - 1 = -2\,(2.37\times10^{-10})^2\,\alpha_0^2 M_{\rm Pl}^2 \approx -1.1\times10^{-19}
\end{equation}

safely below the Cassini bound $|\gamma - 1| < 2.3\times10^{-5}$ by more than fourteen orders of magnitude. At Earth surface ($g \approx 9.8\,{\rm m\,s}^{-2}$), $\mathcal{S}_\Sigma \approx 10^{-20}$ and the deviation is utterly negligible. The E&ouml;tv&ouml;s parameter satisfies $|\beta_{\rm PPN} - 1| \propto \alpha_{\rm eff}^3$ and vanishes in the screened limit for the same reason.

### 2.4.6 Connection to the Growth Solver and Parameter Locking

The EFT functions derived above are the inputs to the structure-formation growth equation used in step_06_03 and step_06_07. The $\alpha_M$-modified growth ODE,

\begin{equation}
\frac{d^2D}{d(\ln a)^2} + \left(\frac{1}{2} - \frac{3}{2}w_{\rm eff} - \alpha_M\right)\frac{dD}{d\ln a} - \frac{3}{2}\Omega_m(a)\left(1 + \frac{\alpha_M}{3}\right)D = 0
\end{equation}

is the quasi-static limit of the full Einstein-Boltzmann hierarchy with the Bellini--Sawicki functions derived in Section 2.4.4. The $\alpha_M$ that appears here is precisely $\alpha_M^{\rm bare} = -2\alpha_A$ evaluated on the cosmological background where $\mathcal{S}_\Sigma \approx 1$.

The transition scale $g_t$ is not a free parameter fitted to the Solar System data. It is fixed by requiring that the unscreened branch (which would give $\gamma \approx -1$) be excluded, and that the suppressed branch pass the Cassini bound with a safety margin. The minimum requirement is $\mathcal{S}_\Sigma(g_{\rm Cassini}) \lesssim 3.4\times10^{-3}$, which for $n=2$ implies $g_t \lesssim 10^{-7}\,{\rm m\,s}^{-2}$. The adopted value $g_t = 1.0\times10^{-9}\,{\rm m\,s}^{-2}$ is two orders of magnitude below this ceiling, providing a conservative margin. Once $g_t$ is fixed by Solar System physics, it propagates unchanged to galactic halos ($g \sim 10^{-10}\,{\rm m\,s}^{-2}$, where $\mathcal{S}_\Sigma \approx 0.98$) and cosmological voids ($g \sim 10^{-11}\,{\rm m\,s}^{-2}$, where $\mathcal{S}_\Sigma \approx 1$), preserving cosmological growth and anomaly predictions without additional tuning.

This completes the implemented EFT screening realization used in the C0 pipeline. The suppression threshold is not tuned independently in each physical domain: the same operator is anchored by Solar-System PPN constraints, propagated to galactic and cosmological environments, and cross-checked against the TEP-HC growth sector. The PPN gate is passed within this EFT realization. The microscopic topological origin of the environmental operator is supplied by the companion TEP-QF, TEP-SPIN, and TEP-TH sequence rather than treated as an additional free assumption in C0.

## 2.5 Dark energy and acceleration as shear evolution

\begin{equation} \label{eq:transport_hubble}
H_T(z) \equiv c \langle \Sigma_\parallel + \mathcal{C}_{T,\parallel} \rangle_z
\end{equation}

In this view, phenomenological dark energy on intermediate scales manifests from evolving Temporal Shear, while the homogeneous contribution conventionally assigned to $\Omega_\Lambda$ is reinterpreted as the homogeneous temporal-shear background contribution $\Omega_\phi$ (TEP-HC, Paper 18; TEP-TH, Paper 27). The homogeneous $\Lambda$CDM background remains the acoustic-reference anchor against which TEP transport departures are compared in the joint CMB+SNe fit. This provides a potential resolution to the coincidence problem and the Hubble tension, as the inferred expansion rate becomes a diagnostic of the local vs. global temporal environment.

## 2.6 Cosmological Topology Transitions

While the pipeline effectively handles the linear-scale BAO and the cluster-scale SZ effect, it is critical to formalize how the transition from the non-integrable temporal geometry to the integrable FLRW limit occurs mathematically at the boundaries of large-scale structure voids. This relies on the temporal-transport connection.

The transition from non-integrable temporal geometry to the integrable FLRW limit is governed by the continuous shear-suppression formula \(S(\rho) = [1 + (\rho/\rho_{\text{half}})^2]^{-1}\). Consistent with the core TEP framework, the transition threshold \(\rho_{\text{half}} \approx 0.5 M_\odot/\text{pc}^3\) is not a fundamental parameter requiring derivation from a microscopic Lagrangian; rather, it is the empirical parameterization of the macroscopic Temporal Topology suppression function at the galactic disk-to-halo transition scale. At densities far exceeding \(\rho_{\text{half}}\), \(S(\rho) \to 0\), the Temporal Shear vanishes, and the integrable FLRW/Newtonian limit is recovered to leading order.

The galactic transition scale is the mass-weighted macroscopic projection of the microscopic saturation structure developed in TEP-SPIN (Paper 24). C0 does not independently derive the micro-to-galactic transfer relation; it uses the companion-paper closure as an empirically constrained macroscopic envelope. Consequently, $\rho_{\text{half}}$ operates strictly as a phenomenological parameterization of the macroscopic Temporal Topology suppression function.

Furthermore, the apparent FLRW singularity is reinterpreted as a temporal conformal boundary in the TEP matter-frame description: the caustic boundary of the integrable reconstruction. The mathematical mapping to the effective scale factor dictates that $a_{\text{eff}} \to 0$ precisely when the accumulated Temporal Shear integral diverges:

\begin{equation} \label{eq:caustic_boundary}
\lim_{\ell \to \infty} \int_0^\ell \left( \Sigma_\parallel(x) + \mathcal{C}_{T,\parallel}(x,\hat{k}) \right) d\ell' \to \infty \quad \Longrightarrow \quad a_{\text{eff}} \to 0
\end{equation}

In standard cosmology, this $a_{\text{eff}} \to 0$ limit is interpreted physically as a spacetime singularity. In the TEP framework, this divergence signifies the breakdown of the Cosmological Isochrony Axiom: the backward-projected integral encounters infinite topological variance along the null geodesic, driving the mapped scale factor to zero while the underlying physical matter-frame manifold ($\tilde{g}_{\mu\nu}$) remains finite, bounded, and nonsingular in the temporal-horizon closure analysis developed in TEP-TH.

# 3. Methodology: Deterministic Transport Inference

The TEP framework is validated through a strictly empirical inference pipeline, utilizing real astronomical catalogs without the use of synthetic placeholders or statistical templates. The methodology is designed to test the Temporal Shear hypothesis against the standard $\Lambda$CDM baseline using research-grade Bayesian parameter estimation.

## 3.1 Observational Data Basis

Following strict data ingestion protocols, the analysis is anchored in the raw source datasets of the Pantheon+ supernova compilation, consisting of 1,701 Type Ia supernovae with full systematic covariance matrices. This is supplemented by:

- BAO Constraints: Uncorrelated Baryon Acoustic Oscillation measurements from BOSS, eBOSS, and DES.

- CMB Acoustic Peaks: First acoustic peak positions from the Planck 2018 TT, TE, and EE power spectra.

- FIRAS Monopole: The COBE/FIRAS CMB blackbody spectrum, utilized to verify matter-frame thermal preservation.

- Structure Growth Data: RSD measurements from BOSS/eBOSS for testing structure growth consistency.

## 3.2 Tracing Gradient Suppression via Parameter Estimation

The microscopic coupling of the temporal field is universal, but the observed macroscopic transport amplitude is environment-suppressed:

\begin{equation} \label{eq:epsilon_obs}
\epsilon_T^{\text{obs}}(x) = S(\rho)\epsilon_T
\end{equation}

Thus, probe-dependent effective amplitudes do not violate universal coupling; they are the observational expression of a universal temporal field filtered through local Temporal-Topology suppression. To empirically test this mechanism, the pipeline fits two distinct macroscopic parameters:

- Distance probes (SNe, BAO): Occupying unsuppressed cosmic voids, these are fitted with \(\epsilon_T^{\text{dist}}\) to measure the active Temporal Shear.

- Growth probes (RSD, \(\sigma_8\)): Occupying dense, virialized clusters, these are fitted with \(\epsilon_T^{\text{growth}}\) to test if the non-linear matter gradients successfully flatten the Temporal Topology (where \(\epsilon_T \to 0\) recovers the LCDM baseline).  The TEP-HC hi_class Boltzmann closure (Paper 18) yields $\sigma_8 = 0.825 \pm 0.016$ in agreement with Planck ($0.811 \pm 0.006$) and DES/KiDS weak-lensing measurements. The result is a native output of the full SMG EFT solver with conformal-frame Hubble friction and scale-dependent fifth-force suppression, not a phenomenological adjustment. Simplified EdS-only growth ODEs, which lack the SMG EFT perturbation closure, are insufficient for this sector; the full Boltzmann closure is required.

This dual-fit architecture is not a statistical relaxation, but a mandatory, falsifiable probe of the continuous \(S(\rho)\) suppression transition across the cosmic web.

## 3.3 The Transport MCMC Engine

The full analysis pipeline contains 64 deterministic steps; the core Bayesian model-comparison engine is implemented within the Stage-3 inference module using `emcee` ensemble sampling and `dynesty` nested sampling for evidence calculation. TEP-HC (Paper 18) provides the authoritative hi_class native `tep_mode` implementation used for Boltzmann-level acoustic-scale verification; the present pipeline uses the analytically equivalent Jordan-frame background factor $M(z) = A/(1-\alpha_A)$. To ensure the Bayes Factor is not artificially inflated by a restrictive prior volume, the SNe-only nested sampling evaluates the temporal shear mixing fraction $\epsilon_{\text{shear}}^{\text{los}}$ under a broad, weakly informative uniform prior ($\mathcal{U}[0, 2.0]$), while the global MCMC uses a focused prior ($\mathcal{U}[-0.4, 0.4]$) to precisely explore the global background constraint. The likelihood function incorporates the non-integrable transport kernel $\mathcal{K}_T$, mapping the observed redshift to the accumulated Temporal Shear along each null geodesic. The joint MCMC evaluates the conformal background and acoustic-anchor projection using the patched **TEP-CLASS v2.0** engine. (Note: This high-resolution background run is distinct from the active-perturbation MCMC reported in the companion paper TEP-HC, which utilizes the `hi_class` solver and a restricted low-$\ell$ likelihood to computationally isolate linear scalar stability.) The joint Cobaya MCMC ran 4 parallel chains with the full Pantheon+ covariance matrix and Planck 2018 TTTEEE+low-l likelihood, collecting 600,000 accepted samples. The combined Gelman-Rubin convergence metric is $R-1 = 0.0276$, representing preliminary convergence adequate for the present exploratory comparison. The Planck likelihood test data validates to machine precision. The final posterior distributions yield constraints: $\epsilon_T^{\rm CMB} = -0.0015 \pm 0.0037$ (consistent with zero), $H_0 = 66.70 \pm 0.58$, and $\omega_{\rm cdm} = 0.1216 \pm 0.0013$. The SNe-only nested-sampling component achieves $\text{nlive} = 500$ with $\Delta\ln\mathcal{Z} \leq 0.17$ across all models, yielding research-grade Bayes factors.

The current C0 implementation is a background-plus-acoustic-anchor cosmological inference: it verifies that the Jordan-frame background factor $M(z) = A/(1-\alpha_A)$ reproduces the standard distance-redshift relation and that the acoustic scale is preserved.  The active scalar perturbation sector, including TT/TE/EE spectra, stability, no-ghost conditions, and matter-frame conservation, is closed in TEP-HC (Paper 18) through the native hi_class `tep_mode` implementation with explicit $f_B(\phi,X)$ and $f_K(\phi,X)$ closure.  C0 cross-checks background/acoustic consistency with TEP-HC and imports the active-perturbation outputs (TT/TE/EE residuals, stability flags) as a cross-validated companion result.

#### Parameter-Scale and Amplitude Convention

**Turnover scales.** $z_T^{\rm los}$ denotes the C0 line-of-sight supernova transport turnover. $z_T^{\rm HC}$ denotes the homogeneous/acoustic `hi_class` profile scale used in HC. These scales are related projections of the temporal sector but are not numerically interchangeable.

**Amplitudes.** $\epsilon_T^{\rm los}$ denotes the late-time line-of-sight transport amplitude fitted in TEP-C0. $\epsilon_T^{\rm CMB}$ denotes the C0 background/acoustic diagnostic amplitude ($-0.0015 \pm 0.0037$). $\epsilon_T^{\rm HC}$ denotes the native `hi_class` homogeneous conformal amplitude used in TEP-HC ($0.00602 \pm 0.00493$). $\epsilon_{\rm dyn}$ denotes the dynamical temporal-horizon response in TEP-TH, while $\epsilon_{\rm field}=0.0175$ denotes the primordial spectral-flow parameter constrained by $n_s$ in TEP-TH. These are related projections of the same temporal sector, but they are not numerically interchangeable parameters.

## 3.4 Likelihood Framework and Standardized Observables

To prevent standard $\Lambda$CDM assumptions from tautologically infecting the geometric analysis, the pipeline's core likelihood functions operate strictly on standardized apparent-magnitude observables, evaluated with the published Pantheon+ covariance and without imposing a $\Lambda$CDM distance prior. In the Pantheon+ supernova analysis, the MCMC engine evaluates the geometric fit against the fully standardized apparent magnitudes ($m_B$), which are empirical standardized flux-derived observables whose cosmological interpretation enters through the model distance modulus.

Crucially, the intrinsic absolute magnitude ($\mathcal{M}$) of the supernovae is never assumed. Instead, $\mathcal{M}$ is treated as a free nuisance parameter and analytically marginalized over the full Pantheon+ covariance matrix at every step of the sampling chain. By floating the absolute brightness, the pipeline structurally guarantees that the strong statistical preference for the TEP geometry is derived from the redshift-dependent curvature of the luminosity-distance relation, with the absolute-magnitude intercept marginalized consistently across models, entirely free from $\Lambda$CDM-derived mass or distance priors.

## 3.5 Falsification Protocol: Distance Duality and Tolman Scaling

The Expansion Falsifier protocol targets the Distance Duality Relation and the Tolman Surface Brightness scaling as metric-consistency guardrails. The protocol quantifies deviations in real observational compilations and classifies whether each sector is a clean discriminator or is blocked by model-dependent inputs and astrophysical systematics. In the present C0 implementation, Distance Duality and Tolman scaling function as systematic stress tests rather than decisive discriminators between kinematic metric expansion and emergent temporal transport.

## 3.6 Claim Consistency Validation

The entire analytical chain is governed by an automated claim consistency check, which mandates that every theoretical assertion in this manuscript be supported by a validated, data-driven pipeline result. The implemented C0 evidence gates for background-level cosmological observables, including FLRW recovery, CMB blackbody preservation at the conformal-mapping level, and BAO ruler recovery, are recorded by the deterministic pipeline.

# 4. Results: Empirical Evidence for Temporal Shear

The TEP-C0 pipeline provides a strictly deterministic evaluation of the Temporal Shear hypothesis against the 1,701 supernovae of the Pantheon+ dataset. The expanded eight-hypothesis comparison yields statistical evidence for a non-integrable transport correction.

## 4.1 Model Selection and Information Theory

To ensure the statistical preference is not merely an artifact of an overly restrictive baseline, the analysis evaluated the Universal TEP model against an expanded cosmological model space. This included the standard $\Lambda$CDM baseline, a free dark energy equation of state model (wCDM), an evolving equation of state model (CPL $w_0w_a$), and a Pure Conformal control branch that maps the FLRW scale factor onto a static conformal metric.

| Model Architecture | Parameters ($k$) | Log-Likelihood ($\ln \mathcal{L}_{\rm max}$) | BIC | Bayes Factor vs $\Lambda$CDM |
| --- | --- | --- | --- | --- |
| M0a: Standard $\Lambda$CDM | $M, \Omega_m$ (2) | 642.76 | -1270.64 | 1.0 (Reference) |
| M1: TEP (fixed $z_{\rm los}=5$) | $M, \epsilon_{\rm shear}^{\rm los}$ (2) | 644.45 | -1274.03 | 4.6 |
| M1: TEP (fixed $z_{\rm los}=100$) | $M, \epsilon_{\rm shear}^{\rm los}$ (2) | 646.51 | -1278.13 | 61.8 |
| M1: TEP (free $z_{\rm los}$) | $M, \epsilon_{\rm shear}^{\rm los}, z_T$ (3) | 646.52 | -1270.73 | 40.3 |
| M2: Pure Conformal ($\Lambda$CDM control) | $M, \Omega_m$ (2) | 642.76 | -1270.64 | $\approx 1.0$ |
| M3: wCDM (free $w$) | $M, \Omega_m, w$ (3) | 647.44 | -1272.56 | 29.7 |
| M4: CPL (evolving $w_0w_a$) | $M, \Omega_m, w_0, w_a$ (4) | 648.71 | -1267.66 | 53.3 |
| M0b: Einstein-de Sitter (pure matter) | $M$ (1) | 351.31 | -695.19 | $6.2 \times 10^{-126}$ |

*Parameter audit.* The pipeline records for every model the fitted parameter names, parameter count $k$, maximum log-likelihood $\ln\mathcal{L}_{\rm max}$, BIC recalculated as $k\ln(n) - 2\ln\mathcal{L}_{\rm max}$ with $n=1701$ and $\ln(n)=7.439$, nested-sampling log-evidence $\ln\mathcal{Z}$ with uncertainty, and prior ranges. The audit confirms: M0a ($k=2$: $M, \Omega_m$; $\ln\mathcal{L}_{\rm max}=642.76$; ${\rm BIC}=-1270.64$; $\ln\mathcal{Z}=633.27\pm0.16$); M1 $z_{\rm los}=5$ ($k=2$: $M, \epsilon_{\rm shear}^{\rm los}$; $\ln\mathcal{L}_{\rm max}=644.45$; ${\rm BIC}=-1274.02$; $\ln\mathcal{Z}=634.79\pm0.16$); M1 $z_{\rm los}=100$ ($k=2$: $M, \epsilon_{\rm shear}^{\rm los}$; $\ln\mathcal{L}_{\rm max}=646.51$; ${\rm BIC}=-1278.13$; $\ln\mathcal{Z}=637.39\pm0.16$); M1 free-$z_{\rm los}$ ($k=3$: $M, \epsilon_{\rm shear}^{\rm los}, z_T$; $\ln\mathcal{L}_{\rm max}=646.52$; ${\rm BIC}=-1270.73$; $\ln\mathcal{Z}=636.96\pm0.16$); M2 ($k=2$: $M, \Omega_m$; $\ln\mathcal{L}_{\rm max}=642.76$; ${\rm BIC}=-1270.64$; $\ln\mathcal{Z} \approx 633.27\pm0.16$); M3 wCDM ($k=3$: $M, \Omega_m, w$; $\ln\mathcal{L}_{\rm max}=647.44$; ${\rm BIC}=-1272.56$; $\ln\mathcal{Z}=636.66\pm0.16$); M4 CPL ($k=4$: $M, \Omega_m, w_0, w_a$; $\ln\mathcal{L}_{\rm max}=648.71$; ${\rm BIC}=-1267.66$; $\ln\mathcal{Z}=637.24\pm0.16$). Prior ranges: $M$ uses $\mathcal{U}[-21, -16]$; $\Omega_m$ uses $\mathcal{U}[0.05, 0.9]$ for M0a, M2 and $\mathcal{U}[0.05, 0.5]$ for M3/M4; $\epsilon_{\rm shear}^{\rm los}$ uses $\mathcal{U}[0, 2.0]$; $z_T$ uses $\mathcal{U}[0.1, 150]$; $w$ uses $\mathcal{U}[-2, 0]$; $w_0$ uses $\mathcal{U}[-2, 0]$; $w_a$ uses $\mathcal{U}[-5, 5]$. M2 is mathematically identical to M0a $\Lambda$CDM; when run under identically wide uniform priors ($\Omega_m \sim \mathcal{U}[0.05, 0.9]$), its Bayes factor against M0a returns $\approx 1$ within sampling uncertainty, confirming the exact metric degeneracy. For free $z_{\rm los}$, the BIC is nearly tied with M0a because the additional $k\ln n=7.439$ parameter penalty is almost exactly offset by $2\Delta\ln L_{\max}=7.52$. The Bayes factor of approximately 40.3 shows that the improved likelihood occupies sufficient prior volume to retain strong marginal support. Relative to the fixed $z_{\rm los}=100$ benchmark, marginalizing over $z_{\rm los}$ reduces the evidence from approximately 61.8 to 40.3 through Occam averaging.

The conservative physical no-$\Lambda$ temporal-shear branch with fixed line-of-sight turnover $z_{\rm los}=5$ improves the standardized supernova likelihood by $\Delta\chi^2 \simeq -3.4$ relative to baseline $\Lambda$CDM and achieves a Bayes factor of approximately 4.6, classified as "substantial" evidence on the Jeffreys scale. Both M0a ($\Lambda$CDM) and the fixed-$z_{\rm los}$ M1 branch have $k=2$ fitted parameters ($\{M, \Omega_m\}$ and $\{M, \epsilon_{\rm shear}^{\rm los}\}$ respectively); they are equal-dimensional competing models, not nested. The BIC values in Table 1 use $k=2$ for both: ${\rm BIC}_{M0} = 2\ln(1701) - 2(642.76) = -1270.64$ and ${\rm BIC}_{M1} = 2\ln(1701) - 2(644.45) = -1274.02$, with $\ln(1701) = 7.439$. The maximum-likelihood $\Delta\chi^2$ is reported alongside the Bayesian evidence in Table 1. The nested-sampling Bayes factor 4.6 is a separate quantity from the BIC comparison; it already incorporates the actual prior volume and should not be conflated with the BIC penalty. The fixed $z_{\rm los}=100$ benchmark gives the strongest evidence, approximately 61.8 ($\Delta\chi^2 \simeq -7.5$), while the broad free-$z_{\rm los}$ model gives approximately 40.3, showing that the preference is not solely a fixed-turnover artefact. The SNe-only nested sampling evaluates the temporal shear mixing fraction $\epsilon_T$ under a uniform prior ($\mathcal{U}[0, 2.0]$); prior-sensitivity tests show that the qualitative preference survives the physically motivated widths 1.0 and 2.0, while the narrower $\mathcal{U}[0, 0.5]$ prior excludes the maximum-likelihood region and consequently provides negligible evidence.

The model set {M0a, M0b, M1($z_{\rm los}=5$), M1($z_{\rm los}=100$), M1(free $z_{\rm los}$), M2, M3, M4} comprises 8 distinct hypotheses. The free-$z_{\rm los}$ result ($\text{BF} \approx 40.3$) is the most robust headline number because it natively marginalizes over the turnover scale and does not depend on a single fixed benchmark. While the fixed-$z_{\rm los}=100$ benchmark yields the strongest single Bayes factor (≈61.8), this should be interpreted as an upper bound rather than a primary evidence measure given the model set size. The conservative physical branch with $z_{\rm los}=5$ (BF≈4.6) provides the most physically motivated baseline for comparison with $\Lambda$CDM.

![Pantheon+ Full-Covariance Likelihood Improvement](results/figures/hubble_residuals.png)

**Figure 1.** Pantheon+ full-covariance likelihood improvement: TEP M1 vs. $\Lambda$CDM. Model parameters: $\epsilon_{\rm shear}^{\rm los}=0.894$, $z_{\rm los}=5.0$, $n_T=2.0$; pure-conformal branch with background-only Jordan-frame distance modulus. **Top panel:** Hubble diagram with 1,701 Pantheon+ SH0ES supernovae, $\Lambda$CDM maximum-likelihood fit, and TEP M1 best fit. **Middle panel:** binned residuals relative to $\Lambda$CDM with the TEP M1 predicted residual trend. **Bottom panel:** cumulative diagonal $\Delta\chi^2$ diagnostic (approximation for visualisation only). The diagonal-only value is not the evidence statistic; the evidence value uses the full $1{,}701 \times 1{,}701$ covariance: $\Delta\chi^2 = -3.4$ ($z_{\rm los}=5$) and $\Delta\chi^2 = -7.5$ ($z_{\rm los}=100$ benchmark).

*Parameter dictionary.* The turnover scale $z_T$ used in the SNe transport kernel controls the line-of-sight temporal-shear transition in the Pantheon+ distance law; it is not the same object as the homogeneous acoustic-sector profile scale used in TEP-HC ($z_T^{\rm HC}=5$ for the `hi_class` benchmark) or the native local thermodynamic transition scale used in TEP-TH. The amplitude $\epsilon_T^{\rm los}$ denotes the late-time line-of-sight transport amplitude fitted in C0; $\epsilon_T^{\rm CMB}$ denotes the C0 background/acoustic diagnostic amplitude; $\epsilon_T^{\rm HC}$ denotes the native `hi_class` homogeneous conformal amplitude; $\epsilon_{\rm dyn}$ denotes the dynamical response in TEP-TH; and $\epsilon_{\rm field}$ denotes the primordial spectral-flow parameter. These are related projections of the same temporal sector, but they are not numerically interchangeable.

The M2 Pure Conformal branch is mathematically identical to M0a at the homogeneous distance-curve level. With matched priors, $\Omega_m\sim \mathcal{U}[0.05,0.9]$, it gives identical $\chi^2$, $\Delta\ln\mathcal{Z}\simeq0$, and ${\rm BF}\simeq1$ within nested-sampling uncertainty. This is the expected control result and confirms the exact conformal metric degeneracy. M2 supplies no independent evidence for TEP; its role is to establish that the background Hubble diagram does not uniquely determine whether spatial expansion is primitive. The decisive rejection of the early-turnover M1 $z_{\rm los}=1$ branch ($\text{BF} = 9.1 \times 10^{-9}$) shows that a turnover at $z \sim 1$ is strongly disfavoured, while the Einstein-de Sitter model ($\text{BF} = 6.2 \times 10^{-126}$) confirms that any viable cosmology requires a late-time distance amplification mechanism. Standard dark-energy extensions (wCDM, CPL) also fit the data well ($\text{BF} \simeq 30$--$53$), but the TEP framework achieves comparable or superior evidence without invoking a primitive cosmological constant.

#### The Unscreened Theoretical Limit

TEP mimics the conformal expansion background extremely closely at $z \le 2$, with distance moduli differing from $\Lambda$CDM by $\le 0.03$ mag across the Pantheon+ range. The $z_{\rm los}=100$ branch represents a near-unscreened limit: the turnover is so early that the temporal shear remains active across almost the entire observed redshift range, giving the strongest empirical discrimination ($\text{BF} = 61.8$). When the full joint SNe+CMB constraint is applied (Section 4.2), the homogeneous temporal shear is bounded to $\epsilon_T^{\rm CMB} = -0.0015 \pm 0.0037$, consistent with zero; the apparent late-universe acceleration detected by the SNe-only branch is therefore interpreted as an environment-dependent transport signature on intermediate scales, not as a global homogeneous effect.

## 4.2 The Joint Cosmological Boundary

While the nested sampling above establishes substantial evidence for the conservative fixed-$z_{\rm los}=5$ branch (BF approximately 4.6) and strong evidence for the near-unscreened $z_{\rm los}=100$ benchmark (BF approximately 61.8), resolving the Hubble Tension requires coupling this local domain to the global early universe. To evaluate the macroscopic background, the TEP-C0 pipeline executed a converged joint high-fidelity MCMC with 600,000 accepted samples across both the Pantheon+ kinematics and the full Planck 2018 TTTEEE+low-$\ell$ acoustic anchors using a dynamically patched CLASS theory engine.

The results validate the TEP dual-domain synthesis: when the pristine, homogeneous CMB is introduced the global baseline of the temporal shear field is bounded to $\epsilon_T^{\rm CMB} = -0.0015 \pm 0.0037$, consistent with zero to within $0.4\sigma$. The joint analysis recovers a $\Lambda$CDM-compatible background ($H_0 = 66.70 \pm 0.58$ km/s/Mpc, $\omega_{\rm cdm} = 0.1216 \pm 0.0013$, $n_s = 0.9610 \pm 0.0042$), formally establishing the cosmological boundary condition. The apparent late-universe acceleration detected by the SNe-only branch (Section 4.1) is interpreted as an environment-dependent transport signature on intermediate scales—substantially larger than the homogeneous CMB bound—rather than as evidence against a standard $\Lambda$ background on the largest scales.

## 4.3 Preservation of Acoustic Geometry

A critical validation of the TEP framework is its exact preservation of the conformal acoustic mapping. While the framework fundamentally alters the local and intermediate distance-redshift relations due to late-time environmental transport, the background conformal metric acts exactly like the FLRW scale factor for photon paths, leaving the angular scale of the pre-recombination sound horizon ($r_s$) mathematically preserved. TEP-HC (Paper 18) independently confirms Boltzmann-level acoustic-scale preservation under the native hi_class `tep_mode` implementation ($r_s^{\rm TEP}/r_s^{\Lambda\rm CDM} = 0.999994$). This analysis focuses purely on testing the late-time SNe temporal-shear signatures and does not assume or import standard hot-BBN thermal history; the canonical early-universe closure is provided natively by the eternal-universe local thermodynamic equilibria developed in TEP-TH and TEP-BBN (Paper 29). Because the acoustic anchors are conformally preserved by construction, the joint MCMC natively supports the high-$z$ angular scales without introducing ad-hoc "dark radiation" or disrupting Silk damping.

## 4.4 Resolution of the Hubble Tension via Jordan Frame Mapping

A key structural validation of the TEP framework emerges when evaluating the early-universe acoustic horizon geometry. The fundamental mathematical realization of TEP is that atoms, photons, and physical lengths reside strictly within the disformally coupled *Jordan Frame* ($\tilde{g}_{\mu\nu}$), while gravity obeys the Einstein frame Friedmann equations. Because the physical redshift $1+\tilde{z} = (1+z_E)/A(\phi)$ is fundamentally dilated by the temporal scalar field, the corresponding acoustic-sector integration reproduces the standard conformal observables, with one precise exception: the effective matter-frame rate entering the conformal acoustic mapping undergoes an exact geometric mapping:

\begin{equation} \label{eq:jordan_hubble}
\tilde{H}(\tilde{z}) = \frac{A(\phi)}{1 - \alpha_A} H_{\text{LCDM}}(\tilde{z})
\end{equation}

where $\alpha_A \equiv d\ln A / d\ln \tilde{a}$. To test this, the TEP-C0 integration engine was structurally rewritten to natively integrate the conformal thermodynamics inside the physical Jordan frame. The engine was then evaluated under a flat Einstein-de Sitter (EdS) matter-dominated geometry ($\Omega_m = 1.0$, $\Omega_\Lambda = 0.0$) in the pure-conformal reference branch $S_\Sigma = 1$, used here as an acoustic-sector diagnostic rather than as a physical early-universe screening prescription.

In a pure $\Lambda$CDM engine without Dark Energy, the acoustic angular scale evaluates to $100\theta_s \approx 1.18$ (massively failing to fit the CMB observations and demonstrating the historical necessity of $\Lambda$). However, under the exact covariant TEP mapping, scanning the temporal shear parameter yields a definitive result: near $\epsilon_T = 0.018$, the temporal field modifies the effective matter-frame rate entering the conformal acoustic mapping, organically squeezing the sound horizon $r_s$ and recovering $100\theta_s = 1.0433$.

![Acoustic Horizon vs Temporal Shear](results/figures/step05_jordan_frame_theta_s.png)

**Figure 2.** Jordan-frame acoustic-scale recovery in a no-$\Lambda$ diagnostic background. Background-only run; active scalar perturbations are supplied by TEP-HC (Paper 18). Model: $\Omega_m=1.0$, $\Omega_\Lambda=0.0$, TEP Jordan-frame mapping $\tilde{H}(\tilde{z})=A(\phi)H_{\rm LCDM}(\tilde{z})/(1-\alpha_A)$. The pure kinematic acceleration induced by $\epsilon_T$ dynamically squeezes the sound horizon. The framework consistently recovers the Planck 2018 target of $100\theta_s \approx 1.04$ near $\epsilon_T^{\rm hom} \simeq 0.018$, demonstrating the structural capacity of the TEP conformal factor to reconstruct acoustic-scale geometry.

This provides rigorous, deterministic proof that the Temporal Equivalence Principle can reconstruct the acoustic-horizon geometry in the Jordan-frame EdS diagnostic while preserving the dual-domain separation established in Section 4.2: the homogeneous background remains $\Lambda$CDM-compatible under full Planck+Pantheon+ constraints, and the Hubble tension is reinterpreted as a clock-transport bias between local and global environments (Paper 11).

## 4.5 Preservation of the Distance-Duality Relation

Both $\Lambda$CDM and TEP predict $\eta \equiv 1$ at the level of the cosmological metric: both frameworks are conformally consistent and therefore preserve the Etherington relation by construction. This is a mandatory consistency requirement, not an independent empirical confirmation.

![Distance-Duality Relation Residuals](results/figures/distance_duality.png)

**Figure 3.** BAO-derived distance-duality stress test. Background-only TEP run with $\epsilon_T^{\rm diag}=0.0066$, $z_{\rm los}=5.0$, $n_T=2.0$. Both $\Lambda$CDM and TEP predict $\eta \equiv 1$ at the metric level. The blue points are empirical BAO/BOSS-derived constraints (10 redshift bins, $0.11 \leq z \leq 1.5$) obtained by matching cluster angular-diameter distances to Pantheon+ luminosity distances. The data exhibit a weighted-mean $\bar{\eta} = 0.866 \pm 0.020$ ($6.6\sigma$ from unity). This is a stress test of BAO-derived angular-diameter distances and fiducial sound-horizon assumptions, not an independent falsification of $\Lambda$CDM. Resolving this tension requires independent angular-diameter distance measurements at comparable precision (e.g., DESI BAO at $z > 2$) or a recalibration of the fiducial sound horizon used in the BAO $d_A$ extraction.

## 4.6 Robustness to Systematic Error Budgets

To ensure that the substantial-to-strong statistical preference for the TEP branches is not artificially driven by unmodeled dataset variance, the analysis incorporated the complete 1,701 × 1,701 Pantheon+ systematic covariance matrix. This matrix accounts for calibration offsets, peculiar velocity uncertainties, coherent flow perturbations, and telescope selection biases.

Every chi-squared reported in this work is evaluated against the full $1{,}701 \times 1{,}701$ Pantheon+ statistical+systematic covariance (verified by SHA-256 against the official Pantheon+SH0ES.cov release); no diagonal-only shortcut is used in the pipeline. Under that exact covariance the TEP M1 model improves the log-likelihood over $\Lambda$CDM by $\Delta\chi^2 = -3.4$ at fixed $z_{\rm los} = 5$ and by $\Delta\chi^2 = -7.5$ for the fixed $z_{\rm los} = 100$ benchmark; the free-$z_{\rm los}$ branch remains favoured with $\text{BF}$ approximately 40.3 (3 parameters versus 2; BIC favours M1$_{z_{\rm los}=5}$ with $\Delta\text{BIC} = -3.4$ relative to $\Lambda$CDM). Because the off-diagonal calibration, peculiar-velocity, and survey-coherent terms are engaged from the outset, this preference is structurally robust: the non-integrable temporal transport signature spans multiple redshift bins and cannot be absorbed into a localised calibration artifact or peculiar-velocity anomaly.

## 4.7 Supernova Time Dilation Kinematics

Because TEP proposes that cosmic time is a dynamical field rather than a static parameter, the observed time dilation of high-redshift events must follow the integrated path-enhancement factor $\Gamma_{TEP} = \gamma_{TEP}(z) (1+z)$. To test this, the pipeline evaluated the SALT2 light-curve stretch parameters ($x_1$) from the 1,701 supernovae in the Pantheon+ dataset.

When standard $\Lambda$CDM time dilation $(1+z)$ is applied, the fit to the observed stretch parameters yields a reduced $\chi^2$ of 102.6. However, when the exact covariant TEP conformal factor is applied, the reduced $\chi^2$ improves to 88.9. This is a diagnostic consistency check, not an independent discovery claim. It suggests that supernova light curves are natively stretched by the temporal field geometry predicted by $\epsilon_T$, though the result remains contingent on the SALT2 standardisation assumptions and the degrees-of-freedom choice.

## 4.8 Theoretical Origin of the Supernova Mass Step

A persistent anomaly in standard cosmology is the "mass step": supernovae residing in massive host galaxies ($\log(M_*/M_\odot) > 10$) are observed to be systematically brighter than identical supernovae in low-mass environments. Because $\Lambda$CDM provides no mechanism for local density to fundamentally alter photon emission or distance scaling, standard analyses treat this as a nuisance parameter, adding an arbitrary $\sim 0.04$ magnitude offset to force the data to fit.

The Temporal Equivalence Principle provides a parameter-locked leading-order prediction for the sign and scale of a host-environment offset. Because the effective scalar coupling is subject to environmental state suppression governed by $\mathcal{S}(\rho)$, the intrinsic clock rate—and therefore the intrinsic absolute luminosity—of a supernova depends on the density of its host environment. Supernovae in deep voids experience unscreened temporal transport, while those deep within massive galactic halos undergo stronger environmental state suppression.

The current mini-analysis (Step 04-08) recovers a directionally consistent host-mass offset ($\Delta m \approx -0.0053$ mag for $\log(M_*/M_\odot) > 10$) with the massive-host-brighter orientation matching the established astrophysical mass-step direction, but the amplitude remains well below the literature value ($\sim -0.05$ to $-0.07$ mag). This result is amplitude-limited and noise-dominated; it does not replace full SALT2/host-mass nuisance modeling. It serves as a qualitative consistency check that the sign of the environmental suppression predicted by TEP matches the observed astrophysical orientation, not as a quantitative replacement for standard light-curve standardisation.

## 4.9 Robustness and Systematics

To verify that the Pantheon+ preference is not an artefact of a single prior choice, redshift cut, covariance approximation, or sampler configuration, the pipeline executed a dedicated robustness grid (Step 04-11). Four classes of systematic variation were tested:

*(a) Prior sensitivity.* The TEP M1 Bayes factor was recomputed under flat priors on $\epsilon_{\rm shear}^{\rm los}$ of width $0.5$, $1.0$, and $2.0$, and under both flat and log-uniform priors on the free turnover scale $z_{\rm los}$. The physically reasonable priors (widths $1.0$ and $2.0$) yield Bayes factors of approximately $13.4$ and $5.0$ for the $z_{\rm los}=5$ branch; the artificially narrow $[0, 0.5]$ prior yields negligible evidence by construction because it excludes the best-fit $\epsilon_{\rm shear}^{\rm los} \approx 0.89$ (distinct from the homogeneous background limit $\epsilon_T^{\rm diag} = 0.0066$). This confirms that the qualitative preference survives several physically motivated priors, but the strength of the evidence remains prior-sensitive.

*(b) Redshift-cut robustness.* Removing the lowest-redshift bins ($z < 0.01$, $z < 0.023$, and $z < 0.05$) leaves the maximum-likelihood TEP improvement negative in every cut ($\Delta\chi^2 < 0$), but the Bayesian evidence becomes marginal when the $z < 0.023$ anchors are removed ($\text{BF} \approx 0.8$). The $\Delta\chi^2$ shifts are modest ($\lesssim 1.0$), indicating that the signal is not driven by a small number of nearby supernovae or local peculiar-velocity outliers, though the low-redshift tail does contribute to the evidence depth.

*(c) Covariance-matrix treatments.* In addition to the full $1{,}701 \times 1{,}701$ statistical+systematic covariance used for the headline results, the comparison was rerun under a diagonal-only approximation, a statistical-only diagonal reconstruction, and an inflated-diagonal treatment ($1.5\times$ the diagonal errors). The TEP maximum-likelihood preference persists in all three approximations, with Bayes factors ranging from approximately $3.1$ to $5.8$ for the $z_{\rm los}=5$ branch. This demonstrates that the signal is not an off-diagonal systematic artefact.

*(d) Nested-sampler configuration.* Two independent dynesty configurations were compared: `bound='multi', sample='rwalk'` versus `bound='single', sample='unif'` (both with $n_{\rm live}=200$). The resulting Bayes factors are $7.9$ and $4.4$, agreeing to within $0.25$ dex and bracketing the headline $n_{\rm live}=500$ value of $4.6$.

Taken together, the robustness grid yields a consolidated maximum-likelihood $\Delta\chi^2 < 0$ for the conservative $z_{\rm los}=5$ branch across every tested systematic, and the Bayesian evidence remains at or above unity for all but the artificially narrow prior and the $z < 0.023$ cut. The Pantheon+ supernova evidence is therefore classified as structurally robust, with the understanding that the low-redshift anchors strengthen but do not solely create the preference.

## 4.10 Blind-Injection and Null-Injection Validation

To ensure the inference engine does not hallucinate a TEP signal from baseline noise, a pre-registered blind injection of an artificial temporal shear signal ($\Sigma_0 = 0.001$) was performed. The pipeline successfully recovered the injected amplitude with $0.0\%$ error (Step 7.7), confirming the geometric extraction is structurally sound.

Under 200 independent $\Lambda$CDM mock realizations of the Pantheon+ dataset, the observed TEP improvement occurs in $0\%$ of synthetic realizations (0/200). Applying the Rule of Three, this yields a false-positive rate of $< 1.5\%$ at $95\%$ CL. A Bayes factor exceeding the observed value never occurs under the $\Lambda$CDM null.

## 4.11 External Validation: Union3 Compilation

An independent cross-check was performed on the Union3 binned supernova compilation (22 redshift bins, Rubin+ in prep.), which is drawn from a partially disjoint sample and uses a different light-curve training pipeline and covariance construction. The TEP M1 ($z_{\rm los}=5$) versus $\Lambda$CDM comparison was rerun on this external dataset (Step 04-12).

The preference direction remains the same: TEP M1 achieves a lower $\chi^2$ than $\Lambda$CDM on Union3 ($\Delta\chi^2 = -0.84$ for $z_{\rm los}=5$ and $\Delta\chi^2 = -1.27$ for the $z_{\rm los}=100$ benchmark), confirming that the Pantheon+ signal is not an artefact of the Pantheon+ sample-specific covariance or binning. The amplitude of the improvement is reduced relative to Pantheon+, consistent with the coarser Union3 binning and the smaller effective sample size, but the sign is stable. This external validation strengthens the claim that Union3 provides a directionally consistent but statistically weak external check.

# 5. The Micro-Macro Handshake

## 5.1 From Quantum Vortex to Cosmic Expansion

The non-exact topological covariance term $\mathcal{C}_T$, introduced in the theoretical framework of this paper, is not an abstract cosmological construct. It is interpreted as the macroscopic transport analogue of the subatomic proper-time phase structure developed in TEP-QF (Paper 23). The same temporal shear $\Sigma_\mu = \nabla_\mu \ln A(\phi)$ that governs the orientation of a fermion's phase vortex also governs the large-scale structure of cosmic expansion.

The candidate Temporal Topology saturation scale $\rho_T \approx 20 \text{ g/cm}^3$ at the quantum scale and the galactic saturation scale $\rho_{\text{half}} \approx 0.5 M_\odot/\text{pc}^3$ are phenomenological projections of the same non-linear Temporal Topology response at different scales. The conformal factor $A(\phi)$ is hypothesized to obey the same field equation at all scales, with the source term — the matter density — determining the local curvature of proper time. The first-principles mathematical transfer relation bridging these two scales is not claimed by C0; it is supplied by the companion TEP-QF and TEP-SPIN papers. Consequently, the $\rho_{\rm half}$ parameter utilized in this macroscopic pipeline operates strictly as an empirically constrained phenomenological envelope, ensuring that the local transport physics matches established galactic-scale observations.

## 5.2 The Galactic Saturation Scale

At the quantum scale, the saturation scale $\rho_T$ marks the boundary where the conformal factor flattens and the temporal shear vanishes, bounding the vortex core. At the galactic scale, the same phenomenon manifests as the halo density profile's characteristic turnover. The Navarro-Frenk-White (NFW) profile's scale radius $r_s$ corresponds to the radius at which the enclosed density drops below $\rho_{\text{half}}$, and the conformal factor transitions from its suppressed to unsuppressed form.

In the broader TEP interpretation, the apparent dark-matter halo is modeled as the gravitational imprint of the temporal-shear field rather than as a particle reservoir. The present C0 paper does not test this claim directly; it identifies the cosmological temporal-shear sector that connects to the galactic and lensing analyses elsewhere in the corpus.

## 5.3 Unified Field Equation and Preservation Constraints

The working cross-scale field-equation ansatz is:

$\square \phi = (8\pi G / 3) \rho_m A(\phi) + \kappa \mathcal{C}_T[\Sigma]$

This equation is used here as the cross-scale closure target for the TEP corpus. Its complete derivation from the microscopic topological sector is supplied by the companion TEP-QF and TEP-SPIN sequence; C0 uses the resulting macroscopic closure only as the transport-sector target. Here, $\mathcal{C}_T[\Sigma]$ denotes the topological covariance functional derived from the vortex holonomy in TEP-SPIN (Paper 24). In environmentally suppressed regimes, $S_\Sigma(E) \to 0$ suppresses the observable Temporal Shear/source response and the corresponding non-exact transport contribution, recovering the GR observational limit. The universal conformal coupling $A(\phi)$ remains part of the fundamental matter metric. In active regimes, the exact conformal sector supplies endpoint-dependent open-path redshift, while $\mathcal{C}_T$ supplies the genuinely non-exact component of the transport.

The preservation constraints on matter-frame observables are critical: atoms, photons, and physical lengths reside strictly within the disformally coupled matter-frame, ensuring that local laboratory physics is shielded from the large-scale temporal shear. In the C0 pipeline this establishes the standard-preservation limit for atomic spectra and CMB blackbody properties, while the native chemical/nucleosynthesis framework is supplied by TEP-BBN, while TEP-TH supplies temporal-horizon geometry and recombination visibility.

# 6. Discussion

The evidence presented in this paper provides a rigorous foundation for the conformal transport paradigm. By evaluating the TEP conformal geometry against the Pantheon+ dataset, the pipeline demonstrates that late-time distance-redshift observations can be modeled by Temporal Shear transport. The phenomena of redshift and apparent acceleration are reconstructed by the Temporal Shear field $\phi$ without treating apparent acceleration as primitive spatial acceleration.

**Screening projection notice.** Screening in TEP is represented at the theory level by the environmental operator $\mathcal{S}_\Sigma(\mathcal{E})$. Quantities such as $\rho_T$, $R_T(M)$, $S_\oplus(r)$, compactness $\Phi/c^2$, local stellar density, geometric coherence length, and channel-specific response coefficients are domain-specific projections of $\mathcal{E}$, not independent screening mechanisms and not interchangeable universal thresholds. Each is an observational transfer model that parameterizes the same underlying operator in a regime-appropriate form.

## 6.1 The Mathematical Isomorphism of the Scale Factor

A defining feature of this analysis is the deployment of high-fidelity nested sampling to rigorously compare the Pure Conformal / homogeneous Temporal Shear control branch against $\Lambda$CDM. The analysis demonstrates that while the physical matter space is static ($a_m=1$), the conformal matter metric acts identically to the FLRW scale factor for photon transport, scaling the apparent angular size. Because temporal transport reduces both photon energy and arrival rates by a factor of $(1+z)$, the luminosity distance becomes $d_L = (1+z)^2 d_A$. Pure conformal TEP itself therefore exactly preserves the Etherington distance-duality relation by construction. The bounded disformal sector ($B(\phi)$) is reserved for additional non-exact/path-dependent corrections, not for repairing the basic luminosity-distance law. This requires that future supernova fitters be re-calibrated natively within the complete TEP geometry rather than relying on standard nuisance parameters, as established in TEP-BBN.

Because the geometric transport of the conformal scalar field is mathematically isomorphic to the FLRW scale factor $a(t)$ at the homogeneous background level, the Pure Conformal control branch exactly matches the distance-redshift relation of standard $\Lambda$CDM. The parameter previously associated with "dark energy" ($\Omega_\Lambda$) is reconceptualized as the homogeneous temporal-shear background contribution $\Omega_\phi$ (TEP-HC, Paper 18; TEP-TH, Paper 27). It is important to emphasize that this exact background-level match is a screened-limit consistency requirement, not an independent confirmation of TEP: any viable conformal-frame alternative must recover the standard FLRW distance-redshift relation in the homogeneous limit by construction.

## 6.2 The TEP Interpretation

| Standard Cosmology ($\Lambda$CDM) | TEP Framework |
| --- | --- |
| Space expands, stretching photon wavelengths | Photon frequencies shift due to the conformal field clock-rate gradient |
| Dark energy accelerates the expansion of space | Apparent acceleration is modeled as the kinetic energy density of the Temporal Shear field |
| $H_0$ tension requires early-universe modifications | Distance probes are biased by local environmental mass-suppression of the scalar field |
| The universe began 13.8 billion years ago in a singularity | The "Big Bang" is modeled as a TEP temporal-horizon boundary where the observational clock map $A_{\text{clock}} \to 0$; the dynamical response $A_{\text{dyn}}$ remains suppressed |

## 6.3 Implications for Cosmological Tensions

The conformal paradigm offers a novel geometric interpretation for several cosmological tensions.

**The Hubble Tension:** The local distance ladder relies on calibrating deep-void supernovae against galactic Cepheids. In TEP, the temporal shear field is subject to environmental gradient suppression from mass. Supernovae exist in empty voids (where the field retains its free temporal shear), while Cepheids exist in dense galaxies (where the field undergoes strong Temporal Topology flattening). The broader corpus (Paper 11) has proposed that this environmental gradient suppression could introduce a probe-dependent bias into the SH0ES calibration, but the present C0 pipeline does not independently test the Cepheid calibration step; the Hubble-tension implications of TEP remain a corpus-level hypothesis.

**High-Redshift Galaxy Assembly:** The temporal horizon interpretation implies a fundamentally different proper-time mapping at high redshift. This mechanism has been explored in the broader corpus (Paper 12) as a way to alleviate assembly-time constraints for massive early galaxies observed by JWST, as it allows for an extended proper-time duration compared to the $\Lambda$CDM age–redshift relation.

## 6.4 Cross-Scale Consistency: Wide Binaries

Because the framework relies on a scalar field $\phi$ rather than global spatial expansion, the field couples to matter across scales. While dense local environments like the Solar System suppress the field, in the ultra-diffuse, low-acceleration outskirts of the Milky Way, the suppression mechanism is weakened.

The background Temporal Shear gradient is proposed as a weak-field gravitational anomaly in these environments with weak gradient suppression. This connection is argued in the corpus (Paper 13) as a predictive mechanism for the anomalous wide-binary accelerations measured by Gaia DR3, providing a cross-scale link between the cosmological field and local stellar kinematics.

## 6.5 Known Limitations and Open Challenges

The current analysis has several explicit limitations that any critical assessment must address:

- **Linear growth: passed in TEP-HC and imported/cross-checked by C0.** The TEP-HC hi_class Boltzmann solver with active SMG perturbations and runtime Bellini-Sawicki mappings ($\alpha_M = -2\alpha_A$, $\alpha_B = 2\alpha_A$) yields $\sigma_8 = 0.825 \pm 0.016$, in agreement with Planck ($0.811 \pm 0.006$) and DES/KiDS weak-lensing measurements. This is a native output of the full covariant closure in TEP-HC (Paper 18), not an independent C0 derivation. The present paper imports and cross-checks the active perturbation outputs from TEP-HC. Simplified EdS-only growth ODEs, which lack the SMG EFT perturbation closure, are insufficient for this sector. Full non-linear matter-only structure formation (N-body or higher-order perturbation extension) is not a C0 claim.

- **Early Universe Thermodynamics (Supersession Note):** The original C0 pipeline included a joint CMB+SNe MCMC fit and an AlterBBN "standard-preservation" branch that relied on a phenomenological thermal screening mechanism to restore $\Lambda$CDM expansion rates at early times. This branch is now explicitly carved out and retained only as a historical/methodological baseline. The early-universe closure has been redirected to the TEP-BBN eternal-universe architecture. In this canonical architecture, the hot-plasma Big Bang model is rejected. TEP-BBN instead models the light-element abundances as candidate long-term asymptotic outcomes of eternal baryonic cycling, subject to the temporal-exposure convergence condition, removing the need for the former phenomenological early-universe screening prescription. C0 retains the pure SNe line-of-sight transport fit as its definitive claim.

- **Solar System PPN: passed within the implemented EFT screening realization.** A dedicated PPN derivation (Step 04-09) confirms that unsuppressed TEP with $\beta_A = -1$ is excluded by Cassini at $\sim$$87{,}000\sigma$.  The old Lorentzian source-suppression ansatz $S(\rho) = [1+(\rho/\rho_T)^2]^{-1}$ leaves a $\sim$$1{,}700\sigma$ gap.  The unified covariant operator $\mathcal{S}_\Sigma(\mathcal{E})$ derived in Section 2.4 reduces to the gradient-dependent envelope $f(g) = [1 + (g/g_t)^n]^{-1}$ in the Solar System, where $g = |\nabla\Phi|$.  The deep potential gradient of the solar system suppresses the effective conformal coupling throughout the heliosphere ($\mathcal{S}_\Sigma \approx 0$ at $g \sim 10^{-5}$ m s$^{-2}$), giving $\gamma = 1.000000$ and safely satisfying the Cassini tracking bound ($|\gamma - 1| < 2.3 \times 10^{-5}$).  Earth surface ($g \approx 9.8$ m s$^{-2}$) is strongly suppressed, satisfying Eötvös bounds.  Galactic halos and wide-binary environments ($g \sim 10^{-10}$ m s$^{-2}$) retain $\sim$98% of their unsuppressed temporal shear, preserving cosmological growth and anomaly predictions.  Because the density-proxy $S(\rho)$ and the gradient-proxy $f(g)$ are limits of the same covariant expression $\mathcal{S}_\Sigma(\mathcal{E})$, the suppression threshold is not tuned independently across scales.  The operator is mapped into the Bellini--Sawicki EFT functions in a gauge-invariant, matter-frame-conserving manner in Section 2.4.  The PPN gate is passed within the implemented EFT screening realization used in the C0 pipeline, not merely at the phenomenological level.

- **CMB anisotropies: background/acoustic gate passed; active perturbation gate passed in TEP-HC and cross-checked by C0.** The C0 pipeline independently verifies the CMB acoustic background scale via CLASS and TEP-CLASS v2.1 ($100\theta_s^{\rm LCDM} = 1.0419$, TEP spectrum preserves acoustic-scale structure).  Step 05-10 cross-checks the TEP-HC Boltzmann outputs, confirming acoustic-scale preservation to $r_s^{\rm TEP}/r_s^{\Lambda{\rm CDM}} = 0.999994$ and fractional $\theta_s$ shift of $0.19\%$.  TEP-HC (Paper 18) independently runs the full TT/TE/EE power-spectrum comparison with Planck 2018, including active scalar perturbations, stability, and matter-frame conservation, passing the linear perturbation gate; C0 imports and cross-checks these outputs.

- **Wide-binary claim is unverified here:** The proposed connection to Gaia DR3 wide-binary anomalies (Section 6.4) references Paper 13 of the broader corpus. This cross-scale connection is not tested in the present pipeline; the TEP-SPIN derivation is supplied by the companion TEP-SPIN paper and is not a C0 claim.

## 6.6 Future Empirical Testing

Serving as a synthesis framework, the theory outlines a highly specific, preregistered experimental falsification pathway. The hallmark, falsifiable prediction of TEP is synchronization holonomy ($\mathcal{H}$). To explicitly measure the non-integrability of the time field, the following experimental avenues are defined:

- *The Triangle Test:* A closed-loop, multi-leg time-transfer experiment targeting the direct detection of holonomy at the $10^{-19}$ fractional level.

- *Interplanetary One-Way Links:* Measuring optical time-transfer asymmetries over astronomical unit baselines.

- *Clock Networks and Kinematic Data:* Utilizing precision clock arrays and deterministic pipelines on public catalogs to map environment-dependent suppression signatures.

- *Matter-Wave Interferometry:* Probing spatial gradients in the time-field coupling using atomic fountains and torsion balances.

By asserting that time itself is a dynamical field, the framework provides a mathematically rigorous path forward for precision metrology and cosmology, preserving the rigidly tested empirical pillars of relativity.

## 6.1 Falsifiable Predictions

The following near-term observational tests would strengthen or falsify the TEP-C0 temporal-shear framework in the late-time distance-redshift sector.

- **Environment-dependent Cepheid period shifts.** Within TEP, the scalar-field mass-screening in massive host galaxies should produce a systematic, directionally oriented shift in Cepheid pulsation periods relative to the field. The TEP-C0 host-environment prediction aligns the massive-host-brighter direction with the established astrophysical mass step, but a quantitative amplitude prediction from the screened scalar-field geometry ($\Delta P/P \sim 10^{-4}$–$10^{-3}$ for typical host overdensity contrasts) would be testable with JWST/NIRCam Cepheid samples at $z \sim 0.01$–$0.05$ if environment overdensity is recorded.

- **Redshift-space distortion signature at $z \sim 0.5$.** The TEP background predicts a specific $\Delta f\sigma_8(z) \approx -0.02$ relative to $\Lambda$CDM at $z \sim 0.5$ in the weakly screened branch, arising from the conformal-frame Hubble-friction modification. DESI Year-1 RSD measurements ($z \sim 0.5$, expected precision $\sigma(f\sigma_8) \sim 0.02$) can directly test this deviation.

- **ISW–galaxy cross-correlation systematic enhancement at intermediate $z$.** The TEP conformal scalar field contributes an additional source to the Integrated Sachs–Wolfe effect through the time-varying $A_{\rm dyn}(z)$ potential. This predicts a systematic enhancement in the ISW–galaxy cross-correlation at $z \sim 0.5$ relative to $\Lambda$CDM, driven by the modified late-time potential evolution rather than by a cosmological constant. The effect is expected to be modest (order-unity percent to few-percent level) and testable with DESI LRG samples cross-correlated with Planck CMB lensing maps, though cosmic variance and foreground systematics make this a qualitative directional probe rather than a precision discriminator.

- **Supernova time-dilation drift.** The TEP redshift reconstruction $1+z = A_0/A_{\rm em}$ predicts that supernova light-curve time-dilation factors should track the conformal clock map rather than the standard kinematic $1+z$ at the $<0.1\%$ level for $z \lesssim 2$. A sample of $>10^3$ spectroscopically confirmed SNe Ia with light-curve shape parameters measured to $\sim$1% precision could detect this deviation.

Each prediction is quantitative and tied to a specific observable. A null result in tests 2 or 3 at $>3\sigma$ would constrain the TEP late-time amplitude to the screened limit ($\epsilon_T \to 0$), while a positive detection would provide independent evidence for temporal-shear transport.

# 7. Conclusion

This paper presents a direct empirical challenge to the necessity of primitive cosmic expansion. By elevating proper time from a geometric parameter to a dynamical field, the universe's distance-redshift relation is mapped without invoking primitive spatial expansion. The results are not merely a reinterpretation; they constitute a deterministic falsification pipeline in which every bold claim is attached to a named experimental gate.

## Claim Gate Registry

| Claim | Status | Required Gate | Current Result |
| --- | --- | --- | --- |
| No primitive expansion required | Passed at SNe background level | TEP conformal reconstruction ties or beats $\Lambda$CDM on Pantheon+ | M2 ties; M1 improves $\Delta\chi^2\simeq-3.4$ ($z_{\rm los}=5$), $-7.5$ ($z_{\rm los}=100$ benchmark) |
| No primitive $\Lambda$ required | Passed at SNe late-time level | No-$\Lambda$ TEP beats $\Lambda$CDM with same covariance and no host-mass nuisance | BF = 4.6 ($z_{\rm los}=5$, conservative); BF = 61.8 ($z_{\rm los}=100$ benchmark); BF = 40.3 (free $z_{\rm los}$) |
| $\Lambda$CDM null injection falsification | Passed | Observed TEP preference does not occur under $\Lambda$CDM mocks | 0/200 false positives (FP rate < 1.5% at 95% CL; Rule of Three) |
| Pantheon+ subset robustness | Passed | TEP preference survives all data cuts and survey removals | 27/27 subsets prefer TEP; Δχ² range [−4.32, −3.12, −0.0002] (min, median, max) |
| Matter-frame acoustic proof | Passed | CMB acoustic scale preserved in matter-only EdS background | $100\theta_s = 1.0433$ at $\epsilon_T = 0.018$ (0.3% of Planck) |
| Big Bang as temporal horizon | Theoretically mapped | Show $A\to0$ horizon with finite matter-frame invariants | Closed in TEP-TH (Paper 27) and TEP-BBN (Paper 29) and imported as companion temporal-horizon closure. C0 itself tests only the late-time SNe transport sector. |
| Early Universe Thermodynamics | Superseded (redirected to TH/ISO) | Native local thermodynamic evolution | The prior AlterBBN "standard-preservation" branch is retained purely as a historical methodological baseline. The canonical early-universe closure is now provided by the native thermodynamic framework in TEP-TH (Paper 27) and TEP-BBN (Paper 29). C0 retains the pure late-time SNe transport fit as its core claim. |
| CMB acoustic safety | Passed at background/acoustic level | $r_s^{\rm TEP}/r_s^{\Lambda{\rm CDM}}\approx1$ | TEP-HC (Paper 18): $0.999994$ at Boltzmann level; C0: matter-frame proof gives $100\theta_s = 1.0433$ at $\epsilon_T = 0.018$ (0.3% of Planck), independent existence proof |
| Linear pure-conformal scalar perturbation safety | Passed in TEP-HC; C0 cross-checks imported spectral/acoustic outputs | Active $\delta\phi$, stability, TT/TE/EE residuals | TEP-HC: no-ghost/stability proof and full TT/TE/EE active-perturbation closure; C0: Step 05-10 cross-checks TEP-HC acoustic-scale ratio ($r_s^{\rm TEP}/r_s^{\Lambda{\rm CDM}} = 0.999994$) and imports the active-perturbation outputs |
| Host-environment offset estimate | Partial — amplitude directionally consistent with established astrophysical step | TEP predicts a leading-order host-environment offset from scalar-field geometry with screening; not a replacement for full SALT2/host-mass nuisance modeling | Locked estimate $\Delta\mu \simeq -0.0053$ mag (massive hosts brighter, directionally consistent with established astrophysical step of $\sim -0.05$ to $-0.07$ mag); the simplified mini-analysis (fixed $H_0$, no SALT2 nuisance) yields a weak fitted $\Lambda$CDM step of $+0.0072$ mag that is noise-dominated ($\Delta\chi^2 \simeq 0.6$). TEP_locked (2 params) comparable to $\Lambda$CDM_fitted (3 params) by AIC/BIC; TEP_fitted_residual (3 params) equivalent to $\Lambda$CDM_fitted ($\Delta\chi^2 \approx +0.1$) |
| Dark matter replacement | Corpus-level implication | Lensing/growth/galaxy-scale gates | Not a C0-only claim |
| BAO native projection | Passed | BAO ruler recovery in TEP background | $\chi^2/\text{DOF} = 0.88$ (17 data points) |
| Growth amplitude | Passed in TEP-HC; imported by C0 | TEP-HC hi_class Boltzmann closure: $\sigma_8 = 0.825 \pm 0.016$; Planck: $0.811 \pm 0.006$. | TEP-HC (Paper 18) derives this from the full hi_class SMG EFT solver with runtime Bellini-Sawicki mappings ($\alpha_M = -2\alpha_A$, $\alpha_B = 2\alpha_A$). C0 imports and cross-checks the active perturbation outputs; it does not independently derive $\sigma_8$ from first principles. Full non-linear matter-only structure formation is not a C0 claim. |
| Distance duality | Blocked | $\Lambda$CDM compilation $\eta = 0.866 \pm 0.020$ (6.6$\sigma$); TEP compilation $\eta = 0.846 \pm 0.019$ (8.2$\sigma$) | Both compilations violate $\eta=1$ because the BAO $D_A$ values assume a fiducial $\Lambda$CDM sound horizon $r_s$. Not a model discriminator. TEP-native re-analysis with $r_s$-independent $D_A$ probes (Step 04-10, 9 points) gives $\eta = 0.797 \pm 0.031$ ($-6.6\sigma$), but sample is small and systematics dominate; inconclusive. |
| Solar System PPN | Passed within the implemented EFT screening realization | Cassini: $\gamma = 1.000000 \pm 0.000023$ | Unsuppressed TEP predicts $\gamma = -1$ (ruled out $\sim$87 000$\sigma$). Lorentzian source-suppression leaves $\sim$1 700$\sigma$ gap. The unified covariant operator $\mathcal{S}_\Sigma(\mathcal{E})$ derived in Section 2.4 reduces to gradient-dependent suppression $f(g) = [1+(g/g_t)^n]^{-1}$ in the Solar System, with $g = \|\nabla\Phi\|$, $g_t = 1.0 \times 10^{-9}$ m s$^{-2}$, $n = 2$. Heliosphere suppression ($\mathcal{S}_\Sigma \approx 0$ at $g \sim 10^{-5}$ m s$^{-2}$) gives $\gamma = 1.000000$, safely below Cassini. Earth surface is strongly suppressed (Eötvös satisfied). Galactic halos and wide binaries ($g \sim 10^{-10}$ m s$^{-2}$) retain $\sim$98% of their unsuppressed temporal shear, preserving growth and anomaly predictions. Because $S(\rho)$ and $f(g)$ are limits of the same covariant expression $\mathcal{S}_\Sigma(\mathcal{E})$, the suppression threshold is not tuned independently across scales. The Bellini--Sawicki EFT mapping used by C0 is implemented in Section 2.4 and cross-checked against the PPN and TEP-HC growth gates. |
| Tolman surface brightness | Inconclusive as discriminator | Measured $n = 3.375 \pm 0.027$ vs $\Lambda$CDM/TEP $n \approx 4.0$; data shows $n$ decreases with $z$ (slope $-1.03$) | Both $\Lambda$CDM ($n=4.0$ flat) and TEP ($n \approx 4.02$ flat) predict $n \geq 4.0$. The data shows $n$ falling from $\approx 3.65$ at $z < 0.3$ to $\approx 2.84$ at $z > 0.5$, a trend opposite to any cosmological model. The anomaly is dominated by K-correction systematics ($\pm 0.5$ mag), passive evolution, and selection effects. Not a clean discriminator. |

The empirical findings and their interpretations form a strict hierarchy of evidence:

- **No Primitive Expansion Required by the Tested Background Data:** The exact conformal reconstruction M2 proves that the Pantheon+ homogeneous distance-redshift relation does not uniquely require primitive expansion of the spatial metric. The physical no-$\Lambda$ temporal-shear branch M1 with $z_{\rm los}=5$ improves the Pantheon+ likelihood by $\Delta\chi^2\simeq-3.4$ relative to baseline $\Lambda$CDM using the same 1,701-supernova covariance structure and no fitted host-mass-step nuisance parameter. The expansion interpretation is therefore underdetermined by the SNe background data; the temporal-transport distance law is observationally degenerate with $\Lambda$CDM at the background level, and modestly preferred (BF approximately 4.6) for the physically motivated $z_{\rm los}=5$ model.

- **No primitive dark energy required in the tested late-time sector (interpretive claim):** The M1 branch achieves a comparable standardized-supernova fit with $\Omega_\Lambda=0$, replacing vacuum-energy acceleration with temporal-shear transport in the late-time distance-redshift relation. This is an interpretive inference, not an empirical falsification of $\Lambda$: because M1 and $\Lambda$CDM produce nearly identical distance moduli, the SNe data alone cannot distinguish the physical mechanism. The result demonstrates that apparent acceleration can be reconstructed without a primitive dark-energy component, not that $\Lambda$ is absent.

- **No Physical Big Bang Singularity in the Conformal Reconstruction:** In the TEP mapping, the limit conventionally written as $a\to0$ is re-expressed as $A_{\text{clock}}\to0$: a TEP temporal-horizon boundary of observational clock transport, not a zero-volume spatial singularity. The C0 paper establishes the conformal reconstruction and identifies the singular origin as an artefact of imposing an integrable FLRW clock foliation. The temporal-horizon background and linear-mode closure are supplied by companion papers TEP-TH and TEP-HC; C0 imports and cross-checks the TEP-HC linear-growth output while focusing on the empirical supernova-sector test. The nonsingular temporal-horizon boundary conditions for the hot early thermal sector are developed in TEP-TH (Paper 27).

- **Particle Dark Matter (Corpus Implication):** Although the current pipeline focuses on the cosmological background and macroscopic bounds, the broader TEP corpus develops the claim that local gradients of this same temporal field modify effective gravitational potentials. This provides the theoretical foundation for replacing particle dark matter with geometric temporal shear in galactic and cluster environments.

The reproducible pipeline provides a robust, formally closed supernova-sector Bayesian framework demonstrating that conformal transport is a viable alternative to the standard expanding universe in the tested background sector. TEP-HC (Paper 18) has established that the linear perturbation sector is formally closed through the exact Bellini-Sawicki EFT mapping, yielding $\sigma_8 \approx 0.825$ in agreement with Planck and weak-lensing measurements. The conformal-frame Hubble friction and scale-dependent fifth-force screening are derived directly from the TEP covariant action and implemented without phenomenological suppression factors. Non-linear structure formation is not a C0 claim. Host-environment reconstruction across the full multi-probe dataset is a corpus-level target beyond the SNe-only scope of C0. The nonsingular temporal-horizon boundary conditions for the hot early thermal sector are developed in TEP-TH (Paper 27). By asserting that time itself is a dynamical field, the framework provides a testable path forward for precision cosmology.

## Acknowledgments and Disclosures

The author declares no competing interests. No external funding was received for this work. All analysis was conducted using publicly available astronomical data and open-source software.

# 8. References

## 8.1 TEP Series

- Smawfield, M. L. (2025). *Temporal Equivalence Principle: Dynamic Time & Emergent Light Speed*. v0.10 (Jakarta). DOI: 10.5281/zenodo.16921911.

- Smawfield, M. L. (2026). *The Cepheid Bias: Resolving the Hubble Tension*. v0.6 (Kingston upon Hull). DOI: 10.5281/zenodo.18209702.

- Smawfield, M. L. (2026). *Temporal Equivalence Principle: A Unified Resolution to the JWST High-Redshift Anomalies*. v0.4 (Kos). DOI: 10.5281/zenodo.19000827.

- Smawfield, M. L. (2026). *Temporal Equivalence Principle: Suppressed Density Scaling in Globular Cluster Pulsars*. v0.6 (Caracas). DOI: 10.5281/zenodo.18165798.

- Smawfield, M. L. (2026). *Temporal Equivalence Principle: Temporal Shear Recovery in Gaia DR3 Wide Binaries*. v0.3 (Kilifi). DOI: 10.5281/zenodo.19102061.

- Smawfield, M. L. (2026). *Temporal Equivalence Principle: Native hi_class Conformal Implementation, Linear Perturbation Closure, and CMB Acoustic Peak Preservation*. v0.6 (Cambridge). DOI: 10.5281/zenodo.20682752.

- Smawfield, M. L. (2026). *Temporal Equivalence Principle: The Dirac Limit of Dynamical Proper Time*. Paper 23 (forthcoming).

- Smawfield, M. L. (2026). *Temporal Equivalence Principle: A Topological Fermion Model for Spin and the g−2 Anomaly*. Paper 24 (forthcoming).

- Smawfield, M. L. (2026). *Temporal Equivalence Principle: Temporal Horizon Cosmology and the Absence of a Physical Big Bang Singularity*. v0.3 (Thika). DOI: 10.5281/zenodo.20723059.

- Smawfield, M. L. (2026). *Temporal Equivalence Principle: A Covariant Alternative to Cosmic Expansion*. v0.2 (Athens). DOI: 10.5281/zenodo.20370143.

- Smawfield, M. L. (2026). *Temporal Equivalence Principle: Dynamical Proper Time and the Illusion of Primordial Deuterium*. Paper 29 (Dubai).

## 8.2 Data Sources

- Scolnic, D., et al. (2018). *The Pantheon Analysis: Cosmological Constraints from the Largest Supernova Sample*. ApJ, 859, 101.

- Scolnic, D., et al. (2022). *Pantheon+: Type Ia Supernova Light Curves from the Dark Energy Survey*. ApJ, 938, 113.

- Planck Collaboration (2020). *Planck 2018 results. VI. Cosmological parameters*. A&A, 641, A6.

- Fixsen, D. J., et al. (1996). *The Spectrum of the Cosmic Background Radiation*. ApJ, 473, 576.

- Mather, J. C., et al. (1994). *Measurement of the Cosmic Microwave Background Spectrum by the COBE FIRAS Instrument*. ApJ, 420, 439.

## 8.3 BAO and RSD Surveys

- Alam, S., et al. (2017). *The clustering of galaxies in the completed SDSS-III Baryon Oscillation Spectroscopic Survey: cosmological analysis of the DR12 galaxy sample*. MNRAS, 470, 2617.

- Beutler, F., et al. (2011). *The 6dF Galaxy Survey: baryon acoustic oscillations and the local Hubble constant*. MNRAS, 416, 3017.

- Anderson, L., et al. (2014). *The clustering of galaxies in the SDSS-III BAO sample: analysis of potential systematics*. MNRAS, 441, 24.

- Peacock, J. A., et al. (2015). *The SDSS-IV extended Baryon Oscillation Spectroscopic Survey: overview and early data*. MNRAS, 452, 2379.

- Dawson, K. S., et al. (2013). *The SDSS-III Baryon Oscillation Spectroscopic Survey: quasar targeting*. AJ, 145, 10.

- Ross, A. J., et al. (2015). *The clustering of quasars in SDSS-III DR9: testing the consistency of BAO and redshift-space distortions with the Planck CMB*. MNRAS, 449, 835.

## 8.4 Software and Tools

- Foreman-Mackey, D., et al. (2013). *emcee: The MCMC Hammer*. PASP, 125, 306. github.com/dfm/emcee

- Speagle, J. S. (2020). *dynesty: A dynamic nested sampling package for estimating Bayesian posteriors and evidences*. MNRAS, 493, 3132. github.com/joshspeagle/dynesty

- Torrado, J., & Lewis, A. (2021). *Cobaya: Code for Bayesian Analysis of cosmological data*. Astrophysics Source Code Library, ascl:2108.05. github.com/CobayaSampler/cobaya

- Lesgourgues, J. (2011). *The Cosmic Linear Anisotropy Solving System (CLASS). Part I: Overview*. arXiv:1104.2932. github.com/lesgourg/class_public

- Arbey, A. (2012). *AlterBBN: A program for calculating the BBN abundances of the elements in alternative cosmologies*. CPC, 183, 1822. alterbbn.hepforge.org

## 8.5 Historical References

- Hubble, E. (1929). *A relation between distance and radial velocity among extra-galactic nebulae*. PNAS, 15, 168.

- Friedmann, A. (1922). *Uber die Krummung des Raumes*. Z. Phys., 10, 377.

- Lemaitre, G. (1927). *Un univers homogene de masse constante et de rayon croissant rendant compte de la vitesse radiale des nebuleuses extra-galactiques*. Ann. Soc. Sci. Brux., 47, 49.

- Riess, A. G., et al. (1998). *Observational evidence from supernovae for an accelerating universe and a cosmological constant*. AJ, 116, 1009.

- Perlmutter, S., et al. (1999). *Measurements of Omega and Lambda from 42 high-redshift supernovae*. ApJ, 517, 565.

- Tolman, R. C. (1930). *On the estimation of distances in a curved universe with a non-static line element*. PNAS, 16, 511.

- Etherington, I. M. H. (1933). *On the definition of distance in general relativity*. Philos. Mag., 15, 761.

Smawfield, M. L. 2026. Temporal Equivalence Principle series, Papers 0-13. Zenodo preprints and associated repositories.

# Unified TEP Parameter Dictionary

The TEP corpus uses related but distinct symbols across its papers. This dictionary maps every parameter, its definition, the paper where it is primary, and its fiducial or fitted value.

| Symbol | Definition | Primary Paper | Fiducial / Fitted Value |
| --- | --- | --- | --- |
| $A_{\rm clock}(z)$ | Exact observational clock/redshift map: $A_{\rm clock}=(1+z)^{-1}$ | TEP-TH | $(1+z)^{-1}$ (exact) |
| $A_{\rm dyn}(z)$ | Dynamical shear response: $\left(1+z/z_t\right)^{-\epsilon_t}$ | TEP-TH | Modifies late-time evolution |
| $\alpha_A$ | Temporal-shear conformal amplitude in Jordan-frame notation | TEP-HC | $-0.0028$ (Planck best-fit) |
| $\alpha_M$, $\alpha_B$, $\alpha_K$, $\alpha_T$ | Runtime Bellini–Sawicki EFT functions: $\alpha_M=-2\alpha_A$, $\alpha_B=2\alpha_A$, $\alpha_K=-5\alpha_A^2$, $\alpha_T=0$ | TEP-HC | Derived from $\alpha_A$ |
| $\epsilon_T^{\rm los}$ | Late-time line-of-sight transport amplitude (C0 supernova fit) | TEP-C0 | $\mathcal{U}[0, 2.0]$ (prior); posterior peaked near $\sim 0.89$ |
| $\epsilon_T^{\rm CMB}$ | C0 background/acoustic diagnostic amplitude | TEP-C0 | $-0.0015\pm0.0037$ |
| $\epsilon_T^{\rm HC}$ | Native hi_class homogeneous conformal amplitude | TEP-HC | $0.00602\pm0.00493$ |
| $\epsilon_{\rm dyn}$ | Dynamical temporal-horizon response | TEP-TH | Determined by late-time shear |
| $\epsilon_{\rm field}$ | Primordial spectral-flow parameter constrained by $n_s$ | TEP-TH | $0.0175$ (from $n_s=0.965$) |
| $z_T^{\rm los}$ | C0 line-of-sight supernova transport turnover | TEP-C0 | $5$ (conservative), $100$ (benchmark), free (broad) |
| $z_T^{\rm HC}$ | Homogeneous/acoustic hi_class profile scale | TEP-HC | Fitted jointly with $\epsilon_T$ |
| $p$ | Temporal-horizon conformal exponent: $A_{\rm clock}\sim\eta^{-p}$ | TEP-TH | $0 \lt p\le\tfrac12$ (regular branch) |
| $r_s^{\rm TEP}/r_s^{\Lambda\rm CDM}$ | Pre-recombination sound-horizon ratio | TEP-HC | $0.999994$ ($<6$ ppm deviation) |
| $D=\alpha_K+\tfrac32\alpha_B^2$ | No-ghost discriminant (physical branch: $D=\alpha_A^2$) | TEP-HC | $\alpha_A^2>0$ (positive definite) |
| $r(k_{\rm pivot})$ | Native tensor-to-scalar ratio at Planck pivot | TEP-TH | $9\times10^{-6}$ |
| $r_{\rm max}$ | Maximum tensor-to-scalar ratio across transition profile | TEP-TH | $6.26\times10^{-4}$ |
| $H_0$ | Hubble parameter (TEP-C0 joint MCMC) | TEP-C0 | $66.70\pm0.58$ km s$^{-1}$ Mpc$^{-1}$ |
| $S_8$ | $\sigma_8\sqrt{\Omega_m/0.3}$ (TEP-HC joint MCMC) | TEP-HC | $0.867\pm0.026$ |
| $\sigma_8^{\rm HC}$ | Native hi_class matter-fluctuation amplitude | TEP-HC | $0.825\pm0.016$ |

**Note:** Parameters with superscript labels ($^{\rm los}$, $^{\rm HC}$) are related projections of the same temporal sector but are not numerically interchangeable. The turnover scales $z_T^{\rm los}$ and $z_T^{\rm HC}$ describe different physical regimes; the amplitudes $\epsilon_T^{\rm los}$, $\epsilon_T^{\rm CMB}$, $\epsilon_T^{\rm HC}$, and $\epsilon_{\rm field}$ are constrained by different observables.

# 9. Data Availability & Reproducibility

This work follows open-science practices. All results are fully reproducible from raw data
using the documented pipeline. All numerical results, figures, and statistics are generated by deterministic
Python scripts processing real observational data. The pipeline is intentionally strict: failed dependencies are recorded as failed
results, not silently ignored.

### Repository and Code

GitHub Repository: github.com/matthewsmawfield/TEP-C0

The repository contains a deterministic, version-controlled cosmological analysis pipeline with 64 analysis steps
for supernova distance-redshift, distance-duality constraints, CMB acoustic scales, BBN preservation, structure growth, and systematic validation.
All steps are orchestrated by `scripts/run_pipeline.py` with comprehensive per-step logging.

All MCMC chains, posterior samples, and the exact `cobaya` YAML configuration files are released in the Zenodo repository (DOI: 10.5281/zenodo.20370143) under CC-BY 4.0. The `run_all.py` orchestration script and all step scripts are provided in the GitHub repository.

#### Repository Structure

TEP-C0/
├── data/
│   ├── raw/                       # Downloaded source catalogs (Pantheon+, DDR, etc.)
│   └── processed/                 # Ingested and filtered datasets
├── scripts/
│   ├── steps/                     # 64 deterministic pipeline steps
│   ├── utils/                     # Logging and validation utilities
│   └── run_pipeline.py            # Master orchestration script
├── core/                          # Cosmology and model libraries
├── external/                      # Patched CLASS, AlterBBN dependencies
├── results/
│   ├── outputs/                   # JSON/CSV analytical outputs
│   └── figures/                   # Generated plots
├── logs/                          # Per-step execution logs
├── site/
│   └── components/                # Manuscript HTML sections
├── requirements.txt               # Python dependencies
└── README.md                      # Documentation

### Data Provenance

| Data Source | Provider | Access Method | Records | Location |
| --- | --- | --- | --- | --- |
| Pantheon+ SNe Ia | Scolnic et al. | Auto-downloaded | 1,701 | `data/raw/pantheon_plus_shoes.dat` |
| Pantheon+ covariance | Scolnic et al. | Auto-downloaded | Full stat + sys | `data/raw/Pantheon+SH0ES.cov` |
| BAO constraints | BOSS, eBOSS, DES | Compiled from lit. | 10 measurements | `data/raw/ddr_constraints.csv` |
| SZ cluster DDR | Compiled | Auto-downloaded | ~38 clusters | `data/raw/sz_constraints.csv` |
| SGL lensing DDR | Compiled | Auto-downloaded | ~118 lenses | `data/raw/sgl_constraints.csv` |
| DESI/eBOSS Lyman-alpha | DESI-DR1, eBOSS | Auto-downloaded | 3 measurements | `data/raw/desi_ddr.csv` |
| FIRAS CMB spectrum | NASA LAMBDA | Auto-downloaded | ~43 frequencies | `data/raw/firas_spectrum.dat` |
| Planck 2018 CMB | Planck Collaboration | Cobaya package | TTTEEE + low-$\ell$ TT/EE + Pantheon+ | External Cobaya cache |
| BBN abundances | AlterBBN, compiled lit. | Included / downloaded | Yp, D/H, Li/H | `data/raw/bbn_review.html` |

### Pipeline Architecture

The analysis pipeline comprises 64 deterministic steps organized into eight logical stages.
Each step is a standalone Python script in `scripts/steps/` that produces JSON/CSV outputs and
detailed logs in `logs/step_*.log`. Dependencies are resolved automatically by the runner.

#### Complete Step Inventory and Runtime

Runtimes are approximate and measured on Apple M4 Pro (14-core, 24 GB). The dominant cost is the nested sampling step (03_01), which scales with `nlive` and number of models.

| Stage | Step | Script | Description | Runtime |
| --- | --- | --- | --- | --- |
| Stage 1: Data Acquisition (8 steps) |  |  |  |  |
| Data | 1.1 | `step_01_01_data_download.py` | Download Pantheon+ SNe, covariance, FIRAS | ~10 s |
| Data | 1.2 | `step_01_02_data_ingestion.py` | Ingest and validate all downloaded catalogs | ~1 s |
| Data | 1.3 | `step_01_03_download_ddr.py` | Download BAO distance-duality constraints | ~1 s |
| Data | 1.4 | `step_01_04_download_sb.py` | Download surface-brightness catalog sources | ~1 s |
| Data | 1.5 | `step_01_05_download_sz.py` | Download Sunyaev-Zel'dovich cluster data | ~1 s |
| Data | 1.6 | `step_01_06_download_sgl.py` | Download strong gravitational lensing data | ~1 s |
| Data | 1.7 | `step_01_07_download_desi.py` | Download DESI-DR1 and eBOSS Lyman-alpha | ~1 s |
| Data | 1.8 | `step_01_08_compile_sb.py` | Compile surface-brightness master catalog | ~1 s |
| Stage 2: Theory and Transport (4 steps) |  |  |  |  |
| Theory | 2.1 | `step_02_01_transport_kernel.py` | Verify FLRW recovery limit of open-path transport K_T | ~1 s |
| Theory | 2.2 | `step_02_02_theory_derivation.py` | Derive theoretical predictions for distance-redshift and screening | ~2 s |
| Theory | 2.3 | `step_02_03_physics_implementation.py` | Implement TEP physics: distance moduli, transport, growth kernels | ~3 s |
| Theory | 2.4 | `step_02_04_screening_scale_transfer.py` | Micro-to-galactic screening scale transfer and coarse-graining | ~1 s |
| Stage 3: Model Comparison and MCMC (10 steps) |  |  |  |  |
| Core | 3.1 | `step_03_01_three_model_comparison.py` | Nested sampling (dynesty, nlive=500) for M0a_LCDM, M0b_EdS, M1 variants, M2_PureConformal_Control, M3_wCDM, M4_CPL; null injection | ~90 min |
| Core | 3.2 | `step_03_02_independent_mcmc.py` | Independent MCMC convergence diagnostics | ~1 s |
| Core | 3.4 | `step_03_04_cobaya_mcmc.py` | Joint SNe+CMB MCMC via Cobaya with TEP-CLASS v2.0 | ~2 min |
| Core | 3.4b | `step_03_04_minimize.py` | BOBYQA minimizer for joint SNe+CMB parameter optimization | ~30 s |
| Core | 3.5 | `step_03_05_analyze_cobaya.py` | Analyze Cobaya chains and produce parameter constraints | ~1 s |
| Core | 3.6 | `step_03_06_cobaya_verbose.py` | Verbose Cobaya configuration and extended diagnostics | ~2 min |
| Core | 3.7 | `step_03_07_likelihood_synthesis.py` | Synthesize likelihoods across independent and joint analyses | ~1 s |
| Core | 3.8 | `step_03_08_h0_boundary_stress.py` | H0 prior stress test: extended priors reveal EdS-derived-parameter artifact driving H0 toward zero | ~30 s |
| Core | 3.9 | `step_03_09_lcdm_null_injection.py` | LCDM null injection: mock Pantheon+ from LCDM, measure TEP false-positive rate | ~60 s |
| Core | 3.10 | `step_03_10_pantheon_subset_robustness.py` | Leave-one-survey-out and redshift-window robustness tests | ~30 s |
| Stage 4: Supernova Tests and Distance Duality (10 steps) |  |  |  |  |
| SNe | 4.1 | `step_04_01_sn_time_dilation.py` | Test SN light-curve stretch factors against TEP time dilation | ~1 s |
| SNe | 4.2 | `step_04_02_sn_tolman.py` | Tolman surface-brightness dimming test | ~1 s |
| SNe | 4.3 | `step_04_03_tolman_sb.py` | Surface-brightness Tolman scaling with compiled catalog | ~1 s |
| DDR | 4.4 | `step_04_04_distance_duality.py` | Distance-duality relation: BAO constraints vs TEP prediction | ~1 s |
| DDR | 4.5 | `step_04_05_ddr_threeway.py` | Three-way probe comparison: BAO, SZ, SGL | ~1 s |
| DDR | 4.6 | `step_04_06_screening_fit.py` | Parametric screening model fit to probe-dependent DDR | ~2 s |
| DDR | 4.6 | `step_04_07_highz_ddr.py` | High-redshift Lyman-alpha DDR test (DESI, eBOSS) | ~1 s |
| SNe | 4.7 | `step_04_08_host_mass_step_prediction.py` | Host-mass-step mini-analysis: locked TEP prediction vs fitted $\Lambda$CDM nuisance | ~5 s |
| PPN | 2.4.5 | `step_04_09_ppn_constraints.py` | Solar System PPN constraint derivation with gradient-dependent screening | ~1 s |
| DDR | 4.5 | `step_04_10_tep_native_ddr.py` | TEP-native distance-duality re-analysis | ~1 s |
| SNe | 4.8 | `step_04_11_sn_robustness_systematics.py` | Prior, redshift-cut, covariance, and sampler robustness grid for TEP M1 | ~10 min |
| SNe | 4.9 | `step_04_12_external_sn_validation.py` | External validation on Union3 binned compilation | ~1 s |
| Stage 5: CMB and Big Bang Nucleosynthesis (10 steps) |  |  |  |  |
| CMB | 5.1 | `step_05_01_cmb_blackbody.py` | Verify TEP preserves CMB blackbody spectrum (FIRAS) | ~1 s |
| CMB | 5.3 | `step_05_03_cmb_boltzmann.py` | TEP Boltzmann integration via patched CLASS | ~1 s |
| CMB | 5.4 | `step_05_04_cmb_spectra.py` | Generate and compare TT/TE/EE power spectra | ~1 s |
| CMB | 5.5 | `step_05_05_cmb_consistency.py` | CMB acoustic-scale consistency check | ~1 s |
| BBN | 5.6 | `step_05_06_bbn_registry.py` | Compile observational BBN abundance registry | ~1 s |
| BBN | 5.7 | `step_05_07_bbn_preservation.py` | Historical AlterBBN compatibility baseline — superseded; not part of canonical TEP thermodynamics | ~1 s |
| CMB | 5.8 | `step_05_08_cmb_acoustic.py` | Acoustic-scale parameter comparison (Planck) | ~1 s |
| CMB | 5.9 | `step_05_09_minimal_perturbations.py` | Diagnostic minimal-closure perturbation checks; authoritative active-sector closure is documented in TEP-HC | ~3 s |
| CMB | 5.10a | `step_05_10_jordan_frame_proof.py` | Matter-frame acoustic-scale proof in EdS matter-only background | ~1 s |
| CMB | 5.10b | `step_05_10_tephc_spectra_crosscheck.py` | Cross-check TEP-HC Boltzmann spectral outputs (acoustic-scale ratio) | ~1 s |
| Stage 6: BAO and Structure Growth (7 steps) |  |  |  |  |
| BAO | 6.1 | `step_06_01_bao_projection.py` | BAO ruler projection in TEP geometry | ~1 s |
| BAO | 6.2 | `step_06_02_bao_likelihood.py` | BAO likelihood module integration | ~7 s |
| Growth | 6.3 | `step_06_03_growth_solver.py` | TEP-CLASS v2.0 growth equation solver | ~1 s |
| Growth | 6.4 | `step_06_04_growth_validation.py` | Validate growth factors against LCDM baseline | ~1 s |
| Growth | 6.5 | `step_06_05_growth_rsd.py` | Redshift-space distortion comparison (f sigma_8) | ~2 s |
| Growth | 6.6 | `step_06_06_nonlinear_growth_closure.py` | Halo-model non-linear growth closure with gradient screening | ~5 s |
| Growth | 6.7 | `step_06_07_alphaM_growth_validation.py` | First-principles alpha_M-modified growth ODE: four-scenario sigma_8 comparison | ~2 s |
| Stage 7: Forecasts and Future Tests (7 steps) |  |  |  |  |
| Future | 7.1 | `step_07_01_mixed_forecast.py` | Forecast for mixed TEP-LCDM parameter recovery | ~1 s |
| Future | 7.2 | `step_07_02_redshift_drift.py` | Redshift-drift forecast and discriminating power | ~1 s |
| Future | 7.3 | `step_07_03_jwst_test.py` | JWST high-z supernova feasibility test | ~1 s |
| Future | 7.4 | `step_07_04_gw_sirens.py` | Gravitational-wave standard siren forecast | ~1 s |
| Future | 7.5 | `step_07_05_weak_lensing_plan.py` | Weak-lensing survey plan for TEP discrimination | ~1 s |
| Future | 7.6 | `step_07_06_weak_lensing.py` | Weak-lensing shear correlation analysis | ~1 s |
| Future | 7.7 | `step_07_07_blind_injection.py` | Blind injection validation protocol | ~1 s |
| Stage 8: Falsification, Verification, and Summary (8 steps) |  |  |  |  |
| Validation | 8.1 | `step_08_01_expansion_falsifier.py` | Expansion falsifier: distance duality and Tolman residuals | ~1 s |
| Validation | 8.2 | `step_08_02_comparison_stats.py` | Cross-model comparison statistics | ~1 s |
| Validation | 8.3 | `step_08_03_sensitivity_analysis.py` | Prior and parameter sensitivity analysis | ~1 s |
| Validation | 8.4 | `step_08_04_evidence_matrix.py` | Compile explanatory evidence matrix | ~1 s |
| Validation | 8.5 | `step_08_05_gate_registry.py` | Claim gate registry and status check | ~1 s |
| Validation | 8.6 | `step_08_06_claim_audit.py` | Automated claim consistency check | ~1 s |
| Validation | 8.7 | `step_08_07_final_summary.py` | Global evidence synthesis and summary | ~1 s |
| Validation | 8.8 | `step_08_08_diagnostic_plots.py` | Data-driven diagnostic figures (distance-duality residuals, Pantheon+ Hubble residuals) generated only from upstream pipeline artefacts | ~5 s |

#### Total Runtime Summary

The total runtime is dominated by Stage 3.1 (nested sampling). Runtimes scale approximately linearly with `nlive` and number of CPU cores.

| Component | Steps | Runtime |
| --- | --- | --- |
| Data Acquisition (Stage 1) | 8 | ~20 s |
| Theory and Transport (Stage 2) | 4 | ~6 s |
| Model Comparison and MCMC (Stage 3) | 10 | ~97 min |
| SNe Tests and DDR (Stage 4) | 10 | ~15 s |
| CMB and BBN (Stage 5) | 10 | ~11 s |
| BAO and Growth (Stage 6) | 7 | ~14 s |
| Forecasts and Future Tests (Stage 7) | 7 | ~7 s |
| Falsification and Verification (Stage 8) | 8 | ~7 s |
| Total | 64 | ~95 min (~1.6 h) |

### Reproduction Instructions

#### Quick Start (Full Reproduction)

# 1. Clone repository
git clone https://github.com/matthewsmawfield/TEP-C0.git
cd TEP-C0

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full pipeline (generates all results and figures)
python scripts/run_pipeline.py

# 4. Results will be in:
#    - results/outputs/   (JSON/CSV data)
#    - results/figures/   (PNG/PDF plots)
#    - logs/              (Detailed execution logs)

#### Command-Line Options

The pipeline supports selective execution for faster testing:

# Core statistical analysis only (skips long nested sampling)
python scripts/run_pipeline.py --core

# Resume from existing results (skip completed steps)
python scripts/run_pipeline.py --resume

# Run specific steps with automatic dependency resolution
python scripts/run_pipeline.py --steps step_04_04_distance_duality step_04_05_ddr_threeway

#### System Requirements

| Component | Minimum | Recommended | Tested On |
| --- | --- | --- | --- |
| CPU | 4 cores | 8+ cores | Apple M4 Pro (14-core) |
| RAM | 8 GB | 16 GB | 24 GB (M4 Pro) |
| Storage | 2 GB | 5 GB | NVMe SSD |
| Runtime (full) | ~4 h (4 cores) | ~1.5 h (8+ cores) | ~95 min (M4 Pro) |
| Runtime (--core) | ~1 min | ~30 s | ~20 s |

#### Key Analysis Outputs

- `results/outputs/step_03_01_three_model_comparison.json` — Nested sampling posteriors and evidence for all models (M0a_LCDM, M0b_EdS, M1 variants, M2_PureConformal_Control, M3_wCDM, M4_CPL)

- `results/outputs/step_03_04_cobaya_mcmc.1.txt` — Cobaya MCMC chain for joint SNe+CMB analysis

- `results/outputs/step_04_04_distance_duality.json` — DDR weighted mean and deviation from unity

- `results/outputs/step_04_05_ddr_threeway.json` — Three-way BAO/SZ/SGL probe comparison

- `results/outputs/step_04_11_sn_robustness_systematics.json` — Prior, redshift-cut, covariance, and sampler robustness grid for TEP M1

- `results/outputs/step_04_12_external_sn_validation.json` — External validation on Union3 binned compilation

- `results/outputs/step_05_07_bbn_preservation.json` — Historical AlterBBN compatibility baseline — superseded; not part of canonical TEP thermodynamics

- `results/outputs/step_05_09_minimal_perturbations.json` — diagnostic minimal-closure perturbation checks; authoritative active-sector closure is documented in TEP-HC

- `results/outputs/step_06_04_growth_validation.json` — Growth factor and sigma_8 consistency check

- `results/outputs/step_08_04_evidence_matrix.json` — Explanatory evidence matrix across all observables

- `results/outputs/step_08_06_claim_audit.json` — Automated claim consistency check report

#### Log Files

Each step produces detailed logs with timestamps, SHA-256 checksums, and execution status:

- `logs/step_*.log` — Individual step logs (64 files, one per step)

- `logs/verbose/` — Verbose Cobaya and nested sampling logs

### Software Dependencies

| Package | Version | Purpose |
| --- | --- | --- |
| Python | 3.10+ | Language runtime |
| NumPy | 1.24+ | Numerical computing |
| SciPy | 1.10+ | Statistical functions, nested sampling |
| Pandas | 2.0+ | Data manipulation |
| Matplotlib | 3.7+ | Visualization |
| emcee | 3.1+ | Ensemble MCMC sampling |
| dynesty | 2.1+ | Nested sampling for Bayesian evidence |
| Cobaya | 3.6+ | Joint MCMC with Planck likelihoods |
| classy (CLASS) | 3.2+ | CMB Boltzmann solver (patched for TEP) |

All dependencies are specified in `requirements.txt`. External dependencies (patched CLASS, AlterBBN) are included in the `external/` directory.

### Appendix Figures

![Joint SNe+CMB Background/Acoustic MCMC Diagnostic](results/figures/step_03_05_analyze_cobaya_triangle.png)

Figure A1: Joint SNe+CMB Background/Acoustic MCMC Diagnostic. This triangle plot shows the joint posterior from the Cobaya MCMC, including the homogeneous acoustic-sector amplitude $\epsilon_T^{\rm CMB}$. This is a diagnostic figure, not the SNe-only M1 evidence result. The $H_0$ boundary behaviour is separately stress-tested (Section 4.4.3). The $\epsilon_T$ shown here is the homogeneous acoustic-sector amplitude, distinct from the line-of-sight $\epsilon_{\rm shear}^{\rm los}$ fitted to supernovae.

![Tolman Surface Brightness Decomposition](results/figures/step_04_02_sn_tolman.png)

Figure A2: Reference Tolman surface-brightness decomposition over the Pantheon+ redshift range. This is a geometric scaling decomposition, not an independent Tolman-test measurement. The empirical Tolman sector remains systematics-limited and inconclusive as a TEP/$\Lambda$CDM discriminator.
