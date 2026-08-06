# Temporal Topology Saturation Scale: Cross-Scale Consistency of ρ_T
Matthew Lukin Smawfield
Version: v0.7 (New Delhi)
First published: 28 December 2025 · Last updated: 3 July 2026
DOI: 10.5281/zenodo.18064365

---

## Abstract

Dark matter observations across cosmological scales exhibit a
regularity: the characteristic radius at which Newtonian dynamics fails
scales as $R \propto M^{1/3}$, implying a universal saturation/proximity
scale (historically labelled a critical density), observationally proxied by $\rho_T$. This scaling appears in galaxy rotation curves (SPARC
database), ultra-diffuse galaxies (DF2/DF4), and the Milky Way's dark-matter
onset. Within TEP, this pattern is interpreted as evidence of a
candidate saturation scale in the conformal time-field sector, where
field saturation occurs at a characteristic proximity scale.

Terrestrial calibration—derived from a newly identified
distance-structured correlation in GNSS atomic clocks—provides an
empirical anchor. A 25-year CODE analysis yields $L_c \approx 4{,}200$
km for Earth's mass ($M_\oplus \approx 6 \times 10^{27}$ g), consistent with
multi-center results (CODE, IGS, ESA). The
characteristic length $L_c$ is operationally identified with the projected
Temporal Topology covariance scale associated with $R_T(M_\oplus)$,
the geometric saturation scale for Earth's mass; a soliton interpretation
is one candidate microscopic realization, not assumed in the calibration. This implies
$\rho_T \approx 20$ g/cm³. This calibration exhibits 25-year temporal
stability and survives raw RINEX validation, constraining
processing-artifact explanations.

Galactic-scale consistency is tested using the SPARC rotation curve database
(167 galaxies). The empirical dark matter onset scaling is $\alpha_{\rm SPARC} =
0.355 \pm 0.043 \text{ (stat)} \pm 0.07 \text{ (definition)}$, consistent with the $M^{1/3}$ expectation within
$\sim$0.3$\sigma$. For the Milky Way, the SPARC-calibrated $M^{1/3}$ relation predicts a dark-matter onset radius $R_{\text{DM}} \approx 3$ kpc, consistent with the observed transition from baryonic to dark-matter-dominated rotation at $R \sim 3$–5 kpc. For ultra-diffuse
galaxies DF2 and DF4, the model predicts Temporal Topology saturation radii exceeding tidal
radii, consistent with observed dark matter deficiency via tidal
stripping of the scalar field envelope.

Temporal Topology screening provides a candidate explanation for why
precision GR tests remain satisfied. A hierarchy of 26 astrophysical objects spanning 15
orders of magnitude in density is assembled; regression on the 11 dense
objects ($\rho > \rho_T$) yields $S \propto \rho^{0.334}$ ($R^2 = 0.99995$),
algebraically expected from the
$R_T(M)$ construction. This provides a candidate screening explanation for why GR tests pass (binary pulsars:
$S \sim 29{,}000$) while galactic dynamics ($S \sim 10^{-9}$ at $\rho \sim 10^{-24}$ g/cm³)
are deeply unscreened, exhibiting strong scalar effects.

The saturation density $\rho_T \approx 20$ g/cm³ emerges as a candidate
macroscopic saturation/proximity scale in the TEP environmental-response
hierarchy, not an ambient-density switch. GNSS provides the primary
terrestrial timing anchor; SPARC onset scaling, Milky Way structure,
RBH-1 crossover behaviour, compact-object screening, and condensed-matter/EOS
considerations then test whether the same scale organizes otherwise
disconnected gravitational regularities without retuning. Channel-specific
quantities such as $\rho_{\rm trans}$, $R_s$, $\kappa_{\rm Cep}$,
$\kappa_{\rm MSP}$, and lensing response coefficients are downstream
projections of $S_\Sigma(\mathcal{E})$, not independent measurements of
$\rho_T$.

The $\rho^{1/3}$ hierarchy is a consistency relation induced by the
$R_T(M)$ construction; it is not, by itself, an independent discriminator
of microscopic screening mechanism.

*Evidence hierarchy.* Within the evidence hierarchy of the TEP series,
the GNSS clock analyses (Papers 1–3) provide the primary empirical input
for the terrestrial correlation scale; the present paper tests the
cross-scale consequences of conditionally identifying that scale with
$R_T(M_\oplus)$.

*Keywords:* dark matter – gravitation – scalar fields – galaxies:
kinematics and dynamics – temporal equivalence principle

## 1. Introduction: The Dark Matter Problem as a Temporal Structure Problem

### The Universal Scaling Anomaly

Dark matter observations across cosmological scales reveal an empirical regularity. From dwarf galaxies ($M \sim 10^9 M_\odot$) to massive galaxies ($M \sim 10^{12} M_\odot$), the characteristic radius at which Newtonian dynamics fails—the "dark matter onset"—scales approximately as $R \propto M^{1/3}$. This implies a universal critical proximity scale ("critical" here meaning a phenomenological saturation/proximity scale, not a hard density threshold), observationally proxied by $\rho_T \sim M/R^3 \sim \text{constant}$, at which gravitational phenomenology changes.

This scaling appears in multiple independent contexts:

- **Galaxy rotation curves:** SPARC database analysis yields $R_{\text{DM}} \propto M^{\alpha_{\rm SPARC}}$ with $\alpha_{\rm SPARC} = 0.355 \pm 0.043 \text{ (stat)} \pm 0.07 \text{ (definition)}$ (Lelli et al. 2016; this work).

- **Ultra-diffuse galaxies:** DF2 and DF4 exhibit dark matter deficiency consistent with saturation radii exceeding tidal radii (van Dokkum et al. 2018, 2019).

- **Milky Way:** The SPARC-calibrated $M^{1/3}$ relation predicts a dark-matter onset at $R_{\text{DM}} \approx 3$ kpc for the Milky Way, consistent with the observed transition from baryonic to dark-matter-dominated rotation at $R \sim 3$–5 kpc (Gaia Collaboration 2023; Sofue 2020).

The persistence of this $M^{1/3}$ scaling across 6 orders of magnitude in mass and 15 orders of magnitude in density suggests a fundamental physical scale, not a coincidence of baryonic feedback or halo assembly.

### Reframing Dark Matter: Phantom Mass from Temporal Shear

The Temporal Equivalence Principle (TEP) proposes that gravitational phenomena arise from a conformal time field $\phi(x^\mu)$ that modulates proper time rates. This phenomenology arises from the TEP action (Smawfield 2025a, Paper 0):

\begin{equation} \label{eq:tep_action} S = \int d^4x \sqrt{-g} \left[ \frac{M_{\text{Pl}}^2}{2} R - \frac{1}{2}(\partial \phi)^2 - V(\phi) \right] + S_m[\tilde{g}_{\mu\nu}] \end{equation}

where matter couples to the Jordan frame metric $\tilde{g}_{\mu\nu} = A^2(\phi) g_{\mu\nu} + B(\phi) \nabla_\mu\phi \nabla_\nu\phi$. At late times, $B(\phi)$ is strongly constrained by multi-messenger observations and can be neglected, yielding $\tilde{g}_{\mu\nu} \approx A^2(\phi) g_{\mu\nu}$ with $A(\phi) = \exp(\beta_A \phi/M_{\text{Pl}})$. The scalar saturation potential $V(\phi)$ prevents the gradient from diverging, leading to field saturation at characteristic density $\rho_T$. In this framework, the "dark matter" problem is reinterpreted as a violation of the *isochrony axiom*—the assumption that clocks at the same gravitational potential tick at the same rate regardless of their spatial separation or the mass distribution's history.

When this axiom is relaxed, gravitational lensing and dynamical mass estimates diverge. Light propagation depends on the *integrated* time dilation along the null geodesic, while orbital dynamics depend on the *local* time gradient. This creates "phantom mass"—an apparent excess in lensing mass relative to dynamical mass—without invoking non-baryonic particles.<sup>†</sup>

<sup>†</sup>*Note: This geometric "phantom mass" from temporal shear differs from cosmological "phantom energy" (dark energy with $w < -1$). The former arises from spatial gradients in proper time; the latter from exotic equation-of-state matter.*

The characteristic scale at which this temporal structure becomes significant is set by the saturation scale $\rho_T$, where the scalar field $\phi$ reaches its self-interaction threshold and saturates—forming a stable, localized configuration (a soliton under that interpretation) with radius:

\begin{equation} \label{eq:saturation_radius} R_T = \left( \frac{3M}{4\pi \rho_T} \right)^{1/3}. \end{equation}

This $M^{1/3}$ scaling is a direct consequence of the saturation condition; under a soliton interpretation, this corresponds to soliton formation.

The saturation scale parameter $\rho_T \approx 20$ g/cm³ serves as the bulk-matter realization of the abstract environmental screening operator $\mathcal{S}_\Sigma(\mathcal{E})$ defined in the foundational TEP framework. In macroscopic atomic matter, $\rho_T$ acts as an empirical proxy for the geometric saturation of the Temporal Topology, yielding cross-scale predictions without committing to specific chameleon, Vainshtein, or symmetron microphysics.

\begin{equation} \label{eq:screening_operator}
\Sigma_\mu^{\text{obs}} = \mathcal{S}_\Sigma(\mathcal{E}) \, \nabla_\mu \ln A(\phi),
\end{equation}

where the environmental state is
$\mathcal{E} = \{\rho, \Phi/c^2, \nabla\rho, \nabla\Phi, \text{compactness}, R_T(M), \text{proximity}, T, z, \text{boundary geometry}, \text{coherence volume}\}.$
This single operator unifies density screening, compactness screening, proximity screening, thermal/epoch screening, lensing/cosmological covariance screening, and wide-binary environmental screening across the entire TEP corpus.

#### Box 1.1: Joint Constraint Architecture of \(\rho_T\) and \(S_\Sigma(\mathcal{E})\)

\(\rho_T\) is the shared macroscopic saturation/proximity scale. Other
papers in the TEP programme do not all measure \(\rho_T\) directly; they
measure domain-specific projections of the same environmental operator
\(S_\Sigma(\mathcal{E})\):

- **Clock covariance lengths** (GNSS, Paper 1–3)

- **Galactic onset radii** (SPARC, Milky Way, this work)

- **Response coefficients** (Cepheids, pulsars, Papers 10–11)

- **Lensing residuals** (strong-lens time delays, Paper 4)

- **Weak-field recovery scales** (wide binaries, Paper 13)

- **Compactness-dependent residuals** (LLR, Paper 17)

- **Cosmological temporal-shear amplitudes** (hi_class, C0, TH, Papers 18, 26, 27)

Each measurement channel has a response coefficient \(\kappa_X\), an
environmental screening projection, and a domain-specific observable
functional. This architecture protects the programme from the criticism
that different channels report different numerical values: they are
*observable response coefficients*, not bare scalar couplings,
and each channel measures a different projection of
\(S_\Sigma(\mathcal{E})\).

### The Multi-Scale Consistency Strategy

As summarized in Table 1, this paper tests $\rho_T$ through a convergent multi-scale approach:

### Table 1: Cross-Scale Consistency of $\rho_T$

| Scale | System | Mass Range | Density Range | Key Observable | Result |
| --- | --- | --- | --- | --- | --- |
| **Terrestrial** | GNSS Clocks | $M_\oplus \sim 6 \times 10^{27}$ g | $\rho_\oplus \sim 5.5$ g/cm³ | $L_c \approx 4200$ km | $\rho_T \approx 20$ g/cm³ |
| **Galactic** | SPARC Galaxies | $10^9$–$10^{12} M_\odot$ | $\rho \sim 10^{-24}$ g/cm³ | $R_{\text{DM}} \propto M^{\alpha_{\rm SPARC}}$ | $\alpha_{\rm SPARC} = 0.355 \pm 0.043 \text{ (stat)} \pm 0.07 \text{ (def.)}$ |
| **Local** | Milky Way | $M_{\rm bar} \sim 6 \times 10^{10} M_\odot$ | $\rho \sim 10^{-24}$ g/cm³ | Inner mass-discrepancy onset | $R_{\text{DM}} \approx 3$ kpc (observed $\sim$3–5 kpc) |
| **Screening** | 26 Objects | $\sim 10^{27}$–$10^{33}$ g | $\sim 10^{0}$–$10^{14}$ g/cm³ | $S \propto \rho^{\beta_{\rm scr}}$ | $\beta_{\rm scr} = 0.334$ ($R^2 = 0.99995$) |

*Note: The convergence of constraints across 18 orders of magnitude in mass (Earth to galaxy) motivates $\rho_T$ as a candidate saturation scale.*

### Paper Structure

The remainder of this paper is organized as follows:

- **Section 2:** Terrestrial calibration via GNSS atomic clock correlations, establishing $L_c \approx 4200$ km → $\rho_T \approx 20$ g/cm³.

- **Section 3:** Galactic consistency test via SPARC rotation curves, confirming $M^{1/3}$ scaling.

- **Section 4:** Screening hierarchy analysis, providing a candidate screening account for why precision GR tests remain satisfied despite scalar-field structure.

- **Section 5:** Physical context for $\rho_T$, including a dimensional identity bridge preserving WEP, a condensed-matter EOS coincidence, and open derivation targets.

- **Section 6:** Universal Scaling as a Cross-Scale Consistency Test, synthesizing constraints across all regimes.

- **Section 7:** Milky Way dark-matter onset at $R_{\text{DM}} \approx 3$ kpc.

- **Section 8:** Discussion of dark matter as phantom mass and cosmological implications.

- **Section 9:** Conclusion and astrophysical applications (including RBH-1, companion paper).

This multi-scale convergence identifies $\rho_T \approx 20$ g/cm³ as a candidate GNSS-anchored saturation parameter (with explicit systematic uncertainty), enabling sharply constrained astrophysical applications such as the RBH-1 runaway black hole candidate (Smawfield 2025h, Paper 7).

*Evidence hierarchy.* Within the evidence hierarchy of the TEP series, the GNSS clock analyses (Papers 1–3) provide the primary empirical input for the terrestrial correlation scale; the present paper tests the cross-scale consequences of conditionally identifying that scale with $R_T(M_\oplus)$. It is a cross-scale normalization test, not an independent primary proof of the TEP framework.

## 2. Terrestrial Calibration: GNSS Atomic Clock Correlations

### Distance-Structured Correlations in Atomic Clocks

The first empirical constraint on $\rho_T$ comes from terrestrial atomic clock networks. Multi-center analysis of GNSS (Global Navigation Satellite System) clock products reveals distance-structured correlations with a characteristic length scale $L_c \approx 4200$ km for Earth's mass. Operationally, $L_c$ denotes the best-fit exponential decay length recovered by fitting clock-residual covariance versus inter-station separation (see Appendix A). Recent analysis of GNSS products (Smawfield 2025b, Paper 1; Smawfield 2025c, Paper 2) reports this correlation across multiple processing centers.

In this corpus, the terrestrial scale $L_c$ is operationally identified with the projected covariance scale associated with the geometric Temporal Topology scale $R_T(M_\oplus)$, not with the in-medium Compton wavelength $\lambda_C(\rho)$ of a candidate completion. The empirical claim is that distance-structured clock correlations decay on the geometric scale $R_T(M)$ of the source mass, rather than on a Yukawa range set by local ambient density.

These correlations exhibit seven independent signatures:

- **Distance structure:** Correlation amplitude scales with inter-station separation.

- **Directional anisotropy:** Stronger correlations along Earth's orbital velocity vector.

- **Orbital velocity coupling:** Annual modulation consistent with $v_\oplus \approx 30$ km/s.

- **CMB-frame alignment:** Dipole structure aligns with CMB rest frame.

- **Multi-center consistency:** CODE, IGS, and ESA products show consistent characteristic scales and directional structure.

- **25-year stability:** Correlation structure persists across 1999–2024 (Smawfield 2025c, Paper 2).

- **Raw RINEX validation:** Pattern survives in unprocessed Single Point Positioning (SPP) analysis, constraining processing artifacts (Smawfield 2025d, Paper 3).

### Operational Calibration of $\rho_T$ from $L_c$

The characteristic length $L_c$ is used as the observed projected covariance scale associated with $R_T(M_\oplus)$. For a spherical mass distribution, the saturation radius is:

\begin{equation} \label{eq:rt_definition} R_T = \left( \frac{3M}{4\pi \rho_T} \right)^{1/3}. \end{equation}

Solving for $\rho_T$:

\begin{equation} \label{eq:rho_t_definition} \rho_T = \frac{3M}{4\pi R_T^3}. \end{equation}

Substituting $M_\oplus \approx 6 \times 10^{27}$ g and $L_c \approx 4200$ km $\approx 4.2 \times 10^8$ cm:

\begin{equation} \label{eq:rho_t_earth} \rho_T = \frac{3 \times (6 \times 10^{27}\,\text{g})}{4\pi \times (4.2 \times 10^8\,\text{cm})^3} \approx 20\,\text{g/cm}^3. \end{equation}

### Systematic Uncertainties

The primary uncertainty is the length scale determination. Three distinct contributions are reported: (i) the 25-year CODE per-measurement statistical uncertainty ($4{,}201 \pm 1{,}967$ km, $\pm 47\%$; Paper 2); (ii) the inter-center systematic spread ($3{,}330$–$4{,}549$ km, $\sim \pm 12\%$; Paper 1), with the caveat that centers share largely overlapping underlying data so this spread is not an independent ensemble; and (iii) the $\mathcal{O}(1)$ prefactor in the transfer sketch (Section 2), treated as a model-defining choice. The adopted operational value is $L_c = 4{,}200$ km with $\pm 500$ km ($\pm 12\%$) reflecting the inter-center range. The $\pm 47\%$ per-measurement statistical uncertainty from the single-center CODE fit is retained as a conservative upper bound on measurement-to-measurement scatter, but the $\pm 12\%$ inter-center spread is preferred as the operational systematic because it reflects the actual range of scales recovered by independent processing pipelines applied to the same underlying data, and the consensus value (4,200 km) lies within that range. The operational mapping from $L_c$ to the projected covariance scale associated with $R_T(M_\oplus)$, together with the use of total Earth mass, are model-defining choices rather than additional empirical uncertainties.

Propagating the length-scale uncertainty alone:

\begin{equation} \label{eq:rho_t_uncertainty} \rho_T = 20 \pm 7 \text{ g/cm}^3 \quad (35\% \text{ systematic}). \end{equation}

This range lies in the same condensed-matter density regime where electron degeneracy becomes dynamically relevant to the equation of state (Section 5.3) and provides the calibration scale for all subsequent tests. The key point is that the $M^{1/3}$ structural form is independent of $\rho_T$; only the normalization changes. Even at the extremes ($L_c = 3{,}700$ km $\rightarrow \rho_T \approx 30$ g/cm³, or $L_c = 4{,}700$ km $\rightarrow \rho_T \approx 14$ g/cm³), the cross-scale consistency tests remain viable because the same $\rho_T$ value must describe all systems simultaneously.

### Discrimination from Geophysical Signals

A critical objection to long-range clock correlations is the presence of tropospheric and ionospheric delays. However, spectral power analysis reveals distinct scale separation:

- **Tropospheric covariance:** Decays at ~100–500 km (weather systems)

- **Ionospheric covariance:** Decays at ~500–2000 km (TEC structures)

- **Scalar field signature:** Persists as a covariance floor at ~4200 km

The 4200 km feature shows no counterpart in ionospheric or tropospheric delay products. Furthermore, the signal aligns with the CMB rest frame (reported at 3.8σ significance in the companion analysis)—a directional dependence absent in atmospheric models. Power spectral analysis of the GPS-only clean baseline confirms this as a persistent background covariance floor, not a geophysical artifact; multi-constellation validation shows consistent signatures.

### Null Tests and Robustness

The GNSS correlation pattern survives multiple null tests:

- **Geophysical coupling:** Correlations persist after removing tidal, atmospheric, and ionospheric signals.

- **Orbital mechanics:** Pattern is independent of satellite constellation geometry.

- **Processing artifacts:** Raw RINEX analysis (Paper 3) constrains "black box" concerns.

- **Temporal stability:** 25-year consistency (Paper 2) disfavors transient instrumental effects.

The convergence of multi-center, multi-decade, and raw-data analyses identifies $L_c \approx 4200$ km as the canonical terrestrial screening length from the 25-year GNSS baseline (Papers 1–2), independent of theoretical interpretation. All cross-regime calibration and forward models in this paper adopt that value. Paper 14 provides an independent MGEX held-out verification on a ~1 yr combined-clock span (recovered scale $\approx 1{,}396 \pm 90$ km, $R^2 \approx 0.49$), confirming signal presence on a shorter baseline and different product; it is cited as verification only, not as the dimensional scale for $\rho_T$ extrapolation.

### Phenomenological Transfer Sketch: Mapping the Static Profile to Clock Covariance

A central question is how the static Earth-sourced Temporal Topology profile
produces the dynamic observable recovered in GNSS clock products: a spatial
covariance length $L_c \approx 4200$ km. The calibration does not require
GNSS satellites to orbit within a saturation shell, nor does it posit a
literal shell at $R_T(M_\oplus)$. Instead, $R_T(M_\oplus)$ defines the
geometric scale controlling the projected covariance kernel of the
Earth-sourced exterior clock-transport field. In this sense, $L_c$ is not an
arbitrary fitted length: it is the observed network projection of the same
mass-dependent scale used in the cross-regime $R_T \propto M^{1/3}$ construction.
This subsection gives an order-of-magnitude argument — a transfer sketch,
not a derivation — for this identification, together with the assumptions
that a full demonstration would have to close (Appendix B.2).

#### What $R_T(M_\oplus)$ is, and is not

$R_T(M) = (3M/4\pi\rho_T)^{1/3}$ is the radius a body of mass $M$ would have
at uniform density $\rho_T$. For Earth this is $\approx 4146$ km. It is a
*scaling variable*, not a location: because Earth lies in the partial-transition
regime, $R_T(M_\oplus)$ is used as a geometric scaling variable rather than
as the radius of a literal material shell (Appendix B.2). The question the
sketch addresses is what spatial scale governs the *exterior* field of such
a partially screened source.

#### The exterior profile scale

In the static exterior problem, the available length scales are the source
radius $R_\oplus$, the geometric saturation scale $R_T(M_\oplus)$, and the
ambient-medium Compton wavelength $\lambda_C(\rho_{\rm amb})$ (very long in
the near-vacuum exterior). Under the corpus's screening ontology the
transition in the screening function $S(\rho)$ is *gradual*, spanning
$\Delta\rho/\rho_T \sim \mathcal{O}(1)$ — a continuous Temporal Topology
with no thin shells (Paper 0, A4). For such a gradual transition, the
exterior field of the partially screened source carries no structure sharper
than the source scale itself, and the characteristic scale over which the
effective source charge accumulates is $R_T(M)$: radii enclosing mean
density above the transition range contribute suppressed source charge,
radii beyond contribute unsuppressed charge, and the changeover in the
*integrated* source — not in any local shell — occurs over a radial span of
order $R_T$. (Note that a *steep* transition would instead produce
thin-shell behaviour with structure on scales $\ll R_T$; the gradual
ontology is therefore not incidental to this argument but required by it.)
Both $R_\oplus = 6371$ km and $R_T(M_\oplus) = 4146$ km lie within a factor
$\sim 1.5$, so at sketch precision the exterior profile varies on the scale
of a few thousand kilometres, with $R_T(M)$ — not $R_\oplus$ — as the
mass-scaling variable the cross-scale law requires.

#### From a static profile to a time-series covariance

A deterministic static field produces constant clock offsets, which
clock-bias estimation absorbs; by itself it generates no time-series
covariance. The sketch therefore requires one explicit assumption:

**Modulated-sampling assumption.** The residual variance analysed in
Papers 1–3 arises from temporal modulation of each station's sampling of
the static exterior profile — diurnal rotation of the ground network
through the profile's anisotropy, satellite orbital motion, and
tropospheric/orientation geometry that multiplies the local conformal
rate $A(\bar{\phi}(\mathbf{x},t))$. Two clocks then acquire *correlated*
time variance to the extent that their modulated samples traverse the
same large-scale field structure.

Writing $\delta\tau_i(t) \approx \tau_0 (\beta_A/M_{\rm Pl})\,
\bar{\phi}(\mathbf{x}_i(t))$ for the modulated residual, the equal-time
network covariance $\langle \delta\tau_A\,\delta\tau_B\rangle_t$ inherits
its baseline dependence from the spatial structure of $\bar{\phi}$: stations
separated by $d$ small compared with the profile's variation scale sample
nearly the same field excursion and correlate strongly; stations separated
by $d$ large compared with that scale decorrelate. The decay length of the
network covariance is then set by the exterior profile scale,

\begin{equation} L_c \sim \mathcal{O}(1) \times R_T(M_\oplus), \end{equation}

with the $\mathcal{O}(1)$ factor determined by the unspecified completion
(profile shape, anisotropy spectrum, sampling geometry). This is the entire
quantitative content of the sketch: a coincidence of scales with an
undetermined order-unity prefactor, *not* a prediction of $L_c = R_T$ to
any stated precision.

#### Implications for the calibration

This sketch binds the clock observable ($L_c$) to the geometric scaling
variable ($R_T$) at the level of dimensional analysis under two explicit
assumptions: the gradual-transition ontology and the modulated-sampling
mechanism. It does not prove that the GNSS correlation is caused by the TEP
scalar — that remains the empirical hypothesis tested by null-model
exclusion (Papers 1–3) — and it does not by itself discriminate $R_T(M)$
from $R_\oplus$ as the controlling scale, since the two differ by only a
factor $\sim 1.5$ for Earth. What *would* discriminate them is the
mass-scaling test: $R_T \propto M^{1/3}$ while a source-radius scale does
not follow that law across the astrophysical systems of Sections 3–7. The
single-potential numerical demonstration that would replace this sketch
with a derivation — an explicit exterior profile and sampling kernel under
a specified completion, with $\Lambda$ fixed by the calibrated $\rho_T$ —
remains the open target recorded in Appendix B.2.

*For detailed methodology and validation, see Appendix A and Papers 1–3.*

## 3. The SPARC Galaxy Analysis: Phantom Mass as Unscreened Time-Field

Galaxies can be understood as "baryon-anchored" solitons—where the same
scalar field structure calibrated from terrestrial GNSS data is pinned to a
baryonic mass concentration. This section extends the test to galactic
scales, where the $M^{1/3}$ scaling is expected to govern the transition
between screened (Newtonian) and unscreened (phantom mass) regimes. The
SPARC database provides an ideal testbed: 175 disk galaxies with
high-quality rotation curves and well-constrained baryonic mass
distributions. The theoretical basis for the phantom mass interpretation is
developed in detail in Smawfield (2025e), which demonstrates how
differential proper-time accumulation in gravitational lensing can produce
apparent dark matter phenomenology.

### Phantom Mass Interpretation

In the TEP framework, the apparent "dark matter" observed in galaxy rotation
curves is identified as *Phantom Mass*—defined as the apparent mass
derived from temporal shear under the assumption of isochrony (Smawfield
2025e). It is not a particle species but a geometric manifestation of the
unscreened proper-time field. A critical distinction must be emphasized:
"Phantom Mass" is an *observational inference artifact* (analogous to
a refractive index effect), not a "ghost field" in the quantum field theory
sense. Ghost fields imply negative kinetic energy and Hamiltonian
instability; phantom mass carries no such pathology. It arises simply from
modeling a bi-metric spacetime with a single-metric prior—the "missing mass"
is the unmodeled temporal shear contribution.

The transition radius $R_{\rm trans}$ marks the boundary between two
regimes:

Inside $R_{\rm trans}$: The local density exceeds the critical screening
density ($\rho > \rho_{\rm trans}$). The time-field is screened, and
gravity follows Newtonian predictions based on visible baryonic matter.

Outside $R_{\rm trans}$: The density drops below $\rho_{\rm trans}$. The
time-field becomes unscreened, and the refractive proper-time gradient
produces an apparent gravitational excess—the "phantom mass"
conventionally attributed to dark matter.

If this interpretation is correct, the radius at which rotation curves
diverge from Newtonian prediction should scale as $R_{\rm DM} \propto M_{\rm
bar}^{1/3}$. While the fundamental core saturation occurs at the proximity
scale observationally proxied by $\rho_T \approx 20$ g/cm³, the halo transition
is governed by a much lower proximity scale, proxied by $\rho_{\rm trans}$,
characteristic of the diffuse field tail.

Invariant: $\rho_T$ denotes the universal saturation
density of the scalar sector that fixes the compact soliton scale via
$R_T \propto (M/\rho_T)^{1/3}$. By contrast, $\rho_{\rm trans}$ is
an *emergent* screening threshold for the onset of halo phenomenology
in diffuse environments; it is not treated as a second fundamental constant
and can depend on coupling, geometry, and baryonic structure.

### Theoretical Expectation from \(\rho_T\)

The TEP scaling law \(R_T \propto M^{1/3}\), anchored by the GNSS
calibration \(\rho_T \approx 20\) g/cm³, has a direct consequence for
galactic structure. In the TEP framework, the dark-matter onset radius
\(R_{\rm DM}\) is identified with the radius at which the local density
drops below the emergent halo threshold \(\rho_{\rm trans}\). Because
\(\rho_{\rm trans}\) is not an independent fundamental constant but an
emergent threshold tied to the same scalar potential that fixes
\(\rho_T\), dimensional analysis yields the same mass scaling:

\begin{equation} \label{eq:dm_scaling} R_{\rm DM} \propto \left(\frac{M_{\rm bar}}{\rho_{\rm trans}}\right)^{1/3}
\propto M_{\rm bar}^{1/3}. \end{equation}

Thus the TEP framework *expects* \(\alpha = 1/3\) for the DM onset
scaling. This is a theoretical consequence of the same saturation scale
that fixes the terrestrial correlation length, not a new free parameter.
The SPARC analysis below tests whether the empirical exponent
\(\alpha_{\rm SPARC}\) is consistent with this expectation.

The analysis was performed *post hoc*: the SPARC data were examined
after the TEP framework was established, so the comparison is a
consistency check rather than a blind prediction. A measured exponent
\(\alpha_{\rm SPARC} \neq 1/3\) (outside measurement uncertainty) would
challenge the framework, while agreement supports it.

### Methodology

The analysis proceeds as follows for each of the 175 SPARC galaxies:

Calculate the total baryonic mass: $M_{\rm bar} = M_* + 1.33 M_{\rm
HI}$, where $M_* = (M/L)_{3.6\mu} \times L_{3.6}$ with $(M/L)_{3.6\mu} =
0.5\,M_\odot/L_\odot$ for disk populations.

Compute the expected Newtonian rotation velocity: $V_{\rm bar}^2 =
V_{\rm gas}^2 + (M/L)_{\rm disk} V_{\rm disk}^2 + (M/L)_{\rm bulge}
V_{\rm bulge}^2$.

Identify the mass discrepancy onset radius $R_{\rm DM}$: the first
radius where $V_{\rm obs}/V_{\rm bar} > 1.3$.

Fit the power-law relation $R_{\rm DM} = k \cdot M_{\rm bar}^\alpha$
across the full sample.

### Results

Of the 175 SPARC galaxies, 167 yield valid mass discrepancy onset radii
spanning five decades in baryonic mass ($10^7$–$10^{12}\,M_\odot$).

#### Robustness to Onset Definition

To address concerns that the "onset radius" is a noise-sensitive functional,
a bootstrap analysis (1000 resamples) was performed across multiple
definition criteria. The scaling exponent $\alpha_{\rm SPARC}$ remains consistent with
the TEP expectation ($\alpha_{\rm SPARC} = 1/3$) across a wide range of velocity ratio
thresholds:

| Definition (Threshold) | Fitted Exponent ($\alpha_{\rm SPARC}$) | Consistency with 1/3 |
| --- | --- | --- |
| Loose ($V_{obs}/V_{bar} > 1.1$) | $0.274 \pm 0.036$ | $1.6\sigma$ |
| Fiducial ($V_{obs}/V_{bar} > 1.3$) | $0.372 \pm 0.041$ | $1.0\sigma$ |
| Strict ($V_{obs}/V_{bar} > 1.5$) | $0.413 \pm 0.045$ | $1.8\sigma$ |

While individual definitions shift the normalization, the slope consistently
clusters around the $M^{1/3}$ expectation. An alternative definition based on
acceleration thresholds ($a < a_0$) yields a steeper slope ($\alpha_{\rm SPARC} \approx
0.5$). This estimator is the $M^{1/3}$ vs. $M^{1/2}$ (TEP vs. MOND-like) discrimination
in miniature; the fact that the verdict is currently estimator-dependent is noted
as an open systematic. Because the acceleration-threshold method is degenerate
with the MOND acceleration scale itself, the kinematic divergence test is retained
as the fiducial estimator. This preference is theoretically motivated: TEP models
phantom mass as a geometric effect driven by temporal shear (a density/proximity
threshold) rather than an instantaneous acceleration threshold, so a purely kinematic
divergence estimator is intrinsically less biased toward MOND-like phenomenologies.

\begin{equation} \label{eq:dm_relation} R_{\rm DM} = k \cdot M_{\rm bar}^{\alpha_{\rm SPARC}} \end{equation}

\begin{equation} \label{eq:sparc_exponent} \alpha_{\rm SPARC} = 0.355 \pm 0.043 \text{ (stat)} \pm 0.07 \text{ (definition)} \end{equation}

The bootstrap-marginalized exponent deviates from the TEP expectation of $\alpha_{\rm SPARC} = 1/3
= 0.333$ by approximately 0.3$\sigma$ under the combined uncertainty—consistent within typical statistical tolerance. The correlation
coefficient $r \approx 0.6$ indicates a significant relationship.
Importantly, this result is robust: individual threshold choices yield
exponents ranging from 0.28 to 0.42, but the ensemble average converges near
1/3.

#### Cross-Regime Consistency

The significance of this result emerges when combined with the Temporal
Topology screening analysis (Section 4). The complete screening hierarchy spans
~15 orders of magnitude in density—from gas giants ($\rho \sim 1$ g/cm³) to
binary pulsars ($\rho \sim 10^{14}$ g/cm³)—while the fitted regression uses the
11 dense objects ($\rho > \rho_T$) spanning ~12 orders of magnitude
($\sim 10^2$–$10^{14}$ g/cm³). The consistency of this exponent across such a range is
striking, though the high $R^2$ is partly a consequence of the definitions
used (see Box 4.1). The result is consistent with the hypothesis that the
$M^{1/3}$ scaling reflects a genuine feature of the gravitational sector,
though systematic uncertainties in rotation curve fitting and baryonic
modeling remain.

Combined with the SPARC galaxy scaling ($\alpha_{\rm SPARC} = 0.355 \pm 0.043 \text{ (stat)} \pm 0.07 \text{ (definition)}$), the
RBH-1 crossover consistency (within $\sim 25\%$ combined uncertainty), and
the Milky Way dark-matter onset ($R_{\text{DM}} \approx 3$ kpc, observed $\sim$3–5 kpc), these results
suggest a universal density-limited structure may span 18 orders of
magnitude in mass—from Earth to galaxy.

### Physical Interpretation

The observed $M^{1/3}$ scaling admits a natural interpretation in terms of
density-limited screening. If the screening transition occurs at a
characteristic density $\rho_{\rm trans}$, dimensional analysis requires:

\begin{equation} \label{eq:trans_radius} R_{\rm trans} \sim \left( \frac{M}{\rho_{\rm trans}} \right)^{1/3}. \end{equation}

Fixing $\alpha_{\rm SPARC} = 1/3$ and fitting only the normalization yields $k \approx
7.9 \times 10^{-4}\,{\rm kpc}/M_\odot^{1/3}$. This normalization corresponds to an
emergent screening density $\rho_{\rm trans} \approx 0.5\,M_\odot/{\rm pc}^3$ ($\sim
3 \times 10^{-23}$ g/cm³) recovered from the fitted normalization $k$ via
Eq.~(\ref{eq:trans_radius}). A completion is conjectured to derive $\rho_{\rm trans}$
from the potential scale $\Lambda$ and coupling $\beta_A$, but this remains an open
target (see Formal Link below). This value is within an order of magnitude of
typical disk densities at the optical radius (0.01–0.1 $M_\odot/{\rm
pc}^3$), providing a physically reasonable anchor for the screening
mechanism.

*Note:* This galactic transition density $\rho_{\rm trans}$ is
distinct from the fundamental core saturation scale $\rho_T \approx 20$
g/cm³ calibrated in Section 2 and cross-checked in Section 5. The core density $\rho_T$ governs the compact
object size (soliton surface), while $\rho_{\rm trans}$ governs the onset of
the diffuse halo effect (screening radius). Both scales follow the same
$M^{1/3}$ structural form, consistent with the environment-dependent nature of the
theory across 15 orders of magnitude in density.

#### Formal Link: From Core Saturation to Galactic Transition

The galactic transition density $\rho_{\rm trans}$ is not an
independent fitting parameter. It emerges from the same scalar field
profile that fixes the core saturation scale $\rho_T$. In the TEP
framework (Box 6.5), the scalar field obeys

\begin{equation} \label{eq:profile_link} \nabla^2 \phi = V'(\phi) + \alpha(\phi)\rho, \end{equation}

with effective mass $m_{\rm eff}^2(\rho) = V''(\phi_{\rm min})$.
The field profile transitions from saturated ($\rho \sim \rho_T$,
$R_T$ scale) to screened ($\rho \gg \rho_T$, Compton wavelength
$\lambda_C \ll R_T$) to unscreened ($\rho \ll \rho_T$,
$\lambda_C \gg R_T$). The galactic halo onset occurs where the
local baryonic density drops to the value $\rho_{\rm trans}$ at
which the scalar-field-induced gravitational acceleration becomes
comparable to the baryonic contribution. Because the field profile
is governed by the same potential scale $\Lambda \equiv \rho_T^{1/4}$
and coupling $\beta_A$ that set the compact-object saturation,
$\rho_{\rm trans}$ is a derived consequence of these parameters,
not a free fit. The SPARC analysis measures the structural
exponent $\alpha_{\rm SPARC}$ and normalization $k$; the
corresponding $\rho_{\rm trans}$ is then recovered from $k$
via Eq.~(\ref{eq:trans_radius}) as a consistency check.

#### Connection to the MOND Acceleration Scale

The fitted screening density implies a characteristic transition
acceleration. At the transition radius, the gravitational acceleration is:

$g_{\rm TEP} = \frac{GM}{R_{\rm trans}^2} \approx 5 \times 10^{-10}\,{\rm
m/s}^2.$

This is within a factor of 4 of the MOND acceleration scale $a_0 \approx 1.2
\times 10^{-10}\,{\rm m/s}^2$. The near-coincidence suggests that the TEP
screening mechanism and the empirical MOND phenomenology may share a common
origin: both describe the transition from screened (Newtonian) to unscreened
(modified) gravity at a characteristic acceleration scale set by
cosmological boundary conditions.

### Connection to the Radial Acceleration Relation

The SPARC database is the source of the celebrated Radial Acceleration
Relation (RAR), which shows a tight empirical coupling between observed
acceleration $g_{\rm obs}$ and baryonic acceleration $g_{\rm bar}$ (McGaugh
et al. 2016). The RAR has been interpreted as evidence for modified gravity
(MOND) or as a consequence of dark matter halo profiles fine-tuned to
baryonic distributions.

The TEP framework offers a third interpretation: the RAR may emerge
from the environment-dependent screening of the proper-time field. At high
accelerations (high densities), screening is complete and $g_{\rm obs}
\approx g_{\rm bar}$. At low accelerations (low densities), the unscreened
time-field gradient contributes additional apparent gravity, producing the
characteristic upturn in the RAR. The $M^{1/3}$ scaling of the transition
radius is the spatial manifestation of this acceleration-dependent
screening.

![Example Rotation Curves](results/figures/figure_6_sparc_examples.png)

Figure 1: Representative SPARC rotation curves. Observed velocities
diverge from baryonic predictions near the TEP transition radius $R_{\rm
trans}$. Shaded regions indicate the screened (Newtonian) zone. The
agreement between $R_{\rm trans}$ and the observed onset $R_{\rm DM}$
supports the screening interpretation.

### Linking RBH-1 to Galactic Halos

The SPARC analysis identifies a cross-scale consistency between the
RBH-1 wake phenomenon and galactic dark matter halos:

RBH-1 represents a "naked saturation"—a Temporal Topology saturation not anchored
to a baryonic host, propagating freely through the intergalactic medium
and inducing star formation via its time-dilation wake.

Galactic halos represent "baryon-anchored solitons"—the same temporal
field structure, but centered on and shaped by the baryonic mass
distribution of the host galaxy.

Both phenomena are consistent with the same $M^{1/3}$ scaling law, calibrated from
terrestrial GNSS observations. The universality of this scaling across
planetary, stellar, galactic, and cosmological mass scales constitutes a
central theoretical consequence of the Temporal Equivalence Principle.

### Residual Analysis: Discriminating Baryonic Feedback vs Field Theory

The RMS scatter of 0.48 dex around the $M^{1/3}$ relation indicates
substantial galaxy-to-galaxy variation. A critical test distinguishes
whether this scatter arises from baryonic feedback processes (Standard
Model) or from the intrinsic field theory mechanism (TEP): correlate the
residuals with baryonic properties versus screening proxies.

#### Methodology

For each of the 167 SPARC galaxies with valid mass discrepancy radii, the
residual from the $M^{1/3}$ expectation is calculated as:

\begin{equation} \label{eq:residual} \text{Residual} = \log_{10}(R_{\rm DM} / R_{\rm exp}) \end{equation}

where $R_{\rm exp} = k \cdot M_{\rm bar}^{1/3}$ with $k = 7.86 \times 10^{-4}$ kpc/$M_\odot^{1/3}$. These
residuals are then correlated with:

*Baryonic properties.* Gas fraction ($f_{\rm gas}$), surface
brightness ($\Sigma$), and inclination, treated as proxies for feedback
efficiency, star formation history, and observational systematics.

*Screening proxy.* Central density ($\rho_{\rm central}$),
treated as a proxy for local screening strength independent of $R_{\rm
DM}$.

#### Results

The residual correlations reveal a clear pattern:

| Property | Correlation (r) | P-value | Interpretation |
| --- | --- | --- | --- |
| Gas Fraction | $-0.164$ | 0.034 | Weak, marginally significant |
| Surface Brightness | $-0.129$ | 0.096 | Weak, not significant |
| Inclination | $+0.297$ | 0.0001 | Moderate (observational systematic) |
| Central Density | $-0.108$ | 0.164 | Weak, not significant |

#### Interpretation

The maximum baryonic correlation is $|r| = 0.297$ (inclination), which
likely reflects observational systematics in rotation curve deprojection
rather than physical feedback. The gas fraction and surface brightness—the
primary tracers of baryonic feedback efficiency—show weak correlations ($|r|
< 0.2$) that are either marginally significant or non-significant.

The central density (screening proxy) shows no significant
correlation ($r = -0.108$, $p = 0.16$).
The weak correlation with central surface density is consistent
with (though does not by itself prove) a geometric origin: if baryonic feedback
were the dominant driver of the scatter in this particular estimator,
stronger covariance with surface brightness would be expected. The present
results remain compatible with a substantial contribution from measurement
uncertainty and geometric effects.

Summary: the residuals do not show strong correlations with gas fraction or
surface brightness, while inclination shows a moderate correlation that is
plausibly attributable to rotation-curve deprojection systematics. Central
density does not show a statistically significant correlation in this
analysis.

![SPARC Residual Analysis](results/figures/figure_7_sparc_residuals.png)

Figure 2: Residual analysis of the SPARC $M^{1/3}$ scaling. (a) Scaling
relation colored by residual. (b-e) Correlations with baryonic
properties and screening proxies. (f) Gaussian residual distribution
($\sigma = 0.48$ dex). (g) Correlations are weak in amplitude ($|r|<0.3$); inclination is statistically significant but plausibly reflects deprojection systematics rather than baryonic feedback physics.

### Scatter Quantification and Comparative Analysis

The observed scatter of $\sigma = 0.48$ dex around the $M^{1/3}$ relation is
substantial (corresponding to a factor of $\sim 3$ variation). Quantifying
the origin of this scatter is essential for distinguishing the TEP signal
from competing hypotheses.

#### Intrinsic vs. Measurement Scatter

Standard error budgeting suggests that a significant fraction of this
scatter is observational. Distance uncertainties in the SPARC sample
typically range from 10–20%, which propagate to a ~0.1–0.2 dex uncertainty
in $R_{\rm DM}$. Inclination corrections and non-circular motions contribute
additional noise. However, the residual scatter ($\sim 0.3$ dex) likely
reflects intrinsic variation in the galaxy population.

#### Comparison with MOND and LCDM+Feedback

MOND: The Modified Newtonian Dynamics framework typically
yields a tighter scatter ($\sim 0.13$ dex) in the Radial Acceleration
Relation (RAR). This is expected because MOND modifies gravity based on
instantaneous local acceleration, a continuous variable. In contrast, this
analysis extracts a single discrete parameter ($R_{\rm DM}$) from each
rotation curve, a method inherently more sensitive to local irregularities
and noise than the integrated RAR analysis. The larger scatter in $R_{\rm
DM}$ does not invalidate the mean scaling; rather, it indicates that the
"onset radius" is a noisy estimator of the underlying screening transition.

LCDM + Feedback (Null Hypothesis): The standard
cosmological model relies on baryonic feedback to explain rotation curve
diversity. In this scenario, the "dark matter onset" is not a fundamental
scale but an emergent property of halo assembly and feedback history.
Hydrodynamic simulations (e.g., NIHAO, FIRE) typically predict a scaling of
$R_{\rm DM} \propto M^{\alpha}$ with $\alpha \approx 0.3$–$0.4$, broadly
consistent with observation. However, they struggle to explain why the
normalization aligns with the specific density $\rho_T \approx 20$ g/cm³
derived from GNSS and terrestrial constraints. The "Null Hypothesis" (that
the $M^{1/3}$ scaling is a feedback coincidence) does not naturally explain the
cross-scale convergence with GNSS and compact-object physics.

#### Distinctive Cross-Scale Linkage

While MOND provides a tighter fit to galaxy kinematics alone, it offers
no explanation for the GNSS clock correlations ($L_c \approx 4200$ km).
LCDM+Feedback can accommodate the
galaxy scaling but treats the normalization as a free parameter,
offering no predictive link to other scales. TEP is distinctive in that
it links the normalization of the galactic relation ($k \approx 7.9
\times 10^{-4}$ kpc/$M_\odot^{1/3}$) to a GNSS-anchored normalization,
enabling a cross-scale consistency check without galaxy-by-galaxy
tuning, within the stated systematic uncertainty.

### Reproducibility

Full analysis code, data products, and methodology documentation are
available in the public repository:

*Repository.*
github.com/matthewsmawfield/TEP-UCD

*SPARC Scaling Analysis.*
`scripts/steps/step_4_sparc_analysis.py`

*SPARC Residual Analysis.*
`scripts/steps/step_7_sparc_residuals.py`

*Input Data.* `data/sparc/` (SPARC database tables)

#### Methodological Details

The "onset radius" $R_{\rm DM}$ is defined as the
*first radial bin* where $V_{\rm obs}/V_{\rm bar} > 1.3$ (fiducial
threshold). Specifically:

For each galaxy, the rotation curve is evaluated at the native SPARC
radial sampling (not interpolated).

The velocity ratio $V_{\rm obs}/V_{\rm bar}$ is computed at each radius.

$R_{\rm DM}$ is the radius of the first bin exceeding the threshold.

Galaxies with non-monotonic or noisy $V_{\rm obs}/V_{\rm bar}$ profiles
are retained; the "first crossing" definition is robust to subsequent
fluctuations.

Galaxies that never exceed the threshold (8 of 175) are excluded from
the fit.

The headline value $\alpha_{\rm SPARC}=0.355\pm0.043$ is the threshold-marginalized bootstrap estimate, obtained by resampling galaxies while marginalizing over the accepted onset definitions. The fiducial ($V_{\rm obs}/V_{\rm bar}>1.3$) definition alone gives $0.372\pm0.041$; the difference is retained as onset-definition systematic uncertainty. Varying the threshold from 1.1 to 1.5 yields exponents from 0.28 to 0.42, bracketing the headline estimate.

## 4. Temporal Topology Screening Analysis

If a scalar field permeates spacetime, why has it not been detected in
precision gravitational experiments? This section answers that question. The
key insight is that the scalar field's influence depends on the geometric
overlap of topological charge cores: in environments where charge cores are
tightly packed, the field is suppressed—a phenomenon called Temporal
Topology screening. Rather than invoking discrete thin-shell boundaries,
screening operates via the continuous spatial profile of the scalar field
(Temporal Topology). The tight geometric packing in deep potential wells
suppresses the local field gradient (Temporal Shear), ensuring short-range
fifth-force suppression while leaving the field light cosmologically.
General Relativity is recovered in the regimes where it has been tested most
stringently.

Screening in TEP is represented at the theory level by the environmental operator
*S*<sub>&Sigma;</sub>(*&Epsilon;*).
Quantities such as
&rho;<sub>T</sub>,
*R*<sub>T</sub>(*M*),
*S*<sub>&oplus;</sub>(*r*),
compactness &Phi;/*c*<sup>2</sup>,
local stellar density,
geometric coherence length,
and channel-specific response coefficients
are domain-specific projections of *&Epsilon;*,
not independent screening mechanisms
and not interchangeable universal thresholds.
Each is an observational transfer model
that parameterizes the same underlying operator
in a regime-appropriate form.

The screening factor $S$ quantifies this suppression. It is defined as the
ratio of the Temporal Topology saturation radius (where the scalar field saturates) to the
physical radius of the object:

\begin{equation} \label{eq:screening_factor} S = \frac{R_T}{R_{phys}}. \end{equation}

The quantity $S$ is used here as a geometric proxy for how deeply the
baryonic source is embedded within a saturated soliton configuration. When
$S \gg 1$, the physical object occupies a small region relative to the
saturated field scale; in this regime the phenomenology assumes that
non-linear response in the scalar sector suppresses gradients in the dense
interior, recovering GR to high precision in local dynamics. When $S \sim
1$, the system lies near the transition between strongly screened and weakly
screened behavior. When $S \ll 1$, the Temporal Topology saturation scale is smaller than the
object and the saturated region does not envelop the full baryonic
configuration; in this regime the model does not assume strong Temporal
Topology suppression a priori, and constraints must be assessed
case-by-case.

\begin{equation} \label{eq:ppn_suppression} \alpha_{\rm PPN}^{\rm eff}=\alpha_0 F(S,\rho,\Phi,\nabla\phi), \qquad F\rightarrow 0 \end{equation}

in the screened or source-charge-suppressed limit. The geometric factor $S$ is not itself the observable PPN coupling. Precision tests constrain $\alpha_{\rm PPN}^{\rm eff}$; $S$ is used here only as a proxy for the nonlinear suppression regime.

### 4.1 The White Dwarf Stress Test

White Dwarfs are the ideal stress test for two reasons. First, their
structure is determined by quantum mechanics (electron degeneracy pressure),
not thermal physics, so their mass-radius relation is calculable from first
principles. Second, their mass-radius scaling runs in the opposite direction
to the saturation scaling (see Figure 3):

White Dwarf (Chandrasekhar): $R_{WD} \propto M^{-1/3}$
— heavier stars are smaller

Soliton (TEP): $R_{sol} \propto M^{+1/3}$ — heavier
fields are larger

![White Dwarf Screening Mechanism](results/figures/figure_3_wd_screening.png)

Figure 3: The "Scissors" Effect. The defining visual signature is the
divergence of scales: as mass increases, the white dwarf physical radius
shrinks ($R_{WD} \propto M^{-1/3}$) while the saturation radius grows
($R_{sol} \propto M^{1/3}$). This divergence drives the screening factor
$S = R_T/R_{\rm phys}$ to large values ($S \gg 1$), recovering
General Relativity within the star.

These opposing scalings create a "scissors" pattern. As mass increases, the saturation radius grows while the physical radius shrinks—the screening factor
diverges. This is the signature of Temporal Topology flattening.

For Sirius B ($M \approx 1.02 M_{\odot}$, $\rho \approx 2.4 \times 10^6$
g/cm³):

\begin{equation} \label{eq:jupiter_radius} R_{phys} \approx 5{,}800 \text{ km} \quad \text{(observed)} \end{equation}

\begin{equation} \label{eq:jupiter_soliton} R_{sol} \approx 289{,}000 \text{ km} \quad \text{(TEP expectation)} \end{equation}

\begin{equation} \label{eq:jupiter_screening} \text{Screening Factor} = \frac{R_T}{R_{phys}} \approx 50\times. \end{equation}

With a mean density \(\rho \approx 2.4 \times 10^6\) g/cm³ far exceeding the
saturation threshold \(\rho_T \approx 20\) g/cm³, the saturation field extends
50 times beyond the physical surface of the star. The dense baryonic matter is
deeply embedded within the scalar field's saturation core, consistent with strong
screening under the stated model assumptions. Within the star, GR dynamics are recovered; the Keplerian mass
measured from binary orbital motion is the true baryonic mass.

### 4.2 The Empirical Screening Law

Extending this analysis across 26 astrophysical objects (planets, brown
dwarfs, main sequence stars, white dwarfs, neutron stars, and binary
pulsars) yields a compact summary of how the defined screening factor varies
with density. Restricting the regression fit to the 11 dense objects
($\rho > \rho_T$, excluding planets and black holes) gives:

\begin{equation} \label{eq:screening_law} S \propto \rho^{\beta_{\rm scr}}, \qquad \beta_{\rm scr} = 0.334 \quad (R^2 = 0.99995). \end{equation}

The exponent $\beta_{\rm scr} = 0.334$ is statistically indistinguishable from 1/3. Under the
stated definitions, this is the expected scaling: if $R_T \propto
M^{1/3}$ and $R_{phys} \propto (M/\rho)^{1/3}$, then:

\begin{equation} \label{eq:screening_derivation} S = \frac{R_T}{R_{phys}} = \frac{M^{1/3}}{M^{1/3}/\rho^{1/3}} =
\rho^{1/3}. \end{equation}

#### Interpretation of the Screening Law

It is important to recognize that the regression $S \propto
\rho^{0.334}$ is primarily a consistency check, not independent evidence
for the scaling law. Since $S \equiv R_T/R_{\text{phys}}$ and
the radii are defined by mass-density relations, the slope $\sim 1/3$ is
algebraically expected given the model assumptions.

The value of this analysis is not to prove the scaling 'ab initio', but
to demonstrate that a *single* saturation scale $\rho_T$ yields
a consistent screening hierarchy across 15 orders of magnitude in
density ($R^2 \approx 1$) without requiring regime-dependent
adjustments. The high $R^2$ confirms internal consistency; it does not
constitute independent confirmation of TEP.

Furthermore, the extreme screening factors observed in binary pulsars
($S \sim 29{,}000$) are consistent with a strongly non-linear suppression
mechanism in the scalar sector. The empirical screening hierarchy
($S$ vs $\rho$) derives from the canonical Temporal Topology mechanism
(Paper 0), where screening operates via the continuous spatial profile
of the scalar field governed by non-linear superposition of field
gradients (Temporal Shear). Box 6.5 (Section 6) sketches a candidate
route from the canonical action to soliton-like saturation, showing how
the interplay between the kinetic term and saturation potential could generate
the characteristic $M^{1/3}$ scaling and screening behavior. The detailed
dynamical derivation is not required for the empirical hierarchy used here.

The empirical screening law is a direct consequence of the $M^{1/3}$ soliton
scaling, providing cross-regime consistency under a single $\rho_T$ rather
than independent confirmation (see Note on Interpretation above).

### 4.3 Complete Screening Hierarchy

![Screening Hierarchy Across 15 Orders of Magnitude](results/figures/figure_4_screening_hierarchy.png)

Figure 4: The Screening Hierarchy. Screening factor $S$ plotted against
density for 26 astrophysical objects. The empirical power law $S \propto
\rho^{0.334}$ explains why dense objects (binary pulsars) are screened
($S \gg 1$) while diffuse systems (galaxies) are weakly screened ($S \ll
1$).

The complete hierarchy of screening factors across all object classes is
visualized in Figure 4 and tabulated below.

| Object Class | Density (g/cm³) | Screening | Physical Meaning |
| --- | --- | --- | --- |
| Gas Giants | 0.7 – 1.6 | 0.3 – 0.4× | $R_T$ smaller than physical radius; scalar contribution expected to be
small and/or below current constraints |
| Main Sequence Stars | 0.6 – 57 | 0.3 – 1.3× | Mixed regime; scalar and baryonic scales comparable |
| Rocky Planets (Earth) | 3.3 – 5.5 | 0.56 – 0.66× | Saturation scale comparable to object radius; GNSS probes this boundary |
| Brown Dwarfs | ~100 | ~1.7× | Just above $\rho_T$; screening onset begins |
| White Dwarfs | $\sim 4 \times 10^5$ – $2.5 \times 10^6$ | 27 – 50× | Baryonic star embedded within saturation scale; Keplerian mass = baryonic mass |
| Neutron Stars | $10^{14}$ | 26,800× | Scalar contribution less than 0.004%; pure GR dynamics |
| Binary Pulsars | $10^{14}$ | 29,500× | GR verified to 0.2%; Nobel Prize 1993 |

Table 4.1: Complete screening hierarchy across astrophysical object
classes. The screening factor increases monotonically with density,
consistent with a Temporal Topology screening hierarchy under the stated
definitions. Binary pulsars provide the strongest validation, with GR
verified to 0.2% precision at screening factors exceeding 29,500×.

### 4.4 Precision GR Tests: Explained, Not Violated

A critical question for any modified gravity theory is: why do precision
tests of General Relativity show no deviation? Temporal Topology provides
the answer: the suppression of Temporal Shear (vanishing field gradient)
reduces the screened PPN scalar charge $\alpha_{\rm PPN}^{\rm eff}$ in dense environments. The five most precise
tests of GR all occur in regimes where screening is operative:

| Test | Observable | Precision | Screening Factor | Status |
| --- | --- | --- | --- | --- |
| Lunar Laser Ranging | Nordtvedt effect | $10^{-13}$ | 0.56× | Calibration boundary |
| Cassini Conjunction | Shapiro delay | $2 \times 10^{-5}$ | 0.42× | Scalar sub-dominant |
| MESSENGER | Perihelion precession | $3 \times 10^{-4}$ | 0.65× | Calibration boundary |
| Hulse-Taylor Pulsar | GW emission | 0.2% | 29,500× | Completely screened |
| Double Pulsar | 7 PPN tests | 0.05% | 26,400× | Completely screened |

Table 4.2: Precision GR tests and their screening factors. All tests
are consistent with GR; TEP explains this via environment-dependent
screening rather than requiring the scalar sector to be absent.

The Hulse-Taylor binary pulsar is particularly significant. Its orbital
decay matches the GR prediction for gravitational wave emission to 0.2%
precision—a result honored with the 1993 Nobel Prize (Taylor & Weisberg
1982). At a density of $\sim 10^{14}$ g/cm³, the screening factor reaches
29,500×. The scalar field contributes less than 0.003% to the orbital
dynamics. This is not a violation of GR tests; it is a requirement of
consistent screening.

### 4.5 Earth as the Calibration Anchor

The screening hierarchy reveals why Earth is the natural calibration
point—and why GNSS clocks can detect what neutron stars cannot.

At $\rho \approx 5.5$ g/cm³, Earth sits just below the critical proximity
scale, proxied by $\rho_T \approx 20$ g/cm³. The screening factor is 0.66×,
meaning the saturation radius (4,200 km) is comparable to the physical radius
(6,371 km). Earth occupies the narrow window where:

**The scalar field is not screened:** Unlike neutron stars
(S = 26,800×), Earth's density is low enough that the soliton extends to
observable scales.

**The soliton is not diffuse:** Unlike gas giants (S =
0.3×), Earth's density is high enough that the soliton concentrates
within the planet's volume.

GNSS satellites orbit at $\sim$20,000 km altitude, sampling clock transport
through the exterior field region associated with the terrestrial Temporal Topology
profile. The measured covariance length $L_c \approx 4200$ km is interpreted as a
characteristic correlation/saturation scale of the Earth-sourced field, not as the
orbital radius of the satellite network. The observed clock correlation
length is therefore not an arbitrary parameter—it is the
characteristic scale where the scalar field's gradient becomes steep enough
to produce measurable timing correlations.

Within this phenomenology, the same saturation scale is expected to govern
both terrestrial clock correlations and compact-object structure. The
terrestrial scale provides an empirical calibration point, and the RBH-1
crossover provides an astrophysical consistency test under the same
parameter choice.

### 4.6 The Critical Proximity Scale

The empirical data suggest a critical proximity scale, observationally
proxied by the density $\rho_T \approx 20$ g/cm³,
corresponding to approximately 4× Earth's mean density. This value has
coincidence-level significance: it lies near the onset of electron degeneracy, where
the equation of state transitions from thermal to quantum pressure support.
This coincidence motivates treating the 10–30 g/cm³ regime as a useful condensed-matter reference scale, while the mechanism setting $\rho_T$ remains an open derivation target.

**Principle:**

$\rho_T \approx 20$ g/cm³ is a saturation scale of the temporal-field topology, not a local on/off switch. Systems approach the GR-like limit when the observable shear/source-charge sector is suppressed, $\mathcal{S}_\Sigma \to 0$, which depends on source structure, environment, and boundary conditions, not $\rho > \rho_T$ alone. The empirical screening hierarchy $S \propto \rho^{0.334}$ (§4.2) is a smooth power-law across 15 orders of magnitude in density, not a step function at $\rho_T$.

#### Box 4.1: Summary — Three Cross-Regime Consistency Tests

The screening analysis provides three cross-regime consistency tests of
the TEP framework under a single $\rho_T$:

**The RBH-1 Crossover:** The saturation radius is
consistent with the Schwarzschild radius at $M \approx 10^7
M_{\odot}$ within combined uncertainties ($R_T/R_S \approx 1.3$). This
provides a tightly constrained cross-regime consistency check under
the fixed $\rho_T$ calibration.

**The Screening Exponent:** The empirical scaling $S
\propto \rho^{\beta_{\rm scr}}$ with $\beta_{\rm scr} = 0.334$ is fitted to the 11 dense objects
($\rho > \rho_T$) spanning ~12 orders of magnitude in density, with $R^2 = 0.99995$. The exponent 1/3 is a
direct consequence of the $M^{1/3}$ saturation scaling. The high $R^2$
partly reflects the definitional structure of $S$ (the $\rho^{1/3}$
dependence is algebraically favoured once $R_T(M)$ enters the
construction); the substantive content is the absence of large deviations,
not the correlation coefficient — see Appendix B.4.

**GR Test Consistency:** All five precision tests of
General Relativity are consistent with screening factors that suppress
scalar contributions below current measurement limits.

The TEP framework does not violate established physics. It extends the
metric structure to include a conformal sector that becomes observable
only in specific density regimes—such as those probed by GNSS atomic
clocks and cosmological soliton-scale systems.

## 5. Physical Constraints on the Proximity Scale

A natural question arises: why does the saturation scale take the
observational proxy value \(\rho_T \approx 20\) g/cm³? This section argues
that the underlying proximity scale is not treated as an arbitrary fit
parameter: it is empirically calibrated, while additional coincidence-level
reference scales and open derivation targets are recorded for completeness.
The GNSS coherence length \(L_c\) is a derived quantity; the fundamental
parameter is the proximity scale itself, of which density is an observable
proxy.

### 5.1 \(\rho_T\) as the Conjectured Fundamental Parameter, \(L_c\) as Derived

The saturation scaling law \(R_T \propto M^{1/3}\)—where a soliton interpretation is one candidate microscopic realization—implies a constant
characteristic density:

\begin{equation} \label{eq:rho_t_constant} \rho_T = \frac{M}{\frac{4}{3}\pi R_T^3} = \text{constant}. \end{equation}

For any object of mass \(M\), the saturation radius is determined by:

\begin{equation} \label{eq:rt_mass} R_T(M) = \left(\frac{3M}{4\pi \rho_T}\right)^{1/3}. \end{equation}

The terrestrial coherence length \(L_c \approx 4200\) km is therefore not an
independently tuned galaxy-scale parameter; operationally, it is the observed
GNSS covariance scale used to calibrate the projected Earth-scale realization
of \(R_T(M_\oplus)\). This relation is inverted in Section 2 to calibrate
\(\rho_T\) from the measured \(L_c\) for Earth:

\begin{equation} \label{eq:lc_earth} L_{c,\oplus} = \left(\frac{3 M_\oplus}{4\pi \rho_T}\right)^{1/3} \approx
4200 \text{ km} \quad \text{for } \rho_T \approx 19\text{–}20 \text{
g/cm}^3. \end{equation}

The question thus reduces to: what physical considerations constrain \(\rho_T\)?

### 5.2 The Dimensional Identity Bridge

A natural concern is whether the proximity of $\rho_T \approx 20$ g/cm³ to the density of terrestrial condensed matter implies a composition-dependent coupling that would violate the Weak Equivalence Principle. Under universal conformal coupling, the scalar sector is sourced by the bulk trace $T = -\rho$, independent of microscopic composition. The saturation scale $\rho_T$ must therefore be a property of the vacuum potential $V(\phi)$, not an emergent property of the local matter.

The electron Fermi wavelength does not cause the scalar field to saturate. If it did, the field would require a direct, particle-specific coupling to the electron number density $n_e$, making the field exert a different force on iron than on hydrogen at the same bulk density. This would introduce a gross WEP violation, already constrained by experiments such as the MICROSCOPE satellite.

Instead, $\rho_T$ is treated as a fundamental constant of the vacuum potential:

\begin{equation} \label{eq:vacuum_scale} \rho_T \equiv \Lambda^4 \approx \alpha^{4/3} m_e^4 \quad \text{(natural units)}. \end{equation}

Because this is a property of the scalar potential itself, the dimensional identity does not introduce composition-dependent coupling; WEP protection then follows from the universal matter coupling and the screened effective scalar charge. The field universally couples to bulk $\rho$ and begins its continuous screening transition when local $\rho$ approaches this vacuum $\rho_T$.

The bulk density of Thomas-Fermi condensed matter is governed by Coulomb packing. The volume per atom scales with the Bohr radius $a_0 = (\alpha m_e)^{-1}$. In the Thomas-Fermi model, the effective atomic radius is $R_{\rm TF} \approx a_0 Z^{-1/3}$, giving a volume $V \approx \frac{4\pi}{3} a_0^3 Z^{-1}$ per atom. With nucleon mass $M \approx A m_p$:

\begin{equation} \label{eq:condensed_density_corrected} \rho_{\rm CM} \approx \frac{3}{4\pi} A Z \, m_p m_e^3 \alpha^3. \end{equation}

For representative rock-forming elements ($A/Z \approx 2$), the product $AZ$ ranges from $\sim 10^2$ (oxygen, $Z=8$) to $\sim 10^3$ (iron, $Z=26$). Evaluating with $m_p/m_e \approx 1836$ and $\alpha \approx 1/137.036$ gives $\rho_{\rm CM} \sim 10^2$–$10^3$ g/cm$^3$, one to two orders of magnitude above the calibrated $\rho_T \approx 20$ g/cm$^3$. The Thomas-Fermi Coulomb-packing estimate therefore does not independently converge on the saturation scale; it situates ordinary condensed matter in a denser regime. The relevant condensed-matter reference that lies near $\rho_T$ is the electron-degeneracy onset threshold (Section 5.3).

### 5.3 The Electron Degeneracy Threshold

Coincidence-level significance attaches to \(\rho_T \approx 20\) g/cm³:
this density lies in the condensed-matter regime where quantum pressure begins
to become dynamically relevant. The crossover from thermal and electrostatic
support to degeneracy-dominated support is gradual, with full degeneracy
domination occurring only at much higher (white-dwarf) densities.

| Density Regime | Dominant Physics | Examples |
| --- | --- | --- |
| \(\rho < 1\) g/cm³ | Gas pressure, thermal | Planets, main-sequence stars |
| \(\rho \sim 1\text{–}10\) g/cm³ | Coulomb/thermal dominated solids and liquids | Rocky planets |
| \(\rho \sim 10\text{–}10^2\) g/cm³ | Degeneracy onset (crossover) | Earth's core, brown dwarfs |
| \(\rho \gtrsim 10^4\) g/cm³ | Electron degeneracy dominated | White dwarfs |

Table 5.1: Equation of state transitions by density. The saturation
density \(\rho_T \approx 20\) g/cm³ lies near the onset of electron
degeneracy.

This coincidence provides a useful condensed-matter reference scale for
where quantum pressure becomes dynamically relevant to the equation of state.
Under the universal conformal-coupling benchmark used throughout this corpus,
the scalar sector is sourced primarily by the non-relativistic mass-density
term, with pressure corrections entering only as a small relativistic
correction in the condensed-matter regime. Electron degeneracy therefore does
not, by itself, provide a mechanism that sets \(\rho_T\), but it motivates the
\(\sim 10\text{–}30\) g/cm³ window as a natural coincidence-level cross-check.

### 5.4 Open Derivation Target: From Potential Scale to Transition Width

Box 6.5 adopts a benchmark completion in which the scalar potential introduces
a saturation scale \(\rho_T \equiv \Lambda^4\). Once \(\rho_T\) (equivalently \(\Lambda\))
is specified, the field equation determines a smooth crossover between weakly
and strongly screened regimes. Determining the detailed width and shape of this
crossover as a function of ambient density—particularly in the terrestrial
transition regime \(\rho \sim 1\text{–}30\) g/cm³—requires an explicit numerical
profile solution under a specified completion. This remains a target for future work.

Many-body condensed-matter physics (including the onset of electron degeneracy)
governs the baryonic equation of state and provides useful reference densities,
but under universal coupling it does not by itself select the saturation scale
\(\rho_T\). A complete theoretical account must instead explain both the value of
\(\rho_T\) and the mapping between the geometric saturation radius \(R_T(M)\) and the
observable correlation length in clock networks.

While Section 5.3 contextualizes *ρ*<sub>T</sub> within macroscopic condensed-matter
physics, establishing its fundamental origin requires examining the scalar coupling
scale at the vacuum level.

### 5.5 Dimensional Analysis: The Scalar Coupling Scale

Box 6.5 identifies the empirical saturation scale with the potential scale,
\(\rho_T c^2 \sim \Lambda^4\) (natural units). Inverting for \(\Lambda\) gives
\(\Lambda \approx 96.4\) keV for \(\rho_T = 20\) g/cm³.

\begin{equation} \label{eq:alpha_density} \rho_T \approx \alpha^{4/3}\,\frac{m_e^4 c^3}{\hbar^3} \approx 22.4 \text{ g/cm}^3. \end{equation}

This correspondence is recorded as a numerical identity: the exponent \(4/3\)
(equivalently \(\Lambda \sim \alpha^{1/3} m_e c^2\)) is not derived within the
present corpus and must be treated as an open target for theoretical work, not
as an independent support.

### 5.6 Status: Phenomenological Constraint with Theoretical Foundation

The present analysis treats \(\rho_T \approx 20\) g/cm³ as an empirically calibrated
parameter, while recording additional coincidence-level reference scales and open
derivation targets:

**Condensed-matter EOS coincidence:** \(\rho_T\) lies near
the density scale where electron degeneracy becomes dynamically relevant to the equation of state

**Numerical identity:** \(\rho_T \approx \alpha^{4/3} m_e^4 c^3/\hbar^3\),
with the \(4/3\) exponent an explicit open target

**Open derivation:** a first-principles account must explain why the saturation scale
\(\rho_T\) and the operational identification of \(L_c\) as the projected covariance scale
associated with \(R_T(M_\oplus)\) govern the observable
clock-correlation envelope

Its use in quantum or topological extensions as a microscopic cutoff or
defect-overlap scale is a conjectural extrapolation of the macroscopic
saturation proxy, not an independent consequence of the present paper.

The key point for the present manuscript is that \(\rho_T\) is not treated as a free
fit parameter: it is GNSS-anchored by clock correlations (Section 2) and then
carried, without retuning, into the cross-scale tests that follow.

#### Box 5.1: Summary — The Saturation Scale: One Calibration, Two Coincidences, One Open Derivation

Operationally, \(\rho_T\) is calibrated from the measured GNSS coherence length
\(L_c\) by treating \(L_c\) as the observed projected covariance scale associated with
\(R_T(M_\oplus)\), and then inverting
\(\rho_T = 3M_\oplus/(4\pi L_c^3)\). The framework’s conjecture—left as an open
derivation target—is that \(\rho_T\) is the fundamental saturation parameter and
that \(L_c\) follows as a derived geometric scale once \(\rho_T\) is fixed.
Two additional numerical correspondences are noted,
but they remain coincidence-level until a first-principles derivation exists:

- Condensed-matter EOS coincidence (electron degeneracy onset)

- Numerical identity \(\rho_T \approx \alpha^{4/3} m_e^4 c^3/\hbar^3\) (exponent underived)

In the present phenomenological calibration, the terrestrial coherence length
is treated as the observed projected covariance realization of the Earth-scale
\(R_T(M_\oplus)\) geometry; deriving this mapping from a specified scalar
potential remains the open target. Both the condensed-matter coincidence and
the numerical identity are contingent on the transfer-sketch \(\mathcal{O}(1)\)
prefactor being close to unity; a prefactor of \(\sim\)2 would shift \(\rho_T\) to
\(\sim\)2.5 g/cm³, evaporating the coincidences. Operationally, GNSS observations
provide the calibration by which \(\rho_T\) is inferred in the most accessible
laboratory: Earth's gravitational field.

## 6. The Universal Density Constraint

A key test is whether the Temporal Topology saturation radius \(R_T\) yields quantitative
expectations for otherwise free scales. Under a soliton interpretation, RBH-1 would correspond to a
gravitational soliton; however, the saturation radius itself is the primary observable, constrained by
the universal density \(\rho_T\) rather than introduced as an adjustable parameter.

### The Universal Density Hypothesis

The Temporal Topology framework tests the empirical claim that a candidate
saturation scale $\rho_T \approx 20$ g/cm³ governs
compact-object structure across mass scales. A soliton interpretation provides one candidate
microscopic realization of this saturation physics. This hypothesis is testable
via two independent windows:

**Planetary Scale:** The Earth-scale Temporal Topology parameter
$R_T(M_\oplus)$ is calibrated through the observed GNSS projected
covariance length ($L_c \approx 4200$ km), serving as the empirical anchor.

**Cosmological Scale:** The extrapolated Temporal Topology saturation radius for
RBH-1 ($M \approx 10^7 M_\odot$) can be compared to the Schwarzschild
diameter, placing the object near the crossover mass where horizon and
soliton interpretations are maximally degenerate.

The central question is whether a single density parameter, validated
cosmologically, remains universal at galactic scales.

### The Saturation Scale as Fundamental Parameter

If the scalar sector saturates at a critical energy density $\rho_T$,
dimensional analysis requires a universal mass–radius relation:

\begin{equation} \label{eq:universal_scaling} R_T(M) = \left( \frac{3M}{4\pi \rho_T} \right)^{1/3}. \end{equation}

#### Box 6.1: The Theoretical Priority of $\rho_T$

A potential misreading of this framework is that "GPS clock noise
determines black hole physics." This inverts the logical and historical
order of the analysis.

**Calibration (Terrestrial):** The GNSS analysis
(Smawfield 2025b) serves as an empirical anchor. The
detection of a correlation length $L_c \approx 4200$ km in atomic
clock data provides an empirical estimate of the saturation scale, conditional on treating $L_c$ as the projected covariance scale associated with $R_T(M_\oplus)$.

**Application (Cosmological):** The resulting $M^{1/3}$
scaling is therefore not an extrapolation of clock noise, but a test
of whether this terrestrially-validated constant
remains universal at galactic scales.

The alignment of the RBH-1 crossover scale is thus an independent
consistency check, not a fitted result.

#### Box 6.2: Null Hypothesis and Look-Elsewhere Effect

A potential concern is that apparent cross-scale correspondences could
arise from post-hoc alignment rather than predictive structure. To
address this possibility, the prior probability of simultaneous
agreement across multiple reference scales is estimated under an
explicit null hypothesis.

Null hypothesis: assume $\rho_T$ is drawn uniformly from the range
$10^{-3}$–$10^{6}$ g/cm³ (spanning interstellar gas to nuclear density).
The GNSS constraint ($R_T(M_\oplus)$ within 50% of 4200 km) permits
$\rho_T$ in the range $\sim$6–160 g/cm³ ($\approx$1.4 dex). The RBH-1
constraint ($R_T(10^7 M_\odot)$ within 50% of $2R_S$) permits
$\rho_T$ in the range $\sim$7–180 g/cm³ ($\approx$1.4 dex). The
overlap is $\sim$7–160 g/cm³ ($\approx$1.4 dex). Under a uniform
logarithmic prior spanning 9 dex, the probability of a randomly
drawn $\rho_T$ satisfying both constraints simultaneously is
$P \approx 1.4/9 \approx 0.15$ ($\sim$15%). While not overwhelmingly
small, this probability is reduced further when the
condensed-matter coincidence (Section 5.3) is included as a third
independent cross-check. Given the GNSS calibration and the
condensed-matter coincidence noted in Section 5, the RBH-1
correspondence provides an independent consistency check of the same underlying
parameter. The convergence of two physically distinct systems—geodesy and cosmology—on the
same density scale across 15 orders of magnitude is not readily explained by independent tuning,
but the universality hypothesis remains
falsifiable: any future measurement that robustly requires a substantially
different $\rho_T$ would exclude this framework.

Independence caveat: the GNSS calibration is currently validated within
the author’s research program (Smawfield 2025b,c,d). Independent
replication by other groups is required before the terrestrial
constraint can be treated as established. The cosmological
inputs referenced here are derived from external published data (van Dokkum et al. 2025).

The GNSS-derived value $\rho_T \approx 19\text{–}20$ g/cm³
(Section 2) lies in the same condensed-matter density regime where electron degeneracy becomes dynamically relevant to the equation of state (Section 5).
The central question is whether this same density scale holds for RBH-1.

#### Box 6.3: Robustness of the GNSS Calibration (Systematics Check)

The terrestrial calibration length $L_c \approx 4200$ km is an
extraordinary claim requiring careful exclusion of geodetic
systematics. The following tests support its physical origin (full
methodology in Smawfield 2025b,c,d):

**Multi-Center Verification:** The correlation
structure persists across independent clock solutions from CODE,
ESA, and IGS (1999–2024), disfavoring software-specific
processing artifacts (e.g., Bernese vs. GIPSY). The recovered $L_c$
spans $\sim$3,330–4,549 km across centers (a $\sim$23\% spread).

**Null Tests:** (i) Randomizing satellite epochs
destroys the correlation ($r^2 < 0.01$, $N = 1000$ shuffles); (ii)
shuffling clock residuals within each satellite eliminates the
spatial structure; (iii) replacing real data with white noise yields
no coherent scale. All three nulls are satisfied at $>5\sigma$.

**Scale Separation:** The 4200 km scale is distinct
from tropospheric correlation lengths (~100–500 km) and orbital
period harmonics (half-sidereal, ~12 hr). Power spectral analysis
shows the 4200 km feature as a persistent background covariance
floor in the GPS-only clean baseline, with no counterpart in
ionospheric or tropospheric delay products; multi-constellation validation shows consistent signatures.

**Dataset:** IGS final clock products (CLK files),
30-second sampling, 1999–2024. Preprocessing: removal of
satellite/receiver clock offsets, relativistic corrections (Sagnac,
gravitational redshift), and reference frame alignment to ITRF2020.

*Limitation:* The present analysis treats $L_c$ as an empirical
calibration parameter. A complete derivation from first
principles—linking the empirical $L_c$ to the geometric saturation radius $R_T(M)$ and to a candidate completion’s field-profile parameters—remains a target for future theoretical work.

#### Box 6.4: Derivation of the Scaling Law

The $R \propto M^{1/3}$ scaling is not an ad hoc ansatz but a direct
consequence of saturation in the scalar sector. Consider a scalar field
$\phi$ coupled to matter density $\rho$ with a potential $V(\phi)$ that
enforces a maximum gradient or energy density.

In the dense limit (compact objects), the scalar field profile
saturates to a core of constant effective energy density $\rho_T$. For a
self-gravitating configuration of total mass $M$, the volume of this
saturated core is constrained by mass conservation:

\begin{equation} \label{eq:mass_conservation} M \approx \frac{4}{3}\pi R_T^3 \rho_T. \end{equation}

Solving for the radius yields the characteristic scaling:

\begin{equation} \label{eq:scaling_law} R_T \propto \left(\frac{M}{\rho_T}\right)^{1/3}. \end{equation}

This relation describes the boundary of the saturated "soliton" region.
Outside this radius, the field decays, recovering Newtonian gravity
(Temporal Topology screening). Inside, the field is phase-locked,
modifying the effective metric (proper time). The "Universal Scaling" is
thus simply the statement that the vacuum has a maximum capacity to
support scalar gradients before saturating at $\rho_T$.

#### Box 6.5: Soliton Formation from the Canonical Action

To move beyond phenomenology, a candidate route from the canonical
Temporal Topology action (Paper 0) to soliton-like saturation is sketched. The action
above contains a canonical kinetic term $-\frac{1}{2}(\partial\phi)^2$
and a saturation potential $V(\phi)$ that prevents gradient divergence.
The interplay between these terms can generate soliton-like saturated configurations with
characteristic radius $R_T \propto M^{1/3}$.

**1. Field Equation and Equilibrium**

For a static, spherically symmetric configuration, the scalar field
equation is:

\begin{equation} \label{eq:field_equation} \nabla^2 \phi = V'(\phi) + \alpha(\phi)\rho \end{equation}

where $\alpha(\phi) \equiv d\ln A/d\phi$ and $\rho$ is the ambient
matter density. In the dense limit, the effective potential
$V_{\rm eff}(\phi; \rho) = V(\phi) + [A(\phi)-1]\rho$ develops a
minimum at $\phi_{\rm min}(\rho)$ with effective mass
$m_{\rm eff}^2(\rho) = V''(\phi_{\rm min})$.

**2. Saturation Mechanism**

For a potential of the form $V(\phi) = \Lambda^4[1 + (\Lambda/\phi)^n]$,
the field reaches an equilibrium value $\phi_{\rm min}(\rho) \propto
\rho^{-1/(n+1)}$. The energy density in the saturated core is:

\begin{equation} \label{eq:effective_density} \rho_{\rm eff} \sim V(\phi_{\rm min}) + \frac{1}{2}(\nabla\phi)^2
\sim \Lambda^4 \equiv \rho_T \approx 20 \text{ g/cm}^3. \end{equation}

This identifies the empirical saturation scale $\rho_T$ with the
potential scale $\Lambda$.

**3. Temporal Topology Screening**

In dense environments ($\rho \gg \rho_T$), the effective mass
$m_{\rm eff}(\rho)$ becomes large, suppressing field gradients
(Temporal Shear) within the Compton wavelength $\lambda_C =
1/m_{\rm eff}$. This creates the characteristic flattening of
the scalar field spatial profile—Temporal Topology screening—that
reconciles precision local tests with cosmological dynamics.
In the saturation regime ($\rho \sim \rho_T$), the Temporal Topology saturation radius
scales as $R_T = (3M/4\pi\rho_T)^{1/3}$, yielding the
observed $M^{1/3}$ dependence.

**4. Coupling and Stability**

Matter couples to the Jordan frame metric $\tilde{g}_{\mu\nu} =
A^2(\phi) g_{\mu\nu}$ with $A(\phi) = \exp(\beta_A \phi/M_{\text{Pl}})$.
Stability is ensured because the canonical kinetic term has the correct
sign for a physical scalar field, and the theory satisfies the Null
Energy Condition for physically realizing solutions.

### Testing the Density Constraint at Cosmological Scales

The saturation scale $\rho_T$ determined from terrestrial clocks is now
tested against RBH-1. Using the revised best-fit mass estimate of $M \approx
2 \times 10^7 M_{\odot}$ (van Dokkum et al. 2025), the predicted Temporal Topology
saturation radius is:

\begin{equation} \label{eq:rbh_radius} R_T = \left( \frac{3M}{4\pi \rho_T} \right)^{1/3} \approx 7.8
\times 10^7 \text{ km}. \end{equation}

The predicted Temporal Topology saturation radius can be compared directly to the Schwarzschild
radius for this mass:

Schwarzschild radius: $R_{\rm S} = 2GM/c^2 \approx 5.9
\times 10^7$ km

Predicted Temporal Topology saturation radius: $R_T =
(3M/4\pi\rho_T)^{1/3} \approx 7.8 \times 10^7$ km

Ratio: $R_T/R_{\rm S} \approx 1.3$

This order-unity correspondence ($R_T \approx 1.3 R_{\rm S}$) arises
because $\rho_T$ lies near the horizon-formation density
threshold in the underlying scalar sector. Within the combined uncertainties
from $M$ and $\rho_T$, which propagate to $\sim 25\%$ uncertainty in $R_T$, the two scales remain consistent at the factor-of-few level. This
places RBH-1 near a crossover regime where horizon and saturated-soliton
interpretations can be observationally degenerate.

This geometric match is the primary prediction of the scaling law. While the
velocity discontinuity ($\Delta v \sim 650$ km/s) provides a secondary
constraint on the *amplitude* of the screening (see Smawfield 2025h,
Paper 7), the *scale* of the object is set fundamentally by the
saturation scale $\rho_T$. The correspondence suggests that what is
conventionally identified as the event horizon scale in GR may correspond to
the saturation boundary of the scalar field in the TEP framework.

In the TEP interpretation, what is conventionally called a "black hole" at
this mass scale is modeled as a saturated soliton core, i.e., a region where
the conformal time-field approaches a maximum gradient set by $\rho_T$. In
this phenomenology, the characteristic radius is set by the saturation
scale rather than by horizon formation. The observed central dimming is
attributed to extreme time dilation (strong redshifting) rather than causal
disconnection.

A further structural implication follows from the different mass scalings.
The soliton relation predicts $R_T \propto M^{1/3}$, whereas the
Schwarzschild radius scales as $R_{\rm S} \propto M$. These opposing
scalings intersect at a unique crossover mass $M_{\times}$ where the two
radii coincide (see Figure D.1). Empirically, the terrestrial calibration
places this crossover near $M_{\times} \approx 10^7\,M_{\odot}$—precisely
where RBH-1 resides. Objects near $M_{\times}$ are expected to be maximally
degenerate between horizon and soliton interpretations, making RBH-1 an
unusually diagnostic system.

### Status of the Scaling Law

The relation $R \propto M^{1/3}$ is a density-limited scaling expected for
compact, self-bound field configurations whose cores approach a finite
saturation scale. Such behavior is familiar in non-topological solitons
and bosonic compact objects (Coleman 1985; Seidel & Suen 1991; Gleiser
1994; Hui et al. 2017).

The central claim is not that terrestrial clocks "predict" black hole sizes,
but that a fundamental density constant $\rho_T$ appears across scales. The
saturation scale is determined empirically from Earth—the most accessible
high-precision gravitational laboratory—and then tested for consistency at
cosmological scales. The present analysis should be read as a
phenomenological identification of an unexpected regularity: the same
density constraint that governs clock correlations in Earth's gravitational
field also governs the characteristic size of a $10^7 M_\odot$ compact
object.

### Dependency Structure and Constraint Isolation

A critical requirement for the robustness of this framework is to
demonstrate that the apparent convergence of $\rho_T$ is not a result of
circular reasoning. Specifically, it must be determined whether the
"independent" tests are truly distinct or if they all implicitly rely on the
same prior assumptions.

To resolve this, a constraint-isolation analysis is performed, isolating
each constraint to determine what value of $\rho_T$ it yields without input
from the others.

#### Test 1: Condensed-Matter Density Coincidence Isolation (Dropping GNSS)

If all GNSS data are discarded, condensed-matter equation-of-state transition scales
(Section 5.3) provide a coincidence-level density estimate.

Result: Condensed-matter transition considerations yield densities in the ~10–30
g/cm³ range, broadly consistent with the GNSS-derived value.

#### Test 2: Terrestrial Isolation (External Calibration Check)

If the condensed-matter coincidence is discarded and reliance is placed *solely* on
the observed GNSS correlation length $L_c \approx 4200$ km for Earth's mass
($M_\oplus$), what density is derived?

\begin{equation} \label{eq:gnss_rho} \rho_T(\text{GNSS}) = \frac{3 M_\oplus}{4\pi L_c^3} \approx 20 \pm 7
\text{ g/cm}^3. \end{equation}

Result: Terrestrial data independently recovers the same
density scale, consistent with the condensed-matter coincidence estimate but derived from a system 51
orders of magnitude more massive.

#### Test 3: Cosmological Isolation (Dropping Local Constraints)

If both condensed-matter considerations and terrestrial inputs are discarded and the density required
to place the RBH-1 Temporal Topology saturation radius at the horizon scale (the crossover
condition) is computed, the result is:

\begin{equation} \label{eq:rbh_rho} \rho_T(\text{RBH-1}) = \frac{3 M_{\text{RBH-1}}}{4\pi (1.3 R_S)^3} \approx
15\text{–}30 \text{ g/cm}^3. \end{equation}

Result: The cosmological constraint points to the same
narrow density window.

#### Conclusion on Circularity

The convergence is not a trivial post-hoc fit. Multiple
physically distinct systems—the Earth (geodesy)
and galactic black holes (cosmology)—are mutually
consistent with the same saturation scale $\rho_T \approx 20$ g/cm³.
The framework remains falsifiable: any future measurement that
robustly requires a substantially different $\rho_T$ would exclude this
universal-density soliton model as formulated here.

## 7. Milky Way Test: Dark Matter Onset

### Expected Onset Radius

The Milky Way provides a local test of the galactic $M^{1/3}$ scaling law established in Section 3. Unlike external galaxies where distance uncertainties dominate, the Milky Way offers high-precision kinematic data. This section tests whether the SPARC-calibrated relation predicts the radius at which the mass discrepancy first becomes significant in the Milky Way.

Using the SPARC-calibrated normalization ($k \approx 7.9 \times 10^{-4}$ kpc/$M_\odot^{1/3}$) and the Milky Way's total baryonic mass estimated at $M_{\text{bar}} \approx 6 \times 10^{10} M_\odot$ (Bland-Hawthorn & Gerhard 2016), the expected dark-matter onset radius is:

\begin{equation} \label{eq:milky_way_dm} R_{\text{DM}} = k \cdot M_{\text{bar}}^{1/3} \approx 7.9 \times 10^{-4} \cdot (6 \times 10^{10})^{1/3} \text{ kpc} \approx 3.1 \text{ kpc}. \end{equation}

This radius marks the scale where the baryonic rotation curve (bulge + disk) first falls below the observed circular velocity, signaling the onset of the phantom mass effect. It is the same quantity $R_{\text{DM}}$ that is measured directly in the SPARC ensemble; the outer Keplerian decline at larger radii is a separate phenomenon governed by the full halo profile, not by the onset scaling alone.

### Gaia DR3 and Local Observations

Gaia Data Release 3 (Gaia Collaboration 2023) provides precise proper motions and radial velocities for millions of stars, enabling construction of the Milky Way rotation curve to greater distances than previously possible. Recent analyses (e.g., Jiao et al. 2023) also report evidence for a Keplerian-like decline beginning near $R \approx 19$ kpc; this outer feature traces the halo edge and is distinct from the inner onset scale tested here.

Key observational features relevant to the $R_{\text{DM}}$ prediction:

- **Baryonic dominance ($R \lesssim 3$ kpc):** The bulge and inner disk dominate; the rotation curve rises steeply and is well described by visible matter alone (Sofue 2020).

- **Dark-matter onset ($R \sim 3$–$5$ kpc):** Beyond the peak of the baryonic contribution, the observed circular velocity ($v_{\text{circ}} \approx 220$ km/s) stays flat while the baryonic model predicts a decline. This is the Milky Way's equivalent of the SPARC $R_{\text{DM}}$ — the radius where the mass discrepancy first becomes significant.

- **Flat regime ($5 \lesssim R \lesssim 15$ kpc):** The rotation curve remains approximately flat, indicating substantial phantom mass contribution.

- **Keplerian decline ($R \gtrsim 19$ kpc):** At large radii the enclosed mass growth slows and $v_{\text{circ}}$ begins to decline. This outer feature is not predicted by the onset scaling $R_{\text{DM}} \propto M^{1/3}$.

### Uncertainty Analysis and Error Budget

| Parameter | Value | Uncertainty | Impact on $R_{\text{DM}}$ |
| --- | --- | --- | --- |
| **MW Baryonic Mass** | $6.0 \times 10^{10} M_\odot$ | $\pm 15\%$ ($0.9 \times 10^{10}$) | $\pm 5\%$ (scaling $M^{1/3}$) |
| **SPARC Constant ($k$)** | $7.9 \times 10^{-4}$ | $\pm 4\%$ (fit error) | $\pm 4\%$ |
| **Model Systematic** | - | $\pm 10\%$ (geometry) | $\pm 10\%$ |
| Total Expected $R_{\text{DM}}$ | 3.1 kpc | $\pm 0.4$ kpc ($\sim$14%) | - |
| **Observed Onset** | $\sim$3–5 kpc | $\pm 1$ kpc | - |

The expected value ($3.1 \pm 0.4$ kpc) and the observed dark-matter onset
scale ($\sim$3–5 kpc) are consistent within the combined uncertainty. Because
the SPARC-calibrated normalization $k$ is the sole empirical input to this
prediction, the Milky Way test is a SPARC-internal consistency check rather
than a GNSS cross-scale test, consistent with the Numerology Firewall
classification (Section 8).

### Comparison with Dark Matter Halo Models

#### NFW Halo vs. TEP Soliton

- **NFW Halo (CDM):** In standard CDM, the inner rotation curve profile depends on the halo concentration $c$ and virial mass $M_{\text{vir}}$, both of which are free parameters. The radius where baryons cease to dominate is not predicted a priori; it emerges from the halo assembly history and can vary significantly between cosmological simulations.

- **TEP Soliton:** The $M^{1/3}$ scaling yields the onset radius directly from the baryonic mass, with no Milky Way-specific tuning. The normalization is fixed by the SPARC ensemble ($k \approx 7.9 \times 10^{-4}$ kpc/$M_\odot^{1/3}$) and gives $R_{\text{DM}} \approx 3$ kpc for the Milky Way. The agreement with the observed transition at $\sim$3–5 kpc supports the universality of the scaling law.

### Future Refinements

Ongoing Gaia data releases will improve the precision of the inner rotation curve, potentially constraining the exact baryonic-to-dark-matter transition radius to $\sim$10% accuracy. Combined with improved Milky Way mass estimates from satellite kinematics and gravitational lensing, this could provide a sub-10% test of the $M^{1/3}$ onset scaling in the Milky Way.

## 8. Discussion: The Nature of the Dark Sector

### Synthesis of Multi-Scale Evidence

The central result of this work is the identification of a single density
scale, $\rho_T \approx 20$ g/cm³, that organizes gravitational anomalies
across 18 orders of magnitude in mass (Earth to galaxy). This scale, originally calibrated
from terrestrial GNSS atomic clock correlations ($L_c \approx 4200$ km), is
consistent with:

The dark matter onset radius in spiral galaxies ($R_{\text{DM}} \propto
M^{1/3}$).

The dark-matter onset radius of the Milky Way ($R_{\text{DM}} \approx 3$ kpc, observed $\sim$3–5 kpc).

The screening hierarchy in binary pulsars vs. galaxies ($S \propto
\rho^{1/3}$).

The convergence of these mutually reinforcing constraints suggests that $\rho_T$ is
not merely a fitting parameter for a specific system, but a candidate
universal parameter of the effective description of the dark sector, within
the stated uncertainties.

#### Claim Hierarchy and Falsification Scope

The empirical content of this work can be read at three distinct levels.
This separation is critical for interpreting the consequences of future
measurements.

**Level 1 (Empirical regularities):** The existence of
approximate $M^{1/3}$ scaling features in rotation-curve onset
radii, and the density-ordered screening hierarchy summarized by $S
= R_T/R_{\text{phys}}$.

**Level 2 (Universal-density soliton model):** The
hypothesis that a single saturation scale $\rho_T$ organizes these
regularities via $R_T(M) = (3M/4\pi\rho_T)^{1/3}$, with
GNSS providing an empirical calibration (subject to independent
replication) and cross-regime consistency tests (SPARC ensemble,
Milky Way, RBH-1 crossover).

**Level 3 (TEP microphysics):** The full dynamical
realization in which temporal shear and a conformal time field
$A(\phi) = \exp(\beta_A\phi/M_{\text{Pl}})$ generate the effective soliton/screening
phenomenology. Failures at Level 2 primarily constrain the soliton
implementation and/or calibration mapping, without necessarily
excluding the broader TEP framework.

### Numerology Firewall: Which Agreements Were Predicted?

The elegance of cross-scale agreement is vulnerable to the accusation of
"curve-fitting across scales." This section provides a candid assessment
of which constraints were predicted *a priori* and which were
identified *post hoc*.

| Comparison | Before/after GNSS calibration? | Free parameters? | Independent? | Weight |
| --- | --- | --- | --- | --- |
| **Earth GNSS $L_c$** | Calibration anchor | yes/no (fit to correlation) | no — defines scale | Calibration |
| **SPARC $M^{1/3}$ scaling** | After | Yes (normalization $k$) | Medium | Moderate |
| **DF2/DF4 UDGs** | After | Yes (extrapolation) | Weak | Illustrative |
| **Milky Way inner mass-discrepancy onset** | After | No new normalization; uses SPARC-calibrated $k$ | Medium | Local consistency check |
| **Wide binaries (if tested)** | Before/after | Yes | Stronger (if replicated) | **High if replicated** |

#### Honesty Principle: The Value of $\rho_T$

The value of $\rho_T$ should be treated as a candidate organizing
scale until at least one held-out mass regime is
predicted *before* analysis and recovered without refitting.
Current cross-scale consistency is motivating but not discriminating;
the $\rho^{1/3}$ hierarchy is a consistency relation induced by the
$R_T(M)$ construction, not an independent discriminator of microscopic
mechanism.

The per-constellation MGEX analysis (Paper 14 extension) provides a
concrete pre-registered discriminator: if GPS, Galileo, and GLONASS
individually recover $\sim$4,200 km while the combined product compresses
to $\sim$1,400 km, the MGEX discrepancy is resolved as a projection artifact.
If per-constellation analysis also yields $\sim$1,400 km, the
$L_c \leftrightarrow R_T(M_\oplus)$ identification fails as calibrated,
and the Level 1/2/3 claim hierarchy absorbs the hit exactly as designed.
Writing this test down before running it converts the corpus's largest
latent vulnerability into its sharpest falsifiable statement.

### Dark Matter as Phantom Mass

In the TEP framework, "dark matter" is reinterpreted not as a particle
species, but as "phantom mass"—an apparent excess inferred when a geometry
with temporal shear (two metrics) is analyzed under the assumption of
isochrony (single metric). The saturation scale $\rho_T$ represents the
scale at which the conformal factor $A(\phi) = \exp(\beta_A\phi/M_{\text{Pl}})$ enters a non-linear,
self-supporting phase (the soliton).

This explains the phenomenology of "cores" in dark matter halos. In standard
CDM simulations, halos tend to form cusps ($\rho \propto r^{-1}$), which are
often in tension with observations of flat cores in dwarf galaxies (the
core-cusp problem). In the soliton picture, the core is a distinct physical
state—a Bose-Einstein condensate-like configuration of the scalar field—that
naturally supports a flat density profile with a characteristic radius
$R_T$.

### Comparison with Alternative Frameworks

The TEP framework shares phenomenological features with both Modified
Newtonian Dynamics (MOND) and Fuzzy Dark Matter (FDM) but is distinguished
by its screening mechanism and cross-scale predictive power.

| Feature | MOND | Fuzzy DM (FDM) | TEP (Soliton) |
| --- | --- | --- | --- |
| **Physical Basis** | Modified Inertia/Gravity | Ultralight Boson ($m \sim 10^{-22}$ eV) | Conformal Time Field Saturation |
| **Fundamental Scale** | Acceleration $a_0 \approx 1.2 \times 10^{-10}$ m/s² | Particle Mass $m_\psi$ | Proximity Scale $\xi_T$ (obs. proxy $\rho_T \approx 20$ g/cm³) |
| **Galactic Scaling** | Fits RAR ($a_0$ is free parameter) | $M_c \propto M_h^{1/3}$ (Core-Halo) | $R \propto M^{1/3}$ (derived from $\rho_T$) |
| **Screening** | External Field Effect (EFE) | None (requires tuning) | Temporal Topology ($S \propto \rho^{1/3}$) |
| **Solar System** | Recovered via interpolation function | Unsuppressed (tension) | Screened ($S \sim 0.6$ for Earth, $>30k$ for Pulsars) |
| **Key Prediction** | Exact RAR | Solitonic Cores | GNSS Correlation Length + Screening Hierarchy |

#### TEP vs. MOND

MOND provides an excellent fit to galaxy rotation curves but offers no
explanation for the GNSS clock correlations. The critical distinction lies in
the triggering mechanism: MOND modifies kinematics at a specific
*acceleration* scale ($a_0 \approx 1.2 \times 10^{-10}$ m/s²), which
is a derived, environment-independent parameter that does not predict
where screening will occur in dense systems. By contrast, TEP is triggered
by a *density/proximity* scale ($\rho_T \approx 20$ g/cm³), enabling it
to explain anomalous dynamics in environments where MOND either makes no
prediction or fails—such as ultra-diffuse galaxies (where the low surface
brightness yields a different acceleration at fixed density) and the Milky
Way core (where the high baryonic density should suppress MOND-like effects
but does not in the data).

TEP recovers MOND-like phenomenology ($a_0 \sim G \Sigma_c$) as an
emergent property of the screening transition, while correctly predicting
the environment-dependent suppression required for precision GR tests.

#### TEP vs. FDM

Standard FDM predicts solitonic cores but fundamentally struggles to recover
the correct structural scales for both dwarf galaxies and the Milky Way simultaneously, requiring contradictory particle mass bounds. TEP naturally and uniquely links the core scale directly to the ambient saturation scale, generating the correct cross-scale hierarchy anchored directly by GNSS empirical measurements.

### Response-Coefficient Hierarchy

A natural criticism of the multi-channel TEP programme is that different
papers report different numerical coefficients. This is expected, and it is
a feature rather than a flaw. In the TEP framework, each measurement channel
carries an *observable response coefficient* \(\kappa_X\) that
absorbs stellar physics, environment, and projection geometry, not the bare
microscopic conformal coupling \(\beta_A\).

Paper 11 (TEP-H0) independently calibrates the weakly screened galactic-disk
response via Cepheid period–luminosity residuals, yielding
\(\kappa_{\rm Cep} \sim 10^6\) (dimensionless, in the appropriate
observable basis). Paper 10 (TEP-COS) measures the globular-cluster
pulsar response \(\kappa_{\rm MSP}^{\rm emp} \approx 3 \times 10^4\)
after dense-cluster geometric suppression. The two values are consistent
with the bare TEP geometric-factor estimate (\(\sim 10^6\)–\(10^7\)) only
after environmental transfer factors are included; they do not assert a
direct one-to-one equality of raw channel coefficients.

This hierarchy protects the programme from the criticism:
*"Why does each paper have a different coefficient?"*
The answer is that they are *observable response coefficients*, not
bare scalar couplings. Each channel measures a different projection of
\(S_\Sigma(\mathcal{E})\):

| Channel | Response coefficient | Environment | Projection |
| --- | --- | --- | --- |
| **GNSS clocks** | \(L_c \approx 4200\) km | Planetary crust/mantle | Radial covariance scale |
| **Cepheids (Paper 11)** | \(\kappa_{\rm Cep} \sim 10^6\) | Galactic disk (weakly screened) | Period–luminosity modulation |
| **Pulsars (Paper 10)** | \(\kappa_{\rm MSP} \sim 10^4\) | Globular cluster (strongly screened) | Spin-down excess (suppressed) |
| **Wide binaries (Paper 13)** | \(\alpha_{\rm sat} \approx 0.37\) | Galactic halo (unscreened) | Velocity-profile saturation |

The cross-paper comparison is therefore through a shared clock-response
structure, not raw equality of coefficients. This preserves the internal
consistency of the programme while acknowledging that each domain probes a
different facet of the same environmental operator.

### Resolving Galactic Anomalies

#### The Radial Acceleration Relation (RAR)

The SPARC analysis (Section 3) confirms that the onset of mass discrepancies
follows the baryonic mass distribution, a key feature of the Radial
Acceleration Relation (McGaugh et al. 2016). In TEP, this coupling is a
natural consequence of the assumed sourcing: the scalar field is sourced by
the trace of the energy-momentum tensor ($T^\mu_\mu$), so the "dark matter"
halo is directly anchored to the baryonic mass. This recovers the MOND-like
phenomenology of the RAR without modifying inertia or requiring new force
laws, through the non-linear response of the scalar sector.

#### Ultra-Diffuse Galaxies (DF2/DF4)

The "dark matter free" galaxies NGC 1052-DF2 and DF4 pose a challenge to
theories where dark matter and baryons are dynamically coupled. TEP resolves
this via "soliton stripping." Unlike a particulate halo, the scalar envelope
can be physically stripped from the baryons during high-velocity encounters
(as proposed by van Dokkum et al. 2022). The remaining baryonic component
would appear devoid of dark matter until it re-equilibrates a new (smaller)
soliton, a process governed by the field relaxation timescale.

### Systematic Uncertainties

The primary uncertainty in the global fit remains the GNSS calibration
length $L_c$. Three distinct contributions comprise this uncertainty: (i) the 25-year CODE
per-measurement statistical uncertainty ($4{,}201 \pm 1{,}967$ km, $\pm 47\%$;
Paper 2); (ii) the inter-center systematic spread ($3{,}330$–$4{,}549$ km,
$\sim \pm 12\%$; Paper 1), with the caveat that centers share largely overlapping
underlying data so this spread is not an independent ensemble; and (iii) the
$\mathcal{O}(1)$ prefactor in the transfer sketch (Section 2), treated as a
model-defining choice. The adopted operational value is $L_c = 4{,}200$ km
with $\pm 500$ km ($\pm 12\%$) reflecting the inter-center range. The
$\pm 47\%$ per-measurement statistical uncertainty from the single-center CODE
fit is retained as a conservative upper bound on measurement-to-measurement
scatter, but the $\pm 12\%$ inter-center spread is preferred as the operational
systematic because it reflects the actual range of scales recovered by
independent processing pipelines applied to the same underlying data, and the
consensus value (4,200 km) lies within that range. Propagating this alone gives
$\rho_T = 20 \pm 7$ g/cm³. The $M^{1/3}$ structural form is
independent of $\rho_T$; only the normalization changes. This uncertainty
is far smaller than the dynamic range over which the model is tested,
preserving the falsifiability of the scaling law.

### Predictions for High-Redshift JWST Observations

The TEP framework makes concrete predictions for the "Little Red
Dots"—compact, massive galaxies at $z > 5$ discovered by JWST. These objects
are a critical testing ground because they are compact enough for scalar-envelope effects to dominate their apparent structure, yet massive enough for JWST to resolve the predicted kpc-scale transition region.

#### Qualitative Predictions for High-Redshift JWST Observations

For a target galaxy with stellar mass $M_* = 10^{10} M_\odot$ (typical of
LRDs), the TEP model makes the following qualitative predictions:

#### 1. Soliton Envelope Scale

The fundamental core saturation density $\rho_T \approx 20$ g/cm³
governs the compact soliton surface, but the observable galactic
transition is set by the diffuse screening density $\rho_{\rm trans}
\approx 3 \times 10^{-23}$ g/cm³ (fitted from SPARC, Section 4). The
predicted emission extent therefore scales as $R_{\rm trans} \propto
M^{1/3}$ at the transition density, not at $\rho_T$. Precise
numerical predictions for high-redshift galaxies require
epoch-specific calibration of $\rho_{\rm trans}$; until then, the
model predicts that resolved gas tracers (e.g., H$\alpha$) should
reveal more extended kinematic structure than expected from a purely
stellar compact core.

#### 2. Velocity Dispersion

The dynamical support speed is set by the soliton potential at the
transition radius: $\sigma_v^2 \sim G M / R_{\rm trans}$. Because
$R_{\rm trans} \propto M^{1/3}$, the velocity dispersion scales as
$\sigma_v \propto M^{1/3}$. This predicts characteristic velocity
dispersions intermediate between the broad-line AGN regime
($>1000$ km/s) and the low-mass dwarf regime ($<50$ km/s). Precise
normalization awaits epoch-specific calibration of $\rho_{\rm trans}$.

#### Discriminant: Soliton vs. Supermassive Black Hole

A leading alternative explanation is that LRDs are reddened Active Galactic
Nuclei (AGNs) powered by supermassive black holes ($M_{\text{BH}} \sim
10^7$–$10^8 M_\odot$).

**AGN Hypothesis:** Emission should be point-like or
confined to the nuclear region ($< 1$ kpc). Broad lines reflect BLR
physics ($v > 1000$ km/s).

**TEP Soliton Hypothesis:** Emission should be extended
over the galactic transition scale (kpc-scale for typical LRD masses,
pending epoch-specific calibration of $\rho_{\rm trans}$). Broad lines
reflect the deep gravitational potential of the soliton well
($\sim$100–200 km/s).

Discriminant: Spatially resolve the H$\alpha$ emission.
Extended kpc-scale emission with velocity dispersions $\sim$100–200 km/s
would disfavor a purely nuclear AGN interpretation, while strongly compact
emission with $v > 1000$ km/s would favor an AGN-dominated explanation.
This constitutes a concrete observational discriminator for this
particular soliton-interpretation channel.

### Implications for Paper 7 (RBH-1)

The GNSS-calibrated $\rho_T$ identified here provides the working normalization for
the analysis of RBH-1 in the companion paper (Smawfield 2025h). By adopting
$\rho_T \approx 20$ g/cm³ (with stated uncertainty), the soliton
interpretation for RBH-1 becomes tightly constrained. The wake structure,
velocity jump, and thermal emission properties can then be compared against
the interaction of the wake with a soliton of characteristic radius $R
\approx 7.8 \times 10^7$ km. Deviations would primarily constrain the
object-specific soliton interpretation and/or the assumed mapping between
$\rho_T$ and the effective radius in this environment, rather than the
broader TEP framework.

### Cross-Domain Constraint Network

| Tier | Role | Channels | Relation to \(\rho_T\) |
| --- | --- | --- | --- |
| **A** | Direct / quasi-direct calibration | GNSS, SPARC/MW, RBH-1, condensed-matter EOS | Enters \(\rho_T\) normalization directly |
| **B** | Screening / response-transfer test | Pulsars (10), Cepheids (11), wide binaries (13), LLR (17), Earth flyby (15) | Constrains \(S_\Sigma(\mathcal{E})\) and transfer functions, not independent \(\rho_T\) |
| **C** | Phantom Mass / conformal transport | Lensing (4), multiply-imaged SNe (19), JWST (12) | Tests the mechanism; not a \(\rho_T\) calibration |
| **D** | Orthogonal architecture test | J0437 (16), hi\_class (18), C0 (26), TH (27), QF (23) | Tests deep TEP structure; does not calibrate \(\rho_T\) |

*Only Tier A contributes to the present \(\rho_T\) normalization. Tiers B–D constrain the environmental operator, response hierarchy, and theoretical viability of the same TEP architecture.*

The TEP programme now spans more than twenty papers across terrestrial,
galactic, compact-object, lensing, and cosmological domains. It is
essential to distinguish which of these constrain \(\rho_T\) directly
and which constrain the broader TEP architecture. The following tiered
classification makes this dependence explicit and prevents the
misreading that every paper independently confirms the same density
number.

#### Tier A — Direct or Quasi-Direct \(\rho_T\) Constraints

These channels calibrate the macroscopic saturation scale most directly:

**GNSS \(\lambda_T\) (Papers 1–3):** The primary
empirical anchor. The 4,200 km correlation length is conditionally
identified with the projected covariance scale associated with
\(R_T(M_\oplus)\), up to an order-unity transfer factor, yielding the
operational estimate \(\rho_T \approx 20\) g/cm³.

**SPARC/Milky Way onset scaling (this work):**
Quasi-direct scaling consistency. The \(M^{1/3}\) scaling of
dark-matter onset radii across 175 spiral galaxies is consistent with
the saturation-radius construction, but does not independently
calibrate \(\rho_T\). The Milky Way test uses the SPARC-calibrated
normalization and is therefore not an independent measurement.

**RBH-1 crossover (Paper 7):** The runaway-black-hole
candidate tests the extrapolation of \(R_T(M)\) to
\(M \sim 10^7 M_\odot\), with wake structure providing an
object-specific consistency test.

**Condensed-matter EOS coincidence (this work, Section 5):**
The proximity of \(\rho_T \approx 20\) g/cm³ to warm-dense-matter /
degeneracy-onset scales provides a coincidence-level physical
plausibility check, not an independent empirical calibration.

#### Tier B — Screening and Response-Transfer Constraints

These channels do not independently calibrate \(\rho_T\), but they
constrain the environmental operator \(S_\Sigma(\mathcal{E})\) and
the transfer functions that connect \(\rho_T\) to observables:

**Globular-cluster pulsars (Paper 10):** The suppressed
density scaling of spin-down excess (\(\Gamma \approx 0.39\) dex/dex,
vs. Newtonian \(0.75\)) constrains how dense environments screen the
temporal-field response. The observable coefficient
\(\kappa_{\rm MSP}\) is a downstream projection, not a direct
\(\rho_T\) measurement.

**Cepheids (Paper 11):** The Cepheid clock-bias response
\(\kappa_{\rm Cep} \sim 10^6\) in galactic disks bridges the
weakly screened regime. It agrees with \(\kappa_{\rm MSP}\) only
after environmental transfer factors are applied, demonstrating that
response coefficients are channel-specific projections.

**Wide binaries (Paper 13):** Gaia DR3 reveals a
weak-field screening transition at a few thousand AU with
environmental ordering (high-\(|Z|\) vs. midplane). This is an
independent *morphology* test of Temporal Shear recovery, not
an independent \(\rho_T\) calibration: the paper explicitly states
that imported \(\rho_T\) and \(g_{\rm TEP}\) enter only as
cross-scale consistency checks.

**Lunar Laser Ranging (Paper 17):** Reports a
compactness-dependent residual channel at high significance, but
remains at the "candidate residual" level pending integrator-level
ephemeris refits. Classified as a *guardrail*: it suggests
the local screened regime may still contain residual-channel
structure, but does not yet provide a completed \(\rho_T\) likelihood
term.

**Earth flyby (Paper 15):** Uses the UCD
\(\rho_T \approx 20\) g/cm³ as a *shared input*, not an
independent validation. Valuable as a downstream posterior predictive
test: if the terrestrial screening profile is correct, flyby anomalies
should organize under the restricted trajectory model.

#### Tier C — Phantom Mass / Conformal Transport Tests

These channels test the *mechanism* by which TEP generates apparent
mass discrepancies, without directly calibrating \(\rho_T\):

**Gravitational lensing (Paper 4):** The Phantom Mass
mechanism shows that standard lensing reconstructions implicitly
assume isochrony; relaxing this projects temporal depth onto the
spatial plane, degenerate with particulate dark matter. This is
*mechanistic support* for the TEP interpretation of the
dark sector, not a \(\rho_T\) calibration.

**Multiply-imaged supernovae (Paper 19):** SN Refsdal
provides directional evidence from blind model residuals for
conformal time-transport in the lensing sector. The amplitude kernel
remains phenomenological; included as a *sign test* for
Phantom Mass, pending derivation of the full transfer kernel.

**JWST high-redshift galaxies (Paper 12):** Tests
potential-depth response at \(z > 5\) using an external response
prior. The signal is not an independent \(\rho_T\) measurement;
it probes whether the same conformal architecture organizes
high-redshift compact galaxies.

#### Tier D — Orthogonal TEP-Architecture Tests

These papers do not calibrate \(\rho_T\) at all, but they test deep
structural claims of the TEP framework:

**J0437 scintillation (Paper 16):** Reports non-zero
phase-closure signal in millisecond pulsar scintillation, rejecting
standard scalar-delay ISM models (\(\psi = 0\)) at high confidence.
This tests *non-integrable time transport* / synchronization
holonomy, one of the deepest TEP claims. It does not calibrate
\(\rho_T\).

**hi_class cosmology (Paper 18):** Demonstrates that
the conformal temporal-transport geometry preserves CMB acoustic
structure, sound-horizon behaviour, and stable linear perturbations.
A *global consistency constraint*, not a \(\rho_T\) term.

**C0 supernova conformal redshift (Paper 26):** Shows
that Pantheon+ distance-redshift data do not uniquely require
primitive spatial expansion. Supports the temporal-transport
interpretation of cosmological redshift.

**TH temporal horizon (Paper 27):** The apparent
\(a_{\rm eff} \to 0\) limit becomes a temporal conformal boundary
rather than a physical singularity. Provides theoretical
closure for the late-time cosmology.

**QF quantum tangent limit (Paper 23):** Shows that
the screened quantum tangent limit is consistent with the TEP
architecture. A *theoretical compatibility* check, not empirical
\(\rho_T\) evidence.

**Summary:** Only Tier A enters the present
\(\rho_T\) normalization argument directly. Tiers B–D constrain
the environmental operator, response hierarchy, and theoretical
viability of the same TEP architecture. This tiered dependence is
stronger than a flat pile of evidence because it acknowledges
exactly where each paper contributes and where it does not.

## 9. Conclusion

### A Universal Organizing Parameter

This paper identifies the saturation scale $\rho_T \approx 20$ g/cm³ as a candidate organizing parameter for gravitational phenomena across planetary, galactic, compact-object, and RBH-scale regimes. The convergence of terrestrial GNSS timing structure, SPARC galaxy onset scaling, Milky Way inner mass-discrepancy structure, and the density-ordered screening hierarchy motivates the hypothesis that $\rho_T$ is a real physical scale in the temporal-field topology, not a system-specific fitting parameter. The screening hierarchy is not treated as an independent discovery of the $1/3$ exponent; rather, it demonstrates that the same $\rho_T$ coherently recovers GR-like behavior in dense systems while permitting unscreened scalar phenomenology in diffuse regimes.

### Key Results

The primary findings are:

- **GNSS Calibration:** Distance-structured correlations in atomic clocks yield $L_c \approx 4200$ km → $\rho_T \approx 20$ g/cm³. This calibration exhibits 25-year stability, multi-center consistency, and survives raw RINEX validation.

- **Galactic Scaling:** SPARC rotation curves yield $\alpha_{\rm SPARC} = 0.355 \pm 0.043 \text{ (stat)} \pm 0.07 \text{ (definition)}$, consistent with the $M^{1/3}$ expectation within $\sim$0.3$\sigma$.

- **Screening Hierarchy:** 26 objects spanning ~15 orders of magnitude in density reveal a consistency relation $S \propto \rho^{1/3}$, algebraically expected from the $R_T(M)$ construction, explaining why GR tests pass (binary pulsars: $S \sim 29{,}000$) while galactic dynamics are deeply unscreened ($S \sim 10^{-9}$ at $\rho \sim 10^{-24}$ g/cm³).

- **Milky Way Test:** The SPARC-calibrated $M^{1/3}$ relation predicts a dark-matter onset radius $R_{\text{DM}} \approx 3$ kpc for the Milky Way, consistent with the observed transition from baryonic to dark-matter-dominated rotation at $R \sim 3$–5 kpc, providing a local scale-consistency check.

### Dark Matter as Phantom Mass

The TEP framework reinterprets dark matter observations as violations of the isochrony axiom—the assumption that spatially separated clocks at the same gravitational potential tick at the same rate. When this axiom is relaxed, gravitational lensing (integrated time dilation) and dynamical mass (local time gradient) diverge, creating "phantom mass" without invoking non-baryonic particles.

This reinterpretation offers a candidate common account of several dark-sector puzzles:

- **Universal scaling:** The $M^{1/3}$ relation follows naturally from the saturation-radius construction.

- **Dark-matter-deficient UDGs:** Soliton stripping provides a possible explanation for DF2/DF4-like systems.

- **Core-cusp structure:** Saturated temporal-field cores offer a possible route to flat effective density profiles.

- **Baryonic Tully-Fisher behaviour:** The $M^{1/3}$ scaling may contribute to BTFR-like phenomenology, though a full BTFR derivation is not completed here.

### Astrophysical Applications

The GNSS-anchored value $\rho_T \approx 20$ g/cm³ enables specific applications for astrophysical systems. The companion paper (Smawfield 2025h, Paper 7) applies this calibration to the RBH-1 runaway black hole candidate, yielding a saturation radius $R_T \approx 7.8 \times 10^7$ km. This provides a testable extrapolation: if the RBH-1 mass is revised by a factor of 3, the expected radius would change by $\sim 44\%$, offering a clear consistency check with observations.

Future applications include:

- **JWST high-redshift galaxies:** "Little Red Dots" and massive quiescent galaxies at $z > 10$ may exhibit soliton signatures.

- **EHT polarimetry:** M87* and Sgr A* polarization patterns could reveal scalar field structure near event horizons.

- **Gravitational waves:** Binary black hole mergers may produce scalar radiation detectable by LISA.

- **Strong lensing time delays:** Phantom mass contributions could resolve the Hubble tension.

### Theoretical Implications

The convergence of terrestrial, compact, galactic, and cosmological constraints on a single density scale suggests a fundamental organization principle governing gravitational phenomenology across many orders of magnitude in mass. This organization is formalized in the Temporal Equivalence Principle (Smawfield 2025a, Paper 0), which posits that gravitational phenomena arise from a conformal time field $\phi(x^\mu)$ coupled to matter via the action:

\begin{equation} \label{eq:tep_action_conclusion} S = \int d^4x \sqrt{-g} \left[ \frac{M_{\text{Pl}}^2}{2} R - \frac{1}{2}(\partial \phi)^2 - V(\phi) \right] + S_m[\tilde{g}_{\mu\nu}] \end{equation}

where $\tilde{g}_{\mu\nu} = A^2(\phi) g_{\mu\nu}$ and $A(\phi) = \exp(\beta_A\phi/M_{\text{Pl}})$ in the late-time conformal limit.

The saturation scale $\rho_T$ emerges as the scale at which the scalar field $\phi$ reaches its self-interaction threshold, saturating the field. Under a soliton interpretation, this would correspond to soliton formation; regardless of microscopic realization, this provides a natural explanation for the $M^{1/3}$ scaling and the phantom mass phenomenon.

### Systematic Uncertainties and Robustness

The primary uncertainty is the GNSS length scale determination. Three distinct contributions are reported: (i) the 25-year CODE per-measurement statistical uncertainty ($4{,}201 \pm 1{,}967$ km, $\pm 47\%$; Paper 2); (ii) the inter-center systematic spread ($3{,}330$–$4{,}549$ km, $\sim \pm 12\%$; Paper 1), with the caveat that centers share underlying data so this spread is not an independent ensemble; and (iii) the $\mathcal{O}(1)$ prefactor in the transfer sketch (Section 2), treated as a model-defining choice. The adopted operational value is $L_c = 4{,}200$ km with $\pm 500$ km ($\pm 12\%$) reflecting the inter-center range. The $\pm 47\%$ per-measurement statistical uncertainty from the single-center CODE fit is retained as a conservative upper bound on measurement-to-measurement scatter, but the $\pm 12\%$ inter-center spread is preferred as the operational systematic because it reflects the actual range of scales recovered by independent processing pipelines applied to the same underlying data, and the consensus value (4,200 km) lies within that range. Propagating this alone gives $\rho_T = 20 \pm 7$ g/cm³. The $M^{1/3}$ structural form is independent of $\rho_T$; only the normalization changes. The present evidence consists of one primary terrestrial timing calibration, one galactic onset-scaling consistency test, one local SPARC-internal Milky Way check, and one algebraic screening-consistency hierarchy. These layers are mutually consistent, but they are not statistically independent measurements of $\rho_T$.

The robustness of the result is demonstrated by:

- **Multi-center consistency:** CODE, IGS, and ESA yield consistent GNSS correlation structure across independent processing centres.

- **Temporal stability:** 25-year consistency disfavors transient effects.

- **Raw data validation:** RINEX analysis constrains processing artifacts.

- **Cross-scale agreement:** Four tests spanning 18 orders of magnitude in mass.

### Future Directions

The identification of $\rho_T \approx 20$ g/cm³ as a candidate cross-scale saturation parameter opens several avenues for future research:

- **Precision tests:** Improved GNSS analysis (longer baselines, more stations) could tighten $L_c$ to $\sim$5% accuracy.

- **High-redshift galaxies:** JWST observations of $z > 10$ systems could test whether soliton formation affects early galaxy assembly.

- **Gravitational wave cosmology:** LISA could detect scalar radiation from binary black hole mergers, providing direct evidence for the time-field sector.

- **Laboratory tests:** Atomic interferometry experiments could search for temporal gradients at sub-meter scales.

### Concluding Remarks

This paper identifies $\rho_T \approx 20$ g/cm³ as a candidate macroscopic saturation/proximity scale in the TEP environmental-response hierarchy. GNSS clock correlations provide the primary terrestrial timing anchor for the Earth-scale covariance length. SPARC onset scaling, Milky Way structure, RBH-1 crossover behaviour, compact-object screening, and condensed-matter/EOS considerations then test whether the same scale organizes otherwise disconnected gravitational regularities without retuning. Channel-specific quantities such as $\rho_{\rm trans}$, $R_s$, $\kappa_{\rm Cep}$, $\kappa_{\rm MSP}$, and lensing response coefficients are not independent measurements of $\rho_T$, but downstream projections of $S_\Sigma(\mathcal{E})$.

This paradigm shift—from dark matter as substance to dark matter as temporal shear—provides a candidate unified organizing framework for the $M^{1/3}$ scaling, screening hierarchy, and phantom mass phenomenon. The GNSS-anchored value of $\rho_T$ enables specific testable applications, including the RBH-1 case study (Paper 7), positioning the Temporal Equivalence Principle as an empirically grounded candidate alternative to the particle dark matter paradigm.

## References

Abedi, J., Dykaar, H., & Afshordi, N. 2017, *Phys. Rev. D*, 96, 082004 (arXiv:1612.00266)

Allen, M. G., Groves, B. A., Dopita, M. A., Sutherland, R. S., & Kewley, L. J. 2008, *ApJS*, 178, 20 (arXiv:0805.0204)

Barro, G., et al. 2025, *From "The Cliff" to "Virgil": Mapping the Spectral Diversity of Little Red Dots with JWST/NIRSpec* (arXiv:2512.15853)

Bland-Hawthorn, J., & Gerhard, O. 2016, *ARA&A*, 54, 529 (Milky Way mass model)

Campanelli, M., Lousto, C. O., Zlochower, Y., & Merritt, D. 2007, *Phys. Rev. Lett.* (arXiv:gr-qc/0702133)

Cardoso, V., Hopper, S., Macedo, C. F. B., Palenzuela, C., & Pani, P. 2016, *Phys. Rev. D*, 94, 084031 (arXiv:1608.08637)

Chen, K., Li, Z., Inayoshi, K., & Ho, L. C. 2025, *ApJ Lett.* (DOI: 10.3847/2041-8213/ae1955; arXiv:2505.22600)

Colpi, M., & Dotti, M. 2011, *Adv. Astron.*, 2011, 1 (arXiv:0906.4339)

Colpi, M., Geppert, U., & Page, D. 2000, *ApJ*, 529, L29 (DOI: 10.1086/312448; arXiv:astro-ph/9912066)

Donato, F., Gentile, G., Salucci, P., et al. 2009, *MNRAS*, 397, 1169 (DOI: 10.1111/j.1365-2966.2009.15004.x; arXiv:0904.4054)

Delvecchio, I., et al. 2025, *A&A* (DOI: 10.1051/0004-6361/202557164; arXiv:2509.07100)

Event Horizon Telescope Collaboration. 2019, *ApJ Lett.*, 875, L1 (DOI: 10.3847/2041-8213/ab0ec7; M87* image)

Event Horizon Telescope Collaboration. 2021, *ApJ Lett.*, 910, L12 (DOI: 10.3847/2041-8213/abe71d; M87* polarization)

Event Horizon Telescope Collaboration. 2021, *ApJ Lett.*, 910, L13 (DOI: 10.3847/2041-8213/abe71e; EHT Paper VII)

Event Horizon Telescope Collaboration. 2022, *ApJ Lett.*, 930, L12 (DOI: 10.3847/2041-8213/ac6674; Sgr A* image)

Heeck, J., et al. 2021, *Phys. Rev. D*, 103, 115004 (arXiv:2009.08463)

Hinterbichler, K., & Khoury, J. 2010, *Phys. Rev. Lett.*, 104, 231301 (DOI: 10.1103/PhysRevLett.104.231301; arXiv:1001.4525)

Khoury, J., & Weltman, A. 2004, *Phys. Rev. Lett.*, 93, 171104 (DOI: 10.1103/PhysRevLett.93.171104; arXiv:astro-ph/0309300)

Hui, L., Ostriker, J. P., Tremaine, S., & Witten, E. 2017, *Phys. Rev. D*, 95, 043541 (DOI: 10.1103/PhysRevD.95.043541; arXiv:1610.08297)

Komossa, S. 2012, *Recoiling black holes: electromagnetic signatures, candidates, and astrophysical implications* (arXiv:1202.1977)

Coleman, S. 1985, *Nucl. Phys. B*, 262, 263 (DOI: 10.1016/0550-3213(85)90286-X)

Gleiser, M. 1994, *Phys. Rev. D*, 49, 2978 (DOI: 10.1103/PhysRevD.49.2978; arXiv:hep-ph/9308279)

Gaia Collaboration. 2023, *A&A*, 674, A1 (DOI: 10.1051/0004-6361/202243940; Gaia Data Release 3)

Gong, Y., Papantonopoulos, E., & Yi, Z. 2018, *Eur. Phys. J. C*, 78, 738 (DOI: 10.1140/epjc/s10052-018-6227-9; arXiv:1711.04102)

Gronke, M., & Oh, S. P. 2018, *MNRAS*, 480, L111 (arXiv:1806.02729)

Hviding, R. E., et al. 2025, *A&A* (DOI: 10.1051/0004-6361/202555816; arXiv:2506.05459)

Seidel, E., & Suen, W.-M. 1991, *Phys. Rev. Lett.*, 66, 1659 (DOI: 10.1103/PhysRevLett.66.1659)

Kusenko, A. 1997, *Phys. Lett. B*, 404, 285 (arXiv:hep-th/9704073)

Lelli, F., McGaugh, S. S., & Schombert, J. M. 2016, *AJ*, 152, 157 (DOI: 10.3847/0004-6256/152/5/157; SPARC Database)

McGaugh, S. S., Lelli, F., & Schombert, J. M. 2016, *Phys. Rev. Lett.*, 117, 201101 (DOI: 10.1103/PhysRevLett.117.201101; arXiv:1609.05917)

McGaugh, S. S., Schombert, J. M., Bothun, G. D., & de Blok, W. J. G. 2000, *ApJ Lett.*, 533, L99 (DOI: 10.1086/312628; arXiv:astro-ph/0003001)

Ogiya, G., & Nagai, D. 2023, *ApJ Lett.*, 958, L5 (arXiv:2309.09031)

Poggianti, B. M., et al. 2019, *ApJ*, 874, 140 (arXiv:1810.05164)

Rybicki, G. B., & Lightman, A. P. 1979, *Radiative Processes in Astrophysics* (Wiley-VCH)

Sanchez Almeida, J., Montes, M., & Trujillo, I. 2023, *A&A* (DOI: 10.1051/0004-6361/202346430; arXiv:2304.12344)

Schive, H.-Y., Chiueh, T., & Broadhurst, T. 2014, *Phys. Rev. Lett.*, 113, 261302 (DOI: 10.1103/PhysRevLett.113.261302; arXiv:1407.7762)

Sofue, Y. 2020, *PASJ*, 72, 81 (DOI: 10.1093/pasj/psaa065; arXiv:2006.15008)

Sutherland, R. S., & Dopita, M. A. 1993, *ApJS*, 88, 253 (DOI: 10.1086/191823)

TDCOSMO Collaboration. 2025, *A&A* (DOI: 10.1051/0004-6361/202555801; arXiv:2506.03023)

van Dokkum, P., et al. 2023, *ApJ Lett.*, 946, L50 (arXiv:2302.04888)

van Dokkum, P., et al. 2025, *JWST Confirmation of a Runaway Supermassive Black Hole via its Supersonic Bow Shock* (arXiv:2512.04166)

Taylor, J. H., & Weisberg, J. M. 1982, *ApJ*, 253, 908 (DOI: 10.1086/159690)

Vainshtein, A. I. 1972, *Phys. Lett. B*, 39, 393 (DOI: 10.1016/0370-2693(72)90147-5)

Weisberg, J. M., & Taylor, J. H. 2005, *Binary Radio Pulsars*, ASP Conf. Ser., 328, 25 (arXiv:astro-ph/0407149)

Westerweck, J., et al. 2018, *Phys. Rev. D*, 97, 124037 (arXiv:1712.09966)

Will, C. M. 2014, *Living Rev. Relativ.*, 17, 4 (DOI: 10.12942/lrr-2014-4; arXiv:1403.7377)

Williams, J. G., Turyshev, S. G., & Boggs, D. H. 2012, *Class. Quantum Grav.*, 29, 184004 (DOI: 10.1088/0264-9381/29/18/184004; arXiv:1203.2150)

Younes, G., et al. 2020, *ApJ Lett.*, 896, L42 (DOI: 10.3847/2041-8213/ab9a48; arXiv:2006.02814)

Younes, G., et al. 2022, *Nature Astronomy*, 7, 339 (arXiv:2210.11518)

Zhang, H., et al. 2025, *Polarization Images of Solitonic Boson Stars* (arXiv:2508.11992)

Zumalacárregui, M., & Bellini, E. 2018, *JCAP*, 2018, 054 (DOI: 10.1088/1475-7516/2018/10/054; arXiv:1705.05302; hi_class cosmology solver)

Pintore, F., et al. 2016, *MNRAS*, 458, 2088 (DOI: 10.1093/mnras/stw449; arXiv:1602.05950)

Ray, P. S., et al. 2019, *ApJ*, 879, 130 (DOI: 10.3847/1538-4357/ab24d8)

Tuo, Y. L., et al. 2024, *ApJ*, 966, 80 (DOI: 10.3847/1538-4357/ad2fb6; arXiv:2403.12137)

Kramer, M., et al. 2006, *Science*, 314, 97 (DOI: 10.1126/science.1132305)

van Dokkum, P., et al. 2018, *Nature*, 555, 629 (DOI: 10.1038/nature25767)

van Dokkum, P., et al. 2019, *ApJ Lett.*, 874, L5 (DOI: 10.3847/2041-8213/ab0d92)

van Dokkum, P., et al. 2022, *Nature*, 605, 435 (DOI: 10.1038/s41586-022-04665-6)

Jiao, Y., et al. 2023, *A&A*, 678, A208 (DOI: 10.1051/0004-6361/202347513; arXiv:2309.00048)

Archibald, R. F., et al. 2013, *Nature*, 497, 591 (DOI: 10.1038/nature12159)

Archibald, R. F., et al. 2017, *ApJ*, 829, L21 (DOI: 10.3847/2041-8205/829/1/L21; arXiv:1608.01007)

de Graaff, A., et al. 2025, *A&A* (arXiv:2503.01891)

Dib, R., & Kaspi, V. M. 2014, *ApJ*, 784, 37 (DOI: 10.1088/0004-637X/784/1/37; arXiv:1401.5738)

Nicolis, A., Rattazzi, R., & Trincherini, E. 2009, *Phys. Rev. D*, 79, 064036 (DOI: 10.1103/PhysRevD.79.064036; arXiv:0811.2197)

Olausen, S. A., & Kaspi, V. M. 2014, *ApJS*, 212, 6 (DOI: 10.1088/0067-0049/212/1/6; arXiv:1309.4167)

Şaşmaz Muş, S., et al. 2014, *MNRAS*, 440, 2916 (DOI: 10.1093/mnras/stu445; arXiv:1402.6054)

### TEP Research Series

Smawfield, M. L. (2025a). *Temporal Equivalence Principle: Dynamic Time & Emergent Light Speed*. Preprint v0.9 (Jakarta). Zenodo. DOI: [10.5281/zenodo.16921911](https://doi.org/10.5281/zenodo.16921911) (Paper 0)

Smawfield, M. L. (2025b). *Global Time Echoes: Distance-Structured Correlations in GNSS Clocks*. Preprint v0.26 (Jaipur). Zenodo. DOI: [10.5281/zenodo.17127229](https://doi.org/10.5281/zenodo.17127229) (Paper 1)

Smawfield, M. L. (2025c). *Global Time Echoes: 25-Year Analysis of CODE Precise Clock Products*. Preprint v0.19 (Cairo). Zenodo. DOI: [10.5281/zenodo.17517141](https://doi.org/10.5281/zenodo.17517141) (Paper 2)

Smawfield, M. L. (2025d). *Global Time Echoes: Raw RINEX Consistency Test*. Preprint v0.6 (Kathmandu). Zenodo. DOI: [10.5281/zenodo.17860166](https://doi.org/10.5281/zenodo.17860166) (Paper 3)

Smawfield, M. L. (2025e). *Temporal-Spatial Coupling in Gravitational Lensing: A Reinterpretation of Dark Matter Observations*. Preprint v0.6 (Tortola). Zenodo. DOI: [10.5281/zenodo.17982540](https://doi.org/10.5281/zenodo.17982540) (Paper 4)

Smawfield, M. L. (2025f). *Global Time Echoes: Empirical Synthesis*. Preprint v0.5 (Singapore). Zenodo. DOI: [10.5281/zenodo.18004832](https://doi.org/10.5281/zenodo.18004832) (Paper 5)

Smawfield, M. L. (2025g). *Temporal Topology Saturation Scale: Cross-Scale Consistency of ρ_T*. Preprint v0.7 (New Delhi). Zenodo. DOI: [10.5281/zenodo.18064365](https://doi.org/10.5281/zenodo.18064365) (Paper 6 — this work)

Smawfield, M. L. (2025h). *The Soliton Wake: Exploring RBH-1 as a Temporal Topology Candidate*. Preprint v0.3 (Blantyre). Zenodo. DOI: [10.5281/zenodo.18059250](https://doi.org/10.5281/zenodo.18059250) (Paper 7)

Smawfield, M. L. (2025). *Global Time Echoes: Optical-Domain Consistency Test via Satellite Laser Ranging*. Preprint v0.3 (Mombasa). Zenodo. DOI: [10.5281/zenodo.18064581](https://doi.org/10.5281/zenodo.18064581) (Paper 8)

Smawfield, M. L. (2025). *What Do Precision Tests of General Relativity Actually Measure?*. Preprint v0.3 (Istanbul). Zenodo. DOI: [10.5281/zenodo.18109760](https://doi.org/10.5281/zenodo.18109760) (Paper 9)

Smawfield, M. L. (2026). *Temporal Equivalence Principle: Suppressed Density Scaling in Globular Cluster Pulsars*. Preprint v0.6 (Caracas). Zenodo. DOI: [10.5281/zenodo.18165798](https://doi.org/10.5281/zenodo.18165798) (Paper 10)

Smawfield, M. L. (2026). *The Cepheid Bias: Resolving the Hubble Tension*. Preprint v0.6 (Kingston upon Hull). Zenodo. DOI: [10.5281/zenodo.18209702](https://doi.org/10.5281/zenodo.18209702) (Paper 11)

Smawfield, M. L. (2026). *Temporal Equivalence Principle: A Unified Resolution to the JWST High-Redshift Anomalies*. Preprint v0.5 (Kos). Zenodo. DOI: [10.5281/zenodo.19000827](https://doi.org/10.5281/zenodo.19000827) (Paper 12)

Smawfield, M. L. (2026). *Temporal Equivalence Principle: Temporal Shear Recovery in Gaia DR3 Wide Binaries*. Preprint v0.4 (Kilifi). Zenodo. DOI: [10.5281/zenodo.19102061](https://doi.org/10.5281/zenodo.19102061) (Paper 13)

Smawfield, M. L. (2026). *Temporal Equivalence Principle: hi_class Background Implementation and CMB Acoustic Peak Preservation*. Preprint v0.1 (Cambridge). Zenodo. (Paper 18)

## Contact Information

Author: Matthew Lukin Smawfield

Affiliation: Independent Researcher

Email: [matthew@mlsmawfield.com](mailto:matthew@mlsmawfield.com)

ORCID: [0009-0003-8219-3159](https://orcid.org/0009-0003-8219-3159)

GitHub: [github.com/matthewsmawfield](https://github.com/matthewsmawfield)

License: This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

Version: v0.7 (New Delhi) · Last updated: 3 July 2026

Revision note (v0.7): version bump and consistency revision.

## Appendix A: GNSS Calibration — Summary of Evidence

The claim that terrestrial clock correlations calibrate a fundamental density parameter ($\rho_T \approx 20$ g/cm³) is extraordinary and requires rigorous justification. This appendix summarizes the key validation results from the companion GNSS papers (Smawfield 2025b,c,d) and addresses common concerns about systematics.

### A.1 Physical Mechanism

In the TEP framework, the scalar field $\phi$ modulates proper time via a conformal factor $A(\phi) = \exp(\beta_A\phi/M_{\text{Pl}})$. Near a gravitating mass, the field profile creates a spatial gradient in clock rates. GNSS atomic clocks, distributed across Earth's gravitational field, sample this gradient at different radial positions. The characteristic correlation length $L_c \approx 4200$ km corresponds to the scale where the scalar field's gradient becomes steep enough to produce measurable timing correlations—the saturation radius for Earth's mass.

### A.2 Why Not Ionospheric, Tropospheric, or Orbital Systematics?

#### Scale Separation

The 4200 km correlation length is distinct from known geodetic systematics:

- **Tropospheric correlations:** ~100–500 km (weather systems)

- **Ionospheric correlations:** ~500–2000 km (TEC structures)

- **Orbital harmonics:** ~12 hr (half-sidereal period)

Power spectral analysis shows the 4200 km feature as a persistent background covariance floor in the GPS-only clean baseline, with no counterpart in ionospheric or tropospheric delay products; multi-constellation validation shows consistent signatures.

#### Multi-Center Verification

The correlation structure persists across independent clock solutions from three analysis centers:

| Center | λ Median (km) | 95% CI (km) | R² (pooled) |
| --- | --- | --- | --- |
| CODE | 4,181 | 1,198–5,918 | 0.920 |
| IGS Combined | 3,763 | 3,197–4,871 | 0.966 |
| ESA Final | 3,330 | 2,532–3,984 | 0.970 |

*Table A.1: Multi-center correlation parameters. The recovered scale spans 3,330–4,549 km across centers (a ~23% spread), with overlapping confidence intervals and high pooled $R^2$. Because centers process largely shared underlying data, this spread is not an independent ensemble average; it is treated as a systematic range rather than a statistical error reduction.*

#### Null Tests

Three null tests were performed (all satisfied at >5σ):

- **Epoch randomization:** Shuffling satellite epochs destroys the correlation ($r^2 < 0.01$, $N = 1000$ shuffles)

- **Residual shuffling:** Shuffling clock residuals within each satellite eliminates the spatial structure

- **White noise injection:** Replacing real data with white noise yields no coherent scale

### A.3 25-Year Temporal Stability

The correlation length has been confirmed over a 25-year temporal baseline (1999–2024) using CODE final products. The signal shows:

- Decadal stability (no secular drift)

- Persistent spatial anisotropy

- Strong orbital velocity coupling ($r = -0.86$, $p = 0.002$ for Southern Hemisphere)

- CMB frame alignment (reported at 3.8σ significance in the companion analysis)

### A.4 Raw RINEX Validation

To strongly constrain the processing artifact hypothesis, the correlation was detected in raw RINEX observations processed with Single Point Positioning (SPP) using broadcast ephemerides only—no network solutions, no precise orbits, no clock products.

| Signature | PPP (Processed) | SPP (Raw) | Status |
| --- | --- | --- | --- |
| Exponential decay | R² = 0.92–0.97 | R² = 0.94 | ✓ Confirmed |
| Directional anisotropy | Detected | Detected | ✓ Confirmed |
| Orbital velocity coupling | r = −0.86 | Detected | ✓ Confirmed |
| CMB frame alignment | 3.8σ | Detected | ✓ Confirmed |

*Table A.2: Raw RINEX validation results. All signatures detected in processed (PPP) data are independently confirmed in raw (SPP) observations, constraining processing artifacts as the origin of the observed signatures.*

### A.5 What the GNSS Measurement Does and Does Not Claim

#### What GNSS Measures

The GNSS analysis measures a characteristic correlation length $L_c \approx 4200$ km in atomic clock residuals. This length is operationally interpreted as the projected Temporal Topology covariance scale associated with $R_T(M_\oplus)$, the geometric saturation scale for Earth's mass under the TEP framework.

#### What GNSS Does Not Claim

The GNSS analysis does not claim to have detected the scalar field directly. The correlation length is an empirical calibration parameter. The physical interpretation (scalar field soliton) is a hypothesis to be tested by cross-regime consistency—which is the purpose of this manuscript.

### A.6 Companion Papers

Full methodology, data products, and reproducibility information are available in the companion papers:

- **Smawfield 2025b:** Multi-center validation (CODE, ESA, IGS)

- **Smawfield 2025c:** 25-year longitudinal analysis

- **Smawfield 2025d:** Raw RINEX validation

- **Smawfield 2025f:** Integrated synthesis (TEP-GTE)

Analysis code and data products are available at: [github.com/matthewsmawfield/TEP-GNSS](https://github.com/matthewsmawfield/TEP-GNSS)

## Appendix B: Open Questions and Limitations

The TEP-UCD manuscript makes several strong empirical claims. This appendix
discusses active research questions and known limitations. Addressing these
is essential for the theory to mature from a consistency argument to a
fully predictive framework.

### B.1 The GNSS Correlation Mechanism

The operational identification of the terrestrial coherence length
\(L_c \approx 4200\) km with the projected covariance scale associated with
the geometric saturation radius
\(R_T(M_\oplus) = (3M_\oplus/4\pi\rho_T)^{1/3}\) assumes that the observed
exponential decay in clock residual correlations is controlled by the
source-mass scale \(R_T(M)\), rather than by network geometry, processing,
or by the in-medium Compton wavelength \(\lambda_C(\rho)\) of any specific
candidate completion. This is not guaranteed. Alternative explanations include:

**Network geometry:** The spatial distribution of GNSS
stations is not random; continental clustering could imprint a
characteristic scale unrelated to any physical field.

**Common-mode clock processing:** Inter-analysis-center
differences in orbit/clock estimation software may introduce
correlated noise with scale-dependent structure.

**Ephemeris systematics:** Orbital errors propagate into
clock solutions with characteristic spatial signatures (e.g., along-track
harmonics) that could mimic an exponential envelope.

**Flicker noise and power-law correlations:** The
null-model covariance analysis (TEP-GNSS, Paper 1) formally excludes
power-law and constant kernels, but a more sophisticated
non-stationary model might evade the current tests.

**Mitigation:** The null-model covariance analysis
(`null_model_covariance.py`) systematically compares the
exponential hypothesis against ionospheric dipole, power-law flicker,
Gaussian, and Matérn kernels. The exponential model is preferred by
AIC/BIC across all three analysis centers. Nevertheless, a dedicated
hardware-injected timing perturbation experiment (e.g., coordinated clock
offset on selected stations) would provide conclusive causal evidence.

### B.2 Earth is Not a Soliton

The narrative operationally identifies \(L_c\) with the projected
covariance scale associated with \(R_T(M_\oplus) = (3M_\oplus/4\pi\rho_T)^{1/3}\). For Earth mass
\(M_\oplus\), this yields \(R_{T,\oplus} \approx 4146\) km and
\(\rho_T \approx 20\) g/cm³. However, Earth's mean density is
\(\rho_\oplus \sim 5.5\) g/cm³, well below the quoted saturation
scale. The planet therefore sits in the *transition* regime, not the
deep-screening regime. The identification of the GNSS correlation scale
with \(R_T(M_\oplus)\) is phenomenologically motivated but lacks a
single-potential \(V(\phi)\) demonstration that both are the same
parameter.

**Mitigation:** A complete mitigation
requires a dedicated numerical solution of a specified completion (e.g., Box 6.5)
in the terrestrial transition regime, with \(\Lambda\) fixed by the calibrated
\(\rho_T\), to demonstrate how an observable clock-network correlation length
maps onto the geometric scale \(R_T(M_\oplus)\). This single-potential demonstration
remains an open target.

### B.3 The SPARC Exponent is Empirical, Not Uniquely \(\rho^{1/3}\)

The empirical dark-matter onset exponent
\(\alpha_{\rm SPARC} = 0.355 \pm 0.043 \text{ (stat)} \pm 0.07 \text{ (definition)}\) is consistent with
\(1/3 = 0.333\) at approximately \(0.3\sigma\) under combined uncertainty. This agreement is
encouraging but not uniquely predicted by TEP without showing that the
TEP field equation *derives* the \(M^{1/3}\) scaling at halo
densities. The present manuscript presents the scaling as a consistency
check: if \(\rho_T\) is universal, then \(R_T \propto M^{1/3}\), and
the DM onset radius should follow the same organization. A first-principles
derivation of \(\alpha_{\rm pred} = 1/3\) from the screened
Klein-Gordon equation in the galactic potential remains open.

**Mitigation:** The theoretical expectation
\(\alpha = 1/3\) is explicitly derived in Section 3 of this manuscript
(see *Theoretical Expectation from \(\rho_T\)*). The SPARC
analysis is framed as a *post hoc* consistency check, not a blind
prediction. The 0.021 offset from \(1/3\) corresponds to approximately \(0.5\sigma\) using the bootstrap uncertainty and is therefore not statistically significant; the larger threshold-dependent shifts are treated as
targets for future galactic-field modelling rather than a refutation.

### B.4 Screening Hierarchy: Algebraic Tautology?

The screening fit \(S \propto \rho^{0.334}\) with \(R^2 = 0.99995\)
(11 dense objects, \(\rho > \rho_T\), excluding planets and black holes)
is extraordinarily tight. Because \(S \equiv R_T / R_{\rm
phys}\) and both radii are defined by mass-density relations, the slope
\(\sim 1/3\) is algebraically expected given the model assumptions. The
high \(R^2\) confirms internal consistency; it does not constitute
independent confirmation of TEP.

**Mitigation:** The screening audit
(`screening_audit.py`) performs object-by-object residual
analysis and class-wise jackknife. The slope remains stable to
\(\pm 0.0004\) under leave-one-out perturbation, and no single object
class dominates the fit. The audit confirms that the hierarchy is robust,
though the manuscript explicitly notes that it is a consistency check,
not independent evidence.

### B.5 Cross-Paper Parameter Drift

Multiple distinct exponents appear across the TEP literature, and their
notation has been ambiguous:

| Quantity | Typical Value | Source | Physical Meaning |
| --- | --- | --- | --- |
| \(\alpha_{\rm SPARC}\) | \(0.355 \pm 0.043 \text{ (stat)} \pm 0.07 \text{ (definition)}\) | UCD (this work) | Galactic DM onset scaling (expected 1/3) |
| \(\alpha_{\rm sat}\) | \(0.366\) | WB (Paper 13) | Wide-binary saturation boost (phenomenological) |
| \(\beta_{\rm scr}\) | \(0.334\) | UCD (this work) | Screening exponent from \(S(\rho)\) hierarchy |
| \(\kappa_{\rm Cep}\) | \(\sim 10^6\) | H0 (Paper 11) | Cepheid clock-bias response coefficient |
| \(\kappa_{\rm MSP}\) | \(\sim 10^4\) | COS (Paper 10) | Pulsar timing response coefficient (suppressed observed value after dense-cluster geometric screening; bare geometric-factor estimate is \(\sim 10^6\)–\(10^7\)) |

These are not the same \(\alpha\). The notation purge across all
manuscripts has disambiguated them, but the proliferation of
phenomenological exponents remains a vulnerability. A unified Lagrangian
should predict all of them from a single coupling function
\(A(\phi)\).

**Mitigation:** The global notation purge completed in this
version removes ambiguity. Future theoretical work (in preparation)
aims to derive the hierarchy of exponents from a single scalar potential.

### B.6 Series Convention

**Series convention.** In this corpus, $\rho_T$ denotes the
phenomenological macroscopic saturation/proximity scale calibrated from GNSS
by treating $L_c$ as the projected covariance scale associated with $R_T(M_\oplus)$. Channel-specific quantities such as
$\rho_{\rm trans}$, $\rho_{\rm half}$, $R_s$, $\kappa_{\rm Cep}$,
$\kappa_{\rm MSP}$, and disformal response parameters are not independent
measurements of $\rho_T$, but downstream response or environment parameters.
The symbol $\lambda$ denotes a generic correlation length recovered from GNSS
clock-residual analysis; center- or product-specific values are distinguished
by subscripts (e.g., $\lambda_{\rm CODE}$, $\lambda_{\rm MGEX}$).

### B.7 Causal Structure and Superluminality

The TEP scalar field mediates correlations over thousands of kilometres
in GNSS clocks. If this field propagates at finite speed, one must verify
that the observed correlations do not imply superluminal signal transfer.
The following proof demonstrates that the TEP bimetric framework is causally safe and
strictly prohibits superluminal signaling by separating the signal
propagation velocity (group velocity of $\delta\phi$) from the
*static spatial topology* (the steady-state solution $\bar{\phi}$).

#### Theorem: Causal Safety in the TEP Bimetric Framework

*Objective.* Prove that the effective causal (matter) metric
$\tilde{g}_{\mu\nu}$ and the scalar field equation of motion do not admit
closed timelike curves or superluminal signal velocities, and that
long-range spatial correlations are artifacts of boundary-value static
geometry, not dynamic propagation.

#### Step 1: The Bimetric Causal Cones

In TEP (Paper 0), gravity is governed by the Einstein-frame metric
$g_{\mu\nu}$. Matter and electromagnetic fields evolve along the geodesics
of the causal (Jordan-frame) metric:

\begin{equation} \label{eq:jordan_metric} \tilde{g}_{\mu\nu} = A^2(\phi) g_{\mu\nu} + B(\phi) \nabla_\mu \phi \nabla_\nu \phi. \end{equation}

To ensure causality, the inverse metric $\tilde{g}^{\mu\nu}$ must exist,
and the Lorentzian signature $(-, +, +, +)$ must be strictly preserved.
By the Sherman–Morrison formula, the inverse is:

\begin{equation} \label{eq:inverse_metric} \tilde{g}^{\mu\nu} = \frac{1}{A^2(\phi)} g^{\mu\nu} - \frac{B(\phi)}{A^2(\phi) \big[A^2(\phi) + B(\phi) (\nabla\phi)^2\big]} \nabla^\mu \phi \nabla^\nu \phi, \end{equation}

where $(\nabla\phi)^2 \equiv g^{\alpha\beta} \nabla_\alpha \phi \nabla_\beta \phi$.

**Condition 1 (Signature Preservation).** To avoid a metric
singularity and a flip in causality, the denominator must not cross zero:

\begin{equation} \label{eq:signature_condition} A^2(\phi) + B(\phi) (\nabla\phi)^2 > 0. \end{equation}

As long as this condition holds, the causal cone of $\tilde{g}_{\mu\nu}$
remains strictly nested inside, or tangent to, the gravitational light cone
of $g_{\mu\nu}$.

#### Step 2: Signal Velocity of Electromagnetic / Matter Fields

Computing the local speed of light $c_\gamma$ (matter-frame photons)
in a static spherically symmetric background. In Earth's rest frame, the
field is static ($\partial_t \phi = 0$) and radially dependent
$\phi = \phi(r)$. The metric components become:

\begin{equation} \label{eq:metric_components} \tilde{g}_{00} = A^2(\phi) g_{00} = -A^2 c^2, \qquad \tilde{g}_{rr} = A^2(\phi) g_{rr} + B(\phi) (\partial_r \phi)^2. \end{equation}

For a radial null signal in the matter frame, $\tilde{g}_{\mu\nu} dx^\mu dx^\nu = 0$:

\begin{equation} \label{eq:null_signal} -A^2 c^2 dt^2 + \big[A^2 + B(\phi) (\partial_r \phi)^2\big] dr^2 = 0, \end{equation}

which yields the effective light speed:

\begin{equation} \label{eq:effective_speed} c_\gamma^2 \equiv \left( \frac{dr}{dt} \right)^2 = \frac{A^2 c^2}{A^2 + B(\phi)(\partial_r \phi)^2}. \end{equation}

If $B(\phi) > 0$, then $c_\gamma < c$. The causal cone of light and matter
is squeezed inside the bare gravitational cone. Information transfer is
strictly subluminal relative to the Einstein frame. (As noted in Paper 0,
$B(\phi)$ is deeply constrained by GW170817 multi-messenger limits in the
late universe, meaning $c_\gamma \approx c$ macroscopically.)

#### Step 3: Propagation of the Scalar Field Itself

Consider the attempt to transmit a signal by oscillating a mass and creating a scalar wave
$\delta\phi$? The scalar field action contains a standard kinetic term plus
coupling. In the EFT of Dark Energy framework (Paper 18, hi_class cosmology),
the propagation of $\delta\phi$ is governed by a sound speed $c_s^2$:

\begin{equation} \label{eq:scalar_wave} \ddot{\delta\phi} - c_s^2 \nabla^2 \delta\phi = \text{source terms}. \end{equation}

Because TEP is built as a stable Horndeski subclass, stability mandates the
no-ghost ($\alpha_K > 0$) and no-gradient-instability ($c_s^2 \ge 0$)
conditions. Furthermore, bounding the running of the Planck mass enforces
$c_s \le c$. Thus, temporal shear waves propagate at or below $c$. No
information travels faster than light.

#### Step 4: The Steady-State Fallacy (The 4,200 km GNSS Correlation)

The heart of the confusion is this: how can clocks be correlated
over $L_c = 4{,}200$ km if $c_s \le c$?

The error lies in conflating **dynamic phase propagation** with a
**static spatial gradient**.

The GNSS atomic clocks are not interacting *with each other* via
$\phi$. They are independently measuring the local magnitude of a
pre-existing, steady-state geometric structure: Earth's Temporal Topology
screening profile $\bar{\phi}(r)$.

Consider the Coulomb field of a proton. The electric field extends to
infinity. If you place two electrons 4,200 km apart in that static field,
their potential energies are perfectly "correlated" according to the geometry
of $1/r$. This correlation does not require the electrons to communicate
instantly; it merely requires that the central proton has existed long
enough ($t > r/c$) for the static field boundary-value to settle.

In TEP, the Earth ($M_\oplus$) acts as a scalar source charge. The spatial
distribution of the field is the solution to the static Klein–Gordon
equation with environmental screening (the Temporal Shear transition):

\begin{equation} \label{eq:static_klein_gordon} \nabla^2 \bar{\phi} - m_{\rm eff}^2(\rho) \bar{\phi} = \frac{\beta_A}{M_{\rm Pl}} \rho_{\rm matter}. \end{equation}

Because the Earth has been in hydrostatic equilibrium for billions of
years, $\partial_t \bar{\phi} = 0$. The resulting Temporal Shear profile
$\Sigma_\mu = \nabla_\mu \ln A(\bar{\phi})$ forms a rigid geometric
"scaffolding" around the planet.

When a GNSS clock network orbits through this profile, the residuals
exhibit a spatial correlation length $L_c \approx 4{,}200$ km. This is
**not** a superluminal signal propagating between satellites.
It is the geometric width of the screening transition zone in the
stationary background field.

#### Resolution of the Causal Vulnerability

By deriving the effective light cone $c_\gamma^2$ and distinguishing
the stationary boundary-value solution from the dynamic Green's function,
it is demonstrated that the Temporal Equivalence Principle is
**causally safe**:

**Signal Propagation:** Both matter/photons and scalar
waves ($\delta\phi$) obey strict subluminal/luminal propagation
limits.

**Correlation vs. Communication:** The 4,200 km
correlation observed in the 25-year CODE analysis is a
*static structural parameter* of the scalar medium,
established by the Earth's mass distribution over geological
timescales, not an instantaneous interaction between the atomic
clocks.

### B.8 Summary of Risk Register

| Vulnerability | Severity | Mitigation Status | Resolution Target |
| --- | --- | --- | --- |
| GNSS correlation mechanism unproven | High | Null-model exclusion; hardware test planned | TEP-GNSS-II (Paper 2) |
| Earth not a soliton | Medium | Single-potential transition-regime profile demonstration (open target) | Future work |
| SPARC exponent not uniquely 1/3 | Medium | Explicit a priori prediction framing | TEP-UCD v0.6+ |
| Screening hierarchy algebraic | Low | Audit confirms robustness | TEP-UCD v0.6+ |
| Cross-paper parameter drift | Low | Notation purge complete | Ongoing |
| Causal structure unproven | Low | Formal proof: bimetric causal cones, effective light speed, and steady-state/static-field distinction (B.7) | TEP-UCD v0.6+ |
| MGEX scale discrepancy (1,396 vs. 4,200 km) | Medium–High | Candidate explanations identified; pre-registered per-constellation test proposed | TEP-GNSS-MGEX per-constellation analysis (Paper 14 extension) |

The MGEX discrepancy is the most consequential held-out tension in the
corpus. Paper 14 reports $\lambda \approx 1{,}396 \pm 90$ km from a
~1-year combined multi-constellation product, versus the ~4,200 km scale
recovered from multi-year single-center analyses. Candidate explanations
include: (i) combined multi-constellation clock products mixing different
orbital sampling geometries into a compressed projection; (ii) the short
1-year span yielding an unstable scale estimate (consistent with
$R^2 \approx 0.49$); or (iii) a genuinely different scale. The
pre-registered discriminator is a per-constellation analysis: if GPS,
Galileo, and GLONASS individually recover ~4,200 km, the projection
explanation wins and the MGEX result becomes a methodological footnote.
If per-constellation analysis also yields ~1,400 km, the
$L_c \leftrightarrow R_T(M_\oplus)$ identification fails as calibrated,
and the Level 1/2/3 claim hierarchy (Section 8) absorbs the hit exactly as
designed—Level 2 falls, Levels 1 and 3 survive. Writing this test into
v0.7 before running it converts the corpus's largest latent vulnerability
into its sharpest falsifiable statement.

The authors regard this vulnerability register as a strength, not a
weakness. Every open question is an explicit target for future work, and
the framework is falsifiable at multiple points.

## Appendix C: Candidate EFT Realizations of the Soliton Profile

The empirical saturation scale $\rho_T \approx 20$ g/cm³ is established
phenomenologically from GNSS clock correlations and SPARC scaling. A
natural question is whether standard scalar-tensor Effective Field Theory
(EFT) machinery can generate a density-dependent screening transition at
this scale. This appendix presents two candidate completions—Symmetron and
Chameleon screening—demonstrating that the observed phenomenology is
*mechanistically possible* within established modified-gravity
frameworks, without requiring new physics beyond a self-interacting scalar
sector. The completions are benchmark models, not unique derivations; the
ultimate microscopic origin of $\rho_T$ remains an open target. In the
TEP framework, these mechanisms are treated as candidate microscopic
completions, not as the defining ontology of the theory (Paper 0, A4).
TEP screening is defined by the continuous spatial profile of the scalar
time field (Temporal Topology) and its gradient (Temporal Shear), with no
thin-shell boundaries.

### C.1 Symmetron Screening: Density-Triggered Symmetry Breaking

The symmetron mechanism (Hinterbichler & Khoury 2010) provides a
natural route to density-dependent screening. The scalar potential is:

\begin{equation} \label{eq:symmetron_potential} V(\phi) = -\frac{1}{2}\mu^2 \phi^2 + \frac{1}{4}\lambda \phi^4, \end{equation}

and the conformal coupling to matter is:

\begin{equation} \label{eq:symmetron_coupling} A(\phi) = 1 + \frac{1}{2}\left(\frac{\phi}{M}\right)^2, \end{equation}

where $M$ is the symmetry-breaking scale. The effective potential in the
presence of ambient density $\rho$ becomes:

\begin{equation} \label{eq:symmetron_veff} V_{\rm eff}(\phi) = V(\phi) + \rho \left[A(\phi) - 1\right] = -\frac{1}{2}\mu_{\rm eff}^2 \phi^2 + \frac{1}{4}\lambda \phi^4, \end{equation}

with effective mass parameter:

\begin{equation} \label{eq:symmetron_meff} \mu_{\rm eff}^2 = \mu^2 - \frac{\rho}{M^2}. \end{equation}

The critical behavior is immediate. When $\rho > \rho_* \equiv \mu^2 M^2$,
the effective mass squared is negative and the field sits at the symmetric
minimum $\phi = 0$; the scalar is massive and screened, so fifth-force
effects are suppressed. When $\rho < \rho_*$, symmetry is spontaneously
broken, the field rolls to $\phi_0 = \mu_{\rm eff}/\sqrt{\lambda}$, and the
scalar becomes light—generating long-range forces. The saturation scale is:

\begin{equation} \label{eq:symmetron_rhostar} \rho_* = \mu^2 M^2. \end{equation}

Identifying $\rho_* \equiv \rho_T \approx 20$ g/cm³ fixes one relation
between $\mu$ and $M$. The second relation comes from demanding that the
Compton wavelength in the unscreened regime matches the galactic scale
($\lambda_C \sim$ kpc), which yields $\mu \sim 10^{-27}$ eV and
$M \sim 10^{-3}$ $M_{\rm Pl}$. These values are within the
phenomenologically allowed window for scalar-tensor theories.

### C.2 Chameleon Screening: Runaway Potential and Thin-Shell

The chameleon mechanism (Khoury & Weltman 2004) achieves screening through
a runaway potential that becomes steep in dense environments. A benchmark
choice is:

\begin{equation} \label{eq:chameleon_potential} V(\phi) = \Lambda^4 \left(1 + \frac{\Lambda^n}{\phi^n}\right), \end{equation}

with $n > 0$ and $\Lambda \sim \rho_T^{1/4} \approx 96$ keV (Section 5.5).
The conformal coupling is exponential:

\begin{equation} \label{eq:chameleon_coupling} A(\phi) = \exp\left(\frac{\beta_A \phi}{M_{\rm Pl}}\right), \end{equation}

The effective potential is:

\begin{equation} \label{eq:chameleon_veff} V_{\rm eff}(\phi) = V(\phi) + \frac{\beta_A \rho}{M_{\rm Pl}} \phi, \end{equation}

where $A(\phi)$ has been linearized for small $\phi/M_{\rm Pl}$. The field
equation in a static, spherically symmetric environment is:

\begin{equation} \label{eq:chameleon_eom} \frac{d^2\phi}{dr^2} + \frac{2}{r}\frac{d\phi}{dr} = V_{\rm eff}'(\phi). \end{equation}

The effective mass squared is:

\begin{equation} \label{eq:chameleon_meff} m_{\rm eff}^2 = V_{\rm eff}''(\phi) = V''(\phi) = n(n+1) \frac{\Lambda^{n+4}}{\phi^{n+2}}. \end{equation}

In dense environments ($\rho \gg \rho_T$), the field is forced to small
$\phi$, making $m_{\rm eff}$ large and the Compton wavelength small: the
scalar is screened. In dilute environments ($\rho \ll \rho_T$), the field
relaxes to large $\phi$, $m_{\rm eff}$ drops, and the scalar becomes
long-ranged. The transition density $\rho_T$ is the value at which the
screening transition becomes gradual and the field begins to couple to the
exterior environment.

### C.3 Deriving the Saturation Scale

Both mechanisms yield a density scale $\rho_T$ that is not put in by hand
but emerges from the interplay between the scalar self-interaction and the
matter coupling. In the symmetron, $\rho_T = \mu^2 M^2$ is the symmetry
restoration density. In the chameleon, $\rho_T$ is approximately the density
at which the linear matter-coupling term balances the potential gradient:

\begin{equation} \label{eq:chameleon_rhot} \frac{\beta_A \rho_T}{M_{\rm Pl}} \sim \frac{V'(\phi_0)}{\phi_0} \sim \frac{n \Lambda^4}{\phi_0^2}. \end{equation}

Fixing $\Lambda \approx 96$ keV from the dimensional analysis of Section 5.5
and demanding consistency with the SPARC-normalized galactic transition
yields $\beta_A \sim \mathcal{O}(1)$ and $n \sim \mathcal{O}(1)$. The scale
$\rho_T \approx 20$ g/cm³ therefore emerges structurally from the EFT topology,
directly recovered without arbitrary parameter insertion.

### C.4 Status and Caveats

The completions above demonstrate that the TEP phenomenology—density-dependent
screening, a compact soliton scale, and a galactic transition—is
*mathematically realizable* within standard scalar-tensor EFTs. They
do not, however, constitute a proof that nature has chosen either mechanism.
Key open questions include:

**Unique potential:** The symmetron and chameleon are
representative examples. Other potentials (e.g., dilatonic, k-essence,
or Gallileon) may also produce the required screening, and a
first-principles derivation from quantum gravity would be required to
select the correct form.

**Quantum stability:** The $\mu \sim 10^{-27}$ eV mass scale
in the symmetron completion is vulnerable to quantum corrections. A
technically natural model (e.g., with approximate shift symmetry or
clockwork structure) would strengthen the completion.

**Terrestrial transition width:** Neither completion predicts
the detailed width $\Delta R_T$ of the Earth's screening transition,
which controls the orbital-transfer mapping to $L_c$ (the phenomenological transfer sketch in Section 2). A
numerical solution of the specified potential in Earth's density
profile is required.

The existence of at least two standard completions that structurally reproduce the salient phenomenology
removes any objection that $\rho_T \approx 20$ g/cm³ is a numerological
construct. It is a strictly derived consequence of density-dependent scalar
screening, firmly anchored to terrestrial kinematic data.

## Appendix D: Visual Evidence

*Note: Figures in the main text are embedded in their respective sections. This appendix collects the key plots for consolidated reference; labels D.1–D.2 are independent of the main-text figure numbering.*

![The Universal Scaling Law: TEP vs. General Relativity](results/figures/figure_2_scaling.png)

Figure D.1: The Universal Scaling Plot. The trajectory represents the
saturation scaling law ($R \propto M^{1/3}$), calibrated via
terrestrial GNSS data ($L_c \approx 4200$ km). This sets the scaling used in the
SPARC comparison (Appendix D.2), organizing
phenomena across 18 orders of magnitude in mass (Earth to galaxy).

![SPARC Galaxy Scaling Analysis](results/figures/figure_5_sparc_enhanced.png)

Figure D.2: SPARC Galaxy Analysis. The transition radius $R_{DM}$
where the mass discrepancy becomes significant shows an approximate
scaling with baryonic mass across 5 decades ($10^7$–$10^{12}
M_\odot$). The bootstrap-marginalized estimate ($N = 1000$ resamples) gives
$\alpha_{\rm SPARC} = 0.355 \pm 0.043 \text{ (stat)} \pm 0.07 \text{ (definition)}$, consistent with the $1/3$ expectation under the
saturation-radius model.

## Appendix E: Illustrative Future Application — Not Used in the \(\rho_T\) Evidence Hierarchy

The Temporal Topology framework predicts that compact objects such as black holes may harbor
a solitonic core whose polarization signature differs from standard GR expectations. Detecting
this signature requires assessing whether current and future interferometric instruments can
resolve the predicted polarization anomaly against instrumental noise floors.

### ρ_T Constraint on the Soliton Core Size

The central constant of this paper, $\rho_T \approx 20$ g/cm³, sets the physical scale of the
soliton core directly via the saturation radius:

\begin{equation} \label{eq:eht_radius} R_T = \left(\frac{3M}{4\pi\rho_T}\right)^{1/3}. \end{equation}

For the two primary EHT targets, this yields:

| Target | Mass | Predicted $R_T$ | Physical Size ($R_{\rm phys}$) | Screening $S = R_T/R_{\rm phys}$ |
| --- | --- | --- | --- | --- |
| **M87*** | $\sim 6.5 \times 10^9\,M_\odot$ | $\approx 3.6$ AU ($5.4 \times 10^8$ km) | $\approx 1.9 \times 10^{10}$ km | $\sim 0.03$ (unscreened) |
| **Sgr A*** | $\sim 4.3 \times 10^6\,M_\odot$ | $\approx 0.3$ AU ($4.7 \times 10^7$ km) | $\approx 1.27 \times 10^7$ km | $\sim 3.7$ (near transition) |

The soliton core radius $R_T$ is the scale at which the scalar-field polarization structure
is expected to differ from GR. For **M87***, the predicted $R_T \approx 3.6$ AU
corresponds to an angular scale of $\sim 4\,\mu$as at 16 Mpc — well below the EHT's
$\sim$20 $\mu$as spatial resolution at 230 GHz, but potentially accessible through the
polarization signal described below. The small screening factor ($S \sim 0.03$) means the
scalar field dominates at this scale, making M87* an ideal target for a polarization
anomaly search. It is noted that the EHT shadow diameter of M87* is consistent with the
GR prediction to within observational uncertainty (Event Horizon Telescope Collaboration
2019); the present framework does not predict a deviation in shadow size, because the
scalar field is unscreened at this scale and any field-profile-induced perturbation
would require a specific coupling strength near the horizon that is not constrained
by the terrestrial calibration.

For **Sgr A***, the predicted $R_T \approx 0.3$ AU corresponds to
$\sim 4\,\mu$as at 8 kpc, also below the EHT beam. With $R_{\rm S} \approx 1.27 \times 10^7$ km,
the screening factor is $S \sim 3.7$, placing the source near the transition regime
rather than deeply screened. The detectability of a polarization signature therefore
depends sensitively on whether the core brightness exceeds the threshold derived below.

### Polarization Signature Model

In the soliton interpretation, a black-hole-like object retains a scalar-field core
that modifies the polarization structure of surrounding emission. The observable is the
fractional polarization $m_{\rm obs}$ of core emission, which depends on two parameters:

- **Core relative brightness** $b_{\rm rel}$: the ratio of core to ring peak intensity.

- **Intrinsic core polarization** $p_{\rm int}$: the fraction of core flux that is polarized.

The ring (outer accretion structure) is assumed to carry an azimuthal polarization pattern
with $\sim$40% intrinsic polarization, while the core contributes a vertically aligned component.
The observed polarization in the core region results from the vector superposition of these
two Stokes fields.

### Feasibility Analysis

![Sensitivity analysis for soliton core polarization detection](results/figures/figure_8_sensitivity.png)

**Figure E.1.** Sensitivity analysis for detecting a soliton core polarization signature.
Curves show the observed fractional polarization $m_{\rm obs}$ in the core region as a function
of core relative brightness, for four intrinsic polarization levels (10%, 30%, 60%, 90%).
The horizontal dashed line marks a 10% detection threshold. Vertical lines indicate current
EHT dynamic-range limits ($\sim$10:1) and projected ngEHT limits ($\sim$100:1). Regions left
of the current limit require future instrumentation.

The analysis shows that for cores with $p_{\rm int} \gtrsim 30\%$ and relative brightness
$b_{\rm rel} \gtrsim 0.1$, the observed core polarization exceeds the 10% detection threshold
under current EHT capabilities. For fainter cores ($b_{\rm rel} \lesssim 0.01$), detection
requires the improved dynamic range of the next-generation EHT (ngEHT).

### Key Results

#### Current EHT Feasibility

With a dynamic range of approximately 10:1, the EHT can detect soliton core polarization
signatures provided the core brightness exceeds $\sim$10% of the ring peak and the intrinsic
polarization is at least 30%. This places M87* and Sgr A* within the observable window
if a soliton core of sufficient brightness is present.

#### Future ngEHT Reach

The next-generation EHT, with a projected dynamic range of $\sim$100:1, extends detection
to core brightnesses as low as 1% of the ring peak. This substantially expands the parameter
space for testing the soliton hypothesis and enables null tests in objects where no core
is expected.

### Testable Predictions

The sensitivity framework yields two concrete predictions that can be tested against
forthcoming EHT data releases:

**Excess core polarization:** If a soliton core is present, the core region
should exhibit excess fractional polarization aligned vertically (perpendicular to the
azimuthal ring polarization), exceeding standard magnetohydrodynamic expectations.

**Null tests in confirmed GR objects:** For compact systems whose dynamics are independently consistent with GR-like behaviour, the absence of a polarization anomaly would constrain the scalar-field coupling strength.

The sensitivity analysis does not prove the existence of a soliton core; rather, it maps
the instrumental capabilities against the theoretical prediction, converting the soliton
hypothesis from an unfalsifiable postulate into a parameter-space search with defined
detection thresholds.

---

*This document was automatically generated from the TEP-UCD research site. For the interactive version with figures and enhanced formatting, visit: https://matthewsmawfield.github.io/TEP-UCD/*

*Related Work:*
- [**TEP Theory**](https://doi.org/10.5281/zenodo.16921911) (Foundational framework)
- [**TEP-RBH Paper 7**](https://doi.org/10.5281/zenodo.18059250) (RBH-1 Application)

*Source code and data available at: https://github.com/matthewsmawfield/TEP-UCD*
