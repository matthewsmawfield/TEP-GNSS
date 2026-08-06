# Temporal Equivalence Principle: Black Holes and the Temporal Horizon
**Matthew Lukin Smawfield**
Version: v0.1 (Bahrain)
First published: 29 July 2026 - Last updated: 29 July 2026
DOI: 10.5281/zenodo.21677827

---

## Abstract

The Temporal Equivalence Principle treats proper time as a dynamical field. Matter, light, and ideal clocks couple to the universal causal metric $\tilde g_{\mu\nu}=A^2(\phi)g_{\mu\nu}+B(\phi)\nabla_\mu\phi\nabla_\nu\phi$. The Einstein-frame metric carries the gravitational dynamics; tensor propagation is determined by the principal symbol of the coupled temporal–geometric equations. This paper develops the strong-field consequence: *a black hole, under TEP, is a temporal well* — a regular spatial region in which the rate of proper time differs radically from the exterior, without physical collapse to an ultradense singular object and without an absolute one-way boundary. A continuous, extreme but finite gradient in proper-time rate is sufficient to produce the full observational phenomenology attributed to a black hole.

Darkness, apparent compactness, large inferred mass, and practical inaccessibility are exterior reconstructions of temporal decoupling. The operational boundary is the Temporal Horizon — the observer-relative threshold beyond which the clock-transfer factor renders signals practically undetectable — not the event horizon. Inferred gravitational mass and local material mass need not coincide; a strong-field phantom mass residual $M_{\rm phantom}^{T} \equiv M_{\rm fit}^{\rm GR} - M_{\rm matter}^{\rm TEP}$ measures this discrepancy, its sign determined by the data. Standard black-hole ontology assumes isochrony: source clocks, photon propagation, and observer clocks mapped onto a single general-relativistic time coordinate. TEP drops that closure. Under TEP, the conventional reconstruction — compact mass, event horizon, singular collapse — is no longer the unique reading of the same observations.

The temporal field cannot sit passively on fixed Schwarzschild geometry: finite curvature and bounded areal radius are mutually exclusive when $g_{\mu\nu}$ is held fixed. Strong temporal structure forces gravitational backreaction. The canonical TEP matter coupling fixes the temporal sector; the leading curvature operator supplies backreaction; the regularising nonlinear coefficients are selected by global regularity and observation. The construction programme is solving the global field equations of the fixed theory — not selecting among competing theories.

Four observational consequences follow from one temporal field: time transfer, mass inference, photon accessibility, and ringdown. Weak-field data recover GR; horizon-scale images constrain the photon-region geometry but do not directly establish an event horizon; gravitational-wave ringdown provides the sharpest test, because TEP replaces the purely ingoing event-horizon condition with propagation through a regular temporal domain, changing the late-time spectral problem. *Cosmological expansion and black-hole collapse are dual misreadings of dynamical proper time.*

Keywords: Temporal Equivalence Principle, temporal well, temporal horizon, dynamical proper time, phantom mass, isochrony axiom, conformal-disformal metric, Schwarzschild incompatibility, apparent compactness, black holes, modified gravity, temporal shear, black-hole observations

# 1. Introduction: Black Holes under the Temporal Equivalence Principle

## 1.1 The Temporal Well

The Temporal Equivalence Principle (TEP) treats proper time as a dynamical field. Matter, light, and ideal clocks couple to the universal causal metric $\tilde g_{\mu\nu}=A^2(\phi)g_{\mu\nu}+B(\phi)\nabla_\mu\phi\nabla_\nu\phi$. The Einstein-frame metric carries the gravitational dynamics; tensor propagation is determined by the principal symbol of the coupled temporal–geometric equations. This paper develops the strong-field consequence: a black hole, under TEP, is a temporal well — a regular spatial region in which the rate of proper time differs radically from the exterior, without physical collapse to an ultradense singular object and without an absolute one-way boundary. A continuous, extreme but finite gradient in proper-time rate is sufficient to produce the full observational phenomenology attributed to a black hole.

Darkness, apparent compactness, large inferred mass, and practical inaccessibility are exterior reconstructions of temporal decoupling. The operational boundary is the Temporal Horizon — the observer-relative threshold beyond which the clock-transfer factor renders signals practically undetectable — not the event horizon. The same principle that reframes cosmology reframes the dark compact sources called black holes. Cosmological expansion and black-hole collapse are dual misreadings of dynamical proper time.

## 1.2 The Principle

TEP distinguishes the Einstein-frame gravitational metric $g_{\mu\nu}$, which carries the gravitational dynamics, from the universal causal matter metric

\begin{equation} \label{eq:intro_1}
\tilde g_{\mu\nu}=A^2(\phi)g_{\mu\nu}+B(\phi)\nabla_\mu\phi\nabla_\nu\phi,
\end{equation}

on which matter, photons, and ideal clocks propagate. The conformal factor $A(\phi)$ maps clock scales; the disformal term $B(\phi)$ can tilt matter cones. In each local Lorentzian orthonormal frame of $\tilde g_{\mu\nu}$, nongravitational physics is locally special relativistic and the locally measured speed of light is invariant. The field $\phi$ is the temporal field — the field that governs the rate of proper time.

The principle is that a continuous, extreme but finite gradient in proper-time rate is sufficient to produce the full observational phenomenology attributed to a black hole: darkness, apparent compactness, large inferred mass, and practical inaccessibility for a distant observer. No absolute one-way boundary is required. No physical compression to ultradense matter is required. The observations themselves do not distinguish a physically compact collapsed body from a sufficiently strong temporal-rate gradient. The physical lapse satisfies

\begin{equation} \label{eq:intro_lapse}
0 < N(r), \qquad N_{\min} = \min_r N(r) \ll N_o,
\end{equation}

where the temporal-rate minimum may occur at a centre, a shell, or an asymmetric basin. The invariant quantity is not the lapse at any particular point but the largest source-to-observer transfer mismatch along physically connected worldlines:

\begin{equation} \label{eq:intro_transfer}
\mathcal{T}_{e\to o} = \frac{\omega_e}{\omega_o} = \frac{(-k_\mu u^\mu)_e}{(-k_\mu u^\mu)_o}, \qquad
\Delta_T = \ln\frac{\mathcal{T}_{e\to o}^{\rm TEP}}{\mathcal{T}_{e\to o}^{\rm GR}}.
\end{equation}

When $|\Delta_T|$ becomes extreme, signals are strongly redshifted, observed processes appear greatly slowed, matter appears to accumulate or freeze, reconstructed trajectories and spatial scales become distorted, a large central mass and compact radius may be inferred, and information becomes practically inaccessible. *Darkness does not require an event horizon; it requires sufficiently extreme temporal decoupling.*

## 1.3 The Observational Reading

Black-hole observations do not measure a singularity, an event horizon, or an ultradense core. They measure angles, frequencies, arrival times, and images. Those data are converted into mass, radius, and causal structure only by solving an inverse problem under a particular spacetime model. The hidden assumption in that inverse problem is the isochrony closure — source clocks, photon propagation, and observer clocks are mapped onto a single general-relativistic time coordinate. Under this closure, the received data are reduced to orbital periods, velocities, and radii, and then inverted through Kepler or GR to yield a large compact mass. The familiar relation

\begin{equation} \label{eq:intro_kepler}
M_{\rm app}^{\rm GR} = \frac{4\pi^2 a_{\rm GR}^3}{G P_{\rm GR}^2}
\end{equation}

returns an apparent mass conditioned on that temporal calibration. It does not independently measure either the local material mass or the proper volume of the source region. The event horizon, the singularity, and the ultradense core are further reconstructions — none is directly observed.

What is observed:

- photons, frequencies, timing, interferometric visibilities, orbital motion.

What is inferred:

- singularities, event horizons, proper volume, local density, compression.

TEP drops the isochrony closure. Under TEP, the conventional reconstruction — compact mass, event horizon, singular collapse — is no longer the unique reading of the same observations. Inferred gravitational mass and local material mass need not coincide; compactness, horizons, and collapse become emergent observational reconstructions rather than directly established physical facts.

## 1.4 The Event Horizon Is Not an Observable

The event horizon is a global causal construct. It is defined as the boundary of the causal past of future null infinity — a statement that requires knowledge of the entire future spacetime. It is not a local observable. No telescope, no interferometer, no gravitational-wave detector has ever measured an event horizon. What is observed is that signals from certain regions become extremely redshifted, delayed, and faint. The conventional interpretation assigns this to a one-way causal boundary. TEP assigns it to an extreme temporal-transfer mismatch. Both interpretations are compatible with the same observational data — that is the non-uniqueness.

The Temporal Horizon $\mathcal{H}_T^{(\Lambda)}[u_o]$ is the operational boundary defined directly from observable temporal accessibility:

\begin{equation} \label{eq:intro_TH}
\mathcal{H}_T^{(\Lambda)}[u_o] = \{x_e : \mathcal{T}_{e\to o} \geq \Lambda\}.
\end{equation}

It is not a null boundary, not one-way, and does not divide spacetime into an outside and an inside. Different observers place it differently because their clock rates differ. It is defined by what is observable — the transfer factor — not by a global property of an assumed spacetime.

## 1.5 Apparent Compactness $\neq$ Physical Compression

The exterior observer infers a compact, dark, massive object because processes deeper in the temporal field are strongly redshifted, signal arrival rates become extremely slow, light paths are strongly distorted, and matter appears to accumulate or freeze within a small apparent radius. But the density $\rho_{\rm inferred} \sim M/(4\pi r^3/3)$ is an exterior-frame inference that assumes the Schwarzschild radial coordinate and exterior clock remain valid measures of the deep region. The locally measured density and local physical volume need not reproduce that inference. The extreme density attributed to a black hole may be an artefact of reconstructing a region with a radically different proper-time rate using exterior spatial and temporal standards.

This interpretation is dual to the Temporal Horizon Cosmology reading (Thika); the strong-field Temporal Horizon is operationally distinct from the cosmological past boundary $\mathscr{T}^-$ (see Section 6.1). On cosmological scales, a temporal-rate gradient produces observed redshift, which standard physics interprets as spatial expansion — the conformal volume element $V_{\rm eff} = A_{\rm clock}^3 a_m^3$ tends to zero, but the underlying matter-frame spatial geometry does not collapse. Around an extreme gravitational source, a temporal-rate gradient produces redshift, lensing and altered trajectories, which standard physics interprets as gravitational collapse and inward suction. In both cases, the conventional spatial narrative is an observational projection of a single underlying dynamical proper-time field. *Cosmological expansion and black-hole collapse are dual misreadings of dynamical proper time.*

## 1.6 The Argument

This paper develops the strong-field consequence of TEP and shows what follows from it. The argument proceeds in five steps.

**First:** the theory is defined (Section 2). The two-metric action, the conformal-disformal matter metric, the frame dictionary, the observational inference problem, and the strong-field phantom mass decomposition. The isochrony closure is identified as the axiom TEP replaces.

**Second:** the analysis of Section 3 shows that the temporal field cannot sit passively on fixed Schwarzschild geometry. Finite curvature and bounded areal radius are mutually exclusive when $g_{\mu\nu}$ is held fixed. Strong temporal structure forces gravitational backreaction. If the temporal field becomes strong enough to create a Temporal Horizon, the geometry cannot remain ordinary Schwarzschild.

**Third:** the required geometric ingredients are shown to be mathematically attainable (Sections 4–5). Regular geometries exist. The leading curvature operator produces real backreaction. The canonical TEP matter coupling fixes the temporal sector; the regularising nonlinear coefficients are selected by global regularity and observation. The construction programme is solving the global field equations of the fixed theory.

**Fourth:** four observational consequences follow from one temporal field (Sections 6–8): time transfer, mass inference, photon accessibility, and ringdown. The Temporal Horizon is the operational boundary defined directly from observable temporal accessibility. Replacing an absorbing horizon boundary with a continuous, strongly time-dilated region changes the ringdown boundary-value problem in principle.

**Fifth:** existing data are confronted as examples of the framework (Section 9). S2 is a weak-field consistency check. Horizon-scale images constrain the photon-region geometry. Gravitational waves provide the strongest future test. None of these is a pillar holding up the theory — they are examples of the framework applied to real data.

## 1.7 Structure of the Paper

- Section 2 introduces the TEP theory: the two-metric action, the matter metric, the frame dictionary, the observational inference problem, and the phantom mass decomposition.

- Section 3 shows why Schwarzschild is not enough: a passive temporal field on fixed Schwarzschild geometry cannot satisfy the conditions for a physically admissible temporal well.

- Section 4 presents existence demonstrations: regular geometry and dynamical coupling as attainability proofs.

- Section 5 establishes global geometry and curvature regularity on the validation benchmark.

- Section 6 derives the causal structure and the Temporal Horizon as the operational boundary.

- Section 7 analyses the perturbation spectrum and the structural ringdown prediction.

- Section 8 presents four observational consequences from one temporal field.

- Section 9 discusses the physical interpretation, data confrontation, and scope.

- Section 10 concludes.

Detailed sGB derivations, numerical QNM solvers, EHT visibility fits, and pipeline manifests are collected in the appendices. The main text develops the TEP temporal well, the Schwarzschild incompatibility, the minimal temporal-well criteria, and the unified phenomenology.

## 1.8 Scope of This Paper

This paper develops the strong-field consequence of TEP: a black hole is a temporal well. The Schwarzschild incompatibility and the minimal temporal-well criteria follow from the TEP principle. The claim is that the phenomena attributed to black holes are produced by an extreme but finite gradient in proper-time rate — without physical collapse, without a singular core, without an absolute one-way boundary.

What is derived from TEP: the conventional black-hole reconstruction is not the unique reading of the observations; the event horizon is not a direct observable; conformal time on fixed Schwarzschild is incompatible with finite curvature and bounded area; the Temporal Horizon is operational and observer-relative; the ringdown boundary condition is altered.

What is constructively demonstrated: regular geometry and curvature-coupled scalar backreaction are attainable; finite-curvature matter metrics are achievable; weak-field S2 recovers GR; the exterior and deep-transit benchmarks isolate the geometric mechanism of the altered spectral problem.

What requires decisive closure: the TEP-selected global field solution; the complete coupled characteristic system; the raw non-isochronous multi-messenger refit; the measured phantom-mass sign.

# 2. TEP Theory: Action, Matter Metric, and Frame Dictionary

## 2.1 Canonical Action and Universal Matter Coupling

The canonical Einstein-frame action is used:

\begin{equation} \label{eq:tep_1}
S=\int d^4x\sqrt{-g}\left[\frac{M_{\rm Pl}^2}{2}R-\frac12(\nabla\Phi)^2-V(\Phi)\right]+S_m[\Psi_m,\tilde g_{\mu\nu}],
\end{equation}

with

\begin{equation} \label{eq:tep_2}
\tilde g_{\mu\nu}=A^2(\Phi)g_{\mu\nu}+B(\Phi)\nabla_\mu\Phi\nabla_\nu\Phi.
\end{equation}

This is the canonical frame: gravity has Einstein–Hilbert form, tensor modes propagate on $g_{\mu\nu}$, and all nongravitational fields couple universally to $\tilde g_{\mu\nu}$. Diffeomorphism invariance of the matter action gives $\tilde\nabla_\mu\tilde T^{\mu\nu}=0$ wherever $\tilde g$ is invertible. This action and matter metric are the foundational TEP structure (Jakarta); the present paper extends them to the strong-field regime. The pipeline uses the dimensionless field $\phi=\Phi/M_*$ and geometrized units $G=c=1$; the numerical normalisation, parameter freeze ($\beta_A=-1$, $B_0=1$), and weak-to-strong-field running are detailed in Appendix B. The lab-scale lock $\beta \simeq -0.013$ is a weak-field convention; strong-field profiles here are not required to hold that value pointwise.

The theory is a defined modified-gravity theory, not a broad framework into which arbitrary completions are inserted. The terms are organised into the fundamental sector and the strong-field EFT sector.

**Fundamental sector — fixed by the TEP principle.**

- The Einstein–Hilbert gravitational action $\frac{M_{\rm Pl}^2}{2}R[g]$: the Einstein–Hilbert term supplies the canonical gravitational kinetic structure. On the Schwarzschild benchmark, tensor propagation reduces to $c_T = 1$, consistent with GW170817; the nonlinear Temporal-Well characteristics follow from the complete coupled principal symbol.

- The canonical scalar kinetic $-\frac12(\nabla\Phi)^2$ and potential $V(\Phi)$: the temporal field $\Phi$ is the dynamical proper-time field. The canonical kinetic term supplies a hyperbolic scalar principal part, while $V''>0$ gives local stability around the selected background.

- The universal conformal matter coupling $A(\Phi) = \exp(\beta_A \Phi/M_{\rm Pl})$: a single function fixes the clock sector for all non-gravitational fields. This is the canonical TEP matter coupling; it is not a free function to be chosen per problem.

- The disformal matter coupling $B(\Phi)\nabla_\mu\Phi\nabla_\nu\Phi$: the disformal rule is fixed by the quartic-Gaussian form (Appendix B), constrained by GW170817 to $|c_\gamma - c_g|/c \lesssim \text{few}\times10^{-15}$ along observed paths. $B$ is not set to zero; it is bounded and structurally fixed.

- The environmental screening/running law $\Sigma_\mu^{\rm obs} = \mathcal S_\Sigma(\mathcal E)\,\Sigma_\mu$ (Section 2.5): the Temporal Topology operator governs weak-to-strong-field running of the effective coupling and reconciles terrestrial tests with cosmological dynamics. This is part of the theory, not an external add-on.

**Strong-field EFT sector — the leading curvature operator and regularising coefficients.**

- The leading curvature operator is the scalar–Gauss–Bonnet coupling $\alpha_{\rm GB}\,\phi\,\mathcal{G}$. This is the lowest-dimension higher-curvature term that couples the temporal field to the gravitational geometry and supplies real backreaction. It is the canonical strong-field extension of the TEP action, not one option among many.

- The regularising nonlinear coefficients — higher-order curvature couplings and potential terms that control the deep-region profile — are selected by the joint requirements of global regularity (finite curvature, bounded areal radius, Lorentzian signature) and observational consistency. These are EFT corrections to the leading operator, not alternative theories.

The canonical TEP matter coupling fixes the temporal sector. The present paper derives the strong-field solution requirements: the Schwarzschild incompatibility (Section 3) forces gravitational backreaction, and the leading curvature operator supplies it (Section 4). The regularising nonlinear coefficients are selected by global regularity and observation. What is under construction is the global field solution of this fixed theory — not a choice among competing theories.

## 2.2 Frame Dictionary and Local Lorentz Limit

| Quantity | Einstein/gravitational frame | Causal matter frame |
| --- | --- | --- |
| Metric | $g_{\mu\nu}$ | $\tilde g_{\mu\nu}$ |
| Propagation | tensor modes | matter, photons, ideal clocks |
| Proper time | $d\tau_g$ | $d\tilde\tau$ |
| Conservation | total Einstein-frame stress tensor | $\tilde\nabla_\mu\tilde T^{\mu\nu}=0$ |
| Validity condition | Lorentzian $g$ | Lorentzian, invertible $\tilde g$ |

At every regular point of the matter frame, Riemann normal coordinates give $\tilde g_{ab}=\eta_{ab}+O(x^2)$, preserving exact local Lorentz invariance and locally invariant $c$. Because the metric is globally non-degenerate, such local inertial frames exist everywhere for all $r > 0$.

## 2.3 Spherical Geometry and Scalar Ansatz

A regular Eddington–Finkelstein chart is used from the beginning, avoiding reliance on Schwarzschild $(t,r)$ coordinates in the deep region. The general static spherical form is

\begin{equation} \label{eq:tep_3}
ds_g^2 = -F(r)\,dv^2 + 2\,G(r)\,dv\,dr + R^2(r)\,d\Omega^2,
\end{equation}

in Eddington-Finkelstein form. This prevents coordinate artifacts from being mistaken for physical freezing. In the exterior, $F(r) = 1 - 2M/r$, $G(r) = 1$, $R(r) = r$, recovering the standard Schwarzschild chart at large $r$; this is a matching convenience, not an event horizon in the TEP solution.

The most general symmetry-compatible scalar profile is

\begin{equation} \label{eq:tep_4}
\phi(v,r) = q\,v + \psi(r).
\end{equation}

For the static case studied here, $q = 0$. Two scalar profiles are used in this paper, serving distinct roles in the argument:

- For the Schwarzschild incompatibility analysis (Section 3), the scalar is prescribed as $\phi(r) = \phi_0\,\ln(r/r_h)\,S(r)$ with a smooth logistic activation $S(r)$ that suppresses the field in the exterior. This prescribed profile is used only to establish the incompatibility: the geometric metric is fixed to Schwarzschild, and the scalar field is prescribed rather than solved from coupled equations.

- For the perturbative sGB exterior branch (Section 4), the scalar is solved from the sGB field equations and is Coulomb-like ($\phi \sim Q_s/r$) in the exterior, with the geometric metric dynamically modified by the temporal field's backreaction.

These are not two competing models. The first is a proof device for the Schwarzschild incompatibility; the second is the leading curvature operator of the strong-field EFT, demonstrating that the backreaction channel is physical. Both are calculations within the fixed TEP theory — the canonical action, the universal conformal coupling, the bounded disformal rule, and the environmental screening law of Section 2.1. A time-dependent scalar with static stress-energy is possible in shift-symmetric constructions and may be essential for rotating generalisations; this is an extension of the scalar ansatz within the same theory, not a replacement of the theory.

## 2.4 Complete Matter Metric and Signature

Expanding the disformal transformation gives $d\tilde s^2 = A^2\,ds_g^2 + B(\psi'(r)\,dr)^2$. The two-dimensional $(v,r)$ determinant is $\det \tilde g_{2D} = -A^4 G^2 - A^2 B F \psi'^2$. The disformal coupling uses an ultra-damped quartic Gaussian ansatz (Appendix B) chosen to ensure conformal dominance in the deep interior: $A^4$ dominates at all radii, so $\det \tilde g_{2D} < 0$ for all $r > 0$.

### Proposition 1 (Invertibility and Signature)

For the bounded strong-field ansatz, the matter metric is Lorentzian and nondegenerate for all $r > 0$. The determinant remains strictly negative as $r \to 0$, where conformal dominance holds asymptotically. The $r \to 0$ limit is an asymptotic spatially enlarged end, not an ordinary manifold point. Full parameter values and the determinant calculation are in Appendix B.

## 2.5 Screening and Temporal Topology

Canonical TEP does not identify screening with a density switch. It represents observable suppression by the environmental operator

\begin{equation} \label{eq:tep_10}
\Sigma_\mu^{\rm obs}=\mathcal S_\Sigma(\mathcal E)\,\Sigma_\mu,
\end{equation}

where $\mathcal E$ includes source structure, compactness, gradients, boundary conditions, and coherence scale. Temporal Topology denotes the continuous spatial and covariance structure of $\ln A$ and its temporal shear, not a discrete change of spacetime topology. The logistic $S(r)$ used in the Schwarzschild incompatibility analysis is a phenomenological radial activation compatible with this vocabulary; the coupled solution does not require $S(r)$ because the scalar profile is determined by the field equations.

## 2.6 The Observational Inference Problem

Because astrophysical mass is an inferred rather than direct measurement, its derivation depends on the assumed time-transfer map. The mass is reconstructed from observed angular positions, spectral shifts, signal periods and image scales under the assumption that source clocks, photon propagation and observer clocks can be mapped onto a single universal time coordinate. This paper identifies this assumption as the *Isochrony Axiom* — the hidden closure in black-hole inference. The TEP framework demonstrates that this reduction is not theory-neutral.

What the telescope directly records is functions of the observer's local proper time $\tilde\tau_o$:

\begin{equation} \label{eq:tep_isochrony_1}
\mathcal D_{\rm obs} = \left\{\theta_x(\tilde\tau_o),\; \theta_y(\tilde\tau_o),\; \nu_o(\tilde\tau_o),\; F_o(\tilde\tau_o)\right\},
\end{equation}

whereas the source trajectory is parameterised by a different local proper time $\tilde\tau_s$. The full source-to-observer map is

\begin{equation} \label{eq:tep_isochrony_2}
\tilde\tau_s \;\longrightarrow\; x^\mu_s(\tilde\tau_s) \;\longrightarrow\; k^\mu \;\longrightarrow\; \tilde\tau_o,
\end{equation}

with observed frequency $\omega_o = -k_\mu u_o^\mu$ and source-to-observer timing map

\begin{equation} \label{eq:tep_isochrony_3}
\frac{d\tilde\tau_o}{d\tilde\tau_s} = \mathcal T_{s\to o}\!\left[\phi,\, A,\, B,\, x_s^\mu,\, u_s^\mu,\, x_o^\mu,\, u_o^\mu\right].
\end{equation}

The standard analysis reconstructs orbital period, velocity, radius and mass by effectively setting $\mathcal T_{s\to o}$ to its GR or isochronous value. The familiar relation

\begin{equation} \label{eq:tep_isochrony_4}
M_{\rm app}^{\rm GR} = \frac{4\pi^2 a_{\rm GR}^3}{G P_{\rm GR}^2}
\end{equation}

therefore returns an apparent mass conditioned on that temporal calibration. It does not independently measure either the local material mass or the proper volume of the source region. The inferred mass is

\begin{equation} \label{eq:tep_isochrony_5}
M_{\rm app}^{\rm GR} = \mathcal F\!\left[\theta(\tilde\tau_o),\, \nu_o(\tilde\tau_o),\, D_{\rm assumed},\, \mathcal T_{s\to o}^{\rm GR},\, g_{\rm GR}\right],
\end{equation}

not a direct measurement of matter. The spectroscopic velocity has the same structure. The observed spectral shift is conventionally decomposed as $z_{\rm obs} = z_{\rm Doppler} + z_{\rm gravitational} + z_{\rm transverse} + \cdots$ and used to infer a radial velocity. But under TEP,

\begin{equation} \label{eq:tep_isochrony_6}
1 + z_{\rm obs} = \frac{(-k_\mu u^\mu)_e}{(-k_\mu u^\mu)_o}
\end{equation}

also contains the temporal-field transfer. Some quantity interpreted as $v_{\rm radial}$ may actually contain motion plus clock-rate difference plus open-path temporal transport. If that temporal component is forced into a Doppler model, it changes the inferred orbital speed and hence the inferred mass. Even the Galactic-centre distance, though largely inferred by combining angular astrometry with spectroscopic velocities rather than from a standard candle, still assumes $v_{\rm transverse} = D\,d\theta/dt$ and compares it with a spectroscopic radial velocity measured in the same effective time standard. Under TEP the correct relation is schematically $v_{\rm transverse}^{\rm local} = D_{\rm TEP}\,(d\theta/d\tilde\tau_o)\,(d\tilde\tau_o/d\tilde\tau_s)$. Even "geometric" orbital parallax is conditional on the mapping between source time and observer time.

The Cepheid analogy is exact at the methodological level. The TEP cosmological paper treats Cepheids as environment-dependent clocks whose altered periods are misread through a universal period–luminosity law, causing distance and $H_0$ bias. The black-hole analogue is: orbital motion supplies a periodic clock; spectral lines supply atomic clocks; variable plasma features supply dynamical clocks; ringdown supplies gravitational clocks. Standard analysis assumes these clocks can be compared through one universal time coordinate. TEP says their observed periods can include $P_{\rm observed} = P_{\rm intrinsic} \times \mathcal T_{\rm path} \times \mathcal T_{\rm environment}$. If those temporal factors are instead attributed to motion or curvature, the inferred mass becomes inflated. This is the same structural mistake as treating Cepheid period contraction as luminosity or distance.

The direct observables are: changing angular positions; changing received spectral frequencies; repeated signal patterns; strong redshift; strong lensing and image distortion; a dark central observational domain. The inferred quantities — a universal source-frame orbital period, a theory-independent physical orbital velocity, a theory-independent linear radius, millions of solar masses of matter, a tiny proper volume, an event horizon, a singularity — are reconstructed through the isochrony axiom. The conventional inference chain is:

\begin{equation} \label{eq:tep_isochrony_7}
\begin{aligned}
&\text{received angles, phases and frequencies} \\
&\quad\Downarrow\quad \text{isochrony axiom + GR transfer + distance calibration} \\
&\text{orbital period, velocity and radius} \\
&\quad\Downarrow\quad \text{Kepler/GR inversion} \\
&\text{large compact mass } (M_{\rm app}^{\rm GR}).
\end{aligned}
\end{equation}

TEP replaces it with:

\begin{equation} \label{eq:tep_isochrony_8}
\begin{aligned}
&\text{received angles, phases and frequencies} \\
&\quad\Downarrow\quad \text{dynamical temporal-transfer reconstruction} \\
&\text{local trajectory + clock-rate field} \\
&\quad\Downarrow\quad \text{material source + temporal contribution.}
\end{aligned}
\end{equation}

The precise statement is not that conventional analysis sets $\mathcal T_{s\to o}=1$ — GR includes gravitational redshift, transverse and line-of-sight Doppler shift, propagation delay, and Shapiro delay. The TEP claim is that conventional analysis fixes the transfer map to its *GR closure*, $\mathcal T_{s\to o} = \mathcal T_{s\to o}^{\rm GR}$. TEP tests whether a reproducible residual temporal field remains after the complete GR mapping is applied:

\begin{equation} \label{eq:tep_Delta_T}
\Delta_T \equiv \ln\frac{\mathcal T_{s\to o}^{\rm TEP}}{\mathcal T_{s\to o}^{\rm GR}},
\end{equation}

Conventional inference is not clock-blind; it is closed under the GR transfer law. Whether $\Delta_T \neq 0$ is required by the data is the central empirical question of the TEP programme.

## 2.7 Strong-Field phantom mass and the Mass Dictionary

This paper defines the strong-field counterpart of phantom mass in the wider TEP framework. At galactic scales, unmodelled temporal shear is interpreted as a dark-matter halo. At black-hole scales, extreme temporal transport can be interpreted as a very large central mass, rapid infall and a compact causal object. The conventional mass may combine a material contribution, temporal-field energy, and a non-isochronous calibration residual. The more exact expression is

\begin{equation} \label{eq:tep_phantom_2}
M_{\rm app}^{\rm GR} = \mathcal F\!\left(M_{\rm matter},\, M_\phi,\, M_{\rm int},\, A,\, B,\, \mathcal T_{s\to o},\, D_{\rm TEP}\right),
\end{equation}

where $M_{\rm matter}$ is the locally measured material content, $M_\phi$ is the temporal-field energy, $M_{\rm int}$ is the interaction energy, and $A, B, \mathcal T_{s\to o}, D_{\rm TEP}$ are the temporal-field profile and transfer functions. The strong-field phantom mass residual is *defined* as the difference between the conventional fitted mass and the TEP material inference:

\begin{equation} \label{eq:tep_phantom_3}
M_{\rm phantom}^{T} \equiv M_{\rm fit}^{\rm GR} - M_{\rm matter}^{\rm TEP}.
\end{equation}

This is a residual definition, not a derived decomposition. The sign of $M_{\rm phantom}^{T}$ is not assumed; it is determined by the data through the sign equation of Section 2.8. The schematic additive form $M_{\rm app}^{\rm GR} = M_{\rm local}^{\rm TEP} + M_{\rm phantom}^{T}$ is shorthand for the residual definition, valid only when $M_\phi$ and $M_{\rm int}$ are absorbed into $M_{\rm phantom}^{T}$.

The paper uses $M$ in several distinct roles, and the distinction matters once the mass itself is in question. The strict dictionary is:

| Symbol | Meaning |
| --- | --- |
| $M_{\rm GR-fit}$ | The parameter returned by a conventional GR reduction of the observations |
| $M_g$ | The asymptotic gravitational charge associated with $g_{\mu\nu}$ |
| $M_{\rm matter}$ | The locally measured material content $M_{\rm matter} = \int_\Sigma \tilde T_{\mu\nu}\tilde u^\mu\tilde u^\nu\,d\tilde V$ |
| $M_\phi$ | The temporal-field energy or effective contribution |
| $M_{\rm phantom}^{T}$ | The difference between the conventional fitted mass and the TEP material inference |
| $M$ | The GR-calibrated exterior mass parameter used in the conditional sGB benchmark (Sections 4–8) |

The statement that a fixed ADM $M$ is "the physical mass measured by a distant observer" is not made in this paper. Under the paper's own thesis, that is precisely what the TEP programme determines. The sGB calculations of Sections 4–8 use $M$ as the GR-calibrated exterior mass parameter; their observable predictions are conditional on that calibration. The central empirical task is to determine how much of the conventional central mass survives as locally measured material content when the Isochrony Axiom is removed. The decisive analysis is a *TEP-native refit of the raw observations*, not a perturbation of a pre-assumed Schwarzschild mass.

## 2.8 Sign and Identifiability of Strong-Field phantom mass

A direct intuition might suggest that because deep clocks run slow, observed periods are longer, so the inferred mass is larger. This overlooks the spatial magnification that accompanies the temporal stretching, and recognising the interplay between the two is the first gate of the entire analysis.

Consider a source orbiting in the deep region with local orbital period $P_s$ and local semi-major axis $a_{\rm local}$. A distant observer measures period $P_o$ and infers radius $a_{\rm GR}$. The temporal transfer stretches the period:

\begin{equation} \label{eq:tep_sign_1}
P_o = \mathcal T_P \, P_s, \qquad \mathcal T_P = \frac{d\tilde\tau_o}{d\tilde\tau_s} > 1.
\end{equation}

The spatial calibration relates the GR-inferred orbit to the local orbit:

\begin{equation} \label{eq:tep_sign_2}
a_{\rm GR} = \mathcal S_a \, a_{\rm local},
\end{equation}

where $\mathcal S_a$ encodes lensing magnification, distance miscalibration, and orbital-dynamics modification — all sourced by the temporal field. The conventional Kepler inversion gives:

\begin{equation} \label{eq:tep_sign_3}
M_{\rm app}^{\rm GR} = \frac{4\pi^2 a_{\rm GR}^3}{G P_o^2} = \frac{4\pi^2 \mathcal S_a^3 a_{\rm local}^3}{G \mathcal T_P^2 P_s^2}.
\end{equation}

The local mass, using the local period and scale, with a possible dynamical-law modification $\mathcal D_{\rm dyn}$ ($\mathcal D_{\rm dyn} = 1$ for Newtonian), is:

\begin{equation} \label{eq:tep_sign_4}
M_{\rm local}^{\rm TEP} = \frac{4\pi^2 a_{\rm local}^3}{G P_s^2 \, \mathcal D_{\rm dyn}}.
\end{equation}

The ratio is:

\begin{equation} \label{eq:tep_sign_5}
\boxed{\frac{M_{\rm app}^{\rm GR}}{M_{\rm local}^{\rm TEP}} = \frac{\mathcal S_a^3 \, \mathcal D_{\rm dyn}}{\mathcal T_P^2}.}
\end{equation}

If the spatial scale is held fixed ($\mathcal S_a = 1$, $\mathcal D_{\rm dyn} = 1$), then $M_{\rm app} = M_{\rm local} / \mathcal T_P^2 < M_{\rm local}$. *Slow deep clocks alone deflate the inferred mass*, producing $M_{\rm phantom}^{T} < 0$. This is the opposite of the phantom mass claim. Positive phantom mass requires the spatial magnification to dominate:

\begin{equation} \label{eq:tep_sign_6}
M_{\rm phantom}^{T} > 0 \quad \Longleftrightarrow \quad \mathcal S_a^3 \, \mathcal D_{\rm dyn} > \mathcal T_P^2.
\end{equation}

Physically, spatial magnification arises because the conformal factor $A(\phi)$ rescales the matter metric relative to the geometric engine. A distant observer using a fixed angular scale and a GR-based distance calibration measures an enlarged effective orbital radius, while the local material mass needed to support that orbit is unchanged. The temporal field therefore inflates the inferred mass when $A(\phi) > 1$ is strong enough to overcome the deflation from slow local clocks.

The spatial calibration $\mathcal S_a$ is not a free parameter. It is determined by photon propagation on $\tilde g_{\mu\nu}$ (lensing magnification), the GR-inferred distance $D_{\rm GR}$ (standard-candle or parallax calibration through the temporal field), and the orbital dynamics on the temporal-well geometry. The critical threshold is $\mathcal S_a > \mathcal T_P^{2/3}$. For moderate temporal transfer $\mathcal T_P \sim 10$, this requires $\mathcal S_a > 4.6$ — a strong but achievable lensing magnification near a strong-field source.

The velocity-based mass estimator gives a complementary relation. With $v_{\rm GR} = \mathcal S_v \, v_{\rm local}$ (spectroscopic velocity calibration):

\begin{equation} \label{eq:tep_sign_7}
\frac{M_{\rm app}^{\rm GR}}{M_{\rm local}^{\rm TEP}} = \mathcal S_v^2 \, \mathcal S_a \, \mathcal D_{\rm dyn}.
\end{equation}

The two estimators (Kepler and velocity) must agree in any consistent theory. Their joint constraint tightens the identifiability of $\mathcal S_a$ and $\mathcal T_P$.

Three outcomes are possible and the pipeline must be allowed to return any of them:

| Outcome | Condition | Interpretation |
| --- | --- | --- |
| $M_{\rm phantom}^{T} > 0$ | $\mathcal S_a^3 \mathcal D_{\rm dyn} > \mathcal T_P^2$ | Spatial magnification dominates. Conventional mass is inflated. $M_{\rm matter} < M_{\rm GR\text{-}fit}$. |
| $M_{\rm phantom}^{T} = 0$ | $\mathcal S_a^3 \mathcal D_{\rm dyn} = \mathcal T_P^2$ | Effects cancel. Conventional mass equals local material mass. TEP reduces to GR for mass inference. |
| $M_{\rm phantom}^{T} < 0$ | $\mathcal S_a^3 \mathcal D_{\rm dyn} < \mathcal T_P^2$ | Temporal stretching dominates. Conventional mass is deflated. $M_{\rm matter} > M_{\rm GR\text{-}fit}$. |

A pipeline structured to return only the first outcome would not constitute a test of the thesis. The TEP-native refit (Section 9, Appendix J) must determine the sign from the data, not assume it.

Two distinct hypotheses must be separated in the observational analysis. The *composition hypothesis* $H_{\rm composition}: M_{\rm matter} < M_{\rm GR\text{-}fit}$ states that the strong-field source contains less material mass than the conventional fit suggests. The *calibration hypothesis* $H_{\rm calibration}: M_g < M_{\rm GR\text{-}fit}$ states that the large exterior gravitational parameter itself is a calibration illusion. These are different claims: a temporal field can contribute real energy and backreaction, so TEP might find $M_{\rm matter} \ll M_{\rm GR\text{-}fit}$ while $M_g \sim M_{\rm GR\text{-}fit}$. In that case there is little compressed matter, but there remains a large physical temporal–geometric charge. That would still be a major TEP result, but it would not mean the entire dynamical mass was merely a data-reduction illusion. The raw-observation fit must test both separately.

A separate question is whether "finite density" at the regular centre implies "no extreme compression." A regular-centre geometry can still represent a physically ultra-dense source with extraordinarily high density. The intended TEP vision is stronger: no extreme local compression. This requires an explicit proper-volume diagnostic. Define the proper volume inside a boundary radius $r_b$:

\begin{equation} \label{eq:tep_proper_volume}
V_{\rm proper}(r_b) = 4\pi\int_0^{r_b} \sqrt{\tilde\gamma_{rr}}\,\tilde R^2(r)\,dr,
\end{equation}

and the local mean density $\bar\rho_{\rm local} = M_{\rm matter} / V_{\rm proper}(r_b)$. The critical diagnostic is the volume ratio:

\begin{equation} \label{eq:tep_CV}
\mathcal C_V \equiv \frac{V_{\rm proper}}{(4\pi/3)\,r_{\rm app}^3},
\end{equation}

where $r_{\rm app}$ is the externally inferred apparent radius. A value $\mathcal C_V \gg 1$ would demonstrate that the externally compact image corresponds to a much larger local proper volume — apparent compactness without physical compression. Without this calculation, "apparent compactness $\neq$ physical compression" remains an interpretation rather than a demonstrated property. The proper-volume diagnostic is a required output of the TEP-native refit.

# 3. Why Schwarzschild Is Not Enough

A natural question: why can TEP not simply add a temporal field onto ordinary Schwarzschild geometry? This section answers that question. The geometric metric is held fixed to Schwarzschild, and the conformal factor takes the power-law form $A = (r_h/r)^{\phi_0}$ in the deep interior. The question is whether the conformal matter metric can simultaneously achieve curvature regularity and bounded areal radius. The answer is no. If the temporal field becomes strong enough to create a Temporal Horizon, the geometry cannot remain ordinary Schwarzschild. The geometry has to respond. The full disformal generalisation — classifying all allowed asymptotics of $A$, $B$, $\phi$ — is a separate mathematical programme; the conformal case already establishes the incompatibility.

## 3.1 The Compatibility Conditions

Consider the conformal matter metric $\tilde g = A^2 g_{\rm Schw}$ with $A = (r_h/r)^{\phi_0}$ in the deep interior, on a fixed Schwarzschild background. The exact Kretschmann scalar scales as

\begin{equation} \label{eq:fbt_1}
\tilde K \sim r^{4\phi_0 - 6}
\end{equation}

(Appendix D), not $r^{12\phi_0-6}$ as the naive conformal formula suggests — the derivative terms from the non-constant conformal factor dominate. The areal radius scales as

\begin{equation} \label{eq:fbt_2}
\rho = A\,r \sim r^{1-\phi_0}.
\end{equation}

Two conditions are required for a regular temporal well:

- Finite curvature ($\tilde K$ finite or vanishing) requires $\phi_0 \geq 3/2$. At $\phi_0 = 3/2$ the leading curvature is finite and constant; for $\phi_0 > 3/2$ the curvature vanishes.

- Bounded areal radius ($\rho$ finite) requires $\phi_0 \leq 1$.

These are mutually exclusive when the geometric metric is held fixed to Schwarzschild. Schwarzschild cannot support the temporal geometry that TEP requires.

## 3.2 The Limiting Cases

The limiting case $\phi_0 = 2$ achieves vanishing curvature: $\tilde K \sim 39r^2/16 \to 0$ (Appendix D). The matter metric is Lorentzian and nondegenerate throughout the open domain $r > 0$ ($\det\tilde g_{2D} < 0$ for all $r > 0$, since $\det\tilde g_{2D} = A^4 \det g_{2D}$ and $A > 0$). However, the areal radius diverges: $\rho \sim r_h^2/r \to \infty$. The $r \to 0$ limit is an asymptotic spatially enlarged end, not an ordinary manifold point. The conformal scaling stretches both time and space — the interior is spatially enlarged, not a black-hole interior.

The complementary limit $\phi_0 = 1$ keeps the areal radius bounded but the curvature diverges: $\tilde K \sim 9/r^2 \to \infty$. The singularity is not resolved.

This is a structural incompatibility within the tested class, not a failure of a particular choice: fixed Schwarzschild geometry cannot support both conditions simultaneously. A systematic gate search over the tested disformal parameter space is reported in Appendix D.

## 3.3 Physical Principle

The incompatibility is the mathematical expression of a physical principle: because gravity depends on how things move through time, changing time must also change gravity. The temporal field cannot sit inertly atop a Schwarzschild geometry; it must backreact on the geometric metric itself. The mutual exclusion of finite curvature and bounded areal radius is proved for conformal matter metrics on fixed Schwarzschild; full disformal asymptotics remain open and do not weaken the backreaction requirement for the clock sector. Within the tested class, a dynamical proper-time field necessarily implies a dynamical gravitational geometry. A fixed Schwarzschild singular geometry cannot satisfy the conditions for a physically admissible temporal well. Any viable strong-field realisation of TEP must arise from a dynamically coupled temporal–geometric solution. Section 4 establishes the required backreaction channel and the regular geometric architecture; their joint nonlinear realisation is selected by the TEP field equations.

## 3.4 Raychaudhuri Equation and the Convergence Condition

On the globally Lorentzian domain, a hypersurface-orthogonal timelike congruence obeys

\begin{equation} \label{eq:fbt_3}
\frac{d\tilde\theta}{d\tilde\tau}=-\frac13\tilde\theta^2-\tilde\sigma_{\mu\nu}\tilde\sigma^{\mu\nu}-\tilde R_{\mu\nu}\tilde u^\mu u^\nu.
\end{equation}

The sign of $\tilde R_{\mu\nu}\tilde u^\mu u^\nu$ is the geometric statement of the timelike convergence condition. On a regular background with a temporal-rate minimum, the effective pressure is negative in the causal matter frame, and the convergence condition fails over the interior domain. This is the Hawking–Penrose escape clause: with the convergence condition violated, the singularity theorems' conclusion of inevitable geodesic incompleteness no longer applies to $(\mathcal M, \tilde g)$. In a two-metric theory, the Penrose–Hawking hypotheses must be checked separately for $(\mathcal M,g)$ and $(\mathcal M,\tilde g)$; incompleteness of Schwarzschild $g$ does not logically prove incompleteness of $\tilde g$. The analysis of this section shows that $\tilde g$ on fixed Schwarzschild cannot be both curvature-regular and spatially bounded; the regular-geometry benchmark (Section 5, Appendix K) shows that $\tilde g$ on a regular background can achieve finite curvature and bounded areal radius.

# 4. Existence Demonstrations

The Schwarzschild incompatibility result determines the admissible class of strong-field solutions: the temporal field must backreact on the geometric metric, and the resulting geometry must simultaneously achieve finite matter-frame curvature, continuous temporal transport, real exterior hair, and dynamical backreaction. The Schwarzschild obstruction is not a general geometric impossibility. Regular geometry, bounded matter-frame curvature, and dynamical scalar backreaction are each explicitly attainable — established separately by the two demonstrations below. Their joint realisation in one dynamically generated solution is the nonlinear closure selected by the TEP action.

## 4.1 Regular Geometry Exists

The Hayward metric $F(r) = 1 - 2Mr^2/(r^3 + 2M\ell^2)$ is a regular geometric background: the Kretschmann scalar is finite everywhere, $K_{\rm Hayward}(0) = 24/\ell^4$, and the effective energy density is finite, $\rho_{\rm eff} \to 3/\ell^2$ at the centre. The radial equation of state is $w_r = -1$ — cosmological-constant-like — with anisotropic tangential pressure. This profile is the signature of a higher-curvature coupling: a scalar field's stress-energy, modified by the Gauss-Bonnet invariant, can in principle produce anisotropic effective pressure with a de Sitter-like radial component.

The demonstration proves that singularity is optional, not mandatory. Once the geometric singularity is absent, the TEP matter metric simultaneously achieves bounded areal radius and finite Kretschmann (Section 5, Appendix K). The burden shifts: GR must justify why the singular reconstruction is mandatory, not why TEP must reproduce every deep-region coordinate chart. The Hayward metric is one example of a regular geometry; it is not the TEP-selected solution.

## 4.2 Dynamical Coupling Exists

The scalar-Gauss-Bonnet (sGB) coupling $\alpha_{\rm GB}\,\phi\,\mathcal{G}$ is the leading curvature operator of the strong-field EFT (Section 2.1). It is the lowest-dimension higher-curvature term that couples the temporal field to the gravitational geometry and supplies real backreaction. The published perturbative solution (Sotiriou \& Zhou 2014, exact $\mathcal{O}(\alpha^2)$) establishes:

- Scalar hair around strong-field sources — the no-hair theorem is evaded because the Gauss-Bonnet coupling provides a geometric source term.

- Real backreaction on the gravitational metric — the temporal field is not epiphenomenal.

- Sector-dependent observables — the shadow is sensitive to the geometric metric at $\mathcal{O}(\eta^2)$ while the ISCO feels the conformal factor at $\mathcal{O}(\eta)$, because null geodesics are conformally invariant while timelike geodesics are not (Appendix L).

What sGB does *not* establish: a regular deep region. The nonlinear solutions of Sotiriou \& Zhou (2014) for the same linear sGB coupling develop a finite-area singularity rather than a regular temporal minimum, and recent simulations (Thaalba et al. 2024) confirm this and explore a possible connection to loss of hyperbolicity. The TEP-selected solution must satisfy the minimal temporal-well criteria: $0 < N(r)$ and $N_{\min} \ll N_o$ everywhere, with no observer-independent one-way boundary. The leading curvature operator supplies the backreaction channel; the regularising nonlinear coefficients that complete the deep-region profile are selected by the joint requirements of global regularity and observation. This is the construction programme — solving the fixed theory, not choosing among theories.

## 4.3 The Inverse Reconstruction

Computing the effective stress-energy required to produce a Hayward-like regular metric identifies $w_r = -1$ as a sufficient regularity target — one sufficient condition, not a uniqueness theorem. The effective energy density is

\begin{equation} \label{eq:coupled_8}
\rho_{\rm eff} = -G^t_t = \frac{12M^2\ell^2}{(r^3 + 2M\ell^2)^2},
\end{equation}

finite at the centre with $w_r = -1$. This is the stress-energy profile that a regular temporal well *would* require; it does not prove that standard linear sGB produces it. The full nonlinear integration is the decisive test. The detailed sGB action, perturbative metric corrections, matter-metric construction, and exterior observable benchmarks are in Appendices B and L.

## 4.4 Status

The Schwarzschild obstruction is not a general geometric impossibility. Regular geometry, bounded matter-frame curvature, and dynamical scalar backreaction are each explicitly attainable. Their joint realisation is the nonlinear closure selected by the TEP action. The canonical matter coupling fixes the temporal sector; the leading curvature operator supplies backreaction; the regularising nonlinear coefficients are selected by global regularity and observation. The construction programme is solving the global field equations of this theory — not selecting among competing theories. The theory is carried by the principle, the action, and the field equations; the global solution is what is under construction.

# 5. Global Geometry and Curvature Regularity

The existence demonstrations of Section 4 require a regular geometric background. This section confirms that finite curvature, bounded areal geometry, and Lorentzian signature can coexist on a regular background. The physical Temporal Well additionally requires $N_{\min}\ll N_o$. Three properties are established; detailed calculations are in Appendix K.

## 5.1 Three Results

- **Global Lorentzian signature.** The radial-block determinant $\det\tilde g_{2D} = -A^4 G^2 - A^2 B\,F\,(\phi')^2$ is strictly negative for all $r > 0$. The determinant remains strictly negative throughout the tested domain, establishing a globally Lorentzian benchmark without a finite-radius signature boundary. This benchmark tests geometric regularity; the physical Temporal Well requires the separate condition ($N_{\min}\ll N_o$).

- **Bounded curvature.** The Kretschmann scalar approaches a finite value at the centre, $\tilde K \to 8/r_h^4$, in contrast to the Schwarzschild value $K_{\rm Schw} = 48\,M^2/r^6 \to \infty$. All curvature invariants remain finite throughout.

- **Bounded areal radius.** The areal radius $\rho = A\,r$ remains finite as $r \to 0$. On a Schwarzschild background the conformal scaling drives $\rho \to \infty$, illustrating the Schwarzschild incompatibility (Section 3): curvature regularity and bounded areal radius are mutually exclusive when the geometric metric is held fixed. On the regular background both conditions are satisfied simultaneously.

## 5.2 Frame Dependence of Density

The classical density estimate $M/(4\pi r^3/3)$ presupposes that the geometric metric governs matter. In a two-metric theory the physical density is defined from the matter stress tensor and a spacelike hypersurface with induced metric $\tilde\gamma_{ij}$: $d\tilde V = \sqrt{\det\tilde\gamma_{ij}}\,d^3x$, $\tilde\rho = \tilde T_{\mu\nu}\,u^\mu u^\nu$. The frame-dependence of volume is a structural feature of TEP: the classical density inference assumes the geometric metric governs matter, which is precisely what TEP questions. The geometric Einstein-tensor component, the matter-frame stress, and the TEP matter stress tensor are distinct objects and must be separated explicitly (Appendix K).

## 5.3 Status

These results establish that finite curvature, bounded areal geometry, and Lorentzian signature can coexist once the fixed Schwarzschild singularity is removed. The nonlinear TEP solution combines these geometric conditions with the defining slow-clock condition ($N_{\min}\ll N_o$). The scalar-on-regular-background benchmark (Appendix K) confirms that the sGB scalar field on a regular background produces a strictly positive lapse, vanishing areal radius at the centre, and Lorentzian signature throughout. The specific numerical values are properties of the benchmark; the structural conclusion is general.

# 6. Causal Structure

The event horizon is not an observable. It is a global causal construct inferred from the spacetime model. No telescope has ever measured an event horizon. What is observed is that signals from certain regions become extremely redshifted, delayed, and faint. The Temporal Horizon is the operational boundary defined directly from observable temporal accessibility. The causal structure of TEP is fundamentally different from the standard black-hole picture: the standard "black hole" is replaced by the *temporal well* — a regular spatial region in which the rate of proper time differs radically from the exterior (Section 1.1). *Darkness does not require an event horizon; it requires sufficiently extreme temporal decoupling.*

## 6.1 The Operational Temporal Horizon

The Temporal Horizon was introduced in the cosmological context (Thika) as the conformal-temporal boundary where $A_{\rm clock} \to 0$. This paper extends the concept to the strong-field regime. The shared name marks a shared mechanism — clock-transport as boundary — not a shared definition. In Thika the Temporal Horizon is the past conformal boundary $\mathscr{T}^-$ where $A_{\rm clock} \to 0$. Here it is the observer-relative accessibility set $\mathcal{H}_T^{(\Lambda)}[u_o] = \{x_e : \mathcal{T}_{e\to o} \geq \Lambda\}$. Cosmological $\mathscr{T}^-$ is not the black-hole surface $\mathcal{H}_T$; both are reconstructions of dynamical proper time. Around a compact source, the relevant boundary is an observer-relative accessibility threshold — the surface beyond which the clock-transfer factor renders signals practically undetectable. The physical lapse $N(r) = \sqrt{-\tilde g_{\mu\nu}\xi^\mu\xi^\nu}$ satisfies $0 < N(r) < 1$ through the deep region, with $N(r) \ll 1$ where the object appears black, but $N(r) \neq 0$ at every finite physical location. The TEP-selected solution must have $N(r) \geq N_{\min} > 0$ everywhere — the lapse has a finite minimum at the temporal-rate minimum, never vanishing at any finite radius. The perturbative sGB benchmark belongs to a horizon-bearing branch that does not realise the temporal well; the full TEP action must produce $N(r) \geq N_{\min} > 0$ everywhere.

The Temporal Horizon is defined by the clock-transfer factor between an emitter $e$ and observer $o$:

\begin{equation} \label{eq:temporal_horizon_def}
\mathcal{T}_{e\to o} = \frac{d\tilde\tau_o}{d\tilde\tau_e} = \frac{\omega_e}{\omega_o} = \frac{(-k_\mu u^\mu)_e}{(-k_\mu u^\mu)_o},
\end{equation}

where $k_\mu$ is the signal wavevector and $u^\mu$ is the observer's four-velocity. For stationary observers with matter-frame lapse $N(x)$, the Killing energy is conserved, giving $\mathcal{T}_{e\to o} = N_o/N_e$. The maximum transfer factor from the temporal-rate minimum is finite: $\mathcal{Z}_{\max} \sim N_o/N_{\min}$ — potentially astronomically large but never divergent. The object appears black because the transfer factor exceeds any practical accessibility threshold, not because signals are absolutely forbidden from escaping. The operational Temporal Horizon at threshold $\Lambda$ is:

\begin{equation} \label{eq:operational_TH}
\mathcal{H}_T^{(\Lambda)}[u_o] = \left\{x_e \in J^-(\gamma_o) : \mathcal{T}_{e\to o} \geq \Lambda\right\}.
\end{equation}

This boundary is not null, not one-way, and does not divide spacetime into an outside and an inside. It changes with the observer's proper-time rate, motion, and worldline: $\mathcal{H}_T^{(\Lambda)}[u_1] \neq \mathcal{H}_T^{(\Lambda)}[u_2]$. The black-hole observables used here are open-path transfer factors $\mathcal{T}_{e\to o}$; they do not require residual closed-loop synchronization holonomy, which remains a disformal/Jakarta programme probe. A freely falling observer moving through the deep temporal region experiences ordinary local space, a normally running local clock, no wall, no horizon crossing, no signature change. The distant observer's Temporal Horizon is not their Temporal Horizon.

## 6.2 Transfer, Refraction, and Darkness

The unifying mechanism is the coordinate propagation rate. In any local Lorentzian frame of $\tilde g_{\mu\nu}$, the locally measured speed of light is exactly $c$ — no local rule is broken. But the physical lapse $N(r)$ drops to an extremely small fraction in the deep temporal region, so from the perspective of a distant fast-clock observer, the coordinate propagation rate of light through that region is drastically slowed: $dr/dt \sim N(r) \ll 1$. This single mechanism — light propagating through a gradient where the rate of time varies — produces three observational consequences that the standard picture attributes to three separate phenomena:

- **Transfer.** The clock-transfer factor $\mathcal{T}_{e\to o} = N_o/N_e$ grows as the emitter moves deeper, reaching a finite maximum $\mathcal{Z}_{\max} \sim N_o/N_{\min}$. When this exceeds any practical threshold, the object appears dark. Darkness is extreme temporal decoupling, not an absorbing boundary.

- **Refraction.** The slowing bends trajectories: light curves when the propagation rate varies through a medium, exactly as in refraction, producing the apparent "inward pull" and gravitational lensing without a sideways force. The conformal factor $A(\phi)$ controls clock-rate transfer and massive-particle response; the disformal term $B(\phi)$ can alter causal cones; the backreacted geometric metric $g_{\mu\nu}[\phi]$ generates light bending and orbital curvature. Lensing is generated by the complete matter metric $\tilde g_{\mu\nu}$, not by conformal clock scaling alone.

- **Extended transit.** Waves crossing the deep region take far longer in coordinate time — the deep-transit ringdown modes show extended damping because the wave transits a zone where proper time runs at a small fraction of the exterior rate (Section 7).

Photons, massive particles, and gravitational waves are all governed by the same unified temporal field — they are different observational probes of the same temporal well, not evidence of fractured metric structure. In the perturbative sGB benchmark, photons and massive particles respond to the temporal field at different coupling orders (null geodesics are conformally invariant, timelike geodesics are not), but this is a property of the mathematical toolkit, not a fundamental law of TEP. Detailed null-expansion calculations, Eddington–Finkelstein null algebra, and the invariant frequency-transfer derivation are in Appendix B.

## 6.3 Minimal Temporal-Well Criteria

A regular-centre profile satisfying the minimal temporal-well criteria approaches, as $r \to 0$:

\begin{equation} \label{eq:causal_6}
F(r) = 1 - f_2 r^2 + O(r^4), \quad R(r) = r + O(r^3), \quad \phi(r) = \phi_c + \phi_2 r^2 + O(r^4),
\end{equation}

\begin{equation} \label{eq:causal_7}
A(\phi_c) = A_c, \quad 0 < A_c < \infty, \quad B(\phi_c) \text{ finite}.
\end{equation}

Then the physical areal radius becomes $\tilde R = A(\phi)R(r) \sim A_c r \to 0$ — an ordinary regular centre with no spatial opening, no tube-like asymptotic end, no other universe, no wall, and finite local geometry. Both null families have finite affine parameter. The centre is at finite optical distance. An emitter at the centre has finite $\mathcal{Z}$ with any causally connected receiver. The temporal-rate minimum need not coincide with a centre — it may occur at a shell or an asymmetric basin — but the invariant is the same: $N_{\min} > 0$ with finite transfer mismatch to any receiver. The TEP object is a temporal well, not a regular-centre black hole.

\begin{equation} \label{eq:causal_8}
\begin{aligned}
&\text{regular Lorentzian geometry} + \text{finite local density} \\
&\qquad + 0 < N(r) \text{ everywhere with } N_{\min} \ll N_o \\
&\qquad + \text{no observer-independent one-way boundary } (N \geq N_{\min} > 0) \\
&\qquad + \text{extreme external redshift}
\end{aligned}
\end{equation}

The Schwarzschild incompatibility analysis (Section 3) shows that the temporal field must backreact on the geometric metric: the fixed-Schwarzschild construction cannot work. The inverse reconstruction (Section 4) identifies $w_r = -1$ as a sufficient regularity condition, demonstrated by the regularity benchmark (Appendix K). The global solution of the fixed TEP theory — the canonical action with the leading sGB curvature operator and the regularising nonlinear coefficients selected by global regularity — must produce a temporal well: a regular spatial domain with finite local density and an extreme temporal-rate gradient, not a regular black hole. The sGB perturbative exterior produces $F = 0$ at a shifted $r_H$ because it is an EFT approximation around the Schwarzschild background; the TEP-selected solution must have $N(r) \geq N_{\min} > 0$ everywhere. The observable predictions (shadow, ISCO, QNM) are computed at $r \sim 3$–$6M$ in the conditional sGB benchmark, far from the would-be horizon at $r \sim 2M$; their robustness to the deep-region profile requires the TEP-native solution. The horizonless temporal well alters the late-time ringdown spectrum (Section 7).

The organising question of TEP is not "How is a black-hole interior regularised?" but rather: *can the phenomena attributed to black holes arise from an extreme relative-time gradient without physical collapse into an ultra-dense object?* The shadow is reproduced by the photon sphere (at $r \sim 3M$, independent of the horizon). The ISCO is reproduced by the orbital structure (at $r \sim 6M$, independent of the horizon). The gravitational-wave ringdown is reproduced by the perturbation spectrum, with a theory-dependent inner boundary condition. The darkness is reproduced by the extreme redshift of signals from the deep temporal region. The apparent inward pull is reproduced by the complete matter metric $\tilde g_{\mu\nu}$. The apparent compactness is reproduced by the exterior observer's reconstruction through strongly redshifted, delayed signals. No event horizon is required, no ultra-dense object is required, and no suction mechanism is required — not a better black hole, but an explanation of why a regular temporal domain is mistaken for one. The duality with cosmological expansion is established in Section 1.5.

In the minimal temporal-well picture, the observer falling toward the deep region experiences ordinary local space, a normally running local clock, finite local density, no wall, no horizon crossing, no signature change. Matter need not be compressed into a vanishing proper volume; the apparent concentration can result from temporal decoupling between the deep region and the exterior. The extreme effect is a finite but potentially enormous transfer mismatch: $\mathcal{Z}_{\max} \sim N_o/N_{\min}$, finite because $N_{\min} > 0$. The signature structure ($\det\tilde g_{2D} = -A^4 G^2 - A^2 F B(\phi')^2 < 0$) is established in Section 2.4 and Appendix B. The divergent-$A$ benchmark ($\phi_0 = 1$) produces a finite-area asymptotic end rather than a regular centre; it serves as a validation benchmark and cautionary example, detailed in Appendix K.

# 7. Perturbation Analysis: Ringdown Structure

TEP replaces the purely ingoing event-horizon condition with propagation through a regular temporal domain. Therefore its late-time spectral problem is not the GR black-hole spectral problem. This is the theory-level structural prediction, independent of the specific coupling. The altered late-time spectrum is an unavoidable consequence. Whether the altered spectrum exhibits longer or shorter damping, weak or strong reflection, echoes or no resolvable echoes depends on the effective potential, characteristic speeds, interior transmission, scalar coupling, boundary regularity, and excitation coefficients of the specific geometry.

## 7.1 The Structural Prediction

In the standard black-hole picture, a single-peaked potential with a purely absorbing horizon produces no echoes. In the TEP temporal-well picture, there is no horizon and no physical boundary — the deep region is continuous, ordinary space in which the rate of proper time drops to a finite minimum. The wave does not strike a wall; it encounters an extreme continuous gradient in the temporal rate. The gradient may turn the wave refractively — the propagation direction curves smoothly as the local clock slows — and the wave may transit through the deep region and emerge back into the exterior. Whether a resonant cavity forms, whether the transit produces longer or shorter damping, and whether scalar–tensor mixing is enhanced or suppressed are all determined by the specific geometry.

The structural prediction is categorical: the boundary condition is changed, and therefore the late-time spectrum is altered. The benchmark calculation on a prescribed Hayward background (Section 7.2) isolates the geometric mechanism: longer damping and amplified isospectrality breaking are the realised behaviour of the present deep-transit benchmark. The nonlinear TEP solution determines the physical realisation.

## 7.2 Benchmark Calculations

Three layers of QNM calculation are reported, each on a distinct geometry. **(1) Fixed-Schwarzschild baseline:** the geometric metric is held at Schwarzschild; gravitational tensor QNMs are the GR baseline, with no TEP shift (Appendix G). **(2) Perturbative sGB exterior:** the Sotiriou–Zhou metric perturbation introduces scalar hair and backreaction; the full coupled axial operator requires derivation from the second variation of the action, and the sign and coefficient of the QNM shift are not determined by the horizon displacement alone (Appendix L). **(3) Prescribed Hayward deep-transit model:** an eigenvalue problem on a horizonless regular geometry, isolating the geometric mechanism of the altered spectral problem. The deep-transit modes find longer damping and amplified isospectrality breaking — the realised behaviour of the present benchmark. The nonlinear TEP solution determines the physical QNM spectrum. The sGB exterior potentials, shadow/ISCO shifts, and QNM horizon-displacement estimates are in Appendix L; the deep-transit eigenvalue calculation is described in this section and documented in the pipeline (Appendix J).

## 7.3 Hyperbolicity and Characteristics

The disformal matter metric must have a well-posed initial-value formulation. The radial-block determinant is strictly negative in the exterior, confirming Lorentzianity, invertibility, and non-degeneracy (Section 2.4, Appendix B). The tensor speed $c_T = 1$ on the Schwarzschild background benchmark, satisfying the GW170817 constraint. The full tensor characteristic metric on the nonlinear TEP solution, global ghost freedom, and scalar sector hyperbolicity in the deep interior require derivation from the complete coupled perturbation system — an open problem, not yet contained in any appendix.

# 8. Four Consequences of One Principle

The principle is that the source-to-observer temporal transfer is a dynamical degree of freedom. The transfer factor

\begin{equation} \label{eq:pred_transfer}
\mathcal{T}_{e\to o} = \frac{\omega_e}{\omega_o} = \frac{(-k_\mu u^\mu)_e}{(-k_\mu u^\mu)_o}
\end{equation}

is the single quantity from which all observational consequences follow. Four consequences, one field:

- **Time transfer** — signals are redshifted, delayed, and diluted by the transfer factor.

- **Mass inference** — the conventional mass is reconstructed under isochrony; a non-isochronous refit yields a phantom mass residual.

- **Photon accessibility** — the shadow and photon region are projections of the transfer field; the Temporal Horizon is the observer-relative accessibility boundary.

- **Ringdown** — replacing an absorbing horizon with a continuous, strongly time-dilated region changes the boundary-value problem in principle.

These are not separate mechanisms bolted onto a GR background. They are different projections of one field. The sGB coupling provides the realised benchmark for the first three; the fourth is a structural prediction independent of the specific coupling. Detailed numerical tables are in Appendix L; the main text retains only the structural points.

## 8.1 Time Transfer

When $|\Delta_T| = |\ln(\mathcal{T}_{e\to o}^{\rm TEP}/\mathcal{T}_{e\to o}^{\rm GR})|$ becomes extreme, signals are strongly redshifted, observed processes appear greatly slowed, and signal arrival rates become extremely slow. The object appears dark because the transfer exceeds any practical accessibility threshold, not because signals are absolutely forbidden. The transfer factor from the temporal-rate minimum is finite: $\mathcal{Z}_{\max} \sim N_o/N_{\min}$, where $N_{\min} \ll N_o$. This may be astronomically large but is not divergent.

In the sGB benchmark, the conformal factor $A = e^{-\phi}$ modifies clock rates at $\mathcal{O}(\eta)$. The tensor speed $c_T = 1$ on the Schwarzschild background in shift-symmetric sGB, satisfying the GW170817 constraint on that background. The Einstein-frame metric carries the gravitational dynamics; tensor propagation is determined by the principal symbol of the coupled temporal–geometric equations. The full tensor characteristic metric on the TEP solution requires derivation from the complete coupled perturbation system (Section 7).

## 8.2 Mass Inference

The conventional mass $M_{\rm app}^{\rm GR}$ is not directly weighed. It is reconstructed from observed angles, frequencies, and periods under the isochrony closure. The TEP phantom mass residual $M_{\rm phantom}^{T} \equiv M_{\rm fit}^{\rm GR} - M_{\rm matter}^{\rm TEP}$ measures the difference between the GR-reconstructed mass and the locally measured material content. Its sign is not assumed: slow deep clocks alone deflate the inferred mass; positive phantom mass requires spatial magnification to dominate temporal stretching (Section 2.8).

The sGB benchmark illustrates the mechanism. The mass-inflation branch ($\alpha_{\rm GB} < 0$) gives $A > 1$ in the exterior — the matter metric is conformally magnified, providing the conformal orientation required for mass inflation. Because pure conformal scaling cancels from null geodesic paths, $A > 1$ does not automatically imply observed angular-scale magnification $\mathcal S_a > 1$; positive phantom mass requires the joint photon–orbit–timing forward model to establish $\mathcal S_a^3 \mathcal D_{\rm dyn} > \mathcal T_P^2$ (Section 2.8). The magnitude scales with the dimensionless coupling $|\eta| = 3|\alpha_{\rm GB}|/M^2$, making corrections largest for low-mass stellar black holes. The fundamental coupling $\alpha_{\rm GB}$ is a dimensionful constant; a single value must be used across all objects. Detailed numbers are in Appendix L.

## 8.3 Photon Accessibility

The shadow and photon region are projections of the transfer field. In the sGB benchmark, the shadow is sensitive to the geometric metric at $\mathcal{O}(\eta^2)$ while the ISCO feels the conformal factor at $\mathcal{O}(\eta)$ — because null geodesics are conformally invariant while timelike geodesics are not. This coupling-order difference is a property of the sGB realisation; the nonlinear TEP solution determines the universal structure.

The Temporal Horizon $\mathcal{H}_T^{(\Lambda)}[u_o]$ is the observer-relative accessibility boundary. It is defined directly from the observable transfer factor, not from a global property of an assumed spacetime. Horizon-scale images constrain how far the photon-region geometry can differ from the GR-calibrated exterior; they do not directly establish an event horizon or measure the local density and proper volume of the obscured region. The pipeline implementation and data products for the visibility-domain fit are documented in Appendix J.

## 8.4 Ringdown

TEP replaces the purely ingoing event-horizon condition with propagation through a regular temporal domain. The late-time spectral problem is therefore not the GR black-hole spectral problem. This is the theory-level prediction. The altered late-time spectrum is an unavoidable consequence. Whether the altered spectrum exhibits longer or shorter damping, weak or strong reflection, echoes or no resolvable echoes depends on the specific geometry.

An altered late-time spectrum is universal to the TEP boundary condition. The calculation on a prescribed Hayward background (Section 7.2) isolates the geometric mechanism: longer damping, two preserved families, and amplified isospectrality breaking are the realised behaviour of the present deep-transit benchmark. The nonlinear TEP solution determines the physical spectrum. The deep-transit modes cannot be directly compared to published sGB QNMs because they are on a horizonless geometry while all published sGB QNMs are on horizon-bearing backgrounds.

## 8.5 Rotating Sources

The static, spherically symmetric ansatz is an imposed symmetry reduction, not a derived property of the theory. A real astrophysical source with angular momentum produces a temporal field that is neither spherical nor static. The slowly-rotating shift-symmetric sGB solution (Delgado et al. 2020) provides the $\mathcal{O}(\beta^2)$ correction to the frame-dragging function from the coupled field equations — an axisymmetric realisation that isolates the spin-coupling mechanism. The nonlinear TEP solution determines the physical rotating-source geometry. The shadow and QNM corrections from spin are documented in the pipeline (Appendix J).

## 8.6 Dynamical Signatures

A binary merger produces a highly asymmetric, dynamical temporal gradient. The coupled system contains axial, gravitational-led polar, and scalar-led temporal mode families; the ringdown can differ from GR because the temporal field participates in the coupled dynamics. During inspiral, the coupled theory permits a $-1$PN scalar dipole channel when the components carry unequal temporal charges — absent in GR because GR has no dynamical time field. These channels are identified as relevant signatures; their quantitative prediction requires the full coupled calculation.

## 8.7 Summary

Four observational consequences follow from one temporal field. The sGB coupling provides the realised benchmark for the first three; the fourth is a structural prediction. None of these is a pillar holding up the theory — they are consequences of the principle that proper time is dynamical. The decisive test is the raw-observable, non-isochronous refit (Section 9).

# 9. Interpretation and Scope

TEP treats proper time as a dynamical field. The strong-field consequence is the temporal well. This section collects the interpretation, confronts existing data as examples of the framework, and defines the scope.

## 9.1 Derived, Demonstrated, Decisive Closure

**Derived from TEP — follows from the principle.**

- The temporal-well interpretation: a black hole is a regular spatial region in which the rate of proper time differs radically from the exterior, without physical collapse, without a singular core, without an absolute one-way boundary.

- The operational Temporal Horizon: the observer-relative accessibility boundary defined directly from observable temporal accessibility.

- Non-unique GR-closed inverse mapping: the conventional black-hole reconstruction is not the unique reading of the same observations once the Isochrony Axiom is dropped.

- Fixed-Schwarzschild incompatibility in the defined conformal class: finite curvature and bounded areal radius are mutually exclusive when $g_{\mu\nu}$ is held fixed.

- Altered horizonless ringdown boundary condition: TEP replaces the purely ingoing event-horizon condition with propagation through a regular temporal domain; the late-time spectral problem is therefore not the GR black-hole spectral problem.

**Constructively demonstrated — attainability proofs and consistency checks.**

- Regular geometry exists: the Hayward metric achieves finite Kretschmann, bounded areal radius, and Lorentzianity simultaneously (Section 5, Appendix K).

- Scalar hair and backreaction exist: the sGB coupling produces real scalar hair and real backreaction on the gravitational metric (Section 4, Appendix L).

- Finite-curvature matter metrics are attainable: the scalar-on-regular-background benchmark produces a regular point centre (Section 4, Appendix K).

- Weak-field S2 recovers GR — a required consistency gate, not a null result (Section 9.3, Appendix J).

- Exterior and deep-transit benchmarks: the sGB shadow/ISCO shifts and the prescribed Hayward deep-transit eigenmodes isolate the geometric mechanism of the altered spectral problem (Sections 7–8, Appendices G, L).

**Decisive closure — the nonlinear selection problem.**

- One global TEP solution: solving the coupled Einstein–scalar–sGB system with the canonical matter coupling to obtain the minimal temporal-well profile.

- Complete characteristics: the full coupled tensor, scalar, and mixed-sector perturbation system on the TEP-selected geometry.

- Raw multi-messenger inference: the non-isochronous refit of the same data GR already uses.

- Measured phantom-mass sign: the joint photon–orbit–timing forward model that fixes $\mathcal S_a^3 \mathcal D_{\rm dyn}$ relative to $\mathcal T_P^2$.

The distinction between what is derived from the TEP principle and what requires explicit construction makes the framework more robust, not less: the conceptual claims cannot be dismissed by finding an issue in one calculation. Bahrain fixes the governing principle, action structure, and global solution conditions. The nonlinear field equations select the realised temporal-well geometry and its observable spectrum. The remaining nonlinear integration does not decide whether the paradigm exists; it selects the unique quantitative realisation of the paradigm already fixed by TEP. The strongest route forward is: one action, one exterior coupling, one global solution, one coupled characteristic system, one observational likelihood, one phantom mass posterior.

## 9.2 Data as Examples

Existing data are not pillars holding up the theory. They are examples of the framework applied to real observations. Three examples are considered.

## 9.3 S2: Weak-Field Consistency

S2 orbits Sgr A* at pericentre $r \sim 1369\,R_s$ — far outside the deep temporal region. Consistency with GR at this scale is required, not disappointing. TEP is a strong-field theory; weak-field recovery is a passed consistency gate. S2 validates weak-field recovery and the non-isochronous inference pipeline; it does not determine the horizon-scale Phantom Mass sign. The fitted coupling is consistent with zero at weak-field precision. The decisive test requires horizon-scale observations. The detailed 7-step pipeline, GR fit, and MCMC posterior are in Appendix J.

## 9.4 EHT: Photon-Region Constraint

Horizon-scale images constrain how far the photon-region geometry can differ from the GR-calibrated exterior. They do not directly establish an event horizon or measure the local density and proper volume of the obscured region. The visibility-domain fit to EHT M87$^\ast$ and Sgr A$^\ast$ data constrains deviations in the photon-region transfer model; the pipeline implementation and data products are documented in Appendix J. The epistemological point is that a horizon-scale image is a constraint on the photon-region transfer field, not a direct observation of an event horizon.

## 9.5 Gravitational Waves: The Strongest Test

TEP replaces the purely ingoing event-horizon condition with propagation through a regular temporal domain. The late-time spectral problem is therefore not the GR black-hole spectral problem — this is the theory-level prediction, independent of the specific coupling. The altered spectrum is an unavoidable consequence; its specific form (longer or shorter damping, echoes or no resolvable echoes, weak or strong scalar–tensor mixing) depends on the geometry. Gravitational-wave ringdown is the strongest test because it directly probes the boundary condition that distinguishes a temporal well from a black hole.

## 9.6 The Unified Statement

*Cosmological expansion and black-hole collapse are dual misreadings of dynamical proper time.* On cosmological scales, the conformal volume element $V_{\rm eff} = A_{\rm clock}^3 a_m^3$ tends to zero while the matter-frame geometry does not collapse (Thika); temporal evolution is interpreted as increasing spatial separation. Around extreme gravitational sources, temporal gradients are interpreted as spatial attraction, compression and causal capture. In both cases, TEP replaces apparent spatial dynamics with the observable consequences of a dynamical proper-time field.

The object is a regular region in which the rate of proper time differs radically from the exterior — not a drain or a singularity, but a continuous spatial domain with an extreme temporal gradient. Matter moving into that gradient appears, from the exterior, to accelerate inward, become increasingly redshifted, slow and fade, collect within a small apparent radius, and become inaccessible. But locally there is no mysterious suction. Matter follows its ordinary local trajectory through a continuous physical environment. The apparent force is analogous to refraction: trajectories curve when the propagation rate varies through a medium, not because matter is being pulled sideways.

## 9.7 Scope

The derived, demonstrated, and decisive-closure claims are collected in Section 9.1. Bahrain defines the nonlinear closure problem and the boundary conditions that uniquely identify the physical strong-field branch. The nonlinear integration selects the realised temporal-well geometry; it does not decide whether the paradigm exists.

# 10. Conclusion

The Temporal Equivalence Principle treats proper time as a dynamical field. This paper develops its strong-field consequence: a black hole, under TEP, is a temporal well — a regular spatial region in which the rate of proper time differs radically from the exterior, without physical collapse to an ultradense singular object and without an absolute one-way boundary. A continuous, extreme but finite gradient in proper-time rate is sufficient to produce the full observational phenomenology attributed to a black hole.

The operational boundary is the Temporal Horizon — the observer-relative threshold beyond which the clock-transfer factor renders signals practically undetectable — not the event horizon. The event horizon is a global causal construct inferred from the spacetime model, not an observable. Apparent compactness arises from exterior-frame reconstruction, not physical compression. Inferred gravitational mass and local material mass need not coincide; a strong-field phantom mass residual $M_{\rm phantom}^{T} \equiv M_{\rm fit}^{\rm GR} - M_{\rm matter}^{\rm TEP}$ measures this discrepancy, its sign determined by the data. Standard black-hole ontology assumes isochrony: source clocks, photon propagation, and observer clocks mapped onto a single general-relativistic time coordinate. TEP drops that closure. Under TEP, the conventional reconstruction — compact mass, event horizon, singular collapse — is no longer the unique reading of the same observations.

The temporal field cannot sit passively on fixed Schwarzschild geometry. Finite curvature and bounded areal radius are mutually exclusive when $g_{\mu\nu}$ is held fixed. Strong temporal structure forces gravitational backreaction. The canonical TEP matter coupling fixes the temporal sector; the leading curvature operator supplies backreaction; the regularising nonlinear coefficients are selected by global regularity and observation. The construction programme is solving the global field equations of the fixed theory — not selecting among competing theories.

Four observational consequences follow from one temporal field: time transfer, mass inference, photon accessibility, and ringdown. TEP replaces the purely ingoing event-horizon condition with propagation through a regular temporal domain; the late-time spectral problem is therefore not the GR black-hole spectral problem. This is the theory-level prediction. An altered late-time spectrum is universal to the TEP boundary condition. The benchmark calculation on a prescribed Hayward background isolates the geometric mechanism: longer damping and amplified isospectrality breaking are the realised behaviour of the present deep-transit benchmark. The nonlinear TEP solution determines the physical spectrum.

Weak-field data recover GR; horizon-scale images constrain the photon-region geometry but do not directly establish an event horizon; gravitational-wave ringdown provides the sharpest test. The decisive programme is the raw non-isochronous multi-messenger refit of the same data GR already uses.

Black holes are not fundamental collapsed objects in TEP. They are the observational appearance of regular strong-field temporal wells reconstructed under exterior temporal standards. Bahrain derives the governing strong-field conditions, proves fixed Schwarzschild insufficient, identifies the required backreaction channel, constructs regular benchmarks, and defines the decisive observational tests. The remaining nonlinear integration does not decide whether the paradigm exists; it selects the unique quantitative realisation of the paradigm already fixed by TEP.

*Cosmological expansion and black-hole collapse are dual misreadings of dynamical proper time.* The standard black-hole object was never what the telescopes measured. The event horizon, the singularity, and the ultradense core are GR reconstructions from observational data under an implicit isochronous transfer model. Under TEP, the same observations are produced by a temporal well — and the conventional reconstruction is no longer the unique reading.

# Appendix A — Conventions and Disformal Identities

This appendix fixes the sign, curvature and disformal conventions used throughout the paper. All identities quoted in the main text are derived here explicitly so that no sign or factor ambiguity survives into the field equations, the curvature invariants of Appendix D, or the perturbation analysis of Appendix G.

## A.1 Metric Signature and Curvature

The metric signature $(-,+,+,+)$ is adopted. The Riemann tensor is defined via the Wald convention

\begin{equation} \label{eq:appA_1}
({\nabla_a\nabla_b - \nabla_b\nabla_a)\,V_c = R_{abc}{}^{d}\,V_d,
\end{equation}

with Ricci tensor $R_{ab} = R_{acb}{}^{c}$ and Ricci scalar $R = g^{ab}R_{ab}$. The Einstein tensor is $G_{ab} = R_{ab} - \tfrac12 R\,g_{ab}$, and the Einstein equations read $G_{\mu\nu}[g] = 8\pi\,T_{\mu\nu}$ in geometrized units $G = c = 1$. With this convention the Schwarzschild Kretschmann scalar is

\begin{equation} \label{eq:appA_2}
K_{\rm Schw} = R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma} = \frac{48\,M^2}{r^6},
\end{equation}

which is the reference used for the curvature comparison in Appendix D.

## A.2 Disformal Transformation

The matter (causal) metric is the disformal image of the geometric metric,

\begin{equation} \label{eq:appA_3}
\tilde g_{\mu\nu} = A^2(\phi)\,g_{\mu\nu} + B(\phi)\,\nabla_\mu\phi\,\nabla_\nu\phi,
\end{equation}

where $A(\phi)$ is the conformal factor and $B(\phi)$ is the disformal coupling. Defining $X \equiv -\tfrac12\,g^{\mu\nu}\nabla_\mu\phi\,\nabla_\nu\phi$, the inverse matter metric is

\begin{equation} \label{eq:appA_4}
\tilde g^{\mu\nu} = A^{-2}\left(g^{\mu\nu} - \frac{B\,\nabla^\mu\phi\,\nabla^\nu\phi}{A^2 - 2BX}\right).
\end{equation}

This is the identity used in the geodesic equations of Appendix E and in the perturbation potentials of Appendix G.

## A.3 Determinant and Invertibility

In four spacetime dimensions the determinant of the disformal metric is

\begin{equation} \label{eq:appA_5}
\det\tilde g = A^{8}\,\det g\,\left(1 - \frac{2BX}{A^2}\right) = A^{8}\,\det g\,\frac{A^2 - 2BX}{A^2}.
\end{equation}

The transformation is therefore invertible if and only if

\begin{equation} \label{eq:appA_6}
A^2 > 0 \qquad\text{and}\qquad A^2 - 2BX \neq 0.
\end{equation}

The 2D $(v,r)$ determinant in Eddington–Finkelstein coordinates is the canonical expression $\det\tilde g_{2D} = -A^4 G^2 - A^2 F B(\phi')^2$ (Section 2.4). The geometric horizon has $\det\tilde g_{2D}=-1.002$ and is Lorentzian. The determinant is strictly negative at every sampled radius from the horizon to the innermost grid point ($r_{\min}=10^{-8}M$) in the exterior region ($F > 0$). Inside horizons ($F < 0$), the disformal term becomes positive and the determinant must be checked with the canonical formula. The disformal transformation is invertible where $A^2 - 2BX \neq 0$; this must be verified separately for the interior. For the $\phi_0 = 2$ fixed-Schwarzschild case, both null families have infinite affine parameter ($\tilde\lambda \sim \int r^{-4}\,dr \to \infty$) because $A \sim r^{-2}$; the $r \to 0$ limit is an asymptotic spatially enlarged end, not a regular point (Appendix E).

## A.4 Connection Difference

The Levi-Civita connection of $\tilde g$ differs from that of $g$ by a tensorial term. Writing $\Sigma_\mu \equiv \nabla_\mu\ln A$ and $\Phi_\mu \equiv \nabla_\mu\phi$, the difference is

\begin{equation} \label{eq:appA_7}
\tilde\Gamma^{\lambda}_{\;\;\mu\nu} - \Gamma^{\lambda}_{\;\;\mu\nu} = 2\,\delta^{(\lambda}_{(\mu}\,\Sigma_{\nu)} - g_{\mu\nu}\,\Sigma^{\lambda} + \frac{B}{A^2-2BX}\Big[\Phi_{\mu}\,\Phi_{\nu}\,\Sigma^{\lambda} + 2\,\Phi^{(\lambda}\,\nabla_{\mu)}\Phi_{\nu)} - g_{\mu\nu}\,\Phi^{\lambda}\,\Sigma_{\rho}\Phi^{\rho}\Big] + \mathcal{O}(\nabla B),
\end{equation}

where $\mathcal{O}(\nabla B)$ collects terms proportional to $\nabla_\mu B$. The conformal piece (the first two terms) is the standard scalar-tensor result; the disformal piece (the bracket) is what produces the modified causal structure and the shear zone in the interior.

## A.5 Stress-Energy Transformation

Matter minimally coupled to $\tilde g_{\mu\nu}$ has stress tensor $\tilde T^{\mu\nu} = -(2/\sqrt{-\tilde g})\,\delta S_m/\delta\tilde g_{\mu\nu}$. When expressed with respect to the geometric metric $g_{\mu\nu}$, this becomes

\begin{equation} \label{eq:appA_8}
T^{\mu\nu} \equiv \frac{2}{\sqrt{-g}}\frac{\delta S_m}{\delta g_{\mu\nu}} = A^{-2}\,\tilde T^{\mu\nu} + \frac{B}{A^2}\,\tilde T^{\alpha\beta}\,\nabla_\alpha\phi\,\nabla_\beta\phi\,g^{\mu\nu} + \text{disformal trace corrections}.
\end{equation}

The first term is the familiar conformal rescaling; the second is the disformal correction that feeds back into the geometric Einstein equations $G_{\mu\nu}[g] = 8\pi\,T_{\mu\nu}$. Conservation is frame-consistent: $\tilde\nabla_\mu\tilde T^{\mu\nu} = 0$ in the matter frame, which maps to a non-standard conservation law in the geometric frame that is sourced by the disformal gradient $\nabla_\mu B$.

## A.6 Summary of Identities

| Object | Expression |
| --- | --- |
| Inverse matter metric | $\tilde g^{\mu\nu} = A^{-2}\big(g^{\mu\nu} - \frac{B\,\nabla^\mu\phi\,\nabla^\nu\phi}{A^2 - 2BX}\big)$ |
| Determinant (4D) | $\det\tilde g = A^{8}\,\det g\,\frac{A^2 - 2BX}{A^2}$ |
| Invertibility | $A^2 > 0$ and $A^2 - 2BX \neq 0$ |
| Lorentzian signature | $\det\tilde g_{2D} = -A^4 G^2 - A^2 F B(\phi')^2 < 0$ in exterior ($F > 0$); must be checked inside horizons |
| Connection split | $\tilde\Gamma - \Gamma = $ conformal $+$ disformal |
| Stress split | $T^{\mu\nu} = A^{-2}\tilde T^{\mu\nu} + $ disformal corrections |

The disformal metric is invertible and Lorentzian in the exterior region ($F > 0$), with $\det\tilde g_{2D} < 0$ at every sampled radius. The pipeline-verified conditions $A^2 > 0$ and $A^2 - 2BX \neq 0$ hold at every sampled radius in the exterior. Inside horizons ($F < 0$), the disformal term $-A^2 F B(\phi')^2$ becomes positive and can oppose the conformal term; the determinant must be checked with the canonical formula. The interior hyperbolicity depends on the outcome of the scalar-on-regular-background benchmark (Section 4, Appendix K).

# Appendix B — Strong-Field EFT Action and Reduced Field Equations

This appendix records the strong-field EFT action, the reduced radial system obtained by substituting the spherical ansatz of Section 2.3, and the parameter set that defines the solutions integrated by the pipeline. The equations are written in geometrized units $G = c = 1$.

## B.1 Strong-Field EFT Action and Frame Assignment

The strong-field TEP action used by Bahrain is the canonical TEP action (Section 2.1) augmented by the leading curvature operator — the shift-symmetric scalar–Gauss–Bonnet coupling:

\begin{equation} \label{eq:appB_1}
S_{\rm BH} = \int d^4x\,\sqrt{-g}\left[\frac{R[g]}{16\pi} - \tfrac12(\nabla\phi)^2 - V(\phi) + f(\phi)\,\mathcal{G}\right] + S_m[\tilde g_{\mu\nu},\Psi_m],
\end{equation}

where $\mathcal{G} = R^2 - 4R_{\mu\nu}R^{\mu\nu} + R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma}$ is the Gauss–Bonnet invariant, $f(\phi) = \eta\,\phi$ is the shift-symmetric coupling, and the matter metric is

\begin{equation} \label{eq:appB_2}
\tilde g_{\mu\nu} = A^2(\phi)\,g_{\mu\nu} + B(\phi)\,\nabla_\mu\phi\,\nabla_\nu\phi.
\end{equation}

The terms are organised as follows. The Einstein–Hilbert term $R/(16\pi)$, the canonical scalar kinetic $-\frac12(\nabla\phi)^2$, the potential $V(\phi)$, and the conformal–disformal matter coupling $S_m[\tilde g]$ constitute the fundamental TEP sector (Section 2.1). The scalar–Gauss–Bonnet coupling $f(\phi)\,\mathcal{G}$ is the leading curvature operator of the strong-field EFT: it is the lowest-dimension higher-curvature term that couples the temporal field to the gravitational geometry and supplies real backreaction. The regularising nonlinear coefficients — higher-order curvature couplings and potential terms that control the deep-region profile — are EFT corrections to this leading operator, selected by the joint requirements of global regularity and observation (Section 4).

The dimensionless coupling is $\eta \equiv 3\alpha_{\rm GB}/M^2$; the mass-inflation branch uses $\eta = -0.1$ (Section 4, step_12 of the pipeline). In code units ($M=1$), $\alpha_{\rm GB} = \eta/3$. The perturbative parameter is $\zeta \equiv \alpha_{\rm GB}^2/M^4 = \eta^2/9 \ll 1$. The shift-symmetric Gauss–Bonnet coupling is $f(\phi)=\alpha_{\rm GB}\phi$, so $df/d\phi=\alpha_{\rm GB}$; the scalar equation therefore carries the source $-\alpha_{\rm GB}\mathcal G$. The potential is $V = 0$ for the sGB branch (massless scalar). The time-dependent generalisation $\phi(v,r) = qv + \psi(r)$ of Section 2.3 uses the same $f(\phi)$.

Variation yields the geometric Einstein equations (now including the Gauss–Bonnet stress-energy contribution), the scalar equation, and the matter conservation law in the matter frame:

\begin{equation} \label{eq:appB_3}
G_{\mu\nu}[g] = 8\pi\,T_{\mu\nu} + T_{\mu\nu}^{(\mathcal{G})}, \qquad \Box_g\phi = \frac{dV}{d\phi} - \frac{df}{d\phi}\,\mathcal{G} + \frac{d\ln A}{d\phi}\,T^{(\rm m)} + \frac{d\ln B}{d\phi}\,\mathcal{D}, \qquad \tilde\nabla_\mu\tilde T^{\mu\nu} = 0,
\end{equation}

where $T_{\mu\nu}^{(\mathcal{G})}$ is the Gauss–Bonnet stress-energy tensor obtained by varying $f(\phi)\mathcal{G}$ with respect to $g^{\mu\nu}$, and $T_{\mu\nu}$ includes the conformal and disformal corrections derived in Appendix A. The scalar equation now contains the Gauss–Bonnet source term $-(df/d\phi)\,\mathcal{G} = -\alpha_{\rm GB}\,\mathcal{G}$. This is the term that sources the scalar in vacuum: the Gauss–Bonnet invariant is nonzero wherever curvature is present, so the scalar is sourced by geometric curvature even when the matter trace $T^{(\rm m)}$ vanishes. The conformal coupling $(d\ln A/d\phi)\,T^{(\rm m)}$ sources the scalar through the matter trace and vanishes in vacuum; it does not provide the geometric vacuum source. The disformal source $\mathcal{D}$ is proportional to $T^{\mu\nu}\nabla_\mu\phi\nabla_\nu\phi$ and is finite in vacuum.

## B.2 Fixed-Background Prototype Ansatz

Horizon-regular Eddington–Finkelstein form is used throughout,

\begin{equation} \label{eq:appB_4}
ds_g^2 = -F(r)\,dv^2 + 2\,G(r)\,dv\,dr + R^2(r)\,d\Omega^2,
\end{equation}

with a static scalar profile $\phi = \phi(r)$ (the shift-symmetric $q\,v$ piece is set to zero for the vacuum solution). The metric functions and scalar field are parameterized as

\begin{equation} \label{eq:appB_5}
A(\phi) = e^{\beta_A \phi}, \qquad B(\phi) = B_0\,|\phi|^{2}/(1 + |\phi|^{2})\,\exp\!\left(-\frac{\phi^4}{2\sigma_B^4}\right), \qquad \phi(r) = \phi_0 \ln\!\left(\frac{r}{r_h}\right) S(r),
\end{equation}

where $S(r) = 1/(1 + e^{(r-r_h)/(\delta\, r_h)})$ is a smooth logistic activation, and $r_h = 2M$ is the geometric horizon. The exterior areal radius is $R(r) = r$, so that $F(r) = 1 - 2M/r$ recovers Schwarzschild exactly at large $r$.

## B.3 Schematic Prototype Equations

Substituting the ansatz, the schematic prototype is written in terms of the independent radial variables

\begin{equation} \label{eq:appB_6}
F(r), \qquad G(r), \qquad R(r), \qquad \phi(r).
\end{equation}

The coupling functions $A(\phi)$ and $B(\phi)$ are evaluated on the solved field profile and are not independent dynamical variables. The $vv$ and $vr$ Einstein components give two first-order constraints,

\begin{equation} \label{eq:appB_7}
\frac{F'}{F} = \frac{2M}{r^2 F} - \frac{2}{r}\left(1 - \frac{1}{G^2}\right) + 8\pi\,\mathcal{S}_{vv}[A,B,\phi],
\end{equation}

\begin{equation} \label{eq:appB_8}
G' = -4\pi\,r\,G^2\,\mathcal{S}_{vr}[A,B,\phi],
\end{equation}

where $\mathcal{S}_{vv}$ and $\mathcal{S}_{vr}$ are the disformal source terms built from $A$, $B$, $\phi'$ and $X = -\tfrac12 F(\phi')^2$. The angular equation fixes $R(r)$ through

\begin{equation} \label{eq:appB_9}
\frac{R''}{R} + \frac{2}{r}\frac{R'}{R} = 8\pi\,\mathcal{S}_{\theta\theta}[A,B,\phi].
\end{equation}

The scalar wave equation reduces to

\begin{equation} \label{eq:appB_10}
\frac{1}{r^2}\frac{d}{dr}\left(r^2\,F\,\phi'\right) = \frac{dV}{d\phi} - \alpha_{\rm GB}\,\mathcal{G}[g] + \frac{d\ln A}{d\phi}\,T^{(\rm m)} + \frac{d\ln B}{d\phi}\,\mathcal{D}[X,\phi],
\end{equation}

where the $-\alpha_{\rm GB}\,\mathcal{G}$ term is the Gauss–Bonnet geometric source, the $(d\ln A/d\phi)\,T^{(\rm m)}$ term is the conformal matter-trace source, and the $(d\ln B/d\phi)\,\mathcal{D}$ term is the disformal source. In vacuum ($T^{(\rm m)} = 0$), the scalar is sourced by the Gauss–Bonnet invariant: $\Box\phi = -\alpha_{\rm GB}\,\mathcal{G}$. For Schwarzschild, $\mathcal{G}_{\rm Schw} = 48M^2/r^6$, giving $\Box_{\rm Schw}\phi = -48\alpha_{\rm GB} M^2/r^6$. The conformal matter-trace source vanishes in vacuum; it is the Gauss–Bonnet coupling, not the conformal coupling, that provides the geometric vacuum source. This is the mechanism by which the scalar acquires hair around strong-field sources and evades the no-hair theorem (Section 4.2).

## B.4 Matching Conditions at the Geometric Horizon

The exterior Schwarzschild limit is approached smoothly at $r_h=2M$ by the logistic profile: $A\approx1$, $B\approx0$, and $\phi\approx0$. The clean run gives $A=1.0005$, $B=3.0\times10^{-7}$, and $\phi=-2.73\times10^{-4}$ at the sampled horizon grid point. This value of $A$ comes from the old logistic-screened fixed-background profile, not from the sGB-coupled exterior with $\beta_A = -1$. With the sGB scalar and $\beta_A = -1$, the mass-inflation branch has $A(2M) \approx 1.063$ for $\eta = -0.1$ (Section 4.3 assessment). The two profiles are different models and cannot be combined. The disformal contribution remains finite because $F\to0$. This is a continuity check for the prescribed ansatz, not a derivation of junction conditions for a fully coupled interior solution.

## B.5 Model Parameters

The pipeline uses the dimensionless field $\phi=\Phi/M_*$ and geometrized units $G=c=1$, followed by the numerical normalization $M=1$. Thus radii and curve parameters are reported in $M$, curvature-squared diagnostics in $M^{-4}$, and three-volumes in $M^3$. $A$ and $\phi$ are dimensionless; $B_0=1$ denotes $B_0=1\,M^2$ in the implemented metric convention, so that $B\,\partial\phi\,\partial\phi$ is dimensionless. The conformal coupling is $A(\phi) = A_0\,e^{\beta_A \phi}$ with $\beta_A = -1$ (the dimensionless strong-field coupling $d\ln A/d\phi$) frozen across all strong-field calculations; this corresponds to the Jakarta convention $A(\Phi) = \exp(\beta_A \Phi/M_{\rm Pl})$ with the identification $\phi = \Phi/M_*$ and $M_* = M_{\rm Pl}/|\beta_A|$. The weak-field corpus value $\beta \simeq -0.013$ (the PPN coupling $d\ln A/d\Phi \cdot M_{\rm Pl}$, consistent with Cassini PPN-$\gamma$ bounds) is a different normalisation of the same coupling; the two are connected by running with $\phi$ or temporal shear, which is part of the action.

Two solution branches are integrated by the pipeline, both within the fixed strong-field EFT action of equation \eqref{eq:appB_1}:

- **Fixed-background prototype** (step_01): the geometric metric is held fixed to Schwarzschild, and the scalar is prescribed as $\phi(r) = \phi_0\ln(r/r_h)\,S(r)$. This branch establishes the Schwarzschild incompatibility (Section 3) and validates the matter-metric signature/invertibility construction. It is not a coupled solution.

- **sGB self-gravitating branch** (step_12): the scalar is solved from the sGB-sourced equation $\Box\phi = -\alpha_{\rm GB}\,\mathcal{G}$ on the Schwarzschild background, and the metric is corrected at $\mathcal{O}(\alpha_{\rm GB}^2/M^4)$ (Sotiriou \& Zhou 2014). In code units ($M=1$) this is $\mathcal{O}(\eta^2)$. This branch establishes scalar hair and real backreaction (Section 4.2). The scalar charge is $Q_s = 2\alpha_{\rm GB}/3$ and the scalar profile is $\phi(r) = (2\alpha_{\rm GB}/3)(1/r + M/r^2 + 4M^2/(3r^3))$.

The parameter set for the fixed-background prototype is:

| Parameter | Symbol | Value | Role |
| --- | --- | --- | --- |
| Disformal amplitude | $B_0$ | $1.0$ | sets disformal coupling strength |
| Mass | $M$ | $1.0$ | geometrized black-hole mass |
| Conformal exponent | $\beta_A$ | $-1.0$ | controls $A(r)$ divergence in the interior |
| Disformal horizon exponent | $\delta$ | $0.05$ | controls $B\to0$ approach at $r_h$ |
| Disformal quartic width | $\sigma_B$ | $1.5$ | controls $B(\phi)$ suppression against violent Coulomb gradients |
| Scalar amplitude | $\phi_0$ | $2.0$ | normalizes the prescribed scalar profile |
| Geometric horizon | $r_h$ | $2.0\,M$ | Schwarzschild radius |

The additional parameter for the sGB self-gravitating branch is:

| Parameter | Symbol | Value | Role |
| --- | --- | --- | --- |
| sGB coupling | $\alpha_{\rm GB}$ | $\eta M^2/3$ | shift-symmetric GB coupling $f(\phi)=\alpha_{\rm GB}\phi$; coefficient of $\mathcal{G}$ in the scalar equation; mass-inflation branch |
| Dimensionless sGB coupling | $\eta$ | $-0.1$ | $\eta = 3\alpha_{\rm GB}/M^2$; mass-inflation branch |
| Dimensionless coupling | $\eta$ | $3\alpha_{\rm GB}/M^2 = -0.1$ | mass-inflation branch |
| Potential | $V(\phi)$ | $0$ | massless scalar in the sGB branch |
| Scalar charge | $Q_s$ | $2\alpha_{\rm GB}/3$ ($2\eta/3$ when $M=1$) | Coulomb-like $1/r$ falloff |

With $\beta_A=-1$ and $\phi_0=2.0$, $A(r)\sim(r/r_h)^{-2}$ in the deep interior, reaching $A\sim4.0\times10^{16}$ at the innermost sampled radius. The ultra-damped (quartic Gaussian) factor $B(\phi)=B_0|\phi|^2/(1+|\phi|^2)\exp(-\phi^4/(2\sigma_B^4))$ is suppressed by the quartic Gaussian envelope in the deep interior, driven below $10^{-300}$ at the innermost sampled radius. The conformal term $A^4$ dominates everywhere, so the determinant is strictly negative and the matter metric is globally Lorentzian with no boundary.

The fixed-background prototype (step_01) is a prescribed-profile metric construction, not a closed six-function radial ODE integration. Its supported result is that the bounded conformal-disformal ansatz is asymptotically Schwarzschild and Lorentzian in the exterior ($\det\tilde g_{2D} < 0$ for $F > 0$). The $\phi_0 = 2$ fixed-Schwarzschild case has vanishing curvature ($\tilde K \to 0$) but the $r \to 0$ limit is an asymptotic spatially enlarged end ($\rho \to \infty$) with both null families at infinite affine parameter — not a regular point (Appendix E). The sGB self-gravitating branch (step_12) solves the scalar from the GB-sourced equation and corrects the metric at $\mathcal{O}(\eta^2)$; it establishes hair and backreaction but not a regular deep region (Section 4.2). The joint realisation — regular geometry, bounded matter-frame curvature, and dynamical scalar backreaction in one solution — is the nonlinear closure selected by the TEP action. The minimal temporal-well criteria are given in Section 6.3.

# Appendix C — Strong-Field Asymptotic Expansion (the $\phi_0=2$ Case)

This appendix derives the interior scaling of the metric functions, the physical radial distance, and the areal radius for the $\phi_0 = 2$ case (curvature-regular but spatially-enlarged; see Appendix D for the Schwarzschild incompatibility result that establishes why this case does not satisfy the temporal-well criteria). All exponents and numerical values are taken directly from the pipeline integration of the reduced system in Appendix B.

#### Context

The $\phi_0 = 2$ case achieves $\tilde K \sim 39r^2/16 \to 0$ (curvature-regular) but has diverging areal radius $\rho \sim r_h^2/r \to \infty$ (spatially-enlarged, not a regular centre). The Schwarzschild incompatibility result (Appendix D) shows this is unavoidable for fixed Schwarzschild: curvature regularity ($\phi_0 > 3/2$) and bounded areal radius ($\phi_0 \leq 1$) are mutually exclusive. The validation benchmark achieves both by allowing the geometric metric to be regular.

## C.1 Interior Scaling

In the deep interior ($r \ll r_h = 2M$), the scalar field follows a logarithmic profile $\phi(r) = \phi_0 \ln(r/r_h)$ with $\phi_0 = 2.0$, and the conformal and disformal factors are

\begin{equation} \label{eq:appC_1}
A(\phi) = A_0\,e^{\beta_A \phi}, \qquad B(\phi) = B_0\,\frac{|\phi|^2}{1+|\phi|^2}\,\exp\!\left(-\frac{\phi^4}{2\,\sigma_B^4}\right),
\end{equation}

with $\beta_A = -1$, $\sigma_B = 1.5$, and disformal coupling $\delta = 0.05$. Substituting the logarithmic $\phi$, the effective radial scalings are

\begin{equation} \label{eq:appC_2}
A(r) \sim A_0\,(r/r_h)^{-2}, \qquad B(r) \to 0, \qquad \phi(r) \sim 2\ln(r/r_h).
\end{equation}

With $\beta_A=-1$ and $\phi_0=2$, the conformal factor diverges as $A\sim(r/r_h)^{-2}$. The disformal factor is suppressed by the quartic cutoff $\exp(-\phi^4/2\sigma_B^4)$: as $|\phi|\to\infty$ the logistic prefactor saturates to unity but the quartic exponential drives $B\to 0$ instantly. The net effect is that the disformal sector vanishes in the deep interior while the conformal sector diverges, establishing conformal dominance throughout the interior.

## C.2 Conformal Factor Profile

The pipeline integration gives the following values of $A(\phi)$ along the radial profile:

| Location | $r/M$ | $A$ |
| --- | --- | --- |
| Geometric horizon | $2.0$ | $1.0005$ |
| Innermost sampled | $10^{-8}$ | $4.0\times10^{16}$ |

The conformal factor is essentially unity at the geometric horizon (deviation $5.5\times10^{-4}$) and reaches $4.0\times10^{16}$ at the innermost sampled radius $r_{\min}=10^{-8}M$. The deep-interior scaling is the inverse-second-power law implied by $\phi_0=2$ and $\beta_A=-1$. Note: this table uses the old logistic-screened fixed-background profile, where $\phi \approx 0$ at the horizon and $A \approx 1$. This is a different model from the sGB-coupled exterior with $\beta_A = -1$, where the mass-inflation branch gives $A(2M) \approx 1.063$ for $\eta = -0.1$ (Section 4.3 assessment). The two profiles cannot be combined.

## C.3 Disformal Factor Profile

The corresponding values of $B(\phi)$ are:

| Location | $r/M$ | $B$ |
| --- | --- | --- |
| Geometric horizon | $2.0$ | $3.0\times10^{-7}$ |
| Innermost sampled | $10^{-8}$ | $<10^{-300}$ |

The disformal factor is $B=3.0\times10^{-7}$ at the geometric horizon and below $10^{-300}$ at the innermost sampled radius. Unlike a purely logistic activation, the quartic Gaussian cutoff $\exp(-\phi^4/2\sigma_B^4)$ drives $B$ to zero in the deep interior rather than saturating to $B_0$. The disformal sector is dynamically irrelevant in the interior; the geometry is governed entirely by the conformal factor.

## C.4 Conformal Dominance and the Absence of a Causal Boundary

The canonical determinant of the two-dimensional $(v,r)$ sector for $ds_g^2 = -F\,dv^2 + 2G\,dv\,dr + R^2\,d\Omega^2$ and a static radial scalar is

\begin{equation} \label{eq:appC_3}
\det\tilde g_{2D} = -A^4 G^2 - A^2 F\,B(\phi')^2.
\end{equation}

For the $\phi_0 = 2$ fixed-Schwarzschild case, $G = 1$ in EF coordinates. In the deep interior, $A\sim(r/r_h)^{-2}\to\infty$ while $B\to 0$ exponentially. The $-A^4 G^2$ term therefore dominates at every sampled radius, and the determinant is strictly negative:

\begin{equation} \label{eq:appC_4}
\det\tilde g_{2D}\big|_{\rm horizon} = -1.002, \qquad \det\tilde g_{2D}\big|_{r_{\min}} = -2.56\times10^{66}.
\end{equation}

The matter metric is Lorentzian and nondegenerate for all $r > 0$. However, a negative determinant for every $r > 0$ proves only that no finite positive radius is degenerate; it does not prove the limit $r = 0$ can be added as a regular point. The $r \to 0$ limit is an asymptotic spatially enlarged end ($\rho \sim r_h^2/r \to \infty$), not a regular manifold point. The determinant must be checked with the canonical formula inside horizons ($F < 0$), where the disformal term becomes positive.

## C.5 Areal Radius and Interior Volume Growth

The areal radius of the matter geometry is $\rho = A\,r$. With $A\sim(r/r_h)^{-2}$, this gives

\begin{equation} \label{eq:appC_5}
\rho \sim \frac{r_h^2}{r} \to \infty \quad\text{as}\quad r\to 0.
\end{equation}

The areal radius diverges: the physical space does not collapse. The constant-$r$ volume element of the matter metric is $d\tilde V/dv = 4\pi A^2 r^2 \sqrt{|F|}$. The pipeline gives

\begin{equation} \label{eq:appC_6}
\frac{d\tilde V}{dv}\bigg|_{0.1M} = 8.76\times10^{4}, \qquad \frac{d\tilde V}{dv}\bigg|_{r_{\min}} = 2.84\times10^{22}.
\end{equation}

Since the geometry is globally Lorentzian, this volume growth is a physical proper volume, not a formal diagnostic. The would-be central singularity of Schwarzschild is replaced by continuous, low-curvature regular space.

## C.6 Curvature Scaling (Corrected)

The exact Kretschmann scalar scales as $\tilde K \sim r^{4\phi_0 - 6} = r^{2}$ for $\phi_0 = 2$ (Appendix D), so the curvature vanishes as $r\to 0$. The pipeline confirms $\tilde K \approx 0.024$ at $0.1M$ and $\tilde K \approx 1.36\times10^{-16}$ at the innermost sampled radius, compared with $K_{\rm Schw} = 4.8\times10^{7}$ and $4.8\times10^{49}$ respectively. The exact formula $\tilde K \sim 39r^2/16$ matches the numerical values to within 4%. Note: the naive conformal formula $\tilde K = A^{-12} K_{\rm Schw} \sim r^{18}$ is incorrect — it omits the derivative terms from the non-constant conformal factor $A(r)$, which dominate for $\phi_0 < 3/2$ and contribute at the same order for $\phi_0 = 2$. The curvature vanishing is real, but the power is $r^2$, not $r^{18}$. Details are recorded in Appendix D.

## C.7 Summary of Asymptotic Behaviour

| Quantity | Scaling | Behaviour as $r\to0$ |
| --- | --- | --- |
| $A(r)$ | $\sim (r/r_h)^{-2}$ | diverges ($\to 4.0\times10^{16}$) |
| $B(r)$ | $\to 0$ (quartic Gaussian cutoff) | vanishes ($< 10^{-300}$) |
| $\phi(r)$ | $\sim 2\ln(r/r_h)$ | diverges logarithmically ($\to -38.2$) |
| $\det\tilde g_{2D}$ | $\sim -A^4 < 0$ | strictly negative (globally Lorentzian) |
| Areal radius $\rho = Ar$ | $\sim r^{-1}$ | diverges ($\to\infty$; spatially-enlarged — see Schwarzschild incompatibility result, App. D) |
| Kretschmann $\tilde K$ | $\sim r^{2}$ (exact; not $r^{18}$) | vanishes ($\to 0$; curvature-regular) |
| Interior volume | $\sim A^2 r^2$ | grows ($\to 2.84\times10^{22}$) |
| Singularity | — | curvature-regular, but areal radius diverges (the $\phi_0=2$ case) |

The interior scaling $A\sim(r/r_h)^{-2}$ with $B\to 0$ establishes conformal dominance: the determinant is strictly negative at every sampled radius, the matter geometry is globally Lorentzian, and the areal radius diverges as $\rho\sim 1/r$. The exact Kretschmann scalar vanishes as $\tilde K\sim r^{2}$ (not $r^{18}$ as the naive conformal formula suggests). The $\phi_0 = 2$ case is curvature-regular but spatially-enlarged; the Schwarzschild incompatibility result (Appendix D) shows this is unavoidable for fixed Schwarzschild, motivating the validation benchmark on a Hayward background.

# Appendix D — Curvature Invariants and the Schwarzschild Incompatibility

This appendix records the exact curvature analysis for the TEP matter metric. It is proved that the naive conformal formula $\tilde K = A^{-12} K_{\rm Schw}$ is incorrect for non-constant conformal factors, the exact Kretschmann $\tilde K \sim r^{4\phi_0 - 6}$ is derived, and the Schwarzschild incompatibility result for the fixed-Schwarzschild construction is established.

## D.1 The Exact Kretschmann Scalar

For the pure conformal matter metric $\tilde g = A^2 g_{\rm Schw}$ with $A = (r_h/r)^{\phi_0}$ in the deep interior ($r \ll r_h$, where $F \approx -2M/r$ and $B \to 0$), the Kretschmann scalar is computed in standard Schwarzschild coordinates $(t, r)$ where the metric is diagonal:

\begin{equation} \label{eq:appD_1}
\tilde g_{tt} = -A^2 F, \quad \tilde g_{rr} = \frac{A^2}{F}, \quad \tilde g_{\theta\theta} = A^2 r^2.
\end{equation}

The four independent orthonormal-frame Riemann components are:

\begin{equation} \label{eq:appD_2}
E = \frac{R_{trtr}}{\tilde g_{tt}\,\tilde g_{rr}}, \quad F_t = \frac{R_{t\theta t\theta}}{\tilde g_{tt}\,\tilde g_{\theta\theta}}, \quad F_r = \frac{R_{r\theta r\theta}}{\tilde g_{rr}\,\tilde g_{\theta\theta}}, \quad G = \frac{R_{\theta\phi\theta\phi}}{\tilde g_{\theta\theta}^2},
\end{equation}

and the Kretschmann scalar is $K = 4E^2 + 8F_t^2 + 8F_r^2 + 4G^2$.

Each component scales as $r^{2\phi_0 - 3}$ (verified by exact SymPy computation), giving:

\begin{equation} \label{eq:appD_3}
\boxed{\tilde K \sim r^{4\phi_0 - 6} \quad \text{as } r \to 0.}
\end{equation}

This is NOT $r^{12\phi_0 - 6}$ as the naive conformal formula $A^{-12} K_{\rm Schw}$ suggests. The conformal formula includes only the leading Schwarzschild term scaled by $A^{-12}$, omitting the derivative terms from the non-constant conformal factor $A(r)$. For $\phi_0 < 3/2$, the derivative terms ($r^{4\phi_0-6}$) dominate over the leading term ($r^{12\phi_0-6}$), and the conformal formula underestimates the curvature.

## D.2 Specific Values (Exact, $M = 1$)

| $\phi_0$ | $\tilde K$ (exact, deep interior) | $\tilde K \to 0$? | $\rho = Ar$ bounded? |
| --- | --- | --- | --- |
| $0.5$ | $\frac{75}{4r^4} + \cdots$ | No ($\to \infty$) | Yes |
| $1.0$ | $\frac{9}{r^2} + \frac{1}{4}$ | No ($\to \infty$) | Yes ($\rho \to r_h$) |
| $1.5$ | $\frac{291}{64}$ (constant) | No ($\to$ const) | No ($\rho \to \infty$) |
| $2.0$ | $\frac{39\,r^2}{16}$ | Yes ($\to 0$) | No ($\rho \sim r_h^2/r \to \infty$) |
| $2.5$ | $\frac{1323\,r^4}{1024}$ | Yes ($\to 0$) | No |

## D.3 The Schwarzschild Incompatibility Result

### Theorem D.1 (Fixed-Schwarzschild)

Within the conformal-disformal TEP framework with geometric metric fixed to Schwarzschild, curvature regularity and bounded physical areal radius are mutually exclusive.

#### Proof

Curvature regularity ($\tilde K \to 0$ as $r \to 0$) requires $4\phi_0 - 6 > 0$, i.e., $\phi_0 > 3/2$. Bounded areal radius ($\rho = A \cdot r = r_h^{\phi_0} r^{1-\phi_0}$ finite as $r \to 0$) requires $1 - \phi_0 \geq 0$, i.e., $\phi_0 \leq 1$. These conditions are mutually exclusive: there is no $\phi_0$ satisfying both $\phi_0 > 3/2$ and $\phi_0 \leq 1$. $\square$

## D.4 Numerical Verification

The exact formula $\tilde K \sim r^{4\phi_0 - 6}$ is verified numerically using the corrected pipeline curvature computation (standard Schwarzschild coordinates, 4-term Kretschmann formula). The power law is recovered to within 0.1% accuracy for all tested $\phi_0$ values:

| $\phi_0$ | Numerical power law | Expected $4\phi_0 - 6$ | Deviation |
| --- | --- | --- | --- |
| 0.50 | $r^{-4.000}$ | $-4.0$ | 0.000 |
| 0.75 | $r^{-3.000}$ | $-3.0$ | 0.000 |
| 1.00 | $r^{-2.001}$ | $-2.0$ | 0.001 |
| 1.25 | $r^{-1.001}$ | $-1.0$ | 0.001 |
| 1.50 | $r^{-0.001}$ | $0.0$ | 0.001 |
| 1.75 | $r^{0.999}$ | $1.0$ | 0.001 |
| 2.00 | $r^{1.999}$ | $2.0$ | 0.001 |
| 2.50 | $r^{3.999}$ | $4.0$ | 0.001 |

The Schwarzschild baseline ($\phi_0 \to 0$, $A \to 1$) recovers $K = 48M^2/r^6$ to within $5 \times 10^{-6}$ (0.0005%). All regression tests pass.

## D.5 Corrected Values for the $\phi_0 = 2$ Prototype

The $\phi_0 = 2$ case (curvature-regular but spatially-enlarged) has the following corrected curvature values, replacing the incorrect $\tilde K \sim r^{18}$ and $\tilde K \sim 10^{-150}$ from the naive conformal formula:

| Location | $r/M$ | $\tilde K$ (corrected) | $\tilde K$ (old, wrong) | $K_{\rm Schw}$ |
| --- | --- | --- | --- | --- |
| Exterior | $10$ | $4.79 \times 10^{-5}$ | $4.79 \times 10^{-5}$ | $4.79 \times 10^{-5}$ |
| Horizon | $2$ | $0.748$ | $0.752$ | $0.752$ |
| Deep interior | $0.1$ | $2.31 \times 10^{-2}$ | $\sim 10^{-150}$ | $4.79 \times 10^{7}$ |
| Innermost | $10^{-8}$ | $1.36 \times 10^{-16}$ | $\sim 10^{-300}$ | $4.80 \times 10^{49}$ |

The corrected values show $\tilde K$ is small but not absurdly tiny. The exact power law $\tilde K \sim 39r^2/16$ gives $\tilde K(0.1) = 0.024$, matching the numerical value $0.023$ to within 4%. The old values ($\sim 10^{-150}$) were artifacts of the incorrect conformal formula.

## D.6 Disformal Term: Status of the Classification

The disformal term $B\nabla\phi\nabla\phi$ contributes $B(\phi')^2$ to $\tilde g_{rr}$. In the deep interior, the Lorentzian condition $\det_{2d} < 0$ forces $B \to 0$ (for logarithmic $\psi$) or $\tilde g_{rr} \to 0$ (for saturating $\psi$). For the tested parameter space, the deep interior curvature is controlled by $A$ alone, and the conformal theorem applies. The full disformal classification — classifying all allowed asymptotics of $A$, $B$, $\phi$ under $\det\tilde g < 0$, $\tilde K < \infty$, and $\tilde R_{\rm area} < \infty$ — is an open mathematical programme (Section 3.3). The statement that the disformal term cannot help is established for the tested class but not as a general theorem.

## D.7 Implication: The Temporal Field Must Modify the Geometry

The Schwarzschild incompatibility result shows that the fixed-Schwarzschild construction is incompatible with the conditions for a physically admissible temporal well. The temporal field cannot merely sit on top of a Schwarzschild black hole — it must also change the black-hole geometry itself. When the geometric metric is allowed to be regular, all regularity conditions hold simultaneously.

# Appendix E — Geodesic Structure

This appendix records the null geodesic structure of the TEP matter metric. The key result is that for a conformal transformation $\tilde g = A^2 g$, null geodesics of $g$ are also null geodesics of $\tilde g$ (same unparameterised curves), but the affine parameters are related by $d\tilde\lambda = A^2\,d\lambda$. This single relation determines all affine-parameter results.

## E.1 Conformal Affine Parameter Relation

For $\tilde g_{\mu\nu} = A^2 g_{\mu\nu}$, a null geodesic $x^\mu(\lambda)$ of $g$ with tangent $k^\mu = dx^\mu/d\lambda$ satisfies $g_{\mu\nu}k^\mu k^\nu = 0$ and the geodesic equation $k^\nu\nabla_\nu k^\mu = 0$. Under the conformal transformation, the same curve is null ($\tilde g_{\mu\nu}k^\mu k^\nu = A^2 g_{\mu\nu}k^\mu k^\nu = 0$), but the affine parameter transforms as

\begin{equation} \label{eq:appE_1}
d\tilde\lambda = A^2\,d\lambda.
\end{equation}

This is the standard conformal affine-parameter relation. The Christoffel symbols transform as $\tilde\Gamma^\mu_{\alpha\beta} = \Gamma^\mu_{\alpha\beta} + \delta^\mu_\alpha \partial_\beta\ln A + \delta^\mu_\beta \partial_\alpha\ln A - g_{\alpha\beta}g^{\mu\nu}\partial_\nu\ln A$, and the geodesic equation in the rescaled parameter is satisfied.

## E.2 Eddington–Finkelstein Null Algebra

In ingoing Eddington–Finkelstein coordinates, the conformal matter metric ($B = 0$) has components

\begin{equation} \label{eq:appE_2}
\tilde g_{ab} = A^2\begin{pmatrix} -F & 1 \\ 1 & 0 \end{pmatrix}, \qquad \tilde g^{ab} = A^{-2}\begin{pmatrix} 0 & 1 \\ 1 & F \end{pmatrix}.
\end{equation}

Note that $\tilde g^{rr} = F/A^2$, not zero. The null condition $\tilde g_{ab}k^a k^b = 0$ gives $dv(-F\,dv + 2\,dr) = 0$, yielding two families:

- Ingoing ($v = \mathrm{const}$, $dv = 0$): the tangent $k^\mu = (0, 1)$. The Christoffel symbol $\tilde\Gamma^r{}_{rr} = 2A'/A$ is nonzero.

- Outgoing ($dr/dv = F/2$): the tangent $k^\mu = (2/F, 1)$, or $dv/dr = 2/F$.

## E.3 Affine Parameters on the Divergent-$A$ Benchmark

On the Hayward benchmark with $A = (r_h/r)^{\phi_0}$ and $\phi_0 = 1$, the conformal factor diverges as $A \sim r^{-1}$ near $r = 0$. The seed (Hayward) radial null affine parameter behaves as $d\lambda \propto dr$ (since $F \to 1$ and the background is regular). Applying the conformal relation:

\begin{equation} \label{eq:appE_3}
\tilde\lambda \sim \int A^2\,dr = \int \frac{dr}{r^{2\phi_0}}.
\end{equation}

For $\phi_0 = 1$: $\tilde\lambda \sim \int r^{-2}\,dr = -1/r \to \infty$ as $r \to 0$. Both radial null families have infinite affine parameter. The finite-area asymptotic end lies at infinite affine distance for both families. This is consistent with the end being an asymptotic boundary, not a regular manifold point.

For $\phi_0 = 2$: $\tilde\lambda \sim \int r^{-4}\,dr \to \infty$. Both families infinite.

For general $\phi_0 > 0$: both families have $\tilde\lambda \to \infty$. The divergence is driven by $A \to \infty$, not by the background geometry.

## E.4 Affine Parameters on the Correct Finite-$A$ Architecture

On the target global solution with $A \to A_c$ finite as $r \to 0$, the conformal relation gives $d\tilde\lambda \to A_c^2\,d\lambda$. Since the seed (regular) geodesic has $d\lambda \propto dr$ and finite affine parameter to $r = 0$, the matter-frame affine parameter is also finite: $\tilde\lambda \to A_c^2 \lambda_{\rm seed}$. Both null families reach the regular centre at finite affine distance. The centre is a regular manifold point; geodesics are extendable through it. This is geodesic completeness in the standard sense.

## E.5 The $\phi_0 = 2$ Case

For $\phi_0 = 2$ on fixed Schwarzschild, $A \sim r^{-2}$ and $\tilde\lambda \sim \int r^{-4}\,dr \to \infty$ for both families. The $r \to 0$ limit is an asymptotic spatially enlarged end ($\rho \sim r_h^2/r \to \infty$), not a regular point. The matter metric is Lorentzian and nondegenerate for all $r > 0$ ($\det\tilde g_{2D} = A^4 \det g_{2D} < 0$ since $A > 0$), but the limit $r = 0$ cannot be added as a regular point. The curvature vanishes ($\tilde K \sim 39r^2/16 \to 0$), but the areal radius diverges — the Schwarzschild incompatibility result (Appendix D) shows this trade-off is unavoidable for fixed Schwarzschild.

## E.6 Signature Structure

The canonical 2D $(v,r)$ determinant for $ds_g^2 = -F\,dv^2 + 2G\,dv\,dr + R^2\,d\Omega^2$ and a static radial scalar is:

\begin{equation} \label{eq:appE_4}
\det\tilde g_{2D} = -A^4 G^2 - A^2 F B(\phi')^2.
\end{equation}

For the pure conformal metric ($B = 0$): $\det\tilde g_{2D} = -A^4 G^2 < 0$ — Lorentzianity inherited analytically from $g$ via $A > 0$. For the full disformal metric, the determinant must be checked with the canonical formula. Inside horizons ($F < 0$), the disformal term $-A^2 F B(\phi')^2$ becomes positive and can oppose the conformal term; all invertibility and hyperbolicity results must be regenerated from this expression.

# Appendix F — Physical Volume and Density Diagnostics

This appendix distinguishes the geometric volume from the matter-frame volume diagnostics. The pipeline computes the exterior cumulative volume and an interior constant-$r$ volume element; it does not solve a self-consistent matter density profile.

## F.1 Volume Elements

The geometric volume is

\begin{equation} \label{eq:appF_1}
V_{\rm geom}(r)=\frac{4\pi}{3}r^3.
\end{equation}

Inside the geometric horizon, the spatial constant-$r$ slice has volume element per unit $v$

\begin{equation} \label{eq:appF_2}
\frac{d\tilde V}{dv}=4\pi A^2r^2\sqrt{|F|}.
\end{equation}

For the clean run this rises from $2.08\times10^2$ at $0.1M$ to $5.50\times10^5$ at $r_{\min}=10^{-8}M$.

## F.2 Geometric Collapse and Physical Interior Growth

$V_{\rm geom}\to0$ as $r\to0$, while the matter-frame interior volume element grows strongly because $A\sim r^{-1}$, reaching $d\tilde V/dv=5.50\times10^5$ at $r_{\min}=10^{-8}M$. The determinant is strictly negative at every sampled radius, so this growth is a physical result on a globally Lorentzian space, not a formal continuation diagnostic. The areal radius diverges ($\rho\sim1/r$), confirming that the Schwarzschild areal coordinate is not a direct measure of physical volume in TEP.

## F.3 Numerical Verification

| Quantity | Geometric frame | Matter-frame diagnostic |
| --- | --- | --- |
| Volume at $r_{\min}$ | $V_{\rm geom}=4.19\times10^{-24}$ | interior element $d\tilde V/dv=5.50\times10^5$ |
| Lorentzian domain | globally Lorentzian | $\det\tilde g_{2D}<0$ at all sampled radii |
| Density | $M/V_{\rm geom}\propto r^{-3}$ | not independently computed by this pipeline |

## F.4 Supported Statement

### Proposition F.1

The $\phi_0 = 2$ fixed-Schwarzschild configuration replaces the geometric zero-volume intuition with a strongly growing matter-frame interior volume element on a Lorentzian space for all $r > 0$. The areal radius diverges ($\rho \sim 1/r \to \infty$), confirming that this configuration is a spatially enlarged asymptotic end, not a bounded solution. The nonlinear TEP solution determines the physical interior volume. The Schwarzschild areal coordinate is not a direct measure of physical volume in TEP. A finite physical density claim requires an explicit stress-energy solution and is not asserted by the present diagnostic pipeline.

# Appendix G — Fixed-Schwarzschild Perturbation Baseline

This appendix records the perturbation calculation on the fixed-Schwarzschild geometric background — the first of three QNM calculations in this manuscript. The three are: **(1) Fixed-Schwarzschild baseline** (this appendix): the geometric metric is held at Schwarzschild; gravitational tensor QNMs are the GR baseline. **(2) Perturbative sGB exterior** (Appendix L): the Sotiriou–Zhou metric perturbation introduces scalar hair and backreaction; the full coupled axial operator requires derivation from the second variation of the action. **(3) Prescribed Hayward deep-transit model** (Section 7.2): a toy eigenvalue problem on a horizonless regular geometry, illustrating one realisation of the altered spectral problem.

## G.1 Geometric Tensor Equation

In the fixed-background limit, the geometric metric is Schwarzschild. Canonical TEP distinguishes the geometric metric $g_{\mu\nu}$ from the matter metric $\tilde g_{\mu\nu}$; gravitational waves propagate on $g_{\mu\nu}$. Conformal and disformal factors in $\tilde g_{\mu\nu}$ therefore do not shift the gravitational Regge–Wheeler or Zerilli spectrum. Decomposing $h_{\mu\nu}$ into axial and polar spherical harmonics and Fourier transforming $\Psi(t,r)=e^{-i\omega t}\Psi(r)$ gives

\begin{equation} \label{eq:appG_1}
\frac{d^2\Psi_i}{dr_*^2}+\left[\omega^2-V_i^{\rm Schw}(r)\right]\Psi_i=0,
\end{equation}

where $r_*$ is the Schwarzschild tortoise coordinate and $i$ denotes the Regge–Wheeler or Zerilli sector. These are the standard geometric GR perturbation equations for this fixed background.

## G.2 Interpretation of Matter-Metric Diagnostics

The pipeline also evaluated surrogate radial functions constructed from the disformal matter metric. Such functions may diagnose matter or electromagnetic propagation, but they are not gravitational-wave potentials in canonical TEP. In particular, applying gravitational QNM boundary conditions to a $\tilde g_{\mu\nu}$ surrogate does not establish a shifted tensor spectrum.

## G.3 Ringdown Spectrum

The gravitational QNMs of this non-spinning fixed-background limit are the Schwarzschild GR baseline. The previously quoted $0.421-0.099\,i$ frequency and 5.6% shift arose from the matter-metric surrogate rather than the geometric gravitational equation and are not gravitational QNM predictions. The coupled solution (Section 4) introduces scalar-led and mixed modes with coupling-specific QNM shifts; the full coupled axial operator requires derivation from the second variation of the action (Appendix L).

## G.4 Stability and Echo Scope

No complete coupled perturbation system was analysed on this fixed background. Background regularity and the shape of a surrogate matter-metric potential are insufficient to prove absence of ghosts, gradient instabilities, or growing scalar and mixed modes. They are likewise insufficient to predict or exclude gravitational-wave echoes. An echo statement requires geometric perturbation evolution with physically specified inner boundary conditions and a waveform-level analysis; those calculations were not performed on this background. The prescribed Hayward deep-transit model (Section 7.2) addresses a different geometry.

## G.5 Summary

| Question | Result for the fixed-Schwarzschild baseline |
| --- | --- |
| GW propagation metric | Geometric $g_{\mu\nu}$ = Schwarzschild |
| Gravitational tensor QNMs | GR baseline; no TEP shift |
| Scalar-led and mixed modes | Computed in the perturbative sGB exterior (Appendix L); full axial operator open |
| Full linear stability | Background regularity verified; coupled-system stability is open (Section 7) |
| Gravitational-wave echoes | Neither predicted nor excluded on this background |

This appendix covers the fixed-Schwarzschild baseline only. The perturbative sGB exterior (Appendix L) and the prescribed Hayward deep-transit model (Section 7.2) are separate calculations on different geometries. The three must not be conflated.

# Appendix H — Ray Tracing and Orbital Mechanics

This appendix records the photon-sphere, ISCO, and QPO calculations together with the EHT shadow-size central-value comparisons. The reported pipeline configuration imposes exterior screening, $A=1$ and $B=0$, over the ray and orbit domains, so these calculations reproduce the Schwarzschild baseline. Gravitational ringdown is separately governed by the geometric metric $g_{\mu\nu}$, which is fixed at Schwarzschild here.

## H.1 Photon Sphere

Null geodesics of the matter metric $\tilde g_{\mu\nu}$ reduce to Schwarzschild null geodesics on the screened exterior used by the pipeline, where $A=1$ and $B=0$. The photon sphere is therefore at

\begin{equation} \label{eq:appH_1}
r_{\rm ph} = 3\,M,
\end{equation}

and the critical impact parameter for the shadow edge is

\begin{equation} \label{eq:appH_2}
b_{\rm crit} = \frac{r_{\rm ph}}{\sqrt{1 - 2M/r_{\rm ph}}} = 3\sqrt{3}\,M = 5.196\,M.
\end{equation}

These are baseline values. The current exterior-null ray tracing does not calculate a subleading disformal correction to the shadow edge.

## H.2 ISCO and Radiative Efficiency

On the exterior-screened baseline, the innermost stable circular orbit (ISCO) for timelike geodesics is

\begin{equation} \label{eq:appH_3}
r_{\rm ISCO} = 6\,M.
\end{equation}

The specific energy and angular momentum at the ISCO are

\begin{equation} \label{eq:appH_4}
E_{\rm ISCO} = \sqrt{\frac{8}{9}} = 0.9428, \qquad L_{\rm ISCO} = 2\sqrt{3}\,M = 3.464\,M.
\end{equation}

The radiative efficiency — the fraction of rest-mass energy liberated by accretion onto the ISCO — is

\begin{equation} \label{eq:appH_5}
\eta = 1 - E_{\rm ISCO} = 1 - \sqrt{\frac{8}{9}} = 0.0572 = 5.72\%.
\end{equation}

These are standard Schwarzschild values resulting from screening over the orbit domain. They do not imply that a nonconstant conformal factor would exactly cancel from timelike geodesics.

## H.3 Redshift Factor at ISCO

The gravitational redshift factor at the ISCO is

\begin{equation} \label{eq:appH_6}
g = \sqrt{1 - \frac{2M}{r_{\rm ISCO}}} = \sqrt{1 - \frac{1}{3}} = \sqrt{\frac{2}{3}} = 0.816.
\end{equation}

The corresponding redshift is

\begin{equation} \label{eq:appH_7}
z = \frac{1}{g} - 1 = \sqrt{\frac{3}{2}} - 1 = 0.225.
\end{equation}

These are the standard Schwarzschild ISCO values, confirming that the iron-line and reflection spectroscopy signatures of the TEP exterior are identical to GR at the ISCO.

## H.4 3:2 QPO Resonance

Quasi-periodic oscillations (QPOs) from black-hole accretion disks often appear in a 3:2 frequency ratio, interpreted as a resonance between the radial and vertical epicyclic frequencies. For the TEP exterior (Schwarzschild), the resonance condition $\Omega_r / \Omega_\theta = 2/3$ is satisfied at

\begin{equation} \label{eq:appH_8}
r_{\rm QPO} = 10.8\,M,
\end{equation}

where the epicyclic frequency ratio is

\begin{equation} \label{eq:appH_9}
\frac{\Omega_r}{\Omega_\theta}\bigg|_{r=10.8\,M} = 0.667 = \frac{2}{3}.
\end{equation}

This is the standard Schwarzschild 3:2 resonance radius. The TEP disformal interior does not affect this radius because it is well outside the horizon.

## H.5 EHT Shadow Comparisons

The shadow angular diameter is computed from the critical impact parameter $b_{\rm crit} = 5.196\,M$ and the source distance and mass. The pipeline evaluates the shadow size for the two EHT targets:

| Source | TEP prediction | EHT measurement | Deviation |
| --- | --- | --- | --- |
| M87* | $39.69\;\mu\text{as}$ | $42 \pm 3\;\mu\text{as}$ | $0.77\sigma$ |
| Sgr A* | $53.26\;\mu\text{as}$ | $48.7 \pm 7.0\;\mu\text{as}$ | $0.65\sigma$ |

The exterior-screened central values, $39.69\;\mu$as for M87* and $53.26\;\mu$as for Sgr A*, lie within the quoted measurement intervals $42\pm3\;\mu$as and $48.7\pm7.0\;\mu$as, respectively. These are central-value checks; a full likelihood analysis would propagate correlated mass, distance, calibration, and imaging systematics. The fixed-background limit reproduces the Schwarzschild baseline; the coupled solution (Section 4) provides TEP-specific shadow deviations of $-0.044\%$ for $\eta=-0.1$ on $g^{\rm sGB}$ (Appendix L).

## H.6 Gravitational Ringdown Baseline

Gravitational waves propagate on the geometric metric, not on the matter metric used for ray tracing. Since the fixed-background limit holds the geometric metric at Schwarzschild, its non-spinning gravitational ringdown is the Schwarzschild GR baseline. GW150914 produced a spinning remnant and must be compared with the appropriate geometric Kerr spectrum. The coupled solution (Section 4) provides coupling-specific QNM shifts of sub-percent to percent-level.

## H.7 Summary of Orbital and Observational Predictions

| Observable | Value | vs Schwarzschild |
| --- | --- | --- |
| Photon sphere | $r = 3\,M$ | identical |
| Critical impact parameter | $b = 5.196\,M$ | identical |
| ISCO | $r = 6\,M$ | identical |
| ISCO energy | $E = \sqrt{8/9} = 0.9428$ | screened baseline |
| ISCO angular momentum | $L = 3.464\,M$ | identical |
| Radiative efficiency | $\eta = 5.72\%$ | identical |
| Redshift factor at ISCO | $g = 0.816$ | identical |
| Redshift at ISCO | $z = 0.225$ | identical |
| 3:2 QPO resonance | $r = 10.8\,M$ | identical |
| M87* shadow | $39.69\;\mu$as | $0.77\sigma$ from EHT |
| Sgr A* shadow | $53.26\;\mu$as | $0.65\sigma$ from EHT |
| GW150914 $f_{220}$ (non-spinning WKB diagnostic) | $206.5\,$Hz | Schwarzschild baseline (0% shift) |
| GW150914 $f_{220}$ (Kerr, $\chi=0.67$) | $267\,$Hz | Kerr baseline; measured $\sim 250\,$Hz |

The exterior orbital structure (photon sphere at $3M$, ISCO at $6M$, 3:2 QPO at $10.8M$) is identical to Schwarzschild. EHT shadows agree at $0.77\sigma$ (M87*) and $0.65\sigma$ (Sgr A*). The gravitational QNM baseline is the Schwarzschild spectrum for non-spinning remnants; for GW150914's spinning remnant ($\chi=0.67$) the appropriate baseline is Kerr, giving $f_{220}\approx 267\,$Hz versus the measured $\sim 250\,$Hz. The non-spinning WKB diagnostic $f_{220}=206.5\,$Hz ($\omega_R^{\rm WKB}=0.3988$) is not the physical prediction for a spinning remnant and carries $\sim 7\%$ WKB systematic error relative to the exact Schwarzschild value $\omega_R^{\rm exact}=0.3737$. There is no coupling-specific ringdown frequency shift in this fixed-background limit.

# Appendix I — Comparison Table

This appendix provides a detailed technical comparison of TEP-BH with the major competing models of singularity resolution. The TEP-BH column is split into four sub-columns reflecting the distinct calculations performed in this paper: the fixed-Schwarzschild theorem case, the Hayward validation benchmark, the linear sGB exterior candidate, and the scalar-on-regular-background benchmark. These must not be combined into a single "TEP" column, because they represent different geometries with different properties.

## I.1 Detailed Comparison

| Property | GR (Schwarzschild/Kerr) | Regular BH (Bardeen/Hayward) | TEP theorem case ($\phi_0=2$, fixed Schw.) | TEP benchmark ($\phi_0=1$, Hayward) | TEP linear sGB (exterior) | TEP regularised interior (solved) |
| --- | --- | --- | --- | --- | --- | --- |
| Exterior metric | Schwarzschild/Kerr (exact) | Schwarzschild (to $M/r$) | Schwarzschild (fixed) | Hayward (prescribed) | Schwarzschild to $\sim$11 parts in $10^4$ at $|\eta|=0.1$ (perturbative) | Asymptotically Schwarzschild (target) |
| Interior metric | Singular ($r\to0$) | de Sitter-like temporal minimum | Spatially enlarged end ($\rho \to \infty$); not a regular point | Finite-area asymptotic end ($\rho \to 1.995$, $\tilde\ell \to \infty$); not a regular point | Not yet integrated; literature: finite-area singularity at $r \approx 0.27\,r_h$ (Sotiriou & Zhou 2014) | Regular point centre: $A(0) = 4.68$, $\rho_{\rm areal}(0) \to 4.68 \times 10^{-8}$, $w_r(0) = -1.000$, $K(0) = 27.1$ ($\eta=-0.1$, $g=1.1M$) |
| Central curvature | $K\to\infty$ | Finite (by construction) | Vanishes ($\tilde K \sim 39r^2/16 \to 0$) but areal radius diverges | Finite ($\tilde K \to 8/r_h^4$) but asymptotic end, not regular centre | Likely divergent at finite-area singularity (literature) | Finite: $K(0) = 27.1$ for $g=1.1M$ |
| Geodesic completeness | No (incomplete at $r=0$) | Model-dependent | Both null families infinite affine ($\tilde\lambda \sim \int r^{-4}\,dr \to \infty$); asymptotic end | Both null families infinite affine ($\tilde\lambda \sim \int r^{-2}\,dr \to \infty$); asymptotic end | Not yet computed; likely incomplete at finite-area singularity | Both families finite affine; regular point centre, extendable geodesics |
| Horizon structure | Event horizon at $r=2M$ | Event + inner horizon | Schwarzschild horizon (fixed) | Hayward outer + inner horizon | sGB apparent horizon (perturbative artifact) | Temporal well: $\tilde N_{\min} = 0.211$ at $r \approx 1.41M$, no finite causal horizon |
| Temporal Horizon | — | — | Not established (theorem case only) | Not established ($\theta_+=0$ is constant-area property, not temporal freeze) | Not yet computed | Operational: $\mathcal{Z}_{\max} \sim N_o/\tilde N_{\min} \sim 4.7$ at $r \approx 1.41M$; $\tilde N > 0$ everywhere |
| Surface / junction | None | None (smooth) | None (but asymptotic end) | None (but asymptotic end) | Not yet determined | None (smooth, regular point centre) |
| Lorentzianity | Yes | Yes | Yes for all $r > 0$ ($\det = A^4 \det g < 0$); limit $r=0$ not a regular point | Yes in exterior ($F > 0$); inside horizon must check canonical determinant | Yes in exterior ($F > 0$) | Yes: $\det\tilde g_{2D} < 0$ throughout the integrated domain |
| Observable shifts | — | — | — | — | Not yet derived (conformal null invariance; disformal circular orbit; tensor characteristic metric all require derivation) | Photons on $\tilde g$, GWs on $\mathcal{G}_{\rm tensor}$; shadow vs ringdown consistency (target) |
| QNM spectrum | Schwarzschild/Kerr | Modified | — | — | Indicative WKB: scalar-led, axial, polar; requires spectral solver and coupled perturbation equations | Exterior benchmark: axial $+0.136\%$ real shift, polar-led $-0.028\%$ real breaking, $+0.067\%$ damping breaking. Full deep-transit solve: three modes (polar-led $0.462-0.032i$, scalar-led $0.416-0.032i$, overtone $0.745-0.035i$), $2.8\times$ longer damping, $+14.8\%$ real breaking, $11.1\%$ mode splitting |
| Shadow size | $b = 5.196\,M$ | Similar to Schwarzschild | — | — | $-0.044\%$ at $\eta=-0.1$ on $g^{\rm sGB}$; conformal invariance means shadow on $\tilde g$ equals shadow on $g$ for pure conformal; ISCO on $\tilde g$ is $+1.95\%$ (opposite sign, $\mathcal{O}(\eta)$, $\sim 44\times$ larger) — a property of the sGB EFT (Appendix L) | From $\tilde g$ photon sphere (target) |
| Gravitational-wave echoes | No | Possible (inner barrier) | — | — | Not yet analysed (requires waveform-level analysis with inner boundary conditions) | Depends on interior outcome |
| Stability / hyperbolicity | Stable | Model-dependent | — | — | Tensor sector: kinetic term positive at leading order, $c_T = 1$ on Schwarzschild background (Horndeski $G_4$ unchanged by sGB at leading order). Full tensor characteristic metric on TEP solution, global ghost freedom, and scalar sector hyperbolicity in deep interior remain open (Thaalba et al. 2024; Section 7.3) | Scalar-sector principal symbol on solved geometry (target) |
| $c_T$ (tensor speed) | $c_T = 1$ | $c_T \to 1$ asymptotically | — | — | $c_T = 1$ on Schwarzschild background (Horndeski $G_4 = M_{\rm Pl}^2/2$, sGB coupling enters through $G_5$); GW170817 satisfied on that background; full tensor characteristic metric on TEP solution pending (Section 7.3) | Established |
| Observational coupling | $M$ (and $a$) | Regularisation parameter | — | — | One $\alpha_{\rm GB}$; $\eta_i = 3\alpha_{\rm GB}/M_i^2$ per source; $|\eta|=0.1$ cannot be universal across masses | Same (target) |

## I.2 What Is Established vs What Is Target

The established results in this paper are:

- The Schwarzschild incompatibility: on fixed Schwarzschild, finite curvature and bounded areal radius are mutually exclusive for the power-law conformal class (Section 3, Appendix D).

- The Hayward benchmark: finite curvature and bounded areal radius are compatible on a regular background, but the $\phi_0 = 1$ benchmark produces a finite-area asymptotic end, not a regular centre (Appendix K).

- The perturbative sGB exterior: scalar hair and real backreaction are established by the published $\mathcal{O}(\alpha^2)$ solution (Sotiriou & Zhou 2014). The non-perturbative exterior mass-function evolution requires rerun with the correct scalar charge.

- The literature evidence that standard linear sGB develops a finite-area interior singularity (Sotiriou & Zhou 2014; Thaalba et al. 2024).

- The minimal temporal-well criteria: $0 < N(r)$ everywhere with $N_{\min} \ll N_o$, regular Lorentzian geometry, no observer-independent one-way boundary, extreme external redshift (Section 6.3).

- The scalar-on-regular-background benchmark with a regular point centre, integrated to $r = 10^{-8}M$ for $\eta=-0.1$ (Section 4, Appendix K).

The remaining target results are:

- The coupled polar–scalar QNM spectrum and isospectrality breaking.

- The observable shifts as quantitative predictions (requires deriving both characteristic metrics).

- The tensor speed $c_T$ derived from the action.

- The observational fit with one universal $\alpha_{\rm GB}$ across all sources.

The comparison table above distinguishes these explicitly. Earlier versions of this appendix combined all TEP calculations into a single column, producing claims (globally Lorentzian interior, geodesic completeness, vanishing central curvature, disformal birefringence, no inner boundary) that belong to specific benchmark or theorem cases, not to the unsolved global TEP solution.

# Appendix J — Reproducibility

Every numerical result quoted in this paper is produced by the automated pipeline, which downloads or verifies the primary data, constructs the prescribed matter metric, computes the diagnostics, and writes the comparison tables. Each clean run writes SHA-256 checksums for the result files. Checksums prove that the result files were not altered after generation; they do not prove that the calculations are correct. Three categories of calculation require particular care in scalar–tensor perturbation theory and are not cited as quantitative evidence in this manuscript: (i) the nonlinear exterior mass-function evolution, which requires a consistent scalar charge and sign convention matched to published sGB solutions; (ii) surrogate WKB QNM frequencies, which require the perturbation potential to be derived from the second variation of the action (not from inserting modified background functions into a GR potential formula); and (iii) matter-frame observables, which must be computed on the matter metric $\tilde g$ rather than the geometric metric $g$. The fixed-background limit is globally Lorentzian for all $r > 0$; the coupled solution is Lorentzian in the exterior ($F > 0$). The deep-interior Lorentzianity of the scalar-on-regular-background benchmark depends on the interior integration (Section 4, Appendix K).

## J.1 Pipeline Architecture

The pipeline consists of 50 steps (`step_00` through `step_49`), each implemented as an independent Python module. Steps 00–11 implement the fixed-background limit (Section 3); steps 12–15 implement the coupled solution (Section 4); steps 16–41 are derivation scripts (geometry, corrected observables, QNM solvers, interior, phantom mass, Kerr-sGB, GW confrontations); step 42 generates figures; steps 43–49 implement the S-star inference pipeline (Section 9, Appendix J). The steps are:

| Step | Inference Step | Output | Description |
| --- | --- | --- | --- |
| step_00 | step_00_data_download | data/raw/, data/processed/ | Download EHT M87* visibilities, compile measurement tables |
| step_01 | step_01_field_equations | step_01_field_equations.{json,csv} | Disformal metric, curvature invariants, geodesic completeness, physical volume/density |
| step_02 | step_02_perturbations | step_02_perturbations.{json,csv} | Regge–Wheeler/Zerilli potentials, QNMs, echo check |
| step_03 | step_03_raytracing | step_03_raytracing.{json,csv} | Photon sphere, shadow angular diameter, ISCO, EHT comparison |
| step_04 | step_04_accretion | step_04_accretion.{json,csv} | Circular geodesics, epicyclic frequencies, radiative efficiency, redshift, QPO |
| step_05 | step_05_observational_constraints | step_05_observational_constraints.{json,csv} | Chi-squared comparison of TEP predictions against EHT and LIGO |
| step_06 | step_06_gw190521_mass_gap | step_06_gw190521_mass_gap.{json,csv} | GW190521 mass gap: LIGO posterior analysis (TEP exterior-null) |
| step_07 | step_07_qpo_frequency_lock | step_07_qpo_frequency_lock.{json,csv} | QPO frequency lock: 3:2 epicyclic resonance ratios (Schwarzschild baseline) |
| step_08 | step_08_jwst_early_smbhs | step_08_jwst_early_smbhs.{json,csv} | JWST early SMBHs: Eddington-limited growth envelope from stellar seeds |
| step_09 | step_09_spin_bias | step_09_spin_bias.{json,csv} | Over-maximal spin bias: observed spin distribution (TEP exterior-null) |
| step_10 | step_10_tde_missing_flares | step_10_tde_missing_flares.{json,csv} | TDE missing flares: sub-Eddington luminosity survey (TEP exterior-null) |
| step_11 | step_11_eht_polarization | step_11_eht_polarization.{json,csv} | EHT polarization: birefringence check (TEP exterior-null) |
| step_12 | step_12_self_gravitating | step_12_self_gravitating.{json,csv} | sGB metric corrections, scalar profile, exterior observables (shadow, ISCO, QNM), $\eta$-scan, GW150914 projection (Section 4) |
| step_13 | step_13_scalar_perturbations | step_13_scalar_perturbations.{json,csv} | Scalar-led, axial, and polar QNM channels; isospectrality breaking; deep-transit analysis (Section 7) |
| step_14 | step_14_kerr_tep | step_14_kerr_tep.{json,csv} | Kerr–TEP shadow and ISCO spin scan; frame-dragging cancellation; GW150914 Kerr ringdown comparison (Section 4) |
| step_15 | step_15_dynamical_signatures | step_15_dynamical_signatures.{json,csv} | PPN parameters, $-1$PN scalar dipole radiation, $c_T = 1$ on the Schwarzschild background, combined $\eta$ constraints (Section 4) |
| step_16 | step_16_exact_geometry | step_16_exact_geometry.{json,csv} | Exact curvature invariants, determinant, inverse metric, areal radius, proper radial distance |
| step_17 | step_17_null_expansions | step_17_null_expansions.json | Null expansions $\theta_\pm$ for TEP matter metric on Schwarzschild and Hayward backgrounds |
| step_18 | step_18_observer_redshift | step_18_observer_redshift.{json,csv} | Gravitational redshift for static observers at different radii |
| step_19 | step_19_observer_frequency_transfer | step_19_observer_frequency_transfer.{json,csv} | Invariant emitter-receiver frequency transfer factor $\mathcal{Z}$ (Section 6) |
| step_20 | step_20_corrected_observables | step_20_corrected_observables.json | Corrected exterior observables (shadow, ISCO, frequency transfer) under fixed-ADM normalization (Appendix L) |
| step_21 | step_21_cT_derivation | step_21_cT_derivation.json | Tensor propagation speed $c_T$ from principal symbol of coupled Einstein-sGB equations |
| step_22 | step_22_quadratic_action | step_22_quadratic_action.json | Exact quadratic action for axial, polar, and scalar perturbation channels (Section 7) |
| step_23 | step_23_characteristic_matrices | step_23_characteristic_matrices.json | Tensor characteristic matrices and principal symbol analysis |
| step_24 | step_24_conformal_invariance_check | step_24_conformal_invariance_check.json | Conformal invariance verification for null geodesics |
| step_25 | step_25_qnm_solver | step_25_qnm_solver.json | Perturbative sGB QNM solver: axial, polar-led, scalar-led modes |
| step_26 | step_26_coupled_spectral_solver | step_26_coupled_spectral_solver.json | Coupled spectral solver for polar-scalar mixed modes |
| step_27 | step_27_qnm_horizon_branch | step_27_qnm_horizon_branch.json | Horizon-bearing sGB QNM: Schwarzschild base + Sotiriou-Zhou $\mathcal{O}(\eta^2)$ correction (Section 7.3) |
| step_28 | step_28_qnm_validation | step_28_qnm_validation.json | QNM spectrum validation against published sGB results (Bryant et al. 2021; Blazquez-Salcedo et al. 2016) (Section 7.3) |
| step_29 | step_29_matrix_leaver | step_29_matrix_leaver.json | Full coupled $2\times 2$ matrix continued-fraction QNM solver with regular inner boundary; three deep-transit modes (Section 7.3) |
| step_30 | step_30_taylor_recurrence | step_30_taylor_recurrence.json | Exact Taylor-series recurrence for background potentials and scalar field coefficients near the origin (Section 7.3) |
| step_31 | step_31_frobenius_analysis | step_31_frobenius_analysis.json | Frobenius regularity analysis at the temporal minimum: $r^{\ell+1}$ regularity, finite tortoise, regularised polar–scalar mixing (Section 7.3) |
| step_32 | step_32_interior_integration | step_32_interior_integration.json | Numerical integration of sGB scalar field on Hayward-class regular background |
| step_33 | step_33_interior_analysis | step_33_interior_analysis.json | Interior analysis: regularity checks, Lorentzian signature, lapse minimum |
| step_34 | step_34_solve_interior | step_34_solve_interior.json | TEP interior solver: sGB scalar on Hayward background, matter metric construction (Section 4, Appendix K) |
| step_35 | step_35_mass_bias_sign | step_35_mass_bias_sign.json | Mass-bias sign equation derivation (Section 2.8) |
| step_36 | step_36_phantom_mass_critical | step_36_phantom_mass_critical.json | Horizon-scale phantom mass: critical $g$ analysis, EHT M87$^\ast$ and Sgr A$^\ast$ shadow comparison (Section 8.2) |
| step_37 | step_37_phantom_mass_raytrace | step_37_phantom_mass_raytrace.json | phantom mass null geodesic ray-tracing (Section 8.2) |
| step_38 | step_38_phantom_mass_scan | step_38_phantom_mass_scan.json | phantom mass parameter scan (Section 8.2) |
| step_39 | step_39_eht_visibility_fit | step_39_eht_visibility_fit.json | EHT visibility-domain joint inference: direct fit of TEP shadow model to calibrated visibilities (Section 9.4) |
| step_40 | step_40_kerr_sgb_true | step_40_kerr_sgb_true.json | True rotating sGB shadow and QNM from Delgado et al. (2020) slowly-rotating solution (Section 8.5) |
| step_41 | step_41_gw250114_confrontation | step_41_gw250114_confrontation.json | GW250114 corrected confrontation with TEP observables |
| step_42 | step_42_generate_figures | results/figures/ | Figure generation (10 figures from paper plan) |
| step_43 | step_43_sstar_data | step_43_sstar_data.json | S-star data acquisition: 145 astrometric + 44 radial-velocity epochs for S2 (Section 9) |
| step_44 | step_44_gr_fit | step_44_gr_fit.json | Conventional GR fit to S-star data (Section 9) |
| step_45 | step_45_mass_bias | step_45_mass_bias.json | Initial mass-bias sign evaluation from GR fit residuals (Section 9) |
| step_46 | step_46_tep_transfer | step_46_tep_transfer.json | TEP transfer-function fit with $\eta_{\rm TEP}$ as free parameter (Section 9) |
| step_47 | step_47_joint_forward | step_47_joint_forward.json | Joint forward-model: composition vs calibration separation, $\mathcal{C}_V$ diagnostic (Section 9) |
| step_48 | step_48_likelihood | step_48_likelihood.json | Formal likelihood comparison: Bayes factor, BIC, AIC, Wilks (Section 9) |
| step_49 | step_49_posterior | step_49_posterior.json | phantom mass posterior via MCMC (Section 9) |

All calculations use the unified parameter set: $B_0 = 1.0$, $M = 1.0$, $\beta_A = -1.0$, $n_B = 2.0$, $\delta = 0.05$, $\sigma_B = 1.5$, $r_h = 2.0$, with the quartic-damped disformal coupling $B(\phi) = B_0\,|\phi|^{n_B}/(1 + |\phi|^{n_B})\,\exp(-\phi^4/2\sigma_B^4)$. The conformal exponent $\beta_A = -1$ is frozen across all calculations — the analysis of Section 3, sGB exterior, and the target global solution — consistent with the wider TEP corpus weak-field value ($\beta \simeq -0.013$, with $\beta_A = -1$ as the strong-field limit). Steps 12–15 use the sGB coupling parameter $\eta$ (primary value $\eta = -0.1$ for the mass-inflation branch) with the exact $\mathcal{O}(\alpha^2)$ perturbative metric corrections of Sotiriou \& Zhou (2014), Eqs. (56)-(63); the $\eta$-scan covers $\eta \in \{-0.05, -0.1, -0.15, -0.2\}$.

#### Convention resolution

The conformal coupling $A = e^{\beta_A \phi}$ with $\beta_A = -1$ is frozen across all calculations. The different scalar profiles interact with this single convention as follows: (i) the analysis of Section 3 uses $\phi = \phi_0 \ln(r/r_h) < 0$ in the interior, giving $A = e^{-\phi} \to \infty$ — the divergent conformal factor that establishes this result; (ii) the sGB exterior uses $\phi \sim Q_s/r$ (Coulomb). TEP selects the mass-inflation branch with $Q_s < 0$ (equivalently $\alpha_{\rm GB} < 0$), so $\phi < 0$ and $A = e^{-\phi} > 1$ — the matter metric is conformally *magnified* relative to $g$; (iii) the target global solution has $\phi \to \phi_c$ finite at the centre, giving $A_c = e^{-\phi_c}$ finite — one candidate profile satisfying the minimal temporal-well criteria of Section 6.3. The strong-field value $\beta_A = -1$ connects to the weak-field corpus value $\beta \simeq -0.013$ through the effective coupling's $\phi$-dependence: if the coupling runs with $\phi$ or temporal shear, that running is part of the action. The quartic-Gaussian $B(\phi)$ should ultimately be derived from the action or screening mechanism, or an admissible class of $B(\phi, X)$ functions should be shown to produce stable results; this is a refinement target, not a structural inconsistency.

## J.2 Output Files

The pipeline produces three categories of output:

- `results/*.json` — structured summary of each step (the canonical machine-readable results);

- `results/*.csv` — data tables for plotting and inspection;

- `data/processed/*.json` — compiled measurement tables (EHT, LIGO QNM, QPO, TDE, spin, JWST) built from the raw downloads;

- `data/raw/` — downloaded EHT/LIGO data (EHT M87* 2019-D01-01 visibilities, LIGO GW190521 posterior);

- `results/pipeline_results.json` — top-level summary with step status, timing, and execution metadata;

- `results/checksums_sha256.json` — SHA-256 checksum of every `results/*.json` and `results/*.csv` file.

## J.3 Checksum Verification

At the end of each run, the pipeline computes the SHA-256 hash of every result file and writes them to `results/checksums_sha256.json`. This covers all result files (step JSON + step CSV + pipeline summary JSON). Any modification to a result file is detectable by re-running the hash. The checksum file itself is regenerated on each run, so the recorded hashes always correspond to the current results.

## J.4 Warning-Free Execution

A clean run is verified with the repository command:

\begin{equation} \label{eq:appJ_1}
\texttt{python scripts/run\_pipeline.py}
\end{equation}

The current diagnostics include a numerical safeguard that masks inverse-determinant curvature expressions where $|\det\tilde g_{2D}|$ is small; in the fixed-background limit this safeguard is never triggered because the determinant is strictly negative everywhere (globally Lorentzian).

## J.5 Data Sources

All observational data is drawn from published, publicly available sources:

| Source | Reference | Used in |
| --- | --- | --- |
| EHT M87* 2019 calibrated visibilities | EHT Collaboration 2019, data product 2019-D01-01 (GitHub) | step_00, step_03, step_05, step_11 |
| LIGO GW190521 posterior | Isi et al. 2020 (Zenodo 4057131) | step_06 |
| QPO measurements (GRS 1915+105) | Strohmayer 2001; Remillard et al. 2002; Homan et al. 2005 | step_07 |
| JWST early SMBH masses | Harikane et al. 2023; Greene et al. 2023; Bogdan et al. 2024 | step_08 |
| Black-hole spin measurements | McClintock et al. 2014; Reynolds 2021; EHT 2019/2022 | step_09 |
| TDE flare catalogs | van Velzen et al. 2021; Holoien et al. 2020 | step_10 |
| EHT polarization | EHT 2021 ApJ 910 L13; EHT 2024 ApJ 964 L26 | step_11 |
| LIGO QNM measurements | compiled in data/processed/ligo_qnm_measurements.json | step_05 |
| sGB perturbative metric corrections | Sotiriou & Zhou 2014, Eqs. (56)-(63) (exact $\mathcal{O}(\alpha^2)$) | step_12, step_14 |
| Kerr QNM / shadow formulas | Bardeen 1973; Leaver 1985 (approximate) | step_14 |
| GW170817 $c_T$ constraint | LIGO/Virgo & Fermi-GBM 2017 | step_15 |
| Binary pulsar dipole bounds | Freire et al. 2012; Antoniadis et al. 2013 (conditional on NS scalarization) | step_15 |

## J.6 Running the Pipeline

The full pipeline is run with a single command:

git clone https://github.com/matthewsmawfield/TEP-BH.git
cd TEP-BH
python scripts/run_pipeline.py

Options include `--start-step`, `--stop-step`, `--skip-steps`, `--no-derive`, `--no-inference`, `--no-figures`, `--list-steps`, and `--continue-on-error`. Full documentation is in `scripts/README.md`.

## J.7 Code Availability

All code is in the GitHub repository:
https://github.com/matthewsmawfield/TEP-BH

The repository contains the complete pipeline (`scripts/`), the processed and raw data (`data/`), the results (`results/`), the manuscript source (`site/components/`), and the figure-generation code (`scripts/steps/step_42_generate_figures.py`). The site is built with `cd site && npm run build` and published at
https://mlsmawfield.com/tep/bh.

The pipeline is 50 steps (step_00–step_49), runs warning-free, and writes SHA-256 checksums for all result files. Steps 00–11 use $\phi_0 = 2.0$, $\delta = 0.05$, $\sigma_B = 1.5$ for the fixed-background limit; steps 12–15 use the sGB coupling $\eta$ (primary $\eta = -0.1$) for the coupled solution, selecting the mass-inflation branch; steps 16–41 are derivation scripts; step 42 generates figures; steps 43–49 implement the S-star inference. All data is from published sources (EHT 2019-D01-01, LIGO Zenodo 4057131, and compiled measurement tables). Full documentation is in the repository `README`.

**S-star inference pipeline.** Steps 43–49 of the unified pipeline implement the primary falsifiable test: a non-isochronous refit of the published S-star data around Sgr A*. Step 43 downloads the machine-readable CDS files for Gillessen et al. (2017, VizieR J/ApJ/837/30, table5.dat and table3.dat); it parses 145 NACO/NTT/Keck/Gemini astrometric epochs and 44 SINFONI/Keck/Gemini radial-velocity epochs for S2 relative to Sgr A*. The steps are: (43) data acquisition, (44) conventional GR fit, (45) initial mass-bias sign evaluation, (46) TEP transfer-function fit, (47) joint forward-model with composition/calibration separation and $\mathcal C_V$ diagnostic, (48) formal likelihood comparison (Bayes factor, BIC, AIC, Wilks), (49) MCMC posterior on $M_{\rm phantom}^T$.

# Appendix K — Regular-Geometry Validation Benchmark

This appendix records the regular-geometry validation benchmark used throughout the main text. The Hayward metric is a controlled mathematical reference, not the TEP-selected solution: it validates the analysis pipeline and confirms that finite curvature and bounded areal geometry are mutually compatible once the geometric singularity is absent. The question TEP asks is whether a dynamical proper-time field can supply the physical origin of that regularisation. The scalar-on-regular-background benchmark (Section 4) is the decisive test of this question. The literature evidence (Sotiriou & Zhou 2014; Thaalba et al. 2024) indicates that standard linear sGB may develop a finite-area singularity rather than a regular centre, motivating the full TEP action.

## K.1 The Hayward Metric

The Hayward metric is

\begin{equation} \label{eq:appK_1}
F(r) = 1 - \frac{2Mr^2}{r^3 + 2M\ell^2},
\end{equation}

where $\ell$ is the regularisation scale. As $r \to 0$, $F \to 1$ (de Sitter-like temporal minimum); as $r \to \infty$, $F \to 1 - 2M/r$ (Schwarzschild). The effective energy density is

\begin{equation} \label{eq:appK_2}
\rho_{\rm eff} = -G^t_t = \frac{12M^2\ell^2}{(r^3 + 2M\ell^2)^2} \to \frac{3}{\ell^2} \quad \text{as } r \to 0,
\end{equation}

finite at the centre. The Kretschmann scalar is $K_{\rm Hayward}(0) = 24/\ell^4$, finite. The equation of state is $w_r = -1$ (cosmological-constant-like radial pressure), with anisotropic tangential pressure.

## K.2 TEP Matter Metric on the Hayward Background

The TEP matter metric $\tilde g = A^2 g_{\rm Hayward}$ with $A = (r_h/r)^{\phi_0}$ and $\phi_0 = 1$ is placed on the Hayward background. The computed invariants are:

| Quantity | Value as $r \to 0$ | Classification |
| --- | --- | --- |
| Kretschmann $\tilde K$ | $8/r_h^4 = 0.5$ (for $M=1$, $r_h=2M$) | Finite — de Sitter-like temporal minimum |
| Areal radius $\rho = Ar$ | $1.995$ | Bounded — finite-area asymptotic end |
| $\det\tilde g_{2D}$ | $< 0$ at 100% of sampled radii | Lorentzian in exterior ($F > 0$) |
| Null expansion $\theta_+$ | $0$ exactly | Constant-area property of the asymptotic end (not temporal freeze) |
| Affine parameter $\tilde\lambda$ (both families) | $\sim \int r^{-2}\,dr \to \infty$ | Infinite affine distance — asymptotic end, not regular point |
| Radial proper distance $\tilde\ell$ | $\sim \int dr/r \to \infty$ | Infinite — tube-like asymptotic end |
| Static-observer clock rate $d\tilde\tau/dt$ | $\to \infty$ | Conformal blueshift (unphysical; benchmark artefact of divergent $A$) |
| Static-observer redshift $z$ | $\to -1$ (blueshift) | Infinite redshift is at the Hayward outer horizon, not the centre — benchmark artefact of the divergent-$A$ test profile, not the TEP target |

## K.3 Classification of the Limiting Region

At an ordinary regular spherical centre one has $\rho \to 0$ as $r \to 0$. Here $\rho \to 1.995 \neq 0$: the limiting areal radius is nonzero. Because $A \sim 1/r$ in the deep interior, the radial proper distance behaves as $\tilde\ell \sim \int dr/r \to \infty$. The limiting region is therefore not a point-like centre but a finite-area asymptotic end (a tube-like limiting region of infinite radial extent). This is still a valid TEP geometry — the curvature is finite, the areal radius is bounded, and the metric is globally Lorentzian — but it must be classified correctly as an asymptotic end rather than a regular point centre. The scalar-on-regular-background benchmark (Section 4) now produces an ordinary regular point centre with $\rho_{\rm areal}(0) \to 4.68 \times 10^{-8}$ for $\eta=-0.1$, $g=1.1M$.

## K.4 What the Benchmark Validates

The benchmark validates two things:

- The analysis pipeline correctly recognises a regular geometric metric and classifies the matter-frame limiting region (finite Kretschmann, bounded areal radius, Lorentzianity inherited from $g$ via $A > 0$).

- Finite curvature and bounded areal geometry are mutually compatible once the geometric singularity is absent — the conditions that were mutually exclusive in the Schwarzschild incompatibility result (Section 3) are simultaneously achievable on a regular background.

The benchmark establishes pipeline correctness and the mutual compatibility of finite curvature and bounded areal geometry on a regular background. The dynamical generation of this geometry by the sGB coupling is the task of the scalar-on-regular-background benchmark (Section 4). The limiting region is a finite-area asymptotic end (Section K.3). The null expansion $\theta_+ = 0$ reflects the constant-area property of the asymptotic end; the invariant frequency-transfer calculation of Section 6 establishes the Temporal Horizon. Both null families have infinite affine parameter ($\tilde\lambda \sim \int r^{-2}\,dr \to \infty$); the asymptotic end is at infinite affine distance.

## K.5 Regularisation Scale Effects

The Kretschmann scalar at the centre depends on the regularisation scale: $K_{\rm Hayward}(0) = 24/\ell^4$. For $\ell = 0.1$ (the benchmark value), $K = 2.4 \times 10^5$; for $\ell = 1.0$, $K = 24$; for $\ell = 0.001$, $K = 2.4 \times 10^{13}$. These are code-unit values ($M = 1$); they do not establish Planck-scale curvature without converting to physical units for a specific black-hole mass. The regularisation scale is a parameter of the benchmark, not a prediction of the theory; the scalar-on-regular-background benchmark with $g = 1.1M$ gives $K(0) = 27.1$ for $\eta=-0.1$ (Section 4).

## K.6 Configuration Scan

A scan over conformal exponent $\phi_0$ on the Hayward background. For a regular (de Sitter-like) seed, the background Kretschmann $K_{\rm Hay} \to$ finite as $r \to 0$, so the dominant contribution to the conformal Kretschmann comes from the $A$-derivative terms: $\tilde K \sim r^{4\phi_0 - 4}$ (not $r^{4\phi_0 - 6}$, which is the Schwarzschild-seed result where $K_{\rm Schw} \sim r^{-6}$ dominates). The areal radius scales as $\rho = Ar \sim r^{1-\phi_0}$ on either seed. The asymptotic classifications are:

| $\phi_0$ | $\rho$ as $r\to 0$ | $\tilde K$ as $r\to 0$ | Lorentzian | Asymptotic classification |
| --- | --- | --- | --- | --- |
| 0.5 | $\to 0$ | $\to \infty$ ($r^{-2}$) | Yes | Point centre, curvature diverges |
| 1.0 | $\to r_h \neq 0$ | $\to$ finite ($r^0$) | Yes | Finite-area asymptotic end, finite curvature |
| 1.25 | $\to \infty$ ($r^{-0.25}$) | $\to 0$ ($r^1$) | Yes | Areal radius diverges, curvature vanishes |
| 2.0 | $\to \infty$ ($r^{-1}$) | $\to 0$ ($r^4$) | Yes | Areal radius diverges, curvature vanishes |

The $\phi_0 = 1$ configuration is the validation benchmark used in the main text: it is the unique configuration that simultaneously achieves bounded (nonzero) areal radius and finite curvature on the Hayward background. The curvature scaling $\tilde K \sim r^{4\phi_0 - 4}$ differs from the Schwarzschild-seed scaling $\tilde K \sim r^{4\phi_0 - 6}$ (Appendix D) because the regular seed has finite background curvature, so the $A$-derivative terms dominate rather than the background Kretschmann. This distinction is critical: the Schwarzschild incompatibility result uses the Schwarzschild exponent, while the benchmark classification uses the Hayward exponent. The two must not be mixed.

# Appendix L — Explicit Derivation of Corrected Exterior Observables

This appendix provides the step-by-step derivation of the corrected exterior observables — the Regge–Wheeler potential, the photon sphere and shadow radius, and the ISCO on the matter metric $\tilde g$ — from the Sotiriou \& Zhou (2014) perturbative sGB solution. The derivation demonstrates the coupling-order difference between the photon and massive-particle sectors explicitly.

## L.1 The sGB-Corrected Metric

The Sotiriou \& Zhou (2014) perturbative solution gives the geometric metric to $\mathcal{O}(\beta^2)$, where $\beta = \alpha_{\rm GB}/r_H^2 = \eta/12$ (since $r_H = 2M$ and $\alpha_{\rm GB} = \eta M^2/3$):

\begin{equation} \label{eq:appL_1}
ds^2_{\rm sGB} = -F(1 + \beta^2 h_2)\,dt^2 + \frac{1 + \beta^2 \sigma_2}{F}\,dr^2 + r^2\,d\Omega^2,
\end{equation}

where $F = 1 - r_H/r$, $x = r_H/r$, and the dimensionless polynomials are (Sotiriou \& Zhou 2014, Eqs. 60–61):

\begin{equation} \label{eq:appL_2}
h_2(x) = -\frac{98}{5}x - \frac{98}{5}x^2 - \frac{274}{15}x^3 - \frac{14}{15}x^4 + \frac{52}{15}x^5 + \frac{20}{3}x^6,
\end{equation}

\begin{equation} \label{eq:appL_3}
\sigma_2(x) = \frac{98}{5}x + \frac{58}{5}x^2 + \frac{38}{5}x^3 - \frac{406}{15}x^4 - \frac{436}{15}x^5 - \frac{92}{3}x^6.
\end{equation}

The would-be horizon shifts to $r_H = 2m = 2M(1 - 19.6\,\beta^2)$ at fixed ADM mass — a property of the horizon-bearing branch on which the perturbative sGB benchmark lives, not a physical TEP temporal-well horizon. The scalar field is $\phi(r) = (2\alpha_{\rm GB}/m)(1/r + m/r^2 + 4m^2/(3r^3))$ with $m = M/(1 + 49\eta^2/360)$ and scalar charge $Q_s = 2\alpha_{\rm GB}/m$. For the TEP mass-inflation branch the scalar charge is negative ($Q_s < 0$), so $\phi < 0$ in the exterior. The conformal factor is $A = e^{-\phi}$, giving the matter metric $\tilde g_{\mu\nu} = A^2 g^{\rm sGB}_{\mu\nu}$ with $A > 1$ (conformal magnification). The true TEP solution is a temporal well: $N \geq N_{\min} > 0$ everywhere, with a finite but extremely small minimum lapse at the centre.

The perturbative regime requires $\beta^2 |h_2(x)| \ll 1$ at all radii of interest. At the would-be horizon ($x = 1$), $h_2(1) = -48.27$, so $\beta^2 |h_2(1)| = 6.94 \times 10^{-5} \times 48.27 = 0.0034 \ll 1$. The value $\eta = -0.1$ ($\beta^2 = 6.94 \times 10^{-5}$) is used throughout this appendix as the TEP mass-inflation benchmark ($\alpha_{\rm GB} < 0$). Note that $|\eta| = 0.1$ exceeds current observational constraints ($\alpha_{\rm GB} < 2.9$ km$^2$ gives $|\eta| < 0.04$ for $M = 10\,M_\odot$); it is used as a perturbative reference value, not as a physically allowed coupling.

## L.2 Regge–Wheeler Potential: Status

A naive application of the scalar gradient $\phi'(r)$ directly into the standard GR Regge–Wheeler potential formula yields a term with dimensions $L^{-4}$ — inconsistent with the potential's $L^{-2}$ dimensions. The correct approach requires deriving the axial perturbation equation from the second variation of the sGB action on the corrected background. In modified gravity, the axial perturbation equation generally cannot be obtained merely by inserting the modified background functions into a GR potential formula; the modified field equations contribute directly to the quadratic perturbation operator.

The dominant effect on the QNM spectrum comes from the would-be horizon shift: $r_H \to 2M(1 - 19.6\,\beta^2)$ shifts $F$ and hence the entire potential. The would-be horizon scale change $r_H \to 2M(1 - 19.6\,\beta^2)$ gives an $\mathcal{O}(\beta^2) \sim \mathcal{O}(10^{-3})$ QNM modification at $\eta = -0.1$ (with $\beta = \eta/12$); the coefficient and sign are not determined by the would-be horizon shift alone. A precise value requires a coupled spectral solver on the corrected background, with the perturbation equations derived from the second variation of the action and validated against published nonperturbative sGB QNM spectra (Witek et al. 2019; Blazquez-Calzadilla et al. 2020; Chen et al. 2024). The structural prediction (scalar-led channel, broken isospectrality at $\mathcal{O}(\eta^2)$) follows from the coupling structure; the sign of the QNM shift is not determined by the would-be horizon shift alone. The deep-transit ringdown structure on the horizonless temporal well is analysed in Section 7.3.

## L.3 Photon Sphere and Shadow (Null Geodesics on $g^{\rm sGB}$)

For null geodesics on $ds^2 = -f\,dt^2 + g\,dr^2 + h\,d\Omega^2$, the impact parameter at a circular null orbit is $b^2 = h/f$. The photon sphere is at the minimum of $b^2(r)$, and the shadow radius is $b_{\rm ph} = \sqrt{h/f}$ at that minimum.

For the geometric metric $g^{\rm sGB}$: $f = F(1 + \beta^2 h_2)$, $h = r^2$. The minimum of $b^2 = r^2/[F(1 + \beta^2 h_2)]$ gives $r_{\rm ph} = 2.9974M$ and $b_{\rm ph} = 5.194M$ at $\eta = -0.1$, a $-0.044\%$ deviation from Schwarzschild ($b = 5.196M$).

For the matter metric $\tilde g = A^2 g^{\rm sGB}$: $f_{\tilde{}} = A^2 f$, $h_{\tilde{}} = A^2 r^2$, so $b^2 = h_{\tilde{}}/f_{\tilde{}} = r^2/f = r^2/[F(1 + \beta^2 h_2)]$ — identical to $g^{\rm sGB}$. The conformal factor $A^2$ cancels exactly. This is conformal invariance of null geodesics: the photon sphere and shadow are the same on $g^{\rm sGB}$ and $\tilde g$. The shadow deviation is $\mathcal{O}(\eta^2)$, determined solely by the metric perturbation.

## L.4 ISCO (Timelike Geodesics on $\tilde g$)

For timelike geodesics on $ds^2 = -f\,dt^2 + g\,dr^2 + h\,d\Omega^2$, the specific angular momentum of a circular orbit is:

\begin{equation} \label{eq:appL_4}
L^2 = \frac{f'\, h^2}{f\, h' - f'\, h},
\end{equation}

and the specific energy is $E^2 = f(1 + L^2/h)$. The ISCO is at the minimum of $L^2(r)$ (marginally stable orbit).

For the geometric metric $g^{\rm sGB}$ ($h = r^2$): $L^2 = f' r^4 / (f \cdot 2r - f' r^2)$. The minimum gives $r_{\rm ISCO} = 5.996M$ at $\eta = -0.1$, a $-0.061\%$ deviation. This is an $\mathcal{O}(\eta^2)$ effect from the metric perturbation alone.

For the matter metric $\tilde g = A^2 g^{\rm sGB}$ ($h = A^2 r^2$): the angular component now includes the conformal factor. The derivative $h' = 2A^2 r + 2A^2 r \cdot (-\phi') r = A^2(2r - 2r^2\phi')$ introduces $\phi'$ at $\mathcal{O}(\eta)$, which does not cancel. The minimum of $L^2(r)$ gives $r_{\rm ISCO} = 6.117M$ at $\eta = -0.1$, a $+1.95\%$ deviation. This is an $\mathcal{O}(\eta)$ effect: the conformal factor $A = e^{-\phi} > 1$ magnifies the effective areal radius $\tilde R = A \cdot r$, pushing the ISCO outward.

## L.5 Coupling-Order Difference in the sGB Benchmark

The key result is the coupling-order difference and sign difference between the shadow and the matter-metric ISCO. Under fixed-ADM mass normalization ($M_{\rm ADM} = M$ held fixed; the would-be horizon shifts to $r_H = 2M(1 - 19.6\,\beta^2)$), the shadow shifts negative while the matter-metric ISCO shifts positive — the conformal magnification pushes massive orbits outward while the geometric metric receives only an $\mathcal{O}(\eta^2)$ inward shift. The ISCO shift is $\sim 44\times$ larger in magnitude:

\begin{equation} \label{eq:appL_5}
\frac{\delta b}{b}\bigg|_{\rm shadow} = -0.044\% \quad (\mathcal{O}(\eta^2), \text{ photons on } g^{\rm sGB} = \tilde g),
\end{equation}

\begin{equation} \label{eq:appL_6}
\frac{\delta r_{\rm ISCO}}{r_{\rm ISCO}}\bigg|_{\tilde g} = +1.95\% \quad (\mathcal{O}(\eta), \text{ massive particles on } \tilde g).
\end{equation}

The coupling-order difference is a mathematical property of the perturbative sGB calculation (Section 6.2), not an indication of fractured metric structure. Matter, photons, and clocks all propagate universally on the single causal matter metric, $\tilde g_{\mu\nu}$. The $\sim 44\times$ coupling-order ratio and sign divergence arise purely from the geometry of the paths: because null geodesics ($d\tilde s^2 = 0$) are conformally invariant, the temporal scaling $A^2(\phi)$ divides out exactly, making the shadow sensitive only to the underlying backreacted geometry at $\mathcal{O}(\eta^2)$. Conversely, massive particles travel on timelike geodesics ($d\tilde s^2 < 0$) and therefore feel the conformal factor $A = e^{-\phi} > 1$ directly, shifting the ISCO at $\mathcal{O}(\eta)$. Both effects are complementary projections of the same unified temporal well. The matter-metric ISCO shift is an order of magnitude larger than the shadow shift ($+1.95\%$ vs $-0.044\%$) because the conformal factor contributes at $\mathcal{O}(\eta)$, not $\mathcal{O}(\eta^2)$. The would-be horizon shift $r_H = 2M(1 - 19.6\,\beta^2)$ is the sGB perturbative signature: the temporal field backreacts on the geometry through the sGB coupling, contracting the horizon of the horizon-bearing branch. The Temporal Horizon is observer-dependent and operational — a distant observer (fast clock) sees the accessibility boundary far from the centre; a deeper observer (stronger conformal field, larger $A$) sees less redshift to any given emitter, and the practical observability boundary moves inward.

The scaling with $\eta$ is:

- Shadow: $\delta b/b \propto \eta^2$ (geometric metric only, conformal factor cancels)

- ISCO on $g^{\rm sGB}$: $\delta r/r \propto \eta^2$ (geometric metric only)

- ISCO on $\tilde g$: $\delta r/r \propto \eta$ (conformal factor dominates)

All numbers are computed at $\eta = -0.1$ in the perturbative regime. Note that $|\eta| = 0.1$ exceeds current observational constraints ($\alpha_{\rm GB} < 2.9$ km$^2$ gives $|\eta| < 0.04$ for $M = 10\,M_\odot$); it is used as a perturbative reference value. Within current bounds, the largest shifts are for $M = 10\,M_\odot$: shadow $-0.007\%$, ISCO $+0.78\%$. The computation uses the Sotiriou \& Zhou (2014) exact $\mathcal{O}(\beta^2)$ metric perturbations and the analytic sGB scalar profile, with no free parameters.

# References

- Smawfield, M. L. (2025). Temporal Equivalence Principle: Dynamic Time & Emergent Light Speed. Paper 0 (Jakarta). DOI: 10.5281/zenodo.16921911.

- Schwarzschild, K. (1916). Über das Gravitationsfeld eines Massenpunktes nach der Einsteinschen Theorie. *Sitzungsber. Preuss. Akad. Wiss.*, 189–196.

- Penrose, R. (1965). Gravitational collapse and space-time singularities. *Phys. Rev. Lett.*, 14, 57–59.

- Hawking, S. W. (1966). The occurrence of singularities in cosmology. *Proc. Roy. Soc. A*, 294, 511–521.

- Hawking, S. W. & Penrose, R. (1970). The singularities of gravitational collapse and cosmology. *Proc. Roy. Soc. A*, 314, 529–548.

- Bekenstein, J. D. (1973). Black holes and entropy. *Phys. Rev. D*, 7, 2333–2346.

- Hawking, S. W. (1975). Particle creation by black holes. *Commun. Math. Phys.*, 43, 199–220.

- Bekenstein, J. D. (1993). Relation between physical and gravitational geometry. *Phys. Rev. D*, 48, 3641–3647. arXiv:gr-qc/9211017.

- Wald, R. M. (1993). Black hole entropy is the Noether charge. *Phys. Rev. D*, 48, R3427–R3431. arXiv:gr-qc/9307038.

- Bardeen, J. M. (1968). Non-singular general relativistic gravitational collapse. In *Proceedings of the 5th International Conference on Gravitation and the Theory of Relativity*, Tbilisi, p. 174.

- Hayward, S. A. (2006). Formation and evaporation of nonsingular black holes. *Phys. Rev. Lett.*, 96, 031103. arXiv:gr-qc/0506126.

- Mazur, P. O. & Mottola, E. (2004). Gravitational vacuum condensate stars. *Proc. Nat. Acad. Sci.*, 101, 9545–9550. arXiv:gr-qc/0109035.

- Mathur, S. D. (2005). The fuzzball proposal for black holes: an elementary review. *Fortsch. Phys.*, 53, 793–827. arXiv:hep-th/0502050.

- Almheiri, A., Marolf, D., Polchinski, J. & Sully, J. (2013). Black holes: complementarity or firewalls? *JHEP*, 02, 062. arXiv:1207.3123.

- Khoury, J. & Weltman, A. (2004). Chameleon cosmology. *Phys. Rev. D*, 69, 044026.

- Event Horizon Telescope Collaboration (2019). First M87 Event Horizon Telescope results. I. The shadow of the supermassive black hole. *ApJL*, 875, L1.

- Event Horizon Telescope Collaboration (2019). First M87 Event Horizon Telescope results. IV. Imaging the central supermassive black hole. *ApJL*, 875, L4.

- Event Horizon Telescope Collaboration (2019). First M87 Event Horizon Telescope results. V. Physical origin of the asymmetric ring. *ApJL*, 875, L5.

- Event Horizon Telescope Collaboration (2022). First Sagittarius A* Event Horizon Telescope results. I. The shadow of the supermassive black hole in the center of the Milky Way. *ApJL*, 930, L12.

- Smawfield, M. L. (2026). Temporal Equivalence Principle: A Covariant Alternative to Cosmic Expansion. Paper 26 (Athens). DOI: 10.5281/zenodo.20370143.

- Smawfield, M. L. (2026). Temporal Equivalence Principle: Native hi_class Conformal Implementation, Linear Perturbation Closure, and CMB Acoustic Peak Preservation. Paper 18 (Cambridge).

- Smawfield, M. L. (2026). Temporal Equivalence Principle: Temporal Horizon Cosmology and the Absence of a Physical Big Bang Singularity. Paper 27 (Thika). DOI: 10.5281/zenodo.20723059.

- LIGO/Virgo Collaboration (2016). Observation of gravitational waves from a binary black hole merger. *Phys. Rev. Lett.*, 116, 241102. DOI: 10.1103/PhysRevLett.116.241102.

- LIGO/Virgo Collaboration (2020). GW190521: A binary black hole merger with a total mass of 150 M<sub>☉</sub>. *Phys. Rev. Lett.*, 125, 101102. DOI: 10.1103/PhysRevLett.125.101102.

- Isi, M., et al. (2020). Posterior samples of GW190521 posterior samples. *Zenodo*. DOI: 10.5281/zenodo.4057131.

- Strohmayer, T. E. (2001). Discovery of high-frequency quasi-periodic oscillations in the black-hole binary GRS 1915+105. *ApJ*, 552, L49.

- Remillard, R. A., et al. (2002). XTE J1550-564: QPOs and black-hole spin. *ApJ*, 564, 962.

- Homan, J., et al. (2005). High-frequency QPOs in H 1743-322. *ApJ*, 624, 1005.

- Morgan, E. H., Remillard, R. A., & Greiner, J. (1997). RXTE observations of GRS 1915+105. *ApJ*, 482, 993.

- Harikane, E., et al. (2023). JWST CEERS: discovery of black-hole candidates at z > 8. *ApJ*, 958, 11.

- Greene, J. E., et al. (2023). CEERS AGN at z ~ 7. *ApJ*, 957, 24.

- Bogdan, A., et al. (2024). Evidence for a black-hole seed at z = 10.1. *Nature Astronomy*.

- Castellano, P., et al. (2024). JWST discovery of GHZ2 at z = 10.6. *Nature Astronomy*.

- Matsuoka, Y., et al. (2019). Subaru discovery of a z = 7.04 quasar. *ApJ*, 872, 2.

- Ba&ntilde;ados, E., et al. (2018). A black-hole mass of 8&times;10<sup>8</sup> M<sub>☉</sub> at z = 7.54. *Nature*, 553, 473.

- McClintock, J. E., Narayan, R., & Steiner, J. F. (2014). Black-hole spin via continuum fitting and reflection spectroscopy. *Space Sci. Rev.*, 183, 295. arXiv:1303.1583.

- Reynolds, C. S. (2021). X-ray reflection spin measurements and systematics. arXiv:2104.10300.

- Gou, L., et al. (2011). Cygnus X-1 spin via continuum fitting. *ApJ*, 742, 85.

- van Velzen, S., et al. (2021). Seventeen tidal disruption events from ZTF. *ApJ*, 908, 4.

- Holoien, T. W.-S., et al. (2016). ASASSN-14li: a nearby TDE. *MNRAS*, 455, 1690.

- Holoien, T. W.-S., et al. (2020). ASASSN-18pg: a luminous TDE. *ApJ*, 903, 151.

- Nicholl, M., et al. (2020). AT2019qiz: a nearby TDE with early-time spectroscopy. *MNRAS*, 499, 482.

- Event Horizon Telescope Collaboration (2021). First M87 Event Horizon Telescope results. VII. Polarization of the ring. *ApJL*, 910, L13.

- Event Horizon Telescope Collaboration (2024). Sgr A* polarimetry with the EHT. *ApJL*, 964, L26.

- Bekenstein, J. D. (2004). Conformal and disformal transformations. *Phys. Rev. D*, 70, 083509.

- Koivisto, T. (2012). Disformal equivalence. *Phys. Rev. D*, 85, 044043.

- Inayoshi, K., Haiman, Z., & Ostriker, J. P. (2020). Hyper-Eddington accretion and rapid growth of massive black holes. *MNRAS*, 496, 4236.

- Sotiriou, T. P. & Zhou, S.-Y. (2014). Black hole hair in generalized scalar-tensor gravity: An explicit example. *Phys. Rev. D*, 90, 124063. arXiv:1408.1698. [Companion to PRL 112, 251102 (2014); contains the exact $\mathcal{O}(\alpha^2)$ perturbative metric corrections used in Section 4. Their nonlinear solutions develop a finite-area singularity at $r \approx 0.27\,r_h$ — see Section 4.2.]

- Kanti, P., Mavromatos, N. E., Rizos, J., Tamvakis, K. & Winstanley, E. (1996). Dilatonic black holes in higher curvature string gravity. *Phys. Rev. D*, 54, 5049. arXiv:hep-th/9511071. [Establishes scalar hair and regular horizons in sGB; "absence of naked singularities" permits singularities hidden inside the horizon. Follow-up (Phys. Rev. D 57, 6255, 1998) confirms "other researchers... demonstrated numerically the existence of curvature singularities" behind the horizon.]

- Kleihaus, B., Kunz, J. & Radu, E. (2011). Rotating black holes in dilatonic Einstein-Gauss-Bonnet theory. *Phys. Rev. Lett.*, 106, 151104. arXiv:1101.2868. [Rotating sGB black holes; domain of existence bounded by singular extremal solutions.]

- Thaalba, F., Franchini, N., Bezares, M. & Sotiriou, T. P. (2024). Dynamics of spherically symmetric black holes in scalar-Gauss-Bonnet gravity with a Ricci coupling. *Phys. Rev. D*, 111, 064054. arXiv:2409.11398. [Confirms finite-area singularity in sGB; explores connection between singularity formation and loss of hyperbolicity; Ricci coupling can mitigate hyperbolicity loss.]

- Torii, T., Maeda, K. & Tamaoki, T. (1999). Rotating non-singular black holes in dilatonic Gauss-Bonnet gravity. *Phys. Rev. D*, 59, 064012.

- Delgado, J. F. M., Herdeiro, C. A. R., & Radu, E. (2020). Spin-induced scalarization and spontaneous scalarization of Kerr black holes in scalar-Gauss–Bonnet gravity. *Phys. Rev. D*, 102, 044041. arXiv:2007.12030.

- Delgado, J. F. M., Herdeiro, C. A. R. & Radu, E. (2020). Spinning black holes in shift-symmetric Horndeski theory. *JHEP*, 04, 180. arXiv:2002.05012. [Slowly-rotating shift-symmetric sGB solution; $\mathcal{O}(\beta^2)$ correction to $W(r)$ and horizon angular velocity; nonperturbative numerical solutions for arbitrary spin. Used in Section 8.5.]

- Bardeen, J. M. (1973). Rapidly rotating stars, disks, and black holes. In *Black Holes (Les Astres Occlus)*, eds. DeWitt & DeWitt, Gordon & Breach, pp. 241–289.

- Leaver, E. W. (1985). An analytical representation for the quasi-normal modes of Kerr black holes. *Proc. Roy. Soc. A*, 402, 285–298.

- Witek, H., et al. (2019). Scalar-Gauss–Bonnet gravity in the strong-field regime. *Phys. Rev. D*, 99, 064035. arXiv:1810.05177.

- Blazquez-Calzadilla, J., et al. (2020). Scalar-Gauss–Bonnet perturbations. *Phys. Rev. D*, 102, 024086. arXiv:2003.02862.

- Chen, Y., et al. (2024). Quasinormal modes of rotating black holes in shift-symmetric Einstein-scalar-Gauss–Bonnet theory. arXiv:2412.09377. [Nonperturbative rotating sGB QNM spectra; validation target for the TEP-sGB QNM pipeline.]

- Aresté Saló, L., Doneva, D. D., Clough, K., Figueras, P. & Yazadjiev, S. S. (2025). Challenges in the nonlinear evolution of unequal mass binaries in scalar-Gauss–Bonnet gravity. *Phys. Rev. D*, in press. arXiv:2507.13046. [Numerical relativity simulations of binary black hole mergers in sGB; ringdown excitation extraction.]

- Kobayashi, T., Yamaguchi, M. & Yokoyama, J. (2011). Galilean creation of the inflationary universe. *Prog. Theor. Phys.*, 126, 511. arXiv:1105.5723. [Principal symbol of scalar–tensor theories; tensor speed derivation used in Section 7.3.]

- Nishizawa, A. & Arai, K. (2019). Generalized framework for testing gravity with gravitational-wave propagation. *Phys. Rev. D*, 99, 104038. arXiv:1901.08249. [c_T formula for modified gravity theories including sGB; GW170817 constraints.]

- LIGO/Virgo & Fermi-GBM Collaborations (2017). Gravitational waves and gamma-rays from a binary neutron star merger: GW170817 and GRB 170817A. *ApJL*, 848, L13.

- Freire, P. C. C., et al. (2012). The relativistic pulsar-white dwarf binary PSR J0348+0432. *MNRAS*, 423, 3328.

- Antoniadis, J., et al. (2013). A massive pulsar in a compact relativistic binary. *Science*, 340, 6131.
