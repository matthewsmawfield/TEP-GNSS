# Temporal Equivalence Principle: Temporal Horizon Cosmology and the Absence of a Physical Big Bang Singularity
**Matthew Lukin Smawfield**
Version: v0.3 (Thika)
First published: 18 June 2026 - Last updated: 7 August 2026
DOI: 10.5281/zenodo.20723059

---

## Abstract

Standard FLRW cosmology extrapolates observed cosmic expansion backward to $a(t)\to0$, producing a Big Bang singularity at finite proper time. This paper demonstrates that this singularity is a reconstruction artifact of imposing a globally isochronous expanding-frame description on a conformal temporal geometry. In the Temporal Equivalence Principle (TEP), the observational role of FLRW expansion is reconstructed through conformal temporal transport: the effective scale factor $a_{\rm eff}$ arises from accumulated open-path conformal temporal shear along cosmological lines of sight rather than from physical expansion of space. TEP-C0 (Paper 26) established the distance-redshift and supernova evidence ; the full nonsingular matter-frame closure is delivered here.

The Temporal Horizon Cosmology framework is developed here, proving, within the temporal-conformal branch defined here, that the apparent $a_{\rm eff}\to0$ limit is not a physical curvature singularity but a temporal horizon. The effective scale factor $a_{\rm eff}$ is driven by the observational clock/redshift mapping $A_{\rm clock}(z)=(1+z)^{-1}$. Proposition 1 establishes curvature regularity of the temporal conformal boundary: for $A_{\rm clock}(\eta)=C\eta^{-p}$ with $0 \lt p\le\tfrac12$, all polynomial curvature invariants vanish at the boundary, timelike proper time diverges, and null geodesics have divergent affine parameter. The temporal-horizon exponent $p$ and the observational clock map are independent boundary conditions: $A_{\rm clock}(z)=(1+z)^{-1}$ is fixed by the redshift definition, while the regularity condition $0 \lt p\le\tfrac12$ is a mathematical requirement for curvature-regularity at the conformal boundary. Here $\eta$ is the temporal-horizon conformal coordinate, oriented so that approach to $\mathscr{T}^{-}$ corresponds to the asymptotic limit in which $A_{\rm clock}\to0$; it is not the standard FLRW conformal time coordinate extrapolated to $a=0$. Figure 1 (Section 4.5) illustrates the resulting conformal-boundary interpretation: the singular lower edge of standard flat $\Lambda$CDM is replaced by a smooth temporal conformal boundary $\mathscr{T}^{-}$, where $A_{\rm clock}\to0$ and curvature invariants vanish. The conformal compactification is smooth, the Weyl tensor vanishes on the boundary, and every causal curve approaches the regular past boundary $\mathscr{T}^{-}$ rather than terminating at a singularity. The temporal horizon is therefore simultaneously curvature-empty, timelike-complete, and null-complete in this branch.

The effective stress-energy tensor of the temporal field violates the Strong Energy Condition, an explicit prerequisite of the Hawking-Penrose singularity theorems. Rather than assuming the early universe was a globally hot expanding plasma, TEP replaces the hot Big Bang framework with Native Local Thermodynamic Evolution in an eternal universe. The temporal-horizon metric provides the geometric boundary where the clock rate vanishes, but it does not claim to uniquely derive primordial abundances. Instead, it delegates the thermal history to the eternal chemical-evolution framework (TEP-BBN, Paper 29), which provides an eternal-universe framework in which chemical states can approach a steady-state asymptotic equilibrium where D/H is no longer uniquely primordial and helium arises through baryonic cycling. The temporal-horizon thermal mapping preserves the observed CMB photon distribution through local decoupling processes without requiring a geometric singularity.

The scalar perturbation spectrum is derived from fluctuations of the clock field, $\zeta=\delta\ln A_{\rm clock}$, yielding a power spectrum $P_{\zeta}(k)\propto k^{n_{s}-1}$ with spectral-flow parameter $n_{s}-1=-2\epsilon_{\rm field}$. The observed Planck value $n_{s}=0.965$ constrains $\epsilon_{\rm field}=0.0175$. Tensor modes are derived directly from the temporal-conformal metric: for $A_{\rm clock}(\eta)\sim\eta^{-p}$ the tensor source term $A_{\rm clock}''/A_{\rm clock}=p(p+1)/\eta^{2}\to 0$ at the horizon, so the tensor equation approaches the Minkowski vacuum. The imported inflationary consistency relation $r=16\epsilon_{\rm field}$ is not assumed. Numerical integration of the native tensor equation across the finite transition profile (Step 09b) yields $r(k_{\rm pivot})=9\times 10^{-6}$ and $r_{\rm max}=6.26\times 10^{-4}$, both well below the BICEP/Keck 2021 bound $r\lt0.036$; tensor power is controlled only by the finite transition region. The late-time homogeneous expansion and acoustic observables (Planck 2018, BOSS DR12) are fully preserved by the conformal temporal mapping as mathematically confirmed in TEP-HC (Paper 18).

The causal matter-frame universe is curvature-regular at the temporal conformal boundary. The apparent Big Bang is a temporal horizon, not a physical curvature singularity. The temporal-horizon geometry supplies a nonsingular framework in which the standard background and acoustic observables are reconstructed by the conformal mapping, while TEP-BBN provides the native chemical-evolution mechanism and a proof of concept for local CMB thermalization; the scalar perturbation shape is reproduced, and the tensor-to-scalar ratio is computed from the native temporal-conformal wave equation, yielding values well below observational bounds.

Code Availability: https://github.com/matthewsmawfield/TEP-TH

Keywords: temporal equivalence principle, temporal horizon cosmology, big bang singularity, static conformal geometry, cosmology, modified gravity, temporal shear

# 1. Introduction: From Big Bang Singularity to Temporal Horizon

Standard FLRW cosmology extrapolates the observed cosmic expansion backward to $(a(t)\to0)$, producing a Big Bang singularity at finite proper time. This singular origin requires an initial hot dense state, followed by BBN nucleosynthesis, recombination, acoustic peak formation, CMB blackbody thermalization, and primordial perturbation generation. Hawking (1966) established, and Hawking and Penrose (1970) later generalized, that such a singularity is mathematically inevitable provided the Strong Energy Condition holds in globally hyperbolic spacetimes.

This paper demonstrates that the observational role of FLRW expansion is reconstructed through conformal temporal transport. In this framework, the distance-redshift relation and CMB acoustic scales are preserved in a static conformal geometry where the effective scale factor $a_{\rm eff}$ arises from accumulated open-path conformal temporal shear along cosmological lines of sight, not from physical expansion of space. The exact observational clock/redshift mapping $A_{\rm clock}(z)=(1+z)^{-1}$ drives the apparent $a_{\rm eff}\to0$ limit. The apparent $(a_{\rm eff}\to0)$ limit is therefore not a physical curvature singularity but a reconstruction artifact of imposing a globally isochronous expanding-frame description on a conformal temporal geometry.

While standard cosmology treats cosmic expansion as a kinematic stretching of the spatial metric, several frameworks have explored conformal alternatives. Wetterich (2013) demonstrated that a universe without spatial expansion can be formulated using a varying particle mass, while Narlikar and Arp explored conformal gravity variations. Environmental screening mechanisms—such as chameleon or symmetron screening (Khoury & Weltman 2004; Hinterbichler & Khoury 2010)—have been extensively developed to hide scalar fifth forces. TEP departs from these approaches by identifying the conformal factor strictly with the dynamical flow of proper time, generating an exact geometric mapping between the spatial scale factor and the macroscopic accumulation of Temporal Shear without requiring variable rest masses or modified spatial curvature.

Prior work established the empirical basis for this static conformal cosmology. TEP-C0 (Paper 26) demonstrated, through nested sampling over 1,701 Pantheon+ supernovae, that a pure conformal reconstruction exactly matches $\Lambda$CDM at the homogeneous distance-modulus level, while the physical no-$\Lambda$ temporal-shear branch improves the standardized supernova likelihood by approximately −3.4 in chi-squared for the conservative line-of-sight turnover $z_{\rm los}=5$ and approximately −7.5 for the fixed $z_{\rm los}=100$ benchmark. TEP-C0 identified full nonsingular matter-frame closure as the remaining open question for the cosmological sequence; the present paper delivers that closure for the temporal-conformal branch.

TEP-HC (Paper 18) implemented the native TEP interpretation directly in the `hi_class` Boltzmann code, deriving the runtime Bellini–Sawicki functions ($\alpha_M=-2\alpha_A$, $\alpha_B=2\alpha_A$, $\alpha_K=-5\alpha_A^2$, $\alpha_T=0$) and verifying that the static conformal geometry preserves the pre-recombination sound horizon with ratio $r_s^{\rm TEP}/r_s^{\Lambda\rm CDM} = 0.999994$ (corresponding to a $<6$ ppm deviation) with an active linear scalar perturbation sector that is stable and observationally negligible. TEP-TH closes the logical loop by demonstrating that the causal matter-frame geometry is regular at the temporal horizon, so the apparent Big Bang is not a physical singularity but an asymptotic boundary of the proper-time field.

#### Parameter-Scale and Amplitude Convention

**Turnover scales.**
**Amplitudes.** $\epsilon_{\rm field}=0.0175$ denotes the primordial spectral-flow parameter constrained by $n_s$. $\epsilon_{\rm dyn}$ denotes the dynamical temporal-horizon response. $\epsilon_T^{\rm los}$ denotes the late-time line-of-sight transport amplitude fitted in TEP-C0. $\epsilon_T^{\rm CMB}$ denotes the C0 background/acoustic diagnostic amplitude. $\epsilon_T^{\rm HC}=0.00602\pm0.00493$ denotes the native `hi_class` homogeneous conformal amplitude reported in TEP-HC. These are related projections of the same temporal sector, but they are not numerically interchangeable parameters.

In standard cosmology, epochs are conventionally defined by the chronological time elapsed since the physical singularity (e.g., "three minutes after the Big Bang" for nucleosynthesis). Because the temporal horizon in the TEP framework is an asymptotic boundary rather than a zero-volume origin, a global linear time coordinate $t$ cannot be extrapolated to a finite $t=0$. Consequently, the sequence of early-universe events is strictly mapped not by chronological time, but by the thermodynamic cooling of the plasma ($T$) and the evolution of the conformal clock-rate field. The history of the universe is preserved, but the chronological stopwatch is replaced by thermodynamic state variables.

TEP-TH addresses this early-universe question by demonstrating that the apparent Big Bang singularity is not a physical boundary but a temporal-horizon limit. The central result is that the apparent singularity corresponds to the limit $A_{\rm clock}\to0$, where the observational clock-rate field vanishes and proper time dilates to zero relative to the present epoch. The thermal history emerges natively through local thermodynamic evolution $\tilde\nabla_\mu \tilde T^{\mu\nu}=0$ in proper time, generating the CMB emission without assuming an initially hot, dense geometric singularity. The "Big Bang" is not a zero-volume, infinite-density origin of space, but a temporal horizon—an asymptotic boundary of the proper-time field.

This paper demonstrates this temporal-horizon replacement through a ten-step pipeline:

- **Temporal-Horizon Mapping**: Establish $A_{\rm clock}(z)=(1+z)^{-1}$ as the exact clock map and $A_{\rm dyn}(z)$ as the physical dynamical response

- **Matter-Frame Curvature**: Prove that for $A_{\rm clock}(\eta)=C\eta^{-p}$ with $0 \lt p\lt1$, all curvature invariants generated from the Ricci sector vanish; since the Weyl tensor vanishes in the conformally flat branch, all polynomial curvature invariants vanish at the temporal conformal boundary

- **Geodesic Completeness**: Demonstrate that for $0 \lt p\le\tfrac12$ the temporal horizon is simultaneously curvature-empty, timelike-complete, and null-complete

- **Effective Stress-Energy**: Evaluate energy conditions and demonstrate Hawking-Penrose consistency via SEC violation

- **BBN Abundance Framework**: Replace expanding thermal plasma assumptions with eternal Native Proper-Time Nucleosynthesis (delegated to TEP-BBN, Paper 29) to evaluate candidate asymptotic abundance solutions

- **Recombination Visibility**: Re-evaluate dynamic proper-time recombination $x_e(\tau)$ and derived physical acoustic scales within the native eternal-universe thermodynamic framework

- **CMB Blackbody Origin**: Attribute the physical origin of the photon distribution to local emission processes and confirm conformal transport preservation

- **Entropy and Arrow of Time**: Demonstrate thermodynamic regularity at the horizon

- **Primordial Perturbation Boundary**: Derive scalar fluctuations from $\zeta=\delta\ln A_{\rm clock}$ and tensor modes from the native temporal-conformal wave equation

- **CMB Anisotropy and LSS Consistency**: Relying on TEP-HC (Paper 18) to demonstrate that TT/TE/EE spectra, matter power spectrum, growth factor, and BAO scales match Planck and BOSS

The combined pipeline shows that the corresponding observational pillars can be reconstructed within the TEP framework. The causal matter-frame universe is curvature-regular at the temporal boundary: for $A_{\rm clock}(\eta)=C\eta^{-p}$ with $0 \lt p\lt1$, curvature invariants vanish at the boundary rather than merely remaining bounded; timelike proper time diverges for $0 \lt p\le 1$; null affine parameter diverges for $0 \lt p\le\tfrac12$; and the Strong Energy Condition is violated—satisfying the mathematical prerequisite established by Hawking and Penrose for a non-singular spacetime. The cleanest fully complete branch is $0 \lt p\le\tfrac12$, in which the temporal horizon is simultaneously curvature-empty, timelike-complete, and null-complete. BBN, recombination, CMB blackbody origin, entropy regularity, and the scalar shape of primordial perturbations are recovered without requiring a physical zero-volume, infinite-density expanding origin. Tensor modes are governed by a native temporal-conformal wave equation whose source term vanishes at the horizon; the imported inflationary consistency relation $r=16\epsilon_{\rm field}$ is not assumed. The apparent Big Bang is a temporal horizon, not a physical curvature singularity.

This work does not independently re-analyse Pantheon+ supernovae or perform the native hi_class Boltzmann perturbation closure; those are addressed in companion papers TEP-C0 (Paper 26) and TEP-HC (Paper 18).

The claim-discipline framework for the TEP corpus, including the scope limitations of canonical precision tests, is established in TEP-EXP (Paper 9).

# 2. Temporal-Horizon Mapping

The central temporal-horizon mapping establishes the relationship between the apparent FLRW singularity and the conformal clock-rate field. In standard FLRW cosmology, the Big Bang corresponds to the limit $a(t)\to0$ at finite proper time. In TEP, the effective scale factor is reconstructed from accumulated open-path conformal temporal shear. Two distinct projections of the temporal field are required to maintain consistency across all redshifts:

\begin{equation} \label{eq:aeff}
a_{\rm eff}(z) = A_{\rm clock}(z)\,a_{\rm m}(z) \, ,
\end{equation}

where $a_{\rm m}(z)=1$ in the clean static matter-frame case. The clock map

\begin{equation} \label{eq:A_clock}
A_{\rm clock}(z) = \frac{1}{1+z}
\end{equation}

is the exact observational clock/redshift mapping that drives $a_{\rm eff}\to0$ as $z\to\infty$. This is the temporal horizon: $A_{\rm clock}\to0$ is the conformal-temporal boundary, not a physical curvature singularity. The dynamical response

\begin{equation} \label{eq:A_dyn}
A_{\rm dyn}(z) = \left(1+\frac{z}{z_{t}}\right)^{-\epsilon_{\rm dyn}}
\end{equation}

is the dynamical shear response that modifies the Hubble parameter and perturbations at late times. It is crucial to distinguish these roles: $A_{\rm clock}$ is fixed by the redshift–distance relation and produces the apparent singularity, while $A_{\rm dyn}$ is a small fitted TEP correction. An observer situated in the early universe would experience local time advancing normally, as curvature invariants remain bounded. The limit $A_{\rm clock}\to0$ is strictly a relative observational boundary: as one looks backward from the present epoch, ancient clocks appear to tick progressively slower. The horizon is the mathematical asymptote where this relative clock rate approaches zero, not a physical location where time itself ceases to exist.

The pipeline computes both projections across redshift $z\in[0,10^{4}]$. The clock map $A_{\rm clock}(z)=(1+z)^{-1}$ reproduces the standard redshift scaling exactly, while the dynamical response $A_{\rm dyn}(z)$ approaches unity at early times. The effective Hubble parameter is therefore

\begin{equation} \label{eq:H_tep_split}
H_{\rm TEP}(z) = \frac{H_{\rm LCDM}(z)}{A_{\rm dyn}(z)} \, ,
\end{equation}

which encodes the late-time TEP shear correction. This separation resolves the apparent tension between $a_{\rm eff}\to0$ at the horizon and the requirement that $H_{\rm TEP}\approx H_{\rm LCDM}$ during the thermal epochs.

The static conformally-flat matter-frame representation is not an assumption; it follows from the TEP disformal coupling. The foundational TEP papers establish the matter coupling through $\tilde{g}_{\mu\nu}=A^{2}g_{\mu\nu}+B\,\nabla_{\mu}\phi\,\nabla_{\nu}\phi$. On the homogeneous cosmological background the field depends only on conformal time, $\phi=\phi(\eta)$, so $\nabla_{\mu}\phi=\delta_{\mu}^{0}\,\phi'$ is purely temporal. The disformal term therefore rescales only $g_{00}$, which a time reparameterisation absorbs, leaving the spatial metric unchanged and the causal metric conformally flat. The static matter frame $a_{\rm m}=1$ is the coordinate choice in which this disformal reduction is manifest.

## 2.1 Sector Consistency with Foundational TEP

The foundational TEP metric is conformal-disformal,

\begin{equation}
\tilde{g}_{\mu\nu} = A^{2}(\phi)\,g_{\mu\nu} + B(\phi)\,\nabla_{\mu}\phi\,\nabla_{\nu}\phi \, .
\end{equation}

The present paper studies the homogeneous cosmological projection of this structure. On an FLRW background ($\phi=\phi(\eta)$), the disformal term is purely temporal and can be absorbed into the lapse by a time reparameterisation. The resulting temporal-horizon theorem is therefore a theorem of the conformal clock-rate sector ($A_{\rm clock}$), not a derivation of the non-exact synchronization-holonomy sector.

This distinction is essential. Open-path temporal transport,

\begin{equation}
\Delta\ln A = \int_{\gamma} d\ln A \, ,
\end{equation}

can generate cosmological redshift and the apparent scale-factor reconstruction. Closed-loop residual synchronization holonomy, however, obeys

\begin{equation}
\oint d\ln A = 0
\end{equation}

for smooth single-valued conformal transport. Therefore TEP-TH does not claim that $A_{\rm clock}$ alone generates nonzero closed-loop holonomy. Nonzero residual holonomy requires disformal or otherwise non-exact synchronization curvature, as developed in the foundational TEP paper (Jakarta). The temporal-horizon result and the synchronization-holonomy programme are complementary probes of different sectors of the same causal matter metric. The homogeneous temporal-shear background contribution $\Omega_\phi$ (TEP-HC, Paper 18) fills the same cosmological background budget as $\Omega_\Lambda$, while the non-exact covariance/topology correction $C_T$ (TEP-C0, Paper 26) provides the transport closure for the open-path redshift reconstruction beyond exact conformal shear.

The redshift map and the conformal-time horizon profile are linked by $A_{\rm clock}(z)=(1+z)^{-1}$ and $A_{\rm clock}(\eta)=C\eta^{-p}$, which together imply

\begin{equation} \label{eq:z_eta}
1+z(\eta) = C^{-1}\eta^{p} \, .
\end{equation}

This relation connects the observational redshift coordinate, used for distance measurements and epoch identification, to the conformal time parameter in which the curvature and geodesic theorems are proved. The constant $C$ is fixed by the condition $A_{\rm clock}=1$ at the present epoch ($z=0$, $\eta=\eta_{0}$), giving $C=\eta_{0}^{-p}$.

# 3. Matter-Frame Geometry and Curvature Regularity

To determine whether the temporal horizon is a physical singularity or a reconstruction artifact, matter-frame curvature invariants are computed. In the causal matter frame the metric is $\tilde{g}_{\mu\nu} = A_{\rm clock}^{2}(\eta)\,g_{\mu\nu}$, where $g_{\mu\nu}$ is the static conformal background and $A_{\rm clock}(\eta)$ is the conformal clock-rate field. The key question is whether curvature invariants remain finite as $A_{\rm clock}\to0$, and whether they in fact vanish for a suitable horizon profile.

The effective scale factor observed by matter is not the raw conformal factor but the product

\begin{equation} \label{eq:aeff_matter}
a_{\rm eff} = A_{\rm clock}\,a_{\rm m} \, ,
\end{equation}

where $a_{\rm m}=1$ in the clean static matter-frame case. The apparent FLRW limit $a_{\rm eff}\to0$ is driven by $A_{\rm clock}\to0$, while the underlying matter-frame coordinate scale factor $a_{\rm m}$ remains finite and non-zero. Consequently the physical matter-frame metric determinant is

\begin{equation} \label{eq:metric_det}
\det(\tilde{g}_{\mu\nu}) = A_{\rm clock}^{8}\,\det(g_{\mu\nu}) = -A_{\rm clock}^{8}\,a_{\rm m}^{6} \, .
\end{equation}

In four spacetime dimensions every metric component acquires a factor $A_{\rm clock}^{2}$, giving $A_{\rm clock}^{8}$ in the determinant. Because $a_{\rm m}$ does not collapse to zero, $\det(\tilde{g})$ vanishes only if $A_{\rm clock}\to0$ with no compensating behaviour. The determinant alone does not prove regularity; one must show that the *physical* volume element, constructed from the inverse metric and the relevant measure, does not degenerate in a way that produces divergent curvature.

The full four-dimensional conformal transformation of the Ricci scalar for $\tilde{g}_{\mu\nu}=A_{\rm clock}^{2}\,g_{\mu\nu}$ is

\begin{equation} \label{eq:ricci_scalar_full}
\tilde{R} = A_{\rm clock}^{-2}\Bigl[ R - 6\,\Box\ln A_{\rm clock} - 6\,(\nabla\ln A_{\rm clock})^{2} \Bigr] \, .
\end{equation}

Equivalently, writing everything in terms of $A_{\rm clock}$ itself,

\begin{equation} \label{eq:ricci_scalar_A}
\tilde{R} = A_{\rm clock}^{-2}R - 6A_{\rm clock}^{-3}\Box A_{\rm clock} \, .
\end{equation}

In the TEP static background $g_{\mu\nu}$ has $R=0$ and $\nabla A_{\rm clock}$ is spatially homogeneous, so the surviving terms are controlled by the background profile $A_{\rm clock}(\eta)$. For the temporal-horizon profile $A_{\rm clock}(\eta)=C\eta^{-p}$ with $0 \lt p\lt1$, the dominant term is $\tilde{R}=6A_{\rm clock}''/A_{\rm clock}^{3}=6p(p+1)C^{-2}\eta^{2p-2}$, which vanishes as $\eta\to\infty$.

The Kretschmann scalar and Ricci-tensor square transform with additional powers of $A_{\rm clock}^{-1}$. For a conformally flat static background the Weyl tensor vanishes, so the full Riemann tensor is built from the Ricci tensor. The conformal transformation of $\tilde{R}_{\mu\nu}$ (Wald, App. D) with $\ln A_{\rm clock}=\ln C-p\ln\eta$ and the static flat background gives the closed-form Ricci-tensor invariant

\begin{equation}
\tilde{R}_{\mu\nu}\tilde{R}^{\mu\nu} = 12p^{2}\bigl(p^{2}+p+1\bigr)\,\frac{\eta^{4p-4}}{C^{4}} \, .
\end{equation}

The Kretschmann invariant in the zero-Weyl case is $\tilde{K}=2\tilde{R}_{\mu\nu}\tilde{R}^{\mu\nu}-\tfrac13\tilde{R}^{2}$, which evaluates to

\begin{equation} \label{eq:kretschmann}
\tilde{K} = 12p^{2}\left(p^{2}+1\right)\frac{\eta^{4p-4}}{C^{4}} \, .
\end{equation}

For $0 \lt p\lt1$, the factor $\eta^{4p-4}\to0$ as $\eta\to\infty$, so both the Kretschmann scalar and the Ricci-tensor invariant vanish at the boundary with an explicit polynomial prefactor in $p$. The scaling is not approximate; it is a closed-form result derived from the conformal transformation of the Ricci tensor on the static background.

## 3.1 Proposition 1 — Curvature Regularity of the Temporal Conformal Boundary

**Proposition 1 (Curvature regularity of the temporal conformal boundary).** Let the observational effective scale factor be $a_{\rm eff}=A_{\rm clock}\,a_{\rm m}$ with $a_{\rm m}$ the finite, non-zero matter-frame coordinate scale factor. Let the clock field take the temporal-horizon profile $A_{\rm clock}(\eta)\sim\eta^{-p}$ in conformal time with $0 \lt p\lt1$. Then, as $\eta\to\infty$ ($A_{\rm clock}\to0$):

- The Ricci scalar $\tilde{R}$ in the matter frame vanishes: $\tilde{R}\sim\eta^{2p-2}\to0$.

- The Kretschmann scalar $\tilde{K}$ and Ricci-tensor invariant $\tilde{R}_{\mu\nu}\tilde{R}^{\mu\nu}$ vanish: $\tilde{K}\sim\tilde{R}_{\mu\nu}\tilde{R}^{\mu\nu}\sim\eta^{4p-4}\to0$.

- The observational conformal volume element $V_{\rm eff}=A_{\rm clock}^{3}a_{\rm m}^{3}$ tends to zero, but this is not a physical collapse of the underlying matter-frame spatial geometry. It reflects the degeneracy of the observational clock/redshift map at the temporal horizon. The finite coordinate scale $a_{\rm m}$ and vanishing curvature invariants show that the boundary is a conformal-temporal endpoint, not a zero-volume curvature singularity.

*Proof sketch.* Substitute $A_{\rm clock}(\eta)=C\eta^{-p}$ into equations \eqref{eq:ricci_scalar_full}–\eqref{eq:kretschmann}. In the static conformal background $R=0$ and spatial gradients vanish by homogeneity. The time derivatives scale as $A_{\rm clock}'\sim -pC\eta^{-p-1}$ and $A_{\rm clock}''\sim p(p+1)C\eta^{-p-2}$. The Ricci scalar \eqref{eq:ricci_scalar_A} contains terms $A_{\rm clock}^{-2}R$ and $A_{\rm clock}^{-3}\Box A_{\rm clock}$. With $R=0$ and homogeneity, the dominant term is $6A_{\rm clock}''/A_{\rm clock}^{3}=6p(p+1)C^{-2}\eta^{2p-2}\to0$ for $0 \lt p\lt1$. The Ricci-tensor invariant has coefficient $12p^{2}(p^{2}+p+1)/C^{4}$ and the Kretschmann invariant has coefficient $12p^{2}(p^{2}+1)/C^{4}$, which satisfy the zero-Weyl identity $\tilde{K}=2\tilde{R}_{\mu\nu}\tilde{R}^{\mu\nu}-\tfrac13\tilde{R}^{2}$; both are strictly positive for all $p>0$ and scale as $\eta^{4p-4}\to0$ for $0 \lt p\lt1$. Numerical evaluation confirms vanishing to machine precision across $\eta\in[1,10^{4}]$. $\square$

For $p\le\tfrac12$ the Kretschmann scalar falls faster than $\eta^{-2}$, ensuring that null affine parameters diverge (Section 4.2).

This establishes the temporal boundary as a regular conformal endpoint, adopting the mathematical architecture of Penrose's conformal compactification and null infinity ($\mathscr{I}^{+}$) (Penrose 2006; Tod 2003, 2015), here applied to the asymptotic past. The boundary is not a curvature singularity; it is a conformal-temporal boundary where the Lorentzian metric becomes degenerate in the observational clock frame.

It is vital to distinguish the TEP temporal horizon ($\mathscr{T}^{-}$) from Penrose's Conformal Cyclic Cosmology (CCC). Where CCC posits an infinite sequence of aeons connected by conformal rescalings at $\mathscr{I}^{+}$ and $\mathscr{I}^{-}$, TEP does not invoke cyclicality. $\mathscr{T}^{-}$ functions as a strict, non-singular asymptotic past boundary for a single, continuous causal history, dictated by the vanishing of the relative conformal clock rate.

The physical interpretation is that the limit $A_{\rm clock}\to0$ is a *temporal horizon*: curvature invariants do not merely remain bounded—they vanish—so the boundary is asymptotically curvature-empty. The local clock rate vanishes relative to the present epoch, but the spatial geometry does not develop pathological curvature. The analogy is not a collapsing sphere reaching zero volume; by analogy with the standard GR event-horizon description, it is an observer falling toward an event horizon whose local clock asymptotically freezes. The causal structure remains intact. Just as an observer crossing a black hole event horizon experiences normal local time while appearing perfectly frozen to a distant observer, the extremely early universe experiences regular local thermodynamic evolution while appearing frozen relative to the present epoch. The "start" of the universe is therefore shielded by a relativistic horizon of clock-transport, preventing infinite extrapolation.

# 4. Geodesic Completeness

A key test for physical singularities is geodesic incompleteness. Null and timelike geodesics are integrated backward from the present epoch toward the temporal horizon. For the temporal-horizon profile $A_{\rm clock}(\eta)=C\eta^{-p}$ with $0 \lt p\le\tfrac12$, both null and timelike geodesics are complete: the null affine parameter diverges and the timelike proper time diverges. For $0 \lt p\lt1$ the timelike branch is already complete; null completeness requires the slightly stronger bound $p\le\tfrac12$. The boundary is curvature-empty in this branch.

Here $\eta$ is the temporal-horizon conformal coordinate, oriented so that approach to $\mathscr{T}^{-}$ corresponds to the asymptotic limit in which $A_{\rm clock}\to0$. It is not the standard FLRW conformal time coordinate extrapolated to $a=0$.

## 4.1 Conformal Christoffel Symbols

Under the conformal transformation $\tilde{g}_{\mu\nu}=A_{\rm clock}^{2}\,g_{\mu\nu}$, the Christoffel symbols transform as

\begin{equation} \label{eq:christoffel}
\tilde{\Gamma}^{\mu}{}_{\alpha\beta}
= \Gamma^{\mu}{}_{\alpha\beta}
+ \delta^{\mu}_{\alpha}\,\partial_{\beta}\ln A_{\rm clock}
+ \delta^{\mu}_{\beta}\,\partial_{\alpha}\ln A_{\rm clock}
- g_{\alpha\beta}\,g^{\mu\nu}\,\partial_{\nu}\ln A_{\rm clock} \, .
\end{equation}

For the static conformal background $g_{\mu\nu}$ with $\partial_{i}A_{\rm clock}=0$ and $A_{\rm clock}=A_{\rm clock}(t)$, the nonzero modified symbols are

\begin{equation}
\tilde{\Gamma}^{0}{}_{00} = \frac{\dot{A}_{\rm clock}}{A_{\rm clock}} \, , \qquad
\tilde{\Gamma}^{0}{}_{ij} = \frac{\dot{A}_{\rm clock}}{A_{\rm clock}}\,\delta_{ij} \, , \qquad
\tilde{\Gamma}^{i}{}_{0j} = \frac{\dot{A}_{\rm clock}}{A_{\rm clock}}\,\delta^{i}_{j} \, ,
\end{equation}

where overdots denote derivatives with respect to the conformal time of the background. These corrections encode how the varying clock rate modifies geodesic acceleration.

## 4.2 Null Geodesics and Affine Length

For radial null geodesics in the static matter frame ($a_{\rm m}=1$), the geodesic equation with the corrected Christoffels gives

\begin{equation} \label{eq:null_geodesics}
\frac{d\eta}{d\tilde{\lambda}} = \frac{1}{A_{\rm clock}^{2}(\eta)} \, , \qquad
\frac{dr}{d\tilde{\lambda}} = \frac{\pm 1}{A_{\rm clock}^{2}(\eta)} \, ,
\end{equation}

where $\tilde{\lambda}$ is the affine parameter in the matter frame. A standard result for conformally related metrics is that if $k^{\mu}=dx^{\mu}/d\lambda_{0}$ is an affinely parametrized null geodesic of the background $g_{\mu\nu}$, then the same path is an affinely parametrized null geodesic of $\tilde{g}_{\mu\nu}=A_{\rm clock}^{2}g_{\mu\nu}$ with parameter $\tilde{\lambda}$ obeying $d\tilde{\lambda}=A_{\rm clock}^{2}\,d\lambda_{0}$ (in four dimensions). The background is static and nonsingular, so $\lambda_{0}$ ranges over $(-\infty,\infty)$. The matter-frame affine parameter to any background coordinate time $t$ is therefore

\begin{equation} \label{eq:affine_null}
\tilde{\lambda}(t) = \int^{t} A_{\rm clock}^{2}(\lambda_{0})\,d\lambda_{0}
= \int^{t} A_{\rm clock}^{2}(\eta)\,d\eta \, ,
\end{equation}

where $\eta$ is the background coordinate time. For the temporal-horizon profile $A_{\rm clock}(\eta)=C\eta^{-p}$ with $0 \lt p\lt1$ as $\eta\to\infty$ (early times), the integrand behaves as $\eta^{-2p}$. The integral $\int^{\infty}\eta^{-2p}\,d\eta$ diverges for $2p\le 1$, i.e. for $0 \lt p\le\tfrac12$. Null geodesics do not terminate in this branch; they extend to $\tilde{\lambda}\to\infty$. For $\tfrac12\lt p \lt 1$ the null affine parameter converges, so null completeness requires $p\le\tfrac12$.

The divergence rates are explicit. Evaluating the integral with a lower cutoff $\eta_{0}\sim\mathcal{O}(1)$,

\begin{equation} \label{eq:null_divergence}
\tilde{\lambda}(\eta)=\int_{\eta_{0}}^{\eta}C^{2}\eta'^{-2p}\,d\eta'=\frac{C^{2}}{1-2p}\Bigl(\eta^{1-2p}-\eta_{0}^{1-2p}\Bigr)\quad(p\neq\tfrac12) \, ,
\end{equation}

so that for $p<\tfrac12$ the null affine parameter diverges as a power law, $\tilde{\lambda}\propto\eta^{1-2p}\to\infty$. At the marginal value $p=\tfrac12$,

\begin{equation} \label{eq:null_log}
\tilde{\lambda}(\eta)=C^{2}\,\ln\!\left(\frac{\eta}{\eta_{0}}\right) \to \infty \quad (p=\tfrac12) \, ,
\end{equation}

diverging only logarithmically. The slower divergence at $p=\tfrac12$ is still sufficient for past-completeness; every finite segment of a null geodesic accumulated since any finite $\eta_{0}$ requires infinite affine parameter to reach $\mathscr{T}^{-}$.

## 4.3 Timelike Geodesics and Proper Time

For timelike geodesics the conformal transformation rescales proper time as $d\tilde{\tau}=A_{\rm clock}\,d\tau$. In the static background a comoving observer has $d\tau=d\eta$, so the total matter-frame proper time from early time $\eta_{i}$ to the present is

\begin{equation} \label{eq:timelike_proper}
\tilde{\tau}_{\rm total} = \int_{\eta_{i}}^{\eta_{0}} A_{\rm clock}(\eta)\,d\eta \, .
\end{equation}

For the temporal-horizon profile $A_{\rm clock}(\eta)=C\eta^{-p}$ with $0 \lt p\lt1$, the integral $\int^{\infty}\eta^{-p}\,d\eta$ *diverges* because $p\le 1$. Timelike proper time to the temporal boundary is therefore *infinite*. This is a major strengthening: timelike geodesics are complete for the entire $0 \lt p\lt1$ branch. In TEP, Proposition 1 guarantees that all curvature invariants *vanish* at the temporal horizon for $0 \lt p\lt1$, so the infinite proper time reflects that observers can continue evolving indefinitely while the geometry becomes asymptotically curvature-empty.

The timelike divergence is also explicit:

\begin{equation} \label{eq:timelike_divergence}
\tilde{\tau}(\eta)=\int_{\eta_{0}}^{\eta}C\eta'^{-p}\,d\eta'=\frac{C}{1-p}\Bigl(\eta^{1-p}-\eta_{0}^{1-p}\Bigr)\quad(p\neq1) \, ,
\end{equation}

which diverges as a power law $\tilde{\tau}\propto\eta^{1-p}\to\infty$ for all $0\lt p\lt1$. The timelike bound $p\lt1$ is weaker than the null bound $p\le\tfrac12$; every timelike observer accumulates infinite proper time before reaching $\mathscr{T}^{-}$. The geodesic equations in the matter frame are regular everywhere:

\begin{equation} \label{eq:timelike_geodesics}
\frac{d^{2}x^{\mu}}{d\tilde{\tau}^{2}} + \tilde{\Gamma}^{\mu}{}_{\alpha\beta}\frac{dx^{\alpha}}{d\tilde{\tau}}\frac{dx^{\beta}}{d\tilde{\tau}} = 0 \, ,
\end{equation}

with $\tilde{\Gamma}$ given by \eqref{eq:christoffel}. Since $\tilde{\Gamma}\sim A_{\rm clock}^{-1}\partial A_{\rm clock}$ is bounded and the four-velocity is normalized, the equations admit smooth solutions that approach the horizon asymptotically. Numerical integration confirms this: both null and timelike geodesics approach $A_{\rm clock}\to0$ without encountering divergent curvature or coordinate breakdown.

## 4.4 Geodesic Completeness Theorem

**Theorem (Temporal-horizon geodesic completeness).** Let the conformal clock field take the profile $A_{\rm clock}(\eta)=C\eta^{-p}$ with $0 \lt p\le\tfrac12$. Then:

- Timelike proper time to the temporal boundary diverges for $0 \lt p\le 1$, hence in particular for $0 \lt p\le\tfrac12$.

- Null affine parameter to the temporal boundary diverges for $0 \lt p\le\tfrac12$.

- All polynomial curvature invariants vanish at the boundary for $0 \lt p\lt1$, hence in particular for $0 \lt p\le\tfrac12$.

- The tensor source term $A_{\rm clock}''/A_{\rm clock}=p(p+1)/\eta^{2}\to 0$ at the boundary.

The cleanest fully complete temporal-horizon branch is therefore $0 \lt p\le\tfrac12$. In this branch the temporal horizon is simultaneously curvature-empty, timelike-complete, and null-complete.

Proposition 1 establishes regularity for the temporal-conformal branch satisfying $0 \lt p\le\tfrac12$. The temporal-horizon exponent $p$ and the observational clock map are independent boundary conditions: $A_{\rm clock}(z)=(1+z)^{-1}$ is fixed by the redshift definition, while the regularity condition $0 \lt p\le\tfrac12$ is a mathematical requirement for curvature-regularity at the conformal boundary. The profile $A_{\rm clock}\sim\eta^{-p}$ with $p=1$ would give the standard FLRW clock map, but that branch ($p=1$) lies outside the curvature-regular window. Within TEP, the regular branch is selected by the requirement that the matter-frame geometry be curvature-regular at the temporal horizon, not by the observational redshift mapping alone.

## 4.5 Conformal Compactification and Penrose-Style Diagram

To illustrate the temporal boundary in the language of standard conformal diagrams, we construct a Penrose-style compactified conformal diagram for the TEP matter-frame spacetime and compare it directly with the standard flat $\Lambda$CDM diagram. The metric in conformal coordinates is

\begin{equation} \label{eq:metric_conformal}
d\tilde{s}^{2} = a_{\rm eff}^{2}(\eta)\left(-d\eta^{2} + dr^{2} + r^{2}\,d\Omega^{2}\right) \, .
\end{equation}

In both the standard and TEP conformal diagrams, time and space are compactified independently so that infinite spacetime maps to a finite rectangle. Let $\eta_{0}$ denote the present-epoch conformal time. For the *standard* flat $\Lambda$CDM cosmology, conformal time runs from $\eta=0$ (Big Bang) to $\eta=\eta_{0}$ (present), and the compactification

\begin{equation} \label{eq:compact_coords_lcdm}
T_{\Lambda} = \arctan\!\left(\frac{\eta}{\eta_{0}}\right) \, , \qquad R = \arctan\!\left(\frac{r}{\eta_{0}}\right) \, ,
\end{equation}

maps $\eta\in[0,\eta_{0}]$ and $r\in[0,\infty)$ to the rectangle $0\le T_{\Lambda}\le\pi/4$, $0\le R\lt\pi/2$. The bottom edge $T_{\Lambda}=0$ is the physical singularity: a spacelike boundary where $a\to0$ and all curvature invariants diverge. Every causal curve terminates there in finite proper time.

For the *TEP* temporal-horizon cosmology, conformal time runs from $\eta=\eta_{0}$ (present) to $\eta\to\infty$ (temporal horizon), and the compactification

\begin{equation} \label{eq:compact_coords_tep}
T = \arctan\!\left(\frac{\eta_{0}}{\eta}\right) \, , \qquad R = \arctan\!\left(\frac{r}{\eta_{0}}\right) \, ,
\end{equation}

maps $\eta\in[\eta_{0},\infty)$ and $r\in[0,\infty)$ to the same rectangle $0\le T\le\pi/4$, $0\le R\lt\pi/2$. The bottom edge $T=0$ is the temporal horizon $\mathscr{T}^{-}$: a *regular* spacelike boundary where $A_{\rm clock}\to0$ and all curvature invariants vanish (Proposition 1). Null geodesics approach it with infinite affine parameter; timelike curves approach it with infinite proper time.

The compactified causal layout is analogous, while the physical content of the lower boundary differs radically.

**Figure 1.** Penrose-style schematic compactified conformal diagrams for (a) standard flat ΛCDM and (b) the TEP temporal-horizon branch. Both diagrams show a compactified causal slab; the upper edge is the present epoch, not future infinity. **Panel (a):** the lower spacelike boundary is the Big Bang singularity, where $a\to0$, curvature diverges ($\tilde{\mathcal{K}}\to\infty$), and past-directed causal curves terminate. **Panel (b):** the corresponding lower boundary is a smooth regular temporal conformal horizon $\mathscr{T}^{-}$, where $A_{\rm clock}\to0$ and curvature invariants vanish ($\tilde{\mathcal{K}}\to0$). Null geodesics are dashed red and timelike curves solid blue. In the TEP panel, the lower boundary is approached only asymptotically: null curves in infinite affine parameter and timelike curves in infinite proper time. Tick marks on the timelike curve denote equal proper-time intervals compressed by the compactification. The compactified causal layout is analogous, but the physical content of the lower boundary differs radically.

The TEP diagram (Panel b) is a rectangle with horizontal top and bottom edges and vertical left and right edges. In this schematic conformal representation, null geodesics are drawn at $45°$ to emphasize the causal structure; a fully metric-derived Penrose compactification would require constructing compactified null coordinates explicitly from the conformal metric. In a generic compactified coordinate representation their apparent slope can vary with position, but they remain causal and approach the spacelike boundary $\mathscr{T}^{-}$ only asymptotically. The boundary of this rectangle comprises:

- **$\mathscr{T}^{-}$ (temporal past infinity):** the horizontal line $T=0$ ($0\le R\lt\pi/2$), corresponding to $\eta\to\infty$ at all finite $r$. This is the TEP temporal horizon. It is *not* a curvature singularity; all polynomial invariants vanish there (Proposition 1), and the conformal factor $a_{\rm eff}\to 0$ merely reflects the vanishing of the relative clock rate.

- **Present epoch:** the horizontal line $T=\arctan(1)=\pi/4$ ($0\le R\lt\pi/2$), corresponding to $\eta=\eta_{0}$ at all finite $r$. This is the observer's present slice.

- **$r=0$ (spatial origin):** the vertical line $R=0$ ($0\le T\le\pi/4$), the symmetry axis of the radial coordinate.

- **$i^{0}$ (spatial infinity):** the vertical line $R\to\pi/2$ ($0\le T\le\pi/4$), approached along spacelike curves with $r\to\infty$.

- **$i^{+}$ (future timelike infinity):** the corner $(T,R)=(\pi/4,0)$, where comoving timelike geodesics reach the present epoch.

Crucially, *every* inextendible causal curve in the TEP diagram has its past limit on $\mathscr{T}^{-}$ and its future limit on the present-epoch boundary; there is no singularity anywhere in the diagram.

**Comparison with standard cosmology.** Figure 1 makes the distinction immediate. In the standard diagram (Panel a), the past boundary is a *spacelike singularity* (the Big Bang) where all curvature invariants diverge and every causal curve terminates in finite proper time. In the TEP diagram (Panel b), the past boundary is $\mathscr{T}^{-}$, a *regular conformal-temporal boundary* where curvature invariants vanish; null geodesics have infinite affine parameter, while timelike curves have infinite proper time. The TEP diagram shares the same compactification topology as the conformal diagram of global de Sitter space: both have a smooth horizontal spacelike past boundary, a smooth horizontal future boundary, and vertical spatial-infinity edges. The compactified causal layout is topologically analogous; the physical mechanism differs—de Sitter's past boundary is $\mathscr{I}^{-}$, an asymptotic surface of an exponentially expanding congruence, whereas TEP's past boundary is $\mathscr{T}^{-}$, the limit where the relative conformal clock rate vanishes on a nonexpanding static background.

**Boundary-regularity interpretation.** For the temporal-horizon profile $A_{\rm clock}(\eta)=C\eta^{-p}$ with $0 \lt p\le\tfrac12$:

- The conformal compactification $(\tilde{M},\tilde{g}_{\mu\nu})$ with conformal boundary $\partial\tilde{M}=\mathscr{T}^{-}\cup\{\text{present}\}\cup i^{0}\cup i^{+}$ is smooth.

- The Weyl tensor vanishes on $\mathscr{T}^{-}$ (conformally flat boundary).

- All inextendible causal curves have past endpoint on the spacelike boundary $\mathscr{T}^{-}$ and future endpoint on the present-epoch surface or future null infinity.

- There exists no trapped surface or incomplete geodesic; the framework complies with the mathematical constraints of the Hawking-Penrose singularity theorems by violating the Strong Energy Condition (Section 5).

The analytic curvature and geodesic results place $\mathscr{T}^{-}$ on the footing of a regular spacelike conformal boundary; Figure 1 gives a Penrose-style schematic representation of that boundary structure. It is a boundary of the manifold, not a singularity within it.

## 4.6 Implications for Quantum Cosmology

The regularity of the temporal conformal boundary $\mathscr{T}^{-}$ removes the standard motivation for a quantum gravitational "beginning" at finite proper time. In standard Big Bang cosmology, the curvature singularity at $a\to0$ forces a breakdown of the classical theory at a finite past, providing the principal physical justification for quantum-cosmological models that attempt to describe a first moment of time. Because TEP's temporal horizon is curvature-empty and timelike-complete, the classical geometry remains valid arbitrarily far into the past; there is no finite-time singularity that demands quantum replacement.

This opens the possibility of a well-defined wavefunction on the regular boundary $\mathscr{T}^{-}$, formulated as a boundary condition on a smooth conformal compactification rather than as an initial condition at a singular point. The boundary data would describe correlations across the temporal horizon, analogous to how boundary-value problems are posed on smooth spatial or null boundaries in standard field theory. Whether such a boundary quantum state reproduces the observed classical late-time cosmology is a well-posed mathematical question; it does not require a speculative resolution of a curvature singularity that does not exist in the TEP matter frame.

# 5. Effective Stress-Energy and Hawking-Penrose Consistency

The Hawking-Penrose singularity theorems establish that a singularity is mathematically inevitable if the Strong Energy Condition (SEC) holds: $\rho + 3p \geq 0$ and $\rho + p \geq 0$. The theorems require the SEC as an explicit assumption; if the SEC is violated, their conclusion of inevitable geodesic incompleteness no longer applies. The effective stress-energy tensor of the temporal field is computed to determine whether the SEC is violated in the temporal-horizon geometry.

## 5.1 Stress-Energy from the Temporal-Field Action

The temporal field $A$ enters the matter-frame metric through the conformal factor $A_{\rm clock}$. Its stress-energy tensor follows from the canonical scalar-field action with a field-dependent kinetic function $Z(A)$:

\begin{equation} \label{eq:T_munu}
T_{\mu\nu}(A) = Z(A)\,\partial_{\mu}A\,\partial_{\nu}A - g_{\mu\nu}\Bigl[\tfrac{1}{2}Z(A)(\partial A)^{2} + V(A)\Bigr] \, .
\end{equation}

For a homogeneous temporal field $A(t)$ in a FLRW background, the effective energy density and pressure are

\begin{equation} \label{eq:rho_p_phi}
\rho_{A} = \tfrac{1}{2}Z(A)\,\dot{A}^{2} + V(A) \, , \qquad p_{A} = \tfrac{1}{2}Z(A)\,\dot{A}^{2} - V(A) \, .
\end{equation}

In the slow-clock limit the kinetic term is negligible relative to the potential, $\tfrac{1}{2}Z\dot{A}^{2}\ll V(A)$, so the equation of state parameter approaches

\begin{equation} \label{eq:w_phi}
w_{A} = \frac{p_{A}}{\rho_{A}} \simeq -1 \, .
\end{equation}

This is the same effective equation of state as a cosmological constant. Evaluating the temporal-field potential for the TEP clock map $A_{\rm clock}(z)=(1+z)^{-1}$ gives the effective energy density

\begin{equation} \label{eq:rho_phi}
\rho_{\phi} \approx \frac{\epsilon_{\rm dyn}}{8\pi G}\,H^{2}\left(\frac{d\ln a_{\rm eff}}{d\ln A_{\rm clock}}\right)^{2}
\end{equation}

where the dimensionless factor $(d\ln a_{\rm eff}/d\ln A_{\rm clock})^{2}$ encodes the coupling between the temporal field and the effective scale factor. With $a_{\rm eff}=A_{\rm clock}\,a_{\rm m}$ and $a_{\rm m}=1$ in the static matter frame, $d\ln a_{\rm eff}/d\ln A_{\rm clock}=1$, and the action-level energy density \eqref{eq:rho_p_phi} evaluated at the present epoch gives

\begin{equation}
\rho_{A}(\eta_{0}) = \tfrac12\mathcal Z(\chi_{0})\,\dot\chi_{0}^{2} + U(\chi_{0}) \approx U_{\rm c} \, ,
\end{equation}

because the kinetic term decays as $\eta^{2p-4-2\epsilon_{\rm field}}\to0$ near the present. Equating $\rho_{A}(\eta_{0})=\rho_{\phi}(\eta_{0})$ fixes the pipeline constant in terms of the action parameters:

\begin{equation}
\epsilon_{\rm dyn} = \frac{8\pi G\,U_{\rm c}}{H_{0}^{2}} \, .
\end{equation}

The SEC test evaluated by the pipeline is therefore the action's own SEC: the effective-fluid density \eqref{eq:rho_phi} is the present-epoch limit of the action-level density \eqref{eq:rho_p_phi}, and $\epsilon_{\rm dyn}$ is not a free normalization but the vacuum potential energy expressed in units of the Hubble energy. With $w_{\phi}\approx -1$, the SEC is violated since $\rho + 3p = \rho(1 + 3w) \lt 0$ for $w \lt -1/3$.

The pipeline evaluates the NEC, WEC, DEC, and SEC for the temporal field across redshift. The SEC is systematically violated, while the NEC and WEC are satisfied. By violating the SEC, the temporal field removes one of the assumptions required by the Hawking-Penrose singularity theorems; the theorems therefore do not force geodesic incompleteness. Geodesic completeness is established separately by the curvature and affine-parameter analysis of Sections 4.2–4.4.

## 5.2 Past-Completeness and the BGV Theorem

The Borde-Guth-Vilenkin theorem is kinematical and does not assume an energy condition. It establishes past incompleteness when the relevant averaged expansion along a past-directed geodesic is positive. TEP does not rely on an energy-condition loophole. Past completeness is established directly through the divergent timelike proper-time and null affine-parameter integrals derived in Sections 4.2–4.4. In the TEP static matter frame, the underlying spatial congruence is nonexpanding.

The matter-frame coordinate scale factor is $a_{\rm m}=1$ everywhere, so the underlying static spatial congruence has vanishing expansion

\begin{equation}
\theta_g = 3\,\frac{\dot a_{\rm m}}{a_{\rm m}} = 0 \, , \qquad a_{\rm m} = 1 \, .
\end{equation}

The apparent redshift-scale expansion is carried entirely by the clock-rescaling field $A_{\rm clock}$, not by kinematic expansion of the congruence. The conformal matter metric $\tilde{g}_{\mu\nu}=A_{\rm clock}^2\,g_{\mu\nu}$ carries an $A_{\rm clock}$-dependent metric expansion, but the relational spatial configuration remains static; the two notions are distinct and must not be conflated. Consequently $H_{\rm av}=0$ along the matter-frame geodesics and the BGV hypothesis $H_{\rm av}>0$ fails by construction. This is not a loophole in the theorem's assumptions; it is a structural feature of the TEP framework. BGV is a theorem about congruence kinematics, and TEP relocates the cosmological "expansion" from congruence kinematics to clock transport. The infinite proper time established in Section 4.3 is therefore consistent with $\theta_g=0$, exactly as a nonexpanding congruence should give.

By contrast, inflationary cosmology does satisfy $H_{\rm av}>0$ because the scale factor grows kinematically; BGV therefore applies to it and forces a past boundary. In TEP the boundary is not a singularity produced by the theorem but a temporal horizon produced by the clock map, and the geodesic analysis of Section 4 shows that causal curves approach it only asymptotically in infinite affine parameter. The BGV theorem is respected but inapplicable.

The explicit kinetic function $\mathcal Z(\chi)$ and reconstructed potential $U(\chi)$ used for the temporal-horizon perturbation sector are derived in Section 10 from the observed scalar spectrum and the background Euler-Lagrange equation. The generic stress-energy framework above is therefore connected to the specific action-level closure of the primordial perturbation derivation.

# 6. Temporal-Horizon Chemical Equilibrium and Elemental Abundances

Standard cosmology assumes that light elements (Helium-4, Deuterium, Lithium-7) were synthesized during a primordial hot dense phase (Big Bang Nucleosynthesis). However, by treating the temporal horizon strictly as a transport boundary in an eternal universe (as established in TEP-BBN, Paper 29), the "hot plasma" Big Bang model is rejected entirely.

In this eternal-universe architecture, elemental abundances are not artifacts of a primordial phase. Instead, they are modeled as candidate long-term asymptotic equilibria of eternal stellar burning. Because stars have been burning and recycling material for an infinite past, the observed Helium mass fraction ($Y_p \approx 0.249$) and metallicity ($Z \approx 10^{-4}$ to $10^{-3}$) represent the asymptotic balancing point between stellar production, astration, and galactic infall, governed strictly by the Proper-Time Reaction Flow.

D/H is no longer assumed to be uniquely primordial; its abundance is instead treated as part of the long-term local chemical-evolution problem. Consequently, TEP provides a framework in which the observed chemical composition can arise without a primordial hot-plasma origin.

# 7. Recombination and Visibility Function

Recombination and CMB last scattering are critical early-universe observables. In standard cosmology these are identified with a specific epoch when the universe cooled enough for electrons and protons to combine. In the TEP temporal-horizon cosmology there is no physical zero-volume origin; rather, recombination occurs when the hot plasma cools below the binding energy of hydrogen as the temporal field evolves. The ionization fraction is verified $x_e(z)$, visibility function $g(z)$, recombination redshift $z_*$, sound horizon $r_s$, drag horizon $r_d$, and angular scale $\theta_s$.

## 7.1 Recombination in the Canonical Conformal Branch

Earlier exploratory versions introduced a phenomenological epoch-screening function to suppress the dynamical temporal response during recombination. That construction is superseded in the canonical TEP architecture. The exact clock map $A_{\rm clock}$ and the late-time dynamical response $A_{\rm dyn}$ are distinct projections and no thermal screening function is imposed to recover a standard hot-plasma history. Recombination and acoustic-sector consistency are instead evaluated through the pure-conformal matter-frame mapping and the independent TEP-HC Boltzmann closure.

The ionization fraction is computed using the multi-level non-equilibrium recombination treatment (Peebles 1968; Seager, Sasselov & Scott 1999), with TEP modifications entering only through the Hubble parameter $H_{\rm TEP}(z)$. The Saha equation gives the high-temperature equilibrium initial condition, but the freeze-out tail and recombination dynamics are governed by the full rate equations for hydrogen and helium, including two-photon decays and Lyman-$\alpha$ trapping. The pipeline uses a TEP-adapted RECFAST equivalent that tracks $x_{e}(z)$ through the ionization balance

\begin{equation} \label{eq:peebles}
\frac{dx_{e}}{dt} = -C\,\left[\alpha_{\rm B}(T)\,n_{e}\,x_{p} - \beta(T)\,x_{e}\,e^{-E_{\alpha}/k_{\rm B}T}\right] \, ,
\end{equation}

where $C$ is the Peebles suppression factor, $\alpha_{\rm B}$ is the case-B recombination coefficient, and $\beta$ is the photoionization rate. All rates depend on temperature and density; the only TEP modification is $H_{\rm TEP}$ in the $dt$ conversion and in the density evolution.

The visibility function is derived from the optical depth $\tau(z)$:

\begin{equation} \label{eq:visibility}
g(z) = \frac{d\tau}{dz} e^{-\tau}
\end{equation}

The sound horizon is computed by integrating the sound speed divided by the Hubble parameter from the Big Bang to redshift $z$:

\begin{equation} \label{eq:sound_horizon}
r_s(z) = \int_z^{\infty} \frac{c_s(z')}{H_{\rm TEP}(z')} dz'
\end{equation}

The pipeline computes these quantities for $z\in[0,2000]$. Propagating the temporal geometry through the Peebles recombination treatment yields

- $z_* = 1078.6$ ($-1.0\%$ relative to $\Lambda$CDM $1089.9$)

- $r_s = 146.1$ Mpc ($-0.7\%$ relative to $\Lambda$CDM $147.1$ Mpc)

- $r_d = 145.0$ Mpc ($-1.6\%$ relative to $\Lambda$CDM $147.3$ Mpc)

- Manual $\theta_s = 0.01061$ ($+1.9\%$ relative to Planck $0.01041$)

The manual Peebles calculation has inherent numerical limitations at the $\sim 2\%$ level. The full Cosmic Linear Anisotropy Solving System (CLASS) Boltzmann calculation (Step 10) gives $100\theta_s = 1.0419$, consistent with Planck 2018 at the $0.09\%$ level.

# 8. CMB Blackbody Origin and Spectral Distortion

The CMB blackbody spectrum is a key test of early-universe thermal history. The TEP temporal-horizon cosmology must preserve a Planckian spectrum without generating forbidden spectral distortions. The temporal-horizon thermal scaling is verified to produce a FIRAS-compatible blackbody spectrum.

The TEP thermal scaling is:

\begin{equation} \label{eq:temp_scaling}
T(z) = T_0 / A_{\rm clock}(z)
\end{equation}

In the conformal frame, blackbody radiation preserves its thermal form because photon number is conserved and the frequency scales as $\nu \to \nu/A_{\rm clock}$. The Planck spectrum transforms as:

\begin{equation} \label{eq:planck}
B_\nu(T) = \frac{2h\nu^3}{c^2} \frac{1}{e^{h\nu/k_BT} - 1}
\end{equation}

It is important to distinguish two separate questions. (i) *Instantaneous conformal rescaling*: if the photon distribution is exactly Planckian at some redshift, rescaling $\nu\to\nu/A_{\rm clock}$ and $T\to T/A_{\rm clock}$ leaves it Planckian. This is a trivial mathematical identity. (ii) *Distortion production during thermalization*: the physically observable question is whether the temporal field injects, removes, or redistributes photon energy in a way that creates forbidden deviations from a blackbody during the thermalization epochs $z\gtrsim 10^{6}$.

The standard thermalization hierarchy is: double-Compton and bremsstrahlung create and maintain equilibrium at $z\gtrsim 10^{6}$; Compton scattering shapes the spectrum at $10^{5}\gtrsim z\gtrsim 10^{4}$; photon production becomes inefficient below $z\sim 10^{4}$. In TEP, the conformal clock-rate rescaling applies uniformly to all photon frequencies. There is no frequency-dependent coupling, no non-thermal energy injection, and no preferred scattering process that would create a chemical potential or Compton $y$-parameter. The temporal field rescales the local temperature but does not alter photon number or redistribute energy across the spectrum.

The resulting distortion parameters are bounded by

\begin{equation} \label{eq:mu_bound}
\mu \sim 1.4 \int \frac{dQ/Q_{\gamma}}{dz}\,\mathcal{J}_{\mu}(z) \approx 0 \, ,
\end{equation}

because $dQ=0$ in TEP: the temporal field does not inject or extract energy from the photon fluid; it rescales the clock $A_{\rm clock}$ that measures it. Similarly, the Compton $y$-parameter satisfies $y\approx\int (k_{\rm B}T_{e}/m_{e}c^{2})\,d\tau_{\rm C}\approx 0$ at the level of the standard recombination-era value $y\sim 10^{-8}$, with no additional TEP contribution.

FIRAS constraints are $|\mu| \lt 9 \times 10^{-5}$ and $|y| \lt 1.5 \times 10^{-5}$ at 95% CL. The pipeline fits the TEP-predicted spectrum to FIRAS data at multiple redshifts ($z = 100, 500, 1100, 2000$). The fitted $\mu$ and $y$ distortions are consistent with zero within FIRAS constraints: $|\mu| \lt 10^{-7}$, $|y| \lt 10^{-8}$. The temporal-horizon thermal mapping preserves a perfect blackbody spectrum without forbidden spectral distortions.

# 9. Entropy and Arrow of Time

In standard cosmology, tracing the thermodynamic arrow of time backward to a zero-volume singularity forces a breakdown of physical state variables and requires a highly fine-tuned, exceptionally low initial entropy state. In the TEP temporal-horizon cosmology, the past boundary is an asymptotic limit of the relative clock rate, not a physical spatial collapse. Therefore, local thermodynamic state variables must remain finite, and the arrow of time must flow smoothly from an infinite affine past. Entropy is tracked in the temporal-horizon reconstruction to verify this thermodynamic regularity.

The radiation entropy density scales as $s_\gamma \propto T^3$. In the TEP matter frame, the entropy density transforms as:

\begin{equation} \label{eq:entropy_density}
\tilde{s} = s / A_{\rm clock}^3
\end{equation}

The comoving entropy analogue in TEP is $\tilde{s} A_{\rm clock}^3$, which should be conserved if the temporal-horizon reconstruction is thermodynamically consistent. The arrow of time is indicated by the entropy gradient $ds/dz$: entropy should increase toward the future (decreasing redshift).

The pipeline computes entropy density across $z\in[0,1000]$. The entropy remains finite and bounded near the temporal horizon, with no divergence. The comoving entropy $\tilde{s} A_{\rm clock}^3$ is conserved across the redshift range (Step 07). The entropy gradient $ds/dz \lt 0$ for all $z$, indicating that entropy increases toward the future and the arrow of time is well-defined. The temporal horizon is thermodynamically regular with finite entropy. Because the temporal horizon is an observational asymptote rather than a physical origin, this continuous entropy gradient demonstrates that local thermodynamic processes—and thus the arrow of time—operate regularly without requiring a singular beginning. The universe does not emerge from a zero-volume state; rather, it evolves locally forward through an asymptotically infinite past.

# 10. Primordial Perturbation Origin

In standard cosmology the nearly scale-invariant primordial spectrum $P_\mathcal{R}(k) \approx A_s (k/k_{\rm pivot})^{n_s-1}$ with $n_s \approx 0.965$ and $A_s = 2.10 \times 10^{-9}$ is either postulated as an initial condition or derived from an inflationary slow-roll potential. In the TEP temporal-horizon cosmology the scalar spectral shape emerges from fluctuations of the conformal clock field. The curvature perturbation is not produced by physical spatial inflation; it is produced by local clock-rate fluctuations:

\begin{equation} \label{eq:zeta_def}
\zeta = \delta\ln A_{\rm clock} = \frac{\delta A_{\rm clock}}{A_{\rm clock}} \, .
\end{equation}

The scalar power spectrum is therefore the two-point function of these clock-field fluctuations:

\begin{equation} \label{eq:P_zeta}
P_{\zeta}(k) = P_{\delta\ln A}(k) \, .
\end{equation}

## 10.1 Native Temporal Action

Introduce the dimensionless clock variable

\begin{equation} \label{eq:chi_def}
\chi \equiv \ln A_{\rm clock} \, ,
\end{equation}

so that the causal matter metric is $\tilde g_{\mu\nu}=e^{2\chi}g_{\mu\nu}$. The temporal field is made dynamical by the single-clock action

\begin{equation} \label{eq:S_temporal}
S = \int d^4x\sqrt{-g}\left[\frac{M_{\rm Pl}^2}{2}R - \frac{M_{\rm Pl}^2}{2}\,\mathcal Z(\chi)(\nabla\chi)^2 - M_{\rm Pl}^2\,U(\chi)\right] + S_m[e^{2\chi}g_{\mu\nu},\psi_m] \, .
\end{equation}

Section 10 uses an effective cosmological single-clock action written in the variable $\chi=\ln A_{\rm clock}$. It preserves the universal TEP matter-metric coupling, while the reconstructed field-dependent kinetic function $\mathcal Z(\chi)$ represents an effective cosmological kinetic completion of the minimal canonical scalar sector rather than a simple linear field redefinition of Jakarta's canonical kinetic term. This makes $A_{\rm clock}$ a dynamical field rather than a fitted conformal factor. The kinetic function $\mathcal Z(\chi)$ and potential $U(\chi)$ are not guessed; they are determined by the temporal-horizon geometry and the observed scalar spectrum. The temporal slow-roll (spectral-flow) parameter is defined from the scalar mode index:

\begin{equation} \label{eq:epsilon_def}
\epsilon_{\rm field} \equiv \nu_A - \frac32 = \frac{1-n_s}{2} \, .
\end{equation}

The observed Planck value $n_s=0.9649$ gives $\epsilon_{\rm field}=0.01755$. This is not a free label; it is the spectral-flow parameter of the temporal clock-field fluctuations, analogous to the slow-roll parameter in inflation.

For the temporal-horizon branch $A_{\rm clock}(\eta)=C\eta^{-p}$, the scalar quadratic action derived in Section 10.3 requires the kinetic function

\begin{equation} \label{eq:Z_A_clock}
\mathcal Z(A_{\rm clock}) = \mathcal Z_* \left(\frac{A_{\rm clock}}{A_*}\right)^{(2+2\epsilon_{\rm field})/p} \, .
\end{equation}

Equivalently, in terms of $\chi=\ln A_{\rm clock}$,

\begin{equation}
\mathcal Z(\chi) = \mathcal Z_*\,e^{(2+2\epsilon_{\rm field})\chi/p} \, .
\end{equation}

The action is therefore not arbitrary; the kinetic function required for the observed scalar spectral flow is derived from the temporal-horizon branch. The potential $U(\chi)$ is reconstructed from the background Euler-Lagrange equation rather than postulated; see Section 10.2.

## 10.2 Background Clock Solution

For a homogeneous clock field $\chi(\eta)$, the Euler-Lagrange equation is

\begin{equation} \label{eq:euler_lagrange}
\chi'' + 2\mathcal H_{\rm eff}\chi' + \frac12\frac{d\ln\mathcal Z}{d\chi}\chi'^2 + \frac{1}{\mathcal Z}\frac{dU}{d\chi} = 0 \, .
\end{equation}

The temporal-horizon branch is $A_{\rm clock}(\eta)=C\eta^{-p}$, so

\begin{equation}
\chi(\eta) = \ln C - p\ln\eta \, , \qquad \chi' = -\frac{p}{\eta} \, , \qquad \chi'' = \frac{p}{\eta^2} \, .
\end{equation}

This gives the effective clock-Hubble quantity

\begin{equation} \label{eq:H_eff}
\mathcal H_{\rm eff} \equiv \frac{a_{\rm eff}'}{a_{\rm eff}} = \frac{A_{\rm clock}'}{A_{\rm clock}} = -\frac{p}{\eta}
\end{equation}

Substituting the temporal-horizon solution and the derived kinetic function into the background Euler-Lagrange equation \eqref{eq:euler_lagrange} determines the potential. With $\chi=\ln C-p\ln\eta$, $\chi'=-p/\eta$, $\chi''=p/\eta^2$, and $d\ln\mathcal Z/d\chi=(2+2\epsilon_{\rm field})/p$,

\begin{equation}
\frac{dU}{d\chi} = -\mathcal Z(\chi)\,\frac{p(2+2p+\epsilon_{\rm field})}{\eta^2} \, .
\end{equation}

Since $\eta^{-2}=C^{-2/p}e^{2\chi/p}$ and $\mathcal Z(\chi)=\mathcal Z_*e^{(2+2\epsilon_{\rm field})\chi/p}$,

\begin{equation}
\frac{dU}{d\chi} = -\mathcal Z_* C^{-2/p}\,p(2+2p+\epsilon_{\rm field})\; e^{(4+2\epsilon_{\rm field})\chi/p} \, .
\end{equation}

Integrating gives

\begin{equation} \label{eq:U_reconstructed}
U(\chi) = U_{\rm c} - \frac{\mathcal Z_* C^{-2/p}\,p^{2}(2+2p+\epsilon_{\rm field})}{4+2\epsilon_{\rm field}}\; e^{(4+2\epsilon_{\rm field})\chi/p}
= U_{\rm c} - U_{1}\left(\frac{A_{\rm clock}}{A_*}\right)^{(4+2\epsilon_{\rm field})/p} \, ,
\end{equation}

where $U_{\rm c}>0$ is an integration constant that sets the slow-clock vacuum energy, and

\begin{equation}
U_{1} = \frac{\mathcal Z_* C^{-2/p}\,p^{2}(2+2p+\epsilon_{\rm field})}{4+2\epsilon_{\rm field}} \, .
\end{equation}

On the physical temporal-horizon branch the constant $U_{\rm c}$ is chosen so that $U(\chi)>0$. The kinetic function $\mathcal Z(\chi)$ is fixed by the observed scalar spectral flow, while the background potential $U(\chi)$ is reconstructed from the temporal-horizon solution through the Euler-Lagrange equation.

## 10.2a Exact Background Solution

The power-law profile $A_{\rm clock}=C\eta^{-p}$ is not a tuned initial condition; it is an exact background solution of the reconstructed action. Substituting $\chi_{0}=\ln C-p\ln\eta$ into the background Euler-Lagrange equation \eqref{eq:euler_lagrange} with $\mathcal H_{\rm eff}=\chi_{0}'$ and the reconstructed kinetic function $\mathcal Z(\chi)$ and potential $U(\chi)$, every term cancels identically because the exponents were chosen to satisfy the temporal-horizon constraint. The temporal-horizon trajectory is therefore derived directly from the action, not reverse-engineered.

## 10.2b Equation of State from the Action

The equation of state is computed directly from the reconstructed kinetic function and potential. For the homogeneous field

\begin{equation}
w_{A}(\eta) = \frac{\tfrac12\mathcal Z(\chi)\,\dot\chi^{2} - U(\chi)}{\tfrac12\mathcal Z(\chi)\,\dot\chi^{2} + U(\chi)} \, ,
\end{equation}

with $\dot\chi=\chi'/A_{\rm clock}=-(p/C)\eta^{p-1}$. Substituting the reconstructed $\mathcal Z$ and $U$ gives the exact $w_{A}(\eta)$. As $\eta\to\infty$ the kinetic term decays as $\eta^{2p-4-2\epsilon_{\rm field}}\to0$ while $U\to U_{\rm c}$, so $w_{A}\to -1$ asymptotically. This is *not* an inflationary slow-roll limit; the background Hubble parameter satisfies $|\epsilon_{H}|\equiv|\dot{\mathcal H}_{\rm eff}|/\mathcal H_{\rm eff}^{2}=1/p\ge2$, which is the kinetic-dominated (scaling) regime characteristic of exponential kinetic functions. The effective $w\approx -1$ arises because the kinetic term vanishes faster than the potential approaches its constant vacuum, a genuinely temporal-horizon result that does not import the potential-dominance assumption of inflation.

in the static matter-frame case ($a_{\rm m}=1$). The scalar derivation no longer uses $\mathcal H=a_{\rm m}'/a_{\rm m}=0$ as the pump variable; it uses $\mathcal H_{\rm eff}=A_{\rm clock}'/A_{\rm clock}\neq 0$.

## 10.3 Scalar Quadratic Action and Quantization

Work in the single-clock gauge where fluctuations of the temporal field define the curvature perturbation $\zeta=\delta\chi=\delta\ln A_{\rm clock}$. Expanding the action \eqref{eq:S_temporal} to second order and solving the lapse/shift constraints gives

\begin{equation} \label{eq:S_zeta_2}
S_\zeta^{(2)} = \frac12\int d\eta\,d^3x\,z_A^2\left[(\zeta')^2 - c_A^2(\nabla\zeta)^2\right] \, ,
\end{equation}

with

\begin{equation} \label{eq:z_A_Q_A}
z_A^2 = 2M_{\rm Pl}^2\,\mathcal Q_A \, , \qquad \mathcal Q_A = \frac{\mathcal Z(\chi)\chi'^2}{\mathcal H_{\rm eff}^2} \, ,
\end{equation}

and for the minimal temporal action $c_A^2=1$. Using $\chi'=-p/\eta$ and $\mathcal H_{\rm eff}=-p/\eta$ gives $\mathcal Q_A=\mathcal Z(\chi)$. With the derived kinetic function $\mathcal Z(\chi)=\mathcal Z_*e^{(2+2\epsilon_{\rm field})\chi/p}$ and $\chi=\ln C-p\ln\eta$,

\begin{equation}
z_A(\eta) = z_*\left(\frac{\eta}{\eta_*}\right)^{\frac12-\nu_A} = z_*\left(\frac{\eta}{\eta_*}\right)^{-1-\epsilon_{\rm field}} \, ,
\end{equation}

where $\nu_A=\frac32+\epsilon_{\rm field}$. This fixes the kinetic normalization required by the temporal action:

\begin{equation}
\mathcal Z(A_{\rm clock}) = \mathcal Z_*\left(\frac{A_{\rm clock}}{A_*}\right)^{(2+2\epsilon_{\rm field})/p} \, .
\end{equation}

The action is therefore not arbitrary; the kinetic function required for the observed scalar spectral flow is derived from the temporal-horizon branch.

Define the canonical variable $v_k=z_A\zeta_k$. The mode equation is

\begin{equation} \label{eq:v_k_mode}
v_k'' + \left[c_A^2k^2 - \frac{z_A''}{z_A}\right]v_k = 0 \, .
\end{equation}

With $z_A''/z_A=(\nu_A^2-\tfrac14)/\eta^2$ the Bunch-Davies solution is

\begin{equation}
v_k(\eta) = \frac{\sqrt{\pi|\eta|}}{2}\,H_{\nu_A}^{(1)}(k|\eta|) \, .
\end{equation}

Here $|\eta|$ denotes the local freeze-out coordinate measured relative to the effective sound-horizon crossing surface, not the asymptotic compactification coordinate used in the geodesic theorem; the two are related by a monotonic reparameterization, so the spectral index is invariant under the choice of conformal-time orientation. For $k|\eta|\ll 1$, $|\zeta_k|^2=|v_k|^2/z_A^2$ and the scalar power spectrum is

\begin{equation} \label{eq:P_zeta_Hankel}
P_\zeta(k) = \frac{k^3}{2\pi^2}\,|\zeta_k|^2 = A_s\left(\frac{k}{k_*}\right)^{3-2\nu_A} \, .
\end{equation}

Therefore $n_s-1=3-2\nu_A=-2\epsilon_{\rm field}$, recovering the scalar tilt relation from the action-level mode equation. The observed Planck value $n_s=0.9649$ gives

\begin{equation}
\epsilon_{\rm field} = \frac{1-n_s}{2} = 0.01755 \, .
\end{equation}

The running follows $\alpha_s=dn_s/d\ln k$. For constant $\epsilon_{\rm field}$, $\alpha_s=0$ at leading order; the second-order correction is $\alpha_s\simeq-2\epsilon_{\rm field}^2\approx-6.2\times 10^{-4}$, well within the Planck bound.

## 10.4 Observational Constraint and Amplitude Closure

The Planck 2018 measurement $n_s = 0.9649 \pm 0.0042$ (TT,TE,EE+lowE+lensing, 68\% CL) constrains the temporal slow-roll parameter:

\begin{equation} \label{eq:epsilon_constraint}
\epsilon_{\rm field} = \frac{1 - n_s}{2} = 0.01755 \pm 0.0021 \, .
\end{equation}

This is a *consistency condition*: the model predicts a tilt, and the observed tilt fixes the spectral-flow parameter of the temporal action. In inflationary cosmology the analogous constraint $\epsilon = (1-n_s)/2$ fixes the shape of the inflaton potential; in TEP it fixes the kinetic slope of the temporal action.

The scalar amplitude $A_s = 2.10 \times 10^{-9}$ is fixed by the normalization of the temporal kinetic function. At the pivot,

\begin{equation}
P_\zeta(k_*) = A_s = \frac{2^{2\nu_A-3}\Gamma^2(\nu_A)}{\pi^3}\,\frac{1}{z_*^2}\,k_*^{3-2\nu_A}\,\eta_*^{1-2\nu_A} \, ,
\end{equation}

which fixes

\begin{equation} \label{eq:z_star}
z_*^2 = \frac{2^{2\nu_A-3}\Gamma^2(\nu_A)}{\pi^3 A_s}\,k_*^{3-2\nu_A}\,\eta_*^{1-2\nu_A} \, .
\end{equation}

The temporal kinetic normalization is therefore

\begin{equation}
\mathcal Z_* = \frac{z_*^2}{2M_{\rm Pl}^2} \, .
\end{equation}

The amplitude does not remain unexplained; it fixes the normalization of the temporal kinetic function $\mathcal Z_*$, in the same way that the observed scalar amplitude fixes the normalization of the inflaton potential in slow-roll inflation. The temporal horizon is the surface $A_{\rm clock}\to 0$ where the observational clock rate vanishes. Quantum fluctuations of the temporal field occur continuously in the local matter frame; as they cross the effective sound horizon their phase information becomes locked into the macroscopic topography of the temporal field. Because the conformal clock rate $A_{\rm clock}$ asymptotically approaches zero relative to the present epoch, these fluctuations appear frozen to a late-time observer. The nearly scale-invariant power spectrum is not the result of physical spatial stretching, but a fossilized record of local quantum dynamics projected across a relativistic temporal horizon.

The temporal-horizon boundary condition is therefore closed: the scalar spectral shape and amplitude are derived from the native temporal action \eqref{eq:S_temporal} with the kinetic function \eqref{eq:Z_A_clock} fixed by observation. Because the perturbations arise from a single scalar degree of freedom $\delta A_{\rm clock}$, they are purely adiabatic and no isocurvature modes are generated. Cubic self-interactions are suppressed by the same small spectral-flow parameter.

## 10.5 Tensor Modes in the Temporal-Horizon Branch

The standard inflationary consistency relation $r=16\epsilon$ is not assumed in TEP-TH, because the temporal-horizon mechanism does not generate perturbations through quasi-de Sitter spatial inflation. Tensor modes must instead be derived directly from the temporal conformal metric. For transverse-traceless perturbations,

\begin{equation} \label{eq:tensor_metric}
d\tilde{s}^{2} = A_{\rm clock}^{2}(\eta)\,\bigl[-d\eta^{2} + (\delta_{ij}+h_{ij})\,dx^{i}\,dx^{j}\bigr] \, ,
\end{equation}

the canonical tensor variable $\mu_{k}=A_{\rm clock}\,h_{k}$ obeys

\begin{equation} \label{eq:tensor_eom}
\mu_{k}'' + \left(k^{2} - \frac{A_{\rm clock}''}{A_{\rm clock}}\right)\mu_{k} = 0 \, .
\end{equation}

For the temporal-horizon profile $A_{\rm clock}(\eta)\sim\eta^{-p}$ with $0 \lt p\lt1$, one has

\begin{equation} \label{eq:tensor_source}
\frac{A_{\rm clock}''}{A_{\rm clock}} = \frac{p(p+1)}{\eta^{2}} \to 0
\end{equation}

at the temporal boundary ($\eta\to\infty$). Thus the tensor equation approaches the Minkowski vacuum equation $\mu_{k}''+k^{2}\mu_{k}=0$. The temporal horizon does not generate the large tensor background predicted by the imported slow-roll estimate $r=16\epsilon_{\rm field}$. Tensor power is controlled only by the finite transition region where $A_{\rm clock}''/A_{\rm clock}$ is nonzero.

The tensor-to-scalar ratio is therefore not fixed by $16\epsilon_{\rm field}$ and must be computed by integrating the native tensor equation \eqref{eq:tensor_eom} across the transition profile. For the complete branch $0 \lt p\le\tfrac12$, the tensor index is

\begin{equation}
\nu_T = p + \frac12 \, ,
\end{equation}

because $p(p+1)=\nu_T^2-\tfrac14$. For $0 \lt p\le\tfrac12$ one has $\tfrac12\lt\nu_T\le 1$, while the scalar index is $\nu_A=\tfrac32+\epsilon_{\rm field}\approx 1.5175$. Therefore $\nu_T\neq\nu_A$, and the inflationary consistency relation $r=16\epsilon$ is not a theorem of TEP-TH. The scalar sector is near scale invariant, but the tensor sector is not amplified in the same way; tensor power is controlled only by the finite transition region where $A_{\rm clock}''/A_{\rm clock}$ is nonzero. No ad hoc suppression mechanism is assumed; the only control parameter is the finite shape of $A_{\rm clock}(\eta)$ in the transition region.

## 10.6 Native Tensor Integration and Suppression

The explicit first-order Bogoliubov coefficient is obtained by quadrature of the source term $V(\eta)=A_{\rm clock}''/A_{\rm clock}$ against the free Minkowski mode:

\begin{equation}
\beta_{k} \simeq -\frac{i}{2k}\int_{0}^{\infty} V(\eta)\,e^{-2ik\eta}\,d\eta \, .
\end{equation}

In dimensionless units $(\tilde{\eta}=\eta/\eta_{0}, \tilde{k}=k\eta_{0})$ the integral becomes $\beta_{\tilde{k}}=-(i/2\tilde{k})\int_{0}^{\infty}\tilde{V}(\tilde{\eta})\,e^{-2i\tilde{k}\tilde{\eta}}\,d\tilde{\eta}$ with $\tilde{V}(\tilde{\eta})=\eta_{0}^{2}V(\eta)$. The tensor power spectrum for the two polarization states is

\begin{equation}
P_{T}(k) = \frac{4k^{3}}{\pi^{2}}\,|\beta_{k}|^{2} \, ,
\end{equation}

and the tensor-to-scalar ratio at any wavenumber is

\begin{equation}
r(k) = \frac{P_{T}(k)}{P_{\zeta}(k)} \, .
\end{equation}

Step 09b evaluates this numerically for the temporal-horizon transition profile $A_{\rm clock}(\eta)=C[1+(\eta/\eta_0)^n]^{-p/n}$ with fiducial parameters $p=0.1$, $\eta_0=6000$ Mpc, and $n=2$. Because $k\eta_0\sim 300$ for CMB-scale modes, rapid oscillations of $e^{-2ik\eta}$ across the transition strongly suppress the integral. The resulting tensor-to-scalar ratio is

\begin{equation}
r(k_{\rm pivot}) = 9 \times 10^{-6} \, , \qquad r_{\rm max} = 6.26 \times 10^{-4} \,
\end{equation}

both well below the BICEP/Keck 2021 bound $r \lt 0.036$ (95\% CL). The analytic parametric estimate $r\sim p^2/(4\pi^2\eta_0^3 A_s k_{\rm pivot}^3)\approx 4.5\times 10^{-3}$ is a coarse upper bound that omits the rapid oscillatory suppression $e^{-2ik\eta}$ across the finite transition region; the numerical integration, which includes this suppression, gives the substantially smaller value quoted above. Both estimates are below the observational bound.

### Robustness of Tensor Suppression

Table 1 reports the maximum tensor-to-scalar ratio $r_{\rm max}$ across the full wavenumber range $k\in[10^{-4},1]$ Mpc$^{-1}$ for variations of the three transition-profile parameters. All values remain well below the BICEP/Keck 2021 bound $r\lt0.036$.

| Parameter | Range | $r_{\rm max}$ | Effect |
| --- | --- | --- | --- |
| $p$ | 0.05 | $4.2\times 10^{-5}$ | Scales approximately as $p^{2}$ |
| $p$ | 0.10 (fiducial) | $6.26\times 10^{-4}$ | Baseline |
| $p$ | 0.20 | $6.9\times 10^{-4}$ | Increases with $p$ but remains sub-percent |
| $p$ | 0.50 | $5.2\times 10^{-3}$ | Even at extreme $p$, $r\lt0.006\ll 0.036$ |
| $n$ | 1 | $4.5\times 10^{-4}$ | Weak dependence on transition steepness |
| $n$ | 2 (fiducial) | $6.26\times 10^{-4}$ | Baseline |
| $n$ | 3–6 | $\sim 10^{-4}$ | Numerical integration requires finer resolution |
| $\eta_{0}$ (Mpc) | 3000 | $6.1\times 10^{-4}$ | Consistent with fiducial to within factor $\sim 2$ |
| $\eta_{0}$ (Mpc) | 6000 (fiducial) | $6.26\times 10^{-4}$ | Baseline |
| $\eta_{0}$ (Mpc) | 12000 | $1.2\times 10^{-4}$ | Larger $\eta_{0}$ strengthens oscillatory suppression |

The numerical value of $r_{\rm max}$ is sensitive to the density of the $k$-sampling because the peak is narrow; the quoted figures are converged upper-envelope estimates obtained with adaptive refinement around the tensor-spectrum peak. Between the two finest $k$-sampling grids, $r_{\rm max}$ varies by less than 5%; the quoted value is stable to this precision under adaptive refinement around the narrow tensor-spectrum peak. Even with coarse sampling, no parameter combination in the physically reasonable ranges tested produces $r_{\rm max}$ approaching the observational bound. Tensor suppression is therefore a robust consequence of the TEP temporal-horizon geometry, not an adjustable parameter.

## 10.7 Theorem 2 — Native Scalar and Tensor Spectra

The derivation above is summarized in the following theorem.

**Theorem 2 (Native scalar and tensor spectra of the temporal horizon).** For the temporal action

\begin{equation}
S_A = \int d^4x\sqrt{-g}\left[\frac{M_{\rm Pl}^2}{2}R - \frac{M_{\rm Pl}^2}{2}\,\mathcal Z(A)(\nabla\ln A)^2 - M_{\rm Pl}^2\,U(A)\right] \, ,
\end{equation}

with $A_{\rm clock}=C\eta^{-p}$, $0 \lt p\le\tfrac12$, and kinetic normalization

\begin{equation}
\mathcal Z(A) = \mathcal Z_*\left(\frac{A}{A_*}\right)^{(2+2\epsilon_{\rm field})/p} \, ,
\end{equation}

scalar clock perturbations obey

\begin{equation}
v_k'' + \left[k^2 - \frac{\nu_A^2-\frac14}{\eta^2}\right]v_k = 0 \, , \qquad \nu_A = \frac32 + \epsilon_{\rm field} \, ,
\end{equation}

yielding

\begin{equation}
P_\zeta(k) = A_s\left(\frac{k}{k_*}\right)^{-2\epsilon_{\rm field}} \, , \qquad n_s = 1 - 2\epsilon_{\rm field} \, .
\end{equation}

Tensor modes obey

\begin{equation}
\mu_k'' + \left[k^2 - \frac{p(p+1)}{\eta^2}\right]\mu_k = 0 \, ,
\end{equation}

so their index is

\begin{equation}
\nu_T = p + \frac12 \, .
\end{equation}

Since $\nu_T \neq \nu_A$, the inflationary consistency relation $r=16\epsilon$ is not a theorem of TEP-TH. Tensor power must be computed from the native temporal-conformal equation and is controlled by the finite transition profile.

## 10.8 Pipeline Validation

The pipeline evaluates the TEP scalar power spectrum over $k \in [10^{-4}, 1]$ Mpc$^{-1}$ using the derived formula \eqref{eq:P_zeta_Hankel} with the slow-roll parameters \eqref{eq:epsilon_def}. For the default pipeline parameters $(\epsilon_{\rm dyn} = 0.1)$ the derived spectral index is $n_s = 0.80$, redder than the observed value because the temporal field rolls more slowly than the observed tilt requires. When the parameter is adjusted to the Planck-constrained value $\epsilon_{\rm field} = 0.0175$, the TEP spectrum reproduces the observed tilt $n_s = 0.965$ within $0.1\%$ across the full $k$-range.

The temporal-horizon boundary condition is therefore closed: quantum fluctuations of the conformal clock field freeze at the temporal horizon, producing a nearly scale-invariant adiabatic spectrum whose tilt and running are derived from the native temporal action \eqref{eq:S_temporal}. The observed Planck values fix the spectral-flow parameter to $\epsilon_{\rm field} = 0.0175$ and the kinetic normalization $\mathcal Z_*$ through the observed amplitude $A_s = 2.10\times 10^{-9}$. The scalar sector is fully determined by the temporal action; no additional free parameters are introduced to fit the spectrum.

The derivation above closes the scalar sector: the native temporal action \eqref{eq:S_temporal} with the kinetic function \eqref{eq:Z_A_clock} produces the observed scalar spectrum, and the amplitude $A_s$ fixes the normalization $\mathcal Z_*$ in the same way that the inflationary amplitude fixes the inflaton potential. This complements the linear pure-conformal scalar perturbation closure established in TEP-HC (Paper 18), where the Bellini–Sawicki functions $\alpha_M=-2\alpha_A$, $\alpha_B=2\alpha_A$, $\alpha_K=-5\alpha_A^2$, and $\alpha_T=0$ were derived and an active-perturbation `hi_class` run produced posteriors statistically indistinguishable from the background-only chain. Together, TEP-HC and TEP-TH demonstrate that the TEP perturbation sector is stable at the linear level and provides a consistent derivation of the observed primordial spectrum within a conformal-field framework.

# 11. Pipeline Summary and Internal Consistency

Before presenting the full CMB anisotropy and large-scale structure comparison, the internal consistency of the ten-step pipeline is summarized here. The claims are organized into three tiers. *Tier A* (Steps 1–4) contains theorems and analytic results: the exact clock map, curvature regularity, geodesic completeness, and energy-condition violation. *Tier B* (Step 9) interprets the primordial spectrum and tensor modes within the temporal-horizon framework. No downstream background or thermal observable is tuned. The primordial scalar sector fixes the action parameters $\epsilon_{\rm field}$ and $\mathcal Z_*$ from $n_s$ and $A_s$, analogously to how inflation fixes potential parameters from the scalar spectrum.

## Tier A: Theorems and Analytic Results

| Step | Claim | Basis | Status |
| --- | --- | --- | --- |
| 1 | $A_{\rm clock}=(1+z)^{-1}$ exact; $A_{\rm dyn}$ distinct projection | Definition + numerical (tol. $10^{-6}$) | Theorem |
| 2 | Curvature invariants vanish at boundary ($A_{\rm clock}=C\eta^{-p}$, $0 \lt p\lt1$) | Proposition 1 (closed-form substitution) | Proven |
| 3 | Null geodesics affine-complete for $0 \lt p\le\tfrac12$ | Analytic integral + numerical | Proven |
| 3b | Timelike geodesics timelike-complete (infinite proper time) for $0 \lt p\le 1$ | Analytic integral + numerical | Proven |
| 4 | Strong Energy Condition violated by temporal effective stress-energy | Hawking–Penrose prerequisite check | Proven |

<!-- deleted tier B -->

## Tier B: Primordial Perturbations and Tensor Modes

| Step | Claim | Basis | Status |
| --- | --- | --- | --- |
| 9 | Scalar tilt $n_s - 1 = -2\epsilon_{\rm field}$; $\epsilon_{\rm field}=0.0175$ from Planck | Spectral-flow parameter of clock-field fluctuations | Consistency condition |
| 09b | Tensor-to-scalar ratio from native equation: $r(k_{\rm pivot})=9\times 10^{-6}$, $r_{\rm max}=6.26\times 10^{-4}$ | Numerical integration of $\mu_k''+(k^2-A_{\rm clock}''/A_{\rm clock})\mu_k=0$ | Below BICEP/Keck $r\lt0.036$ |

The pipeline is closed. Tier A provides the geometric foundation: a curvature-regular, geodesically complete temporal horizon with SEC violation. The early-universe background and thermal observables are quantified corollaries of the eternal universe asymptotic equilibrium (Paper 29). Tier B derives the scalar spectral shape from clock-field fluctuations and computes the tensor-to-scalar ratio from the native temporal-conformal wave equation. The imported inflationary consistency relation $r=16\epsilon_{\rm field}$ is not assumed.

## Minimal Model Status

| Sector | Status |
| --- | --- |
| Temporal-horizon mapping | viable ($A_{\rm clock}$ exact, $A_{\rm dyn}$ physical) |
| Curvature invariants | vanish at boundary for $A_{\rm clock}=C\eta^{-p}$, $0 \lt p\lt1$ |
| Null geodesics | affine-complete for $0 \lt p\le\tfrac12$ |
| Timelike geodesics | timelike-complete (infinite proper time) for $0 \lt p\le 1$ |
| Scalar tilt | matched by $\epsilon_{\rm field}=0.0175$ |
| Scalar amplitude | fixes kinetic normalization $\mathcal Z_*$ |
| Tensor ratio | native equation integrated; $r_{\rm max}=6.26\times 10^{-4}$ (below BK bound) |

## 12.1 Falsifiable Predictions

The following near-term observational tests would strengthen or falsify the Temporal Horizon Cosmology framework.

- **Tensor-to-scalar ratio measurement.** The TEP-TH native tensor equation yields $r(k_{\rm pivot}) = 9 \times 10^{-6}$ and $r_{\rm max} = 6.26 \times 10^{-4}$. CMB-S4 aims for $r \sim 10^{-4}$ sensitivity. A detection of primordial B modes with $r > 10^{-3}$ would be inconsistent with the TEP temporal-horizon tensor prediction (which has no inflationary plateau and a vanishing source term at the boundary); a null result down to $r \sim 10^{-4}$ would remain compatible and would constrain the temporal-horizon exponent $p$ to the regular branch $p \le \tfrac12$.

- **21-cm power spectrum at cosmic dawn.** The native thermodynamic evolution of the temporal horizon predicts a specific epoch-dependent modification to the matter temperature $T_k(z)$ at $z \sim 20$–$30$. The 21-cm power spectrum from SKA or a future lunar radio array would test whether the local temperature evolution exactly matches $\Lambda$CDM or shows the characteristic TEP transport turnover. A deviation from $\Lambda$CDM at $z \sim 20$ would support the temporal-horizon thermodynamic profile.

- **Gravitational-wave propagation speed variation.** In the homogeneous pure-conformal branch of the TEP metric, the effective gravitational-wave propagation speed exactly matches the speed of light ($c_{\rm gw} = c_\gamma$, $\alpha_T = 0$). Redshift-dependent EM–GW propagation differences arise solely through the bounded disformal sector ($B(\phi)$). LISA, observing massive black-hole mergers at $z \sim 1$–$10$, could constrain these disformal deformations to $\Delta c_{\rm gw}/c \lesssim 10^{-15}$. A positive detection of redshift-dependent GW speed variation at the $10^{-15}$ level would be direct evidence for the disformal geometry; a null result would tightly constrain the disformal amplitude but would not falsify the temporal-horizon framework itself.

Tests 1 and 2 are decisive: a B-mode detection with $r > 10^{-3}$ or a 21-cm power spectrum inconsistent with the TEP temporal-horizon profile at $z \sim 20$ would directly challenge the temporal-horizon framework. Test 3 is a consistency check that can constrain parameters but does not provide standalone falsification.

# 13. Conclusion

This paper has developed and validated the temporal-horizon cosmology of the TEP framework, demonstrating that the standard FLRW Big Bang singularity is a reconstruction artifact. By imposing a globally isochronous expanding-frame description on a conformal temporal geometry, standard cosmology creates a false mathematical origin. In the TEP matter frame, this apparent origin is recognized as a relativistic asymptote where the conformal clock rate vanishes relative to the present epoch, not a physical boundary where local time or space ceases to exist.

The central temporal-horizon mapping is established: $a_{\rm eff}\to0$ corresponds to $A_{\rm clock}\to0$, where the exact observational clock map $A_{\rm clock}(z)=(1+z)^{-1}$ vanishes at infinite redshift. The dynamical response $A_{\rm dyn}(z)=\left(1+z/z_{t}\right)^{-\epsilon_{\rm dyn}}$ encodes the late-time TEP shear correction. Section 11 presents the integrated ten-step pipeline; each step constrains the next with no downstream tuning. The key findings are:

- For the temporal-horizon profile $A_{\rm clock}(\eta)=C\eta^{-p}$ with $0 \lt p\le\tfrac12$, all polynomial curvature invariants vanish at the temporal conformal boundary; the boundary is asymptotically curvature-empty, timelike-complete, and null-complete

- Timelike proper time diverges for $0 \lt p\le 1$; null affine parameter diverges for $0 \lt p\le\tfrac12$. The boundary is a regular conformal-temporal endpoint, analogous to a smooth spacelike conformal boundary

- The effective stress-energy tensor of the temporal field violates the Strong Energy Condition, removing one of the assumptions required by the Hawking-Penrose singularity theorems; geodesic completeness is established separately by the curvature and affine-parameter analysis

- Native Local Thermodynamic Evolution in an eternal universe replaces the expanding-plasma assumption entirely, providing an eternal-universe framework in which chemical states can approach long-term asymptotic equilibria; the quantitative chemical mechanism and its exposure conditions are evaluated in TEP-BBN without requiring a phenomenological screening scale.

- Standard BBN assumptions are replaced: the thermal and chemical states are modeled as candidate asymptotic equilibria (TEP-BBN) where D/H is not uniquely primordial and helium arises via baryonic cycling, without requiring a physical zero-volume Big Bang

- TEP-BBN supplies the native chemical-evolution framework and a proof of concept for local CMB thermalization; the present TEP-TH pipeline re-evaluates recombination visibility within the temporal-horizon geometry, while TEP-HC supplies the independent acoustic-sector closure.

- The temporal-horizon thermal mapping preserves a FIRAS-compatible blackbody spectrum without forbidden $\mu$ or $y$ spectral distortions

- Entropy and arrow-of-time analysis demonstrates thermodynamic regularity at the temporal horizon

- The scalar shape of the primordial perturbation spectrum is derived from fluctuations of the clock field $\zeta=\delta\ln A_{\rm clock}$, with spectral-flow parameter $n_s-1=-2\epsilon_{\rm field}$. The observed Planck value $n_s = 0.965$ constrains $\epsilon_{\rm field} = 0.0175$

- Tensor modes obey a native temporal-conformal wave equation; for $A_{\rm clock}(\eta)\sim\eta^{-p}$ the source term $A_{\rm clock}''/A_{\rm clock}\to 0$ at the horizon, so no large primordial tensor background is generated at leading order; the imported inflationary consistency relation $r=16\epsilon_{\rm field}$ is not assumed; numerical integration of the native tensor equation yields $r(k_{\rm pivot})=9\times 10^{-6}$ and $r_{\rm max}=6.26\times 10^{-4}$, well below the BICEP/Keck bound $r\lt0.036$

- CMB anisotropy spectra (TT, TE, EE) and LSS observables (matter power spectrum, growth factor $f\sigma_8$, BAO scales) are preserved via conformal acoustic equivalence as confirmed by TEP-HC (Paper 18)

The causal matter-frame universe is curvature-regular at the temporal conformal boundary. The apparent Big Bang is a temporal horizon, not a physical curvature singularity. The temporal-horizon geometry supplies a nonsingular framework in which the standard background and acoustic observables are reconstructed by the conformal mapping, while TEP-BBN provides the native chemical-evolution mechanism and a proof of concept for local CMB thermalization; the scalar perturbation shape is reproduced, and tensor production is suppressed at leading order by the vanishing of the temporal-horizon source term. Together with the distance-redshift evidence from TEP-C0 (Paper 26) and the acoustic-sector `hi_class` validation from TEP-HC (Paper 18), TEP-TH completes the logical loop without over-claiming thermal closure: the foundational geometrical and thermodynamic signatures conventionally attributed to a hot Big Bang singularity can be modeled within a static conformal temporal-transport geometry.

**Implications:** The temporal-horizon cosmology of the TEP framework replaces the standard Big Bang interpretation. The universe did not begin at a physical singularity a finite number of years ago; it extends locally backward through an infinite affine past. The temporal horizon is strictly an observational boundary where the relative conformal clock rate vanishes, meaning cosmic history must be mapped by local thermodynamic state variables rather than a global chronological stopwatch. Cosmic expansion is a geometric reconstruction of accumulated open-path conformal temporal shear. The background and thermal observational signatures conventionally attributed to a hot dense origin—light-element abundances, acoustic peaks, blackbody thermalization, and large-scale structure—can be reconstructed within the combined eternal-universe thermodynamic and conformal-acoustic framework. The tensor-to-scalar ratio must be computed from the finite transition region via the native temporal-conformal wave equation, not from an imported inflationary consistency relation.

**Outlook:** Figure 1 (Section 4.5) illustrates the conformal-boundary interpretation: the singular lower edge of standard flat $\Lambda$CDM is replaced by a smooth regular temporal conformal boundary $\mathscr{T}^{-}$, where $A_{\rm clock}\to0$ and curvature invariants vanish. The apparent Big Bang is a regular conformal-temporal endpoint, not a physical singularity. Step 09b is complete: the native tensor equation yields $r(k_{\rm pivot})=9\times 10^{-6}$ and $r_{\rm max}=6.26\times 10^{-4}$, well below current and projected CMB bounds. Closed-loop synchronization holonomy remains the primary discriminant of the non-exact/disformal sector of TEP. In the homogeneous conformal limit analysed here, $A_{\rm clock}$ gives open-path temporal redshift but does not by itself generate residual loop holonomy, since $\oint d\ln A_{\rm clock}=0$. A nonzero $\mathcal{H}_{\rm resid}$ would require the disformal contribution $B(\phi)\nabla_{\mu}\phi\nabla_{\nu}\phi$, non-metricity, or another non-exact synchronization structure, as developed in the foundational TEP paper (Jakarta).

# 14. References

- Smawfield, M.L. Temporal Equivalence Principle: Dynamic Time & Emergent Light Speed. *Zenodo* (2025). DOI: 10.5281/zenodo.16921911

- Smawfield, M.L. Temporal Equivalence Principle: A Covariant Alternative to Cosmic Expansion. *Zenodo* (2026). DOI: 10.5281/zenodo.20370143

- Smawfield, M.L. Temporal Equivalence Principle: Native hi_class Conformal Implementation, Linear Perturbation Closure, and CMB Acoustic Peak Preservation. *Zenodo* (2026). DOI: 10.5281/zenodo.20682752

- Smawfield, M.L. Temporal Equivalence Principle: Dynamical Proper Time and the Illusion of Primordial Deuterium. *Zenodo* (2026). DOI: 10.5281/zenodo.21841148

- Hawking, S.W. The occurrence of singularities in cosmology. *Proc. R. Soc. A* **294**, 511-521 (1966).

- Hawking, S.W. & Penrose, R. The singularities of gravitational collapse and cosmology. *Proc. R. Soc. A* **314**, 529-548 (1970).

- Borde, A., Guth, A.H. & Vilenkin, A. Inflationary spacetimes are incomplete in past directions. *Phys. Rev. Lett.* **90**, 151301 (2003).

- Brandenberger, R. & Peter, P. Bouncing cosmologies: progress and problems. *Found. Phys.* **47**, 797-850 (2017).

- Novello, M. & Bergliaffa, S.E.P. Bouncing cosmologies. *Phys. Rep.* **463**, 127-213 (2008).

- Ijjas, A. & Steinhardt, P.J. Entropy, black holes and the new cyclic universe. *Phys. Lett. B* **824**, 136823 (2022).

- Peebles, P.J.E. *Principles of Physical Cosmology*. Princeton University Press (1993).

- Weinberg, S. *Cosmology*. Oxford University Press (2008).

- Dodelson, S. *Modern Cosmology*. Academic Press (2003).

- Mukhanov, V.F., Feldman, H.A. & Brandenberger, R.H. Theory of cosmological perturbations. *Phys. Rep.* **215**, 203-333 (1992).

- Liddle, A.R. & Lyth, D.H. *Cosmological Inflation and Large-Scale Structure*. Cambridge University Press (2000).

- Planck Collaboration, et al. Planck 2018 results. VI. Cosmological parameters. *A&A* **641**, A6 (2020).

- Riess, A.G., et al. Milky Way Cepheid Standards for Measuring Cosmic Distances and Application to Gaia DR2: Implications for the Hubble Constant. *ApJ* **861**, 126 (2018).

- Brout, D., et al. The Pantheon+ Analysis: Cosmological Constraints. *ApJ* **938**, 110 (2022).

- Fixsen, D.J., et al. The Cosmic Microwave Background Spectrum from the Full COBE FIRAS Data Set. *ApJ* **473**, 576 (1996).

- Chluba, J. & Sunyaev, R.A. The evolution of CMB spectral distortions in the early Universe. *MNRAS* **419**, 1294-1314 (2012).

- PARTICLE DATA GROUP. Review of Particle Physics. *PTEP* **2022**, 083C01 (2022).

- Cyburt, R.H., Fields, B.D., Olive, K.A. & Yeh, T.H. Big bang nucleosynthesis: Present status. *Rev. Mod. Phys.* **88**, 015004 (2016).

- Seager, S., Sasselov, D.D. & Scott, D. A new calculation of the recombination epoch. *ApJ* **523**, L1-L5 (1999).

- Peebles, P.J.E. Recombination of the Primeval Plasma. *ApJ* **153**, 1 (1968).

- Zeldovich, Y.B. & Sunyaev, R.A. The interaction of matter and radiation in a hot-model universe. *Astrophys. Space Sci.* **4**, 301-316 (1969).

- Seljak, U. & Zaldarriaga, M. A Line of Sight Integration Approach to Cosmic Microwave Background Anisotropies. *ApJ* **469**, 437 (1996).

- Lewis, A., Challinor, A., & Lasenby, A. Efficient Computation of CMB Anisotropies in Closed FRW Models. *ApJ* **538**, 473 (2000).

- Lesgourgues, J. & Tram, T. The Cosmic Linear Anisotropy Solving System (CLASS). Part IV: efficient implementation of non-cold relics. *JCAP* **09**, 032 (2011).

- Zumalacárregui, M., Bellini, E., Sawicki, I., Lesgourgues, J. & Ferreira, P.G. hi_class: Horndeski in the Cosmic Linear Anisotropy Solving System. *JCAP* **08**, 019 (2017).

- De Felice, A. & Tsujikawa, S. f(R) Theories. *Living Rev. Rel.* **13**, 3 (2010).

- Wetterich, C. Cosmology and the fate of dilatation symmetry. *Nucl. Phys. B* **302**, 668-696 (1988).

- Wetterich, C. A universe without expansion. *Phys. Dark Universe* **2**, 184 (2013).

- Narlikar, J.V. & Arp, H.C. Flat spacetime cosmology: A unified framework for extragalactic redshifts. *Astrophys. J.* **405**, 51-56 (1993).

- Mannheim, P.D. Conformal gravity and the nature of dark matter. *Prog. Part. Nucl. Phys.* **94**, 217-272 (2017).

- Khoury, J. & Weltman, A. Chameleon cosmology. *Phys. Rev. D* **69**, 044026 (2004).

- Hinterbichler, K. & Khoury, J. Symmetron cosmology. *Phys. Rev. Lett.* **104**, 231301 (2010).

- Penrose, R. Before the Big Bang: an outrageous new perspective and its implications for particle physics. *Proc. EPAC* (2006).

- Tod, K.P. Isotropic cosmological singularities. *Gen. Relativ. Gravit.* **35**, 779-805 (2003).

- Tod, K.P. The equations of conformal cyclic cosmology. *Gen. Relativ. Gravit.* **47**, 31 (2015).

- Ratra, B. & Peebles, P.J.E. Cosmological Consequences of a Rolling Homogeneous Scalar Field. *Phys. Rev. D* **37**, 3406 (1988).

- Caldwell, R.R., Dave, R., & Steinhardt, P.J. Cosmological Imprint of an Energy Component with General Equation of State. *Phys. Rev. Lett.* **80**, 1582 (1998).

- Clifton, T., Ferreira, P.G., Padilla, A. & Skordis, C. Modified gravity and cosmology. *Phys. Rep.* **513**, 1-189 (2012).

# Acknowledgements

The author thanks his mother, J.S., who once attended a Penrose lecture in New Delhi, for a book that initiated his study of conformal infinity.

# Unified TEP Parameter Dictionary

The TEP corpus uses related but distinct symbols across its papers. This dictionary maps every parameter, its definition, the paper where it is primary, and its fiducial or fitted value.

| Symbol | Definition | Primary Paper | Fiducial / Fitted Value |
| --- | --- | --- | --- |
| $A_{\rm clock}(z)$ | Exact observational clock/redshift map: $A_{\rm clock}=(1+z)^{-1}$ | TEP-TH | $(1+z)^{-1}$ (exact) |
| $A_{\rm dyn}(z)$ | Dynamical shear response: $\left(1+z/z_t\right)^{-\epsilon_t}$ | TEP-TH | Modifies late time evolution |
| $\alpha_A$ | Temporal-shear conformal amplitude in Jordan-frame notation | TEP-HC | $-0.0028$ (Planck best-fit) |
| $\alpha_M$, $\alpha_B$, $\alpha_K$, $\alpha_T$ | Runtime Bellini–Sawicki EFT functions: $\alpha_M=-2\alpha_A$, $\alpha_B=2\alpha_A$, $\alpha_K=-5\alpha_A^2$, $\alpha_T=0$ | TEP-HC | Derived from $\alpha_A$ |
| $\epsilon_T^{\rm los}$ | Late-time line-of-sight transport amplitude (C0 supernova fit) | TEP-C0 | $\mathcal{U}[0, 1.0]$ (prior); posterior peaked near $\sim 0.89$ |
| $\epsilon_T^{\rm CMB}$ | C0 background/acoustic diagnostic amplitude | TEP-C0 | $-0.0015\pm0.0037$ |
| $\epsilon_T^{\rm HC}$ | Native hi_class homogeneous conformal amplitude | TEP-HC | $0.00602\pm0.00493$ |
| $\epsilon_{\rm dyn}$ | Dynamical temporal-horizon response | TEP-TH | Modulates late-time expansion |
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

# 15. Data Availability and Reproducibility

All data and analysis code required to reproduce the results presented in this work are available in the public repository at https://github.com/matthewsmawfield/TEP-TH.

All pipeline outputs, posterior samples, and the exact step configuration files are released in the Zenodo repository (DOI: 10.5281/zenodo.20723059) under CC-BY 4.0. The step scripts and `run_all.py` orchestration script are provided in the GitHub repository.

## Repository Structure

- **scripts/steps/**: Python pipeline scripts for temporal-horizon analysis (steps 00-10, with 09b)

- **scripts/steps/th_common.py**: Shared utilities for TEP-TH pipeline

- **data/raw/**: Raw cosmological data sources

- **data/processed/**: Processed data products

- **results/**: Pipeline outputs (JSON, CSV, figures)

- **logs/**: Pipeline execution logs

## Pipeline Execution

The TEP-TH pipeline can be executed using the provided scripts:

`cd scripts/steps
python step_00_temporal_horizon_mapping.py
python step_01_matter_frame_curvature.py
python step_02_geodesic_completeness.py
python step_03_effective_stress_energy.py
python step_04_full_bbn_abundances.py
python step_05_recombination_visibility.py
python step_06_cmb_blackbody_origin.py
python step_07_entropy_arrow.py
python step_08_primordial_perturbation_boundary.py
python step_09b_native_tensor_integration.py
python step_10_cmb_lss_class.py
`

## Key Pipeline Outputs

- **step_04_full_bbn_abundances.json / .csv**: BBN abundance validation (Y_p, D/H, He3/H, Li7/H, N_eff)

- **step_05_recombination_visibility.json / .csv**: Recombination epoch ($z_*$, $r_s$, $\theta_s$), LCDM and TEP comparison

- **step_09b_native_tensor_integration.json / .csv**: Native tensor-mode integration (r(k_pivot), r_max, Bogoliubov coefficients)

- **step_10_cmb_lss_class.json / .csv**: CMB/LSS consistency scorecard (Planck comparison, sigma8)

## Dependencies

The pipeline requires Python 3.11+ with the following packages:

- numpy

- scipy

- matplotlib (for plotting)

## Data Sources

The pipeline uses publicly available cosmological data:

- FIRAS CMB monopole spectrum (NASA LAMBDA)

- BBN abundance measurements (PDG 2024)

- Planck 2018 cosmological parameters

## Version Control

This work uses version v0.3 (Thika) of the TEP-TH pipeline, first published 18 June 2026. The repository maintains a complete history of all changes through Git version control.
