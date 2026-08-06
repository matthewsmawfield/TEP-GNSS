# Temporal Equivalence Principle: The Dirac Limit of Dynamical Proper Time
**Matthew Lukin Smawfield**
Version: v0.2 (Qatar)
First published: 12 June 2026 · Last updated: 2 July 2026
DOI: 10.5281/zenodo.20572697

---

## Abstract

Standard relativistic quantum mechanics, including the Klein–Gordon and Dirac equations, is recovered here as the screened, flat-frame tangent limit of a deeper dynamical proper-time phase transport governed by the Temporal Equivalence Principle (TEP). By treating proper time *τ* as a dynamical scalar field *φ* rather than a universal parameter, three foundational structures are derived and two are geometrically reinterpreted.

(1) The phase action *S = −mc^{2} ∫ dτ̃* emerges as the primitive geometric driver, with mass appearing in the primitive action as the parameter governing the oscillator frequency, *ω_{0} = mc^{2} / ℏ*, modulated by the conformal factor in the causal matter metric *g̃_{μν}*. (2) The Klein-Gordon equation is derived from the minimal geometric Lagrangian in the causal metric and verified via WKB / eikonal expansion; its eikonal limit recovers the *g̃*-Hamilton-Jacobi equation, not via operator substitution. (3) The Dirac operator is recovered as the local Clifford/tetrad representation in the isochronous background — it emerges in the limit where temporal shear *Σ_{μ}* and disformal coupling *B(φ)* are negligible. (4) Spin-1/2 is reinterpreted as temporal-orientation holonomy of the proper-time phase frame, and antiparticles as reversed proper-time orientation within the temporal-orientation bundle, eliminating the need for a macroscopic two-sheeted spacetime. (5) The spinor structure of relativistic quantum mechanics may be reinterpreted geometrically: Dirac's 1928 spinor encoded temporal-orientation holonomy without access to a dynamical proper-time geometry. These results recover standard relativistic quantum mechanics as the screened tangent-space limit of the TEP causal geometry, in the regime where the temporal field is frozen and locally constant over the interaction scale; the saturation scale *ρ_{T} ≈ 20 g/cm^{3}* is empirically calibrated in TEP-UCD (Paper 6).

Keywords: quantum foundations, Dirac equation, proper time, phase transport, spin, antimatter, scalar-tensor theories, geometric quantum mechanics, temporal equivalence principle, Hamilton-Jacobi equation, Klein-Gordon equation

## 1. Introduction: The Background-Dependency of Relativistic Quantum Theory

### 1.1 The External-Metric Postulate

Every equation of quantum mechanics — from the Schrödinger equation to the Dirac equation, from the path integral to the operator formalism of Quantum Field Theory — is formulated on a prescribed spacetime background. The metric *g_{μν}* is not dynamical in the quantum sector; it is an external input, fixed either as the Minkowski metric *η_{μν}* in flat-space QFT or as a classical solution to Einstein's equations in QFT in curved spacetime. In neither case does the quantum field back-react on the temporal geometry. Proper time *τ* is computed from a metric that is unaffected by the quantum state of the matter fields it governs.

**Screening projection notice.** Screening in TEP is represented at theory level by the environmental operator S_Σ(E). Quantities such as ρ_T, R_T(M), S_⊕(r), compactness Φ/c^2, local stellar density, thermal epoch, coherence length, proximity, and boundary geometry are domain-specific projections of E, not independent screening mechanisms and not interchangeable universal thresholds.

**Evidence status.** This paper is classified as a *candidate microscopic completion* in the TEP corpus. It derives the Klein–Gordon and Dirac operators as the screened tangent limit of a dynamical proper-time geometry, but full Born-rule recovery, gauge-invariant interaction structure, and quantitative QED matching remain open closure tasks. The antiparticle reinterpretation, wavefunction-collapse model, and entanglement ontology presented here are theoretical proposals, not empirically validated replacements for standard quantum mechanics. The results should be evaluated as structural exploration rather than as direct empirical proof of TEP.

This is not a bug in the formalism; it is a deliberate approximation. The energy scales of known particle physics are so far below the Planck scale that gravitational back-reaction is negligible. However, the approximation becomes a structural limitation when one asks whether the quantum state itself — the phase coherence, the entanglement, the spin orientation — might be fundamentally tied to the geometry that the metric describes. In standard relativistic quantum theory, the metric is a stage; the quantum fields are actors. The Temporal Equivalence Principle reverses this hierarchy.

**Conceptual continuity with Einstein's geometric programme.** This reversal places TEP in conceptual continuity with Einstein's search for a continuous geometric foundation beneath the discrete formal machinery of quantum theory. Einstein's later unified-field programme can be read not as a failure of the geometric instinct itself, but as evidence that the available geometry remained tied too closely to spacetime curvature while quantum phase, spin, and matter–antimatter structure were treated as algebraic additions. TEP shifts the geometric object from the gravitational metric alone to the causal proper-time geometry governing matter-clock phase transport. In this sense, the present paper does not replace the empirical success of relativistic quantum mechanics; it asks whether the Klein–Gordon and Dirac structures are the flat-frame, screened expression of the continuous geometry Einstein sought but could not formulate in terms of dynamical proper time.

A critical distinction must be made. General Relativity *does* recognise local proper-time variance — gravitational time dilation is a well-measured phenomenon. A clock near a massive body ticks slower than a distant clock. However, GR treats this variance as a curvature effect within a single, globally defined spacetime manifold. The metric *g_{μν}* is universal; all matter and light propagate on the same geometric background. The temporal variance is integrable: in many non-rotating idealised settings one can construct a synchronous foliation, and the proper time along any path is computed from the same metric. Beneath this lies a deeper structural assumption: *global isochrony*, the premise that all local proper-time clocks within a spacetime neighbourhood can be synchronised to a single universal time parameter. It is this assumption of global isochrony — not the curvature itself — that is the exact structural limitation that dynamical proper-time geometry dismantles.

The Temporal Equivalence Principle goes further. It treats proper time not merely as a parameter computed from a prescribed metric, but as a *dynamical scalar field* *φ* governed by its own field equations, coupled to matter density through a conformal–disformal metric. The metric that governs quantum phase transport is not the observed metric *g_{μν}*; it is the causal matter metric *g̃_{μν}*, which differs from the observed metric by a conformal factor *A(φ)* that modulates with local matter density. In standard relativistic quantum theory, even in curved spacetime, the metric is externally prescribed and the quantum field is quantised on that fixed background. The Bogoliubov transformation between vacua encodes particle creation (the Unruh and Hawking effects), but the metric itself remains classical and unaffected. This paper targets the structural assumption that the metric governing quantum phase transport is externally prescribed rather than dynamically coupled to the matter fields it governs.

### 1.2 The Inventions It Forces

When a dynamical, density-dependent temporal landscape is forced into the framework of a fixed background metric, the theory must invent compensating structures to account for the discrepancy:

- *Wavefunction collapse:* The sudden, non-unitary reduction of the quantum state upon measurement may reflect the inability of a fixed-background formalism to accommodate the geometric interaction between a probe and the local temporal shear field. In a dynamical proper-time geometry, the measurement process is reinterpreted as the equilibration of probe and system to a shared temporal contour: as the probe approaches the interaction region, the joint shear field relaxes to a common isochronous manifold. The apparent collapse may be reinterpreted as the geometric relaxation of two initially independent temporal topologies to a single synchronised phase, rather than a discontinuous jump. The fluid dynamics of the temporal shear field are proposed to conserve geometric stress; what standard quantum mechanics records as projection onto an eigenstate is, in the TEP framework, the settling of the combined system onto a contour of constant proper-time phase. The amplitude-curvature term that appears in the WKB expansion of the geometric Klein–Gordon equation (§2.4) is the natural point of contact with the quantum-potential term of de Broglie–Bohm mechanics: TEP identifies it as a transport effect of the causal phase geometry, not as a hidden-variable ontology. A rigorous derivation of this correspondence — including the recovery of the Born rule from geometric statistics — from the full TEP field equations, including the non-linear disformal sector, is beyond the scope of this paper and is addressed in TEP-KIN (Paper 25).

- *Intrinsic spin:* In standard quantum mechanics, spin-1/2 is described by an abstract SU(2) representation without reference to spatial extent. In the TEP framework, spin-1/2 is the temporal-orientation holonomy of a localized topological charge — a geometric, not algebraic, property.

- *Antiparticles as separate ontological species:* The Dirac equation's negative-energy solutions are reinterpreted as a separate particle species. In the TEP framework, the antiparticle sector is the reversed proper-time phase orientation in the orientation-conjugate branch of the temporal-orientation bundle.

- *Virtual particles as force carriers:* The exchange of momentum between particles across empty space is modeled by the emission and absorption of unobservable virtual bosons. In the TEP framework, forces may be reinterpreted as routed through the continuous geometry of the disformally tilted light cone. The derivation of the Maxwell equations and full quantum electrodynamic interactions from the TEP disformal geometry is not provided in this paper; it is addressed in TEP-SPIN (Paper 24).

### 1.3 The Temporal Equivalence Principle

The Temporal Equivalence Principle (TEP) treats proper time as a dynamical scalar field *φ* governed by a conformal–disformal metric (Paper 0). Matter clocks tick at rates set by the conformal factor *A(φ) = exp(β_{A} φ / M_{Pl})*. The critical saturation scale *ρ_{T} ≈ 20 g/cm^{3}* (empirically calibrated in TEP-UCD, Paper 6, New Delhi, and cross-checked across compact-object and galactic scales) is the Temporal Topology saturation scale at which screening effects saturate; it is a property of the theory's non-linear regime and not a binary threshold. In the screened limit, where the observable shear sector *Σ_{μ}^{obs} → 0* with *A(φ)* normalized to its ambient environmental value, the locally observable Temporal Shear is suppressed and standard physics is recovered. In the unscreened regime, the full geometric structure of the Temporal Topology is manifest. The disformal coupling *B(φ)* controls the tilting of the light cone by the temporal gradient; in the screened limit the observable disformal response is suppressed and interactions become isotropic.

What "screened" means for a scattering event requires care. Following Paper 0 (§7), recovery of standard physics is controlled by suppression of the observable shear sector, *Σ_{μ}^{obs} = 𝒮_{Σ}(ℰ) Σ_{μ} → 0*, not by comparison of any local density against *ρ_{T}*; high-energy probes are represented by channel-specific response coefficients *κ_{X}* rather than by the ambient bulk-density screening function. For collider processes the relevant statement is kinematic: the scalar field's configuration is set by boundary conditions over its correlation length (*L_{c} ≈ 4 × 10^{3} km* in the terrestrial calibration of Paper 6), so across the ~10^{−18} m, ~10^{−27} s extent of a TeV-scale interaction the field is frozen and locally constant — *Σ_{μ} ≈ 0* and *B(φ)(∇φ)^{2} ≈ 0* over the entire interaction region. A frozen, locally constant field is precisely the tangent-space limit of Section 3. The KG–Dirac and perturbative-QFT descriptions therefore remain exact for currently accessible particle-physics experiments, while long-baseline, long-integration interferometry — which accumulates phase over scales comparable to field gradients — is the probe class that can resolve the unscreened structure.

The foundational geometric framework of TEP was established in the original theory paper (Smawfield 2026, Paper 0, DOI: 10.5281/zenodo.16921911), which introduced the full disformal metric *g̃_{μν} = A^{2}(φ) g_{μν} + B(φ) ∇_{μ}φ ∇_{ν}φ* with *A(φ) = exp(β_{A} φ / M_{Pl})*, demonstrated local Lorentz invariance as a theorem, and introduced the sector ontology of Temporal Topology (spatial/covariance structure of *ln A(φ)*), Temporal Shear (*Σ_{μ} = ∇_{μ} ln A*), and the saturation scale *ρ_{T}*. This paper works in the conformal limit *B(φ) = 0* and adopts the foundational convention throughout: the symbol *A(φ)* denotes the original conformal factor of the Jakarta theory, so the causal matter metric reads *g̃_{μν} = A^{2}(φ) g_{μν}*. The screened limit *A(φ) → 1* recovers the standard Minkowski background; the defect core, where the conformal response saturates and the orientation sector carries winding, marks the topological defects identified in Section 4. The conformal limit suffices for the KG–Dirac tangent-limit derivation because a non-zero disformal response does not invalidate the conformal recovery; it marks the breakdown of the isotropic tangent limit and introduces direction-dependent causal corrections. The full disformal structure becomes essential for interaction routing, synchronization holonomy, interferometric observables, and multi-messenger tests, addressed in companion papers.

Companion papers in this series develop TEP across a wide range of mass densities, from subatomic scales (TEP-SPIN) through laboratory interferometry (TEP-KIN) to cosmological distances (TEP-C0). This paper addresses the quantum regime directly, arguing that the KG–Dirac sector of relativistic quantum theory is the flat-frame, isochronous tangent limit of a deeper dynamical proper-time geometry.

Paper 0 establishes local Lorentz invariance as a theorem: in local freely falling frames of *g̃_{μν}*, physics reduces to special relativity (Axiom A2). The present paper is the quantum-sector instantiation of that theorem. Sections 2–3 exhibit the explicit KG–Dirac tangent limit and, critically, the leading departure from it — the temporal-shear coupling *Σ_{μ}* in the conformal Dirac operator (§3.2.1) — so that the recovery of standard relativistic quantum mechanics is not asserted but constructed, with its first correction term in hand.

### 1.4 Structure of This Paper

Section 2 establishes the phase action and recovers the Klein-Gordon relation as the Euler-Lagrange equation of the minimal scalar-field Lagrangian in the causal metric. Section 3 delivers the central derivation: the Dirac operator is recovered as the local tetrad representation in the screened limit. Section 4 reinterprets spin and antiparticle structure as geometric orientations. Section 5 concludes and outlines the synthesis with companion papers.

### 1.5 Broader Framework Context

Beyond the tangent-space derivations developed below, TEP carries conceptual implications that extend across the companion series. Bell's theorem rules out any theory in which quantum correlations are explained by pre-assigned local hidden variables satisfying the usual separability and factorisation assumptions. TEP does not attempt to restore such a theory. Instead, it challenges the deeper premise that entangled systems are fundamentally separable subsystems embedded in an inert, externally prescribed background.

In the TEP framework, the interval between entangled charges is not an empty stage but part of the same dynamical temporal geometry that governs phase transport. Entanglement is therefore interpreted not as a superluminal signal between isolated particles, but as a non-separable geometric contour in the conformal–disformal proper-time field. Measurement perturbs one boundary of a shared temporal configuration; the observed correlation reflects the global constraint structure of that configuration rather than the transmission of information through space.

This view is best described as geometric non-separability rather than conventional local realism. It preserves the empirical force of Bell-type violations while relocating their ontological meaning: Bell excludes separable local hidden-variable completions of quantum mechanics, but it does not by itself exclude a realist theory in which the relevant physical object is an extended temporal-geometric structure rather than two independent point systems. The full disformal treatment of measurement, entanglement, and geometric stress propagation is developed in TEP-KIN and TEP-SPIN. The present paper establishes the tangent-space quantum limit on which those extensions build.

Temporal-topology drag denotes the influence of large-scale temporal-field configurations on local particle dynamics, mediated by the conformal factor *A(φ)* rather than spacetime curvature; environment-dependent mass emerges from modulation of the proper-time oscillator frequency by *A(φ)*. What standard fixed-background quantum theory models as virtual force carriers and statistical wavefunctions may be effective descriptions of geometric stress propagation in a contiguous temporal fluid. These phenomena furnish concrete experimental signatures developed in TEP-SPIN (Paper 24) and TEP-KIN (Paper 25).

**Cosmological clock-rate interpretation.** The same proper-time oscillator law also supplies the local quantum input for the cosmological papers in this series. If atomic emission frequencies are governed by *ω_{local}(φ) = ω_{0} A(φ)*, then spectra emitted across cosmological gradients of the temporal field may contain a conformal clock-rate component in addition to standard Doppler or metric-expansion interpretations. The present paper does not develop that cosmological model; it records the quantum-sector mechanism. The full redshift, distance-ladder, and effective-dark-energy interpretation is developed in TEP-C0.

### 1.6 Conventions and Notation

Unless otherwise stated, we use metric signature *(+, −, −, −)*. Paper 0 works in the *(−,+,+,+)* signature; this paper uses *(+, −, −, −)*, and all imported expressions have been converted. Natural units *c = ℏ = 1* are used inside derivations; dimensional factors are restored in summary equations. With *x^{0} = ct*, the Klein–Gordon mass term is *m^{2}c^{2} / ℏ^{2}*, and the Dirac mass term is *mc / ℏ*. Throughout, "screened limit" denotes *Σ_{μ}^{obs} → 0* with *A(φ)* normalized to its ambient environmental value (Paper 0, §7); the shorthand *A → 1* refers to this normalization.

## 2. Proper-Time Phase Transport

### 2.1 The Phase Action

The fundamental action for a massive particle in the TEP framework is:

S = −mc^{2} ∫ dτ̃

where *τ̃* is the dynamical proper time measured along the particle's worldline in the causal matter metric *g̃_{μν}*. This is not an abstract phase — it is the *physical accumulated phase* of the matter-clock oscillator, whose frequency is set by the local conformal factor *A(φ)*.

Dimensional analysis confirms the primitive status of this action. The dimensions are:

[S] = [m] [c^{2}] [τ̃] = M L^{2} T^{−1} = [ℏ]

Thus *S/ℏ* is dimensionless, as required for a phase. Given Axiom A2 of Paper 0 — all non-gravitational processes evolve in the proper time of *g̃* — this action is the unique Lorentz-invariant, first-order geometric functional of the proper-time interval. The mass *m* appears not as an inertial parameter but as the frequency of phase accumulation: *ω_{0} = mc^{2} / ℏ*.

The causal matter metric is related to the observed metric *g_{μν}* by the conformal transformation:

g̃_{μν} = A^{2}(φ) g_{μν}

In the screened limit, where the observable shear sector *Σ_{μ}^{obs} → 0* with *A(φ)* normalized to its ambient environmental value, *A(φ) → 1* and *g̃_{μν} → g_{μν}*. In the unscreened regime, the particle's phase accumulates along geodesics of the causal metric, not the observed metric.

### 2.2 The Massive Particle as Proper-Time Oscillator

A massive particle is defined geometrically as a stable, local proper-time oscillator governed by the causal metric. The oscillator's natural frequency in the flat-frame limit is:

ω_{0} = mc^{2} / ℏ

In the full TEP geometry, this frequency is modulated by the conformal factor:

ω_{local}(φ) = ω_{0} · A(φ)

Mass parameter and conformal rescaling. The bare mass *m* enters the primitive action *S = −mc^{2} ∫ dτ̃*. The local oscillator frequency is *ω_{local} = mc^{2} A(φ) / ℏ*. The Klein-Gordon equation uses the bare parameter *m* because the conformal modulation is already encoded in the causal d'Alembertian *□̃* (via *A^{2}* in the metric). The effective inertial mass measured by local clock response reduces to *m* when *A → 1* in the screened limit.

This local oscillator law is also the quantum-sector input used by the cosmological papers in this series. Since atomic emission frequencies are governed by the same matter-clock phase rate, *ω_{local}(φ) = ω_{0} A(φ)*, spectra emitted across large gradients of the temporal field need not be interpreted solely as Doppler or metric-expansion effects. In the present paper this observation is not developed into a cosmological model; it simply records the local mechanism by which conformal proper-time modulation enters atomic frequency standards. The cosmological redshift interpretation is developed in TEP-C0 (Paper 26), where the accumulated variation of *A(φ)* along cosmological baselines is compared with the standard expansion history.

### 2.3 The g̃-Hamilton-Jacobi Equation

The Hamilton-Jacobi equation for the proper-time phase action in the causal metric is:

g̃^{μν} ∂_{μ}S ∂_{ν}S = m^{2}c^{2}

This is the fundamental classical equation governing the phase transport of a massive particle. It is not an operator equation — it is the Hamilton-Jacobi equation in a curved geometry. The correspondence between the wave equation and this classical equation is constructed geometrically (§2.4) rather than imposed by operator substitution; the quantum sector itself enters through Axiom A2.

Metric signature convention. Throughout this paper, the metric signature is *(+, −, −, −)*. In the flat Minkowski limit, with *x^{0} = ct* and *η_{μν} = diag(1, −1, −1, −1)*, the Hamilton-Jacobi equation takes the explicit form:

(∂_{0}S)^{2} − |∇S|^{2} = A^{2}(φ) m^{2}c^{2}

### 2.4 Deriving the Klein-Gordon Equation from Geometric First Principles

In the following derivation we set *c = ℏ = 1*; dimensional factors are restored in the summary equations. The Klein-Gordon equation is derived as the Euler-Lagrange equation of the minimal scalar-field Lagrangian in the causal metric. The derivation proceeds in three steps.

#### Step 1: The geometric scalar-field Lagrangian.

The phase field *Ψ* is a complex scalar field propagating on the causal manifold with metric *g̃_{μν}*. The covariant, second-order Lagrangian for a massive scalar field that reduces to the standard Klein-Gordon Lagrangian in the flat limit is:

L = ½ ( g̃^{μν} ∂_{μ}Ψ^{*} ∂_{ν}Ψ − m^{2} |Ψ|^{2} )

The metric tensor *g̃^{μν}* raises indices; the inverse metric is *g̃^{μν} = A^{−2}(φ) g^{μν}*. The bare mass parameter *m* enters here because it is the only dimensionful parameter available from the primitive action *S = −mc^{2} ∫ dτ̃*. This Lagrangian follows directly from the causal metric *g̃_{μν}* via minimal coupling; its form is fixed by the geometric structure established in Section 2.1. The linear structure of the resulting wave equation follows from the stationarity condition on this action, exactly as in standard field theory, but now formulated in the causal geometry.

#### Step 2: Euler-Lagrange equation.

Varying the action *S = ∫ L √−g̃ d^{4}x* with respect to *Ψ^{*}* gives:

∂_{μ} ( √−g̃ g̃^{μν} ∂_{ν}Ψ ) − √−g̃ m^{2} Ψ = 0

Dividing by *√−g̃* yields the causal d'Alembertian form:

□̃ Ψ + m^{2} Ψ = 0

where *□̃ = (1 / √−g̃) ∂_{μ}( √−g̃ g̃^{μν} ∂_{ν} )* is the d'Alembertian operator constructed from the causal metric *g̃_{μν}*.

#### Step 3: Connection to the Hamilton-Jacobi equation via WKB / eikonal expansion.

To verify that this linear wave equation is consistent with the classical Hamilton-Jacobi equation derived in Section 2.3, insert the WKB / eikonal ansatz:

Ψ = R e^{iS/ℏ} , R, S ∈ ℝ

Substituting into *(□̃ + m^{2}) Ψ = 0* and dividing by *e^{iS/ℏ}* yields an equation that can be organised by powers of *ℏ*. Before collecting orders, recall the exact decomposition of the causal d'Alembertian established in Step 2. With *g̃_{μν} = A^{2}(φ) η_{μν}* in the flat-background limit,

□̃ = A^{−2}(φ) (□_{M} + 2 Σ^{μ} ∂_{μ})

where *□_{M} = ∂_{0}^{2} − ∇^{2}* is the Minkowski d'Alembertian. The conformal factor *A(φ)* is the same scalar field-dependent rescaling introduced in Section 2.1; throughout the eikonal expansion it is treated as a prescribed geometric background, not as part of the *ℏ*-expansion of *Ψ*. Its derivatives *∂_{μ}A = A Σ_{μ}* enter only through the second, temporal-shear term.

At leading order *O(ℏ^{−2})*, the real part recovers the Hamilton-Jacobi equation. The conformal bookkeeping at this order is explicit. From the WKB ansatz, *∂_{ν}Ψ = (i / ℏ) Ψ ∂_{ν}S + O(ℏ^{0})*, so the shear term is *O(ℏ^{−1})* and does not contribute at *O(ℏ^{−2})*. The conformal factor therefore enters the leading order exclusively through the kinetic prefactor *A^{−2}(φ)* in the first term: gradients of the phase *S* are contracted with the inverse causal metric *g̃^{μν} = A^{−2}(φ) η^{μν}* before any quantum corrections appear. The mass term carries no conformal prefactor (it is the bare parameter *m* from the primitive action). Collecting the *ℏ^{−2}* coefficients therefore gives

−A^{−2}(φ) R [(∂_{0}S)^{2} − |∇S|^{2}] + m^{2} R = 0

Dividing by *R* and multiplying by *−A^{2}(φ)* immediately yields the *g̃*-Hamilton-Jacobi equation in natural units, with the conformal mass shift *A^{2}(φ)* on the right-hand side:

(∂_{0}S)^{2} − |∇S|^{2} = A^{2}(φ) m^{2}

The appearance of *A^{2}(φ)* at this order is not an independent postulate; it is the classical signature of propagation on the causal manifold. In the screened limit *A → 1*, the equation reduces to the standard Hamilton-Jacobi form; in the unscreened regime, the effective inertia of the phase front is modulated by the local conformal clock-rate rescaling before any *O(ℏ^{−1})* transport or *O(ℏ^{0})* quantum-potential terms enter.

At next order *O(ℏ^{−1})*, the imaginary part gives the transport equation in the causal metric:

∂_{0}(R^{2} ∂_{0}S) − ∇ · (R^{2} ∇S) + 2A^{−1}R^{2} η^{μν}(∂_{μ}A)(∂_{ν}S) = 0

which expresses conservation of the probability current *j^{μ} = R^{2} ∂^{μ}S* modified by the temporal shear *Σ_{μ} = ∂_{μ} ln A*. In the screened limit *A → 1*, *∂_{μ}A → 0*, this reduces to the standard transport equation.

At finite order *O(ℏ^{0})*, the quantum-potential term with shear coupling appears:

A^{−2} □_{M} R / R + 2A^{−3} η^{μν}(∂_{μ}A) (∂_{ν}R) / R

which reduces to *□_{M} R / R* in the screened limit and is suppressed by *ℏ^{2}* relative to the Hamilton-Jacobi term.

This is the natural point of contact with the quantum-potential term familiar from de Broglie–Bohm formulations [11]: the standard amplitude-curvature term remains, while TEP supplies an additional shear-coupled geometric correction. TEP does not require a hidden-variable ontology; it identifies the potential-like term as a transport effect of the causal phase geometry.

The Klein-Gordon equation is thus *derived* from the minimal geometric Lagrangian in the causal metric, and its eikonal limit is verified to coincide with the *g̃*-Hamilton-Jacobi equation. The inputs are the causal metric *g̃_{μν}*, the bare mass *m* from the primitive action, and the standard scalar-field Lagrangian minimally coupled to that metric. No operator substitution is required.

In the screened limit *g̃_{μν} → η_{μν}* and the standard Klein-Gordon equation is recovered. In the unscreened regime, the causal d'Alembertian encodes the full geometric structure of the temporal field, including additional temporal-shear coupling terms proportional to *∂_{μ}A*.

## 3. The Dirac Operator as a Screened Limit

### 3.1 The Standard Dirac Equation

The Dirac equation, the cornerstone of relativistic quantum mechanics, is:

(iγ^{μ}∂_{μ} − m) ψ = 0

where *γ^{μ}* are the Dirac matrices satisfying the Clifford algebra *{γ^{μ}, γ^{ν}} = 2η^{μν}*. This equation is noted for its Lorentz covariance, its prediction of antimatter, and its role as the foundation of Quantum Electrodynamics.

### 3.2 Algebraic Flat-Space Recovery

TEP attributes the difficulty of geometrising the quantum sector to a metric misattribution: particle phase transport was mapped onto the gravitational metric while proper time remained a universal parameter. TEP demonstrates that the Klein–Gordon and Dirac equations are the screened, flat-frame tangent limits of a deeper dynamical proper-time phase transport. The geometric operator successfully reduces to the familiar Dirac operator only when the temporal background is artificially flattened. By treating proper time τ as a dynamical scalar field φ and deriving the action from the causal matter metric *g̃_{μν}*, the geometric language of a deeper temporal topology is naturally revealed.

The standard Dirac equation is recovered as the local Clifford/tetrad representation in the isochronous (screened) limit. This is an exact mathematical result: the geometric operator reduces to the familiar Dirac operator when the temporal background is flat.

In curved spacetime, spinors cannot be defined directly on the manifold. They require a local frame (tetrad) *e^{a}_{μ}* at each point, related to the metric by:

g_{μν} = e^{a}_{μ} e^{b}_{ν} η_{ab}

The Dirac matrices in curved spacetime are defined as *γ^{μ} = e_{a}^{μ} γ^{a}*, where *γ^{a}* are the flat-space Dirac matrices. The Dirac operator in curved spacetime becomes:

(iγ^{μ}∇_{μ} − m) ψ = 0

where *∇_{μ}* is the spin-covariant derivative, incorporating the spin connection *ω^{ab}_{μ}*.

The flattening conditions. The standard Dirac equation *(iγ^{μ}∂_{μ} − m) ψ = 0* makes two critical assumptions that are never stated explicitly:

- *The temporal shear vanishes:* *Σ_{μ} = ∇_{μ} ln A(φ) = 0*. This means the conformal factor is constant, and the causal metric is identical to the observed metric.

- *The observable disformal response is suppressed:* *B(φ)(∇φ)² → 0*. This means the light-cone tilt becomes phenomenologically negligible, and all interactions are effectively isotropic in the screened regime.

When these two conditions are imposed, the causal metric *g̃_{μν}* reduces to the Minkowski metric *η_{μν}*, the tetrad field becomes the identity, and the spin-covariant derivative reduces to the ordinary partial derivative. The full geometric Dirac operator:

(i e^{μ}_{a} γ^{a} ∇_{μ} − m) ψ = 0

collapses to the standard form:

(iγ^{μ}∂_{μ} − m) ψ = 0

### 3.2.1 The Conformal TEP Dirac Operator Before Screening

Before imposing the flattening conditions, the conformal sector of the causal metric is *g̃_{μν} = A^{2}(φ) η_{μν}*. The corresponding tetrad and curved-space Dirac matrices are:

g̃_{μν} = A^{2}(φ) η_{μν}, ẽ^{a}_{μ} = A(φ) δ^{a}_{μ}, ẽ_{a}^{μ} = A^{−1}(φ) δ_{a}^{μ}, γ̃^{μ} = A^{−1}(φ) γ^{μ}.

The spin-covariant Dirac operator on this conformal geometry therefore takes the unrescaled four-dimensional conformal form

D̃ ψ = i A^{−1} γ^{μ} (∂_{μ} + 3/2 Σ_{μ}) ψ − m ψ = 0,

where *Σ_{μ} = ∂_{μ} ln A(φ)* is the temporal shear. Under the Weyl-rescaled spinor convention *χ = A^{3/2} ψ*, the derivative shear term can be absorbed into the rescaled spinor field. The equation may then be written equivalently as

A^{−5/2} i γ^{μ} ∂_{μ} χ − m A^{−3/2} χ = 0,

or, after multiplying by *A^{5/2}*,

i γ^{μ} ∂_{μ} χ − m A χ = 0.

Thus the temporal-shear contribution can be represented either as an explicit spin-connection term in the unrescaled matter-clock spinor *ψ*, or as a conformal mass modulation in the Weyl-rescaled spinor *χ*. In the matter-clock convention used here, the explicit *Σ_{μ}* term is retained because it displays the temporal-shear correction directly. In the screened limit:

A(φ) → 1, Σ_{μ} → 0 ⇒ i γ^{μ} ∂_{μ} ψ − m ψ = 0.

This *Σ_{μ}*-dependent term is the explicit conformal TEP correction: standard Dirac dynamics is recovered only when the temporal-shear contribution vanishes in the screened limit.

### 3.2.2 Disformal Departure from Conformal Collapse

The conformal sector *g̃_{μν} = A^{2}(φ) η_{μν}* alone admits tetrad rescaling *e^{a}_{μ} → A(φ) δ^{a}_{μ}*. The rank-one disformal correction *B(φ) ∇_{μ}φ ∇_{ν}φ* cannot be absorbed into a single conformal factor: for generic *∇_{μ}φ ≠ 0*, off-diagonal components such as *g̃_{0x} = B(φ) ∂_{t}φ ∂_{x}φ* are non-zero unless *B(φ) = 0*. Even with temporal shear suppressed (*A(φ) = 1*, *Σ_{μ} = 0*), the metric departs from Minkowski through *g̃_{00} − 1 = B(φ)(∂_{t}φ)^{2}* and *g̃_{xx} + 1 = B(φ)(∂_{x}φ)^{2}*. Standard Clifford collapse therefore requires both flattening conditions of Section 3.2. SymPy Audit A.4 in `results/sympy_audit.log` verifies this signpost; the full disformal tensor audit (inverse metric, null-cone tilt, Christoffel symbols, synchronization holonomy) is in TEP-KIN (Paper 25), `results/disformal_kinematics_audit.log`.

Bridge to the full matter metric. The conformal tangent-limit proof of Sections 2–3 is not an abandonment of the Jakarta ansatz *g̃_{μν} = A^{2}(φ) g_{μν} + B(φ) ∇_{μ}φ ∇_{ν}φ*; it isolates the tangent-space algebra that the KG–Dirac sector of relativistic quantum theory already assumes. Particle-physics experiments operate in the screened regime where *B(φ)(∇φ)^{2} → 0*; in the highly screened regime of particle colliders, where the scalar field is frozen and locally constant over the interaction scale (§1.3), this disformal suppression is not merely a convenient approximation but practically exact. The Dirac and Klein-Gordon derivations of this paper are therefore the exact limiting forms of the same geometric operator written against the full *g̃_{μν}*. Outside that limit, the disformal sector does not modify the Clifford algebra by a small perturbative correction; it replaces the isotropic Minkowski inner product with a direction-dependent causal structure. TEP-KIN (Paper 25) carries the full metric through: the symbolic inverse *g̃^{μν}*, the effective refractive index *n*_{eff} that tilts null cones, the Christoffel symbols that route geodesics through *B(φ)* gradients, and the synchronization-holonomy sector that distinguishes conformal transport (exact one-form *d ln A*) from genuinely non-integrable disformal transport. The narrative continuity is therefore explicit: Paper 23 establishes what the KG–Dirac sector of relativistic quantum theory recovers when both temporal shear and observable disformal response are suppressed; Paper 25 establishes how interactions, measurement, and interferometric observables behave when they are not.

Exact tensor algebra. The derivation proceeds by exact tensor algebra. The individual steps — tetrad collapse to the identity, vanishing of the spin connection, and reduction of the covariant derivative to the ordinary partial derivative — are the standard flat-space limit of the Fock–Ivanenko formalism in curved-spacetime quantum mechanics. Every term in the geometric operator is tracked through the flattening limit; no approximation is made. The standard Dirac operator is the exact local Clifford/tetrad representation in the isochronous background. This tensor-algebraic derivation has been verified symbolically using SymPy; see `results/sympy_audit.log`.

### 3.3 The Screened Limit

The flattening conditions correspond to the screened limit, where the observable shear sector *Σ_{μ}^{obs} → 0* with *A(φ)* normalized to its ambient environmental value (Paper 0, §7). In this regime:

- *A(φ) → 1* (conformal factor approaches unity)

- *Σ_{μ} = ∇_{μ} ln A(φ) → 0* (temporal shear vanishes)

- *B(φ)(∇φ)² → 0* (observable disformal response suppressed)

The standard Dirac equation is thus the *screened limiting case* of the geometric operator, valid in the regime where the geometric structure of the temporal field is negligible. For collider processes the relevant statement is kinematic: the scalar field's configuration is set by boundary conditions over its correlation length (*L_{c} ≈ 4 × 10^{3} km* in the terrestrial calibration of Paper 6), so across the ~10^{−18} m, ~10^{−27} s extent of a TeV-scale interaction the field is frozen and locally constant — *Σ_{μ} ≈ 0* and *B(φ)(∇φ)^{2} ≈ 0* over the entire interaction region. A frozen, locally constant field is precisely the tangent-space limit derived above. The KG–Dirac and perturbative-QFT descriptions therefore remain exact for currently accessible particle-physics experiments, while long-baseline, long-integration interferometry — which accumulates phase over scales comparable to field gradients — is the probe class that can resolve the unscreened structure.

The operator collapse of the curved-space Dirac equation into the flat-space Dirac limit serves as the quantum-geometric realization of the abstract environmental operator $\mathcal{S}_\Sigma(\mathcal{E})$. The saturation of the Temporal Topology at the Compton scale $r_c$ acts here as the physical proxy for $\mathcal{S}_\Sigma(\mathcal{E})$, yielding the standard flat-space behavior without committing to specific chameleon or symmetron microphysics.

### 3.4 Geometric Reinterpretation of Spinor Structure

The KG–Dirac tangent-limit derivation shows that the standard Dirac equation is recovered within the TEP framework as the screened limit. The TEP framework further provides a geometric reinterpretation of the spinor structure that Dirac introduced in 1928.

The algebraic structure of the Dirac equation — the Clifford algebra, the spinor representation, the charge conjugation and parity operations — all emerge from the tetrad structure. When the tetrad is trivial (flat background), these operations appear as abstract algebraic symmetries. When the tetrad is non-trivial (curved temporal background), they are revealed as geometric orientation operations on the proper-time manifold.

In the TEP framework, the spinor is proposed as a mathematical encoding of temporal-orientation holonomy. The Clifford algebra is proposed as the local algebra of frame rotations in the temporal orientation bundle. The gamma matrices are proposed as the generators of infinitesimal rotations in the proper-time phase frame. Within this paper the proposal is supported at the abelian level (Appendix A.5); the non-abelian completion is developed in TEP-SPIN (Paper 24). This reinterpretation follows from the geometric structure of the temporal manifold and reveals the physical origin of the algebraic structure that Dirac discovered.

The Dirac equation is thus recovered within the TEP framework as the screened tangent-space limit when temporal shear and disformal coupling are negligible. This characterizes the standard theory as the tangent limit of a deeper geometric structure.

## 4. Spin and Antiparticles as Geometric Orientations

### 4.1 Spin-1/2 as Temporal-Orientation Holonomy

In the TEP framework, spin-1/2 arises as temporal-orientation holonomy of topological charge defects in the temporal field. In standard quantum mechanics, spin-1/2 is described by an abstract SU(2) representation without reference to spatial extent. The SU(2) spinor encodes angular momentum *ℏ/2* algebraically, not as a physical rotation in space.

In the TEP framework, spin is reinterpreted as temporal-orientation holonomy. A fermion is modeled as a localized defect in the temporal orientation bundle, with two sectors distinguished. The scalar amplitude sector: the core is a region of saturated conformal response, with *ln A(φ)* smooth and single-valued, so that *Σ_{μ} = ∇_{μ} ln A* remains an exact one-form and *∮ Σ_{μ} dx^{μ} = 0* around every loop — in exact agreement with the conformal-exactness theorem of Paper 0 (§3.3). The conformal sector carries no holonomy, and the causal metric *g̃_{μν} = A^{2}(φ) g_{μν}* is single-valued everywhere. The orientation sector: the defect regime is postulated to carry a circle-valued orientation phase on the temporal orientation bundle — additional structure beyond the single real scalar *φ*, in the same sense that the conjugate orientation bundle of §4.3 is postulated structure. Its winding around the defect core is the topological invariant identified with spin. The local temporal shear orients the defect relative to the matter-clock congruence but does not itself carry the winding. This division of labour places the holonomy in exactly the sector Paper 0 (§3.3) reserves for non-exact transport structure beyond the conformal one-form.

Singularity regulation. The defect core, where the conformal response saturates, is adopted here as a strict topological point defect for holonomy purposes: the postulated orientation phase winds around the topological core, and Stokes' theorem is inapplicable precisely because the orientation connection is not globally exact. The precise codimension of the idealized singularity in the orientation bundle is not fixed by the single scalar field *φ* alone; the physically relevant scale is the Compton-radius smearing discussed below. This idealization parallels the Dirac-string construction for a magnetic monopole or the *1/r* vortex of a superfluid; it is not regulated by perturbative renormalization of a zero-dimensional source. Physically, the fermion carries finite Compton-scale extent *r_{c} = ℏ/(mc)*; the scalar field equation with potential *V(φ)* saturates the conformal response at the core, smearing the curvature of the orientation connection, *d&thetasym;*, over this radius rather than admitting a bare ultraviolet divergence. A loop penetrating this smeared core (*r < r_{c}*) therefore encounters redistributed curvature rather than a point singularity, yet the enclosed holonomy does not degrade continuously: the integer winding *n = ±1* in the orientation bundle is a topological invariant, and any closed contour encircling the defect yields the same orientation holonomy *∮ &vartheta;_{μ} dx^{μ} = ±4π* regardless of how deeply it penetrates the core. Observable spin and transport are computed in the exterior regime *r &gg; r_{c}*, where the *1/r* shear profile applies; the interior regularization is derived in TEP-SPIN (Paper 24).

The "spin-1/2" property follows from the half-integer coupling of the matter-clock phase to the orientation sector. The postulated connection *&vartheta;_{μ}* carries winding *∮ &vartheta;_{μ} dx^{μ} = n · 4π*; the phase couples with weight *ℏ/2*, so the minimal winding *n = ±1* yields *ΔS/ℏ = ±2π*, preserving single-valuedness of *Ψ = exp(iS/ℏ)*. A *2π* spatial rotation of the defect changes the orientation phase by *±π*, producing the characteristic spinor sign flip *Ψ → −Ψ*. The *4π* periodicity is therefore a consequence of the half-integer weight, not an additional postulate. A full recovery of fermionic exchange antisymmetry requires the multi-particle configuration-space treatment developed in TEP-SPIN. The two "spins" are the two possible orientations of the topological charge relative to the local matter-clock congruence:

- *Spin up:* The topological charge rotates in the same sense as the local temporal shear circulation.

- *Spin down:* The topological charge rotates in the opposite sense.

On this construction, *SU(2)* is proposed as the holonomy group of the orientation bundle — postulated structure beyond the scalar sector — with the Pauli matrices *σ_{i}* as its generators. The *U(1)* winding exhibited in Appendix A.5 is the abelian core of that proposal; the non-abelian completion is developed in TEP-SPIN (Paper 24).

### 4.2 The g-Factor as Geometric Ratio

The anomalous magnetic moment *g − 2* arises in QED from loop corrections involving virtual photons. In the TEP framework, the g-factor is a geometric ratio that measures the deviation from the flat-frame approximation, and the QED loop expansion is interpreted as the screened-limit representation of a deeper geometric correction associated with topological charge structure. Far from the charge core, where the orientation-sector curvature becomes negligible, the geometric correction vanishes and *g → 2*, recovering the Dirac value. The geometric contribution to the g-factor is addressed in TEP-SPIN (Paper 24).

### 4.3 Antiparticles as Reversed Proper-Time Orientation

In standard quantum mechanics, antimatter is introduced as a separate field of particles with opposite charge. The Dirac equation's negative-energy solutions are reinterpreted via the Feynman-Stückelberg interpretation as positive-energy antiparticles moving backward in coordinate time.

In the TEP framework, antimatter is not an independent ontological species, nor does it require a macroscopic 'second sheet' of the universe. It is interpreted as reversed proper-time phase orientation localized strictly within the internal phase space (the temporal orientation bundle) of the topological defect.

The macroscopic spacetime manifold remains a single, continuous, orientable Lorentzian manifold. The 'double cover' required to support opposing temporal orientations exists entirely within the temporal-orientation bundle of the scalar field. Where a particle's phase advances as *+dτ̃* aligned with the ambient temporal shear, its antiparticle advances as *−dτ̃* with an inverted phase relative to the ambient shear.

The Decoupling Vertex and Topological Conservation: At the pair production vertex, the spacetime manifold does not tear or branch. The electromagnetic coupling furnishes the energy budget for a null phase contour to resolve into two orientation-conjugate timelike excitations: a forward-oriented matter-clock mode and a reverse-oriented charge-conjugate mode. In screened laboratory language these appear as particle and antiparticle; in TEP ontology they are opposite phase orientations of the same continuous proper-time geometry.

Laboratory realisation: pair production. The abstract seam *J_{br}* admits a concrete laboratory identification in photon-induced pair production, *γ + Z → e^{−} + e^{+} + Z*, where a high-energy photon converts in the Coulomb field of a nucleus *Z*. The process is traced here in laboratory coordinates *(t, x)*; the orientation assignment is a property of the temporal phase bundle, not of the spatial direction of motion in the lab frame.

*(i) Pre-junction approach.* Before the vertex, the photon propagates on the future null cone with *k_{μ} k^{μ} = 0* in *g̃*. Its phase action *S* advances along the null generator; no timelike proper-time oscillator is yet present because the photon is massless. The covector *k_{μ} = ∂_{μ}S* is future-directed and null, placing the photon on the branch locus *J_{br}* rather than on either orientation branch.

*(ii) Junction event.* At the interaction vertex—the spacetime event at which the electromagnetic coupling supplies sufficient energy–momentum budget to populate the electron mass shell—the local covector remains null on the photon mass shell but is no longer globally continuable on a single orientation branch: the connected phase contour must decouple to support two timelike, on-shell continuations. This vertex is a point of *J_{br}*. It is localised to a single laboratory event (a point in *(t, x)*), not extended along a spatial hypersurface; detectors record the bifurcation only through the emergence of two timelike tracks downstream of that event.

*(iii) Bifurcation into orientation-conjugate excitations.* From the connected contour on the decoupling vertex emerge two branches. The electron continues with timelike *k^{μ}*, accumulating phase as *+dτ̃* along a future-directed worldline aligned with the ambient temporal shear. The positron continues with reversed proper-time orientation within the temporal-orientation bundle, *dτ̃ → −dτ̃* relative to the matter-clock congruence. Both worldlines advance forward in coordinate time *t*; the distinction is geometric, not a reversal of laboratory causality. The Feynman–Stückelberg reinterpretation is recovered as the screened-laboratory projection of a past-directed proper-time orientation: the positron appears in detectors as a particle of opposite charge propagating forward in *t*, while its phase transport is reversed strictly within its internal phase frame. In the screened limit, all standard QED observables—track curvature in a magnetic field, bremsstrahlung spectra, coincidence timing—are recovered; the TEP content is the geometric identification of the creation vertex as a topological decoupling event.

*(iv) Post-creation separation and entanglement.* As the pair separates, each branch carries a localized topological charge (Section 4.1) on its respective orientation branch. The phase contour rooted at the *J_{br}* event remains a single connected geometric object: TEP-KIN (Paper 25) identifies this as the macroscopic entanglement link between the bifurcated charges. The junction event is therefore not only the orientation decoupling but also the origin of the shared orientation-bundle holonomy that enforces quantum correlations between the outgoing tracks.

*(v) Annihilation as recombination.* The reverse process, *e^{−} + e^{+} → γ + γ*, is recombination on *J_{br}*: timelike contours from the matter and antimatter orientation branches converge at a common null event, the shared phase contour closes, and a single future-directed null continuation emerges. Creation and annihilation are time-reversed navigations of the same topological seam.

Textual spacetime schematic. The same navigation can be read directly in the laboratory *(t, x)* plane, with *x* along the photon beam axis and *t* increasing upward. For *t < t_{V}*, the incident photon follows a null generator at slope *dt/dx = ±1/c* (future light cone); its phase contour lies on the branch locus *J_{br}*. At the vertex *V = (t_{V}, x_{V})*, that null generator meets *J_{br}*—a single event, not a spatial line or wall. For *t > t_{V}*, two timelike worldlines branch from *V*, each with *|dx/dt| < c* and both advancing in coordinate time. The electron worldline *γ_{e^{−}}* carries forward orientation *dτ̃ > 0*; the positron worldline *γ_{e^{+}}* carries reversed orientation *dτ̃ < 0* in the matter-clock congruence. The two branches may separate symmetrically in *x* (for example, emission at equal angles about the beam axis) while remaining future-directed in *t*. A track photograph or time-resolved coincidence measurement records only these timelike curves in *(t, x)*; the orientation labels are not imprinted on the tracks but are fixed by the sign of phase accumulation along each worldline. Annihilation is the time-reverse reading of the same diagram: *γ_{e^{−}}* and *γ_{e^{+}}* converge at a vertex *V′* on *J_{br}*, and the outgoing photon again follows a null generator from *V′*. The junction is thereby visible only as a fork or merge in the causal structure of recorded worldlines, localised to the vertex events *V* and *V′*.

Crucially, CPT symmetry is *preserved locally*. The local CPT theorem remains valid at every point in spacetime: a full rotation in the local orientation bundle (C × P × T) returns the system to its original state. Charge conjugation C is realised as a local reflection in the orientation bundle (reversing the direction of proper-time phase accumulation), not as a global spacetime transition. A particle and its antiparticle correspond to opposite orientations within the temporal-orientation bundle; this separation is a consequence of the internal phase geometry, not of the local C operation itself. This construction is not a substitute for the screened QED calculation of pair-production amplitudes; it supplies the proposed TEP geometric interpretation of the same laboratory process. This suggests a possible topological origin for the observed matter-antimatter asymmetry: rather than requiring a symmetry-breaking event in the early universe, the dominance of matter may reflect an orientation bias in the cosmological temporal-orientation bundle, favouring one proper-time phase orientation over its conjugate in certain density regimes. A quantitative prediction of the baryon-to-photon ratio from this topological bias, evaluated via MCMC parameter estimation, is developed in TEP-C0 (Paper 26, Athens).

### 4.4 Unification of C, P, and T

Charge conjugation (C), parity (P), and time reversal (T) are unified as orientation operations in the temporal-orientation bundle:

- *C (Charge conjugation):* Local reflection in the orientation bundle — maps a matter-clock excitation to its charge-conjugate orientation mode by reversing proper-time phase orientation into the orientation-conjugate branch.

- *P (Parity):* Spatial reflection — reverses the handedness of the topological charge in the temporal landscape.

- *T (Time reversal):* Phase reversal — reverses the direction of proper-time accumulation, mapping future-directed phase transport to past-directed. It belongs to the same orientation-bundle sector as charge conjugation but is a distinct operation in the product group C × P × T.

The CPT theorem, a central result in quantum field theory, is reinterpreted geometrically: a full rotation in the orientation bundle (C × P × T) returns the system to its original state. In the standard framework, the theorem is derived rigorously from Lorentz invariance, locality, and the spin-statistics connection (Lüders-Pauli, Jost). The TEP framework preserves the local content of this derivation and proposes a geometric origin: the theorem is manifest as the orientability of the temporal manifold.

### 4.5 Experimental Implications

The TEP framework makes concrete, falsifiable predictions that distinguish it from standard fixed-background quantum theory. In the unscreened regime, where the scalar field varies appreciably over the probe baseline so that the observable shear sector *Σ_{μ}^{obs}* is non-negligible, the conformal factor *A(φ)* deviates from unity. High-energy particle collisions are deep in the screened limit because the scalar field is frozen and locally constant over the interaction scale (Paper 0, §7); long-baseline, long-integration atomic interferometry, which accumulates phase over scales comparable to field gradients, is the probe class that can resolve the unscreened structure.

Order-of-magnitude estimate: For a conformal factor *A(φ) = exp(β_{A} φ / M_{Pl})*, the fractional mass shift is parametrically *δm/m ~ (1 − A) &sim; β_{A} φ / M_{Pl}*. In a realistic model the scalar field acquires a non-zero vacuum expectation value *φ_{0}* from cosmological boundary conditions or apparatus geometry; local matter density then screens this background value, with the unscreened field in ultra-high vacuum approaching *φ_{0}* rather than scaling with the ambient density *ρ*. The linear estimate *φ / M_{Pl} ~ ρ / ρ_{T}*, used here for orientation only, assumes no independent potential for *φ* and should be understood as a lower bound: if *φ_{0}* is set by cosmology at a larger value, the signal is proportionally larger. With the conservative linear estimate and the coupling bounded by the corpus's own unscreened-regime constraint, *β_{A} &lsim; 3 × 10^{−3}* (Cassini PPN–γ; Paper 0, §7), the predicted fractional mass shift is *δm/m &lsim; 1.5 × 10^{−19}*, corresponding to *δΦ &lsim; 1.5 × 10^{−11} rad* in the 10 m rubidium configuration. At current sensitivities of *10^{−4}–10^{−5} rad / √Hz* this is not detectable on realistic integration times. The linear estimate therefore functions as a falsifiability floor, not a near-term detection forecast. Two routes raise the signal within the framework: a cosmologically set background value *φ_{0}* exceeding the linear estimate, which must be computed from the profile solution Paper 6 (§5.4) identifies as an open target; or channel-specific response *κ_{X}* exceeding the PPN channel's, which would have to be derived, not assumed (Paper 0, §7). Both routes are bounded above by existing clock-network constraints. The near-term experimental leverage of the framework lies instead in the closed-loop synchronization holonomy *H_{resid}* (Paper 0, §3.3, §10; TEP-KIN), an observable the conformal mass-shift channel cannot mimic and which standard fixed-background quantum theory predicts to be exactly zero; it is sourced by the non-exact disformal transport sector and measured via closed-loop clock-network protocols (Paper 0, §10; TEP-KIN).

This estimate is phenomenological rather than a full interferometer phase calculation; the precise observable depends on pulse sequence, common-mode cancellation, and whether the temporal-shear contribution enters propagation, laser phase, or both. The companion paper TEP-KIN (Paper 25) develops specific experimental protocols for laboratory interferometry and has provided empirical support via graphene Aharonov-Bohm data. TEP-SPIN (Paper 24) addresses subatomic structure and spin-dependent measurements.

### 4.6 The Spinor: A Historical Reinterpretation

In 1928, Dirac derived the spinor as the mathematical object required to linearize the Klein-Gordon equation. The spinor was introduced as a four-component complex vector that transforms under the Lorentz group via a double-valued representation. Dirac showed that the spinor "internal space" was necessary to accommodate both positive- and negative-energy solutions while preserving Lorentz covariance. The algebraic machinery — Clifford algebra, gamma matrices, charge conjugation — was taken as fundamental.

The TEP framework offers a geometric reinterpretation. Dirac was attempting to describe physical temporal-orientation holonomy without access to a dynamical proper-time geometry. The "internal space" of the spinor is understood as an encoding of geometric orientation data (the winding of the orientation phase, oriented by the local temporal shear) into an algebraic object defined on a flat, isochronous background.

The four components of the Dirac spinor correspond to:

- *Upper two components:* Particle with spin up/down relative to the winding of the orientation phase, oriented by the local temporal shear.

- *Lower two components:* Antiparticle with reversed proper-time phase orientation (the "negative-energy" solutions).

In the TEP framework, these are proposed as the four possible combinations of (topological charge orientation) × (phase direction) within the temporal-orientation bundle. The Clifford algebra is proposed as the local algebra of frame rotations in the temporal orientation bundle, not an abstract symmetry. The gamma matrices are proposed as the generators of infinitesimal rotations in the proper-time phase frame, not fundamental operators. Dirac's spinor was an effective mathematical workaround for a geometric structure that physics had not yet developed.

This reinterpretation is not a dismissal of Dirac's work; it is an elevation. The Dirac equation remains an accurate description of relativistic quantum mechanics in the screened limit. But its algebraic structure, which appeared fundamental in 1928, is now revealed as the natural geometric language of a deeper temporal topology.

## 5. Conclusion

This paper derives three foundational structures of relativistic quantum theory and geometrically reinterprets two others, all as the screened tangent-space limit of the Temporal Equivalence Principle:

- *The phase action* *S = −mc^{2} ∫ dτ̃* is the primitive geometric driver, with mass appearing as the parameter governing the oscillator frequency *ω_{0} = mc^{2} / ℏ*, modulated by the conformal factor in the causal matter metric *g̃_{μν}*.

- *The Klein-Gordon equation* *(□̃ + m^{2}c^{2} / ℏ^{2}) Ψ = 0* is derived from the minimal geometric Lagrangian in the causal metric and verified via WKB / eikonal expansion; its eikonal limit recovers the *g̃*-Hamilton-Jacobi equation, not via operator substitution.

- *The Dirac operator* *iγ^{μ}∂_{μ} − mc / ℏ* is recovered as the local Clifford/tetrad representation in the isochronous background. It emerges as the limiting case of the geometric operator when temporal shear *Σ_{μ}* and disformal coupling *B(φ)* are negligible.

- *Spin and antiparticle structure* are proposed to admit geometric reinterpretations as orientation structures in the internal temporal-orientation bundle, rather than being treated as primitive internal quantum properties. Antiparticles are interpreted as reversed proper-time phase orientation in the orientation-conjugate branch; CPT is preserved locally while matter and antimatter correspond to opposite phase orientations of the same continuous proper-time geometry.

- *The spinor* is reinterpreted geometrically: Dirac's 1928 spinor encoded temporal-orientation holonomy without access to a dynamical proper-time geometry.

These results provide the quantum-foundation layer for the full TEP framework. All tensor-algebraic derivations have been verified symbolically using SymPy; the audit log is available in `results/sympy_audit.log`. The companion papers develop the implications for subatomic structure (TEP-SPIN), interaction kinematics (TEP-KIN), and cosmological synthesis (TEP-C0).

The standard relativistic quantum framework is recovered as the screened limit of the Temporal Equivalence Principle. In the screened limit, where the observable shear sector *Σ_{μ}^{obs} → 0* with *A(φ)* normalized to its ambient environmental value, the conformal factor approaches unity and the temporal shear is suppressed, reproducing the familiar Minkowski background. In the unscreened regime, the full geometric structure is manifest, and new phenomena become accessible. A non-zero disformal response *B(φ)* introduces direction-dependent causal corrections — tilting the effective light cone and producing non-isotropic scattering — that vanish in the screened limit but become critical for interaction routing and interferometric observables. The broader conceptual implications — geometric entanglement, temporal-topology drag, and the geometric reinterpretation of Bell non-separability — are introduced in Section 1.5 and developed in the companion papers TEP-SPIN and TEP-KIN. In this restricted but precise sense, the result restores a recognisably Einsteinian ambition to the quantum sector: the algebraic structures of relativistic quantum mechanics are not discarded, but recovered as limiting expressions of an underlying continuous geometry.

## References

- Smawfield, M. L. (2025). *Temporal Equivalence Principle: Dynamic Time & Emergent Light Speed*. Preprint v0.9 (Jakarta). Zenodo. DOI: 10.5281/zenodo.16921911 (Paper 0)

- Smawfield, M. L. (2025). *Temporal Topology Saturation Scale: Cross-Scale Consistency of ρ_{T}*. Preprint v0.6 (New Delhi). Zenodo. DOI: 10.5281/zenodo.18064365 (Paper 6)

- Smawfield, M. L. (2026). *Temporal Equivalence Principle: Disformal Kinematics and the Measurement Landscape*. Preprint v0.1 (Kuala Lumpur). Zenodo (Paper 25)

- Smawfield, M. L. (2026). *Temporal Equivalence Principle: A Topological Fermion Model for Spin and the g−2 Anomaly*. Preprint v0.1 (Paris). Zenodo (Paper 24)

- Smawfield, M. L. (2026). *Temporal Equivalence Principle: A Covariant Alternative to Cosmic Expansion*. Preprint v0.1 (Athens). Zenodo. DOI: 10.5281/zenodo.20370144 (Paper 26)

- Einstein, A. (1949). *Autobiographical Notes*. In P. A. Schilpp (Ed.), *Albert Einstein: Philosopher-Scientist*. Open Court.

- Einstein, A. (1956). *The Meaning of Relativity*. 5th ed. Princeton University Press.

- Dirac, P. A. M. (1928). The quantum theory of the electron. *Proc. R. Soc. A* 117(778), 610–624.

- Klein, O. (1926). Quantentheorie und fünfdimensionale Relativitätstheorie. *Z. Phys.* 37, 895–906.

- Gordon, W. (1926). Der Comptoneffekt nach der Schrödingerschen Theorie. *Z. Phys.* 40, 117–133.

- Bohm, D. (1952). A suggested interpretation of the quantum theory in terms of "hidden" variables. *Phys. Rev.* 85, 166–179.

- Stückelberg, E. C. G. (1941). Relativistic invariance in the interaction of particles. *Helv. Phys. Acta* 14, 372–383.

- Feynman, R. P. (1949). The theory of positrons. *Phys. Rev.* 76(6), 749–759.

- Penrose, R. & Rindler, W. (1984). *Spinors and Space-Time*. Vol. 1. Cambridge University Press. §6.7.

- Fock, V. & Ivanenko, D. (1929). Géométrie quantique linéaire et déplacement parallèle. *Comptes Rendus de l'Académie des Sciences*, 188, 1470–1472.

- Birrell, N. D. & Davies, P. C. W. (1982). *Quantum Fields in Curved Space*. Cambridge University Press.

- Parker, L. & Toms, D. (2009). *Quantum Field Theory in Curved Spacetime*. Cambridge University Press.

- Wald, R. M. (1994). *Quantum Field Theory in Curved Spacetime and Black Hole Thermodynamics*. University of Chicago Press.

- Nakahara, M. (2003). *Geometry, Topology and Physics*. 2nd ed. Institute of Physics Publishing.

- Frankel, T. (2011). *The Geometry of Physics*. 3rd ed. Cambridge University Press.

- Lüders, G. (1954). On the equivalence of invariance under time reversal and under particle-antiparticle conjugation. *Kgl. Danske Videnskab. Selskab. Mat.-Fys. Medd.* 28(5), 1–17; Pauli, W. (1955). Exclusion principle and quantum mechanics. *Nobel Lecture*; Jost, R. (1957). Eine Bemerkung zum CPT-Theorem. *Helv. Phys. Acta* 30, 409–416.

- Streater, R. F. & Wightman, A. S. (1964). *PCT, Spin and Statistics, and All That*. Benjamin/Cummings.

- Weinberg, S. (1995). *The Quantum Theory of Fields*. Vol. 1. Cambridge University Press. §2.1–2.7.

## Appendix A: Detailed Derivations

### A.1 Tetrad Formalism in Curved Spacetime

In curved spacetime, the Dirac spinor cannot be defined directly on the manifold. A local orthonormal frame (tetrad) *e^{a}_{μ}* is required at each point, satisfying:

g_{μν} = e^{a}_{μ} e^{b}_{ν} η_{ab}

where *η_{ab}* is the Minkowski metric. The inverse tetrad *e_{a}^{μ}* satisfies *e^{a}_{μ} e_{a}^{ν} = δ^{ν}_{μ}*. The Dirac matrices in curved spacetime are defined as:

γ^{μ} = e_{a}^{μ} γ^{a}

where *γ^{a}* are the constant flat-space Dirac matrices satisfying the Clifford algebra *{γ^{a}, γ^{b}} = 2η^{ab}*.

### A.2 Spin Connection

The spin-covariant derivative incorporates the spin connection *ω^{ab}_{μ}*:

∇_{μ} = ∂_{μ} + (1/4) ω^{ab}_{μ} σ_{ab}

where *σ_{ab} = (1/2)[γ_{a}, γ_{b}]* are the Lorentz generators. The spin connection is related to the tetrad by:

ω^{ab}_{μ} = e^{a}_{ν} ∇_{μ} e_{b}^{ν}

In the flat-frame limit where the tetrad becomes the identity (*e^{a}_{μ} → δ^{a}_{μ}*), the spin connection vanishes and the covariant derivative reduces to the ordinary partial derivative.

### A.3 Conformal Transformation of the Dirac Operator

Under the conformal transformation *g̃_{μν} = A^{2}(φ) g_{μν}*, the tetrad rescales as *e^{a}_{μ} → A(φ) e^{a}_{μ}*. The Dirac operator therefore acquires both a direct tetrad rescaling and additional terms from the modified spin connection. The explicit transformation is non-trivial because the spin connection depends on derivatives of *A(φ)*; see e.g. Penrose & Rindler [14]. When *A(φ) = 1* (screened limit, where the observable shear sector *Σ_{μ}^{obs} → 0* with *A(φ)* normalized to its ambient environmental value), the conformal transformation is trivial and the standard Dirac operator is recovered.

The full tetrad algebra factorisation — from the geometric Dirac operator *(i e^{μ}_{a} γ^{a} ∇_{μ} − m) ψ = 0* to the standard flat-space form *(iγ^{μ}∂_{μ} − m)ψ = 0* under the flattening conditions *Σ_{μ} = 0*, *B(φ)(∇φ)^{2} → 0* — has been verified symbolically using SymPy. The audit log documenting each algebraic step (Clifford algebra verification, tetrad identity, spin connection vanishing, operator collapse, and disformal signpost) is available in `results/sympy_audit.log`. The corresponding derivation scripts are in `scripts/derivations/`.

### A.4 Disformal Departure from Conformal Collapse

On the full Jakarta ansatz *g̃_{μν} = A^{2}(φ) g_{μν} + B(φ) ∇_{μ}φ ∇_{ν}φ*, the conformal sector alone is tetrad-rescalable. The disformal term is rank-one and generates off-diagonal matter-frame components *g̃_{μν} = B(φ) ∇_{μ}φ ∇_{ν}φ* whenever *∇_{μ}φ ∇_{ν}φ* has support in both temporal and spatial directions. Identity tetrad collapse therefore requires suppression of the observable disformal response, not merely vanishing temporal shear. SymPy Audit A.4 verifies this obstruction at the metric level; the complete disformal kinematics audit (inverse metric, *n*_{eff}, Christoffel symbols, holonomy sector) is implemented in TEP-KIN (Paper 25), which carries the full *g̃_{μν}* through interaction routing and measurement while this paper establishes the screened tangent limit recovered when *B(φ)(∇φ)^{2} → 0*.

### A.5 Holonomy of the Orientation Bundle

The temporal field carries an orientation bundle with structure group *O(1,3)*. Away from topological defects, the temporal shear *Σ_{μ} = ∇_{μ} ln A(φ)* is a smooth gradient, and by Stokes' theorem:

∮_{C} Σ_{μ} dx^{μ} = ∮_{C} d(ln A) = 0

for any contractible loop in a simply connected region where *A(φ)* is smooth and single-valued, as established in the original TEP theory, which assumes the exponential form *A(φ) = exp(β_{A} φ / M_{Pl})* everywhere. The conformal sector therefore carries no holonomy; the metric *g̃_{μν} = A^{2}(φ) g_{μν}* is single-valued everywhere.

The holonomy resides in the postulated orientation sector. In the defect regime the orientation bundle carries a connection one-form *&vartheta;_{μ}* with curvature concentrated at the core; around any loop encircling the core,

∮_{C} &vartheta;_{μ} dx^{μ} = n · 4π, n ∈ ℤ

The matter-clock phase couples to this sector with half-integer weight, *dS &sup; (ℏ/2) &vartheta;_{μ} dx^{μ}*, so the minimal winding *n = ±1* yields *ΔS/ℏ = ±2π* — preserving single-valuedness of *Ψ = exp(iS/ℏ)* — while a *2π* spatial rotation of the defect yields *ΔS/ℏ = ±π*, the sign flip characteristic of spinors. The half-integer weight is postulated here as the minimal consistent charge; its derivation, and the non-abelian *SU(2)* structure of the full spinor representation, require the multi-defect treatment of TEP-SPIN (Paper 24). The single scalar *φ* generates neither the winding nor the non-abelian structure; both belong to the postulated orientation sector.

### A.6 SymPy Audit Environment

The symbolic derivations were executed in a reproducible Python environment with the following pinned dependencies: Python 3.11, SymPy 1.12, NumPy 1.26, and SciPy 1.11. The primary derivation scripts — `derive_klein_gordon.py`, `derive_dirac_limit.py`, `derive_spin_holonomy.py`, and `sympy_audit.py` — are located in `scripts/derivations/`. The master audit verifies: (i) Clifford algebra closure for the rescaled Dirac matrices; (ii) tetrad identity *g_{μν} = e^{a}_{μ} e^{b}_{ν} η_{ab}* under conformal rescaling; (iii) spin-connection vanishing in the flat-frame limit *Σ_{μ} = 0*; (iv) operator collapse from the curved-space Dirac operator to the standard flat-space form; and (v) the disformal signpost audit (rank-one obstruction to conformal collapse). To reproduce the audit independently, run `python scripts/run_all.py --audit` from the repository root; the master audit writes the step-by-step log to `results/sympy_audit.log`. The full disformal tensor audit is in TEP-KIN (Paper 25).


## 6. Data Availability & Reproducibility

This work follows open-science practices. All theoretical derivations, symbolic checks, and reproducibility materials are fully reproducible using the documented code.


### Repository and Code

GitHub Repository: github.com/matthewsmawfield/TEP-QF

The repository contains the analytical derivations and symbolic verification scripts for the TEP quantum-field framework, conformal Dirac-operator derivation, and spin–antimatter coupling.


### Repository Structure



```
TEP-QF/
├── scripts/
│   └── derivations/
│       ├── derive_klein_gordon.py
│       ├── derive_dirac_limit.py
│       ├── derive_spin_holonomy.py
│       └── sympy_audit.py
├── results/
│   └── sympy_audit.log
├── requirements.txt
├── CITATION.bib
└── README.md
```


### Software Environment

Key packages: NumPy, SciPy, SymPy, Matplotlib. The scripts have been tested on Python 3.10+.


### License

All code and manuscripts are released under CC-BY-4.0.