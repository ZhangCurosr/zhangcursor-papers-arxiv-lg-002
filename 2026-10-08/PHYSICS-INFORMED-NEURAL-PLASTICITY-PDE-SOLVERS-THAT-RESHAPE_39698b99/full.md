# PHYSICS-INFORMED NEURAL PLASTICITY: PDE SOLVERS THAT RESHAPE THEMSELVES

Chun-Wun Cheng<sup>∗</sup> University of Cambridge cwc56@cam.ac.uk

Bingcheng Hu<sup>∗</sup> Shanghai Normal University 1000534759@smail.shnu.edu.cn

Angelica I. Aviles-Rivero<sup>†</sup> YMSC, Tsinghua University aviles-rivero@tsinghua.edu.cn

## ABSTRACT

Physics-informed neural PDE solvers adapt their parameters to satisfy governing equations, yet their representational structure typically remains fixed throughout training. This rigidity is poorly matched to PDE solutions with strongly heterogeneous complexity across space and space–time, leaving capacity insufficient where the physics is difficult and redundant where it is simple. We introduce physics-informed neural plasticity, a paradigm in which the representation itself reshapes during optimization in response to unresolved physics. We instantiate this principle with Representation Capacity Adaptation for PDEs (ReCAP), a Gaussian-localized solver that dynamically redistributes capacity through local enrichment, residual-directed splitting, gate-based pruning, and function-aware merging. ReCAP uses responsibility-weighted error indicators and the geometry of residual energy to determine where and how to refine. To limit the disturbance introduced by splitting, we introduce quiet-child refinement, which initializes new components by transporting the parent representation while controlling instantaneous functional perturbation. We further establish conditional a posteriori reliability and structural-stability guarantees linking localized physics residuals to solution error and stable refinement. Across five challenging 3D and 4D PDE benchmarks against 11 physics-informed solvers, ReCAP achieves the lowest relative $L ^ { 2 }$ error on every problem, reducing error by 10.7%–27.5% relative to the strongest competing result. These results suggest that physics-informed solvers need not merely learn their parameters—they can learn how their representational capacity should be organized.

## 1 INTRODUCTION

Physics-informed neural networks (PINNs) approximate solutions to partial differential equations (PDEs) by optimizing differentiable function representations against governing equations and available initial or boundary data. Their mesh-free formulation and unified treatment of forward and inverse problems make them attractive when conventional discretization is difficult or solution data are scarce (Raissi et al., 2019; Karniadakis et al., 2021). Yet accurate training remains challenging for high-frequency, multiscale, and strongly nonlinear solutions, where spectral bias and optimization difficulties can leave localized structures unresolved even when the overall loss appears small (Rahaman et al., 2019; Wang et al., 2021; Krishnapriyan et al., 2021). This challenge is especially pronounced when solution complexity varies across space or space–time: smooth regions may coexist with oscillatory waves, localized peaks, or rapidly evolving fronts that demand markedly different approximation resources.

Yet most existing approaches share a key limitation: the parameters learn, but the representation itselflargely staysfixed. Optimization strategies, loss balancing, causal training, and residual-adaptive sampling can substantially improve PINN training (Wu et al., 2023; Daw et al., 2023; Wang et al., 2024a), but they generally operate within a representational budget chosen before training. Adaptive sampling changes where the PDE is enforced, but not what the model can represent locally. As a result, capacity may remain redundant in already-resolved regions while being insufficient where the physics is hardest to resolve.

We argue that physics-informed PDE solvers should instead exhibit neural plasticity: the ability to reshape their representational capacity during training in response to unresolved physics. Rather than optimizing a fixed representation, a plastic solver jointly learns the solution and reorganizes the capacity used to express it.

Localized neural representations provide a natural basis for such plasticity. FBPINNs decompose the domain among overlapping subnetworks (Moseley et al., 2023); PIG learns the locations and shapes of Gaussian features while fixing their number at initialization (Kang et al., 2025); and AB-PINNs introduce learnable RBF-defined subdomains with residual-driven addition of local subnetwork (Botvinick-Greenhouse et al., 2026).

These developments motivate a broader research question: how can physics-informed solvers coordinate where capacity is needed, what form it should take, and when it has become redundant, while limiting disruption to the learned solution?

To realize this principle, we introduce Representation Capacity Adaptation for PDEs (ReCAP), a Gaussian-localized physics-informed solver whose representation can enrich, split, merge, and prune itself during training. Each component carries a learned center and scale, a contribution gate, and a local polynomial approximation. Component responsibilities turn global physics errors into local adaptation signals, allowing ReCAP to redistribute capacity rather than simply increase it. Difficult regions receive higher local polynomial order or additional localized components, while weak or functionally redundant components are removed or consolidated.

These operations implement a common principle: adapt the local approximation space to unresolved physics through p-like enrichment, h-like refinement, and coarsening.

For such plasticity to be useful, structural changes must be both targeted and stable. ReCAP uses responsibility-weighted physics errors to identify unresolved components and a residual-weighted covariance to orient splitting along the dominant spread of unresolved residual energy. Merging is function-aware, requiring both effective overlap and agreement of local predictions rather than geometric proximity alone. Structural refinement can nevertheless perturb an already learned solution. We therefore introduce a quiet-child update: the trained parent is retained, its polynomial is translated into the child’s shifted and rescaled coordinates, and the child is introduced through a small gate. This creates new representational capacity while controlling its immediate effect on the learned function.

We complement ReCAP with theoretical analysis connecting local adaptation signals to global solution accuracy and characterizing the stability of structural refinement. Across challenging threeand four-dimensional PDE benchmarks, our experiments further show that dynamically redistributing representational capacity can outperform both fixed-capacity and competing physics-informed solvers. Together, these results support neural plasticity as a principled approach to adapting where and how capacity is deployed during physics-informed learning. Our contributions are:

• We formulate physics-informed neural plasticity, where representational capacity becomes an adaptive variable of PDE optimization rather than a fixed design choice, and introduce ReCAP to jointly adapt local approximation order, component geometry, and component count.

• We develop a responsibility-driven adaptation mechanism that couples local p-enrichment with residual-directed h-like splitting. Splitting is oriented by the leading eigenvector of a responsibility- and residual-weighted covariance, directing new capacity along the dominant spread of unresolved physics.

• We introduce function-aware coarsening and quiet-child refinement: merging requires both effective overlap and agreement of local predictions, while polynomial coordinate translation and gated child insertion create new capacity with controlled instantaneous perturbation of the learned solution.

• We establish conditional a posteriori reliability connecting mass-weighted local physics indicators to solution error under PDE stability and partition-of-unity responsibilities. For normalized mixtures with a linear transport operator, we further derive a gate-dependent bound on the instantaneous loss increase caused by additive quiet-child refinement, clarifying how capacity can grow with controlled disruption.

• Across five 3D/4D PDE benchmarks and 11 competing physics-informed solvers, ReCAP achieves the lowest relative $L ^ { 2 }$ error on every benchmark.

## 2 RELATED WORK

Physics-informed optimization and sampling. PINNs enforce differential equations through a training objective (Raissi et al., 2019; Karniadakis et al., 2021). Gradient imbalance, spectral bias, and difficult optimization landscapes motivate improved training schemes and architectures (Wang et al., 2021; Rahaman et al., 2019; Krishnapriyan et al., 2021; Cho et al., 2023; Wang et al., 2023; 2024a). Residual-adaptive sampling and retain-resample-release sampling direct equation evaluations toward poorly resolved regions (Wu et al., 2023; Daw et al., 2023). These methods alter training signals or their optimization; ReCAP additionally changes the available local basis. Sampling and representation adaptation are complementary, and the experiments examine their cumulative effect.

Local representations and adaptive decomposition. Radial basis functions provide trainable localized approximations (Broomhead & Lowe, 1988). Domain-decomposed PINNs, including XPINNs and FBPINNs, organize approximation over local regions (Jagtap & Karniadakis, 2020; Moseley et al., 2023). PIG combines trainable Gaussian features with a lightweight neural network (Kang et al., 2025), while AB-PINNs learn local subdomains and support residual-driven addition of new ones (Botvinick-Greenhouse et al., 2026). ReCAP builds on this local-representation perspective by coordinating component insertion and removal with local polynomial-order changes. Its splitting rule uses residual geometry, its merge criterion checks local functional agreement, and its child initialization accounts explicitly for the coordinate dependence of polynomial coefficients.

Adaptive approximation spaces. Classical hp methods allocate degrees of freedom by changing element size and approximation order (Babuska & Suri ˇ , 1994); hp-VPINNs bring domain decomposition and hp refinement into variational physics-informed learning (Kharazmi et al., 2021). ReCAP uses analogous operations on learned Gaussian components rather than a fixed mesh. This analogy motivates the algorithm but does not transfer finite-element convergence guarantees to it.

In particular, the residual scores used for adaptation are not local solution-error bounds: their reliability requires a stability estimate for the PDE and appropriate mass weighting, as made explicit in Section 3.4.

## 3 METHODOLOGY

ReCAP realizes physics-informed neural plasticity by adapting the approximation space together with its trainable parameters. At each stage, the structural state $\mathcal { S } = ( M , p _ { 1 } , \hdots , p _ { M } )$ specifies the number of Gaussian components and their active polynomial orders; the continuous parameters Θ specify their locations, scales, gates, and local approximations. Training alternates between optimizing Θ at fixed S and updating the representation using localized physics errors. This separates two complementary tasks: fitting the current approximation space and changing where that space has capacity.

The adaptation rules act at different levels. Polynomial enrichment increases local expressivity without adding components; splitting introduces a more localized component; pruning and merging reduce weakly active or redundant components. The base objective $\mathcal { L } _ { \mathrm { b a s e } }$ determines how the PDE and its conditions are enforced, whereas the adaptation scores determine which components are modified. These scores need not be additional loss terms. Figure 1 summarizes the loop. $\mathsf { A p - }$ pendix A details the component updates, and Appendix B contains the theoretical statements and proofs.

![](images/7715c3b68841d453c9f05e983bff55f80c05cff0b4c188b60d1bfcfcae49d806.jpg)  
Figure 1: Unlike conventional PINNs with fixed representational capacity, ReCAP reshapes its representation during training through local enrichment, splitting, merging, and pruning, allocating capacity where the PDE requires it.

## 3.1 REPRESENTATION AND LOCAL ERROR INDICATORS

On a spatial or spatiotemporal domain $\Omega \subset \mathbb { R } ^ { d }$ , consider $\mathcal { N } [ u ] ( z ) = 0$ with initial or boundary conditions $B [ u ] = g$ . Here d counts all input coordinates, including time when present, and $r _ { \Theta } =$ $\mathcal { N } [ u _ { \Theta } ]$ is the physics residual. For $\mathrm { F M 3 D } , z = ( t , x , y )$ and $r _ { \Theta } = \partial _ { t } u _ { \Theta } + a \partial _ { x } u _ { \Theta } + b \partial _ { y } u _ { \Theta }$ . The other benchmarks use their respective differential operators, given in Appendix C.

Each component $G _ { m } ~ = ~ ( \mu _ { m } , \sigma _ { m } , g _ { m } , c _ { m } , p _ { m } )$ has a center, positive diagonal scale, gate logit, polynomial coefficients, and active degree. Fixed input normalization is absorbed into z in the representation formulas; residual evaluation must retain the corresponding chain-rule factors. Define the complete total-degree index set $\mathcal { A } _ { p } = \left\{ \alpha \in \mathbb { N } _ { 0 } ^ { d } : | \alpha | \leq p \right\}$ and

$$
\xi _ { m } = \left( z - \mu _ { m } \right) \oslash \sigma _ { m } , \qquad \ell _ { m } ( z ) = \sum _ { \alpha \in A _ { p _ { m } } } c _ { m , \alpha } \xi _ { m } ^ { \alpha } , \qquad | A _ { p } | = { \binom { d + p } { p } } .\tag{1}
$$

Here $\begin{array} { r } { \xi ^ { \alpha } = \prod _ { q } \xi _ { q } ^ { \alpha _ { q } } } \end{array}$ and ⊘ denotes element-wise division. A local degree change therefore changes the number of active basis terms, independently of the number of Gaussians. The described threedimensional implementation preallocates the complete basis through total degree three, including cross terms, and masks inactive degrees.

Let $\pi _ { m } ( z ) \geq 0$ denote the effective responsibility used to assign evaluation points to component $m _ { : }$ and write the prediction as $u _ { \Theta } = \dot { \mathcal { A } _ { \Theta } ( \{ \pi _ { m } , \ell _ { m } , g _ { m } \} _ { m = 1 } ^ { M } ) }$ . The update rules use the full prediction and component responsibilities, together with local values or derivatives where required. When internal branches or channels are present, the adaptation statistics average the corresponding quantities across them. The analysis in Section 3.4 states separately when normalized responsibilities or a normalized gated-mixture output are required.

For residual points ${ \mathcal C } = \{ z _ { i } \} _ { i = 1 } ^ { N _ { c } }$ and value observations $\mathcal { D } _ { \Gamma } = \{ ( z _ { i } ^ { \Gamma } , u _ { i } ^ { \Gamma } ) \} _ { i = 1 } ^ { N _ { \Gamma } }$ , define

$$
\eta _ { m } ^ { \mathrm { r e s } } = \frac { \sum _ { i } \pi _ { m } ( z _ { i } ) | r _ { \Theta } ( z _ { i } ) | ^ { 2 } } { \sum _ { i } \pi _ { m } ( z _ { i } ) + \varepsilon } , \qquad \eta _ { m } ^ { \mathrm { b d r y } } = \frac { \sum _ { i } \pi _ { m } ( z _ { i } ^ { \mathrm {Gamma } } ) | u _ { \Theta } ( z _ { i } ^ { \mathrm { r } } ) - u _ { i } ^ { \mathrm { \Gamma } } | ^ { 2 } } { \sum _ { i } \pi _ { m } ( z _ { i } ^ { \mathrm { r } } ) + \varepsilon } .\tag{2}
$$

The normalization separates the intensity of local PDE violation from the amount of responsibility mass assigned to a component: a small unresolved region need not be overshadowed by a broad component. The stabilizer $\varepsilon > 0$ attenuates scores with negligible support. These are empirical averages over the evaluation points, so their interpretation also depends on how those points are sampled. Initial and boundary observations are combined when initial conditions are imposed softly; exactly enforced conditions contribute no mismatch. The displayed data term applies to prescribed values; derivative conditions require their corresponding operator mismatch. To compare overlapping local approximations, form a k-nearest-neighbor graph of centers and evaluate

$$
\begin{array} { r l } & { J _ { m n } = \frac { \sum _ { i = 1 } ^ { N _ { j } } \pi _ { m } ( z _ { i } ) \pi _ { n } ( z _ { i } ) \| \nabla \ell _ { m } ( z _ { i } ) - \nabla \ell _ { n } ( z _ { i } ) \| _ { 2 } ^ { 2 } } { \sum _ { i = 1 } ^ { N _ { j } } \pi _ { m } ( z _ { i } ) \pi _ { n } ( z _ { i } ) + \varepsilon } , } \\ & { \eta _ { m } ^ { \mathrm { j u m p } } = \frac { \sum _ { n \in \mathcal { N } _ { k } ( m ) } J _ { m n } } { | \mathcal { N } _ { k } ( m ) | } . } \end{array}\tag{3}
$$

The gradients are taken with respect to the same global input coordinates, so discrepancies compare the local functions in a common coordinate system. The responsibility product weights the comparison toward overlap, with ε attenuating pairs of negligible overlap mass. Despite the notation “jump,” this is a disagreement score between overlapping models, rather than an interface jump of the global predictor. The adaptation score is $\eta _ { m } = \eta _ { m } ^ { \mathrm { { r e s } } } + \bar { \lambda } _ { \mathrm { b d r y } } \eta _ { m } ^ { \mathrm { { b d r y } } } + \lambda _ { \mathrm { j u m p } } \eta _ { m } ^ { \mathrm { { j u m p } } }$ ; the weights are nonnegative and either optional term can be disabled. These scores rank candidates; the reliability result below instead uses mass-weighted residual and condition indicators.

## 3.2 POLYNOMIAL ENRICHMENT, PRUNING, AND FUNCTIONAL MERGING

Let $\widetilde { \eta } = \mathrm { m e d i a n } _ { m } \eta _ { m }$ . Relative thresholding identifies components whose scores are large compared with the current representation, without prescribing a fixed absolute residual tolerance. A component satisfying $\eta _ { m } > \tau _ { \mathrm { s p l i t } } \widetilde { \eta }$ is eligible for $p \mathrm { - }$ -enrichment, $p _ { m } \gets \operatorname* { m i n } ( p _ { m } + 1 , p _ { \operatorname* { m a x } } )$ . When the pre-update degree satisfies $p _ { m } < p _ { \operatorname* { m a x } } ,$ this activates preallocated coefficients while preserving its center, scale, and component count. If the newly activated coefficients are zero, enrichment preserves the predictor exactly at the update and makes additional polynomial directions available to subsequent optimization.

With $\gamma _ { m } = \mathrm { s i g m o i d } ( g _ { m } )$ , pruning targets $\gamma _ { m } < \tau _ { \mathrm { g a t e } } ,$ optionally also requiring $\eta _ { m } < \tau _ { \mathrm { m e r g e } } \widetilde { \eta } .$ Candidates are ordered by gate and then error, while retaining at least $M _ { \mathrm { m i n } }$ components. This is a gate-based selection heuristic: in a normalized mixture, a small absolute gate alone does not bound a component’s relative influence. Merging instead screens low-error components for both effective overlap and agreement of local predictions. At evaluation points $\{ z _ { q } \} _ { q = 1 } ^ { N _ { e } } .$ , use

$$
O _ { i j } = \frac { 1 } { N _ { e } } \sum _ { q } \pi _ { i } ( z _ { q } ) \pi _ { j } ( z _ { q } ) , \qquad D _ { i j } = \frac { \sum _ { q } \pi _ { i } ( z _ { q } ) \pi _ { j } ( z _ { q } ) | \ell _ { i } ( z _ { q } ) - \ell _ { j } ( z _ { q } ) | ^ { 2 } } { \sum _ { q } \pi _ { i } ( z _ { q } ) \pi _ { j } ( z _ { q } ) + \varepsilon } .\tag{4}
$$

Accept only pairs with $O _ { i j } > \tau _ { \mathrm { o v e r l a p } }$ and $D _ { i j } < \tau _ { \mathrm { s i m } } .$ , rank them by decreasing overlap, increasing discrepancy, and increasing combined error, then select disjoint pairs. Define the average responsibilities $\begin{array} { r } { q _ { m } = N _ { e } ^ { - 1 } \sum _ { q } \pi _ { m } \bar { ( } z _ { q } ) } \end{array}$ and weights $w _ { i } = q _ { i } / ( q _ { i } + q _ { j } + \varepsilon )$ and $w _ { j } = q _ { j } / ( q _ { i } + q _ { j } + \varepsilon )$

The merged geometry uses the following regularized moment-based update:

$$
\begin{array} { r l } & { \mu _ { i j } = w _ { i } \mu _ { i } + w _ { j } \mu _ { j } , } \\ & { \sigma _ { i j } ^ { \odot 2 } = w _ { i } [ \sigma _ { i } ^ { \odot 2 } + ( \mu _ { i } - \mu _ { i j } ) ^ { \odot 2 } ] } \\ & { \qquad + w _ { j } [ \sigma _ { j } ^ { \odot 2 } + ( \mu _ { j } - \mu _ { i j } ) ^ { \odot 2 } ] . } \end{array}\tag{5}
$$

Here $\odot 2$ denotes element-wise squaring. With $\varepsilon = 0$ and positive total mass, these formulas match the weighted mean and diagonal variance; for $\varepsilon > 0 .$ , the weights sum to less than one. The same weights combine coefficients, $c _ { i j } = w _ { i } c _ { i } + w _ { j } c _ { j }$ , and gate probabilities, $\gamma _ { i j } = w _ { i } \gamma _ { i } + w _ { j } \gamma _ { j }$ , before $g _ { i j } = \log \mathrm { i t } ( \gamma _ { i j } )$ . Any latent code is averaged likewise, and $p _ { i j } = \operatorname* { m a x } ( p _ { i } , p _ { j } )$ . This merge initialization is heuristic: functional agreement before merging does not ensure an unchanged predictor, because the polynomials use different local coordinates and the responsibilities also change. Appendix A gives the full rules.

## 3.3 RESIDUAL-DIRECTED SPLITTING AND COORDINATE-AWARE INITIALIZATION

The remaining capacity $M _ { \mathrm { m a x } } - M ,$ a fractional budget, and a hard per-event cap limit splitting. The geometric baseline selects the parent’s largest-scale axis. The residual-directed alternative uses $\eta _ { m } ^ { \mathrm { r e s } }$ for candidate selection and constructs

$$
\begin{array} { r l r l } & { \omega _ { i m } = \pi _ { m } ( z _ { i } ) | r _ { \Theta } ( z _ { i } ) | ^ { 2 } , } & & { \bar { z } _ { m } = \cfrac { \sum _ { i } \omega _ { i m } z _ { i } } { \sum _ { i } \omega _ { i m } + \varepsilon } , } \\ & { } & & { C _ { m } = \cfrac { \sum _ { i } \omega _ { i m } \left( z _ { i } - \bar { z } _ { m } \right) \left( z _ { i } - \bar { z } _ { m } \right) ^ { \top } } { \sum _ { i } \omega _ { i m } + \varepsilon } . } \end{array}\tag{6}
$$

Choose a unit leading eigenvector $v _ { m } \in \arg \operatorname* { m a x } _ { \| v \| _ { 2 } = 1 } v ^ { \top } C _ { m } v$ . Responsibilities emphasize the region associated with the parent, while squared residual emphasizes unresolved physics within that region. The resulting direction tracks the dominant spread of residual energy in the normalized input coordinates; it need not align with a coordinate axis or the parent’s largest scale. Negligible residual mass or an invalid covariance triggers the geometric fallback. A deterministic subset may be selected using residual energy and responsibility. With exact normalization by $\begin{array} { r } { W = \sum _ { i } \omega _ { i m } > 0 } \end{array}$ , this direction maximizes weighted projected variance (Proposition B.2); this characterizes the geometry of the refinement, without asserting that it maximizes subsequent error reduction.

Set $s _ { m } ~ = ~ ( v _ { m } ^ { \top } \mathrm { d i a g } ( \sigma _ { m } ^ { 2 } ) v _ { m } ) ^ { 1 / 2 }$ and clamp the tentative displacement $\alpha _ { \mathrm { s p l i t } } s _ { m } v _ { m }$ in parentnormalized coordinates. The default anchor is the trained parent center; an optional residualweighted anchor is also clamped. The child’s scale is $\mathrm { c l i p } ( \beta _ { \sigma } \sigma _ { m } , \sigma _ { \mathrm { m i n } } , \sigma _ { \mathrm { m a x } } )$ , with $0 < \beta _ { \sigma } < 1$ and $0 < \sigma _ { \mathrm { m i n } } \le \sigma _ { \mathrm { m a x } }$ . The full displacement and fallback rules appear in Appendix A.

Because coefficients refer to local coordinates, simply copying them changes the polynomial. Parent and child coordinates instead satisfy

$$
\xi _ { p } = d + b \odot \xi _ { c } , \qquad d = ( \mu _ { c } - \mu _ { p } ) \oslash \sigma _ { p } , \qquad b = \sigma _ { c } \oslash \sigma _ { p } .\tag{7}
$$

The multivariate binomial expansion gives

$$
c _ { \beta } ^ { c } = \sum _ { \alpha \in A _ { p _ { p } } \atop \alpha \geq \beta } c _ { \alpha } ^ { p } { \binom { \alpha } { \beta } } d ^ { \alpha - \beta } b ^ { \beta } , \qquad { \binom { \alpha } { \beta } } = \prod _ { q } { \binom { \alpha _ { q } } { \beta _ { q } } } .\tag{8}
$$

Here $\alpha \ge \beta$ is component-wise, and the child retains the parent’s active degree. The complete totaldegree basis is closed under this shift and diagonal rescaling, giving $\ell _ { c } ( z ) = \ell _ { p } ( z )$ (Lemma B.3). For example, shifting and rescaling a linear polynomial generally changes both its slope in child coordinates and its constant coefficient; copying the coefficient vector misses this correction. Inactive higher-order coefficients must be zero or excluded, since they can translate into active lower-order terms. Translation therefore preserves the local function; controlling the global predictor additionally requires controlling the inserted component’s weight.

The default additive split retains the parent and initializes a quiet child:

$$
\gamma _ { c } = \mathrm { c l i p } ( \operatorname* { m i n } \{ \rho _ { g } \gamma _ { p } , \gamma _ { 0 } , \gamma _ { \operatorname* { m a x } } \} , \gamma _ { \operatorname* { m i n } } , \gamma _ { \operatorname* { m a x } } ) , \qquad g _ { c } = \mathrm { l o g i t } ( \gamma _ { c } ) , \qquad \rho _ { g } \ll 1 .\tag{9}
$$

The bounds $0 < \gamma _ { \mathrm { m i n } } \leq \gamma _ { \mathrm { m a x } } < 1$ make the logit finite. The gate is intended to limit the child’s initial contribution while leaving its geometry and polynomial available for subsequent specialization. Together, translation and gating address different sources of disturbance: translation removes the coordinate-change error, while gating controls the change in aggregation. Section 3.4 quantifies this effect for a normalized mixture. Optional symmetric replacement shifts both offspring oppositely, shrinks and translates both, and divides the gate; it changes the parent as well and is outside that bound.

## 3.4 CONDITIONAL RELIABILITY AND CONTROLLED ADAPTATION

Residual accuracy implies solution accuracy only under stability in the chosen norms (Hu et al., 2022; Jiang et al., 2026), not for arbitrary PDEs or losses (Wang et al., 2022). For this analysis, let Q be the PDE domain and Γ the condition domain, equipped with probability measures $\mu _ { Q } , \mu _ { \Gamma } ;$ let $\kappa _ { \Gamma } \geq 0$ . Suppose $L u = f , B u = g ,$ , and

$$
\begin{array} { r } { \| v - u \| _ { X } ^ { 2 } \leq C _ { \mathrm { s t a b } } ^ { 2 } \big ( \| L v - f \| _ { L ^ { 2 } ( \mu _ { Q } ) } ^ { 2 } + \kappa _ { \Gamma } \| B v - g \| _ { L ^ { 2 } ( \mu _ { \Gamma } ) } ^ { 2 } \big ) . } \end{array}\tag{10}
$$

Theorem 3.1 (Conditional ReCAP reliability). Under equation 10, assume $\begin{array} { r } { 0 \leq \pi _ { m } \leq 1 , \sum _ { m } \pi _ { m } = } \end{array}$ 1, and $\varepsilon > 0$ . For masses $\begin{array} { r } { s _ { m } = \int _ { Q } \pi _ { m } d \mu _ { Q } } \end{array}$ and $\begin{array} { r } { s _ { m } ^ { \Gamma } = \int _ { \Gamma } \pi _ { m } d \mu _ { \Gamma } } \end{array}$ , define continuum indicators by the corresponding ε-regularized weighted averages of $| r | ^ { 2 }$ and $| b | ^ { 2 }$ , where $r = L u _ { \Theta } - f$ and $b = B u _ { \Theta } - g$ . Then

$$
\| u _ { \Theta } - u \| _ { X } ^ { 2 } \leq C _ { \mathrm { s t a b } } ^ { 2 } \left[ \sum _ { m } ( s _ { m } + \varepsilon ) \eta _ { m } ^ { \mathrm { r e s } } + \kappa _ { \Gamma } \sum _ { m } ( s _ { m } ^ { \Gamma } + \varepsilon ) \eta _ { m } ^ { \mathrm { b d r y } } \right] .\tag{11}
$$

The independent-sample certificate, including itsfinite-sample correction terms and assumptions, is stated and proved in Theorem B.1 in Appendix B.

The key identity is $\begin{array} { r } { \sum _ { m } ( s _ { m } + \varepsilon ) \eta _ { m } ^ { \mathrm { r e s } } = \| r \| _ { L ^ { 2 } ( \mu _ { Q } ) } ^ { 2 } \colon } \end{array}$ mass weighting reconstructs the global residual norm, whereas unweighted scores identify where to adapt. The theorem transfers an assumed PDE stability estimate to these localized quantities; it does not establish that estimate for every benchmark or make the jump score an error bound. The general condition residual $b = B u _ { \Theta } - g$ reduces to the value mismatch in equation 2 when B is a value trace. Independent certification samples are essential: reusing training points requires uniform generalization control (Hu et al., 2022) or continuous-domain verification (Eiras et al., 2024). The proof (Theorem B.1) and an explicit transport estimate appear in Appendix B.

Controlling the effect of a new child. For one output channel, assume a normalized gated mixture with Gaussian weights $\phi _ { m }$

$$
u ^ { - } = \frac { \sum _ { m } q _ { m } \ell _ { m } } { D } , \qquad q _ { m } = \gamma _ { m } \phi _ { m } , \qquad D = \sum _ { m } q _ { m } .\tag{12}
$$

Additive insertion of a translated child with $\ell _ { c } = \ell _ { p }$ and $q _ { c } = \gamma _ { c } \phi _ { c }$ gives $\begin{array} { r } { u ^ { + } - u ^ { - } = \frac { q _ { c } } { D + q _ { c } } ( \ell _ { p } - } \end{array}$ $u ^ { - } )$ . The change depends on the child’s relative weight and parent–mixture disagreement. Other aggregation rules require a separate perturbation analysis.

Theorem 3.2 (Additive quiet-child stability). Let $L = \partial _ { t } + \beta \cdot \nabla$ and use equation 12. Retain the parent unchanged and translate the child so that $\ell _ { c } \ = \ \ell _ { p } .$ Assume the gate is constant in $z , \ 0 \ \leq \ \gamma _ { c } \ \leq \ \epsilon _ { g } ,$ and $0 ~ \le ~ \phi _ { c } ~ \le ~ 1 , ~ D ~ \ge ~ d _ { 0 } ~ > ~ 0$ on $Q \cup \Gamma ,$ , with $\| L \phi _ { c } \| _ { L ^ { \infty } ( Q ) } \leq K _ { \phi }$ and $\| L D \| _ { L ^ { \infty } ( Q ) } \ \leq \ K _ { D } .$ . For $J ( v ) \ = \ \lVert L v - f \rVert _ { L ^ { 2 } ( Q ) } ^ { 2 } + \kappa _ { \Gamma } \lVert v - g \rVert _ { L ^ { 2 } ( \Gamma ) } ^ { 2 } , \ p u t \ h \ = \ \ell _ { p } - u ^ { - }$ and $K _ { \alpha } = K _ { \phi } / d _ { 0 } + K _ { D } / d _ { 0 } ^ { 2 }$ . Define the finite constant

$$
\begin{array} { r } { C _ { \mathrm { s p l i t } } = \left[ \left( K _ { \alpha } \| h \| _ { L ^ { 2 } ( Q ) } + d _ { 0 } ^ { - 1 } \| L h \| _ { L ^ { 2 } ( Q ) } \right) ^ { 2 } + \kappa _ { \Gamma } d _ { 0 } ^ { - 2 } \| h \| _ { L ^ { 2 } ( \Gamma ) } ^ { 2 } \right] ^ { 1 / 2 } . } \end{array}\tag{13}
$$

The post-split predictor satisfies

$$
\begin{array} { r l r } {  { \sqrt { J ( u ^ { + } ) } \le \sqrt { J ( u ^ { - } ) } + \epsilon _ { g } C _ { \mathrm { s p l i t } } , } } \\ & { } & { J ( u ^ { + } ) - J ( u ^ { - } ) \le 2 \epsilon _ { g } C _ { \mathrm { s p l i t } } \sqrt { J ( u ^ { - } ) } + \epsilon _ { g } ^ { 2 } C _ { \mathrm { s p l i t } } ^ { 2 } . } \end{array}\tag{14}
$$

If admissible child gates can approach zero, the refined class contains the old class in its closure, so its optimal achievable loss cannot increase.

The proof appears with Theorem B.4 in Appendix B. For controlled $C _ { \mathrm { s p l i t } }$ , a smaller gate limits the immediate loss increase. The constant also exposes the relevant conditions: adequate mixture coverage through $d _ { 0 }$ , controlled child derivatives through $K _ { \phi }$ , and agreement between the parent and current mixture through h. A small absolute gate alone is therefore insufficient when the denominator is very small or the child is excessively narrow. A fixed positive gate floor also precludes the zero-gate argument for class inclusion. This result concerns additive insertion for the stated transport objective, rather than monotonic training loss or a guarantee for nonlinear and higher-order benchmark operators. Zero-initialized $p \cdot$ -enrichment preserves the current function exactly; symmetric replacement, pruning, and merging require separate perturbation analysis.

## 4 EXPERIMENTAL RESULTS

Benchmarks, baselines and implementation details. We evaluate ReCAP on five PDE benchmarks: flow mixing (FM3D), Helmholtz (H3D), nonlinear diffusion (ND3D), and Klein–Gordon equations (KG3D and KG4D). The dimension counts all input coordinates, including time: H3D has three spatial coordinates, FM3D, ND3D, and KG3D have two spatial coordinates and time, and KG4D has three spatial coordinates and time. Appendix C provides the problem definitions. All experiments run on a single NVIDIA RTX 4090, with baselines reimplemented in the same hardware and software environment. We compare against eleven methods: PINN (Raissi et al., 2019), ActNet (Guilhoto & Perdikaris, 2025), PirateNets (Wang et al., 2024a), SPINN (Cho et al., 2023), JAXPI (Wang et al., 2023), CoupledNet (Meng et al., 2026), CausalPINN (Wang et al., 2024b), DB-PINN (Zhou et al., 2025), SINN (Yu & Oseledets, 2026), Scale-PINN (Chiu et al., 2026), and PIG (Kang et al., 2025). ReCAP hyperparameters and computational costs appear in Appendices D and F, respectively. We report relative $L ^ { 2 }$ error to compare solution accuracy across these benchmarks.

Table 1: Reported relative $L ^ { 2 }$ errors on five PDE benchmarks (lower is better). The lowest error in each column is highlighted in green , and the second-lowest is underlined. The final row gives ReCAP’s percentage error reduction relative to the strongest baseline, computed from the displayed values. A dash indicates an unreported result.
<table><tr><td>Method</td><td>FM3D</td><td>H3D</td><td>ND3D</td><td>KG3D</td><td>KG4D</td></tr><tr><td>PINN (Raissi et al., 2019)</td><td>0.001968</td><td>0.527911</td><td>0.007788</td><td>0.042415</td><td>0.009683</td></tr><tr><td>SPINN (Cho et al., 2023)</td><td>0.007995</td><td>0.066970</td><td>0.005182</td><td>0.003773</td><td>0.004786</td></tr><tr><td>JAXPI (Wang et al., 2023)</td><td>0.000328</td><td>0.110786</td><td>0.002229</td><td>0.007136</td><td>0.003942</td></tr><tr><td>PirateNets (Wang et al., 2024a)</td><td>0.000291</td><td>0.077632</td><td>0.002505</td><td>0.003860</td><td>0.007206</td></tr><tr><td>CausalPINN (Wang et al., 2024b)</td><td>0.000383</td><td></td><td>0.006638</td><td>0.021933</td><td>0.008271</td></tr><tr><td>DB-PINN (Zhou et al., 2025)</td><td>0.002053</td><td>0.587973</td><td>0.005622</td><td>0.007994</td><td>0.017544</td></tr><tr><td>PIG (Kang et al., 2025)</td><td>0.000305</td><td>0.184739</td><td>0.001791</td><td>0.004206</td><td>0.005286</td></tr><tr><td>ActNet (Guilhoto &amp; Perdikaris, 2025)</td><td>0.001787</td><td>0.093405</td><td>0.004842</td><td>0.006199</td><td>0.005991</td></tr><tr><td>CoupledNet (Meng et al., 2026)</td><td>0.000856</td><td>0.158846</td><td>0.006488</td><td>0.070929</td><td>0.009005</td></tr><tr><td>SINN (Yu &amp; Oseledets, 2026)</td><td>0.003955</td><td>0.700347</td><td>0.003598</td><td>0.011043</td><td>0.059185</td></tr><tr><td>Scale-PINN (Chiu et al., 2026)</td><td>0.034121</td><td>0.565737</td><td>0.082126</td><td>0.092426</td><td>0.093574</td></tr><tr><td>ReCAP (Ours)</td><td>0.000211</td><td>0.059810</td><td>0.001574</td><td>0.003181</td><td>0.003463</td></tr><tr><td>Improvement</td><td>27.5%</td><td>10.7%</td><td>12.1%</td><td>15.7%</td><td>12.2%</td></tr></table>

![](images/25481279980f524152396b15ad70a41d16c1d113e8d9c95c7d971c327c9f3162.jpg)

(b) KG3D  
![](images/4a1b1b0b8fc55322c3ec2a4e27fb072880e1de539439556292c8972c3bf78ddf.jpg)

(c) FM3D  
![](images/41aab09372f1738388f7a40d9866143d56de6ecc932bf0405ca4e531af98be3c.jpg)

![](images/f7bc448bc582e889ac94dbe9cf90c1600e43194d035620827f3c2dc2aebf07d0.jpg)

t = 1.5  
![](images/549411d9313415e1d386491383fe9ac277940eb74fb1b18a23bd5e02fbe4038e.jpg)

![](images/afbfce65f6329824c586e59325e4f76ea1c706d92856094e2b4ced4c78b676e1.jpg)

![](images/09cba0ecfde60f9757de4e674a9b175621c4fbd194fec46d28475e2d362f6bdd.jpg)

![](images/9e343fd32c51b83aac15c0f44aa6c77c38a795d518edfedeb3747f6320710351.jpg)

![](images/35c944984cbf23e7c722253e3b5be905aaeca4eec14191493739cfbd5fbd399b.jpg)  
x

![](images/118130abaa3954e7c1741aa3dd6e2a0a537a366c3cca5541bdf182f2c7fde7d9.jpg)  
Figure 2: Qualitative solution and error comparisons. (a) ND3D at $t = 0 . 4 \colon$ reference solution and ReCAP prediction. (b) KG3D: ReCAP prediction and pointwise absolute error. (c) FM3D at $t = 1 . 5 \colon$ predictions and corresponding absolute-error maps for PirateNets, PIG, and ReCAP. ReCAP accurately resolves the dominant solution structures while maintaining small and spatially localized errors, including around rapidly varying regions.

Main Results. ReCAP achieves the lowest reported relative $L ^ { 2 }$ error on each of the five benchmarks in Table 1. Compared with the strongest baseline on each problem, ReCAP reduces error by 27.5% on FM3D (PirateNets), 10.7% on H3D (SPINN), 12.1% on ND3D (PIG), 15.7% on KG3D (SPINN), and 12.2% on KG4D (JAXPI). These comparisons establish the improvement on the tested three- and four-dimensional problems; the ablations below examine how the representation and its adaptation contribute to the observed behavior.

The strongest baseline varies across the benchmarks: PirateNets, SPINN, PIG, and JAXPI each provide the best competing result on at least one problem. ReCAP improves over these problemspecific reference points across transport, elliptic, nonlinear diffusion, and wave equations. This breadth supports its effectiveness on the tested problems; the ablations below examine whether the gains depend on adaptive allocation beyond component-count scaling.

![](images/d1f0c7e9b811da5f47b0dddaabf69a3d7bfe6006424088683b52fc06a806719f.jpg)

![](images/289ba5f0bd2cd36c6bae41281c5e84137ccdc864e4e65bea949710d90885bcc3.jpg)

![](images/1ea09bf27d8b3e6e46646a0b85879ecb8b4c82b33ce609ed50a63c06daf1f919.jpg)  
Figure 3: KG3D ablations. Left: cumulative error reduction from adding M2–M4 to the base representation M1. Middle: capacity-controlled comparisons. Right: splitting-threshold sensitivity, with $\tau _ { \mathrm { s p l i t } } = 1 . 2$ giving the lowest error among the tested values.

Figure 2 provides spatial evidence complementary to the aggregate error metrics. On ND3D, Re-CAP closely reproduces the reference solution, including the localized peaks and surrounding lowamplitude structure. For KG3D, the predicted field preserves the pronounced spatiotemporal variations while the corresponding pointwise error remains small over most of the domain. The FM3D comparison further highlights the difference between methods: although all three models recover the overall spiral structure, ReCAP exhibits substantially less background error and concentrates the remaining error near the most rapidly varying region around the spiral core and fronts.

These visualizations complement the aggregate errors by showing how ReCAP resolves localized solution structures and where prediction errors remain. Additional solution visualizations are provided in Appendix E.

Ablation study. Figure 3 examines ReCAP’s key design choices on KG3D, including (i) the cumulative contributions of its adaptation components, (ii) the benefits of adaptive capacity allocation beyond particle-count scaling, and (iii) sensitivity to the splitting threshold.

Cumulative component ablation. We first evaluate the incremental benefits of the adaptation mechanisms on KG3D (Figure 3, left). Starting from M1, a dynamic Gaussian-particle representation with hard constraints, we successively introduce particle splitting for local refinement (M2), pruning and merging for redundancy control (M3), and residual-based adaptive collocation sampling (M4). The relative $\breve { L } ^ { 2 }$ error decreases at every stage, from $6 . 7 5 \times 1 0 ^ { - 3 } \mathrm { t o } \dot { 5 } . 0 6 \times 1 0 ^ { - 3 } , 4 . 0 7 \times \dot { 1 0 } ^ { - 3 }$ and finally $3 . 7 1 \times 1 0 ^ { - 3 }$ , corresponding to a 45.0% reduction over M1.

The reductions continue after splitting, supporting refinement together with redundancy control and adaptive sampling. Since the components are introduced cumulatively, the observed reductions quantify incremental gains conditional on preceding components, rather than their independent effects.

Adaptive allocation versus particle-count scaling. We next examine whether increasing the particle count is sufficient to recover ReCAP’s accuracy (Figure 3, middle). PIG-small uses the initial Gaussian count, whereas PIG-large matches the final count reached by ReCAP. Two additional controls restrict how capacity is allocated: Fixed- $\boldsymbol { p } _ { \mathrm { m a x } }$ assigns the maximum polynomial order uniformly, while Split-only uses adaptive splitting without polynomial-order refinement or coarsening. ReCAP achieves a relative $L ^ { 2 }$ error of 3. $1 8 1 \times 1 0 ^ { - 3 }$ , reducing the error by 24.4% relative to PIGlarge, 43.1% relative to $\mathrm { F i x e d - } p _ { \mathrm { m a x } } .$ , and 17.8% relative to Split-only. Thus, matching the final particle count does not close the accuracy gap, and neither uniformly maximal polynomial order nor splitting alone matches the complete method. Together, these comparisons support ReCAP’s combined strategy of local spatial refinement, polynomial-order adaptation, and redundancy control over the tested alternatives.

Splitting-threshold sensitivity. We evaluate sensitivity to the splitting threshold over $\tau _ { \mathrm { s p l i t } } ~ \in$ {0.4, 0.8, 1.2, 1.6, 2.0} on KG3D (Figure 3, right). The best tested setting, $\tau _ { \mathrm { s p l i t } } = 1 . 2$ , achieves a relative $L ^ { 2 }$ error of $3 . 7 7 2 \times 1 0 ^ { - 3 }$ , a 20.9% reduction relative to $\tau _ { \mathrm { s p l i t } } = 0 . 8$ . The response is non-monotonic: although 0.8 produces the largest error, the remaining thresholds, 0.4, 1.6, and 2.0, yield errors between $3 . { \bar { 8 } } 7 0 \times { \bar { 1 } } 0 ^ { - 3 }$ and $3 . 9 4 0 \ \times 1 0 ^ { - 3 }$ , all within 4.5% of the minimum.

The sweep therefore favors $\tau _ { \mathrm { s p l i t } } = 1 . 2$ , with several other thresholds giving similar accuracy. These values correspond to the sensitivity sweep; the main-benchmark results are reported separately in Table 1.

## 5 CONCLUSION

ReCAP treats representational capacity as an adaptive resource for physics-informed PDE solving. It couples residual-guided polynomial enrichment and splitting with function-aware coarsening, using coordinate-aware quiet-child initialization to introduce new capacity. Across five three- and fourdimensional PDE benchmarks, ReCAP achieves the lowest reported relative $L ^ { 2 }$ error; capacitycontrolled comparisons further support the benefit of targeted allocation beyond increasing component count alone. Under the stated assumptions, the analysis connects mass-weighted physics indicators to solution error and bounds the perturbation from additive refinement. Together, these results support physics-informed neural plasticity: jointly learning the solution and adapting where, how, and how much capacity is used to represent it.

## AI USE STATEMENT

In this work, we used generative AI tools to assist in developing and writing proofs of mathematical claims, provide feedback on methodological exposition, and refine the interpretation and presentation of results. Additionally, we used these tools to polish the writing, improve clarity and organization, We have reviewed all AI-assisted work, including checking the assumptions, mathematical statements, and individual steps of the proofs, and verifying that the methodological descriptions and interpretations are consistent with our implementation and experimental evidence. We take responsibility for the final content of this work, including all text, claims, and artifacts produced with the aid of generative AI.

## ACKNOWLEDGMENTS

AIAR gratefully acknowledges support from the Tsinghua University Dushi Program and the Tsinghua University Initiative Scientific Research Program. This work is also supported by the Yau Mathematical Sciences Center, Tsinghua University. CWC was supported by the Swiss National Science Foundation (SNSF) under Grant No. 20HW-1 220785.

## REFERENCES

Ivo Babuska and Manil Suri. The p and h-p versions of the finite element method, basic principlesˇ and properties. SIAM Review, 36(4):578–632, 1994. doi: 10.1137/1036141. 3

Jonah Botvinick-Greenhouse, Wael H Ali, Mouhacine Benosman, and Saviz Mowlavi. AB-PINNs: Adaptive-basis physics-informed neural networks for residual-driven domain decomposition. Machine Learning: Science and Technology, 7:045025, 2026. doi: 10.1088/2632-2153/ae8638. 2, 3

David S. Broomhead and David Lowe. Multivariable functional interpolation and adaptive networks. Complex Systems, 2:321–355, 1988. 3

Pao-Hsiung Chiu, Jian Cheng Wong, Chin Chun Ooi, Chang Wei, Yuchen Fan, and Yew-Soon Ong. Scale-PINN: Learning efficient physics-informed neural networks through sequential correction, 2026. URL https://arxiv.org/abs/2602.19475. 7, 8

Junwoo Cho, Seungtae Nam, Hyunmo Yang, Seok-Bae Yun, Youngjoon Hong, and Eunbyung Park. Separable physics-informed neural networks. Advances in Neural Information Processing Systems, 36:23761–23788, 2023. 3, 7, 8

Arka Daw, Jie Bu, Sifan Wang, Paris Perdikaris, and Anuj Karpatne. Mitigating propagation failures in physics-informed neural networks using retain-resample-release (R3) sampling. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 7264–7302. PMLR, 2023. URL https://proceedings. mlr.press/v202/daw23a.html. 2, 3

Francisco Eiras, Adel Bibi, Rudy R Bunel, Krishnamurthy Dj Dvijotham, Philip Torr, and M. Pawan Kumar. Efficient error certification for physics-informed neural networks. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 12318–12347. PMLR, 2024. URL https://proceedings.mlr. press/v235/eiras24a.html. 7, 19

Leonardo Ferreira Guilhoto and Paris Perdikaris. Deep learning alternatives of the Kolmogorov superposition theorem. In The Thirteenth International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ c2469e35d469e3c0eca09dbe484eb474-Abstract-Conference.html. 7, 8

Zheyuan Hu, Ameya D Jagtap, George Em Karniadakis, and Kenji Kawaguchi. When do extended physics-informed neural networks (XPINNs) improve generalization? SIAM Journal on Scientific Computing, 44(5):A3158–A3182, 2022. doi: 10.1137/21M1447039. 6, 7, 18, 19

Ameya D Jagtap and George E Karniadakis. Extended physics-informed neural networks (XPINNs): A generalized space-time domain decomposition based deep learning framework for nonlinear partial differential equations. Communications in Computational Physics, 28(5):2002–2041, 2020. doi: 10.4208/cicp.OA-2020-0164. 3

Deqing Jiang, Justin Sirignano, and Samuel N Cohen. Global convergence of deep galerkin and PINN methods for solving partial differential equations. SIAM Journal on Financial Mathematics, 17(2):620–645, 2026. doi: 10.1137/24M1701502. 6, 18

Namgyu Kang, Jaemin Oh, Youngjoon Hong, and Eunbyung Park. PIG: Physics-Informed Gaussians as Adaptive Parametric Mesh Representations. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id= y5B0ca4mjt. 2, 3, 7, 8

George E Karniadakis, Ioannis G Kevrekidis, Lu Lu, Paris Perdikaris, Sifan Wang, and Liu Yang. Physics-informed machine learning. Nature Reviews Physics, 3(6):422–440, 2021. doi: 10.1038/ s42254-021-00314-5. 1, 3

Ehsan Kharazmi, Zhongqiang Zhang, and George Em Karniadakis. hp-VPINNs: Variational physics-informed neural networks with domain decomposition. Computer Methods in Applied Mechanics and Engineering, 374:113547, 2021. doi: 10.1016/j.cma.2020.113547. 3

Aditi S. Krishnapriyan, Amir Gholami, Shandian Zhe, Robert Kirby, and Michael W Mahoney. Characterizing possible failure modes in physics-informed neural networks. Advances in Neural Information Processing Systems, 34, 2021. 1, 3

Lingshi Meng, Haosen Shi, and Sinno Jialin Pan. Deep coupling learning for solving PDEs. In Proceedings of the 43rd International Conference on Machine Learning, 2026. URL https: //openreview.net/forum?id=4pfxoYPQSY. 7, 8

Ben Moseley, Andrew Markham, and Tarje Nissen-Meyer. Finite basis physics-informed neural networks (FBPINNs): a scalable domain decomposition approach for solving differential equations. Advances in Computational Mathematics, 49(4):62, 2023. doi: 10.1007/s10444-023-10065-9. 2, 3

Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred Hamprecht, Yoshua Bengio, and Aaron Courville. On the spectral bias of neural networks. In Proceedings ofthe 36th International Conference on Machine Learning, volume 97 of Proceedings ofMachine Learning Research, pp. 5301–5310. PMLR, 2019. URL https://proceedings.mlr.press/v97/ rahaman19a.html. 1, 3

Maziar Raissi, Paris Perdikaris, and George E. Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378:686–707, 2019. doi: 10.1016/j. jcp.2018.10.045. 1, 3, 7, 8

Panos Tamamidis and Dennis N. Assanis. Evaluation of various high-order-accuracy schemes with and without flux limiters. International Journal for Numerical Methods in Fluids, 16(10):931– 948, 1993. 21

Chuwei Wang, Shanda Li, Di He, and Liwei Wang. Is L<sup>2</sup> physics informed loss always suitable for training physics informed neural network? Advances in Neural Information Processing Systems, 35:8278–8290, 2022. 6, 18

Sifan Wang, Yujun Teng, and Paris Perdikaris. Understanding and mitigating gradient flow pathologies in physics-informed neural networks. SIAM Journal on Scientific Computing, 43(5):A3055– A3081, 2021. doi: 10.1137/20M1318043. 1, 3

Sifan Wang, Shyam Sankaran, Hanwen Wang, and Paris Perdikaris. An expert’s guide to training physics-informed neural networks. arXiv preprint arXiv:2308.08468, 2023. 3, 7, 8

Sifan Wang, Bowen Li, Yuhan Chen, and Paris Perdikaris. PirateNets: Physics-informed deep learning with residual adaptive networks. Journal ofMachine Learning Research, 25(402):1–51, 2024a. URL https://jmlr.org/papers/v25/24-0313.html. 2, 3, 7, 8

Sifan Wang, Shyam Sankaran, and Paris Perdikaris. Respecting causality for training physicsinformed neural networks. Computer Methods in Applied Mechanics and Engineering, 421: 116813, 2024b. doi: 10.1016/j.cma.2024.116813. 7, 8

Chenxi Wu, Min Zhu, Qinyang Tan, Yadhu Kartha, and Lu Lu. A comprehensive study of nonadaptive and residual-based adaptive sampling for physics-informed neural networks. Computer Methods in Applied Mechanics and Engineering, 403:115671, 2023. doi: 10.1016/j.cma.2022. 115671. 2, 3

Tianchi Yu and Ivan Oseledets. Spectral-informed neural networks outperform spectral methods in high-dimensional PDEs. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id=KAHCMPsPeI. 7, 8

Chenhong Zhou, Jie Chen, Zaifeng Yang, and Ching Eng Png. Dual-balancing for physics-informed neural networks. In Proceedings ofthe Thirty-Fourth International Joint Conference on Artificial Intelligence, IJCAI ’25, pp. 7164–7172, 2025. doi: 10.24963/ijcai.2025/797. URL https: //www.ijcai.org/proceedings/2025/797. 7, 8

## Supplementary Material Physics-Informed Neural Plasticity: PDE Solvers That Reshape Themselves

## CONTENTS

A Detailed Representation and Adaptation Rules 13   
A.1 Problem Formulation . 13   
A.2 Adaptive Gaussian Local Representation 14   
A.3 Physics-Guided Local Error Estimation 14   
A.4 Adaptive Representation Refinement and Coarsening 15   
B Further Theoretical Results 18   
C PDE Benchmarks 21   
C.1 (2+1)D Flow Mixing 21   
C.2 3D Helmholtz Equation 22   
C.3 (2+1)D Nonlinear Diffusion 22   
C.4 Klein–Gordon Equations 22   
D Experimental Settings 22   
E Visualization 23   
F Computational Cost 26

## A DETAILED REPRESENTATION AND ADAPTATION RULES

The following construction expands the representation and component-update rules summarized in Section 3.

## A.1 PROBLEM FORMULATION

Let $\Omega \subset \mathbb { R } ^ { d }$ denote a spatial or spatiotemporal domain, and consider a differential equation of the form $\mathcal { N } [ u ] ( { \mathbf { z } } ) = 0 , \quad \quad \mathbf { z } \in \Omega$ , subject to initial or boundary conditions $B [ u ] ( { \mathbf { z } } ) = g ( { \mathbf { z } } )$ z ∈ ∂Ω. We denote the learned solution by $u _ { \Theta } .$ , where Θ contains both the continuously optimized parameters of the base model and the parameters of the adaptive Gaussian representation. The corresponding physics residual is $r _ { \Theta } ( \mathbf { z } ) \bar { = } \mathcal { N } [ u _ { \Theta } ] ( \mathbf { z } )$

The adaptation mechanism is independent of the precise composition of the base training objective. We therefore denote that objective abstractly by $\mathcal { L } _ { \mathrm { b a s e } } ( \Theta )$ . Between structural adaptation events, all currently active continuous parameters are optimized by gradient-based updates to $\mathcal { L } _ { \mathrm { b a s e } }$

For example, the FM3D benchmark uses coordinates ${ \bf z } = ( t , x , y )$ , with linear transport residual

$$
\boldsymbol { r } _ { \Theta } ( t , x , y ) = \partial _ { t } u _ { \Theta } ( t , x , y ) + a \partial _ { x } u _ { \Theta } ( t , x , y ) + b \partial _ { y } u _ { \Theta } ( t , x , y ) ,\tag{15}
$$

equation 15 is the residual for FM3D. For the other benchmarks, the same adaptation rules use the corresponding PDE residual specified in Appendix C.

## A.2 ADAPTIVE GAUSSIAN LOCAL REPRESENTATION

At a given optimization stage, the representation contains M Gaussian-localized components $\mathcal { G } =$ $\left\{ G _ { m } \right\} _ { m = 1 } ^ { M }$ . For concise notation, we write $G _ { m } = ( \mu _ { m } , \pmb { \sigma } _ { m } , g _ { m } , \pmb { c } _ { m } , p _ { m } )$ , where $\pmb { \mu _ { m } } ~ \in ~ \mathbb { R } ^ { d }$ is the component center, $\pmb { \sigma } _ { m } \in \mathbb { R } _ { + } ^ { d }$ is its diagonal scale, $g _ { m } \in \mathbb { R }$ is a gate logit, $\mathbf { c } _ { m }$ contains local polynomial coefficients, and $p _ { m }$ is the currently active polynomial order.

After absorbing the fixed coordinate normalization used by the model into $\mathbf { z } ,$ the normalized coordinate of component m is $\pmb { \xi } _ { m } ( \mathbf { z } ) = ( \mathbf { z } - \pmb { \mu } _ { m } ) \oslash \pmb { \sigma } _ { m }$ , where ⊘ denotes element-wise division. The component represents a local polynomial

$$
\ell _ { m } ( \mathbf { z } ) = \sum _ { \alpha \in \mathcal { A } _ { p m } } c _ { m , \alpha } \xi _ { m } ( \mathbf { z } ) ^ { \alpha } , \qquad \mathcal { A } _ { p _ { m } } = \big \{ \alpha \in \mathbb { N } _ { 0 } ^ { d } : | \alpha | \leq p _ { m } \big \} .\tag{16}
$$

Here, $\begin{array} { r } { \pmb { \xi ^ { \alpha } } = \prod _ { q = 1 } ^ { d } \xi _ { q } ^ { \alpha _ { q } } } \end{array}$ . In the three-dimensional setting, the basis contains all monomials up to total degree three, including quadratic and cubic cross terms, while the local order $p _ { m }$ determines which basis terms are active for each component.

Let $\pi _ { m } ( \mathbf { z } ) \geq 0$ denote the effective responsibility of component m. Any additional internal branch or channel index is averaged out when computing the adaptation statistics, including responsibilities, centers, local values, and local gradients, and is omitted from the notation for clarity. Since the adaptation procedure is agnostic to the specific responsibility and output-aggregation rules, we express the model prediction abstractly as $u _ { \Theta } ( { \bf z } ) = \mathcal { A } _ { \Theta } \left( \{ \pi _ { m } ( { \bf z } ) , \ell _ { m } ( { \bf z } ) , g _ { m } \} _ { m = 1 } ^ { M } \right)$

Our adaptation procedure requires only that the base model expose the prediction $u _ { \Theta }$ , the effective responsibilities $\left\{ \pi _ { m } \right\}$ , and, when necessary, the component-wise local values $\{ \ell _ { m } \}$

## A.3 PHYSICS-GUIDED LOCAL ERROR ESTIMATION

The global physics loss measures aggregate equation and condition mismatch, but does not identify which components require additional capacity. We localize these quantities using the learned component responsibilities; a solution-error interpretation additionally requires the stability assumptions in Appendix B.

Let ${ \mathcal { C } } = \{ { \mathbf { z } } _ { i } \} _ { i = 1 } ^ { N _ { c } }$ denote the collocation set. The component-wise residual indicator is

$$
\eta _ { m } ^ { \mathrm { r e s } } = \frac { \sum _ { i = 1 } ^ { N _ { c } } \pi _ { m } ( \mathbf { z } _ { i } ) \left| r _ { \Theta } ( \mathbf { z } _ { i } ) \right| ^ { 2 } } { \sum _ { i = 1 } ^ { N _ { c } } \pi _ { m } ( \mathbf { z } _ { i } ) + \varepsilon } .\tag{17}
$$

Thus, a large value of $\eta _ { m } ^ { \mathrm { r e s } }$ indicates a large weighted PDE residual in the region for which component m is responsible.

When value-based initial or boundary observations are available, we add a condition-error indicator. Let $\mathcal { D } _ { \Gamma } = \{ ( \mathbf { z } _ { i } ^ { \Gamma } , u _ { i } ^ { \Gamma } ) \} _ { i = 1 } ^ { N _ { \Gamma } }$ denote the condition points. Then

$$
\eta _ { m } ^ { \mathrm { b d r y } } = \frac { \sum _ { i = 1 } ^ { N _ { \Gamma } } \pi _ { m } ( \mathbf { z } _ { i } ^ { \Gamma } ) \left| { u } _ { \Theta } ( \mathbf { z } _ { i } ^ { \Gamma } ) - u _ { i } ^ { \Gamma } \right| ^ { 2 } } { \sum _ { i = 1 } ^ { N _ { \Gamma } } \pi _ { m } ( \mathbf { z } _ { i } ^ { \Gamma } ) + \varepsilon } .\tag{18}
$$

In the implementation, initial and boundary points are combined when the initial condition is imposed softly. If the initial condition is already hard-constrained by the model, only the remaining boundary points are used in equation 18.

Residual error alone may fail to identify neighboring components whose local approximations are mutually inconsistent. We therefore construct a k-nearest-neighbor graph between Gaussian centers. Let $\mathcal { N } _ { k } ( m )$ denote the neighbors of component m. For a pair $( m , n )$ , define their overlap-weighted gradient discrepancy as

$$
J _ { m n } = \frac { \sum _ { i = 1 } ^ { N _ { j } } \pi _ { m } ( \mathbf { z } _ { i } ) \pi _ { n } ( \mathbf { z } _ { i } ) \left\| \nabla \ell _ { m } ( \mathbf { z } _ { i } ) - \nabla \ell _ { n } ( \mathbf { z } _ { i } ) \right\| _ { 2 } ^ { 2 } } { \sum _ { i = 1 } ^ { N _ { j } } \pi _ { m } ( \mathbf { z } _ { i } ) \pi _ { n } ( \mathbf { z } _ { i } ) + \varepsilon } .\tag{19}
$$

The component-wise inconsistency indicator is then

$$
\eta _ { m } ^ { \mathrm { j u m p } } = \frac { 1 } { | \mathcal { N } _ { k } ( m ) | } \sum _ { n \in \mathcal { N } _ { k } ( m ) } J _ { m n } .\tag{20}
$$

The product $\pi _ { m } \pi _ { n }$ weights the comparison toward points at which both components are relevant. The denominator regularizer attenuates pairs with negligible overlap mass. Gradients are taken in a common global coordinate system, including the chain-rule factors from each component’s local coordinates. This is a local disagreement score, not a literal interface jump of the global predictor.

The complete adaptation score is

$$
\eta _ { m } = \eta _ { m } ^ { \mathrm { r e s } } + \lambda _ { \mathrm { b d r y } } \eta _ { m } ^ { \mathrm { b d r y } } + \lambda _ { \mathrm { j u m p } } \eta _ { m } ^ { \mathrm { j u m p } } ,\tag{21}
$$

where $\lambda _ { \mathrm { b d r y } } , \lambda _ { \mathrm { j u m p } } \geq 0$ and either optional term can be disabled. Unlike a global loss, equation 21 assigns the current physics error to individual learned components and therefore provides a criterion for targeted changes in representation complexity.

## A.4 ADAPTIVE REPRESENTATION REFINEMENT AND COARSENING

The localized error indicators in Section A.3 determine how representation capacity is redistributed across the domain. At each adaptation event, we first compute the robust reference value

$$
\widetilde { \eta } = \mathrm { m e d i a n } \left\{ \eta _ { m } \right\} _ { m = 1 } ^ { M } .
$$

The median limits the influence of a small number of extreme residuals and makes the adaptation criteria relative to the current distribution of component-wise errors. Based on these statistics, the representation is modified through polynomial enrichment, pruning, merging, and Gaussian splitting.

Polynomial-order refinement. Before introducing additional Gaussian components, we first attempt to increase the approximation order of high-error components. A component satisfying

$$
\eta _ { m } > \tau _ { \mathrm { s p l i t } } \widetilde { \eta }\tag{22}
$$

is eligible for local polynomial enrichment, with

$$
p _ { m } \gets \operatorname* { m i n } \left\{ p _ { m } + 1 , p _ { \operatorname* { m a x } } \right\} .\tag{23}
$$

When the pre-update degree satisfies $p _ { m } < p _ { \operatorname* { m a x } } ,$ , this operation activates the next group of preallocated polynomial terms without increasing the number of Gaussian components. It therefore provides local p-adaptation: the active polynomial basis expands while the Gaussian geometry and component count remain unchanged. Setting the newly activated coefficients to zero preserves the current predictor.

Gate-based pruning. Pruning targets components with small gate probabilities. Let

$$
\gamma _ { m } = \mathrm { s i g m o i d } ( g _ { m } )
$$

denote the effective gate probability, averaged over internal branches when applicable. Component m becomes a pruning candidate when

$$
\gamma _ { m } < \tau _ { \mathrm { g a t e } } .\tag{24}
$$

An optional conservative criterion additionally requires the component to have low localized error,

$$
\eta _ { m } < \tau _ { \mathrm { m e r g e } } \widetilde { \eta } .
$$

When the number of candidates exceeds the removal budget, components with the smallest gates are prioritized, with localized error used as an additional ordering criterion. Pruning is constrained so that at least $M _ { \mathrm { m i n } }$ components remain, preventing excessive collapse of the representation.

Functional merging. Pruning removes weakly active components individually, whereas merging coarsens redundant components that represent similar local functions. Low-error components satisfying

$$
\eta _ { m } < \tau _ { \mathrm { m e r g e } } \widetilde { \eta }
$$

are considered for merging. Importantly, geometric proximity alone is not sufficient: two components must exhibit both non-negligible effective overlap and similar local predictions.

Using a set of evaluation points $\{ \mathbf { z } _ { q } \} _ { q = 1 } ^ { N _ { e } }$ , we define the effective overlap between components i and j as

$$
O _ { i j } = \frac { 1 } { N _ { e } } \sum _ { q = 1 } ^ { N _ { e } } \pi _ { i } ( \mathbf { z } _ { q } ) \pi _ { j } ( \mathbf { z } _ { q } ) ,\tag{25}
$$

and their overlap-weighted functional discrepancy as

$$
D _ { i j } = \frac { \sum _ { q = 1 } ^ { N _ { e } } \pi _ { i } ( \mathbf { z } _ { q } ) \pi _ { j } ( \mathbf { z } _ { q } ) \left| \boldsymbol { \ell } _ { i } ( \mathbf { z } _ { q } ) - \boldsymbol { \ell } _ { j } ( \mathbf { z } _ { q } ) \right| ^ { 2 } } { \sum _ { q = 1 } ^ { N _ { e } } \pi _ { i } ( \mathbf { z } _ { q } ) \pi _ { j } ( \mathbf { z } _ { q } ) + \varepsilon } .\tag{26}
$$

A pair is accepted only if

$$
O _ { i j } > \tau _ { \mathrm { o v e r l a p } } , \qquad D _ { i j } < \tau _ { \mathrm { s i m } } .\tag{27}
$$

Thus, merging is restricted to components that are simultaneously relevant in the same region and already describe similar local functions. Candidate pairs are ranked according to large overlap, small functional discrepancy, and small combined error. Selected pairs are disjoint, so each component participates in at most one merge during an adaptation event.

To construct the merged component, define the average effective responsibilities and corresponding regularized weights as

$$
q _ { m } = \frac { 1 } { N _ { e } } \sum _ { q = 1 } ^ { N _ { e } } \pi _ { m } ( \mathbf { z } _ { q } ) , \qquad w _ { i } = \frac { q _ { i } } { q _ { i } + q _ { j } + \varepsilon } , \qquad w _ { j } = \frac { q _ { j } } { q _ { i } + q _ { j } + \varepsilon } .\tag{28}
$$

The merged center and diagonal variance use the following moment-based update. It gives exact weighted moment matching when $\varepsilon = 0$ and $q _ { i } + q _ { j } > 0 ;$ ; positive ε makes the weights sum to less than one:

$$
\begin{array} { r l } & { \mu _ { i j } = w _ { i } \pmb { \mu } _ { i } + w _ { j } \pmb { \mu } _ { j } , } \\ & { \sigma _ { i j } ^ { \odot 2 } = w _ { i } \left[ \pmb { \sigma } _ { i } ^ { \odot 2 } + \left( \pmb { \mu } _ { i } - \pmb { \mu } _ { i j } \right) ^ { \odot 2 } \right] } \\ & { \qquad + w _ { j } \left[ \pmb { \sigma } _ { j } ^ { \odot 2 } + \left( \pmb { \mu } _ { j } - \pmb { \mu } _ { i j } \right) ^ { \odot 2 } \right] . } \end{array}\tag{29}
$$

(30)

The polynomial coefficients and gate probabilities are combined using the same weights, while the larger polynomial order is retained:

$$
\mathbf { c } _ { i j } = w _ { i } \mathbf { c } _ { i } + w _ { j } \mathbf { c } _ { j } ,\tag{31}
$$

$$
\gamma _ { i j } = w _ { i } \gamma _ { i } + w _ { j } \gamma _ { j } , \qquad g _ { i j } = \mathrm { l o g i t } ( \gamma _ { i j } ) ,\tag{32}
$$

$$
p _ { i j } = \operatorname* { m a x } \{ p _ { i } , p _ { j } \} .\tag{33}
$$

Any associated latent code is likewise averaged. The functional-similarity test in Equation equation 27 screens candidates before this heuristic initialization. It does not imply function preservation: coefficient vectors refer to different local coordinates, and the merged geometry and responsibilities also differ. The quiet-child perturbation theorem does not cover this merge rule.

Residual-guided Gaussian splitting. Complementing polynomial enrichment, splitting introduces additional localized representation capacity in unresolved regions. The number of newly created components is limited by the remaining capacity $M _ { \mathrm { m a x } } - M$ , a fractional split budget, and a hard per-adaptation cap. These constraints prevent a single adaptation event from producing a large architectural change.

Largest-scale direction. A simple geometric strategy selects a high-error parent and splits it along its largest spatial or spatiotemporal scale. Specifically,

$$
k _ { m } ^ { \star } = \arg \operatorname* { m a x } _ { q \in \{ 1 , \dots , d \} } \sigma _ { m , q } ,\tag{34}
$$

and the corresponding split direction is $\mathbf { v } _ { m } = \mathbf { e } _ { k _ { m } ^ { \star } }$ . This produces anisotropic refinement along the direction in which the parent Gaussian currently has the broadest support.

Residual-covariance direction. A more informative alternative uses the distribution of unresolved physics error to determine the split direction. In this mode, split candidates are selected using

the residual-only indicator $\eta _ { m } ^ { \mathrm { r e s } }$ . For each candidate component m, define its residual-weighted responsibility as

$$
\omega _ { i m } = \pi _ { m } ( \mathbf { z } _ { i } ) \left| r _ { \Theta } ( \mathbf { z } _ { i } ) \right| ^ { 2 } .\tag{35}
$$

The corresponding residual-weighted center and covariance are

$$
\overline { { \mathbf { z } } } _ { m } = \frac { \sum _ { i } \omega _ { i m } \mathbf { z } _ { i } } { \sum _ { i } \omega _ { i m } + \varepsilon } ,\tag{36}
$$

$$
\mathbf { C } _ { m } = \frac { \sum _ { i } \omega _ { i m } \left( \mathbf { z } _ { i } - \overline { { \mathbf { z } } } _ { m } \right) \left( \mathbf { z } _ { i } - \overline { { \mathbf { z } } } _ { m } \right) ^ { \top } } { \sum _ { i } \omega _ { i m } + \varepsilon } .\tag{37}
$$

The split direction is chosen as the principal eigenvector

$$
\mathbf { v } _ { m } \in \arg \operatorname* { m a x } _ { \| \mathbf { v } \| _ { 2 } = 1 } \mathbf { v } ^ { \top } \mathbf { C } _ { m } \mathbf { v } .\tag{38}
$$

Hence, refinement is oriented along the direction in which the responsibility-weighted residual exhibits the largest spread, rather than along a random or purely geometric direction. If the residual mass is negligible or the covariance is numerically invalid, the implementation falls back to a geometric direction, typically the largest-scale axis. When only a subset of residual points is used, these points can additionally be selected deterministically according to their residual energy and responsibility, avoiding additional randomness from point subsampling.

For either splitting strategy, we define the directional scale of the parent by

$$
s _ { m } = \sqrt { \mathbf { v } _ { m } ^ { \top } \mathrm { d i a g } \left( \pmb { \sigma } _ { m } ^ { \odot 2 } \right) \mathbf { v } _ { m } } .\tag{39}
$$

A tentative displacement is $\alpha _ { \mathrm { s p l i t } } s _ { m } \mathbf { v } _ { m }$ . To prevent the newborn component from being placed far outside the effective support of its parent, we apply the scale-normalized clamp

$$
\mathrm { C l a m p } _ { \pmb { \sigma } } ( \pmb { \delta } ) = \pmb { \delta } \operatorname* { m i n } \left\{ 1 , \frac { \delta _ { \operatorname* { m a x } } } { \lVert \pmb { \delta } \rVert _ { \pmb { \delta } } \pmb { \sigma } \rVert _ { 2 } + \varepsilon } \right\} .
$$

The final displacement is therefore

$$
\begin{array} { r } { \Delta \pmb { \mu } _ { m } = \mathrm { C l a m p } _ { \pmb { \sigma } _ { m } } \left( \alpha _ { \mathrm { s p l i t } } s _ { m } \mathbf { v } _ { m } \right) . } \end{array}\tag{40}
$$

By default, the trained parent center is used as the split anchor, $\mathbf { a } _ { m } = \pmb { \mu } _ { m }$ . The residual-weighted center $\overline { { \mathbf { z } } } _ { m }$ may alternatively be used, while constraining its displacement from the parent in normal ized parent-scale units. The child geometry is then initialized as

$$
\pmb { \mu } _ { m ^ { \prime } } = \mathbf { a } _ { m } + \Delta \pmb { \mu } _ { m } , \qquad \pmb { \sigma } _ { m ^ { \prime } } = \mathrm { c l i p } \left( \beta _ { \sigma } \pmb { \sigma } _ { m } , \sigma _ { \mathrm { m i n } } , \sigma _ { \mathrm { m a x } } \right) ,\tag{41}
$$

where $0 < \beta _ { \sigma } < 1$ and $0 < \sigma _ { \mathrm { m i n } } \le \sigma _ { \mathrm { m a x } }$ . Reducing the scale of the child enables it to specialize to a smaller unresolved region.

Near-function-preserving split initialization. A structural refinement should ideally increase future representation capacity without causing a large instantaneous perturbation to the learned solution. Simply copying the parent polynomial coefficients to a displaced and rescaled child does not achieve this, because the coefficients are defined relative to the component’s normalized local coordinates.

Let the parent and child coordinates be

$$
\pmb { \xi } _ { p } = ( \mathbf { z } - \pmb { \mu } _ { p } ) \oslash \pmb { \sigma } _ { p } , \qquad \pmb { \xi } _ { c } = ( \mathbf { z } - \pmb { \mu } _ { c } ) \oslash \pmb { \sigma } _ { c } .
$$

These coordinates satisfy the affine relation

$$
\pmb { \xi } _ { p } = \mathbf { d } + \mathbf { b } \odot \pmb { \xi } _ { c } , \qquad \mathbf { d } = \left( \pmb { \mu } _ { c } - \pmb { \mu } _ { p } \right) \oslash \pmb { \sigma } _ { p } , \qquad \mathbf { b } = \pmb { \sigma } _ { c } \oslash \pmb { \sigma } _ { p } .\tag{42}
$$

Suppose the active parent polynomial is

$$
\ell _ { p } ( \pmb { \xi } _ { p } ) = \sum _ { \pmb { \alpha } \in \mathcal { A } _ { p } } c _ { \pmb { \alpha } } ^ { p } \pmb { \xi } _ { p } ^ { \pmb { \alpha } } .
$$

Substituting Equation equation 42 and applying the multi-index binomial expansion yields the child coefficients

$$
c _ { \beta } ^ { c } = \sum _ { \alpha \in \mathcal { A } _ { p } } c _ { \alpha } ^ { p } \binom { \alpha } { \beta } \mathbf { d } ^ { \alpha - \beta } \mathbf { b } ^ { \beta } ,\tag{43}
$$

where

$$
{ \binom { \alpha } { \beta } } = \prod _ { q = 1 } ^ { d } { \binom { \alpha _ { q } } { \beta _ { q } } } ,
$$

and $\alpha \succeq \beta$ denotes component-wise inequality. For a complete monomial basis through the active degree, this translation makes the child polynomial represent the same function of the global coordinate as the parent polynomial at initialization.

Preserving the local polynomial does not, however, guarantee that the complete model prediction is exactly unchanged because the output aggregation also depends on the Gaussian responsibilities and gates. To further suppress the perturbation caused by inserting the child, the default refinement leaves the trained parent unchanged and initializes the child with a small gate probability. Let $\gamma _ { p } = \mathrm { s i g m o i d } ( g _ { p } )$ . The child probability is initialized schematically as

$$
\gamma _ { c } = \mathrm { c l i p } \left( \operatorname* { m i n } { \left\{ \rho _ { g } \gamma _ { p } , \ \gamma _ { 0 } , \ \gamma _ { \mathrm { m a x } } \right\} } , \gamma _ { \mathrm { m i n } } , \gamma _ { \mathrm { m a x } } \right) , \qquad g _ { c } = \mathrm { l o g i t } ( \gamma _ { c } ) ,\tag{44}
$$

where $\rho _ { g } \ll 1$ and $0 < \gamma _ { \mathrm { m i n } } \leq \gamma _ { \mathrm { m a x } } < 1$ . Under the normalized-mixture assumptions of Theorem B.4, the perturbation is bounded in terms of the child gate, mixture coverage, and component derivatives. The update aims for

$$
u _ { \Theta ^ { + } } ( \mathbf { z } ) \approx u _ { \Theta ^ { - } } ( \mathbf { z } )\tag{45}
$$

at the instant of insertion. Subsequent gradient updates can then gradually activate and specialize the newborn component in the unresolved region.

The implementation additionally supports a symmetric replacement strategy in which the parent and child are placed on opposite sides of the split anchor, both scales are reduced, both local polynomials are translated into their new coordinate systems, and the original gate probability is divided between them. The additive quiet-child strategy is the stated default: it preserves the trained parent parameters, and its instantaneous perturbation can be bounded under the assumptions in Theorem B.4.

## B FURTHER THEORETICAL RESULTS

We provide an a posteriori interpretation of the localized ReCAP indicators and show that the default additive Gaussian split produces a controlled instantaneous perturbation. The analysis separates three questions: whether the local scores control solution error, whether the residual-covariance direction maximizes residual-weighted projected coordinate variance, and how inserting a new component perturbs the current physics objective. The stability premise below follows the standard principle that residual accuracy implies solution accuracy only when the underlying PDE is stable in the chosen residual norm (Hu et al., 2022; Jiang et al., 2026). This premise is not automatic for arbitrary PDEs or losses (Wang et al., 2022). The full statements and proofs are given below.

Let u solve $\mathcal { L } u = f$ in $Q$ and $B u = g$ on Γ. Let $\mu _ { Q } , \mu _ { \Gamma }$ be probability measures, $\kappa _ { \Gamma } \geq 0$ , and $\varepsilon > 0$ For a candidate v, assume

$$
\begin{array} { r } { \| \boldsymbol { v } - \boldsymbol { u } \| _ { { \boldsymbol { X } } } ^ { 2 } \leq C _ { \mathrm { s t a b } } ^ { 2 } \left( \| \mathcal { L } \boldsymbol { v } - \boldsymbol { f } \| _ { L ^ { 2 } ( \mu _ { Q } ) } ^ { 2 } + \kappa _ { \Gamma } \| \mathcal { B } \boldsymbol { v } - \boldsymbol { g } \| _ { L ^ { 2 } ( \mu _ { \Gamma } ) } ^ { 2 } \right) . } \end{array}\tag{46}
$$

Let the effective Gaussian responsibilities satisfy $0 \leq \pi _ { m } \leq 1$ and $\textstyle \sum _ { m = 1 } ^ { M } \pi _ { m } = 1$ . Define

$$
s _ { m } = \int _ { Q } \pi _ { m } d \mu _ { Q } ,
$$

$$
\eta _ { m } ^ { \mathrm { r e s } } = \frac { \int _ { Q } \pi _ { m } | \boldsymbol { r } _ { \Theta } | ^ { 2 } d \mu _ { Q } } { s _ { m } + \varepsilon } ,\tag{47}
$$

$$
s _ { m } ^ { \Gamma } = \int _ { \Gamma } \pi _ { m } d \mu _ { \Gamma } ,
$$

$$
\eta _ { m } ^ { \mathrm { b d r y } } = \frac { \int _ { \Gamma } \pi _ { m } | b _ { \Theta } | ^ { 2 } d \mu _ { \Gamma } } { s _ { m } ^ { \Gamma } + \varepsilon } ,\tag{48}
$$

where $r _ { \Theta } = \mathcal { L } u _ { \Theta } - f$ and $b _ { \Theta } = B u _ { \Theta } - g$

Theorem B.1 (Mass-weighted ReCAP reliability certificate). Under equation 46 and the partitionof-unity condition,

$$
\| u _ { \Theta } - u \| _ { X } ^ { 2 } \leq C _ { \mathrm { s t a b } } ^ { 2 } \left[ \sum _ { m = 1 } ^ { M } ( s _ { m } + \varepsilon ) \eta _ { m } ^ { \mathrm { r e s } } + \kappa _ { \Gamma } \sum _ { m = 1 } ^ { M } ( s _ { m } ^ { \Gamma } + \varepsilon ) \eta _ { m } ^ { \mathrm { b d r y } } \right] .\tag{49}
$$

Moreover, freeze u<sub>Θ</sub> and draw independent certification samples $Z _ { i } \sim \mu _ { Q } , i = 1 , \ldots , N _ { c } ,$ , and $Y _ { j } \sim \mu _ { \Gamma } , j = 1 , . . . , N _ { \Gamma }$ . For $N _ { c } , \dot { N } _ { \Gamma } \ge 1$ , let the empirical masses and indicators be defined by replacing the integrals above with sample sums, and set

$$
\widehat { \mathcal { E } } _ { \mathrm { R e C A P } } ^ { 2 } = \frac { 1 } { N _ { c } } \sum _ { m } ( \widehat { s } _ { m } + \varepsilon ) \widehat { \eta } _ { m } ^ { \mathrm { r e s } } + \frac { \kappa _ { \Gamma } } { N _ { \Gamma } } \sum _ { m } ( \widehat { s } _ { m } ^ { \Gamma } + \varepsilon ) \widehat { \eta } _ { m } ^ { \mathrm { b d r y } } .\tag{50}
$$

$I f | r _ { \Theta } | \le R _ { \infty }$ on Q and $| b _ { \Theta } | \le G _ { \infty }$ on Γ, then, for $0 < \delta < 1$ , with probability at least $1 - \delta ,$

$$
\| u _ { \Theta } - u \| _ { X } \leq C _ { \mathrm { s t a b } } \left[ \widehat { \mathcal { E } } _ { \mathrm { R e C A P } } ^ { 2 } + R _ { \infty } ^ { 2 } \sqrt { \frac { \log ( 4 / \delta ) } { 2 N _ { c } } } + \kappa _ { \Gamma } G _ { \infty } ^ { 2 } \sqrt { \frac { \log ( 4 / \delta ) } { 2 N _ { \Gamma } } } \right] ^ { 1 / 2 } .\tag{51}
$$

Proof. By definition and the partition-of-unity identity,

$$
\sum _ { m } ( s _ { m } + \varepsilon ) \eta _ { m } ^ { \mathrm { r e s } } = \sum _ { m } \int _ { Q } \pi _ { m } | r _ { \Theta } | ^ { 2 } d \mu _ { Q } = \int _ { Q } \left( \sum _ { m } \pi _ { m } \right) | r _ { \Theta } | ^ { 2 } d \mu _ { Q } = \| r _ { \Theta } \| _ { L ^ { 2 } ( \mu _ { Q } ) } ^ { 2 } .\tag{52}
$$

The same calculation gives $\begin{array} { r } { \sum _ { m } ( s _ { m } ^ { \Gamma } + \varepsilon ) \eta _ { m } ^ { \mathrm { b d r y } } = \| b _ { \Theta } \| _ { L ^ { 2 } ( \mu _ { \Gamma } ) } ^ { 2 } } \end{array}$ . Substitution into equation 46 proves equation 49.

For every realization of the certification samples, the same algebra gives

$$
\frac { 1 } { N _ { c } } \sum _ { m } ( \widehat { s } _ { m } + \varepsilon ) \widehat { \eta } _ { m } ^ { \mathrm { r e s } } = \frac { 1 } { N _ { c } } \sum _ { i = 1 } ^ { N _ { c } } | r _ { \Theta } ( Z _ { i } ) | ^ { 2 } ,\tag{53}
$$

and likewise on Γ. Since the predictor is frozen before the certification samples are drawn, Hoeffding’s inequality yields, with probability at least $1 - \delta / 2$

$$
\| r _ { \Theta } \| _ { L ^ { 2 } ( \mu _ { Q } ) } ^ { 2 } \le \frac { 1 } { N _ { c } } \sum _ { i } | r _ { \Theta } ( Z _ { i } ) | ^ { 2 } + R _ { \infty } ^ { 2 } \sqrt { \frac { \log ( 4 / \delta ) } { 2 N _ { c } } } ,\tag{54}
$$

and, with probability at least $1 - \delta / 2$

$$
\| b _ { \Theta } \| _ { L ^ { 2 } ( \mu _ { \Gamma } ) } ^ { 2 } \leq \frac { 1 } { N _ { \Gamma } } \sum _ { j } | b _ { \Theta } ( Y _ { j } ) | ^ { 2 } + G _ { \infty } ^ { 2 } \sqrt { \frac { \log ( 4 / \delta ) } { 2 N _ { \Gamma } } } .\tag{55}
$$

A union bound and equation 46 prove equation 51.

If $\kappa _ { \Gamma } = 0$ , the condition terms and their sample-count requirements may be omitted.

Theorem B.1 shows that mass-weighted local indicators reconstruct the global physics loss. The unweighted component score remains useful for ranking refinements, but it is not itself a global certificate. The independent-sample requirement is essential: reusing training points requires a uniform generalization argument such as those developed for PINNs and XPINNs (Hu et al., 2022), or a continuous-domain verification method (Eiras et al., 2024).

Explicit transport stability. For the linear transport operator $\mathcal { L } = \partial _ { t } + \beta \cdot \nabla$ on a space–time cylinder $Q = ( 0 , T ) \times \Omega _ { x } , \mathrm { i e t } \Gamma _ { - } = \{ x \in \partial \Omega _ { x } : \bar { \beta ( x ) } \cdot \bar { n } ( x ) < 0 \} , \Sigma _ { - } = ( 0 , T ) \times \Gamma _ { - } ,$ , where n is the outward unit normal, and $c _ { \beta } = 1 + \| \nabla \cdot \beta \| _ { L ^ { \infty } ( \Omega _ { x } ) } . \operatorname { I f } e = u _ { \Theta } - u , r = \mathcal { L } e , e _ { 0 } = e ( 0 , \cdot )$ , and $e _ { - } = e | _ { \Sigma }$ , then

$$
\begin{array} { r } { \| e \| _ { L ^ { 2 } ( Q ) } ^ { 2 } \leq T e ^ { c _ { \beta } T } \left( \| e _ { 0 } \| _ { L ^ { 2 } ( \Omega _ { x } ) } ^ { 2 } + \| r \| _ { L ^ { 2 } ( Q ) } ^ { 2 } + \| e _ { - } \| _ { L ^ { 2 } ( \Sigma _ { - } ; | \beta \cdot n | ) } ^ { 2 } \right) . } \end{array}\tag{56}
$$

Indeed, multiplying $\partial _ { t } e + \beta \cdot \nabla e = r$ by $2 e .$ , integrating by parts, discarding the nonnegative outflow term, and applying Young’s inequality gives

$$
\frac { d } { d t } \| e ( t ) \| _ { L ^ { 2 } ( \Omega _ { x } ) } ^ { 2 } \leq c _ { \beta } \| e ( t ) \| _ { L ^ { 2 } ( \Omega _ { x } ) } ^ { 2 } + \| r ( t ) \| _ { L ^ { 2 } ( \Omega _ { x } ) } ^ { 2 } + \int _ { \Gamma _ { - } } | \beta \cdot n | e _ { - } ^ { 2 } d S .\tag{57}
$$

Gronwall’s inequality followed by integration over $t \in ( 0 , T )$ proves equation 56. The calculation assumes sufficient regularity for these derivatives and traces. The displayed estimate uses volume and flux-weighted boundary measures; normalizing these to probability measures rescales the stability constants.

Proposition B.2 (Optimal residual-covariance direction). For a selected component, define $\omega _ { i } =$ $\begin{array} { r } { \pi _ { m } ( \hat { z } _ { i } ) | r _ { \Theta } ( z _ { i } ) | ^ { 2 } , \dot { W } = \sum _ { i } \omega _ { i } > 0 , \bar { z } = W ^ { - 1 } \sum _ { i } \omega _ { i } z _ { i } , } \end{array}$ , and $\begin{array} { r } { C = W ^ { - 1 } \sum _ { i } \omega _ { i } ( z _ { i } - \bar { z } ) ( z _ { i } - \bar { z } ) ^ { \top } } \end{array}$ . If $v _ { 1 }$ is a unit eigenvector associated with $\lambda _ { \operatorname* { m a x } } ( \overline { { C } } )$ , then

$$
v _ { 1 } \in \underset { \Vert v \Vert _ { 2 } = 1 } { \arg \operatorname* { m a x } } \frac { 1 } { W } \sum _ { i } \omega _ { i } \bigl ( v ^ { \top } ( z _ { i } - \bar { z } ) \bigr ) ^ { 2 } .\tag{58}
$$

Proof. The objective equals $v ^ { \top } C v$ . Writing $\boldsymbol { C } = \boldsymbol { V } \boldsymbol { \Lambda } \boldsymbol { V } ^ { \intercal }$ and $a = V ^ { \top }$ v gives $\begin{array} { r } { \boldsymbol { v } ^ { \top } \boldsymbol { C } \boldsymbol { v } = \sum _ { k } \lambda _ { k } \boldsymbol { a } _ { k } ^ { 2 } \le } \end{array}$ $\begin{array} { r } { \lambda _ { \operatorname* { m a x } } ( C ) \sum _ { k } a _ { k } ^ { 2 } = \lambda _ { \operatorname* { m a x } } ( C ) } \end{array}$ , with equality for a leading eigenvector. □

We next quantify the stability of the default additive split. Assume one output channel has the normalized gated mixture form

$$
u ^ { - } ( z ) = \frac { \sum _ { m = 1 } ^ { M } q _ { m } ( z ) \ell _ { m } ( z ) } { D ( z ) } , \qquad q _ { m } = \gamma _ { m } \phi _ { m } , \qquad D = \sum _ { m = 1 } ^ { M } q _ { m } .\tag{59}
$$

This is the aggregation assumption for the perturbation result below. A different decoder, branch combination, or hard-constraint transformation requires bounds for the corresponding output map.

Lemma B.3 (Exact polynomial translation). Let $\xi _ { p } = ( z - \mu _ { p } ) \oslash \sigma _ { p }$ and $\xi _ { c } = ( z - \mu _ { c } ) \oslash \sigma _ { c } .$ Set $d = ( \mu _ { c } - \mu _ { p } ) \oslash \sigma _ { p }$ and $\begin{array} { r } { b = \sigma _ { c } \oslash \sigma _ { p } . \ I f \ell _ { p } = \overleftarrow { \sum } _ { | \alpha | \leq p } c _ { \alpha } ^ { p } \dot { \xi } _ { p } ^ { \alpha } } \end{array}$ and the complete monomial basis through degree p is used, then the child coefficients

$$
c _ { \beta } ^ { c } = \sum _ { | \alpha | \leq p } c _ { \alpha } ^ { p } { \binom { \alpha } { \beta } } d ^ { \alpha - \beta } b ^ { \beta }\tag{60}
$$

satisfy $\ell _ { c } ( z ) = \ell _ { p } ( z )$ for every z.

Proof. The coordinates satisfy $\xi _ { p } = d + b \odot \xi _ { c }$ . Substituting this identity into the parent polynomial and applying the multi-index binomial theorem gives

$$
\ell _ { p } = \sum _ { | \alpha | \leq p } c _ { \alpha } ^ { p } \sum _ { \beta \preceq \alpha } { \binom { \alpha } { \beta } } d ^ { \alpha - \beta } b ^ { \beta } \xi _ { c } ^ { \beta } .\tag{61}
$$

Collecting the coefficient of each $\xi _ { c } ^ { \beta }$ gives equation 60. Completeness of the basis ensures that every resulting monomial is present. 口

When component-wise local orders are used, the translation must be applied only to source coefficients whose degree is active for the parent, or all inactive stored coefficients must be zero. Otherwise an inactive higher-order coefficient can translate into an active lower-order coefficient, and exact preservation need not hold.

Theorem B.4 (Shock-controlled quiet-child refinement). Translate a new child from parent p according to Lemma $B . 3 ,$ leave the parent unchanged, and let $q _ { c } = \gamma _ { c } \phi _ { c }$ with $0 \le \gamma _ { c } \le \epsilon _ { g }$ and $\gamma _ { c }$ constant in z. Assume $\mathcal { L } = \partial _ { t } + \bar { \beta } \cdot \nabla , 0 \leq \phi _ { c } \leq 1$ on $Q \cup \Gamma$ , and that thefollowing lower bound on D holds on $Q \cup \Gamma$

$$
D \geq d _ { 0 } > 0 , \qquad \| \mathcal { L } \phi _ { c } \| _ { L ^ { \infty } ( Q ) } \leq K _ { \phi } , \qquad \| \mathcal { L } D \| _ { L ^ { \infty } ( Q ) } \leq K _ { D } .\tag{62}
$$

Set $h = \ell _ { p } - u ^ { - } , K _ { \alpha } = K _ { \phi } / d _ { 0 } + K _ { D } / d _ { 0 } ^ { 2 } ,$ and

$$
\begin{array} { r } { C _ { \mathrm { s p l i t } } = \left[ \left( K _ { \alpha } \| h \| _ { L ^ { 2 } ( Q ) } + d _ { 0 } ^ { - 1 } \| \mathcal { L } h \| _ { L ^ { 2 } ( Q ) } \right) ^ { 2 } + \kappa _ { \Gamma } d _ { 0 } ^ { - 2 } \| h \| _ { L ^ { 2 } ( \Gamma ) } ^ { 2 } \right] ^ { 1 / 2 } . } \end{array}\tag{63}
$$

$$
F o r \ : J ( v ) = \| \mathcal { L } v - f \| _ { L ^ { 2 } ( Q ) } ^ { 2 } + \kappa _ { \Gamma } \| v - g \| _ { L ^ { 2 } ( \Gamma ) } ^ { 2 } , t h e p o s t { - s p l i t } { p r e d i c t o r \ : s a t i s f i e s }
$$

$$
\sqrt { J ( u ^ { + } ) } \leq \sqrt { J ( u ^ { - } ) } + \epsilon _ { g } C _ { \mathrm { s p l i t } } ,\tag{64}
$$

$$
J ( u ^ { + } ) - J ( u ^ { - } ) \leq 2 \epsilon _ { g } C _ { \mathrm { s p l i t } } \sqrt { J ( u ^ { - } ) } + \epsilon _ { g } ^ { 2 } C _ { \mathrm { s p l i t } } ^ { 2 } .\tag{65}
$$

If the child gate can approach zero, then the closure of the refined hypothesis class contains the pre-split class and its optimal achievable loss cannot increase.

Proof. By Lemma B.3, the new child represents $\ell _ { p }$ in global coordinates. Directly from equation 59,

$$
u ^ { + } - u ^ { - } = \alpha h , \qquad \alpha = \frac { q _ { c } } { D + q _ { c } } .\tag{66}
$$

Since $q _ { c } \leq \epsilon _ { g }$ and $D \geq d _ { 0 } , 0 \leq \alpha \leq \epsilon _ { q } / d _ { 0 }$ . Moreover,

$$
\mathcal { L } \alpha = \frac { D \mathcal { L } q _ { c } - q _ { c } \mathcal { L } D } { ( D + q _ { c } ) ^ { 2 } } .\tag{67}
$$

The gate is constant with respect to z at insertion, so $\| \mathcal { L } q _ { c } \| _ { \infty } \le \epsilon _ { g } K _ { \phi }$ . Therefore

$$
\| \mathcal { L } \alpha \| _ { \infty } \le \epsilon _ { g } \left( \frac { K _ { \phi } } { d _ { 0 } } + \frac { K _ { D } } { d _ { 0 } ^ { 2 } } \right) = \epsilon _ { g } K _ { \alpha } .\tag{68}
$$

Applying the product rule to equation 66 gives

$$
\begin{array} { r } { \| \mathcal { L } ( u ^ { + } - u ^ { - } ) \| _ { L ^ { 2 } ( Q ) } \leq \epsilon _ { g } \left( K _ { \alpha } \| h \| _ { L ^ { 2 } ( Q ) } + d _ { 0 } ^ { - 1 } \| \mathcal { L } h \| _ { L ^ { 2 } ( Q ) } \right) , } \end{array}\tag{69}
$$

$$
\| u ^ { + } - u ^ { - } \| _ { L ^ { 2 } ( \Gamma ) } \leq \epsilon _ { g } d _ { 0 } ^ { - 1 } \| h \| _ { L ^ { 2 } ( \Gamma ) } .\tag{70}
$$

Thus the product norm of the residual and boundary perturbations is at most $\epsilon _ { g } C _ { \mathrm { s p l i t } }$ . Minkowski’s inequality proves equation 64; squaring it proves equation 65. Finally, retaining every old parameter and sending the child gate logit to −∞ makes $q _ { c } \to 0 ,$ , so every pre-split predictor lies in the closure of the refined class. □

Theorem B.4 applies to the additive, parent-preserving split. It does not directly cover the optional symmetric replacement strategy. Likewise, pruning and merging require separate perturbation guards. For local order enrichment, setting every newly activated polynomial coefficient to zero makes the insertion exactly function-preserving and enlarges the hypothesis class.

## C PDE BENCHMARKS

We specify the governing equations, domains, and reference solutions or initial data for the five benchmarks. The suffix 3D or 4D counts all input coordinates, including time where applicable.

## C.1 (2+1)D FLOW MIXING

FM3D is a linear transport benchmark for the advection and rotational mixing of a passive scalar in a two-dimensional velocity field:

$$
\begin{array} { r l r } & { u _ { t } + a ( x , y ) u _ { x } + b ( x , y ) u _ { y } = 0 , \quad ( x , y , t ) \in [ - 4 , 4 ] ^ { 2 } \times [ 0 , 4 ] , } & \\ & { a ( x , y ) = - \displaystyle \frac { v _ { \mathrm { t a n } } ( r ) } { v _ { \mathrm { m a x } } } \frac { y } { r } , \qquad b ( x , y ) = \displaystyle \frac { v _ { \mathrm { t a n } } ( r ) } { v _ { \mathrm { m a x } } } \frac { x } { r } , } & \\ & { v _ { \mathrm { t a n } } ( r ) = \mathrm { s e c h } ^ { 2 } ( r ) \operatorname { t a n h } ( r ) , \qquad r = \sqrt { x ^ { 2 } + y ^ { 2 } } , \qquad v _ { \mathrm { m a x } } = 0 . 3 8 5 . } & \end{array}\tag{71}
$$

The analytic solution is

$$
u ^ { \star } ( x , y , t ) = - \operatorname { t a n h } \left[ \frac { y } { 2 } \cos ( \omega ( r ) t ) - \frac { x } { 2 } \sin ( \omega ( r ) t ) \right] , \qquad \omega ( r ) = \frac { v _ { \mathrm { t a n } } ( r ) } { r v _ { \mathrm { m a x } } } ,\tag{72}
$$

with initial condition $u ( x , y , 0 ) = - \operatorname { t a n h } ( y / 2 )$ (Tamamidis & Assanis, 1993). The apparent singularities at $r = 0$ are removable: $a ( 0 , 0 ) = \dot { b } ( 0 , 0 ) = 0$ and $\omega ( 0 ) = 1 / v _ { \mathrm { m a x } }$ . These values follow from $v _ { \mathrm { t a n } } ( r ) / r \to 1 { \mathrm { a s } } r \to 0$

## C.2 3D HELMHOLTZ EQUATION

H3D is a stationary elliptic benchmark on $\Omega _ { x } = [ - 1 , 1 ] ^ { 3 }$

$$
\Delta _ { \bf x } u + k ^ { 2 } u = q \quad \mathrm { i n } \Omega _ { x } , \qquad u = 0 \quad \mathrm { o n } \partial \Omega _ { x } .\tag{73}
$$

The manufactured solution and source are

$$
u ^ { \star } ( { \bf x } ) = \prod _ { i = 1 } ^ { 3 } \sin ( a _ { i } \pi x _ { i } ) , \qquad q ( { \bf x } ) = \left[ k ^ { 2 } - \pi ^ { 2 } \sum _ { i = 1 } ^ { 3 } a _ { i } ^ { 2 } \right] u ^ { \star } ( { \bf x } ) ,\tag{74}
$$

where $k = 1$ and $( a _ { 1 } , a _ { 2 } , a _ { 3 } ) = ( 4 , 4 , 3 )$ . Thus $q = ( 1 - 4 1 \pi ^ { 2 } ) u ^ { \star }$ , and the prescribed homogeneous boundary condition is satisfied exactly by $u ^ { \star }$

## C.3 (2+1)D NONLINEAR DIFFUSION

ND3D describes solution-dependent diffusion in two spatial dimensions and time:

$$
\begin{array} { r l } & { u _ { t } = 0 . 0 5 \left( \lVert \nabla _ { \mathbf { x } } u \rVert _ { 2 } ^ { 2 } + u \Delta _ { \mathbf { x } } u \right) , \quad ( x , y , t ) \in [ - 1 , 1 ] ^ { 2 } \times [ 0 , 1 ] , } \\ & { u ( \mathbf { x } , 0 ) = 0 . 2 5 g \bigg ( \mathbf { x } ; 0 . 2 , 0 . 3 , \frac { 1 } { \sqrt { 1 0 } } \bigg ) + 0 . 4 g \bigg ( \mathbf { x } ; - 0 . 1 , - 0 . 5 , \frac { 1 } { \sqrt { 1 5 } } \bigg ) } \\ & { \qquad + 0 . 3 g \bigg ( \mathbf { x } ; - 0 . 5 , 0 , \frac { 1 } { \sqrt { 2 0 } } \bigg ) , } \end{array}\tag{75}
$$

where $\mathbf { x } = ( x , y )$ and $g ( \mathbf { x } ; a , b , \sigma ) = \exp \bigl ( - [ ( x - a ) ^ { 2 } + ( y - b ) ^ { 2 } ] / \sigma ^ { 2 } \bigr )$ . The gradient and Laplacian act only on the spatial coordinates. The initial condition contains three localized peaks with different amplitudes, widths, and locations.

## C.4 KLEIN–GORDON EQUATIONS

For KG3D and KG4D, let $d _ { s } \in \{ 2 , 3 \}$ be the number of spatial coordinates, $\Omega _ { x } = [ - 1 , 1 ] ^ { d _ { s } }$ , and $t \in [ 0 , 1 0 ]$ . The total input dimension used in Section 3 is $\bar { d } = d _ { s } + 1$ . Define

$$
S ( \mathbf { x } ) = \sum _ { i = 1 } ^ { d _ { s } } x _ { i } , \qquad P ( \mathbf { x } ) = \prod _ { i = 1 } ^ { d _ { s } } x _ { i } , \qquad u ^ { \star } ( \mathbf { x } , t ) = S ( \mathbf { x } ) \cos ( 2 t ) + P ( \mathbf { x } ) \sin ( 2 t ) .\tag{76}
$$

The corresponding manufactured problem is

$$
\begin{array} { r l r } & { u _ { t t } - \Delta _ { \mathbf { x } } u + u ^ { 2 } = f , } & { ( \mathbf { x } , t ) \in \Omega _ { x } \times ( 0 , 1 0 ] , } \\ & { u ( \mathbf { x } , 0 ) = S ( \mathbf { x } ) , \qquad u _ { t } ( \mathbf { x } , 0 ) = 2 P ( \mathbf { x } ) , } & { \mathbf { x } \in \Omega _ { x } , } \\ & { u ( \mathbf { x } , t ) = u ^ { \star } ( \mathbf { x } , t ) , } & { ( \mathbf { x } , t ) \in \partial \Omega _ { x } \times [ 0 , 1 0 ] , } \end{array}\tag{77}
$$

where $f = ( u ^ { \star } ) ^ { 2 } - 4 u ^ { \star }$ , since $u _ { t t } ^ { \star } = - 4 u ^ { \star }$ and $\Delta _ { \mathbf { x } } u ^ { \star } = 0$ . Thus $d _ { s } = 2$ gives the stated KG3D solution and $d _ { s } = 3$ gives KG4D. The initial-velocity condition follows by differentiating the manufactured solution; it is part of the complete second-order-in-time problem specification.

## D EXPERIMENTAL SETTINGS

In this section, we summarize the experimental configurations used for all benchmarks. The key hyperparameters, including the number of training epochs and other model-specific settings, are reported in Table 2.

Table 2: Key hyperparameters for the benchmarks.
<table><tr><td>Hyperparameter</td><td>FM3D</td><td>H3D</td><td>ND3D</td><td>KG3D</td><td>KG4D</td></tr><tr><td>Training epochs Learning rate</td><td>80,000 5 × 10 -3</td><td>50,000 1 × 10−3</td><td>100,000 2 × 10−3</td><td>50,000 5 × 10−3</td><td>100,000 1 × 10−3</td></tr><tr><td>Batch size</td><td>1,024</td><td>1,024</td><td>1,024</td><td>1,024</td><td>512</td></tr><tr><td>Hidden features</td><td>48</td><td>16</td><td>32</td><td>48</td><td>128</td></tr><tr><td>Initial number of Gaussians</td><td>2,000</td><td>3,000</td><td>600</td><td>3,000</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>2,000</td></tr><tr><td>Minimum/maximum Gaussians</td><td>600/4,000</td><td>600/5,000</td><td>300/1,000</td><td>600/4,000</td><td>200/2,000</td></tr><tr><td>Polynomial degree</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td></tr><tr><td>Gaussian scale parameter</td><td>0.15</td><td>0.15</td><td>0.20</td><td>0.15</td><td>0.20</td></tr><tr><td>Adaptation start</td><td>3,000</td><td>1,000</td><td>3,000</td><td>2,000</td><td>1,000</td></tr><tr><td>Adaptation interval</td><td>2,000</td><td>2,000</td><td>2,000</td><td>3,000</td><td>2,000</td></tr><tr><td>Split fraction</td><td>0.02</td><td>0.03</td><td>0.03</td><td>0.015</td><td>0.03</td></tr><tr><td>Adaptive sampling interval</td><td>3,000</td><td>3,000</td><td>3,000</td><td>4,000</td><td>3,000</td></tr><tr><td>Tsplit</td><td>1.3</td><td>1.3</td><td>1.3</td><td>1.5</td><td>1.3</td></tr><tr><td>Tmerge</td><td>0.7</td><td>0.5</td><td>0.5</td><td>0.4</td><td>0.5</td></tr></table>

## E VISUALIZATION

In this section, we provide qualitative visualizations of the predicted solutions and corresponding absolute-error distributions on representative PDE benchmarks, with comparisons against reference solutions and selected baselines in Figures 4, 5, and 6.

For FM3D, ReCAP accurately captures the evolving rotational mixing structures over time and closely follows the reference solution. The corresponding error maps show that the remaining errors are mainly localized around the increasingly complex spiral interfaces, while ReCAP maintains lower and more concentrated errors than the representative baselines.

For H3D and KG3D, ReCAP also reproduces the dominant spatial patterns across different slices and time instances. In both cases, the predicted solutions remain close to the reference fields, and the absolute-error maps indicate reduced deviations compared with SPINN and PirateNets, supporting its accuracy on the tested equations.

![](images/444c3b6e1c9cf61ae4a1d31aafde5fdf0c8be375b7e410f69e32d29abe58c9a7.jpg)  
Figure 4: FM3D solution and absolute-error snapshots over the evaluated time points, comparing the reference field with ReCAP and representative baselines.

![](images/22210bbdd5052b05e25a42b7123328b7689ea35d9e9bc1ed3a87a901c03abfc6.jpg)  
Figure 5: H3D reference solution, model predictions, and absolute-error maps on representative two-dimensional slices.

![](images/fac31b8be51c461b92de57334466f7010ed9a5008031cc4738fbf4c3d064a414.jpg)  
Figure 6: KG3D reference solution, model predictions, and absolute-error maps on representative two-dimensional slices.

## F COMPUTATIONAL COST

Figure 7 compares reported floating-point operation counts on KG3D. ReCAP uses 0.53 GFLOPs in this comparison, matching SINN and falling below ActNet (1.64), PirateNets (2.59), and CoupledNet (4.38), while Scale-PINN uses 0.27 GFLOPs. These operation counts do not establish wall-clock training-time savings.

KG3D: Computational cost comparison

![](images/30fad6860f0e0402a2f7bb1880fa486f6bbdb3013a8d2469c46bdb633770d282.jpg)  
Figure 7: Reported computational cost on KG3D in GFLOPs (lower is better). ReCAP has the same reported operation count as SINN; Scale-PINN has the lowest count among the displayed methods.