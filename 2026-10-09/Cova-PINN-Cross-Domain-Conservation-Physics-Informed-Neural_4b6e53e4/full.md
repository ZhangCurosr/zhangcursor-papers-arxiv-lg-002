# Cova-PINN: Cross-Domain Conservation Physics-Informed Neural Network for Fluid-Solid Conjugate Heat Transfer in Complex Geometries

Weizheng Zhang<sup>∗1</sup>, Xunjie Xie<sup>∗1</sup>, Hao Pan<sup>2</sup>, and Lin Lu<sup>1</sup>

<sup>1</sup>Shandong University <sup>2</sup>Tsinghua University

## Abstract

Multi-domain physics-informed neural networks (PINNs) flexibly model medium-specific representations to solve fluid– solid conjugate heat transfer (CHT). However, standard multidomain PINNs enforce governing equations and interface conditions on separately sampled domain supports, which can yield plausible temperature fields but inaccurate end-to-end energy transfer and outlet temperatures. We propose Cova-PINN, a multi-domain PINN framework that aligns conservation support with thermal interaction paths in complex geometries. Cova-PINN jointly optimizes cross-domain composite control-volume balances at the local scale and paired-wall closure at the global exchanger scale. We evaluate Cova-PINN on four triply periodic minimal surface (TPMS) heat exchangers and a geometrically distinct DualMS design against CHT specific, optimization-oriented, and complex-geometry PINN baselines under a common protocol. Relative to the closest baseline, MUSA-PINN-CHT, Cova-PINN reduces average outlet-temperature and device-level closure errors across the four TPMS topologies by 37.7% and 60.2%, respectively, while also improving full-field and heat-duty accuracy, with consistent gains on DualMS.

## 1 Introduction

Physics-informed machine learning turns governing equations and boundary conditions into objectives for learning PDE-governed fields with limited high-fidelity supervision (Raissi, Perdikaris, and Karniadakis 2019; Karniadakis et al. 2021). PINNs are a widely used realization, and multiphysics variants often assign separate subnetworks to diferent media or state variables (Jagtap, Kharazmi, and Karniadakis 2020; Jagtap and Karniadakis 2020; Lee et al. 2025). This modularity supports medium-specific representations, but local losses on separate spatial supports need not make the learned fields globally consistent—especially when a conserved quantity crosses several learned domains.

Conjugate heat transfer (CHT) makes this issue precisely diagnosable: at steady state without internal heat generation, heat released by the hot stream must cross the solid and be equal to that absorbed by the cold stream. Exact fields inherit this device-scale balance from the local equations and interface conditions, whereas PINNs optimize finite, nonconvex objectives with documented training pathologies (Krishnapriyan et al. 2021; Wang, Yu, and Perdikaris 2022; Hao et al. 2024). Because these objectives do not directly constrain outlet bulk temperatures or paired heat rates, plausible fields may still misrepresent exchanger-level transfer. We call this separation the cross-domain conservation gap. It is especially consequential in complex exchangers with tortuous channels and distributed fluid–solid interaction paths.

Prior work improves domain decomposition and weak enforcement (Jagtap, Kharazmi, and Karniadakis 2020; Jagtap and Karniadakis 2020; Kharazmi, Zhang, and Karniadakis 2021), multi-network CHT training (Lee et al. 2025), pointwise optimization (Wu et al. 2024; Duan et al. 2025), and conservation in complex internal flows (Zhang et al. 2026). However, these methods do not explicitly align thermal conservation support with the complete hot-fluid–solid–coldfluid path. The key design question is therefore not only whether to impose an integral constraint, but where its support lies relative to the interaction topology.

We introduce Cova-PINN, a five-network multi-domain framework whose conservation support follows this interaction path. After independently training and freezing the two MUSA-PINN flow networks (Zhang et al. 2026), Cova-PINN jointly optimizes local, wall-centered composite-controlvolume balances and global paired-wall closure between the solid-side heat rates. A solid-mean temperature regularizer conditions early optimization and then decays to zero. Figure 1 summarizes the framework.

Our contributions are threefold:

• We demonstrate that domain-wise pointwise training can yield inaccurate exchanger-level energy transfer even when it reconstructs plausible temperature fields.

• We introduce a support-aligned global–local conservation co-optimization framework: cross-domain control volumes couple finite-region transfer, paired-wall closure coordinates exchanger-scale transfer, and annealed solidmean regularization conditions early optimization.

• Across triply periodic minimal surface (TPMS) topologies and DualMS (Yeranee and Rao 2022; Zhang et al. 2025), Cova-PINN reduces field, outlet, heat-duty, and closure errors; ablations support all three components.

![](images/73b2cc2604a73b3f22f89e3ac8e7dbd2131a96cf430340cf9e545f09c082944d.jpg)  
Figure 1: Overview of Cova-PINN. (a) Energy follows the hot-fluid–solid–cold-fluid path, while separately supported pointwise residuals need not close its device-scale balance. (b) After independently training and freezing the two MUSA-PINN flow networks, the three temperature networks are jointly optimized by pointwise CHT physics and coordinated local–global conservation objectives, with annealed solid-mean regularization during early training.

## 2 Related Work

Multi-domain PINNs for conjugate heat transfer. Multinetwork physics-informed models arise naturally when equations or material properties difer across subdomains. cPINN, XPINN, and FBPINN distribute a PDE solution across networks coupled through interfaces or overlapping supports (Jagtap, Kharazmi, and Karniadakis 2020; Jagtap and Karniadakis 2020; Moseley, Markham, and Nissen-Meyer 2023). More recently, MDPINN-GD partitions complex three-dimensional intersecting channels into regular subdomains and supplements their coupling with global inlet– outlet mass-flow conservation (Hwang 2025). These methods demonstrate the complementary value of local decomposition and global constraints within a single physical domain. Multi-medium thermal models instead associate networks with material regions: M-PINN treats heterogeneous thermal layers (Zhang et al. 2022), while I-PINNs and IG-PINNs target sharp interfaces and discontinuous coeficients (Sarma et al. 2024; Zheng, Huang, and Yi 2025). For CHT, modular PINNs have been applied to finned heat sinks and manifold microchannels (Lu et al. 2024; Zhang, Tu, and Yan 2024), and E-MPINN combines medium-specific subnetworks with physics-guided training (Lee et al. 2025). We therefore select E-MPINN as the task-specific multi-network CHT baseline. Cova-PINN further aligns local conservation support with the complete fluid–solid–fluid interaction path and imposes paired-wall closure at the exchanger scale.

Weak-form and conservation-aware physics-informed solvers. A second line of research reformulates physics supervision over finite regions rather than enforcing diferential residuals only at individual points. The Deep Ritz method minimizes variational energy functionals (E and Yu 2018), weak adversarial networks enforce weak formulations using learned test functions (Zang et al. 2020), and hp-VPINNs impose piecewise variational residuals over decomposed subdomains (Kharazmi, Zhang, and Karniadakis 2019, 2021). More closely related to our setting, control-volume and finitevolume PINNs integrate governing equations over local cells and enforce the resulting flux balances on their boundaries, thereby strengthening finite-region conservation and reducing reliance on high-order pointwise derivatives (Wei et al. 2025a,b). These methods establish integral residuals and control-volume constraints as efective forms of physics supervision. A less explored question in multi-network multiphysics learning is how the support of such constraints should be chosen relative to the topology of physical interaction. Existing formulations typically apply weak constraints within a single PDE domain or according to a predefined geometric decomposition. In contrast, Cova-PINN places wall-centered composite supports across adjoining material regions along the hot-fluid–solid–cold-fluid interaction path, combines the intersected media in a shared local energy balance, and complements these local constraints with device-scale heat-rate closure across the paired heat-transfer walls.

PINN optimization in complex geometries. Even correctly specified continuous physics may be poorly realized by finite, nonconvex PINN training because of stif gradient competition, uneven residual learning, and samplingdependent failure modes (Wang, Teng, and Perdikaris 2021; Krishnapriyan et al. 2021; Wang, Yu, and Perdikaris 2022; Hao et al. 2024). RoPINN expands pointwise optimization to calibrated neighborhoods, whereas CoPINN orders samples from easy to hard using a learned dificulty measure (Wu et al. 2024; Duan et al. 2025); we include them as representative recent PINN optimization baselines. These methods improve how local objectives are sampled or optimized, but do not change which physical domains a conservation law couples. For tortuous internal flows, MUSA-PINN places multiscale weak-form control volumes along complex transport pathways (Zhang et al. 2026). We use it as the complex-geometry flow baseline and closest structural comparator, while Cova-PINN extends conservation supervision from separate fluid domains to the coupled thermal interaction path. Taken together, prior work has advanced heterogeneous representation, interface transmission, integral conservation, and trainability in complex domains. Our work addresses their intersection by aligning conservation support with a multi-domain thermal interaction path.

## 3 Problem Formulation

## 3.1 Physical Setting and Energy Transfer

Figure 1(a) shows the physical setting and introduces the notation used below. The heat exchanger contains a hot-fluid region $\Omega _ { h } ,$ , a separating solid $\Omega _ { s }$ , and a cold-fluid region $\Omega _ { c } .$ Their two contact surfaces are denoted by $\Gamma _ { h s }$ (hot fluid– solid) and $\Gamma _ { s c }$ (solid–cold fluid). These labels are needed because the three regions are represented by diferent temperature networks, while heat must pass consistently across both interfaces.

The same figure also shows the resulting energy-transfer path: heat leaves $\Omega _ { h }$ , crosses $\Gamma _ { h s } ,$ conducts through $\Omega _ { s } ,$ crosses $\Gamma _ { s c } .$ , and enters $\Omega _ { c }$ . Thus, one physical process spans three separately learned domains. We consider steady forced convection with constant material properties, negligible buoyancy, and no internal heat generation; both streams use the same fluid.

We use dimensionless variables to keep the formulation compact. Temperature is scaled so that the hot and cold inlets have values 1 and $0 ,$ respectively. For a fluid stream $f \in \{ h , c \}$ , the Péclet number $P e _ { f }$ measures advection relative to thermal difusion, while $\kappa _ { s }$ is the solid-to-fluid conductivity ratio. The complete nondimensionalization, material parameters, and boundary conditions are given in $\mathsf { A p - }$ pendix A.

With no internal heat generation, the dimensionless energy equations are

$$
\mathbf { u } _ { f } \cdot \nabla T _ { f } - \frac { 1 } { P e _ { f } } \nabla ^ { 2 } T _ { f } = 0 , \quad f \in \{ h , c \} , \qquad \nabla ^ { 2 } T _ { s } = 0 .\tag{1}
$$

The first equation describes convection and difusion in each fluid, and the second describes conduction in the solid. Because our method enforces energy balance over finite regions, we next require a quantity whose boundary integral measures the net energy crossing such a region. We therefore define the dimensionless energy-flux density

$$
\begin{array} { l l } { { \bf J } _ { f } = P e _ { f } { \bf u } _ { f } T _ { f } - \nabla T _ { f } , } & { f \in \{ h , c \} , } \\ { { \bf J } _ { s } = - \kappa _ { s } \nabla T _ { s } , } & { \nabla \cdot { \bf J } _ { m } = 0 , \quad m \in \{ h , s , c \} . } \end{array}\tag{2}
$$

Here $\mathbf { J } _ { f }$ combines convective and conductive transport in a fluid, whereas ${ \bf J } _ { s }$ contains only solid conduction. All three fluxes use a common scale, so their surface integrals can be added directly in the cross-domain control volumes introduced in Sec. 4.

Let $\mathbf { n } _ { f }$ and $\mathbf { n } _ { s }$ point outward from the adjoining fluid and solid regions. Conjugate heat transfer requires

$$
\begin{array} { r l } { \mathbf { u } _ { f } = \mathbf { 0 } , } & { \quad T _ { f } = T _ { s } , } \\ { \mathbf { J } _ { f } \cdot \mathbf { n } _ { f } + \mathbf { J } _ { s } \cdot \mathbf { n } _ { s } = 0 } & { \quad \mathrm { o n ~ } \Gamma _ { h s } \mathrm { ~ a n d ~ } \Gamma _ { s c } , } \end{array}\tag{3}
$$

These three conditions impose no slip, temperature continuity, and normal heat-flux continuity, respectively. Conditions on the remaining boundaries are tabulated in Appendix A.

## 3.2 Sequential Multi-Network Learning Setting

We represent the two flow fields and three temperature fields with independent neural networks:

$$
\begin{array} { r l } & { ( \mathbf { u } _ { f } , p _ { f } ) = { \mathcal { N } } _ { f } ^ { \mathrm { H o w } } ( \mathbf { x } ; \psi _ { f } ) , \quad f \in \{ h , c \} , } \\ & { \quad \quad T _ { m } = { \mathcal { N } } _ { m } ^ { \mathrm { t e m p } } ( \mathbf { x } ; \phi _ { m } ) , ~ m \in \{ h , s , c \} . } \end{array}\tag{4}
$$

Let $\phi : = ( \phi _ { h } , \phi _ { s } , \phi _ { c } )$ collect the parameters of the three temperature networks. The flow networks are trained first, after which their velocity predictions $\bar { \mathbf { u } } _ { h }$ and $\bar { \mathbf { u } } _ { c }$ are frozen. The thermal stage therefore solves a sequential CHT problem with fixed advection rather than a bidirectionally coupled thermofluid problem. The flow-stage PDE, boundary, and complete MUSA objectives are defined in Appendix F, Eqs. (40)–(42).

The standard pointwise thermal objective penalizes Eq. (1), the thermal boundary conditions, and the two thermal interface conditions in Eq. (3):

$$
\mathcal { L } _ { \mathrm { p t } } = \lambda _ { \mathrm { P D E } } \mathcal { L } _ { \mathrm { P D E } } ^ { T } + \lambda _ { \mathrm { b c } } \mathcal { L } _ { \mathrm { b c } } ^ { T } + \lambda _ { \Gamma , T } \mathcal { L } _ { \Gamma , T } + \lambda _ { \Gamma , q } \mathcal { L } _ { \Gamma , q } .\tag{5}
$$

Here $\mathcal { L } _ { \mathrm { P D E } } ^ { T } , \mathcal { L } _ { \mathrm { b c } } ^ { T } , \mathcal { L } _ { \Gamma , T }$ , and $\mathcal { L } _ { \Gamma , q }$ are empirical mean-squared residuals for the energy equations, thermal boundary conditions, interface temperature continuity, and interface normalflux continuity, respectively. Each domain or boundary component is averaged over its own sample set before weighting. Their complete samplewise definitions are given in $\mathsf { A p - }$ pendix F, Eqs. (43)–(48); network and optimization settings and loss weights are listed in Tables 8 and 9. This common pointwise thermal objective underlies every CHT method in our comparison.

Because these residuals are evaluated on separate domainand interface-specific sample sets, no individual pointwise residual directly spans the complete transfer path in Fig. 1(a).

## 3.3 End-to-End Conservation Requirement

For the steady exchanger considered here, whose exterior solid boundary is adiabatic and whose domains contain no internal heat generation, the heat released by the hot stream must equal the heat absorbed by the cold stream. Let $Q _ { h }$ and $Q _ { c }$ denote these heat rates, inferred from the inlet and outlet bulk temperatures and normalized by $Q _ { 0 } = C _ { \operatorname* { m i n } } \Delta T$ , where $C _ { \mathrm { m i n } } = \operatorname* { m i n } ( \dot { m } _ { h } c _ { p , f } , \dot { m } _ { c } c _ { p , f } )$ . The device-level balance is

$$
Q _ { h } = Q _ { c } , \qquad G _ { \mathrm { g l o b a l } } = | Q _ { h } - Q _ { c } | = 0 .\tag{6}
$$

We use three related but distinct levels of energy notation. The flux density $\mathbf { J } _ { m }$ enters finite-region control-volume balances. The solid-side wall integrals $\widehat { Q } _ { h  s }$ and $\widehat { Q } _ { s  c } ,$ normalized by $Q _ { \mathrm { r e f } }$ , form the Global training objective in Sec. 4.

By contrast, $Q _ { h }$ and $Q _ { c }$ are inlet–outlet heat rates normalized by $Q _ { 0 }$ and are used only for device-level evaluation. Their complete bulk-temperature definitions are given in $\mathsf { A p - }$ pendix G.

Integrating the local energy equations over the three media and applying interface flux continuity yields Eq. (6). The finite-sample multi-network objective in Eq. (5), however, evaluates domain and interface residuals on separate spatial supports and does not explicitly optimize this cross-domain integral relation. Therefore, minimizing the pointwise objective does not by itself guarantee device-level energy conservation, i.e.:

$$
{ \mathcal { L } } _ { \mathrm { p t } } ( \phi ) \to 0 \quad \not \Rightarrow \quad G _ { \mathrm { g l o b a l } } ( \phi ) \to 0 .\tag{7}
$$

This separation can manifest as inaccurate outlet bulk temperatures and nonzero $G _ { \mathrm { g l o b a l } }$ . We call it the cross-domain conservation gap. Section 4 turns the corresponding finiteregion and device-level conservation relations into optimization objectives.

## 4 Cova-PINN

## 4.1 Overview and Global–Local Co-Optimization

Figure 1(b) summarizes the two-stage procedure. Under the temperature-independent flow assumption in Sec. 3.2, Stage 1 trains the two flow networks with MUSA-PINN (Zhang et al. 2026) and freezes $\bar { \mathbf { u } } _ { h }$ and $\bar { \bf u } _ { c } .$ Stage 2 updates only the thermal parameters $\phi$ and retains the standard energy equations, boundary conditions, and interface transmission in the Pointwise block. On this basis, Cova-PINN performs global–local conservation co-optimization: Local couples the temperature networks through shared finite-region balances along the thermal interaction path, while Global closes the accumulated heat transfer at the exchanger scale. $R e g$ supports this two-scale objective by anchoring the soliddomain mean temperature against early drift toward implausible levels; its weight then vanishes so that the final solution is governed by the CHT and conservation objectives.

## 4.2 Local Cross-Domain Composite Control Volumes

The Local module addresses a limitation of $\mathcal { L } _ { \mathrm { { p t } } } \mathrm { { : } }$ its domain equations and pairwise interface conditions are evaluated on separate supports and therefore provide no shared finite-region balance over the interacting media. We construct such supports by sampling $N _ { \mathrm { c v } }$ spherical parent regions $B _ { j } = \bar { B ( \mathbf { c } _ { j } , r ) , j } = \bar { 1 } , \bar { . . . } , N _ { \mathrm { c v } }$ , with centers $\mathbf { c } _ { j }$ on the heat-transfer walls $\Gamma _ { \mathrm { w } } : = \Gamma _ { h s } \cup \Gamma _ { s c } .$ The radius is selected relative to the nominal wall thickness so that these wall-centered supports extend into adjoining material regions along the thermal interaction path. Intersections define $\bar { V } _ { m , j } = \bar { B _ { j } } \cap \Omega _ { m }$ and the composite volume $V _ { j } = B _ { j } \cap \Omega$ whose outer boundary is partitioned into material-labeled patches $S _ { m , j } ~ = ~ \partial \bar { V _ { j } } \cap \bar { \Omega } _ { m }$ . Spherical normals are analytic, enabling direct area-uniform flux quadrature; the implemented radius, boundary handling, and sampling details appear in Appendix C.

Using the common-scale energy fluxes in Eq. (2), integration over the material portions gives the compact balance

$$
\begin{array} { r l } & { 0 = \displaystyle \sum _ { m \in \{ h , s , c \} } \displaystyle \int _ { V _ { m , j } } \nabla \cdot \mathbf { J } _ { m } \mathrm { d } V } \\ & { \stackrel { \mathrm { G a u s s } } { = } \displaystyle \sum _ { m \in \{ h , s , c \} } \displaystyle \int _ { \partial V _ { m , j } } \mathbf { J } _ { m } \cdot \mathbf { n } _ { m , j } \mathrm { d } A } \\ & { \quad \mathrm { i n t e r f a c e ~ } \displaystyle \underline { { \underline { { \mathrm { c o } } } } } \mathrm { s e r v a i o n } \displaystyle \sum _ { m \in \{ h , s , c \} } \displaystyle \int _ { S _ { m , j } } \mathbf { J } _ { m } \cdot \mathbf { n } _ { j } \mathrm { d } A = : R _ { j } ^ { \mathrm { c v } } ( \phi ) . } \end{array}\tag{8}
$$

No penetration removes advective interface transfer, while normal-flux continuity cancels internal interface contributions. Thus $\mathcal { L } _ { \mathrm { l o c a l } }$ uses only the material-labeled outer patches; pointwise losses retain temperature and flux transmission.

Each surface integral is estimated by an area-uniform sample mean times the component area, without local-area weights. With $\widehat { R } _ { j } ^ { \mathrm { c v } }$ denoting the fixed-scale estimator,

$$
\mathcal { L } _ { \mathrm { l o c a l } } = \frac { 1 } { N _ { \mathrm { c v } } } \sum _ { j = 1 } ^ { N _ { \mathrm { c v } } } \left( \widehat { R } _ { j } ^ { \mathrm { c v } } \right) ^ { 2 } .\tag{9}
$$

Each composite residual aggregates flux contributions from all material regions intersected by its support and jointly updates their corresponding temperature networks. Across the wall-centered support set, these coupled local updates connect the hot-fluid–solid–cold-fluid interaction path. Appendix C gives the complete estimator.

## 4.3 Global Paired-Wall Heat-Rate Closure

The Local module coordinates cross-domain transfer over finite neighborhoods, but these local balances do not directly close the total heat entering and leaving the solid over the complete exchanger. The Global module supplies this complementary device-scale constraint by pairing $\Gamma _ { h s }$ and $\Gamma _ { s c }$ and evaluating both heat rates from the solid. Using $\mathbf J _ { s } = - \kappa _ { s } \nabla T _ { s }$ from Eq. (2), with $\mathbf { n } _ { s } ^ { h }$ and $\mathbf { n } _ { s } ^ { c }$ pointing from the solid toward the hot and cold fluids, the positive entering and leaving rates are

$$
\widehat { Q } _ { h  s } ( \phi _ { s } ) = - \int _ { \Gamma _ { h s } } \mathbf { J } _ { s } { \cdot } \mathbf { n } _ { s } ^ { h } \mathrm { d } A ,\tag{10}
$$

$$
\widehat { Q } _ { s  c } ( \phi _ { s } ) = \int _ { \Gamma _ { s c } } \mathbf { J } _ { s } { \cdot } \mathbf { n } _ { s } ^ { c } \mathrm { d } A .\tag{11}
$$

Steady energy balance requires these rates to match. Evaluating both from solid-side fluxes provides one accounting domain across the paired walls.

The corresponding residual and loss are

$$
R ^ { \mathrm { w a l l } } ( \phi _ { s } ) = \widehat { Q } _ { h \to s } - \widehat { Q } _ { s \to c } ,\tag{12}
$$

$$
\begin{array} { r } { { \mathcal L } _ { \mathrm { g l o b a l } } = \left( R ^ { \mathrm { w a l l } } ( \phi _ { s } ) \right) ^ { 2 } . } \end{array}\tag{13}
$$

The loss directly updates $\phi _ { s } ;$ pointwise normal-flux conditions in Eq. (3) transmit its efect to the fluid temperature networks. The inlet–outlet gap $G _ { \mathrm { g l o b a l } }$ , reported as $E _ { \mathrm { c l o } }$ , remains an independent diagnostic rather than a training term. Estimator details are given in Appendix D.

## 4.4 Annealed Solid-Mean Temperature Regularization

The Local and Global modules form the core two-scale conservation objective. During early optimization, however, the independently initialized temperature networks can drift toward incompatible or implausible overall levels before crossdomain coupling is established. The Reg module anchors the solid-domain mean temperature during this initial phase without prescribing its spatial structure:

$$
\begin{array} { c l c r } { \displaystyle \overline { T } _ { s } ( \phi _ { s } ) = \frac 1 { \left| \Omega _ { s } \right| } \int _ { \Omega _ { s } } T _ { s } ( \mathbf x ; \phi _ { s } ) \mathrm d V , } \\ { \displaystyle T _ { \mathrm { i n , a v g } } = \frac { \dot { m } _ { h } T _ { h , \mathrm { i n } } + \dot { m } _ { c } T _ { c , \mathrm { i n } } } { \dot { m } _ { h } + \dot { m } _ { c } } . } \end{array}\tag{14}
$$

Because both streams use the same fluid, this is also heatcapacity-rate weighting. For volume-uniform solid samples $\mathcal { X } _ { s } ^ { \mathbf { \check { \cal { R } } } } , \tilde { N _ { s } ^ { \cal { R } } } = | \mathcal { X } _ { s } ^ { \cal { R } } |$ ，

$$
\begin{array} { c } { { \displaystyle \hat { \overline { { T } } } _ { s } = \frac 1 { N _ { s } ^ { R } } \sum _ { \mathbf x _ { i } \in \mathcal X _ { s } ^ { R } } T _ { s } ( \mathbf x _ { i } ; \phi _ { s } ) , } } \\ { { \ } } \\ { { \mathcal L _ { \mathrm { m e a n - r e g } } = \left( \widehat { \overline { { T } } } _ { s } - T _ { \mathrm { i n , a v g } } \right) ^ { 2 } . } } \end{array}\tag{15}
$$

This zero-order regularizer preserves the spatial degrees of freedom of $T _ { s }$ and influences the fluid networks only through the coupled physics losses.

At thermal update t, the complete objective is

$$
\mathcal { L } ( t ) = \mathcal { L } _ { \mathrm { p t } } + \lambda _ { L } \mathcal { L } _ { \mathrm { l o c a l } } + \lambda _ { G } \mathcal { L } _ { \mathrm { g l o b a l } } + \lambda _ { R } ( t ) \mathcal { L } _ { \mathrm { m e a n - r e g } } .\tag{16}
$$

The conservation weights $\lambda _ { L }$ and $\lambda _ { G }$ remain fixed throughout thermal training. With $N _ { T }$ thermal updates, $N _ { R } \ =$ $N _ { T } / 2$ , and $\lambda _ { R } ^ { ( 0 ) } = 1 , \lambda _ { R } ( t ) = \lambda _ { R } ^ { ( 0 ) } \operatorname* { m a x } ( 1 - t / N _ { R } , 0 )$ decreases linearly to zero over the first half of training and remains inactive thereafter. The implemented loss weights and method-specific conservation quadrature are reported in Appendix F.

## 5 Experiments

## 5.1 Experimental Setup

Problem setting. We use Primitive, Gyroid, Diamond, and IWP TPMS exchangers, each containing five streamwise unit cells in $[ 0 , 5 ] \times [ 0 , \bar { 1 } ] \times [ 0 , 1 ] .$ , plus a more complex, geometrically distinct DualMS U-shaped exchanger (Zhang et al. 2025). Figure 2 visualizes the coupled three-dimensional domains of the TPMS family. In every TPMS case, both streams enter at $x = 0$ on the left and exit at $x = 5$ on the right, giving the left-to-right parallel-flow configuration shown in the figure. Their channel connectivity and fluid–solid interface orientation vary substantially across topologies, providing geometrically distinct supports for the same CHT equations and training protocol. Both streams operate at $R e \mathrm { ~ = ~ } 1 0 0$ with inlet temperatures of $8 0 ^ { \circ } \mathrm { C }$ and $\mathrm { 2 0 ^ { \circ } C }$ . ANSYS CFX provides reference fields; geometry, boundary conditions, and solver settings are detailed in Appendix E.

![](images/4e24c5d5dd28c9114c27f329f28cc70f33862f4646db3e51c7b43f5cafc7fe2b.jpg)

![](images/ec704510465177bac45c6162cf662d3d53498718ef71fb986f982ea2372018a6.jpg)  
Figure 2: TPMS exchanger geometries. Top: Primitive and Gyroid; bottom: Diamond and IWP. Each exchanger contains five streamwise unit cells. Red and blue denote the hot- and cold-fluid domains, while dark gray denotes the separating solid wall.

Baselines. We compare with the CHT-specific E-MPINN (Lee et al. 2025), optimization-oriented RoPINN and CoPINN (Wu et al. 2024; Duan et al. 2025), and geometryaware MUSA-PINN (Zhang et al. 2026). MUSA-PINN-CHT uses MUSA-PINN for both flows and the pointwise CHT objective for all three temperatures, making it the closest baseline.

All methods are evaluated with matched network capacity, complete CHT physics, and the same total number of optimization updates. Appendix F.2 details the baseline adaptations, sampling protocol, staged versus joint training, and computational cost.

Implementation details. Each subnetwork uses 3→60 Fourier features (Tancik et al. 2020) followed by five width-256 tanh layers. Training uses SOAP (Wang et al. 2025) and PyTorch 2.4.1 with CUDA 12.4 on an NVIDIA A40. Network and optimization settings are listed in Appendix Table 8.

Evaluation metrics. We report four complementary metrics. $E _ { \mathrm { f i e l d } }$ is the pooled temperature MAE over all three media and measures distributed field reconstruction. Because exchanger performance depends on cumulative energy transfer, $E _ { \mathrm { o u t } }$ averages the hot- and cold-stream bulk outlet-temperature errors; common CFX mass-flux weights isolate temperature accuracy from learned-flow diferences. A small closure error can nevertheless hide similarly biased heat rates, so $E _ { Q }$ compares the normalized hot-side heat release $Q _ { h }$ and cold-side heat gain $Q _ { c }$ with the CFX reference:

$$
E _ { Q } = 1 0 0 \sqrt { \frac { ( Q _ { h } ^ { \mathrm { p r e d } } - Q _ { h } ^ { \mathrm { r e f } } ) ^ { 2 } + ( Q _ { c } ^ { \mathrm { p r e d } } - Q _ { c } ^ { \mathrm { r e f } } ) ^ { 2 } } { 2 } } \mathcal { Y } _ { 0 } .
$$

Finally, $E _ { \mathrm { c l o } } ~ = ~ 1 0 0 | Q _ { h } ^ { \mathrm { p r e d } } - Q _ { c } ^ { \mathrm { p r e d } } | \%$ directly measures device-level energy imbalance. Temperatures are reported in kelvin; $Q _ { h }$ and $Q _ { c }$ are normalized by $Q _ { 0 } = C _ { \mathrm { m i n } } \Delta T$ Appendix G gives the exact estimators.

## 5.2 Comparative Analysis on TPMS Exchangers

Overall performance. Table 1 shows that Cova-PINN is best in every reported TPMS cell. Relative to MUSA-PINN-CHT, it reduces average $E _ { \mathrm { f i e l d } } , \ E _ { \mathrm { o u t } }$ , and $E _ { \mathrm { c l o } }$ by 28.6%,

<table><tr><td></td><td></td><td colspan="3">Primitive (P)</td><td colspan="3">Gyroid (G)</td><td colspan="3">Diamond (D)</td><td colspan="3">IWP</td></tr><tr><td>Method</td><td>Ref.</td><td> $E _ { \mathrm { f i e l d } }$  (K)</td><td> $E _ { \mathrm { o u t } } \left( \mathrm { K } \right)$ </td><td> $E _ { \mathrm { c l o } } ( \% )$ </td><td> $E _ { \mathrm { f i e l d } } ~ ( \mathrm { K } )$ </td><td> $E _ { \mathrm { o u t } } \left( \mathrm { K } \right)$ </td><td> $E _ { \mathrm { c l o } } ( \% )$ </td><td> $E _ { \mathrm { f i e l d } }$  (K)</td><td> $E _ { \mathrm { o u t } } \left( \mathrm { K } \right)$ </td><td> $E _ { \mathrm { c l o } } ( \% )$ </td><td> $E _ { \mathrm { f i e l d } }$  (K)</td><td> $E _ { \mathrm { o u t } }$  (K)</td><td> $E _ { \mathrm { c l o } } ( \% )$ </td></tr><tr><td>E-MPINN</td><td>IJHMT’25</td><td>2.95</td><td>0.793</td><td>4.59</td><td>4.28</td><td>9.358</td><td>7.26</td><td>5.93</td><td>7.451</td><td>8.04</td><td>6.94</td><td>10.487</td><td>239.33</td></tr><tr><td>RoPINN-CHT</td><td>NeurIPS&#x27;24</td><td>5.49</td><td>2.366</td><td>25.36</td><td>4.34</td><td>9.265</td><td>12.75</td><td>5.63</td><td>7.394</td><td>12.63</td><td>10.99</td><td>9.600</td><td>10.20</td></tr><tr><td>CoPINN-CHT</td><td>ICML&#x27;25</td><td>3.16</td><td>0.869</td><td>5.86</td><td>4.53</td><td>12.108</td><td>12.28</td><td>5.84</td><td>7.435</td><td>5.75</td><td>6.97</td><td>10.334</td><td>4092.33</td></tr><tr><td>MUSA-PINN-CHT</td><td>ICML&#x27;26</td><td>1.75</td><td>1.117</td><td>11.88</td><td>4.20</td><td>2.108</td><td>1.29</td><td>3.65</td><td>0.937</td><td>2.87</td><td>5.04</td><td>2.326</td><td>5.96</td></tr><tr><td>Cova-PINN</td><td>Ours</td><td>1.40</td><td>0.316</td><td>1.01</td><td>2.96</td><td>1.890</td><td>0.92</td><td>3.12</td><td>0.828</td><td>2.73</td><td>2.98</td><td>1.006</td><td>4.09</td></tr><tr><td>IMP.</td><td></td><td>20.0%</td><td>60.2%</td><td>77.9%</td><td>29.5%</td><td>10.3%</td><td>28.8%</td><td>14.5%</td><td>11.7%</td><td>4.8%</td><td>40.9%</td><td>56.7%</td><td>31.3%</td></tr></table>

Table 1: End-to-end accuracy on four TPMS exchangers. We report full-field temperature, outlet bulk-temperature, and device-level closure errors. Lower is better; bold and underline mark the best and second-best results. IMP. is Cova-PINN’ improvement over the strongest baseline. Exact definitions and heat-duty results are given in Appendix G and H.

![](images/35ecf7f53016bbeb043ba64b4e9948424fb98ac4da17fe47a91ed1ee25b78c31.jpg)

![](images/f4a0b367d9316fd5e8b4e61b6085d5685a2790cc12ec22f07ab9086bb80ac721.jpg)

![](images/8b8196aa7f782ca971f53a3c3006491dc976838398b03729bc17f9fbfd502fc9.jpg)

![](images/b8bfb64da27c749397cf687c3757fef20b60ad2378fd234a219e9c28f874f214.jpg)

![](images/a37946d1b9fd897159d4fa5ca776f8e6ed62f00633a3a6e69fa353a81b1ad349.jpg)

![](images/6f12ec71ee940e8e9d7821b504517d4638e5ff0e0b67823b4090670496272040.jpg)

![](images/4f728fd47bdd5d8a31002292dd33cf9120d795974fd932cb38ff867c584e389e.jpg)

![](images/1cf4281f3707f1e6436cd66e55b94a68dde15249d1dd4daabfffa8e4ae432211.jpg)

![](images/e53d78026c5219c80cb8657889c053bf40d4e12822b1ac44c1ccc5a2bc7e8b3a.jpg)

![](images/5997afade47e073c1fb568047b7e06af6d5866ce3abe7502e8c6d66840e558c5.jpg)

![](images/5d463f3b6c328010e1e562e43f18863f64ecd8d610af6874c3841d3556d59599.jpg)

![](images/a0d8b8934ff4861523bf78090901efc62adb0aa6db7a2a245bb795318cdd14c3.jpg)  
Figure 3: Outlet-temperature comparison on the Diamond exchanger at $x = 5 .$ Top: predicted temperatures from the five methods and CFX; bottom: the corresponding absolute-error fields, followed by $\Delta E = \left| T _ { \mathrm { M U S A } } - T _ { \mathrm { C F X } } \right| - \left| T _ { \mathrm { C o v a } } - T _ { \mathrm { C F X } } \right|$ in the rightmost panel. Prediction and absolute-error panels use matched color ranges. Positive red values in $\Delta E$ indicate lower Cova-PINN error, whereas negative blue values indicate lower MUSA-PINN-CHT error.

37.7%, and 60.2%, respectively. The independently evaluated $E _ { Q }$ also falls by 44.8%, confirming more accurate exchanged heat rather than closure alone.

Performance across geometries. The gains persist across interface topologies: Primitive shows 60.2% lower outlet error and 77.9% lower closure error than the strongest baselines, while on IWP Cova-PINN reduces MUSA-PINN-CHT’s field/outlet errors from 5.04/2.326 to 2.98/1.006 K. It also achieves the lowest closure on Gyroid and Diamond.

Figure 3 compares all methods on the Diamond outlet plane $x \ = \ 5 ,$ , complementing the bulk outlet metric with spatially resolved errors. Having established the full-baseline comparison at the outlet, Figure 4 focuses on MUSA-PINN-CHT, the closest structural baseline, and follows the internal temperature field through all five cells at $z = 0 . 5$ . Its signed error-diference map localizes the aggregate improvement: red regions favor Cova-PINN, whereas blue regions favor MUSA-PINN-CHT. The full longitudinal comparison with all baselines is provided in Appendix H.2.

## 5.3 Cross-Family Evaluation on DualMS

To test Cova-PINN on a more complex geometry outside the periodic TPMS family, we use the DualMS exchanger, in which the hot and cold streams traverse separate U-shaped passages in opposite directions. This counter-flow arrangement changes both interface topology and transport pathway; Figure 5 displays the common mid-plane of the two passages (Table 2).

<table><tr><td>Method</td><td> $\scriptstyle { E _ { \mathrm { f i e l d } } }$  (K)</td><td> $\mathbf { \mathbf { \mathbf { { E _ { o u t } } } } }$  (K)</td><td> $E _ { Q }$  (%)</td><td> $E _ { \mathrm { c l o } } ( \% )$ </td></tr><tr><td>MUSA-PINN-CHT</td><td>5.996</td><td>4.250</td><td>7.037</td><td>13.575</td></tr><tr><td>Cova-PINN</td><td>4.802</td><td>1.354</td><td>1.998</td><td>3.590</td></tr></table>

Table 2: Cross-family evaluation on the geometrically distinct DualMS exchanger. Cova-PINN is compared with MUSA-PINN-CHT, its closest complex-geometry baseline, using field, outlet, heat-duty, and closure errors. Lower is better for all metrics.

Cova-PINN follows CFX more closely along the curved pathway and reduces E<sub>field</sub>, $E _ { \mathrm { o u t } }$ $E _ { Q }$ and $E _ { \mathrm { c l o } }$ by 19.9%, 68.1%, 71.6%, and $7 3 . 6 \%$ , respectively. The larger exchanger-level gains show that transfer fidelity warrants evaluation beyond field MAE.

![](images/66d4bae363c61b61ee431d8eb8da261d5fcd6b51996ec3a23aa236f9f24d27c8.jpg)  
Figure 4: Longitudinal temperature comparison on the Diamond exchanger at $z = 0 . 5 .$ . From top to bottom: CFX, MUSA-PINN-CHT, and Cova-PINN temperatures, followed by $\Delta E = \vert T _ { \mathrm { M U S A } } - T _ { \mathrm { C F X } } \vert - \vert T _ { \mathrm { C o v a } } - T _ { \mathrm { C F X } } \vert$ . The temperature panels share one color range. Positive red values indicate lower Cova-PINN error; negative blue values indicate lower MUSA-PINN-CHT error.

<table><tr><td>Variant  $E _ { \mathrm { f i e l d } }$ </td><td>(K)  $E _ { \mathrm { o u t } }$  (K)</td><td> $E _ { \mathrm { c l o } }$  (%)</td></tr><tr><td>Full Cova-PINN</td><td>2.98</td><td>1.006 4.09</td></tr><tr><td>w/o Local CV (L)</td><td>4.81</td><td>2.157 6.59</td></tr><tr><td>w/o Global Closure (G)</td><td>5.00 2.335</td><td>7.32</td></tr><tr><td>w/o Solid-Mean Reg. (R)</td><td>4.56 4.568</td><td>23.87</td></tr></table>

Table 3: Component-wise ablation of Cova-PINN. The controlled study uses the IWP benchmark throughout. Starting from full Cova-PINN, each variant removes loca composite-control-volume conservation (L), global pairedwall heat-rate closure (G), or annealed solid-mean regularization (R), while keeping all other training settings fixed. Lower is better for all metrics.

## 5.4 Ablation Studies

We isolate the roles of the finite-region, device-scale, and early-stage objectives through a controlled leave-onecomponent-out study on a fixed TPMS benchmark.

Table 3 shows that every removal degrades all three reported metrics. Removing G gives the largest field error (5.00 K), whereas removing R most strongly harms outlet and closure accuracy (4.568 K and 23.87%). $E _ { Q }$ likewise rises from 4.21% to 4.36%, 4.75%, and 12.71% without L, G, and R, respectively. These distinct degradation patterns confirm that the finite-region, device-scale, and early-stage objectives make complementary contributions.

![](images/13c2c7a3e3bd74d59b5c200564045a29c8b134dcf3caec467a8bfc1943ef47c7.jpg)  
Figure 5: Temperature field comparison on DualMS. All panels show the common mid-plane of the two counter-flow U-shaped passages. Top: MUSA-PINN-CHT, Cova-PINN, and CFX temperatures. Bottom: the two absolute-error fields and $\Delta E = \bar { E } _ { \mathrm { M U S A } } - E _ { \mathrm { C o v a } } ;$ positive red values indicate lower Cova-PINN error, whereas negative blue values indicate lower MUSA-PINN-CHT error.

## 6 Discussion and Conclusion

In this paper, we presented Cova-PINN, a support-aligned multi-network framework for fluid–solid CHT in complex geometries. Its global–local conservation co-optimization coordinates finite-region transfer through wall-centered composite control volumes and exchanger-scale transfer through paired-wall closure, while annealed solid-mean regularization conditions early training. Relative to MUSA-PINN-CHT, it reduces average field, outlet, heat-duty, and closure errors across four TPMS topologies by 28.6%, 37.7%, 44.8%, and 60.2%, respectively. On the more complex DualMS exchanger, the corresponding reductions are 19.9%, 68.1%, 71.6%, and 73.6%. Component ablations further verify the complementary roles of all three objectives. These results establish conservation-support alignment as an efective principle for multi-domain physics learning.

Limitations. The present study considers steady, constantproperty CHT with temperature-independent flow fields; transient transport and temperature-dependent flow–thermal feedback remain outside its scope. Local-control-volume and paired-wall quadrature also increase training cost (Appendix I).

Future Work. Future work will extend support-aligned conservation to coupled and transient thermofluid problems, learn adaptive control-volume placement from geometry and residuals, and develop sparse quadrature. Parameterized studies across operating conditions and material contrasts will assess transferability.

Duan, S.; Wu, W.; Hu, P.; Ren, Z.; Peng, D.; and Sun, Y. 2025. CoPINN: Cognitive physics-informed neural networks. In Forty-second International Conference on Machine Learning.

E, W.; and Yu, B. 2018. The deep Ritz method: a deep learning-based numerical algorithm for solving variational problems. Communications in Mathematics and Statistics, 6(1): 1–12.

Hao, Z.; Yao, J.; Su, C.; Su, H.; Wang, Z.; Lu, F.; Xia, Z.; Zhang, Y.; Liu, S.; Lu, L.; et al. 2024. Pinnacle: A comprehensive benchmark of physics-informed neural networks for solving pdes. Advances in Neural Information Processing Systems, 37: 76721–76774.

Hwang, L. K. 2025. Physics-informed machine learning with domain decomposition and global dynamics for threedimensional intersecting flows. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Jagtap, A. D.; and Karniadakis, G. E. 2020. Extended physics-informed neural networks (XPINNs): A generalized space-time domain decomposition based deep learning framework for nonlinear partial diferential equations. Communications in Computational Physics, 28(5).

Jagtap, A. D.; Kharazmi, E.; and Karniadakis, G. E. 2020. Conservative physics-informed neural networks on discrete domains for conservation laws: Applications to forward and inverse problems. Computer Methods in Applied Mechanics and Engineering, 365: 113028.

Karniadakis, G. E.; Kevrekidis, I. G.; Lu, L.; Perdikaris, P.; Wang, S.; and Yang, L. 2021. Physics-informed machine learning. Nature Reviews Physics, 3(6): 422–440.

Kharazmi, E.; Zhang, Z.; and Karniadakis, G. E. 2019. Variational physics-informed neural networks for solving partial diferential equations. arXiv preprint arXiv:1912.00873.

Kharazmi, E.; Zhang, Z.; and Karniadakis, G. E. 2021. hp-VPINNs: Variational physics-informed neural networks with domain decomposition. Computer Methods in Applied Mechanics and Engineering, 374: 113547.

Krishnapriyan, A.; Gholami, A.; Zhe, S.; Kirby, R.; and Mahoney, M. 2021. Characterizing possible failure modes in physics-informed neural networks. Advances in neural information processing systems, 34: 26548–26560.

Lee, J.; Shin, S.; Choi, H.; Lee, A.; Park, B.; and Lee, S. 2025. Extended multiphysics-informed neural network for conjugate heat transfer problems. International Journal of Heat and Mass Transfer, 246: 127098.

Lu, Z.; Li, Y.; He, C.; Ren, J.; Yu, H.; Zhang, B.; and Chen, Q. 2024. Multi-objective inverse design of finned heat sink system with physics-informed neural networks. Computers & Chemical Engineering, 180: 108500.

Moseley, B.; Markham, A.; and Nissen-Meyer, T. 2023. Finite basis physics-informed neural networks (FBPINNs): a scalable domain decomposition approach for solving diferential equations: B. Moseley et al. Advances in Computational Mathematics, 49(4): 62.

Raissi, M.; Perdikaris, P.; and Karniadakis, G. E. 2019. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations. Journal of Computational physics, 378: 686–707.

Sarma, A. K.; Roy, S.; Annavarapu, C.; Roy, P.; and Jagannathan, S. 2024. Interface PINNs (I-PINNs): A physicsinformed neural networks framework for interface problems. Computer Methods in Applied Mechanics and Engineering, 429: 117135.

Tancik, M.; Srinivasan, P.; Mildenhall, B.; Fridovich-Keil, S.; Raghavan, N.; Singhal, U.; Ramamoorthi, R.; Barron, J.; and Ng, R. 2020. Fourier features let networks learn high frequency functions in low dimensional domains. Advances in neural information processing systems, 33: 7537–7547.

Wang, S.; bhartari, A. K.; Li, B.; and Perdikaris, P. 2025. Gradient Alignment in Physics-informed Neural Networks: A Second-Order Optimization Perspective. In The Thirtyninth Annual Conference on Neural Information Processing Systems.

Wang, S.; Teng, Y.; and Perdikaris, P. 2021. Understanding and mitigating gradient flow pathologies in physics-informed neural networks. SIAM Journal on Scientific Computing, 43(5): A3055–A3081.

Wang, S.; Yu, X.; and Perdikaris, P. 2022. When and why PINNs fail to train: A neural tangent kernel perspective. Journal ofComputational Physics, 449: 110768.

Wei, C.; Fan, Y.; Wong, J. C.; Ooi, C. C.; Wang, H.; and Chiu, P.-H. 2025a. FFV-PINN: A fast physics-informed neural network with simplified finite volume discretization and residual correction. Computer Methods in Applied Mechanics and Engineering, 444: 118139.

Wei, C.; Fan, Y.; Zhou, Y.; Liu, X.; Li, C.; Li, X.; and Wang, H. 2025b. Physics-informed neural network based on control volumes for solving time-independent problems. Physics of Fluids, 37(3).

Wu, H.; Luo, H.; Ma, Y.; Wang, J.; and Long, M. 2024. Ropinn: Region optimized physics-informed neural networks. Advances in Neural Information Processing Systems, 37: 110494–110532.

Yeranee, K.; and Rao, Y. 2022. A review of recent investigations on flow and heat transfer enhancement in cooling channels embedded with triply periodic minimal surfaces (TPMS). Energies, 15(23): 8994.

Zang, Y.; Bao, G.; Ye, X.; and Zhou, H. 2020. Weak adversarial networks for high-dimensional partial diferential equations. Journal of Computational Physics, 411: 109409.

Zhang, B.; Wu, G.; Gu, Y.; Wang, X.; and Wang, F. 2022. Multi-domain physics-informed neural network for solving forward and inverse problems of steady-state heat conduction in multilayer media. Physics of Fluids, 34(11).

Zhang, W.; Pan, H.; Lu, L.; Duan, X.; Yan, X.; Wang, R.; and Du, Q. 2025. DualMS: Implicit Dual-Channel Minimal Surface Optimization for Heat Exchanger Design. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, 1–10.

Zhang, W.; Xie, X.; Pan, H.; Duan, X.; Sun, B.; Du, Q.; and Lu, L. 2026. MUSA-PINN: Multi-scale Weak-form Physics-Informed Neural Networks for Fluid Flow in Complex Geometries. In Forty-third International Conference on Machine Learning.

Zhang, X.; Tu, C.; and Yan, Y. 2024. Physics-informed neural network simulation of conjugate heat transfer in manifold microchannel heat sinks for high-power IGBT cooling. International Communications in Heat and Mass Transfer, 159: 108036.

Zheng, J.; Huang, Y.; and Yi, N. 2025. IG-PINNs: Interfacegated physics-informed neural networks for solving elliptic interface problems. Journal of Computational Physics, 114540.

## Appendix Contents

A Complete Problem Specification 11   
A.1 Domains, Assumptions, and Nondimension  
alization . . 11   
A.2 Governing Equations and Interface Conditions 11   
A.3 Boundary and Material Configuration 11   
B Geometry Construction 12   
B.1 TPMS Benchmark Family 12   
B.2 DualMS Geometry 12   
C Spherical Composite Control Volumes 12   
C.1 Composite Outer-Boundary Construction 12   
C.2 Area-Uniform Sampling and Total-Area   
Weighting . . . . 12   
C.3 Interface Cancellation . 13   
D Paired-Wall Heat-Rate Estimator 13   
E Reference Solutions 13   
E.1 Reference Solver 13   
F Experimental Configuration 13   
F.1 Networks, Base Losses, and Sequential   
Training . 13   
F.2 Baseline Selection and Fairness . 14   
G Evaluation Metrics 15   
G.1 Temperature-Field and Outlet Errors . . . 15   
G.2 Heat-Rate Accuracy and Device Closure . . 15   
H Complete Results 16   
H.1 Reference-Normalized Heat-Duty Accuracy 16   
H.2 Full Longitudinal Field Comparison . . . . 16   
I Computational Cost 16

## A Complete Problem Specification A.1 Domains, Assumptions, and Nondimensionalization

Let $\Omega _ { h } , \Omega _ { s } .$ , and $\Omega _ { c } \subset \mathbb { R } ^ { 3 }$ be the mutually disjoint interiors of the hot-fluid, solid, and cold-fluid regions. Their conjugate interfaces are $\Gamma _ { h s } = \partial \Omega _ { h } \cap \partial \Omega _ { s }$ and $\Gamma _ { s c } = \partial \Omega _ { s } \cap \partial \Omega _ { c }$ We consider steady incompressible forced convection with constant material properties, negligible buoyancy, and no volumetric heat sources. The hot and cold streams use the same fluid with properties $\rho _ { f } , c _ { p , f } , \mu _ { f }$ , and $k _ { f }$ . Temperature does not feed back into momentum, so the two flow fields are learned first and frozen throughout thermal training.

Dimensional quantities carry the superscript d, whereas symbols without this superscript are dimensionless. With geometric coordinate scale $L _ { 0 } ,$ , inlet mean speed $U _ { 0 , f }$ for fluid $f \in \{ h , c \}$ , and $\Delta T = T _ { h , \mathrm { i n } } ^ { \mathrm { d } } - T _ { c , \mathrm { i n } } ^ { \mathrm { d } } ,$ the variables are nondimensionalized as

$$
\begin{array} { r l r } & { \mathbf { x } = \displaystyle \frac { \mathbf { x } ^ { \mathrm { d } } } { L _ { 0 } } , } & { \mathbf { u } _ { f } = \displaystyle \frac { \mathbf { u } _ { f } ^ { \mathrm { d } } } { U _ { 0 , f } } , } \\ & { p _ { f } = \displaystyle \frac { p _ { f } ^ { \mathrm { d } } - p _ { f , \mathrm { o u t } } ^ { \mathrm { d } } } { \rho _ { f } U _ { 0 , f } ^ { 2 } } , } & { T _ { m } = \displaystyle \frac { T _ { m } ^ { \mathrm { d } } - T _ { c , \mathrm { i n } } ^ { \mathrm { d } } } { \Delta T } , } \end{array}\tag{17}
$$

where $m \in \{ h , s , c \}$ . Let $C _ { f } = \dot { m } _ { f } c _ { p , f } , C _ { \mathrm { m i n } } =$ min $( C _ { h } , C _ { c } ) _ { }$ , and $Q _ { 0 } \stackrel { \cdot } { = } C _ { \mathrm { m i n } } \Delta \dot { T }$ . The dimensionless flow groups and conductivity ratios are

$$
\begin{array} { r l r } & { R e _ { f } = \frac { \rho _ { f } U _ { 0 , f } L _ { 0 } } { \mu _ { f } } , } & { P e _ { f } = \frac { \rho _ { f } c _ { p , f } U _ { 0 , f } L _ { 0 } } { k _ { f } } , } \\ & { \kappa _ { m } = \cfrac { k _ { m } } { k _ { f } } , \quad m \in \{ h , s , c \} , } & { \kappa _ { h } = \kappa _ { c } = 1 . } \end{array}\tag{18}
$$

The characteristic length is the inlet hydraulic diameter,

$$
L _ { 0 } = D _ { h } = { \frac { 4 A _ { \mathrm { i n } } } { P _ { \mathrm { i n } } } } ,\tag{19}
$$

where $A _ { \mathrm { i n } }$ and $P _ { \mathrm { i n } }$ are the area and wetted perimeter of the inlet cross-section. The topology-specific dimensional values of $L _ { 0 }$ are reported in Table 6.

## A.2 Governing Equations and Interface Conditions

The dimensionless flow equations in each fluid are

$$
\begin{array} { r l r } { \nabla \cdot \mathbf { u } _ { f } = 0 , } & { } & \\ { ( \mathbf { u } _ { f } \cdot \nabla ) \mathbf { u } _ { f } + \nabla p _ { f } - \displaystyle \frac { 1 } { R e _ { f } } \nabla ^ { 2 } \mathbf { u } _ { f } = \mathbf { 0 } , \qquad f \in \{ h , c \} . } \end{array}\tag{20}
$$

With no internal heat generation, the energy equations are

$$
\begin{array} { c } { { \mathbf { u } _ { f } \cdot \nabla T _ { f } - \frac { 1 } { P e _ { f } } \nabla ^ { 2 } T _ { f } = 0 , \quad f \in \{ h , c \} , } } \\ { { \nabla ^ { 2 } T _ { s } = 0 . } } \end{array}\tag{21}
$$

For integral balances across media, all fluxes are expressed using the common reference heat-flux scale $k _ { f } \Delta T / \bar { L } _ { 0 } :$

$$
\begin{array} { l l } { { \bf J } _ { f } = P e _ { f } { \bf u } _ { f } T _ { f } - \nabla T _ { f } , } & { f \in \{ h , c \} , } \\ { { \bf J } _ { s } = - \kappa _ { s } \nabla T _ { s } , } & { \nabla \cdot { \bf J } _ { m } = 0 , \quad m \in \{ h , s , c \} . } \end{array}\tag{22}
$$

<table><tr><td>Boundary</td><td>Flow condition</td><td>Thermal condition</td></tr><tr><td>Hot inlet</td><td>Uniform,  $| \mathbf { u } _ { h } ^ { \mathrm { d } } | = 1 \mathrm { m } \mathrm { s } ^ { - 1 }$ </td><td> $T _ { h } ^ { \mathrm { d } } = 8 0 ^ { \circ } \mathrm { C }$ </td></tr><tr><td>Cold inlet</td><td>along +x Uniform, |ud| = 1 m s−1 along +x</td><td> $T _ { c } ^ { \mathrm { { \ddot { d } } } } = 2 0 { } ^ { \circ } \mathrm { C }$ </td></tr><tr><td>Fluid outlets</td><td>Mean gauge pressure pd = 0 Pa</td><td>No direct thermal constraint</td></tr><tr><td> $\Gamma _ { h s } , \Gamma _ { s c }$ </td><td>No slip</td><td>Temperature and normal-flux continuity</td></tr><tr><td>External solid boundary</td><td></td><td>Adiabatic</td></tr></table>

Table 4: Boundary conditions for the four TPMS benchmarks. All methods use the same flow and thermal conditions; no direct thermal constraint is imposed at the fluid outlets.

<table><tr><td>Quantity</td><td>Value</td></tr><tr><td> $R e _ { h } / R e _ { c }$ </td><td> $1 0 0 / 1 0 0$ </td></tr><tr><td> $T _ { h , \mathrm { i n } } ^ { \mathrm { d } } / T _ { c , \mathrm { i n } } ^ { \mathrm { d } }$ </td><td> $8 0 / 2 0 \ ^ { \circ } \mathrm { C }$ </td></tr><tr><td> $U _ { 0 , h } ^ { \mathrm { d } } / U _ { 0 , c } ^ { \mathrm { d } }$ </td><td> $1 / 1 \mathrm { m s } ^ { - 1 }$ </td></tr><tr><td> $\rho _ { f }$ </td><td> $1 \mathrm { k g } \mathrm { m } ^ { - 3 }$ </td></tr><tr><td> $c _ { p , f }$ </td><td> $1 0 0 0 \mathrm { J } \mathrm { k g } ^ { - 1 } \mathrm { K } ^ { - 1 }$ </td></tr><tr><td> $k _ { f }$ </td><td> $0 . 6 0 6 9 \mathrm { W } \mathrm { m } ^ { - 1 } \mathrm { K } ^ { - 1 }$ </td></tr><tr><td> $k _ { s }$ </td><td> $2 3 7 \mathrm { W } \mathrm { m } ^ { - 1 } \mathrm { K } ^ { - 1 }$ </td></tr><tr><td> $\kappa _ { s } = k _ { s } / k _ { f }$ </td><td>390.509</td></tr><tr><td> $L _ { 0 } , \mu _ { f } , P e _ { f }$ </td><td>Topology-specific; Table 6</td></tr><tr><td> $Q _ { \mathrm { r e f } }$ </td><td> $k _ { f } \Delta T L _ { 0 }$ </td></tr><tr><td></td><td></td></tr><tr><td> $Q _ { 0 }$ </td><td> $C _ { \mathrm { m i n } } \Delta T$ </td></tr></table>

Table 5: Shared operating conditions and thermal material parameters for the TPMS benchmarks. Both streams use the listed values in every topology; topology-specific $L _ { 0 } .$ $\mu _ { f }$ , and $P e _ { f }$ are reported in Table 6.

Their surface integrals therefore share the heat-rate scale $Q _ { \mathrm { r e f } } = k _ { f } \Delta T L _ { 0 }$ . The reporting scale $Q _ { 0 }$ is used separately for the outlet-based energy-imbalance metric in Sec. G. Define $\Gamma _ { h } : = \Gamma _ { h s }$ and $\Gamma _ { c } : \bar { = } \Gamma _ { s c }$ . If $\mathbf { n } _ { f }$ and $\mathbf { n } _ { s }$ point outward from the adjoining fluid and solid regions, respectively, the conjugate conditions on $\Gamma _ { f }$ , for $f \in \left\{ h , c \right\}$ , are

$$
\mathbf { u } _ { f } = \mathbf { 0 } , \qquad T _ { f } = T _ { s } , \qquad \nabla T _ { f } \cdot \mathbf { n } _ { f } + \kappa _ { s } \nabla T _ { s } \cdot \mathbf { n } _ { s } = 0 .\tag{23}
$$

## A.3 Boundary and Material Configuration

Table 4 summarizes the boundary conditions. The phrase “no slip” applies to fluid velocity on a fluid–solid wall; the solid does not have a velocity field.

For DualMS, the same inlet speed magnitudes and temperatures, outlet gauge pressure, interface conditions, and external adiabatic condition are applied to its labeled ports and walls; inlet velocity directions follow the inward normals of the corresponding ports.

The four principal TPMS cases use $R e _ { h } = R e _ { c } = 1 0 0 $ $T _ { h , \mathrm { i n } } ^ { \mathrm { d } } = 8 0 ^ { \circ } \mathrm { C } ,$ , and $T _ { c , \mathrm { i n } } ^ { \mathrm { d } } = 2 0 ^ { \circ } \mathrm { C }$ , hence $\Delta T = 6 0 \mathrm { K }$ . The hot and cold streams have identical constant properties within each benchmark. Density, heat capacity, and conductivity are shared across topologies and are listed in Table 5. Because the inlet hydraulic diameter $L _ { 0 }$ is topology dependent, the viscosity is set as $\mu _ { f } = { \rho _ { f } U _ { 0 , f } L _ { 0 } } / { R e _ { f } }$ to retain $R e _ { f } = 1 0 0 ;$ the resulting $L _ { 0 } , \mu _ { f }$ , and $P e _ { f }$ values are reported in Table 6.

<table><tr><td>Geometry</td><td>Array</td><td>Lo (m)</td><td> $R e _ { h } = R e _ { c }$ </td><td> $\mu _ { f } \ ( \mathbf { P a s } )$ </td><td> $P e _ { f }$ </td></tr><tr><td>Primitive (P)</td><td> $5 \times 1 \times 1 0 . 5 2 2 0$ </td><td></td><td>100</td><td> $5 . 2 2 0 \times 1 0 ^ { - 3 }$ </td><td>860.109</td></tr><tr><td>Gyroid (G)</td><td> $5 \times 1 \times 1 0 . 4 4 9 0$ </td><td></td><td>100</td><td> $4 . 4 9 0 \times 1 0 ^ { - 3 }$ </td><td>739.825</td></tr><tr><td>Diamond (D)</td><td> $5 \times 1 \times 1 0 . 3 9 1 0$ </td><td></td><td>100</td><td> $3 . 9 1 0 \times 1 0 ^ { - 3 }$ </td><td>644.258</td></tr><tr><td>IWP</td><td> $5 \times 1 \times 1 0 . 4 3 5 6$ </td><td></td><td>100</td><td> $4 . 3 5 6 \times 1 0 ^ { - 3 }$ </td><td>717.746</td></tr></table>

Table 6: Topology-specific reference lengths and dimensionless flow parameters for the TPMS benchmarks. Here $L _ { 0 } = 4 A _ { \mathrm { i n } } / \bar { P } _ { \mathrm { i n } }$ . With $\rho _ { f } = 1 \mathrm { k g m ^ { - 3 } }$ and $U _ { 0 , f } = 1 \mathrm { m } \mathrm { s } ^ { - 1 }$ the listed viscosity gives $R e _ { h } = R e _ { c } = 1 0 0$ , and $P e _ { f } =$ $\rho _ { f } c _ { p , f } U _ { 0 , f } L _ { 0 } / k _ { f }$

## B Geometry Construction

## B.1 TPMS Benchmark Family

Primitive (P), Gyroid (G), Diamond (D), and IWP form the main benchmark family. Let $\mathbf { x } = ( x , y , z )$ ) denote the dimensionless coordinates used to construct the unit cells. Each geometry contains five unit cells tiled along x and is constructed in

$$
\begin{array} { r } { \mathcal { D } = [ 0 , 5 ] \times [ 0 , 1 ] \times [ 0 , 1 ] . } \end{array}\tag{24}
$$

Both streams enter at $x = 0$ and move along the positive x direction toward outlets at $x = 5 ,$ , giving a parallel-flow configuration. Table 6 reports the topology-specific hydraulic diameters and the resulting flow parameters.

The geometry source uses the following periodic implicit functions. For compactness, let $c _ { x } = \cos ( 2 \pi x )$ and $s _ { x } ~ = ~ \sin ( 2 \pi x )$ , with cyclic definitions for $y , z ,$ , and let $c _ { 2 x } = \cos ( 4 \pi x )$

$$
\Phi _ { P } = c _ { x } + c _ { y } + c _ { z } ,\tag{25}
$$

$$
\Phi _ { G } = s _ { x } c _ { y } + s _ { y } c _ { z } + s _ { z } c _ { x } ,\tag{26}
$$

$$
\Phi _ { D } = s _ { x } s _ { y } s _ { z } + s _ { x } c _ { y } c _ { z } + c _ { x } s _ { y } c _ { z } + c _ { x } c _ { y } s _ { z } ,\tag{27}
$$

$$
\Phi _ { \mathrm { I W P } } = 2 ( c _ { x } c _ { y } + c _ { y } c _ { z } + c _ { z } c _ { x } ) - ( c _ { 2 x } + c _ { 2 y } + c _ { 2 z } ) .\tag{28}
$$

Equations (25)–(28) reproduce the four implicit functions used to construct the benchmark geometries. The same nominal wall thickness, 0.03 in normalized coordinates, is used for all four topologies. The two connected fluid regions produced by the sheet construction are assigned to the hot and cold streams before volume meshing. These geometries are treated as distinct benchmark domains; the experiments do not compare their intrinsic heat-exchanger performance against one another.

## B.2 DualMS Geometry

DualMS (Zhang et al. 2025) is used as a geometrically distinct cross-family case. We adopt the fixed U-shaped design generated with the 300-point design configuration and use the resulting watertight solid and two fluid volumes directly for sampling and CFX meshing. Every method is trained independently on this geometry; no geometry optimization is performed as part of the present study.

## C Spherical Composite Control Volumes

## C.1 Composite Outer-Boundary Construction

Let $\Gamma _ { \mathrm { w } } : = \Gamma _ { h s } \cup \Gamma _ { s c }$ denote the two conjugate heat-transfer walls. We sample the center $\mathbf { c } _ { j }$ on $\Gamma _ { \mathrm { w } }$ so that candidate parent regions are preferentially placed along the thermal interaction path. A fixed-radius spherical parent region is

$$
\begin{array} { l } { { \displaystyle B _ { j } = B ( \mathbf { c } _ { j } , r ) = \{ \mathbf { x } : \| \mathbf { x } - \mathbf { c } _ { j } \| _ { 2 } \leq r \} } , } \\ { { \displaystyle \mathbf { c } _ { j } \in \Gamma _ { \mathrm { w } } , \qquad r = 0 . 0 5 . } } \end{array}\tag{29}
$$

Let Ω denote the complete device domain with $\overline { { \Omega } } = \overline { { \Omega } } _ { h } \cup$ $\overline { { \Omega } } _ { s } \cup \overline { { \Omega } } _ { c }$ . Intersecting the parent region with each medium gives

$$
V _ { m , j } = B _ { j } \cap \Omega _ { m } , \qquad V _ { j } = \bigcup _ { m \in \{ h , s , c \} } V _ { m , j } = B _ { j } \cap \Omega .\tag{30}
$$

At each refresh, 50 centers are sampled randomly from the available solid-wall samples. The radius 0.05 exceeds the nominal wall thickness 0.03, so the resulting composite supports extend across adjoining material regions along the thermal interaction path. The exact configuration is summarized in Table 7.

The sampled boundary is the outer boundary of the composite volume, partitioned into material-labeled patches

$$
\begin{array} { r l r } & { { S } _ { m , j } = \partial V _ { j } \cap \overline { { \Omega } } _ { m } , \quad } & { m \in \{ h , s , c \} , } \\ & { { \partial } V _ { j } = { S } _ { h , j } \cup { S } _ { s , j } \cup { S } _ { c , j } . } & \end{array}\tag{31}
$$

The partition is understood up to measure-zero intersection curves. The internal patches $\bar { B _ { j } } \cap \Gamma _ { h s }$ and $B _ { j } \cap \Gamma _ { s c }$ are not included in $S _ { m , j }$ and are never sampled by the local compositecontrol-volume loss. If a ball lies inside the device envelope, $\partial V _ { j } = \partial B _ { j }$ and each nonempty $S _ { m , j }$ is a material-labeled spherical patch. A ball that reaches an inlet, outlet, wall, or other exterior device boundary is not rejected: the reached physical-boundary patches are retained and integrated in the same way as the other components of $\partial V _ { j }$

On a spherical outer patch, the outward unit normal is analytic:

$$
\mathbf { n } _ { j } ^ { \mathrm { s p h } } ( \mathbf { x } ) = \frac { \mathbf { x } - \mathbf { c } _ { j } } { r } .\tag{32}
$$

On a reached physical-boundary patch, the corresponding outward geometry normal is used.

## C.2 Area-Uniform Sampling and Total-Area Weighting

For each parent ball, $N _ { B } ~ = ~ 1 0 \small { , } 0 0 0$ candidate boundary points are sampled and classified by their material and boundary labels. No candidates are drawn from the internal interfaces $\Gamma _ { h s } \mathrm { o r } \Gamma _ { s c } .$ . Each nonempty outer-boundary component $s \subset \partial V _ { j }$ with area-uniform samples $\{ \mathbf { x } _ { k } \} _ { k = 1 } ^ { K _ { s } }$ is evaluated as

$$
\widehat { I } ( S ) = \widehat { \frac { A } { K s } } \sum _ { k = 1 } ^ { K s } g _ { m } ( \mathbf { x } _ { k } ) .\tag{33}
$$

Here $\widehat { A } _ { S }$ is the total area of the component. Every accepted point within one component has equal weight. Only the component’s total area multiplies the sample mean; no trianglewise or pointwise local-area factor is used. Estimates from separately sampled components are added only after this total-area weighting.

<table><tr><td>Item</td><td>Value</td></tr><tr><td>Center support</td><td> $\Gamma _ { \mathrm { w } } = \Gamma _ { h s } \cup \Gamma _ { s c }$ </td></tr><tr><td>Center distribution</td><td>Random sampling from solid-wall samples</td></tr><tr><td>Fixed radius r</td><td>0.05</td></tr><tr><td>Training control volumes  $N _ { \mathrm { c v } }$ </td><td>50</td></tr><tr><td>Candidate points  $N _ { B }$ </td><td>10,000 per ball</td></tr><tr><td>Exterior-boundary handling</td><td>Retain and integrate reached patches</td></tr><tr><td>Internal-interface candidates</td><td>None for  $\mathcal { L } _ { \mathrm { l o c a l } }$ </td></tr><tr><td>Sample refresh frequency</td><td>Every 500 thermal updates</td></tr><tr><td>Local-loss weight</td><td>10</td></tr><tr><td>Scale weights</td><td>None (single radius)</td></tr></table>

Table 7: Sampling and quadrature configuration for local composite control volumes. The same settings are used in all Cova-PINN experiments; the single fixed radius requires no scale aggregation.

For the local energy constraint, $g _ { m } = \mathbf { J } _ { m } \cdot \mathbf { n } _ { j }$ and

$$
\begin{array} { r l } { \displaystyle \widehat { R } _ { j } ^ { \mathrm { c v } } = \sum _ { \boldsymbol { s } \in \mathcal { P } _ { j } } \widehat { I } ( \boldsymbol { s } ) , } & { { } } \\ { \displaystyle \mathcal { L } _ { \mathrm { l o c a l } } = \frac { 1 } { N _ { \mathrm { c v } } } \sum _ { j = 1 } ^ { N _ { \mathrm { c v } } } \left( \widehat { R } _ { j } ^ { \mathrm { c v } } \right) ^ { 2 } , \qquad N _ { \mathrm { c v } } = 5 0 . } \end{array}\tag{34}
$$

Here $\mathcal { P } _ { j }$ is the set of nonempty, material-labeled outerboundary components of $V _ { j }$ , including any reached physical exterior patch but excluding internal fluid–solid interfaces. The implementation uses one radius and therefore requires no scale aggregation or scale weights. Centers and candidate samples are regenerated every 500 thermal updates.

## C.3 Interface Cancellation

For the derivation only, define $V _ { m , j } = V _ { j } \cap \Omega _ { m }$ . Integrating the diferential conservation law over each material portion, applying the divergence theorem, and summing gives

$$
\begin{array} { r l } {  { 0 = \sum _ { m \in \{ h , s , c \} } \int _ { V _ { m , - j } } \nabla \cdot \mathbf { J } _ { m } \mathrm { d } V } } \\ & { = \sum _ { m \in \{ h , s , c \} } \int _ { \mathcal { N } _ { m - j } } \mathbf { J } _ { m } \cdot \mathbf { n } _ { m , j } \mathrm { d } A } \\ & { = \sum _ { m \in \{ h , s , c \} } \int _ { S _ { m - j } } \mathbf { J } _ { m } \cdot \mathbf { n } _ { j } \mathrm { d } A } \\ & { + \int _ { \Gamma _ { k \wedge \sigma } \backslash V _ { j } } ( \mathbf { J } _ { k } \cdot \mathbf { n } _ { h } + \mathbf { J } _ { s } \cdot \mathbf { n } _ { s } ^ { h } ) \mathrm { d } A } \\ & { + \int _ { \Gamma _ { k \wedge \sigma } \backslash V _ { j } } ( \mathbf { J } _ { s } \cdot \mathbf { n } _ { s } ^ { c } + \mathbf { J } _ { c } \cdot \mathbf { n } _ { c } ) \mathrm { d } A . } \end{array}\tag{35}
$$

No penetration eliminates advective transport through each fluid–solid interface, and the normal conductive-flux condition makes the final two integrals vanish for an exact CHT solution. The remaining first term is precisely the outerboundary residual used for training. Thus the local loss enforces a finite-region consequence of the coupled PDE and interface system; it does not directly evaluate an interfaceflux jump. The separate pointwise temperature- and fluxcontinuity penalties remain essential because an outer balance alone cannot exclude compensating interface errors.

## D Paired-Wall Heat-Rate Estimator

The global closure term evaluates both interface heat rates from the solid network using $\mathbf { J } _ { s } ( \mathbf { x } ; \boldsymbol { \phi } _ { s } ) = - \kappa _ { s } \nabla T _ { s } ( \mathbf { x } ; \boldsymbol { \phi } _ { s } )$ Let $\mathbf { n } _ { s } ^ { h }$ and $\mathbf { n } _ { s } ^ { c }$ point outward from the solid toward the hot and cold fluids. With area-uniform samples $\{ \mathbf { x } _ { h s , k } \} _ { k = 1 } ^ { N _ { h s } } \subset \Gamma _ { h s }$ and $\{ \mathbf { x } _ { s c , k } \} _ { k = 1 } ^ { N _ { s c } } \subset \Gamma _ { s c } ,$ , the two full-interface integrals are estimated by

$$
\widehat { Q } _ { h  s } ( \phi _ { s } ) = - \frac { | \Gamma _ { h s } | } { N _ { h s } } \sum _ { k = 1 } ^ { N _ { h s } } \mathbf { J } _ { s } ( \mathbf { x } _ { h s , k } ; \phi _ { s } ) \cdot \mathbf { n } _ { s } ^ { h } ( \mathbf { x } _ { h s , k } ) ,\tag{36}
$$

$$
\widehat { Q } _ { s  c } ( \phi _ { s } ) = \frac { | \Gamma _ { s c } | } { N _ { s c } } \sum _ { k = 1 } ^ { N _ { s c } } \mathbf { J } _ { s } ( \mathbf { x } _ { s c , k } ; \phi _ { s } ) \cdot \mathbf { n } _ { s } ^ { c } ( \mathbf { x } _ { s c , k } ) .\tag{37}
$$

Each sample has equal weight within its interface, and the sample mean is multiplied by the corresponding total interface area; no triangle-wise or pointwise local-area factor is used. The resulting residual and loss are

$$
\begin{array} { r l } & { R ^ { \mathrm { w a l l } } ( \phi _ { s } ) = \widehat { Q } _ { h \to s } ( \phi _ { s } ) - \widehat { Q } _ { s \to c } ( \phi _ { s } ) , } \\ & { \quad \mathcal { L } _ { \mathrm { g l o b a l } } = \left( R ^ { \mathrm { w a l l } } ( \phi _ { s } ) \right) ^ { 2 } . } \end{array}\tag{38}
$$

Both estimators depend directly only on $\phi _ { s }$ . The fluid temperature networks receive their influence indirectly through the pointwise temperature- and normal-flux-continuity losses on $\Gamma _ { h s }$ and $\Gamma _ { s c }$ . The implementation uses a total of $\dot { N _ { h s } } + N _ { s c } =$ 120,000 area-uniform samples across the two interfaces.

## E Reference Solutions

## E.1 Reference Solver

Reference temperature and flow fields are generated using ANSYS CFX 2023 R1. We use its coupled finite-volume solver on unstructured tetrahedral meshes and the High Resolution advection scheme. Five prismatic inflation layers are placed next to the walls, with first-layer height 0.002 and growth rate 1.2. The global element size is 0.02 and is refined to a maximum of 0.01 near the boundaries, giving approximately $1 0 ^ { 7 }$ volume elements for each TPMS case. Straight inlet and outlet extensions of three TPMS unitcell lengths are included. Iterations continue until the RMS equation residuals fall below $1 0 ^ { - 8 }$ . Material properties and boundary conditions follow Tables 5 and 4. Reference fields are used only for post-training evaluation and do not afect network optimization or any training decision.

## F Experimental Configuration

## F.1 Networks, Base Losses, and Sequential Training

Two independent networks $\mathcal { N } _ { h } ^ { \mathrm { f i o w } } ( \mathbf { x } ; \psi _ { h } )$ and $\mathcal { N } _ { c } ^ { \mathrm { f i o w } } ( \mathbf { x } ; \psi _ { c } )$ represent $\left( { \bf u } _ { h } , p _ { h } \right)$ and $\left( \mathbf { u } _ { c } , p _ { c } \right)$ , while $\mathcal { N } _ { m } ^ { \mathrm { t e m p } } ( \bar { \bf x } ; \phi _ { m } )$ represents the temperature in each medium $m \in \{ h , s , c \}$ . Let $\phi : = ( \phi _ { h } , \phi _ { s } , \phi _ { c } )$ collect the three temperature-network parameter vectors. Each method uses the same field decomposition and complete CHT boundary/interface conditions. Cova first trains the two flow networks using MUSA-PINN and then freezes them; only $\phi$ is updated in the thermal stage. The flow and thermal objectives below are therefore optimized sequentially, not added into one simultaneous loss.

Flow-stage base terms. For each $f \in \{ h , c \}$ , define the continuity and momentum residuals

$$
r _ { f } ^ { \mathrm { d i v } } = \nabla \cdot \mathbf { u } _ { f } , \qquad \mathbf { r } _ { f } ^ { \mathrm { m o m } } = ( \mathbf { u } _ { f } \cdot \nabla ) \mathbf { u } _ { f } + \nabla p _ { f } - \frac { 1 } { R e _ { f } } \nabla ^ { 2 } \mathbf { u } _ { f } .\tag{39}
$$

Let $\mathcal { X } _ { f } ^ { u } ~ \subset ~ \Omega _ { f }$ contain $N _ { f } ^ { u }$ interior points. The standard strong-form flow loss is

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { P D E } } ^ { u } = \displaystyle \sum _ { f \in \{ h , c \} } \frac { \lambda _ { \mathrm { d i v } } } { N _ { f } ^ { u } } \sum _ { \mathbf { x } \in \mathcal { X } _ { f } ^ { u } } \vert r _ { f } ^ { \mathrm { d i v } } ( \mathbf { x } ) \vert ^ { 2 } } \\ & { \quad \quad \quad \quad + \displaystyle \sum _ { f \in \{ h , c \} } \frac { \lambda _ { \mathrm { m o m } } } { N _ { f } ^ { u } } \sum _ { \mathbf { x } \in \mathcal { X } _ { f } ^ { u } } \Vert \mathbf { r } _ { f } ^ { \mathrm { m o m } } ( \mathbf { x } ) \Vert _ { 2 } ^ { 2 } . } \end{array}\tag{40}
$$

For the prescribed-inlet-velocity, mean-outlet-pressure, and no-slip-wall realization used by the MUSA-based flow stage, let $\bar { \chi _ { f , \mathrm { i n } } ^ { u } } , \ X _ { f , \mathrm { o u t } } ^ { u } ,$ , and $\mathcal { X } _ { f , \mathrm { w } } ^ { u }$ be the corresponding boundary samples; $\chi _ { f , \mathrm { w } } ^ { u }$ includes the fluid–solid interface and every other no-slip fluid wall. Because the pressure scaling in Eq. (17) sets the prescribed dimensionless mean outlet pressure to zero, the boundary loss is

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { b c } } ^ { u } = \displaystyle \sum _ { f \in \{ \bar { h } , \bar { c } \} } \left( \lambda _ { \mathrm { i n } } ^ { u } \mathcal { L } _ { f , \mathrm { i n } } ^ { u } + \lambda _ { \mathrm { o u t } } ^ { u } \mathcal { L } _ { f , \mathrm { o u t } } ^ { u } + \lambda _ { \mathrm { w } } ^ { u } \mathcal { L } _ { f , \mathrm { w } } ^ { u } \right) , } \\ & { \mathcal { L } _ { f , \mathrm { i n } } ^ { u } = \displaystyle \frac { 1 } { N _ { f , \mathrm { i n } } ^ { u } } \displaystyle \sum _ { \mathbf { x } \in \mathcal { X } _ { f , \mathrm { i n } } ^ { u } } \| \mathbf { u } _ { f } ( \mathbf { x } ) - \mathbf { u } _ { f , \mathrm { i n } } ( \mathbf { x } ) \| _ { 2 } ^ { 2 } , } \\ & { \mathcal { L } _ { f , \mathrm { o u t } } ^ { u } = \displaystyle \left| \frac { 1 } { N _ { f , \mathrm { o u t } } ^ { u } } \displaystyle \sum _ { \mathbf { x } \in \mathcal { X } _ { f , \mathrm { o u t } } ^ { u } } p _ { f } ( \mathbf { x } ) \right| ^ { 2 } , } \\ & { \mathcal { L } _ { f , \mathrm { w } } ^ { u } = \displaystyle \frac { 1 } { N _ { f , \mathrm { w } } ^ { u } \times \sum _ { \mathbf { x } \in \mathcal { X } _ { f , \mathrm { w } } ^ { u } } } \| \mathbf { u } _ { f } ( \mathbf { x } ) \| _ { 2 } ^ { 2 } . } \end{array}\tag{41}
$$

Here $N _ { f , b } ^ { u } = | \mathcal { X } _ { f , b } ^ { u } |$ . MUSA-PINN-CHT and Cova use the flow objective

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { f l o w } } ^ { \mathrm { M U S A } } ( t ) = \mathcal { L } _ { \mathrm { P D E } } ^ { u } + \mathcal { L } _ { \mathrm { b c } } ^ { u } + \mathcal { L } _ { \mathrm { w k , d i v } } + \gamma ( t ) \mathcal { L } _ { \mathrm { w k , m o m } } , } \end{array}\tag{42}
$$

where the last two terms are the multiscale weak continuity and momentum losses and $\gamma ( t )$ is the stage switch defined by MUSA-PINN (Zhang et al. 2026). The other end-to-end baselines train their own flow networks with their published mechanisms, as specified below; they do not receive flow weights from another method.

Thermal-stage pointwise terms. With the frozen velocities $\bar { \mathbf { u } } _ { f }$ , define

$$
\begin{array} { r l } & { r _ { f } ^ { T } = \bar { \mathbf { u } } _ { f } \cdot \nabla T _ { f } - \frac { 1 } { P e _ { f } } \nabla ^ { 2 } T _ { f } , \quad f \in \{ h , c \} , } \\ & { r _ { s } ^ { T } = \nabla ^ { 2 } T _ { s } . } \end{array}\tag{43}
$$

At each thermal update, let $\mathcal { X } _ { f } ^ { T } \subset \Omega _ { f }$ and $\mathcal { X } _ { s } ^ { T } \subset \Omega _ { \varepsilon }$ be the finite energy-equation collocation sets. With $N _ { f } ^ { T } ~ = ~ | \mathcal { X } _ { f } ^ { T } |$ and $N _ { s } ^ { T } = | \mathcal { X } _ { s } ^ { T }$ |, the strong-form energy loss is

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { P D E } } ^ { T } = \displaystyle \sum _ { f \in \{ h , c \} } \frac { 1 } { N _ { f } ^ { T } } \sum _ { \mathbf { x } \in \mathcal { X } _ { f } ^ { T } } | r _ { f } ^ { T } ( \mathbf { x } ) | ^ { 2 } } \\ & { \quad \quad \quad \quad + \displaystyle \frac { 1 } { N _ { s } ^ { T } } \sum _ { \mathbf { x } \in \mathcal { X } _ { s } ^ { T } } | r _ { s } ^ { T } ( \mathbf { x } ) | ^ { 2 } . } \end{array}\tag{44}
$$

Let $\boldsymbol { B } _ { T }$ be the set of non-interface thermal boundary components. For each $\beta \in B _ { T }$ , let $m ( \beta ) \in \{ h , s , c \}$ denote its medium, $\mathcal { X } _ { \beta } ^ { T }$ its $N _ { \beta } ^ { T }$ samples, and $\mathcal { B } _ { \beta } [ T _ { m ( \beta ) } ] = g _ { \beta }$ its prescribed boundary operator. Its empirical residual is evaluated explicitly as

$$
\mathcal { L } _ { \mathrm { b c } } ^ { T } = \sum _ { \beta \in \mathcal { B } _ { T } } \frac { \omega _ { \beta } } { N _ { \beta } ^ { T } } \sum _ { { \bf x } \in \mathcal { X } _ { \beta } ^ { T } } \big \vert \mathcal { B } _ { \beta } [ T _ { m ( \beta ) } ] ( { \bf x } ) - g _ { \beta } ( { \bf x } ) \big \vert ^ { 2 } ,\tag{45}
$$

where $\omega _ { \beta }$ is the relative weight of boundary component $\beta .$ Under the temperature scaling in Eq. (17), the two inlet residuals are $T _ { h } - \mathrm { i }$ and $T _ { c }$ , respectively. The external-solid residual is $\kappa _ { s } \nabla T _ { s } \cdot \mathbf { n } _ { s }$ . Table 4 specifies the boundary operator imposed on every component.

For each interface $\bar { \Gamma _ { f } }$ defined in Sec. A, let $\mathcal { X } _ { \Gamma _ { f } }$ contain $N _ { \Gamma _ { f } }$ paired interface points. The empirical interface losses are

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { \Gamma , T } = \sum _ { f \in \{ h , c \} } \frac { 1 } { N _ { \Gamma _ { f } } } \sum _ { \mathbf { x } \in \mathcal { X } _ { \Gamma _ { f } } } | T _ { f } - T _ { s } | ^ { 2 } , } \\ { \displaystyle \mathcal { L } _ { \Gamma , q } = \sum _ { f \in \{ h , c \} } \frac { 1 } { N _ { \Gamma _ { f } } } \sum _ { \mathbf { x } \in \mathcal { X } _ { \Gamma _ { f } } } | \nabla T _ { f } \cdot \mathbf { n } _ { f } + \kappa _ { s } \nabla T _ { s } \cdot \mathbf { n } _ { s } | ^ { 2 } . } \end{array}\tag{47}
$$

All fields and gradients in Eqs. (46)–(47) are evaluated at the summation point x. Thus the complete pointwise thermal objective is

$$
\mathcal { L } _ { \mathrm { p t } } = \lambda _ { \mathrm { P D E } } \mathcal { L } _ { \mathrm { P D E } } ^ { T } + \lambda _ { \mathrm { b c } } \mathcal { L } _ { \mathrm { b c } } ^ { T } + \lambda _ { \Gamma , T } \mathcal { L } _ { \Gamma , T } + \lambda _ { \Gamma , q } \mathcal { L } _ { \Gamma , q } .\tag{48}
$$

Each component is normalized by its own sample count before applying the loss weights. The reported network, optimization, and method-specific quadrature settings are summarized in Table 8.

## F.2 Baseline Selection and Fairness

The end-to-end comparison includes E-MPINN (Lee et al. 2025), CHT adaptations of RoPINN (Wu et al. 2024) and CoPINN (Duan et al. 2025), MUSA-PINN-CHT (Zhang et al. 2026), and Cova-PINN. E-MPINN provides a CHT-specific multi-network baseline; RoPINN-CHT and CoPINN-CHT represent region-based and curriculum-style PINN optimization; and MUSA-PINN-CHT provides the closest structural baseline for complex-geometry flow. It uses MUSA-PINN for both flow networks and the standard pointwise objective for the three temperature networks.

All methods use the same five-field decomposition, network capacity, complete CHT physics, SOAP with a learning rate of $1 0 ^ { - 3 }$ , and 10,000 total updates. The baselines jointly optimize their flow and temperature networks for the full schedule. Cova-PINN uses 5,000 flow-stage updates followed by 5,000 thermal-stage updates with the learned flow fields frozen during thermal optimization. The base strong-form collocation sets are identical across methods; MUSA-PINN-CHT and Cova-PINN additionally use the same weak-form fluid sampling configuration. Method-specific operations required by RoPINN, CoPINN, and Cova-PINN are retained. The final iterate is evaluated without early stopping or model selection. Thus the comparison matches total optimization updates rather than wall-clock time; the resulting costs are reported in Table 11.

<table><tr><td>Category</td><td>Flow stage</td><td>Thermal stage</td></tr><tr><td>Input/output</td><td> $( x , y , z ) \mapsto ( u , v , w , p )$ </td><td> $( x , y , z ) \mapsto T _ { m }$ </td></tr><tr><td>Networks</td><td>Two, one per fluid</td><td>Three, one per medium</td></tr><tr><td>Architecture</td><td>3→60 Fourier features, then 5× 256 tanh layers</td><td>3→60 Fourier features, then 5× 256 tanh layers</td></tr><tr><td>Input encoding</td><td>Linear Fourier frequencies in [1, 2.5] (Tancik et al. 2020)</td><td>Linear Fourier frequencies in [1, 2.5] (Tancik et al. 2020)</td></tr><tr><td>Initialization</td><td>Xavier-normal weights; zero biases</td><td>Xavier-normal weights; zero biases</td></tr><tr><td>Optimizer</td><td>SOAP (Wang et al. 2025), default parameters</td><td>SOAP (Wang et al. 2025), default parameters</td></tr><tr><td>Learning rate</td><td> $1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>Updates</td><td>5,000</td><td>5,000</td></tr><tr><td>Pointwise loss weights</td><td>See Table 9</td><td>See Table 9</td></tr><tr><td>Local CV quadrature</td><td></td><td> $r = 0 . 0 5 ;$  50 balls; 10,000 candidates/ball; refresh every 500 updates</td></tr><tr><td>Paired-wall quadrature</td><td></td><td>120,000 samples across both solid-side interfaces</td></tr><tr><td>Solid-mean schedule</td><td></td><td> $N _ { R } = 2 , 5 0 0 ; \lambda _ { R } ( t ) = \lambda _ { R } ^ { ( 0 ) } \operatorname* { m a x } ( 1 - t / N _ { R } , 0 )$ </td></tr></table>

Table 8: Cova-PINN network, optimization, and conservation-quadrature configuration. The flow networks are trained first and frozen before thermal optimization; all subnetworks use the same Fourier-feature MLP backbone.
<table><tr><td>Stage/objective</td><td>Weight(s)</td><td>Value(s)</td></tr><tr><td>Flow strong continuity and momentum</td><td> $\lambda _ { \mathrm { d i v } } , \lambda _ { \mathrm { m o m } }$ </td><td>1, 0.1</td></tr><tr><td>Flow inlet, outlet, and no-slip wall</td><td> $\lambda _ { \mathrm { i n } } ^ { u } , \lambda _ { \mathrm { o u t } } ^ { u } , \lambda _ { \mathrm { w } } ^ { u }$ </td><td>10, 10, 10</td></tr><tr><td>MUSA weak continuity (large/medium/small)</td><td> $\lambda ^ { L / M / S }$   $\lambda _ { \mathrm { w k , d i v } }$ </td><td>100, 100, 100</td></tr><tr><td>MUSA weak momentum (large/medium/small)</td><td> $\Upsilon \Upsilon / \overrightarrow { M } / S$   $\lambda _ { \mathrm { w k , m o m } }$ </td><td>4, 25, 100</td></tr><tr><td>Thermal PDE and boundary conditions</td><td> $\lambda _ { \mathrm { P D E } } , \lambda _ { \mathrm { b c } }$ </td><td>1, 1</td></tr><tr><td>Interface temperature and heat flux</td><td> $\lambda _ { \Gamma , T } , \lambda _ { \Gamma , q }$ </td><td>1, 1</td></tr><tr><td>Local composite control volumes</td><td> $\lambda _ { L }$ </td><td>10 (fixed)</td></tr><tr><td>Global paired-wall closure</td><td> $\lambda _ { G }$ </td><td>1 (fixed)</td></tr><tr><td>Solid-mean regularization</td><td> $\lambda _ { R } ( t )$ </td><td> $\lambda _ { R } ^ { ( 0 ) } = 1 ;$  linear decay to 0 by update 2,500</td></tr></table>

Table 9: Loss weights used by Cova-PINN and the baseline objectives. Strong-form weights are shared across methods; MUSA weak-form flow weights are shared by MUSA-PINN-CHT and Cova-PINN, while the local, global, and regularization weights apply to Cova-PINN.

## G Evaluation Metrics

## G.1 Temperature-Field and Outlet Errors

Let ${ \boldsymbol { \mathcal { X } } } _ { m }$ contain the valid uniformly sampled volumeevaluation points in medium $m \in \{ h , s , c \}$ , and let $N _ { \Omega } =$ $\sum _ { m } | \mathcal { X } _ { m } |$ . The pooled full-field temperature error used in the main paper is

$$
E _ { \mathrm { f i e l d } } = \frac { \Delta T } { N _ { \Omega } } \sum _ { m \in \{ h , s , c \} } \sum _ { \mathbf x \in \mathcal X _ { m } } \left| T _ { m } ^ { \mathrm { p r e d } } ( \mathbf x ) - T _ { m } ^ { \mathrm { r e f } } ( \mathbf x ) \right| .\tag{49}
$$

Because the temperature networks output dimensionless values, multiplication by $\Delta T$ converts the error to kelvin. Equa-

tion (49) matches the reported volume\_mae: every valid evaluation point contributes once, rather than first averaging the three media.

For outlet quadrature points $\mathbf { x } _ { f , k }$ with area weights $\begin{array} { r } { a _ { f , k } , } \end{array}$ define a bulk-temperature estimator using a specified velocity field $\mathbf { v } _ { f } \colon$

$$
\mathcal { B } _ { f } [ T ; \mathbf { v } _ { f } ] = \frac { \sum _ { k } a _ { f , k } \rho _ { f } \bigl ( \mathbf { v } _ { f } \bigl ( \mathbf { x } _ { f , k } \bigr ) \cdot \mathbf { n } _ { f } \bigr ) T \bigl ( \mathbf { x } _ { f , k } \bigr ) } { \sum _ { k } a _ { f , k } \rho _ { f } \bigl ( \mathbf { v } _ { f } \bigl ( \mathbf { x } _ { f , k } \bigr ) \cdot \mathbf { n } _ { f } \bigr ) } .\tag{50}
$$

The headline outlet metric is the mean absolute bulk outlettemperature error averaged over the hot and cold streams. It uses the same reference velocity field in the prediction and reference estimators:

$$
e _ { \mathrm { o u t } , f } = \Delta T \left| \mathcal { B } _ { f } [ T _ { f } ^ { \mathrm { p r e d } } ; \mathbf { u } _ { f } ^ { \mathrm { r e f } } ] - \mathcal { B } _ { f } [ T _ { f } ^ { \mathrm { r e f } } ; \mathbf { u } _ { f } ^ { \mathrm { r e f } } ] \right| ,\tag{51}
$$

$$
E _ { \mathrm { o u t } } = { \frac { 1 } { 2 } } \left( e _ { \mathrm { o u t } , h } + e _ { \mathrm { o u t } , c } \right) .\tag{52}
$$

The common mass-flux weights isolate outlet temperature accuracy from diferences among independently trained flow networks. $E _ { \mathrm { o u t } }$ is reported in kelvin.

## G.2 Heat-Rate Accuracy and Device Closure

For end-to-end energy diagnostics, each prediction instead uses its own velocity field in Eq. (50). Let $\chi _ { f } = C _ { f } / C _ { \mathrm { m i n } }$

<table><tr><td>Method</td><td>P</td><td>G</td><td>D</td><td>IWP</td></tr><tr><td>E-MPINN</td><td>3.03</td><td>3.76</td><td>13.09</td><td>58.15</td></tr><tr><td>RoPINN-CHT</td><td>16.12</td><td>6.89</td><td>13.81</td><td>14.43</td></tr><tr><td>CoPINN-CHT</td><td>3.79</td><td>6.61</td><td>12.96</td><td>167.03</td></tr><tr><td>MUSA-PINN-CHT</td><td>7.95</td><td>3.14</td><td>1.63</td><td>5.74</td></tr><tr><td>Cova-PINN</td><td>1.57</td><td>2.92</td><td>1.49</td><td>4.21</td></tr></table>

Table 10: Reference-normalized two-sided heat-duty error $E _ { Q }$ (%) on four TPMS exchangers. Lower is better; bold and underline denote the best and second-best results. Each value evaluates the complete end-to-end model produced by the corresponding training protocol.

The normalized hot-side heat release and cold-side heat gain are

$$
Q _ { h } = \chi _ { h } \left( T _ { h , \mathrm { i n } } ^ { b } - T _ { h , \mathrm { o u t } } ^ { b } \right) ,\tag{53}
$$

$$
Q _ { c } = \chi _ { c } \left( T _ { c , \mathrm { o u t } } ^ { b } - T _ { c , \mathrm { i n } } ^ { b } \right) .\tag{54}
$$

where $T _ { f , \mathrm { o u t } } ^ { b } = \mathcal { B } _ { f } [ T _ { f } ^ { \mathrm { p r e d } } ; \mathbf { u } _ { f } ^ { \mathrm { p r e d } } ]$ for a prediction and the analogous reference fields are used for CFX. The normalization is $Q _ { 0 } = C _ { \operatorname* { m i n } } \Delta T$

The reference-normalized two-sided heat-duty error is

$$
E _ { Q } = 1 0 0 \sqrt { \frac { ( Q _ { h } ^ { \mathrm { p r e d } } - Q _ { h } ^ { \mathrm { r e f } } ) ^ { 2 } + ( Q _ { c } ^ { \mathrm { p r e d } } - Q _ { c } ^ { \mathrm { r e f } } ) ^ { 2 } } { 2 } } \mathcal { Y } _ { 0 } .\tag{55}
$$

Unlike a closure residual, $E _ { Q }$ evaluates whether the amount of heat transferred on each side agrees with the CFX reference.

The independently evaluated device imbalance is

$$
E _ { \mathrm { c l o } } = 1 0 0 | Q _ { h } ^ { \mathrm { p r e d } } - Q _ { c } ^ { \mathrm { p r e d } } | \% .\tag{56}
$$

This outlet-based metric is independent of the paired-wall closure term used for training. None of $E _ { \mathrm { o u t } } , E _ { Q } $ , or $E _ { \mathrm { c l o } }$ is included in the training objective; all three are evaluated only after the fixed training budget.

## H Complete Results

This section complements the main comparison with the complete TPMS heat-duty results.

## H.1 Reference-Normalized Heat-Duty Accuracy

Table 10 reports $E _ { Q }$ for every method–topology pair. Unlike device closure, this metric compares the predicted heat duty on both streams against the CFX reference and therefore measures whether the transferred energy is quantitatively accurate.

## H.2 Full Longitudinal Field Comparison

Figure 6 extends the Diamond mid-plane comparison in the main paper to every baseline. Under matched temperature and absolute-error color ranges, the figure shows how prediction quality evolves through all five streamwise unit cells rather than only at the outlet.

<table><tr><td>Method</td><td>Training stage</td><td>Updates</td><td>Time (h)</td></tr><tr><td>E-MPINN</td><td>Joint</td><td>10,000</td><td>20.43</td></tr><tr><td>RoPINN-CHT</td><td>Joint</td><td>10,000</td><td>27.87</td></tr><tr><td>CoPINN-CHT</td><td>Joint</td><td>10,000</td><td>21.85</td></tr><tr><td>MUSA-PINN-CHT</td><td>Joint</td><td>10,000</td><td>32.76</td></tr><tr><td>Cova-PINN (flow)</td><td>Flow stage</td><td>5,000</td><td>25.12</td></tr><tr><td>Cova-PINN (thermal)</td><td>Thermal stage</td><td>5,000</td><td>16.18</td></tr><tr><td>Cova-PINN (total)</td><td>Sequential total</td><td>10,000</td><td>41.30</td></tr></table>

Table 11: End-to-end wall-clock cost under a common 10,000-update schedule. All runs use one NVIDIA A40 GPU. Baselines train jointly, whereas Cova-PINN reports its sequential flow and thermal stages together with their total.

## I Computational Cost

We report wall-clock time to quantify the additional optimization cost of the staged Cova-PINN pipeline and its conservation quadrature. All measurements use one NVIDIA A40 GPU and the fixed 10,000-update protocol described in Sec. F.

Times measure the fixed training schedule rather than time to a convergence threshold. The comparison matches total update count, not wall-clock time or the number of updates applied to each individual field. Cova-PINN’s additional cost reflects its staged optimization and method-specific localcontrol-volume and paired-wall quadrature.

![](images/6fe797ed735fc6b794b72aa8d28a77665a78216764c7f825ed78aa66fe156bed.jpg)

![](images/6855ecba359f291809227a604c08e4685b645e0f86bd948f39783304ce8a1da2.jpg)

![](images/77e2ebdd7d41ca561f74ebfb51dba2680715338cc6b4f16dea38631e8693e12e.jpg)

![](images/04805b170f99ebe3684abcbd5bbc90ff0757c5e9aa4cb862abd02afcbbe3de64.jpg)

![](images/1254a81f93fa6c7c54fab52cdc8781e4585b3cdfb4277750cf062e9b9071ade7.jpg)

![](images/e7472dc9ae05c9d675fa6c01825a881a8368ba8d5fc7357098e2ef8e7ef962c5.jpg)

![](images/830e1bc4be07c9c9e7128f34baa324be07ed54f8eab0ea2df48369c6a9da161c.jpg)

![](images/1be8d5599ec3b006a4a3492b9e82c4b8044c9a9a26a1c2b5459177fe099ca7b2.jpg)

![](images/15462760f17510ec338adb226591cd2c63d0e2cc6c10f12d9ef5fa782025fffd.jpg)

![](images/b2e5eccfc9d42bd6e953d4fee55fedbf2467ea71a30427c646ee4d266d8f8a3f.jpg)

![](images/e121aea30dd5719ed59b7cc52a6a801b73f731f9f696c30e644578cfd20a2e58.jpg)

![](images/c45eb3bfabc2223e78dcbba21024d11fcac7e0207ff76876ebead6410a831c9d.jpg)  
Figure 6: Full longitudinal comparison on the Diamond exchanger $\mathbf { a t } \ z \ = \ 0 . 5 .$ . The left column shows the predicted temperature fields of all methods and the CFX reference; the right column shows the corresponding pointwise absolute-error fields. Temperature and absolute-error panels use matched color ranges. The final panel reports $\begin{array} { r } { \Delta \mathbf { \bar { E } } = | T _ { \mathrm { M U S A } } - T _ { \mathrm { C F X } } | - } \end{array}$ $\left| T _ { \mathrm { C o v a } } - T _ { \mathrm { C F X } } \right|$ ; positive red values indicate lower Cova-PINN error, whereas negative blue values indicate lower MUSA-PINN-CHT error.