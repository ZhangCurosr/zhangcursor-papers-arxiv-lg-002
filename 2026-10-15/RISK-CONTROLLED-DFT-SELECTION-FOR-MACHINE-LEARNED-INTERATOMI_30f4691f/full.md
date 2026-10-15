# RISK-CONTROLLED DFT SELECTION FOR MACHINE-LEARNED INTERATOMIC POTENTIALS

A PREPRINT

Kun Ryu<sup>1,†</sup> Yeo Jin Jung<sup>2,†</sup> Shyue Ping Ong<sup>3</sup> Claire Donnat<sup>2</sup> Ying Shirley Meng<sup>1,∗</sup>

<sup>1</sup>Pritzker School of Molecular Engineering, The University of Chicago

<sup>2</sup>Department of Statistics, The University of Chicago

<sup>2</sup>Department of Material Science and Engineering, National University of Singapore

<sup>†</sup>Equal contribution <sup>∗</sup>Corresponding author

## ABSTRACT

Automated materials discovery requires evaluating large pools of candidate configurations, with density functional theory (DFT) as a major throughput bottleneck. Pretrained machine-learned interatomic potentials (MLIPs) approximate DFT cheaply, but their errors vary by orders of magnitude across configurations, making it difficult to decide when to trust the MLIP predictions and when to fall back to DFT. We frame this decision as a risk-controlled selection problem and solve it within a conformal risk control (CRC) framework. Each configuration is mapped from atomic numbers and Cartesian coordinates to a fixed geometric descriptor, which a lightweight predictor uses to estimate MLIP–DFT disagreement score in energy, average force, and maximum force. CRC then calibrates this score into a selection threshold for DFT labeling such that the MLIP error on the unselected configurations, which keep their MLIP labels, stays below a user-chosen fraction α of the expected error. Under an identical guarantee across eight dataset–MLIP pairs, our predictor routes 7% and 14% fewer configurations to DFT at α = 0.3 than committee baselines. Because committee disagreement overlooks errors the MLIPs share, our gains are largest when the DFT labels and MLIP training data differ in DFT functional and basis set, and narrow when the two agree. Without model access, the structure-only predictor also ranks errors as good as target MLIP’s latent representation and outperforms widely used descriptors like Smooth Overlap of Atomic Positions (SOAP) and Atom-Centered Symmetry Functions (ACSF).

## 1 Introduction

Density functional theory (DFT) provides accurate energies and atomic forces from electronic structure and is a central computational tool across chemistry and materials science [Kohn and Sham, 1965]. However, conventional DFT scales approximately as O(N<sup>3</sup>) with system size, where N denotes the number of atoms, limiting the number of atomic configurations that can be evaluated in large-scale applications such as molecular dynamics and high-throughput screening [Bowler and Miyazaki, 2012, Curtarolo et al., 2013]. Machine-learned interatomic potentials (MLIPs) address this bottleneck by approximating DFT energies and forces at a small fraction of the computational cost, typically with linear scaling in N for local architectures [Behler and Parrinello, 2007, Unke et al., 2021, Batatia et al., 2022]. Larger training datasets and increased model capacity have substantially improved the accuracy and generality of MLIPs, making pretrained models practical surrogates for DFT across broad regions of chemical space [Batatia et al., 2025, Wood et al., 2025]. However, no pretrained MLIP is uniformly accurate, and prediction errors can vary substantially across molecules, geometries, and chemical environments [Fu et al., 2023, Deng et al., 2025, Perez et al., 2025].

This creates a basic decision problem whenever an MLIP should be used to accelerate a DFT workflow. The main question then becomes: Given a large pool of molecular configurations, which configurations can safely be evaluated with the MLIP and which should be sent to DFT? Evaluating every configuration with DFT increases the computational cost, while accepting every MLIP prediction results in configurations with large energy or force errors. A natural strategy is to estimate the error for each configuration and send only the unreliable configurations to DFT [Jinnouchi et al., 2019, Liu et al., 2026]. The most common estimate is the disagreement within a committee of MLIPs [Smith et al., 2018, Schran et al., 2020]. However, disagreement only flags errors on which the models differ, and errors that all models share remain invisible. The challenge is that the true error is unavailable without performing the expensive DFT calculation [Perez et al., 2025]. Moreover, an error predictor that ranks configurations well does not by itself provide a finite-sample guarantee on the error remaining among configurations that are not evaluated with DFT [Angelopoulos et al., 2024]. Predicting error is not the same as knowing where to stop trusting the MLIP.

![](images/bfc0b051f5f02acad639398982f92237262b9a8ce3f35f8bbd58c19bab28b8a6.jpg)  
Figure 1: Overview of the MLIP–DFT selection framework. A molecular configuration R (caffeine in this example) is converted into a chemistry-derived structural vector and passes through a lightweight predictor to estimate energy and force residuals. These estimates are normalized and combined into a predicted error score sˆ(R). Conformal risk control then calibrates a threshold that retains reliable MLIP predictions and sends higher-risk configurations to DFT. The score and the capped loss are fixed before calibration.

In this work, we treat MLIP–DFT routing as risk-controlled selection for a frozen pretrained MLIP and solve it using conformal risk control (CRC) [Bates et al., 2021, Angelopoulos et al., 2024] (Figure 1). The method consists of two steps. First, a lightweight predictor maps a structural representation rooted in analytical-chemistry consensus to the expected MLIP–DFT error in three channels: energy, average force, and maximum force. The channels are combined into a single predicted error score. Second, CRC uses a labeled calibration set to turn the score into a threshold, and every configuration scoring above this threshold is sent to DFT for labeling. Note the exchangeable unit of calibration is a molecule identity (the set of all sampled geometries of one molecule). Routing decisions, however, are made for individual configurations that potentially share the same molecule identity. Calibration controls validity. Under the exchangeability of molecule identities and for any error score fixed before calibration, the expected error left in the MLIP predictions of a test identity is at most a user-chosen fraction α of the average per-identity error on the fitting set. The quality of the error score predictor, or ranker, controls the efficiency, that is, how many DFT calls are needed to meet this budget, while CRC decides the score threshold that validates the risk control.

Our contributions are as follows: (i) Risk-controlled routing. We cast MLIP–DFT routing as conformal risk control over molecular identities, calibrating a threshold for any given MLIP–DFT error score, so that the configurations scoring above it are sent to DFT while the MLIP error on the rest stays within a user-set budget. Score quality affects the number of DFT calls needed, not the guarantee (Section 2.4). (ii) A model-agnostic error ranker. We propose a lightweight error predictor built on a structure-only, chemistry-informed descriptor. It ranks MLIP errors about as well as the target $\mathbf { M L } \mathbf { \dot { I P } } ^ { \prime } \mathbf { s }$ own latent representation (Sections 2.2–2.3). (iii) A guarantee-matched comparison of DFT cost. At the same level of risk control, we compare different error scores by the number of DFT calls they require across eight dataset–MLIP pairs. Our structure-based error score predictor requires fewer DFT calls than committee-based baselines. The gain is largest when the user’s DFT reference differs from the MLIPs’ training reference, and the gap closes when they match. We also show that partial DFT replacement can break error cancellation in energy differences (Section 3.2, Appendix J.1).

Related work. The closest precedents are Liu et al. [2026], who send structures to DFT using an empirically chosen threshold on ensemble uncertainty, and Conformal Arbitrage [Overman and Bayati, 2025], which also calibrates a deferral threshold with CRC. We differ by using a separately trained error predictor and controlling a molecular identity-level loss. Appendix A gives a full review of the related literature.

## 2 Method

## 2.1 Setup and notation

Let R denote a configuration with $N = N ( R )$ atoms, atomic numbers $Z _ { j } ( R )$ and positions $\mathbf { x } _ { j } ( R ) \in \mathbb { R } ^ { 3 } , j = 1 , \dotsc , N .$ DFT assigns R an energy $e ^ { * } ( R ) \in \mathbb { R }$ and per-atom forces $\mathbf { f } _ { j } ^ { * } ( R ) \in \mathbb { R } ^ { \tilde { 3 } } ,$ ; a pretrained $\bf M L I P$ outputs predictions $\hat { e } ( R )$

and $\hat { \mathbf { f } } _ { j } ( R )$ cheaply. The MLIP leaves residuals,

$$
\delta ( R ) = e ^ { * } ( R ) - \hat { e } ( R ) \in \mathbb { R } , \qquad \mathbf { g } _ { j } ( R ) = \mathbf { f } _ { j } ^ { * } ( R ) - \hat { \mathbf { f } } _ { j } ( R ) \in \mathbb { R } ^ { 3 } , \quad j = 1 , \dots , N .\tag{1}
$$

MLIP and DFT labels do not share an isolated-atom energy convention, so we work with the aligned energy residual $\Delta E ( R ) = \delta ( R ) - \phi ( R ) ^ { \top } \hat { a } .$ , where $\phi ( R )$ holds element counts and an intercept, and aˆ contains the composition coefficients, fitted on ${ \mathcal { D } } _ { \mathrm { f i t } }$ and frozen before calibration (Appendix D). Because $\phi ( R ) ^ { \top } \hat { a }$ depends only on composition, the alignment leaves force residuals and energy differences between configurations of the same composition unchanged. Here, DFT is the reference the user trusts, which may use a different functional or basis set than the ${ \bf M L I P } \mathrm {  { s } }$ training reference; the residual therefore also captures this systematic difference.

The routing problem. Running DFT on a configuration drives its residual to zero, but costs far more than an MLIP forward pass. Given a frozen MLIP and an unlabeled pool of configurations, our goal is to send only a small subset of uncertain configurations to DFT and to accept MLIP predictions for the rest. However, a problem arises: deciding whether a configuration’s residual is large enough to be sent to DFT requires the very DFT calculation we are trying to avoid. We therefore need a label-free proxy for the residual, together with a principled rule for how large that proxy must be before a DFT calculation is worth its cost.

Data splits. We use three disjoint sets. ${ \mathcal { D } } _ { \mathrm { f i t } }$ and $\mathcal { D } _ { \mathrm { c a l i b } }$ are DFT-labeled molecule identities that the MLIP was not trained on. We use ${ \mathcal { D } } _ { \mathrm { f i t } }$ to fit the MLIP error score predictor (Section 2.3) and $\mathcal { D } _ { \mathrm { c a l i b } }$ to calibrate the selection threshold (Section 2.4). $\mathcal { D } _ { \mathrm { t e s t } }$ is the unlabeled test pool where the selection rule is applied. All configurations of the same molecule identity are assigned to the same split (Section 2.4).

Our method proceeds in two steps. First, we use the DFT-labeled set ${ \mathcal { D } } _ { \mathrm { f i t } }$ to define a single error score that combines the MLIP energy residual and force residuals, and train a lightweight predictor sˆ to estimate that error score from structural information alone; every configuration in $\mathcal { D } _ { \mathrm { t e s t } }$ then receives an estimated error score. Second, we use the error scores on $\mathcal { D } _ { \mathrm { c a l i b } }$ to calibrate a risk-controlling threshold, and every configuration in $\mathcal { D } _ { \mathrm { t e s t } }$ scoring above the threshold is sent to DFT (Section 2.4). Sections 2.2 and 2.3 describe the two halves of the first step, and Section 2.4 explains the conformal threshold and its risk-control guarantee.

## 2.2 Structural vector

We define a structural vector constructed using only atomic numbers $Z _ { j }$ and Cartesian coordinates $\mathbf { x } _ { j } .$ . No other features, including DFT quantities, learned embeddings, or MLIP features, enter its construction. The feature families are motivated by structural concepts from physical and analytical chemistry [Musil et al., 2021] and by established fixed-form descriptors such as atom-centered symmetry functions, the Coulomb matrix, and SOAP [Behler, 2011, Rupp et al., 2012, Bartók et al., 2013]. They encode composition, local coordination, pair-distance distortion, short-range crowding, angular distortion, radial structure, and elemental composition of neighboring coordination shells. These are geometric variables that chemists routinely use to reason about molecular structure from nuclear magnetic resonance, Fourier-transform infrared spectroscopy, and Raman spectroscopy [Geue et al., 2022, Stavitski and Weckhuysen, 2010].

The descriptor concatenates nine fixed feature families into 350 dimensions (Table 2 in Appendix B). Coarse, discrete families count elements, element-resolved coordination states, and compressed, nominal, or stretched element-pair contacts. Continuous families then resolve what the counts hide: atomic-number-weighted short contacts, the few worst local environments, a smooth radial distribution, unweighted inverse-distance proximity at several powers, soft crowding across four distance scales, and the elemental identity of each atom’s neighborhood. Exact definitions are also given in Appendix B.

## 2.3 A joint energy–force error predictor

Using the structural vector of Section 2.2 as the input, we train a lightweight error predictor that estimates how accurate the MLIP energy and force prediction is. For a configuration R, the predictor estimates three targets: energy residual, average force residual, and maximum force residual,

$$
u _ { E } ( R ) = | \Delta E ( R ) | , \qquad u _ { \mathrm { a v g } } ( R ) = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \| \mathbf { g } _ { j } ( R ) \| _ { 2 } , \qquad u _ { \mathrm { m a x } } ( R ) = \operatorname* { m a x } _ { j } \| \mathbf { g } _ { j } ( R ) \| _ { 2 } ,\tag{2}
$$

where $\Delta E ( R )$ and ${ \bf g } _ { j } ( R )$ are the aligned energy and per-atom force residuals of (1). The three targets capture different failure modes, each of which matters in practical molecular dynamics (MD) [Fu et al., 2023]. $u _ { E }$ detects a wrong total energy, $u _ { \mathrm { a v g } }$ detects forces that are broadly inaccurate and mislead the trajectory, and $u _ { \mathrm { m a x } }$ detects a single badly mispredicted atom that crashes the simulation. Because the three residual targets scale differently with atom count $N ,$ , each is divided by a size-dependent scale $\hat { c } _ { t } N ^ { \hat { \beta } _ { t } }$ fitted on ${ \mathcal { D } } _ { \mathrm { f i t } }$ and then frozen, for $t \in \{ E , \mathrm { a v g } , \mathrm { m a x } \}$ (detailed in Appendix D). We write $u _ { t }$ for the normalized targets and $\hat { u } _ { t }$ for their predictions. The normalized residuals are combined into a single error score:

$$
\hat { s } ( R ) = \sum _ { t } w _ { t } \hat { u } _ { t } ( R ) , \qquad s ( R ) = \sum _ { t } w _ { t } u _ { t } ( R ) , \qquad w _ { t } \geq 0 , \qquad \sum _ { t } w _ { t } = 1 ,\tag{3}
$$

where $s ( R )$ denotes the oracle error score. The weights are fixed before calibration and specify the relative importance of energy, average-force, and maximum-force errors for the intended application—e.g. a user who cares more about force accuracy than energy accuracy can set w accordingly. All experiments in this paper use equal weights $w _ { t } = 1 / 3$

## 2.4 Configuration selection by conformal risk control

The error predictor sˆ can rank configurations but carries no guarantee about the true error of the unselected configurations—even a perfect ranking would not say where to stop, since that is a decision about tolerable MLIP error against DFT cost, not a property of the predictor. CRC sidesteps this: the user instead chooses $\alpha ,$ the fraction of the mean per-identity MLIP error they are willing to tolerate, and calibration returns the threshold λ<sup>ˆ</sup> on sˆ that achieves it. This exposes an explicit tradeoff between retained MLIP error and the number of DFT calls—tolerating more error means fewer calls. Sweeping α traces this cost–accuracy curve (Figure $^ { 2 , }$ middle row), and the practitioner picks an operating point on it rather than guessing one in advance.

We follow the conformal risk control (CRC) framework of Angelopoulos et al. [2024], with one adaptation: the exchangeable unit is a molecule identity, not a configuration [Dunn et al., 2023]. We write g for an identity and $\mathcal { R } _ { g }$ for its set of configurations, and assign whole identities to splits $\mathcal { D } _ { \mathrm { f i t } } , \mathcal { D } _ { \mathrm { c a l i b } } , \mathcal { D } _ { \mathrm { t e s t } }$ uniformly at random. When a dataset does not record the parent identity of a configuration, that configuration forms its own group (see Appendix D). We make the following exchangeability assumption:

Assumption 1 (Exchangeability of molecule identities). The calibration identities ${ \mathcal { D } } _ { \mathrm { c a l i b } } = \{ g _ { i } \} _ { i = 1 } ^ { n }$ and the test identities $\mathcal { D } _ { \mathrm { t e s t } } = \{ g _ { i } \} _ { i = n + 1 } ^ { n + m }$ , each carrying its finite configuration set $\mathcal { R } _ { g _ { i } }$ with finite scores, are exchangeable.

Selection, on the other hand, still happens one configuration at a time—a molecule identity may have some of its configurations sent to DFT and the rest left to the MLIP. Let $s ( R )$ be the true error score of Section 2.3 and ${ \hat { s } } ( R )$ its estimate. CRC requires a bounded loss [Bates et al., 2021, Angelopoulos et al., 2024], but s is commonly heavy-tailed, so we fix a truncation level $B _ { \mathrm { c f g } } > 0$ and set

$$
\ell ( R ) = \operatorname* { m i n } \{ s ( R ) , B _ { \mathrm { c f g } } \} .\tag{4}
$$

$B _ { \mathrm { c f g } }$ is a data-driven tolerance: the level past which a configuration’s error is unacceptable and its exact size no longer matters. Since the distribution of error scores is skewed on several of our datasets, we set $B _ { \mathrm { c f g } }$ to the 99.9% quantile of sˆ on ${ \mathcal { D } } _ { \mathrm { f i t } }$ for robustness, rather than to its maximum. At a threshold $\lambda ,$ we send to DFT every test configuration whose predicted error score exceeds λ,

$$
S _ { \lambda } = \{ R \in { \mathcal { D } } _ { \mathrm { t e s t } } : \hat { s } ( R ) > \lambda \} ,\tag{5}
$$

and define the loss of a molecule identity $g$ as the total truncated error of its configurations that are not selected, capped at $B ,$

$$
L ( \lambda ; g ) = \operatorname* { m i n } \Bigl \{ \sum _ { R \in { \mathcal { R } } _ { g } } \ell ( R ) \mathbf { 1 } \{ \hat { s } ( R ) \leq \lambda \} , \ B \Bigr \} .\tag{6}
$$

The loss combines the true truncated error $\ell ( R )$ with the predicted error score ${ \hat { s } } ( R )$ , which enters only through the ordering it induces. As a function of λ, $L ( \lambda ; \dot { g } )$ is non-decreasing and right-continuous, and it is zero once λ is small enough that every configuration goes to DFT—the conditions required by Angelopoulos et al. [2024] after reversing the sign of λ. We sum rather than average over R $\mathcal { R } _ { g }$ because an identity with more unselected configurations left to the MLIP carries proportionally more error exposure.

CRC additionally requires an upper bound on the loss of a single exchangeable unit. Capping at B inside (6) supplies that bound by construction: $\operatorname { s u p } _ { \lambda } L ( \lambda ; g ) ~ \leq ~ B$ . The two constants play distinct roles: $B _ { \mathrm { c f g } }$ bounds the loss of an individual configuration, whereas $B$ bounds the loss of an entire identity. Appendix D details how each is estimated on ${ \mathcal { D } } _ { \mathrm { f i t } }$ . Our goal is to choose λ<sup>ˆ</sup> on a calibration set $\mathcal { D } _ { \mathrm { c a l i b } }$ of n identities so that the expected loss of a test identity $g _ { n + 1 }$ is at most a user-specified level $\epsilon _ { 0 } \colon \mathbb { E } \bigl [ L ( \hat { \lambda } ; g _ { n + 1 } ) \bigr ] \leq \epsilon _ { 0 }$

Table 1: Evaluation datasets. A configuration is a geometry and a molecule identity is the exchangeable unit; configurations per identity differ by nearly an order of magnitude across datasets. MACE-OFF23 contains 228 single-configuration groups without a recorded parent identity.
<table><tr><td>Dataset</td><td></td><td></td><td># configs. # identities # atoms (mean)</td><td>elements</td></tr><tr><td>COMP6-GDB07to09-FULL [Smith et al., 2018]</td><td>36,000</td><td>1,500</td><td>7–27 (17.0) H, C, N, O</td><td></td></tr><tr><td>MACE-OFF23-FULL [Eastman et al., 2023]</td><td>50,195</td><td>15,769</td><td></td><td>3–150 (30.9) H, C, N, O, F, P, S, Cl, Br, I</td></tr><tr><td>QCML-6950-9 [Ganscha et al., 2025]</td><td>62,550</td><td>6,950</td><td></td><td>2–23 (11.0) H, C, N, O, F, P, S, Cl, Br, I</td></tr></table>

Definition 1 (Finite calibrated selection threshold). Let ${ \mathcal { D } } _ { \mathrm { c a l i b } } = \{ g _ { i } \} _ { i = 1 } ^ { n }$ contain n identities, and define the finite extended set of observed configuration scores

$$
{ \mathcal { H } } _ { n } = \{ - \infty \} \cup \bigcup _ { i = 1 } ^ { n } \{ { \hat { s } } ( R ) : R \in { \mathcal { R } } _ { g _ { i } } \} .\tag{7}
$$

Given the budget $\epsilon _ { 0 } > 0 \leq$ , let

$$
\mathcal { F } _ { n } = \left\{ \lambda \in \mathcal { H } _ { n } : \frac { \sum _ { i = 1 } ^ { n } L ( \lambda ; g _ { i } ) + B } { n + 1 } \le \epsilon _ { 0 } \right\} .\tag{8}
$$

The threshold is

$$
\hat { \lambda } = \left\{ \begin{array} { l l } { \operatorname* { m a x } \mathcal { F } _ { n } , } & { \mathcal { F } _ { n } \neq \emptyset , } \\ { - \infty , } & { \mathcal { F } _ { n } = \emptyset . } \end{array} \right.\tag{9}
$$

The selection rule is a threshold on sˆ, so configurations with identical scores are retained or rejected together and only distinct score values are reachable as thresholds. The term $B / ( n + 1 )$ is the finite-sample correction for the unseen test identity: if $\epsilon _ { 0 } < B / ( n + 1 )$ , then $\hat { \lambda } = - \infty$ , and the whole pool is sent to DFT—the rule is never infeasible; it simply forgoes all savings. The calibrated $\hat { \lambda }$ then controls the risk both for a single test identity and for the test pool as a whole.

Proposition 1 (Risk control per identity). Let Assumption 1 hold, let sˆ, $B _ { \mathrm { c f g } }$ and B be fixed given $\mathcal { D } _ { \mathrm { f i t } } .$ . Given the threshold $\hat { \lambda } ( 9 )$ ,for a test pool $\mathcal { D } _ { \mathrm { t e s t } }$ with $m = | \mathcal { D } _ { \mathrm { t e s t } } |$ and a test identity $g _ { n + 1 }$

$$
\mathbb { E } \big [ L \big ( \hat { \lambda } ; g _ { n + 1 } \big ) \big ] \ \leq \ \epsilon _ { 0 } , \qquad \mathbb { E } \left[ \sum _ { g \in \mathcal { D } _ { \mathrm { t e s t } } } L ( \hat { \lambda } ; g ) \right] \ = \ \sum _ { g \in \mathcal { D } _ { \mathrm { t e s t } } } \ \mathbb { E } \big [ L ( \hat { \lambda } ; g ) \big ] \ \leq \ m \epsilon _ { 0 } .\tag{10}
$$

The proof follows that of Angelopoulos et al. [2024], and we provide the full proof in Appendix C.

We set $\epsilon _ { 0 } = \alpha \bar { \ell } _ { \mathrm { f i t } }$ , where $\bar { \ell } _ { \mathrm { f i t } }$ is the mean loss per identity on ${ \mathcal { D } } _ { \mathrm { f i t } }$ . Thus, in expectation, the total error carried by the unselected configurations of a test identity, capped at B, is at most an α fraction of the mean per-identity error of the fitting population. Choosing $\alpha = 0 . 1$ therefore leaves about a tenth of the error mass in the pool and removes the rest by DFT. Because the bound is anchored to the fitting population rather than the test pool, this does not assert that the realized pool retains at most an α fraction of its own error; Section 3 reports both quantities. The expectation in Proposition 1 is over the exchangeable calibration and test identities, with the fitting stage held fixed. Because $\bar { \ell } _ { \mathrm { f i t } }$ is fixed by the fitting stage, the bound holds for each α chosen before calibration. It is not a high-probability bound for a single calibration draw, and it is not simultaneous over several values of α (Remark 2).

Remark 1 (Role of the predictor). Proposition 1 holds for any sˆ fixed given ${ \mathcal { D } } _ { \mathrm { f i t } }$ , regardless of the predictor’s quality. An uninformative predictor gives $\mathbb { E } [ \ell \mathbf { 1 } \dot { \{ s \leq \lambda \} } ] = \mathbb { E } [ \ell ] \mathbb { P } ( \hat { s } \leq \dot { \lambda } )$ , so the budget forces $\mathbb { P } ( \hat { s } \leq \lambda ) \lesssim \hat { \epsilon } _ { 0 } / \mathbb { E } [ \ell ]$ and nearly the entire pool is sent to DFT. Predictor quality therefore governs DFT cost, not the validity of risk control; the quantity we are interested in is how many DFT evaluations are saved under a fixed budget $\epsilon _ { 0 }$ . The guarantee also covers scores chosen before calibration from ${ \mathcal { D } } _ { \mathrm { f i t } }$ or prior knowledge, such as whether the user’s DFT matches the ${ \bf M L I P } \ ' _ { \bf S }$ training reference.

## 3 Experiments

## 3.1 Setup

We compare our score against four baselines that differ only in the predicted error score sˆ used in $\mathrm { ( 6 ) } \colon \mathrm { ( i ) }$ Random, where sˆ is uniform and sets the reference cost; (ii) Raw committee, where sˆ is the rank-combined disagreement among three pretrained MLIP architectures on predicted energy and forces, a standard uncertainty proxy for interatomic potentials [Smith et al., 2018, Schran et al., 2020, Liu et al., 2026]; (iii) Supervised committee, where sˆ is a learned combination of the same three-model disagreement signals, produced by a residual-regression network trained on $\mathcal { D } _ { \mathrm { f i t } } \mathrm { : }$ and (iv) Oracle, where $\hat { s } = s$ is the true error, a retrospective reference for the cost of perfect ranking rather than a proved lower bound.

All are wrapped in the identical CRC procedure of Section 2.4, so all satisfy the same risk guarantee (further details in Appendix D). Note that each committee requires three pretrained- ${ \bf \cdot M L I } { \bf \dot { P } }$ evaluations per configuration, whereas ours requires only one MLIP evaluation and the structural descriptor. We report these evaluation counts rather than wall-clock runtimes.

Datasets. We evaluate our method on three molecular datasets that contain DFT labels (Table 1) and three pretrained MLIP models: MACE-OFF23(M) [Kovács et al., 2025], AIMNet2 [Anstine et al., 2025], and AceFF-2/TensorNet2 [Farr et al., 2026, Simeon and De Fabritiis, 2023]. We do not evaluate MACE-OFF23(M) on MACE-OFF23-FULL, whose source it was trained on, leaving 8 dataset–MLIP combinations, or cells. Each MLIP is evaluated once per dataset; datasets and MLIPs use different electronic-structure references, so residuals include reference differences as well as model error (Appendix D).

Splits and calibration. We split each dataset by molecule identity into $\mathcal { D } _ { \mathrm { f i t } } , \mathcal { D } _ { \mathrm { c a l i b } } , \mathcal { D } _ { \mathrm { t e s t } }$ at $4 0 / 2 0 / 4 0$ , so that all configurations of a given identity remain in the same split. The error predictor of Section 2.3 is trained on ${ \mathcal { D } } _ { \mathrm { f i t } }$ and then frozen, along with energy alignment, size normalization, $B _ { \mathrm { c f g } } , \dot { B } ,$ and $\bar { \ell } _ { \mathrm { f i t } }$ . For each cell, we draw 20 random $\mathcal { D } _ { \mathrm { c a l i b } } , \mathcal { D } _ { \mathrm { t e s t } }$ resamples and sweep $\alpha \in \{ 0 . 0 5 , 0 . 1 0 , \hdots , 0 . 5 0 \}$ , the target fraction of error retained. Proposition 1 converts into the calibration budget $\epsilon _ { 0 } = \alpha \ell _ { \mathrm { f i t } }$

Fitting the predictor. Routing depends on the predictor only through the order it induces, so we ask it to rank well, not to predict accurately. The prediction $\hat { u } _ { t }$ in Eq. 3 is the average of three small multilayer perceptrons (seeds 0, 1, 2; two hidden layers of 256 units) trained independently on ${ \mathcal { D } } _ { \mathrm { f i t } }$ . Each network b minimizes an $\ell _ { 1 }$ residual loss on the three normalized channels plus a pairwise ranking loss on its own composite score $\begin{array} { r } { q _ { b } ( \boldsymbol { R } ) = \sum _ { t } w _ { t } \hat { u } _ { b , t } ( \boldsymbol { R } ) } \end{array}$ . The ranking targets are the truncated error scores $\ell _ { i } = \ell ( \bar { R _ { i } } )$ of Eq. (4) in Section $2 . 4 ,$ , the same quantity that CRC controls. For a mini-batch B with a uniformly permuted pairing π, let $\bar { \mathcal { P } } = \{ i \in \mathcal { B } : \ell _ { i } \neq \ell _ { \pi ( i ) } \}$

$$
\mathcal { L } _ { b } = \frac { 1 } { | \mathcal { B } | } \sum _ { i \in \mathcal { B } } \sum _ { t } w _ { t } \big | \hat { u } _ { b , t } ( R _ { i } ) - u _ { t } ( R _ { i } ) \big | + \frac { \lambda _ { \mathrm { r a n k } } } { | \mathcal { P } | } \sum _ { i \in \mathcal { P } } \log \Bigl [ 1 + e ^ { - \left( q _ { b } ( R _ { i } ) - q _ { b } ( R _ { \pi ( i ) } ) \right) \mathrm { s i g n } ( \ell _ { i } - \ell _ { \pi ( i ) } ) } \Bigr ] .\tag{11}
$$

The ranking loss term pushes configurations with larger true loss toward larger scores; the regression term keeps the three channels interpretable. Architecture, seeds, and the value of $\lambda _ { \mathrm { { r a n k } } }$ are chosen on a validation split internal to ${ \mathcal { D } } _ { \mathrm { f i t } }$ and frozen before calibration (Appendix I).

Metrics. For each $\alpha ,$ we report three quantities: (i) the DFT calls $k ( \alpha ) = | S _ { \hat { \lambda } ( \alpha ) } |$ , the number of configurations routed to DFT at that α, reported as a percentage of the test pool; (ii) the risk ratio $\begin{array} { r } { r ( \alpha ) = \sum _ { q \in \mathcal { D } _ { \mathrm { t e s t } } } L ( \hat { \lambda } ( \alpha ) ; g ) / \left( | \mathcal { D } _ { \mathrm { t e s t } } | \epsilon _ { 0 } \right) } \end{array}$ which measures how close the realized loss comes to the calibration budget and satisfies $\mathbb { E } [ r ( \alpha ) ] \le 1$ by Proposition 1; and (iii) the realized retained loss $\begin{array} { r } { \rho ( \alpha ) = \sum _ { R \in \mathcal D _ { \mathrm { t e s t } } \backslash S _ { \hat { \lambda } ( \alpha ) } } \ell ( R ) / \sum _ { R \in \mathcal D _ { \mathrm { t e s t } } } \ell ( R ) } \end{array}$ , the error left uncorrected relative to the pool’s total error rather than to $\epsilon _ { 0 }$

Since calls weight all configurations equally, we also report the $N ^ { 3 }$ -weighted DFT cost proxy $\begin{array} { l l } { { C _ { N ^ { 3 } } ( \alpha ) } } & { { = } } \end{array}$ $\begin{array} { r } { 1 0 0 \sum _ { R \in S _ { \hat { \Lambda } ( \alpha ) } } \breve { N ( } R ) ^ { 3 } / \sum _ { R \in \mathcal { D } _ { \mathrm { t e s t } } } N ( R ) ^ { 3 } } \end{array}$ (a size-scaling proxy, not wall time). We also measure the normalized gain,

$$
G = \frac { \mathrm { A U C } _ { \mathrm { r a n d } } - \mathrm { A U C } _ { \hat { s } } } { \mathrm { A U C } _ { \mathrm { r a n d } } - \mathrm { A U C } _ { \mathrm { o r a c l e } } } , \qquad \mathrm { A U C } _ { \hat { s } } = \frac { 1 } { \alpha _ { \mathrm { m a x } } - \alpha _ { \mathrm { m i n } } } \int k ( \alpha ) d \alpha\tag{12}
$$

This represents how much of the random-to-oracle headroom is captured by score sˆ. Note $G = 0$ and $G = 1$ correspond to random and oracle, respectively, and $G < 0$ means it is worse than random selection (anti-selective).

## 3.2 Results

Same guarantee, but lower DFT cost. Every score passes through the identical CRC rule, so Proposition 1 gives all the same guarantee. With validity fixed by design, models are compared using DFT cost at certified risk. In the top row of Fig. 2, over all cells/scores/targets, the mean risk ratio is at most 1.0359 and the largest single draw is 1.333. Recall that the guarantee is on the expectation, not on each draw. The finite-sample correction $B / ( n { + } 1 )$ ensures the metric is fair. Without it, the score violates the budget in roughly 52% of calibration draws across $n = 5 { - } 1 0 0$ identities. But with this correction, violations drop to 0.9% at $n = 1 0$ , trading safety for higher DFT cost when calibration data is scarce (Appendix E). Averaged uniformly over the eight cells (middle row), ours captures $G = 0 . 8 1$ of the random-to-oracle headroom, compared to 0.50 and 0.34 for the raw and supervised committees, respectively. At $\alpha = 0 . 3 0$ , it routes 57.8% of the pool, against 61.9%, 67.5%, and 71.6% for raw committee, supervised committee, and random, respectively. Our DFT call was comparable with oracle (54.4%). Under $N ^ { 3 }$ weighting (bottom row) and $\alpha \ge 0 . 1 5$ , ours is cheaper than the raw committee and random in all eight cells and than the supervised committee in seven of eight cells (Appendix F). The realized retained fraction $\rho ( \alpha )$ tracks α for every score (Fig. 5, Appendix E). At $\alpha = 0 . 3 0 \mathrm { . }$ , ours retains $\rho = 0 . 2 8 9$ , closest to the target, while the supervised committee (0.251) spends extra DFT on error the user would have tolerated.

![](images/c0383a8333d053d98a98b2797ef30733c3395ec7f25ded7f8eb5d17d14bb603a.jpg)  
Figure 2: Every score meets the same guarantee; the DFT cost differs. Three representative cells. Top row: risk ratio $r ( \alpha )$ against the bound $r = 1$ . Middle row: DFT calls $k ( \alpha )$ , with the random-to-oracle headroom shaded and each cell’s G boxed. Bottom row: $N ^ { 3 } .$ -weighted DFT cost (% of all-DFT), a size-scaling proxy, not wall time. Lines are means over 20 calibration/test reallocations, and bands are ±1 SD for ours and the committees. Dotted verticals mark $\alpha = 0 . 3 0$

Disagreement misses error that the MLIPs share. A residual mixes model error with any mismatch between the MLIP’s training reference and the dataset’s training reference. In such cases, the three MLIPs share common failure modes. Because they agree on the wrong answer, disagreement-based metrics collectively miss the correct answer. By contrast, our structure-only score captures this shared error from ${ \mathcal { D } } _ { \mathrm { f i t } }$ . Fig. 3(a) shows that on COMP6 (mismatch between DFT–MLIP training references), ours reaches $G = 0 . 8 2 – 0 . 8 9$ versus 0.14–0.27 for the committees. On QCML (slightly less of a mismatch), 0.75–0.84 versus 0.32–0.53. One exception is QCML × MACE-OFF23(M), where the raw committee wins (0.91), while the supervised one falls below random (−0.28). We also observed that a few catastrophic, model-specific failures carried nearly all the error (Gini 1.00, against ≤ 0.18 on COMP6). In comparison, on MACE-OFF23 cells (high alignment between DFT–MLIP training references) the supervised committee leads (0.76

Latent mean Supervised committee (3 MLIPs)

Ours (1 MLIP) Raw committee (3 MLIPs)

Supervised ANI committee

(a) The gap depends on the reference  
![](images/74f5aac00705e8908fa7e079a3852d955ce7af2d6ea646b673224092b170aa45.jpg)

(b) Less DFT at the same realized loss  
![](images/ccf145a4a5e04ebf1722f5115c0ba8f499e5d1e7751c274920e14d43c48d9592.jpg)

(c) Every score through the same CRC  
![](images/fe97d066d50dda2a1e0f986751402e1456969d1924715ddebe1fce43d581b89d.jpg)  
Figure 3: Structure beats disagreement where the MLIPs share error; the gap closes as references converge. (a) Per-cell G. Brackets give the relation between dataset and MLIP training references. Right of the dashed line, ANI-2x × COMP6 is a different MLIP and population, compared with its native and supervised ANI committees. The supervised committee on QCML × MACE-OFF23(M) (G = −0.28) is drawn clipped. (b) DFT saved by ours at matched realized loss ρ (committee minus ours, pp of pool), averaged over the common attained ρ range; the mean excludes ANI-2x. (c) Equal-cell mean G (small markers: cells) for every score under the same CRC rule; the right column gives MLIP evaluations per configuration and required model access.

and 0.82 versus 0.71 and 0.76). If disagreement misses shared error, committees slightly recover once the training references match. ANI-2x was trained using COMP6’s standard reference. There, its own committee reaches $G = 0 . 6 9$ and ours reaches 0.66 (whether ANI-2x saw these configurations during training can not be established). The shared error is also not just the standard dispersion correction. A score built from D3(BJ) alone reaches only $G \leq 0 . 1 2$ and adding D3(BJ) to the reference changes our G by at most 0.02 (Appendix G), meaning that our score is not just relearning the dispersion correction. Finally, the committees are not weak by design. Two-member committees average $G \leq 0 . 5 \check { 1 }$ , and committees built from the target’s own model family never beat ours in the five cells where they exist (Appendix H). At matched realized loss $\rho ,$ ours still saves 3.0 and 4.0 percentage points of DFT against the raw and supervised committees, positive in 6 of 8 cells each (Fig. 3(b), Appendix F). The reference relation is known before calibration, so the score can be chosen without touching the guarantee (Remark 1). If unsure whether the DFT matches the MLIPs’ training reference, the hybrid score (our descriptor plus the three disagreement channels) is always a safe choice. It is at least as good as ours in all eight cells, at the cost of running three MLIPs. This pattern comes from comparing nine settings that are not from a controlled experiment, supporting this explanation. In particular, MACE-OFF23 cells have both a closer reference and larger molecules, so they cannot say which of the two helps the committees.

Structure recovers what model internals know. In Fig. 3(c), ours $( G = 0 . 8 1 )$ is on par with the target MLIP’s own representation of 0.82, which needs model access and is fragile to pooling (max pooling: 0.72). It also far exceeds generic descriptors frequently used in chemistry like Atom-Centered Symmetry Functions (ACSF, 0.56) and the Smooth Overlap of Atomic Positions (SOAP, 0.48). Only the latent mean and the hybrid (0.83) are marginally higher, and among scores needing no model internals, ours is the strongest. CRC sees a score only through its order, and across all scores G rises with the test Spearman correlation between sˆ and s. Three checks show where this ranking comes from (Appendix I). First, no single feature family is significantly essential over another. Removing any one of them lowers G by at most 0.03. Second, more fitting data helps. G rises from 0.54 with 2.5% of identities to 0.81 with 40%. Third, the pairwise ranking loss is not needed: weights of 0, 0.1, and 1 give nearly the same G of 0.807, 0.806, and 0.808, respectively. As an extension beyond the guarantee, we were curious on what routing does to relative energies (energy differences between geometries of one molecule). $\mathrm { A t } \alpha = 0 . 3 0$ , ours cuts their MAE from the all-MLIP level by at least 36.4% in every cell, yet picking each molecule’s lowest-energy geometry succeeds less often than with no DFT on the three COMP6 cells. One molecule’s MLIP errors tend to share sign and cancel in differences, and replacing some with DFT removes that cancellation (Appendix J). This is a property of partial DFT substitution rather than of any particular score, so a guarantee on energy and force errors does not by itself cover the relative energies. Because Proposition 1 applies to any bounded loss that decreases as the threshold tightens (Remark 3), our framework can certify these objectives directly. The required DFT compute depends on the target: 79–97% for relative-energy errors, but only 8–61% for selection failure (Appendix K).

## 4 Conclusion

We cast MLIP–DFT routing as risk-controlled selection. A predictor ranks configurations by expected error, and conformal risk control decides where DFT is deployed. Under exchangeability of molecule identities, the expected capped error left to MLIP stays within a chosen fraction of the fitting-set mean for any score fixed before calibration. Ranking quality therefore changes only DFT cost, and scores can be compared on that cost alone. This comparison matters most when the user’s DFT reference differs from the MLIPs’ training reference. In this scenario, the MLIPs share much of their error and agree on the wrong answer. Averaged over eight dataset–MLIP cells, a structure-only score fitted on labeled residuals captures 0.81 of the random-to-oracle headroom, against 0.50 and 0.34 for three-MLIP committees. It also matches the target MLIP’s own latent features without accessing them. When the references do match, committees recover and demonstrate improved results. The choice of score should therefore follow the reference relation, which is known before calibration. Two results define where the method applies: (i) the fitting and calibration labels are amortized over repeated screening of a population, and (ii) partial DFT replacement can break error cancellation in relative energies, so these objectives should be certified directly, which the framework supports. We provide further limitations and broader impacts in Appendix M and N.

## AI Use Statement

In this work, we used generative AI tools to assist in the writing of proofs, provide feedback on research methodology and experiments, implement methods, and interpret results. We have not used generative AI tools to generate synthetic data, develop the theoretical framework, formulate the mathematical claims, propose hypotheses, design the experimenta methodology, translate, or perform qualitative data analysis. Additionally, we used generative AI tools to edit software code, draft parts of this paper, produce and refine figures, edit the manuscript for readability, and search for related literature. The research problem, the proposed approach, and all claims are the authors’ own: every proof was verified independently of the assistant that proposed it, all AI-assisted code was read and its outputs reproduced, and al references were checked against their sources. We take full responsibility for the final content of this work.

## Ethics Statement

This work involves no human subjects, no personally identifiable information, and no new data collection; all datasets and pretrained potentials are publicly released, cited, and used under their licenses (Appendix D). Our method reduces the number of density functional theory (DFT) calculations required at a controlled MLIP annotation risk, and makes downstream molecular screening more accessible to groups without large computing allocations. The main limitations are: Proposition 1 holds in expectation and only under Assumption 1, so a cutoff calibrated on one pool carries no guarantee on configurations from a different distribution, and it controls average risk rather than per-configuration correctness. The method may also be less reliable on datasets with extreme MLIP residuals, since the per-configuration loss is capped at $B _ { \mathrm { c f g } }$ . We state these limitations in Appendix M. We are aware of no other foreseeable harms, conflicts of interest, or fairness concerns.

## Reproducibility Statement

All experiments use public datasets and pretrained potentials, with sources, versions, and preprocessing documented in Appendix D. The structural descriptor is specified in full in Appendix B; the error predictor architecture, training objective, and hyperparameters appear in Section 2.3 and Appendix D; and the calibration procedure appears in Section 2.4, with its proof in Appendix C. The data splits, predictor seeds, and the 20 calibration/test reallocations are described in Appendix D. Code for the selector, the calibration procedure, and all scripts producing the reported tables and figures will be made publicly available.

## References

Anastasios N. Angelopoulos and Stephen Bates. Conformal prediction: A gentle introduction. Foundations and Trends in Machine Learning, 16(4):494–591, 2023. doi: 10.1561/2200000101.

Anastasios N. Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, and Tal Schuster. Conformal risk control. In The Twelfth International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/hash/ f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html.

Anastasios N. Angelopoulos, Stephen Bates, Emmanuel J. Candès, Michael I. Jordan, and Lihua Lei. Learn then test: Calibrating predictive algorithms to achieve risk control. The Annals ofApplied Statistics, 19(2):1641–1662, 2025. doi: 10.1214/24-AOAS1998.

Dylan M. Anstine, Roman Zubatyuk, and Olexandr Isayev. AIMNet2: a neural network potential to meet your neutral, charged, organic, and elemental-organic needs. Chemical Science, 16(23):10228–10244, 2025. doi: 10.1039/d4sc08572h.

Albert P. Bartók, Risi Kondor, and Gábor Csányi. On representing chemical environments. Physical Review B, 87(18): 184115, 2013. doi: 10.1103/PhysRevB.87.184115.

Ilyes Batatia, Dávid Péter Kovács, Gregor N. C. Simm, Christoph Ortner, and Gábor Csányi. MACE: Higher order equivariant message passing neural networks for fast and accurate force fields. In Advances in Neural Information Processing Systems, volume 35, pages 11423–11436, 2022. doi: 10. 52202/068431-0830. URL https://proceedings.neurips.cc/paper\_files/paper/2022/hash/ 4a36c3c51af11ed9f34615b81edb5bbc-Abstract-Conference.html.

Ilyes Batatia, Philipp Benner, Yuan Chiang, Alin M. Elena, Dávid P. Kovács, Janosh Riebesell, Xavier R. Advincula, Mark Asta, Matthew Avaylon, William J. Baldwin, Fabian Berger, Noam Bernstein, Arghya Bhowmik, Filippo Bigi, Samuel M. Blau, Vlad Carare, Michele Ceriotti, Sanggyu Chong, James P. Darby, Sandip De, Flaviano Della˘ Pia, Volker L. Deringer, Rokas Elijošius, Zakariya El-Machachi, Edvin Fako, Fabio Falcioni, Andrea C. Ferrari, John L. A. Gardner, Mikołaj J. Gawkowski, Annalena Genreith-Schriever, Janine George, Rhys E. A. Goodall, Jonas Grandel, Clare P. Grey, Petr Grigorev, Shuang Han, Will Handley, Hendrik H. Heenen, Kersti Hermansson, Cheuk Hin Ho, Stephan Hofmann, Christian Holm, Jad Jaafar, Konstantin S. Jakob, Hyunwook Jung, Venkat Kapil, Aaron D. Kaplan, Nima Karimitari, James R. Kermode, Panagiotis Kourtis, Namu Kroupa, Jolla Kullgren, Matthew C. Kuner, Domantas Kuryla, Guoda Liepuoniute, Chen Lin, Johannes T. Margraf, Ioan-Bogdan Magdau, Angelos˘ Michaelides, J. Harry Moore, Aakash A. Naik, Samuel P. Niblett, Sam Walton Norwood, Niamh O’Neill, Christoph Ortner, Kristin A. Persson, Karsten Reuter, Andrew S. Rosen, Louise A. M. Rosset, Lars L. Schaaf, Christoph Schran, Benjamin X. Shi, Eric Sivonxay, Tamás K. Stenczel, Christopher Sutton, Viktor Svahn, Thomas D. Swinburne, Jules Tilly, Cas van der Oord, Santiago Vargas, Eszter Varga-Umbrich, Tejs Vegge, Martin Vondrák, Yangshuai Wang, William C. Witt, Thomas Wolf, Fabian Zills, and Gábor Csányi. A foundation model for atomistic materials chemistry. The Journal ofChemical Physics, 163(18):184110, 2025. doi: 10.1063/5.0297006.

Stephen Bates, Anastasios N. Angelopoulos, Lihua Lei, Jitendra Malik, and Michael I. Jordan. Distribution-free, risk-controlling prediction sets. Journal ofthe ACM, 68(6):1–34, 2021. doi: 10.1145/3478535. Article 43.

Jörg Behler. Atom-centered symmetry functions for constructing high-dimensional neural network potentials. The Journal ofChemical Physics, 134(7):074106, 2011. doi: 10.1063/1.3553717.

Jörg Behler and Michele Parrinello. Generalized neural-network representation of high-dimensional potential-energy surfaces. Physical Review Letters, 98(14):146401, 2007. doi: 10.1103/PhysRevLett.98.146401.

Filippo Bigi, Sanggyu Chong, Michele Ceriotti, and Federico Grasselli. A prediction rigidity formalism for low-cost uncertainties in trained neural networks. Machine Learning: Science and Technology, 5(4):045018, 2024. doi: 10.1088/2632-2153/ad805f.

Jenna A. Bilbrey, Jesun S. Firoz, Mal-Soon Lee, and Sutanay Choudhury. Uncertainty quantification for neural network potential foundation models. npj Computational Materials, 11(1):109, 2025. doi: 10.1038/s41524-025-01572-y.

D. R. Bowler and T. Miyazaki. O(N) methods in electronic structure calculations. Reports on Progress in Physics, 75 (3):036503, 2012. doi: 10.1088/0034-4885/75/3/036503.

Beatriz Cordero, Verónica Gómez, Ana E. Platero-Prats, Marc Revés, Jorge Echeverría, Eduard Cremades, Flavia Barragán, and Santiago Alvarez. Covalent radii revisited. Dalton Transactions, (21):2832–2838, 2008. doi: 10.1039/B801115J.

Gábor Csányi, T. Albaret, M. C. Payne, and A. De Vita. “Learn on the fly”: A hybrid classical and quantum-mechanical molecular dynamics simulation. Physical Review Letters, 93(17):175503, 2004. doi: 10.1103/PhysRevLett.93. 175503.

Stefano Curtarolo, Gus L. W. Hart, Marco Buongiorno Nardelli, Natalio Mingo, Stefano Sanvito, and Ohad Levy. The high-throughput highway to computational materials design. Nature Materials, 12(3):191–201, 2013. doi: 10.1038/nmat3568.

Bowen Deng, Yunyeong Choi, Peichen Zhong, Janosh Riebesell, Shashwat Anand, Zhuohan Li, KyuJung Jun, Kristin A. Persson, and Gerbrand Ceder. Systematic softening in universal machine learning interatomic potentials. npj Computational Materials, 11(1):9, 2025. doi: 10.1038/s41524-024-01500-6.

Robin Dunn, Larry Wasserman, and Aaditya Ramdas. Distribution-free prediction sets for two-layer hierarchical models. Journal ofthe American Statistical Association, 118(544):2491–2502, 2023. doi: 10.1080/01621459.2022.2060112.

Peter Eastman, Pavan Kumar Behara, David L. Dotson, Raimondas Galvelis, John E. Herr, Josh T. Horton, Yuezhi Mao, John D. Chodera, Benjamin P. Pritchard, Yuanqing Wang, Gianni De Fabritiis, and Thomas E. Markland. SPICE, a dataset of drug-like molecules and peptides for training machine learning potentials. Scientific Data, 10(1):11, 2023. doi: 10.1038/s41597-022-01882-6.

Stephen E. Farr, Stefan Doerr, Antonio Mirarchi, Francesc Sabanes Zariquiey, and Gianni De Fabritiis. Aceff: A state-of the-art machine learning potential for small molecules, 2026. URL https://arxiv.org/abs/2601.00581.

Xiang Fu, Zhenghao Wu, Wujie Wang, Tian Xie, Sinan Keten, Rafael Gómez-Bombarelli, and Tommi S. Jaakkola. Forces are not enough: Benchmark and critical evaluation for machine learning force fields with molecular simulations. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview.net/ forum?id=A8pqQipwkt.

Stefan Ganscha, Oliver T. Unke, Daniel Ahlin, Hartmut Maennel, Sergii Kashubin, and Klaus-Robert Müller. The QCML dataset, quantum chemistry reference data from 33.5M DFT and 14.7B semi-empirical calculations. Scientific Data, 12:406, 2025. doi: 10.1038/s41597-025-04720-7.

Niklas Geue, Richard E. P. Winpenny, and Perdita E. Barran. Structural characterisation methods for supramolecular chemistry that go beyond crystallography. Chemical Society Reviews, 51:8–27, 2022. doi: 10.1039/D0CS01550D.

Yu Gui, Ying Jin, and Zhimei Ren. Conformal alignment: Knowing when to trust foundation models with guarantees. In Advances in Neural Information Processing Systems, volume 37, pages 73884–73919, 2024. doi: 10.52202/079017-2350. URL https://proceedings.neurips.cc/paper\_files/paper/2024/ hash/870ccde24673d3970a680bb48496ed63-Abstract-Conference.html.

Cheuk Hin Ho, Christoph Ortner, and YangShuai Wang. Flexible uncertainty calibration for machine-learned interatomic potentials. npj Computational Materials, 12(1):225, 2026. doi: 10.1038/s41524-026-02080-3.

Yuge Hu, Joseph Musielewicz, Zachary W. Ulissi, and Andrew J. Medford. Robust and scalable uncertainty estimation with conformal prediction for machine-learned interatomic potentials. Machine Learning: Science and Technology, 3 (4):045028, 2022. doi: 10.1088/2632-2153/aca7b1.

Shih-Peng Huang, Nontawat Charoenphakdee, Yuta Tsuboi, Yong-Bin Zhuang, and Wenwen Li. P-DRUM: Post-hoc descriptor-based residual uncertainty modeling for machine learning potentials, 2025. URL https://arxiv. org/abs/2509.02927. NeurIPS 2025 Workshop on Machine Learning and the Physical Sciences (non-archival).

Ying Jin and Emmanuel J. Candès. Selection by prediction with conformal p-values. Journal of Machine Learning Research, 24(244):1–41, 2023. URL http://jmlr.org/papers/v24/22-1176.html.

Ryosuke Jinnouchi, Ferenc Karsai, and Georg Kresse. On-the-fly machine learning force field generation: Application to melting points. Physical Review B, 100(1):014105, 2019. doi: 10.1103/PhysRevB.100.014105.

Matthias Kellner and Michele Ceriotti. Uncertainty quantification by direct propagation of shallow ensembles. Machine Learning: Science and Technology, 5(3):035006, 2024. doi: 10.1088/2632-2153/ad594a.

Walter Kohn and Lu Jeu Sham. Self-consistent equations including exchange and correlation effects. Physical Review, 140(4A):A1133–A1138, 1965. doi: 10.1103/PhysRev.140.A1133.

Dávid Péter Kovács, J. Harry Moore, Nicholas J. Browning, Ilyes Batatia, Joshua T. Horton, Yixuan Pu, Venkat Kapil, William C. Witt, Ioan-Bogdan Magdau, Daniel J. Cole, and Gábor Csányi. MACE-OFF: Short-range ˘ transferable machine learning force fields for organic molecules. Journal ofthe American Chemical Society, 147(21): 17598–17611, 2025. doi: 10.1021/jacs.4c07099.

Salem Lahlou, Moksh Jain, Hadi Nekoei, Victor I. Butoi, Paul Bertin, Jarrid Rector-Brooks, Maksym Korablyov, and Yoshua Bengio. DEUP: Direct epistemic uncertainty prediction. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview.net/forum?id=eGLdVRvvfQ.

Janice Lan, Aini Palizhati, Muhammed Shuaibi, Brandon M. Wood, Brook Wander, Abhishek Das, Matt Uyttendaele, C. Lawrence Zitnick, and Zachary W. Ulissi. AdsorbML: a leap in efficiency for adsorption energy calculations using generalizable machine learning potentials. npj Computational Materials, 9:172, 2023. doi: 10.1038/s41524-023-01121-5.

Zhenwei Li, James R. Kermode, and Alessandro De Vita. Molecular dynamics with on-the-fly machine learning of quantum-mechanical forces. Physical Review Letters, 114(9):096405, 2015. doi: 10.1103/PhysRevLett.114.096405.

Kai Liu, Zixiong Wei, Wei Gao, Poulumi Dey, Marcel H. F. Sluiter, and Fei Shuang. Heterogeneous ensemble enables a universal uncertainty metric for atomistic foundation models. npj Computational Materials, 12(1):34, 2026. doi: 10.1038/s41524-025-01905-x.

Shams Mehdi, Ilkwon Cho, and Olexandr Isayev. Knowing when to trust machine-learned interatomic potentials, 2026. URL https://arxiv.org/abs/2605.00640.

Felix Musil, Andrea Grisafi, Albert P. Bartók, Christoph Ortner, Gábor Csányi, and Michele Ceriotti. Physicsinspired structural representations for molecules and materials. Chemical Reviews, 121(16):9759–9815, 2021. doi: 10.1021/acs.chemrev.1c00021.

William Overman and Mohsen Bayati. Conformal arbitrage: Risk-controlled balancing of competing objectives in language models. In Advances in Neural Information Processing Systems, volume 38, pages 70305–70337, 2025. doi: 10.52202/085713-2364.

Danny Perez, Aparna P. A. Subramanyam, Ivan Maliyov, and Thomas D. Swinburne. Uncertainty quantification for misspecified machine learned interatomic potentials. npj Computational Materials, 11(1):263, 2025. doi: 10.1038/s41524-025-01758-4.

Andrew A. Peterson, Rune Christensen, and Alireza Khorshidi. Addressing uncertainty in atomistic machine learning. Physical Chemistry Chemical Physics, 19(18):10978–10985, 2017. doi: 10.1039/C7CP00375G.

Evgeny V. Podryabinkin and Alexander V. Shapeev. Active learning of linearly parametrized interatomic potentials. Computational Materials Science, 140:171–180, 2017. doi: 10.1016/j.commatsci.2017.08.031.

Raghunathan Ramakrishnan, Pavlo O. Dral, Matthias Rupp, and O. Anatole von Lilienfeld. Big data meets quantum chemistry approximations: The ∆-machine learning approach. Journal ofChemical Theory and Computation, 11(5): 2087–2096, 2015. doi: 10.1021/acs.jctc.5b00099.

Janosh Riebesell, Rhys E. A. Goodall, Philipp Benner, Yuan Chiang, Bowen Deng, Gerbrand Ceder, Mark Asta, Alpha A. Lee, Anubhav Jain, and Kristin A. Persson. A framework to evaluate machine learning crystal stability predictions. Nature Machine Intelligence, 7(6):836–847, 2025. doi: 10.1038/s42256-025-01055-1.

Matthias Rupp, Alexandre Tkatchenko, Klaus-Robert Müller, and O. Anatole von Lilienfeld. Fast and accurate modeling of molecular atomization energies with machine learning. Physical Review Letters, 108(5):058301, 2012. doi: 10.1103/PhysRevLett.108.058301.

Christoph Schran, Krystof Brezina, and Ondrej Marsalek. Committee neural network potentials control generalization errors and enable active learning. The Journal ofChemical Physics, 153(10):104105, 2020. doi: 10.1063/5.0016004.

Guillem Simeon and Gianni De Fabritiis. Tensornet: cartesian tensor representations for efficient learning of molecular potentials. In Proceedings ofthe 37th International Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA, 2023. Curran Associates Inc.

Justin S. Smith, Ben Nebgen, Nicholas Lubbers, Olexandr Isayev, and Adrian E. Roitberg. Less is more: Sampling chemical space with active learning. The Journal of Chemical Physics, 148(24):241733, 2018. doi: 10.1063/1. 5023802.

Eli Stavitski and Bert M. Weckhuysen. Infrared and raman imaging of heterogeneous catalysts. Chemical Society Reviews, 39:4615–4625, 2010. doi: 10.1039/C0CS00064G.

Oliver T. Unke, Stefan Chmiela, Huziel E. Sauceda, Michael Gastegger, Igor Poltavsky, Kristof T. Schütt, Alexandre Tkatchenko, and Klaus-Robert Müller. Machine learning force fields. Chemical Reviews, 121(16):10142–10186, 2021. doi: 10.1021/acs.chemrev.0c01111.

Jonathan Vandermause, Steven B. Torrisi, Simon Batzner, Yu Xie, Lixin Sun, Alexie M. Kolpak, and Boris Kozinsky. On-the-fly active learning of interpretable bayesian force fields for atomistic rare events. npj Computational Materials, 6:20, 2020. doi: 10.1038/s41524-020-0283-z.

Joshua A. Vita, Amit Samanta, Fei Zhou, and Vincenzo Lordi. LTAU-FF: Loss trajectory analysis for uncertainty in atomistic force fields. Machine Learning: Science and Technology, 6(1):015048, 2025. doi: 10.1088/2632-2153/ adb4b9.

Vladimir Vovk, Alexander Gammerman, and Glenn Shafer. Algorithmic Learning in a Random World. Springer, New York, NY, 2005. doi: 10.1007/b106715.

Brandon M. Wood, Misko Dzamba, Xiang Fu, Meng Gao, Muhammed Shuaibi, Luis Barroso-Luque, Kareem Abdelmaqsoud, Vahe Gharakhanyan, John Kitchin, Daniel S. Levine, Kyle Michel, Anuroop Sriram, Taco S. Cohen, Abhishek Das, Sushree Sahoo, Ammar Rizvi, Zachary Ulissi, and Larry Zitnick. UMA: A family of universal models for atoms. In Advances in Neural Information Processing Systems, volume 38, 2025. doi: 10.52202/085713-4310. URL https://proceedings.neurips.cc/paper\_files/paper/2025/ hash/bbf23e81b0ad7637fe9a731d0b676ca6-Abstract-Conference.html.

Zixuan Wu, So Won Jeong, Yating Liu, Yeo Jin Jung, and Claire Donnat. Filtering with confidence: When data augmentation meets conformal prediction, 2025. URL https://arxiv.org/abs/2509.21479.

Han Xu, Taoyong Cui, Chenyu Tang, Jinzhe Ma, Dongzhan Zhou, Yuqiang Li, Xiang Gao, Xingao Gong, Wanli Ouyang, Shufei Zhang, and Mao Su. Evidential deep learning for interatomic potentials. Nature Communications, 17:937, 2026. doi: 10.1038/s41467-025-67663-y.

Yuzhi Zhang, Haidi Wang, Weijie Chen, Jinzhe Zeng, Linfeng Zhang, Han Wang, and Weinan E. DP-GEN: A concurrent learning platform for the generation of reliable deep learning based potential energy models. Computer Physics Communications, 253:107206, 2020. doi: 10.1016/j.cpc.2020.107206.

## A Related Works

Configuration-level uncertainty for pretrained MLIPs. Pretrained MLIPs are trained on large and chemically diverse data [Kovács et al., 2025, Batatia et al., 2025, Wood et al., 2025], yet they still fail on individual geometries and retain systematic errors at scale [Fu et al., 2023, Deng et al., 2025]. Configuration-level uncertainty is commonly estimated from committee disagreement [Peterson et al., 2017, Schran et al., 2020], Bayesian or evidential models [Vandermause et al., 2020, Xu et al., 2026], or distances in a learned latent space [Hu et al., 2022]. For large pretrained models, cheaper estimators avoid training full ensembles [Bilbrey et al., 2025, Bigi et al., 2024, Kellner and Ceriotti, 2024]. Liu et al. [2026] combine a heterogeneous ensemble of universal potentials into one uncertainty metric and use an empirically chosen cutoff on it to decide which structures need DFT labels. Such estimates are often overconfident and must be calibrated against labeled errors before use [Perez et al., 2025]. Raw and supervised committees represent this lineage as baselines. In our framework, however, the cutoff on any score is calibrated by a CRC framework rather than chosen empirically.

Post-hoc error and reliability prediction for trained MLIPs. There are several methods that share the idea of training a predictor that estimates the error of a pretrained model. DEUP trains a secondary predictor that estimates out of-sample error [Lahlou et al., 2023], whereas ∆-learning learns a signed correction to a cheap baseline [Ramakrishnan et al., 2015]. For MLIPs, recent post-hoc methods draw on the target model’s latent representation or training history [Vita et al., 2025, Huang et al., 2025, Mehdi et al., 2026]. P-DRUM is closest in target because it regresses energy and force residuals from the descriptors of a trained MACE model [Huang et al., 2025]. PROBE instead recasts energy reliability as selective classification on frozen per-atom representations [Mehdi et al., 2026]. Our predictor uses a structure-only descriptor whose construction does not depend on the target MLIP (Section 2.2), and a separate predictor is fitted for each dataset–MLIP cell. It needs no model internals but forgoes the model-specific information that a latent representation may carry. Only its ordering enters the routing decision, and the threshold is calibrated separately (Section 2.4).

Uncertainty-guided DFT calls, acquisition, and screening. Learn-on-the-fly molecular dynamics has long coupled quantum-mechanical calculations with a force model that is refitted or extended during the simulation [Csányi et al., 2004, Li et al., 2015]. Modern on-the-fly schemes call DFT when a Bayesian error estimate is large and add the result to the training set [Jinnouchi et al., 2019, Vandermause et al., 2020]. The DFT result can also replace the unreliable prediction at that step [Jinnouchi et al., 2019]. Active-learning pipelines similarly select configurations by committee disagreement or extrapolation grade and retrain on the new labels [Podryabinkin and Shapeev, 2017, Smith et al., 2018, Zhang et al., 2020]. Using the DFT value of a flagged configuration is therefore not specific to our setting. What differs is that our target MLIP is never retrained and each deployment pool is routed in a single pass. Our cutoff also carries a finite-sample guarantee on the expected capped retained loss, whereas these cutoffs are set by rule or tuned empirically. Screen-then-verify workflows use machine-learned models to pre-filter candidates for DFT, as in AdsorbML and the Matbench Discovery benchmark [Lan et al., 2023, Riebesell et al., 2025]. They select on predicted energy or stability rather than on predicted error. Our selection-success metric is analogous to the AdsorbML success rate (Appendix J).

Table 2: Feature families of the structural vector. The canonical Coordination and PairState blocks contain 60 and 165 coordinates; a fixed compaction removes eight and nine of them, leaving 52 and 156. The descriptor contains the nine families and has 350 dimensions.
<table><tr><td>Feature family</td><td>Dim.</td><td>Type</td><td>Description</td></tr><tr><td>ElementCount</td><td>10</td><td>Discrete</td><td>Global counts of the allowed elements.</td></tr><tr><td>Coordination</td><td>52</td><td>Discrete</td><td>Element-resolved proximity-graph coordination states.</td></tr><tr><td>PairState</td><td>156</td><td>Discrete</td><td>Element-pair contacts classified as compressed, near nominal separation, or elongated.</td></tr><tr><td>ShortRange</td><td>21</td><td>Continuous</td><td>Atomic-number-weighted severity of short interatomic contacts.</td></tr><tr><td>Extreme</td><td>12</td><td>Continuous</td><td>Most severe local proximity, strain, and angular-distortion environments.</td></tr><tr><td>RBF</td><td>18</td><td>Continuous</td><td>Smooth multiscale pair-distance channels with global and local pooling.</td></tr><tr><td>InverseDistance</td><td>36</td><td>Continuous</td><td>Element-independent geometric proximity across several inverse-distance scales.</td></tr><tr><td>SoftCrowding</td><td>24</td><td>Continuous</td><td>Smooth filling of local coordination shells over several radii.</td></tr><tr><td>ZPotential</td><td>21</td><td>Continuous</td><td>Neighbor-Z-weighted description of the local chemical environment.</td></tr></table>

Conformal calibration and risk control. Conformal prediction calibrates prediction sets or intervals to finite-sample coverage under exchangeability [Vovk et al., 2005, Angelopoulos and Bates, 2023]. Risk-controlling prediction sets and Learn-then-Test control more general risks with high probability [Bates et al., 2021, Angelopoulos et al., 2025], while CRC controls the expected value of a bounded monotone loss [Angelopoulos et al., 2024]. Related calibration rules select trustworthy foundation-model outputs, filter augmented data, and select candidates with conformal p-values [Gui et al., 2024, Wu et al., 2025, Jin and Candès, 2023]. For MLIPs, Hu et al. [2022] calibrate energy intervals with split conformal prediction, and Ho et al. [2026] learn a conformal-inspired scaling of per-atom force uncertainty without a formal coverage guarantee. Both calibrate the uncertainty of individual predictions, whereas our calibration sets a DFT-routing cutoff to control loss of an identity. For our guarantee to hold, we only need exchangeability across molecule identities because each identity contributes to a single bounded loss.

## B The 350-dimensional structural descriptor

The descriptor is a deterministic function of atomic numbers $Z _ { j }$ and Cartesian coordinates $\mathbf { x } _ { j }$ . It uses no reference labels, MLIP predictions, learned embeddings, or fitted parameters. Its nine feature families summarize composition, coordination, distance distortion, close contacts, and local environments (Table 2). These are geometric proxies motivated by chemical structure; they do not compute electronic structure.

## B.1 Geometric quantities and summary conventions

The fixed element order is H, C, N, O, F, P, S, Cl, Br, I, with atomic numbers [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]. Covalent radii $\rho ( Z )$ in Å are [0.31, 0.76, 0.71, 0.66, 0.57, 1.07, 1.05, 1.02, 1.20, 1.39] [Cordero et al., 2008]; carbon uses the $s p ^ { 3 }$ value. Define

$$
r _ { j k } = \| { \bf x } _ { j } - { \bf x } _ { k } \| _ { 2 } , \qquad q _ { j k } = \frac { r _ { j k } } { \rho ( Z _ { j } ) + \rho ( Z _ { k } ) } , \qquad { \widetilde r } _ { j k } = \operatorname* { m a x } ( r _ { j k } , 0 . 3 5 \mathring { \mathrm { A } } ) .
$$

The graph $G _ { 0 }$ contains all pairs with $q _ { j k } \leq 1 . 2 0$ . A second graph $G$ uses the same candidate edges but restricts H and F to at most one connection, retaining the lowest $( q _ { j k } , e )$ where e is the edge identifier used to break distance ties. Coordination and pair-state counts use $G _ { 0 } ;$ local strain and angular summaries use G.

For an atomwise or pairwise vector $x ,$ define the ordered summaries

$$
\begin{array} { r l } & { S _ { 7 } ( x ) = [ \mathrm { s u m } , \mathrm { m e a n } , \mathrm { s t d } _ { \mathrm { p o p } } , \mathrm { m a x } , Q _ { 0 . 9 0 } , \mathrm { r m s } , \mathrm { c o u n t } ] , } \\ & { \mathcal { U } _ { 6 } ( x ) = [ \mathrm { s u m } , \mathrm { m e a n } , \mathrm { s t d } _ { \mathrm { p o p } } , \mathrm { m a x } , Q _ { 0 . 9 0 } , \mathrm { r m s } ] . } \end{array}
$$

The operators top1, top2, and top3 sort values in descending order and pad with zeros if fewer than three values exist. On empty edge or pair sets, all summaries and fractions return zero. These conventions fix the dimension for small structures.

## B.2 Composition, coordination, and pair states: 218 coordinates

ElementCount records ten element counts. For each element, Coordination counts atoms with $G _ { 0 }$ degrees $0 , 1 , 2 , 3 , 4 , \geq$ 5, giving 60 canonical coordinates. For each of the 55 unordered element pairs, PairState counts $G _ { 0 }$ edges in three regions: $q < 0 . 9 5 , 0 . 9 5 \leq q \leq 1 . 0 5$ , and $1 . 0 5 < q \le 1 . 2 0$ , giving 165 canonical coordinates. These bins describe distances relative to covalent-radius sums, without assigning bond order or bond energy.

Concatenate the $1 0 + 6 0 + 1 6 5 = 2 3 5$ canonical coordinates and remove the fixed zero-based indices

$$
[ 1 5 , 1 6 , 2 2 , 3 3 , 3 8 , 3 9 , 4 0 , 4 6 , 1 3 3 , 1 5 4 , 1 7 2 , 1 7 3 , 1 7 4 , 1 8 1 , 1 8 2 , 1 8 3 , 1 8 4 ] .
$$

Preserve the order of all remaining coordinates. Eight exclusions belong to Coordination and nine to PairState, leaving $1 0 + 5 2 + 1 5 6 = 2 1 8$ inputs.

## B.3 Short-range and extreme local environments: 33 coordinates

For powers $p = 2 , 4 , 6 ,$ , define

$$
S _ { j } ^ { ( p ) } = \sum _ { k \neq j } \frac { Z _ { j } Z _ { k } } { \widetilde { r } _ { j k } ^ { p } } .
$$

Append $S _ { 7 } ( S ^ { ( p ) } )$ for each power, giving 21 ShortRange coordinates; the count entry equals N. Higher powers emphasize closer contacts. Extreme uses three atomwise quantities:

$$
R _ { j } = S _ { j } ^ { ( 2 ) } , \qquad W _ { j } = \sum _ { { e \in G } \atop { e \ni j } } \exp \left[ - { \frac { ( q _ { e } - 1 ) ^ { 2 } } { 2 ( 0 . 2 0 ) ^ { 2 } } } \right] ( q _ { e } - 1 ) ^ { 2 } ,
$$

$$
B _ { j } = \sum _ { \stackrel { k < \ell } { k , \ell \in \mathcal { N } _ { G } ( j ) } } | \sin ( \theta _ { k j \ell } - \theta _ { 0 } ) | , \qquad \theta _ { 0 } = \left\{ \begin{array} { l l } { \pi , } & { \mathrm { d e g } _ { G } ( j ) = 2 , } \\ { 2 \pi / 3 , } & { \mathrm { d e g } _ { G } ( j ) = 3 , } \\ { \mathrm { a r c c o s } ( - 1 / 3 ) , } & { \mathrm { d e g } _ { G } ( j ) \geq 4 . } \end{array} \right.
$$

Set $B _ { j } = 0$ below degree two. For each of R, W, B, append $[ \mathrm { t o p 1 } , \mathrm { t o p 2 } , \mathrm { t o p 3 } , \mathrm { s u m } ]$ , giving 12 coordinates. The angular reference is degree-based, not an orbital hybridization assignment. Its non-monotone sine penalty can vanish for a collapsed degree-two angle; the distance features still record close contacts.

## B.4 Radial and inverse-distance summaries: 54 coordinates

For nine ordered centers

$$
( \mu _ { 1 } , \ldots , \mu _ { 9 } ) = ( 0 . 5 6 8 , 0 . 7 2 8 , 0 . 8 8 8 , 1 . 0 4 8 , 1 . 2 0 8 , 1 . 3 6 8 , 1 . 5 2 8 , 1 . 6 8 8 , 1 . 8 4 8 ) ,
$$

define

$$
g _ { \mu } ( q ) = \exp \bigl [ - \frac { 1 } { 2 } ( ( q - \mu ) / 0 . 1 2 4 ) ^ { 2 } \bigr ] , \quad P _ { \mu } = \sum _ { j < k } g _ { \mu } ( q _ { j k } ) , \quad M _ { \mu } = \operatorname * { m a x } _ { j } \sum _ { k \neq j } g _ { \mu } ( q _ { j k } ) .
$$

The RBF block is $[ P _ { \mu _ { 1 } } , M _ { \mu _ { 1 } } , . . . , P _ { \mu _ { 9 } } , M _ { \mu _ { 9 } } ]$ . For each $p = 1 , \ldots , 6$ , InverseDistance appends $\mathcal { U } _ { 6 } ( \{ \widetilde { r } _ { i k } ^ { - p } : j < k \} )$ giving 36 coordinates. These unweighted proximity summaries distinguish distance effects from the atomic-number weights in ShortRange.

## B.5 Soft crowding and neighboring elements: 45 coordinates

For the ordered scale pairs (q<sub>0</sub>, w) = (0.85, 0.12), (1.00, 0.16), (1.20, 0.22), (1.50, 0.30), define

$$
C _ { j } ^ { ( q _ { 0 } , w ) } = \sum _ { k \neq j } \frac { 1 } { 1 + \exp [ ( q _ { j k } - q _ { 0 } ) / w ] } .
$$

At each scale append [mean $C , { \mathrm { s t d } } _ { \mathrm { p o p } } C ,$ , min C, max $C , Q _ { 0 . 9 0 } ( C ) ]$ . Then append, in order,

$$
\begin{array} { r l } & { \mathrm { m e a n } ( C ^ { 0 . 8 5 , 0 . 1 2 } - C ^ { 1 . 0 0 , 0 . 1 6 } ) , \quad \mathrm { m e a n } ( C ^ { 1 . 0 0 , 0 . 1 6 } - C ^ { 1 . 2 0 , 0 . 2 2 } ) , } \\ & { \mathrm { m a x } ( C ^ { 0 . 8 5 , 0 . 1 2 } - C ^ { 1 . 2 0 , 0 . 2 2 } ) , \quad \mathrm { s t d } _ { \mathrm { p o p } } ( C ^ { 0 . 8 5 , 0 . 1 2 } - C ^ { 1 . 5 0 , 0 . 3 0 } ) . } \end{array}
$$

These 24 SoftCrowding coordinates use heuristic distance scales, not validated physical coordination-shell boundaries. Finally, for $p = 2 , 3 , 4 .$ , define $\begin{array} { r } { L _ { j } ^ { ( p ) } = \sum _ { k \neq j } Z _ { k } / \widetilde { r } _ { j k } ^ { p } } \end{array}$ and append $S _ { 7 } ( L ^ { ( p ) } )$ . The resulting 21 ZPotential coordinates retain neighbor element identity without multiplying by the central atomic number. The complete dimension is $2 1 8 + 2 1 { \overset { \cdot } { + } } 1 2 + 1 8 + 3 6 + 2 4 { \overset { \cdot } { + } } 2 1 = 3 5 0$

## C Proof of Proposition 1

ProofofProposition 1. Fix $\mathcal { D } _ { \mathrm { f i t } } , \hat { s }$ , target normalization, weights $w _ { t } ,$ , bounding constants $B _ { \mathrm { c f g } }$ and $B ,$ and $\epsilon _ { \mathrm { 0 } } .$ For each $\lambda ,$ the map $g \mapsto L ( \lambda ; g )$ is a fixed measurable function of a molecule and Assumption 1 makes $\{ L ( \lambda ; g _ { i } ) \} _ { i \in { \mathcal { D } } _ { \operatorname { c a l i b } } } \cup$ $\{ L ( \lambda ; g _ { n + 1 } { \bar { ) } } \}$ exchangeable. By construction, $L ( \lambda ; g _ { i } )$ is a non-decreasing function of λ, $0 \leq \mathsf { \bar { L } } ( \lambda ; g _ { i } ) \leq \bar { B } ,$ and $L ( - \infty ; g _ { i } ) = 0$

Let

$$
\mathcal { H } _ { n + 1 } = \mathcal { H } _ { n } \cup \{ \hat { s } ( R ) : R \in \mathcal { R } _ { g _ { n + 1 } } \}
$$

where ${ \mathcal { H } } _ { n }$ follows Eq. 7 and

$$
\begin{array} { r } { \lambda ^ { \star } = \operatorname* { m a x } \Bigl \{ \lambda \in \mathcal { H } _ { n + 1 } : \frac { 1 } { n + 1 } \sum _ { i = 1 } ^ { n + 1 } L ( \lambda ; g _ { i } ) \leq \epsilon _ { 0 } \Bigr \} , } \end{array}
$$

which denotes λ in the oracle case when $L ( \lambda ; g _ { n + 1 } )$ is known. The set is non-empty since the condition always holds for $\lambda = - \infty$ so $\lambda ^ { * }$ is well-defined.

For any $\lambda \in \mathcal { F } _ { n }$ we have $\lambda \in \mathcal { H } _ { n + 1 }$ and

$$
\frac { 1 } { n + 1 } \sum _ { i = 1 } ^ { n + 1 } L _ { i } ( \lambda ) \leq \frac { \sum _ { i = 1 } ^ { n } L _ { i } ( \lambda ) + B } { n + 1 } \leq \epsilon _ { 0 } ,
$$

so $\lambda \leq \lambda ^ { \star }$ ; taking the maximum, $\hat { \lambda } \le \lambda ^ { \star }$ for $\hat { \lambda }$ defined in Eq. 9 (trivially so if ${ \mathcal { F } } _ { n } = \emptyset$

Both $\mathcal { H } _ { n + 1 }$ and $\textstyle \sum _ { i = 1 } ^ { n + 1 } L _ { i }$ are invariant under permutations of $g _ { 1 } , \ldots , g _ { n + 1 }$ , so $\lambda ^ { \star }$ is a symmetric function of them. Assumption 1 then makes $\mathbb { E } [ L ( \lambda ^ { \star } ; g _ { i } ) ]$ ] constant in i, whence

$$
\begin{array} { r } { \mathbb { E } \big [ L ( \lambda ^ { \star } , g _ { n + 1 } ) \big ] = \mathbb { E } \Big [ \frac { 1 } { n + 1 } \sum _ { i = 1 } ^ { n + 1 } L ( \lambda ^ { \star } , g _ { i } ) \Big ] \leq \epsilon _ { 0 } . } \end{array}
$$

Since $\hat { \lambda } \le \lambda ^ { \star }$ and $L _ { n + 1 }$ is non-decreasing, $L ( \hat { \lambda } ; g _ { n + 1 } ) \leq L ( \lambda ^ { \star } ; g _ { n + 1 } )$ point-wise, giving $\mathbb { E } [ L ( \hat { \lambda } ; g _ { n + 1 } ) ] \le \epsilon _ { 0 }$ . Each $g \in \mathcal { D } _ { \mathrm { t e s t } }$ is exchangeable with $\mathcal { D } _ { \mathrm { c a l i b } } ,$ , so this bound applies and linearity of expectation yields $\begin{array} { r } { \mathbb { E } [ \sum _ { g \in \mathcal { D } _ { \mathrm { t e s t } } } L ( \hat { \lambda } ; g ) ] \le } \end{array}$ $m \epsilon _ { 0 } .$ , with no independence required among test molecules. □

Remark 2 (Ties, fallback, and scope). The argument is finite and needs neither continuity of sˆ nor absence of ties. When no finite candidate is feasible but $\epsilon _ { 0 } \geq \bar { B ^ { \prime } } ( n + 1 ) , \lambda = - \infty$ still satisfies the criterion; when $\epsilon _ { 0 } < B / ( n + 1 )$ the criterion is infeasible, and the rule in Eq. 9 still sends every configuration to DFT. Our method controls expected capped identity-level retained loss for an α fixed before calibration and it is worth noting that it does not guarantee $\rho ( \alpha ) \leq \alpha$ for a particular pool and does not bound uncapped physical energy or force error.

Remark 3 (Other bounded losses). The proof only requires that $L ( \lambda ; g )$ is non-decreasing in $\lambda ,$ bounded by $B ,$ zero at $\lambda = - \infty$ , and fixed given ${ \mathcal { D } } _ { \mathrm { f i t } }$ . Any per-configuration loss $\ell ^ { \prime }$ with these properties can replace ℓ, and any aggregation over molecules that preserves monotonicity and the bound can replace the capped sum.

## D Datasets, experimental protocol, and baselines

We first specify what is fitted, what is frozen, and what calibration is allowed to use. All primary scores are calibrated against the same loss, with the same splits and target grid. Table 1 summarizes the three evaluation datasets.

Splits and resampling. Each dataset is split by its stored molecule identity into $\mathcal { D } _ { \mathrm { f i t } } , \mathcal { D } _ { \mathrm { c a l i b } } , \mathcal { D } _ { \mathrm { t e s t } }$ at $4 0 / 2 0 / 4 0$ . Every fitting-derived quantity and the score predictor are estimated on ${ \mathcal { D } } _ { \mathrm { f i t } }$ once and frozen. Calibration and test identities are reallocated in 20 random resamples per combination, with $\alpha \in \{ 0 . 0 5 , 0 . 1 0 , \ldots , 0 . 5 0 \}$ . On COMP6, each resample calibrates on 300 identities (7,200 configurations) and evaluates on 600 (14,400); on QCML, it calibrates on 1,390 (12,510) and evaluates on 2,780 (25,020). On MACE-OFF23, 6,308 fitting groups contain 19,654 configurations with 1–229 configurations per group; calibration and test sets contain 3,154 and 6,307 groups, with test-pool size $2 0 , 3 7 2 . 8 \pm 3 3 3 . 5$ configurations across resamples. The MACE-OFF23 population has 15,769 groups: 15,541 known parent identities and 228 configurations with unavailable parent identity, each kept as a singleton group. These singletons are not known to be distinct molecules; the exchangeability assumption is stated for the groups as executed. Total atom counts are 612,360 for COMP6, 1,551,806 for MACE-OFF23, and 687,204 for QCML.

Predictor fitting. Each of the three scorers has hidden widths (256, 256), SiLU activations, dropout 0.1, and three Softplus outputs. Training minimizes (11) with AdamW, learning rate $1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 4 }$ , batch size 256, and at most 300 epochs. Checkpoints use the residual validation $L _ { 1 }$ loss with patience 25 and minimum improvement $1 0 ^ { - 5 }$ the ranking loss is not used for early stopping. Feature centering, scaling, and removal of constant coordinates use only the internal training rows, while the residual normalization and loss transforms use the full ${ \mathcal { D } } _ { \mathrm { f i t } }$ . Scorers are trained independently with seeds 0, 1, 2. The arithmetic mean of the three scores is used directly. The same frozen checkpoints serve all 20 calibration/test allocations, and there is no refit after selection.

Fitting set-derived quantities. The following are estimated on ${ \mathcal { D } } _ { \mathrm { f i t } }$ and then frozen at calibration: the per-element energy offsets aˆ that place the MLIP and DFT energies on a common baseline; the size exponents $\hat { \beta } _ { t }$ and normalizing factors $\hat { c } _ { t }$ for the three channels $t \in \{ E , \mathrm { a v g } , \mathrm { m a x } \}$ , which together define the error score s; the truncation level $B _ { \mathrm { c f g } } ,$ , the $9 9 . 9 \%$ quantile of s over ${ \mathcal { D } } _ { \mathrm { f i t } }$ , gives the loss $\ell = \dot { \operatorname* { m i n } } ( s , B _ { \mathrm { c f g } } )$ of Eq. (4). Because molecule identity is the exchangeable unit, the remaining two quantities are calculated over unique identities in $\mathcal { D } _ { \mathrm { f i t } } \mathrm { : }$ : the bound $B \doteq \operatorname* { m a x } _ { g \in \mathcal { T } _ { \mathrm { f i t } } } \ \sum _ { i \in g } \ell _ { i }$ used in Eq. (8), and the mean per-identity loss $\begin{array} { r } { \bar { \ell } _ { \mathrm { f i t } } = | \mathcal { I } _ { \mathrm { f i t } } | ^ { - 1 } \sum _ { g \in \mathcal { T } _ { \mathrm { f i t } } } \sum _ { i \in g } \ell _ { i } } \end{array}$ setting the budget $\epsilon _ { 0 } = \alpha \bar { \ell } _ { \mathrm { f i t } }$ . Here, $\mathcal { T } _ { \mathrm { f i t } }$ denotes the set of molecule identities of ${ \mathcal { D } } _ { \mathrm { f i t } }$ . Beyond these, the committees fit inter-model energy offsets to align the scales of different model predictions, and our model trains the score predictor sˆ, both from ${ \mathcal { D } } _ { \mathrm { f i t } }$ . Random estimates nothing—sˆ is a uniform draw—and oracle estimates nothing further, since $\hat { s } = s . \ D _ { \mathrm { c a l i b } }$ is spent entirely on choosing the threshold $\hat { \lambda }$ via Eq. (9).

Per-element energy alignment. A frozen public MLIP checkpoint and a reference DFT dataset do not share an isolated-atom energy convention. Their total energies therefore differ by a fixed amount per atom of each element, and that constant is far larger than the model error we want to measure. The offset of a molecule is the sum of the per-element constants over its atoms, so it is exactly linear in the element counts:

$$
e _ { \mathrm { t r u e } } ( R ) - e _ { \mathrm { p r e d } } ( R ) = \phi ( R ) ^ { \top } a + \varepsilon ( R ) ,\tag{13}
$$

where $\phi ( R ) \in \mathbb { R } ^ { K + 1 }$ holds the counts of the elements together with an intercept, and a collects the per-element offsets $a _ { z }$ and that intercept. Regressing the energy difference on ϕ over ${ \mathcal { D } } _ { \mathrm { f i t } }$ recovers ${ \hat { a } } ,$ and $\Delta E ( R ) = e _ { \mathrm { t r u e } } ( R ) - e _ { \mathrm { p r e d } } ( R ) -$ $\phi ( R ) ^ { \top } \hat { a }$ is the aligned residual that enters the energy channel of s. Because the correction is linear in composition alone, it removes the convention exactly and cannot absorb any geometry-dependent error; it can, however, absorb a composition-linear part of systematic model or reference-level bias, so it does not equate different electronic-structure methods. We fit aˆ by iteratively trimmed least squares rather than plain least squares. A handful of MLIP blow-ups, reaching $| \Delta E | \sim 1 0 ^ { 8 } \mathrm { e V }$ on QCML, would otherwise dominate the normal equations, and the offset would not be recovered. Writing Trim $\mathrm { { C } } _ { I _ { \mathrm { f i t } } } ( y )$ for this operator with $y _ { i } = y ( R _ { i } )$ and $\phi _ { i } = \phi ( R _ { i } ) ;$ : initialize $I _ { 0 } = I _ { \mathrm { f i t } }$ , the fitting configuration rows, and for $t = 0 , \ldots , 5$ compute

$$
\begin{array} { l } { { a ^ { ( t ) } \in \arg \operatorname* { m i n } _ { a } \displaystyle \sum _ { i \in I _ { t } } ( y _ { i } - \phi _ { i } ^ { \top } a ) ^ { 2 } , \qquad e _ { i } ^ { ( t ) } = y _ { i } - \phi _ { i } ^ { \top } a ^ { ( t ) } , } } \\ { { \ } } \\ { { \tau _ { t } = \operatorname* { m a x } \Big \{ 5 Q _ { 0 , 9 8 } \Big ( \{ | e _ { i } ^ { ( t ) } | : i \in I _ { t } \} \Big ) , 1 0 ^ { - 3 } \mathrm { e V } \Big \} , \qquad J _ { t } = \{ i \in I _ { \mathrm { f i t } } : | e _ { i } ^ { ( t ) } | < \tau _ { t } \} . } } \end{array}
$$

Return the current coefficients if $| J _ { t } | \leq K + 1 \mathrm { o r } t = 5 ;$ otherwise set $I _ { t + 1 } = J _ { t }$ and refit. The threshold quantile uses the current retained rows, but $J _ { t }$ is evaluated over all original fitting rows, so a trimmed row can re-enter. At most six fits are performed.

Target residual normalization. Because the three residual targets in Eq. (2) can scale differently with atom count $N .$ we first remove their systematic size dependence. Let $r _ { t } ( R )$ denote the raw residual for target t, with $r _ { E } ( R ) = | \Delta E ( R )$ the aligned energy residual. For each target separately, the size exponent $\hat { \beta } _ { t }$ is the least-squares slope, with intercept, of log $r _ { t } ( R _ { i } )$ on log $N ( R _ { i } )$ over the fitting rows with $r _ { t } ( R _ { i } ) > 0$ . Writing $v _ { i t } = r _ { t } ( R _ { i } ) / N ( R _ { i } ) ^ { \hat { \beta } _ { t } }$ , the scale is the root mean square over the rows at or below the 0.99 quantile,

$$
q _ { t } = Q _ { 0 . 9 9 } \bigl ( \bigl \{ v _ { i t } : i \in I _ { \mathrm { f i t } } \bigr \} \bigr ) , \quad K _ { t } = \bigl \{ i \in I _ { \mathrm { f i t } } : v _ { i t } \leq q _ { t } \bigr \} , \quad \hat { c } _ { t } = \left[ \frac { 1 } { | K _ { t } | } \sum _ { i \in K _ { t } } v _ { i t } ^ { 2 } \right] ^ { 1 / 2 } ,\tag{14}
$$

where $Q _ { 0 . 9 9 }$ is the linearly interpolated empirical quantile with an inclusive boundary. Trimming estimates $\hat { c } _ { t }$ only: it removes no configuration from any set and does not clip $u _ { t }$ . The true and predicted residuals are then normalized using the same frozen quantities,

$$
u _ { t } ( R ) = \frac { r _ { t } ( R ) } { \hat { c } _ { t } N ( R ) ^ { \hat { \beta } _ { t } } } , \qquad \hat { u } _ { t } ( R ) = \frac { \hat { r } _ { t } ( R ) } { \hat { c } _ { t } N ( R ) ^ { \hat { \beta } _ { t } } } ,\tag{15}
$$

which removes systematic size dependence and places the three targets on comparable numerical scales. The fitted $\hat { \beta } _ { t }$ and $\hat { c } _ { t }$ are then frozen throughout calibration and inference. The predictor outputs $\hat { u } _ { t }$ directly, so a predicted residual in physical units is recovered by $\hat { r } _ { t } ( R ) = \hat { c } _ { t } N ( R ) ^ { \hat { \beta } _ { t } } \hat { u } _ { t } ( R )$

Committee-based scores. We provide more details on how we calculate sˆ for the committee baselines. The committee ranks configurations by disagreement among the three MLIP models used in Section 3. Let $m = 1 , \ldots , M$ denote each model $( M = 3 )$ , and let $\hat { e } _ { m } ( R )$ and $\hat { \mathbf { f } } _ { m , j } ( R ) \in \mathbb { R } ^ { 3 }$ be model $m \mathrm { { s } }$ predicted total energy and force on configuration R. We use the same frozen public checkpoints for these models as in other experiments in the paper; the committee contains MACE-OFF23(M) as a member on the two MACE-OFF23 cells, where it is excluded as a target.

For each model, raw total energies differ in scale by composition-linear offsets—up to $1 . 9 \times 1 0 ^ { 5 } \mathrm { e V }$ per iodine—that would dominate any genuine disagreement. We therefore place every member on a common scale before measuring the spread of predictions. Let $\phi ( R ) \in \mathbb { R } ^ { K + 1 }$ be the element-count vector of $R ,$ take member 1 as the reference, and fit

$$
\widehat { b } _ { m } = \mathrm { T r i m L S } _ { I _ { \mathrm { f i t } } } ( \widehat { e } _ { 1 } - \widehat { e } _ { m } ) , \qquad \widetilde { e } _ { m } ( R ) = \widehat { e } _ { m } ( R ) + \phi ( R ) ^ { \top } \widehat { b } _ { m } ,\tag{16}
$$

with $\tilde { e } _ { 1 } = \hat { e } _ { 1 }$ , using the same robust trimming as the energy alignment; only model outputs and element counts enter Eq. (16), so the aligned score remains label-free. We then calculate the spread of $\widetilde { e } _ { m } ( R )$ and $\hat { \mathbf { f } } _ { m , j }$ . The energy spread is

$$
\begin{array} { r } { d _ { E } ( R ) = \Big ( \frac { 1 } { M } \sum _ { m } \big ( \tilde { e } _ { m } ( R ) - \bar { e } ( R ) \big ) ^ { 2 } \Big ) ^ { 1 / 2 } , \qquad \bar { e } = \frac { 1 } { M } \sum _ { m } \tilde { e } _ { m } , } \end{array}\tag{17}
$$

and we summarize the atoms two ways to match the two force channels of s:

$$
d _ { \mathrm { a v g } } ( R ) = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \sigma _ { j } ( R ) , \qquad d _ { \mathrm { m a x } } ( R ) = \operatorname * { m a x } _ { j \in [ N ] } \sigma _ { j } ( R ) ,\tag{18}
$$

where

$$
\begin{array} { r } { \sigma _ { j } ( R ) = \left( \frac { 1 } { M } \sum _ { m } \bigl \| \hat { \mathbf { f } } _ { m , j } ( R ) - \bar { \mathbf { f } } _ { j } \bigr \| _ { 2 } ^ { 2 } \right) ^ { 1 / 2 } , \qquad \bar { \mathbf { f } } _ { j } = \frac { 1 } { M } \sum _ { m } \hat { \mathbf { f } } _ { m , j } . } \end{array}\tag{19}
$$

The three channels carry incomparable units, so the raw committee replaces each channel by its rank among the fitting configurations. With $n _ { \mathrm { f i t } }$ fitting configurations,

$$
\widehat { F } _ { c , \mathrm { f i t } } ( x ) = \frac { 1 } { n _ { \mathrm { f i t } } } \sum _ { R ^ { \prime } \in \mathcal { R } _ { \mathrm { f i t } } } \mathbf { 1 } \{ d _ { c } ( R ^ { \prime } ) \leq x \} , \qquad \widehat { s } _ { \mathrm { c o m m } } ( R ) = \sum _ { c } w _ { c } \widehat { F } _ { c , \mathrm { f i t } } ( d _ { c } ( R ) ) ,\tag{20}
$$

with the same weights w that define s in Section 2.3; the inclusive inequality fixes the treatment of ties. The supervised committee uses the same committee features but learns how to combine them: it replaces the fixed rank combination in Eq. (20) with one residual-regression network, with the same hidden widths and output channels as an individual structural scorer, trained with seed 0 on ${ \mathcal { D } } _ { \mathrm { f i t } }$ from the frozen channels $d _ { E } , d _ { \mathrm { a v g } } , d _ { \mathrm { m a x } }$ with the $L _ { 1 }$ objective. It thus tests whether a learned score on committee information can match the structural predictor. Both committees require three pretrained-MLIP evaluations per configuration.

We also experiment with two more baselines. Random passes a frozen randomized ranking through CRC and serves as a lower reference. Oracle passes the true score $\hat { s } = s$ through the same protocol and is retrospective: computing it requires exactly the labels the procedure is meant to avoid, so it is a cost reference rather than a deployable selector.

Electronic-structure references and force conventions. The residuals of Eq. (1) compare each MLIP with the dataset’s evaluation reference. At least one component of that reference differs from the ML $\varLambda  { \mathrm { P } }  { \mathrm { s } }$ training reference in every combination (Table 3). Residuals therefore include systematic differences between electronic-structure levels as well as MLIP approximation error; the composition-linear alignment removes an offset but does not make functionals, basis sets or dispersion treatments equivalent, and force residuals are not aligned. MACE-OFF23(M) is excluded as a target on the MACE-OFF23 population because of shared training-source provenance (25 exact configurations are recorded as shared with the associated training/validation source); complete training membership is not established for every model, and no blanket unseen or out-of-distribution claim is made. All comparisons use physical forces $\mathbf { F } = - \nabla _ { R } E$ in $\mathrm { e V } / \mathring \mathrm { A }$ and energies in eV. COMP6’s upstream array named forces stores the energy gradient, $+ \nabla _ { \boldsymbol { R } } E ;$ ingestion negates that raw vector exactly and converts Hartree/Å to eV/Å with factor 27.211386245988. MACE-OFF23 and QCML references already have physical-force semantics and receive no sign change. Absolute energies are never pooled across different electronic-structure references.

Table 3: Electronic-structure references. Each dataset is paired with the reference used to train each target MLIP; at least one component differs in every combination [Smith et al., 2018, Kovács et al., 2025, Anstine et al., 2025, Farr et al., 2026, Ganscha et al., 2025].
<table><tr><td>dataset</td><td>evaluation reference</td><td>target MLIP</td><td>MLIP training reference</td></tr><tr><td rowspan="3">COMP6</td><td rowspan="3">ωB97X/6-31G(d), Gaussian 09</td><td>MACE-OFF23(M)</td><td>ωB97M-D3(BJ)/def2-TZVPPD</td></tr><tr><td>AIMNet2</td><td>ωB97M-D3/def2-TZVPP</td></tr><tr><td>AceFF-2</td><td>ωB97M-V/def2-TZVPPD, VV10</td></tr><tr><td rowspan="3">QCML</td><td rowspan="3">PBE0, FHI-aims</td><td>MACE-OFF23(M)</td><td>ωB97M-D3(BJ)/def2-TZVPPD</td></tr><tr><td>AIMNet2</td><td>ωB97M-D3/def2-TZVPP</td></tr><tr><td>AceFF-2</td><td>ωB97M-V/def2-TZVPPD, VV10</td></tr><tr><td rowspan="2"></td><td rowspan="2">MACE-OFF23ωB97M-D3(BJ)/def2-TZVPPD, PSI4</td><td>AIMNet2</td><td>ωB97M-D3/def2-TZVPP</td></tr><tr><td>AceFF-2</td><td>ωB97M-V/def2-TZVPPD, VV10</td></tr></table>

## E The shared guarantee

This section extends the top row of Fig. 2 to all eight cells and asks what the finite-sample term of Definition 1 costs.   
The risk ratio $r ( \alpha )$ and the realized retained loss $\rho ( \alpha )$ are defined in Section 3.1 and Appendix ??.

Across all cells, scores, targets, and reallocations, the largest mean risk ratio over all reallocations is 1.0359 and the largest single draw is 1.333 (Fig. 4); the guarantee bounds an expectation, so individual draws and finite means may exceed 1. At small α the ratio falls well below 1 for every score. The cutoff keeps the calibration loss under $( n + 1 ) \epsilon _ { 0 } - B$ , and $\epsilon _ { 0 } = \alpha \ell _ { \mathrm { f i t } }$ shrinks with α while the identity bound B does not, so the fixed term $B / ( n { + } 1 )$ consumes most of a small budget. The retained fraction $\rho ( \alpha )$ tracks α for every score (Fig. 5); points below the diagonal mean that the rule spent more DFT than the target required.

Fig. 6 holds the ranking score fixed and varies the calibration rule and the number n of calibration identities, at $\alpha = 0 . 3 0$ A plug-in cutoff that omits $B / ( n { + } 1 )$ has mean risk/budget 1.01–1.06 for $n = 5 – 1 0 0$ and exceeds the budget in about 52% of calibration draws; a cutoff fixed on ${ \mathcal { D } } _ { \mathrm { f i t } }$ alone exceeds it in 62%. Finite CRC stays below budget on average (0.17 and 0.9% violations at $n = 1 0 )$ but routes the whole pool at n = 5 and 74.6% at $n = 1 0 0$ , against 57.8% at the full calibration size. Violation frequencies describe the observed draws; CRC makes no high-probability claim.

![](images/2d13974cf4534c8819e096a57f5963657015bb9b9949f06d68a5cedb2e1a555c.jpg)  
Ours (1 MLIP) Raw committee (3 MLIPs) Supervised committee (3 MLIPs) Oracle Random r = 1

Figure 4: Every score meets the same risk guarantee in every cell. Risk ratio r(α) (realized capped test loss over the budget $\epsilon _ { 0 } = \alpha \ell _ { \mathrm { f i t } } )$ for all eight cells and five scores. Faint lines are the 20 calibration/test reallocations; bold lines and bands are means ±1 SD; the dashed line marks $r = 1$ . Single draws above 1 are expected because the bound is on the expectation. On QCML × MACE-OFF23(M) the supervised committee routes the whole pool for $\alpha \leq 0 . 2 5 ,$ , so its r is exactly 0 there.

![](images/93f3cbdf39cb93aadd60c86c4f0b24495a4f129991715117af726141aad72b80.jpg)  
Ours (1 MLIP) Raw committee (3 MLIPs) Supervised committee (3 MLIPs) Oracle Random ρ = α  
Figure 5: The realized retained loss tracks the target for every score. Retained fraction $\rho ( \alpha )$ , the capped loss left uncorrected over the pool’s own total, for all eight cells; the dashed line is $\rho = \alpha , \ast$ visual reference rather than a guarantee. Faint lines: the 20 reallocations; bold: mean ±1 SD. Points below the diagonal (MACE-OFF23 cells; the tied supervised committee on QCML) mean that more DFT was spent than the target required.

![](images/ad3694f29c09176e233443ed13aaa35553c4e9ff2c8b844949949df179a52a82.jpg)

![](images/04628006d89cce3b88886ee5d528b6553f8f4374e28881c977bda569c69a8ed0.jpg)

![](images/ffe0f062fda6a587a8152c250ad7897efd36cb461ae4e4be06538b321b1b7018.jpg)  
Figure 6: The finite correction keeps the budget; its price is DFT when calibration is scarce. (a) Realized risk/budget and (b) fraction of calibration draws exceeding the budget at $\alpha = 0 . 3 0$ , for finite CRC, a plug-in cutoff without $B / ( n { + } 1 )$ , and a cutoff fixed on $\mathcal { D } _ { \mathrm { { f i t } } } ;$ equal-cell means over eight cells (200 draws for $n \leq 1 0 0 )$ . (c) Deployment DFT at $\alpha = 0 . 3 0 \colon$ faint lines are cells on their own calibration grids, the bold line the equal-cell mean over the five shared n. The ranking score is fixed throughout; log x axes.

## F DFT cost in every cell

This section extends the middle and bottom row of Fig. 2 and Fig. 3(b) to every cell. Table 4 gives the equal-cell means, Table 5 the per-cell G, and Table 6 the matched-resource comparisons.

$\mathrm { A t } \alpha = 0 . 3 0$ , ours routes less DFT than both committees in five of eight cells (Fig. 7); the raw committee is closer to the oracle on $\mathbf { Q C M L } \times \mathbf { M A C E - O F F } 2 3 ( \mathbf { M } )$ , and the committees match or beat ours on the two MACE-OFF23 cells. On QCML × MACE-OFF23(M) the supervised committee routes the entire pool for $\alpha \leq 0 . 2 5$ , and in 14 of 20 reallocations at $\alpha = 0 . 3 0$ . Note the rule permits this conservative fallback, and it is why that score has $G < 0$ . Weighting each call by $N ^ { 3 }$ (Fig. 8), ours is cheaper than the raw committee and random in all eight cells for $\alpha \ge 0 . 1 5$ and cheaper than the supervised committee in seven, the exception being MACE-OFF23 × AceFF-2. At equal realized loss (Fig. 9), ours saves DFT against each committee in six of eight cells.

Table 4: Main results, equal-weight means over the eight cells. G is the normalized gain Eq. (12); DFT and $\rho$ are the deployment DFT fraction and realized retained loss at $\alpha = 0 . 3 0$ . The last three columns compare each committee with ours over the common attained support of every cell: DFT saved by ours at matched ρ (Appendix F.1), and the relative-energy MAE and selection-success change of ours at matched deployment DFT (Appendix J).
<table><tr><td colspan="3"></td><td colspan="2">at α = 0.30</td><td colspan="3">committee vs. ours at matched resources</td></tr><tr><td>Method</td><td>MLIP evals</td><td>G</td><td>DFT [%]</td><td>ρ</td><td>DFT saved [pp of pool]</td><td>rel.-energy MAE [meV]</td><td>selection success [pp]</td></tr><tr><td>Ours</td><td>1</td><td>0.808</td><td></td><td>57.8 0.289</td><td></td><td></td><td></td></tr><tr><td>Raw committee</td><td>3</td><td>0.497</td><td></td><td>61.9 0.282</td><td>+3.0</td><td>-2.0</td><td>+0.7</td></tr><tr><td>Supervised committee</td><td>3</td><td>0.341</td><td></td><td>67.5 0.251</td><td>+4.0</td><td>-9.7</td><td>+1.5</td></tr><tr><td>Random</td><td>1</td><td>0.000</td><td></td><td>71.6 0.284</td><td></td><td></td><td></td></tr><tr><td>Oracle (retrospective)</td><td></td><td>1.000</td><td></td><td>54.4 0.288</td><td></td><td></td><td></td></tr></table>

## F.1 Matched-resource comparisons

Equal α does not imply equal realized loss: at $\alpha = 0 . 3 0$ the supervised committee retains $\rho = 0 . 2 5 1$ against 0.289 for ours (Table 4), so part of its extra DFT buys a lower retained loss. For each cell we therefore form the 20-reallocation mean curves at the ten α values. The common ρ interval $I _ { \rho }$ is the intersection of the ranges attained by ours and the two committees; random, oracle, and the all-MLIP and all-DFT endpoints do not extend it. Within $I _ { \rho } ,$ piecewise-linear

![](images/146af549ec5ace519240ee84108dbf75366b1e53834803ab05167b767f41f337.jpg)  
Figure 7: Within the shared guarantee, ours needs the fewest DFT calls of the deployable scores in most cells. DFT calls k(α) for all eight cells; the gray band is the random-to-oracle headroom and each panel prints G for ours and the two committees. Means over 20 reallocations with ±1 SD bands for ours and the committees. The raw committee wins QCML × MACE-OFF23(M) and the committees match or beat ours on MACE-OFF23. The oracle is retrospective, not deployable.

![](images/7ea5e75ad15145f2c4d91d24de523aaa46668beb4d61fec4dc74880d42f7b0ac.jpg)  
Figure 8: Weighting each DFT call by $N ^ { 3 }$ preserves, and on QCML widens, the advantage of ours. $N ^ { 3 } .$ -weighted DFT cost, the summed cubed atom counts of routed configurations over the same sum for the pool, for all eight cells; the dotted line marks $\alpha = 0 . 3 0$ . Means over 20 reallocations with ±1 SD bands for ours and the committees. The proxy captures size scaling, not measured wall time, and the comparison fixes α, not the realized loss.

![](images/ff5aaa8e94f2bfa20257ea8b068e326b5131ab76fc1f41d0aace4dd757359a0c.jpg)  
Figure 9: At equal realized loss, ours still calls DFT least in most cells. DFT calls against realized retained loss $\rho ;$ each point is one target α at the 20-reallocation mean, with ±1 SD whiskers in both coordinates. The shaded interval is the common attained ρ range of ours and the two committees, over which Fig. 3(b) averages the DFT difference. On QCML × MACE-OFF23(M), the supervised committee’s first five α values collapse to one all-DFT point at ρ = 0.

Table 5: Normalized gain G per cell. Each G uses the 20-reallocation mean DFT-call curve on the α grid. Among ours and the two committees, bold marks the best and underlining the second best; the last three columns are label-free controls (Appendix H), which never exceed ours. The final row is the equal-cell mean.
<table><tr><td rowspan="2">Cell</td><td colspan="3">primary scores</td><td colspan="3">label-free controls</td></tr><tr><td>Ours</td><td>Raw</td><td>Supervised</td><td>kNN-1</td><td></td><td>kNN-5 Max-force</td></tr><tr><td>COMP6 × MACE-OFF23(M)</td><td>0.879</td><td>0.223</td><td>0.274</td><td>0.261</td><td>0.267</td><td>0.134</td></tr><tr><td> $\mathrm { C O M P 6 } \times \mathrm { A I M N e t 2 }$ </td><td>0.822</td><td>0.237</td><td>0.255</td><td>0.236</td><td>0.244</td><td>0.131</td></tr><tr><td> $\mathrm { C O M P 6 } \times \mathrm { A c e F F } { - 2 }$ </td><td>0.888</td><td>0.136</td><td>0.183</td><td>0.226</td><td>0.232</td><td>0.079</td></tr><tr><td> ${ \bf Q } { \bf C M L } \times { \bf M A C E . O F F } 2 3 ( { \bf M } )$ </td><td>0.820</td><td>0.910</td><td>-0.278</td><td>0.362</td><td>0.385</td><td>0.245</td></tr><tr><td> $\mathrm { Q C M L } \times \mathrm { A I M N e t 2 }$ </td><td>0.748</td><td>0.529</td><td>0.395</td><td>0.371</td><td>0.389</td><td>0.322</td></tr><tr><td> $\mathbf { Q C M L } \times \mathbf { A c e F F { - } } 2$ </td><td>0.837</td><td>0.523</td><td>0.317</td><td>0.333</td><td>0.361</td><td>0.163</td></tr><tr><td> $\mathbf { M A C E - O F F } 2 3 \times \mathbf { A I M N e t } 2$ </td><td>0.706</td><td>0.716</td><td>0.763</td><td>0.471</td><td>0.471</td><td>0.419</td></tr><tr><td> $\mathbf { M A C E - O F F } 2 3 \times \mathbf { A c e F F - 2 }$ </td><td>0.763</td><td>0.705</td><td>0.815</td><td>0.246</td><td>0.272</td><td>0.315</td></tr><tr><td>Mean</td><td>0.808</td><td>0.497</td><td>0.341</td><td>0.313</td><td>0.327</td><td>0.226</td></tr></table>

interpolation defines each curve, and for baseline b

$$
\overline { { \Delta \mathrm { D F T } _ { b } } } = \frac { 1 } { \left| I _ { \rho } \right| } \int _ { I _ { \rho } } \left[ \mathrm { D F T } _ { b } ( \rho ) - \mathrm { D F T } _ { \mathrm { o u r s } } ( \rho ) \right] d \rho ,\tag{21}
$$

integrated exactly over the union of curve knots, so positive values favor ours. Matched deployment DFT uses the analogous common DFT interval, with differences $\mathrm { M } \dot { \mathrm { A E } } _ { \mathrm { o u r s } } - \mathrm { M } \mathrm { A } \mathrm { E } _ { b }$ and $\mathrm { S u c c e s s } _ { \mathrm { o u r s } } - \mathrm { S u c c e s s } _ { b }$ (Appendix J). Cell values are averaged with equal weight. No extrapolation or endpoint extension is used, and these are descriptive comparisons over common support, not new CRC guarantees.

Table 6: Matched resources over common attained support. (a) Comparisons with the raw and supervised committees; arrows mark the direction favoring ours. DFT saving is baseline minus ours; MAE and success changes are ours minus baseline. The bold mean weights the eight cells equally; pp denotes percentage points. (b) Common intervals used for the comparisons.
<table><tr><td colspan="8">(a) Matched-resource comparisons</td></tr><tr><td rowspan="3">Dataset</td><td rowspan="3">MLIP</td><td colspan="2">DFT saving ↑ [pp]</td><td colspan="2">MAE change ↓ [meV]</td><td colspan="2">Success change ↑ [pp]</td></tr><tr><td>Raw</td><td>Sup.</td><td>Raw</td><td>Sup.</td><td>Raw</td><td>Sup.</td></tr><tr><td>+5.85</td><td>+5.28</td><td>-16.12</td><td>-16.60</td><td>+2.19</td><td>+2.73</td></tr><tr><td rowspan="3">COMP6</td><td>MACE-OFF23(M)</td><td>+5.01</td><td>+4.77</td><td>-14.13</td><td>-14.40</td><td>+2.10</td><td>+2.51</td></tr><tr><td>AIMNet2 AceFF-2</td><td>+6.47</td><td>+5.94</td><td>-14.63</td><td>-15.43</td><td>+2.67</td><td>+2.91</td></tr><tr><td></td><td></td><td></td><td></td><td>-36.87</td><td></td><td></td></tr><tr><td rowspan="3">QCML</td><td>MACE-OFF23(M)</td><td>-2.51</td><td>+8.45 +4.24</td><td>+16.13 +5.56</td><td>+3.25</td><td>-1.19 -0.55</td><td>+3.93 -0.32</td></tr><tr><td>AIMNet2 AceFF-2</td><td>+3.50 +5.49</td><td>+5.69</td><td>-1.80</td><td>-3.69</td><td>+0.34</td><td>+0.58</td></tr><tr><td>AIMNet2</td><td>-0.04</td><td></td><td>+5.40</td><td>+4.30</td><td>-0.33</td><td></td></tr><tr><td rowspan="2">MACE-OFF23</td><td>AceFF-2</td><td>+0.39</td><td>-0.94 -1.34</td><td>+3.57</td><td>+1.50</td><td>+0.15</td><td>-0.21 +0.09</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">Mean</td><td colspan="2">+3.02 +4.01</td><td>-2.00</td><td>-9.74</td><td>+0.67</td><td>+1.53</td></tr><tr><td colspan="2">(b) Common attained support</td><td colspan="6"></td></tr><tr><td colspan="2">Dataset MLIP</td><td colspan="6">Deployment DFT IDFT [%]</td></tr><tr><td rowspan="4">COMP6</td><td colspan="2"></td><td colspan="2">Realized loss  $I _ { \rho }$  [0.048, 0.479]</td><td colspan="3">[48.17, 92.28]</td></tr><tr><td colspan="2">MACE-OFF23(M) AIMNet2</td><td colspan="2">[0.047, 0.480]</td><td colspan="3">[48.22, 92.78]</td></tr><tr><td colspan="2">AceFF-2</td><td colspan="2">[0.047, 0.478]</td><td colspan="3">[49.95, 92.63]</td></tr><tr><td colspan="2">MACE-OFF23(M)</td><td colspan="2">[0.038, 0.476]</td><td colspan="3">[27.06, 87.11]</td></tr><tr><td colspan="2">QCML</td><td colspan="4"></td><td colspan="2">[38.79, 92.22]</td></tr><tr><td colspan="2"></td><td colspan="2">AIMNet2</td><td colspan="2">[0.042, 0.492] [0.048, 0.487]</td><td colspan="2">[38.39, 90.11]</td></tr><tr><td colspan="2">MACE-OFF23 AIMNet2</td><td colspan="2">AceFF-2</td><td colspan="2">[0.027, 0.459]</td><td colspan="2">[39.90, 94.57]</td></tr><tr><td colspan="2">AceFF-2</td><td colspan="2"></td><td colspan="2">[0.021, 0.442]</td><td colspan="2">[41.23, 94.94]</td></tr></table>

## G Why the gap opens and closes

This section tests the reading of Fig. 3(a): the structural score wins where the MLIPs share error with one another, and disagreement wins where failures are rare, catastrophic, and model-specific.

Matched reference. ANI-2x is a different MLIP whose eight released members were trained at the nominal COMP6 reference, ωB97X/6-31G(d). We use the mean of members 0–7 as the target, the native committee is their disagreement (population standard deviation), and the supervised ANI committee applies the supervised-committee construction to the same ensemble. If disagreement misses shared error, it should recover here. It does, slightly (Fig. 10): at α = 0.30 ours has $G = 0 . 6 6$ and routes 57.9%, against 0.69 and 57.2% for both ANI committees and 70.7% for random, and at matched $\rho$ ours needs 0.71 and 0.75 pp more DFT. Training membership of these configurations could not be established, so this is not an unseen-data test.

Isolated dispersion control. D3(BJ) is an empirical correction for long-range dispersion that the evaluation and training references treat differently. A score built from the D3(BJ) term alone reaches $G = 0 . 0 5 – 0 . 1 2$ in the six COMP6 and QCML cells, while adding D3(BJ) to the reference and refitting leaves structural G at 0.75–0.90, with each cell changing by at most 0.02 and DFT at α = 0.30 by under 1 pp (Fig. 11). This is a diagnostic control, not proof that all electronic-structure mismatch is absent: functional form, basis set, and other effects remain.

Residual concentration. The Gini coefficient of the pool’s residual mass is ≤ 0.18 on COMP6, where error is diffuse, and 1.00 on QCML × MACE-OFF23(M), where a handful of configurations carry essentially all of it (Table 7, Fig. 12). The latter is the one cell where the raw committee beats ours.

random–oracle headroom

Shared and target-specific components. For each configuration, the consensus residual is the mean of the three MLIPs’ signed aligned energy residuals, or of their per-atom force-residual vectors, and the target-specific residual is the target’s residual minus that consensus, with force norms taken after vector subtraction. On the resample-0 test pools, the mean Spearman correlation of ours over the eight cells is 0.47/0.54/0.45 with the consensus energy, mean-force, and maximum-force magnitudes, against 0.28/0.45/0.36 with the target-specific ones. The raw committee correlates more with the target-specific force components (0.89/0.86) than with the consensus ones (0.65/0.62). The target-specific part equals the deviation of the target’s prediction from the ensemble mean, so the committee’s association with it is partly built in; these are descriptive associations, not a causal decomposition.

Model internals and hybrids. The latent scores feed the target MLIP’s per-atom invariant features (256 dimensions), mean- or max-pooled over atoms, to the same predictor and CRC protocol as ours. The hybrid appends the three disagreement channels of the raw committee to our 350-dimensional descriptor. Native committees use members of the target’s own family: AIMNet2 members 0–3 with member 0 as the frozen target, and MACE-OFF23 S/M/L with M as the target; they are undefined for AceFF-2 targets and for MACE-OFF23(M) on the MACE-OFF23 population. Latent mean pooling (mean $G = 0 . 8 2 )$ exceeds ours in five of eight cells, while max pooling (0.72) drops to 0.60–0.68 on COMP6 (Fig. 13). The hybrid (0.83) is at or above ours in every cell, by +0.11 and +0.07 on MACE-OFF23 and by at most 0.02 elsewhere; the QCML gains are too small to call improvements. Native committees reach only 0.14–0.24 on COMP6, and the native supervised committee has G = −1.07 on QCML × MACE-OFF23(M). Fig. 14 gives every source in every cell.

![](images/0179a770c1b282c3860fc5ef1dcd2921d1969345acde06466a5cf7886cab016c.jpg)

![](images/d40be5bcc7cd4373c6864db4c9d8e99c51b5c54bc83b577bcbaf63e5f5923063.jpg)

![](images/ba8f112c3ab790051b93f92713abe8d52d8db46322726f70b7e0b92d2b8f3339.jpg)  
Figure 10: With a matched reference, committees edge ahead, and the structural score stays far above random. ANI-2x × COMP6: (a) DFT calls with the random-to-oracle headroom shaded; (b) risk ratio r(α); (c) G and DFT calls at $\alpha = 0 . 3 0$ for ours, the native ANI committee (members 0–7; target is their mean), and the supervised ANI committee. Means over 20 reallocations with ±1 SD bands for the three selectors. ANI-2x training membership for these configurations is unverified, so this is not an unseen-data test.

(a) D3 alone ranks weakly

![](images/ba88c38fca538baf91beaf35bc87eaa0f21fdce0d1755df28bb81e54c346aa52.jpg)

(b) Ranking survives + D3  
![](images/df3897bd80e6489157836506234aa15f1fa6bdbdd9646e1575420b84ef6aa829.jpg)

(c) Cost unchanged  
![](images/8b0779c5a16558745ce0bbc30057b5b8893642e5e42e6a776182c891f319e4e4.jpg)  
COMP6 QCML MACE-OFF23(M) AIMNet2 AceFF-2  
Figure 11: An isolated D3(BJ) term does not explain the structural ranking. (a) Structural G against the G of a D3(BJ)-only score (dashed: equal G). (b) Structural G and (c) DFT calls at $\alpha = 0 . 3 0$ with the original reference and after adding D3(BJ) to the reference, joined per cell. Six COMP6 and QCML cells; cell statistics over 20 reallocations. Adjusting the reference changes the loss and refits the predictor, so before and after G use their own random and oracle anchors. A diagnostic, not proof that referenc mismatch is absent.

Table 7: Residual concentration and frozen loss bounds. Resample-0 test pool. Gini uses untruncated configuration scores s; “Top 10%” is the untruncated score mass in the highest-mass 10% of identities. $m _ { \mathrm { m a x } }$ is the maximum fitting-identity size, $B _ { \mathrm { c f g } }$ the configuration truncation level, and B the identity bound of Eq. (6). The last two columns are in meV.
<table><tr><td>Dataset</td><td>MLIP</td><td>Gini Top 10%</td><td></td><td> $m _ { \mathrm { m a x } }$ </td><td> $B _ { \mathrm { c f g } }$ </td><td>B Med.</td><td> $\lvert \Delta E \rvert$ </td><td> $p _ { 9 9 } | \Delta E |$ </td></tr><tr><td>COMP6</td><td>MACE-OFF23(M)</td><td>0.18</td><td>0.16</td><td>24</td><td>3.4</td><td>45.4</td><td>139</td><td>516</td></tr><tr><td rowspan="4">QCML</td><td>AIMNet2</td><td>0.16</td><td>0.15</td><td>24</td><td>2.2</td><td>46.4</td><td>132</td><td>474</td></tr><tr><td>AceFF-2</td><td>0.16</td><td>0.15</td><td>24</td><td>2.1</td><td>47.4</td><td>132</td><td>443</td></tr><tr><td>MACE-OFF23(M)</td><td>1.00</td><td>1.00</td><td></td><td>948.2</td><td>158.5</td><td>485</td><td>3,895</td></tr><tr><td>AIMNet2</td><td>0.45</td><td>0.40</td><td></td><td>934.3</td><td>122.6</td><td>126</td><td>839</td></tr><tr><td rowspan="3">MACE-OFF23</td><td>AceFF-2</td><td>0.46</td><td>0.40</td><td></td><td>911.0</td><td>42.4</td><td>202</td><td>1,151</td></tr><tr><td>AIMNet2</td><td>0.30</td><td>0.42</td><td>229</td><td>19.3</td><td>3210.2</td><td>41</td><td>277</td></tr><tr><td>AceFF-2</td><td>0.30</td><td>0.45</td><td>229</td><td>25.0</td><td>273.3</td><td>82</td><td>396</td></tr></table>

![](images/22935940eb0d3ba7ca0f6c5fc6f196593f02d3cee08c7e973c97ed3645c221da.jpg)

![](images/9867df107264fc95aa2afaf5da0577c0a01a21aa8cb343dee6cc5bf5becc539a.jpg)

![](images/30fe70d13af2ccc4fec81fc9f8ee8977b2fbd88ec69f465bb8eb3e5f2f34d103.jpg)  
Figure 12: Residual scales differ by orders of magnitude across cells, with catastrophic tails only on QCML. Aligned energy error, mean atom-force error, and maximum atom-force error of the target MLIP on the complete resample-0 test pools. Thin lines span minimum to maximum, thick lines the 5th–95th percentiles, markers the medians. Asinh axes, linear below 0.01, so zero and the largest ${ \mathrm { Q C M L } } \times { \mathrm { M A C E . O F F } } { 2 3 } { ( \mathrm { M } ) }$ force error $( \bar { 2 } . 5 \times 1 0 ^ { 7 } ~ \mathrm { e V / \mathring { A } } )$ are both visible; nothing is omitted.

(a) Latent mean rivals structure  
![](images/52569889ec8193409296cd2d6daace7ad275d6dff98c7cbfce52eb84e823d3f0.jpg)  
(c) Native committees vary by cell

(b) Disagreement adds most on MACE-OFF23  
![](images/f6ac011ab6e66d97d52683c5285a29d2d5036dffd6698a721ef020db8873f627.jpg)  
Figure 13: Target-model latents rival but do not dominate the structural score, and disagreement adds most on MACE-OFF23. (a) Per-cell G for ours, latent mean, latent max, and the hybrid (y axis zoomed to 0.55–1). (b) Hybrid minus ours. (c) Native raw and supervised committees against ours in the five cells where they are defined; the native supervised value −1.07 on QCML × MACE-OFF23(M) is drawn clipped. Cell statistics over 20 reallocations. The small QCML hybrid gains are not claimed as improvements.

![](images/1370fefc94a7c1fd4c69949fb2ed2869787f0346fb38d511fa507a5a325b1c07.jpg)  
Native raw committee  
Native supervised committee

(a) Structure and latents lead in most cells  
![](images/c29b219958fc7d04ef385e78ba0005bec559bbc819a3dc1d3080ceadb01dabde.jpg)

(b) Fewer DFT calls where G is high  
![](images/a8ffab233f1e1621f431a6f13d1f5f1cabf3aa1afeec96792cce131c77d251e5.jpg)  
Figure 14: Every score source in every cell. (a) G (diverging map centered at 0) and (b) DFT calls at α = 0.30 (fixed 0–100 scale) for ours, latent mean and max, the hybrid, the raw and supervised committees, and the native raw and supervised committees. Cell statistics over 20 reallocations. Hatched NA entries are undefined native committees, not zeros.

## H Are the committees and controls strong enough?

This section asks whether the gap in Fig. 3(a, c) reflects weak baselines. Two-member committees keep the target and one other model, or drop the target; the latter still needs three distinct MLIPs, since the target must be evaluated. No two-member subset approaches ours on average (Fig. 15): mean G is 0.445–0.494 for raw and 0.244–0.509 for supervised subsets, against 0.808 for ours. Membership effects are cell-specific; the supervised AIMNet2 + AceFF-2 subset gains +0.17 on average, driven by QCML × MACE-OFF23(M) (+0.86), where the full supervised committee is below random. Some subsets beat ours in individual QCML × MACE-OFF23(M) and MACE-OFF23 cells.

The model-agnostic controls pass through the same loss and CRC rule (Fig. 16): kNN-1 and kNN-5, the descriptor-space distance to the nearest fitting configuration or the mean distance to the nearest five (G = 0.31 and 0.33); max predicted force, the target MLIP’s largest predicted atomic force magnitude (0.23); a composition + atom count score and a learned atom count score, each trained with our predictor’s targets and objective on element counts and atom count, or atom count alone (0.31 and −0.17); and raw atom count (−0.31). Ours exceeds every control in every cell. Composition and size carry some signal but do not account for the structural gain.

(a) No subset approaches ours on average  
![](images/9895cd44de214ca2e520f9c04268f32c3d3f038ed789d49bc7872be907f5fe09.jpg)

(b) Membership effects are cell-specific  
![](images/81158b92b1ceaebe5c45716929de6ba99a0e1c50018bd72ee37a9932463d390b.jpg)  
Figure 15: No two-member committee approaches the structural score on average; membership effects are cell-specific. (a) G for ours, the full raw and supervised committees (K=3), and each two-member subset (K=2), per cell and as the equal-cell mean. A black corner triangle marks subsets that exclude the target MLIP; these still need three distinct models, so they are not cheaper than K=3. (b) $\Delta G = G _ { K = 2 } - G _ { K = 3 }$ per cell and subset, raw (green) and supervised (blue), with equal-cell means at right.

![](images/6fdbae55e6dfb32591d505edb5b8440a73ce2807d7e281c0e0b813d01fccee0a.jpg)  
Figure 16: The structural score beats every simple model-agnostic control in every cell. Per-cell G for ours and six controls under the same loss and CRC rule: kNN-1 and kNN-5 descriptor-space distance, the target MLIP’s maximum predicted force, learned composition + atom count, learned atom count, and raw atom count. Gray lines join the same cell (marker: dataset; color: MLIP); bars are equal-cell means. Negative G (worse than random) is shown unclipped and does not imply that size is irrelevant.

## I What gives the structural score its ranking

This section explains Fig. 3(c). CRC uses a score only through the order it induces, so ranking quality is what becomes DFT savings. Across all scores and cells, G rises with the mean test Spearman correlation between sˆ and s (Fig. 17). The predictor ranks residuals even where magnitude regression fails, as on QCML × MACE-OFF23(M), where energy R<sup>2</sup> is 0.007 but the Spearman correlation is 0.66 (Fig. 18).

Representation. The full descriptor reaches mean G = 0.81 against 0.56 for ACSF and 0.48 for SOAP, and routes 57.8% at α = 0.30 against 61.8% and 63.0% (Figs. 19 and 20). This concerns the tested residual-ranking task, not descriptor quality in general. Removing one of the nine feature families changes mean G by at most 0.03 (Coordination, −0.028; PairState, −0.019; RBF, −0.013; the rest within ±0.006), so the ranking information is spread across families; small positive changes are point estimates, not improvements.

Training choices. More fitting labels improve ranking: mean G rises from 0.54 with 2.5% of population identities to 0.81 with 40% (Fig. 21a). Each fitting size refits the loss quantities as well as the predictor, so each size defines its own proxy loss. The pairwise ranking loss does not materially help at the frozen weight: $\lambda _ { \mathrm { r a n k } } \in \{ 0 , 0 . 1 , 1 \}$ give mean G = 0.807, 0.806, and 0.808, and $\lambda _ { \mathrm { r a n k } } = 1 0$ gives 0.70 (Fig. 21b). The frozen $\lambda _ { \mathrm { { r a n k } } } = 1$ is retained because it was selected before test evaluation.

Fitting-only selection. Architecture, hyperparameters, the three seeds, $\lambda _ { \mathrm { { r a n k } } } = 1$ , and arithmetic averaging were chosen on ${ \mathcal { D } } _ { \mathrm { f i t } }$ and its fixed internal validation rows by the equal-cell mean Spearman correlation with the capped loss ℓ, then frozen before calibration and testing. The internal split is by configuration, so validation identities also appear in training rows; it supports selection, not an identity-level generalization test. The diagnostics above were computed after freezing and did not enter selection.

![](images/6911e852edfb59f233dfe0192426fccca450e9740e3e6ca5005276c6e1a57338.jpg)  
Figure 17: Scores that rank configurations better capture more of the random-to-oracle headroom. G against the mean test Spearman correlation between score and true error, per cell (marker: dataset) and score (color): the five main scores plus latent mean, latent max, hybrid, SOAP, and ACSF (hollow). Spearman values are means over 20 test reallocations; G uses the mean DFT-call curve. Oracle and random sit near (1, 1) and (0, 0).

![](images/cbebfe7191f3bf779991da16b8de65c4f0a2de251cdf40000d3fc3bf88c63b77.jpg)  
Figure 18: The predictor ranks energy residuals even where it misestimates their size. True against predicted aligned |∆E| (energy channel of the frozen predictor) for every resample-0 test configuration; asinh axes, linear below 0.01 eV; dashed line $y = x ,$ Boxes give n, Spearman correlation, MAE, and $R ^ { 2 }$ on raw residuals with outliers kept. The QCML × MACE-OFF23(M) axi reaches about $1 0 ^ { \circ } \mathrm { e V }$ because of a few catastrophic residuals, which is also why its $R ^ { 2 }$ is near 0.

![](images/6884aeda34b266cb730b3b228a369367ff74b4420f240b6f6e2966a9e6d2825c.jpg)

![](images/529a8c6cf56b36922bd3ddf1d1f778ed1d190aff91c3fa162b88f6b64719c7cd.jpg)

(c) Fewer DFT calls  
![](images/52474aa23c60d97d257046e91620d2e60ab7f780abb835bba463254c8f649bf4.jpg)  
Figure 19: The full descriptor outperforms SOAP and ACSF, and no single feature family carries the signal. (a) Equal-cell mean G (bars) and per-cell values (dots) for the full 350-dimensional descriptor, SOAP, and ACSF. (b) ∆G (drop minus full) for each single-family drop, ordered by mean effect. (c) DFT calls at $\alpha = 0 . 3 0$ (y axis zoomed to 40–75%). All use the frozen fitting protocol and 20 reallocations; small positive ∆G values are not significant improvements.

(a) Normalized gain G
<table><tr><td rowspan=1 colspan=11>COMP6        QCML  MACE-OFF23</td></tr><tr><td rowspan=1 colspan=1>Full 350-D</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.81</td></tr><tr><td rowspan=1 colspan=1>SOAP</td><td rowspan=1 colspan=1>0.41</td><td rowspan=1 colspan=1>0.40</td><td rowspan=1 colspan=1>0.30</td><td rowspan=1 colspan=1>0.54</td><td rowspan=1 colspan=1>0.52</td><td rowspan=1 colspan=1>0.55</td><td rowspan=1 colspan=1>0.56</td><td rowspan=1 colspan=1>0.57</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.48</td></tr><tr><td rowspan=1 colspan=1>ACSF</td><td rowspan=1 colspan=1>0.51</td><td rowspan=1 colspan=1>0.47</td><td rowspan=1 colspan=1>0.47</td><td rowspan=1 colspan=1>0.66</td><td rowspan=1 colspan=1>0.55</td><td rowspan=1 colspan=1>0.73</td><td rowspan=1 colspan=1>0.52</td><td rowspan=1 colspan=1>0.57</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.56</td></tr><tr><td rowspan=1 colspan=1>drop Coordination</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.72</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.78</td></tr><tr><td rowspan=1 colspan=1>drop PairState</td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.79</td></tr><tr><td rowspan=1 colspan=1>drop RBF</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.79</td></tr><tr><td rowspan=1 colspan=1>drop SoftCrowding</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.80</td></tr><tr><td rowspan=1 colspan=1>drop Extreme</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.68</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.80</td></tr><tr><td rowspan=1 colspan=1>drop ElementCount</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.81</td></tr><tr><td rowspan=1 colspan=1>drop ZPotential</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.81</td></tr><tr><td rowspan=1 colspan=1>drop InverseDistance</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.72</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.81</td></tr><tr><td rowspan=1 colspan=1>drop ShortRange</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.81</td></tr><tr><td rowspan=4 colspan=1></td><td rowspan=2 colspan=1>MAC(()</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=2 colspan=1>MAC((M)</td><td rowspan=2 colspan=1>AIMt2</td><td rowspan=2 colspan=1>A2</td><td rowspan=2 colspan=1>AIMt2</td><td rowspan=2 colspan=1>A2</td><td rowspan=2 colspan=2>meəan</td></tr><tr><td rowspan=1 colspan=1>AIMt2</td><td rowspan=1 colspan=1>A-2</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td></tr><tr><td rowspan=1 colspan=1>-1.0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.5</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.</td><td rowspan=1 colspan=3>5        1.0</td></tr></table>

(b) DFT calls at α = 0.30 [% of pool]
<table><tr><td rowspan=1 colspan=1>60.2</td><td rowspan=1 colspan=1>61.3</td><td rowspan=1 colspan=1>61.1</td><td rowspan=1 colspan=1>46.9</td><td rowspan=1 colspan=1>56.5</td><td rowspan=1 colspan=1>53.3</td><td rowspan=1 colspan=1>61.1</td><td rowspan=1 colspan=1>61.9</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>57.8</td></tr><tr><td rowspan=1 colspan=1>66.6</td><td rowspan=1 colspan=1>66.5</td><td rowspan=1 colspan=1>68.1</td><td rowspan=1 colspan=1>54.8</td><td rowspan=1 colspan=1>60.8</td><td rowspan=1 colspan=1>59.0</td><td rowspan=1 colspan=1>63.5</td><td rowspan=1 colspan=1>65.0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>63.0</td></tr><tr><td rowspan=1 colspan=1>65.7</td><td rowspan=1 colspan=1>66.1</td><td rowspan=1 colspan=1>66.0</td><td rowspan=1 colspan=1>51.4</td><td rowspan=1 colspan=1>60.5</td><td rowspan=1 colspan=1>55.7</td><td rowspan=1 colspan=1>64.4</td><td rowspan=1 colspan=1>64.8</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>61.8</td></tr><tr><td rowspan=1 colspan=1>60.3</td><td rowspan=1 colspan=1>61.3</td><td rowspan=1 colspan=1>61.4</td><td rowspan=1 colspan=1>49.6</td><td rowspan=1 colspan=1>57.4</td><td rowspan=1 colspan=1>53.6</td><td rowspan=1 colspan=1>61.3</td><td rowspan=1 colspan=1>62.1</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>58.4</td></tr><tr><td rowspan=1 colspan=1>60.9</td><td rowspan=1 colspan=1>61.7</td><td rowspan=1 colspan=1>61.8</td><td rowspan=1 colspan=1>47.0</td><td rowspan=1 colspan=1>56.7</td><td rowspan=1 colspan=1>53.4</td><td rowspan=1 colspan=1>61.2</td><td rowspan=1 colspan=1>62.0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>58.1</td></tr><tr><td rowspan=1 colspan=1>60.9</td><td rowspan=1 colspan=1>61.3</td><td rowspan=1 colspan=1>61.5</td><td rowspan=1 colspan=1>47.2</td><td rowspan=1 colspan=1>56.6</td><td rowspan=1 colspan=1>53.3</td><td rowspan=1 colspan=1>61.4</td><td rowspan=1 colspan=1>62.1</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>58.0</td></tr><tr><td rowspan=1 colspan=1>60.2</td><td rowspan=1 colspan=1>61.3</td><td rowspan=1 colspan=1>61.1</td><td rowspan=1 colspan=1>47.1</td><td rowspan=1 colspan=1>56.8</td><td rowspan=1 colspan=1>53.3</td><td rowspan=1 colspan=1>60.9</td><td rowspan=1 colspan=1>62.0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>57.8</td></tr><tr><td rowspan=1 colspan=1>60.3</td><td rowspan=1 colspan=1>61.2</td><td rowspan=1 colspan=1>61.2</td><td rowspan=1 colspan=1>47.0</td><td rowspan=1 colspan=1>56.5</td><td rowspan=1 colspan=1>53.2</td><td rowspan=1 colspan=1>61.3</td><td rowspan=1 colspan=1>61.9</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>57.8</td></tr><tr><td rowspan=1 colspan=1>60.3</td><td rowspan=1 colspan=1>61.3</td><td rowspan=1 colspan=1>61.4</td><td rowspan=1 colspan=1>47.1</td><td rowspan=1 colspan=1>56.4</td><td rowspan=1 colspan=1>53.2</td><td rowspan=1 colspan=1>61.1</td><td rowspan=1 colspan=1>61.9</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>57.8</td></tr><tr><td rowspan=1 colspan=1>60.0</td><td rowspan=1 colspan=1>61.5</td><td rowspan=1 colspan=1>61.3</td><td rowspan=1 colspan=1>46.9</td><td rowspan=1 colspan=1>56.6</td><td rowspan=1 colspan=1>53.1</td><td rowspan=1 colspan=1>61.1</td><td rowspan=1 colspan=1>62.0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>57.8</td></tr><tr><td rowspan=1 colspan=1>60.4</td><td rowspan=1 colspan=1>61.4</td><td rowspan=1 colspan=1>61.2</td><td rowspan=1 colspan=1>46.7</td><td rowspan=1 colspan=1>56.3</td><td rowspan=1 colspan=1>53.2</td><td rowspan=1 colspan=1>60.7</td><td rowspan=1 colspan=1>62.1</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>57.8</td></tr><tr><td rowspan=1 colspan=1>60.3</td><td rowspan=1 colspan=1>61.3</td><td rowspan=1 colspan=1>61.2</td><td rowspan=1 colspan=1>46.8</td><td rowspan=1 colspan=1>56.4</td><td rowspan=1 colspan=1>53.3</td><td rowspan=1 colspan=1>61.2</td><td rowspan=1 colspan=1>61.9</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>57.8</td></tr><tr><td rowspan=2 colspan=1>MAC()</td><td rowspan=2 colspan=1>AIMt2</td><td rowspan=1 colspan=1>2</td><td rowspan=2 colspan=1>MACC()</td><td rowspan=2 colspan=1>AIMt2</td><td rowspan=2 colspan=1>A-2</td><td rowspan=2 colspan=1>AIMt2</td><td rowspan=2 colspan=2>AF-2</td><td rowspan=2 colspan=1>mean</td></tr><tr><td rowspan=1 colspan=1>A2</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1></td><td></td><td></td><td></td></tr></table>

Figure 20: Descriptor effects are consistent across cells. (a) G (diverging scale centered at 0) and (b) DFT calls at $\alpha = 0 . 3 0$ (fixed 0–100 scale) for every representation and single-family drop in every cell; the last column is the equal-cell mean. Rows below the rule are drops from the full descriptor, ordered by mean ∆G.

![](images/cb8f16a5a0128ee9020fd19d530873d58f9f18ce129435d8acb2058bc3371cb7.jpg)  
Figure 21: More fitting labels improve ranking; the ranking-loss weight matters only when large. (a) G against fitting-set size $( 2 . 5 \mathrm { - } 4 0 \%$ of population identities, nested subsets of the original fitting pool), one thin line per cell and the bold equal-cell mean. (b) G against the ranking-loss weight $\lambda \in \{ 0 , 0 . 1$ , 1 (frozen), 10} for the three-seed ensemble (solid) and seed 0 (dashed). Lines join cell values and are not uncertainty intervals.

## J What routing does to chemical observables

This section extends the relative-energy results of Section 3.2. These quantities are outside Proposition 1.

Definitions. Deployment DFT is the percentage of the test pool routed to DFT. Hybrid energies take the reference energy for routed configurations and the MLIP energy, with the frozen fitting-only alignment, otherwise; the all-MLIP endpoint uses raw predictions, since a composition-linear offset cancels within a molecule. For each test identity with at least two configurations, relative-energy MAE averages the absolute errors of all unordered energy differences, then averages identities equally. Selection success is the fraction of identities whose lowest hybrid-energy configuration lies within 1 kcal/mol (43.3641 meV) of the identity’s reference minimum. All 600 COMP6 and 2,780 QCML test identities qualify; MACE-OFF23 eligibility varies across reallocations and excludes singleton groups.

Findings. At matched deployment DFT (Fig. 22; procedure in Appendix F.1), ours lowers relative-energy MAE by 2.0 and 9.7 meV against the raw and supervised committees, better in 4 and 5 of eight cells; the supervised mean is dominated by QCML × MACE-OFF23(M) (−36.9 meV). Selection success rises by 0.7 and 1.5 pp, better in 5 and 6 of eight. The COMP6 curves separate clearly, while QCML and MACE-OFF23 largely overlap (Fig. 23). At α = 0.30, ours reduces relative-energy MAE from the all-MLIP endpoint by at least 36.4% in every cell, yet selection success falls below all-MLIP on the three COMP6 cells. Routing removes 73–99.7% of the full-pool mean energy error and 65–99.8% of the mean force error, but rare large force errors remain among retained configurations, up to 50.300 eV/Å on MACE-OFF23 × AceFF-2 (Fig. 24). Truncation at $B _ { \mathrm { c f g } }$ affects at most 0.159% of configurations but removes 97.116% of the raw score mass on QCML × MACE-OFF23(M); CRC controls only the capped loss.

(a) Lower relative-energy error at matched DFT  
![](images/37d85e31fcd24247f88462b13947600ebc81e2efece235be30796977d60025d3.jpg)  
Figure 22: At matched deployment DFT, ours lowers relative-energy error and raises selection success on average, but not in every cell. Per-cell difference, ours minus committee, against the raw (green) and supervised (blue) committees: (a) relative-energy MAE (lower is better); (b) selection success (higher is better). The right group is the equal-cell mean. Each value averages the difference of the 20-reallocation mean curves over the cell’s common attained deployment-DFT interval. Descriptive, not a new guarantee.

![](images/7b28bce27ba91f904d5f5df28aa2c3dcb4d39782ed136850fbae9be02e2f8f35.jpg)  
Figure 23: On COMP6, ours reaches a given relative-energy error and selection success with less deployment DFT than either committee; elsewhere the curves largely overlap. Per cell, relative-energy MAE (asinh axis, linear below 10 meV) and selection success against deployment DFT, with all-MLIP (square) and all-DFT (star) endpoints and a ring at ours, α = 0.30. Means over 20 reallocations at 5% DFT steps, each split at its nearest attainable point; bands ±1 SD. The saved sweep has no oracle. Success axes are zoomed per cell. 40

after routing, ours at α = 0.30 per-pool maximum, mean ± SD

(c) Rare large force errors are retained COMP6 QCML  
![](images/747124cd2e3f4c79f9a30e417f11de5f505e47737e3a3fd6f5e52d6980012dff.jpg)

(b) … and most force error  
![](images/a19453c73dc9c392afbe1dc1354813b710a54f27e015a9192389750e53cfe457.jpg)  
(d) Few configurations hit the cap

![](images/db7de210a1f8b47dca2b848df496509290dc706b3c0a164451a699a6146640da.jpg)

![](images/4d0325bcf3ee68af9aef977f0e828bab2faad5ed923be3dc1cefab3e402aa727.jpg)  
configurations truncated at $B _ { \mathrm { c f g } }$  
Figure 24: Routing at α = 0.30 removes most physical error, but rare large force errors remain. (a) Full-pool mean aligned energy error and (b) mean atom-force error before routing and after routing with ours; numbers give the relative reduction (%), triangles the retained 99th percentile. (c) Retained maximum-force error: 99th percentile, per-pool maximum (mean ± SD), and largest over 20 pools. (d) Share of the pool truncated at $B _ { \mathrm { c f g } } ,$ as configurations and as raw score mass. Log axes. Uncapped tails are outside the guarantee.

## J.1 Signed-residual cancellation

The MLIP residuals of one molecule’s configurations often share sign, and same-sign residuals cancel in an energy difference. Routing one configuration to DFT sets its residual to zero and removes that cancellation. Before routing, the identity-balanced correlation of paired residuals is 0.92–0.96 on COMP6 and $\mathbf { M A C E - O F F } 2 3 \times \mathbf { A c e F F - } 2$ , and routing with ours at $\alpha = 0 . 3 0$ lowers it to 0.55–0.60 (Fig. 25). On QCML the all-MLIP correlation is near zero and rises after routing because the catastrophic residuals are removed. As a result, a molecule’s relative-energy loss need not fall monotonically as DFT is added: 45–95% of eligible identities have a non-monotone routing path for relative-energy MAE and 1–46% for selection failure (Fig. 26). These aggregate diagnostics are consistent with lost cancellation but do not identify the cause of each selection failure.

![](images/5e8594270f9935518677be8e5ba44032469d560c0088f2bba611ccedf5234e3c.jpg)  
Figure 25: Signed residuals of one molecule’s configurations are strongly correlated on COMP6 and MACE-OFF23 × AceFF-2, so energy differences benefit from cancellation that partial routing removes. Weighted density of signed aligned residual pairs $( \Delta E _ { i } , \Delta E _ { j } )$ within an identity (up to 64 pairs per eligible identity, both orientations), in asinh $( \dot { \Delta } E / 0 . \dot { 1 } \mathrm { e V } )$ coordinates; the dashed diagonal is exact cancellation. Annotations give the paired Pearson r before routing → after ours at $\alpha = 0 . 3 0$ , averaged over 20 reallocations. One shared log color scale of pair fraction per bin.

![](images/a1242eacb3baed630411fad2d59702434f5d0b3a62a14301383cd0e6fe73438d.jpg)

![](images/4388ff9886aa1c63e2a1cb060b91f1b92d3df5d929b1d6c21940e5e7331ac425.jpg)  
Figure 26: Partial routing can raise a molecule’s relative-energy error before full DFT drives it to zero, and such path are common. (a) One saved COMP6 × AceFF-2 identity: Loss A (identity MAE of relative energies) and its running-maximum envelope against the number of configurations routed; an illustration, not a prevalence estimate. (b) Per cell, the percentage of identities whose routing path is non-monotone for Loss A and for Loss B (selection failure).

## K Certifying other losses directly

Remark 3 allows any bounded monotone loss in place of $\ell ,$ so downstream objectives can be certified directly instead of through the proxy. We consider two identity-level losses on identities with at least two configurations. Loss A is the relative-energy MAE of the identity, capped at the largest fitting-identity path loss, with budget $\epsilon _ { A }$ a multiple of each cell’s fitting all-MLIP mean. Loss B is the indicator that the selected lowest-energy configuration lies more than 1 kcal/mol above the true minimum, with a probability budget $\epsilon _ { B }$ . Because both can be non-monotone in the cutoff (Appendix J.1), we compare M, CRC on the running-maximum monotone envelope of the loss; W-max and W-mean, CRC with whole-identity routing by the maximum or mean configuration score; L, Learn-then-Test on the raw loss over a fixed fitting-set grid with $\delta = 0 . 1 0 ; P .$ the original proxy rule at $\alpha = 0 . 3 0$ , which carries no downstream certificate; and N, a naive empirical cutoff with no certificate. M and W bound the expectation, whereas L holds with probability at least 0.90 over the calibration draw, so their costs are not directly comparable. DFT percentages here count eligible non-singleton configurations.

Findings. Strict physical-unit losses, certified with the same score and CRC rule, cost almost all savings (Fig. 27): at the smallest budgets the equal-cell mean DFT is 55.0% for aligned energy (0.05 kcal/mol/atom), 85.3% for raw energy, and 94.3% for maximum force (0.005 eV/Å), and a strict binary loss needs 97.5–99.9%. The proxy guarantee does not transfer downstream: P exceeds the Loss-A budget (0.3× the fitting mean) in five of eight cells, with ratios 1.01–2.14, and the Loss-B budget (0.05) in one (Fig. 28c). Direct certification costs depend on the objective (Fig. 28a, b): at the main Loss-A budget, M, W-max, and L route 86.1%, 79.3%, and 96.7%, while at $\epsilon _ { B } = 0 . 0 5$ they route 61.0%, 8.4%, and 21.8%; W-max is cheap partly because all-MLIP already meets the selection budget in several cells. Expectationlevel methods can exceed the budget in individual repeats (Fig. 28d, e). Relative Loss-A budgets correspond to different absolute accuracies across cells. In particular, the saved QCML × MACE-OFF23(M) fitting mean is 14,761 eV, so its main budget is 4,428.5 eV, and on all three QCML Loss-A panels every certified method routes 100% (Fig. 29). We report these values as saved; the export does not establish the cause of the extreme scale.

(a) Raw energy: costly  
![](images/e047e99a451f6805a946c9d2b2f1fadb47a614439c831fc2a6b45f0be5a94176.jpg)

(b) Aligned energy: cheaper  
![](images/ea215f32244931771bb56536ac2ad6a12d4248860876732828a27f6bf242348e.jpg)  
(d) Strict binary physical loss: almost all DFT

(c) Max force: costly  
![](images/01ccc67beacc666e1741652fbec095a71b70f13ae759bb12aa78488684c7f0fa.jpg)

![](images/2d17fe7da91a1d3ba85cca253a93f69db56a2e2417ef8a1e4ce307aaf8e2efbe.jpg)  
Figure 27: Strict physical-unit budgets cost almost all of the DFT savings. (a–c) DFT calls against an absolute budget for raw and aligned energy (kcal/mol/atom) and maximum force error $( \mathrm { e V } / \mathring \mathrm { A } )$ , with the same structural score and CRC rule; thin lines are cells, the bold line the equal-cell mean of 20-reallocation means. (d) A strict binary loss (aligned $| \Delta E | > 1$ kcal/mol or maximum force $> 0 . 1 0 \mathrm { e V } / \mathring { \mathrm { A } } ;$ budget 0.10 violations per identity) at each cell’s calibration size. Thresholds are examples, not universal tolerances.

(a) Loss A: certification is costly  
![](images/3607a43f5ea5e9c57ab102edc7a359a5a6ab02988e0eead8f244a79a592b17e1.jpg)

(b) Loss B: whole-identity is cheap  
![](images/7b83b95b72d4b37a68b012e8e075afafacaa9bf6175c1fd719366bce58e8b7a7.jpg)

(c) Proxy P can miss the budget  
![](images/f95943a01bf40c38ded4f7bcd89c49125df5008f478c5513ddc3973d153ef00a.jpg)  
(e) Loss B at $\varepsilon _ { B } = 0 . 0 5 \mathrm { : }$ certified ≲ 1

(d) Loss A at $\varepsilon _ { A } = 0 . 3 0 \times$ mean: certified ≲ 1  
![](images/de22bf805dffd426b297c43c3e9e2c71c400b172ca74a91d235b308197fcbb41.jpg)

![](images/89f9d96f8e9b77cb5315205b8a7565ff52f2af6c1ab8398133ebb847e1d38668.jpg)  
W-max (whole identity, CRC) L (Learn-then-Test) P (proxy CRC, α = 0.30) P: Loss B (c)  
Figure 28: The proxy guarantee does not carry over to downstream losses, and direct control has its own cost. (a, b) Eligible DFT against the budget for Loss A and Loss B under M, W-max, and L; the dashed line is the uncertified proxy rule P at α = 0.30. Equal-cell means of 20-repeat means. (c) Realized risk/budget of P per cell (log axis); the tiny QCML × MACE-OFF23(M) Loss-A ratio reflects its extreme fitting anchor. (d, e) Per-cell realized risk/budget of every method at the main budgets, mean ± SD over 20 repeats. L is high-probability; M and W are expectation-level.

Loss A — within-identity relative-energy MAE (budget relative to fitting all-MLIP mean)  
![](images/66b02790bc16956dd31d230ef1260a34d68002398cf69b36fdce5a6861b0b377.jpg)

![](images/3d671e90262d3da6c1eaa8b2e4e5d7e644fd7f5ad5dd5020488d0247b2ca1118.jpg)

Loss B — conformer-selection failure > 1 kcal/mol (probability budget)  
![](images/bcccd9c271b7101a67389e0c8aa9c55ea7cd414f65d7aeb67bf2fdc2c17e5fed.jpg)

![](images/3d5b75341add10b18a63e4e176e418cbf22eaf00e6390f93b67b9a245e89d2d2.jpg)  
Figure 29: Downstream certification cost depends strongly on the cell and the objective. Eligible DFT against the budget in every cell, Loss A (top two rows) and Loss B (bottom two), for M, W-max, W-mean, L, and N (mean ±1 SD over 20 repeats); P is a horizontal reference. ‡ Data issue, plotted without alteration: the saved QCML × MACE-OFF23(M) Loss-A fitting mean <sub>is</sub> 14,761 <sub>eV (main budget</sub> 4,428.5 <sub>eV), and on all three QCML Loss-A panels every certified method routes</sub> 100%<sub>. Budgets are</sub>46 relative to fitting anchors, not a common absolute accuracy.

## L Label cost and transfer

The reported savings are per reused pool and exclude the labels of ${ \mathcal { D } } _ { \mathrm { f i t } }$ and $\mathcal { D } _ { \mathrm { c a l i b } }$ . Charging them up front, and counting each cell in full-pool equivalents, the fitted selector breaks even after $P = 2$ comparable reused pools with 2.5% or 10% of identities for fitting and after $P = 4$ with 40% (Fig. 30); for a single pool it costs 1.17–2.07 full-pool equivalents, more than labeling the pool outright. Reuse assumes comparable pools, and the predictor must be fitted per population: a frozen structural score moved to another population keeps mean G = 0.17 (range −0.09 to 0.36, positive in 12 of 14 directed transfers) after target recalibration, against 0.80 in domain, and the QCML predictor anchored on COMP6 stays at $G \leq 0 . 1 5 ( \mathrm { F i g . } 3 1 )$

![](images/8cf7dca0052e6220fcb114fcb7fd7226139da25baef19d36f6229e33bf8be29a.jpg)  
Figure 30: Up-front labels pay back after 2–4 reused pools. Total DFT in full-pool equivalents (fitting and calibration labels up front, plus routed deployment labels) against the number P of comparable reused pools, for fitting at 2.5%, 10%, and 40% of identities and for all-DFT $( y = P )$ . Equal-cell means, each cell normalized by one deployment pool; rings mark the first saved integer $P$ at or below all-DFT. Cost-model projections in configuration counts, not measured wall time.

![](images/511c511cdd8479bc168c87f366cc232cdcc127aeb2b1e1f184e1e7b4630928b3.jpg)

(b)  
![](images/0e31f8161388d8bd144d7d31992170d1fad724e121ae4217b3d8f291a2258e57.jpg)  
(c)

![](images/eae0ec262a70e2d5365738efc3c3dac680458bcb929f6f9102514463b331900f.jpg)

![](images/db6ead031914c12869d08ddb7d5859b4decfca043d5b2858a58769a4ff61fe25.jpg)

![](images/5253784f2eba493bc24ffcd6dd5ea926716137f033d72fe895c156e97d88e88e.jpg)  
Figure 31: A frozen structural score transfers poorly across populations. (a–c) G of the frozen score from each source population (rows) on each target population (columns), per target MLIP, after target recalibration; boxed diagonals are in-domain, NA marks non-cells. (d) Transfer G against the target’s in-domain G for all 14 directed transfers (dashed: equality). (e) Anchored QCML → COMP6: G against the number of target-anchor identities for the target-trained (solid) and frozen QCML (dashed) predictors. Target anchors and calibration labels are charged, so this is not zero-label deployment.

## M Limitations

Scope of the guarantee. As is standard for conformal risk control, Proposition 1 is a marginal, expectation-level bound that holds within the dataset–MLIP cell used for fitting; a new population or target MLIP requires refitting and recalibration (Appendix L). The bound applies to the capped loss, and $B _ { \mathrm { c f g } }$ can be set to an application tolerance. Where rare chemical families or single calibration draws must be protected, high-probability or group-conditional variants apply at additional DFT cost (Appendix K).

Cost accounting. Reported savings are per deployment pool. The one-time fitting and calibration labels are amortized after two to four comparable pools (Appendix L), so the method targets repeated screening of a population.

Reference mismatch. The residuals include differences between electronic-structure levels, which is the setting the method is designed for. When references match, committee disagreement is competitive, and the hybrid score combines both signals (Appendix G).

Downstream objectives. The proxy loss does not certify relative energies or conformer selection. Because the framework accepts any bounded monotone loss, these objectives can be certified directly (Appendix K).

Domain. The descriptor covers isolated molecules built from ten elements; periodic systems require extending its distance definitions.

## N Broader Impacts

Routing lowers the DFT cost of repeated molecular screening. It also replaces empirically chosen uncertainty thresholds with a stated, auditable loss and budget, which may help pretrained MLIPs enter workflows that require documented error control. Two risks follow. First, cheaper screening can enlarge campaigns, so total compute need not fall. Second, if routed labels are used to retrain the MLIP, regions the selector ranks low may remain unlabeled, and retraining requires refitting the predictor and recalibrating the threshold. The largest practical risk is overreading the guarantee as assurance of stable trajectories, correct conformer energies, or the absence of rare large errors, none of which it covers. Reports of routed results should state the controlled loss, its caps, and the calibration population.