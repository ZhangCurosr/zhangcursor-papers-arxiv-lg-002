# SHORE: Stability-Hybrid On-Policy Reinforcement Learning for Online Bin Packing

Chang Shi<sup>1</sup> and Ann Majewicz Fey<sup>1</sup>

Abstract— Online 3D bin packing with a processing buffer requires choosing which box to pack, its orientation, and its placement while keeping the stack stable under unknown future arrivals. A collapse ends the episode and forfeits the remaining volume. Existing methods use geometric rules or hard stability masks, limiting how risk can influence the joint choice of box, orientation, and placement. We propose Stability-Hybrid On-Policy Reinforcement Learning (SHORE), which combines a learned stability probability of a placement with an onpolicy estimate of its return conditional on stability. A 3D U-Net learns stability from executed outcomes and simulated alternatives, revealing outcomes of actions underrepresented in early rollouts. A discrete on-policy actor–critic uses stability predictions in both its action values and policy prior, with gates that limit the effect of an untrained or changing model. In deployment, SHORE does not require additional search or simulation overhead. In Isaac Gym, SHORE reaches 73% space utilization by 0.4M environment steps, while the same learner without stability modeling reaches that level at about 1M. On the expanded 60-box set, SHORE achieves 0.734 space utilization versus 0.588 for the GOPT state-of-the-art baseline, with a higher placement-failure frequency (1.03% vs. 0.63%), trading more risk for greater packed volume. The results show faster learning and higher utilization on the more demanding realistic box arrival set. Ablations suggest that simulated alternatives provide useful early supervision, while executed outcomes alone sustain later performance. We also demonstrate the method on a real robot to verify the simplicity of integrating the method to real bin packing pipelines.

## I. INTRODUCTION

Online 3D bin packing requires selecting and placing boxes as they arrive, without knowing the complete arrival sequence. In warehouse settings, a small processing buffer allows for limited reordering prior to placement (Fig. 1). The objective is to pack densely while keeping the stack stable, which depends on the box dimensions, physics, and the order, orientation, and location of placement.

Physical stability depends on more than the visible top surface. Supporting material properties and the arrangement of multiple supporting boxes can affect a placement’s outcome. One box collapse can leave the bin disorganized and hurt the long-term packing compactness. Therefore, there have been works trying to integrate physics-aware stability information into packing decision making.

Reinforcement learning (RL) packers usually handle stability with an action mask. Most masks are computed using geometric or physics-based rules on the height map or the stacking structure [1], [2]. Instead, [3] and [4] learn the stability mask through simulation outcomes together with a

Proximal Policy Optimization (PPO) policy [5], but apply the masks as hard filters on the placement of a box the policy has already chosen. Stability therefore cannot influence which box to pack, leading to a policy that is not jointly optimal. Moreover, a placement is either allowed or forbidden, with no trade-off between its risk and the volume it adds. This motivates designing a more advanced joint policy in which stability awareness can directly affect selection among boxes, orientations, and placements in the long run.

We propose SHORE, which separates the prediction of the outcome of the current placement from the value of the subsequent packing, but intelligently fuses the stability information back to guide the long-term joint policy learning. A 3D U-Net reads bin occupancy and the oriented box properties to estimate whether the whole stack satisfies the stability test after the placement. A separate actionconditioned critic learns the return if the current placement holds, including its immediate reward and subsequent packing, from on-policy rollouts. The stability estimate modulates this value and provides a soft prior to the joint actions of the actor (Fig. 2). In this on-policy actor–critic method, the critic scores every valid (box, orientation, placement) triple, and the actor follows the exact expected gradient over all of them, so the stability estimate shapes which box is packed and how it is turned, not only where it goes. At inference, SHORE chooses each placement with one pass of the actor and of the stability network.

In short, our contributions are:

• A physics-aware RL strategy that uses learned placement-stability probabilities as continuous weights on action values for policy improvement over joint box– orientation–placement actions, so that the packing agent can trade the risk of an unstable placement against the volume it adds.

• A highly parallel, Isaac Gym-based bin packing environment that enables fast, large-scale on-policy RL training.

• An empirical study of the efficiency of this method on a realistic box distribution, together with an initial realrobot deployment test.

## II. RELATED WORK

Online 3D Bin Packing. This problem setting involves making packing decisions without knowledge of incoming items, leading to unique challenges in achieving optimality and efficiency. The early efforts on this problem are mostly studies about developing adaptive heuristics and algorithms that can handle the sequential nature of item arrivals. For example, [6] and [7] utilize deepest-bottom-left (DBL) heuristics, while [8] introduces Heightmap-Minimization heuristics.

![](images/66b7ecb7b955a91513c565b9893fe2afccc3e40a22b221a05adc9979fcb35667.jpg)  
Fig. 1: Illustration of the online bin packing setting with a finite processing buffer. The packing agent selects a box from the buffer, its orientation, and its placement while future arrivals remain unknown.

Recent work in this area has moved beyond classical heuristics to incorporate deep learning, physics constraints, and more generalizable models. [9] use DRL to solve the specific setting with an arbitrary number of bins and any bin size, and show RL + search heuristics outperform traditional online heuristics across a range of instances. [10] tackle the problem with constrained DRL, framing the problem as a constrained Markov decision process under limited lookahead and enforcing physical stability, and show strong improvements over previous heuristics. [11] designs a complete robot pipeline to narrow the gap of heuristic DRL framework between simulation and real world. More recently, [2] introduces a DRL model that not only places items but also generalizes to bins of varying sizes via a transformer architecture that fuses item and bin subspace features. [12] integrate a physics heuristic with object rearrangement and stable placement into a DRL framework to reduce wasted space and improve utilization. However, the design of good heuristics remains a challenge, which leads to the emergence of learning-based action masking models.

Action masking models. Action masking has emerged as a practical strategy to tackle the enormous, state-dependent action space in the online 3D bin packing setting. [13] propose a feasibility-prediction module that learns a binary “feasibility mask” over placement choices and uses that mask to modulate actor probabilities during training, which substantially speeds learning and enforces collision/stability constraints. Building on the same idea, [14] allows custom action-mask definitions including reorientation and empty maximal spaces (EMSs), and compares masking strategies with search-based planners. [1] instead compute the feasibility mask for the placement actions based on the stacking tree analysis. [4] incorporate physics properties into the mask calculation to improve sample efficiency and reduce unsafe actions during exploration. [2] uses pairwise actionmasking between EMSs and orientations to guarantee that the policy only samples pairs, further improving generalization in the online setting. [3] learn the mask online from physics simulations, jointly with the policy. All of these masks are hard filters on placements. SHORE instead keeps a geometric mask only for fit and support, and uses a learned stability probability inside the action value, where it can be traded against packed volume and affects which box is packed.

## III. PRELIMINARIES

We consider online 3D single-bin packing with a processing buffer of size N. Only the first N unplaced boxes are visible, and placing one admits the next arrival. The full simulator state $\xi _ { t }$ includes the physical configuration and the entire arrival queue. It evolves as an MDP $\mathcal { M } =$ $( \mathcal { X } , \mathcal { A } , P , \bar { r } , \gamma )$ , where $\mathcal { X }$ is the full state space, A the joint action space, $P$ the transition kernel, and $\bar { r } ( \boldsymbol { \xi } , a ) = \mathbb { E } [ r _ { t }$ $\xi _ { t } = \xi , a _ { t } = a ]$ the expected reward. We use $\gamma = 1$ and maximize $J ( \theta ) \dot { = } \mathbb { E } _ { \pi } [ \sum _ { t } \gamma ^ { t } r _ { t } ]$ , the expected packed-volume fraction. The policy receives only the observation $s _ { t }$ defined below, so its decision problem is partially observable. Arrival order is exogenous; the agent controls only the reordering allowed by the buffer.

## A. State and Observation

The policy observation is $s _ { t } = ( C _ { t } , d _ { t } )$ : the voxelized bin configuration and the properties of the visible boxes. The unseen arrival queue and unresolved contact details belong to $\xi _ { t } .$ , not $s _ { t }$ . We use s as shorthand for this observation in the policy, critic and stability model. The bin has physical dimensions $( L _ { p } , W _ { p } , H _ { p } )$ and is discretized into $( l _ { p } , w _ { p } , h _ { p } )$ cells. $C _ { t }$ records voxel densities, and its height map $\eta _ { t } ~ \in$ $\{ 0 , \ldots , h _ { p } \} ^ { l _ { p } \times w _ { p } }$ gives the top occupied cell of each column. The buffer $d _ { t }$ contains each visible box’s dimensions and density. Following GOPT [2], these observations are encoded as two token sets (Fig. 2):

• Item tokens. There is one token for each pair $u = ( b , j )$ of a visible box b and an orientation $j \in \mathcal { O }$ , where O contains the six axis-aligned orientations. Token $\mathbf { x } _ { u }$ contains the oriented extents, expressed in voxel units and normalized by $( l _ { p } , w _ { p } , h _ { p } )$ , the similarly normalized unrotated extents, a heavy flag derived from density, a presence flag and a selectability flag. Duplicate tokens remain in the observation but are not selectable.

• Placement tokens. There are at most K empty maximal spaces (EMSs), each extending to the bin’s maximum height. For EMS k, let $( x _ { 1 } , y _ { 1 } , z _ { 1 } )$ and $( x _ { 2 } , y _ { 2 } , h _ { p } )$ be its corners in voxel coordinates. Its token $\mathbf { y } _ { k }$ contains these first five coordinates normalized by the corresponding grid dimensions, followed by support fractions $\rho _ { k } ^ { \mathrm { f i a t } }$ and $\overline { { \rho _ { k } ^ { \mathrm { r i g i d } } } }$ . These measure the part of its base supported at height $z _ { 1 }$ , and the part supported there by the bin floor or a heavy box, respectively. They are the only density-layout information provided to the actor and critic. EMSs are sorted by $( z _ { 1 } , y _ { 1 } , x _ { 1 } )$ and truncated to K; unused slots are zero-padded and cannot be selected.

## B. Action

An action $\boldsymbol { a } = \left( \boldsymbol { u } , \boldsymbol { k } \right)$ pairs an item token $u = ( b , j )$ with an EMS k. Box b is rotated to orientation j and placed with its minimum corner at $( x _ { 1 } , y _ { 1 } )$ of EMS k, resting on the highest column under its footprint. Placements are therefore limited to EMS corners rather than every grid cell, and box order, orientation and placement are chosen jointly at every step. A rule-based mask $m ( s , a ) \in \{ 0 , 1 \}$ admits a pair if and only if u is selectable, the oriented box fits inside the EMS (and hence inside the bin and below $H _ { p } ) _ { \ast }$ , and the placement passes GOPT’s geometric support test [2]. The test requires that more than half of the footprint rests at the placement height, or that the footprint center lies within the convex hull of the supported cells. The valid action set is $A ( s ) =$ $\{ a : m ( s , a ) = 1 \}$ , with $\lvert \mathcal { A } ( s ) \rvert \le 6 N K$

## C. Reward Design

The goal is to maximize space utilization while every placement stays stable. The reward at each step is the placed box’s share of the bin volume:

$$
r _ { t } = \chi _ { t } \frac { V _ { b _ { t } } } { V _ { \mathrm { b i n } } } ,\tag{1}
$$

where $b _ { t }$ is the selected box, $V _ { b _ { t } }$ its volume, $V _ { \mathrm { b i n } } \quad =$ $L _ { p } W _ { p } H _ { p } ,$ , and $\chi _ { t } \in \{ 0 , 1 \}$ is the observed stability outcome: it is one exactly when Eq. (2) holds. An unstable placement earns zero and ends the episode (Sec. III-E). With $\gamma = 1$ the undiscounted reward sum is the stably packed volume fraction at termination, bounded by the total sequence volume (0.691 for the base set and 0.866 for the expanded set). The reward of a successful placement depends only on its box volume; order, orientation and placement affect subsequent rewards through future fit and stability. Learners use the scaled reward $c _ { r } r _ { t } .$ , with $c _ { r } = 1 0$

Near the end of an episode the buffer can hold fewer than N boxes. Empty slots are marked absent and masked, so the policy cannot select them. No invalid-selection penalty is needed, unlike the penalty used in [3].

## D. Stability and Feasibility Criteria

We make the two criteria used throughout the paper explicit. Stability (evaluated after placement). A placement is executed by releasing the box 2 mm above its target position in the chosen orientation and simulating the scene forward for $\tau = 8 0 \mathrm { m s }$ . Let $\mathbf { p } _ { i } ^ { \mathrm { t g t } } , \mathbf { p } _ { i } ^ { \mathrm { s e t t l e d } } \in \mathbb { R } ^ { 3 }$ denote box i’s target and settled positions, and let $\mathcal { T } _ { t }$ contain all boxes in the stack, including the new box. Stability requires

$$
\operatorname* { m a x } _ { i \in \mathscr { T } _ { t } } \left\| \mathbf { p } _ { i } ^ { \mathrm { s e t t l e d } } - \mathbf { p } _ { i } ^ { \mathrm { t g t } } \right\| _ { 2 } \leq \epsilon _ { s } .\tag{2}
$$

The threshold $\epsilon _ { s }$ is a positional displacement tolerance; the test checks movement of both the new box and previously placed boxes. Feasibility (evaluated before placement). Feasibility is the geometric mask $m ( s , a )$ of the action definition. Apart from excluding duplicate tokens, it encodes only fit and GOPT’s support test, both computed from the height map, so it does not predict Eq. (2), which depends on box densities and on the response of the whole stack. Stability is learned instead: SHORE trains a separate model of the probability that a placement satisfies Eq. (2) (Sec. IV-C) and uses it in both the critic and the policy.

## E. Episode End

An episode ends in one of three ways: (i) completion, when every box in the sequence has been placed; (ii) infeasibility, when the method’s feasibility filter admits no placement for the next decision; and (iii) instability, when an executed placement violates Eq. (2). For SHORE, GOPT+ and GOPT, the feasibility filter is purely geometric: it is the rule-based mask $m ( s , a )$ of Sec. III-B, which admits a placement only if the oriented box fits inside the EMS and passes GOPT’s support test, both computed from the height map. SHORE and GOPT+ apply it to every visible box in every orientation, so an infeasible ending means that no visible box fits anywhere with enough support; GOPT applies it only to the buffer head in its stored orientation. For OL Mask, the filter of the selected box also includes the learned stability mask, so its episodes can also end when the mask rejects every placement as unstable. An infeasible ending can therefore mean that no box fits in the remaining space, or that the filter rejects placements that would in fact be stable, and the evaluation does not separate the two.

## IV. METHOD

SHORE trains an actor–critic over the joint action $a =$ $( u , k )$ with a discretized version of REPPO [15] (Secs. IV-A– IV-B) and adds a learned estimate $S ( s , a )$ of the probability that placement a holds, i.e., satisfies Eq. (2) (Secs. IV-C–IV-D). S enters learning in two places: it weights the critic’s value (critic weighting, Sec. IV-E) and it acts as a prior on the actor (actor prior, Sec. IV-F). Two safeguards keep an untrained or changing S from disturbing the learner (Sec. IV-G). Fig. 2 gives an overview, and Algorithm 1 lists one training iteration.

The stability model addresses a specific weakness of the plain learner. A collapse pays 0 and ends the episode, so a plain critic must learn, for every action, the product of two different quantities: the chance that the placement holds, which depends on the densities and support under the footprint, and the value of the packing that follows. The token observation shows EMS rectangles and two support fractions but not the interior of the stack, and collapses are rare, so the critic learns this product slowly and only for the actions the policy executes.

![](images/70e73e2c5a8b2d746727a935ae5d3b88a0a8ae688bf5e0625151a2738532ebb4.jpg)  
Fig. 2: SHORE framework. The bin configuration $C _ { t }$ and the buffer $d _ { t }$ are encoded as EMS tokens and (box, orientation) item tokens, and the rule-based mask $m$ keeps the geometrically valid pairs. The actor, a Packing Transformer, scores every valid pair and samples the action $a = \left( u , k \right)$ , which Isaac Gym executes and adjudicates by Eq. (2). A second Packing Transformer, the critic, values every pair in one pass, and a discretized REPPO learner updates both from on-policy rollouts (Sec. IV-B).

## A. Policy and Critic Networks

The actor and the critic each have their own Packing Transformer backbone [2]. MLPs embed the item tokens $\mathbf { x } _ { u }$ and EMS tokens $\mathbf { y } _ { k }$ into $\mathbf { h } _ { u } , \mathbf { e } _ { k } \ \in \ \mathbb { R } ^ { 1 2 8 }$ , and three encoder blocks apply self-attention within each token type and cross-attention between items and EMSs. Actor and critic parameters are denoted θ and ϕ. With learned affine projections followed by LeakyReLU, denoted $F _ { 1 } , F _ { 2 }$ , the base policy before adding stability information is

$$
\begin{array} { r l } & { \ell _ { \theta } ( s , u , k ) = \langle F _ { 1 } ( \mathbf { h } _ { u } ) , F _ { 2 } ( \mathbf { e } _ { k } ) \rangle , } \\ & { \pi _ { \theta } ^ { \mathrm { b a s e } } ( a \mid s ) = \frac { m ( s , a ) e ^ { \ell _ { \theta } ( s , a ) } } { \sum _ { a ^ { \prime } } m \left( s , a ^ { \prime } \right) e ^ { \ell _ { \theta } ( s , a ^ { \prime } ) } } . } \end{array}\tag{3}
$$

The critic forms a pair feature $\mathbf { f } _ { \phi } ( s , u , k ) = \mathrm { M L P } _ { \phi } ( P _ { h } \mathbf { h } _ { u } +$ $P _ { e } { \bf e } _ { k } + P _ { s } [ \bar { \bf h } ; \bar { \bf e } ] )$ , where $P _ { h } , P _ { e } , P _ { s }$ are learned projections and the bars denote token means. Its head predicts a categorical distribution $p _ { \phi } ( i { \bf \theta } \mid s , a )$ on $n _ { b } ~ = ~ 1 5 1$ atoms $z _ { i }$ spanning [−40, 40], with mean $\begin{array} { r } { Q _ { \phi } ( s , a ) = \sum _ { i } z _ { i } p _ { \phi } ( i \mid s , a ) } \end{array}$

The HL-Gauss encoding $\operatorname { H L G } ( g )$ maps a scalar target $g$ to histogram-bin probabilities from $\mathcal { N } ( g , \sigma _ { \mathrm { H L } } ^ { 2 } )$ , after clipping g to the atom support and normalizing the retained bin masses, with $\sigma _ { \mathrm { H L } }$ equal to 0.75 bin widths [16]. One forward pass scores all valid actions. Policy distributions are defined only for nonempty $\boldsymbol { \mathcal { A } } ( \boldsymbol { s } )$ ; terminal states have zero continuation value.

## B. REPPO-D Backbone

We use the discrete REPPO formulation of [15, Appendix C], denoted REPPO-D here. Its action-value expectation, entropy and KL divergence are evaluated exactly over the finite action set. In our setting, the sums cover the geometrically valid joint actions $\boldsymbol { \mathcal { A } } ( \boldsymbol { s } )$ , with the networks of Sec. IV-A scoring every pair in one pass. This discrete update is inherited from REPPO, rather than a contribution of SHORE.

We retain REPPO’s on-policy soft λ-returns, HL-Gauss critic loss with latent self-prediction, and KL-gated actor update with adaptive entropy temperature α and KL multiplier $\beta ;$ their definitions are given in [15, Secs. 3.1– 3.3 and Appendix C]. The rollout-generating distribution is denoted $\pi _ { \mathrm { o l d } }$ . It remains fixed as the reference throughout the minibatch updates. Return targets use the scaled reward $c _ { r } r _ { t }$ zero continuation at terminal states, and value bootstrapping at rollout boundaries.

SHORE changes the action value, the actor’s prior and entropy regularizer, and the entropy target, as detailed below.

Its soft returns use the modified value in Eq. (11), and its actor uses Eq. (10) inside the inherited KL gate. Setting $w _ { Q } = w _ { \pi } = 0$ recovers our masked REPPO-D backbone, called SHORE − actor prior − critic weighting in the experiments. Training settings are listed in Table VII.

## C. Stability Model

$S _ { \psi } ( s , a ) \in [ 0 , 1 ]$ , with parameters ψ, estimates the probability of satisfying Eq. (2). The voxel grid hides sub-cell offsets, velocities and contact states, so identical observed grids can have different placement outcomes. A probability supports expected-value estimation under this partial observability.

The network takes the voxel grid and one oriented box. Its input has six channels on the $l _ { p } \times w _ { p } \times h _ { p }$ grid: occupancy, the voxel densities of $C _ { t }$ , and the box’s length, width, height and density, each broadcast as a constant channel. Extents are normalized by the grid dimensions, and densities by the maximum box density. A 3D U-Net [17] with the architecture of the OL Mask network [3] (8 base feature maps, 3 levels, group normalization) ends in a convolution that collapses the height axis, giving 16 feature channels on the $l _ { p } \times w _ { p }$ grid. Five heads, each a $3 \times 3$ convolution, a ReLU and a $1 \times 1$ convolution, map these features to logit maps $g _ { \psi } ^ { ( n ) }$ $n = 1 , \ldots , 5$ , with one logit per placement corner. For action $\boldsymbol { a } = \left( \boldsymbol { u } , \boldsymbol { k } \right)$ , each head reads the map computed for item token u at the corner $( x _ { 1 } , y _ { 1 } )$ of EMS k:

$$
\begin{array} { r l } & { p _ { \psi , n } \big ( s , a ) = \mathrm { s i g m o i d } \big ( g _ { \psi } ^ { ( n ) } ( s , a ) \big ) , } \\ & { \quad S _ { \psi } \big ( s , a \big ) = \frac { 1 } { 5 } \sum _ { n = 1 } ^ { 5 } p _ { \psi , n } \big ( s , a \big ) , } \\ & { \quad \nu _ { \psi } \big ( s , a \big ) = h _ { \mathrm { b i n } } \big ( S _ { \psi } \big ( s , a \big ) \big ) } \\ & { \quad \quad - \frac { 1 } { 5 } \sum _ { n = 1 } ^ { 5 } h _ { \mathrm { b i n } } \big ( p _ { \psi , n } \big ( s , a \big ) \big ) , } \end{array}\tag{4}
$$

where $h _ { \mathrm { b i n } } ( p ) = - p \log p - ( 1 - p ) \log ( 1 - p )$ is binary entropy; all logarithms are natural. Entropy evaluation clips probabilities to $[ 1 0 ^ { - 6 } , 1 ~ - ~ 1 0 ^ { - 6 } ]$ for numerical stability. A learning copy has parameters ψ and a separate acting copy has parameters ψ<sup>¯</sup>. Below, $S ~ = ~ S _ { \bar { \psi } }$ and $\begin{array} { r } { \nu \ = \ \nu _ { \bar { \psi } } } \end{array}$ denote the acting predictions, held fixed during actor–critic updates (Sec. IV-G). For numerical stability, the policy and its regularizer use $\widetilde { S } ( s , a ) = \operatorname* { m a x } \{ S ( s , a ) , \epsilon _ { \log } \}$ , with $\epsilon _ { \mathrm { l o g } } =$ $1 0 ^ { - 6 } ;$ critic weighting uses the unmodified S. The head disagreement ν (BALD [18]) estimates epistemic uncertainty: it is high when the heads disagree, whereas uniformly uncertain heads can have low disagreement. It is used only to choose which placements to label (Sec. IV-D). One network pass covers every EMS of an item token, and tokens with equal extents and density share a pass, so a state costs at most $6 N = 3 0$ passes. Invalid actions get $S = 1$ and $\nu = 0$

A separate convolutional network can use local density and support information absent from the actor–critic tokens. It also isolates supervised updates on buffered labels from the critic’s on-policy representation.

## D. Stability Labels

A label is the binary outcome of a placement, $y = 1$ if it held and $y = 0$ if it collapsed, stored with its state in a buffer $\mathcal { D } _ { S }$ . Recorded physical outcomes can be reused across policy updates; predictive accuracy is checked on recent held-out placements as the visited states change. They come from two sources, collected during the rollout phase of Algorithm 1:

• Executed placements. Every placement executed in a rollout is labeled by its real outcome, with label weight $\zeta \ = \ 2 .$ . A random 10% of executed outcomes go to a holdout set, excluded from supervised training and used to assess the learning copy $S _ { \psi }$ before activation (Sec. IV-G).

• Simulated placements. At every step, up to $B \ =$ $B _ { s } + B _ { r }$ other valid placements per environment are tested in a copy of the environment and labeled by their simulated outcome, with label weight $\zeta = 1$ . They are never executed. The copies run in a pool of Isaac Gym environments alongside the rollout environments, so they settle in the same batched simulation. Up to $B _ { s } = 6$ placements are selected by descending score $\pi ( a \mid s ) ^ { \gamma _ { \pi } } \nu ( s , a )$ , with $\gamma _ { \pi } ~ = ~ 0 . 5$ and at most two per item token, so that labels go to placements the policy might choose and S is unsure about. Up to $B _ { r } = 2$ more are drawn uniformly without replacement from the remaining candidates, so that $S$ cannot keep confirming its own confident mistakes. Every eighth step, the executed placement is also simulated. Its simulated outcome is added to the training buffer and compared with the executed outcome to measure copy bias. Thus the holdout excludes executed outcomes, but can overlap simulated training queries at the same state– action pair.

The buffer keeps the labels of the most recent 40,000 visited states. After each rollout’s actor–critic update phase, $S _ { \psi }$ takes 64 Adam steps on a weighted binary cross-entropy with a bootstrap mask drawn once when each label is stored: independently, each head sees that label with probability ${ \frac { 1 } { 2 } } .$ Keeping this mask fixed across reuse encourages head diversity:

$$
\mathcal { L } _ { S } ( \psi ) = \frac { \sum _ { i , n } \mu _ { i n } \zeta _ { i } \operatorname { B C E } _ { \log \mathrm { i t } } \left( y _ { i } , g _ { \psi } ^ { ( n ) } ( s _ { i } , a _ { i } ) \right) } { \operatorname* { m a x } \{ 1 , \sum _ { i , n } \mu _ { i n } \zeta _ { i } \} } ,\tag{5}
$$

where i indexes sampled labels, $\begin{array} { r } { \mu _ { i n } \sim \mathrm { B e r n o u l l i } ( \frac { 1 } { 2 } ) } \end{array}$ decides whether head n sees label $i , \zeta _ { i }$ is its source weight, and $\mathrm { B C E _ { l o g i t } }$ is binary cross-entropy evaluated from logits. Executed labels weigh more because a simulated copy reports collapses that the real step does not have, mostly on tall stacks. With $B _ { s } = B _ { r } = 0$ and the periodic copy check disabled, S learns from executed placements only, and SHORE makes no simulator calls beyond the rollout itself.

## E. Stability-Factored Critic

For a fixed policy and fixed return targets, let $G _ { t }$ be the soft λ-return used to train the critic. It includes the scaled immediate reward and future soft values, but not the current action’s entropy term. Thus $G _ { t } = 0$ on an immediate collapse. Define $S _ { \mathrm { t r u e } } ( s , a ) = \operatorname* { P r } ( \chi _ { t } = 1 \mid s _ { t } = s , a _ { t } = a )$ and $Q _ { G } ( s , a ) = \mathbb { E } [ G _ { t } \mid s _ { t } = s , a _ { t } = a ]$ . Where $S _ { \mathrm { t r u e } } > 0 .$

the law of total expectation gives $Q _ { G } ~ = ~ S _ { \mathrm { t r u e } } { \bar { U } } _ { G }$ , with $\bar { U } _ { G } \ = \ \mathbb { E } [ G _ { t } \ \vert \ s _ { t } \ = \ s , a _ { t } \ = \ a , \chi _ { t } \ = \ 1 ]$ . SHORE writes the critic network’s mean as $U _ { \phi }$ and forms

$$
Q ( s , a ) = U _ { \phi } ( s , a ) \big ( 1 - w _ { Q } + w _ { Q } S ( s , a ) \big ) ,\tag{6}
$$

which replaces $Q _ { \phi }$ in the soft value and the actor objective, with no gradient through S. The strength $w _ { Q } \in [ 0 , 1 ]$ ramps from 0 to 1 (Sec. IV-G). To move the critic target from the unconditional return toward the return conditional on survival, SHORE uses

$$
\mathcal { L } _ { U } = \frac { \sum _ { t \in \mathcal { B } _ { \mathrm { A C } } } \omega _ { t } \mathrm { C E } \big ( \mathrm { H L G } ( G _ { t } ) , p _ { \phi } ( \cdot \mid s _ { t } , a _ { t } ) \big ) } { \operatorname* { m a x } \{ \epsilon _ { \mathrm { l o s s } } , \sum _ { t \in \mathcal { B } _ { \mathrm { A C } } } \omega _ { t } \} }\tag{7}
$$

where $\boldsymbol { B } _ { \mathrm { A C } }$ is an actor–critic minibatch of rollout transitions, $\epsilon _ { \mathrm { l o s s } } ~ = ~ 1 0 ^ { - 6 }$ guards the denominator, CE is categorical cross-entropy, $\mathcal { L } _ { \mathrm { a u x } }$ is REPPO’s unchanged latent selfprediction loss, and $\omega _ { t } ~ = ~ 1$ when $\chi _ { t } ~ = ~ 1$ and $1 - w _ { Q }$ otherwise. At a fixed $( s , a )$ , the weighted scalar-target mean is $Q _ { G } / ( 1 - w _ { Q } + w _ { Q } S _ { \mathrm { t r u e } } )$ whenever the denominator is positive. If the critic represents this mean exactly, Eq. (6) yields

$$
Q _ { \mathrm { i d e a l } } ( s , a ) = Q _ { G } ( s , a ) \frac { 1 - w _ { Q } + w _ { Q } S ( s , a ) } { 1 - w _ { Q } + w _ { Q } S _ { \mathrm { t r u e } } ( s , a ) } .\tag{8}
$$

This equals $Q _ { G }$ when $S = S _ { \mathrm { t r u e } }$ pointwise, or when $w _ { Q } =$ 0. At $w _ { Q } ~ = ~ 1$ , surviving samples train $U _ { \phi }$ toward $\bar { U } _ { G }$ which still includes the risk of later collapses. The identity explains the weighting design under exact mean fitting; finite data, function approximation, auxiliary training and HL-Gauss encoding can introduce additional error. In particular, low aggregate calibration error does not establish pointwise equality. If $w _ { Q } = 1$ and $S _ { \mathrm { t r u e } } = 0$ , all sample weights vanish and this conditional target is undefined.

This is where the per-action critic matters. A standard PPO critic predicts a state value; the per-action critic here permits stability to enter both value estimation and policy learning. Because $Q$ is a probability times a value, a slightly risky placement that adds much more volume can be worth more than a safe one, a trade that a hard mask cannot make.

## F. Stability Prior on the Policy

S also weights the policy directly. We write π for the full policy, with dependence on $\theta , \bar { \psi } , w _ { \pi }$ implicit:

$$
\pi ( a \mid s ) = \frac { m ( s , a ) e ^ { \ell _ { \theta } ( s , a ) } \widetilde { S } ( s , a ) ^ { w _ { \pi } } } { \sum _ { a ^ { \prime } } m ( s , a ^ { \prime } ) e ^ { \ell _ { \theta } ( s , a ^ { \prime } ) } \widetilde { S } ( s , a ^ { \prime } ) ^ { w _ { \pi } } } ,\tag{9}
$$

with a prior strength $w _ { \pi } \in [ 0 , 1 ]$ that ramps like $w _ { Q }$ . The prior is soft, unlike a hard mask. A valid placement that S wrongly rates as unsafe retains nonzero sampling probability and can therefore receive a corrective executed label. When every candidate is risky, the policy renormalizes over the least risky ones instead of emptying the action set. The policy KL is computed on these full, stability-adjusted action probabilities (Sec. IV-G).

Keeping REPPO-D’s ordinary entropy regularizer would make the actor counteract this prior. The ordinary entropyregularized objective pulls π toward $\exp ( Q / \alpha )$ , so the actor would have to cancel the prior using voxel information it cannot see. SHORE instead regularizes relative to the prior, with sums over $a \in \mathcal { A } ( s )$

$$
\mathcal { T } _ { S } ( \theta ; s ) = \sum _ { a } \pi ( a \mid s ) \Big [ \alpha \log \frac { \pi ( a \mid s ) } { \widetilde { S } ( s , a ) w _ { \pi } } - Q ( s , a ) \Big ] .\tag{10}
$$

Up to a term that does not depend on $\begin{array} { r l } { \theta , } & { { } \mathcal { I } _ { S } } \end{array} \quad =$ $\alpha \mathrm { K L } ( \pi \parallel \pi _ { S } ) - \mathbb { E } _ { \pi } [ Q ]$ with $\pi _ { S } \propto m \widetilde { S } ^ { w _ { \pi } }$ . For fixed $Q , S , w _ { \pi }$ and α $> 0 ,$ its per-state minimizer over valid-action distributions is $\pi _ { \mathrm { i m p } } \propto \widetilde { S } ^ { w _ { \pi } } \exp ( Q / \alpha )$ . Equation (9) represents this distribution when $\ell _ { \theta } ( s , a ) = Q ( s , a ) / \alpha + \kappa ( s )$ , where $\kappa ( s )$ is any action-independent offset. Thus the ideal logits need not cancel the stability prior; neural approximation and the KL gate constrain the actual update. The soft value carries the same regularizer,

$$
V ( s ) = \sum _ { a } \pi ( a \mid s ) \Big [ Q ( s , a ) - \alpha \log \frac { \pi ( a \mid s ) } { \widetilde { S } ( s , a ) ^ { w _ { \pi } } } \Big ] ,\tag{11}
$$

with zero value at terminal states.

Entropy target. We retain REPPO-D’s temperature update, which compares the policy’s Shannon entropy $\mathcal { H } ( \pi ) =$ $\textstyle - \sum _ { a \in A ( s ) } \pi ( a \mid s ) \log \pi ( a \mid s )$ with a target, but set

$$
\begin{array} { r } { \mathcal { H } _ { \mathrm { t g t } } ( s ) = \frac { 1 } { 2 } \mathcal { H } \big ( \pi _ { S } ( \cdot \mid s ) \big ) . } \end{array}\tag{12}
$$

A target based only on the number of geometrically valid actions would demand the same exploration even when most candidates are predicted unstable. This target instead adapts exploration to the concentration of the stability prior. It reduces to $\textstyle { \frac { 1 } { 2 } } \log | { \mathcal { A } } ( s )$ | when $w _ { \pi } = 0$ , because the prior is then uniform over valid actions.

## G. Safeguards for a Learned Stability Model

Stability activation gate and ramp. An untrained S is noise, and as a prior it would bias early exploration. Both strengths therefore follow one schedule, $w _ { \pi } = w _ { Q } = w ,$ with

$$
w ( n _ { \mathrm { e n v } } ) = \mathrm { c l i p } \big ( ( n _ { \mathrm { e n v } } - n _ { \mathrm { g a t e } } ) / N _ { \mathrm { r a m p } } , 0 , 1 \big ) ,\tag{13}
$$

where $n _ { \mathrm { e n v } }$ counts total training environment steps, $N _ { \mathrm { r a m p } } =$ $1 0 ^ { 5 }$ is the ramp length, and $\mathrm { c l i p } ( x , 0 , 1 )$ truncates x to $[ 0 , 1 ] .$ The gate step $n _ { \mathrm { g a t e } }$ is the first training-iteration boundary at which the updated learning copy $1 - S _ { \psi }$ detects collapses among the held-out executed placements with an AUROC of at least 0.8, measured on at least 50 collapses. Once open, the gate stays open. Until it opens, $w ~ = ~ 0$ and every equation above reduces to the learner of Sec. IV-B: $\pi \propto m e ^ { \ell _ { \theta } } , Q = U _ { \phi }$ , all critic samples have weight 1, and $\begin{array} { r } { \mathcal { H } _ { \mathrm { t g t } } = \frac { 1 } { 2 } \log \left| \mathcal { A } ( s ) \right| } \end{array}$

The activation gate protects early exploration from an uninformative stability model. Its AUROC criterion tests discrimination, not probability calibration; ECE is assessed separately in the experiments.

What the policy KL gate protects. The inherited REPPO-D gate limits actor updates relative to the rollout policy, where the on-policy critic has training support. In SHORE, it evaluates $\operatorname { K L } ( \pi _ { \mathrm { o l d } } \parallel \pi )$ over the full joint policy of Eq. (9), including the stability prior. The gate is applied per state: when its KL is below $\varepsilon _ { \mathrm { K L } }$ , the actor follows $\mathcal { I } _ { S }$ ; otherwise, only the KL pullback is applied. These statewise losses are averaged over the minibatch. The acting stability model and the strengths $w _ { Q } , w _ { \pi }$ stay fixed throughout the rollout and its actor–critic updates. This gate controls policy change; it neither tests physical stability nor guarantees that a placement holds.

Algorithm 1 SHORE: one training iteration   
Require: actor $\theta ,$ critic $\phi ,$ learning/acting stability models $\psi , \bar { \psi } ,$   
label buffer and holdout set   
1: Set w<sub>Q</sub>, $, w _ { \pi }$ from the activation ramp $( \mathrm { E q } . \ ( 1 3 ) )$   
2: Freeze $S _ { \bar { \psi } }$ and both strengths for phases $_ { 1 - 2 }$   
3: Snapshot rollout policy $\pi _ { \mathrm { o l d } }$ using Eq. (9)   
1. Collect on-policy data and stability labels   
4: for each rollout step, in parallel environments do   
5: Sample $a _ { t } \sim \pi _ { \mathrm { o l d } } ( \cdot \mid s _ { t } )$   
6: Label candidate actions in copies of simulator state $\xi _ { t }$ using   
the selection rule of Sec. IV-D   
7: Execute $a _ { t } ;$ store transition and stability outcome   
8: Add simulated labels to the buffer; split executed labels   
between the buffer and holdout set   
9: end for   
2. Update the actor and critic   
10: Compute soft λ-returns using Eq. (11)   
11: for each actor–critic minibatch update do   
12: Update $U _ { \phi }$ with the weighted loss $\left( \operatorname { E q . } \left( 7 \right) \right)$   
13: Update θ with Eq. (10) inside the REPPO-D KL gate   
relative to $\pi _ { \mathrm { o l d } }$   
14: Update REPPO-D’s dual variables, using the entropy target   
in Eq. (12)   
15: end for   
3. Learn stability and prepare the next rollout   
16: Train $S _ { \psi }$ on buffered labels with Eq. (5)   
17: Evaluate $S _ { \psi }$ on held-out executed outcomes; record $n _ { \mathrm { g a t e } }$ if   
the activation criterion first passes   
18: Set the next rollout’s strengths; move $S _ { \bar { \psi } }$ toward $S _ { \psi }$ subject to   
the stability-model KL check (Sec. IV-G)

Controlling stability-model updates. Updating the acting stability model changes the policy even with the actor fixed, so it needs separate control. After training $S _ { \psi } ,$ , we interpolate $\bar { \psi }  \bar { \psi } + \lambda _ { S } ( \psi - \bar { \psi } )$ . A bisection search chooses a step $\lambda _ { S } \in$ [0, 1] whose induced policy KL, from the pre-interpolation policy to the candidate policy, stays within the mean bound $\delta _ { S }$ on recent rollout states. The updated actor and the next rollout’s prior strength are fixed during this check. If that strength is zero, the acting copy takes the learned weights directly. This bounds the measured policy drift caused by the model interpolation; it does not bound the separate change from advancing the strength ramp. It complements the actor’s KL gate. The bound and state-sample size are given in Table VII.

## H. Deployment

The critic, the label buffer and the simulator are used only during training. At deployment, SHORE computes the geometric mask, one actor pass and the stabil-

ity network’s passes for the visible items. It then executes arg $\begin{array} { r } { \operatorname { n a x } _ { a \in \mathcal { A } ( s ) } [ \ell _ { \theta } ( s , a ) + w _ { \pi } \log \widetilde { S } ( s , a ) ] } \end{array}$ , the mode of Eq. (9). No search is run and no placement is simulated.

## V. EXPERIMENTS

The experiments answer three questions:

1) Does explicit stability modeling accelerate policy learning and improve the trade-off between packed volume and collapse risk, beyond the benefit of buffer selection and rotation?

2) Where must the stability estimate act: in the critic, in the policy, or in both?

3) Do simulated stability labels help beyond the labels of executed placements, and is choosing them by uncertainty better than choosing them at random?

## A. Baselines

All methods share the simulator, box sets, stability criterion (Eq. (2)) and evaluation protocol (Sec. V-C). The baselines keep their own action spaces and rewards. Except where stated, they place every box in its stored orientation, as in their original implementations, while SHORE and GOPT+ choose among six orientations.

Masked-PPO variants. OL Mask [3] learns a stability mask online from physics simulations, jointly with a PPO policy. It is the strongest prior method in this setting and the closest to ours, since it also learns stability from physics. Nomask removes masking entirely. Random Selection randomly picks the box from the buffer with only the placement decided by the policy network; Random Placement places at random positions with only box selection determined by the policy.

Learning-based state of the art. GOPT [2] is a transformer-based DRL packer using Empty Maximal Spaces (EMSs — the maximal axis-aligned empty boxes remaining in the bin) as placement candidates. We use the official implementation (Packing Transformer, EMS placement generator, and masked PPO) and swap only its geometric environment for ours, so that placements are executed and adjudicated by the same physics: unstable placements terminate episodes exactly as for the other methods. Since GOPT is a strict online packer without buffer selection, it packs the buffer head at each step. GOPT+ extends GOPT to the action space of SHORE. The same Packing Transformer chooses any visible box in any of its six orientations, i.e., a joint (box, orientation, EMS) action, and is trained with GOPT’s masked PPO [5], with the clipped ratio applied separately to the (box, orientation) choice and to the EMS choice. Its rollout size, learning rate and number of updates are matched to SHORE − actor prior − critic weighting, so GOPT+ has the actor network and the action space of SHORE − actor prior − critic weighting and differs from it in the learner.

To assess sensitivity to box geometry, the expanded eval uation uses the 12 geometries in Tab. II, with five boxes per geometry (60 arrivals per episode), alongside the fourtype base set. Density is assigned independently of geometry: each odd-numbered geometry contributes three light and two heavy boxes and each even-numbered geometry two light and three heavy, giving 30 light and 30 heavy boxes per episode. Light and heavy boxes use the two materials of the base set $( 1 0 0 \mathrm { k g / m ^ { 3 } }$ with rigidity 0.5, and 5000 kg/m<sup>3</sup> with rigidity 3; Tab. I). The 60 boxes fill 86.6% of the bin volume, against 69.1% for the base set, so the expanded set is both more heterogeneous and tighter. Every learned method is retrained from scratch on the expanded set with unchanged hyperparameters, and no policy is transferred between box sets. The training budget of each method is given with its results.

TABLE I: Box specifications in the simulation environment.
<table><tr><td>Dimension (cm)</td><td>Density (kg/m³)</td><td>Rigidity</td><td>Counts</td></tr><tr><td> $6 \times 6 \times 4$ </td><td>100</td><td>0.5</td><td>10</td></tr><tr><td> $6 \times 6 \times 6$ </td><td>100</td><td>0.5</td><td>10</td></tr><tr><td> $6 \times 6 \times 6$ </td><td>5000</td><td>3</td><td>10</td></tr><tr><td> $8 \times 6 \times 6$ </td><td>5000</td><td>3</td><td>10</td></tr></table>

TABLE II: Box geometries for the FBA [19]-inspired synthetic benchmark. Each geometry appears five times per episode. Density is assigned separately from geometry.
<table><tr><td>ID</td><td>Reference (in)</td><td>Simulation (cm)</td><td>Count</td></tr><tr><td>B01</td><td> $8 \times 8 \times 6$ </td><td> $\overline { { 4 \times 4 \times 3 } }$ </td><td>5</td></tr><tr><td>B02</td><td> $1 0 \times 8 \times 6$ </td><td> $5 \times 4 \times 3$ </td><td>5</td></tr><tr><td>B03</td><td> $1 2 \times 8 \times 6$ </td><td> $6 \times 4 \times 3$ </td><td>5</td></tr><tr><td>B04</td><td> $1 2 \times 1 0 \times 6$ </td><td> $6 \times 5 \times 3$ </td><td>5</td></tr><tr><td>B05</td><td> $1 2 \times 1 2 \times 8$ </td><td> $6 \times 6 \times 4$ </td><td>5</td></tr><tr><td>B06</td><td> $1 4 \times 1 0 \times 8$ </td><td> $7 \times 5 \times 4$ </td><td>5</td></tr><tr><td>B07</td><td> $1 6 \times 1 0 \times 6$ </td><td> $8 \times 5 \times 3$ </td><td>5</td></tr><tr><td>B08</td><td> $1 6 \times 1 2 \times 8$ </td><td> $8 \times 6 \times 4$ </td><td>5</td></tr><tr><td>B09</td><td> $1 6 \times 1 2 \times 1 2$ </td><td> $8 \times 6 \times 6$ </td><td>5</td></tr><tr><td>B10</td><td> $1 8 \times 1 2 \times 8$ </td><td> $9 \times 6 \times 4$ </td><td>5</td></tr><tr><td>B11</td><td> $1 8 \times 1 4 \times 1 0$ </td><td> $9 \times 7 \times 5$ </td><td>5</td></tr><tr><td>B12</td><td> $2 0 \times 1 6 \times 1 2$ </td><td> $1 0 \times 8 \times 6$ </td><td>5</td></tr></table>

## B. Simulation Setup

We run all training and evaluation in Isaac Gym [20] with the GPU-accelerated PhysX backend. A placement is executed as a release 2 mm above the target position in the selected orientation, without a full arm trajectory or added Gaussian position or rotation perturbations. This deliberate simplification keeps the property that matters for the decision problem, whether the stack holds, while abstracting away manipulator dynamics. A single box released on the bin settles onto its target with zero measurable drift.

For sim-to-sim transfer, we calibrated Isaac Gym’s PhysX contact behavior against high-fidelity MuJoCo using paired placement replays (Appendix A). This aligns the faster simulator with MuJoCo’s placement outcomes while retaining Isaac Gym’s GPU-parallel capacity for large-scale RL training. Agreement is high but imperfect, so reported results remain specific to the calibrated Isaac Gym model.

In implementation, the bin is $2 5 \times 2 5$ cm with a maximum stacking height of 20 cm, discretized into $2 5 \times 2 5 \times 2 0$ voxels of 1 cm for placement choice. The length of the processing buffer is set to 5. Our box configuration uses light and heavy densities of 100 and $\mathrm { 5 0 0 0 k g / m ^ { 3 } }$ , respectively, following the released implementation of [3]. This density contrast makes density-aware ordering important. The experimental settings used for our comparisons are listed here rather than assumed to match all settings reported in the original paper. All remaining hyperparameters are listed in Tab. VII (Appendix A).

With $N = 5$ and $K = 8 0 ,$ each state has at most $6 N K =$ 2,400 joint actions, against $6 N l _ { p } w _ { p } = 1 8 , 7 5 0$ if every cell of the $2 5 \times 2 5$ grid were a placement choice. This action space is still large enough to make learning challenging.

Four types of boxes with different dimensions (length, width, and height), densities, and rigidities are used. While the density parameter controls how heavy the box is, the rigidity parameter controls the deformation degree of the box under contact forces; both impact placement stability. The detailed box specification is listed in Tab. I. The mixture of light deformable and heavy rigid boxes is what makes stability depend on more than geometry: for example, packing the light $6 \times 6 \times 4$ boxes first produces a soft top surface, and any heavy $8 \times 6 \times 6$ box arriving later must either be placed on that yielding surface (typically unstable) or terminate the episode. A policy that knows which surfaces will hold can choose its boxes and placements to keep rigid support available.

The total volume of all boxes in the base set is 0.691 of the bin volume, which caps utilization even if every box is packed successfully. This near-saturation ceiling makes the base set a weak test of further packing gains. We therefore also evaluate on the expanded FBA set, whose broader range of 12 carton dimensions better represents varied fulfillmentbox sizes and raises the all-boxes-packed ceiling to 0.866. The expanded set offers more headroom for comparing packing strategies and makes box choice and orientation more consequential. The base set remains useful for checking whether the method’s behavior transfers to a simpler, nearly saturated inventory.

## C. Evaluation Metrics

Space utilization is the total stably packed box volume divided by bin volume. Mean episode length counts boxes successfully placed before termination.

Completion rate and unstable rate report the fractions of episodes that end with every box placed and with a collapse (Sec. III-E). We do not report the share of infeasible endings as a metric: what it measures depends on each method’s feasibility filter (Sec. III-E), so it is not comparable across methods. Collapses per placement is the number of collapses divided by the number of placement attempts (placed boxes plus collapses) over a method’s 50 evaluation episodes, in percent. An episode has at most one collapse. Thus, within a training seed, if $f _ { \mathrm { c o l } }$ is the fraction of episodes ending in collapse and L<sup>¯</sup> is the mean number of successfully placed boxes, collapses per attempt is $f _ { \mathrm { c o l } } / ( \bar { L } + f _ { \mathrm { c o l } } )$ before conversion to percent. This normalizes failures by the number of executed attempts; it is not an independent outcome from episode length and unstable rate. It estimates the observed failure frequency under each policy’s visited states, rather than risk on a common set of stacks. A low per-attempt frequency can still accompany frequent episode failures when many placements must succeed in succession.

TABLE III: Performance comparison on the base box set. All methods are evaluated using the 1M trained checkpoint over 50 episodes with randomized arrival order and identical seeds. Episode metrics average over evaluation episodes. Space utilization and termination rates are dimensionless; episode length counts placed boxes and collapses per placement is in percent. Metrics are defined in Sec. V-C. Completion rate is the fraction of episodes in which all boxes are placed.
<table><tr><td></td><td>OL Mask [3]</td><td>GOPT [2]</td><td>GOPT+</td><td>SHORE(Ours)</td></tr><tr><td>Space utilization(↑)</td><td>0.495</td><td>0.661</td><td>0.683</td><td>0.657</td></tr><tr><td>Mean Episode Length(↑)</td><td>28.68</td><td>38.25</td><td>39.53</td><td>38.18</td></tr><tr><td>Unstable Rate(↓)</td><td>0.14</td><td>0.18</td><td>0.05</td><td>0.16</td></tr><tr><td>Collapses per placement (%)(↓)</td><td>0.49</td><td>0.48</td><td>0.12</td><td>0.42</td></tr><tr><td>Completion rate(↑)</td><td>0.00</td><td>0.67</td><td>0.95</td><td>0.84</td></tr></table>

TABLE IV: Performance on the expanded box set (12 geometries in Tab. II, 60 arrivals per episode). All methods are evaluated using the 1M trained checkpoint over 50 episodes with randomized arrival order and identical seeds. Learned methods report mean ± standard deviation over three training seeds. No method completes the 60-box sequence, so the completion rate is 0 and omitted, and every episode that does not end in a collapse ends with no feasible placement (Sec. III-E). Collapses per placement is in percent (Sec. V-C).
<table><tr><td></td><td>OL Mask‡ [3]</td><td>GOPT [2]</td><td>GOPT+</td><td>SHORE(Ours)</td></tr><tr><td>Space utilization(↑)</td><td> $0 . 2 4 2 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $0 . 5 8 8 { \pm } 0 . 0 1 7$ </td><td> $0 . 7 1 5 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $\mathbf { 0 . 7 3 4 } \mathrm { \pm 0 . 0 0 6 }$ </td></tr><tr><td>Mean Episode Length(↑)</td><td> $1 6 . 4 5 { \pm } 1 . 0 7$ </td><td> $4 1 . 0 8 { \pm } 1 . 0 6 $ </td><td> $5 0 . 5 5 { \pm } 0 . 6 9$ </td><td> ${ \pm 0 . 8 3 \pm 0 . 5 3 }$ </td></tr><tr><td>Unstable Rate(↓)</td><td>0.99±0.01</td><td> $\mathbf { 0 . 2 6 { \scriptstyle \pm 0 . 0 6 } }$ </td><td> $0 . 6 1 { \pm } 0 . 0 8$ </td><td> $0 . 5 3 { \pm } 0 . 0 5$ </td></tr><tr><td>Collapses per placement (%)(↓)</td><td>5.69±0.30</td><td> $\mathbf { 0 . 6 3 \pm } 0 . 1 5$ </td><td> $1 . 2 0 { \pm } 0 . 1 7 $ </td><td> $1 . 0 3 { \pm } 0 . 1 0$ </td></tr></table>

Stability-model metrics. The ablations (Sec. V-E) evaluate the learning copy $S _ { \psi }$ on held-out executed outcomes. These outcomes are excluded from supervised training, but simulated labels can cover the same state–action pairs (Sec. IV-D); the evaluation therefore does not establish generalization to entirely unseen placements. It also does not directly assess the acting copy $S _ { \bar { \psi } }$

AUROC, the area under the receiver operating characteristic curve for $1 - S _ { \psi }$ detecting collapses, measures ranking: the probability that a collapsed placement receives a lower prediction than a successful one, with half credit for ties. A score of 0.5 is chance and 1 is perfect ranking. This metric opens the activation gate. Expected calibration error (ECE) groups predictions into ten equal-width bins and averages the absolute difference between each bin’s mean prediction and observed success frequency, weighted by bin size. Zero means perfect agreement at this bin resolution. Probability accuracy matters because the critic uses $S _ { \bar { \psi } }$ multiplicatively, but low ECE does not prove the pointwise accuracy required by Eq. (8). The metrics are complementary: with rare collapses, a constant high prediction can have low ECE while ranking at chance; good ranking can coexist with pessimistic probabilities. Both metrics use the most recent held-out outcomes after each training iteration: about 4,000, including 37–66 collapses at the reported ablation checkpoint.

Although the box type distribution is shared between training and evaluation (a realistic assumption for a warehouse with standardized box sizes), the arrival order is independently randomized in every evaluation episode, so evaluation sequences are almost surely never seen during training; the combinatorial space of arrival orders is $\left( _ { 1 0 , 1 0 , 1 0 , 1 0 } ^ { 4 0 } \right) \approx \mathrm { \bar { 4 } } . 7 \times$ $1 0 ^ { 2 1 }$ . All metrics are averaged over 50 evaluation episodes with the same fixed episode seeds across all methods, using deterministic (argmax) actions and no search for the learned policies.

## D. Results

Separating action flexibility from stability awareness. The base-set comparison is close to its packing ceiling, while the expanded set tests the method in a more demanding regime: varied geometries constrain future fit, and material properties are independent of shape, so geometry alone cannot identify reliable support. At the base-set checkpoint, GOPT+ has higher utilization and completion than the singleseed SHORE run, despite SHORE’s earlier rise in training reward. The curves therefore suggest faster early learning in this run, not a base-set evaluation advantage or a seedrobust difference. The expanded set provides the stronger test of whether modeling stability helps when box choice, orientation, and future fit interact.

The comparison between GOPT and GOPT+ tests the value of broadening decisions beyond the buffer head and its fixed orientation. GOPT+ is our extension of the official GOPT baseline: it can choose a visible box, its orientation, and an EMS placement, so order, orientation, and location all affect the next packing state. Its stronger performance indicates that exposing these choices can improve packing on the expanded set and supports the importance of flexible, coordinated decisions. This is evidence for the value of the expanded decision space, though the comparison does not isolate action flexibility alone: GOPT+ also uses a different

![](images/e7e2e5bb5adffc7475389898bf881214b2ba3a5297e02524309f6aca0c3b282e.jpg)  
Fig. 3: Training curves on the base box set for the methods of Table III, up to 1M environment steps, computed from training episodes as in Fig. 4. (a) Mean reward of training episodes (packed volume fraction). (b) Fraction of training episodes that end in a collapse. (c) Collapses per placement attempt (log scale). The original GOPT runs did not log how training episodes end, so GOPT appears only in (a).

PPO update, with clipped ratios applied separately to itemorientation and placement choices.

Against this stronger diagnostic baseline, SHORE attains higher mean utilization while experiencing fewer collapses both per episode and per placement attempt, despite sharing its action space and actor backbone. The comparison therefore tests what stability-aware valuation and policy guidance add after the joint choices are available. To isolate stability modeling, the relevant control is REPPO-D, which shares the learner but removes both SHORE components. Its eventual utilization matches SHORE within training-seed variation. The evidence thus points to earlier acquisition and a better packing–risk balance, rather than a higher final utilization

ceiling.

Why stability supervision helps learning. Figure 4(a) and the intermediate deterministic checkpoint evaluations show that SHORE reaches its high-utilization regime earlier than the learner without S. The advantage persists across intermediate checkpoints and appears under both sampled training actions and deterministic evaluation, so it is not merely a consequence of exploration noise in training returns. This pattern supports the hypothesis that separating immediate physical survival from continuation value makes learning easier. Without S, the critic must infer both effects from returns; a failed placement also removes the opportunity to observe later packing decisions. In SHORE, supervised placement outcomes train a dedicated model with access to the internal density layout, and that information immediately affects the values and probabilities of joint box–orientation– placement choices. The later catch-up of the base learner is consistent with S supplying useful information sooner, while both ultimately optimize the same packed-volume objective. This learning advantage is measured in rollout environment steps.

Stability without sacrificing useful packing. Table IV shows that SHORE packs more volume than GOPT+ with fewer failures per attempt, while their episode lengths are comparable. Its lower episode-collapse share thus accompanies a lower observed placement failure frequency, rather than simply fewer opportunities to fail. This supports using S to guide productive joint actions. The two collapse metrics summarize the same failures at different scales; their agreement clarifies the packing–risk trade-off rather than providing independent statistical evidence.

Figure 4(c) shows how this trade-off develops during learning. After the initial improvement, SHORE sustains fewer collapses per attempt than the learner without S, alongside higher packed-volume reward in panel (a). GOPT+ approaches its placement-level failure frequency later, consistent with explicit stability supervision accelerating the acquisition of useful physical constraints. OL Mask reveals why both collapse metrics are needed: its failures per attempt decline while its episode-collapse share stays near saturation. It sustains more successful placements before failure, but still fails before completion. Panel (c) therefore reveals progress that panel (b) obscures. These sampled training trajectories complement the deterministic evaluation comparison; they do not measure deployment-policy performance directly.

Balancing stability against packed volume. The expanded set shows two ways to get this trade-off wrong. GOPT is the conservative extreme. It has the lowest observed collapse frequency, yet packs much less volume than the methods that choose from the whole buffer. Its low collapse share is partly a consequence of stopping early: it places only the buffer head in its stored orientation, so most of its episodes end once that box has no admissible placement, after about 41 boxes, and each episode contains fewer placements that could fail. Its restricted actions also make each placement more conservative: it never turns a box onto a smaller face and never packs the last, tightest part of the bin. The training curves show the same regime throughout learning (Fig. 4): about 30% of GOPT’s training episodes end in a collapse, the fewest of any method, while its training reward stays near 0.5. GOPT+ is the aggressive counterpart. With buffer selection and rotation it packs more volume, but incurs a higher collapse rate. Because a collapse ends the episode and forfeits every box not yet packed, neither extreme is optimal: avoiding risk leaves space unused, and ignoring risk lets a single failed placement discard the volume the remaining boxes would have added. SHORE’s action value prices this trade-off directly, as the probability that a placement holds times the return if it holds. With the same action space and backbone as GOPT+, it attains higher utilization while experiencing fewer episode collapses and fewer failures per placement. The aim is thus not the lowest collapse rate, which GOPT shows can be bought by stopping early, but the most packed volume for the risk taken. OL Mask fails on both counts: its hard mask admits placements that do not hold, and 99% of its episodes end in a collapse after about 16 boxes, although its different architecture, orientation restriction and training procedure prevent isolating soft versus hard masking as the cause.

![](images/97e55f41e211cfb580ccd855693ccfaf222e5a6d6c518e040b25926ceaaa9f2a.jpg)

No method completes the expanded sequences. For SHORE, fewer collapse endings accompany more geometrically infeasible endings, identifying preservation of usable space and support as a remaining challenge. Its soft prior does not remove geometrically valid actions; earlier choices can nevertheless leave no admissible placement. Infrequent placement failures can still interrupt a long sequence, while avoiding collapse alone does not preserve room for the remaining boxes. Normalization accounts for the number of attempts but not differences in the states visited; establishing lower risk on the same candidate placements would require a separate controlled comparison.

Figure 5 illustrates these distinct failure mechanisms under identical arrivals. We use the final checkpoint from the training seed with the median final score, and select the episode closest to the mean rewards of GOPT and SHORE. GOPT stops because the buffer head has no admissible placement, illustrating the restriction that buffer selection and rotation address. GOPT+ uses that flexibility to continue, but eventually exhausts admissible placements for every visible box. OL Mask instead admits a placement that destabilizes the stack, illustrating that passing a learned filter is not a physical guarantee. SHORE builds a substantially filled stack before a heavy placement fails; the remaining failure makes clear that its soft prior manages risk rather than eliminating it. The scene also shows why box count and utilization can disagree: SHORE packs fewer boxes than GOPT+ here but more volume. Evaluating the joint policy therefore requires the volume objective as well as the failure mode, rather than counting placements alone.

## E. Ablations

The ablations change one part of SHORE at a time on the expanded set, with all other settings fixed, and fall into two groups. For the component ablation (Table V), AP

Fig. 4: Training curves on the expanded (FBA) box set. (a) Mean reward of training episodes (packed volume fraction), with actions sampled from each method’s training policy. (b) Fraction of training episodes that end in a collapse; the others end with no feasible placement. (c) Collapses per placement attempt during training (log scale), pooled over the training episodes in each window. All three panels are computed from training episodes. A method whose episodes collapse long before the bin fills stays near 1 in (b) even as its collapses per placement fall in (c). The curves end at 1M steps, the checkpoint evaluated in Table IV.

(actor prior only) sets $w _ { Q } = 0 ,$ , CW (critic weighting only) sets $w _ { \pi } ~ = ~ 0 ,$ , and REPPO-D switches off both SHORE components $( w _ { \pi } = w _ { Q } = 0 )$ , leaving the discretized REPPO backbone of Sec. IV-B. The second group tests where the stability labels come from (Table VI), with both components in place: executed placements only $( B _ { s } ~ = ~ B _ { r } ~ = ~ 0 .$ , no simulator calls beyond the rollout), one simulated label per step $( B _ { s } = 1 , B _ { r } = 0 )$ , eight uniformly random simulated labels $( B _ { s } = 0 , B _ { r } = 8 )$ , or the default eight, six chosen by $\pi ^ { \gamma _ { \pi } } \nu$ and two at random $( B _ { s } = 6 , B _ { r } = 2 )$ . All variants are compared at the same checkpoint, 901,120 environment steps. Every run that trains a stability model was resumed from this checkpoint, and a resume empties the label buffer, so a later checkpoint would compare variants at different distances from a reset. Each variant is scored with the protocol of Sec. V-C, and its stability model is also scored by AUROC (how well S separates placements that collapse from placements that hold) and ECE (how closely S matches the observed rate of holding), both defined in Sec. V-C.

![](images/b10519d30661c27bfae3ad303163f197963d06ea67f4cd4fe3992f3e80ce02f9.jpg)

(a) OL Mask: 40 boxes left on the table.  
![](images/1bfd4300d53c8d421950dcbb7712191fcdb66c605bfad0e68274f6734de54f3b.jpg)

(b) GOPT: 15 boxes left on the table.  
![](images/e783700946bb00e7ccf6b29c1c3f6ee73f2a5a55fa1af26fd6d8e339a603cd82.jpg)

(c) GOPT+: 5 boxes left on the table.  
![](images/1cc8f8636338e2541aff566718336df8b0ac39396b2e26836c5c47ef45fed79b.jpg)  
(d) SHORE: 5 boxes left on the table.  
Fig. 5: An illustration of the scenes of one evaluation episode on the expanded box set, with the same box arrivals for all four methods, rendered in Isaac Gym after the episode ends. Each panel shows the final stack (left) and the table from above (right) with its stack and the boxes still waiting on the table. Pale boxes are light and dark boxes heavy; the red slab is the bin floor. OL Mask stopped when the last placement ended unstable and its collapse knocked boxes off the stack onto the table beside the bin. GOPT, GOPT stopped when on buffer boxes can find feasible placement. SHORE stopped when the last placement fails the stability test, but the unstability did not leads to stack collapse.

TABLE V: Ablation of where the stability estimate $S$ acts, on the expanded box set. Each variant removes the named components from SHORE (Sec. V-E); the variants that keep S use the default labels of Table VI. AUROC (higher: S ranks collapses below placements that hold more often) and expected calibration error (ECE; lower: the values of $S$ are closer to the observed rate of holding) score the checkpoint’s S on held-out executed placements (Sec. V-C); they are n/a for REPPO-D, which has no stability model to score.
<table><tr><td>Variant</td><td>(↑)</td><td>(↓)</td><td>Score Collapse AUROC ECE (↑)</td><td>(↓)</td></tr><tr><td>SHORE</td><td>0.734</td><td>0.53</td><td>0.85</td><td>0.040</td></tr><tr><td>CW</td><td>0.708</td><td>0.60</td><td>0.86</td><td>0.051</td></tr><tr><td>AP</td><td>0.699</td><td>0.58</td><td>0.82</td><td>0.042</td></tr><tr><td>REPPO-D*</td><td>0.699</td><td>0.62</td><td>n/a</td><td>n/a</td></tr></table>

TABLE VI: Ablation of the stability label source on the expanded box set, with both the actor prior and the critic weighting in place. $B _ { s }$ simulated labels per step are chosen by $\pi ^ { \gamma _ { \pi } } \nu$ and $B _ { r }$ uniformly at random; executed placements are always labeled.
<table><tr><td>Labels</td><td> $B _ { s } / B _ { r }$ </td><td>(↑)</td><td>(↓)</td><td>Score Collapse AUROC (↑)</td><td>ECE (4)</td></tr><tr><td>SHORE (default)</td><td>6/2</td><td>0.734</td><td>0.54</td><td>0.85</td><td>0.040</td></tr><tr><td>Executed only</td><td>0/0</td><td>0.731</td><td>0.52</td><td>0.77</td><td>0.009</td></tr><tr><td>1 simulated</td><td>1/0</td><td>0.716</td><td>0.52</td><td>0.78</td><td>0.004</td></tr><tr><td>8 random</td><td>0/8</td><td>0.732</td><td>0.52</td><td>0.79</td><td>0.035</td></tr></table>

The component and label-source variants are exploratory single-seed comparisons, whereas SHORE and SHORE − actor prior − critic weighting average multiple training seeds. Their small differences in packing or collapse share should therefore be treated as directional evidence. The reported metrics describe the learning copy, whereas decisions use the interpolated acting copy. Stability-model metrics also fluctuate across training iterations and are measured on each policy’s own held-out executed outcomes, rather than a common candidate set. They diagnose behavior on the actions each policy encounters; they do not establish uniform accuracy across the joint action space.

Complementary roles of value weighting and the action prior. Table V supports complementary roles for the actor prior and critic weighting: retaining either alone does not recover the full method’s packing performance. Critic weighting makes the value reflect the chance of losing both the current reward and the continuation, but its influence on action selection must pass through actor learning. The prior changes relative action probabilities directly, using voxel information unavailable to the actor, but by itself does not separate survival probability from the critic’s return estimate conditional on survival. Their combination supplies both a physically informed action value and an immediate bias toward actions likely to hold. The observed ordering is consistent with this division of labor, although repeated seeds are needed to establish the interaction reliably.

![](images/7a9ab12cdc4e73e03af8ba636fb9e2685baef7f6c0f8f2bd04f656257dfdb643.jpg)  
Fig. 6: Real-robot packing with the UFactory Lite6. Key frames from the right camera view, in temporal order from left to right: the empty bin (yellow) and the stack after each of four successive placements. Boxes are picked from the table on the left of each frame.

The CW variant is especially informative: its stability model ranks outcomes at least as well as the full model at the reported checkpoint, yet its packing performance is lower. Prediction quality alone therefore does not explain the full method’s advantage; how the estimate enters decision making matters. This is empirical support for the integration of S into both modules, rather than evidence that a more accurate classifier automatically yields a better packer. The value factorization in Sec. IV-E supplies the mathematical justification under pointwise accurate stability probabilities and exact mean fitting; these ablations test its practical usefulness with an imperfect learned model. They do not independently validate the stability activation gate or stability trust region, which are retained in the tested variants.

What additional simulated labels contribute. Table VI shows that executed outcomes alone suffice to reach comparable packing performance at the ablation checkpoint. Together with the earlier checkpoint advantage of the default label scheme, this suggests that simulated alternatives mainly help acquire useful stability information before the rollout has supplied enough examples. They are not required to sustain the observed late-stage packing quality. Uncertaintydirected sampling also gives no demonstrated packing advantage over uniform sampling at this budget. The evidence therefore supports the stability model more strongly than it supports a particular acquisition rule or the necessity of extra simulator calls.

The ranking and calibration results explain why more labels need not improve control. The default simulatedlabel scheme yields stronger outcome ranking but poorer probability calibration than learning from executed outcomes alone. The recorded copy bias provides a plausible mechanism: simulated copies sometimes report collapse where the executed placement holds, pushing S toward pessimistic probabilities. A ranking metric can improve despite this shift, while the multiplicative critic weighting depends on the probabilities themselves. Conversely, low ECE on mostly successful executed placements need not imply good discrimination of rare failures (Sec. V-C). These findings support assessing both properties and caution against using either as a surrogate for packing performance. They also identify a concrete improvement path: reduce copy bias and assess label acquisition at matched simulation budgets, while retaining executed outcomes as the calibration reference.

## VI. REAL ROBOT DEPLOYMENT

We demonstrate the packing policy on a UFactory Lite6 arm with a vacuum gripper. The system uses a ZED 2i stereo camera and a hand-mounted camera. Fig. 6 shows one illustrated placement sequence. The yellow area serves as the packing bin. The sequence of cubes arrives in the order from right to left. This demonstration shows that our policy can be easily integrated into a robot packing pipeline.

## VII. CONCLUSION AND FUTURE WORK

We presented SHORE, an on-policy reinforcement learning method for physics-aware online 3D bin packing, which incorporates learned stability information through a policy prior and continuous weighting of action-conditioned critic values. The experiments support faster acquisition of high utilization policies than learning without explicit stability modeling. Against GOPT+, which shares the joint action space and actor backbone, SHORE achieves higher mean utilization with fewer collapses per episode and per placement attempt. The training curves further show that the stability model reduces failures per attempt relative to the same learner without it. These findings support using stability information in both action valuation and selection to improve the packing–risk trade-off; the eventual catch-up of the base learner limits claims of a higher final performance ceiling.

Meanwhile, the method is limited by its observation and candidate representations, approximate stability probabilities, and simulator fidelity. A collapse is attributed to the triggering placement even when earlier choices created the fragile structure; those earlier choices receive information only through returns, and the value network cannot see the raw voxel grid of the bin. Collapses per placement account for the number of attempts, but remain specific to the states each policy visits; earlier infeasible termination can prevent exposure to difficult late-stage placements. Stability metrics must therefore be interpreted together with utilization and termination modes. Besides, there are more sim-toreal factors that we did not take into account, such as contact-parameter mismatch, bin wall collisions, and robot reachability.

For future work, it is better to specifically distinguish exogenous factors like incoming box distribution from endogenous factors, and tackle the problem from the exogenous MDP perspective [21]. Techniques can include, but are not limited to, exogenous state decomposition [22] or training a predictive model of the exogenous factors.

## REFERENCES

[1] H. Zhao, C. Zhu, X. Xu, H. Huang, and K. Xu, “Learning practically feasible policies for online 3d bin packing,” Science China Information Sciences, vol. 65, no. 1, p. 112105, 2022.

[2] H. Xiong, C. Guo, J. Peng, K. Ding, W. Chen, X. Qiu, L. Bai, and J. Xu, “Gopt: Generalizable online 3d bin packing via transformerbased deep reinforcement learning,” IEEE Robotics and Automation Letters, vol. 9, no. 11, pp. 10 335–10 342, 2024.

[3] T. Zhang, Z. Wu, Y. Chen, Y. Wang, B. Liang, S. Moura, M. Tomizuka, M. Ding, and W. Zhan, “Physics-aware robotic palletization with online masking inference,” in 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025, pp. 6139–6145.

[4] Z. Wu, Y. Li, W. Zhan, C. Liu, Y.-H. Liu, and M. Tomizuka, “Efficient reinforcement learning of task planners for robotic palletization through iterative action masking learning,” IEEE Robotics and Automation Letters, vol. 9, no. 11, pp. 9303–9310, 2024.

[5] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.

[6] K. Karabulut and M. M. <sup>˙</sup>Inceoglu, “A hybrid genetic algorithm for ˘ packing in 3d with deepest bottom left with fill method,” in International Conference on Advances in Information Systems. Springer, 2004, pp. 441–450.

[7] C. T. Ha, T. T. Nguyen, L. T. Bui, and R. Wang, “An online packing heuristic for the three-dimensional container loading problem in dynamic environments and the physical internet,” in European Conference on the Applications of Evolutionary Computation. Springer, 2017, pp. 140–155.

[8] F. Wang and K. Hauser, “Stable bin packing of non-convex 3d objects with a robot manipulator,” in 2019 International Conference on Robotics and Automation (ICRA). IEEE, 2019, pp. 8698–8704.

[9] R. Verma, A. Singhal, H. Khadilkar, A. Basumatary, S. Nayak, H. V. Singh, S. Kumar, and R. Sinha, “A generalized reinforcement learning algorithm for online 3d bin-packing,” arXiv preprint arXiv:2007.00463, 2020.

[10] H. Zhao, Q. She, C. Zhu, Y. Yang, and K. Xu, “Online 3d bin packing with constrained deep reinforcement learning,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, no. 1, 2021, pp. 741–749.

[11] Z. Yang, S. Yang, S. Song, W. Zhang, R. Song, J. Cheng, and Y. Li, “Packerbot: Variable-sized product packing with heuristic deep reinforcement learning,” in 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2021, pp. 5002– 5008.

[12] P. Zhou, Z. Gao, C. Li, and N. Y. Chong, “An efficient deep reinforcement learning model for online 3d bin packing combining object rearrangement and stable placement,” in 2024 24th International Conference on Control, Automation and Systems (ICCAS). IEEE, 2024, pp. 964–969.

[13] H. Zhao, Y. Yu, and K. Xu, “Learning efficient online 3d bin packing on packing configuration trees,” in International conference on learning representations, 2021.

[14] A. V. Puche and S. Lee, “Online 3d bin packing reinforcement learning solution with buffer,” in 2022 ieee/rsj international conference on intelligent robots and systems (iros). IEEE, 2022, pp. 8902–8909.

[15] C. Voelcker, A. Brunnbauer, M. Hussing, M. Nauman, P. Abbeel, R. Grosu, E. Eaton, A.-m. Farahmand, and I. Gilitschenski, “Relative entropy pathwise policy optimization,” in International Conference on Learning Representations, vol. 2026, 2026, pp. 35 761–35 790.

[16] J. Farebrother, J. Orbay, Q. Vuong, A. A. Ta¨ıga, Y. Chebotar, T. Xiao, A. Irpan, S. Levine, P. S. Castro, A. Faust, et al., “Stop regressing: Training value functions via classification for scalable deep rl,” arXiv preprint arXiv:2403.03950, 2024.

[17] O. C¸ ic¸ek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ron-<sup>¨</sup> neberger, “3d u-net: learning dense volumetric segmentation from sparse annotation,” in International conference on medical image computing and computer-assisted intervention. Springer, 2016, pp. 424–432.

[18] N. Houlsby, F. Huszar, Z. Ghahramani, and M. Lengyel, “Bayesian ac-´ tive learning for classification and preference learning,” arXiv preprint arXiv:1112.5745, 2011.

[19] D. Carter, “Amazon FBA box size rules: Every limit sellers need to know in 2026,” https://goaura.com/blog/amazon-fba-box-size, Apr. 2026, aura. Accessed: September 14, 2026.

[20] V. Makoviychuk, L. Wawrzyniak, Y. Guo, M. Lu, K. Storey, M. Macklin, D. Hoeller, N. Rudin, A. Allshire, A. Handa, et al., “Isaac gym: High performance gpu-based physics simulation for robot learning,” arXiv preprint arXiv:2108.10470, 2021.

[21] J. Wan, S. R. Sinclair, D. Shah, and M. J. Wainwright, “Exploiting exogenous structure for sample-efficient reinforcement learning,” arXiv preprint arXiv:2409.14557, 2024.

[22] T. Dietterich, G. Trimponias, and Z. Chen, “Discovering and removing exogenous state variables and rewards for reinforcement learning,” in International Conference on Machine Learning. PMLR, 2018, pp. 1262–1270.

## APPENDIX A

SIMULATION CALIBRATION AND HYPERPARAMETERS

Sim-to-sim calibration against MuJoCo. We used a system-identification-style paired-replay method: identical placement sequences were run in MuJoCo and Isaac Gym, and PhysX parameters were tuned to match per-placement stability verdicts. This increased agreement from 67.0% before calibration to 94.8% after calibration. MuJoCo served as the high-fidelity reference for sim-to-sim transfer; largescale training and evaluation use Isaac Gym.

Hyperparameters. Tab. VII lists the experiment settings and hyperparameters of SHORE. The upper block covers the environment and discretized REPPO learner (Sec. IV-B); the lower block covers the stability model and its labels.

TABLE VII: Experiment settings and hyperparameters grouped by function.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Simulation</td><td>5</td></tr><tr><td>Buffer size N</td><td>(25, 25) cm</td></tr><tr><td>Bin length and width  $( L _ { p } , W _ { p } )$ </td><td>20 cm</td></tr><tr><td>Maximum stacking height  $H _ { p }$ </td><td></td></tr><tr><td>Bin voxels  $( l _ { p } , w _ { p } , h _ { p } )$ </td><td>(25, 25, 20) (1 cm bins) None</td></tr><tr><td>Added placement noise Vertical release offset</td><td>2 mm</td></tr><tr><td></td><td>2  / 80 ms</td></tr><tr><td>Simulation timestep / settling time τ</td><td>2 cm</td></tr><tr><td>Stability threshold  $\epsilon _ { s }$ </td><td>16 / 4</td></tr><tr><td>PhysX solver iterations (pos. / vel.)</td><td></td></tr><tr><td>PhysX contact / rest offset</td><td>2 mm / 0</td></tr><tr><td>PhysX maximum depenetration speed</td><td>1 m/s</td></tr><tr><td>Simulation pool</td><td>288 environments</td></tr><tr><td>Action space and REPPO-D learner EMS candidates K</td><td>80</td></tr><tr><td>Discount γ / λ-return λ</td><td>1 / 0.95</td></tr><tr><td>Reward scale</td><td>10</td></tr><tr><td> $c _ { r }$  KL bound</td><td>0.1</td></tr><tr><td> $\varepsilon _ { \mathrm { K L } }$ </td><td>0.01 / 0.01</td></tr><tr><td>Initial entropy temperature α / KL multiplier</td><td></td></tr><tr><td>Entropy target  $\mathcal { H } _ { \mathrm { t g t } } ( s )$ </td><td> ${ \scriptstyle { \frac { 1 } { 2 } } } { \mathcal { H } } ( \pi _ { S } )$ </td></tr><tr><td>HL-Gauss atoms / support</td><td> $1 5 1 ^ { \circ } / \left[ - 4 0 , 4 0 \right]$ </td></tr><tr><td>Learning rate (AdamW) / gradient clip</td><td> $3 \times 1 0 ^ { - 4 } ~ / ~ 0 . 5$ </td></tr><tr><td>Rollout (environments × steps)</td><td> $3 2 \times 1 2 8$ </td></tr><tr><td>Epochs × minibatches (size)</td><td> $4 \times 8 ~ ( 5 1 2 )$ </td></tr><tr><td>Stability model and labels</td><td></td></tr><tr><td>Stability heads / U-Net maps, levels</td><td>5 / 8, 3</td></tr><tr><td></td><td></td></tr><tr><td>Simulated labels per step  $B _ { s } + B _ { r }$ </td><td>6 + 2</td></tr><tr><td>Label selection exponent  $\gamma _ { \pi }$ </td><td>0.5</td></tr><tr><td>Label weight (executed / simulated)</td><td>2 / 1</td></tr><tr><td>Holdout share of executed labels</td><td>0.1</td></tr><tr><td>Label buffer (states)</td><td>40,000</td></tr><tr><td>Stability updates per iteration (lr; batch)</td><td> $6 4 ~ ( 1 0 ^ { - 3 } ; 2 5 6 )$ </td></tr><tr><td>Activation gate: AUROC / held-out collapses</td><td> $0 . 8 ~ / ~ 5 0$ </td></tr><tr><td>Stability ramp length  $N _ { \mathrm { r a m p } }$ </td><td>105 env steps</td></tr><tr><td>Stability trust region  $\delta _ { S }$  / states</td><td>0.02 / 256</td></tr></table>