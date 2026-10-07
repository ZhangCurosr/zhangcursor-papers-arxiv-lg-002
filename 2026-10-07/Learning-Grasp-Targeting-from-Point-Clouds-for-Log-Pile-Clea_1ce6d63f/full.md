# Learning Grasp Targeting from Point Clouds for Log Pile Clearing on a Hydraulic Crane

George Sideris<sup>1</sup>, Lucas Bessai<sup>1</sup>, Heshan Fernando<sup>2</sup>, Elie Ayoub<sup>2</sup>, Nicolas Lemieux<sup>2</sup> and Inna Sharf<sup>1</sup>

Abstract— In mill yards, log loaders clear dense piles by a sequence of bundle grasps: hundreds of logs rest in contact, and each removal changes the pile available to the next grasp. A learned policy chooses where to place and orient the grapple from unsegmented point clouds and runs on a trailer-mounted hydraulic forestry crane. The policy classifies at which observed point to grasp and predicts depth and grapple orientation there. The same network outputs support behavior cloning (BC), reinforcement learning (RL), and deployment. BC learns from successful top-of-pile demonstrations; RL explores for improvements by fine-tuning the cloned policy (BC→RL) or by training from scratch. In simulation, BC clears 98 of 100 piles of 200 logs, while BC→RL improves load stability. Twelve field trials compare a geometric heuristic, RL from scratch, BC, and BC→RL through complete grasp-transport-deposit cycles. BC and BC→RL deposit 93.8% and 88.9% of pooled inventory, against 80.4% for the heuristic. BC→RL deposits logs on 83.6% of its cycles, against 79.6% for the heuristic and 65.7% for BC, while its simulated stability gain does not carry over to the crane testbed. Trained entirely in simulation and run unchanged on the crane, the learned policies clear more than the hand-filtered heuristic while observing unfiltered clouds that still contain the storage rack’s rails and poles.

## I. INTRODUCTION

Log handling is repetitive work in forestry and mill yards, where automation could ease operator shortages and reduce exposure to unstable loads [1], [2]. A grapple-equipped crane repeatedly grasps and lifts bundles from a dense pile, in which hundreds of logs rest against one another, then carries them to the mill’s infeed deck. Each such grasp, carry, and deposit forms a cycle, and every cycle changes the scene for the next decision: closing the grapple moves the captured bundle and the surrounding material, and as the pile depletes, broad mounds give way to isolated logs beside rails and poles.

The decision studied here is where to place and orient the grapple, given a partial, noisy point cloud of the pile. The learned policy runs on a trailer-mounted hydraulic forestry crane with a passively suspended grapple (Fig. 1). A common controller executes the resulting grasp, transport, and deposition; learning concerns only the grasp decision.

A top-of-pile rule is the natural start, but a camera cloud requires distinguishing logs from rack structure and stray returns; geometric filters depend on scene geometry and calibration, and a segmentation network needs its own labels. Learning the cloud-to-grasp mapping trains material selection and grasp prediction together, without segmentation.

Behavior cloning (BC) learns from successful grasps of a simulated expert that knows the log poses and targets the highest log. Reinforcement learning (RL) explores for improvements: fine-tuning seeks better bundle handling, while RL from scratch explores for more effective grasp sequences beyond the expert’s top-of-pile strategy.

The contributions of this paper are:

• A learned grasp policy for sequential clearing of packed log piles from unsegmented point clouds. It classifies at which observed point to grasp and predicts the depth and the yaw, the grapple’s rotation about the vertical axis, at that point; one network serves BC, RL fine-tuning, and deployment (Fig. 2). The point selection matters: a regression head trained on the same demonstrations fails to clear the piles (Sec. V-B).

• Twelve field trials on the crane testbed in which BC and BC→RL, trained entirely in simulation and run with unchanged weights on unfiltered clouds, deposit more inventory than the deployed geometric heuristic on two of three pile shapes and overall.

• A comparison of the geometric heuristic, RL from scratch, BC, and BC→RL in simulation and on the crane testbed showing that the learned policies’ clearing advantage carries over while fine-tuning’s handling gain does not, with field failures classified by cause and trial phase.

![](images/6b26ada5e0b978376b243d91b1e4039fa998417fe112db47c59ffba31d122c5d.jpg)  
Fig. 1. Trailer-mounted hydraulic forestry crane, suspended grapple, source rack, and destination trailer. The policy selects the grasp; a common controller transports and deposits the logs.

![](images/1673b020b042d62c8cc6eba075e317faf19d27f742d8ea7e2ebaa936938dac89.jpg)  
Fig. 2. Shared scoring policy through BC, RL, and deployment. Orange blocks are trained; blue blocks carry weights unchanged. Gray arrows carry weights between phases. The RL panel shows fine-tuning from BC; the dashed arrow is BC deployed directly, skipping (b). The privileged state $\xi _ { k }$ contains the positions and yaws of up to 64 logs and is available only to the critic during training. RL from scratch also trains the encoder, enters (b) without BC weights, and deploys both of its blocks. MLP: multilayer perceptron; PPO: proximal policy optimization.

## II. RELATED WORK

Forestry automation includes crane motion planning [3], [4], learned control [5], vision-guided single-log grasping [6], and loading by an unmanned forwarder [7]. Learned grasp planning has also been evaluated for small log clusters on a real loader [8], for groups of two to five logs [9], and for clusters of up to seven [10]. These works automate motion for a given target or plan one grasp on a static scene; none addresses sequential clearing of a pile whose state evolves under its own actions or reports pile-scale hardware trials.

Learned grasping addresses isolated objects and clutter [11], [12]. Sequential bin picking [13] and push-grasp policies [14] explicitly account for changes induced by earlier actions. That literature assumes loose, separable clutter and pinch grasps that singulate one object; mill-yard logs are packed in layers, nothing can be singulated, and the grapple takes bunches in one power grasp that reshapes the pile.

PointNet [15] provides per-point and global cloud features; point-cloud policies support imitation [16] and simulationto-real RL [17], and HACMan [18] selects point-cloud contact locations with continuous motion parameters for nonprehensile manipulation. Demonstration-initialized RL [19], proximal policy optimization (PPO) [20], and critics with privileged observations [21] are the learning tools.

## III. METHOD

## A. Grasp representation

At each cycle, indexed by k, the policy receives a point cloud $P = \{ p _ { i } \}$ , where $p _ { i }$ is the three-dimensional position of observed point i. It chooses a grasp position $t = ( x , y , z )$ and yaw ψ in the crane base frame. A fixed crane controller executes the target; the next cloud shows the changed pile.

A pile can present several separated graspable regions. Regressing a single coordinate with a squared-error loss can average those alternatives into a gap. Classifying over observed points preserves separate candidates. Section V-B tests this choice against a regression head on the same data and backbone.

A PointNet encoder applies a shared multilayer perceptron (MLP), 3 → 64 → 128 → 256 with batch normalization and an exponential linear unit after each layer, to every point, giving the per-point feature $h _ { i }$ . Max pooling over points and a 256 → 256 projection with the same normalization and activation produce the global feature g. A second shared MLP reads, for every point, the concatenation $[ h _ { i } ; g ; p _ { i } ]$ of its 256-wide feature, the 256-wide global feature, and its position, 515 values in all, and three linear outputs on its last layer predict, at that point, selection score $s _ { i } .$ , vertical offset $\delta _ { i } .$ , and doubled-angle yaw encoding $q _ { i } ~ = ~ ( q _ { i } ^ { c } , q _ { i } ^ { s } )$ Figure 2 gives all actor widths and outputs. The head input has dropout probability 0.2 during BC training.

Targets are restricted to a predefined grasping volume within the source rack, called the action box (Fig. 3, middle). Padded input slots and points outside that volume are masked. At deployment, the highest-scoring unmasked point, with index ˆı, determines the target position t<sup>ˆ</sup> and yaw ψ<sup>ˆ</sup>:

$$
\begin{array} { r l } & { \hat { \imath } = \arg \operatorname* { m a x } _ { i } s _ { i } , } \\ & { \hat { \imath } = ( p _ { \hat { \imath } , x } , p _ { \hat { \imath } , y } , p _ { \hat { \imath } , z } + \delta _ { \hat { \imath } } ) , } \\ & { \hat { \psi } = \frac { 1 } { 2 } \arg \operatorname* { m a x } _ { i } q _ { \hat { \imath } } ^ { s } , q _ { \hat { \imath } } ^ { c } ) . } \end{array}\tag{1}
$$

The doubled-angle yaw respects the equivalence of grasps separated by half a turn [11]. The vertical offset allows the target to lie below the selected return. Selecting among observed points restricts horizontal targets to measured surfaces, which can include rack structure and noise. The policy must therefore learn which material is graspable (Fig. 4).

## B. Behavior cloning

A privileged expert selects the highest simulated log and aligns the grapple with its axis. Its successful cycles supply 11,366 cloud–target demonstrations from 500 simulated pileclearing sequences, or episodes. Two commissioning probes on the crane found that surface-level commands raked the pile, while targets 0.25–0.30 m below the surface closed on material. The recorded targets t are log centers; their height $t _ { z }$ is relabeled to the training label $t _ { z } ^ { \prime }$ with depth $d = 0 . 2 5$ m:

$$
t _ { z } ^ { \prime } = \operatorname* { m a x } ( t _ { z } + r _ { \log } - d , \ z _ { \mathrm { l a b e l } } ) ,\tag{2}
$$

where $r _ { \mathrm { l o g } } = 0 . 0 5 6$ m estimates the surface above the log center and $z _ { \mathrm { l a b e l } } = - 1 . 2 0$ m is the lowest label allowed. These relabeled depths train BC and initialize its RL finetuning, BC→RL; RL from scratch never sees them.

Let $i ^ { * }$ index the point horizontally nearest the expert target. The target distribution assigns equal probability y<sub>i</sub> to points within 0.12 m of the target, always including $i ^ { * }$ and zero elsewhere. The BC loss combines point selection, depth, and yaw errors with a rack-pole penalty:

![](images/77b74cf0608fd8415a4ba2362250d2f534ec02217bbc8c192b9918ebbdf5fbfd.jpg)  
Fig. 3. Simulation: parallel environments (left), rack and frames (middle), and actuated links with passive suspension (right). ψ is grapple yaw.

![](images/2bdd547eacd9a38486d137f348debce520504fe76d4e64f3d2acfce8c80ad337.jpg)  
Fig. 4. Scores on simulated (left) and crane (right) clouds. Brightness gives score rank, gray marks points outside the action box, and red gives the decoded target and yaw.

$$
\begin{array} { r l } & { \mathcal { L } = \displaystyle - \sum _ { i } y _ { i } \log \operatorname { s o f t m a x } ( s ) _ { i } + \mathcal { L } _ { \delta } + 0 . 5 \mathcal { L } _ { q } + \mathcal { L } _ { \mathrm { n e g } } , } \\ & { \mathcal { L } _ { \delta } = \big ( \delta _ { i ^ { * } } - ( t _ { z } ^ { \prime } - p _ { i ^ { * } , z } ) \big ) ^ { 2 } , } \\ & { \mathcal { L } _ { q } = \| q _ { i ^ { * } } - ( \cos 2 \psi , \sin 2 \psi ) \| _ { 2 } ^ { 2 } . } \end{array}\tag{3}
$$

Pole columns are shortened randomly, and $\mathcal { L } _ { \mathrm { n e g } }$ averages $\log ( 1 + e ^ { s _ { i } } )$ over their remaining points to discourage selection of visible rack poles. Joint horizontal translations move cloud and target together. Camera-ray noise uses standard deviation $\sigma ( r ) = 0 . 0 0 1 4 r ^ { 2 }$ m at range r in meters [22].

BC uses AdamW with initial learning rate and weight decay $1 0 ^ { - 4 }$ , batch size 64, and gradient-norm clipping at 1.0. The learning rate halves when the loss on a held-out 15% validation split of the demonstrations plateaus; early stopping retains the saved network weights, or checkpoint, with the lowest validation loss.

## C. Reinforcement learning

The expert selects the highest log without optimizing the resulting bundle. RL explores using a reward that values both the number of logs and their handling. During training, policy π samples point index i, depth δ, and yaw encoding q from cloud P:

$$
\begin{array} { r } { \pi ( i , \delta , q \mid P ) = \operatorname { C a t } ( i \mid \operatorname { s o f t m a x } ( s / \tau ) ) \phantom { x x x x x x x x x x x x x x x x x x x x x x x x x x x } } \\ { \cdot \mathcal { N } ( \delta \mid \delta _ { i } , \sigma _ { \delta } ^ { 2 } ) \mathcal { N } ( q \mid q _ { i } , \sigma _ { q } ^ { 2 } I _ { 2 } ) . } \end{array}\tag{4}
$$

Temperature $\tau ~ = ~ 1 . 0$ sets the score distribution; learned $\sigma _ { \delta }$ and $\sigma _ { q }$ control depth and yaw exploration, and $I _ { 2 }$ is the identity matrix. Sampling a point concentrates horizontal exploration on observed surfaces; Section V-B compares this with a Gaussian policy over grasp coordinates, which can explore anywhere in the box.

The cycle reward R combines the number of logs held in the grapple after lifting, n, their alignment score relative to the tong axis, $\alpha ,$ and grapple stability, ς:

$$
R = \left\{ { \begin{array} { l l } { n \alpha \varsigma , } & { n > 0 , } \\ { - 1 , } & { n = 0 . } \end{array} } \right.\tag{5}
$$

Both handling scores are defined in Sec. IV-B. Aligned, level loads earn more; an empty cycle is charged.

Each grasp affects later rewards by changing the pile, and PPO maximizes the return discounted with $\gamma = 0 . 9 9$ over an episode of at most H = 30 cycles. An asymmetric critic estimates remaining return from privileged state $\xi _ { k } \in \mathbb { R } ^ { 2 5 6 }$ which contains the positions and yaws of up to the 64 highest logs, sorted by height. The actor sees only the point cloud, while the critic receives the privileged state; this difference defines the asymmetric actor–critic. Only the actor deploys.

Fine-tuning freezes the BC encoder and normalization statistics, then updates the head with narrow exploration and no entropy bonus. RL from scratch trains the full architecture with broader exploration and an entropy bonus. The different recipes suit each strategy, so the comparison between finetuning and RL from scratch does not isolate the effect of BC initialization.

Both use 40 parallel environments on one 48 GB-class cloud GPU per run (NVIDIA A40 or comparable), general ized advantage estimation with trace parameter $\lambda = 0 . 9 5$ four minibatches per update, and gradient-norm clipping at 1.0. Fine-tuning uses policy-ratio clip $\epsilon = 0 . 1$ , a fixed learning rate of $5 \times 1 0 ^ { - 5 }$ , four epochs, eight rollout cycles per environment, entropy coefficient $\beta = 0$ , and initial depth and yaw scales $\sigma _ { \delta } = \sigma _ { q } = 0 . 0 5$ . RL from scratch uses $\epsilon = 0 . 2$ an initial learning rate of $3 \times 1 0 ^ { - 4 }$ , five epochs, sixteen rollout cycles per environment, $\beta = 0 . 0 1$ , and $\sigma _ { \delta } = \sigma _ { q } = 0 . 1 5$

## IV. EXPERIMENTAL SETUP

## A. Physical system and observations

A common finite state machine executes every target. GAZE holds a settled viewing pose; HOVER moves above the target; ALIGN sets yaw; DESCEND lowers the grapple; CLOSE closes the tongs; and LIFT raises the bundle. Inverse kinematics supplies joint setpoints to the programmable logic controller. The crane then transports at constant height and releases above the camera-measured fill height of the trailer compartment; these engineered motions deposit a stable, well-aligned bundle reliably. A cycle succeeds only after depositing at least one log. Simulation assesses and removes held logs after the lift.

The action box covers the rack’s $2 . 0 \times 7 . 0$ m footprint and 1.4 m height; commands are clamped at the rack floor. A mast-mounted Stereolabs ZED X stereo camera<sup>1</sup> supplies depth, and simulation matches its resolution and projection. Each stream becomes a crane-frame cloud cropped to the action box plus a 0.5 m horizontal margin and sampled to 2048 points. Clouds with fewer points are zero-padded and masked; the margin exposes rails, poles, and ground.

The camera mount is calibrated against the crane’s own grapple: recorded joint angles place its CAD model by forward kinematics, and the mount transform is fitted to the observed grapple by iterative closest point [23] with a floor-levelness term; fiducial markers [24] on the rack posts, refined on the post columns of the cloud, locate the rack. Simulation uses this viewing geometry (Fig. 5).

## B. Simulation and grasp outcomes

Training uses Isaac Lab [25], built on Isaac Sim, chosen for GPU-parallel PhysX rigid-body dynamics of hundreds of contacting logs across many environments and for on-GPU depth rendering that matches the deployed camera. The crane model includes actuated arm joints, grapple yaw and tongs, and passive suspension. Log geometry follows ten physical measurements: mean diameter 0.113 m, length 2.45 m, and taper approximately 1 cm of diameter per meter. Mass follows mesh volume at a density of 850 kg/m<sup>3</sup> (green softwood) [26]. Six size variants populate each 200-log episode, settled under gravity into randomized flat, mound, double-mound, ramp, or jagged profiles. Log contacts use published wood-on-wood friction coefficients [27]. Contact and suspension parameters are fixed.

A simulated log is counted as held when its center is within 1.5 m of the grapple after lifting. For n held logs, alignment α uses the misalignment angle $\theta _ { \alpha _ { i } }$ between the unit tong axis ${ \hat { y } } ^ { \mathbf { g } }$ , the grapple axis along which a held log lies when the tongs close squarely on it, and each unit log axis $\hat { y } _ { i } ^ { \ell }$ ; stability ς uses the grapple tilt angle $\theta _ { \varsigma }$ (Fig. 6):

$$
\alpha = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \cos ^ { 8 } \theta _ { \alpha _ { i } } , \qquad \varsigma = \operatorname * { m a x } ( 0 , \cos \theta _ { \varsigma } ) ^ { 4 } .\tag{6}
$$

Here $\theta _ { \varsigma }$ is the mean tilt over a one-second post-lift window, taken after settling or at the hold timeout. Both scores lie in [0, 1]; the exponents sharpen them so that only nearly aligned logs and a nearly level load score close to one (Fig. 6, insets).

## C. Compared policies and training

BC (scoring BC), BC→RL, and RL from scratch share the scoring head. The geometric heuristic is the rule-based baseline deployed on the crane. It applies the expert’s topof-pile strategy to the cloud: within a tight 1024-point crop with hand-placed pole exclusions, it takes the highest point in the middle half of the rack width that has three neighbors within 0.15 m, fits a local log axis for yaw or falls back to the rack direction, and grasps below that point at the expert’s 0.25 m in simulation and a field-tuned 0.30 m on the crane.

![](images/3c782e2a13be2e4eb85eb182b46b3037b92207d9cb69cb926c44cf64f7eb2a51.jpg)

Fig. 5. Mast-camera views on the crane (left) and in simulation (right), using the calibrated camera and rack placement.  
![](images/8f15b214314ffa80f8a565db97392c11bea0737d2f8e65e965c1435592b8fbca.jpg)  
Fig. 6. Alignment and stability. Superscripts w, g, and ℓ denote world, grapple, and log frames; insets show the shaped scores in (6).

Two alternative heads isolate the representation (Sec. V-B): under BC, a regression head (regression BC) maps the global feature $g$ to one box-normalized position and yaw by squared error, with the demonstrations, preprocessing, backbone, and optimizer of the scoring head unchanged; under RL from scratch, a Gaussian head over grasp coordinates replaces categorical point selection, sharing environment and reward.

Fine-tuning and RL from scratch use three training seeds and 50k cycles per run. Checkpoints are screened over 40 deterministic episodes, and the one emptying most racks receives the 100-episode evaluation and deployment.

## D. Evaluation protocols

In simulation, six policies run over 100 episodes on shared pile seeds from the training distribution. Episodes end at an empty rack or horizon H = 30 cycles. Deterministic evaluation selects the highest-scoring point and mean depth and yaw. Full counts emptied racks; success requires lifting at least one log. Removed inventory excludes logs knocked out of bounds. The metric $c _ { 9 5 }$ counts cycles until 95% of the initial inventory has been removed. Its mean includes only episodes reaching that threshold: 99 for the expert, 46 for regression BC, and 100 for every other policy. Cycles to termination count grasp decisions until the episode ends.

On the physical crane testbed, the geometric heuristic, BC, BC→RL, and RL from scratch each run once on a single mound, double mound, and flat rack, giving twelve trials. Figure 7 shows these shapes in simulation and on the testbed.

Rebuilt piles share shape categories but are not identical paired states. Reconstructed initial inventories range from 199 to 219 logs. Analysis ends at the first empty rack, three consecutive empty cycles, or 30 cycles. Results for RL from scratch include manual depth offsets (Sec. V-D).

Clear is the share of initial inventory deposited in the trailer; knocked-off logs are not cleared. An empty rack can therefore have Clear below 100%. Here $c _ { 9 5 }$ counts cycles until 95% is deposited, and n, logs per grasp, is the mean over a trial’s successful cycles. Because a closed grapple hides individual poses, recordings score alignment from drooping or falling logs (1), poor (2), acceptable (3), good (4), to a compact, well-aligned bundle (5). Pooled rows treat the three trials as one campaign: Succ. and Clear are ratios of the summed counts, Cycles is the total, and $n ,$ alignment, and ς average over all successful cycles; $c _ { 9 5 }$ is per trial only.

## V. RESULTS

## A. Imitation, exploration, and simulated clearing

BC and its fine-tuned policy achieve the highest fullclearing rates on the shared simulated piles (Table I). BC training reduces loss and median horizontal grasp error (Fig. 8). In training, fine-tuning reaches higher mean return and handling than learning from scratch (Fig. 9).

Scoring BC finishes more racks than its privileged expert. Unfinished expert episodes retain a median of two logs, typically isolated against the floor. The common closing sweep can miss the same log repeatedly, leaving the height rule to select it again. BC learns only from successful expert cycles, which may explain its stronger completion.

Fine-tuning changes handling more clearly than comple tion. It clears two paired piles that BC misses and misses one that BC clears. Stability rises by 0.035 ± 0.003 (paired mean ± standard error), knocked-off logs approximately halve, and cycle success falls 2.8 ± 0.8 percentage points. The reward permits this trade-off between grasping more often and carrying a more stable load. The c<sub>95</sub> and termination columns separate bulk removal from clearing the last remaining logs.

Figure 10 illustrates the regional strategy discovered by RL from scratch, which works one area down before moving on. Consecutive targets in the first five cycles are a median 0.4 m apart, against 1.5 m for the expert and fine-tuned policy, and its targets remain farther from the highest point over complete episodes. Exploration departs from the demonstrated strategy, but under the tested reward and budget it does not improve completion over BC and has the weakest handling.

## B. Representation and design ablations

The controlled BC comparison tests whether learning a score at each observed point improves on directly predicting grasp coordinates. Across three training seeds, the scoring head empties 98, 98, and 99 of 100 racks; the regression head empties 33, 0, and 0. For the regression run in Table I, unfinished episodes retain a median 45 logs, so regression fails while much of the pile remains, well before the difficult final remainder.

TABLE I. Simulation on 100 shared piles (mean ± standard deviation). n: logs per successful grasp; α: alignment; ς: stability; Succ.: successfulcycle rate; Full: emptied-rack rate; c<sub>95</sub>: cycles to 95% removal in episodes reaching it.
<table><tr><td>Method</td><td>n</td><td>α</td><td>S</td><td>Succ. (%)</td><td>Full(%)</td><td> $c _ { 9 5 }$ </td><td>Cycles</td></tr><tr><td>Privileged expert</td><td>11.51 ± 1.18</td><td>0.977 ± 0.010</td><td>0.948 ± 0.016</td><td> $7 6 . 3 \pm 1 6 . 3$ </td><td>63</td><td> $1 6 . 1 \pm 1 . 7$ </td><td>23.9 ± 5.2</td></tr><tr><td>Geometric heuristic</td><td> $1 1 . 2 3 \pm 1 . 0 5$ </td><td>0.974 ± 0.010</td><td>0.931 ± 0.017</td><td> $7 9 . 8 \pm 9 . 9$ </td><td>89</td><td> $1 6 . 9 \pm 1 . 6$ </td><td>22.9 ± 3.7</td></tr><tr><td>RL from scratch</td><td> $1 1 . 0 1 \pm 1 . 1 5$ </td><td>0.950 ± 0.016</td><td> $0 . 8 6 2 \pm 0 . 0 2 0$ </td><td> $8 1 . 3 \pm 9 . 6$ </td><td>87</td><td> $1 8 . 3 \pm 2 . 1$ </td><td>23.0 ± 4.1</td></tr><tr><td>BC, regression head</td><td> $1 0 . 6 5 \pm 1 . 9 2$ </td><td> $0 . 9 7 9 \pm 0 . 0 1 1$ </td><td>0.951 ± 0.015</td><td> $6 0 . 8 \pm 1 9 . 8$ </td><td>33</td><td> $2 0 . 4 \pm 4 . 6$ </td><td>27.5 ± 4.1</td></tr><tr><td>BC, scoring head</td><td> $1 1 . 7 7 \pm 1 . 0 0$ </td><td>0.980 ± 0.008</td><td>0.920 ± 0.024</td><td> $9 1 . 1 \pm 6 . 2 $ </td><td>98</td><td> $1 5 . 2 \pm 1 . 4$ </td><td>18.9 ± 2.5</td></tr><tr><td>BC→RL</td><td> $1 1 . 8 9 \pm 0 . 8 1$ </td><td>0.984 ± 0.006</td><td>0.955 ± 0.017</td><td> $8 8 . 3 \pm 5 . 6 $ </td><td>99</td><td> $1 6 . 2 \pm { 1 . 3 }$ </td><td>19.2 ± 1.9</td></tr></table>

Squared-error regression can average alternative expert targets. Between two similar mounds, this produces a target with no material beneath it. Across 1784 recorded decisions from three regression runs, 14.6–24.7% have no observed point within 0.5 m horizontally; 5.1–8.6% reissue such a target within 0.15 m of the previous one. Scoring BC has neither diagnostic failure over 2140 decisions.

A separate three-seed, 25k-cycle comparison of RL from scratch holds the environment and reward fixed while chang ing categorical point selection to Gaussian coordinate exploration. Table II separates sampled-action training performance from deterministic completion at each seed’s best screened checkpoint; Fig. 11 shows return. The categorical head’s screened completion rates are 97.5%, 57.5%, and 0%; the Gaussian head empties no rack on any seed. This supports exploring only observed points, although learning from scratch varies widely by seed.

The reward ablation compares (5) with three alternatives for successful cycles: normalized multiplication $1 0 f \alpha \varsigma$ and normalized addition $f + \alpha + \varsigma ,$ , which reward taking what the target offers instead of seeking the tallest stack, and log count n, which drops the quality terms. Here $f = \mathrm { m i n } ( n / \tilde { n } , 1 )$ and n˜ is the number of logs stacked at the target, capped at 15. Every variant keeps the empty-cycle penalty. The deployed reward gives the highest mean training clearing; removing quality terms increases bundle size and lowers stability (Table II, Fig. 12).

The fine-tuning ablations change either encoder freezing or the critic’s observation. Leaving the encoder trainable lowers training return and widens the observed spread in screened completion, with rates from 72.5% to 100%. With the encoder frozen, the critic given privileged log poses finishes with higher mean training return and a narrower late seed band than the critic on the actor’s point cloud (Fig. 11).

Offline replay compares regression BC with two scoring variants on recorded crane clouds (Fig. 13). One variant was trained on tight rack crops, the other on the 0.5 m margin crop that exposes surrounding structure. Regression again predicts intermediate targets between mounds, while margintrained scoring chooses a mound point. Tight-crop scoring repeatedly selects rails and poles it never saw in training. This motivated deploying the margin-trained policy.

## C. Clearing and handling on the crane testbed

BC recovers the most pooled inventory; fine-tuning gives the highest cycle success (Table III). Figure 14 gives each trial; Fig. 15 the pooled progression and empty-cycle causes.

BC and BC→RL deposit more inventory than the geometric heuristic on two shapes and over the pooled campaign.

Privileged expert  
![](images/997d918059da03899e1d179b0b86a0a4ece637aebdad8d2ded33bf1c4bf39303.jpg)

![](images/efbd38e090436063370b7da1e4b28dbf7c7ec90563e61558e12c39b46d164874.jpg)

![](images/f6546adf3c38c0a51d331ca56dc83e8b6f2ab1d12a1b768fa175c4337a9b33ca.jpg)

![](images/cb455aab51b96acfbdc97e1b4f7d49baa545d7e22e7ccd5f975f5e3bdc4fe728.jpg)

![](images/86c3466644b7f5e0b4cd81bcb7c9e2e27c6438359e9f21c2b03020ad48ad02b4.jpg)

![](images/0db740d0aac81c3419e9a6ad3645fddd08bcfb541dfe50278a3fd19f91165a24.jpg)  
Fig. 7. Single mound, double mound, and flat rack in simulation (top) and before crane trials (bottom). Categories match, but each physical pile is rebuilt The source rack is in front of the destination trailer.

![](images/3ab729389c74313ef2617eec9e61ba615523e22546ada50c2a6bf4b6a2076ff5.jpg)

![](images/2293e97639df43c6fbcbc7b35e583b0a22361e4cb0e398a66ed3ba5a8ca24952.jpg)

Fig. 8. BC loss (left) and validation grasp error (right), the median horizontal distance from the decoded grasp position to the expert target, mean and standard deviation over three seeds. Training uses fresh camera noise; validation is clean. Diamonds mark the deployed checkpoint.  
![](images/179449bf725c5a9d1684f4c28657bbc8517d086e4ced1bca5bc1e54b045efcb7.jpg)

![](images/ae567681aad63192fa15b30bce5983b5492dc1a1bccd8290e9fe436b3022e383.jpg)  
Fig. 9. RL return and handling, mean and standard deviation over three seeds. Training samples actions; evaluation uses the highest score and predicted depth and yaw. Diamonds mark checkpoints.

The flat rack makes the difference clearest: both learned policies empty it, while the heuristic stops at 62.0% Clear. The double mound reverses it: the heuristic deposits everything.

RL from scratch falls behind early on the double mound. A matched first-fifteen-cycle comparison, reached by every heuristic, BC, and fine-tuned trial, gives success rates of 80%, 78%, and 89%, respectively. The heuristic and finetuned rates are within two percentage points of their simulated rates. This early-window agreement does not establish that later clearing or handling transfers equally well.

The fine-tuned policy’s simulated stability advantage does not transfer: its stability is below BC and the heuristic on every shape, by 0.02 to 0.06, about the size of its simulated gain. RL from scratch has the lowest stability on all three shapes, with the lowest pooled alignment and logs per successful cycle. Observed handling failures include grasps near log ends and loss of part of the bundle during lifting.

Fig. 10. Targets from the expert and RL from scratch on one shared pile. Red points mark targets by cycle; color gives remaining height. Reported spacing medians use all episodes.  
TABLE II. Three-seed ablations (mean ± standard deviation). Clear and handling use sampled cycles 12.5k–25k; Argmax full uses the best 40- episode deterministic screen per seed.
<table><tr><td>Variant</td><td>Clear(%)</td><td>n</td><td>α</td><td>S</td><td>Argmax full (%)</td></tr><tr><td>Output head</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Categorical over points (used)</td><td>94.0 ± 4.8</td><td>9.3 ± 1.0</td><td>0.79 ± 0.04</td><td>0.69 ± 0.03</td><td>51.7 ± 40.0</td></tr><tr><td>Gaussian over coordinates</td><td>90.6 ± 1.2</td><td>9.9 ± 0.1</td><td>0.84 ± 0.01</td><td>0.65 ± 0.01</td><td>0.0 ± 0.0</td></tr><tr><td>Reward</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Multiplicative, unnormalized (used)</td><td>94.0 ± 4.8</td><td>9.3 ± 1.0</td><td>0.79 ± 0.04</td><td>0.69 ± 0.03</td><td>51.7 ± 40.0</td></tr><tr><td>Multiplicative, normalized</td><td>76.4 ± 14.5</td><td>8.8 ± 0.6</td><td>0.83 ± 0.02</td><td>0.70 ± 0.00</td><td>5.8 ± 5.1</td></tr><tr><td>No quality terms</td><td>86.2 ± 18.0</td><td>11.2 ± 0.7</td><td>0.84 ± 0.02</td><td>0.63 ± 0.02</td><td>30.8 ± 38.3</td></tr><tr><td>Additive, normalized</td><td>90.4 ± 10.6</td><td>8.9 ± 0.7</td><td>0.81 ± 0.01</td><td>0.66 ± 0.01</td><td>13.3 ± 12.5</td></tr><tr><td>Fine-tuning recipe</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Frozen encoder, asymmetric critic (used)</td><td>100.0 ± 0.0</td><td>14.1 ± 0.7</td><td>0.86 ± 0.00</td><td>0.87 ± 0.02</td><td>96.7 ± 2.4</td></tr><tr><td>Encoder left trainable</td><td>99.9 ± 0.0</td><td>13.9 ± 0.8</td><td>0.87 ± 0.00</td><td>0.86 ± 0.03</td><td>90.0 ± 12.4</td></tr><tr><td>Critic on the point cloud</td><td>100.0 ± 0.0</td><td>13.7 ± 0.6</td><td>0.86 ± 0.01</td><td>0.88 ± 0.03</td><td>95.0 ± 7.1</td></tr></table>

TABLE III. Twelve crane trials. n: logs per depositing cycle; Align.: 1–5 bundle score; ς: stability over valid successful cycles; these three are mean ± standard deviation over the trial’s successful cycles; Succ.: depositing-cycle rate; Clear: deposited inventory. RL from scratch includes depth offsets.
<table><tr><td>Policy</td><td>n</td><td>Align.</td><td>S</td><td>Succ. (%)</td><td>Clear (%)</td><td>C95</td><td>Cycles</td></tr><tr><td>Single mound</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Heuristic</td><td>13.2 ± 4.2</td><td>4.42 ± 0.95</td><td>0.947 ± 0.049</td><td>80.0</td><td>79.0</td><td></td><td>15</td></tr><tr><td>RL from scratch</td><td>9.1 ± 5.6</td><td>4.05 ± 0.89</td><td>0.781 ± 0.116</td><td>86.4</td><td>86.5</td><td></td><td>22</td></tr><tr><td>BC</td><td>12.6 ± 6.2</td><td>4.23 ± 0.97</td><td>0.942 ± 0.033</td><td>72.2</td><td>82.4</td><td></td><td>18</td></tr><tr><td>BC→RL</td><td>12.5 ± 4.6</td><td>4.08 ± 1.07</td><td>0.918 ± 0.059</td><td>81.2</td><td>81.5</td><td></td><td>16</td></tr><tr><td>Double mound</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Heuristic</td><td>12.6 ± 7.1</td><td>4.44 ± 1.06</td><td>0.937 ± 0.032</td><td>84.2</td><td>100.0</td><td>17</td><td>19</td></tr><tr><td>RL from scratch</td><td>9.9 ± 6.0</td><td>2.90 ± 1.45</td><td>0.777 ± 0.155</td><td>71.4</td><td>49.5</td><td>一</td><td>14</td></tr><tr><td>BC</td><td>12.7 ± 6.3</td><td>4.00 ± 1.37</td><td>0.933 ± 0.055</td><td>56.7</td><td>98.6</td><td>22</td><td>30</td></tr><tr><td>BC→RL</td><td>13.2 ± 3.9</td><td>4.15 ± 0.95</td><td>0.898 ± 0.039</td><td>76.5</td><td>85.5</td><td></td><td>17</td></tr><tr><td>Flat rack</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Heuristic</td><td>11.3 ± 5.3</td><td>4.36 ± 0.77</td><td>0.969 ± 0.018</td><td>73.3</td><td>62.0</td><td></td><td>15</td></tr><tr><td>RL from scratch</td><td>10.3 ± 5.1</td><td>3.56 ± 1.17</td><td>0.732 ± 0.132</td><td>70.4</td><td>98.0</td><td>24</td><td>27</td></tr><tr><td>BC</td><td>12.9 ± 6.9</td><td>4.19 ± 0.88</td><td>0.952 ± 0.038</td><td>72.7</td><td>99.5</td><td>19</td><td>22</td></tr><tr><td>BC→RL</td><td>10.1 ± 6.4</td><td>4.10 ± 1.26</td><td>0.894 ± 0.083</td><td>90.9</td><td>99.5</td><td>18</td><td>22</td></tr><tr><td>Pooled</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Heuristic</td><td>12.4 ± 5.9</td><td>4.41 ± 0.95</td><td>0.949 ± 0.038</td><td>79.6</td><td>80.4</td><td></td><td>49</td></tr><tr><td>RL from scratch</td><td>9.8 ± 5.5</td><td>3.62 ± 1.21</td><td>0.763 ± 0.133</td><td>76.2</td><td>78.0</td><td></td><td>63</td></tr><tr><td>BC</td><td>12.8 ± 6.5</td><td>4.13 ± 1.11</td><td>0.942 ± 0.044</td><td>65.7</td><td>93.8</td><td></td><td>70</td></tr><tr><td>BC→RL</td><td>11.6 ± 5.5</td><td>4.11 ± 1.13</td><td>0.901 ± 0.067</td><td>83.6</td><td>88.9</td><td></td><td>55</td></tr></table>

![](images/6557a97dd1ab5176cf025c88e4e83941918556ca66cea4fb7b7ee1d067cb2860.jpg)

![](images/34d8cd3672a4d22ef954c5611f60120add02064e97837111385e9d7e77e8d341.jpg)

Fig. 11. Training return over three seeds: (a) categorical and Gaussian coordinate exploration; (b) encoder and critic variants during fine-tuning. Curves share reward within each panel.  
![](images/674b7e578290002e8d6c918317b92f46c1e3a9aeed8f5ee2176148790f085316.jpg)  
Fig. 12. Reward ablation, RL from scratch: clearing, bundle size, alignment, and stability, mean ± standard deviation over three seeds.

## D. Where physical clearing fails

Empty-cycle causes distinguish structure or noise targeting, closures blocked by too much material in the tongs (chokes), commands that close above material (depth faults), and control faults. Figure 15(b) groups these causes by trial third. All six heuristic structure/noise cycles occur in the final third. Pole returns survive the heuristic’s hand-placed, axis-aligned exclusion boxes because the poles are thick and the boxes sit on a calibration that drifts, and the cloud also carries stray noise returns. Once such points stand above the remaining logs, the height rule repeatedly chooses them; an empty grasp leaves little change to prompt a different target.

Learning does not stop structure targeting, but it ends the trap. BC has 13 structure/noise cycles in 70, against six in 49 for the heuristic. On the flat rack, however, BC returns to material after each of its three structure cycles and finishes. BC→RL has three structure cycles in 55, all on the base rails, the horizontal members at the rack ends, and never a pole. These match logs in diameter and height; the rack-pole penalty does not cover them. The real rack exposes ground between its base rails; the simulated rack has a solid floor at rail height.

Failures remain even when the target lands on logs. In its 30-cycle double-mound trial, BC has no gap-targeting cycle, yet produces four chokes on the steep mound, one control fault, and eight late structure cycles. The choked commands lie within the successful depth band of the campaign. On a steep flank, that depth can enclose more material than on a level pile, blocking closure; simulation shows no chokes.

![](images/990d67bfc96dd9b1cf3106489edc77b9b7e2877f7150999acd7c141d58dde0a9.jpg)  
Fig. 13. Offline targets on full (left) and depleted (right) clouds, colored by height: regression (black), margin-trained scoring (red), tight-crop scoring (blue). Inner box: tight rack crop; outer box: 0.5 m margin crop.

BC→RL completes thirteen successful single-mound cycles before closing above remaining logs on one side of the rack. The trial ends in that empty-cycle run. Residual camera and kinematic errors are plausible contributors, but this trial does not separate them from the policy’s depth prediction.

RL from scratch shows a systematic depth bias: with applied offsets removed, its mound targets lie a median 0.17 m above the observed surface. BC and BC→RL learn depth from demonstrations relabeled using crane commissioning; RL from scratch learns it only in simulation. The mound trials use a 0.15 m downward offset and still grasp near the surface. On the flat rack, the offset follows two above-surface closes and is increased later, giving a median depth of 0.21 m below the surface, within the other trials’ 0.18–0.33 m band. Its 98.0% Clear includes that adaptation, and it also deposits the most inventory on the single mound.

## VI. DISCUSSION

BC and BC→RL recover more inventory than the geometric heuristic on two pile shapes and across the pooled campaign while using a wider cloud without a geometric support filter or separate segmentation network. Structure selection still causes failures, particularly for BC, but on the flat rack BC returns to material after each structuretargeting cycle, while the heuristic’s repeated selection of residual structure ends its trial. Both were trained entirely in simulation and run unchanged on the crane, and still cleared more than the heuristic.

Neither RL recipe yields one best physical policy: BC recovers the most pooled inventory, and BC→RL has the highest pooled cycle success but loses its simulated stability advantage on the crane testbed.

![](images/c81bac0f40e695b4d835eed4fb3a42d1de3f159f1aa2f56f7e5b06ff127a86ed.jpg)  
Fig. 14. Cumulative logs moved in each field trial. Crosses mark empty cycles, dots a cleared rack; RL from scratch includes depth offsets.

![](images/40424db95109957fdbe00ac58fff7d52bdc3b6cf11474e21f5970bb333abc4bd.jpg)

![](images/e11ae7aaf37e49b7495cfb54db63f7fd5e9d38e1031cfd6bddbabeeddf5c7084.jpg)  
Fig. 15. Field campaign: (a) cumulative logs moved, mean ± standard deviation over one trial per shape, ended trials holding their final value; (b) empty-cycle causes by early (E), middle (M), and late (L) thirds.

Beyond log handling, the results motivate evaluating task completion and handling separately over the full depletion of a pile: better simulated handling need not yield better physical handling.

The evidence has limits: one crane and rack, one rebuilt pile per policy and shape. Three trials per policy cannot establish cross-platform performance or identify the cause of the stability result, although the ordering held on every shape.

Simulation misses physical chokes and the field stability ordering. Possible sources include the crane’s hydraulic and suspension dynamics, unmeasured friction, and the bark and knots of real logs, which the smooth rigid assets lack. Future work should measure how transfer depends on these contact parameters, correct the remaining depth faults with small amounts of real-trial data, extend the rack-pole penalty to the base rails, and randomize contact properties during training.

## VII. CONCLUSION

One point-selection, depth, and yaw network serves BC, RL, and deployment: it learns in simulation and executes complete crane cycles with unchanged weights. BC and BC→RL recover more inventory than the geometric heuristic overall with less hand-tuned filtering. Fine-tuning’s stability gain did not carry over to the crane testbed: handling is where the sim-to-real gap remains, and closing it, with real-trial data or more accurate crane dynamics modeling, is the next step.

## REFERENCES

[1] O. Lindroos, P. La Hera, and C. Haggstr ¨ om, “Drivers of advances in ¨ mechanized timber harvesting – a selective review of technological innovation,” Croatian Journal of Forest Engineering, vol. 38, no. 2, pp. 243–258, 2017.

[2] R. Visser and O. F. Obi, “Automation and robotics in forest harvesting operations: Identifying near-term opportunities,” Croatian Journal of Forest Engineering, vol. 42, no. 1, pp. 13–24, 2021.

[3] D. Ortiz Morales, S. Westerberg, P. X. La Hera, U. Mettin, L. Freidovich, and A. S. Shiriaev, “Increasing the level of automation in the forestry logging process with crane trajectory planning and control,” Journal of Field Robotics, vol. 31, no. 3, pp. 343–363, 2014.

[4] I. Jebellat, G. Sideris, R. Saif, and I. Sharf, “Designing experimental setup emulating log-loader manipulator and implementing antisway trajectory planner,” in 2025 IEEE International Conference on Robotics and Automation (ICRA), 2025, pp. 1488–1494.

[5] J. Andersson, K. Bodin, D. Lindmark, M. Servin, and E. Wallin, “Reinforcement learning control of a forestry crane manipulator,” in 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2021, pp. 2121–2126.

[6] T. Semberg, A. Nilsson, R. Bjorheden, and L. Hansson, “Real-time¨ target point identification and automated log grasping by a forwarder, using a single stereo camera for both object detection and boom-tip control,” Silva Fennica, vol. 58, 2024.

[7] P. La Hera, O. Mendoza-Trejo, O. Lindroos, H. Lideskog, T. Lindback,¨ S. Latif, S. Li, and M. Karlberg, “Exploring the feasibility of autonomous forestry operations: Results from the first experimental unmanned machine,” Journal of Field Robotics, vol. 41, pp. 942–965, 2024.

[8] E. Ayoub, P. Levesque, and I. Sharf, “Grasp planning with CNN for log-loading forestry machine,” in 2023 IEEE International Conference on Robotics and Automation (ICRA), 2023, pp. 11 802–11 808.

[9] E. Wallin, V. Wiberg, and M. Servin, “Multi-log grasping using reinforcement learning and virtual visual servoing,” Robotics, vol. 13, no. 1, 2024.

[10] A. Falldin, T. L¨ ofstedt, T. Semberg, E. Wallin, and M. Servin, “Syn-¨ thesizing multi-log grasp poses in cluttered environments,” Journal of Intelligent & Robotic Systems, vol. 112, p. 52, 2026.

[11] D. Morrison, P. Corke, and J. Leitner, “Closing the Loop for Robotic Grasping: A Real-time, Generative Grasp Synthesis Approach,” in Proc. of Robotics: Science and Systems (RSS), 2018.

[12] J. Mahler, J. Liang, S. Niyaz, M. Laskey, R. Doan, X. Liu, J. A. Ojea, and K. Goldberg, “Dex-Net 2.0: Deep learning to plan robust grasps with synthetic point clouds and analytic grasp metrics,” in Robotics: Science and Systems (RSS), 2017.

[13] J. Mahler and K. Goldberg, “Learning deep policies for robot bin picking by simulating robust grasping sequences,” in Proceedings of the 1st Annual Conference on Robot Learning, ser. Proceedings of Machine Learning Research, S. Levine, V. Vanhoucke, and K. Goldberg, Eds., vol. 78. PMLR, 13–15 Nov 2017, pp. 515–524.

[14] A. Zeng, S. Song, S. Welker, J. Lee, A. Rodriguez, and T. Funkhouser, “Learning synergies between pushing and grasping with selfsupervised deep reinforcement learning,” in 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2018, pp. 4238–4245.

[15] C. R. Qi, H. Su, K. Mo, and L. J. Guibas, “PointNet: Deep learning on point sets for 3D classification and segmentation,” in 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 77–85.

[16] Y. Ze, G. Zhang, K. Zhang, C. Hu, M. Wang, and H. Xu, “3D diffusion policy: Generalizable visuomotor policy learning via simple 3D representations,” in Robotics: Science and Systems (RSS), 2024.

[17] Y. Qin, B. Huang, Z.-H. Yin, H. Su, and X. Wang, “DexPoint: Generalizable point cloud reinforcement learning for sim-to-real dexterous manipulation,” in Conference on Robot Learning (CoRL), 2022.

[18] W. Zhou, B. Jiang, F. Yang, C. Paxton, and D. Held, “HACMan: Learning hybrid actor-critic maps for 6D non-prehensile manipulation,” in Proceedings of The 7th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 229. PMLR, 2023, pp. 241–265.

[19] A. Rajeswaran, V. Kumar, A. Gupta, G. Vezzani, J. Schulman, E. Todorov, and S. Levine, “Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations,” in Proceedings of Robotics: Science and Systems (RSS), 2018.

[20] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” CoRR, vol. abs/1707.06347, 2017.

[21] L. Pinto, M. Andrychowicz, P. Welinder, W. Zaremba, and P. Abbeel, “Asymmetric actor critic for image-based robot learning,” in Proceedings of Robotics: Science and Systems, 2018.

[22] L. E. Ortiz, E. V. Cabrera, and L. M. G. Gonc¸alves, “Depth data error modeling of the ZED 3D vision sensor from Stereolabs,” ELCVIA: Electronic Letters on Computer Vision and Image Analysis, vol. 17, no. 1, pp. 1–15, 2018.

[23] P. J. Besl and N. D. McKay, “A method for registration of 3-D shapes,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 14, no. 2, pp. 239–256, 1992.

[24] S. Garrido-Jurado, R. Munoz-Salinas, F. J. Madrid-Cuevas, and M. J. ˜ Mar´ın-Jimenez, “Automatic generation and detection of highly reliable´ fiducial markers under occlusion,” Pattern Recognition, vol. 47, no. 6, pp. 2280–2292, 2014.

[25] M. Mittal, P. Roth, J. Tigue et al., “Isaac Lab: A GPU-accelerated simulation framework for multi-modal robot learning,” arXiv:2511.04831, 2025.

[26] Forest Products Laboratory, “Wood handbook: Wood as an engineering material,” USDA Forest Service, Tech. Rep. FPL-GTR-282, 2021.

[27] E. A. Avallone, T. Baumeister III, and A. M. Sadegh, Eds., Marks’ Standard Handbook for Mechanical Engineers. McGraw-Hill, 2006.