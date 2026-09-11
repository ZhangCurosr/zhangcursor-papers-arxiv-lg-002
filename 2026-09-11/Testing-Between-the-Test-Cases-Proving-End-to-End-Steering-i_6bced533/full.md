# Testing Between the Test Cases: Proving End-to-End Steering in Conditions You Never Drove

Menuka Ghalan<sup>1</sup>, Charles Rodgers<sup>1</sup>, and Zachary D. Asher<sup>2</sup>

<sup>1</sup>Department of Computer Science, Western Michigan University, Kalamazoo, MI, USA

<sup>2</sup>Department of Mechanical and Aerospace Engineering, Western Michigan University, Kalamazoo, MI, USA

## Abstract

AI-based automated vehicle testing is challenging because a model that passes every test condition can still fail in the real world. Formal verification offers a way to directly address this gap. On a simulated highway and an arterial road we trained two small end-to-end steering networks each in CARLA, one on clear conditions alone and one on clear, fog, night and low sun. All four models were driven against a 2.19 ft lane-departure budget. Without driving again, we used bound propagation, a formal method that reads the trained weights, to compute how far steering can drift at every disturbance strength between two captured images. One calculation covers more than a campaign could drive: on the arterial it spans 133 poses, where ten intensities each would be 10<sup>133</sup> combinations, in minutes on one GPU. Not only did formal verification find conditions that broke the clear-trained policy without simulation testing, it provided some preliminary evidence for potential failures between the test cases. Our overall conclusion is that formal verification is a viable complement to simulation, and could be adopted as a part of verification and validation for automated driving.

## 1 Introduction

Autonomous vehicle (AV) development has moved from rulebased pipelines to data-driven ones, where safety and scalability have been the forces behind every shift along the way [1]. End-to-End (E2E) neural networks are the current expression of that shift. In an E2E model, a single network maps camera pixels directly to a driving command such as a behavior-cloned steering controller [2] or a reinforcement-learned lane follower [3]. The argument for E2E design is that optimizing the whole software stack against the driving objective avoids the errors that compound through separately trained perception, tracking and planning modules and they are easier to scale [4]. Modern E2E models now extend to jointly optimized stacks that still expose intermediate representations [5] and to vectorized planners built for efficiency [6]. The datasets used to train and benchmark these policies have grown accordingly [7].

Compliance is where the E2E design becomes awkward. ISO 26262 [8] allocates functional-safety requirements to components whose failure modes can be enumerated and it needs adapting before it can accommodate machine learning components [9]. ISO 21448 [10] addresses the Safety of the Intended Functionality (SOTIF), the hazards that remain when nothing has malfunctioned, while ISO 8800 [11] and UL 4600 [12] extend the argument to AI components specifically and to autonomous products as a whole. The difficulty common to all of them is that the E2E policies that perform best are the ones least amenable to the enumeration these standards assume.

The prevailing answer today is empirical, a test-and-patch cycle whose evidence operators assemble into an explicit safety case [13, 14]. Simulation carries much of that load, and the Car Learning to Act (CARLA) simulator is the standard open platform for it [15, 16]. What simulation cannot do is enumerate. Change one parameter and the whole campaign runs again. Data-driven policies generalize poorly on the long-tail scenarios that fall outside their training distribution [1]. A current solution is to collect dedicated adverse-condition datasets and utilize them in training [17, 18]. This has been extended more recently to generated inputs that enlarge the training distribution [19]. Camera-driven policies also face inputs no campaign samples so including adversarial examples that transfer across architectures is recommended [20]. Despite these efforts, unanticipated physical conditions still disrupt exactly the decision windows a statistical safety case assumes are covered, because no amount of input data covers the uncountably many ways a disturbance can manifest. The problem is not that the test cases are wrong, but that they are finite.

Formal verification is the alternative. Testing runs a system on chosen inputs and reports what happened on them. Formal verification takes a set of inputs, described mathematically, and computes a guarantee holding for every input in that set, including ones that have not been physically run. For a neural network the practical form is bound propagation: the method reads the trained weights, pushes an input range through the layers, and arrives at a range the output cannot leave. If that range sits inside a safe limit, every input in the set is safe, and the argument is arithmetic.

The gap from testing is not one of degree. A disturbance that varies continuously has uncountably many settings; a testing campaign visits finitely many, and so does a generated one. Older surveys of industrial practice put formal methods mostly at specification and design, with tooling among the barriers [21] even though model checking is long established for finite-state control logic [22]. For neural networks the tooling has matured quickly, from complete satisfiability-modulotheories (SMT) methods [23, 24] to reachability frameworks for learning-enabled systems [25], and on to closed-loop controllers by partitioning the workspace so the imaging function is affine on each piece [26].

What limits formal methods is the disturbance modeling, not the mathematical solvability. Verifiers are overwhelmingly exercised on $\ell _ { p }$ balls around an image [27, 28], a set whose radius is chosen for tractability and whose tightest relaxation is itself bounded [29]. A ball large enough to contain a night-time image also contains images no camera can form, so a bound over it says nothing about real-world disturbances experienced by automated vehicles. While verification is one thread in a wider effort spanning autoformalization [30], interpretability [31] and statistical safety-case confidence [32], we target the perception-to-control bound, which is directly relevant for automated driving using E2E models.

In this work we apply neural-network formal verification to E2E steering under degraded visibility, and we make three contributions. All of the evidence is from the CARLA simulator, on two roads, with a camera as the only sensor and steering as the only learned command. We show that a bound taken only at the conditions a campaign captures is not sufficient, because it clears a policy whose worst case lies at an intermediate strength. We give a disturbance family that brings those intermediate intensities within a verifier’s reach. And we reconcile a per-frame bound with closed-loop driving using vehicle dynamics. We seek neither a new verification algorithm [33] nor a new planning benchmark [34], but the mapping from a physical disturbance to a set a neural network verifier can bound. The result is a practical route from a physical disturbance to safety-case evidence about conditions nobody drove, using a method automotive validation has barely begun to adopt where one bound covers more of the operating envelope than a test program can drive.

## 2 Related Work

Formal verification of automated driving models addresses the gap between empirical AI performance and strict safety standards. Koopman and Wagner argue in 2018 that the confidence deployment requires cannot be reached by vehicle-level testing alone [35], having first mapped where the ISO 26262 V-model strains against autonomous driving in 2016 [36]. For example, it has been shown that to demonstrate a reduction on an interstate with one fatal accident per 662 million km at 5% significance would require some 6.62 billion test kilometers [37]. Wagner has since carried that accounting to fleet-scale safety-case assessment [38]. Burgio et al. [39] detail the open challenges of verification, in which AVs are heterogeneous systems combining white-box and black-box AI components. Mitra et al. [40] show how the high-dimensional nature of visual perception data fundamentally limits traditional mathematical guarantees. Pan et al. [41] pair large language models with model-driven engineering, deriving formal models from eventchain descriptions to validate generated components at the system level. But formal verification applied to E2E driving policies, against the disturbances a vehicle actually meets, remains largely unexplored.

Foundational formal methods have, by contrast, verified substantial non-AI software. The interactive theorem provers Coq [42], Isabelle/HOL [43] and Lean [44] carry machine-checked proofs for the CompCert C compiler [45] and the seL4 microkernel [46], and they remain relevant for certifying deterministic components under ISO 26262 [8]. They target explicit rule-based logic, so deep learning needs a different paradigm.

When applying formal methods, the disturbance model, not the solver, is what limits applicability to automated driving. Bernardeschi et al. [47] evaluate E2E driving robustness against imperceptible image modifications and report probabilistic guarantees, but macroscopic physical deviations such as global illumination drops remain open. Most directly related, Mohapatra et al. [48] verify robustness against parametric semantic perturbations such as brightness and contrast in place of $\ell _ { \infty }$ pixel balls. We adopt that parametric view, take the endpoints from simulator renders at a fixed camera pose instead of an analytic image model, and then reconcile the per-frame bound with closed-loop driving.

Our verifier is the CROWN family [49, 50] with input-space branch and bound. Bound propagation is white-box in the weights: it needs the trained parameters and the architecture which is a different access requirement from testing and is open to a developer or to any assessor given the weights. Our disturbance family is built from pose-paired simulator renders, so for this work the bound needs a renderer even though it never needs to drive. Semidefinite relaxations such as SDP-CROWN [51] tighten bounds on high-dimensional $\ell _ { 2 }$ balls, a regime our one-parameter set does not enter as it stands, though the wider set we propose in the Conclusion would. The approach complements verifiers based on abstract interpretation [52], control reachability [53, 54] and uncertainty-aware analysis of E2E driving [55].

## 3 Methodology

The goal is a per-frame formal verification result whose verdict matches what the vehicle does in closed-loop simulation, without actually running the simulation. This is applied in two operational design domains (ODDs): a highway and an urban arterial in CARLA (§3.1). The policy is a small distilled student, since bound-propagation cost scales with size (§3.2). Disturbances come from rendered frames, not an analytic model, and the verified set is the one-parameter family between two of them (§3.3). The safety test uses the sustained steering deviation rather than the peak, which is what a lane-keeping limit is derived for (§3.4).

## 3.1 Simulator, Vehicle, and ODD

The study runs in the open source CARLA simulator version 0.9.16 [15]. A simulator is required here, because the disturbance family of §3.3 needs a clear frame and a degraded frame from identical camera poses, which a real drive cannot supply. The ego vehicle is a Tesla Model 3 from CARLA’s standard blueprint library, a passenger car whose bounding box sets the lane-departure budget of §3.4. To keep the study simple, the network steers and nothing else: longitudinal speed is held at a constant $\nu _ { x } = 8 . 9 4 0 8$ m/s (20 mph) by a proportional-integral (PI) controller at a 5 Hz control rate $( \Delta t = 0 . 2 \mathrm { s } )$ , and the only sensor is a front RGB camera at 640 480 with $\mathrm { \textbf { a 9 0 } ^ { \circ } }$ field of view. Repeatability is not free in CARLA, which is built on a game engine tuned for smooth graphics rather than for reproducing a lap. We have thoroughly addressed this by ensuring the simulator is stepping on a fixed timestep, each steering command lands on the tick it was issued for, and background streaming of texture detail is switched off.

We chose two CARLA maps that are different ODDs in the vocabulary of ISO 34503 [56], where road type and road geometry are first-class ODD attributes. Town04 is a gradeseparated highway loop and Town06’s outer loop is an urban arterial with tighter corners. In the terms of Koopman and Fratrik [57] these are two separate ODDs and their results are likely to disagree. Both routes are drawn in Fig. 1.

1.32 mi one-way intersection, excluded  
![](images/aaf49a5304e4bba083523ba3333fa13eeb709167ab87172f5fdc836d8da761fc.jpg)  
(a) Town04, a grade-separated highway loop driven in both directions, where each direction stops short of the same intersection from its own side.

![](images/762e84371da7c109012ad9b2950411d8102ce0abcb64a8263b68d81ecca911cb.jpg)  
(b) Town06, one continuous 1.42 mi lap of an urban arterial, with both intersections driven by the PPC expert and excluded from every measurement.

Figure 1: The two roads: a grade-separated highway loop driven both ways, and one continuous lap of an urban arterial whose two intersections are excluded from every measurement.

The Town04 highway study was completed first, and it is where the criterion and its one calibrated constant were chosen: the closed-loop reaction horizon $T _ { \mathrm { c l } }$ . It stands for how long a steering error is assumed to persist before the vehicle either corrects it or leaves its lane because of it. The Town06 arterial study then used that same value, without refitting it. A lap is one traversal of the full route, and it fails if the vehicle leaves its lane at any point along the way. A test case is one policy driven under one condition. Each test case is driven three independent laps, and those three laps are certified together as a single unit. The three laps helped us diagnose many internal errors as we developed the test case procedure.

## 3.2 AI Model Development

There is a well-known tension in formally verifying an E2E lane-keeping policy: bound-propagation cost scales with network size, so a verifiable policy must be small, yet a small network is hard to train to expert quality. We resolve this tension with a teacher student design (Fig. 2) in which the large teacher is only a distillation source and is never verified. A PilotNet-class network [2] of 5 convolutional + 4 fully-connected layers, rectified-linear (ReLU) units only, on a 200 66 red-green-blue (RGB) crop ( 107k ReLU neurons) is used for the teacher.

The student is a ReLU-only convolutional neural network (CNN), 3 strided convolutions and 2 fully-connected (FC) layers, whose region of interest is fixed from ground-truth segmentation. By cropping tightly, we keep lane markings legible at low resolution and also enable a small RGB input size. Table 1 provides the size details of each of the four models, ranging from 5,152 ReLU neurons on Town04 to 101,888 on Town06, which needed a larger input to pass closed-loop testing. Each road utilizes two students distilled from its own teacher, one trained on the clear condition alone and one on the four conditions of §3.3, clear, fog, night and low sun. We call the final models the clear-trained and the mixed-trained model throughout. The mixed-trained model is wider on both roads because it represents four conditions rather than one. Network width is more effective than depth, which at a matched neuron count widens the certified bounds by 2.3 to 3.7 .

Table 1: The four verified models. All share the topology of Fig. 2 and differ only in these numbers.
<table><tr><td>Road</td><td>Student</td><td>Input</td><td>(c1, c2, c3)</td><td>FCw</td><td>ReLU</td></tr><tr><td>Town04</td><td>clear-trained mixed-trained</td><td>84×28 84×28</td><td>(8,16,16) (24,48,48)</td><td>32 96</td><td>5,152 15,456</td></tr><tr><td>Town06</td><td>clear-trained mixed-trained</td><td>168×56 168×56</td><td>(16,32,32) (32,64,64)</td><td>64 128</td><td>50,944 101,888</td></tr></table>

To begin model development, the teacher is first fitted by behavior cloning on PPC-derived expert camera and label pairs over the route, then refined by Dataset Aggregation (DAgger) [58]. Through the DAgger process, the current model iteration drives the full route and when it wanders off the lane, the PPC is queried for the correct recovery action which is then added to the training set. The teacher is retrained on the aggregate and the process repeats until the teacher drives within budget. The student is then distilled from the converged teacher, and runs its own DAgger rounds until it too drives within budget.

We learned several lessons throughout this process. The mixed model DAgger rounds need a warm start, meaning that each round fine-tunes the checkpoint from the round before it, at a reduced learning rate, instead of training again from scratch. Additionally, when a policy needs more capacity, it is better to widen it than to feed it a larger image, because widening adds parameters without enlarging the set the verifier has to bound. Lastly, it pays to keep refining the teacher through DAgger after it already drives within budget: the student distilled from it comes out 2.7 closer.

![](images/5f34e21fe115c7f8db782dd5389b39e5ed2bdee994a4131f0f877d5df2075300.jpg)  
Figure 2: All four models share this topology, differing only in the widths tabulated in Table 1. Each is distilled from a PilotNet-class teacher trained on pure-pursuit labels, and only the student model is verified.

## 3.3 Disturbance Modeling

Each disturbance must be a set the verifier can bound, and it must provoke the same response from the policy as the real condition does. We therefore take the clear and disturbed images from the exact same point in the simulator itself rather than from a photometric model or a real-world dataset. For each pose p on the route we capture a clear frame $x _ { p } ^ { \mathrm { c l e a r } }$ and a condition frame $x _ { p } ^ { \mathrm { c o n d } }$ . Each one is the array of pixel values the network receives on a 0 to 1 brightness scale rendered at full sensor resolution and only then cropped and downsampled. These captured frames are the ones the verifier reads, and we check per pose that they are what the model saw while driving.

Between those two endpoints we declare

$$
\begin{array} { r } { x _ { p } ( s ) = x _ { p } ^ { \mathrm { c l e a r } } + s \big ( x _ { p } ^ { \mathrm { c o n d } } - x _ { p } ^ { \mathrm { c l e a r } } \big ) , \qquad s \in [ 0 , 1 ] , } \end{array}\tag{1}
$$

in which s is a single dial for the strength of the disturbance, so Eq. (1) stands for a whole family of disturbances rather than one. $\mathbf { A } \mathbf { t } s = 0$ the image is the clear frame, at s = 1 the rendered disturbance is at full strength, and in between the disturbance is a blend of the two. Figure 3 shows how this construction captures an entire family of disturbances through one equation. The verifier is then asked about one unknown number instead of thousands of independent pixels, which is what makes it possible to cover every strength in the range at once instead of the single strength a driven lap happens to sample. Cropping and downsampling are linear, so blending the two small network inputs gives exactly the image we would get by blending the two full-resolution renders and shrinking the result. The dial s stays inside [0, 1] because outside it the pixel values run past black or white.

![](images/7b6d44ab7659d5ad7dd6b33d8090da5449776c7012a420cefbe20c5fcd92bcc6.jpg)  
Figure 3: Eq. (1) as pictures. The two ends are renders at the same camera pose, the two frames between them are blends, and s dials the strength.

We study four conditions on this axis: clear (the $s = 0$ baseline), fog, night, and low sun. Figure 4 shows each as the network receives it, with the mean absolute change it makes to a pixel. Low sun sounds the mildest and is the strongest disturbance on the arterial, because that sun angle puts the whole road in terrain shadow with no artificial light. With the sun below the horizon the simulator switches the vehicle’s headlights on and lights the road directly, brightening about a quarter of the pixels while the rest go darker still. The whole road being in shadow is also why we call the condition low sun rather than shadows.

![](images/57c7099a41bab42c9443cad3075ab6e5ab5b65017faab4baa8eba174f9c2b488.jpg)  
(a) Town04, 84 28  
(b) Town06, 168 56  
Figure 4: The four conditions as each network receives them. Each panel gives ∆x , the mean absolute change from the clear condition, averaged over every pixel and every pose on the 0 to 1 brightness scale.

Three ideas we tried and abandoned are worth noting. The first was an analytic Koschmieder fog model [59] which is the attenuation-plus-airlight form that single-image dehazing inverts [60]. Its appeal is that the one free parameter is physical. It reproduced CARLA’s fog images respectably, at a road-ROI $R ^ { 2 } = 0 . 8 4 8$ , and was still unusable, because looking like real fog is not the same as steering the policy like real fog. The network reacted far more strongly to the model than to the rendered fog it stood in for, and analytic models of night [61] fail that test the same way. The second was a real-world dataset. The Adverse Conditions Dataset with Correspondences (ACDC) [17] pairs every adverse frame with a matching view of the same scene in clear conditions, but the two come from separate drives, so the difference between them carries viewpoint and scene change as well as the condition, where Eq. (1) needs it to be the condition alone. The third was rain, which CARLA renders with a temporally random element that a two-endpoint family cannot represent.

## 3.4 The Safety Criterion

Success for a lane-keeping policy means no part of the vehicle leaves its lane. With the vehicle centered, the room on either side is half the lane width minus half the vehicle width. The cross-track error (CTE), measured from vehicle center to lane center, therefore has a budget of

$$
\mathrm { C T E } _ { \mathrm { b u d g e t } } = \frac { w _ { \mathrm { l a n e } } - w _ { \mathrm { v e h } } } { 2 } .\tag{2}
$$

The lane width $w _ { \mathrm { l a n e } } = 3 . 5 0 0 \mathrm { m }$ is measured constant along both routes, and the vehicle width $w _ { \mathrm { v e h } } = 2 . 1 6 4$ m is the CARLA bounding box of the Tesla Model 3, mirrors included. That leaves 0.668 m (2.19 ft) of allowable CTE.

Converting that distance into a steering limit starts from the kinematic bicycle model. The vehicle holds the constant longitudinal speed $\nu _ { x }$ and a steering angle $\delta$ on a wheelbase L which turns it at the yaw rate $\dot { \psi } .$ This relationship is then linearized for small steering angles as

$$
\dot { \psi } = \frac { \nu _ { x } } { L } \tan \delta \approx \frac { \nu _ { x } } { L } \delta .\tag{3}
$$

Here $\nu _ { x } = 8 . 9 4 0 8$ m/s (discussed in §3.1) and the Tesla Model 3’s wheelbase is $L = 3 . 0 0 5 \mathrm { m }$ . Integrating Eq. (3) with a constant steering error ∆δ in place of δ gives the heading error that steering error produces after t seconds,

$$
\Delta \psi ( t ) = \int _ { 0 } ^ { t } \frac { \nu _ { x } } { L } \Delta \delta d \tau = \frac { \nu _ { x } } { L } \Delta \delta t ,\tag{4}
$$

where <sub>τ</sub> is the integration variable. The speed itself does not change. The heading error only tilts the velocity off the lane direction, so the component across the lane is $\dot { y } = \nu _ { x }$ sin∆<sub>ψ</sub> $\nu _ { x } \Delta \psi .$ . Integrating that lateral velocity, with Eq. (4) substituted for ∆<sub>ψ</sub>, gives the lateral offset the steering error accumulates,

$$
y ( T ) = \int _ { 0 } ^ { T } \dot { y } d t = \int _ { 0 } ^ { T } \frac { \nu _ { x } ^ { 2 } } { L } \Delta \delta t d t = \frac { \nu _ { x } ^ { 2 } T ^ { 2 } } { 2 L } \Delta \delta ,\tag{5}
$$

where T is how long the steering error persists. Setting y(T) equal to the CTE budget and solving for ∆δ gives the largest steering deviation the vehicle can absorb, written the way the network writes steering, as a fraction of its configured limit $\delta _ { \mathrm { m a x } } = 7 0 ^ { \circ }$

$$
\delta _ { \mathrm { t o l } } = { \frac { 2 L \mathrm { C T E } _ { \mathrm { b u d g e t } } } { \nu _ { x } ^ { 2 } T _ { \mathrm { c l } } ^ { 2 } \delta _ { \mathrm { m a x } } } } = 0 . 0 1 2 0 .\tag{6}
$$

The value we use for $T$ is $T _ { \mathrm { c l } } .$ , the closed-loop reaction horizon. Fundamentally it represents how long a systematic steering error survives in the frames the model’s own driving produces. We fit it once on the Town04 highway, from the steering bias at which laps actually began to leave the lane, and reused that value, $T _ { \mathrm { c l } } = 1 . 8 5 \mathrm { s }$ , unchanged on the Town06 arterial road. The verdicts do not turn on its exact value. Every highway verdict holds across a factor-of-1.7 window in $T _ { \mathrm { c l } }$ that contains the roughly 1.5 s perception-reaction time the driver-response literature reports for surprise events [62].

## 3.5 Formal Verification

Formal verification provides a maximum bound on a property over a declared set. Such a bound answers not what the network outputs on one image, but the most it could output on any image in a set. Hand it the whole family of Eq. (1) rather than one disturbance strength at a time, and it returns an interval guaranteed to contain every steering value the network can produce anywhere in that family. This is computed by propagating the set through the layers as algebra rather than as sampled points, with each ReLU replaced by a linear upper and lower envelope. Note that the formal verification literature uses the word ‘certified’ to mean that the bound stayed under a desired threshold. The words ‘certified’ and performance ‘certificate’ are not meant in the regulatory sense because they do not approve a vehicle for road use. Formal verification’s evidence can be one part of a safety case, but it is not a substitute for one.

CROWN [49, 50] is the established algorithm we use for formal verification, in its base form rather than the refinements <sub>α</sub>- CROWN [63], β-CROWN [27] and SDP-CROWN [51], whose gains come from high-dimensional input sets and from branching over ReLU activations, neither of which a one-parameter family offers. What matters for validation is that the interval it returns is one-sided, meaning that it can be wider than the true range of outputs but never narrower. That looseness grows with the width of the set being bounded, so we do not ask for one bound over the whole family at once. We cut the strength range into sub-intervals, bound each separately, and take the worst which is still a bound on the whole family. Narrower pieces give tighter bounds, which is what makes the result sharp enough to certify a policy that drives cleanly. The number of pieces is a free parameter and we use 16 sub-intervals per pose on both roads.

The bound and the budget now meet. The verifier runs at P poses along the whole lap, one every eighth control step, and at each pose it reports how far the disturbance can move the steering away from the value the same network produces on the clear frame as

$$
\Delta _ { p } ( s ) = \delta _ { p } ( s ) - \delta _ { p } ( 0 ) ,\tag{7}
$$

where $\delta _ { p } ( s )$ is the steering on the blended image $x _ { p } ( s )$ at any strength from the clear frame at $s = 0$ to the full condition at $s = 1$ . The clear frame is a reference and not ground truth. Subtracting it separates what the disturbance does from how well the policy steers in the first place, which the clear-condition drives answer on their own. Equation (6) is likewise not a limit on one frame. It is the steering error that would use up the whole lane budget if it were held for $T _ { \mathrm { c l } }$ seconds, so what we compare against it has to be a sustained error rather than a spike. We therefore average the per-pose deviations along the lap,

$$
\bar { \Delta } ( { \bf s } ) = \frac 1 P \sum _ { p = 1 } ^ { P } \Delta _ { p } ( s _ { p } ) ,\tag{8}
$$

which is the steady bias the disturbance leaves in the steering. A spike that reverses sign within a few frames averages away, just as it washes out of the vehicle’s path, while a deviation that keeps its sign survives both. Each pose is free to sit at a different strength, so the vector $\mathbf { s } = ( s _ { 1 } , \ldots , s _ { P } )$ holds one strength per pose, and the policy is certified when no assignment of strengths pushes the average outside the corridor as

$$
\operatorname* { m a x } _ { \mathbf { s } \in [ 0 , 1 ] ^ { P } } \left. \bar { \Delta } ( \mathbf { s } ) \right. \leq \delta _ { \mathrm { t o l } } .\tag{9}
$$

CARLA renders each condition uniformly along the road, so this set is deliberately larger than anything a driving campaign can test.

There are a few things to note about this approach. Cutting the strength range into more pieces buys more than a more elaborate bounding algorithm does. The <sub>α</sub>-optimized refinement ran far longer and gave back almost nothing on a family this narrow. How many pieces matters, though. With only four, one policy-condition test case’s bound came out at 1.07 times the tolerance. We also began with the wrong statistic. Testing the single worst frame instead of the road average threw out policies that drive cleanly, and it ranked two test cases in the opposite order. No choice of threshold repairs that.

![](images/d229644e674c280d1f8b7fb0a60d93b680c6bf0be39910c4e80ad72c1c6b2225.jpg)  
(a) Town04 highway, clear-trained model

![](images/315ed5682b8bf97d2fa5e5130251c0283afeee3a13ce61568510081118654eed.jpg)  
(c) Town04 highway, mixed-trained model

## 4 Results

The first step in collecting results is to drive every policy until its trained behavior is established in closed-loop simulation (§4.1). Additional closed-loop test case verdicts are then compared to formal verification where it was determined that formal verification can correctly predict the outcomes (§4.2). Additional analysis of the formal verification shows that it may also be identifying failures between test cases (§4.3). We also discuss the ODD difference between the Town04 highway route and the Town06 urban arterial route (§4.4).

## 4.1 Closed-Loop Driving

We proceed to formal verification only once a model holds its lane on every condition it was trained on. The mixed-trained model must drive clear, fog, night and low sun within the CTE budget in closed loop; the clear-trained model must drive the clear condition, though it does not need to be tested in fog, night or low sun before proceeding to formal verification.

Figure 5 shows where along each route the CTE accumulates at every control step. A trace ends the first time it passes 3 ft, because past that the test case has failed and how much further it goes does not provide any usable information for this work. We observe several interesting behaviors. First the cleartrained model has near zero CTE during clear conditions in Town04 (highway) and Town06 (arterial). The clear-trained model also cannot handle mixed conditions (fog, night, low sun). Likewise the mixed-trained models have near zero CTE during all mixed conditions. However, Town06 does somehow seem to be more challenging because the clear-only model fails faster in mixed conditions and the mixed model has slightly higher CTE overall in mixed conditions.

![](images/f17d74b8106baab520306706890a005f5bfabc5f3d16e8ecd4f96449d2603bfe.jpg)  
(b) Town06 arterial, clear-trained model

![](images/11acec9baed4a5fd3edf54e1588bb4f5c3e16d999cf8e6a12f245298bac80d1e.jpg)  
(d) Town06 arterial, mixed-trained model  
Figure 5: Cross-track error along the route against the 2.19 ft budget (dashed). Each trace stops the first time it passes 3 ft, and the gray bands are Town06’s two bridged intersections, which no policy steers.

## 4.2 Verification

After the closed-loop driving of §4.1 is complete we are ready to run formal verification. Every model-condition-town test was bounded with CROWN over the whole disturbance family and scored against the criterion of $\operatorname { E q . }$ (9) which is a mathematical process only and involves no further simulation. We are mathematically testing between the test cases. The CROWN verifier algorithm reads the trained weights and the frames already captured. The result is a bound on the worst sustained steering deviation any disturbance in the family can produce.

Table 2 puts that formal verification result beside what the same model did in closed-loop testing. The two clear test cases carry a certificate that is vacuous because clear is the point every disturbance is measured from and there is nothing there for the bound to compare against. In Town04 formal verification correctly predicts that the clear-trained model is suitable for fog which is confirmed through closed loop testing. Every other Town04 comparison agrees. Town06 is much more challenging. The worst CTE values are higher overall and the clear-trained model does not work in fog which formal verification also correctly predicts. The interesting case is the mixedtrained model in Town06, which passes closed-loop testing in fog and at night and which formal verification will not certify. Next we can explore this in more detail.

Table 2: Every test case of both studies, one model on one road under one condition. Note that exceeds means the model left its lane.
<table><tr><td>Model</td><td>Condition</td><td>Closed-loop driving</td><td>Worst CTE (ft)</td><td>Formal verification</td></tr><tr><td></td><td colspan="4">Town04, the highway</td></tr><tr><td rowspan="4">Clear-trained</td><td>clear</td><td>PASS</td><td>0.91</td><td>certified†</td></tr><tr><td>fog</td><td>PASS</td><td>1.32</td><td>certified</td></tr><tr><td>night</td><td>FAIL</td><td>exceeds</td><td>not certified</td></tr><tr><td>low sun</td><td>FAIL</td><td>exceeds</td><td>not certified</td></tr><tr><td rowspan="4">Mixed-trained</td><td>clear</td><td>PASS</td><td>0.39</td><td>certified†</td></tr><tr><td>fog</td><td>PASS</td><td>0.37</td><td>certified</td></tr><tr><td>night</td><td>PASS</td><td>0.85</td><td>certified</td></tr><tr><td>low sun</td><td>PASS</td><td>0.54</td><td>certified</td></tr><tr><td colspan="5">Town06, the arterial</td></tr><tr><td rowspan="4">Clear-trained</td><td>clear</td><td>PASS</td><td>1.37</td><td>certified†</td></tr><tr><td>fog</td><td>FAIL</td><td>exceeds</td><td>not certified</td></tr><tr><td>night</td><td>FAIL</td><td>exceeds</td><td>not certified</td></tr><tr><td>low sun</td><td>FAIL</td><td>exceeds</td><td>not certified</td></tr><tr><td rowspan="4">Mixed-trained</td><td>clear</td><td>PASS</td><td>1.30</td><td>certified†</td></tr><tr><td>fog</td><td>PASS</td><td>1.95</td><td>not certified</td></tr><tr><td>night</td><td>PASS</td><td>1.00</td><td>not certified</td></tr><tr><td>low sun</td><td>PASS</td><td>2.19</td><td>certified</td></tr></table>

<sup>†</sup>Vacuous. Clear is the point every disturbance is measured from, so there is nothing here for the bound to compare against and the deviation is zero by construction. The certificate is true but is not informative.

Bounds are intuitively reported as road-averaged bias as Eq. (8) divided by the tolerance of Eq. (6),

$$
\bar { \Delta } _ { \mathrm { r e l } } ( \mathbf { s } ) = \frac { \bar { \Delta } ( \mathbf { s } ) } { \delta _ { \mathrm { t o l } } } ,\tag{10}
$$

which means that if a test case has a maximum output between 1 and +1 it is within tolerance. But a value of 2 means twice the tolerance (i.e., violates the threshold by 2 ). Raw steering units would put every number near 0.01 and would not be comparable between the two roads. The intervals behind Table 2 are plotted in Fig. $^ { 6 , }$ one bar per test case. Certified test cases sit well inside the corridor, refused highway test cases escape it by up to 4 , and the arterial’s clear-trained night test case escapes by 13 . Both clear-trained models sit a long way outside the corridor at night, which matches what they do on the road, where both leave their lane on every run.

![](images/969ac4eca69ac80db7786a92b822a33d75cef4f32512f6ac812853b74a3fadee.jpg)  
(a) Town04, six test cases, each bar the envelope of the two driven directions.

![](images/3d195bcdfb122edcfcfe36cdc4ecea7f9f4dd8c39c18bcdcc65b6e33ee0e9014.jpg)  
Figure 6: Certified sustained steering bias for every test case in the relative units of Eq. (10), black where the test case certifies and red where it does not.

Note that each arterial bar in Fig. 6 is a statement about $[ 0 , 1 ] ^ { P }$ with $P = 1 3 3$ , so ten intensities per pose would be $1 0 ^ { 1 3 3 }$ combinations, which is beyond expensive to drive; it is impossible to drive and impossible to sample. A bar costs a few minutes of GPU time, which is roughly what driving one test case in simulation costs. One for one there is no speedup, but per unit of evidence there is no contest regarding the information that formal verification provides versus closed-loop testing. The bound is a statement about the whole disturbance family, and the closed-loop test is a statement about one disturbance value.

## 4.3 Between the Test Cases

We can uncover details beyond Fig. 6 by going inside the disturbance family and asking specifically what disturbance level (the value s) causes the worst output for uncertified test cases. Figure 7 adds this information with three markers on each row. The circle is the steering bias at the rendered condition, s = 1, which is what endpoint testing checks. The square is the worst output from any one strength s of Eq. (1) applied to the whole route, marked with the $s ^ { \star }$ that produced it, and it is the strongest disturbance a single drive could apply. The triangle is the worst the road average can reach when the strength is free to change from one pose to the next, every pose pushing the steering the same way so nothing cancels. It is larger than anything a single drive can apply, since no disturbance rendering setting varies along a road pose by pose. A triangle is drawn only where the single-strength search left a test case unresolved, which is why panel (a) has just one, on low sun. Panel (a) is the Town04 highway, and its circles and squares coincide in all six test cases: no single strength anywhere in the family is worse than the rendered condition, so driving the condition would have found every failure a single strength can produce. The one low-sun triangle is the exception, and it lies outside the corridor.

![](images/c588c66711450c6fa8fabf23d2918c5ccf528c168766a5279b3792eb3dd2d141.jpg)  
(a) Town04 highway, six test cases. No single strength is worst in the interior; the one profile that leaves the corridor varies along the road.

![](images/a994cce2502a5c8df588f9dda0679eed7e23b3c205a7dc8e37add829a85ac03c.jpg)  
Figure 7: Where inside each disturbance family the worst sustained bias falls, red where it leaves the tolerance corridor.

Panel (b) is the arterial, and the markers come apart. Let’s start with the clear-trained model under fog and under low sun. At the rendered condition the bias sits comfortably inside the corridor, at 0.69 and 0.37, which are the two circles, so a bound computed only there would have issued two clean certificates. Both test cases leave the lane on every attempted lap and the squares explain it. The worst single strength is not the rendered one but an intermediate one, $s ^ { \star } = 0 . 4 1$ under fog and $s ^ { \star } = 0 . 6 0$ under low sun, where the bias reaches 1.01 and 1.87 and leaves the corridor. The failure was between the test cases the whole time, and quantifying over the family is what found it. Now the two test cases that seemed to contradict the method, fog and night on the mixed-trained model. Their circles and squares are both inside the corridor, which is why the model drives clean: a lap applies one strength to the whole road, and no single strength breaks it. Their triangles are outside, at 1.179 and 1.210. Those are not artifacts of a loose bound but exhibited profiles, strengths assigned pose by pose at which the model’s own steering leaves the corridor, so no sound method could certify either test case. The bound and the drive are answering different questions, and the bound is asking the larger one.

The details regarding the AI model failures at the intermediate disturbance levels are worthy of their own closed-loop testing campaign to further expand and improve the application of formal verification to the problem of autonomous driving. As this research continues to evolve, this will be a focus in future studies. Preliminary results suggest that formal verification method is able to identify failures that are not captured by the closed-loop testing, and this information could be used to improve the robustness of the AI models directly.

## 4.4 The Difference Is the ODD

We explored several alternatives to explain the differences in Town04 and Town06 results. It is not the network size, because the fog bounds get wider as the network shrinks rather than narrower. It is not the input projection either, since capturing the arterial road at the highway’s own resolution still leaves fog far outside the tolerance. And it is not one unlucky draw: a pre-registered sweep of sixteen independently distilled mixed-trained models, two widths and eight seeds, drove 83 laps against a stricter gate than the ledger’s and none passed, with fog rejecting every seed that reached it. The teacher in Town06 is not comfortable in fog either, holding 1.53–1.68 ft against the 2.19 ft budget, at 70–77% of it. So although fog is drivable on this road, the models struggle with it more than capacity or the random draw explains.

What is left as a variable is the road itself. Road type and geometry are first-class ODD attributes under ISO 34503, and the two routes differ in exactly the attributes §3.1 chose them to differ in: the arterial is 74–79% straight against the highway’s 51–56%, with a minimum radius of 22–27 m against 45–63 m. Nothing carries from one combination of road attributes to another for free, so the arterial is a domain adjacent to the highway rather than a replication of it, and fog on it is harder to certify at every capacity and resolution we measured.

## 5 Conclusion

We trained two small camera-only steering models in simulation, one on the clear condition and one on clear, fog, night, and low sun conditions. This was done for the Town04 highway and the Town06 arterial. We the apply formal verification using a straight line in image space between two frames captured at the same camera position, so a single number sets how strong the disturbance is and the verifier can be asked about all of it at once. The verifier bounds the steering error that persists along the route which causes failures. It correctly predicted every highway outcome and also identifies potential failure points that were never tested in closed loop.

This paper provides evidence that formal verification viable for a narrow aspect of automated driving verification. Formal verification covers more conditions in a few minutes than a test program could drive in its lifetime, but the family of policies under test must be developed as a mathematical set. Future work involves widening the mathematical set to include more disturbances such as rain, snow, glare, and dust, as well as combinations of these disturbances. Additionally we seek to continue testing the predicted failures from formal verification to further expand its utility.

## Data Availability

The pipeline, instruments, checkpoints and every artifact behind the numbers in this paper are released as a versioned software record whose concept DOI resolves to the latest version [64], mirrored on GitHub under the AD-Assurance-Lab organization. The captured frames the bounds are computed on are published separately [65].

## Abbreviations

AI artificial intelligence   
AV autonomous vehicle   
CARLA Car Learning to Act   
CNN convolutional neural network   
CTE cross-track error   
DAgger dataset aggregation   
E2E end-to-end   
FC fully connected   
FOV field of view   
GPU graphics processing unit   
MSE mean squared error   
ODD operational design domain   
PI proportional-integral   
PPC pure-pursuit controller   
ReLU rectified linear unit   
RGB red, green, blue   
SDP semidefinite programming   
SMT satisfiability modulo theories   
SOTIF safety of the intended functionality

## References

[1] H. X. Liu, Z. Cao, X. Yan, S. Feng, and Q. Lu, “Autonomous vehicles: A critical review (2004-2024) and a vision for the future,” IEEE Transactions on Intelligent Vehicles, 2025.

[2] M. Bojarski, D. Del Testa, D. Dworakowski, B. Firner, B. Flepp, P. Goyal, L. D. Jackel, M. Monfort, U. Muller, J. Zhang, X. Zhang, J. Zhao, and K. Zieba, “End to end learning for self-driving cars,” arXiv preprint arXiv:1604.07316, 2016.

[3] A. Kendall, J. Hawke, D. Janz, P. Mazur, D. Reda, J.-M. Allen, V.-D. Lam, A. Bewley, and A. Shah, “Learning to drive in a day,” in IEEE International Conference on Robotics and Automation (ICRA), 2019, pp. 8233–8239.

[4] L. Chen, P. Wu, K. Chitta, B. Jaeger, A. Geiger, and H. Li, “End-to-end autonomous driving: Challenges and frontiers,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 12, pp. 10 164– 10 183, 2024.

[5] Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, L. Lu, X. Jia, Q. Liu, J. Dai, Y. Qiao, and H. Li, “Planningoriented autonomous driving,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 14 232–14 242.

[6] B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang, “Vad: Vectorized scene representation for efficient autonomous driving,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 8306–8316.

[7] C. Liu, M. Zhu, Z. Zhang, L. Song, X. Zhao, Q. Luo, Q. Wang, C. Guo, and K. Su, “Tad-e2e: A large-scale end-to-end autonomous driving dataset,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

[8] International Organization for Standardization, “Road vehicles — functional safety,” ISO, Tech. Rep. ISO 26262, 2018.

[9] R. Salay, R. Queiroz, and K. Czarnecki, “An analysis of ISO 26262: Using machine learning safely in automotive software,” arXiv preprint arXiv:1709.02435, 2017.

[10] International Organization for Standardization, “Road vehicles — safety of the intended functionality,” ISO, Tech. Rep. ISO 21448, 2022.

[11] ——, “Road vehicles — safety and artificial intelligence,” ISO, Tech. Rep. ISO/PAS 8800, 2024.

[12] Underwriters Laboratories, “Standard for safety for the evaluation of autonomous products,” UL, Tech. Rep. UL 4600, 2023.

[13] F. Favarò, L. Fraade-Blanar, S. Schnelle, T. Victor, M. Peña, J. Engstrom, J. Scanlon, K. Kusano, and D. Smith, “Building a credible case for safety: Waymo’s approach for the determination of absence of unreasonable risk,” Waymo LLC, Tech. Rep., 2023.

[14] Waymo, “Sharing our safety framework for fully autonomous operations,” Waypoint – The Official Waymo Blog, 2020. [Online]. Available: https://waymo.com/blog/2020/10/sharing-our-safety-framework/

[15] A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V. Koltun, “CARLA: An open urban driving simulator,” in Proceedings of the 1st Annual Conference on Robot Learning, 2017, pp. 1–16.

[16] G. P. Vivan, N. Goberville, Z. D. Asher, N. Brown, and J. F. Rojas, “No cost autonomous vehicle advancements in carla through ros,” SAE Technical Paper, Tech. Rep. 2021-01-0106, 2021.

[17] C. Sakaridis, D. Dai, and L. Van Gool, “Acdc: The adverse conditions dataset with correspondences for semantic driving scene understanding,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 6765–6774.

[18] C. Sakaridis, H. Wang, K. Li, R. Zurbrügg, A. Jadon, W. Abbeloos, D. O. Reino, L. Van Gool, and D. Dai, “Acdc: The adverse conditions dataset with correspondences for robust semantic driving scene perception,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 3, pp. 2970–2985, 2026.

[19] NVIDIA, “NVIDIA OmniDreams: Real-time generative world model for closed-loop autonomous vehicle simulation,” arXiv preprint arXiv:2606.03159, 2026.

[20] D. Fernandez, P. MohajerAnsari, A. Salarpour, and M. D. Pese, “Understanding adversarial transferability in vision-language models for autonomous driving: A cross-architecture analysis,” SAE International, SAE Technical Paper 2026-01-0170, 2026.

[21] J. Woodcock, P. G. Larsen, J. Bicarregui, and J. Fitzgerald, “Formal methods: Practice and experience,” ACM Computing Surveys (CSUR), vol. 41, no. 4, pp. 1–36, 2009.

[22] E. M. Clarke, O. Grumberg, D. Kroening, D. Peled, and H. Veith, Model Checking, 2nd ed. MIT Press, 2018.

[23] G. Katz, C. Barrett, D. L. Dill, K. Julian, and M. J. Kochenderfer, “Reluplex: An efficient smt solver for verifying deep neural networks,” in International Conference on Computer Aided Verification (CAV), 2017, pp. 97–117.

[24] G. Katz, D. A. Huang, D. Ibeling, K. Julian, C. Lazarus, R. Lim, P. Shah, S. Thakoor, H. Wu, A. Zeljic, D. L. Dill, M. J. Kochenderfer, and C. Bar-´ rett, “The marabou framework for verification and analysis of deep neural networks,” in International Conference on Computer Aided Verification (CAV), 2019, pp. 443–452.

[25] H.-D. Tran, D. M. Lopez, P. Musau, X. Yang, L. V. Nguyen, W. Xiang, and T. J. Johnson, “Nnv: The neural network verification tool for deep neural networks and learning-enabled cyber-physical systems,” in International Conference on Computer Aided Verification (CAV). Springer, 2020, pp. 3–17.

[26] X. Sun, H. Khedr, and Y. Shoukry, “Formal verification of neural network controlled autonomous systems,” in Proceedings ofthe 22nd ACM International Conference on Hybrid Systems: Computation and Control (HSCC), 2019, pp. 147–156.

[27] S. Wang, H. Zhang, K. Xu, X. Lin, S. Jana, C.-J. Hsieh, and J. Z. Kolter,

“Beta-crown: Efficient bound propagation with per-neuron split constraints for neural network robustness verification,” Advances in Neural Information Processing Systems (NeurIPS), vol. 34, pp. 29 909–29 921, 2021.

[28] G. Singh, T. Gehr, M. Püschel, and M. Vechev, “An abstract domain for certifying neural networks,” Proceedings of the ACM on Programming Languages (POPL), vol. 3, no. POPL, pp. 1–30, 2019.

[29] H. Salman, G. Yang, H. Zhang, C.-J. Hsieh, and P. Zhang, “A convex relaxation barrier to tight robustness verification of neural networks,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 32, 2019, pp. 9835–9846.

[30] C. Szegedy, “A promising path towards autoformalization and general artificial intelligence,” in Intelligent Computer Mathematics (CICM), ser. Lecture Notes in Computer Science, vol. 12236. Springer, 2020, pp. 3– 20.

[31] Anthropic, “Tracing the thoughts of a large language model,” Anthropic Research, Mar. 2025, companion articles: “On the Biology of a Large Language Model” and “Circuit Tracing”. [Online]. Available: https://www.anthropic.com/research/tracing-thoughts-language-model

[32] P. Bishop, A. Povyakalo, and L. Strigini, “Bootstrapping confidence in future safety based on past safe operation,” in IEEE 33rd International Symposium on Software Reliability Engineering (ISSRE), 2022, pp. 97– 108, arXiv:2110.10718.

[33] M. Althoff, O. Stursberg, and M. Buss, “Online verification of cognitive car decisions,” in IEEE Intelligent Vehicles Symposium (IV), 2007.

[34] M. Althoff, M. Koschi, and S. Manzinger, “Commonroad: Composable benchmarks for motion planning on roads,” in IEEE 20th International Conference on Intelligent Transportation Systems (ITSC), 2017, pp. 1–8.

[35] P. Koopman and M. Wagner, “Toward a framework for highly automated vehicle safety validation,” SAE International, SAE Technical Paper 2018-01-1071, 2018.

[36] ——, “Challenges in autonomous vehicle testing and validation,” SAE International Journal of Transportation Safety, vol. 4, no. 1, pp. 15–24, 2016.

[37] W. Wachenfeld and H. Winner, “The release of autonomous vehicles,” in Autonomous Driving: Technical, Legal and Social Aspects. Springer, 2016, pp. 425–449.

[38] M. Wagner, “Agile safety case assessments for autonomous vehicle fleets,” SAE Technical Paper, Tech. Rep. 2026-01-0521, 2026.

[39] P. Burgio, A. Ferrando, and M. Villani, “Open challenges in the formal verification of autonomous driving,” in Proceedings ofthe Sixth International Workshop on Formal Methods for Autonomous Systems (FMAS), ser. EPTCS, vol. 411, 2024, pp. 191–200.

[40] S. Mitra, C. Pas˘ areanu, P. Prabhakar, S. A. Seshia, R. Mangal, Y. Li,˘ C. Watson, D. Gopinath, and H. Yu, “Formal verification techniques for vision-based autonomous systems – a survey,” in Principles of Verification: Cycling the Probabilistic Landscape, ser. Lecture Notes in Computer Science. Springer, 2025, vol. 15262, pp. 89–108.

[41] F. Pan, Y. Song, L. Wen, N. Petrovic, K. Lebioda, and A. Knoll, “Automating automotive software development: A synergy of generative AI and model-based methods,” arXiv preprint arXiv:2505.02500, 2025, v2.

[42] Y. Bertot and P. Castéran, Interactive Theorem Proving and Program Development. Coq’Art: The Calculus ofInductive Constructions, ser. Texts in Theoretical Computer Science. Springer, 2004.

[43] T. Nipkow, L. C. Paulson, and M. Wenzel, Isabelle/HOL: A Proof Assistant for Higher-Order Logic, ser. Lecture Notes in Computer Science. Springer, 2002, vol. 2283.

[44] L. De Moura, S. Kong, J. Avigad, F. Van Doorn, and J. von Raumer, “The lean theorem prover (system description),” in International Conference on Automated Deduction. Springer, 2015, pp. 378–388.

[45] X. Leroy, “Formal verification of a realistic compiler,” Communications ofthe ACM, vol. 52, no. 7, pp. 107–115, 2009.

[46] G. Klein, K. Elphinstone, G. Heiser, J. Andronick, D. Cock, P. Derrin, D. Elkaduwe, K. Engelhardt, R. Kolanski, M. Norrish, T. Sewell, H. Tuch, and S. Winwood, “seL4: Formal verification of an OS kernel,” in Proceedings of the ACM SIGOPS 22nd Symposium on Operating Systems Principles (SOSP), 2009, pp. 207–220.

[47] C. Bernardeschi, G. Lami, F. Merola, and F. Rossi, “Verifying robustness of neural networks in vision-based end-to-end autonomous driving,” IEEE Access, vol. 13, pp. 71 688–71 704, 2025.

[48] J. Mohapatra, T.-W. Weng, P.-Y. Chen, S. Liu, and L. Daniel, “Towards verifying robustness of neural networks against a family of semantic per-

turbations,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 244–252.

[49] H. Zhang, T.-W. Weng, P.-Y. Chen, C.-J. Hsieh, and L. Daniel, “Efficient neural network robustness certification with general activation functions,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 31, 2018, pp. 4939–4948.

[50] K. Xu, Z. Shi, H. Zhang, Y. Wang, K.-W. Chang, M. Huang, B. Kailkhura, X. Lin, and C.-J. Hsieh, “Automatic perturbation analysis for scalable certified robustness and beyond,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 33, 2020, pp. 21 899– 21 910.

[51] H.-M. Chiu, H. Chen, H. Zhang, and R. Y. Zhang, “Sdp-crown: Efficient bound propagation for neural network verification with tightness of semidefinite programming,” arXiv preprint arXiv:2506.06665, 2025.

[52] T. Gehr, M. Mirman, D. Drachsler-Cohen, P. Tsankov, S. Chaudhuri, and M. Vechev, “AI2: Safety and robustness certification of neural networks with abstract interpretation,” in 2018 IEEE Symposium on Security and Privacy (SP). IEEE, 2018, pp. 3–18.

[53] R. Ivanov, J. Weimer, R. Alur, G. J. Pappas, and I. Lee, “Verisig: verifying safety properties of hybrid systems with neural network controllers,” in Proceedings of the 22nd ACM International Conference on Hybrid Systems: Computation and Control (HSCC), 2019, pp. 169–178.

[54] K. D. Julian and M. J. Kochenderfer, “Reachability analysis for neural network aircraft collision avoidance systems,” Journal of Guidance, Control, and Dynamics, vol. 44, no. 6, pp. 1132–1142, 2021.

[55] R. Michelmore, M. Wicker, L. Laurenti, L. Cardelli, Y. Gal, and M. Kwiatkowska, “Uncertainty quantification with statistical guarantees in end-to-end autonomous driving control,” in 2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020, pp. 7344– 7350.

[56] International Organization for Standardization, “ISO 34503:2023 road vehicles — test scenarios for automated driving systems — specification for operational design domain,” Standard, 2023, supersedes BSI PAS 1883:2020.

[57] P. Koopman and F. Fratrik, “How many operational design domains, objects, and events?” in SafeAI 2019: AAAI Workshop on Artificial Intelligence Safety, 2019.

[58] S. Ross, G. J. Gordon, and J. A. Bagnell, “A reduction of imitation learning and structured prediction to no-regret online learning,” in Proceedings of the 14th International Conference on Artificial Intelligence and Statistics (AISTATS), ser. PMLR, vol. 15, 2011, pp. 627–635.

[59] S. G. Narasimhan and S. K. Nayar, “Vision and the atmosphere,” International Journal of Computer Vision, vol. 48, no. 3, pp. 233–254, 2002.

[60] K. He, J. Sun, and X. Tang, “Single image haze removal using dark channel prior,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 33, no. 12, pp. 2341–2353, 2011.

[61] C. Wei, W. Wang, W. Yang, and J. Liu, “Deep retinex decomposition for low-light enhancement,” in British Machine Vision Conference (BMVC), 2018.

[62] M. Green, “How long does it take to stop? Methodological analysis of driver perception-brake times,” Transportation Human Factors, vol. 2, no. 3, pp. 195–216, 2000.

[63] K. Xu, H. Zhang, S. Wang, Y. Wang, X. Lin, S. Jana, and C.-J. Hsieh, “Fast and complete: Enabling complete neural network verification with rapid and massively parallel incomplete verifiers,” in International Conference on Learning Representations (ICLR), 2021.

[64] M. Ghalan, C. Rodgers, and Z. D. Asher, “Testing between the test cases: Proving end-to-end steering in conditions you never drove,” Zenodo, 2026. [Online]. Available: https://doi.org/10.5281/zenodo.22101297

[65] ——, “Steering verification captures,” Hugging Face Datasets, 2026. [Online]. Available: https://huggingface.co/datasets/ AD-Assurance-Lab/steering-verification-captures