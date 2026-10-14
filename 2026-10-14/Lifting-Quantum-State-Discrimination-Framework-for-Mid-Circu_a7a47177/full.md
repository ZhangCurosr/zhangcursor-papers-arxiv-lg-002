# Lifting: Quantum State Discrimination Framework for Mid-Circuit Measurements

Chi Zhang,<sup>1,</sup> ∗ Neel Vora,<sup>2,</sup> † Yilun Xu,<sup>2,</sup> ‡ Gang Huang,<sup>2,</sup> § Fan Chen,<sup>1,</sup> ¶ and Lei Jiang<sup>1,</sup> ∗∗

<sup>1</sup>Indiana University Bloomington, Bloomington, Indiana, USA

<sup>2</sup>Lawrence Berkeley National Laboratory, Berkeley, California, USA

(Dated: October 14, 2026)

Quantum computing promises new capabilities beyond classical computation. Mid-circuit measurements (MCMs) enable error correction and outcome-dependent control, but their reliability is limited by noise on superconducting hardware. These operations depend on accurate quantum state discrimination (QSD), which assigns measured signals to state labels. Neural network-based (NNbased) QSD improves this assignment, yet its error remains bounded below by the Bayes risk of the available readout distributions. A complementary challenge is state deterioration as successive measurements accumulate and circuit execution advances, making static calibration less representative of the workload. We present Lifting, a QSD framework that addresses both challenges through repetition encoding, neural decoding, and in situ calibration. Encoding supplies joint in-phase and quadrature (I–Q) readout signals from the target and auxiliary qubits, while contextual calibration trains the decoder for the conditions encountered during execution. Our theoretical analysis shows that retaining additional channels from a fixed joint distribution cannot increase Bayes risk and establishes suficient conditions for contraction of an overlap-based upper bound. Experiments across three IBM processors evaluate the framework on repeated stabilizer measurement, parity-check, and teleportation workloads.

## I. INTRODUCTION

Quantum computing promises new approaches to computational tasks that are dificult for classical computers, motivating sustained progress in superconducting quantum processors [1]. Realizing this promise requires reliable operations throughout a computation. Mid-circuit measurements (MCMs) and real-time feedback are central to this efort, enabling quantum error correction and dynamic circuits that adapt their operations to measurement outcomes [2, 3]. However, measurement remains a major source of error on superconducting hardware. Its efects extend beyond an incorrect reported outcome to disturbances of the measured qubit and neighboring qubits, limiting the reliability of subsequent computation [4].

An MCM measures a qubit before circuit execution ends, allowing its outcome to guide subsequent operations. On superconducting hardware, dispersive readout obtains this outcome by probing a resonator whose response depends on the qubit state. The resulting microwave signal is demodulated into in-phase and quadrature (I–Q) components. Quantum state discrimination (QSD) then assigns these continuous signals to discrete state labels, typically 0 and 1 [5]. This makes QSD a classification problem at the interface between quantum evolution and classical control. An incorrect assignment can select the wrong feedback operation or produce an incorrect correction frame, allowing a readout error to afect the remainder of the circuit.

Neural network-based (NN-based) QSD has improved this interface in several complementary ways. Lienhard et al. improved multiplexed discrimination by accounting for readout crosstalk [6], Maurya et al. developed a more compact architecture using matched filters and a neural network [7], and Vora et al. implemented low-latency discrimination on a field-programmable gate array (FPGA) for real-time control [8]. Nevertheless, every discriminator operating on the same readout signals is subject to the same fundamental limit. The Bayes risk is the minimum achievable classification error for given class distributions and prior probabilities. When those distributions overlap, even an optimal classifier must sometimes assign the wrong label. Increasing network capacity or improving training cannot reduce error below this limit [9]. Reducing error below the Bayes risk of single-qubit readout therefore requires changing the information available to the decoder, rather than improving the neural network alone.

A complementary challenge is state deterioration during circuit execution, reflected here in changing I–Q distributions and declining discrimination accuracy. Calibration obtained from isolated state-preparation-andmeasurement (SPAM) circuits can consequently become poorly matched to the signals encountered within the workload. Figure 1 illustrates this behavior in the fivecycle qubit-reuse stabilizer (QR-Stabilizer) workload. In this workload, state deterioration becomes more severe as the number of successive measurements increases and circuit execution advances in time. Later I–Q distribu tions depart further from the SPAM references, while dis crimination accuracy declines. This dependence on circuit history difers from the longer-term parameter drift addressed by calibration-monitoring methods [10, 11].

Residual photons provide one physical mechanism for such history dependence. Photons remaining in a readout resonator after a measurement can continue to disturb the qubit and interfere with subsequent operations, motivating both established and recent cavity-depletion techniques [12, 13]. Santos and Uzdin also explicitly account for quantum-state degradation during repeated measurements in their mitigation framework [14]. These results motivate treating the measurement context as part of the discrimination problem and acquiring in situ calibration that reflects the preceding operations and measurement timing.

We present Lifting, a QSD framework that combines repetition encoding, neural decoding, in situ calibration, and optional confidence filtering. Repetition encoding couples the measured qubit to auxiliary qubits and supplies joint I–Q readout signals to the neural decoder. Signals that overlap in a single-qubit I–Q plane can become distinguishable through information carried by the auxiliary channels. This changes the classification problem and creates an opportunity to reduce its Bayes risk. In situ calibration addresses the complementary problem of state deterioration by training the decoder on references acquired in the measurement context. Confidence filtering further allows applications to trade retained-shot coverage for more reliable accepted outcomes.

Our contributions connect these principles to practical readout on superconducting hardware. First, we characterize within-circuit state deterioration and establish calibration context as an important consideration for NNbased QSD. Second, we provide a theoretical foundation for decoding joint I–Q readout signals, showing that retaining additional channels from a fixed joint distribution cannot increase Bayes risk and establishing suficient conditions for contraction of an overlap-based upper bound. Third, we identify a practical encoding choice that balances discrimination accuracy against the cost of auxiliary qubits and encoding operations. Finally, experiments on IBM processors demonstrate how neural decoding and in situ calibration improve workload-specific circuit success in representative MCM workloads.

The remainder of this paper is organized as follows. Section II reviews dispersive readout and the Bayes-risk formulation. Section III examines within-circuit readoutdistribution changes. In Section IV, we present the theoretical framework of our joint soft-information protocol and its high-dimensional embedding. Section V details the experimental setup and evaluates the protocol’s performance on IBM quantum processors. Finally, Section VI concludes the paper.

## II. MCM CIRCUITS AND THE BAYES RISK BOUND

This section reviews MCM workloads, dispersive readout, NN-based QSD, and the Bayes-risk limit of a fixed observation space.

## A. MCM Circuit

A mid-circuit measurement is performed before circuit execution ends. Its outcome can guide subsequent gates, update a correction frame, or support repeated syndrome extraction. Reset and reuse provide another application: a measured physical qubit can be reinitialized and assigned a later role, reducing the physical-qubit footprint [15–17]. Reset is therefore part of some MCM workloads, rather than a requirement for every MCM.

Reliable MCM execution depends on both the quantum operations and the interpretation of their readout. An assignment error changes the inferred classical outcome and can lead to incorrect feedback or frame tracking. Reset infidelity concerns the state left by reinitializa tion. It is a separate error mechanism, even though active reset may use a measured outcome to select a correction. On superconducting devices, measurement backaction, crosstalk, and decoherence during measurement and feedback delays can further afect circuit performance [4, 18]. The workloads considered here cover repeated measurement and reuse, parity extraction, and teleportation with corrections interpreted through a Pauli frame.

## B. Dispersive Readout in Dynamic Circuits

In superconducting qubit systems, dispersive readout is the standard technique for measuring the state of a qubit. This method relies on the interaction between the qubit and a readout resonator, where the qubit state shifts the resonator frequency, allowing the state to be inferred from the transmitted or reflected microwave signal. The Hamiltonian describing this interaction in the dispersive regime is given by

$$
H / \hbar = \omega _ { r } a ^ { \dagger } a + \frac { \omega _ { q } } { 2 } \sigma _ { z } + \chi a ^ { \dagger } a \sigma _ { z } ,\tag{1}
$$

where $\omega _ { r }$ and $\omega _ { q }$ are the resonator and qubit frequencies, respectively, $a ^ { \dagger }$ and a are the creation and annihilation operators of the resonator mode, $\sigma _ { z }$ is the Pauli-z operator of the qubit, and χ is the dispersive shift.

After the interaction, the transmitted or reflected microwave signal carries information about the qubit state. The signal is typically amplified, integrated, and demodulated, yielding a voltage signal whose in-phase and quadrature (I–Q) components encode the qubit state information. On IBM hardware, a readout signal is typically a point on the I–Q plane, as shown in Fig. 1(c). A discriminator, such as a threshold or machine-learningbased classifier, processes the I–Q signal to infer the qubit state, typically distinguishing between the ground and excited states.

In an MCM circuit, however, Eq. (1) does not fully capture the measurement context. During circuit execution, the measured qubit can be afected not only by intrinsic relaxation during lifetime $T _ { 1 }$ and dephasing during time $T _ { 2 }$ , but also by mid-circuit measurement backaction, crosstalk from neighboring qubits, and measurementinduced errors. Residual photons in a readout resonator can also disturb subsequent qubit operations, motivating cavity-depletion techniques [12, 13]. Prior drift studies examine changes in device parameters over calibration windows [10, 11]. CaliQEC instead enables in situ device calibration during surface-code operation [19]. It addresses a diferent calibration task from learning a readout discriminator at a particular MCM cycle. In MCM circuits, by contrast, state deterioration reflected in these readout changes can occur on microsecond-level timescales within a single circuit execution.

## C. Quantum State Discrimination from SPAM Calibration

For superconducting platforms, QSD converts the measured analog signal into a classical state assignment. Lienhard et al. demonstrate NN-based QSD for multiplexed readout, improving assignment accuracy by accounting for system-dependent nonidealities such as crosstalk [6]. Maurya et al. combine hierarchical matched filters with a compact neural network to reduce the hardware cost of discrimination [7]. Vora et al. implement neural inference on an FPGA to support real-time MCM control [8]. These works address complementary accuracy, scalability, and latency challenges.

Our comparison concerns the calibration data supplied to a discriminator. The SPAM baseline in this paper is trained on isolated prepared-state references. The contextual alternative includes preceding workload operations and matches the measurement cycle. The QR-Stabilizer experiment shows that these acquisition conditions can produce diferent readout distributions. We evaluate this calibration dependence directly in the QR-Stabilizer experiments.

Encoding-based methods provide a complementary approach. Repetition-style readout schemes add auxiliary measurements and decode their hard assignments [20, 21]. Coherent Pauli checks instead use check circuits, including one-sided checks for measurement-error mitigation [22]. They are not an ordinary repetition-majority decoder. Soft-information decoding retains analog confidence for code-specific decoding [23]. These distinctions motivate comparing hard repetition decoding with neural decoding of joint I–Q readout signals in the expanded observation space.

## D. Bayes Risk Bound of Discrimination

Because QSD is a classification problem, a quantitative way to understand this limitation is through the concept of Bayes risk [9], which gives the minimum possible error probability for a classification task. In superconductingqubit readout, this means that any discriminator assigning an analog measurement observation to a quantumstate label has an error rate no lower than the Bayes risk associated with its observation space, regardless of architecture or training strategy.

Formally, we have

Definition II.1 (Bayes risk). Given random variables $( X , Y )$ taking values in an observation space and a label set ${ \mathcal { V } } ,$ respectively, let $g ^ { * }$ be the Bayes classifier defined by [9, p. 5]

$$
g ^ { * } ( x ) = \arg \operatorname* { m a x } _ { y \in \mathcal { Y } } P ( Y = y \mid X = x ) ,\tag{2}
$$

where $P ( Y = y ~ \vert ~ X = x )$ is the posterior probability of class y given observation $x .$ The Bayes risk $R ^ { * }$ is then the probability of error of this optimal classifier:

$$
R ^ { * } = P \bigl ( g ^ { * } ( X ) \neq Y \bigl ) .\tag{3}
$$

When there are only two classes, say $\mathcal { V } = \{ 0 , 1 \}$ , which is the exact case for single-qubit quantum state discrimination, the Bayes classifier in Eq. (2) takes the particularly simple form

$$
\begin{array} { r } { g ^ { * } ( x ) = \left\{ \begin{array} { l l } { 1 } & { \mathrm { i f } \ P ( Y = 1 \mid X = x ) \ge 1 / 2 , } \\ { 0 } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}
$$

For equal class priors, $P ( Y = 0 ) = P ( Y = 1 ) = 1 / 2$ , the corresponding Bayes risk in Eq. (3) has the form

$$
R ^ { * } = \frac { 1 } { 2 } - \frac { 1 } { 4 } \int | f _ { 0 } ( x ) - f _ { 1 } ( x ) | d x ,\tag{4}
$$

where $f _ { y } ( x )$ is the conditional probability density of the continuous observation X given $Y = y$ . For any classifier $g ( x )$ , its error probability satisfies $R ( g ) \geq R ^ { * } [ 9$ , Theorem 2.1].

For d readout channels, let $\mathbf { x } \in \mathbb { R } ^ { 2 d }$ collect their joint I–Q readout signals, let $\pi _ { y } ~ = ~ P ( Y ~ = ~ y ) ~$ be the class prior, and let $f _ { y } ^ { ( d ) }$ be the corresponding class-conditional density. The Bayes risk can also be written as

$$
R _ { 2 d } ^ { * } = \int _ { \mathbb { R } ^ { 2 d } } \operatorname* { m i n } \{ \pi _ { 0 } f _ { 0 } ^ { ( d ) } ( \mathbf { x } ) , \pi _ { 1 } f _ { 1 } ^ { ( d ) } ( \mathbf { x } ) \} d \mathbf { x } .\tag{5}
$$

Because the Bayes risk is a fundamental lower bound on the error probability, it limits how well any quantum state discriminator can perform on single-qubit data, regardless of the sophistication of the training strategy or architecture. Even in situ training at a particular measurement cycle can only move the discriminator toward the cycle-specific lower bound. It cannot reduce the error below the Bayes risk determined by the overlapping distributions. Thus, once MCM dynamics cause the analog distributions for 0 and 1 to overlap, no discriminator trained on the same single-qubit observation space can reduce its error below $R ^ { * }$

The next section makes this limitation concrete by examining state deterioration through changes in the readout distribution of a reused qubit relative to SPAM calibration during a representative IBM superconductinghardware MCM circuit.

## III. PHENOMENOLOGY OF STATE DETERIORATION IN MCM CIRCUITS

We now examine the state deterioration that occurs in MCM circuits on IBM superconducting-qubit hardware. The background in Section II establishes that SPAMtrained discriminators are limited by the Bayes risk of their observed single-qubit I–Q distributions. Here, we show that the relevant I–Q distributions in an MCM circuit can difer sharply from the SPAM calibration distribution used for training.

The QR-Stabilizer readout distribution departs increasingly from the SPAM reference as the circuit proceeds, with the final-cycle observations spread broadly across the I–Q plane (Fig. 1). We observe this trend on ibm fez using the acquisitions described in Appendix B 1. This evidence of state deterioration motivates matching calibration to the measurement context.

Possible contributors to state deterioration and the observed distributional changes include relaxation and dephasing of the measured data qubit, measurement backaction, and gate errors. Residual photons in the readout resonator can also perturb subsequent operations [12, 13]. Errors on the phase-kickback ancilla can also afect the data qubit through their two-qubit interaction. The present data do not separately isolate these mechanisms.

To characterize state deterioration through changes in the MCM readout signals in Fig. 1(c) relative to the SPAM calibration data in Fig. 1(b), we compute the Bayes risk between the distribution of the readout signals in the MCM circuit and the distribution of the SPAM calibration signals. If we denote the distribution of the MCM readout signals by $f _ { 1 } ^ { c _ { j } }$ , where the ideal result for the measured data qubit is 1 at measurement cycle $c _ { j } ,$ and the distribution of the SPAM calibration signals by $f _ { \mathrm { 0 } } ^ { \mathrm { S P A M } }$ , then according to Eq. (4), the Bayes risk is

$$
R ^ { * } = \frac { 1 } { 2 } - \frac { 1 } { 4 } \int _ { \mathbf { x } \in \mathbb { R } ^ { 2 } } | f _ { 1 } ^ { c _ { j } } ( \mathbf { x } ) - f _ { 0 } ^ { \mathrm { S P A M } } ( \mathbf { x } ) | d \mathbf { x } .
$$

Figure 1(d) shows increasing SPAM-trained NN-based QSD error alongside the cycle-dependent overlap estimate and the transferred digital-error estimate. The overlap calculation pairs a static 0 reference with each ideal-1 target cycle, whereas the NN-based QSD error is evaluated on target-workload shots. Their separation is therefore not an estimate of the classifier’s excess risk above a Bayes optimum for the same task.

The evidence of state deterioration in Fig. 1(b,c) motivate in situ calibration, whose benefit is tested directly in Section IV A. Independently of this experiment, Bayes risk limits classification for any fixed set of observations. Adding an auxiliary readout ofers a way to change those observations and potentially reduce their overlap. The gap between the plotted curves alone does not establish that this change is necessary.

## IV. FRAMEWORK

Lifting combines context-matched calibration, repetition encoding, and neural decoding. Calibration adapts the decoder to the workload, while encoding supplies auxiliary observations that can distinguish outcomes overlapping in a single-qubit I–Q plane. We first describe these components, then derive an overlap bound and introduce optional confidence selection for evaluated shots.

## A. In Situ Calibration for State Deterioration

The QR-Stabilizer measurements in Section III show state deterioration through changes in readout distribu tions within a circuit. We therefore acquire calibration data at the workload’s measurement locations, including the preceding operations that shape the readout context. This procedure aims to approximate the conditions seen by target shots. Preparation diferences and hardware variability can still produce a mismatch.

The calibration circuits in Fig. 2(a) reproduce the relevant circuit depth, spectator activity, and measurement timing while preparing reference labels $y \in \{ 0 , 1 \}$ . For a single readout channel, they produce the labeled dataset

$$
\mathcal { D } _ { \mathrm { c a l } } = \{ ( { \bf x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n } ,\tag{6}
$$

where each $\mathbf { x } _ { i } ~ \in ~ \mathbb { R } ^ { 2 }$ represents a single-shot measurement vector, and $y _ { i } \in \{ 0 , 1 \}$ is its corresponding known preparation label.

Figure 3 compares the SPAM-trained and cyclematched single-channel decoders on QR-Stabilizer target shots. The visible improvement is concentrated in the early measurements, while later-cycle errors remain high in this experiment. This limited benefit motivates adding readout information as well as changing calibration. The constructed risk curve provides a separate overlap diagnostic rather than a matched lower bound for these target-shot errors.

## B. Repetition Encoding

For a given MCM circuit, Lifting leaves the circuit unchanged except at measurement locations. Immediately before measuring a target qubit, the protocol allocates (d  1) auxiliary physical qubits and forms a (d, 1) repetition-style encoding, as shown in Fig. 2(b). If the target qubit is

$$
\left| \psi \right. = a \left| 0 \right. + b \left| 1 \right. ,
$$

then we apply (d 1) controlled-NOT (CNOT) gates from the target to the auxiliary qubits, producing

$$
a \left| 0 \right. + b \left| 1 \right. \longmapsto a \left| 0 \right. ^ { \otimes d } + b \left| 1 \right. ^ { \otimes d } .\tag{7}
$$

![](images/db74c0536dc15944063d3def2f43f7e2017d7f77b34a907dfc5c692e68155a64.jpg)

![](images/4e63891849e53c33705ef38c46be2369edc82158809de515c44945ee565f5893.jpg)

(d)  
![](images/51666bfb37a2bb2bb3dca557bacf1988f2f90782b039ec5c63fada1317ce0097.jpg)

(c)  
![](images/07c92cc079ce20713b35f3156dda78c1978a74a6c2126c6b9dd4eae7b8678bb1.jpg)  
FIG. 1. State deterioration reflected in readout-distribution changes in the five-cycle QR-Stabilizer workload. (a) Core circuit with a measured data qubit and a phase-kickback ancilla. Lifting proxy wires are omitted. (b) SPAM references for preparation labels 0 (blue) and 1 (red) on qubit 51 of ibm fez. 10,000 shots per class are sampled from the reference acquisitions for display. (c) Target-readout I–Q distributions ordered from cycles $c _ { 1 }$ to $c _ { 5 }$ . (d) Equal-prior kernel density estimation (KDE) Bayes-risk estimates comparing the SPAM-0 reference with each ideal-1 target cycle, SPAM-trained NN-based QSD error rates, and transferred digital-error estimates. The curve labeled ibm fez transfers the accuracy ofset between NN-based QSD and digital readout from a separate acquisition. It is not native digital readout of the same target shots.

![](images/92d2e47e5684811902bfb57ccc09b5bb28798e5af1e8f74e33d4c8ca9fe0c230.jpg)  
FIG. 2. Lifting workflow. (a) Context-matched calibration executions prepare reference labels and place encoded readout at the workload’s measurement locations. (b) Local repetition encoding couples the labeled MCM qubit to d − 1 auxiliary ancillas initialized to |0⟩. Concurrent readout supplies their joint I–Q readout signals. (c) The d I–Q pairs are concatenated into a 2d-component feature vector. Reference labels supervise neural training, and inference assigns a logical outcome to each target observation. Optional confidence selection retains a subset of observations before reporting retained-shot accuracy.

This encoding is the same physical resource used by active readout-error mitigation and repetition-code-based schemes [20, 21, 24]. The diference is in the decoder. Instead of independently thresholding each qubit and applying majority voting, all d qubits are measured concurrently and the joint I–Q readout vector is decoded.

The auxiliary qubits must be initialized to 0 before the CNOT fanout. When auxiliaries are reused within a circuit, a reset prepares them for the next encoded measurement. Imperfect initialization and added gates can reduce the benefit of the auxiliary observations, so their costs must be assessed on hardware.

The d encoded qubits are then measured simultaneously using multiplexed dispersive readout. Each shot produces d pairs of in-phase and quadrature values,

$$
\mathbf { x } ^ { ( d ) } = ( I _ { 1 } , Q _ { 1 } , \ldots , I _ { d } , Q _ { d } ) \in \mathbb { R } ^ { 2 d } .\tag{8}
$$

This vector is the basic input to analog filtering and neural decoding. With the calibrated distributions and the joint I–Q readout space defined, we can now derive the limits of their distinguishability.

![](images/ce291ac50b0854a6f7a76e07dbb5d3382812928ce520d79152c28d995c903898.jpg)  
FIG. 3. Error rates across five MCM cycles: the KDE Bayesrisk estimate, SPAM-trained NN-based QSD, the transferred ibm fez digital estimate from Fig. 1(d), and single-channel NN-based QSD with cycle-matched in situ calibration. The risk curve compares constructed reference–target distributions. The decoder curves evaluate target shots. Contextual calibration changes the training references, while this comparison retains the single-channel observation space.

The repetition encoding in Fig. 2(b) also applies to the in situ calibration circuits in Fig. 2(a). Replacing their ordinary measurements with encoded readout produces 2d-dimensional joint I–Q readout vectors for training. Equation (6) then becomes

$$
\begin{array} { r } { \mathcal { D } _ { \mathrm { c a l } } = \left\{ \left( \mathbf { x } _ { i } ^ { ( d ) } , y _ { i } \right) \right\} _ { i = 1 } ^ { n } , } \end{array}\tag{9}
$$

where each $\mathbf { x } _ { i } ^ { ( d ) } \in \mathbb { R } ^ { 2 d }$ is the 2d-dimensional joint I–Q readout vector corresponding to the ith shot and $y _ { i } \in$ $\{ 0 , 1 \}$ is the associated class label.

## C. Upper Bounds on the Joint Bayes Risk

An additional qubit can separate outcomes that appear indistinguishable in the original observation space. Let $f _ { y } ^ { ( d ) }$ denote the density of the first d readout channels for label y, and let $\rho _ { d } ~ = ~ \mathrm { B C } ( f _ { 0 } ^ { ( d ) } , f _ { 1 } ^ { ( d ) } )$ be their Bhattacharyya coeficient (BC), which measures distributional overlap [25]. The following result separates the benefit of retaining more observations from the stronger assumption that each added channel uniformly contracts overlap.

Theorem IV.1. Consider joint I–Q readout vectors consisting of the first d channels of one fixed joint distribution, with equal class priors and class-conditional densities. The optimal error $R _ { 2 d } ^ { * }$ cannot increase when another channel is retained. If the conditional overlap of each added channel k is uniformly bounded by $\gamma _ { \mathrm { m a x } , k } < 1$

(a)  
![](images/1165f84276372fecd888dc5bc02372e4354e4f536f83cd6ff9d78f883ee991e9.jpg)  
FIG. 4. Qualitative geometry of marginal and conditional overlap. (a) The preceding $d - 1$ readout channels have marginal Bhattacharyya overlap $\rho _ { d - 1 } = \mathrm { B C } ( f _ { 0 } ^ { ( d - 1 ) } , f _ { 1 } ^ { ( d - 1 ) } )$ which upper-bounds balanced-prior Bayes risk after multiplication by $1 / 2$ . (b) Conditional overlap in the added I–Q channel is bounded by $\gamma _ { \mathrm { m a x , } d } .$ Shaded regions schematically illustrate the factorization.

then

$$
R _ { 2 d } ^ { * } \leq \frac { 1 } { 2 } \rho _ { d } \leq \frac { 1 } { 2 } \rho _ { 1 } \prod _ { k = 2 } ^ { d } \gamma _ { \operatorname* { m a x } , k } .\tag{10}
$$

If all factors are bounded by a common $q < 1$ , this upper bound decreases geometrically as $\textstyle { \frac { 1 } { 2 } } \rho _ { 1 } q ^ { d - { \bar { 1 } } }$

To see the geometry, divide the observation into the preceding channels $\mathbf { u } \stackrel { } { = } \mathbf { x } ^ { ( d - 1 ) }$ and the added channel $\mathbf { z } = \mathbf { x } _ { d }$ . Their joint density factors as

$$
f _ { y } ^ { ( d ) } ( \mathbf { u } , \mathbf { z } ) = f _ { y } ^ { ( d - 1 ) } ( \mathbf { u } ) f _ { y } ( \mathbf { z } \mid \mathbf { u } ) .\tag{11}
$$

At a fixed location u, the added channel provides another view of the two classes (Fig. 4). Its conditional overlap $\gamma _ { d } ( \mathbf { u } )$ weights the overlap remaining in the preceding channels, giving

$$
\begin{array} { l } { \displaystyle { \rho _ { d } = \int \sqrt { f _ { 0 } ^ { ( d - 1 ) } ( { \bf u } ) f _ { 1 } ^ { ( d - 1 ) } ( { \bf u } ) } \gamma _ { d } ( { \bf u } ) d { \bf u } } } \\ { \displaystyle { \quad \le \gamma _ { \mathrm { m a x } , d } \rho _ { d - 1 } . } } \end{array}
$$

Repeating this step gives the product bound. The exact optimal error is non-increasing because a decoder can always ignore the new channel. Strict improvement requires the added observations to split a region previously assigned one label into regions favoring diferent classes, with positive probability. Appendix A gives the precise conditions and the Gaussian special case.

The theorem compares coordinates of a fixed distribution. Physically increasing the encoding distance also changes gates and measurement conditions, so separately executed distance sweeps need not share the required marginals. The hardware evaluation therefore examines the benefit of joint I–Q readout signals together with the costs of implementing them.

## D. Decoding

## 1. Analog Confidence Filtering

Confidence filtering selects evaluated shots whose decoded outcomes are suficiently well supported by their analog observations. It is optional and trades retained shot coverage for accuracy. It is neither an assumption of Theorem IV.1 nor a modification of the reference labels used for training. The QR-Stabilizer implementation uses a Gaussian confidence score. The quantum error correction (QEC) workload considered here consists of five parity-check rounds and is denoted QEC-5. This workload and teleportation use neural posterior confidence, as specified in Appendix B 2.

Let $\mathcal { D } _ { \mathrm { c a l } } ^ { 0 }$ and $\mathcal { D } _ { \mathrm { c a l } } ^ { 1 }$ be the subsets of the in situ calibration dataset in Eq. (9) containing observations labeled 0 and 1, respectively. For QR-Stabilizer, using a multivariate Gaussian approximation for each reference class, we can estimate the mean and covariance of the joint I–Q readout vector:

$$
\hat { \pmb { \mu } } _ { y } = \frac { 1 } { | \mathcal { D } _ { \mathrm { c a l } } ^ { y } | } \sum _ { { \bf x } _ { i } \in \mathcal { D } _ { \mathrm { c a l } } ^ { y } } { \bf x } _ { i } ,\tag{12}
$$

$$
\hat { \mathbf { \Sigma } } _ { y } = \frac { 1 } { | \mathcal { D } _ { \mathrm { c a l } } ^ { y } | - 1 } \sum _ { \mathbf { x } _ { i } \in \mathcal { D } _ { \mathrm { c a l } } ^ { y } } ( \mathbf { x } _ { i } - \hat { \pmb { \mu } } _ { y } ) ( \mathbf { x } _ { i } - \hat { \pmb { \mu } } _ { y } ) ^ { T } .\tag{13}
$$

With a small diagonal regularizer $\lambda I ,$ define the Gaussian score

$$
\begin{array} { l } { s _ { y } ( { \bf x } ) = - \displaystyle \frac { 1 } { 2 } ( { \bf x } - { \hat { \pmb \mu } _ { y } } ) ^ { T } ( \hat { \pmb \Sigma } _ { y } + \lambda I ) ^ { - 1 } ( { \bf x } - { \hat { \pmb \mu } _ { y } } ) } \\ { \displaystyle - \displaystyle \frac { 1 } { 2 } \log \operatorname* { d e t } ( \hat { \pmb \Sigma } _ { y } + \lambda I ) + \log \hat { \pi } _ { y } , } \end{array}\tag{14}
$$

where $\hat { \pi } _ { y } = | \mathcal { D } _ { \mathrm { c a l } } ^ { y } | / | \mathcal { D } _ { \mathrm { c a l } } |$ is the empirical class prior. The posterior confidence is

$$
c ( \mathbf { x } ) = \operatorname* { m a x } _ { y \in \{ 0 , 1 \} } { \frac { \exp ( s _ { y } ( \mathbf { x } ) ) } { \exp ( s _ { 0 } ( \mathbf { x } ) ) + \exp ( s _ { 1 } ( \mathbf { x } ) ) } } .\tag{15}
$$

An evaluated QR-Stabilizer shot is retained if $c ( \mathbf { x } ) \geq$ τ at every measurement cycle, for a chosen threshold τ. Equivalently, one can threshold the absolute loglikelihood ratio $| s _ { 1 } ( \mathbf x ) - s _ { 0 } ( \mathbf x ) |$ . This formulation is the analog counterpart of digital detection or postselection: it rejects outcomes whose calibration-derived likelihood is too close to the decision boundary. Unlike majority voting, however, the confidence is computed in the full 2d-dimensional I–Q feature space.

## 2. Neural Decoding

The overlap analysis describes the optimal classification limit for the chosen observations. A neural decoder provides a flexible approximation to the corresponding decision rule. The following existence result, proved in

Claim A.1 of Appendix A, motivates this choice without guaranteeing that finite training reaches the optimum.

Proposition IV.2. For any $\epsilon > 0 .$ , there exists a neural network $f _ { \pmb { \theta } } \in \mathcal { H } _ { 2 d }$ such that the misclassification risk $R ( g _ { \theta } ) ~ = ~ { \cal P } ( g _ { \theta } ( X ) ~ \not = ~ Y )$ of the thresholded classifier $g _ { \theta } ( \mathbf { x } ) = 1 \{ f _ { \theta } ( \mathbf { x } ) \geq 1 / 2 \}$ is bounded by $R _ { 2 d } ^ { * } + \epsilon ,$ , where $R _ { 2 d } ^ { * }$ is the Bayes risk in Theorem IV.1 and $\mathcal { H } _ { 2 d }$ is a sufficiently expressive family of continuous feedforward neural networks operating on the 2d-dimensional joint I–Q readout space.

Using the labeled calibration observations, we train a neural decoder

$$
f _ { \pmb { \theta } } : \mathbb { R } ^ { 2 d }  [ 0 , 1 ] ,
$$

where $f _ { \pmb { \theta } } ( \mathbf { x } )$ estimates the probability that the logical outcome is 1. We use a multilayer perceptron (MLP), consistent with prior NN-based QSD methods and FPGA-oriented single-shot discrimination systems [6–8]. The input layer has 2d nodes, one for each I–Q component in Eq. (8). The hidden layers use rectified linear activations, and the output layer uses a sigmoid activation for binary logical readout. The MCM experiments use one hidden layer of 32 neurons. The output heads and training settings are specified in Appendix B 2.

The decoder is trained on the in situ calibration set by minimizing binary cross entropy, as illustrated in Fig. 2(c),

$$
\mathcal { L } ( \pmb \theta ) = - \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left[ y _ { i } \log f _ { \pmb \theta } ( \mathbf { x } _ { i } ) + ( 1 - y _ { i } ) \log ( 1 - f _ { \pmb \theta } ( \mathbf { x } _ { i } ) ) \right] .
$$

During inference, each encoded measurement produces a joint I–Q readout vector $\mathbf { x } \in \mathbb { R } ^ { 2 d }$ . The logical single-shot output is

$$
\hat { y } = \left\{ \begin{array} { l l } { 1 , } & { f _ { \theta } ( \mathbf { x } ) \ge 1 / 2 , } \\ { 0 , } & { f _ { \theta } ( \mathbf { x } ) < 1 / 2 . } \end{array} \right.
$$

The decoder therefore replaces the conventional pipeline of independent qubit discrimination followed by majority voting with a direct learned decision boundary in the joint I–Q readout space.

## V. EXPERIMENTAL RESULTS

## A. Hardware Readout Geometry and Empirical Accuracy Model

We examine the benefit of joint I–Q readout signals on IBM hardware in two steps. First, we compare the target-only and joint I–Q readout distributions from a distance-two acquisition. We then use a fitted empirical model to describe how assignment accuracy changes with the physical encoding distance, where additional gates and readout channels can introduce competing costs.

Figure 5 shows less overlap in the quadraticdiscriminant scores computed from joint I–Q readout signals than in the target-only scores. The equal-prior KDE risk estimate falls from 14.7% for the target channel to 5.9% for the joint readout. This supports the benefit of retaining auxiliary information from the same acquisition.

For the distance sweep, we use a phenomenological product model inspired by the overlap analysis. Let $\epsilon _ { Y , k }$ be the empirical marginal assignment-error input for preparation label $Y$ , with $k = 0$ denoting the target and $k = 1 , \ldots , d - 1$ its auxiliaries. Define the fitted auxiliary factors

$$
g _ { Y , k } = ( 2 \epsilon _ { Y , k } ) ^ { \kappa } , \qquad 0 < \kappa < 1 ,\tag{16}
$$

and the modeled class-specific error

$$
{ \widehat { P } } _ { \mathrm { e r r } } ( Y ) = \epsilon _ { Y , 0 } \prod _ { k = 1 } ^ { d - 1 } g _ { Y , k } .\tag{17}
$$

The factors $g _ { Y , k }$ summarize how the marginal error inputs combine in the fit. They are empirical model factors, rather than conditional Bhattacharyya overlaps. Thus, this fitted model and the measured risk reduction illustrate the value of auxiliary signals without establishing the conditional-overlap assumptions of Theorem IV.1.

We include an asymmetric attenuation factor to represent the implementation cost of the encoding. With $e _ { g } ^ { ( j ) }$ denoting the two-qubit gate-error input for encoding interaction $j ,$ , the balanced-class accuracy model is

$$
\begin{array} { r l } & { A _ { \mathrm { f i t } } = \displaystyle \frac { 1 } { 2 } \Bigg [ 1 - \widehat { P } _ { \mathrm { e r r } } ( 0 ) } \\ & { \qquad + \displaystyle \prod _ { j = 1 } ^ { d - 1 } ( 1 - \alpha e _ { g } ^ { ( j ) } ) \big ( 1 - \widehat { P } _ { \mathrm { e r r } } ( 1 ) \big ) \Bigg ] . } \end{array}\tag{18}
$$

This model uses $0 \le \epsilon _ { Y , k } \le 1 / 2$ and $0 \leq \alpha e _ { g } ^ { ( j ) } \leq 1$ , so the factors and modeled accuracy remain in $[ 0 , 1 ]$ . The extra attenuation of the class-one branch represents an asymmetric implementation cost without assigning it to a specific noise mechanism.

Figure 6 shows that the fitted model follows the hardware accuracy trend over the tested Fez distances. The parameters $\kappa \ : = \ : 0 . 4 5$ and $\alpha \ = \ 9 . 0 3$ are fitted to this sweep. The trend is consistent with a balance between useful auxiliary information and the errors introduced by implementing a larger encoding.

## B. Comparison with Repetition and Temporal Schemes

We first evaluate how the repetition encoding scales with the code distance d for spatial readout and sequential temporal acquisition. Figure 7 compares the baseline readout, a distance-d repetition code, a temporal repeated-measurement scheme, and our encoding on three IBM superconducting processors. The experiments use neighboring qubits on each device, selecting a target qubit and then adding nearby auxiliary qubits as d increases. The baseline reports the average native assignment accuracy of the selected physical qubits. The spatial repetition comparator independently discriminates the concurrent readouts and applies a hard vote. The temporal comparator applies neural decoding to d sequential observations of one qubit. Lifting applies neural decoding to the joint I–Q readout signals from the d encoded qubits in Section IV B. These schemes increase physical-qubit count or measurement time diferently as d grows.

(a)  
![](images/d672b6fb059ffec9763bf0f3eada584ee181c68201f07db8fcb90098d4af7c3a.jpg)

(c) <sub>2D</sub> <sub>Decision</sub>  
![](images/dd1264c1876db393c42a36948df3862c32492c94789aecf30267253ef07ff0a5.jpg)  
(d)<sub>4D Joint Decision</sub>

(b) Proxy  
![](images/72449114009d64896e796cf70bef366b621fe20cd474b35f9182dfdcec00c8f6.jpg)

![](images/d1f740f28eb250e2912d33522b6637cdf07cd3c83da5fc3c69c883efa0e88030.jpg)  
I (arb.)

FIG. 5. Target-only and joint I–Q readout discrimination for $d = 2$ on ibm fez, using 10,000 shots per preparation label. (a,b) Target and auxiliary (proxy) I–Q scatter, with blue and red denoting reference labels 0 and $^ { 1 , }$ respectively. $^ { \mathrm { ( c , d ) } }$ Signed quadratic-discriminant scores $s _ { 1 } \mathrm { ~ - ~ } s _ { 0 }$ from the twodimensional target observations and four-dimensional joint $_ \mathrm { I - Q }$ readout signals. Annotated risks are equal-prior KDEbased estimates of distributional overlap, rather than held-out classifier error rates.  
![](images/6940f1ed7f8cb441f71f0a1e3d5632c3c1f6d62318c7e08d94806e0231e5a3a3.jpg)  
FIG. 6. IBM Fez assignment accuracy versus encoding distance $d ,$ comparing hardware measurements (emp.) with the fitted empirical model in $\operatorname { E q . }$ (18). The curve labeled theo. uses $\kappa = 0 . 4 5$ and $\alpha = 9 . 0 3$ , fitted for the displayed distance sweep.

The hard-vote comparator benefits unevenly from increasing distance, with a pronounced loss at $d = 2$ and no consistent gain from $d = 3$ to $d = 4$ . At even distances, conflicting assignments can produce a tie, so the result depends on how that tie is resolved. Additional gate and readout errors can further ofset the benefit of another qubit.

Lifting generally has higher assignment accuracy than hard repetition decoding, while the curves coincide at $d \ = \ 1$ At distance two, the relative error reductions over distance-two hard repetition decoding are 80.4%, 82.5%, and 79.8% on ibm fez, ibm kingston, and ibm marrakesh, respectively. Neural decoding of joint I–Q readout signals can use confidence information lost by independent hard assignments, and these results sup port distance two as a practical choice with one auxiliary qubit. Larger distances provide smaller or devicedependent gains.

Lifting also generally exceeds the temporal comparator, with coincident starting points and a near tie on Fez at the largest distance. Although both neural decoders receive 2d scalar features, spatial encoding uses more qubits and entangling gates, whereas temporal acquisition uses repeated measurements and more time. Their diferent circuits and acquisition conditions make this a comparison of implemented schemes, rather than an isolated test of feature dimension or a single noise mechanism.

## C. Distance-Two Accuracy and Auxiliary-Qubit Cost

Figure 8 examines the selected distance-two operating point across qubit pairs with diferent native assignment accuracies. The bars base0 and base1 identify the individual-qubit baselines, while hard repetition decoding and Lifting process the same two-qubit encoding. The pairs include both similar-quality qubits and cases with a substantial diference between their native accuracies.

Lifting exceeds hard repetition decoding for every displayed pair, including pairs with unequal native accuracies. Against the individual-qubit baselines, Lifting is higher in many cases and approximately ties the stronger qubit in some high-accuracy pairs. The consistent gain over hard repetition supports using analog confidence to resolve ambiguous two-qubit records. With one auxil iary qubit, distance two provides a practical accuracy– resource tradeof for the subsequent MCM experiments.

## D. Performance on MCM Workloads

We next evaluate distance-two Lifting on QR-Stabilizer, QEC-5, and teleportation, using the circuits and reference labels defined in Appendix B 1. These workloads test repeated measurement and reuse, parity-based error tracking, and recovery of a teleported computational-basis input, respectively. We distinguish the accuracy of each individual measurement from the success rate of the circuit under its workload-specific criterion.

Figure 9 shows the per-cycle discrimination accuracy for each measurement in QR-Stabilizer: both series decline as the circuit proceeds, while Lifting with in situ calibration remains higher throughout. The downward trend is consistent with the state deterioration identified in Sec. III, which makes later measurements harder to classify. Because NN-based QSD is evaluated on target shots and Lifting on held-out calibration shots, their separation does not establish a paired improvement on the same workload shots.

Figure 10 instead reports circuit success rates under the criteria in Appendix B 1: all QR-Stabilizer outcomes must be correct, QEC-5 must identify the final correction frame, and teleportation must recover the prepared input bit. For QEC-5 and teleportation, corrections are tracked in software rather than applied through physical conditional gates. These criteria test the expected circuit behavior through decoded measurement records. They do not measure corrected-state or arbitrary-state teleportation fidelity.

For QR-Stabilizer, Lifting with in situ calibration has the highest unfiltered success rate on both devices, whereas contextual calibration alone changes singlechannel NN-based QSD little and repetition encoding alone gives a smaller, device-dependent gain. On ibm fez, combining encoding and contextual calibration raises the success rate from 54.2% to 58.2%. For these QR-Stabilizer experiments, the combined configuration improves success more than single-channel contextual re training alone.

For QEC-5, Lifting with in situ calibration also achieves the highest unfiltered success rate on both devices, although its advantage over NN-based QSD with in situ calibration is small on ibm marrakesh. Lifting with static calibration is comparable to SPAM-trained NN-based QSD on ibm fez and lower on ibm marrakesh. Thus, matching the decoder to the repeated parity-check circuit is important to realizing the benefit of the extra readout channel in these results.

For teleportation, all Lifting variants outperform SPAM-trained NN-based QSD on both devices, but the ordering of the unfiltered variants depends on the device. Lifting with only Alice’s proxy and Lifting with both Alice’s and Bob’s proxies are nearly equal on ibm fez, whereas the latter is higher on ibm marrakesh. Both variants use in situ calibration, so this comparison indicates a device-dependent benefit from adding Bob’s

![](images/845a57ac6c2641da30a728763b0266b127bf13df35d8f216deeb1fe7370b81aa.jpg)  
(a) ibm fez.

![](images/6de5a4aa3dbe81e312a6e3cd6a827647e3db7b1c71138c44b4e0d3e7086bdbab.jpg)  
(b) ibm kingston.

![](images/0e1f6cd724a3ebe19f4b77ad1df3b09fdd98c5d0ef754301b1ed1562720971af.jpg)  
(c) ibm marrakesh.

FIG. 7. Assignment accuracy versus distance on (a) ibm fez, (b) ibm kingston, and (c) ibm marrakesh. The baseline averages native assignment accuracy over the selected physical qubits. Spatial repetition (rep-d) uses independent hard assignments followed by voting. The temporal comparator (tem-d) jointly decodes d sequential I–Q observations of one qubit. Lifting (lif-d) jointly decodes concurrent I–Q observations from d repetition-encoded qubits. The spatial schemes increase physical-qubit count with d, whereas the temporal scheme increases measurement time. Both neural schemes use 2d scalar inputs.  
![](images/b6e3230b50261947e1514df2cd379b773df8fa2012e661da058f78fb8d379027.jpg)  
(a) ibm fez.

![](images/dd9d5d66de71bb72cfa6e789b2e1af7d8a17914ea8bb2e4ae483cd2b4d138687.jpg)  
(b) ibm kingston.

![](images/3a2ba478d5b8dd4ecef4755bd16b97fb9cb9a604494a530335d375a89858c01c.jpg)  
(c) ibm marrakesh.

FIG. 8. Distance-two assignment accuracy for physical-qubit pairs on (a) ibm fez, (b) ibm kingston, and (c) ibm marrakesh. Bracketed labels identify the selected qubit pairs. The base0 and base1 bars report native assignment accuracy of the first and second qubits, respectively. The rep-2 bars use hard repetition decoding. Lifting, labeled lif-2, uses neural decoding of joint I–Q readout signals from the same two-qubit encoding. Pair selection spans heterogeneous native readout accuracies.  
![](images/fb81118057a7b22da25b96607490a97200a7614c84cbb3a791fedb64d2d8da21.jpg)  
FIG. 9. Per-cycle QR-Stabilizer discrimination accuracy on ibm fez, in measurement order c –c . NN-based QSD uses target shots. Lifting with in situ calibration uses held-out calibration shots (Appendix B 2).

proxy readout.

Confidence filtering gives the highest retained-shot success rate in every workload and device shown. It removes ambiguous records, making the accepted outcomes more reliable at the cost of discarding shots. These green bars therefore describe a tradeof between accuracy and retained-shot coverage, with the workload-specific selection rules given in Appendix B 2.

## VI. DISCUSSION

We have presented Lifting, a quantum state discrimination framework for mid-circuit measurements on noisy superconducting processors. Repetition encoding supplies joint I–Q readout signals that allow the neural decoder to distinguish outcomes that overlap in a singlequbit readout plane. Complementary in situ calibration addresses the mismatch associated with state deterioration during circuit execution. The theoretical analysis establishes the benefit of retaining additional readout channels under a fixed joint distribution and identifies suficient conditions for contraction of an overlap-based upper bound on Bayes risk.

Experiments across three IBM processors demonstrate improved circuit success in representative MCM workloads, with distance two providing a practical balance between accuracy and auxiliary-qubit cost. The largest relative gain is approximately 13.5% in repeated stabilizer measurement, where confidence-filtered Lifting raises circuit success from the statically calibrated NN-based QSD baseline of 54.2% over all evaluated shots to 61.5% among retained shots. Across the workloads, the results support matching both the auxiliary readout channels and calibration context to the circuit.

![](images/06675df306c91ddf75e1bf1326a82c57c0e49b6b2d47849c30a8122e6eb5ce44.jpg)  
FIG. 10. Circuit success rates for (a) QR-Stabilizer, (b) QEC-5, and (c) teleportation. Success requires a correct complete QR Stabilizer trace, final QEC Pauli-frame bit, or recovered computational-basis input, respectively, as defined in Appendix B 1 Green bars report success among retained shots. Decoder configurations and filtering rules are given in Appendix B 2.

## Appendix A: Proof of Theorem IV.1

We first establish monotonicity of the exact Bayes risk, then derive the conditional-overlap bound. The argument applies to general class-conditional densities. Gaussian models provide a useful geometric special case [9, 25, 26].

## 1. Joint I–Q Readout Signals and Exact Risk

Fix a joint I–Q readout vector $\mathbf { X } ^ { \left( D \right) } \in \mathbb { R } ^ { 2 D }$ and a binary label Y. For $d \leq D$ , let $f _ { y } ^ { ( d ) }$ be the density of its first d channels conditioned on $Y \ = \ y ,$ with priors $\pi _ { y } = P ( Y = y ) $ . All lower-dimensional densities below are marginals of this same joint law. The Bayes risk is

$$
R _ { 2 d } ^ { * } = \int \operatorname* { m i n } \{ \pi _ { 0 } f _ { 0 } ^ { ( d ) } ( \mathbf { x } ) , \pi _ { 1 } f _ { 1 } ^ { ( d ) } ( \mathbf { x } ) \} d \mathbf { x } .\tag{A1}
$$

Write $\mathbf { u } = \mathbf { x } ^ { ( d - 1 ) }$ and $\mathbf { z } = \mathbf { x } _ { d }$ . Since the integral of a pointwise minimum is no larger than the minimum of

the two integrals,

$$
\begin{array} { l } { { \displaystyle R _ { 2 d } ^ { * } = \int \int \int \operatorname* { m i n } \{ \pi _ { 0 } f _ { 0 } ^ { ( d ) } ( { \bf u } , { \bf z } ) , \pi _ { 1 } f _ { 1 } ^ { ( d ) } ( { \bf u } , { \bf z } ) \} d { \bf z } d { \bf u } } } \\ { { \displaystyle ~ \leq \int \operatorname* { m i n } \{ \pi _ { 0 } f _ { 0 } ^ { ( d - 1 ) } ( { \bf u } ) , \pi _ { 1 } f _ { 1 } ^ { ( d - 1 ) } ( { \bf u } ) \} d { \bf u } } } \\ { { \displaystyle ~ = R _ { 2 ( d - 1 ) } ^ { * } } . } \end{array}
$$

Geometrically, retaining the added coordinate allows a finer decision boundary. The original classifier remains available by ignoring that coordinate, so optimal performance cannot worsen.

Strict improvement requires more than distinct conditional distributions. To state the condition precisely, define

$$
h ( \mathbf { u } , \mathbf { z } ) = \pi _ { 1 } f _ { 1 } ^ { ( d ) } ( \mathbf { u } , \mathbf { z } ) - \pi _ { 0 } f _ { 0 } ^ { ( d ) } ( \mathbf { u } , \mathbf { z } ) .
$$

The reduction in risk is

$$
\begin{array} { l } { { \displaystyle R _ { 2 ( d - 1 ) } ^ { * } - R _ { 2 d } ^ { * } = \frac { 1 } { 2 } \int \left[ \int | h ( \mathbf { u } , \mathbf { z } ) | d \mathbf { z } \right. } \ ~ } \\ { { \displaystyle \left. - \left| \int h ( \mathbf { u } , \mathbf { z } ) d \mathbf { z } \right| \right] d \mathbf { u } } . } \end{array}\tag{A2}
$$

This quantity is positive precisely when, on a set of u of positive measure, $h ( \mathbf { u } , \cdot )$ has both positive and negative parts with nonzero integrals. In that case, the added observation resolves a single marginal decision into regions favoring diferent classes. Conditional distributions can difer without crossing this decision threshold, leaving the exact risk unchanged.

## 2. Conditional Overlap and the Product Bound

Define the Bhattacharyya coeficient [25]

$$
\rho _ { d } = \int \sqrt { f _ { 0 } ^ { ( d ) } ( \mathbf { x } ) f _ { 1 } ^ { ( d ) } ( \mathbf { x } ) } d \mathbf { x } .\tag{A3}
$$

The inequality min $( a , b ) \leq { \sqrt { a b } }$ gives

$$
R _ { 2 d } ^ { * } \leq \sqrt { \pi _ { 0 } \pi _ { 1 } } \rho _ { d } .
$$

For the balanced priors in Theorem IV.1, the prefactor is $1 / 2$ . Equality is allowed, including when the two class distributions are disjoint and both sides vanish.

On the shared marginal support, factor the densities as

$$
f _ { y } ^ { ( d ) } ( \mathbf { u } , \mathbf { z } ) = f _ { y } ^ { ( d - 1 ) } ( \mathbf { u } ) f _ { y } ( \mathbf { z } \mid \mathbf { u } ) .
$$

The conditional coeficient is

$$
\gamma _ { d } ( \mathbf { u } ) = \int \sqrt { f _ { 0 } ( \mathbf { z } \mid \mathbf { u } ) f _ { 1 } ( \mathbf { z } \mid \mathbf { u } ) } d \mathbf { z } .\tag{A4}
$$

By Cauchy–Schwarz, $0 \leq \gamma _ { d } ( \mathbf { u } ) \leq 1$ , with equality at one exactly when the two conditional densities agree almost everywhere. Substitution into $\operatorname { E q . }$ . (A3) yields

$$
\rho _ { d } = \int \sqrt { f _ { 0 } ^ { ( d - 1 ) } ( \mathbf { u } ) f _ { 1 } ^ { ( d - 1 ) } ( \mathbf { u } ) } \gamma _ { d } ( \mathbf { u } ) d \mathbf { u } .\tag{A5}
$$

Thus, only locations where the marginal distributions overlap contribute to the contraction.

Let $\nu _ { d - 1 }$ be the finite measure with density $\sqrt { f _ { 0 } ^ { ( d - 1 ) } f _ { 1 } ^ { ( d - 1 ) } }$ relative to du. The uniform contraction assumption means

$$
\gamma _ { d } ( \mathbf { u } ) \leq \gamma _ { \operatorname* { m a x } , d } < 1 \quad \mathrm { f o r } ~ \nu _ { d - 1 } \mathrm { - a l m o s t ~ e v e r y ~ } \mathbf { u } .
$$

Equivalently, the essential supremum of $\gamma _ { d }$ under this overlap measure is below one. If $\nu _ { d - 1 }$ is zero, the preceding observations already separate the classes and the risk is zero. Otherwise, the assumption is additional to pointwise separation: values below one at every location may still approach one arbitrarily closely.

Applying this bound in Eq. (A5) gives

$$
\rho _ { d } \leq \gamma _ { \operatorname* { m a x } , d } \rho _ { d - 1 } .\tag{A6}
$$

Iteration and the balanced-prior risk bound establish

$$
R _ { 2 d } ^ { * } \leq \frac { 1 } { 2 } \rho _ { d } \leq \frac { 1 } { 2 } \rho _ { 1 } \prod _ { k = 2 } ^ { d } \gamma _ { \operatorname* { m a x } , k } ,\tag{A7}
$$

proving Eq. (10). A common $q \in ( 0 , 1 )$ with $\gamma _ { \mathrm { m a x } , k } \leq q$ gives geometric decay of this upper bound. Individual factors below one do not by themselves ensure a common geometric rate, and contraction of the upper bound does not establish strict reduction of the exact risk.

## 3. Gaussian Geometry and Physical Interpretation

For nonsingular Gaussian laws, the Bhattacharyya distance is [25]

$$
\begin{array} { l } { { \displaystyle { D _ { B } = \frac { 1 } { 8 } ( \mu _ { 1 } - \mu _ { 0 } ) ^ { T } \Sigma ^ { - 1 } ( \mu _ { 1 } - \mu _ { 0 } ) } } } \\ { { \displaystyle ~ + \frac { 1 } { 2 } \log \frac { \operatorname* { d e t } \Sigma } { \sqrt { \operatorname* { d e t } \Sigma _ { 0 } \operatorname* { d e t } \Sigma _ { 1 } } } , ~ \Sigma = \frac { \Sigma _ { 0 } + \Sigma _ { 1 } } { 2 } . } } \end{array}\tag{A8}
$$

Their coeficient is $\exp ( - D _ { B } )$ . Both terms are nonnegative: the first is a positive-definite quadratic form, and the second follows from concavity of the log determinant. Hence $D _ { B } = 0$ exactly when both means and both covariance matrices coincide. Separation may arise through means, covariances, or both.

If each class-conditional joint law of $( \mathbf { u } , \mathbf { z } )$ is Gaussian, its conditional law of z given u is Gaussian [26, Sec. 2.3.1]. Equation (A8), evaluated using the conditional means and covariances, then gives $\gamma _ { d } ( \mathbf { u } ) = \exp [ - D _ { B , d } ( \mathbf { u } ) ]$ . A uniform lower bound $D _ { B , d } ( \mathbf { u } ) \geq \delta _ { d } > 0$ on the overlap support is suficient for contraction with $\gamma _ { \operatorname* { m a x } , d } = e ^ { - \delta _ { d } }$ Pointwise positivity alone is insuficient.

A Gaussian marginal does not imply a Gaussian conditional law. Moreover, conditioning a Gaussian mixture generally produces another mixture, so the single-Gaussian formula cannot be applied directly to it. The density-based proof above applies to such mixtures without this restriction.

The requirement that readout vectors share one fixed joint distribution is essential when interpreting hardware results. Retaining an auxiliary channel from the same acquisition adds information without changing the marginal target distribution. Executing a larger encoding circuit can instead change that distribution through additional gates, initialization, and measurement activity. Comparisons across physical encoding distances must therefore assess these implementation costs separately from the theorem.

Claim A.1. Let µ be the input probability distribution on $\mathbb { R } ^ { 2 d }$ , and let $\mathcal { H } _ { 2 d }$ be a family of continuous network scores dense in $L ^ { 1 } ( \mu )$ . For any $\epsilon > 0$ , there is an $h _ { \theta } \in \mathcal { H } _ { 2 d }$ whose thresholded classifier $g _ { \theta } ( \mathbf { x } ) = \mathbf { 1 } \{ h _ { \theta } ( \mathbf { x } ) \geq 1 / 2 \}$ has risk $R ( g _ { \theta } ) = P ( g _ { \theta } ( X ) \neq Y ) \leq R _ { 2 d } ^ { * } + \epsilon$

Proof. Choose a measurable Bayes decision $g ^ { * } ( \mathbf { x } ) \ =$ ${ \bf 1 } \{ \pi _ { 1 } f _ { 1 } ( { \bf x } ) \ge \pi _ { 0 } f _ { 0 } ( { \bf x } ) \}$ . The input distribution is a finite measure because its total mass is one. A standard suficient approximation family uses arbitrarily many hidden units with a bounded, continuous, nonconstant activation, such as a sigmoid [27, Theorems 1, 2]. By $L ^ { 1 } ( \mu )$ density, choose $h _ { \theta }$ with $\begin{array} { r } { \dot { \int } | h _ { \theta } - g ^ { * } | d \mu < \epsilon / 2 } \end{array}$ . Whenever the thresholded decisions difer, $| h _ { \theta } - g ^ { * } | \ge 1 / 2$ . Hence

$$
\begin{array} { c } { { R ( g _ { \theta } ) - R _ { 2 d } ^ { * } \le P ( g _ { \theta } ( X ) \ne g ^ { * } ( X ) ) } } \\ { { \le 2 \displaystyle \int | h _ { \theta } - g ^ { * } | d \mu < \epsilon . } } \end{array}
$$

This is an existence result for an expressive family, not a finite-training guarantee. □

## Appendix B: Experimental Setup and Device Characteristics

## 1. Devices, Circuits, and Reference Labels

The experiments use ibm fez, ibm kingston, and ibm marrakesh, with the device coverage specified in each figure. Preparation labels identify reference circuits. They are not independent verification of each shot’s instantaneous physical state. Static SPAM references prepare $0 / 1$ for single-channel readout or 00/11 for joint I–Q readout. Contextual references include preceding workload operations and are matched to the measurement cycle.

QR-Stabilizer (Fig. 11(a)) uses data qubit 51, phasekickback ancilla 52, and, for Lifting, proxy 50. The routed reference variant also uses proxy 58. Five cycles apply Hadamard (H), controlled-NOT (CX, equivalent to CNOT), and H gates to the data qubit and measure it, with a reset before cycles two through five. Reference preparation selects the ideal all-zero or all-one trace. The target trace is 11111. Each reference circuit is sampled 20,000 times, and each target circuit 10,000 times. The SPAM scatter plot displays a sampled subset of 10,000 shots per reference class.

QEC-5 is a parity-check workload with $N = 5$ rounds. It prepares a Bell pair on data qubits 49 and 50 and measures their $Z Z$ parity through syndrome qubit 51. Lifting adds proxy 52. The syndrome qubit and proxy are reset for each round. Even- and odd-parity references train the contextual decoders. Figure 11(b) shows the conditional correction: after measuring parity bit $c _ { j } ,$ an $X$ correction is applied to the first data qubit if $c _ { j } = 1$ . Here $E _ { j }$ denotes a possible bit flip on that qubit. A single parity check detects odd parity but cannot identify which of two data qubits flipped. The displayed correction therefore assumes the designated error location used in this workload.

The hardware executions defer this correction and retain the uncorrected syndrome history. The target inserts an X error before the fourth round, giving ideal syndrome trace 00011. To describe the corresponding Pauli frame, let $s _ { j }$ be a decoded syndrome, set $s _ { 0 } = 0$ , and define the detection event $e _ { j } = s _ { j } \oplus s _ { j - 1 }$ . The accumulated frame bit is

$$
f _ { N } = \bigoplus _ { j = 1 } ^ { N } e _ { j } = s _ { N } , \qquad F _ { N } = X _ { 4 9 } ^ { f _ { N } } .\tag{B1}
$$

Thus the inferred correction can be recorded classically instead of applying a conditional gate after each round. In a terminal computational-basis measurement, applying this frame would replace the raw first-data-qubit bit $b _ { 4 9 }$ by $b _ { 4 9 } \oplus f _ { N }$ The plotted software-frame criterion checks the inferred frame bit against its ideal value. Because the sequence of exclusive-OR (XOR) operations telescopes, this is the final-syndrome classification metric in Fig. 10(b). It does not evaluate a physically corrected Bell state. With active correction, the ideal syndrome would return to zero after the correction, whereas the uncorrected hardware trace remains odd after the injected error.

The teleportation-related circuit uses payload 50, Alice 51, Alice proxy 52, Bob 58, and Bob proxy 71. Figure 11(c) shows teleportation of an arbitrary input ψ . After Bell-pair preparation and the Bell measurement, outcomes $c _ { \mathrm { P } }$ and $c _ { \mathrm { A } }$ leave Bob in $X ^ { c _ { \mathrm { { A } } } } Z ^ { c _ { \mathrm { { P } } } } \left| \psi \right.$ up to a global phase. Classical feedforward applies X conditioned on $c _ { \mathrm { A } }$ and then $Z$ conditioned on $c _ { \mathrm { P } }$ , recovering ψ . Equivalently, the two bits specify a Pauli frame that can be propagated through subsequent Cliford operations and used to reinterpret final measurements. For a computational-basis readout, only the X component changes the measured bit: $b _ { \mathrm { o u t } } = b _ { \mathrm { B } } \oplus c _ { \mathrm { A } }$ . The $Z$ component afects phase and is needed for general quantumstate recovery, but does not change this measurement probability.

In the experiments, reference circuits prepare payload 0 or 1 . The target independently acquires the same circuit as the input- 1 reference. Alice–Bob entanglement and the Bell-measurement operations precede Bob’s delayed readout. There is no physical outcomeconditioned $X / Z$ correction. Instead, the neural decoder uses the joint Alice–Bob I–Q readout signals, together with the proxy channels specified in Appendix B 2, to infer the input bit directly. This learns the computationalbasis input classification associated with the frame relation rather than explicitly XORing separately thresholded bits. The payload measurement is acquired but is not a decoder feature, consistent with the phase frame being irrelevant to this basis-specific metric. The experiment therefore evaluates ofline input classification, not arbitrary-state teleportation fidelity.

QEC-5 and teleportation acquisitions use a delay of 2000 ns on Fez and 2480 ns on Marrakesh to model the measurement and classical-processing window. Each reference or target circuit has 10,000 shots, collected in five shufled blocks of 2000. The conditional gates in Fig. 11(b,c) specify the corrections accounted for by Pauli-frame tracking in the experiments. Lifting auxiliary wires are omitted from the schematics.

Figure 12 shows the physical subgraph used by these layouts. The native connections are 49–50, 50–51, 51–52, 51–58, and 58–71. In particular, qubits 49 and 51 are not directly coupled: their logical parity-check interaction requires routing or gate decomposition through the hardware connectivity. The graph distinguishes native couplings from the logical gates in the circuit schematics.

![](images/0ec7cd83e3496e3cf5ef449e96617e0699541f8a39ae137258695085015f2238.jpg)  
(a) QR-Stabilizer.

![](images/ab6c8067e55297062880ebe2d1fddfcaa18ac787a1139cef646778b7a9d058e2.jpg)  
(b) QEC-5.

![](images/1fdde0053605f9c338c311b7ba093f2389e6e65701a3d42a56568dd3e9ee3ce9.jpg)  
(c) Teleportation.

FIG. 11. MCM workloads. (a) Repeated stabilizer measurement with reset and reuse. (b) Repeated parity extraction with an outcome-conditioned X correction. (c) Teleportation with outcome-conditioned X and Z corrections. Double lines carry classical measurement outcomes. Lifting auxiliary wires are omitted. The experiments account for the corrections in (b,c) through the Pauli-frame interpretation described in the text.  
![](images/7307c349d773c90abbbe6a3cb5b56d6bd5ce604ead04b26d0e115f446ae6ad08.jpg)  
FIG. 12. Physical connectivity used by the MCM workloads on ibm fez, ibm kingston, and ibm marrakesh. Nodes identify physical qubits, and edges denote native couplings. The workload-specific qubit assignments are given in the text.

## 2. Decoders and Evaluation Populations

All three MCM workloads—QR-Stabilizer, QEC-5, and teleportation—use the same decoder architecture and training settings. The PyTorch decoder has one hidden layer of 32 neurons with rectified linear unit (ReLU) activations. Single-channel models use one binary head, and Lifting uses two heads supervised with labels 00/11. Reported predictions use the data-qubit head. Training uses Adam with learning rate 0.003, batches of 256, 50 epochs, and an 80/20 shufled reference split.

For QR-Stabilizer, features are scaled by a power of two derived from their maximum absolute value. Pooled references set the training scale, while each inference batch computes its own scale. Figure 9 compares targetshot accuracy for NN-based QSD with held-out calibration accuracy for Lifting with in situ calibration. The two populations are distinct.

For QEC-5 and teleportation, features are standardized using a scaler fitted on pooled references. For QEC-

5, NN-based QSD uses the syndrome channel of the en coded target circuit and discards its proxy features, with either static SPAM or in situ calibration. Lifting uses joint syndrome–proxy references acquired under the corresponding calibration condition. For teleportation, NN based QSD uses the same four Alice–Bob I–Q features under either calibration condition. Both Lifting variants use in situ calibration. Adding Alice’s proxy gives six inputs, while including both Alice’s and Bob’s proxies gives eight. The two unfiltered Lifting bars therefore compare diferent input channels rather than an isolated calibration change. In the teleportation panel, the compact legend labels “Lift” and “Lift (in situ)” identify the Alice-proxy and both-proxy variants, respectively.

In Fig. 10(a), NN-based QSD and Lifting use singlechannel and joint I–Q references, respectively. Each is evaluated with static SPAM calibration and with cycle-matched in situ calibration. Circuit success in Fig. 10 is defined separately for each workload. QR-Stabilizer whole-trace accuracy is the fraction of target shots for which all five predictions equal the ideal trace. QEC-5 accuracy is final-syndrome correctness. The reported software-frame XOR reduces to that final bit. Teleportation-related accuracy is the fraction of input-1 target shots classified as 1. QR-Stabilizer filtering fits Gaussian reference distributions with empirical priors and covariance regularizer $1 0 ^ { - 6 } I .$ , and requires posterior confidence at least 0.75 in every cycle. Its coverage is data dependent. The other two workloads rank shots by $| P ( Y = 1 \mid \mathbf { x } ) - 1 / 2 |$ and retain the most confident 90%, with ties potentially increasing coverage. Filtered accuracies use retained shots as their denominator.

## 3. Distributional Estimates and Comparator Definitions

The Bayes-risk curves are equal-prior plug-in estimates from Gaussian kernel densities fitted to normalized I– Q samples. In Fig. 1(d), each target-cycle distribution is compared with the SPAM-0 reference, so this constructed discrimination task difers from a contemporaneous two-class workload acquisition. The digital-error curve transfers the measured accuracy ofset between NN-based QSD and digital readout from a separate Fez acquisition to the plotted target-shot NN-based QSD accuracy. Figure 5 instead compares target-only and joint KDE risk estimates. Its signed histograms come from quadratic discriminant scores.

The distance sweep compares native assignment accuracy, hard decoding of spatial repetition readouts, neural decoding of sequential temporal observations, and neural decoding of joint I–Q readout signals from spatial encoding. Spatial distance increases physical-qubit count, whereas temporal distance increases the number of mea surements on one qubit. The fitted accuracy curve uses $\kappa = 0 . 4 5$ and $\alpha = 9 . 0 3$ for the displayed Fez configurations. These fitted parameters describe that sweep rather than establish the conditional-overlap assumptions on hardware.

[1] F. Arute, K. Arya, R. Babbush, D. Bacon, J. C. Bardin, R. Barends, R. Biswas, S. Boixo, F. G. S. L. Brandao, D. A. Buell, B. Burkett, Y. Chen, Z. Chen, B. Chiaro, R. Collins, W. Courtney, A. Dunsworth, E. Farhi, B. Foxen, A. Fowler, C. Gidney, M. Giustina, R. Graf, K. Guerin, S. Habegger, M. P. Harrigan, M. J. Hartmann, A. Ho, M. Hofmann, T. Huang, T. S. Humble, S. V. Isakov, E. Jefrey, Z. Jiang, D. Kafri, K. Kechedzhi, J. Kelly, P. V. Klimov, S. Knysh, A. Korotkov, F. Kostritsa, D. Landhuis, M. Lindmark, E. Lucero, D. Lyakh, S. Mandr\`a, J. R. Mc-Clean, M. McEwen, A. Megrant, X. Mi, K. Michielsen, M. Mohseni, J. Mutus, O. Naaman, M. Neeley, C. Neill, M. Y. Niu, E. Ostby, A. Petukhov, J. C. Platt, C. Quintana, E. G. Riefel, P. Roushan, N. C. Rubin, D. Sank, K. J. Satzinger, V. Smelyanskiy, K. J. Sung, M. D. Trevithick, A. Vainsencher, B. Villalonga, T. White, Z. J. Yao, P. Yeh, A. Zalcman, H. Neven, and J. M. Martinis, Quantum supremacy using a programmable superconducting processor, Nature 574, 505 (2019).

[2] V. V. Sivak, A. Eickbusch, B. Royer, S. Singh, I. Tsioutsios, S. Ganjam, A. Miano, B. L. Brock, A. Z. Ding, L. Frunzio, S. M. Girvin, R. J. Schoelkopf, and M. H. Devoret, Real-time quantum error correction beyond breakeven, Nature 616, 50 (2023).

[3] A. D. C´orcoles, M. Takita, K. Inoue, S. Lekuch, Z. K. Minev, J. M. Chow, and J. M. Gambetta, Exploiting Dynamic Quantum Circuits in a Quantum Algorithm with Superconducting Qubits, Phys. Rev. Lett. 127, 100501 (2021).

[4] D. Hothem, J. Hines, C. Baldwin, D. Gresh, R. Blume-Kohout, and T. Proctor, Measuring error rates of midcircuit measurements, Nat. Commun. 16, 5761 (2025).

[5] A. Blais, A. L. Grimsmo, S. M. Girvin, and A. Wallraf, Circuit quantum electrodynamics, Rev. Mod. Phys. 93, 025005 (2021).

[6] B. Lienhard, A. Veps¨al¨ainen, L. C. Govia, C. R. Hoffer, J. Y. Qiu, D. Rist\`e, M. Ware, D. Kim, R. Winik, A. Melville, B. Niedzielski, J. Yoder, G. J. Ribeill, T. A. Ohki, H. K. Krovi, T. P. Orlando, S. Gustavsson, and W. D. Oliver, Deep-Neural-Network Discrimination of Multiplexed Superconducting-Qubit States, Phys. Rev. Appl. 17, 014024 (2022).

[7] S. Maurya, C. N. Mude, W. D. Oliver, B. Lienhard, and S. Tannu, Scaling Qubit Readout with Hardware Eficient Machine Learning Architectures, in Proc. 50th Annu. Int. Symp. Comput. Archit. (2023) pp. 1–13, arXiv:2212.03895 [quant-ph].

[8] N. R. Vora, Y. Xu, A. Hashim, N. Fruitwala, H. N. Nguyen, H. Liao, J. Balewski, A. Rajagopala, K. Nowrouzi, Q. Ji, K. B. Whaley, I. Siddiqi, P. Nguyen, and G. Huang, ML-Powered FPGA-based Real-Time Quantum State Discrimination Enabling Mid-circuit Measurements (2024), arXiv:2406.18807 [quant-ph].

[9] L. Devroye, L. Gy¨orfi, and G. Lugosi, A Probabilistic Theory of Pattern Recognition (Springer, New York, 1996).

[10] T. Proctor, M. Revelle, E. Nielsen, K. Rudinger, D. Lobser, P. Maunz, R. Blume-Kohout, and K. Young, Detecting and tracking drift in quantum information processors, Nat. Commun. 11, 5396 (2020).

[11] K. Korhonen, H. Vappula, A. Glos, M. Cattaneo, Z. Zimbor´as, E.-M. Borrelli, M. A. C. Rossi, G. Garc´ıa-P´erez, and D. Cavalcanti, Practical techniques for high-precision measurements on near-term quantum hardware and applications in molecular energy estimation, Npj Quantum Inf. 11, 110 (2025).

[12] D. T. McClure, H. Paik, L. S. Bishop, M. Stefen, J. M. Chow, and J. M. Gambetta, Rapid Driven Reset of a Qubit Readout Resonator, Phys. Rev. Appl. 5, 011001 (2016).

[13] R.-Z. Zhao, Z.-A. Zhao, T.-L. Wang, P. Wang, S. Zhang, X.-Y. Yang, H.-F. Zhang, Z.-F. Li, Y. Wu, S.-R. Liu, P. Duan, and G.-P. Guo, Single-Step Phase-Engineered Pulse for Active Readout Cavity Reset in Superconducting Circuits (2026), arXiv:2512.08393 [quant-ph].

[14] J. P. Santos and R. Uzdin, Drift-resilient mid-circuit measurement and state preparation error mitigation for dynamic circuits (2025), arXiv:2506.11270 [quant-ph].

[15] M. DeCross, E. Chertkov, M. Kohagen, and M. Foss-Feig, Qubit-Reuse Compilation with Mid-Circuit Measurement and Reset, Phys. Rev. X 13, 041057 (2023).

[16] S. Brandhofer, I. Polian, and K. Krsulich, Optimal Qubit Reuse for Near-Term Quantum Computers, in 2023 IEEE Int. Conf. Quantum Comput. Eng. QCE, Vol. 01 (2023) pp. 859–869.

[17] H. Kim, E. Jang, S. Choi, Y. Kim, and W. W. Ro, QR-Map: A Map-Based Approach to Quantum Circuit Abstraction for Qubit Reuse Optimization, in Proc. 52nd Annu. Int. Symp. Comput. Archit., ISCA ’25 (Association for Computing Machinery, New York, NY, USA, 2025) pp. 1568–1582.

[18] M. Zhong, Z. Zhang, X. Ren, C. Zhu, S. Niu, and Z. Liang, A Compilation Framework for Quantum Circuits with Mid-Circuit Measurement Error Awareness (2025), arXiv:2511.10921 [quant-ph].

[19] X. Fang, K. Yin, Y. Zhu, J. Ruan, D. Tullsen, Z. Liang, A. Sornborger, A. Li, T. Humble, Y. Ding, and Y. Shi, CaliQEC: In-situ Qubit Calibration for Surface Code Quantum Error Correction, in Proc. 52nd Annu. Int. Symp. Comput. Archit. (ACM, Tokyo Japan, 2025) pp. 1402–1416.

[20] R. Hicks, B. Kobrin, C. W. Bauer, and B. Nachman, Active readout-error mitigation, Phys. Rev. A 105, 012419 (2022).

[21] J. M. G¨unther, F. Tacchino, J. R. Wootton, I. Tavernelli, and P. K. Barkoutsos, Improving readout in quantum simulations with repetition codes, Quantum Sci. Technol. 7, 015009 (2021).

[22] E. van den Berg, S. Bravyi, J. M. Gambetta, P. Jurcevic, D. Maslov, and K. Temme, Single-shot error mitigation by coherent Pauli checks, Phys. Rev. Res. 5, 033193

(2023).

[23] M. D. Hanisch, B. Het´enyi, and J. R. Wootton, Soft information decoding with superconducting qubits, APS Open Sci. 1, 000019 (2026).

[24] I. Sohn, C. Lee, W. Song, K. Bae, and W. Lee, Quantum error mitigation via structural encoding with classical error correction codes, EPJ Quantum Technol. 13, 25 (2026).

[25] K. Fukunaga, Introduction to Statistical Pattern Recognition, 2nd ed., Computer Science and Scientific Computing (Academic Press, Boston, 1990).

[26] C. M. Bishop, Pattern Recognition and Machine Learning, Information Science and Statistics (Springer, New York, 2006).

[27] K. Hornik, Approximation capabilities of multilayer feed forward networks, Neural Netw. 4, 251 (1991).