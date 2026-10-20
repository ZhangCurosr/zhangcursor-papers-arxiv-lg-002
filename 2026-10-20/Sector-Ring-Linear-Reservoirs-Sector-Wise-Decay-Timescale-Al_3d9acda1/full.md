This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.

# Sector-Ring Linear Reservoirs: Sector-Wise Decay-Timescale Allocation With State Reuse Across Candidate Settings

YUDAI EBATO<sup>1</sup>, SOU NOBUKAWA<sup>2,3,4</sup>, YUSUKE SAKEMI<sup>3,5</sup>, TAKASHI KANAMARU<sup>6</sup>, NINA SVIRIDOVA<sup>5,7</sup>, AND KAZUYUKI AIHARA<sup>3,5</sup>

<sup>1</sup>Graduate School of Information and Computer Science, Chiba Institute of Technology, Narashino, Chiba 275-0016, Japan <sup>2</sup>Department of Computer Science, Chiba Institute of Technology, Narashino, Chiba 275-0016, Japan

<sup>3</sup>Research Center for Mathematical Engineering, Chiba Institute of Technology, Narashino, Chiba 275-0016, Japan <sup>4</sup>Department of Preventive Intervention for Psychiatric Disorders, National Institute of Mental Health, National Center of Neurology and Psychiatry, Kodaira, Tokyo 187-8551, Japan

<sup>5</sup>International Research Center for Neurointelligence (WPI-IRCN), The University of Tokyo, Bunkyo-ku, Tokyo 113-0033, Japan

<sup>6</sup>Department of Mechanical Science and Engineering, School of Advanced Engineering, Kogakuin University, Hachioji, Tokyo 192-0015, Japan

<sup>7</sup>Department of Intelligent Systems, Tokyo City University, Setagaya-ku, Tokyo 158-8557, Japan

Corresponding author: Yudai Ebato (e-mail: yudaiebato@gmail.com).

This work was supported in part by a project, JPNP14004, commissioned by the New Energy and Industrial Technology Development Organization (NEDO); in part by JSPS KAKENHI Grant-in-Aid for Scientific Research (B) (Grant Numbers JP25K03198, SN; JP25K00148, YS); in part by Grant-in-Aid for Transformative Research Areas (A) (Grant Number JP25H02626, SN); and in part by Grant-in-Aid for JSPS Fellows (Grant Number JP25KJ2052, YE).

ABSTRACT In reservoir computing, performance depends on whether the fixed dynamics of the reservoir suit the task, and the appropriate memory length and frequency response differ across tasks. The reservoir settings are found by a search on each task, and driving the reservoir with the input and collecting its states for every candidate setting dominates the cost of the search. In a reservoir with a linear recurrence, the dynamics decompose into modes, one per eigenvalue, and the eigenvalue distribution becomes a design variable. We propose the Sector-Ring Linear Reservoir (SRLR), in which the eigenvalue angles are fixed at evenly spaced positions around a circle and one decay timescale, chosen from a few values, is assigned to each sector of consecutive angles. To keep the scale of each mode’s state equal across candidates, the input weights are scaled using the power spectrum of the training input. To reduce the cost of the search, the states of the modes for each decay-timescale value are computed once and reused across candidates. On prediction and reconstruction benchmarks built from chaotic systems, the SRLR achieved, in all eight cases, a lower mean test error than a reference model in which all modes share one decay timescale, and was competitive with nonlinear echo state network and leaky-integrator echo state network baselines. The selected decay-timescale profiles changed with the prediction target and horizon. The state reuse reduced the number of computed mode trajectories in the search by 99.8% relative to per-candidate recomputation.

INDEX TERMS Echo state network, linear reservoir, reservoir computing, sector-ring linear reservoir.

## I. INTRODUCTION

Reservoir computing (RC) is a framework for time-series processing that uses a fixed, typically nonlinear, recurrent dynamical system called the reservoir [1]–[3]. The reservoir expands the input history into a high-dimensional state trajectory, and only a readout from that trajectory is trained, usually by linear regression. Echo State Networks (ESNs) and Liquid State Machines are representative examples [1], [2]. The framework has been applied effectively to time-series prediction, signal processing, and the prediction and reconstruction of dynamical systems, including chaotic ones [4], [5]. Training the readout is a linear least-squares problem and is faster and cheaper than gradient-based training of recurrent networks, and this training efficiency is one of the main advantages cited for RC [3], [6], [7]. On the other hand, because the recurrent dynamics are fixed, all the information the readout can use must already be present in the reservoir state. Performance therefore depends on whether the dynamics of the reservoir suit the task.

Useful reservoir dynamics are task-dependent. Two of the main factors that affect performance are the timescales over which the reservoir retains past input (its memory) and the frequency components of the input to which it responds (its frequency response). The amount of memory required differs with the prediction target [8], and the timescales and frequency components to which the reservoir should respond have likewise been adjusted for each task [9]–[11].

The hyperparameter settings that realize such dynamics are found by search on the task, and this search accounts for most of the computational cost of applying RC. Practical guides treat the main hyperparameters (the spectral radius, the leaking rate, and the input scaling) as quantities to be tuned together for the task [7], [9], and applying RC to a task means evaluating many candidate settings on that task. In training an ESN, driving the reservoir with the input and collecting its states is the stage that dominates the cost, and the search repeats this stage for every candidate [12]. The reservoir must be driven anew for each candidate because the recurrence is nonlinear and the effect of a setting appears only once the reservoir has been driven. This motivates a design variable that relates more directly to the recurrent dynamics and makes each candidate cheaper to evaluate.

One way to address this problem is to make the recurrence linear, that is, to use a linear reservoir. Linear systems theory then applies to the recurrence, and both the interpretation of the dynamics and the computation of the states become simpler. The state trajectory of a linear recurrence can be analyzed as a superposition of modes, components that evolve independently of one another, each corresponding to an eigenvalue [13], [14]. The radius (magnitude) of a stable discretetime eigenvalue determines the decay timescale of its mode, that is, how long the mode retains past input, and the angle determines the oscillatory, frequency-like component of its response [13], [15]. On the computational side, the recurrences of the individual modes can be advanced in parallel across modes or by parallel scan algorithms over time [16], [17], as recent diagonal and parallel formulations of linear reservoirs demonstrate [18], [19]. The eigenvalue distribution thus becomes an explicit design variable that acts on each mode separately. Using this variable to search and compare candidate settings, however, requires further conditions.

Making the eigenvalue distribution a practical design variable for a reservoir requires three conditions. The first is that the eigenvalue distribution can be specified by a few parameters. If each eigenvalue is chosen freely [14], [20], the search space is too large to search at low computational cost. The second is control of the scale, that is, the variance, of the state of each mode. Even under the same input, different eigenvalue settings give the state of each mode a different scale, and unless this variation is controlled, the differences among candidates include a factor other than the eigenvalue distribution. The third is that each candidate can be evaluated at low computational cost. The eigenvalue distribution adds parameters to tune on top of the existing hyperparameters; consequently, the number of candidates, and with it the number of state trajectories to compute, grows sharply. What is needed, therefore, is a design that specifies the eigenvalue distribution with a few parameters, keeps the scale of each mode’s state equal across candidates, and can be searched at low computational cost.

In this paper, we propose the Sector-Ring Linear Reservoir (SRLR), a design that meets these conditions. The SRLR is a reservoir whose recurrent update is linear and in which nonlinearity is introduced by a fixed feature map before the readout. The angles of the modes’ eigenvalues are fixed at evenly spaced positions around a circle, and one decay timescale is assigned to each sector of consecutive angles; a candidate is therefore specified by one value per sector, that is, by a few parameters. The input weight of each mode is scaled using the power spectrum of the training input so that the scale of each mode’s state is equal across candidates. Moreover, because each per-sector value is chosen from a small set and the trajectory of each mode depends only on the value assigned to its sector, the trajectory of each mode can be computed once per value and reused across candidates. Our preliminary conference paper studied eigenvalue control in a linear reservoir (the eigenvalue-controlled ESN) using random eigenvalue distributions constrained to angular sectors [21]. The present paper develops that approach into the formulation just described: fixed angles, one decay timescale shared within each sector, scaling of the input weights using the power spectrum, and reuse of each mode’s trajectory across candidates for each timescale value.

We evaluate the SRLR in two comparisons on prediction and reconstruction benchmark tasks built from chaotic dynamical systems. First, we test whether allowing the decay timescale to differ across sectors improves accuracy, by comparing the SRLR with the case in which all modes share one decay timescale, that is, the SRLR with a single sector. We call this case the common-radius reference. Second, we compare the SRLR with the ESN and leaky-integrator ESN (LI-ESN) baselines. We also report the selected decay-timescale profiles (the per-sector values chosen by the search), the scale of each mode’s state, and the computational cost of the candidate search, and we compare the input-weight scaling with two alternative input-weight designs. The remainder of the paper is organized as follows. Section II reviews related work, Section III presents the models, tasks, and experimental protocol, Section IV reports the two comparisons and the additional analyses, Section V interprets the selected profiles and discusses the scope of the evidence, and Section VI concludes the paper.

## II. RELATED WORK

## A. TASK-DEPENDENT RESERVOIR DYNAMICS

The amount of memory that a task requires is specific to that task. How long the memory of a reservoir lasts affects how well it solves a particular problem, and too much or too little memory can degrade accuracy [8]. Even among tasks that use the same input signal, the required memory changes with the prediction target and with the prediction horizon, that is, the number of steps ahead at which the target is predicted [22]. Moreover, the total ability of a reservoir to retain past inputs and to transform them nonlinearly, its information-processing capacity, is finite, and because memory and nonlinear transformation share it, memory cannot be increased without reducing nonlinear transformation [23]–[25].

The timescales and the frequency response of a reservoir have also been adjusted to the task. Leaky integration sets a single timescale of the state update for the whole reservoir, and band-pass filters or filter neurons set the frequency response of each unit [9], [10], [26]. There are also designs that give different timescales to individual neurons, to groups of neurons, or to sub-reservoirs [11], [27], [28], designs that set the decay timescale of each mode independently [29], and designs that split the input into bands fed to separate reservoirs [30].

The dynamics of a reservoir are determined not only by its settings but also by the driving input. Even the echo state property, that is, the property that a driven reservoir forgets its initial condition, depends on the driving input [31], [32]. Measures computed with generic test inputs, such as white noise, do not necessarily describe performance under the task input; indeed, the total information-processing capacity correlates only weakly with task performance [33]. Moreover, in nonlinear reservoirs the measured memory capacity, which quantifies how well past inputs can be reconstructed from the state, changes with the input amplitude alone [34], [35]. In view of this input dependence, memory has also been measured with the actual input signal; for example, the delay capacity quantifies memory using the autocorrelation of the states of the reservoir driven with the actual input [8], [36]. Selecting the settings that suit a task therefore requires driving the reservoir with the input of that task and evaluating it for each candidate setting.

## B. LINEAR RESERVOIRS

Cycle and ring reservoirs specify the eigenvalue distribution and the decay timescale with a few parameters, namely evenly spaced angles and one shared spectral radius, and such reservoirs have been shown to achieve accuracy comparable to that of randomly generated ESNs. The Simple Cycle Reservoir (SCR) couples nonlinear units in a ring through one shared weight, and because it uses no random weights it can be constructed deterministically. On benchmarks it achieved accuracy comparable to that of randomly generated ESNs [37]. The single shared weight gives every eigenvalue the same magnitude, equal to the spectral radius; in the linear SCR all modes therefore share one decay timescale [15], [38]. The SRLR retains these evenly spaced angles and the small number of parameters, and its common-radius reference, like a linear SCR, has all eigenvalues at one magnitude, the spectral radius. The SRLR differs from these ring designs and from its own common-radius reference in that it varies the decay timescale across sectors.

A reservoir whose recurrence is linear and diagonalizable can be formulated in diagonal form, in which the recurrent matrix is diagonal and the eigenvalues of the individual modes are parameterized directly. In the diagonal form each mode evolves independently, one per eigenvalue, and the state trajectory is their superposition. This form has been used to analyze reservoirs by decomposition into modes [14] and, recently, as the architecture of linear reservoirs and of reservoirs whose diagonal recurrence has complex-valued entries (complex-diagonal ESNs) [18], [19]. In the complex-diagonal formulations the modes are parameterized independently of one another, and the decay timescale and the angle of each mode can be chosen separately.

A linear recurrence also has the advantage that its computation over time can be parallelized. Because each update in a linear recurrence is an affine transformation, a matrix product plus an input term, consecutive updates can be combined into one affine transformation and computed in parallel over time. A parallel scan algorithm called an associative scan uses this composition, and the number of sequential steps then scales logarithmically with the sequence length T [16], [17]. In the diagonal form the matrix products reduce to elementwise products, which makes the composition computationally inexpensive, and this parallelization has yielded speedups for linear recurrent networks and has been applied to complexdiagonal ESNs [17], [19].

The combination of a linear reservoir with a nonlinear readout, that is, a fixed nonlinear map applied to the state before the trained linear readout, has been adopted in several lines of research and is supported both theoretically and empirically. Theoretically, linear recurrent dynamics followed by sufficiently expressive nonlinear maps are universal approximators for broad classes of sequence-to-sequence maps [39]–[42]. Empirically, competitive accuracy in this form has been reported for next-generation reservoir computing, which reads out nonlinear functions of time-delayed inputs [43], for nonlinear readouts in linear reservoir networks in a benchmark study [44], and for complex-diagonal ESNs [19].

The choice of eigenvalues also changes the scale of each mode’s state. Per-mode normalizations have therefore been proposed that scale the input weight of each mode so that the variance of each mode’s state is equal across modes, as in the normalization parameter γ of the Linear Recurrent Unit (LRU) [45] and the per-mode scaling of complex-diagonal ESNs [19]. Because these assume a white-noise input, we scale the input weights using the power spectrum of the training input so that different eigenvalue distributions can be compared with the state variance of each mode held equal across candidates.

## III. METHODS

This section describes the reservoir models and the common framework in which they are evaluated (Section III-A), and the benchmark tasks, the evaluation protocol, and the candidate-evaluation computation of the SRLR (Section III-B). The training, validation, and test windows and the mode seeds referred to in Section III-A are defined by the evaluation protocol of Section III-B2; a model seed is the random seed that fixes the random elements of a model, namely the weights of the ESN and the LI-ESN (Section III-A2), the mixing matrix of the SRLR (Section III-A3d), and the input weights of the uniform random input-weight design of the SRLR (Section III-A3c).

We conduct two comparisons. The first tests whether allowing the decay timescale to differ across sectors improves accuracy, by comparing the SRLR with its common-radius reference. The second tests how the SRLR as a whole, with its linear recurrence, compares with the nonlinear ESN and LI-ESN baselines. In addition, we compare the inputweight scaling of the SRLR with two alternative input-weight designs (Section III-A3c), each under its own independent search.

## A. RESERVOIR-MODEL DESIGN AND EVALUATION FRAMEWORK

## 1) Common Reservoir-Evaluation Framework

This subsection describes the common framework in which all reservoir models are compared. The models compared are the ESN, the LI-ESN, the Sector-Ring Linear Reservoir (SRLR), and the common-radius reference, which is the SRLR with a single sector (Section III-A3b). Each model updates its reservoir state $\mathbf { h } _ { t } \in \mathbb { R } ^ { N }$ from the preprocessed input $\widetilde { u } _ { t }$ (Section III-B2) by its own recurrence and forms a feature vector $\boldsymbol { \chi } _ { t } ~ \in ~ \mathbb { R } ^ { N + 1 }$ by a fixed feature map, where N is the number of reservoir units. In every model the feature vector $x _ { t }$ consists of N entries formed from the state, followed by a constant entry of 1 (the bias entry); its specific form is given in the subsection for each model. The feature vector is mapped to predictions by a readout $\hat { \mathbf { y } } _ { t } = \mathbf { W } _ { \mathrm { o u t } } ^ { \top } \boldsymbol { x } _ { t }$ , trained for each model by the same ridge regression, and prediction quality is scored by the normalized root-mean-square error (NRMSE) under a protocol shared by all models (Section III-B).

Only the readout weights are trained, by ridge regression. For each model, the feature vectors over the training window (of length $T _ { \mathrm { t r } }$ samples, Section III-B2) are stacked as rows into the feature matrix $\Phi _ { \mathrm { t r } } ~ \in ~ \mathbb { R } ^ { T _ { \mathrm { t r } } \times ( N + 1 ) }$ , and the corresponding targets into $\mathbf { Y } _ { \mathrm { t r } } \in \mathbb { R } ^ { T _ { \mathrm { t r } } \times d _ { \mathrm { y } } }$ , where $d _ { y }$ is the target dimension $( d _ { y } = 1$ for all tasks in this paper, Section III-B1). The readout weights $\mathbf { W _ { \mathrm { o u t } } } \in \mathbb { R } ^ { ( N + 1 ) \times \bar { d } _ { \mathrm { y } } }$ are then obtained in closed form as

$$
\mathbf { W } _ { \mathrm { o u t } } = ( \boldsymbol { \Phi } _ { \mathrm { t r } } ^ { \mathsf { T } } \boldsymbol { \Phi } _ { \mathrm { t r } } + \beta \mathbf { I } ) ^ { - 1 } \boldsymbol { \Phi } _ { \mathrm { t r } } ^ { \mathsf { T } } \mathbf { Y } _ { \mathrm { t r } } ,\tag{1}
$$

where $\beta > 0$ is the ridge regularization parameter and I is the identity matrix. Because the last entry of ${ \boldsymbol { x } } _ { t }$ is the constant 1, the last column of $\Phi _ { \mathrm { t r } }$ is a column of ones, and the last row of $\mathbf { W _ { \mathrm { o u t } } }$ is the bias of the readout; this bias is regularized by the same $\beta$ in (1), and no separate unregularized intercept is fitted. The parameter $\beta$ is one of the model-specific settings selected on the validation window (Section III-B2).

## 2) ESN and LI-ESN Baselines

The first baseline model is the ESN [1]. The ESN updates the state $\mathbf { h } _ { t } \in \mathbb { R } ^ { N }$ of its N reservoir units from the preprocessed scalar input $\widetilde { u } _ { t }$ as

$$
\mathbf h _ { t } = { \mathrm { t a n h } } ( \mathbf W _ { \mathrm { r e c } } \mathbf h _ { t - 1 } + \mathbf w _ { \mathrm { i n } } { \widetilde u } _ { t } ) , \qquad \mathbf h _ { 0 } = \mathbf 0 ,\tag{2}
$$

where tanh is applied element-wise, $\mathbf { w } _ { \mathrm { i n } } \in \mathbb { R } ^ { N }$ is the input weight vector, and $\mathbf { W } _ { \mathrm { r e c } } \in \mathbb { R } ^ { N \times N }$ is the recurrent weight matrix. For each model seed (Section III-B2), each input weight $w _ { \mathrm { i n } , i }$ is drawn uniformly from $\left( - s _ { \mathrm { i n } } , s _ { \mathrm { i n } } \right)$ , where $s _ { \mathrm { i n } }$ is the input scaling. The recurrent weight matrix $\mathbf { W } _ { \mathrm { r e c } }$ is a sparse random matrix constructed as follows. A matrix Wf with entries $\widetilde { W } _ { i j } = \xi _ { i j } Z _ { i j }$ is first formed by drawing $\xi _ { i j }$ from Bernoulli(0.1) and $Z _ { i j }$ uniformly from $( - 1 , 1 )$ . The matrix $\widetilde { \mathbf { W } }$ is then divided by its spectral radius $\rho ( \mathbf { \bar { W } } )$ , the largest absolute eigenvalue, and multiplied by $r ,$ giving $\mathbf { W } _ { \mathrm { r e c } } = ( r / \rho ( \widetilde { \mathbf { W } } ) ) \widetilde { \mathbf { W } }$ , whose spectral radius is therefore r. For each model seed, a single random stream draws the input weights first and then the entries $\xi _ { i j }$ and $Z _ { i j }$ of $\widetilde { \mathbf { W } } ;$ no per-row rescaling of the recurrent weights is applied, so that the spectral-radius scaling above is the only scaling of $\mathbf { W } _ { \mathrm { r e c } }$ . The state update (2) has no bias term and no output feedback [7], that is, the readout output is not fed back into the state. The feature vector is the state with the bias entry appended, $\chi _ { t } = [ \mathbf { h } _ { t } ^ { \mathsf { T } } , 1 ] ^ { \mathsf { T } } \in \mathbb { R } ^ { N + 1 }$ . The hyperparameters selected on the validation window are the input scaling $s _ { \mathrm { i n } }$ , the spectral radius r, and the ridge parameter $\beta .$

The second baseline model is the LI-ESN [7], [9]. It differs from the ESN in the state update,

$$
\begin{array} { c } { \mathbf { h } _ { t } = \left( 1 - \alpha _ { \ell } \right) \mathbf { h } _ { t - 1 } + \alpha _ { \ell } \operatorname { t a n h } ( \mathbf { W } _ { \mathrm { r e c } } \mathbf { h } _ { t - 1 } + \mathbf { w } _ { \mathrm { i n } } \widetilde u _ { t } ) , } \\ { 0 < \alpha _ { \ell } \leq 1 , } \end{array}\tag{3}
$$

where the leaking rate $\alpha _ { \ell }$ adjusts the timescale of the state update: the new state is a weighted average of the previous state and the tanh update. The LI-ESN uses the same weight construction and feature vector as the ESN (the spectral radius r is that of $\mathbf { W } _ { \mathrm { r e c } }$ , as in the ESN) and adds the leaking rate $\alpha _ { \ell }$ to the hyperparameters selected on the validation window.

## 3) Sector-Ring Linear Reservoir

## a: Rotation–Decay Block Representation

As shown in Fig. 1, the SRLR implements its recurrent matrix as a block-diagonal matrix of $2 \times 2$ rotation–decay blocks, so that the eigenvalue distribution of a real recurrent matrix can be designed directly. With $N = 2 M$ , block $j ( j = 1 , \dots , M )$ holds a state $\mathbf { x } _ { j , t } \in \mathbb { R } ^ { 2 }$ driven by the preprocessed scalar input $\widetilde { u } _ { t }$ through the input weights $\mathbf { w } _ { j } \in \mathbb { R } ^ { 2 }$ , and is updated as

$$
\begin{array} { r } { \begin{array} { c c c } { \mathbf { x } _ { j , t } = \mathbf { R } _ { j } \mathbf { x } _ { j , t - 1 } + \mathbf { w } _ { j } \widetilde { u } _ { t } , } & { \mathbf { x } _ { j , 0 } = \mathbf { 0 } , } \\ { \mathbf { R } _ { j } = r _ { j } \mathbf { R } ( \theta _ { j } ) , } & { \mathbf { R } ( \theta ) = \left[ \cos \theta \begin{array} { c c c } { - \sin \theta } \\ { \sin \theta } & { \cos \theta } \end{array} \right] , } \end{array} } \end{array}\tag{4}
$$

where $\mathbf { R } ( \theta )$ is the $2 \times 2$ rotation matrix. Each block realizes one complex-conjugate eigenvalue pair $r _ { j } e ^ { \pm i \theta _ { j } }$ , where $r _ { j }$ is the amplitude decay factor per update and $\theta _ { j }$ the phase increment per update. Below, the state $\mathbf { x } _ { j , t }$ of block j is called the state of modej (a mode in the sense of Section I, a component that evolves independently, and in this real form one mode carries one conjugate eigenvalue pair). Because the decay of each mode is specified directly by the radius $r _ { j } .$ , leaky integration as in the LI-ESN is not used. The reservoir state is the concatenation of the states of the modes, $\mathbf { h } _ { t } = [ \mathbf { x } _ { 1 , t } ^ { \mathsf { T } } , \ldots , \mathbf { x } _ { M , t } ^ { \mathsf { T } } ] ^ { \mathsf { T } } \in \mathbb { R } ^ { N }$ b: SRLR Eigenvalue Parameterization and Common-Radius Reference

![](images/cf35251f020e5ae23c15acb31b25c5dcfef6a88bd3857a8d4c1d049ebaa7c01b.jpg)  
FIGURE 1. Concept of the Sector-Ring Linear Reservoir. The preprocessed input eu enters through input weights scaled using the power spectral density (PSD) of the training input (Section III-A3c). Rotation–decay blocks assigned to sectors generate the states of the modes, which a fixed orthogonal mixing and a fixed tanh map transform into features for the ridge readout. Here q(ϕ) = [cos ϕ, sin ϕ]<sup>T</sup> denotes the unit direction of the input weights of one mode. In the pipeline and the bottom row, H denotes the states h stacked in rows over time, so that the rows of HQ are the mixed states Q<sup>T</sup>h (Section III-B3). The upper-right callout summarizes the candidate-evaluation computation (Section III-B3): a state bank indexed by the half-life value (the per-sector decay timescale, Section III-A3b) reuses states across candidates. An associative scan is invoked only for a half-life value that the candidate requires and the bank does not yet hold.

We now describe the eigenvalue distribution of the SRLR. The angles are fixed at the midpoints

$$
\theta _ { j } = \left( j - { \textstyle \frac 1 2 } \right) \frac { \pi } { M } , \qquad j = 1 , \hdots , M ,\tag{5}
$$

of the M equal subintervals of $( 0 , \pi )$ . The conjugate eigenvalue of each pair lies at angle −θ<sub>j</sub>, and the eigenvalue pairs are therefore symmetric about the real axis. Placing the angles at the midpoints avoids the angles 0 and π, at which $\mathbf { R } ( \theta ) =$ ±I and the block would have a repeated real eigenvalue, so that its two coordinates would carry the same response up to a scale factor. The interval $( 0 , \pi )$ is then divided into K consecutive sectors of equal width, and block j belongs to sector

$$
s ( j ) = \left\lfloor { \frac { K \theta _ { j } } { \pi } } \right\rfloor \in \{ 0 , \dots , K - 1 \} ,\tag{6}
$$

where ⌊·⌋ is the floor function (Fig. 1). The angles and the sector membership are thus fixed by (5) and (6), with no random draw. Below, k denotes a sector index, and $s ( j )$ denotes the index of the sector containing blockj. When M is a multiple of K, each sector contains exactly M/K modes. In the experiments of this paper, $N \ = \ 6 4 , M \ = \ 3 2$ , and $K \ = \ 8 .$ , so that each sector contains four modes. The k-th sector is assigned one positive half-life $\eta _ { k }$ , and the vector $\pmb { \eta } = \left( \eta _ { 0 } , \dots , \eta _ { K - 1 } \right)$ of the sector half-lives is called the halflife profile (the decay-timescale profile of Section I). The half-life is the number of updates over which the amplitude of a mode decays by half, and it serves as the decay timescale of the mode. In terms of a time constant, the amplitude decays as $e ^ { - t / \tau _ { k } }$ with $\tau _ { k } = \eta _ { k } /$ ln 2. The radius of blockj is determined by the half-life of its sector as

$$
r _ { j } = 2 ^ { - 1 / \eta _ { s ( j ) } } ,\tag{7}
$$

so that its amplitude halves every $\eta _ { s ( j ) }$ updates. Since $\eta _ { s ( j ) } > 0$ gives $0 < r _ { j } < 1$ , every eigenvalue lies inside the unit circle. When the reservoir is driven with the same input from any two initial states, the difference between the two state trajectories vanishes asymptotically, which gives this linear recurrence the echo state property [32].

The common-radius reference is the special case of the SRLR with a single sector, $K \ = \ 1$ , in which all blocks share one half-life η and hence one radius $r ~ = ~ 2 ^ { - 1 / \eta _ { 0 } }$ As in a linear SCR, all of its eigenvalues have the same magnitude r and evenly spaced angles (Section II-B). The rest of the construction, namely the fixed angles, the inputweight scaling, the fixed feature map, and the common ridge readout and scoring, is unchanged. The search spaces and computational budgets of the two models are described in Section III-B2.

## c: Input-Weight Scaling From the Training-Input Power Spectrum

In a linear reservoir such as the SRLR, the input weights require attention in addition to the eigenvalues. Mode j responds most strongly to input components whose normalized angular frequency is close to $\theta _ { j }$ [13], and the variance of its state is determined by the power spectral density (PSD) of the preprocessed training input weighted by the frequency response of block j. Consequently, the scale of each mode’s state changes with the eigenvalue setting. Existing normalizations that assume white-noise input [45] (Section II-B) do not account for this frequency-selective response to an input that is not white. We therefore scale the input weights using the PSD of the preprocessed training input.

To control this variation in scale across candidate eigenvalue distributions, the input weights are set analytically so that the states of all modes have a common predicted scale. The input weights of block j are written as the product of a scalar gain $g _ { j }$ and a unit direction $\mathbf { q } ( \phi _ { j } )$ ，

$$
{ \bf w } _ { j } = g _ { j } { \bf q } ( \phi _ { j } ) , \qquad { \bf q } ( \phi ) = \left[ \cos \phi \atop \sin \phi \right] .\tag{8}
$$

For each block, the direction $\phi _ { j }$ is determined from the PSD of the preprocessed training input so that the two coordinates of the mode’s state have equal predicted variance. With $\kappa _ { j }$ $( > 0 )$ denoting the predicted variance at unit gain and $\sigma _ { * }$ the target scale (the target standard deviation of each coordinate), the gain is $g _ { j } ~ = ~ \sigma _ { * } / \sqrt { \kappa _ { j } }$ , which gives both coordinates the target variance $\sigma _ { * } ^ { 2 }$ . In this prediction, the variance of the state of mode $j$ is obtained by integrating the one-sided Welch PSD [46] of the preprocessed training input, weighted by the frequency response of block $j ,$ over $[ 0 , f _ { s } / 2 ]$ , where $f _ { s }$ is the sampling frequency of the input; in the derivation the frequency axis is normalized to $f _ { s } ~ = ~ 2 \pi$ , so that $f$ is the normalized angular frequency in radians per update. The derivation is given in Appendix $\mathbf { A } .$

To distinguish the designs in the comparison, the design above is referred to as the PSD-scaled design, and two alternative input-weight designs are also evaluated in the same SRLR, with the fixed angles, the sector-wise half-life parameterization, the fixed feature map, and the ridge readout unchanged. The γ-normalized design, named after the $\gamma$ normalization of [45], determines the direction $\phi _ { j }$ and the gain $g _ { j }$ by the same procedure as above but uses, instead of the PSD of the training input, the flat one-sided PSD of unitvariance white noise, $P ( f ) = 2 / f _ { s }$ . Its input weights therefore do not depend on the training input, and they equalize the scale of each mode’s state under the assumption of whitenoise input, the assumption made by the normalizations cited in Section II-B. The uniform random design draws each entry of the input weights of all modes uniformly from (−1, 1) once per model seed, holds them fixed, and multiplies them by an input scaling $s _ { \mathrm { i n } } .$ . Because this design has no $\sigma _ { * }$ and no permode variance adjustment, $s _ { \mathrm { i n } }$ is selected on the validation window, as for the ESN (Section III-B2).

## d: Fixed Nonlinear Feature Map and Model Settings

Unlike the ESN and the LI-ESN, the SRLR has a linear recurrent state update, and its nonlinearity is introduced by a fixed feature map (Section II-B). The feature map uses tanh, the same function that the ESN and LI-ESN use inside their recurrence. Here it is applied once, outside the recurrence, to a fixed orthogonal mixing of the state to form the feature vector, which is

$$
\chi _ { t } = [ \operatorname { t a n h } ( \mathbf { Q } ^ { \mathsf { T } } \mathbf { h } _ { t } ) ^ { \mathsf { T } } , 1 ] ^ { \mathsf { T } } \in \mathbb { R } ^ { N + 1 } , \qquad \mathbf { Q } ^ { \mathsf { T } } \mathbf { Q } = \mathbf { I } ,\tag{9}
$$

where $\mathbf { Q } \in \mathbb { R } ^ { N \times N }$ is a fixed orthogonal mixing matrix, tanh is applied element-wise, and the final entry 1 is the bias entry.

The fixed orthogonal matrix Q supplies the mixing between modes, so that each feature is a nonlinear function of many modes rather than of a single mode. Because the modes do not interact in the block-diagonal recurrence, $\mathbf { Q } ^ { \top }$ instead maps the reservoir state $\mathbf { h } _ { t }$ into dense mixed coordinates, preserving its Euclidean norm, before the element-wise tanh. The matrix $\mathbf { Q }$ is generated once per model seed as a Haar-distributed orthogonal matrix (the Q factor of the QR decomposition of a standard-normal matrix, with its column signs fixed), and is then held fixed, together with the tanh map, across all candidate evaluations of that seed.

Overall, three settings determine an SRLR model: the halflife profile $\eta$ determines the sector radii of the recurrence, the target scale $\sigma _ { * }$ determines the input-weight scaling, and β determines the ridge readout. The angles, the feature map, and the scoring procedure are held fixed. For the uniform random input-weight design (Section III-A3c), the input scaling $s _ { \mathrm { i n } }$ replaces $\sigma _ { * }$ among the three settings.

## B. BENCHMARK DESIGN AND SRLR CANDIDATE EVALUATION

## 1) Benchmark Time Series and Tasks

We evaluate all models on supervised prediction and reconstruction tasks built from the Lorenz [47] and Rössler [48] systems. The Lorenz system is

$$
\begin{array} { r l } & { \dot { x } _ { \mathrm { L } } = \sigma _ { \mathrm { L } } ( y _ { \mathrm { L } } - x _ { \mathrm { L } } ) , } \\ & { \dot { y } _ { \mathrm { L } } = x _ { \mathrm { L } } ( \rho _ { \mathrm { L } } - z _ { \mathrm { L } } ) - y _ { \mathrm { L } } , } \\ & { \dot { z } _ { \mathrm { L } } = x _ { \mathrm { L } } y _ { \mathrm { L } } - \beta _ { \mathrm { L } } z _ { \mathrm { L } } , } \end{array}
$$

with $\sigma _ { \mathrm { L } } = 1 0 , \rho _ { \mathrm { L } } = 2 8 , \beta _ { \mathrm { L } } = 8 / 3$ , and the Rössler system is

$$
\begin{array} { r l } & { \dot { x } _ { \mathrm { R } } = - y _ { \mathrm { R } } - z _ { \mathrm { R } } , } \\ & { \dot { y } _ { \mathrm { R } } = x _ { \mathrm { R } } + a _ { \mathrm { R } } y _ { \mathrm { R } } , } \\ & { \dot { z } _ { \mathrm { R } } = b _ { \mathrm { R } } + z _ { \mathrm { R } } ( x _ { \mathrm { R } } - c _ { \mathrm { R } } ) , } \end{array}
$$

with $a _ { \mathrm { R } } ~ = ~ 0 . 1 5 , ~ b _ { \mathrm { R } } ~ = ~ 0 . 2 , ~ c _ { \mathrm { R } } ~ = ~ 1 0$ . Each system is integrated using a fixed-step fourth-order Runge–Kutta method from the initial state $( x _ { S } ( 0 ) , y _ { S } ( 0 ) , z _ { S } ( 0 ) ) = ( 1 , 1 , 1 )$ $S \in \{ \mathrm { L } , \mathrm { R } \}$ , with an internal step of 0.002 and a sampling interval of 0.02. After discarding the initial point and an initial transient of 2,000 saved samples, one trajectory of 22,012 samples is retained. For each system S we build four observed-input tasks that differ in the prediction target and the horizon: $x _ { S } ( t ) ~  ~ x _ { S } ( t + 1 ) , x _ { S } ( t ) ~  ~ x _ { S } ( t + 1 2 )$ $x _ { S } ( t )  z _ { S } ( t )$ , and $x _ { S } ( t )  z _ { S } ( t + 1 2 )$ , giving eight cases in total. Here the horizon is the number of saved samples from the input to the target, and horizons 1 and 12 correspond to 0.02 and 0.24 continuous-time units. Below, the $x _ { S }  x _ { S }$ tasks are called prediction tasks, and the $x _ { S }  z _ { S }$ tasks, whose target is a variable that the model does not observe, are called reconstruction tasks; the task $x _ { S } ( t )  z _ { S } ( t )$ is reconstruction at horizon 0. All tasks are open-loop: the model always receives the observed input $x _ { S } ( t )$ , and its predictions are never fed back.

2) Training, Validation, Candidate Search, and Test Protocol The input time series of each system is divided chronologically into warmup, training, validation, and test windows. In terms of the time index of the retained trajectory of Section III-B1, these are [0, 2000) for reservoir warmup, [2000, 12000) for training, [12000, 17000) for validation, and [17000, 22000) for test, with the final samples [22000, 22012) retained only as a target margin for the 12-step horizon. All preprocessing statistics are computed from the training window alone. The input is standardized by its training mean and standard deviation, and this transform is applied to the warmup, training, and validation and test inputs alike. The targets are standardized by their training statistics, and this transform is applied to the training targets and to the targets of the scored window. This defines the preprocessed input $\widetilde { u } _ { t }$ used by all models. Every run resets the reservoir state to zero and drives the reservoir with the 2,000 warmup inputs without scoring, so that the influence of the zero initial state has decayed before states are collected (the echo state property, Section II-A). States are then collected over the training window, and the run continues chronologically into one of the two scored passes (validation or test) described below. One fixed trajectory and one fixed chronological split are used per system, with no cross-validation.

The hyperparameters of each model are tuned separately for every task and model seed. All reservoir models use $N ~ = ~ 6 4$ units, and the search coordinates (the hyperparameters tuned by the search) are $( s _ { \mathrm { i n } } , r , \beta )$ for the ESN and $( s _ { \mathrm { i n } } , r , \alpha _ { \ell } , \beta )$ for the LI-ESN. An SRLR candidate is the tuple $\mathbf { c } ~ = ~ ( \sigma _ { * } , \eta , \beta )$ , where the half-life profile $\eta$ assigns one half-life to each of the K sectors. For the two alternative input-weight designs of the SRLR (Section III-$\mathbf { A } 3 \mathbf { c } )$ , the candidate is $( \sigma _ { * } , \eta , \beta )$ for the γ-normalized design and $( s _ { \mathrm { i n } } , \eta , \beta )$ for the uniform random design, with the same range of $s _ { \mathrm { i n } }$ as for the ESN. The common-radius reference is searched in its own coordinates $( \sigma _ { * } , r , \beta )$ . The shared radius r is one continuous parameter, and this model is the $K = 1$ special case of Section III-A3b. Its range of $r$ in Table 1, $[ \bar { 2 } ^ { - 4 } , 2 ^ { - 1 / 8 } ]$ , corresponds under (7) to half-lives from 0.25 to 8, the smallest and largest values allowed for η<sub>k</sub>. For every system, task, and model seed $\begin{array} { c c c } { { m } } & { { = } } & { { 0 , \ldots , 9 . } } \end{array}$ , we run an independent hyperparameter search using the Tree-structured Parzen Estimator (TPE) [49], a sequential model-based optimizer that proposes each new candidate from the validation scores of the candidates evaluated so far, implemented in the hyperparameter-optimization library Optuna [50], with sampler seed $4 2 + m$ . The six search configurations, namely the four models and the two alternative input-weight designs of the SRLR (Section III-A3c), use the same evaluationbudget schedule and stopping rule. Each search starts with 2,048 evaluations (one evaluation scores one candidate on the validation window) and is extended in 1,024-evaluation increments until the stopping rule below is met or the search reaches 8,192 evaluations. The searches remain independent across systems, tasks, and model seeds, and stopping is decided for each search configuration and system–task case on the basis of all ten of its model seeds; consequently, the ten searches of one search configuration and case end at the same evaluation count, which may differ between cases. The search coordinates and domains of each search configuration are summarized in Table 1, and the realized evaluation counts are given per case in Appendix B. The optimization histories of these searches are shown in Appendix B.

The stopping rule is as follows. For each search, let $E ^ { * } ( n )$ denote the lowest validation NRMSE among the first n evaluations, and define the relative improvement over the previous $\Delta n = 1 , 0 2 4$ evaluations as

$$
\delta ( n ) = \frac { E ^ { * } ( n - \Delta n ) - E ^ { * } ( n ) } { E ^ { * } ( n - \Delta n ) } .\tag{10}
$$

The rule is checked at $n = 3 , 0 7 2 , 4 , 0 9 6 , \ldots , 8 , 1 9 2$ , and the searches of a case stop at the first n at which the median of $\delta ( n )$ over the ten model seeds of that case falls below 0.5%. The searches of a case that has not met the rule at 8,192 evaluations are stopped at that count and reported as having reached the evaluation budget.

Model settings are selected on the validation window and then evaluated once on the test window with all settings fixed. During the search, each candidate is scored on the validation window: the state is reset, the reservoir is driven with the warmup inputs without scoring, the readout is fitted on the training window, and the run continues into the validation window. For each case and model seed, the candidate with the lowest finite validation NRMSE is selected (Section III-B4). The final test repeats the same chronological pass with the selected settings and no tuning. The reservoir is driven with the validation inputs without scoring so that its state remains synchronized with the input sequence, and the NRMSE is computed on the test window only.

## 3) SRLR Candidate-Evaluation Computation

Because the SRLR searches over eigenvalue distributions, the number of candidate combinations is large and the search is computationally expensive. We therefore reduce this cost by reusing states across candidates through a state bank indexed by the discrete half-life values allowed for $\eta _ { k }$ in Table 1 and kept over one search (one system, task, and model seed). For a candidate $\mathbf { c } ~ = ~ ( \sigma _ { * } , \eta , \beta )$ of the PSD-scaled design, Algorithm 1 returns its validation NRMSE using the state bank $\boldsymbol { B }$ of the current search, which starts empty and is kept across candidates, and the mixing matrix Q of the model seed. The states for a half-life value are generated at $\sigma _ { * } = 1$ by an associative scan only when the bank does not yet hold them (lines 1–7), and the candidate is then assembled from the stored states (lines 8–11). In Algorithm 1, $\mathbf { H } _ { \mathrm { t r } } \in \mathbb { R } ^ { T _ { \mathrm { t r } } \times N }$ and $\mathbf { H } _ { \mathrm { v } } ~ \in ~ \mathbb { R } ^ { T _ { \mathrm { v } } \times N }$ are the states $\mathbf { h } _ { t }$ stacked in rows over the training window and the validation window (of length $T _ { \mathrm { v } }$ samples), so that the row form HQ corresponds to ${ \mathbf { Q } } ^ { \mathsf { T } } { \mathbf { h } } _ { t }$ in (9). The matrix $\mathbf { Y } _ { \mathrm { v } } \in \mathbb { R } ^ { T _ { \mathrm { v } } \times d _ { \mathrm { y } } }$ denotes the validation targets stacked in the same way, and 1 in line 12 is a column of ones.

TABLE 1. Search spaces. Realized evaluation counts are given per case in Appendix B. Here (lin) and (log) denote uniform sampling on a linear and a logarithmic scale, respectively.  
Search configuration Coordinates and domains   
ESN $s _ { \mathrm { i n } } \in [ 0 . 1 , 3 . 0 ]$ (lin), $r \in [ 0 . 2 , 1 . 5 ]$ (lin), $\beta \in [ 1 0 ^ { - 9 } , 1 0 ^ { - 3 } ]$ (log)   
LI-ESN ESN coordinates plus $\alpha _ { \ell } \in [ 0 . 0 0 1 , 1 ]$ (lin)   
SRLR, PSD-scaled input weights $\sigma _ { * } \in [ 0 . 2 5 , 4 ] ( \mathrm { l i n } ) , \beta$ (log, as above), each $\eta _ { k } \in \{ 0 . 2 5 , 0 . 5 , 1 , 2 , 4 , 8 \}$   
Common-radius reference $\sigma _ { * } , \beta$ as above, shared $r \in [ 2 ^ { - 4 } , 2 ^ { - 1 / 8 } ]$ (approximately [0.0625, 0.9170]; lin, continuous)   
SRLR, γ-normalized input weights $\sigma _ { * } , \beta , \eta$ as above   
SRLR, uniform random input weights $\underline { { s _ { \mathrm { i n } } \in [ 0 . 1 , 3 . 0 ] } }$ (lin), β, η as above   
Note: The evaluated SRLR has ${ \overline { { K = 8 } } } .$

Algorithm 1 SRLR candidate evaluation with a state bank.   
1: for each distinct half-life value ℓ in η do   
2: if B has no entry for ℓ then   
3: Set the half-life of every sector to ℓ and $\sigma _ { * }$ to 1, and   
compute the input weights $\mathbf { w } _ { j }$ of all M modes from   
the training-input PSD by (8) (Section III-A3c).   
4: From the zero state, drive the reservoir with the   
warmup, training, and validation inputs in sequence   
by the associative scan (Section III-B3a).   
5: Store the states of all M modes over the training and   
validation windows in $\boldsymbol { B }$ under $\ell .$   
6: end if   
7: end for   
8: for $k = 0$ to $K - 1$ do   
9: Copy from $B [ \eta _ { k } ]$ the state columns of the modes in   
sector $k$ into the unit-scale state matrices $\mathbf { H } _ { \mathrm { t r } } ^ { \mathrm { u n i t } }$ and   
H<sup>unit</sup>.   
10: end for   
11: $\mathbf { H } _ { \mathrm { t r } }  \sigma _ { * } \mathbf { H } _ { \mathrm { t r } } ^ { \mathrm { u n i t } } , \mathbf { H } _ { \mathrm { v } }  \sigma _ { * } \mathbf { H } _ { \mathrm { v } } ^ { \mathrm { u n i t } }$   
12: $\Phi _ { \mathrm { t r } } \gets [ \operatorname { t a n h } ( \mathbf { H } _ { \mathrm { t r } } \mathbf { Q } ) , \mathbf { 1 } ] , \Phi _ { \mathrm { v } } \gets [ \operatorname { t a n h } ( \mathbf { H } _ { \mathrm { v } } \mathbf { Q } ) , \mathbf { 1 } ] \log ( 9 )$   
13: Obtain $\mathbf { W _ { \mathrm { o u t } } }$ from $\Phi _ { \mathrm { t r } }$ and $\mathbf { Y } _ { \mathrm { t r } }$ with $\beta$ by (1).   
14: return the NRMSE of $\Phi _ { \mathrm { v } } \mathbf { W } _ { \mathrm { o u t } }$ against $\mathbf { Y } _ { \mathrm { v } }$ by (12).

## a: Associative-Scan State Generation

The associative scan [16], [17] reorganizes the temporal recurrence of the modes at one half-life value so that, with unlimited parallel processors, the number of sequential steps along time grows only logarithmically with the number of updates (a logarithmic span). The state of each mode is first represented as one complex number $\zeta _ { j , t } ~ = ~ x _ { j , t } ^ { ( 1 ) } + i x _ { j , t } ^ { ( 2 ) }$ with eigenvalue $\lambda _ { j } ~ = ~ r _ { j } e ^ { i \theta _ { j } }$ and input term $\bar { d _ { j , t } } ~ = ~ ( w _ { j } ^ { ( 1 ) } +$ $i w _ { j } ^ { ( 2 ) } ) \widetilde { u } _ { t }$ , where $x _ { j , t } ^ { ( 1 ) }$ and $x _ { j , t } ^ { ( 2 ) }$ denote the two coordinates of $\mathbf { x } _ { j , t }$ , and $w _ { j } ^ { ( 1 ) }$ and $w _ { i } ^ { ( 2 ) }$ those of $\mathbf { w } _ { j }$ . The rotation–decay update then becomes the element-wise complex recurrence $\zeta _ { j , t } = \lambda _ { j } \zeta _ { j , t - 1 } + d _ { j , t }$ . Each update is an element-wise affine map ζ 7→ a ⊙ ζ + b on the vector of the states of all modes, $\boldsymbol { \zeta _ { t } } \ = \ ( \zeta _ { 1 , t } , \dots , \zeta _ { M , t } )$ , with $\mathbf { a } ~ = ~ ( \lambda _ { 1 } , \ldots , \lambda _ { M } )$ , the same for every update, and $\mathbf { b } = ( d _ { 1 , t } , \ldots , d _ { M , t } )$ at update $t ,$ where ⊙ denotes the element-wise product, and two such maps compose as

$$
( \mathbf { a } _ { 2 } , \mathbf { b } _ { 2 } ) \circ ( \mathbf { a } _ { 1 } , \mathbf { b } _ { 1 } ) = ( \mathbf { a } _ { 2 } \odot \mathbf { a } _ { 1 } , \ \mathbf { a } _ { 2 } \odot \mathbf { b } _ { 1 } + \mathbf { b } _ { 2 } ) ,\tag{11}
$$

which is associative. An associative scan over these elements therefore reproduces the serial recurrence, up to floatingpoint differences. For $T _ { \mathrm { d r v } }$ updates (the number of inputs driven in one scan, here the warmup, training, and validation inputs in sequence; line 4 of Algorithm 1), the implemented scan runs in $O ( \log T _ { \mathrm { d r v } } )$ stages of ${ \cal O } ( T _ { \mathrm { d r v } } N )$ element-wise operations, that is, $O ( T _ { \mathrm { d r v } } N$ log $T _ { \mathrm { d r v } } )$ operations in total, with ${ \cal O } ( T _ { \mathrm { d r v } } N )$ memory for the states.

## b: Across-Candidate State Reuse by Half-Life Value

The state bank reuses states across SRLR candidates. This reuse is possible because the SRLR restricts each $\eta _ { k }$ to the finite set of half-life values in Table 1. The γ-normalized design uses the same procedure with its own bank, where line 3 of Algorithm 1 uses the flat white-noise PSD. The common-radius search cannot use the bank because its radius takes continuous values, and the uniform random design is not supported by the bank in the present implementation. Because the input gains g of Section III-A3c are proportional to $\sigma _ { * }$ and the recurrence is linear, the states for any $\sigma _ { * }$ are obtained by computing the states once at $\sigma _ { * } = 1$ and multiplying them by $\sigma _ { * }$ . A candidate with half-life profile $\eta$ (Section III-B2) is then assembled by selecting, for each sector, the stored state trajectories for the half-life value assigned to that sector and scaling them by $\sigma _ { * }$ (Algorithm 1, lines 8–11). The readout is fitted and scored as usual, with $\beta$ changing only the ridge solve. The states are independent of the model seed, which enters only through the mixing matrix $\mathbf { Q }$ of the feature map (Section III-A3d). The bank is kept per search so that the searches remain independent across tasks and model seeds, as for the other search configurations (Section III-B2), and so that the reported cost corresponds to a single task and model seed, as in practical use; sharing a bank across the searches of a system, whose states are identical, would reduce the count further in this benchmark. In each search, the number of mode trajectories computed thus falls from CM (all M modes recomputed for each of the C candidates evaluated in that search) to LM (computed once for each of the L distinct halflife values requested in that search). Here L is at most the number of allowed half-life values (six), however large C is.

## 4) Evaluation Metrics and Aggregation

For a scored window of length T, with $\hat { y } _ { t }$ the prediction and $y _ { t }$ the target at time t, the NRMSE is defined as

$$
{ \mathrm { N R M S E } } = { \sqrt { \frac { { \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } ( { \hat { y } } _ { t } - y _ { t } ) ^ { 2 } } { { \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } ( y _ { t } - { \bar { y } } ) ^ { 2 } } } } , \qquad { \bar { y } } = { \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } y _ { t } ,\tag{12}
$$

where the mean y is computed on the same scored window. This is the root-mean-square error divided by the population standard deviation of the target on that window. Each case is summarized by the mean and population standard deviation (SD) of the NRMSE across the ten model seeds.

## IV. RESULTS

## A. SRLR VERSUS THE COMMON-RADIUS REFERENCE

We first compare the SRLR $( K = 8 ,$ Section III-A3b) with its common-radius reference. Fig. 2 shows the mean test NRMSE for the eight system–task cases, and Table 2 reports the corresponding means and population SDs over the ten model seeds. The SRLR achieved a lower mean NRMSE than the common-radius reference in all eight cases. The reduction in mean NRMSE relative to the common-radius reference, computed from the unrounded means, ranged from 11.6% to 47.4% across the cases.

## B. COMPARISON WITH THE ESN AND LI-ESN BASELINES

We then compare the SRLR with the ESN and LI-ESN baselines. Fig. 3 shows the mean test NRMSE for the eight cases, and Table 3 reports the corresponding means and population SDs over the ten model seeds. The SRLR achieved the lowest mean NRMSE in five of the eight cases, the LI-ESN in three, and the ESN in none. The SRLR had the lowest mean in all four $x \ \to \ x$ prediction cases and in the Rössler $x \ \to \ z$ reconstruction case at horizon 12, and the LI-ESN had the lowest mean in the two x → z reconstruction cases at horizon 0 and in the Lorenz x → z reconstruction case at horizon 12.

## C. SELECTED HALF-LIFE PROFILES

Fig. 4 shows the half-life profiles selected by the validation searches for the SRLR, for the ten model seeds of each task. Within each system, the tasks share the observed input sequence, yet the selected profiles changed with the prediction target and the horizon. The profiles also differed across the ten model seeds.

## D. COMPARISON OF INPUT-WEIGHT DESIGNS

We also compare the three input-weight designs of the SRLR (Section III-A3c), namely the PSD-scaled, γ-normalized, and uniform random designs, each with the settings selected by its own search. Fig. 5 shows the mean test NRMSE for the eight cases, and Table 4 reports the corresponding means and population SDs over the ten model seeds. The PSDscaled design achieved the lowest mean NRMSE in four of the eight cases, the γ-normalized design in two, and the uniform random design in two. The largest differences among the three designs were in the Lorenz $x \ \to \ x$ prediction cases at both horizons, where the PSD-scaled design had the lowest mean, and in the $x \ \to \ z$ reconstruction cases of both systems at horizon 0, where it had the highest mean. In these four cases, the highest mean exceeded the lowest by about 26% to 28%. In the remaining four cases, the three means were within 10% of one another. The ordering of the three designs thus depended on the task.

## E. SCALE OF THE STATES OF THE MODES UNDER THE THREE INPUT-WEIGHT DESIGNS

Next, we examine how the three input-weight designs affect the scale of the state of each mode, using the profiles selected for the PSD-scaled design in Section IV-C as fixed settings. For each of the 40 task-by-seed profiles per system, the recurrence and the observed input sequence are fixed, only the input weights change, and the states of the modes are measured before the orthogonal mixing, the feature map, the readout, and the scoring. Each run starts from the zero state, drives the reservoir with the 2,000 warmup inputs without scoring, and retains the 10,000 states of the training window. For each mode $j ~ = ~ 1 , \dots , 3 2$ , the two state coordinates are treated separately, and the scale of each coordinate is its population SD over the retained states, $s _ { j , c } ~ = ~ \sqrt { \mathrm { V a r } ( x _ { j , t } ^ { ( c ) } ) }$ $c \in \{ 1 , 2 \}$ . This gives 64 coordinate scales per profile under each design. Each profile under each design is summarized by the ratio of the 90th to the 10th percentile (P90/P10) of its 64 coordinate scales, with both percentiles computed by linear interpolation. Because a gain common to all modes $( \sigma _ { * }$ for the PSD-scaled and γ-normalized designs and $s _ { \mathrm { i n } }$ for the uniform random design) multiplies all 64 coordinate scales alike, the ratio does not depend on its value. A ratio close to 1 indicates little relative variation in coordinate scale between P10 and P90 within the profile.

Fig. 6 shows this ratio. The PSD-scaled design gave a lower ratio than the other two designs in all 40 Lorenz profiles and all 40 Rössler profiles. The median ratio was close to 1 for the PSD-scaled design (1.00 for both systems), compared with 9.86 and 10.2 on Lorenz and 34.0 and 12.3 on Rössler for the γ-normalized and uniform random designs, respectively.

## F. COMPUTATIONAL COST OF THE CANDIDATE SEARCH

Finally, we report the computational cost of the validation searches (Fig. 7). Each of the four search configurations in Fig. 7(a), namely the SRLR, the ESN, the LI-ESN, and the common-radius reference, evaluated 245,760 candidates in total (80 searches of 3,072 evaluations each), and every evaluation returned a finite validation NRMSE. The searches ran on a 14-core Apple M4 Pro processor, with the ten model seeds of each case run concurrently. Their full operational wall time was 3.29 h for the SRLR, 10.93 h for the ESN, 9.96 h for the LI-ESN, and 2.30 h for the common-radius reference. Over the 80 SRLR searches, recomputing all 32 modes for every candidate would have required $M \sum C =$ 7,864,320 mode trajectories, where the sum runs over the 80 searches. The state banks computed $M \sum L = 1 5 , 3 6 0 .$ , that is, 32 for each of the $L ~ \leq ~ 6$ half-life values requested in each search (L = 6 in every search), a reduction of 99.8% (Section III-B3b).

![](images/256ed17b081faa521fdad5c174ce70b8899fb4e1020c649d9d0bb3e8abf44123.jpg)  
FIGURE 2. Test NRMSE of the SRLR and the common-radius reference on (a) the four Lorenz tasks and (b) the four Rössler tasks. For each system, the panels show, in order, x → x prediction at horizon ${ \bf l } , { \pmb x }  { \pmb x }$ prediction at horizon $1 2 , x \to z$ reconstruction at horizon 0, and x → z reconstruction at horizon 12. Bars show means over ten model seeds, and error bars show population SD.

TABLE 2. Test NRMSE for the common-radius reference and the SRLR: mean ± population SD over ten model seeds. The lowest mean in each row is in bold.
<table><tr><td>Task</td><td>Common-radius reference</td><td>SRLR</td></tr><tr><td colspan="3">Lorenz</td></tr><tr><td> $x  x ,$  horizon 1 x → x, horizon 12</td><td> $( 8 . 2 9 \pm 1 . 6 3 ) \times 1 0 ^ { - 6 }$   $0 . 0 2 8 3 \pm 0 . 0 0 1 9 0$ </td><td> $( 4 . 3 6 \pm 0 . 5 0 7 ) \times 1 0 ^ { - 6 }$   $\mathbf { 0 . 0 2 0 9 } \pm 0 . 0 0 1 6 7$ </td></tr><tr><td>x → z, horizon 0  $x  z ,$  horizon 12</td><td> $0 . 1 3 2 \pm 0 . 0 0 7 4 8$   $0 . 1 6 2 \pm 0 . 0 0 8 6 2$ </td><td> $\mathbf { 0 . 1 0 9 \pm 0 . 0 1 1 2 }$   $\mathbf { 0 . 1 4 4 \pm 0 . 0 1 2 1 }$ </td></tr><tr><td colspan="3">Rössler</td></tr><tr><td>x → x, horizon 1</td><td> $( 4 . 1 2 \pm 0 . 4 3 5 ) \times 1 0 ^ { - 7 }$ </td><td> $( \mathbf { 3 . 2 2 \pm 0 . 1 7 5 } ) \times 1 0 ^ { - 7 }$ </td></tr><tr><td>x → x, horizon 12</td><td> $( 3 . 0 9 \pm 0 . 4 8 6 ) \times 1 0 ^ { - 4 }$ </td><td> $( \mathbf { 2 . 6 0 \pm 0 . 5 4 9 } ) \times 1 0 ^ { - 4 }$ </td></tr><tr><td> $x  z ,$  horizon 0</td><td> $( 6 . 9 3 \pm 0 . 6 4 5 ) \times 1 0 ^ { - 3 }$ </td><td> $( 5 . 6 5 \pm 0 . 2 4 1 ) \times 1 0 ^ { - 3 }$ </td></tr><tr><td>x → z, horizon 12</td><td> $0 . 0 1 8 5 \pm 0 . 0 0 1 0 1$ </td><td> $\mathbf { 0 . 0 1 5 5 \pm 0 . 0 0 1 6 9 }$ </td></tr></table>

In the separate controlled measurements of Fig. 7(b) and (c), made on the same machine with one worker process and one Basic Linear Algebra Subprograms (BLAS) thread, one evaluated candidate took on average 10.6 ms for the SRLR, 38.3 ms for the ESN, 55.8 ms for the LI-ESN, and 25.2 ms for the common-radius reference. The largest difference among the components was in the reservoir-state stage, which took

2.8 ms for the SRLR, 35.3 ms for the ESN, and 52.9 ms for the LI-ESN. The construction of the readout features, the largest component of the SRLR evaluation, took 4.1 ms. In Fig. 7(d), the time for Optuna to generate the next candidate increased as the searches accumulated candidates, most strongly for the SRLR. The full operational wall time of panel (a) also includes this candidate-generation time (Table 5), which was longer for the SRLR than for the common-radius reference in every block; this contributes to the shorter full wall time of the common-radius reference than of the SRLR despite its longer time per evaluated candidate. Table 5 summarizes the operations timed in panels (a), (b)–(c), and (d).

![](images/ad5e0fbf792b2fe9fc9799d8fd15b9a2682656766819d572d9b9ea696c1e52e1.jpg)  
FIGURE 3. Test NRMSE of the ESN, the LI-ESN, and the SRLR on (a) the four Lorenz tasks and (b) the four Rössler tasks, in the task order of Fig. 2. Bars show means over ten model seeds, and error bars show population SD.

TABLE 3. Test NRMSE for the ESN, the LI-ESN, and the SRLR: mean ± population SD over ten model seeds. The lowest mean in each row is in bold.
<table><tr><td>Task</td><td>ESN</td><td>LI-ESN</td><td>SRLR</td></tr><tr><td colspan="4">Lorenz</td></tr><tr><td> $x \to x , \mathrm { h o r i z o n } \ 1$ </td><td> $( 1 9 . 5 \pm 3 . 0 9 ) \times 1 0 ^ { - 6 }$ </td><td> $( 1 9 . 7 \pm 3 . 3 3 ) \times 1 0 ^ { - 6 }$ </td><td> $( 4 . 3 6 \pm 0 . 5 0 7 ) \times 1 0 ^ { - 6 }$ </td></tr><tr><td> $x \to x , { \mathrm { h o r i z o n ~ } } 1 2$ </td><td> $0 . 0 5 3 6 \pm 0 . 0 0 4 1 8$ </td><td> $0 . 0 3 9 5 \pm 0 . 0 0 4 5 6$ </td><td> $\mathbf { 0 . 0 2 0 9 } \pm 0 . 0 0 1 6 7$ </td></tr><tr><td> $x \to z , { \mathrm { h o r i z o n } } 0$ </td><td> $0 . 0 8 5 3 \pm 0 . 0 2 0 0$ </td><td> $\mathbf { 0 . 0 3 2 8 \pm 0 . 0 1 0 0 }$ </td><td> $0 . 1 0 9 \pm 0 . 0 1 1 2$ </td></tr><tr><td> $x  z ,$  horizon 12</td><td> $0 . 2 0 4 \pm 0 . 0 1 6 7$ </td><td> $\mathbf { 0 . 1 2 6 \pm 0 . 0 1 0 4 }$ </td><td> $0 . 1 4 4 \pm 0 . 0 1 2 1$ </td></tr><tr><td colspan="4">Rössler</td></tr><tr><td> $x  x ,$  horizon 1</td><td> $( 1 6 . 0 \pm 3 . 7 9 ) \times 1 0 ^ { - 7 }$ </td><td> $( 1 5 . 7 \pm 3 . 7 8 ) \times 1 0 ^ { - 7 }$ </td><td> $( 3 . 2 2 \pm 0 . 1 7 5 ) \times 1 0 ^ { - 7 }$ </td></tr><tr><td> $x  x ,$  horizon 12</td><td> $( 1 4 . 8 \pm 3 . 5 8 ) \times 1 0 ^ { - 4 }$ </td><td> $( 1 2 . 3 \pm 3 . 6 0 ) \times 1 0 ^ { - 4 }$ </td><td> $( 2 . 6 0 \pm 0 . 5 4 9 ) \times 1 0 ^ { - 4 }$ </td></tr><tr><td> $x  z ,$  horizon 0</td><td> $( 6 1 . 3 \pm 2 5 . 7 ) \times 1 0 ^ { - 3 }$ </td><td> $( { \bf 3 . 4 2 \pm 0 . 6 3 6 } ) \times 1 0 ^ { - 3 }$ </td><td> $( 5 . 6 5 \pm 0 . 2 4 1 ) \times 1 0 ^ { - 3 }$ </td></tr><tr><td> $x  z ,$  horizon 12</td><td> $0 . 0 6 4 8 \pm 0 . 0 1 2 7$ </td><td> $0 . 0 4 4 4 \pm 0 . 0 0 4 2 6$ </td><td> $\mathbf { 0 . 0 1 5 5 \pm 0 . 0 0 1 6 9 }$ </td></tr></table>

## V. DISCUSSION

## A. BENCHMARK OUTCOMES AND TASK-DEPENDENT TIMESCALES

The two comparisons show that allocating the decay timescale sector by sector is a useful degree of freedom, and that the SRLR with this freedom is, as a whole, competitive with the nonlinear baselines. First, the SRLR achieved a lower mean test NRMSE than the common-radius reference in all eight system–task cases (Fig. 2). Because the two differ only in whether the decay timescale may vary across sectors, this difference can be attributed to that freedom. Second, in the comparison with the ESN and LI-ESN, the lowest mean NRMSE was achieved by the SRLR in five of the eight cases, by the LI-ESN in three, and by the ESN in none (Fig. 3). The

SRLR, whose recurrence is linear, is therefore competitive with the separately tuned nonlinear baselines.

Suitable reservoir dynamics are task-dependent, even for a single input sequence. Within each system, the four tasks share the observed input sequence, and the validation-selected half-life profiles changed with the prediction target and the horizon (Fig. 4). The two nonlinear baselines show the same dependence. Leaky integration, which distinguishes the LI-ESN from the ESN, is a standard way to adjust the timescale of the state update of an ESN to the task [9] and can substantially improve prediction in some tasks [36]. The mean-error gap between the separately tuned ESN and LI-ESN, however, differed across the target–horizon cases, and in the two x → x prediction cases at horizon 1 their means were close (Fig. 3, Table 3). Both patterns are consistent with the prior observation that, under a shared input, different prediction targets require and select different reservoir timescales [11], and with the view that the useful fading memory (the memory of past inputs, which decays with time) is set by the target computation [8].

(a) Lorenz  
![](images/a7c31055e3fc40e8a84cef9f899f4567a3aac4279bfccf001a501ab1a0c443b0.jpg)

![](images/341816380adcde2bfe15bde1e36c260a393a8f00b2b6bd0010023bee9d43db9a.jpg)

![](images/e70587d4d4f696af98629c671a3b638e4f6259e8166e943e4c886e0cbb269392.jpg)

![](images/12b108af1007d6d42b76e9e0d50e4a6664eaac9f5ffead91f38ec5e6ef1959b7.jpg)

(b) Rössler  
![](images/d2c2a15aeb71edefa63e938d1b2e9804edc2b9c2dbbc74477769851bc12c53bb.jpg)

![](images/595fff18b25864e8b36f0823c44ca3a3d14f33d2032bb079eeeafa513a90f8f2.jpg)

![](images/3495ebd79bed843fc36cf255048a0013115f45d70252fb07094a924792476d53.jpg)

![](images/8e12e7d3afc948bb5d48bba627fd72b58fe911efa5ce0a7b4102be2d6d1326f1.jpg)  
FIGURE 4. Half-life profiles selected by the validation searches for the SRLR on (a) the four Lorenz tasks and (b) the four Rössler tasks, in the task order of Fig. 2. Each task panel is a heatmap with the six half-life values η ∈ {0.25, 0.5, 1, 2, 4, 8} (in reservoir updates) as rows and the sectors 0–7 as columns (sector 0 contains the smallest angles θ , near 0, and sector 7 the largest, near π; see (6)). The color of a cell indicates the fraction of the ten model seeds that selected that half-life for that sector, and cells selected by at least five seeds are labeled with the seed count.

(a) Lorenz  
![](images/f8c40b653c119b96ea28540bb1f1ace1aa3638f5593a7cdb2c00f681475c3b46.jpg)

![](images/6e004a669667ed489b1f53eb6a2ff0ab7c9cfc0f1d491147829ddce89d623ebe.jpg)

![](images/c4f23885e3f760cbed052b317a2d4393ef683f4d622790cdbe2907b10b9e3a09.jpg)

![](images/9087d0c41964687a5df28f62206e1f90e1d09216e3285cb9479e516f6bfb0e08.jpg)

(b) Rössler  
![](images/7c0a290ff896ef29b1b1d5f4cc3dd7dab4ecf37221a33cfd7a757061afbaaf16.jpg)

![](images/3f5e509000ed89ce65f25a377ee39ec41ffa5230fe6993fa99e80c8efc885c26.jpg)

![](images/ac0542b117ad3f01968fb1124aa437b3b4f0bc32f77059a17f928e91d958fa38.jpg)

![](images/02fdb1476d159d32e776639bc5b1d2f7369fec04fb3deaf393c5935644967c8e.jpg)  
FIGURE 5. Test NRMSE of the three input-weight designs of the SRLR (PSD-scaled, γ-normalized, and uniform random) on (a) the four Lorenz tasks and (b) the four Rössler tasks, in the task order of Fig. 2, each design with the settings selected by its own search. Bars show means over ten model seeds, and error bars show population SD.

TABLE 4. Test NRMSE for the three input-weight designs of the SRLR: mean ± population SD over ten model seeds. The lowest mean in each row is in bold.
<table><tr><td>Task</td><td>PSD-scaled</td><td>γ-normalized</td><td>Uniform random</td></tr><tr><td colspan="4">Lorenz</td></tr><tr><td>x → x, horizon 1</td><td> $( 4 . 3 6 \pm 0 . 5 0 7 ) \times 1 0 ^ { - 6 }$ </td><td> $( 5 . 5 6 \pm 0 . 6 9 1 ) \times 1 0 ^ { - 6 }$ </td><td> $( 4 . 5 9 \pm 0 . 4 2 4 ) \times 1 0 ^ { - 6 }$ </td></tr><tr><td>x → x, horizon 12</td><td> $\mathbf { 0 . 0 2 0 9 } \pm 0 . 0 0 1 6 7$ </td><td> $0 . 0 2 4 5 \pm 0 . 0 0 2 2 2$ </td><td> $0 . 0 2 6 5 \pm 0 . 0 0 2 8 6$ </td></tr><tr><td> $x \to z , { \mathrm { h o r i z o n } } 0$ </td><td> $0 . 1 0 9 \pm 0 . 0 1 1 2$ </td><td> $0 . 0 9 2 3 \pm 0 . 0 1 1 3$ </td><td> $\mathbf { 0 . 0 8 6 4 } \pm 0 . 0 0 9 9 0$ </td></tr><tr><td> $x  z ,$  horizon 12</td><td> $\mathbf { 0 . 1 4 4 \pm 0 . 0 1 2 1 }$ </td><td> $0 . 1 4 7 \pm 0 . 0 0 6 1 4$ </td><td> $0 . 1 5 4 \pm 0 . 0 0 8 0 6$ </td></tr><tr><td colspan="4">Rössler</td></tr><tr><td>x → x, horizon 1</td><td> $( { \bf 3 . 2 2 } \pm 0 . 1 7 5 ) \times 1 0 ^ { - 7 }$ </td><td> $( 3 . 4 4 \pm 0 . 4 6 9 ) \times 1 0 ^ { - 7 }$ </td><td> $( 3 . 2 3 \pm 0 . 3 6 0 ) \times 1 0 ^ { - 7 }$ </td></tr><tr><td>x → x, horizon 12</td><td> $( 2 . 6 0 \pm 0 . 5 4 9 ) \times 1 0 ^ { - 4 }$ </td><td> $( 2 . 5 8 \pm 0 . 4 4 1 ) \times 1 0 ^ { - 4 }$ </td><td> $( 2 . 3 7 \pm 0 . 6 6 6 ) \times 1 0 ^ { - 4 }$ </td></tr><tr><td>x → z, horizon 0</td><td> $( 5 . 6 5 \pm 0 . 2 4 1 ) \times 1 0 ^ { - 3 }$ </td><td> $( 4 . 4 8 \pm 0 . 2 4 7 ) \times 1 0 ^ { - 3 }$ </td><td> $( 4 . 8 8 \pm 0 . 4 6 4 ) \times 1 0 ^ { - 3 }$ </td></tr><tr><td>x → z, horizon 12</td><td> $0 . 0 1 5 5 \pm 0 . 0 0 1 6 9$ </td><td> $\mathbf { 0 . 0 1 4 8 \pm 0 . 0 0 1 3 1 }$ </td><td> $0 . 0 1 5 6 \pm 0 . 0 0 0 9 8 7$ </td></tr></table>

![](images/7200e15204a5128d58a7a7f7ca71439629378fb660c019e0cbe3dd0c46651d4a.jpg)

![](images/2dceb2171d26641dc9090e2d872f5ae11a688363eaa597e079b239630161796b.jpg)  
FIGURE 6. Ratio of the 90th to the 10th percentile (P90/P10) of the 64 coordinate scales within each profile, where each scale is the population SD of one coordinate of the states of the modes, for the PSD-scaled, γ-normalized, and uniform random input-weight designs (Section III-A3c) on (a) the Lorenz tasks and (b) the Rössler tasks. Each thin line connects the values of one profile (40 profiles, one per task and model seed) across the three designs. Large outlined symbols mark medians, and vertical bars span the interquartile range. The dotted line marks a ratio of 1.

## B. INPUT-WEIGHT SCALING, STATE SCALE, AND PREDICTION PERFORMANCE

The PSD scaling of the input weights was introduced to equalize the scale of the states of the modes across candidates, so that the scale of the states of the modes does not change with the eigenvalue distribution among candidates. In Section IV-E, we tested how closely the realized states follow this design by fixing the recurrence, the input sequence, and the mode seed and changing only the input weights. In all 40 Lorenz and all 40 Rössler profiles, the PSD-scaled design gave a lower P90/P10 ratio than the other two designs, with median ratios close to 1 (Fig. 6). The PSD scaling therefore suppresses the variation in coordinate scale, so that differences in scale do not enter the comparison of decay-timescale profiles.

In terms of prediction performance, the comparison in Section IV-D shows how the three designs differ. The PSD-scaled design achieved the lowest mean NRMSE in four of the eight cases, and among these four the differences among the three designs were largest in the Lorenz $x \ \to \ x$ prediction cases. In the $x  z$ reconstruction cases at horizon 0, by contrast, it had the highest mean, about 26% above the lowest of the three means (Fig. 5). This indicates that equalizing the scale of the states of the modes does not by itself determine accuracy: the three designs distribute the input differently across the modes, and the distribution that suited each task differed. The role of the PSD scaling that the present results support is therefore the one tested in Section IV-E, namely keeping differences in scale out of the comparison of decay-timescale profiles. Future research directions include allocating the input gain of each mode according to the prediction target or the horizon, as in task-tailored reservoir design [51].

## C. SEED-LEVEL VARIABILITY AND UNRESOLVED CONTRIBUTORS

In addition to varying with the prediction target and the horizon, the selected profiles differed across the ten model seeds (Fig. 4). Because each model seed generates its own fixed orthogonal mixing matrix Q and runs its own finite TPE search, both can contribute to this variation. The present evidence does not separate the two contributions, and how each of them affects the selected profiles and the prediction performance is a question for future research.

(a) Wall time of the searches  
![](images/17a63bea71d5cfd2441e1d4c9cc28ea8ca997735fc461498b423eeffca82ab5d.jpg)  
(c) Components of the evaluation time

(b) Controlled evaluation time  
![](images/56e2727fb48d75306b2a3191ca8e46c7cdec6719d880521657b00408aa2a43c5.jpg)

![](images/9780e86dbe28fa3376608e96111d53f9af5c11422d67b54a7d6ff156dc627748.jpg)

(d) Optuna candidate generation  
![](images/39df38b22b80327f6d20fc7fcf0ab9a96a5bac3e320551d6fb7bcac2e5e12e80.jpg)  
each 1,024-candidate chunk  
FIGURE 7. Computational cost of the validation searches. (a) Full operational wall time and number of evaluations that returned a finite validation NRMSE for the SRLR, the ESN, the LI-ESN, and the common-radius reference (80 searches per search configuration). The searches of the four configurations ran sequentially, the ten seeds of each case ran concurrently, and other workloads on the machine were not controlled. (b) Time per evaluated candidate for the first 30 of the 128 replayed candidates (the first 128 candidates recorded in each search, re-evaluated in their recorded order) in each of 80 searches per search configuration, one per system, task, and model seed, measured with one worker process and one BLAS thread on one machine, using a fresh process for each search and, for the SRLR, a fresh state bank per search. Lines show medians across searches, and shading covers the middle 50% (25th–75th percentile range). For the SRLR, all six half-life values of the state bank had been computed by the third evaluation in all 80 searches, and a half-life value computed in an earlier evaluation could be reused before the bank was complete. (c) Mean evaluation-time components from the same measurements as panel (b), over all 128 replayed candidates (10,240 per search configuration). (d) Median time for Optuna to generate the next candidate before evaluation, recorded during the searches, over successive blocks of 1,024 candidates, separately for each case and for the four search configurations of panel (a) (10,240 measurements per point, 1,024 candidates × ten seeds). Solid and dashed lines denote Lorenz and Rössler, respectively.

TABLE 5. Operations included in the timing measurements reported in Fig. 7. ‘‘Measured’’ denotes the interval shown directly, whereas ‘‘Included’’ denotes work contained in the full operational wall time of panel (a), excluding time spent writing summaries after the recorded end of the searches.
<table><tr><td>Operation group</td><td>(a) Full operational wall time</td><td>(b)-(c) Evaluation timing</td><td>(d) Candidate generation</td></tr><tr><td>Optuna candidate generation</td><td>Included</td><td></td><td>Measured</td></tr><tr><td>Candidate evaluation</td><td>Included</td><td>Measured in (b); decomposed in (c)</td><td></td></tr><tr><td>Result reporting and storage</td><td>Included</td><td></td><td></td></tr><tr><td>Worker and process management, checkpoints, and finalization of the searches</td><td>Included</td><td></td><td></td></tr></table>

## D. COMPUTATIONAL COST AND SCALABILITY

Because the search for the SRLR has ten coordinates, eight of them the sector half-lives, against three for the ESN and four for the LI-ESN, evaluating candidates at low computational cost was a practical concern. In the recorded searches, the SRLR, the ESN, and the LI-ESN each evaluated 245,760 candidates, and the SRLR did so in 3.29 h, compared with 10.93 h for the ESN and 9.96 h for the LI-ESN (Fig. 7(a)). In the controlled per-candidate measurement, the SRLR also took 10.6 ms per candidate, against 38.3 ms for the ESN and 55.8 ms for the LI-ESN (Fig. 7(b) and (c)). The difference is concentrated in the reservoir-state stage (Fig. 7(c)), which is consistent with the implemented candidate evaluation (Algorithm 1). It computes the linear recurrence by an associative scan [16], [17], as in recent efficient linear-recurrence models [19], [45], and reuses the states for each half-life value across candidates, so that the recurrence is not recomputed for every candidate. Over the 80 searches, the state banks reduced the number of computed mode trajectories by 99.8% relative to per-candidate recomputation (Section IV-F).

The time for Optuna to generate the next candidate also increased as the searches progressed, most strongly for the SRLR (Fig. 7(d)). Because TPE builds its proposal distribution for each search coordinate from the accumulated evaluation history [49], both the length of that history and the number of search coordinates contribute to the cost of generating a candidate. The observed growth is consistent with this mechanism. For searches over many candidates, efficient candidate generation is therefore as important as efficient candidate evaluation.

The present evaluation used a reservoir of N = 64 units, and extending it to larger N will require further cost reduction. With the state computation reduced, the next bottleneck is the construction of the readout features (the largest component of the SRLR evaluation, 4.1 ms, Fig. 7(c)), in which applying the dense N ×N orthogonal matrix Q at every time step costs $O ( T N ^ { 2 } )$ . One option is to use Hadamard matrices. A composition of Hadamard and diagonal matrices, applied via the fast Hadamard transform, reduces this cost to O(TN log N) [52]. Whether such a replacement preserves the accuracy of the SRLR is an open problem.

## E. SCOPE AND FUTURE DIRECTIONS

These conclusions are based on eight system–task cases, two dynamical systems with four tasks each, evaluated over ten model seeds under a single protocol, with the SRLR at K = 8. Future research directions include evaluating the SRLR on a wide range of systems and on further prediction targets and horizons defined over a shared input. They also include relating the decay timescales selected by the SRLR to the timescales selected by nonlinear reservoirs such as the LI-ESN.

To make the eigenvalue distribution the design variable, the SRLR restricts its recurrence to one block-diagonal construction that is normal (it commutes with its transpose), whereas the randomly generated recurrences of standard ESNs are typically non-normal. Non-normal structure can introduce transient amplification through non-orthogonal modes [53], [54], a behavior that the present parameterization does not cover. How sector-wise decay-timescale allocation can be extended to non-normal recurrences is an open problem.

## VI. CONCLUSION

In this paper, we proposed the Sector-Ring Linear Reservoir (SRLR), which makes the eigenvalue distribution of its linear recurrence a design variable by dividing the eigenvalue angles into sectors and assigning one decay timescale, expressed as a half-life, to each sector. To compare candidate eigenvalue distributions with the scale of each mode’s state kept equal, the input weights are scaled using the power spectrum of the training input. To evaluate candidates at low computational cost, the states for each half-life value are reused across candidates. In two comparisons on prediction and reconstruction tasks built from chaotic systems, the SRLR achieved a lower mean test NRMSE than its common-radius reference in all eight cases, which supports sector-wise allocation of the decay timescale as a useful degree of freedom. It was also competitive with the nonlinear ESN and LI-ESN baselines. The selected half-life profiles changed with the prediction target and the horizon, showing that suitable reservoir dynamics are task-dependent. The PSD scaling of the input weights kept the scale of the states of the modes close to equal within each selected profile, so that differences in scale did not enter the comparison of decay-timescale profiles, and its effect on accuracy depended on the task. Compared with two alternative input-weight designs, the PSD-scaled design achieved the lowest mean NRMSE in four of the eight cases. The state reuse reduced the number of computed mode trajectories in the search by 99.8% relative to per-candidate recomputation. Future research will address a wider range of systems and tasks, the extension to non-normal recurrences, and further cost reduction for larger reservoirs.

## APPENDIX A INPUT-WEIGHT SCALING

This appendix derives the input-weight scaling used in Section III-A3c: for each block, the direction $\phi _ { j }$ that equalizes the predicted variances of the two coordinates of the mode’s state, and the gain $g _ { j }$ that sets them to the target variance $\sigma _ { * } ^ { 2 }$ The scaling uses the training input only through its one-sided PSD. The target scale $\sigma _ { * }$ is one of the three settings of an SRLR candidate and is selected on the validation window (Section III-B2).

Because the block recurrence of Section III-A3a acts on the input as a linear filter, its predicted response can be written in the frequency domain [13]. Let

$$
\begin{array} { r l r } { \displaystyle { z ( f ) = \exp \left( - i \frac { 2 \pi f } { f _ { s } } \right) } , } & { { } } & { } \\ { \displaystyle { \mathbf { G } _ { j } ( f ) = [ \mathbf { I } - z ( f ) \mathbf { R } _ { j } ] ^ { - 1 } } , } & { { } 0 \leq f \leq \frac { f _ { s } } { 2 } . } \end{array}
$$

For unit gain and the direction $\mathbf { q } ( \phi ) = [ \cos \phi , \sin \phi ] ^ { \mathsf { T } }$ , the predicted state covariance of block j is

$$
\mathbf { C } _ { j } ( \phi ) = \operatorname { R e } \int _ { 0 } ^ { f _ { s } / 2 } \widehat { P } _ { u } ( f ) \mathbf { G } _ { j } ( f ) \mathbf { q } ( \phi ) \mathbf { q } ( \phi ) ^ { \mathsf { T } } \mathbf { G } _ { j } ( f ) ^ { * } d f ,
$$

where <sup>∗</sup> denotes the conjugate transpose. Here $\widehat { P } _ { u }$ is a nonnegative one-sided Welch PSD [46] of the training input after standardization by the training-window mean and SD, computed with a Hann window, 1,024-sample segments, 50% overlap, and constant segment-mean detrending, with no further centering applied to the full series. The sampling rate is normalized to $f _ { s } ~ = ~ 2 \pi$ , and the density is rescaled so that its trapezoidal integral equals the population variance of the same standardized input.

Writing

$$
\begin{array} { l } { { a ( f ) = 1 - z ( f ) r _ { j } \cos \theta _ { j } , } } \\ { { \ b ( f ) = z ( f ) r _ { j } \sin \theta _ { j } , } } \\ { { D ( f ) = a ( f ) ^ { 2 } + b ( f ) ^ { 2 } , } } \end{array}
$$

the filtered direction is

$$
\mathbf { G } _ { j } ( f ) \mathbf { q } ( \phi ) = \frac { 1 } { D ( f ) } \left[ \begin{array} { l } { a ( f ) \cos \phi - b ( f ) \sin \phi } \\ { b ( f ) \cos \phi + a ( f ) \sin \phi } \end{array} \right] ,
$$

and with the three integrals

$$
\begin{array} { r l } & { S _ { j } = \displaystyle \int _ { 0 } ^ { f _ { s } / 2 } \widehat { P } _ { u } ( f ) \frac { | a | ^ { 2 } + | b | ^ { 2 } } { | D | ^ { 2 } } d f , } \\ & { \Delta _ { j } = \displaystyle \int _ { 0 } ^ { f _ { s } / 2 } \widehat { P } _ { u } ( f ) \frac { | a | ^ { 2 } - | b | ^ { 2 } } { | D | ^ { 2 } } d f , } \\ & { X _ { j } = \displaystyle \int _ { 0 } ^ { f _ { s } / 2 } \widehat { P } _ { u } ( f ) \frac { \mathrm { R e } \{ a \overline { { b } } \} } { | D | ^ { 2 } } d f , } \end{array}
$$

in which the integrand arguments $( f )$ are omitted for brevity, the two unit-gain coordinate variances reduce to

$$
\begin{array} { r } { V _ { 1 , j } ( \phi ) = \displaystyle \frac { S _ { j } } { 2 } + \frac { \Delta _ { j } } { 2 } \cos { 2 \phi } - X _ { j } \sin { 2 \phi } , } \\ { V _ { 2 , j } ( \phi ) = \displaystyle \frac { S _ { j } } { 2 } - \frac { \Delta _ { j } } { 2 } \cos { 2 \phi } + X _ { j } \sin { 2 \phi } . } \end{array}
$$

The variances are equal whenever the 2ϕ-sinusoid vanishes, and a deterministic choice is

$$
\phi _ { j } = \frac 1 2 \left( \mathrm { a t a n 2 } ( - \Delta _ { j } , - 2 X _ { j } \right) \bmod \pi ) .
$$

For this direction, $V _ { 1 , j } ( \phi _ { j } ) = V _ { 2 , j } ( \phi _ { j } ) = S _ { j } / 2 ;$ the unit-gain predicted variance of Section III-A3c is therefore $\kappa _ { j } = S _ { j } / 2 .$ and the gain

$$
g _ { j } = \sigma _ { * } \sqrt { \frac { 2 } { S _ { j } } } = \frac { \sigma _ { * } } { \sqrt { \kappa _ { j } } }
$$

gives both predicted coordinate variances the target value $\sigma _ { * } ^ { 2 }$

The derivation requires $0 < r _ { j } < 1$ , an integrable nonnegative $\widehat { P } _ { u } , D ( f ) \neq 0$ on the integration interval, and $S _ { j } > 0 ,$ with $\sigma _ { * } > 0$ . If $\Delta _ { j } = X _ { j } = 0$ , every direction equalizes the variances. The scaling concerns only the predicted variances of the state coordinates before the mixing by Q, and does not imply whitening of the realized features.

## APPENDIX B EVALUATION COUNTS AND VALIDATION SEARCH PROGRESS

This appendix reports the evaluation count at which the searches of each search configuration and system–task case stopped under the stopping rule of Section III-B2 (Table 6), and the optimization histories of the searches (Figs. 8 and 9). Because the ten model seeds of a case stop at the same count, each entry of Table 6 is a single count. All 48 configuration– case pairs met the stopping rule before reaching the budget of 8,192 evaluations. Forty-four pairs stopped at 3,072 evaluations, the first count at which the rule is checked, and four at 4,096. The four were the uniform random design on Lorenz x → x at horizon 1 and on Rössler x → x at horizons 1 and 12, and the γ-normalized design on Rössler x → z at horizon 12.

Fig. 8 shows, for each case, the median over the ten model seeds of the best validation NRMSE found so far, $E ^ { * } ( n )$ of Section III-B2, as a function of the number n of evaluated candidates, with the interquartile range shown as a band, for the six search configurations. The median best validation NRMSE decreased early in the searches, and later improvements were smaller. The stopping statistic of Section III-B2 is the median over seeds of the per-seed relative improvement $\delta ( n )$ , which differs from the relative improvement of the median curve in Fig. 8. For example, for the SRLR with the PSD-scaled design on Rössler x → x at horizon 1, the median curve decreased by 3.61% between 2,048 and 3,072 evaluations, whereas the median of the per-seed improvements was 0.488%, below the threshold of 0.5%.

Fig. 9 shows the same progression for the SRLR with the PSD-scaled design, separately for each model seed. Within a case, the ratio of the highest to the lowest final best validation NRMSE across the ten seeds ranged from 1.16 to 1.86 across the eight cases, and the rate of improvement also varied across seeds. The selected profiles likewise differed across seeds (Section IV-C).

TABLE 6. Final evaluation counts of the validation searches for each search configuration and system–task case. Each entry is the count shared by the ten model seeds of that case. All 48 configuration–case pairs met the stopping rule of Section III-B2 before reaching the budget of 8,192 evaluations.
<table><tr><td>Task</td><td>ESN</td><td>LI-ESN</td><td>SRLR (PSD-scaled)</td><td>Common-radius reference</td><td>SRLR (γ-normalized)</td><td>SRLR (uniform random)</td></tr><tr><td>Lorenz</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>x → x, horizon 1</td><td>3,072</td><td>3,072</td><td>3,072</td><td>3,072</td><td>3,072</td><td>4,096</td></tr><tr><td>x → x, horizon 12</td><td>3,072</td><td>3,072</td><td>3,072</td><td>3,072</td><td>3,072</td><td>3,072</td></tr><tr><td>x → z, horizon 0</td><td>3,072</td><td>3,072</td><td>3,072</td><td>3,072</td><td>3,072</td><td>3,072</td></tr><tr><td>x → z, horizon 12</td><td>3,072</td><td>3,072</td><td>3,072</td><td>3,072</td><td>3,072</td><td>3,072</td></tr><tr><td>Rössler</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>x → x, horizon 1</td><td>3,072</td><td>3,072</td><td>3,072</td><td>3,072</td><td>3,072</td><td>4,096</td></tr><tr><td>x → x, horizon 12</td><td>3,072</td><td>3,072</td><td>3,072</td><td>3,072</td><td>3,072</td><td>4,096</td></tr><tr><td>x → z, horizon 0</td><td>3,072</td><td>3,072</td><td>3,072</td><td>3,072</td><td>3,072</td><td>3,072</td></tr><tr><td>x → z, horizon 12</td><td>3,072</td><td>3,072</td><td>3,072</td><td>3,072</td><td>4,096</td><td>3,072</td></tr></table>

(a) Lorenz  
![](images/c578fc274bd7a76f6b8e7d361fca6b79e1d9dc41b1ad6d5ea3d1b0926a686bdb.jpg)  
(b) Rössler

![](images/47a4329613f57c47095f86346542fb5663552c3ab8fa30564fc913d94e347298.jpg)

![](images/1a131f3cc3adb1a28eb2d1f8d0ccc5d5c944ce28dc4a75f6b117a1d647a3dff3.jpg)

![](images/2f81b1c0416efd0434b6ff4785d98130ddfe84c948c78665ca45a631da33908b.jpg)

![](images/83c5ef3ac4ae84e84dd8e38ca8f36d55be293cbd0c4870b89cd7a3b12add72a8.jpg)

![](images/879b6f48df9cfd76333375e01d4aa8e2cb3dc889bfa77dd26eb1ccbd347c43b9.jpg)

![](images/990a4f74e5be7e0a5ffc63f285de0b130a748bc326e4d6b842a88ab733c4ba6c.jpg)

![](images/52e7d3f21e57e8344e6b1b4e8e1b7429a29d915037786869b720be4ea0e0c4a7.jpg)  
ESN LI-ESN Common-radius reference SRLR (PSD-scaled) SRLR (γ-normalized) SRLR (uniform random)

FIGURE 8. Best validation NRMSE found so far versus the number of evaluated candidates for the six search configurations on the eight system–task cases ((a) Lorenz, (b) Rössler). Lines show the median across the ten model seeds, and bands span the interquartile range. Each curve ends at the evaluation count in Table 6. The vertical axis is truncated at the top so that the later part of the searches is visible, and the first evaluations of each curve lie above the plotted range.  
(a) Lorenz  
![](images/d04d3ed56d6970d3879bf1f435fd2b0b66ec576a1a63d5b8afe2ecb0a877fb2e.jpg)

![](images/dd9bccda344d70653162e6cc8713cf6b582fd032a461a7a9a492349f2d95b121.jpg)

![](images/9048a96f4e7125375a280432a33ac3d68c15a98e669f03a0f09541ef2e98c301.jpg)  
(b) Rössler

![](images/db3bc89b666d66bfcff1099f4a4a3b87dd297024f5b2b614648ccf4ec86c1378.jpg)

![](images/bfd6bbcdf55785e03eeb817628d05c6fcf3a179633aae3dbbcb7d547780a4bc6.jpg)

![](images/3b8914ed5068fe0842cd340d5ae6764c298ea75f2fee482719c1791038baad84.jpg)

![](images/b4d769dd691e61387b9ca6854a3e35e3fa892b544ee7b2facb89d2713c3abf1b.jpg)

![](images/5eead985611793cdb7c852a1620a4490225eb57c3e508a99d471f898c1343c1b.jpg)  
Seed 0 Seed 1 Seed 2 Seed 3 Seed 4 Seed 5 Seed 6 Seed 7 Seed 8 Seed 9  
FIGURE 9. Best validation NRMSE found so far versus the number of evaluated candidates for the SRLR with the PSD-scaled design, shown separately for the ten model seeds ((a) Lorenz, (b) Rössler). Each line ends at the evaluation count of its case in Table 6, and a marker shows the final value of each line. The vertical axis is truncated at the top as in Fig. 8.

## ACKNOWLEDGMENT

The authors used Claude Fable 5.1 (Anthropic) through Claude Code and GPT-5.6 Sol and GPT-6 Astra (OpenAI) through Codex to assist in drafting and editing the text, in typesetting, and in writing the experiment and figure-plotting code. All content was reviewed and verified by the authors, who take full responsibility for it.

## REFERENCES

[1] H. Jaeger, ‘‘The ‘echo state’ approach to analysing and training recurrent neural networks,’’ German National Research Center for Information Technology, Tech. Rep. 148, 2001.

[2] W. Maass, T. Natschläger, and H. Markram, ‘‘Real-time computing without stable states: A new framework for neural computation based on perturba tions,’’ Neural Computation, vol. 14, no. 11, pp. 2531–2560, 2002.

[3] M. Lukoševičius and H. Jaeger, ‘‘Reservoir computing approaches to recurrent neural network training,’’ Computer Science Review, vol. 3, no. 3, pp. 127–149, 2009.

[4] H. Jaeger and H. Haas, ‘‘Harnessing nonlinearity: Predicting chaotic systems and saving energy in wireless communication,’’ Science, vol. 304, no. 5667, pp. 78–80, 2004.

[5] J. Pathak, B. Hunt, M. Girvan, Z. Lu, and E. Ott, ‘‘Model-free prediction of large spatiotemporally chaotic systems from data: A reservoir computing approach,’’ Physical Review Letters, vol. 120, no. 2, 2018, Art. no. 024102.

[6] G. Tanaka, T. Yamane, J. B. Héroux, R. Nakane, N. Kanazawa, S. Takeda, H. Numata, D. Nakano, and A. Hirose, ‘‘Recent advances in physical reservoir computing: A review,’’ Neural Networks, vol. 115, pp. 100–123, 2019.

[7] M. Lukoševičius, ‘‘A practical guide to applying echo state networks,’’ in Neural Networks: Tricks of the Trade, ser. Lecture Notes in Computer Science, G. Montavon, G. B. Orr, and K.-R. Müller, Eds. Berlin, Heidelberg: Springer, 2012, vol. 7700, pp. 659–686.

[8] T. L. Carroll, ‘‘Optimizing memory in reservoir computers,’’ Chaos: An Interdisciplinary Journal of Nonlinear Science, vol. 32, no. 2, 2022, Art. no. 023123.

[9] H. Jaeger, M. Lukoševičius, D. Popovici, and U. Siewert, ‘‘Optimization and applications of echo state networks with leaky-integrator neurons,’ Neural Networks, vol. 20, no. 3, pp. 335–352, 2007.

[10] F. Wyffels, B. Schrauwen, D. Verstraeten, and D. Stroobandt, ‘‘Band-pass reservoir computing,’’ in IEEE International Joint Conference on Neural Networks (IJCNN), 2008, pp. 3204–3209.

[11] G. Tanaka, T. Matsumori, H. Yoshida, and K. Aihara, ‘‘Reservoir computing with diverse timescales for prediction of multiscale dynamics,’’ Physical Review Research, vol. 4, no. 3, 2022, Art. no. L032014.

[12] M. Lukoševičius and A. Uselis, ‘‘Efficient implementations of echo state network cross-validation,’’ Cognitive Computation, vol. 15, no. 5, pp. 1470–1484, 2023.

[13] A. V. Oppenheim, R. W. Schafer, and J. R. Buck, Discrete-Time Signal Processing, 2nd ed. Upper Saddle River, NJ: Prentice Hall, 1999.

[14] C. Nathe, E. Del Frate, T. L. Carroll, L. M. Pecora, A. Shirin, and F. Sorrentino, ‘‘Reservoir computers modal decomposition and optimization,’’ 2021, arXiv:2101.07219.

[15] R. S. Fong, B. Li, and P. Tiňo, ‘‘Linear simple cycle reservoirs at the edge of stability perform Fourier decomposition of the input driving signals,’’ Chaos: An Interdisciplinary Journal ofNonlinear Science, vol. 35, no. 4, 2025, Art. no. 043109.

[16] G. E. Blelloch, ‘‘Prefix sums and their applications,’’ School of Computer Science, Carnegie Mellon University, Tech. Rep. CMU-CS-90-190, 1990.

[17] E. Martin and C. Cundy, ‘‘Parallelizing linear recurrent neural nets over sequence length,’’ in International Conference on Learning Representations, 2018.

[18] R. de Coudenhove, Y. Bendi-Ouis, A. Strock, and X. Hinaut, ‘‘Fast linear reservoirs via diagonalization,’’ 2026, arXiv:2602.19802.

[19] M. Pinna, G. Lagomarsini, A. Ceni, and C. Gallicchio, ‘‘ParalESN: Enabling parallel information processing in reservoir computing,’’ in Proceedings of the 43rd International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 306. PMLR, 2026, pp. 98 863–98 883.

[20] S. Tangerami, N. A. Mecholsky, and F. Sorrentino, ‘‘Optimizing the network topology of a linear reservoir computer,’’ Systems & Control Letters, vol. 214, 2026, Art. no. 106462.

[21] Y. Ebato, S. Nobukawa, Y. Sakemi, T. Kanamaru, N. Sviridova, and K. Aihara, ‘‘Optimizing reservoir computing performance via eigenvalue control in echo state networks,’’ in Proceedings of the 2025 International Symposium on Nonlinear Theory and Its Applications (NOLTA2025), Naha, Okinawa, Japan, Oct. 2025, pp. 336–339.

[22] L. Jaurigue and K. Lüdge, ‘‘Reducing reservoir computer hyperparameter dependence by external timescale tailoring,’’ Neuromorphic Computing and Engineering, vol. 4, no. 1, 2024, Art. no. 014001.

[23] J. Dambre, D. Verstraeten, B. Schrauwen, and S. Massar, ‘‘Information processing capacity of dynamical systems,’’ Scientific Reports, vol. 2, 2012, Art. no. 514.

[24] M. Inubushi and K. Yoshimura, ‘‘Reservoir computing beyond memorynonlinearity trade-off,’’ Scientific Reports, vol. 7, 2017, Art. no. 10199.

[25] Y. Sakemi, K. Morino, T. Leleu, and K. Aihara, ‘‘Model-size reduction for reservoir computing by concatenating internal states through time,’ Scientific Reports, vol. 10, 2020, Art. no. 21794.

[26] G. Holzmann and H. Hauser, ‘‘Echo state networks with filter neurons and a delay&sum readout,’’ Neural Networks, vol. 23, no. 2, pp. 244–256, 2010.

[27] S. Yoshida, T. Iinuma, S. Nobukawa, E. Watanabe, and T. Isokawa, ‘‘Multitimescale processing with heterogeneous assembly echo state networks,’’ in Neural Information Processing (ICONIP 2024), Part III, ser. Lecture Notes in Computer Science, vol. 15288. Singapore: Springer, 2025, pp. 156–168.

[28] L. Manneschi, M. O. A. Ellis, G. Gigante, A. C. Lin, P. Del Giudice, and E. Vasilaki, ‘‘Exploiting multiple timescales in hierarchical echo state networks,’’ Frontiers in Applied Mathematics and Statistics, vol. 6, 2021, Art. no. 616658.

[29] J. Beuria and A. Shukla, ‘‘Lindblad-inspired multi-timescale reservoir computing with separable rotation and dissipation,’’ 2026, arXiv:2608.04028.

[30] Z. Li, Y. Liu, and G. Tanaka, ‘‘Multi-reservoir echo state networks with Hodrick–Prescott filter for nonlinear time-series prediction,’’ Applied Soft Computing, vol. 135, 2023, Art. no. 110021.

[31] I. B. Yildiz, H. Jaeger, and S. J. Kiebel, ‘‘Re-visiting the echo state property,’’ Neural Networks, vol. 35, pp. 1–9, 2012.

[32] G. Manjunath and H. Jaeger, ‘‘Echo state property linked to an input: Exploring a fundamental characteristic of recurrent neural networks,’’ Neural Computation, vol. 25, no. 3, pp. 671–696, 2013.

[33] T. Hülser, F. Köster, K. Lüdge, and L. Jaurigue, ‘‘Deriving task specific performance from the information processing capacity of a reservoir computer,’’ Nanophotonics, vol. 12, no. 5, pp. 937–947, 2023.

[34] H. Jaeger, ‘‘Short term memory in echo state networks,’’ German National Research Center for Information Technology, Tech. Rep. 152, 2002.

[35] G. Ballarin, L. Grigoryeva, and J.-P. Ortega, ‘‘Memory capacity of nonlinear recurrent networks: Is it informative?’’ in Geometric Science of Information, ser. Lecture Notes in Computer Science, vol. 16035. Cham: Springer, 2026, pp. 53–64.

[36] Y. Ebato, S. Nobukawa, Y. Sakemi, H. Nishimura, T. Kanamaru, N. Sviri dova, and K. Aihara, ‘‘Impact of time-history terms on reservoir dynamics and prediction accuracy in echo state networks,’’ Scientific Reports, vol. 14, no. 1, 2024, Art. no. 8631.

[37] A. Rodan and P. Tiňo, ‘‘Minimum complexity echo state network,’’ IEEE Transactions on Neural Networks, vol. 22, no. 1, pp. 131–144, 2011.

[38] P. Tiňo, ‘‘Dynamical systems as temporal feature spaces,’’ Journal of Machine Learning Research, vol. 21, no. 44, pp. 1–42, 2020.

[39] L. Gonon and J.-P. Ortega, ‘‘Reservoir computing universality with stochastic inputs,’’ IEEE Transactions on Neural Networks and Learning Systems, vol. 31, no. 1, pp. 100–112, 2020.

[40] A. Orvieto, S. De, C. Gulcehre, R. Pascanu, and S. L. Smith, ‘‘Universality of linear recurrences followed by non-linear projections: Finite-width guarantees and benefits of complex eigenvalues,’’ in Proceedings of the 41st International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 235. PMLR, 2024, pp. 38 837–38 863.

[41] B. Li, R. S. Fong, and P. Tiňo, ‘‘Simple cycle reservoirs are universal,’ Journal of Machine Learning Research, vol. 25, no. 158, pp. 1–28, 2024.

[42] R. S. Fong, B. Li, and P. Tiňo, ‘‘Universality of real minimal complexity reservoir,’’ in Proceedings of the AAAI Conference on Artificial Intelli gence, vol. 39, no. 16, 2025, pp. 16 622–16 629.

[43] D. J. Gauthier, E. Bollt, A. Griffith, and W. A. S. Barbosa, ‘‘Next generation reservoir computing,’’ Nature Communications, vol. 12, 2021, Art. no. 5564.

[44] G. Lagomarsini, A. Ceni, and C. Gallicchio, ‘‘Benchmarking nonlinear readouts in linear reservoir networks,’’ in Artificial Neural Networks and Machine Learning. ICANN 2025 International Workshops and Specia

Sessions, ser. Lecture Notes in Computer Science, vol. 16072. Cham: Springer, 2026, pp. 176–187.

[45] A. Orvieto, S. L. Smith, A. Gu, A. Fernando, C. Gulcehre, R. Pascanu, and S. De, ‘‘Resurrecting recurrent neural networks for long sequences,’’ in Proceedings ofthe 40th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 202. PMLR, 2023, pp. 26 670–26 698.

[46] P. D. Welch, ‘‘The use of fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms,’’ IEEE Transactions on Audio and Electroacoustics, vol. 15, no. 2, pp. 70–73, 1967.

[47] E. N. Lorenz, ‘‘Deterministic nonperiodic flow,’’ Journal of the Atmospheric Sciences, vol. 20, no. 2, pp. 130–141, 1963.

[48] O. E. Rössler, ‘‘An equation for continuous chaos,’’ Physics Letters A, vol. 57, no. 5, pp. 397–398, 1976.

[49] J. S. Bergstra, R. Bardenet, Y. Bengio, and B. Kégl, ‘‘Algorithms for hyperparameter optimization,’’ in Advances in Neural Information Processing Systems, vol. 24, 2011, pp. 2546–2554.

[50] T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, ‘‘Optuna: A nextgeneration hyperparameter optimization framework,’’ in Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2019, pp. 2623–2631.

[51] P. V. Aceituno, G. Yan, and Y.-Y. Liu, ‘‘Tailoring echo state networks for optimal learning,’’ iScience, vol. 23, no. 9, 2020, Art. no. 101440.

[52] F. X. X. Yu, A. T. Suresh, K. M. Choromanski, D. N. Holtmann-Rice, and S. Kumar, ‘‘Orthogonal random features,’’ in Advances in Neural Information Processing Systems, vol. 29, 2016, pp. 1975–1983.

[53] G. Hennequin, T. P. Vogels, and W. Gerstner, ‘‘Non-normal amplification in random balanced neuronal networks,’’ Physical Review E, vol. 86, no. 1 2012, Art. no. 011909.

[54] G. Kerg, K. Goyette, M. Puelma Touzel, G. Gidel, E. Vorontsov, Y. Bengio, and G. Lajoie, ‘‘Non-normal recurrent neural network (nnRNN): Learning long time dependencies while improving expressivity with transient dynamics,’’ in Advances in Neural Information Processing Systems, vol. 32, 2019, pp. 13 613–13 623.