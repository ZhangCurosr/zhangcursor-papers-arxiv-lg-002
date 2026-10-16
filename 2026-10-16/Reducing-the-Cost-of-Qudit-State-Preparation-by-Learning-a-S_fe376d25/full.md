# Reducing the Cost of Qudit State Preparation by Learning a Sparse Structure for SNAP Gates

Marco Pennazio<sup>1</sup>, Leonardo Ignazio Pagliochini<sup>1</sup>, Riccardo Pellini<sup>1</sup>, Maurizio Ferrari Dacrema<sup>1</sup> and Paolo Cremonesi<sup>1</sup>

<sup>1</sup>Politecnico di Milano, Milano, Italy

## Abstract

Preparing quantum states in bosonic qudits with SNAP–Displacement sequences requires both optimizing the gate parameters and deciding which Fock levels should be controlled by each SNAP gate. A fully parameterized ansatz assigns an independent phase to every Fock level in every gate, leading to a number of phases that grows steeply. This constitutes a problem for short to medium term systems as controlling many phases requires complex control pulses which are subject to increased control errors. Previous work reduced this cost by restricting the trainable phases to predefined sparse patterns, such as a diagonal or a grid, which however may not be optimal for a specific target state. In this work, we address this limitation by introducing a method that learns which phases are needed together with their values, using a diferentiable mask. A sparsity penalty encourages the removal of unnecessary phases, while temperature annealing gradually turns the masks into binary decisions. We compare the proposed method with fully parameterized and predefined sparse ansatzes on Fourier and Haar-random target states. The evaluation considers both the minimum number of phases required to reach diferent infidelity thresholds and the overall trade-of between fidelity and phases required, measured through Pareto frontiers and hypervolume. The results show that learning the location of the phases together with their values provides competitive or improved fidelity–resource trade-ofs, although the advantage depends on the target state and qudit dimension.

## Keywords

Bosonic Quantum Computing, Qudit State Preparation, SNAP–Displacement Protocol, Sparse Ansatz

## 1. Introduction

Bosonic quantum computing encodes quantum information in superconducting radio-frequency cavities. These cavities can store diferent numbers of photons, with each photon-number level corresponding to a quantum state called a Fock state. While a qubit uses two levels, conventionally denoted by |0 and |1⟩, a cavity can use many levels. By selecting the first � Fock states |0⟩ . . . |� − 1⟩, a single cavity can implement a �-level quantum system, or qudit [1, 2, 3]. These � Fock states form a �-dimensional Hilbert space, which defines the computational space of the qudit. The potential advantages of encoding quantum information with qudits have been studied in many fields, including quantum simulation and quantum error correction. In particular, access to multiple levels can reduce the number of quantum devices required to represent the same information and, for some tasks, simplify the resulting circuits [4, 5, 6, 7]. While other technologies can also be used to create qudits, radio-frequency cavities ofer both a large number of accessible Fock states and long coherence times [1, 8, 9] with current implementations able to control the qudit up to � = 20.

Among the universal gate sets proposed for bosonic systems, a widely used one combines cavity displacement operations with Selective Number-dependent Arbitrary Phase (SNAP) gates [1, 10]. Displacements redistribute the population among the Fock levels, while SNAP gates apply independently controlled phases to the Fock levels. Alternating the two gates allows for universal control over the qudit and has been used for both state preparation and unitary synthesis [11, 12, 13, 8]. The gate parameters of a SNAP–Displacement sequence can be obtained through various methods: constructive compilation, numerical optimal control, or variational optimization [11, 13, 12]. Preparing a target state with high fidelity remains a dificult problem. Constructive methods can decompose the preparation into a sequence of gates, but the number of gates can increase rapidly with the qudit dimension and the required fidelity. Variational methods can use shorter sequences, but a fully parameterized ansatz assigns one independent phase to each Fock level in every SNAP gate resulting in �� independent phase parameters for � SNAP gates, which is a steep growth.

Besides increasing the time required to run the ansatz on the quantum device, a large number of phases creates an important drawback for practical applications. Controlling many Fock levels requires to engineer control pulses with several closely spaced frequency components, which can make SNAP gates longer and harder to calibrate [1, 12, 8], increasing the accumulation of control errors. Developing methods to identify the minimum number of phases required to prepare a state with the desired fidelity is therefore an active research area. Previous work addressed this problem by defining sparse SNAP–Displacement ansatzes in which only a subset of the phases can be used, according to heuristic patterns [14]. The Diagonal (adaptive), Diagonal (multiple), and Grid ansatzes were shown to reduce the number of active phases while maintaining high fidelity. However, the locations of the active phases are fixed before training and a pattern may retain phases that are not needed or exclude phases that would be useful for a specific target state.

In this work, we draw on diferentiable masking methods from machine learning to overcome this restriction and make the selection of SNAP phases adaptive. The proposed Learnable method starts from the predefined Diagonal (adaptive) ansatz and associates each of the active phases with a trainable mask coeficient. These mask coeficients are optimized jointly with the phase parameters. A regularization term encourages the removal of unnecessary phases, while temperature annealing makes the learned mask progressively closer to a binary selection. The Learnable method therefore learns which phases within the starting ansatz should be retained for a given target state, as well as their optimal values.

We compare the proposed Learnable ansatz with the Full, Diagonal (adaptive), Diagonal (multiple), and Grid ansatzes using the same target states, qudit dimensions, and simulation setting. We first compare the minimum number of SNAP phases required to reach diferent infidelity thresholds. We then examine the Pareto frontiers to determine where each ansatz provides the best trade-of between infidelity and phase count, and use hypervolume to summarize these frontiers over both the complete explored range and the region satisfying $I \leq 1 0 ^ { - 3 }$ . Overall, Learnable reduces the number of SNAP phases in several configurations and provides its clearest advantage for Fourier-5 at intermediate infidelities, while the results for Haar random states are more mixed.

## 2. Background

The state preparation problem consists in finding a sequence of gates that transforms an initial state, typically |0⟩, into a target state $| \psi _ { \mathrm { t a r g e t } } \rangle$ . The quality of a prepared state $| \psi _ { \mathrm { p r e p } } \rangle$ is measured through the fidelity $F = | \langle \psi _ { \mathrm { t a r g e t } } | \psi _ { \mathrm { p r e p } } \rangle | ^ { 2 }$ , or equivalently through the infidelity $I = 1 - F$ . A generic state of a �-dimensional qudit is a superposition of its � Fock states, $\textstyle | \psi \rangle = \sum _ { f = 0 } ^ { d - 1 } c _ { f } | f \rangle$ , where $c _ { f } \in \mathbb { C }$ and $\begin{array} { r } { \sum _ { f = 0 } ^ { d - 1 } | c _ { f } | ^ { 2 } = 1 } \end{array}$ . In this paper we focus in particular on quantum states defined over a single qudit.

## 2.1. The SNAP–Displacement Protocol

The SNAP–Displacement protocol is a widely used approach for state preparation and unitary synthesis. It alternates displacement and SNAP gates to control the amplitudes and phases of the Fock states [1, 10]. A displacement gate changes the population distribution � across the Fock levels and is defined as:

$$
D ( \alpha ) = \exp \left( \alpha \hat { a } ^ { \dagger } - \alpha ^ { * } \hat { a } \right) ,\tag{1}
$$

where $\alpha \in \mathbb { C }$ is the displacement parameter. The operator �^ is called annihilation operator and removes one photon from the cavity, acting as ${ \hat { a } } | f \rangle = { \sqrt { f } } | f - 1 \rangle$ and $\hat { a } | 0 \rangle = 0$ , while the operator $\hat { a } ^ { \dagger }$ is called creation operator and adds one photon $\hat { a } ^ { \dagger } | f \rangle = \sqrt { f + 1 } | f + 1 \rangle$

A SNAP gate applies an independent phase to each Fock state:

$$
S ( \vec { \theta } ) = \sum _ { f = 0 } ^ { \infty } e ^ { i \theta _ { f } } \vert f \rangle \langle f \vert ,\tag{2}
$$

where $\vec { \theta } = ( \theta _ { 0 } , \theta _ { 1 } , \ldots )$ and $\theta _ { f } \in$ R is the phase applied to Fock level $f .$

Note that the Hilbert space of the cavity is infinite, while the hardware can directly control only a finite number of Fock levels, which correspond to the dimension � of the qudit. However, displacement gates can also populate higher levels beyond what the hardware can control, and numerical simulations must take this efect into account to accurately represent the physical state of the cavity. This leakage is typically undesirable because it moves part of the population outside the intended qudit, reducing the preparation fidelity. The simulations therefore include a finite number of additional levels above the first $d ,$ called bumper states, which are not part of the qudit, but allow the simulation to account for population leaking outside it.

The SNAP–Displacement protocol consists of an alternating sequence of � SNAP gates and $B + 1$ displacement gates, and can be written as:

$$
U ( \vec { \alpha } , \Theta ) = D ( \alpha _ { B } ) S ( \vec { \theta } _ { B - 1 } ) D ( \alpha _ { B - 1 } ) \cdot \cdot \cdot S ( \vec { \theta _ { 0 } } ) D ( \alpha _ { 0 } ) ,\tag{3}
$$

where $\vec { \alpha } = ( \alpha _ { 0 } , \ldots , \alpha _ { B } )$ contains the displacement parameters. Each vector $ { \vec { \theta _ { b } } }$ contains the phase angles applied by the SNAP gate at block $b ,$ while $\Theta = \bar { [ \theta _ { b , f } ] } \in \mathbb { R } ^ { B \times d }$ collects the phases applied to the first � Fock levels across the whole protocol. The phases of the bumper states are set to zero. Starting from |0⟩, the state prepared by the protocol is $| \psi _ { \mathrm { p r e p } } \rangle = U ( \vec { \alpha } , \Theta ) | 0 \rangle$

In the fully parameterized ansatz, every entry of Θ is independently optimized. The protocol therefore contains �� SNAP phases and $B + 1$ complex displacement parameters. Since this work studies how to reduce the resources required by SNAP gates, we use the number of active, i.e., non-zero, SNAP phases as the main measure of the resources required.

The parameters of a SNAP–Displacement sequence can be determined through constructive compilation, numerical optimal control, or variational optimization [11, 13, 12]. In this work, we follow the variational approach and optimize the displacement parameters and SNAP phases to maximize the fidelity with the target state.

## 2.2. Predefined Sparse Phase Ansatzes

Previous work proposed reducing the number of SNAP phases by optimizing only those in specific locations [14]. The proposed patterns were motivated by the observation that, starting from |0⟩, displacement gates tend to populate higher Fock levels progressively as the protocol advances. Therefore, phases applied to high Fock levels during the first blocks may have little or no efect because those levels have negligible population.

The SNAP phases can be arranged as a matrix $\Theta = [ \theta _ { b , f } ] \in \mathbb { R } ^ { B \times d }$ , where the rows correspond to the protocol blocks and the columns to the Fock levels. A fixed binary structural mask $M ^ { \mathrm { s t r u c t } } \in \{ 0 , 1 \}$ �×� controls which SNAP phases can be optimized, with the actual unitary used for state preparation being $U ( \vec { \alpha } , \widetilde { \Theta } )$ , with $\widetilde { \Theta } = M ^ { \mathrm { s t r u c t } } \odot \Theta$ , where ⊙ denotes element-wise multiplication. Entries with $M _ { b , f } ^ { \mathrm { s t r u c t } } = 1$ are learnable, while those with $M _ { b , f } ^ { \mathrm { s t r u c t } } = 0$ are fixed to zero.

The following heuristic structures were considered in [14]:

<sup>•</sup> Full: all phases are learnable, so that $M _ { b , f } ^ { \mathrm { s t r u c t } } = 1$ for every block and Fock level.

<sup>•</sup> Diagonal (adaptive): the learnable phases lie within a band around the diagonal connecting the first and last Fock levels across the protocol. The diagonal follows $\begin{array} { r } { f = \frac { d - 1 } { B - 1 } b . } \end{array}$ , while a width hyperparameter determines the size of the band.

<sup>•</sup> Diagonal (multiple): several parallel diagonal bands are used. Their number and width are set as hyperparameters.

<sup>•</sup> Grid: diagonal and anti-diagonal bands are combined, producing a grid of learnable phase locations. The number and width of the bands are set as hyperparameters.

These structures reduce the number of learnable phases, but the structural mask is fixed once the heuristic and its hyperparameters have been selected. The gate parameter optimization determines the values of the learnable phases and may set some of them to zero, but it cannot change which entries of Θ can be optimized. Our Learnable method which we introduce in the next section starts from the structural mask of the Diagonal (adaptive) ansatz and learns which phases within this mask should be retained.

## 3. Learnable Phase Locations

The heuristic ansatzes described in Section 2.2 determine which SNAP phases are learnable before training. Our goal is to make this selection adaptive, so that the ansatz can retain only the phases that are useful for the target state rather than relying entirely on a predefined pattern. In our proposed Learnable method, we start from the mask defined by the Diagonal (adaptive) structure and introduce a second mask whose values are learned together with the gate parameters.

We refer to the mask defined by the Diagonal (adaptive) heuristic as the structural mask, $M ^ { \mathrm { s t r u c t } } \in \{ 0 , 1 \} ^ { B \times d }$ . This mask limits the phase locations considered by the method. A second mask, referred to as the learned mask, $M ^ { \mathrm { l e a r n } } \in [ 0 , 1 ] ^ { B \times d }$ , determines which of these phases are retained. The two masks are combined through element-wise multiplication.

## 3.1. Diferentiable Learned Mask

A binary mask cannot be optimized directly using gradient-based methods, since changing one of its entries from zero to one is a discrete operation. We therefore use a continuous soft mask during training and convert it into a binary mask only after the optimization is complete. For each entry for which $M _ { b , f } ^ { \mathrm { s t r u c t } } = 1$ , we introduce a trainable logit $z _ { b , f }$ , which is an unconstrained real valued parameter that is mapped to the interval [0, 1] through a sigmoid function. The corresponding entry of the learned mask is defined as

$$
M _ { b , f } ^ { \mathrm { l e a r n } } = \sigma \left( \frac { z _ { b , f } } { \tau } \right) ,\tag{4}
$$

where � is the sigmoid function and $\tau > 0$ is a temperature parameter. Positive values of $z _ { b , f }$ favour retaining the phase, while negative values favour suppressing it. Similarly, the value of $M _ { b , f } ^ { \mathrm { l e a r n } }$ controls how much of the corresponding phase is retained, with values close to one leaving the phase almost unchanged, and values close to zero suppressing it. The structural and learned masks are combined into the efective mask

$$
M ^ { \mathrm { e f f } } = M ^ { \mathrm { s t r u c t } } \odot M ^ { \mathrm { l e a r n } } ,\tag{5}
$$

The efective protocol is therefore $U ( \vec { \alpha } , \widetilde { \Theta } )$ where the SNAP phases used are

$$
\widetilde { \Theta } = M ^ { \mathrm { e f f } } \odot \Theta .\tag{6}
$$

If $M _ { b , f } ^ { \mathrm { s t r u c t } } = 0$ , the corresponding phase remains fixed to zero. If $M _ { b , f } ^ { \mathrm { s t r u c t } } = 1$ , its contribution is controlled by the learned mask.

Since the soft-mask values depend continuously on the logits, the logits can be optimized together with the gate parameters using gradient-based methods. The logits are initialized to $z _ { b , f } = 6$ . With the initial temperature $\tau _ { \mathrm { s t a r t } } = 1$ , this gives learned-mask values close to one, so the optimization starts from almost the complete structural mask.

During training, the temperature is reduced according to a geometric annealing schedule. At training step �, the temperature is

$$
\tau _ { t } = \tau _ { \mathrm { s t a r t } } \left( \frac { \tau _ { \mathrm { e n d } } } { \tau _ { \mathrm { s t a r t } } } \right) ^ { q _ { t } } , \qquad q _ { t } = \mathrm { m i n } \left( 1 , \frac { t } { N _ { \mathrm { a n n e a l } } } \right) ,\tag{7}
$$

where $N _ { \mathrm { a n n e a l } }$ is the number of steps over which annealing is performed. In our experiments, the temperature decreases from $\tau _ { \mathrm { s t a r t } } = 1 \ \mathrm { t o } \ \tau _ { \mathrm { e n d } } = 0 . 0 5$ over 10 000 training steps and remains fixed afterward. At the beginning of training, the higher temperature gives a smoother dependence of the mask on the logits. As the temperature decreases, the sigmoid becomes steeper and the mask values move closer to zero or one. This avoids making sharp decisions before the phase and displacement parameters have had time to adapt. Annealing sharpens the mask values, while the joint objective introduced in Section 3.2 determines whether retaining or suppressing each phase is preferred.

At the end of training, the learned soft mask is converted into a binary hard mask using a threshold �:

$$
M _ { b , f } ^ { \mathrm { b i n } } = \left\{ \begin{array} { l l } { 1 , } & { M _ { b , f } ^ { \mathrm { l e a r n } } \ge \gamma , } \\ { 0 , } & { M _ { b , f } ^ { \mathrm { l e a r n } } < \gamma . } \end{array} \right.\tag{8}
$$

In our experiments, we use $\gamma = 1 0 ^ { - 3 }$ . The final binary efective mask is

$$
M ^ { \mathrm { e f f , b i n } } = M ^ { \mathrm { s t r u c t } } \odot M ^ { \mathrm { b i n } } \in \{ 0 , 1 \} ^ { B \times d } .\tag{9}
$$

The entries for which $M _ { b , f } ^ { \mathrm { e f f , b i n } } = 1$ identify the phases retained by the final mask. In order to mitigate the risk that the mask hardening process pushes the gate parameters in a suboptimal region, a short fine tuning phase of 200 steps is carried out using a learning rate reduced by a factor of ten. The final gate parameters are those used to compute the fidelity between the prepared state and the target one.

The diferentiable mask defines how phases can be retained or suppressed, but it does not by itself determine which choice is preferred. This is controlled by the joint optimization objective, which balances the fidelity of the prepared state with the number of the retained mask entries.

## 3.2. Joint Parameter Optimization

The displacement parameters, SNAP phases, and learned mask logits are optimized jointly. Following Ferrari Dacrema et al. [14], the gate parameters are obtained from unconstrained real-valued variables, which can be updated using gradient-based optimization. Each displacement gate is associated to a complex valued parameter $\alpha _ { b }$ which is represented in polar coordinates using two real parameters $r _ { b } , s _ { b } \in \mathbb { R }$

$$
\alpha _ { b } = \beta _ { b } e ^ { i \varphi _ { b } } , \qquad \beta _ { b } = \mathrm { R e L U } ( r _ { b } ) , \qquad \varphi _ { b } = \pi \operatorname { t a n h } ( s _ { b } ) .\tag{10}
$$

This mapping ensures that the displacement magnitude is non-negative and that its phase lies in $[ - \pi , \pi ]$ Similarly, each SNAP phase selected by the structural mask is obtained from an unconstrained parameter $t _ { b , f } \in \mathbb { R }$ as

$$
\theta _ { b , f } = \pi \operatorname { t a n h } ( t _ { b , f } ) ,\tag{11}
$$

which restricts the phase to $[ - \pi , \pi ]$ . The logits $z _ { b , f }$ introduced in Section 3.1 are optimized together with these gate parameters.

The displacement parameters, SNAP phases, and mask logits are optimized by maximizing the fidelity of the prepared state while penalizing large gate parameters and the number of phases retained by the mask. The complete objective is

$$
{ \mathcal { T } } = F \left( { \vec { \alpha } } , { \widetilde { \Theta } } \right) - R _ { \mathrm { g a t e } } - R _ { \mathrm { m a s k } } .\tag{12}
$$

The first regularization term is applied to the gate parameters:

$$
R _ { \mathrm { g a t e } } = \lambda _ { 1 } \left( \lVert \vec { \alpha } \rVert _ { 1 } + \lVert \widetilde { \Theta } \rVert _ { 1 } \right) + \lambda _ { 2 } \left( \lVert \vec { \alpha } \rVert _ { 2 } ^ { 2 } + \lVert \widetilde { \Theta } \rVert _ { 2 } ^ { 2 } \right) .\tag{13}
$$

The $\ell _ { 1 }$ term encourages some gate parameters to become zero, while the $\ell _ { 2 }$ term limits large phase values and displacement amplitudes. We aim to set as many SNAP phase angles as possible to zero, as well as limiting large displacements because they can increase leakage to higher Fock levels and sensitivity to photon loss, which is a dominant noise mechanism in bosonic cavities. The weights $\lambda _ { 1 }$ and $\lambda _ { 2 }$ control the gate regularization. In the experiments for our Learnable method both are set to zero, so sparsity is controlled only through the learned mask. For the other baseline ansatzes, their values are instead searched with hyperparameter optimization as described in Section 4.5. Note that the regularization is applied to the gate parameters, rather than to the unconstrained variables from which they are computed (see Eq. 10 and 11), because only the gate parameters represent the physical operations that need to be performed.

The second regularization term controls the learned mask:

$$
R _ { \mathrm { m a s k } } = \lambda _ { \mathrm { m a s k } } \sum _ { b , f } M _ { b , f } ^ { \mathrm { e f f } } + \lambda _ { \mathrm { e n t } } \sum _ { b , f } H \left( M _ { b , f } ^ { \mathrm { e f f } } \right) ,\tag{14}
$$

where $H ( m ) = -$ � log $m - ( 1 - m ) \log ( 1 - m )$ is the binary entropy, using the convention 0 log $0 = 0$ so that $H ( 0 ) = H ( 1 ) = 0$

Since the efective mask is continuous during training, the sum $\sum _ { b , f } M _ { b , f } ^ { \mathrm { e f f } }$ provides a diferentiable approximation of the number of phases retained by the mask. Minimizing this term favours mask values close to zero, while the fidelity term prevents phases that are needed for state preparation from being removed. The weight $\lambda _ { \mathrm { m a s k } }$ therefore controls the trade-of between fidelity and the number of retained phases. The entropy term has a diferent role. Binary entropy is zero when a mask value is zero or one and is largest when it is equal to 0.5. Penalizing this term therefore discourages intermediate values that do not favor retention or suppression. Together, the two mask terms encourage unnecessary phases to move towards zero and useful phases to remain close to one. The temperature schedule introduced in Section 3.1 makes this separation progressively sharper. Note that the mask regularization and the $\ell _ { 1 }$ gate regularization act on diferent aspects of the ansatz. The mask regularization determines whether a phase remains learnable in the final ansatz, whereas the $\ell _ { 1 }$ term acts on the numerical value of a phase and may set it to zero even when it is retained by the mask.

The unconstrained displacement variables, SNAP phase variables, and mask logits are updated jointly using Adam. The regularization weights $\lambda _ { 1 } , \lambda _ { 2 } , \lambda _ { \mathrm { { m a s k } } }$ , and $\lambda _ { \mathrm { e n t } }$ remain fixed during an individual training run and are selected through the hyperparameter optimization procedure described in Sections 4.4 and 4.5.

After training, the hard efective mask from Equation 9 is applied. Entries equal to one identify the phases retained by the mask, while entries equal to zero fix the corresponding phases to zero. The gate parameters are then optimized for a further 200 steps, as described in Section 4.3. Greedy refinement and structural compression are subsequently applied as described in Section 4.4.

## 4. Methodology

## 4.1. Experimental Setting

We evaluate our proposed Learnable method on the same target states considered by [14]: Fourier-5 and Haar random states defined for qudit dimensions $d \in \{ 8 , 1 6 , 3 2 \}$ . The SNAP–Displacement protocol starts from the vacuum state |0⟩. The Fourier-5 states are obtained by applying the �-dimensional Quantum Fourier Transform to the Fock state |5⟩. The resulting state has equal amplitudes over the � Fock levels and phases that vary regularly across them. The Haar random target states are complex pure states sampled according to the Haar measure.

Although the target states are defined over the first � Fock levels, as we previously explained displacement operations can populate levels higher than �. The numerical simulations are performed in an extended Hilbert space containing $n _ { \mathrm { b u m p e r } } = \operatorname* { m a x } ( 6 4 , 2 d )$ bumper states in addition to the � levels of the qudit. The total dimension used in the simulations is ${ \widetilde { d } } = d + n _ { \mathrm { b u m p e r } }$ . The target state is represented in the extended Hilbert space by assigning zero amplitude to all bumper states. Fidelity is evaluated over the complete space, so population leaking to the bumper states reduces the overlap with the target state.

We perform all simulations in double precision. The number of phases required by each method is evaluated at the target infidelity thresholds $1 0 ^ { - 1 } , 1 0 ^ { - 2 } , 1 0 ^ { - 3 } , 1 0 ^ { - \hat { 4 } }$ , and $1 0 ^ { - 5 }$

## 4.2. Evaluation Protocol

Our goal is to determine whether the Learnable ansatz can reach a given prepared state fidelity using fewer SNAP phases than the heuristic ansatzes, and whether this advantage remains consistent across a wide range of infidelity thresholds. The first stage is to identify the optimal hyperparameter configuration, as we describe in Sections 4.4 and 4.5. Since a small number of random initializations can produce large outliers with very poor fidelity, we repeat the training over many random initializations and define the infidelity � of each configuration as the mean infidelity over the 75% of repetitions with the highest fidelity. Similarly, the number of phases associated to that hyperparameter configuration is the mean across the selected repetitions. These value are used throughout the analysis. We first compare the ansatzes at a set of target infidelities. For each target state, qudit dimension, and threshold �, we consider all hyperparameter configurations satisfying $I \leq \epsilon$ and report the minimum number of SNAP phases among them. This analysis directly measures the number of phases required to reach a given preparation accuracy.

A reduction in the number of phases does not necessarily produce the same reduction in the physical duration of the ansatz, since the duration also depends on how the phases are distributed among the SNAP gates. We therefore report its estimated duration. Following Bornman et al. [15], the duration of a SNAP gate in block � is estimated as

$$
T _ { S _ { b } } \approx 1 0 \sqrt { p ( \theta _ { b } ) } \mu \mathrm { s } ,\tag{15}
$$

where $p ( \theta _ { b } )$ is the number of non-zero phase angles in that SNAP gate. The duration of a Displacement gate is much smaller and can be safely ignored. The ansatz duration is computed by adding the duration of all the SNAP gates in the ansatz. The duration associated to a hyperparameter configuration is the mean duration of the selected repetitions.

The analysis at fixed infidelity thresholds provides a direct comparison between ansatzes but only at those predefined thresholds. To provide a more complete picture, we also examine the Pareto frontiers. Each configuration is represented by a point $( p , I )$ , where $p$ is the number of SNAP phases and � is the aggregated infidelity. Both quantities should be minimized, so points closer to the lower-left region represent better trade-ofs. A point belongs to the Pareto frontier when no other evaluated configuration has both a lower phase count and a lower infidelity, with at least one strict improvement. The frontier contains the configurations for which improving one objective requires worsening the other. It therefore shows the regions in which each ansatz provides the best balance between accuracy and phase count.

While the Pareto frontiers show where these diferences occur and are useful for visualization, comparing several frontiers across target states and qudit dimensions is not straightforward. We use the hypervolume indicator (HV) to summarize each frontier through a single value [16, 17]. To compute the HV, a reference point is selected whose infidelity and number of phases are individually worse than those of all the evaluated configurations. Each point on the frontier defines a rectangular region extending towards this reference point, and the hypervolume is the area covered by the union of these regions. A larger hypervolume indicates a better trade-of, since it corresponds to lower infidelities, fewer phases, or both. For readability, the hypervolume values reported in the tables are multiplied by 100.

We compute the hypervolume in two settings. The first includes the complete range of infidelities explored during the experiments and is directly comparable with [14]. The second includes only configurations satisfying $I \leq 1 0 ^ { - 3 }$ and focuses on the region where the Pareto frontiers show the main advantage of the Learnable ansatz. The normalization and reference point are computed independently

for the two settings. Their absolute hypervolume values are therefore not compared across settings;   
comparisons are made only between ansatzes within the same setting and qudit dimension.

To quantify uncertainty, the reference point for each setting is computed using all repetitions and is then kept fixed during bootstrap resampling. For each hyperparameter configuration, the independent repetitions are sampled with replacement, and the mean infidelity is recomputed using the 75% of repetitions with the highest fidelity. The Pareto frontier and its hypervolume are then recomputed. We repeat this procedure 1000 times and report the mean hypervolume and its 95% confidence interval.

## 4.3. Training Protocol

For each ansatz and hyperparameter configuration, the gate parameters are optimized through gradient ascent using Adam. In the Learnable ansatz, the mask logits are optimized jointly with the displacement parameters and SNAP phases according to the objective defined in Section 3.2. The learning rate and regularization weights remain fixed within each training run. Their values are either fixed as part of the experimental setting or selected through the hyperparameter optimization procedures described in Sections 4.4 and 4.5.

Each training run is limited to $1 0 ^ { 4 }$ optimization steps. Early stopping is applied when the best fidelity does not improve by at least 10<sup>−6</sup> for $1 0 ^ { 3 }$ consecutive steps. Reaching the target infidelity does not terminate the training run, since the threshold is used by the outer hyperparameter optimization to evaluate the final configuration. Since the optimization is sensitive to the initial gate parameters, each configuration is trained from multiple independent random initializations, also referred to as multistart training. Each baseline configuration is evaluated using 100 initializations, following [14], while each Learnable configuration is evaluated using 10 initializations.

During each repetition, the parameter configuration achieving the highest fidelity is retained. For the Learnable ansatz, the temperature of the soft mask is updated at each optimization step using the schedule defined in Equation 7. After training, the soft mask is converted into the hard efective mask defined in Equation 9. The retained gate parameters are then optimized for a further 200 steps using a learning rate reduced by a factor of ten, and the fidelity is recomputed using the resulting ansatz.

## 4.4. Hyperparameter Optimization

In a near term setting, the fidelity that can be achieved in practice is limited not only by the expressivity of the ansatz, but also by noise and other experimental imperfections. Increasing the number of phases to improve the fidelity beyond the range relevant to the application may therefore be counterproductive and result in a longer and more complex ansatz which is more subject to noise and control errors. We assume that this range is known and represent it through a target infidelity �. The aim of the hyperparameter optimization is then to identify a sparse ansatz whose final infidelity satisfies $I \leq \epsilon .$ In our experiments $\epsilon \in \{ 1 0 ^ { - 5 } , 1 0 ^ { - 4 } , 1 0 ^ { - 3 } \}$

A separate hyperparameter optimization is performed for each target state, qudit dimension, and value of �. The hyperparameters of the Learnable ansatz are selected sequentially, and each configuration considered during the search is evaluated using the training protocol described in Section 4.3. The procedure assumes that increasing the width � of the structural mask generally improves fidelity, while increasing $\lambda _ { \mathrm { m a s k } }$ generally removes more phases at the cost of fidelity. These relations are not guaranteed because every configuration requires a separate non-convex optimization. The search works as follows. First, we fix the number of blocks to $B = d$ in all experiments, so it is not included in the search. The remaining hyperparameters are selected in two stages:

1. select the width � of the structural mask;

2. select the mask regularization weight $\lambda _ { \mathrm { m a s k } } .$

During the first stage, the learned mask is disabled. Starting from $w = 1$ , the width is increased until a value satisfying $I \leq \epsilon$ is found, or the maximum width of 10 is reached. The interval between the largest tested width that does not satisfy the threshold and the smallest width that satisfies it is then refined through two binary search steps. The smallest tested width satisfying the target infidelity is selected.

During the second stage, the selected width is kept fixed and the learned mask is enabled. We do not apply regularization directly to the gate parameters, while the entropy regularization weight is fixed to $\lambda _ { \mathrm { e n t } } = 0 . 1$ . The mask regularization weight is the only regularization coeficient selected by this search. The search first evaluates $\lambda _ { \mathrm { m a s k } } = 0$ , for which the learned mask is disabled and only the structural mask is used. Positive values are then tested starting from $1 0 ^ { - 5 }$ and increasing by a factor of 10 until the target infidelity is no longer satisfied or the maximum value of 1 is reached. The interval between the largest accepted value and the first rejected value is refined through four binary search steps. A final comparison is performed between the selected value and values 2% above and below it. The largest tested value satisfying $I \leq \epsilon$ is selected.

The gate parameters obtained from the width search are used to initialize each configuration evaluated during the search for $\lambda _ { \mathrm { m a s k } }$ . Each tested value of $\lambda _ { \mathrm { m a s k } }$ therefore starts from the same gate parameters rather than from the result obtained for the preceding value.

After selecting � and $\lambda _ { \mathrm { m a s k } }$ , the pipeline includes two final refinement steps. First, greedy refinement attempts to remove the remaining phases one at a time, accepting a removal only when the resulting ansatz still satisfies $I \ \leq \ \epsilon$ . Accepted removals are retained during the subsequent tests. Second, structural compression removes any SNAP gate whose phase vector is entirely zero and combines its two adjacent displacement gates into a single displacement. Figure 1 summarizes the complete procedure. The block search shown in the figure is omitted in our experiments, since we fix $B = d$

Each hyperparameter optimization run is allocated a maximum wall time of four days. If the optimization does not complete within this limit, the search is terminated and the analysis uses all configurations completed before the interruption. This limitation mainly afects the experiments with $d = 3 2 .$ , for both the $F o u r i e r – 5$ and Haar random states.

Unlike the baseline optimization described in Section 4.5, which uses diferent scalarization weights to explore the trade-of between fidelity and phase count, the sequential procedure searches directly for a sparse configuration satisfying a specified infidelity threshold. This keeps the search focused on the accuracy range relevant to the application.

## 4.5. Baseline Hyperparameter Optimization

All baseline architectures (Full, Diagonal (adaptive), Diagonal (multiple), and Grid) are optimized following the optimization protocol used in Ferrari Dacrema et al. [14], combining a scalarized loss and a Tree-structured Parzen Estimator (TPE) which is allowed to explore 50 cases.

The hyperparameter optimization of the baselines includes two sets of hyperparameters. The first controls the learning process, including the learning rate and the regularization weights $\lambda _ { 1 }$ and $\lambda _ { 2 }$ The second defines the structure of the ansatz and includes, for example, the number of blocks $B ,$ , the diagonal width �, the number of diagonals $m _ { \mathrm { d i a g } }$ , and the number of anti-diagonals $m _ { \mathrm { a n t i } }$ , depending on the baseline architecture.

These hyperparameters, particularly those controlling the ansatz structure, afect both the fidelity of the prepared state and the required resources, such as the number of blocks and non-zero SNAP phases. To balance these competing objectives, [14] formulated a multi-objective optimization problem that maximizes the fidelity while minimizing the number of non-zero SNAP phases. Following a standard scalarization approach [18], the two objectives are combined into a single scalar objective used to evaluate each hyperparameter configuration ℎ:

$$
L _ { h } ( \vec { \alpha } _ { h } , \Theta _ { h } , \beta ) = F ( \vec { \alpha } _ { h } , \Theta _ { h } ) - \beta \frac { p ( \Theta _ { h } ) } { 2 d - 2 } ,\tag{16}
$$

where $F ( \vec { \alpha } _ { h } , \Theta _ { h } )$ is the fidelity obtained after optimizing the gate parameters for a given hyperparameter configuration $h , p ( \Theta _ { h } )$ denotes the number of non-zero SNAP phase angles, and $\beta$ is a penalty coeficient controlling the trade-of between preparation fidelity and resource requirements. Its value is fixed before each hyperparameter optimization run and is not selected by TPE. The normalization factor

![](images/c921d217007b8fa8e38e18327a3646df98a8ae6b0a326c8f0b144e2e24d1222c.jpg)  
Figure 1: Overview of the optimization pipeline for the Learnable ansatz. In the experiments reported in this work, the number of blocks is fixed to $B = d ,$ so the block search shown in the first stage is omitted. The remaining stages include width selection, learned-mask optimization, greedy refinement, and structural compression.

2� − 2 corresponds to the number of real degrees of freedom of a pure quantum state of dimension �, excluding normalization and global phase, allowing a consistent interpretation of $\beta$ across diferent qudit dimensions.

Note that the same paper [14] also introduced a second scalarized loss penalizing the duration of the ansatz and observed that the duration is a much more dificult quantity to minimize. For this reason, in this study we focus on penalizing the number of phases.

Diferent values of � produce diferent operating points, ranging from high-fidelity solutions requiring many non-zero SNAP phases to sparser solutions with lower resource requirements. We therefore perform a separate hyperparameter optimization for each value

$$
\beta \in \{ 0 , 1 0 ^ { - 6 } , 1 0 ^ { - 5 } , 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 1 0 ^ { - 1 } \} .
$$

For each value of �, the Tree-structured Parzen Estimator (TPE) explores at most 50 hyperparameter configurations. Using the same scalarization weights as [14] allows for a direct comparison between our Learnable method and the architectures they proposed.

## 5. Results and Discussion

We evaluate the trade-of between preparation fidelity and the number of retained SNAP phases through three complementary analyses. First, for each target infidelity, we identify the ansatz requiring the minimum number of phases. Second, we compare the Pareto frontiers to examine the trade-of over the complete set of hyperparameter configurations explored. Third, we use hypervolume to summarize this trade-of both over the complete infidelity range and within the region satisfying $I \leq 1 0 ^ { - 3 }$ , where the Pareto frontiers indicate Learnable is the most efective.

## 5.1. Number of SNAP Phases per Target Infidelity

This first analysis compares the number of SNAP phases required by each ansatz to reach a given target infidelity. For each target state, qudit dimension, and threshold $\epsilon ,$ we consider all evaluated configurations satisfying $I \leq \epsilon$ and report the minimum number of SNAP phases. Lower values indicate that the same state preparation fidelity is reached using fewer phases, indicating the method exhibits a better trade-of.

As shown in Tables 1 and 2, Learnable requires fewer SNAP phases than all the baselines in 7 of the 15 Fourier-5 configurations. Its improvements are most consistent at $d = 1 6$ , where it performs best at three of the five infidelity thresholds. For the Haar random state, Learnable obtains 5 strict improvements and 2 ties. In this case, the strongest results are obtained at $d = 8 .$ , where it requires the fewest phases at four of the five thresholds, while the results become more mixed as � increases.

For the Fourier-5 state, see Table 1, Learnable is particularly efective for the larger qudit dimensions and the more demanding reachable thresholds. $\mathrm { A t } d = 3 2$ , it requires 81 phases for $I \leq 1 0 ^ { - 3 }$ and 153 phases for $I \leq 1 0 ^ { - 4 }$ , compared with 152 and 348 phases, respectively, for the best baseline. These values correspond to reductions of approximately 47% and 56%. A similar trend is observed at $d = 1 6 $ where Learnable reduces the phase count from 54 to 30 at $I \leq 1 0 ^ { - 3 }$ and from 72 to 50 at $I \leq 1 0 ^ { - 4 }$

For the Haar random state, see Table 2, the improvements are more moderate and depend more strongly on the qudit dimension. $\quad \operatorname { A t } d = 8 .$ Learnable achieves the lowest phase count for thresholds between $1 0 ^ { - 2 }$ and $1 0 ^ { - 4 }$ 10 ; in particular, it requires 16 phases at $I \leq 1 0 ^ { - 4 }$ , compared with 24 for the best baseline. At $d = 1 6$ and the same threshold, the phase count decreases from 66 to 51. Conversely, Learnable is less competitive at $d = 3 2$ , where it does not achieve the minimum phase count at any of the reported thresholds. For example, at $I \leq 1 0 ^ { - 4 }$ , Learnable requires 153 phases, while the best baseline requires 142. Learnable also does not reach $I \leq 1 0 ^ { - 5 }$ for $d = 1 6$ and $d = 3 2$ within the allotted computational budget of four days and the hyperparameter optimization is terminated before its completion.

The ansatz durations reported in the same tables show that a lower phase count can often be obtained without increasing the duration substantially. For the Fourier-5 state at $d = 1 6$ and $I \leq 1 0 ^ { - 3 }$ Learnable reduces the number of phases from 54 to 30, while its duration of 214 $\mu \mathrm { s }$ remains within 10% of the shortest baseline duration of $1 9 5 \mu \mathrm { s }$ . For the Haar random state at $d = 8$ and $I \leq 1 0 ^ { - 4 }$ Learnable uses 16 phases and has a duration of 111 $\mu \mathrm { s } ,$ , compared with 24 phases and 117 �s for the best baseline. The behavior is particularly favorable for the Fourier-5 state at $d = 3 2$ $\mathrm { 4 t } \ I \ \leq 1 0 ^ { - 3 }$ Learnable achieves both the lowest phase count and the shortest duration, requiring 81 phases and 495 �s. $\mathrm { A t } I \le 1 0 ^ { - 4 }$ , it requires 153 phases and 701 $\mu \mathrm { s } ,$ , compared with 348 phases and 911 $\mu \mathrm { s }$ for the best baseline in terms of phase count. However, fewer phases do not always imply a shorter ansatz. For the Haar random state at $d = 1 6$ and $I \leq 1 0 ^ { - 4 }$ , Learnable reduces the phase count from 66 to 51, but its duration increases from the best baseline value of $1 8 5 \mu \mathrm { s }$ to $2 8 3 \mu \mathrm { s } .$ The least favorable case is observed for the Haar random state at $d = 3 2$ and $I \leq 1 0 ^ { - 4 }$ , where Learnable minimizes neither the phase count nor the duration, requiring 153 phases and 701 $\mu \mathrm { s } ,$ , compared with 142 phases and a minimum baseline duration of 283 $\mu \mathrm { s }$ . This confirms that the ansatz duration depends not only on the total number of active phases, but also on their distribution across the SNAP gates and circuit blocks.

## 5.2. Pareto Frontiers

The previous analysis considers a set of fixed infidelity thresholds. To examine the complete trade-of between preparation accuracy and the number of SNAP phases, we also compare the Pareto frontiers obtained by the diferent ansatzes. A configuration belongs to the Pareto frontier when no other evaluated configuration achieves both a lower infidelity and a lower number of phases.

Figures 2 and 3 show the Pareto frontiers for the Fourier-5 and Haar random states at $d = 1 6$ and $d = 3 2$ . For ${ \mathrm { F o u r i e r } } - 5 ,$ Learnable provides a clear advantage at intermediate infidelities, approximately between $1 0 ^ { - 2 }$ and $1 0 ^ { - 4 }$ , where it reaches similar or lower infidelities using fewer SNAP phases. It should be noted that this corresponds broadly to the target infidelities used for the hyperparameter optimization of Learnable. This behavior is visible at both dimensions and is particularly pronounced at $d = 3 2$ . For the Haar random state, the advantage is smaller, Learnable is competitive at $d = 1 6 ,$ while at $d = 3 2$ the frontiers largely overlap and no ansatz performs consistently better over the complete range. In general, the diferences are less marked at the lowest and highest infidelities. This indicates that Learnable is most efective in the intermediate infidelity region. Its reduced advantage at very low infidelities may be due to the increased dificulty of jointly optimizing the gate parameters and the learned mask when higher accuracy is required.

Table 1
<table><tr><td rowspan="2">d</td><td rowspan="2">Infidelity ≤</td><td colspan="2">Full T [µs]</td><td rowspan="2">Diagonal</td><td rowspan="2"> $( \mathrm { a d a p t i v e } )$   $\textsf { T } [ \mu s ]$ </td><td rowspan="2">Diagonal Phases</td><td rowspan="2"> $\mathrm { ( m u l t i p 1 e ) }$   $\textsf { T } [ \mu s ]$ </td><td colspan="2">Grid</td><td colspan="2">Learnable</td></tr><tr><td>Phases</td><td>Phases</td><td>Phases</td><td>T [µs]</td><td>Phases</td><td>T [µs]</td></tr><tr><td rowspan="5">8</td><td> $1 0 ^ { - 1 }$ </td><td>8</td><td>28</td><td>2</td><td>20</td><td>4</td><td>40</td><td>1</td><td>10</td><td>6</td><td>60</td></tr><tr><td> $1 0 ^ { - 2 }$ </td><td>16</td><td>56</td><td>10</td><td>77</td><td>10</td><td>60</td><td>10</td><td>76</td><td>7</td><td>70</td></tr><tr><td> $1 0 ^ { - 3 }$ </td><td>24</td><td>85</td><td>16</td><td>88</td><td>15</td><td>92</td><td>16</td><td>87</td><td>8</td><td>80</td></tr><tr><td> $1 0 ^ { - 4 }$ </td><td>32</td><td>113</td><td>22</td><td>114</td><td>19</td><td>106</td><td>20</td><td>113</td><td>21</td><td>127</td></tr><tr><td> $1 0 ^ { - 5 }$ </td><td>40</td><td>141</td><td>28</td><td>128</td><td>22</td><td>115</td><td>28</td><td>139</td><td>34</td><td>164</td></tr><tr><td rowspan="5">16</td><td> $\overline { { 1 0 ^ { - 1 } } }$ </td><td>16</td><td>40</td><td>4</td><td>40</td><td>2</td><td>20</td><td>3</td><td>24</td><td>14</td><td>135</td></tr><tr><td> $1 0 ^ { - 2 }$ </td><td>48</td><td>120</td><td>16</td><td>143</td><td>21</td><td>149</td><td>26</td><td>155</td><td>14</td><td>135</td></tr><tr><td> $1 0 ^ { - 3 }$ </td><td>80</td><td>200</td><td>54</td><td>220</td><td>76</td><td>214</td><td>55</td><td>195</td><td>30</td><td>214</td></tr><tr><td> $1 0 ^ { - 4 }$ </td><td>80</td><td>200</td><td>72</td><td>253</td><td>87</td><td>228</td><td>77</td><td>255</td><td>50</td><td>277</td></tr><tr><td> $1 0 ^ { - 5 }$ </td><td>128</td><td>320</td><td>110</td><td>346</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td rowspan="5">32</td><td> $\overline { { 1 0 ^ { - 1 } } }$ </td><td>64</td><td>113</td><td>10</td><td>103</td><td>10</td><td>77</td><td>12</td><td>68</td><td>32</td><td>323</td></tr><tr><td> $1 0 ^ { - 2 }$ </td><td>96</td><td>170</td><td>34</td><td>242</td><td>44</td><td>306</td><td>35</td><td>204</td><td>55</td><td>379</td></tr><tr><td> $1 0 ^ { - 3 }$ </td><td>288</td><td>510</td><td>152</td><td>591</td><td>213</td><td>814</td><td>195</td><td>752</td><td>81</td><td>495</td></tr><tr><td> $1 0 ^ { - 4 }$ </td><td>544</td><td>963</td><td>348</td><td>911</td><td>663</td><td>1435</td><td>555</td><td>1304</td><td>153</td><td>701</td></tr><tr><td> $1 0 ^ { - 5 }$ </td><td>-</td><td>一</td><td></td><td>-</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td><td>-</td></tr></table>

Comparison of the minimum number of non-zero phases required to reach diferent infidelity thresholds for the Fourier-5 target state.  
Table 2

<table><tr><td rowspan="2"></td><td rowspan="2">Infidelity ≤</td><td colspan="2">Full</td><td rowspan="2">Phases</td><td rowspan="2">Diagonal (adaptive)  $\textsf { T } [ \mu s ]$ </td><td rowspan="2">Phases</td><td rowspan="2">Diagonal (multiple)  $\textsf { T } [ \mu s ]$ </td><td colspan="2">Grid</td><td colspan="2">Learnable</td></tr><tr><td>Phases</td><td> $\textsf { T } [ \mu s ]$ </td><td>Phases</td><td> $\textsf { T } [ \mu s ]$ </td><td>Phases</td><td> $\textsf { T } [ \mu s ]$ </td></tr><tr><td rowspan="5">8</td><td> $1 0 ^ { - 1 }$ </td><td>16</td><td>56</td><td>6</td><td>60</td><td>7</td><td>58</td><td>8</td><td>56</td><td>8</td><td>80</td></tr><tr><td> $1 0 ^ { - 2 }$ </td><td>24</td><td>85</td><td>16</td><td>98</td><td>15</td><td>92</td><td>15</td><td>100</td><td>11</td><td>87</td></tr><tr><td> $1 0 ^ { - 3 }$ </td><td>32</td><td>113</td><td>19</td><td>104</td><td>19</td><td>106</td><td>22</td><td>120</td><td>15</td><td>107</td></tr><tr><td> $1 0 ^ { - 4 }$ </td><td>32</td><td>113</td><td>24</td><td>117</td><td>24</td><td>128</td><td>26</td><td>132</td><td>16</td><td>111</td></tr><tr><td> $1 0 ^ { - 5 }$ </td><td>40</td><td>141</td><td>34</td><td>142</td><td>34</td><td>154</td><td>45</td><td>164</td><td>34</td><td>164</td></tr><tr><td rowspan="5">16</td><td> $\overline { { 1 0 ^ { - 1 } } }$ </td><td>32</td><td>80</td><td>16</td><td>131</td><td>21</td><td>134</td><td>20</td><td>106</td><td>16</td><td>161</td></tr><tr><td> $1 0 ^ { - 2 }$ </td><td>48</td><td>120</td><td>30</td><td>155</td><td>37</td><td>148</td><td>35</td><td>169</td><td>32</td><td>223</td></tr><tr><td> $1 0 ^ { - 3 }$ </td><td>64</td><td>160</td><td>44</td><td>161</td><td>57</td><td>182</td><td>54</td><td>241</td><td>42</td><td>256</td></tr><tr><td> $1 0 ^ { - 4 }$ </td><td>80</td><td>200</td><td>66</td><td>243</td><td>70</td><td>186</td><td>69</td><td>185</td><td>51</td><td>283</td></tr><tr><td> $1 0 ^ { - 5 }$ </td><td>112</td><td>280</td><td>98</td><td>296</td><td>-</td><td>-</td><td>108</td><td>275</td><td>-</td><td>-</td></tr><tr><td rowspan="5">32</td><td> $\overline { { 1 0 ^ { - 1 } } }$ </td><td>96</td><td>170</td><td>36</td><td>274</td><td>49</td><td>240</td><td>38</td><td>212</td><td>72</td><td>452</td></tr><tr><td> $1 0 ^ { - 2 }$ </td><td>96</td><td>170</td><td>64</td><td>210</td><td>96</td><td>192</td><td>85</td><td>275</td><td>76</td><td>473</td></tr><tr><td> $1 0 ^ { - 3 }$ </td><td>128</td><td>226</td><td>96</td><td>238</td><td>129</td><td>251</td><td>127</td><td>225</td><td>121</td><td>614</td></tr><tr><td> $1 0 ^ { - 4 }$ </td><td>160</td><td>283</td><td>142</td><td>393</td><td>197</td><td>370</td><td>161</td><td>308</td><td>153</td><td>701</td></tr><tr><td> $1 0 ^ { - 5 }$ </td><td>448</td><td>793</td><td>673</td><td>1428</td><td>-</td><td>-</td><td>320</td><td>593</td><td>一</td><td>-</td></tr></table>

Comparison of the minimum number of non-zero phases required to reach diferent infidelity thresholds for the Haar random target state.

## 5.3. Hypervolume Analysis

While the Pareto frontiers show where the diferences between the ansatzes occur, their comparison becomes less immediate when several target states and qudit dimensions are considered. We therefore use hypervolume to summarize the overall trade-of between infidelity and the number of SNAP phases. Within the same analysis and qudit dimension, a larger hypervolume indicates a better trade-of.

Following Ferrari Dacrema et al. [14], we first compute the hypervolume over the complete range of infidelities explored during the experiments. Since the Pareto frontiers show that Learnable is most competitive in the intermediate region, we also perform a second analysis including only configurations satisfying $I \leq 1 0 ^ { - 3 }$ . The two hypervolumes are computed independently, and their absolute values should therefore not be compared across the two analyses. Comparisons are made only between ansatzes within the same qudit dimension and panel. Tables 3 and 4 report the mean hypervolume and the corresponding 95% confidence intervals obtained from 1000 bootstrap samples.

![](images/89db3d8200f5c4fff0b3336ada956da60d19e90c4f34d590b3fb565eba4346ee.jpg)  
(a) � = 16.

![](images/70a77aa35bbeff51e09ddd720a5db1d0636773520e198222e467b486032cd7f9.jpg)  
(b) $d = 3 2 .$

Figure 2: Pareto frontiers comparing the trade-of between infidelity and the number of SNAP phases for the Fourier-5 target state.  
![](images/bd95a27e813188dc3b0ee855b780bac2d950c95aa9298ebfc8681da0a9dc1ba9.jpg)  
(a) � = 16.

![](images/e195c03256fefe09bf448c2264d6c40e718b2e4c470967e1311b9f5e08cd9aad.jpg)  
(b) $d = 3 2 .$  
Figure 3: Pareto frontiers comparing the trade-of between infidelity and the number of SNAP phases for the Haar random target state.

Over the complete range, Diagonal (adaptive) obtains the highest mean hypervolume for both target states and all the considered dimensions. For Fourier-5, Learnable remains close to Diagonal (adaptive) at $d = 1 6$ and $d = 3 2$ , with overlapping confidence intervals, but does not improve on its mean hypervolume. For the Haar random state, the intervals also overlap at $d = 8$ and $d = 1 6 .$ At � = 32, however, Learnable obtains a clearly lower value, with 32.1 [30.2, 33.8] compared with 36.2 [35.8, 36.8] for Diagonal (adaptive).

The restricted analysis gives a diferent result for Fourier-5. Learnable obtains the highest mean hypervolume at all three dimensions. $\quad \mathrm { A t ~ } d = 8 .$ , its confidence interval overlaps those of the comparison ansatzes, so the small diference in the mean does not indicate a clear advantage. The improvement is more marked at $d = 1 6$ , where Learnable obtains 55.0 [53.0, 57.5] compared with 48.9 [48.2, 49.7] for Diagonal (adaptive), and at $d = 3 2$ , where it obtains 41.7 [39.1, 43.6] compared with 32.8 [32.1, 33.7]. In both cases, the confidence intervals do not overlap. For the Haar random state, restricting the analysis does not produce the same consistent improvement. At � = 8, Learnable has a slightly higher mean than Diagonal (adaptive), 73.1 compared with 72.9, but

<table><tr><td>d</td><td>Full</td><td>Diagonal (adaptive)</td><td>Diagonal (multiple)</td><td>Grid</td><td>Learnable</td></tr><tr><td>8</td><td>51.9 [51.5, 52.3]</td><td>57.5 [56.9, 58.0]</td><td>54.1 [53.3, 54.9]</td><td>57.0 [56.4, 57.6]</td><td>54.6 [52.6, 56.9]</td></tr><tr><td>16</td><td>40.2 [39.8, 40.7]</td><td>46.6 [46.1, 47.2]</td><td>43.3 [42.6, 44.2]</td><td>44.9 [44.3, 45.7]</td><td>44.8 [43.3, 46.6]</td></tr><tr><td>32</td><td>27.4 [27.0, 27.9]</td><td>35.2 [34.6, 35.9]</td><td>32.9 [32.4, 33.5]</td><td>32.3 [31.8, 32.9]</td><td>34.8 [33.1, 36.3]</td></tr></table>

(a) Complete explored infidelity range.
<table><tr><td rowspan=1 colspan=1>d</td><td rowspan=1 colspan=1>Full</td><td rowspan=1 colspan=1>Diagonal (adaptive)</td><td rowspan=1 colspan=1>Diagonal (multiple)</td><td rowspan=1 colspan=1>Grid</td><td rowspan=1 colspan=1>Learnable</td></tr><tr><td rowspan=2 colspan=1>816</td><td rowspan=1 colspan=1>64.9 [64.4, 65.6]</td><td rowspan=1 colspan=1>68.4 [67.7, 69.2]</td><td rowspan=1 colspan=1>65.2 [64.2, 66.4]</td><td rowspan=1 colspan=1>66.4 [65.7, 67.2]</td><td rowspan=1 colspan=1>69.9 [66.7, 72.8]</td></tr><tr><td rowspan=1 colspan=1>46.6 [46.0, 47.2]</td><td rowspan=2 colspan=1>48.9 [48.2, 49.7]32.8 [32.1, 33.7]</td><td rowspan=2 colspan=1>41.4 [40.5, 42.6]27.1 [26.4, 27.9]</td><td rowspan=1 colspan=1>46.2 [45.4, 47.2]</td><td rowspan=2 colspan=1>55.0 [53.0, 57.5]41.7 [39.1, 43.6]</td></tr><tr><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>26.8 [26.3, 27.4]</td><td rowspan=1 colspan=1>27.8 [27.2, 28.7]</td></tr></table>

(b) Configurations satisfying $I \leq 1 0 ^ { - 3 } .$

## Table 3

Bootstrap hypervolume for the Fourier-5 target state, computed using 1000 bootstrap samples. The two analyses are computed independently, so their absolute hypervolume values are not directly comparable. Values should be compared across ansatzes within the same row and panel.
<table><tr><td>d</td><td>Full</td><td>Diagonal (adaptive)</td><td>Diagonal (multiple)</td><td></td><td>Grid</td><td>Learnable</td></tr><tr><td>8</td><td>49.1 [48.4, 50.0]</td><td>53.9 [53.2, 54.7]</td><td>52.3 [51.2, 53.4]</td><td></td><td>53.1 [51.9, 54.1]</td><td>53.1 [49.8, 55.5]</td></tr><tr><td>16</td><td>41.1 [40.6, 41.7]</td><td>45.3 [44.6, 46.0]</td><td>42.5 [41.6, 43.6]</td><td></td><td>44.1 [43.5, 44.8]</td><td>44.0 [41.9, 46.7]</td></tr><tr><td>32</td><td>32.6 [32.2, 33.0]</td><td>36.2 [35.8, 36.8]</td><td>33.7 [32.9, 34.6]</td><td></td><td>35.3 [34.5, 36.1]</td><td>32.1 [30.2, 33.8]</td></tr></table>

(a) Complete explored infidelity range.
<table><tr><td rowspan=1 colspan=1>d</td><td rowspan=1 colspan=1>Full</td><td rowspan=1 colspan=1>Diagonal (adaptive)</td><td rowspan=1 colspan=2>Diagonal (multiple)</td><td rowspan=1 colspan=2>Grid</td><td rowspan=1 colspan=2>Learnable</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>67.7 [66.6, 68.9]</td><td rowspan=1 colspan=1>72.9 [71.9, 74.1]</td><td rowspan=1 colspan=2>70.5 [68.8, 72.2]</td><td rowspan=1 colspan=2>70.8 [69.1, 72.2]</td><td rowspan=1 colspan=2>73.1 [67.5, 76.2]</td></tr><tr><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>56.9 [56.1, 57.7]</td><td rowspan=1 colspan=1>59.7 [58.7, 60.8]</td><td rowspan=1 colspan=1>54.9 [53.6, 56.6]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>57.6 [5</td><td rowspan=1 colspan=1>6.6, 58.6]</td><td rowspan=1 colspan=1>58.3 [5</td><td rowspan=1 colspan=1>5.2, 62.5]</td></tr><tr><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>45.0 [44.5, 45.6]</td><td rowspan=1 colspan=1>46.7 [46.0, 47.5]</td><td rowspan=1 colspan=2>42.5 [41.5, 43.8]</td><td rowspan=1 colspan=2>45.0 [43.9, 46.2]</td><td rowspan=1 colspan=2>41.6 [38.8, 44.3]</td></tr></table>

(b) Configurations satisfying $I \leq 1 0 ^ { - 3 } .$

## Table 4

Bootstrap hypervolume for the Haar random target state, computed using 1000 bootstrap samples. The two analyses are computed independently, so their absolute hypervolume values are not directly comparable. Values should be compared across ansatzes within the same row and panel.

their confidence intervals overlap substantially. At � = 16, Diagonal (adaptive) has the higher mean, although the intervals again overlap. At � = 32, Diagonal (adaptive) remains clearly better, obtaining 46.7 [46.0, 47.5] compared with 41.6 [38.8, 44.3] for Learnable.

Overall, the two analyses show that Learnable is most competitive in the accuracy region where the Pareto frontiers indicate its main advantage. Although Diagonal (adaptive) performs better over the complete infidelity range, restricting the analysis to $I \leq 1 0 ^ { - 3 }$ reveals clear improvements for Fourier-5, particularly at � = 16 and � = 32. The results for the Haar random state are more mixed, indicating that the benefit of learning the mask depends on the target state.

## 6. Conclusions

This work investigates how learning based methods can be used to select the SNAP phases required for bosonic state preparation. Building on the heuristic sparse ansatzes introduced by Ferrari Dacrema et al. [14], we extended the Diagonal (adaptive) ansatz with a diferentiable mask that is optimized together with the gate parameters. The method starts from the phases allowed by the Diagonal (adaptive) structure and learns which of them should be retained.

We evaluated the proposed Learnable ansatz on Fourier-5 and Haar random target states for qudit dimensions up to � = 32. The comparison considered the minimum number of phases required to reach diferent target infidelities, the corresponding Pareto frontiers, and the hypervolume over both the complete explored range and the region satisfying $I \leq 1 0 ^ { - 3 }$

The results show that Learnable can reduce the number of SNAP phases required to reach a given infidelity. The improvements are clearest for Fourier-5, particularly at � = 16 and � = 32, where the method provides a better trade-of in the intermediate infidelity region. This is also reflected in the restricted hypervolume analysis, where Learnable obtains the highest mean hypervolume for Fourier-5 at all considered dimensions. For the Haar random state, the results are more mixed: Learnable remains competitive and improves several individual configurations, but Diagonal (adaptive) performs better in some dimensions and regions of the frontier.

Overall, the results indicate that learning which phases to retain can improve predefined sparse ansatzes, although the benefit depends on the target state and on the range of infidelities considered. The current method can only remove phases from the initial Diagonal (adaptive) structure. Future work may therefore consider learning phase locations over a broader initial structure, diferent mask architectures and studying its behavior under realistic noise and hardware constraints.

## Acknowledgments

We acknowledge that the scholarship of Marco Pennazio has been funded by ABB S.p.A. We acknowledge ISCRA for awarding this project access to the LEONARDO supercomputer, owned by the EuroHPC Joint Undertaking, hosted by CINECA (Italy).

## Declaration on Generative AI

During the preparation of this work, the authors used GPT-5.6 Sol in order to paraphrase and reword, as well as for grammar and spelling checks. After using these tools, the authors reviewed and edited the content as needed and take full responsibility for the publication’s content.

## References

[1] R. W. Heeres, B. Vlastakis, E. Holland, S. Krastanov, V. V. Albert, L. Frunzio, L. Jiang, R. J. Schoelkopf, Cavity state manipulation using photon-number selective phase gates, Phys. Rev. Lett. 115 (2015) 137002. URL: https://link.aps.org/doi/10.1103/PhysRevLett.115.137002. doi:10.1103/ PhysRevLett.115.137002.

[2] S. Krastanov, V. V. Albert, C. Shen, C.-L. Zou, R. W. Heeres, B. Vlastakis, R. J. Schoelkopf, L. Jiang, Universal control of an oscillator with dispersive coupling to a qubit, Phys. Rev. A 92 (2015) 040303. URL: https://link.aps.org/doi/10.1103/PhysRevA.92.040303. doi:10.1103/PhysRevA.92. 040303.

[3] S. Chakram, K. He, A. V. Dixit, A. E. Oriani, R. K. Naik, N. Leung, H. Kwon, W.-L. Ma, L. Jiang, D. I. Schuster, Multimode photon blockade, Nature Physics 18 (2022) 879–884. doi:10.1038/ s41567-022-01630-y.

[4] Y. Wang, Z. Hu, B. C. Sanders, S. Kais, Qudits and high-dimensional quantum computing, Frontiers in Physics Volume 8 - 2020 (2020). URL: https://www.frontiersin.org/journals/physics/articles/10. 3389/fphy.2020.589504. doi:10.3389/fphy.2020.589504.

[5] B. Royer, S. Singh, S. Girvin, Encoding qubits in multimode grid states, PRX Quantum 3 (2022) 010335. doi:10.1103/PRXQuantum.3.010335.

[6] Y. Xu, Y. Wang, E.-J. Kuo, V. V. Albert, Qubit-oscillator concatenated codes: Decoding formalism and code comparison, PRX Quantum 4 (2023) 020342. doi:10.1103/PRXQuantum.4.020342.

[7] E. O. Kiktenko, A. S. Nikolaeva, A. K. Fedorov, Colloquium: Qudits for decomposing multiqubit gates and realizing quantum algorithms, Rev. Mod. Phys. 97 (2025) 021003. URL: https://link.aps. org/doi/10.1103/RevModPhys.97.021003. doi:10.1103/RevModPhys.97.021003.

[8] T. Roy, T. Kim, A. Romanenko, A. Grassellino, Qudit-based quantum computing with srf cavities at fermilab, volume LATTICE2023, Fermi National Accelerator Laboratory (FNAL), Batavia, IL (United States), 2024. URL: https://www.osti.gov/biblio/2318671. doi:10.22323/1.453.0127.

[9] L. Krayzman, C. U. Lei, S. Ganjam, J. Teoh, L. Frunzio, R. J. Schoelkopf, Superconducting quantum memory with a suspended coaxial resonator, Applied Physics Letters 124 (2024). doi:10.1063/5. 0203906.

[10] R. W. Heeres, P. Reinhold, N. Ofek, L. Frunzio, L. Jiang, M. H. Devoret, R. J. Schoelkopf, Implementing a universal gate set on a logical qubit encoded in an oscillator, Nature Communications 8 (2017) 94. URL: https://doi.org/10.1038/s41467-017-00045-1. doi:10.1038/s41467-017-00045-1.

[11] T. Fösel, S. Krastanov, F. Marquardt, L. Jiang, Eficient cavity control with snap gates, arXiv preprint arXiv:2004.14256 (2020).

[12] M. Kudra, M. Kervinen, I. Strandberg, S. Ahmed, M. Scigliuzzo, A. Osman, D. P. Lozano, M. O. Tholén, R. Borgani, D. B. Haviland, G. Ferrini, J. Bylander, A. F. Kockum, F. Quijandría, P. Delsing, S. Gasparinetti, Robust preparation of wigner-negative states with optimized snap-displacement sequences, PRX Quantum 3 (2022) 030301. URL: https://link.aps.org/doi/10.1103/PRXQuantum.3. 030301. doi:10.1103/PRXQuantum.3.030301.

[13] J. Job, Eficient, direct compilation of su (n) operations into snap & displacement gates, arXiv preprint arXiv:2307.11900 (2023).

[14] M. Ferrari Dacrema, D. M. Kürkçüoğlu, A. Li, T. Roy, S. Zorzetti, Sparse phase ansatzes for resourceeficient qudit state preparation via the SNAP-displacement protocol (2026). arXiv:2603.12203.

[15] N. Bornman, T. Roy, J. A. Job, N. Anand, G. N. Perdue, S. Zorzetti, M. S. Alam, Benchmarking the performance of a high-q cavity qudit using random unitaries, Quantum Science and Technology 10 (2025) 025062.

[16] A. P. Guerreiro, C. M. Fonseca, L. Paquete, The hypervolume indicator: Computational problems and algorithms, ACM Comput. Surv. 54 (2021). URL: https://doi.org/10.1145/3453474. doi:10. 1145/3453474.

[17] A. Auger, J. Bader, D. Brockhof, E. Zitzler, Hypervolume-based multiobjective optimization: Theoretical foundations and practical implications, Theoretical Computer Science 425 (2012) 75–103. URL: https://www.sciencedirect.com/science/article/pii/S0304397511002428. doi:https: //doi.org/10.1016/j.tcs.2011.03.012, theoretical Foundations of Evolutionary Computation.

[18] C. Bazgan, S. Ruzika, C. Thielen, D. Vanderpooten, The power of the weighted sum scalarization for approximating multiobjective optimization problems, Theory of Computing Systems 66 (2022) 395– 415. URL: https://doi.org/10.1007/s00224-021-10066-5. doi:10.1007/s00224-021-10066-5.