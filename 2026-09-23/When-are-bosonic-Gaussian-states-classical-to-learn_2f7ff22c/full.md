# When are bosonic Gaussian states classical to learn?

Senrui Chen<sup>∗†1</sup>, Antonio Anna Mele<sup>‡2</sup>, Francesco Anna Mele<sup>§1</sup>, and John Preskill<sup>¶1</sup>

<sup>1</sup>Institute for Quantum Information and Matter, Caltech, Pasadena, CA 91125, USA <sup>2</sup>Dahlem Center for Complex Quantum Systems, Freie Universität Berlin, 14195 Berlin, Germany

September 22, 2026

## Abstract

A fundamental question in physics is: When does classical behavior emerge from quantum systems? Bosonic Gaussian states provide a natural setting to explore this quantum-classical boundary, as they capture both the classical field behavior and the intrinsic quantum nature of light. Here, we address this problem from a learning-theoretic perspective by asking: When are bosonic Gaussian states classical to learn? That is, under what conditions (if any) can an n-mode bosonic Gaussian state be learned with as few samples, and with operations as simple, as are needed to learn a classical 2n-variate Gaussian distribution? We establish a smooth crossover in learnability governed by the state’s thermal fluctuations:

1. Cold Gaussian states are non-classical to learn. When the covariance matrix satisfies $\Sigma \ \leq$ $\textstyle { \left( { \frac { 1 } { 2 } } + O ( { \frac { 1 } { n } } ) \right) } \mathbb { I }$ , i.e. close to the vacuum covariance, tomography under single-copy (i.e., nonentangled) measurements fundamentally requires $\Omega ( n ^ { 3 } )$ copies—strictly exceeding the sample complexity $\Theta ( n ^ { 2 } )$ of learning classical Gaussian distributions. We show that this hardness persists even when few-copy entangled measurements are allowed.

2. Warm Gaussian states are classical to learn. When thermal fluctuations exceed the vacuum noise, parameterized by $\Sigma \geq ( \frac { 1 } { 2 } + \nu ) \mathbb { I }$ for any parameter $\nu > 0 ,$ , we prove that single-copy tomography requires $N = \Theta \left( n ^ { 2 } \operatorname* { m i n } ( n , 1 + \nu ^ { - 1 } ) \right)$ copies. This bound is tight and is achieved by simple, non-adaptive, unentangled heterodyne measurements. Crucially, for $\nu = \Omega ( 1 )$ ), the sample complexity drops to $\Theta ( n ^ { 2 } )$ , matching the classical case.

Our results tightly characterize a quantum-to-classical crossover in the learnability of bosonic Gaussian states, reveal a novel connection between fundamental physics and statistical learning theory, and have implications for real-world quantum sensing experiments.

## Contents

1 Introduction 2   
1.1 Background 4   
1.2 Results 5   
1.3 Discussion 8   
1.4 Related works 9   
1.5 Outlook 10   
2 Technical Summary 11   
2.1 Encode finite-dimensional states into Gaussian states . 11   
2.2 Bayesian posterior anti-concentration for bosons 13   
2.3 Learning warm Gaussian states and a new trace-distance bound . 14   
Acknowledgments 15   
3 Additional notations and preliminaries 17   
4 k-copy lower bound for cold Gaussian states 18   
5 Tight single-copy lower bound for all-temperature Gaussian states 28   
5.1 Definitions . 28   
5.2 A relative local parametrization 29   
5.3 Reduce posterior anti-concentration to likelihood ratio bound 32   
5.4 Likelihood ratio bound – fixed transcripts 33   
5.5 Likelihood ratio bound II – high probability argument 37   
5.6 Putting everything together . 37   
5.7 Delayed proof for Lemma 12 40   
Alternative single-copy lower bound for non-adaptive scheme 44   
7 Upper bound on learning warm Gaussian states 52   
7.1 A one-sided relative covariance trace-distance bound 52   
7.2 Heterodyne tomography and sample complexity 59   
8 Matching lower bound for learning warm Gaussian states 63

## 1 Introduction

The world is governed by the laws of quantum mechanics, but our daily life appears to be classical. It is now well accepted that such macroscopic classical behavior emerges from underlying microscopic quantum systems [Zur03]. This phenomenon was first understood through investigations into the nature of light: At the macroscopic level, light behaves like a classical electromagnetic field as characterized by Maxwell’s equations. At the microscopic level, light consists of photons and thus has quantized energy. The theoretical formalization and experimental validation built around the wave-particle duality of light are fundamental to the establishment of today’s quantum physics.

In modern quantum information language, light can be described by bosonic continuous-variable quantum systems. For an n-mode bosonic system, the quantum state is either described via a quasiprobability distribution over 2n quadrature observables, known as the Wignerfunction, or via the photon number occupation basis, known as the Fock basis. These two representations are equivalent and reflect the viewpoints of waves and particles, respectively. A class of bosonic quantum states that appear ubiquitously in nature (thanks to a central-limit-type argument [Gla63]) is the class of Gaussian states, defined as those whose Wigner functions are Gaussian probability distributions. Gaussian states describe the most classical-field-like state of light. Indeed, they have 2n well-defined quadrature expectation values (given by the Gaussian mean) with added Gaussian noise (given by the Gaussian covariance). At the same time, they also capture the quantum nature of light, in that the covariance matrix is bounded from below due to the uncertainty principle. The seminal works by Glauber and Sudarshan [Gla63, Sud63] investigated the quantum-classical boundary for states of light. In particular, Glauber observed that an asymptotically large thermal fluctuation renders the quantum fluctuation negligible in comparison and thus makes the state essentially a classical random field.

The goal of the current work is to study when classical behavior emerges in quantum systems from the perspective of statistical learning theory. The basic question is the following: Given N identical and independent copies of an unknown physical state (which can be a quantum state or a classical distribution) sampled from a certain family, how large must N be to ensure that an observer can learn a classical description of the state to ε distinguishability distance<sup>1</sup> with high probability (say 2/3). The answer is referred to as the sample complexity of the learning task.

For a family of n-mode bosonic Gaussian states, since their Wigner functions are given by 2n-variate classical Gaussian distributions, it is natural to compare the sample complexity of learning both families. In particular, we say there is an emergent classical behavior if (1) the quantum and classical families have the same sample complexity; and (2) optimal learning of the quantum family can be achieved via simple “classical-like” measurements. While there can be multiple diferent definitions for classical-like measurement, here we require the measurement to at least be non-entangled, i.e., no joint measurements across multiple copies are allowed. See Fig. 1 for an illustration. Intuitively, when these criteria are satisfied, the family of quantum states is essentially classical to learn.

The sample complexity for learning a 2n-dimensional classical Gaussian distribution is known to be $N = \Theta ( n ^ { 2 } / \varepsilon ^ { 2 } )$ . In fact, simple empirical estimators for sample mean and covariance sufice to achieve the optimal scaling [ABDH<sup>+</sup>20, DMR20]. Very recently, the sample complexity for learning an n-mode bosonic Gaussian state has also been found to be $N = \Theta ( n ^ { 2 } / \bar { \varepsilon ^ { 2 } } )$ [CFG<sup>+</sup>26]. However, the only known protocol to achieve this uses a highly entangled measurement, violating the criterion of using “classical-like” measurements. In contrast, the best currently known non-entangled protocol uses $N = \widetilde O ( n ^ { 3 } / \varepsilon ^ { 2 } )$ copies [BMEM25], which does not match the classical learning complexity. We thus ask the following question,

## Can bosonic Gaussian states be learned using $\Theta ( n ^ { 2 } / \varepsilon ^ { 2 } )$ copies via classical-like measurements? If not, under what additional conditions does this hold?

Beyond the theoretical interest of exploring the quantum-to-classical crossover, finding simpler sampleoptimal learning algorithms for bosonic Gaussian states is of practical relevance. Indeed, learning and estimation of Gaussian states is a crucial subroutine for quantum sensing, with examples including gravitational wave and dark matter detection. It is also useful for benchmarking photonic quantum computers and quantum networks. For those practical quantum devices, a multi-copy entangled measurement is often out of reach. It is thus highly desirable to consider more experimentally friendly learning protocols without compromising the eficiency.

![](images/8d23ee9efa6fffbad691a1b5ab39e25b8ca70abaebb3bb285aeec0d4cdde3612.jpg)  
Figure 1: Illustration of entangled vs. non-entangled learning schemes. An experimental device (on the left) generates many i.i.d. copies of an unknown quantum state of interest, which can be processed in two diferent ways: (a) With a quantum computer, one can conduct entangled measurements across all copies, which is the most powerful measurement allowed by quantum mechanics; (b) With a classical computer only, one can conduct only single-copy non-entangled measurements. Note that each measurement may be chosen adaptively based on outcomes of previous measurements.

## 1.1 Background

We first introduce some background in bosonic quantum information that is necessary to state our results. Bosons are quantum particles whose states are invariant under exchange. An n-mode bosonic system consists of diferent photon number sectors, with the m-photon Hilbert space given by $\mathcal { H } _ { m } ^ { ( n ) } \cong \mathrm { S y m } ^ { m } ( \mathbb { C } ^ { n } )$ and the full (infinite-dimensional) Hilbert space given by $\mathcal { H } ^ { ( n ) } = \bigoplus _ { m = 0 } ^ { \infty } \mathcal { H } _ { m } ^ { ( n ) }$ . A natural basis for $\mathcal { H } ^ { ( n ) }$ is the Fock basis $\{ \vert m \rangle \equiv \left. m _ { 1 } \right. \otimes \cdots \otimes \left. m _ { n } \right. \} _ { m \in \mathbb { N } ^ { n } }$ which specifies the number of photons in each mode. We will also use the ladder operators $\{ \hat { a } _ { j } , \hat { a } _ { j } ^ { \dagger } \} _ { j = 1 } ^ { n }$ , defined as

$$
\hat { a } _ { j } \left| m \right. = \sqrt { m _ { j } } \left| m - e _ { j } \right. , \quad \hat { a } _ { j } ^ { \dag } \left| m \right. = \sqrt { m _ { j } + 1 } \left| m + e _ { j } \right. ,\tag{1}
$$

and the photon number operators $\hat { n } _ { j } : = \hat { a } _ { j } ^ { \dagger } \hat { a } _ { j }$ which satisfy ${ \hat { n } } _ { j } \left| m \right. = m _ { j } \left| m \right.$ . The total photon number is given by $\begin{array} { r } { \hat { n } : = \sum _ { j = 1 } ^ { n } \hat { n } _ { j } } \end{array}$ . We use $\mathcal { D } ( \mathcal { H } ^ { ( n ) } )$ to denote the set of all density operators on $\mathcal { H } ^ { ( n ) }$

We also introduce the 2n quadrature operators $\hat { \pmb { r } } : = ( \hat { q } _ { 1 } , \cdots , \hat { q } _ { n } , \hat { p } _ { 1 } , \cdots , \hat { p } _ { n } )$ defined as

$$
\hat { q } _ { j } = \frac { \hat { a } _ { j } + \hat { a } _ { j } ^ { \dagger } } { \sqrt { 2 } } , \quad \hat { p } _ { j } = \frac { \hat { a } _ { j } - \hat { a } _ { j } ^ { \dagger } } { i \sqrt { 2 } } ,\tag{2}
$$

which satisfy the canonical commutation relation $[ \hat { r } _ { k } , \hat { r } _ { l } ] = i \Omega _ { k l } \mathrm { I }$ where $\Omega : = { \binom { 0 _ { n } } { - \mathbb { I } _ { n } } } \mathbb { I } _ { n } \bigg )$ . Now, for an operator $\hat { O }$ acting on $\mathcal { H } ^ { ( n ) }$ , define its characteristic function as $\chi _ { \hat { O } } ( \alpha ) : = \mathrm { T r } ( \hat { O } e ^ { i \alpha ^ { T } \Omega \hat { r } } )$ and the Wigner

function $\mathsf { a s } ^ { 2 }$

$$
W _ { \hat { O } } ( \beta ) : = \frac { 1 } { ( 2 \pi ) ^ { 2 n } } \int \mathrm { d } ^ { 2 n } \pmb { \alpha } e ^ { i \pmb { \alpha } ^ { T } \Omega \beta } \chi _ { \hat { O } } ( \pmb { \alpha } ) , \quad \forall \pmb { \beta } \in \mathbb { R } ^ { 2 n } .\tag{3}
$$

For any density operator $\rho \in \mathcal { D } ( \mathcal { H } ^ { ( n ) } )$ , its Wigner function satisfies two important properties: (i) Normalization, i.e. $\begin{array} { r } { \int \mathrm { d } ^ { 2 n } \beta W _ { \rho } ( \beta ) = 1 } \end{array}$ . Note that $W _ { \rho }$ can in general contain negative values, and is therefore called a quasi-probability distribution. (ii) All observable expectation values can be computed using $\begin{array} { r } { \mathrm { T r } ( \rho \hat { O } ) = ( 2 \pi ) ^ { n } \int \mathrm { d } ^ { 2 n } { \beta } W _ { \rho } ( \beta ) W _ { \hat { O } } ( \beta ) } \end{array}$ ; thus the Wigner function completely describes a bosonic state.

A Gaussian state is a state whose Wigner function is a Gaussian distribution, i.e. $W _ { \rho } = \mathcal { N } ( \mu , \Sigma )$ Such a state is completely determined by its first and second moments,

$$
\mu _ { j } = \operatorname { T r } ( \rho { \hat { r } } _ { j } ) , \quad \Sigma _ { k l } : = { \frac { 1 } { 2 } } \operatorname { T r } ( \rho \{ { \hat { r } } _ { k } - \mu _ { k } , { \hat { r } } _ { l } - \mu _ { l } \} ) ;\tag{4}
$$

thus we denote a Gaussian state by $\rho ( { \boldsymbol { \mu } } , { \boldsymbol { \Sigma } } )$ ). A pair $( \mu , \Sigma )$ defines a valid Gaussian state if and only if $\Sigma + i \Omega / 2 \geq 0 \ [ \mathrm { W P G P ^ { + } 1 2 } ]$ . A subclass of Gaussian states of particular interest is called passive Gaussian states (or, gauge-invariant Gaussian states), which are Gaussian states whose density operators are block diagonal in the total photon number sectors. Passive Gaussian states are always zero-mean, and the eigenvalues of Σ coincide with the symplectic eigenvalues; each equals $\begin{array} { r } { \hat { n } _ { j } + \frac { 1 } { 2 } } \end{array}$ , where $\hat { n } _ { j }$ is the expected photon occupation of the corresponding eigenmode, a proxy for its temperature. Equivalently, they are states that can be obtained by applying a passive Gaussian unitary on products of thermal states.

An alternative representation of bosonic states is the Glauber-Sudarshan P function [Gla63, Sud63] defined as $\begin{array} { r } { \rho = \int P ( \alpha ) | \alpha \rangle \langle \alpha | \mathrm { d } ^ { 2 n } } \end{array}$ α where |α⟩ is the coherent state (i.e., eigenstates of $\{ \hat { a } _ { j } \} )$ . Conventionally, a state with positive P function is viewed as a “classical state”, as it is a mixture of coherent states and thus has no entanglement or squeezing. However, such a state is not automatically classical to learn, much as separable states may require entangled measurements for optimal learning [DLT02].<sup>3</sup> Specific to Gaussian states, a positive P function is equivalent to requiring $\bar { \Sigma } > \frac 1 2 \mathbb { I }$ . Another condition we will use is $\nu _ { - } \mathbb { I } \leq \Sigma - { \frac { 1 } { 2 } } \mathbb { I } \leq \nu _ { + } \mathbb { I }$ , which implies the expected number of photons in each mode of the centered state is bounded between ν and $\nu _ { + }$ . This is because $\begin{array} { r } { \operatorname { T r } ( \hat { n } _ { j } \rho ) = \frac { 1 } { 2 } \operatorname { T r } ( ( \hat { q } ^ { 2 } + \hat { p } ^ { 2 } - 1 ) \rho ) } \end{array}$ . We will thus refer to $\nu _ { \pm }$ as the photon number or temperature parameter, the latter with slight abuse of terminology (but accurate for passive Gaussian states).

All of our theorems stated in Sec. 1.2 hold for general POVM measurements on n-mode bosonic systems, which are allowed to be non-Gaussian. Here, we review one specific type ofGaussian measurement, heterodyne measurement, whose outcome distribution on an n-mode Gaussian state $\rho ( { \boldsymbol { \mu } } , { \boldsymbol { \Sigma } } )$ is a $2 n \cdot$ variate classical Gaussian $\begin{array} { r } { \mathcal { N } ( \boldsymbol { \mu } , \boldsymbol { \Sigma } + \frac { 1 } { 2 } \mathbb { I } ) } \end{array}$ ; that is, the state’s Wigner function convoluted with additional Gaussian noise.

## 1.2 Results

We first present our technical results and later discuss the physical implications and practical applications. Our first result sets a limitation for non-entangled measurements in Gaussian state learning:

Theorem 1 (Lower bound against single-copy measurements). A single-copy, possibly adaptive scheme that can learn any n-mode Gaussian state $\rho ( { \boldsymbol { \mu } } , { \boldsymbol { \Sigma } } )$ to ε trace distance with probability at least $2 / 3$ requires

$N = \Omega ( n ^ { 3 } / \varepsilon ^ { 2 } )$ copies. This holds even under the promise that $\rho$ is passive and that $\Sigma \leq ( \frac { 1 } { 2 } + O ( \frac { 1 } { n } ) )  { \mathbb { I } }$ Furthermore, when restricted to non-adaptive schemes, the same lower bound holds even if the spectrum of Σ is a priori known.

We refer to Gaussian states satisfying $\Sigma \leq ( \frac { 1 } { 2 } + O ( \frac { 1 } { n } ) )  { \mathbb { I } }$ as cold Gaussian states: their expected total number of photons is $O ( 1 )$ , so they are close to the (zero-temperature) vacuum. For passive Gaussian states, Theorem 1 matches the performance of heterodyne measurement [BMEM25], which is a nonentangled, non-adaptive Gaussian measurement. For general Gaussian states, it matches the upper bound given by the adaptive single-copy Gaussian scheme in [BMEM25] up to a doubly-logarithmic energy factor. This result also shows that entangled measurements provide a provable advantage in learning Gaussian states, achieving $N = \Theta ( n ^ { 2 } / \varepsilon ^ { 2 } )$ instead [CFG<sup>+</sup>26]. Crucially, the hardness of learning can be established using an extremely low-temperature family of passive states whose total number of photons in expectation is $O ( 1 )$ .

One might wonder if Theorem 1 can be circumvented via few-copy entangled measurements, as there are examples where 2-copy measurements are exponentially more eficient than 1-copy measurements [HKP21, CCHL22]. Our second result gives a negative answer:

Theorem 2 (Lower bound against k-copy entangled measurements). There exist universal constants $\varepsilon _ { 0 } > 0$ and $n _ { 0 } \in \mathbb { N }$ such that the following holds. Let $n \geq n _ { 0 } ,$ , let $0 < \varepsilon < \varepsilon _ { 0 } ,$ , and let $k \geq 1$ be any integer. Any possibly adaptive scheme that learns every n-mode Gaussian state $\rho ( { \boldsymbol { \mu } } , { \boldsymbol { \Sigma } } )$ to trace-distance error ε with probability at least $2 / 3 ,$ using only k-copy measurements, requires

$$
N = \Omega \left( \frac { n ^ { 3 } } { \varepsilon ^ { 2 } \operatorname* { m i n } \{ \sqrt { k } , n \} } \right)\tag{5}
$$

copies. Here each measurement acts on at most k fresh copies, and no quantum memory is retained between measurements. This holds even under the promise that $\rho$ is passive and that $\begin{array} { r } { \Sigma \leq ( \frac { 1 } { 2 } + O ( \frac { 1 } { n } ) ) \mathbb { I } } \end{array}$

In particular, for any constant $k ,$ learning via k-copy measurements still requires $N = \Omega ( n ^ { 3 } / \varepsilon ^ { 2 } )$ copies.   
Again, this result holds even for low-temperature passive Gaussian states.

Taken together, Theorem 1 and Theorem 2 give a negative answer to the first part of our main question. That is, general bosonic Gaussian states cannot be learned using classical-like sample complexity via classical-like measurements. The proof idea behind both theorems is inspired by the literature of learning finite-dimensional quantum states with single-copy or few-copy measurements [LN25, CHL<sup>+</sup>23, CLL24, KLMR26, NZ26], together with novel techniques necessary to make things work for bosons. Specifically, Theorem 2 is proved by establishing that tomography of n-dimensional mixed states reduces to tomography of n-mode passive Gaussian states, and thus the finite-dimensional no-go theorem of [KLMR26] applies (with the latter being a refinement of [CLL24]). The case $k = 1$ also proves the adaptive part of Theorem 1. The direct bosonic analyses discussed later further address the temperature dependence and the known-spectrum promise for non-adaptive schemes (see Sec. 2.2 for an overview and Secs. 5 and 6 for the corresponding proofs).

Now we turn to the question of what additional conditions are needed for Gaussian states to be classical to learn. Intuitively, a state with suficiently large fluctuations compared to vacuum should look classical [Gla63]. We precisely quantify this intuition in the following theorem:

Theorem 3 (Tight transition of learnability). Under the promise that an n-mode Gaussian state $\rho ( { \boldsymbol { \mu } } , { \boldsymbol { \Sigma } } )$ satisfies $\Sigma \geq ( \frac { 1 } { 2 } + \nu )  { \mathbb { I } } f o r$ some $\nu > 0$ that can depend on $n ,$ the necessary and suficient number of copies to learn $\rho t o \varepsilon$ trace distance with probability at least $2 / 3$ using single-copy measurements is given by

$$
N = \Theta \biggl ( \frac { n ^ { 2 } } { \varepsilon ^ { 2 } } \operatorname* { m i n } \{ n , 1 + \nu ^ { - 1 } \} \biggr ) .\tag{6}
$$

![](images/baba37dc901fa2a46f87bd754015df80b369fa16238f227fc269564a18a50f7b.jpg)  
Figure 2: Quantum-to-classical crossover in the learnability of bosonic Gaussian states. The vertical axis denotes the sample complexity for learning to ε trace distance with $2 / 3$ probability using single-copy $( \mathrm { i . e . }$ . non-entangled) measurements. The horizontal axis denotes a minimal-temperature parameter ν such that the family of Gaussian states is promised to satisfy $\Sigma \geq ( \frac { 1 } { 2 } + \nu ) \mathbb { I }$ . Our Theorem 3 gives a tight characterization of the sample complexity in the entire regime of $\nu ~ \in ~ \left( 0 , \infty \right)$ as ${ \cal N } = \Theta \big ( n ^ { 2 } \varepsilon ^ { - 2 } \operatorname* { m i n } \{ n , 1 + \nu ^ { - 1 } \} \big )$ , and reveals a smooth learnability crossover from $\nu \sim 1 / n$ to $\nu \sim 1$

Or, equivalently,

$$
N = \left\{ \Theta \left( \frac { n ^ { 3 } } { \varepsilon ^ { 2 } } \right) , \qquad i = \nu < \frac { 1 } { n } , \right.\tag{7}
$$

This bound can be achieved by heterodyne measurements for all values of $\nu > 0 .$

As introduced in Sec. 1.1, ν is a parameter that lower bounds the minimal (expected) number of photons in each mode. Alternatively, it is a lower bound on the minimal fluctuation of a Gaussian state in any quadrature, ofset by the vacuum variance ${ \frac { 1 } { 2 } } .$ . Our results tightly characterize the quantum-to-classical crossover in the learnability of bosonic Gaussian states for any temperature parameter $\nu > 0 . ^ { 4 }$ We refer to Gaussian states with $\nu = \Omega ( 1 )$ as warm Gaussian states: every eigenmode carries a constant number of thermal photons. Such warm Gaussian states can be learned by simple classical-like measurements with classical-like sample complexity. We have answered the second part of our main question: a constant number of thermal photons per mode sufices for Gaussian states to be classical to learn. We illustrate the crossover in Fig. 2.

To our knowledge, the only previously known bound for single-copy learning of Gaussian states is a uniform upper bound of $O ( n ^ { 3 } / \varepsilon ^ { \bar { 2 } } )$ for heterodyne measurements [BMEM25, FRSF26], which we show is tight only in the $\nu = O ( 1 / n )$ regime. (Here we focus on $\nu > 0$ so there is no additional energy-dependent factor.) The lower bound in all regimes and the upper bound for $\nu = \omega ( 1 / n )$ are our new contributions.

<table><tr><td>Experiments</td><td>Estimated n</td><td>Estimated ν</td><td>Regime</td></tr><tr><td>ADMX axion search  $[ \mathrm { D F K ^ { + } } 1 8 ]$ </td><td> $n \simeq 2 6 0$ </td><td> $\nu \simeq 4 . 3$ </td><td>Warm</td></tr><tr><td></td><td> $\Delta f = 2 5 \mathrm { k H z } ,$ </td><td> $f \simeq 6 5 0 \mathrm { M H z } ,$ </td><td> $\nu > 1$ </td></tr><tr><td></td><td> $\delta f = 9 6 \mathrm { H z } .$ </td><td> $T _ { \mathrm { c a v } } \simeq 1 5 0 \mathrm { m K } .$ </td><td></td></tr><tr><td>Photon-counting axion</td><td> $n \simeq 1 2$ </td><td> $\nu \simeq 3 . 2 \times 1 0 ^ { - 4 }$ </td><td>Cold</td></tr><tr><td>haloscope  $[ \mathrm { B B D V ^ { + } 2 5 } ]$ </td><td> $\Delta f \simeq 1 . 1 \mathrm { M H z } ,$ </td><td> $f \simeq 7 . 3 7 \mathrm { G H z } ,$ </td><td> $\nu \ll 1 / n$ </td></tr><tr><td></td><td> $\delta f \simeq 9 5 \mathrm { k H z } .$ </td><td> $T _ { \mathrm { e f f } } \simeq 4 4 \mathrm { m K } .$ </td><td></td></tr></table>

Table 1: Examples of mode counts and photon occupations in two dark-matter search experiments, together with their classification into the warm and cold regimes according to our criteria. Here, $\Delta f$ denotes the analysis or efective detection bandwidth, and $n \simeq \Delta f / \delta f$ . For $[ \mathrm { D F K ^ { + } } 1 8 ] , \delta f$ is the spectralbin width; for $[ \mathrm { B B D V ^ { + } } 2 5 ]$ , it is the efective frequency scale $1 / \tau$ associated with the detection window $\tau = 1 0 . 5 \mu \mathrm { s }$ . For both cases ν is estimated using the Bose-Einstein statistics $\nu = ( \exp ( h f / k _ { B } T ) - 1 ) ^ { - 1 }$

## 1.3 Discussion

Our results answer the question posed at the outset. Bosonic Gaussian states are classical to learn when each eigenmode contains at least a constant number of thermal photons in expectation. They are maximally hard to learn when the occupation number is of order $1 / n$ per eigenmode, so that the total number of thermal photons is constant. There is a smooth crossover between these two regimes, corresponding to $\nu \sim 1 / n$ and $\nu \sim 1$

Our results can be intuitively understood as a manifestation of wave-particle duality. In the cold regime $( \nu \sim 1 / n )$ , since the total photon number is a constant, the state’s behavior is largely dominated by the single-particle sector, which is isomorphic to an n-level quantum particle. Tomography of such an n-dimensional state using single-copy measurements is known to require $\Theta ( n ^ { 3 } )$ samples because single-copy measurements cannot be aligned with the state’s unknown eigenbasis, and measurements in diferent bases are mutually incompatible $[ \mathrm { C H L } ^ { + } 2 3 ]$ . See Sec. 2.1 for a more rigorous development of this intuition.

In the warm regime $( \nu \sim 1 )$ , the higher-photon-number sectors become significant, which causes the state to more closely resemble a classical electromagnetic wave, and the complexity of learning classical Gaussian distributions, $\Theta ( n ^ { 2 } )$ , kicks in. Note that in both the $\nu \sim 1 / n$ and the $\nu \sim 1$ cases, the noise introduced by heterodyne measurements $\textstyle { \frac { 1 } { 2 } } \mathbb { I }$ is comparable to the state’s fluctuation Σ, which means one can learn Σ to the same relative precision using similar sample complexity in both cases. The factor of n separation really arises in converting the covariance estimator to a density-operator estimator with trace-distance error ε. See Sec. 2.3 for further discussion.

Let us now connect our results to real-world quantum sensing settings, by citing examples from dark-matter search experiments in the literature. In each example, we estimate the number of modes n and the average photon number per mode ν. Then, by comparing ν with $1 / n$ and 1, we infer whether the experiments operate in the cold or warm regime in the sense we have defined. To be clear, our rigorous results are for full tomography of an unstructured Gaussian state family (with temperature constraints), which do not immediately apply to those practical setups where the states of interest can lie in a much lower-dimension subspace. Nevertheless, we believe this comparison provides guidance on whether the experiments operate in a quantum or classical regime.

The comparison is summarized in Table 1. For both experiments, n denotes the number of efective frequency-bin modes.<sup>5</sup> The average photon number per mode ν is estimated using Bose-Einstein statistics (see $\mathrm { [ B B D V ^ { + } 2 5 }$ , App J]), $\nu = ( \exp ( h f / k _ { B } T ) - 1 ) ^ { - 1 }$ . By our criteria, the ADMX experiment $[ \mathrm { D F K ^ { + } } 1 8 ]$ operates in the warm regime where heterodyne-type readout is provably optimal among single-copy measurements. In contrast, the axion haloscope $[ \mathrm { B B D V ^ { + } 2 5 } ]$ is in the very cold regime (with ν far below the crossover scale $1 / n )$ , where quantum-limited Gaussian readout carries an unavoidable penalty, and indeed that experiment opts for non-Gaussian photon counting instead. These examples demonstrate that, depending on the experimental setup, a practical sensing task can operate on either side of our crossover. They also show that practical experiments are genuinely multi-mode, with n ranging from tens to hundreds of bosonic modes.

## 1.4 Related works

Bosonic and Gaussian state learning. Bosonic systems have a long history in research on quantum mechanics, quantum optics, and quantum information theory. Tomography of bosonic quantum states has long been investigated both in theory and experiment [LR09]. A recent series of works has begun to study the non-asymptotic sample complexity for learning bosonic systems with provable guarantees in trace distance closeness; see [Mel26] for a review. In particular, $[ \mathrm { M M B } ^ { + } 2 5 ]$ shows that learning general energy-constrained bosonic states is extremely ineficient, with a sample complexity that scales exponentially in the number of modes. In contrast, energy-constrained Gaussian states, a practically motivated subclass of bosonic states $[ \mathsf { W P G P } ^ { + } 1 2 , \mathsf { M M B } ^ { + } 2 5 ]$ , can be learned using a polynomial number of samples.

Subsequent works have significantly advanced the understanding of Gaussian state learning. [BMEM25] proposes an algorithm based on adaptive single-copy Gaussian measurements that learns any nmode Gaussian state with energy at most $E$ to ε trace distance using ${ \widetilde O } ( n ^ { 3 } / \varepsilon ^ { 2 } + n$ log log E) samples. $[ \mathrm { C M F ^ { + } 2 6 } ]$ proves this $\Omega ( n ^ { 3 } / \varepsilon ^ { 2 } )$ scaling is inevitable for any scheme based on Gaussian measurements (more generally, any Wigner-positive measurement), even for learning the subclass of passive Gaussian states. They also show that non-Gaussian measurements can break the $n ^ { 3 }$ barrier, achieving ${ \widetilde O } ( n ^ { 2 } / \varepsilon ^ { 2 } +$ n log log E) for learning passive Gaussian states. More recently, $[ \mathrm { C F G ^ { + } 2 6 } ]$ completely settles the sample complexity of Gaussian state learning to be $\Theta ( n ^ { 2 } / \varepsilon ^ { 2 } )$ . A concurrent work [CGYZ26] obtains similar complexity for pure Gaussian states, and also proves the surprising Ω(log log E) energy dependence is inevitable when restricted to Gaussian measurements. It is worth mentioning that an important foundation for these results is the recently developed set of bounds on the trace distance between Gaussian states [BMM<sup>+</sup>25, Hol24, BMEM25, FRSF26].

While previous works have established a separation between Gaussian and non-Gaussian measurements for Gaussian state learning $[ \mathrm { C M F ^ { + } 2 6 } , \mathrm { C F G ^ { + } 2 6 } ]$ , our work establishes a separation between entangling and non-entangling measurements for the same task. Combining these two aspects, we conclude that the $n ^ { 2 }$ sample complexity for learning general Gaussian states is only possible with both non-Gaussian and entangling measurements. Removing either one of these two resources restores the $n ^ { 3 }$ scaling. On the other hand, for learning warm Gaussian states as in Theorem 3, both separations disappear, and a Gaussian, non-entangling, non-adaptive scheme sufices to achieve the optimal $n ^ { 2 }$ scaling. Together, these give a complete picture of when bosonic Gaussian states are classical to learn.

Quantum learning theory in finite-dimensional systems. Our work belongs to the rapidly growing research program of quantum learning theory. One central focus there is to find provable advantages of quantum resources $( \mathrm { e . g . }$ , entanglement, magic, quantum memory) in learning certain properties of certain families of quantum states or processes (see e.g. [HBC<sup>+</sup>22, LBØ<sup>+</sup>25, HCMP26]). Most of these works are developed for finite-dimensional quantum systems. Among those, a particularly relevant series of works explore the sample complexity for learning d-dimensional states using non-entangled or partially-entangled measurements [CHL<sup>+</sup>23, LN25, CLL24, KLMR26] as opposed to fully-entangled measurements [HHJ<sup>+</sup>17, OW16, PSTW25]. As will be reviewed in Sec. 2.1 and 2.2, the techniques for establishing our lower bounds are inspired by those works, with crucial distinctions unique to the bosonic setting. To our knowledge, the quantum-to-classical crossover of learnability has not been explored in finite-dimensional systems. Indeed, the hard instances for full tomography of d-dimensional states are perturbations of the maximally mixed state, which already has infinite-temperature [CHL<sup>+</sup>23]; hence there is no warmer family of states that is easier to learn. How to properly generalize the crossover to finite-dimensional systems is thus an interesting open question.

Quantum sensing and imaging. Bosonic and Gaussian quantum information theory are directly relevant to real-world quantum sensing and imaging applications. As some examples, squeezed Gaussian states are implemented in state-of-the-art gravitational-wave interferometers to suppress quantum noise [MWG<sup>+</sup>20]. Dark-matter searches can be modeled as finding a weak narrow-band bosonic signal in a wide-band thermal background [BPK<sup>+</sup>21]. Spatially incoherent optical sources studied in quantum imaging can be described by passive Gaussian states [TNL16], etc. These tasks motivate studying eficient characterization and parameter estimation of multi-mode bosonic Gaussian states.

Parameter estimation for bosonic states has long been studied in quantum metrology, mainly using the tools of (quantum) Fisher information and the (quantum) Cramér-Rao bound [GLM11, Mon13, FRG25]. Those tools focus on the asymptotic regime where the number of samples is abundant and the precision is extremely high, so that a local analysis around the ground truth is suficient. In contrast, quantum learning theory mainly operates in the non-asymptotic regime, which is particularly relevant when there are many underlying model parameters but a limited number of available samples (see [Wai19] for discussion). Nevertheless, quantum metrology and quantum learning theory share similar overarching goals. It is a fruitful direction to further connect these two research thrusts (see e.g. [CGZ26, CGYZ26] for some attempts along this line).

Emergent classicality from quantum systems There are diferent notions of emergent classicality in quantum systems. A traditional dynamical point of view is that quantum states become classical due to decoherence, induced by interaction with the environment [Zur03]. More recently, people have investigated notions such as the temperature above which quantum Hamiltonian exhibits classical properties. As one example, [BLMT24] shows that the Gibbs state of a local Hamiltonian becomes separable above a constant threshold of temperature (called the “sudden death of entanglement”). See also [PZC26] for the sudden death of other quantum properties including magic and classical tractability.

Specific to bosonic systems, a quantum-optical notion of classicality is given by the non-negativity of the Glauber–Sudarshan P function [Gla63, Sud63]. A quantum state with a positive P function is a mixture of coherent states with diferent amplitudes, which means it has no entanglement or squeezing. Every passive Gaussian state has a non-negative P function. Also, every bosonic state becomes P-positive if it undergoes an isotropic additive Gaussian noise channel in all quadratures.

Our work introduces a complementary statistical learning-theoretic notion. We call a quantum state family classical to learn if it can be reconstructed using the same sample complexity and “classical-like” measurements as for its classical analogue. Under this criterion, we see that cold passive Gaussian states are non-classical to learn, and they only become classical to learn when above a temperature threshold. This is in sharp contrast to the P function criterion, which views all passive Gaussian states as classical.

## 1.5 Outlook

Many interesting open problems are left for future work. Here, we list some promising directions.

Closing the k-entangled gap. While we obtain a tight bound for single-copy measurements in Gaussian state learning, for k-entangled measurements we only obtain a lower bound, mainly to show constant k cannot help circumvent the $\Omega ( n ^ { 3 } )$ lower bound. We have not pursued a matching k-entangled upper bound, because our focus is on “classical-like” measurements. By Theorem 2, improving on the $\Theta ( n ^ { 3 } )$ sample complexity requires k to grow with $n ,$ a regime where k-copy measurements become impractical. Still, it would be interesting to obtain a tight sample-complexity characterization for k-entangled measurements as has been explored in finite-dimensional systems [CLL24, PSTW25, KLMR26, NZ26]. Extending those results to the bosonic case seems possible but challenging. One dificulty is that the random purification channel for bosonic systems is much more complicated compared to the finitedimensional case [CFG<sup>+</sup>26].

Beyond full-state tomography. The task we study is learning the full description of a general Gaussian state (the only assumption is on the minimal fluctuations). In practical quantum sensing and benchmarking applications, the states to be learned usually come from a much more structured subset, and one may only want to learn a few properties rather than a complete description. It is thus an interesting open direction to generalize the quantum-classical learnability separation beyond full-state tomography. In particular, while heterodyne measurements are optimal among single-copy measurements for full tomography, they may be far from optimal for structured estimation tasks in the cold regime, where tailored non-Gaussian protocols can provide substantial advantages [TNL16].

## 2 Technical Summary

In this section, we give a high-level overview of the techniques used to establish our main results.

## 2.1 Encode finite-dimensional states into Gaussian states

A key ingredient in Theorem 2 is a novel protocol that shows that finite-dimensional tomography reduces to passive Gaussian-state tomography. To describe this reduction, we focus on passive Gaussian states, which sufice to establish all of our lower bounds. We write the n-mode bosonic Fock space as $\mathcal { H } ^ { ( n ) } = \bigoplus _ { m = 0 } ^ { \infty } \mathcal { H } _ { m } ^ { ( n ) }$ , where $\mathcal { H } _ { m } ^ { ( n ) } \cong \mathrm { S y m } ^ { m } ( \mathbb { C } ^ { n } )$ is the sector with exactly m photons. With respect to this decomposition, every n-mode passive Gaussian state admits a particularly simple parametrization [DG13, Sec. 17.1.4], which we refer to as the fugacity matrix representation. Namely, for any n-by-n Hermitian matrix R satisfying $0 \leq { \mathsf { R } } < { \mathbb { I } } _ { n }$ , called the fugacity matrix, the corresponding passive Gaussian state is

$$
\rho = \mathrm { d e t } (  { \mathbb { I } _ { n } } -  { \mathsf { R } } ) \bigoplus _ { m = 0 } ^ { \infty } \Gamma _ { m } (  { \mathsf { R } } ) ,\tag{8}
$$

where $\Gamma _ { m } ( \mathsf { R } ) : = \mathsf { R } ^ { \otimes m } | _ { \mathsf { S y m } ^ { m } ( \mathbb { C } ^ { n } ) } \in \mathcal { L } ( \mathcal { H } _ { m } ^ { ( n ) } )$ is the m-th symmetric power of R acting on the m-photon sector. In particular, $\Gamma _ { 0 } ( \mathsf { R } ) = 1$ and $\Gamma _ { 1 } ( \mathsf { R } ) = \mathsf { R }$ . Conversely, every passive Gaussian state admits a unique representation of this form.

Our key observation is that this representation provides a natural low-energy embedding of finitedimensional states into passive Gaussian states. Concretely, for every n-dimensional density matrix $\sigma \in { \mathcal { D } } ( \mathbb { C } ^ { n } )$ , let $\rho _ { \sigma }$ denote the n-mode passive Gaussian state with fugacity matrix $\mathsf { R } = \sigma / 4$ . The following three properties make this encoding useful:

1. Low photon number. The encoded state satisfies $\operatorname { T r } ( \rho _ { \sigma } \hat { n } ) \leq 1 / 3$ . Moreover, if $\sigma \leq 2 \mathbb { I } _ { n } / n$ (as in the hard family used to prove the finite-dimensional lower bound in [CLL24, KLMR26]), then $\begin{array} { r } { \Sigma _ { \rho _ { \sigma } } \leq \left( \frac { 1 } { 2 } + \frac { 1 } { 2 n - 1 } \right) \operatorname { I } _ { 2 n } } \end{array}$ , so the passive Gaussian state $\rho _ { \sigma }$ is “cold”.

2. Recover σ from $\rho _ { \sigma }$ . Let $\widehat { \rho }$ be any n-mode state satisfying $\frac 1 2 \| \widehat { \rho } - \rho _ { \sigma } \| _ { 1 } \leq \varepsilon \leq 1 / 1 0 0$ . Let $P _ { 1 }$ denote the projector onto the one-photon sector $\mathcal { H } _ { 1 } ^ { ( n ) } \cong \mathbb { C } ^ { n }$ . Then the normalized one-photon block

$$
\widehat { \sigma } : = \frac { P _ { 1 } \widehat { \rho } P _ { 1 } } { \mathrm { T r } ( P _ { 1 } \widehat { \rho } ) }
$$

is well defined and satisfies $\begin{array} { r } { \frac 1 2 \| \widehat { \sigma } - \sigma \| _ { 1 } \leq 1 6 \varepsilon } \end{array}$

3. Simulate $\rho _ { \sigma }$ from copies of $\sigma .$ . There is an algorithm, independent of $\sigma _ { z }$ , that exactly outputs one copy of $\rho _ { \sigma }$ using a random number T of copies of $\sigma ,$ , with E $T \leq 4 / 9$ (so, only a constant number of copies of $\sigma$ is needed on average); see lemma 5.

These three properties give a direct black-box reduction from finite-dimensional tomography to passive Gaussian-state tomography. Suppose that A is a possibly adaptive k-copy protocol that learns every passive Gaussian state in the cold family $\mathrm { t o ~ } \varepsilon$ trace distance using at most N copies, with probability at least $2 / 3 .$ . Starting from copies of an unknown n-dimensional state $\sigma \leq 2 \mathbb { I } _ { n } / n$ , we use the construction of the simulator above to generate, on demand, the copies of $\rho _ { \sigma }$ required by ${ \mathcal { A } } .$ . We then run each measurement block of A on the simulated copies and, from the final estimate ${ \widehat { \rho } } ,$ recover an estimate of $\sigma$ by taking its normalized one-photon block.

The only subtlety is that producing one copy of $\rho _ { \sigma }$ requires a random number of copies of $\sigma .$ Controlling this random cost while remaining within the allowed measurement model requires some care; the details are given in the proof of theorem 4 in the appendix. The resulting finite-dimensional protocol uses $O ( N )$ copies of $\sigma$ in total and learns $\sigma$ to 16ε trace distance with probability at least $1 / 2$ At the same time, the protocol retains the dependence on $k \colon$ each measurement performed by A acts on at most k Gaussian copies, and simulating it may require several finite-dimensional measurement blocks, since failed batches are discarded and repeated. Let $t _ { j }$ denote the (random) number of copies of σ used in the j-th such block, including failed batches. As shown in the appendix, uniformly over $\sigma \leq 2 \mathbb { I } _ { n } / n$

$$
\mathbb { E } \sum _ { j } t _ { j } \operatorname* { m i n } \{ \sqrt { t _ { j } } , n \} = O \Big ( N \operatorname* { m i n } \{ \sqrt { k } , n \} \Big ) .
$$

The left-hand side is precisely the quantity appearing in the finite-dimensional lower bound of [KLMR26], in the form stated in lemma 1. We can therefore apply that result to the induced finite-dimensional protocol. Since the recovery step learns $\sigma$ to accuracy 16ε, the lemma gives

$$
\Omega \left( \frac { n ^ { 3 } } { ( 1 6 \varepsilon ) ^ { 2 } } \right) \leq O \left( N \operatorname* { m i n } \{ \sqrt { k } , n \} \right) .
$$

Absorbing the constant factor into the universal constants and rearranging yields

$$
N = \Omega \left( \frac { n ^ { 3 } } { \varepsilon ^ { 2 } \operatorname* { m i n } \{ \sqrt { k } , n \} } \right) .
$$

The full argument is given in Sec. $4 . ^ { \ 6 }$

## 2.2 Bayesian posterior anti-concentration for bosons

The above reduction-based proof already gives the $\Omega ( n ^ { 3 } / \varepsilon ^ { 2 } )$ single-copy lower bound for cold Gaussian states. To establish the temperature-dependent lower bound, we analyze single-copy measurements directly on bosonic systems.

Our proof of the lower bound of Theorem 3 (and also the adaptive part of Theorem 1) follows the Bayesian posterior anti-concentration approach introduced in [CHL<sup>+</sup>23], but needs to address several significant technical challenges to extend this approach to bosonic systems. At a high level, we design an appropriate prior distribution of passive bosonic Gaussian states of appropriate temperature ν. We then show that, with high probability over a ground-truth state sampled from the prior and over N outcomes from any (potentially adaptive) measurement schedules, the posterior distribution will not concentrate over a ε-trace distance ball centered at the ground-truth state unless $\begin{array} { r } { N = \Omega ( \frac { n ^ { 2 } } { \nu \varepsilon ^ { 2 } } ) } \end{array}$ , which then implies the desired lower bound of sample complexity.

To be concrete, consider the following family of passive Gaussian states defined via fugacity:

$$
\mathsf { R } _ { X } : = r ( \mathbb { I } + \xi X ) , \quad \mathrm { w h e r e } \ r \asymp \frac { \nu } { 1 + \nu } , \quad \xi \asymp \varepsilon \sqrt { n \nu } ,\tag{9}
$$

and $X \in { \mathrm { H e r m } } ( \mathbb { C } ^ { n } )$ is sampled from the traceless Gaussian Unitary Ensemble conditioned on $\| X \| _ { \mathrm { o p } } \leq 4$ This choice ensures every $\rho _ { \mathsf { R } _ { X } }$ has symplectic eigenvalues lower bounded by ν and hence $\Sigma \geq ( \frac { 1 } { 2 } + \nu ) \mathbb { I }$ A key technical Lemma 12 we introduce states that learning $\rho _ { \mathrm { R } _ { X } }$ to ε trace distance implies learning X to $O ( n )$ trace distance. Therefore, we just have to show hardness of learning X. Intuitively, the space of X has volume of order $n ^ { 2 }$ that needs to be compensated in the posterior tilt.

While $\mathsf { R } _ { X }$ looks almost the same as the n-dimensional states ensemble used in $[ \mathrm { C H L } ^ { + } 2 3 ]$ except for the r factor, a crucial diference is that $\rho _ { \mathsf { R } _ { X } }$ is not linear in X, unlike the finite-dimensional case. As a consequence, while in the finite-dimensional case it sufices to consider a linear additive perturbation on the ground-truth $( \mathrm { i } . \mathrm { e } . , \rho _ { 0 } + Z )$ to prove posterior anti-concentration, such perturbation becomes hard to work with in our case. Instead, we introduce the following relative local parameterization Φ around any ground-truth $X _ { 0 }$

$$
\mathsf { R } _ { \Phi _ { X _ { 0 } } ( Z ) } : = r \frac { ( \mathbb { I } + \varepsilon X _ { 0 } ) ^ { \frac { 1 } { 2 } } ( \mathbb { I } + \varepsilon Z ) ( \mathbb { I } + \varepsilon X _ { 0 } ) ^ { \frac { 1 } { 2 } } } { \mathrm { T r } [ ( \mathbb { I } + \varepsilon X _ { 0 } ) ( \mathbb { I } + \varepsilon Z ) / n ] } ,\tag{10}
$$

where $Z \in { \mathrm { H e r m } } ( \mathbb { C } ^ { n } )$ is a traceless operator satisfying $\| Z \| _ { \mathrm { o p } } \le 0 . 1$ denoting the perturbation, and it is clear that $\mathsf { R } _ { \Phi _ { X _ { 0 } } ( 0 ) } = \mathsf { R } _ { X _ { 0 } }$ . This parameterization allows us to decouple Z from $X _ { 0 }$ when analyzing the m-photon sector, where we can single out the factor of $\Gamma _ { m } ( \mathbb { I } + \varepsilon Z )$ . We also show that Φ is a bi-Lipchitz map with bounded Jacobian determinant, so it will not significantly alter the Lebesgue volume that is needed in the proof.

The next crucial step is to control the averaged log likelihood ratio in order to control the posterior tilt. As passive Gaussian states are block-diagonal in total photon number sectors, the optimal measurements can without loss of generality be assumed to first project into a fixed total photon number sector. To bound the likelihood ratio in, say, the m-photon sector, we need to compute moments of Haar measure associated with the m-th symmetric power representation of $\mathbb { U } ( n )$ . This is in contrast to the finitedimensional case, where one only needs to evaluate the moments with the standard representations of $\mathbb { U } ( n )$ [CHL<sup>+</sup>23]. In our proof, we evaluate and bound those moments by connecting them to Schur polynomials, a central object in the representation theory with many useful properties [FH13]. We also use the log-Sobolev inequalities for the unitary group [MM13]. Consequently, we are able to bound the averaged posterior tilt by measuring each state copy as $O ( \varepsilon ^ { 2 } / n )$ . This implies $N = \Omega ( n ^ { 3 } / \varepsilon ^ { 2 } )$ copies are necessary to ensure a suficiently large tilt that compensates the $n ^ { 2 }$ prior volume factor. The complete proof is presented in Sec. 5.

Other lower bounds. In Sec. 6, we also present an alternative proof for Theorem 1 that only works for non-adaptive measurements, but holds even if the spectrum of Σ is assumed to be a priori known. This proof is also considerably simpler. The main technique is the Fano’s argument: first construct a ε-net of cold passive Gaussian states of size $2 ^ { \Theta ( n ^ { 2 } ) }$ , then show that with N rounds of non-adaptive measurement one can only extract $N \cdot O ( \varepsilon ^ { 2 } / n )$ bits of information from the net, and finally use Fano’s inequality to conclude $N = \Omega ( n ^ { 3 } / \varepsilon ^ { 2 } )$ is necessary for learning to succeed. Our proof resembles the finite-dimensional tomography bound studied in [LN25], plus that we still need to use Schur polynomials to help evaluate certain moments in a fixed photon number sector, and to carefully control the probability of high-photon-number event.

In Sec. 8, we prove a matching lower bound of $\Omega ( n ^ { 2 } / \varepsilon ^ { 2 } )$ for learning Gaussian states with promise $\Sigma \geq ( \frac { 1 } { 2 } + \nu ) \mathbb { I }$ for arbitrarily large ν using arbitrary measurements, thus completing the proof for Theorem 3. The lower bound is based on standard Fano’s argument and Holevo theorem. Similar proof techniques are used in $\mathrm { [ C M F ^ { + } 2 6 }$ , FRSF26].

## 2.3 Learning warm Gaussian states and a new trace-distance bound

We now give a high-level overview of the upper bound in Theorem 3. Details are presented in Sec. 7. In our convention, heterodyne detection on an n-mode Gaussian state $\rho ( { \boldsymbol { \mu } } , { \boldsymbol { \Sigma } } )$ produces a classical Gaussian sample $Y \sim { \mathcal { N } } ( \mu , C )$ with covariance $C : = \Sigma + \mathbb { I } / 2$ [BMEM25, FRSF26]. Thus, after the measurement, the statistical problem is classical: from independent samples we estimate the mean $\mu$ and covariance $C$ of a 2n-dimensional Gaussian distribution.

The dificulty is to turn these classical estimates back into an accurate quantum state. There are two related issues. First, simply subtracting the heterodyne noise $\mathbb { I } / 2$ from an empirical estimate of $C$ need not produce a valid quantum covariance matrix. Second, the map from a covariance matrix to the corresponding Gaussian state becomes increasingly sensitive near the pure-state boundary. We deal with the first problem by modifying the estimator, and with the second by exploiting the warmness promise.

The key idea is to slightly inflate the empirical covariance before subtracting the heterodyne noise. Let $\widehat { \mu }$ and $\widehat { C }$ be the empirical mean and centered empirical covariance of N heterodyne outcomes. Standard Gaussian concentration [BMEM25, Lemma 9, Eqs. (91), (94), and (95)] gives $( 1 - \zeta ) C \preceq \widehat { C } \preceq ( 1 + \zeta ) C$ with high probability, where $\zeta = O ( \sqrt { ( n + \log ( 1 / \delta ) ) / N } + ( n + \log ( 1 / \delta ) ) / N )$ . Instead of using $\hat { C } - \mathbb { I } / 2$ we define $\widehat { \Sigma } : = \widehat { C } / ( 1 - \zeta ) - { \mathbb { I } } / 2$ . On the same event,

$$
0 \preceq \widehat { \Sigma } - \Sigma \preceq \frac { 2 \zeta } { 1 - \zeta } C .\tag{11}
$$

Thus $\widehat { \Sigma } \succeq \Sigma \colon$ the estimator can only add covariance to the true state. In particular, $\widehat { \Sigma }$ is automatically physical. More importantly, this one-sided relation is exactly the property that allows us to obtain a sharper trace-distance estimate.

Our main technical ingredient is the following bound. Let $\bar { \Sigma }$ have symplectic eigenvalues at least $\nu _ { - } > 1 / 2$ , and set $\gamma _ { - } : = 1 - ( 2 \nu _ { - } ) ^ { - 1 }$ . Whenever $\Sigma ^ { \prime } \succeq \bar { \Sigma }$

$$
\frac { 1 } { 2 } \| \rho ( 0 , \Sigma ^ { \prime } ) - \rho ( 0 , \bar { \Sigma } ) \| _ { 1 } \leq \frac { 1 } { 2 \sqrt { 2 \gamma _ { - } } } \| \bar { \Sigma } ^ { - 1 / 2 } ( \Sigma ^ { \prime } - \bar { \Sigma } ) \bar { \Sigma } ^ { - 1 / 2 } \| _ { \mathrm { F } } .\tag{12}
$$

The important point is that the prefactor depends only on the distance from the pure-state boundary, and not on the number of modes or on the largest covariance eigenvalue.

Let us briefly sketch the proof of the trace distance bound and explain why the one-sided assumption helps. Similar approaches using resolvents, symmetric relative entropy, and Pinsker’s inequality are first explored in [FRSF26]. Yet, our tighter analysis is necessary to obtain the optimal temperature-dependent upper bound. A faithful Gaussian state can be written as a Gibbs state of a quadratic Hamiltonian, whose Hamiltonian matrix satisfies $H ( \Sigma ) = J f ( \Sigma J )$ , where $J = i \Omega$ and $\begin{array} { r } { f ( z ) = \int _ { - 1 / 2 } ^ { 1 / 2 } ( z - t ) } \end{array}$ <sup>−1</sup>dt [BBP15, Eq. (6)]. After putting the reference covariance Σ<sup>¯</sup> in Williamson normal form, $\Sigma ^ { \prime } \succeq \bar { \Sigma }$ becomes positivity of a relative perturbation $E \succeq 0$ . This positivity implies that the perturbation cannot reduce the spectral gap controlling the relevant resolvents, and therefore cannot move the estimate towards the singular pure-state boundary.

Keeping the exact t-dependent resolvent gap throughout the integral gives a Hamiltonian stability bound proportional to $1 / \gamma _ { - }$ . For a general two-sided perturbation, one instead needs a local smallerror assumption to control the perturbed resolvent, leading to a weaker dependence on the gap. Finally, the symmetric relative-entropy identity and Pinsker’s inequality [Wat18, Theorem 5.41] turn the Hamiltonian estimate into Eq. (12). The complete proof is given in Section 7.1.

We can now return to tomography. Under the promise $\Sigma \succeq ( 1 / 2 + \nu ) \mathbb { I }$ , the heterodyne covariance $C = \Sigma + \mathbb { I } / 2$ is comparable to Σ up to a factor depending only on $\nu .$ Therefore Eq. (11) implies that the normalized covariance error $\Sigma ^ { - 1 / 2 } ( \widehat { \Sigma } - \Sigma ) \bar { \Sigma } ^ { - 1 / 2 }$ has operator norm of order ζ. Since this is a $2 n \times 2 n$ matrix, its Frobenius norm is larger by at most a factor of order $\sqrt { n }$ . The one-sided trace-distance bound then shows that the covariance contribution scales as $O ( \sqrt { 1 + 1 / \nu } \zeta \sqrt { n } )$ . Since $\zeta = O ( \sqrt { ( n + \log ( 1 / \delta ) ) / N } )$ in the relevant regime, this is the dominant source of error and leads to the quadratic dependence on the number of modes.

The first moment is cheaper to estimate. The equal-covariance Gaussian fidelity formula [BBP15, Eq. (9)], together with the Fuchs–van de Graaf inequality [Wat18, Theorem 3.36], gives $\frac { 1 } { 2 } \| \rho ( \widehat { \mu } , \Sigma ) -$ $\begin{array} { r } { \rho ( \mu , \Sigma ) \| _ { 1 } \leq \frac { 1 } { 2 } \| \Sigma ^ { - 1 / 2 } ( \widehat { \mu } { - } \mu ) \| _ { 2 } } \end{array}$ . Gaussian concentration makes the right-hand side only $O ( \sqrt { ( n + \log ( 1 / \delta ) ) / N } )$ , which is smaller than the covariance contribution by a factor of order $\sqrt { n }$ . In particular, no assumption on the size of the displacement $\mu$ is required.

Finally, we insert the intermediate state $\rho ( \widehat { \mu } , \Sigma )$ and use the triangle inequality: the diference between $\rho ( \widehat { \mu } , \widehat { \Sigma } )$ and $\rho ( \widehat { \mu } , \Sigma )$ is the covariance error controlled by our new one-sided bound, while the diference between $\rho ( \widehat { \mu } , \Sigma )$ and $\rho ( { \boldsymbol { \mu } } , { \boldsymbol { \Sigma } } )$ is the first-moment error above. Combining the two contributions gives

$$
N = O \left( \left( 1 + \frac { 1 } { \nu } \right) \frac { n \big ( n + \log ( 1 / \delta ) \big ) } { \epsilon ^ { 2 } } \right) .\tag{13}
$$

For constant failure probability and ν bounded below by an absolute positive constant, this becomes $O ( n ^ { 2 } / \epsilon ^ { 2 } )$ , so independent heterodyne measurements achieve the same sample complexity as learning a classical Gaussian distribution.

Combining this estimate with the general heterodyne upper bound of Ref. [BMEM25, Theorem 10] gives $O ( n ^ { 2 } \operatorname* { m i n } \{ n , 1 + 1 / \nu \} / \epsilon ^ { 2 } )$ samples for constant failure probability. Thus the available upper bound interpolates between the general ${ \cal O } ( n ^ { 3 } / \epsilon ^ { 2 } )$ regime near the vacuum and the $O ( n ^ { 2 } / \epsilon ^ { 2 } )$ warm regime.

## Acknowledgments

We thank Lennart Bittel, Sitan Chen, Yanbei Chen, Marco Fanizza, Yaroslav Herasymenko, Hsin-Yuan Huang, Alfred Li, Zachary Mann, Zhihan Zhang, Andrew Zhao for helpful discussions. S.C., F.A.M. and J.P. acknowledge funding provided by the Institute for Quantum Information and Matter, an NSF Physics Frontiers Center (NSF Grant PHY-2317110), and the U.S. Department of Energy, Ofice of Science, National Quantum Information Science Research Centers, Quantum Systems Accelerator. A.A.M. and F.A.M. thank California Institute of Technology (Caltech) for the hospitality in March 2026, where part of this work was developed with the other authors. A.A.M. acknowledges support from a 2025 Google

PhD Fellowship. A.A.M. further acknowledges support from the BMFTR projects DAQC, MuniQC-Atoms, QuSol, Hybrid++, and PasQuops; the Clusters of Excellence ML4Q and MATH+; QuantERA; the Munich Quantum Valley; Berlin Quantum; the Quantum Flagship projects Millenion and Pasquans2; the DFG through CRC 183 and SPP 2514; the European Research Council through DebuQC; and PraktiQOM.

AI usage declaration. The conceptualization of the project and the identification and selection of gaps in the literature were carried out by the authors in March 2026. ChatGPT 5.5 was subsequently used to assist in closing technical gaps in preliminary proof attempts. More recently, GPT-5.6 Sol and GPT-6 were used to strengthen several of our technical results, and to help check the mathematical arguments and improve the exposition. All results are verified by the authors, who take full responsibility for the content of this work.

Note added. The analysis of the high-temperature regime in Hamiltonian learning of Gaussian states will be considered, with methods related to our trace distance bound, in an independent work [EMD26]. We thank the authors for communicating with us about this.

One day before we submitted this manuscript, an independent preprint [Rub26] appeared on arXiv that reports lower bounds related to ours.

## 3 Additional notations and preliminaries

Facts from representation theory. For $X \in \operatorname { G L } _ { n } ( \mathbb { C } )$ , we denote by $\Gamma _ { m } ( X )$ the symmetric power representation of X on $\mathcal { H } _ { m } ^ { ( n ) } \equiv \mathrm { S y m } ^ { m } ( \mathbb { C } ^ { n } )$ , and by Γ(X) the direct sum of actions for all $m ,$ , i.e., $\Gamma ( X ) : = \bigoplus _ { m = 0 } ^ { \infty } \Gamma _ { m } ( X )$ . In particular, for $U \in \mathbb { U } ( n )$ , Γ(U) is the passive Gaussian unitary associated with U, whose action on the annihilation operators is specified by $\Gamma ( U ) ^ { \dagger } a _ { i } \Gamma ( U ) = \textstyle \sum _ { j } U _ { i j } a _ { j }$ . The projector onto $\mathcal { H } _ { m } ^ { ( n ) }$ is given by $P _ { m } ^ { ( n ) } = \Gamma _ { m } ( \mathbb { I } )$

More generally, let λ be a partition of m of size no more than n sorted in a non-increasing order. We use $S _ { \lambda } ^ { ( n ) } \subseteq \mathbb { C } ^ { n }$ to denote the Weyl module associated with λ, which is an irreducible representation of ${ \mathrm { G L } } _ { n } ( \mathbb { C } )$ . Formally, $S _ { \lambda } ^ { ( n ) }$ can be defined as the image of the Young symmetrizer $c _ { \gamma }$ in $\mathbb { C } ^ { n }$ . See [FH13, Lecture 6]. We denote by $\Gamma _ { \lambda } ( X )$ ) the natural induced action of $X \in \operatorname { G L } _ { n } ( \mathbb { C } )$ on $S _ { \lambda } ^ { ( n ) }$ , and by $P _ { \lambda } ^ { ( n ) } = \Gamma _ { \lambda } ( \mathbb { I } )$ the projector onto $S _ { \lambda } ^ { ( n ) }$

Facts about passive Gaussian states. We use $\tau _ { \nu }$ to denote a single-mode thermal state with average photon number ν. In the Fock basis, $\tau _ { \nu }$ can be expressed as

$$
\tau _ { \nu } = \frac { 1 } { \nu + 1 } \sum _ { m = 0 } ^ { \infty } \left( \frac { \nu } { \nu + 1 } \right) ^ { m } | m \rangle \langle m | .\tag{14}
$$

We will use the photon-number sector decomposition of the n-mode bosonic Hilbert space,

$$
\mathcal { H } ^ { ( n ) } = \bigoplus _ { m = 0 } ^ { \infty } \mathcal { H } _ { m } ^ { ( n ) } ,\tag{15}
$$

where $\mathcal { H } _ { m } ^ { ( n ) } \cong \mathrm { S y m } ^ { m } ( \mathbb { C } ^ { n } )$ is the subspace of the Hilbert space with exactly m photons. The superscript (n) is sometimes omitted when there is no ambiguity.

A passive linear optical transformation, or a passive Gaussian unitary, is described by a unitary $U =$ $( u _ { i j } ) \in \mathrm { U } ( n )$ with the following action on the annihilation operators:

$$
U ^ { \dagger } \hat { a } _ { i } U = \sum _ { j = 1 } ^ { n } u _ { i j } \hat { a } _ { j } , \quad \mathrm { f o r } i = 1 , \cdots , n .\tag{16}
$$

View this as a representation of $\mathbb { U } ( n )$ on the n-mode bosonic Hilbert space $\mathcal { H } ^ { ( n ) }$ , it is known that Eq. (15) is the decomposition into irreducible representations. That is, $\mathbb { U } ( n )$ acts irreducibly on each photon-number sector $\mathcal { H } _ { m } ^ { ( n ) }$

A passive Gaussian state is a Gaussian state with zero mean and covariance matrix Σ satisfying $\Sigma =$ $K D K ^ { T }$ , where K is a symplectic orthogonal matrix and $D =$ diag $\{ d _ { 1 } , \cdot \cdot \cdot , d _ { n } , d _ { 1 } , \cdot \cdot \cdot , d _ { n } \} \ge 1 / 2 \mathbb { I }$ The average total photon number of the state is given by $\textstyle \sum _ { i = 1 } ^ { n } ( d _ { i } - 1 / 2 )$ . All passive Gaussian states are block-diagonal in the photon-number sector decomposition. This is, $\rho = \bigoplus _ { m = 0 } ^ { \infty } p _ { m } \rho _ { m } .$ , where $\rho _ { m }$ is a density matrix supported on $\mathcal { H } _ { m } ^ { ( n ) }$ and $p _ { m }$ is the probability of having m photons. One way to see this is to note that any passive Gaussian state can be generated by applying a passive Gaussian unitary to a product of single-mode thermal states. The latter are block-diagonal in the photon-number sector decomposition, and the passive Gaussian unitary preserves the total photon number.

An alternative way to represent a passive Gaussian state is via the photon number occupation matrix $\mathsf { N } \in \mathbb { C } ^ { n \times n }$ , defined as [HW99]

$$
\mathsf { N } _ { i j } : = \mathrm { T r } ( \rho \hat { a } _ { j } ^ { \dagger } \hat { a } _ { i } )\tag{17}
$$

A simple relation exists between the photon number occupation matrix N and the covariance matrix Σ:

$$
\Sigma = \frac { 1 } { 2 } \mathbb { I } _ { 2 n } + \Phi _ { \mathrm { r e } } ( { \mathsf { N } } ) \equiv \frac { 1 } { 2 } \mathbb { I } _ { 2 n } + \left( \begin{array} { c c } { { \mathrm { R e } } { \mathsf { N } } } & { { - \mathrm { I m } } { \mathsf { N } } } \\ { { \mathrm { I m } } { \mathsf { N } } } & { { \mathrm { R e } } { \mathsf { N } } } \end{array} \right) .\tag{18}
$$

We refer to $\Phi _ { \mathrm { r e } } : \mathbb { C } ^ { n \times n }  \mathbb { R } ^ { 2 n \times 2 n }$ as the realification map. It is also the natural isomorphism from $\mathbb { U } ( n )$ to $\mathrm { S P } ( 2 n ) \cap \mathrm { O } ( 2 n )$ . The correctness of the above equation can be verified by noticing that $\langle \hat { a } _ { i } \hat { a } _ { j } \rangle =$ $\langle \hat { a } _ { i } ^ { \dagger } \hat { a } _ { j } ^ { \dagger } \rangle = 0$ for all passive Gaussian states and then directly use the relation between $( \hat { p } _ { i } , \hat { q } _ { i } )$ and $\hat { a } _ { i }$ .

## 4 k-copy lower bound for cold Gaussian states

The goal of this section is to establish theorem 2 from the main text, whose proof is given in theorem 4 below. The proof is a black-box reduction from finite-dimensional tomography to passive Gaussian-state tomography. The reduction has three ingredients. First, we encode an n-dimensional state $\sigma$ into a passive Gaussian state $\rho _ { \sigma }$ with low mean total photon number. Second, we show that $\sigma$ can be stably recovered from $\rho _ { \sigma }$ . Third, we show how to generate one copy of $\rho _ { \sigma }$ exactly from a random number of copies of $\sigma .$ To combine these ingredients, we group independent simulation trials into batches, with the finite-dimensional input size chosen before each batch is measured. A k-copy Gaussian learner using N copies then induces a finite-dimensional learner with $O ( N )$ total copies and with a controlled "expected weighted block $\cos \mathrm { t } ^ { \prime \prime }$ (as defined below in Lemma 1). The result follows from the finite-dimensional lower bound in [KLMR26, Eq. (40)] (recalled below), which is a refinement of [CLL24].

We work with the k-copy measurement model for quantum state tomography introduced in [CLL24], defined as follows. The input copies are processed in blocks of at most k fresh copies. At each round, the protocol applies a POVM to the current block, and the choice of POVM may depend on all previous classical outcomes. No quantum system is retained from one round to the next. Thus, the protocol may be arbitrarily adaptive, but only through classical information. This coincides with the model of separate, adaptively chosen t-entangled measurements in [CLL24, Theorem 1.2] and with [KLMR26, Definition 1.1].

We use the following weighted form of the finite-dimensional lower bound in [KLMR26, Eq. (40)].

Lemma 1 (Finite-dimensional lower bound with variable block sizes). There exist universal constants $\varepsilon _ { \mathrm { f d } } > 0$ and $d _ { 0 } \in \mathbb { N }$ such that the following holds. Let $d \geq d _ { 0 }$ and $0 < \varepsilon < \varepsilon _ { \mathrm { f d } }$ . Consider a protocol with a deterministic total copy budget $M \in \mathbb { N } .$ . In round $j ,$ it applies a POVM to an integer number $t _ { j } \geq 1$ of fresh copies. Both $t _ { j }$ and the POVM are chosen before accessing those copies, using only previous classical outcomes and internal randomness. No quantum memory is retained between rounds. The number of rounds $T$ may be random, but $\textstyle \sum _ { j = 1 } ^ { T } t _ { j } \leq M$ almost surely.

Suppose that, for every $\sigma \in \mathcal { D } ( \mathbb { C } ^ { d } )$ satisfying $\sigma \leq 2 \mathbb { I } _ { d } / d ,$ , the protocol outputs a state $\widehat { \sigma }$ such that

$$
\operatorname* { P r } _ { { \boldsymbol { \sigma } } } \left( \frac { 1 } { 2 } \| \widehat { \boldsymbol { \sigma } } - { \boldsymbol { \sigma } } \| _ { 1 } \leq \varepsilon \right) \geq \frac { 1 } { 2 } .
$$

Define the weight of a block of size t by $w _ { d } ( t ) : = t \sqrt { \operatorname* { m i n } \{ t , d ^ { 2 } \} }$ . Then

$$
\operatorname* { s u p } _ { \sigma \in \mathcal { D } ( \mathbb { C } ^ { d } ) } \mathbb { E } \left[ \sum _ { j = 1 } ^ { T } w _ { d } ( t _ { j } ) \right] = \Omega \left( \frac { d ^ { 3 } } { \varepsilon ^ { 2 } } \right) .\tag{19}
$$

Here $\mathrm { P r } _ { \sigma }$ and $\mathbb { E } \sigma$ are taken over the protocol’s measurement outcomes and internal randomness, with $\sigma$ held fixed. Thus, the supremum is over input states, whereas the expectation is over runs of the protocol on each fixed state. The sum includes every executed measurement round.

Proof. Let $\Delta \sim \mu$ have the prior from [KLMR26, Sec. 6.1], and set $\sigma _ { \Delta } : = \mathbb { I } _ { d } / d + \Delta$ . The prior is supported on traceless Hermitian matrices satisfying $\| \Delta \| _ { \mathrm { o p } } \leq C \varepsilon / d$ for a universal constant C. Choose $\varepsilon _ { \mathrm { f d } }$ so that $C \varepsilon _ { \mathrm { f d } } \leq 1 / 2$ ; then $\Pi _ { d } / ( 2 d ) \le \sigma _ { \Delta } \le 3 \mathbb { I } _ { d } / ( 2 d ) \le 2 \mathbb { I } _ { d } / d .$ . Run the protocol on $\sigma _ { \Delta }$ , and let $Y$ be its full classical transcript, including internal randomness.

Since the protocol uses at most M copies in total, each executed block also satisfies $t _ { j } \leq M$ . Thus it fits Definition 1.1 of [KLMR26] with both the total copy budget and the allowed maximum block size set to M. This allows us to apply Proposition 7.1 of [KLMR26]. We use its first inequality in Eq. (57) of [KLMR26], which retains the actual block sizes $t _ { j }$ rather than replacing them by the upper bound M. Combining this with Corollary 6.2 of [KLMR26], which allows success probability $1 / 2$ , gives

$$
\Omega ( d ^ { 2 } ) \leq I ( \Delta ; Y ) \leq O \left( \frac { \varepsilon ^ { 2 } } { d } \underset { \Delta \sim \mu } { \mathbb { E } } \left[ \underset { \sigma _ { \Delta } } { \mathbb { E } } \left[ \sum _ { j = 1 } ^ { T } w _ { d } ( t _ { j } ) \right] \right] \right) .
$$

Here only the outer expectation averages over the prior; the inner expectation is over the protocol run on the fixed state $\sigma _ { \Delta }$ . Since every $\sigma _ { \Delta }$ satisfies the promise, the prior average is at most the supremum in (19). Rearranging proves the claim. □

We now develop our argument, showing how n-dimensional mixed state tomography reduces to n-mode passive Gaussian-state tomography. Let us start with the following characterization of passive Gaussian states (see also [DG13, Sec. 17.1.4]):

Lemma 2 (Fugacity matrix representation). There is a bijection between n-mode passive Gaussian states $\rho$ and $n { - } b y { - } n$ Hermitian matrices $0 \leq { \mathsf { R } } < { \mathbb { I } }$

Given R, which we call the fugacity matrix, the corresponding passive Gaussian state is

$$
\rho = \operatorname* { d e t } ( \mathbb { I } - \mathbb { R } ) \Gamma ( \mathbb { R } ) ,\tag{20}
$$

where $\Gamma ( X ) : = \bigoplus _ { m = 0 } ^ { \infty } \Gamma _ { m } ( X )$ and $\Gamma _ { m } ( X ) : = X ^ { \otimes m } | _ { \mathrm { S y m } ^ { m } ( \mathbb { C } ^ { n } ) }$ , with $\Gamma _ { 0 } ( X ) : = 1$

Conversely, if $P _ { 0 }$ and $P _ { 1 }$ denote the projectors onto the vacuum and one-photon sectors, respectively, then

$$
{ \sf R } = \frac { P _ { 1 } \rho P _ { 1 } } { \mathrm { T r } ( P _ { 0 } \rho ) } ,
$$

where the one-photon sector is naturally identified with $\mathbb { C } ^ { n }$

Moreover, the mean total photon number of $\rho$ is

$$
\begin{array} { r } { \mathrm { T r } ( \rho \hat { n } ) = \mathrm { T r } [ { \sf R } ( { \mathbb I } - { \sf R } ) ^ { - 1 } ] . } \end{array}\tag{21}
$$

Proof. Consider the spectral decomposition $\mathsf { R } = U \mathrm { d i a g } ( r _ { 1 } , \ldots , r _ { n } ) U ^ { \dagger }$ , where $r _ { j } \in [ 0 , 1 )$ for every $j$ Using the multiplicativity of Γ and the action of $\Gamma ( \operatorname { d i a g } ( r _ { 1 } , \dots , r _ { n } ) )$ on the Fock basis, we obtain

$$
\begin{array} { l } { \displaystyle \operatorname* { d e t } ( \mathbb { I } - \mathsf { R } ) \Gamma ( \mathsf { R } ) \overset { \mathrm { ( i ) } } { = } \Gamma ( U ) \left( \displaystyle \sum _ { m \in \mathbb { N } ^ { n } } \prod _ { j = 1 } ^ { n } ( 1 - r _ { j } ) r _ { j } ^ { m _ { j } } | m \rangle \langle m | \right) \Gamma ( U ) ^ { \dag } } \\ { \displaystyle \overset { \mathrm { ( i i ) } } { = } \Gamma ( U ) \bigotimes _ { j = 1 } ^ { n } \left( ( 1 - r _ { j } ) \displaystyle \sum _ { \ell = 0 } ^ { \infty } r _ { j } ^ { \ell } | \ell \rangle \langle \ell | \right) \Gamma ( U ) ^ { \dag } } \\ { \displaystyle \overset { \mathrm { ( i i i ) } } { = } \Gamma ( U ) \bigotimes _ { j = 1 } ^ { n } \tau _ { \frac { r _ { j } } { 1 - r _ { j } } } \Gamma ( U ) ^ { \dag } . } \end{array}\tag{22}
$$

Here, (i) uses the spectral decomposition of $\mathsf { R } , \Gamma ( U ^ { \dagger } ) = \Gamma ( U ) ^ { \dagger }$ , and $\begin{array} { r } { \operatorname* { d e t } ( \mathbb { I } - \mathbb { R } ) = \prod _ { i = 1 } ^ { n } ( 1 - r _ { j } ) \colon } \end{array}$ ; (ii) factorizes the sum over the Fock basis; and (iii) uses the definition of a single-mode thermal state. Thus, every Hermitian matrix $0 \leq { \mathsf { R } } < { \mathbb { I } }$ defines a passive Gaussian state.

Moreover, since $\Gamma ( U )$ preserves the total photon number and $\tau _ { \nu }$ has mean photon number $\nu ,$ the last line gives $\textstyle \operatorname { T r } ( \rho { \hat { n } } ) = \sum _ { i = 1 } ^ { n } r _ { j } / ( 1 - r _ { j } ) = \operatorname { T r } [ \operatorname { R } ( \mathbb { I } - \mathbb { R } ) ^ { - 1 } ]$

Conversely, as recalled after $\operatorname { E q } .$ . (16), every passive Gaussian state can be written as

$$
\rho = \Gamma ( U ) \bigotimes _ { j = 1 } ^ { n } \tau _ { \nu _ { j } } \Gamma ( U ) ^ { \dagger }\tag{23}
$$

for some $U \in \mathbb { U } ( n )$ and $\nu _ { j } \geq 0$ . Define $r _ { j } : = \nu _ { j } / ( 1 + \nu _ { j } )$ and $\mathsf { R } : = U \mathrm { d i a g } ( r _ { 1 } , \ldots , r _ { n } ) U ^ { \dagger }$ . Then $r _ { j } \in [ 0 , 1 )$ and $r _ { j } / ( 1 - r _ { j } ) = \nu _ { j }$ , so reversing the calculation above gives $\rho = \operatorname* { d e t } ( \mathbb { I } - \mathbb { R } ) \Gamma ( \mathbb { R } )$

Finally, since $\Gamma _ { 0 } ( \mathsf { R } ) = 1$ and $\Gamma _ { 1 } ( \mathsf { R } ) = \mathsf { R }$ , we have $\mathrm { T r } ( P _ { 0 } \rho ) = \operatorname* { d e t } ( \mathbb { I } - \mathbb { R } )$ and $P _ { 1 } \rho P _ { 1 } = \operatorname* { d e t } ( \mathbb { I } - \mathsf { R } ) \mathsf { R }$ Hence $\mathsf { R } = P _ { 1 } \rho P _ { 1 } / \operatorname { T r } ( P _ { 0 } \rho )$ , which proves uniqueness and therefore the claimed bijection. □

We now use the fugacity representation to embed n-dimensional states into passive Gaussian states. Our key idea is that any n-dimensional density matrix σ can be encoded into an n-mode passive Gaussian state with low mean photon number, so that the corresponding n-mode Gaussian state behaves similarly to the encoded n-dimensional state. We will use the following encoding throughout this section.

Definition 3 (Encoding of a finite-dimensional state into a passive Gaussian state). For every ndimensional state $\sigma \in { \mathcal { D } } ( \mathbb { C } ^ { n } )$ , we denote by $\rho _ { \sigma }$ the n-mode passive Gaussian state with fugacity matrix $\mathsf { R } = \sigma / 4$ , namely $\rho _ { \sigma } : = \operatorname* { d e t } ( \mathbb { I } - \sigma / 4 ) \Gamma ( \sigma / 4 )$

This definition is well posed: since $0 \le \sigma \le \mathbb { I }$ , we have $0 \leq \sigma / 4 < \mathbb { I }$ , and hence lemma 2 guarantees that $\rho _ { \sigma }$ is a valid passive Gaussian state. Moreover, by lemma $2 , \operatorname { T r } ( \rho _ { \sigma } \hat { n } ) = \operatorname { T r } [ ( \sigma / 4 ) ( \mathbb { I } - \sigma / 4 ) ^ { - 1 } ] \leq$ $\textstyle { \frac { 1 } { 3 } } \operatorname { T r } \sigma = { \frac { 1 } { 3 } }$ . Thus, the encoded state has low mean total photon number.

We next show that the encoding can be stably inverted. The normalized state in the one-photon sector of $\rho _ { \sigma }$ is exactly $\sigma ,$ and this remains true up to a controlled error if $\rho _ { \sigma }$ is replaced by a nearby state.

Lemma 4 (Recovering σ from $\rho _ { \sigma } )$ . Let $\sigma ~ \in ~ { \mathcal { D } } ( \mathbb { C } ^ { n } )$ , let $\widehat { \rho } \in \mathcal { D } ( \mathcal { H } ^ { ( n ) } )$ , and let $0 < \varepsilon \leq 1 / 1 0 0 .$ . If $\begin{array} { r } { \frac { 1 } { 2 } \| \widehat { \rho } - \rho _ { \sigma } \| _ { 1 } \leq \varepsilon . } \end{array}$ , then $\widehat { p } : = \operatorname { T r } ( P _ { 1 } \widehat { \rho } ) > 0 .$ , and the normalized one-photon block

$$
\widehat { \sigma } : = \frac { P _ { 1 } \widehat { \rho } P _ { 1 } } { \widehat { p } }
$$

satisfies $\begin{array} { r } { \frac 1 2 \| \widehat { \sigma } - \sigma \| _ { 1 } \leq 1 6 \varepsilon } \end{array}$

In particular, if $\ b \sigma ^ { \prime } \in \mathcal { D } ( \mathbb { C } ^ { n } )$ and $\begin{array} { r } { \frac { 1 } { 2 } \| \rho _ { \sigma } - \rho _ { \sigma ^ { \prime } } \| _ { 1 } \leq \varepsilon , } \end{array}$ then $\begin{array} { r } { \frac 1 2 \| \sigma - \sigma ^ { \prime } \| _ { 1 } \le 1 6 \varepsilon } \end{array}$

Proof. Let $p _ { \sigma } : = \mathrm { T r } ( P _ { 1 } \rho _ { \sigma } )$ . Since $\Gamma _ { 1 } ( \sigma / 4 ) = \sigma / 4$ , we have $P _ { 1 } \rho _ { \sigma } P _ { 1 } = p _ { \sigma } \sigma _ { . }$ , where $\begin{array} { r } { p _ { \sigma } = \frac { 1 } { 4 } \operatorname* { d e t } ( \mathbb { I } - \sigma / 4 ) } \end{array}$ If $\lambda _ { 1 } , \ldots , \lambda _ { n }$ are the eigenvalues of $\sigma ,$ then de $\begin{array} { r } { ( \mathbb { I } - \sigma / 4 ) = \prod _ { j } ( 1 - \lambda _ { j } / 4 ) \ge 1 - \frac { \hat { 1 } } { 4 } \sum _ { j } \lambda _ { j } = 3 / 4 } \end{array}$ Hence $p _ { \sigma } \geq 3 / 1 6$

By Hölder’s inequality for Schatten norms,

$$
\| P _ { 1 } \widehat { \rho } P _ { 1 } - p _ { \sigma } \sigma \| _ { 1 } \leq \| P _ { 1 } \| _ { \infty } ^ { 2 } \| \widehat { \rho } - \rho _ { \sigma } \| _ { 1 } \leq 2 \varepsilon .
$$

Taking the trace and using $\left| \operatorname { T r } X \right| \leq \left\| X \right\| _ { 1 }$ gives $| \widehat { p } - p _ { \sigma } | \leq 2 \varepsilon$ . Using the assumption $\varepsilon \le 1 / 1 0 0$ , we

have $\widehat { p } \geq 3 / 1 6 - 2 \varepsilon > 1 / 8$ , so $\widehat { \sigma }$ is well defined. Moreover, adding and subtracting $p _ { \sigma } \sigma / { \widehat p } ,$ we obtain

$$
\begin{array} { r l } & { \| \widehat { \sigma } - \sigma \| _ { 1 } = \left\| \frac { P _ { 1 } \widehat { \rho } P _ { 1 } } { \widehat { p } } - \sigma \right\| _ { 1 } } \\ & { \qquad \leq \frac { \| P _ { 1 } \widehat { \rho } P _ { 1 } - p _ { \sigma } \sigma \| _ { 1 } } { \widehat { p } } + \left| \frac { p _ { \sigma } } { \widehat { p } } - 1 \right| \| \sigma \| _ { 1 } } \\ & { \qquad = \frac { \| P _ { 1 } \widehat { \rho } P _ { 1 } - p _ { \sigma } \sigma \| _ { 1 } + | p _ { \sigma } - \widehat { p } | } { \widehat { p } } } \\ & { \qquad \leq \frac { 4 \varepsilon } { \widehat { \mathcal { P } } } \leq 3 2 \varepsilon , } \end{array}
$$

where we used $\| \sigma \| _ { 1 } = 1$ , the bounds above, and $\widehat { p } > 1 / 8$ . Thus $\begin{array} { r } { \frac 1 2 \| \widehat { \sigma } - \sigma \| _ { 1 } \leq 1 6 \varepsilon } \end{array}$

The last claim follows by taking $\widehat \rho = \rho _ { \sigma ^ { \prime } }$ , whose normalized one-photon block is exactly $\sigma ^ { \prime }$

We now show that the encoded passive Gaussian state $\rho _ { \sigma }$ can be generated exactly from copies of the original finite-dimensional state $\sigma$ . The construction is simple: we sample a photon number, project the corresponding number of copies of σ onto the symmetric subspace, and repeat only if the projection fails. The sampling probabilities are chosen so that the first successful trial outputs exactly $\rho _ { \sigma }$

We state the following lemma mainly to present the simulation algorithm in a clean and self-contained way. The lemma itself is not used as a black box in the proof of the k-copy lower bound; there, we rely on more detailed properties of the simulation than just the expectation bound stated below.

Lemma 5 (Simulate $\rho _ { \sigma }$ with copies of $\sigma )$ . For every $\sigma \in { \mathcal { D } } ( \mathbb { C } ^ { n } )$ , there is an algorithm (independent of σ) that exactly outputs one copy of $\rho _ { \sigma }$ using a random number T of copies of $\sigma ,$ with $\begin{array} { r } { \mathbb { E } _ { \sigma } T \le \frac { 4 } { 9 } . } \end{array}$

Proof. Set $d _ { \sigma } : = \operatorname* { d e t } ( \mathbb { I } - \sigma / 4 )$ . By the definition of $\rho _ { \sigma }$ and the homogeneity $\Gamma _ { \ell } ( \sigma / 4 ) = 4 ^ { - \ell } \Gamma _ { \ell } ( \sigma )$

$$
\rho _ { \sigma } = d _ { \sigma } \bigoplus _ { \ell = 0 } ^ { \infty } 4 ^ { - \ell } \Gamma _ { \ell } ( \sigma ) .\tag{24}
$$

More generally, taking the trace of the fugacity representation with fugacity matrix xσ gives

$$
\sum _ { \ell = 0 } ^ { \infty } x ^ { \ell } \operatorname { T r } \Gamma _ { \ell } ( \sigma ) = \operatorname * { d e t } ( \mathbb { I } - x \sigma ) ^ { - 1 } , \qquad 0 \leq x < 1 .\tag{25}
$$

A single trial of the simulator proceeds as follows. Sample an integer $L \geq 0$ according to $\Pr ( L = \ell ) =$ $\frac { 3 } { 4 } 4 ^ { - \ell } . \mathrm { I f } \ L = 0 \nonumber$ , regard the trial as successful and output the vacuum without consuming any copy of $\sigma . \mathrm { I f } \ L \geq 1$ , take L fresh copies of $\sigma$ and project them onto $\operatorname { S y m } ^ { L } ( \mathbb { C } ^ { n } )$ . If the projection succeeds, output the resulting state, identifying $\operatorname { S y m } ^ { L } ( \mathbb { C } ^ { n } )$ with the L-photon sector $\mathcal { H } _ { L } ^ { ( n ) }$ ; otherwise, discard these copies and start a new trial.

For a fixed value $L = \ell ,$ the unnormalized post-measurement state is $\Gamma _ { \ell } ( \sigma )$ , so the projection succeeds with probability Tr $\Gamma _ { \ell } ( \sigma )$ . Hence, using (25) with $x = 1 / 4$ , the success probability of one trial is

$$
q _ { \sigma } = \frac { 3 } { 4 } \sum _ { \ell = 0 } ^ { \infty } 4 ^ { - \ell } \mathrm { T r } \Gamma _ { \ell } ( \sigma ) = \frac { 3 } { 4 d _ { \sigma } } .
$$

Conditioned on success, and after discarding the classical value of $L ,$ the output state of one trial is

$$
\frac { 1 } { q _ { \sigma } } \frac { 3 } { 4 } \bigoplus _ { \ell = 0 } ^ { \infty } 4 ^ { - \ell } \Gamma _ { \ell } ( \sigma ) = d _ { \sigma } \bigoplus _ { \ell = 0 } ^ { \infty } 4 ^ { - \ell } \Gamma _ { \ell } ( \sigma ) = \rho _ { \sigma } .
$$

Since $d _ { \sigma } \leq 1$ , we have $q _ { \sigma } \ge 3 / 4$

Moreover, let R denote the index of the first successful trial, and let $L _ { j }$ be the number of copies of σ sampled in the j-th trial. The total number of copies used by the simulator is therefore

$$
T = \sum _ { j = 1 } ^ { R } L _ { j } = \sum _ { j \geq 1 } L _ { j } \mathbb { 1 } _ { \{ R \geq j \} } .
$$

The event $\{ R \geq j \}$ means that the first $j - 1$ trials have all failed. It therefore depends only on the randomness of the preceding trials and is independent of the fresh length $L _ { j }$ sampled in trial j. Hence

$$
{ \underset { \sigma } { \mathbb { E } } } T = \sum _ { j \geq 1 } { \underset { \sigma } { \mathbb { E } } } { \Big [ } L _ { j } \mathbb { 1 } _ { \{ R \geq j \} } { \Big ] } = \sum _ { j \geq 1 } { \underset { \sigma } { \operatorname* { P r } } } ( R \geq j ) \ { \underset { \varepsilon } { \mathbb { E } } } L .
$$

Since every trial succeeds independently with probability $q _ { \sigma }$ , we have $\operatorname* { P r } _ { \sigma } ( R \geq j ) = ( 1 - q _ { \sigma } ) ^ { j - 1 }$ Therefore, using E $L = 1 / 3 _ { \mathrm { { ; } } }$

$$
\frac { \mathbb { E } } { \sigma } T = \mathbb { E } L \sum _ { j \geq 1 } ( 1 - q _ { \sigma } ) ^ { j - 1 } = \frac { \mathbb { E } L } { q _ { \sigma } } = \frac { 1 / 3 } { 3 / ( 4 d _ { \sigma } ) } = \frac { 4 } { 9 } d _ { \sigma } \leq \frac { 4 } { 9 } .
$$

Since $q _ { \sigma } \ge 3 / 4$ , the first successful trial is reached almost surely. Together with the calculation above, this proves the claim. □

We now prove the main result of this section, stated in Theorem 4, via the black-box reduction below, whose main idea is as follows. Suppose that $\mathcal { A }$ is a k-copy protocol that learns every passive Gaussian state in the promised cold family using N copies. We use A as a subroutine to learn an unknown n-dimensional state $\sigma ,$ so that the finite-dimensional lower bound in lemma 1 can be applied. Starting from copies of $\sigma ,$ we generate the copies of $\rho _ { \sigma }$ requested by ${ \mathcal { A } } ,$ run each measurement block of A on the simulated inputs, and finally recover $\sigma$ from the one-photon block of the estimate returned by ${ \mathcal { A } } .$ . The only complication is that generating a copy of the passive Gaussian state $\rho _ { \sigma }$ requires a random number of copies of σ. We address this issue carefully in the proof below.

Thus, finite-dimensional tomography reduces to passive Gaussian-state tomography: an unexpectedly eficient Gaussian learner would give an unexpectedly eficient finite-dimensional learner. The reduction is summarized in fig. 3.

Theorem 4 (Few-copy lower bound for cold Gaussian states). There exist universal constants $\varepsilon _ { 0 } > 0$ and $n _ { 0 } \in \mathbb { N }$ such that thefollowing holds. Let $n \geq n _ { 0 } , 0 < \varepsilon < \varepsilon _ { 0 } ,$ , and let $k \geq 1$ be any integer. Any possibly adaptive protocol that learns every n-mode Gaussian state to trace-distance error ε with probability at least $2 / 3 ,$ , using only k-copy measurements, requires

$$
N = \Omega \left( \frac { n ^ { 3 } } { \varepsilon ^ { 2 } \operatorname* { m i n } \{ \sqrt { k } , n \} } \right)\tag{26}
$$

copies. This remains true under the promise that the unknown state is passive and its covariance matrix satisfies $\begin{array} { r } { \Sigma \leq ( \frac { 1 } { 2 } + \frac { 1 } { 2 n - 1 } )  { \mathrm { I I } _ { 2 n } } } \end{array}$

Proof. Let $d _ { 0 }$ and $\varepsilon _ { \mathrm { f d } }$ be as in lemma 1. Choose ${ n _ { 0 } } ~ \ge ~ \operatorname* { m a x } \{ d _ { 0 } , 2 \}$ and $0 < \varepsilon _ { 0 } \le 1 / 1 0 0$ such that $1 6 \varepsilon _ { 0 } \leq \varepsilon _ { \mathrm { f d } }$ . Let A be a possibly adaptive k-copy learner for the promised passive Gaussian states, using at most N copies and achieving trace-distance error ε with probability at least $2 / 3$ . We use it to learn an unknown $\sigma \in { \mathcal { D } } ( \mathbb { C } ^ { n } )$ satisfying $\sigma \leq 2 \mathbb { I } _ { n } / n$ , following fig. 3. First, $\rho _ { \sigma }$ belongs to the promised Gaussian family. Indeed, its fugacity matrix is $\mathsf { R } = \sigma / 4 \leq \mathbb { I } _ { n } / ( 2 n )$ . By the thermal normal form in lemma $^ { 2 , }$ its ordinary covariance eigenvalues are $\begin{array} { r } { \frac { 1 } { 2 } + r _ { j } / ( 1 - r _ { j } ) } \end{array}$ , each repeated twice, where $r _ { j }$ are the eigenvalues of R. Hence

<table><tr><td colspan="2">Black-box reduction</td><td rowspan="2">passive Gaussian state ρσ estimate ô</td></tr><tr><td colspan="2">finite-dimensional state σ,  $\sigma { \in } \widecheck { D } ( \mathbb { C } ^ { n } )$ </td></tr><tr><td>1. Simulate</td><td>R=σ/4 Starting from copies of the unknown state</td><td>finite-dimensional tomography  $\sigma ,$  generate exact simulated copies of its</td></tr><tr><td>2. Learn</td><td rowspan="3">passive Gaussian encoding 5  $\rho _ { \sigma }$  proof below. Apply the k-copy Gaussian tomography protocol A to the simulated copies of obtaining an estimate  ${ \widehat { \rho } } .$ </td><td rowspan="3">using the construction in lemma  $^ { 5 , }$  as detailed in the  $\rho _ { \sigma }$   ${ \widehat { \rho } } ,$ </td></tr><tr><td>3. Recover</td></tr><tr><td>Return the normalized one-photon block of σ =</td></tr><tr><td rowspan="2">By lemma Result</td><td rowspan="2">Tr(P1ρ)  $^ { 4 , }$  an ε-accurate estimate of</td></tr><tr><td> $\rho _ { \sigma }$  yields an  $O ( \varepsilon )$  -accurate estimate of  $\sigma .$  2 dimensional learner must satisfy the lower bound in lemma 1, thus implying the corresponding lower bound in the theorem below in the (passive) Gaussian setting.</td></tr><tr><td>dimensional learner using weighted cost  $O ( N$  min</td><td>A Gaussian learner using N copies and k-copy measurements induces a finite-  $O ( N )$  copies and measurement blocks with total expected  $\{ { \sqrt { k } } , n \}$  ), uniformly over the promised inputs. This finite-</td></tr></table>

Figure 3: Main steps of the black-box reduction in theorem 4 from finite-dimensional tomography to passive Gaussian-state tomography. A tomography protocol for the encoded passive Gaussian state $\rho _ { \sigma }$ can be used as a subroutine to learn the underlying finite-dimensional state $\sigma .$

$$
\Sigma _ { \rho _ { \sigma } } \leq \left( \frac { 1 } { 2 } + \frac { 1 } { 2 n - 1 } \right) \mathbb { I } _ { 2 n } .
$$

To simulate ${ \mathcal { A } } ,$ we must choose each finite-dimensional block size before accessing its fresh inputs. We therefore group the single trials from lemma 5 into batches. Recall that one trial samples $L \geq 0$ with $\operatorname* { P r } ( L = { \overline { { \ell } } } ) = { \overset { \cdot } { \frac { 3 } { 4 } } } 4 ^ { - \ell }$ and projects L fresh copies of $\sigma$ onto $\operatorname { S y m } ^ { L } ( \mathbb { C } ^ { n } )$ , with $L = 0$ counted as a success and producing the vacuum. Its success probability is $q _ { \sigma } = 3 / [ 4 \operatorname* { d e t } ( \mathbb { I } - \sigma / 4 ) ] \geq 3 / 4$ , and its subnormalized successful output, averaged over $L ,$ is $q _ { \sigma } \rho _ { \sigma }$ . All independence statements below are for a fixed input $\sigma .$ Whenever A requests a measurement on $1 \leq b \leq k$ Gaussian copies, sample 2b independent lengths $L _ { 1 } , \dots , L _ { 2 b }$ and set

$$
t : = \operatorname* { m a x } \left\{ 1 , \sum _ { a = 1 } ^ { 2 b } L _ { a } \right\} .
$$

Take t fresh copies and perform the trials on disjoint groups, discarding the extra copy when all lengths vanish. If at least b trials succeed, apply the requested POVM to the first b successful outputs and pass only its outcome to ${ \mathcal { A } } .$ Otherwise, discard the entire batch and repeat without advancing A. Every quantum system is discarded at the end of each batch. For each preselected length vector, these operations form one POVM on t fresh copies, so the construction respects the block model. Let $p _ { b } ( \sigma )$ be the probability that a batch succeeds. Since each trial succeeds with probability $q _ { \sigma } \ge 3 / 4$ , it fails with probability at most $1 / 4$ . Let $F$ denote the number of failed trials among the 2b trials in the batch. By linearity of expectation,

$$
\mathbb { E } F = 2 b ( 1 - q _ { \sigma } ) \leq \frac { b } { 2 } .
$$

The batch fails precisely when fewer than b trials succeed, or equivalently when $F \geq b + 1$ . Markov’s inequality therefore gives

$$
1 - p _ { b } ( \sigma ) = \operatorname* { P r } ( F \geq b + 1 ) \leq \frac { \mathbb { E } F } { b + 1 } \leq \frac { b } { 2 ( b + 1 ) } \leq \frac { 1 } { 2 } .
$$

Thus every batch succeeds with probability at least $1 / 2$

We next verify that, conditioned on success, the batch supplies exactly b independent copies of $\rho _ { \sigma }$ . Fix a particular success/failure pattern with s successful trials. For one trial, after averaging over its sampled length $L ,$ the subnormalized successful output is $q _ { \sigma } \rho _ { \sigma }$ , while the failure event has total probability $1 - q _ { \sigma }$ Since the 2b trials are independent, after discarding the failed outputs the corresponding subnormalized state on the s successful outputs is

$$
q _ { \sigma } ^ { s } ( 1 - q _ { \sigma } ) ^ { 2 b - s } \rho _ { \sigma } ^ { \otimes s } .
$$

Whenever $s \geq b ,$ , we keep the first b successful outputs and discard the remaining $s - b _ { i }$ leaving

$$
q _ { \sigma } ^ { s } ( 1 - q _ { \sigma } ) ^ { 2 b - s } \rho _ { \sigma } ^ { \otimes b } .
$$

Summing this expression over all success patterns with $s \geq b$ simply sums their probabilities, whose total is $p _ { b } ( \sigma )$ . Hence the subnormalized state retained by a successful batch is

$$
p _ { b } ( \sigma ) \rho _ { \sigma } ^ { \otimes b } ,
$$

and therefore, conditioned on batch success, the retained state is exactly $\rho _ { \sigma } ^ { \otimes b }$ . This identity holds after averaging over the sampled lengths; it need not hold after conditioning on the batch size t. We therefore do not pass the lengths or the trial success flags to A: its choices of block sizes and POVMs use only its own previous measurement outcomes and internal randomness. Repeating failed batches uses independent fresh inputs, so the first successful batch has this same conditional output state. Consequently, the classical record seen by A, and hence its final estimate, has exactly the same distribution as in a run on fresh copies of $\rho _ { \sigma }$

We first analyze the exact simulation without imposing a total copy cutof. Consider one batch used to simulate a request of b Gaussian copies. Let

$$
X : = \sum _ { a = 1 } ^ { 2 b } L _ { a } ,
$$

so that, by construction, $t = \operatorname* { m a x } \{ 1 , X \}$ . Since the $\cal { L } _ { a } { } ^ { \ ' } s$ are independent and satisfy E $L = 1 / 3$ and $\mathrm { V a r } ( L ) = 4 / 9$ , we have

$$
\operatorname { \mathbb { E } } X = { \frac { 2 b } { 3 } } , \qquad \operatorname { \mathbb { E } } X ^ { 2 } = \operatorname { V a r } ( X ) + ( \operatorname { \mathbb { E } } X ) ^ { 2 } = { \frac { 8 b } { 9 } } + { \frac { 4 b ^ { 2 } } { 9 } } .
$$

Moreover, $t \leq 1 + X$ and $t ^ { 2 } \le 1 + X ^ { 2 }$ . Therefore,

$$
\mathbb { E } t \le 1 + \frac { 2 b } { 3 } \le \frac { 5 b } { 3 } , \qquad \mathbb { E } t ^ { 2 } \le 1 + \frac { 4 b ^ { 2 } + 8 b } { 9 } \le \frac { 7 b ^ { 2 } } { 3 } ,
$$

where the last inequalities use $b \geq 1$

We next bound the weighted block cost appearing in lemma 1. Recall that

$$
w _ { n } ( t ) = t \sqrt { \operatorname* { m i n } \{ t , n ^ { 2 } \} } = \operatorname* { m i n } \{ t ^ { 3 / 2 } , n t \} .
$$

Using separately the two terms in this minimum gives

$$
\mathbb { E } w _ { n } ( t ) \leq \operatorname* { m i n } \{ \mathbb { E } t ^ { 3 / 2 } , n \mathbb { E } t \} .
$$

Since $x \mapsto x ^ { 3 / 4 }$ is concave, Jensen’s inequality gives

$$
\begin{array} { r } { \mathbb { E } t ^ { 3 / 2 } = \mathbb { E } ( t ^ { 2 } ) ^ { 3 / 4 } \leq ( \mathbb { E } t ^ { 2 } ) ^ { 3 / 4 } \leq C b ^ { 3 / 2 } , } \end{array}
$$

while the bound on E t gives

$$
n \mathbb { E } t \leq C n b .
$$

Combining the two estimates,

$$
\begin{array} { r } { \mathbb { E } w _ { n } ( t ) \leq C \operatorname* { m i n } \{ b ^ { 3 / 2 } , n b \} = C b \sqrt { \operatorname* { m i n } \{ b , n ^ { 2 } \} } , } \end{array}
$$

where $C > 0$ is a universal constant, which may change from line to line.

These estimates concern a single batch. A request for b Gaussian copies may require several independent batches before one succeeds. For the analysis, view these batches as an infinite independent sequence, of which only the batches up to the first success are actually performed. Let $R _ { b }$ be the index of the first successful batch, and let $t _ { \ell }$ be the size of batch ℓ in this sequence. Here $\mathbb { E } \sigma$ averages over the simulation’s randomness and measurement outcomes for a fixed input $\sigma ,$ while $\mathbb { E } t$ and $\mathbb { E } w _ { n } ( t )$ average over the lengths sampled for one fresh batch.

Since $\{ R _ { b } \geq \ell \}$ means that the first $\ell - 1$ batches have failed, it depends only on the previous batches and is therefore independent of the fresh batch size $t _ { \ell } .$ . Hence

$$
\underset { \sigma } { \mathbb { E } } \left[ \sum _ { \ell = 1 } ^ { R _ { b } } t _ { \ell } \right] = \sum _ { \ell \geq 1 } ( 1 - p _ { b } ( \sigma ) ) ^ { \ell - 1 } \underset { \ell } { \mathbb { E } } t = \frac { \underset { \mathbb { E } } { \mathbb { E } } t } { p _ { b } ( \sigma ) } \leq 2 \underset { \ell } { \mathbb { E } } t ,
$$

and similarly

$$
\mathbb { E } \left[ \sum _ { \ell = 1 } ^ { R _ { b } } w _ { n } ( t _ { \ell } ) \right] = \sum _ { \ell \ge 1 } ( 1 - p _ { b } ( \sigma ) ) ^ { \ell - 1 } \mathbb { E } w _ { n } ( t ) = \frac { \mathbb { E } w _ { n } ( t ) } { p _ { b } ( \sigma ) } \le 2 \mathbb { E } w _ { n } ( t ) .
$$

In summary, failed batches increase both the expected number of copies and the expected weighted cost of simulating one request by at most a factor of two.

We now sum these bounds over all the Gaussian measurement blocks requested by A. Let $b _ { i }$ denote the number of Gaussian copies used in the i-th request. Since $\mathcal { A }$ is a k-copy protocol using at most N copies in total, every request actually made satisfies $1 \leq b _ { i } \leq k _ { : }$ , and $\textstyle \sum _ { i } b _ { i } \leq N$ on every run. In particular, A makes at most $N$ requests.

For each request $i ,$ let $S _ { i }$ be the total number of finite-dimensional copies of σ used to simulate that request, including all failed batches before the first successful one. Similarly, let $W _ { i }$ be the sum of the weighted costs $w _ { n } ( t )$ of all batches used for that request. For bookkeeping, set $b _ { i } = S _ { i } = W _ { i } = 0$ after $\mathcal { A }$ terminates, and let all sums over i run from 1 to $N$

The size $b _ { i }$ may depend on previous outcomes and internal randomness. Let $H _ { i }$ denote the classical information available after A has chosen its i-th block and before its simulation starts, including any randomness used for that choice. Conditional on $H _ { i ; }$ , the size $b _ { i }$ and the requested POVM are fixed, and the simulation uses fresh copies and new independent randomness. The single-request bounds therefore give

$$
\mathbb { E } [ S _ { i } | H _ { i } ] \leq 2 \cdot \frac { 5 } { 3 } b _ { i } = \frac { 1 0 } { 3 } b _ { i } ,
$$

and

$$
\begin{array} { r } { \mathbb E [ W _ { i } | H _ { i } ] \le C b _ { i } \sqrt { \operatorname* { m i n } \{ b _ { i } , n ^ { 2 } \} } . } \end{array}
$$

The factor 2 in the first bound accounts for the possible failed batches before the first successful one; in the second bound the same factor has been absorbed into the universal constant C.

We can now add these bounds over all requests. Let

$$
S : = \sum _ { i } S _ { i }
$$

be the total number of finite-dimensional copies used by the exact simulation. By the law of total expectation and the conditional bounds above,

$$
\begin{array} { r l } & { \frac { \mathbb { E } } { \sigma } S = \frac { \mathbb { E } } { \sigma } \displaystyle \sum _ { i } S _ { i } } \\ & { \quad \quad = \displaystyle \sum _ { i } \mathbb { E } \left[ \mathbb { E } [ S _ { i } \mid H _ { i } ] \right] } \\ & { \quad \quad \le \frac { 1 0 } { 3 } \frac { \mathbb { E } } { \sigma } \displaystyle \sum _ { i } b _ { i } } \\ & { \quad \quad \le \frac { 1 0 N } { 3 } . } \end{array}
$$

Here the last inequality uses $\textstyle \sum _ { i } b _ { i } \leq N$ on every run. Similarly, since the total weighted cost is the sum of the $W _ { i } { ' } s _ { i }$

$$
\begin{array} { r l } {  { \frac { \mathbb { E } } { \sigma } \sum _ { j } w _ { n } ( t _ { j } ) = \frac { \mathbb { E } } { \sigma } \sum _ { i } W _ { i } } } \\ & { = \sum _ { i } \frac { \mathbb { E } } { \sigma } \Big [ \frac { \mathbb { E } [ W _ { i } \mid H _ { i } ] } { \sigma } \Big ] } \\ & { \le C \underbrace { \mathbb { E } } _ { \sigma } \sum _ { i } b _ { i } \sqrt { \operatorname* { m i n } \{ b _ { i } , n ^ { 2 } \} } } \\ & { \le C \sqrt { \operatorname* { m i n } \{ k , n ^ { 2 } \} } \ \frac { \mathbb { E } } { \sigma } \sum _ { i } b _ { i } } \\ & { \le C N \sqrt { \operatorname* { m i n } \{ k , n ^ { 2 } \} } . } \end{array}
$$

Here the second-to-last inequality uses $b _ { i } \leq k$ for every request. Notice that i indexes the Gaussian measurement blocks requested by ${ \mathcal { A } } ,$ whereas $j$ indexes all the finite-dimensional batches used to simulate them, including failed batches. We have therefore proved

$$
\begin{array} { c } { { \displaystyle \mathbb { E } S \le \frac { 1 0 N } { 3 } , } } \\ { { \displaystyle \mathbb { E } \sum _ { \sigma } w _ { n } ( t _ { j } ) \le C N \sqrt { \operatorname* { m i n } \{ k , n ^ { 2 } \} } . } } \end{array}\tag{27}
$$

Both bounds hold uniformly for every $\sigma \leq 2 \mathbb { I } _ { n } / n$ . In particular, the total copy cost $S$ is finite almost surely.

The only remaining problem is that the total number $S$ of finite-dimensional copies is so far controlled only in expectation, whereas lemma 1 requires a deterministic total copy budget. To fix this, set

$$
M _ { 0 } : = 4 0 N .
$$

Before starting each new batch, its size t has already been sampled and is therefore known. If executing that batch would make the total number of used copies exceed $M _ { 0 }$ , we abort without taking those copies.

To compare this capped protocol with the exact simulation, run the two protocols using the same random choices and measurement outcomes up to the possible abort. If the capped protocol aborts, then the exact simulation would have used more than $M _ { 0 }$ copies. Markov’s inequality and (27) therefore give

$$
\operatorname* { P r } _ { \sigma } ( \mathrm { a b o r t } ) \le \operatorname* { P r } _ { \sigma } ( S > M _ { 0 } ) \le \frac { \mathbb { E } _ { \sigma } S } { M _ { 0 } } \le \frac { 1 0 N / 3 } { 4 0 N } = \frac { 1 } { 1 2 } .
$$

The cutof cannot increase the weighted cost: under the coupling above, the capped protocol performs exactly the same batches as the exact simulation until it possibly aborts, and after an abort it simply stops. Since all weights $w _ { n } ( t _ { j } )$ are nonnegative, the weighted cost of the capped protocol is therefore no larger than that of the exact simulation on every run. Taking expectations and using the bound above gives

$$
\underline { { \mathbb { E } } } \mathrm { e d } \sum _ { j } w _ { n } ( t _ { j } ) \le \underline { { \mathbb { E } } } \sum _ { j } w _ { n } ( t _ { j } ) \le C N \sqrt { \operatorname* { m i n } \{ k , n ^ { 2 } \} } .
$$

Moreover, under the coupling above, the capped and exact simulations are identical whenever no abort occurs. The exact simulation reproduces the output distribution of ${ \mathcal { A } } ,$ and hence returns an estimate $\widehat { \rho }$ satisfying $D _ { \mathrm { t r } } ( \widehat { \rho } , \rho _ { \sigma } ) \leq \varepsilon$ with probability at least $2 / 3$ . Therefore, the capped protocol can fail to produce such an estimate only if either the exact simulation fails or an abort occurs. Using the union bound and $\operatorname* { P r } _ { \sigma } ( \mathrm { a b o r t } ) \leq 1 / 1 2$ , its success probability is at least

$$
1 - { \frac { 1 } { 3 } } - { \frac { 1 } { 1 2 } } = { \frac { 7 } { 1 2 } } > { \frac { 1 } { 2 } } .
$$

Whenever the simulation finishes with an estimate ${ \widehat { \rho } } ,$ we return the normalized one-photon block

$$
{ \widehat { \sigma } } : = { \frac { P _ { 1 } { \widehat { \rho } } P _ { 1 } } { \operatorname { T r } ( P _ { 1 } { \widehat { \rho } } ) } } .
$$

On abort, or if the denominator vanishes, we return an arbitrary fixed state in ${ \mathcal { D } } ( \mathbb { C } ^ { n } )$ . This is classical postprocessing of the description of $\widehat { \rho }$ and requires no additional copies. By lemma 4, whenever $\widehat { \rho }$ is within trace distance ε of $\rho _ { \sigma . }$ , the denominator is positive and $\widehat { \sigma }$ is within trace distance 16ε of $\sigma .$ . Thus the capped finite-dimensional protocol learns every $\sigma \leq 2 \mathbb { I } _ { n } / n$ to error 16ε with probability at least $7 / 1 2 > 1 / 2$

We can therefore apply lemma 1. Indeed, the capped protocol has a deterministic total copy budget $M _ { 0 } ,$ , chooses each block size and POVM before accessing its fresh copies, and retains no quantum memory between blocks. Moreover, by our initial choice of $n _ { 0 }$ and $\varepsilon _ { 0 } .$ , we have $n \geq d _ { 0 }$ and $1 6 \varepsilon < 1 6 \varepsilon _ { 0 } \leq \varepsilon _ { \mathrm { f d } }$ Hence

$$
\Omega \left( \frac { n ^ { 3 } } { ( 1 6 \varepsilon ) ^ { 2 } } \right) \leq \operatorname* { s u p } _ { \sigma \in \mathcal { D } ( \mathbb { C } ^ { n } ) } \underset { \sigma , \mathrm { c a p p e d } } { \mathbb { E } } \sum _ { j } w _ { n } ( t _ { j } ) .
$$

Combining this lower bound with the weighted-cost upper bound obtained above gives

$$
\Omega \left( \frac { n ^ { 3 } } { ( 1 6 \varepsilon ) ^ { 2 } } \right) \leq C N \sqrt { \operatorname* { m i n } \{ k , n ^ { 2 } \} } .
$$

Since $\sqrt { \operatorname* { m i n } \{ k , n ^ { 2 } \} }$ = min $\{ { \sqrt { k } } , n \}$ , rearranging yields

$$
N = \Omega \left( \frac { n ^ { 3 } } { \varepsilon ^ { 2 } \operatorname* { m i n } \{ \sqrt { k } , n \} } \right) ,
$$

as claimed.

Setting $k = 1$ gives $N = \Omega ( n ^ { 3 } / \varepsilon ^ { 2 } )$ , proving the adaptive part of Theorem 1. The next section establishes the dependence on temperature by analyzing single-copy measurements directly (without the reduction with finite-dimensional tomography).

## 5 Tight single-copy lower bound for all-temperature Gaussian states

Theorem 5. For any suficiently small $\varepsilon > 0 _ { z }$ , any single-copy adaptive scheme that can learn any n-mode passive Gaussian state, given the promise that $\Sigma \geq ( \frac { 1 } { 2 } + \nu ) \mathbb { I }$ for any $\textstyle { \frac { 1 } { n } } \leq \nu \leq 1$ , to trace distance precision ε with at least $2 / 3$ probability requires at least $N = \Omega \left( \frac { n ^ { 2 } } { \nu \varepsilon ^ { 2 } } \right)$ copies.

For ν as small as $1 / n _ { : }$ , the above theorem gives a lower bound of $\Omega ( n ^ { 3 } / \varepsilon ^ { 2 } )$ ; For ν as large as 1, it gives the classical lower bound of $\Omega ( n ^ { 2 } / \varepsilon ^ { 2 } )$ ; In the intermediate regime $\nu \in ( 1 / n , 1 )$ , it gives a smooth crossover. For all $\nu ,$ this lower bound is achieved by heterodyne measurement, see Theorem 8.

ProofSketch. Our prooffollows the Bayesian posterior anticoncentration approach developed in $[ \mathrm { C H L } ^ { + } 2 3 ]$ At a high-level, we design an appropriate prior distribution of passive bosonic Gaussian states. Then we show that, with high probability over a ground-truth state sampled from the prior and over $N$ outcomes from any (potentially adaptive) measurement schedules, the posterior distribution will not concentrate over a ε-trace distance ball centered at the ground-truth unless $N = \Omega ( n ^ { 2 } / ( \nu \varepsilon ^ { 2 } ) )$ ). This then implies the desired learning lower bound. Several novel techniques from representation theory and diferential geometry are needed to make this argument work in our bosonic Gaussian case, which are not needed in the original finite-dimensional case [CHL<sup>+</sup>23].

The proof for Theorem 5 will be presented in the following subsections.

## 5.1 Definitions

Throughout the proof, we use $C$ or $C _ { 1 } , C _ { 2 } , C _ { 3 } , \cdot \cdot \cdot \ \mathrm { o r } \ C ^ { \prime } , C ^ { \prime \prime } , \cdot \cdot \cdot$ to denote positive constants that are independent of n or ε. We sometimes slightly abuse the notation of $C ^ { \prime } s$ to denote diferent constants, but those should be clear from the context and do not afect the final scaling.

We need the following facts from random matrix theory to introduce our prior, similar to $\mathrm { [ C H L ^ { + } 2 3 }$ CLL24].

Definition 6. A Gaussian Unitary Ensemble, denoted by $\mathrm { G U E } ( n ) .$ , is a distribution over n-by-n Hermitian matrices where a sample $G \sim \mathrm { G U E } ( n )$ has independent Gaussian entries on and above its diagonal:

$$
\begin{array} { l l } { G _ { j j } \sim { \mathcal { N } } ( 0 , 2 / n ) , } & { \forall i \in [ n ] . } \\ { G _ { j _ { 1 } j _ { 2 } } \sim { \mathcal { N } } ( 0 , 1 / n ) + i { \mathcal { N } } ( 0 , 1 / n ) , } & { \forall 1 \leq j _ { 1 } < j _ { 2 } \leq n . } \end{array}\tag{28}
$$

A traceless Gaussian Unitary Ensemble, denoted by $G ^ { \prime } \sim \mathrm { G U E } ^ { * } ( n )$ , is given by $G ^ { \prime } : = G - { \frac { \operatorname { T r } G } { n } } \mathbb { I }$

Lemma 7 (E.g. [AGZ10, Section 2.6.2]). For $G \sim \mathrm { G U E } ^ { * } ( n )$ , it holds that $\| G \| _ { \mathrm { o p } } \leq 3$ with probability at least $1 - e ^ { - \Omega ( n ) }$

Let $\xi \in ( 0 , 1 / 3 2 )$ be a suficiently small parameter to be chosen later, which should be understood as of the same order as $\varepsilon / \sqrt { n \nu }$ . Define the following family of fugacity matrices:

$$
\mathsf { R } _ { X } : = r ( \mathbb { I } + \xi X ) \quad \mathrm { f o r } \quad X \in \mathrm { H e r m } _ { 0 } ( n ) ,\tag{29}
$$

where ${ \mathrm { H e r m } } _ { 0 } ( n )$ denotes the space of n-by-n traceless Hermitian matrices. Also define $G _ { \mathrm { s u p p } } : = \{ X \in $ $\mathrm { H e r m } _ { 0 } ( n ) : \| X \| _ { \mathrm { o p } } \leq 4 \}$ and $G _ { \mathrm { g o o d } } : = \{ X \in \mathrm { H e r m } _ { 0 } ( n ) : \| X \| _ { \mathrm { o p } } \leq 3 \}$ . The parameter r is chosen as

$$
r : = \frac { \nu } { ( 1 - 4 \xi ) ( 1 + \nu ) } .\tag{30}
$$

Using $\xi \leq \textstyle { \frac { 1 } { 3 2 } }$ and $\nu \leq 1$ , one can verify that $\begin{array} { r } { \frac { \nu } { 1 + \nu } \mathbb { I } \leq R _ { X } \leq \frac { 2 \nu } { 1 + 2 \nu } \mathbb { I } } \end{array}$ for all $X \in G _ { \mathrm { s u p p } }$ . Thus, this family of passive Gaussian states satisfy $\begin{array} { r } { ( \frac { 1 } { 2 } + \nu ) \mathbb { I } \leq \Sigma \leq ( \frac { 1 } { 2 } + 2 \nu ) \mathbb { I } } \end{array}$ . We will also denote $\bar { \nu } : = r / ( 1 - r )$ for later use. It is straightforward to verify that $\nu \leq \bar { \nu } \leq \frac { 4 } { 3 } \nu$

Sample $X \sim \operatorname { G U E } ^ { * } ( n )$ conditioned on $X \in G _ { \mathrm { s u p p } }$ . We take the resulting distribution of X as our prior distribution, denoted by $X \sim \mu$ . The density of X with respect to the Lebesgue measure on ${ \mathrm { H e r m } } _ { 0 } ( n )$ is given by <sup>7</sup>

$$
\mu ( X ) = Z _ { \mu } ^ { - 1 } \exp \left( - { \frac { n } { 4 } } \| X \| _ { \mathrm { F } } ^ { 2 } \right) \mathbb { 1 } ( \| X \| _ { \mathrm { o p } } \leq 4 ) .\tag{31}
$$

Here $Z _ { \mu }$ is the normalization factor. Note this induces a prior distribution over passive Gaussian states $\rho _ { \mathsf { R } _ { X } }$ . Thanks to Lemma 7, we also have $\operatorname* { P r } _ { \mu } ( X \in G _ { \mathrm { g o o d } } ) \geq 1 - e ^ { - \Omega ( n ) }$ . Another fact to be used later is the flatness condition of the density: $Z _ { \mu } ^ { - 1 } \exp ( - 4 n ^ { 2 } ) \leq \mu ( X ) \leq Z _ { \mu } ^ { - 1 }$ for all $X \in G _ { \mathrm { s u p p } }$

A non-entangled adaptive learning protocol on N copies can be specified by a family of sequences of POVMs, $\left\{ \left\{ M _ { x _ { 1 } } \right\} _ { x _ { 1 } } , \cdots , \left\{ M _ { x _ { t } } ^ { \left( x _ { < t } \right) } \right\} _ { x _ { t } } , \cdots , \left\{ M _ { x _ { N } } ^ { \left( x _ { < N } \right) } \right\} _ { x _ { N } } \right\}$ . Here, the t-th POVM can depend on the first $t - 1$ measurement outcomes $\mathbf { \mathscr { x } } _ { < t : }$ , indicated by the superscripts. An alternative description is given by the learning tree formalism, see e.g. [CHL<sup>+</sup>23, CCHL22]. We use ${ \pmb x } : = { \pmb x } _ { 1 : N }$ to denote the complete transcript of measurement outcomes.

Fix an adaptive protocol. Let $\tau _ { X } ( { \pmb x } )$ denote the likelihood of observing x when the underlying state is $\rho _ { \mathsf { R } _ { X } }$ . Let $\nu _ { x }$ denote the posterior distribution of X after observing x. By Bayes’ rule:

$$
\nu _ { \pmb { x } } ( X ) = \frac { \mathcal { T } _ { X } ( \pmb { x } ) \mu ( X ) } { \int _ { G _ { \mathrm { s u p p } } } \mathcal { T } _ { X ^ { \prime } } ( \pmb { x } ) \mu ( X ^ { \prime } ) \mathrm { d } X ^ { \prime } } .\tag{32}
$$

Finally, for $X _ { 0 } \in \mathrm { H e r m } _ { 0 } ( n )$ , define the trace-norm ball $B _ { \mathrm { t r } , a } ( X _ { 0 } ) : = \{ X \in \mathrm { H e r m } _ { 0 } ( n ) \ : \ \| X - X _ { 0 } \| _ { \mathrm { t r } } \leq a \}$ and the operator-norm ball $B _ { \mathrm { o p } , a } ( X _ { 0 } ) : = \{ X \in \mathrm { H e r m } _ { 0 } ( n ) \ : \ \| X - X _ { 0 } \| _ { \mathrm { o p } } \leq a \}$ . These will later be used to show posterior anticoncentration.

## 5.2 A relative local parametrization

To show posterior anticoncentration around any ground-truth $\rho _ { 0 } , [ \mathrm { C H L } ^ { + } 2 3 ]$ directly upper bound the posterior ratio (for some constant $C > 1 )$ :

$$
\frac { \nu _ { x } ( B _ { \mathrm { t r } , \varepsilon } ( \rho _ { 0 } ) ) } { \nu _ { x } ( B _ { \mathrm { t r } , C \varepsilon } ( \rho _ { 0 } ) ) }
$$

The denominator can be viewed as an additive perturbation $\rho _ { 0 } + Z$ for $Z \in B _ { \mathrm { o p } , C \varepsilon } ( 0 )$ . In our case, it is more convenient to work with a form of relative perturbation instead, which will become clear in later sections. For any $X _ { 0 } \in G _ { \mathrm { s u p p } }$ , we define

$$
\begin{array} { r l } & { \Phi _ { X _ { 0 } } : \mathrm { H e r m } _ { 0 } ( n ) \mapsto \mathrm { H e r m } _ { 0 } ( n ) , } \\ { s . t . } & { \mathbb { I } + \xi \Phi _ { X _ { 0 } } ( Z ) = \frac { ( \mathbb { I } + \xi X _ { 0 } ) ^ { \frac { 1 } { 2 } } ( \mathbb { I } + \xi Z ) ( \mathbb { I } + \xi X _ { 0 } ) ^ { \frac { 1 } { 2 } } } { 1 + \xi ^ { 2 } \mathrm { T r } ( X _ { 0 } Z ) / n } . } \end{array}\tag{33}
$$

The denominator normalizes the trace, so $\Phi _ { X _ { 0 } } ( Z )$ is traceless whenever $X _ { 0 }$ and $Z$ are traceless. One can also verify that Φ $\boldsymbol { x } _ { 0 } ( 0 ) = \boldsymbol { X } _ { 0 }$ . The following lemma describes the basic properties of this relative parametrization: it is locally bi-Lipschitz and has a controlled Jacobian.

Lemma 8. For any $X _ { 0 } \in G _ { \mathrm { g o o d } }$ and $Z , Z _ { 1 } , Z _ { 2 } \in B _ { \mathrm { o p } , b } ( 0 )$ with $b \leq 0 . 1 $ , and for every $p \in [ 1 , \infty ]$

$$
\begin{array} { r } { \| \mathcal { D } \Phi _ { X _ { 0 } } ( Z ) [ H ] - H \| _ { p } \leq C \xi \| H \| _ { p } , } \end{array}\tag{34}
$$

for any $H \in { \mathrm { H e r m } } _ { 0 } ( n )$ . Here $\begin{array} { r } { \mathcal { D } \Phi _ { X _ { 0 } } ( Z ) [ H ] : = \frac { d } { d t } | _ { t = 0 } \Phi _ { X _ { 0 } } ( Z + t H ) . \ \parallel \cdot \parallel _ { p } } \end{array}$ denotes the Schatten p-norm, with $\| \cdot \| _ { \infty } = \| \cdot \| _ { \mathrm { o p } }$ . Consequently,

$$
\frac { 1 } { 2 } \| Z _ { 1 } - Z _ { 2 } \| _ { p } \leq \| \Phi _ { X _ { 0 } } ( Z _ { 1 } ) - \Phi _ { X _ { 0 } } ( Z _ { 2 } ) \| _ { p } \leq 2 \| Z _ { 1 } - Z _ { 2 } \| _ { p } ,\tag{35}
$$

Also,

$$
\Phi _ { X _ { 0 } } ( Z ) \in G _ { \mathrm { s u p p } } .\tag{36}
$$

Finally, the Jacobian determinant of $\Phi _ { X _ { 0 } }$ satisfies

$$
J _ { \Phi _ { X _ { 0 } } } ( Z ) : = | \mathrm { d e t } ( { \mathcal { D } } \Phi _ { X _ { 0 } } ( Z ) ) | \geq \exp ( - C ^ { \prime } n ^ { 2 } ) .\tag{37}
$$

Proof. For notational simplicity, define

$$
S : = ( \mathbb { I } + \xi X _ { 0 } ) , \quad T _ { Z } : = ( \mathbb { I } + \xi Z ) , \quad d _ { Z } : = 1 + \xi ^ { 2 } \operatorname { T r } ( X _ { 0 } Z ) / n .\tag{38}
$$

First compute the derivative,

$$
\begin{array} { l } { { \mathcal { D } \Phi _ { X _ { 0 } } ( Z ) [ H ] = \displaystyle \frac { d } { d t } \bigg | _ { t = 0 } \frac { 1 } { \xi } \left( d _ { Z + t H } ^ { - 1 } S ^ { \frac 1 2 } ( T _ { Z } + t \xi H ) S ^ { \frac 1 2 } - \mathbb { I } \right) } } \\ { { \mathrm { ~ } = d _ { Z } ^ { - 1 } S ^ { \frac 1 2 } H S ^ { \frac 1 2 } - \displaystyle \frac { \xi d _ { Z } ^ { - 2 } } { n } S ^ { \frac 1 2 } T _ { Z } S ^ { \frac 1 2 } \operatorname { T r } X _ { 0 } H . } } \end{array}\tag{39}
$$

Thus,

$$
\| \mathcal { D } \Phi _ { X _ { 0 } } ( Z ) [ H ] - H \| _ { p } \leq \Big \| d _ { Z } ^ { - 1 } S ^ { \frac 1 2 } H S ^ { \frac 1 2 } - H \Big \| _ { p } + \frac { \xi d _ { Z } ^ { - 2 } } { n } \| S ^ { \frac 1 2 } T _ { Z } S ^ { \frac 1 2 } \| _ { p } | \mathrm { T r } X _ { 0 } H | .\tag{40}
$$

For the first term, note that

$$
| d _ { Z } ^ { - 1 } - 1 | \le C _ { 0 } | d _ { Z } - 1 | \le C _ { 0 } \| X _ { 0 } \| _ { \mathrm { o p } } \| Z \| _ { \mathrm { o p } } \xi ^ { 2 } \le C _ { 1 } \xi ^ { 2 }\tag{41}
$$

And, by defining $B = S ^ { \frac { 1 } { 2 } } - \mathbb { I }$ , for suficiently small $\xi ,$

$$
\| B \| _ { \mathrm { o p } } = \| ( \mathbb I + \xi X _ { 0 } ) ^ { \frac { 1 } { 2 } } - \mathbb I \| _ { \mathrm { o p } } \leq \xi \| X _ { 0 } \| _ { \mathrm { o p } } \leq C _ { 2 } \xi .\tag{42}
$$

Therefore, the first term can be upper bounded as

$$
\begin{array} { r l } & { \left\| d _ { Z } ^ { - 1 } S ^ { \frac 1 2 } H S ^ { \frac 1 2 } - H \right\| _ { p } \leq \left\| ( d _ { Z } ^ { - 1 } - 1 ) S ^ { \frac 1 2 } H S ^ { \frac 1 2 } \right\| _ { p } + \left\| S ^ { \frac 1 2 } H S ^ { \frac 1 2 } - H \right\| _ { p } } \\ & { \qquad \leq | d _ { Z } ^ { - 1 } - 1 | \| S \| _ { \mathrm { o p } } \| H \| _ { p } + 2 \| H \| _ { p } \| B \| _ { \mathrm { o p } } + \| H \| _ { p } \| B \| _ { \mathrm { o p } } ^ { 2 } } \\ & { \qquad \leq C _ { 3 } \xi \| H \| _ { p } . } \end{array}\tag{43}
$$

For the second term, let $q \in [ 1 , \infty ]$ be such that $1 / q + 1 / p = 1$

$$
\begin{array} { r l } {  { \frac { \xi d _ { Z } ^ { - 2 } } { n } \| S ^ { \frac { 1 } { 2 } } T _ { Z } S ^ { \frac { 1 } { 2 } } \| _ { p } | \operatorname { T r } X _ { 0 } H | \overset { \mathrm { ( i ) } } { \leq } \frac { \xi d _ { Z } ^ { - 2 } } { n } \| S ^ { \frac { 1 } { 2 } } T _ { Z } S ^ { \frac { 1 } { 2 } } \| _ { p } \| X _ { 0 } \| _ { q } \| H \| _ { p } } } \\ & { \overset { \mathrm { ( i i ) } } { \leq } \frac { \xi d _ { Z } ^ { - 2 } } { n } n ^ { \frac { 1 } { p } } \| S ^ { \frac { 1 } { 2 } } T _ { Z } S ^ { \frac { 1 } { 2 } } \| _ { \mathrm { o p } } n ^ { \frac { 1 } { q } } \| X _ { 0 } \| _ { \mathrm { o p } } \| H \| _ { p } } \\ & { \leq C _ { 4 } \xi \| H \| _ { p } . } \end{array}\tag{44}
$$

Here (i) is by Hölder’s inequality; (ii) uses $\| \cdot \| _ { p } \leq n ^ { \frac { 1 } { p } } \| \cdot \| _ { \mathrm { o p } }$ . Combining both terms yields, as claimed,

$$
\begin{array} { r } { \| \mathcal { D } \Phi _ { X _ { 0 } } ( Z ) [ H ] - H \| _ { p } \leq C \xi \| H \| _ { p } . } \end{array}\tag{45}
$$

Now, let $\Delta : = Z _ { 2 } - Z _ { 1 }$ . The fundamental theorem of calculus gives that

$$
\begin{array} { r l r } {  { \Phi _ { X _ { 0 } } ( Z _ { 2 } ) - \Phi _ { X _ { 0 } } ( Z _ { 1 } ) = \int _ { 0 } ^ { 1 } \mathcal { D } \Phi _ { X _ { 0 } } ( Z _ { 1 } + t \Delta ) [ \Delta ] \mathrm { d } t } } \\ & { } & { = \Delta + \int _ { 0 } ^ { 1 } ( \mathcal { D } \Phi _ { X _ { 0 } } ( Z _ { 1 } + t \Delta ) [ \Delta ] - \Delta ) \mathrm { d } t . } \end{array}\tag{46}
$$

Thus, by the convexity of the Schatten p-norm,

$$
\| \Phi _ { X _ { 0 } } ( Z _ { 2 } ) - \Phi _ { X _ { 0 } } ( Z _ { 1 } ) \| _ { p } \leq \int _ { 0 } ^ { 1 } \| \mathcal { D } \Phi _ { X _ { 0 } } ( Z _ { 1 } + t \Delta ) [ \Delta ] \| _ { p } \mathrm { d } t \leq ( 1 + C \xi ) \| \Delta \| _ { p } .\tag{47}
$$

$$
\| \Phi _ { X _ { 0 } } ( Z _ { 2 } ) - \Phi _ { X _ { 0 } } ( Z _ { 1 } ) \| _ { p } \ge \| \Delta \| _ { p } - \int _ { 0 } ^ { 1 } \| \mathcal { D } \Phi _ { X _ { 0 } } ( Z _ { 1 } + t \Delta ) [ \Delta ] - \Delta \| _ { p } \mathrm { d } t \ge ( 1 - C \xi ) \| \Delta \| _ { p } .\tag{48}
$$

Take ξ suficiently small such that $C \xi \le 1 / 2$ gives as claimed,

$$
\frac 1 2 \| Z _ { 2 } - Z _ { 1 } \| _ { p } \leq \| \Phi _ { X _ { 0 } } ( Z _ { 2 } ) - \Phi _ { X _ { 0 } } ( Z _ { 1 } ) \| _ { p } \leq 2 \| Z _ { 2 } - Z _ { 1 } \| _ { p }\tag{49}
$$

In particular, this implies

$$
\begin{array} { r } { \| \Phi _ { X _ { 0 } } ( Z ) \| _ { \mathrm { o p } } \leq \| \Phi _ { X _ { 0 } } ( Z ) - \Phi _ { X _ { 0 } } ( 0 ) \| _ { \mathrm { o p } } + \| X _ { 0 } \| _ { \mathrm { o p } } \leq 2 \| Z \| _ { \mathrm { o p } } + \| X _ { 0 } \| _ { \mathrm { o p } } < 4 , } \end{array}\tag{50}
$$

which implies $\Phi _ { X _ { 0 } } ( Z ) \in G _ { \mathrm { s u p p } }$ . Finally, note what we have shown implies

$$
( 1 - C \xi ) \| H \| _ { 2 } \leq \| \mathcal { D } \Phi _ { X _ { 0 } } ( Z ) [ H ] \| _ { 2 } \leq ( 1 + C \xi ) \| H \| _ { 2 } ,\tag{51}
$$

which equivalently says the singular values of $\mathcal { D } \Phi _ { X _ { 0 } } ( Z )$ as a linear map on ${ \mathrm { H e r m } } _ { 0 } ( n )$ is within $[ 1 -$ $C \xi , 1 + C \xi ]$ . The Jacobian determinant of $\Phi _ { X _ { 0 } }$ is thus lower bounded by

$$
J _ { \Phi _ { X _ { 0 } } } : = \vert \operatorname * { d e t } \mathcal { D } \Phi _ { X _ { 0 } } ( Z ) \vert \ge ( 1 - C \xi ) ^ { n ^ { 2 } - 1 } \ge \exp ( - C ^ { \prime } n ^ { 2 } )
$$

uniformly in $Z \in B _ { \mathrm { o p } , b } ( 0 )$

We define the following ball induced by Φ ${ \mathrm { \Sigma } } _ { X _ { 0 } } \colon B _ { \Phi , b } ( X _ { 0 } ) : = \{ \Phi _ { X _ { 0 } } ( Z ) : Z \in B _ { \mathrm { o p } , b } ( 0 ) \}$ . By Lemma 8, $B _ { \Phi , b } ( X _ { 0 } ) \subseteq G _ { \mathrm { s u p p } }$ for every $X _ { 0 } \in G _ { \mathrm { g o o d } }$

## 5.3 Reduce posterior anti-concentration to likelihood ratio bound

The following proposition is our starting point for showing posterior anticoncentration. Here, a and b are suficiently small positive constants that will be chosen later.

Proposition 1. Fix $X _ { 0 } \in G _ { \mathrm { g o o d } }$ . For any $X \in G _ { \mathrm { s u p p } }$ and outcomes $\mathbf { \delta } _ { \mathbf { \mathbf { \mathbf { x } } } , \mathbf { \delta } }$ define the likelihood ratio

$$
L _ { X } ( \pmb { x } ) : = \frac { \mathcal { T } _ { X } ( \pmb { x } ) } { \mathcal { T } _ { X _ { 0 } } ( \pmb { x } ) } .\tag{52}
$$

Then, with probability at least $1 - e ^ { - n ^ { 2 } }$ over $\mathbf { \boldsymbol { x } } \sim \mathcal { T } _ { \boldsymbol { X } _ { 0 } }$

$$
\frac { \nu _ { x } ( B _ { \mathrm { t r } , n a } ( X _ { 0 } ) ) } { \nu _ { x } ( B _ { \Phi , b } ( X _ { 0 } ) ) } \leq \exp ( - \Omega ( n ^ { 2 } ) ) \left( \underset { Z \sim \mathrm { U n i f } ( B _ { \mathrm { o p } , b } ( 0 ) ) } { \mathbb { E } } L _ { \Phi _ { X _ { 0 } } ( Z ) } ( \boldsymbol { x } ) \right) ^ { - 1 } .\tag{53}
$$

Proof. Write $X : = \Phi _ { X _ { 0 } } ( Z )$ . By Bayes’ rule,

$$
\frac { \nu _ { x } ( B _ { \mathrm { t r } , n a } ( X _ { 0 } ) ) } { \nu _ { x } ( B _ { \Phi , b } ( X _ { 0 } ) ) } = \frac { \int _ { B _ { \mathrm { t r } , n a } ( X _ { 0 } ) } L _ { X } ( \pmb { x } ) \mu ( X ) \mathrm { d } X } { \int _ { B _ { \Phi , b } ( X _ { 0 } ) } L _ { X } ( \pmb { x } ) \mu ( X ) \mathrm { d } X } .\tag{54}
$$

For the denominator, Lemma 8 gives both injectivity of $\Phi _ { X _ { 0 } }$ on $B _ { \mathrm { o p } , b } ( 0 )$ and $B _ { \Phi , b } ( X _ { 0 } ) \subseteq G _ { \mathrm { s u p p } } ,$ so one can change the variable $Z = \Phi _ { X _ { 0 } } ^ { - 1 } ( X )$ . Thus,

$$
\begin{array} { r l } { \displaystyle \int _ { B _ { \Phi , b } ( X _ { 0 } ) } L _ { X } ( \pmb { x } ) \mu ( X ) \mathrm { d } X \stackrel { \mathrm { ( i ) } } { = } \int _ { B _ { \Phi , b } ( 0 ) } L _ { \Phi _ { X _ { 0 } } ( Z ) } ( \pmb { x } ) \mu ( \Phi _ { X _ { 0 } } ( Z ) ) J _ { \Phi _ { X _ { 0 } } } ( Z ) \mathrm { d } Z } & { } \\ { \overset { \mathrm { ( i i ) } } { \geq } Z _ { \mu } ^ { - 1 } \exp ( - C _ { 1 } n ^ { 2 } ) \int _ { B _ { \Phi , b } ( 0 ) } L _ { \Phi _ { X _ { 0 } } ( Z ) } ( \pmb { x } ) \mathrm { d } Z } & { } \\ { \overset { \mathrm { ( i i i ) } } { = } Z _ { \mu } ^ { - 1 } \exp ( - C _ { 1 } n ^ { 2 } ) \mathrm { V o l } ( B _ { \mathrm { o p } , b } ) \underset { Z \sim \mathrm { U n i f } ( B _ { \mathrm { o p } , b } ( 0 ) ) } { \mathbb { E } } L _ { \Phi _ { X _ { 0 } } ( Z ) } ( \pmb { x } ) . } \end{array}\tag{55}
$$

Here, (i) uses d $X = J _ { \Phi _ { X _ { 0 } } } ( Z ) \mathrm { d } Z$ where $J _ { \Phi _ { X _ { 0 } } } ( Z )$ is the Jacobian determinant; (ii) uses the support guarantee $\Phi _ { X _ { 0 } } ( Z ) \in G _ { \mathrm { s u p p } } ,$ , the Jacobian determinant lower bound from Lemma 8, and the expression of the prior density; In (iii), $\mathrm { V o l } ( B _ { \mathrm { o p } , b } )$ denotes the Lebesgue volume of the operator-norm ball. We omit the center as it is irrelevant of the volume.

For the numerator, taking the expectation of $\mathbf { \mathit { x } } \sim \tau _ { X _ { 0 } }$ yields

$$
\underset { x \sim 7 X _ { 0 } } { \mathbb { E } } \int _ { B _ { \mathrm { f r } , n a } ( X _ { 0 } ) } L _ { X } ( \pmb { x } ) \mu ( X ) \mathrm { d } X = \int _ { B _ { \mathrm { t r } , n a } ( X _ { 0 } ) } \mu ( X ) \mathrm { d } X = \mu ( B _ { \mathrm { t r } , n a } ( X _ { 0 } ) ) .\tag{56}
$$

Therefore, Markov’s inequality implies that, with probability at least $1 - \exp ( - n ^ { 2 } )$ over $x \sim \tau _ { X _ { 0 } }$ , it holds that<sup>8</sup>

$$
\int _ { B _ { \mathrm { t r } , n a } ( X _ { 0 } ) } L _ { X } ( \pmb { x } ) \mu ( X ) \mathrm { d } X \leq \exp ( n ^ { 2 } ) \mu ( B _ { \mathrm { t r } , n a } ( X _ { 0 } ) )\tag{57}
$$

$$
\leq \exp ( n ^ { 2 } ) Z _ { \mu } ^ { - 1 } \mathrm { V o l } ( B _ { \mathrm { t r } , n a } ) .
$$

Putting the two parts together,

$$
\frac { \nu _ { x } ( B _ { \mathrm { t r } , n a } ( X _ { 0 } ) ) } { \nu _ { x } ( B _ { \Phi , b } ( X _ { 0 } ) ) } \leq \exp ( ( 1 + C _ { 1 } ) n ^ { 2 } ) \frac { \mathrm { V o l } ( B _ { \mathrm { t r } , n a } ) } { \mathrm { V o l } ( B _ { \mathrm { o p } , b } ) } \left( \underset { Z \sim \mathrm { U n i f } ( B _ { \mathrm { o p } , b } ( 0 ) ) } { \mathbb { E } } L _ { \Phi _ { X _ { 0 } } ( Z ) } ( \pmb { x } ) \right) ^ { - 1 } .\tag{58}
$$

Now, use the following volume ratio bound:<sup>9</sup>

Lemma $9 \ ( \mathrm { [ C H L ^ { + } 2 3 }$ , CLL24]). For the trace-norm and operator-norm ball defined in ${ \mathrm { H e r m } } _ { 0 } ( n )$

$$
\frac { \mathrm { V o l } ( B _ { \mathrm { t r } , 1 } ) } { \mathrm { V o l } ( B _ { \mathrm { o p } , 1 / n } ) } \le \exp ( 1 0 n ^ { 2 } ) .\tag{59}
$$

By homogeneity,

$$
{ \frac { { \mathrm { V o l } } ( B _ { \mathrm { t r } , n a } ) } { { \mathrm { V o l } } ( B _ { \mathrm { o p } , b } ) } } = { \frac { ( n a ) ^ { n ^ { 2 } - 1 } } { ( n b ) ^ { n ^ { 2 } - 1 } } } { \frac { { \mathrm { V o l } } ( B _ { \mathrm { t r } , 1 } ) } { { \mathrm { V o l } } ( B _ { \mathrm { o p } , 1 / n } ) } } \leq \exp ( - ( n ^ { 2 } - 1 ) \log ( b / a ) + 1 0 n ^ { 2 } ) .\tag{60}
$$

By choosing $b / a$ a suficiently large constant and putting the above back to the posterior ratio, we conclude that

$$
\frac { \nu _ { x } ( B _ { \mathrm { t r } , n a } ( X _ { 0 } ) ) } { \nu _ { x } ( B _ { \Phi , b } ( X _ { 0 } ) ) } \leq \exp ( - \Omega ( n ^ { 2 } ) ) \left( \underset { Z \sim \mathrm { U n i f } ( B _ { \mathrm { o p } , b } ( 0 ) ) } { \mathbb { E } } L _ { \Phi _ { X _ { 0 } } ( Z ) } ( x ) \right) ^ { - 1 } .\tag{61}
$$

Recall this holds with probability at least $1 - e ^ { - n ^ { 2 } }$ over $\mathbf { \mathit { x } } \sim \tau _ { X _ { 0 } }$ for fixed $X _ { 0 } \in G _ { \mathrm { g o o d } } .$

## 5.4 Likelihood ratio bound – fixed transcripts

Thanks to Proposition 1, we just need E<sub>Z</sub> $L _ { \Phi _ { X _ { 0 } } ( Z ) } ( \pmb { x } ) \ge \exp ( - o ( n ^ { 2 } ) )$ with high probability for posterior anticoncentration around $X _ { 0 }$ , where $Z \sim \mathrm { U n i f } ( B _ { \mathrm { o p } , b } ( 0 ) )$ . We now start to lower bound this average likelihood ratio. Lemma 8 ensures that every $\Phi _ { X _ { 0 } } ( Z )$ appearing here lies in $G _ { \mathrm { s u p p } }$

Since passive Gaussian states are block-diagonal in total photon number sectors, one can without loss of generality assume all the POVMs have the same block-diagonal structure. Furthermore, we can assume all POVM elements are rank-1, for the same reason as before. We thus denote the t-th POVM by $\left\{ | \phi _ { z _ { t } , k _ { t } } ^ { ( \pmb { x } _ { < t } ) } \rangle \langle \phi _ { z _ { t } , k _ { t } } ^ { ( \pmb { x } _ { < t } ) } | \right\} _ { z _ { t } , k _ { t } }$ , with outcome $x _ { t } : = ( z _ { t } , k _ { t } )$ where $k _ { t }$ indicates the total number of photons and $| \phi _ { z _ { t } , k _ { t } } ^ { ( \pmb { x } _ { < t } ) } \rangle \in \mathcal { H } _ { k _ { t } } ^ { ( n ) }$ is an unnormalized vector satisfying $\begin{array} { r } { \int \mathrm { d } z _ { t } | \phi _ { z _ { t } , k _ { t } } ^ { ( \pmb { x } _ { < t } ) } \rangle \langle \phi _ { z _ { t } , k _ { t } } ^ { ( \pmb { x } _ { < t } ) } | = P _ { k _ { t } } ^ { ( n ) } } \end{array}$ for all $k _ { t } \in \mathbb { N } _ { i }$ , where we recall $P _ { k } ^ { ( n ) }$ is the projector onto the k-photon sector of the Hilbert space. Again, the superscript of $\scriptstyle { \mathbf { { \pmb { x } } } } _ { < t }$ indicates adaptivity.

Now, fix $X _ { 0 } \in G _ { \mathrm { g o o d } }$ and a transcript of outcomes x. Recall $X : = \Phi _ { X _ { 0 } } ( Z )$

$$
\begin{array} { r } { \log _ { \frac { U } { Z } } L _ { \Phi _ { X _ { 0 } } ( Z ) } ( \pmb { x } ) = \log _ { \frac { U } { Z } } \displaystyle \prod _ { t = 1 } ^ { N } \frac { \langle \phi _ { z _ { t } , k _ { t } } ^ { ( \pmb { x } _ { < t } ) } | \rho _ { \mathsf { R } _ { X } } | \phi _ { z _ { t } , k _ { t } } ^ { ( \pmb { x } _ { < t } ) } \rangle } { \langle \phi _ { z _ { t } , k _ { t } } ^ { ( \pmb { x } _ { < t } ) } | \rho _ { \mathsf { R } _ { X _ { 0 } } } | \phi _ { z _ { t } , k _ { t } } ^ { ( \pmb { x } _ { < t } ) } \rangle } } \\ { \overset { \mathrm { ( i ) } } { \geq } \displaystyle \sum _ { t = 1 } ^ { N } \mathbb { E } \log \frac { \langle \phi _ { z _ { t } , k _ { t } } ^ { ( \pmb { x } _ { < t } ) } | \rho _ { \mathsf { R } _ { X } } | \phi _ { z _ { t } , k _ { t } } ^ { ( \pmb { x } _ { < t } ) } \rangle } { \langle \phi _ { z _ { t } , k _ { t } } ^ { ( \pmb { x } _ { < t } ) } | \rho _ { \mathsf { R } _ { X _ { 0 } } } | \phi _ { z _ { t } , k _ { t } } ^ { ( \pmb { x } _ { < t } ) } \rangle } . } \end{array}\tag{62}
$$

(i) is by Jensen’s inequality. Thus, we just need to bound the average log likelihood ratio for each copy. In the following, we focus on the t-th term and temporarily rewrite the POVM as $\{ | z , k \rangle \langle z , k | \} _ { z , k }$ for notational simplicity. By definition of $\rho _ { \mathsf { R } }$ , we have

$$
\begin{array} { r } { \frac { \mathbb { E } } { Z } \log \frac {  z , k \big | \rho _ { \mathsf { R } _ { X } } \big | z , k  } { \langle z , k \big | \rho _ { \mathsf { R } _ { X _ { 0 } } } \big | z , k \rangle } = \frac { \mathbb { E } } { Z } \log \frac { \operatorname* { d e t } ( \mathbb { I } - \mathsf { R } _ { X } ) } { \operatorname* { d e t } ( \mathbb { I } - \mathsf { R } _ { X _ { 0 } } ) } + \frac { \mathbb { E } } { Z } \log \frac { \langle z , k \big | \Gamma _ { k } ( \mathsf { R } _ { X } ) \big | z , k  } { \langle z , k \big | \Gamma _ { k } ( \mathsf { R } _ { X _ { 0 } } ) \big | z , k \rangle } . } \end{array}\tag{63}
$$

Recall that $\mathsf { R } _ { X _ { 0 } } = r ( \mathbb { I } + \xi X _ { 0 } )$ and $\mathsf { R } _ { X } = r ( \mathbb { I } + \xi X ) = r \frac { ( \mathbb { I } + \xi X _ { 0 } ) ^ { \frac { 1 } { 2 } } ( \mathbb { I } + \xi Z ) ( \mathbb { I } + \xi X _ { 0 } ) ^ { \frac { 1 } { 2 } } } { 1 + \xi ^ { 2 } \operatorname { T r } ( X _ { 0 } Z ) / n } .$

Bound for the first term. Notice that (recall we denote $\textstyle { \bar { \nu } } : = { \frac { r } { 1 - r } } )$

$$
\log \operatorname * { d e t } ( \mathbb { I } - r ( \mathbb { I } + \xi X ) ) = n \log ( 1 - r ) + \log \operatorname * { d e t } ( \mathbb { I } - \bar { \nu } \xi X ) .\tag{64}
$$

Only the second term depends on X. We bound it as follows,

$$
\begin{array} { l } { \displaystyle \log \operatorname* { d e t } ( \mathbb { I } - \bar { \nu } \xi X ) \big \vert \triangleq \left. \displaystyle \sum _ { k = 2 } ^ { \infty } \frac { 1 } { k } ( \bar { \nu } \xi ) ^ { k } \mathrm { T r } X ^ { k } \right. } \\ { \displaystyle \qquad \leq \displaystyle \sum _ { k = 2 } ^ { \infty } \frac { 1 } { k } ( \bar { \nu } \xi ) ^ { k } n \| X \| _ { \mathrm { o p } } ^ { k } . } \\ { \displaystyle \qquad \overset { \mathrm { ( i i ) } } { \leq } 1 6 n \bar { \nu } ^ { 2 } \xi ^ { 2 } \displaystyle \sum _ { k = 0 } ^ { \infty } ( 4 \bar { \nu } \xi ) ^ { k } } \\ { \displaystyle \qquad = 1 6 n \bar { \nu } ^ { 2 } \xi ^ { 2 } ( 1 - 4 \bar { \nu } \xi ) ^ { - 1 } } \\ { \displaystyle \qquad \overset { \mathrm { ( i i i ) } } { \leq } 3 2 \bar { n } \bar { \nu } ^ { 2 } \xi ^ { 2 } . } \end{array}\tag{65}
$$

Here, (i) uses the Taylor expansion for log det. The $k = 1$ term disappears as X is traceless; (ii) uses Lemma $^ { 8 , }$ which gives $\| X \| _ { \mathrm { o p } } \leq 4$ for $X = \Phi _ { X _ { 0 } } ( Z )$ for all Z we consider; (iii) uses the observation that $\bar { \nu } \xi \le 1 / 2 4$ . Combining this with the same bound for $X _ { 0 } \in G _ { \mathrm { g o o d } }$ and the triangle inequality,

$$
\log { \frac { \operatorname * { d e t } ( \mathbb { I } - \mathsf { R } _ { X } ) } { \operatorname * { d e t } ( \mathbb { I } - \mathsf { R } _ { X _ { 0 } } ) } } \geq - 6 4 n \bar { \nu } ^ { 2 } \xi ^ { 2 } .\tag{66}
$$

Since this bounds holds uniformly for all $X = \Phi _ { X _ { 0 } } ( Z )$ , it also holds in expectation.

Bound for the second term. This term (before averaging over $Z )$ can be further decomposed as

$$
\begin{array} { r l } & { \quad \log \frac {  z , k  \Gamma _ { k } ( \mathsf { R } _ { X } )  z , k  } {  z , k  \Gamma _ { k } ( \mathsf { R } _ { X _ { 0 } } )  z , k  } } \\ & { = \log \frac {  z , k  \Gamma _ { k } ( \mathbb { I } + \xi X _ { 0 } ) ^ { \frac { 1 } { 2 } } \Gamma _ { k } ( \mathbb { I } + \xi Z ) \Gamma _ { k } ( \mathbb { I } + \xi X _ { 0 } ) ^ { \frac { 1 } { 2 } }  z , k  } {  z , k  \Gamma _ { k } ( \mathbb { I } + \xi X _ { 0 } )  z , k  } - k \log ( 1 + \frac { \xi ^ { 2 } } { n } \operatorname { T r } ( X _ { 0 } Z ) ) . } \end{array}\tag{67}
$$

The second part is non-negative after averaging over $Z \sim \operatorname { U n i f } ( B _ { \mathrm { o p } , b } ( 0 ) )$

$$
- \frac { \mathbb { E } } { z } k \log \left( 1 + \frac { \xi ^ { 2 } } { n } \operatorname { T r } ( X _ { 0 } Z ) \right) \overset { \mathrm { ( i ) } } { \geq } - k \log \left( 1 + \frac { \xi ^ { 2 } } { n } \frac { \mathbb { E } } { z } \operatorname { T r } ( X _ { 0 } Z ) \right) \overset { \mathrm { ( i i ) } } { = } 0 .\tag{68}
$$

(i) uses the concavity of log; (ii) is because the distribution of $Z$ is centrally symmetric.

To evaluate the first part, define a new normalized vector in $\mathcal { H } _ { k } ^ { ( n ) }$

$$
\vert \bar { \psi } \rangle : = \frac { \Gamma _ { k } ( \mathbb { I } + \xi X _ { 0 } ) ^ { \frac { 1 } { 2 } } \left. z , k \right. } { \sqrt { \left. z , k \right. \Gamma _ { k } ( \mathbb { I } + \xi X _ { 0 } ) \left. z , k \right. } } .\tag{69}
$$

The first part can then be concisely written as

$$
\begin{array} { r } { \frac { \ d } { \ d z } \operatorname { \mathbb { E } } \log \left. \bar { \psi } \big | \Gamma _ { k } ( \mathbb { I } + \xi Z ) \big | \bar { \psi } \right. . } \end{array}
$$

Let the spectrum decomposition of Z be $Z = U D U ^ { \dagger }$ where $D = \mathrm { d i a g } ( d )$ . We first fix D and average over $U _ { z }$ , which is possible as the distribution of Z is invariant under unitary conjugation. Define

$$
F ( U ) = \log \left. \bar { \psi } | \Gamma _ { k } ( \mathbb { I } + \xi U D U ^ { \dagger } ) | \bar { \psi } \right. .\tag{70}
$$

We need the following lemma to control the concentration of $F$

Lemma 10 (Haar log-Sobolev concentration, modified from [MM13]). Let $U \sim \operatorname { H a a r } ( \mathbb { U } ( n ) )$ , and equip $\mathbb { U } ( n )$ with the geodesic distance induced by the Frobenius (i.e., Hilbert–Schmidt) norm. Suppose that $F : \mathbb { U } ( n ) \to$ R is L-Lipschitz with respect to this geodesic distance. Then, for every $\lambda > 0$

$$
\log _ { U } \mathbb { E } \exp \biggl ( \lambda ( F ( U ) - \underset { U } { \mathbb { E } } F ( U ) ) \biggr ) \leq \frac { 3 \pi ^ { 2 } \lambda ^ { 2 } L ^ { 2 } } { 4 n } .\tag{71}
$$

In particular,

$$
\mathbb { E } F ( U ) \geq \log \varmathbb { E } e ^ { F ( U ) } - \frac { 3 \pi ^ { 2 } L ^ { 2 } } { 4 n } .\tag{72}
$$

Proof. By [MM13, Theorem 15 and the following paragraph], Haar measure $\mu$ on $\mathbb { U } ( n )$ satisfies

$$
\mathrm { E n t } _ { \mu } ( g ^ { 2 } ) \leq \frac { 3 \pi ^ { 2 } } { n } \mathbb { E } | \nabla g | ^ { 2 }\tag{73}
$$

for this geodesic metric, where

$$
\operatorname { E n t } _ { \mu } ( Y ) : = \operatorname { \mathbb { E } } _ { \mu } ( Y \log Y ) - ( \operatorname { \mathbb { E } } _ { \mu } Y ) \log ( \operatorname { \mathbb { E } } _ { \mu } Y ) .\tag{74}
$$

Set

$$
H : = F - \operatorname * { \mathbb { E } } _ { \mu } F , \qquad \psi ( \lambda ) : = \log \operatorname * { \mathbb { E } } _ { \mu } e ^ { \lambda H } .\tag{75}
$$

Since H difers from $F$ by a constant and $F$ is L-Lipschitz, $| \nabla H | = | \nabla F | \leq L$ . Thus, for $g = e ^ { \lambda H / 2 }$

$$
| \nabla g | ^ { 2 } = \frac { \lambda ^ { 2 } } 4 e ^ { \lambda H } | \nabla H | ^ { 2 } \leq \frac { \lambda ^ { 2 } L ^ { 2 } } 4 e ^ { \lambda H } .\tag{76}
$$

Applying the preceding logarithmic Sobolev inequality therefore gives, for $\lambda > 0$

$$
\mathrm { E n t } _ { \mu } ( e ^ { \lambda H } ) \leq \frac { 3 \pi ^ { 2 } \lambda ^ { 2 } L ^ { 2 } } { 4 n } \mathbb { E } e ^ { \lambda H } .\tag{77}
$$

On the other hand,

$$
\begin{array} { r l } & { \frac { \mathrm { E n t } _ { \mu } ( e ^ { \lambda H } ) } { \mathbb { E } _ { \mu } e ^ { \lambda H } } = \lambda \frac { \mathbb { E } _ { \mu } ( H e ^ { \lambda H } ) } { \mathbb { E } _ { \mu } e ^ { \lambda H } } - \log \frac { \mathbb { E } } { \mu } e ^ { \lambda H } } \\ & { \quad \quad = \lambda \psi ^ { \prime } ( \lambda ) - \psi ( \lambda ) . } \end{array}\tag{78}
$$

Combining the last two equations yields

$$
\lambda \psi ^ { \prime } ( \lambda ) - \psi ( \lambda ) \leq \frac { 3 \pi ^ { 2 } \lambda ^ { 2 } L ^ { 2 } } { 4 n } .\tag{79}
$$

Hence

$$
\left( \frac { \psi ( \lambda ) } { \lambda } \right) ^ { \prime } = \frac { \lambda \psi ^ { \prime } ( \lambda ) - \psi ( \lambda ) } { \lambda ^ { 2 } } \leq \frac { 3 \pi ^ { 2 } L ^ { 2 } } { 4 n } .\tag{80}
$$

Since $\psi ( 0 ) = \psi ^ { \prime } ( 0 ) = 0$ , integration proves the claim for $\lambda > 0$ . The final claim is the case $\lambda = 1$

We now bound the Lipschitz constant for $F .$ . Let $U _ { \theta } = e ^ { \theta K } U$ for any $K ^ { \dagger } = - K$ . Define

$$
B _ { \theta } : = U _ { \theta } ( \mathbb { I } + \xi D ) U _ { \theta } ^ { \dagger } , \quad T _ { \theta } : = \langle \bar { \psi } | \Gamma _ { k } ( B _ { \theta } ) | \bar { \psi } \rangle .\tag{81}
$$

Hence $F ( U _ { \theta } ) = \log T _ { \theta }$ . Since $\Gamma _ { k }$ denotes the k-th symmetric tensor power,

$$
T _ { \theta } = \left. \Psi \right| B _ { \theta } ^ { \otimes k } \left| \Psi \right. ,\tag{82}
$$

where |Ψ⟩ is the first-quantized wave function of $| \bar { \psi } \rangle$ , and dots denote diferentiation with respect to $\theta .$ Thus,

$$
\begin{array} { r l } & { | \dot { T } _ { \theta } | = | \langle \Psi | \big ( B _ { \theta } ^ { \frac { 1 } { 2 } } \big ) ^ { \otimes k } \big ( \underset { j = 1 } { \overset { k } { \sum } } \mathbb { I } ^ { \otimes j - 1 } \otimes B _ { \theta } ^ { - \frac { 1 } { 2 } } \dot { B } _ { \theta } B _ { \theta } ^ { - \frac { 1 } { 2 } } \otimes \mathbb { I } ^ { \otimes k - j } \big ) \big ( B _ { \theta } ^ { \frac { 1 } { 2 } } \big ) ^ { \otimes k } | \Psi \rangle | } \\ & { \qquad \leq k \| B _ { \theta } ^ { - \frac { 1 } { 2 } } \dot { B } _ { \theta } B _ { \theta } ^ { - \frac { 1 } { 2 } } \| _ { \mathrm { o p } } \cdot T _ { \theta } . } \end{array}\tag{83}
$$

Further note that $\dot { B } _ { \theta } = [ K , B _ { \theta } ]$ and that the singular values of $B _ { \theta }$ are bounded within, say, [0.9, 1.1],

$$
\left| \frac { \mathrm { d } } { \mathrm { d } \theta } F ( U _ { \theta } ) \right| = \frac { | \dot { T } _ { \theta } | } { T _ { \theta } } \le 2 k \| \dot { B } _ { \theta } \| _ { \mathrm { o p } } \le 4 k \xi \| K \| _ { \mathrm { o p } } \le 4 k \xi \| K \| _ { \mathrm { F } }\tag{84}
$$

The Lipschitz constant for F thus satisfies $L \leq 4 k \xi$

It remains for us to bound log $\mathbb { E } _ { U } e ^ { F ( U ) } = \log \mathbb { E } _ { U } \langle \bar { \psi } | \Gamma _ { k } ( \mathbb { I } + \xi U D U ^ { \dagger } ) | \bar { \psi } \rangle$ . Recall that $( \Gamma _ { k } , \mathrm { S y m } ^ { k } ( \mathbb { C } ^ { n } ) )$ is an irreducible representation of $\mathbb { U } ( n )$ . As a direct consequence of Schur’s Lemma and the fact that Schur polynomials give the characters of $\Gamma _ { k }$

$$
\frac { \mathbb { E } } { U } \langle \psi | \Gamma _ { k } ( U ( \mathbb { I } + \xi D ) U ^ { \dagger } ) | \psi \rangle = \langle \psi | \frac { \operatorname { T r } \Gamma _ { k } ( \mathbb { I } + \xi D ) } { \operatorname { T r } P _ { k } ^ { ( n ) } } P _ { k } ^ { ( n ) } | \psi \rangle = \bar { h } _ { k } ( \mathbf { 1 } + \xi d ) .\tag{85}
$$

Here $\bar { h } _ { k } ( \pmb { y } ) = h _ { k } ( \pmb { y } ) / h _ { k } ( \pmb { 1 } )$ is the normalized complete homogeneous polynomial. Equivalently, $h _ { k } = s _ { ( k ) }$ is a Schur polynomial corresponding to the one-row partition. A useful characterization of the normalized complete homogeneous polynomials is as follows:

Lemma 11. [HLP52, Proof of Theorem 220] Let $k \in \mathbb N$ and $\pmb { y } = ( y _ { 1 } , \dotsb { \mathit { \Omega } } , y _ { n } )$ such that $y _ { j } > 0$ for all $j \in [ n ]$ . Let $P : = ( P _ { 1 } , \cdot \cdot \cdot , P _ { n } ) \sim$ Dirichlet $\mathopen { } \mathclose \bgroup \left( 1 , \cdots , 1 \aftergroup \egroup \right)$ ). Equivalently, $_ { r }$ is sampled uniformly from the probability simplex. Then it holds that

$$
{ \bar { h } } _ { k } ( { \pmb y } ) = \mathbb { E } ( \sum _ { P } ^ { n } P _ { j } y _ { j } ) ^ { k } \equiv \mathbb { E } _ { P } ^ { } ( P \cdot { \pmb y } ) ^ { k } .\tag{86}
$$

Using this lemma, we obtain

$$
\begin{array} { r } { \underline { { \mathbb { E } } } e ^ { F ( U ) } = \underline { { \mathbb { E } } } ( P \cdot ( { \bf 1 } + \xi d ) ) ^ { k } \overset { \mathrm { ( i ) } } { \geq } ( ( \underline { { \mathbb { E } } } P ) \cdot ( { \bf 1 } + \xi d ) ) ^ { k } \overset { \mathrm { ( i i ) } } { = } 1 . } \end{array}\tag{87}
$$

Here (i) uses Jensen’s inequality and (ii) uses the fact that Z is traceless. We can now use Lemma 10 to conclude

$$
\operatorname { \mathbb { E } } _ { U } F ( U ) \geq - { \frac { 1 2 \pi ^ { 2 } k ^ { 2 } \xi ^ { 2 } } { n } } ,\tag{88}
$$

This proves our bound for the second term of the fixed-transcript likelihood ratio. Putting the two terms together, we obtain

Proposition 2 (Likelihood Ratio Bound for fixed transcript). Fix $X _ { 0 } \in \mathrm { ~ } G _ { \mathrm { g o o d } }$ . For any transcript of measurement outcomes $\pmb { x } : = ( z , \pmb { k } )$ where $k _ { t }$ denotes the measured photon number of the t-th copy,

$$
\log _ { \cal Z } L _ { \Phi _ { X _ { 0 } } ( Z ) } ( x ) \geq - C \xi ^ { 2 } \sum _ { t = 1 } ^ { N } \left( n \nu ^ { 2 } + \frac { k _ { t } ^ { 2 } } { n } \right) ,\tag{89}
$$

where $Z \sim \mathrm { U n i f } ( B _ { \mathrm { o p } , b } ( 0 ) )$ and C is some positive constant.

## 5.5 Likelihood ratio bound II – high probability argument

Now we show a high probability bound for the log-likelihood over measurement transcripts. Note that, although the measurement can be adaptive, the total photon number for each copy is sampled from an i.i.d. distribution. More concretely, let $Y = \Phi _ { X _ { 0 } } ( Z )$ for any $Z \in B _ { \mathrm { o p } , b } ( 0 )$ . Let k be a random variable corresponding to the total photon number of $\rho _ { \mathsf { R } _ { Y } }$ . That is, $k \sim q _ { Y }$ where $q _ { Y } ( k ) : = \operatorname* { d e t } ( \mathbb { I } -$ $\mathsf { R } _ { Y } ) \operatorname { T r } ( \Gamma _ { k } ( \mathsf { R } _ { Y } ) )$

Denote the eigenvalues of $\mathsf { R } _ { Y }$ by $\{ r _ { j } \} _ { j = 1 } ^ { n }$ . As is clear from the proof of Lemma 2, k equals to the sum of n independent geometric random variables ${ \mathrm { G e o m } } ( 1 - r _ { j } )$ . That is,

$$
k = \sum _ { j = 1 } ^ { n } M _ { j } , \quad { \mathrm { ~ w h e r e ~ } } \operatorname* { P r } ( M _ { j } = m ) = ( 1 - r _ { j } ) r _ { j } ^ { m } , \quad m = 0 , 1 , 2 , \cdots\tag{90}
$$

Recall that $\begin{array} { r } { \frac { \nu } { 1 + \nu } \leq r _ { j } \leq \frac { 2 \nu } { 1 + 2 \nu } } \end{array}$ . Combining with the moments of geometric random variables:

$$
\mathbb { E } k ^ { 2 } \leq \sum _ { j = 1 } ^ { n } \operatorname { V a r } M _ { j } + ( \sum _ { j = 1 } ^ { n } \mathbb { E } M _ { j } ) ^ { 2 } \leq 1 2 \nu ^ { 2 } n ^ { 2 } .\tag{91}
$$

Combining this with Proposition 2 gives us the following:

Proposition 3 (High-probability Likelihood Ratio Bound). Fix $X _ { 0 } ~ \in ~ G _ { \mathrm { g o o d } }$ and $\delta \in \mathsf { \Gamma } ( 0 , 1 )$ . With probability at least $1 - \delta$ over x $\sim \mathcal { T } _ { X _ { 0 . } }$ 2

$$
\underset { Z } { \mathbb { E } } L _ { \Phi _ { X _ { 0 } } ( Z ) } ( \pmb { x } ) \geq \exp \left( - \frac { C ^ { \prime \prime } } { \delta } N n \nu ^ { 2 } \xi ^ { 2 } \right) ,\tag{92}
$$

where $C ^ { \prime \prime }$ is some positive constant.

Proof. Apply Markov’s inequality to Proposition 2 and then exponentiate both sides.

## 5.6 Putting everything together

Proposition 4 (Posterior anti-concentration). Assume $N = o ( n / ( \nu ^ { 2 } \xi ^ { 2 } ) )$ . Then, for any $X _ { 0 } \in G _ { \mathrm { g o o d } }$ and a small absolute constant $\delta _ { 0 } \in ( 0 , 0 . 0 1 ]$ , with probability at least $1 - e ^ { - n ^ { 2 } } - \delta _ { 0 }$ over x $\sim \mathcal { T } _ { X _ { 0 } } ,$

$$
\nu _ { \pmb { x } } ( B _ { \mathrm { t r } , n a } ( X _ { 0 } ) ) \leq \exp ( - \Omega ( n ^ { 2 } ) ) .\tag{93}
$$

Proof. Combining Proposition 1 and Proposition 3 with a union bound, the following holds with probability at least $1 - e ^ { - n ^ { 2 } } - \delta _ { 0 }$ for any fixed $X _ { 0 } \in G _ { \mathrm { g o o d } }$

$$
\begin{array} { r l } { \displaystyle \frac { \nu _ { x } ( B _ { \mathrm { t r } , n a } ( X _ { 0 } ) ) } { \nu _ { x } ( B _ { \Phi , b } ( X _ { 0 } ) ) } \leq \exp \left( - \Omega ( n ^ { 2 } ) + \frac { C ^ { \prime \prime } } { \delta _ { 0 } } N n \nu ^ { 2 } \xi ^ { 2 } \right) } & { } \\ { \displaystyle } & { = \exp \left( - \Omega ( n ^ { 2 } ) + \frac { C ^ { \prime \prime } } { \delta _ { 0 } } o ( n ^ { 2 } ) \right) } \\ { \displaystyle } & { = \exp \left( - \Omega ( n ^ { 2 } ) \right) . } \end{array}\tag{94}
$$

The proposition follows by relaxing the denominator $\nu _ { \pmb { x } } ( B _ { \Phi , b } ( X _ { 0 } ) ) \leq 1$

Proposition 5 (Hardness of learning $X _ { 0 } )$ . Let $\{ \rho _ { \mathsf { R } _ { X _ { 0 } } } : X _ { 0 } \in G _ { \operatorname { s u p p } } \}$ be the family ofpassive Gaussian states as defined before, with the prior distribution $X _ { 0 } \sim \mu$ . Let A denote a (possibly randomized) algorithm that takes as input the transcript of outcomes x and output an estimator $\mathcal { A } ( \pmb { x } ) \in \mathrm { H e r m } _ { 0 } ( n )$ for X<sub>0</sub>. Given that $N = o ( n / ( \nu ^ { 2 } \xi ^ { 2 } ) )$ , one must have

$$
\operatorname* { P r } _ { \substack { A , X _ { 0 } \sim \mu , \pmb x \sim T _ { X _ { 0 } } } } \left( \| \pmb { \mathcal { A } } ( \pmb x ) - X _ { 0 } \| _ { \mathrm { t r } } \leq n a / 2 \right) \leq 0 . 1 + o ( 1 ) .\tag{95}
$$

Here A is also used to denote the internal randomness of the algorithm.

Proof. By Proposition 4 and the fact that $\operatorname* { P r } _ { X _ { 0 } \sim \mu } ( X _ { 0 } \in G _ { \mathrm { g o o d } } ) \geq 1 - e ^ { - \Omega ( n ) } \quad$

$$
\operatorname* { P r } _ { X _ { 0 } \sim \mu , \boldsymbol { x } \sim \mathcal { T } _ { X _ { 0 } } } [ \nu _ { \boldsymbol { x } } ( B _ { \mathrm { t r } , n a } ( X _ { 0 } ) ) \leq \exp ( - \Omega ( n ^ { 2 } ) ) ] \geq 1 - \delta _ { 0 } - o ( 1 ) .\tag{96}
$$

Combined with the trivial bound $\nu _ { \mathbf { x } } ( B _ { \mathrm { t r } , n a } ( X _ { 0 } ) ) \leq 1$ ，

$$
\underset { X _ { 0 } \sim \mu , \pmb { x } \sim \mathcal { T } _ { X _ { 0 } } } { \mathbb { E } } \nu _ { \pmb { x } } ( B _ { \mathrm { t r } , n a } ( X _ { 0 } ) ) \leq \delta _ { 0 } + o ( 1 ) .\tag{97}
$$

By Bayes’ rule, compared to sampling first $X _ { 0 } \sim \mu$ and then $\begin{array} { r } { \mathbf { \boldsymbol { x } } \sim \mathcal { T } _ { X _ { 0 } : } } \end{array}$ , it is equivalent to first sampling $X ^ { \prime } \sim \mu$ and $\mathbf { \boldsymbol { x } } \sim \mathcal { T } _ { \boldsymbol { X ^ { \prime } } }$ , and then sampling $X _ { 0 } \sim \nu _ { x }$ . Thus, we can rewrite the above as

$$
\operatorname* { \mathbb { E } } _ { X ^ { \prime } \sim \mu , \pmb { x } \sim \mathcal { T } _ { X ^ { \prime } } } \operatorname* { \mathbb { E } } _ { X _ { 0 } \sim \nu _ { x } } \nu _ { \pmb { x } } ( B _ { \mathrm { t r } , n a } ( X _ { 0 } ) ) \leq \delta _ { 0 } + o ( 1 ) .\tag{98}
$$

Now, for any fixed x, sample independently $X _ { 1 } , X _ { 2 } \sim \nu _ { x }$ . Let $\scriptstyle A ( { \pmb x } )$ be a realization of the estimator conditioned on its internal randomness. Consider the following two events:

$$
\bullet B _ { 1 } \colon X _ { 1 } \in B _ { \mathrm { t r } , n a / 2 } ( A ( \pmb { x } ) ) \ \mathrm { a n d } \ X _ { 2 } \in B _ { \mathrm { t r } , n a / 2 } ( A ( \pmb { x } ) ) .
$$

$$
\bullet E _ { 2 } \colon X _ { 2 } \in B _ { \mathrm { t r } , n a } ( X _ { 1 } ) .
$$

By the triangle inequality, $E _ { 1 } \Rightarrow E _ { 2 }$ , and thus $\operatorname* { P r } ( E _ { 1 } ) \leq \operatorname* { P r } ( E _ { 2 } )$ . That is,

$$
\begin{array} { r } { \nu _ { \pmb { x } } ( B _ { \mathrm { t r } , n a / 2 } ( \pmb { \mathcal { A } } ( \pmb { x } ) ) ) ^ { 2 } \leq \underset { X _ { 1 } \sim \nu _ { \pmb { x } } } { \mathbb { E } } \nu _ { \pmb { x } } ( B _ { \mathrm { t r } , n a } ( X _ { 1 } ) ) . } \end{array}\tag{99}
$$

Taking the expectation over $X ^ { \prime } \sim \mu , x \sim \mathcal { T } _ { X ^ { \prime } }$ and the internal randomness of A on both sides,

$$
\begin{array} { r l } & { \underset { x , \mathcal { A } } { \mathbb { E } } \nu _ { x } ( B _ { \mathrm { t r } , n a / 2 } ( \mathcal { A } ( \pmb { x } ) ) ) ^ { 2 } \leq \underset { x } { \mathbb { E } } \underset { X _ { 1 } \sim \nu _ { x } } { \mathbb { E } } \nu _ { x } ( B _ { \mathrm { t r } , n a } ( X _ { 1 } ) ) } \\ & { \qquad \quad \leq \delta _ { 0 } + o ( 1 ) . } \end{array}\tag{100}
$$

By Cauchy-Schwarz,

$$
\begin{array} { r } { \underset { { \boldsymbol { x } } , \boldsymbol { A } } { \mathbb { E } } \nu _ { \boldsymbol { x } } ( B _ { \mathrm { t r } , n a / 2 } ( \boldsymbol { A } ( \boldsymbol { x } ) ) ) \leq \sqrt { \delta _ { 0 } + o ( 1 ) } . } \end{array}\tag{101}
$$

Again by Bayes’ rule, the L.H.S. equals to,

$$
\operatorname* { P r } _ { A , X _ { 0 } \sim \mu , \mathbf { x } \sim T _ { X _ { 0 } } } \left( X _ { 0 } \in B _ { \mathrm { t r } , n a / 2 } ( \mathcal { A } ( \pmb { x } ) ) \right) = \operatorname* { P r } _ { \substack { A , X _ { 0 } \sim \mu , \mathbf { x } \sim T _ { X _ { 0 } } } } \left( \lVert A ( \pmb { x } ) - X _ { 0 } \rVert _ { \mathrm { t r } } \leq n a / 2 \right) .
$$

The proof is finalized by noting that $\delta _ { 0 } \leq 0 . 0 1$

Finally, we need the following lemma that relates estimation of X to estimation of $\rho _ { \mathsf { R } _ { X } }$

Lemma 12 (Trace distance bound via fugacity matrix). For every $a > 0 ,$ , there exist constants $C _ { a } >$ $0 , \alpha _ { a } > 0$ such that, whenever $\xi \le \alpha _ { a } / \sqrt { n \nu } , i f X _ { 1 } , X _ { 2 } \in G _ { \mathrm { s u p p } }$ and

$$
\| X _ { 2 } - X _ { 1 } \| _ { \mathrm { t r } } \geq a n .\tag{102}
$$

Then,

$$
\| \rho _ { \mathsf { R } _ { X _ { 2 } } } - \rho _ { \mathsf { R } _ { X _ { 1 } } } \| _ { \mathrm { t r } } \geq C _ { a } \xi \sqrt { n \nu } .\tag{103}
$$

We postpone the proof for Lemma 12 to Sec. 5.7. Below, we prove the main lower bound.

Proof of Theorem 5. Fix $b = 0 . 1$ and choose an suficiently small constant $a > 0$ such that Proposition 1 holds with appropriate constants. Also fix $\delta = 0 . 0 1$

Assume for contradiction that $N = o ( n ^ { 2 } / ( \nu \varepsilon ^ { 2 } ) )$ ) copies sufice to learn an n-mode passive Gaussian state $\rho \operatorname { t o } \varepsilon$ trace distance with $2 / 3$ probability using a single-copy adaptive protocol (denoted by $\mathcal { T } )$ Consider the family $\{ \rho _ { \mathsf { R } _ { X } } : X \in G _ { \mathsf { s u p p } } \}$ with prior $\mu ,$ where we set $\xi : = \frac { 6 \varepsilon } { C _ { a / 2 } \sqrt { n \nu } }$ for $C _ { a / 2 }$ to be the constant from Lemma 12. By taking the theorem’s constant upper bound on ε small enough, this choice satisfies all earlier smallness assumptions on ξ. Note this family satisfies $\begin{array} { r } { ( \frac { 1 } { 2 } + \nu ) \mathbb { I } \leq \Sigma \leq ( \frac { 1 } { 2 } + 2 \nu ) \mathbb { I } } \end{array}$ I as has been shown above. Denote the measurement outcome by ${ \mathbf { } } x ,$ and the state estimator by ${ \hat { \rho } } ( { \pmb x } )$ that allows internal randomness. By assumption,

$$
\operatorname* { P r } _ { X \sim \mu , \pmb { x } \sim \mathcal { T } _ { X } , \hat { \rho } } \left( \frac { 1 } { 2 } \| \hat { \rho } ( \pmb { x } ) - \rho _ { \mathsf { R } _ { X } } \| _ { \mathrm { t r } } \leq \varepsilon \right) \geq 2 / 3 .\tag{104}
$$

We now define an estimator for $X$ from $\hat { \rho } ( { \pmb x } )$ . We do so by introducing a finite covering net to avoid complication from measure theory. Let $\eta ~ = ~ \varepsilon$ . Since $G _ { \mathrm { s u p p } }$ is compact and the map $X \mapsto \rho _ { \mathsf { R } _ { X } }$ is continuous in trace norm (which can be shown by appropriate photon number cutof), the image $\{ \rho _ { \mathsf { R } _ { X } } : X \in G _ { \mathsf { s u p p } } \}$ has an $\eta \cdot$ -covering net in trace norm: Let $\mathcal { N } _ { \eta } \subset G _ { \mathrm { s u p p } }$ be a finite set such that for every $X \in G _ { \mathrm { s u p p } }$ there exists $\mathcal { N } ( X ) \in \mathcal { N } _ { \eta }$ such that

$$
\| \rho _ { \mathsf { R } _ { X } } - \rho _ { \mathsf { R } _ { N ( X ) } } \| _ { \mathrm { t r } } \leq \eta .\tag{105}
$$

Our estimator of X is defined as follows,

$$
\hat { X } ( \pmb { x } ) : = \arg \operatorname* { m i n } _ { Y \in \mathcal { N } _ { \eta } } \| \hat { \rho } ( \pmb { x } ) - \rho _ { \mathsf { R } _ { Y } } \| _ { \mathrm { t r } } ,\tag{106}
$$

where we break ties in an arbitrary fixed order. Now, conditioned on the event that $\| \hat { \rho } ( { \pmb x } ) - \rho _ { \mathsf { R } _ { X } } \| _ { \mathrm { t r } } / 2 \le \varepsilon ,$

$$
\begin{array} { r l } & { \| \rho _ { \mathsf { R } _ { \hat { X } ( \pmb { x } ) } } - \rho _ { \mathsf { R } _ { X } } \| _ { \mathrm { t r } } \overset { \mathrm { ( i ) } } { \leq } \| \rho _ { \mathsf { R } _ { \hat { X } ( \pmb { x } ) } } - \hat { \rho } ( \pmb { x } ) \| _ { \mathrm { t r } } + \| \hat { \rho } ( \pmb { x } ) - \rho _ { \mathsf { R } _ { X } } \| _ { \mathrm { t r } } } \\ & { \qquad \overset { \mathrm { ( i i ) } } { \leq } \| \rho _ { \mathsf { R } _ { \hat { X } ( \pmb { X } ) } } - \hat { \rho } ( \pmb { x } ) \| _ { \mathrm { t r } } + \| \hat { \rho } ( \pmb { x } ) - \rho _ { \mathsf { R } _ { X } } \| _ { \mathrm { t r } } } \\ & { \qquad \overset { \mathrm { ( i i i ) } } { \leq } \| \rho _ { \mathsf { R } _ { \hat { X } ( \pmb { X } ) } } - \rho _ { \mathsf { R } _ { X } } \| _ { \mathrm { t r } } + 2 \| \hat { \rho } ( \pmb { x } ) - \rho _ { \mathsf { R } _ { X } } \| _ { \mathrm { t r } } } \\ & { \qquad \overset { \mathrm { ( i v ) } } { \leq } \eta + 4 \varepsilon = 5 \varepsilon < C _ { a / 2 } \xi \sqrt { n \nu } . } \end{array}\tag{107}
$$

Here, (i) and (iii) is by the triangle inequality; (ii) uses the definition of $\hat { X } ( { \pmb x } )$ ; (iv) uses our assumption and the definition of the η-covering net. Lemma 12 then implies that

$$
\| \rho _ { \mathsf { R } _ { { \hat { X } } ( \pmb { x } ) } } - \rho _ { \mathsf { R } _ { X } } \| _ { \mathrm { t r } } < C _ { a / 2 } \xi \sqrt { n \nu } \implies \| { \hat { X } } ( \pmb { x } ) - X \| _ { \mathrm { t r } } < a n / 2 .\tag{108}
$$

That is, $\| \hat { X } ( \pmb { x } ) - X \| _ { \mathrm { t r } } < a n / 2$ holds with probability at least $2 / 3$ . However, Proposition 5 states that with $N = o ( n / ( \nu ^ { 2 } \xi ^ { 2 } ) ) = o ( n ^ { 2 } / ( \nu \varepsilon ^ { 2 } ) )$ samples, this can only succeed with no more than $0 . 1 + o ( 1 )$ probability. This leads to contradiction. We thus conclude that $N = \Omega ( n ^ { 2 } / ( \nu \varepsilon ^ { 2 } ) )$ samples are necessary for the learning task. □

## 5.7 Delayed proof for Lemma 12

We first introduce two lemmas that are needed for our concentration argument.

Lemma 13 (A coarse Lévy concentration bound). Let W be a real-valued random variable and set

$$
s ^ { 2 } : = 1 + \operatorname { V a r } ( W ) .\tag{109}
$$

For $h > 0 ,$ define its concentration function

$$
\mathcal { Q } _ { W } ( h ) : = \operatorname* { s u p } _ { t \in \mathbb { R } } \operatorname* { P r } ( t - h < W \leq t ) .\tag{110}
$$

$I f 1 \leq h \leq s ,$ then

$$
Q _ { W } ( h ) \geq c { \frac { h } { s } }\tag{111}
$$

for an absolute constant $c > 0 .$

Proof. Let $\mu : = \mathbb { E } W$ . Since $s ^ { 2 } = 1 + \mathrm { V a r } ( W )$ , Chebyshev’s inequality gives

$$
\operatorname* { P r } ( | W - \mu | < 2 s ) \geq 1 - { \frac { \operatorname { V a r } ( W ) } { 4 s ^ { 2 } } } \geq { \frac { 3 } { 4 } } .\tag{112}
$$

The interval $( \mu - 2 s , \mu + 2 s )$ can be covered by at most

$$
\left\lceil \frac { 4 s } { h } \right\rceil \leq \frac { 5 s } { h } .\tag{113}
$$

Here, the last inequality uses $h \leq s .$ . At least one of these intervals therefore has probability at least $3 h / ( 2 0 s )$ , proving Eq. (111). This is the elementary form of the concentration–variance principle introduced by Lévy; see, e.g., [FHS90] for sharp versions and extensions. □

Lemma 14 (Threshold separation under convolution order). Let $K _ { - }$ and $K _ { + }$ be nonnegative integervalued random variables. Suppose that on an auxiliary probability space one can write

$$
{ \cal K } _ { + } \stackrel { \mathrm { ~ d ~ } } { = } { \cal K } _ { - } + { \cal D } , \qquad { \cal D } \ i s \ n o n n e g a t i v e \ a n d \ i n t e g e r { \cdot } \nu a l u e d , \qquad { \cal D } \ i s \ i n d e p e n d e n t \ o f { \cal K } _ { - } .\tag{114}
$$

Set

$$
\delta : = \mathbb { E } D , \qquad s ^ { 2 } : = 1 + \mathrm { V a r } ( K _ { - } ) .\tag{115}
$$

Assume that $\delta > 0 , \delta \leq s ,$ and

$$
\mathbb { E } D ^ { 2 } \le C _ { D } ( \delta + \delta ^ { 2 } )\tag{116}
$$

for some fixed constant $C _ { D }$ . Then there exists $t \in \mathbb { R }$ such that

$$
\operatorname* { P r } ( K _ { + } > t ) - \operatorname* { P r } ( K _ { - } > t ) \ge c _ { C _ { D } } \frac { \delta } { s } ,\tag{117}
$$

where $c _ { C _ { D } } > 0$ depends only on $C _ { D }$

Proof. The representation in Eq. (114) is precisely the convolution-order relation $K _ { - } \le _ { \mathrm { c o n v } } K _ { + }$ . See [SS07, Section 1.D]. For every $h > 0$ and $t \in \mathbb { R } _ { : }$ , monotonicity of the increment gives

$$
\begin{array} { r l } { \operatorname* { P r } ( K _ { + } > t ) - \operatorname* { P r } ( K _ { - } > t ) = \operatorname* { P r } ( K _ { - } \leq t < K _ { - } + D ) ~ } & { } \\ { \geq \operatorname* { P r } ( D \geq h ) \operatorname* { P r } ( t - h < K _ { - } \leq t ) , } \end{array}\tag{118}
$$

where independence is used in the second line. Taking the supremum over $t ,$

$$
\operatorname* { s u p } _ { t } [ \operatorname* { P r } ( K _ { + } > t ) - \operatorname* { P r } ( K _ { - } > t ) ] \ge \operatorname* { P r } ( D \ge h ) \mathcal { Q } _ { K _ { - } } ( h ) .\tag{119}
$$

We choose h according to the size of δ. If $0 ~ < ~ \delta ~ \leq ~ 2$ , take $h = 1$ . Since D is nonnegative and integer-valued,

$$
\begin{array} { r l } & { \delta = \mathbb { E } [ D \mathbf { 1 } _ { \{ D > 0 \} } ] } \\ & { \quad \leq ( \mathbb { E } D ^ { 2 } ) ^ { 1 / 2 } \operatorname* { P r } ( D > 0 ) ^ { 1 / 2 } . } \end{array}\tag{120}
$$

Equation (116) thus gives

$$
\operatorname* { P r } ( D \geq 1 ) \geq \frac { \delta ^ { 2 } } { \mathbb { E } D ^ { 2 } } \geq c _ { C _ { D } } \delta .\tag{121}
$$

Because $1 \leq s ,$ Lemma 13 gives $\mathcal { Q } _ { K _ { - } } ( 1 ) \geq c / s$ . Substitution into Eq. (119) yields Eq. (117).

If $\delta > 2 ,$ , take $h = \delta / 2$ . The assumption $\delta < s$ ensures $1 < h \leq s$ . Paley–Zygmund gives

$$
\operatorname* { P r } \biggl ( D \geq \frac { \delta } { 2 } \biggr ) \geq \frac { ( 1 - 1 / 2 ) ^ { 2 } ( { \mathbb { E } } D ) ^ { 2 } } { { \mathbb { E } } D ^ { 2 } } \geq c _ { C _ { D } } .\tag{122}
$$

By Lemma 13, $\mathcal { Q } _ { K _ { - } } ( \delta / 2 ) \ge c \delta / s$ . Equation (119) again yields Eq. (117). Finally, because $K _ { - }$ and $K _ { + }$ are integer-valued, the supremum of their tail-probability diference is attained at some threshold $t . \sqsubset$

Now we proceed to the proof of Lemma 12.

Proof of Lemma 12. Assume that $\| X _ { 1 } - X _ { 2 } \| _ { \mathrm { t r } } \geq n a$ . We will construct a two-outcome photon-counting measurement whose outcome distributions under $\rho _ { \mathsf { R } _ { X _ { \mathrm { 1 } } } }$ and $\rho _ { \mathsf { R } _ { X _ { 2 } } }$ have total variation distance at least a constant multiple of $\xi \sqrt { n \nu }$ . Below, $c , C > 0$ denote absolute constants, while $c _ { a } , C _ { a } ^ { \prime } > 0$ may depend only on a.

For $X \in G _ { \mathrm { s u p p } }$ , recall the photon occupation matrix

$$
\begin{array} { r l } & { { \sf N } _ { X } : = { \sf R } _ { X } ( { \mathbb I } - { \sf R } _ { X } ) ^ { - 1 } } \\ & { \quad \quad = \bar { \nu } ( { \mathbb I } + \xi X ) ( { \mathbb I } - \bar { \nu } \xi X ) ^ { - 1 } } \\ & { \quad \quad = ( 1 + \bar { \nu } ) ( { \mathbb I } - \bar { \nu } \xi X ) ^ { - 1 } - { \mathbb I } , } \end{array}\tag{123}
$$

where we used $\bar { \nu } = r / ( 1 - r )$ . Define

$$
\Delta : = \mathsf { N } _ { X _ { 1 } } - \mathsf { N } _ { X _ { 2 } } .\tag{124}
$$

The resolvent identity gives

$$
\Delta = ( 1 + \bar { \nu } ) \bar { \nu } \xi ( \mathbb { I } - \bar { \nu } \xi X _ { 1 } ) ^ { - 1 } ( X _ { 1 } - X _ { 2 } ) ( \mathbb { I } - \bar { \nu } \xi X _ { 2 } ) ^ { - 1 } .\tag{125}
$$

Since $\| X _ { i } \| _ { \mathrm { o p } } \leq 4 , \bar { \nu } \leq 4 / 3$ , and $\xi \le 1 / 3 2$ , the two resolvent factors in the preceding display and their inverses have uniformly bounded operator norms. In particular, it can be rewritten as

$$
X _ { 1 } - X _ { 2 } = \frac { ( \mathbb { I } - \bar { \nu } \xi X _ { 1 } ) \Delta ( \mathbb { I } - \bar { \nu } \xi X _ { 2 } ) } { ( 1 + \bar { \nu } ) \bar { \nu } \xi }\tag{126}
$$

and using the ideal property of the trace norm, we obtain

$$
\begin{array} { r l } & { \| \Delta \| _ { \mathrm { t r } } \geq c \bar { \nu } \xi \| X _ { 1 } - X _ { 2 } \| _ { \mathrm { t r } } \geq c _ { a } n \bar { \nu } \xi , } \\ & { \| \Delta \| _ { \mathrm { o p } } \leq C \bar { \nu } \xi . } \end{array}\tag{127}
$$

The same spectral bounds applied to Eq. (123) show that, uniformly over $X \in G _ { \mathrm { s u p p } } ,$

$$
c { \bar { \nu } } \mathbb { I } \leq \mathbb { N } _ { X } \leq C { \bar { \nu } } \mathbb { I } .\tag{128}
$$

Let $\Pi _ { + }$ denote the projector onto the positive-eigenvalue subspace of $\Delta ,$ and define $\Delta _ { + } : = \Pi _ { + } \Delta \Pi _ { + }$ For a Hermitian matrix, at least one of the positive parts of $\Delta$ and $- \Delta$ has trace at least $\| \Delta \| _ { \mathrm { t r } } / 2 .$ By interchanging $X _ { 1 }$ and $X _ { 2 }$ if necessary, and then redefining $\Pi _ { + }$ and $\Delta _ { + }$ accordingly, we may therefore assume that

$$
\mathrm { T r } \Delta _ { + } \geq \frac { 1 } { 2 } \Vert \Delta \Vert _ { \mathrm { t r } } ,\tag{129}
$$

Regard the following compressions as operators on ran $\Pi _ { + }$ :

$$
B _ { + } : = \Pi _ { + } { \sf N } _ { X _ { 1 } } \Pi _ { + } , \qquad B _ { - } : = \Pi _ { + } { \sf N } _ { X _ { 2 } } \Pi _ { + } .\tag{130}
$$

Set

$$
m : = \mathrm { r a n k } ( \Pi _ { + } ) , \qquad \delta _ { \mathrm { p h } } : = \mathrm { T r } ( B _ { + } - B _ { - } ) = \mathrm { T r } \Delta _ { + } .\tag{131}
$$

By construction,

$$
B _ { + } - B _ { - } = \Pi _ { + } \Delta \Pi _ { + } = \Delta _ { + } \geq 0 , \qquad \delta _ { \mathrm { p h } } \geq c _ { a } n \bar { \nu } \xi .\tag{132}
$$

In particular, this implies $B _ { + } \geq B _ { }$ <sub>−</sub>. Moreover,

$$
m \ge \frac { \delta _ { \mathrm { p h } } } { \| \Delta \| _ { \mathrm { o p } } } \ge c _ { a } n , \qquad c \bar { \nu } \Pi _ { + } \le B _ { \pm } \le C \bar { \nu } \Pi _ { + } .\tag{133}
$$

We now measure the total photon number in the mode subspace ran $\Pi _ { + }$ . Equivalently, one may first apply a passive Gaussian unitary that maps this subspace to the first m modes, and then photoncount those modes. Let $\widehat { K }$ denote the corresponding photon-number operator, and let $K _ { X }$ denote its measurement outcome on $\rho _ { \mathsf { R } _ { X } }$ . For $X \in G _ { \mathrm { s u p p } }$ , write $B _ { X } : = \Pi _ { + } \mathsf { N } _ { X } \Pi _ { + }$ , viewed as an operator on ran $\Pi _ { + }$ , so that $B _ { X _ { 1 } } = B _ { + }$ and $B _ { X _ { 2 } } = B _ { - }$ . The reduced state on these $m$ modes is again a passive Gaussian state, and its photon occupation matrix is $B _ { X }$ : taking the marginal simply restricts the twopoint function $\mathsf { N } _ { X }$ to ran $\Pi _ { + }$ . Let $\lambda _ { 1 } ( X ) \leq \cdots \leq \lambda _ { m } ( X )$ be the eigenvalues of $B _ { X }$ . A passive Gaussian unitary within ran Π diagonalizes $B _ { X }$ without changing the total photon number. By the thermal normal form (Eq. (14)), in that basis the reduced state is a product of single-mode thermal states with mean photon numbers $\lambda _ { i } ( X )$ . Hence the individual mode counts $M _ { i , X }$ are independent geometric random variables satisfying

$$
\begin{array} { r l r } & { } & { \displaystyle \operatorname* { P r } ( M _ { i , X } = k ) = \frac { 1 } { 1 + \lambda _ { i } ( X ) } \left( \frac { \lambda _ { i } ( X ) } { 1 + \lambda _ { i } ( X ) } \right) ^ { k } , \qquad k \in \{ 0 , 1 , 2 , \ldots \} , } \\ & { } & { \displaystyle \mathbb { E } z ^ { M _ { i , X } } = g _ { \lambda _ { i } ( X ) } ( z ) : = \frac { 1 } { 1 + \lambda _ { i } ( X ) ( 1 - z ) } , \qquad 0 \le z \le 1 . } \end{array}\tag{134}
$$

The total count is their sum, and independence makes the probability generating functions multiply. Therefore,

$$
\begin{array} { l } { \displaystyle { \cal K } _ { X } \triangleq \sum _ { i = 1 } ^ { m } M _ { i , X } , } \\ { \displaystyle \mathop { { \mathbb E } _ { X _ { X } } } z ^ { K _ { X } } = \prod _ { i = 1 } ^ { m } \frac { 1 } { 1 + \lambda _ { i } ( X ) ( 1 - z ) } = \prod _ { \mathrm { r a n } \Pi _ { + } } ^ { \mathrm { d e t } } \left( \mathbb { I } _ { \mathrm { r a n } \Pi _ { + } } + ( 1 - z ) B _ { X } \right) ^ { - 1 } . } \end{array}\tag{135}
$$

Set $\lambda _ { i } ^ { + } : = \lambda _ { i } ( X _ { 1 } )$ and $\lambda _ { i } ^ { - } : = \lambda _ { i } ( X _ { 2 } )$ . Since $B _ { + } \geq B _ { - }$ , Weyl’s monotonicity theorem yields ${ \lambda } _ { i } ^ { + } \geq { \lambda } _ { i } ^ { - }$ for every i. Writing $M _ { i } ^ { + } : = M _ { i , X _ { 1 } }$ and $M _ { i } ^ { - } : = M _ { i , X _ { 2 } }$ , Eq. (135) gives

$$
K _ { + } : = K _ { X _ { 1 } } \stackrel { \mathrm { d } } { = } \sum _ { i = 1 } ^ { m } M _ { i } ^ { + } , \qquad K _ { - } : = K _ { X _ { 2 } } \stackrel { \mathrm { d } } { = } \sum _ { i = 1 } ^ { m } M _ { i } ^ { - } ,\tag{136}
$$

where the variables within each sum are independent and $M _ { i } ^ { \pm }$ is geometric on $\{ 0 , 1 , 2 , \ldots \}$ with mean ${ \lambda } _ { i } ^ { \pm }$

We next construct the convolution coupling required by Lemma 14. By Eq. (133), $\lambda _ { i } ^ { + } \geq \lambda _ { i } ^ { - } \geq c \bar { \nu } > 0$ Taking the ratio between the PGFs of $M _ { i } ^ { \bar { \pm } }$ gives

$$
\begin{array} { c } { { \displaystyle \frac { g _ { \lambda _ { i } ^ { + } } ( z ) } { g _ { \lambda _ { i } ^ { - } } ( z ) } = \displaystyle \frac { 1 + \lambda _ { i } ^ { - } ( 1 - z ) } { 1 + \lambda _ { i } ^ { + } ( 1 - z ) } } } \\ { { = \displaystyle \frac { \lambda _ { i } ^ { - } } { \lambda _ { i } ^ { + } } + \left( 1 - \displaystyle \frac { \lambda _ { i } ^ { - } } { \lambda _ { i } ^ { + } } \right) g _ { \lambda _ { i } ^ { + } } ( z ) . } } \end{array}\tag{137}
$$

Since $0 < \lambda _ { i } ^ { - } / \lambda _ { i } ^ { + } \leq 1$ , the right-hand side is a convex combination of the PGF 1 of the point mass at zero and the geometric PGF $g _ { \lambda _ { i } ^ { + } }$ . Hence it is itself a PGF of a random variable, denoted by $J _ { i } \geq 0$ . On an auxiliary probability space, we may consequently choose mutually independent variables $M _ { i } ^ { - }$ and $J _ { i }$ such that

$$
M _ { i } ^ { - } + J _ { i } \stackrel { \mathrm { d } } { = } M _ { i } ^ { + } .\tag{138}
$$

Defining

$$
K _ { - } : = \sum _ { i = 1 } ^ { m } M _ { i } ^ { - } , \qquad J : = \sum _ { i = 1 } ^ { m } J _ { i } , \qquad K _ { + } : = K _ { - } + J ,\tag{139}
$$

realizes the two photon-count distributions on a common probability space, with $J \geq 0$ independent of $K _ { - }$ . Since $J _ { i }$ is a convex combination between the point mass at 0 and a geometric random variable with mean ${ \lambda } _ { i } ^ { + }$ , we can obtain that (recall a geometric random variable of mean λ has variance $\lambda ( 1 + \lambda ) )$ )

$$
\begin{array} { l } { \mathbb { E } J _ { i } = \lambda _ { i } ^ { + } - \lambda _ { i } ^ { - } , } \\ { \mathbb { E } J _ { i } ^ { 2 } = ( \lambda _ { i } ^ { + } - \lambda _ { i } ^ { - } ) ( 1 + 2 \lambda _ { i } ^ { + } ) . } \end{array}\tag{140}
$$

Consequently,

$$
\begin{array} { l } { \displaystyle \mathbb { E } J = \sum _ { i = 1 } ^ { m } ( \lambda _ { i } ^ { + } - \lambda _ { i } ^ { - } ) = \delta _ { \mathrm { p h } } , } \\ { \displaystyle \mathbb { E } J ^ { 2 } = \delta _ { \mathrm { p h } } ^ { 2 } + \sum _ { i = 1 } ^ { m } \mathrm { V a r } ( J _ { i } ) \leq C ( \delta _ { \mathrm { p h } } + \delta _ { \mathrm { p h } } ^ { 2 } ) , } \end{array}\tag{141}
$$

where the last inequality uses $\lambda _ { i } ^ { + } \leq C$

Lemma 14 shows that the relevant scale is $\delta _ { \mathrm { p h } } / s$ . We now estimate this ratio and verify the remaining hypothesis $\delta _ { \mathrm { p h } } \leq s$ . Set

$$
\begin{array} { l } { { \displaystyle s ^ { 2 } : = 1 + \mathrm { V a r } ( K _ { - } ) } } \\ { { \displaystyle ~ = 1 + \sum _ { i = 1 } ^ { m } \lambda _ { i } ^ { - } ( 1 + \lambda _ { i } ^ { - } ) . } } \end{array}\tag{142}
$$

Eqs. (133) and $n \bar { \nu } \geq n \nu \geq 1$ imply

$$
c _ { a } n { \bar { \nu } } \leq s ^ { 2 } \leq C n { \bar { \nu } } .\tag{143}
$$

On the other hand, Eqs. (127) and (132) give

$$
c _ { a } n \bar { \nu } \xi \leq \delta _ { \mathrm { p h } } \leq m \| \Delta \| _ { \mathrm { o p } } \leq C n \bar { \nu } \xi .\tag{144}
$$

Therefore,

$$
c _ { a } \xi \sqrt { n \bar { \nu } } \leq \frac { \delta _ { \mathrm { p h } } } { s } \leq C _ { a } ^ { \prime } \xi \sqrt { n \bar { \nu } } .\tag{145}
$$

Since $\bar { \nu } \leq 4 \nu / 3 ,$ , the hypothesis $\xi \le \alpha _ { a } / \sqrt { n \nu }$ ensures $\delta _ { \mathrm { p h } } / s \le C _ { a } ^ { \prime } \sqrt { 4 / 3 } \alpha _ { a }$ . By decreasing $\alpha _ { a }$ if necessary, we may and do assume that $\delta _ { \mathrm { p h } } \leq s$

The coupling in Eq. (139), the moment estimate in Eq. (141), and the bounds above satisfy all the requirements of Lemma 14 with $D = J$ and $\delta = \delta _ { \mathrm { p h } }$ . Therefore, there exists a threshold $\tau \in \mathbb { R }$ such that

$$
\operatorname* { P r } _ { \rho _ { \mathrm { R } _ { X _ { 1 } } } } \left( K _ { X _ { 1 } } > \tau \right) - \operatorname* { P r } _ { \rho _ { \mathrm { R } _ { X _ { 2 } } } } \left( K _ { X _ { 2 } } > \tau \right) \geq c \frac { \delta _ { \mathrm { p h } } } { s } \geq c _ { a } \xi \sqrt { n \bar { \nu } } .\tag{146}
$$

Coarse-graining the photon count into the two outcomes $\{ \widehat { K } > \tau \}$ and $\{ \widehat { K } \leq \tau \}$ gives a classical total variation distance equal to the absolute value of the left-hand side of Eq. (146). By data processing for trace distance and $\bar { \nu } \geq \nu ,$

$$
\frac { 1 } { 2 } \| \rho _ { \mathsf { R } _ { X _ { 1 } } } - \rho _ { \mathsf { R } _ { X _ { 2 } } } \| _ { \mathrm { t r } } \geq c _ { a } \xi \sqrt { n \bar { \nu } } \geq c _ { a } \xi \sqrt { n \nu } .\tag{147}
$$

Renaming $2 c _ { a }$ as $C _ { a }$ proves the stated full trace-norm bound.

## 6 Alternative single-copy lower bound for non-adaptive scheme

This section presents an alternative proof for the single-copy lower bound. This proof only works for non-adaptive schemes, but it holds even if the symplectic eigenvalues of the unknown Gaussian states are a priori known. For simplicity, here we focus only on cold Gaussian states $( { \mathrm { i } } . { \mathrm { e } } . , \nu = \Theta ( 1 / n ) )$ and do not try to derive a ν-dependent lower bound.

Theorem 6. For any single-copy non-adaptive scheme that can learn any n-mode passive Gaussian state $\rho ( 0 , \Sigma )$ , with the promise that $\begin{array} { r } { \frac { 1 } { n } \mathbb { I } \leq \Sigma - \frac { 1 } { 2 } \mathbb { I } \leq \frac { 2 } { n } \mathbb { I } } \end{array}$ , to trace distance precision $\varepsilon \le 0 . 0 1$ with at least $2 / 3$ probability, the necessary number of copies is $N \stackrel { \cdot \cdot } { = } \Omega ( n ^ { 3 } / \varepsilon ^ { 2 } )$ . This holds even if the symplectic eigenvalues of the state are a priori known

This bound can be achieved by the heterodyne measurement [BMEM25].

The following representation theory result is crucial to our proof.

Lemma 15. [FH13, Theorem 6.3] For any $X \in \operatorname { G L } _ { n } ( \mathbb { C } )$ , the trace of $\operatorname { T } _ { \lambda } ( X )$ on $S _ { \lambda } ^ { ( n ) }$ is given by the Schur polynomial on the eigenvalues $x _ { 1 } , \cdots , x _ { n }$ of X on $\mathbb { C } ^ { n } .$ , i.e.,

$$
\operatorname { T r } ( \Gamma _ { \lambda } ( X ) ) = s _ { \lambda } ( x _ { 1 } , \cdot \cdot \cdot , x _ { n } ) .\tag{148}
$$

In particular, the dimension of $S _ { \lambda } ^ { n }$ is given by $\mathrm { T r } ( P _ { \lambda } ^ { ( n ) } ) = \mathrm { T r } ( \Gamma _ { \lambda } ( \mathbb { I } _ { n } ) ) = s _ { \lambda } ( 1 , \cdot \cdot \cdot , 1 ) .$

Proof of Theorem 6. Consider passive Gaussian states of the following form:

$$
\rho _ { U } : = \Gamma ( U ) \rho _ { 0 } \Gamma ( U ) ^ { \dagger } , \quad \mathrm { w h e r e } ~ \rho _ { 0 } : = \tau _ { \nu _ { 1 } } ^ { \otimes s } \otimes \tau _ { \nu _ { 0 } } ^ { \otimes s } .\tag{149}
$$

Here $\begin{array} { r } { s = n / 2 , \nu _ { 0 } = \frac { 1 } { n } , \nu _ { 1 } = \frac { 1 + \xi } { n } } \end{array}$ , where $\xi \le 1$ is a parameter to be chosen later.

We need the following fact: There exists an ensemble of unitary $\{ U _ { a } \} _ { a = 1 } ^ { M }$ with $M = 2 ^ { \Theta ( n ^ { 2 } ) }$ such that

$$
\mathrm { T r } \left( \Pi _ { > s } U _ { a } ^ { \dagger } U _ { b } \Pi _ { \le s } U _ { b } ^ { \dagger } U _ { a } \right) \ge \frac { n } { 8 } , \quad \forall a \ne b \in [ M ] ,\tag{150}
$$

which is used to define our packing net. Here $\Pi { \_ } s$ and $\Pi _ { > s }$ are the projectors onto the last $n - s$ modes and the first s modes, respectively. This is directly implied by Lemma 3.2, Lemma 3.3 and Corollary 3.4 from [LN25]. Basically, it is known that

$$
\operatorname* { P r } _ { U \sim \mathrm { H a a r } } \left( \mathrm { T r } \left( \Pi _ { > s } U \Pi _ { \le s } U \right) \le \frac { n } { 8 } \right) \le \exp ( - \frac { n ^ { 2 } } { 3 2 } ) .\tag{151}
$$

Suppose one has obtained a collection of M $U _ { i }$ that satisfies Eq. (150). Now sample a Haar random $U$ and put it into the ensemble. The probability that the condition is still satisfied is at least $\textstyle 1 - M \exp ( - { \frac { n ^ { 2 } } { 3 2 } } )$ by the union bound. One can thus inductively construct an ensemble of size $M = 2 ^ { \Theta ( n ^ { 2 } ) }$ that satisfies Eq. (150) with a non-zero probability.

To show the packing property, for any $a \neq b \in [ M ]$ , consider the following measurement: First undo $U _ { a } .$ then project the last s mode to vacuum. For $\rho _ { U _ { a } ; }$ , the probability of seeing vacuum is clearly,

$$
\mathrm { P r } ( \mathrm { V a c } _ { > s } | U _ { a } ) = ( 1 + \nu _ { 0 } ) ^ { - s } .\tag{152}
$$

For $\rho _ { U _ { b } ; }$ , define $W : = U _ { a } ^ { \dagger } U _ { b }$ . Then, using the fidelity formula for Gaussian states,

$$
\begin{array} { r l } & { \mathrm { P r } ( \mathrm { V a c } _ { > s } | U _ { b } ) = \operatorname* { d e t } \Bigl ( ( 1 + \nu _ { 0 } )  { \mathrm { I } } + ( \nu _ { 1 } - \nu _ { 0 } )  { \mathrm { I } } _ { > s } W  { \mathrm { I } } _ { \le s } W ^ { \dagger }  { \mathrm { I } } _ { > s } \Bigr ) ^ { - 1 } } \\ & { \stackrel { \mathrm { ( i ) } } { \le } ( 1 + \nu _ { 0 } ) ^ { - s } \left( 1 + \frac { \nu _ { 1 } - \nu _ { 0 } } { 1 + \nu _ { 0 } } \mathrm { T r } (  { \mathrm { I } } _ { > s } W  { \mathrm { I } } _ { \le s } W ^ { \dagger }  { \mathrm { I } } _ { > s } ) \right) ^ { - 1 } } \\ & { \stackrel { \mathrm { ( i i ) } } { \le } ( 1 + \nu _ { 0 } ) ^ { - s } \left( 1 + \frac { \xi } { 1 6 } \right) ^ { - 1 } . } \\ & { \stackrel { \mathrm { ( i i i ) } } { \le } ( 1 + \nu _ { 0 } ) ^ { - s } \left( 1 - \frac { \xi } { 3 2 } \right) . } \end{array}\tag{153}
$$

Here (i) uses det $( \mathbb { I } + X ) \geq 1 + \operatorname { T r } X$ for any $X \geq 0 ; ( \mathrm { i i } )$ uses the condition from Eq. (150); (iii) uses the fact that $( 1 + x ) ^ { - 1 } \leq 1 - x / 2$ for any $x \in [ 0 , 1 ]$ and that $\xi \le 1$ . Thus,

$$
\begin{array} { l } { { \displaystyle D _ { \mathrm { t r } } ( \rho _ { U _ { a } } , \rho _ { U _ { b } } ) \sum _ { \ge \mathrm { \tiny ~ P r } } ( \mathrm { V a c } _ { > s } | U _ { a } ) - \mathrm { P r } ( \mathrm { V a c } _ { > s } | U _ { b } ) } } \\ { ~ \ge \displaystyle \frac { \xi } { 3 2 } ( 1 + \nu _ { 0 } ) ^ { - s } } \\ { ~ \le \displaystyle \frac { ( \mathrm { i i } ) } { 3 2 } \xi \exp ( - s \nu _ { 0 } ) } \\ { ~ \ge \displaystyle \frac { \xi } { 6 4 } . } \end{array}\tag{154}
$$

Here (i) uses the data processing inequality for trace distance; (ii) uses $\log ( 1 + \nu _ { 0 } ) \leq \nu _ { 0 }$ . Thus the packing separation is at least $\xi / 6 4 .$ . For a target trace-distance accuracy ε, we may choose $\xi = 6 4 \varepsilon$ to obtain an ε-packing net. Since $\varepsilon \le 0 . 0 1$ , this choice satisfies $\xi \le 1$ . We keep ξ explicit below and substitute $\xi = \Theta ( \varepsilon )$ at the end; changing the packing convention only rescales ξ by another absolute constant and does not afect the final scaling.

Next, we will use Fano’s argument. Consider the following communication task between Alice and Bob: Alice samples $a \in [ M ]$ uniformly at random and sends N copies of $\rho _ { U _ { a } }$ to Bob, who then performs a single-copy non-adaptive measurement and outputs $\hat { a } \in [ M ]$ as his guess of a. If there exists a singlecopy non-adaptive scheme that can use N copies to learn any passive Gaussian state to trace distance precision $c _ { 1 } / 9$ with $2 / 3$ success probability, then Bob can guess correctly with at least $2 / 3$ average success probability. Denote Bob’s measurement outcomes by ${ \pmb y } : = ( y _ { 1 } , \dotsb , y _ { N } )$ . By Fano’s inequality, we must have

$$
I ( a : y ) \geq I ( a : \hat { a } ) \geq \Omega ( \log M ) = \Omega ( n ^ { 2 } ) .\tag{155}
$$

We now use an argument from [LN25] and $[ \mathrm { H H J ^ { + } 1 7 } ]$ . For any $W \in \mathbb { U } ( n )$ , the ensemble $\{ \rho _ { W U _ { a } } \} _ { a = 1 } ^ { M }$ also satisfies pairwise trace distance at least $c _ { 1 } / 8$ thanks to the unitary invariance of trace distance. For a fixed non-adaptive scheme, use $\pmb { y } _ { W }$ to denote the measurement outcome when the ensemble is $\{ \rho _ { W U _ { a } } \} _ { a = 1 } ^ { M }$ . Also, use z to denote the measurement outcome on $\rho _ { U } ^ { \otimes N }$ with a Haar random U. Let $p ( \cdot | V )$ be the measurement outcome distribution on $\rho _ { V }$ for any $V \in \mathbb { U } ( n )$ . We claim that,

$$
\operatorname { \mathbb { E } } _ { W \sim \mathrm { H a a r } } I ( a : { \pmb y } _ { W } ) \le I ( U : z ) .\tag{156}
$$

Indeed,

$$
\begin{array} { r l } & { \frac { \mathbb { E } } { W } I ( a : y _ { W } ) \overset { \mathrm { ( i ) } } { = } \frac { \mathbb { E } } { W } H ( \underset { a } { \mathbb { E } } p ( \cdot | W U _ { a } ) ) - \underset { W , a } { \mathbb { E } } H ( p ( \cdot | W U _ { a } ) ) } \\ & { \qquad \overset { \mathrm { ( i i ) } } { \leq } H ( \underset { W , a } { \mathbb { E } } p ( \cdot | W U _ { a } ) ) - \underset { W , a } { \mathbb { E } } H ( p ( \cdot | W U _ { a } ) ) } \\ & { \qquad \overset { \mathrm { ( i i i ) } } { = } H \left( \underset { U \sim \mathrm { H a a r } } { \mathbb { E } } p ( \cdot | U ) \right) - \underset { U \sim \mathrm { H a a r } } { \mathbb { E } } H ( p ( \cdot | U ) ) } \\ & { \qquad = I ( U : z ) . } \end{array}\tag{157}
$$

Here, (i) is by definition of mutual information; (ii) uses the concavity of entropy; (iii) use the invariance of Haar measure. The above inequality implies that there exists at least one $W _ { * }$ such that $I ( a : y _ { W _ { * } } ) \leq$ $I ( U : z )$ . Thus, we can replace our packing net by $\{ \rho _ { W _ { * } U _ { a } } \} _ { a = 1 } ^ { M } . ^ { 1 0 }$ In the following, we derive an upper bound on $I ( U : z )$

Since Bob’s scheme is non-adaptive, each outcome $z _ { j }$ is independent conditioned on U. Thus,

$$
\begin{array} { l } { \displaystyle I ( U : z ) \stackrel { ( ) } { = } \displaystyle \sum _ { j = 1 } ^ { N } I ( U : z _ { j } | z _ { < j } ) } \\ { \displaystyle \qquad = \sum _ { j = 1 } ^ { N } H ( z _ { j } | z _ { < j } ) - H ( z _ { j } | U , z _ { < j } ) } \\ { \displaystyle \qquad \stackrel { ( ) ) } { \le } \displaystyle \sum _ { j = 1 } ^ { N } H ( z _ { j } ) - H ( z _ { j } | U ) } \\ { \displaystyle \qquad = \sum _ { j = 1 } ^ { N } I ( U : z _ { j } ) . } \end{array}\tag{158}
$$

Here (i) uses the chain rule of mutual information; (ii) uses the conditional independence of $z _ { j } { } ^ { \dag } s$ on U and the fact that conditioning does not increase entropy. What’s left is to find an appropriate upper bound on $I ( U : z )$ for any single-copy measurement outcome z. Use $p ( z | U )$ to denote the probability

density of the measurement $\{ M _ { z } \}$ when applying the scheme to $\rho _ { U }$ . We have

$$
\begin{array} { r l } &  \Omega ( n ^ { 2 } ) = I ( U : z ) \stackrel { \mathrm { ( i ) } } { = } \frac { \mathbb { E } } { U } \mathrm { K L } \bigg ( p ( \cdot | U ) \Big | \Big | \begin{array} { l } { \frac { \mathbb { E } } { U ^ { \prime } } p ( \cdot | U ^ { \prime } ) \Big ) } \\ { \stackrel { \mathrm { ( i i ) } } { \leq } \frac { \mathbb { E } } { U } \chi ^ { 2 } \bigg ( p ( \cdot | U ) \Big | \Big | \begin{array} { l } { \frac { \mathbb { E } } { U ^ { \prime } } p ( \cdot | U ^ { \prime } ) \Big ) } \end{array} } \\ & { \qquad = \bigg ( \int d z \frac { \mathbb { E } _ { U } p ^ { 2 } ( z | U ) } { \mathbb { E } _ { U ^ { \prime } } p ( z | U ^ { \prime } ) } \bigg ) - 1 = : \star . } \end{array} \end{array}\tag{159}
$$

Here, (i) is by the definition of mutual information; (ii) uses the relation between KL divergence and $\chi ^ { 2 }$ -divergence. The expectation over U should be understood as the Haar average. The goal is to upper bound ⋆.

Any single-copy measurement can be described by a POVM $\{ M _ { z } \} _ { z }$ such that $\textstyle \int d z M _ { z } = \mathbb { I }$ , where each $M _ { z }$ is a PSD operator acting on $\mathcal { H } ^ { ( n ) }$ . We have $p ( z | U ) = \mathrm { T r } ( M _ { z } \rho _ { U } )$ . Since $\rho { } U$ is block-diagonal in the photon-number sector decomposition, replacing $\{ M _ { z } \} _ { z }$ by $\{ \Sigma _ { m = 0 } ^ { \infty } P _ { m } ^ { ( n ) } M _ { z } P _ { m } ^ { ( n ) } \}$ will not change the measurement outcome distributions, where $P _ { m } ^ { ( n ) }$ is the projector onto the m-photon sector $\mathcal { H } _ { m } ^ { ( n ) }$ Therefore, we can assume without loss of generality that each $M _ { z }$ is block-diagonal in the photon-number sector.

Furthermore, we can always assume each $M _ { z }$ is rank-1, because otherwise one can replace $M _ { z }$ by its spectral decomposition and treat each eigen-projector as a separate measurement outcome, which will not decrease the RHS of Eq. (159). Consequently, we only need to consider POVMs of the form $\{ | z ; m \rangle \langle z ; m | \} _ { z , m }$ such that $| z ; m \rangle \in \mathcal { H } _ { m } ^ { ( n ) }$ and $\begin{array} { r } { \int d z | z ; m \rangle \langle z ; m | = P _ { m } ^ { ( n ) } } \end{array}$ for all $m \in \mathbb { N } .$

Write $\rho _ { 0 } : = \oplus _ { m = 0 } ^ { \infty } q _ { m } \rho _ { 0 , m }$ where $\rho _ { 0 , m } \in \mathcal { L } ( \mathcal { H } _ { m } ^ { ( n ) } )$ and $\mathrm { T r } ( \rho _ { 0 , m } ) = 1$ . Here $q _ { m }$ is the probability distribution of $\rho _ { 0 }$ in the m-photon sector. This also implies $\rho _ { U } = \bigoplus _ { m = 0 } ^ { \infty } q _ { m } \Gamma _ { m } ( U ) \rho _ { 0 , m } \Gamma _ { m } ( U ) ^ { \dagger }$ . Thus (suppose we have chosen the optimal measurement),

$$
\begin{array} { l } { { \star + 1 = \displaystyle \sum _ { m = 0 } ^ { \infty } q _ { m } \int d z \frac { \mathbb { E } _ { U } \left. z , m \right| \Gamma _ { m } \left( U \right) \rho _ { 0 , m } \Gamma _ { m } \left( U \right) ^ { \dagger } \left| z , m \right. ^ { 2 } } { \mathbb { E } _ { U } \left. z , m \right| \Gamma _ { m } \left( U \right) \rho _ { 0 , m } \Gamma _ { m } \left( U \right) ^ { \dagger } \left| z , m \right. } } } \\ { { \mathrm { ~ } = : \displaystyle \sum _ { m = 0 } ^ { \infty } q _ { m } \int d z \frac { \mathbb { E } _ { U } \left. z , m \right| \rho _ { U , m } \left| z , m \right. ^ { 2 } } { \mathbb { E } _ { U } \left. z , m \right| \rho _ { U , m } \left| z , m \right. } . } } \end{array}\tag{160}
$$

To proceed, let us first derive an exact formula of $\rho _ { 0 , m }$

$$
\begin{array} { r l } & { \rho _ { 0 , m } \propto \displaystyle \sum _ { | m | = m } \left( \frac { \nu _ { 1 } } { \nu _ { 1 } + 1 } \right) ^ { | m \le s | } \left( \frac { \nu _ { 0 } } { \nu _ { 0 } + 1 } \right) ^ { | m > s | } | m \rangle \langle m | } \\ & { \qquad \propto \displaystyle \sum _ { k = 0 } ^ { m } \alpha ^ { k } \left( \displaystyle \sum _ { | m \le s | = k } | m _ { \le s } \rangle \langle m _ { \le s } | \right) \otimes \left( \displaystyle \sum _ { | m _ { > s } | = m - k } | m _ { > s } \rangle \langle m _ { > s } | \right) } \\ & { \qquad = \displaystyle \sum _ { k = 0 } ^ { m } \alpha ^ { k } P _ { k } ^ { ( \le s ) } \otimes P _ { m - k } ^ { ( > s ) } . } \end{array}\tag{161}
$$

Here we have defined $\begin{array} { r } { \alpha : = ( \frac { \nu _ { 1 } } { \nu _ { 1 } + 1 } ) / ( \frac { \nu _ { 0 } } { \nu _ { 0 } + 1 } ) > 1 } \end{array}$ for notational simplicity. Crucially, the last line can be interpreted as $\Gamma _ { m } ( g _ { \alpha } )$ where

$$
\begin{array} { r } { g _ { \alpha } : = \mathrm { d i a g } ( \underbrace { \alpha , \cdots , \alpha } _ { s } , \underbrace { 1 , \cdots , 1 } _ { s } ) \in \mathrm { G L } _ { n } ( \mathbb { C } ) . } \end{array}\tag{162}
$$

Indeed, $\Gamma _ { m } ( g _ { \alpha } ) | m \rangle = \alpha ^ { | m \leq s | } | m \rangle$ for any $| { \boldsymbol { m } } | = m$ , which can also be seen via the tensor product action of $g _ { \alpha }$ on $( \mathbb { C } ^ { n } ) ^ { \otimes m }$ and then symmetrizing. The equivalence between this and the last line of Eq. (161) is obvious. Taking into account the normalization factor, we conclude that

$$
\rho _ { 0 , m } = \Gamma _ { m } ( g _ { \alpha } ) / \operatorname { T r } \left( \Gamma _ { m } ( g _ { \alpha } ) \right) .\tag{163}
$$

Now we compute the following Haar averages:

$$
\mathbb { E } \Gamma _ { m } ( U ) \rho _ { 0 , m } \Gamma _ { m } ( U ) ^ { \dagger } \overset { \mathrm { ( i ) } } { = } \frac { \mathrm { T r } ( \rho _ { 0 , m } P _ { m } ^ { ( n ) } ) } { \mathrm { T r } P _ { m } ^ { ( n ) } } P _ { m } ^ { ( n ) } = \frac { 1 } { \mathrm { T r } P _ { m } ^ { ( n ) } } P _ { m } ^ { ( n ) } .\tag{164}
$$

$$
\begin{array} { r l } { \frac { \mathbb { E } } { U } \Gamma _ { n } ( U ) \big ) ^ { \theta _ { n } } \rho _ { 0 , n ^ { \prime } } ^ { \frac { \gamma } { \alpha ^ { 2 } } } \Gamma _ { n } ( U ) \big \} ^ { 1 . 5 ; \alpha ^ { 2 } } } & { \Gamma _ { n } ^ { \theta _ { n } } \bigg [ \Gamma _ { 0 } ^ { \theta _ { n } } \Gamma _ { 0 } ^ { \theta _ { n } } \bigg ] ^ { 1 / \alpha } } \\ & { \quad = \frac { \mathbb { E } } { \Gamma _ { n - 1 } } \frac { \mathbb { E } } { \Gamma _ { 1 } \Gamma _ { 0 } ^ { \theta _ { n } } \Gamma _ { n - 1 } ^ { \theta _ { n } } } \bigg [ \Gamma _ { 0 } ^ { \theta _ { n } } \Gamma _ { n - 1 , \mathrm { e } } ^ { \theta _ { n } } \bigg ] } \\ & { \quad = \frac { \mathbb { E } } { \Gamma _ { n - 1 } } \frac { \mathbb { E } } { \Gamma _ { n } ^ { \theta _ { n } } \Gamma _ { n - 1 } ^ { \theta _ { n } } } \bigg [ \Gamma _ { 0 } ^ { \theta _ { n } } \Gamma _ { n - 1 , \mathrm { e } } ^ { \theta _ { n } } \bigg ] } \\ & { \quad \stackrel { \mathrm { G i v } } { = } \frac { \mathbb { E } } { \Gamma _ { n - 1 } } \frac { \mathbb { E } \mathbb { E } } { \Gamma _ { n } ^ { \theta _ { n } } \Gamma _ { n - 1 } ^ { \theta _ { n } } } \bigg [ \Gamma _ { 0 } ^ { \theta _ { n } } \Gamma _ { n - 1 , \mathrm { e } } ^ { \theta _ { n } } \bigg ] } \\ &  \quad \stackrel { \mathrm { G i v } } { \leq } ( \underset { 1 \leq n \leq n } { \operatorname* { m a x } } \frac { \mathbb { E } \Gamma _ { 1 } ^ { \theta _ { n } } \Gamma _ { 0 } ^ { \theta _ { n } } \Gamma _ { n - 1 , \mathrm { e } } ^ { \theta _ { n } } }  \mathbb { E } \Gamma _ { 0 } ^ { \theta _ { n } } \Gamma _  n - 1 , \mathrm  e  \end{array}\tag{165}
$$

Here (i) is by Schur’s Lemma; (ii) is by Schur’s Lemma and Pieri’s rule: $\mathrm { S y m } ^ { m } ( \mathbb { C } ^ { n } ) ^ { \otimes 2 } \cong \bigoplus _ { r = 0 } ^ { m } S _ { ( 2 m - r , r ) } ^ { ( n ) }$ as the irreducible representation decomposition of ${ \mathrm { G L } } _ { n } ( \mathbb { C } )$ ; (iii) uses Pieri’s rule again; (iv) replace all coeficients with a uniform upper bound; Finally, (v) uses Pieri’s rule again in the reversed direction. Note that the expression inside the bracket can be understood as normalized Schur polynomials, thanks to Lemma 15,

$$
\frac { \mathrm { T r } \Gamma _ { ( 2 m - r , r ) } ( g _ { \alpha } ) } { \mathrm { T r } P _ { ( 2 m - r , r ) } ^ { ( n ) } } = \frac { s _ { ( 2 m - r , r ) } ( \alpha , \cdots , \alpha , 1 , \cdots , 1 ) } { s _ { ( 2 m - r , r ) } ( 1 , \cdots , 1 , 1 , \cdots , 1 ) } = : \bar { s } _ { ( 2 m - r , r ) } ( \alpha , \cdots , \alpha , 1 , \cdots , 1 ) .\tag{166}
$$

The following result, conjectured in [CGS11] and proven in [Sra16], characterizes the monotonicity of normalized Schur polynomials:

Lemma 16 ([Sra16, CGS11]). Let λ and $\mu$ be partitions of m of size no more than n. For any $x _ { 1 } , \cdots , x _ { n } \geq$ $0 ,$

$$
{ \bar { s } } _ { \lambda } ( x _ { 1 } , \cdots , x _ { n } ) \leq { \bar { s } } _ { \mu } ( x _ { 1 } , \cdots , x _ { n } ) { \mathrm { i f ~ a n d ~ o n l y ~ i f ~ } } \lambda \preceq \mu .\tag{167}
$$

Here $\lambda \preceq \mu$ means $\lambda$ is majorized by $\mu .$ . That is, $\begin{array} { r } { \sum _ { j = 1 } ^ { k } \lambda _ { j } \le \sum _ { j = 1 } ^ { k } \mu _ { j } } \end{array}$ for all $1 \leq k \leq n$ . Note that we assume λ and $\mu$ are sorted in a non-increasing order.

This immediately yields that $r = 0$ maximize the last line of Eq. (165). Putting everything into the

expression of ⋆:

$$
\begin{array} { r l } & { \star + 1 \leq \displaystyle \sum _ { m = 0 } ^ { \infty } q _ { m } \int d z \frac { \langle z , m | z , m \rangle ^ { 2 } } { \langle z , m | z , m \rangle } \frac { \mathrm { T r } \mathrm { T } _ { 2 m } ^ { ( n ) } \langle g _ { 0 } \rangle } { \mathrm { T r } { P } _ { 2 m } ^ { ( n ) } \mathrm { T } ^ { 2 } \mathrm { T } _ { m } ( g _ { 0 } ) } } \\ & { \quad \quad \quad = \displaystyle \sum _ { m = 0 } ^ { \infty } q _ { m } \frac { \mathrm { T r } ^ { 2 } { P } _ { m } ^ { ( n ) } \mathrm { T r } \mathrm { I } _ { 2 m } ^ { ( n ) } \langle g _ { 0 } \rangle } { \mathrm { T r } ^ { 2 } \mathrm { T } _ { m } ^ { ( 2 ) } \mathrm { T r } _ { 0 } ^ { ( 2 ) } } } \\ & { \quad \quad \quad = \displaystyle \sum _ { m = 0 } ^ { \infty } q _ { m } \frac { \bar { x } _ { 0 } m ( \alpha , \cdots , \alpha , 1 , \cdots , 1 ) } { \bar { x } _ { m } ( \alpha , \cdots , \alpha , 1 , \cdots , 1 ) ^ { 2 } } \cdot } \\ & { \quad \quad \quad = \displaystyle \sum _ { m = 0 } ^ { \infty } q _ { m } \frac { \bar { x } _ { 0 } m ( \alpha _ { \mathrm { s } } ) } { \bar { x } _ { m } ( \alpha _ { \mathrm { s } } ) ^ { 2 } } , } \end{array}\tag{168}
$$

where we define $\pmb { \alpha } _ { s } : = ( \alpha , \cdot \cdot \cdot , \alpha , 1 , \cdot \cdot \cdot , 1 )$ for notational simplicity.

Now we derive a more explicit characterization of $\bar { s } _ { m } ( \alpha _ { s } )$

$$
\bar { s } _ { m } ( \alpha _ { s } ) = \frac { \mathrm { T r } \Gamma _ { m } ( g _ { \alpha } ) } { \mathrm { T r } P _ { m } ^ { ( n ) } } \overset { ( \mathrm { i } ) } { = } \sum _ { k = 0 } ^ { m } \alpha ^ { k } \frac { \binom { k + s - 1 } { k } \binom { m - k + s - 1 } { m - k } } { \binom { m + n - 1 } { m } } \overset { ( \mathrm { i i } ) } { = } \mathbb { E } [ \alpha ^ { K } ] .\tag{169}
$$

Here (i) uses $\begin{array} { r } { \Gamma _ { m } ( \alpha ) = \sum _ { k = 0 } ^ { m } \alpha ^ { k } P _ { k } ^ { ( \leq s ) } \otimes P _ { m - k } ^ { ( > s ) } } \end{array}$ and the dimension formula for symmetric subspaces; For (ii), we introduce K as a random variable with a Beta-binomial distribution (see e.g. [JKK05]), i.e.,

$$
K \sim \operatorname { B e t a B i n o m i a l } ( m , s , s ) .
$$

Equivalently, let $P \sim \mathrm { B e t a } ( s , s )$ , then $K | P \sim$ Binomial $( m , P )$ . Using the moment generating function of the binomial distribution, we obtain that

$$
\bar { s } _ { m } ( \pmb { \alpha } _ { s } ) = \mathbb { E } \mathbb { E } [ e ^ { K \log \alpha } | P ] = \mathbb { E } _ { P } ^ { } ( 1 + ( \alpha - 1 ) P ) ^ { m }\tag{170}
$$

Same for $\bar { s } _ { 2 m } ( \alpha _ { s } )$

Next, we derive some properties of $q _ { m }$ . View m as a random variable of distribution $q _ { m }$ . Since $\rho _ { 0 } =$ $\tau _ { \nu _ { 1 } } ^ { \otimes s } \otimes \tau _ { \nu _ { 0 } } ^ { \otimes s }$ , we can write $m = m _ { 1 } + m _ { 0 }$ where $m _ { 1 } \sim \mathrm { N B } ( \mathrm { s } , 1 / ( \nu _ { 1 } + 1 ) )$ and $m _ { 0 } \sim \mathrm { N B } ( \mathrm { s } , 1 / ( \nu _ { 0 } + 1 ) )$ are two independent negative binomial random variables. The first and second moment and moment generating function of m can then be derived as

$$
\begin{array} { r l } & { \underset { m \sim q } { \mathbb { E } } ^ { \lambda } m = ( \nu _ { 0 } + \nu _ { 1 } ) s . } \\ & { \underset { m \sim q } { \mathbb { E } } m ^ { 2 } = ( \nu _ { 0 } ^ { 2 } + \nu _ { 0 } + \nu _ { 1 } ^ { 2 } + \nu _ { 1 } ) s + ( \nu _ { 0 } + \nu _ { 1 } ) ^ { 2 } s ^ { 2 } = O ( 1 ) . } \\ & { \underset { m \sim q } { \mathbb { E } } e ^ { t m } = ( 1 - \nu _ { 0 } ( e ^ { t } - 1 ) ) ^ { - s } ( 1 - \nu _ { 1 } ( e ^ { t } - 1 ) ) ^ { - s } . } \end{array}\tag{171}
$$

We are now ready to bound $\star$ . We will divide and bound separately the sum over m into $m \leq { \sqrt { n } }$ and

m $> { \sqrt { n } } .$ For $m \leq { \sqrt { n } } ,$

$$
\begin{array} { r l } { \frac { \sqrt { \pi } \mu _ { 0 } ( { \bf { C } } _ { \alpha } ) } { \sqrt { \pi } \mu _ { 0 } ( { \bf { C } } _ { \alpha } ) ^ { 2 } } - \frac { \sqrt { \pi } \mu _ { 0 } ( { \bf { C } } _ { 1 } + ( { \bf { C } } _ { 0 } - 1 ) \rho ) ^ { 2 n } } { \left( \frac { \sqrt { \pi } \mu _ { 0 } ( { \bf { C } } _ { 1 } + ( { \bf { C } } _ { 1 } - 1 ) \rho ) ^ { 2 } } { 4 \pi } \right) ^ { 2 } } } & { } \\ { \frac { \sqrt { \pi } \mu _ { 0 } ( { \bf { C } } _ { 2 } + ( { \bf { C } } _ { 0 } - 1 ) \rho ) ^ { 2 n } } { ( ( { \bf { C } } _ { 1 } + 1 ) \rho ) ^ { 2 } } } & { } \\ { \frac { \sqrt { \pi } \mu _ { 0 } ( { \bf { C } } _ { 1 } + ( { \bf { C } } _ { 1 } - 1 ) \rho ) ^ { 2 n } } { ( ( { \bf { C } } _ { 2 } + 1 ) \rho ) ^ { 2 n } } } & { } \\ { = \frac { \sqrt { \pi } } { \mu _ { 0 } } \left( 1 + 2 \frac { \alpha - 1 } { \sqrt { \pi } } \left( P - \frac { 1 } { 2 } \right) \right) ^ { 2 n } } & { } \\ { \frac { \sqrt { \pi } \mu _ { 0 } ( { \bf { C } } _ { 1 } - 1 ) } { ( ( P - \frac { 1 } { 2 } + 1 ) \rho ) ^ { 2 } } } & { } \\ { \frac { \sqrt { \pi } \mu _ { 0 } ( { \bf { C } } _ { 2 } + ( { \bf { C } } _ { 0 } - 1 ) \rho ) } { ( ( { \bf { C } } _ { 1 } - 1 ) \rho ) ( { \bf { C } } _ { 1 } + 1 ) ^ { 2 } } } & { } \\  \frac  \sqrt { \pi } \mu _ { 0 } ( { \bf { C } } _ { 2 } + ( { \bf { C } } _ { 0 } - 1 ) \rho  \end{array}\tag{172}
$$

Here, (i) uses Jensen’s inequality and the fact that E $P = 1 / 2 ;$ (ii) uses the sub-gaussianity of Beta distribution as given in the following lemma:

Lemma 17. [MA17, Theorem 1, simplified] For any $s > 0 , P \sim \mathrm { B e t a } ( s , s )$ is sub-Gaussian with variance proxy $1 / ( 4 ( 2 s + 1 ) )$ ). That $i s ,$

$$
\mathbb { E } \exp ( \lambda ( P - \mathbb { E } P ) ) \leq \exp \left( \frac { \lambda ^ { 2 } } { 8 ( 2 s + 1 ) } \right) .\tag{173}
$$

(iii) is by substituting α with $\nu _ { 0 }$ and $\nu _ { 1 }$ , and then with n and $\xi .$ In particular, it is straightforward to verify that $( \alpha - 1 ) / ( \alpha + 1 ) \le \xi / 2$ . (iv) uses our assumption that $m \leq { \sqrt { n } }$ and $\xi \le 1$ and the fact that $\exp ( x ) \leq 1 +$ 2x for $x \in [ 0 , 1 / 2 ]$ . Therefore,

$$
\sum _ { m \leq \sqrt { n } } q _ { m } \left( \frac { \bar { s } _ { 2 m } ( \alpha _ { s } ) } { \bar { s } _ { m } ( \alpha _ { s } ) ^ { 2 } } - 1 \right) \leq \sum _ { m \leq \sqrt { n } } q _ { m } \frac { m ^ { 2 } \xi ^ { 2 } } { n } \leq \operatorname * { \mathbb { E } } _ { m \sim q } [ m ^ { 2 } ] \frac { \xi ^ { 2 } } { n } = O \left( \frac { \xi ^ { 2 } } { n } \right) .\tag{174}
$$

The last equality uses the second order moment of $m \sim q$ derived above.

For $m > { \sqrt { n } } .$ , we use a diferent upper bound:

$$
\begin{array} { r l } & { \frac { \overline { { s } } _ { 2 m } ( \alpha _ { s } ) } { \overline { { s } } _ { m } ( \alpha _ { s } ) ^ { 2 } } - 1 = \frac { \mathrm { V a r } _ { P } ( 1 + ( \alpha - 1 ) P ) ^ { m } } { \left( \mathbb { E } _ { P } \left( 1 + ( \alpha - 1 ) P \right) ^ { m } \right) ^ { 2 } } } \\ & { \qquad \stackrel { \mathrm { ( i ) } } { \leq } \frac { ( \alpha ^ { m } - 1 ) ^ { 2 } / 4 } { ( ( \alpha + 1 ) / 2 ) ^ { 2 m } } } \\ & { \qquad \stackrel { \mathrm { ( i i ) } } { \leq } m ^ { 2 } ( \alpha - 1 ) ^ { 2 } 4 ^ { m } } \\ & { \qquad \stackrel { \mathrm { ( i i i ) } } { \leq } m ^ { 2 } \xi ^ { 2 } 4 ^ { m } . } \end{array}\tag{175}
$$

For (i), the denominator uses Jensen’s inequality, while the numerator uses the fact that $\mathrm { V a r } [ X ] \le$ (max X−min $X ) ^ { 2 } / 4$ for any bounded random variable X. (ii) uses the inequality $\alpha ^ { m } - 1 \le m ( \alpha - 1 ) \alpha ^ { m - 1 }$

which can be seen by diferentiating $x ^ { m }$ . (iii) uses that $\alpha - 1 \leq \xi$ that can be directly verified. Therefore,

$$
\begin{array} { r l } { \displaystyle \sum _ { m > \sqrt { n } } q _ { m } ( \frac { \tilde { \sigma } _ { 2 m } ( \alpha _ { s } ) } { \tilde { s } _ { m } ( \alpha _ { s } ) ^ { 2 } } - 1 ) \leq \displaystyle \sum _ { m > \sqrt { n } } q _ { m } m ^ { 2 } \xi ^ { 2 } 4 ^ { m } = \displaystyle \operatorname* { l } _ { m < \ell } \mathbb { I } [ \mathfrak { z } _ { m > \sqrt { n } } ] m ^ { 2 } \xi ^ { 2 } 4 ^ { m } ] } & { \medskip } \\ & { \overset { ( ) } { \leq } \displaystyle \operatorname* { m e x } _ { m < \ell } \mathbb { E } [ e ^ { m - \sqrt { n } } m ^ { 2 } \xi ^ { 2 } 4 ^ { m } ] } \\ & { \leq \xi ^ { 2 } \epsilon ^ { - \sqrt { n } } \displaystyle \operatorname* { l } _ { m < \ell } [ e ^ { ( 2 + \mathrm { i } \mathfrak { s } q _ { 4 } ) m } ] } \\ & { \overset { ( ) } { = } \xi ^ { 2 } \epsilon ^ { - \sqrt { n } } \displaystyle ( 1 - \nu _ { 1 } ( 4 \epsilon ^ { 2 } - 1 ) ) ^ { - s } ( 1 - \nu _ { 0 } ( 4 \epsilon ^ { 2 } - 1 ) ) ^ { - s } } \\ & { \leq \xi ^ { 2 } \epsilon ^ { - \sqrt { n } } \exp ( - n \log ( 1 - \frac { \xi + 1 } { n } ( 4 \epsilon ^ { 2 } - 1 ) ) ) } \\ & { \overset { ( ) } { = } \mathcal { O } ( \xi ^ { 2 } \epsilon ^ { - \sqrt { n } } ) . } \end{array}\tag{176}
$$

Here (i) uses $\mathbb { 1 } _ { [ m > \sqrt { n } ] } \leq e ^ { m - \sqrt { n } } ;$ ; (ii) uses the moment generating function of $m \sim q$ derived above; (iii) follows from $\xi \le 1$ and the bound log $\left( 1 - x \right) \geq - C x$ for $0 \leq x \leq x _ { 0 } < 1$ , where $C$ is a constant that only depends on $x _ { 0 }$ . This holds for suficiently large n.

Combing the two parts, we conclude that

$$
\star = O \left( \frac { \xi ^ { 2 } } { n } \right) .\tag{177}
$$

Together with the lower bound of Eq. (159), we obtain that the necessary number of samples must satisfy

$$
N = \Omega \left( \frac { n ^ { 3 } } { \xi ^ { 2 } } \right) = \Omega \left( \frac { n ^ { 3 } } { \varepsilon ^ { 2 } } \right) ,\tag{178}
$$

where the last equality uses our choice $\xi = \Theta ( \varepsilon )$ . This completes the proof for Theorem $6 .$ □

## 7 Upper bound on learning warm Gaussian states

In this section, we prove the upper bound in Theorem 3. We use the quadrature ordering $\hat { r } \textrm { = }$ $( \widehat { q } _ { 1 } , \ldots , \widehat { q } _ { n } , \widehat { p } _ { 1 } , \ldots , \widehat { p } _ { n } )$ , the symplectic form $\Omega \ = \ \left( \begin{array} { c c } { { 0 } } & { { \mathbb I _ { n } } } \\ { { - \mathbb I _ { n } } } & { { 0 } } \end{array} \right)$ , and the covariance convention so that the vacuum covariance matrix is $\mathbb { I } / 2$ as also used in previous sections.

The proof is based on a simple property of the estimator used below. After slightly inflating the empirical heterodyne covariance, the estimated covariance $\widehat { \Sigma }$ satisfies $\widehat { \Sigma } \succeq \Sigma$ on the concentration event. Thus, the estimation error only adds noise to the state. We first show that such one-sided covariance perturbations lead to a particularly stable trace-distance bound. We then apply this bound to the empirical mean and covariance obtained from heterodyne measurements.

The proof strategy based on the resolvent-based covariance-to-Hamiltonian analysis and the quantum Pinsker inequality is first explored in [FRSF26] for bosonic Gaussian states and Hamiltonian learning. Our Theorem 7 improves upon [FRSF26, Theorem 4.1] for one-sided covariance perturbation, thanks to a tighter analysis on the relative Frobenius norm. This improvement is necessary to obtain our optimal temperature-dependent upper bound. Our analysis of the heterodyne estimators builds upon [BMEM25].

## 7.1 A one-sided relative covariance trace-distance bound

A centered Gaussian state whose symplectic eigenvalues are strictly larger than $1 / 2$ is faithful $( \mathrm { i . e . }$ , it has trivial kernel) and can be written as a Gibbs state of a positive quadratic Hamiltonian. More precisely,

$$
\rho ( 0 , \Sigma ) = \frac { \exp \Bigl [ - \frac { 1 } { 2 } \hat { r } ^ { T } H ( \Sigma ) \hat { r } \Bigr ] } { \mathrm { T r } \exp \Bigl [ - \frac { 1 } { 2 } \hat { r } ^ { T } H ( \Sigma ) \hat { r } \Bigr ] } .\tag{179}
$$

To express the Hamiltonian matrix $H ( \Sigma )$ in terms of the covariance matrix, set $J : = i \Omega$ . Since $\Omega ^ { T } = - \Omega$ and $\Omega ^ { 2 } = - \mathbb { I }$ , the matrix J is Hermitian and unitary. In our covariance convention, Eqs. (5)–(6) of Ref. [BBP15] give

$$
H ( \Sigma ) = J f ( \Sigma J ) , \qquad f ( z ) : = 2 \operatorname { a r c c o t h } ( 2 z ) = \log \left( { \frac { z + { \frac { 1 } { 2 } } } { z - { \frac { 1 } { 2 } } } } \right) .\tag{180}
$$

The form of $f$ that will be useful below is its resolvent representation. For every $z \not \in [ - 1 / 2 , 1 / 2 ]$ ， direct integration gives

$$
f ( z ) = \int _ { - 1 / 2 } ^ { 1 / 2 } \frac { d t } { z - t } .\tag{181}
$$

Indeed, $\begin{array} { r } { \int _ { - 1 / 2 } ^ { 1 / 2 } ( z - t ) ^ { - 1 } d t = [ - \log ( z - t ) ] _ { - 1 / 2 } ^ { 1 / 2 } = \log ( ( z + 1 / 2 ) / ( z - 1 / 2 ) ) . } \end{array}$

We can apply the same identity to the matrix ΣJ. To see this explicitly, observe first that ΣJ is similar to the Hermitian matrix $\mathrm { \dot { \Sigma } } ^ { 1 / 2 } J \Sigma ^ { 1 / 2 }$ : indeed, $\Sigma J = \Sigma ^ { 1 / 2 } ( \Sigma ^ { \bar { 1 / 2 } } J \Sigma ^ { \bar { 1 / 2 } } ) \Sigma ^ { - 1 / 2 }$ . Hence ΣJ is diagonalizable and has real spectrum. Moreover, if $\Sigma = S D S ^ { T }$ is a Williamson decomposition $[ \mathrm { W P G P ^ { + } } 1 2 $ Eq. (44)], with $D = \operatorname { d i a g } ( \nu _ { 1 } , \dots , \nu _ { n } , \nu _ { 1 } , \dots , \nu _ { n } )$ in our quadrature ordering, then symplecticity of S gives $\Sigma J = S ( D J ) S ^ { - 1 }$ . Since D commutes with J and J has eigenvalues ±1, the eigenvalues of ΣJ are precisely $\{ \pm \nu _ { j } \} _ { j = 1 } ^ { n }$

For a faithful Gaussian state, every $\nu _ { j } > 1 / 2 ,$ , so the spectrum of $\Sigma J$ is disjoint from $[ - 1 / 2 , 1 / 2 ]$ Consequently, $\Sigma J -$ tI is invertible for every $t \in [ - 1 / 2 , 1 / 2 ]$ . Diagonalizing ΣJ and applying Eq. (181)

to each eigenvalue therefore gives

$$
f ( \Sigma J ) = \int _ { - 1 / 2 } ^ { 1 / 2 } ( \Sigma J - t \mathbb { I } ) ^ { - 1 } d t .\tag{182}
$$

This resolvent representation is the starting point for the stability estimate below.

We now state the trace-distance bound that will be used in the tomography proof.

Theorem 7 (One-sided relative covariance bound). Let Σ<sup>¯</sup> be an n-mode covariance matrix whose symplectic eigenvalues are all at least $\nu _ { - } > 1 / 2 ,$ and let $\Sigma ^ { \prime }$ be another covariance matrix satisfying $\Sigma ^ { \prime } \succeq \bar { \Sigma } .$ . Define

$$
\gamma _ { - } : = 1 - \frac { 1 } { 2 \nu _ { - } } , \qquad \Delta _ { \mathrm { r e l } } : = \Bigl \| \bar { \Sigma } ^ { - 1 / 2 } ( \Sigma ^ { \prime } - \bar { \Sigma } ) \bar { \Sigma } ^ { - 1 / 2 } \Bigr \| _ { \mathrm { F } } .\tag{183}
$$

Then

$$
\frac { 1 } { 2 } \left\| \rho ( 0 , \Sigma ^ { \prime } ) - \rho ( 0 , \bar { \Sigma } ) \right\| _ { 1 } \leq \frac { 1 } { 2 \sqrt { 2 \gamma _ { - } } } \Delta _ { \mathrm { r e l } } .\tag{184}
$$

Proof. By Williamson’s theorem, there is a real symplectic matrix S such that

$$
\bar { \Sigma } = S D S ^ { T } , \qquad D : = \mathrm { d i a g } ( \bar { \nu } _ { 1 } , \dots , \bar { \nu } _ { n } , \bar { \nu } _ { 1 } , \dots , \bar { \nu } _ { n } ) , \qquad \bar { \nu } _ { j } \geq \nu _ { - } .\tag{185}
$$

We also have that D commutes with $J = i \Omega$

We express $\Sigma ^ { \prime }$ in the same coordinates and measure its error relative to D:

$$
\Sigma _ { 0 } : = S ^ { - 1 } \Sigma ^ { \prime } S ^ { - T } ,
$$

$$
\begin{array} { r } { E : = D ^ { - 1 / 2 } ( \Sigma _ { 0 } - D ) D ^ { - 1 / 2 } . } \end{array}\tag{186}
$$

The order relation $\Sigma ^ { \prime } \succeq \bar { \Sigma }$ is preserved under congruence by an invertible matrix. Therefore $\Sigma _ { 0 } \succeq D$ and hence $E \succeq 0$ . Equivalently,

$$
\Sigma _ { 0 } = D ^ { 1 / 2 } ( \mathbb { I } + E ) D ^ { 1 / 2 } , \qquad E \succeq 0 .\tag{187}
$$

This positivity is the main reason for introducing the one-sided estimator.

The matrix E has exactly the same Frobenius norm as the relative error in the statement of the theorem. To see this, define ${ \cal O } : = \bar { \Sigma } ^ { - 1 / 2 } S D ^ { 1 / 2 }$ . Since $\bar { \Sigma } = S D S ^ { T }$ , we have $O O ^ { T } = \mathbb { I } ,$ so O is orthogonal. Moreover,

$$
\bar { \Sigma } ^ { - 1 / 2 } ( \Sigma ^ { \prime } - \bar { \Sigma } ) \bar { \Sigma } ^ { - 1 / 2 } = O E O ^ { T } .\tag{188}
$$

The Frobenius norm is invariant under orthogonal conjugation, and consequently

$$
\Delta _ { \mathrm { r e l } } = \| E \| _ { \mathrm { F } } .\tag{189}
$$

Notice also that $\Sigma _ { 0 } \succeq D \succeq \nu _ { - }$ I. We now show that this ordinary matrix lower bound implies the same lower bound on the symplectic eigenvalues of $\Sigma _ { 0 }$

Let $M > 0$ have symplectic eigenvalues $\nu _ { 1 } ( M ) , \ldots , \nu _ { n } ( M )$ . By Williamson’s theorem, the eigenvalues of ΩM are $\{ \pm i \nu _ { j } ( M ) \} _ { j = 1 } ^ { n }$ . Since $M ^ { 1 / 2 } \Omega M ^ { 1 / 2 }$ is similar to ΩM, it has the same eigenvalues. Moreover, $M ^ { 1 / 2 } \Omega M ^ { 1 / 2 }$ is real antisymmetric and therefore normal, so its singular values are the absolute values of its eigenvalues. Hence

$$
\nu _ { \mathrm { m i n } } ( M ) = s _ { \mathrm { m i n } } \Big ( M ^ { 1 / 2 } \Omega M ^ { 1 / 2 } \Big ) .
$$

Using the standard inequality $s _ { \mathrm { m i n } } ( A B ) \geq s _ { \mathrm { m i n } } ( A ) s _ { \mathrm { m i n } } ( B )$ twice, we obtain

$$
s _ { \mathrm { m i n } } \Big ( M ^ { 1 / 2 } \Omega M ^ { 1 / 2 } \Big ) \geq s _ { \mathrm { m i n } } ( M ^ { 1 / 2 } ) ^ { 2 } s _ { \mathrm { m i n } } ( \Omega ) .
$$

If $M \succeq m \mathbb { I } .$ , then $s _ { \operatorname* { m i n } } ( M ^ { 1 / 2 } ) = \sqrt { \lambda _ { \operatorname* { m i n } } ( M ) } \geq \sqrt { m }$ , while $\Omega ^ { T } \Omega = \mathbb { I }$ implies $s _ { \mathrm { m i n } } ( \Omega ) = 1$ . Therefore

$$
\nu _ { \operatorname* { m i n } } ( M ) \geq m .
$$

Applying this to $M = \Sigma _ { 0 }$ and $m = \nu _ { - }$ shows that every symplectic eigenvalue of $\Sigma _ { 0 }$ is at least $\nu _ { - }$ Since $\bar { \Sigma _ { 0 } } = \bar { S } ^ { - 1 } \Sigma ^ { \prime } S ^ { - T }$ is related to $\Sigma ^ { \prime }$ by a symplectic congruence, and symplectic eigenvalues are invariant under such congruences, the same lower bound holds for $\Sigma ^ { \prime } .$

Thus both $\bar { \Sigma }$ and $\Sigma ^ { \prime }$ have symplectic eigenvalues strictly larger than $1 / 2$ . The corresponding Gaussian states are therefore faithful, and the Hamiltonian formula in Eq. (180) is well defined for both.

We now compare the two Hamiltonian matrices. Set

$$
A : = \Sigma _ { 0 } J , B : = D J .\tag{190}
$$

For $t \in [ - 1 / 2 , 1 / 2 ]$ , the commutation relation $D J = J D$ gives

$$
\begin{array} { r l } & { D ^ { 1 / 2 } ( B - t \mathbb { I } ) ^ { - 1 } D ^ { 1 / 2 } = ( J - t D ^ { - 1 } ) ^ { - 1 } , } \\ & { D ^ { 1 / 2 } ( A - t \mathbb { I } ) ^ { - 1 } D ^ { 1 / 2 } = \big ( ( \mathbb { I } + E ) J - t D ^ { - 1 } \big ) ^ { - 1 } . } \end{array}\tag{191}
$$

Define $a ( t ) : = 1 - | t | / \nu _ { - }$ . Since $| t | \leq 1 / 2$ and $\nu _ { - } > 1 / 2$ , we have $a ( t ) > 0$ . To bound the first resolvent in Eq. (191), we use the fact that J is unitary. Right multiplication by J therefore leaves singular values unchanged, while it turns the denominator into a Hermitian matrix:

$$
( J - t D ^ { - 1 } ) J =  { \mathbb { I } } - t D ^ { - 1 } J .
$$

Since $D ^ { - 1 }$ commutes with $J ,$ the matrix $D ^ { - 1 } { \mathcal { J } }$ is Hermitian. Moreover, on the two-dimensional subspace corresponding to the $( q _ { j } , p _ { j } )$ coordinates,

$$
D ^ { - 1 } J = \frac { 1 } { \bar { \nu } _ { j } } \left( { 0 \atop - i } \ { 0 \atop 0 } \right) ,
$$

so its eigenvalues are $\pm 1 / \bar { \nu } _ { j }$ . It follows that the eigenvalues of $\mathbb { I } - t D ^ { - 1 } J$ are $1 \mp t / \bar { \nu } _ { j }$ . Since $\bar { \nu } _ { j } \geq \nu _ { - }$ and $t \in [ - 1 / 2 , 1 / 2 ]$ , all of them are bounded below by

$$
a ( t ) : = 1 - \frac { | t | } { \nu _ { - } } > 0 .
$$

Hence $\mathbb { I } - t D ^ { - 1 } J \succeq a ( t ) \mathbb { I }$ . Equivalently, the smallest singular value of $J - t D ^ { - 1 }$ is at least $a ( t )$ , and therefore

$$
\left. D ^ { 1 / 2 } ( B - t \mathbb { I } ) ^ { - 1 } D ^ { 1 / 2 } \right. _ { \mathrm { o p } } = \left. ( J - t D ^ { - 1 } ) ^ { - 1 } \right. _ { \mathrm { o p } } \leq \frac { 1 } { a ( t ) } .
$$

The same argument gives the corresponding bound for the perturbed resolvent. Indeed,

$$
\begin{array} { r } { \big ( ( \mathtt { I } + E ) J - t D ^ { - 1 } \big ) J = \mathtt { I } + E - t D ^ { - 1 } J . } \end{array}
$$

The matrix on the right is Hermitian, and the one-sided assumption enters precisely here: since $E \succeq 0$

$$
\mathbb { I } + E - t D ^ { - 1 } J \succeq \mathbb { I } - t D ^ { - 1 } J \succeq a ( t ) \mathbb { I } .
$$

Thus the positive relative perturbation $E$ can only increase the relevant spectral gap. Since multiplication by J again preserves singular values, we conclude that

$$
\left\| D ^ { 1 / 2 } ( A - t \mathbb { I } ) ^ { - 1 } D ^ { 1 / 2 } \right\| _ { \mathrm { o p } } , \quad \left\| D ^ { 1 / 2 } ( B - t \mathbb { I } ) ^ { - 1 } D ^ { 1 / 2 } \right\| _ { \mathrm { o p } } \leq \frac { 1 } { a ( t ) } .\tag{192}
$$

This is the only place in the proof where the order relation $\Sigma ^ { \prime } \succeq \bar { \Sigma }$ is needed. In Williamson coordinates, it becomes $E \succeq 0$ , which ensures that the perturbed resolvent is no closer to the singular interval $[ - 1 / 2 , 1 / 2 ]$ than the unperturbed one.

Using the matrix resolvent representation

$$
f ( X ) = \int _ { - 1 / 2 } ^ { 1 / 2 } ( X - t \mathbb { I } ) ^ { - 1 } d t ,
$$

which applies to both A and $B ,$ we have

$$
f ( A ) - f ( B ) = \int _ { - 1 / 2 } ^ { 1 / 2 } \Bigl [ ( A - t \mathbb { I } ) ^ { - 1 } - ( B - t \mathbb { I } ) ^ { - 1 } \Bigr ] d t .
$$

We now use the elementary resolvent identity $X ^ { - 1 } - Y ^ { - 1 } = X ^ { - 1 } ( Y - X ) Y ^ { - 1 }$ with $X = A - t \mathbb { I }$ and $Y = B - t \mathbb { I }$ . This gives

$$
( A - t \mathbb { I } ) ^ { - 1 } - ( B - t \mathbb { I } ) ^ { - 1 } = ( A - t \mathbb { I } ) ^ { - 1 } ( B - A ) ( B - t \mathbb { I } ) ^ { - 1 } ,
$$

and therefore

$$
f ( A ) - f ( B ) = \int _ { - 1 / 2 } ^ { 1 / 2 } ( A - t \mathbb { I } ) ^ { - 1 } ( B - A ) ( B - t \mathbb { I } ) ^ { - 1 } d t .\tag{193}
$$

We next express the perturbation $B - A$ in terms of the relative covariance error $E .$ . Since $\Sigma _ { 0 } =$ $D ^ { 1 / 2 } ( \mathbb { I } + E ) D ^ { \bar { 1 } / 2 }$ and $D$ commutes with J,

$$
B - A = ( D - \Sigma _ { 0 } ) J = - D ^ { 1 / 2 } E J D ^ { 1 / 2 } .
$$

Substituting this into the integrand of Eq. (193) and inserting the weights $D ^ { 1 / 2 }$ on both sides gives

$$
\begin{array} { r l } & { \boldsymbol { D } ^ { 1 / 2 } ( \boldsymbol { A } - t \mathbb { I } ) ^ { - 1 } ( \boldsymbol { B } - \boldsymbol { A } ) ( \boldsymbol { B } - t \mathbb { I } ) ^ { - 1 } \boldsymbol { D } ^ { 1 / 2 } } \\ & { \qquad = - \Big [ \boldsymbol { D } ^ { 1 / 2 } ( \boldsymbol { A } - t \mathbb { I } ) ^ { - 1 } \boldsymbol { D } ^ { 1 / 2 } \Big ] \boldsymbol { E } \boldsymbol { J } \Big [ \boldsymbol { D } ^ { 1 / 2 } ( \boldsymbol { B } - t \mathbb { I } ) ^ { - 1 } \boldsymbol { D } ^ { 1 / 2 } \Big ] . } \end{array}\tag{194}
$$

We can now apply $\| X Y Z \| _ { \mathrm { F } } ~ \le ~ \| X \| _ { \mathrm { o p } } \| Y \| _ { \mathrm { F } } \| Z \| _ { \mathrm { o p } }$ . Since J is unitary, $\| E J \| _ { \mathrm { F } } = \| E \| _ { \mathrm { F } }$ , while Eq. (192) bounds each weighted resolvent by $1 / a ( t )$ , with $a ( t ) = 1 - | t | / \nu .$ <sub>−</sub>. Hence, for every $t \in$ $[ - 1 / 2 , 1 / 2 ]$ 2

$$
\begin{array} { r l } & { \left\| D ^ { 1 / 2 } ( A - t \mathbb { I } ) ^ { - 1 } ( B - A ) ( B - t \mathbb { I } ) ^ { - 1 } D ^ { 1 / 2 } \right\| _ { \mathrm { F } } } \\ & { \qquad \leq \frac { \| E \| _ { \mathrm { F } } } { a ( t ) ^ { 2 } } = \frac { \| E \| _ { \mathrm { F } } } { ( 1 - | t | / \nu _ { - } ) ^ { 2 } } . } \end{array}
$$

Taking the Frobenius norm in Eq. (193) and using the triangle inequality for the integral, we obtain

$$
\begin{array} { r l } & { \left\| D ^ { 1 / 2 } \big ( f ( A ) - f ( B ) \big ) D ^ { 1 / 2 } \right\| _ { \mathrm { F } } \leq \| E \| _ { \mathrm { F } } \int _ { - 1 / 2 } ^ { 1 / 2 } \frac { d t } { \big ( 1 - | t | / \nu _ { - } \big ) ^ { 2 } } } \\ & { \qquad = \frac { 1 } { \gamma _ { - } } \| E \| _ { \mathrm { F } } . } \end{array}\tag{195}
$$

To evaluate the integral, we use symmetry around $t = 0$

$$
\begin{array} { r l r } {  { \int _ { - 1 / 2 } ^ { 1 / 2 } \frac { d t } { ( 1 - | t | / \nu _ { - } ) ^ { 2 } } = 2 \int _ { 0 } ^ { 1 / 2 } \frac { d t } { ( 1 - t / \nu _ { - } ) ^ { 2 } } } } \\ & { } & { = 2 \nu _ { - } ( \frac { 1 } { \gamma _ { - } } - 1 ) = \frac { 1 } { \gamma _ { - } } , } \end{array}
$$

where $\gamma _ { - } = 1 - ( 2 \nu _ { - } ) ^ { - 1 }$

The exact t-dependence is important here. Replacing $a ( t )$ by its minimum value $\gamma _ { - }$ before integrating would give the weaker factor $1 / \gamma _ { - } ^ { 2 }$ . Keeping the full resolvent gap throughout the integral instead gives the sharper $1 / \gamma _ { - }$ dependence.

We now translate the matrix-function estimate Eq. (195) into a bound on the corresponding quadratic Hamiltonians. Recall from Eq. (180) that

$$
H ( \Sigma _ { 0 } ) = J f ( A ) , \qquad H ( D ) = J f ( B ) ,
$$

because $A = \Sigma _ { 0 } J$ and $B = D J$ . Therefore

$$
\begin{array} { r } { D ^ { 1 / 2 } \big ( H ( \Sigma _ { 0 } ) - H ( D ) \big ) D ^ { 1 / 2 } = D ^ { 1 / 2 } J \big ( f ( A ) - f ( B ) \big ) D ^ { 1 / 2 } } \\ { = J D ^ { 1 / 2 } \big ( f ( A ) - f ( B ) \big ) D ^ { 1 / 2 } , } \end{array}
$$

where in the second equality we used that D commutes with $J ,$ and hence so does $D ^ { 1 / 2 }$ . Since J is unitary, left multiplication by $J$ does not change the Frobenius norm. Thus, using Eq. (195) and $\| E \| _ { \mathrm { F } } = \Delta _ { \mathrm { r e l } }$ , we obtain

$$
\left. D ^ { 1 / 2 } ( H ( \Sigma _ { 0 } ) - H ( D ) ) D ^ { 1 / 2 } \right. _ { \mathrm { F } } \leq \frac { 1 } { \gamma _ { - } } \Delta _ { \mathrm { r e l } } .\tag{196}
$$

We next return from Williamson coordinates to the original covariance matrices $\Sigma ^ { \prime }$ and $\Sigma .$ The Hamiltonian matrix transforms covariantly under the same symplectic change of coordinates. In particular, since $\Sigma _ { 0 } = S ^ { - 1 } \Sigma ^ { \prime } S ^ { - T }$ , we claim that

$$
H ( \Sigma _ { 0 } ) = S ^ { T } H ( \Sigma ^ { \prime } ) S .
$$

To see this, recall that S is symplectic, so $S ^ { T } J S = J $ . Equivalently, $S ^ { T } J = J S ^ { - 1 }$ and $S ^ { - T } J = J S$ Hence

$$
\Sigma _ { 0 } J = S ^ { - 1 } \Sigma ^ { \prime } S ^ { - T } J = S ^ { - 1 } ( \Sigma ^ { \prime } J ) S .
$$

Thus $\Sigma _ { 0 } J$ is similar to $\Sigma ^ { \prime } J$ . The same similarity relation is inherited by the matrix function $f .$ . Indeed, for every $t \in [ - 1 / 2 , 1 / 2 ]$ 2

$$
( \Sigma _ { 0 } J - t \mathbb { I } ) ^ { - 1 } = S ^ { - 1 } ( \Sigma ^ { \prime } J - t \mathbb { I } ) ^ { - 1 } S ,
$$

and integrating the resolvent representation of $f$ gives $f ( \Sigma _ { 0 } J ) = S ^ { - 1 } f ( \Sigma ^ { \prime } J ) S .$

Using the covariance-to-Hamiltonian formula and $J S ^ { - 1 } = S ^ { T } J _ { : }$ , we therefore obtain

$$
H ( \Sigma _ { 0 } ) = J f ( \Sigma _ { 0 } J ) = J S ^ { - 1 } f ( \Sigma ^ { \prime } J ) S = S ^ { T } J f ( \Sigma ^ { \prime } J ) S = S ^ { T } H ( \Sigma ^ { \prime } ) S .
$$

Applying exactly the same argument to $D = S ^ { - 1 } \bar { \Sigma } S ^ { - T }$ gives $H ( D ) = S ^ { T } H ( \bar { \Sigma } ) S$

Recall now the orthogonal matrix $O = \bar { \Sigma } ^ { - 1 / 2 } S D ^ { 1 / 2 }$ introduced above. From its definition, $O ^ { T } \bar { \Sigma } ^ { 1 / 2 } =$ $D ^ { 1 / 2 } S ^ { T }$ and $\bar { \Sigma } ^ { 1 / 2 } O = \bar { S } D ^ { 1 / 2 }$ . Therefore

$$
\begin{array} { r l } & { O ^ { T } \bar { \Sigma } ^ { 1 / 2 } \bigl ( H \bigl ( \Sigma ^ { \prime } \bigr ) - H ( \bar { \Sigma } ) \bigr ) \bar { \Sigma } ^ { 1 / 2 } O } \\ & { \quad \quad = D ^ { 1 / 2 } S ^ { T } \bigl ( H \bigl ( \Sigma ^ { \prime } \bigr ) - H ( \bar { \Sigma } ) \bigr ) S D ^ { 1 / 2 } } \\ & { \quad \quad = D ^ { 1 / 2 } \bigl ( H \bigl ( \Sigma _ { 0 } \bigr ) - H ( D ) \bigr ) D ^ { 1 / 2 } . } \end{array}\tag{197}
$$

Since O is orthogonal, conjugation by $O$ preserves the Frobenius norm. Combining Eqs. (197) and (196) therefore yields

$$
\left\| \bar { \Sigma } ^ { 1 / 2 } \big ( H ( \Sigma ^ { \prime } ) - H ( \bar { \Sigma } ) \big ) \bar { \Sigma } ^ { 1 / 2 } \right\| _ { \mathrm { F } } \leq \frac { 1 } { \gamma _ { - } } \Delta _ { \mathrm { r e l } } .\tag{198}
$$

It remains to convert this Hamiltonian estimate into a trace-distance estimate. For brevity, write $H ^ { \prime } : = H ( \Sigma ^ { \prime } )$ and $\bar { H } : = H ( \bar { \Sigma } )$ . For any Hamiltonian matrix $H _ { \ast }$ define

$$
K _ { H } : = \frac { 1 } { 2 } \hat { \pmb { r } } ^ { T } H \hat { \pmb { r } } , \qquad Z _ { H } : = \mathrm { T r } ( e ^ { - K _ { H } } ) , \qquad \rho _ { H } : = \frac { e ^ { - K _ { H } } } { Z _ { H } } .
$$

Thus $K _ { H }$ is the quadratic Hamiltonian operator and $Z _ { H }$ is its partition function. In particular, log $\rho _ { H } =$ $- K _ { H } - \log Z _ { H }$

Using the definition $D ( \rho | | \sigma ) = \mathrm { T r } [ \rho ( \log \rho - \log \sigma ) ]$ , we obtain

$$
\begin{array} { l } { { { \cal D } \Big ( \rho ( 0 , \Sigma ^ { \prime } ) \Big \| \rho ( 0 , \bar { \Sigma } ) \Big ) = \mathrm { T r } \big [ \rho ( 0 , \Sigma ^ { \prime } ) ( K _ { \bar { H } } - K _ { H ^ { \prime } } ) \big ] + \log Z _ { \bar { H } } - \log Z _ { H ^ { \prime } } , } } \\ { { { \cal D } \Big ( \rho ( 0 , \bar { \Sigma } ) \Big \| \rho ( 0 , \Sigma ^ { \prime } ) \Big ) = \mathrm { T r } \Big [ \rho ( 0 , \bar { \Sigma } ) ( K _ { H ^ { \prime } } - K _ { \bar { H } } ) \Big ] + \log Z _ { H ^ { \prime } } - \log Z _ { \bar { H } } . } } \end{array}
$$

Adding the two expressions cancels the partition-function terms, leaving

$$
\begin{array} { r l } & { D \Big ( \rho ( 0 , \Sigma ^ { \prime } ) \Big | \Big | \rho ( 0 , \bar { \Sigma } ) \Big ) + D \Big ( \rho ( 0 , \bar { \Sigma } ) \Big | \Big | \rho ( 0 , \Sigma ^ { \prime } ) \Big ) } \\ & { \qquad = \mathrm { T r } \Big [ \big ( \rho ( 0 , \Sigma ^ { \prime } ) - \rho ( 0 , \bar { \Sigma } ) \big ) ( K _ { \bar { H } } - K _ { H ^ { \prime } } ) \Big ] . } \end{array}\tag{199}
$$

We now evaluate the expectation of a quadratic Hamiltonian. For every real symmetric matrix G and every centered state with covariance matrix Σ,

$$
\mathrm { T r } \Big [ \rho ( 0 , \Sigma ) \frac { 1 } { 2 } \widehat { r } ^ { T } G \widehat { r } \Big ] = \frac { 1 } { 2 } \mathrm { T r } ( G \Sigma ) .
$$

Indeed, because G is symmetric,

$$
\widehat { r } ^ { T } G \widehat { r } = \frac { 1 } { 2 } \sum _ { j , k } G _ { j k } \{ \widehat { r } _ { j } , \widehat { r } _ { k } \} ,
$$

since the antisymmetric commutator contribution cancels. The identity then follows directly from the definition of the covariance matrix.

Applying this identity to Eq. (199) gives the exact symmetric relative-entropy identity

$$
\begin{array} { r l } & { D \Big ( \rho ( 0 , \Sigma ^ { \prime } ) \Big | \Big | \rho ( 0 , \bar { \Sigma } ) \Big ) + D \Big ( \rho ( 0 , \bar { \Sigma } ) \Big | \Big | \rho ( 0 , \Sigma ^ { \prime } ) \Big ) } \\ & { \qquad = \frac { 1 } { 2 } \operatorname { T r } \Big [ ( \bar { H } - H ^ { \prime } ) ( \Sigma ^ { \prime } - \bar { \Sigma } ) \Big ] . } \end{array}\tag{200}
$$

The left-hand side is nonnegative, so the trace on the right-hand side is nonnegative as well. We may therefore upper bound it by its absolute value. Inserting the relative weights $\bar { \Sigma } ^ { 1 / \bar { 2 } }$ and $\bar { \Sigma } ^ { - 1 / 2 }$ and using cyclicity of the trace gives

$$
\begin{array} { r l } & { \mathrm { T r } \left[ ( \bar { H } - H ^ { \prime } ) ( { \Sigma } ^ { \prime } - \bar { \Sigma } ) \right] } \\ & { \quad \quad = - \mathrm { T r } \left[ \bar { \Sigma } ^ { 1 / 2 } ( H ^ { \prime } - \bar { H } ) \bar { \Sigma } ^ { 1 / 2 } \bar { \Sigma } ^ { - 1 / 2 } ( { \Sigma } ^ { \prime } - \bar { \Sigma } ) \bar { \Sigma } ^ { - 1 / 2 } \right] . } \end{array}
$$

Applying the Frobenius Cauchy–Schwarz inequality $| \operatorname { T r } ( X Y ) | \leq \| X \| _ { \mathrm { F } } \| Y \| _ { \mathrm { F } }$ therefore gives

$$
\begin{array} { r l } & { \left| \operatorname { T r } \left[ ( \bar { H } - H ^ { \prime } ) ( \Sigma ^ { \prime } - \bar { \Sigma } ) \right] \right| } \\ & { \qquad \leq \left\| \bar { \Sigma } ^ { 1 / 2 } ( H ^ { \prime } - \bar { H } ) \bar { \Sigma } ^ { 1 / 2 } \right\| _ { \mathrm { F } } \left\| \bar { \Sigma } ^ { - 1 / 2 } ( \Sigma ^ { \prime } - \bar { \Sigma } ) \bar { \Sigma } ^ { - 1 / 2 } \right\| _ { \mathrm { F } } } \\ & { \qquad \leq \frac { 1 } { \gamma _ { - } } \Delta _ { \mathrm { r e l } } ^ { 2 } , } \end{array}\tag{201}
$$

where the last inequality uses Eq. (198) and the definition of $\Delta _ { \mathrm { r e l } }$ . Consequently,

$$
D \Big ( \rho ( 0 , \Sigma ^ { \prime } ) \Big | \Big | \rho ( 0 , \bar { \Sigma } ) \Big ) + D \Big ( \rho ( 0 , \bar { \Sigma } ) \Big | \Big | \rho ( 0 , \Sigma ^ { \prime } ) \Big ) \leq \frac { 1 } { 2 \gamma _ { - } } \Delta _ { \mathrm { r e l } } ^ { 2 } .\tag{202}
$$

Finally, quantum Pinsker’s inequality [Wat18, Theorem 5.41], with natural logarithms, gives $D ( \rho | | \sigma ) \geq$ ${ \frac { 1 } { 2 } } \| \rho - \sigma \| _ { 1 } ^ { 2 }$ . Applying it once in each direction and adding the resulting inequalities yields

$$
D ( \rho | | \sigma ) + D ( \sigma | | \rho ) \geq \| \rho - \sigma \| _ { 1 } ^ { 2 } .
$$

Taking $\rho = \rho ( 0 , \Sigma ^ { \prime } )$ and $\sigma = \rho ( 0 , \bar { \Sigma } )$ and combining this with Eq. (202), we obtain

$$
\left. \rho ( 0 , \Sigma ^ { \prime } ) - \rho ( 0 , \bar { \Sigma } ) \right. _ { 1 } ^ { 2 } \leq \frac { 1 } { 2 \gamma _ { - } } \Delta _ { \mathrm { r e l } } ^ { 2 } .
$$

Taking square roots and recalling that the trace distance is one half of the trace norm gives

$$
\frac { 1 } { 2 } \left\| \rho ( 0 , \Sigma ^ { \prime } ) - \rho ( 0 , \bar { \Sigma } ) \right\| _ { 1 } \leq \frac { 1 } { 2 \sqrt { 2 \gamma _ { - } } } \Delta _ { \mathrm { r e l } } ,
$$

which is precisely Eq. (184).

We also need to control errors in the first moment. This part is simpler and does not require the warmness assumption.

Lemma 18 (First-moment perturbation). For every valid covariance matrix Σ and all $\mu , \mu ^ { \prime } \in \mathbb { R } ^ { 2 n }$

$$
\frac { 1 } { 2 } \left\| \rho ( \boldsymbol { \mu } ^ { \prime } , \Sigma ) - \rho ( \boldsymbol { \mu } , \Sigma ) \right\| _ { 1 } \leq \frac { 1 } { 2 } \left\| \Sigma ^ { - 1 / 2 } ( \boldsymbol { \mu } ^ { \prime } - \boldsymbol { \mu } ) \right\| _ { 2 } .\tag{203}
$$

Proof. Let $\delta \mu : = \mu ^ { \prime } - \mu$ . The equal-covariance specialization of the Gaussian root-fidelity formula [BBP15, Eq. (9)] gives

$$
{ \cal F } \big ( \rho ( \mu ^ { \prime } , \Sigma ) , \rho ( \mu , \Sigma ) \big ) = \exp \left[ - \frac { 1 } { 8 } \delta \mu ^ { T } \Sigma ^ { - 1 } \delta \mu \right] .
$$

The Fuchs–van de Graaf inequality [Wat18, Theorem 3.36] and $1 - e ^ { - x } \leq x$ for $x \geq 0$ therefore imply

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { 2 } \left\| \rho ( \mu ^ { \prime } , \Sigma ) - \rho ( \mu , \Sigma ) \right\| _ { 1 } \leq \sqrt { 1 - \exp \biggl [ - \displaystyle \frac { 1 } { 4 } \delta \mu ^ { T } \Sigma ^ { - 1 } \delta \mu \biggr ] } } \\ & { \qquad \leq \displaystyle \frac { 1 } { 2 } \sqrt { \delta \mu ^ { T } \Sigma ^ { - 1 } \delta \mu } , } \end{array}
$$

which proves the claim.

## 7.2 Heterodyne tomography and sample complexity

We now apply Theorem 7 to the covariance estimator obtained from heterodyne measurements.

Theorem 8 (Heterodyne tomography under a vacuum gap). Let $\rho ( { \boldsymbol { \mu } } , { \boldsymbol { \Sigma } } )$ be an n-mode Gaussian state satisfying

$$
\Sigma \succeq \left( \frac { 1 } { 2 } + \nu \right) \mathbb { I }\tag{204}
$$

for some $\nu > 0 .$ . For every $\epsilon , \delta \in ( 0 , 1 )$ , there is a non-adaptive protocol using only independent heterodyne measurements that always outputs a valid Gaussian state $\widehat { \rho }$ and, with probability at least $1 - \delta ,$ satisfies

$$
\frac { 1 } { 2 } \left\| \widehat { \rho } - \rho ( \mu , \Sigma ) \right\| _ { 1 } \leq \epsilon
$$

using

$$
N = O \left( \left( 1 + \frac { 1 } { \nu } \right) \frac { n ( n + \log ( 1 / \delta ) ) } { \epsilon ^ { 2 } } \right)\tag{205}
$$

copies.

Proof. In our covariance convention, a heterodyne measurement on $\rho ( \mu , \Sigma )$ produces a classical Gaussian outcome

$$
Y \sim { \mathcal { N } } ( \mu , C ) , \qquad C : = \Sigma + { \frac { 1 } { 2 } } \mathbb { I } .\tag{206}
$$

This is the standard heterodyne law [BMEM25, Eq. (39)], after the fixed permutation from the interleaved quadrature ordering used there to the grouped ordering used in this manuscript.

From N independent heterodyne outcomes $Y _ { 1 } , \dots , Y _ { N }$ , define the empirical mean and the centered empirical covariance by

$$
\widehat { \mu } : = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } Y _ { j } ,
$$

$$
\widehat { C } : = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } ( Y _ { j } - \widehat { \mu } ) ( Y _ { j } - \widehat { \mu } ) ^ { T } .\tag{207}
$$

The centering is important when $\mu \neq 0 :$ the uncentered second moment estimates $C + \mu \mu ^ { T }$ , whereas the theorem places no restriction on the size of $\mu .$

Set

$$
\chi : = \sqrt { 2 n } + \sqrt { 2 \log ( 2 / \delta ) } ,
$$

$$
\zeta : = \frac { 2 \chi } { \sqrt { N } } + \frac { 2 \chi ^ { 2 } } { N } .\tag{208}
$$

The Gaussian concentration bound of [BMEM25, Lemma 9, Eqs. (91), (94), and (95)] implies that, with probability at least $1 - \delta$ ,

$$
\begin{array} { l } { \displaystyle \left\| C ^ { - 1 / 2 } ( \widehat { \mu } - \mu ) \right\| _ { 2 } \leq \frac { \chi } { \sqrt { N } } , } \\ { \displaystyle ( 1 - \zeta ) C \preceq \widehat { C } \preceq ( 1 + \zeta ) C . } \end{array}\tag{209}
$$

Let $\mathcal { E }$ denote the event on which both inequalities in Eq. (209) hold. Thus $\operatorname* { P r } ( { \mathcal { E } } ) \geq 1 - \delta$ . We analyze the estimator on $\mathcal { E }$ throughout the rest of the proof. Notice that both estimates hold on the same event, so we do not need to assume that $\widehat { \mu }$ and $\widehat { C }$ are independent.

Assume $\zeta < 1$ and define the inflated covariance estimator

$$
\widetilde \Sigma : = \frac { \widehat { C } } { 1 - \zeta } - \frac { 1 } { 2 } \mathbb { I } .\tag{210}
$$

The purpose of the factor $( 1 - \zeta ) ^ { - 1 }$ is to turn the two-sided statistical error on $\widehat { C }$ into a one-sided error on the quantum covariance. The protocol outputs

$$
\widehat { \rho } : = \left\{ \begin{array} { l l } { \rho ( \widehat { \mu } , \widetilde { \Sigma } ) , } & { \mathrm { i f ~ } \zeta < 1 \mathrm { ~ a n d ~ } \widetilde { \Sigma } + \frac { i } { 2 } \Omega \succeq 0 , } \\ { \rho ( 0 , \mathbb { I } / 2 ) , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

The second branch is included only to guarantee that every possible measurement transcript produces a valid Gaussian state. We will now show that it is never used on E.

Indeed, the lower covariance bound in Eq. (209) gives $\widehat { C } / ( 1 - \zeta ) \succeq C$ . Recalling that $C = \Sigma + \mathbb { I } / 2$ this immediately implies $\widetilde { \Sigma } \succeq \Sigma$ . Similarly, the upper covariance bound gives

$$
\widetilde { \Sigma } - \Sigma = \frac { \widehat { C } } { 1 - \zeta } - C \preceq \frac { 1 + \zeta } { 1 - \zeta } C - C = \frac { 2 \zeta } { 1 - \zeta } C .
$$

Hence, on $\mathcal { E } ,$

$$
0 \preceq \widetilde { \Sigma } - \Sigma \preceq \frac { 2 \zeta } { 1 - \zeta } C .\tag{211}
$$

In particular,

$$
\widetilde { \Sigma } + \frac { i } { 2 } \Omega = \left( \Sigma + \frac { i } { 2 } \Omega \right) + ( \widetilde { \Sigma } - \Sigma ) \succeq 0 ,
$$

because $\Sigma$ is a valid covariance matrix and $\widetilde { \Sigma } - \Sigma \succeq 0$ . Thus $\widetilde { \Sigma }$ is physical on $\mathcal { E } ,$ and the first branch of the estimator is selected.

We next compare the heterodyne covariance $C$ with the quantum covariance $\Sigma .$ . Define

$$
\alpha _ { \nu } : = 1 + \frac { 1 } { 1 + 2 \nu } ,
$$

$$
\gamma _ { \nu } : = \frac { 2 \nu } { 1 + 2 \nu } .\tag{212}
$$

The promise $\Sigma \succeq ( 1 / 2 + \nu ) \mathbb { I }$ is equivalent to $( 1 + 2 \nu ) \mathbb { I } / 2 \preceq \Sigma .$ and therefore $\Pi / 2 \preceq \Sigma / ( 1 + 2 \nu )$ . It follows that

$$
C = \Sigma + \frac { 1 } { 2 } \mathbb { I } \preceq \left( 1 + \frac { 1 } { 1 + 2 \nu } \right) \Sigma = \alpha _ { \nu } \Sigma .\tag{213}
$$

The same promise also implies that every symplectic eigenvalue of Σ is at least $1 / 2 + \nu ,$ by the ordinary-to-symplectic eigenvalue comparison proved above. Hence Theorem 7 applies to the pair $( \Sigma , \tilde { \Sigma } )$ with

$$
\nu _ { - } = \frac { 1 } { 2 } + \nu , \qquad \gamma _ { - } = 1 - \frac { 1 } { 1 + 2 \nu } = \gamma _ { \nu } .
$$

We first bound the relative covariance error entering that theorem. Conjugating Eq. (211) by $\Sigma ^ { - 1 / 2 }$ and then using Eq. (213) gives

$$
0 \preceq \Sigma ^ { - 1 / 2 } ( \widetilde { \Sigma } - \Sigma ) \Sigma ^ { - 1 / 2 } \preceq \frac { 2 \alpha _ { \nu } \zeta } { 1 - \zeta } \mathbb { I } .
$$

Let

$$
R : = \Sigma ^ { - 1 / 2 } ( \widetilde { \Sigma } - \Sigma ) \Sigma ^ { - 1 / 2 } .
$$

The matrix R is positive semidefinite and has size $2 n \times 2 n$ . The previous inequality says that every eigenvalue of R lies between 0 and $2 \alpha _ { \nu } \zeta / ( 1 - \zeta )$ . Since for a positive semidefinite matrix $\| \cal R \| _ { \mathrm { F } } ^ { 2 }$ is the sum of the squares of its eigenvalues, we obtain

$$
\Bigl \| \Sigma ^ { - 1 / 2 } \bigl ( \widetilde { \Sigma } - \Sigma \bigr ) \Sigma ^ { - 1 / 2 } \Bigr \| _ { \mathrm { F } } \leq \frac { 2 \alpha _ { \nu } \zeta } { 1 - \zeta } \sqrt { 2 n } .\tag{214}
$$

We can now apply Theorem 7. Since a common displacement is implemented by the same unitary on both states, trace distance is invariant under it. Therefore

$$
\begin{array} { r } { \left. \rho ( \widehat { \mu } , \widetilde { \Sigma } ) - \rho ( \widehat { \mu } , \Sigma ) \right. _ { 1 } = \left. \rho ( 0 , \widetilde { \Sigma } ) - \rho ( 0 , \Sigma ) \right. _ { 1 } . } \end{array}
$$

Combining Theorem 7 with Eq. (214) yields

$$
\frac { 1 } { 2 } \left\| \rho ( \widehat { \mu } , \widetilde { \Sigma } ) - \rho ( \widehat { \mu } , \Sigma ) \right\| _ { 1 } \leq \frac { \alpha _ { \nu } } { \sqrt { \gamma _ { \nu } } } \frac { \zeta } { 1 - \zeta } \sqrt { n } .\tag{215}
$$

This is the contribution to the error coming from covariance estimation.

We next control the first moment. From $C \preceq \alpha _ { \nu } \Sigma$ , inversion reverses the positive-semidefinite order, so $C ^ { - 1 } \succeq \alpha _ { \nu } ^ { - 1 } \Sigma ^ { - 1 }$ , or equivalently $\Sigma ^ { - 1 } \preceq \alpha _ { \nu } C ^ { - 1 }$ . Consequently,

$$
\begin{array} { r } { \left\| \Sigma ^ { - 1 / 2 } ( \widehat { \mu } - \mu ) \right\| _ { 2 } \leq \sqrt { \alpha _ { \nu } } \left\| C ^ { - 1 / 2 } ( \widehat { \mu } - \mu ) \right\| _ { 2 } . } \end{array}
$$

Lemma 18 and the mean concentration bound in Eq. (209) therefore give

$$
\frac { 1 } { 2 } \| \rho ( \widehat { \mu } , \Sigma ) - \rho ( \mu , \Sigma ) \| _ { 1 } \leq \frac { \sqrt { \alpha _ { \nu } } } { 2 } \frac { \chi } { \sqrt { N } } .\tag{216}
$$

Importantly, this bound depends only on the normalized estimation error ${ \widehat { \mu } } - \mu$ and not on the size of $\mu$ itself. Thus the theorem allows arbitrary first moments.

We now combine the two errors. On $\mathcal { E } ,$ the output is $\widehat { \rho } = \rho ( \widehat { \mu } , \widetilde { \Sigma } )$ . Inserting the intermediate state $\rho ( \widehat { \mu } , \Sigma )$ and applying the triangle inequality gives

$$
\frac { 1 } { 2 } \left\| \widehat { \rho } - \rho ( \mu , \Sigma ) \right\| _ { 1 } \leq \frac { 1 } { 2 } \left\| \rho ( \widehat { \mu } , \widetilde { \Sigma } ) - \rho ( \widehat { \mu } , \Sigma ) \right\| _ { 1 } + \frac { 1 } { 2 } \left\| \rho ( \widehat { \mu } , \Sigma ) - \rho ( \mu , \Sigma ) \right\| _ { 1 }\tag{217}
$$

$$
\leq \frac { \alpha _ { \nu } } { \sqrt { \gamma _ { \nu } } } \frac { \zeta } { 1 - \zeta } \sqrt { n } + \frac { \sqrt { \alpha _ { \nu } } } { 2 } \frac { \chi } { \sqrt { N } } .\tag{218}
$$

It remains to choose N so that the two terms in Eq. (218) are together at most ϵ. Recall that

$$
\chi = \sqrt { 2 n } + \sqrt { 2 \log ( 2 / \delta ) } , ~ \zeta = \frac { 2 \chi } { \sqrt { N } } + \frac { 2 \chi ^ { 2 } } { N } .
$$

For convenience, set

$$
x : = \frac { \chi } { \sqrt { N } } ,
$$

so that $\zeta = 2 x + 2 x ^ { 2 }$ . We choose N such that

$$
N \geq \frac { 6 4 \alpha _ { \nu } ^ { 2 } } { \gamma _ { \nu } } \frac { n \chi ^ { 2 } } { \epsilon ^ { 2 } } .\tag{219}
$$

Equivalently,

$$
x = \frac { \chi } { \sqrt { N } } \leq \frac { \epsilon \sqrt { \gamma _ { \nu } } } { 8 \alpha _ { \nu } \sqrt { n } } .
$$

Since $\epsilon < 1 , \gamma _ { \nu } \leq 1 , \alpha _ { \nu } \geq 1$ , and $n \geq 1$ , the right-hand side is at most $1 / 8$ . Hence $x \leq 1 / 8$

For $0 \leq x \leq 1 / 8 _ { \cdot }$ , we have

$$
\frac { \zeta } { 1 - \zeta } = \frac { 2 x + 2 x ^ { 2 } } { 1 - 2 x - 2 x ^ { 2 } } \leq 4 x .
$$

Thus the covariance contribution in Eq. (218) satisfies

$$
\frac { \alpha _ { \nu } } { \sqrt { \gamma _ { \nu } } } \frac { \zeta } { 1 - \zeta } \sqrt { n } \leq \frac { 4 \alpha _ { \nu } } { \sqrt { \gamma _ { \nu } } } x \sqrt { n } \leq \frac { 4 \alpha _ { \nu } } { \sqrt { \gamma _ { \nu } } } \frac { \epsilon \sqrt { \gamma _ { \nu } } } { 8 \alpha _ { \nu } \sqrt { n } } \sqrt { n } = \frac { \epsilon } { 2 } .
$$

The first-moment contribution is even smaller:

$$
\frac { \sqrt { \alpha _ { \nu } } } { 2 } \frac { \chi } { \sqrt { N } } = \frac { \sqrt { \alpha _ { \nu } } } { 2 } x \leq \frac { \epsilon \sqrt { \gamma _ { \nu } } } { 1 6 \sqrt { \alpha _ { \nu } n } } \leq \frac { \epsilon } { 1 6 } ,
$$

where the last inequality again uses $\gamma _ { \nu } \leq 1 , \alpha _ { \nu } \geq 1$ , and $n \geq 1$ . Therefore, on the event $\mathcal { E } _ { i }$

$$
\frac { 1 } { 2 } \| \widehat { \rho } - \rho ( \mu , \Sigma ) \| _ { 1 } \leq \frac { \epsilon } { 2 } + \frac { \epsilon } { 1 6 } < \epsilon .
$$

Since $\Pr ( \mathcal { E } ) \geq 1 - \delta _ { \mathrm { m } }$ , the estimator succeeds with the required probability.

Finally, using $( a + b ) ^ { 2 } \leq 2 a ^ { 2 } + 2 b ^ { 2 } .$

$$
\chi ^ { 2 } \leq 4 \left( n + \log \frac { 2 } { \delta } \right) ,
$$

while

$$
\frac { \alpha _ { \nu } ^ { 2 } } { \gamma _ { \nu } } = \frac { 2 ( 1 + \nu ) ^ { 2 } } { \nu ( 1 + 2 \nu ) } \leq 2 \left( 1 + \frac { 1 } { \nu } \right) .
$$

Substituting these bounds into Eq. (219) gives

$$
N = O \left( \left( 1 + \frac { 1 } { \nu } \right) \frac { n \big ( n + \log ( 1 / \delta ) \big ) } { \epsilon ^ { 2 } } \right) ,
$$

which is $\operatorname { E q } .$ . (205). For every fixed $\nu > 0$ and constant failure probability, this is $O _ { \nu } ( n ^ { 2 } / \epsilon ^ { 2 } )$ □

Corollary 19 (Interpolation in the vacuum gap). Let $\rho ( { \boldsymbol { \mu } } , { \boldsymbol { \Sigma } } )$ be an n-mode Gaussian state satisfying

$$
\Sigma \succeq \left( \frac { 1 } { 2 } + \nu \right) \mathbb { I }
$$

for some $\nu > 0 .$ . For constant failure probability, there is a non-adaptive protocol using only independent heterodyne measurements that learns $\rho ( { \boldsymbol { \mu } } , { \boldsymbol { \Sigma } } )$ to trace-distance error ϵ using

$$
N = O \left( \frac { n ^ { 2 } } { \epsilon ^ { 2 } } \operatorname* { m i n } \left\{ n , 1 + \frac { 1 } { \nu } \right\} \right)\tag{220}
$$

copies.

Proof. We combine two heterodyne upper bounds. First, Theorem 8 gives

$$
N = O \left( \left( 1 + \frac { 1 } { \nu } \right) \frac { n ^ { 2 } } { \epsilon ^ { 2 } } \right)
$$

for constant failure probability. On the other hand, the general heterodyne tomography bound of [BMEM25, Theorem 10] applies without using the vacuum-gap promise and gives

$$
N = O \left( { \frac { n ^ { 3 } } { \epsilon ^ { 2 } } } \right) .
$$

Since both protocols use only independent heterodyne measurements, we may simply use whichever of the two bounds is smaller for the given value of $\nu .$ Therefore

$$
N = O \left( \operatorname* { m i n } \left\{ \frac { n ^ { 3 } } { \epsilon ^ { 2 } } , \left( 1 + \frac { 1 } { \nu } \right) \frac { n ^ { 2 } } { \epsilon ^ { 2 } } \right\} \right) = O \left( \frac { n ^ { 2 } } { \epsilon ^ { 2 } } \operatorname* { m i n } \left\{ n , 1 + \frac { 1 } { \nu } \right\} \right) ,
$$

which proves Eq. (220).

Corollary 19 makes explicit how the available heterodyne upper bound changes with the distance from the vacuum boundary. When $\nu = O ( 1 / n )$ , the general ${ \cal O } ( \bar { n ^ { 3 } } / \epsilon ^ { 2 } )$ bound is at least as good as the new estimate. When $\nu = \Theta ( 1 )$ , the second case gives $O ( n ^ { 2 } / \epsilon ^ { 2 } )$ . Thus the upper bound interpolates between the cold and warm regimes.

## 8 Matching lower bound for learning warm Gaussian states

Theorem 9. There exists constants $C > 0$ and $\varepsilon _ { 0 } > 0$ such that the following holds: For any $\nu > 0 ,$ $0 < \varepsilon < \varepsilon _ { 0 } ,$ and suficiently large $n ,$ any scheme that can learn any passive Gaussian state $\rho ( 0 , \Sigma )$ to ε trace distance with $2 / 3$ probability with the promise that $\Sigma \geq ( \frac { 1 } { 2 } + \nu ) \mathbb { I }$ requires at least $N \geq C n ^ { 2 } / \varepsilon ^ { 2 }$ copies.

Proof. Without loss of generality, let $n$ be a suficiently large even number and let $s : = n / 2$ . Consider a similar n-mode passive Gaussian ensemble used in Sec. 6:

$$
\rho _ { U } : = \Gamma ( U ) \rho _ { 0 } \Gamma ( U ) ^ { \dagger } , \quad \rho _ { 0 } : = \tau _ { \nu + \Delta } ^ { \otimes s } \otimes \tau _ { \nu } ^ { \otimes s } ,\tag{221}
$$

where $U \in \{ U _ { a } \in \mathbb { U } ( n ) \} _ { a = 1 } ^ { M }$ is a size $M = 2 ^ { \Theta ( n ^ { 2 } ) }$ ensemble that satisfies

$$
\mathrm { T r } \left( \Pi _ { > s } U _ { a } ^ { \dagger } U _ { b } \Pi _ { \le s } U _ { b } ^ { \dagger } U _ { a } \right) \ge \frac { n } { 8 } , \quad \forall a \ne b \in [ M ] ,\tag{222}
$$

the existence of which has been shown in Sec. 6. We assume $\nu \geq 1$ without loss of generality, as a lower bound for a larger ν automatically implies a lower bound for a smaller $\nu . \Delta$ is chosen as

$$
\Delta : = C _ { 1 } \varepsilon \nu / { \sqrt { n } } ,\tag{223}
$$

with $C _ { 1 } > 0$ a constant to be decided later. All $\rho _ { U }$ in this ensemble clearly satisfy $\Sigma \geq ( \frac { 1 } { 2 } + \nu ) \mathbb { I }$

We first show this ensemble is a ε-packing net in trace distance. Assume $0 < \varepsilon < \varepsilon _ { 0 }$ with $\varepsilon _ { 0 }$ being a suficiently small constant. Say $\varepsilon _ { 0 } : = 1 / 1 8 0 0$ . For $a ,$ b such that $D _ { \mathrm { t r } } ( \rho _ { U _ { a } } , \rho _ { U _ { b } } ) \geq 3 \varepsilon _ { 0 }$ it trivially holds that $D _ { \mathrm { t r } } ( \rho _ { U _ { a } } , \rho _ { U _ { b } } ) > 3 \varepsilon$ . Otherwise, consider applying a heterodyne measurement on $\rho _ { U _ { a } }$ . The outcome distribution is a 2n-dimensional zero-mean Gaussian $\textstyle { \mathcal { N } } ( 0 , { \tilde { \Sigma } } _ { a } )$ with covariance matrix given by (denote the covariance matrix of $\rho _ { U _ { a } }$ by $\Sigma _ { a } )$ :

$$
\tilde { \Sigma } _ { a } : = \Sigma _ { a } + \frac { 1 } { 2 } \mathrm { I } = G _ { U _ { a } } \left( ( 1 + \nu ) \mathrm { I } + \Delta \Pi _ { \leq s } \right) ^ { \oplus 2 } G _ { U _ { a } } ^ { T } .\tag{224}
$$

We start to lower bound the pairwise trace distance.

$$
\begin{array} { r l } & { D _ { \mathrm { t r } } \big ( \rho _ { U _ { a } } , \rho _ { U _ { b } } \big ) \overset { \mathrm { ( i ) } } { \geq } \mathrm { T V } ( \mathcal { N } ( 0 , \tilde { \Sigma } _ { a } ) , \mathcal { N } ( 0 , \tilde { \Sigma } _ { b } ) ) } \\ & { \qquad \quad \stackrel { \mathrm { ( i i ) } } { \geq } C _ { 2 } \| \tilde { \Sigma } _ { a } ^ { - \frac { 1 } { 2 } } ( \tilde { \Sigma } _ { b } - \tilde { \Sigma } _ { a } ) \tilde { \Sigma } _ { a } ^ { - \frac { 1 } { 2 } } \| _ { \mathrm { F } } } \\ & { \qquad \stackrel { \mathrm { ( i i i ) } } { \geq } C _ { 2 } \frac { 1 } { 1 + { \nu } + \Delta } \| \tilde { \Sigma } _ { b } - \tilde { \Sigma } _ { a } \| _ { \mathrm { F } } . } \end{array}\tag{225}
$$

Here, (i) uses the data-processing inequality; (2) uses the TV distance bounds for classical Gaussians [DMR18, AAL23] (here the TV distance is upper bounded by $1 / 6 0 0$ by assumption and hence the inequality can be used); (3) uses the fact that the maximal singular value of $\tilde { \Sigma } _ { a }$ is $( 1 + \nu + \Delta )$ ). Next, define $W : = U _ { a } ^ { \dagger } U _ { b }$

$$
\begin{array} { r l } & { \| \tilde { \boldsymbol { \Sigma } } _ { b } - \tilde { \boldsymbol { \Sigma } } _ { a } \| _ { \mathrm { F } } ^ { 2 } \stackrel { \mathrm { ( i ) } } { = } 2 \Delta ^ { 2 } \mathrm { T r } \Big ( ( \Pi _ { \leq s } - W \Pi _ { \leq s } W ^ { \dagger } ) ^ { 2 } \Big ) } \\ & { \qquad = 4 \Delta ^ { 2 } \mathrm { T r } \Big ( \Pi _ { \leq s } W \Pi _ { > s } W ^ { \dagger } \Big ) } \\ & { \qquad \stackrel { \mathrm { ( i i ) } } { \geq } \frac { 1 } { 2 } \Delta ^ { 2 } n . } \end{array}\tag{226}
$$

Here, (i) uses the realification map (see Sec. 3); (ii) uses the defining property of our unitary ensemble (see Eq. (150)). Putting the above two inequalities together,

$$
D _ { \mathrm { t r } } ( \rho _ { U _ { a } } , \rho _ { U _ { b } } ) \geq \frac { C _ { 2 } C _ { 1 } } { 3 \sqrt { 2 } } \varepsilon ,\tag{227}
$$

where we have used $\nu \geq 1$ and the definition of $\Delta .$ Setting $C _ { 1 } : = 9 \sqrt { 2 } / C _ { 2 }$ yields $D _ { \mathrm { t r } } ( \rho _ { U _ { a } } , \rho _ { U _ { b } } ) \geq 3 \varepsilon$

Now, we use the standard Fano’s method to prove the claimed lower bound: Let Alice and Bob be two players of a communication game. Alice chooses $a \in [ M ]$ at uniform random, then sends $\rho _ { U _ { a } } ^ { \otimes { N } }$ to Bob, whose can perform arbitrary quantum measurements to guess the value of a with maximal success probability. Suppose there is a learning protocol satisfies the assumption of Theorem 9 exists. Then, Bob can guess correctly with probability at least $2 / 3$ . Denote Bob’s guess by aˆ. Fano’s inequality gives that [Cov99]:

$$
I ( a : \hat { a } ) \geq \frac { 2 } { 3 } \log M - \log 2 = \Omega ( n ^ { 2 } ) .\tag{228}
$$

On the other hand, $I ( a : { \hat { a } } )$ is upper bounded by the Holevo $\chi$ quantity of the ensemble $\{ \rho _ { U _ { a } } ^ { \otimes N } : a \sim$ Unif $( [ M ] ) ]$ thanks to Holevo theorem [Hol73]:

$$
\begin{array} { r l } & { \chi : = D ( \underset { a } { \mathbb { E } } | a \rangle \langle a | \otimes \rho _ { U _ { a } } ^ { \otimes N } | | \underset { a } { \mathbb { E } } | a \rangle \langle a | \otimes \underset { b } { \mathbb { E } } \rho _ { U _ { b } } ^ { \otimes N } ) } \\ & { \quad \overset { \mathrm { ( i ) } } { \leq } N \underset { a , b } { \mathbb { E } } D ( \rho _ { U _ { a } } \| \rho _ { U _ { b } } ) } \\ & { \quad \overset { \mathrm { ( i i ) } } { \leq } N \underset { a , b } { \mathrm { m a x } } \frac { 1 } { 2 } { \mathrm { T r } } [ ( \Sigma _ { a } - \Sigma _ { b } ) ( H _ { b } - H _ { a } ) ] . } \end{array}\tag{229}
$$

Here (i) uses the convexity, joint convexity, and tensorization properties of the quantum relative entropy $D ;$ (ii) is an upper bound on the relative entropy between Gaussian states given in [FRSF26], where $H _ { a }$ is the parent Hamiltonian of $\rho _ { U _ { a } }$ . For the passive Gaussian state, the Hamiltonian is given by

$$
H _ { a } : = f ( \Sigma _ { a } ) , \quad { \mathrm { w h e r e ~ } } f ( x ) : = { \frac { 1 } { 2 } } \log ( { \frac { 2 x + 1 } { 2 x - 1 } } ) , \forall x > { \frac { 1 } { 2 } } .\tag{230}
$$

See $\mathrm { [ C M F ^ { + } 2 6 }$ , Appendix C.2] for more details. Therefore,

$$
\begin{array} { r l } & { \mathrm { T r } \big [ ( \Sigma _ { a } - \Sigma _ { b } ) ( H _ { b } - H _ { a } ) \big ] \stackrel { \mathrm { ( i ) } } { = } \Delta \left( f ( \frac { 1 } { 2 } + \nu ) - f ( \frac { 1 } { 2 } + \nu + \Delta ) \right) 2 \mathrm { T r } \Big ( ( \Pi _ { \leq s } - W \Pi _ { \leq s } W ^ { \dagger } ) ^ { 2 } \Big ) } \\ & { \qquad \stackrel { \mathrm { ( i i ) } } { \leq } \frac { \Delta ^ { 2 } } { \nu ^ { 2 } } n = C _ { 1 } \varepsilon ^ { 2 } . } \end{array}\tag{231}
$$

Here, (i) is by direct calculation and then uses the realification map; For (ii), one can verify that $f ^ { \prime } ( x ) < 0$ and $f ^ { \prime \prime } ( x ) > 0$ . Thus,

$$
f ( \frac { 1 } { 2 } + \nu ) - f ( \frac { 1 } { 2 } + \nu + \Delta ) \leq - \Delta f ^ { \prime } ( \frac { 1 } { 2 } + \nu ) \leq \frac { \Delta } { 2 \nu ^ { 2 } } .\tag{232}
$$

The factor of n appears by using $| \Pi _ { \le s } - W \Pi _ { \le s } W ^ { \dagger } | \le \mathtt { I } _ { n }$ . Putting things together, we have

$$
\Omega ( n ^ { 2 } ) \leq I ( a : \hat { a } ) \leq \chi \leq C _ { 1 } N \varepsilon ^ { 2 } .\tag{233}
$$

Therefore, $N = \Omega ( n ^ { 2 } / \varepsilon ^ { 2 } )$ . Note that the prefactor hidden in Ω can be chosen as an absolute constant. This completes the proof for Theorem 9. □

## References

[AAL23] Jamil Arbas, Hassan Ashtiani, and Christopher Liaw. Polynomial time and private learning of unbounded gaussian mixture models. In International Conference on Machine Learning, pages 1018–1040. PMLR, 2023.

[ABDH<sup>+</sup>20] Hassan Ashtiani, Shai Ben-David, Nicholas JA Harvey, Christopher Liaw, Abbas Mehrabian, and Yaniv Plan. Near-optimal sample complexity bounds for robust learning of gaussian mixtures via compression schemes. Journal of the ACM (JACM), 67(6):1–42, 2020.

[AGZ10] Greg W. Anderson, Alice Guionnet, and Ofer Zeitouni. An Introduction to Random Matrices, volume 118 of Cambridge Studies in Advanced Mathematics. Cambridge University Press, 2010.

[BBDV<sup>+</sup>25] Caterina Braggio, L Balembois, R Di Vora, Z Wang, J Travesedo, L Pallegoix, G Carugno, A Ortolan, G Ruoso, U Gambardella, et al. Quantum-enhanced sensing of axion dark matter with a transmon-based single microwave photon counter. Physical Review X, 15(2):021031, 2025.

[BBP15] Leonardo Banchi, Samuel L. Braunstein, and Stefano Pirandola. Quantum fidelity for arbitrary gaussian states. Physical Review Letters, 115(26):260501, 2015.

[BLMT24] Ainesh Bakshi, Allen Liu, Ankur Moitra, and Ewin Tang. Structure learning of hamiltonians from real-time evolution. In 2024 IEEE 65th Annual Symposium on Foundations of Computer Science (FOCS), pages 1037–1050. IEEE, 2024.

[BMEM25] Lennart Bittel, Francesco A Mele, Jens Eisert, and Antonio A Mele. Energy-independent tomography of gaussian states. arXiv preprint arXiv:2508.14979, 2025.

[BMM<sup>+</sup>25] Lennart Bittel, Francesco Anna Mele, Antonio Anna Mele, Salvatore Tirone, and Ludovico Lami. Optimal estimates of trace distance between bosonic gaussian states and applications to learning. Quantum, 9:1769, June 2025.

[BPK<sup>+</sup>21] Kelly M Backes, Daniel A Palken, S Al Kenany, Benjamin M Brubaker, SB Cahn, A Droster, Gene C Hilton, Sumita Ghosh, H Jackson, Steve K Lamoreaux, et al. A quantum enhanced search for dark matter axions. Nature, 590(7845):238–242, 2021.

[CCHL22] Sitan Chen, Jordan Cotler, Hsin-Yuan Huang, and Jerry Li. Exponential separations between learning with and without quantum memory. In 2021 IEEE 62nd Annual Symposium on Foundations of Computer Science (FOCS), pages 574–585. IEEE, 2022.

[CFG<sup>+</sup>26] Senrui Chen, Marco Fanizza, Filippo Girardi, Ludovico Lami, Francesco Anna Mele, Michael Walter, and Freek Witteveen. Optimal tomography of bosonic and fermionic gaussian states. arXiv preprint arXiv:2607.11847, 2026.

[CGS11] Allison Cuttler, Curtis Greene, and Mark Skandera. Inequalities for symmetric means. European Journal of Combinatorics, 32(6):745–761, 2011.

[CGYZ26] Sitan Chen, Weiyuan Gong, Qi Ye, and Zhihan Zhang. The log log jam in gaussian state tomography. arXiv preprint arXiv:2607.12983, 2026.

[CGZ26] Senrui Chen, Weiyuan Gong, and Sisi Zhou. Instance-optimal high-precision shadow tomography with few-copy measurements: A metrological approach. arXiv preprint arXiv:2602.04952, 2026.

[CHL<sup>+</sup>23] Sitan Chen, Brice Huang, Jerry Li, Allen Liu, and Mark Sellke. When does adaptivity help for quantum state learning? In 2023 IEEE 64th Annual Symposium on Foundations of Computer Science (FOCS), pages 391–404. IEEE, 2023.

[CLL24] Sitan Chen, Jerry Li, and Allen Liu. An optimal tradeof between entanglement and copy complexity for state tomography. In Proceedings of the 56th Annual ACM Symposium on Theory of Computing, pages 1331–1342, 2024. Full version available at arXiv:2402.16353; theorem and section numbers refer to the full arXiv version.

[CMF<sup>+</sup>26] Senrui Chen, Francesco Anna Mele, Marco Fanizza, Alfred Li, Zachary Mann, Hsin-Yuan Huang, Yanbei Chen, and John Preskill. Towards sample-optimal learning of bosonic gaussian quantum states. arXiv preprint arXiv:2603.18136, 2026.

[Cov99] Thomas M Cover. Elements of information theory. John Wiley & Sons, 1999.

[DFK<sup>+</sup>18] Nick Du, N Force, R Khatiwada, E Lentz, R Ottens, LJ Rosenberg, Gray Rybka, G Carosi, N Woollett, D Bowring, et al. Search for invisible axion dark matter with the axion dark matter experiment. Physical review letters, 120(15):151301, 2018.

[DG13] Jan Dereziński and Christian Gérard. Mathematics of Quantization and Quantum Fields. Cambridge Monographs on Mathematical Physics. Cambridge University Press, 2013.

[DLT02] David P DiVincenzo, Debbie W Leung, and Barbara M Terhal. Quantum data hiding. IEEE Transactions on Information Theory, 48(3):580–598, 2002.

[DMR18] Luc Devroye, Abbas Mehrabian, and Tommy Reddad. The total variation distance between high-dimensional gaussians with the same mean. arXiv preprint arXiv:1810.08693, 2018.

[DMR20] Luc Devroye, Abbas Mehrabian, and Tommy Reddad. The minimax learning rates of normal and ising undirected graphical models. Electronic Journal of Statistics, 14(1):2338–2361, 2020.

[EMD26] Enrique Escobar Fernandez-Marcote, Marco Fanizza, and Daniel Stilck França. Advancing Gaussian hamiltonian and graph learning. Manuscript in preparation, 2026.

[FH13] William Fulton and Joe Harris. Representation theory: a first course. Springer Science & Business Media, 2013.

[FHS90] R. D. Foley, T. P. Hill, and M. C. Spruill. A generalization of Lévy’s concentration-variance inequality. Probability Theory and Related Fields, 86(1):53–62, 1990.

[FRG25] Matteo Fadel, Noah Roux, and Manuel Gessner. Quantum metrology with a continuousvariable system. Reports on Progress in Physics, 88(10):106001, 2025.

[FRSF26] Marco Fanizza, Cambyse Rouzé, and Daniel Stilck França. Eficient Hamiltonian, structure and trace distance learning of Gaussian states. Nature Communications, September 2026.

[Gla63] Roy J Glauber. Coherent and incoherent states of the radiation field. Physical Review, 131(6):2766, 1963.

[GLM11] Vittorio Giovannetti, Seth Lloyd, and Lorenzo Maccone. Advances in quantum metrology. Nature photonics, 5(4):222–229, 2011.

[HBC<sup>+</sup>22] Hsin-Yuan Huang, Michael Broughton, Jordan Cotler, Sitan Chen, Jerry Li, Masoud Mohseni, Hartmut Neven, Ryan Babbush, Richard Kueng, John Preskill, et al. Quantum advantage in learning from experiments. Science, 376(6598):1182–1186, 2022.

[HCMP26] Hsin-Yuan Huang, Soonwon Choi, Jarrod R McClean, and John Preskill. Vast world of quantum advantage. Physical Review X, 16(3):030501, 2026.

[HHJ<sup>+</sup>17] Jeongwan Haah, Aram W. Harrow, Zhengfeng Ji, Xiaodi Wu, and Nengkun Yu. Sampleoptimal tomography of quantum states. IEEE Transactions on Information Theory, page 1, 2017.

[HKP21] Hsin-Yuan Huang, Richard Kueng, and John Preskill. Information-theoretic bounds on quantum advantage in machine learning. Physical Review Letters, 126(19):190505, 2021.

[HLP52] Godfrey Harold Hardy, John Edensor Littlewood, and George Pólya. Inequalities. Cambridge university press, 1952.

[Hol73] Alexander Semenovich Holevo. Bounds for the quantity of information transmitted by a quantum communication channel. Problemy Peredachi Informatsii, 9(3):3–11, 1973.

[Hol24] AS Holevo. On estimates of trace-norm distance between quantum gaussian states. arXiv preprint arXiv:2408.11400, 2024.

[HW99] Alexander S Holevo and Reinhard F Werner. Evaluating capacities of bosonic gaussian channels. arXiv preprint quant-ph/9912067, 1999.

[JKK05] Norman L Johnson, Adrienne W Kemp, and Samuel Kotz. Univariate discrete distributions. John Wiley & Sons, 2005.

[KLMR26] Ufuk Keskin, Jason Luo, Mahbod Majid, and Matthew Radzihovsky. Tight lower bounds for state tomography with limited entanglement. arXiv preprint arXiv:2609.05718, 2026.

[LBØ<sup>+</sup>25] Zheng-Hao Liu, Romain Brunel, Emil EB Østergaard, Oscar Cordero, Senrui Chen, Yat Wong, Jens AH Nielsen, Axel B Bregnsbo, Sisi Zhou, Hsin-Yuan Huang, et al. Quantum learning advantage on a scalable photonic platform. Science, 389(6767):1332–1335, 2025.

[LN25] Angus Lowe and Ashwin Nayak. Lower bounds for learning quantum states with single-copy measurements. ACM Transactions on Computation Theory, 17(1):1–42, 2025.

[LR09] Alexander I Lvovsky and Michael G Raymer. Continuous-variable optical quantum-state tomography. Reviews of modern physics, 81(1):299–332, 2009.

[MA17] Olivier Marchal and Julyan Arbel. On the sub-gaussianity of the beta and dirichlet distributions. Electronic Communications in Probability, 22(paper no. 54):1–14, 2017.

[Mel26] Francesco Anna Mele. Advances in quantum learning theory with bosonic systems. arXiv preprint arXiv:2605.08082, 2026.

[MM13] Elizabeth S. Meckes and Mark W. Meckes. Spectral measures of powers of random matrices. Electronic Communications in Probability, 18(78):1–13, September 2013.

[MMB<sup>+</sup>25] Francesco A Mele, Antonio A Mele, Lennart Bittel, Jens Eisert, Vittorio Giovannetti, Ludovico Lami, Lorenzo Leone, and Salvatore FE Oliviero. Learning quantum states of continuousvariable systems. Nature Physics, pages 1–7, 2025.

[Mon13] Alex Monras. Phase space formalism for quantum estimation of gaussian states. arXiv preprint arXiv:1303.3682, 2013.

[MWG<sup>+</sup>20] L McCuller, C Whittle, D Ganapathy, K Komori, M Tse, A Fernandez-Galiana, L Barsotti, Peter Fritschel, M MacInnis, F Matichard, et al. Frequency-dependent squeezing for advanced ligo. Physical review letters, 124(17):171102, 2020.

[NZ26] Ashwin Nayak and Xingyu Zhou. Optimal low-rank quantum state tomography with bounded-sample joint measurements. arXiv preprint arXiv:2609.10514, 2026.

[OW16] Ryan O’Donnell and John Wright. Eficient quantum tomography. In Proceedings of the Forty-eighth Annual ACM Symposium on Theory of Computing, STOC ’16, pages 899–912, New York, NY, USA, 2016. ACM.

[PSTW25] Angelos Pelecanos, Jack Spilecki, Ewin Tang, and John Wright. Mixed state tomography reduces to pure state tomography. Preprint arXiv:2511.15806, 2025.

[PZC26] Harald Putterman, Alexander Zlokapa, and Jordan Cotler. When quantum thermal states look classical. arXiv preprint arXiv:2607.28536, 2026.

[Rub26] Ron Rubin. Entangled measurements are necessary for optimal tomography of mixed fermionic Gaussian states and of bosonic Gaussian states near the vacuum. arXiv preprint arXiv:2609.23189v1, 2026.

[Sra16] Suvrit Sra. On inequalities for normalized schur functions. European Journal of Combinatorics, 51:492–494, 2016.

[SS07] Moshe Shaked and J. George Shanthikumar. Stochastic Orders. Springer Series in Statistics. Springer, New York, NY, 2007.

[Sud63] E. C. George Sudarshan. Equivalence of semiclassical and quantum mechanical descriptions of statistical light beams. Physical Review Letters, 10(7):277, 1963.

[TNL16] Mankei Tsang, Ranjith Nair, and Xiao-Ming Lu. Quantum theory of superresolution for two incoherent optical point sources. Physical Review X, 6(3):031033, 2016.

[Wai19] Martin J Wainwright. High-dimensional statistics: A non-asymptotic viewpoint, volume 48. Cambridge university press, 2019.

[Wat18] John Watrous. The Theory of Quantum Information. Cambridge University Press, Cambridge, 2018.

[WPGP<sup>+</sup>12] Christian Weedbrook, Stefano Pirandola, Raúl García-Patrón, Nicolas J Cerf, Timothy C Ralph, Jefrey H Shapiro, and Seth Lloyd. Gaussian quantum information. Reviews of Modern Physics, 84(2):621–669, 2012.

[Zur03] Wojciech Hubert Zurek. Decoherence, einselection, and the quantum origins of the classical. Reviews of modern physics, 75(3):715, 2003.