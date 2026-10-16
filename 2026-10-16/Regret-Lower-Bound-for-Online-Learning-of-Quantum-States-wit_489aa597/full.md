# Regret Lower Bound for Online Learning of Quantum States with Logarithmic Loss, with Applications to Quantum State Tomography and Confidence Sequences for Matrix Means

Yen-Huan Li<sup>1,2,3</sup> and Wei-Fu Tseng<sup>1</sup>

<sup>1</sup>Department of Computer Science and Information Engineering, National Taiwan University <sup>2</sup>Department of Mathematics, National Taiwan University <sup>3</sup>Center for Quantum Science and Engineering, National Taiwan University

## Abstract

We study the problem ofonline learning ofquantum states with logarithmic loss, which is a quantum generalization of online portfolio selection and has applications to quantum state tomography, the task of estimating an unknown quantum state from measurement outcomes. We first establish a regret lower bound, showing that the regret rates of Exponential Weights and the algorithm of Tseng et al. (2025) have optimal polynomial dependence on the dimension. Our analysis leads to two further results. First, our lower-bound proof leads to a quantum state tomography algorithm that is sample optimal up to polylogarithmic factors in the sample size. Second, we develop an algorithm for constructing confidence sequences for the mean ofa matrix-valued random variable. A numerical experiment shows that the latter algorithm is competitive with existing methods.

## 1 Introduction

In the mathematical formalism of quantum mechanics, a quantum state is represented by a density matrix, which is a Hermitian positive semidefinite matrix with unit trace. A measurement is represented by a positive operator-valued measure (POVM), which is a collection of Hermitian positive semidefinite matrices that sum to the identity matrix. Given a density matrix $\rho$ and a POVM $\{ M _ { \iota } \mid \iota \in \mathcal { S } \}$ indexed by a set $\mathcal { I } _ { : }$ , the measurement outcome is a random variable η taking values in $\mathcal { I }$ whose distribution is given by the Born rule:

$$
\mathsf { P } \big ( \eta = \iota \big ) = \mathrm { t r } \big ( M _ { \iota } \rho \big ) , \quad \mathrm { f o r \ a l l \ } \iota \in \mathcal { I } ,\tag{1}
$$

For convenience, we will write $\mathcal { D } _ { d }$ for the set of $d \times d$ density matrices.

Consider the following multi-round game of online learning of quantum states with logarithmic loss between two players, Physicist and Nature. The game consists of T rounds. At each round $1 \leq t \leq T$

• Physicist announces a density matrix $\rho _ { t } \in \mathcal { D } _ { d } ;$

• Nature then announces a non-zero $d \times d$ Hermitian positive semidefinite matrix $A _ { t }$ whose eigenvalues lie $\mathrm { i n } ^ { 1 } [ 0 , 1 ] $

• finally, Physicist suffers the loss $f _ { t } ( \rho _ { t } ) \mathrel { \mathop : } = - \log \operatorname { t r } \left( A _ { t } \rho _ { t } \right)$

Physicist aims to achieve small regret against any strategy of Nature. The regret is defined as

$$
\mathrm { R e g r e t } _ { T } : = \sum _ { t = 1 } ^ { T } f _ { t } ( \rho _ { t } ) - \operatorname* { m i n } _ { \rho \in \mathcal { D } _ { d } } \sum _ { t = 1 } ^ { T } f _ { t } ( \rho ) .
$$

For convenience, throughout this paper we refer to this game simply as online learningofquantum states (OLQS), although online learning of quantum states has also been studied under other loss functions (Aaronson et al., 2018, 2019; Bansal et al., 2025; Chen et al., 2024; Yang et al., 2020).

The game was independently proposed and studied by Lin et al. (2021b) and Zimmert et al. (2022). The work of Lin et al. (2021b) was motivated by quantum state tomography, the task of estimating an unknown quantum state from measurement outcomes (Paris and Reháˇcek<sup>ˇ</sup> , 2004). Suppose that a fixed POVM is used for the measurements and that $M _ { t }$ is the POVM element corresponding to the outcome observed at round t. Then, $f _ { t } ( \rho )$ is precisely the negative log-likelihood of the observed outcome under the candidate state $\rho .$ Thus, OLQS can be viewed as an online formulation of maximum-likelihood quantum state tomography (Hradil, 1997). In addition, it is easily checked that any no-regret algorithm for OLQS, together with an online-to-batch conversion (Hazan, 2023; Orabona, 2026), can be used as a stochastic optimization method for approximating the maximum-likelihood estimate in quantum state tomography.

OLQS is also closely connected to online portfolio selection introduced by Cover (1991), as noted by both Lin et al. (2021b) and Zimmert et al. (2022). Indeed, suppose that the matrices $M _ { t }$ are all diagonal in a fixed common basis. Since tr $\left( M _ { t } \rho \right)$ then depends only on the diagonal entries of $\rho$ in this basis, Physicist may, without loss of generality, restrict attention to density matrices that are diagonal in the same basis. Identifying the diagonal entries of $\rho _ { t }$ with a portfolio vector $x _ { t }$ in the probability simplex and those of $M _ { t }$ with a return vector $a _ { t } \in [ 0 , 1 ] ^ { d }$ , we obtain $f _ { t } ( \rho _ { t } ) = - \log \langle a _ { t } , x _ { t } \rangle$ , which is exactly the logarithmic loss in online portfolio selection.

Since neither the logarithmic loss nor its gradient is Lipschitz continuous, regret guarantees of textbook online convex optimization algorithms do not directly apply to online portfolio selection. Indeed, optimal online portfolio selection, in both regret rate and computational complexity, has been an open problem for more than three decades. Cover and Ordentlich (1996) and Ordentlich and Cover (1998) established that the minimax-optimal regret rate is O(d logT) and this rate can be achieved by the Universal Portfolios algorithm, an instance of Exponential Weights (Hazan et al., 2007; van der Hoeven et al., 2018; Vovk and Watkins, 1998). Unfortunately, the computational complexity of Universal Portfolios scales poorly with the time horizon (Kalai and Vempala, 2002), with a per-round runtime of $O ( d ^ { 4 } T ^ { 1 4 } )$ (Luo et al., 2018).

Table 1 summarizes existing state-of-the-art algorithms for online portfolio selection and their counterparts for OLQS. The best-known algorithm for online portfolio selection is perhaps VB-FTRL (Jézéquel et al., 2026), which achieves a regret rate of O(d log(d + T)) with a per-round runtime of $O ( d ^ { 2 } T )$ . When $T \gg d ,$ , the regret rate of VB-FTRL is already optimal, but the dependence of its per-round runtime on $T$ becomes undesirable.

A trade-off between regret and computational complexity is achieved by PAE+DONS (Mhammedi and Rakhlin, 2022) and BISONS (Zimmert et al., 2022), which attain regret rates of $O ( d ^ { 2 }$ polylog(T )) with per-round runtimes of O(poly(d)polylog(T )). Another trade-off between regret and computational complexity is provided by the barrier subgradient method (Nesterov, 2011), Soft-Bayes (Orseau et al., 2017), and LB-FTRL (Tsai et al., 2023c). These algorithms achieve regret rates of $\widetilde { O } ( \sqrt { d T } )$ ) with per-round runtimes of $\widetilde O ( d )$

Following the convention in quantum information theory, we will refer to online portfolio selection and OLQS as the classical case and the quantum case, respectively. As shown in Table 1, existing algorithms for OLQS are generalizations of their classical counterparts. Such generalizations can be techni cally challenging because of the noncommutative nature of the quantum setting.

<table><tr><td rowspan=1 colspan=2>Online Portfolio Selection</td><td rowspan=1 colspan=2>Online Learning of Quantum States</td></tr><tr><td rowspan=1 colspan=1>Algorithm</td><td rowspan=1 colspan=1>Regret Rate</td><td rowspan=1 colspan=1>Algorithm</td><td rowspan=1 colspan=1>Regret Rate</td></tr><tr><td rowspan=1 colspan=1>Universal Portfolios(Cover, 1991)</td><td rowspan=1 colspan=1>O(dlog T)</td><td rowspan=1 colspan=1>Exponential Weights(Zimmert et al., 2022)</td><td rowspan=1 colspan=1> $O ( d ^ { 2 } \log T )$ </td></tr><tr><td rowspan=1 colspan=1>VB-FTRL(Jézéquel et al., 2026)</td><td rowspan=1 colspan=1> $O ( d \log ( d + T ) )$ </td><td rowspan=1 colspan=1>Q-VB-FTRL(Tseng et al., 2025)</td><td rowspan=1 colspan=1> $O ( d ^ { 2 } \log ( d + T ) )$ </td></tr><tr><td rowspan=1 colspan=1>PAE+DONS(Mhammedi and Rakhlin, 2022)</td><td rowspan=1 colspan=1> $O ( d ^ { 2 } \log ^ { 5 } T )$ </td><td rowspan=1 colspan=2>No known quantum counterpart.</td></tr><tr><td rowspan=1 colspan=1>BISONS(Zimmert et al., 2022)</td><td rowspan=1 colspan=1> $O ( d ^ { 2 } \log ^ { 2 } T )$ </td><td rowspan=1 colspan=1>SCHRÖDINGER&#x27;s BISONS(Zimmert et al., 2022)</td><td rowspan=1 colspan=1> $O ( d ^ { 3 } \log ^ { 2 } T )$ </td></tr><tr><td rowspan=1 colspan=1>Barrier subgradient method(Nesterov, 2011)</td><td rowspan=3 colspan=1> $\widetilde { O } ( \sqrt { d T } )$ </td><td rowspan=1 colspan=1>Q-LB-OMD(Tsai et al., 2023a)</td><td rowspan=3 colspan=1> $\widetilde { O } ( \sqrt { d T } )$ </td></tr><tr><td rowspan=1 colspan=1>Soft-Bayes(Orseau et al., 2017)</td><td rowspan=1 colspan=1>Q-Soft-Bayes(Lin et al., 2021b)</td></tr><tr><td rowspan=1 colspan=1>LB-FTRL(Tsai et al., 2023c)</td><td rowspan=1 colspan=1>Q-LB-FTRL(Tsai et al., 2023b)</td></tr></table>

Table 1: Existing algorithms for online portfolio selection and online learning of quantum states with logarithmic loss (OLQS), together with their regret upper bounds. Rows group related online portfolio selection algorithms and their quantum counterparts. Here, d denotes the number of assets in online portfolio selection and the dimension of density matrices in OLQS, and T denotes the time horizon. The notation Oe suppresses polylogarithmic factors in d and T . LB-FTRL and Q-LB-FTRL in the last row use linearized losses.

We observe an interesting separation in Table 1:

• Q-LB-OMD (Tsai et al., 2023a), Q-Soft-Bayes (Lin et al., 2021b), and Q-LB-FTRL (Tsai et al., 2023b) inherit the same regret rates as their classical counterparts.

• Exponential Weights (Zimmert et al., 2022), Q-VB-FTRL (Tseng et al., 2025), and SCHRODINGER’s BISONS (Zimmert et al., 2022) suffer an additional factor of d in their regret bounds compared with their classical counterparts.

Both phenomena can be explained. Q-LB-OMD, Q-Soft-Bayes, and Q-LB-FTRL were developed so that their algorithms and regret bounds reduce to those of their classical counterparts when all matrices involved are diagonal. In contrast, the analyses ofExponential Weights, Q-VB-FTRL, and SCHRÖDINGER’s BISONS use a vectorization argument that treats a d × d density matrix as a vector of dimension $\Theta ( d ^ { 2 } )$ , leading to the additional factor of d in their regret upper bounds.

The aforementioned observation naturally leads to the following question:

Is the additional factor of d in the regret bound for Exponential Weights in OLQS intrinsic to the problem, or is it merely an artifact of the analysis?

The answer to this question also has implications for the significance of the Q-VB-FTRL algorithm. No explicit implementation of Exponential Weights for OLQS is currently available in the literature. Q-VB-FTRL, on the other hand, admits an implementation with O(Tpoly(d)) per-round runtime while achieving an $O ( d ^ { 2 } \log ( d + T ) )$ regret bound. Therefore, if the additional factor of d in these regret bounds could be removed, then the significance of Q-VB-FTRL as a computationally efficient algorithm with nearoptimal dependence on the dimension would also be called into question.

In this paper, we establish the following regret lower bound for OLQS.

Theorem 1.1. Suppose that $T \gg d ^ { 2 } \log d .$ . Then no algorithm for OLQS can achieve a worst-case regret bound of O(d<sup>γ</sup> logT ) for any $\gamma < 2$

This result establishes that the regret rates of both Exponential Weights and Q-VB-FTRL have the correct polynomial dependence on the dimension.

Our proof strategy exploits the connection between OLQS and quantum state tomography. We first develop an online learning approach to quantum state tomography, reducing quantum state tomography to OLQS via an online-to-batch conversion, and derive a high-probability upper bound on the estimation error in terms of the regret of the OLQS algorithm used. Then, existing high-probability lower bounds on the estimation error from the quantum information theory literature (Acharya et al., 2025; Chen et al., 2023; Haah et al., 2017; Lowe and Nayak, 2025) immediately imply that the regret of any OLQS algorithm must be sufficiently large.

Our result also yields two applications.

1. As a byproduct of our proof strategy, we develop a novel algorithm for quantum state tomography that is sample-optimal up to polylogarithmic factors in the sample size.

2. Generalizing the connection between confidence sequences for estimating the mean of a random vector and online portfolio selection, we develop an algorithm that constructs confidence sequences for the mean of a random matrix. Theorem 1.1 establishes that the regret bound for Exponential Weights has optimal polynomial dependence on the dimension, motivating our use of this bound in the confidence-sequence construction in Section 4.

Notation The d ×d identity matrix is denoted by $I _ { d }$ . The transpose and conjugate transpose of a matrix M are denoted by $M ^ { \mathsf { T } }$ and $M ^ { * }$ , respectively. The trace norm of a matrix M is denoted by $\| M \| _ { 1 }$ , as it coincides with the Schatten 1-norm. The 1-norm ofa vector v is denoted by $\| \nu \| _ { 1 }$ . For any d-dimensional probability vectors $p$ and $q ,$ , their Kullback-Leibler divergence is denoted by $D _ { \mathrm { K L } } ( p , q )$ and defined as

$$
D _ { \mathrm { K L } } ( p , q ) : = \sum _ { i = 1 } ^ { d } p [ i ] \log \frac { p [ i ] } { q [ i ] } ,
$$

where $p [ i ]$ and $q [ i ]$ denote the i-th entries of $p$ and $q ,$ respectively.

## 2 Preliminaries on Quantum State Tomography and Quantum Measurements

Quantum state tomography refers to the task of estimating an unknown density matrix from measurements and measurement outcomes, and is essential for building reliable quantum computing devices. Throughout this paper, we will only consider the following non-adaptive, single-copy formulation of quantum state tomography, a commonly studied setting because of its relative ease of implementation in practice: Suppose we have n independent copies of an unknown density matrix $\rho ^ { \natural } \in { \mathcal { D } } _ { d }$ . We measure each of the copies independently using a fixed POVM<sup>2</sup> $\mathcal { M } = \{ M _ { \iota } \vert \iota \in \mathcal { I } \} \subset \mathbb { C } ^ { d \times d }$ , for some finite index set $\mathcal { I } _ { : }$ , and observe the measurement outcomes $\eta _ { 1 } , . . . , \eta _ { n }$ as independent and identically distributed (i.i.d.) random variables taking values in $\mathcal { I }$ . Then, according to the Born rule (1), the negative log-likelihood function is given by

$$
\begin{array} { r } { L ( \rho ; \iota ) : = - \log \mathrm { t r } \big ( M _ { \iota } \rho \big ) , \quad \mathrm { f o r ~ a l l } \iota \in \mathcal { I } , } \end{array}
$$

and the maximum-likelihood estimate is given by minimizing the empirical average of the negative loglikelihood functions over the data (measurement outcomes). In this paper, however, we will consider an online learning approach to quantum state tomography, rather than the maximum-likelihood one.

We will measure the performance of any estimate ${ \hat { \rho } } _ { n } ,$ based on n copies of the unknown density matrix, using the trace-norm error $\left\| { \hat { \rho } } _ { n } - \rho ^ { \natural } \right\| _ { 1 }$ . Under this performance measure, a sample complexity lower bound is already known.

Theorem 2.1 (Haah et al. (2017); Lowe and Nayak (2025)). For $0 < \varepsilon \le 1 / 2$ , any quantum state tomography algorithm usingnonadaptive, single-copymeasurements thatestimates everyd×d densitymatrix to tracenorm error at most ε with probability at least 2/3 requires $n = \Omega ( d ^ { 3 } / \varepsilon ^ { 2 } )$ copies of the unknown state.

The sample complexity is highly dependent on the POVM used. In this paper, we will consider the POVM induced by complex projective 4-designs, whose POVMnorm constant will be useful in our analysis.

Definition 2.2 (Complex projective t-design). Let $S _ { d } : = \{ x \in \mathbb { C } ^ { d } \mid x ^ { * } x = 1 \}$ , and let d $\mathcal { S } _ { d }$ denote its normalized unitarily invariant probability measure. For any positive integer t, a complex projective t-design of dimension d and cardinality m (Ambainis and Emerson, 2007; Matthews et al., 2009; Scott, 2006) consists of a finite set $\{ x _ { 1 } , \ldots , x _ { m } \} \subset { \mathcal { S } } _ { d }$ such that

$$
\sum _ { i = 1 } ^ { m } \frac { 1 } { m } ( x _ { i } x _ { i } ^ { * } ) ^ { \otimes t } = \int _ { x \in \mathcal { S } _ { d } } ( x x ^ { * } ) ^ { \otimes t } \mathrm { d } \mathcal { S } _ { d } ,
$$

where ⊗ denotes the tensor product. The induced POVM is given by the set of operators

$$
\{ ( d / m ) x _ { 1 } x _ { 1 } ^ { * } , . . . , ( d / m ) x _ { m } x _ { m } ^ { * } \} .
$$

Definition 2.3 (POVM norm constant (Kueng et al., 2016; Matthews et al., 2009)). For any POVM $\mathcal { M } =$ $\{ M _ { \iota } \mid \iota \in \mathcal { S } \}$ , where $\mathcal { I }$ is a finite index set, the associated POVMnorm constant is defined as

$$
\lambda _ { \mathcal { M } } = \operatorname* { i n f } _ { \rho \neq \sigma \atop \rho , \sigma \in \mathcal { D } _ { d } } \frac { \sum _ { i \in \mathcal { I } } \left| \operatorname { t r } \left( M _ { i } \rho \right) - \operatorname { t r } \left( M _ { i } \sigma \right) \right| } { \left\| \rho - \sigma \right\| _ { 1 } } .
$$

Theorem 2.4 (Matthews et al. (2009, (23))). The POVM norm constant of the POVM induced by complex projective 4-designs is at least $1 / ( 3 \sqrt { d } )$

Remark 2.5. Matthews et al. (2009) proved that the largest POVM norm constant is $\Theta ( 1 / { \sqrt { d } } )$ . The POVM induced by complex projective 4-designs is hence optimal in this sense. Indeed, Matthews et al. (2009) showed that the uniform POVM also attains this optimal rate. Nevertheless, defining the uniform POVM formally requires the general measure-theoretic definition of POVM, as it has an infinite outcome space. For the purpose of this paper, it suffices to consider the POVM induced by complex projective 4-designs, which has a finite outcome space.

Define the risk function as the expected negative log-likelihood:

$$
R ( \rho ) : = \mathsf E _ { \eta _ { 1 } } \left[ L ( \rho ; \eta _ { 1 } ) \right] .
$$

The excess risk of any estimate $\hat { \rho } _ { n }$ —not necessarily the maximum-likelihood one—is then given by

$$
\mathrm { E R } ( \hat { \rho } _ { n } ) : = R ( \hat { \rho } _ { n } ) - \operatorname* { m i n } _ { \rho \in \mathcal { D } _ { d } } R ( \rho ) .
$$

We will use the following inequality in analysing the sample complexity of the proposed online learningbased quantum state tomography algorithm.

Proposition 2.6. In any non-adaptive, single-copy quantum state tomography experiment with a fixed POVM M satisfying $\lambda _ { \mathcal { M } } > 0$ , any estimate $\hat { \rho } _ { n }$ must satisfy

$$
\left\| \hat { \rho } _ { n } - \rho ^ { \natural } \right\| _ { 1 } \leq \frac { \sqrt { 2 \mathrm { E R } ( \hat { \rho } _ { n } ) } } { \lambda _ { \mathcal { M } } } .
$$

In particular, if the POVM is the one induced by complex projective 4-designs, then

$$
\left\| \hat { \rho } _ { n } - \rho ^ { \natural } \right\| _ { 1 } \leq 3 \sqrt { 2 d \mathrm { E R } ( \hat { \rho } _ { n } ) } .
$$

Proposition 2.6 should be familiar to experts in related fields. Nevertheless, we do not find an appropriate reference, so we provide a proof for completeness.

Proof of Proposition 2.6. A direct calculation gives

$$
\mathrm { E R } ( \hat { \rho } _ { n } ) = D _ { \mathrm { K L } } ( p _ { \rho ^ { \sharp } } , p _ { \hat { \rho } _ { n } } ) ,
$$

where $p _ { \rho }$ denotes the probability distribution of the measurement outcome when the measured density matrix is $\rho .$ That is, $p _ { \rho } : = ( \mathrm { t r } \big ( M _ { \iota } \rho \big ) ) _ { \iota \in \mathcal { I } }$ , which is a probability vector in $\mathbb { R } ^ { | \mathcal { I } | }$ . By Pinsker’s inequality and the definition of POVM norm constants (Definition 2.3), we write

$$
\begin{array} { r } { \lambda _ { \mathcal { M } } \Big \| \hat { \rho } _ { n } - \rho ^ { \sharp } \Big \| _ { 1 } \leq \Big \| p _ { \hat { \rho } _ { n } } - p _ { \rho ^ { \sharp } } \Big \| _ { 1 } \leq \sqrt { 2 \mathrm { E R } ( \hat { \rho } _ { n } ) } , } \end{array}
$$

which proves the proposition.

## 3 Regret Lower Bound for Online Learning of Quantum States and a Quantum State Tomography Algorithm

This section proposes an online learning approach to quantum state tomography. According to Proposi tion 2.6, the estimation error is controlled by the excess risk; hence, it appears sufficient to implement an OLQS algorithm followed by an online-to-batch conversion, which translates the algorithm’s regret guarantee into a high-probability excess risk guarantee. Nevertheless, it is easily checked that the logarithmic loss $f _ { t } ( \rho ) : = - \log \mathrm { t r } \bigl ( A _ { t } \rho \bigr )$ in OLQS can be arbitrarily large near the boundary of ${ \mathcal { D } } _ { d } ,$ , so standard onlineto-batch conversion based on the Azuma-Hoeffding inequality does not directly apply. Moreover, the logarithmic loss is 1-exp-concave, so we expect to achieve a fast rate of ${ \tilde { O } } ( 1 / T )$ for the excess risk (Koren and Levy, 2015; Mehta, 2017). Recall the definition of exp-concavity:

Definition 3.1 (Exp-concavity (Hazan et al., 2007; Kivinen and Warmuth, 1999)). A function $f$ is called η-exp-concave if the function $\mathrm { e } ^ { - \eta f }$ is concave.

We show below that the online-to-batch conversion scheme for exp-concave losses proposed by Mehta (2017) suffices. Although this online-to-batch conversion scheme requires the loss functions to be bounded, this issue can be solved if we shrink the density matrices slightly toward the normalized identity matrix $I _ { d } / d ,$ called the maximally mixed state in quantum information theory. While Foster et al. (2018) also use this shrinkage in a computationally more expensive online-to-batch conversion based on the work of Mehta (2017), we show that it is unnecessary to use their scheme.

For any $\rho \in \mathcal { D } _ { d }$ , define the shrunk density matrix

$$
\tilde { \rho } : = \left( 1 - \frac { 1 } { n } \right) \rho + \frac { I _ { d } } { n d } .
$$

Also define the set of shrunk density matrices:

$$
\tilde { \mathcal { D } } _ { d } : = \{ \tilde { \rho } \mid \rho \in \mathcal { D } _ { d } \} .
$$

Consider the following online learning-based scheme for quantum state tomography.

• Let there be n rounds in the OLQS game.

• Let the matrices $A _ { t }$ in the OLQS be set as $M _ { \eta _ { t } } / \operatorname { t r } \bigl ( M _ { \eta _ { t } } \bigr )$ , where $\eta _ { t }$ are measurement outcomes as defined in the previous section. Note that trace normalization does not affect the regret value since the loss functions are logarithmic.

• Implement an online learning algorithm in this OLQS game, and obtain iterates $\rho _ { 1 } , \rho _ { 2 } , . . . , \rho _ { n }$

• Output the empirical average of shrunk iterates $\hat { \rho } _ { n } : = ( \tilde { \rho } _ { 1 } + \tilde { \rho } _ { 2 } + \cdots + \tilde { \rho } _ { n } ) / n .$

Lemma 3.2. Suppose that $n \geq 2 .$ . For any $\rho \in \mathcal { D } _ { d }$ such that $\mathbf { \nabla } \cdot \mathbf { r } \bigl ( A _ { t } \rho \bigr ) > 0$ for all t, it holds that

$$
f _ { t } ( \tilde { \rho } ) - f _ { t } ( \rho ) \leq \frac { 1 } { n - 1 } , \quad f o r a l l 1 \leq t \leq n ,
$$

and

$$
\operatorname* { m i n } _ { \tilde { \rho } \in \tilde { \mathcal { D } } _ { d } } R ( \tilde { \rho } ) - \operatorname* { m i n } _ { \rho \in \mathcal { D } _ { d } } R ( \rho ) \leq \frac { 1 } { n - 1 } .
$$

Proof. Let $\mu = 1 / n$ . We write

$$
f _ { t } ( \tilde { \rho } ) - f _ { t } ( \rho ) = \log \left( \frac { \mathrm { t r } \big ( A _ { t } \rho \big ) } { ( 1 - \mu ) \mathrm { t r } \big ( A _ { t } \rho \big ) + \frac { \mu } { d } \mathrm { t r } ( A _ { t } ) } \right) \leq \log \left( \frac { 1 } { 1 - \mu } \right) .
$$

By the inequality above, we write

$$
\operatorname* { m i n } _ { \tilde { \rho } \in \tilde { \mathcal { D } } _ { d } } R ( \tilde { \rho } ) \leq R ( \rho ) + \log \left( \frac { 1 } { 1 - \mu } \right) .
$$

The lemma then follows from the fact that

$$
\log \left( { \frac { 1 } { 1 - \mu } } \right) = \log \left( { \frac { n } { n - 1 } } \right) = \log \left( 1 + { \frac { 1 } { n - 1 } } \right) \leq { \frac { 1 } { n - 1 } } .
$$

Lemma 3.2 implies that

$$
\widetilde { \mathrm { R e g r e t } } _ { n } : = \sum _ { t = 1 } ^ { n } f _ { t } ( \tilde { \rho } _ { t } ) - \operatorname* { m i n } _ { \tilde { \rho } \in \tilde { \mathcal { D } } _ { d } } \sum _ { t = 1 } ^ { n } f _ { t } ( \tilde { \rho } ) \leq \mathrm { R e g r e t } _ { n } ( \tilde { \rho } ) + \frac { n } { n - 1 } .
$$

Define $\widetilde { \mathrm { R e g r e t } } _ { n } ^ { + } : = \operatorname* { m a x } \{ \widetilde { \mathrm { R e g r e t } } _ { n } , 0 \}$ . Note that the loss functions $f _ { t }$ are bounded by log(nd) and 1-expconcave over $\tilde { D } _ { d }$ . Applying the online-to-batch conversion scheme of Mehta (2017, Corollary 2) and Lemma 3.2, we obtain that for any $n \geq 3$

$$
\begin{array} { r l } & { R \bigg ( \frac { \bar { \rho } _ { 1 } + \cdots + \bar { \rho } _ { n } } { n } \bigg ) - \underset { \rho \in \mathcal { G } _ { d } } { \operatorname* { m i n } } R ( \rho ) } \\ & { \quad = R \bigg ( \frac { \bar { \rho } _ { 1 } + \cdots + \bar { \rho } _ { n } } { n } \bigg ) - \underset { \bar { \rho } \in \bar { \mathcal { G } } _ { d } } { \operatorname* { m i n } } R ( \bar { \rho } ) + \underset { \bar { \rho } \in \bar { \mathcal { Q } } _ { d } } { \operatorname* { m i n } } R ( \bar { \rho } ) - \underset { \rho \in \mathcal { Q } _ { d } } { \operatorname* { m i n } } R ( \rho ) } \\ & { \quad \leq \frac { \widetilde { \mathrm { R e g r e t } } _ { n } ^ { + } } { n } + 4 \frac { \sqrt { \widetilde { \mathrm { R e g r e t } } _ { n } ^ { + } } } { n } \sqrt { \bigg ( 1 + \log { ( n d ) } \bigg ) \log \bigg ( \frac { 4 \log { n } } { \delta } \bigg ) } + } \\ & { \quad \quad ~ 1 6 \Big ( 1 + \log { ( n d ) } \bigg ) \frac { \log \bigg ( \frac { 4 \log { n } } { \delta } \bigg ) } { n } + \frac { 1 } { n - 1 } . } \end{array}
$$

with probability at least $1 - \delta$

Setting $\delta = 1 / 3$ and plugging in the regret rate bounds for Exponential Weights and Q-VB-FTRL, we immediately obtain the following high-probability estimation error bound, which is nearly optimal according to Theorem 2.1.

Theorem 3.3. Suppose that $n \gg d . \mathrm { ~ } I f n \ge 3 ,$ , the proposed online learning-based scheme for quantum state tomography with either Exponential Weights or Q-VB-FTRL, initialized at $\rho _ { 1 } = I _ { d } / d ,$ yields an estimate $\hat { \rho } _ { n }$ that satisfies

$$
\left\| \hat { \rho } _ { n } - \rho ^ { \natural } \right\| _ { 1 } = \tilde { O } \left( \sqrt { \frac { d ^ { 3 } \log n } { n } } \right)
$$

with probability at least 2/3. Here, the notation $\tilde { O }$ suppresses polylogarithmic factors in log(n).

In addition, the resulting algorithm is sample-optimal up to polylogarithmic factors in n.

Now, we are ready to prove Theorem 1.1.

Proof of Theorem 1.1. Suppose that there exists an OLQS algorithm with a regret rate of $O ( d ^ { \gamma } \log T )$ for some $\gamma < 2$ . Then, adopting this algorithm with the arguments above, there exists a quantum state tomography algorithm that achieves an estimation error of $\tilde { O } ( \sqrt { \left( d ^ { 1 + \gamma } \log n \right) / n } )$ with probability at least 2/3, which violates the known sample complexity lower bound in Theorem 2.1. □

Remark 3.4. Note that Theorem 1.1 does not determine the minimax regret rate of OLQS. In particular, it rules out regret bounds of the form $O ( d ^ { \gamma }$ log T ) for $\gamma < 2$ , but does not establish a matching $\Omega ( d ^ { 2 }$ logT ) lower bound for the regret bound achieved by Exponential Weights. Since OLQS generalizes online portfolio selection, whose minimax regret rate is $\Theta ( d \log T )$ , the minimax regret of OLQS is also Ω(d logT ). However, this lower bound, even when combined with Theorem 1.1, is insufficient to establish that the minimax regret rate of OLQS is $\Theta ( d ^ { 2 } \log T )$ . We leave the determination of the exact minimax regret rate of OLQS to future work.

Consider a real version of the OLQS game, in which the matrices $A _ { t }$ are real and the action set $\mathcal { D } _ { d }$ is replaced by $\mathcal { D } _ { d } ^ { \mathbb { R } }$ , the set of real symmetric positive semidefinite matrices with unit trace.

Corollary 3.5. Theorem 1.1 continues to hold for the real version of the OLQS game.

The corollary will be used in the next section to justify our approach to constructing confidence sequences for estimating the mean of a real random matrix. Its proof, deferred to Appendix $\mathrm { A , }$ shows that any algorithm for the real version in dimension 2d induces an algorithm for the original OLQS game in dimension d whose worst-case regret is no greater than that of the real algorithm.

## 4 Confidence Sequence for Random Matrix Mean Estimation

Let $( \xi _ { t } ) _ { t \in \mathbb { N } }$ be a stochastic process satisfying $\mathsf { E } [ \xi _ { t } \mid \xi _ { 1 } , \ldots , \xi _ { t - 1 } ] = \mu ^ { \mathsf { \Delta } }$ for every $t \in \mathbb { N } ,$ , where $\mu ^ { \natural }$ is an unknown parameter. We say that a sequence of sets $( \mathcal { C } _ { t } ) _ { t \in \mathbb { N } }$ is a (1−α)-confidence sequence for $\mu ^ { \natural }$ if each $\mathscr { C } _ { t }$ depends only on $\xi _ { 1 } , . . . , \xi _ { t }$ and

$$
\operatorname { P } \left( { \mu { \mathrm { ? } } } \in { \mathcal { C } } _ { t } { \mathrm { ~ f o r ~ a l l ~ } } t \in \mathbb { N } \right) \geq 1 - \alpha .
$$

Thus, confidence sequences may be viewed as an anytime-valid extension of confidence sets. The reader is referred to, for example, the works of Howard et al. (2021), Ramdas et al. (2022), and Ramdas and Wang (2025) for historical reviews and expositions of related concepts.

We are particularly interested in a relatively recent line of research that constructs confidence sequences by associating the task with online portfolio selection (Orabona and Jun, 2024; Ryu and Wornell, 2024; Ryu and Bhatt, 2024). Orabona and Jun (2024) and Ryu and Bhatt (2024) considered only the case where $\xi _ { t }$ are scalar-valued random variables, as their constructions explicitly rely on special properties of Universal Portfolios in the 2-dimensional case. Ryu and Wornell (2024) considered the case where $\xi _ { t }$ are vector-valued random variables by reducing the construction of confidence sequences to solving general online portfolio selection problems.

We aim to extend this line of research to the case where $\xi _ { t }$ are matrix-valued random variables. Although confidence sequences for the mean of a random matrix can be constructed by vectorizing the observations and applying the algorithm of Ryu and Wornell (2024), the numerical result in Section 5 shows that the resulting confidence sequences yield larger estimated confidence-set volumes than our construction.

Suppose that $\xi _ { t }$ are $\mathcal { D } _ { d }$ -valued random variables. Define the following “matrix division” between $\xi _ { t }$ and $\mu { : }$

$$
\xi _ { t } ( \mu ) : = \mu ^ { - \dag / 2 } \xi _ { t } \mu ^ { - \dag / 2 } , \quad \mathrm { f o r ~ a l l } \ \mu \in \mathcal { D } _ { d } ,
$$

where $\mu ^ { - \dagger / 2 }$ denotes the square root of the Moore-Penrose generalized inverse of the matrix $\mu .$

For every $\mu \in \mathcal { D } _ { d }$ , consider the OLQS game in which Nature announces $A _ { t } = \xi _ { t } ( \mu )$ in round t, and denote the output of Physicist in round t by $\rho _ { t } ( \mu )$ . Consider the stochastic process $( W _ { t } ( \mu ) ) _ { t \geq 0 }$ such that $W _ { 0 } = 1$ and

$$
W _ { t } ( \mu ) = W _ { t - 1 } ( \mu ) \operatorname { t r } \left( \xi _ { t } ( \mu ) \rho _ { t } ( \mu ) \right) , \quad \mathrm { f o r ~ a l l ~ } t \in \mathbb { N } ,
$$

which is non-negative because both $\xi _ { t } ( \mu )$ and $\rho _ { t } ( \mu )$ are positive semi-definite. Since $\rho _ { t + 1 } ( \mu ^ { \natural } )$ is completely determined by $\xi _ { 1 } , . . . , \xi _ { t }$ , we have

$$
\begin{array} { r l } & { \mathsf { E } \left[ W _ { t + 1 } ( \mu ^ { \sharp } ) \Big | \xi _ { 1 } , \ldots , \xi _ { t } \right] = W _ { t } ( \mu ^ { \sharp } ) \operatorname { t r } \left( \left( \mu ^ { \sharp } \right) ^ { - \dagger / 2 } \mu ^ { \sharp } \left( \mu ^ { \sharp } \right) ^ { - \dagger / 2 } \rho _ { t + 1 } ( \mu ^ { \sharp } ) \right) } \\ & { \qquad \le W _ { t } ( \mu ^ { \sharp } ) . } \end{array}
$$

Hence, $( W _ { t } ( \mu ^ { \natural } ) ) _ { t \geq 0 }$ is a non-negative supermartingale.

Theorem 4.1 (Ville’s inequality (Ville, 1939)). Let $( \eta _ { t } ) _ { t \geq 0 }$ be a non-negative supermartingale. Then, it holds that

$$
\mathsf { P } \left( \mathsf { s u p } \eta _ { t } \geq c \right) \leq \frac { \mathsf { E } \left[ \eta _ { 0 } \right] } { c } , \quad f o r a n y c > 0 .
$$

Applying Ville’s inequality to the stochastic process $( W _ { t } ( \mu ^ { \natural } ) ) _ { t \geq 0 }$ , we immediately get the following theorem.

Theorem 4.2. The sequence $( \mathcal { C } _ { t } ) _ { t \in \mathbb { N } }$ defined as

$$
\mathcal { C } _ { t } = \left\{ \left. \mu \in \mathcal { D } _ { d } \right. \bigg | \ : W _ { t } ( \mu ) < \frac { 1 } { \alpha } \right\} ,\tag{2}
$$

is a $( 1 - \alpha )$ -confidence sequence for $\mu ^ { \natural }$

Remark 4.3. Suppose that the observations $\xi _ { t }$ are almost surely diagonal. Then $\mu ^ { \natural }$ is also diagonal, so we may restrict the candidate means to diagonal density matrices. In this case, our construction specializes to that of Ryu and Wornell (2024).

Define the regret for the sequence of observations $( \xi _ { t } ( { \mu } ) ) _ { t \in \mathbb { N } }$ as

$$
\mathrm { R e g r e t } _ { t } ( \mu ) : = \sum _ { \tau = 1 } ^ { t } \left( - \log \mathrm { t r } \big ( \xi _ { \tau } ( \mu ) \rho _ { \tau } ( \mu ) \big ) \right) - L _ { t } ^ { \star } ( \mu ) , \quad L _ { t } ^ { \star } ( \mu ) : = \operatorname* { m i n } _ { \rho \in \mathcal { D } _ { d } } \sum _ { \tau = 1 } ^ { t } \left( - \log \mathrm { t r } \big ( \xi _ { \tau } ( \mu ) \rho \big ) \right) .
$$

Then, the confidence set $\mathscr { C } _ { t } \left( 2 \right)$ can be expressed equivalently as

$$
\mathcal { C } _ { t } = \left\{ \mu \in \mathcal { D } _ { d } \big | \mathrm { R e g r e t } _ { t } ( \mu ) > \log \alpha - L _ { t } ^ { \star } ( \mu ) \right\} .
$$

The quantity $L _ { t } ^ { \star } ( \mu )$ can be efficiently approximated using existing algorithms (Li and Cevher, 2019; Lin et al., 2021a; Tsai et al., 2024). Computing Regret (µ), however, requires implementing an OLQS algorithm. Given Theorem 1.1, Exponential Weights and Q-VB-FTRL are natural choices, but both are computationally expensive.

Corollary 4.4. The sequence of sets $( \overline { { \mathcal { C } } } _ { t } ) _ { t \in \mathbb { N } }$ given by

$$
\overline { { \mathscr { C } } } _ { t } : = \left\{ \mu \in \mathscr { D } _ { d } \bigg | \overline { { \mathrm { R e g r e t } _ { t } ( \mu ) } } > \log \alpha - L _ { t } ^ { \star } ( \mu ) \right\} ,\tag{3}
$$

where $\overline { { \mathrm { R e g r e t } _ { t } ( \mu ) } }$ is any upper bound on Regret (µ) realized by any OLQS algorithm, i $\varsigma a ( 1 { - } \alpha )$ -confidence sequence $f o r \mu ^ { \natural }$

The construction does not require implementing any OLQS algorithm but requires a tight regret bound. Exploiting exp-concavity of logarithmic losses, the result of Hazan et al. (2007, Theorem 7) implies that the regret of Exponential Weights is no larger than $( d ^ { 2 } - 1 ) ( 1 + \log ( T + 1 ) )$ in the OLQS game. Below, we provide a sharper regret bound whose proof, deferred to Appendix B, essentially follows the strategy of Blum and Kalai (1999, Theorem 1).

Theorem 4.5. LetX bea boundedclosedconvexsetwith non-emptyrelativeinteriorin afinite-dimensional Euclidean space. Letp bethedimension ofitsaffinehull. Consideran onlineconvexoptimization problem ofT rounds, wheretheaction setis X andthelossfunctions areη-exp-concaveforsomeη $> 0$ . Then, Exponential Weights with learning rate η and the uniform prior achieves a regret no greater than (1/η)log $\binom { T + p } { p }$

Combining Corollary 4.4 and Theorem 4.5, we obtain the following explicit constructions of confidence sequences for both the complex and real cases. The proof is deferred to Appendix C.

Corollary 4.6. The sequence ofsets $( { \overline { { \boldsymbol { \mathscr { C } } _ { t } } } } ) _ { t \in \mathbb { N } }$ defined in Corollary 4.4 with $\overline { { \mathrm { R e g r e t } _ { t } ( \mu ) } } = \log \binom { t + p } { p }$ , where $p = d ^ { 2 } - 1$ , is a $( 1 - \alpha )$ -confidence sequence.

Ifit is known that the random matrices $\xi _ { t }$ are almost surely real, then we can set $p = d ^ { 2 } / 2 + d / 2 - 1$ and replace $\mathcal { D } _ { d }$ with $\mathcal { D } _ { d } ^ { \mathbb { R } }$ in (3) and the definition of $\cdot _ { L _ { t } ^ { \star } ( \mu ) }$

Remark 4.7. Our construction is not limited to the case where $\xi _ { t }$ are random density matrices. More generally, suppose that $\xi _ { t }$ are Hermitian matrices satisfying $A \leq \xi _ { t } \leq B$ for known Hermitian matrices A and B. If $A \neq B$ , define $c : = \operatorname { t r } ( B - A )$ and the affine transformation

$$
\begin{array} { r l } & { \xi _ { t } \longmapsto \widetilde { \xi } _ { t } : = \left( \begin{array} { c c } { \xi _ { t } - A } & { 0 } \\ { 0 } & { 1 - \operatorname { t r } \left( \frac { \xi _ { t } - A } { c } \right) } \end{array} \right) \in { \mathcal { D } } _ { d + 1 } . } \end{array}
$$

A confidence sequence for $\mathsf { E } \left[ \widetilde { \xi } _ { t } \right]$ then immediately yields one for $\mathsf { E } \left[ \boldsymbol { \xi } _ { t } \right]$ through the inverse affine transformation.

![](images/39882e983fdc6c9d40f4a725e09e0600baa4ce8f5d1cce563a743aa77902daec.jpg)  
Figure 1: Comparison of the mean relative Hilbert–Schmidt volumes of confidence sequences for covariance matrix estimation. OLQS denotes our construction following the second claim of Corollary 4.6. See Section 5 and Appendix D for details.

## 5 Numerical Experiment

We conduct a numerical experiment to compare our confidence sequence construction with existing methods for covariance matrix estimation. Let $( \nu _ { t } ) _ { t \in \mathbb { N } }$ be i.i.d. mean-zero random vectors in $\mathbb { R } ^ { d }$ with covariance matrix Σ. Our goal is to construct a confidence sequence for Σ. This problem fits our framework by setting $\xi _ { t } = \nu _ { t } \nu _ { t } ^ { \top }$ and $\mu ^ { \natural } = \Sigma$ . Figure 1 compares our construction with those of Ryu and Wornell (2024, Eqs. (15)–(16)), implemented using matrix vectorization and the regret bound of Cover and Ordentlich (1996), Howard et al. (2021, Eq. (4.10)), and Wang and Ramdas (2025, Theorem 4.2). The latter two methods are based on matrix concentration inequalities rather than online-learning regret bounds.

We set $d = 8$ and consider $T = 2 0 0$ observations, with confidence level $1 - \alpha = 0 . 9 5$ . The random vectors $( \nu _ { t } ) _ { t \in \mathbb { N } }$ are generated such that $\| \nu _ { t } \| _ { 2 } = 1$ , the covariance matrix Σ lies in ${ \mathcal { D } } _ { d } ^ { \mathbb { R } } ,$ and the support of the distribution of $\xi _ { t } = \nu _ { t } \nu _ { t } ^ { \top }$ contains non-commuting matrices.

The performance is measured by the relative Hilbert–Schmidt volume, defined as the ratio between the Hilbert–Schmidt volumes of a confidence set and $\mathcal { D } _ { d } ^ { \mathbb { R } }$ (Zyczkowski and Sommers<sup>˙</sup> , 2003). For each method, we intersect its confidence sets obtained up to time t and restrict the resulting set to ${ \mathcal { D } } _ { d } ^ { \mathbb { R } } ,$ to ensure that the resulting sets are nested over time. We approximate these volumes by Monte Carlo sampling and average the estimates over 1000 independent repetitions of the experiment. The error bars in Figure 1 represent pointwise 95% Student-t confidence intervals for the mean relative volume.

Figure 1 shows that our construction yields substantially smaller estimated relative volumes than the three competing implementations in this experiment. In particular, the construction based on matrix vectorization and the regret bound of Cover and Ordentlich (1996) produces considerably larger esti mated confidence-set volumes, suggesting that this adaptation ofRyu and Wornell (2024) is less effective in the considered setting. For Howard et al. (2021) and Wang and Ramdas (2025), we follow the parameter choices specified in their papers without further tuning. In comparison, our construction requires no tuning parameter beyond the confidence level.

Further details of the numerical experiment can be found in Appendix D.

## 6 Concluding Remarks

We have shown that no algorithm for OLQS can achieve a regret bound ofO(d<sup>γ</sup> logT) for any γ < 2, establishing the near-optimality of Exponential Weights and the Q-VB-FTRL algorithm of Tseng et al. (2025) in terms of the polynomial dependence of their regret rates on d. Our proof strategy also yields a novel online learning-based quantum state tomography algorithm that is sample-optimal up to polylogarithmic factors in the sample size. Furthermore, viewing OLQS as a quantum generalization of online portfolio selection, we have extended the connection between online portfolio selection and confidence sequence construction to matrix-valued parameters. The regret bound for Exponential Weights yields confidence sequences whose candidate membership can be assessed by convex optimization without implementing an OLQS algorithm. Figure 1 illustrates the empirical competitiveness of our construction in covariance matrix estimation.

Several research problems follow. First, our result characterizes only the optimal polynomial dependence of regret rate on the dimension and does not determine the minimax-optimal regret rate, as discussed in Remark 3.4. Deriving the minimax-optimal regret rate is a natural direction for future work. In particular, deriving sharper regret upper bounds, ideally with optimal constants, could lead to tighter confidence sequences via Corollary 4.4. Second, although our confidence sequence appears competitive empirically, we have not established any convergence rate in terms of the sizes of its confidence sets, nor have we proved its optimality under any criterion. Addressing these questions would help clarify the theoretical advantages of online learning-based confidence sequence construction. Finally, the online learning-based approach may extend to more general parameter spaces by considering losses of the form −logL(x), where L is a linear functional positive on the parameter space. Developing a unified theory for such constructions would be an interesting direction for future work.

## AI Disclosure

The strategies for deriving the regret lower bounds and constructing the confidence sequences were developed by the authors. Nevertheless, the original online-to-batch scheme adopted by the authors was that of Foster et al. (2018); GPT-6 Pro noted that the scheme of Mehta (2017) suffices. The regret bound originally derived by the authors for Theorem 4.5 was (d/η)log(T + 1) + (1/η), looser than the current bound. GPT-6 Pro noted that the analysis of Blum and Kalai (1999) can be extended to exp-concave online convex optimization, yielding an improved regret bound. GPT-6 Pro generated the Julia code for the numerical experiments, while the experimental settings and parameter choices were determined by the authors. Appendix D was initially generated by GPT-6 after multiple rounds of conversations and then modified by the authors. The authors also used GPT-6 to correct grammatical errors in the draft.

## Acknowledgments

This work was supported by the National Science and Technology Council of Taiwan under Grant Nos. NSTC 112-2628-E-002-019-MY3 and NSTC 115-2221-E-002-208, by the research project “Geometry of

Quantum Learning and Optimization” at National Taiwan University under Grant No. NTU-CC-115L893706, and by the Young Theoretical Scientist Award ofthe Mathematics Division ofthe National Center for Theoretical Sciences of Taiwan.

## References

Scott Aaronson, Xinyi Chen, Elad Hazan, Satyen Kale, and Ashwin Nayak. Online learning of quantum states. In Adv. Neural Information ProcessingSystems 31, 2018.

Scott Aaronson, Xinyi Chen, Elad Hazan, Satyen Kale, and Ashwin Nayak. Online learning of quantum states. J. Stat. Mech.: Theory Exp., page 124019, 2019.

Jayadev Acharya, Abhilash Dharmavarapu, Yuhan Liu, and Nengkun Yu. Pauli measurements are not optimal for single-copy tomography. In STOC’25: Proc. 57th Annu. ACM Symp. Theory of Computing, pages 719–729, 2025.

Andris Ambainis and Joseph Emerson. Quantum t-designs: t-wise independence in the quantum world. In 22ndAnnu. IEEEConf. Computational Complexity (CCC’07), 2007.

Akshay Bansal, Ian George, Soumik Ghosh, Jamie Sikora, and Alice Zheng. Online learning of a panoply of quantum objects. Quantum Mach. Intell., 7(78), 2025.

Avrim Blum and Adam Kalai. Universal portfolios with and without transaction costs. Mach. Learn., 35: 193–205, 1999.

Sitan Chen, Brice Huang, Jerry Li, Allen Liu, and Mark Selke. When does adaptivity help for quantum state learning? In IEEE 64th Annu. Symp. Foundations of Computer Science (FOCS), 2023.

Xinyi Chen, Elad Hazan, Tongyang Li, Zhou Lu, Xinzhao Wang, and Rui Yang. Adaptive online learning of quantum states. Quantum, 8:1471, 2024.

Thomas M. Cover. Universal portfolios. Math. Financ., 1(1):1–29, 1991.

Thomas M. Cover and Erick Ordentlich. Universal portfolios with side information. IEEE Trans. Inf. Theory, 42(2):348–363, 1996.

Steven T. Flammia, David Gross, Yi-Kai Liu, and Jens Eisert. Quantum tomography via compressed sensing: Error bounds, sample complexity and efficient estimators. NewJ. Phys., 14, 2012.

Dylan J. Foster, Satyen Kale, Haipeng Luo, Mehryar Mohri, and Karthik Sridharan. Logistic regression: The importance of being improper. In Proc. 31stConf. LearningTheory, 2018.

Michel X. Goemans and David P. Williamson. Approximation algorithms for MAX-3-CUT and other problems via complex semidefinite programming. J. Comput. Syst. Sci., 68:442–470, 2004.

David Gross, Yi-Kai Liu, Steven T. Flammia, Stephen Becker, and Jens Eisert. Quantum state tomography via compressed sensing. Phys. Rev. Lett., 105, 2010.

Jeongwan Haah, Aram W. Harrow, Zhengfeng Ji, Xiaodi Wu, and Nengkun Yu. Sample-optimal tomography of quantum states. IEEE Trans. Inf. Theory, 63(9):5628–5641, 2017.

Elad Hazan. Introduction to online convex optimization. 2023. arXiv:1909.05207v3.

Elad Hazan, Amit Agarwal, and Satyen Kale. Logarithmic regret algorithms for online convex optimiza tion. Mach. Learn., 69:169–192, 2007.

Steven R. Howard, Aaditya Ramdas, Jon McAuliffe, and Jasjeet Sekhon. Time-uniform, nonparametric, nonasymptotic confidence sequences. Ann. Stat., 49(2):1055–1080, 2021.

Z. Hradil. Quantum-state estimation. Phys. Rev. A, 55(3), 1997.

Rémi Jézéquel, Dmitrii Ostrovskii, and Pierre Gaillard. Efficient and near-optimal online portfolio selection. Math. Oper. Res., 51(2):1350–1384, 2026.

Adam Kalai and Santosh Vempala. Efficient algorithms for universal portfolios. J. Mach. Learn. Res., 3: 423–440, 2002.

Jyrki Kivinen and Manfred K. Warmuth. Averaging expert predictions. In Paul Fischer and Hans Ulrich Simon, editors, EuroCOLT’99, pages 153–167, Berlin, 1999. Springer.

Tomer Koren and Kfir Levy. Fast rates for exp-concave empirical risk minimization. In Adv. Neural Information ProcessingSystems 28 (NIPS 2015), 2015.

Richard Kueng, Huangjun Zhu, and David Gross. Distinguishing quantum states using Clifford orbits. 2016. arXiv:1609.08595.

Yen-Huan Li and Volkan Cevher. Convergence of the exponentiated gradient method with Armijo line search. J. Optim. TheoryAppl., 181(2):588–607, 2019.

Chien-Ming Lin, Hao-Chung Cheng, and Yen-Huan Li. Maximum-likelihood quantum state tomography by Cover’s method with non-asymptotic analysis. In 25th Annu. Conf. Quantum Information Processing, 2021a. arXiv:2110.00747.

Chien-Ming Lin, Yu-Ming Hsu, and Yen-Huan Li. An online algorithm for maximum-likelihood quantum state tomography. In 24th Annu. Conf. Quantum Information Processing, 2021b. arXiv:2012.15498.

Angus Lowe and Ashwin Nayak. Lower bounds for learning quantum states with single-copy measurements. ACM Trans. Comput. Theory, 17(7):1–42, 2025.

Haipeng Luo, Chen-Yu Wei, and Kai Zheng. Efficient online portfolio with logarithmic regret. In Adv. Neural Information Processing Systems 31, 2018.

William Matthews, Stephanie Wehner, and Andreas Winter. Distinguishability of quantum states under restricted families of measurements with an application to quantum data hiding. Commun. Math. Phys., 291:813–843, 2009.

Nishant Mehta. Fast rates with high probability in exp-concave statistical learning. In Proc. 20th Int. Conf. Artificial Intelligence and Statistics, 2017.

Zakaria Mhammedi and Alexander Rakhlin. Damped Online Newton Step for Portfolio Selection. In Proc. 35th Conf. Learning Theory, pages 5561–5595, 2022.

Yurii Nesterov. Barrier subgradient method. Math. Program., Ser. B, 127:31–56, 2011.

Francesco Orabona. Online Learning: A Modern Introduction Using Convex Optimization. 2026. arXiv:1912.13213v10.

Francesco Orabona and Kwang-Sung Jun. Tight concentrations and confidence sequences from the re gret of universal portfolio. IEEE Trans. Inf. Theory, 70(1):436–455, 2024.

Erik Ordentlich and Thomas M. Cover. The cost of achieving the best portfolio in hindsight. Math. Oper. Res., 23(4):960–982, 1998.

Laurent Orseau, Tor Lattimore, and Shane Legg. Soft-Bayes: Prod for mixtures of experts with log-loss. In Proc. 28th Int. Conf. Algorithmic Learning Theory, pages 372–399, 2017.

V Al Osipov, H-J Sommers, and K Zyczkowski. Random Bures mixed states and the distribution of their<sup>˙</sup> purity. J. Phys. A: Math. Theor., page 055302, 2010.

Matteo Paris and Jaroslav Reháˇcek, editors. <sup>ˇ</sup> Quantum State Estimation. Springer, Berlin, 2004.

Aaditya Ramdas and Ruodu Wang. Hypothesis Testing with E-Values. 2025. arXiv:2410.23614v6.

Aaditya Ramdas, Johannes Ruf, Martin Larsson, and Wouter M. Koolen. Admissible anytime-valid sequential inference must rely on nonnegative martingales. 2022. arXiv:2009.03167v3.

C. A. Riofrío, D. Gross, S. T. Flammia, T. Monz, D. Nigg, R. Blatt, and J. Eisert. Experimental quantum compressed sensing for a seven-qubit system. Nature Commun., 2017.

J. Jon Ryu and Alankrita Bhatt. On confidence sequences for bounded random processes via universal gambling strategies. IEEETrans. Inf. Theory, 70(10):7143–7161, 2024.

J. Jon Ryu and Gregory W. Wornell. Gambling-based confidence sequences for bounded random vectors. In Proc. 41st Int. Conf. Machine Learning, 2024.

A. J. Scott. Tight informationally complete quantum measurements. J. Phys. A: Math. Gen., 39:13507– 13530, 2006.

Chung-En Tsai, Hao-Chung Cheng, and Yen-Huan Li. Online self-concordant and relatively smooth minimization, with applications to online portfolio selection and learning quantum states. In Proc. 34th Int. Conf. Algorithmic Learning Theory, 2023a.

Chung-En Tsai, Hao-Chung Cheng, and Yen-Huan Li. Faster stochastic first-order method for maximumlikelihood quantum state tomography. In 26th Conf. Quantum Information Processing, 2023b.

Chung-En Tsai, Ying-Ting Lin, and Yen-Huan Li. Data-dependent bounds for online portfolio selection without Lipschitzness and smoothness. In Adv. Neural Information Processing Systems 36 (NeurIPS 2023), 2023c.

Chung-En Tsai, Hao-Chung Cheng, and Yen-Huan Li. Fast minimization of expected logarithmic loss via stochastic dual averaging. In Proc. 27th Int. Conf. Artificial Intelligence and Statistics, pages 2908–2916, 2024.

Wei-Fu Tseng, Kai-Chun Chen, Zi-Hong Xiao, and Yen-Huan Li. Online learning of quantum states with logarithmic loss via VB-FTRL. In Proc. 36th Int. Conf. Algorithmic Learning Theory, pages 1285–1312, 2025.

Dirk van der Hoeven, Tim van Erven, and Wojciech Kotłowski. The many faces of exponential weights in online learning. In Proc. 31stConf. LearningTheory, 2018.

Jean Ville. Étude critique de la notion de collectif. Gauthier-Villars, Paris, 1939.

V. Vovk and C. Watkins. Universal portfolio selection. In Proc. 11th Annu. Conf. Computational Learning Theory, pages 12–23, 1998.

Hongjian Wang and Aaditya Ramdas. Sharp matrix empirical Bernstein inequalities. In Adv. Neural Information ProcessingSystems 38 Main Conference (NeurIPS 2025), 2025.

Feidiao Yang, Jiaqing Jiang, Jialin Zhang, and Xiaoming Sun. Revisiting online quantum state learning. In Proc. AAAI Conf. Artificial Intelligence, 2020.

Julian Zimmert, Naman Agarwal, and Satyen Kale. Pushing the efficiency-regret Pareto frontier for online learning of portfolios and quantum states. In Proc. 35th Conf. Learning Theory, pages 182–226, 2022.

Karol Zyczkowski and Hans-Jürgen Sommers. Hilbert-Schmidt volume of the set of mixed quantum<sup>˙</sup> states. 2003. arXiv:quant-ph/0302197v3.

## A Proof of Corollary 3.5

We prove the corollary by showing that any algorithm for the real version ofthe OLQS game in dimension 2d can be transformed into an algorithm for the original OLQS game in dimension d, and that both algorithms incur the same regret values on the corresponding data sequences. Then, any regret lower bound of the original OLQS game also implies a regret lower bound of its real version.

Let $d \in \mathbb { N } .$ Denote by $\mathcal { D } _ { 2 d } ^ { \mathbb { R } }$ the set of $( 2 d ) \times ( 2 d )$ real symmetric positive semidefinite matrices with unit trace. For any matrix $M \in \mathbb { C } ^ { d \times d }$ , define the realification map as

$$
\Phi ( M ) : = { \binom { \operatorname { R e } M } { \operatorname { I m } M } } \quad { \mathrm { R e } } M { \mathrm { ) } } ,
$$

where ReM and ImM denote the real and imaginary parts of the matrix M, respectively.

Lemma A.1 (Goemans and Williamson (2004, Section 3)). A Hermitian complex matrix M is positive semidefinite ifand only $i f \Phi ( M )$ is positive semidefinite.

For any matrix

$$
\rho ^ { \mathbb { R } } = \left( \begin{array} { l l } { P } & { Q } \\ { Q ^ { \top } } & { R } \end{array} \right) \in \mathcal { D } _ { 2 d } ^ { \mathbb { R } } ,
$$

where P and R are symmetric matrices in $\mathbb { R } ^ { d \times d }$ , define its “complexification” as

$$
\Psi ( \rho ^ { \mathbb { R } } ) : = ( P + R ) - i \left( Q - Q ^ { \top } \right) ,
$$

where i denotes $\sqrt { - 1 }$

Lemma A.2. The matrix $\Psi ( \rho ^ { \mathbb { R } } )$ is a density matrix.

Proof. It is easily checked that the matrix $\Psi ( \rho ^ { \mathbb { R } } )$ is Hermitian with unit trace. Note that for every $\nu \in \mathbb { C } ^ { d }$

$$
\nu ^ { * } \Psi ( \rho ^ { \mathbb { R } } ) \nu = \nu ^ { * } ( P + R ) \nu - i \nu ^ { * } ( Q - Q ^ { \top } ) \nu = \tilde { \nu } ^ { * } \rho ^ { \mathbb { R } } \tilde { \nu } ,
$$

where

$$
\tilde { \nu } : = \left( \begin{array} { c } { { \nu } } \\ { { - i \nu } } \end{array} \right) .
$$

Since $\rho ^ { \mathbb { R } }$ is positive semidefinite, $\tilde { \nu } ^ { * } \rho ^ { \mathbb { R } } \tilde { \nu } \geq 0$ . Hence, the matrix $\Psi ( \rho ^ { \mathbb { R } } )$ is also positive semidefinite.

Lemma A.3. For any Hermitian matrix $M \in \mathbb { C } ^ { d \times d }$ and $\rho ^ { \mathbb { R } } \in \mathcal { D } _ { 2 d } ^ { \mathbb { R } }$ , it holds that

$$
\operatorname { t r } \bigl ( M \Psi ( \rho ^ { \mathbb { R } } ) \bigr ) = \operatorname { t r } \bigl ( \Phi ( M ) \rho ^ { \mathbb { R } } \bigr ) .
$$

Proof. Use the notations above. We write

$$
\begin{array} { r l } & { \operatorname { t r } \bigl ( M \Psi ( \rho ^ { \mathbb { R } } ) \bigr ) = \operatorname { t r } \bigl ( ( \operatorname { R e } M ) ( P + R ) + ( \operatorname { I m } M ) ( Q - Q ^ { \top } ) \bigr ) + } \\ & { \qquad i \operatorname { t r } \bigl ( ( \operatorname { I m } M ) ( P + R ) - ( \operatorname { R e } M ) ( Q - Q ^ { \top } ) \bigr ) } \\ & { = \operatorname { t r } \bigl ( ( \operatorname { R e } M ) ( P + R ) + ( \operatorname { I m } M ) ( Q - Q ^ { \top } ) \bigr ) } \\ & { = \operatorname { t r } \bigl ( \Phi ( M ) \rho ^ { \mathbb { R } } \bigr ) . } \end{array}
$$

The second equality follows from the fact that both M and $\Psi ( \rho ^ { \mathbb { R } } )$ are Hermitian, so the imaginary part of the trace must vanish; the third equality follows from direct calculation. □

Lemma A.4. The mappingΨ is onto.

Proof. For any $\rho \in \mathcal { D } _ { d }$ , define

$$
\rho ^ { \mathbb { R } } : = \frac { 1 } { 2 } \left( \operatorname { R e } \rho \mathrm { \quad } - \operatorname { I m } \rho \right) .
$$

Note that $\rho ^ { \mathbb { R } }$ is symmetric, positive semidefinite by Lemma $\mathrm { A . 1 }$ , and unit-trace, so it lies in $\mathcal { D } _ { 2 d } ^ { \mathbb { R } }$ . In addition, it is easily checked that $\Psi ( \rho ^ { \mathbb { R } } ) = \rho$ □

For any algorithm $\mathcal { A } ^ { \mathbb { R } }$ for Physicist in the real version of the OLQS game in dimension $2 d ,$ consider the following algorithm for Physicist in the original OLQS game in dimension d: At each round t, obtain the next prediction $\rho _ { t } ^ { \mathbb { R } }$ from $\mathcal { \dot { A } } ^ { \mathbb { R } }$ and announce $\rho _ { t } = \Psi ( \rho _ { t } ^ { \mathbb { R } } )$ . After Nature reveals $A _ { t }$ , feed $\Phi ( A _ { t } ) \ \mathrm { t o } \ \mathcal { A } ^ { \mathbb { R } }$ Note that this is a valid input: The matrix $\Phi ( A _ { t } )$ is positive semidefinite by Lemma A.1; applying Lemma A.1 again, we write

$$
I _ { 2 d } - \Phi ( A _ { t } ) = \Phi ( I - A _ { t } ) \geq 0 ,
$$

which shows that all eigenvalues of $\Phi ( A _ { t } )$ are at most 1.

Lemma A.2 ensures that $\rho _ { t }$ is a density matrix, so the induced algorithm for the original OLQS game is valid. In addition, Lemma A.3 ensures that the cumulative losses in the original OLQS game and its real version are the same; Lemma A.4 ensures that the minimum cumulative losses in hindsight in the original OLQS game and its real version over $\mathcal { D } _ { d }$ and $\mathcal { D } _ { 2 d } ^ { \mathbb { R } }$ , respectively, are also the same. Hence, the two algorithms incur the same regret on corresponding data sequences $A _ { 1 } , . . . , A _ { T }$ and $\Phi ( A _ { 1 } ) , . . . , \Phi ( A _ { T } )$ . A regret rate of $O ( d ^ { \gamma } \log T )$ for any $\gamma < 2$ for the real version of OLQS would therefore imply a regret rate of $O ( d ^ { \gamma } \log T )$ for the original OLQS game, contradicting Theorem 1.1.

## B ProofofTheorem 4.5

Let us specify the algorithm and define the necessary notations first. Denote by $f _ { t }$ the loss function at round t. For any Lebesgue-measurable ${ \mathcal { S } } \subseteq { \mathcal { X } }$ , let vol(S) denote its Lebesgue measure on the affine hull of X , identified as $\mathbb { R } ^ { p }$ . Exponential Weights with the uniform prior iterates as follows.

• Let $\pi _ { 1 }$ be the density of the uniform distribution over $\mathcal { X } ;$ that is, $\pi _ { 1 } ( x ) = 1 / \mathrm { v } \mathbf { 0 } \mathrm { l } ( \mathcal { X } )$ for all $x \in \mathcal X$

• For every $t \in \mathbb { N } ,$ output $\boldsymbol { x } _ { t } = \mathsf { E } _ { X \sim \pi _ { t } } \left[ X \right]$ and update

$$
\pi _ { t + 1 } ( x ) \propto \pi _ { t } ( x ) \mathrm { e } ^ { - \eta f _ { t } ( x ) } , \quad \mathrm { f o r \ a l l \ } x \in \mathcal { X } .
$$

Then, we prove the theorem. Define $f _ { 1 : T } ( x ) : = f _ { 1 } ( x ) + \dots + f _ { T } ( x )$ . The iteration rule of Exponential Weights gives the telescoping identity

$$
\prod _ { t = 1 } ^ { T } \int _ { \mathcal X } \mathbf { e } ^ { - \eta f _ { t } ( x ) } \pi _ { t } ( x ) \mathrm { d } x = \int _ { \mathcal X } \prod _ { t = 1 } ^ { T } \mathbf { e } ^ { - \eta f _ { t } ( x ) } \pi _ { 1 } ( x ) \mathrm { d } x = \int _ { \mathcal X } \mathbf { e } ^ { - \eta f _ { 1 : T } ( x ) } \pi _ { 1 } ( x ) \mathrm { d } x .
$$

By the η-exp-concavity of $f _ { t } ,$ we write

$$
\mathbf { e } ^ { - \eta \sum _ { t = 1 } ^ { T } f _ { t } ( x _ { t } ) } = \prod _ { t = 1 } ^ { T } \mathbf { e } ^ { - \eta f _ { t } ( x _ { t } ) } \geq \prod _ { t = 1 } ^ { T } \int _ { \mathcal { X } } \mathbf { e } ^ { - \eta f _ { t } ( x ) } \pi _ { t } ( x ) \mathrm { d } x .
$$

Let $U$ be a random variable with the uniform probability distribution over $\mathcal { X }$ . Then, we write

$$
\int _ { \mathcal { X } } \mathrm { e } ^ { - \eta f _ { 1 : T } ( x ) } \pi _ { 1 } ( x ) \mathrm { d } x = \mathsf { E } \left[ \mathrm { e } ^ { - \eta f _ { 1 : T } ( U ) } \right] .
$$

We next derive a lower bound on the expectation. Let $x _ { \star }$ be any comparator in $\mathcal { X }$ . For any $0 \leq \alpha \leq 1$ define

$$
\mathcal { X } _ { \alpha } : = \{ \alpha x _ { \star } + ( 1 - \alpha ) x \mid x \in \mathcal { X } \} .
$$

Note that vol $( \mathcal { X } _ { \alpha } ) = ( 1 - \alpha ) ^ { p } \mathbf { v _ { 0 } } \mathbf { l } ( \mathcal { X } )$ . In addition, for any $\boldsymbol { x } _ { \alpha } \in \mathcal { X } _ { \alpha }$ , by the η-exp-concavity of $f _ { t } ,$ , we have

$$
\mathrm { e } ^ { - \eta f _ { 1 : T } ( x _ { \alpha } ) } \geq \alpha ^ { T } \mathrm { e } ^ { - \eta f _ { 1 : T } ( x _ { \star } ) } .
$$

Hence, we obtain

$$
\begin{array} { r } { \mathsf { P } \left( \mathbf { e } ^ { - \eta f _ { 1 : T } ( U ) } \geq \alpha ^ { T } \mathbf { e } ^ { - \eta f _ { 1 : T } ( x _ { \star } ) } \right) \geq ( 1 - \alpha ) ^ { p } . } \end{array}
$$

Since $0 \leq \mathbf { e } ^ { - \eta f _ { 1 : T } ( U ) }$ almost surely, we write

$$
\begin{array} { r l } & { \mathsf { E } \left[ \mathrm { e } ^ { - \eta f _ { 1 : T } ( U ) } \right] \ge \displaystyle \int _ { 0 } ^ { \mathrm { e } ^ { - \eta f _ { 1 : T } ( x _ { \star } ) } } \mathsf { P } \left( \mathrm { e } ^ { - \eta f _ { 1 : T } ( U ) } \ge t ^ { \prime } \right) \mathrm { d } t ^ { \prime } } \\ & { \qquad = \mathrm { e } ^ { - \eta f _ { 1 : T } ( x _ { \star } ) } \displaystyle \int _ { 0 } ^ { 1 } \mathsf { P } \left( \frac { \mathrm { e } ^ { - \eta f _ { 1 : T } ( U ) } } { \mathrm { e } ^ { - \eta f _ { 1 : T } ( x _ { \star } ) } } \ge t \right) \mathrm { d } t } \\ & { \qquad \ge \mathrm { e } ^ { - \eta f _ { 1 : T } ( x _ { \star } ) } \displaystyle \int _ { 0 } ^ { 1 } ( 1 - t ^ { 1 / T } ) ^ { p } \mathrm { d } t . } \end{array}
$$

The integral in the last line is an instance of the Euler integral and equals $1 / { \binom { T + p } { p } }$ . Combining the results above, we obtain

$$
\mathbf { e } ^ { - \eta \sum _ { t = 1 } ^ { T } f _ { t } ( x _ { t } ) } \geq \frac { \mathbf { e } ^ { - \eta f _ { 1 : T } ( x _ { \star } ) } } { \binom { T + p } { p } } .
$$

Taking logarithms yields

$$
\sum _ { t = 1 } ^ { T } f _ { t } ( x _ { t } ) \leq f _ { 1 : T } ( x _ { \star } ) + \frac { 1 } { \eta } \log \binom { T + p } { p } .
$$

Since $\boldsymbol { x } _ { \star } \in \mathcal X$ is arbitrary, taking the infimum over $x _ { \star }$ proves the theorem.

## C Proof of Corollary 4.6

The corollary is obtained by applying the regret bound for Exponential Weights in Theorem 4.5 to Corol lary 4.4. We regard the matrix spaces as vector spaces over R and apply Theorem 4.5 in the affine hull of each decision set.

It is easily checked that $\mathcal { D } _ { d }$ is a bounded closed convex set with non-empty relative interior and that the logarithmic loss is 1-exp-concave. To specify a density matrix, it suffices to specify its diagonal entries, which needs $( d - 1 )$ real numbers given the trace constraint, as well as its strictly upper-triangular entries, which needs $( d ^ { 2 } - d )$ real numbers. Hence, the dimension ofthe affine hull of $\mathcal { D } _ { d }$ is at most $d ^ { 2 } - 1$ The first claim of the corollary then follows from Theorem 4.5.

For the real case, it is easily checked that $\mathcal { D } _ { d } ^ { \mathbb { R } }$ is a bounded closed convex set with non-empty relative interior, the dimension of whose affine hull is at most $d ^ { 2 } / 2 + d / 2 - 1$ . Since the conditional mean $\mu ^ { \natural }$ must be real when the observations $\xi _ { t }$ are real almost surely, we have

$$
\mathsf { P } \left( \mu ^ { \sharp } \in \overline { { \mathcal { C } } } _ { t } \mathrm { f o r } \mathrm { a l l } t \right) = \mathsf { P } \left( \mu ^ { \sharp } \in \mathbb { D } _ { d } ^ { \mathbb { R } } \cap \overline { { \mathcal { C } } } _ { t } \mathrm { f o r } \mathrm { a l l } t \right) ,
$$

showing that we can replace $\mathcal { D } _ { d }$ with $\mathbb { D } _ { d } ^ { \mathbb { R } }$ . This restricts the parameter space to real density matrices and allows us to use the smaller affine dimension in the regret bound. Finally, note that if the minimum cumulative loss $L _ { T } ^ { \star }$ is attained at a (complex) density matrix ${ \boldsymbol { \rho } } _ { T } ^ { \star }$ , then it is also attained at the point

$$
\tilde { \rho } _ { T } ^ { \star } : = \frac { \overline { { \rho _ { T } ^ { \star } } } + \rho _ { T } ^ { \star } } { 2 } ,
$$

where $\overline { { { \rho } _ { T } ^ { \star } } }$ denotes the conjugate of ${ \boldsymbol { \rho } } _ { T } ^ { \star }$ , because for any real symmetric positive semidefinite matrix M, we have

$$
\mathrm { t r } \big ( { \cal M } \rho _ { T } ^ { \star } \big ) = \mathrm { t r } \big ( { \cal M } \tilde { \rho } _ { T } ^ { \star } \big ) .
$$

Hence, we can use the regret bound of Exponential Weights over $\mathcal { D } _ { d } ^ { \mathbb { R } }$ and replace $\mathcal { D } _ { d }$ with $\mathcal { D } _ { d } ^ { \mathbb { R } }$ in the definition of $L _ { T } ^ { \star } ( \mu )$

## D Details of Numerical Experiment

For Ryu and Wornell (2024, Eqs. (15)–(16)), we map each real density matrix to a vector with 35 coordinates in [0,1], which is then converted into a probability vector with 36 coordinates using their construction. Since evaluating Universal Portfolios is computationally expensive in high dimensions, we instead apply Corollary 4.4 using the regret bound of Cover and Ordentlich (1996, Theorem 2) for Universal Portfolios with a Dirichlet- $( 1 / 2 , . . . , 1 / 2 )$ prior. For both our method and this adaptation ofRyu and Wornell (2024), we approximately minimize the cumulative logarithmic loss over the respective decision sets using entropic mirror descent with Armijo line search (Li and Cevher, 2019).

For Howard et al. (2021, Eq. (4.10)), we set $\eta = 2$ and $s = 1 . 4$ , as in their Eq. (3.4), and $b = 1$ , since the generated random vectors satisfy ∥ $\nu _ { t } \| _ { 2 } = 1$ . Their bound controls the estimation error $\| \widehat { \Sigma } _ { t } - \Sigma \| _ { \mathrm { o p } } ,$ but its radius depends on $\| \Sigma \| _ { \mathbf { o p } } ,$ where Σ is the unknown covariance matrix. To obtain a computable confidence set, we evaluate the bound at each candidate covariance matrix $\mu ,$ retaining µ if

$$
\| \widehat { \Sigma } _ { t } - \mu \| _ { \mathrm { o p } } < r _ { t } ( \mu ) ,
$$

where $r _ { t } ( \mu )$ is the radius in their Eq. (4.10) with Σ replaced by $\mu .$ This preserves the time-uniform coverage guarantee because, when $\mu = \Sigma$ , the membership condition coincides with the original concentra tion inequality.

For Wang and Ramdas (2025, Theorem 4.2), we construct the confidence sequence using a weighted average of the observed matrices. We choose the weights following the authors’ recommendation after their Corollary 4.3 for the case where the total number of observations is not known in advance, using their empirical matrix variance estimate in $\operatorname { E q . }$ (27). We also use their estimate $\widehat { \xi } _ { t }$ of the next observation in Eq. (32), with $\bar { V } _ { 0 } = I _ { d } / 4$ and $\widehat { \xi } _ { 1 } = 0$ as initial values. To obtain a two-sided operator-norm confidence sequence, we apply their one-sided bound to both tails with error probability $\alpha / 2$ each.

Let $u _ { j } = e _ { j }$ for $1 \leq j \leq d$ , where $e _ { j }$ denotes the j-th element in the standard basis in $\mathbb { R } ^ { d }$ , and define

$$
u _ { d + 1 } = \frac { ( 1 , . . . , d ) ^ { \top } } { \sqrt { \textstyle \sum _ { k = 1 } ^ { d } k ^ { 2 } } } , \qquad u _ { d + 2 } = \frac { \left( ( - 1 ) ^ { k } ( k + \frac { 1 } { 2 } ) \right) _ { k = 1 } ^ { d } } { \sqrt { \textstyle \sum _ { k = 1 } ^ { d } ( k + \frac { 1 } { 2 } ) ^ { 2 } } } .
$$

Each random vector is generated as $\nu _ { t } = \epsilon _ { t } u _ { t _ { t } }$ , where $( \epsilon _ { t } ) _ { t \in \mathbb { N } }$ is a sequence of i.i.d. Rademacher random variables and $\left( \boldsymbol { \iota } _ { t } \right) _ { t \in \mathbb { N } }$ is an independent sequence of i.i.d. random variables with

$$
p _ { j } : = \mathsf { P } ( \iota _ { t } = j ) \propto \mathrm { e } ^ { - 0 . 3 8 ( j - 1 ) } , \quad \mathrm { f o r } \mathrm { a l l } 1 \leq j \leq d + 2 .
$$

Consequently, E $\mathbf { \sigma } _ { \nu _ { t } } ] = 0 , \| \nu _ { t } \| _ { 2 } = 1$ , and

$$
\Sigma = \mathrm { d i a g } ( p _ { 1 } , . . . , p _ { d } ) + p _ { d + 1 } u _ { d + 1 } u _ { d + 1 } ^ { \mathsf { T } } + p _ { d + 2 } u _ { d + 2 } u _ { d + 2 } ^ { \mathsf { T } } .
$$

We estimate the relative Hilbert–Schmidt volume by sampling candidate density matrices of the form $G G ^ { \mathsf { T } } / \operatorname { t r } \left( G G ^ { \mathsf { T } } \right)$ , where $G \in \mathbb { R } ^ { d \times ( d + 1 ) }$ has independent standard Gaussian entries. This construction generates matrices uniformly distributed with respect to the Hilbert–Schmidt volume measure on real density matrices (Osipov et al., 2010, Section 4, Eq. (25)).