This work has been accepted for publication in the 2026 IEEE 2nd International Conference on Quantum Artificial Intelligence (QAI). © 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works.

# Measurement-Efficient Differentiable Quantum Architecture Search for Combinatorial Optimization

Lukas Theißinger

University of Bonn

Germany

lukas.theissinger@uni-bonn.de

Thore Gerlach European Space Agency Netherlands

Christian Bauckhage   
Lamarr Institute   
Germany

Abstract—Differentiable quantum architecture search (DQAS) is a promising framework for the automated design of quantum circuits, particularly for variational quantum optimization algorithms. However, its practical deployment on quantum hardware is limited by the large number of circuit measurements required during optimization, making hardware execution costly.

In this work, we show that for a broad class of combinatorial optimization problems and commonly used rotational gate parameterizations, the measurement cost of DQAS can be significantly reduced without changing the optimization objective. We derive the proposed measurement reduction scheme theoretically and validate it experimentally on 3-SAT and MaxCut benchmark problems. Our approach reduces the requested gradient measurement cost by about 39 to 41% while introducing only negligible classical post-processing overhead, lowering the practical cost of executing DQAS on quantum hardware.

Index Terms—Differentiable quantum architecture search, variational quantum algorithms, diagonal Hamiltonians, 3-SAT, MaxCut, combinatorial optimization, gradient estimation, measurement cost, parameter-shift rule.

## I. INTRODUCTION

Quantum variational algorithms are a leading approach for using near-term quantum devices on optimization and simulation tasks. In combinatorial optimization, many canonical problems, including 3-SAT, MaxCut, graph coloring, Maxk-SAT, graph partitioning and weighted constraint satisfaction, are NP-hard and arise naturally in scheduling, logistics, network design and resource allocation. A key reason these problems are attractive for quantum optimization is that many of them admit compact encodings as Ising or quadratic unconstrained binary optimization (QUBO) Hamiltonians, whose ground states correspond to optimal or feasible high-quality solutions of the original discrete problem [1], [2]. These encodings make combinatorial optimization a natural setting for variational quantum algorithms such as the variational quantum eigensolver and the quantum approximate optimization algorithm [3]–[6].

A variational quantum algorithm optimizes the parameters of a parameterized quantum circuit so as to minimize the expectation value of a problem Hamiltonian. In the standard setting, the circuit architecture is chosen before training and remains fixed; only the continuous gate parameters are updated. This separation between architecture design and parameter optimization is convenient, but it makes performance highly dependent on a manually selected ansatz. Moreover, gradientbased training on quantum hardware is itself measurementintensive: analytic parameter-shift rules typically estimate a single rotation-gate derivative from two shifted circuit evaluations [7], [8].

Differentiable Quantum Architecture Search (DQAS) addresses the fixed-architecture limitation by jointly optimizing continuous gate parameters and discrete circuit choices [9]. DQAS introduces a differentiable relaxation of gate selection, allowing architecture weights and gate parameters to be trained together with gradient-based optimization. This places DQAS in a broader line of work that treats ansatz structure as part of the optimization problem, including structure optimization for parameterized quantum circuits, adaptive ansatz construction in ADAPT-VQE and qubit-ADAPT-VQE and differentiable quantum architecture search methods such as QuantumDARTS [10]–[13]. These methods differ in how they choose circuit structure: ADAPT-style methods greedily grow the ansatz from an operator pool using gradient information, whereas DQASstyle methods relax discrete architectural choices so that they can be optimized directly.

Despite this flexibility, architecture-adaptive variational algorithms remain expensive to execute on quantum hardware. Each optimization step requires repeated circuit executions to estimate expectation values and gradients, while the associated classical optimization is typically cheap by comparison. Measurement overhead is therefore a central practical bottleneck, especially for methods that evaluate many candidate gates or operators. This issue has been studied explicitly in ADAPT-VQE, where gradient-measurement cost can dominate and specialized measurement strategies are needed to make adaptive ansatz construction practical [14].

In this work, we show that the gradient-measurement overhead of DQAS can be reduced substantially for an important class of combinatorial optimization Hamiltonians. We consider Hamiltonians that are diagonal in the computational basis, including Ising and QUBO formulations used for MaxCut, 3- SAT, Max-k-SAT, graph partitioning and weighted constraint satisfaction. For DQAS circuits built from parameterized Pauli-string rotation gates, we derive a measurement-saving gradient estimator that replaces a naively applied two-shift parameter-shift estimate with a one-shift estimate whenever the suffix of the circuit has the required diagonal or Clifford structure. The method leaves the optimization objective unchanged: it reduces the number of quantum circuit measurements needed to estimate gradients, while adding only cheap classical post-processing. We prove the correctness of the estimator and evaluate it on 3-SAT and MaxCut benchmarks, where it reduces the requested gradient-measurement cost by approximately 39-41% with negligible classical overhead.<sup>1</sup>

## II. RELATED WORK

A central limitation of variational quantum algorithms is that their performance depends strongly on the chosen ansatz. Several lines of work therefore treat circuit structure as an optimization variable rather than a fixed design choice. Structure optimization methods optimize the placement and ordering of parameterized gates [10], while quantum architecture search frameworks search over candidate circuit layouts for variational algorithms [15], [16]. Differentiable Quantum Architecture Search (DQAS) makes this search gradient-based by relaxing discrete gate choices into differentiable architecture parameters, so that circuit parameters and architecture weights can be optimized jointly [9]. QuantumDARTS follows a similar philosophy for tasks including MaxCut, ground-state estimation and classification [13]. Adaptive ansatz methods such as ADAPT-VQE and qubit-ADAPT-VQE instead grow circuits greedily from operator pools using gradient information [11], [12]. Recent work has studied how to reduce the resulting ADAPT-VQE gradient-measurement cost by exploiting Pauli structure and shared measurement information [14]. Our work targets the corresponding cost inside differentiable architecture search: the shifted circuit evaluations needed to compute gradients for many candidate gates and parameters.

Analytic gradient estimation for variational circuits is commonly based on parameter-shift rules, which express derivatives of expectation values as combinations of expectation values evaluated at shifted parameter values [7], [8]. For gates generated by Pauli strings, the standard rule uses two shifted evaluations per parameter. Later work generalized these rules to broader generator spectra and developed single-component gradient rules for more general variational settings [17], [18]. These results provide general-purpose gradient estimators, but they do not exploit the specific structure that arises in DQAS for diagonal combinatorial Hamiltonians. Our method uses this additional structure: when the cost Hamiltonian is diagonal in the computational basis and the circuit suffix has the required diagonal or Clifford form, one of the two shifted evaluations can be replaced by classical post-processing of a single shifted measurement distribution.

Measurement cost is a major bottleneck in variational quantum algorithms, and many prior approaches reduce this cost by changing how Hamiltonian expectation values are measured. Common strategies include grouping mutually compatible Pauli terms, formulating measurement grouping as a cliquecover problem, using unitary partitioning and constructing deterministic measurement schedules for families of Pauli observables [19]–[22]. These techniques are most useful when the Hamiltonian contains many non-commuting Pauli terms. For the diagonal Ising and QUBO Hamiltonians considered in this work, the Hamiltonian terms are already simultaneously measurable in the computational basis. The dominant avoidable cost is therefore not the number of measurement bases for the objective, but the number of distinct shifted circuits required for DQAS gradient estimation. Our contribution targets this different source of overhead by reducing two-shift gradient evaluations to one-shift evaluations under structural conditions that hold for an important class of architecture-search circuits.

## III. FUNDAMENTALS

## A. Combinatorial Optimization as a Diagonal Hamiltonian

Many discrete optimization objectives can be encoded as cost Hamiltonians that are diagonal in the computational basis. This is the standard representation used in Ising and QUBO formulations of combinatorial optimization problems and also underlies QAOA-style cost Hamiltonians [1], [2], [4], [23]. Since our measurement-saving rule relies on this diagonal structure, we review the construction explicitly on 3-SAT, a canonical NP-complete constraint satisfaction problem [24].

Consider a 3-SAT instance with Boolean variables $x _ { 1 } , \ldots , x _ { n }$ and clauses $c = 1 , \ldots , m$ . Rather than treating $3 { \mathrm { - } } \mathbf { S } \mathbf { A } \mathbf { T }$ only as a decision problem, we use the equivalent optimization objective that counts violated clauses:

$$
f ( x ) = \sum _ { c = 1 } ^ { m } v _ { c } ( x ) , \qquad v _ { c } ( x ) = \left\{ 1 , \mathrm { ~ i f ~ } c \mathrm { ~ v i o l a t e d ~ b y ~ } x , \right.
$$

A satisfying assignment exists exactly when min $\iota _ { x } f ( x ) = 0$ The corresponding quantum cost Hamiltonian is defined by assigning each computational basis state the classical objective value, $H _ { f } | x \rangle ~ = ~ f ( x ) | x \rangle$ . Equivalently, $\begin{array} { r l } { H _ { f } } & { { } = } \end{array}$ $\textstyle \sum _ { x \in \{ 0 , 1 \} ^ { n } } f ( x ) | x \rangle \langle x |$ , which makes the diagonal structure explicit. Of course, this diagonal representation is exponential in the number of qubits. For 3-SAT, this Hamiltonian can also be written compactly as a sum of clause projectors. For example, the clause $C = ( x _ { 1 } \vee { \lnot x _ { 2 } } \vee x _ { 5 } )$ is violated only when $x _ { 1 } = 0 , \ x _ { 2 } = 1 , \ x _ { 5 } = 0$ . Its clause Hamiltonian is therefore

$$
H _ { C } = \frac { I + Z _ { 1 } } { 2 } \frac { I - Z _ { 2 } } { 2 } \frac { I + Z _ { 5 } } { 2 } .
$$

This operator contributes one unit of energy exactly on assignments that violate $C$ and zero otherwise. Summing over clauses gives $\begin{array} { r } { H _ { f } ~ = ~ \sum _ { c = 1 } ^ { m } H _ { c } } \end{array}$ . Each $H _ { c }$ is a product of I and $Z$ operators, so all clause terms commute and are simultaneously diagonal in the computational basis.

Given a parameterized quantum circuit $U ( \theta )$ , the variational objective is

$$
F ( { \boldsymbol { \theta } } ) = \langle 0 | U ( { \boldsymbol { \theta } } ) ^ { \dagger } H _ { f } U ( { \boldsymbol { \theta } } ) | 0 \rangle .\tag{1}
$$

If the output state $U ( \theta ) | 0 \rangle$ is measured in the computational basis, it induces a distribution $p _ { \theta }$ over bit strings. Because $H _ { f }$ is diagonal, the objective can be estimated purely from sampled bit strings as $\overset { \vartriangle } { \boldsymbol { F } } ( \boldsymbol { \theta } ) = \mathbb { E } _ { \boldsymbol { X } \sim p _ { \boldsymbol { \theta } } } \left[ \boldsymbol { f } ( \boldsymbol { X } ) \right]$ . In practice, one repeatedly samples $X \sim p _ { \theta }$ , evaluates the classical cost $f ( X )$ and averages the resulting values.

## B. Differentiable Quantum Architecture Search (DQAS)

Differentiable Quantum Architecture Search (DQAS) can be used to minimize the objective in eq. (1). It extends variational quantum algorithms by jointly optimizing the circuit architecture and the continuous gate parameters. Instead of optimizing a single fixed circuit, DQAS maintains a parameterized probability distribution $p _ { \phi } ( a )$ over candidate circuit architectures a. During each optimization step, architectures are sampled from this distribution, their circuit parameters are optimized and the resulting objective values are used to update the architecture distribution.

For a sampled architecture $^ { a , }$ the circuit parameters θ are optimized by minimizing the objective

$$
F _ { a } ( \theta ) = \langle 0 | U _ { a } ( \theta ) ^ { \dagger } H _ { f } U _ { a } ( \theta ) | 0 \rangle ,
$$

using the parameter-shift rule. For rotation gates generated by Pauli operators, the derivative with respect to a parameter $\theta _ { i }$ is given by

$$
\frac { \partial F _ { a } ( \theta ) } { \partial \theta _ { i } } = \frac { 1 } { 2 } \left[ F _ { a } \left( \theta + \frac { \pi } { 2 } e _ { i } \right) - F _ { a } \left( \theta - \frac { \pi } { 2 } e _ { i } \right) \right] ,
$$

where $e _ { i }$ is the unit vector in the i-th parameter direction. Each gradient evaluation therefore requires two additional expectation-value estimates beyond the evaluation of the original objective.

After the circuit parameters have been updated, the objective values of the sampled architectures are used to estimate the gradient of the architecture distribution $p _ { \phi }$ using the REINFORCE estimator. Since this optimization is performed entirely on the classical processor, its computational cost is negligible compared to the repeated execution of quantum circuits.

The dominant cost of DQAS therefore stems from quantum measurements. For every sampled architecture, one measurement-based estimate of the objective is required together with two additional measurements for each parametershift gradient evaluation. As the number of sampled ar chitectures and circuit parameters increases, these repeated expectation-value evaluations quickly become the computational bottleneck. Reducing this measurement overhead is therefore essential for making DQAS practical on near-term quantum hardware. In this work, we make a first important step in this direction.

## IV. FINDING SIMPLER GRADIENTS

## A. Pauli-string rotation gradients

We first consider the common case where the parametrized gate at position j is generated by a Pauli string V, with $\bar { R } _ { V } ( \theta ) \ \stackrel { \scriptscriptstyle - } { = } \ \exp ( - i \theta V / 2 ) , \ V ^ { \scriptscriptstyle \dagger } \ = \ \bar { V } \ \mathrm { a n d } \ V ^ { 2 } \ = \ \bar { I _ { \scriptscriptstyle + } }$ . Let $U _ { < j }$ denote the part of the circuit before this gate and $U _ { > j }$ the part after it. We absorb these two parts into the input state and the observable, respectively: $| \phi _ { < j } \rangle = U _ { < j } | 0 \rangle$ and

$H _ { > j } = U _ { > j } ^ { \dagger } H U _ { > j }$ . Then the objective can be written locally around the j-th gate as

$$
F ( \theta ) = \langle \phi _ { < j } | R _ { V } ( \theta ) ^ { \dagger } H _ { > j } R _ { V } ( \theta ) | \phi _ { < j } \rangle .
$$

Differentiating gives

$$
F ^ { \prime } ( \theta ) = \langle \phi _ { < j } | R _ { V } ( \theta ) ^ { \dagger } \frac { i } { 2 } [ V , H _ { > j } ] R _ { V } ( \theta ) | \phi _ { < j } \rangle .\tag{2}
$$

Equivalently, if we define $| \phi _ { j } ( \theta ) \rangle = R _ { V } ( \theta ) | \phi _ { < j } \rangle$ , then

$$
F ^ { \prime } ( \theta ) = \frac { i } { 2 } \langle \phi _ { j } ( \theta ) | [ V , H _ { > j } ] | \phi _ { j } ( \theta ) \rangle .\tag{3}
$$

The exact zero-gradient condition follows from eq. (3): $[ V , H _ { > j } ] ~ = ~ 0$ . When this holds, $F ^ { \prime } ( \theta ) ~ = ~ 0$ and no measurement is required for this parameter. Although checking this condition may require constructing the dressed observable $H _ { > j }$ , there are two cheap sufficient tests.

First, if V commutes with the original observable H and with every gate after position j, then V also commutes with $H _ { > j }$ . Second, if V also commutes with H and every gate after position $j$ commutes with H, then again $[ V , H _ { > j } ] = 0 ;$ therefore, the gradient is again exactly zero.

These conditions are especially cheap to check for the Zonly Hamiltonians that arise in many combinatorial optimization problems. In this setting, the Pauli decomposition of H is given directly by the problem instance, so we do not need to construct or measure any new observable. Checking whether V commutes with H only requires checking whether V commutes with each Pauli term already present in this known decomposition. Similarly, checking whether a later Pauli-rotation gate commutes with H only requires checking commutation between its generator and these same known Pauli terms. Thus both tests are purely classical preprocessing steps and add negligible overhead compared with estimating the gradient by circuit measurements.

More generally, suppose that the dressed observable has a Pauli decomposition

$$
H _ { > j } = \sum _ { \ell } h _ { \ell } P _ { \ell } , \qquad h _ { \ell } \in \mathbb { R } , \qquad P _ { \ell } ^ { \dagger } = P _ { \ell } , \qquad P _ { \ell } ^ { 2 } = I .
$$

Substituting this into eq. (3) yields

$$
F ^ { \prime } ( \theta ) = \frac { 1 } { 2 } \sum _ { \ell } h _ { \ell } \langle \phi _ { j } ( \theta ) | i [ V , P _ { \ell } ] | \phi _ { j } ( \theta ) \rangle .
$$

Since both V and $P _ { \ell }$ are Pauli strings, they either commute or anticommute. If $[ V , P _ { \ell } ] = 0$ , then the corresponding term contributes nothing to the gradient. If $V P _ { \ell } = - P _ { \ell } V$ , then $[ V , P _ { \ell } ] = 2 V P _ { \ell }$ . Therefore only Pauli terms that anticommute with V contribute:

$$
F ^ { \prime } ( \theta ) = \sum _ { \ell \in \cal { A } ( V ) } h _ { \ell } \langle \phi _ { j } ( \theta ) | i V P _ { \ell } | \phi _ { j } ( \theta ) \rangle ,\tag{4}
$$

where $\mathcal { A } ( V ) = \{ \ell : V P _ { \ell } = - P _ { \ell } V \}$

For every $\begin{array} { r l r l } { \ell } & { { } } & { \in } & { { } \ A ( V ) } \end{array}$ , we have the identity $R _ { V } ( \pi / 2 ) ^ { \dagger } P _ { \ell } R _ { V } ( \pi / 2 ) = i V P _ { \ell }$ . Therefore,

$$
\begin{array} { l } { { \displaystyle F ^ { \prime } ( \theta ) = \sum _ { \ell \in A ( V ) } h _ { \ell } \langle \phi _ { j } ( \theta ) | R _ { V } ( \pi / 2 ) ^ { \dagger } P _ { \ell } R _ { V } ( \pi / 2 ) | \phi _ { j } ( \theta ) \rangle } } \\ { { \displaystyle \quad = \sum _ { \ell \in A ( V ) } h _ { \ell } \langle \phi _ { < j } | R _ { V } ( \theta + \pi / 2 ) ^ { \dagger } P _ { \ell } R _ { V } ( \theta + \pi / 2 ) | \phi _ { < j } \rangle . } } \end{array}
$$

Equivalently, if we define the anticommuting part of the Hamiltonian as $\begin{array} { r } { H _ { \mathcal { A } ( V ) } = \sum _ { \ell \in \mathcal { A } ( V ) } h _ { \ell } P _ { \ell } } \end{array}$ , then

$$
F ^ { \prime } ( \theta ) = \langle \phi _ { j } ( \theta + \pi / 2 ) | H _ { A ( V ) } | \phi _ { j } ( \theta + \pi / 2 ) \rangle .\tag{5}
$$

Thus the gradient can be estimated by measuring only these anticommuting terms and shifting the parameter by $\pi / 2$

## B. Cheaper gradients for Z-only Hamiltonians

For combinatorial optimization Hamiltonians that are diagonal in the computational basis, the gradient can often be measured at substantially lower cost. In the 3-SAT case, we have seen that the problem Hamiltonian consists only of Ztype Pauli strings $\begin{array} { r } { H = \sum _ { P _ { \ell } } h _ { \ell } P _ { \ell } , \ P _ { \ell } \in \{ I , Z \} ^ { \otimes n } } \end{array}$ . The same structure appears in many other combinatorial optimization problems, including MaxCut, QUBO, Ising models and graph coloring. In these settings, the Pauli decomposition of H is not an additional object that must be constructed or measured term by term from scratch; it is already specified by the problem instance.

1) Commuting suffix: Starting again from eq. (3), assume that all gates applied after position j commute with the original Hamiltonian H. Then $H _ { > j } \ = \ U _ { > i } ^ { \dagger } H U _ { > j } \ = \ H$ . Thus the gradient from eq. (5) simplifies. It no longer requires the dressed Hamiltonian $H _ { > j }$ , but now consists of the original problem Hamiltonian. Since H is diagonal, $H _ { \mathcal { A ( V ) } }$ is also Zdiagonal and it can be measured in the computational basis, just like the original combinatorial optimization Hamiltonian. Thus the gradient can be calculated by sampling $X \sim p _ { \theta + \pi / 2 }$ and averaging the diagonal entries $f _ { A ( V ) } ( X )$ of $H _ { \mathcal { A ( V ) } }$ . This has no additional costs, since no gate type or non-standard measurement basis is required.

Compared with the standard parameter-shift rule, $F ^ { \prime } ( \theta ) =$ $\overset { 1 } { \underset { 2 } { \_ } } ( F ( \overset { - } { \theta _ { + } } \pi / 2 ) - F ( \theta - \pi / 2 ) )$ , this method requires only one shifted circuit evaluation. Moreover, the measurement only needs the Hamiltonian terms that anticommute with $V ;$ the commuting terms are known classically to have zero contribution to the gradient.

2) Clifford suffix: A more general useful case occurs when the suffix after position j does not commute with H, but is a Clifford circuit. Let the suffix be $C _ { > j }$ . The objective function becomes $\langle C _ { > j } \phi _ { j } ( \theta ) | H | C _ { > j } \phi _ { j } ( \theta ) \rangle$ ⟩. Therefore, the gradient is

$$
F ^ { \prime } ( \theta ) = \langle C _ { > j } \phi _ { j } ( \theta + \pi / 2 ) | H _ { A _ { C } ( V ) } | C _ { > j } \phi _ { j } ( \theta + \pi / 2 ) \rangle _ { \nparallel }
$$

where $H _ { \mathbf { \mathcal { A } } _ { C } ( V ) }$ keeps the terms $h _ { \ell } P _ { \ell }$ of H for which $C _ { > j } ^ { \dagger } P _ { \ell } C _ { > j }$ anticommutes with V.

Hence, the gradient can be calculated by measuring the circuit $C _ { > j } | \phi _ { j } ( \theta + \pi / 2 ) \rangle$ . Compared to two of the naive parameter-shift rule this requires only one measurement in the computational basis. On the other hand, this introduces a quantum cost, because even though the quantum circuit technically stays the same length, because the clifford part was a suffix of the circuit it is usually handled classically as post-processing. This can no longer be done if this clifford part is introduced at the beginning of the circuit. However, it does not introduce any new measurement groups, which a classical handling of the clifford part could add additionally. In summary, if the transformed observable does not remain measurable in one cheap group and if the Clifford suffix is shallow, then physically applying the suffix may be better, than trying to handle it classically only.

## C. Measurement-Efficient DQAS

The gradient identities above lead to a direct modification of DQAS. The architecture-sampling procedure, objective evaluation and architecture update are kept unchanged. ME-DQAS differs from ordinary DQAS only in how it estimates gradients with respect to circuit parameters θ.

For each parameterized Pauli rotation in a sampled architecture, the algorithm first checks whether the derivative can be certified to vanish by commutation. If not, it checks whether the suffix structure allows the gradient to be estimated from one shifted circuit, either because the suffix preserves the diagonal Hamiltonian or because it is Clifford. Only gates that satisfy none of these conditions use the ordinary two-shift parameter-shift estimator.

The resulting procedure is shown in algorithm 1. The highlighted lines are the algorithmic changes relative to baseline DQAS.

The checks in algorithm 1 are purely structural. They use the known Pauli decomposition of the diagonal problem Hamiltonian and the generators of the gates appearing after the differentiated parameter. Thus they add only a small overhead and do not change the objective being optimized.

The first branch implements the zero-gradient certificates from eq. (3). The second branch applies the one-shift rule from eq. (5) when the circuit suffix commutes with $H _ { f } ,$ , so that only the anticommuting diagonal terms contribute to the derivative. The third branch handles the Clifford-suffix case, where the suffix can be propagated through Pauli observables while preserving Pauli structure. In all remaining cases, ME-DQAS falls back to the standard two-shift parameter-shift rule.

Consequently, ME-DQAS preserves the DQAS search objective and update structure. Its effect is limited to reducing the number of shifted circuit evaluations requested for eligible θ-gradient components. This is the measurement cost reported in the experiments below.

The probability that a gradient falls into one of the cheaper branches is problem- and architecture-dependent. It increases when the Hamiltonian is sparse or diagonal, when the gate pool contains many diagonal, commuting or Clifford operations and when the sampled suffix after the differentiated gate preserves this structure. It decreases when dense Hamiltonians or non-Clifford mixing gates make commutation with the suffix unlikely. Circuit depth therefore has two competing effects: longer suffixes create more chances to break the required structure, but a searched architecture may also learn suffixes rich in commuting or Clifford gates. We measure these effects empirically in the breakdown experiment below.

![](images/3d6891232999787249a1ba631fa214c01a60923b65744fd90d99f774f0b15632.jpg)

![](images/bcc9f926b529cf2ca035ac001591409f5e2d83a5774303626144fb7dee186b6d.jpg)

![](images/607a7cb7e5ba94011b6a360ec7c39bef8298a57f0150c9a1c953f2ed3eef67c1.jpg)  
Fig. 1. Main experimental result for 100 matched 3-SAT instances and 100 matched MaxCut instances. Each row compares baseline DQAS with ME-DQAS under the same global shot budget. Bars show means with 95% confidence intervals. The three panels measure final solution quality, how many DQAS updates fit into the budget and the requested full θ-gradient shot cost relative to baseline parameter shift.

![](images/fa1d3491cdf238c95c3ed6d119e02f6ce0f71b2354e51af6ca88662ac0c1ffdf.jpg)  
Fig. 2. Source of the requested full θ-shot reduction in fig. 1. The dominant contribution comes from gradients that are certified to be exactly zero; the diagonal one-shift rule gives the next largest contribution. The Clifford-suffix branch is negligible for this gate set.

## V. EXPERIMENTS

## A. Protocol and Metrics

We evaluate ME-DQAS against baseline DQAS using paired, shot-budgeted simulations. In each pair, both algorithms receive the same problem instance, initial $\theta ,$ architecture-distribution seed, batch size, learning rates and global shot budget. The main difference is the estimator used for θ-gradients: baseline DQAS applies the two-shift parameter-shift rule to every sampled parametric gate, while ME-DQAS uses the zero-gradient, one-shift diagonal, oneshift Clifford or fallback branch from algorithm 1.

The benchmarks use two diagonal objectives. For 3-SAT, the loss is the number of violated clauses. For MaxCut, the prob lem is written as a minimization task by counting uncut edges. In both benchmarks, $n = 8 .$ , the circuit length is $L = 8$ and each architecture position chooses from $\{ R _ { x } , R _ { y } , R _ { z } , \mathrm { C Z } \}$ Random 3-SAT formulas and random MaxCut graphs use $\begin{array} { r } { m \ = \ \alpha n } \end{array}$ local objective terms with $\alpha ~ \in ~ \{ 1 . 5 , 2 . 0 \}$ . All runs use global budget $B = 7 5 0 { , } 0 0 0 { \mathrm { ~ } }$ , batch size K = 128, $S _ { f } = S _ { \theta } = 5 0$ , one θ-update per DQAS iteration, architecture learning rate 0.3, θ-learning rate 0.15, initial θ scale 0.4 and a cap of 100 iterations.

Algorithm 1 ME-DQAS   
Require: L circuit components; operator pool ${ \overline { { \mathcal { O } } } } ;$ model $\overline { { { P _ { \phi } } _ { } } } ;$   
batch size $K ;$ Hamiltonian $H _ { f }$   
1: while not converged do   
2: Sample a batch of $K$ circuits from $P _ { \phi }$   
3: Compute the objective eq. (1) for each circuit in the   
batch   
4: for each parameterized gate $R _ { V } ( \theta _ { j } )$ in the sampled   
circuits do   
5: Let $U _ { > j }$ denote the part of the circuit after $R _ { V } ( \theta _ { j } )$   
6: Compute available commutation certificates:   
7: $c _ { V H } : = [ V , H _ { f } ] = 0$   
8: $c _ { V U } : = [ V , U _ { > j } ] = 0$   
9: c<sub>HU</sub> $: = [ H _ { f } , U _ { > j } ] = 0$   
10: if c<sub>VH</sub> and (c<sub>VU</sub> or c<sub>HU</sub>) then   
11: Set $\nabla _ { \theta _ { i } } f = 0$   
12: else if $c _ { H U }$ then   
13: Estimate $\nabla _ { \theta _ { j } } f$ from one shifted circuit at $\theta _ { j } + \pi / 2$   
14: else if $U _ { > j }$ is Clifford then   
15: Estimate $\nabla _ { \theta _ { j } } f$ using the circuit $C _ { > j } | \phi _ { j } ( \theta + \pi / 2 ) \rangle$   
16: else   
17: Estimate $\nabla _ { \theta _ { j } } f$ using the ordinary two-shift   
parameter-shift estimator   
18: end if   
19: end for   
20: Compute the gradients with respect to $\phi$   
21: Update θ and $\phi$ with gradient-based optimizers   
22: end while   
23: return circuit found with the lowest objective evaluation

Figure 1 reports final normalized best-so-far loss, the number of completed objective updates and the requested full θ- gradient shot-cost ratio. For quality, each run is normalized by the exact objective range,

$$
\widetilde { F } _ { t } = \frac { \operatorname* { m i n } _ { \tau \le t } F _ { \tau } - F _ { \mathrm { m i n } } ^ { \mathrm { e x a c t } } } { F _ { \mathrm { m a x } } ^ { \mathrm { e x a c t } } - F _ { \mathrm { m i n } } ^ { \mathrm { e x a c t } } } .
$$

ME-DQAS improves all three metrics. It reduces the requested θ-gradient shot cost to $0 . 5 8 5 7 \pm 0 . 0 0 1 4$ of baseline on 3-SAT and $0 . 6 0 6 4 \pm 0 . 0 0 4 6$ on MaxCut, corresponding to 41.4% and 39.4% fewer requested gradient shots. Under the same global shot budget, this buys roughly five additional DQAS updates: $1 4 . 0 5 \pm 0 . 0 4$ instead of $8 . 9 9 \pm 0 . 0 2$ on 3-SAT and 13.54 ± 0.13 instead of $8 . 5 6 \pm 0 . 1 0$ on MaxCut. The additional search progress coincides with lower final normalized loss on both benchmarks.

Figure 2 shows that most savings come from complete gradient removal, not from a uniform constant-factor reduction. The zero-gradient certificate saves 33.1% of baseline θ-shot cost on 3-SAT and 31.2% on MaxCut; the diagonal one-shift rule adds 8.3% and 8.1%. This matches the structural discussion above. With a Z-diagonal Hamiltonian and gate set $\{ R _ { x } , R _ { y } , R _ { z } , \mathrm { C Z } \}$ , many suffixes contain diagonal or diagonal-preserving operations, while the Clifford branch is rarely available for arbitrary-angle rotations. ME-DQAS therefore changes the measurement budget, not the objective landscape: it evaluates the same objective and uses the same outer DQAS update rule, but measures eligible θ-gradients more cheaply.

## VI. CONCLUSION

We introduced ME-DQAS, a measurement-efficient variant of differentiable quantum architecture search for diagonal combinatorial Hamiltonians. The method uses commutation certificates and one-shift rules to reduce the number of circuit evaluations requested for eligible θ-gradients while leaving the DQAS objective, architecture sampling and outer update rule unchanged.

On paired 3-SAT and MaxCut benchmarks, ME-DQAS reduced the requested full θ-gradient shot cost to 58.6% and 60.6% of baseline DQAS. Under the same global shot budget, this produced roughly five additional architecture-search updates and better final normalized best-so-far loss. The fraction of eligible gradients will depend on the Hamiltonian, gate pool, suffix structure and hardware noise, but the mechanism is general: when the cost Hamiltonian and circuit suffix expose commutation or Clifford structure, DQAS gradients need not be measured with the full two-shift rule.

## REFERENCES

[1] A. Lucas, “Ising formulations of many NP problems,” Frontiers in Physics, vol. 2, p. 5, 2014.

[2] F. Glover, G. Kochenberger, and Y. Du, “Quantum bridge analytics I: A tutorial on formulating and using QUBO models,” 4OR, vol. 17, no. 4, pp. 335–371, 2019.

[3] A. Peruzzo, J. McClean, P. Shadbolt, M.-H. Yung, X.-Q. Zhou, P. J. Love, A. Aspuru-Guzik, and J. L. O’Brien, “A variational eigenvalue solver on a photonic quantum processor,” Nature Communications, vol. 5, no. 1, p. 4213, 2014.

[4] E. Farhi, J. Goldstone, and S. Gutmann, “A quantum approximate optimization algorithm,” 2014, arXiv:1411.4028. [Online]. Available: https://arxiv.org/abs/1411.4028

[5] J. R. McClean, J. Romero, R. Babbush, and A. Aspuru-Guzik, “The theory of variational hybrid quantum-classical algorithms,” New Journal of Physics, vol. 18, no. 2, p. 023023, 2016.

[6] M. Cerezo, A. Arrasmith, R. Babbush, S. C. Benjamin, S. Endo, K. Fujii, J. R. McClean, K. Mitarai, X. Yuan, L. Cincio, and P. J. Coles, “Variational quantum algorithms,” Nature Reviews Physics, vol. 3, no. 9, pp. 625–644, 2021.

[7] K. Mitarai, M. Negoro, M. Kitagawa, and K. Fujii, “Quantum circuit learning,” Physical Review A, vol. 98, no. 3, p. 032309, 2018.

[8] M. Schuld, V. Bergholm, C. Gogolin, J. Izaac, and N. Killoran, “Evaluating analytic gradients on quantum hardware,” Physical Review A, vol. 99, no. 3, p. 032331, 2019.

[9] S.-X. Zhang, C.-Y. Hsieh, S. Zhang, and H. Yao, “Differentiable quantum architecture search,” Quantum Science and Technology, vol. 7, no. 4, p. 045023, 2022.

[10] M. Ostaszewski, E. Grant, and M. Benedetti, “Structure optimization for parameterized quantum circuits,” Quantum, vol. 5, p. 391, 2021.

[11] H. R. Grimsley, S. E. Economou, E. Barnes, and N. J. Mayhall, “An adaptive variational algorithm for exact molecular simulations on a quantum computer,” Nature Communications, vol. 10, no. 1, p. 3007, 2019.

[12] H. L. Tang, V. O. Shkolnikov, G. S. Barron, H. R. Grimsley, N. J. Mayhall, E. Barnes, and S. E. Economou, “Qubit-ADAPT-VQE: An adaptive algorithm for constructing hardware-efficient ansatze on a¨ quantum processor,” PRX Quantum, vol. 2, no. 2, p. 020310, 2021.

[13] W. Wu, G. Yan, X. Lu, K. Pan, and J. Yan, “QuantumDARTS: Differentiable quantum architecture search for variational quantum algorithms,” in Proceedings of the 40th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 202. PMLR, 2023, pp. 37 745–37 764. [Online]. Available: https://proceedings.mlr.press/v202/wu23v.html

[14] P. G. Anastasiou, N. J. Mayhall, E. Barnes, and S. E. Economou, “How to really measure operator gradients in ADAPT-VQE,” 2023, arXiv:2306.03227. [Online]. Available: https://arxiv.org/abs/2306.03227

[15] Y. Du, T. Huang, S. You, M.-H. Hsieh, and D. Tao, “Quantum circuit architecture search for variational quantum algorithms,” npj Quantum Information, vol. 8, no. 1, p. 62, 2022.

[16] H. Wang, Y. Ding, J. Gu, Y. Lin, D. Z. Pan, F. T. Chong, and S. Han, “QuantumNAS: Noise-adaptive search for robust quantum circuits,” in 2022 IEEE International Symposium on High-Performance Computer Architecture (HPCA), 2022, pp. 692–708.

[17] D. Wierichs, J. Izaac, C. Wang, and C. Y.-Y. Lin, “General parametershift rules for quantum gradients,” Quantum, vol. 6, p. 677, 2022.

[18] T. Hubregtsen, F. Wilde, S. Qasim, and J. Eisert, “Single-component gradient rules for variational quantum algorithms,” Quantum Science and Technology, vol. 7, no. 3, p. 035008, 2022.

[19] V. Verteletskyi, T.-C. Yen, and A. F. Izmaylov, “Measurement optimization in the variational quantum eigensolver using a minimum clique cover,” The Journal of Chemical Physics, vol. 152, no. 12, p. 124114, 2020.

[20] A. F. Izmaylov, T.-C. Yen, R. A. Lang, and V. Verteletskyi, “Unitary partitioning approach to the measurement problem in the variational quantum eigensolver method,” Journal of Chemical Theory and Computation, vol. 16, no. 1, pp. 190–195, 2020.

[21] A. Zhao, A. Tranter, W. M. Kirby, S. F. Ung, A. Miyake, and P. J. Love, “Measurement reduction in variational quantum algorithms,” Physical Review A, vol. 101, no. 6, p. 062322, 2020.

[22] H.-Y. Huang, R. Kueng, and J. Preskill, “Efficient estimation of pauli observables by derandomization,” Physical Review Letters, vol. 127, no. 3, p. 030503, 2021.

[23] S. Hadfield, Z. Wang, B. O’Gorman, E. G. Rieffel, D. Venturelli, and R. Biswas, “From the quantum approximate optimization algorithm to a quantum alternating operator ansatz,” Algorithms, vol. 12, no. 2, p. 34, 2019.

[24] R. M. Karp, “Reducibility among combinatorial problems,” in Complexity of Computer Computations, R. E. Miller, J. W. Thatcher, and J. D. Bohlinger, Eds. New York: Plenum Press, 1972, pp. 85–103.