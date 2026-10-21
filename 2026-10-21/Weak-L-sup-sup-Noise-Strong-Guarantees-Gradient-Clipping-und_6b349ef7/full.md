# Weak-L<sup>α</sup> Noise, Strong Guarantees: Gradient Clipping under Afine Heavy Tails

Alyona Nekrasova<sup>1</sup> Makar Sidorov<sup>2</sup> Savelii Chezhegov<sup>3</sup> Alexander Gasnikov<sup>4</sup> Margarita Preobrazhenskaia<sup>2</sup> Igor Preobrazhensky<sup>2</sup> Eduard Gorbunov<sup>5</sup>

<sup>1</sup>Independent researcher <sup>2</sup>Yaroslavl State University

<sup>3</sup>Moscow Independent Research Institute of Artificial Intelligence <sup>4</sup>Innopolis University <sup>5</sup>Mohamed bin Zayed University of Artificial Intelligence

October 21, 2026

## Abstract

Gradient clipping is a key tool for obtaining high-probability convergence guarantees for stochastic first-order methods under heavy-tailed noise. Existing analyses typically assume a bounded central moment of order $\alpha \in ( 1 , 2 ]$ , which excludes distributions with tails of order u<sup>−α</sup> at the critical exponent. In this paper, we consider smooth convex optimization under afine heavy-tailed noise, whose scale may grow with the gradient norm. We derive high-probability convergence guarantees for clipped SGD (Clip-SGD) and its accelerated variant Clip-SSTM under a weak-L<sup>α</sup> tail condition, which is strictly weaker than the corresponding moment bound. For $1 < \alpha < 2$ , the clipping bounds have the same powers of the clipping level as in the moment case, and both methods attain the additive stochastic rate $K ^ { - ( \alpha - 1 ) / \alpha }$ . The gradient-dependent noise contributes a faster-decaying term of order $K ^ { - 2 ( \alpha - 1 ) / \alpha }$ , up to logarithmic factors. At $\alpha = 2 .$ , we characterize the additional logarithmic factor on the Lorentz scale and establish its necessity for the additive error at fixed confidence. We also analyze Clip-SGD with prescribed clipping levels and derive finite-horizon guarantees for reaching a neighborhood determined by the clipping bias. For additive noise, matching lower bounds show that this neighborhood is unavoidable, up to constants and the natural error cap. Numerical experiments with Pareto noise illustrate the theoretical results.

## 1 Introduction

Gradient clipping rescales a stochastic gradient whenever its norm exceeds a prescribed level λ. It is a standard tool for stabilizing the training of machine learning models. It was introduced to handle exploding gradients in recurrent neural networks (Pascanu et al., 2013) and is now used in largescale training pipelines, including GPT-3 (Brown et al., 2020) and LLaMA (Touvron et al., 2023). It is also a basic component of diferentially private SGD (Abadi et al., 2016). One explanation for its usefulness is that stochastic gradient noise can be heavy-tailed, as observed in deep learning and language-model training (S¸im¸sekli et al., 2019; Zhang et al., 2020b). Moreover, SGD iterates can have heavy-tailed stationary distributions even for light-tailed data (G¨urb¨uzbalaban et al., 2021).

Table 1: Summary of known and new high-probability complexity results for Clip-SGD and Clip-SSTM with the clipping level tuned to the horizon. Complexity is the total number of oracle calls needed for a method to guarantee f− $\cdot f ^ { * } \leq \varepsilon$ with probability at least $1 - \beta ;$ the constants depending only on α and the logarithmic factors are omitted for simplicity. Notation: $\zeta : = g ( x , \xi ) - \nabla f ( x )$ is the noise, $\alpha \in ( 1 , 2 ]$ is its tail index and σ its level; $R \geq \| x ^ { 0 } - x ^ { * } \|$ is any upper bound on the initial distance; $\beta \in ( 0 , 1 )$ is the failure probability. Under the afine conditions, $\begin{array} { r } { \mathbb { E } \| \zeta \| ^ { \alpha } \leq \sigma _ { 0 } ^ { \alpha } + \sigma _ { 1 } ^ { \alpha } \| \nabla f ( x ) \| ^ { \alpha } } \end{array}$ (Assumption 3.4) or $\mathbb { P } \{ \| \zeta \| > u \} \le ( \sigma _ { 0 } ^ { \alpha } + \sigma _ { 1 } ^ { \alpha } \| \nabla f ( x ) \| ^ { \alpha } ) u ^ { - \alpha }$ for all $u > 0$ (Assumption 3.5). The error $f - f ^ { * }$ is measured at the averaged iterate $\bar { x } ^ { K }$ for Clip-SGD and at the last iterate $y ^ { K }$ for Clip-SSTM. The results of this paper are highlighted in blue.
<table><tr><td>Method</td><td>Result</td><td>Noise condition</td><td>Oracle calls</td><td></td></tr><tr><td>Clip-SGD</td><td>Sadiev et al. (2023)</td><td> $\mathbb { E } \| \zeta \| ^ { \alpha } \leq \sigma ^ { \alpha } , \alpha \in ( 1 , 2 ]$ </td><td> $\left\{ { \frac { L R ^ { 2 } } { \varepsilon } } , \left( { \frac { \sigma R } { \varepsilon } } \right) ^ { \frac { \alpha } { \alpha - 1 } } \right\}$  max</td><td></td></tr><tr><td>Clip-SGD</td><td>Theorem 5.1</td><td>Assumption 3.4 or 3.5</td><td>max</td><td> $\left\{ \begin{array} { l } { { \frac { L R ^ { 2 } } { \varepsilon } , \left( \frac { \sigma _ { 0 } R } { \varepsilon } \right) ^ { \frac { \alpha } { \alpha - 1 } } , \left( \frac { \sigma _ { 1 } ^ { 2 } L R ^ { 2 } } { \varepsilon } \right) ^ { \frac { \alpha } { 2 \left( \alpha - 1 \right) } } } } \end{array} \right\}$ </td></tr><tr><td></td><td>Clip-SSTM Gorbunov et al. (2020)</td><td> $\mathbb { E } \| \zeta \| ^ { 2 } \leq \sigma ^ { 2 }$ </td><td>max</td><td> $\left\{ \begin{array} { l l } { \sqrt { \frac { L R ^ { 2 } } { \varepsilon } } , \frac { \sigma ^ { 2 } R ^ { 2 } } { \varepsilon ^ { 2 } } } \end{array} \right\}$ </td></tr><tr><td></td><td>Clip-SSTM Sadiev et al. (2023)</td><td> $\mathbb { E } \| \zeta \| ^ { \alpha } \leq \sigma ^ { \alpha } , \alpha \in ( 1 , 2 ]$ </td><td>max</td><td> $\left\{ { \sqrt { \frac { L R ^ { 2 } } { \varepsilon } } } , \left( { \frac { \sigma R } { \varepsilon } } \right) ^ { \frac { \alpha } { \alpha - 1 } } \right\}$ </td></tr><tr><td></td><td>Clip-SSTM uniform scale (Cor. I.9) Assumption 3.4 or 3.5</td><td></td><td>max</td><td> $\left\{ \sqrt { \frac { L R ^ { 2 } } { \varepsilon } } , \left( \frac { \sigma _ { 0 } R } { \varepsilon } \right) ^ { \frac { \alpha } { \alpha - 1 } } , \left( \frac { \sigma _ { 1 } L R ^ { 2 } } { \varepsilon } \right) ^ { \frac { \alpha } { \alpha - 1 } } \right\}$ </td></tr><tr><td>Clip-SSTM Theorem 5.2</td><td></td><td>Assumption 3.4 or 3.5</td><td>max</td><td> $\begin{array} { r } { \left\{ \sqrt { \frac { L R ^ { 2 } } { \varepsilon } } , \left( \frac { \sigma _ { 0 } R } { \varepsilon } \right) ^ { \frac { \alpha } { \alpha - 1 } } , \left( \frac { \sigma _ { 1 } ^ { 2 } L R ^ { 2 } } { \varepsilon } \right) ^ { \frac { \alpha ^ { \prime } } { 2 ( \alpha - 1 ) } } \right\} } \end{array}$ </td></tr></table>

Under a bounded central moment of order $\alpha \in ( 1 , 2 ]$ , clipped SGD (Clip-SGD) converges with high probability and logarithmic dependence on the inverse failure probability, whereas SGD without clipping may fail when the variance is infinite (Zhang et al., 2020b; Gorbunov et al., 2020; Sadiev et al., 2023). Section 2 discusses extensions of these results.

Two limitations of much of the existing high-probability theory motivate this work. First, the moment assumption $\mathbb { E } \| g ( x , \xi ) - \nabla f ( x ) \| ^ { \alpha } \leq \sigma ^ { \alpha }$ is stronger than the tail control used in our analysis. Pareto distributions and non-Gaussian α-stable distributions with $\alpha \in ( 1 , 2 )$ have tails of order $u ^ { - \alpha }$ but an infinite α-th moment; the latter have been proposed as models of stochastic gradient noise in deep learning (S¸im¸sekli et al., 2019). A moment theorem therefore applies only at a smaller exponent, with a slower polynomial rate when that exponent is fixed. The weak- $L ^ { \alpha }$ (Marcinkiewicz) condition $\mathbb { P } \{ \| g ( x , \xi ) - \nabla f ( x ) \| > u \} \le \sigma ^ { \alpha } u ^ { - \alpha }$ controls the tail directly.

Second, many high-probability analyses tune the clipping level to the noise parameters and the horizon and assume a uniform noise scale on the relevant region. Practical implementations often use a prescribed constant level, and the noise scale may vary with the gradient. Although prior work addresses some of these features separately (Section 2), their combination under a local weak-tail condition motivates the following question:

What do clipped methods guarantee under a pure tail condition on the noise, with a noise scale that grows with the gradient norm, and with a clipping level chosen by the user?

In this paper, we address this question for Clip-SGD and its accelerated variant Clip-SSTM.

Our contributions. We analyze Clip-SGD and Clip-SSTM under afine moment and weak- $L ^ { \alpha }$ noise conditions, which allow the noise scale to depend on the gradient norm. Table 1 compares the guarantees for horizon-dependent tuning with prior results. Table 3 summarizes the guarantees and lower bounds for prescribed clipping levels. Our contributions are as follows.

• Clipping under tail conditions. We bound the clipping bias and the second moment of the clipped error under a weak- $L ^ { \alpha }$ tail condition. The bounds have the same powers of the clipping level as in the moment case, with an unavoidable logarithmic factor at $\alpha = 2$ (Lemma 4.1).

• New high-probability results for Clip-SGD and Clip-SSTM. With horizon-dependent tuning, both methods attain high-probability rates with explicit constants (Theorems 5.1 and 5.2). In the nontrivial stochastic regime, the additive term is optimal in K, $\sigma _ { 0 }$ , and $R ;$ the corresponding confidence dependence is optimal under the moment condition and under the weak-tail condition for $\alpha < 2$ (Theorem I.18 and Propositions H.2 and H.3). At the weak- $L ^ { 2 }$ endpoint, the additive term contains the necessary factor $\sqrt { \ln ( e K ) }$ at fixed confidence (Corollary I.19). The gradientdependent noise contributes a faster-decaying term. For Clip-SSTM, this follows from a weighted bound on the true gradients; we do not establish optimality of the multiplicative term.

• Prescribed clipping levels. For suficiently large prescribed clipping levels, we bound the error of the averaged iterate by a clipping-bias neighborhood plus terms that vanish with the horizon (Theorem 5.3). For arbitrary positive clipping levels, we obtain a best-iterate guarantee under additive or suficiently small multiplicative noise; the weak-tail version requires $\alpha \ : < \ : 2$ (Theorem 5.4). Additive-noise lower bounds show that the neighborhood is unavoidable up to α-dependent constants and truncation at $L R ^ { 2 }$ (Theorem 5.5). For additive noise, a clipping level proportional to $( K + 1 ) ^ { 1 / \alpha }$ , held constant within each run, recovers the rate $K ^ { - ( \alpha - 1 ) / \alpha }$ up to logarithmic factors.

• Numerical experiments. In Section 6 and Appendix J, we illustrate the derived results in numerical experiments with Pareto noise.

## 2 Related Work

We discuss the closely related works below and defer a more detailed overview to Appendix B.

High-probability bounds under heavy-tailed noise. High-probability guarantees with logarithmic dependence on the inverse failure probability were first obtained for clipped and truncated stochastic methods under bounded variance by Nazin et al. (2019) and Gorbunov et al. (2020). Sadiev et al. (2023) extended the analysis to bounded central moments of order $\alpha \in ( 1 , 2 ]$ . For smooth convex problems, they derive the rate $\widetilde { O } ( L R ^ { 2 } / K + \sigma R K ^ { - ( \alpha - 1 ) / \alpha } )$ for Clip-SGD and its accelerated counterpart for Clip-SSTM. These are the closest constant-scale results: Theorems 5.1 and 5.2 recover them when $\sigma _ { 1 } = 0$ and the moment condition holds (Table 1).

The weak-L<sup>p</sup> condition itself is used by El Bakkali et al. (2026) for zeroth-order non-convex optimization under global sample-wise smoothness and a constant noise scale. Their methods clip scalar directional derivatives and use mini-batches or a momentum scheme with a warm-start batch. We consider first-order methods for convex problems with one oracle call per iteration, under a local tail condition whose scale may grow with the gradient norm. Our assumption requires neither symmetry nor a density. Appendix B discusses the full range of bounded-moment results, structural noise assumptions, and methods without clipping, including the accelerated convex method of He et al. (2025).

Afine noise, fixed clipping levels, and lower bounds. Afine variance is classical in the analysis of SGD (Bottou et al., 2018; Gower et al., 2019; Khaled and Richt´arik, 2023). Most high-probability analyses under heavy-tailed noise use a constant scale on the relevant region; state-dependent finite-variance bounds appear in Tsai et al. (2022); Das et al. (2024), and afine moment bounds are considered in the concurrent work of Lobanov et al. (2026).

For a fixed clipping level, Koloskova et al. (2023) characterize the clipping bias and derive tight in-expectation guarantees under bounded variance. Khah et al. (2026) obtain high-probability bounds for the best iterate with an arbitrary clipping level under a bounded moment assumption. Table 3 compares their regime $\lambda > 4 L R$ . We extend prescribed-level guarantees to afine weak- $L ^ { \alpha }$ noise and derive matching additive-noise lower bounds, distinguishing averaged and best iterates.

Stochastic convex lower bounds $_ \mathrm { g o }$ back to Nemirovski and Yudin (1983). Vural et al. (2022) establish the optimal heavy-tailed rate $K ^ { - ( \alpha - 1 ) / \alpha }$ , and Liu (2026a) derive high-probability lower bounds for non-smooth convex problems. Our lower bounds concern smooth unconstrained problems and capture the confidence dependence and the additional logarithmic factors at the weak- $L ^ { 2 }$ endpoint.

Concurrent work. Lobanov et al. (2026) analyze a restarted accelerated method with clipping and projection under $( H _ { 0 } , H _ { 1 } )$ -smoothness (Alimisis et al., 2026) and a moment bound combining additive, gradient-dependent, and function-gap-dependent noise. Their method uses one sample per iteration. For L-smooth objectives and no function-gap-dependent term, their convex complexity has the same polynomial dependence as Theorem 5.2, including the dependence on $\sigma _ { 1 }$ , whose optimality remains open in both works.

Their analysis covers generalized smoothness, strong convexity, and noise decaying with the function gap. We analyze Clip-SGD and Clip-SSTM without restarts or projections under a tail condition on a ball, including the weak- $L ^ { 2 }$ endpoint with infinite variance, and study prescribed clipping levels. Appendix B gives the precise assumptions, complexity comparison, and relation between the lower bounds.

## 3 Problem Setup

We consider the following stochastic optimization problem and assumptions.

Notation. We use the standard notation: $\langle \cdot , \cdot \rangle$ denotes the standard inner product in $\mathbb { R } ^ { d }$ and $\| \cdot \|$ is the induced Euclidean norm, $B _ { r } ( { \boldsymbol x } ) : = \{ { \boldsymbol y } \in \mathbb { R } ^ { d } : \| { \boldsymbol y } - { \boldsymbol x } \| \leq r \}$ is the ball with center at x and radius $r , ( a ) _ { + } : = \operatorname* { m a x } \{ a , 0 \}$ , and ${ \bf 1 } _ { A }$ is the indicator of an event $A$ . For a σ-algebra G, E[· | G] denotes the conditional expectation w.r.t. $\mathcal { G }$

Problem. We consider the stochastic optimization problem

$$
\operatorname* { m i n } _ { \boldsymbol { x } \in \mathbb { R } ^ { d } } f ( \boldsymbol { x } ) ,\tag{1}
$$

where $f : \mathbb { R } ^ { d }  \mathbb { R }$ is Borel measurable and accessible only through the stochastic oracle specified below. We assume that a minimizer $x ^ { * }$ exists and write $f ^ { * } : = f ( x ^ { * } )$ . Following Sadiev et al. (2023), we impose the smoothness, convexity, and noise assumptions only on a ball around a minimizer. Given $R \geq \| x ^ { 0 } - x ^ { * } \| > 0$ , we use $Q = B _ { 2 R } ( x ^ { * } )$ for Clip-SGD and $Q = B _ { 3 R } ( x ^ { * } )$ for Clip-SSTM. The analysis shows that the relevant iterates remain in these balls with high probability. No smoothness or convexity is required outside $Q$

Assumptions on $f .$ We start with the standard assumptions on the objective function.

Assumption 3.1 (Local L-smoothness). We assume that $f$ is L-smooth on $Q ,$ , i.e., $\parallel \nabla f ( x ) -$ $\nabla f ( y ) \| \leq L \| x - y \|$ for all $x , y \in Q$

Assumption 3.2 (Gradient bound). We assume that $\| \nabla f ( x ) \| ^ { 2 } \leq 2 L \left( f ( x ) - f ^ { * } \right)$ for all $x \in Q .$

Assumption 3.3 (Local convexity). We assume that $f$ is convex on $Q .$ , i.e., $f ( y ) \geq f ( x ) +$ $\langle \nabla f ( x ) , y - x \rangle$ for all $x , y \in Q$

If $f$ is convex and L-smooth on $\mathbb { R } ^ { d }$ , then Assumptions 3.1–3.3 hold for any R. Here these properties are required only on $Q ,$ which contains the relevant iterates with high probability. Assumption 3.2 follows from Assumptions 3.1 and 3.3, since $Q$ is a ball centered at $x ^ { * }$ (Lemma E.1). We state it separately to simplify the comparison with Sadiev et al. (2023).

Stochastic oracle. Let $\mathcal { G } _ { 0 } : = \sigma (  { \boldsymbol { { x } } } ^ { 0 } )$ and $\mathcal { G } _ { k } : = \sigma ( x ^ { 0 } , \xi ^ { 0 } , \ldots , \xi ^ { k - 1 } )$ for $k \geq 1$ . The oracle is a jointly measurable, finite vector field $\boldsymbol { g } ( \boldsymbol { x } , \boldsymbol { \xi } ) \in \mathbb { R } ^ { d }$ satisfying $\mathbb { E } _ { \xi } [ g ( x , \xi ) ] = \nabla f ( x )$ for all $x \in Q$ . It need not be the gradient of a random function; expected-loss minimization with $g = \nabla f _ { \xi }$ is a special case. At iteration $k ,$ the method samples $\xi ^ { k } \sim \mathcal { D }$ independently of $\mathcal { G } _ { k }$ and queries the oracle at a $\mathcal { G } _ { k } .$ -measurable point. The noise assumptions hold only on $Q ;$ outside $Q _ { i }$ , the oracle is an arbitrary finite jointly measurable extension. Appendix E explains why these pointwise assumptions apply to adaptive queries and why the extension does not enter the bounds. We consider two afine noise models, whose scale may grow with the gradient norm.

Assumption 3.4 (Afine moment condition). We assume that there exist $\sigma _ { 0 } , \sigma _ { 1 } \geq 0$ and $\alpha \in$ (1, 2] such that for all $x \in Q$ we have $\mathbb { E } _ { \xi } [ g ( x , \xi ) ] = \nabla f ( x )$ and

$$
\begin{array} { r } { \mathbb { E } _ { \xi } \| g ( x , \xi ) - \nabla f ( x ) \| ^ { \alpha } \leq \sigma _ { 0 } ^ { \alpha } + \sigma _ { 1 } ^ { \alpha } \| \nabla f ( x ) \| ^ { \alpha } . } \end{array}\tag{2}
$$

Algorithm 1 Clip-SGD   
Require: starting point $x ^ { 0 }$ , stepsize $\gamma > 0$ , clipping levels $\{ \lambda _ { k } \} _ { k \ge 0 }$ , number of iterations $K + 1$   
1: for $k = 0 , 1 , \ldots , K$ do   
2: Sample $\xi ^ { k } \sim \mathcal { D }$ independently of $\mathcal { G } _ { k } ;$ set $\begin{array} { r } { g ^ { k } = g ( x ^ { k } , \xi ^ { k } ) } \end{array}$   
3: $\widetilde { g } ^ { k } = { \mathsf { c l i p } } ( g ^ { k } , \lambda _ { k } )$   
4: $\boldsymbol { x } ^ { k + 1 } = \boldsymbol { x } ^ { k } - \gamma \widetilde { g } ^ { k }$   
5: end for   
Ensure: $\begin{array} { r } { \bar { x } ^ { K } = \frac { 1 } { K + 1 } \sum _ { k = 0 } ^ { K } x ^ { k } } \end{array}$

Assumption 3.5 (Afine weak- $L ^ { \alpha }$ (Marcinkiewicz) condition). We assume that there exist $\sigma _ { 0 } , \sigma _ { 1 } \geq 0$ and $\alpha \in ( 1 , 2 ]$ such that for all $x \in Q$ we have $\mathbb { E } _ { \xi } [ g ( x , \xi ) ] = \nabla f ( x )$ and, for all $u > 0$

$$
\mathbb { P } \left\{ \| g ( x , \xi ) - \nabla f ( x ) \| > u \right\} \leq \frac { \sigma _ { 0 } ^ { \alpha } + \sigma _ { 1 } ^ { \alpha } \| \nabla f ( x ) \| ^ { \alpha } } { u ^ { \alpha } } .\tag{3}
$$

Condition (3) restricts only the tail of the noise. Equivalently, $\| \| g ( x , \xi ) - \nabla f ( x ) \| \| _ { L ^ { \alpha , \infty } } ^ { \alpha } \ \leq$ $\sigma _ { 0 } ^ { \alpha } + \sigma _ { 1 } ^ { \alpha } \| \nabla f ( x ) \| ^ { \alpha }$ , where $\| \cdot \| _ { L ^ { \alpha , \infty } }$ denotes the weak- $L ^ { \alpha }$ quasi-norm. For $\alpha > 1$ , this quasi-norm is equivalent to the norm of the Marcinkiewicz space $M _ { \alpha }$ , which is a classical rearrangement invariant Banach space (Kre˘ın et al., 1982; Bennett and Sharpley, 1988); see Appendix C for the details. By Markov’s inequality, Assumption 3.4 implies Assumption 3.5 with the same $\sigma _ { 0 } , \sigma _ { 1 } , \alpha ,$ but the converse implication does not hold at the same exponent: weak- $L ^ { \alpha }$ noise may have infinite α-th moment and, for $\alpha = 2$ , infinite variance (see the example in Section 3). When $\sigma _ { 1 } = 0$ Assumption 3.4 recovers the standard heavy-tailed noise assumption (Zhang et al., 2020b; Sadiev et al., 2023).

Example: Pareto noise at the critical index. Weak- $L ^ { \alpha }$ noise can have an infinite α-th moment. For example, non-Gaussian symmetric α-stable distributions with $\alpha \in ( 1 , 2 )$ , proposed as models of stochastic gradient noise in deep learning (S¸im¸sekli et al., 2019), have tails $\mathbb { P } \{ | Y | > u \} \sim$ $c _ { \alpha } u ^ { - \alpha }$ as $u \to \infty$ (Samorodnitsky and Taqqu, 1994). For a concrete example, let $\sigma > 0 , \alpha \in ( 1 , 2 ]$ and $\xi = \sigma \varepsilon \mathsf { R }$ , where ε is an independent Rademacher sign and $\mathbb { P } \{ { \sf R } > u \} =$ min $\{ 1 , u ^ { - \alpha } \}$ . Then $\mathbb { E } \xi = 0 , \| \xi \| _ { L ^ { \alpha , \infty } } = \sigma$ , and $\mathbb { E } | \xi | ^ { \alpha } = \infty$ , while $\mathbb { E } | \xi | ^ { q } = \sigma ^ { q } \alpha / ( \alpha - q )$ for $0 < q < \alpha$ (Proposition C.7). Thus, the moment condition applies only at a smaller exponent, whereas the weak-tail condition holds at the critical exponent with the exact scale. The Pareto tail bound is attained for $u \geq \sigma ,$ and a d-dimensional version is $\xi = \sigma { \mathsf { R } } U$ , where U is independent and uniform on the unit sphere. Our experiments use symmetric or centered asymmetric Pareto noise in scalar or one-directional form, with an afine scale $S ( x )$

Considered methods. For $\lambda > 0$ , the clipping operator is defined as follows: ${ \mathsf { c l i p } } ( v , \lambda ) : = $ min $\{ 1 , \lambda / \| v \| \}$ v for $v \neq 0$ and $\mathtt { c l i p } ( 0 , \lambda ) : = 0$ . We study the convergence of clipped SGD (Clip-SGD, Algorithm 1) in terms of its averaged iterate $\textstyle { \bar { x } } ^ { K } : = { \frac { 1 } { K + 1 } } \sum _ { k = 0 } ^ { K } x ^ { k }$ , and of its accelerated variant Clip-SSTM, which we introduce in Section 5.1.

## 4 Clipping under Afine Heavy-Tailed Noise

The key technical ingredient of the high-probability analysis of clipped methods is a lemma bounding the bias and the variance of the clipped stochastic gradient. In this section, we generalize Lemma 5.1 from Sadiev et al. (2023) to the setting of this paper, i.e., to the noise satisfying either the afine weak- $L ^ { \alpha }$ condition or the afine α-th moment condition. For brevity, we denote $A ( x ) : = \sigma _ { 0 } ^ { \alpha } + \sigma _ { 1 } ^ { \alpha } \| x \| ^ { \alpha }$

Lemma 4.1 (Clipping under afine heavy-tailed noise). Let $\alpha \in ( 1 , 2 ] , \lambda > 0$ , and let X be a random vector in $\mathbb { R } ^ { d }$ such that $\mathbb { E } X = x , \| x \| \leq \lambda / 2$ , and either (a) $\mathbb { P } \{ \| X - x \| > u \} \leq A ( x ) u ^ { - \alpha }$ for all $u > 0$ , or $( \mathrm { b } ) \ \mathbb { E } \| X - x \| ^ { \alpha } \leq A ( x )$ . Let $\bar { X } : = { \mathsf { c l i p } } ( X , \lambda )$ . Then $\| { \widetilde { X } } - \mathbb { E } { \widetilde { X } } \| \leq 2 \lambda$ almost surely, $\mathbb { E } \Vert \widetilde { X } - \mathbb { E } \widetilde { X } \Vert ^ { 2 } \leq \mathbb { E } \Vert \widetilde { X } - \widetilde { x } \Vert ^ { 2 }$ , and

$$
\| \mathbb { E } \widetilde { X } - x \| \le \frac { b A ( x ) } { \lambda ^ { \alpha - 1 } } , \qquad \mathbb { E } \| \widetilde { X } - x \| ^ { 2 } \le v \lambda ^ { 2 - \alpha } A ( x ) ,\tag{4}
$$

where $( b , v ) = ( b _ { \alpha } , v _ { \alpha } ( ( 3 \lambda / 2 ) ^ { \alpha } / A ( x ) ) )$ in case (a) (the second bound reads 0 if $A ( x ) = 0 )$ and $( b , v ) = ( b _ { \alpha } ^ { \mathrm { m o m } } , v _ { \alpha } ^ { \mathrm { m o m } } )$ in case (b), with

$$
\begin{array} { r } { v _ { \alpha } ( s ) : = \left( \frac { 3 } { 2 } \right) ^ { 2 - \alpha } \operatorname* { m i n } \left\{ \frac { 2 } { 2 - \alpha } , 1 + \frac { 2 } { \alpha } \ln ( 1 + s ) \right\} , } \end{array}\tag{5}
$$

$$
b _ { \alpha } : = \frac { 2 ^ { \alpha - 1 } } { \alpha - 1 } , \qquad b _ { \alpha } ^ { \mathrm { m o m } } : = \frac { 2 ^ { \alpha - 1 } ( \alpha - 1 ) ^ { \alpha - 1 } } { \alpha ^ { \alpha } } ,\tag{6}
$$

and $v _ { \alpha } ^ { \mathrm { m o m } } : = ( \frac { 3 } { 2 } ) ^ { 2 - \alpha }$ , for $s \in [ 0 , \infty ]$ where $\textstyle { \frac { 2 } { 2 - \alpha } } : = + \infty$ for $\alpha = 2 ;$ here $\begin{array} { r } { \frac { 1 } { 2 } \le b _ { \alpha } ^ { \mathrm { m o m } } \le 1 } \end{array}$ and $\begin{array} { r } { 1 \leq v _ { \alpha } ^ { \mathrm { m o m } } \leq \frac { 3 } { 2 } . } \end{array}$

Both cases also hold conditionally on a σ-algebra G with respect to which x is measurable (Corollary D.5). Appendix D gives the separate statements, proofs, and endpoint remarks. The bias is bounded by $b A ( x ) \lambda ^ { 1 - \alpha }$ , and the second moment by $v A ( x ) \lambda ^ { 2 - \alpha }$ . For $\alpha < 2$ , these bounds have the same powers of λ as under the moment assumption considered by Sadiev et al. (2023). Case (a) applies to the strictly larger class L<sup>α,∞</sup> (Propositions C.5 and C.7). Its constants reflect the weaker assumption: $\begin{array} { r } { v _ { \alpha } ( s ) \le v _ { \alpha } ( \infty ) = \frac { 2 } { 2 - \alpha } ( \frac { 3 } { 2 } ) ^ { 2 - \alpha } } \end{array}$ for $\alpha < 2$ , while at $\alpha = 2$ the logarithmic factor in (5) is unavoidable (Remark D.3).

## 5 Main Results

In this section, we present high-probability convergence guarantees for Clip-SGD with a constant clipping level tuned as in Sadiev et al. (2023). To handle both noise models simultaneously, let $( b , v )$ be the clipping constants from Lemma 4.1 and $c : = 1 8 0 \operatorname* { m a x } \{ v , b / 3 \}$ . Under Assumption 3.4, we have $( b , v ) = ( b _ { \alpha } ^ { \mathrm { m o m } } , v _ { \alpha } ^ { \mathrm { m o m } } ) { \mathrm { ~ a n d ~ } } c = c _ { \alpha } ^ { \mathrm { m o m } } : = 1 8 0 ( \frac 3 2 ) ^ { 2 - \alpha } \in [ 1 8 0 , 2 7 0 )$ . Under Assumption 3.5, we have $( b , v ) = ( b _ { \alpha } , v _ { \alpha } ( \Lambda ) ) \mathrm { a n d } c = c _ { \alpha } ( \Lambda ) : = 1 8 0 \operatorname* { m a x } \{ \bar { v _ { \alpha } } ( \Lambda ) , b _ { \alpha } / 3 \}$ , where Λ is the ratio of the clipping level to the average noise scale: for the tuned clipping level, $\Lambda = \Lambda _ { K } : = \operatorname* { m a x } \{ 8 0 ( K + 1 ) , \Lambda _ { 2 \sqrt { 2 } L R } \}$ where $\Lambda _ { \lambda }$ is the explicit ratio (29) from Appendix F $( \Lambda _ { \lambda } = ( 3 \lambda / ( 2 \sigma _ { 0 } ) ) ^ { \alpha } \mathrm { i f } \sigma _ { 1 } = 0$ and $\sigma _ { 0 } > 0 )$ . For $\alpha < 2 , c _ { \alpha } ( \Lambda )$ is bounded by $c _ { \alpha } ( \infty ) = \operatorname* { m a x } \{ 1 8 0 v _ { \alpha } ( \infty ) , 6 0 b _ { \alpha } \}$ , which depends only on $\alpha ,$ whereas $c _ { 2 } ( \Lambda ) = 1 8 0 ( 1 + \ln ( 1 + \Lambda ) )$ .

Theorem 5.1 (High-probability convergence of Clip-SGD). Let Assumptions 3.1, 3.2, and 3.3 hold for $Q = B _ { 2 R } ( x ^ { * } ) , R \geq \| x ^ { 0 } - x ^ { * } \| > 0$ and let either Assumption 3.4 or Assumption 3.5 hold with $\alpha \in ( 1 , 2 ]$ Let c be defined as above with $\Lambda = \Lambda _ { K } , \beta \in ( 0 , 1 ) , K \geq 0 ,$ and $\begin{array} { r } { \ell : = \ln \frac { 4 ( K + 1 ) } { \beta } } \end{array}$ Then there exists an explicit stepsize $\gamma = \gamma _ { * } ( K , \beta )$ (see Theorem F.1 in Appendix F) such that with probability at least $1 - \beta$ the iterates produced by Algorithm 1 with $\begin{array} { r } { \lambda _ { k } \equiv \lambda : = \frac { R } { 4 0 \gamma \ell } } \end{array}$ satisfy $x ^ { k } \in B _ { \sqrt { 2 } R } ( x ^ { * } )$ for all $k = 0 , 1 , \ldots , K + 1$ and

$$
f ( \bar { x } ^ { K } ) - f ^ { * } \leq \operatorname* { m a x } \Bigg \{ \frac { 1 6 0 \sqrt { 2 } L R ^ { 2 } \ell } { K + 1 } , ~ \frac { 2 \left( c 4 0 ^ { \alpha - 1 } \right) ^ { \frac { 1 } { \alpha } } \sigma _ { 0 } R \ell ^ { \frac { \alpha - 1 } { \alpha } } } { ( K + 1 ) ^ { \frac { \alpha - 1 } { \alpha } } } , ~ \frac { 2 \left( c 2 ^ { \alpha } 4 0 ^ { \alpha - 1 } \right) ^ { \frac { 2 } { \alpha } } \sigma _ { 1 } ^ { 2 } L R ^ { 2 } \ell ^ { \frac { 2 ( \alpha - 1 ) } { \alpha } } } { ( K + 1 ) ^ { \frac { 2 ( \alpha - 1 ) } { \alpha } } } \Bigg \} .\tag{7}
$$

The full statement (Theorem F.1) gives the bound $2 R ^ { 2 } / ( \gamma ( K + 1 ) )$ for every admissible stepsize $\gamma \le \gamma _ { * }$ Choosing $\gamma = \gamma _ { * }$ yields (7); Corollary F.2 gives the oracle complexity summarized in Table 1. Writing $r = ( \alpha - 1 ) / \alpha$ , the three terms in the rate have polynomial orders $K ^ { - 1 } , K ^ { - r }$ , and $K ^ { - 2 r }$ : they correspond to the deterministic error, additive noise, and gradient-dependent noise, respectively.

The horizon logarithm in ℓ comes from a union bound. A maximal inequality replaces ℓ by $\ln ( 4 / \beta )$ under the moment condition and under the weak-tail condition for $\alpha < 2$ with $v = v _ { \alpha } ( \infty )$ (Theorem I.18; see also Nguyen et al., 2023). Proposition H.3 establishes optimality of this confidence dependence for the additive stochastic term. The same replacement applies to Clip-SSTM. For prescribed clipping levels, it is valid under either noise condition for every $\alpha \in ( 1 , 2 ]$ , because the clipping level does not depend on ℓ.

Discussion. Under Assumption 3.4 with $\sigma _ { 1 } = 0 _ { \it { \Delta } }$ , the stochastic term $\sigma _ { 0 } R K ^ { - ( \alpha - 1 ) / \alpha }$ cannot be improved, up to the logarithmic factor and the constant, by any algorithm making $K + 1$ oracle calls, even if one asks for the guarantee with constant probability: Proposition H.2 (Appendix H) provides a convex L-smooth instance on R with $\| x ^ { 0 } - x ^ { * } \| = R$ and $\mathbb { E } | \zeta | ^ { \alpha } \leq \sigma _ { 0 } ^ { \alpha }$ such that the error is at least ${ \scriptstyle \frac { 1 } { 4 } } \operatorname* { m i n } \{ \sigma _ { 0 } R ( K + 1 ) ^ { - ( \alpha - 1 ) / \alpha } , 2 L R ^ { 2 } \}$ with probability at least $1 / 4 ;$ the proof relies on the two-point argument of Bubeck et al. (2013). Since Assumption 3.4 implies Assumption 3.5 with the same parameters, the same lower bound holds for the larger class as well. That is, for $\alpha < 2$ the weaker assumption costs nothing in the rate and afects only the α-dependent constants. $\mathrm { { N e x t } , }$ the terms depending on $\sigma _ { 1 }$ decay faster in K than the σ -term; for $\alpha = 2$ under Assumption 3.4, the last term in (7) is of the order $\sigma _ { 1 } ^ { 2 } L R ^ { 2 } \ell / K$ , which together with the first term recovers the well-known $( 1 + \sigma _ { 1 } ^ { 2 } ) L R ^ { 2 } / K$ scaling under afine variance.

The endpoint $\alpha = 2$ and the Pareto example. Under Assumption 3.5 with $\alpha = 2$ , the noise may have infinite variance at every point (Section 3), meaning that the analyses relying on the bounded second moment are not applicable. Nevertheless, Theorem 5.1 gives the rate of the finitevariance case, max $\{ \frac { L R ^ { 2 } \ell } { K } , \sigma _ { 0 } R \sqrt { \frac { c \ell } { K } } , \frac { c \sigma _ { 1 } ^ { 2 } L R ^ { 2 } \ell } { K } \}$ up to absolute constants, with the logarithmic factor $c = c _ { 2 } ( \Lambda _ { K } ) = O ( \ln ( K + 1 ) + \mathrm { \dot { l n } } ( 1 + \Lambda _ { 2 \sqrt { 2 } L R } ) )$ ; the same holds for Clip-SSTM (Theorem 5.2). A logarithmic factor is unavoidable both in the clipping lemma (Remark D.3) and in the rate: at a fixed confidence level, no algorithm using K stochastic gradients can avoid the factor $\sqrt { \ln ( e K ) }$ in the additive term over the weak- $L ^ { 2 }$ class, and, more generally, the factor $[ \ln ( e K ) ] ^ { 1 / \dot { 2 } - 1 / \dot { s } }$ on the Lorentz scale $L ^ { 2 , s }$ (Proposition I.16 in Appendix I.8; this order is attained, see Corollary I.19). Therefore, for the Pareto example from Section 3, the weak- $L ^ { \alpha }$ analysis gives the exponent $( \alpha - 1 ) / \alpha$ with a K-independent constant for $\alpha < 2$ and the finite-variance exponent $1 / 2$ up to a logarithmic factor for $\alpha = 2$ , whereas the moment-based analysis can be applied only with some $q < \alpha$ and $\begin{array} { r } { \sigma _ { 0 } = ( \frac { \alpha } { \alpha - q } ) ^ { 1 / q } \sigma { : } } \end{array}$ a fixed $q$ leads to the slower exponent $( q - 1 ) / q$ , and choosing $q  \alpha$ with the growth of the horizon recovers the exponent $( \alpha - 1 ) / \alpha$ only up to a factor of the order $( \ln K ) ^ { 1 / \alpha }$ (see Corollary F.3 in Appendix F).

Algorithm 2 Clip-SSTM   
Require: starting point $x ^ { 0 }$ , number of iterations $K \geq 1$ , parameter $a \geq 1$ , clipping levels $\{ \lambda _ { k } \} _ { k = 0 } ^ { K - 1 }$   
1: Set $A _ { 0 } = 0$ and $y ^ { 0 } = z ^ { 0 } = x ^ { 0 }$ (the supplied starting point)   
2: for $k = 0 , 1 , \ldots , K - 1$ do   
3: $\begin{array} { r } { \omega _ { k + 1 } = \frac { k + 2 } { 2 a L } , A _ { k + 1 } = A _ { k } + \omega _ { k + 1 } } \end{array}$   
4: $x ^ { k + 1 } = ( \widetilde { A _ { k } } y ^ { k } + \omega _ { k + 1 } z ^ { k } ) / A _ { k + 1 }$   
5: Sample $\xi ^ { k } \sim \mathcal { D }$ independently of $\mathcal { G } _ { k } ;$ set $\widetilde { g } _ { k } = { \mathsf { c l i p } } ( g ( x ^ { k + 1 } , \xi ^ { k } ) , \lambda _ { k } )$   
6: $z ^ { k + 1 } = z ^ { k } - \omega _ { k + 1 } \widetilde { g } _ { k }$   
7: $y ^ { k + 1 } = ( A _ { k } y ^ { k } + \omega _ { k + 1 } z ^ { k + 1 } ) / A _ { k + 1 }$   
8: end for   
Ensure: $y ^ { K }$

## 5.1 Acceleration: Clip-SSTM

Next, we consider the accelerated version of Clip-SGD called Clipped Stochastic Similar Triangles Method (Clip-SSTM) (Gorbunov et al., 2020; Sadiev et al., 2023), see Algorithm 2. For a parameter $a \ge 1$ , the method uses the weights $\begin{array} { r } { \omega _ { k + 1 } : = \frac { k + 2 } { 2 a L } } \end{array}$ and $A _ { k + 1 } : = A _ { k } + \omega _ { k + 1 } .$ , so that $A _ { k } = { \frac { k ( k { + } 3 ) } { 4 a L } }$ We analyze Clip-SSTM under either of the two noise models with $\alpha \in ( 1 , 2 ]$ and with the clipping constants (b, v) from Section $5 ;$ under Assumption 3.5, $v = v _ { \alpha } ( \Lambda _ { K } ^ { \mathrm { S } } )$ for an explicit $\Lambda _ { K } ^ { \mathrm { S } } \geq 2 0 0 b K$ defined in (61) (Appendix I), which plays the same role as $\Lambda _ { K }$ in Theorem 5.1.

Theorem 5.2 (High-probability convergence of Clip-SSTM). Let Assumptions 3.1, 3.2, and 3.3 hold for $Q = B _ { 3 R } ( x ^ { * } ) , R \geq \| x ^ { 0 } - x ^ { * } \| > 0$ and let either Assumption 3.4 or Assumption 3.5 hold with $\alpha \in ( 1 , 2 ]$ . Let $K \geq 1$ be an integer, $\beta \in ( 0 , 1 ]$ $H : = \ln \frac { 4 K } { \beta }$ , and $r : = { \frac { \alpha - 1 } { \alpha } }$ Then there exist an explicit parameter $a _ { * } = a _ { * } ( K , \beta )$ and constants $c _ { 0 , \alpha } , c _ { 1 , \alpha } ,$ determined by $( b , v )$ and α (see Theorem I.1 in Appendix I), such that on one event of probability at least $1 - \beta$ all points $x ^ { k } , y ^ { k } , z ^ { k } , k = 0 , \dots , K$ , produced by Algorithm 2 with $a = a _ { * }$ and $\begin{array} { r } { \lambda _ { k } = \frac { R } { 3 0 \omega _ { k + 1 } H } } \end{array}$ lie in $B _ { 2 R } ( x ^ { * } )$ and

$$
f ( y ^ { K } ) - f ^ { * } \leq \operatorname* { m a x } \Big \{ \frac { 1 0 5 8 4 0 0 L R ^ { 2 } H ^ { 2 } } { K ^ { 2 } } , \frac { 1 2 c _ { 0 , \alpha } \sigma _ { 0 } R H ^ { r } } { K ^ { r } } , \frac { 6 c _ { 1 , \alpha } \sigma _ { 1 } ^ { 2 } L R ^ { 2 } H ^ { 2 r } } { K ^ { 2 r } } \Big \} .\tag{8}
$$

Under Assumption 3.4, and under Assumption 3.5 with $\alpha \ : < \ : 2$ (if one computes $v = v _ { \alpha } ( \infty )$ which is allowed), the constants ${ \mathit { c } } _ { 0 , \alpha }$ and $c _ { 1 , \alpha }$ depend only on α (at $\alpha = 2$ under Assumption 3.4, $c _ { 0 , 2 } = \sqrt { 2 7 0 0 }$ and $c _ { 1 , 2 } = 4 9 \cdot 5 4 0 0 )$ ; at $\alpha = 2$ under Assumption $3 . 5 , c _ { 0 , 2 } = O ( \sqrt { v } )$ and $c _ { 1 , 2 } = O ( v )$ with $v = 1 + \ln ( 1 + \Lambda _ { K } ^ { \mathrm { S } } ) = O ( \ln ( e K ) + \ln ( 1 + \Lambda _ { K } ^ { 0 } ) )$ , where $\Lambda _ { K } ^ { 0 }$ is defined in (61).

The oracle complexity implied by (8) is given in Table 1 (see Corollary I.2). When $\sigma _ { 1 } = 0$ bound (8) has the same powers of K as the moment-based results of Gorbunov et al. (2020) (α = 2)

and Sadiev et al. (2023), but it holds for the larger class $L ^ { \alpha , \infty , }$ ; at $\alpha = 2$ under Assumption 3.5, it gives the finite-variance rates for the noise whose variance may be infinite, with the logarithmic factor v, which Proposition I.13 (Appendix I.8) refines on the Lorentz scale $L ^ { 2 , s }$

Role of the direct Lorentz analysis. For fixed $L , R , \sigma _ { 0 } > 0$ , additive noise, and fixed con fidence, the moment-embedding route in Corollary I.19 improves the direct Lorentz bound by a factor $\sqrt { \ln ( e K ) }$ . Its additive term matches the lower bound in Proposition I.16 in the stochastic regime. The direct analysis remains useful because it gives explicit finite-horizon parameters under the original afine Lorentz condition. Appendix I.8 gives the detailed comparison. Neither result establishes optimality of the multiplicative term or of the joint dependence on K and a vanishing failure probability.

Why the afine structure is preserved. The key idea of the proof is to bound the true gradients along the trajectory by induction via the function gap: on the induction event, the potential of SSTM gives $A _ { t } ( f ( y ^ { t } ) - f ^ { * } ) \leq \frac { 3 } { 2 } R ^ { 2 }$ , and Assumptions 3.1–3.2 turn this into $\omega _ { t + 1 } \| \nabla f ( x ^ { t + 1 } ) \| \ \leq$ $7 R / { \sqrt { a } }$ , which implies $\begin{array} { r } { \sum _ { t < T } \omega _ { t + 1 } ^ { \alpha } \| \nabla f ( x ^ { t + 1 } ) \| ^ { \alpha } \leq 7 ^ { \alpha } R ^ { \alpha } T / a ^ { \alpha / 2 } } \end{array}$ (see Lemma I.5 in Appendix I.4), whereas the crude bound $\| \nabla f ( x ^ { t + 1 } ) \| \leq 2 L R$ gives only $R ^ { \alpha } K ^ { \alpha + 1 } / a ^ { \alpha }$

This weighted estimate preserves the gradient-dependent noise scale throughout the clipping analysis. As a result, the multiplicative contribution is bounded by $\sigma _ { 1 } ^ { 2 } L R ^ { 2 } ( { \bar { H } } / { K } ) ^ { 2 r }$ , compared with $\sigma _ { 1 } L R ^ { 2 } ( H / K ) ^ { r }$ when the noise is bounded uniformly on $Q$ (Corollary I.9). This improves the bound’s dependence on $K ;$ it does not establish optimality of the multiplicative term. Faster rates are known under additional multiplicative-noise structure (Appendix I.7).

## 5.2 Prescribed Clipping Levels

The clipping levels in Theorems 5.1 and 5.2 depend on the parameters of the problem and on the horizon, whereas in practice λ is a constant chosen by the user. In this section, we allow the user to prescribe λ. Theorem 5.3 bounds the error of the computable averaged iterate $\bar { x } ^ { K }$ for any $\lambda \ge 2 \sqrt { 2 } L R ;$ ; this is a geometric condition guaranteeing that the true gradient is never clipped in $B _ { \sqrt { 2 } R } ( x ^ { * } )$ , and it involves neither $\beta$ nor $K$ (nor the noise, except for the case $\alpha = 2$ with $\sigma _ { 1 } > 0 )$ Theorem 5.4 covers any $\lambda > 0$ (for additive noise or for multiplicative noise below an absolute threshold; under Assumption 3.5, only for $\alpha < 2 )$ , but it bounds the best iterate $\mathrm { m i n } _ { t \le K } f ( x ^ { t } )$ which cannot be selected using the gradient information only. In both results, only the clipping level is free: the stepsizes still depend on the noise and confidence parameters. Below, $( b , v )$ are the clipping constants from Section 5 with $\Lambda = \Lambda _ { \lambda }$ under Assumption 3.5; only b afects the size of the neighborhood.

Theorem 5.3 (Clip-SGD with a prescribed clipping level). Let Assumptions 3.1, 3.2, and $3 . 3$ hold for $Q = B _ { 2 R } ( x ^ { * } ) , R \geq \| x ^ { 0 } - x ^ { * } \| > 0$ , and let either Assumption 3.4 or Assumption 3.5 hold with $\alpha \in ( 1 , 2 ]$ . Let $\beta \in ( 0 , 1 ) , K \geq 0 , \ell : = \ln \frac { 4 ( K + 1 ) } { \beta }$ and let $\lambda \ge 2 \sqrt { 2 } L R$ be arbitrary (with $\lambda \ge 2 4 0 b \sigma _ { 1 } ^ { 2 } L R$ if $\alpha = 2$ and $\sigma _ { 1 } > 0 )$ . Then there exists an explicit stepsize $\begin{array} { r } { \gamma = \gamma ( \lambda , K , \beta ) \le \frac { R } { 4 0 \lambda \ell } } \end{array}$ (see Corollary F.5 in Appendix F) such that with probability at least $1 - \beta$ the iterates produced by Algorithm 1 with $\lambda _ { k } \equiv \lambda$ satisfy $x ^ { k } \in B _ { \sqrt { 2 } R } ( x ^ { * } )$ for all $k = 0 , 1 , \ldots , K + 1$ and

$$
f ( \bar { x } ^ { K } ) - f ^ { * } \le \operatorname* { m a x } \left\{ \frac { 8 0 \lambda R \ell } { K + 1 } , \mathcal { N } _ { \lambda } , \mathcal { V } _ { \lambda , K } \right\} ,\tag{9}
$$

where the neighborhood

$$
\mathcal { N } _ { \lambda } : = \operatorname* { m a x } \left\{ \frac { 1 2 0 b R \sigma _ { 0 } ^ { \alpha } } { \lambda ^ { \alpha - 1 } } , 2 \left( \frac { 6 0 \cdot 2 ^ { \alpha } b R L ^ { \frac { \alpha } { 2 } } \sigma _ { 1 } ^ { \alpha } } { \lambda ^ { \alpha - 1 } } \right) ^ { \frac { 2 } { 2 - \alpha } } \right\}\tag{10}
$$

does not depend on K (its second term is absent for $\alpha = 2 )$ , whereas $\mathcal { V } _ { \lambda , K }$ vanishes as $K  \infty :$ for $\sigma _ { 1 } = 0 , \mathcal { V } _ { \lambda , K } = 1 2 0 \sqrt { 2 } R \sqrt { v \lambda ^ { 2 - \alpha } \sigma _ { 0 } ^ { \alpha } \ell / ( K + 1 ) }$ , and for $\sigma _ { 1 } > 0$ it has three more terms, of the orders $K ^ { - 1 } , K ^ { - 1 }$ and $K ^ { - 2 / ( 4 - \alpha ) }$ up to logarithmic factors. If $\sigma _ { 1 } = 0$ and $\lambda = \lambda _ { 0 } ( K + 1 ) ^ { 1 / \alpha }$ with $\lambda _ { 0 } > 0$ and $K + 1 \ge ( 2 \sqrt { 2 } L R / \lambda _ { 0 } ) ^ { \alpha }$ , then, with probability at least $1 - \beta , f ( \bar { x } ^ { K } ) - f ^ { * } \leq \Xi R ( K + 1 ) ^ { - ( \alpha - 1 ) / \alpha }$ with $\Xi : = 8 0 \lambda _ { 0 } \ell + 1 2 0 b \sigma _ { 0 } ^ { \alpha } \lambda _ { 0 } ^ { 1 - \alpha } + 1 2 0 \sqrt { 2 v \ell } \sigma _ { 0 } ^ { \alpha / 2 } \lambda _ { 0 } ^ { 1 - \alpha / 2 }$

Theorem 5.3 concerns a separate run for each horizon K, with the stepsize chosen for that horizon. As K increases, the bound approaches ${ \mathcal { N } } _ { \lambda } ;$ the slowest vanishing term has order $K ^ { - 1 / 2 }$ up to logarithms. The additive part of $\mathcal { N } _ { \lambda }$ is proportional to the clipping-bias constant b from Lemma 4.1; the factor 120 comes from the optimization analysis. This neighborhood does not depend on the variance constant v, including its logarithm at $\alpha = 2$ . Its dependence on $R , \ \sigma _ { 0 }$ and λ is unimprovable up to α-dependent constants in the nontrivial regime (Theorem 5.5). The multiplicative neighborhood decreases faster in λ.

For $\sigma _ { 1 } = 0$ and $\lambda = \lambda _ { 0 } ( K + 1 ) ^ { 1 / \alpha }$ , any $\lambda _ { 0 } > 0$ recovers the polynomial rate from Theorem 5.1; Appendix F.1 quantifies the cost of misspecifying the noise level. Under the moment condition, the non-private specialization of Khah et al. (2026) gives the same three-term order for the best iterate when $\lambda > 4 L R$ , after combining it with the initial-point bound $f ( x ^ { 0 } ) - f ^ { * } \leq L R ^ { 2 } / 2$ (see the discussion of Table 3).

Arbitrary clipping levels. When $\lambda < 2 \sqrt { 2 } L R$ , the true gradient may be clipped along the trajectory, and the proof of Theorem 5.3 does not work. The next result covers any $\lambda > 0$ , for additive noise and for multiplicative noise below an absolute threshold. Its neighborhood depends on λ only through the bias scale

$$
\bar { b } _ { \lambda } : = \frac { \sigma _ { 0 } ^ { \alpha } } { \operatorname* { m a x } \{ \lambda , \sigma _ { 0 } \} ^ { \alpha - 1 } } ,\tag{11}
$$

which equals $\sigma _ { 0 } ^ { \alpha } / \lambda ^ { \alpha - 1 }$ for $\lambda \ge \sigma _ { 0 }$ and saturates at $\sigma _ { 0 }$ for $\lambda < \sigma _ { 0 }$ . Let $( \kappa _ { 1 } , \kappa _ { 2 } ) : = ( 1 , 1 )$ under Assumption 3.4 and $\begin{array} { r } { ( \kappa _ { 1 } , \kappa _ { 2 } ) : = \left( \frac { \alpha } { \alpha - 1 } , \frac { 2 } { 2 - \alpha } \right) } \end{array}$ under Assumption 3.5 with $\alpha \ : < \ : 2$ , and let $c _ { \perp } : =$ $2 ^ { \alpha - 1 } ( \kappa _ { 2 } + 3 \kappa _ { 1 } )$ and $\begin{array} { r } { \bar { \sigma } _ { 1 } : = \frac { 1 } { 2 \cdot 2 8 ^ { 1 / \alpha } } } \end{array}$

Theorem 5.4 (Any clipping level). Let Assumptions 3.1, 3.2, and 3.3 hold for $Q = B _ { 2 R } ( x ^ { * } )$ $R \geq \| x ^ { 0 } - x ^ { * } \| > 0$ , and let Assumption 3.4 (with $\alpha \in ( 1 , 2 ] )$ or Assumption 3.5 (with $\alpha \in ( 1 , 2 ) )$ hold with $\sigma _ { 1 } \leq \bar { \sigma } _ { 1 }$ Let $\lambda > 0$ be arbitrary, $\beta \in ( 0 , 1 ) , \ K \geq 0 .$ and $\ell : = \dot { \ln \frac { 4 ( K + 1 ) } { \beta } }$ Then there exists an explicit stepsize $\gamma = \gamma ( \lambda , K , \beta )$ (see Theorem G.1 in Appendix G) such that with probability at least $1 - \beta$ the iterates produced by Algorithm 1 with $\lambda _ { k } \equiv \lambda$ satisfy

$$
\operatorname* { m i n } _ { 0 \leq t \leq K } f ( x ^ { t } ) - f ^ { * } \leq \operatorname* { m a x } \left\{ \varepsilon _ { N } , \varepsilon _ { M } , \mathcal { V } _ { \lambda , K } ^ { \prime } \right\} ,\tag{12}
$$

where $\varepsilon _ { N } : = 4 8 \sqrt { 2 } c _ { \perp } R \bar { b } _ { \lambda } , \varepsilon _ { M } : = 3 8 4 c _ { \perp } \sigma _ { 1 } ^ { \alpha } L R ^ { 2 } \operatorname* { m i n } \{ 1 , 9 6 c _ { \perp } \sigma _ { 1 } ^ { \alpha } \}$ , and $\mathcal { V } _ { \lambda , K } ^ { \prime }$ collects the terms vanishing as $K  \infty$ , of the orders $K ^ { - 1 / 2 } , K ^ { - 1 }$ and $K ^ { - 2 }$ up to logarithmic factors, which are given explicitly in Theorem G.1.

Theorem 5.4 bounds the first time an iterate reaches the stated neighborhood, so its stepsize need not decrease as $1 / K$ . The tuning still depends on the horizon. The guarantee concerns the best iterate in objective value, as in Khah et al. (2026); selecting an output with the same guarantee requires function values or an additional validation procedure.

For $\sigma _ { 1 } > 0$ , the bound has the additional term $\varepsilon _ { M } = O ( \sigma _ { 1 } ^ { 2 \alpha } L R ^ { 2 } )$ , which does not decrease in λ. The key new ingredient of the proof is a clipping lemma for arbitrary clipping levels (Lemma G.2): when $\| \nabla f ( x ) \| > \lambda / 2$ and $\| \nabla f ( x ) \| \gtrsim \sigma _ { 0 }$ , the mean clipped gradient has a component of size at least min $\{ \| \nabla f ( x ) \| , \lambda \} / 6$ along $\nabla f ( x )$ , provided that $\sigma _ { 1 } \leq \bar { \sigma } _ { 1 }$ , and an orthogonal component bounded by a quantity of order $\lambda A ( \nabla f ( x ) ) / \operatorname* { m a x } \{ \| \nabla f ( x ) \| , \lambda \} ^ { \alpha }$ . The analysis of Khah et al. (2026) compares the mean clipped gradient with ${ \mathsf { c l i p } } ( \nabla f ( x ) , \lambda / 2 )$ . It covers arbitrary $\lambda > 0$ , with additional terms involving $\zeta _ { \lambda } : = \operatorname* { m a x } \{ 0 , 2 L R - \lambda / 2 \}$ when $\lambda < 4 L R$ . Table 3 compares the regime $\lambda > 4 L R$ , where $\zeta _ { \lambda } = 0$ . We also notice that some restriction on $\sigma _ { 1 }$ is necessary for small clipping levels: for the multiplicative oracle of Faw et al. (2023) $( \sigma _ { 0 } = 0 , \sigma _ { 1 } \le 9 )$ on $\begin{array} { r } { f ( x ) = \frac { L } { 2 } x ^ { 2 } } \end{array}$ , Clip-SGD with any fixed $\lambda > 0$ and $\gamma _ { k } \le 1 / L$ started from $x ^ { 0 } = \lambda / L$ stays at distance $\geq \lambda / ( 2 L )$ from $x ^ { * }$ forever with probabili $\mathrm { { y } \geq 2 / 3 }$ (Proposition H.6).

## 5.3 Lower Bounds for a Fixed Clipping Level

The next result shows that the additive-noise neighborhood from Theorem 5.4, and hence the first part of $\mathcal { N } _ { \lambda }$ in Theorem 5.3, cannot be improved for any $\lambda > 0$ , up to the factors depending on α and the trivial level $L R ^ { 2 }$ . More precisely, for any $\lambda > 0$ we construct a one-dimensional problem with additive noise such that Clip-SGD with $\lambda _ { k } \equiv \lambda$ and any deterministic stepsizes does not approach the solution.

Theorem 5.5 (Lower bound for a fixed clipping level). Let $L , R , \sigma _ { 0 } , \lambda > 0 , \alpha \in ( 1 , 2 ]$ , and let $\begin{array} { r } { \varepsilon _ { \lambda } : = \frac { 1 } { 4 8 \cdot 4 ^ { \alpha } } } \end{array}$ min $\{ R \bar { b } _ { \lambda } , L R ^ { 2 } \}$ with $\bar { b } _ { \lambda } = \operatorname* { m i n } \{ \sigma _ { 0 } , \sigma _ { 0 } ^ { \alpha } / \lambda ^ { \alpha - 1 } \}$ from (11). Then there exist a convex L-smooth function $f : \mathbb { R } \to \mathbb { R }$ with minimizer $x ^ { * } = 0 ,$ , a point $x ^ { 0 }$ with $| x ^ { 0 } - x ^ { * } | = R$ , and an oracle $g ( x , \xi ) = f ^ { \prime } ( x ) + \zeta$ with $\mathbb { E } \zeta = 0$ and $\mathbb { E } | \zeta | ^ { \alpha } \leq \sigma _ { 0 } ^ { \alpha } \mathrm { ( i . e . }$ , Assumptions 3.1–3.5 hold on R with $\sigma _ { 1 } = 0 )$ such that Clip-SGD with $\lambda _ { k } \equiv \lambda$ and any deterministic stepsizes $\gamma _ { k } > 0$ satisfies $\mathbb { E } f ( \hat { x } ) - f ^ { * } \geq \varepsilon _ { \lambda }$ for every iterate ${ \hat { x } } = x ^ { k }$ and every convex combination ˆx of $x ^ { 0 } , \ldots , x ^ { K }$ with deterministic weights, in particular, for $\bar { x } ^ { K }$ ; moreover, $\mathbb { E } f ( x ^ { k } ) - f ^ { * }$ grows linearly in $\begin{array} { r } { \sum _ { j < k } \gamma _ { j } . \mathrm { ~ I f ~ } \gamma _ { k } \le \frac { R } { 1 6 \lambda } } \end{array}$ for all $k ,$ then, with probability at least $3 / 4 , f ( x ^ { k } ) - f ^ { * } \geq \varepsilon _ { \lambda } / 2$ and $f ( \bar { x } ^ { k } ) - f ^ { * } \geq \varepsilon _ { \lambda } / 2$ for all $k \geq 0$ . The explicit growth rate is given in Theorem H.1 $( \mathrm { A p p e n d i x ~ H } )$

Comparison with the upper bounds. Let $\sigma _ { 1 } = 0 .$ , and assume that the conditions of the upperbound result under consideration are satisfied. Let $\bar { b } _ { \lambda } = \operatorname* { m i n } \{ \sigma _ { 0 } , \sigma _ { 0 } ^ { \alpha } / \lambda ^ { \alpha - 1 } \}$ . Theorem 5.5 gives the lower bound $\varepsilon _ { \lambda } = ( 4 8 \cdot 4 ^ { \alpha } ) ^ { - 1 } \operatorname* { m i n } \{ R \bar { b } _ { \lambda } , L R ^ { 2 } \}$ in expectation for every iterate and every deterministic average under arbitrary deterministic stepsizes; if $\gamma _ { k } \leq R / ( 1 6 \lambda )$ , it also gives the lower bound $\varepsilon _ { \lambda } / 2$ with probability at least $3 / 4$ , simultaneously for all iterates and their averages. This restriction on the stepsizes is satisfied by the tuning $\gamma \leq R / ( 4 0 \lambda \ell )$ used in the upper bounds. In the regime $R \bar { b } _ { \lambda } \leq L R ^ { 2 }$ , the best-iterate neighborhood $\varepsilon _ { N } = 4 8 \sqrt { 2 } c _ { \perp } R \bar { b } _ { \lambda }$ from Theorem $5 . 4 .$ , at the confidence level $\beta < 1 / 4$ , matches this high-probability lower bound up to the factor $2 \cdot 4 8 ^ { 2 } \sqrt { 2 } \cdot 4 ^ { \alpha } c _ { \perp }$ , which depends only on α. For the averaged iterate, we impose $\lambda \ge 2 \sqrt { 2 } L R$ and the nontrivial-regime condition $R \sigma _ { 0 } ^ { \alpha } / \lambda ^ { \alpha - 1 } \leq L R ^ { 2 }$ These conditions imply $\sigma _ { 0 } < \lambda$ , and, therefore, the neighborhood $1 2 0 b R \sigma _ { 0 } ^ { \alpha } / \lambda ^ { \alpha - 1 }$ from Theorem 5.3 matches the same lower bound up to the factor $1 1 5 2 0 \cdot 4 ^ { \alpha } b ;$ see Table 3 in Appendix H. Note that these comparisons concern only the neighborhoods: the lower bound does not depend on K and does not match the terms vanishing with the horizon.

The proof (Appendix H) relies on a Huber function whose slope is smaller than the clipping bias of a two-point noise, so that the mean clipped field has the same sign everywhere and pushes the iterates away from $x ^ { * }$ . Consequently, a uniform guarantee in expectation $\mathbb { E } f ( \bar { x } ^ { K } ) - f ^ { * } \leq \varepsilon$ over the class of problems with additive noise and parameters $( L , R , \sigma _ { 0 } , \alpha )$ , where $\varepsilon <$ min $\{ R \sigma _ { 0 } , L R ^ { 2 } \} / ( 4 8$ $4 ^ { \alpha } )$ , requires $\lambda \ge \sigma _ { 0 } ( \sigma _ { 0 } R / ( 4 8 \cdot 4 ^ { \alpha } \varepsilon ) ) ^ { 1 / ( \alpha - 1 ) }$ regardless of the deterministic stepsizes; on the other hand, for the high-probability criterion with $\lambda = \sigma _ { 0 } ( 1 2 0 b \sigma _ { 0 } R / \varepsilon ) ^ { 1 / ( \alpha - 1 ) }$ (if this value is at least $2 \sqrt { 2 } L R )$ , the neighborhood from Theorem 5.3 equals $\varepsilon ,$ and the bound becomes at most ε after $K =$ $O _ { \alpha } ( v \ell ( \bar { \sigma } _ { 0 } R / \varepsilon ) ^ { \alpha / ( \alpha - 1 ) } )$ ) iterations, which is the complexity from Table 1. Moreover, the expected error of the last iterate also grows linearly in the total stepsize, which makes the scale of the bias condition from Theorem 5.3 necessary for the averaged iterate with a constant stepsize, and the bias persists even on strongly convex quadratics (see Appendix H.1).

## 6 Numerical Experiments

We illustrate the efect of clipping under weak- $L ^ { \alpha }$ noise with an infinite α-th moment. On a quadratic problem with asymmetric noise, we compare the plateaus produced by prescribed clipping levels with the decreasing errors obtained using a horizon-dependent level. Appendix J reports additional experiments with symmetric noise, ridge regression, and the full theoretical parameter choices for Clip-SGD and Clip-SSTM.

Protocol. We use $\alpha = 3 / 2$ , 1024 independent runs for each configuration, and horizons $n \le 1 0 ^ { 6 }$ oracle calls, and we report the mean of $f ( \bar { x } ^ { n - 1 } ) - f ^ { * }$ together with pointwise approximate 95% confidence bands; the horizon-dependent configurations are rerun for each horizon, with $\beta = 0 . 0 5$ The noise is built from a Pareto random variable $P$ with $\mathbb { P } \{ P > t \} = t ^ { - \alpha }$ for $t \geq 1$ (so that $\mathbb { E } P = 3$ and $\mathbb { E } P ^ { \alpha } = \infty )$ : $g ( x ) = x + S ( x ) Z$ with the centered asymmetric $Z = ( P - 3 ) / 2$ and $S ( x ) = ( \sigma _ { 0 } ^ { \alpha } + \sigma _ { 1 } ^ { \alpha } | x | ^ { \alpha } ) ^ { 1 / \alpha }$ . This oracle satisfies the afine weak-tail inequality with $( \sigma _ { 0 } , \sigma _ { 1 } )$ , and its α-th moment is infinite whenever the noise scale is non-zero.

Influence of the clipping level. We consider the quadratic $f ( x ) = x ^ { 2 } / 2$ on R $( L = 1 , x ^ { * } = 0$ $x ^ { 0 } = 1 )$ with the asymmetric noise and $( \sigma _ { 0 } , \sigma _ { 1 } ) = ( 0 . 0 5 , 0 . 0 0 1 )$ , and compare the constant clipping levels $\lambda \in \{ 0 . 1 , 0 . 5 , 1 , 4 \}$ with the level $\lambda ( n ) : = R / ( 4 0 \gamma _ { * } ( n ) \ell _ { n } )$ from Theorem 5.1, computed for each horizon n with $\beta = 0 . 0 5$ and the stepsize $\gamma _ { * } ( n )$ from Theorem F.1; this level grows as $n ^ { 1 / \alpha }$ up to logarithmic factors. All variants use $\gamma = 0 . 0 5$ (see Figure 1).

![](images/a6740cea4415d67971775d891d80aaf579564c5f60f7172f9329ae9e1fc8d96e.jpg)  
Figure 1: Influence of the clipping level under centered asymmetric Pareto noise on $f ( x ) = x ^ { 2 } / 2$

Figure 1 shows distinct plateaus for constant clipping levels, with smaller plateaus at larger levels. The dotted lines show objective values at zeros of the mean clipped field and provide a reference for these plateaus. With $\lambda ( n )$ , the observed error continues to decrease over the tested horizons, consistently with the displayed reference slope and the heuristic $n ^ { - 2 / 3 }$ balance between squared bias and the variance of averaged clipped noise. Increasing the level reduces clipping bias while increasing the second-moment bound. All configurations use the empirical stepsize $\gamma = 0 . 0 5$ so this comparison isolates the clipping level rather than testing the theorem’s full parameter prescription.

## 7 Conclusion

We derived high-probability guarantees for Clip-SGD and Clip-SSTM under afine weak- $. L ^ { \alpha }$ noise. For $\alpha \ < \ 2$ the tail condition preserves the polynomial rates available under a bounded α-th moment. At $\alpha = 2$ , the additive term requires a logarithmic correction; our Lorentz-scale upper and lower bounds identify its optimal asymptotic order at fixed confidence and fixed positive problem parameters. The analysis also separates the additive and gradient-dependent noise contributions and gives guarantees for prescribed clipping levels. In the additive case, matching lower bounds characterize the resulting clipping-bias neighborhood up to constants and truncation at $L R ^ { 2 }$

Several questions remain open. These include lower bounds for the multiplicative term, sharper guarantees for small clipping levels under multiplicative noise, and the joint dependence on the horizon and the failure probability at the endpoint.

## AI use statement

The main results presented in this document were obtained by the authors. We used a generative AI assistant to improve the writing, strengthen the initial core results, review mathematical arguments, and check references. The authors take responsibility for the final content of this work.

## References

Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., and Zhang, L. (2016). Deep learning with diferential privacy. In Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security, pages 308–318.

Alimisis, F., Islamov, R., and Lucchi, A. (2026). Why do we need warm-up? a theoretical perspective. In Proceedings of the 43rd International Conference on Machine Learning.

Armacki, A., Sharma, P., Joshi, G., Bajovi´c, D., Jakoveti´c, D., and Kar, S. (2023). High-probability convergence bounds for nonlinear stochastic gradient descent under heavy-tailed noise. arXiv preprint arXiv:2310.18784.

Armacki, A., Yu, S., Bajovi´c, D., Jakoveti´c, D., and Kar, S. (2026). Large deviation upper bounds and improved MSE rates of nonlinear SGD: Heavy-tailed noise and power of symmetry. SIAM Journal on Optimization, 36(1):32–59.

Armacki, A., Yu, S., Sharma, P., Joshi, G., Bajovi´c, D., Jakoveti´c, D., and Kar, S. (2025). Highprobability convergence bounds for online nonlinear stochastic gradient descent under heavytailed noise. In Proceedings of the 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings of Machine Learning Research, pages 1774–1782. PMLR.

Astashkin, S. V. and Sukochev, F. A. (2010). Independent functions and the geometry of Banach spaces. Russian Mathematical Surveys, 65(6):1003–1081.

Attia, A. and Koren, T. (2023). SGD with AdaGrad stepsizes: Full adaptivity with high probability to unknown parameters, unbounded gradients and afine variance. In International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 1147–1171. PMLR.

Barsbey, M., Sefidgaran, M., Erdogdu, M., Richard, G., and S¸im¸sekli, U. (2021). Heavy tails in SGD and compressibility of overparametrized neural networks. In Advances in Neural Information Processing Systems, volume 34, pages 29364–29378.

Bennett, C. and Sharpley, R. C. (1988). Interpolation of operators, volume 129 of Pure and Applied Mathematics. Academic Press, Boston.

Bottou, L., Curtis, F. E., and Nocedal, J. (2018). Optimization methods for large-scale machine learning. SIAM Review, 60(2):223–311.

Braverman, M. S. (1994). Independent random variables and rearrangement invariant spaces, volume 194 of London Mathematical Society Lecture Note Series. Cambridge University Press.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. (2020). Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pages 1877–1901.

Bubeck, S., Cesa-Bianchi, N., and Lugosi, G. (2013). Bandits with heavy tail. IEEE Transactions on Information Theory, 59(11):7711–7717.

Chen, X., Wu, S. Z., and Hong, M. (2020). Understanding gradient clipping in private SGD: A geometric perspective. In Advances in Neural Information Processing Systems, volume 33, pages 13773–13782.

Chezhegov, S., Klyukin, Y., Semenov, A., Beznosikov, A., Gasnikov, A., Horv´ath, S., Tak´aˇc, M., and Gorbunov, E. (2025). Clipping improves Adam-Norm and AdaGrad-Norm when the noise is heavy-tailed. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 10269–10333. PMLR.

Cutkosky, A. and Mehta, H. (2021). High-probability bounds for non-convex stochastic optimization with heavy tails. In Advances in Neural Information Processing Systems, volume 34, pages 4883– 4895.

Das, A., Nagaraj, D., Pal, S., Suggala, A. S., and Varshney, P. (2024). Near-optimal streaming heavy-tailed statistical estimation with clipped SGD. In Advances in Neural Information Processing Systems, volume 37, pages 8834–8900.

Davis, D., Drusvyatskiy, D., Xiao, L., and Zhang, J. (2021). From low probability to high confidence in stochastic convex optimization. Journal of Machine Learning Research, 22(49):1–38.

Dzhaparidze, K. and Van Zanten, J. (2001). On Bernstein-type inequalities for martingales. Stochastic Processes and their Applications, 93(1):109–117.

Efron, B., Hastie, T., Johnstone, I., and Tibshirani, R. (2004). Least angle regression. The Annals of Statistics, 32(2):407–499. With discussion.

El Bakkali, T., Chayti, E. M., Zhang, Q., Rahali, I., and Saadi, O. (2026). Stochastic zeroth-order optimization under heavy-tailed noise. arXiv preprint arXiv:2605.17394.

El Bakkali El Kadi, T., Chezhegov, S., Beznosikov, A., Horv´ath, S., and Gorbunov, E. (2025). High-probability convergence of clipped SGD under heavy-tailed noise and (L<sub>0</sub>, L<sub>1</sub>)-smoothness. arXiv preprint arXiv:2505.20817.

Fatkhullin, I., H¨ubler, F., and Lan, G. (2025). Can SGD handle heavy-tailed noise? arXiv preprint arXiv:2508.04860.

Faw, M., Rout, L., Caramanis, C., and Shakkottai, S. (2023). Beyond uniform smoothness: A stopped analysis of adaptive SGD. In The Thirty Sixth Annual Conference on Learning Theory, volume 195 of Proceedings of Machine Learning Research, pages 89–160. PMLR.

Faw, M., Tziotis, I., Caramanis, C., Mokhtari, A., Shakkottai, S., and Ward, R. (2022). The power of adaptivity in SGD: Self-tuning step sizes with unbounded gradients and afine variance. In Conference on Learning Theory, volume 178 of Proceedings of Machine Learning Research, pages 313–355. PMLR.

Freedman, D. A. (1975). On tail probabilities for martingales. The Annals of Probability, 3(1):100– 118.

Gaash, O., Levy, K. Y., and Carmon, Y. (2025). Convergence of clipped SGD on convex $( L _ { 0 } , L _ { 1 } ) \ –$ smooth functions. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Gorbunov, E., Danilova, M., Dobre, D., Dvurechenskii, P., Gasnikov, A., and Gidel, G. (2022). Clipped stochastic methods for variational inequalities with heavy-tailed noise. In Advances in Neural Information Processing Systems, volume 35, pages 31319–31332.

Gorbunov, E., Danilova, M., and Gasnikov, A. (2020). Stochastic optimization with heavy-tailed noise via accelerated gradient clipping. In Advances in Neural Information Processing Systems, volume 33, pages 15042–15053.

Gorbunov, E., Danilova, M., Shibaev, I., Dvurechensky, P., and Gasnikov, A. (2024a). Highprobability complexity bounds for non-smooth stochastic convex optimization with heavy-tailed noise. Journal of Optimization Theory and Applications, 203(3):2679–2738.

Gorbunov, E., Sadiev, A., Danilova, M., Horv´ath, S., Gidel, G., Dvurechensky, P., Gasnikov, A., and Richt´arik, P. (2024b). High-probability convergence for composite and distributed stochastic minimization and variational inequalities with heavy-tailed noise. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 15951–16070. PMLR.

Gorbunov, E., Tupitsa, N., Choudhury, S., Aliev, A., Richt´arik, P., Horv´ath, S., and Tak´aˇc, M. (2025). Methods for convex (L<sub>0</sub>, L<sub>1</sub>)-smooth optimization: Clipping, acceleration, and adaptivity. In International Conference on Learning Representations.

Gower, R. M., Loizou, N., Qian, X., Sailanbayev, A., Shulgin, E., and Richt´arik, P. (2019). SGD: General analysis and improved rates. In International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 5200–5209. PMLR.

Gupta, K., Siegel, J. W., and Wojtowytsch, S. (2024). Nesterov acceleration despite very noisy gradients. In Advances in Neural Information Processing Systems, volume 37.

G¨urb¨uzbalaban, M., S¸im¸sekli, U., and Zhu, L. (2021). The heavy-tail phenomenon in SGD. In International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 3964–3975. PMLR.

He, C., Li, B., and Lu, Z. (2025). Accelerated stochastic first-order method for convex optimization under heavy-tailed noise. arXiv preprint arXiv:2510.11676. Version 2, September 2026.

Hodgkinson, L. and Mahoney, M. (2021). Multiplicative noise and heavy tails in stochastic optimization. In International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 4262–4274. PMLR.

Holland, M. J. and Ikeda, K. (2019). Eficient learning with robust gradient descent. Machine Learning, 108(8):1523–1560.

H¨ubler, F., Fatkhullin, I., and He, N. (2025). From gradient clipping to normalization for heavy tailed SGD. In Proceedings of the 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings of Machine Learning Research, pages 2413–2421. PMLR.

Jakoveti´c, D., Bajovi´c, D., Sahu, A. K., Kar, S., Miloˇsevi´c, N., and Stamenkovi´c, D. (2023). Nonlinear gradient mappings and stochastic optimization: A general framework with applications to heavy-tail noise. SIAM Journal on Optimization, 33(2):394–423.

Jiao, Z. and Keller-Ressel, M. (2024). Emergence of heavy tails in homogenized stochastic gradient descent. In Advances in Neural Information Processing Systems, volume 37, pages 14066–14092.

Khah, S. V., Chezhegov, S., Farahmand, S., Horv´ath, S., and Gorbunov, E. (2026). Diferentially private clipped-SGD: High-probability convergence with arbitrary clipping level. In Proceedings of the 29th International Conference on Artificial Intelligence and Statistics, volume 300 of Proceedings of Machine Learning Research, pages 4762–4770. PMLR.

Khaled, A. and Richt´arik, P. (2023). Better theory for SGD in the nonconvex world. Transactions on Machine Learning Research.

Kikuchi, M. (1999). Averaging operators and martingale inequalities in rearrangement invariant function spaces. Canadian Mathematical Bulletin, 42(3):321–334.

Kikuchi, M. (2004). New martingale inequalities in rearrangement-invariant function spaces. Proceedings of the Edinburgh Mathematical Society, 47(3):633–657.

Koloskova, A., Hendrikx, H., and Stich, S. U. (2023). Revisiting gradient clipping: Stochastic bias and tight convergence guarantees. In International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 17343–17363. PMLR.

Kornilov, N., Dorn, Y., Lobanov, A., Kutuzov, N., Shibaev, I., Gorbunov, E., Nazin, A., and Gasnikov, A. (2024). Median clipping for zeroth-order non-smooth convex optimization and multiarmed bandit problem with heavy-tailed symmetric noise. arXiv preprint arXiv:2402.02461.

Kornilov, N., Shamir, O., Lobanov, A., Dvinskikh, D., Gasnikov, A., Shibaev, I., Gorbunov, E., and Horv´ath, S. (2023). Accelerated zeroth-order method for non-smooth stochastic convex optimization problem with infinite variance. In Advances in Neural Information Processing Systems, volume 36, pages 64083–64102.

Kornilov, N., Zmushko, P., Semenov, A., Ikonnikov, M., Gasnikov, A., and Beznosikov, A. (2025). Sign operator for coping with heavy-tailed noise in non-convex optimization: High probability bounds under (L<sub>0</sub>, L<sub>1</sub>)-smoothness. arXiv preprint arXiv:2502.07923.

Kre˘ın, S. G., Petunin, Y. I., and Semenov, E. M. (1982). Interpolation of linear operators, volume 54 of Translations of Mathematical Monographs. American Mathematical Society.

Lee, S. H., Zaheer, M., and Li, T. (2025). Eficient distributed optimization under heavy-tailed noise. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 33833–33882. PMLR.

Li, S. and Liu, Y. (2022). High probability guarantees for nonconvex stochastic gradient descent with heavy tails. In International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 12931–12963. PMLR.

Lindenstrauss, J. and Tzafriri, L. (1979). Classical Banach Spaces II: Function Spaces, volume 97 of Ergebnisse der Mathematik und ihrer Grenzgebiete. Springer-Verlag.

Liu, Z. (2026a). Clipped gradient methods for nonsmooth convex optimization under heavy-tailed noise: A refined analysis. In International Conference on Learning Representations.

Liu, Z. (2026b). In-expectation convergence of stochastic gradient methods under heavy-tailed noise. arXiv preprint arXiv:2606.00520.

Liu, Z., Zhang, J., and Zhou, Z. (2023). Breaking the lower bound with (little) structure: Acceleration in non-convex stochastic optimization with heavy-tailed noise. In The Thirty Sixth Annual Conference on Learning Theory, volume 195 of Proceedings of Machine Learning Research, pages 2266–2290. PMLR.

Liu, Z. and Zhou, Z. (2023). Stochastic nonsmooth convex optimization with heavy-tailed noises: High-probability bound, in-expectation rate and initial distance adaptation. arXiv preprint arXiv:2303.12277.

Liu, Z. and Zhou, Z. (2025). Nonconvex stochastic optimization under heavy-tailed noises: Optimal convergence without gradient clipping. In International Conference on Learning Representations.

Lobanov, A., Dvinskikh, D., and Gasnikov, A. (2026). Accelerated stochastic method under (H<sub>0</sub>, H<sub>1</sub>)-smoothness and heavy-tailed noise. arXiv preprint arXiv:2609.33415.

Lobanov, A. and Koloskova, A. (2026). Avoiding bias in clipped SGD for overparameterized models under generalized smoothness. arXiv preprint arXiv:2605.14800.

Lugosi, G. and Mendelson, S. (2019). Mean estimation and regression under heavy-tailed distributions: A survey. Foundations of Computational Mathematics, 19(5):1145–1190.

Mai, V. V. and Johansson, M. (2021). Stability and convergence of stochastic gradient clipping: Beyond Lipschitz continuity and smoothness. In International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 7325–7335. PMLR.

Nazin, A. V., Nemirovsky, A. S., Tsybakov, A. B., and Juditsky, A. B. (2019). Algorithms of robust stochastic optimization based on mirror descent method. Automation and Remote Control, 80(9):1607–1627.

Nemirovski, A., Juditsky, A., Lan, G., and Shapiro, A. (2009). Robust stochastic approximation approach to stochastic programming. SIAM Journal on Optimization, 19(4):1574–1609.

Nemirovski, A. S. and Yudin, D. B. (1983). Problem Complexity and Method Eficiency in Optimization. Wiley.

Nguyen, T. D., Nguyen, T. H., Ene, A., and Nguyen, H. (2023). Improved convergence in high probability of clipped gradient methods with heavy tailed noise. In Advances in Neural Information Processing Systems, volume 36, pages 24191–24222.

Nguyen, T. H., S¸im¸sekli, U., G¨urb¨uzbalaban, M., and Richard, G. (2019). First exit time analysis of stochastic gradient descent under heavy-tailed gradient noise. In Advances in Neural Information Processing Systems, volume 32.

Parletta, D. A., Paudice, A., Pontil, M., and Salzo, S. (2024). High probability bounds for stochastic subgradient schemes with heavy tailed noise. SIAM Journal on Mathematics of Data Science, 6(4):953–977.

Pascanu, R., Mikolov, T., and Bengio, Y. (2013). On the dificulty of training recurrent neural networks. In International Conference on Machine Learning, volume 28 of Proceedings of Machine Learning Research, pages 1310–1318. PMLR.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., and Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. <sup>´</sup> Journal of Machine Learning Research, 12(85):2825–2830.

Prasad, A., Suggala, A. S., Balakrishnan, S., and Ravikumar, P. (2020). Robust estimation via robust gradient estimation. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 82(3):601–627.

Puchkin, N., Gorbunov, E., Kutuzov, N., and Gasnikov, A. (2024). Breaking the heavy-tailed noise barrier in stochastic optimization problems. In International Conference on Artificial Intelligence and Statistics, volume 238 of Proceedings of Machine Learning Research, pages 856–864. PMLR.

Robbins, H. and Siegmund, D. (1971). A convergence theorem for non negative almost supermartingales and some applications. In Optimizing methods in statistics, pages 233–257. Elsevier.

Sadiev, A., Danilova, M., Gorbunov, E., Horv´ath, S., Gidel, G., Dvurechensky, P., Gasnikov, A., and Richt´arik, P. (2023). High-probability bounds for stochastic optimization and variational inequalities: the case of unbounded variance. In International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 29563–29648. PMLR.

Samorodnitsky, G. and Taqqu, M. S. (1994). Stable Non-Gaussian Random Processes: Stochastic Models with Infinite Variance. Chapman & Hall.

Shestakov, A., Tak´aˇc, M., and Gorbunov, E. (2026). From optimization to generalization under heavy-tailed data: The role of gradient clipping. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 111656–111690. PMLR.

S¸im¸sekli, U., Sagun, L., and G¨urb¨uzbalaban, M. (2019). A tail-index analysis of stochastic gradient noise in deep neural networks. In International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 5827–5837. PMLR.

Sun, T., Liu, X., and Yuan, K. (2025). Revisiting gradient normalization and clipping for nonconvex SGD under heavy-tailed noise: Necessity, suficiency, and acceleration. Journal of Machine Learning Research, 26(237):1–42.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozi\`ere, B., Goyal, N., Hambro, E., Azhar, F., et al. (2023). LLaMA: Open and eficient foundation language models. arXiv preprint arXiv:2302.13971.

Tsai, C.-P., Prasad, A., Balakrishnan, S., and Ravikumar, P. (2022). Heavy-tailed streaming statistical estimation. In International Conference on Artificial Intelligence and Statistics, volume 151 of Proceedings of Machine Learning Research, pages 1251–1282. PMLR.

Vaswani, S., Bach, F., and Schmidt, M. (2019). Fast and faster convergence of SGD for overparameterized models and an accelerated perceptron. In Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics, volume 89 of Proceedings of Machine Learning Research, pages 1195–1204. PMLR.

Vural, N. M., Yu, L., Balasubramanian, K., Volgushev, S., and Erdogdu, M. A. (2022). Mirror descent strikes again: Optimal stochastic convex optimization under infinite noise variance. In Conference on Learning Theory, volume 178 of Proceedings of Machine Learning Research, pages 65–102. PMLR.

Wang, B., Zhang, H., Ma, Z., and Chen, W. (2023). Convergence of AdaGrad for non-convex objectives: Simple proofs and relaxed assumptions. In The Thirty Sixth Annual Conference on Learning Theory, volume 195 of Proceedings of Machine Learning Research, pages 161–190. PMLR.

Wang, H., G¨urb¨uzbalaban, M., Zhu, L., S¸im¸sekli, U., and Erdogdu, M. A. (2021). Convergence rates of stochastic gradient descent under infinite noise variance. In Advances in Neural Information Processing Systems, volume 34, pages 18866–18877.

Yang, Y., Lu, K., and Wang, L. (2026). High probability convergence of distributed clipped stochastic gradient descent with heavy-tailed noise. Systems & Control Letters, 209:106358.

Yu, S., Jakoveti´c, D., and Kar, S. (2026). Smoothed gradient clipping and error feedback for decentralized optimization under symmetric heavy-tailed noise. SIAM Journal on Optimization, 36(2):703–728.

Zaman, M. U. and Mukherjee, A. (2026). Convergence of stochastic gradient methods under heavytailed noise and H¨older smoothness. arXiv preprint arXiv:2609.12785.

Zhang, J. and Cutkosky, A. (2022). Parameter-free regret in high probability with heavy tails. In Advances in Neural Information Processing Systems, volume 35, pages 8000–8012.

Zhang, J., He, T., Sra, S., and Jadbabaie, A. (2020a). Why gradient clipping accelerates training: A theoretical justification for adaptivity. In International Conference on Learning Representations.

Zhang, J., Karimireddy, S. P., Veit, A., Kim, S., Reddi, S., Kumar, S., and Sra, S. (2020b). Why are adaptive methods good for attention models? In Advances in Neural Information Processing Systems, volume 33, pages 15383–15393.

Zhu, T., Xu, Y., and Ji, X. (2025). Stochastic weakly convex optimization under heavy-tailed noises. arXiv preprint arXiv:2507.13283.

## Weak-L<sup>α</sup> Noise, Strong Guarantees: Gradient Clipping under Afine Heavy Tails Supplementary Material

## Contents

A Notation 24   
B Extended Related Work 26   
C Weak-L<sup>α</sup> and Marcinkiewicz Spaces 31   
C.1 Proof of Proposition C.3 33   
C.2 Proof of Proposition C.4 33   
C.3 Proof of Proposition C.5 33   
C.4 The Pareto Example 34   
C.5 Proof of Proposition C.7 34   
D Proofs of the Clipping Lemmas 34   
D.1 Separate Statements and Remarks 34   
D.2 Proof of Lemma D.1 36   
D.3 Proof of Lemma D.2 37   
D.4 Summing the Variance Bounds 38   
E Auxiliary Results 38   
F Upper Bounds for Clip-SGD: Full Statements and Proofs 41   
F.1 Full Statements 41   
F.2 A General Bound and the Proofs 44   
G Clip-SGD with an Arbitrary Clipping Level: Proof of Theorem G.1 52   
G.1 Full Statement 52   
G.2 Clipping with an Arbitrary Level 53   
G.3 The Mean Clipped Field 54   
G.4 Proof of Theorem G.1 55   
H Lower Bounds for a Fixed Clipping Level: Full Statement, Comparison and   
Proofs 57   
H.1 Full Statement and Comparison with Existing Results 57   
H.2 Proof of Theorem H.1 60   
H.3 Optimality of the Stochastic Term 61   
H.4 Strongly Convex Quadratics 64   
H.5 Strong Multiplicative Noise 65   
I Proofs for Clip-SSTM 65   
I.1 Full Statements . 65   
I.2 Adaptive Query Points 67   
I.3 Potential Inequality 68   
I.4 Weighted Bound on the True Gradients 69   
I.5 Proof of Theorem I.1 . 70   
I.6 Oracle Complexity 74   
I.7 Comparison with a Uniform Noise Scale 74   
I.8 The Lorentz Scale at the Endpoint 76   
I.9 Removing the Horizon Logarithm 82   
J Additional Experiments 85

## A Notation

In Table 2, we summarize the notation and the constants used in the statements of the paper, in the order of their appearance.

Table 2: Notation.
<table><tr><td>Symbol</td><td>Definition</td><td>Introduced in</td></tr><tr><td>Problem, oracle and method</td><td></td><td></td></tr><tr><td> $\langle \cdot , \cdot \rangle , \parallel \cdot \parallel , B _ { r } ( x ) , ( a ) _ { + } ,$   $\mathbf { 1 } _ { A }$ </td><td>Euclidean inner product and norm; closed ball of radius r around x; positive part; indicator of A</td><td>Section 3</td></tr><tr><td> $f , g ( x , \xi ) , \mathcal { D } , x ^ { * } , f ^ { * }$ </td><td>objective, unbiased stochastic oracle with  $\xi \sim \mathcal { D }$   $( \mathbb { E } _ { \xi } g ( x , \xi ) = \nabla f ( x )$  on  $Q )$  , a minimizer  $x ^ { * }$  and</td><td>(1), Section 3</td></tr><tr><td> $x ^ { 0 } , R$ </td><td> $f ^ { * } = f ( x ^ { * } )$  starting point and a number  $R \geq \| x ^ { 0 } - x ^ { * } \| > 0$ </td><td>Section 3</td></tr><tr><td> $Q = B _ { \rho R } ( x ^ { * } )$ </td><td>ball on which all assumptions are imposed;  $\rho = 2$  for Clip-SGD,  $\rho = 3$  for Clip-SSTM</td><td>Section 3</td></tr><tr><td> $L$ </td><td>local smoothness constant on  $Q$ </td><td>Assumption 3.1</td></tr><tr><td> $\mathcal { G } _ { k }$ </td><td>σ-algebra generated by  $x ^ { 0 } , \xi ^ { 0 } , \ldots , \xi ^ { k - 1 } \ ( \mathcal { G } _ { 0 } = \sigma ( x ^ { 0 } ) )$ </td><td>Section 3</td></tr><tr><td> $\sigma _ { 0 } , \sigma _ { 1 } , \alpha$ </td><td>additive and multiplicative noise scales and tail index,  $\alpha \in ( 1 , 2 ]$ </td><td>Assumptions 3.4, 3.5</td></tr><tr><td> $\| Y \| _ { L ^ { \alpha , \infty } } , L ^ { \alpha , \infty }$ </td><td>weak  $L ^ { \alpha }$  quasi-norm  $\textstyle \operatorname* { s u p } _ { u > 0 } u \mathbb { P } \{ | Y | > u \} ^ { 1 / \alpha }$  and the corresponding space</td><td>Definition C.1,</td></tr><tr><td> $d _ { Y } , Y ^ { * } , Y ^ { * * }$ </td><td>distribution function  $d _ { Y } ( u ) = \mathbb { P } \{ | Y | > u \}$  2 non-increasing rearrangement  $Y ^ { \ast } ( t ) = \operatorname* { i n f } \{ u > 0 : d _ { Y } ( u ) \leq t \}$  and its average</td><td>(13) Appendix C</td></tr><tr><td> $\| Y \| _ { M _ { \alpha } } , M _ { \alpha }$ </td><td> $\begin{array} { r } { Y ^ { * * } ( t ) = \frac { 1 } { t } \int _ { 0 } ^ { t } Y ^ { * } } \end{array}$  Marcinkiewicz norm  $\mathrm { s u p } _ { 0 < t \leq 1 } t ^ { 1 / \alpha } Y ^ { \ast \ast } ( t )$  and space</td><td>Definition C.2</td></tr><tr><td> $\| Y \| _ { 2 , s } , L ^ { 2 , s }$ </td><td> $( \alpha > 1 )$   $\begin{array} { r } { ( \int _ { 0 } ^ { 1 } [ t ^ { 1 / 2 } Y ^ { * } ( t ) ] ^ { s } \frac { \mathrm { d } t } { t } ) ^ { 1 / s } , 2 \leq s < \infty , } \end{array}$  Lorentz quasi-norm</td><td>Appendix I.8</td></tr><tr><td> $\mathsf { c l i p } ( v , \lambda )$ </td><td> $\| Y \| _ { 2 , \infty } = \| Y \| _ { L ^ { \alpha , \infty } } | _ { \alpha = 2 }$  , and the corresponding space clipping operator min  $\{ 1 , \lambda / \| v \| \} v$ </td><td>Section 3</td></tr><tr><td> $\lambda , \lambda _ { k } , \gamma , K$ </td><td>clipping level(s), stepsize, horizon: Clip-SGD makes</td><td>Algorithms 1, 2</td></tr><tr><td> $g ^ { k } , \widetilde { g } ^ { k } , \bar { x } ^ { K }$ </td><td> $K + 1$  iterations  $k = 0 , \ldots , K .$  , Clip-SSTM makes K stochastic gradient, clipped stochastic gradient, averaged Algorithm 1</td><td></td></tr><tr><td> $x ^ { k } , y ^ { k } , z ^ { k } , a , \omega _ { k + 1 } , A _ { k }$ </td><td>iterate  $\textstyle { \frac { 1 } { K + 1 } } \sum _ { k = 0 } ^ { K } x ^ { k }$  of Clip-SGD</td><td>Algorithm 2, (60)</td></tr><tr><td></td><td>Clip-SSTM sequences; parameter  $a \geq 1 ;$  weights  $\begin{array} { r } { \omega _ { k + 1 } = \frac { k + 2 } { 2 a L } , A _ { k + 1 } = A _ { k } + \omega _ { k + 1 } , A _ { k } = \frac { k ( k + 3 ) } { 4 a L } } \end{array}$ </td><td></td></tr><tr><td> $\beta$ </td><td>failure probability of the high-probability statements</td><td>Theorem F.1</td></tr><tr><td> $\ell , H , h$ </td><td> $\begin{array} { r } { \ell = \ln \frac { 4 ( K + 1 ) } { \beta } ( \mathsf { C l i p - S G D } ) , H = \ln \frac { 4 K } { \beta } ( \mathsf { C l i p - S S T M } ) , } \end{array}$  h = ln  $\frac { 4 } { \beta }$  (horizon-free confidence)</td><td>Theorems F.1, I.1, (90)</td></tr><tr><td> $A ( x )$ </td><td>affine noise scale  $\sigma _ { 0 } ^ { \alpha } + \sigma _ { 1 } ^ { \alpha } \| x \| ^ { \alpha }$  (so that  $A ( \nabla f ( x ) )$  bounds the tail or the moment of the noise at x)</td><td>Section 4</td></tr><tr><td> $\theta _ { k } , \theta _ { k } ^ { u } , \theta _ { k } ^ { b }$ </td><td>error of the clipped gradient  $\widetilde { g } ^ { k } - \nabla f ( x ^ { k } )$  and its decomposition into the martingale part  $\tilde { g } ^ { k } - \mathbb { E } [ \tilde { g } ^ { k } \mid \mathcal { G } _ { k } ]$  and the bias  $\mathbb { E } [ \widetilde { g } ^ { k } \mid \mathcal { G } _ { k } ] - \nabla f ( \bar { x ^ { k } } )$ </td><td>(25)</td></tr><tr><td> $R _ { k } , \delta _ { k }$  Clipping constants</td><td> $\| x ^ { k } - x ^ { * } \|$  and  $f ( x ^ { k } ) - f ^ { * }$ </td><td>Appendix F</td></tr><tr><td> $b _ { \alpha }$ </td><td> $. L ^ { \alpha }$   $\frac { 2 ^ { \alpha - 1 } } { \alpha - 1 }$  bias constant under the weak- condition,</td><td>Lemma D.1</td></tr><tr><td> $v _ { \alpha } ( s ) , v _ { \alpha }$ </td><td>variance constant under the weak  $. L ^ { \alpha }$  condition,  $( { \textstyle { \frac { 3 } { 2 } } } ) ^ { 2 - \alpha } \operatorname* { m i n } \{ { \frac { 2 } { 2 - \alpha } } , 1 + { \frac { 2 } { \alpha } } \ln ( 1 + s ) \} { \mathrm { ~ f o r ~ } } s \in [ 0 , \infty ]$ </td><td>(5), Section 4</td></tr><tr><td></td><td> $\begin{array} { r } { ( \frac { 2 } { 2 - \alpha } = + \infty \mathrm { ~ a t ~ } \alpha = 2 ) ; v _ { \alpha } = v _ { \alpha } ( \infty ) = \frac { 2 } { 2 - \alpha } ( \frac { 3 } { 2 } ) ^ { 2 - \alpha } } \end{array}$  for  $\alpha < 2$ </td><td></td></tr><tr><td> $b _ { \alpha } ^ { \mathrm { m o m } } , v _ { \alpha } ^ { \mathrm { m o m } }$   $( b , v )$ </td><td>bias and variace constants under the moment  $\frac { 2 ^ { \alpha - 1 } ( \alpha - 1 ) ^ { \alpha - 1 } } { \alpha ^ { \alpha } } \in [ \frac { 1 } { 2 } , 1 ]$  condition, and  $( { \frac { 3 } { 2 } } ) ^ { 2 - \alpha } \in [ 1 , { \frac { 3 } { 2 } } ]$  generic clipping constants:  $( b _ { \alpha } ^ { \mathrm { m o m } } , v _ { \alpha } ^ { \mathrm { m o m } } )$  under</td><td>(6) Section 5</td></tr><tr><td></td><td>Assumption  $3 . 4 , ( b _ { \alpha } , v _ { \alpha } ( \Lambda ) )$  under Assumption 3.5, with  $\Lambda = \Lambda _ { K }$  (Theorem F.1), Λλ (Theorem F.4),  $\Lambda _ { K } ^ { \mathrm { S } }$  (Theorem I.1) or  $\Lambda _ { \gamma , \lambda }$  (Proposition F.6)</td><td></td></tr><tr><td> $c , c _ { \alpha } ^ { \mathrm { m o m } } , c _ { \alpha } ( \Lambda )$   $\Lambda _ { \lambda } , \Lambda _ { K }$ </td><td>rate constant  $c = 1 8 0 \operatorname* { m a x } \{ v , b / 3 \}$   $c _ { \alpha } ^ { \mathrm { m o m } } = 1 8 0 ( \textstyle { \frac { 3 } { 2 } } ) ^ { 2 - \alpha } \in [ 1 8 0 , \dot { 2 } 7 0 )$  under Assumption 3.4,  $c _ { \alpha } ( \Lambda ) = 1 8 0 \tilde { \operatorname* { m a x } } \{ v _ { \alpha } ( \Lambda ) , b _ { \alpha } / 3 \}$  under Assumption 3.5  $\begin{array} { r } { \Lambda _ { \lambda } = \frac { ( K + 1 ) ( 3 \lambda / 2 ) ^ { \alpha } } { ( K + 1 ) \sigma _ { 0 } ^ { \alpha } + 2 ^ { \alpha / 2 } \sigma _ { 1 } ^ { \alpha } ( L R ) ^ { \alpha } } } \end{array}$  (ratio of the clipping level to</td><td>(28), Section 5</td></tr><tr><td></td><td>the average noise scale; 0 if  $\sigma _ { 0 } = \sigma _ { 1 } = 0 )$   $\Lambda _ { K } = \operatorname* { m a x } \{ 8 0 ( K + 1 ) , \Lambda _ { 2 \sqrt { 2 } L R } \}$ </td><td>(29)</td></tr><tr><td> $\Lambda _ { \gamma , \lambda }$   $\Lambda _ { K } ^ { \mathrm { S } } , \Lambda _ { K } ^ { 0 }$ </td><td> $\begin{array} { r } { \frac { ( K + 1 ) ( 3 \lambda / 2 ) ^ { \alpha } } { \Sigma _ { K + 1 } } \leq \Lambda _ { \lambda } } \end{array}$  , the ratio used in the proofs for  $\mathsf { C l i p { - } S G D }$ </td><td>(38)</td></tr><tr><td></td><td> $\Lambda _ { K } ^ { \mathrm { S } } = \operatorname* { m a x } \{ 2 0 0 b K , \Lambda _ { K } ^ { 0 } \}$  with  $\begin{array} { r } { \Lambda _ { K } ^ { 0 } = [ ( \frac { \sigma _ { 0 } ( K + 1 ) } { 1 7 6 4 0 L R H } ) ^ { \alpha } + ( \frac { \sigma _ { 1 } } { 3 } ) ^ { \alpha } ] ^ { - 1 } } \end{array}$  (0 if  $\sigma _ { 0 } = \sigma _ { 1 } = 0 )$  , the analog of  $\Lambda _ { K }$  for Clip-SSTM</td><td>(61)</td></tr><tr><td colspan="3">Clip-SGD with the tuned clipping level (Theorem F.1, Corollary F.2)</td></tr><tr><td> $\gamma _ { L } , \gamma _ { 0 } , \gamma _ { 1 } , \gamma _ { 2 }$ </td><td>stepsize thresholds  ${ \frac { 1 } { 8 0 \sqrt { 2 } L \ell } } , { \frac { R } { ( c 4 0 ^ { \alpha - 1 } ( K + 1 ) \ell ^ { \alpha - 1 } ) ^ { 1 / \alpha } \sigma _ { 0 } } } ,$   $\begin{array} { r } { \frac { \star } { ( c 2 ^ { \alpha } 4 0 ^ { \alpha - 1 } \ell ^ { \alpha - 1 } ) ^ { 2 / \alpha } \sigma _ { 1 } ^ { 2 } L ( K + 1 ) ^ { ( 2 - \alpha ) / \alpha } } , \frac { \star } { ( c 2 ^ { \alpha / 2 } 4 0 ^ { \alpha - 1 } \ell ^ { \alpha - 1 } ) ^ { 1 / \alpha } \sigma _ { 1 } L } \ddag \frac { \star } { ( c 2 ^ { \alpha } 4 0 ^ { \alpha - 1 } \ell ^ { \alpha - 1 } ) ^ { 1 / \alpha } \sigma _ { 1 } L } \ddag \frac { \star } { ( c 2 ^ { \alpha } 4 0 ^ { \alpha - 1 } \ell ^ { \alpha - 1 } ) ^ { 1 / \alpha } \sigma _ { 1 } L } } \end{array}$ </td><td>Theorem F.1</td></tr><tr><td> $\Delta _ { L } , \Delta _ { 0 } , \Delta _ { 1 } \ ( \Delta _ { 2 } )$ </td><td>the level is  $\begin{array} { r } { \lambda = \frac { { \bar { R } } } { 4 0 \gamma \ell } } \end{array}$  the three terms of the rate,  $\begin{array} { r } { \Delta _ { j } = \frac { 2 R ^ { 2 } } { \gamma _ { j } ( K + 1 ) } } \end{array}$ </td><td>Corollary F.2</td></tr><tr><td> $\Sigma _ { T } , \Phi _ { T }$ </td><td> $\Sigma _ { T } = \sigma _ { 0 } ^ { \alpha } T + \sigma _ { 1 } ^ { \alpha } 2 ^ { \alpha } L ^ { \alpha / 2 } R ^ { \alpha } \gamma ^ { - \alpha / 2 } T ^ { ( 2 - \alpha ) / 2 } + \sigma _ { 1 } ^ { \alpha } 2 ^ { \alpha / 2 } L ^ { \alpha } R ^ { \alpha }$  (accumulated noise scale over  $T$  steps) and  $\Phi _ { T } = \gamma \lambda ^ { 1 - \alpha } \Sigma _ { T }$  (accumulated bias scale)</td><td>(37)</td></tr><tr><td>Clip-SSTM (Theorem I.1, Corollaries I.2, I.9)</td><td></td><td></td></tr><tr><td> $\textstyle { \frac { \alpha - 1 } { \alpha } }$  r</td><td>, the exponent of the additive term  $\begin{array} { r } { q _ { \alpha } = \operatorname* { m i n } \lbrace \frac { 1 } { 1 3 5 0 v 3 0 ^ { \alpha - 2 } } , \frac { 1 } { 1 5 b 3 0 ^ { \alpha - 1 } } \rbrace , c _ { 0 , \alpha } = ( 2 / q _ { \alpha } ) ^ { 1 / \alpha } , } \end{array}$ </td><td>Theorem I.1</td></tr><tr><td> $q _ { \alpha } , c _ { 0 , \alpha } , c _ { 1 , \alpha }$ </td><td> $c _ { 1 , \alpha } = 4 9 ( 4 / q _ { \alpha } ) ^ { 2 / \alpha } { \mathrm { ~ ( a t ~ } } \alpha = 2 { \mathrm { ~ u n d e r ~ A s s u m p t i o n ~ } } 3 . 4 \colon$ </td><td>Theorem I.1</td></tr><tr><td> $a _ { * } ( K , \beta )$ </td><td> $\begin{array} { r } { q _ { 2 } = \frac { 1 } { 1 3 5 0 } , c _ { 0 , 2 } = \sqrt { 2 7 0 0 } , c _ { 1 , 2 } = 4 9 \cdot 5 4 0 0 ) } \end{array}$   $\operatorname* { m a x } \{ 1 7 6 4 0 0 H ^ { 2 } , c _ { 0 , \alpha } \frac { \sigma _ { 0 } } { L R } ( K + 1 ) K ^ { 1 / \alpha } H ^ { r } , c _ { 1 , \alpha } \sigma _ { 1 } ^ { 2 } K ^ { 2 / \alpha } H ^ { 2 r } \}$  the prescribed value of  $a ;$  the levels are  $\begin{array} { r } { \lambda _ { k } = \frac { R } { 3 0 \omega _ { k + 1 } H } } \end{array}$ </td><td>Theorem I.1</td></tr><tr><td> $C _ { \alpha } , M _ { \varepsilon }$ </td><td> $C _ { \alpha } = \operatorname* { m a x } \{ 1 0 5 8 4 0 0 , 1 2 c _ { 0 , \alpha } , 6 c _ { 1 , \alpha } \}$  and the horizon scale  $M _ { \varepsilon }$  of the oracle complexity</td><td>Corollary I.2</td></tr><tr><td> $\bar { \sigma } , a _ { \mathrm { u n i } }$ </td><td>uniform noise scale  $( \sigma _ { 0 } ^ { \alpha } + ( 3 L R \sigma _ { 1 } ) ^ { \alpha } ) ^ { 1 / \alpha } \ \mathrm { o n } \ Q = B _ { 3 R } ( x ^ { * } )$  and the corresponding value of a</td><td>Section 5.1, Corollary I.9</td></tr><tr><td> $C _ { p , s } , p _ { K }$ </td><td> $\begin{array} { r } { C _ { p , s } = ( \frac { 2 ( s - p ) } { s ( 2 - p ) } ) ^ { 1 / p - 1 / s } } \end{array}$  (embedding constant  $\| Y \| _ { p } \leq C _ { p , s } \| Y \| _ { 2 , s } )$  and the horizon-dependent exponent  $\stackrel { \cdot } { p _ { K } } = 2 - \operatorname* { m i n } \{ \textstyle { \frac { 1 } { 2 } } , \frac { 1 } { \ln ( e K ) } \}$  of the moment route</td><td>Remark I.15</td></tr><tr><td>Prescribed clipping levels (Theorems F.4, G.1, H.1, Corollary F.5)</td><td></td><td></td></tr><tr><td> $S _ { 0 } , S _ { 1 } , S _ { 2 } , \Sigma _ { K + 1 } ( \gamma )$ </td><td> $S _ { 0 } = \sigma _ { 0 } ^ { \alpha } ( K + 1 ) , S _ { 1 } = 2 ^ { \alpha } \sigma _ { 1 } ^ { \alpha } L ^ { \alpha / 2 } R ^ { \alpha } ( K + 1 ) ^ { ( 2 - \alpha ) / 2 } ,$   $S _ { 2 } = 2 ^ { \bar { \alpha } / 2 } \sigma _ { 1 } ^ { \alpha } ( L R ) ^ { \alpha } , \mathrm { a n d } \Sigma _ { K + 1 } ( \gamma ) = S _ { 0 } + \bar { S } _ { 1 } \gamma ^ { - \alpha / 2 } + S _ { 2 }$  (equal to  $\Sigma _ { K + 1 }$  of (37))</td><td>Theorem F.4</td></tr><tr><td> $\gamma _ { b } ^ { \lambda } , \gamma _ { v } ^ { \lambda }$ </td><td>explicit stepsize thresholds for the bias condition (31) and the variance condition (32)</td><td>(53)</td></tr><tr><td> $\mathcal { N } _ { \lambda } , \mathcal { V } _ { \lambda , K } , \Xi$ </td><td> $\begin{array} { r } { \cdot \{ \frac { 1 2 0 b R \sigma _ { 0 } ^ { \alpha } } { \lambda ^ { \alpha - 1 } } , 2 \big ( \frac { 6 0 \cdot 2 ^ { \alpha } b R L ^ { \alpha / 2 } \sigma _ { 1 } ^ { \alpha } } { \lambda ^ { \alpha - 1 } } \big ) ^ { 2 / ( 2 - \alpha ) } \} , } \end{array}$  neighborhood max the terms vanishing with  $K$  , and the constant of the rate for  $\lambda = \lambda _ { 0 } ( K + 1 ) ^ { \bar { 1 / \alpha } }$ </td><td>(34), (35), (36)</td></tr><tr><td> ${ \bar { b } } _ { \lambda }$ </td><td>bias scale  $\begin{array} { r } { \frac { \sigma _ { 0 } ^ { \alpha } } { \operatorname* { m a x } \{ \lambda , \sigma _ { 0 } \} ^ { \alpha - 1 } } = \operatorname* { m i n } \{ \frac { \sigma _ { 0 } ^ { \alpha } } { \lambda ^ { \alpha - 1 } } , \sigma _ { 0 } \} } \end{array}$ </td><td>(11)</td></tr><tr><td> $( \kappa _ { 1 } , \kappa _ { 2 } ) , c _ { \perp } , \bar { \sigma } _ { 1 }$ </td><td>(1, 1) under Assumption  ${ 3 . 4 , ( \frac { \alpha } { \alpha - 1 } , \frac { 2 } { 2 - \alpha } ) }$  under Assumption 3.5 with  $\alpha < 2 ; c _ { \perp } = 2 ^ { \alpha - 1 } ( \kappa _ { 2 } + 3 \kappa _ { 1 } ) ;$  threshold  $\begin{array} { r } { \bar { \sigma } _ { 1 } = \frac { 1 } { 2 \cdot 2 8 ^ { 1 / \alpha } } } \end{array}$  on the multiplicative noise</td><td>Section 5.2</td></tr><tr><td>À,  $V _ { \lambda } , \varepsilon _ { N } , \varepsilon _ { M }$ </td><td> $\bar { A } = \sigma _ { 0 } ^ { \alpha } + \sigma _ { 1 } ^ { \alpha } ( \sqrt { 2 } L R ) ^ { \alpha } , V _ { \lambda } = \operatorname * { m i n } \{ 4 \lambda ^ { 2 } , \kappa _ { 2 } ( 2 \lambda ) ^ { 2 - \alpha } \bar { A } \}$  (variance scale of the clipped gradient), neighborhoods  $\varepsilon _ { N } = 4 8 \sqrt { 2 } c _ { \perp } R \bar { b } _ { \lambda }$  and</td><td>Theorem G.1</td></tr><tr><td> $\varepsilon _ { \lambda } , \ s _ { \lambda }$ </td><td> $\varepsilon _ { M } = 3 8 4 c _ { \perp } \sigma _ { 1 } ^ { \alpha } L R ^ { 2 } \operatorname* { m i n } \{ 1 , 9 6 c _ { \perp } \sigma _ { 1 } ^ { \alpha } \}$  lower bound  $\frac { 1 } { 4 8 { \cdot } 4 ^ { \alpha } }$  min  $\{ \frac { R \sigma _ { 0 } ^ { \alpha } } { \lambda ^ { \alpha - 1 } } , R \sigma _ { 0 } , L R ^ { 2 } \}$   $\begin{array} { r } { \frac { 1 } { 4 ^ { \alpha + 1 } } \operatorname* { m i n } \{ \frac { \sigma _ { 0 } ^ { \alpha } } { \lambda ^ { \alpha - 1 } } , \lambda \} } \end{array}$ </td><td>and growth rate Theorem H.1</td></tr></table>

## B Extended Related Work

We expand the comparison from Section 2, distinguishing the noise assumptions, the dependence of its scale on the current point, the clipping schedule, and the type of guarantee. Our analysis concerns smooth convex problems and high-probability bounds. The non-convex, non-smooth, light-tailed, and in-expectation results below provide complementary context.

Light-tailed noise and boosting. Logarithmic dependence on the inverse failure probability is classical under sub-Gaussian noise (Nemirovski et al., 2009) and extends to sub-Weibull noise, which has all moments (Li and Liu, 2022). For strongly convex problems, Davis et al. (2021) obtain highconfidence guarantees from low-confidence solvers using robust distance estimation. Their concrete stochastic-gradient instantiations assume light tails or bounded variance, whereas the abstract reduction applies whenever suitable solvers for the regularized subproblems are available. Below, we focus on direct analyses of clipping and on alternative methods for heavy-tailed noise.

Clipped methods for convex problems under bounded moments. Under bounded variance, Nazin et al. (2019) analyze truncated Mirror Descent, and Gorbunov et al. (2020) analyze

Clip-SGD and Clip-SSTM for smooth convex and strongly convex problems. Related guarantees cover monotone and structured non-monotone variational inequalities (Gorbunov et al., 2022), non-smooth convex problems (Gorbunov et al., 2024a), and subgradient schemes (Parletta et al., 2024). Sadiev et al. (2023) replace bounded variance with a bounded central moment of order $\alpha \in ( 1 , 2 ]$ for smooth convex, strongly convex, and monotone variational inequality problems. We recover their smooth convex rates when $\sigma _ { 1 } = 0$ (Table 1). The smooth convex analyses of Gorbunov et al. (2020); Sadiev et al. (2023) tune the clipping level to the horizon and bound the noise scale by a constant on a ball around the solution.

Nguyen et al. (2023) remove the horizon logarithm from the confidence factor for clipped Mirror Descent and its accelerated variant. Liu and Zhou (2023) obtain a similar confidence dependence for non-smooth convex problems, together with in-expectation rates and adaptation to the initial distance. Our Theorem I.18 gives the corresponding reduction via a maximal Freedman inequality. Liu (2026a) refine the non-smooth analysis using a generalized efective dimension and derive highprobability and in-expectation lower bounds, which we compare with ours in Section 5.

Further extensions cover composite and distributed problems and variational inequalities (Gorbunov et al., 2024b), distributed Clip-SGD over networks (Yang et al., 2026), and, in expectation, distributed methods with local updates (Lee et al., 2025). Clipping has also been analyzed for AdaGrad-Norm and Adam-Norm (Chezhegov et al., 2025), convex $( L _ { 0 } , L _ { 1 } )$ -smooth problems (El Bakkali El Kadi et al., 2025), weakly convex problems (Zhu et al., 2025), and zeroth-order nonsmooth convex optimization (Kornilov et al., 2023). Other settings include parameter-free online learning (Zhang and Cutkosky, 2022), diferential privacy with an arbitrary clipping level (Khah et al., 2026), and streaming estimation (Tsai et al., 2022; Das et al., 2024). These guarantees use moment or variance bounds.

Concurrent work. Lobanov et al. (2026) (arXiv, 27 September 2026) analyze RCSAG – a restarted clipped stochastic accelerated gradient method – for diferentiable convex and strongly convex objectives under $( H _ { 0 } , H _ { 1 } )$ -smoothness, i.e., $f ( y ) \leq f ( x ) + \langle \nabla f ( x ) , y - x \rangle + ( H _ { 0 } + H _ { 1 } ( f ( x ) -$ $f ^ { * } ) ) \lVert y - x \rVert ^ { 2 }$ for $\| y - x \| \leq r _ { H }$ (L-smoothness corresponds to $H _ { 1 } = 0$ with $L = 2 H _ { 0 } )$ , and a conditionally unbiased oracle satisfying, in our notation, $\begin{array} { r } { \mathbb { E } \| g ( x , \xi ) - \nabla f ( x ) \| ^ { \alpha } \leq \sigma _ { 0 } ^ { \alpha } + \sigma _ { 1 } ^ { \alpha } \| \nabla f ( x ) \| ^ { \alpha } + } \end{array}$ $\sigma _ { 2 } ^ { \alpha } ( f ( x ) - f ^ { * } ) ^ { \alpha / 2 } , \alpha \in ( 1 , 2 ]$ , at every query point (their Assumption 3). This is a moment condition: for $\sigma _ { 2 } = 0$ , it coincides with Assumption 3.4 imposed on $\mathbb { R } ^ { d }$ , and otherwise it has an additional term decaying with the function gap

In the convex case, the method runs a logarithmic number of phases starting from an upper bound on $f ( x ^ { 0 } ) - f ^ { * }$ ; each phase starts from a point whose gap is at most the current bound, runs an accelerated scheme of Nesterov type with one clipped sample per iteration, whose auxiliary sequence is projected onto a bounded convex set $Q \ni x ^ { 0 } , x ^ { * }$ of diameter D, with the clipping level and the number of iterations chosen according to this bound, D, and the noise and smoothness parameters, and halves the bound with high probability (their Theorem 8, which relies on a maximal Freedman inequality for a stopped potential).

The total number of oracle calls (their Theorem 9) is ${ \widetilde O } ( D \sqrt { H _ { 0 } / \varepsilon } + \sqrt { H _ { 1 } } D J _ { H } + ( \sigma _ { 0 } D / \varepsilon ) ^ { \alpha / ( \alpha - 1 ) } +$ $( \sigma _ { 1 } D [ \sqrt { H _ { 0 } / \varepsilon } + \sqrt { H _ { 1 } } J _ { H } ^ { ( \alpha - 1 ) / \alpha } ] ) ^ { \alpha / ( \alpha - 1 ) } + ( \sigma _ { 2 } D / \sqrt { \varepsilon } ) ^ { \alpha / ( \alpha - 1 ) } + S )$ , where $S = \lceil \log _ { 2 } ( \Delta _ { 0 } / \varepsilon ) \rceil$ for an initial bound $\Delta _ { 0 } \geq f ( x ^ { 0 } ) - f ^ { * }$ and $0 < \varepsilon \le \Delta _ { 0 }$ , and $J _ { H } \leq S$ counts the phases with $H _ { 1 } 2 ^ { - s } \Delta _ { 0 } > H _ { 0 } ;$ for $H _ { 1 } = 0$ and $\sigma _ { 2 } = 0$ , this has the same polynomial dependence, up to constants and logarithmic factors, as the complexity of Theorem 5.2 in Table 1, with $D = 2 R$ in place of R (their Appendix G.1 takes $Q = B _ { R } ( x ^ { 0 } )$ when R is known), including the dependence on $\sigma _ { 1 }$ , whose optimality is open in both works.

Under strong convexity (their Theorem 10), the method uses shrinking balls centered at the restart points, and only the $\sigma _ { 0 } \mathrm { - t e r m }$ is polynomial in $1 / \varepsilon$ . For the noise satisfying $\mathbb { E } \| g ( x , \xi ) -$ $\nabla f ( x ) \| ^ { \alpha } \lesssim ( f ( x ) - f ^ { * } ) ^ { \varrho }$ , the stochastic cost is polynomial, logarithmic, or bounded in $1 / \varepsilon$ depending on whether $\varrho$ is below, at, or above $\alpha$ for convex and $\alpha / 2$ for strongly convex objectives (their Theorem 11). Their lower bounds for the convex case and the $\sigma _ { 0 ^ { - } }$ and $\sigma _ { \mathrm { { 2 ^ { - } } } } \mathrm { { t e r m s } }$ (their Theorems 13 and 15) are based on a two-point construction with a rare informative response, similar to the one used in our Propositions H.2 and H.3, and have the factor $\ln ( 1 / ( 2 \rho ) )$ in terms of the failure probability $\rho .$

The two works are complementary: Lobanov et al. (2026) consider generalized smoothness, strong convexity, and the noise decaying with the function gap, under moment bounds, and analyze a restarted and projected method whose parameters depend on the bound on $f ( x ^ { 0 } ) - f ^ { * } , D , \alpha$ $\sigma _ { i } ,$ , and $H _ { i }$ (the horizon-driven variant from their Appendix I removes the last three from the inner updates, and their Appendix $\mathrm { G }$ constructs the initial bound from stochastic gradients); we consider Clip-SGD and Clip-SSTM without restarts and projections, under the tail condition on a ball, and also cover the $\mathrm { w e a k } { - } L ^ { 2 }$ endpoint with infinite variance, prescribed clipping levels, and the corresponding lower bounds.

Beyond moment bounds: symmetry, medians, and tail conditions. Several approaches relax moment assumptions by imposing structure on the noise. Under symmetry, Jakoveti´c et al. (2023) analyze a general class of non-linear SGD methods (clipping, sign, normalization, and their component-wise versions) for strongly convex objectives under the noise whose components have a symmetric density and finite first moment only, and derive almost sure convergence and mean-squared-error rates; Armacki et al. (2023) derive high-probability bounds for non-convex and strongly convex problems in the same noise model, and Armacki et al. (2025) extend this framework to online learning, treating sign, quantization, and clipping non-linearities in a unified way, for symmetric noise without any moment assumption and for mixtures of symmetric and non-symmetric components, with convergence to a neighborhood in the latter case; Armacki et al. (2026) derive large-deviation and mean-squared-error bounds under a symmetric density that is positive near zero and possibly unbounded moments; Yu et al. (2026) obtain mean-squared-error rates for smoothed clipping with error feedback for decentralized strongly convex problems; and Kornilov et al. (2024) use median clipping for zeroth-order convex problems and bandits under symmetric noise with bounded κ-th moment for some $\kappa > 0$ , in which case the noise may have no finite expectation.

Puchkin et al. (2024) stabilize the stochastic gradients via smoothed medians of means before clipping and derive, under a structural condition on the marginal densities of the noise (a symmetric heavy-tailed part plus an antisymmetric remainder with finite second moment), high-probability rates for (quasi-)strongly convex problems of the order $\widetilde { O } ( K ^ { - 1 / 2 } )$ in general, which improves upon the heavy-tailed rate ${ \cal O } ( K ^ { - 2 ( \alpha - 1 ) / \alpha } )$ for $\alpha < 4 / 3$ , and of the order $\widetilde O ( K ^ { - 1 } )$ for symmetric noise.

A tail condition also appears in Shestakov et al. (2026): there, a regularly varying tail $\begin{array} { r } { \mathbb { P } \{ \| g ( x ^ { * } , \xi ) - \nabla f ( x ^ { * } ) \| \ge u \} \sim c u ^ { - \alpha } } \end{array}$ at the minimizer quantifies how heavy-tailed data afect the finite-sum minimization and the generalization of ClipSGD, with the bounds in expectation over the algorithm.

The weak- $L ^ { p }$ tail condition itself appears in El Bakkali et al. (2026), who impose it on the sample-gradient noise with a constant scale on $\mathbb { R } ^ { d }$ (their Assumption 1.3) for zeroth-order optimization of smooth non-convex objectives under global sample-wise L-smoothness (their Assumption 1.2): their method clips the scalar directional derivatives of a two-point estimator before aggregation, and they derive high-probability guarantees for finding ε-stationary points using $\widetilde { \cal O } ( d ^ { p / ( 2 ( p - 1 ) ) } \varepsilon ^ { - ( 3 p - 2 ) / ( p - 1 ) } $ ) function evaluations, which gives $\widetilde { O } ( d \varepsilon ^ { - 4 } )$ at $p = 2$ under a condition allowing infinite variance.

Their analysis of the base method controls the error of the estimator at every step separately, via a vector Bernstein inequality for the mini-batch and a union bound over the steps; therefore, their base method requires mini-batches of size $\widetilde { \Theta } \big ( ( \sqrt { d } ( \sqrt { L ( f ( x ^ { 0 } ) - f ^ { * } ) } + \sigma ) / \varepsilon ) ^ { p / ( p - 1 ) } \big )$ at each of its $O ( L ( f ( x ^ { 0 } ) - f ^ { * } ) / \varepsilon ^ { 2 } )$ steps, where $\sigma$ is their noise scale (their Remark $2 . 4 )$ ; at $p = 2$ , both the batch size and the number of steps grow as $\varepsilon ^ { - 2 }$

Their momentum variant reduces the running batch size in proportion to one minus the momentum parameter, down to a single two-point estimate per step (two function evaluations with shared randomness), at the price of the stepsize reduced in the same proportion, a warm-start batch of the full size, and the same total number of function evaluations up to logarithmic factors and the additive warm-start cost (their Theorem 3.1 and Remark 3.2).

Our Assumption 3.5 difers from the symmetry-based and minimizer-based conditions in that it is a weak- $L ^ { \alpha }$ bound on the whole ball $Q ,$ without any symmetry or density assumptions; it has the same tail form as the condition from El Bakkali et al. (2026), in the afine and loca version, and we use it for first-order methods on convex problems with one oracle call per iteration, where the polynomial rates coincide with those for the moment case. We characterize the additional logarithmic factor at the weak- $L ^ { 2 }$ endpoint with infinite variance, analyze prescribed clipping levels, and derive lower bounds.

Methods without clipping can also converge under infinite variance (Wang et al., 2021; Fatkhullin et al., 2025; Liu, 2026b). These results and the in-expectation analysis under H¨older smoothness by Zaman and Mukherjee (2026) use moment bounds. For convex composite problems, He et al. (2025) analyze an accelerated stochastic proximal method without clipping under a uniform bounded-moment assumption. Their heavy-tailed high-probability bounds have polynomial confidence dependence; logarithmic dependence is obtained under stronger sub-Weibull assumptions. Their bounded-domain and proximal setting difers from our local analysis of unconstrained first-order methods.

Non-convex problems and alternatives to clipping. Zhang et al. (2020b) show that clipped SGD attains the optimal in-expectation rate $K ^ { - ( \alpha - 1 ) / ( 3 \alpha - 2 ) }$ for the gradient norm under a bounded α-th moment assumption. Cutkosky and Mehta (2021) derive high-probability bounds by combining clipping, normalization, and momentum. Liu et al. (2023) improve the analysis and obtain acceleration beyond the general lower bound when the objective is an expectation of smooth sample functions.

H¨ubler et al. (2025) analyze normalized SGD with mini-batches or momentum without clipping: they obtain in-expectation convergence without knowledge of the problem parameters and highprobability guarantees for the mini-batched variant. Liu and Zhou (2025) show that normalized SGD with momentum and mini-batches achieves the optimal in-expectation rate when the tail index is known; when it is unknown, their rate is $K ^ { - ( \alpha - 1 ) / ( 2 \alpha ) }$ . Sun et al. (2025) clarify the roles of normalization and clipping, and Kornilov et al. (2025) analyze sign methods with high probability under $( L _ { 0 } , L _ { 1 } )$ )-smoothness. Li and Liu (2022) obtain high-probability guarantees for non-convex

SGD under sub-Weibull noise. These results concern non-convex objectives and complement our smooth convex analysis.

Afine and multiplicative noise. Afine variance (Bottou et al., 2018), expected smoothness (Gower et al., 2019), and the ABC condition (Khaled and Richt´arik, 2023) are standard assumptions in the in-expectation analysis of SGD. Adaptive methods under afine variance are studied by Faw et al. (2022); Wang et al. (2023) and, under $( L _ { 0 } , L _ { 1 } )$ -smoothness, by Faw et al. (2023). These works concern non-convex objectives and include high-probability guarantees with polynomial dependence on the inverse failure probability. Attia and Koren (2023) obtain high-probability bounds under an almost sure afine noise bound, with a sub-Gaussian extension; these assumptions exclude infinite variance.

State-dependent finite-variance bounds also appear in streaming estimation. For mean estimation and linear regression, Tsai et al. (2022) analyze clipped SGD with a decaying stepsize and variance growing quadratically with the distance to the solution. Their clipping level scales as $\sqrt { N }$ and depends on the confidence level, and their squared-error bound has order $1 / N$ with logarithmic confidence dependence. Das et al. (2024) derive high-probability guarantees for smooth strongly convex objectives under a covariance bound growing quadratically with the distance to the solution (their Assumption 2 and Theorem 2). Under infinite variance, Wang et al. (2021) analyze unmodified SGD with state-dependent noise and a finite p-th moment, $p \in [ 1 , 2 )$ , for a class of strongly convex objectives satisfying a p-positive-definiteness condition on the Hessian. They obtain $L ^ { p }$ rates and a generalized central limit theorem.

Multiplicative noise also explains how heavy-tailed iterates can arise from light-tailed data (G¨urb¨uzbalaban et al., 2021; Hodgkinson and Mahoney, 2021); Jiao and Keller-Ressel (2024) study this phenomenon in homogenized SGD. Heavy-tailed gradient noise is documented and modeled by S¸im¸sekli et al. (2019); Nguyen et al. (2019); Zhang et al. (2020b), and Barsbey et al. (2021) relate heavy tails to compressibility and generalization. Under global strong growth or multiplicative finite-variance conditions, accelerated methods attain faster rates in expectation (Vaswani et al., 2019; Gupta et al., 2024); see the comparison in Appendix I.

Our assumptions combine a gradient-dependent scale with heavy tails. Replacing this scale by $\sigma = \sigma _ { 0 } + \sigma _ { 1 } \operatorname* { s u p } _ { Q } \| \nabla f \|$ recovers a constant-scale condition but loses the separation between the additive and multiplicative contributions (Corollary I.9).

Prescribed clipping levels and clipping bias. Clipping with a fixed level may introduce a persistent bias. Under bounded variance, Koloskova et al. (2023) derive tight in-expectation guarantees for convergence to a neighborhood and show that decreasing the stepsize alone cannot remove the bias. Chen et al. (2020) study this efect in private SGD through the symmetry of the gradient distribution. Mai and Johansson (2021) establish stability estimates beyond Lipschitz continuity, including a fixed-threshold variant.

Zhang et al. (2020a) explain how clipping can accelerate optimization under $( L _ { 0 } , L _ { 1 } ) .$ smoothness. For convex objectives in this class, Gorbunov et al. (2025) analyze gradient descent with smoothed clipping and its stochastic finite-sum variant under a common-minimizer assumption, in expectation. Gaash et al. (2025) derive high-probability bounds for clipped SGD with double sampling under sub-Gaussian noise, and El Bakkali El Kadi et al. (2025) obtain such bounds under a bounded α-th moment without exponentially large factors. Under strong growth and interpolation, Lobanov and Koloskova (2026) analyze clipping and normalization under generalized smoothness and show that a suitable batch size can eliminate the error floor. Their heavy-tailed extension concerns normalized SGD and uses a moment bound.

Under standard smoothness and a bounded α-th moment, Khah et al. (2026) obtain highprobability guarantees for the best iterate with any clipping level. Our Theorem F.4 controls the averaged iterate under afine weak- $L ^ { \alpha }$ noise for $\lambda \ge 2 \sqrt { 2 } L R$ (and $\lambda \ge 2 4 0 b \sigma _ { 1 } ^ { 2 } L R$ when $\alpha = 2$ and $\sigma _ { 1 } > 0 )$ Theorem G.1 covers any $\lambda > 0$ for the best iterate when $\sigma _ { 1 } ~ \le ~ \bar { \sigma } _ { 1 }$ (under $\mathrm { A s } _ { - }$ sumption 3.5, for $\alpha < 2 )$ . For additive noise, Theorem H.1 gives the unavoidable neighborhood $\frac { 1 } { 4 8 { \cdot } 4 ^ { \alpha } }$ min $\{ R \sigma _ { 0 } ^ { \alpha } / \lambda ^ { \alpha - 1 } , R \sigma _ { 0 } , L R ^ { 2 } \}$ through a distribution satisfying the moment condition.

Lower bounds. Stochastic convex lower bounds go back to Nemirovski and Yudin (1983). Under infinite variance, Vural et al. (2022) prove that $K ^ { - ( \alpha - 1 ) / \alpha }$ is optimal for non-smooth convex problems and that Mirror Descent attains it in expectation. Zhang et al. (2020b) derive the non-convex counterpart $K ^ { - ( \alpha - 1 ) / ( 3 \alpha - 2 ) }$ . Liu (2026a) obtain high-probability and in-expectation lower bounds for non-smooth convex problems in terms of two noise scales.

Our Propositions H.2 and H.3 concern smooth unconstrained problems, give explicit constants, and capture the failure probability. Proposition I.16 establishes the necessity of the logarithmic factors on the Lorentz scale at $\alpha = 2$ , in the regimes stated there. We do not derive a lower bound for the multiplicative-noise term.

Robust estimation. Another approach replaces each stochastic gradient with a robust estimate computed from a batch, using, for example, coordinate-wise Catoni-type estimators (Holland and Ikeda, 2019) or median-of-means-type estimators (Prasad et al., 2020). Lugosi and Mendelson (2019) survey mean estimation under heavy tails. These analyses use batches to estimate the mean gradient, whereas the methods studied here clip individual stochastic gradients.

## C Weak- $L ^ { \alpha }$ and Marcinkiewicz Spaces

In this section, we recall the basic facts about weak- $L ^ { \alpha }$ and Marcinkiewicz spaces that we use in the paper. Rearrangement-invariant function spaces are the natural language for power-law tails. Marcinkiewicz spaces are one of the basic classes of such spaces, along with Lorentz and Orlicz spaces; we refer to the monographs by Kre˘ın et al. (1982), Bennett and Sharpley (1988), Lindenstrauss and Tzafriri (1979), and Braverman (1994) for the details. The connection between rearrangement-invariant spaces and probabilistic inequalities for independent random variables is surveyed by Astashkin and Sukochev (2010), and martingale inequalities in such spaces are studied by Kikuchi (1999, 2004). In this paper, we use the Marcinkiewicz space with fundamental function $\varphi ( t ) = t ^ { 1 / \alpha }$ ; as a set, it coincides with weak- $L ^ { \alpha } = L ^ { \alpha , \infty }$

Let Y be a real random variable on $( \Omega , \mathcal { F } , \mathbb { P } )$ with tail function $d _ { Y } ( u ) : = \mathbb { P } \{ | Y | > u \} , u \geq 0$ Its non-increasing rearrangement $Y ^ { * } : ( 0 , 1 ]  [ 0 , \infty ]$ and maximal rearrangement are defined as follows:

$$
Y ^ { \ast } ( t ) : = \operatorname* { i n f } \{ u > 0 : \ d _ { Y } ( u ) \leq t \} ,
$$

$$
Y ^ { * * } ( t ) : = \frac { 1 } { t } \int _ { 0 } ^ { t } Y ^ { * } ( s ) \mathrm { d } s .
$$

Definition C.1 (Weak- $L ^ { \alpha }$ quasi-norm). For $\alpha > 0$ , let

$$
\| Y \| _ { L ^ { \alpha , \infty } } : = \operatorname* { s u p } _ { u > 0 } u \mathbb { P } \{ | Y | > u \} ^ { 1 / \alpha } = \operatorname* { s u p } _ { 0 < t \leq 1 } t ^ { 1 / \alpha } Y ^ { * } ( t ) ,\tag{13}
$$

and let $L ^ { \alpha , \infty }$ be the set of all Y with $\| Y \| _ { L ^ { \alpha , \infty } } < \infty$ .

The functional (13) is positively homogeneous and rearrangement invariant, but, in general, it does not satisfy the triangle inequality, i.e., it is only a quasi-norm (Bennett and Sharpley, 1988). Nevertheless, for $\alpha > 1$ , the same class of random variables admits an equivalent norm.

Definition C.2 (Marcinkiewicz norm). For $\alpha > 1$ and a random variable $Y ,$ let $\| Y \| _ { M _ { \alpha } } : =$ $\mathrm { s u p } _ { 0 < t \leq 1 } t ^ { 1 / \alpha } Y ^ { \ast \ast } ( t )$ The space $M _ { \alpha }$ of all $Y$ with $\| Y \| _ { M _ { \alpha } } < \infty$ is the Marcinkiewicz space with fundamental function $\varphi ( t ) = t ^ { 1 / \alpha }$

Proposition C.3 (Equivalence of weak- $L ^ { \alpha }$ and $M _ { \alpha } )$ . For $1 < \alpha < \infty$ , the spaces $L ^ { \alpha , \infty }$ and $M _ { \alpha }$ coincide as sets and

$$
\Vert Y \Vert _ { L ^ { \alpha , \infty } } \leq \Vert Y \Vert _ { M _ { \alpha } } \leq \frac { \alpha } { \alpha - 1 } \left. Y \right. _ { L ^ { \alpha , \infty } } .\tag{14}
$$

Proposition C.4 (Fundamental function and monotonicity). For every event $A \in \mathcal { F } , \| \mathbf { 1 } _ { A } \| _ { M _ { \alpha } } =$ $\mathbb { P } ( A ) ^ { 1 / \alpha }$ If $| Y | \le | Z |$ almost surely, then $\| Y \| _ { M _ { \alpha } } \leq \| Z \| _ { M _ { \alpha } }$ The norm depends only on the distribution of $| Y | , { \mathrm { i . e . , } } M _ { \alpha }$ is rearrangement invariant.

Therefore, the absence of the triangle inequality for the tail quasi-norm does not cause any dificulties in the clipping lemma (Lemma D.1): it is convenient to prove the lemma using the quasi-norm, and one can always pass to the equivalent Marcinkiewicz norm. The next statement clarifies the exact relation between the moment and the weak- $L ^ { \alpha }$ settings.

Proposition C.5 (Embeddings). Let $\alpha > 0$

1. If $Y \in L ^ { \alpha }$ , then $\| Y \| _ { L ^ { \alpha , \infty } } \leq ( \mathbb { E } | Y | ^ { \alpha } ) ^ { 1 / \alpha }$ .

2. If $Y \in L ^ { \alpha , \infty }$ and $0 < q < \alpha$ , then $Y \in L ^ { q }$ and

$$
\mathbb { E } | Y | ^ { q } \leq \frac { \alpha } { \alpha - q } \| Y \| _ { L ^ { \alpha , \infty } } ^ { q } .\tag{15}
$$

In general, $Y \in L ^ { \alpha , \infty }$ does not imply $Y \in L ^ { \alpha }$

Remark C.6 (Direction of the implications). For the same exponent $\alpha ,$ weak- $L ^ { \alpha }$ control does not imply a bound on the $\alpha { - } \mathrm { t h }$ moment. The correct statements are the following: strong $L ^ { \alpha }$ control implies weak- $L ^ { \alpha }$ control with the same constant, and weak- $L ^ { \alpha }$ control implies finiteness of every moment of order $q < \alpha$ with the explicit constant (15). In particular, Assumption 3.5 implies Assumption 3.4 with $\alpha$ replaced by any $q \in ( 1 , \alpha )$ and $\sigma _ { j }$ replaced by $( \alpha / ( \alpha - q ) ) ^ { 1 / q } \sigma _ { j } .$ $j \in \{ 0 , 1 \}$ (it is suficient to use (15) and $( s + t ) ^ { q / \alpha } \leq s ^ { q / \alpha } + t ^ { q / \alpha } )$ . That is, it is the result obtained under the weaker assumption that is stronger.

## C.1 Proof of Proposition C.3

Proof. Since $Y ^ { * }$ is non-increasing, we have $\begin{array} { r } { Y ^ { * * } ( t ) = \frac { 1 } { t } \int _ { 0 } ^ { t } Y ^ { * } ( s ) \mathrm { d } s \geq Y ^ { * } ( t ) } \end{array}$ , which gives the first inequality in (14).

Next, let $M : = \| Y \| _ { L ^ { \alpha , \infty } }$ . If $M = \infty$ , the second inequality in (14) is immediate. Otherwise, $Y ^ { * } ( s ) \le M s ^ { - 1 / \alpha }$ for every $s \in ( 0 , 1 ]$ , and therefore

$$
Y ^ { \ast \ast } ( t ) \leq \frac { M } { t } \int _ { 0 } ^ { t } s ^ { - 1 / \alpha } \mathrm { d } s = \frac { \alpha } { \alpha - 1 } M t ^ { - 1 / \alpha } .
$$

Multiplying both sides by $t ^ { 1 / \alpha }$ and taking the supremum over $t ,$ we get the second inequality in (14). Finally, the completeness of the space with norm $\lVert \cdot \rVert _ { M _ { \alpha } }$ is a standard property of Marcinkiewicz spaces (Bennett and Sharpley, 1988; Braverman, 1994). Thus, for $\alpha > 1$ , the weak- $L ^ { \alpha }$ quasi-norm is equivalent to a Banach norm, which concludes the proof. □

## C.2 Proof of Proposition C.4

Proof. Let $s : = \mathbb { P } ( A )$ . For $Y = \mathbf { 1 } _ { A }$ , we have

$$
Y ^ { * } ( t ) = \left\{ { \begin{array} { l l } { 1 , } & { 0 < t < s , } \\ { 0 , } & { s \leq t \leq 1 , } \end{array} } \right. \quad \quad Y ^ { * * } ( t ) = \left\{ { \begin{array} { l l } { 1 , } & { 0 < t \leq s , } \\ { s / t , } & { s < t \leq 1 , } \end{array} } \right.
$$

since $d _ { Y } ( u ) = s$ for $0 < u < 1$ and $d _ { Y } ( u ) = 0$ for $u \geq 1$ , meaning that $Y ^ { * } ( t ) = 0$ as soon as $t \geq s$ (in particular, at $t = s )$ . Thus, $\begin{array} { r } { \operatorname* { s u p } _ { 0 < t < 1 } t ^ { 1 / \alpha } Y ^ { * * } ( t ) = s ^ { 1 / \alpha } } \end{array}$ . Next, if $| Y | \leq | Z | .$ , then $Y ^ { * } \le Z ^ { * }$ and, as a consequence, $Y ^ { * * } \le Z ^ { * * }$ , which gives the lattice monotonicity of the norm. Finally, if $| Y |$ and $| Z |$ are equidistributed, then $Y ^ { * } = Z ^ { * }$ , and the norms coincide, which concludes the proof. □

## C.3 Proof of Proposition C.5

Proof. The first claim follows from Markov’s inequality: $\mathbb { P } \{ | Y | > u \} = \mathbb { P } \{ | Y | ^ { \alpha } > u ^ { \alpha } \} \le \mathbb { E } | Y | ^ { \alpha } / u ^ { \alpha }$ Thus, $u \mathbb { P } \{ | Y | > u \} ^ { 1 / \alpha } \leq ( \mathbb { E } | Y | ^ { \alpha } ) ^ { 1 / \alpha }$ , and it remains to take the supremum over $u > 0$

For the second estimate, let $M : = \| Y \| _ { L ^ { \alpha , \infty } }$ . If $M = 0$ , then $Y = 0$ almost surely and the claim is immediate; if $M = \infty$ , the estimate with an infinite right-hand side is trivial. It remains to consider $0 < M < \infty$ . Then $\mathbb { P } \{ | Y | > u \} \le \operatorname* { m i n } \{ 1 , M ^ { \alpha } / u ^ { \alpha } \}$ , and the layer-cake representation gives

$$
\begin{array} { l } { \displaystyle \mathbb { E } | Y | ^ { q } = q \int _ { 0 } ^ { \infty } u ^ { q - 1 } \mathbb { P } \{ | Y | > u \} \mathrm { d } u \leq q \int _ { 0 } ^ { M } u ^ { q - 1 } \mathrm { d } u + q M ^ { \alpha } \int _ { M } ^ { \infty } u ^ { q - \alpha - 1 } \mathrm { d } u } \\ { \displaystyle \quad = M ^ { q } + \frac { q } { \alpha - q } M ^ { q } = \frac { \alpha } { \alpha - q } M ^ { q } . } \end{array}
$$

To prove the last claim for every $\alpha > 0 .$ , take a nonnegative random variable $Y$ with $\mathbb { P } \{ Y \ >$ $u \} = \operatorname* { m i n } \{ 1 , u ^ { - \alpha } \}$ for $u > 0$ . Then $\| Y \| _ { L ^ { \alpha , \infty } } = 1$ , whereas the layer-cake formula gives $\mathbb { E } Y ^ { \alpha } \geq$ $\begin{array} { r } { \dot { \alpha } \int _ { 1 } ^ { \infty } u ^ { - 1 } \dot { \mathrm { d } } u = \infty } \end{array}$ . This counterexample requires no centering or finite first moment and concludes the proof. □

## C.4 The Pareto Example

The following example, which we summarize in Section 3, shows that the strict inclusion $L ^ { \alpha } \subsetneq L ^ { \alpha , \infty }$ is attained by the simplest heavy-tailed oracle. For $L , \sigma > 0$ , we consider the one-dimensional quadratic problem $\begin{array} { r } { f ( x ) = \frac { L } { 2 } x ^ { 2 } , \nabla f ( x ) = L x . } \end{array}$ , with

$$
g ( x , \xi ) = L x + \xi , \qquad \xi = \sigma \varepsilon \mathsf { R } ,\tag{16}
$$

where $\varepsilon$ is a Rademacher variable independent of $\mathsf { R } ,$ , and R has the Pareto tail:

$$
\mathbb { P } \{ \mathsf { R } > u \} = \left\{ \begin{array} { l l } { 1 , } & { 0 < u < 1 , } \\ { u ^ { - \alpha } , } & { u \geq 1 , } \end{array} \right. \qquad 1 < \alpha \leq 2 .\tag{17}
$$

Proposition C.7 (Pareto noise at the critical index). For the noise $( 1 6 ) – ( 1 7 ) , \mathbb { E } \xi = 0 , \| \xi \| _ { L ^ { \alpha , \infty } } =$ $\sigma$ and $\mathbb { E } | \xi | ^ { \alpha } = \infty$ . Moreover, $\begin{array} { r } { \mathbb { E } | \xi | ^ { q } = \sigma ^ { q } \frac { \alpha } { \alpha - q } } \end{array}$ for every $0 < q < \alpha$

That is, the moment-based theory with exponent α is not applicable to this example, while Assumption 3.5 holds with $\sigma _ { 0 } = \sigma , \sigma _ { 1 } = 0$ , and the exact scale.

## C.5 Proof of Proposition C.7

Proof. By the independence of $\varepsilon$ and R and the symmetry of ε, we have $\mathbb { E } \xi = \sigma \mathbb { E } \varepsilon \mathbb { E } \mathsf { R } = 0$ , where $\mathbb { E } \mathsf { R } < \infty$ since $\alpha > 1$ . Next, since $| \xi | = \sigma \mathsf { R } .$ , for u $\geq \sigma$ we have $\mathbb { P } \{ | \xi | > u \} = \mathbb { P } \{ \mathsf { R } > u / \sigma \} = ( \sigma / u ) ^ { \alpha }$ and for $0 < u < \sigma$ this probability equals 1. Thus, $\begin{array} { r } { \operatorname* { s u p } _ { u > 0 } u \mathbb { P } \{ | \xi | > u \} ^ { 1 / \alpha } = \sigma } \end{array}$

Finally, the density of R on $[ 1 , \infty )$ is $p ( r ) = \alpha r ^ { - \alpha - 1 }$ . Therefore, $\begin{array} { r } { \mathbb { E } \mathbb { R } ^ { \alpha } = \alpha \int _ { 1 } ^ { \infty } r ^ { - 1 } \mathrm { d } r = \infty } \end{array}$ , while for $0 < q < \alpha$ we have $\begin{array} { r } { \mathbb { E } \mathbb { R } ^ { q } = \alpha \int _ { 1 } ^ { \infty } r ^ { q - \alpha - 1 } \mathrm { d } r = \frac { \alpha } { \alpha - q } } \end{array}$ . Multiplying the latter by $\sigma ^ { q }$ , we get the last claim, which concludes the proof. □

## D Proofs of the Clipping Lemmas

## D.1 Separate Statements and Remarks

In this section, we provide the proofs of the clipping lemmas. Lemma 4.1 summarizes the following two statements, which we prove in Sections D.2 and D.3; both of them also hold conditionally (see Corollary D.5).

Lemma D.1 (Clipping under the weak- $L ^ { \alpha }$ condition). Let $1 < \alpha \leq 2$ and $\lambda > 0$ , and let X be a random vector in $\mathbb { R } ^ { d }$ with $\mathbb { E } X = x$ and $\mathbb { P } \{ \| X - x \| > u \} \leq A ( x ) u ^ { - \alpha }$ for all $u > 0$ . Let $\widetilde { X } : = { \mathsf { c l i p } } ( X , \lambda )$ and assume $\| x \| \leq \lambda / 2$ Then $\| { \widetilde { X } } - \mathbb { E } { \widetilde { X } } \| \leq 2 \lambda$ almost surely, $\mathbb { E } \Vert \widetilde { X } - \mathbb { E } \widetilde { X } \Vert ^ { 2 } \leq$ $\mathbb { E } \| \widetilde { X } - x \| ^ { 2 }$

$$
\| \mathbb { E } \widetilde { X } - x \| \leq b _ { \alpha } \frac { A ( x ) } { \lambda ^ { \alpha - 1 } } ,\tag{18}
$$

$$
\mathbb { E } \Vert \widetilde { X } - x \Vert ^ { 2 } \leq v _ { \alpha } \biggl ( \frac { ( 3 \lambda / 2 ) ^ { \alpha } } { A ( x ) } \biggr ) \lambda ^ { 2 - \alpha } A ( x ) ,\tag{19}
$$

where the right-hand side of (19) is 0 if $A ( x ) = 0 ;$ , and $b _ { \alpha } , v _ { \alpha } ( \cdot )$ are defined in $( 6 ) \AA { - } \bigl ( 5 \bigr )$ .

Lemma D.2 (Clipping under the moment condition). Let $\lambda > 0$ , let X be a random vector in $\mathbb { R } ^ { d }$ , and let $\tilde { X } : = { \mathsf { c l i p } } ( X , \lambda )$ Then $\lVert \widetilde { X } - \mathbb { E } \widetilde { X } \rVert \leq 2 \lambda$ almost surely. If, in addition, $\mathbb { E } X = x$ $\mathbb { E } \| X - x \| ^ { \alpha } \leq A ( x ) <$ ∞ for some $\alpha \in ( 1 , 2 ]$ , and $\| x \| \leq \lambda / 2$ , then, with $b _ { \alpha } ^ { \mathrm { m o m } }$ and $v _ { \alpha } ^ { \mathrm { m o m } }$ from (6),

$$
\| \mathbb { E } \widetilde { X } - x \| \leq \frac { b _ { \alpha } ^ { \mathrm { m o m } } A ( x ) } { \lambda ^ { \alpha - 1 } } ,\tag{20}
$$

$$
\begin{array} { r } { \mathbb { E } \| \widetilde { X } - \mathbb { E } \widetilde { X } \| ^ { 2 } \leq \mathbb { E } \| \widetilde { X } - x \| ^ { 2 } \leq v _ { \alpha } ^ { \mathrm { m o m } } \lambda ^ { 2 - \alpha } A ( x ) . } \end{array}\tag{21}
$$

Here $1 / 2 \le b _ { \alpha } ^ { \mathrm { m o m } } \le 1$ and $1 \leq v _ { \alpha } ^ { \mathrm { m o m } } \leq 3 / 2 ;$ in particular, $b _ { 2 } ^ { \mathrm { m o m } } = 1 / 2$ and $v _ { 2 } ^ { \mathrm { m o m } } = 1$

The constant $v _ { \alpha } ( s )$ . For $\alpha < 2$ , we have $\begin{array} { r } { v _ { \alpha } ( s ) \le v _ { \alpha } : = v _ { \alpha } ( \infty ) = \frac { 2 } { 2 - \alpha } ( \frac { 3 } { 2 } ) ^ { 2 - \alpha } } \end{array}$ , and, therefore, (19) implies $\begin{array} { r } { \mathbb { E } \| \widetilde { X } - x \| ^ { 2 } \leq v _ { \alpha } \lambda ^ { 2 - \alpha } A ( x ) } \end{array}$ , where the constant does not depend on λ but blows up as $\alpha \uparrow 2$ . The logarithmic branch in (5) is finite for every $s < \infty$ , including the case $\alpha = 2$ . It is the smaller one if and only if ln $\begin{array} { r } { ( 1 + s ) < \frac { \alpha ^ { 2 } } { 2 ( 2 - \alpha ) } } \end{array}$ , i.e., unless λ exceeds the tail scale $A ( x ) ^ { 1 / \alpha }$ by a factor exponential in $( 2 - \alpha ) ^ { - 1 }$ . When the noise scale varies along the trajectory, the logarithms are combined using concavity: $s \mapsto s v _ { \alpha } ( C / s )$ is concave and non-decreasing on $[ 0 , \infty )$ for every $C \geq 0$ (see Lemma D.6 in Appendix D), and, therefore, only the average noise scale enters the logarithm.

Remark D.3 (The endpoint $\alpha = 2 )$ . At $\alpha = 2$ and $A ( x ) > 0$ Lemma D.1 gives $\mathbb { E } \| \widetilde { X } - x \| ^ { 2 } \leq$ $A ( x ) \bigl ( 1 { + } \ln \left( 1 { + } \frac { 9 \lambda ^ { 2 } } { 4 A ( x ) } \right) \bigr )$ ; the bound is zero if $A ( x ) = 0$ The logarithm is necessary even for centered scalar noise. Indeed, let $A > 0 , u _ { 0 } : = A ^ { 1 / \alpha }$ , and let $Y \geq 0$ satisfy $\mathbb { P } \{ Y > u \} = \operatorname* { m i n } \{ 1 , A / u ^ { \alpha } \}$ with $1 < \alpha \leq 2$ . For an independent Rademacher sign ϵ, the variable $X : = \epsilon Y$ has mean zero and weak- $L ^ { \alpha }$ quasi-norm $A ^ { 1 / \alpha }$ . For $\lambda \geq u _ { 0 }$ , direct integration gives

$$
\mathbb { E } | \mathsf { c } \mathtt { l } \mathtt { i } \mathtt { p } ( X , \lambda ) | ^ { 2 } = \mathbb { E } \operatorname* { m i n } \{ Y ^ { 2 } , \lambda ^ { 2 } \} = \left\{ \begin{array} { l l } { \displaystyle u _ { 0 } ^ { 2 } + \frac { 2 A } { 2 - \alpha } \big ( \lambda ^ { 2 - \alpha } - u _ { 0 } ^ { 2 - \alpha } \big ) , } & { 1 < \alpha < 2 , } \\ { A \big ( 1 + 2 \ln ( \lambda / \sqrt { A } ) \big ) , } & { \alpha = 2 . } \end{array} \right.
$$

At $\alpha = 2$ the second expression grows logarithmically with $\lambda ,$ while $\mathbb { E } X ^ { 2 } = \infty$ . For $\alpha < 2$ dividing the first expression by $A \lambda ^ { 2 - \alpha }$ and taking the supremum over $\lambda \geq u _ { 0 }$ gives $2 / ( 2 - \alpha )$ . Thus, any second-moment bound of the form $v A \lambda ^ { 2 - \alpha }$ with v independent of λ must have $v \geq 2 / ( 2 - \alpha )$

For each fixed ratio $\lambda / u _ { 0 }$ , the expression instead has the finite logarithmic limit as $\alpha \uparrow 2 .$ In contrast, Lemma D.2 holds at $\alpha = 2$ with $v _ { \mathrm { 2 } } ^ { \mathrm { m o m } } = 1$

Remark D.4 (The endpoint $\alpha = 1 )$ . The factor $( \alpha - 1 ) ^ { - 1 }$ in the bias bound is necessary within the class of centered noise with a finite mean. To see this, fix $1 < \alpha \leq 2$ and let

$$
\lambda _ { \alpha } : = \left( \frac { 2 \alpha } { \alpha - 1 } \right) ^ { 1 / ( \alpha - 1 ) } , \qquad p : = \lambda _ { \alpha } ^ { - \alpha } , \qquad d : = \frac { 1 } { 2 ( 1 - p ) } .
$$

Let $Y \ge \lambda _ { \alpha }$ have the Pareto tail $\mathbb { P } \{ Y > u \} = ( \lambda _ { \alpha } / u ) ^ { \alpha }$ for $u \geq \lambda _ { \alpha }$ , and let X equal an independent copy of Y with probability p and equal −d otherwise. Since $p \mathbb { E } Y = { p \alpha \lambda _ { \alpha } } / ( \alpha - 1 ) = 1 / 2 = ( 1 - p ) d ,$ we have E $X = 0$ . Also, $\lambda _ { \alpha } \geq 4 , p \leq 1 / 4$ , and $d < 1$ . For $u < d$ the tail bound $\mathbb { P } \{ | X | > u \} \le u ^ { - \alpha }$ is trivial; for $d \leq u < \lambda _ { \alpha }$ it follows from $p \leq u ^ { - \alpha }$ ; and for $u \geq \lambda _ { \alpha }$ it is an equality. Therefore, $\| X \| _ { L ^ { \alpha , \infty } } = 1$ , while

$$
\left| \mathbb { E } \mathsf { c } \mathsf { l } \mathsf { i p } ( X , \lambda _ { \alpha } ) \right| = \frac { 1 } { 2 } - p \lambda _ { \alpha } = \frac { 1 } { 2 \alpha } = \frac { \lambda _ { \alpha } ^ { 1 - \alpha } } { \alpha - 1 } .
$$

Consequently, any bias constant valid uniformly over centered weak- $L ^ { \alpha }$ noise is at least $1 / ( \alpha - 1 )$ $\mathrm { A t } ~ \alpha = 1$ , weak-tail control alone need not even give a finite mean: for $A > 0$ and $\mathbb { P } \{ Y > u \} =$ min $\{ 1 , A / u \}$ , we have $\begin{array} { r } { \mathbb { E } ( Y - a ) _ { + } = \int _ { a } ^ { \infty } \mathbb { P } \{ Y > u \} \mathrm { d } u = \infty } \end{array}$ for every $a > 0$

## D.2 Proof of Lemma D.1

Proof. Let $Y : = \| X - x \|$ and $A : = A ( x )$ . By assumption,

$$
{ \mathbb P } \{ Y > u \} \leq A u ^ { - \alpha } , \qquad u > 0 .\tag{22}
$$

Step 1: boundedness. By the definition of clipping, we have $\| \widetilde { X } \| \leq \lambda$ almost surely. By Jensen’s inequality, $\| \mathbb { E } \widetilde { X } \| \leq \mathbb { E } \| \widetilde { X } \| \leq \lambda$ . Thus, $\| \widetilde { X } - \mathbb { E } \widetilde { X } \| \le \| \widetilde { X } \| + \| \mathbb { E } \widetilde { X } \| \le 2 \lambda$

Step 2: bias. Since $\mathbb { E } X = x$ , we have $\| \mathbb { E } \widetilde { X } - x \| = \| \mathbb { E } ( \widetilde { X } - X ) \| \le \mathbb { E } \| \widetilde { X } - X \|$ . For the clipping operator, we have $\| \widetilde { X } - X \| = ( \| X \| - \lambda ) _ { + }$ . Since $\| x \| \leq \lambda / 2$ , we have $\| X \| \le Y + \lambda / 2$ and, therefore, $( \| X \| - \lambda ) _ { + } \leq ( Y - \lambda / 2 ) _ { + }$ . Using the identity $\begin{array} { r } { \mathbb { E } ( Y - a ) _ { + } = \int _ { a } ^ { \infty } \mathbb { P } \{ Y > u \} } \end{array}$ du with $a = \lambda / 2$ and (22), we get

$$
\| \mathbb { E } \widetilde { X } - x \| \le A \int _ { \lambda / 2 } ^ { \infty } u ^ { - \alpha } \mathrm { d } u = \frac { 2 ^ { \alpha - 1 } } { \alpha - 1 } \frac { A } { \lambda ^ { \alpha - 1 } } .
$$

Step 3: second moment around x. We start with showing that

$$
\begin{array} { r } { \| \widetilde { X } - x \| \le \operatorname* { m i n } \left\{ \| X - x \| , \ \frac { 3 \lambda } { 2 } \right\} . } \end{array}\tag{23}
$$

The bound by $3 \lambda / 2$ follows from $\| { \widetilde { X } } \| \leq \lambda$ and $\| x \| \leq \lambda / 2$ . For the first bound, if $\| X \| \leq \lambda$ , then ${ \widetilde { X } } = X$ . Otherwise, let $s : = \| X \| > \lambda$ and $e : = X / s ;$ then $\widetilde X = \lambda e$ and

$$
\| X - x \| ^ { 2 } - \| \lambda e - x \| ^ { 2 } = ( s - \lambda ) ( s + \lambda - 2 \langle e , x \rangle ) \geq 0 ,
$$

since $\langle e , x \rangle \leq \| x \| \leq \lambda / 2$ . Next, let $a : = 3 \lambda / 2$ . By (23) and the layer-cake representation, we have

$$
\mathbb { E } \Vert \widetilde { X } - x \Vert ^ { 2 } \leq \mathbb { E } \operatorname* { m i n } \{ Y ^ { 2 } , a ^ { 2 } \} = 2 \int _ { 0 } ^ { a } u \mathbb { P } \{ Y > u \} \mathrm { d } u .
$$

If $A = 0$ , then $Y = 0$ almost surely and both sides vanish; therefore, we assume that $A > 0$ . For $\alpha < 2 , ( 2 2 )$ gives E min $\begin{array} { r } { \{ Y ^ { 2 } , a ^ { 2 } \} \le 2 A \int _ { 0 } ^ { a } u ^ { 1 - \alpha } \mathrm { d } u = \frac { 2 } { 2 - \alpha } A a ^ { 2 - \alpha } } \end{array}$ . Next, we consider the logarithmic branch and let $u _ { 0 } : = A ^ { 1 / \alpha }$ . If $a \leq u _ { 0 }$ , then E min $\{ Y ^ { 2 } , a ^ { 2 } \} \leq a ^ { 2 } \leq A a ^ { 2 - \alpha }$ , since $a ^ { \alpha } \leq A$ . If $a > u _ { 0 }$ we use $\mathbb { P } \{ Y > u \} \le 1$ on $( 0 , u _ { 0 } )$ and (22) on $( u _ { 0 } , a )$

$$
\begin{array} { l } { { \displaystyle \mathbb { E } \operatorname* { m i n } \{ Y ^ { 2 } , a ^ { 2 } \} \leq u _ { 0 } ^ { 2 } + 2 A \int _ { u _ { 0 } } ^ { a } u ^ { 2 - \alpha } \frac { \mathrm { d } u } { u } \leq A u _ { 0 } ^ { 2 - \alpha } + 2 A a ^ { 2 - \alpha } \ln \frac { a } { u _ { 0 } } } } \\ { { \displaystyle \leq A a ^ { 2 - \alpha } \left( 1 + \frac { 2 } { \alpha } \ln \left( 1 + \frac { a ^ { \alpha } } { A } \right) \right) , } } \end{array}
$$

since $u _ { 0 } ^ { 2 } = A u _ { 0 } ^ { 2 - \alpha } \le A a ^ { 2 - \alpha }$ and $\textstyle \ln ( a / u _ { 0 } ) = { \frac { 1 } { \alpha } } \ln ( a ^ { \alpha } / A )$ . In all cases, we have E min $\{ Y ^ { 2 } , a ^ { 2 } \} \leq$ $A a ^ { 2 - \alpha }$ min $\textstyle \{ { \frac { 2 } { 2 - \alpha } } , 1 + { \frac { 2 } { \alpha } } \ln ( 1 + { \frac { a ^ { \alpha } } { A } } ) \} = v _ { \alpha } ( a ^ { \alpha } / A ) \lambda ^ { 2 - \alpha } A$ , which is exactly (19). $S t e p \ 4 \colon$ centered second moment. Finally, in the Euclidean space, we have $\mathbb { E } \Vert \widetilde { X } - x \Vert ^ { 2 } =$ $\mathbb { E } \| \widetilde { X } - \mathbb { E } \widetilde { X } \| ^ { 2 } + \| \mathbb { E } \widetilde { X } - x \| ^ { 2 }$ , implying E∥ $| \widetilde { X } - \bar { \mathbb { E } } \widetilde { X } | | ^ { 2 } \leq \mathbb { E } \| \widetilde { X } - x \| ^ { 2 }$ , which concludes the proof.

Corollary D.5 (Conditional versions). Let G be a σ-algebra, let x be G-measurable, and let $\mathbb { E } [ X \mid { \mathcal { G } } ] = x$ and $\| x \| \leq \lambda / 2$ . If, almost surely, $\mathbb { P } \{ \| X - x \| > u \mid \mathcal { G } \} \leq A ( x ) u ^ { - \alpha }$ for all $u > 0$ (resp. $\mathbb { E } [ \| X - x \| ^ { \alpha } \ | \ { \mathcal { G } } ] \leq A ( x ) )$ , then the conclusions of Lemma D.1 (resp. Lemma D.2) hold almost surely with E[·] replaced by $\mathbb { E } [ \cdot \mid { \mathcal { G } } ] ;$ ; in particular, $\| \widetilde { X } - \mathbb { E } [ \widetilde { X } \mid \mathcal { G } ] \| \leq 2 \lambda$ almost surely.

Proof. All steps of the proofs of Lemmas D.1 and D.2 remain valid after replacing expectations and probabilities with their conditional counterparts given G. The quantity $A ( x )$ is G-measurable and, therefore, plays the role of a fixed coeficient in the conditional integral bounds. The conditional tail bounds can be assumed to hold for all $u > 0$ simultaneously on one event of full probability: it is suficient to take versions for rational u and extend them by monotonicity and right-continuity. Then, the conditional Tonelli theorem gives, for deterministic $\begin{array} { r } { c > 0 , \mathbb { E } [ ( Y - c ) _ { + } \mid \mathcal { G } ] = \int _ { c } ^ { \infty } \mathbb { P } \{ Y > u \mid } \end{array}$ G} du and E[min $\begin{array} { r } { \{ Y ^ { 2 } , c ^ { 2 } \} \mid \mathcal { G } \} = 2 \int _ { 0 } ^ { c } u \mathbb { P } \{ Y > u \mid \mathcal { G } \} } \end{array}$ du, which we use with $c = \lambda / 2$ and $c = 3 \lambda / 2$ For Lemma D.2, the two pointwise inequalities from its proof have deterministic coeficients and remain valid after taking conditional expectations, with $A ( x )$ replaced by any finite G-measurable upper bound on $\mathbb { E } [ \| X - x \| ^ { \alpha } \mid { \mathcal { G } } ]$ . Finally, the conditional variance decomposition reads $\mathbb { E } [ \| \widetilde { X } - x \| ^ { 2 }$ $\mathcal { G } \mathbf { \bar { ] } } = \mathbb { E } [ \| \widetilde { X } - \mathbb { E } [ \widetilde { X } \ | \ \mathcal { G } ] \| ^ { 2 } \ | \mathcal { G } ] + \| \mathbb { E } [ \widetilde { X } \ | \ \mathcal { G } ] - x \| ^ { 2 }$ , which concludes the proof. □

## D.3 Proof of Lemma D.2

Proof. The bound $\lVert \widetilde { X } - \mathbb { E } \widetilde { X } \rVert \leq 2 \lambda$ is proved in the same way as in Step 1 of the proof of Lemma D.1. Let $Y : = \| X - x \|$ and $c : = \lambda / 2$ . As in Step 2 of the proof of Lemma D.1, we have $\| \mathbb { E } \widetilde { X } - x \| \leq$ $\mathbb { E } ( Y - c ) _ { + }$ . For $y > c$ , we have

$$
\frac { \mathrm { d } } { \mathrm { d } y } \frac { y - c } { y ^ { \alpha } } = y ^ { - \alpha - 1 } \big ( \alpha c - ( \alpha - 1 ) y \big ) ,
$$

meaning that the maximum of $\left( y - c \right) _ { + } / y ^ { \alpha }$ over $y > 0$ is attained at $y = \alpha c / ( \alpha - 1 )$ and equals $c ^ { 1 - \alpha } ( \alpha - 1 ) ^ { \alpha - 1 } / \alpha ^ { \alpha }$ . Thus, $\begin{array} { r } { ( Y - c ) _ { + } \leq \frac { ( \alpha - 1 ) ^ { \alpha - 1 } } { \alpha ^ { \alpha } } c ^ { 1 - \alpha } Y ^ { \alpha } } \end{array}$ pointwise, and, taking expectations with $c ^ { 1 - \alpha } = 2 ^ { \alpha - 1 } \lambda ^ { 1 - \alpha }$ , we get (20).

Next, to bound the second moment, we use the contraction (23) (its proof uses only $\| x \| \leq \lambda / 2 )$ , which gives $\| \widetilde { X } - x \| ^ { 2 } \leq \operatorname* { m i n } \{ Y ^ { 2 } , d ^ { 2 } \}$ with $d : = 3 \lambda / 2$ . For $y \ge 0 , d > 0$ , and $1 < \alpha \leq 2$ , we have

$$
\operatorname* { m i n } \{ y ^ { 2 } , d ^ { 2 } \} \leq d ^ { 2 - \alpha } y ^ { \alpha } :
$$

for $y \leq d ,$ , this follows from $y ^ { 2 - \alpha } \leq d ^ { 2 - \alpha }$ , and, for $y \geq d ,$ from $d ^ { \alpha } \leq y ^ { \alpha }$ . Taking expectations, we get $\mathbb { E } \Vert \widetilde { X } - x \Vert ^ { 2 } \leq ( 3 / 2 ) ^ { 2 - \alpha } \lambda ^ { 2 - \alpha } A ( x )$ , which is the last inequality in (21). The bound on $\mathbb { E } \| \widetilde { X } - \mathbb { E } \widetilde { X } \| ^ { 2 }$ follows from the variance decomposition, as in Step 4 of the proof of Lemma D.1. Finally, $\frac { \mathrm { d } } { \mathrm { d } \alpha }$ ln $\begin{array} { r } { b _ { \alpha } ^ { \mathrm { m o m } } ~ = ~ \ln \frac { 2 ( \alpha - 1 ) } { \alpha } ~ \leq ~ 0 } \end{array}$ on $( 1 , 2 ]$ and $b _ { \alpha } ^ { \mathrm { m o m } } \ \to \ 1 \ \mathrm { a s } \ \alpha \ \downarrow \ 1$ , and, therefore, $1 / 2 \ =$ $b _ { 2 } ^ { \mathrm { m o m } } \leq b _ { \alpha } ^ { \mathrm { m o m } } \leq 1$ ; the bounds on $v _ { \alpha } ^ { \mathrm { m o m } }$ follow immediately from the definition, which concludes the proof. □

## D.4 Summing the Variance Bounds

Lemma D.6 (Summing the variance bounds). Let $\alpha \in ( 1 , 2 ]$ and $C \geq 0$ , and let $\phi ( s ) : = s v _ { \alpha } ( C / s )$ for $s > 0$ and $\phi ( 0 ) : = 0$ . Then ϕ is concave and non-decreasing on $[ 0 , \infty )$ Consequently, for all $T \geq 1 , \bar { B } > 0$ and $A _ { 0 } , \dotsc , A _ { T - 1 } \geq 0$ with $\textstyle \sum _ { t < T } A _ { t } \leq { \bar { B } }$

$$
\sum _ { t = 0 } ^ { T - 1 } \phi ( A _ { t } ) \leq \bar { B } v _ { \alpha } \bigg ( \frac { T C } { \bar { B } } \bigg ) .\tag{24}
$$

Proof. For $s > 0 .$ , we have $\begin{array} { r } { \phi ( s ) = ( \frac { 3 } { 2 } ) ^ { 2 - \alpha } \operatorname* { m i n } \{ \frac { 2 s } { 2 - \alpha } , \ : s + \frac { 2 } { \alpha } h ( s ) \} } \end{array}$ with $h ( s ) : = s \ln ( 1 + C / s )$ , where the first term is absent for $\alpha = 2$ . If $C = 0$ , then $h \equiv 0 . { \mathrm { ~ I f ~ } } C > 0$ , then $\begin{array} { r } { h ^ { \prime } ( s ) = \ln ( 1 + \frac { C } { s } ) - \frac { C } { s + C } > 0 } \end{array}$ , since $\textstyle \ln ( 1 + y ) > { \frac { y } { 1 + y } }$ for $y > 0$ , and $\begin{array} { r } { h ^ { \prime \prime } ( s ) = - \frac { C ^ { 2 } } { s ( s + C ) ^ { 2 } } < 0 } \end{array}$ . Moreover, $0 \leq h ( s ) = s \ln ( s + C ) - s \ln s \to 0$ as $s \downarrow 0$ . Thus, with $h ( 0 ) : = 0$ , the function h is continuous, concave, and non-decreasing on $[ 0 , \infty )$ . Both branches defining $\phi$ have these properties. Their minimum is concave because its hypograph is the intersection of their convex hypographs; it is also continuous and non-decreasing. The same conclusion holds when only one branch is present. Next, let $\begin{array} { r } { B : = \sum _ { t < T } A _ { t } ; } \end{array}$ ; then $B \le \bar { B }$ by assumption $( \hat { B }$ is the upper bound from the statement, and $\phi$ is non-decreasing). By Jensen’s inequality and monotonicity, we get $\begin{array} { r } { \sum _ { t < T } \phi ( A _ { t } ) \le T \phi ( B / T ) \le T \phi ( \bar { B } / T ) = \bar { B } v _ { \alpha } ( T C / \bar { B } ) } \end{array}$ , which concludes the proof. □

## E Auxiliary Results

Measurability conventions. In the setting of Section 3, the oracle $g ( x , \xi )$ is a jointly measurable and finite vector field with $\mathbb { E } _ { \xi } [ g ( x , \xi ) ] = \nabla f ( x )$ for $x \in Q ;$ we do not assume any sample-wise gradient representation. The noise $\boldsymbol { \mathfrak { g } } ( \boldsymbol { x } , \boldsymbol { \xi } ) - \nabla \boldsymbol { f } ( \boldsymbol { x } )$ is controlled only at the points $x \in Q ;$ ; Lemma I.3 (see Appendix I.2) transfers these pointwise conditions to random query points in $Q$ . Outside Q, $g$ is an arbitrary jointly measurable and finite extension, $f$ is not necessarily diferentiable, and, for definiteness, we set $\nabla f ( x ) : = 0$ there; this extension is used only to define auxiliary random variables on the whole probability space, and it never enters the bounds, which are derived on the events where the relevant points belong to $Q$

We assume that $f : \mathbb { R } ^ { d } $ R is Borel measurable; no diferentiability, smoothness, or convexity is required outside $Q$ . Consequently, the objective gaps used in the auxiliary random variables and stopping times are measurable on the whole probability space.

The gradient bound. Assumption 3.2 is stated separately to simplify the comparison with Sadiev et al. (2023); however, it is implied by the other two local assumptions, since $Q$ is a ball centered at the minimizer.

Lemma E.1 (Gradient bound from local smoothness and convexity). Let f be diferentiable on $Q = B _ { r } ( x ^ { * } ) , r > 0$ , where $x ^ { * }$ is a minimizer of f on $\mathbb { R } ^ { d }$ , and let Assumptions 3.1 and 3.3 hold on Q. Then $\| \nabla f ( x ) \| ^ { 2 } \leq 2 L ( f ( x ) - f ^ { * } )$ for all $x \in Q$

Proof. Step 1: cocoercivity on $Q$ . We claim that $\begin{array} { r l } { \langle \nabla f ( x ) - \nabla f ( z ) , x - z \rangle \geq \frac { 1 } { L } \| \nabla f ( x ) - \nabla f ( z ) \| ^ { 2 } } \end{array}$ for all $x , z \in Q . \ ( \mathrm { a } )$ First, we assume that $f$ is twice continuously diferentiable on an open convex set U on which Assumptions 3.1 and 3.3 hold, and let $x , z \in U$ . Then $\nabla f ( x ) - \nabla f ( z ) = A ( x - z )$ with $\begin{array} { r } { A : = \int _ { 0 } ^ { 1 } \nabla ^ { 2 } f ( z + t ( x - z ) ) } \end{array}$ dt. Convexity gives $\nabla ^ { 2 } f \succeq 0$ on $U$ and the Lipschitz bound gives $\nabla ^ { 2 } f \preceq L I$ on $U ;$ hence, $0 \preceq A \preceq L I$ . Since A and $L I - A$ commute, we have $A ( L I - A ) \succeq 0$ ${ \mathrm { i . e . , ~ } } A ^ { 2 } \preceq L A$ , and, therefore, $\| A ( x - z ) \| ^ { 2 } \leq L \left. A ( x - z ) , x - z \right.$ , which is exactly the claim. (b) In the general case, let $\rho _ { \delta }$ be a smooth probability density supported in $B _ { \delta } ( 0 )$ and $f _ { \delta } : = f * \rho _ { \delta }$ on $U _ { \delta } : = \{ x : \| x - x ^ { * } \| < r - \delta \}$ . For $x \in U _ { \delta }$ and $\| u \| \leq \delta ,$ , we have $x - u \in Q$ , meaning that $f _ { \delta }$ is welldefined and smooth on $U _ { \delta } .$ , convex on $U _ { \delta }$ as an average of the convex functions $x \mapsto f ( x - u )$ , and $\nabla f _ { \delta } = ( \nabla f ) * \rho _ { \delta }$ is L-Lipschitz on $U _ { \delta } ;$ moreover, $\| \nabla f _ { \delta } ( x ) - \nabla f ( x ) \| \leq L \delta$ for $x \in U _ { \delta }$ . Applying (a) to $f _ { \delta }$ on $U _ { \delta }$ and letting $\delta \to 0$ , we get the claim for all $x , z$ from the open ball $\{ \| x - x ^ { * } \| < r \}$ , and then for all $x , z \in Q$ by the continuity of $\nabla f$ on $Q$

Step 2. Let $x \in Q$ . Since $x ^ { * }$ is an interior point of Q minimizing $f \mathrm { ~ o n ~ } \mathbb { R } ^ { d }$ , we have $\nabla f ( x ^ { * } ) = 0$ and Step 1 with $z = x ^ { * }$ gives $\begin{array} { r l } { \langle \nabla f ( x ) , x - x ^ { * } \rangle \geq \frac { 1 } { L } \| \nabla f ( x ) \| ^ { 2 } } & { { } } \end{array}$ . Next, let $\begin{array} { r } { y : = x - \frac { 1 } { L } \nabla f ( x ) } \end{array}$ . Then

$$
\| y - x ^ { * } \| ^ { 2 } = \| x - x ^ { * } \| ^ { 2 } - { \frac { 2 } { L } } \left. \nabla f ( x ) , x - x ^ { * } \right. + { \frac { 1 } { L ^ { 2 } } } \| \nabla f ( x ) \| ^ { 2 } \leq \| x - x ^ { * } \| ^ { 2 } - { \frac { 1 } { L ^ { 2 } } } \| \nabla f ( x ) \| ^ { 2 } \leq r ^ { 2 } ,
$$

meaning that $y \in Q$ and the segment $[ x , y ]$ belongs to $Q$ . Along this segment, the gradient is L-Lipschitz; hence, $\begin{array} { r } { f ( y ) \leq f ( x ) + \langle \nabla f ( x ) , y - x \rangle + \frac { L } { 2 } \| y - x \| ^ { 2 } = f ( x ) - \frac { 1 } { 2 L } \| \nabla f ( x ) \| ^ { 2 } } \end{array}$ , and $f ( y ) \geq f ^ { * }$ concludes the proof. □

Throughout the appendix, we use the following decomposition of the error of the clipped stochastic gradient (with the convention $\nabla f : = 0$ outside $Q ,$ meaning that all quantities below are defined on the whole probability space):

$$
\theta _ { k } : = \widetilde { g } ^ { k } - \nabla f ( { x } ^ { k } ) = \theta _ { k } ^ { u } + \theta _ { k } ^ { b } , \qquad \theta _ { k } ^ { u } : = \widetilde { g } ^ { k } - \mathbb { E } [ \widetilde { g } ^ { k } \ \vert \ \mathcal { G } _ { k } ] , \qquad \theta _ { k } ^ { b } : = \mathbb { E } [ \widetilde { g } ^ { k } \ \vert \ \mathcal { G } _ { k } ] - \nabla f ( { x } ^ { k } ) ,\tag{25}
$$

where $\mathbb { E } [ \theta _ { k } ^ { u } \mid \mathcal { G } _ { k } ] = 0$ by construction.

Lemma E.2 (Basic inequality). Let Assumptions 3.2 and 3.3 hold on $Q _ { i }$ let $0 < \gamma \leq 1 / ( 2 L )$ and let $x ^ { 0 } , \ldots , x ^ { t - 1 } \in Q$ for some integer $t \geq 1$ . Then

$$
\gamma \sum _ { l = 0 } ^ { t - 1 } \left( f ( x ^ { l } ) - f ^ { * } \right) \leq \| x ^ { 0 } - x ^ { * } \| ^ { 2 } - \| x ^ { t } - x ^ { * } \| ^ { 2 } - 2 \gamma \sum _ { l = 0 } ^ { t - 1 } \left. x ^ { l } - x ^ { * } - \gamma \nabla f ( x ^ { l } ) , \theta _ { l } \right. + \gamma ^ { 2 } \sum _ { l = 0 } ^ { t - 1 } \| \theta _ { l } \| ^ { 2 } ,\tag{26}
$$

and the left-hand side of (26) is at least $\gamma t \left( f ( \bar { x } ^ { t - 1 } ) - f ^ { * } \right)$ , where $\begin{array} { r } { \bar { x } ^ { t - 1 } : = \frac { 1 } { t } \sum _ { l = 0 } ^ { t - 1 } x ^ { l } } \end{array}$

Proof. By the update rule, we have $x ^ { l + 1 } = x ^ { l } - \gamma \nabla f ( x ^ { l } ) - \gamma \theta _ { l }$ . Expanding the squared distance to $x ^ { * }$ , we get

$$
\begin{array} { r l } & { \| x ^ { l + 1 } - x ^ { * } \| ^ { 2 } = \| x ^ { l } - x ^ { * } \| ^ { 2 } - 2 \gamma \left. x ^ { l } - x ^ { * } , \nabla f ( x ^ { l } ) \right. + \gamma ^ { 2 } \| \nabla f ( x ^ { l } ) \| ^ { 2 } } \\ & { \qquad - 2 \gamma \left. x ^ { l } - x ^ { * } - \gamma \nabla f ( x ^ { l } ) , \theta _ { l } \right. + \gamma ^ { 2 } \| \theta _ { l } \| ^ { 2 } . } \end{array}
$$

By convexity, we have $\langle x ^ { l } - x ^ { * } , \nabla f ( x ^ { l } ) \rangle \geq f ( x ^ { l } ) - f ^ { * }$ . By Assumption 3.2 and $\gamma \leq 1 / ( 2 L )$ , we also have $\gamma ^ { 2 } \| \nabla f ( x ^ { l } ) \| ^ { 2 } \le 2 \dot { L } \gamma ^ { 2 } \big ( f ( x ^ { l } ) - f ^ { * } \big ) \overset { , } { \le } \gamma \big ( f ( x ^ { l } ) - f ^ { * } \big )$ . Therefore,

$$
\begin{array} { r } { \gamma \left( f ( x ^ { l } ) - f ^ { * } \right) \leq \Vert x ^ { l } - x ^ { * } \Vert ^ { 2 } - \Vert x ^ { l + 1 } - x ^ { * } \Vert ^ { 2 } - 2 \gamma \left. x ^ { l } - x ^ { * } - \gamma \nabla f ( x ^ { l } ) , \theta _ { l } \right. + \gamma ^ { 2 } \Vert \theta _ { l } \Vert ^ { 2 } . } \end{array}
$$

Summing $\mathrm { u p }$ the above inequality over $l = 0 , \ldots , t - 1$ , we get (26). The last claim follows from Jensen’s inequality for the convex function $f$ on the convex set $Q \ni \bar { x } ^ { t - 1 }$ , which concludes the proof. □

Lemma E.3 (Freedman’s inequality). Let $( \mathcal { F } _ { i } ) _ { i \geq 0 }$ be a filtration and let $X _ { i }$ be F<sub>i</sub>-measurable with $\mathbb { E } [ X _ { i } \mid { \mathcal { F } } _ { i - 1 } ] = 0$ and $| X _ { i } | \le c$ almost surely. Let $v _ { i } : = \mathbb { E } [ X _ { i } ^ { 2 } \mid \mathcal { F } _ { i - 1 } ]$ . Then for all $b , G > 0$

$$
\mathbb { P } \left\{ \left| \sum _ { i = 1 } ^ { n } X _ { i } \right| > b \mathrm { a n d } \sum _ { i = 1 } ^ { n } v _ { i } \leq G \right\} \leq 2 \exp \left( - \frac { b ^ { 2 } } { 2 G + 2 c b / 3 } \right) .\tag{27}
$$

The same bound holds for the larger event $\begin{array} { r } { \left\{ \exists m \ \leq \ n : \ | \sum _ { i = 1 } ^ { m } X _ { i } | > b \right. } \end{array}$ and $\textstyle \sum _ { i = 1 } ^ { m } v _ { i } \leq G \}$ (maximal form).

Proof. This is the martingale version of Bernstein’s inequality (Freedman, 1975; Dzhaparidze and Van Zanten, 2001); for completeness, we provide the full proof. For $c = 0$ , the claim is trivial; therefore, we assume that $c > 0$ . Since m! $\geq 2 \cdot 3 ^ { m - 2 }$ for $m \geq 2$ , for $| x | \le c$ and $0 < \mu < 3 / c$ we have

$$
e ^ { \mu x } \leq 1 + \mu x + \frac { \mu ^ { 2 } x ^ { 2 } } { 2 } \sum _ { j = 0 } ^ { \infty } \left( \frac { \mu c } { 3 } \right) ^ { j } = 1 + \mu x + \frac { \mu ^ { 2 } x ^ { 2 } } { 2 ( 1 - \mu c / 3 ) } .
$$

Let $\begin{array} { r } { \psi ( \mu ) : = \frac { \mu ^ { 2 } } { 2 ( 1 - \mu c / 3 ) } } \end{array}$ . Taking conditional expectations and using $1 + u \leq e ^ { u }$ , we get $\mathbb { E } [ e ^ { \mu X _ { i } } \mid { \mathcal { F } } _ { i - 1 } ] \leq$ $\exp ( \psi ( \mu ) v _ { i } )$ . Thus, $\begin{array} { r } { M _ { m } : = \exp \left( \mu \sum _ { i \leq m } X _ { i } - \psi ( \mu ) \sum _ { i \leq m } v _ { i } \right) } \end{array}$ is a non-negative supermartingale with $M _ { 0 } = 1$ , implying $\mathbb { E } M _ { n } \le 1$ . On the event $\{ \sum _ { i < n } X _ { i }  > b , \ \sum _ { i < n } v _ { i } \leq G \}$ , we have $M _ { n } \mathrm { ~ > ~ }$ $e ^ { \mu b - \psi ( \mu ) G } $ ; by Markov’s inequality, the probability of this event is at most $e ^ { - \mu b + \psi ( \mu ) G }$ . The choice $\begin{array} { r } { \mu = \frac { b } { G + c b / 3 } < \frac { 3 } { c } } \end{array}$ gives $\begin{array} { r } { - \mu b + \psi ( \mu ) G = - \frac { b ^ { 2 } } { 2 G + 2 c b / 3 } } \end{array}$ . Applying the same bound to $- X _ { i }$ and taking the union of the two events, we get (27). To prove the maximal form, note that on $\{ \exists m \leq n :$ $\begin{array} { r } { \sum _ { i \leq m } X _ { i } > b , \sum _ { i \leq m } v _ { i } \leq G \} } \end{array}$ we have max $\ L _ { m \leq n } M _ { m } > e ^ { \mu b - \psi ( \mu ) G }$ , and Ville’s maximal inequality for the non-negative supermartingale $\begin{array} { r } { ( M _ { m } ) , \operatorname { \mathbb { P } } \{ \operatorname* { m a x } _ { m \leq n } M _ { m } \geq a \} \leq \operatorname { \mathbb { E } } M _ { 0 } / a } \end{array}$ , gives the same bound $e ^ { - \mu b + \psi ( \mu ) G }$ (Freedman, 1975, Theorem 1.6), which concludes the proof. □

Lemma E.4 (Jensen for powers). For $0 ~ < ~ r ~ \leq ~ 1$ and non-negative $\begin{array} { r } { a _ { 1 } , \dotsc , a _ { n } , \dotsc \dotsc \dotsc a _ { i } ^ { r } \dotsc \dotsc } \end{array}$ $\begin{array} { r } { n ^ { 1 - r } \left( \sum _ { i = 1 } ^ { n } a _ { i } \right) ^ { r } } \end{array}$

Proof. The function $t \mapsto t ^ { r }$ is concave for $0 < r \leq 1$ , and, by Jensen’s inequality, $\begin{array} { r } { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } a _ { i } ^ { r } \leq } \end{array}$ $\textstyle { \left( { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } a _ { i } \right) } ^ { r }$ . Multiplying both sides by n, we get the claim. □

# F Upper Bounds for Clip-SGD: Full Statements and Proofs

## F.1 Full Statements

Throughout this section, $( b , v ) , c ,$ and $\Lambda _ { K }$ are defined as in Section 5. More precisely, under Assumption 3.4, since $b _ { \alpha } ^ { \mathrm { m o m } } / 3 < 1 \leq v _ { \alpha } ^ { \mathrm { m o m } }$ , the variance branch of c is active and

$$
\begin{array} { r } { c = c _ { \alpha } ^ { \mathrm { m o m } } : = 1 8 0 \left( \frac { 3 } { 2 } \right) ^ { 2 - \alpha } \in [ 1 8 0 , 2 7 0 ) , \qquad c _ { 2 } ^ { \mathrm { m o m } } = 1 8 0 ; } \end{array}\tag{28}
$$

under Assumption 3.5, $c = c _ { \alpha } ( \Lambda _ { K } )$ with $\Lambda _ { K } : = \operatorname* { m a x } \{ 8 0 ( K + 1 ) , \Lambda _ { 2 \sqrt { 2 } L R } \}$ , where, for $\lambda > 0$

$$
\Lambda _ { \lambda } : = \frac { ( K + 1 ) ( 3 \lambda / 2 ) ^ { \alpha } } { ( K + 1 ) \sigma _ { 0 } ^ { \alpha } + 2 ^ { \alpha / 2 } \sigma _ { 1 } ^ { \alpha } ( L R ) ^ { \alpha } }\tag{29}
$$

$( \Lambda _ { \lambda } : = 0 { \mathrm { ~ i f ~ } } \sigma _ { 0 } = \sigma _ { 1 } = 0 )$ ; in particular, $\Lambda _ { \lambda } = ( 3 \lambda / ( 2 \sigma _ { 0 } ) ) ^ { \alpha }$ if $\sigma _ { 1 } = 0$ and $\sigma _ { 0 } > 0$

Theorem F.1 (Clip-SGD with the tuned level: full statement of Theorem 5.1). Let Assumptions 3.1–3.3 hold for $Q = B _ { 2 R } ( x ^ { * } )$ , where $R \geq \| x ^ { 0 } - x ^ { * } \| > 0 \quad$ , and let either Assumption 3.4 or Assumption 3.5 hold with $\alpha \in ( 1 , 2 ]$ ; let c be defined as in Section 5 with $\Lambda = \Lambda _ { K }$ under Assumption 3.5. Let $\beta \in \mathsf { \Gamma } ( 0 , 1 )$ $K \in \{ 0 , 1 , \ldots \}$ , $\begin{array} { r } { \ell : = \ln \frac { 4 ( K + 1 ) } { \beta } } \end{array}$ and let the stepsize satisfy $0 < \gamma \leq \operatorname* { m i n } \{ \gamma _ { L } , \gamma _ { 0 } , \gamma _ { 1 } , \gamma _ { 2 } \}$ , where

$$
\begin{array} { l } { \gamma _ { L } : = \frac { 1 } { 8 0 \sqrt { 2 } L \ell } , } \\ { \gamma _ { 0 } : = \frac { R } { \left( c 4 0 ^ { \alpha - 1 } ( K + 1 ) \ell ^ { \alpha - 1 } \right) ^ { 1 / \alpha } \sigma _ { 0 } } , } \\ { \gamma _ { 1 } : = \frac { 1 } { \left( c 2 ^ { \alpha } 4 0 ^ { \alpha - 1 } \ell ^ { \alpha - 1 } \right) ^ { 2 / \alpha } \sigma _ { 1 } ^ { 2 } L \left( K + 1 \right) ^ { \frac { 2 - \alpha } { \alpha } } } , } \\ { \gamma _ { 2 } : = \frac { 1 } { \left( c 2 ^ { \alpha / 2 } 4 0 ^ { \alpha - 1 } \ell ^ { \alpha - 1 } \right) ^ { 1 / \alpha } \sigma _ { 1 } L } , } \end{array}
$$

with $\gamma _ { 0 } : = + \infty$ if $\sigma _ { 0 } = 0$ and $\gamma _ { 1 } = \gamma _ { 2 } : = +$ ∞ if $\sigma _ { 1 } = 0$ . Then, with probability at least $1 - \beta _ { \mathrm { { \ell } } }$ , the iterates produced by Algorithm 1 with $\textstyle \lambda _ { k } \equiv \lambda : = { \frac { R } { 4 0 \gamma \ell } }$ satisfy $x ^ { k } \in B _ { \sqrt { 2 } R } ( x ^ { * } )$ for all $k = 0 , 1 , \ldots , K + 1$ and

$$
f ( \bar { x } ^ { K } ) - f ^ { * } \leq \frac { 2 R ^ { 2 } } { \gamma ( K + 1 ) } .\tag{30}
$$

Corollary F.2 (Rate and complexity). Let the conditions of Theorem F.1 hold and $\gamma ~ =$

min $\{ \gamma _ { L } , \gamma _ { 0 } , \gamma _ { 1 } , \gamma _ { 2 } \}$ . Then, with probability at least $1 - \beta$ , we have $f ( \bar { x } ^ { K } ) - f ^ { * } \leq \operatorname* { m a x } \{ \Delta _ { L } , \Delta _ { 0 } , \Delta _ { 1 } \}$ , where

$$
\begin{array} { r l } & { \Delta _ { L } : = \frac { 1 6 0 \sqrt { 2 } L R ^ { 2 } \ell } { K + 1 } , } \\ & { \Delta _ { 0 } : = \frac { 2 \left( c 4 0 ^ { \alpha - 1 } \right) ^ { 1 / \alpha } \sigma _ { 0 } R \ell ^ { \left( \alpha - 1 \right) / \alpha } } { ( K + 1 ) ^ { \left( \alpha - 1 \right) / \alpha } } , } \\ & { \Delta _ { 1 } : = \frac { 2 \left( c 2 ^ { \alpha } 4 0 ^ { \alpha - 1 } \right) ^ { 2 / \alpha } \sigma _ { 1 } ^ { 2 } L R ^ { 2 } \ell ^ { 2 \left( \alpha - 1 \right) / \alpha } } { ( K + 1 ) ^ { 2 \left( \alpha - 1 \right) / \alpha } } . } \end{array}
$$

That is, Clip-SGD achieves $f ( \bar { x } ^ { K } ) - f ^ { * } \leq \varepsilon$ with probability at least $1 - \beta$ after

$$
K = \widetilde { \cal O } \left( \operatorname* { m a x } \left\{ \frac { L R ^ { 2 } } { \varepsilon } , \left( \frac { \sigma _ { 0 } R } { \varepsilon } \right) ^ { \frac { \alpha } { \alpha - 1 } } , \left( \frac { \sigma _ { 1 } ^ { 2 } L R ^ { 2 } } { \varepsilon } \right) ^ { \frac { \alpha } { 2 ( \alpha - 1 ) } } \right\} \right)
$$

iterations, where $\widetilde O$ hides factors depending only on α and logarithmic factors in $1 / \beta$ and the problem parameters (including $c = c _ { \alpha } ( \Lambda _ { K } )$ under Assumption 3.5).

We notice that the threshold $\gamma _ { 2 }$ from Theorem F.1 is redundant: with $\Delta _ { 2 } : = 2 R ^ { 2 } / ( ( K + 1 ) \gamma _ { 2 } )$ the proof of Corollary F.2 gives $\Delta _ { 2 } \leq \operatorname* { m a x } \{ \Delta _ { L } , \Delta _ { 1 } \} / 1 5 , \mathrm { i . e . , } \gamma _ { 2 } \geq 1 5 \operatorname* { m i n } \{ \gamma _ { L } , \gamma _ { 1 } \}$ , and, therefore, it never determines the stepsize. The logarithmic dependence on the horizon in ℓ comes from the union bound over the $K + 1$ steps in the proof: under Assumption 3.4, and under Assumption 3.5 with $\alpha < 2$ and $v = v _ { \alpha } ( \infty )$ , Theorem F.1 holds with ℓ replaced by $\ln ( 4 / \beta )$ (see Theorem I.18 in Appendix I.9, which relies on a maximal inequality, as in Nguyen et al. (2023) for the moment case), and the resulting dependence on $\beta$ is optimal (see Proposition H.3). The same holds for Theorem I.1; moreover, Theorem F.4 and Corollary F.5 admit the replacement under either assumption and for every $\alpha \in ( 1 , 2 ]$ (see item 1 of Theorem I.18), since their clipping level is prescribed and $\Lambda _ { \lambda }$ does not depend on ℓ. With the horizon-dependent constant $v _ { \alpha } ( \Lambda _ { K } )$ , in particular, at $\alpha = 2 .$ , the logarithm remains; the endpoint is reached without it via the moment route (see Corollary I.19).

Corollary F.3 (Pareto noise). Consider the problem from Appendix C.4 with $\sigma > 0$ Then Assumptions 3.1–3.3 and 3.5 hold with $\sigma _ { 0 } = \sigma$ and $\sigma _ { 1 } = 0$ , although $\mathbb { E } | \xi | ^ { \alpha } = \infty$ Therefore, by Corollary F.2, with probability at least $1 - \beta$ , we have

$$
f ( \bar { x } ^ { K } ) - f ^ { * } = O _ { \alpha } \left( \frac { L R ^ { 2 } \ell } { K } + \frac { c ^ { 1 / \alpha } \sigma R \ell ^ { ( \alpha - 1 ) / \alpha } } { K ^ { ( \alpha - 1 ) / \alpha } } \right) ,
$$

where $c = c _ { \alpha } ( \Lambda _ { K } ) \leq c _ { \alpha } ( \infty )$ for $\alpha < 2$ , and $c = O ( 1 + \ln ( K + 1 ) + \ln ( 1 + L R / \sigma ) )$ for $\alpha = 2$ In contrast, Assumption 3.4 holds for this problem only with α replaced by some $q \in ( 1 , \alpha )$ and $\begin{array} { r } { \sigma _ { 0 } = \left( \frac { \alpha } { \alpha - q } \right) ^ { 1 / q } \sigma } \end{array}$ For a fixed $q ,$ Theorem F.1 gives the slower rate $K ^ { - ( q - 1 ) / q } ;$ if $q$ approaches $\alpha$ with the horizon, one recovers the exponent $( \alpha - 1 ) / \alpha$ only up to an additional factor of the order $( \ln K ) ^ { 1 / \alpha }$ , which comes from the constant $( \alpha / ( \alpha - q ) ) ^ { 1 / q }$

For example, for $\alpha = 3 / 2$ , the weak- $L ^ { 3 / 2 }$ analysis gives the exponent $1 / 3$ with a constant independent of K, while the moment-based analysis gives only $1 / 4$ for the fixed $q = 4 / 3$ , and $1 / 3$ up to the factor (ln $K ) ^ { 2 / 3 }$ for a horizon-dependent $q .$ For $\alpha = 2$ , the noise has infinite variance, and the weak- $. L ^ { 2 }$ analysis gives the exponent $1 / 2$ of the finite-variance case up to a logarithmic factor, in the original weak- $L ^ { 2 }$ class and with explicit finite-horizon parameters. We also notice that, for $\alpha < 2$ close to 2, the logarithmic branch of $v _ { \alpha }$ improves the constant for a fixed horizon as well: $c _ { \alpha } ( \Lambda _ { K } ) \leq c _ { \alpha } ( \infty )$ for every $K _ { i }$ with a strict gain when ln $\Lambda _ { K } \ll ( 2 - \alpha ) ^ { - 1 }$ (since $\Lambda _ { K } \geq 8 0$ the logarithmic branch can be active only for $\alpha > 1 . 6 8 )$ ; since $c _ { \alpha } ( \Lambda _ { K } ) \to c _ { \alpha } ( \infty )$ as $K  \infty$ , this improves the bound for a fixed horizon but not uniformly in $K$

Theorem F.4 (Prescribed clipping level: full statement). Let Assumptions 3.1–3.3 hold for $Q = B _ { 2 R } ( x ^ { * } )$ , where $R \geq \| x ^ { 0 } - x ^ { * } \| > 0$ , and let either Assumption 3.4 or Assumption 3.5 hold with $\alpha \in ( 1 , 2 ]$ . Let $\begin{array} { r } { \beta \in ( 0 , 1 ) , K \in \{ 0 , 1 , \dots \} , \ell : = \ln \frac { 4 ( K + 1 ) } { \beta } } \end{array}$ , let $\lambda \ge 2 \sqrt { 2 } L R$ be arbitrary, and let

$$
\begin{array} { c } { { S _ { 0 } : = \sigma _ { 0 } ^ { \alpha } ( K + 1 ) , \qquad S _ { 2 } : = 2 ^ { \alpha / 2 } \sigma _ { 1 } ^ { \alpha } ( L R ) ^ { \alpha } , } } \\ { { S _ { 1 } : = 2 ^ { \alpha } \sigma _ { 1 } ^ { \alpha } L ^ { \alpha / 2 } R ^ { \alpha } ( K + 1 ) ^ { \frac { 2 - \alpha } { 2 } } . } } \end{array}
$$

Let the stepsize $\begin{array} { r } { 0 < \gamma \le \frac { R } { 4 0 \lambda \ell } } \end{array}$ satisfy the following bias and variance conditions:

$$
6 0 b \gamma \lambda ^ { 1 - \alpha } \Sigma _ { K + 1 } ( \gamma ) \leq 3 R ,\tag{31}
$$

$$
7 2 0 0 \ell v \gamma ^ { 2 } \lambda ^ { 2 - \alpha } \Sigma _ { K + 1 } ( \gamma ) \leq 3 R ^ { 2 } ,\tag{32}
$$

where $\Sigma _ { K + 1 } ( \gamma ) : = S _ { 0 } + S _ { 1 } \gamma ^ { - \alpha / 2 } + S _ { 2 }$ . Then, with probability at least $1 - \beta$ , the iterates produced by Algorithm 1 with $\lambda _ { k } \equiv \lambda$ satisfy $x ^ { k } \in B _ { \sqrt { 2 } R } ( x ^ { * } )$ for all $k = 0 , 1 , \ldots , K + 1$ and

$$
f ( \bar { x } ^ { K } ) - f ^ { * } \leq \frac { 2 R ^ { 2 } } { \gamma ( K + 1 ) } .
$$

Both left-hand sides are non-decreasing in $\gamma$ . Requiring each of the three terms of $\Sigma _ { K + 1 } ( \gamma )$ to contribute at most one third, we get the explicit thresholds $\gamma \leq \operatorname* { m i n } \{ \gamma _ { b } ^ { \lambda } , \gamma _ { v } ^ { \lambda } \}$ from (53) below. For $\alpha = 2$ , the $S _ { \mathrm { 1 ^ { - } p a r t } }$ of (31) does not depend on $\gamma$ and requires $\lambda \ge 2 4 0 b \sigma _ { 1 } ^ { 2 } L R$

Corollary F.5 (Neighborhood and horizon-dependent levels: full statement of Theorem 5.3). Let the conditions of Theorem F.4 hold, let either $\alpha < 2$ , or $\alpha = 2$ and $\lambda \ge 2 4 0 b \sigma _ { 1 } ^ { 2 } L R$ (in the latter case, the $S _ { 1 } { \mathrm { - t e r m } }$ of $\gamma _ { b } ^ { \lambda }$ and the second term of $\mathcal { N } _ { \lambda }$ are absent), and let $\begin{array} { r } { \gamma : = \operatorname* { m i n } \{ \frac { R } { 4 0 \lambda \ell } , \gamma _ { b } ^ { \lambda } , \gamma _ { v } ^ { \lambda } \} } \end{array}$ with the thresholds from (53). Then, with probability at least $1 - \beta .$ , we have

$$
f ( \bar { x } ^ { K } ) - f ^ { * } \le \operatorname* { m a x } \left\{ \frac { 8 0 \lambda R \ell } { K + 1 } , \mathcal { N } _ { \lambda } , \mathcal { V } _ { \lambda , K } \right\} ,\tag{33}
$$

where the neighborhood

$$
\mathcal { N } _ { \lambda } : = \operatorname* { m a x } \left\{ \frac { 1 2 0 b R \sigma _ { 0 } ^ { \alpha } } { \lambda ^ { \alpha - 1 } } , 2 \left( \frac { 6 0 \cdot 2 ^ { \alpha } b R L ^ { \frac { \alpha } { 2 } } \sigma _ { 1 } ^ { \alpha } } { \lambda ^ { \alpha - 1 } } \right) ^ { \frac { 2 } { 2 - \alpha } } \right\}\tag{34}
$$

does not depend on $K .$ , while

$$
\begin{array} { c } { { \displaystyle \mathcal { V } _ { \lambda , K } : = \operatorname * { m a x } \{ 1 2 0 \sqrt { 2 } R \sqrt { \frac { v \lambda ^ { 2 - \alpha } \sigma _ { 0 } ^ { \alpha \ell } } { K + 1 } } , \frac { 1 2 0 \sqrt { 2 } b \sigma _ { 1 } ^ { \alpha } L R ^ { 2 } } { K + 1 } , \frac { 1 2 0 \sqrt { 2 } R \sqrt { v \lambda ^ { 2 - \alpha } S _ { 2 } \ell } } { K + 1 } , 2 0 \sqrt { \frac { v \lambda ^ { 2 } } { K } } \} \sigma _ { 0 } ^ { \lambda , K } \sigma _ { 0 } ^ { \lambda , K } } } \\ { { \displaystyle \frac { 2 R ^ { 2 } } { K + 1 } ( \frac { 7 2 0 0 \ell v \lambda ^ { 2 - \alpha } S _ { 1 } } { R ^ { 2 } } ) ^ { \frac { 2 } { 4 - \alpha } } \} } } \end{array}\tag{35}
$$

vanishes as $K  \infty$ (its terms are of the orders $K ^ { - 1 / 2 } , K ^ { - 1 } , K ^ { - 1 }$ , and $K ^ { - 2 / ( 4 - \alpha ) }$ , up to logarithmic factors). Moreover, if $\sigma _ { 1 } = 0$ and $\lambda = \lambda _ { 0 } ( K + 1 ) ^ { 1 / \alpha }$ with $\lambda _ { 0 } > 0$ and $K + 1 \ge ( 2 \sqrt { 2 } L R / \lambda _ { 0 } ) ^ { \alpha }$ then, with probability at least $1 - \beta$ , we have

$$
f ( \bar { x } ^ { K } ) - f ^ { * } \leq \frac { \Xi R } { ( K + 1 ) ^ { ( \alpha - 1 ) / \alpha } } ,\tag{36}
$$

where $\Xi : = 8 0 \lambda _ { 0 } \ell + 1 2 0 b \sigma _ { 0 } ^ { \alpha } \lambda _ { 0 } ^ { 1 - \alpha } + 1 2 0 \sqrt { 2 v \ell } \sigma _ { 0 } ^ { \alpha / 2 } \lambda _ { 0 } ^ { 1 - \alpha / 2 }$ . For $\sigma _ { 1 } > 0$ , the same choice of $\lambda$ adds the terms decaying faster than $( K + 1 ) ^ { - ( \alpha - 1 ) / \tilde { \alpha } }$

Discussion. If we write $\lambda _ { 0 } = \rho \sigma _ { 0 }$ in (36), then the price of misspecifying the noise level is the factor $8 0 \rho \ell + 1 2 0 b \rho ^ { 1 - \alpha } + 1 2 0 \sqrt { 2 v \ell } \rho ^ { 1 - \alpha / 2 }$ in front of $\sigma _ { 0 } R ( K + 1 ) ^ { - ( \alpha - 1 ) / \alpha }$ , which grows linearly in the case of overestimation $( \rho \to \infty )$ and as $\rho ^ { - ( \alpha - 1 ) }$ in the case of underestimation $( \rho \to 0 )$ . We emphasize that Theorems F.1 and F.4 both follow from one general bound (Proposition F.6 below), where $\gamma$ and $\lambda$ are coupled only through the bias condition, involving $b ,$ and the variance condition, involving $v ;$ under Assumption 3.4, Theorem F.1 is the special case $\lambda = R / ( 4 0 \gamma \ell )$ of Theorem $\mathrm { F . 4 . }$ while under Assumption 3.5 the variance constants $v _ { \alpha } ( \Lambda _ { K } )$ and $v _ { \alpha } ( \Lambda _ { \lambda } )$ are not comparable in general, meaning that neither of the theorems contains the other one. For $\sigma _ { 1 } = 0$ and the moment condition, the bound of Khah et al. (2026), for $\lambda > 4 L R$ and the best iterate, reduces to the same three terms as in Corollary F.5 after applying the initial-point bound $f ( x ^ { 0 } ) - f ^ { * } \leq L R ^ { 2 } / 2$ (see the discussion of Table 3): $\mathrm { ~ a ~ } \lambda R \ell / K$ term, a neighborhood of the order $R \sigma _ { 0 } ^ { \alpha } / \lambda ^ { \alpha - 1 }$ , and a cross term of the order $R \lambda ^ { 1 - \alpha / 2 } \sigma _ { 0 } ^ { \alpha / 2 } \sqrt { \ell / K }$ ; Corollary F.5 gives them for the averaged iterate, under afine weak- $. L ^ { \alpha }$ noise and with explicit constants, and, at $\alpha = 2$ under Assumption 3.5, only the cross term has the logarithm of $v _ { 2 }$

## F.2 A General Bound and the Proofs

Throughout this section, we use the notation $\begin{array} { r } { \ell : = \ln \frac { 4 ( K + 1 ) } { \beta } , R _ { k } : = \| x ^ { k } - x ^ { * } \| } \end{array}$ , and $\delta _ { k } : = f ( x ^ { k } ) - f ^ { * } \geq$ 0, and the decomposition (25). Note that $\ell > \ln { 4 } > ^ { \prime } 1$ since $4 ( K + 1 ) / \beta > 4$ . For $\gamma , \lambda > 0$ and $T \geq 1$ , we define

$$
\Sigma _ { T } : = \sigma _ { 0 } ^ { \alpha } T + \sigma _ { 1 } ^ { \alpha } 2 ^ { \alpha } L ^ { \alpha / 2 } R ^ { \alpha } \gamma ^ { - \alpha / 2 } T ^ { \frac { 2 - \alpha } { 2 } } + \sigma _ { 1 } ^ { \alpha } 2 ^ { \alpha / 2 } L ^ { \alpha } R ^ { \alpha } , \qquad \Phi _ { T } : = \gamma \lambda ^ { 1 - \alpha } \Sigma _ { T } .\tag{37}
$$

Both quantities are non-decreasing in $T ,$ while $\Sigma _ { T } / T$ is non-increasing in $T .$ . Next, let $( b , v ) : =$ $( b _ { \alpha } ^ { \mathrm { m o m } } , v _ { \alpha } ^ { \mathrm { m o m } } )$ under Assumption 3.4 and $( b , v ) : = ( b _ { \alpha } , v _ { \alpha } ( \Lambda _ { \gamma , \lambda } ) )$ under Assumption 3.5, where

$$
\Lambda _ { \gamma , \lambda } : = \frac { ( K + 1 ) ( 3 \lambda / 2 ) ^ { \alpha } } { \Sigma _ { K + 1 } } \qquad ( \Lambda _ { \gamma , \lambda } : = 0 \mathrm { i f } \sigma _ { 0 } = \sigma _ { 1 } = 0 ) ,\tag{38}
$$

and let $c : = 1 8 0 \operatorname* { m a x } \{ v , b / 3 \}$ , meaning that

$$
c \geq 1 8 0 v \qquad \mathrm { a n d } \qquad c \geq 6 0 b .\tag{39}
$$

Since $\Sigma _ { K + 1 } \geq ( K + 1 ) \sigma _ { 0 } ^ { \alpha } + 2 ^ { \alpha / 2 } \sigma _ { 1 } ^ { \alpha } ( L R ) ^ { \alpha }$ , we have $\Lambda _ { \gamma , \lambda } \leq \Lambda _ { \lambda }$ , where $\Lambda _ { \lambda }$ is defined in (29); hence, under Assumption 3.5, this c is at most $c _ { \alpha } ( \Lambda _ { \lambda } )$

Proposition F.6 (A general high-probability bound). Let Assumptions 3.1–3.3 hold for $Q =$ $B _ { 2 R } ( x ^ { * } )$ , where $R \geq \| x ^ { 0 } - x ^ { * } \| > 0$ , and let either Assumption 3.4 or Assumption 3.5 hold with $\alpha \in ( 1 , 2 ]$ . Let $\beta \in ( 0 , 1 ) , K \in \{ 0 , 1 , . . . \}$ , and let $\gamma , \lambda > 0$ satisfy, with $( b , v )$ defined above,

$$
\lambda \geq 2 \sqrt { 2 } L R , \qquad \gamma \lambda \leq \frac { R } { 4 0 \ell } , \qquad b \Phi _ { K + 1 } \leq \frac { R } { 2 0 } , \qquad v \gamma ^ { 2 } \lambda ^ { 2 - \alpha } \Sigma _ { K + 1 } \leq \frac { R ^ { 2 } } { 2 4 0 0 \ell } .\tag{40}
$$

Then, with probability at least $1 - \beta _ { i }$ , the iterates produced by Algorithm 1 with $\lambda _ { k } \equiv \lambda$ satisfy $x ^ { k } \in B _ { \sqrt { 2 } R } ( x ^ { * } )$ for all $k = 0 , 1 , \ldots , K + 1$ and $\begin{array} { r } { f ( \bar { x } ^ { \bar { K } } ) - f ^ { * } \leq \frac { 2 R ^ { 2 } } { \gamma ( K + 1 ) } } \end{array}$

The third condition in (40) controls the accumulated clipping bias, and the fourth one controls the accumulated variance. Both of them follow from the single condition $\Phi _ { K + 1 } \leq 3 R / c \mathrm { : }$ by (39), $b \Phi _ { K + 1 } \leq 3 b R / c \leq R / 2 0$ , and $\begin{array} { r } { v \gamma ^ { 2 } \lambda ^ { 2 - \alpha } \Sigma _ { K + 1 } = v \gamma \lambda \Phi _ { K + 1 } \leq \frac { R } { 4 0 \ell } \cdot \frac { 3 v R } { c } \leq \frac { R ^ { 2 } } { 2 4 0 0 \ell } } \end{array}$ . This is exactly how we derive Theorem F.1 below. For a prescribed $\lambda ,$ keeping the two conditions separate is what makes the neighborhood in Theorem F.4 independent of the variance constant v (and of its logarithm at $\alpha = 2 )$ .

Proof of Proposition F.6. Our proof is induction-based: by induction, we show that the iterates do not leave the ball $B _ { \sqrt { 2 } R } ( x ^ { * } )$ with high probability. We start with the consequences of (40). By the first two conditions and $\ell > 1$ , we have

$$
\gamma \leq \frac { R } { 4 0 \ell \lambda } \leq \frac { 1 } { 8 0 \sqrt { 2 } L } < \frac { 1 } { 2 L } , \qquad L \gamma \leq \sqrt { 2 } - 1 , \qquad \gamma \lambda \leq \frac { R } { 4 0 \ell } \leq \frac { R } { 4 0 } , \qquad \sqrt { 2 } L R \leq \frac { \lambda } { 2 } .\tag{41}
$$

Events and induction hypothesis. To formulate the statement rigorously, we introduce probability event $E _ { k }$ for each $k = 0 , 1 , \ldots , K + 1$ as follows: $E _ { 0 } : = \Omega$ and, for $k = 1 , \ldots , K + 1$

$$
E _ { k } : = \left\{ \forall t = 1 , \ldots , k : \ - 2 \gamma \sum _ { l = 0 } ^ { t - 1 } \left. x ^ { l } - x ^ { * } - \gamma \nabla f ( x ^ { l } ) , \theta _ { l } \right. + \gamma ^ { 2 } \sum _ { l = 0 } ^ { t - 1 } \lVert \theta _ { l } \rVert ^ { 2 } \leq R ^ { 2 } \ \mathrm { ~ a n d ~ } \ R _ { t } \leq \sqrt { 2 } R \right\} .\tag{42}
$$

We will prove by induction that $\begin{array} { r } { \mathbb { P } ( E _ { k } ) \ge 1 - \frac { k \beta } { K + 1 } } \end{array}$ for all $k = 0 , 1 , \ldots , K + 1$ . The base of the induction $( k = 0 )$ is trivial. Next, we assume that the statement holds for $k = T - 1$ with some $\begin{array} { r } { T \in \{ 1 , \dotsc , K + 1 \} , \mathrm { i . e . , } \mathbb { P } ( E _ { T - 1 } ) \geq 1 - \frac { ( T - 1 ) \beta } { K + 1 } } \end{array}$ . Given this, we need to prove that $\begin{array} { r } { \mathbb { P } ( E _ { T } ) \geq 1 - \frac { T \beta } { K + 1 } . } \end{array}$ Throughout the proof, empty sums are equal to zero.

Localization. First, we show that $E _ { T - 1 }$ implies that the iterates $x ^ { 0 } , \ldots , x ^ { T }$ belong to Q. Indeed, on $E _ { T - 1 }$ , we have $R _ { t } \leq \sqrt { 2 } R$ for $t = 1 , \dots , T - 1$ and $R _ { 0 } \leq R _ { ☉ }$ , and, therefore, $x ^ { t } \in Q$ for $t \leq T - 1$ Moreover, by (41), $R _ { T } \leq R _ { T - 1 } + \gamma \Vert \widetilde { g } ^ { T - 1 } \Vert \leq \sqrt { 2 } R + \gamma \lambda \leq 2 R$ , i.e., $x ^ { T } \in Q$ . Since $x ^ { * }$ is an interior minimizer of the diferentiable convex function f on $Q _ { i }$ we have $\nabla f ( x ^ { * } ) = 0$ . Thus, $E _ { T - 1 }$ implies that, for $t = 0 , \ldots , T - 1$ ，

$$
\| \nabla f ( x ^ { t } ) \| \leq L R _ { t } \leq \sqrt { 2 } L R \leq \frac { \lambda } { 2 } , \qquad \| x ^ { t } - x ^ { * } - \gamma \nabla f ( x ^ { t } ) \| \leq \sqrt { 2 } R ( 1 + L \gamma ) \leq 2 R .\tag{43}
$$

Before we proceed, we introduce a new notation: for $t = 0 , \ldots , T - 1$ , we define the $\mathcal { G } _ { t } { \mathrm { - m e a s u r a b l e } }$ vectors

$$
\begin{array} { r } { \eta _ { t } : = \left\{ \begin{array} { l l } { x ^ { t } - x ^ { * } - \gamma \nabla f ( x ^ { t } ) , } & { \mathrm { i f ~ } \| x ^ { t } - x ^ { * } - \gamma \nabla f ( x ^ { t } ) \| \leq 2 R , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}
$$

By construction, $\| \eta _ { t } \| \le 2 R$ almost surely, and $E _ { T - 1 }$ implies $\eta _ { t } = x ^ { t } - x ^ { * } - \gamma \nabla f ( x ^ { t } )$ for $t \leq T - 1$

Basic inequality. By (41), Lemma E.2 can be applied on $E _ { T - 1 }$ with $t = T$ . Since $f ( \bar { x } ^ { T - 1 } ) \geq f ^ { * }$ it gives

$$
R _ { T } ^ { 2 } \leq R ^ { 2 } - 2 \gamma \sum _ { t = 0 } ^ { T - 1 } \left. x ^ { t } - x ^ { * } - \gamma \nabla f ( x ^ { t } ) , \theta _ { t } \right. + \gamma ^ { 2 } \sum _ { t = 0 } ^ { T - 1 } \| \theta _ { t } \| ^ { 2 } .\tag{44}
$$

Clipping bounds. By construction, we have $\| \theta _ { t } ^ { u } \| \ \leq \ 2 \lambda$ almost surely. Next, on $E _ { T - 1 }$ , for $t \leq T - 1$ , we have $\| \nabla f ( x ^ { t } ) \| \leq \lambda / 2$ by (43). Formally, let $\mathcal { C } _ { t } : = \{ x ^ { t } \in Q , \ \| \nabla f ( x ^ { t } ) \| \le \lambda / 2 \} \in \mathcal { G } _ { t } ;$ then $E _ { T - 1 } \subseteq { \mathcal { C } } _ { t }$ , and let $\check { x } ^ { t } : = x ^ { t } \mathbf { 1 } _ { \mathcal { C } _ { t } } + x ^ { * } \mathbf { 1 } _ { \mathcal { C } _ { t } ^ { c } } \in Q$ . By Lemma I.3, Corollary D.5 can be applied with $\mathcal { G } = \mathcal { G } _ { t } , x = \nabla f ( \check { x } ^ { t } )$ , and $X = g ( \check { x } ^ { t } , \xi ^ { t } )$ ; here we use Lemma D.2 under Assumption 3.4 and Lemma D.1 under Assumption 3.5. The clipped vector coincides with $\widetilde { g } ^ { t }$ on $\mathcal { C } _ { t } ;$ therefore, on $\mathcal { C } _ { t } .$ the conditional bias and second moment of $\widetilde { g } ^ { t }$ satisfy the same bounds (see Step 2 of Appendix I.5 for the details). Here we always condition on $\mathcal { G } _ { t }$ and never on $E _ { T - 1 }$ . Let $A _ { t } : = \sigma _ { 0 } ^ { \alpha } + \sigma _ { 1 } ^ { \alpha } ( 2 L \delta _ { t } ) ^ { \alpha / 2 }$ and let $\phi _ { \lambda } ( s ) : = v _ { \alpha } ^ { \mathrm { m o m } } \varepsilon$ under Assumption 3.4 and $\phi _ { \lambda } ( s ) : = s v _ { \alpha } ( ( 3 \lambda / 2 ) ^ { \alpha } / s )$ under Assumption 3.5 (with $\phi _ { \lambda } ( 0 ) : = 0 )$ . By Assumption 3.2, we have $\| \nabla f ( x ^ { t } ) \| ^ { \alpha } \leq ( 2 L \delta _ { t } ) ^ { \alpha / 2 }$ , meaning that the noise scale at $x ^ { t }$ is at most $A _ { t }$ . Since the right-hand sides of the clipping lemmas are non-decreasing in $A ( x )$ (see Lemma D.6 for $\phi _ { \lambda } )$ , we get

$$
\| \theta _ { t } ^ { b } \| \leq \frac { b A _ { t } } { \lambda ^ { \alpha - 1 } } , \qquad \mathbb { E } \left[ \| \theta _ { t } ^ { u } \| ^ { 2 } \ | \ \mathcal { G } _ { t } \right] \leq \lambda ^ { 2 - \alpha } \phi _ { \lambda } ( A _ { t } ) .\tag{45}
$$

Upper bound for $\textstyle \sum _ { t } \delta _ { t } ^ { \alpha / 2 }$ . On $E _ { T - 1 }$ , L-smoothness on Q and $\begin{array} { r } { \nabla f ( x ^ { * } ) = 0 \mathrm { g i v e } \delta _ { T - 1 } \leq \frac { L } { 2 } R _ { T - 1 } ^ { 2 } \leq } \end{array}$ $L R ^ { 2 } . { \mathrm { ~ H ~ } } T = 1$ , the sum over $t = 0 , \dots , T - 2$ is empty. If $T \geq 2 .$ , Lemma E.2 with $t = T - 1$ and the first condition in (42) give $\begin{array} { r } { \gamma \sum _ { t = 0 } ^ { T - 2 } \delta _ { t } \leq R _ { 0 } ^ { 2 } + \bar { R ^ { 2 } } \leq 2 R ^ { 2 } } \end{array}$ . Applying Lemma E.4 with $r = \alpha / 2$ and $n = T - 1$ , we obtain

$$
\sum _ { t = 0 } ^ { T - 2 } \delta _ { t } ^ { \alpha / 2 } \leq ( T - 1 ) ^ { \frac { 2 - \alpha } { 2 } } \left( \sum _ { t = 0 } ^ { T - 2 } \delta _ { t } \right) ^ { \alpha / 2 } \leq T ^ { \frac { 2 - \alpha } { 2 } } \left( \frac { 2 R ^ { 2 } } { \gamma } \right) ^ { \alpha / 2 } .
$$

The final upper bound also holds for $T = 1$ , since the empty sum is zero. Therefore, on $E _ { T - 1 }$ , with $\Sigma _ { T }$ defined in (37), we have

$$
\sum _ { t = 0 } ^ { T - 1 } A _ { t } = \sum _ { t = 0 } ^ { T - 1 } \left( \sigma _ { 0 } ^ { \alpha } + \sigma _ { 1 } ^ { \alpha } ( 2 L \delta _ { t } ) ^ { \alpha / 2 } \right) \leq \Sigma _ { T } .\tag{46}
$$

Moreover, $E _ { T - 1 }$ implies

$$
\sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left[ \Vert \theta _ { t } ^ { u } \Vert ^ { 2 } \mid \mathcal { G } _ { t } \right] \leq v \lambda ^ { 2 - \alpha } \Sigma _ { T } .\tag{47}
$$

Indeed, under Assumption 3.4, this follows from (45) and (46). Under Assumption 3.5, if $\Sigma _ { T } = 0$ 2 then all $A _ { t } ~ = ~ 0$ and both sides vanish. Otherwise, Lemma D.6 with $C = ( 3 \lambda / 2 ) ^ { \alpha }$ and $\bar { B } \ =$ $\Sigma _ { T }$ , the inequality $T / \Sigma _ { T } \ \leq \ ( K + 1 ) / \Sigma _ { K + 1 }$ , and the monotonicity of v<sub>α</sub> give $\begin{array} { r } { \sum _ { t < T } \phi _ { \lambda } ( A _ { t } ) } \end{array} \leq$ $\Sigma _ { T } v _ { \alpha } ( T C / \Sigma _ { T } ) \le \Sigma _ { T } v _ { \alpha } ( \Lambda _ { \gamma , \lambda } ) = v \Sigma _ { T }$

The key inequalities. Since $\Sigma _ { T }$ and $\Phi _ { T }$ are non-decreasing in $T$ and $T \leq K + 1$ , the last two conditions in (40) imply

$$
b \Phi _ { T } = b \gamma \lambda ^ { 1 - \alpha } \Sigma _ { T } \leq \frac { R } { 2 0 }\tag{48}
$$

and

$$
v \gamma ^ { 2 } \lambda ^ { 2 - \alpha } \Sigma _ { T } \leq \frac { R ^ { 2 } } { 2 4 0 0 \ell } .\tag{49}
$$

Decomposition into five terms. Using $\theta _ { t } = \theta _ { t } ^ { u } + \theta _ { t } ^ { b } , \ \lVert \theta _ { t } \rVert ^ { 2 } \leq 2 \lVert \theta _ { t } ^ { u } \rVert ^ { 2 } + 2 \lVert \theta _ { t } ^ { b } \rVert ^ { 2 }$ , and $\eta _ { t } ~ =$ $x ^ { t } - x ^ { * } - \gamma \nabla f ( x ^ { t } )$ on $E _ { T - 1 }$ , we derive from (44) that $E _ { T - 1 }$ implies

$$
- 2 \gamma \sum _ { t = 0 } ^ { T - 1 } \langle x ^ { t } - x ^ { * } - \gamma \nabla f ( x ^ { t } ) , \theta _ { t } \rangle + \gamma ^ { 2 } \sum _ { t = 0 } ^ { T - 1 } \| \theta _ { t } \| ^ { 2 } \leq \Phi + \mathcal { D } + \mathcal { D } + \Phi , \quad \quad R _ { T } ^ { 2 } \leq R ^ { 2 } + \Phi + \mathcal { D } + \mathcal { D } + \Phi ,\tag{50}
$$

where

$$
\begin{array} { r l } { \bar { \mathbb { O } } : = - 2 \gamma \displaystyle \sum _ { t = 0 } ^ { T - 1 } \left. \theta _ { t } ^ { u } , \eta _ { t } \right. , \quad \quad \bar { \mathbb { O } } : = - 2 \gamma \displaystyle \sum _ { t = 0 } ^ { T - 1 } \left. \theta _ { t } ^ { b } , \eta _ { t } \right. , \quad } & { \bar { \mathbb { O } } : = 2 \gamma ^ { 2 } \displaystyle \sum _ { t = 0 } ^ { T - 1 } \left( \| \theta _ { t } ^ { u } \| ^ { 2 } - \mathbb { E } \left[ \| \theta _ { t } ^ { u } \| ^ { 2 } \mid \mathcal { G } _ { t } \right] \right) , } \\ { \bar { \mathbb { Q } } : = 2 \gamma ^ { 2 } \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left[ \| \theta _ { t } ^ { u } \| ^ { 2 } \mid \mathcal { G } _ { t } \right] , \quad } & { \bar { \mathbb { O } } : = 2 \gamma ^ { 2 } \displaystyle \sum _ { t = 0 } ^ { T - 1 } \| \theta _ { t } ^ { b } \| ^ { 2 } . } \end{array}
$$

That ${ \mathrm { i s } } ,$ it remains to derive the upper bounds for ${ \textcircled { 1 } } , { \textcircled { 2 } } , { \textcircled { 3 } } , { \textcircled { 4 } } , { \textcircled { 5 } }$ . More precisely, we show that each of these terms is at most $R ^ { 2 } / 5$ in absolute value on $E _ { T - 1 }$ intersected with two events of probability at least $\begin{array} { r } { 1 - \frac { \beta } { 2 ( K + 1 ) } } \end{array}$ each.

Upper bound for ②. On $E _ { T - 1 }$ , by (45), (46), and (48), we have

$$
| \mathfrak { Q } | \le 2 \gamma \sum _ { t = 0 } ^ { T - 1 } \| \theta _ { t } ^ { b } \| \| \eta _ { t } \| \le 4 \gamma R \sum _ { t = 0 } ^ { T - 1 } \| \theta _ { t } ^ { b } \| \le 4 b R \gamma \lambda ^ { 1 - \alpha } \Sigma _ { T } = 4 R b \Phi _ { T } \le \frac { R ^ { 2 } } { 5 } .
$$

Upper bound for ④. On $E _ { T - 1 }$ , by (47) and (49), we have

$$
\textcircled { 4 } \leq 2 v \gamma ^ { 2 } \lambda ^ { 2 - \alpha } \Sigma _ { T } \leq \frac { R ^ { 2 } } { 1 2 0 0 \ell } \leq \frac { R ^ { 2 } } { 5 } .
$$

Upper bound for $\textcircled{5}$ . On $E _ { T - 1 }$ , since the summands are non-negative, by (45) and (48), we have

$$
\mathfrak { G } \le 2 \left( \gamma \sum _ { t = 0 } ^ { T - 1 } \lVert \theta _ { t } ^ { b } \rVert \right) ^ { 2 } \le 2 \left( b \Phi _ { T } \right) ^ { 2 } \le \frac { R ^ { 2 } } { 2 0 0 } \le \frac { R ^ { 2 } } { 5 } .
$$

Upper bound for ①. To estimate this sum, we will use Bernstein’s inequality in the martingale form (Lemma E.3). Let $X _ { t } : = - 2 \gamma \left. \theta _ { t } ^ { u } , \eta _ { t } \right.$ . The summands have conditional expectations equal to zero: since $\eta _ { t }$ is $\mathcal { G } _ { t }$ -measurable and $\mathbb { E } [ \theta _ { t } ^ { u } \mid \mathcal { G } _ { t } ] = 0$ , the sequence $\{ X _ { t } \} _ { t = 0 } ^ { T - 1 }$ is a martingale diference sequence w.r.t. $\{ \mathcal { G } _ { t + 1 } \} _ { t = 0 } ^ { T - 1 }$ . Next, the summands are bounded:

$$
| X _ { t } | \leq 2 \gamma \| \theta _ { t } ^ { u } \| \| \eta _ { t } \| \leq 8 \gamma \lambda R \leq \frac { R ^ { 2 } } { 5 \ell } = : c _ { 1 } .
$$

Finally, conditional variances $\sigma _ { t } ^ { 2 } : = \mathbb { E } [ X _ { t } ^ { 2 } \mid { \mathcal { G } } _ { t } ] \leq 4 \gamma ^ { 2 } \| \eta _ { t } \| ^ { 2 } \mathbb { E } [ \| \theta _ { t } ^ { u } \| ^ { 2 } \mid { \mathcal { G } } _ { t } ] \leq 1 6 \gamma ^ { 2 } R ^ { 2 } \mathbb { E } [ \| \theta _ { t } ^ { u } \| ^ { 2 } \mid { \mathcal { G } } _ { t } ]$ are bounded on $E _ { T - 1 } .$ by (47) and (49),

$$
\sum _ { t = 0 } ^ { T - 1 } \sigma _ { t } ^ { 2 } \le 1 6 R ^ { 2 } v \gamma ^ { 2 } \lambda ^ { 2 - \alpha } \Sigma _ { T } \le \frac { 1 6 R ^ { 4 } } { 2 4 0 0 \ell } = \frac { R ^ { 4 } } { 1 5 0 \ell } = : G .\tag{51}
$$

Applying Lemma E.3 with $b _ { 1 } = R ^ { 2 } / 5 , c _ { 1 }$ , and $G _ { i }$ , we get

$$
\mathbb { P } \left\{ | \Phi | > \frac { R ^ { 2 } } { 5 } \mathrm { ~ a n d ~ } \sum _ { t = 0 } ^ { T - 1 } \sigma _ { t } ^ { 2 } \le G \right\} \le 2 \exp \left( - \frac { b _ { 1 } ^ { 2 } } { 2 G + 2 c _ { 1 } b _ { 1 } / 3 } \right) = 2 \exp ( - \ell ) = \frac { \beta } { 2 ( K + 1 ) } .
$$

The above is equivalent to $\begin{array} { r } { \mathbb { P } ( E _ { \mathfrak { D } } ) \geq 1 - \frac { \beta } { 2 ( K + 1 ) } } \end{array}$ for the event

$$
E _ { \mathbb { O } } : = \left\{ \mathrm { e i t h e r } \sum _ { t = 0 } ^ { T - 1 } \sigma _ { t } ^ { 2 } > G \mathrm { o r } | \mathbb { O } | \leq \frac { R ^ { 2 } } { 5 } \right\} .
$$

Moreover, by (51), $E _ { T - 1 } \cap E _ { \mathbb { 0 } }$ implies $| \mathbb { O } | \le R ^ { 2 } / 5$

Upper bound for ③. Let $Y _ { t } : = 2 \gamma ^ { 2 } \left( \Vert \theta _ { t } ^ { u } \Vert ^ { 2 } - \mathbb { E } [ \Vert \theta _ { t } ^ { u } \Vert ^ { 2 } \ | \mathcal { G } _ { t } ] \right)$ . Then $\{ Y _ { t } \} _ { t = 0 } ^ { T - 1 }$ is a martingale diference sequence w.r.t. $\{ \mathcal { G } _ { t + 1 } \} _ { t = 0 } ^ { T - 1 }$ , and, since $\| \theta _ { t } ^ { u } \| ^ { 2 } \leq 4 \lambda ^ { 2 }$ , the summands are bounded:

$$
| Y _ { t } | \leq 2 \gamma ^ { 2 } \left( \Vert \theta _ { t } ^ { u } \Vert ^ { 2 } + \mathbb { E } [ \Vert \theta _ { t } ^ { u } \Vert ^ { 2 } \ | \ \mathcal { G } _ { t } ] \right) \leq 1 6 \gamma ^ { 2 } \lambda ^ { 2 } \leq \frac { R ^ { 2 } } { 1 0 0 \ell ^ { 2 } } \leq \frac { R ^ { 2 } } { 5 \ell } = c _ { 1 } .
$$

Next, for a random variable $0 \le Z \le B$ , we have $\begin{array} { r } { \mathbb { E } ( Z - \mathbb { E } Z ) ^ { 2 } \leq \mathbb { E } Z ^ { 2 } \leq B \mathbb { E } Z } \end{array}$ . With $Z = \| \theta _ { t } ^ { u } \| ^ { 2 }$ and $B = 4 \lambda ^ { 2 }$ , the conditional variances satisfy $\widetilde { \sigma } _ { t } ^ { 2 } : = \mathbb { E } [ Y _ { t } ^ { 2 } \mid \mathcal { G } _ { t } ] \le 1 6 \gamma ^ { 4 } \lambda ^ { 2 } \mathbb { E } [ \| \theta _ { t } ^ { u } \| ^ { 2 } \mid \mathcal { G } _ { t } ]$ . On $E _ { T - 1 }$ by (51), we have

$$
\sum _ { t = 0 } ^ { T - 1 } \widetilde { \sigma } _ { t } ^ { 2 } \leq 1 6 \gamma ^ { 2 } \lambda ^ { 2 } \cdot \gamma ^ { 2 } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } [ \Vert \theta _ { t } ^ { u } \Vert ^ { 2 } \ \vert \ \mathcal { G } _ { t } ] \leq \frac { R ^ { 2 } } { 1 0 0 \ell ^ { 2 } } \cdot \frac { 1 } { 1 6 R ^ { 2 } } \cdot \frac { R ^ { 4 } } { 1 5 0 \ell } \leq G .
$$

Similarly to the case of $\textcircled{1} ,$ , Lemma E.3 implies that the event

$$
E _ { \odot } : = \left\{ \mathrm { e i t h e r } \sum _ { t = 0 } ^ { T - 1 } \widetilde { \sigma } _ { t } ^ { 2 } > G \mathrm { o r } | \mathfrak { Q } | \le \frac { R ^ { 2 } } { 5 } \right\}
$$

satisfies $\begin{array} { r } { \mathbb { P } ( E _ { \mathfrak { G } } ) \geq 1 - \frac { \beta } { 2 ( K + 1 ) } } \end{array}$ , and $E _ { T - 1 } \cap E _ { \mathfrak { 3 } }$ implies $\lvert \textcircled { 3 } \rvert \leq R ^ { 2 } / 5$

Completing the induction. That is, we derived the upper bounds for $\textcircled{1} - \textcircled{5}$ . More precisely, $E _ { T - 1 } \cap E _ { \mathbb { 0 } } \cap E _ { \mathbb { 3 } }$ implies that all five bounds hold. By (50), the left-hand side of the first condition in (42) with $t = T$ is at most $R ^ { 2 }$ , and $R _ { T } ^ { 2 } \le R ^ { 2 } + 5 \cdot \frac { R ^ { 2 } } { 5 } = 2 R ^ { 2 }$ . Together with the conditions for $t \leq T - 1$ contained in $E _ { T - 1 }$ , this means that $E _ { T - 1 } \cap E _ { \mathbb { 0 } } \cap E _ { \mathbb { 3 } } \subseteq E _ { T }$ , and the union bound gives

$$
\mathbb { P } ( E _ { T } ) \geq 1 - \mathbb { P } ( E _ { T - 1 } ^ { c } ) - \mathbb { P } ( E _ { \mathfrak { D } } ^ { c } ) - \mathbb { P } ( E _ { \mathfrak { D } } ^ { c } ) \geq 1 - \frac { ( T - 1 ) \beta } { K + 1 } - \frac { \beta } { K + 1 } = 1 - \frac { T \beta } { K + 1 } .
$$

This finishes the induction. In particular, $\mathbb { P } ( E _ { K + 1 } ) \ge 1 - \beta .$

Conclusion of the proof of Proposition F.6. Finally, $E _ { K + 1 }$ implies that all iterates $x ^ { 0 } , \ldots , x ^ { K + 1 }$ belong to $B _ { \sqrt { 2 } R } ( x ^ { * } )$ . Lemma E.2 with $t = K + 1$ and the first condition in (42) with $t = K + 1 { \mathrm { ~ g i v e ~ } } \gamma ( K + 1 ) \left( { \bar { f } } ( { \bar { x } } ^ { K } ) - f ^ { * } \right) \leq R _ { 0 } ^ { 2 } + R ^ { 2 } \leq 2 R ^ { 2 }$ , which concludes the proof. □

Proof of Theorem F.1. Let $c _ { K }$ denote the constant c from Theorem F.1, i.e., $c _ { K } = c _ { \alpha } ^ { \mathrm { m o m } }$ under Assumption 3.4 and $c _ { K } = c _ { \alpha } ( \Lambda _ { K } )$ under Assumption 3.5, and let $\gamma _ { * } : = \operatorname* { m i n } \{ \gamma _ { L } , \gamma _ { 0 } , \gamma _ { 1 } , \gamma _ { 2 } \}$ computed with $c _ { K }$ . Let $\gamma \le \gamma _ { * }$ and $\begin{array} { r } { \lambda \ : = \ : \frac { R } { 4 0 \gamma \ell } } \end{array}$ . Then $\begin{array} { r } { \gamma \lambda = \frac { R } { 4 0 \ell } } \end{array}$ , and $\begin{array} { r } { \lambda \ge \frac { 8 0 \sqrt { 2 } L \ell R } { 4 0 \ell } = 2 \sqrt { 2 } L R } \end{array}$ since $\gamma \le \gamma _ { L }$ . It remains to verify that $\Phi _ { K + 1 } \stackrel { \cdot } { \leq } 3 R / c$ for the constant $c = 1 8 0 \operatorname* { m a x } \{ v , b / 3 \}$ from Appendix F, which implies the last two conditions in (40) (see the discussion after Proposition F.6).

Step 1: the bound with $c _ { K }$ . Raising $\gamma \leq \gamma _ { 0 } , \gamma \leq \gamma _ { 1 }$ and $\gamma \leq \gamma _ { 2 }$ to the powers $\alpha , \alpha / 2$ and $\alpha ,$ respectively, we get

$$
\begin{array} { r l r } { \gamma ^ { \alpha } ( 4 0 \ell ) ^ { \alpha - 1 } \sigma _ { 0 } ^ { \alpha } ( K + 1 ) \le \displaystyle \frac { R ^ { \alpha } } { c _ { K } } , } & { \ \gamma ^ { \alpha / 2 } ( 4 0 \ell ) ^ { \alpha - 1 } \sigma _ { 1 } ^ { \alpha } 2 ^ { \alpha } L ^ { \alpha / 2 } ( K + 1 ) ^ { \frac { 2 - \alpha } { 2 } } \le \displaystyle \frac { 1 } { c _ { K } } , } & \\ { \gamma ^ { \alpha } ( 4 0 \ell ) ^ { \alpha - 1 } \sigma _ { 1 } ^ { \alpha } 2 ^ { \alpha / 2 } L ^ { \alpha } \le \displaystyle \frac { 1 } { c _ { K } } . } & \end{array}\tag{52}
$$

Using $\lambda ^ { 1 - \alpha } = ( 4 0 \gamma \ell / R ) ^ { \alpha - 1 }$ and (52), we get

$$
\begin{array} { l } { { \Phi _ { K + 1 } = R ^ { 1 - \alpha } \gamma ^ { \alpha } ( 4 0 \ell ) ^ { \alpha - 1 } \sigma _ { 0 } ^ { \alpha } ( K + 1 ) + R \gamma ^ { \alpha / 2 } ( 4 0 \ell ) ^ { \alpha - 1 } \sigma _ { 1 } ^ { \alpha } 2 ^ { \alpha } L ^ { \alpha / 2 } ( K + 1 ) ^ { \frac { 2 - \alpha } { 2 } } } } \\ { { \qquad + R \gamma ^ { \alpha } ( 4 0 \ell ) ^ { \alpha - 1 } \sigma _ { 1 } ^ { \alpha } 2 ^ { \alpha / 2 } L ^ { \alpha } \le \displaystyle \frac { 3 R } { c _ { K } } . } } \end{array}
$$

Under Assumption 3.4, we have $c = c _ { K } ;$ ; thus, (40) holds, and Proposition F.6 gives the result, including (30).

Step 2: Assumption 3.5 and $\gamma = \gamma _ { * }$ . Let $\begin{array} { r } { \lambda _ { * } : = \frac { R } { 4 0 \gamma _ { * } \ell } } \end{array}$ and $\Lambda _ { * } : = \Lambda _ { \gamma _ { * } , \lambda _ { * } }$ . We show that $c \leq c _ { K }$ at $\gamma = \gamma _ { * }$ . If $\sigma _ { 0 } = \sigma _ { 1 } = 0$ , then $\Sigma _ { K + 1 } = 0 , \Phi _ { K + 1 } = 0$ and, by the convention in $( 3 8 ) , c = c _ { \alpha } ( 0 ) \leq c _ { K } ;$ thus, Steps 2 and 3 hold trivially, and, therefore, we assume that $\Sigma _ { K + 1 } > 0$ in both steps. If $\alpha \leq 1 . 1$ , then $b _ { \alpha } / 3 \geq b _ { 1 . 1 } / 3 > 3 . 5 > v _ { 1 . 1 } ( \infty ) \geq v _ { \alpha } ( \infty ) \geq v _ { \alpha } ( s )$ for all $s ,$ since $b _ { \alpha }$ is decreasing and $\begin{array} { r } { v _ { \alpha } ( \infty ) = \frac { 2 } { 2 - \alpha } ( \frac { 3 } { 2 } ) ^ { 2 - \alpha } } \end{array}$ is increasing in α. In this case, $c _ { \alpha } ( \cdot ) \equiv 6 0 b _ { \alpha }$ , and there is nothing to prove. Next, let $\alpha > 1 . 1$ , implying $b _ { \alpha } < b _ { 1 . 1 } < 1 0 . 8 ;$ we show that $\Lambda _ { * } \leq \Lambda _ { K }$ . With $\begin{array} { r } { \lambda _ { j } : = \frac { R } { 4 0 \gamma _ { j } \ell } } \end{array}$ for $j \in \{ L , 0 , 1 , 2 \}$ , we have $\lambda _ { * } = \operatorname* { m a x } _ { j } \lambda _ { j } . \mathrm { ~ I f ~ } \lambda _ { * } = \lambda _ { L } = 2 \sqrt { 2 } L R _ { \mathrm { ~ } }$ , then $\Lambda _ { * } \leq \Lambda _ { 2 \sqrt { 2 } L R } \leq \Lambda _ { K } . \ \mathrm { I f } \ \lambda _ { * } = \lambda _ { 0 }$ 2 then $\Sigma _ { K + 1 } \geq ( K + 1 ) \sigma _ { 0 } ^ { \alpha }$ and the definition of γ<sub>0</sub> give

$$
\Lambda _ { * } \leq \left( \frac { 3 \lambda _ { 0 } } { 2 \sigma _ { 0 } } \right) ^ { \alpha } = \left( \frac { 3 } { 2 } \right) ^ { \alpha } \frac { c _ { K } ( K + 1 ) } { 4 0 \ell } .
$$

If $\lambda _ { * } ~ = ~ \lambda _ { 2 }$ , the same bound follows from $\Sigma _ { K + 1 } \ \geq \ 2 ^ { \alpha / 2 } \sigma _ { 1 } ^ { \alpha } ( L R ) ^ { \alpha }$ and the definition of $\gamma _ { 2 }$ . If $\lambda _ { * } = \lambda _ { 1 }$ , then $\gamma _ { * } = \gamma _ { 1 }$ , and the same bound follows from $\bar { \Sigma _ { K + 1 } } \geq \sigma _ { 1 } ^ { \alpha } 2 ^ { \alpha } L ^ { \alpha / 2 } R ^ { \alpha } \gamma _ { 1 } ^ { - \alpha / 2 } ( K + 1 ) ^ { \frac { 2 - \alpha } { 2 } }$ and the definition of $\gamma _ { 1 }$ . It remains to show that $( \frac { 3 } { 2 } ) ^ { \alpha } \frac { c _ { K } ( K + 1 ) } { 4 0 \ell } \ \leq \ \Lambda _ { K }$ . The map $\Lambda \mapsto \Lambda / c _ { \alpha } ( \Lambda )$ is non-decreasing, since $\textstyle \Lambda \mapsto \Lambda / ( 1 + { \frac { 2 } { \alpha } } \ln ( 1 + \Lambda ) )$ is increasing and the other branches of $c _ { \alpha }$ are constant. Since $\Lambda _ { K } \geq 8 0 ( K + 1 )$ , it is suficient to show that $( \frac { 3 } { 2 } ) ^ { \alpha } c _ { \alpha } ( 8 0 ( K + 1 ) ) \leq 3 2 0 0 \ell$ . We have $( \frac { 3 } { 2 } ) ^ { \alpha } \cdot 1 8 0 v _ { \alpha } ( s ) \leq 4 0 5 ( 1 + 2 \ln ( 1 + s ) )$ , and $4 0 5 ( 1 + 2 \ln ( 1 + 8 0 ( \bar { K ^ { - } } 1 ) ) ) < 3 2 0 0 \ln ( 4 ( K + 1 ) ) < 3 2 0 0 \ell _ { \cdot }$ since the diference of the two sides of the first inequality is increasing in K and positive for $K = 0$ Moreover, $( \frac { 3 } { 2 } ) ^ { \alpha } \cdot 6 0 b _ { \alpha } < 2 . 2 5 \cdot 6 0 \cdot 1 0 . 8 < 3 2 0 0 \ln 4 < 3 2 0 0 \ell$ . Thus, $\Lambda _ { * } \leq \Lambda _ { K }$ , i.e., $c \leq c _ { K }$ , and $\Phi _ { K + 1 } c \leq 3 R$ at $\gamma = \gamma _ { * }$ by Step 1.

Step 3: Assumption 3.5 and $\gamma \leq \gamma _ { * }$ . Let $u ( \gamma ) : = \gamma ^ { \alpha } \Sigma _ { K + 1 }$ , which is non-decreasing in $\gamma ,$ and $\begin{array} { r } { C ^ { \prime } : = ( K + 1 ) ( \frac { 3 R } { 8 0 \ell } ) ^ { \alpha } } \end{array}$ . For $\begin{array} { r } { \lambda = \frac { R } { 4 0 \gamma \ell } } \end{array}$ , we have $\begin{array} { r } { \Phi _ { K + 1 } = ( \frac { 4 0 \ell } { R } ) ^ { \alpha - 1 } u ( \gamma ) } \end{array}$ and $\Lambda _ { \gamma , \lambda } = C ^ { \prime } / u ( \gamma )$ , which gives

$$
\Phi _ { K + 1 } c = 1 8 0 ( \frac { 4 0 \ell } { R } ) ^ { \alpha - 1 } \operatorname* { m a x } \{ u v _ { \alpha } ( \frac { C ^ { \prime } } { u } ) , \ : \frac { u b _ { \alpha } } { 3 } \} \Biggl \Biggl . | _ { u = u ( \gamma ) } .
$$

By Lemma D.6, $u \mapsto u v _ { \alpha } ( C ^ { \prime } / u )$ is non-decreasing, and, therefore, $\Phi _ { K + 1 }$ c is non-decreasing in $\gamma .$ Since $\Phi _ { K + 1 } c \leq 3 R$ at $\gamma = \gamma _ { * }$ by Step 2, the same holds for every $\gamma \leq \gamma _ { * }$ . Thus, (40) holds, and Proposition F.6 gives the result, including (30), which concludes the proof. □

Proof of Corollary F.2. For $\gamma = \operatorname* { m i n } \{ \gamma _ { L } , \gamma _ { 0 } , \gamma _ { 1 } , \gamma _ { 2 } \}$ , we have $\begin{array} { r } { \frac { 2 R ^ { 2 } } { \gamma ( K + 1 ) } = \operatorname* { m a x } _ { j \in \{ L , 0 , 1 , 2 \} } \frac { 2 R ^ { 2 } } { \gamma _ { j } ( K + 1 ) } } \end{array}$ and direct computations give $\begin{array} { r } { \frac { 2 R ^ { 2 } } { \gamma _ { j } ( K + 1 ) } = \Delta _ { j } } \end{array}$ for $j \in \{ L , 0 , 1 \}$ and

$$
\frac { 2 R ^ { 2 } } { \gamma _ { 2 } ( K + 1 ) } = \Delta _ { 2 } : = \frac { 2 \left( c 2 ^ { \alpha / 2 } 4 0 ^ { \alpha - 1 } \right) ^ { 1 / \alpha } \sigma _ { 1 } L R ^ { 2 } \ell ^ { ( \alpha - 1 ) / \alpha } } { K + 1 } .
$$

For example, ${ \textstyle \frac { 2 R ^ { 2 } } { \gamma _ { 1 } ( K + 1 ) } } = 2 \left( c 2 ^ { \alpha } 4 0 ^ { \alpha - 1 } \right) ^ { 2 / \alpha } \sigma _ { 1 } ^ { 2 } L R ^ { 2 } \ell ^ { 2 ( \alpha - 1 ) / \alpha } ( K + 1 ) ^ { { \frac { 2 - \alpha } { \alpha } } - 1 }$ with $\textstyle { \frac { 2 - \alpha } { \alpha } } - 1 = - { \frac { 2 ( \alpha - 1 ) } { \alpha } }$ The term $\Delta _ { 2 }$ is redundant. If $\sigma _ { 1 } = 0 .$ , then $\Delta _ { 1 } = \Delta _ { 2 } = 0$ and the claim is immediate. If $\sigma _ { 1 } > 0$ 2 then $\Delta _ { L } \Delta _ { 1 } > 0$ , and, with $\textstyle r : = { \frac { \alpha - 1 } { \alpha } } \leq { \frac { 1 } { 2 } }$ , we have

$$
\sqrt { \Delta _ { L } \Delta _ { 1 } } = \sqrt { 3 2 0 \sqrt { 2 } } \left( c 2 ^ { \alpha } 4 0 ^ { \alpha - 1 } \right) ^ { 1 / \alpha } \sigma _ { 1 } L R ^ { 2 } \frac { \ell ^ { r + 1 / 2 } } { ( K + 1 ) ^ { r + 1 / 2 } } ,
$$

$$
\frac { \Delta _ { 2 } } { \sqrt { \Delta _ { L } \Delta _ { 1 } } } = \frac { 2 \cdot 2 ^ { 1 / 2 } } { 2 \sqrt { 3 2 0 \sqrt { 2 } } } \ell ^ { - 1 / 2 } ( K + 1 ) ^ { r - 1 / 2 } \leq \frac { 1 } { 1 5 } ,
$$

since $\ell > 1$ and $r \leq \frac 1 2$ ; hence, $\Delta _ { 2 } \leq \sqrt { \Delta _ { L } \Delta _ { 1 } } \leq \operatorname* { m a x } \{ \Delta _ { L } , \Delta _ { 1 } \}$ , and $\begin{array} { r } { \frac { 2 R ^ { 2 } } { \gamma ( K + 1 ) } = \operatorname* { m a x } \{ \Delta _ { L } , \Delta _ { 0 } , \Delta _ { 1 } \} } \end{array}$ To derive the complexity bound, we use the fact that each of the three terms has the form const · $\ln ^ { p } ( K / \beta ) / K ^ { q }$ with $p \geq 0$ and $q > 0$ . Such a term is at most ε as soon as $K \geq A \left( { \frac { \mathrm { c o n s t } } { \varepsilon } } \right) ^ { 1 / q } \ln ^ { p / q } { \frac { \mathrm { c o n s t } } { \varepsilon \beta } }$ for a suficiently large constant $A \ = \ A ( p , q )$ , which depends only on α through the exponents $( p , q )$ (and blows up as $\alpha \downarrow 1$ , where $q \downarrow 0 )$ , since in this case l $\begin{array} { r } { \mathrm { n } ( K / \beta ) = O \left( \ln { \frac { \mathrm { c o n s t } } { \varepsilon \beta } } \right) } \end{array}$ . Under Assumption 3.5, $c = c _ { \alpha } ( \Lambda _ { K } ) \leq \operatorname* { m a x } \{ 2 7 0 ( 1 + 2 \ln ( 1 + 8 0 ( K + 1 ) ) + 2 \ln ( 1 + \Lambda _ { 2 { \sqrt { 2 } L R } } ) ) , 6 0 b _ { \alpha } \}$ adds a factor that is polylogarithmic in K and in the problem parameters, which can be handled in the same way. Taking the maximum over the three terms with $\begin{array} { l } { \displaystyle ( q , p ) = ( 1 , 1 ) , \ : \left( \frac { \alpha - 1 } { \alpha } , \frac { \alpha - 1 } { \alpha } \right) } \end{array}$ , and $\textstyle \left( { \frac { 2 ( \alpha - 1 ) } { \alpha } } , { \frac { 2 ( \alpha - 1 ) } { \alpha } } \right)$ , we get the stated complexity, which concludes the proof. □

Proof of Theorem F.4. The first condition in (40) is assumed, and the second one follows from $\begin{array} { r } { \gamma \le \frac { R } { 4 0 \lambda \ell } } \end{array}$ . Under Assumption 3.5, the variance constant from Proposition F.6 is $v _ { \alpha } ( \Lambda _ { \gamma , \lambda } ) \leq$ $v _ { \alpha } ( \Lambda _ { \lambda } ) = v ;$ , since $\Lambda _ { \gamma , \lambda } \leq \Lambda _ { \lambda }$ and $v _ { \alpha } ( \cdot )$ is non-decreasing. By (37), we have $\Sigma _ { K + 1 } = S _ { 0 } + S _ { 1 } \gamma ^ { - \alpha / 2 } +$ $S _ { 2 } = \Sigma _ { K + 1 } ( \gamma )$ , meaning that (31) is the third condition in (40), bΦ $\kappa { + } 1 \leq R / 2 0$ , and (32) is the fourth one, $v \gamma ^ { 2 } \lambda ^ { 2 - \alpha } \Sigma _ { K + 1 } \leq R ^ { 2 } / ( 2 4 0 0 \ell )$ . Therefore, the result follows from Proposition F.6. □

Explicit thresholds. Let

$$
\begin{array} { r l } & { \gamma _ { b } ^ { \lambda } : = \operatorname* { m i n } \left\{ \frac { R \lambda ^ { \alpha - 1 } } { 6 0 b S _ { 0 } } , \left( \frac { R \lambda ^ { \alpha - 1 } } { 6 0 b S _ { 1 } } \right) ^ { \frac { 2 } { 2 - \alpha } } , \frac { R \lambda ^ { \alpha - 1 } } { 6 0 b S _ { 2 } } \right\} , } \\ & { \gamma _ { v } ^ { \lambda } : = \operatorname* { m i n } \left\{ \frac { R } { \sqrt { 7 2 0 0 \ell v \lambda ^ { 2 - \alpha } S _ { 0 } } } , \left( \frac { R ^ { 2 } } { 7 2 0 0 \ell v \lambda ^ { 2 - \alpha } S _ { 1 } } \right) ^ { \frac { 2 } { 4 - \alpha } } , \frac { R } { \sqrt { 7 2 0 0 \ell v \lambda ^ { 2 - \alpha } S _ { 2 } } } \right\} , } \end{array}\tag{53}
$$

with the convention that a term with $S _ { j } ~ = ~ 0$ equals +∞; for $\alpha = 2$ , the second term of $\gamma _ { b } ^ { \lambda }$ is replaced by +∞ and the condition $\lambda \ge 6 0 b S _ { 1 } / R = 2 4 0 b \sigma _ { 1 } ^ { 2 } L R$ is imposed instead. Then every $\gamma \leq$ min $\{ \gamma _ { b } ^ { \lambda } , \gamma _ { v } ^ { \lambda } \}$ satisfies (31) and (32). Indeed, (31) holds if each of the three terms of $b \gamma \lambda ^ { 1 - \alpha } \Sigma _ { K + 1 } ( \gamma )$ is at most $R / 6 0$

$$
\begin{array} { r c c c l } { b \gamma \lambda ^ { 1 - \alpha } S _ { 0 } \leq \displaystyle \frac { R } { 6 0 } } & { \Longleftrightarrow } & { \gamma \leq \displaystyle \frac { R \lambda ^ { \alpha - 1 } } { 6 0 b S _ { 0 } } , } & & & \\ { b \gamma ^ { 1 - \alpha / 2 } \lambda ^ { 1 - \alpha } S _ { 1 } \leq \displaystyle \frac { R } { 6 0 } } & { \Longleftrightarrow } & { \gamma ^ { \frac { 2 - \alpha } { 2 } } \leq \displaystyle \frac { R \lambda ^ { \alpha - 1 } } { 6 0 b S _ { 1 } } } & { \Longleftrightarrow } & { \gamma \leq \left( \displaystyle \frac { R \lambda ^ { \alpha - 1 } } { 6 0 b S _ { 1 } } \right) ^ { \frac { 2 } { 2 - \alpha } } , } \\ { b \gamma \lambda ^ { 1 - \alpha } S _ { 2 } \leq \displaystyle \frac { R } { 6 0 } } & { \Longleftrightarrow } & { \gamma \leq \displaystyle \frac { R \lambda ^ { \alpha - 1 } } { 6 0 b S _ { 2 } } , } \end{array}
$$

where the middle line is for $\alpha \ < \ 2 $ ; for $\alpha = 2$ , the middle term equals $b S _ { 1 } / \lambda = 4 b \sigma _ { 1 } ^ { 2 } L R ^ { 2 } / \lambda$ which is at most $R / 6 0$ since $\lambda \ge 2 4 0 b \sigma _ { 1 } ^ { 2 } L R$ . Similarly, (32) holds if each of the three terms of $v \gamma ^ { 2 } \lambda ^ { 2 - \alpha } \Sigma _ { K + 1 } ( \gamma )$ is at most $R ^ { 2 } / ( 7 2 0 0 \ell )$ :

$$
\begin{array} { r c l } { { v \gamma ^ { 2 } \lambda ^ { 2 - \alpha } S _ { 0 } \le \frac { R ^ { 2 } } { 7 2 0 0 \ell } } } & { { \longleftrightarrow } } & { { \gamma \le \frac { R } { \sqrt { 7 2 0 0 \ell v \lambda ^ { 2 - \alpha } S _ { 0 } } } , } } \\ { { v \gamma ^ { 2 - \alpha / 2 } \lambda ^ { 2 - \alpha } S _ { 1 } \le \frac { R ^ { 2 } } { 7 2 0 0 \ell } } } & { { \longleftrightarrow } } & { { \gamma \le \left( \frac { R ^ { 2 } } { 7 2 0 0 \ell v \lambda ^ { 2 - \alpha } S _ { 1 } } \right) ^ { \frac { 2 } { 4 - \alpha } } , } } \\ { { v \gamma ^ { 2 } \lambda ^ { 2 - \alpha } S _ { 2 } \le \frac { R ^ { 2 } } { 7 2 0 0 \ell } } } & { { \Longleftrightarrow } } & { { \gamma \le \frac { R } { \sqrt { 7 2 0 0 \ell v \lambda ^ { 2 - \alpha } S _ { 2 } } } , } } \end{array}
$$

(for $\alpha = 2$ , the middle line reads $\gamma \leq R ^ { 2 } / ( 7 2 0 0 \ell v S _ { 1 } )$ , which is the stated term).

Proof of Corollary F.5. By the previous paragraph, $\begin{array} { r } { \gamma = \operatorname* { m i n } \{ \frac { R } { 4 0 \lambda \ell } , \gamma _ { b } ^ { \lambda } , \gamma _ { v } ^ { \lambda } \} } \end{array}$ is admissible in Theorem F.4, and $\frac { 2 R ^ { 2 } } { \gamma ( K + 1 ) }$ is the maximum of $\frac { 2 R ^ { 2 } } { ( K + 1 ) }$ divided by each of the seven thresholds (six for

$\alpha = 2$ , where the $S _ { 1 }$ bias threshold is $+ \infty$ and the corresponding term below is absent). The first one gives $\begin{array} { r } { \frac { 2 R ^ { 2 } } { \frac { R } { 4 0 \lambda \ell } ( K + 1 ) } = \frac { 8 0 \lambda R \ell } { K + 1 } } \end{array}$ . The three bias thresholds give

$$
\begin{array} { r l r } & { } & { \frac { 2 R ^ { 2 } \cdot 6 0 b S _ { 0 } } { R \lambda ^ { \alpha - 1 } ( K + 1 ) } = \frac { 1 2 0 b R \sigma _ { 0 } ^ { \alpha } } { \lambda ^ { \alpha - 1 } } , \frac { 2 R ^ { 2 } } { K + 1 } \left( \frac { 6 0 b S _ { 1 } } { R \lambda ^ { \alpha - 1 } } \right) ^ { \frac { 2 } { 2 - \alpha } } = 2 \left( \frac { 6 0 \cdot 2 ^ { \alpha } b R L ^ { \alpha / 2 } \sigma _ { 1 } ^ { \alpha } } { \lambda ^ { \alpha - 1 } } \right) ^ { \frac { 2 } { 2 - \alpha } } , } \\ & { } & { \frac { 2 R ^ { 2 } \cdot 6 0 b S _ { 2 } } { R \lambda ^ { \alpha - 1 } ( K + 1 ) } = \frac { 1 2 0 \cdot 2 ^ { \alpha / 2 } b \sigma _ { 1 } ^ { \alpha } L ^ { \alpha } R ^ { \alpha + 1 } } { \lambda ^ { \alpha - 1 } ( K + 1 ) } \le \frac { 1 2 0 \cdot 2 ^ { ( 3 - 2 \alpha ) / 2 } b \sigma _ { 1 } ^ { \alpha } L R ^ { 2 } } { K + 1 } \le \frac { 1 2 0 \sqrt 2 b \sigma _ { 1 } ^ { \alpha } L R ^ { 2 } } { K + 1 } , } \end{array}
$$

where the second identity uses $S _ { 1 } ^ { 2 / ( 2 - \alpha ) } = ( 2 ^ { \alpha } \sigma _ { 1 } ^ { \alpha } L ^ { \alpha / 2 } R ^ { \alpha } ) ^ { 2 / ( 2 - \alpha ) } ( K + 1 )$ and $R ^ { 2 } ( R ^ { \alpha - 1 } ) ^ { 2 / ( 2 - \alpha ) } ~ =$ $R ^ { 2 / ( 2 - \alpha ) }$ , and the last line uses $\lambda \overset { \smile } { \geq } 2 ^ { 3 / 2 } L R$ and $\alpha > 1$ . The three variance thresholds give, with ${ \sqrt { 7 2 0 0 } } = 6 0 { \sqrt { 2 } }$

$$
\begin{array} { c } { { \frac { 2 R \sqrt { 7 2 0 0 \ell v \lambda ^ { 2 - \alpha } S _ { 0 } } } { K + 1 } = 1 2 0 \sqrt { 2 } R \sqrt { \frac { v \lambda ^ { 2 - \alpha } \sigma _ { 0 } ^ { \alpha } \ell } { K + 1 } } , } } \\ { { \frac { 2 R ^ { 2 } } { K + 1 } \left( \frac { 7 2 0 0 \ell v \lambda ^ { 2 - \alpha } S _ { 1 } } { R ^ { 2 } } \right) ^ { \frac { 2 } { 4 - \alpha } } , \qquad \frac { 1 2 0 \sqrt { 2 } R \sqrt { v \lambda ^ { 2 - \alpha } S _ { 2 } \ell } } { K + 1 } . } } \end{array}
$$

This proves (33). The orders in K follow from $S _ { 1 } \propto ( K + 1 ) ^ { ( 2 - \alpha ) / 2 } ;$ : the fourth term of $\mathcal { V } _ { \lambda , K }$ is of the order $( K + 1 ) ^ { \frac { 2 - \alpha } { 4 - \alpha } - 1 } = ( K + 1 ) ^ { - \frac { 2 } { 4 - \alpha } }$ . Next, if $\sigma _ { 1 } = 0$ and $\lambda = \lambda _ { 0 } ( K + 1 ) ^ { 1 / \alpha }$ , then $K + 1 \geq$ $( 2 \sqrt { 2 } L R / \lambda _ { 0 } ) ^ { \alpha }$ is equivalent to $\lambda \ge 2 \sqrt { 2 } L R$ , and, substituting λ, we get $\begin{array} { r } { \frac { 8 0 \lambda R \ell } { K + 1 } = \frac { 8 0 \lambda _ { 0 } R \ell } { ( K + 1 ) ^ { ( \alpha - 1 ) / \alpha } } ; } \end{array}$ $\begin{array} { r } { \frac { 1 2 0 b R \sigma _ { 0 } ^ { \alpha } } { \lambda ^ { \alpha - 1 } } = \frac { 1 2 0 b \sigma _ { 0 } ^ { \alpha } \lambda _ { 0 } ^ { 1 - \alpha } R } { ( K + 1 ) ^ { ( \alpha - 1 ) / \alpha } } } \end{array}$ , and

$$
1 2 0 \sqrt { 2 } R \sqrt { \frac { v \lambda ^ { 2 - \alpha } \sigma _ { 0 } ^ { \alpha \ell } } { K + 1 } } = 1 2 0 \sqrt { 2 v \ell } \sigma _ { 0 } ^ { \alpha / 2 } \lambda _ { 0 } ^ { 1 - \alpha / 2 } R ( K + 1 ) ^ { \frac { 2 - \alpha } { 2 \alpha } - \frac { 1 } { 2 } } = \frac { 1 2 0 \sqrt { 2 v \ell } \sigma _ { 0 } ^ { \alpha / 2 } \lambda _ { 0 } ^ { 1 - \alpha / 2 } R } { ( K + 1 ) ^ { ( \alpha - 1 ) / \alpha } } .
$$

Bounding the maximum of the three quantities by their sum, we get (36). Finally, for $\sigma _ { 1 } > 0$ the remaining four terms are of the orders $( K + 1 ) ^ { - { \frac { 2 ( \alpha - 1 ) } { \alpha ( 2 - \alpha ) } } }$ (the second part of $\mathcal { N } _ { \lambda } ) , ( K + 1 ) ^ { - 1 }$ $( K + 1 ) ^ { \frac { 2 - \alpha } { 2 \alpha } - 1 }$ , and $( K + 1 ) ^ { \frac { 2 ( 2 - \alpha ) } { \alpha ( 4 - \alpha ) } - \frac { 2 } { 4 - \alpha } } = ( K + 1 ) ^ { - \frac { 4 ( \alpha - 1 ) } { \alpha ( 4 - \alpha ) } }$ , and each exponent is smaller than − α−1 α for $\alpha \in ( 1 , 2 )$ , which concludes the proof. □

## G Clip-SGD with an Arbitrary Clipping Level: Proof of Theorem G.1

## G.1 Full Statement

Theorem G.1 (Any clipping level: full statement of Theorem 5.4). Let Assumptions 3.1–3.3 hold for $Q = B _ { 2 R } ( x ^ { * } )$ , where $R \geq \| x ^ { 0 } - x ^ { * } \| > 0$ , and let Assumption 3.4 (with $\alpha \in ( 1 , 2 ] )$ or Assumption 3.5 (with $\alpha \in ( 1 , 2 ) )$ hold with $\sigma _ { 1 } \leq \bar { \sigma } _ { 1 }$ . Let $\lambda > 0$ be arbitrary, $\beta \in ( 0 , 1 )$ $\begin{array} { r } { K \in \{ 0 , 1 , \dotsc \} , \ell : = \ln \frac { 4 ( K + 1 ) } { \vartheta } , \bar { A } : = \sigma _ { 0 } ^ { \alpha } + \sigma _ { 1 } ^ { \alpha } ( \sqrt { 2 } L R ) ^ { \alpha } , V _ { \lambda } : = \operatorname* { m i n } \{ 4 \lambda ^ { 2 } , \kappa _ { 2 } ( 2 \lambda ) ^ { 2 - \alpha } \bar { A } \} } \end{array}$ , and let the stepsize satisfy

$$
0 < \gamma \leq \operatorname* { m i n } \left\{ { \frac { 1 } { 4 8 L } } , { \frac { R } { 4 0 \lambda \ell } } , { \frac { R } { \sqrt { 1 2 0 0 \ell V _ { \lambda } ( K + 1 ) } } } \right\} ,
$$

where the last term is $+ \infty \operatorname { i f } \bar { A } = 0$ . Then, with probability at least $1 - \beta$ , the iterates produced by Algorithm 1 with $\lambda _ { k } \equiv \lambda$ satisfy

$$
\operatorname* { m i n } _ { 0 \leq t \leq K } f ( x ^ { t } ) - f ^ { * } \leq \operatorname* { m a x } \bigg \{ \varepsilon _ { N } , \varepsilon _ { M } , \frac { 4 R ^ { 2 } } { \gamma ( K + 1 ) } , \frac { 4 6 0 8 L R ^ { 4 } } { \gamma ^ { 2 } \lambda ^ { 2 } ( K + 1 ) ^ { 2 } } \bigg \} ,
$$

where $\varepsilon _ { N } : = 4 8 \sqrt { 2 } c _ { \perp } R \bar { b } _ { \lambda }$ and $\varepsilon _ { M } : = 3 8 4 c _ { \perp } \sigma _ { 1 } ^ { \alpha } L R ^ { 2 } \operatorname* { m i n } \{ 1 , 9 6 c _ { \perp } \sigma _ { 1 } ^ { \alpha } \}$ .

For $\gamma$ equal to the minimum, the last two terms are

$$
O \bigg ( \frac { L R ^ { 2 } + \lambda R \ell } { K } + R \sqrt { \frac { \ell V _ { \lambda } } { K } } + \frac { L ^ { 3 } R ^ { 4 } } { \lambda ^ { 2 } K ^ { 2 } } + \frac { L R ^ { 2 } \ell ^ { 2 } } { K ^ { 2 } } + \frac { \ell L R ^ { 2 } V _ { \lambda } } { \lambda ^ { 2 } K } \bigg )
$$

with an absolute constant, and they vanish as $K  \infty$ (note that ${ V _ { \lambda } \leq \operatorname * { m a x } \{ 4 , 2 ^ { 2 - \alpha } \kappa _ { 2 } \} \lambda ( \bar { b } _ { \lambda } + }$ $\sigma _ { 1 } ^ { \alpha } ( \sqrt { 2 } L R ) ^ { \alpha } \lambda ^ { 1 - \alpha } ) )$ . This is exactly the term $\mathcal { V } _ { \lambda , K } ^ { \prime }$ from Theorem 5.4.

Throughout this section, $\sigma _ { 1 } \leq \bar { \sigma } _ { 1 } = 1 / ( 2 \cdot 2 8 ^ { 1 / \alpha } )$ , and $( \kappa _ { 1 } , \kappa _ { 2 } ) , c _ { \perp } , \bar { b } _ { \lambda } , \bar { A }$ , and $V _ { \lambda }$ are defined as in Section 5.2; $A ( x ) = \sigma _ { 0 } ^ { \alpha } + \sigma _ { 1 } ^ { \alpha } \| x \| ^ { \alpha }$ is defined as in Section 4, and $( b , v )$ are the clipping constants. We will use the following three consequences of the noise assumptions. Let z be a random vector with $\mathbb { P } \{ \| z \| > u \} \ \le \ A u ^ { - \alpha }$ for all $u > 0$ (under Assumption 3.5) or with $\mathbb { E } \Vert z \Vert ^ { \alpha } \leq A$ (under Assumption 3.4). Then, for every $c > 0$ , we have

$$
\mathbb { P } \{ \| z \| > c \} \le A c ^ { - \alpha } , \qquad \mathbb { E } \left[ \| z \| \mathbf { 1 } _ { \{ \| z \| > c \} } \right] \le \kappa _ { 1 } A c ^ { 1 - \alpha } , \qquad \mathbb { E } \operatorname* { m i n } \{ \| z \| , c \} ^ { 2 } \le \kappa _ { 2 } A c ^ { 2 - \alpha } .\tag{54}
$$

Indeed, under Assumption 3.5, the second bound follows from $\begin{array} { r } { \mathbb { E } [ Y \mathbf { 1 } _ { \{ Y > c \} } ] = c \mathbb { P } \{ Y > c \} + \int _ { c } ^ { \infty } \mathbb { P } \{ Y > } \end{array}$ u} du, and the third one follows from E min $\{ Y , c \} ^ { 2 } = 2 \int _ { 0 } ^ { c } u \mathbb { P } \{ Y \stackrel { . } { > } u \}$ du. Under Assumption 3.4, the first bound is Markov’s inequality, and the other two follow from ${ \cal Y } { \bf 1 } _ { \{ Y > c \} } \leq { \cal Y } ^ { \alpha } c ^ { 1 - \alpha }$ and min $\{ Y , c \} ^ { 2 } \leq Y ^ { \alpha } c ^ { 2 - \alpha }$ , where $Y : = \| z \|$ and $\alpha \leq 2$

## G.2 Clipping with an Arbitrary Level

Lemma G.2 (Clipping with an arbitrary level). Let $a \in \mathbb { R } ^ { d }$ with $G : = \| a \| > 0$ , let $\lambda > 0$ $D : = \operatorname* { m a x } \{ G , \lambda \} , \hat { a } : = a / G$ and $\boldsymbol { P } _ { \perp } : = \boldsymbol { I } - \hat { a } \hat { a } ^ { \top }$ . Let z be a random vector with $\mathbb { E } z ~ = ~ 0$ satisfying (54) with some $A \geq 0$ , and let $\widetilde X : = { \mathsf { c l i p } } ( a + z , \lambda )$ Then the following statements hold:

$$
\mathrm { ( i ) } \ \operatorname { \mathbb { E } } \| \widetilde { X } - \mathbf { c } \mathbf { 1 } \mathbf { i } _ { \mathsf { p } } ( a , \lambda ) \| ^ { 2 } \leq \operatorname { \mathbb { E } } \operatorname* { m i n } \{ \| z \| , 2 \lambda \} ^ { 2 } \leq \operatorname* { m i n } \{ 4 \lambda ^ { 2 } , \kappa _ { 2 } ( 2 \lambda ) ^ { 2 - \alpha } A \} ;
$$

(ii) $\| P _ { \perp } \mathbb { E } \widetilde { X } \| \leq c _ { \perp } \lambda A D ^ { - \alpha }$ ;

$$
\begin{array} { r } { \mathrm { ( i i i ) ~ i f ~ } G \geq \lambda / 2 \mathrm { ~ a n d ~ } \mathbb { P } \{ \| z \| > G / 2 \} \leq \frac { 1 } { 1 4 } , \mathrm { t h e n ~ } \Big \langle \mathbb { E } \widetilde { X } , \hat { a } \Big \rangle \geq \frac { 1 } { 6 } \operatorname* { m i n } \{ G , \lambda \} . } \end{array}
$$

Proof. (i) The map $\mathsf { c l i p } ( \cdot , \lambda )$ is the Euclidean projection onto $B _ { \lambda } ( 0 ) ;$ therefore, it is 1-Lipschitz, and both $\widetilde { X }$ and $\mathsf { c l i p } ( a , \lambda )$ belong to $B _ { \lambda } ( 0 )$ . Thus, $\| \widetilde { X } - \mathsf { c l i p } ( a , \lambda ) \| \le \operatorname* { m i n } \{ \| z \| , 2 \lambda \}$ , and the claim follows from (54) with $c = 2 \lambda$

(ii) Let $y : = a + z , M ( v ) : = \operatorname* { m a x } \{ \| v \| , \lambda \}$ , and $s ( v ) : = \lambda / M ( v ) = \operatorname* { m i n } \{ 1 , \lambda / \| v \| \}$ ; then $\widetilde { X } = s ( y ) y$ Since $P _ { \bot } a = 0$ and $\mathbb { E } P _ { \perp } z = 0$ , we have $P _ { \perp } \mathbb { E } \tilde { X } = \mathbb { E } [ s ( y ) P _ { \perp } z ] = \mathbb { E } [ ( s ( y ) - s ( a ) ) P _ { \perp } z ]$ . The map $v \mapsto M ( v )$ is 1-Lipschitz and $\boldsymbol { M } ( \boldsymbol { a } ) = \boldsymbol { D }$ ; hence, $| s ( y ) - s ( a ) | = \lambda | M ( a ) - M ( y ) | / ( M ( a ) M ( y ) ) \leq$

$\lambda \| z \| / ( D M ( y ) )$ . On the event $\{ \| z \| \le D / 2 \}$ , we have $M ( y ) ~ \geq ~ D / 2 { \mathrm { ~ ~ ( i f ~ } } G ~ \geq ~ \lambda$ , then $\| y \| \ge$ $G - \| z \| \geq G / 2$ ; otherwise, $M ( y ) \geq \lambda = D )$ . Thus, $\| ( s ( y ) - s ( a ) ) P _ { \perp } z \| \leq 2 \lambda \| z \| ^ { 2 } / D ^ { 2 }$ on this event, and, by (54),

$$
\mathbb { E } \left[ \| ( s ( y ) - s ( a ) ) P _ { \perp } z \| \mathbf { 1 } _ { \{ \| z \| \leq D / 2 \} } \right] \leq \frac { 2 \lambda } { D ^ { 2 } } \mathbb { E } \operatorname* { m i n } \left\{ \| z \| , \frac { D } { 2 } \right\} ^ { 2 } \leq 2 ^ { \alpha - 1 } \kappa _ { 2 } \lambda A D ^ { - \alpha } .
$$

On the event $\{ \| z \| > D / 2 \}$ , we use $\| s ( y ) P _ { \perp } z \| = \| P _ { \perp } \widetilde { X } \| \leq \lambda < 2 \lambda \| z \| / D$ and $s ( a ) \| P _ { \bot } z \| \le \lambda \| z \| / D$ Thus, $\| ( s ( y ) - s ( a ) ) P _ { \perp } z \| \leq 3 \lambda \| z \| / D$ on this event, and

$$
\mathbb { E } \left[ \Vert ( s ( y ) - s ( a ) ) P _ { \perp } z \Vert _ { \mathbf { 1 } _ { \{ \Vert z \Vert > D / 2 \} } } \right] \leq \frac { 3 \lambda } { D } \mathbb { E } \left[ \Vert z \Vert \mathbf { 1 } _ { \{ \Vert z \Vert > D / 2 \} } \right] \leq 3 \cdot 2 ^ { \alpha - 1 } \kappa _ { 1 } \lambda A D ^ { - \alpha } .
$$

Summing up the two bounds, we get (ii).

(iii) Let $\begin{array} { r } { p : = \mathbb { P } \{ \| z \| > G / 2 \} \le \frac { 1 } { 1 4 } } \end{array}$ . On the event $\{ \| z \| \leq G / 2 \}$ , we have $\langle y , \hat { a } \rangle = G + \langle z , \hat { a } \rangle \geq G / 2$ and $\| y \| \leq 3 G / 2 ;$ hence, $\begin{array} { r } { s ( y ) \geq \operatorname* { m i n } \{ 1 , \frac { 2 \lambda } { 3 G } \} } \end{array}$ and $\left. \widetilde { X } , \hat { a } \right. = s ( y ) \left. y , \hat { a } \right. \ge \operatorname* { m i n } \{ \frac { G } { 2 } , \frac { \lambda } { 3 } \} \ge \frac { 1 } { 3 }$ min{G, λ}. On the complementary event, we have $\left. { \widetilde { X } } , { \hat { a } } \right. \geq - \| { \widetilde { X } } \| \geq - \lambda$ . Since $G \ge \lambda / 2$ , we have $\lambda \ \leq$ $2 \operatorname* { m i n } \{ G , \lambda \}$ , and, therefore,

$$
\left. \mathbb { E } \widetilde { X } , \hat { a } \right. \geq \frac { 1 - p } { 3 } \operatorname* { m i n } \{ G , \lambda \} - p \lambda \geq \left( \frac { 1 } { 3 } - \frac { 7 p } { 3 } \right) \operatorname* { m i n } \{ G , \lambda \} \geq \frac { 1 } { 6 } \operatorname* { m i n } \{ G , \lambda \} .
$$

In the proof of Theorem G.1, we apply these deterministic statements pointwise at random points via the freezing formula (see below). This allows us to avoid conditional versions with random thresholds such as $G / 2$ and $D / 2$

## G.3 The Mean Clipped Field

Lemma G.3 (Mean clipped field). Let the assumptions of Theorem G.1 hold, and let $x \in$ $B _ { \sqrt { 2 } R } ( x ^ { * } ) , u : = x - x ^ { * } , \delta : = f ( x ) - f ^ { * } , G : = \| \nabla f ( x ) \|$ , and $X : = g ( x , \xi )$ with $\xi \sim \mathcal { D }$ . Let

$$
w : = 1 \mathrm { i f } G \leq \frac { \lambda } { 2 } , \qquad w : = \frac { \operatorname* { m i n } \{ G , \lambda \} } { 6 G } \mathrm { i f } G > \frac { \lambda } { 2 } ,
$$

and let $\varepsilon _ { N } = 4 8 \sqrt { 2 } c _ { \perp } R \bar { b } _ { \lambda }$ and $\varepsilon _ { M } = 3 8 4 c _ { \perp } \sigma _ { 1 } ^ { \alpha } L R ^ { 2 } \operatorname* { m i n } \{ 1 , 9 6 c _ { \perp } \sigma _ { 1 } ^ { \alpha } \}$ be defined as in Theorem G.1. If $\delta > \operatorname* { m a x } \{ \varepsilon _ { N } , \varepsilon _ { M } \}$ , then $\langle \mathtt { E c l i p } ( X , \lambda ) , u \rangle \ge \frac { 3 } { 4 } w \delta$

Proof. Let $\widetilde { X } : = { \mathsf { c l i p } } ( X , \lambda )$ and $z : = X - \nabla f ( x )$ ; then $\mathbb { E } z = 0$ and (54) holds with $A = A ( \nabla f ( x ) ) =$ $\sigma _ { 0 } ^ { \alpha } + \sigma _ { 1 } ^ { \alpha } G ^ { \alpha }$ . By convexity, we have $\langle \nabla f ( x ) , u \rangle \geq \delta$ , and $\delta \leq G \Vert u \Vert \leq { \sqrt { 2 } } R G$ . In particular, $G > 0$ By Assumption 3.2, we also have $\delta \geq G ^ { 2 } / ( 2 L )$ . We start with the following consequence of $\delta > \varepsilon _ { M } :$ for every $C \in ( 0 , 9 6 \sqrt { 2 } c _ { \perp } ]$ , we have

$$
{ \cal C } \sigma _ { 1 } ^ { \alpha } R { \cal G } \leq \delta .\tag{55}
$$

Indeed, if $\varepsilon _ { M } = 3 8 4 c _ { \perp } \sigma _ { 1 } ^ { \alpha } L R ^ { 2 }$ , then either $G \geq 2 C L \sigma _ { 1 } ^ { \alpha } R ,$ and then $C \sigma _ { 1 } ^ { \alpha } R G \le G ^ { 2 } / ( 2 L ) \le \delta .$ , or $G < 2 C L \sigma _ { 1 } ^ { \alpha } R .$ , and then $\delta \leq \sqrt { 2 } R G < 2 \sqrt { 2 } C \sigma _ { 1 } ^ { \alpha } L R ^ { 2 } \leq 3 8 4 c _ { \perp } \sigma _ { 1 } ^ { \alpha } L R ^ { 2 } = \varepsilon _ { M }$ , which is excluded. If $\varepsilon _ { M } = 3 6 8 6 4 c _ { \scriptscriptstyle \mathrm { ~ l ~ } } ^ { 2 } \sigma _ { 1 } ^ { 2 \alpha } L R ^ { 2 }$ , then $C \sigma _ { 1 } ^ { \alpha } R G \leq C \sigma _ { 1 } ^ { \alpha } R \sqrt { 2 L \delta } \leq \delta ,$ since $\delta > \varepsilon _ { M } \geq 2 C ^ { 2 } \sigma _ { 1 } ^ { 2 \alpha } L R ^ { 2 }$ (note that $2 ( 9 6 \sqrt { 2 } c _ { \perp } ) ^ { 2 } = 3 \bar { 6 } 8 6 4 c _ { \perp } ^ { 2 } ,$ ). Next, we consider three cases.

Case $G \leq \lambda / 2$ . By Lemma D.1 or Lemma D.2, we have $\begin{array} { r } { \| \mathbb { E } \widetilde { X } - \nabla f ( x ) \| \le b A \lambda ^ { 1 - \alpha } = b \sigma _ { 0 } ^ { \alpha } \lambda ^ { 1 - \alpha } + } \end{array}$ $b \sigma _ { 1 } ^ { \alpha } G ^ { \alpha } \lambda ^ { 1 - \alpha }$ , and also $\| \mathbb { E } \tilde { X } - \nabla f ( x ) \| \le \lambda + G \le \frac { 3 } { 2 } \lambda$ . If $\lambda \ge \sigma _ { 0 }$ , then the first bound contains $b \sigma _ { 0 } ^ { \alpha } \lambda ^ { 1 - \alpha } = b \bar { b } _ { \lambda } ;$ otherwise, the second one is at most $\begin{array} { r } { \frac { 3 } { 2 } \sigma _ { 0 } = \frac { 3 } { 2 } \bar { b } _ { \lambda } } \end{array}$ . In both cases, since $G \leq \lambda / 2$ implies $G ^ { \alpha } \lambda ^ { 1 - \alpha } \leq 2 ^ { 1 - \alpha } G$ , we have

$$
\begin{array} { r } { \| \mathbb { E } \widetilde { X } - \nabla f ( x ) \| \le \operatorname* { m a x } \left\{ b , \frac { 3 } { 2 } \right\} \bar { b } _ { \lambda } + 2 ^ { 1 - \alpha } b \sigma _ { 1 } ^ { \alpha } G . } \end{array}
$$

Thus, by $\| u \| \leq \sqrt { 2 } R$ , max $\langle b , \frac { 3 } { 2 } \} \leq c _ { \perp }$ , and (55) with $C = 8 \sqrt 2 \cdot 2 ^ { 1 - \alpha } b \leq 8 \sqrt 2 c _ { \perp }$ , we get

$$
\left. \mathbb { E } \widetilde { X } , u \right. \ge \delta - \sqrt { 2 } R c _ { \perp } \bar { b } _ { \lambda } - \sqrt { 2 } \cdot 2 ^ { 1 - \alpha } b \sigma _ { 1 } ^ { \alpha } R G \ge \delta - \frac { \varepsilon _ { N } } { 4 8 } - \frac { \delta } { 8 } \ge \frac { 3 } { 4 } \delta = \frac { 3 } { 4 } w \delta .
$$

Here max $\{ b , { \frac { 3 } { 2 } } \} \leq c _ { \perp }$ since $c _ { \perp } \geq 2 ^ { \alpha - 1 } \cdot 4 \geq 4 ;$ under Assumption 3.5, $c _ { \perp } \geq 3 \cdot 2 ^ { \alpha - 1 } \kappa _ { 1 } = 3 \alpha b _ { \alpha } \geq b _ { \alpha }$ and under Assumption $3 . 4 , c _ { \perp } = 2 ^ { \alpha + 1 } \ge 4 > 1 \ge b$

Case $G > \lambda / 2$ and $G \geq G _ { 1 } : = 2 \cdot 2 8 ^ { 1 / \alpha } \sigma _ { 0 }$ . By $( 5 4 ) , G \geq G _ { 1 }$ , and $\sigma _ { 1 } \leq \bar { \sigma } _ { 1 }$ , we have

$$
\mathbb { P } \{ \| z \| > G / 2 \} \le 2 ^ { \alpha } A G ^ { - \alpha } = \frac { 2 ^ { \alpha } \sigma _ { 0 } ^ { \alpha } } { G ^ { \alpha } } + 2 ^ { \alpha } \sigma _ { 1 } ^ { \alpha } \le \frac { 1 } { 2 8 } + \frac { 1 } { 2 8 } = \frac { 1 } { 1 4 } ,
$$

and Lemma G.2(iii) gives $\left. \mathbb { E } \widetilde { X } , \hat { a } \right. \ge \frac { 1 } { 6 } \operatorname* { m i n } \{ G , \lambda \} = w G$ , where ${ \hat { a } } : = \nabla f ( x ) / G$ . Since $\langle \hat { a } , u \rangle \geq$ $\delta / G \geq 0$ , the component of $\mathbb { E } \widetilde { X }$ along ˆa contributes at least wδ to $\left. \mathbb { E } \widetilde { X } , u \right.$ . Next, let $D : =$ max $\{ G , \lambda \}$ ; then $w = \lambda / ( 6 D )$ in this case, $D \geq \operatorname* { m a x } \{ \lambda , \sigma _ { 0 } \}$ (since $G _ { 1 } \geq \sigma _ { 0 } )$ , and $D < 2 G$ (since $\lambda < 2 G )$ . By Lemma $\mathrm { G . 2 ( i i ) }$ , the orthogonal component contributes at least

$$
\begin{array} { r l r } & { } & { - \sqrt { 2 } R c _ { \perp } \lambda A D ^ { - \alpha } = - \sqrt { 2 } R c _ { \perp } \lambda \sigma _ { 0 } ^ { \alpha } D ^ { - \alpha } - \sqrt { 2 } R c _ { \perp } \lambda \sigma _ { 1 } ^ { \alpha } \left( \frac { G } { D } \right) ^ { \alpha } } \\ & { } & { \geq - w \cdot 6 \sqrt { 2 } c _ { \perp } R \sigma _ { 0 } ^ { \alpha } D ^ { 1 - \alpha } - w \cdot 6 \sqrt { 2 } c _ { \perp } R \sigma _ { 1 } ^ { \alpha } D . } \end{array}
$$

The first term is at least $- w \cdot 6 \sqrt { 2 } c _ { \perp } R \bar { b } _ { \lambda } = - w \varepsilon _ { N } / 8 \geq - w \delta / 8$ , since $D \geq \operatorname* { m a x } \{ \lambda , \sigma _ { 0 } \}$ . The second one is at least $- w \cdot 1 2 \sqrt { 2 } c _ { \perp } R \sigma _ { 1 } ^ { \alpha } G \geq - w \delta / 8$ , due to $D < 2 G$ and (55) with $C = 9 6 \sqrt { 2 } c _ { \perp }$ . Thus, $\begin{array} { r } { \left. \mathbb { E } \widetilde { X } , u \right. \ge w ( \delta - \delta / 8 - \delta / 8 ) = \frac { 3 } { 4 } w \delta . } \end{array}$

Case $\lambda / 2 < G < G _ { 1 } .$ This case is impossible when $\delta > \varepsilon _ { N }$ . Indeed, it requires $\lambda < 2 G _ { 1 } =$ $4 \cdot 2 8 ^ { 1 / \alpha } \sigma _ { 0 }$ . Then max $\{ \lambda , \sigma _ { 0 } \} \le 4 \cdot 2 8 ^ { 1 / \alpha } \sigma _ { 0 }$ , and, therefore, $\sigma _ { 0 } \leq ( 4 \cdot 2 8 ^ { 1 / \alpha } ) ^ { \alpha - 1 } \bar { b } _ { \lambda }$ . Therefore,

$$
\delta \leq \sqrt { 2 } R G < \sqrt { 2 } R G _ { 1 } = 2 \cdot 2 8 ^ { 1 / \alpha } \sqrt { 2 } R \sigma _ { 0 } \leq 2 \cdot 2 8 \cdot 4 ^ { \alpha - 1 } \sqrt { 2 } R \bar { b } _ { \lambda } = 1 4 \cdot 4 ^ { \alpha } \sqrt { 2 } R \bar { b } _ { \lambda } \leq \varepsilon _ { N } ,
$$

where the last step uses $c _ { \perp } \geq 2 ^ { \alpha - 1 } \cdot 4 = 2 ^ { \alpha + 1 }$ (since $\kappa _ { 1 } , \kappa _ { 2 } \geq 1 )$ and $1 4 \cdot 4 ^ { \alpha } \leq 4 8 \cdot 2 ^ { \alpha + 1 }$ , i.e., $1 4 \cdot 2 ^ { \alpha } \leq 9 6$ , for $\alpha \leq 2$ . This concludes the proof. □

## G.4 Proof of Theorem G.1

Let $\varepsilon _ { N }$ and $\varepsilon _ { M }$ be defined as in Lemma G.3, let $\varepsilon _ { \ast } : = \operatorname* { m a x } \{ \varepsilon _ { N } , \varepsilon _ { M } \}$ , let us introduce the notation $R _ { t } : = \| x ^ { t } - x ^ { * } \| , u _ { t } : = x ^ { t } - x ^ { * } , \delta _ { t } : = f ( x ^ { t } ) - f ^ { * } , G _ { t } : = \| \nabla f ( x ^ { t } ) \|$ , and let $w _ { t }$ be the weight from Lemma G.3 at $x ^ { t }$ . Let $\widetilde { g } ^ { t } : = \mathsf { c l i p } ( g ^ { t } , \lambda ) , ~ \theta _ { t } ^ { u } : = \widetilde { g } ^ { t } - \mathbb { E } [ \widetilde { g } ^ { t } ~ | ~ \mathcal { G } _ { t } ]$ (as in (25)), and $\widetilde { m } _ { t } : = \mathbf { \Gamma }$ $\| \widetilde { \boldsymbol { g } } ^ { t } - \mathsf { c l i p } ( \nabla f ( \boldsymbol { x } ^ { t } ) , \lambda ) \| ^ { 2 }$ . Let

$$
\tau : = \operatorname* { i n f } \{ t \geq 0 : ~ \delta _ { t } \leq \varepsilon _ { * } \} , \qquad \iota _ { t } : = { \mathbf 1 } _ { \{ t < \tau \} } , \qquad \eta _ { t } : = u _ { t } { \mathbf 1 } _ { \{ R _ { t } \leq \sqrt { 2 } R \} } .
$$

Since $\delta _ { t }$ is $\mathcal { G } _ { t ^ { - } } \mathrm { m e a s u r a b l e } , \tau$ is a stopping time, and $\mathbf { } _ { l , \mathrm { ~ } } w _ { t }$ , and $\eta _ { t }$ are $\mathcal { G } _ { t } { \mathrm { - m e a s u r a b l e } }$

One-step inequality. Let t be such that $\iota _ { t } = 1$ and $R _ { t } \leq \sqrt { 2 } R$ . Then we have

$$
R _ { t + 1 } ^ { 2 } \leq R _ { t } ^ { 2 } - \gamma w _ { t } \delta _ { t } - 2 \gamma \left. \theta _ { t } ^ { u } , \eta _ { t } \right. + 2 \gamma ^ { 2 } \widetilde { m } _ { t } .\tag{56}
$$

Indeed, $R _ { t + 1 } ^ { 2 } = R _ { t } ^ { 2 } - 2 \gamma \left. \widetilde { g } ^ { t } , u _ { t } \right. + \gamma ^ { 2 } \| \widetilde { g } ^ { t } \| ^ { 2 } , \ u _ { t } = \eta _ { t }$ , and $\| \widetilde { g } ^ { t } \| ^ { 2 } \ \le \ 2$ min $\{ G _ { t } , \lambda \} ^ { 2 } + 2 \widetilde { m } _ { t }$ , since $\| \mathbf { c } \mathbf { 1 i p } ( \nabla f ( x ^ { t } ) , \lambda ) \| = \operatorname* { m i n } \{ \dot { G } _ { t } , \lambda \}$ . Let $F ( x ) : = \mathbb { E } _ { \xi } { \mathsf { c l i p } } ( g ( x , \xi ) , \lambda )$ and $\Phi ( x ) : = \mathbb { E } _ { \xi } \| { \mathsf { c l i p } } ( g ( x , \xi ) , \lambda ) -$ cl $\cdot { \tt i p } ( \nabla f ( x ) , \lambda ) \| ^ { 2 }$ . These functions are bounded and measurable. Since $x ^ { t }$ is $\mathcal { G } _ { t } .$ -measurable and $\boldsymbol { \xi } ^ { t }$ is independent of $\mathcal { G } _ { t }$ , the freezing formula (as in the proof of Lemma I.3) gives $\mathbb { E } [ \widetilde { g } ^ { t } \mid \mathcal { G } _ { t } ] = F ( x ^ { t } )$ and $\mathbb { E } [ \widetilde { m } _ { t } \mid { \mathcal G } _ { t } ] = \Phi ( x ^ { t } )$ almost surely. Thus, Lemma G.3, applied at the point $x ^ { t } ( \omega ) \in B _ { \sqrt { 2 } R } ( x ^ { * } )$ gives $\begin{array} { r } { \left. \mathbb { E } [ \widetilde { g } ^ { t } \mid \mathcal { G } _ { t } ] , u _ { t } \right. \ge \frac { 3 } { 4 } w _ { t } \delta _ { t } } \end{array}$ . Finally, $2 \gamma ^ { 2 }$ min $\begin{array} { r } { \{ G _ { t } , \lambda \} ^ { 2 } \le \frac 1 2 \gamma w _ { t } \delta _ { t } } \end{array}$ . For $G _ { t } \leq \lambda / 2$ , this follows from $G _ { t } ^ { 2 } \leq \dot { 2 } L \delta _ { t }$ and $\gamma \leq 1 / ( 8 L )$ . For $G _ { t } > \lambda / 2$ , it is equivalent to $2 4 \gamma$ min $\{ G _ { t } , \lambda \} G _ { t } \leq \delta _ { t }$ , which follows from min $\{ G _ { t } , \lambda \} G _ { t } \leq G _ { t } ^ { 2 } \leq 2 L \delta _ { t }$ and $\gamma \leq 1 / ( 4 8 L )$

Variance bounds. If $R _ { t } \leq \sqrt { 2 } R$ , then $\| \nabla f ( x ^ { t } ) \| \le L R _ { t } \le \sqrt { 2 } L R$ , implying $A ( \nabla f ( x ^ { t } ) ) \leq \bar { A }$ and Lemma $\mathrm { G . 2 ( i ) }$ , applied at the point $x ^ { t } ( \omega )$ with $A = A ( \nabla f ( x ^ { t } ) )$ via the freezing formula, gives $\mathbb { E } [ \| \theta _ { t } ^ { u } \| ^ { 2 } \ | \ \mathcal { G } _ { t } ] \leq \mathbb { E } [ \widetilde { m } _ { t } \ | \ \mathcal { G } _ { t } ] \leq V _ { \lambda } .$ , where we use that the bound in Lemma $\mathrm { G . 2 ( i ) }$ is non-decreasing in A. Moreover, we always have $\| \theta _ { t } ^ { u } \| \leq 2 \lambda$ and $\widetilde { m } _ { t } \le 4 \lambda ^ { 2 }$

Events and induction. As in the proof of Proposition F.6, our proof is induction-based. For $\begin{array} { r } { T \ge 1 , \mathrm { l e t } \ S _ { T } : = - 2 \gamma \sum _ { t < T } \iota _ { t } \left. \theta _ { t } ^ { u } , \eta _ { t } \right. + 2 \gamma ^ { 2 } \sum _ { t < T } \iota _ { t } \widetilde m _ { t } } \end{array}$ . We introduce probability event $E _ { k }$ for each $k = 0 , 1 , \ldots , K + 1$ as follows: $E _ { 0 } : = \Omega$ and, for $k = 1 , \ldots , K + 1$ ，

$$
E _ { k } : = \Big \{ \forall T \in \{ 1 , \dots , k \} : \ S _ { T } \leq R ^ { 2 } \ \mathrm { ~ a n d ~ } \ \Big ( T \leq \tau \ \Rightarrow \ R _ { T } \leq \sqrt { 2 } R \Big ) \Big \} .
$$

We will prove by induction that $\begin{array} { r } { \mathbb { P } ( E _ { k } ) ~ \ge ~ 1 - ~ \frac { k \beta } { K + 1 } } \end{array}$ for all $k = 0 , 1 , \ldots , K + 1$ The base of the induction is trivial. Next, we assume that the statement holds for $k = T - 1$ with some $T \in \{ 1 , \ldots , K + 1 \}$ and prove it for $k = T$ . On $E _ { T - 1 }$ , every $t < T$ with $\iota _ { t } = 1$ satisfies $R _ { t } \leq \sqrt { 2 } R$ $( \mathrm { f o r } ~ t = 0 , ~ R _ { 0 } \leq R )$ , meaning that $x ^ { t } \in Q$ and the variance bounds hold for all active $t < T$

Let $X _ { t } : = - 2 \gamma _ { t } \langle \theta _ { t } ^ { u } , \eta _ { t } \rangle$ . This is a martingale diference sequence with $| X _ { t } | \le 4 \sqrt { 2 } \gamma \lambda R \le$ $\frac { R ^ { 2 } } { 5 \ell } \ = : c _ { 1 }$ , since $\gamma \lambda \le R / ( 4 0 \ell )$ . On $E _ { T - 1 }$ , its conditional variances satisfy $\begin{array} { r } { \sum _ { t < T } \mathbb { E } [ X _ { t } ^ { 2 } \mid \mathcal { G } _ { t } ] \ \le } \end{array}$ $\begin{array} { r } { 8 \gamma ^ { 2 } R ^ { 2 } V _ { \lambda } ( K + 1 ) \le \frac { R ^ { 4 } } { 1 5 0 \ell } = : G _ { F } } \end{array}$ , since $\begin{array} { r } { \gamma ^ { 2 } V _ { \lambda } ( K + 1 ) \le \frac { R ^ { 2 } } { 1 2 0 0 \ell } } \end{array}$ . Next, let $Y _ { t } : = 2 \gamma ^ { 2 } \iota _ { t } ( \widetilde { m } _ { t } - \mathbb { E } [ \widetilde { m } _ { t } \mid \mathcal { G } _ { t } ] )$ Then $Y _ { t }$ is a martingale diference with $\begin{array} { r } { | Y _ { t } | \le 8 \gamma ^ { 2 } \lambda ^ { 2 } \le \frac { R ^ { 2 } } { 2 0 0 \ell ^ { 2 } } \le c _ { 1 } } \end{array}$ , and, since $\iota _ { t }$ is G<sub>t</sub>-measurable with $\iota _ { t } ^ { 2 } = \iota _ { t } , \mathbb { E } [ Y _ { t } ^ { 2 } \mid { \mathcal G } _ { t } ] = 4 \gamma ^ { 4 } \iota _ { t } \operatorname { V a r } ( \widetilde { m } _ { t } \mid { \mathcal G } _ { t } ) \leq 4 \gamma ^ { 4 } \iota _ { t } \mathbb { E } [ \widetilde { m } _ { t } ^ { 2 \operatorname { v a r } } \mid { \mathcal G } _ { t } ] \leq 1 6 \gamma ^ { 4 } \lambda ^ { 2 } \iota _ { t } \mathbb { E } [ \widetilde { m } _ { t } \mid { \mathcal G } _ { t } ]$ . On $E _ { T - 1 }$ every active $t < T$ satisfies $x ^ { t } \in Q$ , and, therefore, $\iota _ { t } \mathbb { E } [ \widetilde { m } _ { t } \ | \ { \mathcal { G } } _ { t } ] \leq \iota _ { t } V _ { \lambda }$ , while for inactive t both sides vanish; hence, on $\begin{array} { r } { E _ { T - 1 } , \sum _ { t < T } \mathbb { E } [ Y _ { t } ^ { 2 } \mid \mathcal { G } _ { t } ] \le 1 6 \gamma ^ { 4 } \lambda ^ { 2 } V _ { \lambda } ( K + 1 ) = 1 6 ( \gamma \lambda ) ^ { 2 } \gamma ^ { 2 } V _ { \lambda } ( K + 1 ) \le G _ { F } } \end{array}$ Lemma E.3 with $b _ { 1 } = R ^ { 2 } / 5 , c _ { 1 }$ , and $G _ { F }$ , applied in the same way as in the proof of Proposition F.6, gives events $E _ { X }$ and $E _ { Y }$ , each of probability at least $\begin{array} { r } { 1 - \frac { \beta } { 2 ( K + 1 ) } } \end{array}$ , such that $\begin{array} { r } { \left| \sum _ { t < T } X _ { t } \right| \le \frac { R ^ { 2 } } { 5 } } \end{array}$ and $\begin{array} { r } { \left| \sum _ { t < T } Y _ { t } \right| \le \frac { R ^ { 2 } } { 5 } } \end{array}$ on $E _ { T - 1 } \cap E _ { X } \cap E _ { Y }$ . Since, in addition, $\begin{array} { r } { 2 \gamma ^ { 2 } \sum _ { t < T } \iota _ { t } \mathbb { E } [ \widetilde { m } _ { t } \mid \mathcal { G } _ { t } ] \le 2 \gamma ^ { 2 } V _ { \lambda } ( K + 1 ) \le \frac { R ^ { 2 } } { 6 0 0 } } \end{array}$ on $E _ { T - 1 }$ , we get $\begin{array} { r } { S _ { T } \le \frac { R ^ { 2 } } { 5 } + \frac { R ^ { 2 } } { 5 } + \frac { R ^ { 2 } } { 6 0 0 } < R ^ { 2 } } \end{array}$ on this event. If, moreover, $T \leq \tau ,$ , then all $t < T$ are active, and, summing up (56) over $t < T$ , we get

$$
R _ { T } ^ { 2 } + \gamma \sum _ { t < T } w _ { t } \delta _ { t } \leq R _ { 0 } ^ { 2 } + S _ { T } \leq 2 R ^ { 2 } .\tag{57}
$$

Thus, $E _ { T - 1 } \cap E _ { X } \cap E _ { Y } \subseteq E _ { T }$ , and the union bound gives $\begin{array} { r } { \mathbb { P } ( E _ { T } ) \geq 1 - \frac { T \beta } { K + 1 } } \end{array}$ , which finishes the induction.

Conclusion. Finally, we consider the event $E _ { K + 1 }$ , which has probability at least $1 - \beta .$ . If $\tau \leq K$ then min $t \le K  \delta _ { t } \le \delta _ { \tau } \le \varepsilon _ { * }$ . Otherwise, (57) with $T = K + 1$ gives $\begin{array} { r } { \sum _ { t < K } w _ { t } \delta _ { t } \le 2 R ^ { 2 } / \gamma } \end{array}$ . Let $\mathcal { T } _ { 1 } : =$ $\{ t \leq K : \ G _ { t } \leq \lambda / 2 \}$ and $\mathcal { T } _ { 2 } : = \{ 0 , \ldots , K \} \setminus \mathcal { T } _ { 1 }$ $\begin{array} { r } { \mathrm { I f ~ } | \mathcal { T } _ { 1 } | \geq \frac { K + 1 } { 2 } } \end{array}$ , then $\begin{array} { r } { \frac { K + 1 } { 2 } \operatorname* { m i n } _ { t } \delta _ { t } \leq \sum _ { t \in \mathcal { T } _ { 1 } } \delta _ { t } \leq \frac { 2 R ^ { 2 } } { \gamma } } \end{array}$ since $w _ { t } = 1$ on T . Otherwise, $\begin{array} { r } { | \mathcal { T } _ { 2 } | > \frac { K + 1 } { 2 } } \end{array}$ . For $t \in \tau _ { 2 }$ , we have min $\{ G _ { t } , \lambda \} \ge \lambda / 2$ and $G _ { t } \leq \sqrt { 2 L \delta _ { t } } ;$ hence, $\begin{array} { r } { w _ { t } \delta _ { t } \geq \frac { \lambda \delta _ { t } } { 1 2 G _ { t } } \geq \frac { \lambda \sqrt { \delta _ { t } } } { 1 2 \sqrt { 2 L } } } \end{array}$ , and

$$
\frac { K + 1 } { 2 } \cdot \frac { \lambda \sqrt { \operatorname* { m i n } _ { t } \delta _ { t } } } { 1 2 \sqrt { 2 L } } \leq \frac { 2 R ^ { 2 } } { \gamma } , \qquad \mathrm { i . e . , } \qquad \operatorname* { m i n } _ { t } \delta _ { t } \leq \frac { 4 6 0 8 L R ^ { 4 } } { \gamma ^ { 2 } \lambda ^ { 2 } ( K + 1 ) ^ { 2 } } .
$$

In all cases, we have min<sub>t</sub> $\begin{array} { r } { \le K \delta _ { t } \le \operatorname* { m a x } \{ \varepsilon _ { N } , \varepsilon _ { M } , \frac { 4 R ^ { 2 } } { \gamma ( K + 1 ) } , \frac { 4 6 0 8 L R ^ { 4 } } { \gamma ^ { 2 } \lambda ^ { 2 } ( K + 1 ) ^ { 2 } } \} } \end{array}$ , which concludes the proof. □

## H Lower Bounds for a Fixed Clipping Level: Full Statement, Comparison and Proofs

## H.1 Full Statement and Comparison with Existing Results

In Table 3, we compare Theorems 5.3, 5.4, and 5.5 with Koloskova et al. (2023) and Khah et al. (2026); see also the comparison in Section 5.3. The table separates the output criterion, the neighborhood, and the terms that vanish with the horizon. We then give the full statement of Theorem 5.5, including the growth of the expected error with the total stepsize.

Here $R \geq \| x ^ { 0 } - x ^ { * } \| , x ^ { t }$ are the iterates with $\lambda _ { k } \equiv \lambda$ , and $\bar { x } ^ { K }$ is their average. The noise is additive $( \sigma _ { 1 } = 0 )$ , with level $\sigma _ { 0 }$ and index $\alpha \in ( 1 , 2 ]$ , under Assumption 3.4 or 3.5. Unless a note states otherwise, the upper bounds hold with probability at least $1 - \beta _ { \mathrm { { \ell } } }$ , with a stepsize chosen for the horizon K. The criterion is bounded by the neighborhood plus the vanishing terms. The vanishing terms suppress constants depending on α and logarithmic factors in K and $1 / \beta$

Our neighborhood terms retain their constants: $b = b _ { \alpha } ^ { \mathrm { m o m } } \in [ 1 / 2 , 1 ]$ under Assumption 3.4, and $b = b _ { \alpha } = 2 ^ { \alpha - 1 } / ( \alpha - 1 )$ under Assumption 3.5. The constant $c _ { \perp } = 2 ^ { \alpha - 1 } ( \kappa _ { 2 } + 3 \kappa _ { 1 } )$ is defined in Section 5.2 and depends only on α.

The first row specializes Theorem 3.3 of Koloskova et al. (2023), which treats non-convex $( L _ { 0 } , L _ { 1 } )$ -smooth functions under $\begin{array} { r } { \mathbb { E } \| g ( x , \xi ) - \nabla f ( x ) \| ^ { 2 } \leq \sigma _ { 0 } ^ { 2 } . } \end{array}$ , to $L _ { 0 } ~ = ~ L$ and $L _ { 1 } ~ = ~ 0$ . It uses $\eta = \operatorname* { m i n } \{ 1 / ( 9 L ) , ( F _ { 0 } / ( L \sigma _ { 0 } ^ { 2 } K ) ) ^ { 1 / 2 } \}$ , where $F _ { 0 } : = f ( x ^ { 0 } ) - f ^ { * } \leq L R ^ { 2 } / 2$ in our setting. Their upper and lower bounds characterize the gradient-norm neighborhood min $\{ \sigma _ { 0 } , \sigma _ { 0 } ^ { 2 } / \lambda \} = \bar { b } _ { \lambda }$ up to absolute constants.

The second row uses the full non-private bound of Khah et al. (2026). With $u : = R ^ { 2 } / ( \gamma ( K + 1 ) )$ their bound is $4 u + 6 4 L u ^ { 2 } / \lambda ^ { 2 }$ . The best-iterate gap is also at most $f ( x ^ { 0 } ) - f ^ { * } \leq L R ^ { 2 } / 2$ . For $\lambda > 4 L R$

$$
\operatorname* { m i n } \left\{ { \frac { L R ^ { 2 } } { 2 } } , 4 u + { \frac { 6 4 L u ^ { 2 } } { \lambda ^ { 2 } } } \right\} \leq 8 u .
$$

Indeed, if $u \le L R ^ { 2 }$ , then $6 4 L u ^ { 2 } / \lambda ^ { 2 } \ \leq$ 4u; otherwise, the first term in the minimum sufices. Choosing γ as the minimum of their full stepsize bounds, with no privacy noise and $\zeta _ { \lambda } = 0$ , gives

$$
u = O \left( \frac { \lambda R \ell } { K + 1 } + \frac { R \sigma _ { 0 } ^ { \alpha } } { \lambda ^ { \alpha - 1 } } + R \sqrt { \frac { \lambda ^ { 2 - \alpha } \sigma _ { 0 } ^ { \alpha } \ell } { K + 1 } } \right) .
$$

This yields the displayed row, including its finite-horizon terms. In particular, the additional quadratic neighborhood term $L R ^ { 2 } \sigma _ { 0 } ^ { 2 \alpha } / \lambda ^ { 2 \alpha }$ in their unsimplified bound is absorbed by this argument.

Table 3: Bounds for Clip-SGD with a prescribed clipping level under additive noise. Our results are highlighted in light blue.
<table><tr><td>Result</td><td>Level</td><td>Criterion</td><td>Neighborhood term</td><td>Vanishing terms</td></tr><tr><td>Koloskova et al. (2023), Thm.  $3 . 3 ^ { ( 1 ) }$ </td><td> $\mathrm { a n y } \lambda > 0$ </td><td> $\mathrm { m i n } _ { t \leq K } \mathbb { E } \| \nabla f ( x ^ { t } ) \|$ </td><td> ${ \cal O } ( \operatorname* { m i n } \{ \sigma _ { 0 } , \sigma _ { 0 } ^ { 2 } / \lambda \} )$ </td><td> $\overline { { ( L F _ { 0 } \sigma _ { 0 } ^ { 2 } / K ) ^ { 1 / 4 } } }$   $+ \operatorname* { m a x } \{ 1 , \sigma _ { 0 } / \lambda \} \sqrt { L F _ { 0 } / K }$   $+ L F _ { 0 } / ( \lambda K )$ </td></tr><tr><td>Khah et al.  $\left( 2 0 2 6 \right) ^ { ( 2 ) }$ </td><td> $\lambda > 4 L R$ </td><td> $\mathrm { m i n } _ { t \le K } f ( x ^ { t } ) - f ^ { * }$ </td><td> $O ( R \sigma _ { 0 } ^ { \alpha } / \lambda ^ { \alpha - 1 } )$ </td><td> $\lambda R / K + R \sqrt { \lambda ^ { 2 - \alpha } \sigma _ { 0 } ^ { \alpha } / K }$ </td></tr><tr><td>Theorem 5.3</td><td> $\lambda \ge 2 \sqrt { 2 } L R$ </td><td> $f ( \bar { x } ^ { K } ) - f ^ { * }$ </td><td> $1 2 0 b R \sigma _ { 0 } ^ { \alpha } / \lambda ^ { \alpha - 1 }$ </td><td> $\lambda R / K + R \sqrt { \lambda ^ { 2 - \alpha } \sigma _ { 0 } ^ { \alpha } / K }$   $( L R ^ { 2 } + \lambda R ) / K$ </td></tr><tr><td>Theorem  $5 . 4 ^ { ( 3 ) }$ </td><td>any  $\lambda > 0$ </td><td> $\mathrm { m i n } _ { t \le K } f ( x ^ { t } ) - f ^ { * }$ </td><td> $4 8 \sqrt { 2 } c _ { \perp } R \times$   $\operatorname* { m i n } \{ \sigma _ { 0 } , \sigma _ { 0 } ^ { \alpha } / \lambda ^ { \alpha - 1 } \}$ </td><td> $+ R \sqrt { \lambda ^ { 2 - \alpha } \sigma _ { 0 } ^ { \alpha } / K }$   $+ \ L ^ { 3 } \dot { R } ^ { 4 } / ( \lambda ^ { 2 } \dot { K } ^ { 2 } )$ </td></tr><tr><td>Koloskova et al. (2023), Thms.  $3 . 1 \mathrm { - } 3 . 2 ^ { ( 4 ) }$ </td><td>any  $\lambda > 0$ </td><td>lower bound:  $\| \nabla f ( { \hat { x } } ) \|$  at the fixed points </td><td> $\ge \operatorname* { m i n } \{ \sigma _ { 0 } / 1 2 , \sigma _ { 0 } ^ { 2 } / ( 6 \lambda ) \}$ </td><td></td></tr><tr><td>Theorem  $\mathrm { { H . 1 ^ { ( 5 ) } } }$ </td><td>any  $\lambda > 0$ </td><td>lower bound:  $\mathbb { E } f ( \hat { x } ) - f ^ { * } ,$  ê an iterate or a</td><td> $\begin{array} { r l } { \ge } & { { } \frac { 1 } { 4 8 \cdot 4 ^ { \alpha } } \times } \end{array}$ </td><td></td></tr><tr><td></td><td></td><td>deterministic average; deterministic stepsizes</td><td> $\operatorname* { m i n } \{ R \sigma _ { 0 } ^ { \alpha } / \lambda ^ { \alpha - 1 } , R \sigma _ { 0 } , L R ^ { 2 } \}$ </td><td> $\left( \lambda _ { 0 } + \sigma _ { 0 } ^ { \alpha } \lambda _ { 0 } ^ { 1 - \alpha } \right.$ </td></tr><tr><td>Thm.  $5 . 3 ~ \mathrm { w i t h } ^ { ( 6 ) }$   $\lambda = \lambda _ { 0 } ( K + 1 ) ^ { 1 / \alpha }$ </td><td>any  $\lambda _ { 0 } > 0 \quad f ( \bar { x } ^ { K } ) - f ^ { * }$ </td><td></td><td></td><td>+ ×  $\sigma _ { 0 } ^ { \alpha / 2 } \lambda _ { 0 } ^ { 1 - \alpha / 2 } )$   $R K ^ { - ( \alpha - 1 ) / \alpha }$ </td></tr></table>

<sup>(1)</sup> Expected gradient norm for non-convex smooth functions under bounded variance $( \alpha = 2 )$  
<sup>(2)</sup> Under Assumption 3.4; simplified using the initial-point bound, as explained in the text.  
<sup>(3)</sup> Under Assumption 3.5, for $\alpha < 2 .$  
<sup>(4)</sup> Gradient-norm lower bound at fixed points of the mean clipped field under bounded variance.  
<sup>(5)</sup> Expected objective gap for every iterate and every convex average with deterministic weights; the text also gives the simultaneous probability comparison.  
<sup>(6)</sup> The level depends on the horizon and stays fixed within each run; the horizon condition is stated in the text.

Theorems 3.1–3.2 of Koloskova et al. (2023) construct, for every $\lambda > 0$ , a one-dimensional strongly convex quadratic with $L = 1$ and two-point noise of variance $\sigma _ { 0 } ^ { 2 } .$ , for which every fixed point ˆx of the mean clipped field, $\mathtt { E c l i p } ( g ( \hat { x } , \xi ) , \lambda ) = 0$ , satisfies the displayed gradient bound. This is a statement about the fixed points governing the small-stepsize limit. The finite-horizon upper bound concerns $\begin{array} { r } { \operatorname* { m i n } _ { t \leq K } \mathbb { E } \| \nabla f ( x ^ { t } ) \| } \end{array}$ . The factor R in our function-value neighborhood reflects the linear growth of the Huber objective away from its quadratic region. For $f ( x ) = L x ^ { 2 } / 2$ , a gradient norm of magnitude B corresponds to a function gap $B ^ { 2 } / ( 2 L )$

Theorem H.1 gives the expected lower bound for every iterate and every convex combination with deterministic weights, under any positive deterministic stepsize sequence. ${ \mathrm { I f ~ } } 0 < \gamma _ { k } \leq R / ( 1 6 \lambda )$ for all k, then with probability at least $3 / 4$ all iterates and all their convex averages simultaneously have error at least half the displayed bound. This also bounds the best iterate on the same event. We use this probability statement to compare with the high-probability upper bounds, whose stepsizes satisfy the required cap. Consequently, the neighborhood R min $\{ \sigma _ { 0 } , \sigma _ { 0 } ^ { \alpha } / \lambda ^ { \alpha - 1 } \}$ is unavoidable up to a factor depending only on α until it reaches the natural scale $L R ^ { 2 }$ . Above that scale, returning the initial point already gives $f ( x ^ { 0 } ) - f ^ { * } \leq L R ^ { 2 } / 2 \mathrm { . }$ ; the lower bound therefore includes the truncation by $L R ^ { 2 }$

In the final row, $\lambda = \lambda _ { 0 } ( K + 1 ) ^ { 1 / \alpha }$ grows with the horizon but remains constant during a run. The condition $K + 1 \ge ( 2 \sqrt { 2 } L R / \lambda _ { 0 } ) ^ { \alpha }$ ensures that Theorem 5.3 applies. This choice recovers the rate in Table 1.

Theorem H.1 (Lower bound for a fixed clipping level: full statement of Theorem 5.5). Let $L , R , \sigma _ { 0 } , \lambda > 0$ and $\alpha \in ( 1 , 2 ]$ , and let

$$
\varepsilon _ { \lambda } : = \frac { 1 } { 4 8 \cdot 4 ^ { \alpha } } \operatorname* { m i n } \left\{ \frac { R \sigma _ { 0 } ^ { \alpha } } { \lambda ^ { \alpha - 1 } } , R \sigma _ { 0 } , L R ^ { 2 } \right\} ,
$$

$$
s _ { \lambda } : = \frac { 1 } { 4 ^ { \alpha + 1 } } \operatorname* { m i n } \left\{ \frac { \sigma _ { 0 } ^ { \alpha } } { \lambda ^ { \alpha - 1 } } , \lambda \right\} .
$$

For<sub>of</sub> $\lambda \geq \sigma _ { 0 }$ , the first minimum is min $\{ R \sigma _ { 0 } ^ { \alpha } / \lambda ^ { \alpha - 1 } , L R ^ { 2 } \}$ and $s _ { \lambda } = \sigma _ { 0 } ^ { \alpha } / ( 4 ^ { \alpha + 1 } \lambda ^ { \alpha - 1 } )$ ; in the notation $\begin{array} { r } { ( 1 1 ) , \ \varepsilon _ { \lambda } = \ \frac { 1 } { 4 8 \cdot 4 ^ { \alpha } } \operatorname* { m i n } \{ R \bar { b } _ { \lambda } , L R ^ { 2 } \} } \end{array}$ and $s _ { \lambda } = \mathrm { m i n } \{ \sigma _ { 0 } , \lambda \} ^ { \alpha } / ( 4 ^ { \alpha + 1 } \lambda ^ { \alpha - 1 } )$ . There exist a convex L-smooth $f : \mathbb { R } \to \mathbb { R }$ with minimizer $x ^ { * } = 0$ , a point $x ^ { 0 }$ with $| x ^ { 0 } - x ^ { * } | = R$ and an oracle $g ( x , \xi ) = f ^ { \prime } ( x ) + \zeta$ with $\mathbb { E } \zeta = 0$ and $\mathbb { E } | \zeta | ^ { \alpha } \leq \sigma _ { 0 } ^ { \alpha } \mathrm { ~ ( i . e . ~ }$ , Assumptions 3.1–3.5 hold on R with $\sigma _ { 1 } = 0 )$ such that Clip-SGD with $\lambda _ { k } \equiv \lambda$ has the following properties.

1. For any deterministic stepsizes $\gamma _ { k } > 0$ (used instead of $\gamma$ at iteration k) and all $k , K \in$ $\{ 0 , 1 , \ldots \}$

$$
\mathbb { E } f ( x ^ { k } ) - f ^ { * } \geq \varepsilon _ { \lambda } \left( 1 + \frac { s _ { \lambda } } { R } \sum _ { j < k } \gamma _ { j } \right) ,
$$

and $\mathbb { E } f ( \hat { x } ) - f ^ { * } \geq \varepsilon _ { \lambda }$ for every convex combination ˆx of $x ^ { 0 } , \ldots , x ^ { K }$ with deterministic weights. For a constant stepsize $\gamma ,$ moreover, $\begin{array} { r } { \mathbb { E } f ( \bar { x } ^ { K } ) - f ^ { * } \geq \varepsilon _ { \lambda } \left( 1 + \frac { s _ { \lambda } \gamma K } { 2 R } \right) } \end{array}$

2. For any deterministic stepsizes $\begin{array} { r } { 0 < \gamma _ { k } \le \frac { R } { 1 6 \lambda } } \end{array}$ , with probability at least $3 / 4 , f ( x ^ { k } ) { - } f ^ { * } \geq \varepsilon _ { \lambda } / 2$ for all $k \in \{ 0 , 1 , \ldots \}$ and $f ( \bar { x } ^ { K } ) - f ^ { * } \geq \bar { \varepsilon } _ { \lambda } / 2$ for all $K \in \{ 0 , 1 , \ldots \}$

Two further consequences. First, due to the linear growth of the expected error (part 1 of Theorem H.1), to keep the expected error of the last iterate, or of the averaged iterate with a constant stepsize, of the order $\varepsilon _ { \lambda }$ , one needs the total stepsize to be ${ O _ { \alpha } ( R \lambda ^ { \alpha - 1 } / \sigma _ { 0 } ^ { \alpha } ) }$ when $\sigma _ { 0 } \le \lambda$ (for the averaged iterate under a general deterministic schedule, the proof controls the weighted sum $\begin{array} { r } { \frac { 1 } { K + 1 } \sum _ { j < K } ( K - j ) \gamma _ { j } } \end{array}$ instead of the total stepsize, since a late step moves only a few of the averaged iterates), which is exactly the scale of the bias condition in Theorem 5.3 (there, the bias threshold for the stepsize is $R \lambda ^ { \alpha - 1 } / ( 6 0 b \sigma _ { 0 } ^ { \alpha } ( K + 1 ) )$ for $\sigma _ { 1 } = 0$ , and a total stepsize $\gamma ( K + 1 )$ much larger than $R \lambda ^ { \alpha - 1 } / ( 6 0 b \sigma _ { 0 } ^ { \alpha } )$ would increase the error); the best iterate in Theorem 5.4 does not require this condition, and part 2 of Theorem H.1 covers its stepsizes. Second, the bias is not caused by the linear growth of the Huber function: for $\begin{array} { r } { f ( x ) = \frac { L } { 2 } x ^ { 2 } } \end{array}$ and the noise from the first construction, Clip-SGD with $\lambda _ { k } \equiv \lambda$ and any stepsizes with $\textstyle \sum _ { k } \gamma _ { k } =$ ∞ and $\textstyle \sum _ { k } \gamma _ { k } ^ { 2 } <$ ∞ converges almost surely to a point $x _ { \lambda } \neq x ^ { * }$ with $f ( x _ { \lambda } ) - f ^ { * } \ge \operatorname * { m i n } \{ \sigma _ { 0 } ^ { 2 \tilde { \alpha } } / \lambda ^ { 2 ( \alpha - 1 ) } , \lambda ^ { 2 } \} / ( \tilde { 8 } \cdot \tilde { 1 } 6 ^ { \alpha } L )$ (see Proposition H.4 in Section H.4).

## H.2 Proof of Theorem H.1

Proof. Huber functions. For $s > 0$ , let

$$
f _ { s } ( x ) : = { \left\{ \begin{array} { l l } { { \frac { L } { 2 } } x ^ { 2 } , } & { | x | \leq s / L , } \\ { s | x | - { \frac { s ^ { 2 } } { 2 L } } , } & { | x | > s / L . } \end{array} \right. }
$$

Then $f _ { s }$ is convex and L-smooth on R, and $f _ { s } ^ { \prime } ( x ) = \mathrm { m a x } \{ - s , \mathrm { m i n } \{ L x , s \} \}$ , implying $| f _ { s } ^ { \prime } | \le s$ Moreover, $x ^ { * } = 0$ is the unique minimizer, $f ^ { * } = 0$ , and $f _ { s }$ is even and non-decreasing in |x|. Assumption 3.2 holds on R: $( f _ { s } ^ { \prime } ( x ) ) ^ { 2 } = 2 L f _ { s } ( x )$ for $| x | \leq s / L$ , and $s ^ { 2 } \leq 2 L s | x | - s ^ { 2 } = 2 L f _ { s } ( x )$ otherwise. Moreover, for every $r \geq 0$ , we have

$$
f _ { s } ( - r ) = f _ { s } ( r ) \geq \operatorname* { m i n } \left\{ \frac { s r } { 2 } , \frac { L r ^ { 2 } } { 2 } \right\} ,\tag{58}
$$

since $\begin{array} { r } { s r - \frac { s ^ { 2 } } { 2 L } \geq \frac { s r } { 2 } } \end{array}$ for $r \geq s / L$ . We use two instances depending on λ. In both of them, $f = f _ { s }$ for a suitable s, and the noise satisfies $\mathbb { E } \zeta = 0$ and $\mathbb { E } | \zeta | ^ { \alpha } \leq \sigma _ { 0 } ^ { \alpha }$ . Thus, Assumption 3.4 holds on R with $( \sigma _ { 0 } , 0 )$ , and so does Assumption 3.5 by Proposition C.5. The iterates use independent copies $\zeta _ { k }$ of ζ.

Instance $A { : } ~ \lambda \geq \sigma _ { 0 } / ( 2 \cdot 3 ^ { 1 / \alpha } ) , ~ x ^ { 0 } = - R . ~ \mathrm { L e t } ~ \sigma _ { \lambda } : = \operatorname* { m i n } \{ \sigma _ { 0 } , \lambda \}$ and

$$
p : = \frac { 1 } { 2 } \left( \frac { \sigma _ { \lambda } } { 4 \lambda } \right) ^ { \alpha } , \qquad \nu : = \frac { 2 p \lambda } { 1 - p } , \qquad s : = \frac { p \lambda } { 2 } .
$$

Since $\sigma _ { \lambda } \leq \lambda$ and $\alpha > 1$ , we have $\begin{array} { r } { p \leq \frac { 1 } { 8 } } \end{array}$ . Let ζ be equal to ν with probability $1 - p$ and to −2λ with probability p. Then $\mathbb { E } \zeta = ( 1 - p ) \nu ^ { \top } 2 p \lambda = 0$ and $p ( 2 \lambda ) ^ { \alpha } = \sigma _ { \lambda } ^ { \alpha } / 2 ^ { \alpha + 1 } \leq \sigma _ { \lambda } ^ { \alpha } / 4$ . Since $\begin{array} { r } { 1 - p \ge \frac { 7 } { 8 } } \end{array}$ 2 $4 ^ { \alpha } \geq 4$ , and $\sigma _ { \lambda } \leq \lambda$ , we have

$$
\nu \leq \frac { 1 6 } { 7 } p \lambda = \frac { 8 } { 7 } \cdot \frac { \sigma _ { \lambda } } { 4 ^ { \alpha } } \left( \frac { \sigma _ { \lambda } } { \lambda } \right) ^ { \alpha - 1 } \leq \frac { 2 } { 7 } \sigma _ { \lambda } ,
$$

hence, $\begin{array} { r } { \mathbb { E } | \zeta | ^ { \alpha } = ( 1 - p ) \nu ^ { \alpha } + p ( 2 \lambda ) ^ { \alpha } \leq \left( \frac { 2 } { 7 } + \frac { 1 } { 4 } \right) \sigma _ { \lambda } ^ { \alpha } \leq \sigma _ { 0 } ^ { \alpha } } \end{array}$ . Since $s \leq \lambda / 1 6$ and $\nu \leq 2 \lambda / 7 .$ , for every $x \in \mathbb { R }$ we have $f ^ { \prime } ( x ) + \nu \in ( - \lambda , \lambda )$ and $f ^ { \prime } ( x ) - 2 \lambda < - \lambda$ . Thus, $\mathsf { c l i p } ( f ^ { \prime } ( x ) + \zeta , \lambda )$ equals $f ^ { \prime } ( x ) + \nu$ with probability $1 - p$ and −λ with probability p. Using $( 1 - p ) \nu = 2 p \lambda$ and $| f ^ { \prime } ( x ) | \leq s$ , we get

$$
h ( x ) : = \mathbb { E } \mathrm { c } \mathrm { 1 } \mathtt { i p } ( f ^ { \prime } ( x ) + \zeta , \lambda ) = ( 1 - p ) ( f ^ { \prime } ( x ) + \nu ) - p \lambda = ( 1 - p ) f ^ { \prime } ( x ) + p \lambda \geq p \lambda - s = s .\tag{59}
$$

Moreover, the variance of cl $\begin{array} { r } { \cdot \mathtt { i p } ( f ^ { \prime } ( x ) + \zeta , \lambda ) \mathrm { ~ i s ~ } p ( 1 - p ) ( f ^ { \prime } ( x ) + \nu + \lambda ) ^ { 2 } \le p \left( \frac { 1 } { 1 6 } + \frac { 2 } { 7 } + 1 \right) ^ { 2 } \lambda ^ { 2 } \le 2 p \lambda ^ { 2 } } \end{array}$ Finally, $\begin{array} { r } { s = \frac { \sigma _ { \lambda } ^ { \alpha } } { 4 ^ { \alpha + 1 } \lambda ^ { \alpha - 1 } } = s _ { \lambda } } \end{array}$ and $\begin{array} { r } { \frac { s R } { 2 } ~ = ~ \frac { R \sigma _ { \lambda } ^ { \alpha } } { 8 \cdot 4 ^ { \alpha } \lambda ^ { \alpha - 1 } } } \end{array}$ . If $\lambda \ge \sigma _ { 0 }$ , then this equals $\frac { R \bar { b } _ { \lambda } } { 8 { \cdot } 4 ^ { \alpha } }$ . Otherwise, $\sigma _ { \lambda } = \lambda \ge \sigma _ { 0 } / 6$ and $\bar { b } _ { \lambda } = \sigma _ { 0 }$ , and, therefore, it is at least $\frac { R \bar { b } _ { \lambda } } { 4 8 { \cdot } 4 ^ { \alpha } }$ . In both cases, $\begin{array} { r } { \frac { s R } { 2 } \geq \frac { R \bar { b } _ { \lambda } } { 4 8 \cdot 4 ^ { \alpha } } } \end{array}$

Instance $B \colon \lambda < \sigma _ { 0 } / ( 2 \cdot 3 ^ { 1 / \alpha } ) , \ x ^ { 0 } = R$ . Let $a : = \sigma _ { 0 } / 3 ^ { 1 / \alpha }$ (meaning that $\lambda < a / 2 )$ and $s : =$ $a - \lambda > a / 2$ . Let ζ be equal to $- a$ with probability $3 / 4$ and to 3a with probability $1 / 4$ . Then $\mathbb { E } \zeta = 0$ and $\begin{array} { r } { \mathbb { E } | \zeta | ^ { \alpha } = \frac { 3 + 3 ^ { \alpha } } { 4 } a ^ { \alpha } \le 3 a ^ { \alpha } = \sigma _ { 0 } ^ { \alpha } } \end{array}$ . For every $x \in \mathbb { R }$ , we have $f ^ { \prime } ( x ) - a \leq s - a = - \lambda$ and $f ^ { \prime } ( x ) + 3 a \geq 3 a - \bar { s = } 2 a + \lambda \geq \lambda$ . Thus, the clipped oracle equals −λ with probability $3 / 4$ and λ with probability $1 / 4$ , independently of $x ,$ and $x ^ { k + 1 } = x ^ { k } + \gamma _ { k } \lambda \chi _ { k }$ with i.i.d. $\chi _ { k } \in \{ - 1 , 1 \}$ $\begin{array} { r } { \mathbb { P } \{ \chi _ { k } = 1 \} = \frac { 3 } { 4 } } \end{array}$ . In particular, $\mathbb { E } [ x ^ { k + 1 } \mid { \mathcal { G } } _ { k } ] = x ^ { k } + \gamma _ { k } \lambda / 2$ , and $\lambda / 2 \geq \lambda / 4 ^ { \alpha + 1 } = s _ { \lambda }$ . Finally, $\bar { b } _ { \lambda } = \sigma _ { 0 }$ and $\begin{array} { r } { \frac { s R } { 2 } \geq \frac { a R } { 4 } \geq \frac { R \sigma _ { 0 } } { 1 2 } \geq \frac { R \bar { b } _ { \lambda } } { 4 8 \cdot 4 ^ { \alpha } } } \end{array}$

Value of the bound. In both instances, $\begin{array} { r } { \frac { s R } { 2 } \geq \frac { R \bar { b } _ { \lambda } } { 4 8 \cdot 4 ^ { \alpha } } } \end{array}$ and $\begin{array} { r } { \frac { L R ^ { 2 } } { 2 } \geq \frac { L R ^ { 2 } } { 4 8 \cdot 4 ^ { \alpha } } } \end{array}$ , and, therefore, (58) gives $f ( r ) \geq \varepsilon _ { \lambda }$ for every $r \geq R ,$ , and $\begin{array} { r } { f ( R / 2 ) \geq \operatorname* { m i n } \{ \frac { s R } { 4 } , \frac { L R ^ { 2 } } { 8 } \} \geq \frac { \varepsilon _ { \lambda } } { 2 } } \end{array}$

Part 1. Let $\textstyle \Gamma _ { k } : = \sum _ { j < k } \gamma _ { j }$ . Since $| x ^ { k + 1 } - x ^ { k } | \leq \gamma _ { k } \lambda$ , for each fixed k we have $| x ^ { k } | \le | x ^ { 0 } | + \lambda \Gamma _ { k }$ thus, each $x ^ { k }$ is bounded by a deterministic constant and, therefore, integrable, and the same holds for every finite convex combination of the iterates. In instance $\mathrm { A } , ( 5 9 )$ gives $\mathbb { E } [ x ^ { k + 1 } \mid { \mathcal G } _ { k } ] \le x ^ { k } - \gamma _ { k } s ;$ hence, $\mathbb { E } x ^ { k } \le - R - s _ { \lambda } \Gamma _ { k }$ . In instance B, we have $\mathbb { E } x ^ { k } = R + \lambda \Gamma _ { k } / 2 \ge R + s _ { \lambda } \Gamma _ { k }$ . In both cases, $| \mathbb { E } x ^ { k } | \geq R + s _ { \lambda } \Gamma _ { k }$ with $\mathbb { E } x ^ { k }$ on the side of $x ^ { 0 }$ . Thus, $| \mathbb { E } \hat { x } | \geq R$ for every convex combination $\hat { x }$ of the iterates with deterministic weights, and $| \mathbb { E } \bar { x } ^ { K } | \ge R + s _ { \lambda } \gamma K / 2$ for a constant stepsize $\gamma ,$ since $\begin{array} { r } { \frac { 1 } { K + 1 } \sum _ { k = 0 } ^ { K } \Gamma _ { k } = \gamma K / 2 } \end{array}$ . Since f is convex, even, and non-decreasing in |x|, Jensen’s inequality gives $\mathbb { E } f ( \hat { x } ) \geq f ( R ) \geq \varepsilon _ { \lambda } , \mathbb { E } f ( x ^ { k } ) \geq f ( R + s _ { \lambda } \Gamma _ { k } )$ , and, for a constant stepsize, $\mathbb { E } f ( \bar { x } ^ { K } ) \ge f ( R + s _ { \lambda } \gamma K / 2 )$ By (58) with $r = R ( 1 + s _ { \lambda } \Gamma _ { k } / R )$ (and, similarly, with $r = R + s _ { \lambda } \gamma K / 2 )$ , we get

$$
f ( R + s _ { \lambda } \Gamma _ { k } ) \geq \operatorname* { m i n } \left\{ \frac { s R } { 2 } , \frac { L R ^ { 2 } } { 2 } \right\} \left( 1 + \frac { s _ { \lambda } \Gamma _ { k } } { R } \right) \geq \varepsilon _ { \lambda } \left( 1 + \frac { s _ { \lambda } \Gamma _ { k } } { R } \right) .
$$

Part 2. Let $\bar { \gamma } : = \operatorname* { s u p } _ { k } \gamma _ { k } \leq R / ( 1 6 \lambda )$ . We show that, with probability at least $\begin{array} { r } { 1 - e ^ { - 2 } \ge \frac { 3 } { 4 } } \end{array}$ , all iterates satisfy $| x ^ { k } | \geq R / 2$ and belong to the side of $x ^ { 0 }$ . Then all averages $\bar { x } ^ { K }$ belong to it as well, and $f ( x ^ { k } ) - f ^ { * } \geq f ( R / 2 ) \geq \varepsilon _ { \lambda } / 2$ and $f ( \bar { x } ^ { K } ) - f ^ { * } \geq \varepsilon _ { \lambda } / 2$ In instance A, let $\begin{array} { r } { e _ { k } : = \mathsf { c l i p } ( f ^ { \prime } ( x ^ { k } ) + \zeta _ { k } , \lambda ) - h ( x ^ { k } ) } \end{array}$ and $\begin{array} { r } { D _ { k } : = \sum _ { j < k } \gamma _ { j } e _ { j } } \end{array}$ . Then $x ^ { k } =$ $\begin{array} { r } { - R - \sum _ { i < k } \gamma _ { j } h ( x ^ { j } ) - D _ { k } \leq - R - s \Gamma _ { k } - D _ { k } } \end{array}$ . Let $\vartheta : = 1 / ( 4 \bar { \gamma } \lambda )$ . Since $\begin{array} { r } { | \vartheta \gamma _ { j } e _ { j } | \le \frac { 1 } { 2 } } \end{array}$ and $e ^ { v } \leq$ $1 + v + v ^ { 2 }$ for $| v | \leq 1$ , we have $\begin{array} { r } { \mathbb { E } [ e ^ { - \vartheta \gamma _ { j } e _ { j } } \ \vert \ \mathcal { G } _ { j } ] \le 1 + \vartheta ^ { 2 } \gamma _ { j } ^ { 2 } \mathbb { E } [ e _ { j } ^ { 2 } \ \vert \ \mathcal { G } _ { j } ] \le \exp ( 2 p \vartheta ^ { 2 } \gamma _ { j } ^ { 2 } \lambda ^ { 2 } ) } \end{array}$ . Thus, $\begin{array} { r } { M _ { k } : = \exp \left( - \vartheta D _ { k } - 2 p \vartheta ^ { 2 } \lambda ^ { 2 } \sum _ { j < k } \gamma _ { j } ^ { 2 } \right) } \end{array}$ is a non-negative supermartingale with $M _ { 0 } = 1$ . On the event $\{ x ^ { k } > - R / 2 \}$ , we hav $\mathrm { ~ e ~ } { - } D _ { k } > \frac { R } { 2 } { + } s \Gamma _ { k }$ . Since $\begin{array} { r } { \sum _ { j < k } \gamma _ { j } ^ { 2 } \le \bar { \gamma } \Gamma _ { k } } \end{array}$ and $2 p \vartheta \lambda ^ { 2 } \bar { \gamma } = \textstyle { \frac { p \lambda } { 2 } } = s$ , on this event we have

$$
M _ { k } \ge \exp \left( \frac { \vartheta R } { 2 } + \vartheta \Gamma _ { k } \left( s - 2 p \vartheta \lambda ^ { 2 } \bar { \gamma } \right) \right) = e ^ { \vartheta R / 2 } .
$$

By Ville’s maximal inequality, $\begin{array} { r } { \mathbb { P } \{ \exists k : \ x ^ { k } > - R / 2 \} \leq e ^ { - \vartheta R / 2 } = e ^ { - R / ( 8 \bar { \gamma } \lambda ) } \leq e ^ { - 2 } . } \end{array}$

In instance B, let $\kappa : = \ln 3 / ( \bar { \gamma } \lambda )$ . The function $\begin{array} { r } { \varphi ( v ) : = \frac { 3 } { 4 } e ^ { - v } + \frac { 1 } { 4 } e ^ { v } } \end{array}$ is convex with $\varphi ( 0 ) =$ $\varphi ( \ln 3 ) = 1$ ; hence, $\mathbb { E } e ^ { - \kappa \gamma _ { k } \lambda \chi _ { k } } = \varphi ( \kappa \gamma _ { k } \lambda ) \le 1$ . Therefore, ${ \tilde { N _ { k } } } : = \mathrm { e x p } { \bar { ( - \kappa ( x ^ { k } - R / 2 ) ) } }$ is a nonnegative supermartingale, and $\mathbb { P } \{ \exists k : \ x ^ { k } < R / 2 \} \le \mathbb { P } \{ \operatorname* { s u p } _ { k } N _ { k } > 1 \} \le N _ { 0 } = 3 ^ { - R / ( 2 \bar { \gamma } \dot { \lambda } ) } \le 3 ^ { - 8 } \le$ $e ^ { - \bar { 2 } }$ , which concludes the proof. □

## H.3 Optimality of the Stochastic Term

Proposition H.2 (Lower bound for the stochastic term). Let $L , R , \sigma _ { 0 } > 0 , \alpha \in ( 1 , 2 ] , K \in$ $\{ 0 , 1 , \ldots \}$ , and

$$
s : = \operatorname* { m i n } \left\{ \frac { \sigma _ { 0 } } { 2 ( K + 1 ) ^ { ( \alpha - 1 ) / \alpha } } , L R \right\} .
$$

Consider any algorithm that makes at most $K + 1$ adaptive, possibly randomized, queries to a stochastic gradient oracle and outputs a point $\hat { x } \in \mathbb { R } ;$ ; the algorithm accesses the unknown instance only via this oracle, without access to exact function values, exact gradients, or an analytical description of the instance. Then there exist two instances $v \in \{ - 1 , + 1 \}$ of problem (1) on $\mathbb { R }$ , with $f _ { v }$ convex and L-smooth on $\mathbb { R } .$ , minimizer $x _ { v } ^ { * } = v R , \ f _ { v } ^ { * } = 0$ common starting point $x ^ { 0 } = 0 ( { \mathrm { i . e . , ~ } } | x ^ { 0 } - x _ { v } ^ { * } | = R )$ , and unbiased oracles $\nabla f _ { v , \xi }$ satisfying Assumption 3.4, and, therefore, Assumption 3.5, on R with the parameters $( \sigma _ { 0 } , 0 )$ such that, for at least one v,

$$
\mathbb { P } \left\{ f _ { v } ( \hat { x } ) - f _ { v } ^ { * } \geq \frac { s R } { 2 } \right\} \geq \frac { 1 } { 4 } , \qquad \mathrm { w h e r e } \qquad \frac { s R } { 2 } = \frac { 1 } { 4 } \operatorname* { m i n } \left\{ \frac { \sigma _ { 0 } R } { ( K + 1 ) ^ { ( \alpha - 1 ) / \alpha } } , ~ 2 L R ^ { 2 } \right\} .
$$

Proof. Let $\textstyle p : = { \frac { 1 } { 2 ( K + 1 ) } }$ and let $f _ { s }$ be the Huber function from the proof of Theorem H.1, with $f _ { s } ^ { \prime } ( y ) = \operatorname* { m a x } \{ - s , \operatorname* { m i n } \{ L y , s \} \}$ . For $v \in \{ - 1 , + 1 \}$ , let $f _ { v } ( x ) : = f _ { s } ( x - v R )$ , which is convex and L-smooth on R, has the unique minimizer vR with $f _ { v } ^ { * } = 0 ,$ , and satisfies $| f _ { v } ^ { \prime } | \le s$ . We write $f _ { v } ^ { \prime } = m +$ vδ with m $: = \textstyle { \frac { 1 } { 2 } } ( f _ { + } ^ { \prime } + f _ { - } ^ { \prime } ) \quad$ and $\begin{array} { r } { \delta : = \frac { 1 } { 2 } ( f _ { + } ^ { \prime } - f _ { - } ^ { \prime } ) } \end{array}$ , where $| \delta | \le s$ , and, similarly, $f _ { v } = M + v D$ with $M : = { \textstyle \frac { 1 } { 2 } } ( f _ { + } + f _ { - } )$ and $\begin{array} { r } { D : = \frac { 1 } { 2 } ( f _ { + } - f _ { - } ) } \end{array}$ , where $M ^ { \prime } = m$ and $D ^ { \prime } = \delta$ . Let $\xi = U \sim \mathrm { U n i f o r m } ( 0 , 1 )$ and

$$
f _ { v , U } ( x ) : = M ( x ) + v D ( x ) \frac { \mathbf { 1 } _ { \{ U \leq p \} } } { p } , \qquad \nabla f _ { v , U } ( x ) = m ( x ) + v \delta ( x ) \frac { \mathbf { 1 } _ { \{ U \leq p \} } } { p } .
$$

Then $\mathbb { E } _ { U } f _ { v , U } = f _ { v }$ and $\mathbb { E } _ { U } \nabla f _ { v , U } ( x ) = f _ { v } ^ { \prime } ( x )$ . The noise $\begin{array} { r } { \nabla f _ { v , U } ( x ) - f _ { v } ^ { \prime } ( x ) = v \delta ( x ) \big ( \frac { \mathbf { 1 } _ { \{ U \leq p \} } } { p } - 1 \big ) } \end{array}$ equals $- v \delta ( x )$ with probability $1 - p$ and $\begin{array} { r } { v \delta ( x ) \frac { 1 - p } { p } } \end{array}$ with probability $p ;$ hence,

$$
\begin{array} { r l } & { \mathbb { E } _ { U } \left| \nabla f _ { v , U } ( x ) - f _ { v } ^ { \prime } ( x ) \right| ^ { \alpha } = | \delta ( x ) | ^ { \alpha } \left[ ( 1 - p ) + p \left( \displaystyle \frac { 1 - p } { p } \right) ^ { \alpha } \right] \leq | \delta ( x ) | ^ { \alpha } \left( 1 + p ^ { 1 - \alpha } \right) } \\ & { \qquad \leq 2 s ^ { \alpha } p ^ { 1 - \alpha } = 2 s ^ { \alpha } \big ( 2 ( K + 1 ) \big ) ^ { \alpha - 1 } \leq \sigma _ { 0 } ^ { \alpha } , } \end{array}
$$

where $1 + p ^ { 1 - \alpha } \leq 2 p ^ { 1 - \alpha }$ since $p < 1 < \alpha ,$ , and the last step follows from the choice of $s .$ Therefore, both instances satisfy Assumption 3.4 on R with $( \sigma _ { 0 } , 0 )$ , and Assumption 3.5 by Markov’s inequality.

Next, we run the algorithm on both instances with the same internal randomness and the same i.i.d. uniforms $U _ { 0 } , \dots , U _ { K }$ (one per query). On the event $E : = \{ U _ { k } > p$ for all $k \leq K \}$ , every query at any point x returns $m ( x )$ under both instances, meaning that the two runs produce the same query points and the same output ˆx. By Bernoulli’s inequality, $\mathbb { P } ( E ) = ( 1 - p ) ^ { K + 1 } \geq 1 - ( K + 1 ) p =$ $\textstyle { \frac { 1 } { 2 } }$ . Since $\begin{array} { r } { \mathbb { P } ( E \cap \{ \hat { x } \leq 0 \} ) + \mathbb { P } ( E \cap \{ \hat { x } > 0 \} ) = \mathbb { P } ( E ) \geq \frac { 1 } { 2 } } \end{array}$ , one of the two probabilities is at least $\textstyle { \frac { 1 } { 4 } }$ If $\mathbb { P } ( E \cap \{ \hat { x } \leq 0 \} ) \geq { \frac { 1 } { 4 } }$ , we take $v = + 1 \colon$ for $x \leq 0$ , we have $x - R \leq - R \leq - s / L$ , since $s \leq L R ,$ and, therefore, $\begin{array} { r } { f _ { + } ( x ) \dot { = } s ( R - x ) - \frac { s ^ { 2 } } { 2 L } \geq s R - \frac { s ^ { 2 } } { 2 L } \geq \frac { s R } { 2 } } \end{array}$ . Otherwise, $\mathbb { P } ( E \cap \{ \hat { x } > 0 \} ) \geq { \frac { 1 } { 4 } }$ , and, for $v = - 1$ and $x > 0$ , we have $\begin{array} { r } { f _ { - } ( x ) = s ( x + R ) - \frac { s ^ { 2 } } { 2 L } \geq \frac { s R } { 2 } } \end{array}$ . In both cases, the event in question is determined by the coupled run, which coincides with the actual run of instance v on $E ,$ , which concludes the proof. □

Proposition H.2 establishes the optimal polynomial dependence of the additive stochastic term $\Delta _ { 0 }$ from Corollary F.2 on $K , \sigma _ { 0 }$ , and R in the regime $\sigma _ { 0 } R ( K + 1 ) ^ { - ( \alpha - 1 ) / \alpha } \stackrel { < } { \sim } L R ^ { 2 }$ , at a fixed confidence level with failure probability below $1 / 4 .$ , within the class of algorithms using $K + 1$ oracle calls; Clip-SGD with $K + 1$ iterations belongs to this class. For the moment class, the upper and lower bounds match up to constants and the factor $\ell ^ { ( \alpha - 1 ) / \alpha }$ ; at the weak- $L ^ { 2 }$ endpoint, the upper bound also contains the tail-dependent logarithmic factor $c _ { 2 } ( \Lambda _ { K } ) ^ { 1 / 2 }$ . It is worth mentioning that the proposition itself does not establish the necessity of these logarithms, the optimal dependence on the failure probability, or the optimality of the multiplicative term; the dependence on the failure probability is studied in Proposition H.3 below, the necessity of the logarithm at $\alpha = 2$ at fixed confidence is shown in Proposition I.16, and nothing is claimed for the multiplicative term. The hard instances have bounded gradients, $| f _ { v } ^ { \prime } | \le s ,$ , and state-dependent noise (its law depends on x through $\delta ( x )$ , which is allowed by Assumption $3 . 4 )$ ; therefore, the lower bound also holds within the subclass of problems with bounded gradients. That is, the proposition concerns a uniformly bounded noise scale $( \sigma _ { 1 } = 0 )$ rather than state-independent additive noise; in contrast, the Lorentz lower bound from Proposition I.16 (see Appendix I.8) is realized by state-independent noise. Related lower bounds are classical in expectation (Nemirovski and Yudin, 1983; Vural et al., 2022), and Liu (2026a) recently proves high-probability lower bounds for non-smooth convex problems under bounded $\alpha \mathrm { - }$ th moments, of the order $( \sigma _ { s } ^ { 2 / \alpha - 1 } \sigma _ { l } ^ { 2 - 2 / \alpha } + \sigma _ { s } \ln ^ { 1 - 1 / \alpha } ( 1 / \delta ) ) D / T ^ { 1 - 1 / \alpha }$ for $T$ oracle calls and failure probability $\delta ,$ where $\sigma _ { s }$ and $\sigma _ { l }$ bound the directional and the full α-th moment of the noise and D is the initial distance; in dimension one, where $\sigma _ { s } = \sigma _ { l }$ , this coincides with the dependence on $K , \ \sigma _ { 0 } , \ R .$ and $\beta$ in Propositions H.2 and H.3, which are derived for the smooth unconstrained problem and have explicit constants, while the dimension-dependent improvements via $\sigma _ { s }$ have no counterpart in our setting.

The same construction with a confidence-dependent revelation probability gives the dependence on the confidence level.

Proposition H.3 (Dependence on the failure probability). Let $L , R , \sigma _ { 0 } > 0 , \alpha \in ( 1 , 2 ] , \beta \in$ $\begin{array} { r } { ( 0 , 1 / 8 ] , \ell _ { \beta } : = \ln \frac { 1 } { 4 \beta } } \end{array}$ , and let $K \in \{ 0 , 1 , \ldots \}$ satisfy $K + 1 \ge \ell _ { \beta }$ For every algorithm as in Proposition H.2, there exist two instances $v \in \{ - 1 , + 1 \}$ with the properties listed there (Assumption 3.4 with $( \sigma _ { 0 } , 0 )$ on R, $| x ^ { 0 } - x _ { v } ^ { * } | = R )$ such that, for at least one $v ,$

$$
\mathbb { P } \left\{ f _ { v } ( \hat { x } ) - f _ { v } ^ { * } \geq \frac { 1 } { 4 } \operatorname* { m i n } \left\{ L R ^ { 2 } , \sigma _ { 0 } R \left( \frac { \ell _ { \beta } } { K + 1 } \right) ^ { \frac { \alpha - 1 } { \alpha } } \right\} \right\} > 2 \beta .
$$

That is, a guarantee holding with probability at least $1 - \beta ^ { \prime }$ for some $\beta ^ { \prime } \leq 1 / 4$ and $\begin{array} { r } { K + 1 \geq \ln { \frac { 1 } { 2 \beta ^ { \prime } } } } \end{array}$ must allow an error of at least $\textstyle { \frac { 1 } { 4 } }$ min $\{ L R ^ { 2 } , \sigma _ { 0 } R ( \ln { \frac { 1 } { 2 \beta ^ { \prime } } } / ( K + 1 ) ) ^ { ( \alpha - 1 ) / \alpha } \}$ on one of these instances; in other words, the factor $( \ln ( 1 / \beta ) / K ) ^ { ( \alpha - 1 ) / \alpha }$ in the additive term in (91) cannot be improved in this range.

Proof. We repeat the proof of Proposition H.2 with the parameters $\begin{array} { r } { p : = \frac { \ell _ { \beta } } { 2 ( K + 1 ) } \le \frac { 1 } { 2 } } \end{array}$ and $s : =$ min $\{ L R , \sigma _ { 0 } ( p ^ { \alpha - 1 } / 2 ) ^ { 1 / \alpha } \}$ ; then $2 s ^ { \alpha } p ^ { 1 - \alpha } \ \leq \ \sigma _ { 0 } ^ { \alpha }$ and both instances satisfy Assumption 3.4 with $( \sigma _ { 0 } , 0 )$ . On the event $E = \{ U _ { k } > p \forall k \leq K \}$ , the two coupled runs coincide, and $\mathbb { P } ( E ) =$ $( 1 - p ) ^ { K + 1 } > e ^ { - 2 p ( K + 1 ) } = e ^ { - \ell _ { \beta } } = 4 \beta _ { \mathrm { { \scriptscriptstyle { \cdot } } } }$ , since $\ln ( 1 - p ) + 2 p$ vanishes at 0 and is increasing on $( 0 , \textstyle { \frac { 1 } { 2 } } )$ . Thus, $\mathbb { P } ( E \cap \{ \hat { x } \leq 0 \} ) > 2 \beta$ or $\mathbb { P } ( E \cap \{ \hat { x } > 0 \} ) > 2 \beta$ , and, on the respective instance, the error is at least $\frac { s R } { 2 }$ , as before. Finally, $\begin{array} { r } { \frac { s R } { 2 } \geq \frac { R } { 2 } \cdot 2 ^ { - \frac { 1 } { \alpha } - \frac { \alpha - 1 } { \alpha } } } \end{array}$ min $\{ L R , \sigma _ { 0 } ( \ell _ { \beta } / ( K + 1 ) ) ^ { ( \alpha - 1 ) / \alpha } \} =$ ${ \textstyle \frac { 1 } { 4 } } \operatorname* { m i n } \{ L R ^ { 2 } , \sigma _ { 0 } R ( \ell _ { \beta } / ( { \bar { K } } + 1 ) ) ^ { ( \alpha - 1 ) / \alpha } \}$ . The consequence follows with $\beta = \beta ^ { \prime } / 2$ , which concludes the proof. □

## H.4 Strongly Convex Quadratics

Proposition H.4 (Bias under strong convexity). Let $L , \sigma _ { 0 } , \lambda > 0 , \alpha \in ( 1 , 2 ]$ , and let $\zeta , \sigma _ { \lambda } , p ,$ and $\nu$ be defined as in instance A from the proof of Theorem H.1 (these definitions make sense for every $\lambda > 0 )$ . Consider $\begin{array} { r } { f ( x ) = \frac { L } { 2 } x ^ { 2 } } \end{array}$ on R with the oracle $g ( x , \xi ) = f ^ { \prime } ( x ) + \zeta ,$ which satisfies Assumptions 3.1–3.5 on R with $( \sigma _ { 0 } , 0 )$ Let $\begin{array} { r } { x _ { \lambda } : = - \frac { p \lambda } { ( 1 - p ) L } } \end{array}$ Then, for any $x ^ { 0 } \in \mathbb { R }$ and any deterministic stepsizes $\gamma _ { k } > 0$ with $\textstyle \sum _ { k } \gamma _ { k } = \infty$ and $\textstyle \sum _ { k } \gamma _ { k } ^ { 2 } < \infty$ Clip-SGD with $\lambda _ { k } \equiv \lambda$ satisfies $x ^ { k } \to x _ { \lambda }$ almost surely, and

$$
\operatorname* { l i m } _ { k \to \infty } f ( x ^ { k } ) - f ^ { * } = \frac { L } { 2 } x _ { \lambda } ^ { 2 } \geq \frac { \sigma _ { \lambda } ^ { 2 \alpha } } { 8 \cdot 1 6 ^ { \alpha } L \lambda ^ { 2 ( \alpha - 1 ) } } = \frac { 1 } { 8 \cdot 1 6 ^ { \alpha } L } \operatorname* { m i n } \left\{ \frac { \sigma _ { 0 } ^ { 2 \alpha } } { \lambda ^ { 2 ( \alpha - 1 ) } } , \lambda ^ { 2 } \right\} \quad \mathrm { a l m o s t ~ s u r e l y } .
$$

The proof relies on the following standard result from stochastic approximation.

Lemma H.5 (Scalar mean field with a unique attracting root). Let $V : \mathbb { R } \times \mathcal { Z } \to \mathbb { R }$ be measurable with $| V | \le B$ , let $\zeta _ { 0 } , \zeta _ { 1 } , \ldots$ . be i.i.d., and let $h ( x ) : = \mathbb { E } V ( x , \zeta _ { 0 } )$ be continuous with $( x - z ) h ( x ) > 0$ for all $x \neq z$ . Let $\gamma _ { k } > 0$ be deterministic with $\textstyle \sum _ { k } \gamma _ { k } = \infty$ and $\textstyle \sum _ { k } \gamma _ { k } ^ { 2 } < \infty$ Then, for every $x ^ { 0 } \in \mathbb { R }$ , the recursion $x ^ { k + 1 } = x ^ { k } - \gamma _ { k } V ( x ^ { k } , \zeta _ { k } )$ converges to z almost surely.

Proof. Continuity and the sign condition imply $h ( z ) = 0$ . Let $\mathcal { G } _ { k } : = \sigma ( \zeta _ { 0 } , \ldots , \zeta _ { k - 1 } )$ and $W _ { k } : =$ $( x ^ { k } - z ) ^ { 2 }$ . For each fixed $k ,$ the bound $\begin{array} { r } { | { \boldsymbol x } ^ { k } | \leq | { \boldsymbol x } ^ { 0 } | + B \sum _ { j < k } \gamma _ { j } } \end{array}$ implies that $W _ { k }$ is integrable. Since $x ^ { k }$ is $\mathcal { G } _ { k }$ -measurable and $\zeta _ { k }$ is independent of $\mathcal { G } _ { k }$ , we have

$$
\begin{array} { r } { \mathbb { E } [ W _ { k + 1 } \mid \mathcal { G } _ { k } ] \le W _ { k } - 2 \gamma _ { k } ( x ^ { k } - z ) h ( x ^ { k } ) + \gamma _ { k } ^ { 2 } B ^ { 2 } . } \end{array}
$$

The last term is summable, and $( x ^ { k } - z ) h ( x ^ { k } ) \geq 0$ . By the Robbins–Siegmund theorem (Robbins and Siegmund, 1971), $W _ { k }$ converges almost surely to a finite limit $W _ { \infty }$ , and $\begin{array} { r } { \sum _ { k } \gamma _ { k } ( x ^ { k } - z ) h ( x ^ { k } ) < \infty } \end{array}$ almost surely; in particular, the whole trajectory is bounded almost surely. On the event $\{ W _ { \infty } > 0 \}$ for all suficiently large $k ,$ the iterates stay in the compact set $\{ x : \sqrt { W _ { \infty } } / 2 \leq | x - z | \leq 2 \sqrt { W _ { \infty } } \}$ On this set, the continuous positive function $( x - z ) h ( x )$ is bounded from below by some $\delta > 0$ and, therefore, $\begin{array} { r } { \sum _ { k } \gamma _ { k } ( x ^ { k } - z ) h ( x ^ { k } ) = \infty } \end{array}$ since $\textstyle \sum _ { k } \gamma _ { k } = \infty$ , which is a contradiction. Thus, $W _ { \infty } = 0$ almost surely, which concludes the proof. □

Proof of Proposition H.4. The noise coincides with the one from instance $\operatorname { A } ;$ therefore, the assumptions hold as shown there. Let $V ( x , \zeta ) : = { \tt c l i p } ( L x + \zeta , \lambda )$ , which satisfies $| V | \leq \lambda$ , and let $h ( x ) : = \mathbb { E } V ( x , \zeta ) = ( 1 - p ) \mathsf { c } \mathsf { 1 i p } ( L x + \nu , \lambda ) + p \mathsf { c 1 i p } ( L x - 2 \lambda , \lambda )$ . The function $h$ is continuous and non-decreasing, since both clipped terms are. We recall that $\begin{array} { r } { p \leq \frac { 1 } { 8 } , \nu \leq \frac { 2 } { 7 } \lambda . } \end{array}$ , and $( 1 - p ) \nu = 2 p \lambda$ $\mathrm { I f } - \lambda - \nu \leq L x \leq \lambda - \nu ,$ , then $L x + \nu \in [ - \lambda , \lambda ]$ and $L x - 2 \lambda < - \lambda$ , and, therefore,

$$
h ( x ) = ( 1 - p ) ( L x + \nu ) - p \lambda = ( 1 - p ) L x + p \lambda .
$$

On this interval, h is strictly increasing and vanishes exactly at $x _ { \lambda }$ , which belongs to the interval since $\begin{array} { r } { \frac { p \lambda } { 1 - p } \le \frac { \lambda } { 7 } < \lambda - \nu } \end{array}$ . By monotonicity, $\begin{array} { r } { h ( x ) \ge h \left( \frac { \lambda - \nu } { L } \right) > 0 } \end{array}$ for $L x > \lambda - \nu ,$ and $h ( x ) \ \leq$ $\begin{array} { r } { h \left( \frac { - \lambda - \nu } { L } \right) = - \lambda < 0 } \end{array}$ for $L x < - \lambda - \nu$ . Thus, $( x - x _ { \lambda } ) h ( x ) > 0$ for all $x \neq x _ { \lambda }$ , and Lemma H.5 gives $x ^ { k } \to x \lambda$ almost surely. By continuity, $\begin{array} { r } { f ( x ^ { k } ) - f ^ { * } \to \frac { L } 2 x _ { \lambda } ^ { 2 } \geq \frac { ( p \lambda ) ^ { 2 } } { 2 L } } \end{array}$ . It remains to notice that $\begin{array} { r } { p \lambda = \frac { \sigma _ { \lambda } ^ { \alpha } } { 2 \cdot 4 ^ { \alpha } \lambda ^ { \alpha - 1 } } } \end{array}$ and $\sigma _ { \lambda } ^ { 2 \alpha } / \lambda ^ { 2 ( \alpha - 1 ) } = \operatorname* { m i n } \{ \sigma _ { 0 } ^ { 2 \alpha } / \lambda ^ { 2 ( \alpha - 1 ) } , \lambda ^ { 2 } \}$ , which concludes the proof. □

## H.5 Strong Multiplicative Noise

The oracle below is the multiplicative oracle from Faw et al. (2023, Proposition 13) with $\epsilon = 2$ and $\sigma _ { 1 } = 9$ in their notation, for which their Lemma 35 shows that clipped and normalized SGD with any constant stepsize make no progress with constant probability on a quadratic; the following proposition extends the clipping-failure argument to bounded deterministic stepsize sequences, with an explicit probability and divergence on the event.

Proposition H.6 (Strong multiplicative noise). Let $\begin{array} { r } { L , \lambda > 0 , \alpha \in ( 1 , 2 ] , f ( x ) = \frac { L } { \gamma } x ^ { 2 } } \end{array}$ on $\mathbb { R }$ , and $g ( x , \xi ) : = L x ( 1 + 3 Z )$ , where ${ \mathbb { P } } \{ Z = - 1 \} = 0 . 9$ and ${ \mathbb { P } } \{ Z = 9 \} = 0 . 1$ . Then $\bar { \mathbb { E } } \bar { g ( } x , \xi ) = f ^ { \prime } ( x )$ and $\mathbb { E } | g ( x , \xi ) - f ^ { \prime } ( x ) | ^ { \alpha } = \sigma _ { 1 } ^ { \alpha } | f ^ { \prime } ( x ) | ^ { \alpha }$ with $\sigma _ { 1 } : = ( 0 . 9 \cdot 3 ^ { \alpha } + 0 . 1 \cdot 2 7 ^ { \alpha } ) ^ { 1 / \alpha } \in ( 5 . 4 , 9 ]$ , meaning that Assumption 3.4 holds on R with $\sigma _ { 0 } = 0$ . Let $x ^ { 0 } \geq \lambda / L$ and let $( \gamma _ { k } ) _ { k \geq 0 }$ be any deterministic stepsizes with $0 < \gamma _ { k } \le 1 / L$ for all k. Then, with probability at least $2 / 3$ Clip-SGD with $\lambda _ { k } \equiv \lambda$ and stepsizes $\gamma _ { k }$ satisfies $x ^ { k } \geq { \frac { \lambda } { 2 L } }$ , and, therefore, $\begin{array} { r } { f ( x ^ { k } ) - f ^ { * } \ge \frac { \lambda ^ { 2 } } { 8 L } } \end{array}$ , for all $k \geq 0 . \mathrm { ~ H f ~ }$ in addition, $\textstyle \sum _ { k } \gamma _ { k } = \infty$ , then $x ^ { k } \to + \infty$ almost surely on this event.

The stepsize caps in Theorems F.1, F.4, and G.1 imply $\gamma \leq 1 / L$ . Their other assumptions exclude the above instance: the first two theorems require $\lambda \ge 2 \sqrt { 2 } L R$ with $R \geq | x ^ { 0 } |$ , which is incompatible with $x ^ { 0 } \geq \lambda / L$ , while Theorem G.1 requires $\sigma _ { 1 } \leq \bar { \sigma } _ { 1 }$

Proof. Unbiasedness follows from $\mathbb { E } Z = 0$ , and $| g ( x , \xi ) - f ^ { \prime } ( x ) | = 3 | Z | | f ^ { \prime } ( x ) |$ gives the moment identity; $\sigma _ { 1 } = \| 3 Z \| _ { L ^ { \alpha } }$ is non-decreasing in $\alpha ,$ equals 5.4 at $\alpha = 1$ , and equals 9 at $\begin{array} { r } { \alpha = 2 . \mathrm { ~ H ~ } x \geq \frac { \lambda } { 2 L } } \end{array}$ then $g ( x , \xi ) \in \{ - 2 L x , 2 8 L x \}$ with $2 L x \geq \lambda .$ , and, therefore, ${ \mathsf { c l i p } } ( g ( x , \xi ) , \lambda )$ equals −λ if $Z = - 1$ and λ if $Z = 9$ . Let $\chi _ { k } : = 1$ if $Z _ { k } = - 1$ and $\chi _ { k } : = - 1$ otherwise; these random variables are i.i.d. with $\mathbb { P } \{ \chi _ { k } = 1 \} = 0 . 9$ . Let $\tau : =$ inf $\textstyle \left\{ k \geq 0 : x ^ { k } < { \frac { \lambda } { 2 L } } \right\}$ . For $k < \tau$ , we have $x ^ { k + 1 } = x ^ { k } + \gamma _ { k } \lambda \chi _ { k }$

Next, let $\bar { \gamma } : = \operatorname* { s u p } _ { k } \gamma _ { k } \leq 1 / L$ and $\kappa : = \ln 9 / ( \bar { \gamma } \lambda )$ . The function $\varphi ( u ) : = 0 . 9 e ^ { - u } + 0 . 1 e ^ { u }$ is convex with $\varphi ( 0 ) = \varphi ( \ln 9 ) = 1$ ; hence, $\varphi \leq 1$ on [0, ln 9]. Since $\kappa \gamma _ { k } \lambda \in ( 0 , \ln 9 ]$ , we have $\mathbb { E } e ^ { - \kappa \gamma _ { k } \lambda \chi _ { k } } =$ $\varphi ( \kappa \gamma _ { k } \lambda ) \ \leq \ 1$ . Thus, $M _ { k } : = \mathrm { e x p } \left( - \kappa \left( x ^ { \mathrm { m i n } \{ k , \tau \} } - \frac { \lambda } { 2 L } \right) \right)$ is a non-negative supermartingale, and $M _ { \tau } > 1 \mathrm { ~ o n ~ } \{ \tau < \infty \}$ . By the maximal inequality for non-negative supermartingales, we get

$$
\mathbb { P } \{ \tau < \infty \} \le \mathbb { P } \left\{ \operatorname* { s u p } _ { k } M _ { k } \ge 1 \right\} \le M _ { 0 } \le \exp \left( - \frac { \kappa \lambda } { 2 L } \right) = 9 ^ { - 1 / ( 2 \bar { \gamma } L ) } \le \frac { 1 } { 3 } .
$$

On $\{ \tau ~ = ~ \infty \}$ , we have $\begin{array} { r } { x ^ { k } = x ^ { 0 } + \lambda \sum _ { i < k } \gamma _ { j } \chi _ { j } } \end{array}$ for all k. Let $\begin{array} { r } { \Gamma _ { k } : = \sum _ { j < k } \gamma _ { j } \to \infty } \end{array}$ . Since $\begin{array} { r } { \sum _ { j \ge 0 } \gamma _ { j } ^ { 2 } / \Gamma _ { j + 1 } ^ { 2 } \ \le \ \bar { \gamma } \left( \frac 1 { \gamma _ { 0 } } + \sum _ { j \ge 1 } \left( \frac 1 { \Gamma _ { j } } - \frac 1 { \Gamma _ { j + 1 } } \right) \right) \ \le \ \frac { 2 \bar { \gamma } } { \gamma _ { 0 } } } \end{array}$ , the martingale $\textstyle \sum _ { j < k } \gamma _ { j } ( \chi _ { j } - 0 . 8 ) / \Gamma _ { j + 1 }$ is bounded in $L ^ { 2 }$ and converges almost surely. By Kronecker’s lemma, $\begin{array} { r } { \frac { 1 } { \Gamma _ { k } } \sum _ { j < k } \gamma _ { j } ( \chi _ { j } - 0 . 8 ) \to 0 ; } \end{array}$ hence, $\begin{array} { r } { \frac { 1 } { \Gamma _ { k } } \sum _ { j < k } \gamma _ { j } \chi _ { j } \ \to \ 0 . 8 } \end{array}$ and $x ^ { k } \to + \infty$ almost surely on $\{ \tau ~ = ~ \infty \}$ , which concludes the proof. □

## I Proofs for Clip-SSTM

## I.1 Full Statements

For a parameter $a \geq 1$ , the weights $\begin{array} { r } { \omega _ { k + 1 } = \frac { k + 2 } { 2 a L } } \end{array}$ and $A _ { k + 1 } = A _ { k } + \omega _ { k + 1 }$ from Algorithm 2 satisfy

$$
A _ { k } = \frac { k ( k + 3 ) } { 4 a L } , \qquad \frac { L \omega _ { k + 1 } ^ { 2 } } { A _ { k + 1 } } = \frac { ( k + 2 ) ^ { 2 } } { a ( k + 1 ) ( k + 4 ) } \leq \frac { 1 } { a } ,\tag{60}
$$

since $( k + 1 ) ( k + 4 ) - ( k + 2 ) ^ { 2 } = k \geq 0$ . We analyze Clip-SSTM under either noise model with $\alpha \in ( 1 , 2 ]$ , with the clipping constants $( b , v )$ from Section 5: under Assumption 3.5, $v = v _ { \alpha } ( \Lambda _ { K } ^ { \mathrm { S } } )$ where

$$
\Lambda _ { K } ^ { \mathrm { S } } : = \operatorname* { m a x } \left\{ 2 0 0 b K , \Lambda _ { K } ^ { 0 } \right\} , \qquad \Lambda _ { K } ^ { 0 } : = \left[ \left( \frac { \sigma _ { 0 } ( K + 1 ) } { 1 7 6 4 0 L R H } \right) ^ { \alpha } + \left( \frac { \sigma _ { 1 } } { 3 } \right) ^ { \alpha } \right] ^ { - 1 }\tag{61}
$$

$( \Lambda _ { K } ^ { 0 } : = 0 \mathrm { ~ i f ~ } \sigma _ { 0 } = \sigma _ { 1 } = 0 )$ plays the role of $\Lambda _ { K }$ from Theorem F.1, and $\begin{array} { r } { H : = \ln { \frac { 4 K } { \beta } } } \end{array}$ is defined as in Theorem I.1 below.

Theorem I.1 (Clip-SSTM: full statement of Theorem 5.2). Let Assumptions 3.1–3.3 hold for $Q = B _ { 3 R } ( x ^ { * } )$ , where $R \geq \| x ^ { 0 } - x ^ { * } \| > 0$ and let either Assumption 3.4 or Assumption 3.5 hold with $\alpha \in ( 1 , 2 ]$ . Let $K \geq 1$ be an integer, $\begin{array} { r } { \beta \in ( 0 , 1 ] , H : = \ln \frac { 4 K } { \beta } , r : = \frac { \alpha - 1 } { \alpha } } \end{array}$ , let $( b , v )$ be defined as above,

$$
q _ { \alpha } : = \operatorname* { m i n } \left\{ { \frac { 1 } { 1 3 5 0 v 3 0 ^ { \alpha - 2 } } } , \ { \frac { 1 } { 1 5 b 3 0 ^ { \alpha - 1 } } } \right\} ,
$$

$$
c _ { 0 , \alpha } : = \left( \frac { 2 } { q _ { \alpha } } \right) ^ { 1 / \alpha } , \qquad c _ { 1 , \alpha } : = 4 9 \left( \frac { 4 } { q _ { \alpha } } \right) ^ { 2 / \alpha } ,
$$

and

$$
a _ { * } ( K , \beta ) : = { \operatorname* { m a x } \left\{ 1 7 6 4 0 0 H ^ { 2 } , \ c _ { 0 , \alpha } \frac { \sigma _ { 0 } } { L R } ( K + 1 ) K ^ { \frac { 1 } { \alpha } } H ^ { r } , \ c _ { 1 , \alpha } \sigma _ { 1 } ^ { 2 } K ^ { \frac { 2 } { \alpha } } H ^ { 2 r } \right\} } .
$$

Then, for any $a \ge a _ { * } ( K , \beta )$ and $\begin{array} { r } { \lambda _ { k } = \frac { R } { 3 0 \omega _ { k + 1 } H } , 0 \leq k < K } \end{array}$ there exists an event of probability at least $1 - \beta$ on which all points $x ^ { k } , y ^ { k } , z ^ { k } , k = 0 , \ldots , K$ produced by Algorithm 2 belong to $B _ { 2 R } ( x ^ { * } )$ and

$$
f ( y ^ { K } ) - f ^ { * } \leq \frac { 6 a L R ^ { 2 } } { K ( K + 3 ) } .\tag{62}
$$

Moreover, for $a = a _ { * } ( K , \beta )$ , on the same event, we have

$$
f ( y ^ { K } ) - f ^ { * } \leq \operatorname* { m a x } \Big \{ \frac { 1 0 5 8 4 0 0 L R ^ { 2 } H ^ { 2 } } { K ^ { 2 } } , \frac { 1 2 c _ { 0 , \alpha } \sigma _ { 0 } R H ^ { r } } { K ^ { r } } , \frac { 6 c _ { 1 , \alpha } \sigma _ { 1 } ^ { 2 } L R ^ { 2 } H ^ { 2 r } } { K ^ { 2 r } } \Big \} .\tag{63}
$$

The constants $c _ { 0 , \alpha } , c _ { 1 , \alpha }$ depend only on α under Assumption 3.4 $( \mathrm { a t ~ } \alpha = 2 , q _ { 2 } = \frac { 1 } { 1 3 5 0 } , c _ { 0 , 2 } =$ $\sqrt { 2 7 0 0 } , c _ { 1 , 2 } = 4 9 \cdot 5 4 0 0 )$ , or under Assumption 3.5 with $\alpha < 2$ if v is replaced with $v _ { \alpha } ( \infty )$ (which is allowed), and are $O ( \sqrt { v } )$ and $O ( v )$ with $v = 1 + \ln ( 1 + \Lambda _ { K } ^ { \mathrm { S } } ) = O ( \ln ( e K ) + \ln ( 1 + \Lambda _ { K } ^ { 0 } ) )$ at $\alpha = 2$ under Assumption 3.5.

Corollary I.2 (Oracle complexity of Clip-SSTM). Let the assumptions of Theorem I.1 hold with $\alpha < 2$ under Assumption 3.5, in which case $q _ { \alpha } , \ c _ { 0 , \alpha } , \ c _ { 1 , \alpha }$ and $a _ { * } ( K , \beta )$ are computed with $v = v _ { \alpha } ( \infty )$ ; let $C _ { \alpha } : = \operatorname* { m a x } \{ 1 0 5 8 4 0 0 , 1 2 c _ { 0 , \alpha } , 6 c _ { 1 , \alpha } \} , \varepsilon > 0$ , and

$$
M _ { \varepsilon } : = \operatorname* { m a x } \left\{ 1 , \sqrt { \frac { C _ { \alpha } L R ^ { 2 } } { \varepsilon } } , \left( \frac { C _ { \alpha } \sigma _ { 0 } R } { \varepsilon } \right) ^ { \frac { 1 } { r } } , \left( \frac { C _ { \alpha } \sigma _ { 1 } ^ { 2 } L R ^ { 2 } } { \varepsilon } \right) ^ { \frac { 1 } { 2 r } } \right\} .
$$

If $K \ \ge \ 4 M _ { \varepsilon }$ ln $\frac { 4 M _ { \varepsilon } } { \beta }$ , then Algorithm 2 with $a = a _ { * } ( K , \beta )$ and $\lambda _ { k }$ defined as in Theorem I.1 guarantees $f ( y ^ { K } ) - f ^ { * } \leq \varepsilon$ with probability at least $1 - \beta$ . In particular,

$$
\widetilde { \cal O } _ { \alpha } \left( \operatorname * { m a x } \left\{ 1 , \sqrt { \frac { L R ^ { 2 } } { \varepsilon } } , \left( \frac { \sigma _ { 0 } R } { \varepsilon } \right) ^ { \frac { \alpha } { \alpha - 1 } } , \left( \frac { \sigma _ { 1 } ^ { 2 } L R ^ { 2 } } { \varepsilon } \right) ^ { \frac { \alpha } { 2 ( \alpha - 1 ) } } \right\} \right)
$$

oracle calls are suficient, where $\widetilde { O } _ { \alpha }$ hides constants depending only on α and logarithmic factors, including those in $1 / \beta$ At $\alpha = 2$ under Assumption 3.5, the same complexity holds with an additional polylogarithmic factor, as in Corollary F.2.

Throughout Sections I.3–I.7, we assume that Assumptions 3.1–3.3 hold on $Q = B _ { 3 R } ( x ^ { * } )$ , either Assumption 3.4 or Assumption 3.5 holds with $\alpha \in ( 1 , 2 ]$ , and $( b , v ) , H , r , q _ { \alpha } , c _ { 0 , \alpha } , c _ { 1 , \alpha } , a _ { * } ( K , \beta )$ $\Lambda _ { K } ^ { \mathrm { S } } , \ \Lambda _ { K } ^ { 0 }$ are defined as in Theorem I.1 and (61); $\nabla f : = 0$ outside $Q .$ as in Appendix E. As in Appendix F, let $\phi _ { \lambda } ( s ) : = v _ { \alpha } ^ { \mathrm { m o m } } s$ under Assumption 3.4 and $\phi _ { \lambda } ( s ) : = s v _ { \alpha } ( ( 3 \lambda / 2 ) ^ { \alpha } / s )$ under Assumption 3.5, with $\phi _ { \lambda } ( 0 ) : = 0$ . For $a \geq 1$ , we define

$$
\bar { \Psi } _ { K } ( a ) : = \frac { \sigma _ { 0 } ^ { \alpha } H ^ { \alpha - 1 } K ( K + 1 ) ^ { \alpha } } { 2 ^ { \alpha } a ^ { \alpha } L ^ { \alpha } R ^ { \alpha } } + \frac { 7 ^ { \alpha } \sigma _ { 1 } ^ { \alpha } H ^ { \alpha - 1 } K } { a ^ { \alpha / 2 } } , \quad \Lambda ( a ) : = \frac { K } { 2 0 ^ { \alpha } H \bar { \Psi } _ { K } ( a ) } \quad ( \Lambda ( a ) : = 0 \mathrm { ~ i f ~ } \bar { \Psi } _ { K } ( a ) = 0 ) ,\tag{64}
$$

which are the deterministic bound on the afine budget from Step 3 below and the corresponding logarithmic ratio; $\bar { \Psi } _ { K }$ is non-increasing and Λ is non-decreasing in a. For the iterates of Algorithm $2 ,$ we define the errors

$$
\begin{array} { r } { \theta _ { k + 1 } : = \widetilde { g } _ { k } - \nabla f ( x ^ { k + 1 } ) = \theta _ { k + 1 } ^ { u } + \theta _ { k + 1 } ^ { b } , \qquad \theta _ { k + 1 } ^ { u } : = \widetilde { g } _ { k } - \mathbb { E } [ \widetilde { g } _ { k } \mid \mathcal { G } _ { k } ] , \qquad \theta _ { k + 1 } ^ { b } : = \mathbb { E } [ \widetilde { g } _ { k } \mid \mathcal { G } _ { k } ] - \nabla f ( x ^ { k + 1 } ) . } \end{array}\tag{65}
$$

Since $\| \widetilde { g } _ { k } \| \le \lambda _ { k }$ , both conditional expectations are defined on the whole probability space, $\mathbb { E } [ \theta _ { k + 1 } ^ { u }$ $\mathcal { G } _ { k } ] = 0$ , and $\lVert \theta _ { k + 1 } ^ { u } \rVert \leq 2 \lambda _ { k }$ almost surely. By induction over the recursions and the joint measurability of the oracle, $y ^ { k } , z ^ { k }$ , and $x ^ { k + 1 }$ are $\mathcal { G } _ { k } .$ -measurable. We never need the unconditional mean of the unclipped oracle at random points outside $Q$

## I.2 Adaptive Query Points

Lemma I.3 (Adaptive query points). Let Assumption 3.1 hold on $Q = B _ { \rho R } ( x ^ { * } )$ for some $\rho > 0$ and let Assumption 3.4 or Assumption 3.5 hold on Q. Let X be a G-measurable random vector, let $\xi \sim \mathcal { D }$ be independent of $\mathcal { G }$ , and let $A _ { X } : = \{ X \in Q \}$ . Then $\mathbf { 1 } _ { A _ { X } } g ( X , \xi ) \in L ^ { 1 }$ and

$$
\mathbb { E } \left[ \mathbf { 1 } _ { A _ { X } } g ( X , \xi ) \mid \boldsymbol { \mathcal { G } } \right] = \mathbf { 1 } _ { A _ { X } } \nabla f ( X ) .\tag{66}
$$

Moreover, almost surely, under Assumption 3.5, for every $u > 0$

$$
\mathbf { 1 } _ { A _ { X } } \mathbb { P } \left\{ \| g ( X , \xi ) - \nabla f ( X ) \| > u \mid \mathcal { G } \right\} \leq \mathbf { 1 } _ { A _ { X } } \frac { \sigma _ { 0 } ^ { \alpha } + \sigma _ { 1 } ^ { \alpha } \| \nabla f ( X ) \| ^ { \alpha } } { u ^ { \alpha } } ,\tag{67}
$$

and under Assumption 3.4, $\mathbf { 1 } _ { A _ { X } } \mathbb { E } \left[ \lVert g ( X , \xi ) - \nabla f ( X ) \rVert ^ { \alpha } \mid \mathcal { G } \right] \leq \mathbf { 1 } _ { A _ { X } } { \big ( } \sigma _ { 0 } ^ { \alpha } + \sigma _ { 1 } ^ { \alpha } \lVert \nabla f ( X ) \rVert ^ { \alpha } { \big ) } .$

Proof. Since $x ^ { * }$ is an unconstrained minimizer of the diferentiable function $f ,$ we have $\nabla f ( x ^ { * } ) = 0$ 2 and Assumption 3.1 gives

$$
\| \nabla f ( x ) \| \leq L \| x - x ^ { * } \| \leq \rho L R , \qquad x \in Q .\tag{68}
$$

Let $M _ { Q } : = ( \sigma _ { 0 } ^ { \alpha } + \sigma _ { 1 } ^ { \alpha } ( \rho L R ) ^ { \alpha } ) ^ { 1 / \alpha }$ . For every $x \in Q$ , we have

$$
\mathbb { E } _ { \xi } \| g ( x , \xi ) \| \le \rho L R + \frac { \alpha } { \alpha - 1 } M _ { Q } .\tag{69}
$$

Indeed, under Assumption 3.4, this follows from Lyapunov’s inequality, $\mathbb { E } _ { \xi } \| g ( x , \xi ) - \nabla f ( x ) \| \le M _ { Q }$ Under Assumption 3.5, for $Y \geq 0$ and $M > 0$ with $\mathbb { P } \{ Y > u \} \le \operatorname* { m i n } \{ 1 , \overleftarrow { M ^ { \alpha } u ^ { - \alpha } } \}$ for all $u > 0$ , we have $\begin{array} { r } { \mathbb { E } Y \le \int _ { 0 } ^ { M } \mathrm { d } u + M ^ { \alpha } \int _ { M } ^ { \infty } u ^ { - \alpha } \mathrm { d } u = \frac { \alpha } { \alpha - 1 } M ; } \end{array}$ for $M = 0 , Y = 0$ almost surely.

Next, the vector $X _ { Q } : = X \mathbf { 1 } _ { A _ { X } } + x ^ { * } \mathbf { 1 } _ { A _ { X } ^ { c } }$ is G-measurable and belongs to $Q$ . By independence and joint measurability, for every jointly measurable $\varphi$ with $\mathbb { E } | \varphi ( X _ { Q } , \xi ) | < \infty$ , we have

$$
\begin{array} { r } { \mathbb { E } \left[ \varphi ( X _ { Q } , \xi ) \mid \mathcal { G } \right] = \mathbb { E } _ { \xi \sim \mathcal { D } } \varphi ( x , \xi ) \big | _ { x = X _ { Q } } . } \end{array}
$$

This formula is verified first for indicators of measurable rectangles in $( x , \xi )$ , then for simple functions, and, finally, for integrable functions by monotone convergence and splitting into positive and negative parts. By (69), $g ( X _ { Q } , \xi )$ is integrable, uniformly in the value of $X _ { Q }$ , and $\mathbb { E } [ g ( X _ { Q } , \xi ) \mid \mathcal { G } ] = \nabla f ( X _ { Q } )$ . Multiplying by ${ \mathbf { 1 } } _ { A _ { X } }$ , on which $X _ { Q } \ : = \ : X$ , we get (66). For the tail indicator $\varphi = \mathbf { 1 } \{ \| g ( x , \xi ) - \nabla f ( x ) \| > u \}$ and for the non-negative function $\varphi = \| g ( x , \xi ) - \nabla f ( x ) \| ^ { \alpha }$ the same formula does not require the integrability of $g .$ . For the tail probabilities, we choose the version supplied by the freezing formula; the pointwise tail assumption then gives the inequality simultaneously for every $u > 0$ . Together with Assumption 3.5 (resp. Assumption 3.4), it gives the remaining claims, which concludes the proof. □

Without additional integrability of $g ( X , \xi )$ , the conditional expectation $\mathbb { E } [ g ( X , \xi ) \mid \mathcal { G } ]$ outside $Q$ is never used. For example, if $g ( x , \xi ) = x$ and X has a Cauchy distribution, then the vector $g ( X , \xi )$ is not integrable, while its restriction to $\{ X \in Q \}$ is. In contrast, the clipped oracle is always bounded and has an ordinary conditional expectation on the whole probability space.

## I.3 Potential Inequality

Lemma I.4 (Potential inequality). Let $a \geq 1$ and let the points $x ^ { k + 1 } , y ^ { k } , y ^ { k + 1 } , z ^ { k } , z ^ { k + 1 } , 0 \leq k <$ $N$ , produced by Algorithm 2 belong to $Q$ . Then, for every $z \in Q$ , we have

$$
\begin{array} { l } { \displaystyle { A _ { N } \left( f ( y ^ { N } ) - f ( z ) \right) \leq \frac { 1 } { 2 } \| z ^ { 0 } - z \| ^ { 2 } - \frac { 1 } { 2 } \| z ^ { N } - z \| ^ { 2 } } } \\ { \displaystyle { \quad \quad + \sum _ { k = 0 } ^ { N - 1 } \omega _ { k + 1 } \left. \theta _ { k + 1 } , z - z ^ { k } + \omega _ { k + 1 } \nabla f ( x ^ { k + 1 } ) \right. + \sum _ { k = 0 } ^ { N - 1 } \omega _ { k + 1 } ^ { 2 } \| \theta _ { k + 1 } \| ^ { 2 } } . } \end{array}\tag{70}
$$

Proof. We fix one step and introduce the notation $v : = \nabla f ( x ^ { k + 1 } )$ and $g : = \widetilde { g } _ { k } = v + \theta _ { k + 1 }$ . The recursions give $\begin{array} { r } { y ^ { k + 1 } - x ^ { k + 1 } = - \frac { \omega _ { k + 1 } ^ { 2 } } { A _ { k + 1 } } g . } \end{array}$ The descent lemma holds on the segment $[ x ^ { k + 1 } , y ^ { k + 1 } ] \subset Q$ 2 since it follows by integrating Assumption 3.1 along the segment. Thus, by (60) and $a \ge 1$ , we have

$$
A _ { k + 1 } f ( y ^ { k + 1 } ) \leq A _ { k + 1 } f ( x ^ { k + 1 } ) - \omega _ { k + 1 } ^ { 2 } \left. v , g \right. + \frac { L \omega _ { k + 1 } ^ { 4 } } { 2 A _ { k + 1 } } \| g \| ^ { 2 } \leq A _ { k + 1 } f ( x ^ { k + 1 } ) - \frac { \omega _ { k + 1 } ^ { 2 } } { 2 } \| g \| ^ { 2 } + \omega _ { k + 1 } ^ { 2 } \left. \theta _ { k + 1 } , g \right. .
$$

The identity $A _ { k } ( x ^ { k + 1 } - y ^ { k } ) = \omega _ { k + 1 } ( z ^ { k } - x ^ { k + 1 } )$ and convexity give

$$
A _ { k + 1 } f \big ( x ^ { k + 1 } \big ) - A _ { k } f \big ( y ^ { k } \big ) \leq \omega _ { k + 1 } \left( f \big ( x ^ { k + 1 } \big ) + \Big \langle v , z ^ { k } - x ^ { k + 1 } \Big \rangle \right) \leq \omega _ { k + 1 } f \big ( z \big ) + \omega _ { k + 1 } \left. v , z ^ { k } - z \right. .
$$

Finally, the update rule for z gives $\begin{array} { r } { \omega _ { k + 1 } \langle g , z ^ { k } - z \rangle - \frac { \omega _ { k + 1 } ^ { 2 } } { 2 } \| g \| ^ { 2 } = \frac { 1 } { 2 } \| z ^ { k } - z \| ^ { 2 } - \frac { 1 } { 2 } \| z ^ { k + 1 } - z \| ^ { 2 } . } \end{array}$ Substituting $v = g - \theta _ { k + 1 }$ and combining the three relations, we get

$$
A _ { k + 1 } f ( y ^ { k + 1 } ) - A _ { k } f ( y ^ { k } ) \leq \omega _ { k + 1 } f ( z ) + \frac { 1 } { 2 } \| z ^ { k } - z \| ^ { 2 } - \frac { 1 } { 2 } \| z ^ { k + 1 } - z \| ^ { 2 } + \omega _ { k + 1 } \left. \theta _ { k + 1 } , z - z ^ { k + 1 } \right. .
$$

It remains to use $z - z ^ { k + 1 } = z - z ^ { k } + \omega _ { k + 1 } \nabla f ( x ^ { k + 1 } ) + \omega _ { k + 1 } \theta _ { k + 1 }$ and to sum up the above inequality over $k = 0 , \ldots , N - 1$ . The terms $A _ { k } f ( y ^ { k } )$ and the squared distances telescope, and $A _ { 0 } = 0$ , which concludes the proof. □

## I.4 Weighted Bound on the True Gradients

For $t \geq 0$ , we define

$$
B _ { t } : = \sum _ { l = 0 } ^ { t - 1 } \omega _ { l + 1 } \left. \theta _ { l + 1 } , x ^ { * } - z ^ { l } + \omega _ { l + 1 } \nabla f ( x ^ { l + 1 } ) \right. + \sum _ { l = 0 } ^ { t - 1 } \omega _ { l + 1 } ^ { 2 } \| \theta _ { l + 1 } \| ^ { 2 } , \qquad B _ { 0 } : = 0 .\tag{71}
$$

These are finite measurable random variables; we do not need their unconditional integrability.

Lemma I.5 (Gradient budget). Let $a \geq 1$ and $T \in \{ 1 , \ldots , K \}$ , and assume that $B _ { t } \leq R ^ { 2 }$ and $\| z ^ { t } - x ^ { * } \| \leq 2 R$ for $0 \leq t < T$ . Then, for every $0 \leq t < T$ , we have

$$
\omega _ { t + 1 } \| \nabla f ( { x } ^ { t + 1 } ) \| \le \frac { 7 R } { \sqrt { a } } , \qquad \mathrm { a n d ~ h e n c e } \qquad \sum _ { t = 0 } ^ { T - 1 } \omega _ { t + 1 } ^ { \alpha } \| \nabla f ( { x } ^ { t + 1 } ) \| ^ { \alpha } \le \frac { 7 ^ { \alpha } R ^ { \alpha } T } { a ^ { \alpha / 2 } } .\tag{72}
$$

Proof. By the recursions and induction, $y ^ { t }$ is a convex combination of $z ^ { 0 } , \ldots , z ^ { t }$ . Thus, $y ^ { t } , x ^ { t + 1 } \in$ $B _ { 2 R } ( x ^ { * } )$ for $t < T$ . For $1 \leq t < T$ , all points from the first t steps belong to $Q$ , and Lemma I.4 with $N = t$ and $z = x ^ { * }$ gives

$$
A _ { t } \left( f ( y ^ { t } ) - f ^ { * } \right) \leq { \frac { 1 } { 2 } } R ^ { 2 } + B _ { t } \leq { \frac { 3 } { 2 } } R ^ { 2 } .\tag{73}
$$

Since $x ^ { t + 1 } - y ^ { t } = \omega _ { t + 1 } ( z ^ { t } - y ^ { t } ) / A _ { t + 1 }$ and $\| z ^ { t } - y ^ { t } \| \leq 4 R$ , Assumptions 3.1 and 3.2 give

$$
\| \nabla f ( x ^ { t + 1 } ) \| \leq L \| x ^ { t + 1 } - y ^ { t } \| + \| \nabla f ( y ^ { t } ) \| \leq { \frac { 4 L R \omega _ { t + 1 } } { A _ { t + 1 } } } + { \sqrt { 2 L ( f ( y ^ { t } ) - f ^ { * } ) } } \leq { \frac { 4 L R \omega _ { t + 1 } } { A _ { t + 1 } } } + R { \sqrt { \frac { 3 L } { A _ { t } } } } .
$$

Therefore, $\begin{array} { r } { \omega _ { t + 1 } \| \nabla f ( x ^ { t + 1 } ) \| \le 4 R \frac { L \omega _ { t + 1 } ^ { 2 } } { A _ { t + 1 } } + R \sqrt { \frac { 3 L \omega _ { t + 1 } ^ { 2 } } { A _ { t } } } } \end{array}$ . The first ratio is at most $1 / a$ by (60). For the second one, we have $\begin{array} { r } { \frac { L \omega _ { t + 1 } ^ { 2 } } { A _ { t } } = \frac { ( t + 2 ) ^ { 2 } } { a t ( t + 3 ) } \leq \frac { 9 } { 4 a } } \end{array}$ , since $9 t ( t + 3 ) - 4 ( t + 2 ) ^ { 2 } = ( t - 1 ) ( 5 t + 1 6 ) \geq 0$ Consequently, $\begin{array} { r } { \omega _ { t + 1 } \| \nabla f ( x ^ { t + 1 } ) \| \le \frac { 4 R } { a } + \frac { 3 \sqrt { 3 } R } { 2 \sqrt { a } } < \frac { 7 R } { \sqrt { a } } } \end{array}$ . For $t = 0$ , we have $x ^ { 1 } = x ^ { 0 }$ and $\omega _ { 1 } = 1 / ( a L )$ and (68) gives $\omega _ { 1 } \| \nabla f ( x ^ { 1 } ) \| \le R / a \le R / \sqrt { a }$ . Raising to the power α and summing up, we get the second claim, which concludes the proof. □

Remark I.6. The crude bound $\| \nabla f ( x ^ { t + 1 } ) \| \le 2 L R$ would give only a budget of the order $R ^ { \alpha } K ^ { \alpha + 1 } / a ^ { \alpha }$ while Lemma I.5 gives $R ^ { \alpha } K / a ^ { \alpha / 2 }$ Replacing the first budget with the second one is exactly what improves the dependence of Theorem I.1 on $\sigma _ { 1 }$

## I.5 Proof of Theorem I.1

As in the proof of Proposition $\mathrm { F } . 6 ,$ our proof is induction-based: by induction, we show that the quantities $B _ { t }$ and the distances $\| z ^ { t } - x ^ { * } \|$ remain bounded with high probability.

Step 1: induction events and applicability of the potential. To formulate the statement rigorously, for $k = 0 , \ldots , K$ , we introduce probability event $E _ { k } \in \mathcal G _ { k }$ as follows: for all $0 \leq t \leq k$

$$
B _ { t } \leq R ^ { 2 } \qquad \mathrm { a n d } \qquad \| z ^ { t } - x ^ { * } \| \leq 2 R .\tag{74}
$$

We will prove by induction that $\mathbb { P } ( E _ { k } ) \ge 1 - k \beta / K$ for all $k = 0 , 1 , \ldots , K$ . The base of the induction $( k = 0 )$ follows from $B _ { 0 } = 0$ and $\| z ^ { 0 } - x ^ { * } \| \leq R$ . Next, we assume that the statement holds for $k = T - 1$ , where $1 \leq T \leq K$ , and prove it for $k = T$ . On $E _ { T - 1 }$ , by the convexity of the ball and the recursions, we have $y ^ { t } , z ^ { t } , x ^ { t + 1 } \in B _ { 2 R } ( x ^ { * } )$ for $t < T$ . Moreover, since $H \geq \ln { 4 } > 1$ , we have

$$
\| z ^ { T } - x ^ { * } \| \leq \| z ^ { T - 1 } - x ^ { * } \| + \omega _ { T } \lambda _ { T - 1 } \leq 2 R + { \frac { R } { 3 0 H } } < 3 R .
$$

The point $y ^ { T }$ is a convex combination of $y ^ { T - 1 } \in B _ { 2 R } ( x ^ { * } )$ and $z ^ { T } \in B _ { 3 R } ( x ^ { * } )$ , implying $y ^ { T } \in Q$ Thus, on $E _ { T - 1 }$ , Lemma I.4 can be applied to the first T steps.

Step 2: the clipping condition. Lemma I.5 and $a \ge 1 7 6 4 0 0 H ^ { 2 }$ , i.e., $\sqrt { a } \geq 4 2 0 H$ , give that $E _ { T - 1 }$ implies $\omega _ { t + 1 } \| \nabla f ( x ^ { t + 1 } ) \| \leq 7 R / { \sqrt { a } } \leq R / ( 6 0 H ) , { \mathrm { i . e . } }$

$$
\| \nabla f ( x ^ { t + 1 } ) \| \le \frac { R } { 6 0 \omega _ { t + 1 } H } = \frac { \lambda _ { t } } { 2 } , \qquad t < T .\tag{75}
$$

Let $D _ { t } : = \sigma _ { 0 } ^ { \alpha } + \sigma _ { 1 } ^ { \alpha } \| \nabla f ( x ^ { t + 1 } ) \| ^ { \alpha }$ . To apply the conditional lemmas correctly, we consider the predictable event $\mathcal { C } _ { t } : = \{ x ^ { t + 1 } \in Q , \ \Vert \nabla f ( x ^ { t + 1 } ) \Vert \leq \lambda _ { t } / 2 \} \in \mathcal { G } _ { t } .$ , the point $\check { x } ^ { t + 1 } : = x ^ { t + 1 } \mathbf { 1 } _ { \mathcal { C } _ { t } } + x ^ { * } \mathbf { 1 } _ { \mathcal { C } _ { t } ^ { c } } \in$ $Q ,$ , and $U _ { t } : = g ( \check { x } ^ { t + 1 } , \xi ^ { t } )$ . Lemma I.3 gives $U _ { t } \in L ^ { 1 }$ and $\mathbb { E } [ U _ { t } \mid { \mathcal { G } } _ { t } ] = \nabla f ( { \check { x } } ^ { t + 1 } )$ , whose norm is at most $\lambda _ { t } / 2$ everywhere. Thus, Corollary D.5 can be applied to $U _ { t }$ . On $\mathcal { C } _ { t }$ , the vectors $\mathsf { c l i p } ( U _ { t } , \lambda _ { t } )$ and $\widetilde { g } _ { t }$ coincide. Since both of them are bounded and $\mathcal { C } _ { t } \in \mathcal { G } _ { t }$ , we have

$$
\begin{array} { r } { \mathbf { 1 } _ { \mathcal { C } _ { t } } \mathbb { E } \left[ \mathsf { c } \mathrm { 1 i p } ( U _ { t } , \lambda _ { t } ) - \widetilde { g } _ { t } \mid \mathcal { G } _ { t } \right] = \mathbb { E } \left[ \mathbf { 1 } _ { \mathcal { C } _ { t } } \left( \mathsf { c } \mathrm { 1 i p } ( U _ { t } , \lambda _ { t } ) - \widetilde { g } _ { t } \right) \mid \mathcal { G } _ { t } \right] = 0 , } \end{array}
$$

and the same argument applies to the conditional second moments of the centered vectors. Therefore, on $\mathcal { C } _ { t }$ , and, as a consequence, on $E _ { T - 1 } \subseteq { \mathcal { C } } _ { t }$ for $t < T$ , we have

$$
\begin{array} { r } { \| \theta _ { t + 1 } ^ { b } \| \leq b D _ { t } \lambda _ { t } ^ { 1 - \alpha } , \qquad \mathbb { E } \left[ \| \theta _ { t + 1 } ^ { u } \| ^ { 2 } \ | \ \mathcal { G } _ { t } \right] \leq \lambda _ { t } ^ { 2 - \alpha } \phi _ { \lambda _ { t } } ( D _ { t } ) , } \end{array}\tag{76}
$$

where $D _ { t } = A ( \nabla f ( x ^ { t + 1 } ) )$ is exactly the noise scale from the lemmas at $x ^ { t + 1 }$ . Here we always condition on $\mathcal { G } _ { t }$ and never on $E _ { T - 1 }$ , which may depend on future samples for $t < T - 1$ . The global bound $\lvert | \theta _ { t + 1 } ^ { u } \rvert | \leq 2 \lambda _ { t }$ was mentioned above.

Step 3: a single afine budget. Let

$$
\Psi _ { T } : = H ^ { \alpha - 1 } R ^ { - \alpha } \sum _ { t = 0 } ^ { T - 1 } \omega _ { t + 1 } ^ { \alpha } D _ { t } .\tag{77}
$$

On $E _ { T - 1 }$ , we have $\begin{array} { r l } { \sum _ { t < T } \omega _ { t + 1 } ^ { \alpha } D _ { t } = \sigma _ { 0 } ^ { \alpha } \sum _ { t < T } \omega _ { t + 1 } ^ { \alpha } + \sigma _ { 1 } ^ { \alpha } \sum _ { t < T } \omega _ { t + 1 } ^ { \alpha } \| \nabla f ( x ^ { t + 1 } ) \| ^ { \alpha } } & { { } } \end{array}$ . The first sum is at most $T ( T + 1 ) ^ { \alpha } / ( 2 ^ { \alpha } a ^ { \alpha } L ^ { \alpha } )$ , and the second one is bounded by Lemma I.5. Thus, since $T \leq K$ , we get

$$
\Psi _ { T } \leq \frac { \sigma _ { 0 } ^ { \alpha } H ^ { \alpha - 1 } K ( K + 1 ) ^ { \alpha } } { 2 ^ { \alpha } a ^ { \alpha } L ^ { \alpha } R ^ { \alpha } } + \frac { 7 ^ { \alpha } \sigma _ { 1 } ^ { \alpha } H ^ { \alpha - 1 } K } { a ^ { \alpha / 2 } } .\tag{78}
$$

The right-hand side of (78) is exactly $\bar { \Psi } _ { K } ( a )$ from (64). If $\sigma _ { 0 } > 0$ , then the second term in the definition of $a _ { * } ( K , \beta )$ gives $\begin{array} { r } { a ^ { \alpha } \ge \frac { 2 } { q _ { \alpha } } \frac { \sigma _ { 0 } ^ { \alpha } } { L ^ { \alpha } R ^ { \alpha } } ( K + 1 ) ^ { \alpha } K H ^ { \alpha - 1 } } \end{array}$ , meaning that the first term in (78) is at most $q _ { \alpha } / 2 ^ { \alpha + 1 }$ ; for $\sigma _ { 0 } = 0$ , it vanishes. Similarly, if $\sigma _ { 1 } > 0$ , then the third term gives $\begin{array} { r } { a ^ { \alpha / 2 } \geq \frac { 4 } { q _ { \alpha } } 7 ^ { \alpha } \sigma _ { 1 } ^ { \dot { \alpha } } K H ^ { \alpha - 1 } } \end{array}$ , meaning that the second term is at most $q _ { \alpha } / 4 ;$ for $\sigma _ { 1 } = 0$ , it vanishes. Thus,

$$
\Psi _ { T } \leq \bar { \Psi } _ { K } ( a ) \leq \frac { q _ { \alpha } } { 2 ^ { \alpha + 1 } } + \frac { q _ { \alpha } } { 4 } < \frac { q _ { \alpha } } { 2 } \leq q _ { \alpha } .\tag{79}
$$

Step $\mathbf { 3 ^ { \prime } } \colon$ the logarithm at $a = a _ { * } ( K , \beta )$ . Under Assumption 3.5, we also need the variance constant $v = v _ { \alpha } ( \Lambda _ { K } ^ { \mathrm { S } } )$ to dominate the actual logarithmic ratio at $a _ { * } : = a _ { * } ( K , \beta )$

$$
\Lambda ( a _ { * } ) \leq \Lambda _ { K } ^ { \mathrm { S } } .\tag{80}
$$

If $\bar { \Psi } _ { K } ( a _ { * } ) = 0$ , then $\Lambda ( { a _ { * } } ) = 0$ . Otherwise, we consider which term attains the maximum in $^ { a _ { * } }$ . If $a _ { * } = 1 7 6 4 0 0 H ^ { 2 }$ , then direct computations from (64) give $\begin{array} { r } { \bar { \Psi } _ { K } ( a _ { * } ) = \frac { K } { H } \big [ \big ( \frac { \sigma _ { 0 } ( K + 1 ) } { 3 5 2 8 0 0 L R H } \big ) ^ { \alpha } + \big ( \frac { \sigma _ { 1 } } { 6 0 } \big ) ^ { \alpha } \big ] } \end{array}$ (here we use $2 \cdot 1 7 6 4 0 0 = 3 5 2 8 0 0$ and $( 1 7 6 4 0 0 H ^ { 2 } ) ^ { \alpha / 2 } = 4 2 0 ^ { \alpha } H ^ { \alpha } )$ ; hence, $\begin{array} { r } { \Lambda ( \boldsymbol { a } _ { * } ) = \left[ 2 0 ^ { \alpha } \big ( \frac { \sigma _ { 0 } ( \dot { \boldsymbol { K } } + 1 ) } { 3 5 2 8 0 0 L R H } \big ) ^ { \alpha } + \right. } \end{array}$ $\begin{array} { r } { 2 0 ^ { \alpha } \bigl ( \frac { \sigma _ { 1 } } { 6 0 } \bigr ) ^ { \alpha } \bigr ] ^ { - 1 } = \Lambda _ { K } ^ { 0 } \leq \Lambda _ { K } ^ { \mathrm { S } } } \end{array}$ . If $\begin{array} { r } { a _ { * } = c _ { 0 , \alpha } \frac { \sigma _ { 0 } } { L R } ( K + 1 ) K ^ { 1 / \alpha } H ^ { r } } \end{array}$ , then the first term of $\bar { \Psi } _ { K } ( a _ { * } )$ equals $\begin{array} { r } { \frac { 1 } { 2 ^ { \alpha } c _ { 0 , \alpha } ^ { \alpha } } = \frac { q _ { \alpha } } { 2 ^ { \alpha + 1 } } } \end{array}$ , and if $a _ { * } = c _ { 1 , \alpha } \sigma _ { 1 } ^ { 2 } K ^ { 2 / \alpha } H ^ { 2 r }$ , then the second term equals $\begin{array} { r } { \frac { 7 ^ { \alpha } } { c _ { 1 , \alpha } ^ { \alpha / 2 } } = \frac { q _ { \alpha } } { 4 } } \end{array}$ . In both cases, $\bar { \Psi } _ { K } ( a _ { * } ) \geq q _ { \alpha } / 2 ^ { \alpha + 1 }$ , which gives

$$
\Lambda ( a _ { * } ) \leq \frac { 2 ^ { \alpha + 1 } K } { 2 0 ^ { \alpha } H q _ { \alpha } } = \frac { 2 ^ { \alpha + 1 } K } { 2 0 ^ { \alpha } H } \operatorname* { m a x } \left\{ 1 3 5 0 v 3 0 ^ { \alpha - 2 } , 1 5 b 3 0 ^ { \alpha - 1 } \right\} .
$$

The second branch is $\begin{array} { r } { \frac { 2 ^ { \alpha + 1 } \cdot 1 5 \cdot 3 0 ^ { \alpha - 1 } } { 9 0 ^ { \alpha } } \frac { b K } { H } = 3 ^ { \alpha } \frac { b K } { H } \leq 9 b K \leq \Lambda _ { K } ^ { \mathrm { S } } } \end{array}$ , since $H > 1$ . For the first branch, $\begin{array} { r } { v \leq ( \frac { 3 } { 2 } ) ^ { 2 - \alpha } ( 1 + \frac { 2 } { \alpha } \ln ( 1 + \Lambda _ { K } ^ { \mathrm { S } } ) ) } \end{array}$ and $2 ^ { \dot { \alpha } + 1 } \cdot 1 3 5 0 \cdot 3 0 ^ { \alpha - 2 } ( \textstyle { \frac { 3 } { 2 } } ) ^ { 2 - \alpha } / 2 0 ^ { \alpha } = 2 ^ { \alpha + 1 }$ $1 3 5 0 \cdot 2 0 ^ { - 2 } \leq 2 7$ give the bound $2 7 \frac { K } { H } \big ( 1 + \mathrm { \bar { 2 } } \ln ( 1 + \Lambda _ { K } ^ { \mathrm { \bar { S } } } ) \big )$ . Since $b = b _ { \alpha } \geq b _ { 2 } = 2$ under Assumption 3.5, we have $\Lambda _ { K } ^ { \mathrm { S } } \geq 4 0 0 K$ and, since $\bar { \Lambda } \mapsto \Lambda / ( 1 { + } 2 \ln ( 1 { + } \Lambda ) )$ is increasing, it is suficient to check that 27K $( 1 + 2 \ln ( 1 + 4 0 0 K ) ) \leq$ 400KH, $\mathrm { i . e . , ~ } 2 7 ( 1 + 2 \ln ( 1 + 4 0 0 K ) ) \leq 4 0 0 H$ . Since $H \geq \ln ( 4 K )$ , this holds at $K = 1 \ ( 3 5 1 < 5 5 4 )$ and the diference $4 0 0 \ln ( 4 K ) - 2 7 ( 1 + 2 \ln ( 1 + 4 0 0 K ) )$ is increasing in $K \geq 1$ , since its derivative equals $4 0 0 / K - 2 1 6 0 0 / ( 1 + 4 0 0 K ) > 0$ . This proves (80).

Step 4: bounded predictable vectors. Before we proceed, we introduce a new notation: on the whole probability space, we define $w _ { t } : = x ^ { * } - z ^ { t } + \omega _ { t + 1 } \nabla f ( x ^ { t + 1 } )$ and $\eta _ { t } : = w _ { t } \mathbf { 1 } _ { \{ \| w _ { t } \| \le 3 R \} }$ . The vector $\eta _ { t }$ is G<sub>t</sub>-measurable and $\| \eta _ { t } \| \le 3 R$ almost surely. On $E _ { T - 1 }$ , by (75), we have $\| w _ { t } \| \leq$ $2 R + R / ( 6 0 H ) < 3 R$ for $t < T$ , and, therefore, $\eta _ { t } = w _ { t }$ on this event.

Step 5: decomposition of $B _ { T }$ . Let $Z _ { t } : = \| \theta _ { t + 1 } ^ { u } \| ^ { 2 }$ . On $E _ { T - 1 }$ , using $\Vert u + v \Vert ^ { 2 } \leq 2 \Vert u \Vert ^ { 2 } + 2 \Vert v \Vert ^ { 2 }$ we get $B _ { T } \leq \textcircled { \mathfrak { 1 } } + \textcircled { \mathfrak { 2 } } + \textcircled { \mathfrak { 3 } } + \textcircled { \mathfrak { 4 } } + \textcircled { \mathfrak { 5 } }$ , where the following quantities are defined globally:

$$
\begin{array} { r l } & { \displaystyle \mathfrak { O } : = \sum _ { t < T } \omega _ { t + 1 } \left. \theta _ { t + 1 } ^ { u } , \eta _ { t } \right. , \qquad \mathfrak { O } : = \sum _ { t < T } \omega _ { t + 1 } \left. \theta _ { t + 1 } ^ { b } , \eta _ { t } \right. , \qquad \mathfrak { O } : = 2 \sum _ { t < T } \omega _ { t + 1 } ^ { 2 } \left( Z _ { t } - \mathbb { E } [ Z _ { t } \mid \mathcal { G } _ { t } ] \right) , } \\ & { \qquad \mathfrak { O } : = 2 \sum _ { t < T } \omega _ { t + 1 } ^ { 2 } \mathbb { E } [ Z _ { t } \mid \mathcal { G } _ { t } ] , \qquad \mathfrak { O } : = 2 \sum _ { t < T } \omega _ { t + 1 } ^ { 2 } \lVert \theta _ { t + 1 } ^ { b } \rVert ^ { 2 } . } \end{array}
$$

That is, it remains to derive the upper bounds for $\textcircled{1} - \textcircled{5}$

Step 6: upper bound for ① (the linear martingale term). To estimate this sum, we will use Bernstein’s inequality in the martingale form (Lemma E.3). The summands $X _ { t } : = \omega _ { t + 1 } \left. \theta _ { t + 1 } ^ { u } , \eta _ { t } \right.$ have conditional expectations equal to zero given $\mathcal { G } _ { t }$ , and they are bounded globally:

$$
| X _ { t } | \leq 6 R \omega _ { t + 1 } \lambda _ { t } = \frac { R ^ { 2 } } { 5 H } = : c _ { 1 } , \qquad b _ { 1 } : = \frac { R ^ { 2 } } { 5 } , \qquad G : = \frac { R ^ { 4 } } { 1 5 0 H } .
$$

Finally, the conditional quadratic variation satisfies $\begin{array} { r } { V _ { 1 } : = \sum _ { \boldsymbol { t } < \boldsymbol { T } } \mathbb { E } [ X _ { t } ^ { 2 } \mid \mathcal { G } _ { t } ] \le 9 R ^ { 2 } S _ { T } } \end{array}$ , where $S _ { T } : =$ $\textstyle \sum _ { t < T } \omega _ { t + 1 } ^ { 2 } \mathbb { E } [ Z _ { t } \mid \mathcal { G } _ { t } ]$ . By the choice of $\lambda _ { t } ,$ , we have $\omega _ { t + 1 } ^ { 2 } \lambda _ { t } ^ { 2 - \alpha } \overline { { = } } \tilde { R } ^ { 2 - \alpha } 3 0 ^ { \alpha - 2 } \dot { H } ^ { \alpha - 2 } \omega _ { t + 1 } ^ { \alpha }$ , and, therefore, by (76), on $E _ { T - 1 }$

$$
S _ { T } \leq R ^ { 2 - \alpha } 3 0 ^ { \alpha - 2 } H ^ { \alpha - 2 } \sum _ { t < T } \omega _ { t + 1 } ^ { \alpha } D _ { t } { \frac { \phi _ { \lambda _ { t } } ( D _ { t } ) } { D _ { t } } } \qquad ( \mathrm { t e r m s ~ w i t h ~ } D _ { t } = 0 \mathrm { ~ a r e ~ } 0 ) .
$$

If $\bar { \Psi } _ { K } ( a ) = 0$ , then $D _ { t } ~ = ~ 0$ for every $t \ < \ T$ on $E _ { T - 1 }$ , and (76) gives $S _ { T } ~ = ~ 0$ . The desired bound on $S _ { T }$ is therefore immediate in this case. In the remaining estimates, assume $\bar { \Psi } _ { K } ( a ) > 0$ so that the budget used in Lemma D.6 is positive. Under Assumption 3.4, we have $\phi _ { \lambda _ { t } } ( D _ { t } ) =$ $v D _ { t }$ with $v ~ = ~ v _ { \alpha } ^ { \mathrm { m o m } }$ (also when $D _ { t } \ = \ 0 )$ , which gives $\begin{array} { r } { \sum _ { t < T } \omega _ { t + 1 } ^ { \alpha } \phi _ { \lambda _ { t } } ( D _ { t } ) = v \sum _ { t < T } \omega _ { t + 1 } ^ { \alpha } D _ { t } = } \end{array}$ $v R ^ { \alpha } H ^ { 1 - \alpha } \Psi _ { T }$ . Under Assumption 3.5, let $q _ { t } : = \omega _ { t + 1 } ^ { \alpha } D _ { t } ;$ ; since $\omega _ { t + 1 } \lambda _ { t } = R / ( 3 0 H )$ , for $D _ { t } > 0$ , we have $( 3 \lambda _ { t } / 2 ) ^ { \alpha } / D _ { t } = C _ { \mathrm { S } } / q _ { t }$ with $C _ { \mathrm { S } } : = ( R / ( 2 0 H ) ) ^ { \alpha }$ , and, therefore, $\omega _ { t + 1 } ^ { \alpha } \phi _ { \lambda _ { t } } ( D _ { t } ) = q _ { t } v _ { \alpha } ( C _ { \mathrm { S } } / q _ { t } )$ When $q _ { t } = 0$ , we define the latter product as 0, consistently with $\phi _ { \lambda _ { t } } ( 0 ) = 0$ . By Lemma D.6 with $\begin{array} { r } { \bar { B } : = R ^ { \alpha } H ^ { 1 - \alpha } \bar { \Psi } _ { K } ( a ) \geq \sum _ { t < T } q _ { t } = R ^ { \alpha } H ^ { 1 - \alpha } \Psi _ { T } \mathrm { ~ ( b y ~ ( 7 9 ) ) } , T \leq K } \end{array}$ , and the monotonicity of $v _ { \alpha }$ , we get

$$
\sum _ { t < T } q _ { t } v _ { \alpha } \bigg ( \frac { C _ { \mathrm { S } } } { q _ { t } } \bigg ) \leq \bar { B } v _ { \alpha } \bigg ( \frac { K C _ { \mathrm { S } } } { \bar { B } } \bigg ) = R ^ { \alpha } H ^ { 1 - \alpha } \bar { \Psi } _ { K } ( a ) v _ { \alpha } ( \Lambda ( a ) ) ,
$$

since $K C _ { \mathrm { S } } / \bar { B } = K / ( 2 0 ^ { \alpha } H \bar { \Psi } _ { K } ( a ) ) = \Lambda ( a )$ . Moreover, $u \mapsto u v _ { \alpha } ( K C _ { \mathrm { S } } / u )$ is non-decreasing (see Lemma D.6) and $\bar { \Psi } _ { K } ( a ) \leq \bar { \Psi } _ { K } ( a _ { * } )$ for $a \geq a _ { * }$ , implying $\begin{array} { r } { \bar { \Psi } _ { K } ( a ) v _ { \alpha } ( \Lambda ( a ) ) \leq \bar { \Psi } _ { K } ( a _ { * } ) v _ { \alpha } ( \Lambda ( a _ { * } ) ) \leq \frac { q _ { \alpha } } { 2 } v } \end{array}$ by (79) and (80). In both cases, by the definition of $q _ { \alpha } .$ , we have

$$
S _ { T } \leq v 3 0 ^ { \alpha - 2 } \frac { R ^ { 2 } } { H } \cdot \frac { q _ { \alpha } } { 2 } \leq \frac { R ^ { 2 } } { 2 7 0 0 H } \leq \frac { R ^ { 2 } } { 1 3 5 0 H } , \qquad V _ { 1 } \leq \frac { R ^ { 4 } } { 1 5 0 H } = G .\tag{81}
$$

Since ${ \frac { b _ { 1 } ^ { 2 } } { 2 G { + } 2 c _ { 1 } b _ { 1 } / 3 } } = H$ , Lemma E.3 gives

$$
\mathbb { P } \left\{ \left| \mathfrak { O } \right| > \frac { R ^ { 2 } } { 5 } , \ V _ { 1 } \le G \right\} \le 2 e ^ { - H } = \frac { \beta } { 2 K } .\tag{82}
$$

Step 7: upper bound for $\textcircled{2}$ (the systematic bias). On $E _ { T - 1 }$ , by (76), the choice of $\lambda _ { t } .$ and (79), we have

$$
| \textcircled { 2 } | \leq 3 R \sum _ { t < T } \omega _ { t + 1 } \| \theta _ { t + 1 } ^ { b } \| \leq 3 R b \sum _ { t < T } \omega _ { t + 1 } D _ { t } \lambda _ { t } ^ { 1 - \alpha } = 3 b 3 0 ^ { \alpha - 1 } R ^ { 2 } \Psi _ { T } \leq \frac { R ^ { 2 } } { 5 } ,\tag{83}
$$

where the last step uses the second term in the definition of $q _ { \alpha }$ . In particular, $\begin{array} { r } { \sum _ { t < T } \omega _ { t + 1 } \Vert \theta _ { t + 1 } ^ { b } \Vert \leq } \end{array}$ $R / 1 5$

Step 8: upper bound for $\textcircled{3}$ (the quadratic martingale term). Let $Y _ { t } : = 2 \omega _ { t + 1 } ^ { 2 } \mathopen { } \mathclose \bgroup ( Z _ { t } - \mathbb { E } [ Z _ { t } \mid$ $\mathcal { G } _ { t } ] )$ . Since $0 \leq Z _ { t } \leq 4 \lambda _ { t } ^ { 2 }$ globally, the summands are bounded:

$$
| Y _ { t } | \leq 8 \omega _ { t + 1 } ^ { 2 } \lambda _ { t } ^ { 2 } = \frac { 8 R ^ { 2 } } { 9 0 0 H ^ { 2 } } \leq \frac { R ^ { 2 } } { 5 H } = c _ { 1 } .
$$

Moreover, $\mathbb { E } [ Y _ { t } ^ { 2 } \mid \mathcal { G } _ { t } ] = 4 \omega _ { t + 1 } ^ { 4 } \operatorname { V a r } ( Z _ { t } \mid \mathcal { G } _ { t } ) \leq 4 \omega _ { t + 1 } ^ { 4 } \mathbb { E } [ Z _ { t } ^ { 2 } \mid \mathcal { G } _ { t } ] \leq 1 6 \omega _ { t + 1 } ^ { 4 } \lambda _ { t } ^ { 2 } \mathbb { E } [ Z _ { t } \mid \mathcal { G } _ { t } ]$ . Since $\omega _ { t + 1 } ^ { 2 } \lambda _ { t } ^ { 2 } =$ $R ^ { 2 } / ( 9 0 0 H ^ { 2 } )$ , on $E _ { T - 1 }$ , we have

$$
V _ { 3 } : = \sum _ { t < T } \mathbb { E } [ Y _ { t } ^ { 2 } \mid \mathcal { G } _ { t } ] \leq \frac { 1 6 R ^ { 2 } } { 9 0 0 H ^ { 2 } } S _ { T } \leq \frac { 1 6 R ^ { 4 } } { 1 2 1 5 0 0 0 H ^ { 3 } } \leq \frac { R ^ { 4 } } { 1 5 0 H } = G ,
$$

where the last inequality uses $H > 1$ . Lemma E.3 with the same $b _ { 1 } , c _ { 1 } , G$ gives

$$
\mathbb { P } \left\{ | \textcircled { 3 } | > \frac { R ^ { 2 } } { 5 } , \ V _ { 3 } \le G \right\} \le \frac { \beta } { 2 K } .\tag{84}
$$

Step 9: upper bounds for $\textcircled{4}$ and $\textcircled{5}$ . On $E _ { T - 1 }$ , by (81), we have $\begin{array} { r } { \textcircled { 4 } = 2 S _ { T } \leq \frac { R ^ { 2 } } { 6 7 5 H } } \end{array}$ . By Step 7 and the non-negativity of the summands, we also have $\begin{array} { r } { \mathfrak { G } \le 2 \left( \sum _ { t < T } \omega _ { t + 1 } \| \theta _ { t + 1 } ^ { b } \| \right) ^ { 2 } \le \frac { 2 R ^ { 2 } } { 2 2 5 } } \end{array}$

Step 10: closing the induction and the final bound. That is, we derived the upper bounds for ①–⑤. Let $E _ { \mathbb { 0 } } : = \{ V _ { 1 } > G \} \cup \{ | \mathbb { O } | \leq R ^ { 2 } / 5 \}$ and $E _ { \mathfrak { d } } : = \{ V _ { 3 } > G \} \cup \{ | \mathfrak { d } | \le R ^ { 2 } / 5 \}$ . By (82) and (84), each of these events has probability at least $1 - \beta / ( 2 K )$ . On $E _ { T - 1 } .$ , we have already shown that $V _ { 1 } , V _ { 3 } \leq G$ . Thus, $E _ { T - 1 } \cap E _ { \mathbb { 0 } } \cap E _ { \mathbb { 3 } }$ implies, by Steps 5–9 and $H > 1$

$$
B _ { T } \leq \left( \frac { 3 } { 5 } + \frac { 1 } { 6 7 5 } + \frac { 2 } { 2 2 5 } \right) R ^ { 2 } = \frac { 4 1 2 } { 6 7 5 } R ^ { 2 } < R ^ { 2 } .
$$

By Step 1, Lemma I.4 can be applied up to step $T .$ . Since $f ( y ^ { T } ) \geq f ^ { * }$ , it gives $\| z ^ { T } - x ^ { * } \| ^ { 2 } \leq$ $\| z ^ { 0 } - x ^ { * } \| ^ { 2 } + 2 B _ { T } < 3 R ^ { 2 } < 4 R ^ { 2 }$ . Therefore, $E _ { T - 1 } \cap E _ { \mathbb { 0 } } \cap E _ { \mathbb { 3 } } \subseteq E _ { T }$ , and the union bound, which does not require the independence of the events, gives

$$
\mathbb { P } ( E _ { T } ) \geq 1 - \frac { ( T - 1 ) \beta } { K } - \frac { \beta } { 2 K } - \frac { \beta } { 2 K } = 1 - \frac { T \beta } { K } .
$$

This finishes the induction. Finally, $E _ { K }$ implies, via the recursions, that all points belong to $B _ { 2 R } ( x ^ { * } )$ , and Lemma I.4 with $z = x ^ { * }$ and $B _ { K } \le R ^ { 2 }$ gives

$$
A _ { K } \left( f ( y ^ { K } ) - f ^ { * } \right) \leq { \frac { 1 } { 2 } } R ^ { 2 } + B _ { K } \leq { \frac { 3 } { 2 } } R ^ { 2 } .
$$

Substituting $A _ { K } = K ( K + 3 ) / ( 4 a L )$ from (60), we get (62). For $a = a _ { * } ( K , \beta )$ , using $K ( K + 3 ) \geq K ^ { 2 }$ $K + 1 \le 2 K$ , and $1 - r = 1 / \alpha .$ , we get

$$
\frac { 6 a _ { * } L R ^ { 2 } } { K ( K + 3 ) } \leq \operatorname* { m a x } \left\{ \frac { 1 0 5 8 4 0 0 L R ^ { 2 } H ^ { 2 } } { K ^ { 2 } } , \frac { 1 2 c _ { 0 , \alpha } \sigma _ { 0 } R H ^ { r } } { K ^ { r } } , \frac { 6 c _ { 1 , \alpha } \sigma _ { 1 } ^ { 2 } L R ^ { 2 } H ^ { 2 r } } { K ^ { 2 r } } \right\} ,
$$

which proves (63) and concludes the proof.

## I.6 Oracle Complexity

Lemma I.7 (Inverting the logarithm). Let $M \geq 1$ and $\beta \in \mathsf { \Gamma } ( 0 , 1 ]$ If $\begin{array} { r } { K \ge 4 M \ln { \frac { 4 M } { \beta } } } \end{array}$ then ln $\begin{array} { r } { \frac { 4 K } { \beta } \leq \frac { K } { M } } \end{array}$

Proof. Let $s : = 4 M / \beta \geq 4$ and $K _ { 0 } : = 4 M \ln s \geq M$ . The function $\varphi ( K ) : = K / M - \ln ( 4 K / \beta )$ is non-decreasing for $K \geq M .$ , since $\varphi ^ { \prime } ( K ) = M ^ { - 1 } - K ^ { - 1 } \geq 0$ . Therefore, it is suficient to check the case $K = K _ { 0 } \colon \varphi ( K _ { 0 } ) \geq 0 \iff$ ln(4s ln s) ≤ 4 ln s ⇐⇒ 4 ln $s \leq s ^ { 3 }$ . The last inequality holds for $s \geq 4$ , since the function $s ^ { 3 } - 4 \ln { s }$ is positive at $s = 4$ and has positive derivative $3 s ^ { 2 } - 4 / s$ on $[ 4 , \infty )$ , which concludes the proof. □

Proof of Corollary I.2. By Lemma I.7, we have $H / K \ \leq \ M _ { \varepsilon } ^ { - 1 }$ . By the definition of $M _ { \varepsilon } .$ $C _ { \alpha } L R ^ { 2 } M _ { \varepsilon } ^ { - 2 } \ \leq \ \varepsilon , \ C _ { \alpha } \sigma _ { 0 } R M _ { \varepsilon } ^ { - r } \ \leq \ \varepsilon ,$ , and ${ \cal C } _ { \alpha } \sigma _ { 1 } ^ { 2 } L R ^ { 2 } M _ { \varepsilon } ^ { - 2 r } \leq \varepsilon$ . Since each of the three constants in (63) is at most $C _ { \alpha } ,$ substituting these inequalities into (63), we get $f ( y ^ { K } ) - f ^ { * } \leq \varepsilon$ . To count the oracle calls, we take $K = \lceil 4 M _ { \varepsilon } \ln ( 4 M _ { \varepsilon } / \beta ) \rceil$ ; the number of calls equals the number of steps K, which concludes the proof. □

Remark I.8 (Why the choice of a matters). Corollary I.2 is not a statement about all $a \ge$ $a _ { * } ( K , \beta )$ . Consider $\beta = 1 / 2 , f ( x ) = L x ^ { 2 } / 2 , x ^ { * } = 0 , x ^ { 0 } = R > 0$ , an exact oracle, and $\sigma _ { 0 } = \sigma _ { 1 } = 0$ For $a \ge \operatorname* { m a x } \{ a _ { * } ( K , \beta ) , K ( K + 3 ) \}$ , we have $L A _ { K } \le 1 / 4$ . By induction, all points $x ^ { k + 1 } , y ^ { k } , z ^ { k }$ up to step K belong to the segment between $3 R / 4$ and $R \colon$ while the points belong to $[ 0 , R ]$ , the clipped gradient belongs to $[ 0 , L R ]$ and $\begin{array} { r } { z ^ { k + 1 } = R - \sum _ { i = 0 } ^ { k } \omega _ { i + 1 } \mathbf { c } \mathbf { 1 } \mathrm { i p } ( L x ^ { j + 1 } , \lambda _ { j } ) \geq R - L R \bar { A } _ { k + 1 } \geq 3 R / 4 } \end{array}$ The upper bound for $z ^ { k + 1 }$ and both bounds for x and $y$ follow from the recursions. Thus, $f ( y ^ { K } ) - f ^ { * } \geq 9 L R ^ { 2 } / 3 2$ , and the accuracy $\varepsilon = L R ^ { 2 } / 8$ is not achieved, no matter how large K is. Due to this reason, the corollary fixes $a = a _ { * } ( K , \beta )$

## I.7 Comparison with a Uniform Noise Scale

Corollary I.9 (Uniform noise scale). Let the assumptions of Theorem I.1 hold and let $\bar { \sigma } : =$ $( \sigma _ { 0 } ^ { \alpha } + ( 3 L R \sigma _ { 1 } ) ^ { \alpha } ) ^ { 1 / \alpha }$ Then the noise assumption in force also holds on Q with the parameters $( \bar { \sigma } , 0 )$ . As a consequence, Algorithm 2 with $a = a _ { \mathrm { u n i } } : = \operatorname* { m a x } \{ 1 7 6 4 0 0 H ^ { 2 } , c _ { 0 , \alpha } \frac { \bar { \sigma } } { L R } ( K + 1 ) K ^ { 1 / \alpha } H ^ { r } \}$ (where, under Assumption 3.5, v and $\Lambda _ { K } ^ { 0 }$ in ${ \mathit { c } } _ { 0 , \alpha }$ are computed with the parameters $( \bar { \sigma } , 0 ) )$ and $\lambda _ { k }$ defined as in Theorem I.1 satisfies, with probability at least $1 - \beta$

$$
f ( y ^ { K } ) - f ^ { * } \leq C _ { \alpha } \operatorname* { m a x } \left\{ L R ^ { 2 } \left( \frac { H } { K } \right) ^ { 2 } , \bar { \sigma } R \left( \frac { H } { K } \right) ^ { r } \right\} , \qquad C _ { \alpha } : = \operatorname* { m a x } \{ 1 0 5 8 4 0 0 , 1 2 c _ { 0 , \alpha } \} .\tag{85}
$$

Moreover, with $a : =$ min $\{ a _ { * } ( K , \beta ) , a _ { \mathrm { u n i } } \}$ , with probability at least $1 - \beta$ , the localization from Theorem I.1 holds and $\begin{array} { r } { f ( y ^ { K } ) - f ^ { * } \leq \frac { 6 L R ^ { 2 } } { K ( K + 3 ) } } \end{array}$ min $\{ a _ { * } ( K , \beta ) , a _ { \mathrm { u n i } } \}$

Proof. By (68), we have $\| \nabla f ( x ) \| \le 3 L R$ on $Q ,$ and, therefore, $\sigma _ { 0 } ^ { \alpha } + \sigma _ { 1 } ^ { \alpha } \| \nabla f ( x ) \| ^ { \alpha } \leq \bar { \sigma } ^ { \alpha }$ for all $x \in Q$ , and both (2) and (3) hold with $( \bar { \sigma } , 0 )$ . Theorem I.1 with the parameters $( \bar { \sigma } , 0 )$ has the threshold $a _ { \mathrm { u n i } }$ and gives (85). To prove the last claim, we notice that a is deterministic. If it equals $a _ { * } ( K , \beta )$ , we apply Theorem I.1 with $( \sigma _ { 0 } , \sigma _ { 1 } ) ;$ otherwise, we apply it with $( \bar { \sigma } , 0 )$ . In both cases, (62) holds with probability at least $1 - \beta ,$ , and no additional union bound is needed, which concludes the proof. □

The factor 3 in $\bar { \sigma }$ ensures that the tail condition holds on the whole set $Q ,$ on which the assumptions of Theorem I.1 are imposed. Since all queries are localized in $B _ { 2 R } ( x ^ { * } )$ , repeating the probabilistic argument would allow a smaller factor, but this is not needed for the comparison of the orders. Since $\bar { \sigma } \leq \sigma _ { 0 } + 3 L R \sigma _ { 1 }$ , the multiplicative part contributes $O ( \sigma _ { 1 } L R ^ { 2 } ( H / K ) ^ { r } )$ to (85). Theorem I.1 replaces this term with $\sigma _ { 1 } ^ { 2 } L R ^ { 2 } ( H / K ) ^ { 2 r }$ . The ratio of the two scales is of the order $\sigma _ { 1 } ( H / K ) ^ { r }$ and tends to zero for fixed $\sigma _ { 1 }$ and $\beta$ as $K  \infty ;$ we do not claim that this is a uniform improvement for all K and $\sigma _ { 1 }$ . For $\sigma _ { 0 } > 0$ , the overall order may be determined by the additive term $\sigma _ { 0 } R ( H / K ) ^ { r }$ , meaning that the improvement of the multiplicative part does not necessarily change the leading power. The comparison concerns the polynomial powers of the horizon; at the weak- $L ^ { 2 }$ endpoint, both bounds also contain the tail-dependent logarithmic coeficients $c _ { 0 , 2 } , c _ { 1 , 2 }$ from Theorem I.1.

Comparison with prior analyses. We emphasize that the polynomial rates themselves are not new: for $\sigma _ { 1 } = 0$ , the exponents of $K$ and ε in (63) coincide with the ones from Gorbunov et al. (2020) at $\alpha = 2$ and from Sadiev et al. (2023) for $\alpha \in ( 1 , 2 ]$ , and, for Clip-SGD, with the ones from Sadiev et al. (2023) (see Table 1). What changes is the setting in which they hold.

First, the noise class: the weak- $L ^ { \alpha }$ condition, and, at $\alpha = 2 .$ , the Lorentz scale, replace the bounded α-th moment, and, therefore, the rates cover the noise whose α-th moment is infinite.

Second, the afine structure: the gradient-dependent part of the noise enters only through the terms $\sigma _ { 1 } ^ { 2 } L R ^ { 2 } ( H / K ) ^ { 2 r }$ (Clip-SSTM) and $\Delta _ { 1 } ~ ( \mathsf { C l i p { - } S G D } )$ , which decay in K faster than the additive term. Afine noise with finite variance $( \alpha = 2 , \sigma _ { 1 } > 0 )$ is the classical setting from Bottou et al. (2018); related afine conditions appear in the analyses of adaptive stepsizes by Faw et al. (2022); Wang et al. (2023) (afine conditional variance, non-convex criteria) and Attia and Koren (2023) (almost surely bounded afine noise in the main condition, with a sub-Gaussian extension). At $\alpha = 2$ , our multiplicative term has the order $\sigma _ { 1 } ^ { 2 } L R ^ { 2 } / K$ , up to logarithmic factors, which matches the variance term from the finite-variance analyses; the criteria and assumptions difer, meaning that this is a comparison of the orders only.

We do not claim that this term is optimal: we do not provide a lower bound for the multiplicative part, and, under purely multiplicative noise with additional structure, e.g., $\sigma _ { 0 } = 0$ and strong convexity, faster guarantees are known (Gower et al., 2019).

Faster polynomial decay is also possible without strong convexity under a global strong growth condition: under Assumption 3.4 with $\alpha \ : = \ : 2$ and $\sigma _ { 0 } ~ = ~ 0$ , unbiasedness gives $\mathbb { E } \| g ( x , \xi ) \| ^ { 2 } \ =$ $\| \nabla f ( x ) \| ^ { 2 } + \mathbb { E } \| g ( x , \xi ) - \nabla f ( x ) \| ^ { 2 } \leq ( 1 + \sigma _ { 1 } ^ { 2 } ) \| \nabla f ( x ) \| ^ { 2 }$ , and Vaswani et al. (2019, Theorem 2) derive, for an accelerated method under global smoothness and strong growth with constant $\rho = 1 + \sigma _ { 1 } ^ { 2 }$ the bound $\mathbb { E } f ( x ^ { K } ) - f ^ { * } \leq 2 \rho ^ { 2 } L \bar { R ^ { 2 } } / K ^ { 2 }$ ; hence, $2 \rho ^ { 2 } L R ^ { 2 } / ( \beta K ^ { 2 } )$ with probability at least $1 - \beta$ by Markov’s inequality, which is a faster polynomial order at fixed confidence with a much worse dependence on $\beta$ than our logarithmic one. Gupta et al. (2024) also obtain acceleration in expectation for smooth convex and strongly convex problems through AGNES, under global multiplicative finite-variance noise and with arbitrary noise-to-gradient ratios. These results concern diferent algorithms and assumptions and show why our multiplicative term should not be regarded as optimal. An analysis bounding the noise scale uniformly over the ball does not retain the separation of the additive and gradient-dependent components (see Corollary I.9).

Third, localization: the smoothness, convexity, and noise assumptions are imposed only on the ball $Q ,$ and all iterates remain in $B _ { 2 R } ( x ^ { * } )$ on the high-probability event (in Theorem G.1, up to the first iterate in the neighborhood). Fourth, the guarantees are high-probability ones with logarithmic dependence on $1 / \beta$ and with explicit constants and parameter choices; the logarithmic dependence on the horizon in ℓ and H is an artifact of the union bound and can be removed outside the Λ-dependent weak-case tuning (see Appendix I.9).

## I.8 The Lorentz Scale at the Endpoint

In this section, we provide a direct finite-horizon analysis of Clip-SSTM under an afine Lorentz condition at the endpoint $\alpha = 2$ . Since $L ^ { 2 , 2 } = L ^ { 2 }$ and $L ^ { 2 , s } \subseteq L ^ { 2 , \infty }$ , the result extends the finitevariance setting and refines the weak- $L ^ { 2 }$ guarantee; for every $s > 2$ , the inclusion $L ^ { 2 } \subsetneq L ^ { 2 , s }$ is strict, meaning that the assumption allows the noise with infinite second moment. For a real random variable $Y$ with non-increasing rearrangement $Y ^ { * }$ (see Appendix C) and $2 \leq s < \infty$ , we define

$$
\| Y \| _ { 2 , s } : = \left( \int _ { 0 } ^ { 1 } \left[ t ^ { 1 / 2 } Y ^ { * } ( t ) \right] ^ { s } \frac { \mathrm { d } t } { t } \right) ^ { 1 / s } , \qquad \| Y \| _ { 2 , \infty } : = \operatorname* { s u p } _ { 0 < t \leq 1 } t ^ { 1 / 2 } Y ^ { * } ( t ) = \| Y \| _ { L ^ { \infty , \infty } } \left| _ { \alpha = 2 } , \qquad \mathrm { ~ a ~ n ~ d ~ } \right\| _ { L ^ { 1 / 2 } ( ( H - t ) ) }
$$

and let $L ^ { 2 , s }$ be the class of $Y$ with $\| Y \| _ { 2 , s } < \infty ; \| Y \| _ { 2 , 2 } ^ { 2 } = \mathbb { E } Y ^ { 2 }$ . Let $\delta _ { s } : = 1 - 2 / s$ for $s < \infty$ and $\delta _ { \infty } : = 1$

Lemma I.10 (Lorentz embeddings). For $2 \ \leq \ s \ < \ \infty , \ \| Y \| _ { 2 , s } \ \leq \ ( \mathbb { E } Y ^ { 2 } ) ^ { 1 / 2 }$ and $\| Y \| _ { 2 , \infty } ^ { 2 } \leq$ $( s / 2 ) ^ { 2 / s } \| Y \| _ { 2 , s } ^ { 2 } \leq 2 \| Y \| _ { 2 , s } ^ { 2 }$ . On an atomless probability space, $L ^ { 2 } \subsetneq L ^ { 2 , s } \subsetneq L ^ { 2 , \infty }$ for $2 < s < \infty .$

Proof. Let $M ^ { 2 } : = \mathbb { E } Y ^ { 2 } = \textstyle \int _ { 0 } ^ { 1 } Y ^ { * } ( u ) ^ { 2 }$ du and $F ( t ) : = t ^ { 1 / 2 } Y ^ { * } ( t )$ Since $Y ^ { * }$ is non-increasing, we have $\begin{array} { r } { F ( t ) ^ { 2 } \leq \int _ { 0 } ^ { t } Y ^ { * } ( u ) ^ { 2 } } \end{array}$ du $\leq M ^ { 2 } ;$ hence, $\begin{array} { r } { \int _ { 0 } ^ { 1 } F ^ { s } \frac { \mathrm { d } t } { t } \le M ^ { s - 2 } \int _ { 0 } ^ { 1 } F ^ { 2 } \frac { \mathrm { d } t } { t } = M ^ { s } } \end{array}$ . The second claim follows from $\begin{array} { r } { \| Y \| _ { 2 , s } ^ { s } \geq Y ^ { * } ( t ) ^ { s } \int _ { 0 } ^ { t } u ^ { s / 2 - 1 } \mathrm { d } u = \frac { 2 } { s } t ^ { s / 2 } Y ^ { * } ( t ) ^ { s } } \end{array}$ for every t and $( s / 2 ) ^ { 2 / s } = \rho ^ { 1 / \rho }$ with $\rho = s / 2 \ge 1$ is at most $e ^ { 1 / e } < 2$ . To prove the strict inclusions, we consider $Y _ { b } ( t ) : = t ^ { - 1 / 2 } [ \ln ( e / t ) ] ^ { - b }$ with $1 / s < b \leq 1 / 2$ (a non-increasing function on $( 0 , 1 ]$ , which is its own rearrangement up to the single point $t = 1$ , which does not afect the integrals): the substitution $u = \ln ( e / t )$ gives $\begin{array} { r } { \| Y _ { b } \| _ { 2 , s } ^ { s } = \int _ { 1 } ^ { \infty } u ^ { - b s } \mathrm { d } u = ( b s - 1 ) ^ { - 1 } < \infty } \end{array}$ and $\begin{array} { r } { \mathbb { E } Y _ { b } ^ { 2 } = \int _ { 1 } ^ { \infty } u ^ { - 2 b } \mathrm { d } u = \infty ; } \end{array}$ moreover, $t ^ { - 1 / 2 }$ has $\| \cdot \| _ { 2 , \infty } = 1$ but $\begin{array} { r } { \int _ { 0 } ^ { 1 } \frac { \mathrm { d } t } { t } = \infty } \end{array}$ , which concludes the proof. □

Monotonicity of the Lorentz scale. For $2 \leq s _ { 1 } < s _ { 2 } \leq \infty \ ( \mathrm { w i t h \ } 1 / \infty : = 0 )$ , we have

$$
\| Y \| _ { 2 , s _ { 2 } } \leq \left( { \frac { s _ { 1 } } { 2 } } \right) ^ { { \frac { 1 } { s _ { 1 } } } - { \frac { 1 } { s _ { 2 } } } } \| Y \| _ { 2 , s _ { 1 } } .\tag{86}
$$

Indeed, let $\| Y \| _ { 2 , s _ { 1 } } < \infty$ and $h ( t ) : = t ^ { 1 / 2 } Y ^ { \ast } ( t )$ . Since $Y ^ { * }$ is non-increasing, for every $t \in ( 0 , 1 )$ , we have $\begin{array} { r } { \| Y \| _ { 2 , s _ { 1 } } ^ { s _ { 1 } } \ge \int _ { 0 } ^ { t } u ^ { s _ { 1 } / 2 - 1 } Y ^ { * } ( u ) ^ { s _ { 1 } } \mathrm { d } u \ge \frac { 2 } { s _ { 1 } } h ( t ) ^ { s _ { 1 } } } \end{array}$ , implying $\| h \| _ { \infty } \leq ( s _ { 1 } / 2 ) ^ { 1 / s _ { 1 } } \| Y \| _ { 2 , s _ { 1 } }$ , which is the case $s _ { 2 } = \infty$ . For finite $\begin{array} { r } { s _ { 2 } , \int _ { 0 } ^ { 1 } h ^ { s _ { 2 } } \frac { \mathrm { d } t } { t } \leq \| h \| _ { \infty } ^ { s _ { 2 } - s _ { 1 } } \int _ { 0 } ^ { 1 } h ^ { s _ { 1 } } \frac { \mathrm { d } t } { t } } \end{array}$ gives the claim. Thus, the classes $L ^ { 2 , s }$ increase with $s ,$ and a noise bound at the index s<sub>1</sub> with parameters $( \sigma _ { 0 } , \sigma _ { 1 } )$ implies the bound at every index $s _ { 2 } > s _ { 1 }$ with the parameters $( s _ { 1 } / 2 ) ^ { 1 / s _ { 1 } - 1 / s _ { 2 } } ( \sigma _ { 0 } , \sigma _ { 1 } )$ ; the embedding of spaces is not an embedding of balls with the same parameters.

Assumption I.1 (Afine $L ^ { 2 , s }$ noise). There exist $s \in [ 2 , \infty ]$ and $\sigma _ { 0 } , \sigma _ { 1 } \geq 0$ such that for all $x \in Q , \mathbb { E } _ { \xi } [ g ( x , \xi ) ] = \nabla f ( x ) \mathrm { ~ a n d ~ } \| \| g ( x , \xi ) - \nabla f ( x ) \| \big \| _ { 2 , s } ^ { 2 } \leq \sigma _ { 0 } ^ { 2 } + \sigma _ { 1 } ^ { 2 } \| \nabla f ( x ) \| ^ { 2 }$

For $s = 2 .$ this is Assumption 3.4 with $\alpha = 2$ , and, for $s = \infty$ , it is Assumption 3.5 with $\alpha = 2 ;$ the classes increase with s by (86), with the noise parameters transformed by the embedding constant when guarantees at diferent indices are compared, and, for $s > 2$ , the noise may have infinite variance (it is suficient to multiply $Y _ { b }$ by an independent Rademacher sign to obtain unbiased examples). The embedding constants from Lemma I.10 are uniform in $s ,$ but, for a fixed distribution, the quasi-norm $\| Y \| _ { 2 , s }$ is not necessarily uniform in $s \mathrm { : }$ for $Y _ { 1 / 2 }$ , it equals $( s / 2 - 1 ) ^ { - 1 / s }$ which diverges as $s \downarrow 2$ . Therefore, when guarantees for one distribution are compared across $s ,$ the parameters $\sigma _ { 0 } , \sigma _ { 1 }$ must be taken relative to the chosen $s ,$ and their growth may ofset a smaller power of the logarithm.

Lemma I.11 (Truncated second moment on the Lorentz scale). Let $Y \geq 0$ with $\| Y \| _ { 2 , s } ^ { 2 } \le D$ for some $s \in [ 2 , \infty ]$ and $D \geq 0$ . Then, for every $c > 0$

$$
\mathbb { E } \operatorname* { m i n } \{ Y ^ { 2 } , c ^ { 2 } \} \leq 2 D \left[ 1 + \ln \left( 1 + \frac { c ^ { 2 } } { D } \right) \right] ^ { \delta _ { s } } ,\tag{87}
$$

with right-hand side 0 if $D = 0$

Proof. If $D = 0 ;$ , then $Y ~ = ~ 0$ almost surely by Lemma I.10. If $0 ~ < ~ c ^ { 2 } ~ \le ~ D _ { \cdot }$ , then the lefthand side is at most $c ^ { 2 } \leq D$ . Next, let $c ^ { 2 } > D > 0$ and $t _ { 0 } : = D / c ^ { 2 }$ . For $2 < s < \infty$ , by E min $\begin{array} { r } { \{ Y ^ { 2 } , c ^ { 2 } \} = \int _ { 0 } ^ { 1 } } \end{array}$ min $\{ Y ^ { * } ( t ) ^ { 2 } , c ^ { 2 } \}$ dt and H¨older’s inequality with exponents $s / 2$ and $s / ( s - 2 )$ we have

$$
\begin{array} { r l r } {  { \mathbb { E } \operatorname* { m i n } \{ Y ^ { 2 } , c ^ { 2 } \} \le c ^ { 2 } t _ { 0 } + \int _ { t _ { 0 } } ^ { 1 } [ t ^ { 1 / 2 } Y ^ { * } ( t ) ] ^ { 2 } \frac { \mathrm { d } t } { t } \le D + ( \int _ { t _ { 0 } } ^ { 1 } [ t ^ { 1 / 2 } Y ^ { * } ( t ) ] ^ { s } \frac { \mathrm { d } t } { t } ) ^ { 2 / s } [ \ln \frac { 1 } { t _ { 0 } } ] ^ { 1 - 2 / s } } } \\ & { } & { \le D \{ 1 + [ \ln \frac { c ^ { 2 } } { D } ] ^ { \delta _ { s } } \} , } \end{array}
$$

which is at most the right-hand side of (87). For $s \ : = \ : 2$ , we have E min $\{ Y ^ { 2 } , c ^ { 2 } \} \le \mathbb { E } Y ^ { 2 } \le D$ Finally, for $s = \infty ,$ , the tail is at most min $\{ 1 , D / u ^ { 2 } \}$ , and $\begin{array} { r } { 2 \int _ { 0 } ^ { c } u \mathbb { P } \{ Y > u \} \mathrm { d } u \leq D [ 1 + \ln ( c ^ { 2 } / D ) ] } \end{array}$ , as in the proof of Lemma D.1. This concludes the proof. □

Lemma I.12 (Concavity with an exponent). Let $C \geq 0$ and $\delta \in [ 0 , 1 ]$ , and let $\phi _ { \delta } ( q ) : = q [ 1 + \ln ( 1 +$ $C / q ) ] ^ { \delta }$ for $q > 0 , \phi _ { \delta } ( 0 ) : = 0$ . Then $\phi _ { \delta }$ is non-decreasing and concave on $[ 0 , \infty )$ Consequently, $\begin{array} { r } { \sum _ { t < T } \phi _ { \delta } ( q _ { t } ) \leq \bar { B } [ 1 + \ln ( 1 + T C / \bar { B } ) ] ^ { \delta } } \end{array}$ whenever $q _ { t } \geq 0$ and $\begin{array} { r } { \sum _ { t < T } q _ { t } \le \bar { B } , \bar { \bar { B } } > 0 . } \end{array}$

Proof. If $C = 0$ , then $\phi _ { \delta } ( q ) = q$ . Next, let $C > 0 , w : = 1 + \ln ( 1 + C / q ) \geq 1$ , and $\rho : = C / ( q + C ) \in$ $( 0 , 1 )$ ; then $\begin{array} { r } { { \frac { \mathrm { d } w } { \mathrm { d } q } } = - { \frac { \rho } { q } } } \end{array}$ and $\begin{array} { r } { { \frac { \mathrm { d } \rho } { \mathrm { d } q } } = - { \frac { \rho ^ { 2 } } { C } } } \end{array}$ . Then $\phi _ { \delta } ^ { \prime } ( q ) = w ^ { \delta - 1 } ( w - \delta \rho ) > 0$ and

$$
\phi _ { \delta } ^ { \prime \prime } ( q ) = - \frac { \delta \rho ^ { 2 } } { q } w ^ { \delta - 2 } \left( w + 1 - \delta \right) \leq 0 .
$$

Since $\phi _ { \delta } ( q ) \leq q ( 1 + \ln ( 1 + C / q ) ) \to 0 { \mathrm { ~ a s ~ } } q \downarrow 0$ , the function is continuous, non-decreasing, and concave on $[ 0 , \infty )$ , and the last claim follows from Jensen’s inequality with equal weights and monotonicity, as in Lemma $_ { \mathrm { { D . 6 , } } }$ which concludes the proof. □

Proposition I.13 (Clip-SSTM on the Lorentz scale). Let Assumptions 3.1–3.3 hold for $Q =$ $B _ { 3 R } ( x ^ { * } )$ , where $R \geq \| x ^ { 0 } - x ^ { * } \| > 0$ , and let Assumption I.1 hold with some $s \in [ 2 , \infty ]$ Then Theorem I.1 holds with $\alpha = 2$ and the clipping constants

$$
b : = 4 , \qquad v : = v _ { s } ( \Lambda _ { K } ^ { \mathrm { S } } ) , \qquad v _ { s } ( \Lambda ) : = 2 \left[ 1 + \ln ( 1 + \Lambda ) \right] ^ { \delta _ { s } } , \qquad \Lambda _ { K } ^ { \mathrm { S } } = \operatorname* { m a x } \{ 8 0 0 K , \Lambda _ { K } ^ { 0 } \} .
$$

In particular, with $a = a _ { * } ( K , \beta ) , \mathcal { L } _ { K } : = 1 + \ln ( 1 + \Lambda _ { K } ^ { \mathrm { S } } )$ , and $v = 2 \mathcal { L } _ { K } ^ { \delta _ { s } }$ on an event of probability at least $1 - \beta$ , we have

$$
f ( y ^ { K } ) - f ^ { * } \leq \operatorname* { m a x } \left\{ \frac { 1 0 5 8 4 0 0 L R ^ { 2 } H ^ { 2 } } { K ^ { 2 } } , 1 2 \sqrt { 2 7 0 0 } v \sigma _ { 0 } R \sqrt { \frac { H } { K } } , 6 \cdot 4 9 \cdot 5 4 0 0 v \sigma _ { 1 } ^ { 2 } L R ^ { 2 } \frac { H } { K } \right\} ;
$$

in other words, the additive term contains $\mathcal { L } _ { K } ^ { \delta _ { s } / 2 }$ and the multiplicative term contains $\mathcal { L } _ { K } ^ { \delta _ { s } }$

Proof. The proof of Theorem I.1 uses the noise assumption only through the conditional clipping bounds (76), i.e., through the following three properties of $( b , v ) \colon \ ( \mathrm { i } )$ for $\| x \| \le \lambda / 2$ and X with EX $= x , \| \mathbb { E } \mathrm { c } \lambda { \mathrm { i } } \mathrm { p } ( X , \lambda ) - x \| \leq b D / \lambda ; { \mathrm { ~ ( i i ) ~ } } \mathbb { E } \| \mathrm { c } \lambda { \mathrm { i } } \mathrm { p } ( X , \lambda ) - x \| ^ { 2 } \leq \phi _ { \lambda } ( D )$ with $\phi _ { \lambda } ( D ) =$ $D v _ { s } ( ( 3 \lambda / 2 ) ^ { 2 } / D )$ , where $v _ { s }$ is non-decreasing and $q \mapsto q v _ { s } ( C / q )$ is concave and non-decreasing for every $C \geq 0$ (this is exactly what Lemma D.6 provides in Step 6, both for the summation and for the monotonicity in $^ { a , }$ and what makes the right-hand side of (ii) non-decreasing in $D )$ ; (iii) the fixed point (80). Here $D = \sigma _ { 0 } ^ { 2 } + \sigma _ { 1 } ^ { 2 } \| x \| ^ { 2 }$ bounds $\| \| X - x \| \| _ { 2 , s } ^ { 2 } ,$ and the conditional versions follow exactly as in Corollary D.5 and Lemma I.3, since the quasi-norm bound is applied pointwise in x via the freezing formula. To prove (i), we use Lemma I.10, which gives $\mathbb { P } \{ \| X - x \| > u \} \le 2 D / u ^ { 2 }$ $\mathrm { i . e . }$ , the tail condition from Lemma D.1 with $\alpha = 2$ and 2D in place of $A ( x )$ , and, therefore, (18) gives the bias bound $b _ { 2 } \cdot 2 D / \lambda = 4 D / \lambda$ . To prove (ii), we use the contraction (23) and Lemma I.11 with $c = 3 \lambda / 2$ , which give $\mathbb { E } \| \mathsf { c } 1 \mathrm { i p } ( X , \lambda ) - x \| ^ { 2 } \leq \mathbb { E }$ min $\{ \| X - x \| ^ { 2 } , ( 3 \lambda / 2 ) ^ { 2 } \} \le D v _ { s } ( ( 3 \lambda / 2 ) ^ { 2 } / D )$ and the required monotonicity and concavity follow from Lemma I.12 with $\delta = \delta _ { s }$ (the factor 2 is harmless). To prove (iii), we repeat Step $3 ^ { \prime }$ with $b = 4$ and $v \leq 2 ( 1 + \ln ( 1 + \Lambda _ { K } ^ { \mathrm { S } } ) )$ : the second branch is $9 b K = 3 \dot { 6 } K \le \Lambda _ { K } ^ { \mathrm { S } }$ , and the first branch is at most $\begin{array} { r l r } {  { \frac { 8 \cdot 1 \dot { 3 } 5 0 \cdot 2 } { 4 0 0 } \frac { K } { H } \big ( 1 + \ln ( 1 + \Lambda _ { K } ^ { \mathrm { S } } ) \big ) } } \end{array} =$ $5 4 \frac { K } { H } \big ( 1 + \ln ( 1 + \Lambda _ { K } ^ { \mathrm { S } } ) \big )$ ; since $\Lambda _ { K } ^ { \mathrm { S } } \geq 8 0 0 K$ and $\Lambda \mapsto \Lambda / ( 1 + \ln ( 1 + \Lambda ) )$ is increasing, it is suficient to show that $5 4 ( 1 + \ln ( 1 + 8 0 0 K ) ) \leq 8 0 0 H$ , which holds for $K = 1 \ ( 4 1 6 < 1 1 0 9$ , using $H \geq \ln 4 )$ and whose two sides have increasing diference in K (the derivative is $8 0 0 / K - 4 3 2 0 0 / ( 1 + 8 0 0 K ) > 0 )$ The displayed bound is (63) at $\alpha = 2 \colon$ since $v \geq 2 > 4 / 3$ , we have $\begin{array} { r } { q _ { 2 } = \operatorname* { m i n } \{ \frac { 1 } { 1 3 5 0 v } , \frac { 1 } { 1 8 0 0 } \} = \frac { 1 } { 1 3 5 0 v } ; } \end{array}$ hence, $c _ { 0 , 2 } = \sqrt { 2 / q _ { 2 } } = \sqrt { 2 7 0 0 v }$ and $c _ { 1 , 2 } = 4 9 \cdot 4 / q _ { 2 } = 4 9$ · 5400v, which concludes the proof. □

Remark I.14 (Sharpness of the exponent). For $2 < s < \infty .$ the exponent $\delta _ { s }$ in (87) cannot be decreased uniformly over the unit ball of $L ^ { 2 , s }$ and over c: for $T \geq 1$ the non-increasing function $Y _ { T } ( t ) : = c _ { T } \operatorname* { m i n } \{ t ^ { - 1 / 2 } , e ^ { T / 2 } \}$ on (0, 1] (its own rearrangement up to the point $t = 1 )$ with $c _ { T } : = ( T + 2 / s ) ^ { - 1 / s }$ satisfies $\| Y _ { T } \| _ { 2 , s } = 1$ and $\mathbb { E } Y _ { T } ^ { 2 } = c _ { T } ^ { 2 } ( 1 + T )$ , which is the truncated second moment at level $c = c _ { T } e ^ { T / 2 } \geq Y _ { T } ;$ it grows as $\hat { T ^ { 1 - 2 / s } }$ , while $\ln ( 1 + c ^ { 2 } ) \sim T$ For $s = \infty$ , the full logarithm is necessary, see Remark D.3. This sharpness statement concerns the truncated second-moment bound over the Lorentz unit ball. Proposition I.16 below separately establishes an optimization lower bound and proves the necessity of the exponent $\textstyle { \frac { 1 } { 2 } } - { \frac { 1 } { s } }$ in the additive noise term at fixed confidence; it does not establish the optimality of the full logarithmic dependence in the Clip-SSTM guarantee or of its multiplicative term.

Remark I.15 (Comparison with the moment guarantee). The orders in K from Proposition I.13 can also be recovered from Theorem I.1 under Assumption 3.4. For $1 < p < 2 \leq s < \infty$ , writing $\begin{array} { r } { \mathbb { E } | Y | ^ { p } = \int _ { 0 } ^ { 1 } [ t ^ { 1 / 2 } Y ^ { * } ( t ) ] ^ { p } t ^ { 1 - p / 2 } \frac { \mathrm { d } t } { t } } \end{array}$ and applying H¨older’s inequality with the exponents $s / p$ and $s / ( s - p )$ , we get

$$
\| Y \| _ { p } \leq C _ { p , s } \| Y \| _ { 2 , s } , \qquad C _ { p , s } : = \left( { \frac { 2 ( s - p ) } { s ( 2 - p ) } } \right) ^ { { \frac { 1 } { p } } - { \frac { 1 } { s } } } ,\tag{88}
$$

and, for $s = \infty , C _ { p , \infty } = ( 2 / ( 2 - p ) ) ^ { 1 / p }$ (see Proposition C.5). As a consequence, Assumption I.1 implies Assumption 3.4 with the exponent p and the parameters $\left( C _ { p , s } \sigma _ { 0 } , C _ { p , s } \sigma _ { 1 } \right)$ . Choose $p _ { K } : =$ $2 - \operatorname* { m i n } \{ { \textstyle { \frac { 1 } { 2 } } } , { \frac { 1 } { \ln ( e K ) } } \} \in [ { \textstyle { \frac { 3 } { 2 } } } , 2 )$ The constants $c _ { 0 , p } , c _ { 1 , p }$ from Theorem I.1 under Assumption 3.4 are bounded uniformly in $p \in \left[ \frac { 3 } { 2 } , 2 \right)$ We next verify the uniform embedding bound $C _ { p _ { K } , s } ~ \le$ $4 [ \ln ( e K ) ] ^ { 1 / 2 - 1 / s }$

Set $T : = \ln ( e K )$ and $p : = p _ { K } . { \mathrm { ~ H ~ } } 1 \leq T \leq 2$ , then $p = 3 / 2$ , the base in (88) is at most 4, and its exponent is at most $2 / 3$ . Hence $C _ { p , s } \leq 4 ^ { 2 / 3 } \leq 4 T ^ { 1 / 2 - 1 / s }$ . If $T \geq 2$ , the base is at most 2T and $1 / p - 1 / s = 1 / 2 - 1 / s + 1 / ( 2 p T )$ . Since $p \geq 3 / 2$ and ln $( 2 T ) / T$ decreases for $T \geq 2$

$$
C _ { p , s } \leq \sqrt { 2 } T ^ { 1 / 2 - 1 / s } \exp \biggl ( \frac { \ln ( 2 T ) } { 2 p T } \biggr ) \leq \sqrt { 2 } e ^ { \ln 4 / 6 } T ^ { 1 / 2 - 1 / s } < 4 T ^ { 1 / 2 - 1 / s } .
$$

The case $s = \infty$ follows by taking the limit. Also, $\left( 2 - p _ { K } \right)$ ln $K \leq 1$ and $H > 1$ give $K ^ { - ( p _ { K } - 1 ) / p _ { K } } \leq$ $e ^ { 1 / ( 2 p _ { K } ) } K ^ { - 1 / 2 }$ and $H ^ { ( p _ { K } - 1 ) / p _ { K } } \leq H ^ { 1 / 2 }$ Applying Theorem I.1 with its prescribed parameters therefore gives, with probability at least $1 - \beta$

$$
f ( y ^ { K } ) - f ^ { * } \lesssim \frac { L R ^ { 2 } H ^ { 2 } } { K ^ { 2 } } + \sigma _ { 0 } R \sqrt { \frac { H } { K } } \left[ \ln ( e K ) \right] ^ { \frac { 1 } { 2 } - \frac { 1 } { s } } + \sigma _ { 1 } ^ { 2 } L R ^ { 2 } \frac { H } { K } \left[ \ln ( e K ) \right] ^ { 1 - \frac { 2 } { s } } , \qquad H = \ln \frac { 4 K } { \beta } ,
$$

The constant in this bound is absolute. For fixed problem parameters and confidence, this reduction gives the same powers of K and ln K as Proposition I.13. The direct proof provides explicit finite-horizon constants and parameter choices under the original Lorentz condition, with $1 + \ln ( 1 + \Lambda _ { K } ^ { \mathrm { S } } )$ in place of $\mathrm { l n } ( e K )$

Theorem I.18 improves the reduction by replacing $H = \ln ( 4 K / \beta )$ with $h = \ln ( 4 / \beta )$ , yielding Corollary I.19. For fixed problem parameters, fixed confidence, and $\sigma _ { 0 } > 0$ , the additive terms in the direct and improved bounds have orders $\sigma _ { 0 } R K ^ { - 1 / 2 } ( \ln K ) ^ { 1 - 1 / s }$ and $\sigma _ { 0 } R K ^ { - 1 / 2 } ( \ln K ) ^ { 1 / 2 - 1 / s }$ respectively. The improved bound therefore removes a factor $\sqrt { \ln { K } }$ and matches the additive lower bound asymptotically (see Remark I.17).

This improvement does not follow by substituting h directly into the parameters of Proposition I.13. It uses the moment theorem with exponent $p _ { K }$ and noise parameters $\left( C _ { p _ { K } , s } \sigma _ { 0 } , C _ { p _ { K } , s } \sigma _ { 1 } \right)$ For $s \ : = \ : 2$ the finite-variance theorem with exponent 2 is also available. The same momentembedding argument applies to Clip-SGD.

An optimization lower bound on the Lorentz scale. Next, we show that the Lorentz logarithm is necessary for optimization itself, in the additive noise class. The oracle model is the same as in Proposition H.2: the algorithm makes at most K adaptive, possibly randomized, calls to a stochastic gradient oracle, which returns fresh independent samples at successive calls; query points, the decision to stop, and the output are measurable functions of the known class parameters, the previous responses, and an independent random seed; no exact function values are supplied. For a real random variable Y with law $P ,$ we write $\| Y \| _ { 2 , s ; P }$ for the quasi-norm defined above (the rearrangement is taken under $P )$ and omit P when it is clear from the context.

Proposition I.16 (Optimization lower bound under additive $L ^ { 2 , s }$ noise). Let $K \geq 1$ be an integer, $L , R , \sigma _ { 0 } > 0$ , and $s \in [ 2 , \infty ]$ For every algorithm in the above oracle model that makes at most K stochastic gradient calls and returns a point ${ \hat { x } } \in \mathbb { R }$ , there exist a convex L-smooth quadratic f on R and an unbiased stochastic gradient oracle $g ( x , \xi )$ with $| x ^ { 0 } - x ^ { * } | = R$ and $\| g ( x , \xi ) - \nabla f ( x ) \| _ { 2 , s } \ \leq \ \sigma _ { 0 }$ for every $x \in \mathbb { R }$ (the law of the noise does not depend on $x ,$ and Assumption I.1 holds on R with $\left( \sigma _ { 0 } , 0 \right) )$ such that

$$
\mathbb { P } \{ f ( \hat { x } ) - f ^ { * } \geq \frac { 1 } { 6 4 } \operatorname* { m i n } \Big \{ L R ^ { 2 } , \frac { \sigma _ { 0 } R } { \sqrt { K } } [ \ln ( e K ) ] ^ { \frac { 1 } { 2 } - \frac { 1 } { s } } \Big \} \Big \} \geq \frac { 1 } { 3 } ,\tag{89}
$$

where the probability is taken $\mathrm { w . r . t }$ the oracle samples and the internal randomness of the algorithm, which may know $L , R , \sigma _ { 0 } , s$ , and $K$ The constants $1 / 6 4$ and $1 / 3$ do not depend on s.

Proof. Without loss of generality, we assume that $x ^ { 0 } = 0$ (this can be achieved by translation). Let $T : = \ln ( e K ) \geq 1 , a : = \sigma _ { 0 } / 4 , c _ { T } : = ( T + 2 / s ) ^ { - 1 / s }$ for $s ~ < ~ \infty$ , and $c _ { T } : = 1$ for $s ~ = ~ \infty$ Let U be uniformly distributed on $( 0 , 1 )$ and ϵ be an independent Rademacher sign, and define the symmetric reference law $P _ { 0 }$ via $Z _ { 0 } : = a \epsilon c _ { T }$ min $\{ U ^ { - 1 / 2 } , e ^ { T / 2 } \}$ . The rearrangement of $\lvert Z _ { 0 } \rvert$ is $a c _ { T }$ min $\{ t ^ { - 1 / 2 } , e ^ { T / 2 } \}$ for $0 < t < 1$ (its value at the single point $t = 1$ is immaterial), and, therefore, for $s < \infty$

$$
\lVert Z _ { 0 } \rVert _ { 2 , s } ^ { s } = a ^ { s } c _ { T } ^ { s } \left( e ^ { s T / 2 } \int _ { 0 } ^ { e ^ { - T } } t ^ { \frac { s } { 2 } - 1 } \mathrm { d } t + \int _ { e ^ { - T } } ^ { 1 } \frac { \mathrm { d } t } { t } \right) = a ^ { s } c _ { T } ^ { s } \left( \frac { 2 } { s } + T \right) = a ^ { s } ,
$$

and, for $s = \infty$ , the supremum equals $a ;$ similarly,

$$
V : = \mathbb { E } _ { 0 } Z _ { 0 } ^ { 2 } = a ^ { 2 } c _ { T } ^ { 2 } ( 1 + T ) , \qquad M : = \operatorname* { s u p } | Z _ { 0 } | = a c _ { T } e ^ { T / 2 } .
$$

Let $\Delta : =$ min $\{ L R , \textstyle { \frac { 1 } { 8 } } \sqrt { V / K } \} , \vartheta : = \Delta / V$ , and $\mu : = \Delta / R \le L$ . For $v \in \{ - 1 , + 1 \}$ , we define the tilted laws $P _ { v }$ via $\begin{array} { r } { \frac { \mathrm { d } \hat { P } _ { v } } { \mathrm { d } P _ { 0 } } ( z ) : = 1 - v \vartheta z } \end{array}$ . They integrate to one since $\mathbb { E } _ { 0 } Z _ { 0 } = 0 .$ , and they are positive since $\begin{array} { r } { \vartheta M \le \frac { M } { 8 \sqrt { K V } } = \frac { e ^ { T / 2 } } { 8 \sqrt { K ( T + 1 ) } } = \frac { \sqrt { e } } { 8 \sqrt { T + 1 } } < \frac { 1 } { 4 } } \end{array}$ . For $Z \sim P _ { v } ,$ , we have $\mathbb { E } _ { v } Z = - v \vartheta V = - v \Delta$

Noise class. For $u \geq 0 .$ , we have $P _ { v } \{ | Z | > u \} = \mathbb { E } _ { 0 } [ ( 1 - v \vartheta Z _ { 0 } ) \mathbf { 1 } _ { \{ | Z _ { 0 } | > u \} } ] = P _ { 0 } \{ | Z _ { 0 } | > u \}$ , since $z \mathbf { 1 } _ { \{ | z | > u \} }$ is odd and $P _ { 0 }$ is symmetric; hence, the rearrangement of $| Z |$ under $P _ { v }$ coincides with the one of $\left| Z _ { 0 } \right|$ under $P _ { 0 }$ , and $\| Z \| _ { 2 , s ; P _ { v } } = a$ $\mathrm { N e x t } , c _ { T } \leq 1$ and $( T + 1 ) / K = ( \ln K + 2 ) / K \leq 2$ for $K \geq 1$ give $\begin{array} { r } { \Delta \leq \frac { a c _ { T } } { 8 } \sqrt { ( T + 1 ) / K } \leq \frac { \sqrt { 2 } } { 8 } a < \frac { a } { 4 } } \end{array}$ . Since $| Z + v \Delta | \le | Z | + \Delta$ , we have $P _ { v } \{ | Z + v \Delta | >$ $u + \Delta \} \le P _ { v } \{ | Z | > u \}$ for all $u \ge 0 , \mathrm { i . e . , ~ } ( \bar { Z } + v \Delta ) ^ { * } ( t ) \le Z ^ { * } ( t ) + \Delta$ for $0 < t < 1$ . For $s < \infty$ Minkowski’s inequality in $L ^ { s } ( ( 0 , 1 ) , \mathrm { d } t / t )$ gives

$$
\| Z + v \Delta \| _ { 2 , s ; P _ { v } } \leq \| Z \| _ { 2 , s ; P _ { v } } + \Delta \left( \int _ { 0 } ^ { 1 } t ^ { \frac { s } { 2 } } \frac { \mathrm { d } t } { t } \right) ^ { 1 / s } = a + \left( \frac { 2 } { s } \right) ^ { 1 / s } \Delta \leq a + \Delta ,
$$

and, for $s = \infty$ , the same bound follows by taking suprema. Thus, $\begin{array} { r } { \| Z + v \Delta \| _ { 2 , s ; P _ { v } } \le a + \Delta < \frac { 5 } { 4 } a = } \end{array}$ $\frac { 5 } { 1 6 } \sigma _ { 0 } < \sigma _ { 0 }$

Instances. Let $f _ { v } ( x ) : = { \textstyle \frac { \mu } { \gamma } } ( x - v R ) ^ { 2 }$ and $\nabla f _ { v , Z } ( x ) : = \mu x + Z$ with $Z \sim P _ { v }$ (as a stochastic function, $\begin{array} { r } { f _ { v , Z } ( x ) = \frac { \mu } { 2 } x ^ { 2 } + Z x \mathbf { \bar { + } } \frac { \mu } { 2 } R ^ { 2 } } \end{array}$ , and $\mathbb { E } _ { v } f _ { v , Z } = f _ { v } )$ . Both objectives are convex and L-smooth on R, with minimizers $v R$ and $f _ { v } ^ { * } = 0 ; \mathbb { E } _ { v } [ \mu x + Z ] = \mu ( x - v R ) = \nabla f _ { v } ( x )$ , and the noise is $\nabla f _ { v , Z } ( x ) - \nabla f _ { v } ( x ) = Z + v \Delta$ , whose law does not depend on x and satisfies the required bound by the previous step.

Indistinguishability. With $P _ { \pm } : = P _ { \pm 1 }$ and $\begin{array} { r } { \chi ^ { 2 } ( P \| Q ) : = \int ( \mathrm { d } P / \mathrm { d } Q - 1 ) ^ { 2 } \mathrm { d } Q } \end{array}$ , we have

$$
\chi ^ { 2 } ( P _ { + } \| P _ { - } ) = \mathbb { E } _ { 0 } \frac { 4 \vartheta ^ { 2 } Z _ { 0 } ^ { 2 } } { 1 + \vartheta Z _ { 0 } } \le \frac { 1 6 } { 3 } \frac { \Delta ^ { 2 } } { V } \le \frac { 1 } { 1 2 K } ,
$$

by $1 + \vartheta Z _ { 0 } \geq 3 / 4$ and $\Delta ^ { 2 } \le V / ( 6 4 K )$ . By independence, $1 + \chi ^ { 2 } ( P _ { + } ^ { \otimes K } | | P _ { - } ^ { \otimes K } ) = ( 1 + \chi ^ { 2 } ( P _ { + } | | P _ { - } ) ) ^ { K } \leq$ $e ^ { 1 / 1 2 }$ , and the Cauchy–Schwarz inequality gives $\begin{array} { r } { \mathrm { T V } ( P _ { + } ^ { \otimes K } , P _ { - } ^ { \otimes K } ) \leq \frac { 1 } { 2 } \sqrt { e ^ { 1 / 1 2 } - 1 } < \frac { 1 } { 4 } } \end{array}$ . The response to a query x is $\mu x + Z$ with $\mu$ common for both hypotheses, meaning that the whole transcript and the output are the same measurable function of K independent observations $Z _ { 1 } , \dots , Z _ { K }$ (unused ones are supplied to the algorithms with fewer calls) and of the seed under both hypotheses; hence, the output laws also have total variation smaller than $1 / 4$ . Let $A : = \{ \hat { x } > 0 \}$ . Then $\mathbb { P } _ { + } ( A ^ { c } ) + \mathbb { P } _ { - } ( A ) = 1 - ( \mathbb { P } _ { + } ( A ) - \mathbb { P } _ { - } ( A ) ) > \frac { 3 } { 4 }$ , meaning that one of the two error probabilities exceeds $\textstyle { \frac { 3 } { 8 } } > { \frac { 1 } { 3 } }$ . On an error under hypothesis v, we have $| \hat { x } - v R | \ge R$ and $\begin{array} { r } { f _ { v } ( \hat { x } ) - f _ { v } ^ { * } \ge \frac { \mu R ^ { 2 } } { 2 } = \frac { \Delta R } { 2 } } \end{array}$ Finally, $c _ { T } \geq ( T + 1 ) ^ { - 1 / s }$ gives $\sqrt { V } = a c _ { T } \sqrt { T + 1 } \ge a ( T + 1 ) ^ { \frac { 1 } { 2 } - \frac { 1 } { s } } \ge a T ^ { \frac { 1 } { 2 } - \frac { 1 } { s } }$ (also for $s = \infty )$ , and, therefore,

$$
\frac { \Delta R } { 2 } = \operatorname * { m i n } \left\{ \frac { L R ^ { 2 } } { 2 } , \frac { R \sqrt { V } } { 1 6 \sqrt { K } } \right\} \geq \frac { 1 } { 6 4 } \operatorname * { m i n } \left\{ L R ^ { 2 } , \frac { \sigma _ { 0 } R } { \sqrt { K } } T ^ { \frac { 1 } { 2 } - \frac { 1 } { s } } \right\} .
$$

Remark I.17 (Scope of the optimization lower bound). For fixed $L , R , \sigma _ { 0 } > 0$ and large $K .$ , the stochastic term in (89) is the smaller one, and, therefore, at any fixed failure probability $\beta < 1 / 3$ no uniform guarantee over the additive $L ^ { 2 , s }$ class can improve the order $\sigma _ { 0 } R [ \ln ( e K ) ] ^ { 1 / 2 - 1 / s } / \sqrt { K }$ this recovers the finite-variance order at $s = 2$ , proves the necessity of the Lorentz logarithm for every $s > 2$ , and gives the factor $\sqrt { \ln ( e K ) }$ at $s = \infty$ Unlike Remark I.14, it is a statement about the optimization error of every algorithm in the stated oracle model. Nevertheless, it does not establish the optimality of the whole Clip-SSTM guarantee: the additive term from Proposition I.13 contains $\sqrt { H } \mathcal { L } _ { K } ^ { 1 / 2 - 1 / s }$ , and, for fixed $\therefore R , \sigma _ { 0 } > 0 , \sigma _ { 1 } = 0$ , and fixed $\beta , \Lambda _ { K } ^ { 0 } =$ $( 1 7 6 4 0 L R H / ( \sigma _ { 0 } ( K + 1 ) ) ) ^ { 2 } = O ( [ \ln ( e K ) ] ^ { 2 } / K ^ { 2 } )$ , meaning that $\Lambda _ { K } ^ { \mathrm { S } } = 8 0 0 K$ for large K and both H and $\mathcal { L } _ { K }$ are of the order ln(eK); therefore, a factor of the order $\sqrt { \ln ( e K ) }$ remains between the upper and the lower bound, and nothing is claimed for the multiplicative term or for the joint dependence on K and a vanishing $\beta .$ The construction is a minimax one: the hard instance, its curvature $\mu ,$ and its noise law depend on $K$ and $s ;$ each noise law is bounded with variance $\begin{array} { r } { \mathrm { V a r } _ { P _ { v } } ( Z + v \Delta ) = V - \Delta ^ { 2 } \in [ \frac { 6 3 } { 1 0 2 4 } , \frac { 1 } { 8 } ] \cdot \bar { \sigma _ { 0 } ^ { 2 } } [ \ln ( e K ) ] ^ { 1 - 2 / s } } \end{array}$ (this follows from $a ^ { 2 } T ^ { 1 - 2 / s } \leq V \leq 2 a ^ { 2 } T ^ { 1 - 2 / s }$ and $\Delta ^ { 2 } \le V / ( 6 4 K ) )$ ), which is not bounded uniformly in K for $s > 2$ That is, no hardness is claimed for one fixed infinite-variance distribution, for a class with a prescribed strong convexity constant, or for centrally symmetric noise (the centered tilted laws are asymmetric).

## I.9 Removing the Horizon Logarithm

The factors $\begin{array} { r } { \ell = \ln \frac { 4 ( K + 1 ) } { \beta } } \end{array}$ and $\begin{array} { r } { H = \ln { \frac { 4 K } { \beta } } } \end{array}$ appear since the proofs of Proposition F.6 and Theorem I.1 distribute the failure probability over the $K + 1 \ ( \mathrm { r e s p } . \ K )$ steps of the induction. This is an artifact of the union bound, not of the heavy tails; under the moment assumption, its removal is studied by Nguyen et al. (2023). Let

$$
h : = \ln { \frac { 4 } { \beta } } > 1 .\tag{90}
$$

Theorem I.18 (Horizon-free confidence). 1. Proposition F.6 remains valid when ℓ is replaced by h in (40), under either noise assumption and for all $\alpha \in ( 1 , 2 ]$ . Consequently, Theorem F.4 and Corollary F.5 hold with ℓ replaced by h throughout.

2. Theorem F.1 and Corollary F.2 hold with ℓ replaced by h throughout (in $\gamma _ { L } , \gamma _ { 0 } , \gamma _ { 1 } , \gamma _ { 2 } , \lambda .$ $\Delta _ { L } , \Delta _ { 0 } , \Delta _ { 1 } )$ , under Assumption 3.4 with $\alpha \in ( 1 , 2 ]$ and under Assumption 3.5 with $\alpha < 2$ if c is computed with $v = v _ { \alpha } ( \infty )$

3. Theorem I.1 and Corollary I.2 hold with H replaced by h throughout (in $\lambda _ { k } , \ a _ { * } ( K , \beta )$ and (63)), under Assumption 3.4 with $\alpha \in ( 1 , 2 ]$ , and under Assumption 3.5 with $\alpha < 2$ if $q _ { \alpha }$ is computed with $v = v _ { \alpha } ( \infty )$

In the cases of items $2$ and 3, the corresponding bound below holds for each method separately with probability at least $1 - \beta$

$$
f ( \bar { x } ^ { K } ) - f ^ { * } \leq C _ { \alpha } \left[ \frac { L R ^ { 2 } h } { K + 1 } + \sigma _ { 0 } R \Big ( \frac { h } { K + 1 } \Big ) ^ { r } + \sigma _ { 1 } ^ { 2 } L R ^ { 2 } \Big ( \frac { h } { K + 1 } \Big ) ^ { 2 r } \right] ,
$$

$$
f ( y ^ { K } ) - f ^ { * } \le C _ { \alpha } \left[ \frac { L R ^ { 2 } h ^ { 2 } } { K ^ { 2 } } + \sigma _ { 0 } R \Big ( \frac h K \Big ) ^ { r } + \sigma _ { 1 } ^ { 2 } L R ^ { 2 } \Big ( \frac h K \Big ) ^ { 2 r } \right] ,\tag{91}
$$

with $\textstyle r = { \frac { \alpha - 1 } { \alpha } }$ and constants $C _ { \alpha }$ depending only on $\alpha ;$ under Assumption 3.4, $C _ { \alpha }$ is bounded uniformly in $\alpha \in [ 3 / 2 , 2 ]$

Proof. Item 1. We modify the proof of Proposition F.6, with h in place of ℓ in (40) and in all quantities derived from it $( c _ { 1 } = R ^ { 2 } / ( 5 h ) , G = R ^ { 4 } / ( 1 5 0 h ) )$ . The consequences (41) use only $h > 1$ The events $E _ { k }$ from (42) are non-increasing in $k , \ E _ { k } \ \in \ { \mathcal G } _ { k }$ , and all deterministic bounds from the proof (localization, (43), (45), (46), (47), (48), (49), the bounds on $\textcircled{2} , \textcircled{4} , \textcircled{5}$ , and the variance bounds (51) and its analogue for ③) hold on $E _ { T - 1 }$ for every $T \leq K + 1$ exactly as before, since they use ℓ only through (40). Instead of the induction, we define

$$
\tau : = \operatorname* { m i n } \{ T \in \{ 1 , \dots , K + 1 \} : \ E _ { T } { \mathrm { ~ f a i l s } } \} \qquad ( \tau : = K + 2 { \mathrm { ~ i f ~ } } E _ { K + 1 } { \mathrm { ~ h o l d s } } ) ,
$$

meaning that $\{ t < \tau \} = E _ { t } \in \mathcal G _ { t }$ for $t \leq K$ , and consider the stopped increments $X _ { t } \mathbf { 1 } _ { E _ { t } }$ and $Y _ { t } \mathbf { 1 } _ { E t } , t = 0 , \ldots , K$ , where $X _ { t } , Y _ { t }$ are the summands of $\textcircled{1} , \textcircled{3}$ . Since $\mathbf { 1 } _ { E _ { t } }$ is G -measurable, these are martingale diference sequences w.r.t. $\left( \mathcal { G } _ { t + 1 } \right)$ , bounded by $c _ { 1 }$ , with conditional variances $\sigma _ { t } ^ { 2 } \mathbf { 1 } _ { E _ { t } }$ and $\widetilde { \sigma } _ { t } ^ { 2 } \mathbf { 1 } _ { E _ { t } }$ . Their total conditional variances are at most G almost surely: on $\{ \tau = T \}$ with $T \leq K + 1$ the event $E _ { T - 1 }$ holds and the sums reduce to $\textstyle \sum _ { t < T } \sigma _ { t } ^ { 2 }$ and $\textstyle \sum _ { t < T } { \widetilde { \sigma } } _ { t } ^ { 2 }$ , which are at most G by (51) and its analogue; on $\{ \tau = K + 2 \}$ , the same holds with $T = K + 1$ . Therefore, the maximal form of Lemma E.3 with $b _ { 1 } = R ^ { 2 } / 5 , c _ { 1 }$ , and $G$ gives, for the two channels,

$$
\mathbb { P } \Big \{ \exists T \le K + 1 : \ \Big | \sum _ { t < T } X _ { t } \mathbf { 1 } _ { E _ { t } } \Big | > \frac { R ^ { 2 } } { 5 } \Big \} \le 2 e ^ { - h } = \frac \beta 2 , \qquad \mathbb { P } \Big \{ \exists T \le K + 1 : \ \Big | \sum _ { t < T } Y _ { t } \mathbf { 1 } _ { E _ { t } } \Big | > \frac { R ^ { 2 } } { 5 } \Big \} \le \frac \beta 2 .
$$

On the complement $E ^ { \mathrm { m a x } }$ of the union of these two events, $\mathbb { P } ( E ^ { \mathrm { m a x } } ) \geq 1 - \beta$ , we assume that $\tau = T \leq K + 1$ . Then $E _ { T - 1 }$ holds, implying $\mathbf { 1 } _ { E _ { t } } = 1$ for $t \leq T - 1$ and the stopped sums up to $T - 1$ coincide with ① and ③ at time $T ;$ hence, $| \mathbb { 1 } | , | \mathfrak { 3 } | \le R ^ { 2 } / 5$ , and $| \textcircled { 2 } | , \textcircled { 4 } , \textcircled { 5 } \leq R ^ { 2 } / 5$ by the deterministic bounds on $E _ { T - 1 }$ . As in the paragraph “Completing the induction”, this gives the condition from (42) at $t \ : = \ : T$ and $R _ { T } \ \leq \ \sqrt { 2 } R$ , meaning that $E _ { T }$ holds, which contradicts $\tau = T$ . Thus, $E _ { K + 1 } \supseteq E ^ { \mathrm { m a x } }$ , and the conclusion of Proposition F.6 follows as before. The proof of Theorem F.4 applies Proposition F.6 with $v = v _ { \alpha } ( \Lambda _ { \lambda } )$ , where $\Lambda _ { \lambda }$ from (29) does not depend on $\ell ,$ and the proof of Corollary F.5 is algebraic in $\ell ;$ both of them remain valid with h.

Item 2. The proof of Theorem F.1 verifies (40) from $\gamma \leq \operatorname* { m i n } \{ \gamma _ { L } , \gamma _ { 0 } , \gamma _ { 1 } , \gamma _ { 2 } \}$ via algebraic manipulations in which ℓ is a parameter (Step 1 in Appendix F); only the fixed-point check from Step 2, which bounds $\Lambda _ { \gamma _ { * } , \lambda _ { * } }$ by $\Lambda _ { K }$ under Assumption 3.5, uses $\ell \geq \ln ( 4 ( K + 1 ) )$ . Under Assumption 3.4, the constant is $c = c _ { \alpha } ^ { \mathrm { m o m } }$ and no fixed point is needed; under Assumption 3.5 with $\alpha < 2$ and $v = v _ { \alpha } ( \infty )$ , (47) holds with this v directly, since $v _ { \alpha } ( \cdot ) \leq v _ { \alpha } ( \infty )$ , and, again, no fixed point is needed. Then Item 1 gives the claim, and Corollary F.2 follows from algebraic manipulations.

Item 3. In the proof of Theorem I.1 (see Appendix I.5), H enters through the parametrization $\lambda _ { t } = R / ( 3 0 \omega _ { t + 1 } H ) , a \geq 1 7 6 4 0 0 H ^ { 2 }$ , the Freedman parameters $c _ { 1 } = R ^ { 2 } / ( 5 H ) , G = R ^ { 4 } / ( 1 5 0 H )$ the inequalities $H > 1$ in Steps 1, 8, and 10, and the fixed-point check from Step $3 ^ { \prime }$ , which is needed only under Assumption 3.5 with $v = v _ { \alpha } ( \Lambda _ { K } ^ { \mathrm { S } } )$ ; in the two cases from the statement, $\phi _ { \lambda _ { t } } ( D _ { t } ) \leq v D _ { t }$ holds directly with the horizon-free v (Lemma D.2, resp. $v _ { \alpha } ( \cdot ) \ \leq \ v _ { \alpha } ( \infty ) )$ , and Step 6 reads $\begin{array} { r } { S _ { T } \le v 3 0 ^ { \alpha - 2 } \frac { R ^ { 2 } } { H } \Psi _ { T } \le \frac { R ^ { 2 } } { 2 7 0 0 H } } \end{array}$ by (79) and the definition of $q _ { \alpha }$ . With h in place of $H$ , all deterministic steps hold on $E _ { T - 1 }$ for every $T \leq K$ as before. Let τ be the first $T \leq K$ at which $E _ { T }$ from (74) fails $( \tau : = K + 1 \mathrm { ~ i f ~ } E _ { K }$ holds); $\{ t < \tau \} = E _ { t } \in \mathcal G _ { t }$ , and the stopped increments $X _ { t } \mathbf { 1 } _ { E _ { t } }$ $Y _ { t } \mathbf { 1 } _ { E _ { t } }$ of ①, ③ are martingale diferences bounded by $c _ { 1 } = R ^ { 2 } / ( 5 h )$ with total conditional variances at most $G = R ^ { 4 } / ( 1 5 0 h )$ almost surely (Steps 6 and 8 on $E _ { \tau - 1 } )$ . The maximal form of Lemma E.3 bounds the probability that some partial sum of either channel exceeds $R ^ { 2 } / 5$ by $\beta / 2$ for each of them. On the complement, if $\tau = T \leq K$ , then $E _ { T - 1 }$ holds, the stopped sums coincide with ①, ③ at time $T _ { \ast }$ Steps 5–9 give $B _ { T } < R ^ { 2 }$ , and Step 1 and Lemma I.4 give $\| z ^ { T } - x ^ { \ast } \| < 2 R$ , meaning that $E _ { T }$ holds, which is a contradiction. Thus, $E _ { K }$ holds with probability at least $1 - \beta$ , and the final bound follows as in Step $1 0 ;$ Corollary I.2 follows from algebraic manipulations. The rates (91) are Corollary F.2 and (63) with $h ;$ under Assumption 3.4, $b _ { \alpha } ^ { \mathrm { m o m } } \in [ 1 / 2 , 1 ]$ and $v _ { \alpha } ^ { \mathrm { m o m } } \in [ 1 , 3 / 2 ]$ meaning that $c _ { \alpha } ^ { \mathrm { m o m } } , \ : q _ { \alpha } , \ : c _ { 0 , \alpha } , \ : c _ { 1 , \alpha }$ , and, as a consequence, $C _ { \alpha }$ are bounded uniformly on $[ 3 / 2 , 2 ]$ ， which concludes the proof. □

Note that Theorem I.18 controls the prefixes of one run tuned for the horizon $K ;$ it is not an anytime statement for all horizons of one infinite trajectory. For the original tuning with $v _ { \alpha } ( \Lambda _ { K } )$ and $v _ { \alpha } ( \Lambda _ { K } ^ { \mathrm { S } } )$ , the fixed-point estimates in Steps 2 and $3 ^ { \prime }$ must be checked again after replacing ℓ or H by h. At $\alpha = 2$ and fixed confidence, the logarithmic terms in these checks cannot in general be absorbed into h for large $K$ . This is not a general obstruction for $\alpha < 2$ , where $v _ { \alpha }$ is bounded. Theorem I.18 uses $v _ { \alpha } ( \infty )$ in that case, and the following corollary treats the endpoint through the moment embedding.

Corollary I.19 (The Lorentz endpoint without the union-bound loss). Let Assumptions 3.1–3.3 hold for $Q = B _ { 3 R } ( x ^ { * } )$ , where $R \geq \| x ^ { 0 } - x ^ { * } \| > 0$ , let Assumption I.1 hold with $s \in [ 2 , \infty ]$ , let $K \ge 1 , \beta \in ( 0 , 1 )$ , and $h = \ln ( 4 / \beta )$ . Let $p _ { K } : = 2 -$ min $\{ { \textstyle \frac { 1 } { 2 } } , { \textstyle \frac { 1 } { \ln ( e K ) } } \}$ and $C _ { p , s }$ be defined as in Remark I.15. Then Algorithm $2$ with the parameters from Theorem I.1 under Assumption 3.4, computed with the exponent $p _ { K }$ , the parameters $\left( C _ { p _ { K } , s } \sigma _ { 0 } , C _ { p _ { K } , s } \sigma _ { 1 } \right)$ , and h in place of $H _ { ; }$ satisfies, with probability at least $1 - \beta$

$$
f ( y ^ { K } ) - f ^ { * } \le C \left[ \frac { L R ^ { 2 } h ^ { 2 } } { K ^ { 2 } } + \sigma _ { 0 } R \sqrt { \frac { h } { K } } \left[ \ln ( e K ) \right] ^ { \frac { 1 } { 2 } - \frac { 1 } { s } } + \sigma _ { 1 } ^ { 2 } L R ^ { 2 } \frac { h } { K } \left[ \ln ( e K ) \right] ^ { 1 - \frac { 2 } { s } } \right]\tag{92}
$$

with an absolute constant $C .$ . Under the same assumptions for $Q = B _ { 2 R } ( x ^ { * } )$ , Algorithm 1 with the parameters from Theorem F.1 (exponent $p _ { K }$ , the same noise parameters, h in place of $\ell , K + 1$ steps) satisfies the same bound for $f ( \bar { x } ^ { K } ) - f ^ { * }$ with the deterministic term $L R ^ { 2 } h / ( K + 1 )$

Proof. We use the prescribed exponent $p _ { K }$ for every $s \in [ 2 , \infty ]$ . In particular, $C _ { p , 2 } = 1$ , so the moment embedding also covers $s = 2$ . By (88) and the subadditivity of $t \mapsto t ^ { p / 2 }$ , Assumption I.1 implies, for $1 < p < 2$ and $x \in Q , \mathbb { E } \Vert g ( x , \xi ) - \nabla f ( x ) \Vert ^ { p } \leq C _ { p , s } ^ { p } ( \sigma _ { 0 } ^ { 2 } + \sigma _ { 1 } ^ { 2 } \Vert \nabla f ( x ) \Vert ^ { 2 } ) ^ { p / 2 } \leq C _ { p , s } ^ { p } ( \sigma _ { 0 } ^ { p } +$ $\sigma _ { 1 } ^ { p } \lVert \nabla f ( x ) \rVert ^ { p } )$ , i.e., Assumption 3.4 with the exponent $p$ and the parameters $C _ { p , s } ( \sigma _ { 0 } , \sigma _ { 1 } )$ . With $p = p _ { K } \in [ \frac { 3 } { 2 } , 2 )$ , we have $C _ { p _ { K } , s } \leq 4 [ \ln ( e K ) ] ^ { \frac { 1 } { 2 } - \frac { 1 } { s } }$ (see Remark I.15), $K ^ { - ( p _ { K } - 1 ) / p _ { K } } \leq e ^ { 1 / ( 2 p _ { K } ) } K ^ { - 1 / 2 }$ (since $\begin{array} { r } { { \frac { 1 } { 2 } } - { \frac { { \bar { p } } - 1 } { p } } = { \frac { 2 - p } { 2 p } } } \end{array}$ and $( 2 - p _ { K } ) \ln K \leq 1 )$ , and $h ^ { ( p _ { K } - 1 ) / p _ { K } } \leq h ^ { 1 / 2 }$ (since $h > 1 )$ . Substituting these inequalities into (91) with the exponent $p _ { K }$ and using the uniform boundedness of $C _ { p }$ on $[ \frac { 3 } { 2 } , 2 ]$ , we get (92); for Clip-SGD, the same substitution in the first bound of (91) applies, which concludes the proof. □

For fixed L, $R , \sigma _ { 0 } > 0 , \sigma _ { 1 } = 0 , s \in [ 2 , \infty ]$ , and $\beta < 1 / 3$ , the stochastic term in (92) matches the lower bound (89) as $K  \infty$ . Thus, $\sigma _ { 0 } \dot { R } K ^ { - 1 / 2 } [ \ln ( e K ) ] ^ { 1 / 2 - 1 / s }$ is the asymptotic minimax order over the additive $L ^ { 2 , s }$ class at fixed confidence, and both Clip-SGD and Clip-SSTM attain this order. The asymptotic qualification accounts for the cap $L R ^ { 2 }$ in the lower bound and the deterministic terms in the upper bounds. That is, not all logarithmic factors from the earlier bounds are statistically necessary: the logarithmic dependence on the horizon in ℓ and H comes from the union bound, while the Lorentz factor has a lower bound of its own. For $1 < \alpha < 2$ , the dependence on $\beta$ in (91) is optimal as well (see Proposition H.3).

## J Additional Experiments

In this section, we present additional experiments complementing Section 6: an experiment with symmetric noise on a quadratic problem, a ridge regression problem on real data with the asymmetric noise from Section 6, and the experiments with the full theoretical parameter prescriptions from Theorems F.1 and I.1 on a smooth convex benchmark with vanishing curvature. We start with the description of the full experimental protocol.

Protocol. We use $\alpha = 3 / 2$ and 1024 independent runs per configuration. The reported horizons reach $1 0 ^ { 6 }$ oracle calls. For ${ \mathsf { C l i p } } { \mathsf { - S G D } }$ , we report the mean of $f ( \bar { x } ^ { n - 1 } ) - f ^ { * }$ , where $\begin{array} { r } { { \bar { x } } ^ { n - 1 } = n ^ { - 1 } \sum _ { k = 0 } ^ { n - 1 } x ^ { k } } \end{array}$ is the averaged iterate after n oracle calls (one call per iteration); for ${ \mathsf { C l i p } } { \mathsf { P } } { \mathsf { - S S T M } }$ , we report the mean of $f ( y ^ { n } ) - f ^ { * }$ , using the output prescribed by Algorithm 2, without additional averaging. The bands are pointwise approximate 95% confidence intervals for the mean. Horizon-dependent configurations restart from the same initial point for each terminal horizon; the theoretical failure parameter is $\beta = 0 . 0 5$

Let P satisfy P $\{ P > t \} = t ^ { - \alpha }$ for $t \geq 1$ ; then $\mathbb { E } P = 3$ and $\mathbb { E } P ^ { \alpha } = \infty$ . We use either $Z = \epsilon P$ , with an independent Rademacher sign ϵ, or the centered asymmetric $Z = ( P { - } 3 ) / 2$ . Both of them satisfy $\mathbb { E } Z = 0$ and $\mathrm { s u p } _ { u > 0 } u \mathbb { P } \{ | Z | > u \} ^ { 1 / \alpha } \leq 1$ . In the scalar experiments, $g ( x ) = x + S ( x ) Z , \ S ( x ) =$ $\left( \sigma _ { 0 } ^ { \alpha } + \sigma _ { 1 } ^ { \alpha } | x | ^ { \alpha } \right) ^ { 1 / \alpha }$ . The vector oracles are specified in each experiment. All of them satisfy the afine weak-tail inequality with the stated $( \sigma _ { 0 } , \sigma _ { 1 } )$ ; whenever the noise scale is non-zero, its α-th moment is infinite.

Small additive noise under symmetric noise. We minimize $f ( x ) = x ^ { 2 } / 2$ on R $( L = 1$ $x ^ { * } = 0 )$ starting from $x ^ { 0 } = 1$ with the symmetric noise, $\sigma _ { 1 } = 0 . 0 3$ , the constant stepsize $\gamma = 0 . 0 5$ and the constant clipping level $\lambda = 1$ , and vary $\sigma _ { 0 } \in \{ 0 , 0 . 0 1 , 0 . 1 , 0 . 5 , 1 \}$ (see Figure 2).

For $\sigma _ { 0 } = 0$ , the noise scale vanishes at $x ^ { * }$ , and the observed error closely tracks the noiseless reference $( \gamma n ) ^ { - 2 } / 2$ over the plotted range. For $\sigma _ { 0 } > 0$ , the curves leave this reference earlier as $\sigma _ { 0 }$ increases and then decay approximately as $1 / n$ , at levels increasing with $\sigma _ { 0 }$ . The exact noiseless averaged error, when clipping is inactive, is $( 1 - ( 1 - \gamma ) ^ { n } ) ^ { 2 } / ( 2 \gamma ^ { 2 } n ^ { 2 } )$ ; the plotted reference is its large-n limit.

Two features of this example go beyond the tail assumption. For symmetric Z and even $S ,$ the mean clipped field $m _ { \lambda } ( x ) : = \mathbb { E } { \mathsf { c l i p } } ( x + S ( x ) Z , \lambda )$ is odd, and, therefore, $m _ { \lambda } ( 0 ) = 0 ;$ ; its zero is unique here, although, in general, $m _ { \lambda } ( x ) \neq \nabla f ( x )$ for $x \neq 0 { : }$ since $\sigma _ { 0 } \leq 1$ and $\sigma _ { 1 } = 0 . 0 3$ , we have $S ( x ) \leq \sigma _ { 0 } + \sigma _ { 1 } | x | < 1 + | x |$ for $x \neq 0 ;$ for $x > 0$ and $a \geq 0$ , the sum ${ \mathsf { c l i p } } ( x + a , 1 ) + { \mathsf { c l i p } } ( x - a , 1 )$ is non-negative and positive when $a < x + 1$ , which is provided by the event $\{ S ( x ) P < x + 1 \}$ of positive probability (the case $S ( x ) = 0$ is immediate); hence, $m _ { 1 } ( x ) > 0$ for $x > 0$ and, by oddness, $m _ { 1 } ( x ) < 0$ for $x < 0$ . Thus, clipping does not displace the stationary point in this experiment, and the stepsize is constant rather than tied to the horizon as in Corollary F.5. That is, the experiment illustrates the dependence on $\sigma _ { 0 } .$ , through which the additive noise enters both the neighborhood $\mathcal { N } _ { \lambda }$ and the vanishing term $\mathcal { V } _ { \lambda , K }$ from Corollary F.5, and not the convergence to $x ^ { * }$ with a fixed stepsize for arbitrary weak-L<sup>α</sup> noise, which fails in general (see Theorem H.1).

![](images/7428e221db715d2eaa4cc70b47e46220c30ca02732fb21edda71b701db4b420c.jpg)  
Figure 2: Mean error of the averaged iterate on $f ( x ) = x ^ { 2 } / 2$ under symmetric Pareto noise

Details of the clipping-level experiment. In the experiment from Figure 1 (see Section 6), $\gamma _ { * } ( n ) = \operatorname* { m i n } \{ \gamma _ { L } , \gamma _ { 0 } , \gamma _ { 1 } , \gamma _ { 2 } \}$ and $\ell _ { n } = \ln ( 4 n / \beta )$ are computed for the horizon n with $\beta = 0 . 0 5$ , which gives $\lambda ( 1 0 ^ { 2 } ) = 2 . 8 , \lambda ( 1 0 ^ { 4 } ) = 3 2 .$ , and $\lambda ( 1 0 ^ { 6 } ) \approx 5 7 0 ;$ each horizon n is a separate run with $\lambda ( n )$ being constant within the run. The dotted lines are the values $f ( x _ { \lambda } )$ at the zero $x _ { \lambda }$ of the mean clipped field x 7→ Ecl $\mathrm { i p } ( g ( x ) , \lambda )$ , computed numerically; $x _ { \lambda } > 0 .$ , since clipping cuts the heavy right tail of $Z .$ They decrease with λ roughly as $\lambda ^ { - 2 ( \alpha - 1 ) }$ , since, on a quadratic, the error at the biased fixed point is the squared bias divided by $2 L ,$ , and the bias scales as $\sigma _ { 0 } ^ { \alpha } / \lambda ^ { \alpha - 1 }$ (see Lemma 4.1). The zero $x _ { \lambda }$ of the mean clipped field is a reference for the observed plateau: with a non-zero constant stepsize, the stochastic iterates may fluctuate around this point, and the limiting error of their average is not necessarily equal to $f ( x _ { \lambda } ) - f ^ { * }$ exactly.

Regression on real data. We minimize $\begin{array} { r } { f ( w ) = \frac { 1 } { 2 m } \| A w - y \| ^ { 2 } + \frac { 0 . 0 1 } { 2 } \| w \| ^ { 2 } } \end{array}$ on the Diabetes dataset (Efron et al., 2004), with standardized features and targets $( m = 4 4 2 , d = 1 0$ ; loaded with scikit-learn, Pedregosa et al., 2011), starting from $w ^ { 0 } = 0$ . Here $L = \lambda _ { \operatorname* { m a x } } ( \nabla ^ { 2 } f ) \approx 4 . 0$ and $\lambda _ { \operatorname* { m i n } } ( \nabla ^ { 2 } f ) \approx 0 . 0 1 9$ . We use the same clipping levels as above, with $R = \| w ^ { 0 } - w ^ { * } \|$ and this L in $\lambda ( n )$ , and the common stepsize $\gamma = 0 . 0 5 / L$ (Figure 3).

The oracle is $g ( w , Z ) = \nabla f ( w ) + S ( w ) Z e$ , where Z is the centered asymmetric Pareto variable, $( \sigma _ { 0 } , \sigma _ { 1 } ) \ = \ ( 0 . 0 5 , 0 . 0 0 1 )$ , and e is a fixed unit eigenvector of the smallest Hessian eigenvalue. The projected scale $S ( w ) : = ( \sigma _ { 0 } ^ { \alpha } + \sigma _ { 1 } ^ { \alpha } | \langle \nabla f ( w ) , e \rangle | ^ { \alpha } ) ^ { 1 / \alpha }$ satisfies Assumption 3.5, since $| \left. \nabla f ( w ) , e \right. | \leq \| \nabla f ( w ) \|$ . This oracle is also the gradient of $f ( w ) + Z \Psi ( \langle w - w ^ { * } , e \rangle )$ , with $\Psi ^ { \prime } ( u ) =$ $( \sigma _ { 0 } ^ { \alpha } + \sigma _ { 1 } ^ { \alpha } \lambda _ { \operatorname* { m i n } } ^ { \alpha } | u | ^ { \alpha } ) ^ { 1 / \alpha }$ , although our model does not require a sample-objective representation.

For small clipping levels, the error first decreases and then rises toward a plateau. This behavior is consistent with a displacement of the equilibrium of the mean clipped field. If $\nabla f ( w _ { \lambda } ) = b e$ , the quadratic structure gives $w _ { \lambda } - w ^ { * } = ( b / \lambda _ { \operatorname* { m i n } } ) e$ and $f ( w _ { \lambda } ) - f ^ { * } = b ^ { 2 } / ( 2 \lambda _ { \operatorname* { m i n } } )$ , illustrating how low curvature can amplify clipping bias. Since the experiment uses a positive constant stepsize, this equilibrium value is a reference for the observed plateau, not an exact formula for the limiting error of the averaged iterates. Larger levels lower the observed plateau, and $\lambda ( n )$ yields decreasing error over the tested horizons.

![](images/59a3d06252d12a03e5e2f4438e788bc0ffb2e67a9db2f165b9d9ff5cf3b928b1.jpg)  
Figure 3: Mean objective gap of the averaged iterate for ridge regression on the Diabetes dataset with asymmetric Pareto noise along the eigenvector of the smallest Hessian eigenvalue

Relation to the theory. The clipping level $\lambda ( n )$ is the one prescribed by Theorem 5.1 for the horizon n, but the stepsize used is the common empirical value rather than the much smaller $\gamma _ { * } ( n )$ , and the constant-level runs use a fixed stepsize instead of the horizon-dependent one from Theorem 5.3; therefore, the comparisons isolate the efect of the clipping level and do not test the high-probability bounds themselves.

For the asymmetric-noise examples, larger clipping levels lower the observed plateaus, while the horizon-dependent level yields decreasing error over the tested horizons. The symmetric-noise example illustrates a diferent case: clipping preserves the zero of the mean field, and the averaged error continues to decrease with a constant level. These observations are consistent with the mechanisms discussed in Sections 5 and 5.2.

The pointwise 95% intervals quantify uncertainty in the estimated mean across runs; they are not 95% guarantees for the error of an individual run. All experiments use $\alpha = 3 / 2$ , so they do not directly test the Lorentz endpoint or compare its two theoretical tunings.

A convex benchmark with vanishing curvature. In the remaining experiments, we use a chain-inspired objective in $d = 1 6$ dimensions. Let $D w = ( w _ { 1 } - w _ { 2 } , \ldots , w _ { d - 1 } - w _ { d } )$ and $M =$ $\lambda _ { \operatorname* { m a x } } ( D ^ { \top } D )$ . We choose a zero-mean linear profile $w _ { i } ^ { * } = c ( ( d + 1 ) / 2 - i ) , 1 \leq i \leq d .$ , with $\| w ^ { * } \| = R$ and let $\tau = \lVert D w ^ { * } \rVert / \sqrt { d - 1 }$ . We set $L = R = 1 , w ^ { 0 } = 0 .$ , and

$$
\begin{array} { l } { { \displaystyle f ( w ) = \frac { 8 L \tau ^ { 2 } } { 9 M } \sum _ { i = 1 } ^ { d - 1 } \psi \biggl ( \frac { [ D ( w - w ^ { * } ) ] _ { i } } { \tau } \biggr ) , } } \\ { { \displaystyle \psi ( t ) = \frac { t ^ { 2 } - \ln ( 1 + t ^ { 2 } ) } { 2 } . } } \end{array}
$$

Since $0 \leq \psi ^ { \prime \prime } ( t ) \leq 9 / 8$ , this objective is globally convex and L-smooth. Moreover, $\nabla ^ { 2 } f ( w ^ { * } ) = 0$ and $\psi ( t ) = t ^ { 4 } / 4 + O ( t ^ { 6 } )$ near zero; therefore, it is not strongly convex even on the active zero-mean subspace. We use either additive noise $g ( w ) = \nabla f ( w ) + \sigma _ { 0 } Z v$ , where $v = \nabla f ( w ^ { 0 } ) / \| \nabla f ( w ^ { 0 } ) \|$ is fixed, or multiplicative noise $g ( w ) = ( 1 + \sigma _ { 1 } Z ) \nabla f ( w )$ , always with $Z = ( P - 3 ) / 2$ . These are the gradients of $f _ { Z } = f + \sigma _ { 0 } Z v ^ { \top }$ w and $f _ { Z } = ( 1 + \sigma _ { 1 } Z ) f$ , respectively, and they satisfy Assumption 3.5 with the stated coeficients.

Three noise regimes for Clip-SGD. We compare $( \sigma _ { 0 } , \sigma _ { 1 } ) = ( 0 , 0 ) , ( 0 , 0 . 0 0 4 ) , ( 0 . 0 0 2 , 0 )$ on this objective, using the full pair γ<sub>∗</sub>(n) and $\lambda ( n ) = R / ( 4 0 \gamma _ { * } ( n ) \ln ( 4 n / \beta ) )$ from Theorem F.1, recomputed for each configuration and horizon (see Figure 4). The mean averaged-iterate gaps are close at short horizons and separate as the horizon increases: the noiseless case decreases most rapidly, followed by the multiplicative and then the additive noise. Since the prescribed stepsizes depend on the noise coeficients, this separation reflects both the stochastic perturbations and the corresponding conservative tuning. The comparison illustrates the three noise regimes from Corollary F.2, whose upper bounds have the orders $n ^ { - 1 } , n ^ { - 2 / 3 }$ , and $n ^ { - 1 / 3 }$ up to logarithmic factors. These orders describe theoretical upper bounds and are not claimed to be fitted empirical slopes for this instance.

![](images/10f5942e024c86910bc317ab3a21b2d624bc157a88bf0caeea694c9823a68590.jpg)  
Figure 4: Mean averaged-iterate gap for Clip-SGD on the smooth convex chain with vanishing curvature, under the full horizon-dependent prescription from Theorem F.1

Sensitivity of Clip-SSTM to the noise. On the same objective, we use the full prescription from Theorem I.1: $a = a _ { * } ( n , \beta )$ and $\lambda _ { k } ~ = ~ R / ( 3 0 \omega _ { k + 1 } H _ { n } )$ where $\omega _ { k + 1 } = ( k + 2 ) / ( 2 a L )$ and $H _ { n } = \ln ( 4 n / \beta )$ . We vary $\sigma _ { 0 } \in \{ 0 , 1 0 ^ { - 5 } , 1 0 ^ { - 4 } , 1 0 ^ { - 3 } \}$ with $\sigma _ { 1 } = 0$ (see Figure 5), and, separately, $\sigma _ { 1 } \in \{ 0 , 1 0 ^ { - 4 } , 3 \cdot 1 0 ^ { - 4 } , 1 0 ^ { - 3 } \}$ with $\sigma _ { 0 } = 0$ (see Figure 6). The two plots share the same noiseless control and cover $1 0 ^ { 3 } \leq n \leq 1 0 ^ { 6 }$ on logarithmic axes. The parameter $a _ { * } ( n , \beta )$ and the associated clipping schedule are recomputed for each noise level and terminal horizon. At the smallest noise levels, the curves closely follow the noiseless control, while larger coeficients lead to markedly slower observed progress. This separation reflects both the stochastic perturbations and the more conservative theoretical tuning required by larger noise coeficients.

![](images/35eff11480b638c1413245973e2b44fddc8cf7bddcba93371cdf0c5789f5c5b6.jpg)

![](images/ee688da7f6f94f0b1c56e6b5834b3840c418e5b5fe7548259b590c114638b46e.jpg)  
Figure 5: Mean output gap $f ( y ^ { n } ) - f ^ { * }$ of Clip-SSTM on the smooth convex chain under the full prescription from Theorem I.1: additive noise $( \sigma _ { 1 } = 0 )$  
Figure 6: Mean output gap $f ( y ^ { n } ) - f ^ { * }$ of Clip-SSTM on the smooth convex chain under the full prescription from Theorem I.1: multiplicative noise $( \sigma _ { 0 } = 0 )$

Software and data. We use the diabetes dataset from Efron et al. (2004), loaded through scikitlearn (Pedregosa et al., 2011). The scikit-learn software is distributed under the BSD-3-Clause license. The synthetic problems are fully specified by the formulas above.