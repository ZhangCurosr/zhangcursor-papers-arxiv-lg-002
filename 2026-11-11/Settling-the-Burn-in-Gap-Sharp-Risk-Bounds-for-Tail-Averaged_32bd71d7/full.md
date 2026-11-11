# Settling the Burn-in Gap: Sharp Risk Bounds for Tail-Averaged SGD in Linear Regression

Qingyue Zhao<sup>∗†</sup> Xuheng Li<sup>∗‡</sup> Quanquan Gu<sup>§</sup>

## Abstract

We study the excess risk of constant-stepsize online stochastic gradient descent (SGD) with tail-averaging for high-dimensional linear regression. Building on the instance-dependent spectral framework of Zou et al. (2021), we establish matching initialization-explicit upper and lower bounds on the total excess risk of tail-averaged SGD in terms of the population covariance spectrum H and initialization $\mathbf { w } _ { 0 } - \mathbf { w } ^ { * }$ whenever a trajectory spectral feedback condition holds. Existing analyses exhibit a fundamental gap in the efective variance: the upper bound of Zou et al. (2021) retains an undamped contribution from the initial error in the head eigenspace I † , whereas their lower bound omits the variance induced by the head initialization altogether. We show that both the existing upper and lower bounds are loose: the single-hop fast-mode variance contribution from the head initialization is exponentially damped during the burn-in phase, whereas fourth-moment energy scattered from the head initialization into slow spectral modes during the early burn-in steps survives undamped into the averaging window. We prove that our spectral feedback condition is satisfied in two concrete constant-stepsize regimes: (i) for arbitrary covariance spectra H and any finite horizon, under an explicit horizon-dependent stepsize chosen prior to execution; and (ii) for polynomially decaying spectra $\lambda _ { i } \asymp i ^ { - a } \ ( a > 1 )$ , under a single horizon-independent positive stepsize. Finally, we complement these positive results with explicit one- and two-dimensional separation constructions proving an exponential-versus-polynomial gap in the lower bound of Zou et al. (2021) and showing why fourth-moment assumptions alone cannot determine the asymptotic exponential decay rate without additional structural conditions. Proof ideas emerged from interactive brainstorming between the authors and the AI models GPT-6 Astra and Gemini 4 Argon; the authors simplified, reorganized, and verified all proofs and take full responsibility for the paper.

## 1 Introduction

Understanding how first-order stochastic optimization generalizes in high- and infinite-dimensional feature spaces is a central theme in modern statistical learning theory. For ofline empirical risk minimization in linear regression, Bartlett et al. (2020) and Tsigler and Bartlett (2023) characterized benign overfitting—when minimum-ℓ<sub>2</sub>-norm interpolators and ridge regression generalize well despite fitting noisy training data—in terms of the efective ranks of the population data covariance operator H (Belkin et al., 2020; Hastie et al., 2022). In the complementary setting of fresh-sample online stochastic approximation, Zou et al. (2021) established an instance-dependent excess risk analysis for constant-stepsize stochastic gradient descent (SGD) with full iterate averaging, demonstrating that online SGD enjoys an implicit spectral regularization efect governed by a closely related spectral partition of H (while noting that for polynomially decaying spectra such as $\lambda _ { i } = i ^ { - 2 }$ , online SGD achieves vanishing excess risk by forgetting noisy labels rather than interpolating training data; see also Wu et al. (2025)).

In large-scale stochastic optimization, however, averaging all iterates from the initial point $\mathbf { w } _ { 0 }$ sufers from slow polynomial decay of its deterministic drift bias $\begin{array} { r } { \frac { 1 } { 2 } \| \mathbb { E } [ \overline { { \mathbf { w } } } _ { 0 : N } ] - \mathbf { w } ^ { * } \| _ { \mathbf { H } } ^ { 2 } } \end{array}$ —decaying at best as $\mathcal { O } ( N ^ { - 2 } )$ when $\lVert \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } \rVert _ { \mathbf { H } ^ { - 1 } } < \infty$ (or in finite dimensions with $\lambda _ { d } > 0 )$ , and even more slowly $( \mathrm { e . g . } , \Theta ( N ^ { - 3 / 2 } )$ when $\lambda _ { i } = i ^ { - 2 }$ and $( \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } ) _ { i } ^ { 2 } = i ^ { - 2 } )$ when slow spectral modes remain uncontracted. To accelerate the decay of initial error while preserving statistical eficiency, Rakhlin et al. (2012) introduced sufix (tail) averaging for stochastic gradient descent, and Jain et al. (2017b, 2018) developed its operator-theoretic analysis for least-squares linear regression; in modern deep learning and language model pretraining, its checkpoint-based counterpart Latest Weight Averaging (LAWA) (Kaddour, 2022; Sanyal et al., 2023) averages recent checkpoints along the trajectory to achieve rapid training speedups. Tail-averaging discards the first s iterates as a burn-in phase and averages the subsequent N iterates:

$$
\overline { { \mathbf { w } } } _ { s : s + N } = \frac { 1 } { N } \sum _ { t = s } ^ { s + N - 1 } \mathbf { w } _ { t } , \qquad \mathrm { w h e r e } \qquad \mathbf { w } _ { t } = \mathbf { w } _ { t - 1 } + \gamma \bigl ( y _ { t } - \bigl < \mathbf { w } _ { t - 1 } , \mathbf { x } _ { t } \bigr > \bigr ) \mathbf { x } _ { t } .
$$

Zou et al. (2021, Theorems 5.1 and 5.2) extended their spectral excess risk analysis to tail-averaged SGD. Let $\{ \lambda _ { i } \} _ { i \geq 1 }$ denote the eigenvalues of H in non-increasing order, and define the efective dimension cutofs $k ^ { * } = \operatorname* { m a x } \{ k : \lambda _ { k } \geq ( \gamma N ) ^ { - 1 } \}$ and $k ^ { \dagger } = \operatorname* { m a x } \{ k : \lambda _ { k } \geq ( \gamma ( s + N ) ) ^ { - 1 } \}$ . Up to universal multiplicative constants, when the burn-in length satisfies $s = \Theta ( N )$ , Zou et al. (2021, Theorem 5.1) upper-bounds the bias-induced variance contribution by

$$
\frac { \alpha \big ( \| \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } \| _ { \mathbf { I } _ { 0 : k ^ { \dagger } } } ^ { 2 } + ( s + N ) \gamma \| \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } \| _ { \mathbf { H } _ { k ^ { \dagger } ; \infty } } ^ { 2 } \big ) } { N \gamma ( 1 - \gamma \alpha \operatorname { t r } ( \mathbf { H } ) ) } \cdot \left( \frac { k ^ { * } } { N } + N \gamma ^ { 2 } \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } \right) ,\tag{1.1}
$$

whereas their lower bound (Zou et al., 2021, Theorem 5.2) yields only

$$
\beta \| \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } \| _ { \mathbf { H } _ { k } \uparrow _ { : \infty } } ^ { 2 } \cdot \left( \frac { k ^ { * } } { N } + N \gamma ^ { 2 } \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } \right) .\tag{1.2}
$$

Comparing (1.1) and (1.2), Zou et al. (2021) observed that the two bounds fail to match whenever the initial error $\mathbf { w } _ { 0 } - \mathbf { w } ^ { * }$ has significant energy in the head subspace $\mathbf { I } _ { 0 : k ^ { \dagger } }$ , and conjectured that the upper bound (1.1) can be improved. The same discrepancy on the head initialization subspace persists in the exponential moving average (EMA) analysis of Li and Gu (2025, Theorems 4.1 and 4.2) and in the simplified diagonal covariance analysis of Meterez et al. (2025).

Why Both Existing Upper and Lower Bounds Are Loose. In this paper, we show that closing the gap between (1.1) and (1.2) requires sharpening both the upper bound in Zou et al. (2021, Theorem 5.1) and the lower bound in Zou et al. (2021, Theorem 5.2). Specifically, expanding the outer product in (1.1) between the initialization subspaces $\{ \mathbf { I } _ { 0 : k ^ { \dagger } } , \mathbf { H } _ { k ^ { \dagger } : \infty } \}$ and the fast and slow variance components $\begin{array} { r } { \{ k ^ { * } / N , N \gamma ^ { 2 } \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } \} } \end{array}$ reveals two distinct physical mechanisms during the burn-in phase:

1. Looseness in the Upper Bound (Undamped Fast-Mode Terms): The product term $\frac { 1 } { \gamma N } \Vert \mathbf { w } _ { 0 } -$ $\mathbf { w } ^ { * } \| _ { \mathbf { I } _ { 0 : k ^ { \dagger } } } ^ { 2 } \cdot \frac { k ^ { * } } { N }$ in (1.1) does not decay with the burn-in length s. Consequently, even when all eigenvalues satisfy $\lambda _ { i } \geq ( \gamma N ) ^ { - 1 } ( k ^ { * } = d$ and $\mathbf { H } _ { k ^ { * } : \infty } = \mathbf { 0 } ) , ( 1 . 1 )$ predicts a polynomial rate $\mathcal { O } ( d / ( \gamma N ^ { 2 } ) )$ instead of exponential burn-in decay (Jain et al., 2018). This looseness arises because prior upper-bound analyses drop the deterministic burn-in contraction acting on multiplicative gradient fluctuations after they are excited. Retaining the exact burn-in propagation shows that the single-hop $( i \leq k ^ { * } , j \leq k ^ { * } )$ contribution is exponentially damped in s through $( \mathbf { I } - \gamma \mathbf { H } ) ^ { s } ( \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } )$ and a burn-in cross-spectral convolution (Theorems 4.1 and 4.2), while multi-hop feedback (including energy that scatters into slow modes $i > k ^ { * }$ and subsequently returns to fast modes) is absorbed into the total excess risk whenever our spectral feedback condition $\chi _ { s , N } \leq \chi _ { 0 } < 1$ holds.

2. Looseness in the Lower Bound (Burn-In Head-to-Tail Transfer): Conversely, whenever slow spectral modes $i > k ^ { * }$ exist $( k ^ { * } < d )$ , the cross-interaction between the initial head error term $( \gamma N ) ^ { - 1 } \| \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } \| _ { \mathbf { I } _ { 0 : k } \mathrm {  f } } ^ { 2 }$ and the slow-spectrum factor $N \gamma ^ { 2 } \textstyle \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 }$ cannot be removed from the total-risk upper bound; instead, it is a genuine lower bound contribution missing from (1.2). During the first $\Theta ( 1 / ( \gamma \lambda _ { j } ) ,$ ) steps of the burn-in phase, an initial error component along a fast eigenvector ${ \bf v } _ { j } ~ ( j \leq k ^ { \dagger } )$ undergoes multiplicative gradient fluctuations $( \mathbf { x } _ { t } \mathbf { x } _ { t } ^ { \top } - \mathbf { H } ) ( \mathbf { w } _ { t - 1 } - \mathbf { w } ^ { * } )$ Under Assumption 3.4, this process scatters fourth-moment energy proportional to $\beta \gamma \| \mathbf { w } _ { 0 } -$ $\mathbf { w } ^ { * } \| _ { \mathbf { I } _ { 0 : k ^ { \dagger } } } ^ { 2 }$ H across all eigendirections of H. While the single-hop energy scattered into fast modes $( i \leq k ^ { * } )$ decays over the remaining burn-in steps, the energy scattered into slow modes $( i > k ^ { \dagger }$ , where $\lambda _ { i } < ( \gamma ( s + N ) ) ^ { - 1 } )$ ) experiences negligible contraction $( 1 - \gamma \lambda _ { i } ) ^ { 2 ( s - 1 - t ) } \geq e ^ { - 2 }$ and survives undamped into the averaging window—even in two-dimensional noiseless Gaussian regression (Proposition $4 . 7 ( 1 ) )$ .

Main Contributions. Our main contributions are summarized as follows (see Table 1 in Section 4.3 for a detailed term-by-term comparison with Zou et al. (2021)):

• Matching Excess Risk Bounds: Under the distributional assumptions of Zou et al. (2021)— Assumptions 3.1–3.3 for the upper bound (Theorem 4.1) and the well-specified model (3.4) with Assumptions 3.1, 3.2, and 3.4 for the lower bound (Theorem 4.2)—we derive an exact trajectory operator-polynomial identity (Lemma 5.1) and control repeated fourth-moment interactions via a scalar spectral feedback ratio $\chi _ { s , N }$ (Lemma 5.2). Whenever $\chi _ { s , N } \leq \chi _ { 0 } < 1$ ， the multi-hop feedback terms are absorbed within a factor of $( 1 - \chi _ { s , N } ) ^ { - 1 }$ , yielding explicit spectral upper and lower bounds for EfectiveBias and EfectiveVar that match up to constants depending only on $\alpha , \beta ,$ , and $1 - \chi _ { 0 }$ across all dimensions, initializations $\mathbf { w } _ { 0 } - \mathbf { w } ^ { * }$ , burn-in lengths $s \geq 0$ , and window sizes $N \geq 2$ (Theorems 4.1–4.2; and up to constants also depending on $c _ { 0 } , C _ { 0 }$ in the canonical specialization $c _ { 0 } N \leq s \leq C _ { 0 } N$ of Corollary 4.5).

• Stepsize Rules for Spectral Feedback Control: Because the tail-averaging filter weights $F _ { t }$ increase during the burn-in phase $1 \leq t \leq s .$ , standard monotone renewal arguments cannot bound $\chi _ { s , N }$ . By expanding $F _ { t }$ via a pair-midpoint identity, we establish two concrete constantstepsize regimes guaranteeing $\chi _ { s , N } \leq 1 / 2$ (Proposition 4.3): (i) for arbitrary spectra H and any finite horizon $T = s + N - 1 \ge 2$ , positive correlation of the kernel modes yields $\chi _ { s , N } \leq$ $\alpha \gamma ^ { 2 } ( T - 1 ) \operatorname { t r } ( \mathbf { H } ^ { 2 } ) ( 1 - \gamma \lambda _ { 1 } ) ^ { - 2 }$ , so any pre-run stepsize $\gamma \leq \operatorname* { m i n } \{ ( 2 \lambda _ { 1 } ) ^ { - 1 } , ( 8 \alpha ( T - 1 ) \operatorname { t r } ( \mathbf { H } ^ { 2 } ) ) ^ { - 1 / 2 } \}$ guarantees $\chi _ { s , N } \leq 1 / 2 ;$ and (ii) splitting the inner convolution at the midpoint bounds $\chi _ { s , N }$ through the kernel doubling constant $\begin{array} { r } { D _ { T } = \operatorname* { s u p } _ { 1 < h < T } k ( ( h - 1 ) / 2 ) / k ( h ) } \end{array}$ . For polynomially decaying spectra $\theta \Lambda i ^ { - a } \leq \lambda _ { i } \leq \Lambda i ^ { - a } ( a > 1 )$ , we prove $D _ { T } \leq D _ { * } ( a , \theta ) < \infty$ uniformly in $T _ { \ast }$ so a single horizon-independent stepsize ensures $\chi _ { s , N } \leq 1 / 2$ across all $s , N$ , and $\mathbf { w } _ { 0 } - \mathbf { w } ^ { * }$ (Corollary 4.4).

• Separation Constructions and Conditional Bounds: We establish two explicit separation results (Proposition 4.7): (i) in two-dimensional noiseless Gaussian regression with $\mathbf { H } =$ di $\mathrm { { s g } } ( 1 , ( 4 \gamma N ) ^ { - 1 } )$ and $\mathbf { w } _ { 0 } - \mathbf { w } ^ { * } = \mathbf { e } _ { 1 }$ , burn-in transfer from mode 1 to mode 2 forces excess risk $\Omega ( 1 / ( \gamma N ^ { 2 } ) )$ , whereas Zou et al. (2021, Theorem 5.2) gives the exponentially small bound $\mathcal { O } ( ( 1 - \gamma ) ^ { 2 N } / ( \gamma ^ { 2 } N ^ { 2 } ) )$ ; and (ii) in one-dimensional regression where $k ( h )$ decays exponentially and $D _ { T } \to \infty$ , two distributions sharing the same $( \mathbf { H } , \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } , \alpha , \beta )$ exhibit an exponential ratio gap $( r _ { 3 } / r _ { 2 } ) ^ { N } \to \infty$ , proving that no single expression depending only on $( { \bf H } , { \bf w } _ { 0 } -$ $\mathbf { w } ^ { * } , \gamma , s , N , \alpha , \beta , \sigma _ { \mathrm { n o i s e } } ^ { 2 } )$ can match both risks without an additional structural condition (such as $\chi _ { s , N } \leq \chi _ { 0 } < 1 )$ or state information (such as $\mathbf { B } _ { s }$ in Proposition 4.8).

Notation. We use lowercase letters $( a , b , \gamma , \lambda )$ to denote scalars, boldface lowercase letters $( \mathbf { w } , \mathbf { x } , \eta )$ to denote vectors in the separable Hilbert space H (where $\mathcal { H } = \mathbb { R } ^ { d } \mathrm { o r } \ell _ { 2 } )$ , boldface uppercase letters $( \mathbf { A } , \mathbf { B } , \mathbf { H } , \mathbf { I } )$ to denote bounded self-adjoint linear operators (matrices) on ${ \mathcal { H } } ,$ and calligraphic letters $( \mathcal { T } , \mathcal { M } , \mathcal { S } , \mathcal { T } , \mathcal { V } )$ to denote linear super-operators or scalar risk functionals. For a vector $\mathbf { w } \in \mathcal { H }$ , we write $( \mathbf { w } ) _ { i } : = \langle \mathbf { v } _ { i } , \mathbf { w } \rangle$ for its i-th coordinate in the eigenbasis of H, and for a positive semi-definite (PSD) operator $\mathrm { ~ \bf ~ A ~ } \succeq \mathrm { ~ \bf ~ 0 ~ }$ , we write $\| \mathbf { w } \| _ { \mathbf { A } } : = \sqrt { \langle \mathbf { w } , \mathbf { A w } \rangle }$ , where $\langle \mathbf { u } , \mathbf { v } \rangle = \mathbf { u } ^ { \top } \mathbf { v }$ . For two vectors $\mathbf { u } , \mathbf { v } \in \mathcal { H }$ , their outer product is the rank-one operator u $\otimes \mathbf { v } : = \mathbf { u } \mathbf { v } ^ { \top }$ ; for two bounded operators A, B on H, A⊗B denotes the linear super-operator acting on operators C via $( \mathbf { A } \otimes \mathbf { B } ) \circ \mathbf { C } : = \mathbf { A C B } ^ { \top }$ (so $( \mathbf { H } \otimes \mathbf { I } + \mathbf { I } \otimes \mathbf { H } ) \circ \mathbf { C } = \mathbf { H } \mathbf { C } + \mathbf { C } \mathbf { H }$ preserves self-adjointness); and for two operators A and B whose product $\mathbf { A } ^ { \top } \mathbf { B }$ is trace-class, their trace inner product is $\langle \mathbf { A } , \mathbf { B } \rangle : = \mathrm { t r } ( \mathbf { A } ^ { \top } \mathbf { B } )$ . Given the spectral decomposition $\begin{array} { r } { \mathbf { H } = \sum _ { i > 1 } \lambda _ { i } \mathbf { v } _ { i } \mathbf { v } _ { i } ^ { \top } } \end{array}$ with $\lambda _ { 1 } \geq \lambda _ { 2 } \geq \dots > 0$ , for any index pair $0 \leq k _ { 1 } \leq k _ { 2 } \leq \infty$ and finite head cutof $0 \leq k < \infty$ , we define the spectral projections and truncated operators

$$
\mathbf I _ { k _ { 1 } : k _ { 2 } } : = \sum _ { k _ { 1 } < i \leq k _ { 2 } } \mathbf v _ { i } \mathbf v _ { i } ^ { \top } , \qquad \mathbf H _ { k _ { 1 } : k _ { 2 } } : = \sum _ { k _ { 1 } < i \leq k _ { 2 } } \lambda _ { i } \mathbf v _ { i } \mathbf v _ { i } ^ { \top } , \qquad \mathbf H _ { 0 : k } ^ { - 1 } : = \sum _ { 1 \leq i \leq k } \lambda _ { i } ^ { - 1 } \mathbf v _ { i } \mathbf v _ { i } ^ { \top } ,
$$

and write $\begin{array} { r } { \| \mathbf { w } \| _ { \mathbf { H } ^ { - 1 } } ^ { 2 } : = \sum _ { i > 1 } \lambda _ { i } ^ { - 1 } ( \mathbf { w } ) _ { i } ^ { 2 } \in [ 0 , \infty ] } \end{array}$ for the inverse-covariance quadratic form. Throughout the paper, any spectral cutof max ${ \mathfrak { x } } \{ k : \lambda _ { k } \geq \tau \}$ is set to 0 when the defining set is empty. For sequences $f _ { n } , g _ { n } \ge 0$ , we write $f _ { n } \lesssim g _ { n }$ (or $f _ { n } = \mathcal { O } ( g _ { n } ) )$ if $f _ { n } \leq C g _ { n }$ for a positive constant $C ,$ $f _ { n } \gtrsim g _ { n }$ (or $f _ { n } = \Omega ( g _ { n } ) )$ if $g _ { n } \lesssim f _ { n } ,$ and $f _ { n } \asymp g _ { n } ~ ( \mathrm { o r } ~ f _ { n } = \Theta ( g _ { n } ) )$ if both $f _ { n } \lesssim g _ { n }$ and $f _ { n } \gtrsim g _ { n }$ hold. Finally, for real numbers $a , b \in \mathbb { R }$ , we write $a \wedge b : = \operatorname* { m i n } \{ a , b \}$ and $a \vee b : = \operatorname* { m a x } \{ a , b \}$

## 2 Related Work

Stochastic Approximation and Sufix/Tail Averaging. Stochastic gradient descent and Polyak–Ruppert iterate averaging have a rich history in stochastic approximation (Robbins and Monro, 1951; Polyak and Juditsky, 1992; Ruppert, 1988; Nemirovski et al., 2009). Because full-iterate averaging forgets initial conditions only at a polynomial rate, Rakhlin et al. (2012) introduced sufix (tail) averaging for strongly convex stochastic optimization. For least-squares regression in finite and non-parametric dimensions, Bach and Moulines (2013), D´efossez and Bach (2015), and Dieuleveut and Bach (2016); Dieuleveut et al. (2017) established convergence guarantees for constant- and decaying-stepsize SGD using operator recursions. Jain et al. $\mathrm { ( 2 0 1 7 a , b ) }$ developed a Markov chain operator perspective for constant-stepsize least-squares SGD, proving minimax optimal rates and analyzing tail-averaging to achieve exponential forgetting of the initial error in finite dimensions. Jain et al. (2018) and Ge et al. (2019) further studied accelerated and step-decay schedules for least-squares SGD, while Lakshminarayanan and Szepesv´ari (2018) examined constant-stepsize linear stochastic approximation with iterate averaging. In high-dimensional least-squares settings,

Paquette et al. (2021, 2022) characterized exact average-case SGD risk trajectories via Volterra integral equations.

Benign Overfitting and Implicit Regularization in Linear Regression. The phenomenon of benign overfitting in ofline linear regression was formalized by Bartlett et al. (2020), who proved matching upper and lower variance bounds and efective-rank characterizations for the minimum-$\ell _ { 2 } { \mathrm { - n o r m } }$ interpolator in terms of the data covariance H. Tsigler and Bartlett (2023) extended these sharp characterizations to ridge regression and minimum- $\cdot \ell _ { 2 } { \mathrm { - n o r m } }$ interpolation, while Hastie et al. (2022) and Muthukumar et al. (2020) studied asymptotic and finite-sample high-dimensional regression regimes (and Chatterji and Long (2021) analyzed overparameterized linear classification). More recently, Wu et al. (2025) established instance-wise risk comparisons showing that implicit regularization via gradient methods can dominate explicit ridge regularization in linear regression.

Sharp Risk Bounds for SGD with Weight Averaging in Linear Regression. Zou et al. (2021) bridged the Markov operator framework of Jain et al. (2017a,b) with the spectral partitioning technique of Bartlett et al. (2020) to derive instance-dependent upper and lower bounds for constant-stepsize online SGD with full iterate averaging (Zou et al., 2021, Theorems 2.1 and 2.2) and tail-averaging (Zou et al., 2021, Theorems 5.1 and 5.2). Related works have studied several complementary directions: Berthier et al. (2020) studied tight non-parametric rates in noiseless linear models; Wu et al. (2022) analyzed the last-iterate risk of SGD with polynomial and step-decay stepsize schedules; Li et al. (2024) established instance-dependent risk bounds for accelerated SGD with Nesterov momentum and tail averaging; Lin et al. (2024) and Li et al. (2025) studied scaling laws and learning-rate schedules under power-law spectral conditions; and Li and Gu (2025) analyzed SGD with exponential moving average (EMA), where a parallel gap between upper and lower bounds on the head initialization subspace appears between their Theorems 4.1 and 4.2. In large-scale deep learning and language model pretraining, checkpoint-based tail averaging—termed Latest Weight Averaging (LAWA) (Kaddour, 2022; Sanyal et al., 2023)—has been used widely to accelerate convergence, motivating a sharp theoretical understanding of tail-averaged SGD. Methodologically, Meterez et al. (2025) recently simplified the proof of Zou et al. (2021) by working in the diagonal eigenbasis of H while retaining the same gap on the head initialization subspace, and Meterez et al. (2026) studied horizon-free learning-rate schedules with weight averaging for anytime pretraining.

## 3 Problem Setup and Preliminaries

We consider linear regression over a separable Hilbert space H (which may be $\mathbb { R } ^ { d }$ or infinitedimensional). Let D be an unknown population distribution over feature-response pairs $( \mathbf { x } , y ) \in \mathcal { H } \times \mathbb { R }$ The population risk of a weight vector $\mathbf { w } \in \mathcal { H }$ is given by the expected squared loss

$$
L ( \mathbf { w } ) = \frac { 1 } { 2 } \mathbb { E } _ { ( \mathbf { x } , y ) \sim \mathcal { D } } \big [ \big ( y - \langle \mathbf { w } , \mathbf { x } \rangle \big ) ^ { 2 } \big ] .
$$

## 3.1 Tail-Averaged Stochastic Gradient Descent

Given an initialization $\mathbf { w } _ { 0 } \in \mathcal { H }$ and a stream of independent and identically distributed (i.i.d.) samples $\{ ( \mathbf { x } _ { t } , y _ { t } ) \} _ { t \geq 1 }$ drawn from D, constant-stepsize SGD updates the weight vector at iteration $t \geq 1$ according to

$$
\mathbf { w } _ { t } = \mathbf { w } _ { t - 1 } + \gamma \big ( y _ { t } - \langle \mathbf { w } _ { t - 1 } , \mathbf { x } _ { t } \rangle \big ) \mathbf { x } _ { t } ,\tag{3.1}
$$

where $\gamma > 0$ is the learning rate (stepsize). Given a burn-in period of length $s \geq 0$ and an averaging window of length $N \geq 1$ , SGD with tail-averaging outputs the average of the iterates from step s to step $s + N - 1$ :

$$
\overline { { \mathbf { w } } } _ { s : s + N } = \frac { 1 } { N } \sum _ { t = s } ^ { s + N - 1 } \mathbf { w } _ { t } .\tag{3.2}
$$

During the first s burn-in iterations, the algorithm executes constant-stepsize SGD updates (3.1) without recording the iterates into the running average; from iteration $t = s$ to $s + N - 1$ , it averages the subsequent N iterates to form $\overline { { \mathbf { w } } } _ { s : s + N }$ in (3.2). When $s = 0$ , (3.2) reduces to standard full-iterate averaging $\begin{array} { r } { \overline { { \mathbf { w } } } _ { 0 : N } = \frac { 1 } { N } \sum _ { t = 0 } ^ { N - 1 } \mathbf { w } _ { t } } \end{array}$ (Zou et al., 2021).

## 3.2 Model Assumptions

Following Zou et al. (2021), we adopt standard second- and fourth-moment regularity conditions on the feature distribution and label noise. Specifically, our upper bounds rely solely on Assumptions 3.1, 3.2, and 3.3, while our lower bounds rely on the well-specified linear model (3.4) together with Assumptions 3.1, 3.2, and 3.4.

Assumption 3.1 (Regularity Conditions; Zou et al. 2021, Assumption 2.1). The second moment operator $\mathbf { H } : = \mathbb { E } _ { ( \mathbf { x } , y ) \sim \mathcal { D } } [ \mathbf { x x } ^ { \top } ]$ is strictly positive definite $( \mathbf { H } \succ \mathbf { 0 } )$ and has finite trace $\mathrm { t r } ( { \bf H } ) < \infty$ Moreover, the population risk $L ( \mathbf { w } )$ admits a unique global minimizer $\mathbf { w } ^ { * } = \mathrm { a r g m i n } _ { \mathbf { w } \in \mathcal { H } } L ( \mathbf { w } )$ satisfying $L ( \mathbf { w } ^ { * } ) < \infty$

Let $\begin{array} { r } { \mathbf { H } = \sum _ { i > 1 } \lambda _ { i } \mathbf { v } _ { i } \mathbf { v } _ { i } ^ { \top } } \end{array}$ be the eigendecomposition of H, where the eigenvalues are ordered non-increasingly as $\lambda _ { 1 } \geq \lambda _ { 2 } \geq \dots > 0$ and $\textstyle \sum _ { i \geq 1 } \lambda _ { i } = \operatorname { t r } ( \mathbf { H } ) < \infty$ . To control the fourth-moment behavior of the stochastic rank-one updates $\mathbf { x } _ { t } \mathbf { x } _ { t } ^ { \mathsf { I } }$ , define the fourth-moment PSD super-operators $\mathcal { M }$ and $\widetilde { \mathcal { M } }$ acting on any symmetric operator A by

$$
\mathcal { M } \circ \mathbf { A } : = \mathbb { E } _ { ( \mathbf { x } , y ) \sim \mathcal { D } } \big [ \mathbf { x x } ^ { \top } \mathbf { A x x } ^ { \top } \big ] , \qquad \widetilde { \mathcal { M } } \circ \mathbf { A } : = \mathbf { H A H } .\tag{3.3}
$$

Assumption 3.2 (Fourth-Moment Upper Bound; Zou et al. 2021, Assumption 2.2). There exists a constant $\alpha > 0$ such that for any PSD operator $\mathbf { A } \succeq \mathbf { 0 }$ , it holds that

$$
\mathcal { M } \circ \mathbf { A } = \mathbb { E } _ { ( \mathbf { x } , y ) \sim \mathcal { D } } \left[ \mathbf { x x } ^ { \top } \mathbf { A x x } ^ { \top } \right] \preceq \alpha \operatorname { t r } ( \mathbf { H A } ) \mathbf { H } .
$$

As shown in Lemma C.4 (restated from Zou et al. (2021, Lemma A.1)), Assumption 3.2 holds with $\alpha = 1 6 \sigma _ { \gamma } ^ { 4 }$ whenever the standardized projections of x are $\sigma _ { z } ^ { 2 } \mathrm { - s u b \mathrm { - } G a u s s i a n }$ $( \mathrm { i . e . , } \mathbb { E } [ \exp ( \langle { \bf u } , { \bf x } \rangle ) ] \leq$ $\exp ( \sigma _ { z } ^ { 2 } \| \mathbf { u } \| _ { \mathbf { H } } ^ { 2 } / 2 )$ for all $\mathbf { u } \in \mathcal { H }$ , or $\mathbf { z } : = \mathbf { H } ^ { - 1 / 2 } \mathbf { x }$ is $\sigma _ { z } ^ { 2 } .$ -sub-Gaussian in finite dimension), and with $\alpha = 3$ when $\mathbf { x } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { H } )$ . Note also that taking $\mathbf { A } = \mathbf { v } _ { 1 } \mathbf { v } _ { 1 } ^ { \top }$ in $\mathbf { H } \mathbf { A } \mathbf { H } = { \widetilde { \mathcal { M } } } \circ \mathbf { A } \preceq { \mathcal { M } } \circ \mathbf { A } \preceq \alpha \operatorname { t r } ( \mathbf { H } \mathbf { A } ) \mathbf { H }$ yields $\lambda _ { 1 } ^ { 2 } \le \alpha \lambda _ { 1 } ^ { 2 }$ , so $\alpha \geq 1$ always holds; moreover, $( \mathcal { M } - \widetilde { \mathcal { M } } ) \circ \mathbf { A } \preceq \mathcal { M } \circ \mathbf { A } \preceq \alpha \mathrm { t r } ( \mathbf { H } \mathbf { A } ) \mathbf { H }$ for all $\mathbf { A } \succeq \mathbf { 0 }$

Assumption 3.3 (Noise Condition; Zou et al. 2021, Assumption 2.3). Define the residual noise $\boldsymbol { \xi } : = \boldsymbol { y } - \langle \mathbf { w } ^ { * } , \mathbf { x } \rangle$ and the noise-weighted covariance operator $\pmb { \Sigma } : = \mathbb { E } _ { ( \mathbf { x } , y ) \sim \mathcal { D } } [ \xi ^ { 2 } \mathbf { x } \mathbf { x } ^ { \top } ]$ . Suppose that

$$
\sigma ^ { 2 } : = \left. \mathbf { H } ^ { - 1 / 2 } \pmb { \Sigma } \mathbf { H } ^ { - 1 / 2 } \right. _ { 2 } < \infty ,
$$

or equivalently, $\pmb { \Sigma } \preceq \sigma ^ { 2 } \mathbf { H }$ in the Loewner PSD order.

Assumption 3.3 accommodates model misspecification and heteroscedastic noise, requiring only that $\mathbb { E } [ \xi \mathbf { x } ] = \nabla L ( \mathbf { w } ^ { * } ) = \mathbf { 0 }$ and $\pmb { \Sigma } \preceq \sigma ^ { 2 } \mathbf { H }$ . For the lower bound analysis, we consider the well-specified homoscedastic linear regression model (slightly relaxing Zou et al. (2021, Eq. (2.1)), which specifies Gaussian noise $\xi \sim \mathcal { N } ( 0 , \sigma _ { \mathrm { n o i s e } } ^ { 2 } )$ , to arbitrary centered finite-variance noise independent of x):

$$
y = \langle \mathbf { w } ^ { * } , \mathbf { x } \rangle + \xi , \qquad \mathrm { w h e r e ~ } \xi \perp \mathbf { x } , \quad \mathbb { E } [ \xi ] = 0 , \quad \mathrm { a n d } \quad \mathbb { E } [ \xi ^ { 2 } ] = \sigma _ { \mathrm { n o i s e } } ^ { 2 } < \infty .\tag{3.4}
$$

Under (3.4), $\pmb { \Sigma } = \sigma _ { \mathrm { n o i s e } } ^ { 2 } \mathbf { H }$ , so Assumption 3.3 holds with equality $( \sigma ^ { 2 } = \sigma _ { \mathrm { n o i s e } } ^ { 2 } )$ . Finally, to establish lower bounds on the fourth-moment bias-variance interaction, we adopt the fourth-moment lower bound condition of Zou et al. (2021).

Assumption 3.4 (Fourth-Moment Lower Bound; Zou et al. 2021, Assumption 2.4). There exists a constant $\beta \in ( 0 , \alpha ]$ such that for any PSD operator $\mathbf { A } \succeq \mathbf { 0 }$ , it holds that

$$
\begin{array} { r } { ( \mathcal { M } - \widetilde { \mathcal { M } } ) \circ \mathbf { A } = \mathbb { E } _ { ( \mathbf { x } , y ) \sim \mathcal { D } } \left[ \mathbf { x x } ^ { \top } \mathbf { A x x } ^ { \top } \right] - \mathbf { H A H } \succeq \beta \operatorname { t r } ( \mathbf { H A } ) \mathbf { H } . } \end{array}
$$

When $\mathbf { x } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { H } )$ , Isserlis’ theorem yields $\mathcal { M } \circ \mathbf { A } = 2 \mathbf { H } \mathbf { A } \mathbf { H } + \mathrm { t r } ( \mathbf { H } \mathbf { A } ) \mathbf { H }$ , so $\begin{array} { r } { ( \mathcal { M } - \widetilde { \mathcal { M } } ) \circ \mathbf { A } = } \end{array}$ $\mathbf { H A H } + \mathrm { t r } ( \mathbf { H A } ) \mathbf { H } \succeq \mathrm { t r } ( \mathbf { H A } ) \mathbf { H }$ and Assumption 3.4 holds with $\beta = 1$ (and with $\beta = 2$ in scalar Gaussian regression $d = 1 )$ .

## 3.3 Bias-Variance Decomposition and Efective Risk Components

Following Jain et al. (2017b), Zou et al. (2021), and Li and Gu (2025), let $\pmb { \eta } _ { t } : = \mathbf { w } _ { t } - \mathbf { w } ^ { * }$ denote the estimation error at iteration t and $\begin{array} { r } { \pmb { \eta } _ { s : s + N } : = \overline { { \mathbf { w } } } _ { s : s + N } - \mathbf { w } ^ { * } = \frac { 1 } { N } \sum _ { t = s } ^ { s + N - 1 } \eta _ { t } } \end{array}$ . By linearity of (3.1), $\pmb { \eta } _ { t } = \pmb { \eta } _ { t } ^ { \mathrm { b i a s } } + \pmb { \eta } _ { t } ^ { \mathrm { v a r i a n c e } }$ , where the noiseless bias process and zero-initialized variance process satisfy

$$
\eta _ { t } ^ { \mathrm { b i a s } } = \left( \mathbf { I } - \gamma \mathbf { x } _ { t } \mathbf { x } _ { t } ^ { \top } \right) \eta _ { t - 1 } ^ { \mathrm { b i a s } } , \qquad \quad \eta _ { 0 } ^ { \mathrm { b i a s } } = \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } ,\tag{3.5}
$$

$$
\eta _ { t } ^ { \mathrm { v a r i a n c e } } = \left( \mathbf { I } - \gamma \mathbf { x } _ { t } \mathbf { x } _ { t } ^ { \top } \right) \eta _ { t - 1 } ^ { \mathrm { v a r i a n c e } } + \gamma \xi _ { t } \mathbf { x } _ { t } , \qquad \eta _ { 0 } ^ { \mathrm { v a r i a n c e } } = \mathbf { 0 } .\tag{3.6}
$$

Here $\xi _ { t } : = y _ { t } - \langle \mathbf { w } ^ { * } , \mathbf { x } _ { t } \rangle$ , and the single-step second-moment operators are $\mathbf { B } _ { t } : = \mathbb { E } [ \pmb { \eta } _ { t } ^ { \mathrm { b i a s } } \otimes \pmb { \eta } _ { t } ^ { \mathrm { b i a s } } ]$ and $\mathbf { C } _ { t } : = \mathbb { E } | \boldsymbol { \eta } _ { t } ^ { }$ variance $\otimes \pmb { \eta } _ { t } ^ { \backslash }$ <sup>variance</sup>] (initialized at ${ \bf B } _ { 0 } = ( { \bf w } _ { 0 } - { \bf w } ^ { * } ) \otimes ( { \bf w } _ { 0 } - { \bf w } ^ { * } )$ and $\mathbf { C } _ { 0 } = \mathbf { 0 } )$ . Letting $\begin{array} { r } { \bar { \pmb { \eta } } _ { s : s + N } ^ { \mathrm { b i a s } } : = \frac { 1 } { N } \dot { \sum _ { t = s } ^ { s + N - 1 } } \pmb { \eta } _ { t } ^ { \mathrm { b i a s } } } \end{array}$ and $\begin{array} { r } { \bar { \eta } _ { s : s + N } ^ { \mathrm { v a r i a n c e } } : = \frac { 1 } { N } \sum _ { t = s } ^ { s + N - 1 } \eta _ { t } ^ { \mathrm { v a r i a n c e } } } \end{array}$ , Lemma C.6 (Jain et al., 2017b; Zou et al., 2021) shows that $\mathbb { E } [ L ( \overline { { \mathbf { w } } } _ { s : s + N } ) ] - L ( \mathbf { w } ^ { * } ) \leq 2$ · bias + 2 · variance under Assumptions 3.1–3.3, and $\mathbb { E } [ L ( \overline { { \mathbf { w } } } _ { s : s + N } ) ] - L ( \mathbf { w } ^ { * } ) =$ bias + variance under the well-specified model (3.4), where

$$
\mathrm { b i a s : } = \frac { 1 } { 2 } \left. \mathbf { H } , \mathbb { E } \left[ \bar { \eta } _ { s : s + N } ^ { \mathrm { b i a s } } \otimes \bar { \eta } _ { s : s + N } ^ { \mathrm { b i a s } } \right] \right. , \qquad \mathrm { v a r i a n c e : } = \frac { 1 } { 2 } \left. \mathbf { H } , \mathbb { E } \left[ \bar { \eta } _ { s : s + N } ^ { \mathrm { v a r i a n c e } } \otimes \bar { \eta } _ { s : s + N } ^ { \mathrm { v a r i a n c e } } \right] \right. .\tag{3.7}
$$

Because the rank-one updates $\mathbf { x } _ { t } \mathbf { x } _ { t } ^ { \top }$ in (3.5) fluctuate around ${ \bf H } = \mathbb { E } [ { \bf x } _ { t } { \bf x } _ { t } ^ { \top } ]$ , the bias process has mean $\mathbb { E } [ \pmb { \eta } _ { t } ^ { \mathrm { b i a s } } ] = ( \mathbf { I } - \gamma \mathbf { H } ) ^ { t } ( \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } )$ together with stochastic fluctuations $\pmb { \eta } _ { t } ^ { \mathrm { b i a s } } - \mathbb { E } [ \pmb { \eta } _ { t } ^ { \mathrm { b i a s } } ]$ driven by multiplicative gradient noise $( \mathbf { x } _ { t } \mathbf { x } _ { t } ^ { \top } - \mathbf { H } ) \pmb { \eta } _ { t - 1 } ^ { \mathrm { b i a s } }$ . Following Zou et al. (2021) and Li and Gu (2025), we therefore decompose bias + variance = EfectiveBias + EfectiveVar into the deterministic drift bias EfectiveBias and the total stochastic variance error EfectiveVar:

$$
\mathrm { E f f e c t i v e B i a s } : = \frac { 1 } { 2 } \| \mathbb { E } \big [ \overline { { \eta } } _ { s : s + N } ^ { \mathrm { b i a s } } \big ] \| _ { \mathbf { H } } ^ { 2 } = \sum _ { i = 1 } ^ { \infty } \frac { ( \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } ) _ { i } ^ { 2 } } { 2 \gamma ^ { 2 } N ^ { 2 } \lambda _ { i } } ( 1 - \gamma \lambda _ { i } ) ^ { 2 s } \big ( 1 - ( 1 - \gamma \lambda _ { i } ) ^ { N } \big ) ^ { 2 } ,\tag{3.8}
$$

$$
\mathrm { E f f e c t i v e V a r : } = \mathrm { ( b i a s - E f f e c t i v e B i a s ) } + \mathrm { v a r i a n c e } = \frac { 1 } { 2 } \mathbb { E } \| \bar { \eta } _ { s : s + N } ^ { \mathrm { b i a s } } - \mathbb { E } \big [ \bar { \eta } _ { s : s + N } ^ { \mathrm { b i a s } } \big ] \| _ { \mathbf { H } } ^ { 2 } + \mathrm { v a r i a n c e } ,\tag{3.9}
$$

where EfectiveBias $\begin{array} { r } { \mathbf { \eta } = \frac { 1 } { 2 } \| \frac { 1 } { N } \sum _ { t = s } ^ { s + N - 1 } ( \mathbf { I } - \gamma \mathbf { H } ) ^ { t } ( \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } ) \| _ { \mathbf { H } } ^ { 2 } } \end{array}$ is the excess risk of deterministic gradient descent averaged over $[ s , s + N - 1 ]$ . The efective variance in (3.9) is the sum of the bias-induced variance bias − EfectiveBias and the additive-noise variance variance.

## 4 Main Results

In this section, we present our main upper and lower bounds on the excess risk $\mathbb { E } [ L ( \overline { { \mathbf { w } } } _ { s : s + N } ) ] - L ( \mathbf { w } ^ { * } )$ of SGD with tail-averaging (Section 4.1), establish concrete stepsize rules guaranteeing the spectral feedback condition for general and polynomially decaying spectra (Section 4.2), specialize our bounds to the canonical regime $s = \Theta ( N )$ to compare directly with Zou et al. (2021) (Section 4.3), and present our separation constructions and conditional bounds (Section 4.4).

Before stating our theorems, recall from Section 3.3 the deterministic drift bias EfectiveBias in (3.8) and the efective variance EfectiveVar in (3.9), and let $T : = s + N - 1$ denote the final iteration index in (3.2). For each step $t \in \{ 1 , \ldots , T \}$ and real lag $h \geq 0$ , we define the tail-averaging filter polynomial $q _ { t } ( \lambda )$ , the receiving sensitivity weight $F _ { t }$ , and the propagation kernel $k ( h ) \ \mathrm { b y ^ { 1 } }$

$$
q _ { t } ( \lambda ) : = \frac { 1 } { N } \sum _ { m = s \vee t } ^ { T } ( 1 - \gamma \lambda ) ^ { m - t } , \qquad F _ { t } : = \sum _ { i = 1 } ^ { \infty } \lambda _ { i } ^ { 2 } q _ { t } ( \lambda _ { i } ) ^ { 2 } , \qquad k ( h ) : = \sum _ { i = 1 } ^ { \infty } \lambda _ { i } ^ { 2 } ( 1 - \gamma \lambda _ { i } ) ^ { 2 h } ,\tag{4.1}
$$

together with the spectral feedback coeficient

$$
\chi _ { s , N } : = \alpha \gamma ^ { 2 } \operatorname* { m a x } _ { 1 \leq r < T } \frac { \sum _ { t = r + 1 } ^ { T } k ( t - r - 1 ) F _ { t } } { F _ { r } } .\tag{4.2}
$$

Here $F _ { t }$ measures the sensitivity of $\overline { { \mathbf { w } } } _ { s : s + N }$ to a stochastic gradient perturbation at step $t ,$ while $\chi _ { s , N }$ measures the worst-case single-hop amplification of fourth-moment energy across the trajectory (see Section 5.1 for a detailed derivation).

## 4.1 Upper and Lower Bounds of Excess Risk

We first present our upper bound on the excess risk of SGD with tail-averaging.

Theorem 4.1 (Upper Bound). Consider SGD with tail-averaging (3.2). Suppose Assumptions $\it 3 . 1$ 3.2, and 3.3 hold, and that the stepsize and spectral feedback coeficient satisfy $\gamma \leq 1 / ( 2 \lambda _ { 1 } )$ and $\chi _ { s , N } < 1$ . Then for any $s \geq 0$ and $N \geq 1$ , the excess risk can be upper bounded as follows:

$$
\begin{array} { r } { \mathbb { E } \big [ L ( \overline { { \mathbf { w } } } _ { s : s + N } ) \big ] - L ( \mathbf { w } ^ { * } ) \leq 2 \cdot \mathrm { E f f e c t i v e B i a s } + 2 \cdot \mathrm { E f f e c t i v e V a r } , } \end{array}
$$

where the factor $o f 2$ can be omitted under the well-specified model (3.4), the efective bias in (3.8) satisfies

$$
\mathrm { E f f e c t i v e B i a s } \leq \frac { 1 } { 2 \gamma ^ { 2 } N ^ { 2 } } \cdot \Vert ( \mathbf { I } - \gamma \mathbf { H } ) ^ { s } ( \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } ) \Vert _ { \mathbf { H } _ { 0 ; k ^ { * } } ^ { - 1 } } ^ { 2 } + \frac { 1 } { 2 } \cdot \Vert ( \mathbf { I } - \gamma \mathbf { H } ) ^ { s } ( \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } ) \Vert _ { \mathbf { H } _ { k ^ { * } ; \infty } } ^ { 2 } ,\tag{4.3}
$$

and the efective variance in (3.9) satisfies<sup>2</sup>

$$
\begin{array} { r l } & { \mathrm { E f f e c t i v e V a r } \le \displaystyle \frac { \alpha \left( \| ( \mathbf { I } - \gamma \mathbf { H } ) ^ { s } ( \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } ) \| _ { \mathbf { I } _ { 0 , k ^ { * } } } ^ { 2 } + N \gamma \| ( \mathbf { I } - \gamma \mathbf { H } ) ^ { s } ( \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } ) \| _ { \mathbf { H } _ { k ^ { * } ; \infty } } ^ { 2 } \right) } { N \gamma ( 1 - \chi _ { s , N } ) } \Big ( \frac { k ^ { * } } { N } + N \gamma ^ { 2 } \displaystyle \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } \Big ) } \\ & { \qquad + \displaystyle \frac { \alpha } { 1 - \chi _ { s , N } } \displaystyle \sum _ { i , j = 1 } ^ { \infty } \operatorname* { m i n } \Big \{ \frac { 1 } { N ^ { 2 } } , \gamma ^ { 2 } \lambda _ { i } ^ { 2 } \Big \} \frac { \lambda _ { j } ( \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } ) _ { j } ^ { 2 } ( 1 - \gamma ( \lambda _ { i } \wedge \lambda _ { j } ) ) ^ { 2 ( s - 1 ) } } { s ^ { - 1 } \vee \gamma | \lambda _ { i } - \lambda _ { j } | } } \end{array}
$$

$$
+ \frac { \sigma ^ { 2 } } { 1 - \chi _ { s , N } } \cdot ( \frac { k ^ { * } } { N } + \gamma \sum _ { k ^ { * } < i \leq k ^ { \dagger } } \lambda _ { i } + ( s + N ) \gamma ^ { 2 } \sum _ { i > k ^ { \dagger } } \lambda _ { i } ^ { 2 } ) ,\tag{4.4}
$$

where the eigenvalue cutofs are defined as

$$
k ^ { * } = \operatorname* { m a x } \Bigl \{ k : \lambda _ { k } \geq \frac { 1 } { \gamma N } \Bigr \} , \qquad k ^ { \dagger } = \operatorname* { m a x } \Bigl \{ k : \lambda _ { k } \geq \frac { 1 } { \gamma ( s + N ) } \Bigr \} .\tag{4.5}
$$

We next establish our matching lower bound on the excess risk of SGD with tail-averaging:

Theorem 4.2 (Lower Bound). Consider SGD with tail-averaging (3.2) under the well-specified model (3.4). Suppose Assumptions 3.1, 3.2, and $\it 3 . 4$ hold, and the hyperparameters satisfy $\gamma \leq 1 / ( 2 \lambda _ { 1 } )$ and $N \geq 2$ . Then for any $s \geq 0$ , the excess risk decomposes as

$$
\mathbb { E } \big [ L ( \overline { { \mathbf { w } } } _ { s : s + N } ) \big ] - L ( \mathbf { w } ^ { * } ) = \mathrm { E f f e c t i v e B i a s } + \mathrm { E f f e c t i v e V a r } ,
$$

where the efective bias in (3.8) satisfies

$$
\mathrm { E f f e c t i v e B i a s } \geq \frac { 1 } { 8 \gamma ^ { 2 } N ^ { 2 } } \cdot \Vert ( \mathbf { I } - \gamma \mathbf { H } ) ^ { s } ( \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } ) \Vert _ { \mathbf { H } _ { 0 : k ^ { * } } ^ { - 1 } } ^ { 2 } + \frac { 1 } { 8 } \cdot \Vert ( \mathbf { I } - \gamma \mathbf { H } ) ^ { s } ( \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } ) \Vert _ { \mathbf { H } _ { k ^ { * } ; \infty } } ^ { 2 } ,\tag{4.6}
$$

and the efective variance in (3.9) satisfies

$$
\begin{array} { r l r } { \mathrm { E f f e c t i v e V a r } \ge \displaystyle \frac { \beta \left( \| ( \mathbf { I } - \gamma \mathbf { H } ) ^ { s } ( \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } ) \| _ { \mathbf { I } _ { 0 ; k } } ^ { 2 } + N \gamma \| ( \mathbf { I } - \gamma \mathbf { H } ) ^ { s } ( \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } ) \| _ { \mathbf { H } _ { k ; \infty , \infty } } ^ { 2 } \right) } { 8 6 4 N \gamma } \Big ( \frac { k ^ { * } } { N } + N \gamma ^ { 2 } \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } \Big ) } & \\ { + \displaystyle \frac { \beta } { 1 0 8 } \sum _ { i , j = 1 } ^ { \infty } \operatorname* { m i n } \Big \{ \frac { 1 } { N ^ { 2 } } , \gamma ^ { 2 } \lambda _ { i } ^ { 2 } \Big \} \frac { \lambda _ { j } ( \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } ) _ { j } ^ { 2 } ( 1 - \gamma ( \lambda _ { i } \wedge \lambda _ { j } ) ) ^ { 2 ( s - 1 ) } } { s ^ { - 1 } \vee \gamma | \lambda _ { i } - \lambda _ { j } | } } & \\ { + \displaystyle \frac { \sigma _ { \mathrm { n o i s e } } ^ { 2 } } { 2 1 6 } \cdot \left( \frac { k ^ { * } } { N } + \gamma \sum _ { k ^ { * } < i \leq k ^ { \sharp } } \lambda _ { i } + ( s + N ) \gamma ^ { 2 } \sum _ { i > k ^ { \sharp } } \lambda _ { i } ^ { 2 } \right) , } & { ( 4 . 7 ) } & \end{array}
$$

with $k ^ { * }$ and $k ^ { \dagger }$ defined in (4.5).

Structure and Interpretation of the Bounds. The proofs of Theorems 4.1 and 4.2 are given in Appendix B. The bounds on EfectiveBias and EfectiveVar separate the following contributions:

• Efective Bias: In (4.3) and (4.6), the initial error $\mathbf { w } _ { 0 } - \mathbf { w } ^ { * }$ is first contracted by $( \mathbf { I } - \gamma \mathbf { H } ) ^ { s }$ during the s burn-in steps, and then enjoys $\mathcal { O } ( ( \gamma ^ { 2 } N ^ { 2 } \lambda _ { i } ) ^ { - 1 } )$ averaging decay on the head subspace $\mathbf { I } _ { 0 : k ^ { * } }$ while remaining unaveraged on the tail subspace $\mathbf { H } _ { k ^ { * } : \infty }$

• Efective Variance: Both (4.4) and (4.7) consist of three matching components: (i) the averaging-window multiplicative variance (first line), which has the exact form of full-averaging SGD restarted from the burn-in drifted vector $( \mathbf { I } - \gamma \mathbf { H } ) ^ { s } ( \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } )$ at step $s ; ( \mathrm { i i } )$ the burn-in cross-spectral transfer variance (second line), where initial error in source mode $j$ scatters fourth-moment energy into mode i during the s burn-in steps with efective interaction duration min $\{ s , ( \gamma | \lambda _ { i } - \lambda _ { j } | ) ^ { - 1 } \}$ and slower-mode damping $( 1 - \gamma ( \lambda _ { i } \wedge \lambda _ { j } ) ) ^ { 2 ( s - 1 ) }$ ; and (iii) the additive label-noise variance (third line), partitioned across $i \leq k ^ { * } , k ^ { * } < i \leq k ^ { \dagger }$ , and $i > k ^ { \dagger }$

Whenever $\chi _ { s , N } \leq \chi _ { 0 } < 1$ for a constant $\chi _ { 0 } < 1$ , every term in Theorems 4.1 and 4.2 matches up to a multiplicative factor depending only on $\alpha , \beta .$ , and $1 - \chi _ { 0 }$ across all dimensions, spectra, initializations, and horizons $s \ge 0 , N \ge 2$ . Moreover, when $s = 0$ , the burn-in sum vanishes and Theorems 4.1 and 4.2 recover Zou et al. (2021, Theorems 2.1 and 2.2) (while extending the lower bound from $N \geq 5 0 0$ to all $N \geq 2 )$

## 4.2 Guaranteeing the Spectral Feedback Condition $\chi _ { s , N } \leq \chi _ { 0 } < 1$

We now show how the spectral feedback condition $\chi _ { s , N } \leq 1 / 2$ can be guaranteed prior to execution, both for arbitrary covariance spectra H (under a horizon-dependent constant stepsize) and for polynomially decaying spectra $\lambda _ { i } \asymp i ^ { - a }$ (under a single horizon-independent constant stepsize). Recall $\begin{array} { r } { k ( h ) = \sum _ { i \geq 1 } \lambda _ { i } ^ { 2 } ( 1 - \gamma \lambda _ { i } ) ^ { 2 h } } \end{array}$ for real $h \geq 0$ , and define its doubling constant over [1, T] by

$$
D _ { T } : = \operatorname* { s u p } _ { 1 \leq h \leq T } { \frac { k ( ( h - 1 ) / 2 ) } { k ( h ) } } .\tag{4.8}
$$

Proposition 4.3 (Pair-Midpoint Convolution Bounds on $\chi _ { s , N }$ and General-Spectrum Stepsize Rule). Suppose Assumption 3.1 holds and $0 < \gamma < 1 / \lambda _ { 1 }$ . Then for every $s \geq 0$ and $N \geq 1$ (with $T = s + N - 1 \geq 0 )$ , the receiving weights $F _ { t }$ in (4.1) satisfy the exact pair-midpoint identity

$$
F _ { t } = \frac { 1 } { N ^ { 2 } } \sum _ { a = s \vee t } ^ { T } \sum _ { b = s \vee t } ^ { T } k \Big ( \frac { a + b } { 2 } - t \Big ) , \qquad \forall t \in \{ 1 , . . . , T \} .\tag{4.9}
$$

Consequently, $\chi _ { s , N } = 0$ whenever $T \leq 1$ , and for all $T = s + N - 1 \geq 2$ , the spectral feedback coeficient $\chi _ { s , N }$ in (4.2) satisfies both:

1. (General-Spectrum Horizon-Dependent Bound): For any covariance operator H and $T \geq 2$

$$
\chi _ { s , N } \leq \frac { \alpha \gamma ^ { 2 } ( T - 1 ) \operatorname { t r } ( \mathbf { H } ^ { 2 } ) } { ( 1 - \gamma \lambda _ { 1 } ) ^ { 2 } } ,\tag{4.10}
$$

so choosing any constant pre-run stepsize satisfying

$$
\gamma \leq \operatorname* { m i n } \left\{ \frac { 1 } { 2 \lambda _ { 1 } } , \frac { 1 } { \sqrt { 8 \alpha ( T - 1 ) \operatorname { t r } ( \mathbf { H } ^ { 2 } ) } } \right\} \quad \Longrightarrow \quad \chi _ { s , N } \leq \frac { 1 } { 2 } .\tag{4.11}
$$

2. (Kernel-Doubling Bound): In terms of the doubling constant $D _ { T }$ in (4.8),

$$
\chi _ { s , N } \leq 2 \alpha D _ { T } \gamma ^ { 2 } \sum _ { n = 0 } ^ { \infty } k ( n ) = 2 \alpha D _ { T } \gamma \sum _ { i = 1 } ^ { \infty } \frac { \lambda _ { i } } { 2 - \gamma \lambda _ { i } } \leq \frac { 2 \alpha D _ { T } \gamma \operatorname { t r } ( \mathbf { H } ) } { 2 - \gamma \lambda _ { 1 } } .\tag{4.12}
$$

Proposition 4.3(1) guarantees $\chi _ { s , N } \leq 1 / 2$ (and hence matching upper and lower bounds in Theorems 4.1 and 4.2) for arbitrary spectra H under the horizon-dependent constant stepsize (4.11). Moreover, when the eigenvalues of H decay polynomially as $\lambda _ { i } \asymp i ^ { - a } ( a > 1 )$ —the standard power-law family studied in Caponnetto and De Vito (2007); Dieuleveut and Bach (2016); Bartlett et al. (2020); Zou et al. (2021); Li and Gu (2025)—the propagation kernel decays polynomially as $k ( h ) \asymp \Lambda ^ { 2 } ( 1 + \gamma \Lambda h ) ^ { - ( 2 - 1 / a ) }$ . In this regime, Proposition 4.3(2) strengthens (4.11) to a single horizon-independent stepsize $\gamma$ ensuring $\chi _ { s , N } \leq 1 / 2$ across all s and $N$

Corollary 4.4 (Uniform Horizon-Free Matching under Polynomial Eigenvalue Decay). Suppose there exist constants $a > 1 , \Lambda > 0 _ { }$ , and $\theta \in ( 0 , 1 ]$ such that $\theta \Lambda i ^ { - a } \leq \lambda _ { i } \leq \Lambda i ^ { - a }$ for all $i \geq 1$ (in infinite dimension) or for $1 \leq i \leq$ d (in finite dimension $d \geq 2$ , over horizons $\gamma \Lambda T \leq ( d / 2 ) ^ { a } )$ . Then there exists a finite constant $D _ { * } = D _ { * } ( a , \theta ) \geq 1$ such that $\gamma \Lambda \leq 1 / 2$ implies $D _ { T } \leq D _ { * }$ , and hence any fixed stepsize satisfying

$$
\gamma \leq \operatorname* { m i n } \left\{ \frac { 1 } { 2 \Lambda } , \frac { 1 } { 4 \alpha D _ { * } \operatorname { t r } ( \mathbf { H } ) } \right\}\tag{4.13}
$$

ensures $\chi _ { s , N } \ \leq \ 1 / 2$ . Consequently, under (3.4) and Assumptions 3.1–3.4, the upper bound of Theorem 4.1 and the lower bound of Theorem 4.2 match up to constants depending only on α and β uniformly over all $\mathbf { w } _ { 0 } - \mathbf { w } ^ { * } , s \geq 0$ , and $N \geq 2$ (and Corollary $4 . 5$ below holds whenever $c _ { 0 } N \leq s \leq C _ { 0 } N$ up to constants depending only on $\alpha , \beta , c _ { 0 } , C _ { 0 } )$

## 4.3 Canonical Regime $s = \Theta ( N )$ and Comparison with Zou et al. (2021)

In the standard tail-averaging regime where $s = \Theta ( N )$ (for instance, $s \ = \ N$ , which averages the second half of a 2N-step run as in Jain et al. (2018) and Zou et al. (2021, Section 5)), the eigenvalue cutofs k<sup>∗</sup> and k<sup>†</sup> have thresholds $( \gamma N ) ^ { - 1 }$ and $( \gamma ( s + N ) ) ^ { - }$ <sup>−1</sup> within a constant factor of each other, so $\| \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } \| _ { \mathbf { H } _ { k ^ { * } ; \infty } } ^ { 2 } \asymp \| ( \mathbf { I } - \gamma \mathbf { H } ) ^ { s } ( \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } ) \| _ { \mathbf { H } _ { k ^ { * } ; \infty } } ^ { 2 }$ and $\gamma \lambda _ { i } \asymp N \gamma ^ { 2 } \lambda _ { i } ^ { 2 } \asymp N ^ { - 1 }$ for all $k ^ { * } < i \leq k ^ { \dagger }$ . Splitting the burn-in variance sum in (4.4) and $( 4 . 7 )$ into fast index modes $( i \leq k ^ { * }$ where min $\{ N ^ { - 2 } , \gamma ^ { 2 } \lambda _ { i } ^ { 2 } \} = N ^ { - 2 } )$ and slow index modes $( i > k ^ { * }$ , where min $\{ N ^ { - 2 } , \gamma ^ { 2 } \lambda _ { i } ^ { 2 } \} = \gamma ^ { 2 } \lambda _ { i } ^ { 2 } )$ yields the following corollary and the term-by-term comparison in Table 1.

Corollary 4.5 (Matching Excess Risk Bounds when $s = \Theta ( N ) )$ . Suppose $c _ { 0 } N \leq s \leq C _ { 0 } N$ for constants $0 < c _ { 0 } \le C _ { 0 } < \infty$ , and suppose the assumptions of Theorems $4 . 1$ and $4 . 2$ hold with $\chi _ { s , N } \leq \chi _ { 0 } < 1$ . Then the excess risk satisfies

$$
\mathrm { E f f e c t i v e B i a s } = \Theta \left( \frac { 1 } { \gamma ^ { 2 } N ^ { 2 } } \cdot \lVert ( { \mathbf I } - \gamma { \mathbf H } ) ^ { s } ( { \mathbf w } _ { 0 } - { \mathbf w } ^ { * } ) \rVert _ { { \mathbf H } _ { 0 ; k ^ { * } } ^ { 2 } } ^ { 2 } + \lVert { \mathbf w } _ { 0 } - { \mathbf w } ^ { * } \rVert _ { { \mathbf H } _ { k ^ { * } ; \infty } } ^ { 2 } \right) ,\tag{4.14}
$$

$$
\begin{array} { r l } { \mathrm { E f f e c t i v e V a r } = \Theta \Bigg ( \displaystyle \frac { \| \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } \| _ { \Omega _ { 1 ; k ^ { * } } } ^ { 2 } + N \gamma \| \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } \| _ { \mathbf { H } _ { k ^ { * } ; \infty } } ^ { 2 } } { N \gamma } \cdot \left( N \gamma ^ { 2 } \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } \right) } & { } \\ { + \displaystyle \frac { \| ( \mathbf { I } - \gamma \mathbf { H } ) ^ { s } ( \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } ) \| _ { \Omega _ { 1 ; k ^ { * } } } ^ { 2 } + N \gamma \| \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } \| _ { \mathbf { H } _ { k ^ { * } ; \infty } } ^ { 2 } } { N \gamma } \cdot \frac { k ^ { * } } { N } } & { } \\ { + \displaystyle \frac { 1 } { N ^ { 2 } } \displaystyle \sum _ { i \leq k ^ { * } } \sum _ { j \leq k ^ { * } } \lambda _ { j } ( \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } ) _ { j } ^ { 2 } \big ( 1 - \gamma ( \lambda _ { i } \wedge \lambda _ { j } ) \big ) ^ { 2 ( s - 1 ) } \operatorname* { m i n } \left\{ s , \frac { 1 } { \gamma | \lambda _ { i } - \lambda _ { j } | } \right\} } & { } \\ { + \sigma _ { \operatorname* { m i s e } } ^ { 2 } \cdot \left( \frac { k ^ { * } } { N } + N \gamma ^ { 2 } \displaystyle \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } \right) \Bigg ) , } & { ( ) } \end{array}\tag{4.15}
$$

where the constants hidden in Θ(·) depend only on $\alpha , \beta , 1 - \chi _ { 0 } , c _ { 0 }$ , and $C _ { 0 }$ .

Remark 4.6 (Resolution of the Total-Risk Gap in Zou et al. (2021) and Role of Multi-Hop Return Paths). Comparing (4.15) with Zou et al. (2021, Theorems 5.1 and 5.2) (see Table 1) clarifies both why the rank-one factored form in prior work could not match and how the terms of (4.15) combine in the total excess risk:

1. Presence of Slow Spectral Modes $( k ^ { * } < d ,$ so $\textstyle \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } > 0 )$ : In Panel B $( i > k ^ { * } )$ , the first line of (4.15) retains the undamped head initialization contribution $\begin{array} { r } { { \frac { 1 } { \gamma N } } \| \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } \| _ { \mathbf { I } _ { 0 : k ^ { * } } } ^ { 2 } \cdot \left( N \gamma ^ { 2 } \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } \right) } \end{array}$ which is missing from Zou et al. (2021, Theorem 5.2). In multi-dimensional problems with $k ^ { * } < d ,$ energy scattered into slow modes $i > k ^ { * }$ can also scatter back into fast modes $i \leq k ^ { * }$ under $\chi _ { s , N } \leq \chi _ { 0 } < 1$ , this multi-hop return contribution is absorbed into the total-risk bound via $( 1 - \chi _ { s , N } ) ^ { - 1 }$ multiplying the full single-hop expression in (4.15).<sup>3</sup>

2. Well-Resolved Spectral Regime $( d \leq k ^ { * }$ , so $\mathbf { H } _ { k ^ { * } : \infty } = \mathbf { 0 } )$ : Conversely, when all eigenvalues satisfy $\lambda _ { i } \geq ( \gamma N ) ^ { - 1 }$ , the slow subspace is empty $\begin{array} { r } { ( \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } = 0 ) } \end{array}$ , so Panel B vanishes and no slow-mode return path exists. Every remaining bias-induced variance term in Panel A (the second and third lines of (4.15)) is exponentially damped in s by $( 1 - \gamma ( \lambda _ { i } \wedge \lambda _ { j } ) ) ^ { 2 ( s - 1 ) }$ eliminating the undamped $\begin{array} { r } { \frac { 1 } { \gamma N } \vline \| \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } \| _ { \mathbf { I } _ { 0 : k ^ { \dagger } } } ^ { 2 } \cdot \frac { k ^ { * } } { N } } \end{array}$ artifact in Zou et al. (2021, Theorem 5.1).

Table 1: Comparison of the fast-mode $( i \leq k ^ { * } )$ and slow-mode $( i > k ^ { * } )$ index components in the total bias-induced variance bound for tail-averaged SGD $\overline { { \mathbf { w } } } _ { s : s + N }$ in the canonical regime $c _ { 0 } N \leq s \leq C _ { 0 } N$ omitting multiplicative constants depending on $\alpha , \beta , c _ { 0 } , C _ { 0 } ,$ , and the stability margin $( 1 - \gamma \alpha \operatorname { t r } ( \mathbf { H } )$ in $\mathrm { Z o u }$ et al. (2021, Theorem 5.1), and $1 - \chi _ { 0 }$ in Corollary 4.5 under $\chi _ { s , N } \leq \chi _ { 0 } < 1 )$
<table><tr><td>Bound</td><td>Panel A: Fast-Mode Index Component  $( i \leq k ^ { * } )$ </td></tr><tr><td>Zou et al. (2021, Theorem 5.1) (Upper)</td><td> $\frac { \| \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } \| _ { \mathbf { I } _ { 0 : k ^ { \dagger } } } ^ { 2 } + N \gamma \| \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } \| _ { \mathbf { H } _ { k ^ { \dagger } : \infty } } ^ { 2 } } { N \gamma } \cdot \frac { k ^ { * } } { N }$ </td></tr><tr><td>Zou et al. (2021, Theorem 5.2) (Lower)</td><td> $\| \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } \| _ { \mathbf { H } _ { k ^ { \dagger } : \infty } } ^ { 2 } \cdot \frac { k ^ { * } } { N }$   $\frac { \| ( \mathbf { I } - \gamma \mathbf { H } ) ^ { s } ( \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } ) \| _ { \mathbf { I } _ { 0 : k ^ { * } } } ^ { 2 } + N \gamma \| \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } \| _ { \mathbf { H } _ { k ^ { * } ; \infty } } ^ { 2 } } { N \gamma } \cdot \frac { k ^ { * } } { N }$ </td></tr><tr><td>Corollary 4.5 (Ours, Upper &amp; Lower)</td><td> $+ \frac { 1 } { N ^ { 2 } } \sum _ { i , j \leq k ^ { * } } \frac { \lambda _ { j } ( \mathbf w _ { 0 } - \mathbf w ^ { * } ) _ { j } ^ { 2 } ( 1 - \gamma ( \lambda _ { i } \wedge \lambda _ { j } ) ) ^ { 2 ( s - 1 ) } } { s ^ { - 1 } \vee \gamma | \lambda _ { i } - \lambda _ { j } | }$ </td></tr><tr><td>Bound</td><td>Panel B: Slow-Mode Index Component  $( i > k ^ { * } )$ </td></tr><tr><td>Zou et al. (2021, Theorem 5.1) (Upper)</td><td> $\frac { \| \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } \| _ { \mathbf { I } _ { 0 : k ^ { \dagger } } } ^ { 2 } + N \gamma \| \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } \| _ { \mathbf { H } _ { k ^ { \dagger } : \infty } } ^ { 2 } } { N \gamma } \cdot \left( N \gamma ^ { 2 } \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } \right)$ </td></tr><tr><td>Zou et al. (2021, Theorem 5.2) (Lower)</td><td> $\| \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } \| _ { \mathbf { H } _ { k } ^ { \star } : \infty } ^ { 2 } \cdot \left( N \gamma ^ { 2 } \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } \right)$   $\frac { \| \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } \| _ { \mathbf { I } _ { 0 : k } ^ { 2 } } ^ { 2 } + \hat { N } \gamma \| \mathbf { w } _ { 0 } ^ { * - w } \mathbf { w } ^ { * } \| _ { \mathbf { H } _ { k } : \infty } ^ { 2 } } { N \gamma } \cdot \left( N \gamma ^ { 2 } \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } \right)$ </td></tr></table>

## 4.4 Sharp Separation Examples and Conditional Averaging-Window Bounds

We conclude Section 4 with two explicit separation constructions (Proposition 4.7) that demonstrate (i) an exponential-versus-polynomial gap between our lower bound in Theorem 4.2 and Zou et al. (2021, Theorem 5.2), and (ii) why fourth-moment assumptions alone cannot determine the asymptotic exponential burn-in rate without an additional structural condition (such as $\chi _ { s , N } \leq \chi _ { 0 } < 1 )$ or state information (such as $\mathbf { B } _ { s }$ in Proposition 4.8).

Proposition 4.7 (Sharp Separation Examples for Cross-Spectral Transfer and Burn-In Decay).

1. (Exponential-vs.-Polynomial Gap in the Lower Bound of Zou et al. (2021)): Fix $0 < \gamma \leq 1 / 6$ and integer $N \ge \operatorname* { m a x } \{ 5 0 0 , 1 / \gamma \}$ For $\mathbf { x } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { H } )$ in $\mathbb { R } ^ { 2 }$ with $\mathbf { H } ~ = ~ \mathrm { d i a g } ( 1 , ( 4 \gamma N ) ^ { - 1 } )$ $y = \langle \mathbf { w } ^ { * } , \mathbf { x } \rangle ~ ( \sigma _ { \mathrm { n o i s e } } ^ { 2 } = 0 ) , ~ \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } = ( 1 , 0 ) ^ { \top }$ , and $s = N$ , constant-stepsize tail-averaged SGD satisfies

$$
\mathbb { E } \big [ L ( \overline { { \mathbf { w } } } _ { N : 2 N } ) \big ] - L ( \mathbf { w } ^ { * } ) \geq \frac { 1 } { 1 0 2 4 \gamma N ^ { 2 } } ,\tag{4.16}
$$

whereas Zou et al. (2021, Theorem 5.2) yields the exponentially small lower bound $( 1 \textrm { -- }$ $\gamma ) ^ { 2 N } / ( 1 0 0 \gamma ^ { 2 } N ^ { 2 } )$

2. (Fourth Moments Shift Scalar Exponential Decay When $\chi _ { s , N }$ Is Uncontrolled): Fix $0 < \gamma < 1 / 3$ and $y = 0 , w ^ { * } = 0 , w _ { 0 } = 1 , H = 1$ . Let $\mathcal { D } _ { 1 }$ be the three-point distribution $\mathbb { P } ( x = 0 ) = 1 / 2$ $\mathbb { P } ( x = \pm \sqrt { 2 } ) = 1 / 4 \ ( w h e r e \ \mathbb { E } [ x ^ { 2 } ] = 1 , \ \mathbb { E } [ x ^ { 4 } ] = 2 )$ and let $\mathcal { D } _ { 2 }$ be $\mathcal { N } ( 0 , 1 )$ (where $\mathbb { E } [ x ^ { 2 } ] = 1$ $\mathbb { E } [ x ^ { 4 } ] = 3 )$ . Both satisfy Assumptions 3.1, 3.2, and $\it 3 . 4$ with $\alpha = 3$ and $\beta = 1$ , yet their exact tail-averaged risks $R _ { \kappa } { ( s , N ) } ~ ( \kappa \in \{ 2 , 3 \} )$ satisfy $R _ { \kappa } ( s , N ) \asymp _ { \gamma } r _ { \kappa } ^ { s } / N ^ { 2 }$ with $r _ { \kappa } = 1 - 2 \gamma + \kappa \gamma ^ { 2 }$ , so

$$
\frac { R _ { 3 } ( N , N ) } { R _ { 2 } ( N , N ) } \gtrsim _ { \gamma } ( \frac { 1 - 2 \gamma + 3 \gamma ^ { 2 } } { 1 - 2 \gamma + 2 \gamma ^ { 2 } } ) ^ { N } \longrightarrow \infty \qquad a s \ N  \infty .\tag{4.17}
$$

Consequently, no single expression depending only on $( \mathbf { H } , \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } , \gamma , s , N , \alpha , \beta , \sigma _ { \mathrm { n o i s e } } ^ { 2 } )$ can match both risks uniformly across N without an additional structural condition (such as $\chi _ { s , N } \leq \chi _ { 0 } < 1 )$ or conditioning on $\mathbf { B } _ { s }$

When $\chi _ { s , N } \geq 1$ (as in the large-s scalar regime of Proposition 4.7(2)), conditioning at the start of the averaging window $t = s$ still yields upper and lower bounds in terms of the burn-in bias covariance $\mathbf { B } _ { s } = \mathbb { E } [ \pmb { \eta } _ { s } ^ { \mathrm { b i a s } } \otimes \pmb { \eta } _ { s } ^ { \mathrm { b i a s } } ]$ from Section 3.3 that match up to constants depending only on $\alpha ,$ $\beta ,$ and $1 - \gamma \alpha \operatorname { t r } ( \mathbf { H } )$ for all $s \geq 0$ and $N \geq 2$ (recovering at the level of orders the result of restarting the full-averaging bounds of Zou et al. (2021) at $\mathbf { B } _ { s }$ while extending coverage to all $N \geq 2 )$

Proposition 4.8 (Conditional Upper and Lower Bounds in Terms of $\mathbf { B } _ { s } )$ . Under Assumptions $\it 3 . 1 -$ 3.3 with $\gamma \leq 1 / ( 2 \alpha \operatorname { t r } ( \mathbf { H } ) )$ , for all $s \geq 0$ and $N \geq 1$ , the excess risk satisfies

$$
\begin{array} { r l } { \mathbb { E } \big [ L ( \overline { { \mathbf { w } } } _ { s : s + N } ) \big ] - L ( \mathbf { w } ^ { * } ) \leq \frac { 1 } { \gamma ^ { 2 } N ^ { 2 } } \operatorname { t r } \big ( \mathbf { H } _ { 0 : k ^ { * } } ^ { - 1 } \mathbf { B } _ { s } \big ) + \operatorname { t r } \big ( \mathbf { H } _ { k ^ { * } : \infty } \mathbf { B } _ { s } \big ) } & { } \\ { + \frac { 2 \alpha } { 1 - \gamma \alpha \operatorname { t r } ( \mathbf { H } ) } \cdot \frac { \operatorname { t r } \big ( \mathbf { I } _ { 0 : k ^ { * } } \mathbf { B } _ { s } \big ) + N \gamma \operatorname { t r } \big ( \mathbf { H } _ { k ^ { * } : \infty } \mathbf { B } _ { s } \big ) } { N \gamma } \cdot \Bigg ( \frac { k ^ { * } } { N } + N \gamma ^ { 2 } \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } \Bigg ) } & { } \\ { + \frac { 2 \sigma ^ { 2 } } { 1 - \gamma \alpha \operatorname { t r } ( \mathbf { H } ) } \cdot \Bigg ( \frac { k ^ { * } } { N } + \gamma \displaystyle \sum _ { k ^ { * } < i \leq k ^ { * } } \lambda _ { i } + ( s + N ) \gamma ^ { 2 } \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } \Bigg ) , } &  \quad ( 4 . 1 8 \ \end{array}
$$

and under (3.4) and Assumptions 3.1, 3.2, and $\ 3 . 4$ with $\gamma \leq 1 / ( 2 \lambda _ { 1 } )$ and $N \geq 2$

$$
\begin{array} { r l } & { \displaystyle \mathbb { E } \big [ L \big ( \overline { { \mathbf { w } } } _ { s : s + N } \big ) \big ] - L \big ( \mathbf { w } ^ { * } \big ) \geq \frac { 1 } { 8 \gamma ^ { 2 } N ^ { 2 } } \mathrm { t r } \big ( \mathbf { H } _ { 0 : k ^ { * } } ^ { - 1 } \mathbf { B } _ { s } \big ) + \frac { 1 } { 8 } \mathrm { t r } \big ( \mathbf { H } _ { k ^ { * } : \infty } \mathbf { B } _ { s } \big ) } \\ & { \qquad + \frac { \beta } { 8 6 4 } \cdot \frac { \mathrm { t r } \big ( \mathbf { I } _ { 0 : k ^ { * } } \mathbf { B } _ { s } \big ) + N \gamma \mathrm { t r } \big ( \mathbf { H } _ { k ^ { * } : \infty } \mathbf { B } _ { s } \big ) } { N \gamma } \cdot \Bigg ( \frac { k ^ { * } } { N } + N \gamma ^ { 2 } \displaystyle \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } \Bigg ) } \\ & { \qquad + \frac { \sigma _ { \mathrm { n o i s e } } ^ { 2 } } { 2 1 6 } \cdot \Bigg ( \frac { k ^ { * } } { N } + \gamma \displaystyle \sum _ { k ^ { * } < i \leq k ^ { \dagger } } \lambda _ { i } + ( s + N ) \gamma ^ { 2 } \displaystyle \sum _ { i > k ^ { \dagger } } \lambda _ { i } ^ { 2 } \Bigg ) . } \end{array}\tag{4.19}
$$

## 5 Overview of Proof Techniques

In this section, we present the analytical framework underlying Theorems 4.1 and 4.2, Proposition 4.3, Corollary 4.4, and Propositions $4 . 7 \mathrm { - } 4 . 8 .$ Recall from Section 3.3 the bias and variance error processes $\{ \eta _ { t } ^ { \mathrm { b i a s } } \} _ { t \ge 0 }$ and $\{ \eta _ { t } ^ { \mathrm { v a r i a n c e } } \} _ { t \ge 0 } \mathrm { i n } ( 3 . 5 ) - ( 3 . 6 )$ and their single-step second-moment operators $\mathbf { B } _ { t } = \mathbb { E } [ \pmb { \eta } _ { t } ^ { \mathrm { b i a s } } \otimes \pmb { \eta } _ { t } ^ { \mathrm { b i a s } } ]$ and $\mathbf { C } _ { t } = \mathbb { E } [ \pmb { \eta } _ { t } ^ { \mathrm { v a r i a n c e } } \otimes \pmb { \eta } _ { t } ^ { \mathrm { v a r i a n c e } } ]$ . Following Jain et al. (2017b) and Zou et al. (2021), we define the linear transition super-operators $\tau$ and $\tilde { \tau }$ acting on symmetric operators A by

$$
\mathcal { T } : = \mathbf { H } \otimes \mathbf { I } + \mathbf { I } \otimes \mathbf { H } - \gamma \mathcal { M } , \qquad \widetilde { \mathcal { T } } : = \mathbf { H } \otimes \mathbf { I } + \mathbf { I } \otimes \mathbf { H } - \gamma \widetilde { \mathcal { M } } ,\tag{5.1}
$$

so that $( \mathcal { T } - \gamma \mathcal { T } ) \circ \mathbf { A } = \mathbb { E } [ ( \mathbf { I } - \gamma \mathbf { x } \mathbf { x } ^ { \top } ) \mathbf { A } ( \mathbf { I } - \gamma \mathbf { x } \mathbf { x } ^ { \top } ) ]$ is the single-step stochastic covariance transition and $( \mathcal { T } - \gamma \widetilde { \mathcal { T } } ) \circ \mathbf { A } = ( \mathbf { I } - \gamma \mathbf { H } ) \mathbf { A } ( \mathbf { I } - \gamma \mathbf { H } )$ is its deterministic counterpart. Their diference isolates the fourth-moment fluctuation super-operator:

$$
( \mathbb { Z } - \gamma \mathbb { T } ) - ( \mathbb { Z } - \gamma \widetilde { \mathbb { T } } ) = \gamma ^ { 2 } ( \mathbb { M } - \widetilde { \mathbb { M } } ) .\tag{5.2}
$$

Let $\mathcal { F } _ { t } : = \sigma ( \{ ( \mathbf { x } _ { u } , y _ { u } ) \} _ { u = 1 } ^ { t } )$ denote the natural filtration generated by the first t samples (with $\mathcal { F } _ { 0 }$ trivial). Because $\left( \mathbf { x } _ { t } , y _ { t } \right)$ is independent of $\mathcal { F } _ { t - 1 }$ and first-order optimality implies $\mathbb { E } [ \xi _ { t } { \mathbf { x } } _ { t } ] = \nabla L ( { \mathbf { w } } ^ { * } ) = { \mathbf { 0 } }$ taking expectations in (3.6) from $\eta _ { \mathrm { 0 } } ^ { \mathrm { v a r i a n c e } } = \mathbf { 0 }$ gives $\mathbb { E } [ \pmb { \eta } _ { t } ^ { \mathrm { v a r i a n c e } } ] = ( \mathbf { I } - \gamma \mathbf { H } ) \mathbb { E } [ \pmb { \eta } _ { t - 1 } ^ { \mathrm { v a r i a n c e } } ] = \mathbf { 0 }$ for all $t \geq 1$ . Together with sample independence, this cancels the cross terms in $\mathbf { C } _ { t } .$ , so taking outer products and expectations in (3.5)–(3.6) yields the exact single-step operator recurrences (Jain et al., 2017b; Zou et al., 2021):

$$
\mathbf { B } _ { t } = ( { \mathcal { T } } - \gamma T ) \circ \mathbf { B } _ { t - 1 } , \qquad \mathbf { C } _ { t } = ( { \mathcal { T } } - \gamma T ) \circ \mathbf { C } _ { t - 1 } + \gamma ^ { 2 } \Sigma , \qquad \forall t \geq 1 .\tag{5.3}
$$

Our analysis is organized into two main components. In Section 5.1, we develop a perturbationtheoretic trajectory decomposition (Lemma 5.1) that unrolls (5.2) across the full horizon and leads directly to a discrete Volterra feedback absorption inequality (Lemma 5.2, whose complete proof we present below). In Section 5.2, we analyze the resulting propagation kernel $k ( h )$ and explain how the pair-midpoint convolution identity and kernel doubling constant $D _ { T }$ control $\chi _ { s , N }$ across general and polynomially decaying spectra.

## 5.1 Perturbation Decomposition and Discrete Volterra Feedback Absorption

In Zou et al. (2021), the tail-averaged bias error is analyzed by splitting the trajectory at the end of the burn-in phase $t = s { \mathrm { : } }$ one first unrolls (5.3) over $0 \leq t \leq s - 1$ via the Duhamel expansion

$$
\mathbf { B } _ { s } = ( \mathbf { I } - \gamma \mathbf { H } ) ^ { s } \mathbf { B } _ { 0 } ( \mathbf { I } - \gamma \mathbf { H } ) ^ { s } + \gamma ^ { 2 } \sum _ { t = 0 } ^ { s - 1 } ( \mathbf { I } - \gamma \mathbf { H } ) ^ { s - 1 - t } \big ( ( \mathcal { M } - \widetilde { \mathcal { M } } ) \circ \mathbf { B } _ { t } \big ) ( \mathbf { I } - \gamma \mathbf { H } ) ^ { s - 1 - t } ,\tag{5.4}
$$

and then restarts a full-averaging bound from $\mathbf { B } _ { s }$ over $[ s , s + N - 1 ]$ . The subsequent relaxations in this two-stage argument cause both the upper- and lower-bound gaps in (1.1)–(1.2): Zou et al. (2021, Lemma D.4) upper-bounds (5.4) by replacing $( \mathbf { I } - \gamma \mathbf { H } ) ^ { 2 ( s - 1 - t ) } \mathbf { H }$ with H (and Zou et al. (2021, Lemma D.5) replaces the averaging-window trace drop $\mathrm { t r } ( \mathbf { B } _ { s , s + N } )$ by $\mathrm { t r } ( \mathbf { B } _ { 0 , s + N } )$ , where $\mathbf { B } _ { a , b } : = \mathbf { B } _ { a } - ( \mathbf { I } - \gamma \mathbf { H } ) ^ { b - a } \mathbf { B } _ { a } ( \mathbf { I } - \gamma \mathbf { H } ) ^ { b - a } )$ , which destroys the remaining burn-in contraction on the fast modes; conversely, Zou et al. (2021, Lemma D.9) lower-bounds (5.4) by dropping the non-negative sum over $t \in \{ 0 , \ldots , s - 1 \}$ altogether, which discards all burn-in fourth-moment energy scattered from the head initialization into slow modes.

Inspired by the operator-polynomial reformulations developed for momentum and exponentia moving average SGD in Li et al. (2024); Li and Gu (2025), we bypass the intermediate boundary $t = s$ and unroll the perturbation identity (5.2) directly across the entire trajectory $r \in \{ 0 , \ldots , s + N - 2 \}$ Physically, a stochastic gradient fluctuation injected at step $t = r + 1$ is propagated by the deterministic transition operator $( \mathbf { I } - \gamma \mathbf { H } ) ^ { m - t }$ to every subsequent iterate m $\in \{ s \vee t , \ldots , T \}$ in the averaging window, accumulating the filter polynomial $\mathbf { Q } _ { r , s , N } = q _ { t } ( \mathbf { H } )$ from (4.1). This yields the following exact covariance decomposition, proved in Appendix A.1.

Lemma 5.1 (Exact Operator-Polynomial Decomposition for Tail-Averaging). Under Assumptions 3.1–3.3, for any stepsize $\gamma > 0$ , burn-in length $s \geq 0$ , and averaging window $N \geq 1$ , define the deterministic operator polynomials

$$
\mathbf P _ { s , N } : = \frac { 1 } { N } \sum _ { t = s } ^ { s + N - 1 } \left( \mathbf I - \gamma \mathbf H \right) ^ { t } , \qquad \mathbf Q _ { r , s , N } : = \frac { 1 } { N } \sum _ { t = s \forall ( r + 1 ) } ^ { s + N - 1 } \left( \mathbf I - \gamma \mathbf H \right) ^ { t - 1 - r } , \quad \forall r \in \{ 0 , \ldots , s + N - 2 \} .\tag{5.5}
$$

Then the tail-averaged bias and variance covariance operators satisfy the exact identities

$$
\mathbb { E } \big [ \bar { \eta } _ { s : s + N } ^ { \mathrm { b i a s } } \otimes \bar { \eta } _ { s : s + N } ^ { \mathrm { b i a s } } \big ] = \mathbf { P } _ { s , N } \mathbf { B } _ { 0 } \mathbf { P } _ { s , N } + \gamma ^ { 2 } \sum _ { r = 0 } ^ { s + N - 2 } \mathbf { Q } _ { r , s , N } \big ( ( { \mathcal { M } } - \widetilde { \mathcal { M } } ) \circ \mathbf { B } _ { r } \big ) \mathbf { Q } _ { r , s , N } ,\tag{5.6}
$$

$$
\mathbb { E } \big [ \bar { \eta } _ { s : s + N } ^ { \mathrm { v a r i a n c e } } \otimes \bar { \eta } _ { s : s + N } ^ { \mathrm { v a r i a n c e } } \big ] = \gamma ^ { 2 } \sum _ { r = 0 } ^ { s + N - 2 } \mathbf { Q } _ { r , s , N } \big ( ( { \mathcal { M } } - \widetilde { { \mathcal { M } } } ) \circ \mathbf { C } _ { r } + \Sigma \big ) \mathbf { Q } _ { r , s , N } .\tag{5.7}
$$

Lemma 5.1 explains why the receiving weights $F _ { t } .$ , propagation kernel $k ( h )$ , and spectral feedback ratio $\chi _ { s , N }$ in (4.1)–(4.2) arise from first principles. Taking the trace inner product of (5.6) and (5.7) with $\scriptstyle { \frac { 1 } { 2 } } \mathbf { H }$ , the deterministic term $\begin{array} { r l } { { \frac { 1 } { 2 } } \langle \mathbf { H } , \mathbf { P } _ { s , N } \mathbf { B } _ { 0 } \mathbf { P } _ { s , N } \rangle } & { { } } \end{array}$ equals EfectiveBias in (3.8), while the remaining stochastic terms sum to EfectiveVar = (bias − EfectiveBias) + variance in (3.9). Under Assumptions 3.2–3.4, the fourth-moment fluctuation operators $( \mathcal { M } - \widetilde { \mathcal { M } } ) \circ \mathbf { B } _ { t - 1 }$ and $( \mathcal { M } - \widetilde { \mathcal { M } } ) \circ \mathbf { C } _ { t - 1 }$ are sandwiched by multiples of $\mathrm { t r } ( { \bf H } { \bf B } _ { t - 1 } )$ H and $\mathrm { t r } ( \mathbf { H } \mathbf { C } _ { t - 1 } ) \mathbf { H }$ . Consequently, each fluctuation excited along H at step $t = r + 1 \in \{ 1 , . . . , T \}$ is weighted at the output by

$$
\langle \mathbf H , \mathbf Q _ { t - 1 , s , N } \mathbf H \mathbf Q _ { t - 1 , s , N } \rangle = \mathrm { t r } \big ( \mathbf H ^ { 2 } q _ { t } ( \mathbf H ) ^ { 2 } \big ) = \sum _ { i = 1 } ^ { \infty } \lambda _ { i } ^ { 2 } q _ { t } ( \lambda _ { i } ) ^ { 2 } = F _ { t } .
$$

When $\mathbf { B } _ { t - 1 }$ is replaced by its unperturbed deterministic drift $( { \bf I } - \gamma { \bf H } ) ^ { t - 1 } { \bf B } _ { 0 } ( { \bf I } - \gamma { \bf H } ) ^ { t - 1 }$ , the resulting single-hop initialization-transfer functional $\mathcal { S } _ { s , N }$ and additive-noise functional $\gamma _ { s , N }$ take the form

$$
\mathcal { S } _ { s , N } : = \frac { \gamma ^ { 2 } } { 2 } \sum _ { t = 1 } ^ { T } \Vert ( \mathbf { I } - \gamma \mathbf { H } ) ^ { t - 1 } ( \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } ) \Vert _ { \mathbf { H } } ^ { 2 } F _ { t } , \qquad \mathcal { V } _ { s , N } : = \frac { \gamma ^ { 2 } } { 2 } \sum _ { t = 1 } ^ { T } F _ { t } .\tag{5.8}
$$

To bound the multi-hop contributions in EfectiveVar, we must control how the covariance traces $f _ { r } : = \mathrm { t r } ( \mathbf { H } \mathbf { B } _ { r } )$ and $g _ { r } : = \mathrm { t r } ( \mathbf { H } \mathbf { C } _ { r } )$ amplify over time. Unrolling $\mathbf { B } _ { t - 1 }$ as in (5.4) (and $\mathbf { C } _ { t - 1 }$ analogously) and taking the trace inner product with H reveals that a fluctuation of magnitude $f _ { r - 1 } \mathbf { H }$ injected at an earlier step $r < t$ is contracted by $( \mathbf { I } - \gamma \mathbf { H } ) ^ { t - r - 1 }$ before being tested against H at step t − 1, producing the two-time trace propagation kernel

$$
\mathrm { t r } \big ( \mathbf H ( \mathbf I - \gamma \mathbf H ) ^ { t - r - 1 } \mathbf H ( \mathbf I - \gamma \mathbf H ) ^ { t - r - 1 } \big ) = \sum _ { i = 1 } ^ { \infty } \lambda _ { i } ^ { 2 } ( 1 - \gamma \lambda _ { i } ) ^ { 2 ( t - r - 1 ) } = k ( t - r - 1 ) .
$$

Thus $\{ f _ { t } \}$ and $\{ g _ { t } \}$ obey a discrete-time Volterra convolution recursion driven by the kernel $k ( t - r - 1 ) - \mathrm { a }$ non-asymptotic counterpart of the Volterra integral equations that govern highdimensional SGD dynamics (Paquette et al., 2021, 2022; Li et al., 2025). Crucially, because the receiving weights $F _ { t }$ increase with t during the burn-in phase $1 \leq t \leq s .$ , bounding $\scriptstyle \sum _ { t = 1 } ^ { T } F _ { t } f _ { t - 1 }$ via an unweighted trace sum $\scriptstyle \sum _ { t = 1 } ^ { T } f _ { t - 1 }$ would require replacing $F _ { t }$ by max $F _ { t }$ and losing burnin contraction. Inspired by the Volterra convolution self-bounding strategy of Li et al. (2025, Appendix C.3.1), we instead bound the F -weighted trajectory sum directly as a self-bounding functional.

Lemma 5.2 (Weighted Trajectory Feedback Absorption). Suppose Assumption 3.1 holds and $0 < \gamma < 1 / \lambda _ { 1 }$ , and recall EfectiveBias, EfectiveVar, $\chi _ { s , N } , S _ { s , N }$ , and $\gamma _ { s , N }$ from (3.8), (3.9), (4.2), and (5.8).

1. Under Assumptions 3.2 and 3.3, $i f \chi _ { s , N } < 1$ , then

$$
\mathrm { E f f e c t i v e V a r } \leq \frac { \alpha } { 1 - \chi _ { s , N } } { S _ { s , N } } + \frac { \sigma ^ { 2 } } { 1 - \chi _ { s , N } } \mathcal V _ { s , N } .\tag{5.9}
$$

2. Under the well-specified model (3.4) and Assumptions 3.2 and $\it 3 . 4$ (without requiring $\chi _ { s , N } < 1 )$

$$
\mathrm { E f f e c t i v e V a r } \geq \beta S _ { s , N } + \sigma _ { \mathrm { n o i s e } } ^ { 2 } \mathcal { V } _ { s , N } .\tag{5.10}
$$

Proof of Lemma 5.2. For each integer $r \geq 0$ , let $f _ { r } : = \mathrm { t r } ( \mathbf { H } \mathbf { B } _ { r } ) , g _ { r } : = \mathrm { t r } ( \mathbf { H } \mathbf { C } _ { r } )$ , and $b _ { r } : = \mathrm { t r } ( \mathbf { H } ( \mathbf { I }$ $\gamma \mathbf { H } ) ^ { 2 r } \mathbf { B } _ { 0 } ) \ = \ \| ( \mathbf { I } - \gamma \mathbf { H } ) ^ { r } ( \mathbf { w } _ { 0 } \ - \ \mathbf { w } ^ { * } ) \| _ { \mathbf { H } } ^ { 2 }$ . Taking the trace inner product of (5.6) and (5.7) in Lemma 5.1 with <sup>1</sup>H and using $\begin{array} { r } { \frac { 1 } { 2 } \langle \mathbf { H } , \mathbf { \bar { P } } _ { s , N } \mathbf { B } _ { 0 } \mathbf { P } _ { s , N } \rangle = \mathrm { E f f e c t i v e B i a s } } \end{array}$ together with the index shift $t = r + 1 \in \{ 1 , . . . , T \}$ , we obtain the exact representation

$$
\mathrm { E f f e c t i v e V a r } = \frac { \gamma ^ { 2 } } { 2 } \sum _ { t = 1 } ^ { T } \left. \mathbf { H } , \mathbf { Q } _ { t - 1 , s , N } \left( \left( \mathcal { M } - \widetilde { \mathcal { M } } \right) \circ \left( \mathbf { B } _ { t - 1 } + \mathbf { C } _ { t - 1 } \right) + \boldsymbol { \Sigma } \right) \mathbf { Q } _ { t - 1 , s , N } \right. .\tag{5.11}
$$

We first prove the upper bound (5.9). Applying Assumptions 3.2 and 3.3 to the inner operator in (5.11) gives $( \mathcal { M } - \widetilde { \mathcal { M } } ) \circ \mathbf { B } _ { t - 1 } \preceq \alpha f _ { t - 1 } \mathbf { H }$ and $\begin{array} { r } { ( \mathcal { M } - \widetilde { \mathcal { M } } ) \circ \mathbf { C } _ { t - 1 } + \boldsymbol { \Sigma } \preceq ( \alpha g _ { t - 1 } + \sigma ^ { 2 } ) \mathbf { I } } \end{array}$ . Substituting these Loewner upper bounds into (5.11) yields

$$
\begin{array} { r l } & { \mathrm { E f f e c t i v e V a r } \le \displaystyle \frac { \gamma ^ { 2 } } { 2 } \sum _ { t = 1 } ^ { T } \bigl [ \alpha ( f _ { t - 1 } + g _ { t - 1 } ) + \sigma ^ { 2 } \bigr ] \langle \mathbf { H } , \mathbf { Q } _ { t - 1 , s , N } \mathbf { H } \mathbf { Q } _ { t - 1 , s , N } \rangle } \\ & { \qquad \quad = \displaystyle \frac { \gamma ^ { 2 } } { 2 } \sum _ { t = 1 } ^ { T } F _ { t } \bigl [ \alpha \bigl ( f _ { t - 1 } + g _ { t - 1 } \bigr ) + \sigma ^ { 2 } \bigr ] = : U , } \end{array}\tag{5.12}
$$

where the equality uses $\langle \mathbf { H } , \mathbf { Q } _ { t - 1 , s , N } \mathbf { H } \mathbf { Q } _ { t - 1 , s , N } \rangle = F _ { t }$ . Next, unrolling the single-step recurrences (5.3) via the perturbation identity (5.2) up to step t − 1 gives

$$
\begin{array} { l } { { \displaystyle { \bf B } _ { t - 1 } = \left( { \bf I } - \gamma { \bf H } \right) ^ { t - 1 } { \bf B } _ { 0 } ( { \bf I } - \gamma { \bf H } ) ^ { t - 1 } + \gamma ^ { 2 } \sum _ { r = 1 } ^ { t - 1 } ( { \bf I } - \gamma { \bf H } ) ^ { t - 1 - r } \left( \left( { \cal M } - \widetilde { \mathcal { M } } \right) \circ { \bf B } _ { r - 1 } \right) \left( { \bf I } - \gamma { \bf H } \right) ^ { t - 1 - r } } , } \\ { { \displaystyle { \bf C } _ { t - 1 } = \gamma ^ { 2 } \sum _ { r = 1 } ^ { t - 1 } ( { \bf I } - \gamma { \bf H } ) ^ { t - 1 - r } \left( \left( { \cal M } - \widetilde { \mathcal { M } } \right) \circ { \bf C } _ { r - 1 } + { \bf E } \right) \left( { \bf I } - \gamma { \bf H } \right) ^ { t - 1 - r } } . } \end{array}
$$

Taking the trace inner product of both expansions with H and applying Assumptions 3.2 and 3.3 together with $\mathrm { t r } ( \mathbf H ( \mathbf I - \gamma \mathbf H ) ^ { t - 1 - r } \mathbf H ( \mathbf I - \gamma \mathbf H ) ^ { t - 1 - r } ) = k ( t - r - 1 )$ , we obtain for every $t \in \{ 1 , \ldots , T \}$ the coupled Volterra trace inequalities

$$
f _ { t - 1 } \leq b _ { t - 1 } + \alpha \gamma ^ { 2 } \sum _ { r = 1 } ^ { t - 1 } k ( t - r - 1 ) f _ { r - 1 } , \qquad g _ { t - 1 } \leq \gamma ^ { 2 } \sum _ { r = 1 } ^ { t - 1 } k ( t - r - 1 ) \big ( \alpha g _ { r - 1 } + \sigma ^ { 2 } \big ) .\tag{5.13}
$$

Substituting (5.13) into the definition of U in (5.12) yields

$$
\begin{array} { l } { { U \leq \displaystyle \frac { \gamma ^ { 2 } } { 2 } \sum _ { t = 1 } ^ { T } F _ { t } \big ( \alpha b _ { t - 1 } + \sigma ^ { 2 } \big ) + \frac { \gamma ^ { 2 } } { 2 } \sum _ { t = 1 } ^ { T } F _ { t } \alpha \gamma ^ { 2 } \sum _ { r = 1 } ^ { t - 1 } k ( t - r - 1 ) \big [ \alpha \big ( f _ { r - 1 } + g _ { r - 1 } \big ) + \sigma ^ { 2 } \big ] } } \\ { { \mathrm { } = \alpha S _ { s , N } + \sigma ^ { 2 } \mathcal { V } _ { s , N } + \displaystyle \frac { \gamma ^ { 2 } } { 2 } \sum _ { r = 1 } ^ { T - 1 } \big [ \alpha \big ( f _ { r - 1 } + g _ { r - 1 } \big ) + \sigma ^ { 2 } \big ] \left( \alpha \gamma ^ { 2 } \sum _ { t = r + 1 } ^ { T } k ( t - r - 1 ) F _ { t } \right) } } \\ { { \mathrm { } \leq \alpha S _ { s , N } + \sigma ^ { 2 } \mathcal { V } _ { s , N } + \chi _ { s , N } \cdot \displaystyle \frac { \gamma ^ { 2 } } { 2 } \sum _ { r = 1 } ^ { T - 1 } F _ { r } \big [ \alpha \big ( f _ { r - 1 } + g _ { r - 1 } \big ) + \sigma ^ { 2 } \big ] } } \end{array}
$$

$$
\leq \alpha S _ { s , N } + \sigma ^ { 2 } \mathcal { V } _ { s , N } + \chi _ { s , N } U ,\tag{5.14}
$$

where the first equality uses the definitions of $\mathcal { S } _ { s , N }$ and $\gamma _ { s , N }$ in (5.8) and interchanges the finite sums over $1 \leq r < t \leq T$ , the second inequality uses $\begin{array} { r } { \alpha \gamma ^ { 2 } \sum _ { t = r + 1 } ^ { T } k ( t - r - 1 ) F _ { t } \leq \chi _ { s , N } F _ { r } } \end{array}$ from the definition of $\chi _ { s , N }$ in (4.2), and the last inequality extends the non-negative sum over r to $T$ . Since $\chi _ { s , N } < 1$ , rearranging (5.14) and combining with (5.12) establishes (5.9). Finally, to prove the lower bound (5.10), Lemma $\mathrm { { C . 5 ( 3 ) } }$ implies B<sub>t</sub> $\mathbf { \Gamma } _ { - 1 } \succeq ( \mathbf { I } - \gamma \mathbf { H } ) ^ { t - 1 } \mathbf { B } _ { 0 } ( \mathbf { I } - \gamma \mathbf { H } ) ^ { t - 1 }$ (so and Applving, Assumption 3.4 and under (3 4)

$$
f _ { t - 1 } \geq b _ { t - 1 } )
$$

$$
( \mathcal { M } - \widetilde { \mathcal { M } } ) \circ \mathbf { C } _ { t - 1 } \succeq \mathbf { 0 }
$$

$$
\pmb { \Sigma } = \sigma _ { \mathrm { n o i s e } } ^ { 2 } \mathbf { H }
$$

$$
\begin{array} { r l r } & { } & { \mathrm { E f f e c t i v e V a r } \ge \displaystyle \frac { \gamma ^ { 2 } } { 2 } \sum _ { t = 1 } ^ { T } \left. { \bf H } , { \bf Q } _ { t - 1 , s , N } \left( \beta f _ { t - 1 } { \bf H } + \sigma _ { \mathrm { n o i s e } } ^ { 2 } { \bf H } \right) { \bf Q } _ { t - 1 , s , N } \right. } \\ & { } & { \ge \displaystyle \frac { \gamma ^ { 2 } } { 2 } \sum _ { t = 1 } ^ { T } F _ { t } \big ( \beta b _ { t - 1 } + \sigma _ { \mathrm { n o i s e } } ^ { 2 } \big ) = \beta S _ { s , N } + \sigma _ { \mathrm { n o i s e } } ^ { 2 } \mathcal { V } _ { s , N } , } \end{array}
$$

which completes the proof of Lemma 5.2.

## 5.2 Propagation Kernel Geometry, Pair-Midpoint Convolution, and Spectral Regimes

With Lemma 5.2 in hand, the remaining analytical challenge is to bound the spectral feedback ratio $\chi _ { s , N }$ in (4.2) without losing the denominator weight $F _ { r }$ . During the burn-in phase $1 \leq r < t \leq s .$ the receiving sensitivity $F _ { t }$ increases with t because later perturbations undergo fewer burn-in contractions before entering the averaging window $[ s , T ]$ . Consequently, bounding $F _ { t } / F _ { r }$ pointwise would incur an exponential blow-up in s.

Our key observation (Proposition 4.3) is that $F _ { t }$ and the step-to-step kernel $k ( h )$ are governed by the same functional $k ( \cdot )$ . Expanding the square of the filter polynomial $\begin{array} { r } { q _ { t } ( \lambda _ { i } ) = \frac { 1 } { N } \sum _ { a = s \vee t } ^ { T } ( 1 - } \end{array}$ $\gamma \lambda _ { i } ) ^ { a - t }$ in $\left( 4 . 1 \right)$ and summing against $\lambda _ { i } ^ { 2 }$ expresses $F _ { t }$ as the average of $k ( \cdot )$ evaluated at the pair midpoint $\textstyle { \frac { a + b } { 2 } } - t \colon$

$$
F _ { t } = \sum _ { i = 1 } ^ { \infty } \lambda _ { i } ^ { 2 } \left( \frac { 1 } { N } \sum _ { a = s \vee t } ^ { T } { \left( 1 - \gamma \lambda _ { i } \right) } ^ { a - t } \right) \left( \frac { 1 } { N } \sum _ { b = s \vee t } ^ { T } { \left( 1 - \gamma \lambda _ { i } \right) } ^ { b - t } \right) = \frac { 1 } { N ^ { 2 } } \sum _ { a , b = s \vee t } ^ { T } k \Big ( \frac { a + b } { 2 } - t \Big ) .
$$

Substituting this pair-midpoint representation into $\textstyle \gamma ^ { 2 } \sum _ { t = r + 1 } ^ { T } k ( t - r - 1 ) F _ { t }$ and setting $j : =$ $t - r - 1$ and $\textstyle h : = { \frac { a + b } { 2 } } - r \in [ 1 , T ]$ reduces the inner sum over t to the discrete self-convolution $\begin{array} { r } { \sum _ { j = 0 } ^ { \operatorname* { m i n } ( a , b ) - r - 1 } k ( j ) k ( h - 1 - j ) } \end{array}$ at total lag $h - 1$ . This structure yields two complementary mechanisms for controlling $\chi _ { s , N }$ (proved in Appendix B.2):

• General Trace-Class Spectra via Mode-Wise Positive Correlation (Proposition $\it 4 . 3 ( 1 ) )$ : Under the main-theorem restriction $\gamma \lambda _ { 1 } \leq 1 / 2 .$ write $a _ { i } : = 1 - \gamma \lambda _ { i } \in [ 1 / 2 , 1 )$ . Because slower modes contract more slowly, the sequences $\{ a _ { i } ^ { 2 j } \} _ { i \geq 1 }$ and $\{ a _ { i } ^ { 2 ( h - 1 - j ) } \} _ { i \geq 1 }$ are both non-decreasing in i. By Chebyshev’s sum inequality under the positive weights $\{ \lambda _ { i } ^ { \overline { { 2 } } } \} _ { i \geq 1 }$ (Lemma C.1),

$$
k ( j ) k ( h - 1 - j ) \le \mathrm { t r } ( \mathbf { H } ^ { 2 } ) k ( h - 1 ) \le \frac { \mathrm { t r } ( \mathbf { H } ^ { 2 } ) } { ( 1 - \gamma \lambda _ { 1 } ) ^ { 2 } } k \Big ( \frac { a + b } { 2 } - r \Big ) \lesssim \mathrm { t r } ( \mathbf { H } ^ { 2 } ) k \Big ( \frac { a + b } { 2 } - r \Big ) .
$$

Summing over the at most $T - 1$ indices $j$ and averaging over $a , b \ge s \vee ( r + 1 ) \ge s \vee r$ reconstructs $\begin{array} { r } { F _ { r } = \frac { 1 } { N ^ { 2 } } \sum _ { a , b = s \vee r } ^ { T } k ( \frac { a + b } { 2 } - r ) } \end{array}$ , canceling the denominator $F _ { r }$ in (4.2) and yielding $\chi _ { s , N } \lesssim \alpha \gamma ^ { 2 } T \mathrm { t r } ( \mathbf { H } ^ { 2 } )$ .

• Polynomial Spectra via Midpoint Convolution Splitting and Kernel Doubling (Proposition $\it 4 . 3 ( 2 )$ and Corollary $4 { \cdot } 4 )$ : In every term of the self-convolution $\begin{array} { r } { \sum _ { j = 0 } ^ { h - 1 } k ( j ) k ( h - 1 - j ) } \end{array}$ , at least one of the two lags $j$ or $h - 1 - j$ is at least $( h - 1 ) / 2$ . Since $k ( \cdot )$ is non-increasing, splitting the convolution at the midpoint $( h - 1 ) / 2$ (a discrete counterpart of regular-variation splitting for power-law Volterra kernels; cf. Li et al. (2025)) factors out $k ( ( h - 1 ) / 2 ) \leq D _ { T } k ( h )$ while the remaining factor sums to $\textstyle \gamma ^ { 2 } \sum _ { n = 0 } ^ { \infty } k ( n ) \asymp \gamma \operatorname { t r } ( \mathbf { H } )$ :

$$
\gamma ^ { \operatorname* { m i n } ( a , b ) - r - 1 } k ( j ) k ( h - 1 - j ) \le 2 k \Big ( \frac { h - 1 } { 2 } \Big ) \gamma ^ { 2 } \sum _ { n = 0 } ^ { \infty } k ( n ) \lesssim D _ { T } \gamma \operatorname { t r } ( \mathbf { H } ) k \Big ( \frac { a + b } { 2 } - r \Big ) .
$$

Averaging over $( a , b )$ again recovers $F _ { r }$ and gives the horizon-free bound $\chi _ { s , N } \lesssim \alpha D _ { T } \gamma \mathrm { t r } ( \mathbf { H } )$ Under two-sided polynomial eigenvalue decay $\lambda _ { i } \asymp \Lambda i ^ { - a } \left( a > 1 \right)$ , integral comparison over the active spectral scale $i \asymp ( \gamma \Lambda h ) ^ { 1 / a }$ shows that $k ( h ) \asymp \Lambda ^ { 2 } ( 1 + \gamma \Lambda h ) ^ { - ( 2 - 1 / a ) }$ decays as a power law, so its doubling ratio $D _ { T }$ is uniformly bounded by a constant $D _ { * } ( a , \theta ) < \infty$ for all $T \geq 1$ (whereas in $d = 1 , k ( h ) = \lambda _ { 1 } ^ { 2 } ( 1 - \gamma \lambda _ { 1 } ) ^ { 2 h }$ decays exponentially so $D _ { T }$ grows exponentially in $T _ { \ast }$ giving rise to the scalar separation in Proposition $4 . 7 ( 2 ) )$

Finally, in Appendix A.2 (Lemma A.1), we evaluate the single-hop functionals EfectiveBias, $\gamma _ { s , N }$ and $\mathcal { S } _ { s , N }$ within universal constants across the spectral cutofs $k ^ { * }$ and $k ^ { \dagger }$ , which combined with Lemma 5.2 yields Theorems 4.1 and 4.2.

## 6 Conclusion

We have established initialization-explicit matching upper and lower excess risk bounds for constantstepsize stochastic gradient descent with tail-averaging in high-dimensional linear regression under the distributional assumptions of Zou et al. (2021). By controlling cumulative fourth-moment trajectory feedback through the spectral feedback ratio $\chi _ { s , N } \leq \chi _ { 0 } < 1$ and bounding $\chi _ { s , N }$ via the pair-midpoint convolution identity, we proved that matching bounds hold for arbitrary covariance spectra under a horizon-dependent constant stepsize chosen before the run, and hold uniformly across all horizons $s \geq 0$ and $N \geq 2$ at a single horizon-independent stepsize for polynomially decaying spectra $\lambda _ { i } \asymp i ^ { - a } ( a > 1 )$ . Complementing these positive results, our two-dimensional Gaussian separation construction demonstrates an exponential-versus-polynomial improvement over the lower bound of Zou et al. (2021, Theorem 5.2) due to burn-in head-to-tail transfer, while our scalar separation construction shows why additional structural conditions (such as $\chi _ { s , N } \leq \chi _ { 0 } < 1 )$ or state information $\left( \mathbf { B } _ { s } \right)$ are required over the unrestricted fourth-moment class when $D _ { T }$ grows exponentially.

## Acknowledgment

The proof ideas are developed through an interactive brainstorm among $\mathrm { G P T _ { - 6 } }$ Astra, Gemini 4 Argon, and the authors. Gemini 4 Argon are employed to further streamline the canonical pillars underlying the proof roadmap. The authors simplified the arguments, re-organized the proofs, verified the details, and prepared the draft with further assistance from Gemini 4 Argon and GPT-6 Astra. The authors take full responsibility for its contents.

## A Trajectory Decomposition and Spectral Evaluation

In this appendix, we establish the exact operator-polynomial decomposition (Lemma 5.1 in Section A.1) and the two-sided spectral evaluations of EfectiveBias, $\gamma _ { s , N }$ , and $\mathcal { S } _ { s , N }$ (Lemma A.1 in Section A.2). Elementary algebraic and convolution lemmas as well as foundational operator lemmas from prior work are collected in Appendix C.

Additional Notations. Throughout the appendices, for any real number $z \in \mathbb { R }$ , we write $\lfloor z \rfloor$ and $\lceil z \rceil$ for the floor and ceiling of z, respectively. Recall from Section 5 the natural filtration $\mathcal { F } _ { t } = \sigma ( \{ ( \mathbf { x } _ { u } , y _ { u } ) \} _ { u = 1 } ^ { t } )$ (with $\mathcal { F } _ { 0 }$ trivial) and the eigenvalue cutofs $k ^ { * }$ = max $\{ k : \lambda _ { k } \ge ( \gamma N ) ^ { - 1 } \}$ and $k ^ { \dagger } = \operatorname* { m a x } \{ k : \lambda _ { k } \geq ( \gamma ( s + N ) ) ^ { - 1 } \}$ from (4.5). To streamline the spectral evaluations across Appendices $\mathrm { A }$ and B, we define the burn-in drifted error vector, initial eigen-coordinates, and window efective-variance multiplier by

$$
\mathbf w _ { s } ^ { \mathrm { d r i f t } } : = \left( \mathbf I - \gamma \mathbf H \right) ^ { s } \left( \mathbf w _ { 0 } - \mathbf w ^ { * } \right) , \qquad \omega _ { j } : = \left( \mathbf w _ { 0 } - \mathbf w ^ { * } \right) _ { j } = \langle \mathbf v _ { j } , \mathbf w _ { 0 } - \mathbf w ^ { * } \rangle , \qquad \Xi _ { N } : = \frac { k ^ { * } } N + N \gamma ^ { 2 } \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } ,\tag{A.1}
$$

and for each $r \in \{ 0 , \ldots , s + N - 2 \}$ , we define the step-(r + 1) scaled receiving weight

$$
\Psi _ { r , s , N } : = \gamma ^ { 2 } F _ { r + 1 } = \frac { 1 } { N ^ { 2 } } \sum _ { i = 1 } ^ { \infty } ( 1 - \gamma \lambda _ { i } ) ^ { 2 ( ( s - 1 - r ) \vee 0 ) } \big ( 1 - ( 1 - \gamma \lambda _ { i } ) ^ { ( s + N - 1 - r ) \wedge N } \big ) ^ { 2 } .\tag{A.2}
$$

## A.1 Proof of Lemma 5.1

Proof of Lemma 5.1. We first establish the bias covariance identity (5.6). For each $r \geq 1$ , define the zero-mean random operator fluctuation

$$
\begin{array} { r } { \pmb { \Delta } _ { r } : = ( \mathbf { I } - \gamma \mathbf { x } _ { r } \mathbf { x } _ { r } ^ { \top } ) - ( \mathbf { I } - \gamma \mathbf { H } ) = \gamma ( \mathbf { H } - \mathbf { x } _ { r } \mathbf { x } _ { r } ^ { \top } ) . } \end{array}
$$

Then the single-step bias recursion (3.5) can be written as $\pmb { \eta } _ { t } ^ { \mathrm { b i a s } } = ( \mathbf { I } - \gamma \mathbf { H } ) \pmb { \eta } _ { t - 1 } ^ { \mathrm { b i a s } } + \Delta _ { t } \pmb { \eta } _ { t - 1 } ^ { \mathrm { b i a s } }$ . Unrolling this linear recursion from step 0 to step $t \geq 0$ gives

$$
\eta _ { t } ^ { \mathrm { b i a s } } = ( \mathbf { I } - \gamma \mathbf { H } ) ^ { t } \eta _ { 0 } ^ { \mathrm { b i a s } } + \sum _ { r = 0 } ^ { t - 1 } ( \mathbf { I } - \gamma \mathbf { H } ) ^ { t - 1 - r } \Delta _ { r + 1 } \eta _ { r } ^ { \mathrm { b i a s } } .\tag{A.3}
$$

Averaging (A.3) over $t \in \{ s , s + 1 , \ldots , s + N - 1 \}$ and exchanging the order of summation between t and $r \in \{ 0 , \ldots , s + N - 2 \}$ , we obtain

$$
\begin{array} { l } { \displaystyle \bar { \eta } _ { s : s + N } ^ { \mathrm { b i a s } } = \frac { 1 } { N } \sum _ { t = s } ^ { s + N - 1 } \left( \mathbf { I } - \gamma \mathbf { H } \right) ^ { t } \eta _ { 0 } ^ { \mathrm { b i a s } } + \frac { 1 } { N } \sum _ { t = s } ^ { s + N - 1 } \sum _ { r = 0 } ^ { t - 1 } ( \mathbf { I } - \gamma \mathbf { H } ) ^ { t - 1 - r } \Delta _ { r + 1 } \eta _ { r } ^ { \mathrm { b i a s } } } \\ { = \mathbf { P } _ { s , N } \eta _ { 0 } ^ { \mathrm { b i a s } } + \displaystyle \sum _ { r = 0 } ^ { s + N - 2 } \left( \frac { 1 } { N } \sum _ { t = s \vee ( r + 1 ) } ^ { s + N - 1 } \left( \mathbf { I } - \gamma \mathbf { H } \right) ^ { t - 1 - r } \right) \Delta _ { r + 1 } \eta _ { r } ^ { \mathrm { b i a s } } } \\ { = \mathbf { P } _ { s , N } \eta _ { 0 } ^ { \mathrm { b i a s } } + \displaystyle \sum _ { r = 0 } ^ { s + N - 2 } \mathbf { Q } _ { r , s , N } \Delta _ { r + 1 } \eta _ { r } ^ { \mathrm { b i a s } } , } \end{array}\tag{A.4}
$$

where the third equality uses the definitions of $\mathbf { P } _ { s , N }$ and $\mathbf { Q } _ { r , s , N }$ in (5.5). Since $\eta _ { r } ^ { \mathrm { b i a s } }$ is $\mathcal { F } _ { r }$ -measurable and $\left( \mathbf { x } _ { r + 1 } , y _ { r + 1 } \right)$ is independent of $\mathcal { F } _ { r }$ with $\mathbb { E } [ \Delta _ { r + 1 } \mid \mathcal { F } _ { r } ] = \mathbf { 0 }$ , the sequence $\{ \mathbf { Q } _ { r , s , N } \Delta _ { r + 1 } \pmb { \eta } _ { r } ^ { \mathrm { b i a s } } \} _ { r = 0 } ^ { s + N - 2 }$

is a martingale diference sequence with respect to $\{ \mathcal { F } _ { r + 1 } \} _ { r \ge 0 }$ . Consequently, for any $0 \leq r < r ^ { \prime } \leq$ $s + N - 2$ ，

$$
\mathbb { E } \left[ \left( \mathbf { Q } _ { r ^ { \prime } , s , N } \Delta _ { r ^ { \prime } + 1 } \eta _ { r ^ { \prime } } ^ { \mathrm { b i a s } } \right) \otimes \left( \mathbf { Q } _ { r , s , N } \Delta _ { r + 1 } \eta _ { r } ^ { \mathrm { b i a s } } \right) \right] = \mathbf { 0 } , \qquad \mathbb { E } \left[ \left( \mathbf { Q } _ { r , s , N } \Delta _ { r + 1 } \eta _ { r } ^ { \mathrm { b i a s } } \right) \otimes \left( \mathbf { P } _ { s , N } \eta _ { 0 } ^ { \mathrm { b i a s } } \right) \right] = \mathbf { 0 } .
$$

Furthermore, for each $r \in \{ 0 , \ldots , s + N - 2 \}$ , conditioning on $\mathcal { F } _ { r }$ and using (3.3) gives

$$
\begin{array} { r l } & { \mathbb { E } \big [ ( \Delta _ { r + 1 } \eta _ { r } ^ { \mathrm { b i a s } } ) \otimes ( \Delta _ { r + 1 } \eta _ { r } ^ { \mathrm { b i a s } } ) \big ] = \gamma ^ { 2 } \mathbb { E } \Big [ \mathbb { E } \big [ ( \mathbf { x } _ { r + 1 } \mathbf { x } _ { r + 1 } ^ { \top } - \mathbf { H } ) ( \eta _ { r } ^ { \mathrm { b i a s } } \otimes \eta _ { r } ^ { \mathrm { b i a s } } ) ( \mathbf { x } _ { r + 1 } \mathbf { x } _ { r + 1 } ^ { \top } - \mathbf { H } ) \mid \mathcal { F } _ { r } \big ] \Big ] } \\ & { \qquad = \gamma ^ { 2 } ( M - \widetilde { M } ) \circ \mathbb { E } \big [ \eta _ { r } ^ { \mathrm { b i a s } } \otimes \eta _ { r } ^ { \mathrm { b i a s } } \big ] = \gamma ^ { 2 } ( M - \widetilde { M } ) \circ \mathbf { B } _ { r } . \qquad ( \widetilde { M } \mathbin { a d d } \mathbf { x } _ { r } ^ { \top } \big ) } \end{array}\tag{.5}
$$

Taking the outer product of $\mathrm { ( A . 4 ) }$ with itself in expectation and substituting (A.5) yields (5.6).

To prove the variance covariance identity (5.7) under Assumption 3.3 (without assuming independence of $\xi _ { t }$ and ${ \bf x } _ { t } )$ , we work with $\mathbf { C } _ { t } = \mathbb { E } [ \pmb { \eta } _ { t } ^ { \mathrm { v a r i a n c e } } \otimes \pmb { \eta } _ { t } ^ { \mathrm { v a r i a n c e } } ] .$ . Since η<sup>variance</sup> is $\mathcal { F } _ { t } .$ -measurable and the tower property with $\operatorname { \mathbb { E } } [ \xi _ { u } \mathbf { x } _ { u } \mid \mathcal { F } _ { u - 1 } ] = \nabla L ( \mathbf { w } ^ { * } ) = \mathbf { 0 }$ implies $\mathbb { E } [ \xi _ { u } { \bf x } _ { u } \mid \mathcal { F } _ { t } ] = { \bf 0 }$ for all $u > t$ taking conditional expectations given $\mathcal { F } _ { t }$ in (3.6) yields $\mathbb { E } [ \pmb { \eta } _ { t ^ { \prime } } ^ { \mathrm { v a r i a n c e } } \ | \ \mathcal { F } _ { t } ] = ( \mathbf { I } - \gamma \mathbf { H } ) ^ { t ^ { \prime } - t } \pmb { \eta } _ { t } ^ { \mathrm { v a r i a n c e } }$ for all $t ^ { \prime } \geq t \geq 0$ , and hence

$$
\mathbb { E } \left[ \eta _ { \ell ^ { \prime } } ^ { \mathrm { v a r i a n c e } } \otimes \eta _ { \ell } ^ { \mathrm { v a r i a n c e } } \right] = \mathbb { E } \left[ \mathbb { E } \left[ \eta _ { \ell ^ { \prime } } ^ { \mathrm { v a r i a n c e } } \mid \mathcal { F } _ { t } \right] \otimes \eta _ { t } ^ { \mathrm { v a r i a n c e } } \right] = ( \mathbf { I } - \gamma \mathbf { H } ) ^ { t ^ { \prime } - t } \mathbf { C } _ { t } , \qquad \forall t ^ { \prime } \geq t \geq 0 .\tag{A.6}
$$

On the other hand, iterating $\mathbf { C } _ { t } = ( \mathcal { T } - \gamma \widetilde { \mathcal { T } } ) \circ \mathbf { C } _ { t - 1 } + \gamma ^ { 2 } \big ( ( \mathcal { M } - \widetilde { \mathcal { M } } ) \circ \mathbf { C } _ { t - 1 } + \boldsymbol { \Sigma } \big )$ from $\mathbf { C } _ { 0 } = \mathbf { 0 }$ gives

$$
\mathbf { C } _ { t } = \gamma ^ { 2 } \sum _ { r = 0 } ^ { t - 1 } ( \mathbf { I } - \gamma \mathbf { H } ) ^ { t - 1 - r } \mathbf { D } _ { r } ( \mathbf { I } - \gamma \mathbf { H } ) ^ { t - 1 - r } , \qquad \mathrm { w h e r e } \quad \mathbf { D } _ { r } : = ( \mathcal { M } - \widetilde { \mathcal { M } } ) \circ \mathbf { C } _ { r } + \boldsymbol { \Sigma } .\tag{A.7}
$$

Substituting (A.7) into $\left( \mathrm { A . 6 } \right)$ (and taking transposes when $t ^ { \prime } < t )$ , we see that for every pair $( t , t ^ { \prime } ) \in \{ s , \ldots , s + N - 1 \} ^ { 2 }$

$$
\mathbb { E } \big [ \eta _ { t ^ { \prime } } ^ { \mathrm { v a r i a n c e } } \otimes \eta _ { t } ^ { \mathrm { v a r i a n c e } } \big ] = \gamma ^ { 2 } \sum _ { r = 0 } ^ { ( t \wedge t ^ { \prime } ) - 1 } ( \mathbf { I } - \gamma \mathbf { H } ) ^ { t ^ { \prime } - 1 - r } \mathbf { D } _ { r } ( \mathbf { I } - \gamma \mathbf { H } ) ^ { t - 1 - r } .
$$

Averaging over all $t , t ^ { \prime } \in \{ s , \ldots , s + N - 1 \}$ and exchanging the sum over $r \in \{ 0 , \ldots , s + N - 2 \}$ with the double sum over $t , t ^ { \prime } \geq s \vee \left( r + 1 \right)$ yields

$$
\begin{array} { r l } & { \mathbb { E } \big [ \overline { { \eta } } _ { s : s + N } ^ { \mathrm { s a r i a n c e } } \otimes \overline { { \eta } } _ { s : s + N } ^ { \mathrm { v a r i a n c e } } \big ] = \gamma ^ { 2 } \displaystyle \sum _ { r = 0 } ^ { s + N - 2 } \left( \frac { 1 } { N } \sum _ { t ^ { \prime } = s \vee ( r + 1 ) } ^ { s + N - 1 } ( { \mathbf I } - \gamma { \mathbf H } ) ^ { t ^ { \prime } - 1 - r } \right) { \mathbf D } _ { r } \left( \frac { 1 } { N } \sum _ { t = s \vee ( r + 1 ) } ^ { s + N - 1 } ( { \mathbf I } - \gamma { \mathbf H } ) ^ { t - 1 - r } \right) } \\ & { \qquad = \gamma ^ { 2 } \displaystyle \sum _ { r = 0 } ^ { s + N - 2 } { \mathbf Q } _ { r , s , N } \big ( ( { \mathcal M } - \widetilde { \mathcal { M } } ) \circ { \mathbf C } _ { r } + { \mathbf \Sigma } \big ) { \mathbf Q } _ { r , s , N } , } \end{array}
$$

which completes the proof of (5.7).

## A.2 Two-Sided Spectral Evaluation of Trajectory Functionals

Lemma A.1 (Two-Sided Spectral Evaluation of EfectiveBias, $\gamma _ { s , N }$ , and $\mathcal { S } _ { s , N } )$ . Suppose Assumption 3.1 holds and $0 < \gamma \leq 1 / ( 2 \lambda _ { 1 } )$ , and recall EfectiveBias, $\mathcal { S } _ { s , N } , \ \mathcal { V } _ { s , N }$ , and $( \mathbf { w } _ { s } ^ { \mathrm { d r i f t } } , \omega _ { j } , \Xi _ { N } )$ from (3.8), (5.8), and (A.1). Define the burn-in cross-spectral transfer functional

$$
\Phi _ { \mathrm { b u r n } } ( s , N ) : = \sum _ { i , j = 1 } ^ { \infty } \operatorname* { m i n } \left\{ \frac { 1 } { N ^ { 2 } } , \gamma ^ { 2 } \lambda _ { i } ^ { 2 } \right\} \lambda _ { j } \omega _ { j } ^ { 2 } \big ( 1 - \gamma ( \lambda _ { i } \wedge \lambda _ { j } ) \big ) ^ { 2 ( s - 1 ) } \operatorname* { m i n } \left\{ s , \frac { 1 } { \gamma | \lambda _ { i } - \lambda _ { j } | } \right\} .\tag{A.8}
$$

Then for all $s \geq 0$ and $N \geq 1$ (and $N \geq 2$ for the lower bounds on $\gamma _ { s , N }$ and $\mathcal { S } _ { s , N } ) , { } ^ { 4 }$

$$
\frac { 1 } { 8 } \left( \frac { 1 } { \gamma ^ { 2 } N ^ { 2 } } \lVert \mathbf { w } _ { s } ^ { \mathrm { d r i f t } } \rVert _ { { \mathbf { H } _ { 0 : k ^ { * } } ^ { - 1 } } } ^ { 2 } + \lVert \mathbf { w } _ { s } ^ { \mathrm { d r i f t } } \rVert _ { { \mathbf { H } _ { k ^ { * } : \infty } } } ^ { 2 } \right) \leq \mathrm { E f f e c t i v e B i a s }
$$

$$
\leq \frac { 1 } { 2 } \left( \frac { 1 } { \gamma ^ { 2 } N ^ { 2 } } \| \mathbf { w } _ { s } ^ { \mathrm { d r i f t } } \| _ { \mathbf { H } _ { 0 : k ^ { * } } ^ { - 1 } } ^ { 2 } + \| \mathbf { w } _ { s } ^ { \mathrm { d r i f t } } \| _ { \mathbf { H } _ { k ^ { * } : \infty } } ^ { 2 } \right) ,\tag{A.9}
$$

$$
\frac { 1 } { 2 1 6 } \left( \frac { k ^ { * } } { N } + \gamma \sum _ { k ^ { * } < i \leq k ^ { \dagger } } \lambda _ { i } + ( s + N ) \gamma ^ { 2 } \sum _ { i > k ^ { \dagger } } \lambda _ { i } ^ { 2 } \right) \leq \mathcal { V } _ { s , N } \leq \frac { k ^ { * } } { N } + \gamma \sum _ { k ^ { * } < i \leq k ^ { \dagger } } \lambda _ { i } + ( s + N ) \gamma ^ { 2 } \sum _ { i > k ^ { \dagger } } \lambda _ { i } ^ { 2 } ,\tag{A.10}
$$

and

$$
\mathcal { S } _ { s , N } \geq \frac { 1 } { 8 6 4 } \left( \frac { 1 } { \gamma N } \| \mathbf { w } _ { s } ^ { \mathrm { d r i f t } } \| _ { \mathbf { I } _ { 0 : k ^ { * } } } ^ { 2 } + \| \mathbf { w } _ { s } ^ { \mathrm { d r i f t } } \| _ { \mathbf { H } _ { k ^ { * } ; \infty } } ^ { 2 } \right) \Xi _ { N } + \frac { 1 } { 1 0 8 } \Phi _ { \mathrm { b u r n } } ( s , N ) ,
$$

$$
S _ { s , N } \leq \frac { 1 } { 2 } \left( \frac { 1 } { \gamma N } \| \mathbf { w } _ { s } ^ { \mathrm { d r i f t } } \| _ { \mathbf { I } _ { 0 : k ^ { * } } } ^ { 2 } + \| \mathbf { w } _ { s } ^ { \mathrm { d r i f t } } \| _ { \mathbf { H } _ { k ^ { * } ; \infty } } ^ { 2 } \right) \Xi _ { N } + \frac { 1 } { 2 } \Phi _ { \mathrm { b u r n } } ( s , N ) .\tag{A.11}
$$

Proof of Lemma A.1. We establish (A.9), (A.10), and (A.11) in three parts.

## Part 1: Evaluation of EfectiveBias

For each mode $i \geq 1$ , the elementary inequalities $\begin{array} { r } { \frac 1 2 ( 1 \wedge \gamma N \lambda _ { i } ) \leq ( 1 - e ^ { - 1 } ) ( 1 \wedge \gamma N \lambda _ { i } ) \leq 1 - ( 1 - \gamma \lambda _ { i } ) ^ { N } \leq } \end{array}$ $1 \wedge ( \gamma N \lambda _ { i } )$ imply

$$
\frac { 1 } { 4 } \left( \mathbf { 1 } _ { \{ i \leq k ^ { * } \} } + \gamma ^ { 2 } N ^ { 2 } \lambda _ { i } ^ { 2 } \mathbf { 1 } _ { \{ i > k ^ { * } \} } \right) \leq \left( 1 - \left( 1 - \gamma \lambda _ { i } \right) ^ { N } \right) ^ { 2 } \leq \mathbf { 1 } _ { \{ i \leq k ^ { * } \} } + \gamma ^ { 2 } N ^ { 2 } \lambda _ { i } ^ { 2 } \mathbf { 1 } _ { \{ i > k ^ { * } \} } .\tag{A.12}
$$

Multiplying (A.12) by $\begin{array} { r } { \frac { \omega _ { i } ^ { 2 } } { 2 \gamma ^ { 2 } N ^ { 2 } \lambda _ { i } } ( 1 - \gamma \lambda _ { i } ) ^ { 2 s } = \frac { ( \mathbf { v } _ { i } ^ { \top } \mathbf { w } _ { s } ^ { \mathrm { d r i f t } } ) ^ { 2 } } { 2 \gamma ^ { 2 } N ^ { 2 } \lambda _ { i } } } \end{array}$ and summing over $i \geq 1$ in (3.8) directly yields (A.9).

## Part 2: Evaluation of $\gamma _ { s , N }$ Across the Three Spectral Regimes

Using $\Psi _ { r , s , N } = \gamma ^ { 2 } F _ { r + 1 }$ from (A.2), we have $\begin{array} { r } { \mathcal { V } _ { s , N } = \frac { 1 } { 2 } \sum _ { r = 0 } ^ { s + N - 2 } \Psi _ { r , s , N } } \end{array}$ . We split this sum into the averaging window $r \in \{ s , \ldots , s + N - 2 \}$ and the burn-in phase $r \in \{ 0 , \ldots , s - 1 \}$

• Averaging-window contribution $( s \le r \le s + N - 2 )$ : Setting $m = s + N - 1 - r \in \{ 1 , \ldots , N - 1 \}$ gives

$$
\begin{array} { r l r } {  { \frac { 1 } { 2 } \sum _ { r = s } ^ { s + N - 2 } \Psi _ { r , s , N } = \frac { 1 } { 2 N ^ { 2 } } \sum _ { m = 1 } ^ { N - 1 } \sum _ { i = 1 } ^ { \infty } \bigl ( 1 - ( 1 - \gamma \lambda _ { i } ) ^ { m } \bigr ) ^ { 2 } } } \\ & { } & { \leq \frac { N - 1 } { 2 N ^ { 2 } } \sum _ { i = 1 } ^ { \infty } \bigl ( 1 \wedge \gamma ^ { 2 } N ^ { 2 } \lambda _ { i } ^ { 2 } \bigr ) \leq \frac { 1 } { 2 } \Xi _ { N } , } \end{array}
$$

where the first inequality uses $1 - ( 1 - \gamma \lambda _ { i } ) ^ { m } \leq 1 \wedge ( \gamma N \lambda _ { i } )$ . Conversely, for $N \geq 2$ , restricting the sum to $r \in \{ s , \ldots , s + \lfloor N / 2 \rfloor - 1 \}$ ensures $m \geq \lfloor N / 2 \rfloor$ , so

$$
\frac { 1 } { 2 } \sum _ { r = s } ^ { s + N - 2 } \Psi _ { r , s , N } \geq \frac { 1 } { 2 N ^ { 2 } } \sum _ { r = s } ^ { s + \lfloor N / 2 \rfloor - 1 } \sum _ { i = 1 } ^ { \infty } \bigl ( 1 - ( 1 - \gamma \lambda _ { i } ) ^ { \lfloor N / 2 \rfloor } \bigr ) ^ { 2 }
$$

$$
\geq \frac { \lfloor N / 2 \rfloor } { 1 8 N ^ { 2 } } \sum _ { i = 1 } ^ { \infty } \bigl ( 1 - ( 1 - \gamma \lambda _ { i } ) ^ { N } \bigr ) ^ { 2 } \geq \frac { 1 } { 2 1 6 } \Xi _ { N } ,
$$

where the second inequality applies Lemma C.2 and the third inequality uses (A.12) together with $\lfloor N / 2 \rfloor \ge N / 3$ for $N \geq 2$ . Combining both bounds gives

$$
\frac { 1 } { 2 1 6 } \left( \frac { k ^ { * } } { N } + N \gamma ^ { 2 } \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } \right) \leq \frac { 1 } { 2 } \sum _ { r = s } ^ { s + N - 2 } \Psi _ { r , s , N } \leq \frac { 1 } { 2 } \left( \frac { k ^ { * } } { N } + N \gamma ^ { 2 } \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } \right) .\tag{A.13}
$$

• Burn-in contribution $( 0 ~ \le ~ r ~ \le ~ s - 1$ , when $s \geq 1 )$ : Summing the geometric series over $0 \leq r \leq s - 1$ yields

$$
\frac { 1 } { 2 } \sum _ { r = 0 } ^ { s - 1 } \Psi _ { r , s , N } = \frac { 1 } { 2 N ^ { 2 } } \sum _ { i = 1 } ^ { \infty } \bigl ( 1 - ( 1 - \gamma \lambda _ { i } ) ^ { N } \bigr ) ^ { 2 } \frac { 1 - ( 1 - \gamma \lambda _ { i } ) ^ { 2 s } } { \gamma \lambda _ { i } ( 2 - \gamma \lambda _ { i } ) } .
$$

Since $\begin{array} { r } { \frac 1 4 ( s \wedge ( \gamma \lambda _ { i } ) ^ { - 1 } ) \leq \frac { 1 - ( 1 - \gamma \lambda _ { i } ) ^ { 2 s } } { \gamma \lambda _ { i } ( 2 - \gamma \lambda _ { i } ) } \leq s \wedge ( \gamma \lambda _ { i } ) ^ { - 1 } } \end{array}$ , applying (A.12) shows that $\textstyle { \frac { 1 } { 2 } } \sum _ { r = 0 } ^ { s - 1 } \Psi _ { r , s , N }$ is sandwiched between $\begin{array} { r } { \frac { 1 } { 3 2 } \sum _ { i \geq 1 } v _ { i } } \end{array}$ and $\begin{array} { r } { \frac { 1 } { 2 } \sum _ { i \geq 1 } v _ { i } } \end{array}$ , where $\begin{array} { r } { v _ { i } : = \frac { 1 } { N ^ { 2 } } ( 1 \wedge \gamma ^ { 2 } N ^ { 2 } \lambda _ { i } ^ { 2 } ) } \end{array}$ min $\{ s , ( \gamma \lambda _ { i } ) ^ { - 1 } \}$ Adapting the three-regime spectral partitioning strategy of Zou et al. (2021, Lemma D.3), we evaluate $v _ { i }$ across the cutofs $k ^ { * }$ and $k ^ { \dagger }$ :

1. Case $1 ~ ( i \leq k ^ { * }$ , so $\gamma \lambda _ { i } \geq N ^ { - 1 } )$ : Here $1 \land \gamma ^ { 2 } N ^ { 2 } \lambda _ { i } ^ { 2 } = 1$ and min $\left\{ s , ( \gamma \lambda _ { i } ) ^ { - 1 } \right\} \leq ( \gamma \lambda _ { i } ) ^ { - 1 } \leq N$ so $\begin{array} { r } { 0 \leq v _ { i } \leq \frac { 1 } { N } } \end{array}$

2. Case $\begin{array} { r } { \textnormal { \textsf { Z } } ( k ^ { * } < i \leq k ^ { \dagger } , \ s o \ ( s + N ) ^ { - 1 } \leq \gamma \lambda _ { i } < N ^ { - 1 } ) } \end{array}$ : Here $1 \wedge \gamma ^ { 2 } N ^ { 2 } \lambda _ { i } ^ { 2 } = \gamma ^ { 2 } N ^ { 2 } \lambda _ { i } ^ { 2 }$ , so $v _ { i } = \gamma ^ { 2 } \lambda _ { i } ^ { 2 } \operatorname* { m i n } \{ s , ( \gamma \lambda _ { i } ) ^ { - 1 } \}$ . In the upper bound, min $\{ s , ( \gamma \lambda _ { i } ) ^ { - 1 } \} \le ( \gamma \lambda _ { i } ) ^ { - 1 }$ gives $v _ { i } \leq \gamma \lambda _ { i }$ In the lower bound, if $s \geq ( \gamma \lambda _ { i } ) ^ { - 1 }$ , then $v _ { i } = \gamma \lambda _ { i } ; \mathrm { i f } s < ( \gamma \lambda _ { i } ) ^ { - 1 } \leq s + N$ , then adding the averaging-window contribution $N \gamma ^ { 2 } \lambda _ { i } ^ { 2 }$ from (A.13) gives $v _ { i } + N \gamma ^ { 2 } \lambda _ { i } ^ { 2 } = ( s + N ) \gamma ^ { 2 } \lambda _ { i } ^ { 2 } \ge \gamma \lambda _ { i }$

3. Case $3 ~ ( i ~ > ~ k ^ { \dagger }$ , so $\gamma \lambda _ { i } ~ < ~ ( s + N ) ^ { - 1 } ~ \leq ~ s ^ { - 1 } ) .$ : Here $1 \wedge \gamma ^ { 2 } N ^ { 2 } \lambda _ { i } ^ { 2 } \ = \ \gamma ^ { 2 } N ^ { 2 } \lambda _ { i } ^ { 2 }$ and min $\{ s , ( \gamma \lambda _ { i } ) ^ { - 1 } \} = s , \mathrm { s o } v _ { i } = s \gamma ^ { 2 } \lambda _ { i } ^ { 2 }$

Summing Cases 1–3 and combining with (A.13) (noting that for $k ^ { * } < i \leq k ^ { \dagger } , N \gamma ^ { 2 } \lambda _ { i } ^ { 2 } \leq \gamma \lambda _ { i }$ in the upper bound) yields (A.10).

## Part 3: Evaluation of $\mathcal { S } _ { s , N }$

Writing $\begin{array} { r } { b _ { r } = \| ( \mathbf { I } - \gamma \mathbf { H } ) ^ { r } ( \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } ) \| _ { \mathbf { H } } ^ { 2 } = \sum _ { j > 1 } \lambda _ { j } \omega _ { j } ^ { 2 } ( 1 - \gamma \lambda _ { j } ) ^ { 2 r } } \end{array}$ , we have $\begin{array} { r } { S _ { s , N } = \frac { 1 } { 2 } \sum _ { r = 0 } ^ { s + N - 2 } b _ { r } \Psi _ { r , s , N } } \end{array}$ We split this sum into the averaging window $r \stackrel {  } { = } s + k ( 0 \leq k \leq N - 2 )$ and the burn-in phase $0 \leq r \leq s - 1$

• Averaging-window contribution $( r ~ = ~ s + k , ~ 0 ~ \le ~ k ~ \le ~ N - 2 )$ : For $N \ \geq \ 2 , 5$ note first from Part 2 that $\Psi _ { s + k , s , N } \leq \Psi _ { s , s , N } \leq \frac { 1 } { N } \Xi _ { N }$ for all $0 \le k \le N - 2$ , while Lemma C.2 and (A.12) give $\Psi _ { s + k , s , N } \geq \frac { 1 } { 3 6 N } \Xi _ { N }$ for all $0 \le k \le \lfloor N / 2 \rfloor - 1$ . Moreover, since $b _ { s + k } =$ $\begin{array} { r } { \sum _ { j \geq 1 } \lambda _ { j } ( \mathbf { v } _ { j } ^ { \top } \mathbf { w } _ { s } ^ { \mathrm { d r i f t } } ) ^ { 2 } ( 1 - \gamma \lambda _ { j } ) ^ { 2 \dot { k } } } \end{array}$ , summing the geometric series over k gives

$$
\gamma \sum _ { k = 0 } ^ { N - 1 } b _ { s + k } = \sum _ { j = 1 } ^ { \infty } \frac { 1 - ( 1 - \gamma \lambda _ { j } ) ^ { 2 N } } { 2 - \gamma \lambda _ { j } } ( \mathbf { v } _ { j } ^ { \top } \mathbf { w } _ { s } ^ { \mathrm { { d r i f t } } } ) ^ { 2 }
$$

$$
\begin{array} { r } { \leq \| \mathbf { w } _ { s } ^ { \mathrm { d r i f t } } \| _ { \mathbf { I } _ { 0 : k ^ { * } } } ^ { 2 } + \gamma N \| \mathbf { w } _ { s } ^ { \mathrm { d r i f t } } \| _ { \mathbf { H } _ { k ^ { * } : \infty } } ^ { 2 } , } \end{array}
$$

where the inequality uses $\begin{array} { r } { \frac { 1 - ( 1 - \gamma \lambda _ { j } ) ^ { 2 N } } { 2 - \gamma \lambda _ { j } } \leq 1 } \end{array}$ on $j \leq k ^ { * }$ (since $2 - \gamma \lambda _ { j } \geq 1 )$ and $\begin{array} { r } { \gamma \sum _ { k = 0 } ^ { N - 1 } \lambda _ { j } ( 1 - } \\ { \gamma \sum _ { k = 0 } ^ { N - 1 } \lambda _ { j } ( 1 - } \end{array}$ $\gamma \lambda _ { j } ) ^ { 2 k } \leq \gamma N \lambda _ { j }$ on $j > k ^ { * }$ . Conversely, restricting the sum to $0 \le k \le \lfloor N / 2 \rfloor - 1$ yields

$$
\begin{array} { r } { \gamma \displaystyle \sum _ { k = 0 } ^ { \lfloor N / 2 \rfloor - 1 } b _ { s + k } = \displaystyle \sum _ { j = 1 } ^ { \infty } \frac { 1 - ( 1 - \gamma \lambda _ { j } ) ^ { 2 \lfloor N / 2 \rfloor } } { 2 - \gamma \lambda _ { j } } ( { \mathbf { v } } _ { j } ^ { \top } { \mathbf { w } } _ { s } ^ { \mathrm { d r i f t } } ) ^ { 2 } } \\ { \geq \displaystyle \frac { 1 } { 6 } \displaystyle \sum _ { j = 1 } ^ { \infty } \big [ 1 - ( 1 - \gamma \lambda _ { j } ) ^ { 2 N } \big ] ( { \mathbf { v } } _ { j } ^ { \top } { \mathbf { w } } _ { s } ^ { \mathrm { d r i f t } } ) ^ { 2 } } \\ { \geq \displaystyle \frac { 1 } { 1 2 } \left( \| { \mathbf { w } } _ { s } ^ { \mathrm { d r i f t } } \| _ { { \mathbf { I } } _ { 0 ; k ^ { * } } } ^ { 2 } + \gamma N \| { \mathbf { w } } _ { s } ^ { \mathrm { d r i f t } } \| _ { { \mathbf { H } } _ { k ^ { * } ; \infty } } ^ { 2 } \right) , } \end{array}
$$

where the first inequality applies Lemma C.2 together with $2 - \gamma \lambda _ { j } \leq 2$ , and the second inequality uses $\begin{array} { r } { 1 - ( 1 - \gamma \lambda _ { j } ) ^ { 2 N } \geq 1 - ( 1 - \gamma \lambda _ { j } ) ^ { N } \geq \frac { 1 } { 2 } ( 1 \wedge \gamma N \lambda _ { j } ) } \end{array}$ . Combining the filter and geometric-sum bounds yields

$$
\begin{array} { r l } & { \frac { 1 } { 8 6 4 } \left( \frac { 1 } { \gamma \mathcal { N } } \| \mathbf { w } _ { s } ^ { \mathrm { d r i f t } } \| _ { \mathbf { I } _ { 0 : k ^ { * } } } ^ { 2 } + \| \mathbf { w } _ { s } ^ { \mathrm { d r i f t } } \| _ { \mathbf { H } _ { k ^ { * } \times \infty } } ^ { 2 } \right) \Xi _ { N } \leq \displaystyle \frac { 1 } { 2 } \sum _ { k = 0 } ^ { N - 2 } b _ { s + k } \Psi _ { s + k , s , N } } \\ & { \qquad \leq \displaystyle \frac { 1 } { 2 } \left( \frac { 1 } { \gamma \mathcal { N } } \| \mathbf { w } _ { s } ^ { \mathrm { d r i f t } } \| _ { \mathbf { I } _ { 0 : k ^ { * } } } ^ { 2 } + \| \mathbf { w } _ { s } ^ { \mathrm { d r i f t } } \| _ { \mathbf { H } _ { k ^ { * } \times \infty } } ^ { 2 } \right) \Xi _ { N } . } \end{array}\tag{A.14}
$$

• Burn-in contribution $( 0 \leq r \leq s - 1$ , when $s \geq 1 ,$ : Expanding $b _ { r }$ and $\Psi _ { r , s , N }$ gives

$$
\frac { 1 } { 2 } \sum _ { r = 0 } ^ { s - 1 } b _ { r } \Psi _ { r , s , N } = \frac { 1 } { 2 N ^ { 2 } } \sum _ { i = 1 } ^ { \infty } \left( 1 - ( 1 - \gamma \lambda _ { i } ) ^ { N } \right) ^ { 2 } \sum _ { j = 1 } ^ { \infty } \lambda _ { j } \omega _ { j } ^ { 2 } \sum _ { r = 0 } ^ { s - 1 } ( 1 - \gamma \lambda _ { j } ) ^ { 2 r } ( 1 - \gamma \lambda _ { i } ) ^ { 2 ( s - 1 - r ) } .
$$

For each mode pair $( i , j )$ , let $p : = ( 1 - \gamma \lambda _ { j } ) ^ { 2 }$ and $q : = ( 1 - \gamma \lambda _ { i } ) ^ { 2 }$ , so that $M : = p \vee q =$ $( 1 - \gamma ( \lambda _ { i } \wedge \lambda _ { j } ) ) ^ { 2 } \in [ 1 / 4 , 1 ]$ (since $\gamma \lambda _ { 1 } \leq 1 / 2 )$ and $M - m = \gamma | \lambda _ { i } - \lambda _ { j } | ( 2 - \gamma ( \lambda _ { i } + \lambda _ { j } ) ) \in$ $[ \gamma | \lambda _ { i } - \lambda _ { j } | , 2 \gamma | \lambda _ { i } - \lambda _ { j } | ]$ . Consequently, $\begin{array} { r } { \frac { 1 } { 8 \gamma | \lambda _ { i } - \lambda _ { j } | } \le \frac { M } { M - m } \le \frac { 1 } { \gamma | \lambda _ { i } - \lambda _ { j } | } } \end{array}$ . Applying Lemma C.3 to the inner geometric convolution over r and (A.12) to $( 1 - ( 1 - \gamma \bar { \lambda } _ { i } ) ^ { N } ) ^ { 2 }$ term-by-term yields

$$
\frac { 1 } { 1 0 8 } \Phi _ { \mathrm { b u r n } } ( s , N ) \leq \frac { 1 - e ^ { - 1 } } { 6 4 } \Phi _ { \mathrm { b u r n } } ( s , N ) \leq \frac { 1 } { 2 } \sum _ { r = 0 } ^ { s - 1 } b _ { r } \Psi _ { r , s , N } \leq \frac { 1 } { 2 } \Phi _ { \mathrm { b u r n } } ( s , N ) .\tag{A.15}
$$

Adding (A.14) and (A.15) establishes (A.11).

## B Proofs of Main Theorems, Propositions, and Corollaries

## B.1 Proof of Theorem 4.1, Theorem 4.2, and Corollary 4.5

Proof of Theorem 4.1 and Theorem 4.2. For Theorem 4.1, Lemma C.6(1) gives $\mathbb { E } [ L ( \overline { { \mathbf { w } } } _ { s : s + N } ) ] \mathrm { ~ - ~ }$ $L ( \mathbf { w } ^ { * } ) \leq 2 \cdot \mathrm { b i a s } + 2 \cdot \mathrm { v a r i a n c e } = 2$ · EfectiveBias + 2 · EfectiveVar under Assumptions 3.1–3.3 (and equality without the factor of 2 under the well-specified model (3.4) by Lemma C.6(2)). Invoking Lemma 5.2(1) and substituting the upper bounds $( \mathrm { A . 9 } ) { - } ( \mathrm { A . 1 1 } )$ from Lemma A.1 directly establishes (4.3) and (4.4).

For Theorem 4.2, Lemma C.6(2) gives the exact decomposition $\mathbb { E } [ L ( \overline { { \mathbf { w } } } _ { s : s + N } ) ] - L ( \mathbf { w } ^ { * } ) \ =$ EfectiveBias + EfectiveVar under (3.4). Invoking Lemma 5.2(2) and substituting the lower bounds (A.9)–(A.11) from Lemma A.1 directly establishes (4.6) and (4.7). □

Proof of Corollary 4.5. Suppose $c _ { 0 } N \leq s \leq C _ { 0 } N$ , and recall $\mathbf { w } _ { s } ^ { \mathrm { d r i f t } } , \ \omega _ { j }$ , and $\Phi _ { \mathrm { b u r n } } ( s , N )$ from $\left( \mathrm { A . 1 } \right)$ and $\left( \mathrm { A } . 8 \right)$ . For every slow mode $i ~ > ~ k ^ { * } .$ , we have $\gamma \lambda _ { i } ~ < ~ N ^ { - 1 } ~ \le ~ C _ { 0 } / s$ , which implies $( 1 - \gamma \lambda _ { i } ) ^ { 2 s } \in [ e ^ { - 4 C _ { 0 } } , 1 ]$ and hence $\| \mathbf { w } _ { s } ^ { \mathrm { d r i f t } } \| _ { { \mathbf { H } _ { k ^ { * } : \infty } } } ^ { 2 } \asymp \| \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } \| _ { \mathbf { H } _ { k ^ { * } : \infty } } ^ { 2 }$ . Similarly, for every intermediate mode $i \in ( k ^ { * } , k ^ { \dagger } ]$ , we have $( ( 1 + C _ { 0 } ) N ) ^ { - 1 } \leq \gamma \lambda _ { i } < N ^ { - 1 }$ , so $\gamma \dot { \lambda _ { i } } \stackrel { . . . } { \sim } N \gamma ^ { 2 } \lambda _ { i } ^ { 2 } \asymp N ^ { - 1 }$

To evaluate $\Phi _ { \mathrm { b u r n } } ( s , N )$ in (A.8), we partition the double sum over $( i , j )$ into three blocks:

• Fast-to-fast block $( i \leq k ^ { * }$ and $j \leq k ^ { * } ) ,$ : Since min $\{ N ^ { - 2 } , \gamma ^ { 2 } \lambda _ { i } ^ { 2 } \} = N ^ { - 2 } \mathrm { o n } i \le k ^ { * }$ , this block equals the third line of (4.15).

• Slow-to-fast block $( i ~ \leq ~ k ^ { * }$ and $j > k ^ { * } )$ : Using min $\lbrace N ^ { - 2 } , \gamma ^ { 2 } \lambda _ { i } ^ { 2 } \rbrace = N ^ { - 2 }$ and $( 1 - \gamma ( \lambda _ { i } \wedge$ $\lambda _ { j } ) ) ^ { 2 ( s - 1 ) } \leq 1$ , we upper-bound this block by

$$
\begin{array} { r l r } {  { \frac { 1 } { N ^ { 2 } } \sum _ { i \le k ^ { * } j > k ^ { * } } \lambda _ { j } \omega _ { j } ^ { 2 } \operatorname* { m i n } \{ s , \frac { 1 } { \gamma | \lambda _ { i } - \lambda _ { j } | } \} \le \frac { 2 \vee C _ { 0 } } { N } \sum _ { i \le k ^ { * } } \sum _ { j > k ^ { * } } \lambda _ { j } \omega _ { j } ^ { 2 } } } \\ & { } & { = ( 2 \vee C _ { 0 } ) \| \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } \| _ { \mathbf { H } _ { k ^ { * } ; \infty } } ^ { 2 } \frac { k ^ { * } } { N } , } \end{array}
$$

where the inequality uses min $\{ s , ( \gamma | \lambda _ { i } - \lambda _ { j } | ) ^ { - 1 } \} \le 2 ( \gamma \lambda _ { i } ) ^ { - 1 } \le 2 N$ when $\lambda _ { j } \ \leq \ \lambda _ { i } / 2$ and min $\{ s , ( \gamma | \lambda _ { i } - \lambda _ { j } | ) ^ { - 1 } \} \le s \le C _ { 0 } N$ when $\lambda _ { j } > \lambda _ { i } / 2$ . This contribution is absorbed into the second line of (4.15).

• Slow receiving block $\mathit { \Omega } ^ { \mathrm { ~ \textit ~ { ~  ~ } ~ } } \mathit { \Omega } ^ { \mathrm { ~ \textit ~ { ~  ~ } ~ } }$ and $j \geq 1 )$ : On $i > k ^ { * }$ , we have min $\{ N ^ { - 2 } , \gamma ^ { 2 } \lambda _ { i } ^ { 2 } \} = \gamma ^ { 2 } \lambda _ { i } ^ { 2 }$ and $\lambda _ { i } \wedge \lambda _ { j } \leq \lambda _ { i } < ( \gamma N ) ^ { - 1 }$ , so $( 1 - \gamma ( \lambda _ { i } \wedge \lambda _ { j } ) ) ^ { 2 ( s - 1 ) } \in [ e ^ { - 4 C _ { 0 } } , 1 ]$ . Moreover, for $j \ \leq \ k ^ { * }$ , we have min $\{ s , ( \gamma | \lambda _ { i } - \lambda _ { j } | ) ^ { - 1 } \} \asymp ( \gamma \lambda _ { j } ) ^ { - 1 }$ (both when $\lambda _ { j } \geq 2 \lambda _ { i }$ and when $\lambda _ { i } < \lambda _ { i } < 2 \lambda _ { i }$ , since $s \asymp N \asymp ( \gamma \lambda _ { j } ) ^ { - 1 }$ in the latter case), whereas for $j > k ^ { * }$ , both $\gamma \lambda _ { i } < N ^ { - 1 }$ and $\gamma \lambda _ { j } < N ^ { - 1 }$ imply min $\{ s , \bar { ( \gamma | \lambda _ { i } - \lambda _ { j } | ) ^ { - 1 } } \} \asymp N$ . Summing over $j \geq 1$ and $i > k ^ { * }$ therefore gives

$$
\begin{array} { r l r } {  { \sum _ { i > k ^ { * } } \gamma ^ { 2 } \lambda _ { i } ^ { 2 } \sum _ { j = 1 } ^ { \infty } \lambda _ { j } \omega _ { j } ^ { 2 } \big ( 1 - \gamma ( \lambda _ { i } \wedge \lambda _ { j } ) \big ) ^ { 2 ( s - 1 ) } \operatorname* { m i n } \{ s , \frac { 1 } { \gamma | \lambda _ { i } - \lambda _ { j } | } \} } } \\ & { } & { \asymp ( \sum _ { i > k ^ { * } } \gamma ^ { 2 } \lambda _ { i } ^ { 2 } ) ( \frac { 1 } { \gamma } \| \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } \| _ { \mathbf { I } _ { 0 : k ^ { * } } } ^ { 2 } + N \| \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } \| _ { \mathbf { H } _ { k ^ { * } \infty } } ^ { 2 } ) } \\ & { } & { = \frac { \| \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } \| _ { \mathbf { I } _ { 0 : k ^ { * } } } ^ { 2 } + N \gamma \| \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } \| _ { \mathbf { H } _ { k ^ { * } \infty } } ^ { 2 } } { N \gamma } \cdot ( N \gamma ^ { 2 } \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } ) , } \end{array}
$$

which yields the first line of (4.15).

Substituting these evaluations into Theorems 4.1 and 4.2 establishes (4.14)–(4.15).

## B.2 Proof of Proposition 4.3 and Corollary 4.4

Proof of Proposition 4.3. Expanding $\begin{array} { r } { q _ { t } ( \lambda _ { i } ) ^ { 2 } = \frac { 1 } { N ^ { 2 } } \sum _ { a = s \vee t } ^ { T } \sum _ { b = s \vee t } ^ { T } ( 1 - \gamma \lambda _ { i } ) ^ { a + b - 2 t } } \end{array}$ in (4.1) and summing against $\lambda _ { i } ^ { 2 }$ yields the pair-midpoint identity (4.9). When $T \leq 1$ , the maximum in (4.2) is empty so $\chi _ { s , N } = 0$ . Suppose $T \geq 2$ and fix any $r \in \{ 1 , \ldots , T - 1 \}$ . Since $s \vee t \geq s \vee ( r + 1 )$ for all $t \geq r + 1$ , substituting (4.9) into $\textstyle \gamma ^ { 2 } \sum _ { t = r + 1 } ^ { T } k ( t - r - 1 ) F _ { t }$ and exchanging the order of summation gives

$$
\gamma ^ { 2 } \sum _ { t = r + 1 } ^ { T } k ( t - r - 1 ) F _ { t } = \frac { \gamma ^ { 2 } } { N ^ { 2 } } \sum _ { a = s \vee ( r + 1 ) } ^ { T } \sum _ { b = s \vee ( r + 1 ) } ^ { T } \sum _ { t = r + 1 } ^ { \operatorname* { m i n } ( a , b ) } k ( t - r - 1 ) k \Big ( \frac { a + b } { 2 } - t \Big ) .\tag{B.1}
$$

For each pair $( a , b )$ , set $j : = t - r - 1 \geq 0$ and $\textstyle h : = { \frac { a + b } { 2 } } - r \in [ 1 , T ]$ , so the inner sum in (B.1) runs over $0 \leq j \leq \operatorname* { m i n } ( a , b ) - r - 1 \leq T - 2$ with $\begin{array} { r } { h - 1 - j = \frac { a + b } { 2 } - t \geq 0 } \end{array}$

For part (1), for every $0 \leq j \leq \operatorname* { m i n } ( a , b ) - r - 1$ , we have

$$
\begin{array} { l } { { \displaystyle k ( j ) k ( h - 1 - j ) \le \mathrm { t r } ( { \bf H } ^ { 2 } ) k ( h - 1 ) } \ ~ } \\ { { \displaystyle ~ = \mathrm { t r } ( { \bf H } ^ { 2 } ) \sum _ { i = 1 } ^ { \infty } \lambda _ { i } ^ { 2 } ( 1 - \gamma \lambda _ { i } ) ^ { 2 h - 2 } \le \frac { \mathrm { t r } ( { \bf H } ^ { 2 } ) } { ( 1 - \gamma \lambda _ { 1 } ) ^ { 2 } } k ( h ) , } } \end{array}
$$

where the first inequality follows from Chebyshev’s sum inequality (Lemma C.1 with $u = j$ and $v = h - 1 - j )$ and the second inequality uses $( 1 - \gamma \lambda _ { i } ) ^ { - 2 } \leq ( 1 - \gamma \lambda _ { 1 } ) ^ { - 2 }$ . Since the inner sum over $j$ in (B.1) contains at most $T - 1$ terms and $\begin{array} { r } { \frac { 1 } { N ^ { 2 } } \sum _ { a , b = s \vee ( r + 1 ) } ^ { T } k ( \frac { a + b } { 2 } - r ) \leq \frac { 1 } { N ^ { 2 } } \sum _ { a , b = s \vee r } ^ { T } k ( \frac { a + b } { 2 } - r ) = F _ { r } } \end{array}$ summing over $a , b \colon$ yields $\begin{array} { r } { \gamma ^ { 2 } \sum _ { t = r + 1 } ^ { T } k ( t - r - 1 ) F _ { t } \leq \frac { \gamma ^ { 2 } ( T - 1 ) \operatorname { t r } ( { \bf H } ^ { 2 } ) } { ( 1 - \gamma \lambda _ { 1 } ) ^ { 2 } } F _ { r } . } \end{array}$ , which establishes (4.10) and (4.11).

For part (2), since k(·) is non-negative and non-increasing on $[ 0 , \infty )$ , splitting the inner convolution over $0 \leq j \leq \operatorname* { m i n } ( a , b ) - r - 1 \leq h - 1$ at the midpoint $( h - 1 ) / 2 :$ yields

$$
\begin{array} { l } { { \displaystyle \sum _ { j = 0 } ^ { \operatorname* { m i n } ( a , b ) - r - 1 } k ( j ) k ( h - 1 - j ) \le \sum _ { 0 \le j \le ( k - 1 ) / 2 } k ( j ) k \Big ( \frac { h - 1 } { 2 } \Big ) + \sum _ { ( h - 1 ) / 2 < j \le h - 1 } k \Big ( \frac { h - 1 } { 2 } \Big ) k ( h - 1 - j ) } } \\ { { \le 2 k \Big ( \frac { h - 1 } { 2 } \Big ) \displaystyle \sum _ { n = 0 } ^ { \infty } k ( n ) \le 2 D _ { T } k ( h ) \displaystyle \sum _ { n = 0 } ^ { \infty } k ( n ) , } } \end{array}
$$

where the first inequality bounds $k ( h - 1 - j ) \leq k ( ( h - 1 ) / 2 )$ on the first half and $k ( j ) \leq k ( ( h - 1 ) / 2 )$ on the second half, and the third inequality uses $k ( ( h - 1 ) / 2 ) \leq D _ { T } k ( h )$ from (4.8). Substituting this bound into (B.1) and using $\begin{array} { r } { \frac { 1 } { N ^ { 2 } } \sum _ { a , b = s \vee ( r + 1 ) } ^ { T } k ( \frac { a + b } { 2 } - r ) \leq F _ { r } } \end{array}$ gives

$$
\gamma ^ { 2 } \sum _ { t = r + 1 } ^ { T } k ( t - r - 1 ) F _ { t } \leq 2 D _ { T } \gamma ^ { 2 } \biggl ( \sum _ { n = 0 } ^ { \infty } k ( n ) \biggr ) F _ { r } .
$$

Multiplying by $\alpha / F _ { r }$ , taking the maximum over $1 \leq r < T$ , and evaluating the geometric series $\begin{array} { r } { \gamma ^ { 2 } \sum _ { n = 0 } ^ { \infty } k ( n ) = \gamma \sum _ { i \geq 1 } \frac { \lambda _ { i } } { 2 - \gamma \lambda _ { i } } \leq \frac { \gamma \operatorname { t r } ( \mathbf { H } ) } { 2 - \gamma \lambda _ { 1 } } } \end{array}$ establishes (4.12). □

Proof of Corollary $4 . 4 \cdot$ . Let $\nu : = 2 - 1 / a \in ( 1 , 2 )$ and $z : = \gamma \Lambda h \geq 0$ . Since $\gamma \lambda _ { i } \leq \gamma \Lambda \leq 1 / 2$ , the elementary inequalities $- 2 x \le \log ( 1 - x ) \le - x$ for $x \in [ 0 , 1 / 2 ]$ imply $e ^ { - 4 \gamma h \lambda _ { i } } \leq ( 1 - \gamma \lambda _ { i } ) ^ { 2 h } \leq e ^ { - 2 \gamma h \lambda _ { i } }$

• Upper bound on k(h): For $\begin{array} { r } { z \in [ 0 , 1 ] , k ( h ) \le k ( 0 ) \le \Lambda ^ { 2 } \sum _ { i > 1 } i ^ { - 2 a } \le 2 ^ { \nu } \Lambda ^ { 2 } ( \sum _ { i > 1 } i ^ { - 2 a } ) ( 1 + z ) ^ { - \nu } } \end{array}$ For $z > 1$ , set $M : = \lceil z ^ { 1 / a } \rceil \geq 2$ and split the sum at $M$ :

$$
\begin{array} { l } { { \displaystyle k ( h ) \le \Lambda ^ { 2 } \sum _ { i = 1 } ^ { M } i ^ { - 2 a } e ^ { - 2 \theta z i ^ { - a } } + \Lambda ^ { 2 } \sum _ { i = M + 1 } ^ { \infty } i ^ { - 2 a } } } \\ { { \displaystyle \qquad \le \Lambda ^ { 2 } ( \theta z ) ^ { - 2 } \sum _ { i = 1 } ^ { M } 1 + \Lambda ^ { 2 } \int _ { M } ^ { \infty } x ^ { - 2 a } \mathrm { d } x } } \\ { { \displaystyle \qquad \le \Lambda ^ { 2 } \left( 2 \theta ^ { - 2 } + \frac { 1 } { 2 a - 1 } \right) z ^ { - \nu } \le C ( a , \theta ) \Lambda ^ { 2 } ( 1 + \gamma \Lambda h ) ^ { - \nu } , } } \end{array}
$$

where the second inequality uses $v ^ { 2 } e ^ { - v } \leq 4 e ^ { - 2 } \leq 1$ with $v = \theta z i ^ { - a }$ on $1 \leq i \leq M$ and integral comparison on $i > M$ , the third inequality uses $M = \lceil z ^ { 1 / a } \rceil \leq 2 z ^ { 1 / a }$ and $M ^ { 1 - 2 a } \leq z ^ { - \nu }$ , and the last inequality uses $z ^ { - \nu } \leq 2 ^ { \nu } ( 1 + z ) ^ { - \nu } { \mathrm { ~ f o r ~ } } z > 1$

• Lower bound on $k ( h )$ $\mathrm { F o r } z \in [ 0 , 1 ]$ , retaining $i = 1$ gives $k ( h ) \geq \theta ^ { 2 } \Lambda ^ { 2 } e ^ { - 4 z } \geq \theta ^ { 2 } e ^ { - 4 } \Lambda ^ { 2 } ( 1 + z ) ^ { - \nu }$ For $1 \leq z \leq ( d / 2 ) ^ { a }$ (with $d = \infty$ in infinite dimension), let $m : = \lfloor z ^ { 1 / a } \rfloor \geq 1$ , so that $2 m \leq d$ and $m + 1 > z ^ { 1 / a }$ . Restricting the sum to $m + 1 \leq i \leq 2 m$ , where $z i ^ { - a } \leq z ( m + 1 ) ^ { - a } < 1$ gives

$$
\begin{array} { c } { { k ( h ) \ge \theta ^ { 2 } \Lambda ^ { 2 } \displaystyle \sum _ { i = m + 1 } ^ { 2 m } i ^ { - 2 a } e ^ { - 4 z i ^ { - a } } \ge \theta ^ { 2 } \Lambda ^ { 2 } e ^ { - 4 } \displaystyle \sum _ { i = m + 1 } ^ { 2 m } i ^ { - 2 a } } } \\ { { \ge \theta ^ { 2 } \Lambda ^ { 2 } e ^ { - 4 } m ( 2 m ) ^ { - 2 a } \ge \theta ^ { 2 } e ^ { - 4 } 2 ^ { - 2 a } \Lambda ^ { 2 } ( 1 + \gamma \Lambda h ) ^ { - \nu } , } } \end{array}
$$

where the last inequality uses $m \leq z ^ { 1 / a } \leq ( 1 + z ) ^ { 1 / a }$ and $1 - 2 a < 0$

Combining the upper and lower kernel bounds, for every $1 \leq h \leq T$ (with $\gamma \Lambda T \leq ( d / 2 ) ^ { a } )$

$$
\frac { k ( ( h - 1 ) / 2 ) } { k ( h ) } \leq \frac { C ( a , \theta ) } { \theta ^ { 2 } e ^ { - 4 } 2 ^ { - 2 a } } \bigg ( \frac { 1 + \gamma \Lambda h } { 1 + \gamma \Lambda ( h - 1 ) / 2 } \bigg ) ^ { \nu } \leq \frac { 2 ^ { \nu } C ( a , \theta ) } { \theta ^ { 2 } e ^ { - 4 } 2 ^ { - 2 a } } = : D _ { * } ( a , \theta ) .
$$

Consequently, $D _ { T } ~ \leq ~ D _ { * }$ , and for any $\gamma$ satisfying (4.13), Proposition 4.3(2) yields $\chi _ { s , N } ~ \leq$ $\begin{array} { r } { \frac { 2 \alpha D _ { * } \gamma \operatorname { t r } ( \mathbf { H } ) } { 2 - \gamma \lambda _ { 1 } } \leq 2 \alpha D _ { * } \gamma \operatorname { t r } ( \mathbf { H } ) \leq 1 / 2 } \end{array}$ □

## B.3 Proof of Proposition 4.7 and Proposition 4.8

Proof of Proposition $4 . 7 .$ For part (1), we have $\lambda _ { 1 } = 1 , \lambda _ { 2 } = ( 4 \gamma N ) ^ { - 1 } , \omega _ { 1 } = 1 , \omega _ { 2 } = 0 , \sigma _ { \mathrm { n o i s e } } ^ { 2 } = 0 \mathrm { . }$ and $\beta = 1$ (since $\mathbf { x } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { H } ) )$ ). Let $m : = \lfloor 1 / ( 4 \gamma ) \rfloor + 1 \le N$ . For every $1 \leq t \leq m$ , we have $t - 1 \leq 1 / ( 4 \gamma )$ , so Bernoulli’s inequality gives $( 1 - \gamma ) ^ { 2 ( t - 1 ) } \geq 1 - 2 \gamma ( t - 1 ) \geq 1 / 2$ . Meanwhile, for every $1 \leq t \leq N$ and $N \leq k \leq 2 N - 1$ , we have $k - t \leq 2 N$ , so $( 1 - \gamma \lambda _ { 2 } ) ^ { k - t } = ( 1 - \textstyle { \frac { 1 } { 4 N } } ) ^ { k - t } \ge 1 - \textstyle { \frac { k - t } { 4 N } } \ge 1 / 2 .$ which implies $\begin{array} { r } { q _ { t } ( \lambda _ { 2 } ) = \frac { 1 } { N } \sum _ { k = N } ^ { 2 N - 1 } ( 1 - \gamma \lambda _ { 2 } ) ^ { k - t } \ge 1 / 2 } \end{array}$ and $\begin{array} { r } { F _ { t } \ge \lambda _ { 2 } ^ { 2 } q _ { t } ( \lambda _ { 2 } ) ^ { 2 } \ge \frac { 1 } { 4 } ( 4 \gamma N ) ^ { - 2 } = \frac { 1 } { 6 4 \gamma ^ { 2 } N ^ { 2 } } } \end{array}$ Substituting these lower bounds into (5.10) of Lemma 5.2 with $\beta = 1$ yields

$$
\begin{array} { r l } & { \displaystyle \mathbb { E } \big [ L ( \overline { { \mathbf { w } } } _ { N : 2 N } ) \big ] - L ( \mathbf { w } ^ { * } ) \geq \mathcal { S } _ { N , N } \geq \frac { \gamma ^ { 2 } } { 2 } \sum _ { t = 1 } ^ { m } ( 1 - \gamma ) ^ { 2 ( t - 1 ) } F _ { t } } \\ & { \qquad \geq \frac { \gamma ^ { 2 } } { 2 } \cdot m \cdot \frac { 1 } { 2 } \cdot \frac { 1 } { 6 4 \gamma ^ { 2 } N ^ { 2 } } \geq \frac { 1 } { 1 0 2 4 \gamma N ^ { 2 } } , } \end{array}
$$

where the last inequality uses $m > 1 / ( 4 \gamma )$ . By contrast, since $\lambda _ { 1 } = 1 \ge ( 2 \gamma N ) ^ { - 1 } > \lambda _ { 2 } = ( 4 \gamma N ) ^ { - 1 }$ we have $k ^ { * } = k ^ { \dagger } = 1$ and $\| \mathbf { w } _ { 0 } - \mathbf { w } ^ { * } \| _ { \mathbf { H } _ { k ^ { \dagger } : \infty } } ^ { 2 } = \lambda _ { 2 } \omega _ { 2 } ^ { 2 } = 0$ , so Zou et al. (2021, Theorem 5.2) drops the variance contribution entirely and retains only the bias lower bound $\begin{array} { r } { \frac { 1 } { 1 0 0 \gamma ^ { 2 } N ^ { 2 } } \| ( \mathbf { I } - \gamma \mathbf { H } ) ^ { N } ( \mathbf { w } _ { 0 } - \mathbf { \sigma } } \end{array}$ $\begin{array} { r } { \mathbf { w } ^ { * } ) \| _ { \mathbf { H } _ { 0 \cdot k ^ { * } } ^ { - 1 } } ^ { 2 } = \frac { ( 1 - \gamma ) ^ { 2 N } } { 1 0 0 \gamma ^ { 2 } N ^ { 2 } } } \end{array}$ , proving (4.16).

For part (2), both $\mathcal { D } _ { 1 }$ and $\mathcal { D } _ { 2 }$ satisfy $\mathbb { E } [ x ^ { 2 } ] = 1 , \mathbb { E } [ x ^ { 4 } ] \leq 3 = \alpha$ , and $\begin{array} { r } { \mathbb { E } [ ( x ^ { 2 } - 1 ) ^ { 2 } ] = \kappa - 1 \ge 1 = \beta } \end{array}$ for $\kappa : = \mathbb { E } [ x ^ { 4 } ] \in \{ 2 , 3 \}$ . Since $y _ { t } = 0$ and $w _ { 0 } = 1$ , the scalar iterate satisfies $w _ { t } = ( 1 - \gamma x _ { t } ^ { 2 } ) w _ { t - 1 }$ with $\mathbb { E } [ 1 - \gamma x _ { t } ^ { 2 } ] = 1 - \gamma > 0$ and $\mathbb { E } [ ( 1 - \gamma x _ { t } ^ { 2 } ) ^ { 2 } ] = 1 - 2 \gamma + \kappa \gamma ^ { 2 } = r _ { \kappa } \in ( 0 , 1 )$ for $0 < \gamma < 1 / 3$ . By independence of $\{ x _ { t } \}$ , we have $\mathbb { E } [ w _ { t } ^ { 2 } ] = r _ { \kappa } ^ { t } \mathrm { ~ a n d ~ } \mathbb { E } [ w _ { t } w _ { t + h } ] = r _ { \kappa } ^ { t } ( 1 - \gamma ) ^ { h }$ for all $t , h \geq 0$ . Expanding $\begin{array} { r } { \overline { { w } } _ { s : s + N } ^ { 2 } = \frac { 1 } { N ^ { 2 } } \sum _ { t = s } ^ { s + N - 1 } w _ { t } ^ { 2 } + \frac { 2 } { N ^ { 2 } } \sum _ { t = s } ^ { \lfloor s + N - 1 } \sum _ { h = 1 } ^ { \lfloor s + N - 1 - t } w _ { t } w _ { t + h } \mathrm { ~ g i v e s } } \end{array}$ the exact risk

$$
R _ { \kappa } ( s , N ) = { \frac { 1 } { 2 } } \mathbb { E } \bigl [ \overline { { w } } _ { s : s + N } ^ { 2 } \bigr ] = { \frac { r _ { \kappa } ^ { s } } { 2 N ^ { 2 } } } \sum _ { j = 0 } ^ { N - 1 } r _ { \kappa } ^ { j } \left( 1 + 2 \sum _ { h = 1 } ^ { N - 1 - j } ( 1 - \gamma ) ^ { h } \right) .
$$

Retaining $j = 0$ and dropping $h \geq 1$ gives $\begin{array} { r } { R _ { \kappa } ( s , N ) \geq \frac { r _ { \kappa } ^ { s } } { 2 N ^ { 2 } } } \end{array}$ , while bounding $\textstyle 1 + 2 \sum _ { h = 1 } ^ { \infty } ( 1 - \gamma ) ^ { h } = { \frac { 2 - \gamma } { \gamma } }$ and $\begin{array} { r } { \sum _ { j = 0 } ^ { \infty } r _ { \kappa } ^ { j } = \frac { 1 } { 1 - r _ { \kappa } } } \end{array}$ gives $\begin{array} { r } { R _ { \kappa } ( s , N ) \leq \frac { ( 2 - \gamma ) r _ { \kappa } ^ { s } } { 2 \gamma ( 1 - r _ { \kappa } ) N ^ { 2 } } } \end{array}$ , establishing (4.17). □

Proof of Proposition $4 . 8 .$ . Applying the martingale-diference expansion of Lemma 5.1 to the shifted bias process $\bar { \{ \eta _ { s + k } ^ { \mathrm { b i a s } } \} } _ { k = 0 } ^ { N - 1 }$ initialized at $\eta _ { s } ^ { \mathrm { b i a s } }$ with $\mathbb { E } [ \pmb { \eta } _ { s } ^ { \mathrm { b i a s } } \otimes \pmb { \eta } _ { s } ^ { \mathrm { b i a s } } ] = \mathbf { B } _ { s }$ yields the conditional identity

$$
\mathbb { E } \big [ \bar { \eta } _ { s : s + N } ^ { \mathrm { b i a s } } \otimes \bar { \eta } _ { s : s + N } ^ { \mathrm { b i a s } } \big ] = \mathbf { P } _ { 0 , N } \mathbf { B } _ { s } \mathbf { P } _ { 0 , N } + \gamma ^ { 2 } \sum _ { k = 0 } ^ { N - 2 } \mathbf { Q } _ { k , 0 , N } \big ( ( \mathcal { M } - \widetilde { \mathcal { M } } ) \circ \mathbf { B } _ { s + k } \big ) \mathbf { Q } _ { k , 0 , N } .\tag{B.2}
$$

Let $a _ { i } ^ { ( s ) } : = \mathbf { v } _ { i } ^ { \top } \mathbf { B } _ { s } \mathbf { v } _ { i } \geq 0$ denote the i-th diagonal entry of $\mathbf { B } _ { s }$ in the eigenbasis of $\mathbf { H } ,$ and recall $\Xi _ { N }$ and $\Psi _ { r , s , N }$ from (A.1)–(A.2). Taking the trace inner product of (B.2) with $\scriptstyle { \frac { 1 } { 2 } } \mathbf { H }$ gives

$$
\mathrm { b i a s } = \frac { 1 } { 2 } \langle \mathbf { H } , \mathbf { P } _ { 0 , N } \mathbf { B } _ { s } \mathbf { P } _ { 0 , N } \rangle + \frac { \gamma ^ { 2 } } { 2 } \sum _ { k = 0 } ^ { N - 2 } \left. \mathbf { H } , \mathbf { Q } _ { k , 0 , N } \left( \left( \mathcal { M } - \widetilde { \mathcal { M } } \right) \circ \mathbf { B } _ { s + k } \right) \mathbf { Q } _ { k , 0 , N } \right. .
$$

For the first term, since $\begin{array} { r } { { \bf P } _ { 0 , N } { \bf v } _ { i } = \frac { 1 - ( 1 - \gamma \lambda _ { i } ) ^ { N } } { \gamma N \lambda _ { i } } { \bf v } _ { i } } \end{array}$ , applying (A.12) yields

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { 8 \gamma ^ { 2 } N ^ { 2 } } \mathrm { t r } \big ( \mathbf H _ { 0 ; k ^ { * } } ^ { - 1 } \mathbf B _ { s } \big ) + \frac 1 8 \mathrm { t r } \big ( \mathbf H _ { k ^ { * } ; \infty } \mathbf B _ { s } \big ) \leq \frac 1 2 \langle \mathbf H , \mathbf P _ { 0 , N } \mathbf B _ { s } \mathbf P _ { 0 , N } \rangle } \\ & { \qquad \leq \frac { 1 } { 2 \gamma ^ { 2 } N ^ { 2 } } \mathrm { t r } \big ( \mathbf H _ { 0 ; k ^ { * } } ^ { - 1 } \mathbf B _ { s } \big ) + \frac 1 2 \mathrm { t r } \big ( \mathbf H _ { k ^ { * } ; \infty } \mathbf B _ { s } \big ) . } \end{array}\tag{B.3}
$$

For the second term, note that $\gamma ^ { 2 } \langle \mathbf { H } , \mathbf { Q } _ { k , 0 , N } \mathbf { H } \mathbf { Q } _ { k , 0 , N } \rangle = \Psi _ { s + k , s , N } . ^ { 6 }$ In the upper bound, we have

$$
\begin{array} { r l } {  { \frac { \gamma ^ { 2 } } { 2 } \sum _ { k = 0 } ^ { N - 2 } \Big \langle \mathbf H , \mathbf Q _ { k , 0 , N } \big ( ( \mathcal M - \widetilde { \mathcal M } ) \circ \mathbf B _ { s + k } \big ) \mathbf Q _ { k , 0 , N } \Big \rangle } \quad } & { } \\ & { \leq \frac { \alpha } { 2 } \sum _ { k = 0 } ^ { N - 2 } \mathrm { t r } ( \mathbf H \mathbf B _ { s + k } ) \Psi _ { s + k , s , N } \leq \frac { \alpha \Xi _ { N } } { 4 \gamma N ( 1 - \gamma \alpha \mathrm { t r } ( \mathbf H ) ) } \mathrm { t r } ( \mathbf B _ { s , s + N } ) } \\ & { \leq \frac { \alpha } { 2 ( 1 - \gamma \alpha \mathrm { t r } ( \mathbf H ) ) } ( \frac { 1 } { \gamma N } \mathrm { t r } \big ( \mathbf I _ { 0 : k * } \mathbf B _ { s } \big ) + \mathrm { t r } \big ( \mathbf H _ { k ^ { * } : \infty } \mathbf B _ { s } \big ) ) \Xi _ { N } , } \end{array}
$$

where the first inequality applies Assumption 3.2, the second inequality uses $\Psi _ { s + k , s , N } \leq \Psi _ { s , s , N } \leq$ $\underline { { \underline { { 1 } } } } \underline { { \Xi } } _ { N }$ and Lemma C.7 on $[ s , s + N ]$ , and the third inequality uses $1 - ( 1 - \gamma \lambda _ { i } ) ^ { 2 N } \leq 1 \wedge ( 2 \gamma N \lambda _ { i } )$ Meanwhile, for the variance error in (5.7), Lemma C.8 and Assumption 3.2 imply $( { \mathcal { M } } - { \widetilde { \mathcal { M } } } ) \circ$ $\begin{array} { r } { \mathbf { C } _ { r } + \pmb { \Sigma } \preceq \frac { \sigma ^ { 2 } } { 1 - \gamma \alpha \operatorname { t r } ( \mathbf { H } ) } \mathbf { H } } \end{array}$ for all $r \geq 0$ , so variance $\begin{array} { r l } {  { \le \frac { \sigma ^ { 2 } } { 1 - \gamma \alpha \operatorname { t r } ( \mathbf { H } ) } \mathcal { V } _ { s , N } } } \end{array}$ . Combining these bounds with $\mathbb { E } [ L ( \overline { { \mathbf { w } } } _ { s : s + N } ) ] - L ( \mathbf { w } ^ { * } ) \leq 2 \cdot \mathrm { b i a s } + 2$ · variance and (A.10) yields (4.18).

In the lower bound $( N \ge 2$ and $\gamma \leq 1 / ( 2 \lambda _ { 1 } ) )$ , let $m : = \lfloor N / 2 \rfloor \geq 1$ and $\mathbf { G } : = \mathbf { I } - \gamma \mathbf { H }$ . Restricting the sum over k to $0 \leq k \leq m - 1$ , we have $\Psi _ { s + k , s , N } ~ \ge ~ \frac { 1 } { 3 6 N } \Xi _ { N }$ from Part 3 of the proof of Lemma A.1, while Assumption 3.4 gives $( \mathcal { M } - \widetilde { \mathcal { M } } ) \circ \mathbf { B } _ { s + k } \succeq \beta \mathrm { t r } ( \mathbf { H } \mathbf { B } _ { s + k } ) \mathbf { H }$ . Summing $\gamma \operatorname { t r } ( \mathbf { H } \mathbf { B } _ { s + k } )$ over $0 \leq k \leq m - 1$ gives

$$
\begin{array} { r l } { \gamma \displaystyle \sum _ { k = 0 } ^ { m - 1 } \mathrm { t r } ( \mathbf H \mathbf B _ { s + k } ) \geq \gamma \displaystyle \sum _ { k = 0 } ^ { m - 1 } \mathrm { t r } ( \mathbf H \mathbf G ^ { 2 k } \mathbf B _ { s } ) = \sum _ { i = 1 } ^ { \infty } \frac { 1 - ( 1 - \gamma \lambda _ { i } ) ^ { 2 m } } { 2 - \gamma \lambda _ { i } } a _ { i } ^ { ( s ) } } & { } \\ { \geq \displaystyle \frac 1 2 \mathrm { t r } ( \mathbf B _ { s } - \mathbf G ^ { 2 m } \mathbf B _ { s } ) \geq \frac 1 6 \mathrm { t r } ( \mathbf B _ { s , s + N } ) } & { } \\ { \geq \displaystyle \frac 1 { 1 2 } \Big ( \mathrm { t r } \big ( \mathbf I _ { 0 : k * } \mathbf B _ { s } \big ) + N \gamma \mathrm { t r } \big ( \mathbf H _ { k ^ { * } : \infty } \mathbf B _ { s } \big ) \Big ) , } \end{array}
$$

where the first inequality uses $\mathbf { B } _ { s + k } = ( \mathcal { T } - \gamma T ) ^ { k } \circ \mathbf { B } _ { s } \succeq \mathbf { G } ^ { k } \mathbf { B } _ { s } \mathbf { G } ^ { k }$ from Lemma $\mathrm { C . 5 ( 3 ) }$ , the second inequality uses $2 - \gamma \lambda _ { i } \leq 2$ , the third inequality applies Lemma C.2, and the fourth inequality uses $\begin{array} { r } { 1 - ( 1 - \gamma \lambda _ { i } ) ^ { 2 N } \geq \frac { 1 } { 2 } ( 1 \wedge \gamma N \lambda _ { i } ) } \end{array}$ . Multiplying this trace bound by $\begin{array} { r } { \frac { \beta } { 2 \gamma } \cdot \frac { 1 } { 3 6 N } \Xi _ { N } = \frac { \beta } { 7 2 \gamma N } \Xi _ { N } } \end{array}$ gives $\begin{array} { r } { \frac { \beta } { 8 6 4 \gamma N } \left( \operatorname { t r } ( \mathbf { I } _ { 0 : k ^ { * } } \mathbf { B } _ { s } ) + N \gamma \operatorname { t r } ( \mathbf { H } _ { k ^ { * } : \infty } \mathbf { B } _ { s } ) \right) \Xi _ { N } } \end{array}$ . Combining this with (B.3), variance $\ge \sigma _ { \mathrm { n o i s e } } ^ { 2 } \mathcal { V } _ { s , N }$ , and (A.10) establishes (4.19). □

## C Auxiliary Lemmas

In this appendix, we collect the general-purpose elementary algebraic and convolution lemmas (Section C.1) and the foundational operator lemmas from Jain et al. (2017a,b) and Zou et al. (2021) (Section C.2) invoked in Appendices $\mathrm { A - B }$

## C.1 Elementary Algebraic and Convolution Lemmas

Lemma C.1 (Chebyshev’s Sum Inequality for Weighted Monotone Sequences). Let $\{ w _ { i } \} _ { i \ge 1 }$ be non-negative weights with $\begin{array} { r } { W : = \sum _ { i = 1 } ^ { \infty } w _ { i } < \infty , } \end{array}$ and let $\{ x _ { i } \} _ { i \geq 1 }$ and $\{ y _ { i } \} _ { i \geq 1 }$ be bounded non-negative sequences that are both non-decreasing in i. Then

$$
\left( \sum _ { i = 1 } ^ { \infty } w _ { i } x _ { i } \right) \left( \sum _ { i = 1 } ^ { \infty } w _ { i } y _ { i } \right) \leq W \sum _ { i = 1 } ^ { \infty } w _ { i } x _ { i } y _ { i } .\tag{C.1}
$$

In particular, under Assumption 3.1 with $0 < \gamma < 1 / \lambda _ { 1 }$ , the propagation kernel $\begin{array} { r } { k ( h ) = \sum _ { i \geq 1 } \lambda _ { i } ^ { 2 } ( 1 - } \end{array}$ $\gamma \lambda _ { i } ) ^ { 2 h }$ satisfies $k ( u ) k ( v ) \leq k ( 0 ) k ( u + v ) = \mathrm { t r } ( { \bf H } ^ { 2 } ) k ( u + v )$ for all $u , v \geq 0$

Proof of Lemma C.1. Since $\{ x _ { i } \}$ and $\{ y _ { i } \}$ are both non-decreasing, $( x _ { i } - x _ { j } ) ( y _ { i } - y _ { j } ) \geq 0$ for all $i , j \geq 1$ . Multiplying by $w _ { i } w _ { j } \geq 0$ and summing over all $i , j \geq 1$ gives

$$
0 \leq \frac { 1 } { 2 } \sum _ { i = 1 } ^ { \infty } \sum _ { j = 1 } ^ { \infty } w _ { i } w _ { j } ( x _ { i } - x _ { j } ) ( y _ { i } - y _ { j } ) = W \sum _ { i = 1 } ^ { \infty } w _ { i } x _ { i } y _ { i } - \left( \sum _ { i = 1 } ^ { \infty } w _ { i } x _ { i } \right) \left( \sum _ { j = 1 } ^ { \infty } w _ { j } y _ { j } \right) ,
$$

which proves (C.1). Taking $w _ { i } = \lambda _ { i } ^ { 2 } , x _ { i } = ( 1 - \gamma \lambda _ { i } ) ^ { 2 u }$ , and $y _ { i } = ( 1 - \gamma \lambda _ { i } ) ^ { 2 v }$ (which are non-decreasing in i since $\lambda _ { 1 } \ge \lambda _ { 2 } \ge \cdots > 0 )$ yields $k ( u ) k ( v ) \leq \mathrm { t r } ( \mathbf { H } ^ { 2 } ) k ( u + v )$ □

Lemma C.2 (Polynomial Exponent-Halving Sandwich). For every $x \in [ 0 , 1 ]$ and integer $N \geq 2$ with $m : = \lfloor N / 2 \rfloor \geq 1$ , it holds that $1 - x ^ { m } \leq 1 - x ^ { N } \leq 3 ( 1 - x ^ { m } )$ . Consequently, for any PSD operator H with $\| \gamma \mathbf { H } \| _ { 2 } \le 1$ and any PSD operator $\mathbf { A } \succeq \mathbf { 0 }$

$$
\frac { 1 } { 3 } \operatorname { t r } \Big ( \big ( \mathbf { I } - ( \mathbf { I } - \gamma \mathbf { H } ) ^ { 2 N } \big ) \mathbf { A } \Big ) \leq \operatorname { t r } \Big ( \big ( \mathbf { I } - ( \mathbf { I } - \gamma \mathbf { H } ) ^ { 2 \lfloor N / 2 \rfloor } \big ) \mathbf { A } \Big ) \leq \operatorname { t r } \Big ( \big ( \mathbf { I } - ( \mathbf { I } - \gamma \mathbf { H } ) ^ { 2 N } \big ) \mathbf { A } \Big ) .\tag{C.2}
$$

Proof of Lemma C.2. Since $m \ = \ \lfloor N / 2 \rfloor \ \geq \ 1$ satisfies $m \le N \le$ 3m and $x \in [ 0 , 1 ]$ , we have $1 - x ^ { m } \leq 1 - x ^ { N } \leq 1 - x ^ { 3 m } = ( 1 - \bar { x ^ { m } } ) ( 1 + x ^ { m } + x ^ { 2 m } ) \leq 3 ( 1 - x ^ { m } )$ . Applying this to each eigenvalue $x = ( 1 - \gamma \lambda _ { i } ) ^ { 2 } \in [ 0 , 1 ]$ yields (C.2). □

Lemma C.3 (Exact Two-Sided Bound for Finite Geometric Convolutions). For any scalars $p , q \in ( 0 , 1 ]$ and any integer $s \geq 1$ , let $M : = p \vee q$ and m $: = p \wedge q$ . Then

$$
( 1 - e ^ { - 1 } ) M ^ { s - 1 } \operatorname* { m i n } \left\{ s , { \frac { M } { M - m } } \right\} \leq \sum _ { r = 0 } ^ { s - 1 } p ^ { r } q ^ { s - 1 - r } \leq M ^ { s - 1 } \operatorname* { m i n } \left\{ s , { \frac { M } { M - m } } \right\} ,\tag{C.3}
$$

where $M / ( M - m )$ is interpreted as +∞ when $p = q$

Proof of Lemma C.3. By symmetry, assume $p \leq q .$ , so $M = q , m = p ,$ and set $u : = p / q \in ( 0 , 1 ]$ . Then $\begin{array} { r } { \sum _ { r = 0 } ^ { s - 1 } \dot { p ^ { r } } q ^ { s - 1 - r } = q ^ { s - 1 } \dot { \sum _ { r = 0 } ^ { s - 1 } } u ^ { r } } \end{array}$ . Since $u \in ( 0 , 1 ]$ , we have $\begin{array} { r } { \sum _ { r = 0 } ^ { s - 1 } u ^ { r } \le s \wedge ( 1 - u ) ^ { - 1 } = } \end{array}$ min $\{ s , q / ( q - p ) \}$ Conversely, if $1 - u \leq 1 / s$ , then $u \geq 1 - 1 / s$ gives $\begin{array} { r } { \sum _ { r = 0 } ^ { s - 1 } u ^ { r } \ge s ( 1 - ( 1 - 1 / s ) ^ { s } ) \ge ( 1 - e ^ { - 1 } ) s ; } \end{array}$ if $1 - u > 1 / s$ , then $u ^ { s } < ( 1 - 1 / s ) ^ { s } < e ^ { - 1 }$ gives $\begin{array} { r } { \sum _ { r = 0 } ^ { s - 1 } u ^ { r } = ( 1 - u ^ { s } ) / ( 1 - u ) \ge ( 1 - e ^ { - 1 } ) q / ( q - p ) } \end{array}$ □

## C.2 Operator Lemmas from Prior Work

Lemma C.4 (Sub-Gaussian Implies Fourth-Moment Upper Bound; Zou et al. 2021, Lemma A.1; Hsu et al. 2014; Vershynin 2018, Proposition 2.5.2). Suppose $\mathbb { E } [ { \mathbf { x } } { \mathbf { x } } ^ { \top } ] = \mathbf { H } \succ \mathbf { 0 }$ and the standardized feature projections are $\sigma _ { z } ^ { 2 } { - } s u b { - } G a u s s i a n ,$ , in the sense that $\mathbb { E } [ \exp ( \langle \mathbf { \bar { u } } , \mathbf { x } \rangle ) ] \leq \exp ( \sigma _ { z } ^ { 2 } \| \mathbf { u } \| _ { \mathbf { H } } ^ { 2 } / 2 )$ for all $\mathbf { u } \in \mathcal { H }$ (equivalently, $\mathbb { E } [ \exp ( \lambda \langle \mathbf { v } , \mathbf { H } ^ { - 1 / 2 } \mathbf { P } _ { m } \mathbf { x } \rangle ) ] \leq \exp ( \lambda ^ { 2 } \sigma _ { z } ^ { 2 } / 2 )$ for all finite-rank spectral projections $\mathbf { P } _ { m } = \mathbf { I } _ { 0 : m }$ , unit vectors $\mathbf { v } \in \mathcal { H }$ , and $\lambda \in \mathbb { R } )$ . Then for any PSD operator $\mathbf { A } \succeq \mathbf { 0 }$ , Assumption 3.2 holds with $\alpha = 1 6 \sigma _ { z } ^ { 4 }$ :

$$
\mathcal { M } \circ \mathbf { A } = \mathbb { E } \big [ ( \mathbf { x } ^ { \top } \mathbf { A } \mathbf { x } ) \mathbf { x } \mathbf { x } ^ { \top } \big ] \preceq 1 6 \sigma _ { z } ^ { 4 } \operatorname { t r } ( \mathbf { H } \mathbf { A } ) \mathbf { H } .
$$

Lemma C.5 (Properties of PSD Super-Operators; Zou et al. 2021, Lemmas 4.1 and B.1; Jain et al. $^ { 2 0 1 7 \mathrm { a } , \mathrm { b } ) }$ . A linear super-operator O on symmetric operators is called a PSD mapping if $\mathbf { A } \succeq \mathbf { 0 }$ implies $\mathcal { O } \circ \mathbf { A } \succeq \mathbf { 0 }$ . Under Assumptions 3.1 and 3.2, the super-operators defined in (3.3) and (5.1) are well-defined on symmetric trace-class operators and satisfy:

1. M and $\widetilde { \mathcal { M } }$ are both PSD mappings.

2. $\mathcal { T } - \gamma \mathcal { T }$ and $\mathcal { I } - \gamma \widetilde { \mathcal { T } }$ are both PSD mappings for all $\gamma > 0$

3. $M - { \widetilde { \mathcal { M } } }$ and $\widetilde { \mathcal { T } } - \mathcal { T } = \gamma ( \mathcal { M } - \widetilde { \mathcal { M } } )$ are both PSD mappings; in particular, $\left( { \mathcal { T } } - \gamma { \mathcal { T } } \right) \circ \mathbf { A } \succeq$ $( \mathcal { T } - \gamma \widetilde { \mathcal { T } } ) \circ \mathbf { A } = ( \mathbf { I } - \gamma \mathbf { H } ) \mathbf { A } ( \mathbf { I } - \gamma \mathbf { H } )$ for all $\mathbf { A } \succeq \mathbf { 0 }$

Lemma C.6 (Bias-Variance Risk Decomposition; Zou et al. 2021, Lemmas B.2, C.1, D.1, and D.6). Consider SGD with tail-averaging (3.2) and recall the definitions of bias and variance in (3.7). Under Assumptions 3.1, 3.2, and 3.3 (which ensure $\mathbb { E } \Vert \pmb { \eta } _ { t } ^ { \mathrm { b i a s } } \Vert _ { \mathbf { H } } ^ { 2 } < \infty$ and $\mathbb { E } \Vert \pmb { \eta } _ { t } ^ { \mathrm { v a r i a n c e } } \Vert _ { \mathbf { H } } ^ { 2 } < \infty$ for all $t \geq 0 )$

1. For any data distribution D, the expected excess risk satisfies

$$
\begin{array} { r l r } {  { \mathbb { E } \big [ L ( \overline { { \mathbf { w } } } _ { s : s + N } ) \big ] - L ( \mathbf { w } ^ { * } ) = \frac { 1 } { 2 } \big \langle \mathbf { H } , \mathbb { E } \big [ \overline { { \eta } } _ { s : s + N } \otimes \overline { { \eta } } _ { s : s + N } \big ] \big \rangle } } \\ & { } & { \leq \big ( \sqrt { \mathrm { b i a s } } + \sqrt { \mathrm { v a r i a n c e } } \big ) ^ { 2 } \leq 2 \cdot \mathrm { b i a s } + 2 \cdot \mathrm { v a r i a n c e } . } \end{array}
$$

2. If the residual noise $\xi _ { t } = y _ { t } - \left. \mathbf { w } ^ { * } , \mathbf { x } _ { t } \right.$ is well-specified as in (3.4) (so $\mathbb { E } [ \xi _ { t } \mid { \bf x } _ { t } ] = 0 )$ , then $\mathbb { E } [ \bar { \pmb { \eta } } _ { s : s + N } ^ { \mathrm { b i a s } } \otimes \bar { \pmb { \eta } } _ { s : s + N } ^ { \mathrm { v a r i a n c e } } ] = { \bf { 0 } }$ and the excess risk decomposes with exact equality:

$$
\mathbb { E } \big [ L ( \overline { { \mathbf { w } } } _ { s : s + N } ) \big ] - L ( \mathbf { w } ^ { * } ) = \mathrm { b i a s + v a r i a n c e } .
$$

Lemma C.7 (Single-Step Trace Identity and Interval Telescoping Inequality; adapted from Zou et al. 2021, Eqs. (B.1) and (D.4)). Under Assumptions 3.1 and 3.2, $i f \gamma < 1 / ( \alpha \mathrm { t r } ( \mathbf { H } ) )$ , then for any integers $0 \leq a < b$ and $\mathbf { B } _ { a , b } : = \mathbf { B } _ { a } - ( \mathbf { I } - \gamma \mathbf { H } ) ^ { b - a } \mathbf { B } _ { a } ( \mathbf { I } - \gamma \mathbf { H } ) ^ { b - a }$ , the bias covariance sequence $\{ \mathbf { B } _ { u } \} _ { u \geq 0 }$ satisfies

$$
\gamma \sum _ { u = a } ^ { b - 1 } \mathrm { t r } ( \mathbf H \mathbf B _ { u } ) \leq \frac { \mathrm { t r } ( \mathbf B _ { a } ) - \mathrm { t r } ( \mathbf B _ { b } ) } { 2 - \gamma \alpha \mathrm { t r } ( \mathbf H ) } \leq \frac { 1 } { 2 \big ( 1 - \gamma \alpha \mathrm { t r } ( \mathbf H ) \big ) } \mathrm { t r } ( \mathbf B _ { a , b } ) .\tag{C.4}
$$

Proof of Lemma C.7. We adapt the telescoping argument of Zou et al. (2021, Eqs. (B.1) and (D.4)) to a general interval $[ a , b ]$ for completeness. For each step $u \geq 0$ , taking the trace of $\mathbf { B } _ { u + 1 } = ( { \mathcal { T } } - { \mathcal { \gamma T } } ) \circ$ ${ \bf B } _ { u } = { \bf B } _ { u } - \gamma ( { \bf H } { \bf B } _ { u } + { \bf B } _ { u } { \bf H } ) + \gamma ^ { 2 } \mathcal { M } \circ { \bf B } _ { u }$ and applying Assumption $3 . 2 \left( \operatorname { t r } ( \mathcal { M } \circ \mathbf { B } _ { u } ) \leq \alpha \operatorname { t r } ( \mathbf { H } ) \operatorname { t r } ( \mathbf { H } \mathbf { B } _ { u } ) \right)$ gives

$$
\mathrm { t r } ( \mathbf { B } _ { u + 1 } ) \leq \mathrm { t r } ( \mathbf { B } _ { u } ) - \gamma \big ( 2 - \gamma \alpha \mathrm { t r } ( \mathbf { H } ) \big ) \mathrm { t r } ( \mathbf { H } \mathbf { B } _ { u } ) .
$$

Summing this inequality over $u \in \{ a , \ldots , b - 1 \}$ and dividing by $2 - \gamma \alpha \operatorname { t r } ( \mathbf { H } ) > 2 ( 1 - \gamma \alpha \operatorname { t r } ( \mathbf { H } ) ) > 0$ yields the first inequality in (C.4). Finally, since $\mathbf { B } _ { b } = ( { \mathcal { T } } - \gamma T ) ^ { b - a } \circ \mathbf { B } _ { a } \succeq ( \mathbf { I } - \gamma \mathbf { H } ) ^ { b - a } \mathbf { B } _ { a } ( \mathbf { I } - \gamma \mathbf { H } ) ^ { b - a }$ by Lemma C.5(3), we have t $\operatorname { r } ( \mathbf { B } _ { a } ) - \operatorname { t r } ( \mathbf { B } _ { b } ) \leq \operatorname { t r } ( \mathbf { B } _ { a , b } )$ , establishing the second inequality in (C.4).

Lemma C.8 (Upper Bounds on Constant-Stepsize Variance Iterates; Zou et al. 2021, Lemmas B.4 and B.5). Under Assumptions ${ \mathcal { B } } . 1 , \ { \mathcal { B } } . { \mathcal { Q } } ,$ and 3.3, if the constant stepsize satisfies $0 < \gamma < 1 / ( \alpha \mathrm { t r } ( \mathbf { H } ) )$ then the variance covariance sequence $\{ \mathbf { C } _ { t } \} _ { t \ge 0 }$ defined in (5.3) satisfies

$$
\mathbf { C } _ { t } \preceq \frac { \gamma \sigma ^ { 2 } } { 1 - \gamma \alpha \operatorname { t r } ( \mathbf { H } ) } \big ( \mathbf { I } - ( \mathbf { I } - \gamma \mathbf { H } ) ^ { t } \big ) \preceq \frac { \gamma \sigma ^ { 2 } } { 1 - \gamma \alpha \operatorname { t r } ( \mathbf { H } ) } \mathbf { I } , \qquad \forall t \geq 0 .
$$

## References

Bach, F. and Moulines, E. (2013). Non-strongly-convex smooth stochastic approximation with convergence rate O(1/n). In Advances in Neural Information Processing Systems (NeurIPS), vol. 26.

Bartlett, P. L., Long, P. M., Lugosi, G. and Tsigler, A. (2020). Benign overfitting in linear regression. Proceedings of the National Academy of Sciences 117 30063–30070.

Belkin, M., Hsu, D. and Xu, J. (2020). Two models of double descent for weak features. SIAM Journal on Mathematics of Data Science 2 1167–1180.

Berthier, R., Bach, F. and Gaillard, P. (2020). Tight nonparametric convergence rates for stochastic gradient descent under the noiseless linear model. In Advances in Neural Information Processing Systems (NeurIPS), vol. 33.

Caponnetto, A. and De Vito, E. (2007). Optimal rates for the regularized least-squares algorithm. Foundations of Computational Mathematics 7 331–368.

Chatterji, N. S. and Long, P. M. (2021). Finite-sample analysis of interpolating linear classifiers in the overparameterized regime. Journal of Machine Learning Research 22 1–30.

Defossez, A.<sup>´</sup> and Bach, F. (2015). Averaged least-mean-squares: Bias-variance trade-ofs and optimal sampling distributions. In Proceedings of the 18th International Conference on Artificial Intelligence and Statistics (AISTATS).

Dieuleveut, A. and Bach, F. (2016). Nonparametric stochastic approximation with large step-sizes. The Annals of Statistics 44 1363–1399.

Dieuleveut, A., Flammarion, N. and Bach, F. (2017). Harder, better, faster, stronger convergence rates for least-squares regression. Journal of Machine Learning Research 18 1–51.

Ge, R., Kakade, S. M., Kidambi, R. and Netrapalli, P. (2019). The step decay schedule: A near optimal, geometrically decaying learning rate procedure for least squares. In Advances in Neural Information Processing Systems (NeurIPS), vol. 32.

Hastie, T., Montanari, A., Rosset, S. and Tibshirani, R. J. (2022). Surprises in highdimensional ridgeless least squares interpolation. The Annals of Statistics 50 949–986.

Hsu, D., Kakade, S. M. and Zhang, T. (2014). Random design analysis of ridge regression. Foundations of Computational Mathematics 14 569–600.

Jain, P., Kakade, S. M., Kidambi, R., Netrapalli, P., Pillutla, V. K. and Sidford, A. (2017a). A Markov chain theory approach to characterizing the minimax optimality of stochastic gradient descent (for least squares). arXiv preprint arXiv:1710.09430 .

Jain, P., Kakade, S. M., Kidambi, R., Netrapalli, P. and Sidford, A. (2018). Accelerating stochastic gradient descent for least squares regression. In Proceedings of the 31st Conference on Learning Theory (COLT). PMLR.

Jain, P., Netrapalli, P., Kakade, S. M., Kidambi, R. and Sidford, A. (2017b). Parallelizing stochastic gradient descent for least squares regression: Mini-batching, averaging, and model misspecification. Journal of Machine Learning Research 18 1–42.

Kaddour, J. (2022). Stop wasting my time! Saving days of ImageNet and BERT training with latest weight averaging. arXiv preprint arXiv:2209.14981 .

Lakshminarayanan, C. and Szepesvari, C. <sup>´</sup> (2018). Linear stochastic approximation: How far does constant step-size and iterate averaging go? In Proceedings of the 21st International Conference on Artificial Intelligence and Statistics (AISTATS). PMLR.

Li, B., Chen, F., Huang, Z., Wang, L. and Wu, L. (2025). Functional scaling laws in kernel regression: Loss dynamics and learning rate schedules. In Advances in Neural Information Processing Systems (NeurIPS), vol. 38. ArXiv preprint arXiv:2509.19189.

Li, X., Deng, Y., Wu, J., Zhou, D. and Gu, Q. (2024). Risk bounds of accelerated SGD for overparameterized linear regression. In International Conference on Learning Representations (ICLR). ArXiv preprint arXiv:2311.14222.

Li, X. and Gu, Q. (2025). Understanding SGD with exponential moving average: A case study in linear regression. arXiv preprint arXiv:2502.14123 .

Lin, L., Wu, J., Kakade, S. M., Bartlett, P. L. and Lee, J. D. (2024). Scaling laws in linear regression: Compute, parameters, and data. In Advances in Neural Information Processing Systems (NeurIPS), vol. 37. ArXiv preprint arXiv:2406.08466.

Meterez, A., Morwani, D., Oncescu, C.-A., Wu, J., Pehlevan, C. and Kakade, S. M. (2025). A simplified analysis of SGD for linear regression with weight averaging. arXiv preprint arXiv:2506.15535 .

Meterez, A., Nair, P. A., Morwani, D., Pehlevan, C. and Kakade, S. M. (2026). Anytime pretraining: Horizon-free learning-rate schedules with weight averaging. arXiv preprint arXiv:2602.03702 .

Muthukumar, V., Narang, A., Subramanian, V., Belkin, M., Hsu, D. and Sahai, A. (2020). Harmless interpolation of noisy data in regression. IEEE Journal on Selected Areas in Information Theory 1 67–83.

Nemirovski, A., Juditsky, A., Lan, G. and Shapiro, A. (2009). Robust stochastic approximation approach to stochastic programming. SIAM Journal on Optimization 19 1574–1609.

Paquette, C., Lee, K., Pedregosa, F. and Paquette, E. (2021). SGD in the large: Averagecase analysis, asymptotics, and stepsize criticality. In Proceedings of the 34th Conference on Learning Theory (COLT). PMLR.

Paquette, C., Paquette, E., Adlam, B. and Pennington, J. (2022). Implicit regularization or implicit conditioning? Exact risk trajectories of SGD in high dimensions. In Advances in Neural Information Processing Systems (NeurIPS), vol. 35.

Polyak, B. T. and Juditsky, A. B. (1992). Acceleration of stochastic approximation by averaging. SIAM Journal on Control and Optimization 30 838–855.

Rakhlin, A., Shamir, O. and Sridharan, K. (2012). Making gradient descent optimal for strongly convex stochastic optimization. In Proceedings of the 29th International Conference on Machine Learning (ICML).

Robbins, H. and Monro, S. (1951). A stochastic approximation method. The Annals of Mathematical Statistics 22 400–407.

Ruppert, D. (1988). Eficient estimations from a slowly convergent Robbins–Monro process. Technical Report 781, Cornell University Operations Research and Industrial Engineering.

Sanyal, S., Neerkaje, A., Kaddour, J., Kumar, A. and Sanghavi, S. (2023). Early weight averaging meets high learning rates for LLM pre-training. arXiv preprint arXiv:2306.03241 .

Tsigler, A. and Bartlett, P. L. (2023). Benign overfitting in ridge regression. Journal of Machine Learning Research 24 1–76.

Vershynin, R. (2018). High-Dimensional Probability: An Introduction with Applications in Data Science. Cambridge University Press.

Wu, J., Bartlett, P. L., Kakade, S. M., Lee, J. D. and Yu, B. (2025). Risk comparisons in linear regression: Implicit regularization dominates explicit regularization. arXiv preprint arXiv:2509.17251 .

Wu, J., Zou, D., Braverman, V., Gu, Q. and Kakade, S. M. (2022). Last iterate risk bounds of SGD with decaying stepsize for overparameterized linear regression. In Proceedings of the 39th International Conference on Machine Learning (ICML). PMLR.

Zou, D., Wu, J., Braverman, V., Gu, Q. and Kakade, S. M. (2021). Benign overfitting of constant-stepsize SGD for linear regression. In Proceedings of the 34th Conference on Learning Theory (COLT). PMLR. Full version: arXiv preprint arXiv:2103.12692.