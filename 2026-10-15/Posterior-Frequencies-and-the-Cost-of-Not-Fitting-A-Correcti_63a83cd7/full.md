# Posterior Frequencies and the Cost of Not Fitting: A Correction to a Memorization Lower Bound

Zhenisbek Assylbekov Department of Mathematical Sciences Purdue University Fort Wayne

Rustem Takhanov   
School of Computing and Artificial Intelligence   
Nazarbayev University

ZASSYLBE@PFW.EDU

RUSTEM.TAKHANOV@NU.EDU.KZ

## Abstract

Feldman’s memorization lower bound assigns a frequency coefficient based on how often a point appears in the training sample. We give an exact three-point counterexample to its conditional and averaged forms, using independent fair labels and a permutation-equivariant learner. A sequence of sampling priors makes the ratio of the proposed penalty to actual excess risk grow exponentially, ruling out a universal positive constant-factor repair. The priors vary with sample size, the offending profiles become rare, and both quantities vanish. Retaining the original model, we obtain an exact replacement using full-sample posterior probabilities. Under exchangeability, these depend on multiplicity and the complete frequency profile. We characterize the largest multiplicity-only coefficients valid for unrestricted learners. The original coefficient has the posterior-mean form of Good’s exact frequency formula and is an occupancy-weighted average of the corrected coefficients.

Keywords: memorization, posterior prediction, Good–Turing, exchangeability, frequency profile

## 1. Introduction

If a learner misclassifies an observed noiseless label, it incurs population error whenever the corresponding input is encountered again. In a model with an unknown sampling distribution, the condi tional cost is therefore the input’s posterior probability given the training sample. Feldman (2021) uses this observation to relate fitting rare examples to generalization. His Theorem 2.3 assigns the same coefficient to every point observed ℓ times in a sample of size n:

$$
\tau _ { n , \ell } = \frac { \mathbb { E } [ Q ^ { \ell + 1 } ( 1 - Q ) ^ { n - \ell } ] } { \mathbb { E } [ Q ^ { \ell } ( 1 - Q ) ^ { n - \ell } ] } ,\tag{1}
$$

where Q is the random probability of a fixed domain point. This is the posterior mean of $Q ,$ conditional only on the number of times the fixed point appears in the sample. It has the same functional form as Good (1953, formula (13), with m = 1).

The learner, however, observes all sample counts. These may reveal additional information about Q. In Feldman’s model, raw weights are drawn independently and then normalized; independence before normalization does not justify ignoring the other counts.

Feldman’s work provides an influential theoretical account of why learning from long-tailed distributions can require memorizing rare examples. Theorem 2.3 is a central step in that account: it connects errors on observed examples to excess population risk through frequency-dependent coefficients. Establishing the precise validity of this connection is therefore important beyond the particular conditioning identity used in its proof. We identify the failed conditioning step, give an exact counterexample to the theorem itself, and recover a valid lower bound within the original model.

The contributions are threefold. First, Section 3 gives an exact three-point counterexample, with independent fair labels and a permutation-equivariant learner. Second, Section 4 shows that the failure can be exponentially large in relative terms: along even sample sizes $n ,$ a sequence of three-entry frequency lists gives

$$
{ \frac { \mathrm { p r o p o s e d a v e r a g e d p e n a l t y } } { \mathrm { a c t u a l e x c e s s ~ r i s k } } } \sim { \frac { 2 } { 1 5 } } \left( { \frac { 4 } { 3 } } \right) ^ { n } .
$$

Thus no universal positive constant multiple of the original penalty is valid. The prior changes with n, the learner errs on an increasingly rare count profile, and both the penalty and excess risk tend to zero. This is a worst-case multiplicative obstruction, not a fixed-prior asymptotic result or a nonvanishing additive gap.

Third, Sections 5–6 give the exact replacement and its sharp multiplicity-only relaxation. Under a permutation-invariant prior, the full-sample posterior depends only on the point’s multiplicity and the frequency profile: the numbers of domain points appearing zero times, once, twice, and so forth. A multiplicity-only coefficient is valid for all learners exactly when it does not exceed the smallest feasible profile coefficient. Feldman’s $\tau _ { n , \ell }$ instead averages these coefficients with occupancy weights, allowing learners to select errors on profiles where the average is too large.

Related work and scope. The posterior decomposition and exchangeability arguments are standard; their role here is to identify the exact correction. The main contributions are the counterexample, its exponential relative amplification, and the sharp criterion for universal learner-dependent bounds. Count-only prediction is closely related to Johnson’s sufficientness postulate: under suit able assumptions across sample sizes, it leads to classical Dirichlet characterizations (Zabell, 1982). Our criterion concerns one fixed sample size and observed categories; it is not a characterization of Dirichlet priors. Section 6.3 makes this distinction precise. Lee et al. (2013) characterize when predictive rules based on species counts are compatible with an exchangeable sampling model. Our question is different: given a sampling model, which posterior probabilities yield valid lower bounds on a learner’s excess risk? Appendix C gives a finite-dimensional Gamma-integral calculation related to posterior analysis of normalized random measures (James et al., 2009); it is a computational aid, not a new posterior method.

We correct Theorem 2.3 of arXiv:1906.05271v4, dated January 10, 2021, while retaining its normalized-weight model. Section 7 identifies specific downstream dependencies and separates them from calculations that do not use the failed conditioning identity. Fixed-prior and typicalprofile approximations, including long-tail guarantees, require additional analysis.

## 2. Model and posterior probabilities

Let $\mathcal { X } = [ N ] = \{ 1 , . . . , N \}$ be a finite domain and let Y be a finite label set with at least two elements. A prior Π is a probability distribution on the simplex

$$
\begin{array} { r } { \Delta _ { N } = \{ d \in [ 0 , 1 ] ^ { N } : \sum _ { x } d _ { x } = 1 \} . } \end{array}
$$

Independently draw a probability vector $D \sim \Pi$ and a labeling function $f \sim \mathcal { F }$ , where $\mathcal { F }$ is any distribution on $y x$ . Independence here is between D and $f ;$ the labels $f ( x )$ may be dependent across points. Conditional on $( D , f )$ , draw $U = ( X _ { 1 } , \ldots , X _ { n } ) \sim D ^ { n }$ and observe

$$
S = ( ( X _ { 1 } , f ( X _ { 1 } ) ) , \ldots , ( X _ { n } , f ( X _ { n } ) ) ) .
$$

All claims conditional on a dataset concern datasets of positive probability. We assume $n \geq 1$ . A learner $A$ maps each dataset to a distribution on $y x$ ; its randomization is independent of the latent variables conditional on the dataset. No computational or hypothesis-class restriction is imposed.

For a realized sample $Z ,$ define the counts and the observed set by

$$
c _ { x } = c _ { x } ( Z ) = \sum _ { t = 1 } ^ { n } \mathbf { 1 } \{ X _ { t } = x \} , \qquad O ( Z ) = \{ x : c _ { x } > 0 \} .
$$

Noiselessness makes the label $y z ( x )$ well-defined for every $x \in O ( Z )$ . Let

$$
L _ { c } ( d ) = \prod _ { x \in \mathcal { X } } d _ { x } ^ { c _ { x } } , \qquad 0 ^ { 0 } : = 1 .
$$

Define $q _ { x } ( Z )$ as the posterior mean of $D _ { x }$ . As shown in Lemma 1 below, it satisfies

$$
q _ { x } ( Z ) : = \mathbb { E } [ D _ { x } \mid S = Z ] = \frac { \mathbb { E } _ { D \sim \Pi } [ D _ { x } L _ { c } ( D ) ] } { \mathbb { E } _ { D \sim \Pi } [ L _ { c } ( D ) ] } .\tag{2}
$$

We also write $q _ { x } ( c )$ because this quantity depends only on the input counts.

Lemma 1 (Posterior factorization) Conditional on $S = Z ,$ , the random variables D and f are independent. The posterior of D depends only on c, and the posterior of f is its prior conditioned on agreeing with all observed labels. In particular, (2) holds

Proof For a fixed ordered dataset, its likelihood given $( D , f )$ is

$$
L _ { c } ( D ) { \bf 1 } \{ f ( x ) = y _ { Z } ( x ) \ \mathrm { f o r } \ \mathrm { e v e r y } \ x \in O ( Z ) \} .
$$

The two factors involve separate variables, and the joint prior is a product. Normalization therefore yields a product posterior. Integrating $D _ { x }$ against its factor gives (2).

Feldman’s sampling prior. Given a list $\pi = ( \pi _ { 1 } , \ldots , \pi _ { N } ) $ of positive numbers, independently choose each $W _ { x }$ uniformly from its entries and set

$$
D _ { x } = { \frac { W _ { x } } { T } } , \qquad T = \sum _ { z = 1 } ^ { N } W _ { z } .\tag{3}
$$

Repeated entries are counted with their multiplicities. The list may sum to one, but the independently sampled weights need not. More generally, one may use any common distribution on positive raw weights. This construction gives an exchangeable prior: permuting the coordinates of D does not change its distribution. We write $\bar { \pi } ^ { N }$ for the marginal law of $D _ { x }$ in the list model, following Feldman. This law generally differs from the uniform law on the raw list.

Count-only conditioning. For a fixed $x ,$ conditional on $\mathit { D _ { x } } \ : = \ : \alpha$ , the count $C _ { x } = c _ { x } ( S )$ has distribution Binomial $( n , \alpha )$ . Consequently,

$$
\mathbb E [ D _ { x } \mid C _ { x } = \ell ] = \frac { \mathbb E [ D _ { x } ^ { \ell + 1 } ( 1 - D _ { x } ) ^ { n - \ell } ] } { \mathbb E [ D _ { x } ^ { \ell } ( 1 - D _ { x } ) ^ { n - \ell } ] } = \tau _ { n , \ell } ( x ) ,\tag{4}
$$

whenever the denominator is positive. Under an exchangeable prior the value does not depend on $x ,$ and we write $\tau _ { n , \ell }$ for this common value. Equation (4) is valid; the assertion $q _ { x } ( Z ) = \tau _ { n , c _ { x } }$ for every dataset is the additional claim that fails.

Risk and error-count conventions. For a dataset $Z ,$ , let

$$
\mathcal { R } ( A \mid Z ) = \mathbb { E } _ { D , f \mid Z } \mathbb { E } _ { h \sim A ( Z ) } \left[ \sum _ { x } D _ { x } \mathbf { 1 } \{ h ( x ) \neq f ( x ) \} \right] , \qquad \mathcal { R } ^ { * } ( Z ) = \operatorname* { i n f } _ { B } \mathcal { R } ( B \mid Z ) .
$$

The infimum is over all learners. Thus ${ \mathcal { R } } ^ { * } ( Z )$ is the minimum posterior risk achievable from the same observed data $Z$ and the specified priors; the predictor is not given the latent labeling function or sampling distribution. Write ${ \mathcal { R } } ( A ) = \mathbb { E } [ { \mathcal { R } } ( A \mid S ) ]$ ] and $\mathcal { R } ^ { * } = \operatorname* { i n f } _ { B } \mathcal { R } ( B )$ for the corresponding averaged quantities. For observed points define

$$
e _ { x } ( A , Z ) = { \mathbb P } _ { h \sim A ( Z ) } \{ h ( x ) \neq y z ( x ) \} , \qquad B _ { \ell } ( A , Z ) = \sum _ { x : c _ { x } = \ell } e _ { x } ( A , Z ) , \quad \ell \ge 1 .
$$

Thus $B _ { \ell }$ counts misclassified distinct points in expectation over the learner’s randomness, with no division by n and no extra factor ℓ. It is Feldman’s err ${ \mathfrak { l } } _ { Z } ( A , \ell )$ . In this notation the original assertion is

$$
\mathcal { R } ( A \mid Z ) - \mathcal { R } ^ { * } ( Z ) \geq \sum _ { \ell = 1 } ^ { n } \tau _ { n , \ell } B _ { \ell } ( A , Z ) ,\tag{5}
$$

for every learner and dataset, together with its averaged version. We now show that both assertions can fail.

## 3. An exact counterexample

We use $N = n = 3$ and the raw list

$$
\pi = ( 1 / 4 , 1 / 4 , 1 / 2 ) .
$$

Thus $W _ { i } = 1 / 4$ with probability $2 / 3$ and $W _ { i } = 1 / 2$ with probability $1 / 3 ,$ , independently for $i =$ 1, 2, 3. Direct averaging over the $2 7$ equally likely choices of list entries gives

$$
\tau _ { 3 , 1 } = \frac { \mathbb { E } [ D _ { 1 } ^ { 2 } ( 1 - D _ { 1 } ) ^ { 2 } ] } { \mathbb { E } [ D _ { 1 } ( 1 - D _ { 1 } ) ^ { 2 } ] } = \frac { 2 2 6 3 1 3 } { 6 8 2 2 6 0 } \approx 0 . 3 3 1 7 1 1 ,\tag{6}
$$

$$
\mathbb { E } [ D _ { 1 } \mid U = ( 1 , 2 , 2 ) ] = { \frac { \mathbb { E } [ D _ { 1 } ^ { 2 } D _ { 2 } ^ { 2 } ] } { \mathbb { E } [ D _ { 1 } D _ { 2 } ^ { 2 } ] } } = { \frac { 1 1 8 2 7 3 } { 3 5 8 1 4 0 } } \approx 0 . 3 3 0 2 4 2 .\tag{7}
$$

In contrast, exchangeability and $\textstyle \sum _ { i } D _ { i } = 1$ imply

$$
\mathbb { E } [ D _ { 1 } \mid U = ( 1 , 2 , 3 ) ] = 1 / 3 .\tag{8}
$$

The first point occurs once in both sequences, but its posterior means differ. Appendix A supplies an exact-arithmetic certificate.

Proposition 2 (Failure with independent fair labels) Let $N = n = 3$ , and let the sampling prior Π be the law of $D _ { i } = { W _ { i } } / { ( W _ { 1 } + W _ { 2 } + W _ { 3 } ) }$ , where the $W _ { i }$ are independent, with P $( W _ { i } = 1 / 4 ) =$ $2 / 3$ and $\mathbb { P } ( W _ { i } = 1 / 2 ) = 1 / 3$ . Independently of $D ,$ draw the labels $f ( 1 ) , f ( 2 ) , f ( 3 )$ independently from Bernoulli $( 1 / 2 )$ : each label is 0 or 1 with equal probability. Each point’s label is drawn once and is shared by all occurrences of that point in the sample. For these sampling and labeling priors, there exists a permutation-equivariant learnerfor which both (5) and its averaged versionfail.

Proof Consider the event $E$ that the counts are a permutation of (1, 2, 0). The learner predicts all observed labels correctly except that, on $E ,$ it flips the label of the unique singleton. On each unseen point it predicts 0 or 1 with equal probability, using independent random bits. This rule is equivariant under permutations of the domain.

The learner is posterior-optimal on every unseen point. Thus, conditional on a sample in $E ,$ , its excess risk is exactly the posterior mass of the singleton, namely (7). Its only observed-point error is that singleton, so the proposed penalty is (6). This is a strict violation. Outside E its excess risk and penalty are both zero. By exchangeability, averaging multiplies both quantities by the same positive factor $\mathbb { P } ( E )$ and preserves the violation. For example, $\mathbb { P } ( E ) = 6 \cdot 3 \mathbb { E } [ D _ { 1 } D _ { 2 } ^ { 2 } ] = 5 9 6 9 / 9 0 0 0$

A constant-zero labeling prior gives an even shorter witness: use the same rule with zero predictions except for the singleton flip. Then the optimal risk is zero. The fair-label construction shows that the failure does not depend on knowing all labels in advance.

The conditioning error. The proof of Lemma A.1 in Feldman (2021, v4) asserts that, given $D _ { x } = \alpha$ , the relative weights of all other points have the law of fresh independent raw weights normalized over those points. For the three-point construction with raw list $\pi = ( 1 / 4 , 1 / 4 , 1 / 2 )$ the identity $D _ { 1 } = W _ { 1 } / ( W _ { 1 } + W _ { 2 } + W _ { 3 } )$ gives $D _ { 1 } = 1 / 2 \iff W _ { 1 } = W _ { 2 } + W _ { 3 }$ . Since each $W _ { i }$ belongs to $\{ 1 / 4 , 1 / 2 \}$ , we have $W _ { 1 } \leq 1 / 2$ and $W _ { 2 } + W _ { 3 } \geq 1 / 2$ . Equality therefore forces $\left( W _ { 1 } , W _ { 2 } , W _ { 3 } \right) = \left( 1 / 2 , 1 / 4 , 1 / 4 \right)$ . Hence $D _ { 2 } / ( 1 - D _ { 1 } ) = D _ { 3 } / ( 1 - D _ { 1 } ) = 1 / 2$ deterministically under that conditioning. In contrast, two fresh independent draws from the raw list can be $1 / 4$ and $1 / 2 .$ , yielding relative weights $1 / 3$ and $2 / 3$ after normalization. The asserted conditional construction is therefore invalid. The counterexample uses the original finite-list model exactly; no limit, approximation, or alternative prior is involved.

## 4. An exponentially large relative failure

The three-point counterexample has a small numerical discrepancy. We now show that this is not a uniform limitation: even with $N = 3$ , the original coefficient can exceed the correct one by an arbitrarily large factor.

For even $n \geq 4$ , put $\varepsilon _ { n } = 2 ^ { - n / 2 }$ and use the probability list

$$
\pi ^ { ( n ) } = \frac { 1 } { 1 + 2 \varepsilon _ { n } } ( \varepsilon _ { n } , \varepsilon _ { n } , 1 ) .\tag{9}
$$

Multiplying every raw weight by the same constant leaves the normalized vector unchanged. We may therefore compute with weights equal to $\varepsilon _ { n }$ with probability $2 / 3$ and to 1 with probability $1 / 3$ Let $\Pi _ { n }$ denote the resulting prior on $D$ . Let $E _ { n }$ be the event that the count vector is a permutation of $( 1 , n - 1 , 0 )$ , and define the singleton’s posterior mass on this event by

$$
q _ { n , 1 } = \mathbb { E } _ { \Pi _ { n } } [ D _ { 1 } \mid ( C _ { 1 } , C _ { 2 } , C _ { 3 } ) = ( 1 , n - 1 , 0 ) ] .
$$

Here 1 denotes a singleton; the profile $( 1 , n - 1 , 0 )$ is implicit. By exchangeability, this is the singleton’s posterior mass for every sample in $E _ { n }$ . Throughout this section, $\tau _ { n , \ell }$ is computed under the sample-size-dependent prior $\Pi _ { n } ;$ the same convention applies in Appendix B. The notation $a _ { n } \sim b _ { n }$ means $a _ { n } / b _ { n } \to 1$ , here along even sample sizes.

Theorem 3 (No universal constant-factor repair) For the priors (9),

$$
q _ { n , 1 } \sim { \frac { 5 } { 4 } } 2 ^ { - n / 2 } , \qquad \tau _ { n , 1 } \sim { \frac { 1 } { 6 } } \left( { \frac { 2 \sqrt { 2 } } { 3 } } \right) ^ { n } , \qquad { \frac { \tau _ { n , 1 } } { q _ { n , 1 } } } \sim { \frac { 2 } { 1 5 } } \left( { \frac { 4 } { 3 } } \right) ^ { n } .\tag{10}
$$

With independent fair binary labels there is a permutation-equivariant learner $A _ { n }$ whose proposed averaged penalty $\begin{array} { r } { P _ { n } = \sum _ { \ell = 1 } ^ { n } \tau _ { n , \ell } \mathbb { E } [ B _ { \ell } ( A _ { n } , S ) ] } \end{array}$ and actual excess risk satisfy

$$
\frac { P _ { n } } { \mathcal { R } ( A _ { n } ) - \mathcal { R } ^ { * } } = \frac { \tau _ { n , 1 } } { q _ { n , 1 } } \longrightarrow \infty .\tag{11}
$$

Consequently, no constant $c > 0 _ { i }$ , independent ofthefrequency list and sample size, makes $\mathcal { R } ( A ) -$ $\begin{array} { r } { \mathcal { R } ^ { * } \geq c \sum _ { \ell } \tau _ { n , \ell } \mathbb { E } [ B _ { \ell } ( A , S ) ] } \end{array}$ validfor all such models and learners.

Proof Write $\varepsilon = \varepsilon _ { n }$ and define the four expectations

$$
\begin{array} { r l r l } & { a _ { n } = \mathbb { E } [ D _ { 1 } ( 1 - D _ { 1 } ) ^ { n - 1 } ] , \qquad } & & { b _ { n } = \mathbb { E } [ D _ { 1 } ^ { 2 } ( 1 - D _ { 1 } ) ^ { n - 1 } ] , } \\ & { z _ { n } = \mathbb { E } [ D _ { 1 } D _ { 2 } ^ { n - 1 } ] , \qquad } & & { v _ { n } = \mathbb { E } [ D _ { 1 } ^ { 2 } D _ { 2 } ^ { n - 1 } ] . } \end{array}
$$

Thus $\tau _ { n , 1 } ~ = ~ b _ { n } / a _ { n }$ and $q _ { n , 1 } \ = \ v _ { n } / z _ { n } .$ . The singleton count alone leaves posterior weight on all-equal raw configurations. Observing that another point occurs $n - 1$ times assigns those configurations a much smaller likelihood. Quantitatively, the eight raw-weight configurations give

$$
a _ { n } \sim \frac { \varepsilon } { 3 } , \qquad b _ { n } \sim \frac { 1 } { 2 7 } \left( \frac { 2 } { 3 } \right) ^ { n - 1 } , \qquad z _ { n } \sim \frac { 4 } { 2 7 } \varepsilon , \qquad v _ { n } \sim \frac { 5 } { 2 7 } \varepsilon ^ { 2 } .\tag{12}
$$

Here is the leading-term calculation. Since $n \varepsilon  0$ , factors of the form $( 1 + O ( \varepsilon ) ) ^ { n }$ tend to one. For $a _ { n } .$ , the configurations $( \varepsilon , 1 , \varepsilon )$ and $( \varepsilon , \varepsilon , 1 )$ each contribute asymptotically $( 4 / 2 7 ) \varepsilon$ , while $( \varepsilon , 1 , 1 )$ contributes asymptotically $\varepsilon / 2 7$ . The all-equal configurations have total probability $1 / 3$ and normalized vector $( 1 / 3 , 1 / 3 , 1 / 3 )$ ; they supply the leading term of $b _ { n }$ . Their contribution to $a _ { n }$ is negligible because $( 2 / 3 ) ^ { n } = o ( \varepsilon )$ . All other contributions to $b _ { n }$ are negligible because $\varepsilon ^ { 2 } =$ $2 ^ { - n } = o ( ( 2 / 3 ) ^ { n } )$ , or decay faster.

For $z _ { n } .$ , only $( \varepsilon , 1 , \varepsilon )$ contributes at order ε. For $v _ { n }$ this configuration contributes asymptotically $( 4 / 2 7 ) \varepsilon ^ { 2 }$ , while (1, 1, ε) contributes

$$
{ \frac { 2 } { 2 7 } } ( 2 + \varepsilon ) ^ { - ( n + 1 ) } \sim { \frac { 1 } { 2 7 } } 2 ^ { - n } = { \frac { 1 } { 2 7 } } \varepsilon ^ { 2 } .
$$

The other terms are smaller. Appendix B lists every configuration’s scaled limit. Taking ratios gives (10).

The learner $A _ { n }$ predicts observed labels correctly except that it flips the unique singleton on $E _ { n } ;$ ; it predicts independent fair bits on unseen points. This is permutation-equivariant and optimal except at the flipped point. Hence its excess risk is $\mathbb { P } ( E _ { n } ) q _ { n , 1 }$ , whereas $P _ { n } = \mathbb { P } ( E _ { n } ) \tau _ { n , 1 }$ . The event has positive probability, proving (11) and the final assertion.

Size of the event and absolute risks. There are six assignments of the counts $( 1 , n - 1 , 0 )$ to named points and n orders for each assignment. Thus

$$
\mathbb { P } ( E _ { n } ) = 6 n z _ { n } \sim { \frac { 8 } { 9 } } n 2 ^ { - n / 2 } ,\tag{13}
$$

and

$$
\mathcal { R } ( A _ { n } ) - \mathcal { R } ^ { * } \sim \frac { 1 0 } { 9 } n 2 ^ { - n } , \qquad P _ { n } \sim \frac { 4 } { 2 7 } n \left( \frac { 2 } { 3 } \right) ^ { n } .\tag{14}
$$

Both quantities vanish. The theorem excludes a uniform multiplicative repair over frequency lists, including on a fixed three-point domain. It does not exclude useful additive-error bounds, bounds on typical profiles, or approximations for a fixed long-tailed prior. The list in (9) depends on n, which is essential to the scope of this conclusion. For $n = 3 2$ , exact enumeration gives $\tau _ { 3 2 , 1 } / q _ { 3 2 , 1 } \approx 1 2 3 4 . 9 $ Appendix B provides further finite-sample values.

## 5. The corrected bound and its profile reduction

For the learner and label posterior, define

$$
a _ { x , y } ( Z ) = \mathbb { P } _ { h \sim A ( Z ) } ( h ( x ) = y ) , \qquad \eta _ { x , y } ( Z ) = \mathbb { P } ( f ( x ) = y \mid S = Z ) .
$$

The posterior label regret at $x$ is

$$
r _ { x } ( A , Z ) = \operatorname* { m a x } _ { y \in \mathcal { V } } \eta _ { x , y } ( Z ) - \sum _ { y \in \mathcal { V } } a _ { x , y } ( Z ) \eta _ { x , y } ( Z ) \ge 0 .\tag{15}
$$

By noiselessness, $r _ { x } ( A , Z ) = e _ { x } ( A , Z )$ when $c _ { x } > 0$

Theorem 4 (Exact posterior decomposition) For every sampling prior Π, independent labeling prior ${ \mathcal { F } } ,$ , learner A, and positive-probability dataset $Z ,$

$$
\mathcal { R } ^ { * } ( Z ) = \sum _ { x \in \mathcal { X } } q _ { x } ( Z ) \left( 1 - \operatorname* { m a x } _ { y \in \mathcal { Y } } \eta _ { x , y } ( Z ) \right) ,\tag{16}
$$

and

$$
\mathcal { R } ( A \mid Z ) - \mathcal { R } ^ { * } ( Z ) = \sum _ { x : c _ { x } > 0 } q _ { x } ( Z ) e _ { x } ( A , Z ) + \sum _ { x : c _ { x } = 0 } q _ { x } ( Z ) r _ { x } ( A , Z ) .\tag{17}
$$

In particular, dropping the second sum gives a lower bound. Equality in that lower bound is achieved by every learner that makes posterior-optimal predictions on unseen points.

Proof By Lemma 1 and the independence of the learner’s randomization,

$$
\mathcal { R } ( A \mid Z ) = \sum _ { x } q _ { x } ( Z ) \left( 1 - \sum _ { y } a _ { x , y } ( Z ) \eta _ { x , y } ( Z ) \right) .
$$

Because the output class contains all functions on $x ,$ , the conditional risk can be minimized sepa rately at each point. Choosing any maximizing label for each $\eta _ { x , y }$ gives (16). Subtracting yields $\textstyle \sum _ { x } q _ { x } r _ { x }$ . For observed points the label posterior is a point mass at $y _ { Z } ( x )$ , giving (17). A maximizing label on each unseen point sets its regret to zero. ■

An alternative proof of the inequality modifies A by correcting all its observed-point predictions and keeping its other predictions. The decrease in risk is exactly the first sum in (17). The modified learner cannot have risk below ${ \mathcal { R } } ^ { * } ( Z )$ . This proof explains the cost of not fitting without any exchangeability assumption.

Sharpness. For any chosen observed point x, a learner can make a deliberate error only at x and use optimal predictions everywhere else. Its excess risk is exactly $q _ { x } ( Z )$ . Thus the individual weights in the lower bound cannot be increased while retaining a statement for all learners. A point observed with positive count necessarily has positive posterior mass under a positive-probability dataset.

Since the data space is finite and pointwise minimizers exist,

$$
\mathcal { R } ^ { * } : = \operatorname* { i n f } _ { B } \mathcal { R } ( B ) = \mathbb { E } [ \mathcal { R } ^ { * } ( S ) ] .
$$

Averaging (17) consequently gives a valid unconditional decomposition as well. The posterior coefficient remains inside the expectation; it cannot in general be replaced by a count-only mean.

## 5.1. Reduction to the frequency profile

Assume from now on that Π is exchangeable. Define the profile

$$
H = ( K _ { 0 } , \ldots , K _ { n } ) , \qquad K _ { j } = \# \{ x : C _ { x } = j \} .
$$

Every profile satisfies $\textstyle \sum _ { j } K _ { j } = N$ and $\textstyle \sum _ { j } j K _ { j } = n$ . The coordinate $K _ { 0 }$ records the number of unseen domain points; N is known in this model. A profile is feasible if it has positive probability. Let $\mathcal { H } _ { N , n }$ denote the finite set of feasible profiles.

Lemma 5 (Profile reduction) There arefunctions $\theta _ { n , \ell } ( h )$ , defined whenever $h \in \mathcal { H } _ { N , n }$ and $h _ { \ell } >$ 0, such that

$$
q _ { x } ( Z ) = \theta _ { n , c _ { x } } ( H ( Z ) ) .
$$

In particular, all points of the same multiplicity have the same posterior mean, regardless of their observed labels.

Proof If two count vectors have the same profile, a permutation of domain points maps one to the other. If their designated coordinates have the same count, the permutation can be chosen to map those coordinates to each other. Apply this change of variables to numerator and denominator of (2); exchangeability preserves the prior. Lemma 1 removes the dependence on labels.

Corollary 6 (Corrected profile-dependent bound) Under an exchangeable sampling prior,

$$
\mathcal { R } ( A \mid Z ) - \mathcal { R } ^ { * } ( Z ) \geq \sum _ { \ell \geq 1 : K _ { \ell } ( Z ) > 0 } \theta _ { n , \ell } ( H ( Z ) ) B _ { \ell } ( A , Z ) .\tag{18}
$$

Its averaged form is

$$
\mathcal { R } ( A ) - \mathcal { R } ^ { * } \geq \mathbb { E } \left[ \sum _ { \ell \geq 1 : K _ { \ell } > 0 } \theta _ { n , \ell } ( H ) B _ { \ell } ( A , S ) \right] .\tag{19}
$$

Both bounds are attained by learners that predict optimally on unseen points, with arbitrary predictions on observed points.

Proof Substitute Lemma 5 into Theorem 4, group the observed points by multiplicity, and then average. ■

The sampling prior must be exchangeable, but the labeling prior and the learner need not be. In particular, label structure does not destroy the reduction of the frequency posterior to the profile. It affects the unseen-point regret and the learner’s pattern of errors.

The corrected coefficients obey the normalization identity

$$
\sum _ { \ell : K _ { \ell } > 0 } K _ { \ell } \theta _ { n , \ell } ( H ) = 1 ,\tag{20}
$$

including $\ell = 0$ . Computing them requires the sampling prior; the profile alone does not specify a prior-free numerical value.

## 6. Good’s formula and sharp count-only bounds

## 6.1. An exact averaging identity

Define the random total probability of the points with multiplicity ℓ by

$$
\mathsf { M } _ { \ell } = \sum _ { x } D _ { x } \mathbf { 1 } \{ C _ { x } = \ell \} .
$$

This quantity depends on both D and the sample. Conditional on the profile, its expectation is

$$
\mathbb { E } [ \mathsf { M } _ { \ell } \mid H ] = K _ { \ell } \theta _ { n , \ell } ( H ) ,\tag{21}
$$

where the product is interpreted as zero when $K _ { \ell } = 0$

Proposition 7 (Relation to the count-only coefficient) For every ℓ with $\mathbb { P } ( C _ { x } = \ell ) > 0 ;$

$$
\tau _ { n , \ell } = \mathbb { E } [ \theta _ { n , \ell } ( H ) \mid C _ { x } = \ell ] = { \frac { \mathbb { E } [ K _ { \ell } \theta _ { n , \ell } ( H ) ] } { \mathbb { E } [ K _ { \ell } ] } } = { \frac { \mathbb { E } [ \mathsf { M } _ { \ell } ] } { \mathbb { E } [ K _ { \ell } ] } } .\tag{22}
$$

Proof Apply the tower property to $\mathbb { E } [ D _ { x } \mid C _ { x } = \ell ]$ and use Lemma 5. Alternatively, exchangeability gives

$$
\mathbb { E } [ \mathsf { M } _ { \ell } ] = N \mathbb { E } [ D _ { x } \mathbf { 1 } \{ C _ { x } = \ell \} ] , \qquad \mathbb { E } [ K _ { \ell } ] = N \mathbb { P } ( C _ { x } = \ell ) .
$$

Their ratio is (4); (21) gives the middle expression.

Thus the averaging is weighted by the number of points in the multiplicity class. An unweighted expectation over profiles is generally different. A profile containing more singletons is more likely under conditioning on a specified point being a singleton.

For Good’s fixed-population model, let $p _ { 1 } , \ldots , p _ { N }$ be fixed population probabilities and choose an index I uniformly from [N], independently of the sample. Then

$$
\mathbb { E } [ p _ { I } \mid C _ { I } = \ell ] = { \frac { \sum _ { i } p _ { i } ^ { \ell + 1 } ( 1 - p _ { i } ) ^ { n - \ell } } { \sum _ { i } p _ { i } ^ { \ell } ( 1 - p _ { i } ) ^ { n - \ell } } } .\tag{23}
$$

This is formula (13), with $m = 1$ , of Good (1953), after changing notation. It has the same functional form as (4). Good averages over a uniformly selected species in a fixed population; Feldman’s coefficient averages over the marginal law of a normalized random frequency.

One may also put a uniform random permutation prior on the fixed vector $( p _ { 1 } , \ldots , p _ { N } )$ . The marginal frequency law is then $N ^ { - 1 } \sum _ { i } \delta _ { p _ { i } }$ , where $\delta _ { p }$ is a unit point mass at $p ,$ so (23) is its countonly posterior mean. The full-profile posterior and Corollary 6 still apply, but the random permutation does not in general make conditioning on the other counts redundant. The empirical Good– Turing estimate, typically written $( \ell + 1 ) K _ { \ell + 1 } / ( n K _ { \ell } )$ when $K _ { \ell } > 0$ and $\ell < n$ , is a further estimation step; none of the exact identities here justify substituting it into a universal learner-dependent bound without additional control.

## 6.2. The largest coefficients valid for every learner

For each attainable positive multiplicity, define

$$
\begin{array} { r } { \underline { { \theta } } _ { n , \ell } = \operatorname* { m i n } \{ \theta _ { n , \ell } ( h ) : h \in \mathcal { H } _ { N , n } , \ h _ { \ell } > 0 \} . } \end{array}\tag{24}
$$

The minimum exists because the set of profiles is finite.

Theorem 8 (Sharp multiplicity-only coefficients) Fix an exchangeable sampling prior, a labeling prior $\mathcal { F }$ independent ofD, $N ,$ and n. Let $b _ { \ell }$ be constants for the attainable positive multiplicities. Thefollowing statements are equivalent:

1. For every learner and every positive-probability dataset,

$$
\mathcal { R } ( A \mid Z ) - { \mathcal { R } } ^ { * } ( Z ) \geq \sum _ { \ell = 1 } ^ { n } b _ { \ell } B _ { \ell } ( A , Z ) .
$$

2. For every learner,

$$
\mathcal { R } ( A ) - \mathcal { R } ^ { * } \geq \sum _ { \ell = 1 } ^ { n } b _ { \ell } \mathbb { E } [ B _ { \ell } ( A , S ) ] .
$$

3. $b _ { \ell } \leq \underline { { \theta } } _ { n , \ell }$ for every attainable positive multiplicity.

Here the learner class is unrestricted, as in Section 2. Terms corresponding to unattainable multiplicities are zero and may be omitted.

Proof The third assertion implies the first by Corollary 6, since $B _ { \ell } \geq 0 ;$ averaging gives the second. For the converse, suppose $b _ { \ell } > \theta _ { n , \ell } ( h )$ at a feasible profile h with $h _ { \ell } > 0$ . Under the fixed labeling prior, start from a posterior-optimal predictor. When $H = h$ , select uniformly one observed point with count ℓ and replace its known label by any different label; leave all other predictions optimal. Conditional on this profile, the excess risk is exactly $\theta _ { n , \ell } ( h )$ and $B _ { \ell } = 1$ . Outside this profile the excess risk and all $B _ { j }$ are zero. Thus the first assertion fails. The unconditional excess risk is $\mathbb { P } ( H = h ) \theta _ { n , \ell } ( h )$ ), whereas the claimed penalty is $\mathbb { P } ( H = h ) b _ { \ell }$ . Feasibility ensures $\mathbb { P } ( H = h ) > 0$ so the second assertion fails too.

Sharpness permits learners to concentrate errors on a selected profile; restricted algorithm or hypothesis classes may admit larger coefficients. The witness need not be permutation-equivariant for arbitrary labeling priors. For independent fair labels, however, fair predictions on unseen points and uniform selection within the multiplicity class give an equivariant witness. Thus symmetry alone does not repair the theorem.

Corollary 9 (Validity of the count-only coefficient) The bounds in Theorem 8 hold with $b _ { \ell } = \tau _ { n , \ell }$ for all attainable positive multiplicities if and only if, for every such $\ell , \theta _ { n , \ell } ( h )$ is constant over all feasible profiles with $h _ { \ell } > 0$

Proof By Proposition 7, $\tau _ { n , \ell }$ is a weighted average of these profile coefficients. Every feasible profile with $h _ { \ell } > 0$ has strictly positive weight, proportional to $h _ { \ell } \mathbb { P } ( H = h )$ . An average is at most the minimum exactly when all its positively weighted terms are equal. Apply Theorem 8.

For the counterexample, the singleton-containing profiles are (1, 1, 1, 0) and $( 0 , 3 , 0 , 0 )$ . Their singleton coefficients are respectively (7) and $1 / 3$ . Thus the sharp universal singleton coefficient is 118273/358140, strictly below $\tau _ { 3 , 1 }$ . The improvement supplied by a realized profile is retained by (18) and lost by taking the minimum in (24).

## 6.3. Relation to Johnson’s sufficientness postulate

The posterior mean $q _ { x } ( c )$ is also the predictive probability that the next input equals x. Requiring it to depend on the sample only through n and $c _ { x }$ is a form of Johnson’s sufficientness postulate. Zabell (1982) develops classical characterizations of Dirichlet prediction from such conditions, imposed across sample sizes with additional assumptions. See also Lee et al. (2013) for the species-sampling setting.

Corollary 9 instead fixes $N , n ,$ and the prior, and characterizes a risk bound valid for every learner using posterior means for positive counts only. Without conditions across sample sizes or for unseen categories, it implies neither a Dirichlet characterization nor uniqueness of the prior.

A case where count-only conditioning is sufficient. For comparison, if $W _ { i }$ have a common Gamma distribution with shape $a > 0$ and rate $b > 0$ , then D has the symmetric Dirichlet distribu tion. Multiplication of its density by $L _ { c }$ gives a Dirichlet posterior with parameters $( a + c _ { 1 } , \ldots , a +$ $c _ { N } )$ , so

$$
\theta _ { n , \ell } ( h ) = \frac { a + \ell } { N a + n } = \tau _ { n , \ell } .
$$

The profile dependence disappears, as required by Corollary 9. This example checks the formulas and isolates a setting where the original proof strategy works. Our correction does not impose this prior or replace Feldman’s finite list by Gamma weights.

## 7. Consequences for the cited results

Unqualified result and equation numbers below refer to Feldman (2021), arXiv version 4; references to our results are identified explicitly.

Disproved statements. Lemma 2.1, proved through Lemma A.1, identifies the full-sample posterior mass with the count-only coefficient. Our counterexample disproves that identity and both forms of Theorem 2.3. Our Theorem 3 also rules out repairing its averaged form by a universal positive constant multiplier.

The subpopulation extension, Theorem 3.2, also fails when subpopulations are point masses on distinct domain points. The two prediction distributions in Definition 3.1 then coincide, making every learner subpopulation-coupled with coupling parameters one. The bound reduces to Theorem 2.3, disproved by our Proposition 2. We do not analyze the coupling argument for nontrivial subpopulations.

Affected proof steps. In the noisy-label Theorem 2.4, equation (7) replaces full posterior masses by the singleton coefficient. The population-risk conclusions of Corollary 4.6 and Theorem 4.8 invoke that bound, so these conversions require separate justification. We do not supply counterexamples to each noisy-label or privacy statement or resolve their other assumptions and conventions.

Calculations independent of the conditioning error. Several intermediate calculations do not use the failed identity. The singleton-count formula and equation (8) in Section 2.5 are marginal occupancy calculations. The first inequality of Lemma 2.6 bounds the count-only coefficient itself; such a bound does not by itself bound the coefficient at every profile. Likewise, the label-prediction arguments in Lemma 4.4 and the first part of Theorem 4.8 do not rely on the frequency-conditioning step. Their conversion into excess population risk is the part requiring care. This dependency analysis does not certify every other statement in those sections.

Available repair and unresolved scope. Our Theorem 4 supplies an exact noiseless replacement. Under the additional fixed-n condition of our Corollary 9, the original noiseless coefficients are valid. More generally, a uniform profile-wise lower bound on $\theta _ { n , \ell }$ would suffice. The present counterexamples do not settle the proposed Zipf or other fixed long-tail conclusions; recovering them requires controlling these posterior coefficients, or the resulting learner-weighted expectation.

## 8. Discussion

Full-sample posterior frequencies give a valid lower bound in the original normalized-weight model for every independent labeling prior and every learner. The sharp-coefficient theorem identifies exactly when count-only coefficients suffice. A next step is to bound the corrected penalty for specified long-tailed priors or algorithm families. This requires controlling the posterior coefficient or its learner-weighted expectation; concentration of the normalization sum alone does not supply that control.

## AI Disclosure

Generative AI was used to assist with mathematical exploration, drafting, and LaTeX preparation.

## References

Vitaly Feldman. Does learning require memorization? a short tale about a long tail. arXiv:1906.05271v4, 2021. URL https://arxiv.org/abs/1906.05271v4. Version 4, January 10, 2021. Conference version appeared in STOC 2020.

I. J. Good. The population frequencies of species and the estimation of population parameters. Biometrika, 40(3–4):237–264, 1953. doi: 10.1093/biomet/40.3-4.237.

Lancelot F. James, Antonio Lijoi, and Igor Prunster. Posterior analysis for normalized random ¨ measures with independent increments. Scandinavian Journal of Statistics, 36(1):76–97, 2009. doi: 10.1111/j.1467-9469.2008.00609.x.

Jaeyong Lee, Fernando A. Quintana, Peter Muller, and Lorenzo Trippa. Defining predictive prob-¨ ability functions for species sampling models. Statistical Science, 28(2):209–222, 2013. doi: 10.1214/12-STS407.

S. L. Zabell. W. E. Johnson’s “sufficientness” postulate. The Annals of Statistics, 10(4):1090–1099, 1982. doi: 10.1214/aos/1176345975.

## Appendix A. Exact-arithmetic certificate

Rescale all raw weights by four, which leaves D unchanged. Let $V _ { i } = 4 W _ { i } \in \{ 1 , 2 \}$ , with probabilities $2 / 3$ and $1 / 3 ,$ , and let $s ( v )$ count the coordinates of $v \in \{ 1 , 2 \} ^ { 3 }$ equal to two. Every expectation used in Section 3 can be computed from the eight-term formula

$$
\mathbb { E } [ g ( D ) ] = \sum _ { v \in \{ 1 , 2 \} ^ { 3 } } { \frac { 2 ^ { 3 - s ( v ) } } { 2 7 } } g \left( { \frac { v } { v _ { 1 } + v _ { 2 } + v _ { 3 } } } \right) .\tag{25}
$$

The eight vectors and their weights are shown in Table 1. This is equivalent to enumeration over the 27 equally likely list-entry choices, but combines repeated entries.

Table 1: Exact normalized vectors in the counterexample.
<table><tr><td>Raw vector v</td><td>Probability</td><td>Normalized vector D</td></tr><tr><td>(1, 1, 1)</td><td>8/27</td><td> $( 1 / 3 , 1 / 3 , 1 / 3 )$ </td></tr><tr><td>(2, 1, 1)</td><td>4/27</td><td>(1/2,1/4,1/4)</td></tr><tr><td>(1, 2, 1)</td><td>4/27</td><td>(1/4,1/2,1/4)</td></tr><tr><td>(1,1, 2)</td><td>4/27</td><td>(1/4,1/4,1/2)</td></tr><tr><td>(2, 2, 1)</td><td>2/27</td><td>(2/5, 2/5, 1/5)</td></tr><tr><td>(2, 1, 2)</td><td>2/27</td><td>(2/5, 1/5,2/5)</td></tr><tr><td>(1, 2,2)</td><td>2/27</td><td>(1/5, 2/5,2/5)</td></tr><tr><td>(2, 2,2)</td><td>1/27</td><td>(1/3, 1/3,1/3)</td></tr></table>

In particular, substituting the four expressions used in the counterexample into (25) gives

$$
\begin{array} { r l r } { \mathbb { E } [ D _ { 1 } ( 1 - D _ { 1 } ) ^ { 2 } ] = \displaystyle \frac { 1 1 3 7 1 } { 8 1 0 0 0 } , } & { \quad } & { \mathbb { E } [ D _ { 1 } ^ { 2 } ( 1 - D _ { 1 } ) ^ { 2 } ] = \displaystyle \frac { 2 2 6 3 1 3 } { 4 8 6 0 0 0 0 } , } \\ { \mathbb { E } [ D _ { 1 } D _ { 2 } ^ { 2 } ] = \displaystyle \frac { 5 9 6 9 } { 1 6 2 0 0 0 } , } & { \quad } & { \mathbb { E } [ D _ { 1 } ^ { 2 } D _ { 2 } ^ { 2 } ] = \displaystyle \frac { 1 1 8 2 7 3 } { 9 7 2 0 0 0 } . } \end{array}
$$

Taking the corresponding ratios yields (6) and (7). Thus the finite counterexample can be checked directly from Table 1, without external code.

## Appendix B. Leading terms for the exponential relative gap

Let $\varepsilon = 2 ^ { - n / 2 } , u _ { n } = ( 2 / 3 ) ^ { n - 1 }$ , and use the definitions of $a _ { n } , b _ { n } , z _ { n } , v _ { n }$ from Theorem 3. Table 2 records the contribution of each raw configuration to the indicated scaled expectation, in the limit along even n. The first row combines the all-small and all-large configurations because they yield the same normalized vector. The other rows have probabilities $4 / 2 7$ (one large weight) or $2 / 2 7$ (two large weights), as indicated.

Table 2: Limits of individual contributions to the four scaled expectations.
<table><tr><td>Raw configuration</td><td>Probability</td><td> $a _ { n } / \varepsilon$   $b _ { n } / u _ { n }$ </td><td> $z _ { n } / \varepsilon$ </td><td> $v _ { n } / \varepsilon ^ { 2 }$ </td></tr><tr><td>All equal</td><td>1/3</td><td>0  $1 / 2 7$ </td><td>0</td><td>0</td></tr><tr><td>(1, ε, ε)</td><td>4/27</td><td>0 0</td><td>0</td><td>0</td></tr><tr><td>(ε, 1, ε)</td><td>4/27</td><td>4/27 0</td><td>4/27</td><td>4/27</td></tr><tr><td>(ε,ε,1)</td><td>4/27</td><td>4/27 0</td><td>0</td><td>0</td></tr><tr><td>(1, 1, ε)</td><td>2/27</td><td>0 0</td><td>0</td><td>1/27</td></tr><tr><td>(1, ε, 1)</td><td>2/27</td><td>0 0</td><td>0</td><td>0</td></tr><tr><td>(ε,1,1)</td><td>2/27</td><td>1/27 0</td><td>0</td><td>0</td></tr><tr><td>Sum</td><td>1</td><td>1/3  $1 / 2 7$ </td><td> $4 / 2 7$ </td><td>5/27</td></tr></table>

For example, the third row contributes

$$
{ \frac { 4 } { 2 7 } } { \frac { \varepsilon } { 1 + 2 \varepsilon } } \left( { \frac { 1 + \varepsilon } { 1 + 2 \varepsilon } } \right) ^ { n - 1 }
$$

to $a _ { n }$ , and $( 4 / 2 7 ) \varepsilon ^ { 2 } ( 1 + 2 \varepsilon ) ^ { - ( n + 1 ) }$ to $v _ { n }$ . The first row contributes $( 1 / 2 7 ) ( 2 / 3 ) ^ { n - 1 }$ to $b _ { n }$ exactly. The fifth row contributes $( 2 / 2 7 ) ( 2 + \varepsilon ) ^ { - ( n + 1 ) }$ to $v _ { n }$ . The limits use $n \varepsilon  0 , ( 2 / 3 ) ^ { n } / \varepsilon  0$ , and $2 ^ { - n } / ( 2 / 3 ) ^ { n } \to 0$ . The remaining terms either have one of these vanishing ratios or contain $\varepsilon ^ { n - 1 }$ whose contribution is smaller than every fixed exponential rate. Since the number of configurations is finite, summing the limits proves (12).

## Appendix C. Evaluating the corrected coefficients

The ratio in (2) defines the corrected weights for any prior. For independently normalized positive weights it can be reduced to integrals in one variable, avoiding enumeration over all raw-weight assignments.

Table 3: Finite-sample values for the family (9), computed with exact rational arithmetic and rounded for display.
<table><tr><td>n</td><td> $\tau _ { n , 1 }$ </td><td> $q _ { n , 1 }$ </td><td> $\tau _ { n , 1 } / q _ { n , 1 }$ </td></tr><tr><td>8</td><td>0.158246</td><td>0.0839837</td><td>1.88424</td></tr><tr><td>16</td><td>0.0609648</td><td>0.00494029</td><td>12.3403</td></tr><tr><td>32</td><td>0.0235565</td><td> $1 . 9 0 7 5 7 \times 1 0 ^ { - 5 }$ </td><td>1234.9</td></tr><tr><td>64</td><td>0.00380177</td><td> $2 . 9 1 0 3 8 \times 1 0 ^ { - 1 0 }$ </td><td> $1 . 3 0 6 2 8 \times 1 0 ^ { 7 }$ </td></tr></table>

Suppose $W _ { 1 } , \dots , W _ { N }$ are independent with a common law on $( 0 , \infty )$ and $D _ { i } = W _ { i } / T$ . For integers $j \geq 0 , t > 0$ , and a profile $h = ( h _ { 0 } , \ldots , h _ { n } )$ , define

$$
\phi _ { j } ( t ) = \mathbb { E } [ W ^ { j } e ^ { - t W } ] , \qquad F _ { h } ( t ) = \prod _ { j = 0 } ^ { n } \phi _ { j } ( t ) ^ { h _ { j } } .
$$

Each $\phi _ { j } ( t )$ is finite for $t > 0$ , even if the raw moments of W are infinite.

Proposition 10 (One-dimensional integral formula) For $h \in \mathcal { H } _ { N , n }$ and $h _ { \ell } > 0 ,$ , including $\ell = 0 ,$

$$
\theta _ { n , \ell } ( h ) = \frac { 1 } { n } \frac { \displaystyle \int _ { 0 } ^ { \infty } t ^ { n } \phi _ { \ell + 1 } ( t ) \phi _ { \ell } ( t ) ^ { h _ { \ell } - 1 } \prod _ { j \neq \ell } \phi _ { j } ( t ) ^ { h _ { j } } \mathrm { d } t } { \displaystyle \int _ { 0 } ^ { \infty } t ^ { n - 1 } F _ { h } ( t ) \mathrm { d } t } .\tag{26}
$$

Both integrals arefinite and the denominator is positive.

Proof For $T > 0$ and an integer $m \geq 1$

$$
T ^ { - m } = { \frac { 1 } { \Gamma ( m ) } } \int _ { 0 } ^ { \infty } t ^ { m - 1 } e ^ { - t T } \mathrm { d } t .
$$

For a count vector c with profile h, independence and Tonelli’s theorem give

$$
\mathbb { E } \left[ \prod _ { i } D _ { i } ^ { c _ { i } } \right] = \frac { 1 } { \Gamma ( n ) } \int _ { 0 } ^ { \infty } t ^ { n - 1 } \prod _ { i } \phi _ { c _ { i } } ( t ) \mathrm { d } t .
$$

The numerator of (2) increases one exponent from ℓ to $\ell + 1$ and replaces ${ \cal T } ^ { - n }$ by $T ^ { - ( n + 1 ) }$ <sup>)</sup>. Apply the same identity with $m = n + 1$ . The ratio of the Gamma factors is $\Gamma ( n ) / \Gamma ( n + 1 ) = 1 / n .$ yielding (26). The expectations of the corresponding products of probabilities are finite, bounded by one, and the denominator is positive. This also verifies the stated integrability without separate raw-moment assumptions.

For a finite raw list, $\begin{array} { r } { \phi _ { j } ( t ) = N ^ { - 1 } \sum _ { r = 1 } ^ { N } \pi _ { r } ^ { j } e ^ { - t \pi _ { r } } } \end{array}$ . Equation (26) is a computational representation; small likelihoods may require rescaling or logarithmic evaluation. For the three-point example, quadrature gives singleton coefficients 0.330242363322 and 0.333333333333 for profiles (1, 1, 1, 0) and (0, 3, 0, 0), matching (7) and (8). With absolute and relative quadrature tolerances $1 0 ^ { - 1 2 }$ , the maximum observed discrepancy over all six feasible profile–multiplicity pairs was at most $1 . 2 \times 1 0 ^ { - 1 6 }$ . This diagnostic is not a certified numerical-error bound. The exact values follow independently from the finite sum in Appendix A; the quadrature comparison is only an implemen tation check.