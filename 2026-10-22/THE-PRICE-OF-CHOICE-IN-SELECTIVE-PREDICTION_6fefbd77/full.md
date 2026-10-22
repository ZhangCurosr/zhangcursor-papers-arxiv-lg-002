# THE PRICE OF CHOICE IN SELECTIVE PREDICTION

Daniel Cieslak´ <sup>∗</sup>   
Gdansk University of Technology´   
Gdansk, Poland ´

Andrzej Czyzewski˙ Gdansk University of Technology´ Gdansk, Poland´

## ABSTRACT

A vendor may claim that some rule from a predeclared menu—a score threshold or a union of K score intervals—lets its selective classifier keep a fraction κ of inputs while erring on at most a fraction α of them. Refuting such an existential claim means ruling out every permitted rule at once. The intersection–union test (IUT) does this without a multiplicity correction, but its sample cost as the menu grows, and how close that cost is to the best valid audit, are not captured by the level argument. We characterize both. For a fixed interior risk target α, menus of K intervals and a risk gap γ, in the small-gap, large-menu regime of our theorem the minimax sample size at each fixed coverage floor $\kappa \in ( 0 , 1 )$ is $K ( 1 - \kappa ) / ( \kappa \gamma ^ { 2 } )$ up to logarithmic factors, and the exact IUT attains it: menu richness costs K times the abstention odds, even though validity costs nothing. The lower bound holds for audits that know the score distribution; the upper bound extends to any menu of VC dimension d. We then explain the linear cost. Every certificate that never rejects while some rule looks feasible on the audit data—the IUT, Bonferroni, DKW and uniform-convergence bounds—pays it even at laws where $\sqrt { K } / \gamma ^ { 2 }$ samples suffice, because the best-looking rule is optimistically biased. We then identify when the gap can and cannot be avoided: against constant-risk alternatives a reference-free audit reaches the testing scale, including the square-root rate on the sign family in its stated small-gap regime, whereas over a rich class with unknown heterogeneous risk the linear cost is unavoidable up to a logarithm. In simulation with ${ \bar { K } } = 4 .$ , the raw menu cost changes 7.7-fold between $\kappa = . 5$ and .95; after dividing by the abstention odds and the theorem-motivated localized-VC logarithmic normalization used in Figure 1, the two endpoint values differ by about 5%. Held-out model families show that the same feasible-mixture mechanism is not confined to synthetic constructions: across three contemporary VLM judges, all 60 split-specific confidence-gate families have positive exact-hiding windows and 28/30 contracts selected on audit-A remain hidden on untouched audit-B; on the cleaner multi-annotator Polaris target, hiding persists on all 20 splits but is much narrower and transfers in 3/10 resplits. Frozen image and multimodal classifiers separately show boundary behavior consistent with the exact IUT guarantee and strong power relative to the practical certificates we compare.

## 1 INTRODUCTION

Before a selective classifier is deployed—in procurement, certification or a regulatory feasibility check—its vendor often claims that it can be configured to meet a standard: some permitted acceptance rule retains at least 90% of inputs while erring on at most 10% of the accepted ones. The menu of permitted rules is fixed in advance, for example thresholds on a confidence score or unions of a few score intervals. An auditor with fresh labeled data who wants to refute the claim cannot test one rule of its choice: showing that one rule fails leaves the vendor free to pick another. The null hypothesis is a union over the menu, and a rejection must rule out every rule at once. This is the dual of risk control and Learn-then-Test, which certify one rule selected by calibration (Chow, 1970; El-Yaniv and Wiener, 2010; Geifman and El-Yaniv, 2017; Bates et al., 2021; Angelopoulos et al., 2024; 2025).

The intersection–union test (IUT) is the natural audit: reject when every rule fails its own valid test.   
Because a true claim has one fixed witness rule, no multiplicity correction is needed (Berger, 1982;

Berger and Hsu, 1996). This invites a tempting conclusion—that the size of the menu is free. It is not. We characterize the statistical price of menu richness, compare the exact IUT with the best valid audit, and identify assumptions under which the gap can close.

The main result in one line. For menus of at most K score intervals, a fixed interior risk target α, a fixed coverage floor $\kappa \in ( 0 , 1 )$ and risk gap γ, the number of audit samples needed to refute every robustly false claim is

$$
N \ \asymp \ \underbrace { 1 - \kappa } _ { \mathrm { a b s t e n i o n ~ o d d s } } \cdot \frac { K } { \gamma ^ { 2 } } \qquad \mathrm { i n ~ t h e ~ s m a l l . g a p , ~ l a r g e - m e n u ~ r e g i m e , ~ u p ~ t o ~ l o ~ g a r i t h m i c ~ f a c t o r s , } \quad ( 1 )
$$

and the exact IUT attains it (Theorem 1). This is the price of the menu, and it arises because the vendor’s freedom lies in which small part of the score line to abstain on (or, for $\kappa < 1 / 2 ,$ to accept): the auditor must rule out that the errors happen to be concentrated there, and the error mass of such a region fluctuates on the scale of that region, not of the whole sample. Richness is therefore cheap at high floors: at $K = 4$ in our simulations the menu cost at $\kappa = . 9$ is 4.4 times smaller than at $\kappa = . 5$ (Figure 1).

Contributions. Table 1 summarizes the results; every lower bound holds for audits that are valid over the full, unrestricted null.

1. The abstention-odds law and optimality of the IUT (Section 3). For any menu of VC dimension d the IUT needs ${ \tilde { O } } ( d ( 1 - \kappa + \gamma ) / ( \kappa \gamma ^ { 2 } ) )$ samples (Proposition 2); for unions of K intervals, in the theorem’s small-gap regime, no valid audit, even one that knows the score distribution, needs fewer than $\Omega ( \bar { K ( 1 - \kappa ) } / ( \kappa \gamma ^ { 2 } \log K ) )$ at any fixed floor κ (Theorem 1). The law covers the high floors used in practice $( \kappa = . 9$ in our experiments), where richness is several times cheaper than at $\kappa = 1 / 2$

2. A plug-in barrier (Section 4). We call a certificate EFC if it never rejects while some rule looks feasible on the audit data; the IUT, Bonferroni, DKW and uniform-convergence bounds all are. On explicit families with k effective groups, every valid EFC certificate needs $\Omega ( k / \gamma ^ { 2 } )$ samples at a law where $\Theta ( \sqrt { k } / \gamma ^ { 2 } )$ suffice (Theorems 2 and 3), and this carries over to interval menus (Remark 1). The obstruction is a winner’s curse: the best empirical rule is optimistically biased by $\sqrt { k / n }$

3. What a better audit must know (Section 5). An unknown score distribution does not force the linear cost: against constant-risk alternatives, a reference-free audit valid for the unrestricted null reaches the testing scale and, on the uniform sign family in the stated small-gap regime, the square-root rate (Theorem 4). By contrast, uncertainty about heterogeneous conditional risk can force the linear cost up to a logarithm over the class of Theorem 5.

4. Evidence (Section 6). Simulations illustrate the coverage-floor dependence and the testing– estimation separation. Held-out families of frozen classifiers and VLM-judge confidence gates exhibit feasible-mixture hiding; on three VLM judges 28/30 audit-A midpoint contracts remain hidden on untouched audit-B, while a clean multi-annotator control retains positive but much smaller windows. Separate frozen-classifier audits show boundary behavior consistent with exact IUT level and strong finite-sample power against Bonferroni, DKW, total-loss and a valid non-EFC universal-inference audit.

For practitioners the message is short. Use the exact IUT: it needs no correction, runs in $O ( M K n )$ time over M score cells, and is minimax up to logarithms in the regime of Theorem 1. Within that regime, budget for the menu with (1): richness is cheap at high coverage floors and the menu penalty grows as the floor decreases. The held-out classifier and VLM-judge studies show why a family-level audit matters in practice: multiple individually failing gates can still hide the reference through feasible mixtures, while the constant-risk square-root shortcut requires additional structure.

Positioning and contributions. Classical analyses of the IUT concern a fixed number of normal means (Lehmann, 1952; Berger, 1989); tests of many moment inequalities and reality checks have an intersection null and pay only log K (White, 2000; Chernozhukov et al., 2019); property testing of function classes separates known-distribution from distribution-free testing (Kearns and Ron, 2000;

Table 1: Minimax sample sizes of valid audits (up to constants; $\tilde { \Omega } , \tilde { O }$ hide logarithms). Row 1: robustly infeasible alternatives $\mathcal { P } _ { \gamma }$ (Section 3); the interval bounds need large ${ \bf \bar { \Psi } } _ { K } ,$ , an interior risk target α bounded away from 0 and 1, and $\gamma \lesssim \operatorname* { m i n } \{ \kappa , 1 - \kappa \} /$ log K. Rows 2–4: the sign families of Section 4 $( \kappa = 1 / \bar { 2 } , M = 2 B$ cells, $k = B$ in rows 3–4, large $B , \gamma \leq \gamma _ { 0 } )$ . “Valid” means level δ over the unrestricted null.
<table><tr><td>Known to auditor</td><td>Power required against</td><td>Minimax sample size</td><td>EFC certificates</td></tr><tr><td>Menu only</td><td> $\mathcal { P } _ { \gamma }$  (robust margin  $\gamma )$ </td><td>K intervals, fixed κ:  $\tilde { \Theta } \big ( \frac { K ( 1 - \kappa ) } { \kappa \gamma ^ { 2 } } \big )$  (Thm. 1)</td><td> $\begin{array} { r } { \mathrm { I U T } ; \tilde { O } \big ( \frac { d ( 1 - \kappa + \gamma ) } { \kappa \gamma ^ { 2 } } \big ) } \end{array}$  , VC dim. d</td></tr><tr><td>Reference  $P _ { 0 }$ </td><td> $P _ { 0 }$ </td><td> $\Theta ( \sqrt { k } / \gamma ^ { 2 } ) ( \mathrm { T h m } . 2 )$ </td><td> $\Omega ( k / \gamma ^ { 2 } ) ; \mathrm { I U T } ; \Theta ( k / \gamma ^ { 2 } )$   $\left( \mathrm { T h m } . { 3 } \right)$ </td></tr><tr><td>Cells only</td><td>constant risk, any marginal</td><td> $\Theta ( \sqrt { B } / \gamma ^ { 2 } ) \operatorname { i f } \gamma \leq ( 2 B ) ^ { - 1 / 4 }$  (Thm. 4)</td><td> $\Omega ( B / \gamma ^ { 2 } ) \mathrm { a t } P _ { 0 }$ </td></tr><tr><td>Cells and marginal</td><td>heterogeneous risk</td><td> $\tilde { \Theta } ( B / \gamma ^ { 2 } ) \left( \mathrm { T h m } . 5 \right)$ </td><td> $\operatorname { I U T } \colon O ( B / \gamma ^ { 2 } )$ </td></tr></table>

Blais et al., 2021). Our null is a union of convex sets of laws with a ratio-type risk constraint, validity is required over all of it, and the question is how the sample size depends on the menu. The lowerbound tools—mixture second moments, identity and independence testing, moment matching—are classical (Ingster, 1987; Valiant and Valiant, 2017; Diakonikolas and Kane, 2016; Cai and Low, 2011). Our contributions are the EFC barrier, the abstention-odds law, and three constructions they require: a padded moment-matching prior whose null draws stay existentially feasible while the alternative draws are robustly infeasible against every measurable set at any prescribed floor; a witness that controls the coverage screen and the risk simultaneously; and a localization of the IUT to the smaller side of the floor.

## 2 THE AUDIT PROBLEM

Observe i.i.d. pairs (S, L) of score and binary loss. Freeze a measurable family A of acceptance rules (the menu) and a contract $( \kappa , \alpha )$ independently of the audit sample. A law Q is feasible if

$$
\exists A \in { \mathcal { A } } : \quad c _ { Q } ( A ) = Q ( S \in A ) \geq \kappa , \qquad r _ { Q } ( A ) = Q ( L = 1 \mid S \in A ) \leq \alpha .\tag{2}
$$

Let $\mathcal { H } ( A )$ be this existential null. It is unrestricted: score probabilities and conditional losses may both change. For a class $\mathcal { P }$ of infeasible laws, the minimax sample size is

$$
N ( \mathcal { P } , \mathcal { H } ; \delta , \beta ) = \operatorname* { i n f } \Big \{ n \geq 1 : \exists \phi _ { n } \in [ 0 , 1 ] , \operatorname* { s u p } _ { Q \in \mathcal { H } } \mathbb { E } _ { Q ^ { n } } \phi _ { n } \leq \delta , \operatorname* { i n f } _ { P \in \mathcal { P } } \mathbb { E } _ { P ^ { n } } \phi _ { n } \geq \beta \Big \} .\tag{3}
$$

$N _ { \mathrm { I U T } }$ is the same quantity with $\phi _ { n }$ restricted to the exact IUT below. For a single law $P _ { 0 }$ we write $N ( P _ { 0 } , { \mathcal { H } } )$ ; the optimal test may then use $P _ { 0 }$ , so this is an information benchmark as in local minimax testing (Balakrishnan and Wasserman, 2019). Larger classes $\mathcal { P }$ encode what the auditor does not know.

The exact IUT. For a rule A, let $T _ { A }$ and $E _ { A }$ count the audit observations it accepts and the accepted errors; $T _ { A } \sim \mathrm { B i n } ( n , c ( A ) )$ and $E _ { A } \mid T _ { A } = t \sim \mathrm { B i n } ( t , r ( A ) )$ ). Let $U _ { A }$ be the one-sided Clopper– Pearson upper bound for coverage at error $\zeta _ { c }$ and $L _ { A }$ the lower bound for risk at error $\zeta _ { r }$ (Clopper and Pearson, 1934), and define the certifiedfrontier

$$
L _ { \mathrm { I U T } } ( A , \kappa ) = \operatorname* { m i n } _ { A : U _ { A } \geq \kappa } L _ { A } , \qquad \operatorname* { m i n } \mathcal { O } = + \infty .\tag{4}
$$

It reads: with confidence $1 - \delta$ , no permitted rule of coverage at least κ has risk below $L _ { \mathrm { I U T } }$

Proposition 1 (Validity without correction). Rejectingfeasibility when $L _ { \mathrm { I U T } } > \alpha$ has level at most $\zeta _ { c } + \zeta _ { r } \leq \delta$ over the unrestricted null.

Under the null a fixed population witness $A ^ { * }$ exists, and rejection requires failure of one of its two bounds. For unions of at most K intervals over M frozen score cells a dynamic program computes (4) exactly in $O ( M K n )$ time (Appendix G). We use $\zeta _ { c } = \zeta _ { r } = \delta / 2$

Plug-in (EFC) certificates. Call A empiricallyfeasible if $T _ { A } \geq \kappa n$ and $E _ { A } \leq \alpha T _ { A }$

Definition 1 (EFC audits). An audit $\phi _ { n }$ is empirical-feasibility consistent (EFC) if $\phi _ { n } = 0$ whenever some $A \in { \mathcal { A } }$ is empirically feasible.

EFC audits never reject a claim that the vendor can apparently meet on the audit data itself, which makes their decisions easy to explain. Every certificate that screens with a coverage upper bound no smaller than the empirical coverage and rejects via risk lower bounds no larger than the empirical risk is EFC. This includes the IUT (for $\zeta _ { c } , \zeta _ { r } \leq 1 / 2 )$ , its Bonferroni variant, DKW and Rademacher uniform-convergence certificates, and the total-loss certificate (Appendix B.1). EFC plays the role of monotonicity in classical IUT theory (Perlman and Wu, 1999).

## 3 THE ABSTENTION-ODDS LAW

For an arbitrary menu we place the power requirement on alternatives that are robustly infeasible:

$$
\mathcal { P } _ { \gamma } ( \mathcal { A } ) = \{ P : \ r _ { P } ( A ) \geq \alpha + \gamma \mathrm { ~ f o r ~ e v e r y ~ } A \in \mathcal { A } \mathrm { ~ w i t h ~ } c _ { P } ( A ) \geq \kappa - \gamma \} .
$$

The class allows arbitrary score marginals and arbitrary, including informative, risk functions. The coverage slack $\gamma$ excludes rules that sit just below the floor with very low risk, which no finite audit can resolve.

Proposition 2 (The IUT for every menu). Let A have VC dimension d, $0 < \delta < \beta < 1 , \kappa \in ( 0 , 1 ] ,$ $0 < \gamma \le \kappa / 2$ and $\alpha + \gamma \leq 1$ . For absolute constants $C \colon ( i )$ the exact IUT has power at least $\beta$ uniformly over $\mathcal { P } _ { \gamma } ( \mathcal { A } )$ once $n \geq C \{ d + \log ( 2 / \delta ) + \log ( 4 / ( 1 - \beta ) ) \} / ( \kappa \gamma ) ^ { 2 } ; ( i i )$ if some $A _ { 0 } \in { \mathcal { A } }$ has coverage at least κ under some score law, every valid audit needs $n \ge 2 ( \beta - \delta ) ^ { 2 } \alpha ( 1 - \alpha ) / \gamma ^ { 2 } ,$ (iii) localization: the exact IUT has power at least $\dot { \boldsymbol { \beta } }$ over $\mathcal { P } _ { \gamma } ( \mathcal { A } )$ whenever

$$
n \geq C \frac { ( 1 - \kappa + \gamma ) d \log ( 2 e n / d ) + \log \{ 8 / ( \delta ( 1 - \beta ) ) \} } { \kappa \gamma ^ { 2 } } .
$$

Part (i) combines VC uniform deviations with Clopper–Pearson inversion; part (ii) is a two-point bound (Appendix B.3). Part (iii) is the refinement that matters at high floors. A rule that passes the coverage screen differs from the whole score line only on a region of probability at most $1 - \kappa + \gamma$ and it accepts at least a fraction $\kappa / 2$ of the data. Writing its error mass as the overall error mass minus the error mass of the abstained region, only the latter varies over the menu. Relative (Bernstein-type) VC deviations bound it by $\sqrt { ( 1 - \kappa + \gamma ) d / n }$ up to logarithms, which for $\kappa \geq 1 / 2$ is also the order of the risk error. When $\kappa < 1 / 2$ the accepted region is the small side, its error mass fluctuates by $\sqrt { \kappa d / n }$ , and dividing by κ gives $\scriptstyle { \sqrt { d / ( \kappa n ) } } ;$ ; both cases are $\sqrt { ( 1 - \kappa + \gamma ) d / ( \kappa n ) }$ up to a factor two. The Clopper–Pearson bounds adapt to both scales automatically, so the IUT needs no tuning.

For interval menus, the next result matches this dependence up to logarithmic factors at any fixed coverage floor.

Theorem 1 (Interval menus at any fixed coverage floor). Let $\boldsymbol { \mathcal { A } } _ { K }$ contain all unions ofat most K intervals ofthe score line, $\kappa \in ( 0 , 1 ) , a \in ( 0 , 1 / \bar { 8 } ] , \alpha \in [ 2 a , 1 - 2 a ]$ and $0 < \delta < \beta < 1$ . There are $c , c ^ { \prime } , C , K _ { 0 } > 0$ depending only on $( a , \delta , \beta )$ such thatfor $\mathbf { \bar { K } } \geq K _ { 0 }$ and $\gamma \leq c ^ { \prime }$ min{κ, 1 − κ}/ log K,

$$
\frac { c K ( 1 - \kappa ) } { \kappa \gamma ^ { 2 } \log K } \ \leq \ N ( \mathcal P _ { \gamma } ( A _ { K } ) , \mathcal H ) \ \leq \ N _ { \mathrm { I U T } } ( \mathcal P _ { \gamma } ( A _ { K } ) ) \ \leq \ \frac { C \{ K ( 1 - \kappa ) \log ( e / ( \kappa \gamma ) ) + 1 \} } { \kappa \gamma ^ { 2 } }
$$

The lower bound holds evenfor audits that know the score distribution.

Thus, for every fixed $\kappa \in ( 0 , 1 )$ in the stated small-gap regime, the IUT is minimax optimal up to logarithmic factors. Moreover, Proposition 2(i) gives $\overset { \cdot } { O } ( \breve { K } / \gamma ^ { 2 } )$ at fixed κ, leaving only the log K gap to the lower bound. Proofidea. The lower bound embeds $K - 1$ consecutive score groups, each split into two halves whose losses are $m \pm d _ { b }$ with a random sign and a random magnitude $d _ { b }$ . Under the null and alternative priors the magnitudes match their first log K even moments but differ in their mean by the best polynomial approximation error of $\sqrt { u } ,$ , so the two mixtures are indistinguishable until $n \stackrel { . } { \sim } K / ( \gamma ^ { 2 } \stackrel { . } { \log } \stackrel { . } { K } )$ (Theorem 5 below). To reach an arbitrary floor, a padding region carries the rest of the score mass: for $\kappa \geq 1 / 2$ it has risk α and is accepted by every good rule, so only a fraction $2 ( 1 - \kappa )$ of the data is informative and the risk gap shrinks by $( 1 - \dot { \kappa } ) \bar { / } \kappa ;$ for $\kappa < 1 / \dot { 2 }$ it has risk one and is never accepted. The delicate step is robust infeasibility: under the alternative prior every measurable set of coverage at least $\kappa - \gamma$ must have risk at least $\alpha + \gamma$ , not only the menu’s rules. Because the conditional risk is piecewise constant, the least-risk sets fill the score line in increasing order of risk, and trimming them changes the risk by at most $2 \gamma D / \kappa ( \mathrm { A }$ ppendix B.4).

Reading the law. The menu cost $K ( 1 - \kappa ) / ( \kappa \gamma ^ { 2 } )$ is a cost ofpower, not of validity: the level proof pays nothing for $| { \mathcal { A } } |$ , yet a richer menu gives the vendor more ways to place the abstained region where the audit sample happens to contain few errors. Part (i) holds for any uniform-convergence certificate with radius of order ${ \sqrt { d / n } } ;$ the endpoint DKW certificate (radius of order $K / \sqrt { n } )$ and policywise Bonferroni (of order $\sqrt { K }$ log $\overline { { M / n } }$ over M cells) have weaker guarantees, and part (iii) uses the Clopper–Pearson bounds’ adaptation to small proportions. The logarithmic gaps and constants remain open; at $\kappa = 1 / 2$ , with the constants of our proof, the new lower bound exceeds the elementary $\Omega ( \sqrt { K } / \gamma ^ { 2 } )$ obstruction from Proposition 3 only for small $\gamma$ and large $K$ . The next two sections explain why the cost is linear in K and when an audit can avoid it.

## 4 A TESTING–ESTIMATION GAP FOR EXISTENTIAL AUDITS

Sign families. Fix $p \in ( 0 , 1 ) , 0 < \gamma < \operatorname* { m i n } \{ p , 1 - p \}$ and $\left( \kappa , \alpha \right) = \left( 1 / 2 , p - \gamma \right)$ . The reference $P _ { 0 }$ has $S \sim \mathrm { U n i f } [ 0 , 1 ]$ and $L \mid S \sim$ Bern $\mathsf { \Omega } _ { \mathsf { l } } ( p )$ ; it is infeasible. For positive weights w in the simplex $\Delta _ { B - 1 }$ , partition [0, 1] into groups $G _ { \ell }$ of mass $w _ { \ell } ,$ , each split into equal halves. Each $\theta \in \{ - 1 , + \dot { 1 } \} ^ { B }$ accepts one half per group, giving $2 ^ { B }$ rules $A _ { \theta } . \mathrm { A }$ feasible component keeps the score law and sets

$$
Q _ { \theta } ( L = 1 \mid S = x ) = p - \gamma s _ { \theta } ( x ) , \qquad s _ { \theta } ( x ) = 1 { \mathrm { ~ o n ~ } } A _ { \theta } , \ - 1 { \mathrm { ~ o f f ~ } } A _ { \theta } .\tag{5}
$$

Every component has coverage $1 / 2 .$ , accepted risk $p - \gamma$ , and the same distance from $P _ { 0 } ;$ mixing them uniformly gives $P _ { 0 }$ exactly. All positive weight profiles share the same single-rule testing problems, incidence structure, VC dimension B and growth function (Propositions $6 { - } 7 ) \div$ ; the families differ only in how their rules overlap. For the plateau profile with k heavy groups, $w _ { \ell } = ( 1 - B ^ { - 2 } ) / k$ for $\ell \leq k$ and the remaining mass $B ^ { - 2 }$ spread over the other groups, the effective dimension is $d _ { \mathrm { e f f } } = \overline { { \Vert } } w \Vert _ { 2 } ^ { - 2 } = k \{ 1 + o ( 1 ) \}$

Theorem 2 (Testing scale). Fix $p \in ( 0 , 1 )$ and $0 < \delta < \beta < 1 . \ I f \gamma = \gamma _ { B } < \mathrm { m i n } \{ p , 1 - p \}$ and $B ^ { - 2 } \leq \gamma _ { B } / 1 6 ,$ , then as $B  \infty ,$ uniformly in $k \leq B$ , the full-null sample size at the reference is $N ( P _ { 0 } , \mathcal { H } _ { B , k } ; \delta , \beta ) = \Theta ( \sqrt { k } / \gamma _ { B } ^ { 2 } )$

The lower bound is Ingster’s second-moment method. For a prior Π over feasible components with $\begin{array} { r } { \mathbb { E } _ { \Pi } Q = P _ { 0 } , 1 + \chi ^ { 2 } ( \mathbb { E } _ { \Pi } Q ^ { n } \| P _ { 0 } ^ { n } ) = \mathbb { E } _ { Q , Q ^ { \prime } \sim \Pi } [ 1 + \Xi ( Q , Q ^ { \prime } ) ] ^ { n } \overset { \cdot } { \mathrm { w i t h } } \Xi = \int ( d Q / d P _ { 0 } ) ( d Q ^ { \dagger } / d P _ { 0 } ) d P _ { 0 } - } \end{array}$ 1 (Lemma 3); here $\begin{array} { r } { \Xi ( \theta , \theta ^ { \prime } ) = \lambda \sum _ { \ell } w _ { \ell } \theta _ { \ell } \theta _ { \ell } ^ { \prime } } \end{array}$ with $\lambda = \gamma ^ { 2 } / [ p ( 1 - p ) ]$ , which gives:

Proposition 3 (Weight-profile obstruction). Every level-δ audit of a sign family has $\mathbb { E } _ { P _ { 0 } ^ { n } } \phi \ \leq$ $\begin{array} { r } { \delta + \frac { 1 } { 2 } \{ \exp ( n ^ { 2 } \lambda ^ { 2 } \| w \| _ { 2 } ^ { 2 } / 2 ) - 1 \} ^ { 1 / 2 } } \end{array}$ , hence $N _ { w } = \Omega ( \sqrt { d _ { \mathrm { e f f } } } / \gamma ^ { 2 } )$

For the upper bound, every law of the full null—whether it changes scores, labels or both—is $\Omega ( \gamma )$ )-far from $P _ { 0 }$ after pushing forward to the $( 4 k + 2 )$ half/label cells of the heavy groups, so identity testing needs $O ( \sqrt { k } / \gamma ^ { 2 } )$ samples (Valiant and Valiant, 2017) (Appendix D).

Theorem 3 (The estimation cost of EFC audits). Fix $p \in ( 0 , 1 )$ and $0 < \delta < \beta < 1$ . There are $\gamma _ { 0 } , c > 0$ depending only on $( p , \delta , \beta )$ , andfor each split below a $C > 0 ,$ such that on the plateaus ofTheorem 2, whenever $\gamma \leq \gamma _ { 0 }$ and $B ^ { - 2 } \leq \gamma / 1 6 \colon \hat { ( i ) }$ every level-δ EFC audit has power below $\beta$ at $P _ { 0 } f o r$ all $n \leq c k / \gamma ^ { 2 } ; ( i i )$ the exact IUT with any fixed split $\zeta _ { c } + \zeta _ { r } = \delta , \zeta _ { c } , \zeta _ { r } \in ( 0 , 1 / 2 ]$ , has power at least $\beta f o r$ all $n \geq C k / \gamma ^ { 2 }$ . Hence $N _ { \mathrm { I U T } } = \Theta ( k / \gamma ^ { 2 } )$ and $N _ { \mathrm { I U T } } / N ( P _ { 0 } , \mathcal { H } _ { B , k } ) = \Theta ( \sqrt { k } )$

Mechanism. In each heavy group choose the half containing more correct audit observations. Conditional on the correct observations and the error count, this rule accepts each error independently with probability $1 / 2 .$ , while its accepted correct count exceeds half of all correct observations by $G$ with $\mathbb { E } G \gtrsim$ min $\{ n , { \sqrt { n k } } \}$ . For $n \ll k / \gamma ^ { 2 }$ this gain overwhelms the population gap nγ and keeps the empirical coverage at least $1 / 2 { : }$ a sample-dependent rule is empirically feasible, so no EFC audit rejects (Appendix C.2). Choosing the half with fewer errors would not control coverage. The optimal audit instead aggregates quadratic evidence whose fluctuation is $\sqrt { k } / n$ against a signal $\gamma ^ { 2 }$ . The EFC bound does not rely on the reference coverage sitting exactly at the floor: with $q \mathrm { . }$ -ary groups and reference coverage $m / q > \kappa$ it persists (Proposition 4, Appendix C.3).

Remark 1 (The barrier on interval menus). Order the 2K halves of the uniform sign family $( k = B =$ $K )$ along the score line, each group’s halves adjacent. Unions of at most K intervals of these cells contain every sign rule, so Theorem $3 ( \mathrm { i } )$ applies: for $\gamma \leq \gamma _ { 0 }$ , every valid EFC audit of this interval menu needs $\bar { \Omega } ( \breve { K } / \gamma ^ { 2 } )$ samples at $P _ { 0 }$ , without a logarithm, and the IUT needs $O ( K / \gamma ^ { 2 } )$ . Theorem 4 below gives a valid non-EFC audit with $O ( \sqrt { K } / \gamma ^ { 2 } )$ samples at the same law for $\gamma \le ( 2 K ) ^ { - 1 / 4 }$

## 5 WHEN CAN THE GAP BE ESCAPED?

The optimal audit of Theorem 2 knows $P _ { 0 } ;$ ; the IUT knows nothing about it. Which missing knowledge forces the linear cost? Validity is always required over the unrestricted null; only the power requirement is placed on a class $\mathcal { P } _ { \cdot }$ . Let $\dot { C } _ { 1 } , \dot { \ldots } , \bar { C } _ { M }$ be score cells frozen independently of the audit sample and A any family of unions of cells.

Unknown score distribution, constant risk: a reference-free testing-scale audit. Let $\mathcal { P } _ { \mathrm { c } } ( \alpha , \gamma )$ contain every law under which L is independent of S and $P ( L = 1 ) \geq \alpha + \gamma$ , with an arbitrary score marginal.

Theorem 4 (Reference-free audit at the testing scale). Fix $\kappa \in ( 0 , 1 ]$ and $0 < \delta < \beta < 1$ . An audit that uses only cell labels and losses, and knows neither the score marginal nor $P ( L = 1 )$ , has level δ over $\mathcal { H } ( A )$ and power at least β uniformly over $\mathcal { P } _ { \mathrm { c } } ( \alpha , \gamma )$ for every $n \geq C _ { \delta , \beta } \{ \sqrt { M } / ( \kappa \gamma ) ^ { 2 } +$ $M ^ { 2 / 3 } / ( \kappa \gamma ) ^ { 4 / 3 } \}$ . On the uniform sign family $( M = 2 k _ { \mathrm { { ; } } }$ $\kappa = 1 / 2 )$ with $\gamma \leq ( 2 k ) ^ { - 1 / 4 }$ this is $\Theta ( \sqrt { k } / \gamma ^ { 2 } )$ , while every valid EFC audit needs $\Omega ( k / \gamma ^ { 2 } )$ at $P _ { 0 } \in \mathcal { P } _ { \mathrm { c } }$

The audit rejects when the empirical loss rate exceeds $\alpha + \gamma / 2$ and an independence tester for cell and loss (Diakonikolas and Kane, 2016) finds no dependence. The key fact is thatfeasibilityforces dependence: if Q has a witness A and $e = Q ( L = 1 ) > \alpha$ , every product law R on cells and labels has $\mathrm { T V } ( \bar { Q , } R ) \geq ( e - \alpha ) \kappa / 3$ , because a product law assigns the witness the overall loss rate (Lemma 2). The audit is valid even for the family of all $2 ^ { M }$ unions of cells, so it matches the testing scale only where that scale is $\sqrt { M } / \gamma ^ { 2 } ;$ ; for small menus the IUT is better. It abstains whenever its independence test detects score–loss dependence. Section 6 separately reports strong score–loss dependence and held-out family-level hiding on frozen model outputs. The next result shows that, over a rich heterogeneous class, no valid audit can uniformly retain the square-root rate.

Unknown heterogeneous risk: the linear cost can be necessary. Consider the uniform sign family with B groups, $\kappa = 1 / 2$ and the uniform score marginal, known to the auditor. For $\eta \in [ 0 , 1 ] ^ { 2 B }$ let $P _ { \eta }$ have loss probability $\eta _ { b , h }$ on half h of group $b ;$ the best risk is $\begin{array} { r } { r ^ { * } ( \eta ) = B ^ { - 1 } \sum _ { b } \operatorname* { m i n } \{ \bar { \eta } _ { b , + } , \bar { \eta } _ { b , - } \} } \end{array}$ Let $\mathcal { P } _ { \mathrm { h e t } } ( \alpha , \gamma ) = \{ P _ { \eta } : \eta \in [ a , 1 - a ] ^ { 2 B } , \ r ^ { * } ( \eta ) \ge \alpha + \gamma \}$

Theorem 5 (The price of unknown heterogeneity). Fix $a \in ( 0 , 1 / 8 ] , \alpha \in [ 2 a , 1 - 2 a ]$ and $0 ~ <$ $\delta < \beta < 1$ . There are $c _ { 1 } , c _ { 2 } , C , B _ { 0 } > 0$ depending only on $( a , \delta , \beta )$ such that for $B \ \geq B _ { 0 }$ and $\gamma \leq c _ { 1 } /$ log $B , \ c _ { 2 } B / ( \gamma ^ { 2 } \log B ) \leq N ( \mathcal { P } _ { \mathrm { h e t } } , \mathcal { H } ) \leq \bar { N } _ { \mathrm { I U T } } ( \mathcal { P } _ { \mathrm { h e t } } ) \leq C B / \gamma ^ { 2 }$ , whereas $N ( P , { \mathcal { H } } ) =$ $O ( \sqrt { B } / \gamma ^ { 2 } )$ for every single $P \in \mathcal { P } _ { \mathrm { h e t } }$

The null and alternative priors set $\eta _ { b , \pm } ~ = ~ m \pm s _ { b } d _ { b }$ with a uniform sign $s _ { b }$ and a magnitude $d _ { b } \in [ 0 , D ]$ from $\nu _ { 0 } \mathrm { ~ o r ~ } \nu _ { 1 }$ . These match $\mathbb { E } d ^ { 2 j }$ for $j \leq L \asymp$ log B but differ in Ed, and hence in $r ^ { * }$ by twice the best degree-L approximation error of $\sqrt { u } \mathrm { o n } \ [ 0 , D ^ { 2 } ]$ , of order $D / L$ (Bernstein, 1914; Lepski et al., 1999; Cai and Low, 2011). Under Poissonization the random signs make each group’s $\chi ^ { 2 }$ series depend only on the even moments, so all terms up to order L cancel (Appendix B.5). Unlike ℓ -functional estimation (Jiao et al., 2018), the priors must keep the null draws existentially feasible. Together, Theorems 4 and 5 show that validity alone does not force the linear cost, while uncertainty about where the classifier errs can force it on a rich heterogeneous class.

## 6 EXPERIMENTS

## 6.1 THE ABSTENTION-ODDS LAW

We audit menus of $K \in \{ 1 , 2 , 4 , 8 , 1 6 \}$ intervals over $M = 6 4$ equal-mass score cells at floors $\kappa \in \{ . 5 , . 7 , . 8 , . 9 , . 9 5 \}$ (and .3 in Appendix H.4). The alternative has constant risk $\alpha + \gamma \left( \alpha = . 1 0 \right.$

$\gamma = . 0 4 )$ , so every rule has the same risk and the law lies in $\mathcal { P } _ { \gamma }$ at every floor; any increase of the sample size with K is therefore a pure cost of menu richness. For each $( K , \kappa )$ the exact IUT audits 250 samples at each of 15 log-spaced sizes between 150 and 15,846, and we record the 80%-power crossing $n _ { 8 0 }$ and the menu cost $n _ { 8 0 } ( K ) - n _ { 8 0 } ( 1 )$

Figure 1 illustrates the coverage-floor dependence in Proposition 2(iii) and Theorem 1. The raw menu cost varies by almost an order of magnitude across floors. The right panel additionally divides by the theorem-motivated localized-VC logarithmic factor discussed in Appendix B.3; that factor is not used in the proofs, and the resulting variation is substantially reduced until the menu saturates. The flattening itself is the localization mechanism at work: only rules that differ on the small abstained region compete. The IUT’s level is not at issue here (it is exact); boundary checks are reported below.

## 6.2 SIGN FAMILIES: THE IUT AGAINST THE INFORMATION ENVELOPE

We fix $B = 2 5 6 ( 2 ^ { 2 5 6 }$ rules), $p = . 2 5 , \kappa = . 5 , k \in \{ 1 , 4 , 1 6 , 6 4 , 2 5 6 \}$ and $\gamma \in \lbrace . 0 4 , . 0 2 \rbrace$ . The exact dynamic program audits $1 { , } 5 0 0$ samples per grid point. The comparator is the Neyman–Pearson curve for $P _ { 0 } ^ { n }$ against the uniform feasible mixture; its population version bounds the power of every valid audit, so it is an information envelope, not an audit. $\mathbf { A t } \gamma = . 0 4$ the IUT’s crossing rises from 2,342 to $3 2 { , } 7 9 6$ samples as k grows from 1 to $2 5 6 ,$ , and the envelope’s from 1,001 to $^ { 7 , 3 8 4 }$ ; halving the gap multiplies all crossings by 3.91–4.17. Fitting $\gamma ^ { 2 } n _ { 8 0 } \approx a _ { 0 } + a _ { 1 } \sqrt { k } + a _ { 2 } k$ gives $a _ { 2 } \approx 0 . 1 3$ for the IUT at both gaps and $a _ { 2 } \approx 0$ for the envelope, as Theorems $_ { 2 - 3 }$ predict. Ten boundary-null checks, one per $( k , \gamma )$ with 20,000 audits each, give rejection rates of 0–4.7%. Across the simulated k values the gap to the envelope is a factor of 2.2–4.4. In the separate fixed-degree $( L = 2 )$ heterogeneity-envelope experiment, the heterogeneous envelope overtakes the known-reference one only at $B \approx 2 . 6 \times 1 \bar { 0 } ^ { 5 }$ (Appendix B.7).

## 6.3 AUDITING THE AUDITORS: VLM-JUDGE RELIABILITY

Automated VLM judges are increasingly used to automate evaluation, and a reliability claim is naturally existential: some confidence gate should retain enough evaluations while keeping serious judge errors below a target. We use the public score-token log-probabilities and human scores released by Kumar et al. (2026). For each item let $L = { \bf 1 } \{ | \hat { y } - y _ { \mathrm { h u m a n } } | > 1 \}$ , matching the source study’s ±1 scoring criterion; a non-1–5 parse is counted as a serious error, with a drop-invalid sensitivity analysis in the Supplementary Note. From the five score-token probabilities we predeclare three label-free confidence channels—maximum probability, negative entropy and top-two margin—and cross each with target coverages .65 and .90, giving six gates at contract floor $\kappa = . 6 0$

For each judge we use ten deterministic three-way resplits. Thresholds are frozen from calibration scores only; audit-A selects the midpoint of its exact fixed-marginal hiding window $I = [ \alpha _ { H } , \alpha _ { \mathrm { i n d } } ]$ and untouched audit-B tests transfer. Thus $W _ { \mathrm { h i d e } } > 0$ means no individual gate is feasible at some target even though feasible alternatives can convexify to the empirical reference. Table 2 reports the result. On MLLM-Judge, every one of the 60 split-specific windows is positive and 28/30 midpoint contracts transfer. To test whether this is merely noisy single-annotator supervision, we repeat the same LLaVA-Critic protocol on Polaris, where the released target averages at least three annotators and the same judge is much more accurate (Kumar et al., 2026). All 20 windows remain positive, but the median held-out width falls from 4.97 to 0.33 percentage points and midpoint transfer falls from 10/10 to 3/10. On a representative Polaris audit-B split, singleton families have zero window, 60% of two-gate and 90% of three-gate subfamilies hide, and every subfamily of size at least four hides (Supplementary Note). The mechanism therefore persists under cleaner supervision, while its magnitude and split stability are population dependent. These are finite-target mechanism diagnostics, not an instance-specific minimax verification of Theorem 5.

## 6.4 FROZEN CLASSIFIER CERTIFICATES AND FAMILY GEOMETRY

We also treat frozen classifier outputs as populations: CIFAR-10/100 ResNet-18, ImageNetV2 ViT-B/16 and CMU-MOSEI MLP outputs (Krizhevsky, 2009; Recht et al., 2019; Bagher Zadeh et al., 2018). Calibration fixes score cells and the audit half supplies labels. $\mathrm { A t } \ \kappa = . 9$ , the exact IUT has no observed boundary rejection above 1.8% and is the most powerful certificate compared except for one $n = 5 0 0 { \mathrm { C I F A R } } { - } 1 0 0$ point within Monte Carlo error; Bonferroni needs several times more data, structured DKW is vacuous here, and a score-agnostic total-loss certificate fails when total error is small relative to $1 - \kappa$ (Table 8, Appendix H.2).

![](images/5f639ce08251ad00616e0594617ebcc69ba386c3d5e6e8c6633a6198d96f8db4.jpg)

![](images/ba35db0e22b6050dff21d7db5592df9b0192940b7c9a93c0349b08e7668dd6de.jpg)  
Figure 1: Menu richness costs K times the abstention odds. Left: menu cost of the exact IUT, scaled by $\gamma ^ { 2 }$ , for five coverage floors. At K = 4 it falls 7.7-fold from $\kappa = . 5$ to $\kappa = . 9 5$ . Right: the same costs divided by $\begin{array} { r } { \frac { 1 - \overline { { \kappa } } } { \kappa } \log \frac { e } { \operatorname* { m i n } \{ \kappa , 1 - \kappa \} } } \end{array}$ lie within 2.32–2.82 at $K = 4$ and within 0.95–1.39 at K = 2. $\mathrm { ~ A t ~ } \kappa \geq . 9$ the curves flatten for $K \ge 8$ because the abstained region spans only $M ( 1 - \kappa ) \leq 6 . 4 $ cells, so extra intervals do not enlarge the realized admissible rule set in this finite benchmark. Monte Carlo crossings from 250 audits per point.

![](images/d92bc1a978b44dfd2764cf711222d2154a21721c8324a924c0a16b8379157072.jpg)

![](images/69376fb1a82e109fd7e948c40349c70abb733feb57a953e155be80514d1fae7d.jpg)  
Figure 2: The IUT and the information envelope have different scales $( B = 2 5 6 )$ . Left: $\gamma ^ { 2 } n _ { 8 0 }$ at two gaps; the curves nearly coincide, as $\gamma ^ { - 2 }$ scaling predicts. Right: local log–log slopes in k. The $\mathrm { I U T } \mathbf { \bar { s } }$ slope rises from 0.25 to 0.72 towards the proven exponent 1, the envelope’s from 0.28 to 0.43 towards $1 / 2 ;$ the range is pre-asymptotic. Crossings are in Table 7 (Appendix H.1).

The constant-risk premise of Theorem 4 is strongly contradicted: a cell–loss independence diagnostic rejects in every one of 200 audits of size 500 on each population. More directly, a separate six-policy, three-way held-out geometry experiment freezes gates from calibration scores only. All 60 audit-A and all 60 audit-B windows are positive; 47/60 midpoint contracts selected on A transfer to untouched B. Singleton subfamilies have zero hiding, 96.7% of pairs hide, and every subfamily of size at least three hides (Supplementary Note, Section S5). Thus the same feasible-mixture mechanism appears on standard classifier families, without claiming that these populations instantiate Theorem 5’s asymptotic hard class.

Finally, as a valid non-EFC reference-free competitor we implement cross-fit universal inference (Wasserman et al., 2020). It has no observed boundary rejections, but at $\Delta = . 0 5$ its first 80%-power grid point is $n = 1 0 , 0 0 0 { - } 5 0 , 0 0 0$ , versus 500–2,500 for the IUT; at $\Delta = . 0 2$ it does not reach 80% power by 50,000 (Table 12, Appendix H.5). This evaluates one practical non-EFC construction, not all non-EFC audits.

Table 2: Auditing the auditors. Held-out exact-hiding windows for six predeclared confidence gates at $\kappa = . 6 0$ and serious judge error $L = { \bf 1 } \{ | \hat { y } - y _ { \mathrm { h u m a n } } | > 1 \}$ . Policies are frozen on a disjoint calibration third. “pos. $\mathbf { A } / \mathbf { B } ^ { \ast }$ counts positive windows over audit-A and audit-B; $\mathrm { \ddot { A } \vec { \Delta } \vec { \Delta } \vec { B } \vec { \Delta } }$ asks whether the midpoint contract selected on A remains hidden on untouched B. Widths are median audit-B windows in percentage points.
<table><tr><td>Target</td><td>judge</td><td>pos. A/B</td><td>A→B med.</td></tr><tr><td>MLLM-Judge</td><td>LLaVA-Critic-7B</td><td>20/20</td><td>10/10 4.97</td></tr><tr><td>MLLM-Judge</td><td>Gemini 2.5 Flash</td><td>20/20</td><td>10/10 5.91</td></tr><tr><td>MLLM-Judge</td><td>Phi-4 reasoning vision</td><td>20/20</td><td>8/10 4.83</td></tr><tr><td>Polaris (multi-annotator)</td><td>LLaVA-Critic-7B</td><td>20/20</td><td>3/10 0.33</td></tr></table>

## 7 RELATED WORK AND LIMITATIONS

Risk control, Learn-then-Test and Pareto testing certify selected rules (Bates et al., 2021; Angelopoulos et al., 2024; 2025; Laufer-Goldshtein et al., 2023); model falsification, subgroup auditing and PAC verification address adjacent goals (Müller et al., 2025; Cherian and Candès, 2024; Mutreja and Shafer, 2026). Intersection–union validity is classical (Berger, 1982; Berger and Hsu, 1996; Lehmann and Romano, 2005); for a fixed number of normal means the IUT is the likelihood-ratio test and non-monotone tests can be uniformly more powerful (Lehmann, 1952; Berger, 1989), at the price of non-monotone rejection regions that can yield counterintuitive inferences (Perlman and Wu, 1999). Moment-inequality and intersection-bound tests (Andrews and Soares, 2010; Chernozhukov et al., 2013; Romano et al., 2014; Bugni et al., 2015) ask whether some parameter satisfies a set of inequalities; with many inequalities, as in reality checks (White, 2000; Hansen, 2005), the cost is logarithmic in their number (Chernozhukov et al., 2019) because that null is an intersection, whereas ours is a union. Inference on winners corrects the selection bias of an empirical optimum (Andrews et al., 2024), the bias that EFC certificates do not correct; sequential tests of whether some arm lies below a threshold address a related existential question (Kaufmann et al., 2018); confidence sequences give anytime versions of the IUT (Howard et al., 2021) (Appendix G). Universal inference gives reference-free composite tests (Wasserman et al., 2020); recent work on testable hypotheses and safe anytime-valid inference provides broader characterizations and tools for composite testing (Larsson et al., 2026b; Ramdas et al., 2023). In property testing, unions of d intervals are testable from $O ( { \sqrt { d } } )$ samples when the input distribution is known but need $\Omega ( d / \log d )$ distribution-free (Kearns and Ron, 2000; Balcan et al., 2012; Blais et al., 2021); related interval and flat structure also appears in identity testing of structured distributions (Diakonikolas et al., 2015), and optimal risk can be estimated faster than learned (Kong and Valiant, 2018). Minimax $L _ { \mathrm { { 1 } } } { \mathrm { { - d i s t a n c e } } }$ estimation and tolerant distribution testing exhibit related linear-over-log sample-complexity phenomena (Jiao et al., 2018; Canonne et al., 2022). Recent work studies calibrated uncertainty for VLM judges and releases the score-token features used in our auditing application (Kumar et al., 2026). Our null is a union of convex sets of laws with ratio-type constraints, losses are noisy, and an unknown input distribution alone does not force the linear scale (Theorem 4), whereas unknown conditional risk can force it on the heterogeneous class of Theorem 5. The Supplementary Note relates the one-observation geometry to convex-set testing (Goldenshluger et al., 2015).

Limitations. Losses are binary, the contract has one coverage floor, and data are i.i.d.; distribution shift is outside our scope. For each fixed coverage floor, the interval-menu bounds are worst-case and match up to logarithms in the stated small-gap, large-K regime; the lower bound is asymptotic in $K ;$ the sharp testing–estimation spectrum and the constant-risk result are proved on sign families; an $\Omega ( d )$ bound for every VC-d family is open. Theorem 4 is an existence result with an untuned tester. The natural-family and VLM-judge studies are finite-target mechanism diagnostics, not instance-specific minimax lower bounds for Theorem 5; the six VLM gates are illustrative and use released benchmark outputs rather than production traffic. The non-EFC experiment evaluates one universal-inference construction rather than all possible non-EFC audits. Nonrejection certifies no rule.

## AI USE STATEMENT

Generative AI was used to improve the clarity and presentation of manuscript claims, assist with software implementation and debugging, and provide critical feedback on claims and arguments during manuscript development. AI-generated suggestions were treated as editorial and technical assistance rather than as evidence or independent validation. The authors are fully responsible for the final claims, proofs, code, citations, analyses, and submission.

## ETHICS STATEMENT

The empirical studies reuse frozen benchmark model outputs and released human annotations and collect no new human-subject data. The work concerns statistical auditability and does not introduce a clinical decision system or an automated evaluator for deployment. Population interpretations require a sampling model appropriate to the intended setting.

## REPRODUCIBILITY STATEMENT

The supplement contains this manuscript’s figures, analysis scripts, exact rational certificates, frozen internal inputs, and derived VLM-audit summaries; the public third-party VLM feature files are referenced by source and are not redistributed. The analyses for Section 5 and Figure 3 are provided with their tests. The release README maps claims to code and distinguishes recomputed results from retained outputs. Appendix H and the Supplementary Note give protocols and limitations.

## REFERENCES

Donald W. K. Andrews and Gustavo Soares. Inference for Parameters Defined by Moment Inequalities Using Generalized Moment Selection. Econometrica, 78(1):119–157, 2010.

Isaiah Andrews, Toru Kitagawa, and Adam McCloskey. Inference on Winners. The Quarterly Journal ofEconomics, 139(1):305–358, 2024.

Anastasios N. Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, and Tal Schuster. Conformal Risk Control. In International Conference on Learning Representations, 2024.

Anastasios N. Angelopoulos, Stephen Bates, Emmanuel J. Candès, Michael I. Jordan, and Lihua Lei. Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control. The Annals of Applied Statistics, 19(2):1641–1662, 2025. doi:10.1214/24-AOAS1998.

AmirAli Bagher Zadeh, Paul Pu Liang, Soujanya Poria, Erik Cambria, and Louis-Philippe Morency. Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph. In Proceedings ofthe 56th Annual Meeting ofthe Associationfor Computational Linguistics, pp. 2236–2246, 2018. doi:10.18653/v1/P18-1208.

Sivaraman Balakrishnan and Larry Wasserman. Hypothesis testing for densities and high-dimensional multinomials: Sharp local minimax rates. The Annals of Statistics, 47(4):1893–1927, 2019. doi:10.1214/18-AOS1729.

Maria-Florina Balcan, Eric Blais, Avrim Blum, and Liu Yang. Active Property Testing. In IEEE 53rd Annual Symposium on Foundations of Computer Science (FOCS), pp. 21–30, 2012.

Stephen Bates, Anastasios Angelopoulos, Lihua Lei, Jitendra Malik, and Michael I. Jordan. Distribution-Free, Risk-Controlling Prediction Sets. Journal of the ACM, 68(6):43:1–43:34, 2021. doi:10.1145/3478535.

Roger L. Berger and Jason C. Hsu. Bioequivalence Trials, Intersection–Union Tests and Equivalence Confidence Sets. Statistical Science, 11(4):283–319, 1996.

Roger L. Berger. Multiparameter hypothesis testing and acceptance sampling. Technometrics, 24(4):295–300, 1982. doi:10.1080/00401706.1982.10487790.

Roger L. Berger. Uniformly More Powerful Tests for Hypotheses Concerning Linear Inequalities and Normal Means. Journal ofthe American Statistical Association, 84(405):192–199, 1989.

Serge Bernstein. Sur la meilleure approximation de |x| par des polynômes de degrés donnés. Acta Mathematica, 37:1–57, 1914.

Eric Blais, Renato Ferreira Pinto Jr., and Nathaniel Harms. VC Dimension and Distribution-Free Sample-Based Testing. In Proceedings ofthe 53rd Annual ACM SIGACT Symposium on Theory of Computing (STOC), 2021.

Stéphane Boucheron, Olivier Bousquet, and Gábor Lugosi. Theory of Classification: A Survey of Some Recent Advances. ESAIM: Probability and Statistics, 9:323–375, 2005.

Federico A. Bugni, Ivan A. Canay, and Xiaoxia Shi. Specification Tests for Partially Identified Models Defined by Moment Inequalities. Journal of Econometrics, 185(1):259–282, 2015.

T. Tony Cai and Mark G. Low. Testing composite hypotheses, Hermite polynomials and optimal estimation of a nonsmooth functional. The Annals of Statistics, 39(2):1012–1041, 2011. doi:10.1214/10-AOS849.

Clément L. Canonne, Ayush Jain, Gautam Kamath, and Jerry Li. The Price of Tolerance in Distribution Testing. In Proceedings ofthe 35th Conference on Learning Theory (COLT), volume 178 of PMLR, pp. 573–624, 2022.

John J. Cherian and Emmanuel J. Candès. Statistical Inference for Fairness Auditing. Journal of Machine Learning Research, 25(149):1–49, 2024.

Victor Chernozhukov, Sokbae Lee, and Adam M. Rosen. Intersection Bounds: Estimation and Inference. Econometrica, 81(2):667–737, 2013.

Victor Chernozhukov, Denis Chetverikov, and Kengo Kato. Inference on Causal and Structural Parameters Using Many Moment Inequalities. The Review ofEconomic Studies, 86(5):1867–1900, 2019.

C. K. Chow. On Optimum Recognition Error and Reject Tradeoff. IEEE Transactions on Information Theory, 16(1):41–46, 1970.

C. J. Clopper and E. S. Pearson. The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial. Biometrika, 26(4):404–413, 1934. doi:10.1093/biomet/26.4.404.

Ilias Diakonikolas and Daniel M. Kane. A New Approach for Testing Properties of Discrete Distributions. In IEEE 57th Annual Symposium on Foundations of Computer Science (FOCS), pp. 685–694, 2016.

Ilias Diakonikolas, Daniel M. Kane, and Vladimir Nikishkin. Testing Identity of Structured Distributions. In Proceedings ofthe Twenty-Sixth Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pp. 1841–1854, 2015. doi:10.1137/1.9781611973730.123.

Ran El-Yaniv and Yair Wiener. On the Foundations of Noise-free Selective Classification. Journal of Machine Learning Research, 11:1605–1641, 2010.

Yonatan Geifman and Ran El-Yaniv. Selective Classification for Deep Neural Networks. In Advances in Neural Information Processing Systems, volume 30, 2017.

Evarist Giné and Vladimir Koltchinskii. Concentration Inequalities and Asymptotic Results for Ratio Type Empirical Processes. The Annals ofProbability, 34(3):1143–1216, 2006.

Alexander Goldenshluger, Anatoli Juditsky, and Arkadi Nemirovski. Hypothesis testing by convex optimization. Electronic Journal ofStatistics, 9(2):1645–1712, 2015. doi:10.1214/15-EJS1054.

Peter Reinhard Hansen. A Test for Superior Predictive Ability. Journal of Business & Economic Statistics, 23(4):365–380, 2005.

Steven R. Howard, Aaditya Ramdas, Jon McAuliffe, and Jasjeet Sekhon. Time-uniform, Nonparametric, Nonasymptotic Confidence Sequences. The Annals of Statistics, 49(2):1055–1080, 2021.

Yu. I. Ingster. Minimax Testing of Nonparametric Hypotheses on a Distribution Density in the $L _ { p }$ Metrics. Theory ofProbability and Its Applications, 31(2):333–337, 1987. doi:10.1137/1131042.

Jiantao Jiao, Yanjun Han, and Tsachy Weissman. Minimax Estimation of the $L _ { 1 }$ Distance. IEEE Transactions on Information Theory, 64(10):6672–6706, 2018. doi:10.1109/TIT.2018.2846245.

R. Kaas and J. M. Buhrman. Mean, median and mode in binomial distributions. Statistica Neerlandica, 34(1):13–18, 1980.

Emilie Kaufmann, Wouter M. Koolen, and Aurélien Garivier. Sequential Test for the Lowest Mean: From Thompson to Murphy Sampling. In Advances in Neural Information Processing Systems 31, 2018.

Michael Kearns and Dana Ron. Testing Problems with Sublearning Sample Complexity. Journal of Computer and System Sciences, 61(3):428–456, 2000.

Weihao Kong and Gregory Valiant. Estimating Learnability in the Sublinear Data Regime. In Advances in Neural Information Processing Systems 31, 2018.

Alex Krizhevsky. Learning Multiple Layers of Features from Tiny Images. MSc thesis, Department of Computer Science, University of Toronto, 2009.

Divake Kumar, Sina Tayebati, Devashri Naik, Ranganath Krishnan, and Amit Ranjan Trivedi. VLM Judges Can Rank but Cannot Score: Task-Dependent Uncertainty in Multimodal Evaluation. arXiv preprint arXiv:2604.25235, 2026.

Martin Larsson, Johannes Ruf, and Aaditya Ramdas. A Complete Characterization of Testable Hypotheses. arXiv preprint arXiv:2601.05217, 2026.

Bracha Laufer-Goldshtein, Adam Fisch, Regina Barzilay, and Tommi Jaakkola. Efficiently Controlling Multiple Risks with Pareto Testing. In International Conference on Learning Representations, 2023.

Lucien Le Cam and Grace Lo Yang. Asymptotics in Statistics: Some Basic Concepts. Springer, 2nd edition, 2000. doi:10.1007/978-1-4612-1166-2.

E. L. Lehmann and Joseph P. Romano. Testing Statistical Hypotheses. Springer, 3rd edition, 2005. doi:10.1007/0-387-27605-X.

E. L. Lehmann. Testing Multiparameter Hypotheses. The Annals of Mathematical Statistics, 23(4):541–552, 1952.

Oleg Lepski, Arkady Nemirovski, and Vladimir Spokoiny. On estimation of the $L _ { r }$ norm of a regression function. Probability Theory and Related Fields, 113(2):221–253, 1999.

Manuel M. Müller, Yuetian Luo, and Rina Foygel Barber. Are All Models Wrong? Fundamental Limits in Distribution-Free Empirical Model Falsification. In Proceedings of the 38th Conference on Learning Theory, volume 291 of PMLR, pp. 4271–4308, 2025.

Pascal Massart. The Tight Constant in the Dvoretzky–Kiefer–Wolfowitz Inequality. The Annals of Probability, 18(3):1269–1283, 1990. doi:10.1214/aop/1176990746.

Mehryar Mohri, Afshin Rostamizadeh, and Ameet Talwalkar. Foundations of Machine Learning. MIT Press, 2nd edition, 2018.

Saachi Mutreja and Jonathan Shafer. PAC Verification of Statistical Algorithms. Theory of Computing, 22(5):1–40, 2026. doi:10.4086/toc.2026.v022a005

Michael D. Perlman and Lang Wu. The Emperor’s New Tests. Statistical Science, 14(4):355–369, 1999.

Aaditya Ramdas, Peter Grünwald, Vladimir Vovk, and Glenn Shafer. Game-Theoretic Statistics and Safe Anytime-Valid Inference. Statistical Science, 38(4):576–601, 2023. doi:10.1214/23-STS894.

Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do ImageNet Classifiers Generalize to ImageNet? In Proceedings of the 36th International Conference on Machine Learning, volume 97 of PMLR, pp. 5389–5400, 2019.

Joseph P. Romano, Azeem M. Shaikh, and Michael Wolf. A Practical Two-Step Method for Testing Moment Inequalities. Econometrica, 82(5):1979–2002, 2014.

Gregory Valiant and Paul Valiant. An Automatic Inequality Prover and Instance Optimal Identity Testing. SIAM Journal on Computing, 46(1):429–455, 2017. doi:10.1137/151002526.

Roman Vershynin. High-Dimensional Probability: An Introduction with Applications in Data Science. Cambridge University Press, 2018.

Larry Wasserman, Aaditya Ramdas, and Sivaraman Balakrishnan. Universal inference. Proceedings of the National Academy of Sciences, 117(29):16880–16890, 2020. doi:10.1073/pnas.1922664117.

Halbert White. A Reality Check for Data Snooping. Econometrica, 68(5):1097–1126, 2000.

Yihong Wu and Pengkun Yang. Minimax rates of entropy estimation on large alphabets via best polynomial approximation. IEEE Transactions on Information Theory, 62(6):3702–3720, 2016.

## A ROADMAP AND NOTATION

Local notation. A few symbols are reused locally with meanings fixed by context: L is the loss, but the polynomial degree in Theorem 5; M is the number of cells, but the error count in the proofs of Theorem 3 and Proposition 4; m is a prior midpoint in Theorem 5 and the number of selected parts in the q-ary construction; K counts intervals, and $K ( Q , Q ^ { \prime } )$ is the overlap kernel of Lemma 3; λ is $\gamma ^ { 2 } / [ p ( \mathsf { \bar { 1 } } - \mathsf { \bar { p } } ) ]$ ] in Section 4 and a Poisson mean in Appendix B.5.

Where the main results are proved. This appendix contains everything the main theorems use. EFC certificates, Theorem 4, Proposition 2 (including the localization in part (iii)), Theorem 1 (Appendix B.4) and Theorem 5: Appendix B. Theorem 3, the Rademacher formula and Proposition 5: Appendix C. Theorem 2 and Propositions 6, 3 and 7: Appendix D. Lemma 3: Appendix E. Proposition 1 and the dynamic programs: Appendix G. The witness power ceiling (Corollary 3) and the collision bound (Theorem 6): Appendix F. Experimental protocols and all tables, including the abstention-odds simulation and the universal-inference audit: Appendix H. A separate Supplementary Note in the supplementary material collects one-observation geometry (exact projections, Hall–TV formulas, the eight-atom pairwise example and natural hiding windows) that the main theorems do not use.

<table><tr><td>Symbol</td><td>Meaning</td><td>Sample- computable?</td></tr><tr><td> $P _ { 0 }$ </td><td>fixed infeasible reference / audit alternative</td><td>reference-</td></tr><tr><td>Q A</td><td>feasible population law under the existential null</td><td>dependent no</td></tr><tr><td> $\mathcal { H } ( \mathcal { A } )$ </td><td>predeclared family of acceptance policies A laws for which at least one A ∈ A meets the contract</td><td>yes implicit</td></tr><tr><td>κ, α</td><td>coverage floor and conditional-risk target</td><td>yes</td></tr><tr><td> $\vec { R _ { n } } = \mathbb { E } _ { Q \sim \Pi } [ Q ^ { n } ]$ </td><td>product mixture induced by a prior over feasible laws</td><td>only with wit- ness</td></tr><tr><td> $\Xi ( Q , Q ^ { \prime } )$ </td><td>likelihood-ratio overlap kernel</td><td>only with compo-</td></tr><tr><td> $d _ { \mathrm { e f f } }$ </td><td>participation-ratio summary inside the matched sign construction</td><td>nents diagnostic proxy</td></tr><tr><td> $N ( P _ { 0 } , \mathcal { H } ; \delta , \beta )$ </td><td>samples needed for level δ and power β</td><td>no</td></tr><tr><td> $L _ { \mathrm { { I U T } } }$ </td><td>reference-free lower risk frontier for an existential contract</td><td>yes</td></tr><tr><td>EFC</td><td>audit that never rejects while some policy is empirically feasible</td><td>property</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td> $\mathcal { P } _ { \mathrm { c } } , \mathcal { P } _ { \mathrm { h e t } }$ </td><td>constant-risk and heterogeneous-risk alternative classes</td><td>power classes</td></tr></table>

## B EFC CERTIFICATES, REFERENCE-FREE TESTING, AND UNKNOWN HETEROGENEITY

## B.1 WHICH CERTIFICATES ARE EFC

Lemma 1 (Clopper–Pearson bounds bracket the empirical proportion). Let $X \sim \operatorname { B i n } ( t , q )$ with observed value x. For $\zeta \leq 1 / 2$ the one-sided Clopper–Pearson upper bound at error $\zeta$ is at least $x / t$ and the lower bound is at most x/t.

Proof. When the mean $t q$ is an integer, it is a median of $\mathrm { B i n } ( t , q )$ (Kaas and Buhrman, 1980). At $q = x / t$ we therefore have $P _ { q } ( X \leq \bar { x } ) \geq 1 / 2 \geq \zeta$ and $P _ { q } ( X \ge x ) \ge 1 / 2 \ge \zeta$ . The upper bound is the largest q with $P _ { q } ( X \leq x ) ^ { \ast } \geq \zeta$ , so it is at least $x / t ;$ the lower bound is symmetric. □

Let A be empirically feasible: $T _ { A } \geq \kappa n$ and $E _ { A } \leq \alpha T _ { A }$

• Exact IUT and policywise Bonferroni. By Lemma 1, $U _ { A } \geq T _ { A } / n \geq \kappa ,$ , so A passes the coverage screen, and $L _ { A } \leq E _ { A } / T _ { A } \leq \alpha$ . Hence $L _ { \mathrm { I U T } } \leq$ α and the audit does not reject. Bonferroni replaces ζ by $\zeta / | A | \leq 1 / 2$ and is covered by the same argument.

• DKW and Rademacher certificates (Appendices $\mathbf { G } . 5 \substack { - \mathbf { G } . 6 ) }$ . Their screen $c _ { n } ( A ) \geq \kappa -$ ϵ is passed by A, and their lower bound $[ m _ { n } ( A ) - \epsilon ] _ { + } / \operatorname* { m i n } \{ 1 , c _ { n } ( A ) + \epsilon \}$ is at most $m _ { n } ( { \bar { A } } ) / c _ { n } ( { \bar { A } } ) = E _ { A } / T _ { A } \leq \alpha$

• Total-loss certificate. Writing eˆ for the empirical loss rate, $\hat { e } \le ( E _ { A } + n - T _ { A } ) / n \le$ $1 - ( 1 - \alpha ) T _ { A } / n \leq 1 - \kappa ( 1 - \alpha )$ . Hence $( \hat { e } + \kappa - 1 ) / \kappa \leq \alpha ;$ its lower confidence version is smaller by Lemma 1, so it does not reject.

Part (i) of Theorem 3 therefore applies to every certificate compared in Section 6.

## B.2 PROOF OF THEOREM 4

Throughout, laws are identified with their pushforwards to $[ M ] \times \{ 0 , 1 \}$ (cell index and loss). This loses nothing for the null: every $A \in { \mathcal { A } }$ is a union of cells, so the contract depends on $Q$ only through this pushforward, and the pushforward of a feasible law is feasible for the induced family. Products on the original space push forward to products.

Lemma 2 (Feasibility forces dependence). Let $Q$ be feasible with witness $A ,$ and let $e = Q ( L =$ $1 ) > \alpha$ . Then $\operatorname { T V } ( Q , \mathbf { \tilde { \mu } } R ) \geq ( e - \mathbf { \tilde { \alpha } } \alpha ) Q ( A ) / 3$ for every product law $R = \nu \otimes \mathrm { B e r n } ( q )$ on $[ M ] \times \{ 0 , 1 \}$

Proof. Let $t = \mathrm { T V } ( Q , R )$ . If $t \geq e \ \mathrm { o r } \ t \geq Q ( A )$ the claim holds because $( e - \alpha ) Q ( A ) / 3 <$ min $\{ e , Q ( A ) \}$ . Otherwise, comparing the events $\{ L = 1 \} , \{ S \in A \}$ and $\{ S \in { \mathrm { \bf \dot { A } } } , { \mathrm { \bf \dot { L } } } { \mathrm { \bf \dot { \Phi } } } = 1 \}$ gives $q \ \geq \ e - \ t , \nu ( A ) \geq Q ( \bar { A } ) - \bar { t }$ and $q \nu ( A ) \ \stackrel { \cdot } { \ } \ { Q } ( \dot { A , } \dot { L } \ \stackrel { \cdot } { = } \ 1 ) + \stackrel { \cdot } { t } \ \leq \ \dot { \alpha } { Q } ( A ) + t .$ Thus $\alpha Q ( A ) + t \geq ( e - t ) ( Q ( A ) - t ) \geq e Q ( A ) - t ( e + Q ( A ) ) \geq e Q ( A ) - 2 t , \operatorname { s o } 3 t \geq ( e - \alpha ) Q ( A )$ . □

Independence tester. For distributions on $[ M ] \times [ 2 ]$ , Diakonikolas and Kane (2016) give a tester that, from $O ( \sqrt { M } / \epsilon ^ { 2 } + M ^ { 2 / 3 } / \epsilon ^ { 4 / 3 } )$ samples, accepts every product law and rejects every law ϵ-far in total variation from all product laws, each with probability at least $2 / 3$ (their $\ell _ { 1 }$ parameter is 2ϵ). Taking the majority of ${ \cal O } ( \bar { \log } ( 1 / \epsilon ^ { \prime } ) ) ,$ ) independent runs reduces both error probabilities to $\epsilon { ' } .$ Denote the amplified output by $\dot { \Psi } \in \{ 0 , 1 \}$ , with $\Psi = 1$ meaning “dependent”.

The audit. Let $\epsilon ^ { \prime } = \operatorname* { m i n } \{ \delta , ( 1 - \beta ) / 2 \}$ and $\epsilon = \kappa \gamma / 1 2$ . Reject feasibility if and only if $\dot { e } \geq \alpha + \gamma / 2$ and $\Psi = 0$ , where eˆ is the empirical loss rate and Ψ uses the independence tester at distance ϵ and error $\epsilon { ' } .$ Neither the score marginal nor $P ( L = 1 )$ is an input. Let $n _ { 0 }$ be the larger of $8 \log ( 1 / \epsilon ^ { \prime } ) / \gamma ^ { 2 }$ and the amplified tester’s sample size; for $n < n _ { 0 }$ the audit does not reject, and for $n \geq$ n<sub>0</sub> it applies the rule above to the first $n _ { 0 }$ observations for the tester.

Level over the unrestricted null. Let $Q \in { \mathcal { H } } ( A )$ and $e = Q ( L = 1 ) . { \mathrm { I f } } e \leq \alpha + \gamma / 4 .$ , Hoeffding’s inequality gives $Q ( \hat { e } \geq \alpha + \gamma / 2 ) \leq \exp ( - n \gamma ^ { 2 } / 8 ) \leq \epsilon ^ { \prime }$ once $\tau \geq 8 \log ( 1 / \epsilon ^ { \prime } ) / \gamma ^ { 2 }$ . If $e > \alpha + \gamma / 4$ Lemma $^ 2$ and $Q ( A ) \geq \kappa$ make $Q$ at least $\kappa \gamma / 1 2 \cdot$ -far from every product law, so $Q ( \Psi = 0 ) \leq \epsilon ^ { \prime }$ . In both cases the rejection probability is at most $\epsilon ^ { \prime } \leq \delta$ . No assumption beyond i.i.d. sampling and frozen cells is used.

Power. Let $P \in \mathcal { P } _ { \mathrm { c } } ( \alpha , \gamma )$ . Its pushforward is a product law, so $P ( \Psi = 1 ) \leq \epsilon ^ { \prime } .$ , and $P ( \hat { e } \ <$ $\alpha + \gamma / 2 ) \le \exp ( - n \gamma ^ { 2 } / 2 ) \le \epsilon ^ { \prime }$ . The power is at least $1 - 2 \epsilon ^ { \prime } \ge \beta$ . Collecting sample sizes, and using $M \geq 2$ and $\kappa \leq 1$ to absorb the $\bar { \log ( 1 / \epsilon ^ { \prime } ) } / \gamma ^ { 2 }$ term, gives the displayed bound with a constant depending on $( \delta , \beta )$

Rates on the sign family. With $M = 2 k$ and $\kappa = 1 / 2 , M ^ { 2 / 3 } \gamma ^ { - 4 / 3 } \leq M ^ { 1 / 2 } \gamma ^ { - 2 }$ if and only if $\gamma \le M ^ { - 1 / 4 }$ , so $N ( \mathcal { P } _ { \mathrm { c } } , \mathcal { H } ) = O ( \sqrt { k } / \gamma ^ { 2 } )$ in that regime. The uniform reference $P _ { 0 }$ with $p = \alpha + \gamma$ lies in $\mathcal { P } _ { \mathrm { c } }$ , so $N ( \mathcal { P } _ { \mathrm { c } } , \mathcal { H } ) \geq N ( P _ { 0 } , \mathcal { H } ) = \Omega ( \sqrt { k } / \gamma ^ { 2 } )$ by Proposition 3 with uniform weights, and Theorem 3(i) gives the EFC lower bound at the same law. Conversely, the IUT upper bound of Appendix C.2 extends to every $P \in \mathcal { P } _ { \mathrm { c } }$ on the sign family: all policies have risk $P ( L = 1 ) \geq \alpha + \gamma$ and on the uniform-concentration event over the $\mathbf { \bar { 2 } } ^ { k }$ policies every policy passing the coverage screen has $T _ { A } \ge \kappa n / 2$ and $E _ { A } - \alpha T _ { A } \geq \gamma T _ { A } - C \sqrt { n k }$ , so $N _ { \mathrm { I U T } } ( \mathcal { P } _ { \mathrm { c } } ) = O ( k / ( \kappa \gamma ) ^ { 2 } )$ ; this justifies the entry in the third row of Table 1. Whether the $M ^ { 2 / 3 } \gamma ^ { - 4 / 3 }$ term is necessary for $\gamma > M ^ { - 1 / 4 }$ is left open.

## B.3 PROOF OF PROPOSITION 2

Two inversions. For $X \sim \operatorname { B i n } ( t , q )$ observed at $x ,$ Hoeffding’s inequality inverted at error $\zeta$ gives that the one-sided Clopper–Pearson upper bound is at most $x / t + \sqrt { \log ( 1 / \zeta ) / ( 2 t ) }$ and the lower bound is at least $x / t - \sqrt { \log ( 1 / \zeta ) / ( 2 t ) }$

Uniform deviations. The classes ${ \mathcal { F } } _ { 1 } = \{ ( s , l ) \mapsto \mathbf { 1 } _ { A } ( s ) : A \in { \mathcal { A } } \}$ and $\mathcal { F } _ { 2 } = \{ ( s , l ) \mapsto l \mathbf { 1 } _ { A } ( s )$ $A \in { \mathcal { A } } \}$ have VC dimension at most d; a set shattered by $\mathcal { F } _ { 2 }$ consists of points with $l = 1$ whose scores are shattered by A. The VC bound for Boolean classes (Vershynin, 2018, Chapter 8) gives E sup $_ { f \in \mathcal { F } _ { i } } \left| P _ { n } f - P f \right| \leq C _ { 0 } \sqrt { d / n }$ , and the bounded-differences inequality gives, with probability at least $1 - ( 1 - \beta ) / 2$ for both classes simultaneously,

$$
\operatorname* { s u p } _ { A \in { \cal A } } | c _ { n } ( A ) - c ( A ) | \leq \epsilon _ { n } , \qquad \operatorname* { s u p } _ { A \in { \cal A } } | m _ { n } ( A ) - m ( A ) | \leq \epsilon _ { n } ,
$$

$$
\epsilon _ { n } = C _ { 0 } \sqrt { d / n } + \sqrt { \log ( 8 / ( 1 - \beta ) ) / ( 2 n ) } ,
$$

where $c _ { n } ( A ) = T _ { A } / n , m ( A ) = P ( S \in A , L = 1 ) { \mathrm { ~ a n d ~ } } m _ { n } ( A ) = E _ { A } / n .$

Part (i). Let $P \in { \mathcal { P } } _ { \gamma } ( A )$ and work on the event above. Suppose A passes the coverage screen, $U _ { A } \geq$ κ. By the first inversion $c _ { n } ( A ) \geq \kappa - \sqrt { \log ( 2 / \delta ) / ( 2 n ) } , \operatorname { s o } c ( A ) \geq \kappa - \sqrt { \log ( 2 / \delta ) / ( 2 n ) } - \epsilon _ { n } \geq \kappa - \gamma$ provided $\sqrt { \log ( 2 / \delta ) / ( 2 n ) } + \epsilon _ { n } \leq \gamma$ . The definition of $\mathcal { P } _ { \gamma }$ then gives $m ( A ) \geq ( \alpha + \gamma ) c ( A )$ , and since $c ( A ) \stackrel { \cdot } { \geq } \kappa - \gamma \geq \kappa / 2 .$

$$
\frac { E _ { A } } { T _ { A } } \geq \frac { m ( A ) - \epsilon _ { n } } { c ( A ) + \epsilon _ { n } } \geq \alpha + \gamma - \frac { 2 \epsilon _ { n } } { c ( A ) } \geq \alpha + \gamma - \frac { 4 \epsilon _ { n } } { \kappa } .
$$

If also $\epsilon _ { n } \leq \kappa / 4$ , then $T _ { A } ~ \ge ~ n \kappa / 4$ , and the second inversion gives $L _ { A } \ge \alpha + \gamma - 4 \epsilon _ { n } / \kappa -$ $\sqrt { 2 \log ( 2 / \delta ) / ( n \kappa ) }$ . All three conditions, and $L _ { A } > \alpha$ , hold once $n \geq C \{ d + \log ( 2 / \delta ) + \log ( 4 / ( 1 -$ $\dot { \beta } ) ) \} / ( \kappa \gamma ) ^ { 2 }$ for an absolute C. Then every screened policy has $L _ { A } > \alpha$ , and if no policy is screened $L _ { \mathrm { I U T } } = + \infty ;$ ; either way the IUT rejects. Its power is at least $1 - ( 1 - \beta ) / 2 \geq \beta$

Part (ii). Choose a score law under which $c ( A _ { 0 } ) \geq \prime$ κ and let P have constant risk $\alpha + \gamma$ . Every policy has risk $\alpha + \gamma$ , so $P \in \mathcal { P } _ { \gamma } ( A )$ . Let Q equal $P$ except that the risk inside $A _ { 0 }$ is α; then $A _ { 0 }$ witnesses feasibility of Q. Since $\mathrm { K L } ( P \| Q ) = c ( A _ { 0 } ) \mathrm { k l } ( \alpha + \gamma \| \alpha ) \leq \gamma ^ { 2 } / \{ \alpha ( 1 - \alpha ) \}$ , Pinsker’s inequality gives $\mathrm { T V } ( P ^ { n } , Q ^ { n } ) \leq \{ n \gamma ^ { 2 } / ( 2 \alpha ( 1 - \alpha ) ) \} ^ { 1 / 2 } .$ A level-δ audit with power $\beta$ at P needs $\mathrm { T V } \dot { ( } P ^ { n } , \mathbf { \dot { Q } } ^ { \overline { { n } } } ) \geq \beta - \dot { \delta }$ , hence $n \ge 2 ( \beta - \delta ) ^ { 2 } \alpha ( 1 - \overset { \cdot } { \alpha } ) / \gamma ^ { 2 }$

Part (iii): localization to the smaller side of the floor. We use Vapnik’s relative-deviation inequalities: for a class ${ \mathcal { F } } \operatorname { o f } \left\{ 0 , 1 \right\}$ -valued functions with growth function $S _ { \mathcal { F } } ,$ , with probability at least $1 \mathrm { ~ - ~ } \eta .$ sup $_ { f } ( P f \bar { \mathrm { ~ - ~ } } P _ { n } f ) / \sqrt { P f } \le \epsilon$ and sup $\bar { \bf \Phi } _ { f } ( P _ { n } f - P f ) / \sqrt { P _ { n } f } \ \leq \ \bar { \epsilon }$ , where $\epsilon = 2 \{ ( \log S _ { \mathcal { F } } ( 2 n ) + \log ( 8 / \eta ) ) / n \} ^ { 1 / 2 }$ (Boucheron et al., 2005, Theorem 5.1). Solving the second inequality for $\dot { P } _ { n } f$ gives, for every ${ \bar { f } } \in { \mathcal { F } }$

$$
| P _ { n } f - P f | \leq \epsilon \sqrt { P f } + \epsilon ^ { 2 } .\tag{6}
$$

Apply this to ${ \mathcal { F } } = \left\{ \mathbf { 1 } _ { A } \right\} \cup \left\{ \mathbf { 1 } _ { A ^ { c } } \right\} \cup \left\{ l \mathbf { 1 } _ { A } \right\} \cup \left\{ l \mathbf { 1 } _ { A ^ { c } } \right\} ( A \in { \mathcal { A } } )$ . Each of the four classes has VC dimension at most d, so lo $\begin{array} { r } { \~ \ ; S _ { \mathcal { F } } ( 2 n ) \le \log 4 + d \log ( 2 e n / d ) } \end{array}$ for $n \geq d$ . Take $\eta = ( 1 - \beta ) / 4$ , and let Hoeffding’s inequality give $| \hat { e } - e | \le \{ \log ( 8 / ( 1 - \beta ) ) / ( 2 n ) \} ^ { 1 / 2 }$ for the overall loss rate with probability at least $\bar { 1 } - ( \bar { 1 } - \beta ) \dot { / } 4$ . On the intersection Ω of these events, which has probability at least $1 - ( 1 - \beta ) / 2$ , every $A \in { \mathcal { A } }$ satisfies, with $c = c ( A ) , m = m ( A )$ and $c _ { n } , m _ { n }$ their empirical versions,

$$
| c _ { n } - c | \leq \epsilon \sqrt { \operatorname* { m i n } \{ c , 1 - c \} } + \epsilon ^ { 2 } ,\tag{D1}
$$

$$
| m _ { n } - m | \leq \operatorname* { m i n } { \big \{ } \epsilon { \sqrt { c } } + \epsilon ^ { 2 } , | { \hat { e } } - e | + \epsilon { \sqrt { 1 - c } } + \epsilon ^ { 2 } { \big \} } ,\tag{D2}
$$

using $c _ { n } - c = - ( c _ { n } ( A ^ { c } ) - c ( A ^ { c } ) ) , m _ { n } - m = ( { \hat { e } } - e ) - ( m _ { n } ( A ^ { c } ) - m ( A ^ { c } ) )$ and $m ( A ^ { c } ) \leq 1 - c$

Clopper–Pearson widths. Let $X \sim \operatorname { B i n } ( t , q )$ be observed at $x , \hat { q } = x / t$ . Since $\partial _ { q } \mathrm { k l } ( \hat { q } , q ) =$ $( q - \hat { q } ) / \{ q ( 1 - q ) \}$ , we have $\begin{array} { r } { \mathrm { k l } ( \hat { q } , q ) \geq ( q - \hat { q } ) ^ { 2 } / \{ 2 \operatorname* { m a x } _ { s \in [ \hat { q } , q ] } s ( 1 - s ) \} \geq ( q - \hat { q } ) ^ { 2 } / ( 2 \operatorname* { m i n } \{ q , 1 - \hat { q } \} ) } \end{array}$ for $q \geq { \hat { q } }$ . The Chernoff bound $P _ { q } ( X \leq x ) \leq \exp \{ - t \mathrm { k l } ( \hat { q } , q ) \}$ therefore gives, for the upper bound U at error ζ,

$$
U - \hat { q } \le \{ 2 \operatorname* { m i n } \{ U , 1 - \hat { q } \} \log ( 1 / \zeta ) / t \} ^ { 1 / 2 } ,\tag{7}
$$

and the lower bound satisfies $\hat { q } - L \le \{ \log ( 1 / \zeta ) / ( 2 t ) \} ^ { 1 / 2 }$ as before.

Write ℓ = log(2/δ), σ = min{κ, 1 − κ + γ} and $u = \epsilon ^ { 2 } + \ell / n$ . Suppose

$$
u \leq c _ { 0 } \gamma ^ { 2 } \kappa / ( 1 - \kappa + \gamma ) , \qquad | \hat { e } - e | \leq \gamma / 4 8 , \qquad n \geq 8 \ell / ( \kappa \gamma ^ { 2 } ) , \qquad c _ { 0 } = 9 6 ^ { - 2 } .\tag{8}
$$

Because $\kappa ^ { 2 } \leq \kappa \leq 1 - \kappa + \gamma$ when $\sigma = \kappa ,$ the first condition implies $u \leq c _ { 0 } \gamma ^ { 2 } / \sigma$ , and since $\sigma \geq \gamma$ $( \mathrm { a s } \gamma \leq \kappa / 2 )$ also $u \leq c _ { 0 } \gamma$

Step 1: the screen admits only policies of coverage at least $\kappa - \gamma$ . Suppose $U _ { A } \geq \kappa$ but $t : = \kappa -$ $c ( A ) > \gamma$ . By (7) applied to the n coverage trials, $\kappa - c _ { n } \leq \{ 2 v \ell / n \} ^ { 1 / 2 }$ with $v = \operatorname* { m i n } \{ U _ { A } , 1 - c _ { n } \}$ and by $( \mathrm { D } 1 ) , c _ { n } - c \le \epsilon \sqrt { \operatorname* { m i n } \{ c , 1 - c \} } + \epsilon ^ { 2 }$ . Now min $\{ c , 1 - c \} \leq$ min $\{ \kappa , 1 - \kappa + t \} \leq ( t / \gamma ) \sigma$ because $t \geq \gamma$ . Using $\epsilon \sqrt { x } \leq ( x + \epsilon ^ { 2 } ) / 2$ in (D1) and $\sqrt { 2 U \ell / n } ~ \le ~ U / 2 + \ell / n$ in (7) gives $1 - c _ { n } \leq 2 ( 1 - \kappa + t ) +$ 2u and $U _ { A } \leq 4 \kappa + 4 u$ , so $v \leq 4 ( t / \gamma ) \sigma \cdot$ + 4u. Adding the two displays,

$$
t \leq { \sqrt { 8 ( t \sigma / \gamma ) u } } + { \sqrt { 8 } } u + { \sqrt { ( t \sigma / \gamma ) u } } + u \leq ( { \sqrt { 8 } } + 1 ) \{ { \sqrt { c _ { 0 } } } + c _ { 0 } \} t < t ,
$$

using $( t \sigma / \gamma ) u \leq c _ { 0 } t \gamma \leq c _ { 0 } t ^ { 2 }$ and $u \leq c _ { 0 } \gamma \leq c _ { 0 } t .$ . This contradiction proves the claim.

Step 2: every screened policy is certified infeasible. Let $U _ { A } \geq \kappa ,$ so $c = c ( A ) \geq \kappa - \gamma \geq \kappa / 2$ , and $P \in \mathcal { P } _ { \gamma } ( \mathcal { A } )$ gives $m \geq ( \alpha + \gamma ) c$ . With $\varrho = \alpha + \gamma / 2$ we have $m - \varrho c \ge \gamma c / 2 .$ , and by $( \mathrm { D } 1 ) { - } ( \mathrm { D } 2 )$

$$
\begin{array} { r } { | ( m _ { n } - \varrho c _ { n } ) - ( m - \varrho c ) | \overset { } { \leq } \Delta : = \operatorname* { m i n } \big \{ 2 \epsilon \sqrt { c } + 2 \epsilon ^ { 2 } , | \hat { e } - e | + 2 \epsilon \sqrt { 1 - c } + 2 \epsilon ^ { 2 } \big \} . } \end{array}
$$

If $\kappa \leq 1 / 2$ , then $\kappa / ( 1 - \kappa + \gamma ) \leq 2 \kappa$ , so $\epsilon ^ { 2 } \le 2 c _ { 0 } \gamma ^ { 2 } \kappa \le \gamma ^ { 2 } c / 2 5 6$ and the first term gives $\Delta \le \gamma c / 8 + \gamma c / 8$ . If $\dot { \kappa } > 1 / 2$ , then $c \geq 1 / 4 , 1 - c \leq 1 - \kappa + \gamma$ and $\epsilon ^ { 2 } \leq c _ { 0 } \gamma ^ { 2 } / ( 1 - \kappa + \gamma )$ , so the second term gives $\Delta \leq \gamma / 4 8 + \gamma / 4 8 + \gamma / 4 8 \leq \gamma c / 4$ . In both cases $m _ { n } - \varrho c _ { n } \ge \gamma c / 4 > 0 ;$ also $c _ { n } \geq c / 2 \geq \kappa / 4$ by (D1). Hence $E _ { A } / T _ { A } \stackrel { \cdot } { \geq } \alpha + \gamma / 2$ with $T _ { A } \geq n \kappa / 4$ , and the Clopper–Pearson lower bound satisfies $L _ { A } \ge \alpha + \gamma / 2 - \{ 2 \ell / ( n \kappa ) \} ^ { 1 / 2 } > \alpha$ by the last condition in (8).

Sample size. On Ω, every screened policy has $L _ { A } > \alpha ,$ , so the IUT rejects; its power is at least $1 - ( 1 - \beta ) / 2 \geq \beta$ . The conditions (8) hold whenever n $\ge C \{ ( 1 - \kappa + \gamma ) \bar  [ d \log ( 2 e n \bar { / } d ) + \log ( 8 / ( 1 -$ $\beta ) ) ] + \log ( 2 / \delta ) + \log ( 8 / ( 1 - \beta ) ) \} / ( \kappa \gamma ^ { 2 } )$ for an absolute constant $C ,$ using $( 1 - \kappa + \gamma ) \leq 1$ and $\kappa \leq 1$ for the terms without d. The left side minus the right side is increasing in n, which proves part (iii).

Remark on the logarithm. The factor $\log ( 2 e n / d )$ comes from the growth function. Localized VC bounds for sets of small probability (Giné and Koltchinskii, 2006) suggest that it can be replaced by $\log ( e / \operatorname* { m i n } \{ \kappa , 1 - \kappa \} )$ ; the experiments of Section 6.1 are consistent with this form, but we do not use it.

## B.4 PROOF OF THEOREM 1

Upper bound. Unions of at most K intervals have VC dimension $d = 2 K$ . Part (iii) of Proposition 2 with $\begin{array} { r } { 1 - \kappa + \gamma \le 2 ( 1 - \kappa ) \left( \mathrm { a s } \gamma \le 1 - \kappa \right) } \end{array}$ ) gives power at least $\beta$ once n $: \geq C \{ K ( 1 - \kappa ) \log ( \bar { 2 } e n / K ) +$ $1 \} / ( \kappa \gamma ^ { 2 } )$ , with C depending on $( \delta , \beta )$ . This condition holds for $n \geq C ^ { \prime } \{ K ( 1 - \kappa ) \log ( e / ( \kappa \gamma ) ) +$ $1 \dot  \} / ( \kappa \gamma ^ { 2 } )$ because the logarithm of such an $n / K$ is at most a constant times $\log ( e / ( \kappa \gamma ) )$

Lower bound: a padded construction. Fix $B = K - 1$ and let $\gamma _ { 4 } , D , m , L , \nu _ { 0 } , \nu _ { 1 }$ be as in Steps 1–2 of the proof of Theorem 5 with this $B ;$ the conditions imposed below imply those of that theorem. Scores are uniform on [0, 1]. A construction region R of mass π is split into B consecutive groups of mass $\pi / B$ , each split into a left and a right half, with loss probability $\eta _ { b , \pm } = m \pm s _ { b } d _ { b }$ on the two halves as in Step 2. Its complement, the padding, has a fixed conditional loss probability $\eta _ { \mathrm { p a d } }$ that is the same under both priors. The inner best risk is $\begin{array} { r } { r _ { \mathrm { i n } } ^ { * } ( \eta ) = m - B ^ { - 1 } \sum _ { b } d _ { b } } \end{array}$ , and the lower halves (those with risk $m - d _ { b } )$ have total mass $\pi / 2$ . We distinguish two cases.

Case $\kappa \geq 1 / 2$ . Take $\pi = 2 ( 1 - \kappa )$ , padding $F = [ 0 , 1 - \pi )$ with $\eta _ { \mathrm { p a d } } = \alpha .$ , and $R = [ 1 - \pi , 1 ]$ The set W formed by F and the lower halves has coverage $1 - \pi + \pi / 2 = \kappa .$ , is a union of at most $B + 1 = K$ intervals, and has risk

$$
\begin{array} { r } { r ( W ) = \alpha + \frac { \pi } { 2 \kappa } \lbrace r _ { \mathrm { i n } } ^ { * } ( \eta ) - \alpha \rbrace = \alpha + \frac { 1 - \kappa } { \kappa } \lbrace r _ { \mathrm { i n } } ^ { * } ( \eta ) - \alpha \rbrace . } \end{array}
$$

Under the null prior, on the event $r _ { \mathrm { i n } } ^ { * } \leq \alpha$ of Step 2, W witnesses feasibility for $\boldsymbol { \mathcal { A } } _ { K }$ . Under the alternative prior, on the event $r _ { \mathrm { i n } } ^ { * } \ge \alpha + \gamma _ { 4 }$ , consider any measurable set. The conditional risk is constant on each half and on F. Every upper half has risk $m + d _ { b } \ge m$ , and every point of W has risk at most $m ,$ because $m - \alpha = \mathbb { E } _ { \nu _ { 0 } } d - g / 4 \geq 3 g / 4 > 0$ . Hence, as in the proof for $\kappa = 1 / 2$ , the least risk at coverage exactly κ is $r ^ { * } : = r ( W ) \geq \dot { \alpha } + ( 1 - \kappa ) \gamma _ { 4 } / \kappa$ , larger coverage cannot lower it, and at coverage $\kappa - s$ with $s \leq \kappa / 2$ the optimal set removes mass s of risk at most m from $W$ so its risk is at least $r ^ { * } - s ( m - r ^ { * } ) / ( \kappa - s ) \ge r ^ { * } - 2 s D / \kappa .$ because $m - r ^ { * } \leq m - \alpha \leq D$ . Put $\gamma = ( 1 - \kappa ) \gamma _ { 4 } / ( 2 \kappa )$ . Taking $s = \gamma$ and using $D \le 1 / 1 6 \le \kappa / 2$ , every measurable set of coverage at least $\kappa - \gamma$ has risk at least $\alpha + 2 \gamma - \gamma = \alpha + \gamma \colon$ : the alternative draws lie in $\mathcal { P } _ { \gamma } ( A _ { K } )$

Case $\kappa < 1 / 2 .$ . Take $\pi = 2 \kappa , R = [ 0 , \pi )$ and padding $H = [ \pi , 1 ]$ with $\eta _ { \mathrm { p a d } } = 1$ . The lower halves form a set $\dot { W }$ of coverage κ that is a union of at most B intervals, with risk $r _ { \mathrm { i n } } ^ { \ast } ;$ under the null prior it witnesses feasibility. Under the alternative, the points of least risk are the lower halves, then the upper halves (risk at least $m )$ , then H (risk $1 > m + D )$ . As before, larger coverage cannot lower the risk and coverage $\kappa - s$ lowers it by at most $2 s D / \kappa$ . Put $\gamma = \gamma _ { 4 } / 2$ and require $D \leq \kappa / 2 ;$ then every set of coverage at least $\kappa - \gamma$ has risk at least $\alpha + \gamma _ { 4 } - \gamma = \alpha + \gamma$

Indistinguishability. Poissonize as in Step 3 of the proof of Theorem 5, with mean $n ^ { \prime }$ . Counts in the padding are independent of counts in R and have the same law under both priors, so the total variation between the two mixtures equals that between the mixtures of the R-counts. These are independent Poisson counts with means $\lambda \eta _ { b , h }$ and $\lambda ( 1 - \eta _ { b , h } )$ ) in each half, now with $\lambda = \pi n ^ { \prime } / ( 2 B )$ , so Steps 4–6 apply verbatim with $\rho = \pi n ^ { \prime } / \{ B m ( 1 - m ) \}$ . The condition $\tau = \rho D ^ { 2 } \leq ( 2 L + 2 ) / e ^ { 3 }$ holds whenever $\pi n ^ { \prime } \leq c B \dot { / } ( \gamma _ { 4 } ^ { 2 } \log \dot { B } )$ . In the case $\dot { \kappa } \geq 1 / 2$ this reads $n ^ { \prime } \le c B ( 1 - \kappa ) / ( 8 \kappa ^ { 2 } \gamma ^ { 2 } \log B )$ and in the case $\kappa < 1 / 2$ it reads $n ^ { \prime } \le c B / ( 8 \kappa \gamma ^ { 2 } \log \dot { B } )$ . Both are at least $c ^ { \prime \prime } K ( 1 - \kappa ) / ( \kappa \gamma ^ { 2 } \log K )$ for $K \geq 3$ . Hence no valid audit with fewer samples has power $\beta$ over $\mathcal { P } _ { \gamma } ( A _ { K } )$

Conditions. Theorem 5 needs $\gamma _ { 4 } \leq c _ { 1 } / \log B$ , the case $\kappa < 1 / 2$ needs $D \leq \kappa / 2 ,$ , and $D =$ $\gamma _ { 4 } / E _ { L } \leq \gamma _ { 4 } L / c _ { * }$ with $L = \lceil \log B \rceil$ . In the case $\kappa \geq 1 / 2 , \gamma _ { 4 } = 2 \kappa \gamma / ( 1 - \kappa ) \leq 2 \gamma / ( 1 - \kappa ) ;$ ; in the case $\kappa < 1 / 2 , \gamma _ { 4 } = 2 \gamma$ . All conditions therefore hold when $\gamma \le c ^ { \prime } \operatorname* { m i n } \{ \kappa , 1 - \kappa \} / \log K$ for a suitable $c ^ { \prime }$ depending only on a, and $K \geq K _ { 0 }$ . The auditor may know the uniform score marginal, the cells and the padding: only the location and size of the risk heterogeneity inside R are random. $\mathbf { A t } \kappa = 1 / 2$ the padding is empty and the construction is the one of Theorem 5 with $B = K - 1$ consecutive groups. □

Numerical checks. The artifact test $\mathsf { t e s t \_ v 6 1 . p y }$ verifies the trimming inequality of both cases on 20,000 random padded instances with $\kappa \in ( . 0 5 , . 9 5 )$ , the width bound (7) for every outcome with $t \leq 1 0 ^ { 4 }$ trials and $\bar { \zeta } \in \{ . 0 2 5 , . 0 5 , . 2 5 \}$ , and Lemma 4 against constrained numerical optimization.

Consequences. The ratio of the bounds in parts (i) and (ii) of Proposition 2 is ${ \cal O } ( d / ( \kappa ^ { 2 } \alpha ( 1 - \alpha ) ) )$ up to $( \delta , \beta )$ terms. On the uniform sign family, the alternative prior draws of Theorem 5 with $r ^ { * } ( \eta ) \stackrel {  } { \stackrel { } { \textrm { = } } \alpha + \gamma }$ lie in $\mathcal { P } _ { \gamma / 2 }$ by the trimming argument of Theorem 1 (case $\kappa = 1 / 2 )$ , which uses $d _ { b } \leq D \leq 1 / 1 6$ ; the whole class $\mathcal { P } _ { \mathrm { h e t } }$ need not, since trimming a high-risk lower half can lower the risk by more. Hence Theorem $5 \mathrm { { s } }$ construction also shows that the upper end is attained up to log d there.

## B.5 PROOF OF THEOREM 5

Constants $c , C$ below depend only on $( a , \delta , \beta )$ unless stated otherwise.

Upper bounds. Let $P _ { \eta } \in \mathcal { P } _ { \mathrm { h e t } }$ . Every policy has coverage exactly $1 / 2$ and risk $r _ { \eta } ( A _ { \theta } ) \geq \alpha + \gamma$ Hoeffding’s inequality and a union bound over the $2 ^ { B }$ policies give, with probability at least $1 - \eta _ { 0 }$ $T _ { A } \geq n / 2 - C \sqrt { n ( B + \log ( 1 / \eta _ { 0 } ) ) }$ and $E _ { A } - \alpha T _ { A } \ge n \gamma / 2 - C \sqrt { n ( B + \log ( 1 / \eta _ { 0 } ) ) }$ for every A. As in the upper-bound part of Appendix C.2, for $n \geq C B / \gamma ^ { 2 }$ every Clopper–Pearson risk lower bound exceeds $\alpha ,$ , so the IUT rejects with probability at least $\beta$ uniformly over the class. For a single known $P \in \mathcal { P } _ { \mathrm { h e t } }$ and a feasible Q with witness $A ,$ , the function $f = \mathop { \mathbf { 1 } _ { A } \cdot \cdot } _ { \mathbf { \delta } } ( L - \alpha )$ has oscillation one, $\mathbb { E } _ { Q } f \le 0$ and $\mathbb { E } _ { P } f = \{ r _ { \eta } ( A ) - \alpha \} / 2 \ge \gamma / 2$ . So every feasible law is $\gamma / 2$ -far after the 4B-cell pushforward, and identity testing gives $N ( P , \mathcal { H } ) = O ( \sqrt { B } / \gamma ^ { 2 } )$

Step 1: moment-matched magnitudes. For $L \geq 1$ let $\begin{array} { r } { E _ { L } = \operatorname* { i n f } _ { g } \operatorname* { s u p } _ { u \in [ 0 , 1 ] } | \sqrt { u } - g ( u ) | } \end{array}$ | over polynomials g of degree at most L. Substituting $u = x ^ { 2 } , E _ { L }$ equals the error of best approximation of |x| on $[ - 1 , 1 ]$ by polynomials of degree $2 L ,$ , and Bernstein’s theorem gives $2 L E _ { L } \to \beta _ { * } \approx 0$ .2802 (Bernstein, 1914). Since each $\begin{array} { r } { E _ { L } > \bar { 0 _ { , } } c _ { * } = \operatorname* { i n f } _ { L > 1 } L E _ { L } > 0 ; } \end{array}$ numerically $L E _ { L } \ge 1 / 8$ for $L  \leq 1 2$ (Appendix B.6). By duality (Cai and Low, 2011, Lemma 1) there are probability measures $\pi _ { 0 } , \pi _ { 1 }$ on [0, 1] with $\textstyle \int u ^ { j } d \pi _ { 0 } = \int u ^ { j } d \pi _ { 1 }$ for $j = 1 , \dots , L$ and $\begin{array} { r } { \int { \sqrt { u } } d \pi _ { 0 } - \int { \sqrt { u } } d \pi _ { 1 } = 2 E _ { L } } \end{array}$ . For $D > 0$ let $\nu _ { i }$ be the law of $D \sqrt { U }$ with $U \sim \pi _ { i }$ . Then $\nu _ { 0 } , \nu _ { 1 }$ are supported on $[ 0 , D ]$ , match $\mathbb { E } d ^ { 2 j }$ for $j \leq L$ , and $g : = \mathbb { E } _ { \nu _ { 0 } } d - \mathbb { E } _ { \nu _ { 1 } } d = 2 D E _ { L }$

Step 2: priors and their contracts. Set $L = \lceil \log B \rceil , D = \gamma / E _ { L }$ and $m = \alpha + \mathbb { E } _ { \nu _ { 0 } } d - g / 4$ Under prior $i ,$ draw independently for each group a sign s uniform on $\{ \pm 1 \}$ and $d _ { b } \sim \nu _ { i } .$ , and set $\eta _ { b , \pm } = m \pm s _ { b } d _ { b }$ . Since $D \leq \gamma L / c _ { * }$ , the condition $\gamma \leq c _ { 1 } / \log B$ with $c _ { 1 }$ small gives $D \leq a / 2 ;$ together with m $\in [ \alpha - D / 4 , \alpha + \dot { D } ]$ and $\alpha \in [ 2 a , 1 - 2 a ]$ this keeps every $\eta _ { b , h }$ in $[ a , 1 - a ]$ . Because $\begin{array} { r } { r ^ { * } ( \eta ) = m - B ^ { - 1 } \sum _ { b } d _ { b } } \end{array}$ , Hoeffding’s inequality gives

$$
\begin{array} { r l } & { \operatorname* { P r } \{ r ^ { * } ( \eta ) > \alpha \} + \operatorname* { P r } \{ r ^ { * } ( \eta ) < \alpha + \gamma \} \leq 2 \exp \{ - B g ^ { 2 } / ( 8 D ^ { 2 } ) \} } \\ & { \phantom { \operatorname* { P r } \{ r ^ { * } ( \eta ) > \alpha \} } = 2 \exp ( - B E _ { L } ^ { 2 } / 2 ) \leq 2 \exp \{ - B c _ { * } ^ { 2 } / ( 2 L ^ { 2 } ) \} , } \end{array}
$$

using $\mathbb { E } _ { 0 } r ^ { * } = \alpha - g / 4 , \mathbb { E } _ { 1 } r ^ { * } = \alpha + 3 g / 4 = \alpha + \gamma + g / 4$ and $\gamma = g / 2$ . Under prior 0 the policy choosing the lower half in each group has coverage $1 / 2 ,$ so on the event $r ^ { * } \leq { \mathfrak { a } }$ the law $P _ { \eta }$ is feasible; under prior 1 and $r ^ { * } \geq \alpha + \gamma$ it lies in $\mathcal { P } _ { \mathrm { h e t } }$

Step 3: Poissonization. If a test with n samples has level δ and power $\beta ,$ , so does one with any $n _ { \mathrm { ~ } } ^ { \prime \prime } \geq n$ samples; take $n ^ { \prime \prime } = \operatorname* { m a x } \{ n , n _ { 1 } \}$ with $n _ { 1 } = \lceil 4 \log ( 4 / ( \beta ^ { ' } - \delta ) ) \rceil$ . Draw $N \sim \mathrm { P o i } ( 2 n ^ { \prime \prime } )$ observations and apply the test to the first $\mathit { \Pi } _ { n ^ { \prime \prime } }$ if $N \geq n ^ { \prime \prime }$ , otherwise accept. The level is unchanged and the power drops by at most $\operatorname* { P r } ( N < n ^ { \prime \prime } ) \leq e ^ { - n ^ { \prime \prime } / 4 } \leq ( \beta - \delta ) / 4$ . Under Poissonization with mean $n ^ { \prime } = 2 n ^ { \prime \prime }$ , the numbers of errors and correct observations in the 2B halves are independent Poisson variables with means $\lambda \eta _ { b , h }$ and $\lambda ( 1 - \eta _ { b , h } ) , \lambda = n ^ { \prime } / ( 2 B )$ ; groups are independent under both priors.

Step 4: one group. Let $F _ { 0 }$ be the law of one group’s four counts at $\eta \equiv m$ , and let $\mu _ { i }$ be its law under prior i. For Poisson likelihood ratios, $\mathbb { E } _ { F _ { 0 } } [ ( d { \bar { P _ { x } } } / d { \dot { F } } _ { 0 } ) ( d P _ { x ^ { \prime } } / d F _ { 0 } ) ] \stackrel { \cdot } { = } \exp \{ \lambda x x ^ { \prime } / \dot { m } + \lambda x x ^ { \prime } / ( 1 - m ) \}$ for error-rate shifts $x , x ^ { \prime }$ of one half. The two halves have shifts $\pm s d .$ , so the group-level kernel is $\exp ( \rho s s ^ { \prime } d d ^ { \prime } )$ with $\rho = 2 \lambda / \{ m ( 1 - m ) \} = n ^ { \prime } / \{ B m ( 1 - m ) \}$ . Averaging over independent signs gives cosh(ρdd<sup>′</sup>) and

$$
\int \frac { ( d \mu _ { 1 } - d \mu _ { 0 } ) ^ { 2 } } { d F _ { 0 } } = \sum _ { j \ge 0 } \frac { \rho ^ { 2 j } } { ( 2 j ) ! } \Big ( \mathbb { E } _ { \nu _ { 1 } } d ^ { 2 j } - \mathbb { E } _ { \nu _ { 0 } } d ^ { 2 j } \Big ) ^ { 2 } \le \sum _ { j > L } \frac { ( \rho D ^ { 2 } ) ^ { 2 j } } { ( 2 j ) ! } .
$$

With $\begin{array} { r } { \tau = \rho D ^ { 2 } , \sum _ { i > L } \tau ^ { 2 j } / ( 2 j ) ! \le e ^ { \tau } \{ e \tau / ( 2 L + 2 ) \} ^ { 2 L + 2 } . \mathrm { ~ I f ~ } \tau \le ( 2 L + 2 ) / e ^ { 3 } } \end{array}$ this is at most $\exp \{ - ( 2 - e ^ { - 3 } ) ( 2 \bar { L } + 2 ) \} \le e ^ { - 3 . 9 ( L + 1 ) }$ . By Cauchy–Schwarz, $\mathrm { T V } ( \mu _ { 0 } , \mu _ { 1 } ) \le { \textstyle \frac { 1 } { 2 } } e ^ { - 1 . 9 5 ( L + 1 ) }$

Step 5: all groups. With $\begin{array} { r } { H ^ { 2 } ( P , Q ) = 1 - \int { \sqrt { d P d Q } } } \end{array}$ we have $H ^ { 2 } \leq \mathrm { T V } , H ^ { 2 } ( \mu _ { 0 } ^ { \otimes B } , \mu _ { 1 } ^ { \otimes B } ) \leq$ $B H ^ { 2 } ( \mu _ { 0 } , \mu _ { 1 } )$ and $\mathrm { T V } \le \sqrt { 2 H ^ { 2 } }$ . Hence $\mathrm { T V } ( \mu _ { 0 } ^ { \otimes B } , \mu _ { 1 } ^ { \otimes B } ) \leq \sqrt { B } e ^ { - 0 . 9 7 5 ( L + 1 ) } \leq B ^ { - 0 . 4 7 } \mathrm { \ f o r ~ } L =$ ⌈log $B \rceil$

Step 6: conclusion. Conditioning the priors on the events of Step 2 changes the mixtures by at most their probabilities. A test of level δ over H and power $\beta$ over $\mathcal { P } _ { \mathrm { h e t } }$ therefore satisfies

$$
\begin{array} { r } { \beta - \delta - \frac { \beta - \delta } { 4 } \leq B ^ { - 0 . 4 7 } + 2 \exp \{ - B c _ { * } ^ { 2 } / ( 2 L ^ { 2 } ) \} , } \end{array}
$$

which fails for $B \geq B _ { 0 }$ whenever the condition $\tau \leq ( 2 L + 2 ) / e ^ { 3 }$ holds. That condition reads $n ^ { \prime } \leq ( 2 L + 2 ) B \bar { m ( 1 - m ) } E _ { I } ^ { 2 } / ( e ^ { 3 } \gamma ^ { 2 } )$ ; with $m ( 1 - m ) \ge a / 2 , E _ { L } \ge c _ { * } / L$ and $L  \leq 2$ log B it holds whenever $n ^ { \prime \prime } \le c _ { 2 } \ddot { B / ( \gamma ^ { 2 } \log B ) }$ ) for a suitable $c _ { 2 } .$ Since $n ^ { \prime \prime } = \operatorname* { m i a x } \{ n , n _ { 1 } \}$ and $n _ { 1 } ~ \leq$ $c _ { 2 } B / ( \gamma ^ { 2 } \log B )$ for $B \geq B _ { 0 }$ , no valid audit with $n \leq c _ { 2 } B / ( \gamma ^ { 2 } \log B )$ samples has power $\beta . \quad \sqcup$

Remarks. The auditor in this lower bound knows the family, the cells and the exact uniform score marginal; only the location and size of the risk heterogeneity are random. The unknown quantity is a sum of $\dot { B }$ nonsmooth terms $\operatorname* { m i n } \{ \eta _ { b , + } , \eta _ { b , - } \}$ , which is why the argument parallels minimax estimation of $\ell _ { 1 } { \mathsf { - t y p e } }$ functionals (Lepski et al., 1999; Cai and Low, 2011; Wu and Yang, 2016).

## B.6 NUMERICAL CHECKS FOR THEOREM 5

The script heterogeneity\_checks.py (i) computes $E _ { L }$ by a linear program on a grid; (ii) solves the dual linear program for $\pi _ { 0 } , \pi _ { 1 }$ and confirms that the optimal moment-matched gap equals $2 E _ { L } ;$ and (iii) compares the $\chi ^ { 2 }$ series of Step 4 with direct summation over Poisson counts for one group (errors only, which has the same structure with $\rho = 2 \lambda / m )$

Table 3: Numerical checks behind Theorem 5. Left: best-approximation error of $\sqrt { u }$ on $[ 0 , 1 ]$ (the monomial LP is reliable in this range; values increase towards Bernstein’s constant $\beta _ { * } \approx 0 . 2 8 0 2 )$ Middle: optimal moment-matched gap from the dual LP versus $2 E _ { L }$ . Right: largest relative error between the closed-form $\chi ^ { 2 }$ series and direct Poisson summation over two parameter settings.
<table><tr><td> $L$ </td><td> $E _ { L }$ </td><td> $2 L E _ { L }$ </td><td>L</td><td>gap</td><td> $2 E _ { L }$ </td><td> $L$ </td><td>max rel. error</td></tr><tr><td>1</td><td>0.12500</td><td>0.2500</td><td>2</td><td>0.13524</td><td>0.13524</td><td>2</td><td>1.4e-13</td></tr><tr><td>2</td><td>0.06762</td><td>0.2705</td><td>3</td><td>0.09186</td><td>0.09186</td><td>3</td><td>1.8e-11</td></tr><tr><td>3</td><td>0.04593</td><td>0.2756</td><td>4</td><td>0.06938</td><td>0.06938</td><td>4</td><td>1.6e-09</td></tr><tr><td>4</td><td>0.03469</td><td>0.2775</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>6</td><td>0.02325</td><td>0.2790</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>8</td><td>0.01747</td><td>0.2795</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>10</td><td>0.01399</td><td>0.2797</td><td></td><td></td><td></td><td></td><td></td></tr></table>

## B.7 PROTOCOL FOR THE ENVELOPE COMPARISON

Setting. Uniform sign family with $B \in \{ 6 4 , 2 5 6 , 1 0 2 4 , 4 0 9 6 , 1 6 3 8 4 , 6 5 5 3 6 , 2 6 2 1 4 4 \}$ , uniform score marginal, $\kappa = 1 / 2 , \alpha = . 2$ and Poissonized sample size n (independent Poisson counts of errors and correct observations in each half).

Heterogeneity envelope. The LP of Appendix B.6 with $L = 2$ gives $\nu _ { 0 }$ with atoms {0.0569, 0.2} and weights $\{ 0 . 9 0 4 , 0 . 0 9 6 \}$ , and $\nu _ { 1 }$ with atoms $\{ 0 , 0 . 1 5 5 4 \}$ and weights $\left\{ 0 . 7 1 9 , 0 . 2 8 1 \right\} ( D = . 2 )$ Then $g = 2 D \dot { E } _ { 2 } , \gamma = g / 2 = 0 . 0 1 3 5 2$ and $m = \stackrel { \cdot } { \alpha } + \mathbb { E } _ { \nu _ { 0 } } d - g / 4 = 0 . 2 6 3 9$ . The statistic is the exact log-likelihood ratio between the two product mixtures, obtained by summing over atoms and signs in each group. For a valid audit $\phi , \bar { \mathbb { E } _ { \mathrm { m i x } _ { 0 } } } \bar { \phi } \le \delta + \mathrm { P r } _ { 0 } ( G _ { 0 } ^ { c } ) = \delta ^ { \prime }$ and $\mathrm { \bar { E } _ { m i x _ { 1 } } } | _ { G _ { 1 } } \bar { \phi } \leq \mathrm { N P } _ { \delta ^ { \prime } } / \operatorname* { P r } _ { 1 } ( G _ { 1 } )$ . We estimate the conditioning probabilities from $2 \times 1 0 ^ { 5 }$ multinomial draws, calibrate the NP threshold at level $\delta ^ { \prime }$ on independent draws from mix , and evaluate power on independent draws from mix .

Known-reference envelope. The reference has constant risk $p = \alpha + \gamma$ . The null mixture averages the hiding components of Equation (5) with the same γ, so every component is feasible and no conditioning is needed. The NP threshold is calibrated at level $\delta = . 0 5$

Replicates and crossings. We use 2,000 replicates for calibration and 2,000 for power per grid point when $B \leq 4 0 9 6$ , and 1,000 of each when $B \geq 1 6 3 8 4 .$ , with sample-size grids around the crossing. The $B = 2 6 2$ ,144 runs used a second machine with different NumPy/SciPy versions; rerunning $B =$ 256 there changed the crossings by at most 1.2%, within Monte Carlo error. The $n _ { 8 0 }$ values are loglinear interpolations of Monte Carlo power estimates; the two envelopes use independent seeds. The script heterogeneity\_envelope.py produces all runs and merge\_het\_envelope.py merges them and draws Figure 3.

## C ESTIMATION COMPLEXITY, IUT INEFFICIENCY, AND REFERENCEADAPTATION

## C.1 EXACT DISTRIBUTION-DEPENDENT RADEMACHER COMPLEXITY

The profiles of Theorem 2 are distinguished by the population Rademacher complexity, $\Re _ { n } \ \asymp$ $\begin{array} { r } { \sum _ { b } \operatorname* { m i n } \{ w _ { b } , \sqrt { w _ { b } / n } \} } \end{array}$ , of order $\sqrt { k / n }$ on a plateau; but it does so at the estimation scale $k / \gamma ^ { 2 }$ whereas the optimal audit works at the testing scale $\sqrt { k } / \gamma ^ { 2 }$ , as in instance-optimal identity testing (Valiant and Valiant, 2017; Balakrishnan and Wasserman, 2019). For $\mathcal { F } _ { w } = \big \{ \mathbf { 1 } _ { A _ { \theta } } : \theta \in \lbrace \mathbf { \bar { - 1 } } , 1 \rbrace ^ { B } \big \}$ define $\begin{array} { r } { \mathfrak { R } _ { n } ( \mathcal { F } _ { w } ) = \mathbb { E } _ { S , \sigma } \operatorname* { s u p } _ { \theta } n ^ { - 1 } \sum _ { i = 1 } ^ { n } \sigma _ { i } \mathbf { 1 } _ { A _ { \theta } } ( S _ { i } ) } \end{array}$ , with independent Rademacher signs $\sigma _ { i } .$ . Let h(s) denote the +/− half within its group. Since $\mathbf { 1 } _ { A _ { \theta } } ( s ) = ( 1 + \theta _ { b } h ( s ) ) / 2 \operatorname { o n } G _ { b }$

![](images/5f401ffe40113a2a158e2d58c80938a5e9051fde887dba7caf91876c86cdb02c.jpg)

![](images/df2b49e0bffb1e521c256d436174be83d07348b8aa3de34a05bda1cbffb541a3.jpg)  
Figure 3: Unknown heterogeneity changes the exponent and eventually dominates. Left: 80%- power crossings of the two envelopes, scaled by $\gamma ^ { 2 }$ . The known-reference crossing grows with fitted exponent 0.48 (theory: $1 / 2 )$ ; the heterogeneity crossing grows with exponent 0.80, approaching $1 - 1 / ( 2 L + 2 ) = 0 . \dot { 8 } 3$ for fixed $L = 2$ (the theorem lets $\mathbf { \boldsymbol { \mathsf { L } } } \approx \log \mathbf { \boldsymbol { B } } )$ . Right: their ratio falls from 12.5 at $B = 6 4$ to 0.71 at $B = 2 6 2 , 1 4 4$ . Because the constant-risk reference belongs to $\mathcal { P } _ { \mathrm { h e t } }$ , the larger crossing is the operative lower bound on $N ( \mathcal { P } _ { \mathrm { h e t } } , \mathcal { H } )$ . Envelopes bound valid audits; they are not audits.

Table 4: Envelope crossings behind Figure 3. Because the constant-risk reference belongs to $\mathcal { P } _ { \mathrm { h e t } }$ the larger of the two crossings is the operative lower bound on $N ( \mathcal { P } _ { \mathrm { h e t } } , \mathcal { H } )$ . The last two columns report how often a prior draw leaves its hypothesis; this probability is charged in the heterogeneity envelope.
<table><tr><td> $B$ </td><td>known  $n _ { 8 0 }$ </td><td>heterogeneity  $n _ { 8 0 }$ </td><td>ratio</td><td> $\mathrm { P r } _ { 0 } ( G _ { 0 } ^ { c } )$ </td><td> $\mathrm { P r } _ { 1 } ( G _ { 1 } ^ { c } )$ </td></tr><tr><td>64</td><td>32,484</td><td>2,600</td><td>12.49</td><td>0.123</td><td>0.238</td></tr><tr><td>256</td><td>58,713</td><td>10,108</td><td>5.81</td><td>0.003</td><td>0.055</td></tr><tr><td>1,024</td><td>111,210</td><td>29,743</td><td>3.74</td><td>0.000</td><td>0.001</td></tr><tr><td>4,096</td><td>205,041</td><td>86,773</td><td>2.36</td><td>0.000</td><td>0.000</td></tr><tr><td>16,384</td><td>417,386</td><td>249,855</td><td>1.67</td><td>0.000</td><td>0.000</td></tr><tr><td>65,536</td><td>846,246</td><td>742,665</td><td>1.14</td><td>0.000</td><td>0.000</td></tr><tr><td>262,144</td><td>1,660,543</td><td>2,327,210</td><td>0.71</td><td>0.000</td><td>0.000</td></tr></table>

$$
\Re _ { n } ( { \mathcal { F } } _ { w } ) = { \frac { 1 } { 2 n } } \sum _ { b = 1 } ^ { B } \mathbb { E } | Z _ { b } | , \qquad Z _ { b } = \sum _ { i = 1 } ^ { n } \sigma _ { i } h ( S _ { i } ) \mathbf { 1 } \{ S _ { i } \in G _ { b } \} .\tag{9}
$$

The constant term has expectation zero. A summand in $Z _ { b } \ \mathrm { i s } \ \pm 1$ with probability $w _ { b } / 2$ each, and zero otherwise. Thus, putting $v = n w _ { b }$

$$
\begin{array} { r } { \mathbb { E } Z _ { b } ^ { 2 } = v , \qquad \mathbb { E } Z _ { b } ^ { 4 } \leq v + 3 v ^ { 2 } . } \end{array}
$$

Hölder’s inequality gives $\begin{array} { r } { \mathbb { E } | Z _ { b } | \geq ( \mathbb { E } Z _ { b } ^ { 2 } ) ^ { 3 / 2 } / ( \mathbb { E } Z _ { b } ^ { 4 } ) ^ { 1 / 2 } \geq v / \sqrt { 1 + 3 v } , } \end{array}$ , while $\mathbb { E } | Z _ { b } | \le \operatorname* { m i n } \{ v , \sqrt { v } \}$ Consequently

$$
\frac { 1 } { 4 } \sum _ { b } \operatorname* { m i n } \{ w _ { b } , \sqrt { w _ { b } / n } \} \leq \Re _ { n } ( \mathcal { F } _ { w } ) \leq \frac { 1 } { 2 } \sum _ { b } \operatorname* { m i n } \{ w _ { b } , \sqrt { w _ { b } / n } \} .\tag{10}
$$

If $n w _ { \mathrm { m i n } } \ge 1$ , this is $\Theta ( \sum _ { b } \sqrt { w _ { b } } / \sqrt { n } )$ . For a plateau and $n \geq k .$ , the heavy groups contribute $\Theta ( \sqrt { k / n } )$ and the residual contributes at most $\varepsilon _ { B } / 2$ . Therefore, around $n \asymp k / \gamma ^ { 2 }$ with $\varepsilon _ { B } \leq \gamma / 1 6 .$ the complexity is of order $\gamma .$ . Distribution-dependent complexity does distinguish our profiles; it is not one of the quantities matched in Theorem 2.

For indicator policies the squared $L ^ { 2 } ( P _ { 0 } ^ { S } )$ distance is

$$
\| \mathbf { 1 } _ { A _ { \theta } } - \mathbf { 1 } _ { A _ { \theta ^ { \prime } } } \| _ { 2 } ^ { 2 } = \sum _ { b } w _ { b } \mathbf { 1 } \{ \theta _ { b } \neq \theta _ { b } ^ { \prime } \} .
$$

At any fixed sufficiently small positive resolution, a plateau has logarithmic covering and packing numbers $\Theta ( k )$ : ignoring residual coordinates gives a $2 ^ { k }$ cover once the squared resolution exceeds $\varepsilon _ { B } .$ , and a constant-relative-distance binary code on the heavy coordinates gives a packing of size $\exp ( c k )$ . Thus the usual metric entropy sees the heavy coordinates. The point is a difference between estimation and testing rates, not failure of every distribution-dependent complexity measure.

## C.2 PROOF OF THE IUT RATE SEPARATION

Table 5: Two statistical scales on a k-plateau (constants and the residual suppressed).
<table><tr><td>Evidence</td><td></td><td>Signal Reference fluctuation Sample scale</td><td></td></tr><tr><td>Best (minimum) empirical policy risk</td><td></td><td> $\sqrt { k / n }$ </td><td> $k / \gamma ^ { 2 }$ </td></tr><tr><td>Quadratic family evidence</td><td> $\begin{array} { c } { { \gamma } } \\ { { \gamma ^ { 2 } } } \end{array}$ </td><td> $\sqrt { k } / n$ </td><td> $\sqrt { k } { \dot { / } } \gamma ^ { 2 }$ </td></tr></table>

Part (ii) of Theorem 3 is proved for the exact Clopper–Pearson IUT with any fixed positive allocation $\zeta _ { c } + \zeta _ { r } = \delta$ . Part (i) is proved for every level-δ EFC audit: on the main range the argument below only uses that the audit does not reject while the constructed policy $\widehat { A }$ is empirically feasible, and level δ is used only in the bounded ranges at the end. Constants below may depend on $p , \delta , \beta$ but not on $B , k , \gamma$

Upper bound. Let R be the residual groups, of reference mass $\varepsilon \leq \gamma / 1 6$ , with $\varepsilon = 0$ for $k = B$ For a heavy-coordinate sign pattern $\theta _ { H }$ , define

$$
T _ { H } = \sum _ { i } { \bf 1 } \{ S _ { i } \in A _ { \theta _ { H } } \ \backslash \ \mathcal { R } \} , \qquad W _ { H } = \sum _ { i } ( L _ { i } - \alpha ) { \bf 1 } \{ S _ { i } \in A _ { \theta _ { H } } \ \backslash \ \mathcal { R } \} .
$$

Their expectations are $n ( 1 - \varepsilon ) / 2$ and $n ( 1 - \varepsilon ) \gamma / 2$ . A union bound over only $2 ^ { k }$ heavy patterns and Hoeffding’s inequality give, with probability at least $1 - \eta ,$ simultaneously

$$
T _ { H } \geq n ( 1 - \varepsilon ) / 2 - C \sqrt { n ( k + \log ( 1 / \eta ) ) } , \qquad W _ { H } \geq n ( 1 - \varepsilon ) \gamma / 2 - C \sqrt { n ( k + \log ( 1 / \eta ) ) } .
$$

Also $N _ { \mathcal { R } } \leq n \varepsilon + C \sqrt { n \log ( 1 / \eta ) }$ on an event of the same order of probability. Any full policy has $T _ { A } \ge T _ { H }$ and $E _ { A } - \alpha T _ { A } \geq W _ { H } - \alpha N _ { \mathcal { R } }$ . Since $\varepsilon \leq \gamma / 1 6 .$ , at $n \geq C ^ { \prime } ( k + \log ( 1 / \eta ) ) / \gamma ^ { 2 }$ these inequalities imply $T _ { A } \ge n / 4$ and $E _ { A } / T _ { A } \geq \alpha + c \gamma$ for every policy. A one-sided Clopper–Pearson lower bound is at least the empirical proportion minus $\sqrt { \log ( 1 / \zeta _ { r } ) / ( 2 T _ { A } ) }$ whenever this difference is positive: this follows by inverting the Hoeffding bound for a binomial upper tail. Increasing $C ^ { \prime }$ makes every risk lower bound exceed $\alpha .$ . Taking $\eta < 1 - \beta$ proves $N _ { \mathrm { I U T } } \leq \dot { C } k / \gamma ^ { 2 }$ . This argument does not pay for the number of residual sign patterns.

A constructive empirical witness for the lower bound. Condition on the total number of errors $M ,$ , and write $m = n - M$ for the number of correct observations. In each heavy group choose the half with more correct observations; fix the + half in every residual group. This defines a policy $\widehat { A }$ using correct observations alone. Conditional on these observations and ${ \bar { M } } ,$ , the error observations have independent uniform scores, and any selected sign policy has score probability $1 / 2$ . Hence

$$
E _ { \widehat { A } } = M / 2 + Z , \qquad \mathbb { E } [ Z \mid M , \mathrm { c o r r e c t ~ d a t a } ] = 0 , \quad \mathbb { E } [ Z ^ { 2 } \mid M , \mathrm { c o r r e c t ~ d a t a } ] = M / 4 .
$$

If $C _ { b , + } , C _ { b , - }$ <sub>−</sub> are the heavy correct counts and $C _ { \mathcal { R } , + }$ is the correct count in the fixed residual halves, then

$$
C _ { \widehat { A } } = m / 2 + G , \qquad G = \frac { 1 } { 2 } \sum _ { b \leq k } \left| C _ { b , + } - C _ { b , - } \right| + C _ { \mathcal { R } , + } - C _ { \mathcal { R } } / 2 .
$$

The last term has conditional mean zero. Each heavy difference has variance $v = m ( 1 - \varepsilon ) / k$ and fourth moment at most $v + 3 v ^ { 2 }$ , so

$$
\mathbb { E } [ G \mid M ] \geq { \frac { m ( 1 - \varepsilon ) } { 2 { \sqrt { 1 + 3 m ( 1 - \varepsilon ) / k } } } } \geq c _ { 0 } \operatorname* { m i n } \{ m , { \sqrt { m k } } \} .
$$

The second inequality holds uniformly for $\varepsilon \le 1 / 2$ . Replacing one correct observation changes the maximum selected correct count by at most one. Efron–Stein therefore gives $\operatorname { V a r } ( G \mid M ) \stackrel { = } { \leq } m / 2$

(the weaker bound 2m also suffices). Chebyshev, binomial concentration for M, and the conditional variance of Z imply, except with probability at most $C _ { 0 } ( 1 / n + 1 / k )$ ,

$$
\begin{array} { r } { G \geq c _ { 1 } \operatorname* { m i n } \{ n , \sqrt { n k } \} , \quad | M - n p | + 2 | Z | \leq c _ { 2 } \operatorname* { m i n } \{ n , \sqrt { n k } \} , } \end{array}
$$

where $c _ { 2 } > 0$ can be chosen sufficiently small relative to $p c _ { 1 }$ by changing $C _ { 0 }$ . Here the first bound also uses $m \geq ( 1 - p ) n / 2 ;$ its failure has probability $O ( 1 / n )$

For $\gamma \leq \gamma _ { 0 } ( p )$ sufficiently small and $n \le c k / \gamma ^ { 2 }$ with c sufficiently small, $n \gamma \leq c _ { 3 } \operatorname* { m i n } \{ n , { \sqrt { n k } } \}$ Since $\alpha = p - \gamma \ge p / 2$ , the same event yields

$$
\begin{array} { r } { T _ { \widehat { A } } = n / 2 + G + Z \geq n / 2 , \qquad E _ { \widehat { A } } - \alpha T _ { \widehat { A } } = \frac { 1 } { 2 } ( M - \alpha n ) + ( 1 - \alpha ) Z - \alpha G \leq 0 . } \end{array}
$$

Thus $\widehat { A }$ is empirically feasible, so no EFC audit rejects; for the exact IUT this is Lemma 1. For large enough $n , k ,$ its rejection probability is below $\beta$ throughout $n \leq c k / \gamma ^ { 2 }$

For completeness, the bounded-k cases follow from the information-theoretic lower bound $N \geq$ $c ^ { \prime } \sqrt { k } / \gamma ^ { 2 }$ , which is at least a constant times $k / \gamma ^ { 2 }$ on any fixed bounded range. For bounded n and sufficiently large $k ,$ the mixture ceiling in Proposition 3 is below $\beta ,$ uniformly over bounded $\gamma \colon \| w \| _ { 2 } ^ { 2 } = O \dot { ( 1 / k ) }$ makes its exponent ${ \cal O } \overline { { ( n ^ { 2 } \gamma ^ { 4 } / k ) } }$ . Combining these ranges proves the claimed uniform lower rate. This proof controls the coverage screen as well as risk; minimizing error counts alone would not suffice.

## C.3 PROOF OF PROPOSITION 4

Every rule of the sign family has reference coverage exactly κ. The estimation cost does not depend on this boundary case.

Proposition 4 (Estimation cost with coverage slack). Split each group ofa k-plateau into q equal parts and let each policy accept m parts per group (Proposition $\delta ) .$ . With reference coverage $m / q ,$ a fixed floor $\kappa < m / q$ and residual mass at most $m \gamma / ( 1 6 q )$ , there are $\gamma _ { 0 } , c _ { 1 } , C _ { 1 } > 0$ depending only on $( p , q , m , \kappa , \delta , \dot { \beta } )$ such that, $f o r \gamma \leq \gamma _ { 0 }$ , every level-δ EFC audit has power below $\beta$ at $P _ { 0 } f o r$ all $n \leq c _ { 1 } k / \gamma ^ { 2 }$ , while the exact IUT has power at least $\beta f o r n \ge C _ { 1 } k / \gamma ^ { 2 }$

The witness chooses, in each heavy group, the m parts with the most correct observations; with slack the coverage screen is passed automatically, and the gain in correct counts again scales as min $\{ n , { \sqrt { n k } } \}$ }. Use the construction of Proposition 8: $P _ { 0 }$ has uniform scores and independent $L \sim$ $\operatorname { B e r n } ( p )$ , each group $G _ { \ell }$ is split into q equal parts, a policy selects m parts per group, $c = m / q > \kappa ,$ $\alpha = p - \gamma$ , and the residual groups have total mass $\varepsilon \leq c \gamma / 1 6$

Upper bound. Repeat the upper-bound argument of Appendix C.2 with the ${ \binom { q } { m } } ^ { k }$ heavy patterns in place of $2 ^ { k } { : }$ : the union bound costs k log $\textstyle { \binom { q } { m } }$ , every policy has $T _ { A } \geq n c / 2$ and $E _ { A } - \alpha T _ { A } \geq n c \gamma / 4$ once $n \geq C k / \gamma ^ { 2 }$ , and the Clopper–Pearson inversion makes every risk bound exceed α.

A top-m inequality. If $x _ { 1 } , \ldots , x _ { q }$ sum to zero, the sum of the m largest is at least $\frac { \operatorname* { m i n } \{ m , q - m \} } { 2 q } \sum _ { a } \left| x _ { a } \right|$ . Indeed, let $\begin{array} { r } { P = \sum _ { a } ( x _ { a } ) _ { + } = \frac { 1 } { 2 } \sum _ { a } \left| x _ { a } \right| } \end{array}$ and let r be the number of positive entries. If $m \leq r$ , the m largest positive entries sum to at least $( m / r ) P \ge ( m / q ) P$ . Otherwise the top m contain all positive entries and the $m - r$ negative entries of smallest magnitude, whose total magnitude is at most ${ \frac { m - r } { q - r } } P _ { }$ , so the sum is at least $\begin{array} { r } { \frac { q - m } { q - r } P \geq \frac { q - m } { q } P } \end{array}$

Witness. Condition on the error count M and on all correct observations; let $m _ { 0 } = n - M$ . In each heavy group choose the m parts with the most correct observations, and fix m parts in every residual group. Conditionally, the error observations have independent uniform scores and every policy has score probability $c ,$ so the accepted errors are $E = c M + Z$ with $\mathbb { E } Z = 0$ and Var $Z = \bar { M } c ( 1 \bar { - } c )$ The accepted correct count is $c m _ { 0 } + G$ , where

$$
G = \sum _ { b \leq k } \Big ( \sum _ { a \in \widehat { C } _ { b } } N _ { b , a } - c N _ { b } \Big ) + R ,
$$

$N _ { b , a }$ is the correct count in part a of group $\begin{array} { r } { b , \ N _ { b } \ = \ \sum _ { a } N _ { b , a } , } \end{array}$ , and $R$ is the centred residual contribution. By the top-m inequality applied to $x _ { a } = N _ { b , a } - N _ { b } / q$ , each heavy term is at least $\frac { \operatorname* { m i n } \{ m , q - m \} } { 2 q } \sum _ { a } \left| N _ { b , a } - N _ { b } / q \right|$ . Each $N _ { b , a } - N _ { b } / q$ is a sum of $m _ { 0 }$ independent centred terms bounded by one, with variance $v = m _ { 0 } w _ { b } ( q - 1 ) / q ^ { 2 }$ and fourth moment at most $v + 3 v ^ { 2 }$ . Hölder’s inequality gives $\mathbb { E } | N _ { b , a } - N _ { b } / q | \geq v / \sqrt { 1 + 3 v } .$ , and therefore $\mathbb { E } [ G \mid M ] \geq c _ { 0 }$ min $\{ m _ { 0 } , \sqrt { m _ { 0 } k } \}$ with $c _ { 0 } > 0$ depending on $( q , m )$ . Moving one correct observation changes the top-m sums by at most one and $c \sum d$ by at most $2 c ,$ so Efron–Stein gives $\operatorname { V a r } ( G \mid M ) \leq 9 m _ { 0 } / 2$

Empirical feasibility. As in Appendix C.2, Chebyshev’s inequality and binomial concentration give, except with probability $C _ { 0 } ( \overset { \cdot } { 1 } / n + 1 / k )$

$$
\begin{array} { r } { G \geq c _ { 1 } ^ { \prime } \operatorname* { m i n } \{ n , \sqrt { n k } \} , \qquad \lvert M - n p \rvert + 2 \lvert Z \rvert \leq c _ { 2 } ^ { \prime } \operatorname* { m i n } \{ n , \sqrt { n k } \} , } \end{array}
$$

with $c _ { 2 } ^ { \prime }$ small relative to $\alpha c _ { 1 } ^ { \prime }$ and to $c - \kappa$ . The accepted count is $T = c n + G + Z \geq c n - | Z | \geq \kappa n$ since $\begin{array} { r } { | Z | \le ( c - \kappa ) \eta } \end{array}$ n on this event. Moreover

$$
E - \alpha T = c ( M - p n ) + c \gamma n + ( 1 - \alpha ) Z - \alpha G \le 0
$$

whenever $c \gamma n \leq c _ { 3 } \operatorname* { m i n } \{ n , { \sqrt { n k } } \}$ , which holds for $n \leq c _ { 1 } k / \gamma ^ { 2 }$ and $\gamma \leq \gamma _ { 0 }$ . The selected policy is then empirically feasible, and no EFC audit rejects. The bounded-k and bounded-n ranges are handled exactly as in Appendix C.2, using the lower bound $\Omega ( \sqrt { k } / \gamma ^ { 2 } )$ and the product-mixture ceiling of Proposition 8. □

## C.4 EXPLICIT CONSTANTS WHEN ONLY THE LOSS RATE IS UNKNOWN

Theorem 4 relies on a generic independence tester. When the score marginal is known and uniform, a pilot-and-collision audit gives explicit constants.

Proposition 5 (Adaptation to an unknown constant risk). On the uniform sign family $B = k ,$ let $P _ { p }$ have uniform scores over the 2k halves and independent $L \sim \operatorname { B e r n } ( p )$ , and let $\mathcal { P } _ { a , \alpha , \gamma } = \{ P _ { p } : p \stackrel { . } { \in }$ $[ a , 1 - a ] , p \geq \alpha + \gamma \}$ . Fix $0 < a < 1 / 3 , \alpha \in [ \bar { a , } 1 - 2 a ]$ ] and $0 < \delta < \beta < 1$ . For sufficiently small $0 < \gamma \leq a _ { \mathrm { { \scriptscriptstyle \mathscr { M } } } }$ , the audit below has level δ over the unrestricted null, does not take $p$ as input, and has power $\beta$ over $\mathcal { P } _ { a , \alpha , \gamma }$ with $O _ { a , \delta , \beta } ( \sqrt { k } / \gamma ^ { 2 } )$ samples, which is optimal.

We prove Proposition 5. The score-space partition is known, but the loss probability p is not an input to the audit. For power, the reference class has uniform scores on the 2k halves and independent $L \sim \operatorname { B e r n } ( p )$ with $p \in [ a , 1 - a ]$ and $p \ge \alpha + \gamma$ . Type-I validity continues to cover every law in the unrestricted existential null, without uniform-score or score–loss independence assumptions.

Set $\epsilon =$ min $\{ \delta , ( 1 - \beta ) / 3 \}$ . Use m pilot observations to estimate $p$ by their mean loss projected onto $[ a , 1 - a ]$ , denoted $\widehat { p . } \ : \mathrm { I f } \ : \widehat { p } < \alpha + \gamma / 2$ , do not reject. Otherwise apply the collision test from Theorem 6 to an independent batch of n observations with the candidate reference $P _ { \widehat { p } } .$ . For each candidate,

$$
\Delta _ { \widehat { p } } = \frac { ( \widehat { p } - \alpha ) ^ { 2 } } { 2 \widehat { p } ( 1 - \widehat { p } ) } \geq \gamma ^ { 2 } / 2 , \qquad p _ { \operatorname* { m i n } } \geq a / ( 2 k ) , \qquad d = 4 k .
$$

All categorical cells are reference-positive. A common sufficient size for the second batch is

$$
n \geq \operatorname* { m a x } \left\{ 2 , \left\lceil \frac { 3 2 } { \epsilon } \operatorname* { m a x } \left\{ \frac { 4 \sqrt { k } } { \gamma ^ { 2 } } , \frac { 2 } { \gamma ^ { 2 } } + \frac { 2 \sqrt { k } } { \gamma \sqrt { a } } \right\} \right\rceil \right\} .\tag{11}
$$

Indeed, $\sqrt { d - 1 } / \Delta \leq 4 \sqrt { k } / \gamma ^ { 2 }$ and $u _ { 0 } / \Delta = 1 / \Delta + 1 / \sqrt { \Delta p _ { \mathrm { m i n } } }$ is bounded by the second expression. Conditional on any pilot, the collision test has level at most ϵ for every feasible law; a failed pilot screen never rejects. Therefore the entire procedure has level at most δ without assuming the power model.

Under $P _ { p } ,$ projection cannot increase squared estimation error, so $\mathbb { E } ( \widehat { p } - p ) ^ { 2 } \leq 1 / ( 4 m )$ . For candidate Bernoulli laws,

$$
\mathrm { K L } ( P _ { p } \| P _ { \widehat { p } } ) = \mathrm { k l } ( p \| \widehat { p } ) \leq \frac { ( p - \widehat { p } ) ^ { 2 } } { a ( 1 - a ) } .
$$

Pinsker and Jensen yield

$$
\mathbb { E } \operatorname { T V } ( P _ { p } ^ { n } , P _ { \widehat { p } } ^ { n } ) \leq \sqrt { \frac { n } { 8 m a ( 1 - a ) } } .
$$

Also $P _ { p } ( \widehat { p } < \alpha + \gamma / 2 ) \leq e ^ { - m \gamma ^ { 2 } / 2 }$ . Taking

$$
m \geq \operatorname* { m a x } \left\{ \frac { n } { 8 a ( 1 - a ) \epsilon ^ { 2 } } , \frac { 2 \log ( 1 / \epsilon ) } { \gamma ^ { 2 } } \right\}\tag{12}
$$

bounds both terms by ϵ. On a passed screen the test has power at least $1 - \epsilon$ under its candidate $P _ { \widehat { p } } ;$ transferring to $P _ { p }$ and averaging gives power at least $1 - 3 \epsilon \geq \beta$ . The total $m + n$ is $O _ { a , \delta , \beta } ( \sqrt { k } / \gamma ^ { 2 } )$ The lower bound follows by fixing $p = \alpha + \gamma$ in the reference class and applying the uniform-profile mixture bound, with constants uniform over $[ a , 1 - a ]$

The statistic can be computed from cell counts in $O ( n + k )$ time using $\begin{array} { r } { U _ { n } ~ = ~ [ \sum _ { i } N _ { j } ( N _ { j } - } \end{array}$ $1 ) / p _ { j } ] / [ n ( n - 1 ) ] - 1$ . Its conservative constants are not claimed to be practically competitive. The result establishes the rate and refutes a universal interpretation of $k / \gamma ^ { 2 }$ as the unavoidable cost of not knowing p. It does not solve adaptation to an arbitrary unknown score marginal or conditional risk function.

## D PROOF OF THE MATCHED SAME-CARDINALITY COMPLEXITY SPECTRUM

## D.1 EXACT SINGLETON SUFFICIENCY AND AUDIT EQUIVALENCE

Every component has exactly

$$
\mathrm { T V } ( P _ { 0 } , Q _ { \theta } ) = \gamma ,\tag{13}
$$

$$
\begin{array} { r } { \mathrm { K L } ( P _ { 0 } \| Q _ { \theta } ) = \frac 1 2 \mathrm { k l } ( p \| p - \gamma ) + \frac 1 2 \mathrm { k l } ( p \| p + \gamma ) , } \end{array}\tag{14}
$$

$$
\begin{array} { r } { \mathrm { K L } ( Q _ { \theta } | | P _ { 0 } ) = \frac { 1 } { 2 } \mathrm { k l } ( p - \gamma | | p ) + \frac { 1 } { 2 } \mathrm { k l } ( p + \gamma | | p ) . } \end{array}\tag{15}
$$

For reference coverage c, accepted risk $r , 0 < \kappa \leq c ,$ , and $\alpha = r - \gamma$ with $0 < \gamma < r$ , the exact projections are as follows. Moreover, putting $d = \operatorname { k l } ( r - \gamma \Vert r )$ and $t ^ { * } = \operatorname* { m a x } \{ \kappa , c e ^ { - d } / ( 1 - c + c e ^ { - d } ) \}$

$$
\operatorname* { i n f } _ { Q \in { \mathcal { H } } ( \{ A \} ) } \mathrm { T V } ( P _ { 0 } , Q ) = c \gamma ,\tag{16}
$$

$$
\operatorname* { i n f } _ { Q \in { \mathcal { H } } ( \{ A \} ) } \mathrm { K L } ( P _ { 0 } \| Q ) = c \mathrm { k l } ( r \| r - \gamma ) ,\tag{17}
$$

$$
\operatorname* { i n f } _ { Q \in { \mathcal { H } } ( \{ A \} ) } \mathrm { K L } ( Q \| P _ { 0 } ) = \ker ( t ^ { * } \| c ) + t ^ { * } d .\tag{18}
$$

The four-cell sufficient-statistic reduction is proved below. The constant-risk case used below has $r = s = p$ and gives these formulas as a special case. Note that the canonical hiding component $Q _ { \theta }$ in Equation (5) is not the closest feasible singleton law: for $c = 1 / 2$ , Equation (13) gives $\mathrm { T V } ( P _ { 0 } , Q _ { \theta } ) = \gamma$ , whereas the singleton projection in Equation (16) is $\gamma / 2$

Proposition 6 (Singleton experiments are matched). For a policy A whosefour $( { \bf 1 } \{ S \in A \} , L )$ reference cells are positive, the unrestricted singleton testing problem at every sample size is determined by this four-cell law. Hence every member of every sign family has the same singleton minimax complexity.

We prove Proposition 6 without assuming that L is independent of the score. Let $M = \mathbf { 1 } \{ S \in A \}$ and $T ( \dot { S } , L ) = ( \dot { M } , L )$ . The singleton contract depends on a law only through the four probabilities of $T$ Fix once and for all a Markov kernel $K ( d s \mid m , l )$ supported on A when $m = 1$ and on $A ^ { c }$ when $m = 0 .$ , taking $K ( B \mid m , l ) = P _ { 0 } ( S \in \dot { B } , \dot { M } = m , \bar { L } = l ) / P _ { 0 } ( M = m , L = l )$ , the normalized restriction of $P _ { 0 }$ to that finite event. Proposition 6 assumes all four cells are positive, so no general disintegration theorem is needed.

Every feasible four-cell law q has a canonical lift $Q _ { q } \mathrm { : }$ draw $T \sim q$ and then $S \sim K ( \cdot \mid T )$ Conversely, $T _ { \# } Q$ is feasible whenever $Q \in { \mathcal { H } } ( \{ A \} )$ . Hence data processing gives one direction for every divergence and every test.

The testing reduction is exact for every sample size. Given any original-sample test $\phi _ { n }$ , define

$$
\psi _ { n } ( t ^ { n } ) = \int \phi _ { n } \ l ( s ^ { n } , l ^ { n } ) \prod _ { i = 1 } ^ { n } K \ l ( d s _ { i } \mid t _ { i } ) .
$$

For every canonical lift $Q _ { q } , \mathbb { E } _ { Q _ { a } ^ { n } } \phi _ { n } = \mathbb { E } _ { q ^ { n } } \psi _ { n }$ , and the same identity holds under $P _ { 0 } ^ { n }$ . Therefore every valid original test induces a four-cell test with the same power, while the converse follows by composition with $T ^ { n }$ . The unrestricted singleton minimax problem thus depends on $( P _ { 0 } , A )$ only through $T _ { \# } P _ { 0 }$ . In particular, two policies with identical reference four-cell laws have identical singleton audit complexity even when $\begin{array} { r } { \dot { \eta } ( s ) = P _ { 0 } ( L = 1 | S = s ) } \end{array}$ is nonconstant.

For the explicit projections, write

$$
c = P _ { 0 } ( M = 1 ) , \quad r = P _ { 0 } ( L = 1 \mid M = 1 ) , \quad s = P _ { 0 } ( L = 1 \mid M = 0 ) , \quad \alpha = r - \gamma .
$$

For TV, $f ( M , L ) = M ( L - \alpha )$ ) has oscillation one. Under $P _ { 0 } , \mathbb { E } f = c \gamma$ , whereas every feasible Q has $\mathbb { E } _ { Q } \dot { f } \le 0 .$ , so $\mathrm { T V } ( P _ { 0 } , Q ) \ge c \gamma$ . Equality is attained in the four-cell problem by preserving $P ( M = { \bar { 1 } } ) = c$ and the outside conditional risk s while changing the accepted risk from $r \tan r - \gamma ;$ the canonical lift preserves TV exactly.

For the KL projections, write $t = Q ( M = 1 ) , u = Q ( L = 1 \mid M = 1 )$ , and $v = Q ( L = 1 \mid M = 0 )$ Data processing and the Bernoulli chain rule give

$$
\begin{array} { r l } & { \mathrm { K L } ( P _ { 0 } \| Q ) \geq \mathrm { k l } ( c \| t ) + c \mathrm { k l } ( r \| u ) + ( 1 - c ) \mathrm { k l } ( s \| v ) , } \\ & { \mathrm { K L } ( Q \| P _ { 0 } ) \geq \mathrm { k l } ( t \| c ) + t \mathrm { k l } ( u \| r ) + ( 1 - t ) \mathrm { k l } ( v \| s ) . } \end{array}
$$

The feasible constraints are $t \geq \kappa$ and $u \leq r - \gamma$ . The forward direction is minimized at $( t , u , v ) =$ $\left( c , r - \gamma , s \right)$ , giving $c \mathrm { k l } ( r \| r - \gamma )$

For the reverse direction first set $( u , v ) = ( r - \gamma , s )$ , optimal for every fixed t. With $d = \operatorname { k l } ( r - \gamma \| r )$ the remaining objective is

$$
h ( t ) = \mathrm { k l } ( t \| c ) + t d , \qquad t \geq \kappa .
$$

Its unconstrained minimizer is $t _ { 0 } = c e ^ { - d } / ( 1 - c + c e ^ { - d } )$ , so the constrained optimizer is $t ^ { * } =$ $\operatorname* { m a x } \{ \kappa , t _ { 0 } \}$ and the minimum is $\ k \mathbf { l } ( t ^ { * } \| c ) + t ^ { * } d .$ . Canonical lifts attain the data-processing bounds because the likelihood ratio depends only on $T .$ . This proves Equation (18). The score-independent special case is obtained by $r = s = p$

## D.2 COMMON COMPONENT INFORMATION

Every policy $A _ { \theta }$ accepts one half of each group, so $P _ { 0 } ( S \in A _ { \theta } ) = 1 / 2$ and under $( 5 ) , Q _ { \theta } ( L = 1 \mid$ $S \in A _ { \theta } ) = p - \gamma = \alpha$ . Half of the score mass shifts the Bernoulli parameter down by $\gamma$ and half shifts it up by γ, independently of group weights. Summing the two-point channel distances gives (15) and $\mathrm { T V } ( P _ { 0 } , Q _ { \theta } ) = \gamma$

For independent uniform signs $\theta , \theta ^ { \prime }$ , the standardized Bernoulli calculation gives

$$
\Xi ( \theta , \theta ^ { \prime } ) = { \frac { \gamma ^ { 2 } } { p ( 1 - p ) } } \sum _ { \ell = 1 } ^ { B } w _ { \ell } \theta _ { \ell } \theta _ { \ell } ^ { \prime } = \lambda \sum _ { \ell = 1 } ^ { B } w _ { \ell } \varepsilon _ { \ell } ,
$$

where $\varepsilon _ { \ell }$ are i.i.d. Rademacher variables. The nonzero covariance eigenvalues are proportional to $w _ { \ell }$ giving $\begin{array} { r } { \dot { d } _ { \mathrm { e f f } } = ( \sum _ { \ell } w _ { \ell } ^ { 2 } ) ^ { - 1 } } \end{array}$

Proposition 7 (Combinatorial invariance of the matched sign families). For any two positive profiles $w , w ^ { \prime } \in \Delta _ { B - 1 }$ , the corresponding sign families have the same nonempty membership-signature incidence structure up to relabeling of the 2B half-group atoms. In particular, both have cardinality $2 ^ { B }$ , VC dimension exactly $B ,$ and growth function $\Breve { \Pi } _ { A } ( \nearrow ) = 2 ^ { \operatorname* { m i n } \{ m , B \} }$ . Thus changing the profile changes only the $P _ { 0 }$ -masses assigned to a common combinatorial set system, not its unweighted class complexity.

Remark 2 (Pairwise matching inside the plateau spectrum). If $\theta ^ { ( \ell ) }$ is obtained from θ by flipping only coordinate ℓ, then $A _ { \theta } { \bar { \triangle _ { A } } } _ { \theta ( \ell ) } = G _ { \ell }$ and hence $P _ { 0 } ( A _ { \theta } \triangle A _ { \theta ^ { ( \ell ) } } ) = w _ { \ell }$ . Thus adjacent pairwise membership laws reveal the plateau weights. A pairwise-matched analogue of Theorem 2 therefore requires a qualitatively different construction; the Supplementary Note gives such a construction for the weaker but already nontrivial question of one-observation family hiding.

## D.3 PROOF OF PROPOSITION 7: FIXED INCIDENCE AND VC COMPLEXITY

For every positive profile, quotient the score space by policy-membership signature. The nonempty signature atoms are exactly the 2B half-groups $G _ { \ell } ^ { + }$ and $G _ { \ell } ^ { - } ;$ : policy $A _ { \theta }$ contains precisely $G _ { \ell } ^ { \theta _ { \ell } }$ from group ℓ. This 2B-atom incidence matrix depends on B and the sign labels but not on the mass profile $w .$ Hence any two profiles have isomorphic quotient set systems; only the reference masses of the atoms differ.

For the VC dimension, choose one point from $G _ { \ell } ^ { + }$ for each of $m \ \leq \ B$ distinct groups. The corresponding m membership bits can be set arbitrarily by the m sign coordinates, so every such m is shattered and ${ \mathrm { V C d i m } } ( { \mathcal { A } } ) { \overset { * } { \geq } } B$ . Conversely, among any $B + 1$ points two lie in the same group. If they lie in the same half, every policy gives them identical membership; if they lie in opposite halves, every policy gives complementary membership. In either case the pair cannot realize all four binary labelings, so no $B + 1$ points are shattered. Thus ${ \mathrm { V C d i m } } ( { \mathcal { A } } ) = { \dot { B } }$

Finally, for $m \le B$ the shattered construction realizes $2 ^ { m }$ traces. For $m > B ,$ , the family contains only $\mathrm { 2 } ^ { \ J }$ policies, while an m-point set containing one point from each $G _ { \ell } ^ { + }$ plus additional distinct points in any already represented half-group still distinguishes all $2 ^ { B }$ sign vectors. Therefore $\mathbf { \dot { I } } \mathrm { I } _ { A } ( m ) = 2 ^ { \mathbf { \dot { m } } \mathrm { i n } \{ m , B \} }$

## D.4 PROOF OF PROPOSITION 3

Let w be any positive group-mass vector and let $\varepsilon _ { \ell } = \theta _ { \ell } \theta _ { \ell } ^ { \prime }$ for independent uniform sign vectors. Then

$$
\Xi ( \theta , \theta ^ { \prime } ) = \lambda \sum _ { \ell = 1 } ^ { B } w _ { \ell } \varepsilon _ { \ell } , \qquad \lambda = \frac { \gamma ^ { 2 } } { p ( 1 - p ) } .
$$

By Lemma $3 , 1 + \chi ^ { 2 } ( R _ { n } \| P _ { 0 } ^ { n } ) = \mathbb { E } ( 1 + \Xi ) ^ { n }$ . Since $1 + x \leq e ^ { x }$ for $x > - 1$ and the $\varepsilon _ { \ell }$ are independent Rademacher variables,

$$
\begin{array} { l } { \displaystyle 1 + \chi ^ { 2 } ( R _ { n } \| P _ { 0 } ^ { n } ) \leq \mathbb { E } \exp \left( n \lambda \sum _ { \ell } w _ { \ell } \varepsilon _ { \ell } \right) } \\ { \displaystyle \qquad = \prod _ { \ell = 1 } ^ { B } \cosh ( n \lambda w _ { \ell } ) \leq \exp \left( \frac { n ^ { 2 } \lambda ^ { 2 } } { 2 } \sum _ { \ell = 1 } ^ { B } w _ { \ell } ^ { 2 } \right) , } \end{array}
$$

where cosh $t \leq e ^ { t ^ { 2 } / 2 }$ . Corollary 3 now gives the ceiling of Proposition 3. Choosing the constant in $n \leq c / ( \lambda \| w \| _ { 2 } )$ so that the right-hand side stays below β yields $N _ { w } = \Omega ( \sqrt { d _ { \mathrm { e f f } } } / \gamma ^ { 2 } )$

Two uniform bounds delimit the entire sign-family class. First, any one canonical component has directed KL $\Theta _ { p } ( \gamma ^ { 2 } )$ from $P _ { 0 } ,$ , so the standard two-point lower bound (Le Cam and Yang, 2000) gives $N _ { w } = \Omega _ { p } ( 1 / \dot { \gamma ^ { 2 } } )$ for every profile. Second, for any witness policy $A _ { \theta }$ , feasibility implies

$$
Q ( A _ { \theta } , L = 0 ) \geq \frac { 1 - \alpha } { 2 } , \qquad P _ { 0 } ( A _ { \theta } , L = 0 ) = \frac { 1 - p } { 2 } ,
$$

so every law in the full existential null is at least $\gamma / 2$ away after the 4B half-group/label pushforward. Generic known-distribution identity testing on 4B categories therefore gives $N _ { w } = O ( \sqrt { B } / \gamma ^ { 2 } )$ uniformly in w. Consequently the ratio between any two complexities in this matched sign-family class is at most $O ( { \sqrt { B } } )$ . Theorem 2 attains every $\sqrt { k }$ order along the plateau subclass and in particular attains this maximal endpoint spread.

## D.5 PROOF OF THEOREM 2: PLATEAU SPECTRUM

For $k < B$ , the plateau profile $w _ { \ell } ^ { ( k ) } = ( 1 - \varepsilon _ { B } ) / k ( \ell \le k ) , w _ { \ell } ^ { ( k ) } = \varepsilon _ { B } / ( B - k ) ( \ell > k ) , \varepsilon _ { B } = B ^ { - 2 } ,$ satisfies

$$
\| w ^ { ( k ) } \| _ { 2 } ^ { 2 } = \frac { ( 1 - \varepsilon _ { B } ) ^ { 2 } } { k } + \frac { \varepsilon _ { B } ^ { 2 } } { B - k } , \qquad \varepsilon _ { B } = B ^ { - 2 } .
$$

Uniformly over $1 \leq k < B$ , the second term is $o ( 1 / k )$ and $( 1 - \varepsilon _ { B } ) ^ { 2 } = 1 + o ( 1 ) \quad$ ; for $k = B$ $\| \boldsymbol { w } ^ { ( B ) } \| _ { 2 } ^ { 2 } = 1 / B$ . Hence $d _ { \mathrm { e f f } } ( w ^ { ( k ) } ) = k \{ 1 + o ( 1 ) \}$ } uniformly in k. Proposition 3 therefore gives

$$
N _ { B , k } = \Omega \left( \frac { \sqrt { k } } { \gamma _ { B } ^ { 2 } } \right) .
$$

It remains to prove the matching upper bound against the full unrestricted existential null. For $k < B$ let

$$
R = \bigcup _ { \ell = k + 1 } ^ { B } G _ { \ell } , \qquad P _ { 0 } ( R ) = \varepsilon _ { B } ,
$$

and retain separately the two halves of each heavy group $G _ { 1 } , \ldots , G _ { k }$ . Push $( S , L )$ to the $( 4 k + 2 )$ categories consisting of heavy-group half/label pairs plus $R \times \{ 0 , 1 \}$ . Let $Q$ be any law in the full null and let $A _ { \theta }$ be a witness. Write $A _ { H } = A _ { \theta } \cap R ^ { c }$ for the accepted halves of the heavy groups.

If $| Q ( R ) - \varepsilon _ { B } | \geq \gamma _ { B } / 1 6$ , the pushed-forward laws already differ in TV by at least $\gamma _ { B } / 1 6$ . Otherwise $Q ( R ) \leq \varepsilon _ { B } + \gamma _ { B } / 1 6$ . Feasibility of $A _ { \theta }$ gives

$$
Q ( A _ { \theta } , L = 0 ) \geq ( 1 - \alpha ) Q ( A _ { \theta } ) \geq \frac { 1 - p + \gamma _ { B } } { 2 } ,
$$

and hence

$$
\begin{array} { l } { { Q ( A _ { H } , L = 0 ) \geq \displaystyle \frac { 1 - p + \gamma _ { B } } { 2 } - Q ( R ) , } } \\ { { P _ { 0 } ( A _ { H } , L = 0 ) = \displaystyle \frac { ( 1 - p ) ( 1 - \varepsilon _ { B } ) } { 2 } . } } \end{array}
$$

Therefore

$$
Q ( A _ { H } , L = 0 ) - P _ { 0 } ( A _ { H } , L = 0 ) \geq \frac { 7 \gamma _ { B } } { 1 6 } - \frac { 1 + p } { 2 } \varepsilon _ { B } \geq \frac { 3 \gamma _ { B } } { 8 }
$$

whenever $\varepsilon _ { B } \leq \gamma _ { B } / 1 6$ . Thus every law in the full null is at TV distance at least $\gamma _ { B } / 1 6$ from $P _ { 0 }$ after a pushforward with $4 k + 2$ categories. Known-distribution identity testing on $m = 4 k + 2$ categories uses ${ \cal O } ( \sqrt { m } / \gamma _ { B } ^ { 2 } ) = { \cal O } ( \sqrt { k } / \gamma _ { B } ^ { 2 } )$ ) samples. For $k = B .$ , the residual is empty and the same argument reduces to the standard 4B pushforward. Combining with the lower bound proves Theorem 2, and $d _ { \mathrm { e f f } } ( w ^ { ( k ) } ) = k \{ 1 + o ( 1 ) \}$ as computed above.

The condition $\varepsilon _ { B } \leq \gamma _ { B } / 1 6$ also makes the vanishing-gap statement stable when $\varepsilon _ { B } / \gamma _ { B }$ tends to a nonzero constant below $1 / 1 6 \colon$ the displayed separation remains a fixed positive fraction of $\gamma _ { B }$ , so no hidden constant degenerates in the pushforward upper bound.

## D.6 BALANCED Q-ARY SPECTRUM WITH POSITIVE COVERAGE SLACK

The binary construction uses reference coverage $1 / 2$ only for notation. The following balanced construction simultaneously extends the matched spectrum to rational reference coverage and removes the boundary condition $P _ { 0 } ( A ) = \kappa$

Proposition 8 (Matched spectrum with positive coverage slack). Fix a rational reference coverage $c = m / q \in ( 0 , 1 )$ with integers $1 \ \leq \ m \ < \ q ,$ a contract floor $0 < \kappa \leq c , p \in ( 0 , 1 )$ , and $0 < \delta < \beta < 1 . { \cal L e t } \gamma = \gamma _ { B }$ satisfy

$$
\gamma _ { B } < p , \qquad \frac { c } { 1 - c } \gamma _ { B } < 1 - p , \qquad B ^ { - 2 } \leq \frac { c \gamma _ { B } } { 1 6 } .
$$

For each $k \in \{ 1 , \ldots , B \}$ use the plateau profile $w ^ { ( k ) }$ of Section 4. There is a family with $\begin{array} { r } { J _ { B } = \binom { q } { m } ^ { B } } \end{array}$ policies, common reference $P _ { 0 } ,$ , common raw support, common contract $\left( \kappa , p - \gamma _ { B } \right)$ , and exact oneobservation hiding such that every policy has reference coverage exactly c and the same unrestricted singleton testing problem, while, uniformly in k,

$$
N _ { B , k } ^ { ( c , \kappa ) } = \Theta _ { p , c , \kappa , \delta , \beta } \left( \frac { \sqrt { k } } { \gamma _ { B } ^ { 2 } } \right) .
$$

Consequently,for everyfixed $c > \kappa$ the construction has fixed positive coverage slack $c - \kappa f o r$ every member, and its endpoint complexity ratio is $\Theta _ { c } ( \sqrt { \log { J _ { B } } } )$

Construction and exact hiding. Partition $[ 0 , 1 ]$ into groups $G _ { 1 } , \ldots , G _ { B }$ of masses $w _ { 1 } , \ldots , w _ { B }$ and split every group into q equal-mass consecutive subintervals $G _ { \ell , 1 } , \ldots , G _ { \ell , q }$ . A local policy chooses an m-subset ${ \dot { C } } _ { \ell } \subset [ { \bar { q } } ] ;$ ; a global policy $\theta = ( C _ { 1 } , \dots , C _ { B } )$ accepts

$$
A _ { \theta } = \bigcup _ { \ell = 1 } ^ { B } \bigcup _ { a \in C _ { \ell } } G _ { \ell , a } .
$$

Thus $P _ { 0 } ( A _ { \theta } ) = m / q = c$ for every member. In this one-dimensional realization each policy is a union of at most mB intervals, so the construction does not rely on nonmeasurable or arbitrarily fragmented acceptance sets, although the interval budget grows with B.

Let

$$
a _ { c } = \frac { c } { 1 - c } \gamma _ { B } , \qquad Q _ { \theta } ( L = 1 \mid S = x ) = \left\{ p - \gamma _ { B } , \quad x \in A _ { \theta } , \right.
$$

The parameter conditions keep both Bernoulli probabilities in $( 0 , 1 )$ . Each canonical component is feasible because it has coverage $c \geq \kappa$ and accepted risk $p - \gamma _ { B } = \alpha$ . Under the uniform prior on the $\left( { \begin{array} { l } { q } \\ { m } \end{array} } \right) ^ { B }$ policies, each score subinterval is selected with probability $m / q = c ,$ , hence its average perturbation $\mathrm { i s } - c \gamma _ { B } + ( 1 - c ) a _ { c } = 0$ . Therefore the mixture barycenter is exactly $P _ { 0 }$

All members have the same singleton problem by Proposition 6. The canonical components are also matched componentwise across profiles:

$$
\begin{array} { r l } & { \mathrm { T V } ( P _ { 0 } , Q _ { \theta } ) = 2 c \gamma _ { B } , } \\ & { \mathrm { K L } ( P _ { 0 } \| Q _ { \theta } ) = c \mathrm { k l } ( p \| p - \gamma _ { B } ) + ( 1 - c ) \mathrm { k l } ( p \| p + a _ { c } ) , } \\ & { \mathrm { K L } ( Q _ { \theta } \| P _ { 0 } ) = c \mathrm { k l } ( p - \gamma _ { B } \| p ) + ( 1 - c ) \mathrm { k l } ( p + a _ { c } \| p ) . } \end{array}
$$

These quantities depend on $\left( p , c , \gamma _ { B } \right)$ but not on the group-mass profile.

Product-mixture lower bound. For a local accepted subset $C \subset [ q ]$ define

$$
h _ { C } ( a ) = \left\{ { \displaystyle - 1 , \qquad a \in C , } \atop { c / ( 1 - c ) , \quad a \not \in C , }  \quad \quad Z ( C , C ^ { \prime } ) = \frac { 1 } { q } \sum _ { a = 1 } ^ { q } h _ { C } ( a ) h _ { C ^ { \prime } } ( a ) . \right.
$$

For independent uniform m-subsets $C , C ^ { \prime } , Z$ is centered and bounded by a constant depending only on c (equivalently on the fixed pair $m , q )$ . Across groups the $Z _ { \ell }$ are independent, and direct Bernoulli calculation gives

$$
\Xi ( \theta , \theta ^ { \prime } ) = \frac { \gamma _ { B } ^ { 2 } } { p ( 1 - p ) } \sum _ { \ell = 1 } ^ { B } w _ { \ell } Z _ { \ell } .
$$

Applying Hoeffding’s lemma inside Lemma 3 yields

$$
1 + \chi ^ { 2 } ( R _ { n } \| P _ { 0 } ^ { n } ) \leq \exp \left( C _ { p , c } n ^ { 2 } \gamma _ { B } ^ { 4 } \sum _ { \ell = 1 } ^ { B } w _ { \ell } ^ { 2 } \right) .
$$

Corollary 3 therefore implies $N _ { B , k } ^ { ( c , \kappa ) } = \Omega _ { p , c , \delta , \beta } ( \sqrt { k } / \gamma _ { B } ^ { 2 } )$ because $\textstyle \sum _ { \ell } ( w _ { \ell } ^ { ( k ) } ) ^ { 2 } = \Theta ( 1 / k )$ uniformly over the plateau spectrum.

Matching upper bound against the full unrestricted null. For $k < B$ , let $R = \cup _ { \ell > k } G _ { \ell }$ , so $P _ { 0 } ( R ) = \bar { \varepsilon } _ { B } = B ^ { - 2 }$ , retain the q subcells of every heavy group separately, and push $( S , L )$ to the $2 ( q k + 1 )$ resulting cell/label categories. Let Q be any law in the full existential null and let $A _ { \theta }$ be a witness. Write $D \stackrel { \smile } { = } A _ { \theta } \cap R ^ { c }$ and $\dot { d } _ { 0 } = P _ { 0 } ( D ) = c ( \dot { 1 } - \varepsilon _ { B } )$ . Put $a = c / 1 6$

If either $| Q ( R ) - \varepsilon _ { B } | \geq a \gamma _ { B }$ or $| Q ( D ) - d _ { 0 } | \geq a \gamma _ { B }$ , the pushed-forward laws already differ in TV by at least $a \gamma _ { B }$ . Otherwise $Q ( R ) \leq \varepsilon _ { B } + a \gamma _ { B }$ and $Q ( D ) \geq d _ { 0 } - a \gamma _ { B }$ . Feasibility gives

$$
\begin{array} { r l } & { Q ( D , L = 0 ) \geq Q ( A _ { \theta } , L = 0 ) - Q ( R ) } \\ & { \qquad \geq ( 1 - p + \gamma _ { B } ) Q ( A _ { \theta } ) - Q ( R ) } \\ & { \qquad \geq ( 1 - p + \gamma _ { B } ) Q ( D ) - Q ( R ) . } \end{array}
$$

Since $P _ { 0 } ( D , L = 0 ) = ( 1 - p ) d _ { 0 }$

$$
\begin{array} { l } { { Q ( D , L = 0 ) - P _ { 0 } ( D , L = 0 ) \geq \gamma _ { B } d _ { 0 } - a \gamma _ { B } ( 2 - p + \gamma _ { B } ) - \varepsilon _ { B } } } \\ { { \mathrm { } \qquad \geq \displaystyle \frac { 1 1 c \gamma _ { B } } { 1 6 } , } } \end{array}
$$

where the last inequality uses $d _ { 0 } \geq 1 5 c / 1 6 , 2 - p + \gamma _ { B } < 3 $ , and $\varepsilon _ { B } \leq c \gamma _ { B } / 1 6$ . Thus every full-null law is separated from $P _ { 0 }$ by at least $c \gamma _ { B } / 1 6$ after a pushforward with $O _ { c } ( k )$ categories.

Known-distribution identity testing gives $O _ { p , c } ( \sqrt { k } / \gamma _ { B } ^ { 2 } )$ samples. For $k = B$ the residual cell is absent and the same argument uses the $2 q B$ subcell/label categories. Combining the two bounds proves the proposition.

Taking, for example, $c = 9 / 1 0$ and any fixed $\kappa < 9 / 1 0$ gives a high-coverage construction with nonzero contract slack. The theorem does not settle the different regime in which the number of allowed intervals K is fixed while $B \to \infty ;$ that structured question is left open.

## D.7 INFORMATIVE NONCONSTANT-REFERENCE SPECTRUM

The score-independent reference in Equation (5) is a convenient special case, not a requirement for the separation. The next construction makes the score genuinely informative about risk and makes selective acceptance strictly useful under the reference law.

Proposition 9 (Matched spectrum under informative nonconstant risk). Fix constants $a , b , \rho > 0$ with $a + b + \rho = 1$ , risks $p _ { C } , p _ { V } , p _ { H } \in ( 0 , 1 )$ , and $h > 0$ such that $p _ { V } \pm h \in ( 0 , 1 )$ . Put

$$
c = a + \rho / 2 , \qquad r _ { A } = \frac { a p _ { C } + ( \rho / 2 ) p _ { V } } { c } , \qquad \bar { p } = a p _ { C } + \rho p _ { V } + b p _ { H } ,
$$

and assume $r _ { A } < \bar { p } .$ Fix $0 < \kappa \leq$ c and $0 < \delta < \beta < 1$ . For any sequence $\gamma _ { B } \downarrow 0$ such that

$$
\tau _ { B } : = \frac { 2 c \gamma _ { B } } { \rho } < \operatorname* { m i n } \{ p _ { V } - h , 1 - p _ { V } - h \} , \qquad \rho B ^ { - 2 } \leq \frac { c \gamma _ { B } } { 8 } ,
$$

there are k-plateaufamilies with $J _ { B } = 2 ^ { B }$ policies, a common reference law with nonconstant $\eta ( s ) =$ $P _ { 0 } ( L = 1 \ \bar { | } \ S = s )$ , common contract $\left( \kappa , r _ { A } - \gamma _ { B } \right)$ , exact one-observation hiding, and identical unrestricted singleton testing problemsfor every member, while uniformly in $k \in \{ \bar { 1 } , \ldots , B \}$

$$
N _ { B , k } ^ { \mathrm { i n f } } = \Theta _ { a , b , \rho , p _ { C } , p _ { V } , p _ { H } , h , \kappa , \delta , \beta } \left( \frac { \sqrt { k } } { \gamma _ { B } ^ { 2 } } \right) .
$$

Every policy has reference coverage c and reference conditional risk $r _ { A } < \bar { p } ,$ so abstention strictly improves risk under $P _ { 0 }$

Reference and policy family. The reference law must remain literally fixed as the plateau profile changes. Let the variable region carry product coordinates $( X , Z ) \ \overset { \cdot } { \in } \ [ 0 , 1 ] \times \{ - 1 , + 1 \}$ , with conditional reference law $X \ \stackrel { \sim } { \sim } \ \mathrm { U n i f } [ \dot { 0 } , \dot { 1 } ]$ , Z uniform and independent of X. Give the disjoint regions C, V, H masses $a , \rho , b ;$ set $\eta = p _ { C }$ on $C , \eta = p _ { H }$ on H, and

$$
\eta ( X , Z ) = p _ { V } + h Z \qquad { \mathrm { o n ~ } } V .
$$

This defines one common $P _ { 0 }$ for every profile. For a mass vector $w ,$ , partition only the X coordinate into groups $G _ { 1 } , \ldots , G _ { B }$ of masses $w _ { 1 } , \ldots , w _ { B }$ and split each $G _ { \ell }$ into equal X-halves $G _ { \ell } ^ { + } , G _ { \ell } ^ { - }$ Each half is crossed with both values of $Z ,$ so it always contains equal masses at risks $p _ { V } - h$ and $p _ { V } + h$ , independently of its width or of the profile. $( \dot { \mathbf { A } }$ standard nonatomic-space isomorphism can encode these product coordinates in a scalar score if desired; interval structure is not claimed for this informative extension.)

For $\theta \in \{ - 1 , + 1 \} ^ { B }$ , let $A _ { \theta }$ accept all of $C$ and $( G _ { \ell } ^ { \theta _ { \ell } } \times \{ - 1 , + 1 \} )$ from every variable group. Then $P _ { 0 } ( A _ { \theta } ) ~ = ~ c$ and every member has reference accepted risk $r _ { A } ;$ its rejected risk is also profile-independent. In particular the four-cell law of $( \mathbf { 1 } \{ S { \overset { . } { \in } } A _ { \theta } \} , L )$ is the same for every $\theta$ and every plateau profile, so Proposition 6 gives exact singleton equivalence without any score–loss independence assumption.

Canonical feasible components and exact hiding. Let $u _ { \theta } ( X , Z ) = + 1$ on the selected X-half of a variable group (for either $Z )$ and −1 on the unselected half. Define $Q _ { \theta }$ by preserving the score marginal and setting

$$
\begin{array} { r } { Q _ { \theta } ( L = 1 \mid S = x ) = \left\{ \begin{array} { l l } { \eta ( x ) , } & { x \in C \cup H , } \\ { \eta ( x ) - \tau _ { B } u _ { \theta } ( x ) , } & { x \in V . } \end{array} \right. } \end{array}
$$

The parameter condition keeps all Bernoulli probabilities in $( 0 , 1 )$ . The accepted error mass decreases by $( \rho / 2 ) \tau _ { B } = c \gamma _ { B }$ , hence $\bar { Q } _ { \theta } ( L = 1 \mid S \in \bar { A } _ { \theta } ) = r _ { A } - \dot { \gamma _ { B } }$ and the component is feasible. Under the uniform prior on signs, each variable point is selected and rejected with equal probability, so the perturbation cancels pointwise and $\mathbb { E } _ { \theta } Q _ { \theta } = P _ { 0 }$ exactly. Canonical TV and both directed KL divergences are independent of the plateau profile because every group contains the same replicated two-level risk pattern.

Overlap lower bound. Define

$$
H _ { V } = { \frac { 1 } { 2 } } \left\{ \frac { 1 } { ( p _ { V } - h ) ( 1 - p _ { V } + h ) } + \frac { 1 } { ( p _ { V } + h ) ( 1 - p _ { V } - h ) } \right\} .
$$

By the finite-cell Fisher representation, for independent signs $\theta , \theta ^ { \prime }$

$$
\Xi ( \theta , \theta ^ { \prime } ) = \Lambda _ { B } \sum _ { \ell = 1 } ^ { B } w _ { \ell } \theta _ { \ell } \theta _ { \ell } ^ { \prime } , \qquad \Lambda _ { B } = \rho \tau _ { B } ^ { 2 } H _ { V } = \frac { 4 c ^ { 2 } H _ { V } } { \rho } \gamma _ { B } ^ { 2 } .
$$

The same Rademacher MGF calculation as Proposition 3 therefore gives $N _ { B , k } ^ { \mathrm { i n f } } ~ = ~ \Omega ( \sqrt { k } / \gamma _ { B } ^ { 2 } )$ uniformly over the plateau spectrum.

Matching upper bound against the full unrestricted null. For $k < B ,$ , collapse the residual variable groups $R = \cup _ { \ell > k } G _ { \ell }$ , whose reference mass is $P _ { 0 } ( R ) = \rho \varepsilon _ { B }$ , while retaining $C , H$ and each heavy half separately; retain the label in every cell. This gives 4 $. k { + 6 }$ categories. Let $Q$ be any full-null law with witness $A _ { \theta } .$ , put $A _ { H } = A _ { \theta } \setminus R$ , and define on the pushforward $g = { \bf 1 } \{ S \in A _ { H } \} ( L - \alpha )$ with $\alpha = r _ { A } - \gamma _ { B }$ . Since $P _ { 0 } ( A _ { \theta } ) \{ r _ { P _ { 0 } } ( A _ { \theta } ) - \alpha \} = c \gamma _ { B }$

$$
\begin{array} { r } { \mathbb { E } _ { P _ { 0 } } g \ge c \gamma _ { B } - P _ { 0 } ( R ) . } \end{array}
$$

Feasibility gives $\mathbb { E } _ { Q } [ 1 \{ S \in A _ { \theta } \} ( L - \alpha ) ] \le 0$ , hence $\mathbb { E } _ { Q } g \le Q ( R )$ . If $| Q ( R ) - P _ { 0 } ( R ) | \geq c \gamma _ { B } / 4$ the residual-mass coordinate already separates the pushforwards by $c \gamma _ { B } / 4$ . Otherwise

$$
\mathbb { E } _ { P _ { 0 } } g - \mathbb { E } _ { Q } g \ge \frac { 3 c \gamma _ { B } } { 4 } - 2 P _ { 0 } ( R ) \ge \frac { c \gamma _ { B } } { 2 } ,
$$

using $\rho \varepsilon _ { B } \leq c \gamma _ { B } / 8$ . Because $g$ has oscillation one, the pushed-forward TV distance is therefore at least $c \gamma _ { B } / 4$ in all cases. Known-distribution identity testing on $O ( k )$ categories gives $O ( \sqrt { k } / \gamma _ { B } ^ { 2 } )$ samples; for $k = B$ there is no residual category. This matches the lower bound. All constants are uniform over k and over every admissible sequence $\gamma _ { B }$ satisfying the displayed conditions.

A concrete choice is $( a , \rho , b ) = ( . 3 0 , . 5 0 , . 2 0 )$ and $( p _ { C } , p _ { V } - h , p _ { V } + h , p _ { H } ) = ( . 0 8 , . 1 6 , . 3 2 , . 6 0 )$ for which $c = . 5 5$ , the accepted reference risk is about .153, the population risk is .264, and the score carries substantial risk information while the matched spectrum remains unchanged in order.

## E PRODUCT-OVERLAP IDENTITY AND FISHER GEOMETRY

Lemma 3 (Product-mixture identity; Ingster, 1987). For dominated feasible components write $K ( Q , Q ^ { \prime } ) \stackrel { } { = } \int ( d Q / d P _ { 0 } ) ( d Q ^ { \prime } / d P _ { 0 } ) d P _ { 0 }$ and $\Xi = K - 1$ . For a prior Π with $\bar { \mathbb { E } _ { \Pi } Q } = P _ { 0 }$ and $R _ { n } = \mathbb { E } _ { \Pi } Q ^ { n } { } _ { ; }$

$$
1 + \chi ^ { 2 } ( R _ { n } \| P _ { 0 } ^ { n } ) = \mathbb { E } _ { Q , Q ^ { \prime } \sim \Pi } [ 1 + \Xi ( Q , Q ^ { \prime } ) ] ^ { n } \quad ( i n t h e e x t e n d e d s e n s e ) .\tag{19}
$$

## E.1 PROOF OF LEMMA 3

Write $h _ { Q } = d Q / d P _ { 0 }$ . Since $Q \ll P _ { 0 }$ almost surely, the product-mixture likelihood ratio is

$$
{ \frac { d R _ { n } } { d P _ { 0 } ^ { n } } } ( x _ { 1 : n } ) = \mathbb { E } _ { Q } \prod _ { i = 1 } ^ { n } h _ { Q } ( x _ { i } ) .
$$

All terms are nonnegative, so Tonelli applies without any per-component $L ^ { 2 }$ assumption. With independent $Q , Q ^ { \prime } \sim \Pi$

$$
\begin{array} { l } { 1 + { \displaystyle \chi ^ { 2 } ( R _ { n } \| P _ { 0 } ^ { n } ) } = \mathbb { E } _ { Q , Q ^ { \prime } } \prod _ { i = 1 } ^ { n } \int h _ { Q } ( x ) h _ { Q ^ { \prime } } ( x ) d P _ { 0 } ( x ) } \\ { ~ = \mathbb { E } _ { Q , Q ^ { \prime } } K ( Q , Q ^ { \prime } ) ^ { n } . } \end{array}
$$

The barycenter condition implies $\mathbb { E } _ { Q } h _ { Q } ( x ) ~ = ~ 1$ for P -almost every x, hence by Tonelli $\begin{array} { r } { \mathbb { E } _ { Q , Q ^ { \prime } } K ( Q , Q ^ { \prime } ) = \int ( \mathbb { E } _ { Q } h _ { Q } ) ^ { 2 } d P _ { 0 } = 1 } \end{array}$ . Therefore $K < \infty$ for Π ⊗ Π-almost every pair, $R _ { 1 } = P _ { 0 }$ and with $\Xi = K - \check { 1 }$

$$
\chi ^ { 2 } ( R _ { 2 } \| P _ { 0 } ^ { 2 } ) = \mathbb { E } K ^ { 2 } - 1 = \mathbb { E } ( K - 1 ) ^ { 2 } = \mathbb { E } \Xi ^ { 2 }
$$

in the extended sense. This is the standard product-mixture second-moment calculation; the paperspecific content is the selective-contract overlap kernel and its role in the matched separation.

Corollary 1 (Approximate hiding under TV mismatch). Let Π be supported on $\mathcal { H } ( A )$ , write $\bar { Q } =$ $\mathbb { E } _ { Q \sim \Pi ^ { Q } }$ , and suppose $\mathrm { T V } ( P _ { 0 } , \bar { Q } ) \leq \varepsilon .$ . Assume $Q \ll \bar { Q }$ almost surely and define $\bar { K } ( Q , Q ^ { \prime } ) =$ $\textstyle \int ( d Q / d \bar { Q } ) ( d Q ^ { \prime } / d \bar { Q } ) d \bar { Q }$ and $\bar { \Xi } = \bar { K } - 1$ . Then every level-δ full-null-valid test satisfies

$$
\mathbb { E } _ { P _ { 0 } ^ { n } } \phi \leq \delta + 1 - ( 1 - \varepsilon ) ^ { n } + \frac { 1 } { 2 } \sqrt { \mathbb { E } [ 1 + \bar { \Xi } ( Q , Q ^ { \prime } ) ] ^ { n } - 1 } \leq \delta + n \varepsilon + \frac { 1 } { 2 } \sqrt { \mathbb { E } [ 1 + \bar { \Xi } ( Q , Q ^ { \prime } ) ] ^ { n } - 1 } .\tag{20}
$$

Corollary 2 (Approximate hiding under quadratic mismatch). Under the same mixture assumptions, additionally suppose $P _ { 0 } \ll \bar { Q }$ and $\chi ^ { 2 } ( P _ { 0 } \mathbf { \bar { \| } } \bar { Q } ) \leq \rho .$ Then every level-δ full-null-valid test satisfies

$$
\mathbb { E } _ { P _ { 0 } ^ { n } } \phi \leq \delta + \frac 1 2 \sqrt { ( 1 + \rho ) ^ { n } - 1 } + \frac 1 2 \sqrt { \mathbb { E } [ 1 + \bar { \Xi } ( Q , Q ^ { \prime } ) ] ^ { n } - 1 } .\tag{21}
$$

## E.2 PROOF OF COROLLARY 1

Let $\bar { Q } = \mathbb { E } _ { Q \sim \Pi } Q$ and $R _ { n } = \operatorname { \mathbb { E } } _ { Q \sim \Pi } Q ^ { n }$ . For any level-δ full-null-valid test,

$$
\begin{array} { r } { \mathbb { E } _ { P _ { 0 } ^ { n } } \phi \leq \mathbb { E } _ { R _ { n } } \phi + \mathrm { T V } ( P _ { 0 } ^ { n } , R _ { n } ) \leq \delta + \mathrm { T V } ( P _ { 0 } ^ { n } , R _ { n } ) . } \end{array}
$$

By the triangle inequality,

$$
\mathrm { T V } ( P _ { 0 } ^ { n } , R _ { n } ) \leq \mathrm { T V } ( P _ { 0 } ^ { n } , \bar { Q } ^ { n } ) + \mathrm { T V } ( \bar { Q } ^ { n } , R _ { n } ) .
$$

A maximal coupling of one draw from $P _ { 0 }$ and $\bar { Q }$ disagrees with probability $\mathrm { T V } ( P _ { 0 } , \bar { Q } ) \leq \varepsilon ;$ coupling the n coordinates independently gives

$$
\mathrm { T V } ( P _ { 0 } ^ { n } , \bar { Q } ^ { n } ) \leq 1 - ( 1 - \varepsilon ) ^ { n } \leq n \varepsilon .
$$

Finally, $\bar { Q }$ is the exact barycenter of Π, so Lemma 3 applied with reference $\bar { Q }$ gives

$$
\mathrm { T V } ( \bar { Q } ^ { n } , R _ { n } ) \leq \frac { 1 } { 2 } \sqrt { \chi ^ { 2 } ( R _ { n } \| \bar { Q } ^ { n } ) } = \frac { 1 } { 2 } \sqrt { \mathbb { E } [ 1 + \bar { \Xi } ( Q , Q ^ { \prime } ) ] ^ { n } - 1 } .
$$

Combining the three displays proves (20). The TV statement is intentionally assumption-light; its linear product penalty is useful only when nε is small.

## E.3 PROOF OF COROLLARY 2

With the same notation and triangle inequality,

$$
\mathrm { T V } ( P _ { 0 } ^ { n } , R _ { n } ) \leq \mathrm { T V } ( P _ { 0 } ^ { n } , \bar { Q } ^ { n } ) + \mathrm { T V } ( \bar { Q } ^ { n } , R _ { n } ) .
$$

The second term is controlled exactly as above by Lemma 3 with reference $\bar { Q }$ . For the first term, product likelihood ratios give

$$
1 + \chi ^ { 2 } ( P _ { 0 } ^ { n } \| \bar { Q } ^ { n } ) = \{ 1 + \chi ^ { 2 } ( P _ { 0 } \| \bar { Q } ) \} ^ { n } \leq ( 1 + \rho ) ^ { n } .
$$

Therefore $\mathrm { T V } ( P _ { 0 } ^ { n } , \bar { Q } ^ { n } ) \leq \textstyle { \frac { 1 } { 2 } } \sqrt { ( 1 + \rho ) ^ { n } - 1 }$ . Combining the two terms and test validity proves (21).   
When $n \rho = o ( 1 ) , ( 1 + \rho ) ^ { n } \dot { ~ - } 1 = O ( n \rho )$ , giving an $O ( { \sqrt { n \rho } } )$ mismatch contribution.

When exact one-observation hiding is possible, minimizing the witness bound in Corollary 3 over all hiding priors yields the family-level obstruction

$$
\Omega _ { n } ( P _ { 0 } , { \mathcal H } ) = \operatorname* { i n f } _ { \substack { \Pi : \Pi ( { \mathcal H } ) = 1 , \mathbb E _ { Q \sim \Pi } [ Q ] = P _ { 0 } } } { \mathrm { T V } } ( P _ { 0 } ^ { n } , \mathbb E _ { Q \sim \Pi } Q ^ { n } ) ,\tag{22}
$$

so every level-δ full-null-valid audit has power at most $\delta + \Omega _ { n } ( P _ { 0 } , \mathcal { H } )$ . Lemma 3 evaluates any chosen hiding witness, while the infimum above removes dependence on a particular decomposition. We do not assume this infimum is attained without additional compactness; in the finite constructions used below, the displayed witness prior is explicit and order-tight.

## E.4 FINITE-CELL FISHER REPRESENTATION

Assume component $Q _ { k }$ preserves cell masses $\mu _ { j }$ and has Bernoulli error rate $\theta _ { k , j }$ in cell $j ,$ , while $P _ { 0 }$ has rate $\eta _ { j } \in ( 0 , 1 )$ . Directly evaluating the two label states shows

$$
\Xi ( Q _ { k } , Q _ { \ell } ) = \sum _ { j } \mu _ { j } \frac { ( \theta _ { k , j } - \eta _ { j } ) ( \theta _ { \ell , j } - \eta _ { j } ) } { \eta _ { j } ( 1 - \eta _ { j } ) } = \sum _ { j } \mu _ { j } g _ { k } ( j ) g _ { \ell } ( j ) .
$$

Thus the overlap matrix is exactly a Gram matrix in the local Bernoulli Fisher metric.

## F AUXILIARY TESTING BOUNDS

For a prior Π over feasible laws with $\mathbb { E } _ { \Pi } Q = P _ { 0 }$ (a hiding prior) and $R _ { n } = \mathbb { E } _ { \Pi } Q ^ { n }$

Corollary 3 (Witness power ceiling). For every full-null-valid level-δ test and any hiding prior,

$$
\begin{array} { r } { \mathbb { E } _ { P _ { 0 } ^ { n } } \phi \le \delta + \mathrm { T V } ( P _ { 0 } ^ { n } , R _ { n } ) \le \delta + \frac 1 2 \sqrt { \mathbb { E } [ 1 + \Xi ( Q , Q ^ { \prime } ) ] ^ { n } - 1 } . } \end{array}\tag{23}
$$

Proof. Every component of the prior is feasible, so a level-δ test satisfies $\mathbb { E } _ { R _ { n } } \phi \le \delta$ . Hence $\begin{array} { r } { \mathbb { E } _ { P _ { 0 } ^ { n } } \phi \le \delta + \mathrm { T V } ( P _ { 0 } ^ { n } , R _ { n } ) ; \mathrm { T V } \le \frac { 1 } { 2 } \sqrt { \chi ^ { 2 } } } \end{array}$ and Lemma 3 give the second inequality. □

## F.1 A COLLISION BOUND FOR SUPPORTED NULLS

Let $p _ { i } > 0$ on $d \geq 2$ reference-positive cells and define $\mathcal { H } _ { + }$ as the feasible categorical laws that put zero mass on all reference-zero cells. Assume $\mathcal { H } _ { + }$ is nonempty and

$$
\Delta = \operatorname* { m i n } _ { q \in \mathcal { H } _ { + } } \sum _ { i } ( q _ { i } - p _ { i } ) ^ { 2 } / p _ { i } > 0 , \quad p _ { \operatorname* { m i n } } = \operatorname* { m i n } _ { i } p _ { i } , \quad u _ { 0 } = 1 + \sqrt { \Delta / p _ { \operatorname* { m i n } } } .
$$

The infimum is over members, not their one-observation convex hull.

Theorem 6 (Supported collision bracket). Set $\epsilon = \operatorname* { m i n } \{ \delta , 1 - \beta \}$ and $c _ { \delta , \beta } = \mathrm { m i n } \{ 1 / 2 , { \scriptstyle \frac { 1 } { 4 } } \log [ 1 / ( 1 -$ $( \beta - \delta ) ^ { 2 } ) ] \}$ . Then

$$
\frac { c _ { \delta , \beta } } { \Delta } \leq N ( p , \mathcal { H } _ { + } ; \delta , \beta ) \leq \operatorname* { m a x } \left\{ 2 , \left\lceil \frac { 3 2 } { \epsilon } \frac { \operatorname* { m a x } \{ \sqrt { d - 1 } , u _ { 0 } \} } { \Delta } \right\rceil \right\} .
$$

The lower bound also holds for the full null. The upper bound is achieved by rejecting feasibility when $U _ { m } < \Delta / 2$ , where

$$
U _ { m } = { \binom { m } { 2 } } ^ { - 1 } \sum _ { a < b } \left( \frac { \mathbf { 1 } \{ X _ { a } = X _ { b } \} } { p _ { X _ { a } } } - 1 \right) .
$$

Proof. Let $f ( i , j ) = \mathbf { 1 } \{ i = j \} / p _ { i } - 1$ . Under $p , \sum _ { j } p _ { j } f ( i , j ) = 0$ and $\mathbb { E } _ { p \otimes p } f ^ { 2 } = d - 1$ . Thus $\mathbb { E } _ { p } U _ { m } = 0$ and $\operatorname { V a r } _ { p } ( U _ { m } ) = ( d - 1 ) / { \binom { m } { 2 } }$ . Under a component q, let $D = \chi ^ { 2 } ( q \| p ) \geq \Delta , r _ { i } = q _ { i } / p _ { i }$ and $u = \operatorname* { m a x } _ { i } r _ { i }$ . Then $\begin{array} { r } { \mathbb { E } _ { q } U _ { m } = D , } \end{array}$

$$
\zeta _ { 1 } = \mathrm { V a r } _ { q } ( r _ { i } - 1 ) \le u D , \qquad \zeta _ { 2 } \le \mathbb { E } _ { q \otimes q } f ^ { 2 } \le 4 ( d - 1 ) + 4 D / p _ { \operatorname* { m i n } } + 1 .
$$

For the last inequality split the sum $\textstyle \sum _ { i } r _ { i } ^ { 2 } ( 1 - p _ { i } ) ^ { 2 }$ at $r _ { i } = 2 ,$ use $r _ { i } ^ { 2 } \leq 4 ( r _ { i } - 1 ) ^ { 2 }$ on the second part, and bound the off-diagonal contribution by one. The U-statistic decomposition gives

$$
\operatorname { V a r } _ { q } ( U _ { m } ) \leq 4 \zeta _ { 1 } / m + 2 \zeta _ { 2 } / [ m ( m - 1 ) ] .
$$

Chebyshev therefore bounds the reference error by $1 6 ( d - 1 ) / ( m ^ { 2 } \Delta ^ { 2 } )$ and the component error by $1 6 u / \bar { ( } m D ) + 1 6 \zeta _ { 2 } / ( m ^ { 2 } D ^ { 2 } )$ for $m \geq 2 .$ . Since $D ^ { ' } \geq p _ { \operatorname* { m i n } } ( u ^ { - } \dot { 1 } ) ^ { 2 }$ , we have $u / D \le \bar { u } _ { 0 } / \Delta$ . Also

$$
\sqrt { \zeta _ { 2 } } / D \leq [ 2 \sqrt { d - 1 } + 2 ( u _ { 0 } - 1 ) + 1 ] / \Delta \leq 5 \operatorname* { m a x } \{ \sqrt { d - 1 } , u _ { 0 } \} / \Delta .
$$

The stated constant controls both errors by $\epsilon .$ Validity is componentwise and therefore survives mixing. For the lower bound use a minimizing q and $H ^ { 2 } ( p , q ) \leq \chi ^ { 2 } ( q \| p ) = \Delta$ . The two-point inequality gives $N \ge ( 1 - H ^ { 2 } ) / ( 2 H ^ { 2 } ) \log [ 1 / ( \bar { 1 - ( \beta - \delta ) } ^ { 2 } ) ] . \bar { 1 \mathrm { f } } \bar { \Delta } \le \bar { 1 } / \hat { 2 }$ this is at least the claimed lower bound; if $\Delta > 1 / 2 ,$ , use $\dot { N } \geq 1$ and $c _ { \delta , \beta } \leq 1 / 2 .$ □

Reference-zero cells. The collision statistic is evaluated only on reference-positive cells. For a full-null extension let Z be the reference-zero cells, $\rho ( q ) = q ( Z )$ , and

$$
\Delta ( \rho _ { 0 } ) = \operatorname* { i n f } _ { \substack { q \in \mathcal { H } : \rho ( q ) \le \rho _ { 0 } } } \chi ^ { 2 } ( q ( \cdot \mid Z ^ { c } ) \| p ) .
$$

Combine the collision rejection with the requirement that no sample falls in $Z .$ . If $\rho ( q ) > \rho _ { 0 }$ , rejection probability is at most $( \bar { 1 } - \rho ( q ) ) ^ { m } \leq e ^ { - \bar { m } \rho _ { 0 } }$ . Otherwise, conditional on avoiding $Z ,$ , the samples are i.i.d. from $q ( \cdot \mid Z ^ { \dot { c } } )$ and the collision proof applies with $\Delta ( \rho _ { 0 } )$ . Thus a valid full-null upper bound uses the maximum of the collision sample bound at this radius and $\log ( 1 / \epsilon ) / \rho _ { 0 }$ , whenever $\Delta ( \rho _ { 0 } ) > 0$ . We do not substitute the supported $\bar { \Delta }$ for this radius in our empirical claims.

## G REFERENCE-FREE INTERSECTION–UNION CERTIFICATE

For a fixed policy A, let $c ( A ) = P ( S \in A )$ and $r ( A ) = P ( L = 1 \mid S \in A )$ Then $T _ { A } \sim$ Bin $\operatorname { \mathbb { 1 } } ( n , c ( A ) )$ ) and, conditional on $T _ { A } = t , E _ { A } \sim \mathrm { B i n } ( t , r ( A ) )$ . Let $U _ { A }$ be a one-sided $( 1 - \zeta _ { c } )$ exact upper bound on $c ( A )$ and $L _ { A }$ a one-sided $\left( 1 - \zeta _ { r } \right)$ exact lower bound on $r ( A )$ , with $\zeta _ { c } + \zeta _ { r } \leq \delta$

## G.1 PROOF OF PROPOSITION 1

If the existential null is true, choose a fixed population witness $A ^ { * }$ satisfying $c ( A ^ { * } ) ~ \geq ~ \kappa$ and $r ( A ^ { * } ) \leq \alpha$ . This witness depends on the population, not on the sample. If the procedure rejects, then either $U _ { A ^ { * } } < \kappa \leq c ( A ^ { * } )$ , or $U _ { A ^ { * } } \geq \kappa ,$ in which case $A ^ { * }$ participates in the minimization and rejection forces $L _ { A ^ { * } } > \alpha \ge r ( A ^ { * } )$ The two one-sided bounds fail with probabilities at most $\zeta _ { c }$ and $\zeta _ { r }$ , respectively. A union bound over these two events gives at most $\zeta _ { c } + \zeta _ { r } \leq \delta$ . No event is required to hold simultaneously for all searched policies. The equal split used in our experiments is therefore a convenience rather than a requirement; any predeclared allocation is valid. More generally, a fixed-policy joint confidence region for $( c ( A ) , r ( { \dot { A } } ) )$ can replace the rectangular pair whenever its simultaneous failure probability is at most δ and the corresponding coverage/risk projections are used in the same screening logic.

Proposition 10 (Anytime-valid existential certificate). For each fixed A, replace $( U _ { A } , L _ { A } )$ by onesided confidence sequences $( U _ { A , t } , L _ { A , t } )$ whose anytime failure probabilities are at most $\zeta _ { c } f o r$ coverage and $\zeta _ { r } f o r$ risk. Reject at the first time t for which min $_ { A : U _ { A , t } \geq \kappa } L _ { A , t } > \alpha$ . Under the existential null, the probability ofever rejecting is at most $\zeta _ { c } + \zeta _ { r } \leq \delta ,$ , again without a multiplicity correction over A.

## G.2 PROOF OF PROPOSITION 10

If the existential null is true, fix a population witness $A ^ { * }$ . Let $\mathcal { E } _ { c }$ be the event that the coverage upper confidence sequence ever falls below the true $c ( A ^ { * } )$ , and let ${ \mathcal { E } } _ { r }$ be the event that the risk lower confidence sequence ever rises above the true $r ( \boldsymbol { A } ^ { * } )$ . By assumption, their probabilities are at most $\zeta _ { c }$ and $\zeta _ { r } ,$ , respectively. If the anytime procedure ever rejects while neither event occurs, then $U _ { A ^ { * } , t } \geq c ( A ^ { * } ) \geq \kappa ,$ , so $\boldsymbol { \dot { A } } ^ { * }$ is in the screened family, while $L _ { A ^ { * } , t } \leq r ( A ^ { * } ) \leq \alpha$ , contradicting rejection. Hence ever-rejection is contained in $\mathcal { E } _ { c } \cup \mathcal { E } _ { r }$ and has probability at most $\zeta _ { c } + \zeta _ { r } \leq \delta .$ . The argument requires simultaneous-in-time validity only for the one fixed population witness, not for all searched policies.

Why a one-policy split-sample comparator is not a valid existential-null test. Selecting one A on one split and rejecting that policy on an independent split proves only that the selected member is infeasible. The null is a union over all $A \in { \mathcal { A } }$ , so another member may still be the population witness. A valid split-sample existential test would still need to reject every potential witness (or otherwise control the union-null logic), restoring the structural problem addressed by the IUT. A “select one, then test $\mathrm { o n e ^ { \gamma } }$ comparator therefore tests a different null hypothesis, and we do not report it.

## G.3 OPERATIONAL POWER ON THEOREM-ADMISSIBLE MATCHED FAMILIES

An additional finite-size control uses $B = 1 6 , p = . 2 5 , \gamma = . 0 8 , \alpha = . 1 7 , \kappa = . 5 ,$ , and $k = 1 , 4 , 1 6$ The residual mass is $B ^ { - 2 } = 1 / 2 5 6$ , which is at most $\gamma / 1 6 = . 0 0 5$ , as required by Theorem 2. The $k = 1 6$ profile is uniform. For every $( k , n )$ , generate independent multinomial counts over the 2B group halves and then binomial errors with common probability $p .$ The boundary-null control instead uses error probability $p - \gamma$ on one selected half of each group and $p + \gamma$ on the other; the selected policy is a population witness.

The sample grid is $n \in \{ 3 0 0 , 5 0 0 , 7 5 0 , 1 0 0 0 , 1 4 0 0 , 2 0 0 0 \}$ for alternatives and $n \in \{ 5 0 0 , 1 4 0 0 \}$ for boundary checks, with 2,000 replicates per point. The artifact records all deterministic seeds. Error bars are pointwise Wilson 95% intervals for Monte Carlo rejection probabilities. The reported $n _ { 8 0 }$ values linearly interpolate neighboring power estimates; no confidence interval or minimax interpretation is attached to a crossing. The endpoint ratio is $1 3 1 8 . 1 1 / 5 7 2 . 2 9 = 2 . 3 0 3$ . All grid values, including null controls, are retained below.

Table 6: Additional sign-family simulation: observed rejection percentages, 2,000 replicates per cell. Null rows refer to a feasible boundary component.
<table><tr><td>Source</td><td>n</td><td> $k = 1$ </td><td> $k = 4$ </td><td> $k = 1 6$ </td></tr><tr><td>Alternative</td><td>300</td><td>42.75</td><td>16.95</td><td>0.75</td></tr><tr><td>Alternative</td><td>500</td><td>75.20</td><td>47.60</td><td>7.00</td></tr><tr><td>Alternative</td><td>750</td><td>91.80</td><td>77.00</td><td>26.85</td></tr><tr><td>Alternative</td><td>1000</td><td>97.45</td><td>92.95</td><td>54.75</td></tr><tr><td>Alternative</td><td>1400</td><td>99.90</td><td>98.90</td><td>86.50</td></tr><tr><td>Alternative</td><td>2000</td><td>99.95</td><td>99.95</td><td>98.35</td></tr><tr><td>Boundary null</td><td>500</td><td>3.50</td><td>2.15</td><td>0.25</td></tr><tr><td>Boundary null</td><td>1400</td><td>3.95</td><td>2.80</td><td>1.30</td></tr></table>

Sign-family dynamic program. Write $n _ { b h } , e _ { b h }$ for counts and errors in half $h \in \{ 0 , 1 \}$ of group b. Initialize $D _ { 0 } ( 0 ) = 0$ and $D _ { 0 } ( t ) = + \infty$ otherwise. The exact recurrence is

$$
D _ { b } ( t ) = \operatorname* { m i n } _ { h \in \{ 0 , 1 \} } \{ D _ { b - 1 } ( t - n _ { b h } ) + e _ { b h } \} ,
$$

with invalid indices assigned infinity. After all B groups, $D _ { B } ( t )$ is the least errors among policies accepting exactly t observations. The lower binomial bound is monotone in errors, so applying the coverage screen and risk bound to $D _ { B }$ computes the IUT over all $2 ^ { B }$ policies in $O ( B \bar { n } )$ time and $O ( n )$ memory. Exhaustive tests compare every state and final decision with direct policy enumeration on small instances. This recurrence changes computation only; Proposition 1 supplies validity.

Coverage-floor and audit-size sensitivity. On the held-out ImageNetV2 audit with $K = 1 6$ and cells frozen on disjoint calibration data, we also evaluate $\kappa \in \{ . 5 0 , . 6 0 , . 7 5 , . 9 0 \}$ and audit sizes $n \in \{ 5 0 0 , 1 0 0 0 , 2 5 0 0 , 5 0 0 0 \}$ . For the full $n = 5 0 0 0$ audit the exact 95% IUT lower frontiers are 4.06%, 6.28%, 10.89%, and 17.71%, respectively. In 40 deterministic held-out subsamples, the corresponding median frontiers at $n = 5 0 0$ are 0.97%, 2.62%, 6.70%, and 13.55%; at $n = 1 0 0 0$ they are 2.41%, 4.20%, 8.79%, and 15.64%. This is a finite-benchmark sensitivity analysis, not a population learning curve, but it makes explicit how the certificate weakens at lower coverage floors and smaller audit sizes.

## G.4 WEIGHTED INTERVAL DYNAMIC PROGRAM

Let cell m contain $n _ { m }$ audit observations and $e _ { m }$ errors. A policy using at most K contiguous intervals selects a subset of cells that is a union of at most K contiguous intervals. Define

$$
e _ { K } ( t ) = \operatorname* { m i n } \left\{ \sum _ { m \in { \cal A } } e _ { m } : \sum _ { m \in { \cal A } } n _ { m } = t , { \cal A } \mathrm { u s e s ~ a t ~ m o s t } { \cal K } \mathrm { ~ i n t e r v a l s } \right\} .
$$

A standard dynamic program over cell index, selected count, number of opened intervals, and whether the current cell is inside an interval computes all $e _ { K } ( t )$ exactly. If zero-count cells are possible, an additional one-bit state distinguishes the structurally empty mask from a nonempty rule that happens to accept no audit observation; this leaves the $\bar { O ( M K n ) }$ state and $O ( K n )$ memory bounds unchanged. Because the one-sided binomial lower bound is monotone in the observed error count, the IUT objective reduces to the one-dimensional minimization reported in Section 4. In the reported experiments the coverage floors are large enough that a $t = 0$ rule cannot pass the coverage screen, but the implementation check retains the distinction for completeness.

## G.5 STRUCTURED DKW SIMULTANEOUS BASELINE

To compare the IUT against a policy-count-free baseline, let $F ( x ) = P ( S \leq x )$ and $G ( x ) = P ( S \leq$ $x , L = 1 )$ . The latter is the CDF on the extended line of the random variable equal to S when $L = 1$ and +∞ otherwise. By the Dvoretzky–Kiefer–Wolfowitz inequality with Massart’s constant (Massart, 1990), allocating failure probability $\delta / 2$ to each CDF gives, simultaneously,

$$
\| F _ { n } - F \| _ { \infty } \leq \epsilon _ { n } , \qquad \| G _ { n } - G \| _ { \infty } \leq \epsilon _ { n } , \qquad \epsilon _ { n } = \sqrt { \frac { \log ( 4 / \delta ) } { 2 n } } .
$$

For a union of at most K intervals, telescoping its $2 K$ endpoints gives $| c _ { n } ( A ) - c ( A ) | \leq \rho _ { K }$ and $| m _ { n } ( A ) - m ( A ) | \leq \rho _ { K }$ with $\rho _ { K } = 2 K \epsilon _ { n } .$ , where $m ( A ) = P ( S \in A , L = 1 )$ . Hence every population policy with $c ( A ) \geq \kappa$ must satisfy $c _ { n } ( A ) \geq \kappa - \rho _ { K }$ and

$$
r ( A ) = \frac { m ( A ) } { c ( A ) } \geq \frac { [ m _ { n } ( A ) - \rho _ { K } ] _ { + } } { \operatorname* { m i n } \{ 1 , c _ { n } ( A ) + \rho _ { K } \} } .
$$

The same weighted interval DP minimizes this expression exactly over the empirical $( t , e _ { K } ( t ) )$ frontier. This comparator exploits the one-dimensional interval structure and contains no factor $| \mathcal { A } _ { M , K } | ;$ it is nevertheless conservative when $K \epsilon _ { n }$ is not small.

## G.6 VC/RADEMACHER SIMULTANEOUS BASELINE

The class of unions of at most K intervals on a line has VC dimension 2K. To compare against a standard uniform-convergence route rather than only endpoint DKW or policywise Bonferroni, define ${ \mathcal { F } } _ { K } = \{ ( s , l ) \mapsto \mathbf { 1 } \{ s \in A \} : A \in { \mathcal { A } } _ { M , K } \}$ and $\mathring { \mathcal { G } } _ { K } = \{ ( s , l ) \mapsto l { \bf 1 } \bar { \{ s \in A \} } : A \in \mathcal { A } _ { M , K } \}$ . For either [0, 1]-valued class F, the standard empirical-Rademacher inequality gives, with probability at least $1 - \eta$

$$
\operatorname* { s u p } _ { f \in \mathcal { F } } \{ P f - P _ { n } f \} \leq 2 \widehat { \mathfrak { R } } _ { n } ( \mathcal { F } ) + 3 \sqrt { \frac { \log ( 2 / \eta ) } { 2 n } } ,
$$

and the reverse deviation follows by applying the same bound to $- \mathcal { F }$ (Mohri et al., 2018). We allocate the data failure budget 0.049 over the four one-sided events for $\mathcal { F } _ { K }$ and $\mathcal { G } _ { K }$ . The two empirical Rademacher expectations are evaluated by 200,000 independent sign draws; for each draw the supremum over at most K intervals is solved exactly by a vectorized interval DP. A Hoeffding upper confidence radius with failure probability $5 \times 1 0 ^ { - 4 }$ is added to each Monte-Carlo mean, leaving total data-plus-numerical failure probability at most 0.05.

Let the resulting two-sided radii be $\epsilon _ { c } ( K )$ for coverage and $\epsilon _ { m } ( K )$ for accepted-error mass. Then every population policy with $c ( A ) \geq \kappa$ obeys

$$
c _ { n } ( A ) \geq \kappa - \epsilon _ { c } ( K ) , \qquad r ( A ) \geq { \frac { [ m _ { n } ( A ) - \epsilon _ { m } ( K ) ] _ { + } } { \operatorname* { m i n } \{ 1 , c _ { n } ( A ) + \epsilon _ { c } ( K ) \} } } ,
$$

and the same weighted interval DP minimizes this lower bound over the class. This is a generic VC/Rademacher comparator, not a claim that the bound is sharp for interval unions.

## H EXPERIMENTAL DETAILS AND COMPLETE EMPIRICAL TABLES

## H.1 MATCHED-FAMILY EFFICIENCY SIMULATION

We fix $\boldsymbol { B } = 2 5 6 , \boldsymbol { p } = . 2 5 , \boldsymbol { \kappa } = . 5 , \boldsymbol { k } \in \{ 1 , 4 , 1 6 , 6 4 , 2 5 6 \}$ and $\gamma \in \{ . 0 4 , . 0 2 \}$ . The finite sample-size grid for IUT multiplies $( . 1 2 k + . 9 \sqrt { k } + 1 . 5 ) / \gamma ^ { 2 } \mathrm { b y } . 4 , . 6 , . 8 , 1 , 1 . 2 5 , 1 . 6 , 2 ,$ , rounded to multiples of $5 0 ;$ every crossing is bracketed. These grids are computational design choices, not fitted rate claims. Each of the 70 alternative grid points uses 1,500 independent audits. Ten boundary checks, one per $( k , \gamma )$ , use the grid point nearest observed 80% power, a boundary-null law and independent randomness, with 20,000 audits each (boundary\_all.py). For $k = 1 , 4 , 1 6 , 6 4 , 2 5 6$ the rejection rates are 4.73, 2.75, 1.44, 0.37 and 0.00% at $\gamma = . 0 4$ and 4.73, 2.59, 1.66, 0.29 and 0.00% at $\gamma = . 0 2 ;$ the largest Wilson upper limit is 5.03%. The alternatives use 105,000 IUT audits in total. Seeds are recorded in the scripts and depend on the configuration.

Table 7: Measured 80%-power sample crossings for fixed $B = 2 5 6$ . NP denotes the simple-mixture envelope, not a full-null-valid test. Ratios compare IUT to that envelope; no minimax constant is inferred.
<table><tr><td colspan="4"> $\gamma = . 0 4$ </td><td colspan="2"> $\gamma = . 0 2$ </td></tr><tr><td> $k$ </td><td>IUT</td><td>NP</td><td>ratio</td><td>IUT</td><td>NP</td></tr><tr><td>1</td><td>2,342</td><td>1,001</td><td>2.34</td><td>9,216</td><td>4,030</td></tr><tr><td>4</td><td>3,298</td><td>1,478</td><td>2.23</td><td>13,437</td><td>5,883</td></tr><tr><td>16</td><td>5,370</td><td>2,363</td><td>2.27</td><td>22,386</td><td>9,374</td></tr><tr><td>64</td><td>12,065</td><td>4,060</td><td>2.97</td><td>48,268</td><td>15,859</td></tr><tr><td>256</td><td>32,796</td><td>7,384</td><td>4.44</td><td>131,679</td><td>29,637</td></tr></table>

For the NP envelope, the grid multiplies $( . 7 \sqrt { k } + 1 ) / \gamma ^ { 2 }$ by .5, .7, .9, 1.1, 1.35, 1.65, 2, rounded to multiples of 25. At each of 70 points, 12,000 independent mixture draws calibrate a nominal 5% likelihood-ratio threshold; 6,000 fresh reference draws estimate power and 6,000 fresh mixture draws estimate size. The factorized likelihood ratio is invariant under flipping any group half, so sampling the all-+ feasible component gives the same LR distribution as sampling the uniform component mixture. We test this symmetry directly. Calibration is independent of both evaluations. Pointwise Wilson intervals for the evaluations condition on the calibrated threshold and do not include its calibration uncertainty. The empirical size column is retained; the fitted threshold is not asserted to have exact 5% size. The population NP envelope bounds full-null-valid power, but the estimated curve is descriptive.

For exact computation, subtract the smaller half count in every group from the accepted-count state. The reachable width is then at most $\begin{array} { r } { 1 + \sum _ { b } | n _ { b , + } - n _ { b , - } | . } \end{array}$ rather than $n + 1$ . A compiled implementation performs the same two-choice recurrence over this interval, with unreachable states at infinity. This compression is exact and is checked against the independent uncompressed dynamic program, including final binomial decisions. Source code and a compiler invocation are supplied; no prebuilt binary is needed.

![](images/5a22530759d7fa5d25456c813a3b228949d809a57a733738216da9c0b05c127d.jpg)

![](images/e24325962d61a56f681e31ac0ed754ca87409ac41bbe3dd3af6853f8b7e3fd67.jpg)  
Figure 4: Exact IUT power on the two risk-gap grids. Bars are pointwise Wilson 95% intervals over 1,500 audits. These are Monte Carlo uncertainties, not confidence bands over the sample-size grid.

The recorded constructive-witness fraction selects the half with more correct observations in each heavy group and a fixed half in residual groups. Whenever it has accepted count at least $n / 2$ and empirical risk at most $\alpha ,$ IUT rejection is impossible; the simulation checks this deterministic implication. The theorem does not require this particular witness to be optimal.

## H.2 STRUCTURED CERTIFICATE COMPARISON ON ALL FROZEN OUTPUTS

Semi-synthetic power study. The protocol is described in Section 6. Scripts: semisynthetic\_power.py, make\_semisyn\_tabl $\stackrel { \mathrm { \tiny ~ e . } } { \mathrm { \tiny ~ x ~ r } } \mathrm { p y } ,$ make\_semisyn\_full.py. Table 8 lists the IUT for every K and the other certificates at $K = 4$

Table 8: Semi-synthetic audits, all configurations (κ = .9; 500 audits per cell for n ≤ 5,000, 300 at n = 10,000, 200 at n = 20,000). Entries are rejection rates at $\alpha = { \bar { r } } ^ { * } - \Delta ;$ the column ∆ = 0 is the level at the true boundary claim. For CIFAR-10, ∆ = .10 gives α < 0 and is rejected trivially.
<table><tr><td>Population</td><td>Method</td><td>n K</td><td>Δ = 0</td><td></td><td>.02</td><td>.05</td><td>.10</td><td>median frontier</td></tr><tr><td>CIFAR-10</td><td>IUT</td><td>500</td><td></td><td>0.006</td><td>0.13</td><td>0.92</td><td>1.00</td><td>0.044</td></tr><tr><td>CIFAR-10</td><td>IUT</td><td>500</td><td>14</td><td>0.000</td><td>0.06</td><td>0.92</td><td>1.00</td><td>0.037</td></tr><tr><td>CIFAR-10</td><td>Bonferroni</td><td>500</td><td>4</td><td>0.000</td><td>0.00</td><td>0.00</td><td>1.00</td><td>0.003</td></tr><tr><td>CIFAR-10</td><td>DKW</td><td>500</td><td>4</td><td>0.000</td><td>0.00</td><td>0.00</td><td>1.00</td><td>0.000</td></tr><tr><td>CIFAR-10</td><td>Total loss</td><td>500</td><td>4</td><td>0.000</td><td>0.00</td><td>0.03</td><td>1.00</td><td>0.000</td></tr><tr><td>CIFAR-10</td><td>IUT</td><td>500</td><td>16</td><td>0.000</td><td>0.05</td><td>0.92</td><td>1.00</td><td>0.037</td></tr><tr><td>CIFAR-10</td><td>IUT</td><td>1,000</td><td>1</td><td>0.004</td><td>0.24</td><td>1.00</td><td>1.00</td><td>0.052</td></tr><tr><td>CIFAR-10</td><td>IUT</td><td>1,000</td><td>4</td><td>0.000</td><td>0.20</td><td>1.00</td><td>1.00</td><td>0.048</td></tr><tr><td>CIFAR-10</td><td>Bonferroni</td><td>1,000</td><td>4</td><td>0.000</td><td>0.00</td><td>0.00</td><td>1.00</td><td>0.013</td></tr><tr><td>CIFAR-10</td><td>DKW</td><td>1,000</td><td>4</td><td>0.000</td><td>0.00</td><td>0.00</td><td>1.00</td><td>0.000</td></tr><tr><td>CIFAR-10</td><td>Total loss</td><td>1,000</td><td>4</td><td>0.000</td><td>0.00</td><td>0.02</td><td>1.00</td><td>0.003</td></tr><tr><td>CIFAR-10</td><td>IUT</td><td>1,000</td><td>16</td><td>0.000</td><td>0.19</td><td>1.00</td><td>1.00</td><td>0.048</td></tr><tr><td>CIFAR-10</td><td>IUT</td><td>2,500</td><td>1</td><td>0.004</td><td>0.67</td><td>1.00</td><td>1.00</td><td>0.061</td></tr><tr><td>CIFAR-10</td><td>IUT</td><td>2,500</td><td>4</td><td>0.000</td><td>0.73</td><td>1.00</td><td>1.00</td><td>0.058</td></tr><tr><td>CIFAR-10</td><td>Bonferroni</td><td>2,500</td><td>4</td><td>0.000</td><td>0.00</td><td>0.90</td><td>1.00</td><td>0.030</td></tr><tr><td>CIFAR-10</td><td>DKW</td><td>2,500</td><td>4</td><td>0.000</td><td>0.00</td><td>0.00</td><td>1.00</td><td>0.000</td></tr><tr><td>CIFAR-10</td><td>Total loss</td><td>2,500</td><td>4</td><td>0.000</td><td>0.00</td><td>0.03</td><td>1.00</td><td>0.010</td></tr><tr><td>CIFAR-10</td><td>IUT</td><td>2,500</td><td>16</td><td>0.000</td><td>0.73</td><td>1.00</td><td>1.00</td><td>0.058</td></tr><tr><td>CIFAR-10</td><td>IUT</td><td>5,000</td><td>1</td><td>0.000</td><td>0.95</td><td>1.00</td><td>1.00</td><td>0.064</td></tr><tr><td>CIFAR-10</td><td>IUT</td><td>5,000</td><td>4</td><td>0.000</td><td>0.99</td><td>1.00</td><td>1.00</td><td>0.062</td></tr><tr><td>CIFAR-10</td><td>Bonferroni</td><td>5,000</td><td>4</td><td>0.000</td><td>0.00</td><td>1.00</td><td>1.00</td><td>0.041</td></tr><tr><td>CIFAR-10</td><td>DKW</td><td>5,000</td><td>4</td><td>0.000</td><td>0.00</td><td>0.00</td><td>1.00</td><td>0.000</td></tr><tr><td>CIFAR-10</td><td>Total loss</td><td>5,000</td><td>4</td><td>0.000</td><td>0.00</td><td>0.01</td><td>1.00</td><td>0.013</td></tr><tr><td>CIFAR-10</td><td>IUT</td><td>5,000</td><td>16</td><td>0.000</td><td>0.99</td><td>1.00</td><td>1.00</td><td>0.062</td></tr><tr><td>CIFAR-100</td><td>IUT</td><td>500</td><td>1</td><td>0.008</td><td>0.06</td><td>0.42</td><td>0.99</td><td>0.212</td></tr><tr><td>CIFAR-100</td><td>IUT</td><td>500</td><td>4</td><td>0.002</td><td>0.02</td><td>0.23</td><td>0.97</td><td>0.199</td></tr><tr><td>CIFAR-100</td><td>Bonferroni</td><td>500</td><td>4</td><td>0.000</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.090</td></tr><tr><td>CIFAR-100</td><td>DKW</td><td>500</td><td>4</td><td>0.000</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.000</td></tr><tr><td>CIFAR-100</td><td>Total loss</td><td>500</td><td>4</td><td>0.004</td><td>0.03</td><td>0.25</td><td>0.94</td><td>0.195</td></tr><tr><td>CIFAR-100</td><td>IUT</td><td>500</td><td>16</td><td>0.002</td><td>0.02</td><td>0.21</td><td>0.96</td><td>0.197</td></tr><tr><td>CIFAR-100</td><td>IUT</td><td>1,000</td><td>1</td><td>0.002</td><td>0.10</td><td>0.78</td><td>1.00</td><td>0.228</td></tr><tr><td>CIFAR-100</td><td>IUT</td><td>1,000</td><td>4</td><td>0.002</td><td>0.04</td><td>0.63</td><td>1.00</td><td>0.217</td></tr><tr><td>CIFAR-100</td><td>Bonferroni</td><td>1,000</td><td>4</td><td>0.000</td><td>0.00</td><td>0.00</td><td>0.01</td><td>0.134</td></tr><tr><td>CIFAR-100</td><td>DKW</td><td>1,000</td><td>4</td><td>0.000</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.000</td></tr><tr><td>CIFAR-100</td><td>Total loss</td><td>1,000</td><td>4</td><td>0.000</td><td>0.02</td><td>0.38</td><td>1.00</td><td>0.207</td></tr><tr><td>CIFAR-100</td><td>IUT</td><td>1,000</td><td>16</td><td>0.002</td><td>0.04</td><td>0.63</td><td>1.00</td><td>0.217</td></tr><tr><td>CIFAR-100</td><td>IUT</td><td>2,500</td><td>1</td><td>0.004</td><td>0.30</td><td>1.00</td><td>1.00</td><td>0.241</td></tr><tr><td>CIFAR-100</td><td>IUT</td><td>2,500</td><td>4</td><td>0.000</td><td>0.21</td><td>0.99</td><td>1.00</td><td>0.234</td></tr><tr><td>CIFAR-100</td><td>Bonferroni</td><td>2,500</td><td>4</td><td>0.000</td><td>0.00</td><td>0.00</td><td>0.96</td><td>0.179</td></tr><tr><td>CIFAR-100</td><td>DKW</td><td>2,500</td><td>4</td><td>0.000</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.000</td></tr><tr><td>CIFAR-100</td><td>Total loss</td><td>2,500</td><td>4</td><td>0.000</td><td>0.01</td><td>0.68</td><td>1.00</td><td>0.217</td></tr><tr><td>CIFAR-100</td><td>IUT</td><td>2,500</td><td>16</td><td>0.000</td><td>0.20</td><td>0.99</td><td>1.00</td><td>0.234</td></tr><tr><td>CIFAR-100</td><td>IUT</td><td>5,000</td><td>1</td><td>0.014</td><td>0.62</td><td>1.00</td><td>1.00</td><td>0.249</td></tr><tr><td>CIFAR-100</td><td>IUT</td><td>5,000</td><td>4</td><td>0.008</td><td>0.55 0.00</td><td>1.00 0.09</td><td>1.00 1.00</td><td>0.243 0.203</td></tr><tr><td>CIFAR-100 CIFAR-100</td><td>Bonferroni DKW</td><td>5,000 5,000</td><td>4 4</td><td>0.000 0.000</td></tr><tr><td>Population</td><td>Method</td><td>n</td><td>K</td><td>∆ = 0</td><td>.02</td><td>.05</td><td>.10</td><td>median</td></tr><tr><td>ImageNetV2</td><td>Bonferroni</td><td>1,000</td><td>4</td><td>0.000</td><td>0.00</td><td>0.00</td><td>0.16</td><td>0.083</td></tr><tr><td>ImageNetV2</td><td>DKW</td><td>1,000</td><td>4</td><td>0.000</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.000</td></tr><tr><td>ImageNetV2</td><td>Total loss</td><td>1,000</td><td>4</td><td>0.000</td><td>0.00</td><td>0.28</td><td>1.00</td><td>0.135</td></tr><tr><td>ImageNetV2</td><td>IUT</td><td>1,000</td><td>16</td><td>0.002</td><td>0.08</td><td>0.80</td><td>1.00</td><td>0.153</td></tr><tr><td>ImageNetV2</td><td>IUT</td><td>2,500</td><td>1</td><td>0.010</td><td>0.27</td><td>1.00</td><td>1.00</td><td>0.174</td></tr><tr><td>ImageNetV2</td><td>IUT</td><td>2,500</td><td>4</td><td>0.000</td><td>0.31</td><td>1.00</td><td>1.00</td><td>0.169</td></tr><tr><td>ImageNetV2</td><td>Bonferroni</td><td>2,500</td><td>4</td><td>0.000</td><td>0.00</td><td>0.00</td><td>1.00</td><td>0.120</td></tr><tr><td>ImageNetV2</td><td>DKW</td><td>2,500</td><td>4</td><td>0.000</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.000</td></tr><tr><td>ImageNetV2</td><td>Total loss</td><td>2,500</td><td>4</td><td>0.000</td><td>0.00</td><td>0.52</td><td>1.00</td><td>0.144</td></tr><tr><td>ImageNetV2</td><td>IUT</td><td>2,500</td><td>16</td><td>0.000</td><td>0.31</td><td>1.00</td><td>1.00</td><td>0.169</td></tr><tr><td>ImageNetV2</td><td>IUT</td><td>5,000</td><td>1</td><td>0.004</td><td>0.52</td><td>1.00</td><td>1.00</td><td>0.180</td></tr><tr><td>ImageNetV2</td><td>IUT</td><td>5,000</td><td>4</td><td>0.000</td><td>0.66</td><td>1.00</td><td>1.00</td><td>0.176</td></tr><tr><td>ImageNetV2</td><td>Bonferroni</td><td>5,000</td><td>4</td><td>0.000</td><td>0.00</td><td>0.28</td><td>1.00</td><td>0.140</td></tr><tr><td>ImageNetV2</td><td>DKW</td><td>5,000</td><td>4</td><td>0.000</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.000</td></tr><tr><td>ImageNetV2</td><td>Total loss</td><td>5,000</td><td>4</td><td>0.000</td><td>0.00</td><td>0.79</td><td>1.00</td><td>0.149</td></tr><tr><td>ImageNetV2</td><td>IUT</td><td>5,000</td><td>16</td><td>0.000</td><td>0.66</td><td>1.00</td><td>1.00</td><td>0.176</td></tr><tr><td>ImageNetV2</td><td>IUT</td><td>10,000</td><td>1</td><td>0.013</td><td>0.79</td><td>1.00</td><td>1.00</td><td>0.187</td></tr><tr><td>ImageNetV2</td><td>IUT</td><td>10,000</td><td>4</td><td>0.003</td><td>0.94</td><td>1.00</td><td>1.00</td><td>0.181</td></tr><tr><td>ImageNetV2</td><td>Bonferroni</td><td>10,000</td><td>4</td><td>0.000</td><td>0.00</td><td>1.00</td><td>1.00</td><td>0.156</td></tr><tr><td>ImageNetV2</td><td>DKW</td><td>10,000</td><td>4</td><td>0.000</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.000</td></tr><tr><td>ImageNetV2</td><td>Total loss</td><td>10,000</td><td>4</td><td>0.000</td><td>0.00</td><td>0.97</td><td>1.00</td><td>0.152</td></tr><tr><td>ImageNetV2</td><td>IUT</td><td>10,000</td><td>16</td><td>0.003</td><td>0.94</td><td>1.00</td><td>1.00</td><td>0.181</td></tr><tr><td>ImageNetV2</td><td>IUT</td><td>20,000</td><td>1</td><td>0.015</td><td>0.99</td><td>1.00</td><td>1.00</td><td>0.193</td></tr><tr><td>ImageNetV2</td><td>IUT</td><td>20,000</td><td>4</td><td>0.000</td><td>1.00</td><td>1.00</td><td>1.00</td><td>0.184</td></tr><tr><td>ImageNetV2</td><td>Bonferroni</td><td>20,000</td><td>4</td><td>0.000</td><td>0.00</td><td>1.00</td><td>1.00</td><td>0.167</td></tr><tr><td>ImageNetV2</td><td>DKW</td><td>20,000</td><td>4</td><td>0.000</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.045</td></tr><tr><td>ImageNetV2</td><td>Total loss</td><td>20,000</td><td>4</td><td>0.000</td><td>0.00</td><td>1.00</td><td>1.00</td><td>0.155</td></tr><tr><td>ImageNetV2</td><td>IUT</td><td>20,000</td><td>16</td><td>0.000</td><td>1.00</td><td>1.00</td><td>1.00</td><td>0.184</td></tr><tr><td>CMU-MOSEI</td><td>IUT</td><td>500</td><td>1</td><td>0.012</td><td>0.12</td><td>0.72</td><td>1.00</td><td>0.104</td></tr><tr><td>CMU-MOSEI</td><td>IUT</td><td>500</td><td>4</td><td>0.002</td><td>0.03</td><td>0.53</td><td>1.00</td><td>0.088</td></tr><tr><td>CMU-MOSEI</td><td>Bonferroni</td><td>500</td><td>4</td><td>0.000</td><td>0.00</td><td>0.00</td><td>0.09</td><td>0.025</td></tr><tr><td>CMU-MOSEI</td><td>DKW</td><td>500</td><td>4</td><td>0.000</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.000</td></tr><tr><td>CMU-MOSEI</td><td>Total loss</td><td>500</td><td>4</td><td>0.000</td><td>0.00</td><td>0.03</td><td>0.79</td><td>0.052</td></tr><tr><td>CMU-MOSEI</td><td>IUT</td><td>500</td><td>16</td><td>0.002</td><td>0.02</td><td>0.48</td><td>1.00</td><td>0.086</td></tr><tr><td>CMU-MOSEI</td><td>IUT</td><td>1,000</td><td>1</td><td>0.006</td><td>0.23</td><td>0.97</td><td>1.00</td><td>0.116</td></tr><tr><td>CMU-MOSEI</td><td>IUT</td><td>1,000</td><td>4</td><td>0.000</td><td>0.10</td><td>0.97</td><td>1.00</td><td>0.103</td></tr><tr><td>CMU-MOSEI</td><td>Bonferroni</td><td>1,000</td><td>4</td><td>0.000</td><td>0.00</td><td>0.00</td><td>0.98</td><td>0.049</td></tr><tr><td>CMU-MOSEI</td><td>DKW</td><td>1,000</td><td>4</td><td>0.000</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.000</td></tr><tr><td>CMU-MOSEI</td><td>Total loss</td><td>1,000</td><td>4</td><td>0.000</td><td>0.00</td><td>0.02</td><td>0.98</td><td>0.058</td></tr><tr><td>CMU-MOSEI</td><td>IUT</td><td>1,000</td><td>16</td><td>0.000</td><td>0.09</td><td>0.96</td><td>1.00</td><td>0.102</td></tr><tr><td>CMU-MOSEI</td><td>IUT</td><td>2,500</td><td>1</td><td>0.004</td><td>0.60</td><td>1.00</td><td>1.00</td><td>0.127</td></tr><tr><td>CMU-MOSEI</td><td>IUT</td><td>2,500</td><td>4</td><td>0.000</td><td>0.52</td><td>1.00</td><td>1.00</td><td>0.117</td></tr><tr><td>CMU-MOSEI</td><td>Bonferroni</td><td>2,500</td><td>4</td><td>0.000</td><td>0.00</td><td>0.08</td><td>1.00</td><td>0.079</td></tr><tr><td>CMU-MOSEI</td><td>DKW</td><td>2,500</td><td>4</td><td>0.000 0.000</td><td>0.00 0.00</td><td>0.00 0.00</td><td>0.00 1.00</td><td>0.000 0.067</td></tr><tr><td>CMU-MOSEI CMU-MOSEI</td><td>Total loss IUT</td><td>2,500 2,500</td><td>4 16</td></tr></table>

Fixed-split frontiers on 12 outputs. We use one predeclared split, seed 3401, for each of the 12 frozen model outputs. Calibration uses half the scores and no audit labels; audit counts use the disjoint half. We report every combination of $K = 1 , 4 , 1 6 { \mathrm { a t } } \kappa = . 9$ . For each configuration, IUT, policywise Bonferroni, and DKW use the nominal error budget .05. The Rademacher calculation uses 200,000 signs for each of the coverage and accepted-error classes; Hoeffding upper confidence corrections cost .0005 each, with the remaining .049 allocated over four one-sided data events. Guarantees are pointwise in K, not simultaneous over all reported configurations. These methods have different inferential targets internally but each resulting frontier is valid for the same existential contract. The global-loss comparator is deliberately described as score-agnostic, and no data-dependent maximum of unadjusted certificates is used.

Table 9: Fixed-split structured comparison at $\kappa = . 9 .$ . All frontiers are percentages, pointwise 95% under the stated sampling assumptions. Numerical uncertainty is included for the empirical-Rademacher comparator.
<table><tr><td>Output / K</td><td>IUT</td><td>Bonf.</td><td>DKW</td><td>Rad.</td><td>Global</td></tr><tr><td>CIFAR-10 / 1</td><td>6.40</td><td>5.64</td><td>0.99</td><td>0.00</td><td>1.37</td></tr><tr><td>CIFAR-10 / 4</td><td>6.22</td><td>4.16</td><td>0.00</td><td>0.00</td><td>1.37</td></tr><tr><td>CIFAR-10 / 16</td><td>6.22</td><td>3.28</td><td>0.00</td><td>0.00</td><td>1.37</td></tr><tr><td>CIFAR-100 / 1</td><td>24.56</td><td>23.16</td><td>18.87</td><td>8.35</td><td>22.25</td></tr><tr><td>CIFAR-100 / 4</td><td>24.29</td><td>20.37</td><td>0.00</td><td>4.29</td><td>22.25</td></tr><tr><td>CIFAR-100 / 16</td><td>24.29</td><td>18.82</td><td>0.00</td><td>1.93</td><td>22.25</td></tr><tr><td>CIFAR-100 (2) / 1</td><td>24.11</td><td>21.62</td><td>17.75</td><td>8.28</td><td>21.50</td></tr><tr><td>CIFAR-100 (2) / 4</td><td>23.70</td><td>19.74</td><td>0.00</td><td>3.93</td><td>21.50</td></tr><tr><td>CIFAR-100 (2) / 16</td><td>23.70</td><td>17.82</td><td>0.00</td><td>1.44</td><td>21.50</td></tr><tr><td>CIFAR-100 (3) / 1</td><td>23.58</td><td>22.20</td><td>17.46</td><td>8.09</td><td>21.45</td></tr><tr><td>CIFAR-100 (3) / 4</td><td>23.58</td><td>19.77</td><td>0.00</td><td>3.77</td><td>21.45</td></tr><tr><td>CIFAR-100 (3) / 16</td><td>23.58</td><td>17.77</td><td>0.00</td><td>1.63</td><td>21.45</td></tr><tr><td>CIFAR-10 (2) / 1</td><td>6.61</td><td>5.26</td><td>1.20</td><td>0.00</td><td>1.39</td></tr><tr><td>CIFAR-10 (2) / 4</td><td>6.57</td><td>4.39</td><td>0.00</td><td>0.00</td><td>1.39</td></tr><tr><td>CIFAR-10 (2) / 16</td><td>6.57</td><td>3.41</td><td>0.00</td><td>0.00</td><td>1.39</td></tr><tr><td>CIFAR-10 (3) / 1</td><td>6.81</td><td>5.24</td><td>0.84</td><td>0.00</td><td>1.17</td></tr><tr><td>CIFAR-10 (3) / 4</td><td>6.17</td><td>3.91</td><td>0.00</td><td>0.00</td><td>1.17</td></tr><tr><td>CIFAR-10 (3) / 16</td><td>6.17</td><td>3.36</td><td>0.00</td><td>0.00</td><td>1.17</td></tr><tr><td>ImageNetV2 / 1</td><td>17.78</td><td>16.55</td><td>12.45</td><td>2.95</td><td>14.89</td></tr><tr><td>ImageNetV2 /4</td><td>17.71</td><td>14.13</td><td>0.00</td><td>0.00</td><td>14.89</td></tr><tr><td>ImageNetV2 / 16</td><td>17.71</td><td>12.58</td><td>0.00</td><td>0.00</td><td>14.89</td></tr><tr><td>ImageNetV2 (2) / 1</td><td>24.83</td><td>22.61</td><td>18.39</td><td>8.27</td><td>22.53</td></tr><tr><td>ImageNetV2 (2) / 4</td><td>24.12</td><td>20.22</td><td>0.00</td><td>4.24</td><td>22.53</td></tr><tr><td>ImageNetV2 (2) / 16</td><td>24.12</td><td>18.83</td><td>0.00</td><td>2.04</td><td>22.53</td></tr><tr><td>ImageNetV2 (3) / 1</td><td>27.43</td><td>25.26</td><td>21.18</td><td>10.35</td><td>25.68</td></tr><tr><td>ImageNetV2 (3) / 4</td><td>27.38</td><td>23.04</td><td>0.00</td><td>6.24</td><td>25.68</td></tr><tr><td>ImageNetV2 (3) / 16</td><td>27.24</td><td>21.20</td><td>0.00</td><td>3.69</td><td>25.68</td></tr><tr><td>MOSEI / 1</td><td>12.44</td><td>9.94</td><td>3.80</td><td>0.00</td><td>6.53</td></tr><tr><td>MOSEI / 4</td><td>11.58</td><td>7.20</td><td>0.00</td><td>0.00</td><td>6.53</td></tr><tr><td>MOSEI / 16</td><td>11.58</td><td>5.45</td><td>0.00</td><td>0.00</td><td>6.53</td></tr><tr><td>MOSEI (2) / 1</td><td>10.80</td><td>8.72</td><td>2.89</td><td>0.00</td><td>5.68</td></tr><tr><td>MOSEI (2) / 4</td><td>10.80</td><td>6.56</td><td>0.00</td><td>0.00</td><td>5.68</td></tr><tr><td>MOSEI (2) / 16</td><td>10.80</td><td>4.86</td><td>0.00</td><td>0.00</td><td>5.68</td></tr><tr><td>MOSEI (3) / 1</td><td>13.75</td><td>11.47</td><td>5.24</td><td>0.00</td><td>8.65</td></tr><tr><td>MOSEI (3) / 4</td><td>13.51</td><td>8.56</td><td>0.00</td><td>0.00</td><td>8.65</td></tr><tr><td>MOSEI (3) / 16</td><td>13.30</td><td>6.55</td><td>0.00</td><td>0.00</td><td>8.65</td></tr></table>

## H.3 HELD-OUT IMAGENETV2 IUT AUDIT

For the reference-free certificate, we use the frozen ImageNetV2 ViT-B/16 output and split it 50/50 with seed 3401. The calibration half defines 57 ordered cells from the stored max\_softmax\_score channel; the audit half contains 5,000 images and supplies all labels/counts. We search unions of at most K contiguous cell intervals and report the 90%-coverage contract. Alongside the exact IUT we report policywise Bonferroni, the structured DKW comparator of Appendix G.5, and the VC/Rademacher comparator of Appendix G.6.

Table 10: Held-out ImageNetV2 audit at $\kappa = . 9 0$ . Lower frontiers are 95% valid under each method’s stated probability budget. “Rad.” is the Monte-Carlo-certified empirical-Rademacher uniform-convergence comparator.
<table><tr><td>K</td><td>policy count</td><td>empirical</td><td>IUT</td><td>Bonf.</td><td>DKW</td><td>Rad.</td></tr><tr><td>1</td><td> $1 . 6 5 \times 1 0 ^ { 3 }$ </td><td>19.996%</td><td>17.783%</td><td>16.552%</td><td>12.451%</td><td>2.947%</td></tr><tr><td>2</td><td> $4 . 2 6 \times 1 0 ^ { 5 }$ </td><td>19.410%</td><td>17.783%</td><td>15.242%</td><td>4.660%</td><td>1.085%</td></tr><tr><td>4</td><td> $1 . 9 6 \times 1 0 ^ { 9 }$ </td><td>19.348%</td><td>17.712%</td><td>14.135%</td><td>0.000%</td><td>0.000%</td></tr><tr><td>8</td><td> $9 . 1 0 \times 1 0 ^ { 1 3 }$ </td><td>19.348%</td><td>17.712%</td><td>13.321%</td><td>0.000%</td><td>0.000%</td></tr><tr><td>16</td><td> $1 . 2 3 \times 1 0 ^ { 1 7 }$ </td><td>19.348%</td><td>17.712%</td><td>12.584%</td><td>0.000%</td><td>0.000%</td></tr></table>

The IUT value stabilizes once additional intervals no longer improve the relevant frontier. DKW accumulates endpoint error linearly in K, while the empirical-Rademacher baseline explicitly controls the VC-2K interval-union class without a raw policy-count correction but is still conservative at this sample size. Thus the comparison does not rest on Bonferroni as the only alternative. A one-policy split-sample select-then-test procedure is not listed because it tests only the selected member and is not level-valid for the existential union null. We do not claim any displayed comparator is minimax optimal for the existential problem. On the supplied CPU implementation $( M = 5 7 , n = 5 0 0 0 )$ median DP times are 1.15, 1.60, 2.43, 4.34, 8.13, and 14.80 ms for $K = 1 , 2 , 4 , 8 , 1 6 , 3 2$ , respectively; absolute timing is environment-specific.

![](images/5c2d1d96dd10464e5a9812c628d1c644a979c93d30dc2aeb5c44032a1cfc73d9.jpg)  
Figure 5: Held-out ImageNetV2 audit at $\kappa = . 9 0 .$ . The exact IUT frontier is shown with policywise Bonferroni, structured DKW, empirical-Rademacher uniform convergence, and the empirical frontier. This illustrates Proposition 1; it is not a minimax-optimality comparison.

Boundary-null implementation check. We additionally simulate 1,000 i.i.d. audits from a finitecell null containing a known boundary witness: 12 equally likely score cel $1 \mathrm s , A ^ { * }$ equal to the first nine cells, $c ( A ^ { * } ) \bar { = } . 7 5 , r ( A ^ { * } ) = . 2 0 , K = 3$ , and $n = 3 0 0$ . The IUT+DP implementation rejects the true null in 9/1,000 runs, an empirical rate of 0.9% with 95% Wilson interval 0.47–1.70%. This is intentionally only an implementation regression check; Proposition 1 supplies the exact finite-sample level guarantee.

Operating-characteristic check. To distinguish conservativeness at the boundary from lack of power, we keep the same $M = 1 2 , K = 3 , n = 3 0 0 , \kappa = . 7 5$ , and target $\alpha = . 2 0$ design and vary the best population risk. Over 2,000 simulations per point, rejection rates are:

<table><tr><td>best population risk</td><td>.20</td><td>.22</td><td>.25</td><td>.30</td><td>.35</td></tr><tr><td>IUT rejection rate</td><td>.0115</td><td>.0555</td><td>.3015</td><td>.8620</td><td>.9955</td></tr></table>

This is a descriptive power curve, not a replacement for Proposition 1’s exact level guarantee. It shows that the 0.9–1.2% boundary rejection rate is a feature of conservative exact one-sided bound rather than a procedure that remains inert away from the boundary.

Family-flexibility power cost. The phrase “no multiplicity correction” concerns the level proof, not immunity of power to the size of the searched family. To isolate this distinction, we simulate a homogeneous alternative with $M = 1 2 , n = 3 0 0 , \kappa = . 7 5$ , target $\alpha = . 2 0$ , and population risk .25 in every score cell. Hence every policy in every interval-union class has exactly the same population risk; increasing K creates no better population witness. Over 3,000 simulations, IUT rejection falls from 23.1% at $K = 1$ to 7.6% at $K = 2 , 4 . 7 \%$ at $K = 3 ,$ and 4.4% at $K = 4$ and 6. The median lower frontier simultaneously falls from 18.17% to 16.03%. Thus family flexibility can materially reduce power even though the exact type-I argument pays no raw |A| factor.

## H.4 ABSTENTION-ODDS SIMULATION

Design. Scores are uniform over $M = 6 4$ equal-mass cells and $L \sim \mathrm { B e r n } ( . 1 4 )$ independently of the score, so every union of cells has risk . $1 4 = \alpha + \gamma$ with $\alpha = . 1 0 , \gamma = . 0 4 ;$ the law lies in $\dot { \mathcal { P } } _ { \gamma } ( A _ { K } )$ for every K and $\kappa ,$ and every rule is exactly as bad as every other. The menus are unions of at most $K \in \{ \dot { 1 } , 2 , 4 , 8 , 1 6 \}$ intervals of cells, and the IUT uses $\bar { \zeta _ { c } } = \zeta _ { r } = . 0 2 5$ . On a grid of log-spaced sizes from 150 to 15,846 (the first 15 points of a 19-point grid up to 60,000; the run was stopped once every power had reached 1) we draw 250 audits per size, with seeds 6101 and 6102 for the two interleaved halves of the grid. One dynamic program per audit serves all $( K , \kappa )$ . Crossings are log-linear interpolations of Monte Carlo power; with 250 audits the power standard error is at most .032, which moves a crossing by a few percent.

Results. Table 11 lists all crossings. The normalization divides by the abstention odds and by $\log ( e / \operatorname* { m i n } \{ \kappa , 1 - \kappa \} )$ , a logarithmic factor suggested by localized VC bounds (Appendix B.3); without that logarithm the normalized costs still vary much less than the raw ones (at $K = 4 \colon 3 . 9$ at $\kappa = . 5$ and 9.8 at $\kappa = . 9 5$ , against a raw variation of 7.7-fold in the opposite direction). At $\kappa = . 3$ the normalized costs are 0.56–0.64 times those at $\kappa = . 5 $ , within the constant factor that separates $( 1 - \kappa ) / \kappa$ from min $\{ \kappa , 1 - \kappa \} / \kappa ^ { 2 } . \mathrm { A t } \kappa \geq . 9$ and $K \geq 8$ the crossings stop growing: the abstained region contains only $M ( 1 - \dot { \kappa } ) \leq 6 . 4$ 4 cells, so no new rule competes. The constant-risk law is not the least favorable law of Theorem 1; the experiment checks the κ-dependence of the IUT’s cost, which is governed by the upper bound. Script: kappa\_law.py; figure and table: make\_kappa\_figure.py, mak $\mathsf { \Pi } _ { \mathsf { - k a p p a \_ t a b l e . p y } }$

Table 11: Abstention-odds simulation: 80%-power crossings $n _ { 8 0 }$ of the exact IUT $( M = 6 4$ equal cells, constant risk . $1 4 , \alpha = . 1 0 , \gamma = . 0 4$ , 250 audits per grid point, log-linear interpolation), and normalized menu costs $\gamma ^ { 2 } \{ n _ { 8 0 } ( K ) - n _ { 8 0 } ( 1 ) \} / \{ \frac { 1 - \kappa } { \kappa } \stackrel {  } { \log } \frac { e } { \operatorname* { m i n } ( \kappa , 1 - \kappa ) } \}$
<table><tr><td>K\κ</td><td>.3</td><td>.5</td><td>.7</td><td>.8</td><td>.9</td><td>.95</td></tr><tr><td>1</td><td>3,832</td><td>1,958</td><td>1,070</td><td>881</td><td>755</td><td>656</td></tr><tr><td>2</td><td>5,764</td><td>3,092</td><td>1,847</td><td>1,402</td><td>973</td><td>839</td></tr><tr><td>4</td><td>8,252</td><td>4,414</td><td>2,737</td><td>1,887</td><td>1,316</td><td>977</td></tr><tr><td>8</td><td>11,961</td><td>6,151</td><td>3,458</td><td>2,584</td><td>1,568</td><td>1,024</td></tr><tr><td>16</td><td>14,050</td><td>7,389</td><td>3,921</td><td>2,732</td><td>1,580</td><td>1,024</td></tr><tr><td>norm.,  $K = 2$ </td><td>0.60</td><td>1.07</td><td>1.32</td><td>1.28</td><td>0.95</td><td>1.39</td></tr><tr><td>norm.,  $K = 4$ </td><td>1.38</td><td>2.32</td><td>2.82</td><td>2.47</td><td>2.45</td><td>2.45</td></tr><tr><td>norm.,  $K = 8$ </td><td>2.53</td><td>3.96</td><td>4.04</td><td>4.18</td><td>3.54</td><td>2.80</td></tr><tr><td>norm.,  $K = 1 6$ </td><td>3.18</td><td>5.13</td><td>4.83</td><td>4.54</td><td>3.60</td><td>2.80</td></tr></table>

## H.5 A VALID NON-EFC AUDIT: CROSS-FIT UNIVERSAL INFERENCE

Identify each law with its pushforward to the 2M cell-by-loss categories; as in Appendix B.2, this loses nothing for the null because every rule is a union of cells. For $A \in { \mathcal { A } }$ let $\mathcal { H } _ { A }$ be the laws under which A meets the contract, so $\mathcal { H } = \dot { \bigcup _ { A } } \mathcal { H } _ { A }$

Lemma 4 (Profile likelihood of the existential null). Let x be category counts with total $n _ { 0 } ,$ , and let cˆ be the empirical coverage of A. When $\hat { c } > 0 ,$ let rˆ be its empirical risk; when $\hat { c } = 0 ;$ , define the product $\hat { c } \mathbf { 1 } \hat { \{ r > \alpha \} } \mathrm { k l } ( \hat { r } , \tilde { \alpha } )$ below to be zero. Then

$$
\operatorname* { s u p } _ { Q \in \mathcal { H } A } \sum _ { i } x _ { i } \log Q _ { i } = \sum _ { i } x _ { i } \log \frac { x _ { i } } { n _ { 0 } } - n _ { 0 } \Big [ \mathbf { 1 } \big \{ \hat { c } < \kappa \big \} \mathbf { k } \mathbf { l } ( \hat { c } , \kappa ) + \hat { c } \mathbf { 1 } \big \{ \hat { r } > \alpha \big \} \mathbf { k } \mathbf { l } ( \hat { r } , \alpha ) \Big ] .
$$

Proof. The constraints of $\mathcal { H } _ { A }$ involve Q only through the four masses $Q ( \tilde { A , 0 } ) , Q ( A , 1 ) , Q ( A ^ { c } , 0 ) , Q ( A ^ { c } , 1 )$ . For fixed masses, the likelihood is maximized by splitting each mass in proportion to the counts, which gives $\begin{array} { r } { \sum _ { i } x _ { i } \log ( x _ { i } / n _ { 0 } ) - n _ { 0 } \operatorname { K L } ( \hat { p } \| Q _ { 4 } ) } \end{array}$ , where pˆ and $Q _ { 4 }$ are the empirical and model four-cell laws. Parametrize $Q _ { 4 } ^ { \cdot } \mathfrak { b } \mathfrak { y } c = Q ( A ) , \ddot { r } = Q ( L = \bar { 1 } \mid A )$ and $s = Q ( L = 1 \mid A ^ { c } )$ . The chain rule gives $\mathrm { K L } ( \hat { p } \| Q _ { 4 } ) = \mathrm { k l } ( \hat { c } , c ) + \hat { c } \mathrm { k l } ( \hat { r } , r ) + ( 1 - \hat { c } ) \mathrm { k l } ( \hat { s } , s )$ with zero-weight conditional terms defined as zero (so the empirical conditional proportion on an empty block may be chosen arbitrarily). The constraints are $c \geq \kappa$ and $r \leq \alpha$ , and the three terms are minimized separately by $c = \operatorname* { m a x } \{ \hat { c } , \kappa \}$ , by $r = \operatorname* { m i n } \{ \hat { r } , \alpha \}$ when $\hat { c } > 0 .$ , and by $s = \hat { s }$ when $\hat { c } < 1 ;$ the corresponding zero-weight choices are immaterial. □

For $t > 0$ , the bracket is increasing in the error count for a fixed accepted count, so the minimal error counts $e _ { K } ( t )$ of the interval dynamic program (Appendix G) suffice. If a nonempty frozen score cell receives zero observations in the evaluated split, a nonempty rule with $t = 0$ is also available and contributes the coverage-only penalty $\operatorname { k l } ( 0 , \kappa )$ ; we include this candidate explicitly. The structurally empty mask is not a witness for the existential null. On small instances the formula agrees with numerical constrained optimization to $1 0 ^ { - 9 }$ , and the dynamic-program minimum, including the zero-count edge case, agrees with brute-force enumeration.

The audit. Split the audit sample into halves $D _ { 0 } , D _ { 1 }$ . Let $\hat { P } _ { 1 }$ be the add-1/2 smoothed empirical law of $D _ { 1 }$ over the 2M categories, $\begin{array} { r } { \Lambda _ { 0 } = \hat { P } _ { 1 } ^ { n _ { 0 } } ( D _ { 0 } ) / \operatorname* { s u p } _ { Q \in \mathcal { H } } Q ^ { n _ { 0 } } ( D _ { 0 } ) } \end{array}$ the split likelihood ratio, $\Lambda _ { 1 }$ its mirror image, and reject when $( \Lambda _ { 0 } + \Lambda _ { 1 } ) / 2 \geq 1 / \dot { \delta }$ . For every $Q \in \mathcal { H } , \mathbb { E } _ { Q } [ \Lambda _ { 0 } ~ | ~ D _ { 1 } ] \leq$ $\mathbb { E } _ { Q } [ \hat { P } _ { 1 } ^ { n _ { 0 } } ( D _ { 0 } ) / Q ^ { n _ { 0 } } ( D _ { 0 } ) \mid D _ { 1 } ] \leq 1 \colon$ explicitly, the latter expectation sums $\hat { P } _ { 1 } ^ { n _ { 0 } }$ only over sample sequences with positive $Q ^ { n _ { 0 } }$ mass, and equals one when $\hat { P } _ { 1 } \ll Q$ . The same holds for $\Lambda _ { 1 }$ , so the average is an e-value and Markov’s inequality gives level δ over the unrestricted null (Wasserman et al., 2020). The audit is not EFC: it may reject although a rule is empirically feasible on the full sample. It is a natural reference-free competitor, and it is not the testing-scale audit of Theorem 4, which requires constant risk.

Results. Table 12 compares it with the IUT on the same draws. No boundary rejection was observed for the universal-inference audit. The IUT’s largest boundary rejection rate in these runs is 4.0% (ImageNetV2, K = 1, n = 50,000, 200 audits), within Monte Carlo error of its exact level; up to $n = 2 0 { , } 0 0 0$ it matches the at most 1.8% of Table 8 within that error. The $\chi ^ { 2 }$ independence test rejected in all 200 audits at $n = 5 0 0$ on every population. The numerator must fit the full cell-by-loss law, which has up to $2 M - 1 = 1 1 3$ free parameters here (105 for CIFAR-10 after ties are merged); this is a substantial finite-sample burden that the IUT does not incur. Script: nonefc\_semisyn.py; table: make\_nonefc\_table.py.

Table 12: A valid non-EFC audit on the frozen populations $( \kappa = . 9 , K = 4$ , 200 audits per point, $n \in \{ 5 0 0 , 1 0 0 0 , 2 5 0 0 , 5 0 0 0 , 1 0 ^ { 4 } , 2 \times 1 0 ^ { 4 } , 5 \times 1 0 ^ { 4 } \} )$ . Entries give the smallest grid size at which the rejection rate of the false claim $\alpha = r ^ { * } - \Delta$ reaches 80%. “Level” is the largest rejection rate of a true boundary claim over all n and $K \in \{ 1 , 4 , 1 6 \}$ . The last column is the median p-value of a $\chi ^ { 2 }$ test of cell–loss independence at $n = 5 0 0 .$
<table><tr><td rowspan="2">Population</td><td rowspan="2"> $r ^ { * }$ </td><td colspan="2">∆ = .05</td><td colspan="2"> $\Delta = . 0 2$ </td><td colspan="2">level split LR</td><td rowspan="2">independence median p</td></tr><tr><td>IUT</td><td>split LR</td><td>IUT</td><td>split LR</td><td>IUT</td><td></td></tr><tr><td>CIFAR-10</td><td>0.074</td><td>500</td><td>10,000</td><td>5,000</td><td>&gt; 50,000</td><td>2.0%</td><td>0.0%</td><td>2e-17</td></tr><tr><td>CIFAR-100</td><td>0.262</td><td>2,500</td><td>50,000</td><td>10,000</td><td>&gt; 50,000</td><td>3.0%</td><td>0.0%</td><td>3e-19</td></tr><tr><td>ImageNetV2</td><td>0.193</td><td>2,500</td><td>50,000</td><td>10,000</td><td>&gt; 50,000</td><td>4.0%</td><td>0.0%</td><td>3e-17</td></tr><tr><td>CMU-MOSEI</td><td>0.136</td><td>1,000</td><td>50,000</td><td>5,000</td><td>&gt; 50,000</td><td>3.5%</td><td>0.0%</td><td>5e-10</td></tr></table>

## I ARTIFACT PROVENANCE AND AUTOMATED CHECKS

The release includes the frozen dependency artifact plus an analysis directory containing the operational benchmark, sign-family simulation, full-null window reconstruction, exact parity certificates, validation tests, and generated manuscript tables. All computational configurations have recorded deterministic seeds. Figures and numerical controls computed from frozen outputs are identified as such; their source scripts and input arrays are included. The parity certificates are verified with rational arithmetic; the sign dynamic program and binomial rejection lookup are checked against independent exhaustive calculations; singleton radii are checked against constrained four-cell numerical optimization. The release README supplies commands, software versions, and claim-to-file mappings.