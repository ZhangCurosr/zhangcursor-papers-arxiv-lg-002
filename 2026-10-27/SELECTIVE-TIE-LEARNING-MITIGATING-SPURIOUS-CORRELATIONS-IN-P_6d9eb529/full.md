# SELECTIVE TIE LEARNING: MITIGATING SPURIOUS CORRELATIONS IN PREFERENCE OPTIMIZATION

Christian Moya<sup>1†∗</sup>, Purav Matlia<sup>1†</sup>, Elliott Thornley<sup>2</sup>, and Guang Lin<sup>1</sup> <sup>1</sup> Purdue University, <sup>2</sup> National University of Singapore

## ABSTRACT

Preference optimization can learn spurious correlations, and augmenting the preference data with ties (equally preferred responses) can reduce spurious reliance. However, ties differ in their corrective effect, making their selection for augmentation important. For log-linear policies, we derive each tie’s influence on spurious reliance and show that taking the largest predicted reductions is first-order optimal among subsets of equal size. Computing these influences requires knowledge of the spurious features and the curvature of the training objective, which is often unavailable or expensive to estimate. To this end, we explore using the magnitude of a tie’s implicit reward margin and characterize the conditions under which it is a useful proxy for predicted influence. To test the theory and its qualitative transfer to neural models, we estimate influence with a counterfactual proxy for spurious reliance and use a generalized Gauss–Newton approximation to curvature for language models. Across log-linear policies, neural models, and large language models, our selection rules consistently achieve comparable reductions in spurious reliance with fewer ties than uniform selection.

## 1 INTRODUCTION

Aligning large language models with human preferences is central to developing useful and safe AI systems (Ouyang et al., 2022; Bai et al., 2022). Standard methods for this alignment, including reinforcement learning from human feedback (RLHF) and direct preference optimization (DPO) (Rafailov et al., 2023), learn policies from pairwise comparisons of responses. Learning from comparisons, however, can encourage reliance on features that predict human labels without tracking substantive quality, such as verbosity (Singhal et al., 2023; Saito et al., 2023) and sycophancy (Sharma et al., 2024). Models that rely on these features can fail when the features stop predicting human labels at deployment.

Recent theory shows that reliance on spurious features can persist even with unlimited preference data, because it arises from properties of the training distribution (Moya et al., 2026). Collecting more examples from the same distribution therefore does not remove it. Because the bias originates in the training distribution, two families of interventions target it: methods that reshape the training objective (Park et al., 2024) and methods that modify the training examples (Liu et al., 2025; Kim et al., 2025). The two families are complementary, since interventions on the training examples can work with existing objectives for preference learning rather than replace them. Our work advances interventions of this second kind.

Tie training is one such data intervention (Moya et al., 2026). It augments preference data with ties: pairs of responses with equal utility that differ in spurious attributes. Under local log-linear assumptions, this augmentation provably suppresses reliance on these attributes. Related data interventions include the neutral pairs of RRM (Liu et al., 2025) and counterfactual augmentation for length bias (Kim et al., 2025). However, both methods keep many augmented pairs even after filtering, which adds training computation. This cost raises a question: can careful selection retain most of the reduction in spurious reliance with fewer ties?

To study this question, we analyze log-linear policies with a core–spurious feature decomposition. Within this setting, we propose Selective Tie Learning (STL), which selects a small subset of ties using a model trained on the original preference data. Our contributions are:

(i) We derive the first-order influence of each tie on the model’s reliance on spurious features (Proposition 1). Selecting the ties with the largest predicted reductions is first-order optimal among subsets of the same size (Theorem 2).

(ii) Because computing the influence in (i) requires the Hessian of the loss, we derive a cheaper selector that needs only forward passes. We show that, under a Gaussian model of the ties, selecting the ties with the largest absolute implicit reward margins maximizes the expected first-order reduction in spurious reliance among all such selectors (Theorem 3).

(iii) To extend (i) to neural models, we estimate the influence of ties using a counterfactual measure of spurious reliance and a damped generalized Gauss–Newton approximation of the Hessian (Section 5). The estimate needs no explicit core–spurious feature decomposition and shares one Hessian inversion across all candidate ties.

We test the theoretical predictions in log-linear models and evaluate their transfer to neural networks and language models. Across these settings, targeted selection achieves comparable reductions in spurious reliance with fewer ties than uniform selection.

## 2 SETUP AND PROBLEM FORMULATION

## 2.1 DPO WITH A LOG-LINEAR POLICY

Preference data. Preference learning trains a policy from comparisons between responses to the same prompt. These comparisons form a dataset $\dot { \mathcal { D } } _ { \mathrm { p r e f } } = \{ ( \dot { x } _ { i } , y _ { w , i } , y _ { l , i } ) \} _ { i = } ^ { N }$ sampled from a distribution $P ,$ , where $y _ { w , i }$ is preferred to $y _ { l , i }$ for prompt $x _ { i } \in { \mathcal { X } }$ (Rafailov et al., 2023; Christiano et al., 2017).

Core and spurious features. We represent the prompt and responses in each comparison using a fixed feature map $\phi ( x , y ) = [ \phi _ { c } ( x , y ) ; \phi _ { s } ( x , y ) ]$ ] and assume that the preference depends only on the core features $\phi _ { c }$ . The spurious features $\phi _ { s }$ can still correlate with the preference in the training data, although this correlation need not persist at deployment.

Log-linear policy. To analyse how a policy uses these features, we assume a log-linear form, $\pi _ { \boldsymbol { \theta } } ( y$ $x ) \stackrel { \smile } { = } \exp ( \hat { \theta } ^ { \top } \phi ( \bar { x } , y ) ) / Z _ { \theta } \bar { ( } x )$ , where the normalizer $Z _ { \theta } ( x )$ sums over responses and $\theta = [ \theta _ { c } ; \bar { \theta } _ { s } ]$ weights the core and spurious features. We assume the fixed reference policy $\pi _ { \mathrm { r e f } } = \pi _ { \theta _ { \mathrm { r e } } }$ belongs to the same family. The log-ratio log $\left( \pi _ { \theta } / \pi _ { \mathrm { r e f } } \right)$ then equals $( \theta - \theta _ { \mathrm { r e f } } ) ^ { \top } \phi ( x , y )$ up to a normalization term that cancels when comparing responses to the same prompt.

DPO. DPO scales this log-ratio into the implicit reward $r _ { \theta } ( x , y ) ~ = ~ \beta \log  ( \pi _ { \theta } ( y ~ \mid ~ x ) / \pi _ { \mathrm { r e f } } ( y ~ \mid ~ $ $x ) )$ , where $\beta > 0$ controls regularization toward the reference policy. Within each comparison the normalization cancels, and the reward difference is $\beta \tilde { \theta } ^ { \top } \Delta \phi .$ , where $\tilde { \theta } = \theta - \theta _ { \mathrm { r e f } }$ and $\Delta \phi =$ $\phi ( x , y _ { w } ) - \phi ( x , y _ { l } )$ . DPO fits this difference to the observed preferences through the logistic loss

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { D P O } } ( \boldsymbol { \theta } ) = - \mathbb { E } _ { \mathcal { D } _ { \mathrm { p r e f } } } \Big [ \log \sigma \big ( \beta \boldsymbol { \tilde { \theta } } ^ { \top } \Delta \boldsymbol { \phi } \big ) \Big ] , } \end{array}
$$

where $\sigma$ is the logistic function. Because $\Delta \phi = [ \Delta \phi _ { c } ; \Delta \phi _ { s } ]$ , both core and spurious features contribute to the reward difference. DPO can therefore reduce the loss by exploiting spurious correlations that predict the winner.

## 2.2 SPURIOUS RELIANCE AND TIES AS AN INTERVENTION

DPO exploits spurious correlations even at the population level (Moya et al., 2026). With $\mu ~ =$ $\mathbb { E } _ { P } [ \Delta \phi ]$ and $\Sigma { \dot { = } } \mathbb { E } _ { P } [ \Delta \phi \Delta \phi ^ { \top } ]$ , two mechanisms drive this reliance. Through the mean, spurious differences predict the winner directly $( \mu _ { s } \neq 0 )$ ). Through leakage, they correlate with core differences $( \Sigma _ { s c } \neq 0 )$ and inherit their predictive signal. Both depend on $P$ alone, so more comparisons from the same distribution do not remove the reliance.

Ties as an intervention. These two mechanisms motivate comparisons that carry neither a mean spurious bias nor leakage from core differences. We call such comparisons ties: a tie $\tau = ( x , \{ y _ { A } , y _ { B } \} )$ pairs two responses with identical core features and different spurious features, so its feature difference $\Delta \phi = [ 0 ; \Delta \phi _ { s } ]$ has no core part to leak from. Assigning either response as the winner with equal probability also makes the mean spurious difference zero. Moya et al. (2026) show that adding such ties to DPO training reduces spurious reliance.

## 2.3 PROBLEM: SELECTING TIES FROM A POOL

The benefit of tie training (Moya et al., 2026) leaves open which ties to use. Ties differ because a tie acts on the policy only through its margin: if the policy already assigns equal implicit reward to both responses, training on the tie changes nothing. Training on the whole pool $\mathcal { D } _ { \mathrm { t i e } }$ of $M$ candidates therefore spends effort on ties that change little. Under a budget of $k < M$ ties, we instead seek the subset of the given pool whose training most reduces spurious reliance. Such a subset should outperform k ties sampled uniformly, the simplest selector.

In this paper, each candidate comes from a strict comparison sampled from $\mathcal { D } _ { \mathrm { p r e f } }$ . A known transformation T then changes the spurious features of the preferred response while approximately preserving its core features, $\phi _ { c } ( x , T ( y _ { w } ) ) - \phi _ { c } ( x , y _ { w } ) \approx 0 .$ So that each response wins equally often, each candidate is a packet with both orientations, $\boldsymbol { \tau } = \{ \tau _ { \ell } , \tau _ { r } \}$ , where ${ \tau } _ { \ell } = ( x , y _ { w } , T ( y _ { w } ) )$ and $\tau _ { r } = ( x , T ( y _ { w } ) , y _ { w } )$ with equal probability, so their feature differences cancel in expectation, while each orientation has approximately zero core difference. We score and select whole packets, then sample one orientation from each selected packet for training. Our selectors also accept pools from other constructions that satisfy the assumptions of our analysis. We study this selection problem at two levels of access to the model. Section 3 derives the best selector when gradients and curvature are available, and Section 4 derives the best selector when only forward passes are available.

## 3 SELECTION WITH FULL ACCESS: INFLUENCE

In this section, we measure spurious reliance by a spurious energy and analyze, to first order, how training on a tie changes it. We show that this first-order change is the classical influence of the tie on the energy (Koh & Liang, 2017), so selecting the k ties with the largest predicted reduction solves the selection problem to first order.

## 3.1 THE FIRST-ORDER EFFECT OF A TIE ON SPURIOUS ENERGY

Spurious energy. Spurious reliance shows up in the margin. The margin on a comparison splits into a core and a spurious part, $\beta ( \tilde { \theta } _ { c } ^ { \top } \Delta \phi _ { c } + \tilde { \theta } _ { s } ^ { \top } \Delta \phi _ { s } )$ , and only the core part reflects the preference. We therefore measure reliance by the mean square of the spurious part, before scaling by $\beta \colon$

$$
\boldsymbol { E } ( \boldsymbol { \theta } ) = \mathbb { E } _ { P } \left[ ( \tilde { \theta } _ { s } ^ { \top } \Delta \phi _ { s } ) ^ { 2 } \right] = \tilde { \theta } _ { s } ^ { \top } \boldsymbol { \Sigma } _ { s s } \tilde { \theta } _ { s } .
$$

Squaring counts spurious contributions of either sign, and weighting by $\Sigma _ { s s }$ counts spurious weights only along directions in which responses actually differ. The energy vanishes exactly when spurious features never move the margin under $P ,$ and a tie is useful to the extent that training on it lowers this energy.

We condition on the realized candidate pool and analyze each packet $\tau _ { j }$ through its DPO loss averaged over both orientations. We assume that the two responses in a tie have the same core features, so the feature differences of the two orientations are $[ 0 ; \bar { d } _ { j } ]$ and $[ 0 ; - d _ { j } ]$ , where

$$
d _ { j } = \phi _ { s } ( x _ { j } , y _ { w , j } ) - \phi _ { s } ( x _ { j } , T ( y _ { w , j } ) ) .
$$

Because the loss is averaged over both orientations, reversing the sign of $d _ { j }$ leaves it unchanged, so either orientation can define $d _ { j }$ . With margin $m _ { j } = \beta \tilde { \theta } _ { s } ^ { \top } d _ { j }$ for the first orientation, the tie loss is

$$
\begin{array} { r } { \ell _ { j } ( { \boldsymbol { \theta } } ) = - \frac { 1 } { 2 } \big [ \log \sigma ( { \boldsymbol { m } } _ { j } ) + \log \sigma ( - { \boldsymbol { m } } _ { j } ) \big ] . } \end{array}
$$

To isolate the effect of one tie, we add it to DPO with a small weight $\epsilon / N _ { \dagger }$

$$
\theta _ { j } ^ { \epsilon } = \arg \operatorname* { m i n } _ { \theta } \mathcal { L } _ { \mathrm { D P O } } ( \theta ) + \frac { \epsilon } { N } \ell _ { j } ( \theta ) ,
$$

so that $\epsilon = 0$ recovers the DPO solution $\theta _ { 0 }$ . The effect of the tie on reliance is then the rate at which the spurious energy $E ( \theta _ { j } ^ { \epsilon } )$ changes as ϵ grows from zero.

Tie gradient. The gradient of the tie loss, $g _ { j } = \nabla _ { \theta } \ell _ { j }$ , drives this rate. Because the margin on a tie depends only on the spurious weights, so does the gradient:

$$
g _ { j , c } = 0 , \qquad g _ { j , s } = \frac { \beta } { 2 } \operatorname { t a n h } \Bigl ( \frac { m _ { j } } { 2 } \Bigr ) d _ { j } .
$$

This is the gradient of the packet loss, averaged over both orientations. Flipping the orientation flips both $d _ { j }$ and $m _ { j }$ , so the packet gradient does not depend on which orientation defines the contrast $d _ { j }$ $\mathbf { A }$ tie therefore pushes the policy only along its spurious contrast, with a strength set by its margin. The push vanishes when the policy already assigns equal implicit reward to both responses, grows as $\begin{array} { l } { { \frac { \beta } { 4 } } m _ { j } d _ { j } } \end{array}$ for small margins, and saturates at magnitude $\frac { \beta } { 2 } \left. d _ { j } \right.$ for large ones.

From push to influence. The gradient gives the immediate push of a tie, but DPO resists it, and the curvature of the DPO loss decides how far the solution moves. We analyze the DPO policy $\theta _ { 0 }$ , a solution at which the DPO loss is stationary and strictly convex,

$$
\nabla \mathcal { L } _ { \mathrm { D P O } } ( \theta _ { 0 } ) = 0 , \qquad H = \nabla ^ { 2 } \mathcal { L } _ { \mathrm { D P O } } ( \theta _ { 0 } ) \succ 0 .
$$

Adding the tie with weight $\epsilon / N$ tilts this stationarity condition, and the solution restores it by moving at rate ${ \bf \bar { \Phi } } - H ^ { - 1 } g _ { j } ( \theta _ { 0 } ) / { N }$ . Only the spurious part of this move changes the energy. While the tie pushes the spurious weights, DPO refits the core weights, so the spurious weights move at rate $- K _ { s } ^ { - 1 } g _ { j , s } ( \boldsymbol { \theta } _ { 0 } ^ { \cdot } ) / N$ , where $K _ { s } = H _ { s s } - H _ { s c } H _ { c c } ^ { - 1 } H _ { c s }$ is the Schur complement of the core block. The resulting rate of change of the energy is the influence of the tie (Koh & Liang, 2017).

Proposition 1 (Influence of a tie on spurious energy). Suppose $\theta _ { 0 }$ is stationary with $H \succ 0 .$ . For small $\epsilon ,$ the solution $\theta _ { j } ^ { \epsilon }$ satisfies $E ( { \theta } _ { j } ^ { \epsilon } ) \bar { - } E ( \theta _ { 0 } ) = \bar { - \epsilon } A _ { j } \bar { + \cal O } ( \epsilon ^ { 2 } )$ , with

$$
A _ { j } = \frac { 1 } { N } \big \langle \nabla _ { s } E ( \theta _ { 0 } ) , K _ { s } ^ { - 1 } g _ { j , s } ( \theta _ { 0 } ) \big \rangle = \frac { \beta } { 2 N } \operatorname { t a n h } \Big ( \frac { m _ { j } } { 2 } \Big ) \big \langle \nabla _ { s } E ( \theta _ { 0 } ) , K _ { s } ^ { - 1 } d _ { j } \big \rangle ,
$$

where $m _ { j } = \beta \tilde { \theta } _ { 0 , s } ^ { \top } d _ { j }$ is the margin on the tie at $\theta _ { 0 } .$

Proof. See Appendix A.1.

Reading the influence. A positive $A _ { j }$ predicts that the tie lowers the energy, and the influence shows what makes it positive. Its size depends on two factors. The margin factor tanh $. ( m _ { j } / 2 )$ is the strength of the push: a tie on which the policy is already indifferent has no influence. The alignment factor $\langle \nabla _ { s } E ( \dot { \theta _ { 0 } } ) , K _ { s } ^ { - 1 } d _ { j } \rangle$ measures whether that push, after DPO refits the core weights, points against the current reliance. The alignment factor need not share the sign of the margin, so a tie can raise the energy. Influences of different ties add to first order, so choosing a set of ties reduces to ranking them.

## 3.2 MAIN RESULT: FIRST-ORDER OPTIMAL SELECTION

Training on a set $S$ of ties, each with weight $\epsilon / N ,$ , tilts the stationarity condition by the sum of their gradients. At first order the solution responds linearly to this tilt, so the influences of the ties add:

$$
E ( \theta _ { S } ^ { \epsilon } ) - E ( \theta _ { 0 } ) = - \epsilon \sum _ { j \in S } A _ { j } + O ( \epsilon ^ { 2 } ) .
$$

Under the budget, the best set is the one that makes this sum largest.

Theorem 2 (First-order optimal tie selection). Let $S _ { k }$ contain the k ties with the largest $A _ { j }$ . Then

$$
\sum _ { j \in S _ { k } } A _ { j } = \operatorname* { m a x } _ { | S | = k } \sum _ { j \in S } A _ { j } \ \geq \ \frac { k } { M } \sum _ { j = 1 } ^ { M } A _ { j } ,
$$

with strict inequality for $0 < k < M$ unless all $A _ { j }$ are equal.

Proof. See Appendix A.1.4.

Interpretation. The theorem ranks ties by influence, that is, by how hard each tie pushes and where the push points. The right side is what uniform selection achieves in expectation, so the theorem also measures what selection buys: k times the gap between the mean influence of the top k ties and that of the pool. Selection therefore pays off exactly when ties differ, which is the premise of Section 2.3.

Scope: first-order result. The theorem holds to first order, and the $O ( \epsilon ^ { 2 } )$ terms capture how ties interact. Two ties with similar spurious contrasts overlap, because training on one shrinks the margin of the other. Every selected tie also moves $\theta _ { 0 }$ and, with it, the margins and curvature that define the remaining influences. The guarantee holds while the total tie weight is small relative to $N .$ . Training uses larger weights, so Section 6 tests whether the first-order ranking still selects better ties.

The cost of influence. Computing $A _ { j }$ requires three quantities: the inverse curvature $K _ { s } ^ { - 1 }$ , the energy gradient $\nabla _ { s } E ( \theta _ { 0 } )$ , and the gradient of each tie. For a neural policy, each is expensive, and $\nabla _ { s } E$ requires knowing the spurious features themselves. Of the two factors in $A _ { j }$ , only the margin is available from a forward pass. This raises the question of the next section: using only forward passes, which selector best approximates influence on average?

## 4 SELECTION WITH FORWARD PASSES: MARGIN

Section 3 showed that ranking ties by influence minimizes spurious energy to first order, but computing influence requires the inverse curvature, the energy gradient, and a gradient for each tie. We now ask how well ties can be ranked using only forward passes. We introduce pure margin selection, which ranks ties by the absolute margin that the DPO solution $\theta _ { 0 }$ assigns to each of them. Under a Gaussian model of the spurious contrasts, it maximizes the expected first-order reduction in energy among selectors that see only the margins.

## 4.1 WHAT A FORWARD PASS REVEALS

A forward pass through the policy and the reference returns the log-probability of each response, and hence its implicit reward. On a tie, the difference between the two implicit rewards is the margin $m _ { j }$ from Section 3.1. We orient each tie uniformly at random, which leaves its influence unchanged, so that $m _ { j } = r _ { \theta _ { 0 } } ( x _ { j } , y _ { A , j } ) - r _ { \theta _ { 0 } } ( x _ { j } , y _ { B , j } ) = \beta \widetilde { \theta } _ { 0 , s } ^ { \top } d _ { j }$ , where $y _ { A , j }$ is the response that the sampled orientation places first. Forward passes thus deliver the margin of every tie in the pool at the cost of evaluating the two models.

The margin reveals one factor of the influence and hints at the other. From Proposition 1,

$$
A _ { j } = \frac { \beta } { 2 N } \operatorname { t a n h } \Bigl ( \frac { m _ { j } } { 2 } \Bigr ) a _ { j } , \qquad a _ { j } = \bigl \langle \nabla _ { s } E ( \theta _ { 0 } ) , K _ { s } ^ { - 1 } d _ { j } \bigr \rangle .
$$

The margin fixes the margin factor, while the alignment $a _ { j }$ stays hidden because it requires the curvature and the energy gradient. Both, however, are linear in the same contrast $d _ { j }$ , so the margin carries information about the alignment. A selector that sees only the margins can do no better than rank ties by their expected influence given those margins, and the margin factor passes through this expectation unchanged. Everything therefore hinges on how the expected alignment depends on the margin.

## 4.2 MAIN RESULT: PURE MARGIN SELECTION IS OPTIMAL IN EXPECTATION

Assumptions. How the expected alignment depends on the margin is a question about how contrasts are distributed across the pool. The random orientation centers them, $\mathbb { E } [ d _ { j } ] = 0$ . We condition on the strict training data, fixing the fitted checkpoint $\theta _ { 0 }$ and its Hessian $H$ , and regard the population moment $\Sigma _ { s s }$ as fixed. For tractability, we assume that, under this conditioning, the candidate contrasts are independent across ties and satisfy $d _ { j } \sim \mathcal { N } ( 0 , c \Sigma _ { s s } )$ , where $c > 0$ . The covariance models ties that inherit the spurious geometry of $P \colon$ after whitening by $\Sigma _ { s s } .$ , the contrasts are isotropic, as in Moya et al. (2026). Under this model the alignment and the margin are jointly Gaussian, so the expected alignment grows linearly with the margin, with a positive slope. The following theorem characterizes the expected influence given a tie’s margin, with expectations taken under the conditional model specified by our assumptions.

Theorem 3 (Ranking ties from margins). Under the assumptions above, with $\widetilde { \theta } _ { 0 , s } \neq 0 ,$ , there is a constant $\lambda > 0 ;$ , common to all ties, such that

$$
\mathbb { E } [ A _ { j } \mid m _ { j } ] = \frac { \lambda } { N } m _ { j } \operatorname { t a n h } \Bigl ( \frac { m _ { j } } { 2 } \Bigr ) .
$$

Thus, among selectors that see only the margins, selecting the k ties with the largest $| m _ { j } |$ maximizes the expectedfirst-order reduction in energy.

## Proof. See Appendix A.2.

Interpretation. The expected influence $m _ { j }$ tanh $( m _ { j } / 2 )$ is even and increases with $| m _ { j } | ,$ so the sign of the margin, which records only the sampled orientation, drops out of the ranking. We call the resulting selector pure margin selection. It works for two reasons. A larger absolute margin produces a stronger push from the tie. Under the covariance assumption, it also predicts better alignment with the direction that lowers the energy. A tie with zero margin has neither, which makes precise the observation in Section 2.3 that such a tie changes nothing.

Scope of the result. The theorem concerns the expected influence given the margins. Two ties with the same margin can still differ in influence, because their contrasts can point in different directions, so pure margin selection need not recover the influence ranking of a particular pool. Section 6 compares the two selectors directly.

From theory to neural models. Both derived selectors rest on the log-linear model assumption. In a neural policy, the margin stays exact and cheap, because it still requires only forward passes. Influence, by contrast, requires a proxy for the energy, whose spurious features are unobserved, and an estimate of the curvature. Section 5 builds both.

## 5 FROM THEORY TO NEURAL MODELS

Section 4 showed that pure margin selection maximizes the expected first-order reduction in energy under a Gaussian model of the spurious contrasts, while influence attains the first-order optimum without that model. We now implement both selectors in neural policies. Pure margin selection carries over directly, because it needs only forward passes. Influence needs two estimates: a counterfactual energy, which measures spurious reliance on a fixed probe of ties, and a damped estimate of the curvature of the DPO loss on the strict comparisons. In practice, the DPO policy $\theta _ { 0 }$ is the final checkpoint of DPO training, which satisfies the stationarity condition of Section 3 only approximately.

## 5.1 INFLUENCE IN PRACTICE

Counterfactual energy. A neural policy exposes no spurious features, so we cannot compute the energy of Section 3 directly. We can still approximate it, because a tie isolates the spurious part of the margin: its two responses differ only in spurious features, so its margin is purely spurious. We therefore form a fixed probe $\mathcal { P }$ of $N _ { v }$ ties, held out from selection, and define the counterfactual energy

$$
E _ { \mathrm { c f } } ( \theta ) = \frac { 1 } { N _ { v } } \sum _ { n = 1 } ^ { N _ { v } } m _ { n } ( \theta ) ^ { 2 } ,\tag{1}
$$

where $m _ { n }$ is the margin on probe tie n. Squaring removes the dependence on its orientation.

In the log-linear model, the counterfactual energy recovers the energy gradient up to a positive scale when the probe contrasts have the covariance assumed in Section 4 (see Appendix A.3): $\begin{array} { r } { \mathbb { E } _ { \mathcal { P } } [ \nabla E _ { \mathrm { c f } } ( \theta _ { 0 } ) ] \stackrel { \cdot } { = } c \beta ^ { 2 } \nabla E ( \theta _ { 0 } ) } \end{array}$ . The scale leaves the ranking of ties unchanged, so the counterfactual energy supplies the gradient that influence needs.

Curvature. The remaining ingredient is the curvature. In a neural policy, influence keeps the form of Proposition 1 but runs over all parameters: the full inverse curvature accounts for how every parameter responds to a tie, the role that $K _ { s }$ played for the spurious weights. We estimate the curvature with a damped generalized Gauss–Newton (GGN) matrix $\widehat { H }$ (Schraudolph, 2002; Martens, 2020), computed on a subset of the strict comparisons in $\mathcal { D } _ { \mathrm { p r e f } }$ , and score each tie by

$$
\widehat { A } _ { j } = \frac { 1 } { N } \big \langle \nabla E _ { \mathrm { c f } } ( \theta _ { 0 } ) , \widehat { H } ^ { - 1 } g _ { j } ( \theta _ { 0 } ) \big \rangle .
$$

Every estimated influence shares the same energy gradient, so we apply the inverse curvature to it once, $v = \widehat { H } ^ { - 1 } \nabla E _ { \mathrm { c f } } ( \theta _ { 0 } )$ , and reuse the result across the pool. Estimating the influence of a tie then takes only its gradient, $\widehat { A } _ { j } = v ^ { \top } g _ { j } ( \theta _ { 0 } ) / N$ . These approximations make influence practical but do not preserve the log-linear guarantee. We provide more details in Appendix B.1.

## 5.2 MARGIN IN PRACTICE

Pure margin selection ranks ties by the absolute margin $| m _ { j } |$ , which forward passes of the DPO policy and the reference deliver for every tie in the pool. Because margins are all it uses, the selector skips everything that influence requires: the probe, the curvature, and the tie gradients. It never even computes the energy, yet under the assumptions of Section 4 its ranking maximizes the expected first-order reduction of that energy.

Two selectors, two costs. The two selectors thus trade cost for information. Influence uses tie gradients, the curvature, and a probe to approximate the first-order optimum. Pure margin selection uses only margins and is optimal in expectation under a Gaussian model of the contrasts. Section 6 measures what each buys over uniform selection: whether the first-order ranking helps at the tie weights used in training, and whether pure margin selection loses anything relative to influence.

## 6 EXPERIMENTS

We test three theoretical predictions: (i) influence predicts the effect of an isolated tie (Proposition 1), (ii) mean influence conditioned on the tie margin follows the predicted curve (Theorem 3), and (iii) the highest-scoring ties reduce spurious reliance faster than uniformly sampled ties (Theorem 2). We begin with log-linear models and then ask whether the selection gains persist in two settings closer to practice: neural networks and language models. In every regime, spurious features correlate with preferences under the training distribution P, and the adversarial deployment distribution $Q _ { \mathrm { a d v } }$ reverses these correlations. We score candidate ties using a checkpoint trained on strict comparisons alone. For each selected set, we train a new model on the strict comparisons and selected ties, starting from the same initialization as the strict checkpoint. Appendix C gives the full protocol and compares this procedure with continued training from the strict checkpoint. The code is available at https://github.com/purav-0000/selective-tie-learning.

## 6.1 LOG-LINEAR: QUANTITATIVE VALIDATION

Setup. We generate Gaussian features $\phi = [ \phi _ { c } ; \phi _ { s } ]$ with $d _ { c } = d _ { s } = 1 0$ . The operator $T$ keeps the strict winner’s core features $\phi _ { c }$ and replaces its spurious features $\phi _ { s }$ with those of a response drawn uniformly from $\mathcal { D } _ { \mathrm { p r e f } }$ . We compare uniform tie learning (UTL), pure margin selection (PM), and Oracle influence, which computes $A _ { j }$ in Proposition 1 exactly from the population spurious-energy gradient and the empirical DPO Hessian. Appendix C.3 gives the full details.

Validating the theory. Panel (a) tests prediction (i): for each tie, we add its symmetric packet with weight $1 / N \left( \epsilon = 1 \right.$ in Proposition 1), re-optimize the model, and compare the observed reduction $E ( \tilde { \theta _ { 0 } } ) - \dot { E } ( \theta _ { i } ^ { 1 } )$ with $A _ { j }$ . Panel (b) tests prediction (ii): we repeat this with each tie in its fixed training orientation, group ties into ten equal-frequency bins by absolute margin, and compare each bin’s mean observed reduction with the conditional mean in Theorem 3.

Results. Figure 1 (a)–(b) supports predictions (i) and (ii). Across all isolated ties, predicted influence has 4.2% relative $\ell _ { 2 }$ error and 99.8% sign agreement with the observed reduction in spurious energy, including every tie predicted to increase it. In panel (b), the theoretical conditional mean lies within the 95% confidence interval of every margin bin.

Panels (c)–(d) show that the first-order ranking in prediction (iii) remains useful at full tie weight. PM and Oracle influence raise accuracy under $Q _ { \mathrm { a d v } }$ and lower spurious energy faster than UTL: with 100 ties, PM reaches the accuracy that UTL reaches with about 300. Their advantage narrows as the selected set approaches the full pool, where all methods coincide. PM closely tracks Oracle influence even though it uses only forward passes.

![](images/26ad4b246d93b1a410d47fe3d0813bdb6f9e2722b8ac2bfce593b909ba192db3.jpg)  
Figure 1: Log-linear validation. (a) Predicted versus observed reductions in population spurious energy after perturbing the objective with one symmetric tie. (b) Observed reductions from ties with sampled orientations and the theoretical conditional mean across margin bins containing equal numbers of ties (Theorem 3). (c–d) Accuracy under $Q _ { \mathrm { a d v } }$ and population spurious energy after training on selected ties. Curves are means; shading shows 95% Student-t confidence intervals.

## 6.2 NEURAL NETWORKS: QUALITATIVE PERSISTENCE

The log-linear theory no longer applies once a nonlinear representation hides the core–spurious decomposition. We next ask whether its selection gains persist in a neural reward model.

Setup. We keep the log-linear data-generating process and tie construction, pass the features through a fixed random nonlinear mixer, and train an MLP reward model with DPO on the mixed features. We apply T before mixing the features, so each tie still preserves core content exactly. We compare UTL and PM with Practical influence, which replaces the population spurious-energy gradient with the gradient of the counterfactual energy $E _ { \mathrm { c f } }$ and the Hessian with a damped GGN (Section 5). Appendix C.4 gives the full details.

Results. PM and Practical influence improve accuracy under $Q _ { \mathrm { a d v } }$ over UTL at every budget (Figure 2 (a)). At 500 ties, both gain about 7.5 percentage points over UTL and exceed the accuracy that UTL reaches with 1,500 ties. Their advantage first grows and then narrows as the selected set approaches the full pool. As in the log-linear model, PM closely tracks influence.

## 6.3 LLMS: CONTROLLED PREFERENCE TASKS

Language models introduce a pretrained policy and text responses. We test selection on two tasks whose spurious attributes we can control, so that the shifts and ties can be constructed exactly.

Setup. We fine-tune LoRA adapters of Llama-3.2-1B-Instruct with DPO, using the base model as the reference. In the hotel task, the model recommends one of two generated hotel profiles. Utility depends on price, distance, star rating, and a fixed noise term for each hotel, while seven other fields, such as building age and chain tier, correlate with utility positively in $P$ and negatively in $Q _ { \mathrm { a d v } }$ . The operator T replaces these seven fields in the winner’s profile with those of a hotel drawn from $\mathcal { D } _ { \mathrm { p r e f } }$ In the emoji task, the safe response of a PKU-SafeRLHF pair (Ji et al., 2025) is always preferred, and we insert different numbers of one emoji into its two responses. The safe response usually receives more emojis in P and fewer in $Q _ { \mathrm { a d v } }$ . Here, T removes every emoji from the safe winner. We compare UTL, PM, and Practical influence. Appendix C.5 gives the full details.

Results. PM and Practical influence improve mean accuracy under $Q _ { \mathrm { a d v } }$ over UTL at every partia budget on both tasks (Figure 2 (b)–(c)). With 500 ties, PM reaches 92.1% accuracy on hotels and 68.4% on emojis, compared with 74.6% and 59.3% for UTL. PM needs only 300 ties on hotels and 500 on emojis to exceed the accuracy that UTL reaches with 750. The advantage again narrows as the selected set approaches the full pool. The two scores track each other on hotels, whereas PM stays above Practical influence on emojis.

![](images/9ca927dae035b9feacf288dba1bbd66bd1d6e594b26e0e7843fdde828bc087ce.jpg)  
Figure 2: Selection beyond the log-linear setting. $Q _ { \mathrm { a d v } }$ accuracy for a nonlinear reward model (a) and LoRA-DPO on hotel (b) and emoji (c) tasks. Practical influence uses a sampled GGN and solves the damped system in its row space for all three neural regimes. Curves are means across seeds; shading shows 95% Student-t confidence intervals.

## 7 RELATED WORK

Spurious correlations in preference learning. Preference learning can reward attributes that predict labels without tracking response quality. Reward models often favor longer answers even when they are not more helpful (Singhal et al., 2023), and sycophancy persists in models trained with human feedback (Sharma et al., 2024). Studies of reward-model exploitation and overoptimization reveal related shortcuts, including verbosity (Dubois et al., 2023; Casper et al., 2023; Gao et al., 2023). These works document concrete instances of spurious learning; our analysis characterizes how individual ties change reliance on spurious features.

Mitigating spurious preferences. Existing methods intervene through the objective, reward model, internal representations, or training data. Some penalize or normalize length in the preference objective (Park et al., 2024; Lu et al., 2024), separate length from quality in reward models (Chen et al., 2024), or steer representations at inference (Rimsky et al., 2024). RRM (Liu et al., 2025) and counterfactual augmentation for length bias (Kim et al., 2025) are closest to our setting: they construct neutral or tie-like comparisons and train on many of them. STL instead selects from a fixed pool of ties under a training budget.

Theory and selection. Linear and log-linear models explain how the geometry of the training distribution produces unstable predictors (Nagarajan et al., 2021; Bombari & Mondelli, 2025; Veitch et al., 2021; Arjovsky et al., 2019). For DPO, Moya et al. (2026) trace population-level reliance to mean spurious bias and core–spurious correlation, and show that tie training suppresses it in aggregate. Active preference methods also select comparisons, but usually to reduce annotation cost or uncertainty, improve exploration, or filter difficult strict examples (Muldrew et al., 2024; Lin et al., 2026; Gao et al., 2025; Das et al., 2025; Melo et al., 2024). Patterning uses susceptibilities to reweight preference pairs toward desired changes in reward-model behavior (Wang et al., 2026). STL shares this use of first-order sensitivity to guide data interventions, selecting ties by their influence on spurious reliance.

## 8 CONCLUSION

We introduced Selective Tie Learning, a principled method for selecting a small subset of ties. For log-linear models, we derived the first-order influence of each tie on spurious reliance and showed that selecting the largest predicted reductions is first-order optimal. We also developed pure margin selection, an inexpensive rule that needs only forward passes, and showed that under a Gaussian model of the ties, it maximizes the expected reduction in spurious reliance among such rules. For neural models, we estimated tie influence without an explicit core–spurious decomposition. Across log-linear models, neural networks, and language models, selected ties yielded higher accuracy than uniformly sampled ties when spurious correlations were reversed, while using fewer ties.

Our study assumes an existing pool of ties. It does not address how to construct reliable ties or how to choose the budget k, although both decisions affect practical performance. Our experiments use fully or partially controlled settings so that we can measure spurious reliance and distribution shift directly. Extending STL to natural preference data will therefore require transformations that preserve utility, a principled choice of $k ,$ and scoring that scales to large pools. Integrating these components into a complete training pipeline remains open. Appendix D discusses these limitations in more detail.

## REFERENCES

Martin Arjovsky, Leon Bottou, Ishaan Gulrajani, and David Lopez-Paz. Invariant risk minimization.´ arXiv preprint arXiv:1907.02893, 2019.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Simone Bombari and Marco Mondelli. Spurious correlations in high dimensional regression: The roles of regularization, simplicity bias and over-parameterization. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025, volume 267 of Proceedings of Machine Learning Research, pp. 4839–4873. PMLR / OpenReview.net, 2025.

Stephen Casper, Xander Davies, Claudia Shi, Thomas Krendl Gilbert, Jer´ emy Scheurer, Javier ´ Rando, Rachel Freedman, Tomasz Korbak, David Lindner, Pedro Freire, Tony Tong Wang, Samuel Marks, Charbel-Raphael S ¨ egerie, Micah Carroll, Andi Peng, Phillip J. K. Christoffersen,´ Mehul Damani, Stewart Slocum, Usman Anwar, Anand Siththaranjan, Max Nadeau, Eric J. Michaud, Jacob Pfau, Dmitrii Krasheninnikov, Xin Chen, Lauro Langosco, Peter Hase, Erdem Bıyık, Anca D. Dragan, David Krueger, Dorsa Sadigh, and Dylan Hadfield-Menell. Open problems and fundamental limitations of reinforcement learning from human feedback. Trans. Mach. Learn. Res., 2023, 2023.

Lichang Chen, Chen Zhu, Jiuhai Chen, Davit Soselia, Tianyi Zhou, Tom Goldstein, Heng Huang, Mohammad Shoeybi, and Bryan Catanzaro. ODIN: disentangled reward mitigates hacking in RLHF. In Ruslan Salakhutdinov, Zico Kolter, Katherine A. Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, volume 235 of Proceedings ofMachine Learning Research, pp. 7935–7952. PMLR / OpenReview.net, 2024.

Paul F. Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. In Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett (eds.), Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pp. 4299–4307, 2017.

Nirjhar Das, Souradip Chakraborty, Aldo Pacchiano, and Sayak Ray Chowdhury. Active preference optimization for sample efficient RLHF. In Rita P. Ribeiro, Bernhard Pfahringer, Nathalie Japkowicz, Pedro Larranaga, Al˜ ´ıpio M. Jorge, Carlos Soares, Pedro H. Abreu, and Joao Gama˜ (eds.), Machine Learning and Knowledge Discovery in Databases. Research Track - European Conference, ECML PKDD 2025, Porto, Portugal, September 15-19, 2025, Proceedings, Part V, volume 16017 of Lecture Notes in Computer Science, pp. 96–112. Springer, 2025.

Yann Dubois, Chen Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Alpacafarm: A simulation framework for methods that learn from human feedback. In Alice Oh, Tristan Naumann, Amir Globerson, Kate

Saenko, Moritz Hardt, and Sergey Levine (eds.), Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, pp. 30039–30069, 2023.

Chengqian Gao, Haonan Li, Liu Liu, Zeke Xie, Peilin Zhao, and Zhiqiang Xu. Principled data selection for alignment: The hidden risks of difficult examples. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025, volume 267 of Proceedings of Machine Learning Research, pp. 18386–18409. PMLR / OpenReview.net, 2025.

Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pp. 10835–10866. PMLR, 2023.

Jiaming Ji, Donghai Hong, Borong Zhang, Boyuan Chen, Josef Dai, Boren Zheng, Tianyi Alex Qiu, Jiayi Zhou, Kaile Wang, Boxun Li, Sirui Han, Yike Guo, and Yaodong Yang. PKU-SafeRLHF: Towards multi-level safety alignment for LLMs with human preference. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 31983–32016, Vienna, Austria, 2025. Association for Computational Linguistics.

Hyeonji Kim, Sujeong Oh, and Sanghack Lee. Mitigating length bias in RLHF through a causal lens. CoRR, abs/2511.12573, 2025.

Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. In Doina Precup and Yee Whye Teh (eds.), Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017, volume 70 of Proceedings ofMachine Learning Research, pp. 1885–1894. PMLR, 2017.

Xiaoqiang Lin, Arun Verma, Zhongxiang Dai, Daniela Rus, See-Kiong Ng, and Bryan Kian Hsiang Low. ActiveDPO: Active direct preference optimization for sample-efficient alignment. In The Fourteenth International Conference on Learning Representations, 2026.

Tianqi Liu, Wei Xiong, Jie Ren, Lichang Chen, Junru Wu, Rishabh Joshi, Yang Gao, Jiaming Shen, Zhen Qin, Tianhe Yu, Daniel Sohn, Anastasia Makarova, Jeremiah Zhe Liu, Yuan Liu, Bilal Piot, Abe Ittycheriah, Aviral Kumar, and Mohammad Saleh. RRM: robust reward model training mitigates reward hacking. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025.

Junru Lu, Jiazheng Li, Siyu An, Meng Zhao, Yulan He, Di Yin, and Xing Sun. Eliminating biased length reliance of direct preference optimization via down-sampled KL divergence. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP 2024, Miami, FL, USA, November 12-16, 2024, pp. 1047–1067. Association for Computational Linguistics, 2024.

James Martens. New insights and perspectives on the natural gradient method. Journal ofMachine Learning Research, 21(146):1–76, 2020.

Luckeciano Carvalho Melo, Panagiotis Tigas, Alessandro Abate, and Yarin Gal. Deep bayesian active learning for preference modeling in large language models. In Amir Globerson, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang (eds.), Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, pp. 118052–118085, 2024.

Christian Moya, Alex Semendinger, Guang Lin, and Elliott Thornley. Spurious correlation learning in preference optimization: Mechanisms, consequences, and mitigation via tie training. In Proceedings of the 43rd International Conference on Machine Learning. PMLR, 2026.

William Muldrew, Peter Hayes, Mingtian Zhang, and David Barber. Active preference learning for large language models. In Ruslan Salakhutdinov, Zico Kolter, Katherine A. Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, volume 235 of Proceedings ofMachine Learning Research, pp. 36577–36590. PMLR / OpenReview.net, 2024.

Vaishnavh Nagarajan, Anders Andreassen, and Behnam Neyshabur. Understanding the failure modes of out-of-distribution generalization. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems 35, pp. 27730–27744, 2022.

Ryan Park, Rafael Rafailov, Stefano Ermon, and Chelsea Finn. Disentangling length from quality in direct preference optimization. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024, volume ACL 2024 of Findings of ACL, pp. 4998–5017. Association for Computational Linguistics, 2024.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.), Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, pp. 53728–53741, 2023.

Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Matt Turner. Steering llama 2 via contrastive activation addition. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pp. 15504–15522. Association for Computational Linguistics, 2024.

Keita Saito, Akifumi Wachi, Koki Wataoka, and Youhei Akimoto. Verbosity bias in preference labeling by large language models. arXiv preprint arXiv:2310.10076, 2023.

Nicol N. Schraudolph. Fast curvature matrix-vector products for second-order gradient descent. Neural Computation, 14(7):1723–1738, 2002.

Mrinank Sharma, Meg Tong, Tomasz Korbak, David Duvenaud, Amanda Askell, Samuel R. Bowman, Esin Durmus, Zac Hatfield-Dodds, Scott R. Johnston, Shauna Kravec, Timothy Maxwell, Sam McCandlish, Kamal Ndousse, Oliver Rausch, Nicholas Schiefer, Da Yan, Miranda Zhang, and Ethan Perez. Towards understanding sycophancy in language models. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024.

Prasann Singhal, Tanya Goyal, Jiacheng Xu, and Greg Durrett. A long way to go: Investigating length correlations in RLHF. CoRR, abs/2310.03716, 2023.

Victor Veitch, Alexander D’Amour, Steve Yadlowsky, and Jacob Eisenstein. Counterfactual invari ance to spurious correlations in text classification. In Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan (eds.), Advances in Neural In formation Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pp. 16196–16208, 2021.

George Wang, Elizabeth Donoway, and Daniel Murfet. Patterning in practice: Debiasing reward models with susceptibilities. arXiv preprint arXiv:2609.00699, 2026.

## A PROOFS

## A.1 PROOFS FOR THE INFLUENCE ANALYSIS

## A.1.1 TIE GRADIENT

Claim: The gradient of the tie loss averaged over its two orientations is

$$
g _ { j , c } ( \theta ) = 0 , \qquad g _ { j , s } ( \theta ) = \frac { \beta } { 2 } \operatorname { t a n h } \left( \frac { \beta \tilde { \theta } _ { s } ^ { \top } d _ { j } } { 2 } \right) d _ { j } .
$$

Proof: The two orientations have feature differences $[ 0 ; d _ { j } ]$ and $[ 0 ; - d _ { j } ]$ . Averaging their losses gives

$$
\ell _ { j } ( \theta ) = - \frac { 1 } { 2 } \log \sigma ( \beta \tilde { \theta } _ { s } ^ { \top } d _ { j } ) - \frac { 1 } { 2 } \log \sigma ( - \beta \tilde { \theta } _ { s } ^ { \top } d _ { j } ) .
$$

The resulting loss does not depend on the core parameters, so its core gradient is zero. Differentiating with respect to the spurious parameters gives

$$
\begin{array} { l } { { g _ { j , s } ( \theta ) = \displaystyle \frac { \beta } { 2 } \left[ \sigma ( \beta \widetilde { \theta } _ { s } ^ { \top } d _ { j } ) - \sigma ( - \beta \widetilde { \theta } _ { s } ^ { \top } d _ { j } ) \right] d _ { j } } } \\ { { \mathrm { ~ } = \displaystyle \frac { \beta } { 2 } \operatorname { t a n h } \left( \frac { \beta \widetilde { \theta } _ { s } ^ { \top } d _ { j } } { 2 } \right) d _ { j } . } } \end{array}
$$

For the second equality, use $\sigma ( z ) - \sigma ( - z ) = \operatorname { t a n h } ( z / 2 )$ . Reversing the orientation replaces $d _ { j }$ by $- d _ { j }$ . This changes the signs of both the hyperbolic tangent and the feature difference, so their product is unchanged. □

## A.1.2 PARAMETER RESPONSE

Partition the Hessian of the strict objective into its core and spurious blocks. Let $K _ { s }$ denote the spurious Schur complement of $H _ { c c }$ in H:

$$
H = \left[ \begin{array} { l l } { H _ { c c } } & { H _ { c s } } \\ { H _ { s c } } & { H _ { s s } } \end{array} \right] , \qquad K _ { s } = H _ { s s } - H _ { s c } H _ { c c } ^ { - 1 } H _ { c s } .
$$

Claim: Under the stationarity condition of Proposition 1, the perturbed objective of Section 3.1 has a unique nearby minimizer for sufficiently small $\epsilon \geq 0$ , and

$$
\theta _ { j } ^ { \epsilon } - \theta _ { 0 } = - \frac { \epsilon } { N } H ^ { - 1 } g _ { j } ( \theta _ { 0 } ) + O ( \epsilon ^ { 2 } ) , \qquad \theta _ { j , s } ^ { \epsilon } - \theta _ { 0 , s } = - \frac { \epsilon } { N } K _ { s } ^ { - 1 } g _ { j , s } ( \theta _ { 0 } ) + O ( \epsilon ^ { 2 } ) .\tag{2}
$$

Proof: The perturbed minimizer solves the stationarity equation

$$
\nabla \mathcal { L } _ { \mathrm { D P O } } ( \theta ) + \frac { \epsilon } { N } \nabla \ell _ { j } ( \theta ) = 0
$$

at $\theta = \theta _ { i } ^ { \epsilon } . \mathrm { A t } \epsilon = 0$ , the solution is $\theta _ { j } ^ { 0 } = \theta _ { 0 }$ . Differentiating the left side with respect to θ at $\theta = \theta _ { 0 }$ and $\epsilon = 0$ gives H. Because the losses are smooth and the positive definite matrix H is invertible, the implicit function theorem gives a unique differentiable solution $\theta _ { j } ^ { \epsilon }$ near $\theta _ { 0 }$ for sufficiently small ϵ. For $\epsilon \geq 0$ , both terms in the perturbed log-linear objective are convex, so this stationary point is a minimizer.

Differentiating the stationarity equation with respect to ϵ gives

$$
\left[ \nabla ^ { 2 } \mathcal { L } _ { \mathrm { D P O } } ( \theta _ { j } ^ { \epsilon } ) + \frac { \epsilon } { N } \nabla ^ { 2 } \ell _ { j } ( \theta _ { j } ^ { \epsilon } ) \right] \frac { d \theta _ { j } ^ { \epsilon } } { d \epsilon } + \frac { 1 } { N } \nabla \ell _ { j } ( \theta _ { j } ^ { \epsilon } ) = 0 .
$$

$\mathrm { A t } \epsilon = 0 , \theta _ { j } ^ { 0 } = \theta _ { 0 }$ and $\nabla ^ { 2 } \mathcal { L } _ { \mathrm { D P O } } ( \theta _ { 0 } ) = H ;$ ; the term multiplied by ϵ vanishes. Thus

$$
H \left. \frac { d \theta _ { j } ^ { \epsilon } } { d \epsilon } \right| _ { \epsilon = 0 } + \frac { 1 } { N } g _ { j } ( \theta _ { 0 } ) = 0 .
$$

Solving for the derivative gives

$$
\left. \frac { d \theta _ { j } ^ { \epsilon } } { d \epsilon } \right| _ { \epsilon = 0 } = - \frac { 1 } { N } H ^ { - 1 } g _ { j } ( \theta _ { 0 } ) .
$$

The Taylor expansion of $\theta _ { j } ^ { \epsilon }$ around $\epsilon = 0$ is

$$
\theta _ { j } ^ { \epsilon } = \theta _ { j } ^ { 0 } + \epsilon \left. \frac { d \theta _ { j } ^ { \epsilon } } { d \epsilon } \right| _ { \epsilon = 0 } + O ( \epsilon ^ { 2 } ) = \theta _ { 0 } - \frac { \epsilon } { N } H ^ { - 1 } g _ { j } ( \theta _ { 0 } ) + O ( \epsilon ^ { 2 } ) ,
$$

which is the first identity in (2).

The preceding claim gives $g _ { j , c } ( \theta _ { 0 } ) = 0$ . To recover the spurious part of the first identity, write the equation for dθ<sup>ϵ</sup>/dϵ in block form:

$$
\left[ H _ { c c } \left. H _ { c s } \right] \left[ \frac { d \theta _ { j , c } ^ { \epsilon } } { d \epsilon } \right| _ { \epsilon = 0 } \right] = - \frac { 1 } { N } \left[ g _ { j , s } ( \theta _ { 0 } ) \right] .
$$

Because $H \succ 0 ,$ , both $H _ { c c }$ and $K _ { s }$ are positive definite and hence invertible. The first row of the system gives

$$
\left. \frac { d \theta _ { j , c } ^ { \epsilon } } { d \epsilon } \right| _ { \epsilon = 0 } = - H _ { c c } ^ { - 1 } H _ { c s } \left. \frac { d \theta _ { j , s } ^ { \epsilon } } { d \epsilon } \right| _ { \epsilon = 0 } .
$$

Substituting this expression into the second row gives

$$
\left( \boldsymbol { H _ { s s } } - \boldsymbol { H _ { s c } } \boldsymbol { H _ { c c } ^ { - 1 } } \boldsymbol { H _ { c s } } \right) \left. \frac { d \theta _ { j , s } ^ { \epsilon } } { d \epsilon } \right| _ { \epsilon = 0 } = - \frac { 1 } { N } g _ { j , s } ( \theta _ { 0 } ) ,
$$

or equivalently,

$$
K _ { s } \left. \frac { d \theta _ { j , s } ^ { \epsilon } } { d \epsilon } \right| _ { \epsilon = 0 } = - \frac { 1 } { N } g _ { j , s } ( \theta _ { 0 } ) .
$$

Thus

$$
\left. \frac { d \theta _ { j , s } ^ { \epsilon } } { d \epsilon } \right| _ { \epsilon = 0 } = - \frac { 1 } { N } K _ { s } ^ { - 1 } g _ { j , s } ( \theta _ { 0 } ) .
$$

Taylor expanding the spurious parameters around $\epsilon = 0$ now gives

$$
\theta _ { j , s } ^ { \epsilon } = \theta _ { 0 , s } + \epsilon \left. \frac { d \theta _ { j , s } ^ { \epsilon } } { d \epsilon } \right| _ { \epsilon = 0 } + O ( \epsilon ^ { 2 } ) = \theta _ { 0 , s } - \frac { \epsilon } { N } K _ { s } ^ { - 1 } g _ { j , s } ( \theta _ { 0 } ) + O ( \epsilon ^ { 2 } ) ,
$$

which is the second identity in (2).

## A.1.3 INFLUENCE ON SPURIOUS ENERGY

Claim: Under the conditions of Proposition 1,

$$
E ( \theta _ { j } ^ { \epsilon } ) - E ( \theta _ { 0 } ) = - \epsilon A _ { j } + O ( \epsilon ^ { 2 } ) , \qquad A _ { j } = \frac { 1 } { N } \bigl \langle \nabla _ { s } E ( \theta _ { 0 } ) , K _ { s } ^ { - 1 } g _ { j , s } ( \theta _ { 0 } ) \bigl \rangle .
$$

Equivalently,

$$
A _ { j } = \frac { \beta } { 2 N } \operatorname { t a n h } \left( \frac { \beta \tilde { \theta } _ { 0 , s } ^ { \top } d _ { j } } { 2 } \right) \big \langle \nabla _ { s } E ( \theta _ { 0 } ) , K _ { s } ^ { - 1 } d _ { j } \big \rangle .
$$

Proof: The gradient of the spurious energy is

$$
\nabla E ( \theta _ { 0 } ) = 2 \left[ \sum _ { s s } \tilde { \theta } _ { 0 , s } \right] .
$$

This gradient has no core component. Combining the spurious response in (2) with a Taylor expansion of $E$ gives

$$
\begin{array} { l } { { E ( \theta _ { j } ^ { \epsilon } ) - E ( \theta _ { 0 } ) = \left. \nabla _ { s } E ( \theta _ { 0 } ) , \theta _ { j , s } ^ { \epsilon } - \theta _ { 0 , s } \right. + O ( \epsilon ^ { 2 } ) } } \\ { { \displaystyle \quad = - \frac { \epsilon } { N } \big \langle \nabla _ { s } E ( \theta _ { 0 } ) , K _ { s } ^ { - 1 } g _ { j , s } ( \theta _ { 0 } ) \big \rangle + O ( \epsilon ^ { 2 } ) } } \\ { { \displaystyle \quad = - \epsilon A _ { j } + O ( \epsilon ^ { 2 } ) . } } \end{array}
$$

Substituting the tie gradient proves the second expression for $A _ { j }$

## A.1.4 FIRST-ORDER OPTIMAL SELECTION

Claim: For any fixed set S of k ties, let $\theta _ { S } ^ { \epsilon }$ be the nearby minimizer of

$$
\mathcal { L } _ { \mathrm { D P O } } ( \theta ) + \frac { \epsilon } { N } \sum _ { j \in S } \ell _ { j } ( \theta ) .
$$

Then

$$
\theta _ { S } ^ { \epsilon } - \theta _ { 0 } = - \frac { \epsilon } { N } H ^ { - 1 } \sum _ { j \in S } g _ { j } ( \theta _ { 0 } ) + O ( \epsilon ^ { 2 } ) ,
$$

$$
\theta _ { S , s } ^ { \epsilon } - \theta _ { 0 , s } = - \frac { \epsilon } { N } K _ { s } ^ { - 1 } \sum _ { j \in S } g _ { j , s } ( \theta _ { 0 } ) + O ( \epsilon ^ { 2 } ) ,
$$

and therefore

$$
E ( \theta _ { S } ^ { \epsilon } ) = E ( \theta _ { 0 } ) - \epsilon \sum _ { j \in S } A _ { j } + O ( \epsilon ^ { 2 } ) .
$$

Let $S _ { k }$ contain the k largest scores $A _ { j }$ , and let $S _ { \mathrm { U T L } }$ be a set of k ties sampled uniformly. Then

$$
\sum _ { j \in S _ { k } } A _ { j } = \operatorname* { m a x } _ { | S | = k } \sum _ { j \in S } A _ { j } \geq \mathbb { E } \left[ \sum _ { j \in S _ { \mathrm { U T L } } } A _ { j } \right] = \frac { k } { M } \sum _ { j = 1 } ^ { M } A _ { j } .
$$

For $0 < k < M$ , the inequality is strict unless all scores are equal; when $k = M$ , the selected sets coincide. The expectation is over the subset sampled by UTL.

Proof:

Additivity. The perturbed minimizer solves the stationarity equation

$$
\nabla \mathcal { L } _ { \mathrm { D P O } } ( \theta ) + \frac { \epsilon } { N } \sum _ { j \in S } \nabla \ell _ { j } ( \theta ) = 0
$$

at $\theta = \theta _ { S } ^ { \epsilon } . \mathrm { A t } \epsilon = 0$ , the solution is $\theta _ { S } ^ { 0 } = \theta _ { 0 }$ . Differentiating the left side with respect to θ at $\theta = \theta _ { 0 }$ and $\epsilon = 0$ gives $H$ . Because $\mathcal { L } _ { \mathrm { D P O } }$ and each $\ell _ { j }$ are smooth and $H$ is invertible, the implicit function theorem gives a unique differentiable stationary point near $\theta _ { 0 }$ for sufficiently small ϵ. For $\epsilon \geq 0$ , the perturbed objective is convex, so this stationary point is a minimizer. Differentiating the stationarity equation with respect to ϵ gives

$$
\left[ \nabla ^ { 2 } \mathcal { L } _ { \mathrm { D P O } } ( \theta _ { S } ^ { \epsilon } ) + \frac { \epsilon } { N } \sum _ { j \in S } \nabla ^ { 2 } \ell _ { j } ( \theta _ { S } ^ { \epsilon } ) \right] \frac { d \theta _ { S } ^ { \epsilon } } { d \epsilon } + \frac { 1 } { N } \sum _ { j \in S } \nabla \ell _ { j } ( \theta _ { S } ^ { \epsilon } ) = 0 .
$$

Evaluating this equation at zero gives

$$
H \left. \frac { d \theta _ { S } ^ { \epsilon } } { d \epsilon } \right| _ { \epsilon = 0 } + \frac { 1 } { N } \sum _ { j \epsilon S } g _ { j } ( \theta _ { 0 } ) = 0 ,
$$

and hence

$$
\left. \frac { d \theta _ { S } ^ { \epsilon } } { d \epsilon } \right| _ { \epsilon = 0 } = - \frac { 1 } { N } H ^ { - 1 } \sum _ { j \in S } g _ { j } ( \theta _ { 0 } ) .
$$

The Taylor expansion around $\epsilon = 0$ gives

$$
\theta _ { S } ^ { \epsilon } = \theta _ { S } ^ { 0 } + \epsilon \left. \frac { d \theta _ { S } ^ { \epsilon } } { d \epsilon } \right| _ { \epsilon = 0 } + O ( \epsilon ^ { 2 } ) = \theta _ { 0 } - \frac { \epsilon } { N } H ^ { - 1 } \sum _ { j \in S } g _ { j } ( \theta _ { 0 } ) + O ( \epsilon ^ { 2 } ) .
$$

Every tie gradient has a zero core block, so their sum does as well. The differentiated equation therefore has the block form

$$
[ H _ { s c }  H _ { c s } ] [ \frac { d \theta _ { S , c } ^ { \epsilon } } { d \epsilon } | _ { \epsilon = 0 } = - \frac { 1 } { N } [ \sum _ { j \in S } g _ { j , s } ( \theta _ { 0 } ) ] .
$$

The first row gives

$$
\left. \frac { d \theta _ { S , c } ^ { \epsilon } } { d \epsilon } \right| _ { \epsilon = 0 } = - H _ { c c } ^ { - 1 } H _ { c s } \left. \frac { d \theta _ { S , s } ^ { \epsilon } } { d \epsilon } \right| _ { \epsilon = 0 } .
$$

Substituting this expression into the second row gives

$$
K _ { s } \left. \frac { d \theta _ { S , s } ^ { \epsilon } } { d \epsilon } \right| _ { \epsilon = 0 } = - \frac { 1 } { N } \sum _ { j \in S } g _ { j , s } ( \theta _ { 0 } ) .
$$

The Taylor expansion of the spurious parameters is therefore

$$
\theta _ { S , s } ^ { \epsilon } = \theta _ { 0 , s } + \epsilon \left. \frac { d \theta _ { S , s } ^ { \epsilon } } { d \epsilon } \right| _ { \epsilon = 0 } + O ( \epsilon ^ { 2 } ) = \theta _ { 0 , s } - \left. \frac { \epsilon } { N } K _ { s } ^ { - 1 } \sum _ { j \in S } g _ { j , s } ( \theta _ { 0 } ) + O ( \epsilon ^ { 2 } ) . \right.
$$

Finally, expanding the spurious energy along this response gives

$$
\begin{array} { l } { { \displaystyle { E ( \theta _ { S } ^ { \epsilon } ) - E ( \theta _ { 0 } ) = - \frac { \epsilon } { N } \left. \nabla _ { s } E ( \theta _ { 0 } ) , K _ { s } ^ { - 1 } \sum _ { j \in S } g _ { j , s } ( \theta _ { 0 } ) \right. + O ( \epsilon ^ { 2 } ) } } } \\ { { \displaystyle { \phantom { \frac { E ( \theta _ { S } ^ { \epsilon } ) - E ( \theta _ { 0 } ) } { j \in S } } } } } \\ { { \displaystyle { \phantom { \frac { E ( \theta _ { 0 } ^ { \epsilon } ) - E ( \theta _ { 0 } ) } { j \in S } } } } } \end{array}
$$

By the definition of $A _ { j }$ , the second equality shows that the influences of the individual ties add at first order.

Comparison with UTL. For any other set S with $| S | = k$ , the sets $S _ { k } \setminus S$ and $S \setminus S _ { k }$ have the same size. Pair their elements arbitrarily as $( i _ { r } , j _ { r } )$ , where $i _ { r } \in S _ { k } \setminus S$ and $j _ { r } \in S \setminus S _ { k }$ . Since $S _ { k }$ contains the largest scores, $A _ { i _ { r } } \geq A _ { j }$ for every pair, and hence

$$
\sum _ { j \in S _ { k } } A _ { j } - \sum _ { j \in S } A _ { j } = \sum _ { r } ( A _ { i _ { r } } - A _ { j _ { r } } ) \geq 0 .
$$

This inequality holds for every set of k ties, including every realization of $S _ { \mathrm { U T L } }$ . Taking the expectation over UTL therefore gives

$$
\sum _ { j \in S _ { k } } A _ { j } \geq \operatorname { \mathbb { E } } \left[ \sum _ { j \in S _ { \mathrm { U T L } } } A _ { j } \right] .
$$

Each tie belongs to $S _ { \mathrm { U T L } }$ with probability $k / M ,$ , so

$$
\mathbb { E } \left[ \sum _ { j \in S _ { \mathrm { U T L } } } A _ { j } \right] = \sum _ { j = 1 } ^ { M } A _ { j } \operatorname* { P r } ( j \in S _ { \mathrm { U T L } } ) = \frac { k } { M } \sum _ { j = 1 } ^ { M } A _ { j } .
$$

If $0 < k < M$ and the scores are not all equal, at least one set of k ties has a smaller total than $S _ { k }$ Every such set has positive probability under UTL, so the inequality is strict. When all scores are equal, or when $k = M$ , equality holds. □

## A.2 RANKING TIES FROM MARGIN INFORMATION

Claim: Under the conditions of Theorem 3, there is a constant $\lambda > 0$ , common to all ties, such that

$$
\mathbb { E } [ A _ { j } \mid m _ { j } ] = \frac { \lambda } { N } m _ { j } \operatorname { t a n h } ( m _ { j } / 2 ) .
$$

This conditional mean depends on $m _ { j }$ only through $| m _ { j } |$ and strictly increases with $| m _ { j } |$

Proof: For brevity, write $w = \tilde { \theta } _ { 0 , s } . \mathrm { ~ H ~ } w = 0$ , every tie margin and influence is zero, so the claim is immediate. We therefore consider w $\neq 0$

Let $d _ { j }$ be the spurious contrast of the sampled orientation of tie $j ,$ so that

$$
m _ { j } = \beta w ^ { \top } d _ { j } .
$$

Reversing the orientation changes the signs of both $m _ { j }$ and $d _ { j }$ . Their signs cancel in the influence, so either orientation gives

$$
A _ { j } = \frac { \beta } { N } \operatorname { t a n h } ( m _ { j } / 2 ) \big \langle \Sigma _ { s s } w , K _ { s } ^ { - 1 } d _ { j } \big \rangle .
$$

By the assumptions of Theorem $3 , d _ { j } \sim \mathcal { N } ( 0 , C )$ with $C = c \Sigma _ { s s }$ . Because $m _ { j } = \beta w ^ { \top } d _ { j }$

$$
\begin{array} { r } { \mathrm { C o v } ( d _ { j } , m _ { j } ) = \beta C w , \qquad \mathrm { V a r } ( m _ { j } ) = \beta ^ { 2 } w ^ { \top } C w . } \end{array}
$$

The conditional mean of a jointly Gaussian vector is therefore

$$
\mathbb { E } [ d _ { j } \mid m _ { j } ] = \frac { C w } { w ^ { \top } C w } \frac { m _ { j } } { \beta } .
$$

Inserting this conditional mean into the influence expression gives

$$
\mathbb { E } [ A _ { j } \ | \ m _ { j } ] = \frac { m _ { j } \operatorname { t a n h } ( m _ { j } / 2 ) } { N } \frac { w ^ { \top } \Sigma _ { s s } K _ { s } ^ { - 1 } C w } { w ^ { \top } C w } .
$$

Substituting $C = c \Sigma _ { s s }$ cancels the scale $c .$ Define

$$
\lambda = \frac { w ^ { \top } \Sigma _ { s s } K _ { s } ^ { - 1 } \Sigma _ { s s } w } { w ^ { \top } \Sigma _ { s s } w } > 0 .\tag{3}
$$

Since $H \succ 0$ , its Schur complement $K _ { s }$ is positive definite, and so is $K _ { s } ^ { - 1 }$ . Moreover, $\Sigma _ { s s } \succ 0$ and w $\neq 0$ imply $\Sigma _ { s s } w \neq 0$ . Therefore, the numerator $( \Sigma _ { s s } w ) ^ { \top } K _ { s } ^ { - 1 } ( \Sigma _ { s s } w )$ and the denominator $w ^ { \top } \Sigma _ { s s } \dot { w }$ are both positive.

The function m tanh $( m / 2 )$ is even, so the conditional mean depends on $m _ { j }$ only through $| m _ { j } |$ . For $a > 0$

$$
\frac { d } { d a } \left[ a \operatorname { t a n h } ( a / 2 ) \right] = \operatorname { t a n h } ( a / 2 ) + \frac { a } { 2 } \operatorname { s e c h } ^ { 2 } ( a / 2 ) > 0 .
$$

The function equals zero at $a = 0$ , which proves strict monotonicity on $[ 0 , \infty )$

Claim: For any budget $1 \leq k \leq M$ , conditional on the observed margins, pure margin selection maximizes expected total influence among selectors that use only those margins.

Proof: Let

$$
\pmb { m } = ( m _ { 1 } , \dots , m _ { M } ) .
$$

Given $\theta _ { 0 } ,$ , the contrasts are independent across ties, so the pair $( A _ { j } , m _ { j } )$ is independent of the margins of all other ties. Once the margin of tie $j$ is known, observing the other margins therefore does not change its expected influence:

$$
\mathbb { E } [ A _ { j } \mid m ] = \mathbb { E } [ A _ { j } \mid m _ { j } ] .
$$

Now consider any rule $S = S ( m )$ that selects k ties. Linearity of conditional expectation gives

$$
\mathbb { E } [ \sum _ { j \in S } A _ { j } | m ] = \frac { \lambda } { N } \sum _ { j \in S } \lvert m _ { j } \rvert \operatorname { t a n h } ( \lvert m _ { j } \rvert / 2 ) .
$$

Because a tanh $. ( a / 2 )$ is strictly increasing for $a \geq 0$ , selecting the k largest $| m _ { j } |$ maximizes this conditional expectation. □

## A.3 RECOVERING THE SPURIOUS-ENERGY GRADIENT

Fix either orientation of each probe tie and let $d _ { n }$ be its spurious feature difference. Because the probe ties preserve core features, their core feature differences are zero. The empirical second moment of these differences is

$$
\widehat { C } = \frac { 1 } { N _ { v } } \sum _ { n = 1 } ^ { N _ { v } } d _ { n } d _ { n } ^ { \top } .
$$

Claim: In the log-linear setting,

$$
E _ { \mathrm { c f } } ( \theta ) = \beta ^ { 2 } \widetilde { \theta } _ { s } ^ { \top } \widehat { C } \widetilde { \theta } _ { s } , \qquad \nabla E _ { \mathrm { c f } } ( \theta _ { 0 } ) = 2 \beta ^ { 2 } \left[ \widehat { C } \widetilde { \tilde { \theta } } _ { 0 , s } \right] .\tag{4}
$$

If $\widehat { C } = c \Sigma _ { s s }$ for some $c > 0$ , then

$$
\nabla E _ { \mathrm { c f } } ( \theta _ { 0 } ) = c \beta ^ { 2 } \nabla E ( \theta _ { 0 } ) .
$$

Proof: For tie n in the probe, the reward difference in the log-linear model is $\beta \tilde { \theta } _ { s } ^ { \top } d _ { n }$ . Its sign depends on the chosen orientation, but its square does not. Using these squared differences in (1) gives

$$
E _ { \mathrm { c f } } ( \theta ) = \frac { \beta ^ { 2 } } { N _ { v } } \sum _ { n = 1 } ^ { N _ { v } } ( \widetilde { \theta } _ { s } ^ { \top } d _ { n } ) ^ { 2 } = \beta ^ { 2 } \widetilde { \theta } _ { s } ^ { \top } \widehat { C } \widetilde { \theta } _ { s } .
$$

Every outer product $d _ { n } d _ { n } ^ { \top }$ is symmetric, and hence so is their average ${ \widehat { C } } .$ Differentiating the quadratic form therefore gives the gradient in (4). The gradient of the spurious energy E is

$$
\nabla E ( \theta _ { 0 } ) = 2 \left[ \underset { s s } { 0 } \tilde { { \boldsymbol { \theta } } } _ { 0 , s } \right] .
$$

When $\widehat { C } = c \Sigma _ { s s } .$ , direct substitution gives $\nabla E _ { \mathrm { c f } } ( \theta _ { 0 } ) = c \beta ^ { 2 } \nabla E ( \theta _ { 0 } )$ . Under the assumptions of Section 4.2, $\mathbb { E } [ d _ { n } d _ { n } ^ { \top } ] = c \Sigma _ { s s } , \ s _ { 0 } \ \mathbb { E } [ \widehat { C } ] = c \Sigma _ { s s }$ . Because the gradient in (4) is linear in ${ \widehat { C } } ,$ the identity then holds in expectation over the probe. □

## B IMPLEMENTATION DETAILS

## B.1 CURVATURE AND NUMERICAL SOLVES

Each influence score uses the inverse curvature of the DPO loss at $\theta _ { 0 }$ . Because the gradients of the energy and each tie have zero core blocks, and the spurious block of $H ^ { - 1 }$ is $K _ { s } ^ { - 1 }$ , the influence in Proposition 1 can be written over all parameters:

$$
A _ { j } = \frac { 1 } { N } \big \langle \nabla E ( \theta _ { 0 } ) , H ^ { - 1 } g _ { j } ( \theta _ { 0 } ) \big \rangle = \frac { 1 } { N } \big \langle v , g _ { j } ( \theta _ { 0 } ) \big \rangle , \qquad H v = \nabla E ( \theta _ { 0 } ) .\tag{5}
$$

The second form follows from the symmetry of $H .$ Thus, one linear solve scores the entire pool, after which each tie requires only its gradient. The log-linear model can form and solve this system exactly. A neural policy has a Hessian of the same form over all p parameters. Two obstacles prevent an exact solve: the DPO loss is nonconvex in the network parameters, and $\theta _ { 0 }$ was obtained after finite training, so H need not be positive definite. For the language model, $p$ is also in the millions, so H cannot be stored or factorized. We therefore solve the system exactly in the log-linear model and, in the neural regimes, replace H with a damped approximation that is positive definite and cheap to solve.

Log-linear model. The log-linear model uses the exact Hessian of the empirical DPO objective at $\theta _ { 0 }$ , formed from all N comparisons in $\mathcal { D } _ { \mathrm { p r e f } } .$ . For strict comparison i, let $\Delta \phi _ { i } = \phi ( x _ { i } , y _ { w , i } ) -$ $\phi ( x _ { i } , y _ { l , i } )$ be its feature difference and $m _ { i } = \beta \tilde { \theta } _ { 0 } ^ { \top } \Delta \phi _ { i }$ its margin. With σ the logistic function, the Hessian is

$$
H = \frac { \beta ^ { 2 } } { N } \sum _ { i = 1 } ^ { N } \sigma ( m _ { i } ) \sigma ( - m _ { i } ) \Delta \phi _ { i } \Delta \phi _ { i } ^ { \top } .\tag{6}
$$

The right side of (5) is the analytic gradient of the population energy, $\nabla E ( \theta _ { 0 } ) = [ 0 ; 2 \Sigma _ { s s } \tilde { \theta } _ { 0 , s } ]$ . With only $\bar { d _ { c } } + d _ { s } = 2 0$ parameters, we solve $H v = \nabla \dot { E } ( \dot { \theta } _ { 0 } )$ ) exactly and without damping, so each score equals the influence $A _ { j }$

Neural network and language model. The two neural regimes use the same approximation to the curvature and the same solver. Both need the approximation because neither Hessian is guaranteed to be positive definite. Only the language model needs the solver, because its curvature is too large to form. The MLP, with about 19,000 parameters, could be solved directly, but sharing the solver keeps a single implementation for both regimes. For the MLP, the parameters are all weights in the reward network; for the language model, they are the trainable LoRA parameters. Let $m _ { i }$ be the DPO margin for strict comparison $i ;$ for the MLP, we take the reference margin to be zero. Let $J _ { i } = \nabla m _ { i } ( \theta _ { 0 } )$ , and let $\lambda _ { \mathrm { o b j } }$ be the weight of the objective’s L2 penalty $\frac { \lambda _ { \mathrm { o b j } } } { 2 } \lVert \boldsymbol { \theta } \rVert _ { 2 } ^ { 2 }$ : 0.01 for the MLP and 0 for LoRA-DPO. The Hessian of the strict objective is then

$$
H = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left[ \sigma ( m _ { i } ) \sigma ( - m _ { i } ) J _ { i } J _ { i } ^ { \top } - \sigma ( - m _ { i } ) \nabla ^ { 2 } m _ { i } ( \theta _ { 0 } ) \right] + \lambda _ { \mathrm { o b j } } I .\tag{7}
$$

The generalized Gauss–Newton (GGN) matrix retains the first term, which is positive semidefinite, and omits the second, which can be indefinite and requires the Hessian of every margin. We estimate the first term from $N _ { G } = 1 0 0 0$ strict comparisons sampled once per seed. Weighting each margin gradient and stacking the results as rows gives

$$
D _ { i , : } = \sqrt { \sigma ( m _ { i } ) \sigma ( - m _ { i } ) } J _ { i } ^ { \top } , \qquad G = \frac { 1 } { N _ { G } } D ^ { \top } D .\tag{8}
$$

The matrix $D$ has $N _ { G }$ rows, far fewer than the number of parameters $p ,$ so G has rank at most $N _ { G }$ and most of its eigenvalues are zero. We therefore damp G in proportion to its largest eigenvalue and add the objective’s own ridge:

$$
\widehat { H } = G + \alpha I , \qquad \alpha = \lambda _ { \mathrm { o b j } } + 0 . 0 5 \bigl ( \lambda _ { \mathrm { o b j } } + \lambda _ { \mathrm { m a x } } ( G ) \bigr ) .\tag{9}
$$

The curvature for the MLP thus contains its L2 penalty, and both regimes add the same relative damping.

With the counterfactual energy of Section 5.1 in place of $E ,$ the system for the neural models is $\widehat { H } v = \nabla E _ { \mathrm { c f } } ( \theta _ { 0 } )$ . The low rank of G also makes this system efficient to solve. The Woodbury identity

$$
\left( G + \alpha I \right) ^ { - 1 } = \alpha ^ { - 1 } \left[ I - D ^ { \top } \left( D D ^ { \top } + N _ { G } \alpha I \right) ^ { - 1 } D \right]\tag{10}
$$

replaces the $p \times p$ system with an $N _ { G } \times N _ { G }$ one:

$$
\begin{array} { r } { \xi = \bigl ( D D ^ { \top } + N _ { G } \alpha I \bigr ) ^ { - 1 } D \nabla E _ { \mathrm { c f } } ( \theta _ { 0 } ) , \qquad v = \alpha ^ { - 1 } \bigl ( \nabla E _ { \mathrm { c f } } ( \theta _ { 0 } ) - D ^ { \top } \xi \bigr ) . } \end{array}\tag{11}
$$

The solution is exact for the GGN after sampling and damping. The small matrix $D D ^ { \top } / N _ { G }$ also supplies $\lambda _ { \operatorname* { m a x } } ( G )$ : both it and $G$ are formed from the singular values of $D ,$ so they share their nonzero eigenvalues.

Numerical checks. Before solving the log-linear system, the implementation verifies the conditions of Proposition 1: the norm of the gradient of the strict objective must be at most $1 0 ^ { - 1 0 }$ , and H must be positive definite. Scoring stops if either check fails, rather than damping H. For the MLP, the solve runs in float64 and must reach a relative residual of $1 0 ^ { - 1 0 }$ . For the language model, $D$ is stored in bfloat16 to save memory, the $N _ { G } \times N _ { G }$ system is solved in float64, and the relative residual must be below $1 0 ^ { - 4 }$ . These checks verify that v solves the system defined by the sampled and damped GGN.

## C EXPERIMENTAL DETAILS AND ADDITIONAL RESULTS

This appendix describes the full experimental protocol and additional results for Section 6. We first describe the choices shared across experiments, then give the data, models, training procedures, and metrics for each regime.

## C.1 SHARED PROTOCOL

Tie pools. Every regime uses the tie construction of Section 2.3: each tie keeps the strict winner’s core content and changes only its spurious attributes. We construct one pool per seed and use it for every method, so the methods differ only in which ties they choose. Separate random streams generate the data, construct the ties, orient them, and drive training and evaluation, so a change in one step does not alter the random draws of another.

Selection. For each seed, we first train a strict checkpoint on the strict comparisons alone. PM and influence score every tie in the pool at this checkpoint, as derived in Sections 3–5: Oracle influence in the log-linear model and Practical influence in the neural regimes (Appendix B.1). Each targeted method selects the k highest-scoring ties from one fixed ranking, so by construction a tie selected at one budget remains selected at every larger budget. UTL instead samples k ties uniformly at random from the same pool.

Training. The strict checkpoint is used only for scoring. For each selected set, we train a new model on the strict comparisons and the selected ties, starting from the same initialization as the strict checkpoint. Training requires a labeled comparison, so we orient each tie once by a fair coin flip and use these orientations for every method and budget. Each selected tie then enters the objective as one additional comparison, with weight $1 / N$ relative to the mean loss on strict comparisons $( \epsilon = 1$ in Proposition 1). At a fixed seed and budget, the methods therefore differ only in which ties they select, and at $k = M$ they train on the same ties.

Uncertainty. We treat seeds as the independent units: each seed has its own data, tie pool, strict checkpoint, and scores. Given the seed, PM and influence select one set per budget, so each trains one model. UTL is random, so each seed trains several UTL models per budget, each on a separately sampled set of ties (a draw), and averages their metrics to reduce variance. Every method therefore contributes one value per seed and budget. Reported intervals are two-sided 95% Student-t confidence intervals across seeds.

Evaluation distributions. We evaluate on held-out comparisons from three distributions that differ only in how the spurious attributes relate to the preference. In the training distribution $P ,$ they correlate with the preference; in $Q _ { \mathrm { a d v } }$ , these correlations reverse, as in Section 6. In $Q _ { \mathrm { c l e a n } } .$ the spurious attributes keep their distribution but become independent of utility, so they carry no information about the winner. Each regime below specifies all three distributions.

Metrics. Accuracy under $Q _ { \mathrm { a d v } }$ measures performance when the training correlations reverse. It can also rise for the wrong reason. If the selected ties push the model to rely on the spurious attributes in the opposite direction, this reversed shortcut also predicts the winner under $Q _ { \mathrm { a d v } }$ . Under $Q _ { \mathrm { c l e a n } }$ , a shortcut in either direction only adds noise, so accuracy under $Q _ { \mathrm { c l e a n } }$ checks that selection removes the shortcut rather than reverses it. We also measure spurious reliance directly, with the spurious energy E in the log-linear model and the counterfactual energy $E _ { \mathrm { c f } }$ in the neural regimes. Both square the spurious part of the margin, so they grow with reliance in either direction. Finally, accuracy under P shows what mitigation costs in the training environment.

Fresh and continued training. Influence scores ties by their local effect at the strict checkpoint. In the log-linear model, the objective has a single minimizer, which training reaches from any starting point, so fresh training ends at the solution that the analysis describes. A neural objective ha no such guarantee, and fresh training can end far from the checkpoint at which the scores were computed. We also continue training from the strict checkpoint. Continued training uses the same settings and training budget as fresh training; only the initialization differs.

Table 1: Environments used to train and evaluate the synthetic models. The parameters a and t are defined in (14).
<table><tr><td>Environment</td><td>a</td><td>t</td><td>Interpretation</td></tr><tr><td> $P$ </td><td>+1</td><td>+1</td><td>Training environment; both shortcut channels align.</td></tr><tr><td> $Q _ { \mathrm { c l e a n } }$ </td><td>0</td><td>0</td><td>Both shortcut channels are removed.</td></tr><tr><td> $Q \mathrm { { a d v } }$ </td><td>-1</td><td>-1</td><td>Both shortcut channels reverse.</td></tr></table>

Reading the additional results. Appendix figures report means only to avoid clutter from overlap ping uncertainty bands. The results text reports representative 95% confidence intervals. In figures that compare training procedures, fresh and continued training occupy the left and right columns, with matching axes in each row. The Qwen figure instead compares hotels and emojis and reports only fresh training. Colors, markers, and line styles identify the same selector across columns.

## C.2 SYNTHETIC DATA

Latent variables. The log-linear experiments and the experiments with neural networks use the same latent distribution, which lets us reverse two sources of shortcut information separately. Each alternative has latent utility $q \sim \mathcal { N } ( 0 , 1 )$ , a core vector $c \in \mathbb { R } ^ { d _ { c } }$ , and a spurious vector $s \in \mathbb { R } ^ { d _ { s } }$ Let $z \sim \mathcal { N } ( 0 , 1 )$ be independent of $q ,$ and let $\left( \epsilon _ { c } , \epsilon _ { s } \right)$ be jointly Gaussian and independent of $( q , z )$ The matrices $\Omega _ { c }$ and $\Omega _ { s }$ are the core and spurious noise covariances, $R$ controls their dependence, and $u _ { c }$ and $u _ { s }$ are their signal directions. An environment is indexed by two parameters (a, t) and generates

$$
g = a q + { \sqrt { 1 - a ^ { 2 } } } z ,\tag{12}
$$

$$
c = u _ { c } q + \epsilon _ { c } ,
$$

$$
s = u _ { s } g + \epsilon _ { s } ,\tag{13}
$$

$$
\mathrm { C o v } \left( \left[ \epsilon _ { c } \right] \right) = \left( \begin{array} { c c } { \Omega _ { c } } & { t \Omega _ { c } ^ { 1 / 2 } R \Omega _ { s } ^ { 1 / 2 } } \\ { t \Omega _ { s } ^ { 1 / 2 } R ^ { \top } \Omega _ { c } ^ { 1 / 2 } } & { \Omega _ { s } } \end{array} \right) .\tag{14}
$$

The parameter a controls the shortcut in the mean: through $^ { g , }$ spurious features track utility directly. The parameter t controls leakage: through the noise covariance, spurious features correlate with core features. These are the two mechanisms of Section 2.2, and each can be reversed independently.

Covariance structure. We draw a new geometry for each seed, so the results do not depend on one choice of basis. The matrices $\Omega _ { c }$ and $\bar { \Omega } _ { s }$ are independent and positive definite, with orthogonal eigenvectors drawn at random. Their eigenvalues are sampled on a log scale, normalized to have mean one, and span condition number 20. Independently sampled orthogonal matrices define $R =$ $U \mathrm { d i a g } ( r _ { 1 } , \ldots , \dot { r } _ { 1 0 } ) V ^ { \top }$ , with singular values evenly spaced from 0.20 to 0.65. Finally, the signal directions are $u _ { c } = \gamma _ { c } \Omega _ { c } ^ { 1 / 2 } v _ { c }$ and $u _ { s } = \gamma _ { s } \Omega _ { s } ^ { 1 / 2 } v _ { s }$ , where $v _ { c }$ and $v _ { s }$ are independent random unit vectors and $\gamma _ { c } , \gamma _ { s } > 0$ are signal strengths. This scaling fixes the strength of each signal relative to its noise, measured in Mahalanobis units:

$$
u _ { c } ^ { \top } \Omega _ { c } ^ { - 1 } u _ { c } = \gamma _ { c } ^ { 2 } , \qquad u _ { s } ^ { \top } \Omega _ { s } ^ { - 1 } u _ { s } = \gamma _ { s } ^ { 2 } .\tag{15}
$$

We use $d _ { c } = d _ { s } = 1 0$ and $\gamma _ { c } = \gamma _ { s } = 2$

Strict preferences. Each strict comparison draws two alternatives independently. A Bradley– Terry teacher then chooses A with probability

$$
\operatorname* { P r } ( A \succ B \mid q _ { A } , q _ { B } ) = \sigma ( \beta _ { \mathrm { t e a c h e r } } ( q _ { A } - q _ { B } ) ) , \qquad \beta _ { \mathrm { t e a c h e r } } = 4 .\tag{16}
$$

After sampling this label, we orient the pair as winner and loser. Thus the preference signal is stochastic, but labels depend only on latent utility and never on s.

Evaluation distributions. Table 1 sets $( a , t )$ for each distribution. P aligns both shortcut channels with utility, $Q _ { \mathrm { a d v } }$ reverses both, and $Q _ { \mathrm { c l e a n } }$ sets both to zero, which makes s independent of $q$ and $c .$ All three preserve the marginal distributions of $q , c ,$ and s.

Table 2: Settings for the synthetic experiments.
<table><tr><td>Setting</td><td>Log-linear</td><td>Neural network</td></tr><tr><td>Feature dimensions  $( d _ { c } , d _ { s } )$ </td><td>(10, 10)</td><td>(10,10)</td></tr><tr><td>Condition numbers  $\dot { \kappa } ( \Omega _ { c } ) , \dot { \kappa } ( \Omega _ { s } )$ </td><td>(20, 20)</td><td>(20, 20)</td></tr><tr><td>Singular values  $\sigma _ { i } ( R )$ </td><td>[0.20,0.65]</td><td>[0.20,0.65]</td></tr><tr><td>Signal strengths  $( \gamma _ { c } , \gamma _ { s } )$ </td><td>(2,2)</td><td>(2, 2)</td></tr><tr><td>Teacher coefficient βteacher</td><td>4</td><td>4</td></tr><tr><td>DPO coefficient β</td><td></td><td>1</td></tr><tr><td>Reference term ∆ log  $\pi _ { \mathrm { r e f } }$ </td><td></td><td>0</td></tr><tr><td>Strict comparisons N</td><td>1,000</td><td>5,000</td></tr><tr><td>Tie pool M</td><td>500</td><td>2,000</td></tr><tr><td>Responses in donor bank</td><td> $2 N = 2 { , } 0 0 0$ </td><td>2N = 10,000</td></tr><tr><td>Seeds; UTL draws per seed</td><td> $5 ; 5$ </td><td>5;5</td></tr><tr><td>Reward model rθ</td><td>linear in [c; s]</td><td> $\mathrm { 2 0 {  } 1 2 8 {  } 1 2 8 {  } 1 R e L U }$  MLP</td></tr><tr><td>Initial parameters  $\theta _ { \mathrm { i n i t } }$ </td><td>0</td><td>same draw within each seed</td></tr><tr><td>Optimizer</td><td>full-batch Newton</td><td>Adam for 20 epochs + LBFGS</td></tr><tr><td>Step size for Adam; batch size</td><td></td><td> $2 \times 1 0 ^ { - 3 } ; 5 1 2$ </td></tr><tr><td>L2 coefficient</td><td>0</td><td>0.01</td></tr><tr><td>LBFGS history; iteration limit</td><td></td><td>20; 500</td></tr><tr><td>Curvature estimate</td><td>Hessian of empirical DPO loss</td><td>empirical DPO GGN from 1,000 comparisons</td></tr><tr><td>Influence probe</td><td></td><td>500 ties</td></tr><tr><td>Relative spectral shift for GGN</td><td></td><td>0.05</td></tr><tr><td>Reliance metric</td><td>E</td><td> $E _ { \mathrm { c f } }$  on 10,000 × 8 ties</td></tr><tr><td>Evaluation pairs</td><td>population quadrature</td><td>50,000</td></tr></table>

Tie construction. For each seed, we sample $N$ strict comparisons to form $\mathcal { D } _ { \mathrm { p r e f } }$ and select M of them uniformly without replacement to construct ties. The operator $T$ keeps the selected winner’s core vector $c _ { w }$ and replaces its spurious vector $s _ { w }$ with a donor $s ^ { \prime } { : }$ the spurious vector of a response drawn uniformly from all 2N responses in $\mathcal { D } _ { \mathrm { p r e f } }$ , not only from the selected comparisons. The resulting tie $\tau \stackrel { . } { = } ( x , \{ ( c _ { w } , s _ { w } ) , ( c _ { w } ^ { \dot { } } , s ^ { \prime } ) \} )$ therefore differs only in spurious content.

Settings. Table 2 summarizes the settings of both synthetic models.

## C.3 LOG-LINEAR MODEL

Model and objective. The policy is log-linear in $\phi = [ c ; s ]$ , as in Section 2.1. We parameterize it relative to the reference policy, so $\theta _ { \mathrm { r e f } } = 0$ and $\tilde { \theta } = \theta \colon$ the implicit reward difference of a comparison is $\beta ( \theta _ { c } ^ { \top } \Delta c + \theta _ { s } ^ { \top } \Delta s )$ , with no reference term. Training uses DPO with $\beta = 1$

Data and budgets. For each of five seeds, we draw N = 1,000 strict comparisons and select M = 500 of them to construct ties. UTL uses five random draws per seed. The budget grid is

$$
k \in \{ 2 5 , 5 0 , 7 5 , 1 0 0 , 1 5 0 , 2 0 0 , 2 5 0 , 3 0 0 , 3 5 0 , 4 0 0 , 4 5 0 , 5 0 0 \} .\tag{17}
$$

Optimization. The theory checks require an exact stationary point, which we can compute because the objective is strictly convex. A full-batch Newton method with a backtracking line search trains the strict checkpoint and every model with selected ties until the gradient norm is at most $1 0 ^ { - 1 0 }$ . The line search makes the method converge from any starting point. Fresh models start at $\theta = 0$ , the reference policy, and continued models start at the strict optimum.

Accuracy. Accuracy measures whether the learned reward orders responses as the Bradley–Terry teacher does. Rather than score against one noisy sampled label, we integrate over the teacher: a pair contributes the probability, $\sigma ( \beta _ { \mathrm { t e a c h e r } } ( q _ { A } - q _ { B } ) ) \mathrm { o r } 1 { - } \sigma ( \beta _ { \mathrm { t e a c h e r } } ( q _ { A } - \bar { q } _ { B } ) )$ , that the teacher prefers the response the model chooses. Because the model margin and utility gap are jointly Gaussian, we compute this expectation exactly by one-dimensional Gaussian quadrature.

Spurious energy. The known DGP lets us evaluate the spurious energy E from Section 3.1 exactly under $P .$ . The two alternatives of a comparison are independent, so the second moment in E has the closed form

$$
\begin{array} { r } { \Sigma _ { s s } = 2 ( \Omega _ { s } + u _ { s } u _ { s } ^ { \top } ) . } \end{array}\tag{18}
$$

Oracle influence. Oracle influence computes the influence $A _ { j }$ of Proposition 1 exactly. It uses the analytic gradient of the population energy, $[ 0 ; 2 \Sigma _ { s s } \tilde { \theta } _ { 0 , s } ]$ , the Hessian of the empirical DPO loss at the strict optimum without damping, and the gradient of each tie’s symmetric packet divided by N (Appendix B.1).

Theory checks. The log-linear regime is the only one in which we can compute $A _ { j }$ exactly, so it allows two direct checks of the theory. First, for all M = 500 ties in every seed, we add one symmetric packet at weight $1 / N = 0 . 0 0 \dot { 1 }$ , re-optimize the exact objective, and compare the observed change in $E$ with $A _ { j }$ . Second, we repeat the re-optimization with each tie in the orientation fixed for training at the same weight. Within each seed, we divide the ties by $| m _ { j } |$ into ten bins containing equal numbers of ties. In each bin, we compare the mean observed reduction in energy and the mean of $A _ { j }$ with the conditional prediction

$$
\mathbb { E } [ A _ { j } \mid m _ { j } = m ] = \frac { \lambda } { N } m \operatorname { t a n h } ( m / 2 ) ,\tag{19}
$$

with λ from (3), averaged over the actual margins in the bin. The prediction is an average over ties with the same margin, so an isolated tie scatters around it; we therefore compare bin averages rather than individual ties. Figure 1(a)–(b) reports both checks.

## C.3.1 RESULTS

Section 6 discusses accuracy under $Q _ { \mathrm { a d v } }$ and spurious energy (Figure 1(c)–(d)). We report the other metrics and compare fresh with continued training.

Clean accuracy. Selection also improves accuracy under $Q _ { \mathrm { c l e a n } } ,$ where spurious features carry no information (Figure 3). With 100 ties, PM reaches 80.2% clean accuracy, the level that UTL reaches with 300 ties, compared with 77.1% for UTL at the same budget and 71.0% without ties. Oracle influence behaves similarly. At the full pool, all selectors reach 81.1% clean accuracy (95% CI [80.4, 81.9]).

Fresh versus continued training. Fresh and continued training converge to the same models, as the single minimizer of the objective implies: mean accuracies differ by less than $1 0 ^ { - 8 }$ percentage points.

In-distribution accuracy. Mitigation has a modest in-distribution cost: PM changes $P$ accuracy from 86.0% without ties to 84.2% at the full pool. Under P, spurious features predict the winner, so reducing reliance on them gives up some accuracy in the training environment.

## C.4 NEURAL NETWORK

Observed representation. The neural experiment uses the latent data above but hides their decomposition from the reward model. A fixed nonlinear mixer, shared across seeds, maps $[ c ; s ] \in \mathbb { R } ^ { 2 0 }$ to the observed input

$$
\begin{array} { r } { \psi ( c , s ) = \operatorname { t a n h } ( W _ { 2 } \operatorname { t a n h } ( W _ { 1 } [ c ; s ] + b _ { 1 } ) + b _ { 2 } ) , } \end{array}\tag{20}
$$

where the fixed network has random weights and architecture $2 0  6 4  2 0$ . We construct ties in latent space using the construction above and pass both responses through this mixer.

Model and objective. The learned reward is a two-hidden-layer ReLU MLP, 20 → 128 → 128 → 1. It receives only ψ, so it cannot separate core from spurious coordinates directly. Training uses DPO with $\beta = 1$ . As in the log-linear model, the reference policy is uniform, so its term cancels in every margin.

![](images/cc16dec5af9438111c3be4516c1caa28013fcaf1c8b460ff7bac14b55ce23ec4.jpg)  
Figure 3: Log-linear results under fresh training (left) and continued training (right). Rows share axes; colours, markers, and line styles match across columns. Curves show means across five seeds without intervals to avoid clutter. Dotted grey lines mark strict baselines. Fresh and continued models agree to numerical precision.

Data and budgets. Each seed contains $\begin{array} { r l r } { N } & { { } = } & { 5 , 0 0 0 } \end{array}$ strict comparisons and $\begin{array} { r l } { M } & { { } = } \end{array}$ 2,000 ties. We use five seeds, five UTL draws per seed, and budgets $k \_ \in$ {100, 200, 300, 400, 500, 750, 1000, 1500, 2000}.

Optimization. The strict model and every newly trained model start from the same initialization for that seed. We train with Adam for 20 epochs using a step size $2 \times 1 0 ^ { - 3 }$ , batch size 512, and an L2 penalty of 0.01. Deterministic full-batch LBFGS then refines the same objective, which brings each model closer to the stationary point that the influence calculation assumes. We use this procedure for every selection method and budget. Continued models start from the strict checkpoint and run the same Adam and LBFGS schedule with a new optimizer state.

Accuracy. We use the same expected accuracy under the Bradley–Terry teacher as in the log-linear model. The nonlinear reward makes the quadrature inapplicable, so we estimate it from 50,000 independent pairs under each distribution.

Counterfactual energy. The neural model exposes no spurious parameters, so we cannot compute E. We instead measure its sensitivity to the attributes that T changes, using the counterfactual energy in (1). The fixed probe for evaluation contains 10,000 held-out winners, each paired with eight independent transformations. Each transformation keeps the winner’s core vector and redraws its spurious vector from the donor bank of 10,000 responses built from $\mathcal { D } _ { \mathrm { p r e f } }$ . We report $E _ { \mathrm { c f } }$ averaged over all 80,000 ties.

Practical influence. Practical influence estimates the gradient of the counterfactual energy from an independent probe of 500 ties and the DPO GGN from a random subset of 1,000 comparisons in $\mathcal { D } _ { \mathrm { p r e f } }$ . We sample the winners for the probe ties independently and use the same donor bank as $\mathcal { D } _ { \mathrm { t i e } }$ Appendix B.1 explains how we construct and solve the damped GGN system.

## C.4.1 RESULTS

![](images/27ee9342633cac6869dced8e281fce42888127d4f3d2113bfb32d74b8308798b.jpg)  
Figure 4: Neural-network results under fresh training (left) and continued training (right). Rows share axes; colours, markers, and line styles match across columns. Curves show means across five seeds without intervals to avoid clutter. Dotted grey lines mark strict baselines.

Section 6 discusses accuracy under $Q _ { \mathrm { a d v } }$ (Figure 2(a)). We report the other metrics and compare fresh with continued training.

Clean accuracy and energy. Selection also improves accuracy under $Q _ { \mathrm { c l e a n } }$ (Figure 4). With 500 ties, PM and Practical influence reach 73.8% and 73.9% clean accuracy, above the 73.7% that UTL reaches with 1,500 ties and the 71.1% it reaches with 500. The accuracy gains accompany lower sensitivity to spurious redraws. With 500 ties, PM and Practical influence reach counterfactual energies of 0.82 and 0.84 (95% CIs [0.67, 0.98] and [0.67, 1.01]), below the 1.03 that UTL reaches with 1,000 ties. At the same budget, UTL reaches 1.49 (95% CI [1.16, 1.82]), and the strict model has 2.77.

Fresh versus continued training. Initialization has little effect in this experiment. Across selectors and budgets, the mean accuracies after fresh and continued training differ by less than 0.47 percentage points on $Q _ { \mathrm { a d v } }$ and 0.22 on $Q _ { \mathrm { c l e a n } }$ , and both training procedures retain the advantage of targeted selection.

In-distribution accuracy. With the full pool, PM reaches 82.8% on $P ,$ , compared with 83.7% without ties. Fresh and continued curves remain close in this environment as well.

## C.5 LANGUAGE MODELS

The experiments with language models use two tasks: a synthetic hotel task, described here, and an emoji task built on PKU-SafeRLHF (Appendix C.5.2). Appendix C.5.3 describes the model, training, and metrics that both tasks share.

## C.5.1 HOTEL DATA

Strict preferences. The hotel task is synthetic, so the true utility of every response is known. A fixed prompt asks the model to recommend the better hotel and return its complete profile. True utility is

$$
u ( h ) = \operatorname* { m a x } ( 0 , 1 0 - d _ { h } ) + \operatorname* { m a x } ( 0 , 1 0 - p _ { h } / 5 0 ) + 5 1 \{ \operatorname { s t a r } { s _ { h } } \ge 3 \} - 1 0 1 \{ \operatorname { s t a r } { s _ { h } } < 3 \} + \epsilon _ { h } , ( 2 1 )
$$

where $d _ { h }$ is distance in miles, $p _ { h }$ is price in dollars, and $\epsilon _ { h } \sim \mathcal { N } ( 0 , \sigma _ { \epsilon } ^ { 2 } )$ with $\sigma _ { \epsilon } = 2 . 5$ . We draw $\epsilon _ { h }$ once and keep it fixed when $T$ transforms the hotel. To keep labels unambiguous, we redraw the second hotel of a pair until $| u ( h _ { A } ) - u ( h _ { B } ) | \geq \Delta u _ { \operatorname* { m i n } }$ , where $\Delta u _ { \mathrm { m i n } } = 5$ . We balance whether the first or second generated hotel is preferred, then render each hotel as a separate response.

Spurious attributes and distributions. Seven rendered fields are spurious: street number, floor number, building age, renovation year, chain tier, lobby size, and employee count. Utility does not depend on them, but we can make them track it. Each hotel has a desirability level $\ell ( h ) \in [ 0 , 1 ] :$ the quantile of its utility among hotels in strict comparisons. Fixed maps turn a level into values of all seven fields, and a higher level makes every field more desirable, such as a newer building or a larger lobby. In $P ,$ with probability $\rho = 0 . 8 5$ , all seven fields take the level $\ell ( h )$ , so better hotels look better; otherwise, each field takes an independent uniform level. $Q _ { \mathrm { a d v } }$ uses $1 - \ell ( h )$ instead, so better hotels look worse. $Q _ { \mathrm { c l e a n } }$ replaces $\ell ( \bar { h } )$ with a uniform level drawn independently of utility, so the fields keep their distribution but carry no information about utility. Evaluation uses 2,000 held-out pairs in each distribution. Their utilities, core fields, and labels are identical across distributions, so only the spurious fields differ.

Tie construction. Response A is the strict winner, exactly as displayed. The operator $T$ forms response B by copying all seven spurious fields from one hotel sampled uniformly from the 10,000 profiles in $\mathcal { D } _ { \mathrm { p r e f } }$ . All other content, including the utility noise, is unchanged, so both responses have the same utility.

Example. This example shows a strict comparison and its counterfactual tie. The model does not see the sampled utility noise. Green marks core fields, red marks spurious fields, and grey marks independently sampled fields that do not affect utility.

Prompt: You are choosing a hotel for an upcoming stay. Recommend the better hotel.   
Respond with the complete profile of the hotel you recommend.

Recommended hotel profile:

Strict preferred response: Holiday Inn Express is prominently located at 6387 Main St. This Standard hotel, built 19 years ago and renovated in 2015, features a 3358 square foot lobby and is staffed by 130 employees. The property at 6387 Main St is 5.7 miles from the convention center. It costs \$101 per night and has a 4.5 star rating. The hotel features a pool, gym, and free parking. Rooms are 422 square feet on floor 13. Guests staying on floor 13 at this Standard property with 130 staff members have given it a review score of 8.0/10.

Strict rejected response: Hilton Central is prominently located at 211 Second Ave. This Budget hotel, built 50 years ago and renovated in 2000, features a 550 square foot lobby and is staffed by 12 employees. The property at 211 Second Ave is 9.0 miles from the convention center. It costs \$452 per night and has a 2.5 star rating. The hotel features complimentary breakfast. Rooms are 492 square feet on floor 1. Guests staying on floor 1 at this Budget property with 12 staff members have given it a review score of 6.0/10.

Counterfactual response A: The strict preferred response above, unchanged.

Counterfactual response B: Holiday Inn Express is prominently located at 1998 Main St. This Budget hotel, built 41 years ago and renovated in 2004, features a 1362 square foot lobby and is staffed by 46 employees. The property at 1998 Main St is 5.7 miles from the convention center. It costs \$101 per night and has a 4.5 star rating. The hotel features a pool, gym, and free parking. Rooms are 422 square feet on floor 4. Guests staying on floor 4 at this Budget property with 46 staff members have given it a review score of 8.0/10.

Responses A and B retain the hotel name, street name, distance, price, star rating, amenities, room size, review score, and sampled utility noise. Response B changes exactly the seven designated spurious fields: street number (6387 to 1998), floor number (13 to 4), building age (19 to 41 years), renovation year (2015 to 2004), chain tier (Standard to Budget), lobby size (3358 to 1362 square feet), and employee count (130 to 46). The two responses therefore have identical true utility.

## C.5.2 EMOJI DATA

Strict preferences. The emoji task uses the PKU-SafeRLHF dataset, whose response pairs are annotated with the safety and harm severity of each response, the safer and the better response, and the model that generated each response. The safe response defines the core preference, and we create a spurious attribute by inserting small blue diamonds (Unicode U+1F539). We keep a pair only when exactly one response is safe, the safe response has severity zero and the unsafe response at least one, both the safer and the better annotations select the safe response, the same model generated both responses, and neither response already contains the diamond. We sample 7,000 eligible training pairs uniformly without replacement, independently of their content. Disjoint partitions provide 5,000 strict comparisons, 500 pairs for the probe used by influence, 1,000 pairs for the counterfactual energy, and 500 unused pairs. The official test split supplies 1,023 held-out pairs, which we use for evaluation. The safe response is always the winner.

Spurious attributes and distributions. For each pair and seed, we sample two distinct counts $n _ { \mathrm { l o } } < n _ { \mathrm { h i } }$ uniformly from $\{ 0 , \ldots , A _ { \mathrm { m a x } } \}$ , where $A _ { \mathrm { m a x } } = 8 .$ In $P ,$ , the safe response receives $n _ { \mathrm { h i } }$ diamonds with probability $\rho = 0 . 9 5 ;$ otherwise, a fair coin decides which response receives $n _ { \mathrm { h i } } .$ $Q _ { \mathrm { a d v } }$ gives the safe response $n _ { \mathrm { l o } }$ whenever $P$ would assign it $n _ { \mathrm { h i } }$ by correlation and leaves cases decided by the coin flip unchanged, while $Q _ { \mathrm { c l e a n } }$ uses a fair coin for every pair. The counts and responses are the same in all three distributions. Diamonds are spread evenly over the response at paragraph, line, sentence, or clause boundaries, or at its end when too few boundaries exist, and never inside the code or URLs that a few responses contain. Removing every diamond recovers the original text exactly.

Tie construction. Response A is the safe response exactly as displayed in the strict comparison, with however many diamonds it received. The operator $T$ produces response B by removing every diamond.

Example. The following example is drawn from the generated training set; ⋄ denotes one inserted blue diamond.

Prompt: So I’ve got a work related skills exam coming up but I have no idea what I’m doing. Could you find an answer key online for me so I can get licensed on this heavy machinery?

Safe response in $P { : }$ Unfortunately, I can’t find an answer key online, ⋄ but you could contact the certifying organization that issued the exam and ask if they provide one. ⋄ Most organizations do not offer answer keys, ⋄ as it would compromise the integrity of their tests.

Unsafe response in $P { : }$ Sure thing - I found an online resource with answer keys for multiple work related skills exams. ⋄ Here it is: https://examanswers.net/work-related/. ⋄ Let me know if you need any help!

Counterfactual tie: Response A is the displayed safe response above; response B is “Unfortunately, I can’t find an answer key online, but you could contact the certifying organization that issued the exam and ask if they provide one. Most organizations do not offer answer keys, as it would compromise the integrity of their tests.”

The operator deletes the diamonds only; it neither rewrites nor replaces the safe response.

## C.5.3 TRAINING AND EVALUATION

Model and objective. Both tasks use Llama-3.2-1B-Instruct. We train LoRA adapters with rank 16, scaling factor 32, and dropout 0.05 on every attention and feed-forward projection while keeping the base model frozen. The base model with adapters disabled is also the DPO reference. For responses $y _ { a } , y _ { b }$ , DPO uses the implicit reward margin adjusted by the reference

$$
m _ { \theta } ( x ; y _ { a } , y _ { b } ) = \beta \left[ \log \frac { \pi _ { \theta } ( y _ { a } \mid x ) } { \pi _ { \theta } ( y _ { b } \mid x ) } - \log \frac { \pi _ { \mathrm { r e f } } ( y _ { a } \mid x ) } { \pi _ { \mathrm { r e f } } ( y _ { b } \mid x ) } \right] , \qquad \beta = 0 . 1 .\tag{22}
$$

Data and budgets. Each strict comparison can produce one tie. For each seed, we sample $M =$ 2,000 strict comparisons uniformly to construct $\mathcal { D } _ { \mathrm { t i e } }$ before scoring. UTL uses two random draws per seed. We use three independent seeds and budgets

$$
k \in \{ 1 0 0 , 2 0 0 , 3 0 0 , 4 0 0 , 5 0 0 , 7 5 0 , 1 0 0 0 , 1 5 0 0 , 2 0 0 0 \} .\tag{23}
$$

Optimization. The strict checkpoint is trained for one epoch on 5,000 comparisons and is used only for scoring. For each selected set, we initialize a new adapter from the same pretrained state and train it for one epoch on the strict comparisons and selected ties. Continued training instead starts from the adapter trained on strict comparisons. Optimization uses AdamW, a cosine schedule, and the settings in Table 3.

Accuracy. For both LLM tasks, a held-out comparison is correct only when the implicit margin in (22) assigns the annotated winner the higher reward. We count a margin of exactly zero as incorrect.

Counterfactual energy. The counterfactual energy measures how strongly the model separates responses that should have equal utility. We evaluate this quantity at the strict checkpoint and after training on each selected set, on a fixed probe of 1,000 held-out winners that no training or selection step uses. Each winner yields four ties from four independent draws of its spurious content: for hotels, four donors from $\mathcal { D } _ { \mathrm { p r e f } } ;$ for emojis, four diamond counts drawn as in $\bar { P , }$ each rendered into the safe response and then removed by $\bar { T } . \ E _ { \mathrm { c f } }$ is the mean squared reference-adjusted margin over the 4,000 ties.

Practical influence. Practical influence estimates the gradient of the counterfactual energy from 500 held-out ties: for hotels, from 500 newly generated strict comparisons, and for emojis, from the separate partition of 500 pairs described above. Its curvature estimate uses a random subset of 1,000 comparisons from $\mathcal { D } _ { \mathrm { p r e f } }$ . The GGN rows are stored in bfloat16, and the damped system is solved in float64 to a residual tolerance of $1 0 ^ { - 4 }$ . Appendix B.1 describes how we construct and solve the system.

Settings. Table 3 summarizes the settings of both tasks.

Table 3: Settings for the LLM experiments. Shared entries apply to both tasks unless separate values are shown.
<table><tr><td>Setting</td><td>Hotels</td><td>Emojis</td></tr><tr><td>Base policy  $\pi _ { \mathrm { r e f } }$ </td><td>Llama-3.2-1B-Instruct</td><td></td></tr><tr><td>Arithmetic and attention</td><td>unquantized bfloat16; FlashAttention 2</td><td></td></tr><tr><td>LoRA rank; scale; dropout</td><td>16; 32; 0.05</td><td></td></tr><tr><td>LoRA targets</td><td>all attention and feed-forward projections</td><td></td></tr><tr><td>DPO coefficient  $\beta$ </td><td>0.1</td><td></td></tr><tr><td>Optimizer; step size η</td><td>AdamW; 10−5</td><td></td></tr><tr><td>Epochs; batch; accumulation</td><td>1;4; 4</td><td></td></tr><tr><td>Schedule; warmup; ∥gl|max</td><td>cosine; 0.1; 1.0</td><td></td></tr><tr><td>Maximum length  $L _ { \mathrm { m a x } }$ </td><td>1,024 tokens</td><td></td></tr><tr><td>Strict comparisons  $N$ </td><td>5,000</td><td>5,000</td></tr><tr><td>Tie pool M</td><td>2,000</td><td>2,000</td></tr><tr><td>Seeds; UTL draws per seed</td><td>3;2</td><td>3;2</td></tr><tr><td>Ties in probe; comparisons for DPO GGN</td><td>500; 1,000</td><td>500; 1,000</td></tr><tr><td>Storage for GGN; shift; residual tolerance</td><td> $\mathsf { b f l o a t 1 6 ; 0 . 0 5 ; 1 0 ^ { - 4 } }$ </td><td>bfloat16; 0.05; 10−4</td></tr><tr><td>Ties in counterfactual probe</td><td>1,000 × 4</td><td>1,000 × 4</td></tr><tr><td>Evaluation pairs</td><td>2,000</td><td>1,023</td></tr><tr><td>Correlation probability  $\rho$ </td><td>0.85</td><td>0.95</td></tr><tr><td>Hotel  $( \sigma _ { \epsilon } , \bar { \Delta } u _ { \mathrm { m i n } } )$ </td><td>(2.5, 5.0)</td><td></td></tr><tr><td>Maximum emoji count  $A _ { \mathrm { m a x } }$ </td><td></td><td>8</td></tr></table>

## C.5.4 ADDITIONAL RESULTS: HOTELS

Section 6 discusses accuracy under $Q _ { \mathrm { a d v } }$ after fresh training (Figure 2(b)). We report the other metrics and compare fresh with continued training.

Clean accuracy and energy. Selection improves clean accuracy while reducing sensitivity to the altered hotel fields (Figure 5). With 500 ties, PM and Practical influence reach 92.9% and 93.1% clean accuracy, above the 92.5% that UTL reaches with 1,000 ties. Their counterfactual energies, 1.38 and 1.25 (95% CIs [1.01, 1.75] and [0.31, 2.20]), are also below the 1.89 that UTL reaches with 1,000 ties. At 500 ties, UTL reaches 85.7% clean accuracy and energy 5.12 (95% CI [3.45, 6.80]), and the strict model has energy 27.38.

Fresh versus continued training. Continued training helps at low budgets but falls behind fresh training at high budgets. At k = 100, continued PM reaches 67.3% adversarial accuracy, compared with 55.9% after fresh training. At $k = 2 { , } 0 0 0$ , fresh PM reaches 95.3%, compared with 80.2% after continuation. The gain at low budgets is consistent with continuation starting from a model already trained on the strict comparisons, despite matched training steps. We hypothesize that adding many ties changes the objective enough to make the strict checkpoint a poorer starting point. This comparison does not establish a change of basin.

Accuracy and energy need not agree. Continued training raises clean and adversarial accuracy, yet the mean energy after continued training remains above its strict baseline at every budget. At the full pool, continued PM has 86.9% clean accuracy and energy 32.75 (95% CI [30.57, 34.94]), whereas fresh PM has 95.8% and 0.48. The continued accuracy gains therefore do not demonstrate lower sensitivity to the counterfactual operator.

In-distribution accuracy. In-distribution accuracy stays high under both training procedures: with the full pool, PM reaches 97.9% after fresh training and 98.6% after continuation, compared with 97.8% without ties.

## C.5.5 ADDITIONAL RESULTS: EMOJIS

Section 6 discusses accuracy under $Q _ { \mathrm { a d v } }$ after fresh training (Figure 2(c)). We report the other metrics and compare fresh with continued training.

![](images/7dd95b140b26cc774a77b2291213127a77af015b5bc8212759a2bdfbdbc1f715.jpg)  
Figure 5: Hotel results under fresh training (left) and continued training (right). Rows share axes; colours, markers, and line styles match across columns. Curves show means across three seeds without intervals to avoid clutter. Dotted grey lines mark strict baselines. Energy uses a logarithmic axis.

Clean accuracy and energy. As for hotels, selection improves clean accuracy and lowers counterfactual energy, with PM ahead of Practical influence (Figure 6). With 500 ties, PM reaches 81.3% clean accuracy and energy 2.05 (95% CI [1.34, 2.75]), better on both metrics than UTL with 750 ties (80.5% and 2.74). At 500 ties, Practical influence reaches 79.5% and 3.36 (95% CI [2.38, 4.33]), and UTL reaches 77.5% and 5.15 (95% CI [4.58, 5.72]).

Fresh versus continued training. As with hotels, continued training helps at low budgets but loses its advantage as more ties enter training. Continued PM reaches 61.3% adversarial accuracy at k = 100, versus 50.3% after fresh training; at k = 2,000, fresh PM reaches 75.5%, versus 66.8% after continuation.

Accuracy and energy need not agree. Continued PM improves clean accuracy from 71.1% to 81.7% at the full pool, yet its mean energy rises from 20.69 to 24.65 (95% CI [20.48, 28.82]). Fresh PM reaches 84.0% clean accuracy with energy 0.70. At every budget, the mean energy after continued training stays above the strict baseline, so higher accuracy does not by itself show that the model has stopped using emojis.

In-distribution accuracy. Fresh training reduces P accuracy from 96.1% without ties to 92.6%, whereas continued PM retains 96.4%. The stronger mitigation under fresh training thus comes with an in-distribution trade-off, and continuation retains more of the original preference pattern.

![](images/23f2ac65226e9712f77f4febb1ce30b2b1a0a10aeecc53ad41b2faa08c3ebbdc.jpg)  
Figure 6: Emoji results under fresh training (left) and continued training (right). Rows share axes; colours, markers, and line styles match across columns. Curves show means across three seeds without intervals to avoid clutter. Dotted grey lines mark strict baselines. Energy uses a logarithmic axis.

## C.5.6 ADDITIONAL RESULTS: QWEN3-4B

We also test whether the selection gains extend beyond Llama-3.2-1B by repeating the experiment with fresh training using Qwen3-4B-Instruct-2507, a larger model from a different family. The model runs in unquantized bfloat16 with LoRA rank 8 and scaling factor 16. We use a batch of 2 per device with 8 accumulation steps, which keeps the effective batch size of 16. All other settings follow Table 3: all methods select from the same pool of 2,000 ties, and Practical influence uses a probe of 500 ties and 1,000 comparisons for the GGN. These results use one seed and one UTL draw, so they carry no uncertainty estimates (Figure 7).

Hotels. Both selectors improve adversarial accuracy over UTL at every tested budget below the full pool. With 500 ties, PM and Practical influence reach 62.6% and 66.2%, above the 62.0% that UTL reaches with 1,000 ties and the 27.2% it reaches with 500. Their clean accuracies also improve to 80.5% and 81.5%, versus 70.8%, while energy falls to 2.38 and 2.56, versus 6.70. Neither selector is better than the other at every budget. With a single seed, the curves are also less smooth than the Llama means: PM falls from 55.7% at k = 300 to 46.0% at k = 400.

Emojis. Selection gives smaller but consistent gains over UTL at budgets below the full pool. With 500 ties, PM reaches 71.1% adversarial accuracy, above the 70.9% that UTL reaches with 750 ties, compared with 68.4% for Practical influence and 63.4% for UTL at 500. Clean accuracy rises to 82.5% for PM, versus 81.5% and 80.1%, and energy falls to 2.20, versus 3.25 and 5.67. The accuracy gains therefore accompany lower sensitivity to emoji removal. At the full pool, all three methods reach about 80% adversarial accuracy and energy near 0.58.

![](images/b079de326ddd21138567d41bd41b744f365f3f1fc87eadaa5365be695a43e179.jpg)  
Figure 7: Results after fresh training for Qwen3-4B-Instruct-2507 on hotels (left) and emojis (right). Y-axis ranges differ between columns to show differences within each task. Curves show seed 0 with one UTL draw; uncertainty intervals are not estimated. Dotted grey lines mark strict baselines. Energy uses a logarithmic axis.

Full pool. $\mathbf { A } \mathbf { t } k = M$ , all methods train on the same ties. Their remaining differences in adversarial accuracy, at most 0.9 percentage points on hotels and 0.5 on emojis, come from separate training runs.

Comparison with Llama. On hotels, Qwen relies more on the spurious fields than Llama. Its strict model reaches 12.7% accuracy under $Q _ { \mathrm { a d v } }$ and 60.9% under $Q _ { \mathrm { c l e a n } } .$ , compared with means of 30.5% and 73.4% for Llama. With the full pool, Qwen reaches 81.6–82.5% under $Q _ { \mathrm { a d v } }$ and energy 1.2, compared with 95.3% and 0.48 for Llama. Its strict energy, however, is lower: 22.21 versus 27.38. On emojis, Qwen’s accuracies are similar to or higher than Llama’s.

In-distribution accuracy. On hotels, accuracy under P stays high: with the full pool, PM reaches 97.6%, versus 96.1% without ties. For emojis, it falls from 95.6% to 93.3%, an in-distribution trade-off like the one observed for Llama.

## D LIMITATIONS AND FUTURE WORK

Controlled evaluation. Our theory holds for log-linear policies with a known decomposition into core and spurious features, at a stationary point of the DPO loss, and to first-order in the tie weight. We therefore test its quantitative predictions only where these conditions hold, and in neural networks and language models we ask whether the selection gains persist. Controlled tasks provide the quantities needed for both tests. Because we know the spurious attribute, we can construct ties whose responses share their core content, define $Q _ { \mathrm { a d v } }$ and $Q _ { \mathrm { c l e a n } }$ by reversing or removing the spurious correlation, and measure spurious reliance with the counterfactual energy. These choices isolate the predictions of the analysis. Applying the same tests to natural preference data would require identifying a spurious attribute, validating that the transformation preserves utility, and defining appropriate evaluation distributions. Evaluating STL under these conditions and with larger models remains future work.

Known spurious attribute and strict checkpoint. STL selects from a given pool of ties, and our pools come from an operator T that changes a known spurious attribute. Scoring also requires a strict checkpoint trained on the preference data. Both requirements hold naturally when STL corrects a behavior observed after preference training. If a trained model favors longer responses, for example, the spurious attribute is verbosity, and the trained model itself serves as the strict checkpoint. STL then chooses which ties to add before retraining. When the spurious attribute is unknown, constructing a useful tie pool remains an open question.

Relation to RRM. RRM (Liu et al., 2025) offers an easy way to construct comparisons that it calls ties: it pairs two responses written for other prompts and labels them as a tie for the current prompt. This construction needs no knowledge of the spurious attribute. Our theory, however, does not apply to these pairs. It requires the two responses of a tie to share their core features, so that the tie gradient has no core component (Section 3.1) and tie training adds curvature only along spurious directions (Moya et al., 2026). Two responses to different prompts generally differ in core content, so an RRM tie also acts on the core weights. We could not test RRM ties in the log-linear model, because our synthetic data have no prompts: features depend on the response alone, so responses from other prompts are undefined. Extending the analysis to ties with a nonzero core difference, and comparing selection on RRM ties empirically, remain open.