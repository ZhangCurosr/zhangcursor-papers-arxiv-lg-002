# Statistical Learning Theory for Direct Preference Optimization

Joel Wills J.Wills@maths.usyd.edu.au   
School of Mathematics and Statistics   
The University of Sydney   
Yuanfan Li yuli0846@uni.sydney.edu.au   
School of Mathematics and Statistics   
The University of Sydney   
Yiming Ying yiming.ying@sydney.edu.au   
School of Mathematics and Statistics   
The University of Sydney   
Ding-Xuan Zhou dingxuan.zhou@sydney.edu.au   
School of Mathematics and Statistics   
The University of Sydney

## Abstract

Direct Preference Optimization (DPO) learns from pairwise preferences through a logistic objective whose score-diference parameterization follows the Bradley–Terry model. We study how failure of this modeling assumption limits DPO, distinguishing structural misspecification from limited model capacity and finite-sample estimation. Our main contribu tion is a characterization of structural approximation error through the triangle-curl of the preference log-odds, which vanishes exactly when preferences admit a Bradley–Terry representation. We upper bound the achievable excess logistic risk by a score-class approximation term and a triangle-curl term, and establish a lower bound proportional to triangle-curl un der additional boundedness assumptions. For ReLU score models, including clipped models, we derive explicit approximation rates showing that increasing capacity can eliminate the score-approximation term while leaving the structural obstruction unchanged. We further establish finite-sample bounds for clipped empirical risk minimizers that separate network approximation, structural misspecification, and estimation error. We place these results within a surrogate-risk framework by identifying contextual preference accuracy (CPA) as a natural measure of agreement between preference judgments and changes in pairwise probability ratios relative to the reference policy. Standard logistic consistency and calibration results connect DPO’s objective to this policy property and transfer our excess-risk upper bounds to CPA, without treating CPA as a complete measure of generated-response quality.

## 1 Introduction

Aligning large language models (LLMs) with human preferences commonly relies on pairwise comparisons, where annotators indicate which of two responses is preferred for a given prompt. Learning from these comparisons requires a model of how preferences relate across responses. The resulting policy is therefore shaped not only by the available data, but also by the restrictions imposed by this model.

Reinforcement Learning from Human Feedback (RLHF) (Christiano et al., 2017; Stiennon et al., 2020; Ouyang et al., 2022; Kaufmann et al., 2023; Lambert, 2025) first fits a reward model to preference data and then optimizes a policy using reinforcement learning. Direct Preference Optimization (DPO) (Rafailov et al., 2023) simplifies this pipeline through a single supervised objective. Its logistic loss models preference log-odds as diferences of policy-to-reference log-density ratios, following the Bradley–Terry (BT) model (Bradley & Terry, 1952).

The distinction between policy expressiveness and preference representability is central. Even an unrestricted scalar score cannot represent arbitrary pairwise preference log-odds: score diferences must sum to zero around every triangle of responses. Cyclic preferences violate this condition, but even preferences with a consistent ordering need not satisfy it. Increasing model capacity therefore need not eliminate misspecification. Recent work studies misspecification arising from restrictions on the policy class under BT preferences (Gopalan et al., 2026). We examine the complementary question of what happens when the preferences themselves do not admit any score-diference representation.

Our central question is how this structural restriction limits the logistic risk achievable by DPO, and how that limitation can be distinguished from insuficient model capacity and finite preference data. We approach this question through statistical learning theory, separating structural misspecification, score-class approximation, and finite-sample estimation.

To connect this risk analysis to policy behavior, we adopt the classical surrogate-risk framework (Bartlett et al., 2006; Zhang, 2004). We consider contextual preference accuracy (CPA), the accuracy of the preference decisions implicit in DPO. At the policy level, CPA measures how often the policy increases the probability of the preferred response relative to the other response, compared with the reference policy. This captures a specific aspect of policy behavior, without guaranteeing improved generated-response quality. Calibration connects excess logistic risk to CPA, allowing our analysis of DPO’s structural and statistical limitations to yield guarantees for this policy property.

Contributions. We formulate DPO within the classical surrogate-risk framework (Bartlett et al., 2006; Zhang, 2004), identifying contextual preference accuracy (CPA) as a natural measure of agreement between preference judgments and policy reweighting relative to the reference. Established consistency and calibration properties of logistic loss connect this criterion to DPO’s objective and allow our excess-risk upper bounds to transfer to CPA. Our contribution here is this formulation and its policy interpretation, rather than new calibration properties of logistic loss.

Our main technical contribution is a characterization of the structural approximation error imposed by DPO’s score-diference parameterization. We introduce a triangle-curl functional of the preference log-odds that vanishes exactly when preferences admit a Bradley–Terry representation. We upper bound the achievable excess logistic risk by a score-class approximation term and a triangle-curl term, and establish a lower bound proportional to triangle-curl under additional boundedness assumptions. For ReLU score models, including clipped models, we derive explicit rates showing that the capacity-dependent approximation term can be made arbitrarily small. We further establish finite-sample bounds for clipped empirical risk minimizers, separating network approximation, structural misspecification, and estimation error.

## 1.1 Related Works

A growing literature develops alternatives to DPO, including methods for noisy preference data (Chowdhury et al., 2024), alternative preference objectives (Azar et al., 2024; Zhao et al., 2023), reference-free optimization (Meng et al., 2024), and joint supervised fine-tuning and preference optimization (Hong et al., 2024). Alongside these algorithmic developments, theoretical work has established guarantees under specific modeling assumptions. However, our analysis addresses the less studied question of how failure of the Bradley–Terry assumption limits DPO.

Our analysis follows the classical surrogate-risk framework in statistical learning theory (SLT) (Bartlett et al., 2006; Zhang, 2004), which studies the relationship between a tractable surrogate loss and a target population risk. Fisher consistency and calibration are well understood for binary classification and ranking, including AUC optimization (Clémençon et al., 2008; Gao & Zhou, 2015; Ying et al., 2016), but analogous analyses of preference optimization objectives for LLM alignment remain limited.

Recent work has also examined the limitations of the Bradley–Terry (BT) model. Pukdee et al. (2026) characterize when pairwise comparison data admits a BT representation and analyze the behavior of BT learning under model misspecification. From a social choice perspective, Dong et al. (2026) show that excess risk decomposes into representation and aggregation errors, the latter arising from Condorcet cycles in pooled preferences. Meanwhile, Gölz et al. (2026) quantify the loss incurred when aggregating heterogeneous BT models into a single policy or ranking. These works highlight fundamental limitations of representing heterogeneous preferences with a single score-based model.

Empirically, Chen et al. (2024) show that preference learning methods such as DPO often produce only modest improvements in preference accuracy, suggesting a persistent gap between optimization objectives and alignment performance that may partly arise from the influence of the reference model. Gopalan et al. (2026) study DPO under Bradley–Terry preferences, showing how restrictions on the policy class can prevent its induced rewards from representing the preference-generating reward. Our analysis addresses a distinct source of misspecification: the preference log-odds may not admit a score-diference representation at all. We quantify this obstruction through triangle-curl and separate it from score-class approximation error. The two perspectives are complementary, distinguishing limitations of the chosen policy class from limitations of the score-diference representation itself.

## 2 Background and Problem Setup

Preference-based alignment methods, including RLHF and DPO, use judgments comparing two candidate responses to the same prompt. We study the preference-prediction problem implicit in the DPO objective and the statistical consequences of its policy parameterization.

Let X denote the prompt space and Y the response space. We observe preference data of the form

$$
( X , Y , Y ^ { \prime } , A ) \sim D , \qquad A \in \{ - 1 , + 1 \} ,
$$

where $A = 1$ indicates that $Y$ is preferred to $Y ^ { \prime }$ , and $A = - 1$ indicates the opposite preference. We assume that, conditional on X, the responses Y and $Y ^ { \prime }$ are independently sampled from a common distribution $p ( \cdot \mid X )$ . Define the conditional preference probability

$$
\eta ( x , y , y ^ { \prime } ) : = \mathbb { P } ( A = 1 \mid X = x , Y = y , Y ^ { \prime } = y ^ { \prime } ) .\tag{1}
$$

A policy assigns a conditional probability distribution $\pi ( \cdot \mid x )$ over responses to each prompt x. We write $\pi ( y \mid x )$ for its probability mass or density, as appropriate. Let $\pi _ { \mathrm { r e f } }$ be a fixed reference policy. Where policyto-reference ratios are used, we consider policies and a reference policy with positive masses or densities on the support of the comparison distribution.

DPO as preference prediction. DPO uses the policy-induced score

$$
s _ { \pi } ( x , y ) : = \tau \log \frac { \pi ( y \mid x ) } { \pi _ { \mathrm { r e f } } ( y \mid x ) } , \qquad \tau > 0 ,
$$

and the corresponding comparison function

$$
f _ { \pi } ( x , y , y ^ { \prime } ) : = s _ { \pi } ( x , y ) - s _ { \pi } ( x , y ^ { \prime } ) .
$$

Its population objective is

$$
- \mathbb { E } _ { ( X , Y , Y ^ { \prime } , A ) \sim D } \left[ \log \sigma \left( A f _ { \pi } ( X , Y , Y ^ { \prime } ) \right) \right] ,\tag{2}
$$

where $\sigma ( t ) = ( 1 + e ^ { - t } ) ^ { - 1 }$ (Rafailov et al., 2023).

Thus, DPO has the form of binary logistic prediction with input $( X , Y , Y ^ { \prime } )$ and label A. The quantity $\sigma ( f _ { \pi } ( x , y , y ^ { \prime } ) )$ is its predicted probability that $y$ is preferred to $y ^ { \prime }$ . Since $\sigma ( f _ { \pi } ( x , y , y ^ { \prime } ) ) > 1 / 2$ if and only if $f _ { \pi } ( x , y , y ^ { \prime } ) > 0$ , the sign of $f _ { \pi }$ determines which response the model predicts to be preferred, with zero corresponding to a tie. This decision compares policy-to-reference density ratios, rather than raw policy probabilities π $\tau ( y \mid x )$

To analyse this prediction problem, we extend the logistic objective to arbitrary measurable triplet-wise predictors $f : \mathcal { X } \times \mathcal { Y } \times \mathcal { Y }  \mathbb { R }$ . For $\ell ( u ) = \log ( 1 + e ^ { - u } )$ , define

$$
\mathcal { L } ( f ) : = \mathbb { E } _ { ( X , Y , Y ^ { \prime } , A ) \sim D } \left[ \ell \left( A f ( X , Y , Y ^ { \prime } ) \right) \right] = - \mathbb { E } _ { ( X , Y , Y ^ { \prime } , A ) \sim D } \left[ \log \sigma \left( A f ( X , Y , Y ^ { \prime } ) \right) \right] .
$$

Substituting $f = f _ { \pi }$ recovers equation 2. DPO therefore minimizes ${ \mathcal { L } } ( f _ { \pi } )$ over the chosen policy class.

Contextual preference accuracy. The sign of $f _ { \pi }$ determines the preference decision implicit in DPO’s logistic objective. To study these decisions, for any measurable triplet-wise predictor $f : \mathcal { X } \times \mathcal { Y } \times \mathcal { Y }  \mathbb { R }$ define its contextual preference accuracy (CPA) as

$$
\begin{array} { r l } & { P ( f ) : = \mathbb { E } _ { X , Y , Y ^ { \prime } } \Big [ \eta ( X , Y , Y ^ { \prime } ) \mathbf { 1 } \{ f ( X , Y , Y ^ { \prime } ) > 0 \} } \\ & { \qquad + \left( 1 - \eta ( X , Y , Y ^ { \prime } ) \right) \mathbf { 1 } \{ f ( X , Y , Y ^ { \prime } ) < 0 \} } \\ & { \qquad + \frac { 1 } { 2 } \mathbf { 1 } \{ f ( X , Y , Y ^ { \prime } ) = 0 \} \Big ] . } \end{array}
$$

This is the population accuracy of the induced preference decisions, with half credit for ties. Under DPO’s parameterization, these decisions describe how the policy reweights responses relative to the reference: for positive policy and reference probabilities,

$$
f _ { \pi } ( x , y , y ^ { \prime } ) > 0 \Longleftrightarrow \frac { \pi ( y \mid x ) } { \pi ( y ^ { \prime } \mid x ) } > \frac { \pi _ { \mathrm { r e f } } ( y \mid x ) } { \pi _ { \mathrm { r e f } } ( y ^ { \prime } \mid x ) } .
$$

Thus, $P ( f _ { \pi } )$ measures how often the policy increases the probability of the preferred response relative to the other response, compared with the reference policy, with half credit for unchanged ratios. In particular, if $P ( f _ { \pi } ) = P ^ { * } : = \operatorname* { s u p } _ { f } P ( f )$ , where the supremum is over all measurable triplet-wise predictors, then Lemma 2 implies that the displayed inequality holds for almost every pair with $\eta ( x , y , y ^ { \prime } ) > 1 / 2$ . However, as discussed by Chen et al. (2024), this density-ratio ordering does not imply $\pi ( y \mid x ) > \pi ( y ^ { \prime } \mid x )$ . Moreover, both responses may lose probability even when their ratio changes in the preferred direction. CPA alone therefore does not establish an improvement in the quality of generated responses.

The key question is how closely DPO’s score-diference predictors can approximate the optimal preference log-odds. We study this through excess logistic risk, using the surrogate-risk framework to connect risk bounds to CPA. This separates the representational limitations of DPO from the calibration properties of the logistic loss.

## 3 Fisher Consistency and Calibration

The consistency and calibration results in this section are standard consequences of logistic surrogate-risk theory (Bartlett et al., 2006; Zhang, 2004); we do not claim these results as novel. Our contribution is the formulation of DPO as a surrogate for CPA and its connection to the policy interpretation developed above. Working with unrestricted triplet-wise predictors separates these properties of the loss from the score-diference approximation issues studied in Section 4. First, we show Fisher consistency: minimizing the logistic risk recovers a predictor that maximizes CPA. Second, we strengthen this connection by establishing a quantitative calibration inequality, showing that excess logistic risk controls excess CPA. We include proofs for completeness.

## 3.1 Fisher Consistency

To establish the Fisher consistency of logistic risk with respect to CPA, we characterize the population minimizers of logistic risk and the population maximizers of CPA. The following lemma characterizes the population minimizers of logistic population risk. It is the direct three-variable (triplet-wise) analog of the characterization of functions minimizing the population logistic risk in binary classification (Bartlett et al., 2006; Zhang, 2004). Here, the optimal object is the preference log-odds.

All proofs of the results that follow can be found in the Appendix.

Lemma 1. A measurable triplet-wise function f minimizes the population logistic risk if and only $i f$

$$
f ( x , y , y ^ { \prime } ) = \log \frac { \eta ( x , y , y ^ { \prime } ) } { 1 - \eta ( x , y , y ^ { \prime } ) } \quad a . s . ,
$$

with the convention that the right hand side $i s + \infty$ when $\eta ( x , y , y ^ { \prime } ) = 1$ and −∞ when $\eta ( x , y , y ^ { \prime } ) = 0$

Define $f ^ { * }$ to be the logistic risk-minimizing preference log-odds function

$$
f ^ { * } ( x , y , y ^ { \prime } ) = \log \frac { \eta ( x , y , y ^ { \prime } ) } { 1 - \eta ( x , y , y ^ { \prime } ) } .
$$

Lemma 2. A measurable triplet-wise function f maximizes $P ( f )$ if and only if

$$
\mathrm { s i g n } ( f ( x , y , y ^ { \prime } ) ) = \mathrm { s i g n } ( 2 \eta ( x , y , y ^ { \prime } ) - 1 ) \quad a . s .
$$

on the set where $\eta ( x , y , y ^ { \prime } ) \neq \frac { 1 } { 2 }$ , with arbitrary tie-breaking allowed on the set where $\eta ( x , y , y ^ { \prime } ) = 1 / 2$ Moreover,

$$
P ^ { * } : = \operatorname* { s u p } _ { f } P ( f ) = \mathbb { E } \left[ \operatorname* { m a x } \{ \eta ( X , Y , Y ^ { \prime } ) , 1 - \eta ( X , Y , Y ^ { \prime } ) \} \right] .
$$

The optimal value $P ^ { * }$ is the pairwise analog of the optimal Bayes accuracy in binary classification and represents the highest achievable agreement with human preferences.

Combining Lemma 1 with Lemma 2, we can establish Fisher consistency in the following theorem. Specifically, minimizing the logistic risk recovers the true preference log-odds $\begin{array} { r } { f ^ { * } ( x , y , y ^ { \prime } ) = \log \frac { \eta ( x , y , y ^ { \prime } ) } { 1 - \eta ( x , y , y ^ { \prime } ) } } \end{array}$ , whose sign coincides almost surely with $\mathrm { s g n } ( 2 \eta ( x , y , y ^ { \prime } ) - 1 )$ , implying that every logistic risk minimizer maximizes the CPA.

Theorem 3. Every minimizer of the population logistic risk maximizes the population CPA:

$$
\arg \operatorname* { m i n } _ { f } { \mathcal { L } } ( f ) \subseteq \arg \operatorname* { m a x } _ { f } P ( f ) ,
$$

where we consider measurable triplet-wise functions $f : \mathcal { X } \times \mathcal { Y } \times \mathcal { Y }  \mathbb { R }$

## 3.2 Calibration Inequality

The Fisher consistency result of Theorem 3 only demonstrates consistency in a qualitative way; it does not quantify how optimization error in the surrogate risk transfers to preference alignment performance. We now strengthen this connection by establishing a calibration inequality, which controls excess CPA in terms of the excess logistic risk.

Define

$$
C ^ { * } : = \operatorname* { i n f } _ { f } \mathcal { L } ( f ) ,
$$

where the infimum is over all measurable triplet-wise predictors $f : \mathcal { X } \times \mathcal { Y } \times \mathcal { Y }  \mathbb { R }$

Theorem 4. For any measurable triplet-wise function $f ,$ there holds

$$
P ^ { * } - P ( f ) \leq \sqrt { 2 } \sqrt { \mathcal { L } ( f ) - \mathcal { L } ^ { * } } .
$$

Remark 5. The calibration inequality implies statistical consistency. For any sequence of measurable tripletwise functions $\{ f _ { n } \} _ { n \geq 1 }$

$$
{ \mathcal { L } } ( f _ { n } ) \to { \mathcal { L } } ^ { * } \quad \Longrightarrow \quad P ( f _ { n } ) \to P ^ { * } .
$$

Thus, convergence of the logistic risk guarantees convergence to the optimal CPA.

## 4 Approximation Error of Score-Diference DPO

For any score function $s : \mathcal { X } \times \mathcal { Y }  \mathbb { R }$ , define

$$
\Delta _ { s } ( x , y , y ^ { \prime } ) : = s ( x , y ) - s ( x , y ^ { \prime } ) .
$$

Section 3 proves consistency and calibration for unrestricted triplet-wise functions, however, DPO restricts the predictor to $f _ { \pi } = \Delta _ { s _ { \pi } }$ for the policy induced score functions $s _ { \pi }$ . By Theorem 4,

$$
P ^ { * } - \operatorname* { s u p } _ { s } P ( \Delta _ { s } ) \leq \sqrt { 2 } \left( \operatorname* { i n f } _ { s } \mathcal { L } ( \Delta _ { s } ) - \mathcal { L } ^ { * } \right) ^ { 1 / 2 } .
$$

Thus, even when restricting triplet-wise functions to the score-diferences, excess CPA is controlled by the surrogate-risk approximation error. From Lemma 1, $f ^ { * }$ is the preference log-odds and global minimizer of $\mathcal { L } ( f )$ , so to understand the surrogate risk approximation error, we need to understand the limitations of approximating $f ^ { * }$ with a score-diference function.

The key message of this section is that restricting to score-diference functions imposes an intrinsic structural limitation on the triplet-wise preference functions that can be represented, regardless of model capacity. A score-diference function assigns each response a scalar latent score and compares responses through score diferences. Consequently, score diferences satisfy a structural consistency condition: for any three responses,

$$
f ( x , y _ { 1 } , y _ { 2 } ) + f ( x , y _ { 2 } , y _ { 3 } ) + f ( x , y _ { 3 } , y _ { 1 } ) = 0 .
$$

Thus, not every triplet-wise preference function admits a score-diference representation, and the ability of a triplet-wise function to admit a score-diference representation is governed by its behavior over all response triples. Naturally, we ask to what extent does $f ^ { * }$ fail this condition and how this afects the performance of DPO.

We formalize this intuition by introducing triangle-curl as a measure of score-model misspecification, using combinatorial curl from combinatorial Hodge theory (Jiang et al., 2011). Theorem 7 characterizes scorediference functions through vanishing triangle-curl; in particular, $\mathcal { T } ( f ^ { * } ) = 0$ if and only if preferences admit a Bradley–Terry representation. Theorem 9 relates this quantity to surrogate-risk approximation error, providing an upper bound and, under additional boundedness assumptions, a lower bound. We then specialize these bounds to ReLU score models and their induced DPO policies in Theorem 11, establishing approximation upper bounds for both unclipped and clipped scores, together with a triangle-curl lower bound for the clipped class under additional boundedness assumptions on the preferences. Finally, Theorem 12 combines the clipped-class approximation bound with finite-sample estimation bounds and transfers the resulting excess-risk guarantee to CPA through calibration.

Triangle-Curl. For a triplet-wise function $f ( x , y , y ^ { \prime } )$ , we apply the combinatorial curl operator from combinatorial Hodge theory (Jiang et al., 2011) to $f ( x , \cdot , \cdot ) ;$

$$
( \operatorname { c u r l } f ) ( x ; y _ { 1 } , y _ { 2 } , y _ { 3 } ) : = f ( x , y _ { 1 } , y _ { 2 } ) + f ( x , y _ { 2 } , y _ { 3 } ) + f ( x , y _ { 3 } , y _ { 1 } ) .
$$

The combinatorial curl characterizes when a triplet-wise function can be represented by a scalar score function. Intuitively, score diferences always telescope around a triangle of responses, so any score-diference model is curl-free. The following lemma shows that the converse also holds: vanishing curl is exactly the condition under which a triplet-wise function admits a score-diference representation.

Lemma 6. Let $f : \mathcal { X } \times \mathcal { Y } \times \mathcal { Y }  \mathbb { R }$ be antisymmetric in its response arguments. Then curl $f = 0$ if and only if there exists a score function $s : \mathcal { X } \times \mathcal { Y }  \mathbb { R }$ such that

$$
f ( x , y , y ^ { \prime } ) = s ( x , y ) - s ( x , y ^ { \prime } )
$$

for all $y , y ^ { \prime } \in \mathcal { V }$

For conditionally i.i.d $Y _ { 1 } , Y _ { 2 } , Y _ { 3 } \ \sim \ p ( \cdot \ | \ X )$ , define the triangle-curl of a square-integrable triplet-wise function f as

$$
\begin{array} { r } { \mathcal { I } ( f ) : = \mathbb { E } _ { X , Y _ { 1 } , Y _ { 2 } , Y _ { 3 } } \left[ ( \mathrm { c u r l } f ) ( X ; Y _ { 1 } , Y _ { 2 } , Y _ { 3 } ) ^ { 2 } \right] . } \end{array}
$$

Because combinatorial curl can be negative, this quantity aggregates the squared combinatorial curl of $f$ across random response triples to measure the overall inability of $f$ to admit a score-diference representation. The key quantity that we are interested in is $\operatorname { \mathcal { T } } ( f ^ { * } )$ . Since $f ^ { * }$ is the preference log-odds, not admitting a score-diference representation is an underlying property of the preferences. In particular, $\eta ( x , y , y ^ { \prime } ) =$ $\sigma ( f ^ { * } ( x , y , y ^ { \prime } ) )$ , and $\operatorname { \mathcal { T } } ( f ^ { * } )$ can be interpreted as measuring the extent to which the preferences do not follow the Bradley–Terry model (Bradley & Terry, 1952):

$$
\eta ( x , y , y ^ { \prime } ) = \sigma ( s ( x , y ) - s ( x , y ^ { \prime } ) ) .
$$

We have a global analogy to Lemma 6 using triangle-curl.

Theorem 7. Let $f : \mathcal { X } \times \mathcal { Y } \times \mathcal { Y }  \mathbb { R }$ be square-integrable and satisfy $f ( x , y , y ^ { \prime } ) = - f ( x , y ^ { \prime } y )$ almost surely. Then $\textstyle { \mathcal { T } } ( f ) = 0$ if and only if there exists a measurable function $s : \mathcal { X } \times \mathcal { Y }  \mathbb { R }$ such that

$$
f ( x , y , y ^ { \prime } ) = s ( x , y ) - s ( x , y ^ { \prime } ) \qquad a . s .
$$

In particular, under Assumption $\ 8 , \ \mathcal { T } ( f ^ { * } ) = 0 \ i f$ and only if preferences almost surely follow the Bradley– Terry model: there exists a measurable function $s : \mathcal { X } \times \mathcal { Y }  \mathbb { R }$ such that

$$
\eta ( x , y , y ^ { \prime } ) = \sigma ( s ( x , y ) - s ( x , y ^ { \prime } ) ) \ \mathrm { ~ a . s . }
$$

Theorem 7 characterizes score-diference representations independently of the learning objective. In particular, let $g ^ { * }$ be a square-integrable antisymmetric population minimizer of an unrestricted preference-learning objective. Then $\boldsymbol { \mathcal { T } } ( \boldsymbol { g } ^ { * } ) = 0$ if and only if $g ^ { * }$ admits an exact score-diference representation. This provides a representability criterion for other preference-learning methods that use score diferences, although its implications for excess risk depend on the chosen loss.

Assumption 8. The preference log-odds function $f ^ { * }$ is square-integrable, and hence finite almost surely, so that $\eta ( x , y , y ^ { \prime } ) \in ( 0 , 1 )$ almost surely. All score functions considered are also assumed to be square-integrable. Finally, preferences are reciprocal, i.e. $\eta ( x , y ^ { \prime } , y ) = 1 - \eta ( x , y , y ^ { \prime } )$ , which implies that $f ^ { * }$ is antisymmetric: $f ^ { * } ( x , y , y ^ { \prime } ) = - f ^ { * } ( x , y ^ { \prime } , y )$ for all $x , y , y ^ { \prime }$

Under Assumption 8, define the centered score potentia

$$
s ^ { * } ( x , y ) : = \mathbb { E } _ { U \sim p ( \cdot | x ) } \big [ f ^ { * } ( x , y , U ) \big ] .\tag{3}
$$

The induced score diference $\Delta _ { s ^ { * } }$ is the $L ^ { 2 }$ projection of $f ^ { * }$ onto the score-diference subspace, while $\mathcal { T } ( f ^ { \ast } ) / 3$ is the squared $L ^ { 2 }$ norm of the orthogonal residual; see Lemmas 15 and 16 in Appendix A.7. In this sense, $\operatorname { \mathcal { T } } ( f ^ { * } )$ measures the intrinsic component of $f ^ { * }$ that cannot be represented as a score diference.

Theorem 9. For any class $s$ of square-integrable score functions, the population excess logistic risk satisfies

$$
\operatorname* { i n f } _ { s \in \mathcal { S } } \big ( \mathcal { L } ( \Delta _ { s } ) - \mathcal { L } ^ { * } \big ) \ \leq \ \frac { 1 } { 8 } \operatorname* { i n f } _ { s \in \mathcal { S } } \mathbb { E } \big [ ( \Delta _ { s } - \Delta _ { s ^ { * } } ) ^ { 2 } \big ] + \frac { 1 } { 2 4 } \mathcal { T } ( f ^ { * } ) .
$$

Moreover, if for every $s \in \mathcal { S } , \ \delta < \sigma ( \Delta _ { s } ( X , Y , Y ^ { \prime } ) ) < 1 - \delta$ and $\delta < \sigma ( f ^ { * } ( X , Y , Y ^ { \prime } ) ) < 1 - \delta$ for some $\delta \in ( 0 , 1 / 2 )$ almost surely, then

$$
{ \frac { \delta ^ { 2 } } { 6 } } { \mathcal { T } } ( f ^ { * } ) \ \leq \ \operatorname* { i n f } _ { s \in { \mathcal { S } } } { \big ( } { \mathcal { L } } ( \Delta _ { s } ) - { \mathcal { L } } ^ { * } { \big ) } .
$$

Theorem 9 reduces the score-diference approximation error into two terms: an approximation error measuring how well the score-diference component $\Delta _ { s ^ { * } }$ of $f ^ { * }$ can be approximated by the score class $S ,$ and an intrinsic error given by the triangle-curl. The first term decreases with model expressiveness, whereas the triangle-curl depends only on the underlying preference distribution. Under the additional boundedness conditions, nonzero triangle-curl yields a positive lower bound on the achievable excess logistic risk. The structural obstruction vanishes exactly when preferences admit a Bradley–Terry representation, although approximation error may remain within a restricted score class.

We now explicitly bound the surrogate risk approximation error, taking S to be a class of deep ReLU neural networks.

Hypothesis space. Let $\Omega : = \mathcal { X } \times \mathcal { Y } \subset \mathbb { R } ^ { d } , \mathrm { ~ } z = ( x , y )$ , and let $\phi ( t ) = \operatorname* { m a x } \{ 0 , t \}$ . For any $m \in \mathbb { N }$ and $\mathbf { v } \in \mathbb { R } ^ { m }$ , define

$$
\phi _ { \mathbf { v } } : \mathbb { R } ^ { m } \to [ 0 , \infty ) ^ { m } , \qquad z \mapsto \big ( \phi ( ( z - \mathbf { v } ) _ { 1 } ) , \dots , \phi ( ( z - \mathbf { v } ) _ { m } ) \big ) .
$$

A ReLU neural network is a function $s : \mathbb { R } ^ { d } $ R of the form

$$
s ( z ) = \mathbf { W } _ { L } \phi _ { \mathbf { v } _ { L } } \mathbf { W } _ { L - 1 } \phi _ { \mathbf { v } _ { L - 1 } } \boldsymbol { \cdot } \boldsymbol { \cdot } \boldsymbol { \cdot } \mathbf { W } _ { 1 } \phi _ { \mathbf { v } _ { 1 } } \mathbf { W } _ { 0 } z ,\tag{4}
$$

where $L \in \mathbb { N } , m _ { 0 } = d , m _ { L + 1 } = 1 , \mathbf { v } _ { i } \in \mathbb { R } ^ { m _ { i } }$ , and $\mathbf { W } _ { i } \in \mathbb { R } ^ { m _ { i + 1 } \times m _ { i } }$ for $0 \leq i \leq L$ . For $L , N \in \mathbb { N }$ , define

$S ( d , L , N ) : = \{ s : \mathbb { R } ^ { d }  \mathbb { R } \ | \ s $ is of the form equation 4 and $m _ { i } \leq N$ for $1 \leq i \leq L \}$

(5)

The corresponding triplet-wise class is

$$
\Delta S ( d , L , N ) : = \{ \Delta _ { s } \mid s \in S ( d , L , N ) \} .
$$

We impose some regularity assumption on the optimal scoring function $s ^ { * }$ . In this work, we consider Hölder continuity assumption, which is standard in the literature (Nagler & Langer, 2026; Schmidt-Hieber, 2020; Lu et al., 2021). For $\beta \in ( 0 , 1 ]$ and $h : \Omega \to \mathbb { R }$ , the Hölder norm of h on Ω is defined as

$$
\| h \| _ { \beta , \Omega } : = \operatorname* { m a x } \left\{ \operatorname* { s u p } _ { z \in \Omega } | h ( z ) | , \operatorname* { s u p } _ { z _ { 1 } \neq z _ { 2 } \in \Omega } \frac { | h ( z _ { 1 } ) - h ( z _ { 2 } ) | } { \| z _ { 1 } - z _ { 2 } \| _ { \infty } ^ { \beta } } \right\} .
$$

Assumption 10. There exist $R > 0 , \beta \in ( 0 , 1 ]$ , and $B > 0$ such that $\Omega = \mathcal { X } \times \mathcal { Y } \subseteq [ - R , R ] ^ { d }$ and $\| s ^ { * } \| _ { \beta , \Omega } \leq B$

Let $\Theta ( d , L , N )$ denote the parameter space corresponding to ReLU networks $s \in \mathcal { S } ( d , L , N )$ . Each parameter vector $\theta \in \Theta ( d , L , N )$ induces a score function $s _ { \theta } ,$ , which in turn induces a policy

$$
\pi _ { \theta } ( y \mid x ) = \frac { \pi _ { \mathrm { r e f } } ( y \mid x ) \exp ( s _ { \theta } ( x , y ) / \tau ) } { Z _ { \theta } ( x ) } , \qquad Z _ { \theta } ( x ) = \int _ { Y } \pi _ { \mathrm { r e f } } ( u \mid x ) \exp ( s _ { \theta } ( x , u ) / \tau ) d u .
$$

We then have the triplet-wise score-diference function $f _ { \pi _ { \theta } } ( x , y , y ^ { \prime } ) = s _ { \theta } ( x , y ) - s _ { \theta } ( x , y ^ { \prime } )$

We cannot invoke the lower bound of Theorem 9 for the score class S, as there is no uniform bound of the score values, and thus $\sigma ( \Delta _ { s } ( x , y , y ^ { \prime } ) )$ can be made arbitrarily close to zero and one. Therefore, we additionally consider bounded ReLU network score functions. For $\theta \in \Theta ( d , L , N )$ , define

$$
s _ { \theta , B } : = \mathrm { c l i p } _ { B } \circ s _ { \theta } , \qquad \mathrm { c l i p } _ { B } ( t ) : = \operatorname* { m a x } \{ - B , \operatorname* { m i n } \{ B , t \} \} ,
$$

so that $\begin{array} { r } { \operatorname* { s u p } _ { ( x , y ) \in \mathcal { X } \times \mathcal { Y } } | s _ { \theta , B } ( x , y ) | \le B } \end{array}$ and B is the same constant from Assumption 10. Let $\pi _ { \boldsymbol { \theta } , B }$ denote the policy induced by $s _ { \boldsymbol { \theta } , B }$ through the same exponential parameterization, so that $f _ { \pi _ { \theta , B } } = \Delta _ { s _ { \theta , B } } .$

Theorem 11. Let Assumptions 8 and 10 hold. Then there exist constants $C _ { 1 } , C _ { 2 } > 0$ , depending only on $d , \beta , B , R ,$ such that for every $L _ { 0 } , N _ { 0 } \ge 2 ,$ , if

$$
\bar { L } = C _ { 1 } L _ { 0 } + C _ { 1 } L _ { 0 } \left\lceil \frac { \sqrt { \log L _ { 0 } } } { \sqrt { \log N _ { 0 } } } \right\rceil , \qquad \bar { N } = C _ { 1 } N _ { 0 } ,
$$

then

$$
\operatorname * { i n f } _ { \theta \in \Theta ( d , \bar { L } , \bar { N } ) } \left[ { \mathcal L } ( f _ { \pi _ { \theta } } ) - { \mathcal L } ^ { * } \right] \ \le \ C _ { 2 } \left( \frac { 1 } { L _ { 0 } ^ { 2 } N _ { 0 } ^ { 2 } } \right) ^ { 2 \beta / d } \ + \ \frac { 1 } { 2 4 } { \mathcal L } ( f ^ { * } ) .
$$

Consequently, for every $\varepsilon \in ( 0 , 1 )$ , there exist network sizes $\bar { L } _ { \varepsilon } , \bar { N } _ { \varepsilon }$ such that

$$
\operatorname* { i n f } _ { \theta \in \Theta ( d , \bar { L } _ { \varepsilon } , \bar { N } _ { \varepsilon } ) } \left[ \mathcal { L } ( f _ { \pi _ { \theta } } ) - \mathcal { L } ^ { * } \right] \ \lesssim \ \varepsilon + \mathcal { T } ( f ^ { * } ) .
$$

The above two upper bounds also hold true when $f _ { \pi _ { \theta } }$ is replaced by $f _ { \pi _ { \theta , B } }$ for the clipped score class. If additionally $a \leq \eta ( X , Y , Y ^ { \prime } ) \leq 1 - a$ almost surely for some $a \in ( 0 , 1 / 2 )$ , then, for any $0 < \delta < \operatorname* { m i n } \{ a , \sigma ( - 2 B ) \}$ }, the excess DPO risk of the clipped score class is bounded below:

$$
\frac { \delta ^ { 2 } } { 6 } \mathcal { T } ( f ^ { * } ) \leq \operatorname* { i n f } _ { \theta \in \Theta ( d , \bar { L } , \bar { N } ) } \left[ \mathcal { L } ( f _ { \pi _ { \theta , B } } ) - \mathcal { L } ^ { * } \right] .
$$

Theorem 11 separates a capacity-dependent approximation term, which vanishes as network size increases, from a structural term determined by triangle-curl. The upper bounds hold for both the original and clipped parameterizations. For the clipped class, if the preference probabilities are bounded away from zero and one, the matching lower bound shows that nonzero triangle-curl creates an irreducible excess-risk floor, regardless of network capacity. Under these additional conditions, triangle-curl therefore controls the best achievable excess DPO risk up to multiplicative constants as the capacity-dependent term vanishes.

## 4.1 Finite-Sample Generalization

We now combine the approximation bounds with a finite-sample estimation bound for empirical risk minimization. We use the clipped parameterization introduced above to ensure that the logistic loss is uniformly bounded.

Fix $L _ { 0 } , N _ { 0 } \ge 2$ , with corresponding network dimensions $\bar { L } , \bar { N }$ from Theorem 11, and write

$$
\mathcal S _ { B } : = \{ s _ { \theta , B } : \theta \in \Theta ( d , \bar { L } , \bar { N } ) \} , \qquad \Delta \mathcal S _ { B } : = \{ \Delta _ { s } : s \in \mathcal S _ { B } \} .
$$

Let $D _ { n } = \{ ( X _ { i } , Y _ { i } , Y _ { i } ^ { \prime } , A _ { i } ) \} _ { i = } ^ { n }$ be i.i.d. samples from $D _ { : }$ under the conditional sampling model specified in Section 2. Define the empirical logistic risk

$$
\widehat { \mathcal { L } } _ { n } ( f ) : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \log \left( 1 + \exp \{ - A _ { i } f ( X _ { i } , Y _ { i } , Y _ { i } ^ { \prime } ) \} \right) .
$$

Suppose a measurable empirical risk minimizer exists, and let

$$
\widehat { \theta } \in \underset { \theta \in \Theta ( d , \bar { L } , \bar { N } ) } { \arg \operatorname* { m i n } } ~ \widehat { \mathcal { L } } _ { n } \big ( f _ { \pi _ { \theta , B } } \big ) .
$$

Write

$$
\mathfrak { R } _ { n } ( S _ { B } ) : = \mathbb { E } \left[ \operatorname* { s u p } _ { s \in S _ { B } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \epsilon _ { i } s ( X _ { i } , Y _ { i } ) \right]
$$

for the expected Rademacher complexity, where $\epsilon _ { 1 } , \ldots , \epsilon _ { n }$ are independent uniform signs, independent of the observations. The expectation is over both the observations and the signs.

Theorem 12 (Finite-sample generalization). Let Assumptions $\boldsymbol { \vartheta }$ and 10 hold. Set $M = \log ( 1 + e ^ { 2 B } )$ . For any $\alpha \in ( 0 , 1 )$ , with probability at least $1 - \alpha$

$$
\begin{array} { r l r } {  { \mathcal { L } ( f _ { \pi _ { \widehat { \theta , B } } } ) - \mathcal { L } ^ { * } \leq C _ { 2 } ( \frac { 1 } { L _ { 0 } ^ { 2 } N _ { 0 } ^ { 2 } } ) ^ { 2 \beta / d } + \frac { 1 } { 2 4 } \mathcal { T } ( f ^ { * } ) } } \\ & { } & { \quad + 8 \Re _ { n } ( S _ { B } ) + 2 M \sqrt { \frac { \log ( 2 / \alpha ) } { 2 n } } , } \end{array}
$$

where $C _ { 2 }$ is the constant in Theorem 11. On the same event,

$$
\begin{array} { r l r } {  { P ^ { * } - P ( f _ { \pi _ { \widehat { \theta , B } } } ) \leq \sqrt { 2 } \Bigg [ C _ { 2 } ( \frac { 1 } { L _ { 0 } ^ { 2 } N _ { 0 } ^ { 2 } } ) ^ { 2 \beta / d } + \frac { 1 } { 2 4 } \mathscr { T } ( f ^ { * } ) } } \\ & { } & \\ & { } & { + 8 \Re _ { n } ( S _ { B } ) + 2 M \sqrt { \frac { \log ( 2 / \alpha ) } { 2 n } } \Bigg ] ^ { 1 / 2 } . } \end{array}
$$

Theorem 12 separates the capacity-dependent approximation term, the structural triangle-curl term, and finite-sample estimation error. Increasing network capacity reduces the first term but can increase the Rademacher complexity. The theorem leaves this trade-of explicit rather than prescribing depth and width as functions of $n .$ Unlike the lower bound in Theorem 11, these upper bounds do not require preference probabilities to be bounded away from zero and one.

For an approximate empirical minimizer with optimization error $\varepsilon _ { \mathrm { o p t } }$ , we have

$$
\widehat { \mathcal { L } } _ { n } ( f _ { \pi _ { \widehat { \theta , B } } } ) \leq \operatorname* { i n f } _ { \theta \in \Theta ( d , \bar { L } , \bar { N } ) } \widehat { \mathcal { L } } _ { n } ( f _ { \pi _ { \theta , B } } ) + \varepsilon _ { \mathrm { o p t } } ,
$$

and the same conclusions of Theorem 12 hold with $\varepsilon _ { \mathrm { o p t } }$ added to the excess-risk bound and inside the square brackets of the CPA bound.

## 5 Conclusion

We studied Direct Preference Optimization through the lens of statistical learning theory. Using standard consistency and calibration properties of logistic risk, we connected DPO’s objective to contextual preference accuracy, which measures agreement between preference judgments and policy reweighting relative to the reference. For score-diference parameterizations, we characterized approximation error through a capacitydependent score-approximation term and an intrinsic triangle-curl term, which vanishes exactly under the Bradley–Terry model. For ReLU score models, the capacity-dependent term can be made arbitrarily small, while triangle-curl yields an irreducible approximation barrier under additional boundedness assumptions. We also established finite-sample excess-risk bounds for clipped empirical risk minimizers, separating approximation, structural misspecification, and estimation error, and transferred these bounds to CPA through calibration.

Future work includes selecting network capacity as a function of sample size to obtain explicit convergence rates, developing richer preference models beyond score diferences that can represent nonzero trianglecurl, and constructing estimators of triangle-curl as a practical diagnostic for violations of the Bradley– Terry assumption. Finally, Assumption 8 excludes deterministic preferences by requiring square-integrable preference log-odds, and hence $\eta \in ( 0 , 1 )$ almost surely. An interesting direction is to relax this assumption, for example by adapting truncation techniques similar to those of Zhang et al. (2024).

## References

Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, Remi Munos, Mark Rowland, Michal Valko, and Daniele Calandriello. A general theoretical paradigm to understand learning from human preferences. In International Conference on Artificial Intelligence and Statistics, pp. 4447–4455. PMLR, 2024.

Peter L. Bartlett, Michael I. Jordan, and Jon D. McAulife. Convexity, classification, and risk bounds. Journal of the American Statistical Association, 101(473):138–156, 2006.

Ralph Allan Bradley and Milton E. Terry. Rank analysis of incomplete block designs: The method of paired comparisons. Biometrika, 39(3/4):324–345, 1952.

Angelica Chen, Sadhika Malladi, Lily H Zhang, Xinyi Chen, Qiuyi Zhang, Rajesh Ranganath, and Kyunghyun Cho. Preference learning algorithms do not learn preference rankings. Advances in Neural Information Processing Systems, 37:101928–101968, 2024.

Sayak Ray Chowdhury, Anush Kini, and Nagarajan Natarajan. Provably robust dpo: Aligning language models with noisy feedback. arXiv preprint arXiv:2403.00409, 2024.

Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

Stéphan Clémençon, Gábor Lugosi, and Nicolas Vayatis. Ranking and empirical minimization of u-statistics. 2008.

Jing Dong, Yaoliang Yu, and Pascal Pourpart. The representation-rationalizability tradeof in reward learning. arXiv preprint arXiv:2606.00291, 2026.

Wei Gao and Zhi-Hua Zhou. On the consistency of auc pairwise optimization. In Proceedings of the 24th International Conference on Artificial Intelligence, pp. 939–945, 2015.

Paul Gölz, Nika Haghtalab, and Kunhe Yang. Distortion of ai alignment: Does preference optimization optimize for preferences? Advances in Neural Information Processing Systems, 38:25969–26007, 2026.

Aditya Gopalan, Sayak Ray Chowdhury, and Debangshu Banerjee. Why dpo is a misspecified estimator and how to fix it. In International Conference on Learning Representations, volume 2026, pp. 88837–88847, 2026.

Jiwoo Hong, Noah Lee, and James Thorne. Orpo: Monolithic preference optimization without reference model. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 11170–11189, 2024.

Xiaoye Jiang, Lek-Heng Lim, Yuan Yao, and Yinyu Ye. Statistical ranking and combinatorial hodge theory. Mathematical Programming, 127(1):203–244, 2011.

Timo Kaufmann, Paul Weng, Viktor Bengs, and Eyke Hüllermeier. A survey of reinforcement learning from human feedback. arXiv preprint arXiv:2312.14925, 2023.

Nathan Lambert. Reinforcement learning from human feedback. arXiv preprint arXiv:2504.12501, 2025.

Jianfeng Lu, Zuowei Shen, Haizhao Yang, and Shijun Zhang. Deep network approximation for smooth functions. SIAM Journal on Mathematical Analysis, 53(5):5465–5506, 2021.

Yu Meng, Mengzhou Xia, and Danqi Chen. Simpo: Simple preference optimization with a reference-free reward. Advances in Neural Information Processing Systems, 37:124198–124235, 2024.

Thomas Nagler and Sophie Langer. Optimal neural network approximation of smooth compositional functions on sets with low intrinsic dimension. arXiv preprint arXiv:2602.03539, 2026.

Long Ouyang, Jefrey Wu, Xu Jiang, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022.

Rattana Pukdee, Maria-Florina Balcan, and Pradeep Ravikumar. What does preference learning recover from pairwise comparison data? arXiv preprint arXiv:2602.10286, 2026.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2023.

Johannes Schmidt-Hieber. Nonparametric regression using deep neural networks with relu activation function. 2020.

Nisan Stiennon, Long Ouyang, Jefrey Wu, et al. Learning to summarize with human feedback. Advances in Neural Information Processing Systems, 33:3008–3021, 2020.

Yiming Ying, Longyin Wen, and Siwei Lyu. Stochastic online auc maximization. In D. Lee, M. Sugiyama, U. Luxburg, I. Guyon, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 29. Curran Associates, Inc., 2016. URL https://proceedings.neurips.cc/paper\_files/paper/ 2016/file/c52f1bd66cc19d05628bd8bf27af3ad6-Paper.pdf.

Tong Zhang. Statistical behavior and consistency of classification methods based on convex risk minimization. The Annals of Statistics, 32(1):56–85, 2004.

Zihan Zhang, Lei Shi, and Ding-Xuan Zhou. Classification with deep neural networks and logistic loss. Journal of Machine Learning Research, 25(125):1–117, 2024.

Yao Zhao, Rishabh Joshi, Tianqi Liu, Misha Khalman, Mohammad Saleh, and Peter J Liu. Slic-hf: Sequence likelihood calibration with human feedback. arXiv preprint arXiv:2305.10425, 2023.

## A Omitted Proofs

For the fixed $\mathrm { t r i p l e } \ ( x , y , y ^ { \prime } )$ , define the conditional logistic risk $\begin{array} { r } { \varphi _ { \eta } ( u ) : = \eta \ell ( u ) + ( 1 - \eta ) \ell ( - u ) } \end{array}$ , where $\eta =$ $\eta ( x , y , y ^ { \prime } )$ and $u \in \mathbb { R }$ . We have $\mathcal { L } ( f ) = \mathbb { E } _ { X , Y , Y ^ { \prime } } \bigl [ \varphi _ { \eta ( X , Y , Y ^ { \prime } ) } \bigl ( f ( X , Y , Y ^ { \prime } ) \bigr ) \bigr ]$ . The conditional logistic risk provides a convenient pointwise characterization of the population risk. We similarly define the conditional CPA $P _ { \eta } ( u ) = \eta { \bf 1 } \{ u > 0 \} + \left( 1 - \eta \right) { \bf 1 } \{ u < 0 \} + \frac { 1 } { 2 } { \bf 1 } \{ u = 0 \}$ , and likewise $P ( f ) = \mathbb { E } _ { X , Y , Y ^ { \prime } } [ P _ { \eta ( X , Y , Y ^ { \prime } ) } ( f ( X , Y , Y ^ { \prime } ) ) ]$ . Let $P ^ { * } = \operatorname* { s u p } _ { f } P ( f )$ and $\begin{array} { r } { P _ { \eta } ^ { * } = \operatorname* { s u p } _ { v \in \mathbb { R } } P _ { \eta } ( v ) } \end{array}$ , where the first supremum is taken over all measurable triplet-wise functions.

## A.1 Proof of Lemma 1

Proof. Let $\eta = \eta ( x , y , y ^ { \prime } )$ and $u = f ( x , y , y ^ { \prime } )$ . Direct calculations show that the derivative of the conditional logistic risk is $\varphi _ { \eta } ^ { \prime } ( u ) = [ - \eta / ( 1 + \exp ( u ) ) + ( 1 - \eta ) / ( 1 + \exp ( - u ) ) ]$ ]. Setting $\varphi _ { n } ^ { \prime } ( u ) = 0$ gives $\eta / ( 1 + \exp ( u ) ) =$ $( 1 { - } \eta ) / ( 1 { + } \mathrm { e x p } ( \dot { - } u ) )$ . Let $a = \exp ( u ) > 0$ . Then $\eta / ( 1 { + } a ) = ( 1 { - } \eta ) / ( 1 { + } a ^ { - 1 } ) \stackrel { . } { = } ( 1 { - } \eta ) a / ( 1 { + } a )$ . Therefore a = $\eta / ( 1 - \eta )$ , and the stationary point $u ^ { * }$ is given by $u ^ { * } = \log { ( \eta / ( 1 - \eta ) ) }$ . It remains to verify that $u ^ { * }$ is indeed the minimizer. Diferentiating again gives $\varphi _ { \eta } ^ { \prime \prime } ( u ) = [ \eta \exp ( u ) / ( 1 + \exp ( u ) ) ^ { 2 } + ( 1 - \eta ) \exp ( - u ) / ( 1 + \exp ( - u ) ) ^ { 2 } \}$ For $\eta \in ( 0 , 1 )$ , this quantity is strictly positive for all finite u. Hence $\varphi _ { \eta }$ is strictly convex, and the stationary point is the unique minimizer.

If $\eta = 1$ , then $\varphi _ { \eta } ( u ) = \ell ( u )$ , which is minimized in the limiting sense by sending $u  + \infty$ . If $\eta = 0$ , then $\varphi _ { \eta } ( u ) = \ell ( - u )$ , which is minimized in the limiting sense by sending $u  - \infty$ . Taking the expectation over the pointwise minimizer $u ^ { * }$ gives the desired result. □

## A.2 Proof of Lemma 2

Proof. Let $\eta = \eta ( x , y , y ^ { \prime } )$ and $u = f ( x , y , y ^ { \prime } )$ . If $u > 0$ , then conditional CPA is η. If $u \ : < \ : 0$ , then the conditional CPA is $1 - \eta .$ . If $u = 0$ , the conditional CPA takes value $1 / 2$ . Therefore, the optimal conditional decision is $u > 0 { \mathrm { ~ i f ~ } } \eta > 1 / 2 , u < 0 { \mathrm { ~ i f ~ } } \eta < 1 / 2$ , and any tie-breaking decision is optimal if $\eta = 1 / 2$ . The optimal conditional value is $P _ { \eta } ^ { * } = \operatorname* { m a x } \{ \eta , 1 - \eta \}$ . Equivalently, on the set $\eta \neq 1 / 2$ , optimality is characterized by $\mathrm { s i g n } ( u ) = \mathrm { s i g n } \bigl ( \eta - \frac { 1 } { 2 } \bigr )$ . The lemma follows from the conditional CPA characterization of CPA. □

## A.3 Proof of Theorem 3

Proof. By Lemma 1, any population minimizer $g ^ { * }$ of the logistic risk satisfies $g ^ { * } = \log ( \eta / ( 1 - \eta ) )$ almost surely, with $( x , y , y ^ { \prime } )$ suppressed. Since the logarithm is strictly increasing, $\mathrm { s i g n } ( g ^ { * } ) = \mathrm { s i g n } ( \eta - { \textstyle \frac { 1 } { 2 } } )$ on the set $\eta \neq 1 / 2$ . Also, if $\eta = 1$ then $g ^ { * } ( x , y , y ^ { \prime } ) = + \infty$ and signs agree, likewise for $\eta = 0$ . By Lemma 2, this is precisely the necessary and suficient condition for maximizing the CPA. □

## A.4 Proof of Theorem 4

Lemma 13. Let $\eta = \eta ( x , y , y ^ { \prime } )$ and $q = \sigma ( u )$ . Then, $P _ { \eta } ^ { * } - P _ { \eta } ( u ) \leq 2 | \eta - q |$

Proof. Recall that $\begin{array} { r } { P _ { \eta } ( u ) = \eta \mathbf { 1 } \{ u > 0 \} + ( 1 - \eta ) \mathbf { 1 } \{ u < 0 \} + \frac { 1 } { 2 } \mathbf { 1 } \{ u = 0 \} } \end{array}$ , and from the proof of Lemma $2 ,$ $P _ { \eta } ^ { * } = \operatorname* { m a x } \{ \eta , 1 - \eta \}$ . Since σ is strictly increasing, $\textstyle u > 0 \iff q > { \frac { 1 } { 2 } }$ and $u < 0 \iff q < { \frac { 1 } { 2 } }$ . We consider three cases. Suppose $q > \frac { 1 } { 2 }$ . Then $u > 0$ , so $P _ { \eta } ( u ) = \eta$ . Therefore $P _ { \eta } ^ { * } - P _ { \eta } ( u ) = \operatorname* { m a x } \{ \eta , 1 - \eta \} - \eta _ { \mathit { \Pi } } $ which if $\eta \geq \frac { 1 } { 2 }$ , this is zero. Otherwise, $\begin{array} { r } { P _ { \eta } ^ { * } - P _ { \eta } ( u ) = 1 - 2 \eta = 2 \big ( \frac { 1 } { 2 } - \eta \big ) < 2 ( q - \eta ) = 2 | \eta - q | } \end{array}$ , since $q > \frac { 1 } { 2 }$ . When $q < \frac { 1 } { 2 }$ , the argument is analogous. If $\begin{array} { r } { q = \frac { 1 } { 2 } } \end{array}$ , then $u = 0$ , and therefore $\begin{array} { r } { P _ { \eta } ( u ) = \frac { 1 } { 2 } } \end{array}$ . Thus, $\begin{array} { r } { P _ { \eta } ^ { * } \tilde { \mathbf { \Xi } } ^ { * } P _ { \eta } ( u ) = \operatorname* { m a x } \tilde { \{ \eta , 1 - \eta \} } - \frac { 1 } { 2 } = \left| \eta - \frac { 1 } { 2 } \right| = \left| \eta - q \right| \leq \tilde { \mathbf { \Xi } } \left| \eta - q \right| } \end{array}$ |. Combining the three cases proves the claim. □

Let $\eta = \eta ( x , y , y ^ { \prime } ) , u = f ( x , y , y ^ { \prime } )$ , and $q = \sigma ( u )$ . The conditional logistic risk can be written as $\varphi _ { \eta } ( u ) =$ $- [ \eta \log q + ( 1 - \eta ) \log ( 1 - q ) ]$ . In the proof of Lemma 1, we show that the conditional logistic risk is minimized when $q = \eta$ . Therefore, in $\mathrm { f } _ { v } \varphi _ { \eta } ( v ) = - \Bigl \lceil \eta \log \eta + ( 1 - \eta ) \log ( 1 - \eta ) \Bigr \rceil$ , where the infimum is over $v \in \mathbb { R }$ . Subtracting gives $\varphi _ { \eta } ( u ) - \operatorname* { i n f } _ { v } \varphi _ { \eta } ( v ) = D _ { \mathrm { K L } } ^ { \ } ( \mathrm { B e r n } ( \eta ) \Vert \ \mathrm { B e r n } ( q ) ) = : \dot { K _ { \eta , q } } ^ { \ }$ . By Pinsker’s inequality, $| \eta - q | \le ( K _ { \eta , q } / 2 ) ^ { 1 / 2 }$ , and thus $| \eta - q | \le [ ( \varphi _ { \eta } ( u ) - \operatorname* { i n f } _ { v } \varphi _ { \eta } ( v ) ) / 2 ] ^ { 1 / 2 }$ . Combining the above inequality with Lemma 13, which says $P _ { \eta } ^ { * } - P _ { \eta } ( u ) \leq 2 | \eta - q | .$ we obtain $P _ { \eta } ^ { * } - P _ { \eta } ( u ) \leq 2 ^ { 1 / 2 } \left( \varphi _ { \eta } ( u ) - \operatorname* { i n f } _ { v } \varphi _ { \eta } ( v ) \right) ^ { 1 / 2 }$ Under the assumption now that $u = f ( X , Y , Y ^ { \prime } )$ , taking the expectation over $( X , Y , Y ^ { \prime } )$ gives $P ^ { * } - P ( f ) =$ $\begin{array} { r } { \mathbb { E } \big [ P _ { \eta } ^ { * } - P _ { \eta } ( u ) \big ] \leq 2 ^ { 1 / 2 } \mathbb { E } \left[ \big ( \varphi _ { \eta } ( u ) - \operatorname* { i n f } _ { v } \varphi _ { \eta } ( v ) \big ) ^ { 1 / 2 } \right] } \end{array}$ . Applying Jensen’s inequality to the right hand side and using $\mathbb { E } [ \varphi _ { \eta } ( u ) ] = \mathcal { L } ( f )$ and $\mathbb { E } [ \operatorname* { i n f } _ { v } \varphi _ { \eta } ( v ) ] = \mathcal { L } ^ { * }$ proves Theorem 4.

## A.5 Proof of Lemma 6

Proof. Suppose $( \operatorname { c u r l } f ) ( x ; y _ { 1 } , y _ { 2 } , y _ { 3 } ) = 0$ for all $x , y _ { 1 } , y _ { 2 } , y _ { 3 }$ . Fix an arbitrary reference response $y _ { 0 } \in Y$ and define $s ( x , y ) : = f ( x , y , y _ { 0 } )$ . Then $f ( x , y , y ^ { \prime } ) + f ( x , y ^ { \prime } , y _ { 0 } ) + f ( x , y _ { 0 } , y ) = 0$ . Using antisymmetry, $f ( x , y _ { 0 } , y ) =$ $- f ( x , y , y _ { 0 } )$ so that $f ( x , y , y ^ { \prime } ) = s ( x , y ) - s ( x , y ^ { \prime } )$ . Conversely, if $f ( x , y , y ^ { \prime } ) = s ( x , y ) - s ( x , y ^ { \prime } )$ , then scores telescope to give curl $f = 0$ □

## A.6 Proof of Theorem 7

Proof. Since $\mathcal { T } ( f ) = \mathbb { E } \big [ ( \operatorname { c u r l } f ) ^ { 2 } \big ] , \mathcal { T } ( f ) = 0$ if and only if curl $f = 0$ almost surely.

$( 1 \Rightarrow 2 )$ . Assume $\textstyle { \mathcal { T } } ( f ) = 0$ . By Fubini’s theorem, for almost every x, (curl $f ) ( x ; y _ { 1 } , y _ { 2 } , y _ { 3 } ) = 0$ for $p ( \cdot \mid x ) ^ { 3 } .$ almost every $( y _ { 1 } , y _ { 2 } , y _ { 3 } )$ . Define $s ( x , y ) : = \mathbb { E } _ { U \sim p ( \cdot | x ) } \big [ f ( x , y , U ) \big ]$ . Fix an x such that (curl $f ) ( x ; y , y ^ { \prime } , u ) = 0$ for $p ( \cdot \mid x ) ^ { 3 } .$ -almost every $( y , y ^ { \prime } , u )$ . Using antisymmetry, $f ( x , u , y ) { \overset { \cdot } { = } } - f ( x , y , u )$ , so that $f ( x , y , y ^ { \prime } ) = f ( x , y , u ) -$ $f ( x , y ^ { \prime } , u )$ . Taking expectations over $U \sim p ( \cdot \mid x ) , f ( x , y , y ^ { \prime } ) = \mathbb { E } [ f ( x , y , U ) ] - \mathbb { E } [ f ( x , y ^ { \prime } , U ) ] = s ( x , y ) - s ( x , y ^ { \prime } )$ almost surely.

$( \mathcal { Z } \Rightarrow \mathit { 1 } )$ . Conversely, suppose there exists a measurable function s such that $f ( x , y , y ^ { \prime } ) = s ( x , y ) - s ( x , y ^ { \prime } )$ almost surely. Then score diferences telescope to give curl $f = 0$ almost surely. □

## A.7 Proof of Theorem 9

We begin with a few lemmas from which the result will follow easily.

Lemma 14. Let S be a class of score functions. There holds

$$
\operatorname* { i n f } _ { s \in \mathcal { S } } \mathcal { L } ( \Delta _ { s } ) - \mathcal { L } ^ { * } \leq \frac { 1 } { 8 } \operatorname* { i n f } _ { s \in \mathcal { S } } \mathbb { E } _ { X , Y , Y ^ { \prime } } \left[ \left( \Delta _ { s } ( X , Y , Y ^ { \prime } ) - f ^ { * } ( X , Y , Y ^ { \prime } ) \right) ^ { 2 } \right] .
$$

Moreover, suppose that for every $s \in \mathcal { S } , \delta < \sigma ( \Delta _ { s } ( X , Y , Y ^ { \prime } ) ) < 1 - \delta$ and $\delta < \sigma ( f ^ { * } ( X , Y , Y ^ { \prime } ) ) < 1 - \delta$ for some $\delta \in ( 0 , 1 / 2 )$ almost surely.

$$
\operatorname* { i n f } _ { s \in \mathcal { S } } \mathcal { L } ( \Delta _ { s } ) - \mathcal { L } ^ { * } \geq \frac { \delta ^ { 2 } } { 2 } \operatorname* { i n f } _ { s \in \mathcal { S } } \mathbb { E } _ { X , Y , Y ^ { \prime } } \left[ \left( \Delta _ { s } ( X , Y , Y ^ { \prime } ) - f ^ { * } ( X , Y , Y ^ { \prime } ) \right) ^ { 2 } \right] .
$$

Proof. Fix the triple $( x , y , y ^ { \prime } )$ and write $\eta = \eta ( x , y , y ^ { \prime } )$ . Using the conditional logistic risk, for any score function s, $\mathcal { L } ( \Delta _ { s } ) = \mathbb { E } \left[ \varphi _ { \eta ( X , Y , Y ^ { \prime } ) } \big ( \Delta _ { s } ( X , Y , Y ^ { \prime } ) \big ) \right]$ . Since $\varphi _ { \eta } ^ { \prime } ( u ) = \sigma ( u ) - \eta , \varphi _ { \eta }$ is minimized at $f ^ { * } ( x , y , y ^ { \prime } ) =$ $\log { \frac { \eta ( x , y , y ^ { \prime } ) } { 1 - \eta ( x , y , y ^ { \prime } ) } }$ . Hence $\mathcal { L } ^ { * } = \mathbb { E } \left[ \varphi _ { \eta ( X , Y , Y ^ { \prime } ) } \big ( f ^ { * } ( X , Y , Y ^ { \prime } ) \big ) \right]$ . For every $s \in S$ , subtracting the last two identities gives

$$
\mathcal { L } ( \Delta _ { s } ) - \mathcal { L } ^ { * } = \mathbb { E } \left[ \varphi _ { \eta ( X , Y , Y ^ { \prime } ) } \big ( \Delta _ { s } ( X , Y , Y ^ { \prime } ) \big ) - \varphi _ { \eta ( X , Y , Y ^ { \prime } ) } \big ( f ^ { * } ( X , Y , Y ^ { \prime } ) \big ) \right] .
$$

A direct calculation gives $\begin{array} { r } { \frac { d } { d u } \varphi _ { \eta } ( u ) = \sigma ( u ) - \eta } \end{array}$ and $\begin{array} { r } { \frac { d ^ { 2 } } { d u ^ { 2 } } \varphi _ { \eta } ( u ) = \sigma ( u ) ( 1 - \sigma ( u ) ) \le \frac { 1 } { 4 } } \end{array}$ . Thus $\varphi _ { \eta }$ is 1/4-smooth. Since $f ^ { * } ( x , y , y ^ { \prime } )$ is the minimizer of $\begin{array} { r } { \varphi _ { \eta } , \varphi _ { \eta } ( \Delta _ { s } ( x , y , y ^ { \prime } ) ) - \varphi _ { \eta } ( f ^ { * } ( x , y , y ^ { \prime } ) ) \leq \big ( \Delta _ { s } ( x , y , y ^ { \prime } ) - f ^ { * } ( x , y , y ^ { \prime } ) \big ) ^ { 2 } / 8 . } \end{array}$ Taking expectations and then the infimum over $s \in S$ proves the upper bound.

For the lower bound, suppose that for every $s \in \mathcal { S } , \delta < \sigma ( \Delta _ { s } ( X , Y , Y ^ { \prime } ) ) < 1 - \delta$ and $\delta < \sigma ( f ^ { \ast } ( X , Y , Y ^ { \prime } ) ) < 1 -$ δ almost surely. Since σ is monotone, for every t between $\Delta _ { s } ( x , y , y ^ { \prime } )$ and $f ^ { * } ( x , y , y ^ { \prime } )$ we have $\delta < \sigma ( t ) < 1 - \delta$ Hence $\begin{array} { r } { \frac { d ^ { 2 } } { d t ^ { 2 } } \varphi _ { \eta } ( t ) = \sigma ( t ) ( 1 - \sigma ( t ) ) \geq \delta ( 1 - \delta ) \geq \delta ^ { 2 } } \end{array}$ . Because $f ^ { * } ( x , y , y ^ { \prime } )$ minimizes $\varphi _ { \eta } .$ Taylor’s theorem gives $\begin{array} { r } { \varphi _ { \eta } ( \Delta _ { s } ( x , y , y ^ { \prime } ) ) - \varphi _ { \eta } ( f ^ { * } ( x , y , y ^ { \prime } ) ) \geq \frac { \delta ^ { 2 } } { 2 } \big ( \Delta _ { s } ( x , y , y ^ { \prime } ) - f ^ { * } ( x , y , y ^ { \prime } ) \big ) ^ { 2 } } \end{array}$ . Taking expectations and then the infimum over $s \in S$ proves the lower bound. □

Lemma 15. Let Assumption 8 hold. Among all square-integrable measurable score functions, the best $L _ { 2 }$ score-diference approximation to $f ^ { * }$ is induced by $s ^ { * }$ . Equivalently, for the joint distribution $P _ { X , Y , Y ^ { \prime } } , \Delta _ { s ^ { * } }$ is the $L _ { 2 } ( P _ { X , Y , Y ^ { \prime } } )$ projection of $f ^ { * }$ onto the space of all score diferences. Consequently, for any score class ${ \mathbf { } } S _ { ; }$

$$
\begin{array} { r } { \underset { s \in \mathcal { S } } { \operatorname* { i n f } } \mathbb { E } \left[ \left( \Delta _ { s } ( X , Y , Y ^ { \prime } ) - f ^ { * } ( X , Y , Y ^ { \prime } ) \right) ^ { 2 } \right] = \underset { s \in \mathcal { S } } { \operatorname* { i n f } } \mathbb { E } \left[ \left( \Delta _ { s } ( X , Y , Y ^ { \prime } ) - \Delta _ { s ^ { * } } ( X , Y , Y ^ { \prime } ) \right) ^ { 2 } \right] } \\ { + \mathbb { E } \left[ \left( \Delta _ { s ^ { * } } ( X , Y , Y ^ { \prime } ) - f ^ { * } ( X , Y , Y ^ { \prime } ) \right) ^ { 2 } \right] . } \end{array}\tag{6}
$$

Proof. Let $R ( x , y , y ^ { \prime } ) : = f ^ { * } ( x , y , y ^ { \prime } ) - \Delta _ { s ^ { * } } ( x , y , y ^ { \prime } )$ . By reciprocity of preferences, $f ^ { * }$ is anti-symmetric: $f ^ { * } ( x , y ^ { \prime } , y ) = - f ^ { * } ( x , y , y ^ { \prime } )$ . Since $\Delta _ { s ^ { * } }$ ∗ is also anti-symmetric, R is anti-symmetric. $\mathrm { B y }$ antisymmetry of $f ^ { * }$ in its responses and the definition of $s ^ { * }$ , we get $\mathbb { E } _ { Y \sim p ( \cdot | x ) } s ^ { * } ( x , Y ) = 0$ , and consequently,

$$
\mathbb { E } _ { U \sim p ( \cdot | x ) } R ( x , y , U ) = \mathbb { E } _ { U } f ^ { * } ( x , y , U ) - s ^ { * } ( x , y ) + \mathbb { E } _ { U } s ^ { * } ( x , U ) = 0 .
$$

By anti-symmetry, $\mathbb { E } _ { U } R ( x , U , y ) = 0$ as well. Now take any square-integrable score s. Then

$$
\begin{array} { r } { \mathbb { E } \left[ R ( X , Y , Y ^ { \prime } ) \big ( s ( X , Y ) - s ( X , Y ^ { \prime } ) \big ) \right] = \mathbb { E } \left[ R ( X , Y , Y ^ { \prime } ) s ( X , Y ) \right] - \mathbb { E } \left[ R ( X , Y , Y ^ { \prime } ) s ( X , Y ^ { \prime } ) \right] . } \end{array}
$$

Using the exchangeability of $Y$ and $Y ^ { \prime }$ conditional on X and the anti-symmetry of R, the second term equals the negative of the first. Therefore the previous display equals $2 \mathbb { E } [ R ( X , Y , Y ^ { \prime } ) s ( X , Y ) ] = 0$ , where the last equality follows from $\mathbb { E } _ { Y ^ { \prime } | X , Y } R ( X , Y , Y ^ { \prime } ) = 0$ . Hence $f ^ { * } - \Delta _ { s ^ { \prime } }$ ∗ is orthogonal to every score diference. This proves that $\Delta _ { s ^ { * } }$ is the $\tilde { L _ { 2 } } ( P _ { X , Y , Y ^ { \prime } } )$ projection of $f ^ { * }$ onto the score-diference subspace. The Pythagorean identity gives equation 6. □

Lemma 16. Let Assumption 8 hold, and let $Y _ { 1 } , Y _ { 2 } , Y _ { 3 }$ be conditionally i.i.d. given X. Then

$$
\mathbb { E } \left[ \left( \Delta _ { s ^ { * } } ( X , Y , Y ^ { \prime } ) - f ^ { * } ( X , Y , Y ^ { \prime } ) \right) ^ { 2 } \right] = \frac { 1 } { 3 } \mathbb { E } _ { X , Y _ { 1 } , Y _ { 2 } , Y _ { 3 } } \left[ ( \operatorname { c u r l } f ^ { * } ) ( X ; Y _ { 1 } , Y _ { 2 } , Y _ { 3 } ) ^ { 2 } \right] .
$$

Proof. Let $R ( x , y , y ^ { \prime } ) : = f ^ { * } ( x , y , y ^ { \prime } ) - \Delta _ { s ^ { * } } ( x , y , y ^ { \prime } )$ . From the proof of Lemma 15, R is anti-symmetric and satisfies $\mathbb { E } _ { U \sim p ( \cdot | x ) } R ( x , y , U ) = 0$ and $\mathbb { E } _ { U \sim p ( \cdot | x ) } R ( x , U , y ) = 0$ . The combinatorial curl of every score-diference function telescopes to zero. Therefore,

$$
( \operatorname { c u r l } f ^ { * } ) ( X ; Y _ { 1 } , Y _ { 2 } , Y _ { 3 } ) = R ( X , Y _ { 1 } , Y _ { 2 } ) + R ( X , Y _ { 2 } , Y _ { 3 } ) + R ( X , Y _ { 3 } , Y _ { 1 } ) .\tag{7}
$$

Expanding the square of the right-hand side of equation 7, the three squared terms have the same expectation. The cross terms vanish:

$$
\mathbb { E } \left[ R ( X , Y _ { 1 } , Y _ { 2 } ) R ( X , Y _ { 2 } , Y _ { 3 } ) \right] = \mathbb { E } _ { X , Y _ { 2 } } \left[ \mathbb { E } _ { Y _ { 1 } | X , Y _ { 2 } } R ( X , Y _ { 1 } , Y _ { 2 } ) \mathbb { E } _ { Y _ { 3 } | X , Y _ { 2 } } R ( X , Y _ { 2 } , Y _ { 3 } ) \right] = 0 .
$$

The other two cross terms are identical. Hence

$$
{  { \mathbb E } } \left[ \left( R ( X , Y _ { 1 } , Y _ { 2 } ) + R ( X , Y _ { 2 } , Y _ { 3 } ) + R ( X , Y _ { 3 } , Y _ { 1 } ) \right) ^ { 2 } \right] = 3 {  { \mathbb E } } \left[ \left( \Delta _ { s ^ { * } } ( X , Y , Y ^ { \prime } ) - f ^ { * } ( X , Y , Y ^ { \prime } ) \right) ^ { 2 } \right] .
$$

Combining this identity with equation 7 proves the claim.

The proof of Theorem 9 is completed by plugging Lemma 15 and Lemma 16 into Lemma 14.

## A.8 Proof of Theorem 11

Proof. Each $\theta \in \Theta ( d , L , N )$ induces a score function $s _ { \theta } \in \mathcal { S } ( d , L , N )$ , and every $s \in \mathcal { S } ( d , L , N )$ is realized by some $\theta \in \Theta ( d , L , N )$ , so the induced triplet-wise hypothesis class satisfies $\{ f _ { \pi _ { \theta } } : \theta \in \Theta ( d , L , N ) \} =$ $\Delta S ( d , L , N )$ . Thus, $\begin{array} { r } { \operatorname* { i n f } _ { \theta \in \Theta ( d , L , N ) } \mathcal { L } ( f _ { \pi _ { \theta } } ) = \operatorname* { i n f } _ { s \in \mathcal { S } ( d , L , N ) } \mathcal { L } ( \Delta _ { s } ) } \end{array}$ , and it sufices to work at the level of score functions.

Apply Lemma 14 with score class ${ \cal S } ( d , \bar { L } , \bar { N } )$ followed by Lemma 15 to obtain

$$
\begin{array} { r l r } {  { \operatorname* { i n f } _ { s \in S ( d , \bar { L } , \bar { N } ) } \big [ \mathcal { L } ( \Delta _ { s } ) - \mathcal { L } ^ { * } \big ] \le \frac { 1 } { 8 } \operatorname* { i n f } _ { s \in S ( d , \bar { L } , \bar { N } ) } \mathbb { E } [ \big ( \Delta _ { s } ( X , Y , Y ^ { \prime } ) - \Delta _ { s ^ { * } } ( X , Y , Y ^ { \prime } ) \big ) ^ { 2 } ] } } \\ & { } & { + \frac { 1 } { 8 } \mathbb { E } \Big [ \big ( \Delta _ { s ^ { * } } ( X , Y , Y ^ { \prime } ) - f ^ { * } ( X , Y , Y ^ { \prime } ) \big ) ^ { 2 } \Big ] } \end{array}\tag{8}
$$

For any score $s ,$ write $e _ { s } ( x , y ) = s ( x , y ) - s ^ { * } ( x , y )$ . Since $Y$ and $Y ^ { \prime }$ are conditionally i.i.d. given $X$

$$
\begin{array} { r l } & { \mathbb { E } _ { X , Y , Y ^ { \prime } } \left[ \left( \Delta _ { s } ( X , Y , Y ^ { \prime } ) - \Delta _ { s ^ { * } } ( X , Y , Y ^ { \prime } ) \right) ^ { 2 } \right] = 2 \mathbb { E } _ { X , Y } \left[ e _ { s } ( X , Y ) ^ { 2 } \right] - 2 \mathbb { E } _ { X } \left[ \left( \mathbb { E } _ { Y | X } [ e _ { s } ( X , Y ) ] \right) ^ { 2 } \right] } \\ & { \qquad \leq 2 \mathbb { E } _ { X , Y } \left[ \left( s ( X , Y ) - s ^ { * } ( X , Y ) \right) ^ { 2 } \right] . } \end{array}
$$

By Theorem 3.1 in Nagler & Langer (2026), there exist constants $C _ { 1 } , C _ { \mathrm { a p p } } > 0$ such that, for all $L _ { 0 } , N _ { 0 } \ge 2$ , there exists $\hat { s } \in \mathcal { S } ( d , \bar { L } , \bar { N } )$ satisfying $\begin{array} { r } { \operatorname* { s u p } _ { ( x , y ) \in \mathcal { X } \times \mathcal { Y } } | \hat { s } ( x , y ) - s ^ { * } ( \stackrel { \cdot \cdot } { x } , y ) | \leq C _ { \mathrm { a p p } } ( 1 / ( L _ { 0 } ^ { 2 } N _ { 0 } ^ { 2 } ) ) ^ { \beta / d } } \end{array}$ Thus, in $\begin{array} { r } { \mathrm { f } _ { s \in S ( d , \bar { L } , \bar { N } ) } \mathbb { E } \left[ \left( \Delta _ { s } ( X , Y , Y ^ { \prime } ) - \Delta _ { s ^ { * } } ( X , Y , Y ^ { \prime } ) \right) ^ { 2 } \right] \leq 2 C _ { \mathrm { a p p } } ^ { 2 } \left( 1 / L _ { 0 } ^ { 2 } N _ { 0 } ^ { 2 } \right) ^ { 2 \beta / d } } \end{array}$ . Combining this bound and Lemma 16 with (8) proves the upper bound for the unclipped score class after setting $C _ { 2 } = C _ { \mathrm { a p p } } ^ { 2 } / 4$ into the constant. Choosing $L _ { 0 } \asymp N _ { 0 } \asymp \epsilon ^ { - d / ( 8 \beta ) }$ makes the first term of order ϵ, and gives the stated ϵ-form of the bound.

For the clipped parameterization, choose $\hat { \theta } \in \Theta ( d , \bar { L } , \bar { N } )$ such that $\begin{array} { r } { s _ { \hat { \theta } } = \hat { s } , } \end{array}$ , and set $\hat { s } _ { B } = \mathrm { c l i p } _ { B } \circ \hat { s }$ . By Assumption 10, $| s ^ { * } ( x , y ) | \leq B$ , so clipping gives

$$
| \hat { s } _ { B } ( x , y ) - s ^ { * } ( x , y ) | \leq | \hat { s } ( x , y ) - s ^ { * } ( x , y ) | .
$$

Consequently, $\begin{array} { r } { \operatorname* { s u p } _ { ( x , y ) \in \mathcal { X } \times \mathcal { Y } } | \hat { s } _ { B } ( x , y ) - s ^ { * } ( x , y ) | \leq C _ { \mathrm { a p p } } \left( 1 / ( L _ { 0 } ^ { 2 } N _ { 0 } ^ { 2 } ) \right) ^ { \beta / d } } \end{array}$ . The preceding conditional-i.i.d. cal culation yields $\begin{array} { r } { \mathbb { E } \left[ \left( \Delta _ { \hat { s } _ { B } } - \Delta _ { s ^ { * } } \right) ^ { 2 } \right] \leq 2 C _ { \mathrm { a p p } } ^ { 2 } \left( 1 / ( L _ { 0 } ^ { 2 } N _ { 0 } ^ { 2 } ) \right) ^ { 2 \beta / d } } \end{array}$ . The upper bound statements for the clipped score class then follow in the same way as the unclipped score class argument.

For the lower bound, fix $0 < \delta < \mathrm { m i n } \{ a , \sigma ( - 2 B ) \}$ }. For every $\theta \in \Theta ( d , \bar { L } , \bar { N } ) , | s _ { \theta , B } | \leq B$ , and hence $| f _ { \pi _ { \theta , B } } | \leq 2 B$ . Thus,

$$
\delta < \sigma ( - 2 B ) \leq \sigma ( f _ { \pi _ { \theta , B } } ) \leq \sigma ( 2 B ) = 1 - \sigma ( - 2 B ) < 1 - \delta .
$$

Moreover,

$$
\delta < a \leq \sigma ( f ^ { * } ) = \eta \leq 1 - a < 1 - \delta \qquad \mathrm { a l m o s t ~ s u r e l y } .
$$

The lower-bound assumptions of Theorem 9 therefore hold uniformly over the clipped class, proving the stated lower bound. □

## A.9 Proof of Theorem 12

Define $\Re _ { n } ( \Delta S _ { B } )$ analogously using the inputs $( X _ { i } , Y _ { i } , Y _ { i } ^ { \prime } )$ . Since (X, Y ) and $( X , Y ^ { \prime } )$ have the same marginal distribution, symmetry of the Rademacher signs gives

$$
\begin{array} { r l } & { \Re _ { n } ( \Delta S _ { B } ) = \mathbb { E } \left[ \underset { s \in \mathcal { S } _ { B } } { \operatorname* { s u p } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \epsilon _ { i } \big ( s ( X _ { i } , Y _ { i } ) - s ( X _ { i } , Y _ { i } ^ { \prime } ) \big ) \right] } \\ & { \qquad \le \mathbb { E } \left[ \underset { s \in \mathcal { S } _ { B } } { \operatorname* { s u p } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \epsilon _ { i } s ( X _ { i } , Y _ { i } ) \right] + \mathbb { E } \left[ \underset { s \in \mathcal { S } _ { B } } { \operatorname* { s u p } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( - \epsilon _ { i } ) s ( X _ { i } , Y _ { i } ^ { \prime } ) \right] } \\ & { \qquad = 2 \Re _ { n } ( \mathcal { S } _ { B } ) . } \end{array}
$$

Let $\ell ( u ) = \log ( 1 + e ^ { - u } )$ , and define the loss class

$$
\mathcal { G } : = \{ g _ { f } : ( x , y , y ^ { \prime } , a ) \mapsto \ell ( a f ( x , y , y ^ { \prime } ) ) \mid f \in \Delta S _ { B } \} .
$$

Every $f \in \Delta S _ { B }$ satisfies $| f | \le 2 B$ , so every $g \in { \mathcal { G } }$ takes values in $[ 0 , M ]$ . Moreover, $| \ell ^ { \prime } ( u ) | \leq 1$ , and hence ℓ is 1-Lipschitz. Applying the Rademacher contraction inequality gives

$$
\mathfrak { R } _ { n } ( \mathcal { G } ) \leq \mathbb { E } \left[ \operatorname* { s u p } _ { f \in \Delta S _ { B } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \epsilon _ { i } A _ { i } f ( X _ { i } , Y _ { i } , Y _ { i } ^ { \prime } ) \right] .
$$

Conditional on the observations, the signs $\epsilon _ { i } A _ { i }$ are independent uniform signs. Consequently,

$$
\Re _ { n } ( { \mathcal { G } } ) \leq \Re _ { n } ( \Delta S _ { B } ) \leq 2 \Re _ { n } ( S _ { B } ) .
$$

Define the one-sided deviations

$$
U _ { + } : = \operatorname* { s u p } _ { f \in \Delta S _ { B } } \big ( \mathcal { L } ( f ) - \widehat { \mathcal { L } } _ { n } ( f ) \big ) , \qquad U _ { - } : = \operatorname* { s u p } _ { f \in \Delta S _ { B } } \big ( \widehat { \mathcal { L } } _ { n } ( f ) - \mathcal { L } ( f ) \big ) .
$$

Standard symmetrization yields

$$
\begin{array} { r } { \mathbb { E } U _ { + } \leq 2 \Re _ { n } ( \mathcal { G } ) , \qquad \mathbb { E } U _ { - } \leq 2 \Re _ { n } ( \mathcal { G } ) . } \end{array}
$$

Since $0 \leq g \leq M$ , replacing one observation changes either $U _ { + }$ or $U _ { - }$ by at most $M / n$ . McDiarmid’s inequality therefore gives, for $t > 0$

$$
\mathrm { P r } \left( U _ { \pm } > \mathbb { E } U _ { \pm } + t \right) \le \exp \left( - \frac { 2 n t ^ { 2 } } { M ^ { 2 } } \right) .
$$

Taking $t = M \sqrt { \log ( 2 / \alpha ) / ( 2 n ) }$ and applying a union bound shows that, with probability at least $1 - \alpha .$

$$
U _ { n } : = \operatorname* { s u p } _ { f \in \Delta S _ { B } } \left. \mathcal { L } ( f ) - \widehat { \mathcal { L } } _ { n } ( f ) \right. = \operatorname* { m a x } \{ U _ { + } , U _ { - } \} \leq 2 \Re _ { n } ( \mathcal { G } ) + M \sqrt { \frac { \log ( 2 / \alpha ) } { 2 n } } \leq 4 \Re _ { n } ( S _ { B } ) + M \sqrt { \frac { \log ( 2 / \alpha ) } { 2 n } } .\tag{9}
$$

On this event, the empirical minimization property implies that, for every $\theta \in \Theta ( d , \bar { L } , \bar { N } )$ ,

$$
\begin{array} { r } { \mathcal { L } ( f _ { \widehat { \boldsymbol { \theta } _ { \beta , B } } } ) \leq \widehat { \mathcal { L } } _ { n } ( f _ { \boldsymbol { \pi } _ { \widehat { \boldsymbol { \theta } _ { \beta , B } } } } ) + U _ { n } \leq \widehat { \mathcal { L } } _ { n } ( f _ { \boldsymbol { \pi } _ { \boldsymbol { \theta } , B } } ) + U _ { n } \leq \mathcal { L } ( f _ { \boldsymbol { \pi } _ { \boldsymbol { \theta } , B } } ) + 2 U _ { n } . } \end{array}
$$

Taking the infimum over θ gives

$$
{ \mathcal L } ( f _ { \pi _ { \widehat { \theta , B } } } ) - { \mathcal L } ^ { * } \leq \operatorname* { i n f } _ { \theta \in \Theta ( d , \bar { L } , \bar { N } ) } \big [ { \mathcal L } ( f _ { \pi _ { \theta , B } } ) - { \mathcal L } ^ { * } \big ] + 2 U _ { n } .
$$

Combining this inequality with the clipped upper bound in Theorem 11 and equation (9) proves the asserted excess-risk bound.

Finally, Theorem 4 implies

$$
P ^ { * } - P ( f _ { \widehat { \theta , B } } ) \leq \sqrt { 2 } \left[ \mathcal { L } ( f _ { \widehat { \theta , B } } ) - \mathcal { L } ^ { * } \right] ^ { 1 / 2 } ,
$$

which proves the CPA bound on the same event.