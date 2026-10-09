# Interval-valued SHAP in Tree-Based Models

Chenrui Zhu<sup>∗</sup>, Vu-Linh Nguyen<sup>∗</sup>, Marie-Hel´ ene Masson\` <sup>∗†</sup>, Sebastien Destercke´ <sup>∗</sup>

∗ Universite de technologie de Compi ´ egne, UMR-CNRS 7253 Heudiasyc\`

Compiegne, France \`

<sup>†</sup> IUT de l’Oise, Universite de Picardie Jules Verne´

Beauvais, France

{chenrui.zhu, vu-linh.nguyen, mylene.masson, sebastien.destercke}@hds.utc.fr

Abstract—Shapley values are among the most popular featureattribution explanations. Efficient approaches for computing/estimating Shapley values for tree-based models, which are state-of-the-art for tabular data sets, have been developed. However, it is known that Shapley values can be (highly) unrobust due to small and realistic changes. In this paper, we propose an imprecise Dirichlet model (IDM) based method to analyze the robustness of Shapley values in decision trees and random forests. Technically, it is done by quantifying and analyzing the interval-valued Shapley values when a few unannotated instances are randomly introduced to the leaves of the trees. The intervalvalued Shapley values can be defined following common principles in handling incomplete data: the pessimistic and averaging principles. We derive various theoretical results that lead to efficient computation of the interval-valued Shapley values. We also show that the proposed method can be straightforwardly generalized to the case of Banzhaf values. We then present various case studies and experiments to illustrate the behaviour of the proposed interval-valued Shapley values and their applications in debiasing uninformative features.

Index Terms—Explainable AI, robust explanations, intervalvalued attributions, tree-based models

## I. INTRODUCTION

Explanation methods aim to make predictive models more transparent by providing human-interpretable summaries of their behavior. Feature-attribution explanations are among the most popular explanations. For each instance, they assign to each feature a real-valued score quantifying its attribution, i.e., its contribution to the prediction given by the predictive model. The most popular feature-attribution explanations are SHapley Additive exPlanations (SHAP), whose attributions are characterized by axioms such as local accuracy and consistency [1], [2]. Efficient approaches, notably TreeSHAP [1], [3], for computing Shapley values for tree-based models, which are state-of-the-art for tabular data sets [4], have been developed. This has led to the broad adoption of Shapley-style explanations for tree-based models, notably gradient-boosted trees and random forests [1], [3].

However, it is known that Shapley values can be fragile, e.g., small, realistic input perturbations can cause large shifts in the attributions [5]. Their fragility would be further amplified in the case of tree-based models due to the fact that small perturbations to the training data can significantly change the trees [6], and thus the attributions. These have led to various attempts to assess the robustness of Shapley values for treebased models [5], [7]. A notable attempt is the imprecise SHAP [7], developed for imprecise-probability classifiers. The approach starts by assuming that the classifiers predict credal sets and employs the KL divergence (instead of the difference) to define the Shapley values. Imprecise outputs of the classifiers naturally lead to interval-valued Shapley values, which essentially reflect the uncertainty (due to insufficient training data). While this is a generic approach with interesting theoretical properties, it is unclear how to leverage it to assess the robustness of Shapley values for tree-based models. In particular, due to differences in their notions of Shapley values, it is unclear under which conditions imprecise Shapley values can cover the Shapley values of tree-based models, which might be a crucial requirement for robustness indicators.

In this paper, we focus on assessing the robustness of Shapley values when the stability of the tree-based models is of utmost importance, notably for the purpose of interpretability. Note that in safety-critical applications, such as in the medical domain [8], [9], understanding the structure of the trees is important. In other words, it might be advisable to guarantee the model stability unless additional data coming at the deployment phase suggests the need for model retraining, e.g., due to unsatisfactory generalization ability. This is because understanding the structure of the trees can be costly, but it can enhance the trustworthiness of the decision support system. Model stability in random forests is also important for interpretability, notably when the structures of its trees are leveraged to extract rules [10].

Note that with fixed structures, the Shapley values for some individuals can also be very fragile due to small perturbations to training data. This will be confirmed in our empirical study. This naturally calls for methods for assessing the Shapley values for the individuals, which should compactly present, for each instance, the possible changes in all the attributions. To serve this purpose, we propose an imprecise Dirichlet model (IDM) [11] based method to analyze the robustness of Shapley values in decision trees and random forests. The overall idea is to associate the robustness of each Shapley value with an interval-valued Shapley value, which covers the original value, induced by introducing a small number of unannotated instances into the training data. This can be implemented following common principles in handling incomplete data: the pessimistic and averaging principles, see, e.g., [12] and references therein. The corresponding problems of computing interval-valued Shapley values are combinatorial optimization problems, where Brute-force search is impractical.

We present various theoretical results that allow efficient computation of interval-valued Shapley values in the case of decision trees. For each instance, we show that the Shapley value associated with each feature can be expressed as a linear combination of the leaf values. The efficiency of the averaging principle further calls for efficiency in computing expected values of multinomial distributions. The efficiency of the pessimistic principle is achieved by leveraging the linearity of the Shapley values and the property that, for each leaf, the lower (upper) leaf value provided by the IDM model is a convex decreasing (concave increasing) function in the number of unannotated instances. We then show that the mentioned results can be extended to the case of Banzhaf values straightforwardly. Finally, we extend the results achieved for decision trees to random forests.

After recalling the basics of Shapley values and tree-based models in Section II, Section III presents our IDM-based method for deriving interval-valued Shapley values. Section IV presents various case studies and experiments to illustrate the behaviour of the proposed interval-valued Shapley values and their applications in debiasing uninformative features [13]. Finally, Section V concludes this work and sketches out potential follow-up work.

## II. BACKGROUND

We consider decision-tree predictors and game-theoretic feature-attribution explanations. $\mathcal { U } : = \{ 1 , \ldots , n \}$ denotes the set of feature indices and $\mathcal { X } \subseteq \mathbb { R } ^ { n }$ the input space. We fix an instance $\pmb { x } = ( x _ { 1 } , \dots , x _ { n } ) \in \pmb { \chi }$ to be explained.

## A. Decision-tree models

A binary decision tree is a rooted directed tree whose internal nodes perform threshold tests on single features. Each internal node v is labeled by a split feature $d ( v ) \in \mathcal { U }$ and a threshold $t h ( v ) \in \mathbb { R }$ . Given an input $\textbf { \em x } \in { \mathcal { X } }$ , the traversal follows the left child if $x _ { d ( v ) } ~ < ~ t h ( v )$ and the right child otherwise, until a leaf is reached. Let $\mathcal { L }$ denote the set of leaves of the tree trained on the training data.

Throughout the paper, we distinguish the tree structure $T ,$ consisting of the topology together with the split variables and thresholds, from the leaf outputs. The structure is kept fixed, while the leaf outputs will later be allowed to vary. Each leaf $\ell \in \mathcal L$ carries a scalar value $w _ { \ell } \in \mathbb { R }$ , such as a regression mean, a target-class probability, or a logit. We collect these values in the vector $w : = ( w _ { \ell } ) _ { \ell \in \mathcal { L } } \in \mathbb { R } ^ { | \mathcal { L } | }$ . In our case, as we are concerned with supervised learning, we will focus on target-class probability.

Let $\ell _ { T } ( { \pmb x } ) \in { \mathcal { L } }$ be the leaf reached by x, and let $R _ { \ell } : =$ $\{ { \pmb x } \in { \mathcal { X } } : \ell _ { T } ( { \pmb x } ) = \ell \}$ be the set of all inputs that would be routed to leaf ℓ. Then the prediction function admits the leaf-sum representation

$$
f _ { T , w } ( \pmb { x } ) = w _ { \ell _ { T } ( \pmb { x } ) } = \sum _ { \ell \in \mathcal { L } } w _ { \ell } \pmb { 1 } \{ \pmb { x } \in R _ { \ell } \} .\tag{1}
$$

i.e., the output is the value attached to the leaf selected by traversal [14].

## B. Feature attribution as a coalitional game

Local feature-attribution methods such as SHAP can be formulated as cooperative games over features [1], [2], [15]. Let $\mathbf { X } \ = \ ( X _ { 1 } , \ldots , X _ { n } ) \ \sim \ D$ be an X-valued background random vector. For each subset $S \subseteq u ,$ , we denote by ${ \bar { S } } : = { \mathcal { U } } \backslash S .$ . We use the interventional (marginal) value function

$$
v _ { w } ( S ) : = \mathbb { E } _ { \mathbf { X } _ { \bar { S } } } \left[ f _ { T , w } \mathopen { } \mathclose \bgroup \left( \pmb { x } _ { S } , \mathbf { X } _ { \bar { S } } \aftergroup \egroup \right) \right] ,\tag{2}
$$

where the hybrid vector $( \pmb { x } _ { S } , \mathbf { X } _ { \bar { S } } )$ fixes the coordinates in $S$ to their values in the explained instance and samples the remaining coordinates from the background distribution $D$ Equivalently, this semantics is often written as $v _ { w } ( S ) \ =$ $\mathbb { E } _ { \mathbf { X } _ { \bar { S } } } [ f _ { T , w } ( \mathbf { X } ) \mid \mathrm { d o } ( \mathbf { X } _ { S } = \pmb { x } _ { S } ) ]$ ], emphasizing that the features in $\tilde { S }$ are intervened on rather than conditioned on.

The baseline is ${ v _ { w } } ( \emptyset ) \ = \ f _ { T , w } ( { \pmb x } )$ . Other missing-feature semantics can also be used, including conditional expectations for dependent features [15] and path-dependent semantics for tree models [1], [16]. Our results apply to any semantics for which $v _ { w } ( S )$ can be written as an expectation of $f _ { T , w }$ under a distribution that does not depend on the leaf-output vector w.

## C. Shapley attributions

Given a set function $v : 2 ^ { \mathcal { U } } \to \mathbb { R }$ , the Shapley attribution of feature $i \in \mathcal { U }$ is:

$$
\phi _ { i } : = \sum _ { S \subseteq \mathcal { U } \backslash \{ i \} } \frac { | S | ! \left( n - | S | - 1 \right) ! } { n ! } \left( v ( S \cup \{ i \} ) - v ( S ) \right) .\tag{3}
$$

The Shapley value is uniquely characterized by axioms [14] such as efficiency, symmetry, linearity, and dummy. A standard convention defines an intercept $\phi _ { 0 } : = v ( \emptyset )$ so that the efficiency identity reads

$$
\begin{array} { r c l } { { } } & { { \displaystyle \phi _ { 0 } + \sum _ { i \in \mathcal { U } } \phi _ { i } \ = \ v ( \mathcal { U } ) \ = \ f _ { T , w } ( x ) , } } \\ { { } } & { { \displaystyle \mathrm { e q u i v a l e n t l y } \quad \ } } & { { \displaystyle \sum _ { i \in \mathcal { U } } \phi _ { i } \ = \ v ( \mathcal { U } ) - v ( \emptyset ) . } } \end{array}\tag{4}
$$

Remark 1: An alternative game-theoretic attribution is the raw Banzhaf value, which averages marginal contributions uniformly

$$
\beta _ { i } : = \frac { 1 } { 2 ^ { n - 1 } } \sum _ { S \subseteq \mathcal { U } \backslash \{ i \} } \big ( v ( S \cup \{ i \} ) - v ( S ) \big ) .
$$

Unlike the Shapley value, the raw Banzhaf value generally does not satisfy the efficiency identity $\textstyle \sum _ { i \in { \mathcal { U } } } \beta _ { i } = v ( { \mathcal { U } } ) - v ( \emptyset )$ Nevertheless, it has attractive computational properties for tree models and often produces feature rankings similar to Shapley attributions in practice [16].

## D. Leaf-linear representation of the game and attributions

Recall from Section II that a fixed tree structure T together with leaf values $\boldsymbol { w } ~ = ~ ( w _ { \ell } ) _ { \ell \in \mathcal { L } }$ defines the predictor $f _ { T , w } .$ Fixing T, an instance $\mathbf { \boldsymbol { x } } \in \mathcal { X }$ and a background distribution D and the missing-feature semantics, for each coalition $S \subseteq { \mathcal { U } } ,$ the induced game value $v _ { w } ( S )$ depends on w through the leaf outputs. This section shows that $v _ { w } ( S )$ is linear in w, and consequently that the attribution scores are linear forms in w. 1) Linearity of the coalition value $v _ { w } ( S )$ : For a coalition $S \subseteq { \mathcal { U } }$ , recall the associated game value

$$
v _ { w } ( S ) \ : = \ \mathbb { E } _ { \mathbf { X } _ { \bar { S } } } \left[ f _ { T , w } \big ( \mathbf { x } _ { S } , \mathbf { X } _ { \overline { { S } } } \big ) \right] .\tag{5}
$$

This is the expected prediction of the tree when the features in S are fixed at their values in the explained instance x, while the remaining features ${ \overline { { S } } } : = { \mathcal { U } } \backslash S$ are integrated out according to the background distribution. Hence, $v _ { w } ( S )$ quantifies the predictive contribution of knowing the features in S. Since the tree prediction is piecewise constant over leaves, $v _ { w } ( S )$ can be written in terms of the probabilities of reaching each leaf. For every leaf $\ell \in { \mathcal { L } }$ , define the reach probability

$$
\begin{array} { r l } & { q _ { \ell } ( S ) : = \mathbb { P } \Big ( \big ( \boldsymbol { x } _ { S } , \mathbf { X } _ { \overline { { S } } } \big ) \in R _ { \ell } \Big ) } \\ & { ~ = \mathbb { E } _ { \mathbf { X } _ { \bar { S } } } \Big [ \mathbf { 1 } \Big \{ \big ( \boldsymbol { x } _ { S } , \mathbf { X } _ { \overline { { S } } } \big ) \in R _ { \ell } \Big \} \Big ] . } \end{array}\tag{6}
$$

Let $q ( S ) : = ( q _ { \ell } ( S ) ) _ { \ell \in \mathcal L } \in [ 0 , 1 ] ^ { | \mathcal L | }$ . The quantity $q _ { \ell } ( S )$ is the probability that the hybrid point $( \pmb { x } _ { S } , \mathbf { X } _ { \overline { { S } } } )$ falls into leaf ℓ. It depends on the tree structure, the explained instance x, the coalition S, the background distribution and the missingfeature semantics, but crucially it does not depend on the leaf values w. Thus $q ( S )$ captures how the coalition S distributes probability mass over the leaves, whereas w determines the prediction associated with each leaf.

Proposition 1: For every coalition $S \subseteq { \mathcal { U } }$

$$
v _ { w } ( S ) = \sum _ { \ell \in \mathcal { L } } q _ { \ell } ( S ) w _ { \ell } = \langle q ( S ) , w \rangle .\tag{7}
$$

In particular, for each fixed S, the map $w \mapsto v _ { w } ( S )$ is linear. Proof 1: Insert the leaf decomposition of $f _ { T , w }$ into (5) and exchange the finite sum and expectation:

$$
\begin{array} { r } { v _ { w } ( S ) = \mathbb { E } _ { \mathbf { X } _ { \overline { { S } } } } \left[ \displaystyle \sum _ { \ell \in \mathcal { L } } w _ { \ell } \mathbf { 1 } \Big \{ \big ( \boldsymbol { x } _ { S } , \mathbf { X } _ { \overline { { S } } } \big ) \in R _ { \ell } \Big \} \right] } \\ { = \displaystyle \sum _ { \ell \in \mathcal { L } } w _ { \ell } \mathbb { E } _ { \mathbf { X } _ { \overline { { S } } } } \Big [ \mathbf { 1 } \Big \{ \big ( \boldsymbol { x } _ { S } , \mathbf { X } _ { \overline { { S } } } \big ) \in R _ { \ell } \Big \} \Big ] . } \end{array}
$$

The last expectation is exactly $q _ { \ell } ( S )$ from (6), which yields (7). ■

Note that Proposition 1 is a conditional statement: the reach probabilities $q _ { \ell } ( S )$ are treated as fixed quantities of a fixed tree. Equivalently, the uncertainty considered in this paper is propagated through the leaf outputs $w _ { \ell } .$ , not through the routing probabilities used to re-estimate $q _ { \ell } ( S )$ . This distinction is important for the uncertainty model introduced later.

2) Linearity of the attribution value $\phi ( w ) .$ : Proposition 1 implies that any linear attribution functional applied to the game $v _ { w }$ is itself linear in w. We state this explicitly for SHAP values.

Proposition 2: For each feature $i \in \mathcal { U } ,$ , the SHAP value is linear in the leaf-value vector w:

$$
\phi _ { i } ( w ) = \sum _ { \ell \in \mathcal { L } } a _ { i , \ell } w _ { \ell } = \langle a _ { i } , w \rangle ,\tag{8}
$$

where the coefficient vector $a _ { i } = ( a _ { i , \ell } ) _ { \ell \in \mathcal { L } } \in \mathbb { R } ^ { | \mathcal { L } | }$ is given by

$$
a _ { i , \ell } = \sum _ { S \subseteq \mathcal { U } \setminus \{ i \} } \frac { | S | ! ( n - | S | - 1 ) ! } { n ! } \Big ( q _ { \ell } ( S \cup \{ i \} ) - q _ { \ell } ( S ) \Big ) .\tag{9}
$$

In particular, $a _ { i }$ depends only on the tree structure, the explained instance, the background distribution and the missingfeature semantics, and not on the leaf values w. Throughout the paper, we use SHAP value to refer to the Shapley value of the feature-attribution game induced by the value function $v _ { w }$ in (2).

Proof 2: By Proposition 1,

$$
v _ { w } ( S \cup \{ i \} ) - v _ { w } ( S ) = \sum _ { \ell \in \mathcal { L } } \left( q _ { \ell } ( S \cup \{ i \} ) - q _ { \ell } ( S ) \right) w _ { \ell } .
$$

Substituting this expression into the Shapley formula (3) and exchanging the finite sums yields $\phi _ { i } ( w ) =$

$$
\sum _ { \ell \in \mathcal { L } } \left[ \sum _ { S \subseteq \mathcal { U } \setminus \{ i \} } \frac { | S | ! ( n - | S | - 1 ) ! } { n ! } \big ( q _ { \ell } ( S \cup \{ i \} ) - q _ { \ell } ( S ) \big ) \right] w _ { \ell } ,
$$

which is exactly (8)–(9).

Remark 2: Banzhaf value is also a linear combination of the marginal differences $v _ { w } ( S \cup \{ i \} ) - v _ { w } ( S )$ , but with uniform weights. Thus, all our subsequent results apply to Banzhaf indices by replacing the Shapley coefficient $a _ { i , \ell }$ with the Banzhaf coefficient

$$
b _ { i , \ell } = \frac { 1 } { 2 ^ { n - 1 } } \sum _ { S \subseteq \mathcal { U } \setminus \{ i \} } \Big ( q _ { \ell } ( S \cup \{ i \} ) - q _ { \ell } ( S ) \Big ) .\tag{10}
$$

For this reason, we will focus on SHAP from now on.

The reach probabilities $q _ { \ell } ( S )$ are fixed once the tree structure, explained instance, background distribution, and missingfeature semantics are fixed. Therefore, once $( a _ { i } ) _ { i \in \mathcal { U } }$ are computed, SHAP values can only vary according to parameters w, reducing the analysis to linear forms. This leaf-linear representation is the key ingredient behind the interval-valued robustness analysis developed in the next section.

## III. INTERVAL-VALUED FEATURE ATTRIBUTION UNDER LEAF UNCERTAINTY

We now study how to introduce uncertainty in the leaf parameters $w _ { \ell } ,$ the target-class probabilities. Rather than considering arbitrary variations, we will connect those to virtual samples: this has the benefit that it is easy for the user to specify a number of additional virtual samples of unknown class and features, and see how SHAP values ϕ<sub>i</sub> vary from it.

## A. From virtual samples to leaf uncertainty

Since the tree structure is fixed, uncertainty is introduced only through the leaf-output vector $\begin{array} { r c l } { w } & { = } & { ( w _ { \ell } ) _ { \ell \in { \mathcal L } } } \end{array}$ . This can model retraining variability or moderate distribution shift affecting leaf estimates. For a target-class probability, let $N _ { \ell }$ be the cover of leaf $\ell ,$ and let $n _ { \ell }$ be the number of samples of the target class in that leaf. The nominal leaf output is therefore $\begin{array} { r } { w _ { \ell } = \frac { n _ { \ell } } { N _ { \ell } } } \end{array}$ . Depending on the value of $N _ { \ell } ,$ , a small variation of $n _ { \ell }$ can have a deep impact on the value of $w _ { \ell } .$

For this reason, it may be useful to check how sensitive our explanations are (taking the form of SHAP values) if we add a small number of instances to the leaf. A simple idea to do that is to add to leaf ℓ a number $m _ { \ell }$ of unclassified instances. This gives rise to the following possible interval of values w<sub>ℓ</sub>:

$$
\left[ { \underline { { w } } } _ { \ell } , { \overline { { w } } } _ { \ell } \right] = \left[ { \frac { n _ { \ell } } { N _ { \ell } + m _ { \ell } } } , { \frac { n _ { \ell } + m _ { \ell } } { N _ { \ell } + m _ { \ell } } } \right] ,
$$

where the lower bound is obtained by considering that all $m _ { \ell }$ instances do not belong to the target class, and the upper by considering that they all belong to the target class. These bounds can either be interpreted in a frequentist way, in which case they are derived from imprecise observations [17], or in a Bayesian way, in which case they would correspond to the Imprecise Dirichlet Model (IDM) [18].

Note that it is difficult for a user to specify a meaningful number $m _ { \ell }$ of unknown virtual instances per leaf. This is why we will consider that the user specifies a unique number s of virtual instances to be distributed among the tree leaves. Given this, an allocation of virtual instances will be a vector

$$
m = ( m _ { \ell } ) _ { \ell \in \mathcal { L } } \in \mathcal { M } ( s ) : = \left\{ m \in \mathbb { N } ^ { | \mathcal { L } | } : \sum _ { \ell \in \mathcal { L } } m _ { \ell } = s \right\} .
$$

Note that the uncertainty level s has a more direct statistical interpretation than a uniform per-leaf allocation used in [19]: it measures how many imprecise observations the explanation is required to tolerate, without assuming where these observations would fall in the tree. As we will see next, this makes the optimisation problems associated with the sensitivity analysis more complex, but we think it is much more understandable from the user’s perspective, which is essential in explainability.

Example 1: We use a toy depth-2 tree with three split features and four leaves as a running example. Consider the allocation $m = ( 1 , 2 , 1 , 2 )$ , corresponding to 6 unannotated instances. The resulting IDM intervals are reported in Table I.

TABLE I: IDM leaf-output intervals.
<table><tr><td>leaf</td><td>nl</td><td> $N _ { \ell }$ </td><td> $w _ { \ell }$   $m _ { \ell }$ </td><td>[we, we]</td></tr><tr><td> $\ell _ { 1 }$ </td><td>20</td><td>60 1/3</td><td>1</td><td>20 21 1</td></tr><tr><td> $\ell _ { 2 }$ </td><td>20 50</td><td>2/5</td><td>2</td><td>50 62-511</td></tr><tr><td> $\ell _ { 3 }$ </td><td>10 30</td><td>1/3</td><td>1</td><td>78</td></tr><tr><td> $\ell _ { 4 }$ </td><td>30 50</td><td>3/5</td><td>2</td><td>3 325 52，</td></tr></table>

## B. Interval-valued attributions

In this section, we study how we can integrate the virtual samples in order to derive interval-valued attributions of each feature. We will consider two settings: one where we have no information about the distribution of virtual samples (unannotated samples) into the leaf, which we will call unknown allocation, and one where we have an idea about how the virtual sample population is distributed within the leaves, which we will call stochastic allocation.

Before detailing each approach, recall that by Section II-D, for each feature $i \in \mathcal { U }$ there exists a coefficient vector $a _ { i } \in$ $\mathbb { R } ^ { | \mathcal { L } | }$ such that

$$
\phi _ { i } ( w ) = \langle a _ { i } , w \rangle \qquad { \mathrm { f o r ~ a l l ~ } } w \in \mathbb { R } ^ { | \mathcal { L } | } ,
$$

meaning that $\phi _ { i }$ is a linear function of w.

1) unknown allocation (pessimistic): We will consider the problem of deriving upper and lower bounds of $\phi _ { i }$ given that any allocation can happen. Let

$$
\operatorname { r a n } ( w ) = \bigtimes _ { \ell \in \mathcal { L } } [ \underline { { w } } _ { \ell } , \overline { { w } } _ { \ell } ] .
$$

Formally, we will aim at solving the following problems:

$$
\begin{array} { r } { \underline { { \phi } } _ { i } ^ { U } = \displaystyle \operatorname* { i n f } _ { m \in \mathcal { M } ( s ) } \operatorname* { i n f } _ { w \in \mathrm { r a n } ( w ) } \langle a _ { i } , w \rangle , } \\ { \overline { { \phi } } _ { i } ^ { U } = \displaystyle \operatorname* { s u p } _ { m \in \mathcal { M } ( s ) } \operatorname* { s u p } _ { w \in \mathrm { r a n } ( w ) } \langle a _ { i } , w \rangle . } \end{array}\tag{11}
$$

(12)

We refer to this as the pessimistic principle: since the allocation is unknown, the interval is enlarged by considering the most adverse allocation for each endpoint. We will focus on solving the lower bound, and will show that the derived solution is analogous to the upper bound. Let us first focus on the inner optimization problem, which is straightforward to solve given its linearity

$$
\underline { { \phi } } _ { i } ^ { i n n } = \operatorname* { i n f } _ { w \in \mathrm { r a n } ( w ) } \langle a _ { i } , w \rangle = \sum _ { \ell \in \mathcal { L } } \left\{ \begin{array} { l l } { a _ { i , \ell } \underline { { w } } _ { \ell } , } & { a _ { i , \ell } \ge 0 , } \\ { a _ { i , \ell } \overline { { w } } _ { \ell } , } & { a _ { i , \ell } < 0 , } \end{array} \right.\tag{13}
$$

So, the problem becomes

$$
\underline { { \phi } } _ { i } ^ { U } = \operatorname* { i n f } _ { m \in \mathcal { M } ( s ) } \underline { { \phi } } _ { i } ^ { i n n }
$$

and we now have to find the allocation m for which the minimum will be reached. In order to do that, we will first show that Equation (13) can be reformulated more conveniently.

Proposition 3: Equation (13) can be reformulated as

$$
\underline { { \phi } } _ { i } ^ { i n n } = \phi _ { i } ( w ) - C _ { \alpha _ { i } ^ { - } } ^ { \mathrm { w c } } ( m )
$$

where

$$
\begin{array} { r l } & { C _ { \alpha _ { i } ^ { - } } ^ { \mathrm { w c } } : = \displaystyle \sum _ { \ell \in \mathcal { L } } \alpha _ { i , \ell } ^ { - } \frac { m _ { \ell } } { N _ { \ell } + m _ { \ell } } , } \\ & { \alpha _ { i , \ell } ^ { - } : = [ a _ { i , \ell } ] _ { + } w _ { \ell } + [ - a _ { i , \ell } ] _ { + } ( 1 - w _ { \ell } ) , } \end{array}\tag{14}
$$

(15)

and $[ u ] _ { + } : = \operatorname* { m a x } \{ u , 0 \}$ , is a correction term depending on m. Proof 3: First note that we can rewrite

$$
{ \underline { { w } } } _ { \ell } = w _ { \ell } - w _ { \ell } { \frac { m _ { \ell } } { N _ { \ell } + m _ { \ell } } } .
$$

Indeed, we have $\begin{array} { r } { w _ { \ell } - w _ { \ell } \frac { m _ { \ell } } { N _ { \ell } + m _ { \ell } } } \end{array}$

$$
\begin{array} { l } { = \displaystyle \frac { n _ { \ell } } { N _ { \ell } } - \frac { n _ { \ell } } { N _ { \ell } } \frac { m _ { \ell } } { N _ { \ell } + m _ { \ell } } = \frac { n _ { \ell } ( N _ { \ell } + m _ { \ell } ) - n _ { \ell } m _ { \ell } } { N _ { \ell } ( N _ { \ell } + m _ { \ell } ) } } \\ { = \frac { N _ { \ell } n _ { \ell } } { N _ { \ell } ( N _ { \ell } + m _ { \ell } ) } = \frac { n _ { \ell } } { N _ { \ell } + m _ { \ell } } = \underline { { w } } _ { \ell } . } \end{array}
$$

Likewise, we can rewrite

$$
\overline { { w } } _ { \ell } = w _ { \ell } + ( 1 - w _ { \ell } ) \frac { m _ { \ell } } { N _ { \ell } + m _ { \ell } } .
$$

Plugging those rewrittings in $\operatorname { E q . } \ ( 1 3 )$ , we get

$$
{ \underline { { \phi } } } _ { i } ^ { i n n } = \sum _ { \ell \in { \mathcal { L } } } w _ { \ell } a _ { i , \ell } + \left\{ { \begin{array} { l l } { - a _ { i , \ell } w _ { \ell } { \frac { m _ { \ell } } { N _ { \ell } + m _ { \ell } } } , } & { a _ { i , \ell } \geq 0 , } \\ { a _ { i , \ell } ( 1 - w _ { \ell } ) { \frac { m _ { \ell } } { N _ { \ell } + m _ { \ell } } } , } & { a _ { i , \ell } < 0 . } \end{array} } \right.\tag{16}
$$

$$
= \phi _ { i } ( w ) - \sum _ { \ell \in \mathcal { L } } \alpha _ { i , \ell } ^ { - } \frac { m _ { \ell } } { N _ { \ell } + m _ { \ell } }\tag{17}
$$

$$
= \phi _ { i } ( w ) - C _ { \alpha _ { i } ^ { - } } ^ { \mathrm { w c } } ( m ) .\tag{18}
$$

So, now, all we have to do is to solve

$$
\begin{array} { l } { { \phi _ { i } ^ { U } = \displaystyle \operatorname* { i n f } _ { m \in \mathcal { M } ( s ) } \phi _ { i } ( w ) - C _ { \alpha _ { i } ^ { - } } ^ { \mathrm { w c } } ( m ) = \phi _ { i } ( w ) - \operatorname* { s u p } _ { m \in \mathcal { M } ( s ) } C _ { \alpha _ { i } ^ { - } } ^ { \mathrm { w c } } ( m ) } } \\ { { \quad \quad = \phi _ { i } ( w ) - C _ { \alpha _ { i } ^ { - } } ^ { \mathrm { w c } } ( s ) , } } \end{array}
$$

which means we need to find the allocation maximizing $C _ { \alpha _ { i } ^ { - } } ^ { \mathrm { w c } } ( m )$ . We first note that

$$
C _ { \alpha _ { i } ^ { - } } ^ { \mathrm { w c } } ( m ) = \sum _ { \ell \in \mathcal { L } } \alpha _ { i , \ell } ^ { - } \frac { m _ { \ell } } { N _ { \ell } + m _ { \ell } } = \sum _ { \ell \in \mathcal { L } } \alpha _ { i , \ell } ^ { - } g _ { l } ( m _ { \ell } )
$$

is a separable function. Moreover, it is easy to check that each function g<sub>l</sub> is concave so that $C _ { \alpha _ { i } ^ { - } } ^ { \mathrm { w c } } ( m )$ is also concave. Finding the optimal m is thus a resource allocation problem which can be solved using a greedy approach, the steps of which can be summarized in the following algorithm:

Algorithm 1 Optimal allocation of virtual samples   
1: Start with $m _ { l } = 0$ for all l   
2: Compute for each l the marginal gain:   
$\begin{array} { r } { \Delta _ { \ell } ( m _ { \ell } + 1 ) : = \alpha _ { i , \ell } ^ { - } \left( \frac { m _ { \ell } + 1 } { N _ { \ell } + m _ { \ell } + 1 } - \frac { m _ { \ell } } { N _ { \ell } + m _ { \ell } } \right) } \\ { = \alpha _ { i , \ell } ^ { - } \frac { N _ { \ell } } { ( N _ { \ell } + m _ { \ell } + 1 ) ( N _ { \ell } + m _ { \ell } ) } ; \quad { \quad } } \end{array}$ (19)   
3: Assign a new sample to the leaf with the largest gain;   
4: Repeat this sequence s times.

Algorithm 1 runs in $O ( s | \mathcal { L } | )$ time with a direct implementation. The optimization problem for the upper bound can be treated likewise, since if we define

$$
\alpha _ { i , \ell } ^ { + } : = [ a _ { i , \ell } ] _ { + } \left( 1 - w _ { \ell } \right) + [ - a _ { i , \ell } ] _ { + } w _ { \ell } ,\tag{20}
$$

Then the inner supremum can be rewritten as

$$
\overline { { { \phi } } } _ { i } ^ { i n n } = \phi _ { i } ( w ) + \sum _ { \ell \in \mathcal { L } } \alpha _ { i , \ell } ^ { + } \frac { m _ { \ell } } { N _ { \ell } + m _ { \ell } } = \phi _ { i } ( w ) + C _ { \alpha ^ { + } } ^ { \mathrm { w c } } ( m )
$$

which is of a form similar to Proposition 3, except that we have $\overline { { { \phi } } } _ { i } ^ { i n n } = \phi _ { i } ( w ) + C _ { \alpha _ { \therefore } ^ { + } } ^ { \mathrm { w c } } ( m )$ , with the need to find the allocation maximising the second term, meaning we can apply Algorithm 1 to solve it.

In practice, the greedy allocation used to compute the lower correction and the one used to compute the upper correction need not coincide; moreover, both allocations may depend on the feature i.

2) Stochastic allocation (averaging): An alternative is to model the allocation probabilistically, in which case, assuming that P(m) is the probability of a given allocation, we have

$$
\underline { { \phi } } _ { i } ^ { S } = \sum _ { m \in \mathcal { M } ( s ) } \mathbb { P } ( m ) \operatorname* { i n f } _ { w \in \mathrm { r a n } ( w ) } \langle a _ { i } , w \rangle : = \mathbb { E } _ { M } [ \underline { { \phi } } _ { i } ^ { i n n } ( M ) ] ,\tag{21}
$$

$$
\overline { { \phi } } _ { i } ^ { S } = \sum _ { m \in \mathcal { M } ( s ) } \mathbb { P } ( m ) \operatorname* { s u p } _ { w \in \operatorname { r a n } ( w ) } \langle a _ { i } , w \rangle : = \mathbb { E } _ { M } [ \overline { { \phi } } _ { i } ^ { i n n } ( M ) ] ,\tag{22}
$$

We refer to this as the averaging principle: instead of optimizing over all feasible allocations, the attribution endpoints are averaged with respect to the allocation distribution $\mathbb { P } ( m )$ We will consider the case where $\begin{array} { r l } { M } & { { } = } \end{array}$ $( M _ { \ell } ) _ { \ell \in \mathcal { L } } \sim$ Multinomial(s, π), is a Multinomial and where $\dot { \pi } = \bar { ( \pi _ { \ell } ) } \ell \in \mathcal { L } \in \Delta ^ { | \mathcal { L } | - 1 }$ is the leaf-placement distribution. Then each marginal satisfies $M _ { \ell } \sim \mathrm { B i n o m i a l } ( s , \pi _ { \ell } )$

Using our previous decompositions of $\underline { { \phi } } _ { i n n }$ and $\overline { { \phi } } _ { i n n }$ we get

$$
\begin{array} { r l r } & { } & { \mathbb { E } _ { M } [ \underline { { \phi } } _ { i } ^ { i n n } ( M ) ] = \phi _ { i } ( w ) - C _ { \alpha _ { i } ^ { - } } ^ { \mathrm { e x p } } ( s ) , } \\ & { } & { \mathbb { E } _ { M } [ \overline { { \phi } } _ { i } ^ { i n n } ( M ) ] = \phi _ { i } ( w ) + C _ { \alpha _ { i } ^ { + } } ^ { \mathrm { e x p } } ( s ) . } \end{array}\tag{23}
$$

where the expected correction for any nonnegative coefficient vector can be rewritten for sign $\in \{ - , + \}$ is

$$
C _ { \alpha _ { i } ^ { \mathrm { s i g n } } } ^ { \mathrm { e x p } } ( s ) = \mathbb { E } _ { M } [ C _ { \alpha _ { i } ^ { \mathrm { s i g n } } } ^ { \mathrm { e x p } } ( M ) ] = \sum _ { \ell \in \mathcal { L } } \Bigl ( \mathbb { E } _ { M } \left[ \frac { M _ { \ell } } { N _ { \ell } + M _ { \ell } } \right] \alpha _ { i , \ell } ^ { \mathrm { s i g n } } \Bigr ) ,
$$

thanks to expectation linearity, and where

$$
\mathbb { E } _ { M } \bigg [ \frac { M \ell } { N _ { \ell } + M _ { \ell } } \bigg ] = \sum _ { r = 0 } ^ { s } \frac { r } { N _ { \ell } + r } \binom { s } { r } \pi _ { \ell } ^ { r } ( 1 - \pi _ { \ell } ) ^ { s - r } .\tag{24}
$$

due to our multinomial assumption.

These expected endpoints $\underline { { \phi } } _ { i } ^ { S } , \overline { { \phi } } _ { i } ^ { S }$ characterize average attribution uncertainty under the placement law π. They should be distinguished from the worst-case endpoints $\underline { { \phi } } _ { i } ^ { U } , \overline { { \phi } } _ { i } ^ { U }$ : the latter optimize over all allocations, whereas the former average over allocations sampled from a specified distribution. Since the correction terms are nonnegative, the expected interval is no wider than the corresponding worst-case envelope. In practice, one may choose π to be, for example, uniform over leaves, $\pi _ { \ell } = 1 / | \mathcal { L } |$ , or proportional to empirical leaf mass, $\pi _ { \ell } = N _ { \ell } / \sum _ { j } N _ { j }$ , depending on the intended interpretation.

Example 2: We continue Example 1. Consider the lower correction for a fixed feature $i ,$ and suppose that the coefficients in the correction term (14) are $\alpha \quad =$ $( 0 . 6 0 , 0 . 4 5 , 0 . 3 0 , 0 . 5 0 )$ . In Table II, under unknown allocation, the adversary chooses the allocation maximizing $C _ { \alpha _ { i } ^ { - } } ^ { \mathrm { w c } } ( m )$ Starting from $\textit { m } = \textit { \textbf { ( 0 , 0 , 0 , 0 ) } }$ , the greedy rule assigns each new virtual sample to the leaf with the largest current marginal gain $\Delta _ { \ell }$ . Under stochastic allocation, we instead assume M ∼ Multinomial $( 6 , \pi )$ , with empirical leaf masses $\pi \ = \ \left( { \frac { 6 0 } { 1 9 0 } } , { \frac { 5 0 } { 1 9 0 } } , { \frac { 3 0 } { 1 9 0 } } , { \frac { 5 0 } { 1 9 0 } } \right)$ . As expected, $C _ { \alpha _ { i } ^ { - } } ^ { \mathrm { w c } } ( 6 )$ is greater than $C _ { \alpha _ { i } ^ { - } } ^ { \mathrm { e x p } } ( 6 )$

TABLE II: Worst-case and stochastic allocation computations.
<table><tr><td rowspan=1 colspan=1>step</td><td rowspan=1 colspan=1>leaf</td><td rowspan=1 colspan=1> $\Delta _ { \ell }$ </td><td rowspan=1 colspan=1>m</td></tr><tr><td rowspan=2 colspan=1>12</td><td rowspan=1 colspan=1> $\overline { { \ell _ { 1 } } }$ </td><td rowspan=1 colspan=1>0.00984</td><td rowspan=1 colspan=1> $\overline { { ( 1 , 0 , 0 , 0 ) } }$ </td></tr><tr><td rowspan=1 colspan=1> $\ell _ { 4 }$ </td><td rowspan=1 colspan=1>0.00980</td><td rowspan=1 colspan=1>(1, 0, 0, 1)</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1> $\ell _ { 3 }$ </td><td rowspan=1 colspan=1>0.00968</td><td rowspan=1 colspan=1>(1, 0, 1, 1)</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1> $\ell _ { 1 }$ </td><td rowspan=1 colspan=1>0.00952</td><td rowspan=1 colspan=1>(2, 0, 1, 1)</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1> $\ell _ { 4 }$ </td><td rowspan=1 colspan=1>0.00943</td><td rowspan=1 colspan=1>(2, 0,1, 2)</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1> $\ell _ { 1 }$ </td><td rowspan=1 colspan=1>0.00922</td><td rowspan=1 colspan=1>(3, 0,1, 2)</td></tr><tr><td rowspan=1 colspan=4> $\overline { { C _ { \alpha _ { i } ^ { - } } ^ { \mathrm { w c } } } }$ (6) ≈ 0.0575</td></tr></table>

<table><tr><td>leaf</td><td> $\pi _ { \ell }$ </td><td></td><td> $\overline { { \mathbb { E } \underbrace { \left[ \frac { M _ { \ell } } { N _ { \ell } + M _ { \ell } } \right] } } }$ </td></tr><tr><td> $\ell _ { 1 }$ </td><td>60/190</td><td colspan="2">0.0303</td></tr><tr><td> $\ell _ { 2 }$ </td><td>50/190</td><td colspan="2">0.0302</td></tr><tr><td> $\ell _ { 3 }$ </td><td>30/190</td><td colspan="2">0.0298</td></tr><tr><td> $\ell _ { 4 }$ </td><td colspan="2">50/190</td><td>0.0302</td></tr><tr><td colspan="4">Cexp(6) ≈ 0.0558 αi</td></tr></table>

## C. Extensions to tree-ensemble

We now discuss how the leaf-uncertainty framework extends to tree ensembles. Let $\mathcal { T } = \{ T _ { 1 } , . . . T _ { T } \}$ be the set of T trees. Let $\tau \tau = ( \ell ^ { 1 } , \dots , \ell ^ { T } )$ be the set of leaves (one per tree) that cover x. The ensemble prediction is

$$
F _ { \mathcal { T } , w } ( x ) = \sum _ { t = 1 } ^ { T } \eta _ { t } f _ { t , w ^ { ( t ) } } ( { \pmb x } )\tag{25}
$$

where $f _ { t , w ^ { ( t ) } } ( x )$ is the prediction of the t-th tree, $w ^ { t }$ is its leafoutput vector, $\eta _ { t }$ is its weight. For a random forest, typically $\eta _ { t } = 1 / T ;$ ; for gradient-boosted trees such as XGBoost, the weights include the learning-rate scaling.

By linearity of attributions in the model output, the singletree leaf-linear representation extends directly to ensembles:

$$
\begin{array} { l } { \displaystyle \phi _ { i } \big ( F _ { \mathcal { T } , w } \big ) = \sum _ { t = 1 } ^ { T } \eta _ { t } \phi _ { i } \big ( f _ { t , w ^ { ( t ) } } \big ) = \sum _ { t = 1 } ^ { T } \eta _ { t } \langle a _ { i } ^ { ( t ) } , w ^ { ( t ) } \rangle } \\ { \displaystyle = \langle a _ { i } ^ { \mathrm { e n s } } , w ^ { \mathrm { e n s } } \rangle , } \end{array}\tag{26}
$$

where $w ^ { \mathrm { e n s } }$ is the concatenation of all leaf-output vectors and $a _ { i } ^ { \mathrm { { e n s } } }$ is the corresponding concatenated coefficient vector. Thus, under fixed tree structures, ensemble attributions remain linear in the collection of leaf outputs.

Let $\mathcal { L } _ { \mathcal { T } } = \mathcal { L } _ { 1 } \times \cdot \cdot \cdot \times \mathcal { L } _ { \mathcal { T } }$ . The ensemble $\tau$ can be interpreted as a (possibly very) large tree, whose leaves are non-empty intersections:

$$
\ell _ { T } = \cap _ { t = 1 } ^ { T } \ell _ { t } \neq \varnothing , \forall \tau _ { T } \in L _ { T } .\tag{27}
$$

For each leaf $\ell _ { T } .$ , the attribution is defined using (26). Therefore, one can, in principle, convert the ensemble into a tree with the leaf output (25) and compute interval-valued attribution following either the pessimistic or averaging principle. However, constructing the tree can be costly.

We therefore propose to proceed with an approximate approach. When proceeding with the pessimistic approach, we relax the requirement that the leaf $\ell \tau \ ( 2 7 )$ should be non-empty and define the attribution using (26) even if the leaf is empty. The greedy algorithm proposed in Algorithm 1 can be easily extended to find an allocation on the (possibly empty) leaves of the ensemble that is optimal under the pessimistic principle. Technically, we start from $m _ { \ell } ^ { t } ~ = ~ 0$ for any ℓ and $t ,$ and repeatedly assign the next example to elements of a set of leaves $\tau _ { T }$ with the highest marginal gains. At each iteration, this set of leaves can be found efficiently. Thanks to the linearity of the attribution (26), we can look at each tree separately and select the leaf with the highest marginal gain on that tree. Since the marginal gains achieved during the allocation procedure can come from invalid, i.e., empty, leaves of the ensemble, the interval-valued attributions given by this approximate approach are an outer approximation of the interval-valued attributions produced by converting the ensemble into a tree and then proceeding with the pessimistic principle.

One can, of course, think of an approximate approach to the averaging principle by averaging the ensemble correction over all the (possibly empty) leaves $\ell \tau$ of the ensemble. This will be equivalent to averaging the corrections of the leaves of the trees. However, the interval-valued attributions are not necessarily either the attributions provided by the exact approach or their outer approximation.

## D. Discussion

In many applications, feature attributions are mainly used through rankings rather than through their exact numerical values. It is therefore natural to ask whether such rankings remain stable under leaf uncertainty. A direct comparison of individual attribution intervals can be misleading, since the lower and upper endpoints of different features may be attained under different leaf-output vectors. For ranking stability, it is more appropriate to compare two features under the same leaf vector w. For two features i and $j ,$ define the pairwise attribution margin $\gamma _ { i j } ( w ) : = \phi _ { i } ( w ) - \phi _ { j } ( w ) = \langle a _ { i } - a _ { j } , w \rangle$ Since $\gamma _ { i j }$ is again linear in the leaf-output vector, the same interval-valued analysis applies by replacing the attribution coefficient $a _ { i }$ with the margin coefficient $a _ { i } - a _ { j }$

Given a nominal ranking $i _ { ( 1 ) } , \ldots , i _ { ( n ) }$ , one can test the stability of the whole ranking by checking the adjacent margins $\gamma _ { i _ { ( k ) } i _ { ( k + 1 ) } } ( w ) , k = 1 , \dots , n - 1$ . If all adjacent lower margins are positive, then the complete ranking is robust at uncertainty level s. If some adjacent comparisons are ambiguous, the method may still certify a coarser tiered ranking. For example, even if features 1 and 2 cannot be robustly ordered, both may remain certified above feature 3, yielding the stable statement $\{ 1 , 2 \} \ \succ _ { s } \ \{ 3 \}$ . This tiered interpretation is often sufficient when the goal is to identify a robust set of top features rather than a fully ordered ranking.

This ranking viewpoint also provides a practical interpretation of s. In the original IDM, s represents prior strength, and Walley [18] recommends small values, typically $s = 1$ or s = 2. In our setting, where s represents a virtual-sample uncertainty budget, ranking stability provides an applicationoriented selection criterion: as s increases, pairwise ranking certificates become harder to maintain, and the robust ordering between two features, or between two groups of features, may eventually fail. For example, at the dataset level, one can fix the nominal top-k set of each instance and track the fraction of instances for which this set remains robustly topranked as s increases from zero. The largest s for which this fraction remains above a prescribed threshold then quantifies how much leaf uncertainty the top-k explanations can tolerate while maintaining the desired level of stability.

Since ranking certification is not the main focus of this work, we leave a detailed study of margin-based ranking stability for single trees and ensembles as future work. In the experiments below, whenever interval comparisons are needed, we use the simpler sufficient criterion ${ \underline { { \phi } } } _ { i } > { \overline { { \phi } } } _ { j }$ to certify that feature i is ranked above feature j.

## IV. EXPERIMENTS

We evaluate the proposed interval-valued attribution framework from the viewpoint of a practitioner who uses featureimportance explanations to interpret tree-based predictions. The objective is not only to show how to interpret and robustify the framework, but also to provide empirical evidence for potential applications. The source code is available in the repository GitHub. We organize the experiments around the following questions:

1) For a fixed instance, which nominally important features remain robust under a given uncertainty level, and which ones become fragile?

2) Does the proposed interval-valued attribution agree with empirical attribution variability under bootstrap model retraining?

3) How does interval-valued attribution vary between confident and ambiguous decision regions?

4) Can the pessimistic principle reduce the apparent importance of uninformative features?

Since our robustness notions are defined for a single explained instance, the primary analysis is local. We first examine individual predictions to show how interval-valued attributions distinguish robust from fragile feature-importance claims. We then aggregate the same instance-level quantities over the test set. These summaries provide a dataset-level view of attribution robustness while preserving the local nature of the proposed interval-valued SHAP.

## A. Experimental protocol

We first use decision trees for Question 1, where the proposed uncertainty model is clearly interpretable. We then consider random forests for the remaining questions, in order to evaluate how the robustness analysis behaves in ensemble settings. We use four public tabular binary-classification datasets: Diabetes (Diab.), Breast Cancer (BreC.), Ionoshpere (Ions.), Mortality (Mort.). The first three datasets are available from OpenML. The Mortality dataset is the case study from [3]. The summary of the datasets is shown in Table III.

TABLE III: Summary of datasets
<table><tr><td></td><td>Diab.</td><td>BreC.</td><td>Ions.</td><td>Mort.</td></tr><tr><td>Features</td><td>8</td><td>30</td><td>34</td><td>79</td></tr><tr><td>Instances</td><td>768</td><td>569</td><td>351</td><td>14264</td></tr></table>

For each dataset, we use a 5-fold cross-validation protocol. Model hyperparameters are selected on the training set using 3-fold nested cross-validation, with balanced accuracy as the selection criterion. For both decision trees and random forests, we tune the minimum number of samples per leaf over {5, 10, 20}. Random forests use 100 trees. We then compute instance-level SHAP intervals on the test set and report averages over the five folds. Experiments were run on Ubuntu 24.04 using a machine with 80 CPU cores (2 × Intel(R) Xeon(R) Gold 6138 CPU @ 2.00GHz).

TABLE IV: Representative instances from the Diabetes dataset, their nominal SHAP values, and magnitude-based intervals. (+) and (−) indicate support for and opposition to the diabetic prediction, respectively  
(a) Representative instances.
<table><tr><td>sample | preg</td><td></td><td>plas</td><td>pres</td><td>skin</td><td>insu</td><td>BMI</td><td>pedi</td><td>age</td></tr><tr><td>1</td><td>2</td><td>108</td><td>62</td><td>32</td><td>56</td><td>25.2</td><td>0.128</td><td>21</td></tr><tr><td>2</td><td>5</td><td>166</td><td>72</td><td>19</td><td>175</td><td>25.8</td><td>0.587</td><td>51</td></tr></table>

(b) Nominal SHAP values and magnitude-based intervals.

<table><tr><td rowspan=1 colspan=1>sample</td><td rowspan=1 colspan=1>feature</td><td rowspan=1 colspan=3>nominal  pessimistic    averaging</td></tr><tr><td rowspan=5 colspan=1>1</td><td rowspan=1 colspan=1>plas</td><td rowspan=1 colspan=3>(-) 0.137 [0.133,0.141] [0.135, 0.138]</td></tr><tr><td rowspan=4 colspan=1>BMIagepedipregpres</td><td rowspan=1 colspan=3>(-) 0.130 [0.122,0.135] [0.128,0.131](-) 0.059 [0.053, 0.073] [0.057, 0.060]</td></tr><tr><td rowspan=1 colspan=3>(-) 0.021 [0.019, 0.022] [0.021, 0.021]</td></tr><tr><td rowspan=1 colspan=3>(-) 0.004 [0.003, 0.005] [0.004, 0.004]</td></tr><tr><td rowspan=1 colspan=1>(+) 0.003</td><td rowspan=1 colspan=2>[0.002, 0.004] [0.003, 0.003]</td></tr><tr><td rowspan=6 colspan=1>2</td><td rowspan=1 colspan=1>plas</td><td rowspan=1 colspan=1>(+) 0.357</td><td rowspan=1 colspan=1>[0.334, 0.365]</td><td rowspan=1 colspan=1>[0.353, 0.359]</td></tr><tr><td rowspan=1 colspan=1>pedi</td><td rowspan=1 colspan=1>(+) 0.108</td><td rowspan=1 colspan=1>[0.099,0.114]</td><td rowspan=1 colspan=1>[0.107, 0.109]</td></tr><tr><td rowspan=4 colspan=1>BMIagepregpres</td><td rowspan=1 colspan=2>(-) 0.093 [0.084, 0.095]</td><td rowspan=1 colspan=1>[0.092, 0.094]</td></tr><tr><td rowspan=1 colspan=3>(+) 0.029 [0.027, 0.034] [0.028, 0.030]</td></tr><tr><td rowspan=1 colspan=2>(-) 0.003 [0.002, 0.003]</td><td rowspan=1 colspan=1>[0.003, 0.003]</td></tr><tr><td rowspan=1 colspan=2>(+) 0.001 [0.000, 0.001]</td><td rowspan=1 colspan=1>[0.001, 0.001]</td></tr></table>

## B. Case study

First, we provide a local understanding of our proposed robustness framework to answer Questions 1. We present a case study on selected instances to see whether SHAP values and their induced rankings remain stable. Specifically, from a user perspective, the key takeaways are: (i) dominance, where a feature remains clearly more influential than another because their intervals are separated; (ii) incomparability, where the relative order of two features is ambiguous because their intervals overlap; and (iii) robust top-k, where a group of features dominates all features outside the group even if the ordering inside the group is ambiguous.

We illustrate first how to interpret interval-valued SHAP scores on the Diabetes dataset, which predicts whether a patient is diabetic from eight clinical variables: pregnancies (preg), plasma glucose (plas), blood pressure (pres), skinfold thickness (skin), insulin (insu), BMI (mass), diabetes pedigree (pedi), and age (age). Their direct clinical meaning makes this dataset suitable for interpretation.

We train a decision tree of depth (6), achieving a test accuracy of (0.76). Table IVa presents two representative patients, one predicted non-diabetic and one predicted diabetic. Table IVb reports their signed nominal SHAP values and the corresponding magnitude-based intervals at (s=2) under the pessimistic and averaging principles.

a) Patient 1 (predicted non-diabetic): The magnitudebased ranking indicate that low plasma glucose, lower BMI, and young age support the non-diabetic prediction. Under the pessimistic principle, the intervals for plas and BMI overlap, so their internal order is ambiguous: either feature may become the most important. By contrast, both remain clearly separated from all remaining features. Hence, while the exact top-1 feature is unstable, the top-2 set is robust, yielding the userfacing conclusion: “glucose and BMI are the two dominant drivers of the prediction, while age has a secondary $\mathit { e f f e c t . } ^ { \prime \prime }$ Under the averaging principle, the intervals are narrower and the intervals for plas and BMI no longer overlap, preserving their nominal order. This comparison provides a more cautious explanation than the nominal ranking alone, because making a non-risky prediction requires relying on several consistently influential features rather than over-interpreting the exact order between $\mathtt { p l a s }$ and BMI. If s increases further, additional interval expansion may eventually cause age to overlap with BMI, weakening the robust top-2 interpretation.

b) Patient 2 (predicted diabetic): Here, plasma glucose and pedigree contribute positively to the diabetic prediction, whereas BMI contributes negatively and therefore acts as a countervailing factor. Unlike Patient 1, the leading feature is unambiguous: plas is robustly top-1, as its interval is well separated from all others. Moreover, the intervals for pedi and BMI remain separated under both principles, although the gap is small. The averaging principle produces narrower intervals but leads to the same ranking conclusion. This supports the interpretation: “the prediction is dominated $b y$ glucose, while the ranking of the secondary drivers might be fragile.” This is also reasonable given the patient’s high plasma glucose value, together with a relatively high pedigree value and a lower BMI.

## C. Bootstrap-based validation of interval-valued attributions

We now move from individual instances to a dataset-level analysis. For Question 2, we examine how bootstrap retraining variability correlates with the interval-valued attributions. More specifically, we ask whether larger attribution intervals induced by our method also tend to exhibit a larger variability under bootstrap retraining. As illustrated in the preceding case study and expected from the theoretical formulation, the pessimistic principle generally produces wider intervals than the averaging one, thereby providing a more conservative assessment of attribution uncertainty. For the sake of simplicity, we therefore focus on pessimistic in the following experiments.

In this experiment, the number of bootstrap replicates is fixed to $B = 1 0 0$ for single trees (T) and for random forests (RF). For each fold, we first train the model on the whole training set and compute, for each test instance x, a vector

$$
\operatorname { l e n } ( \pmb { x } ) = ( \operatorname { l e n } _ { 1 } , \dots , \operatorname { l e n } _ { n } ) ,
$$

where len ${ \bf \Phi } _ { i } : = \overline { { { \phi } } } _ { i } - \underline { { { \phi } } } _ { i }$ is the proposed attribution interval length. Inside each fold, for each bootstrap replicate $b \in$ $\{ 1 , \cdots , B \}$ , we retrain the model on a bagging-style bootstrap sample (sampling observations uniformly with replacement) of the training set and compute the nominal attribution $\phi _ { i } ^ { ( b ) }$ for the same instance. A bootstrap-based vector is

$$
\operatorname { l e n } ^ { \mathrm { b o o t } } ( \pmb { x } ) = ( \mathrm { l e n } _ { 1 } ^ { \mathrm { b o o t } } , \dots , \mathrm { l e n } _ { n } ^ { \mathrm { b o o t } } ) ,
$$

where $\begin{array} { r } { \mathrm { { l e n } } _ { i } ^ { \mathrm { b o o t } } : = \ \operatorname* { m a x } _ { 1 \leq b \leq \mathcal { B } } \phi _ { i } ^ { ( b ) } - \operatorname* { m i n } _ { 1 \leq b \leq \mathcal { B } } \phi _ { i } ^ { ( b ) } } \end{array}$ is the bootstrap-based attribution interval length by the empirical min-max. Then we compute the Spearman correlation [20] of the two vectors given by the two approaches. This correlation evaluates whether the two approaches rank attribution length similarly according to attribution variability. The reported values are obtained by averaging the instance-level correlations over all test instances and folds.

The idea is to see if the proposed interval-valued attributions are aligned with the empirical variability produced by the bootstrap retraining. The results given in Table V confirm this relation. Across $s \in \{ 1 , 2 , 3 \}$ , the correlations varied by at most 0.01 over all dataset–model combinations, showing that the observed association is not specific to a certain s. So we only report the results for $s \ = \ 2$ . The correlations are moderately high for single trees and high for random forests. Since the proposed intervals avoid repeated model retraining, they provide a computationally convenient proxy for assessing bootstrap attribution sensitivity.

TABLE V: Correlation between SHAP interval lengths and bootstrap variability.
<table><tr><td>Dataset</td><td>Model</td><td>Spearman</td></tr><tr><td>Diab.</td><td>T</td><td>0.57</td></tr><tr><td>Diab.</td><td>RF</td><td>0.83</td></tr><tr><td>BreC.</td><td>T</td><td>0.36</td></tr><tr><td>BreC.</td><td>RF</td><td>0.92</td></tr><tr><td>Ions.</td><td>T</td><td>0.40</td></tr><tr><td>Ions.</td><td>RF</td><td>0.80</td></tr><tr><td>Mort.</td><td>T</td><td>0.55</td></tr><tr><td>Mort.</td><td>RF</td><td>0.90</td></tr></table>

TABLE VI: Correlation between the margin-based certainty score mar(x) and pessimistic SHAP interval length.
<table><tr><td>Dataset</td><td>Spearman</td></tr><tr><td>Diab.</td><td>-0.74</td></tr><tr><td>BreC.</td><td>-0.46</td></tr><tr><td>Ions.</td><td>-0.65</td></tr><tr><td>Mort.</td><td>-0.72</td></tr></table>

## D. Attribution in certain and ambiguous regions

To answer Question 3, we study whether attribution uncertainty is related to the model’s local prediction certainty. Here, we use a simple operational proxy based on the predicted class probabilities. For an instance x, let ${ \hat { p } } _ { k } ( { \pmb x } )$ denote the predicted probability of class $k ,$ and let $\hat { p } _ { ( 1 ) } ( \pmb { x } )$ and $\hat { p } _ { ( 2 ) } ( \pmb { x } )$ be the largest and second largest predicted probabilities. We define the margin score mar $\mathbf { \boldsymbol { \cdot } } ( \mathbf { \boldsymbol { x } } ) : = \hat { p } _ { ( 1 ) } ( \mathbf { \boldsymbol { x } } ) - \hat { p } _ { ( 2 ) } ( \mathbf { \boldsymbol { x } } )$ . For binary classification, this reduces to $\begin{array} { r } { m a r ( \pmb { x } ) = | \hat { p } _ { 1 } ( \pmb { x } ) - \hat { p } _ { 0 } ( \pmb { x } ) | = } \end{array}$ $| 2 \hat { p } _ { 1 } ( { \pmb x } ) - 1 | .$ . A large value of $m a r ( { \pmb x } )$ indicates that the prediction is confident, whereas a small value indicates that the instance lies in a more ambiguous region of the model. A single decision tree partitions the input space into discrete leaves and produces piecewise-constant class probabilities. As a result, its confidence scores are often too coarse to define meaningful certain and ambiguous regions based on the classprobability margin. In contrast, a random forest averages the outputs of many trees and typically yields a smoother range of confidence scores. We therefore do not use the same marginbased proxy for both model classes, and focus this analysis on random forests. Although this remains only a confidencebased proxy, it is natural in our setting: the confidence score is directly determined by leaf outputs in each tree, and by averaged leaf outputs in the forest.

For each instance x, we compare this margin score with the average attribution interval length across all features. We then compute the correlation between mar(x) and the length over the test instances. Consistently with Section IV-C, we use $s = 2$ for the reported results. Table VI shows a clear negative association for random forests: instances with smaller margin scores tend to have wider attribution intervals, which is expected as predictions in more ambiguous regions are also more fragile from the viewpoint of feature attribution.

## E. Application: debiasing uninformative features

Tree-based feature-importance measures are known to favor high-cardinality variables, especially in finite samples. The reason is that such variables offer more candidate splits, and therefore, have more opportunities to produce spurious impurity reductions or unstable local contributions. This effect has been studied for MDI, where [21] show that noisy features can receive non-zero importance, especially in deep trees with small leaves, and propose $\mathrm { M D I ^ { \mathrm { o o b } } } ^ { \mathrm { } }$ , which evaluates importance using out-of-bag samples. More recently, [13] show that global SHAP scores can suffer from a similar bias and propose $\mathrm { S H A P ^ { S h r u n k } }$ , a sample-splitting correction that shrinks in-sample SHAP values according to their agreement with out-of-sample SHAP values. They also propose a smoothed $\mathrm { S H A P ^ { o o b } }$ based on in-bag/out-of-bag samples. For Question $^ { 4 , }$ we study whether the proposed interval-valued attribution can be used as a conservative debiasing proxy for reducing the apparent importance of uninformative features.

We follow the synthetic protocol of [13]. The dataset contains five features $X _ { i }$ with different cardinalities: $X _ { 1 }$ is continuous and normally distributed while $X _ { 2 } , \cdots , X _ { 5 }$ are categorical features with 2,4,10,20 categories, respectively. In the setting, only $X _ { 2 }$ is informative, with $P ( Y = 1 | X _ { 2 } = 1 ) =$ $0 . 5 - \rho , ~ P ( Y = 1 | X _ { 2 } = 2 ) = 0 . 5 + \rho ,$ where $\rho \in [ 0 , 0 . 5 ]$ controls the signal strength. A higher $\rho$ indicates a higher importance level of $X _ { 2 }$ on the values of the class variable Y. The remaining features are generated independently of the values of $Y$ and are considered uninformative. For each value of $\rho ,$ we generate the prediction $Y .$ , split the data into the train/test set with a 75%/25% ratio, and train the model on the training set. To isolate the effect of $\rho ,$ we keep the feature values and the train/test split fixed across different values of $\rho ,$ regenerate only Y, and retrain the model. For our method, we fix $s = 2$

1) Global study: Motivated by the same phenomenon observed in [13], we consider a global feature-level analysis. Instead of recomputing SHAP values on held-out or out-of-bag samples, we use the lower attribution magnitude induced by the pessimistic principle. Specifically, for feature i, we define

$$
\mathrm { S H A P } _ { i } ^ { \mathrm { P e s s } } ( s ) = \frac { 1 } { J } \sum _ { j = 1 } ^ { J } \operatorname* { i n f } _ { \phi \in [ \underline { { \phi } } _ { i } ^ { U } ( x _ { j } ; s ) , \overline { { \phi } } _ { i } ^ { U } ( x _ { j } ; s ) ] } | \phi | ,
$$

where J is the number of explained test instances. In other words, $\mathrm { S H A P ^ { P e s s } }$ penalizes features whose apparent importance is not stable. It should not be interpreted as an out-ofsample validation score, since it does not directly test whether the attribution transfers to unseen samples. Rather, it provide a conservative proxy for attribution stability. Since highcardinality variables can create small or unstable leaves, this proxy can capture part of the same finite-sample fragility targeted by sample-splitting corrections. Using the same signalstrength protocol, we compute nominal SHAP, $\mathrm { S H A P ^ { \mathrm { \bar { P e s s } } } }$ and $\mathrm { \bar { S } H A \mathrm { \bar { P } ^ { S h r u n k } } }$ for each fold and average the results over folds. The results are shown in Figure 1. We observe that $\mathrm { S H A P ^ { P e s s } }$ reduces the importance assigned to uninformative high-cardinality features.

![](images/27f768e4d6008304e0f89bb19c5d2d3d0c081120d72d3643f02c5039cdd47272.jpg)  
Fig. 1: Comparison of (row) nominal SHAP, SH $\mathrm { A P ^ { P e s s } }$ $\mathrm { S H A P ^ { S h r u n k } }$ , and $\mathrm { S H A P ^ { o o b } }$ , where only $X _ { 2 }$ is informative, for different signal strengths (column) $\rho \in [ 0 , 0 . 3 ]$

The goal of this experiment is not to claim that the pessimistic principle fully removes all sources of high-cardinality bias. Rather, we ask whether the proposed intervals are sufficient to identify features whose importance is fragile. The results suggest that interval-valued SHAP can provide a tractable and interpretable debiasing proxy without retraining the model or recomputing SHAP on additional data splits.

2) Feature ranking: To further evaluate the global feature rankings as a classification task, we follow the feature-ranking setting of [13], [21], where they treat feature selection as a binary ranking problem since the informative features are known by construction: informative features are assigned label 1, and uninformative features are assigned label 0. This evaluation is useful in synthetic experiments because the informative set is controlled. However, in practical feature selection, declaring a feature completely uninformative is often too strong: a feature with small importance may still have a non-negligible contribution under the model mechanism. So we define a

TABLE VII: Average AUC scores for separating informative from uninformative features.
<table><tr><td>SHAP</td><td> $\mathrm { S H A P ^ { S a f e } }$ </td><td> $\mathrm { S H A P ^ { P e s s } }$ </td><td> $\mathrm { S H A P ^ { S h r u n k } }$ </td><td> $\mathrm { S H A P ^ { o o b } }$ </td></tr><tr><td>0.82</td><td>0.81</td><td>0.83</td><td>0.79</td><td>0.85</td></tr></table>

similar safe SHAP as

$$
\mathrm { S H A P } _ { i } ^ { \mathrm { S a f e } } ( s ) = \frac { 1 } { J } \sum _ { j = 1 } ^ { J } \operatorname* { s u p } _ { \phi \in [ \underline { { \phi } } _ { i } ^ { U } ( x _ { j } ; s ) , \overline { { \phi } } _ { i } ^ { U } ( x _ { j } ; s ) ] } | \phi | ,
$$

In the experiment, we generate $n ~ = ~ 2 0 0 0$ observations with 50 discrete features. The k-th feature $X _ { k }$ takes values in $\{ 0 , 1 , \ldots , k \}$ , and therefore has $k + 1$ distinct categories. We randomly select a set ${ \mathcal { S } } \subseteq \{ 1 , \ldots , 1 0 \}$ of five informative features, while the remaining features are uninformative. All features are generated independently. The binary response is then sampled according to

$$
P ( Y = 1 \mid X ) = \operatorname { L o g i s t i c } \left( { \frac { 2 } { 5 } } \sum _ { k \in { \mathcal { S } } } { \frac { X _ { k } } { k } } - 1 \right) .
$$

For each SHAP score, we compute the AUC for separating these two groups and then report the averaged AUC score over 5 repetitions. Equivalently, the AUC estimates the probability that a randomly chosen informative feature receives a larger importance score than a randomly chosen uninformative feature. Higher AUC therefore indicates better separation between true signal features and noisy features. The results are reported in Table VII. Overall, both $\mathrm { \dot { S } H A P ^ { P e s s } }$ and $\mathrm { S H A P ^ { S a f e } }$ achieve competitive performance compared with other methods. The AUC of $\mathrm { S H A P ^ { S a f e } }$ is slightly lower than that of $\mathrm { S H A P ^ { P e s s } }$ which is expected. Indeed, $\mathrm { S H A P ^ { S a f e } }$ has a different objective: instead of aggressively suppressing uncertain features, it is designed to retain features that may still be important under admissible perturbations. In this sense, it provides a safer criterion for feature selection, especially when removing a potentially informative feature would be undesirable.

## V. CONCLUSION

In this paper, we propose an IDM-based method to analyze the robustness of SHAP values in tree-based models. It quantifies interval-valued SHAP when a few virtual samples are introduced into the tree leaves. These intervals, defined under the pessimistic and averaging principles, can be computed efficiently. We present case studies and experiments illustrating their behavior and relationship with the empirical variability produced by bootstrap retraining. We also show how our method can better estimate the importance of uninformative features.

As future work, exact computation under both principles could be extended to tree ensembles. One approach is to translate an ensemble into a potentially large single tree and apply the exact solutions presented in Section III. However, constructing this tree and handling its many leaves may be costly. Alternatively, the optimal allocation problem could be formulated as an integer nonlinear program. The case $s = 1$ may be handled efficiently using heuristic algorithms, whereas general values of s and large ensembles remain challenging.

## ACKNOWLEDGMENTS

This work was partially supported by the Junior Professor Chair in Trustworthy AI (Ref. ANR-R311CHD).

## REFERENCES

[1] S. M. Lundberg, G. G. Erion, S.-I. Lee, Consistent individualized feature attribution for tree ensembles, arXiv preprint arXiv:1802.03888 (2018).

[2] S. M. Lundberg, S.-I. Lee, A unified approach to interpreting model predictions, in: Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, Curran Associates Inc., Red Hook, NY, USA, 2017, p. 4768–4777.

[3] S. M. Lundberg, G. Erion, H. Chen, A. DeGrave, J. M. Prutkin, B. Nair, R. Katz, J. Himmelfarb, N. Bansal, S.-I. Lee, Explainable ai for trees: From local explanations to global understanding, arXiv preprint arXiv:1905.04610 (2019).

[4] L. Grinsztajn, E. Oyallon, G. Varoquaux, Why do tree-based models still outperform deep learning on typical tabular data?, Advances in neural information processing systems 35 (2022) 507–520.

[5] C. C. Kasbohrer, S. Mair, L. Jiang, Assessing the fragility of shap-¨ based model explanations using counterfactuals, in: Northern Lights Deep Learning Conference 2026.

[6] M. Last, O. Maimon, E. Minkov, Improving stability of decision trees, International journal of pattern recognition and artificial intelligence 16 (02) (2002) 145–159.

[7] L. V. Utkin, A. V. Konstantinov, K. A. Vishniakov, An imprecise shap as a tool for explaining the class probability distributions under limited training data, arXiv preprint arXiv:2106.09111 (2021).

[8] M. Banerjee, E. Reynolds, H. B. Andersson, B. K. Nallamothu, Treebased analysis: a practical approach to create clinical decision-making tools, Circulation: Cardiovascular Quality and Outcomes 12 (5) (2019) e004879.

[9] Q. Xu, W. Xie, B. Liao, C. Hu, L. Qin, Z. Yang, H. Xiong, Y. Lyu, Y. Zhou, A. Luo, Interpretability of clinical decision support systems based on artificial intelligence from technological and medical perspective: a systematic review, Journal of healthcare engineering 2023 (1) (2023) 9919269.

[10] E. T. Tetteh, B. Zielosko, Optimization of decision rules derived from random forest, Procedia Computer Science 270 (2025) 5453–5462.

[11] J.-M. Bernard, An introduction to the imprecise dirichlet model for multinomial data, International Journal of Approximate Reasoning 39 (2) (2005) 123–150, imprecise Probabilities and Their Applications.

[12] T.-H. Do, V.-L. Nguyen, Y. Grandvalet, Probabilistic multi-dimensional classification with incomplete data at the prediction time, in: The 29th International Conference on Artificial Intelligence and Statistics, 2026.

[13] M. Loecher, Debiasing shap scores in random forests, AStA Advances in Statistical Analysis 108 (2) (2024) 427–440.

[14] C. Molnar, Interpretable Machine Learning, 2nd Edition, Leanpub, 2022.

[15] K. Aas, M. Jullum, A. Løland, Explaining individual predictions when features are dependent: More accurate approximations to shapley values, Artificial Intelligence 298 (2021) 103502.

[16] A. Karczmarz, T. Michalak, A. Mukherjee, P. Sankowski, P. Wygocki, Improved feature importance computation for tree models based on the banzhaf value, in: J. Cussens, K. Zhang (Eds.), Proceedings of the Thirty-Eighth Conference on Uncertainty in Artificial Intelligence, Vol. 180 of Proceedings of Machine Learning Research, PMLR, 2022, pp. 969–979.

[17] A. P. Dempster, Upper and lower probabilities induced by a multivalued mapping, in: Classic works of the Dempster-Shafer theory of belief functions, Springer, 2008, pp. 57–72.

[18] P. Walley, Inferences from multinomial data: Learning about a bag of marbles, Journal of the Royal Statistical Society: Series B (Methodological) 58 (1) (1996) 3–34.

[19] H. Zhang, B. Quost, M.-H. Masson, Cautious random forests: a new decision strategy and some experiments, in: A. Cano, J. De Bock, E. Miranda, S. Moral (Eds.), Proceedings of the Twelveth International Symposium on Imprecise Probability: Theories and Applications, Vol. 147 of Proceedings of Machine Learning Research, PMLR, 2021, pp. 369–372.

[20] Chapter 2 - descriptive statistics ii: Bivariate and multivariate statistics, in: Statistics for Biomedical Engineers and Scientists, Academic Press, 2019, pp. 23–56.

[21] X. Li, Y. Wang, S. Basu, K. Kumbier, B. Yu, A debiased mdi feature importance measure for random forests, Advances in Neural Information Processing Systems 32 (2019).