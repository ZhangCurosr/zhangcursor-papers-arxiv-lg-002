# LEARNING THE ROBUSTNESS MECHANISM WITH BILEVEL OPTIMIZATION

Yiyang Shen   
Department of Informatics   
University of Iowa   
yiyang-shen@uiowa.edu   
Qihang Lin   
Tippie College of Business   
University of Iowa   
qihang-lin@uiowa.edu Weiran Wang   
Department of Computer Science University of Iowa   
weiran-wang@uiowa.edu

## ABSTRACT

We propose a distributionally robust learning framework where parameters defining the robustness mechanism are learned from held-out data instead of extensively tuned. Using bilevel optimization with both upper and lower level minimax problems, we create two instances of our framework to tackle setups with and without group labels in the training set. Theoretically, we provide sample complexity analysis for our robustness mechanism learning paradigm, showing that it achieves generalization guarantees comparable to exhaustive grid search while being more computationally efficient. Empirically, we evaluate our framework under a challenging setup when both intra-group and inter-group test distribution shifts occur at the same time, thereby demonstrating the efficacy and scalability of our method.

## 1 INTRODUCTION

Many machine learning methods require optimizing model parameters to minimize the empirical risk or average sample loss. The empirical risk minimization (ERM) paradigm assumes that unseen data are sampled from the same distribution as seen training data are. Since ERM weighs all sam ples equally, it is particularly vulnerable to subpopulation shift, where the training samples consist of several groups divided by spurious attributes whose proportions are different from those of the test data (Sagawa et al., 2020; Shen et al., 2021; Cai et al., 2021; Yang et al., 2023; Yu et al., 2024). In real-world applications such as healthcare (Zech et al., 2018; Badgeley et al., 2019), fairness (Buolamwini & Gebru, 2018; Mehta et al., 2024; Lei et al., 2024), robotics (Ryu & Mehr, 2024), and autonomous driving (Zhang et al., 2017; Azizi et al., 2025), the classifier parameters may inadvertently depend on spurious attributes, causing failures when the testing environment is different.

Distributionally robust optimization (DRO, Duchi et al. (2021)) along with many out-of-distribution generalization algorithms (Arjovsky et al., 2019; Sohoni et al., 2020; Krueger et al., 2021) are developed to address this issue. A simplified setup for DRO is Group DRO (GDRO, Sagawa et al. (2020)), which assumes grouping among samples and robustifies the model by minimizing the training loss of the group with the highest loss or worst training accuracy. Take the widely used CelebrityAttributes (CelebA) benchmark as an example, an ERM-trained model tend to correlate golden hair color (class labels) with female (attribute) celebrities, leading to severe performance degradation on minority groups, male celebrities with blond hair. Thus, GDRO explicitly minimizes the loss of the group incurring high loss. This has inspired a growing line of research which aims at developing outof-distribution (OOD) generalization algorithms for the GDRO setup(Ahmed et al., 2021; Creager et al., 2021; Piratla et al., 2022; Izmailov et al., 2022; Nam et al., 2022; Seo et al., 2022; Asgari et al., 2022; Zhang et al., 2022a; Ghosal & Li, 2023; Paranjape et al., 2023; Wu et al., 2023; Deng et al., 2023; Han & Zou, 2024; Jain et al., 2024; LaBonte et al., 2023; Pezeshki et al., 2024; Jeong et al., 2025; Jo et al., 2026). Most methods fall under two general categories of setup. One stream focuses on the more ideal setup where all attribute labels are known, so the algorithms know the ground truth group membership of samples during training. The other focuses on a weakly group-supervised or group-unsupervised setup which is more realistic since attribute labels that define groups are often unavailable. Typically, a worst-group identification model is trained (e.g., worst loss samples in ERM, attribute prediction) during the first stage with or without a small amount of group-labeled data; in the second stage, robust training is done using those pseudo-labeled groups, e.g., GDRO.

GDRO is suited to address shifts in group distributions, i.e., inter-group subpopulation shift, which leads to failures concentrated in minority groups. However, treating groups as fixed distributions overlooks another type of uncertainty: the conditional distribution within a group, i.e., intra-group subpopulation shift (Ben-Tal & Nemirovski, 2002; Devroye et al., 2013; Duchi & Namkoong, 2021). For example, a minority group at training time may contain a small variation of environment, while the test distribution gives underrepresented variants of the same group. DRO with hierarchical ambiguity set (HDRO, Jo et al. (2026)) addresses this issue by introducing an adversarial perturbation within each group, which controls how much intra-group variation the model protects against. However, the appropriate amount of robustness signals per group is generally unknown, may not be uniform across groups, and requires extensive tuning. Similarly, when group membership is unknown at training time, worst group membership predictor typically is crucial to robustness and requires tuning as well.

As such, while effective at tackling various adversarial conditions, advanced robustness mechanisms require users to manually specify the type of uncertainty, and the level of uncertainty the trained model should be robust against under different evaluation distributions. This introduced many more potentially sensitive models and hyper-parameters, making exhaustive tuning expensive and naturally raises the following question:

Can we treat the robustness mechanism itself as an active, integral component to be learned during active training?

Our affirmative answer contributes to DRO research in three aspects:

1. We propose a bilevel adaptive tuning framework that directly learns a given robustness mechanism based on the validation set, with a minimax problem for tuning robustness parameters on the upper level and a minimax problem for robust training on the lower level. With two instantiations, we show such bilevel problem can be tractably solved by a first-order method proposed by Shen et al. (2026).

2. While most existing theory is concerned with the optimization complexity of solving such empirical problem to certain optimality conditions, such as saddle point and ϵ-KKT point (Lu & Mei, 2024), we derive generalization guarantees for our bilevel framework, providing the sample complexity for learning robust model which is new to the best of our knowledge. The guarantee shows advantage of continuous optimization of the hyperparameters avoids discretization error compared to grid search.

3. We validate and enhance the evaluation setting proposed by Jo et al. (2026) where both inter-group and intra-group minority group test distribution shifts are manifest, showing significant improvement on worst-group prediction using our methods both with and without group labels at training time.

## 2 A BILEVEL ADAPTIVE FRAMEWORK FOR GDRO

## 2.1 GROUP DISTRIBUTIONALLY ROBUST OPTIMIZATION

Suppose there are $G$ groups in the data distribution. Let $( x , y , a )$ be a data point, where x is the feature vector, y is the target variable, and $a \in { \mathcal { A } }$ is an attribute label that defines the group the data point belongs to together with known y. We consider a predictive task where the goal is to predict y based on x through a model $f _ { W , \theta } ( x ) : = W h _ { \theta } ( x )$ , where $h _ { \theta } ( x )$ is a mapping parameterized by θ that produces a representation of x while W is a matrix that defines a linear model that produces the prediction $W h _ { \theta } ( \bar { x } )$ for y.

Let $D _ { \mathrm { t r } } ^ { ( g ) } = \{ ( x _ { g , i } ^ { \mathrm { t r } } , y _ { g , i } ^ { \mathrm { t r } } ) \} _ { i = 1 } ^ { n _ { g } ^ { \mathrm { t r } } }$ be a set of $n _ { g } ^ { \mathrm { t r } }$ training samples from group g for $g \in \{ 1 , \ldots , G \}$ , and $\ell ( f _ { W , \theta } ( x ) , y )$ be a loss function that measures the discrepancy between the prediction $f _ { W , \theta } ( x )$ and the target y. The average training loss on group g is $\begin{array} { r } { L _ { g } ^ { \mathrm { t r } } ( W ; \theta ) : = \frac { 1 } { n _ { \sigma } ^ { \mathrm { t r } } } \sum _ { i = 1 } ^ { n _ { g } ^ { \mathrm { t r } } } \ell \left( f _ { W , \theta } ( x _ { g , i } ^ { \mathrm { t r } } ) , y _ { g , i } ^ { \mathrm { t r } } \right) } \end{array}$ The GDRO model for learning $W$ and θ can be formulated as

$$
( W ^ { \star } , \theta ^ { \star } ) \in \arg \operatorname* { m i n } _ { W , \theta } \operatorname* { m a x } _ { q \in \Delta _ { \mathcal { G } } } \left\{ \sum _ { g = 1 } ^ { G } q _ { g } L _ { g } ^ { \mathrm { t r } } ( W ; \theta ) - \frac { \eta } { 2 } \left\| q - \frac { 1 } { G } { \bf 1 } \right\| _ { 2 } ^ { 2 } + \frac { \lambda } { 2 } \| W \| _ { F } ^ { 2 } + \frac { \lambda } { 2 } \| \theta \| _ { 2 } ^ { 2 } \right\} ,\tag{1}
$$

where $\Vert \cdot \Vert _ { F }$ denotes the Frobenius norm, $\Delta _ { G } : = \{ q = ( q _ { 1 } , \dots , q _ { G } ) | q _ { g } \geq 0 , 1 \leq g \leq G , \mathbf { 1 } ^ { \top } q = 1 \}$ $\eta \geq 0$ is the robustness parameter (Huang et al., 2021; Zhang et al., 2022b), and $\lambda \geq 0$ is the regularization parameter. Here, the goal is to achieve a robust performance of $f _ { W , \theta }$ by minimizing a weighted loss over groups with more weight put on the groups of larger average losses. Note that η controls how far the group weight q may deviate from uniform weighting, $\operatorname { i . e , 1 / } G$ . Naturally, it is critical to select η in (1) to achieve the best out-of-sample performance. Typically, a validation set is used, denoted by $D _ { \mathrm { v a l } } ^ { ( g ) } = \{ ( x _ { g , i } ^ { \mathrm { v a l } } , y _ { g , i } ^ { \mathrm { v a l } } ) \} _ { i = 1 } ^ { n _ { g } ^ { \mathrm { v a l } } }$ for group $g \in \{ 1 , \ldots , G \}$ , to evaluate the robustness of the performance of $f _ { W , \theta }$ , for example, in its largest validation loss among the groups, i.e., <sup>1</sup>

$$
\operatorname* { m a x } _ { p \in \Delta _ { G } } \sum _ { g = 1 } ^ { G } p _ { g } L _ { g } ^ { \mathrm { v a l } } ( W ^ { \star } ; \theta ^ { \star } ) , \quad \mathrm { w h e r e } \quad L _ { g } ^ { \mathrm { v a l } } ( W ; \theta ) : = \frac { 1 } { n _ { g } ^ { \mathrm { v a l } } } \sum _ { i = 1 } ^ { n _ { g } ^ { \mathrm { v a l } } } \ell \left( f _ { W , \theta } ( x _ { g , i } ^ { \mathrm { v a l } } ) , y _ { g , i } ^ { \mathrm { v a l } } \right) .\tag{2}
$$

Then a value of $\eta$ is selected from a grid to minimize (2). While widely used, this approach requires training a model for each candidate of η and does not directly extend to the setting where the attribute label a is missing from most of the training data.

## 2.2 WARM-UP: BILEVEL GROUP DRO (BI-GDRO)

An adaptive bilevel group DRO method can be used to address the challenges caused by tuning. Instead of training the full model as in (1) and selecting η based on (2), we integrate the training and the parameter tuning into a bilevel optimization model as follows

$$
\begin{array} { r l r } {  { \operatorname* { m i n } _ { W , \theta , \eta \geq 0 } \operatorname* { m a x } _ { \rho \in \Delta _ { G } } \sum _ { g = 1 } ^ { G } p _ { g } L _ { g } ^ { \mathrm { v a l } } ( W ; \theta ) + \frac { \lambda } { 2 } \| \theta \| _ { 2 } ^ { 2 } } } \\ & { } & { \mathrm { s . t . } \ W \in \arg \operatorname* { m i n } _ { W ^ { \prime } } \operatorname* { m a x } _ { q \in \Delta _ { G } } \{ \sum _ { g = 1 } ^ { G } q _ { g } L _ { g } ^ { \mathrm { t r } } ( W ^ { \prime } ; \theta ) - \frac { \eta } { 2 } \| q - \frac { 1 } { G } { \bf 1 } \| _ { 2 } ^ { 2 } + \frac { \lambda } { 2 } \| W ^ { \prime } \| _ { F } ^ { 2 } \} . } \end{array}\tag{3}
$$

(4)

Different from (1) and (2), η becomes a continuous upper-level decision variable in (3) without being limited in a finite grid. Parameter θ is another upper-level decision variable while the linear model $W$ is optimized in the lower-level problem (4). This way, W is learned from the training data for any given θ and $\eta ,$ while θ and $\eta$ are learned by optimizing the performance of $f _ { W , \theta }$ on the validation set. Note that, with this design, the lower-level problem (4) becomes convex in $W ^ { \prime }$ and concave in $q ,$ which is required by most algorithms for bilevel optimization<sup>2</sup>, although the upper-level objective function in (3) can be nonconvex jointly in $W$ and θ.

This bilevel minimax hyperparameter optimization model has been studied by Shen et al. (2026). However, as we show below, the framework like (3) can be extended to learn additional elements of a robustness mechanism that is more general than (1), such as the radius of an ambiguity set and the latent group structure itself when group attribute labels are unavailable during training.

## 2.3 BILEVEL-HIERARCHICAL DRO (BI-HDRO)

GDRO can be extended into hierarchical DRO (HDRO) whose ambiguity set has a hierarchical structure (Jo et al., 2026). Let $P _ { g }$ be the empirical distribution on $D _ { \mathrm { t r } } ^ { ( g ) }$ . The HDRO in our notation

can be formulated as

$$
\operatorname* { m i n } _ { W , \theta } \operatorname* { m a x } _ { Q \in \mathcal { Q } } \mathbb { E } _ { ( X , Y ) \sim Q } [ \ell \left( f _ { W , \theta } ( X ) , Y \right) ] ,\tag{5}
$$

where (X, Y) denotes a random data point, ${ \mathbb E } _ { ( X , Y ) \sim Q }$ denotes the expectation taken over $( X , Y )$ when (X, Y) follows distribution $Q ,$ and

$$
\mathcal { Q } = \left\{ \sum _ { g = 1 } ^ { G } q _ { g } Q _ { g } : \ q \in \Delta _ { G } , \ W _ { \infty } ( Q _ { g } , P _ { g } ) \leq \epsilon _ { g } , \ \mathrm { f o r } g = 1 , \ldots , G \right\}
$$

is the hierachical ambiguity set, where $Q _ { g }$ is a distribution of $( X , Y ) , W _ { \infty } ( Q _ { g } , P _ { g } )$ is the $\infty \mathrm { \Omega }$ Wasserstein distance between $Q _ { g }$ and $P _ { g } ,$ , and $\epsilon _ { g }$ is a radius. As in (1), weights q model the changes in group proportions, i.e., inter-group shift, while $Q _ { g }$ is used to further to accommodate shifts within each group, i.e., intra-group shift. Note that (5) is reduced to (1) when $\epsilon _ { g } = 0$ since $Q _ { g } = P _ { g }$

Direct optimization over the distributions of $Q _ { g }$ is generally intractable. Therefore, Jo et al. (2026) (Theorem 4.1) proposed solving the following upper approximation of (5)

$$
( W ^ { \star } , \theta ^ { \star } ) \in \arg \operatorname* { m i n } _ { W , \theta } \operatorname* { m a x } _ { q \in \Delta _ { G } } \sum _ { g = 1 } ^ { G } q _ { g } L _ { g } ^ { \mathrm { t r } , \epsilon _ { g } } ( W ; \theta )\tag{6}
$$

$$
\mathrm { w h e r e } \quad L _ { g } ^ { \mathrm { t r } , \epsilon _ { g } } ( W ; \theta ) : = \frac { 1 } { n _ { g } ^ { \mathrm { t r } } } \sum _ { i = 1 } ^ { n _ { g } ^ { \mathrm { t r } } } \left[ \operatorname* { m a x } _ { z : \| z - h _ { \theta } ( x _ { g , i } ^ { \mathrm { t r } } ) \| \le \epsilon _ { g } } \ell \big ( W z , y _ { g , i } ^ { \mathrm { t r } } \big ) \right] .\tag{7}
$$

Note that $L _ { q } ^ { \mathrm { t r } } ( W ; \theta ) \leq L _ { g } ^ { \mathrm { t r } , \epsilon _ { g } } ( W ; \theta )$ . In (6), we minimize the largest loss over groups when the latent representation of each sample can be adversarially perturbed within a ball of radius $\epsilon _ { g }$ . The perturbation makes the model more robust to test-distribution intra-group shift.

Problems with HDRO (1) Solving the inner maximization over z for each data point to evaluate $L _ { g } ^ { \mathrm { t r } , \epsilon _ { g } } ( W ; \theta )$ is computationally challenging when $n _ { g } ^ { \mathrm { t r } }$ is large. Jo et al. (2026) proposed a heuristic method that performs one step of gradient ascent over z from $h _ { \theta } ( x _ { g , i } ^ { \mathrm { t r } } )$ , which only solves the inner maximization suboptimally and thus approximates $L _ { g } ^ { \mathrm { t r } , \epsilon _ { g } } ( W ; \theta )$ and its gradient poorly. $( 2 ) \epsilon _ { g }$ requires additional tuning for each $g .$ Although Jo et al. (2026) (Appendix D.3) proposed tuning a scalar ϵ with $\epsilon _ { g } = \epsilon / \sqrt { n _ { g } ^ { \mathrm { t r } } }$ , this remains heuristic and still requires grid search over ϵ based on a performance metric on the validation set such as (6).

Our Solution (1) For each $^ { g , }$ we propose a modification of $L _ { g } ^ { \mathrm { t r } , \epsilon _ { g } } ( W ; \theta )$ , denoted by $\tilde { L } _ { g } ^ { \mathrm { t r } , \epsilon _ { g } } ( W ; \theta ; u )$ with an additional variable u. The specific form of $\tilde { L } _ { g } ^ { \mathrm { t r } , \epsilon _ { g } }$ depends on the prediction task and the loss function $\ell ( \cdot , \cdot )$ . We show that, for a binary classification problem where ℓ is either the hinge loss or the logistic loss, $\tilde { L } _ { g } ^ { \mathrm { t r } , \epsilon _ { g } } ( W ; \theta ; u )$ is jointly convex in $W$ and u, and (6) equals

$$
\operatorname* { m i n } _ { W , \theta , u } \operatorname* { m a x } _ { q \in \Delta _ { G } } \sum _ { g = 1 } ^ { G } q _ { g } \tilde { L } _ { g } ^ { \mathrm { t r } , \epsilon _ { g } } ( W ; \theta ; u ) \mathrm { s . t . } r ( W , u ) \leq 0 ,\tag{8}
$$

where $r ( W , u )$ is a jointly convex function of $W$ and u. For a multiclass classification problem where ℓ is the cross-entropy loss, we show that the corresponding $\tilde { L } _ { g } ^ { \mathrm { t r } , \epsilon _ { g } } ( W ; \theta ; u _ { g } )$ and $r ( W , u )$ are still jointly convex in W and u but (8) is only an upper bound of (6). Therefore, we propose solving (8) as the gradient of $\tilde { L } _ { g } ^ { \mathrm { t r } , \epsilon _ { g } }$ can be evaluated exactly without solving the inner maximization problems. Furthermore, in all the aforementioned cases, we can show that projection to the constraint set defined by the inequality $r ( W , u ) \leq 0$ has a closed form, meaning that (8) is not computationally more difficult than (6). We present the details in Appendix C. (2) Similar to (3), we can tune $\epsilon _ { g }$ in a bilevel optimization model based on the performance on the validation set after adding $\{ \epsilon _ { g } \} _ { g = 1 } ^ { \check { G } }$ as

upper-level decision variables just like η:

$$
\begin{array} { r l r } { \displaystyle \operatorname* { m i n } _ { \substack { W , \theta , u , \eta \geq 0 , \left\{ \epsilon _ { g } \right\} _ { g = 1 } ^ { G } \operatorname* { m a x } _ { g \in \Delta _ { G } } \sum _ { g = 1 } ^ { G } p _ { g } L _ { g } ^ { \mathrm { v a l } } \left( W ; \theta \right) + \frac { \lambda } { 2 } \| \theta \| _ { 2 } ^ { 2 } } } } & { \displaystyle ( 9 ) } \\ { \mathrm { s . t . } \ W \in \arg \operatorname* { m i n } _ { W ^ { \prime } , u ^ { \prime } } \operatorname* { m a x } _ { \substack { q \in \Delta _ { G } } } \left\{ \sum _ { g = 1 } ^ { G } q _ { g } \tilde { L } _ { g } ^ { \mathrm { t r . } \epsilon _ { g } } \left( W ^ { \prime } ; \theta ; u ^ { \prime } \right) - \frac { \eta } { 2 } \left\| q - \frac { 1 } { G } \mathbf { 1 } \right\| _ { 2 } ^ { 2 } + \frac { \lambda } { 2 } \| W ^ { \prime } \| _ { F } ^ { 2 } \right\} , } & \\ { \mathrm { s . t . } \ r ( W ^ { \prime } , u ^ { \prime } ) \leq 0 , } \end{array}
$$

where we’ve replaced the loss $L ^ { \mathrm { t r } , \epsilon _ { g } }$ in (4) to be $\tilde { L } ^ { \mathrm { t r } , \epsilon _ { g } }$

## 2.4 BILEVEL-PROBABILISTIC GROUP DRO (BI-PG-DRO)

In real-world scenarios, it is possible that only a very small portion of data has attribute label a so we are not able to formulate $\bar { L } _ { a } ^ { \mathrm { t r } } ( W ; \theta )$ using all data points due to the lack of group information. To address this issue, Ghosal & Li (2023) proposed PG-DRO, a robustness mechanism that uses a small amount of attribute-labeled training data to train a soft group predictor to generate pseudomembership labels before using a robust model such as GDRO for training (Sagawa et al., 2020).

Formally, let $D ^ { \mathrm { u l } } = \{ ( x _ { i } ^ { \mathrm { u l } } , y _ { i } ^ { \mathrm { u l } } ) \} _ { i = 1 } ^ { n ^ { \mathrm { u l } } }$ be a separate subset without group labels. We assume $n _ { g } ^ { \mathrm { t r } } \ll$ $n ^ { \mathrm { u l } }$ for any $\boldsymbol { g } = ( a , y )$ , where attribute label a and class label $y$ jointly determines the group. PG DRO introduces another classification model $\tilde { f } _ { \phi } ( x )$ parameterized by ϕ and train $\tilde { f } _ { \phi } ( x )$ on $D _ { \mathrm { t r } } ^ { ( g ) }$ to predict the attribute label $a \in { \mathcal { A } }$ based on x. For each data sample $( x _ { i } ^ { \mathrm { u l } } , y _ { i } ^ { \mathrm { u l } } )$ from $D ^ { \mathrm { u l } }$ , we assume $\tilde { f } _ { \phi } ( x _ { i } ^ { \mathrm { u l } } ) = ( \gamma _ { i 1 } ( \phi ) , \gamma _ { i 2 } ( \phi ) , \ldots , \gamma _ { i G } ( \phi ) ) ^ { \top }$ where $\gamma _ { i g } ( \phi )$ is the predicted probability of $( x _ { i } ^ { \mathrm { u l } } , y _ { i } ^ { \mathrm { u l } } )$ being in group g for each g, since class label is known. Using this conditional probability as soft group labels, we can assign a fraction of $( x _ { i } ^ { \mathrm { u l } } , y _ { i } ^ { \mathrm { u l } } )$ to each group, yielding the probabilistic loss

$$
L _ { g } ^ { \mathrm { P G } } ( W ; \theta , \phi ) = \frac { \sum _ { i = 1 } ^ { n ^ { \mathrm { u l } } } \gamma _ { i g } ( \phi ) \ell \left( f _ { W , \theta } ( x _ { i } ^ { \mathrm { u l } } ) , y _ { i } ^ { \mathrm { u l } } \right) } { \sum _ { i = 1 } ^ { n ^ { \mathrm { u l } } } \gamma _ { i g } ( \phi ) + \epsilon } ,
$$

where ϵ is a smoothing parameter to avoid a zero denominator. Then PG-DRO solves

$$
( W ^ { \star } , \theta ^ { \star } ) \in \arg \operatorname* { m i n } _ { W , \theta } \operatorname* { m a x } _ { q \in \Delta _ { G } } \sum _ { g = 1 } ^ { G } q _ { g } L _ { g } ^ { \mathrm { P G } } ( W ; \theta , \phi ) .\tag{10}
$$

However, PG-DRO requires additional training for $\tilde { f } _ { \phi } ( x )$ . For a more efficient training approach, we propose to integrate the training of $f _ { W , \theta } ( x )$ and $\tilde { f } _ { \phi } ( x )$ as well as the tuning of the robustness parameter into a bilevel optimization model below

$$
\operatorname* { m i n } _ { W , \theta , \phi , \eta \geq 0 } \operatorname* { m a x } _ { p \in \Delta _ { G } } \sum _ { g = 1 } ^ { G } p _ { g } L _ { g } ^ { \mathrm { v a l } } ( W ; \theta ) + \frac { \lambda } { 2 } \| \theta \| _ { 2 } ^ { 2 } + \beta \mathrm { K L } ( \pi _ { \mathrm { t r } } | | \pi _ { \phi } )\tag{11}
$$

$$
\mathrm { s . t . } \ W \in \arg \operatorname* { m i n } _ { W ^ { \prime } } \operatorname* { m a x } _ { q \in \Delta _ { \mathcal { G } } } \left\{ \sum _ { q = 1 } ^ { G } q _ { g } L _ { g } ^ { \mathrm { P G } } ( W ^ { \prime } ; \theta , \phi ) - \frac { \eta } { 2 } \left\| q - \frac { 1 } { G } { \bf 1 } \right\| _ { 2 } ^ { 2 } + \frac { \lambda } { 2 } \| W ^ { \prime } \| _ { F } ^ { 2 } \right\} .\tag{12}
$$

Here, $\begin{array} { r } { \pi _ { \phi } = ( \sum _ { i = 1 } ^ { n ^ { \mathrm { u l } } } \gamma _ { i g } ( \phi ) / n ^ { \mathrm { u l } } ) _ { g = 1 } ^ { G } } \end{array}$ is the proportion of data points predicted to be in group g by model $\tilde { f } _ { \phi } , \pi _ { \mathrm { t r } } = ( { n _ { g } ^ { \mathrm { t r } } } / ( { \sum _ { g ^ { \prime } = 1 } ^ { G } n _ { g ^ { \prime } } ^ { \mathrm { t r } } } ) ) _ { g = 1 } ^ { G }$ is an estimation of the prior distribution of the group labels, and $\mathrm { K L } ( \pi _ { \mathrm { t r } } | | \pi _ { \phi } )$ is the Kullback-Leibler (KL) divergence between $\pi _ { \phi }$ and $\pi _ { \mathrm { t r } }$ . Different from PG-$\mathrm { D R O } , \tilde { f } _ { \phi }$ in (11) is not trained separately on a binary cross-entropy (BCE) loss to predict $a .$ Instead, ϕ optimized in the upper-level in (11) such that the produced $\gamma _ { i g }$ helps ensure a good performance of the resulting $f _ { W , \theta } ( x )$ on the validation set, and a high prediction accuracy of $\tilde { f } _ { \phi }$ is not necessary. One may replace $\mathrm { K L } ( \pi _ { \mathrm { t r } } | | \pi _ { \phi } )$ in (11) to the training loss of $\tilde { f } _ { \phi }$ in predicting the group label $g .$ Empirical findings (Appendix E.3) show that this has little impact on the numerical performance but using $\mathrm { K L } ( \pi _ { \mathrm { t r } } | | \pi _ { \phi } )$ as the regularizer makes the optimization more lightweight.

## 2.5 BILEVEL MINIMAX ALGORITHM

Despite their different formulations, Problems $( 3 ) , ( 9 )$ , and (11) can be solved using the first-order method proposed by Shen et al. (2026). We provide the algorithm’s pseudocode in Appendix A and summarize it here. Firstly, these problems are instances of the following bilevel minimax problem

$$
\operatorname* { m i n } _ { \alpha , W , q } \left\{ \operatorname* { m a x } _ { p } F ( \alpha , p , W , q ) \ \bigg | \ ( W , q ) \in \arg \operatorname* { m i n } _ { \widetilde { W } } \operatorname* { m a x } _ { \widetilde { q } } \widetilde { F } ( \alpha , \widetilde { w } , \widetilde { q } ) \right\} ,\tag{13}
$$

where α denotes the collection of all primal upper-level decision variables, including $\theta , \eta , \epsilon _ { g }$ and $\phi$ in the three bilevel models above, $p$ is the upper-level group weight, q is the lower-level group weight, and $W$ is the parameter of the linear classifier within model $f _ { W , \theta }$ . Here, $F$ and $\overrightharpoon { F }$ are different objectives. The primal and dual value functions of the lower-level minimax problem of (13) are denoted by

$$
V _ { \mathrm { P } } ( \alpha , W ) : = \operatorname* { m a x } _ { \widetilde { q } } \widetilde { F } ( \alpha , W , \widetilde { q } ) \quad \mathrm { a n d } \quad V _ { \mathrm { D } } ( \alpha , q ) : = \operatorname* { m i n } _ { \widetilde { W } } \widetilde { F } ( \alpha , \widetilde { W } , q ) ,
$$

respectively. Therefore, Problem (13) can be equivalently written as the single-level constrained problem with a primal-dual gap constraint

$$
\operatorname* { m i n } _ { \alpha , W , q } \left\{ \operatorname* { m a x } _ { p } F ( \alpha , p , W , q ) \biggm | V _ { \mathrm { P } } ( \alpha , W ) - V _ { \mathrm { D } } ( \alpha , q ) \leq 0 \right\} .
$$

Lastly, introducing a penalty parameter $\rho > 0$ yields the penalized minimax problem

$$
\operatorname* { m i n } _ { \alpha , W , q } \operatorname* { m a x } _ { p } \left\{ F ( \alpha , p , W , q ) + \rho \left[ V _ { \mathrm { P } } ( \alpha , W ) - V _ { \mathrm { D } } ( \alpha , q ) \right] \right\} = \operatorname* { m i n } _ { \alpha , W , q } \operatorname* { m a x } _ { p , \widetilde { W } , \widetilde { q } } P _ { \rho } \left( \alpha , p , W , q , \widetilde { W } , \widetilde { q } \right) ,\tag{14}
$$

where $P _ { \rho }$ denotes the corresponding penalized objective. To compute a ϵ-primal-dual stationary point of $P _ { \rho }$ which is nonconvex-concave, we apply the inexact proximal-point method as in Shen et al. (2026). The original nonconvex-concave problem is thereby reduced to a sequence of approximately solved strongly-convex-strongly-concave minimax subproblems, each of which solved using the stochastic accelerated primal-dual (SAPD) algorithm by treating minimizing and maximizing variables as separate blocks (Zhang et al., 2022b).

## 2.6 GENERALIZATION THEORY

We establish a generalization theory for our continuous bilevel hyperparameter tuning framework. For clarity of exposition, we present the results in this section without the encoder $\theta ;$ the full extensions and analysis details are provided in Appendix F. Formally, we analyze the problem of finding the optimal multidimensional hyperparameters $\hat { \psi } \in \Psi \left( \mathrm { e . g . , } \psi = \left( \lambda , \eta \right) \right.$ for Group DRO, or $\boldsymbol { \psi } = ( \lambda , \eta , \epsilon )$ for Bi-HDRO) that minimize the worst-group validation loss:

$$
\hat { \psi } = \arg \operatorname* { m i n } _ { \psi \in \Psi } \operatorname* { m a x } _ { p \in \Delta _ { G } } \sum _ { g = 1 } ^ { G } p _ { g } L _ { g } ^ { \mathrm { v a l } } ( \widehat { W } _ { \psi } ) ,\tag{15}
$$

where the robust model parameters $\widehat { W } _ { \psi }$ are trained via the lower-level robust objective:

$$
\widehat { W } _ { \psi } = \arg \operatorname* { m i n } _ { W } \operatorname* { m a x } _ { q \in \Delta _ { G } } \left[ \sum _ { g = 1 } ^ { G } q _ { g } L _ { g } ^ { \mathrm { t r } } ( W ) - \frac { \eta } { 2 } \left. q - \frac { 1 } { G } \mathbf { 1 } \right. ^ { 2 } + \frac { \lambda } { 2 } \Vert W \Vert ^ { 2 } \right] .\tag{16}
$$

Under standard assumptions on the learning problem (e.g., Lipschitz and bounded convex loss, bounded continuous hyperparameter space, and strongly convex lower-level regularization), we establish that the lower-level optimization algorithm induces a bounded, Lipschitz-continuous hypothesis space with respect to the continuous hyperparameters.

Informally, we prove a Continuous Oracle Inequality for our robust tuning frameworks, demonstrating that tuning continuous multidimensional hyperparameters (such as the robustness penalty η, the $L _ { 2 }$ regularization coefficient λ, and group-specific perturbation radii ϵ) on a validation set allows our algorithm to achieve an optimal bias-variance trade-off without suffering from discretization error or grid-search penalties. Our main technical tools rely on the uniform stability of the lower-level predictor (due to strong convexity) and the Rademacher complexity of the algorithmic hypothesi class (due to algorithmic Lipschitzness) (Shalev-Shwartz & Ben-David, 2014).

Theorem 2.1 (Informal Continuous Oracle Inequality for Bilevel DRO). Let $\hat { \psi }$ be the multidimensional continuous hyperparameters tuned via the upper-level continuous validation process. Let $\widehat { W } _ { \widehat { \psi } }$ be the corresponding robust model trained in the lower level. With high probability over the training and validation sets, the true worst-group risk $\begin{array} { r } { L _ { \mathcal { D } } ^ { w o r s t } ( \widehat { W } _ { \widehat { \psi } } ) : = \operatorname* { m a x } _ { g \in [ G ] } L _ { \mathcal { D } , g } ( \widehat { W } _ { \widehat { \psi } } ) } \end{array}$ is bounded by:

$$
\begin{array} { r l } & { L _ { \mathcal { D } } ^ { w o r s t } ( \widehat { W } _ { \widehat { \psi } } ) \leq \underbrace { L _ { \mathcal { D } } ^ { w o r s t } ( W ^ { * } ) } _ { R e f e r n e c R i s k } + \underbrace { \mathcal { O } \left( \sqrt { \frac { \log G } { \operatorname* { m i n } _ { g } n _ { g } ^ { t r } } } + \sqrt { \frac { \log G } { \operatorname* { m i n } _ { g } n _ { g } ^ { \mathrm { v a l } } } } \right) } _ { S t a t i c i z a l G a p s } } \\ & { + \underbrace { \operatorname* { m i n } _ { \psi } \left( \underbrace { \mathcal { O } ( \lambda + \eta ) } _ { A p p o x i m a t i o n B i c i \psi } + \underbrace { \mathcal { O } \left( \frac { 1 } { \lambda \eta \operatorname* { m i n } _ { g } n _ { g } ^ { \mathrm { t r } } } \right) } _ { S t a b i l i t y G a p ( \psi ) } \right) } _ { C o n t i n t i a g n S \ : I t e r a l p } + \underbrace { \mathcal { O } \left( \sqrt { \frac { k } { \operatorname* { m i n } _ { g } n _ { g } ^ { \mathrm { v a l } } } \log \left( \rho _ { A , \psi } \right) } \right) } _ { C o n t i n t i a g n \ : I e r a l y } } \end{array}
$$

where min<sub>g</sub> $\mathrm { ~ , ~ } n _ { g } ^ { \mathrm { t r } }$ and min ${ } _ { g } n _ { g } ^ { \mathrm { v a l } }$ are the sizes ofthe smallest groups in the training and validation sets respectively, G is the number of groups, k is the dimensionality of the hyperparameter space $( e . g .$ $k = 2$ for standard GDRO, $k = G + 2$ for HDRO), and $\rho _ { \mathcal { A } , \psi }$ is the algorithmic Lipschitz constant of the lower-level optimization. The Approximation Bias scales with the algorithmic regularization penalties λ and $\eta ,$ while the Stability Gap (derived via uniform stability) shrinks as regularization increases,fundamentally capturing the bias-variance trade-offparameterized by ψ.

Remark 2.2 (Algorithmic Lipschitz Constant). For standard Group DRO where $\pmb { \psi } = ( \lambda , \eta )$ , the Lipschitz constant is bounded by $\begin{array} { r } { \rho _ { \mathcal { A } } = \mathcal { O } \big ( \frac { 1 } { \lambda _ { \operatorname* { m i n } } ^ { 2 } } + \frac { 1 } { \lambda _ { \operatorname* { m i n } } \eta _ { \operatorname* { m i n } } ^ { 2 } } \big ) } \end{array}$ (see Corollary F.6), where $\lambda _ { \operatorname* { m i n } } > 0$ and $\eta _ { \mathrm { m i n } } > 0$ are lower bounds of search spaces. When extending to Bi-HDRO with tunable perturbation radii $\boldsymbol { \psi } = ( \lambda , \eta , \epsilon )$ , the mapping remains Lipschitz continuous with the constant expanding by $\begin{array} { r } { \mathcal { O } \big ( \frac { 1 } { \lambda _ { \operatorname* { m i n } } } + \frac { 1 } { \lambda _ { \operatorname* { m i n } } \eta _ { \operatorname* { m i n } } } \big ) } \end{array}$ due to the norm-bounded inner perturbations (see Lemma F.9). Furthermore, as detailed in the appendix, this framework extends to the scenario where the weights of a deep neural network encoder (upper level decision variables) are also treated as tuning parameters. Similar bounds on uniform stability and algorithmic Lipschitz continuity hold in this high-dimensional regime (see Corollaries F.7 and F.12).

This result confirms that continuous bilevel tuning discovers the theoretically optimal configuration for any unknown reference predictor $W ^ { * }$ . Crucially, the statistical penalty for tuning over a continuous space scales logarithmically with the algorithmic Lipschitz constant. This constant dictates an “effective grid s $\mathrm { i z e ^ { , \vec { \mathbf { \mu } } _ { - } } }$ —the finite number of distinguishable models within the search space—allowing us to bypass discretization error while paying a statistical penalty no worse than a discrete grid search. Furthermore, unlike exhaustive grid search which suffers from exponential computational complexity in high dimensions, our scheme enables efficient continuous optimization over multidi mensional hyperparameter spaces (full assumptions and proofs are provided in Appendix F.5).

## 3 RELATED WORKS

GDRO GDRO (Sagawa et al., 2020) aims at minimizing the training loss on the group with least training signals due to the spurious attribute. Empirically, minimizing the worst group training loss per se does not translate to robustness over test data, necessitating a tuned weight decay term (Sagawa et al., 2020). DFR (Kirichenko et al., 2023) and AFR (Qiu et al., 2023) improve on GDRO by retraining the convex classifier (last layer of a deep neural network) using a group-balanced set (Ren et al., 2018), which they show to improve the worst-group robustness even with a ERM-trained model. When both inter-group and intra-group uncertainty exist, HDRO (Jo et al., 2026) perturbs the latent representation with group-dependent radii before performing classification in the last layer. The perturbation radii is fixed and sensitive, so extensive tuning is necessary.

Other methods focus on using a small amount of group-labeled data to achieve similar worst-case oracle performance. SSA (Nam et al., 2022) first trains a hard group predictor before running GDRO on pseudolabeled data, and PG-DRO (Ghosal & Li, 2023) instead use soft group prediction and robust training on soft labels. Notably, AGRO (Paranjape et al., 2023) jointly trains an adversarial soft group prediction model by changing group assignments to increase robust classifier’s groupwise prediction loss. CnC (Zhang et al., 2022a) aligns samples with the same class but different attributes in a two-stage contrastive learning framework. DISC (Wu et al., 2023) partitions data with a “concept bank” which consists of candidate spurious attributes. GIC (Han & Zou, 2024) has three stages and identifies spurious features by comparing the training set with a carefully selected reference dataset. D3M (Jain et al., 2024) removes examples that disproportionately degrade worstgroup accuracy. Remarkably, XRM (Pezeshki et al., 2024) requires no auxiliary datasets whatsoever and instead identifies spurious attributes by training twin classifiers with mutually exclusive training splits and discover attributes using worst-performing samples before running GDRO.

Bilevel Optimization Bilevel optimization methods has been widely applied to hyperparameter tuning (Bennett et al., 2008; Franceschi et al., 2018), meta-learning (Franceschi et al., 2018; Bertinetto et al., 2019; Rajeswaran et al., 2019), reinforcement learning (Hong et al., 2023; Yang et al., 2024; Li et al., 2024a;b), and neural architecture search (Liu et al., 2019).

Among hyperparameter tuning applications, validation set performance is optimized in the upper level to improve model generalizability over the training set (Domke, 2012; Maclaurin et al., 2015; Franceschi et al., 2017; 2018; Shaban et al., 2019; Feurer & Hutter, 2019; Lorraine et al., 2020). In our work, we treat the robustness mechanism, which may contain non-convex neural networks, as hyperparameters to be optimized. Inspired by Lu & Mei (2024) and Lu & Mei (2026) which solved bilevel optimization problem via deterministic minimax optimization, Shen et al. (2026) addressed a more general bilevel optimization problem when both upper and lower level are minimax problems and extended it to a stochastic case. The fact that GDRO itself is a minimax problem naturally makes such bilevel-minimax algorithm good solver candidates.

## 4 EXPERIMENTS

In this section, we describe the modified datasets under both inter-group and intra-group subpopulation shift before showing the performance on two instantiations of our bilevel robust mechanism learning framework for DRO, namely Bi-HDRO and Bi-PG-DRO.

## 4.1 DATASETS WITH TEST DISTRIBUTION SHIFT

We use datasets where both intra-group distribution and inter-group distribution shifts exist. Since baseline methods have reached similar performance under inter-group subpopulation shift, and further tuning offers no improvements, we do not investigate it here. Dataset details in Appendix D.1.

Shifted CMNIST Class label is digit (0-4 or 5-9), and the spurious attribute is color. We rotate minority group samples (red, lebel 1 (digit 5-9)) by 90<sup>◦</sup> in the validation and test sets.

Shifted CelebA Class label is hair color, and spurious attribute is gender. We include only noglasses images in training and validation, and only with-glasses images at test time for minority group (male with blond hair).

Shifted CivilComments Class label is toxicity and spurious attribute is black/white. Since “black” and “white” attributes may be co-mentioned, for the minority toxic/black group, we include no “white” attribute in the training set, whereas in the test set all entries have the “white” attribute.

## 4.2 RESULTS ON BI-HDRO

In this setup, group membership is known at all times, so we use GDRO (Sagawa et al., 2020), DFR<sup>Tr</sup> (Kirichenko et al., 2023) and PDE (Deng et al., 2023) as baselines since they need access to group information at training time. See details in Appendix D.3.

Validity of our enhanced setup As seen in the top half of Table 1, intra-group shift on top of existing inter-group shift indeed decimates the performance of strong baselines, making the doublyshifted datasets valuable benchmarks. We further validated the improvement of HDRO compared to existing baselines and show that employing ambiguity sets in such setup provides tangible benefits across all datasets. For example, HDRO improves from 59.2 of GDRO to 72.4 on CelebA. However, we used 4 ϵ and $5 C$ values (part of the scaling term $C / n _ { g } ^ { \mathrm { t r } }$ in Sagawa et al., 2020, Eq (5)) for HDRO tuning, yielding 16 combinations in grid search.

Table 1: Accuracy over 3 runs under shifted distributions for Bi-HDRO and its baselines
<table><tr><td></td><td colspan="2">CMNIST</td><td colspan="2">CelebA</td><td colspan="2">CivilComments</td></tr><tr><td>Method</td><td>Worst</td><td> $\operatorname { A v g }$ </td><td>Worst</td><td> $\operatorname { A v g }$ </td><td>Worst</td><td>Avg</td></tr><tr><td>GDRO</td><td> $7 1 . 3 { \pm } 1 . 3 $ </td><td> $7 2 . 7 { \scriptstyle \pm 1 . 6 }$ </td><td> $5 9 . 2 { \pm } 0 . 1$ </td><td> $9 2 . 7 \pm 0 . 1 $ </td><td> $3 4 . 8 { \scriptstyle \pm 6 . 0 }$ </td><td> $8 7 . 8 { \pm } 1 . 4 $ </td></tr><tr><td> $\mathbf { D F R } ^ { \mathrm { T r } }$ </td><td> $6 2 . 9 { \scriptstyle \pm 8 . 3 }$ </td><td> $6 8 . 9 { \scriptstyle \pm 4 . 3 }$ </td><td> $6 5 . 5 { \scriptstyle \pm 4 . 8 }$ </td><td> $8 9 . 4 \pm 0 . 3$ </td><td> $4 0 . 8 { \scriptstyle \pm 2 . 7 }$ </td><td> $8 7 . 9 2 0 . 9 $ </td></tr><tr><td>PDE</td><td> $6 2 . 8 { \pm } 7 . 8$ </td><td> $6 9 . 1 \pm 3 . 8$ </td><td> $3 5 . 9 2 5 . 4$ </td><td> $9 2 . 0 { \scriptstyle \pm 0 . 6 }$ </td><td> $3 9 . 0 { \scriptstyle \pm 3 . 9 }$ </td><td> $8 1 . 8 { \scriptstyle \pm 0 . 8 }$ </td></tr><tr><td>HDRO</td><td> $7 2 . 7 \pm 0 . 2$ </td><td> $7 6 . 3 { \scriptstyle \pm 3 . 1 }$ </td><td> $7 2 . 4 \pm 3 . 0$ </td><td> $9 1 . 4 { \pm } 0 . 2 $ </td><td> $4 0 . 8 \pm 3 . 1$ </td><td> $8 8 . 0 { \pm } 1 . 8 $ </td></tr><tr><td>Fixed €</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $-  \mathrm { B i - G D R O } ( \epsilon = 0 )$ </td><td> $7 3 . 9 { \scriptstyle \pm 0 . 9 }$ </td><td> $7 4 . 8 { \scriptstyle \pm 0 . 8 }$ </td><td> $7 7 . 0 { \scriptstyle \pm 4 . 3 }$ </td><td> $9 0 . 1 \pm 0 . 5$ </td><td> $5 7 . 9 { \scriptstyle \pm 5 . 1 }$ </td><td> $7 9 . 1 \pm 1 . 3$ </td></tr><tr><td>– Grid search €</td><td> $7 2 . 3 { \pm } 2 . 1$ </td><td> $7 3 . 9 { \pm } 1 . 8$ </td><td> $8 2 . 5 { \scriptstyle \pm 2 . 0 }$ </td><td> $9 0 . 7 \pm 1 . 3$ </td><td> $5 4 . 7 \pm 2 . 3$ </td><td> $8 0 . 0 { \scriptstyle \pm 1 . 2 }$ </td></tr><tr><td>– Grid search η and €</td><td> $7 3 . 7 \pm 0 . 7$ </td><td> $7 5 . 0 { \pm } 1 . 2$ </td><td> $8 2 . 1 \pm 2 . 5$ </td><td> $9 0 . 4 \pm 0 . 8 $ </td><td> $5 7 . 9 { \scriptstyle \pm 6 . 5 }$ </td><td> $7 8 . 0 \pm 3 . 4$ </td></tr><tr><td>Bi-HDRO (learnable €)</td><td> $7 4 . 0 { \scriptstyle \pm 0 . 3 }$ </td><td> $7 6 . 4 \pm 0 . 4$ </td><td> $8 1 . 8 { \scriptstyle \pm 3 . 5 }$ </td><td> $8 9 . 2 { \pm } 1 . 0$ </td><td> ${ \bf 5 9 . 5 } \pm 3 . 5$ </td><td> $7 8 . 5 { \scriptstyle \pm 2 . 6 }$ </td></tr><tr><td>Bi-HDRO (learnable  $\epsilon _ { g } )$ </td><td> ${ \bf 7 4 . 2 \pm 0 . 3 }$ </td><td> $7 6 . 9 2 1 . 6 $ </td><td> $\mathbf { 8 2 . 8 } \pm \mathbf { 3 . 1 }$ </td><td> $9 0 . 1 \pm 1 . 2 $ </td><td> $5 7 . 4 { \pm 2 . 0 }$ </td><td> $7 8 . 5 { \scriptstyle \pm 0 . 6 }$ </td></tr></table>

Benefit of parameter tuning In the bottom half of Table 1, we show that (1) Bi-HDRO performs better than HDRO (lower level) alone; and (2) ablating individual $\epsilon _ { g }$ and η as manually tuned and fixed components shows the superior performance of Bi-HDRO.

## 4.3 RESULTS ON BI-PG-DRO

In this setting, group labels are not available at training time, so we uniformly sample a small fraction of group-labeled validation data (5% from CMNIST, 15% from CelebA, and 3% from CivilComments) and create two splits, where one is used to tune the attribute prediction model and the other is used to tune the validation set. For comparison fairness among baseline methods, we made sure attribute prediction training sees the same split, and the other split is for manual model tuning. We use ERM, $\mathbf { \dot { G I C } } ^ { C _ { y } }$ (Han & Zou, 2024), XRM (Pezeshki et al., 2024), AGRO (Paranjape et al., 2023), and SSA (Nam et al., 2022) as our baselines since they need minimal or no group-labeled data at training time. See details in Appendix D.4.

Group-labeled validation data boost our performance With validation tuning, fixed $\eta ,$ and no attribute prediction regularization, our method already performs better than all baseline methods.

Implicitly tuning auxiliary model yields superior performance Compared to explicitly using a cross-entropy loss to train the attribute predictor, implicitly tuning the attribute predictor by minimizing KL divergence works just as well. Finally, we show that the complete bilevel method (Bi-PG-DRO) tuning η works the best when combined with $\mathrm { K L , }$ outperforming all baseline methods.

## 5 CONCLUSION

We proposed a DRO framework to learn the robustness mechanism that requires minimal tuning and provided two instantiantions. There are two future directions from this work. First, our bilevel framework is still somewhat restrictive in that the algorithm proposed by Shen et al. (2026) requires lower level training to be convex-concave. By assuming Kurdyka-Łojasiewicz (KL) condition so that lower level becomes a non-convex-concave problem, a more general class of applications ensues, but no such algorithm exists yet. Second, this class of algorithms may be further extended to machine unlearning and language model alignment tasks (Fan et al., 2025; Wu et al., 2025; Asif & Amiri, 2026).

Table 2: Accuracy over 3 runs under shifted subpopulations for Bi-PG-DRO and its baselines
<table><tr><td></td><td colspan="2">CMNIST</td><td colspan="2">CelebA</td><td colspan="2">CivilComments</td></tr><tr><td>Method</td><td>Worst</td><td>Avg</td><td>Worst</td><td>Avg</td><td>Worst</td><td>Avg</td></tr><tr><td>ERM</td><td>1.6±1.7</td><td>15.7±6.1</td><td>25.0±2.6</td><td>95.4±0.1</td><td>36.8±4.6</td><td>91.3±0.8</td></tr><tr><td>GICC{</td><td>25.7±8.8</td><td>48.0±13.4</td><td>47.1±9.2</td><td>90.8±0.8</td><td>54.2±4.9</td><td>86.8±1.4</td></tr><tr><td>XRM</td><td>68.8±3.8</td><td>72.2±3.2</td><td>51.4±2.8</td><td>89.7±0.2</td><td>25.9±6.4</td><td>89.6±1.5</td></tr><tr><td>AGRO</td><td>22.8±4.9</td><td>32.4±2.2</td><td>22.1±1.8</td><td>95.4±0.2</td><td>24.5±3.5</td><td>91.4±0.4</td></tr><tr><td>SSA</td><td>70.0±2.9</td><td>72.3±1.7</td><td>58.6±2.3</td><td>89.7±0.1</td><td>27.4±8.6</td><td>87.1±1.9</td></tr><tr><td>PG-DRO</td><td>64.3±2.0</td><td>70.7±0.7</td><td>66.7±1.0</td><td>91.6±0.4</td><td>16.1±3.1</td><td>82.3±3.0</td></tr><tr><td>Fixed η = 1.0</td><td>67.8±1.5</td><td>74.1±1.6</td><td>69.8±6.9</td><td>92.6±0.2</td><td>66.2±1.1</td><td>84.3±1.1</td></tr><tr><td>+ BCE</td><td>69.5±1.4</td><td>75.3±4.0</td><td>73.3±5.4</td><td>91.8±2.2</td><td>65.0±1.3</td><td>79.3±1.4</td></tr><tr><td>+ KL</td><td>71.0±0.8</td><td>80.1±0.9</td><td>73.3±4.8</td><td>92.5±0.3</td><td>68.0±0.3</td><td>84.7±1.1</td></tr><tr><td>Bi-PG-DRO</td><td>71.7±0.8</td><td>79.5±1.5</td><td>76.7±3.4</td><td>92.2±0.3</td><td>68.2±0.2</td><td>83.6±1.9</td></tr></table>

## AI USE STATEMENT

In this work, we used generative AI tools to implement methods, clean and reformat dataset, and assist in the writing of proofs.

We have not used generative AI tools to help develop theoretical models or conceptual frameworks, formulate mathematical claims, provide critical ingredients for proving mathematical claims, propose or refine hypotheses, design or provide feedback on research methodology or experiments, assist with translation, support qualitative and thematic data analysis, or interpret results.

## Generating synthetic datasets is not applicable to this work.

We have reviewed all AI-assisted work. LLM-generated code was verified and tested for correctness. LLM-generated proof steps are judiciously reviewed and revised by all authors. We take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## REFERENCES

Faruk Ahmed, Yoshua Bengio, Harm Van Seijen, and Aaron Courville. Systematic generalisation with group invariant predictions. In International Conference on Learning Representations, 2021.

M. Arjovsky, L. Bottou, I. Gulrajani, and D. Lopez-Paz. Invariant risk minimization. arXiv preprint arXiv:1907.02893, 2019.

Saeid Asgari, Aliasghar Khani, Fereshte Khani, Ali Gholami, Linh Tran, Ali Mahdavi Amiri, and Ghassan Hamarneh. Masktune: Mitigating spurious correlations by forcing to explore. In Advances in Neural Information Processing Systems, 2022.

Sadia Asif and Mohammad Mohammadi Amiri. OFMU: OPTIMIZATION-DRIVEN FRAME-WORK FOR MACHINE UNLEARNING. In International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=ZDuyNJI56H.

Kasra Azizi, Kumar Anurag, and Wenbin Wan. Towards resilient tracking in autonomous vehicles: A distributionally robust input and state estimation approach. IFAC-PapersOnLine, 59(3):31–36, 2025. ISSN 2405-8963. doi: https://doi.org/10.1016/j.ifacol.2025.07.006. 12th IFAC Symposium on Intelligent Autonomous Vehicles IAV 2025.

Marcus A Badgeley, John R Zech, Luke Oakden-Rayner, Benjamin S Glicksberg, Manway Liu, William Gale, Michael V McConnell, Bethany Percha, Thomas M Snyder, and Joel T Dudley. Deep learning predicts hip fracture using confounding patient and healthcare variables. NPJ Digital Medicine, 2(1):31, 2019.

Aharon Ben-Tal and Arkadi Nemirovski. Robust optimization-methodology and applications. Mathematical Programming, 92:453–480, 2002.

Kristin P. Bennett, Gautam Kunapuli, Jing Hu, and Jong-Shi Pang. Bilevel optimization and machine learning. In IEEE World Congress on Computational Intelligence, pp. 25–47. Springer, 2008.

Luca Bertinetto, Joao F. Henriques, Philip Torr, and Andrea Vedaldi. Meta-learning with differentiable closed-form solvers. In International Conference on Learning Representations, 2019.

Olivier Bousquet and Andre Elisseeff. Stability and generalization.´ The Journal ofMachine Learning Research, 2:499–526, 2002.

Joy Buolamwini and Timnit Gebru. Gender shades: Intersectional accuracy disparities in commercial gender classification. In Proceedings of the Conference on Fairness, Accountability and Transparency, pp. 77–91, 2018.

Tianle Cai, Ruiqi Gao, Jason Lee, and Qi Lei. A theory of label propagation for subpopulation shift. In Proceedings of the International Conference on Machine Learning, pp. 1170–1182, 2021.

Koby Crammer and Yoram Singer. On the algorithmic implementation of multiclass kernel-based vector machines. In Journal ofmachine learning research, 2001.

Elliot Creager, Jorn-Henrik Jacobsen, and Richard Zemel. Environment inference for invariant¨ learning. In Proceedings of the International Conference on Machine Learning, pp. 2189–2200, 2021.

John M Danskin. The theory of max-min and its application to weapons allocation problems. Econometrics and Operations Research, 5, 1967.

Yihe Deng, Yu Yang, Baharan Mirzasoleiman, and Quanquan Gu. Robust learning with progressive data expansion against spurious correlation. In Advances in Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=9QEVJ9qm46.

Luc Devroye, Laszl ´ o Gy ´ orfi, and G ¨ abor Lugosi. ´ A Probabilistic Theory of Pattern Recognition. Springer Science & Business Media, 2013.

Justin Domke. Generic methods for optimization-based modeling. In Proceedings of the International Conference on Artificial Intelligence and Statistics, 2012.

J. C. Duchi, P. W. Glynn, and H. Namkoong. Statistics of robust optimization: A generalized empirical likelihood approach. Mathematics ofOperations Research, 2021.

John C Duchi and Hongseok Namkoong. Learning models with uniform performance via distributionally robust optimization. The Annals of Statistics, 49(3):1378–1406, 2021.

Chongyu Fan, Jinghan Jia, Yihua Zhang, Anil Ramakrishna, Mingyi Hong, and Sijia Liu. Towards LLM unlearning resilient to relearning attacks: A sharpness-aware minimization perspective and beyond. In Proceedings of the International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=zZjLv6F0Ks.

Matthias Feurer and Frank Hutter. Hyperparameter optimization. In Automated Machine Learning, pp. 3–33. Springer, 2019.

Luca Franceschi, Michele Donini, Paolo Frasconi, and Massimiliano Pontil. Forward and reverse gradient-based hyperparameter optimization. In Proceedings of the International Conference on Machine Learning, pp. 1165–1173, 2017.

Luca Franceschi, Paolo Frasconi, Saverio Salzo, Riccardo Grazzi, and Massimiliano Pontil. Bilevel programming for hyperparameter optimization and meta-learning. In Proceedings of the International Conference on Machine Learning, pp. 1568–1577, 2018.

Soumya Suvra Ghosal and Yixuan Li. Distributionally robust optimization with probabilistic group. In Proceedings ofthe AAAI Conference on Artificial Intelligence, pp. 11809–11817, 2023.

Yujin Han and Difan Zou. Improving group robustness on spurious correlation requires preciser group inference. In Proceedings of the International Conference on Machine Learning, pp. 17480–17504, 2024. URL https://openreview.net/forum?id=KycvgOCBBR.

Mingyi Hong, Hoi-To Wai, Zhaoran Wang, and Zhuoran Yang. A two-timescale stochastic algorithm framework for bilevel optimization. SIAM Journal on Optimization, 33(1):147–180, 2023.

Feihu Huang, Xidong Wu, and Heng Huang. Efficient mirror descent ascent methods for nonsmooth minimax problems. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan (eds.), Advances in Neural Information Processing Systems, volume 34, pp. 10431– 10443. Curran Associates, Inc., 2021.

Pavel Izmailov, Polina Kirichenko, Nate Gruver, and Andrew G Wilson. On feature learning in the presence of spurious correlations. In Advances in Neural Information Processing Systems, 2022.

Arthur Jacot, Franck Gabriel, and Clement Hongler. Neural tangent kernel: Convergence and gener-´ alization in neural networks. In Advances in neural information processing systems, volume 31, 2018.

Saachi Jain, Kimia Hamidieh, Kristian Georgiev, Andrew Ilyas, Marzyeh Ghassemi, and Aleksander Madry. Improving subgroup robustness via data selection. In Advances in Neural Information Processing Systems, 2024.

Jinyong Jeong, Hyungu Kahng, and Seoung Bum Kim. Multi-expert distributionally robust optimization for out-of-distribution generalization. In Advances in Neural Information Processing Systems, volume 38, Main Conference, pp. 101081–101116, 2025. doi: 10.52202/085713-3384.

Sung Ho Jo, Seonghwi Kim, and Minwoo Chae. Mitigating spurious correlation via distributionally robust learning with hierarchical ambiguity sets. In International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=mruL1LDjzV.

Bingyi Kang, Saining Xie, Marcus Rohrbach, Zhicheng Yan, Albert Gordo, Jiashi Feng, and Yannis Kalantidis. Decoupling representation and classifier for long-tailed recognition. In International Conference on Learning Representations, 2020.

Polina Kirichenko, Pavel Izmailov, and Andrew Gordon Wilson. Last layer re-training is sufficient for robustness to spurious correlations. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=Zb6c8A-Fghk.

Pang Wei Koh, Shiori Sagawa, Henrik Marklund, Sang Michael Xie, Marvin Zhang, Akshay Balsubramani, Weihua Hu, Michihiro Yasunaga, Richard Lanas Phillips, Irena Gao, Tony Lee, Etienne David, Ian Stavness, Wei Guo, Berton Earnshaw, Imran Haque, Sara M Beery, Jure Leskovec, Anshul Kundaje, Emma Pierson, Sergey Levine, Chelsea Finn, and Percy Liang. Wilds: A benchmark of in-the-wild distribution shifts. In Marina Meila and Tong Zhang (eds.), Proceedings of the International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pp. 5637–5664. PMLR, 18–24 Jul 2021.

David Krueger, Ethan Caballero, Joern-Henrik Jacobsen, Amy Zhang, Jonathan Binas, Dinghuai Zhang, Remi Le Priol, and Aaron Courville. Out-of-distribution generalization via risk extrapolation (rex). In Proceedings ofthe International Conference on Machine Learning, pp. 5815–5826, 2021.

Tyler LaBonte, Vidya Muthukumar, and Abhishek Kumar. Towards last-layer retraining for group robustness with fewer annotations. In Advances in Neural Information Processing Systems, 2023.

Haoyu Lei, Amin Gohari, and Farzan Farnia. On the inductive biases of demographic parity-based fair learning algorithms. In Negar Kiyavash and Joris M. Mooij (eds.), Uncertainty in Artificial Intelligence, volume 244 of Proceedings ofMachine Learning Research, pp. 2205–2225. PMLR, 15–19 Jul 2024.

Chenliang Li, Siliang Zeng, Zeyi Liao, Jiaxiang Li, Dongyeop Kang, Alfredo Garcia, and Mingyi Hong. Learning reward and policy jointly from demonstration and preference improves alignment. arXiv preprint arXiv:2406.06874, 2024a.

Jiaxiang Li, Siliang Zeng, Hoi To Wai, Chenliang Li, Alfredo Garcia, and Mingyi Hong. Getting more juice out of the SFT data: Reward learning from human demonstration improves SFT for LLM alignment. In Advances in Neural Information Processing Systems, 2024b.

Hanxiao Liu, Karen Simonyan, and Yiming Yang. Darts: Differentiable architecture search. In International Conference on Learning Representations, 2019.

Jonathan Lorraine, Paul Vicol, and David Duvenaud. Optimizing millions of hyperparameters by implicit differentiation. In Proceedings of the International Conference on Artificial Intelligence and Statistics, 2020.

Zhaosong Lu and Sanyou Mei. First-order penalty methods for bilevel optimization. SIAM Journal on Optimization, 34(2):1937–1969, 2024. doi: 10.1137/23M1566753. URL https://doi. org/10.1137/23M1566753.

Zhaosong Lu and Sanyou Mei. Solving bilevel optimization via sequential minimax optimization. Mathematics of Operations Research, 2026. doi: 10.1287/moor.2024.0521. URL https:// doi.org/10.1287/moor.2024.0521. Published online January 6, 2026.

Dougal Maclaurin, David Duvenaud, and Ryan Adams. Gradient-based hyperparameter optimization through reversible learning. In Proceedings of the International Conference on Machine Learning, 2015.

Raghav Mehta, Changjian Shui, and Tal Arbel. Evaluating the fairness of deep learning uncertainty estimates in medical image analysis. In Medical Imaging with Deep Learning, volume 227 of Proceedings ofMachine Learning Research, pp. 1453–1492. PMLR, 10–12 Jul 2024.

Junhyun Nam, Jaehyung Kim, Jaeho Lee, and Jinwoo Shin. Spread spurious attribute: Improving worst-group accuracy with spurious attribute estimation. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=\_F9xpOrqyX9.

Bhargavi Paranjape, Pradeep Dasigi, Vivek Srikumar, Luke Zettlemoyer, and Hannaneh Hajishirzi. AGRO: Adversarial discovery of error-prone groups for robust optimization. In International Conference on Learning Representations, 2023.

Mohammad Pezeshki, Diane Bouchacourt, Mark Ibrahim, Nicolas Ballas, Pascal Vincent, and David Lopez-Paz. Discovering environments with XRM. In Proceedings of the International Conference on Machine Learning, 2024. URL https://openreview.net/forum?id= gPStP3FSY9.

Vihari Piratla, Praneeth Netrapalli, and Sunita Sarawagi. Focus on the common good: Group distributional robustness follows. In International Conference on Learning Representations, 2022.

Shikai Qiu, Andres Potapczynski, Pavel Izmailov, and Andrew Gordon Wilson. Simple and fast group robustness by automatic feature reweighting. In Proceedings of the International Conference on Machine Learning, pp. 28448–28467, 2023. URL https://openreview.net/ forum?id=s5F1a6s1HS.

Aravind Rajeswaran, Chelsea Finn, Sham M Kakade, and Sergey Levine. Meta-learning with implicit gradients. In Advances in Neural Information Processing Systems, volume 32, 2019.

Mengye Ren, Wenyuan Zeng, Bin Yang, and Raquel Urtasun. Learning to reweight examples for robust deep learning. In Proceedings of the International Conference on Machine Learning, pp. 4334–4343. PMLR, 2018.

Kanghyun Ryu and Negar Mehr. Integrating predictive motion uncertainties with distributionally robust risk-aware control for safe robot navigation in crowds. In IEEE International Conference on Robotics and Automation (ICRA), pp. 2410–2417, 2024.

Shiori Sagawa, Pang Wei Koh, Tatsunori B. Hashimoto, and Percy Liang. Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generaliza tion. In International Conference on Learning Representations, 2020.

Seonguk Seo, Joon-Young Lee, and Bohyung Han. Unsupervised learning of debiased representations with pseudo-attributes. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 16742–16751, 2022.

Amir Shaban, Ching-An Cheng, Nathan Hatch, and Byron Boots. Truncated back-propagation for bilevel optimization. In Proceedings of the International Conference on Artificial Intelligence and Statistics, 2019.

Shai Shalev-Shwartz and Shai Ben-David. Understanding Machine Learning: From Theory to Algorithms. Cambridge university press, 2014.

Yiyang Shen, Yutian He, Weiran Wang, and Qihang Lin. Penalty-based first-order methods for bilevel optimization with minimax and constrained lower-level problems. arXiv preprint arXiv:2605.08006, 2026.

Zheyan Shen, Jiashuo Liu, Yue He, Xingxuan Zhang, Renzhe Xu, Han Yu, and Peng Cui. Towards out-of-distribution generalization: A survey. arXiv preprint arXiv:2108.13624, 2021.

Nimit Sohoni, Jared Dunnmon, Geoffrey Angus, Albert Gu, and Christopher Re. No subclass left´ behind: Fine-grained robustness in coarse-grained classification problems. In Advances in Neural Information Processing Systems, 2020.

Junkang Wu, Yuexiang Xie, Zhengyi Yang, Jiancan Wu, Jiawei Chen, Jinyang Gao, Bolin Ding, Xiang Wang, and Xiangnan He. Towards robust alignment of language models: Distributionally robustifying direct preference optimization. In International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=CbfsKHiWEn.

Shirley Wu, Mert Yuksekgonul, Linjun Zhang, and James Zou. Discover and cure: Concept-aware mitigation of spurious correlation. In Proceedings of the International Conference on Machine Learning, pp. 37765–37786, 2023.

Yan Yang, Bin Gao, and Ya-xiang Yuan. Bilevel reinforcement learning via the development of hyper-gradient without lower-level convexity. arXiv preprint arXiv:2405.19697, 2024.

Yuzhe Yang, Haoran Zhang, Dina Katabi, and Marzyeh Ghassemi. Change is hard: A closer look at subpopulation shift. In Proceedings of the International Conference on Machine Learning, pp. 39584–39622, 2023.

Han Yu, Jiashuo Liu, Xingxuan Zhang, Jiayun Wu, and Peng Cui. A survey on evaluation of out-ofdistribution generalization. ArXiv, 2024.

John R Zech, Marcus A Badgeley, Manway Liu, Anthony B Costa, Joseph J Titano, and Eric Karl Oermann. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study. PLoS Medicine, 15(11):e1002683, 2018.

Michael Zhang, Nimit S Sohoni, Hongyang R Zhang, Chelsea Finn, and Christopher Re. Correct-N-Contrast: A contrastive approach for improving robustness to spurious correlations. In Proceedings ofthe International Conference on Machine Learning, pp. 26484–26516, 2022a.

Xuan Zhang, Necdet Serhat Aybat, and Mert Gurbuzbalaban. Sapd+: An accelerated stochastic method for nonconvex-concave minimax problems. In Advances in Neural Information Processing Systems, volume 35, pp. 21668–21681, 2022b.

Yang Zhang, Philip David, and Boqing Gong. Curriculum domain adaptation for semantic segmentation of urban scenes. In Proceedings ofthe IEEE International Conference on Computer Vision, pp. 2020–2030, 2017.

## A A FIRST-ORDER METHOD FOR BILEVEL MINIMAX PROBLEMS

Algorithm 1 A First-Order Method for (13)   
Require: Initial iterates $( \alpha ^ { 0 } , W ^ { 0 } , q ^ { 0 } )$ and $( p ^ { 0 } , \widetilde { W } ^ { 0 } , \widetilde { q } ^ { 0 } )$ and number of proximal iterations K.   
1: Set $\rho , \rho _ { 1 } , \rho _ { 2 }$ and relevant parameters according to Shen et al., 2026, Algorithm 1.   
2: for $k = 0 , \ldots , K - 1$ do   
3: Construct the proximal penalized objective   
$\mathcal { \bar { P } } _ { k } = P _ { \rho } ( \alpha , p , W , q , \widetilde { W } , \widetilde { q } ) + \frac { \rho _ { 1 } } { 2 } \| ( \alpha , W , q ) - ( \alpha ^ { k } , W ^ { k } , q ^ { k } ) \| ^ { 2 } - \frac { \rho _ { 2 } } { 2 } \| ( p , \widetilde { W } , \widetilde { q } ) - ( p ^ { k } , \widetilde { W } ^ { k } , \widetilde { q } ^ { k } ) \| ^ { 2 }$   
4: Solve the resulting strongly-convex-strongly-concave problem with SAPD (Zhang et al.,   
2022b):   
$( ( \alpha ^ { k + 1 } , W ^ { k + 1 } , q ^ { k + 1 } ) , ( p ^ { k + 1 } , \widetilde W ^ { k + 1 } , \tilde { q } ^ { k + 1 } ) ) \gets \mathrm { S A P D } ( \bar { \mathcal { P } } _ { k } )$   
5: end for   
6: return $( \alpha ^ { k ^ { \prime } } , W ^ { k ^ { \prime } } , q ^ { k ^ { \prime } } )$ with k<sup>′</sup> sampled uniformly from $\{ 1 , \ldots , K \}$

While the original work has many hyperparameters on the algorithmic level, we stress that those are tuned once and can then be applied to all datasets used in this paper and both of our proposed methods (See Appendix D.2 for details). We further note that validation accuracy is used for model selection rather than objective convergence, which is prohibitively expensive in deep learning experiments.

## B CLOSED FORM OF LOWER LEVEL ADVERSARY

We can use the definition of training loss in Section 2.1,

$$
\begin{array} { r } { L ^ { \mathrm { t r } } ( W ; \theta ) = \left( L _ { 1 } ^ { \mathrm { t r } } ( W ; \theta ) , \dots , L _ { G } ^ { \mathrm { t r } } ( W ; \theta ) \right) ^ { \top } , } \end{array}
$$

to obtain the adversarial subproblem,

$$
q ^ { \star } \in \arg \operatorname* { m a x } _ { q \in \Delta _ { G } } \left\{ q ^ { \top } L ^ { \mathrm { t r } } ( W ; \theta ) - \frac { \eta } { 2 } \left\| q - \frac { 1 } { G } { \bf 1 } \right\| _ { 2 } ^ { 2 } \right\} .
$$

For $\eta > 0 .$ we can complete the square:

$$
q ^ { \top } L ^ { \mathrm { t r } } - \frac { \eta } { 2 } \left. q - \frac { 1 } { G } { \bf 1 } \right. _ { 2 } ^ { 2 } = - \frac { \eta } { 2 } \left. q - \left( \frac { 1 } { G } { \bf 1 } + \frac { L ^ { \mathrm { t r } } } { \eta } \right) \right. _ { 2 } ^ { 2 } + C ,
$$

where C does not depend on q. Therefore,

$$
q ^ { \star } ( W , \theta , \eta ) = \Pi _ { \Delta _ { G } } \left( \frac { 1 } { G } \mathbf { 1 } + \frac { L ^ { \mathrm { t r } } ( W ; \theta ) } { \eta } \right) ,
$$

where component-wise $\begin{array} { r } { q _ { g } ^ { \star } = \left[ \frac { 1 } { G } + \frac { L _ { g } ^ { \mathrm { t r } } ( W ; \theta ) } { \eta } - \nu \right] _ { + } } \end{array}$ , and ν is chosen such that $\textstyle \sum _ { g = 1 } ^ { G } q _ { g } ^ { \star } = 1$

While there is a closed form solution, computing it requires the losses of all groups and hence a full pass of the entire dataset, which is prohibitively expensive during stochastic training.

## C PERTURBATION FORMULATIONS

## C.1 CLOSED-FORM PERTURBATION FOR BINARY CLASSIFICATION

We show below that closed form can be derived for latent representation perturbation when the task is binary classification. We use the binary cross entropy (BCE) loss for all experiments but show the formulation for hinge loss as well.

Case 1: the last layer is linear and L is hinge loss. Let the final layer be binary linear classification: $\begin{array} { r } { f _ { \theta } ^ { L } ( z ) = w ^ { \top } \dot { z } + b . } \end{array}$ . The hinge loss is

$$
\begin{array} { r } { L ( f _ { \theta } ^ { L } ( z ) , y ) = \operatorname* { m a x } ( 0 , 1 - y ( w ^ { \top } z + b ) ) . } \end{array}
$$

The inner maximization in equation becomes

$$
\operatorname* { s u p } _ { \| \boldsymbol { z } ^ { \prime } - \boldsymbol { z } \| \le \epsilon } \operatorname* { m a x } \bigl ( 0 , 1 - y ( w ^ { \top } \boldsymbol { z } ^ { \prime } + b ) \bigr ) .
$$

Write $z ^ { \prime } = z + \delta , \| \delta \| \leq \epsilon$ , then

$$
\operatorname* { s u p } _ { \| \delta \| \leq \epsilon } \operatorname* { m a x } \big ( 0 , 1 - y ( w ^ { \top } z + b ) - y w ^ { \top } \delta \big ) .
$$

Since $x \mapsto \operatorname* { m a x } ( 0 , x )$ is monotonically increasing,

$$
\mathbf { \Sigma } = \operatorname* { m a x } \left( 0 , 1 - y ( w ^ { \top } z + b ) + \operatorname* { s u p } _ { \| \delta \| \leq \epsilon } ( - y w ^ { \top } \delta ) \right) .
$$

Because of $y \in \{ \pm 1 \}$ and the support function of the norm ball,

$$
\operatorname* { s u p } _ { \| \boldsymbol { \delta } \| \leq \epsilon } ( - \boldsymbol { y } \boldsymbol { w } ^ { \top } \boldsymbol { \delta } ) = \operatorname* { s u p } _ { \| \boldsymbol { \delta } \| \leq \epsilon } \boldsymbol { w } ^ { \top } \boldsymbol { \delta } = \boldsymbol { w } ^ { \top } \boldsymbol { \delta } ^ { * } = \boldsymbol { w } ^ { \top } ( \epsilon \frac { \boldsymbol { w } } { \| \boldsymbol { w } \| ^ { 2 } } ) = \epsilon \| \boldsymbol { w } \| _ { * } ,
$$

where $\| \cdot \|$ <sub>∗</sub> is the dual norm. Therefore, the inner robust hinge loss has closed form:

$$
\operatorname* { s u p } _ { \| \boldsymbol { z } ^ { \prime } - \boldsymbol { z } \| \leq \epsilon } L ( f _ { \theta } ^ { L } ( \boldsymbol { z } ^ { \prime } ) , \boldsymbol { y } ) = \operatorname* { m a x } \big ( 0 , 1 - \boldsymbol { y } ( \boldsymbol { w } ^ { \top } \boldsymbol { z } + b ) + \epsilon \| \boldsymbol { w } \| _ { * } \big )
$$

Case 2: the last layer is linear and L is logistic/binary cross entropy (BCE) loss. Using the same notations as above, the logistic loss is

$$
\begin{array} { r } { L ( f _ { \theta } ^ { L } ( z ) , y ) = \log \bigl ( 1 + \exp ( - y ( w ^ { \top } z + b ) ) \bigr ) . } \end{array}
$$

Due to the monotonicity of the exponent of the maximized perturbation, we have

$$
\operatorname* { s u p } _ { \| \boldsymbol { \delta } \| \leq \epsilon } \log \big ( 1 + \exp \left( - y ( w ^ { \top } ( \boldsymbol { z } + \boldsymbol { \delta } ) + \boldsymbol { b } ) \right) \big ) \Leftrightarrow \operatorname* { i n f } _ { \| \boldsymbol { \delta } \| \leq \epsilon } y ( w ^ { \top } ( \boldsymbol { z } + \boldsymbol { \delta } ) + \boldsymbol { b } ) = y ( w ^ { \top } \boldsymbol { z } + \boldsymbol { b } ) + \operatorname* { i n f } _ { \| \boldsymbol { \delta } \| \leq \epsilon } y w ^ { \top } \boldsymbol { \delta } .
$$

As derived in Case 1, $\begin{array} { r } { \operatorname* { i n f } _ { \| \delta \| \leq \epsilon } y w ^ { \top } \delta = - \epsilon \| w \| _ { * } } \end{array}$ , so we can substitute this back to the maximized perturbation, forming

$$
\operatorname* { s u p } _ { \| \delta \| \leq \epsilon } \log \big ( 1 + \exp ( - y ( w ^ { \top } ( z + \delta ) + b ) ) \big ) = \log \big ( 1 + \exp \big ( - ( y ( w ^ { \top } z + b ) - \epsilon \| w \| _ { * } ) \big ) \big ) .
$$

∥w∥ makes the both losses non-smooth, so we smoothen it using a substitution scaler u. In the logistic example, it becomes

$$
J ( w , u ) = \sum _ { i = 1 } ^ { n } \log \big ( 1 + \exp ( - y _ { i } ( w ^ { \top } x _ { i } + b ) + \epsilon u ) \big ) \quad \mathrm { s . t . } \quad \| w \| _ { * } \leq u ,\tag{17}
$$

which is jointly convex in $( w , u )$

As a result, we perform projected descent when optimizing Eq (17). Define the second-order cone $K = \{ ( w , u ) : \| \bar { w } \| _ { 2 } \leq \bar { u } \}$ , the Euclidean projection is then

$$
\operatorname* { m i n } _ { ( w , u ) \in \mathcal { K } } \frac { 1 } { 2 } \| w - v \| _ { 2 } ^ { 2 } + \frac { 1 } { 2 } ( u - t ) ^ { 2 } ,\tag{18}
$$

the closed form of which falls into three categories, where $( v , t )$ falls inside, outside and “behind”, and outside but near the boundary of the cone, i.e.,

$$
\Pi _ { K } ( v , t ) = \left\{ \begin{array} { l l } { ( v , t ) , } & { \| v \| _ { 2 } \leq t } \\ { ( 0 , 0 ) , } & { \| v \| _ { 2 } \leq - t } \\ { \left( \frac { \| v \| _ { 2 } + t } { 2 \| v \| _ { 2 } } v , \frac { \| v \| _ { 2 } + t } { 2 } \right) , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.
$$

## C.2 PERTURBATION FOR MULTI-CLASS CLASSIFICATION

Next, we show that latent representation perturbation can still be done, although in a relaxed form, when the task is multi-class.

Let the final layer be linear with K classes:

$$
w _ { k } ( z ) = \mathbf { w } _ { k } ^ { \top } z + b _ { k } , \qquad k = 1 , \ldots , K ,
$$

and let the true label be $y \in \{ 1 , \ldots , K \}$

Case 1: multi-class hinge loss. We use the multi-class hinge loss (Crammer & Singer, 2001):

$$
L ( w ( z ) , y ) = \operatorname* { m a x } \left\{ 0 , \operatorname* { m a x } _ { j \neq y } \left[ 1 + ( w _ { j } ( z ) + b _ { j } ) - ( w _ { y } ( z ) + b _ { y } ) \right] \right\} ,
$$

where $w _ { y } ( z )$ is the score of the correct class, and the inner max aims to find the largest violation over all incorrect classes. For a perturbation $z ^ { \prime } = z + \delta , \| \delta \| \leq \epsilon .$ , we have

$$
w _ { j } ( z + \delta ) - w _ { y } ( z + \delta ) = ( w _ { j } - w _ { y } ) ^ { \top } z + ( b _ { j } - b _ { y } ) + ( w _ { j } - w _ { y } ) ^ { \top } \delta .
$$

Therefore,

$$
\operatorname* { s u p } _ { \| \delta \| \leq \epsilon } L ( w ( z + \delta ) , y ) = \operatorname* { s u p } _ { \| \delta \| \leq \epsilon } \operatorname* { m a x } \left\{ 0 , \operatorname* { m a x } _ { j \neq y } \left[ 1 + ( w _ { j } - w _ { y } ) ^ { \top } z + b _ { j } - b _ { y } + ( w _ { j } - w _ { y } ) ^ { \top } \delta \right] \right\} .
$$

Since the maximum is over finitely many (K) affine functions, the supremum can be exchanged with the finite maximum:

$$
= \operatorname* { m a x } \left\{ 0 , \operatorname* { m a x } _ { j \neq y } \left[ 1 + ( w _ { j } - w _ { y } ) ^ { \top } z + b _ { j } - b _ { y } + \operatorname* { s u p } _ { \| \delta \| \leq \epsilon } ( w _ { j } - w _ { y } ) ^ { \top } \delta \right] \right\} .
$$

Using the support function of the norm ball,

$$
\operatorname* { s u p } _ { \| \delta \| \leq \epsilon } ( w _ { j } - w _ { y } ) ^ { \top } \delta = \epsilon \| w _ { j } - w _ { y } \| _ { * } .
$$

Thus the robust multi-class hinge loss has the closed form

$$
\operatorname* { s u p } _ { \| \boldsymbol { z } ^ { \prime } - \boldsymbol { z } \| \leq \epsilon } L ( w ( \boldsymbol { z } ^ { \prime } ) , \boldsymbol { y } ) = \operatorname* { m a x } \left\{ 0 , \operatorname* { m a x } _ { j \neq \boldsymbol { y } } \big [ 1 + w _ { j } ( \boldsymbol { z } ) - w _ { \boldsymbol { y } } ( \boldsymbol { z } ) + \epsilon \| \mathbf { w } _ { j } - \mathbf { w } _ { \boldsymbol { y } } \| _ { * } \big ] \right\} .
$$

Equivalently, one may introduce auxiliary variables $u _ { j }$ satisfying

$$
\| w _ { j } - w _ { y } \| _ { * } \leq u _ { j } , \qquad j \neq y ,
$$

and write the robust loss as

$$
\operatorname* { m a x } \left\{ 0 , \operatorname* { m a x } _ { j \neq y } \left[ 1 + w _ { j } ( z ) - w _ { y } ( z ) + \epsilon u _ { j } \right] \right\} .
$$

This form is convex in the final-layer parameters for fixed features $z .$

Case 2: multi-class logistic / softmax cross-entropy loss. For true class y, the softmax crossentropy loss is

$$
L ( w ( z ) , y ) = - \log \frac { \exp ( w _ { y } ( z ) ) } { \sum _ { k = 1 } ^ { K } \exp ( w _ { k } ( z ) ) } = \log \left( \sum _ { k = 1 } ^ { K } \exp ( w _ { k } ( z ) - w _ { y } ( z ) ) \right) .
$$

Define

$$
a _ { k } = w _ { k } ( z ) - w _ { y } ( z ) , \qquad v _ { k } = { \mathbf w } _ { k } - { \mathbf w } _ { y } .
$$

Then

$$
w _ { k } ( z + \delta ) - w _ { y } ( z + \delta ) = a _ { k } + v _ { k } ^ { \top } \delta ,
$$

with $a _ { y } = 0$ and $v _ { y } = 0$ . The robust loss is

$$
\operatorname* { s u p } _ { \| \delta \| \leq \epsilon } \log \left( \sum _ { k = 1 } ^ { K } \exp ( a _ { k } + v _ { k } ^ { \top } \delta ) \right) .
$$

A useful variational representation is obtained from the Fenchel form of log-sum-exp:

$$
\log \sum _ { k = 1 } ^ { K } \exp ( r _ { k } ) = \operatorname* { s u p } _ { p \in \Delta _ { K } } \left\{ p ^ { \top } r + H ( p ) \right\} ,
$$

where

$$
H ( p ) = - \sum _ { k = 1 } ^ { K } p _ { k } \log p _ { k } .
$$

Applying this with $\boldsymbol { r } _ { k } = \boldsymbol { a } _ { k } + \boldsymbol { v } _ { k } ^ { \top } \delta$ gives

$$
\underset { | | \delta | | \leq \epsilon } { \operatorname* { s u p } } \log \sum _ { k = 1 } ^ { K } \exp ( a _ { k } + v _ { k } ^ { \top } \delta ) = \underset { p \in \Delta _ { K } } { \operatorname* { s u p } } \left\{ \sum _ { k = 1 } ^ { K } p _ { k } a _ { k } + H ( p ) + \epsilon \left\| \sum _ { k = 1 } ^ { K } p _ { k } v _ { k } \right\| _ { * } \right\} .\tag{19}
$$

However, this is not a closed-form perturbation of the same type as the binary case. Continuing from (19), we derive an efficient upper bound relaxation as follows

$$
\begin{array} { r l } & { \displaystyle \operatorname* { s u p } _ { \| \delta \| \leq \epsilon } \log \displaystyle \sum _ { k = 1 } ^ { K } \exp ( a _ { k } + v _ { k } ^ { \top } \delta ) } \\ & { \leq \displaystyle \operatorname* { s u p } _ { p \in \Delta _ { K } } \left. \displaystyle \sum _ { k = 1 } ^ { K } p _ { k } a _ { k } + H ( p ) + \epsilon \sum _ { k = 1 } ^ { K } p _ { k } \| v _ { k } \| _ { * } \right. \quad \mathrm { ( f e n s e n ' s ~ i n e q u a l i t y ~ o n ~  { \left| \cdot \right\| _ { * } } ) } } \\ & { = \displaystyle \operatorname* { s u p } _ { p \in \Delta _ { K } } \left. \displaystyle \sum _ { k = 1 } ^ { K } p _ { k } \left( a _ { k } + \epsilon \| v _ { k } \| _ { * } \right) + H ( p ) \right. \quad \mathrm { ( G r o u p i n g ~ l i n e a r ~ t e r m s ) } } \\ & { = \displaystyle \log \displaystyle \sum _ { k = 1 } ^ { K } \exp \left( a _ { k } + \epsilon \| v _ { k } \| _ { * } \right) \quad \mathrm { ( R e v e r s e ~ F e n c h e l ~ c o n j u g a t e ) } } \end{array}
$$

## D EXPERIMENTAL DETAILS

## D.1 SHIFTED DATASETS

Shifted CMNIST Dataset statistics are presented in Table 3. In the original HDRO experiment (Jo et al., 2026), rotation is only applied to test set and its distribution significantly differs from the validation distribution, causing large variance across different random seeds (8%-10%) and tuning on such validation set would make little sense. In our setup, rotation is applied to both validation and test split of the minority group (red, label 1). This is done to reduce the extreme large variance observed during model tuning and stabilize prediction performance. Note that only rotations are applied, and samples are not moved.

Table 3: CMNIST Statistics.
<table><tr><td>Group</td><td>Train</td><td>Val</td><td>Test</td></tr><tr><td>y = 0, green</td><td>2,998</td><td>2,591</td><td>8,966</td></tr><tr><td>y = 1, green</td><td>11,781</td><td>2,513</td><td>1,013</td></tr><tr><td>y = 0, red</td><td>12,130</td><td>2,465</td><td>1,068</td></tr><tr><td>y = 1, red</td><td>3,091</td><td>2,431</td><td>8,953</td></tr><tr><td>Total</td><td>30,000</td><td>10,000</td><td>20,000</td></tr></table>

Shifted CelebA Dataset statistics are presented in Table 4. Our shifting procedure is the same as that of Jo et al. (2026). For minority group (blond male), 164 without glasses are moved from test to train, 90 with glasses are moved from train to test, and 10 with glasses are moved from validation to test.

Table 4: CelebA Statistics
<table><tr><td>Group</td><td>Train</td><td>Val</td><td>Test</td></tr><tr><td>non-blond female non-blond male</td><td>71,629 66,874</td><td>8,535 8,276</td><td>9,767 7,535</td></tr><tr><td>blond female</td><td>22,880</td><td>2,874</td><td>2,480</td></tr><tr><td>blond male (before shift) blond male (after shift)</td><td>1,387 1,461</td><td>182 172</td><td>180 116</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Total (before shift) Total (after shift)</td><td>162,770 162,844</td><td>19,867 19,857</td><td>19,962 19,898</td></tr></table>

Shifted CivilComments Dataset statistic presented in Table 5. Similar to the test distribution shift of CMNIST and CelebA, we created a shifted version of the CivilComments dataset (Koh et al., 2021), which is originally used for toxic language classification. Intra-group shift on this dataset has not been studied previously to our knowledge. For the shifted version, 1,278 minority group (toxic, black) samples with white=1 are moved from train to test, and 905 minority group samples with white=0 are moved from test to train. After shifting, train minority group contains only non-whiteannotated samples, and test minority contains only white annotated samples. The fact that attributes black=1 and white=1 are not mutually exclusive allows such shift to happen.

Table 5: CivilComments Statistics
<table><tr><td>Group</td><td>Train</td><td>Val</td><td>Test</td></tr><tr><td>non-toxic, non-black</td><td>231,738</td><td>39,006</td><td>115,223</td></tr><tr><td>non-toxic, black</td><td>6,785</td><td>1,119</td><td>3,335</td></tr><tr><td>toxic, non-black</td><td>27,404</td><td>4,522</td><td>13,687</td></tr><tr><td>toxic, black (before shift)</td><td>3,111</td><td>533</td><td>1,537</td></tr><tr><td>toxic, black (after shift)</td><td>2,738</td><td>533</td><td>1,910</td></tr><tr><td>Total (before shift)</td><td>269,038</td><td>45,180</td><td>133,782</td></tr><tr><td>Total (after shift)</td><td>268,665</td><td>45,180</td><td>134,155</td></tr></table>

Shifted Waterbirds Waterbirds (Sagawa et al., 2020) is one of the most commonly used benchmark in this line of DRO research, but we do not use it here because (1) the dataset itself has known mislabeled attributes (Asgari et al., 2022), making performance metrics less informative; (2) the above issue is compounded with the small size of the training set (4,795 total, with 56 in the minority group of waterbird with land background); and (3) the performance has saturated with or without intra-group shifts on existing methods, rendering comparisons banal.

## D.2 SETTINGS

In Table 6, we provide the important settings (hyperparameters and architectures) used in this work. We emphasize that those hyperparameters are relatively insensitive to the dataset, so we only tune them on CMNIST for one method and use the same values for the other two datasets. $\epsilon _ { g }$ is cliped [0,1] to follow the HDRO preset range.

Table 6: Settings and Hyperparameters for our methods.
<table><tr><td>Setting</td><td>CMNIST</td><td>CelebA</td><td>CivilComments</td></tr><tr><td>Model</td><td>ResNet-50</td><td>ResNet-50</td><td>DistilBERT</td></tr><tr><td>Weight decay λ</td><td> $1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 1 }$ </td><td>10-3</td></tr><tr><td>Lower Train/Upper Validation batch size</td><td>256</td><td>128</td><td>64</td></tr><tr><td> $L _ { \nabla h }$  in Shen et al. (2026)</td><td></td><td>10000</td><td></td></tr><tr><td>Iterations of SAPD per outer iteration T</td><td></td><td>1</td><td></td></tr><tr><td>ηinit</td><td></td><td>1.0</td><td></td></tr><tr><td>Gradient multiplier of η</td><td></td><td>10</td><td></td></tr><tr><td>Gradient multiplier of validation simplex u</td><td></td><td>100</td><td></td></tr><tr><td> $\epsilon _ { \mathrm { i n i t } }$  for Bi-HDRO</td><td></td><td>96/255</td><td></td></tr><tr><td>KL coefficient β for Bi-PG-DRO</td><td></td><td>10</td><td></td></tr></table>

## D.3 BI-HDRO

## D.3.1 ABLATION PROCEDURE

To show the advantage of automatically tuning both η and ϵ $( \mathrm { o r } \epsilon _ { g } )$ in Bi-HDRO, (1) we fixed ϵ to be 0 and leave η learnable, which recovers Bi-GDRO; (2) we performed grid search with fixed ϵ over {60,72,84,96,108}/255 when $\eta = 0 .$ , and (3) we performed grid search ϵ with the above schedule and η over {0.01, 0.1, 1.0, 10.0}.

## D.3.2 BASELINES

$\mathbf { D F R } ^ { \mathrm { T r } }$ (Kirichenko et al., 2023) A two-stage method where the training set is partitioned 80-20 for different purposes. (1) The larger chunk of data is used to train ERM with uniform sampling. (2) The smaller group-balanced subset to retrain last layer. For DFR<sup>Val</sup>, they include all minority group data in validation and sample the same amount from other groups. For fairness of group information access, we compare our method to $\mathrm { D F R ^ { T r } }$ only. Default hyperparameters from their implementation are used.

PDE (Deng et al., 2023) A two-stage method. (1) Warmup the entire model using group-balanced subset where all groups have same size as the minority group. (2) Progressively add more training data (progressive data expansion) for training the entire model as well as using existing warmup subset. We tune the warm-up epochs in {10,20} and added samples in {50,100,500}.

HDRO (Jo et al., 2026) A single-level min-max-max method described in Section 2.3. We use the implementation from the authors and tune ϵ in the range {60,72,84,96}/255 and $C \in \{ 0 , 1 , 2 , 3 \}$ across all three of our datasets.

## D.4 BI-PG-DRO

## D.4.1 ABLATION PROCEDURE

We ablated two types of regularization: BCE loss for attribute predictor, KL prior regularization for attribute predictor. We performed a grid search {1,10,30} for these two terms.

## D.4.2 BASELINES

$\mathbf { G I C } ^ { C _ { y } }$ (Han & Zou, 2024) A three-stage method. (1) Feature extraction. (2) Train attribute predictor maximizing spurious attribute label KL between predicted label and true label (Eq (11) in their paper, hence the superscript). (3) Train GDRO. We tune the γ (Eq (9) in their paper) in {2,5,10} for our shifted datasets.

XRM (Pezeshki et al., 2024) A two-stage method. (1) Train two auxilary models with mutually exclusive held-in and held-out split of the training set. Notably, it does not rely on any attribute labels from, for example, the validation set. Instead, it flips training set labels so that minority samples are identified, since they are confidently misclassified in the held-out set. (2) Train GDRO with pseudo-group-labels. We sample three hyperparameter combination candidates, choose one with the highest flip rates, and train GDRO using 3 different seeds.

![](images/3b2892307db0763495440d0106154a29f19605d54154676f4f325ec5a0c5c6ae.jpg)  
(a) CMNIST

![](images/5e30ca4989abd090d09921f583ca6eb84948196c112c619b4fd3803d7c2d4ec6.jpg)  
(b) CelebA

![](images/2df7c5e4a1ea8c80a4fba945b9f0776c651d9941caa7c6c5db02ad12d96b5453.jpg)  
(c) CivilComments  
Figure 1: Validation Curve of HDRO and Bi-HDRO.

AGRO (Paranjape et al., 2023) A greedy unified method that jointly trains attribute predictor and robust model. It learns soft group memberships that makes robust training (GDRO) difficult. We use the hyperparameters described in their Table 7 (and 4 slices for CMNIST), but tune their α in {0.2,0.3,0.4} due to our different setup (intra-group shift).

SSA (Nam et al., 2022) A two-stage method. (1) Train a attribute label prediction model and performs hard prediction to generate pseudo-group-label and (2) Train GDRO. Default hyperparameters are used. We tune the model with $\mathsf { \bar { C } } \in \{ 0 , 1 , 2 , 3 \}$ across all three of our datasets.

PG-DRO (Ghosal & Li, 2023) A two-stage method described in Section 2.4. Essentially SSA but with probabilistic group labels. We tune the model with $C \in \{ 0 , 1 , 2 , 3 \}$ across all three of our datasets.

## E ADDITIONAL RESULTS

## E.1 EFFICIENCY

Validation performance. In Figure 1, we show that Bi-HDRO reaches optimal validation accuracy quicker than HDRO does. For CelebA, HDRO took about 6,000 optimization steps to reach level similar to Bi-HDRO. Additionally, HDRO requires extensive grid search to obtain such results, while Bi-HDRO requires only one run.

Runtime. We show below in Table 7 that Bi-HDRO takes shorter time to run. HDRO would require much longer runtime than stated if grid search is taken into account. CMNIST in general takes longer to achieve reasonable performance.

Table 7: Wall-clock time for one run in seconds.
<table><tr><td></td><td>CMNIST</td><td>CelebA</td><td>CivilComments</td></tr><tr><td>HDRO</td><td>405</td><td>3145</td><td>625</td></tr><tr><td> $\mathbf { B i - H D R O } \left( \epsilon _ { g } \right)$ </td><td>6674</td><td>129</td><td>3912</td></tr></table>

## E.2 DOES PERTURBATION HELP SOFT GROUP ASSIGNMENTS (BI-PG-DRO)?

We decide not to perturb the latent space in Bi-PG-DRO because empirically, fixed perturbation alone degrades model performance, and $\epsilon _ { g }$ converges to 0 when automatically tuned. On CMNIST, we found that perturbation $\epsilon _ { g }$ all converged to 0 regardless of initial values (Figure 2). We attempted 3 setups: initial value is 0, initial value is 96/255, initial value is 0 but unbounded from above. Validation performance worsens in Bi-PG-DRO with active perturbation. We observed similar performance degradation on CelebA and CivilComments. Based on this result, we decide not to include perturbation with probabilistic group membership.

![](images/8dea1852461b9a3112b2231b3149fe82b7bc15a24810b4773784cf2be356c555.jpg)  
Figure 2: $\epsilon _ { g }$ converges to 0 for Bi-PG-DRO. First row: $\epsilon _ { \mathrm { i n i t } } = 0 .$ Second row: $\epsilon _ { \mathrm { i n i t } } = 9 6 / 2 5 5 .$ Third row: Unclipped $\epsilon _ { \mathrm { i n i t } } = 0$ . Last column is the shifted minority group.

## E.3 SHOULD MEMBERSHIP INFERENCE BE PRECISE FOR WORST-GROUP PERFORMANCE?

As long as worst-group validation loss is focused on during bilevel optimization, predicted group labels need not be precise. In Figure 3, we show that post-hoc attribute accuracy over group-unlabeled training set without upper level regularization is poor, but the validation performance does not degrade as much. Moreover, we show that KL regularization instead of explicit BCE training provides closer attribute prediction accuracy results and gives a small boost in validation worst-group accuracy.

![](images/ae1aeea4e27a5d53b40f5ec0bdee6ad433bc453e89ba0af580905e47bb9939d0.jpg)

![](images/50de9c94936d37f17fa7d009bf530e0d5fcfd0842932a446668ed898af2aa7c1.jpg)

![](images/05ed0c3128b49f31302f756e36ffa8fe1fc2e0efdc91449f10b6164914c6cd69.jpg)

![](images/c72804dd1eb98dfa3283d40bf981418e3c366da4cbc61014311c82a3678ff913.jpg)  
(a) CMNIST

![](images/48b90ed0bd0600322667d44f37dff13110563404f817eaa56864677c665d3d70.jpg)  
(b) CelebA

![](images/49fcdba9883f9630042b53eb2f6e6d4a58e43279237655c450a23631e22285aa.jpg)  
(c) CivilComments  
Figure 3: Attribute prediction accuracy on group-unlabeled training data (top) and worst-group validation accuracy (bottom) across three datasets organized by column.

## F GENERALIZATION THEORY FOR CONTINUOUS BILEVEL HYPERPARAMETER TUNING

In this section, we analyze the generalization guarantees of continuous bilevel hyperparameter tuning. Our primary theoretical goal is to establish a Continuous Oracle Inequality for Group Distributionally Robust Optimization (DRO). This inequality characterizes how optimizing hyperparameters on a validation set allows the algorithm to seamlessly navigate the fundamental trade-off between structural complexity and worst-case group robustness.

Roadmap. Because establishing continuous generalization bounds requires multiple statistical learning tools, our analysis proceeds in four stages.

1. We formalize the bilevel setup and explicitly prove that the lower-level optimization algorithm induces a bounded, Lipschitz-continuous hypothesis space with respect to the hyperparameters (Appendix F.1).

2. We establish the mathematical machinery required to bound the upper-level validation error across the entire continuous hyperparameter path using Rademacher complexity (Appendix F.3).

3. As a theoretical warm-up, we apply this machinery to standard Regularized Empirical Risk Minimization, which isolates how the upper-level continuous tuning error cleanly decouples from the lower-level uniform stability (Appendix F.4).

4. We derive our main result: the Group DRO Oracle Inequality (Appendix F.5).

5. Finally, we extend this continuous generalization theory to Hierarchical DRO (Bi-HDRO), demonstrating that tuning multi-dimensional perturbation radii preserves the algorithmic Lipschitz continuity and resulting Oracle Inequalities (Appendix F.6).

## F.1 SETUP AND ASSUMPTIONS

Throughout this section, we make the following formal assumptions about the learning problem and the bilevel setup:

1. Loss Function: The instantaneous loss function $\ell ( W , z )$ is convex and L-Lipschitz with respect to W. Furthermore, its absolute value is bounded by $M \left( \mathrm { i . e . , } \vert \ell ( W , z ) \vert ^ { \cdot } \le M ) \right.$ over a bounded optimization domain of radius $B \left( \mathrm { i . e . , } \| W \| \leq \dot { B } \right)$

2. Continuous Hyperparameter Space: The generic hyperparameter $\phi$ is tuned over a continuous k-dimensional bounded domain $\Phi \ { \overset { \mathbf { \textstyle ~ \subset ~ } } { \subset } } \ \mathbb { R } ^ { k }$ . We assume the maximum $\ell _ { 2 }$ distance between any two hyperparameters in Φ is bounded by a diameter R.

3. Strong Convexity of Regularization: The lower-level objective includes a regularization term $\bar { \Omega } _ { \lambda } ( W )$ that is λ-strongly convex with respect to W.

4. Algorithmic Mapping and Lipschitz Continuity: The lower-level optimization acts as a mapping algorithm $\mathcal { A } : \Phi  \bar { \mathbb { R } ^ { d } }$ on a training set $S _ { \mathrm { t r a i n } }$ of size $n ^ { \mathrm { t r } }$ , outputting a parameter $\widehat { W } _ { \phi } = \mathcal { A } ( \phi )$ . We require this mapping A to be $\rho _ { \mathcal { A } }$ -Lipschitz continuous with respect to ϕ in the $\ell _ { 2 }$ norm: $\lVert \boldsymbol { \mathcal { A } } ( \boldsymbol { \dot { \phi } } _ { 1 } ) - \boldsymbol { \mathcal { A } } ( \boldsymbol { \phi } _ { 2 } ) \rVert \leq \rho _ { \boldsymbol { \mathcal { A } } } \lVert \boldsymbol { \phi } _ { 1 } - \boldsymbol { \dot { \phi } } _ { 2 } \rVert$ . While formalized here as a high-level requirement for our general theorems, we explicitly demonstrate later that this continuity is a consequence of strong convexity (Assumption 3) for our specific learning objectives.

Based on this mapping, the upper-level problem evaluates the induced predictors on an independent validation set $S _ { \mathrm { v a l } }$ of size $n ^ { \mathrm { v a l } }$ . We define the effective algorithmic hypothesis class explored by the validation process as the k-dimensional continuous manifold induced by A:

$$
{ \mathcal { H } } _ { \Phi } = \{ { \widehat { W } } _ { \phi } : \phi \in \Phi \}\tag{20}
$$

To establish our generalization guarantees, we rely on the classical framework of uniform stability.   
For completeness, we defer the standard formal definitions and stability derivations to Appendix F.7.

## F.2 ALGORITHMIC LIPSCHITZ CONTINUITY EXAMPLES

To provide concrete intuition for Assumption 4 (Algorithmic Lipschitz Continuity), we walk through two representative cases: a warm-up for standard Regularized Loss Minimization, and a formal proof for our primary application of Group DRO.

Example 1: (Warm-up) Standard Regularized Loss Minimization. Consider a standard bilevel formulation for Regularized Empirical Risk Minimization (ERM), where the goal is to tune the continuous regularization penalty. Here, the one-dimensional continuous hyperparameter is the regularization weight $\lambda \in \Lambda \bar { = } \left[ \lambda _ { \operatorname* { m i n } } , \lambda _ { \operatorname* { m a x } } \right]$ where $\lambda _ { \operatorname* { m i n } } > 0$ . The lower-level objective minimizes the regularized empirical risk over the training set:

$$
F _ { \lambda } ( W ) = L ^ { \operatorname { t r } } ( W ) + \Omega _ { \lambda } ( W )\tag{21}
$$

yielding the optimal predictor ${ \widehat { W } } _ { \lambda } = \arg \operatorname* { m i n } _ { W } F _ { \lambda } ( W )$ . The upper-level problem evaluates these predictors to minimize the unregularized risk on a validation set: min $. \lambda { \in } \Lambda \ L ^ { \mathrm { v a l } } ( { \widehat { W } } _ { \lambda } )$

Because the lower-level objective $F _ { \lambda } ( W )$ is λ-strongly convex (Assumption 3), the Lipschitz continuity of the mapping $\lambda \mapsto \widehat { W } _ { \lambda }$ is naturally guaranteed without requiring differentiability of the loss. By the property of strong convexity at the optimum $\widehat { W } _ { \lambda }$ , for any W we have $F _ { \lambda } ( W ) \ \geq$ $\begin{array} { r } { F _ { \lambda } ( \widehat { W } _ { \lambda } ) + \frac { \lambda } { 2 } \| W - \widehat { W } _ { \lambda } \| ^ { 2 } } \end{array}$ . Evaluating this for $\lambda _ { 1 }$ at $W = \widehat { W } _ { \lambda _ { 2 } }$ and for $\lambda _ { 2 }$ at $W = \widehat { W } \lambda _ { 1 }$ gives two inequalities:

$$
\begin{array} { r l } & { F _ { \lambda _ { 1 } } ( \widehat { W } _ { \lambda _ { 2 } } ) \geq F _ { \lambda _ { 1 } } ( \widehat { W } _ { \lambda _ { 1 } } ) + \frac { \lambda _ { 1 } } { 2 } \| \widehat { W } _ { \lambda _ { 2 } } - \widehat { W } _ { \lambda _ { 1 } } \| ^ { 2 } } \\ & { F _ { \lambda _ { 2 } } ( \widehat { W } _ { \lambda _ { 1 } } ) \geq F _ { \lambda _ { 2 } } ( \widehat { W } _ { \lambda _ { 2 } } ) + \frac { \lambda _ { 2 } } { 2 } \| \widehat { W } _ { \lambda _ { 1 } } - \widehat { W } _ { \lambda _ { 2 } } \| ^ { 2 } } \end{array}
$$

Expanding $F _ { \lambda } ( W ) = L ^ { \operatorname { t r } } ( W ) + \Omega _ { \lambda } ( W )$ and summing them, the empirical loss terms $L ^ { \mathrm { t r } } ( \widehat { W } _ { \lambda _ { 1 } } )$ and $L ^ { \mathrm { t r } } ( \widehat { W } _ { \lambda _ { 2 } } )$ exactly cancel out on both sides. Rearranging the remaining regularization terms leaves:

$$
\frac { \lambda _ { 1 } + \lambda _ { 2 } } { 2 } \| \widehat { W } _ { \lambda _ { 1 } } - \widehat { W } _ { \lambda _ { 2 } } \| ^ { 2 } \leq \left( \Omega _ { \lambda _ { 1 } } ( \widehat { W } _ { \lambda _ { 2 } } ) - \Omega _ { \lambda _ { 1 } } ( \widehat { W } _ { \lambda _ { 1 } } ) \right) + \left( \Omega _ { \lambda _ { 2 } } ( \widehat { W } _ { \lambda _ { 1 } } ) - \Omega _ { \lambda _ { 2 } } ( \widehat { W } _ { \lambda _ { 2 } } ) \right)\tag{22}
$$

Consider the general proximal regularization case where $\Omega _ { \lambda } ( W ) = \textstyle { \frac { \lambda } { 2 } } \| W - \mathbf { a } \| ^ { 2 }$ for some reference vector a (standard Ridge is recovered when ${ \bf a } = { \bf 0 } )$ . The right side simplifies to $\frac { \lambda _ { 1 } - \lambda _ { 2 } } { 2 } ( | | \widehat { W } _ { \lambda _ { 2 } }  -$ $\mathbf { a } \| ^ { 2 } - \| \widehat { W } _ { \lambda _ { 1 } } - \mathbf { a } \| ^ { 2 } )$ ). Using the algebraic identity $\| x \| ^ { 2 } - \| y \| ^ { 2 } = \langle x - y , x + y \rangle$ , we can rewrite this difference as:

$$
\frac { \lambda _ { 1 } - \lambda _ { 2 } } { 2 } \langle \widehat { W } _ { \lambda _ { 2 } } - \widehat { W } _ { \lambda _ { 1 } } , \widehat { W } _ { \lambda _ { 2 } } + \widehat { W } _ { \lambda _ { 1 } } - 2 \mathbf { a } \rangle
$$

Applying the Cauchy-Schwarz inequality, and noting that the average $\frac { \lambda _ { 1 } + \lambda _ { 2 } } { 2 }$ is strictly lowerbounded by $\lambda _ { \mathrm { m i n } } , \mathrm { y i e l d s }$

$$
\lambda _ { \operatorname* { m i n } } \| \widehat { W } _ { \lambda _ { 1 } } - \widehat { W } _ { \lambda _ { 2 } } \| ^ { 2 } \leq \frac { | \lambda _ { 1 } - \lambda _ { 2 } | } { 2 } \| \widehat { W } _ { \lambda _ { 1 } } - \widehat { W } _ { \lambda _ { 2 } } \| \| \widehat { W } _ { \lambda _ { 1 } } + \widehat { W } _ { \lambda _ { 2 } } - 2 { \mathsf { a } } \|\tag{23}
$$

By the triangle inequality, and assuming the optimization domain is bounded by a constant radius B (Assumption 1), the second term is bounded by $\begin{array} { r } { \| \widehat { W } _ { \lambda _ { 1 } } \| + \| \widehat { W } _ { \lambda _ { 2 } } \| + 2 \| \mathbf { a } \| \leq 2 ( B + \| \mathbf { a } \| ) } \end{array}$ . Dividing by $\| \widehat { W } _ { \lambda _ { 1 } } - \widehat { W } _ { \lambda _ { 2 } } \|$ proves that the mapping A is strictly bounded by a constant $\rho _ { \mathcal { A } } = ( B + \| \mathbf { a } \| ) / \lambda _ { \operatorname* { m i n } }$ and is therefore $\rho _ { \mathcal { A } } { \mathrm { - L i p s c h i t z } }$

$$
\begin{array} { r } { \| \widehat { W } _ { \lambda _ { 1 } } - \widehat { W } _ { \lambda _ { 2 } } \| \leq \rho _ { { \cal A } } | \lambda _ { 1 } - \lambda _ { 2 } | } \end{array}\tag{24}
$$

Alternatively, if the loss and regularizer are strictly twice continuously differentiable, one can recover this same Lipschitz constant $\rho _ { \mathcal { A } } \ \leq \ B / \lambda _ { \operatorname* { m i n } }$ by directly bounding the spectral norm of the Implicit Function Theorem Jacobian: $\begin{array} { r } { \frac { d \widehat { W } _ { \lambda } } { d \lambda } = - [ \nabla _ { W } ^ { 2 } F _ { \lambda } ( \widehat { W } _ { \lambda } ) ] ^ { - 1 } \nabla _ { \lambda W } ^ { 2 } F _ { \lambda } ( \widehat { W } _ { \lambda } ) } \end{array}$

Example 2: Group Distributionally Robust Optimization (DRO). As a concrete application, consider a linearized Group DRO setting where we optimize a linear classifier W directly on fixed inputs. Given training data partitioned into G groups with empirical group losses $L ^ { \operatorname { t r } } ( \dot { W } ) \in \mathbb { R } ^ { G }$ the lower level optimizes the classifier against a worst-case group distribution $q \in \Delta _ { G }$ (the probability simplex weighting the groups). The deviation of this worst-case distribution from the uniform distribution is penalized by a robustness hyperparameter $\eta \in H = [ \eta _ { \mathrm { m i n } } , \eta _ { \mathrm { m a x } } ]$ where $\eta _ { \mathrm { m i n } } > 0$ The lower-level problem is:

$$
\widehat { W } _ { \eta } = \arg \operatorname* { m i n } _ { W } \operatorname* { m a x } _ { q \in \Delta _ { G } } \left[ q ^ { T } L ^ { \mathrm { t r } } ( W ) - \frac { \eta } { 2 } \left\| q - \frac { 1 } { G } { \bf 1 } \right\| ^ { 2 } + \frac { \lambda } { 2 } \| W \| ^ { 2 } \right]\tag{25}
$$

where λ is a fixed $L _ { 2 }$ regularization coefficient. In the upper level, we tune the robustness hyperparameter $\eta \in H$ using an independent validation set $D _ { \mathrm { v a l } }$ (partitioned into $G$ groups) to minimize the worst-group validation loss. Here, η naturally assumes the role of the generic continuous tuning hyperparameter ϕ (with $k = 1 )$ from Section F.1, while λ serves strictly as the fixed strong convexity constant required for stability:

$$
\operatorname* { m i n } _ { \eta \in H } \operatorname* { m a x } _ { g \in [ G ] } L _ { g } ^ { \mathrm { v a l } } ( \widehat { W } _ { \eta } )\tag{26}
$$

By continuously tuning η, the bilevel formulation dynamically discovers the optimal trade-off between average-case empirical risk and worst-group robustness on unseen data.

Remarkably, this formulation strictly satisfies the algorithmic Lipschitz condition (Assumption 4).   
We formally state this property below.

Lemma F.1 (Algorithmic Lipschitz Continuity of Group DRO). Suppose the instantaneous group losses $L _ { g } ^ { \mathrm { t r } } ( W )$ are L-Lipschitz and bounded by M over a bounded optimization domain of radius B (Assumption 1). For any fixed strong convexity regularizer $\lambda > 0$ (Assumption 3), the maxmarginalized lower-level objective $J _ { \eta } ( W )$ induces an algorithmic mapping $\widehat { W } _ { \eta }$ that is $\rho _ { \mathcal { A } } – L i p s c h i t z$ continuous with respect to the robustness hyperparameter $\eta \ge \eta _ { \mathrm { m i n } } > 0$ , with explicit constant:

$$
\rho _ { \mathcal { A } } \leq \frac { G L M } { \lambda \eta _ { \mathrm { m i n } } ^ { 2 } }\tag{27}
$$

Proof. Let $J _ { \eta } ( W ) = \operatorname* { m a x } _ { q \in \Delta _ { G } } F ( W , q , \eta )$ be the max-marginalized lower-level objective, where ${ \cal F } ( W , q , \eta ) \stackrel { . } { = } q ^ { T } { \cal L } ^ { \mathrm { t r } } ( W ) \stackrel { . } { - } \frac { \eta } { 2 } \| q - \textstyle \frac { 1 } { G } { \bf 1 } \| ^ { 2 } + \frac { \lambda } { 2 } \| W \| ^ { 2 }$ . By Danskin’s theorem (Danskin, 1967), the gradient of the max-marginalized function is simply the gradient of the objective evaluated at the optimal inner variable. Thus, its gradient is $\nabla J _ { \eta } ( \mathsf { \tilde { W } } ) \ = \ \nabla _ { W } F ( W , q _ { \eta } ^ { * } ( W ) , \eta ) \ =$

$J ( W ) q _ { n } ^ { * } ( W ) + \lambda W$ , where $J ( W )$ is the $d \times G$ Jacobian matrix of group losses. By Assumption 1, each group loss is L-Lipschitz, so the operator norm of $J ( W )$ is bounded by its Frobenius norm $\sqrt { \sum _ { g = 1 } ^ { G } \| \nabla L _ { g } ^ { \mathrm { t r } } ( W ) \| ^ { 2 } } \leq \sqrt { G } L$

Next, we bound how much this gradient shifts with respect to $\eta .$ The optimal inner distribution is exactly the simplex projection $q _ { \eta } ^ { * } \mathbf { \bar { ( } } W ) = \Pi _ { \Delta _ { G } } \left( \mathbf { u } _ { \eta } ( W ) \right)$ , where $\begin{array} { r } { \dot { \mathbf { u } } _ { \eta } ( W ) \stackrel { \cdot } { = } \frac { 1 } { n } L ^ { \mathrm { t r } } ( W ) + \frac { 1 } { G } \mathbf { 1 } } \end{array}$ . Because the projection $\Pi _ { \Delta _ { G } }$ is non-expansive (1-Lipschitz), the shift in the optimal inner distribution is bounded by the shift in the unprojected vector: $\begin{array} { r } { \| q _ { \eta _ { 1 } } ^ { * } ( W ) - q _ { \eta _ { 2 } } ^ { * } ( W ) \| \overset { \cdot } { \leq } \| \mathbf { u } _ { \eta _ { 1 } } ( W ) - \mathbf { u } _ { \eta _ { 2 } } ( W ) \| = } \end{array}$ $\begin{array} { r l } {  { \big \| \big ( \frac { 1 } { \eta _ { 1 } } - \frac { 1 } { \eta _ { 2 } } \big ) L ^ { \mathrm { t r } } ( W ) \big \| } } \end{array}$ . By Assumption 1, each group loss is bounded by $M ,$ so $\| L ^ { \operatorname { t r } } ( W ) \| \leq \sqrt { G } M$ Requiring $\eta \geq \eta _ { \mathrm { m i n } } > 0 .$ , we have $| 1 / \eta _ { 1 } - 1 / \eta _ { 2 } | \leq | \eta _ { 1 } - \eta _ { 2 } | / \eta _ { \mathrm { m i n } } ^ { 2 }$ . This directly provides the shift bound: $\begin{array} { r } { \| q _ { \eta _ { 1 } } ^ { * } ( W ) - q _ { \eta _ { 2 } } ^ { * } ( W ) \| \le \frac { \sqrt { G } M } { \eta _ { \cdots } ^ { 2 } \vert \eta _ { 1 } - \eta _ { 2 } \vert } } \end{array}$ , which in turn limits the overall gradient shift to $\begin{array} { r } { \| \nabla J _ { \eta _ { 1 } } ( W ) - \nabla J _ { \eta _ { 2 } } ( W ) \| \le \frac { G L M } { \eta _ { \mathrm { m i n } } ^ { 2 } } | \eta _ { 1 } - \eta _ { 2 } | . } \end{array}$

Finally, let $\widehat { W } _ { \eta _ { 1 } }$ and $\widehat { W } _ { \eta _ { 2 } }$ be the optimal lower-level classifiers for hyperparameters $\eta _ { 1 }$ and $\eta _ { 2 }$ . Because $J _ { \eta }$ is λ-strongly convex, its gradient is λ-strongly monotone:

$$
\langle \nabla J _ { \eta _ { 1 } } ( \widehat { W } _ { \eta _ { 1 } } ) - \nabla J _ { \eta _ { 1 } } ( \widehat { W } _ { \eta _ { 2 } } ) , \widehat { W } _ { \eta _ { 1 } } - \widehat { W } _ { \eta _ { 2 } } \rangle \geq \lambda \Vert \widehat { W } _ { \eta _ { 1 } } - \widehat { W } _ { \eta _ { 2 } } \Vert ^ { 2 }
$$

For any closed convex domain (in particular $\| W \| \leq B )$ , the first-order optimality condition dictates:

$$
\begin{array} { r } { \langle - \nabla J _ { \eta _ { 1 } } ( \widehat { W } _ { \eta _ { 1 } } ) , \widehat { W } _ { \eta _ { 1 } } - \widehat { W } _ { \eta _ { 2 } } \rangle \geq 0 } \\ { \langle \nabla J _ { \eta _ { 2 } } ( \widehat { W } _ { \eta _ { 2 } } ) , \widehat { W } _ { \eta _ { 1 } } - \widehat { W } _ { \eta _ { 2 } } \rangle \geq 0 } \end{array}
$$

Summing the above three inequalities and applying the Cauchy-Schwarz inequality yields:

$$
\begin{array} { r l } { \lambda \| \widehat { W } _ { \eta _ { 1 } } - \widehat { W } _ { \eta _ { 2 } } \| ^ { 2 } \leq \langle \nabla J _ { \eta _ { 2 } } ( \widehat { W } _ { \eta _ { 2 } } ) - \nabla J _ { \eta _ { 1 } } ( \widehat { W } _ { \eta _ { 2 } } ) , \widehat { W } _ { \eta _ { 1 } } - \widehat { W } _ { \eta _ { 2 } } \rangle } & { } \\ { \leq \| \nabla J _ { \eta _ { 2 } } ( \widehat { W } _ { \eta _ { 2 } } ) - \nabla J _ { \eta _ { 1 } } ( \widehat { W } _ { \eta _ { 2 } } ) \| \| \widehat { W } _ { \eta _ { 1 } } - \widehat { W } _ { \eta _ { 2 } } \| } & { } \end{array}\tag{28}
$$

Dividing by $\lambda \| \widehat { W } _ { \eta _ { 1 } } - \widehat { W } _ { \eta _ { 2 } } \|$ and plugging in our bound for the gradient shift establishes the explicit Lipschitz constant $\rho _ { \mathcal { A } } .$ □

Thus, tuning the DRO robustness parameter over a continuous space is theoretically well-behaved and Lipschitz-stable as long as the search space is bounded away from zero $( \eta \ge \eta _ { \mathrm { m i n } } > 0 )$ .

## F.3 UNIFORM CONVERGENCE OVER MULTI-DIMENSIONAL CONTINUOUS HYPOTHESIS SPACES

Before specializing to specific learning algorithms like ERM or Group DRO, we first establish a general uniform convergence bound for the k-dimensional continuous hypothesis class ${ \mathcal { H } } _ { \Phi }$ . By leveraging the Lipschitz continuity of the algorithmic mapping, we can bound the Rademacher complexity of this manifold as a function of the hyperparameter space dimensionality k.

Lemma F.2 (Rademacher Complexity of k-dimensional Algorithmic Hypothesis Class). Suppose the loss function ℓ is L-Lipschitz and bounded by M (Assumption 1). Let Φ $\subset \mathbb { R } ^ { k }$ be a bounded kdimensional hyperparameter space with $\ell _ { 2 }$ diameter R (Assumption 2). If the lower-level mapping $\mathcal { A } ( \phi ) = \widehat { W } _ { \phi } i s \rho _ { \mathcal { A } }$ -Lipschitz (Assumption 4), the empirical Rademacher complexity of the validation loss class over H<sub>Φ</sub> on a set $S _ { \nu a l }$ ofsize $n ^ { \mathrm { v a l } }$ is bounded by:

$$
\hat { \mathcal { R } } _ { S _ { v a l } } ( \mathcal { H } _ { \Phi } ) \leq \frac { 6 M } { \sqrt { n ^ { \mathrm { v a l } } } } \sqrt { k } \left( \sqrt { \log \left( \frac { 3 \rho _ { A } R L } { M } \right) } + 2 \sqrt { \log 2 } \right)\tag{29}
$$

Proof. The proof relies on bounding the continuous covering number and applying discrete chaining. For a k-dimensional space Φ with $\ell _ { 2 }$ diameter R, its r-covering number in $\ell _ { 2 }$ norm is bounded by $N ( r , \Phi ) \leq ( 3 R / r ) ^ { k }$ for $r \leq R$ . This follows from a standard volumetric argument: a maximal r-separated set of size N in Φ induces N disjoint balls of radius $r / 2$ . Since Φ has diameter $R ,$ these disjoint balls are entirely contained within a larger ball of radius $\dot { R } + r / 2$ . Comparing their volumes yields $N ( r / 2 ) ^ { k } \leq ( R + r / 2 ) ^ { k }$ , which implies $N \leq ( 1 + 2 R / r ) ^ { k } \leq ( 3 R / r ) ^ { k }$ when $r \leq R$ . Due to the $\rho _ { \mathcal { A } }$ -Lipschitz property of the mapping, an r-cover of Φ projects to an $( \rho _ { \mathcal { A } } \cdot r )$ -cover of the hypothesis class ${ \mathcal { H } } _ { \Phi }$ . Thus, the covering number of the effective hypothesis class is bounded by $\dot { N ( r , \mathcal { H } _ { \Phi } ) } \le ( 3 \rho _ { A } R / r ) ^ { k }$

Let $A \subset \mathbb { R } ^ { n ^ { \mathrm { v a l } } }$ be the set of loss evaluations on $S _ { \mathrm { v a l } } = \{ z _ { 1 } , \ldots , z _ { n ^ { \mathrm { v a l } } } \}$ for all predictors in ${ \mathcal { H } } _ { \Phi }$ The maximum $\ell _ { 2 }$ norm of any vector in A is $C = M { \sqrt { n ^ { \mathrm { v a l } } } }$ . Since the loss is L-Lipschitz, the $\ell _ { 2 }$ distance between two loss vectors $\mathbf { a } _ { \phi _ { 1 } }$ and $\mathbf { a } _ { \phi _ { 2 } }$ generated by predictors $\widehat { W } _ { \phi _ { 1 } }$ and $\widehat { \overline { { W } } } _ { \phi _ { 2 } }$ is bounded by $\| \mathbf { a } _ { \phi _ { 1 } } - \mathbf { a } _ { \phi _ { 2 } } \| _ { 2 } \leq L \sqrt { n ^ { \mathrm { v a l } } } \| \widehat { W } _ { \phi _ { 1 } } - \widehat { W } _ { \phi _ { 2 } } \| _ { 2 }$ . Therefore, guaranteeing an ϵ-cover of A requires at most an r-cover of ${ \mathcal { H } } _ { \Phi }$ with $\begin{array} { r } { r = \frac { \epsilon } { L \sqrt { n ^ { \mathrm { v a l } } } } . } \end{array}$ . Thus, the covering number of A is bounded by $N ( \epsilon , A ) \leq$ $\begin{array} { r } { N ( r , \mathcal { H } _ { \Phi } ) \le \left( \frac { 3 \rho _ { \mathcal { A } } R L \sqrt { n ^ { \mathrm { v a l } } } } { \epsilon } \right) ^ { k } } \end{array}$

By the discrete chaining lemma (Shalev-Shwartz & Ben-David, 2014, Lemma 27.4), we evaluate the complexity over discrete scales $\epsilon _ { i } = C 2 ^ { - i }$ . For any $i \geq 1$

$$
\sqrt { \log N ( C 2 ^ { - i } , A ) } \leq \sqrt { k \log \left( \frac { 3 \rho _ { A } R L \sqrt { n ^ { \mathrm { v a l } } } } { M \sqrt { n ^ { \mathrm { v a l } } } } 2 ^ { i } \right) } \leq \sqrt { k } ( \alpha + \beta i )\tag{30}
$$

where $\alpha = \sqrt { \log \left( \frac { 3 \rho _ { A } R L } { M } \right) }$ and $\beta = \sqrt { \log { 2 } }$ . Evaluating the full infinite chaining sum gives the explicit Rademacher complexity (Shalev-Shwartz & Ben-David, 2014, Lemma 27.5):

$$
\hat { \mathcal { R } } _ { S _ { \mathrm { v a l } } } ( \mathcal { H } _ { \Phi } ) \leq \frac { 6 C } { n ^ { \mathrm { v a l } } } \sqrt { k } ( \alpha + 2 \beta ) = \frac { 6 M } { \sqrt { n ^ { \mathrm { v a l } } } } \sqrt { k } \left( \sqrt { \log \left( \frac { 3 \rho _ { A } R L } { M } \right) } + 2 \sqrt { \log 2 } \right)\tag{31}
$$

This lemma provides a deterministic uniform convergence bound. By applying standard Rademacher concentration bounds (Shalev-Shwartz & Ben-David, 2014, Theorem 26.5) combined with McDiarmid’s inequality, the uniform deviation su $\mathsf { p } _ { \phi \in \Phi } | L _ { \mathcal { D } } ( \widehat { W } _ { \phi } ) - L ^ { \mathrm { v a l } } ( \widehat { W } _ { \phi } ) | \leq \epsilon _ { \mathrm { v a l } }$ holds with probability $1 - \delta$ , where:

$$
\epsilon _ { \mathrm { v a l } } = \frac { 1 2 M } { \sqrt { n ^ { \mathrm { v a l } } } } \sqrt { k } \left( \sqrt { \log \left( \frac { 3 \rho _ { A } R L } { M } \right) } + 2 \sqrt { \log 2 } \right) + M \sqrt { \frac { 2 \log ( 2 / \delta ) } { n ^ { \mathrm { v a l } } } }\tag{32}
$$

## F.4 WARM-UP: CONTINUOUS ORACLE INEQUALITY FOR REGULARIZED ERM

In this subsection, we formalize the generalization guarantees for Regularized Empirical Risk Minimization (ERM), where the generic continuous tuning parameter is instantiated as the $L _ { 2 }$ regularization coefficient (i.e., we set $\phi = \lambda$ with $k = 1 )$ . In this setting, the lower level trains a regularized model $\widehat { W } _ { \lambda }$ on a training set $S _ { \mathrm { t r a i n } } = \{ z _ { 1 } ^ { \mathrm { t r a i n } } , \dots , z _ { n ^ { \mathrm { t r } } } ^ { \mathrm { t r a i n } } \}$ drawn i.i.d. from a single data distribution $\mathcal { D } \colon$

$$
\widehat { W } _ { \lambda } = \arg \operatorname* { m i n } _ { W } \left( \frac { 1 } { n ^ { \mathrm { t r } } } \sum _ { i = 1 } ^ { n ^ { \mathrm { t r } } } \ell ( W , z _ { i } ^ { \mathrm { t r a i n } } ) + \Omega _ { \lambda } ( W ) \right)\tag{33}
$$

The upper level evaluates this continuous hypothesis class $\mathcal { H } _ { \Lambda } = \{ \widehat { W } _ { \lambda } : \lambda \in \Lambda \}$ on an independent validation set $S _ { \mathrm { v a l } } = \{ z _ { 1 } ^ { \mathrm { v a l } } , \dots , z _ { n ^ { \mathrm { v a l } } } ^ { \mathrm { v a l } } \}$ also drawn from D:

$$
\hat { \lambda } = \arg \operatorname* { m i n } _ { \lambda \in \Lambda } \frac { 1 } { n ^ { \mathrm { v a l } } } \sum _ { j = 1 } ^ { n ^ { \mathrm { v a l } } } \ell ( \widehat { W } _ { \lambda } , z _ { j } ^ { \mathrm { v a l } } )\tag{34}
$$

This serves as the foundational continuous oracle inequality, clearly distinct from the worst-case Group DRO formulation analyzed subsequently.

The following theorem combines the lower-level high-probability generalization bound for a fixed λ with the uniform convergence bound over the 1-dimensional continuous hypothesis class $\mathcal { H } _ { \Lambda }$

Theorem F.3 (Continuous Oracle Inequality for ERM). Suppose Assumptions 1–4 hold: the loss ℓ is L-Lipschitz and bounded by M, the regularization $\Omega _ { \lambda }$ is λ-strongly convex, the continuous tuning interval Λ has length R, and the lower-level mapping $\mathcal { A } ( \lambda ) = \widehat { W } _ { \lambda }$ is $\rho _ { \mathcal { A } }$ -Lipschitz. Let $W ^ { * }$ be an arbitrary reference predictor (e.g., the population risk minimizer). Let $\hat { \lambda } \in \Lambda$ be the hyperparameter chosen by minimizing the validation risk over $\mathcal { H } _ { \Lambda }$ . With probability at least $1 - \delta$ over the random draw ofboth $S _ { t r a i n }$ and $S _ { \nu a l } ,$ , the true risk $L _ { \mathcal { D } } ( \widehat { W } _ { \widehat { \lambda } } ) = \mathbb { E } _ { z \sim \mathcal { D } } [ \ell ( \widehat { W } _ { \widehat { \lambda } } , z ) ]$ satisfies:

$$
\begin{array} { r } { L _ { \mathcal { D } } ( \widehat { W } _ { \widehat { \lambda } } ) \leq L _ { \mathcal { D } } ( W ^ { * } ) + \displaystyle \operatorname* { m i n } _ { \lambda \in \Lambda } \left( \Omega _ { \lambda } ( W ^ { * } ) + \displaystyle \frac { 2 L ^ { 2 } } { \lambda n ^ { \mathrm { t r } } } + \left( \frac { 4 L ^ { 2 } } { \lambda n ^ { \mathrm { t r } } } + \frac { 4 M } { n ^ { \mathrm { t r } } } \right) \sqrt { \displaystyle \frac { n ^ { \mathrm { t r } } \log ( 4 / \delta ) } { 2 } } \right) } \\ { + \displaystyle \frac { 2 4 M } { \sqrt { n ^ { \mathrm { v a l } } } } \left( \sqrt { \log \left( \frac { 3 \rho _ { A } R L } { M } \right) } + 2 \sqrt { \log 2 } \right) + 2 M \sqrt { \frac { 2 \log ( 4 / \delta ) } { n ^ { \mathrm { v a l } } } } } \end{array}
$$

Interpretation of the Continuous Oracle Inequality. In standard Regularized Loss Minimization (as noted in Shalev-Shwartz & Ben-David (2014, Corollary 13.8)), finding the optimal hyperparameter λ requires prior knowledge of the optimal predictor’s norm $\| W ^ { * } \| . \operatorname { I f } \| W ^ { * } \|$ is known, one can analytically set λ to perfectly balance the bias (the $\Omega _ { \lambda } ( W ^ { * } )$ term) and the variance (the stability gap, which scales as $\bar { O } ( 1 / \lambda \bar { m } ) ,$ ), thereby achieving an optimal generalization rate of $O ( 1 / \sqrt { m } )$

However, in practice, the true optimal predictor $W ^ { * }$ and its norm are unknown. The standard approach to circumvent this is Structural Risk Minimization (SRM), where one trains models on a finite, discrete grid of λ values and selects the best one using a validation set. While SRM guarantees learning, it restricts the solution to the predefined grid, introducing discretization error. Furthermore, as established by Shalev-Shwartz & Ben-David (2014, Theorem 11.2), to theoretically guarantee that the validation set does not overfit to any model in the finite grid, one applies the union bound. Plugging the union bound over all |Grid| discrete options into Hoeffding’s inequality incurs a uniform convergence penalty scaling with $O \left( \sqrt { \log ( \left| G r i d \right| ) / n ^ { \mathrm { v a l } } } \right)$ . Consequently, attempting to reduce discretization error by making the grid denser degrades the theoretical generalization guarantee.

Our Continuous Oracle Inequality demonstrates that continuous bilevel tuning effectively acts as an “oracle,” automatically discovering the theoretically optimal bias-variance tradeoff $\mathrm { m i n } _ { \lambda \in \Lambda } ( \Omega _ { \lambda } ( W ^ { * } ) + \epsilon _ { \mathrm { t r a i n } } ( \lambda ) )$ for the unknown reference predictor $W ^ { * }$ Crucially, it achieves this without discretization error, paying only a logarithmic uniform convergence penalty $O \big ( \sqrt { \log ( \rho _ { \var A } R L / M ) / n ^ { \mathrm { v a l } } } \big )$ Conceptually, the inner term $\rho _ { \mathcal { A } } R L / M$ acts as an “effective grid size”—representing the finite number of distinguishable models within the continuous interval— allowing us to bypass discretization error while paying a statistical penalty no worse than a dense discrete grid. Furthermore, this continuous formulation allows us to directly traverse the hyperparameter space using efficient continuous optimization techniques, such avoiding the prohibitive computational cost of repeatedly training independent models associated with standard grid search.

Proof. Given the Lipschitz continuity of the algorithmic mapping established in the assumptions, the proof decomposes the true risk using the uniform convergence of the validation loss (via our general Rademacher bound) and the stability of the lower-level algorithm.

Step 1: Upper-Level Uniform Convergence. By substituting $k = 1$ into the general uniform deviation bound derived in Equation (32), the two-sided uniform deviation $\operatorname* { s u p } _ { \lambda \in \Lambda } | L _ { \mathcal { D } } ( \widehat { W } _ { \lambda } ) -$ $L ^ { \mathrm { v a l } } ( \widehat { W } _ { \lambda } ) \vert \le \epsilon _ { \mathrm { v a l } }$ holds with probability $1 - \delta / 2$ , where:

$$
\epsilon _ { \mathrm { v a l } } = \frac { 1 2 M } { \sqrt { n ^ { \mathrm { v a l } } } } \left( \sqrt { \log \left( \frac { 3 \rho _ { A } R L } { M } \right) } + 2 \sqrt { \log 2 } \right) + M \sqrt { \frac { 2 \log ( 4 / \delta ) } { n ^ { \mathrm { v a l } } } }\tag{35}
$$

Step 2: Final Continuous Oracle Inequality. Let $W ^ { * }$ be any fixed reference predictor (such as the population risk minimizer). We define the optimal regularization parameter for this predictor as $\begin{array} { r } { \lambda ^ { * } = \arg \operatorname* { m i n } _ { \lambda \in \Lambda } \left( \Omega _ { \lambda } ( W ^ { * } ) + \epsilon _ { \mathrm { t r a i n } } ( \lambda ) \right) } \end{array}$ , where $\epsilon _ { \mathrm { t r a i n } } ( \lambda )$ will be defined shortly. Since $\hat { \lambda }$ minimizes the empirical validation risk, we have $L ^ { \mathrm { v a l } } ( \widehat { W } _ { \widehat { \lambda } } ) \leq L ^ { \mathrm { v a l } } ( \widehat { W } _ { \lambda ^ { * } } )$ . Applying the uniform deviation bound (which holds over all $\mathcal { H } _ { \Lambda }$ with probability $1 - \delta / 2 )$ to both $\hat { \lambda }$ and $\lambda ^ { * }$ yields:

$$
\begin{array} { r l } & { L _ { \mathcal { D } } ( \widehat { W } _ { \widehat { \lambda } } ) \leq L ^ { \mathrm { v a l } } ( \widehat { W } _ { \widehat { \lambda } } ) + \epsilon _ { \mathrm { v a l } } } \\ & { \qquad \leq L ^ { \mathrm { v a l } } ( \widehat { W } _ { \lambda ^ { * } } ) + \epsilon _ { \mathrm { v a l } } } \\ & { \qquad \leq L _ { \mathcal { D } } ( \widehat { W } _ { \lambda ^ { * } } ) + 2 \epsilon _ { \mathrm { v a l } } } \end{array}
$$

To bound $L _ { \mathcal { D } } ( \widehat { W } _ { \lambda ^ { * } } )$ , we compare it against the fixed $W ^ { * }$ . By the empirical optimality of $\widehat { W } _ { \lambda ^ { * } }$ ∗ , we have $L ^ { \mathrm { t r } } ( \widehat { W } _ { \lambda ^ { * } } ) + \Omega _ { \lambda ^ { * } } ( \widehat { W } _ { \lambda ^ { * } } ) \leq L ^ { \mathrm { t r } } ( W ^ { * } ) + \Omega _ { \lambda ^ { * } } ( W ^ { * } )$ . Because $\Omega _ { \lambda } \geq 0$ , we can decompose the true risk as:

$$
\begin{array} { r l } & { L _ { \mathcal { D } } ( \widehat { W } _ { \lambda ^ { * } } ) = L ^ { \mathrm { t r } } ( \widehat { W } _ { \lambda ^ { * } } ) + \Big ( L _ { \mathcal { D } } ( \widehat { W } _ { \lambda ^ { * } } ) - L ^ { \mathrm { t r } } ( \widehat { W } _ { \lambda ^ { * } } ) \Big ) } \\ & { \qquad \leq L ^ { \mathrm { t r } } ( W ^ { * } ) + \Omega _ { \lambda ^ { * } } ( W ^ { * } ) + \Big ( L _ { \mathcal { D } } ( \widehat { W } _ { \lambda ^ { * } } ) - L ^ { \mathrm { t r } } ( \widehat { W } _ { \lambda ^ { * } } ) \Big ) } \\ & { \qquad = L _ { \mathcal { D } } ( W ^ { * } ) + \Omega _ { \lambda ^ { * } } ( W ^ { * } ) + \underbrace { \Big ( L _ { \mathcal { D } } ( \widehat { W } _ { \lambda ^ { * } } ) - L ^ { \mathrm { t r } } ( \widehat { W } _ { \lambda ^ { * } } ) \Big ) } _ { \mathrm { S u b i l i t y ~ g a p } } + \underbrace { \Big ( L ^ { \mathrm { t r } } ( W ^ { * } ) - L _ { \mathcal { D } } ( W ^ { * } ) \Big ) } _ { \mathrm { H o e f f f i d i n g ~ g a p } } } \end{array}
$$

By Lemma F.15 with probability $1 - \delta / 4$ , the stability gap is bounded by:

$$
\frac { 2 L ^ { 2 } } { \lambda ^ { * } n ^ { \mathrm { t r } } } + \left( \frac { 4 L ^ { 2 } } { \lambda ^ { * } n ^ { \mathrm { t r } } } + \frac { 2 M } { n ^ { \mathrm { t r } } } \right) \sqrt { \frac { n ^ { \mathrm { t r } } \log ( 4 / \delta ) } { 2 } }
$$

Simultaneously, since $W ^ { * }$ is fixed independent of $S _ { \mathrm { t r a i n } }$ , we apply Hoeffding’s inequality (Shalev-Shwartz & Ben-David, 2014, Lemma $\bar { \mathbf { B } } . 6 )$ . Because the absolute loss is bounded by M, the loss variables fall in a range of 2M. For a one-sided bound with local confidence $\delta _ { \mathrm { l o c a l } } = \delta / 4$ , Hoeffding’s inequality exactly bounds the second gap by $\begin{array} { r } { M \sqrt { \frac { 2 \log ( 4 / \delta ) } { n ^ { \mathrm { t r } } } } = \frac { 2 M } { n ^ { \mathrm { t r } } } \sqrt { \frac { n ^ { \mathrm { t r } } \log ( 4 / \delta ) } { 2 } } } \end{array}$ with probability $1 - \delta / 4$ . Summing these bounds gives:

$$
L _ { \mathcal { D } } ( \widehat { W } _ { \lambda ^ { * } } ) \leq L _ { \mathcal { D } } ( W ^ { * } ) + \Omega _ { \lambda ^ { * } } ( W ^ { * } ) + \underbrace { \frac { 2 L ^ { 2 } } { \lambda ^ { * } n ^ { \mathrm { t r } } } + \left( \frac { 4 L ^ { 2 } } { \lambda ^ { * } n ^ { \mathrm { t r } } } + \frac { 4 M } { n ^ { \mathrm { t r } } } \right) \sqrt { \frac { n ^ { \mathrm { t r } } \log ( 4 / \delta ) } { 2 } } } _ { : = \epsilon _ { \mathrm { t r a i n } } ( \lambda ^ { * } ) }\tag{36}
$$

By the definition of $\lambda ^ { * }$ , this is exactly $\begin{array} { r } { L _ { \mathcal { D } } ( W ^ { * } ) + \operatorname* { m i n } _ { \lambda \in \Lambda } \left( \Omega _ { \lambda } ( W ^ { * } ) + \epsilon _ { \mathrm { t r a i n } } ( \lambda ) \right) } \end{array}$ . Substituting this bound directly into $L _ { \mathcal { D } } ( \widehat { W } _ { \widehat { \lambda } } ) \leq L _ { \mathcal { D } } ( \widehat { W } _ { \lambda ^ { * } } ) + 2 \epsilon _ { \mathrm { v a l } }$ , and applying the union bound over all three events (total probability $1 - \delta )$ , we obtain the Continuous Oracle Inequality. Expanding $\epsilon _ { \mathrm { t r a i n } } ( \lambda )$ and $2 \epsilon _ { \mathrm { v a l } }$ precisely matches the theorem statement, completing the proof. □

## F.5 CONTINUOUS ORACLE INEQUALITY FOR GROUP DRO

While Theorem F.3 outlines the oracle inequality for standard ERM over a unified dataset, applying this continuous generalization bound to the Group DRO formulation requires substituting the sample complexities. In this setting, the generic continuous tuning parameter is instantiated as the robustness penalty $( \mathrm { i . e . }$ , we set $\phi = \eta$ with $k = 1 )$ , while the $L _ { 2 }$ regularization coefficient λ is held fixed strictly to satisfy the required strong convexity for algorithmic stability. Because DRO is evaluated on a worst-case basis at both levels, uniform stability in the lower level and uniform convergence in the upper level are strictly bottlenecked by the most scarcely represented groups.

Lemma F.4 (Modified Uniform Stability of Group DRO). Let $\begin{array} { r l } { J _ { \eta } ( W ; S ) } & { { } = } \end{array}$ $\begin{array} { r } { \operatorname* { m a x } _ { q \in \Delta _ { G } } [ \sum _ { g = 1 } ^ { G } q _ { g } L _ { g } ^ { \mathrm { t r } } ( W ) - \frac { \eta } { 2 } \| q - \frac { 1 } { G } { \bf 1 } \| ^ { 2 } + \frac { \lambda } { 2 } \| W \| ^ { 2 } ] } \end{array}$ be the lower-level Group DRO ob jective. Under the assumptions of bounded and Lipschitz continuous group losses (Assumption 1), the lower-level Group DRO algorithm $\mathbf { \mathcal { A } } ( S ) \mathbf { \mathcal { = } } \arg$ min<sub>W</sub> $J _ { \eta } ( W ; S )$ is uniformly stable with modified constant:

$$
\beta _ { D R O } = { \frac { 2 L ^ { 2 } } { \lambda \operatorname* { m i n } _ { g } n _ { g } ^ { \mathrm { t r } } } } \left( 1 + { \frac { M { \sqrt { G } } } { \eta _ { \mathrm { m i n } } } } \right)\tag{37}
$$

where mi $\mathrm { l } _ { g } n _ { g } ^ { \mathrm { t r } }$ is the size of the smallest group in the training set $S .$

Proof. This proof follows the same standard gradient-based strong convexity argument as the standard stability result (Lemma F.14), but must account for the coupled min-max optimization over all groups. When the training set S is perturbed by changing exactly one example z into $z ^ { \prime }$ , this perturbation occurs in exactly one group, say group k. Because each instantaneous loss is bounded by M, the empirical loss of group k changes by at most $\begin{array} { r } { \frac { 1 } { n _ { k } ^ { \mathrm { t r } } } | \ell ( W , z ) - \ell ( W , z ^ { \prime } ) | \leq \frac { 2 M } { \operatorname* { m i n } _ { g } n _ { g } ^ { \mathrm { t r } } } } \end{array}$

To bound the uniform stability, we evaluate how much the gradient of $J _ { \eta }$ shifts due to this perturbation. By Danskin’s theorem, the gradient is exactly $\begin{array} { r } { \nabla J _ { \eta } ( W ; S ) = \sum _ { g = 1 } ^ { G } q _ { g } \nabla L _ { g } ^ { \mathrm { t r } } ( W ) + \lambda W } \end{array}$ , where $\begin{array} { r } { q = \Pi _ { \Delta _ { G } } \left( \frac { 1 } { \eta } L ^ { \mathrm { t r } } ( W ) + \frac { 1 } { G } \mathbf { 1 } \right) } \end{array}$ are the optimal simplex weights. Let $S ^ { \prime }$ be the perturbed dataset with corresponding optimal weights $q ^ { \prime }$ . The shift in the gradient of the loss term is bounded by:

$$
\begin{array} { r l } & { \displaystyle \left\| \displaystyle \sum _ { g = 1 } ^ { G } q _ { g } \nabla L _ { g } ^ { \mathrm { t r } } ( S ) - \displaystyle \sum _ { g = 1 } ^ { G } q _ { g } ^ { \prime } \nabla L _ { g } ^ { \mathrm { t r } } ( S ^ { \prime } ) \right\| } \\ & { \displaystyle \leq \underbrace { \left\| \sum _ { g = 1 } ^ { G } q _ { g } \left( \nabla L _ { g } ^ { \mathrm { t r } } ( S ) - \nabla L _ { g } ^ { \mathrm { t r } } ( S ^ { \prime } ) \right) \right\| } _ { \mathrm { D i r e c t ~ g r a d i e n t ~ s h i f t } } + \underbrace { \left\| \sum _ { g = 1 } ^ { G } ( q _ { g } - q _ { g } ^ { \prime } ) \nabla L _ { g } ^ { \mathrm { t r } } ( S ^ { \prime } ) \right\| } _ { \mathrm { W e i g h t ~ p e r u r b a t i o n ~ s h i f t } } } \end{array}
$$

For the first term, only group $k \mathrm { { : } }$ empirical gradient changes (by at most $\frac { 2 L } { \operatorname* { m i n } _ { g } n _ { g } ^ { \mathrm { t r } } } )$ . Since $q _ { k } \leq 1$ this direct shift is bounded by $\frac { 2 L } { \operatorname* { m i n } _ { g } n _ { g } ^ { \mathrm { t r } } }$ . For the second term, the simplex projection is 1-Lipschitz, so the shift in optimal weights is bounded by the scaled shift in the group losses: $\| q - q ^ { \prime } \| \leq$ $\begin{array} { r } { \frac { 1 } { \eta } \| L ^ { \mathrm { t r } } ( S ) - L ^ { \mathrm { t r } } ( \bar { S } ^ { \prime } ) \| \le \frac { 1 } { \eta _ { \mathrm { m i n } } } \frac { 2 M } { \operatorname* { m i n } _ { g } n _ { q } ^ { \mathrm { t r } } } } \end{array}$ . Multiplying by the Jacobian norm of the group losses $( \sqrt { G } L )$ the second term is bounded by $\frac { 2 M L \sqrt { G } } { \eta _ { \mathrm { m i n } } \ : \mathrm { m i n } _ { g } \ : n _ { g } ^ { \mathrm { t r } } }$

Summing these, the entire gradient shifts by at most $\begin{array} { r } { \frac { 2 L } { \operatorname* { m i n } _ { g } n _ { g } ^ { \mathrm { t r } } } \left( 1 + \frac { M \sqrt { G } } { \eta _ { \mathrm { m i n } } } \right) : = \Delta } \end{array}$ . Because $J _ { \eta }$ is λ-strongly convex, applying the exact same strong monotonicity displacement argument from Equation (43) in Lemma F.14 guarantees $\begin{array} { r } { \| \widehat { W } _ { \eta } ( S ) - \widehat { W } _ { \eta } ( S ^ { \prime } ) \| \le \frac { \Delta } { \lambda } } \end{array}$ . Multiplying by the L-Lipschitz constant of the worst-group loss yields the final modified uniform stability constant β<sub>DRO</sub>. □

Theorem F.5 (Group DRO Continuous Oracle Inequality). Let min<sub>g</sub> n<sup>tr</sup> and $\operatorname* { m i n } _ { g } n _ { g } ^ { \mathrm { v a l } }$ denote the sizes ofthe smallest groups in the training and validation sets, respectively, where both sets consist of G groups. For any reference predictor $\Breve { W } ^ { * }$ , let $\begin{array} { r } { \Omega _ { \eta } ( W ^ { \ast } ) = \frac { \lambda } { 2 } \| W ^ { \ast } \| ^ { 2 } + \frac { \dot { \eta } } { 2 } } \end{array}$ be the approximation bias penalty. Let $\begin{array} { r } { \widehat { \eta } = \arg \operatorname* { m i n } _ { \eta \in H } f _ { \nu a l } ( \widehat { W } _ { \eta } ) } \end{array}$ be the hyperparameter chosen by minimizing the empirical validation worst-group risk $f _ { \nu a l }$ over the continuous hypothesis class H<sub>H</sub>, where $H = [ \eta _ { \mathrm { m i n } } , \eta _ { \mathrm { m a x } } ] .$ Let $\begin{array} { r } { \beta _ { D R O } = \frac { 2 L ^ { 2 } } { \lambda \operatorname* { m i n } _ { g } n _ { g } ^ { \mathrm { t r } } } \left( 1 + \frac { M \sqrt { G } } { \eta _ { \mathrm { m i n } } } \right) } \end{array}$ be the modified uniform stability. With probability at least $1 - \delta ,$ the true worst-group risk $L _ { \mathcal { D } } ^ { w o r s t } ( \widehat { W } _ { \hat { \eta } } ) = \operatorname* { m a x } _ { g } L _ { \mathcal { D } , g } ( \widehat { W } _ { \hat { \eta } } )$ satisfies:

$$
\begin{array} { r l } & { L _ { \mathcal { D } } ^ { w o r s t } ( \widehat { W } _ { \widehat { \eta } } ) \leq L _ { \mathcal { D } } ^ { w o r s t } ( W ^ { * } ) + M \displaystyle \sqrt { \frac { 2 \log \left( 4 G / \delta \right) } { \operatorname* { m i n } _ { g } n _ { g } ^ { \mathrm { t r } } } } + 2 M \displaystyle \sqrt { \frac { 2 \log \left( 4 G / \delta \right) } { \operatorname* { m i n } _ { g } n _ { g } ^ { \mathrm { v a l } } } } } \\ & { \qquad + \displaystyle \operatorname* { m i n } _ { \eta \in H } \left( \Omega _ { \eta } ( W ^ { * } ) + \beta _ { D R O } + \Sigma \sqrt { \frac { \log \left( 4 G / \delta \right) } { 2 } } \right) } \\ & { \qquad + \displaystyle \frac { 2 4 M } { \sqrt { \operatorname* { m i n } _ { g } n _ { g } ^ { \mathrm { v a l } } } } \left( \sqrt { \log \left( \frac { 3 \rho _ { A } R L } { M } \right) } + 2 \sqrt { \log 2 } \right) } \end{array}
$$

where $\begin{array} { r } { \rho _ { \mathcal { A } } \leq \frac { G L M } { \lambda \eta _ { \sf m i n } ^ { 2 } } } \end{array}$ is the Lipschitz constant of the DRO algorithmic mapping, and $\Sigma ^ { 2 } = 4 n ^ { \mathrm { t r } } \beta _ { D R O } ^ { 2 } +$ $\begin{array} { r } { 8 M \beta _ { D R O } + \frac { 4 \overrightarrow { M } ^ { 2 } } { \operatorname* { m i n } _ { g } n _ { g } ^ { \mathrm { t r } } } } \end{array}$ bounds the stability variance.

Proof. The proof mirrors the three-step structure of Theorem F.3, isolating the exact points where worst-case group bounds modify the complexities.

Step 1: Lower-Level Uniform Stability (via Union Bound). By Lemma F.4, the lower-level Group DRO objective is uniformly stable with the modified constant $\begin{array} { r } { \beta _ { \mathrm { D R O } } = \frac { 2 L ^ { 2 } } { \lambda \operatorname* { m i n } _ { g } n _ { q } ^ { \mathrm { t r } } } \left( 1 + \frac { M \sqrt { G } } { \eta _ { \mathrm { m i n } } } \right) } \end{array}$ . Because the empirical worst-group risk is a maximum over empirical averages, its expected generalization gap is not bounded directly by $\beta _ { \mathrm { D R O } }$ due to Jensen’s inequality $( \mathbf { \bar { m a x } } _ { g } \mathbb { E } [ \cdot ] ^ { - } \leq \mathbb { E } [ \mathbf { \bar { m a x } } _ { g } \cdot ] )$ Instead, we decouple the maximum operator. By the subadditivity of the maximum, the worst-group generalization gap is bounded by the maximum of the individual group generalization gaps:

$$
\begin{array} { r } { L _ { \mathcal { D } } ^ { \mathrm { w o r s t } } ( \widehat { W } _ { \eta } ) - L _ { \mathrm { w o r s } } ^ { \mathrm { t r } } ( \widehat { W } _ { \eta } ) = \displaystyle \operatorname* { m a x } _ { g } L _ { \mathcal { D } , g } ( \widehat { W } _ { \eta } ) - \operatorname* { m a x } _ { g } L _ { g } ^ { \mathrm { t r } } ( \widehat { W } _ { \eta } ) \leq \displaystyle \operatorname* { m a x } _ { g \in [ \mathcal { G } ] } \underbrace { \Big ( L _ { \mathcal { D } , g } ( \widehat { W } _ { \eta } ) - L _ { g } ^ { \mathrm { t r } } ( \widehat { W } _ { \eta } ) \Big ) } _ { : = Z _ { g } ( S ) } } \end{array}
$$

For any specific group $^ { g , }$ the standard expected-loss stability result applies perfectly: $\mathbb { E } _ { S } [ Z _ { g } ( S ) ] \le$ $\beta _ { \mathrm { D R O } }$ . To bound the maximum over all groups with high probability, we rigorously evaluate the sensitivity of $Z _ { g } ( S )$ to a single point perturbation. Suppose we perturb exactly one training example $z _ { i }  z ^ { \prime }$ to form $S ^ { ( i ) }$ . Let $\widehat { W }$ and $\widehat W ^ { ( i ) }$ be the optimal predictors for S and $S ^ { ( i ) }$ , respectively. By the triangle inequality, the sensitivity of the group generalization gap is bounded by:

$$
\begin{array} { r } { | Z _ { g } ( S ) - Z _ { g } ( S ^ { ( i ) } ) | \le \underbrace { \left| L _ { \mathcal { D } , g } ( \widehat { W } ) - L _ { \mathcal { D } , g } ( \widehat { W } ^ { ( i ) } ) \right| } _ { \mathrm { T r u e ~ r i s k ~ s e n s i t i v i t y } } + \underbrace { \left| L _ { g } ^ { \mathrm { t r } } ( \widehat { W } ) - L _ { g } ^ { \mathrm { t r } , ( i ) } ( \widehat { W } ^ { ( i ) } ) \right| } _ { \mathrm { E m p i r i c a l ~ r i s k ~ s e n s i t i v i t y } } } \end{array}
$$

For the true risk sensitivity, the uniform stability property guarantees that the absolute loss difference on any arbitrary point z is deterministically bounded by β<sub>DRO</sub>. Therefore, its expectation over the target group distribution $z \sim \mathcal { D } _ { g }$ is identically bounded: $\begin{array} { r } { \left| L _ { \mathcal { D } , g } ( \widehat { W } ) - L _ { \mathcal { D } , g } ( \widehat { W } ^ { ( i ) } ) \right| \ \leq } \end{array}$ $\begin{array} { r } { \mathbb { E } _ { z \sim \mathcal { D } _ { q } } [ | \ell ( \widehat { W } , z ) - \ell ( \widehat { W } ^ { ( i ) } , z ) | ] \leq \beta _ { \mathrm { D R O } } } \end{array}$ . For the empirical risk sensitivity, the bound depends on whether the perturbed index i belongs to group $g ( i \in G _ { g } ) ;$

1. Case 1 $( i \notin G _ { g } ) { \ : }$ The subset of points belonging to group g is identical between S and $S ^ { ( i ) }$ The empirical risk shifts solely due to the change in the algorithmic output $\widehat { W }$ . Averaging the uniform stability bound over these ${ n } _ { g } ^ { \mathrm { t r } }$ unperturbed points yields an empirical shift of exactly $\beta _ { \mathrm { D R O } }$

2. Case $2 ( i \in G _ { g } ) \colon$ Group $g$ shares $n _ { g } ^ { \mathrm { t r } } - 1$ identical points between the two sets, but one point differs $( z _ { i }  z ^ { \prime } )$ . For the identical points, the loss difference is bounded by $\beta _ { \mathrm { D R O } }$ . For the single swapped point, the loss difference is naively bounded by 2M (Assumption 1). Averaging these yields:

$$
\begin{array} { r l } & { \Big | L _ { g } ^ { \mathrm { t r } } \big ( \widehat W \big ) - L _ { g } ^ { \mathrm { t r } , ( i ) } \big ( \widehat W ^ { ( i ) } \big ) \Big | } \\ & { \le \frac { 1 } { n _ { g } ^ { \mathrm { t r } } } \displaystyle \sum _ { \substack { j \in G _ { g } , j \neq i } } \underbrace { \vert \ell ( \widehat W , z _ { j } ) - \ell ( \widehat W ^ { ( i ) } , z _ { j } ) \vert } _ { \le \beta _ { \mathrm { p h o } } } + \frac { 1 } { n _ { g } ^ { \mathrm { t r } } } \underbrace { \vert \ell ( \widehat W , z _ { i } ) - \ell ( \widehat W ^ { ( i ) } , z ^ { \prime } ) \vert } _ { \le 2 M } } \\ & { \le \frac { n _ { g } ^ { \mathrm { t r } } - 1 } { n _ { g } ^ { \mathrm { t r } } } \beta _ { \mathrm { D R O } } + \frac { 2 M } { n _ { g } ^ { \mathrm { t r } } } \le \beta _ { \mathrm { D R O } } + \frac { 2 M } { n _ { g } ^ { \mathrm { t r } } } } \end{array}
$$

Summing the true and empirical sensitivities, the bounded difference constants $c _ { i }$ for $Z _ { g } ( S )$ satisfy $\begin{array} { r } { c _ { i } \le 2 \bar { \beta _ { \mathrm { D R O } } } + \frac { 2 M } { n _ { q } ^ { \mathrm { t r } } } } \end{array}$ for $i \in G _ { g }$ , and $c _ { i } \leq 2 \beta _ { \mathrm { D R O } }$ for $i \not \in G _ { g }$ . The sum of squared differences over all $n ^ { \mathrm { t r } }$ independent examples is bounded by:

$$
\begin{array} { r l } { \displaystyle \sum _ { i = 1 } ^ { n ^ { \mathrm { t r } } } c _ { i } ^ { 2 } = \sum _ { i \notin G _ { g } } \left( 2 \beta _ { \mathrm { D R O } } \right) ^ { 2 } + \sum _ { i \in G _ { g } } \left( 2 \beta _ { \mathrm { D R O } } + \frac { 2 M } { n _ { g } ^ { \mathrm { t r } } } \right) ^ { 2 } } \\ { \displaystyle } & { ~ = n ^ { \mathrm { t r } } ( 2 \beta _ { \mathrm { D R O } } ) ^ { 2 } + n _ { g } ^ { \mathrm { t r } } \cdot 2 ( 2 \beta _ { \mathrm { D R O } } ) \frac { 2 M } { n _ { g } ^ { \mathrm { t r } } } + n _ { g } ^ { \mathrm { t r } } \frac { 4 M ^ { 2 } } { ( n _ { g } ^ { \mathrm { t r } } ) ^ { 2 } } \le 4 n ^ { \mathrm { t r } } \beta _ { \mathrm { D R O } } ^ { 2 } + 8 M \beta _ { \mathrm { D R O } } + \frac { 4 M ^ { 2 } } { \operatorname* { m i n } _ { g } n _ { g } ^ { \mathrm { t r } } } } \\ { \displaystyle } & { : = \Sigma ^ { 2 } } \end{array}
$$

Applying the one-sided McDiarmid’s inequality (Lemma F.16) to bound the deviation of $Z _ { g } ( S )$ above its expectation, and noting that the expected generalization gap is bounded by uniform stability $\mathbb { E } _ { S } [ Z _ { g } ( S ) ] \overset { * } { \leq } \beta _ { \mathrm { D R O } }$ , we obtain that with probability at least $\textstyle 1 - { \frac { - \delta } { 4 G } } \colon$

$$
Z _ { g } ( S ) \leq \beta _ { \mathrm { D R O } } + \Sigma \sqrt { \frac { \log ( 4 G / \delta ) } { 2 } }
$$

Applying a union bound over all $G$ groups ensures that this bound holds simultaneously for the maximum ma $\mathrm { x } _ { g } Z _ { g } ( S )$ with total failure probability $\delta / 4$

$$
L _ { \mathcal { D } } ^ { \mathrm { w o r s t } } ( \widehat { W } _ { \eta } ) \leq L _ { \mathrm { w o r s t } } ^ { \mathrm { t r } } ( \widehat { W } _ { \eta } ) + \beta _ { \mathsf { D R O } } + \Sigma \sqrt { \frac { \log ( 4 G / \delta ) } { 2 } }\tag{38}
$$

Step 2: Upper-Level Uniform Convergence Decomposition. The upper-level empirical objective is the worst-group validation loss $\bar { f _ { \mathrm { v a l } } } ( W ) = \bar { \operatorname * { m a x } _ { g \in [ G ] } } L _ { g } ^ { \mathrm { v a l } } ( W )$ , and the target population risk is $L _ { \mathcal { D } } ^ { \mathrm { w o r s t } } ( W ) = \operatorname* { m a x } _ { g \in [ G ] } L _ { \mathcal { D } , g } ( W )$ . By the non-expansive property of the maximum operator (Lemma F.19), the uniform deviation over the continuous path $\mathcal { H } _ { H }$ cleanly decomposes. Furthermore, because the supremum and finite maximum operators commute, we can isolate the supremum to each individual group:

$$
\begin{array} { r l } { \displaystyle \operatorname* { s u p } _ { \eta \in { \cal H } } \left. f _ { \mathrm { v a l } } ( \widehat W _ { \eta } ) - L _ { \mathcal { D } } ^ { \mathrm { w o r s t } } ( \widehat W _ { \eta } ) \right. = \displaystyle \operatorname* { s u p } _ { \eta \in { \cal H } } \left. \operatorname* { m a x } _ { g \in [ G ] } L _ { g } ^ { \mathrm { v a l } } ( \widehat W _ { \eta } ) - \operatorname* { m a x } _ { g \in [ G ] } L _ { \mathcal { D } , g } ( \widehat W _ { \eta } ) \right. } & { } \\ { \displaystyle \leq \operatorname* { s u p } _ { \eta \in { \cal H } } \operatorname* { m a x } _ { g \in [ G ] } \left. L _ { g } ^ { \mathrm { v a l } } ( \widehat W _ { \eta } ) - L _ { \mathcal { D } , g } ( \widehat W _ { \eta } ) \right. } & { } \\ { \displaystyle } & { = \operatorname* { m a x } _ { g \in [ G ] } \operatorname* { s u p } _ { \eta \in { \cal H } } \left. L _ { g } ^ { \mathrm { v a l } } ( \widehat W _ { \eta } ) - L _ { \mathcal { D } , g } ( \widehat W _ { \eta } ) \right. } \end{array}
$$

This isolates the continuous uniform convergence problem entirely into G independent group-wise continuous uniform convergence bounds.

Step 3: Rademacher Complexity and Union Bound. For any individual group $^ { g , }$ bounding the continuous 1-dimensional path $\mathcal { H } _ { H }$ relies on the Rademacher complexity of the independent subset $S _ { \mathrm { v a l } , g }$ of size $n _ { q } ^ { \mathrm { v a l } }$ . Following exactly the uniform deviation derivation in Lemma F.2 (Equation (32) with $k = 1 )$ , we evaluate the continuous uniform deviation bound for group $g .$ To ensure this bound holds simultaneously across all $G$ groups with a total failure probability of $\delta / 2 ,$ we apply a discrete union bound allocating confidence $\delta / \bar { ( 2 G ) }$ to each group. As a result, for all $g \in [ G ]$ with probability $1 - \delta / 2 \cdot$

$$
\operatorname* { s u p } _ { \eta \in H } \left| L _ { g } ^ { \mathrm { v a l } } ( \widehat { W } _ { \eta } ) - L _ { \mathcal { D } , g } ( \widehat { W } _ { \eta } ) \right| \leq \varprojlim _ { g } \left( \sqrt { \log \left( \frac { 3 \rho _ { A } R L } { M } \right) } + 2 \sqrt { \log 2 } \right) + M \sqrt { \frac { 2 \log ( 4 G / \delta ) } { n _ { g } ^ { \mathrm { v a l } } } }
$$

Taking the maximum over all groups conservatively bounds the overall uniform deviation by the worst-case group size min $, n _ { g } ^ { \mathrm { v a \overline { { l } } } }$ :

$$
\operatorname* { m a x } _ { g \in [ G ] } \operatorname* { s u p } _ { \eta \in H } \left| L _ { g } ^ { \mathrm { v a l } } ( \widehat { W } _ { \eta } ) - L _ { \mathcal { D } , g } ( \widehat { W } _ { \eta } ) \right| \leq \operatorname* { m a x } _ { g \in [ G ] } \epsilon _ { \mathrm { v a l } , g } : = \epsilon _ { \mathrm { v a l } } ^ { \mathrm { w o r s t } }
$$

Step 4: Final Decomposition. Let $\epsilon _ { \mathrm { t r a i n } } ( \eta ) = \beta _ { \mathrm { D R O } } + \Sigma \sqrt { \frac { \log ( 4 / \delta ) } { 2 } }$ denote the lower-level stability gap bound from Step 1. We define η<sup>∗</sup> = arg min $\ L _ { \eta \in H } \left( \Omega _ { \eta } ( W ^ { * } ) + \epsilon _ { \mathrm { t r a i n } } ( \eta ) \right)$ as the optimal hyperparameter for the reference predictor $W ^ { * }$ . Because $\begin{array} { r } { \widehat { \eta } \ = \ \arg \operatorname* { m i n } _ { \eta \in H } f _ { \mathrm { v a l } } ( \widehat { W } _ { \eta } ) } \end{array}$ minimizes the empirical worst-group validation loss $f _ { \mathrm { v a l } }$ , we have $f _ { \mathrm { v a l } } ( \widehat { W } _ { \widehat { \eta } } ) \leq f _ { \mathrm { v a l } } ( \widehat { W } _ { \eta ^ { * } } )$ . Recall that $f _ { \mathrm { v a l } } ( W ) = \operatorname* { m a x } _ { g } L _ { g } ^ { \mathrm { v a l } } ( W )$ . Applying the uniform deviation bound (which holds over all $\eta \in H$ with probability $1 - \delta / 2 )$ to both ηˆ and $\eta ^ { * }$ yields:

$$
\begin{array} { r l } { L _ { \mathcal { D } } ^ { \mathrm { w o r s t } } ( \widehat { W } _ { \widehat { \eta } } ) \leq f _ { \mathrm { v a l } } ( \widehat { W } _ { \widehat { \eta } } ) + \epsilon _ { \mathrm { v a l } } ^ { \mathrm { w o r s t } } } & { } \\ { \leq f _ { \mathrm { v a l } } ( \widehat { W } _ { \eta ^ { * } } ) + \epsilon _ { \mathrm { v a l } } ^ { \mathrm { w o r s t } } } & { } \\ { \leq L _ { \mathcal { D } } ^ { \mathrm { w o r s t } } ( \widehat { W } _ { \eta ^ { * } } ) + 2 \epsilon _ { \mathrm { v a l } } ^ { \mathrm { w o r s t } } } \end{array}
$$

To bound the target $L _ { \mathcal { D } } ^ { \mathrm { w o r s t } } ( \widehat { W } _ { \eta ^ { * } } )$ , we decompose the true risk via the empirical training risks. Crucially, while the algorithm optimizes the 3-term Group DRO objective $J _ { \eta } ( W )$ , defined as:

$$
J _ { \eta } ( W ) = \operatorname* { m a x } _ { q \in \Delta _ { G } } \left[ \sum _ { g = 1 } ^ { G } q _ { g } L _ { g } ^ { \mathrm { t r } } ( W ) - \frac { \eta } { 2 } \left\| q - \frac { 1 } { G } \mathbf { 1 } \right\| ^ { 2 } \right] + \frac { \lambda } { 2 } \| W \| ^ { 2 }
$$

we can rigorously relate its minimizer back to the pure unregularized worst-group loss by bounding the $\begin{array} { r } { - \frac { \eta } { 2 } \lVert q - \frac { 1 } { G } \mathbf { 1 } \rVert ^ { 2 } } \end{array}$ penalty term. Let $\begin{array} { r } { \Omega _ { \eta ^ { * } } ( W ^ { * } ) = \frac \lambda 2 \| W ^ { * } \| ^ { 2 } + \frac { \eta ^ { * } } { 2 } } \end{array}$ encompass the deterministic penalties. By the optimality of $\widehat { W } _ { \eta ^ { * } }$ on $J _ { \eta ^ { * } }$ , we have $J _ { \eta ^ { * } } ( \widehat { W } _ { \eta ^ { * } } ) \leq J _ { \eta ^ { * } } ( W ^ { * } )$

For the left side, the maximum over $q \in \Delta _ { G }$ is lower bounded by evaluating it at the specific onehot vector $q = \mathbf { e } _ { k }$ corresponding to the worst group k = arg max<sub>g</sub> $L _ { q } ^ { \mathrm { t r } } ( \widehat { W } _ { \eta ^ { * } } )$ . The η-penalty for this one-hot vector evaluates to exactly $\begin{array} { r } { \frac { \eta ^ { * } } { 2 } \| \mathbf { e } _ { k } - \frac { 1 } { G } \mathbf { 1 } \| ^ { 2 } = \frac { \eta ^ { * } } { 2 } ( 1 - \frac { 1 } { G } ) \overset {  } { \le } \frac { \eta ^ { * } } { 2 } } \end{array}$ . Thus, retaining the non-negative regularizer $\begin{array} { r } { \frac \lambda 2 \| \widehat { W } _ { \eta ^ { * } } \| ^ { 2 } \geq 0 } \end{array}$ , we have:

$$
J _ { \eta ^ { * } } ( \widehat { W } _ { \eta ^ { * } } ) \geq L _ { \mathrm { w o r s t } } ^ { \mathrm { t r } } ( \widehat { W } _ { \eta ^ { * } } ) - \frac { \eta ^ { * } } { 2 } + \frac { \lambda } { 2 } \| \widehat { W } _ { \eta ^ { * } } \| ^ { 2 } \geq L _ { \mathrm { w o r s t } } ^ { \mathrm { t r } } ( \widehat { W } _ { \eta ^ { * } } ) - \frac { \eta ^ { * } } { 2 }
$$

For the right side, because the η-penalty term $\begin{array} { r l } {  { - \frac { \eta ^ { * } } { 2 } \| q - \frac { 1 } { G } \mathbf { 1 } \| ^ { 2 } } } \end{array}$ is strictly non-positive for any $q ,$ we can trivially upper bound the inner maximum by dropping the penalty entirely:

$$
J _ { \eta ^ { * } } ( W ^ { * } ) \leq \operatorname* { m a x } _ { q \in \Delta _ { G } } \left[ \sum _ { g } q _ { g } L _ { g } ^ { \mathrm { t r } } ( W ^ { * } ) \right] + \frac { \lambda } { 2 } \| W ^ { * } \| ^ { 2 } = L _ { \mathrm { w o r s t } } ^ { \mathrm { t r } } ( W ^ { * } ) + \frac { \lambda } { 2 } \| W ^ { * } \| ^ { 2 }
$$

Chaining these two inequalities $\begin{array} { r } { ( L _ { \mathrm { w o r s t } } ^ { \mathrm { t r } } ( \widehat { W } _ { \eta ^ { * } } ) - \frac { \eta ^ { * } } { 2 } \leq J _ { \eta ^ { * } } ( \widehat { W } _ { \eta ^ { * } } ) \leq J _ { \eta ^ { * } } ( W ^ { * } ) \leq L _ { \mathrm { w o r s t } } ^ { \mathrm { t r } } ( W ^ { * } ) + } \end{array}$ $\frac { \lambda } { 2 } \parallel W ^ { * } \parallel ^ { 2 } )$ securely isolates the empirical risks:

$$
L _ { \mathrm { w o r s t } } ^ { \mathrm { t r } } ( \widehat { W } _ { \eta ^ { * } } ) \leq L _ { \mathrm { w o r s t } } ^ { \mathrm { t r } } ( W ^ { * } ) + \Omega _ { \eta ^ { * } } ( W ^ { * } )
$$

We can now algebraically inject this upper bound into the true risk:

$$
\begin{array} { r l } & { L _ { \mathcal { D } } ^ { \mathrm { w o r s t } } ( \widehat { W } _ { \eta ^ { * } } ) = L _ { \mathrm { w o r s t } } ^ { \mathrm { t r } } ( \widehat { W } _ { \eta ^ { * } } ) + \Big ( L _ { \mathcal { D } } ^ { \mathrm { w o r s t } } ( \widehat { W } _ { \eta ^ { * } } ) - L _ { \mathrm { w o r s t } } ^ { \mathrm { t r } } ( \widehat { W } _ { \eta ^ { * } } ) \Big ) } \\ & { \qquad \leq L _ { \mathrm { w o r s t } } ^ { \mathrm { t r } } ( W ^ { * } ) + \Omega _ { \eta ^ { * } } ( W ^ { * } ) + \underbrace { \Big ( L _ { \mathcal { D } } ^ { \mathrm { w o r s t } } ( \widehat { W } _ { \eta ^ { * } } ) - L _ { \mathrm { w o r s t } } ^ { \mathrm { t r } } ( \widehat { W } _ { \eta ^ { * } } ) \Big ) } _ { \mathrm { s t a b i l i t y ~ g a p ~ } } } \\ & { \qquad = L _ { \mathcal { D } } ^ { \mathrm { w o r s t } } ( W ^ { * } ) + \Omega _ { \eta ^ { * } } ( W ^ { * } ) + \mathrm { S t a b i l i t y ~ g a p } + \underbrace { \Big ( L _ { \mathrm { w o r s t } } ^ { \mathrm { t r } } ( W ^ { * } ) - L _ { \mathcal { D } } ^ { \mathrm { w o r s t } } ( W ^ { * } ) \Big ) } _ { \mathrm { H o e f f d i n g ~ g a p } } } \end{array}
$$

Simultaneously, since $W ^ { * }$ is fixed independent of $S _ { \mathrm { t r a i n } }$ , we bound the one-sided deviation of its pure unregularized worst-group training loss. For any group $g \in \left[ G \right]$ , the one-sided Hoeffding’s inequality bounds the deviation $L _ { q } ^ { \mathrm { t r } } ( W ^ { * } ) - L _ { \mathcal { D } , g } ( W ^ { * } )$ . Applying a union bound over all $G$ training groups with total confidence $\delta / 4$ yields the Hoeffding gap:

$$
L _ { \mathrm { w o r s t } } ^ { \mathrm { t r } } ( W ^ { * } ) - L _ { \mathcal { D } } ^ { \mathrm { w o r s t } } ( W ^ { * } ) \leq \operatorname* { m a x } _ { g \in [ G ] } \left( L _ { g } ^ { \mathrm { t r } } ( W ^ { * } ) - L _ { \mathcal { D } , g } ( W ^ { * } ) \right) \leq M \sqrt { \frac { 2 \log ( 4 G / \delta ) } { \operatorname* { m i n } _ { g } n _ { g } ^ { \mathrm { t r } } } }
$$

Combining the uniform convergence over $\mathcal { H } _ { H }$ (probability $1 - \delta / 2 )$ , the stability gap for $\eta ^ { * }$ evaluated in Step 1 (probability $1 - \delta / 4 )$ , and the Hoeffding bound for the fixed reference $W ^ { * }$ (probability $1 -$ $\delta / 4 )$ , the total failure probability sums exactly to δ, yielding the final continuous oracle inequality. □

Corollary F.6 (Joint Hyperparameter Tuning of λ and η). Suppose the assumptions ofTheorem $F . 5$ hold. Let the joint hyperparameter vector be $\psi = ( \dot { \lambda } , \eta ) \overset { \cdot \cdot } { \in } \Psi = \Lambda \times \dot { H } \subset \mathbb { R } ^ { \tilde { 2 } }$ , where $\Lambda =$ $[ \lambda _ { \operatorname* { m i n } } , \lambda _ { \operatorname* { m a x } } ] , H = [ \eta _ { \operatorname* { m i n } } , \eta _ { \operatorname* { m a x } } ]$ , and $R = d i a m ( \Psi )$ . Let ψ<sup>ˆ</sup> = arg min<sub>ψ∈Ψ</sub> $f _ { \nu a l } ( \widehat { W } _ { \psi } )$ . The lowerlevel algorithmic mapping isjointly Lipschitz with respect to ψ with constant $\begin{array} { r } { \rho _ { \mathcal { A } } \le \frac { \dot { L } } { \lambda _ { \operatorname* { m i n } } ^ { 2 } } + \frac { G L M } { \lambda _ { \operatorname* { m i n } } \eta _ { \mathrm { m i n } } ^ { 2 } } . } \end{array}$

With probability at least $1 - \delta ,$ the true worst-group risk of the jointly tuned predictor satisfies:

$$
\begin{array} { r l } & { L _ { \mathcal { D } } ^ { w o r t } ( \widehat { W } _ { \widehat { \psi } } ) } \\ & { \leq L _ { \mathcal { D } } ^ { w o r t } ( W ^ { * } ) + M \sqrt { \frac { 2 \log \left( 4 G / \delta \right) } { \operatorname* { m i n } _ { g } n _ { g } ^ { \mathrm { t r } } } } + 2 M \sqrt { \frac { 2 \log \left( 4 G / \delta \right) } { \operatorname* { m i n } _ { g } n _ { g } ^ { \mathrm { s q 1 } } } } } \\ & { + \underset { \lambda \in \Lambda , \eta \in H } { \operatorname* { m i n } } \left( \Omega _ { \lambda , \eta } ( W ^ { * } ) + \beta _ { D R o } ( \lambda , \eta ) + \Sigma ( \lambda , \eta ) \sqrt { \frac { \log \left( 4 G / \delta \right) } { 2 } } \right) } \\ & { + \frac { 2 4 M } { \sqrt { \operatorname* { m i n } _ { g } n _ { g } ^ { \mathrm { v a l } } } } \sqrt { 2 } \left( \sqrt { \log \left( \frac { 3 \rho _ { A } R L } { M } \right) } + 2 \sqrt { \log 2 } \right) } \end{array}
$$

where $\begin{array} { r } { \Omega _ { \lambda , \eta } ( W ^ { * } ) = \frac { \lambda } { 2 } \| W ^ { * } \| ^ { 2 } + \frac { \eta } { 2 } } \end{array}$ , and β<sub>DRO</sub>(λ, η) = 2L<sup>2</sup> 1 + <sup>M√G</sup> η

Proof of Corollary F.6. To establish the joint continuous uniform convergence bound, we must prove the mapping $\psi \mapsto \widehat { W } _ { \psi }$ is Lipschitz over Ψ. We directly apply the exact strong monotonicity argument used in Lemma F.14. Let $J ( W ; \lambda , \eta )$ denote the lower-level Group DRO objective. For two hyperparameter configurations $\pmb { \psi } = ( \lambda , \eta )$ and $\psi ^ { \prime } = ( \lambda ^ { \prime } , \eta ^ { \prime } )$ , let $\widehat { W }$ and $\widehat { W } ^ { \prime }$ be their respective minimizers.

Because $J ( W ; \lambda , \eta )$ is λ-strongly convex with respect to W, its gradient is λ-strongly monotone. Evaluating at the two optima and exploiting the first-order optimality condition $\nabla J ( \widehat { W } ; \lambda , \eta ) =$ $\nabla J ( \widehat { W } ^ { \prime } ; \lambda ^ { \prime } , \eta ^ { \prime } ) = 0$ , we bound the shift in the predictors by the shift in the gradients evaluated at the fixed point ${ \widehat { W } } ^ { \prime }$

$$
\begin{array} { r l } & { \lambda \| \widehat { W } - \widehat { W } ^ { \prime } \| ^ { 2 } \leq \langle \nabla J ( \widehat { W } ^ { \prime } ; \lambda , \eta ) - \nabla J ( \widehat { W } ; \lambda , \eta ) , \widehat { W } ^ { \prime } - \widehat { W } \rangle } \\ & { \qquad \leq \langle \nabla J ( \widehat { W } ^ { \prime } ; \lambda , \eta ) - \nabla J ( \widehat { W } ^ { \prime } ; \lambda ^ { \prime } , \eta ^ { \prime } ) , \widehat { W } ^ { \prime } - \widehat { W } \rangle } \\ & { \qquad \leq \| \nabla J ( \widehat { W } ^ { \prime } ; \lambda , \eta ) - \nabla J ( \widehat { W } ^ { \prime } ; \lambda ^ { \prime } , \eta ^ { \prime } ) \| \| \widehat { W } - \widehat { W } ^ { \prime } \| } \end{array}
$$

Dividing by $\lambda \| \widehat { W } \mathrm { ~ - ~ } \widehat { W } ^ { \prime } \|$ isolates the deviation. The exact gradient of the objective is $\begin{array} { r } { \nabla J ( W ; \lambda , \dot { \eta } ) = \sum q _ { g } \nabla L _ { a } ^ { \mathrm { t r } } ( W ) + \lambda W } \end{array}$ . Thus, the gradient shift decomposes into a regularization shift and a robust weight shift:

$$
\| \nabla J ( \widehat { W } ^ { \prime } ; \lambda , \eta ) - \nabla J ( \widehat { W } ^ { \prime } ; \lambda ^ { \prime } , \eta ^ { \prime } ) \| \leq | \lambda - \lambda ^ { \prime } | \| \widehat { W } ^ { \prime } \| + \left\| \sum _ { g = 1 } ^ { G } ( q _ { g } - q _ { g } ^ { \prime } ) \nabla L _ { g } ^ { \mathrm { t r } } ( \widehat { W } ^ { \prime } ) \right\|
$$

For the first term, the optimality condition for ${ \widehat { W } } ^ { \prime }$ implies $\begin{array} { r } { \lambda ^ { \prime } \widehat { W } ^ { \prime } = - \sum q _ { g } ^ { \prime } \nabla L _ { g } ^ { \mathrm { t r } } ( \widehat { W } ^ { \prime } ) } \end{array}$ . Since $\lVert \nabla L _ { g } ^ { \mathrm { t r } } \rVert \leq L$ , we have $\begin{array} { r } { \| \widehat { W } ^ { \prime } \| \leq \frac { L } { \lambda ^ { \prime } } \leq \frac { L } { \lambda _ { \operatorname* { m i n } } } } \end{array}$ . For the second term, following the weight perturbation derivation in Lemma ${ \mathrm { F } } . 4 ,$ the optimal simplex weights shift by at most $\| q - q ^ { \prime } \| _ { 2 } \leq$ $\begin{array} { r l } {  { \| ( \frac { 1 } { \eta } - \frac { 1 } { \eta ^ { \prime } } ) L ^ { \mathrm { t r } } ( \widehat { W } ^ { \prime } ) \| _ { 2 } } } \end{array}$ . Since $| L _ { g } ^ { \mathrm { t r } } | \le M$ , this is bounded by ${ \frac { | \eta - \eta ^ { \prime } | } { \eta _ { \mathrm { m i n } } ^ { 2 } } } \sqrt { G } M$ . Multiplying by the Jacobian norm $( \sqrt { G } L )$ bounds the robust weight shift by $\frac { G M L } { \eta _ { \mathrm { m i n } } ^ { 2 } } | \eta - \eta ^ { \prime } |$

Combining these and dividing by $\lambda \geq \lambda _ { \operatorname* { m i n } }$ yields the final perturbation bound:

$$
\vert \vert \widehat { W } - \widehat { W } ^ { \prime } \vert \vert \leq \frac { L } { \lambda _ { \operatorname* { m i n } } ^ { 2 } } \vert \lambda - \lambda ^ { \prime } \vert + \frac { G L M } { \lambda _ { \operatorname* { m i n } } \eta _ { \operatorname* { m i n } } ^ { 2 } } \vert \eta - \eta ^ { \prime } \vert \leq \left( \frac { L } { \lambda _ { \operatorname* { m i n } } ^ { 2 } } + \frac { G L M } { \lambda _ { \operatorname* { m i n } } \eta _ { \operatorname* { m i n } } ^ { 2 } } \right) \Vert \psi - \psi ^ { \prime } \Vert _ { 2 }
$$

This establishes the joint Lipschitz constant $\rho _ { \mathcal { A } }$ over $\Psi$ . Injecting this $\rho _ { \mathcal { A } }$ into a 2-dimensional variant of Lemma F.2 yields the $\sqrt { 2 }$ dimension scaling on the validation uniform deviation. The minimization trades off this against the local stability gap $\beta _ { \mathrm { { D R O } } } ( \lambda , \eta )$ , completing the proof. □

Corollary F.7 (Joint Tuning with Deep Encoder Parameters). Suppose the assumptions of Corollary F.6 hold. Further assume that the input features are produced by a deep encoder $z = f _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ parameterized by $\theta \in \Theta$ , where $\boldsymbol { \Theta } \subset \mathbb { R } ^ { \hat { d } _ { \theta } }$ . Assume $( l )$ the encoder output ${ \bar { f } } _ { \theta } ( x )$ is $L _ { f } – L i p s c h i t z$

with respect to $\theta ,$ and (2) the classification loss gradient $\nabla _ { W } \ell ( W , z )$ is $L _ { g r a d }$ -Lipschitz with respect to z. Let the extended joint hyperparameter vector be $\psi = ( \lambda , \eta , \theta ) \in \Psi \times \Theta$

The lower-level algorithmic mapping is jointly Lipschitz with respect to ψ with constant:

$$
\rho _ { \mathcal { A } } \leq \frac { L } { \lambda _ { \operatorname* { m i n } } ^ { 2 } } + \frac { G L M } { \lambda _ { \operatorname* { m i n } } \eta _ { \operatorname* { m i n } } ^ { 2 } } + \frac { L _ { g r a d } L _ { f } } { \lambda _ { \operatorname* { m i n } } }\tag{39}
$$

Crucially, the lower-level uniform stability constant $\beta _ { D R O } ( \lambda , \eta )$ remains unchanged, as $\theta$ is fixed during the lower-level optimization. Consequently, the true worst-group risk bound takes the identical structural form as Corollary F.6, with the dimension factor k increasing to $2 + d _ { \theta }$ and the covering radius scaling to diam $( \dot { \Psi } \times \Theta )$ .

Proof. To derive the joint algorithmic Lipschitz constant, we follow the exact strong monotonicity argument used in Corollary F.6. Let $\begin{array} { r } { J ( W ; \lambda , \eta , \theta ) = \sum q _ { g } ( \eta ) L _ { a } ^ { \mathrm { t r } } ( W , \theta ) + \frac { \lambda } { 2 } \| W \| ^ { 2 } } \end{array}$ denote the lowerlevel objective. For any two joint configurations $\boldsymbol { \psi } = ( \lambda , \eta , \ddot { \theta ) }$ and $\psi ^ { \prime } \overset { = } { = } ( \lambda ^ { \prime } , \eta ^ { \prime } , \theta ^ { \prime } )$ , the optimal predictors shift according to the total gradient shift at the fixed point ${ \widehat { W } } ^ { \prime }$

$$
\lambda \| \widehat { W } - \widehat { W } ^ { \prime } \| \leq \| \nabla J ( \widehat { W } ^ { \prime } ; \lambda , \eta , \theta ) - \nabla J ( \widehat { W } ^ { \prime } ; \lambda ^ { \prime } , \eta ^ { \prime } , \theta ^ { \prime } ) \|
$$

By the triangle inequality, this gradient shift decomposes into three distinct perturbations corresponding to the regularization, the group weights, and the encoder representations:

$$
\begin{array} { r } { \| \nabla J ( \widehat { W } ^ { \prime } ; \lambda , \eta , \theta ) - \nabla J ( \widehat { W } ^ { \prime } ; \lambda ^ { \prime } , \eta ^ { \prime } , \theta ^ { \prime } ) \| \leq \underbrace { \left| \lambda - \lambda ^ { \prime } \right| \| \widehat { W } ^ { \prime } \| } _ { \leq \frac { L } { \lambda _ { \operatorname* { m i n } } } | \lambda - \lambda ^ { \prime } | } + \underbrace { \left| \underset { g = 1 } { \overset { G } { \sum } } ( q _ { g } - q _ { g } ^ { \prime } ) \nabla _ { W } L _ { g } ^ { \mathrm { t r } } ( \widehat { W } ^ { \prime } , \theta ) \right| } _ { \leq \frac { G L M } { \eta _ { \operatorname* { m i n } } ^ { 2 M } } | \eta - \eta ^ { \prime } | } } \\ { + \underbrace { \left| \underset { g = 1 } { \overset { G } { \sum } } q _ { g } ^ { \prime } \left( \nabla _ { W } L _ { g } ^ { \mathrm { t r } } ( \widehat { W } ^ { \prime } , \theta ) - \nabla _ { W } L _ { g } ^ { \mathrm { t r } } ( \widehat { W } ^ { \prime } , \theta ^ { \prime } ) \right) \right| } _ { \mathrm { E n c o d e r ~ s i l i f t } } } \end{array}
$$

The first two bounds follow identically from Corollary F.6. For the third term, because the simplex weights satisfy $\textstyle \sum q _ { q } ^ { \prime } = 1$ , the encoder shift is bounded by the maximum gradient deviation across groups. Applying the smoothness of the loss and the Lipschitz property of the encoder, this shift evaluates to:

$$
\operatorname* { m a x } _ { g } \| \nabla _ { W } L _ { g } ^ { \mathrm { t r } } ( \widehat { W } ^ { \prime } , \theta ) - \nabla _ { W } L _ { g } ^ { \mathrm { t r } } ( \widehat { W } ^ { \prime } , \theta ^ { \prime } ) \| \leq L _ { \mathrm { g r a d } } L _ { f } \| \theta - \theta ^ { \prime } \|
$$

Combining these three bounds and dividing by $\lambda \geq \lambda _ { \operatorname* { m i n } }$ yields the final mapping deviation:

$$
\bigl \| \widehat { W } - \widehat { W } ^ { \prime } \bigr \| \leq \biggl ( \frac { L } { \lambda _ { \operatorname* { m i n } } ^ { 2 } } + \frac { G L M } { \lambda _ { \operatorname* { m i n } } \eta _ { \operatorname* { m i n } } ^ { 2 } } + \frac { L _ { \operatorname { g r a d } } L _ { f } } { \lambda _ { \operatorname* { m i n } } } \biggr ) \| \psi - \psi ^ { \prime } \| _ { 2 }
$$

For the stability term, uniform stability measures the sensitivity of the learning algorithm to a single training point perturbation for a fixed hyperparameter configuration. Because $\bar { \theta }$ is an upper-level variable, it acts as a constant mapping $x \mapsto z$ during the lower-level optimization. Provided the loss $\ell ( W , z )$ is L-Lipschitz over the bounded representation space, the stability constant $\beta _ { \mathrm { D R O } }$ relies solely on the loss properties and the fixed regularization λ, remaining identical to the linear case.

Remark F.8 (Neural Tangent Kernel). While this generic continuous treatment seamlessly maintains the logical flow of our bilevel framework, the sample complexity bounds could be further tightened by incorporating specialized neural network generalization theories, such as the Neural Tangent Kernel (NTK) (Jacot et al., 2018).

## F.6 EXTENSION TO HIERARCHICAL DRO (BI-HDRO)

The continuous generalization theory seamlessly extends to Hierarchical DRO (HDRO) where the continuous hyperparameters being tuned include the multi-dimensional inner perturbation radii $\epsilon =$ $( \epsilon _ { 1 } , \dots , \epsilon _ { G } )$ . In this formulation, the joint hyperparameter is $\pmb { \psi } = ( \lambda , \eta , \epsilon )$ . As long as the smoothed robust loss $\dot { \ell } _ { \mathrm { r o b } , g } ( W , \epsilon _ { g } )$ is used, the lower-level mapping remains Lipschitz continuous.

Lemma F.9 (Algorithmic Lipschitz Continuity of Bi-HDRO). Assume the smoothed robust loss $\ell _ { r o b , g } ( W , \epsilon _ { g } ) = \bar { \mathbb { E } } [ \operatorname* { s u p } _ { \| \delta _ { g } \| \leq \epsilon _ { g } } \bar { \ell } ( W , z + \delta _ { g } , y ) ]$ has bounded gradient shifts with respect to $\epsilon _ { g } ,$ , satisfying $\| \nabla _ { W } \ell _ { r o b , g } ( W , \epsilon _ { g } ^ { ( 1 ) } ) - \nabla _ { W } \ell _ { r o b , g } ( W , \epsilon _ { g } ^ { ( 2 ) } ) \| _ { 2 } \leq C _ { \epsilon } | \epsilon _ { g } ^ { ( 1 ) } - \epsilon _ { g } ^ { ( 2 ) } |$ , and is $L _ { \epsilon } – L i p s c h i t z$ with respect to $\epsilon _ { g } .$ . Let the lower-level objective maintain λ-strong convexity via $L _ { 2 }$ regularization. Then, the HDRO algorithmic mapping $\widehat { W } _ { \psi }$ is jointly Lipschitz with resp ect to the combined hyperparameter $\pmb { \psi } = ( \lambda , \eta , \epsilon )$ with the joint Lipschitz constant bounded by:

$$
\rho _ { \mathcal { A } , \psi } \leq \underbrace { \frac { L } { \lambda _ { \operatorname* { m i n } } ^ { 2 } } + \frac { G L M } { \lambda _ { \operatorname* { m i n } } \eta _ { \operatorname* { m i n } } ^ { 2 } } } _ { S h i f t f r o m { \lambda , \eta } } + \underbrace { \frac { C _ { \epsilon } } { \lambda _ { \operatorname* { m i n } } } + \frac { \sqrt { G } L L _ { \epsilon } } { \lambda _ { \operatorname* { m i n } } \eta _ { \operatorname* { m i n } } } } _ { S h i f t f r o m { \epsilon } }\tag{40}
$$

Remark F.10 (Validity of the Lipschitz Assumption for Classification). The assumption that the robust loss has bounded gradient shifts with respect to $\epsilon _ { g }$ naturally holds for the exact closed-form perturbations used in binary classification (derived in Appendix C). Analytically solving the inner adversarial maximization min $\| \delta \| \le \epsilon ^ { \gamma } ( W ^ { \top } ( z + \delta ) + b )$ yields the robust margin $y ( \dot { W } ^ { \top } z + b ) { - \epsilon \| W \| _ { * } }$ where $\| W \| _ { * }$ is the dual norm of the perturbation constraint.

Consider $L _ { 2 }$ norm perturbations where the dual norm is simply $\| W \| _ { * } = \| W \| _ { 2 }$ . For the globally smooth robust logistic/BCE loss $\ell _ { \mathrm { r o b } } ( W , \epsilon ) = \log ( 1 + \exp ( A ) )$ , where $\begin{array} { r } { A = - y ( \ddot { W } ^ { \top } z + b ) + \check { \epsilon } \| W \| _ { 2 } } \end{array}$ the gradient with respect to $W$ everywhere is $\begin{array} { r } { \nabla _ { W } \ell _ { \mathrm { r o b } } ( W , \epsilon ) = \sigma ( A ) \left( - y z + \epsilon \frac { W } { \| W \| _ { 2 } } \right) } \end{array}$ . Using the $1 / 4 { \cdot } \mathrm { I }$ Lipschitz continuity of the sigmoid function $\sigma ( \cdot )$ , the gradient shift between two perturbation radii $\epsilon ^ { ( 1 ) }$ and $\epsilon ^ { ( 2 ) }$ is explicitly bounded by the triangle inequality:

$$
\begin{array} { r l } & { \| ( \sigma ( A ^ { ( 1 ) } ) - \sigma ( A ^ { ( 2 ) } ) ) ( - y z ) + ( \sigma ( A ^ { ( 1 ) } ) \epsilon ^ { ( 1 ) } - \sigma ( A ^ { ( 2 ) } ) \epsilon ^ { ( 2 ) } ) \frac { W } { \| W \| _ { 2 } } \| _ { 2 } } \\ & { \le \frac { 1 } { 4 } | A ^ { ( 1 ) } - A ^ { ( 2 ) } | \| z \| _ { 2 } + 1 \cdot | \epsilon ^ { ( 1 ) } - \epsilon ^ { ( 2 ) } | + \epsilon _ { \operatorname* { m a x } } \frac { 1 } { 4 } | A ^ { ( 1 ) } - A ^ { ( 2 ) } | } \\ & { = \left( 1 + \frac { 1 } { 4 } \| W \| _ { 2 } \| z \| _ { 2 } + \frac { \epsilon _ { \operatorname* { m a x } } } { 4 } \| W \| _ { 2 } \right) | \epsilon ^ { ( 1 ) } - \epsilon ^ { ( 2 ) } | . } \end{array}
$$

Because the lower-level objective enforces λ-strong convexity via $\frac { \lambda } { 2 } \lVert W \rVert _ { 2 } ^ { 2 }$ , the optimal weights $\| W \| _ { 2 }$ are bounded by a constant $M _ { w }$ . Assuming bounded $\bar { \| \boldsymbol { z } \| _ { 2 } }$ , this ensures the gradient shift is strictly bounded by $\dot { C } _ { \epsilon } | \epsilon ^ { ( 1 ) } - \epsilon ^ { ( 2 ) } |$ where $C _ { \epsilon }$ is a finite constant. Furthermore, this justifies the $L _ { \epsilon } -$ Lipschitzness of the loss value itself: the derivative $\begin{array} { r } { \frac { \partial \ell _ { \mathrm { r o b } } } { \partial \epsilon } = \sigma ( A ) \lVert W \rVert _ { 2 } } \end{array}$ is strictly bounded by $\| W \| _ { 2 }$ meaning the robust loss is L -Lipschitz with $L _ { \epsilon } \leq \dot { M } _ { w }$ . Thus, all Lipschitz continuity assumptions are rigorously satisfied globally in our implementation.

Proof. Let $\begin{array} { r } { F ( W , q , \epsilon ) = \sum _ { q = 1 } ^ { G } q _ { g } \ell _ { \mathrm { r o b } , g } ( W , \epsilon _ { g } ) - \frac { \eta } { 2 } \| q - \frac { 1 } { G } { \bf 1 } \| ^ { 2 } + \frac { \lambda } { 2 } \| W \| ^ { 2 } } \end{array}$ be the lower-level HDRO objective. By Danskin’s theorem, the gradient of the max-marginalized objective $J _ { \epsilon } ( W )$ with respect to W is $\begin{array} { r } { \nabla _ { W } J _ { \epsilon } ( W ) = \sum _ { q = 1 } ^ { G } q _ { q } ^ { * } \nabla _ { W } \ell _ { \mathrm { r o b } , g } ( W , \epsilon _ { g } ) + \lambda W } \end{array}$ , where $\begin{array} { r } { q ^ { * } = \Pi _ { \Delta _ { G } } ( \frac { 1 } { \eta } \ell _ { \mathrm { r o b } } ( W , \epsilon ) + \frac { 1 } { G } { \bf 1 } ) } \end{array}$ with $\ell _ { \mathrm { r o b } } ( W , \epsilon ) \in \mathbb { R } ^ { G }$ denoting the vector of robust losses across all groups.

If the perturbation hyperparameter shifts from $\epsilon ^ { ( 1 ) } \mathrm { t o } \ \epsilon ^ { ( 2 ) }$ , the gradient shift is bounded by the triangle inequality:

$$
\begin{array} { r l } & { \| \nabla _ { W } J _ { \epsilon ^ { ( 1 ) } } ( W ) - \nabla _ { W } J _ { \epsilon ^ { ( 2 ) } } ( W ) \| _ { 2 } } \\ & { \le \left\| \displaystyle \sum _ { g = 1 } ^ { G } q _ { g } ^ { ( 1 ) } \left( \nabla _ { W } \ell _ { \mathrm { r o b } , g } ^ { ( 1 ) } - \nabla _ { W } \ell _ { \mathrm { r o b } , g } ^ { ( 2 ) } \right) \right\| _ { 2 } + \left\| \displaystyle \sum _ { g = 1 } ^ { G } \left( q _ { g } ^ { ( 1 ) } - q _ { g } ^ { ( 2 ) } \right) \nabla _ { W } \ell _ { \mathrm { r o b } , g } ^ { ( 2 ) } \right\| _ { 2 } } \\ & { \le \displaystyle \sum _ { g = 1 } ^ { G } q _ { g } ^ { ( 1 ) } C _ { \epsilon } | \epsilon _ { g } ^ { ( 1 ) } - \epsilon _ { g } ^ { ( 2 ) } | + \displaystyle \sum _ { g = 1 } ^ { G } | q _ { g } ^ { ( 1 ) } - q _ { g } ^ { ( 2 ) } | \underbrace { \| \nabla _ { W } \ell _ { \mathrm { r o b } , g } ^ { ( 2 ) } \| _ { 2 } } _ { \le L } } \\ & { \le C _ { \epsilon } \| \epsilon ^ { ( 1 ) } - \epsilon ^ { ( 2 ) } \| _ { 2 } + L \| q ^ { ( 1 ) } - q ^ { ( 2 ) } \| _ { 2 } } \\ & { \le C _ { \epsilon } \| \epsilon ^ { ( 1 ) } - \epsilon ^ { ( 2 ) } \| _ { 2 } + L \sqrt { G } \| q ^ { ( 1 ) } - q ^ { ( 2 ) } \| _ { 2 } } \end{array}
$$

Because the simplex projection $\Pi _ { \Delta _ { G } }$ is 1-Lipschitz, the shift in the adversarial weights is strictly bounded by the shift in the robust loss terms scaled by the fixed penalty $\eta \geq \eta _ { \mathrm { m i n } }$

$$
\lVert q ^ { ( 1 ) } - q ^ { ( 2 ) } \rVert _ { 2 } \le \frac { 1 } { \eta _ { \mathrm { m i n } } } \lVert \ell _ { \mathrm { r o b } } ( W , \epsilon ^ { ( 1 ) } ) - \ell _ { \mathrm { r o b } } ( W , \epsilon ^ { ( 2 ) } ) \rVert _ { 2 } \le \frac { L _ { \epsilon } } { \eta _ { \mathrm { m i n } } } \lVert \epsilon ^ { ( 1 ) } - \epsilon ^ { ( 2 ) } \rVert _ { 2 }
$$

Substituting this bound into the gradient shift yields a total shift bounded by $\begin{array} { r } { \left( C _ { \epsilon } + \frac { \sqrt { G } L L _ { \epsilon } } { \eta _ { \mathrm { m i n } } } \right) \Vert \epsilon ^ { ( 1 ) } - } \end{array}$ $\epsilon ^ { ( 2 ) } \parallel _ { 2 }$ . Because the objective is λ-strongly convex, applying the exact same strong monotonicity argument from Lemma F.1 divides this gradient shift by $\lambda ,$ proving the mapping is Lipschitz continuous with respect to ϵ. Summing this ϵ-specific constant with the Lipschitz bounds for λ and η derived in Corollary 3.8 establishes the combined joint Lipschitz constant $\rho _ { \mathcal { A } , \psi }$ over the entire hyperparameter space. □

Remark F.11 (Uniform Stability of Bi-HDRO). While tuning ϵ expands the algorithmic Lipschitz constant, the uniform stability of the lower-level algorithm remains unchanged. For a fixed configuration, Bi-HDRO optimizes the robust loss $\bar { \ell _ { \mathrm { r o b } , g } ( W ) } = \operatorname* { s u p } _ { \| \delta \| < \epsilon } \ell ( \bar { W , } z + \delta )$ . Because the global constants L and M bound the base loss across all possible inputs, they naturally bound any perturbed input $z + \delta$ . Thus, Bi-HDRO inherits the exact same stability constant $\beta _ { \mathrm { D R O } } ( \lambda , \eta ) =$ $\begin{array} { r } { \frac { 2 L ^ { 2 } } { \lambda \operatorname* { m i n } _ { g } n _ { g } ^ { \mathrm { t r } } } \left( 1 + \frac { M \sqrt { G } } { \eta } \right) } \end{array}$ as standard group DRO.

Corollary F.12 (Joint Tuning of Bi-HDRO with Deep Encoder Parameters). Suppose the assumptions of Lemma F.9 hold. Further assume the input features are generated by a deep encoder $z = f _ { \boldsymbol { \theta } } ( x )$ parameterized by $\theta \in \Theta$ , such that the encoder output is L<sub>f</sub>-Lipschitz with respect to θ, and the gradient of the robust loss $\nabla _ { W } \ell _ { r o b , g } ( W , \theta , \epsilon _ { g } )$ representation z. Let the fully extended joint hyperparameter vector be $\psi = ( \lambda , \eta , \epsilon , \theta ) \in \Psi \times \Theta$

The Bi-HDRO algorithmic mapping is jointly Lipschitz with respect to ψ with constant:

$$
\rho _ { { A , \psi } } \leq \underbrace { \frac { L } { \lambda _ { \operatorname* { m i n } } ^ { 2 } } + \frac { G L M } { \lambda _ { \operatorname* { m i n } } \eta _ { \operatorname* { m i n } } ^ { 2 } } } _ { S h i f t f r o m { \lambda , \eta } } + \underbrace { \frac { C _ { \epsilon } } { \lambda _ { \operatorname* { m i n } } } + \frac { \sqrt { G } L L _ { \epsilon } } { \lambda _ { \operatorname* { m i n } } \eta _ { \operatorname* { m i n } } } } _ { S h i f t f r o m { \epsilon } } + \underbrace { \frac { L _ { g r a d } ^ { r o b } L _ { f } } { \lambda _ { \operatorname* { m i n } } } } _ { S h i f t f r o m { \theta } }\tag{41}
$$

Moreover, as established in the preceding remark, the uniform stability constant $\beta _ { H D R O } ( \lambda , \eta )$ remains identical.

Proof. The proof follows immediately by combining the derivation of Lemma F.9 with the triangle inequality decomposition established in Corollary F.7. The gradient shift now contains a fourth additive term arising from the variation in θ, which evaluates to ma $\tau _ { g } \parallel \nabla _ { W } \ell _ { \mathrm { r o b } , g } ( \widehat { W } ^ { \prime } , \theta , \epsilon _ { g } ) \ -$ $\nabla _ { W } \ell _ { \mathrm { r o b } , g } \bigl ( \widehat { W } ^ { \prime } , \theta ^ { \prime } , \epsilon _ { g } \bigr ) \| \leq L _ { \mathrm { g r a d } } ^ { \mathrm { r o b } } L _ { f } \| \theta - \theta ^ { \prime } \|$ . Dividing by the strong convexity constant $\lambda _ { \mathrm { m i n } }$ yields the additive θ-shift term. □

## F.7 BACKGROUND: UNIFORM STABILITY

To establish generalization guarantees, we rely on the framework of uniform stability. We first explicitly recall the uniform stability of the lower-level algorithm, adapting the standard analysi for Tikhonov regularization (Shalev-Shwartz & Ben-David, 2014, Section 13.3) to our general $\lambda -$ strongly convex regularizer $\Omega _ { \lambda }$

Definition F.13 (Uniform Stability (Bousquet & Elisseeff, 2002)). A learning algorithm A is $\beta -$ uniformly stable with respect to a loss function ℓ if, for any two training sets $\bar { S } , \bar { S } ^ { ( \bar { i } ) }$ of size $m$ that differ by exactly one example, and for any arbitrary test point z, the following holds:

$$
\operatorname* { s u p } _ { z } | \ell ( A ( S ) , z ) - \ell ( A ( S ^ { ( i ) } ) , z ) | \leq \beta\tag{42}
$$

Lemma F.14 (Uniform Stability of λ-Strongly Convex RLM). Assume the loss function $\ell ( W , z )$ is convex and L-Lipschitz with respect to $\bar { W } .$ . Let $\Omega _ { \lambda } ( W )$ be a λ-strongly convex regularization function. Then the Regularized Loss Minimization rule A(S) = arg min<sub>W</sub> $( L _ { S } ( W ) + \bar { \Omega } _ { \lambda } ( W ) )$ is β-uniformly stable with $\begin{array} { r } { \beta = \frac { 2 L ^ { 2 } } { \lambda m } } \end{array}$

Proof. Let $\begin{array} { c c l } { S } & { = } & { \left( z _ { 1 } , \ldots , z _ { m } \right) } \end{array}$ be a training set, $z ^ { \prime }$ an additional example, and $\begin{array} { r l } { S ^ { ( i ) } } & { { } = } \end{array}$ $( z _ { 1 } , \ldots , z _ { i - 1 } , z ^ { \prime } , z _ { i + 1 } , \ldots , z _ { m } )$ . Denote $f _ { S } ( W { \bar { ) } } = L _ { S } ( W ) + \Omega _ { \lambda } ( W )$ . Because $f _ { S }$ is λ-strongly convex, its gradient is λ-strongly monotone. Evaluating this for the optimal predictors $W = \boldsymbol { \mathcal { A } } ( \boldsymbol { \bar { S } } )$ and $W ^ { ( i ) } = \mathcal { A } ( S ^ { ( i ) } )$ yields:

$$
\lambda \| W ^ { ( i ) } - W \| ^ { 2 } \leq \langle \nabla f _ { S } ( W ^ { ( i ) } ) - \nabla f _ { S } ( W ) , W ^ { ( i ) } - W \rangle
$$

In view of the first-order optimality conditions of $W$ and $W ^ { ( i ) }$ , we have:

$$
\begin{array} { r } { \langle - \nabla f _ { S } ( W ) , W ^ { ( i ) } - W \rangle \leq 0 } \\ { \langle \nabla f _ { S ^ { ( i ) } } ( W ^ { ( i ) } ) , W ^ { ( i ) } - W \rangle \leq 0 } \end{array}
$$

Summing these three inequalities yields:

$$
\begin{array} { r } { \lambda \| W ^ { ( i ) } - W \| ^ { 2 } \leq \langle \nabla f _ { S } ( W ^ { ( i ) } ) - \nabla f _ { S ^ { ( i ) } } ( W ^ { ( i ) } ) , W ^ { ( i ) } - W \rangle } \end{array}
$$

Applying the Cauchy-Schwarz inequality, we can bound the distance strictly by the shift in the gradients:

$$
\lambda \| W ^ { ( i ) } - W \| \le \| \nabla f _ { S } ( W ^ { ( i ) } ) - \nabla f _ { S ^ { ( i ) } } ( W ^ { ( i ) } ) \|\tag{43}
$$

Expanding the empirical risk gradients, the shift is exactly:

$$
\| \nabla f _ { S } ( W ^ { ( i ) } ) - \nabla f _ { S ^ { ( i ) } } ( W ^ { ( i ) } ) \| = \left\| \frac { \nabla \ell ( W ^ { ( i ) } , z _ { i } ) - \nabla \ell ( W ^ { ( i ) } , z ^ { \prime } ) } { m } \right\| \le \frac { 2 L } m
$$

Dividing by λ gives the optimal parameter displacement $\| W ^ { ( i ) } - W \| \leq \frac { 2 L } { \lambda m }$ . Finally, the $L _ { - }$ Lipschitzness of ℓ implies that for any test point z, the difference in loss is bounded by:

$$
| \ell ( A ( S ^ { ( i ) } ) , z ) - \ell ( A ( S ) , z ) | \leq L \| A ( S ^ { ( i ) } ) - A ( S ) \| \leq \frac { 2 L ^ { 2 } } { \lambda m }\tag{44}
$$

Thus, the learning rule is $\frac { 2 L ^ { 2 } } { \lambda m }$ -uniformly stable.

Lemma F.15 (High Probability Generalization via Stability). Let the learning algorithm A be $\beta \mathrm { \cdot }$ uniformly stable, and assume the loss function ℓ is bounded by M. Then, for any $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta$ over the random draw ofa training set $S _ { t r a i n }$ ofsize $n ^ { \mathrm { t r } }$ , the true risk ofthe output hypothesis is bounded by:

$$
L _ { \mathcal { D } } ( \mathcal { A } ( S _ { t r a i n } ) ) \le L ^ { \mathrm { t r } } ( \mathcal { A } ( S _ { t r a i n } ) ) + \beta + \left( 2 \beta + \frac { 2 M } { n ^ { \mathrm { t r } } } \right) \sqrt { \frac { n ^ { \mathrm { t r } } \log ( 1 / \delta ) } { 2 } }\tag{45}
$$

For the λ-strongly convex Regularized Loss Minimization rule defined in Lemma $F . l 4 ,$ , we substitute $\begin{array} { r } { \beta = \frac { 2 L ^ { 2 } } { \lambda n ^ { \mathrm { t r } } } } \end{array}$

Proof. Let $f ( S _ { \mathrm { t r a i n } } ) = L _ { \mathcal { D } } ( A ( S _ { \mathrm { t r a i n } } ) ) - L ^ { \mathrm { t r } } ( A ( S _ { \mathrm { t r a i n } } ) )$ denote the generalization gap. A fundamental result in stability theory (Shalev-Shwartz & Ben-David, 2014, Section 13.2) guarantees that the expected generalization gap is bounded by the uniform stability: $\mathbb { E } _ { S _ { \mathrm { t r a i n } } } [ f ( S _ { \mathrm { t r a i n } } ) ] { \stackrel { - } { \le } } \beta$

To obtain a high-probability bound, we analyze the sensitivity of $f ( S _ { \mathrm { t r a i n } } )$ to the replacement of a single training example. Let $S _ { \mathrm { t r a i n } }$ and $S _ { \mathrm { t r a i n } } ^ { ( i ) }$ be two training sets differing by exactly one example $z _ { i }  z ^ { \prime }$ . By definition of $\beta .$ -uniform stability, the loss on any arbitrary point z changes by at most $\beta \colon$

$$
\operatorname* { s u p } _ { z } | \ell ( A ( S _ { \mathrm { t r a i n } } ) , z ) - \ell ( A ( S _ { \mathrm { t r a i n } } ^ { ( i ) } ) , z ) | \le \beta\tag{46}
$$

Taking the expectation over $z \sim \mathcal { D }$ , the difference in true risk is bounded by this uniform difference:

$$
| L _ { \mathcal { D } } ( \mathcal { A } ( S _ { \mathrm { t r a i n } } ) ) - L _ { \mathcal { D } } ( \mathcal { A } ( S _ { \mathrm { t r a i n } } ^ { ( i ) } ) ) | \le \mathbb { E } _ { z \sim \mathcal { D } } \left[ \operatorname* { s u p } _ { z ^ { \prime } } | \ell ( \mathcal { A } ( S _ { \mathrm { t r a i n } } ) , z ^ { \prime } ) - \ell ( \mathcal { A } ( S _ { \mathrm { t r a i n } } ^ { ( i ) } ) , z ^ { \prime } ) | \right] \le \beta\tag{47}
$$

Furthermore, we can bound the change in the empirical risk between the two sets. Noting that $S _ { \mathrm { t r a i n } }$ and $S _ { \mathrm { t r a i n } } ^ { ( i ) }$ share $n ^ { \mathrm { t r } } - 1$ identical points and only differ at the i-th point $( z _ { i } \ \mathbf { V } \mathbf { S } \ z ^ { \prime } )$ , we have:

$$
\begin{array} { r l } & { | L ^ { \mathrm { t r } } ( \mathcal { A } ( S _ { \mathrm { t r a n } } ) ) - L ^ { \mathrm { t r } , ( \lambda ) } ( \mathcal { A } ( S _ { \mathrm { t r a n } } ^ { ( i ) } ) ) | = \displaystyle \left| \frac { 1 } { n ^ { \mathrm { t r } } } \sum _ { j = 1 } ^ { n ^ { \mathrm { t r } } } \left( \ell ( \mathcal { A } ( S _ { \mathrm { t r a n } } ) , z _ { j } ) - \ell ( \mathcal { A } ( S _ { \mathrm { t r a n } } ^ { ( i ) } ) , z _ { j } ^ { ( i ) } ) \right) \right| } \\ & { \quad \quad \quad \quad \quad \quad \leq \displaystyle \frac { 1 } { n ^ { \mathrm { t r } } } \sum _ { j \neq i } ^ { n } \frac { | \ell ( \mathcal { A } ( S _ { \mathrm { t r a n } } ) , z _ { j } ) - \ell ( \mathcal { A } ( S _ { \mathrm { t r a n } } ^ { ( i ) } ) , z _ { j } ) | } { \xi \beta ( \mathrm { m i d o m s t a b i l i t y } ) } } \\ & { \quad \quad \quad \quad \quad + \displaystyle \frac { 1 } { n ^ { \mathrm { t r } } } \frac { \left\{ \ell ( \mathcal { A } ( S _ { \mathrm { t r a n } } ) , z _ { i } ) - \ell ( \mathcal { A } ( S _ { \mathrm { t r a n } } ^ { ( i ) } ) , z ^ { \prime } ) \right\} } { \xi 2 M ( \mathrm { o u a d o s t a b } ) } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \leq \displaystyle \frac { n ^ { \mathrm { t r } } - 1 } { n ^ { \mathrm { t r } } } \beta + \frac { 2 M } { n ^ { \mathrm { t r } } } \leq \beta + \frac { 2 M } { n ^ { \mathrm { t r } } } } \end{array}
$$

Consequently, the change in the function f when one point is perturbed is bounded by:

$$
\begin{array} { r l } & { \lvert f ( S _ { \mathrm { t r a i n } } ) - f ( S _ { \mathrm { t r a i n } } ^ { ( i ) } ) \rvert \le \lvert L _ { \mathcal { D } } ( \mathcal { A } ( S _ { \mathrm { t r a i n } } ) ) - L _ { \mathcal { D } } ( \mathcal { A } ( S _ { \mathrm { t r a i n } } ^ { ( i ) } ) ) \rvert + \lvert L ^ { \mathrm { t r } } ( \mathcal { A } ( S _ { \mathrm { t r a i n } } ) ) - L ^ { \mathrm { t r } , ( i ) } ( \mathcal { A } ( S _ { \mathrm { t r a i n } } ^ { ( i ) } ) ) \rvert } \\ & { \qquad \le \beta + \bigg ( \beta + \frac { 2 M } { n ^ { \mathrm { t r } } } \bigg ) = 2 \beta + \frac { 2 M } { n ^ { \mathrm { t r } } } } \end{array}
$$

Thus, $f ( S _ { \mathrm { t r a i n } } )$ satisfies the bounded differences property with constant $\begin{array} { r } { c = 2 \beta + \frac { 2 M } { n ^ { \mathrm { t r } } } } \end{array}$ . Applying the one-sided McDiarmid’s inequality (Lemma F.16), we have that with probability at least $1 - \delta \colon$

$$
f ( S _ { \mathrm { t r a i n } } ) \leq \mathbb { E } [ f ( S _ { \mathrm { t r a i n } } ) ] + c { \sqrt { \frac { n ^ { \mathrm { t r } } \log ( 1 / \delta ) } { 2 } } } \leq \beta + \left( 2 \beta + { \frac { 2 M } { n ^ { \mathrm { t r } } } } \right) { \sqrt { \frac { n ^ { \mathrm { t r } } \log ( 1 / \delta ) } { 2 } } }\tag{48}
$$

which yields the final result.

## F.8 HELPFUL LEMMAS

For completeness, we include the explicit derivations for properties utilized in the main theorems.

Lemma F.16 (McDiarmid’s Inequality). Let $X _ { 1 } , \ldots , X _ { m }$ be independent random variables, and let $f ( X _ { 1 } , \ldots , X _ { m } )$ be a function that satisfies the bounded differences property with constants $c _ { 1 } , \ldots , c _ { m } .$

$$
| f ( x _ { 1 } , \dots , x _ { i } , \dots , x _ { m } ) - f ( x _ { 1 } , \dots , x _ { i } ^ { \prime } , \dots , x _ { m } ) | \leq c _ { i }\tag{49}
$$

Then for any $\epsilon > 0 ,$ , the one-sided deviation is bounded by:

$$
P ( f ( X _ { 1 } , \ldots , X _ { m } ) - \mathbb { E } [ f ] \geq \epsilon ) \leq \exp \left( - { \frac { 2 \epsilon ^ { 2 } } { \sum _ { i = 1 } ^ { m } c _ { i } ^ { 2 } } } \right)\tag{50}
$$

By symmetry, the two-sided absolute deviation is bounded by:

$$
P ( | f ( X _ { 1 } , \ldots , X _ { m } ) - \mathbb { E } [ f ] | \geq \epsilon ) \leq 2 \exp \left( - { \frac { 2 \epsilon ^ { 2 } } { \sum _ { i = 1 } ^ { m } c _ { i } ^ { 2 } } } \right)\tag{51}
$$

Lemma F.17 (Sub-Gaussian Variance Proxy). Let $X _ { 1 } , \ldots , X _ { m }$ be independent random variables, and let $f ( X _ { 1 } , \ldots , X _ { m } )$ be a function that satisfies the bounded differences property with constants $c _ { 1 } , \ldots , c _ { m } .$

$$
| f ( x _ { 1 } , \dots , x _ { i } , \dots , x _ { m } ) - f ( x _ { 1 } , \dots , x _ { i } ^ { \prime } , \dots , x _ { m } ) | \leq c _ { i }\tag{52}
$$

Then the random variable $Z = f ( X _ { 1 } , \ldots , X _ { m } )$ is a sub-Gaussian random variable with variance proxy $\textstyle \sigma ^ { 2 } = { \frac { 1 } { 4 } } \sum _ { i = 1 } ^ { m } c _ { i } ^ { 2 }$

Proof. By the one-sided McDiarmid’s inequality (Lemma F.16), for any $t \geq 0$ , the probability of deviation from the expected value is bounded by:

$$
P ( Z - \mathbb { E } [ Z ] \geq t ) \leq \exp \left( - { \frac { 2 t ^ { 2 } } { \sum _ { i = 1 } ^ { m } c _ { i } ^ { 2 } } } \right)\tag{53}
$$

A random variable $Z$ is formally defined as sub-Gaussian with variance proxy $\sigma ^ { 2 }$ if its tail distribution satisfies $\begin{array} { r } { P ( Z - \mathbb { E } [ Z ] \geq t ) \leq \exp \left( - \frac { t ^ { 2 } } { 2 \sigma ^ { 2 } } \right) } \end{array}$ . By equating the exponents of the bounds, we have:

$$
{ \frac { t ^ { 2 } } { 2 \sigma ^ { 2 } } } = { \frac { 2 t ^ { 2 } } { \sum _ { i = 1 } ^ { m } c _ { i } ^ { 2 } } } \implies 2 \sigma ^ { 2 } = { \frac { 1 } { 2 } } \sum _ { i = 1 } ^ { m } c _ { i } ^ { 2 } \implies \sigma ^ { 2 } = { \frac { 1 } { 4 } } \sum _ { i = 1 } ^ { m } c _ { i } ^ { 2 }
$$

Lemma F.18 (Maximal Inequality for Sub-Gaussian Random Variables). Let $Z _ { 1 } , \ldots , Z _ { n }$ be afinite collection of sub-Gaussian random variables, where each $Z _ { i }$ has variance proxy $\sigma ^ { 2 }$ and expectation $\mu _ { i } = \mathbb { E } [ Z _ { i } ]$ . The expected maximum of these random variables is bounded by:

$$
\mathbb { E } \left[ \operatorname* { m a x } _ { 1 \leq i \leq n } Z _ { i } \right] \leq \operatorname* { m a x } _ { 1 \leq i \leq n } \mu _ { i } + \sigma \sqrt { 2 \log n }\tag{54}
$$

Proof. Let $Y _ { i } = Z _ { i } - \mu _ { i }$ . By definition, each $Y _ { i }$ is a zero-mean sub-Gaussian random variable with variance proxy $\sigma ^ { 2 }$ , satisfying the moment generating function bound $\begin{array} { r } { \mathbb { E } [ \exp ( s Y _ { i } ) ] \le \exp \left( \frac { s ^ { 2 } \sigma ^ { 2 } } { 2 } \right) } \end{array}$ for any $s > 0$ . We wish to bound $\mathbb { E } [ \operatorname* { m a x } _ { i } Y _ { i } ]$

By Jensen’s inequality, since the exponential function is strictly convex for $s > 0 \colon$

$$
\begin{array} { r l } { \exp \left( s \mathbb { E } \left[ \underset { 1 \leq i \leq n } { \operatorname* { m a x } } Y _ { i } \right] \right) \leq \mathbb { E } \left[ \exp \left( s \underset { 1 \leq i \leq n } { \operatorname* { m a x } } Y _ { i } \right) \right] } & { } \\ { = \mathbb { E } \left[ \underset { 1 \leq i \leq n } { \operatorname* { m a x } } \exp ( s Y _ { i } ) \right] } & { } \\ { \leq \mathbb { E } \left[ \underset { i = 1 } { \overset { n } { \sum } } \exp ( s Y _ { i } ) \right] } & { } \\ { = \underset { i = 1 } { \overset { n } { \sum } } \mathbb { E } [ \exp ( s Y _ { i } ) ] \leq n \exp \left( \frac { s ^ { 2 } \sigma ^ { 2 } } { 2 } \right) } \end{array}
$$

Taking the natural logarithm of both sides and dividing by s yields:

$$
\mathbb { E } \left[ \operatorname* { m a x } _ { 1 \leq i \leq n } Y _ { i } \right] \leq \frac { \log n } { s } + \frac { s \sigma ^ { 2 } } { 2 }\tag{55}
$$

To minimize this upper bound, we select the optimal parameter $s = \sqrt { 2 \log n / \sigma ^ { 2 } }$ . Substituting this into the inequality gives:

$$
\mathbb { E } \left[ \operatorname* { m a x } _ { 1 \leq i \leq n } Y _ { i } \right] \leq \frac { \log n } { \sqrt { 2 \log n } / \sigma } + \frac { \sigma ^ { 2 } \sqrt { 2 \log n / \sigma ^ { 2 } } } { 2 } = \sigma \sqrt { \frac { \log n } { 2 } } + \sigma \sqrt { \frac { \log n } { 2 } } = \sigma \sqrt { 2 \log n }\tag{56}
$$

Finally, because max<sub>i</sub> $Z _ { i } \le \operatorname* { m a x } _ { i } \mu _ { i } + \operatorname* { m a x } _ { i } Y _ { i } .$ , applying the expectation yields $\mathbb { E } [ \operatorname* { m a x } _ { i } Z _ { i } ] \ \leq$ max<sub>i</sub> $\mu _ { i } + \mathbb { E } [ \operatorname* { m a x } _ { i } Y _ { i } ] \leq \operatorname* { m a x } _ { i } \mu _ { i } + \sigma \sqrt { 2 \log n } .$ □

Lemma F.19 (Non-Expansive Property of the Maximum Operator). For any two finite sets of real numbers $A = \{ A _ { 1 } , \dotsc , A _ { G } \}$ and $\mathbf { \bar { \boldsymbol { B } } } = \{ B _ { 1 } , \ldots , B _ { G } \}$ , the absolute difference between their maxi mums is bounded by the maximum oftheir element-wise absolute differences:

$$
\left| \operatorname* { m a x } _ { g \in [ G ] } A _ { g } - \operatorname* { m a x } _ { g \in [ G ] } B _ { g } \right| \leq \operatorname* { m a x } _ { g \in [ G ] } \left| A _ { g } - B _ { g } \right|\tag{57}
$$

Proof. Without loss of generality, assume that ma $\mathfrak { c } _ { g } A _ { g } \geq \operatorname* { m a x } _ { g } B _ { g }$ . Let $k = \arg \operatorname* { m a x } _ { g } A _ { g }$ be the index that achieves the maximum for A. We can then write the difference as:

$$
\operatorname* { m a x } _ { g } A _ { g } - \operatorname* { m a x } _ { g } B _ { g } = A _ { k } - \operatorname* { m a x } _ { g } B _ { g }
$$

Since the maximum of the set $B$ must be at least as large as any specific element in $B ,$ , we know that ma $\mathfrak { c } _ { g } B _ { g } \geq B _ { k }$ . Substituting this lower bound can only increase the difference:

$$
A _ { k } - \operatorname* { m a x } _ { g } B _ { g } \leq A _ { k } - B _ { k }
$$

Since a quantity is always bounded by its absolute value, and the k-th element’s difference is bounded by the maximum absolute difference across all elements, we have:

$$
A _ { k } - B _ { k } \leq | A _ { k } - B _ { k } | \leq \operatorname* { m a x } _ { g } | A _ { g } - B _ { g } |
$$

This establishes the bound, completing the proof.

□