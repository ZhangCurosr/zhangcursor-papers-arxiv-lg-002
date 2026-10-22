# Preventing Model Extraction in Contextual Linear Optimization

Adam N. Elmachtoub

Tianyi Lin Columbia University

Wenxuan Liu

## Abstract

Consider a contextual linear optimization problem, where a predict-then-optimize (PTO) system returns optimal decisions without revealing the cost vectors used in its downstream linear program. Our focus is on how to prevent attackers, who can submit context queries, from extracting the underlying prediction model. Although the cost-to-decision map is many-to-one, observations across contexts can reveal the underlying prediction model. For linear models, we provide suficient conditions where the attacker can learn the model up to positive scaling by only observing the decision. These conditions motivate an adaptive attack that alternates inverse optimization with boundary sampling, i.e., using context queries leading to decision boundaries of the current attacker model. To prevent extraction, we propose a mechanism that perturbs decisions only when the predicted cost vector is near a decision boundary. We establish a regret bound in terms of the perturbation radius. For the attackers that fit the defended responses as optimal decisions under the evaluation distribution, we also prove a lower bound on expected extraction error. Experiments on synthetic instances and real data quantify the security-performance tradeof of the defense mechanism and illustrate that our defense can increase the extraction error.

## 1 INTRODUCTION

Many decision-making problems can be formulated as optimization problems whose parameters are unknown and must be inferred from contexts. For example, a recommendation system uses user features to predict click probabilities before optimizing which items to display. In portfolio optimization, asset returns must be estimated before making allocation decisions. This class of problems is referred to as contextual optimization (Sadana et al., 2025). A widely adopted approach to contextual optimization is the predict-then-optimize (PTO) framework (Elmachtoub and Grigas, 2022). In the linear optimization setting studied here, PTO proceeds in two stages: (i) a machine learning (ML) model predicts the unknown objective cost vector from a given context and (ii) the predicted cost vector is used to solve the downstream linear optimization problem and produce a decision. Due to its modularity and empirical efectiveness, PTO has been successfully applied across a broad range of application domains, including recommendation systems, e-commerce, logistics, and resource allocation.

This setting raises a model-security question: can an attacker recover the hidden prediction model by interacting only with the optimization system’s decision outputs? Such a threat is known as a model extraction attack: the attacker repeatedly queries the deployed system with contexts (perhaps through an API) and uses the returned decisions to build a substitute prediction model. Successful extraction can allow the attacker to replicate the service while bypassing access costs or usage restrictions, and moreover, can provide a stepping stone for user privacy or inference attacks (Narayanan and Shmatikov, 2008; Papernot et al., 2017; Shokri et al., 2017; Salem et al., 2019; Juuti et al., 2019) and information leakage (Tan et al., 2016; Zhang et al., 2023). At first glance, the optimization layer appears to make extraction dificult. The attacker does not observe predicted costs directly, and the map from costs to optimal decisions is many-to-one in linear optimization. This intuition is incomplete: the same polyhedral geometry that hides costs in the interior of normal cones creates boundary contexts where small changes in predicted costs change the returned decision. This leads to two central questions: (i) when do decisionlevel observations identify the underlying prediction model, and (ii) how can we mitigate extraction while preserving decision quality?

Existing approaches do not answer these questions directly. Prior work on privacy in contextual decisionmaking has studied diferential privacy (DP) (Lei et al.,

2024), which protects training data but does not directly defend the deployed predictive model against extraction. A second related line is inverse optimization (Chan et al., 2025), which infers unknown parameters of an optimization problem from observed optimal decisions. While inverse optimization includes both fixed-parameter and contextual formulations, our focus is on whether decision-level observations identify the deployed predictor. More importantly, because the decision map is many-to-one, decision consistency does not in general imply prediction recovery.

In this paper, we focus on model extraction in contextual linear optimization. First, we study vulnerability through a loss calibration condition that links global inverse feasibility consistency to prediction recovery. Motivated by this condition, we then develop an adaptive attack that combines inverse optimization with boundary sampling to reconstruct the hidden predictor using only decision-level observations. This insight motivates a defense that perturbs the returned decision only when the deployed predictor’s cost vector is near a decision boundary. Our theory establishes a regret bound for linear predictors and a lower bound on expected prediction error under a response-matching condition. Our experiments on synthetic instances and real data examine the security-performance trade-of.

Contributions. We formalize model extraction in contextual linear optimization under the predict-thenoptimize paradigm and study when the decision-level observations identify the predictor and how to mitigate extraction while limiting the loss in decision quality. Our contributions are summarized as follows:

• We identify loss calibration as a vulnerability criterion: if it holds, global inverse feasibility consistency is suficient for prediction recovery up to positive scaling. For linear predictors and unrestricted contextual queries, we use boundary contexts to establish this identification result under suficient geometric and regularity conditions. This motivates an adaptive attack that alternates inverse optimization with boundary sampling, which targets contexts near decision boundaries of the current attacker model.

• We propose a decision perturbation defense that randomizes only near decision boundaries and prove a regret upper bound that is linear in the boundary radius. For attackers that fit the defended responses as optimal decisions under the evaluation distribution, we also prove a lower bound on expected prediction error. This lower bound is nondecreasing in the boundary radius.

• We evaluate the attack and defense on synthetic instances and a short video recommendation task.

In the tested settings, boundary sampling achieves lower prediction error than the non-adaptive inverse-optimization baseline, while the proposed defense can increase the prediction error of the evaluated attacks at modest regret.

Related work. Privacy in contextual optimization has been studied through diferential privacy, including personalized pricing and revenue management (Chen et al., 2022, 2023; Lei et al., 2024; Mochaourab et al., 2021; Sun et al., 2025; Chowdhury and Zhou, 2023; Fu et al., 2026; Selvi et al., 2026). These works are complementary to ours: DP protects data privacy, whereas we study model extraction attacks that target the deployed predictor itself. In model extraction attacks, the goal is to recover a substitute model via query access to a target system (Wang and Gong, 2018; Reith et al., 2019; Orekondy et al., 2019; Milli et al., 2019; Chandrasekaran et al., 2020; Pal et al., 2020; Truong et al., 2021; Huang et al., 2024; Ferry et al., 2024; Khouna et al., 2025; Gorgé et al., 2026). Our work is also related to adaptive query attacks (Tramèr et al., 2016; Zhao et al., 2025). In our setting, the attacker observes downstream decisions but not predicted cost vectors. Since the prediction-to-decision map is generally many-to-one, we examine how the geometry of the downstream feasible set afects predictor recovery.

Another related line concerns inverse optimization, which infers unknown problem parameters from observed decisions (Chan et al., 2014; Aswani et al., 2018; Babier et al., 2021; Sun et al., 2023; Besbes et al., 2025; Chan et al., 2025; Patel and Vempala, 2026). We use inverse optimization to study when decision consistency identifies the underlying predictor.

Notation. We write $x \mathrm { ~ / / ~ } y$ for parallel vectors and $\angle ( x , y )$ for the angle between two nonzero vectors. The polyhedral cone generated by $a _ { 1 } , \ldots , a _ { k }$ is denoted by polycone $\{ a _ { 1 } , \ldots , a _ { k } \}$ . For a polytope $S \subseteq \mathbb { R } ^ { n }$ , we let V denote its vertex set. For a vertex $w \in V ,$ , we denote cone(w) as its normal cone: the set of cost vectors $c \in \mathbb { R } ^ { n }$ satisfying $c ^ { \top } ( u - w ) \geq 0$ for all $u \in S .$ . We also denote by $N _ { V } ( w ) = \{ v \in V$ : dim(cone(v)∩cone(w)) = n − 1} the set of vertices whose normal cones share a facet with cone(w). For $c \neq 0 ,$ its angular distance to cone(w) is $\begin{array} { r } { \angle ( c , \mathrm { c o n e } ( w ) ) = \operatorname* { i n f } _ { u \in \mathrm { c o n e } ( w ) \setminus \{ 0 \} } \angle ( c , u ) } \end{array}$

## 2 MODEL FORMULATION

We study model extraction in contextual linear optimization. This setting involves three roles: a decisionmaker, an attacker, and some users. For each user, the decision-maker maps contexts to an action through a predict-then-optimize (PTO) pipeline. Given a cost vector $c \in \mathbb { R } ^ { n }$ , the downstream decision is obtained by solving a linear program in the form of

$$
z ^ { \star } ( c ) = \operatorname* { m i n } _ { w \in S } c ^ { \top } w ,\tag{1}
$$

where $S = \{ w \in \mathbb { R } ^ { n } : A w \leq b \}$ is a compact, fulldimensional polytope. We assume that each inequality in $A w \le b$ defines a distinct facet of S, and denote the nonzero rows of A by $a _ { 1 } ^ { \top } , \ldots , a _ { m } ^ { \top }$ . We denote the optimal solution set by $\bar { W ^ { \star } ( c ) } = \bar { \mathrm { a r g m i n } } _ { w \in S } c ^ { \top }$ w and let $w ^ { \star } ( c ) \in W ^ { \star } ( c ) \cap V$ be an optimal vertex selected by a fixed tie-breaking rule. Compactness ensures existence of an optimizer, while full dimensionality allows us to study cost directions in the ambient space<sup>1</sup>.

In the PTO pipeline, the user’s true cost vector is not observed. The decision-maker instead learns a prediction model ${ \hat { c } } = p ( x )$ from contexts $\boldsymbol { x } \in \mathbb { R } ^ { d }$ and exposes the induced decision rule $\tilde { w } ( x ) = w ^ { \star } ( p ( x ) ) \in W ^ { \star } ( p ( x ) )$ (perhaps through an $\mathrm { A P I } ) ^ { 2 } . \mathrm { \dot { A } }$ user submits a context x and receives the decision $\tilde { w } ( x )$ . We focus on the linear prediction $\mathrm { m o d e l ^ { 3 } }$ in the form of $p ( x ) = P ^ { D M } x ,$ , where $\mathsf { \bar { P } } ^ { D M } \in \mathbb { R } ^ { n \times d }$ is the decision-maker’s prediction matrix. The attacker knows the linear structure but not $P ^ { D M }$ ， and learns a model $p ^ { a } ( x ) = P ^ { a } x$ with $P ^ { a } \in \mathbb { R } ^ { n \times d }$ . We consider $d > n \geq 2 ,$ assume rank $( P ^ { D M } ) = n$ , and exclude the zero attacker matrix $P ^ { a } = 0$

Attackers. A key step is to define the attacker’s prediction loss. Since positive scaling of the cost vector does not change the optimal solution set of a linear program, the attacker succeeds whenever it recovers a prediction model $P ^ { a } = \gamma P ^ { D M }$ for some $\gamma > 0$ . We then measure prediction error by angular distance. Since $d > n .$ , prediction matrices have nontrivial kernels. We extend the angular loss by setting $\angle ( 0 , 0 ) = 0$ and $\begin{array} { r } { \angle ( u , v ) = \frac { \pi } { 2 } } \end{array}$ when exactly one of u and v is zero.

Definition 2.1 (Prediction Loss). Given a context x and prediction matrices $P ^ { a }$ and $P ^ { D M }$ , the prediction loss is ${ \cal L } _ { \mathrm { p r e d } } ( P ^ { a } , P ^ { D M } , x ) = \angle ( P ^ { a } x , P ^ { D M } x )$ . For $D ~ = ~ \{ ( x _ { i } , \dot { w _ { i } } ) \} _ { i = 1 } ^ { N } ,$ , the empirical prediction loss is $\begin{array} { r } { L _ { \mathrm { p r e d } } ( P ^ { a } , P ^ { D M } ; D ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \angle ( P ^ { a } x _ { i } , P ^ { D M } x _ { i } ) } \end{array}$

We assume that the attacker can make N queries $x _ { i } ~ \in ~ \mathbb { R } ^ { d }$ , each chosen using the preceding queryresponse pairs. Without defense, query $x _ { i }$ returns $w _ { i } = w ^ { \star } ( p ( x _ { i } ) )$ . After observing $D ^ { a } = \{ ( x _ { i } , w _ { i } ) \} _ { i = 1 } ^ { N }$ the attacker returns a prediction matrix $P ^ { a } = \mathcal { A } ( D ^ { a } )$

where A is his algorithm. Fixing a test distribution $Q ,$ we evaluate $\mathbb { E } [ L _ { \mathrm { p r e d } } ( P ^ { a } , P ^ { D M } , \bar { x } ) ]$ , where the expectation is taken over an independent evaluation context $x \sim Q$ and any attack or defense randomness. The attacker aims to reduce this prediction loss using only information available through the query interface. The following proposition shows that traditional model extraction on classification model fits into the predictthen-optimize model.

Proposition 2.2. For the simplex $S = \{ w \in \mathbb { R } _ { + } ^ { n }$ $\textstyle \sum _ { j = 1 } ^ { n } w _ { j } \ = 1 \}$ , we define p(x) as the negative class utility vector of an n-class classifier. With matched tiebreaking rules, returning the optimal vertex $w ^ { \star } ( p ( x ) ) =$ $e _ { i }$ is equivalent to returning label i for a n-classification problem with prediction model p(x).

Defense. To defend against extraction, the decisionmaker may strategically modify the query interface to restrict the information revealed to adversarial users. Instead of returning the exact optimizer $w ^ { \star } ( p ( x ) )$ , the decision-maker (API) outputs a perturbed decision w˜(x), so that an attacker querying $x _ { i }$ observes $w _ { i } = \tilde { w } ( x _ { i } )$ . The decision-maker aims to prevent reconstruction of the prediction model while maintaining near-optimal decision quality.

This interaction admits a Stackelberg interpretation: the decision-maker commits to a response mechanism before the attacker chooses its queries and learning procedure. For a specified attack, the security-performance trade-of is measured by

Security: $\mathbb { E } [ L _ { \mathrm { p r e d } } ( P ^ { a } , P ^ { D M } , x ) ]$

Performance: $\mathbb { E } [ p ( x ) ^ { \top } ( \tilde { w } ( x ) - w ^ { \star } ( p ( x ) ) ) ]$

The expectations are taken over an independent $x \sim$ $Q$ and any attack or defense randomness. The first criterion favors larger attacker prediction error, whereas the second favors smaller regret under the deployed predicted costs. We consider the expected regret when $\mathbb { E } _ { x \sim Q } \| p ( x ) \| _ { 2 } < + \infty$

## 3 MODEL EXTRACTION

We characterize how decision-level query responses can reveal the underlying prediction model. Throughout this section, we assume that no defensive mechanism is deployed, so the decision-maker returns the true optimal decision $\tilde { w } ( x ) = w ^ { \star } ( P ^ { D M } x )$

## 3.1 Vulnerability identification

The attacker first assesses whether the decision-maker is vulnerable to model extraction, where vulnerability means that the prediction model can be recovered using decision-level queries. Since $P ^ { D M } \mathcal { _ { 3 } }$ x is never observed, the attacker cannot directly evaluate $L _ { \mathrm { p r e d } } ( \cdot , P ^ { D M } , x )$ Instead, the attacker optimizes a surrogate loss that depends on $P ^ { D M }$ only through the observed decision $w ^ { \star } ( P ^ { D M } x )$ . A class of surrogate loss functions comes from inverse optimization (Chan et al., 2025). Given a query-response dataset $D = \{ ( x _ { i } , w _ { i } ) \} _ { i = 1 } ^ { N }$ , we define the inverse feasible set as follows:

$$
{ \mathcal { P } } _ { \operatorname { i n v } } ( D ) = \{ P \neq 0 ^ { n \times d } : w _ { i } \in W ^ { \star } ( P x _ { i } ) , \forall ( x _ { i } , w _ { i } ) \in D \} .
$$

We allow $P x _ { i } = 0$ at individual contexts but exclude the zero prediction matrix. This is the set of prediction matrices whose induced costs make every observed decision optimal.

Definition 3.1. A nonnegative loss $L ( \cdot , P ^ { D M } ; D )$ defined on nonzero matrices is an inverse feasibility loss for $D { \mathrm { ~ i f ~ } } L ( P , P ^ { D M } ; D ) = 0 \Longleftrightarrow P \in { \mathcal { P } } _ { \mathrm { i n v } } ( D ) . { \mathrm { ~ W e } }$ also write ${ \cal L } ( \dot { P _ { \bf \ L } } P ^ { D M } ; x ) \stackrel { _ { \ L } } { = } { \cal L } ( P , P ^ { D M } ; \{ ( x , w ^ { \star } ( { \bar { P } } ^ { D M } x ) ) \} )$ for a single context $x \in \mathbb { R } ^ { d }$

Achieving zero inverse feasibility loss enforces decision consistency on the observed queries. However, decision consistency alone does not in general imply prediction recovery since diferent cost vectors can induce the same optimal decision. We thus need to identify when inverse feasibility losses are appropriate surrogates for the prediction loss.

Definition 3.2. An inverse feasibility loss calibrates the prediction loss if, for any attacker model $P ^ { a }$ , we have that $L ( P ^ { a } , P ^ { D M } ; x ) = 0$ for all $x \not \in$ ker $( P ^ { D M } )$ if and only if $L _ { \mathrm { p r e d } } ( P ^ { a } , P ^ { D M } ; x ) \ = \ 0$ for all $x \not \in$ $\ker ( P ^ { D M } )$ . Equivalently, global decision consistency implies recovery of the predicted cost directions, up to the positive scaling ambiguity of linear objectives.

Loss calibration is an exact global identification property: zero inverse feasibility loss on all nonkernel contexts identifies the predictor up to positive scaling. It does not itself bound prediction error from a small loss on a finite dataset. The following lemma shows that for a fixed downstream problem, calibration holds for either all inverse feasibility losses or none.

Lemma 3.3. The loss calibration property holds for either all inverse feasibility losses or none. $I f S$ is an n-dimensional hypercube, the loss calibration property holds for none.

We provide a suficient condition under which the loss calibration property holds in the following assumption.

Assumption 3.4. There exist $n + 1$ facet normals $\{ a _ { k _ { i } } \} _ { i = 1 } ^ { n + \hat { 1 } }$ so that $a _ { k _ { 1 } } , \ldots , a _ { k _ { n } }$ are linearly independent and $\begin{array} { r } { a _ { k _ { n + 1 } } = \sum _ { i = 1 } ^ { n } \beta _ { i } a _ { k _ { i } } } \end{array}$ where $\beta _ { i } \neq 0$ for all i.

Theorem 3.5. Suppose Assumption 3.4 holds. Then, if a nonzero attacker model $P ^ { a }$ satisfies $w ^ { \star } ( P ^ { D M } x ) \in$ $W ^ { \star } ( P ^ { a } x )$ for all x /∈ ker $( P ^ { D M } )$ , we have $P ^ { \dot { a } } = \gamma P ^ { \dot { D } M }$ for some $\gamma > 0$ . Consequently, every inverse feasibility loss calibrates the prediction loss.

Algorithm 1 Attacking scheme   
1: Input: $t = 0 , P ^ { 0 } \neq 0 ^ { n \times d } , \Delta > 0$ and $L ( P , P ^ { D M } ; D )$   
2: Initialize: $\begin{array} { r } { P ^ { 0 }  \frac { P ^ { 0 } } { \parallel P ^ { 0 } \parallel _ { F } } } \end{array}$ and $D ^ { 0 } \gets \mathrm { S a m p l i n g } ( P ^ { 0 } )$   
3: while there exist $( x _ { i } , w _ { i } ) \in D ^ { t }$ and a vertex w such   
that $( P ^ { t } x _ { i } ) ^ { \top } ( w _ { i } - w _ { k } ) > \Delta$ do   
4: $\begin{array} { r } { P ^ { t + 1 } \in \operatorname { a r g m i n } _ { P \neq 0 } \dot { L } ( P , P ^ { D M } ; D ^ { t } ) } \end{array}$   
5: $\begin{array} { r } { P ^ { t + 1 } \gets \frac { P ^ { t + 1 } } { \parallel P ^ { t + 1 } \parallel _ { F } } . } \end{array}$   
6: $D ^ { t + 1 } \gets \ddot { D } ^ { t } \cup \ddot { \mathrm { S a m p l i n g } } ( P ^ { t + 1 } ) .$   
7: $t  t + 1 .$   
8: Return: $P ^ { t } .$

The geometry behind Theorem 3.5 also motivates the use of boundary contexts. Indeed, we define $\begin{array} { r } { r _ { i } = - \frac { a _ { i } } { \lVert a _ { i } \rVert } . } \end{array}$ A boundary context associated with facet i satisfies $P ^ { a } \bar { x } = \alpha r _ { i }$ for some $\alpha > 0$ . At this cost, every vertex of that facet is optimal. For an attacker model, nearby contexts can enter each incident normal cone. Global decision consistency constrains the true cost direction at these boundary contexts.

Proposition 3.6. Suppose that the rank of P<sup>a</sup> is n and $w ^ { \star } ( \bar { P } ^ { D M } x ) \in W ^ { \star } ( { P } ^ { a } x )$ for all $x \notin \ker ( \dot { P } ^ { D M } )$ . Then, $L _ { \mathrm { p r e d } } ( P ^ { a } , P ^ { D M } , \bar { x } ) = 0$ at every boundary context x¯.

Proposition 3.6 shows that global decision consistency fixes the cost direction at boundary contexts. To establish Theorem 3.5, global consistency first gives $P ^ { a } =$ $B P ^ { D M }$ , and cone preservation implies $B a _ { i } = \lambda _ { i } a _ { i }$ with $\lambda _ { i } \geq 0$ . Applying B to the relation in Assumption 3.4 gives $\beta _ { i } ( \lambda _ { k _ { i } } - \lambda _ { k _ { n + 1 } } ) = 0$ for every i. Since all $\beta _ { i }$ are nonzero, these eigenvalues coincide, so B is a scalar multiple of the identity. The scalar is positive because $P ^ { a } \neq 0$

Assumption 3.4 holds for various application problems with $S = \{ w : 0 \leq w \leq 1 , a ^ { \top } w \leq b \}$ when $a _ { i } > 0$ for all i and $\textstyle 0 < b < \sum _ { i } a _ { i }$ . The lower-coordinate facets have normals $- e _ { 1 } , \ldots , - e _ { n }$ and the budget facet has normal a. Since $\begin{array} { r } { a = \sum _ { i } ( - a _ { i } ) ( - e _ { i } ) } \end{array}$ , this assumption holds. By contrast, the hypercube has the normals $\{ \pm e _ { i } \} _ { i = 1 } ^ { n }$ and does not satisfy Assumption 3.4. Although failing a suficient condition does not prove that calibration fails, Lemma 3.3 shows that calibration fails for this set. For comparison, on the Euclidean unit ball, the optimal decision is $\begin{array} { r } { w ^ { \star } ( c ) = - \frac { c } { \| c \| } } \end{array}$ for $c \neq 0 .$ , so it reveals the cost direction. This implies that vulnerability is not restricted to the polyhedral setting.

## 3.2 Algorithmic framework

Under Assumption 3.4, global decision consistency implies prediction recovery. This motivates the adaptive attack in Algorithm 1 that alternates inverse optimization with sampling. The algorithm stops when every observed decision is ∆-optimal under the current attacker model $P ^ { t }$ . The following proposition shows that this procedure stops after finite steps.

Proposition 3.7. Suppose $\| \cal P ^ { 0 } \| _ { F } = 1$ , each inverseoptimization update achieves zero inverse feasibility loss, and every sampling call returns finitely many queries. Assume $\| x _ { i } \| \leq R$ for every query in the run, for some finite $R > 0$ . For $\Delta > 0$ , Algorithm 1 terminates after at most $( 1 + \frac { 2 R L _ { e } } { \Delta } ) ^ { n c }$ inverse optimization updates, where $L _ { e } = \operatorname* { m a x } _ { u , v \in V } \left\| u - v \right\|$

The tolerance $\Delta$ controls empirical suboptimality after Frobenius normalization. The bound counts inverse optimization updates and our result guarantees finite termination. A smaller $\Delta$ makes the stopping criterion stricter and may require more updates.

Inverse optimization. The inverse optimization step minimizes an inverse feasibility loss on the current dataset D<sup>t</sup>. Any inverse feasibility loss can be used in this framework and we use the suboptimality loss $L _ { \mathrm { S O } } ( P , P ^ { D M } , x _ { i } ) = ( P x _ { i } ) ^ { \top } ( w _ { i } - w ^ { \star } ( P x _ { i } ) )$ where $w _ { i } = w ^ { \star } ( P ^ { \mathrm { D M } } x _ { i } )$ is the observed decision. We define $\begin{array} { r } { L _ { \mathrm { S O } } ( P , \dot { P } ^ { D M } ; D ) = \sum _ { ( x _ { i } , w _ { i } ) \in D } L _ { \mathrm { S O } } ( P , P ^ { D M } , x _ { i } ) } \end{array}$ . The empirical loss is zero exactly when $w _ { i } \in W ^ { \star } ( P x _ { i } )$ for every observation. Babier et al. (2021) have shown that minimizing the empirical suboptimality loss admits the convex reformulation in the form of

$$
\begin{array} { r l } { \underset { P , \lambda _ { i } } { \operatorname* { m i n } } } & { \sum _ { i = 1 } ^ { N } | ( P x _ { i } ) ^ { \top } w _ { i } - b ^ { \top } \lambda _ { i } | , } \\ { \mathrm { s . t . } } & { A ^ { \top } \lambda _ { i } = P x _ { i } , \lambda _ { i } \leq 0 \mathrm { ~ f o r ~ a l l ~ } i . } \end{array}\tag{2}
$$

The above homogeneous formulation admits $P = 0$ . In the experiments, we solve two convex optimization problems with the respective constraints $\begin{array} { r } { \sum _ { i = 1 } ^ { \bar { n } } \sum _ { j = 1 } ^ { d } P _ { i j } \ge 1 } \end{array}$ or $\begin{array} { r } { \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { d } P _ { i j } \leq - 1 } \end{array}$ before normalizing the fitted matrix in Frobenius norm. These constraints exclude zero prediction matrices but do not cover matrices whose entries sum to zero. Appendix A.2 discusses this restriction and alternative inverse feasibility losses.

Sampling. Decision consistency on a finite dataset does not necessarily imply small prediction loss, even under loss calibration, since many cost vectors can induce the same downstream optimizer. The sampling step therefore focuses on contexts where this ambiguity is easiest to expose.

Our key idea is to query boundary and near-boundary contexts. Using the directions $r _ { i }$ defined in Section 3.1, an ϵ-near-boundary context satisfies $P ^ { a } x \neq 0$ and $\angle ( P ^ { a } x , r _ { i } ) < \epsilon$ for some i. Such queries can expose prediction misalignment even when interior contexts return the same decision (see Figure 1). Formally, we define ${ \cal I } ( P ^ { a } , P ^ { D M } ) = \{ x : L _ { \mathrm { S O } } ( P ^ { a } , P ^ { \dot { D M } } , x ) > 0 \}$ . A context in this set reveals a decision inconsistency. The following proposition summarizes the observation.

![](images/3ad2eaec9e73504399a498f14dde8f3624ecf47bc8efa9deb2bbb663a1631a1d.jpg)

(a) near-boundary context  
![](images/4f306e0ec9d19c1771b707d99ad69ae15360164c10a590edbde39b4347eb4db7.jpg)  
(b) non-boundary context  
Figure 1: Illustration of prediction misalignment. Arrows show negative cost vectors. The two models return diferent decisions in (a) and the same decision in (b).

Proposition 3.8. Suppose that Assumption 3.4 holds, the rank of $P ^ { a }$ is n and $L _ { \mathrm { p r e d } } ( P ^ { a } , P ^ { D M } , x ) > 0$ for some x. Then, for every $\epsilon > 0$ , there is an ϵ-nearboundary context $\tilde { x } \in I ( \hat { P ^ { a } } , \hat { P ^ { D M } } )$ .

Proposition 3.8 shows that, whenever the attacker model $P ^ { a }$ is misaligned with $P ^ { D M }$ , some near-boundary query can expose this misalignment through nonzero inverse feasibility loss. Boundary sampling is designed to choose informative queries. Given $P ^ { a }$ , the attacker queries contexts whose predicted costs point along the inward facet directions $r _ { i } .$ , and queries nearby contexts obtained by moving slightly from these boundary directions into adjacent normal cones. We use a parameter M to control proximity to these directions, where larger M places the sampled cost directions closer to the corresponding facet rays. Appendix A.3 specifies the boundary sampling construction.

We now state the convergence guarantee for Algorithm 1 with boundary sampling.

Theorem 3.9 (Informal). Suppose Assumption 3.4 holds. Run Algorithm 1 with a full-rank initialization and boundary sampling as specified in Appendix A.3. Assume that each inverse optimization update returns a full-rank matrix with zero inverse feasibility loss on the accumulated dataset, and that the samples satisfy the uniform boundedness and conditioning requirements of Theorem A.3 across all runs. Choose $M = M ( \Delta )$ with $M ( \Delta ) \to + \infty$ as $\Delta \downarrow 0 ,$ , and suppose that $( M , \Delta )$ satisfies the stopping condition in Theorem A.3. Each run terminates, and its output satisfies $L _ { \mathrm { p r e d } } ( P _ { M ( \Delta ) , \Delta } ^ { a } , P ^ { D M } , x )  0$ for every $x \bar { \notin } \ker ( P ^ { D M } )$

Algorithm 2 Prediction perturbation scheme   
1: Input: $x , P ^ { D M }$ and σ.   
2: Sample $\boldsymbol \eta \sim \mathcal { N } ( \boldsymbol 0 , \sigma ^ { 2 } \boldsymbol I _ { n } )$   
3: Return: $\tilde { w } ( x ) = w ^ { \star } ( \dot { P } ^ { D M } x + \eta ) .$

Theorem 3.9 establishes asymptotic prediction recovery under uniform sampling regularity. This regularity prevents the boundary representatives from becoming asymptotically uninformative. The stopping condition couples M and $\Delta$ so that the algorithm cannot stop while a prescribed near-boundary response disagrees with its unique predicted optimizer. Appendix A.3 gives the construction and the formal conditions.

## 4 PREVENTING MODEL EXTRACTION

We introduce a prediction perturbation baseline that adds Gaussian noise to the predicted cost before optimization (see Algorithm 2). This scheme perturbs costs without using their distance to a decision boundary. Independent resampling allows an attacker to estimate the response distribution by repeating a query.

We then propose a decision perturbation defense. Since near-boundary queries can expose prediction misalignment, we modify responses in these regions by selecting neighboring decisions whose normal cones are close to the predicted cost. Algorithm 3 first computes the true optimizer and then checks whether the predicted cost $\bar { P } ^ { D M } x$ lies close to the normal cone of any neighboring vertex. If no neighboring normal cone is within angular distance $\epsilon , \mathcal { W } = \{ w _ { 0 } \}$ and the decision-maker returns the true optimal decision. Otherwise, the algorithm returns a vertex from W, which contains $w _ { 0 }$ and all eligible neighboring vertices. For each fixed deployed model and boundary radius, the cache is initialized once and retained across queries. Each new candidate set is assigned a uniformly sampled member using fresh randomness. The cache prevents repeated queries from yielding fresh independent perturbations: once a candidate set W has been assigned an output, the same output is returned whenever that set appears again.

The parameter $\epsilon > 0$ controls which neighboring decisions are eligible for perturbation. Increasing ϵ enlarges each candidate set. Below, we bound the regret incurred by the defense scheme in Algorithm 3 .

Algorithm 3 Decision perturbation scheme   
1: Input: $x , P ^ { D M }$ and $\epsilon \geq 0 .$   
2: Compute $c = P ^ { D M } x$ and $w _ { 0 } = w ^ { \star } ( c )$   
3: i $\textbf { f } c = 0 \mathrm { ~ o r ~ } \epsilon = 0$ then   
4: Return: w<sub>0</sub>.   
5: Initialize: $\mathcal { W } = \{ w _ { 0 } \}$   
6: for w $\ b { \ d M } _ { V } ( \ b { w } _ { 0 } )$ do   
7: $\mathbf { i f } \ \angle ( c ,$ cone $( \dot { w } ) ) \leq \epsilon$ then   
8: ${ \mathcal { W } } \gets { \mathcal { W } } { \dot { \cup } } \{ { \dot { w } } \} .$   
9: if W has been recorded then   
10: Return: record(W).   
11: else   
12: Sample w˜ uniformly from W.   
13: $\operatorname { r e c o r d } ( \mathcal { W } )  \tilde { w } .$   
14: Return: w˜.

Proposition 4.1. Define the regret at context x by Regret $( w ; x ) = ( P ^ { D M } x ) ^ { \top } ( w - \bar { w } ^ { \star } ( P ^ { D M } x ) )$ . Under Algorithm ${ \it 3 , f o r }$ every cache realization and every test distribution Q with $\mathbb { E } _ { x \sim Q } [ \| P ^ { D M } x \| ] < + \infty .$ , we have

$$
\begin{array} { r } { \mathbb { E } _ { { x } \sim { Q } } [ \mathrm { R e g r e t } ( \tilde { w } ( x ) ; x ) ] \le L _ { e } \epsilon \mathbb { E } _ { { x } \sim { Q } } [ \| P ^ { D M } x \| ] , } \end{array}
$$

where $L _ { e } = \operatorname* { m a x } _ { u , v \in V } \left\| u - v \right\|$

Proposition 4.1 provides a regret upper bound that is linear in $\epsilon > 0 .$ . This bound also depends on the feasible set’s diameter and the predicted cost’s scale.

We quantify the prediction error incurred by attackers that fit the defended responses. For a context x, we write $c = P ^ { D M } x$ and $w _ { 0 } = w ^ { \star } ( c )$ . For $c \neq 0$ and $\epsilon > 0 .$ we define

$$
\begin{array} { r } { \mathcal { W } _ { \epsilon } ( x ) = \{ w _ { 0 } \} \cup \{ w \in N _ { V } ( w _ { 0 } ) : \angle ( c , \mathrm { c o n e } ( w ) ) \leq \epsilon \} . } \end{array}
$$

In this case, $\mathcal { W } _ { \epsilon } ( x )$ is the final candidate set in Algorithm 3 before cache lookup. When $c = 0 ~ { \mathrm { o r } } ~ \epsilon = 0 .$ we set $\mathcal { W } _ { \epsilon } ( x ) = \{ w _ { 0 } \}$ . For each fixed cache realization and boundary radius, we let E denote the set of predicted costs c for which the algorithm returns a vertex diferent from $w ^ { \star } ( c )$

Theorem 4.2. For a context x with $P ^ { a } x \neq 0$ and $P ^ { D M } x \neq 0 .$ , we let $\phi ( x )$ and $\psi ( x )$ be the angular distances from these costs to the nearest normal cones neighboring cone $\left( w ^ { \star } ( P ^ { a } x ) \right)$ and $c o n e ( w ^ { \star } ( P ^ { D M } x ) )$ , respectively, with margin zero at ties. For every cache realization, $i f \ \tilde { w } ( x ) \in W ^ { \star } ( P ^ { a } x )$ and $P ^ { D M } x \in \mathcal { E }$ , we have

$$
L _ { \mathrm { p r e d } } ( P ^ { a } , P ^ { D M } , x ) \ge \psi ( x ) + \phi ( x ) .
$$

In particular, the loss is at least ϵ if also $\phi ( { \boldsymbol { x } } ) \geq \epsilon$ . The fitting condition holds in particular if

$$
( P ^ { a } x ) ^ { \top } ( \tilde { w } ( x ) - w ^ { \star } ( P ^ { a } x ) ) < \| P ^ { a } x \| \ell _ { e } \sin ( \phi ( x ) ) ,\tag{3}
$$

where $\ell _ { e } = \operatorname* { m i n } _ { u \neq v \in V } \left\| u - v \right\|$ . Fix Q independently of the cache. Suppose that, for almost every realization of the defense and attacker randomness, both costs are

nonzero and $\tilde { w } ( x ) \in W ^ { \star } ( P ^ { a } x )$ for Q-almost every x. Then, we have

$$
\begin{array} { r } { \mathbb { E } [ L _ { \mathrm { p r e d } } ( P ^ { a } , P ^ { \mathrm { D M } } , x ) ] \geq B _ { \epsilon } ( Q ) , } \end{array}
$$

where $\begin{array} { r } { B _ { \epsilon } ( Q ) = \mathbb { E } _ { x \sim Q } [ \psi ( x ) ( 1 - \frac { 1 } { | \mathcal { W } _ { \epsilon } ( x ) | } ) ] } \end{array}$ . On the $l e f t -$ hand side, the expectation is taken over the defense and attacker randomness and an independent $x \sim Q$ . For fixed $P ^ { D M }$ and $Q , B _ { \epsilon } ( Q )$ is nondecreasing in ϵ.

Theorem 4.2 formalizes the defense efect for responsefitting attackers: viewing a perturbed decision as optimal can lead to moving away from the true cost direction. Under the population fitting condition, it provides an expected prediction-error lower bound that is nondecreasing in ϵ, while Proposition 4.1 bounds the regret cost. Section 5 evaluates the efect of ϵ empirically.

## 5 EXPERIMENTS

We evaluate the proposed attack and defense methods on synthetic instances and a short-video recommendation task. We compare adaptive sampling attacks with inverse optimization baselines and examine the securityregret trade-of induced by the defensive algorithm. We report the angular prediction loss in Definition 2.1 and the decision-maker’s regret in Proposition 4.1.

Synthetic instances. We compare three attacks: boundary sampling (BndS), uniform sampling (Unif), and an inverse optimization baseline method that uses a single nonadaptive fit (INV). Here, the feasible set is $S = \bar { \{ \boldsymbol { w } \in [ 0 , 1 ] ^ { \bar { n } } : a ^ { \top } \boldsymbol { w } \leq b \} }$ , with $a _ { j } = 1 + 0 . 5 ( j - 1 )$ for $j = 1 , \dotsc , n$ and $\textstyle b = 0 . 6 \sum _ { j = 1 } ^ { n } a _ { j }$ The deployed model is fitted by linear regression to noisy synthetic data. Unif samples directions uniformly on the unit sphere and maps them to contexts using the current attacker model, so it remains adaptive. All three attacks use the suboptimality-loss fit in Eq. (2), and the INV comparisons use matched sample sizes. Figure 2 reports prediction loss after three adaptive iterations for BndS and Unif, together with the INV comparison, and the sample sizes required to attain prediction loss below 0.001 as the decision dimension n and feature dimension d vary. BndS achieves near-zero median prediction loss across the tested settings. Unif performs similarly at smaller feature dimensions, but has substantially higher loss at d = 15 and d = 20 after three iterations. BndS also uses fewer samples than Unif to reach the loss threshold at these larger feature dimensions. Across the tested decision dimensions, BndS has lower median prediction loss than INV in the displayed comparisons.

Figure 3 compares prediction loss under query budgets up to 3000 for BndS, Unif, and a mixture of the two sampling methods (MIX). MIX uses equal numbers of boundary and uniform samples. For $\epsilon = 0 .$ , we use the interface without any perturbation. All three strategies attain near-zero prediction loss at the largest evaluated budget. For $\epsilon = 0 . 0 0 1 5$ , prediction loss decreases as the budget grows, but at 3000 queries all three strategies have higher median loss than in the undefended setting. MIX has the lowest or near-lowest median loss at the displayed budgets. These results demonstrate finitebudget mitigation for the evaluated attacks, and our defense efectively reduces model extraction accuracy even under large query budget.

Figure 4 reports attack-specific prediction loss-regret trade-ofs. In particular, Figure 4a compares attacks under decision perturbation: the attacker who aims to get more boundary information (BndS(M=50)) sufers from larger prediction under defense. Figure 4b shows decision perturbation increases the attacker’s prediction loss at modest regret and is more efective than prediction perturbation. These comparisons concern the reported attacks and parameter settings.

Short video recommendation. We examine the performance of our framework on a real dataset. We consider a short video recommendation problem constructed from the public dataset KuaiRand-1K (Gao et al., 2022). We use 800 users for training and 200 for testing, with $d = 1 8$ contextual features and $n = 5$ video types. Recommendations satisfy $w \in [ 0 , 1 ] ^ { 5 }$ and $\textstyle \sum _ { i = 1 } ^ { 5 } w _ { i } \leq 3$ . In this experiment, the user contextual information is used to fit a linear predictor of clickthrough rates, and recommendations are selected by solving a resource-allocation problem using these predictions (see Appendix A.5). The cost vector in Eq. (1) is the negative of the predicted click-through-rate vector in this example.

Figure 4c shows that BndS has lower prediction loss than INV at the lowest plotted regret levels. At larger regret levels, both attacks incur greater prediction loss, with a more pronounced increase for BndS. This is compatible with the mechanism in Theorem 4.2: making boundary query causes perturbed responses, and fitting perturbed responses as optimal decisions can cause higher prediction loss.

## 6 CONCLUSION

This paper studies model extraction and defense in contextual linear optimization under the PTO paradigm. We characterize vulnerability through loss calibration, develop an adaptive inverse optimization attack with boundary sampling, and propose a near-boundary decision-perturbation defense. Our theory bounds the regret of the defense and provides a lower bound on expected prediction error for attackers that fit the defended responses as optimal decisions under the evaluation distribution. This lower bound is nondecreasing in the perturbation degree. Experiments show that the defense can increase prediction error for the evaluated attacks within the tested query budgets. Future work includes analyzing attackers that explicitly model the defense and extending the analysis to nonlinear contextual optimization.

![](images/e9b4029b460c5a463f1887b121f75d7ea54a0b94e522a4a48ded8a5b0aa30fb4.jpg)

![](images/56c82bfa4ca99fd0c1d7145898bd330f8724d0827a632cbe909a34f79f416a6b.jpg)

![](images/28819ea11a314c98cc961ff8cd9b9eb4529ef004a322ba57049254ac55389046.jpg)

![](images/8354e9d0e1b49ea65cd99a9ee47963f42219c2797df36905691b3167e8f2a892.jpg)

Figure 2: Attack performance on synthetic data as the decision dimension n (left) and feature dimension d (right) vary. The first and third panels report prediction loss. The second and fourth report the number of context-decision observations required to attain loss below 0.001. Prediction loss points and error bars show medians and $2 5 ^ { \mathrm { t h } }  – 7 5 ^ { \mathrm { t h } }$ percentiles over 100 repetitions per parameter setting.  
![](images/49e949743d85564e98c7a289fd09bcf39f0b7d1f8126fde1c4f63d66c1418301.jpg)

![](images/a4a9f9db62bd5519f7ad4719ec37455502baa7b25760a20f02692d347e0be23b.jpg)  
Figure 3: Prediction loss under increasing query budgets for BndS $( M = 2 0 )$ , Unif, and MIX, with $\epsilon = 0 \ \mathrm { ( l e f t ) }$ and $\epsilon = 0 . 0 0 1 5$ (right). Curves show medians over five repetitions with $d = 1 0$ and $n = 5$

![](images/41b9dfb499639b10ad0c4e1dea54ae1b8dbdc5aaff57113dda9913a2c916a7a4.jpg)  
(a)

![](images/77e2d50618a9a4e116f67586e2a6c8376a21ea7339d7031e19b6ca65a1df78fb.jpg)  
(b)

![](images/c44dd7f98c44d43145c4958d736b517ec480aae38fc90d81b24b3e147deb78d7.jpg)  
(c)  
Figure 4: Empirical prediction loss-regret trade-ofs: (a) decision perturbation against diferent attacks on synthetic instances; (b) prediction versus decision perturbation on synthetic instances; and (c) decision perturbation on the short-video recommendation task. Points show median regret and median prediction loss, and shaded bands show the $2 5 ^ { \mathrm { t h } }  – 7 5 ^ { \mathrm { t h } }$ percentiles of prediction loss over 100 repetitions per parameter setting.

## References

A. Aswani, Z-J. Shen, and A. Siddiq. Inverse optimization with noisy data. Operations Research, 66(3): 870–892, 2018.

A. Babier, T. C. Y. Chan, T. Lee, R. Mahmood, and D. Terekhov. An ensemble learning framework for model fitting and evaluation in inverse linear optimization. INFORMS Journal on Optimization, 3(2): 119–138, 2021.

O. Besbes, Y. Fonseca, and I. Lobel. Contextual inverse optimization: Ofline and online learning. Operations Research, 73(1):424–443, 2025.

T. C. Y. Chan, T. Craig, T. Lee, and M. B. Sharpe. Generalized inverse multiobjective optimization with application to cancer therapy. Operations Research, 62(3):680–695, 2014.

T. C. Y. Chan, R. Mahmood, and I. Y. Zhu. Inverse optimization: Theory and applications. Operations Research, 73(2):1046–1074, 2025.

V. Chandrasekaran, K. Chaudhuri, I. Giacomelli, S. Jha, and S. Yan. Exploring connections between active learning and model extraction. In USENIX Security Symposium, pages 1309–1326, 2020.

X. Chen, D. Simchi-Levi, and Y. Wang. Privacypreserving dynamic personalized pricing with demand learning. Management Science, 68(7):4878– 4898, 2022.

X. Chen, S. Miao, and Y. Wang. Diferential privacy in personalized pricing with nonparametric demand models. Operations Research, 71(2):581–602, 2023.

S. R. Chowdhury and X. Zhou. Diferentially private reward estimation from preference based feedback. In ICML Workshop on The Many Facets of Preference-Based Learning, 2023.

A. N. Elmachtoub and P. Grigas. Smart “predict, then optimize”. Management Science, 68(1):9–26, 2022.

J. Ferry, R. Fukasawa, T. Pascal, and T. Vidal. Trained random forests completely reveal your dataset. In ICML, pages 13545–13569. PMLR, 2024.

X. Fu, N. Chen, P. Gao, and Y. Li. Privacy-preserving personalized recommender systems. Manufacturing & Service Operations Management, 28(1):271–289, 2026.

C. Gao, S. Li, Y. Zhang, J. Chen, B. Li, W. Lei, P. Jiang, and X. He. KuaiRand: An unbiased sequential recommendation dataset with randomly exposed videos. In CIKM, pages 3953–3957, 2022.

A. Gorgé, J. Ferry, S. Gambs, and T. Vidal. Training set reconstruction from diferentially private forests: How efective is DP? In SaTML, pages 1136–1171. IEEE, 2026.

H. Huang, Y. Li, B. Jiang, B. Jiang, L. Liu, Z. Liu, R. Sun, and S. Liang. Enhancing the resilience of LLMs against grey-box extractions. In ICML Workshop on Next Generation of AI Safety, 2024.

M. Juuti, S. Szyller, S. Marchal, and N. Asokan. PRADA: Protecting against DNN model stealing attacks. In IEEE European Symposium on Security and Privacy, pages 512–527. IEEE, 2019.

A. Khouna, J. Ferry, and T. Vidal. From counterfactuals to trees: Competitive analysis of model extraction attacks. In NeurIPS, pages 2721–2756, 2025.

Y. Lei, S. Miao, and R. Momot. Privacy-preserving personalized revenue management. Management Science, 70(7):4875–4892, 2024.

S. Milli, L. Schmidt, A. D. Dragan, and M. Hardt. Model reconstruction from model explanations. In FAccT, pages 1–9. ACM, 2019.

R. Mochaourab, S. Sinha, S. Greenstein, and P. Papapetrou. Robust counterfactual explanations for privacy-preserving SVM. In ICML Workshop on Socially Responsible Machine Learning, 2021.

A. Narayanan and V. Shmatikov. Robust deanonymization of large sparse datasets. In IEEE Symposium on Security and Privacy, pages 111–125. IEEE, 2008.

T. Orekondy, B. Schiele, and M. Fritz. Knockof nets: Stealing functionality of black-box models. In CVPR, pages 4954–4963, 2019.

S. Pal, Y. Gupta, A. Shukla, A. Kanade, S. Shevade, and V. Ganapathy. Activethief: Model extraction using active learning and unannotated public data. In AAAI, pages 865–872, 2020.

N. Papernot, P. McDaniel, I. Goodfellow, S. Jha, Z. B. Celik, and A. Swami. Practical black-box attacks against machine learning. In Asia Conference on Computer and Communications Security, pages 506– 519. ACM, 2017.

S. Patel and S. Vempala. Tight bounds for learning polyhedra with a margin. ArXiv Preprint: 2604.14614, 2026.

R. N. Reith, T. Schneider, and O. Tkachenko. Eficiently stealing your machine learning models. In ACM Workshop on Privacy in the Electronic Society, pages 198–210, 2019.

R. T. Rockafellar. Convex Analysis, volume 28. Princeton University Press, 1970.

U. Sadana, A. Chenreddy, E. Delage, A. Forel, E. Frejinger, and T. Vidal. A survey of contextual optimization methods for decision-making under uncertainty. European Journal of Operational Research, 320(2): 271–289, 2025.

A. Salem, Y. Zhang, M. Humbert, P. Berrang, M. Fritz, and M. Backes. ML-Leaks: Model and data independent membership inference attacks and defenses on machine learning models. In Network and Distributed System Security Symposium. The Internet Society, 2019.

A. Selvi, H. Liu, and W. Wiesemann. Diferential privacy via distributionally robust optimization. Operations Research, 74(1):356–376, 2026.

R. Shokri, M. Stronati, C. Song, and V. Shmatikov. Membership inference attacks against machine learning models. In IEEE Symposium on Security and Privacy, pages 3–18. IEEE, 2017.

C. Sun, S. Liu, and X. Li. Maximum optimality margin: A unified approach for contextual linear programming and inverse linear programming. In ICML, pages 32886–32912. PMLR, 2023.

L. Sun, K. Cong, J. Q. Chooi, and R. Li. DP-AdamW: Investigating decoupled weight decay and bias correction in private deep learning. In ICML Workshop on Reliable and Responsible Foundation Models, 2025.

K. H. Tan, W. P. Wong, and L. Chung. Information and knowledge leakage in supply chain. Information Systems Frontiers, 18(3):621–638, 2016.

B. Tang and E. B. Khalil. Cave: A cone-aligned approach for fast predict-then-optimize with binary linear programs. In CPAIOR, pages 193–210. Springer, 2024.

F. Tramèr, F. Zhang, A. Juels, M. K. Reiter, and T. Ristenpart. Stealing machine learning models via prediction APIs. In USENIX Security Symposium, pages 601–618, 2016.

J-B. Truong, P. Maini, R. J. Walls, and N. Papernot. Data-free model extraction. In CVPR, pages 4771– 4780, 2021.

B. Wang and N. Z. Gong. Stealing hyperparameters in machine learning. In IEEE Symposium on Security and Privacy, pages 36–52. IEEE, 2018.

C. Zhang, B. Liu, G. G. Cai, and T. Huang. When ignorance is bliss: The retailer’s intelligence hazard under information sharing and exchanging. Transportation Research Part E: Logistics and Transportation Review, 180:103356, 2023.

K. Zhao, L. Li, K. Ding, N. Z. Gong, Y. Zhao, and Y. Dong. A systematic survey of model extraction attacks and defenses: State-of-the-art and perspectives. ArXiv Preprint: 2508.15031, 2025.

# Supplementary Material: Preventing Model Extraction in Contextual Linear Optimization

## A ADDITIONAL TECHNICAL MATERIAL

## A.1 Condition of loss calibration

We present the condition of loss calibration and provide additional insights into how the decision-maker can assess vulnerability. For a full-rank attacker matrix $P ^ { a }$ , Proposition 3.6 implies that $w ^ { \star } ( P ^ { D M } x ) \in W ^ { \star } ( P ^ { a } x )$ for every $x \not \in \ker ( P \overset { \partial \cdot } { \partial ^ { M } } )$ ensures positive alignment of $P ^ { a } \bar { x }$ and $P ^ { D M } \bar { x }$ at every boundary context x¯ of $P ^ { a }$ . Recall that $\begin{array} { r } { r _ { i } = - \frac { a _ { i } } { \left\| a _ { i } \right\| } } \end{array}$ . For a full-rank attacker matrix $P ^ { a }$ , we define the candidate space

$$
\mathcal { P } _ { c } ( P ^ { a } ) = \{ P \in \mathbb { R } ^ { n \times d } : P \bar { x } \in \mathrm { s p a n } ( r _ { i } ) \mathrm { ~ f o r ~ e v e r y ~ } i = 1 , 2 , \dots , m \mathrm { ~ a n d ~ e v e r y ~ } \bar { x } \in \mathbb { R } ^ { d } \mathrm { ~ s a t i s f y i n g ~ } P ^ { a } \bar { x } = r _ { i } \} .
$$

The candidate space is a linear subspace that contains $P ^ { a }$ . Its definition requires only that the images of the boundary contexts lie on the corresponding lines. In particular, this space includes the zero matrix, although the zero matrix is excluded from the attacker class.

If $P ^ { a }$ has full row rank and satisfies global inverse feasibility, Proposition 3.6 gives $P ^ { D M } \in \mathcal { P } _ { c } ( P ^ { a } )$ . If this space is 1-dimensional, $P ^ { D M } = \gamma P ^ { a }$ for some scalar γ. Since $P ^ { D M } \neq 0$ , this scalar is nonzero and the positive boundary alignment in Proposition 3.6 implies $\gamma > 0$ . Thus the two matrices have zero prediction loss at every context. Then, we connect Assumption 3.4 to the dimension of $\mathcal { P } _ { c } ( P ^ { a } )$ by the following result:

Proposition A.1. Suppose that Assumption $\it 3 . 4$ holds. Then, for every full-rank matrix $P ^ { a } \in \mathbb { R } ^ { n \times d }$ , we have ${ \mathcal { P } } _ { c } ( P ^ { a } ) = \operatorname { s p a n } \{ P ^ { a } \}$ . In particular, the dimension of $\mathcal { P } _ { c } ( P ^ { a } )$ is 1.

Proposition A.1 ensures that $\mathcal { P } _ { c } ( P ^ { D M } )$ is 1-dimensional under Assumption 3.4. The following corollary shows that this property is suficient for loss calibration.

Corollary A.2. If $\mathcal { P } _ { c } ( P ^ { D M } )$ is 1-dimensional, every inverse feasibility loss calibrates the prediction loss.

Corollary A.2 provides a diagnostic for the decision-maker. For each facet i, we choose $d - n + 1$ linearly independent contexts $\{ \bar { x } _ { i , j } \} _ { j = 1 } ^ { d - n + 1 }$ satisfying $P ^ { D M } \bar { x } _ { i , j } = r _ { i }$ for all $j = 1 , 2 , \ldots , d - n + 1$ and set $X _ { i } = [ \bar { x } _ { i , 1 } , \dotsc , \bar { x } _ { i , d - n + 1 } ]$ . Since $P ^ { D M }$ has full row rank, the linear space $H _ { i } = \{ x \in \mathbb { R } ^ { d } : P ^ { D M } x \in \mathrm { s p a n } ( r _ { i } ) \}$ has dimension $d - n + 1$ , and the columns of $X _ { i }$ form a basis of $H _ { i }$ . Such representatives exist: if $y _ { i }$ satisfies $\dot { P } ^ { D M } y _ { i } = r _ { i }$ and $z _ { 1 } , \dotsc , z _ { d - n }$ form a basis of ker $( P ^ { D M } )$ , one may take $y _ { i } , y _ { i } + z _ { 1 } , . . . , y _ { i } + z _ { d - n }$

Since $r _ { i }$ is a unit vector, the condition $P \bar { x } _ { i , j } \in \mathrm { s p a n } ( r _ { i } )$ is equivalent to $( I _ { n } - r _ { i } r _ { i } ^ { \top } ) P \bar { x } _ { i , j } = 0$ . This implies

$$
\mathcal { P } _ { c } ( P ^ { D M } ) = \{ P \in \mathbb { R } ^ { n \times d } : ( I _ { n } - r _ { i } r _ { i } ^ { \top } ) P X _ { i } = 0 \mathrm { ~ f o r ~ e v e r y ~ } i = 1 , 2 , \ldots , m \} .
$$

Thus, the decision-maker can determine the dimension of the candidate space by solving a linear system in the entries of $P .$ . If its solution space is 1-dimensional, Corollary A.2 implies loss calibration. This diagnostic uses $P ^ { D M }$ and is available to the decision-maker.

## A.2 Inverse optimization methods

We present other loss functions motivated by inverse optimization and the approaches to solve them. Any inverse feasibility loss function can be incorporated into our framework, although the resulting empirical optimization problems are diferent and may require diferent solution methods.

Scale normalization of the suboptimality loss function. The suboptimality-loss formulation in Eq. (2) is homogeneous and admits $P = 0$ . In the experiments, we solve two convex programs with the respective constraints

$$
\sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { d } P _ { i j } \geq 1 \ \mathrm { o r } \ \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { d } P _ { i j } \leq - 1 .
$$

Each constraint excludes the zero matrix. Every matrix with a nonzero entry sum has a positive rescaling satisfying one of these constraints, so zero-loss feasibility is preserved for such matrices. Matrices with zero entry sum are excluded. Since positive rescaling changes the value of the suboptimality loss, the fitted matrix is normalized in Frobenius norm before applying the stopping test in Algorithm 1.

Distance loss function. The distance loss measures the distance from an observed decision to the optima solution set under $P ;$

$$
L _ { d } ( P , P ^ { D M } ; x _ { i } ) = \operatorname* { m i n } _ { w \in W ^ { \star } ( P x _ { i } ) } \| w - w _ { i } \| .\tag{4}
$$

We define the empirical distance loss by

$$
\hat { L } _ { d } ( P , P ^ { D M } ; D ) = \sum _ { i = 1 } ^ { N } L _ { d } ( P , P ^ { D M } ; x _ { i } ) .
$$

Recall that the set $W ^ { \star } ( P x _ { i } )$ is nonempty and compact. Thus, the minimum in Eq. (4) is attained and we have $L _ { d } ( P , P ^ { D M } ; x _ { i } ) = 0$ if and only if $w _ { i } \in W ^ { \star } ( P x _ { i } )$ ). By nonnegativity, the empirical loss vanishes exactly on ${ \mathcal { P } } _ { \mathrm { i n v } } ( D )$ when restricted to nonzero matrices. Since $W ^ { \star } ( 0 ) = S$ , this also holds when $P x _ { i } = 0$

The distance loss is not continuous in general. Following the duality-based approach of Aswani et al. (2018), we consider an optimality-gap relaxation for the fixed feasible set S. For $\epsilon \geq 0 .$ , we define

$$
\begin{array} { r l } { \hat { L } _ { d } ( P , P ^ { D M } ; D , \epsilon ) = } & { \operatorname* { m i n } _ { \{ \tilde { w } _ { i } , \lambda _ { i } \} _ { i = 1 } ^ { N } } ~ \sum _ { i = 1 } ^ { N } \| w _ { i } - \tilde { w } _ { i } \| , } \\ { \mathrm { s . t . ~ } ~ } & { A ^ { \top } \lambda _ { i } = P x _ { i } , ~ i = 1 , \ldots , N , } \\ & { ( P x _ { i } ) ^ { \top } \tilde { w } _ { i } - b ^ { \top } \lambda _ { i } \leq \epsilon , ~ i = 1 , \ldots , N , } \\ & { A \tilde { w } _ { i } \leq b , ~ i = 1 , \ldots , N , } \\ & { \lambda _ { i } \leq 0 , ~ i = 1 , \ldots , N . } \end{array}
$$

Strong linear programming duality makes these constraints equivalent to $( P x _ { i } ) ^ { \top } \tilde { w } _ { i } - z ^ { \star } ( P x _ { i } ) \leq \epsilon$ and $\tilde { w } _ { i } \in S$ for every $i = 1 , 2 , \dots , N$ . This implies $\hat { L } _ { d } ( P , P ^ { D M } ; D , 0 ) = \hat { L } _ { d } ( P , P ^ { D M } ; D )$ . For $\epsilon > 0 .$ , the relaxed loss is continuous in $P$ but need not be an inverse feasibility loss. For every fixed $P ,$ we have

$$
\hat { L } _ { d } ( P , P ^ { D M } ; D , \epsilon ) \uparrow \hat { L } _ { d } ( P , P ^ { D M } ; D ) \mathrm { a s } \epsilon \downarrow 0 .
$$

When minimization is over a fixed nonempty compact set of nonzero matrices, every accumulation point of global minimizers of the relaxed loss as $\epsilon \downarrow 0$ minimizes the exact empirical distance loss over the same set.

CaVE loss function. Following the approach of Tang and Khalil (2024), we define the cone-aligned vector estimation (CaVE) loss function as follows,

$$
L _ { C } ( P ^ { a } , P ^ { D M } ; x _ { i } ) = \left\{ \begin{array} { l l } { { 1 - \cos ( \angle ( P ^ { a } x _ { i } , \mathrm { c o n e } ( w _ { i } ) ) ) , } } & { { \mathrm { i f ~ } P ^ { a } x _ { i } \not = 0 , } } \\ { { 0 , } } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right.\tag{5}
$$

Its empirical value is

$$
L _ { C } ( P ^ { a } , P ^ { D M } ; D ) = { \textstyle \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } L _ { C } ( P ^ { a } , P ^ { D M } ; x _ { i } ) .
$$

For nonzero costs, zero angular distance to the closed normal cone is equivalent to being in that cone. At zero cost, every feasible decision is optimal. For every nonzero matrix $P ^ { a }$ , we have

$$
L _ { C } ( P ^ { a } , P ^ { D M } ; D ) = 0 \Longleftrightarrow w _ { i } \in W ^ { \star } ( P ^ { a } x _ { i } ) { \mathrm { ~ f o r ~ a l l ~ } } i = 1 , \dots , N .
$$

```latex
Algorithm 4 Learning the prediction matrix via ADMM
1: Input: $P ^ { a , 0 }$ satisfying $\textstyle \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { d } P _ { i j } ^ { a , 0 } = 1$ and $| P _ { i j } ^ { a , 0 } | \leq C$ for all $i , j , \rho > 0$ , and $T \geq 1$
2: Initialize: $t = 0$ and $P _ { i } ^ { 0 } = P ^ { a , \bar { 0 } } , \mu _ { i } ^ { 0 } \bar { = } 0 ,$ and $\begin{array} { r } { \lambda _ { i } ^ { 0 } = \frac { \mathbf { 1 } } { \lVert A _ { I _ { i } } ^ { \top } \mathbf { 1 } \rVert } \mathrm { ~ f o r ~ } i = 1 , \ldots , N . } \end{array}$
3: while $t < T$ do
4: $P ^ { a , t + 1 } \in$ argmin $\begin{array} { r } { \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { d } P _ { i j } = 1 , | P _ { i j } | \leq C } \end{array}$ for all $_ { i , j } L _ { \rho } ( \{ P _ { i } ^ { t } \} , \{ \lambda _ { i } ^ { t } \} , P , \{ \mu _ { i } ^ { t } \} )$
5: for $i \in \{ 1 , \ldots , N \}$ do
6: $\lambda _ { i } ^ { t + 1 } \stackrel {  } { \in } \mathrm { a r g m i n } _ { \lambda _ { i } \in \Lambda _ { i } } L _ { \rho } ( P _ { i } ^ { t } , \lambda _ { i } , P ^ { a , t + 1 } , \mu _ { i } ^ { t } ) .$
7: $\begin{array} { r } { P _ { i } ^ { t + 1 } \in \operatorname * { a r g m i n } _ { P _ { i } } \bar { L _ { \rho } } ( P _ { i } , \lambda _ { i } ^ { t + 1 } , P ^ { a , t + 1 } , \mu _ { i } ^ { t } ) . } \end{array}$
8: $\mu _ { i } ^ { t + 1 } = \mu _ { i } ^ { t } + \rho ( \dot { P _ { i } ^ { t + 1 } } - P ^ { a , t + 1 } ) .$
9: $t  t + 1 .$
10: Return: $P ^ { a , t } .$
```

Thus, $L _ { C }$ is an inverse feasibility loss. The loss can be discontinuous at a zero predicted cost.

For each observed vertex $w _ { i } ,$ we let $I _ { i } = \{ j : a _ { i } ^ { \top } w _ { i } = b _ { j } \}$ and let $A _ { I _ { i } }$ contain the corresponding rows of $A .$ Then, we have cone $( w _ { i } ) = \{ - A _ { I _ { i } } ^ { \top } \lambda _ { i } : \lambda _ { i } \geq 0 \}$ . Here, $\lambda _ { i } \geq 0$ contains the coeficients of the inward active normals $- a _ { j }$ for $j \in I _ { i }$ . For $P ^ { a } x _ { i } \neq 0$ , its angular loss can therefore be written as

$$
\begin{array} { r l } { L _ { C } ( P ^ { a } , P ^ { D M } ; x _ { i } ) = } & { \underset { \underset { \Vert \boldsymbol { A } _ { I _ { i } } ^ { \top } \lambda _ { i } \Vert = 1 } { \operatorname* { m i n } } } { \operatorname* { m i n } } \left( 1 + \frac { ( \boldsymbol { A } _ { I _ { i } } ^ { \top } \lambda _ { i } ) ^ { \top } P ^ { a } x _ { i } } { \Vert P ^ { a } x _ { i } \Vert } \right) . } \end{array}
$$

With a numerical bound $C \geq \frac { 1 } { n d }$ , we minimize the empirical loss subject to an entry-sum normalization and bounds on the matrix entries:

$$
\begin{array} { r l } { \operatorname* { m i n } _ { P ^ { a } , \lambda _ { 1 } , \dots , \lambda _ { N } } } & { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \bigg ( 1 + \frac { ( A _ { I _ { i } } ^ { \top } \lambda _ { i } ) ^ { \top } P ^ { a } x _ { i } } { \| P ^ { a } x _ { i } \| } \bigg ) } \\ { \mathrm { s . t . } \qquad } & { \lambda _ { i } \geq 0 , \| A _ { I _ { i } } ^ { \top } \lambda _ { i } \| = 1 , i = 1 , \dots , N , } \\ & { \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { d } P _ { i j } ^ { a } = 1 , } \\ & { | P _ { i j } ^ { a } | \leq C , i = 1 , \dots , n , j = 1 , \dots , d . } \end{array}\tag{6}
$$

In this formulation and the expressions below, the entire angular summand is defined to be zero when its predicted cost is zero, as in Eq. (5). The matrix constraints restrict the admissible predictors.

Introducing a local matrix $P _ { i }$ for each observation gives the consensus formulation as follows,

$$
\begin{array} { r l } { \operatorname* { m i n } _ { P ^ { a } , \{ P _ { i } , \lambda _ { i } \} _ { i = 1 } ^ { N } } } & { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( 1 + \frac { ( A _ { I _ { i } } ^ { \top } \lambda _ { i } ) ^ { \top } P _ { i } x _ { i } } { \| P _ { i } x _ { i } \| } \right) } \\ { \mathrm { s . t . ~ } ~ } & { \lambda _ { i } \geq 0 , ~ \| A _ { I _ { i } } ^ { \top } \lambda _ { i } \| = 1 , ~ i = 1 , \ldots , N , } \\ & { P _ { i } = P ^ { a } , ~ i = 1 , \ldots , N , } \\ & { \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { d } P _ { i j } ^ { a } = 1 , } \\ & { | P _ { i j } ^ { a } | \leq C , ~ i = 1 , \ldots , n , ~ j = 1 , \ldots , d . } \end{array}\tag{7}
$$

For $\rho > 0$ , the augmented Lagrangian function is defined as follows,

$$
L _ { \rho } ( \{ P _ { i } \} , \{ \lambda _ { i } \} , P ^ { a } , \{ \mu _ { i } \} ) = \sum _ { i = 1 } ^ { N } \left[ { \frac { 1 } { N } } \left( 1 + { \frac { ( A _ { I } ^ { \top } { \lambda _ { i } } ) ^ { \top } P _ { i } x _ { i } } { \| P _ { i } x _ { i } \| } } \right) + \langle \mu _ { i } , P _ { i } - P ^ { a } \rangle _ { F } + { \frac { \rho } { 2 } } \| P _ { i } - P ^ { a } \| _ { F } ^ { 2 } \right] ,
$$

where $\mu _ { i } \in \mathbb { R } ^ { n \times d }$ is the multiplier for $P _ { i } = P ^ { a }$ . The ADMM method in Algorithm 4 alternates between updates of $P ^ { a } , \lambda _ { i } ,$ and $P _ { i }$ , followed by the multiplier updates. In the local updates of Algorithm 4, $L _ { \rho } ( P _ { i } , \lambda _ { i } , P ^ { a } , \mu _ { i } )$ denotes the $i ^ { \mathrm { t h } }$ summand of the augmented Lagrangian function. We also write $\Lambda _ { i } = \{ \lambda _ { i } \in \mathbb { R } ^ { | I _ { i } | } : \lambda _ { i } \geq 0 , \| A _ { I _ { i } } ^ { \top } \lambda _ { i } \| = 1 \}$

The local minimizations can be approximated by first-order updates. For $P _ { i } ^ { t } x _ { i } \neq 0$ , a projected gradient step for $\lambda _ { i }$ and a gradient step for $P _ { i }$ are

$$
\begin{array} { r c l } { \lambda _ { i } ^ { t + 1 } } & { \in } & { \Pi _ { \Lambda _ { i } } \left( \lambda _ { i } ^ { t } - \frac { \alpha } { N } \frac { A _ { I _ { i } } P _ { i } ^ { t } x _ { i } } { \| P _ { i } ^ { t } x _ { i } \| } \right) , } \\ { P _ { i } ^ { t + 1 } } & { = } & { P _ { i } ^ { t } - \beta \left( \frac { 1 } { N } \left( \frac { A _ { I _ { i } } ^ { \top } \lambda _ { i } ^ { t + 1 } } { \| P _ { i } ^ { t } x _ { i } \| } - \frac { ( A _ { I _ { i } } ^ { \top } \lambda _ { i } ^ { t + 1 } ) ^ { \top } P _ { i } ^ { t } x _ { i } } { \| P _ { i } ^ { t } x _ { i } \| ^ { 3 } } P _ { i } ^ { t } x _ { i } \right) x _ { i } ^ { \top } + \mu _ { i } ^ { t } + \rho ( P _ { i } ^ { t } - P ^ { a , t + 1 } ) \right) , } \end{array}\tag{8}
$$

![](images/02e87f8b015f85a1beb3d03d3c9593c497b26aa2748faf77dddf4a867db0b6cd.jpg)  
(a) ADMM

![](images/d141e81e1e13e090278522d98e4ee4e4d006176c3776ceb96b0fcebd486c0da4.jpg)  
(b) Tang and Khalil’s method  
Figure 5: Convergence of the ADMM algorithm and Tang and Khalil’s method. The red line represents the optimization loss, defined as the proportion of incorrect decisions $w ^ { \star } ( P ^ { a } x _ { i } ) \neq w ^ { \star } ( P ^ { D M } x _ { i } )$ over the dataset.

Table 1: Comparison of three approaches for minimizing the CaVE loss.
<table><tr><td>Algorithm</td><td>CaVE loss</td><td>predLoss</td><td>optLoss</td></tr><tr><td>ADMM</td><td>0.0007018</td><td>0.0356</td><td>0.1450</td></tr><tr><td>Tang and Khalil&#x27;s method</td><td>0.0000032</td><td>0.0539</td><td>0.1150</td></tr><tr><td>Global</td><td>0.0000030</td><td>0.0378</td><td>0.0650</td></tr></table>

where $\alpha , \beta > 0$ are step sizes and $\Pi _ { \Lambda _ { i } }$ denotes Euclidean projection onto the feasible set $\Lambda _ { i } .$ . The projection may have multiple solutions, in which case any minimizer can be selected. These updates require nonzero loca predicted costs and ADMM is used as a numerical method without a global convergence guarantee.

The experiment shows that our solution approach can achieve near-zero optimality gap. Apart from ADMM, we tried the heuristic algorithm (Tang and Khalil, 2024) and the global solver in scipy package to solve the optimization problem. Figure 5 shows the convergence of ADMM and Tang and Khalil’s method. We observe that both methods achieve near zero CaVE loss. Table 1 compares all methods and shows that the global solver attains the smallest reported CaVE loss. Furthermore, if we encounter some context such that $\| P ^ { a } x _ { i } \| = 0$ , then we discard this context in practice.

## A.3 Sampling algorithms

We specify the sampling step of Algorithm 1. Indeed, for a full-rank matrix $P ^ { a }$ , the uniform sampling method in Algorithm 5 draws a direction uniformly from the unit sphere in $\mathbb { R } ^ { n }$ and chooses a context whose predicted cost has that direction. Since the context is chosen using $P ^ { a }$ , the sampling rule is adaptive, unlike a nonadaptive batch of contexts used by the inverse optimization benchmark. Uniform sampling is simple, and its sample size controls the number of observations added at each iteration. It does not target the decision boundaries of the downstream problem. To exploit this geometry, we use the boundary sampling method in Algorithm 6.

We enumerate the vertices as $V = \{ w _ { 1 } , \ldots , w _ { | V | } \}$ and write $C _ { k } = \mathrm { c o n e } ( w _ { k } )$ . For each vertex $w _ { k } \in V$ , we let $I _ { k } = \{ i \in \{ 1 , . . . , m \} : a _ { i } ^ { \top } w _ { k } = b _ { i } \}$ . Then, for each facet $i ,$ we define the set of incident vertex indices by $\mathcal { C } _ { i } = \left\{ k \in \left\{ 1 , \dots , | V | \right\} : i \in I _ { k } \right\}$ and set $K = \operatorname* { m a x } _ { 1 \leq k \leq | V | } \left| I _ { k } \right|$ . Using the inward directions $\begin{array} { r } { r _ { i } = - \frac { a _ { i } } { \left\| a _ { i } \right\| } } \end{array}$ , we have

$$
C _ { k } = \mathrm { p o l y c o n e } \{ r _ { i } : i \in I _ { k } \} , \quad \bigcap _ { k \in \mathcal { C } _ { i } } C _ { k } = \{ \alpha r _ { i } : \alpha \geq 0 \} .
$$

Figure 6 illustrates the facet-vertex incidences. For each facet i, Algorithm 6 chooses $d - n + 1$ linearly independent boundary contexts satisfying $P ^ { a } \bar { x } _ { i 0 j } = r _ { i }$ for all $j = 1 , \ldots , d - n + 1$ and sets $X _ { i } = [ \bar { x } _ { i 0 1 } , \dots , \bar { x } _ { i 0 , d - n + 1 } ]$ . These columns span the $( d - n + 1 )$ -dimensional linear space $\{ x \in \mathbb { R } ^ { d } : P ^ { a } x \in \mathrm { s p a n } ( r _ { i } ) \}$ . For $M \geq 1$ , each $k \in \mathcal { C } _ { i } .$ , and

![](images/869b376eff04206bf111f3b84816e5af5407ab8efa54f6352f2780e6abc86968.jpg)  
Figure 6: (a) the tight constraints (also the boundaries of normal cone) incident to the a normal cone $C _ { k } . \mathrm { ~ ( b ) }$ the normal cones (also the vertex) incident to the constraint $a _ { i }$

each $j = 1 , \ldots , d - n + 1$ , the associated near-boundary context is

$$
\begin{array} { r } { \tilde { x } _ { i k j } = \frac { M \bar { x } _ { i 0 j } + \sum _ { \ell \in I _ { k } \setminus \{ i \} } \bar { x } _ { \ell 0 j } } { \Vert M r _ { i } + \sum _ { \ell \in I _ { k } \setminus \{ i \} } r _ { \ell } \Vert } . } \end{array}
$$

The vector in the denominator is a positive combination of all active inward normals at $w _ { k }$ , so it belongs to int $\left( C _ { k } \right)$ and is nonzero. Thus, we have

$$
\begin{array} { r } { P ^ { a } \tilde { x } _ { i k j } = \frac { M r _ { i } + \sum _ { \ell \in I _ { k } \setminus \{ i \} } r _ { \ell } } { \| M r _ { i } + \sum _ { \ell \in I _ { k } \setminus \{ i \} } r _ { \ell } \| } \in \mathrm { i n t } ( C _ { k } ) . } \end{array}
$$

Thus, $w _ { k }$ is the unique optimizer under the attacker cost at $\tilde { x } _ { i k j }$ , and $P ^ { a } \tilde { x } _ { i k j }  r _ { i }$ as $M \to + \infty$ . The algorithm queries all boundary contexts and all of these near-boundary contexts.

In what follows, we describe the stopping condition mentioned in Theorem 3.9. Indeed, we define

$$
\eta _ { S } = \operatorname* { m i n } _ { 1 \leq k \leq | V | } \operatorname* { m i n } _ { v \in V \setminus \{ w _ { k } \} } \sum _ { \ell \in I _ { k } } r _ { \ell } ^ { \top } ( v - w _ { k } ) .
$$

Clearly, $\eta _ { S } \geq 0 .$ For $v \neq w _ { k }$ , at least one is strictly positive, because the active facet normals at $w _ { k }$ span $\mathbb { R } ^ { n }$ Thus, $\eta _ { S } > 0$ . For $M \geq 1 , k \in \mathcal { C } _ { i } ,$ and $v \neq w _ { k }$ , we have

$$
\begin{array} { r } { ( P ^ { a } \tilde { x } _ { i k j } ) ^ { \top } ( v - w _ { k } ) = \frac { ( M - 1 ) r _ { i } ^ { \top } ( v - w _ { k } ) + \sum _ { \ell \in I _ { k } } r _ { \ell } ^ { \top } ( v - w _ { k } ) } { \| M r _ { i } + \sum _ { \ell \in I _ { k } \setminus \{ i \} } r _ { \ell } \| } \geq \frac { \eta _ { S } } { M + K - 1 } . } \end{array}
$$

We impose the suficient stopping condition as follows,

$$
\Delta ( M + K - 1 ) < \eta _ { S } .\tag{9}
$$

If a near-boundary query returns any vertex other than $w _ { k }$ , its suboptimality loss under $P ^ { a }$ then exceeds $\Delta .$ , so Algorithm 1 cannot terminate at that sampling call.

Theorem A.3. Suppose Assumption 3.4 holds. For each $\Delta > 0$ in a family tending to zero, run Algorithm 1 with a full-rank initialization and Algorithm 6 with $M = M ( \Delta ) \geq 1$ . Assume that every inverse optimization update returns a full-rank matrix with zero inverse feasibility loss on the accumulated dataset. Suppose there exist constants $R > 0$ and $\kappa > 0$ , independent of the iteration and of $\Delta$ , such that every queried context satisfies $\| x \| \leq R$ and $\sigma _ { \operatorname* { m i n } } ( X _ { i } ) \geq \kappa$ for all $i = 1 , \ldots , m$ at every sampling call. Here, $\sigma _ { \operatorname* { m i n } } ( X _ { i } )$ denotes the smallest of the $d - n + 1$ singular values of $X _ { i }$ . Assume that $M ( \Delta ) \to + \infty$ as $\Delta \downarrow 0$ and that Eq. (9) holds for all suficiently small $\Delta$ . Then, every run terminates after at most $\begin{array} { r } { \left( 1 + \frac { 2 R L _ { e } } { \Delta } \right) ^ { n d } } \end{array}$ inverse optimization updates and its normalized output satisfies $\begin{array} { r } { P _ { M ( \Delta ) , \Delta } ^ { a } \to \frac { { P ^ { D M } } } { \parallel P ^ { D M } \parallel _ { F } } } \end{array}$ . As a consequence, we have $L _ { \mathrm { p r e d } } ( P _ { M ( \Delta ) , \Delta } ^ { a } , P ^ { D M } , x )  0$ for every x $\notin$ ker(P<sup>DM</sup>).

Algorithm 5 Uniform sampling (Unif)   
Input: A full-rank matrix $P ^ { a }$ and a positive integer $N _ { t }$   
Initialize: $D  \varnothing .$   
for $k = 1 . . . N _ { t }$ do   
Sample $\iota \sim { \mathcal { N } } ( 0 , I _ { n } )$ and set u $\begin{array} { r } {  \frac { u } { \| u \| } . } \end{array}$   
Choose $\boldsymbol { x } \in \mathbb { R } ^ { d }$ satisfying $P ^ { a } x = \underline { { u } } .$   
Query x and observe $w = w ^ { \star } ( P ^ { D M } x )$   
$\bar { D } \gets \mathsf { D } \cup \{ ( x , w ) \}$   
Return: D.

```latex
Algorithm 6 Boundary sampling (BndS)
Input: A full-rank matrix $P ^ { a }$ and a parameter $M \geq 1$
Initialize: $D \gets \emptyset$
for $i = 1 , \ldots , m$ do
Choose linearly independent $\bar { x } _ { i 0 1 } , \ldots , \bar { x } _ { i 0 , d - n + 1 }$ satisfying $P ^ { a } \bar { x } _ { i 0 , j } = r _ { i }$ for all $j = 1 , \ldots , d - n + 1 .$
for $j = 1 , \ldots , \stackrel { \cdot } { d } - n \stackrel { \cdot } { + } 1$ do
Query $\bar { x } _ { i 0 j }$ and observe $\bar { w } _ { i 0 j } = w ^ { \star } ( P ^ { D M } \bar { x } _ { i 0 j } )$
$\bar { D }  \bar { D } \cup \bar { \{ ( \bar { x } _ { i 0 j } , \bar { w } _ { i 0 j } ) \} }$
for $i = 1 , \ldots , m { \bf d o }$
for $j = 1 , \ldots , d - n + 1$ do
for $k \in \mathcal { C } _ { i }$ do
x˜<sub>ikj</sub> ← <sup>ℓ∈Ik\{i}</sup> <sub>∥Mri+Pℓ∈I \{i} rℓ∥</sub> . $\begin{array} { r } { \underset { : = : \infty } { \smile }  \frac { M \bar { x } _ { i 0 j } + \sum _ { \ell \in I _ { k } \setminus \{ i \} } \bar { x } _ { \ell 0 j } } { \pi _ { i } } } \end{array}$
Query $\tilde { x } _ { i k j }$ and observe $\tilde { w } _ { i k j } = w ^ { \star } ( P ^ { D M } \tilde { x } _ { i k j } )$
$\bar { D }  \bar { D } \cup \bar { \{ ( \tilde { x } _ { i k j } , \tilde { w } _ { i k j } ) \} }$
Return: D.
```

The bound on ∥x∥ yields a positive lower bound on the distance between distinct nonterminal iterates, which ensures finite termination. And the uniform lower bound $\sigma _ { \operatorname* { m i n } } ( X _ { i } ) \geq \kappa > 0$ ensures that the columns of $X _ { i }$ remain linearly independent in the limit. A suficient parameter choice is $M ( \Delta ) \to \infty$ and $\Delta M ( \Delta ) \to 0$ as $\Delta \downarrow 0 .$ . Indeed, these limits imply $\Delta ( M ( \Delta ) + K - 1 )  0$ , so Eq. (9) holds eventually. For example, $M ( \Delta ) = \operatorname* { m a x } \{ 1 , \Delta ^ { - 1 / 2 } \}$ satisfies these conditions. Theorem A.3 concerns this coupled limit. A smaller $\Delta$ increases the stated upper bound on the number of inverse optimization updates.

## A.4 Restricted context region

We consider the undefended interface with query contexts restricted to a set $\mathcal { X } \subseteq \mathbb { R } ^ { d }$ . The following result gives a suficient condition for identifying the prediction matrix from decisions observed in this region.

Lemma A.4. Let $J \subseteq \{ 1 , \dots , m \}$ be a set of facet indices such that $\{ a _ { i } : i \in J \}$ contains $n + 1$ normals satisfying Assumption $\ 3 . 4 \cdot$ Suppose that, for each $i \in J _ { i }$ , there exists ${ \bar { x } } _ { i } \in \operatorname { i n t } ( { \mathcal { X } } )$ with $P ^ { \bar { D } M } \bar { x } _ { i } = \alpha _ { i } r _ { i }$ for some $\alpha _ { i } > 0$ . If a nonzero matrix $P ^ { a }$ satisfies $w ^ { \star } ( P ^ { D M } x ) \in W ^ { \star } ( P ^ { a } x )$ for every $x \in \mathcal { X } \backslash$ ker $( P ^ { D M } ) , P ^ { a } = \gamma \dot { P } ^ { D M }$ for some $\gamma > 0$

For boundary sampling, fix a set $J \subseteq \{ 1 , \dots , m \}$ of facet indices. At each sampling call, choose the boundary representatives in Algorithm 6 for all facets, but query only $\bar { x } _ { i 0 j }$ and $\tilde { x } _ { i k j }$ for all $i \in J , j = 1 , \dots , d - n + 1$ and $k \in \mathcal { C } _ { i }$ . Every query in these selected blocks is retained and must lie in X. Boundary representatives for facets outside J may still be needed to form the near-boundary mixtures.

Lemma A.5. Suppose $\{ a _ { i } : i \in J \}$ contains n + 1 normals satisfying Assumption $\ 3 . 4 \cdot$ For each $\Delta > 0$ in a family tending to zero, run Algorithm 1 with a full-rank initialization and the restricted sampling rule above, using $M = M ( \Delta ) \geq 1$ . Assume every inverse optimization update returns a full-rank matrix with zero inverse feasibility loss on the accumulated dataset. Suppose there exist constants $R > 0$ and $\kappa > 0$ , independent of the iteration and of $\Delta _ { i }$ , such that every queried context and every boundary representative used to construct these queries has norm at most $R ,$ and $\sigma _ { \operatorname* { m i n } } ( X _ { i } ) \geq \kappa$ for all $i \in J$ at every sampling call. The norm bound includes auxiliary representatives for facets outside J. Assume that $M ( \Delta ) \to + \infty$ as $\Delta \downarrow 0$ and $\Delta ( M ( \Delta ) + K - 1 ) < \eta _ { S }$ for all suficiently small $\Delta .$ . Then, every run terminates after at most $\begin{array} { r } { \left( 1 + \frac { 2 R L _ { e } } { \Delta } \right) ^ { n d } } \end{array}$ inverse optimization updates and its normalized output satisfies $\begin{array} { r } { P _ { M ( \Delta ) , \Delta } ^ { a } \to \frac { { P ^ { D M } } } { \parallel P ^ { D M } \parallel _ { F } } } \end{array}$ . As a consequence, we have $L _ { \mathrm { p r e d } } ( P _ { M ( \Delta ) , \Delta } ^ { a } , P ^ { D M } , x )  0$ for every x /∈ ker $( P ^ { D M } )$

## A.5 Experiment setup

Synthetic instances. In the experiment, the decision-maker first learns the prediction model $P ^ { D M }$ by performing a linear regression on a randomly generated dataset $D = \{ x _ { i } , c _ { i } \}$ , where all coordinates of $x _ { i }$ are independently drawn from U[0, 20]. The responses satisfy $c _ { i } = P ^ { \mathrm { t r u e } } x _ { i } + \varepsilon$ , where $\varepsilon \sim \mathcal { N } ( 0 , I _ { n } )$ . Here, $P ^ { \mathrm { t r u e } }$ is a randomly generated ground-truth prediction matrix, unknown to the decision-maker, with $P _ { i j } ^ { \mathrm { t r u e } } \sim \mathcal { U } [ - 1 0 , 1 0 ]$ Given a query context x, the decision-maker will predict the cost vector $P ^ { D M } x$ and solve the downstream linear optimization problem m $\mathrm { i n } _ { w \in S } ( P ^ { D M } x ) ^ { \top } w$ . The downstream optimization problem is a fractional Knapsack problem:

$$
\begin{array} { c l } { \operatorname* { m i n } _ { w \in \mathbb { R } ^ { n } } } & { ( P ^ { D M } x ) ^ { \top } w } \\ { \mathrm { s . t . } } & { a ^ { \top } w \leq b , \ \mathbf { 0 } \leq w \leq \mathbf { 1 } . } \end{array}\tag{10}
$$

where $a \in \mathbb { R } ^ { n }$ is a vector whose entries start at 1 and increase by 0.5 successively and $\textstyle b = 0 . 6 \sum _ { i = 1 } ^ { n } a _ { i }$ . We use the fractional Knapsack problem because it models resource allocation decisions, including those arising in recommendation systems. The number of vertices can grow exponentially with n, which afects the size of a complete boundary sampling call. The optimization problem in Eq. (2), however, is a linear program using the inequality representation of the polytope and does not require vertex enumeration.

We compare boundary sampling (BndS), uniform sampling (Unif), and a nonadaptive inverse optimization baseline (INV). Unif and BndS alternate sampling and inverse optimization as in Algorithm 1, using the sampling constructions in Algorithms 5 and 6, respectively. INV performs one inverse optimization fit on a nonadaptive batch of contexts, with sample sizes matched in the reported comparisons. Unif draws predicted cost directions uniformly from the unit sphere in $\mathbb { R } ^ { n }$ and maps them to contexts using the current attacker matrix. The INV batch is selected without adapting to an updated attacker matrix. Unless otherwise specified, all three methods use the suboptimality-loss formulation in Eq. (2).

For the attack comparisons, we perform 100 repetitions, using a diferent deployed matrix $P ^ { D M }$ in each repetition.   
We report the median prediction loss and the interval between its 25th and 75th percentiles.

In the query-budget experiment, we use $d = 1 0$ and $n = 5$ , with 100 queries per iteration. MIX allocates 50 queries to boundary sampling and 50 to uniform sampling. Each parameter setting is repeated five times, and we report the median prediction loss. Here, BndS and the boundary component of MIX use a budgeted variant of Algorithm 6 rather than its complete query set, while Theorem A.3 is proved for the complete construction.

For each defensive parameter setting, we perform 100 repetitions and record the attacker’s prediction loss and the decision-maker’s regret. The parameter is ϵ for decision perturbation and σ for prediction perturbation. Each point in a trade-of plot reports the median regret and median prediction loss over the repetitions, with the 25th and 75th percentiles of prediction loss shown as an interval.

All experiments were conducted on a machine with two AMD EPYC 7643 48-core processors, with each job allocated one CPU core and 16 GB of memory.

Real data. The KuaiRand-1K dataset (Gao et al., 2022) contains 1000 users and approximately four million videos. Its standard policy portion contains 11713045 interactions. We use the records of 800 users for training and the remaining users for testing. Each user is represented by a feature vector of dimension $d = 1 8$ . We cluster the videos into $n = 5$ types and then estimate each user’s click probability for every video type from the interaction logs. Each log entry records the interaction between a specific user and a specific video. Using the cluster label of the video and the observed interaction, we update the user’s click-through rate for the corresponding video type. The platform fits a linear predictor of click-through rates and defines $P ^ { D \tilde { M } }$ as the negative of its coeficient matrix. Thus, the predicted click-through-rate vector is ${ } _ { - P ^ { D M } x }$ . With $\begin{array} { r } { k = \left\lceil \frac { n } { 2 } \right\rceil } \end{array}$ , the recommendation is obtained from

$$
\begin{array} { r l } { \operatorname* { m i n } _ { w \in \mathbb { R } ^ { n } } } & { ( P ^ { D M } x ) ^ { \top } w } \\ { \mathrm { s . t . } } & { \sum _ { i = 1 } ^ { n } w _ { i } \leq k , \ \mathbf { 0 } \leq w \leq \mathbf { 1 } . } \end{array}\tag{11}
$$

Prediction loss is evaluated using the attacker and deployed cost vectors, and regret is measured under the fitted cost vector $P ^ { D M } \mathcal { _ { 3 } }$ x rather than realized clicks.

![](images/33eaafd142d71729a8d261c51cd1b810b909612fafe03fbb8a536c2b2f803a72.jpg)

![](images/2161f2a616e049676fdb0d86f9c33d2247834ebfc0b847f6a033bbea930effdb.jpg)  
Figure 7: The out-of-sample prediction loss and iterations for the attacking framework in diferent attacker’s parameter settings.

![](images/40065d76e58816d0aa37961c540c7dce5e97c01b4059cf2a0f53c4e451379212.jpg)  
(a)

![](images/6cb9c727f47e29d62e66265e7d16d7eb39d5b167fb8ea910c99fd975de3e646d.jpg)  
(b)

![](images/697a1f12d6dced0fbb6a73172bf77aae41558cfe73186ce84d1fae13846c3314.jpg)  
(c)  
Figure 8: (a) Prediction loss; (b) Sample size for BndS and Unif against MLP and SVR predictors fitted to the linear synthetic data; and (c) Prediction loss of a linear attacker against MLP and SVR predictors as the quadratic coeficient γ varies.

## A.6 Additional experimental results

Attacker’s coeficient. Figure 7 compares prediction loss and the number of iterations for diferent values of M and $\Delta .$ . In the tested settings, larger ∆ is generally associated with fewer iterations and greater prediction loss. For fixed M, both quantities change little once ∆ is suficiently small. The condition $\Delta ( M + K - 1 ) < \eta _ { S }$ is suficient for the stopping argument in Theorem A.3 but the figure does not establish that it is a sharp threshold.

Proposition 3.7 bounds the number of inverse optimization updates rather than specifying its realized value. With a common query-norm bound R, the bound has no explicit dependence on M, although the observed number of iterations can depend on M. Prediction recovery in Theorem A.3 concerns a coupled limit in M and ∆ under uniform sampling regularity.

Attacker’s robustness to misspecification. On the synthetic dataset, we evaluate the linear attack against two nonlinear deployed model classes: a multilayer perceptron (MLP) with one hidden layer and ReLU activation, and support vector regression (SVR) with a Gaussian kernel. The hyperparameters, including hidden-layer width and kernel bandwidth, are selected by cross-validation.

Figure 8a reports prediction-loss values below 0.01 for the evaluated linear attacks against both deployed models while Figure 8b reports their sample sizes. These results show that a linear attacker can approximate the deployed prediction directions in these experiments even when the decision-maker uses a nonlinear model class.

![](images/8e8db8b2d07c1e461c520d2184c9fbbbb1b8e0abc1cd95fdbd222bfac53d8f6c.jpg)  
(a) Prediction Loss

![](images/a915a08e0e751e8ba54aa3eeae15a068a899a52a9a13635c617fa6e2e2006b2f.jpg)  
(b) Time

Figure 9: Prediction loss (a) and runtime per iteration (b) of Unif on fractional Knapsack instances of diferent dimensions.  
![](images/53057a3e858fec79fec81f321ea0aa4a0b874c446ebfb8a3e8585f9ab36cd4b6.jpg)  
Figure 10: Security-performance trade-of under the fractional Knapsack problem with $n = 2 0$ , using at most three neighboring vertices per query.

We also evaluate the linear attacker on data with a quadratic component:

$$
c = P ^ { \mathrm { t r u e } } x + \gamma B \phi _ { 2 } ( x ) + \epsilon .
$$

Here, $\phi _ { 2 } ( x )$ collects all quadratic terms of $x , P _ { i j } ^ { \mathrm { t r u e } } \sim \mathcal { U } [ - 1 0 , 1 0 ] , B _ { i j }$ ∼ Bernoulli(0.5), and ϵ is a zero-mean Gaussian noise vector. The decision-maker fits the MLP or SVR model to these data, while the attacker retains its linear model class. Figure 8c shows generally larger prediction loss for larger values of γ over the tested range.

Large-scale optimization problem. A call to Algorithm 6 makes $\begin{array} { r l } {  { ( d - n + 1 ) ( m + \sum _ { k = 1 } ^ { | V | } | I _ { k } | ) } } \end{array}$ query calls, counting repeated contexts separately. This count can grow exponentially with n through the number of vertices. Context construction, oracle evaluations, and inverse optimization also contribute to the computational cost. The cost of Algorithm 3 likewise depends on how neighboring vertices and their angular eligibility are computed. We evaluate uniform sampling and a restricted-neighbor defense on larger fractional Knapsack instances.

To control the number of queries in larger problems, we use uniform sampling on the fractional Knapsack problem in Eq. (10), with $d = 2 n$ and N = 5n new queries per iteration. The number of new queries is therefore linear in n, while the size of the inverse-fitting problem depends on the accumulated dataset. This choice does not by itself establish a polynomial bound on total attack time. Figure 9a reports small prediction losses without defense, and Figure 9b reports increasing time per iteration as n grows.

![](images/7c804c668a52cdbe94c408c259ff3c0f62eea98ae9f0c7ac018fd5fdf00811ab.jpg)

![](images/b6e7bbf7bb58d51c101ad489a549066ffd51023f497ac95a94821499d22fd215.jpg)  
Figure 11: Prediction loss (left) and decision-maker’s regret (right) as functions of the defensive parameter ϵ.

For the defender, we evaluate decision perturbation restricted to at most three neighboring vertices per query against a uniform-sampling attacker on fractional Knapsack instances with $n = 2 0$ . Figure 10 shows a securityperformance trade-of for this implementation. The regret bound in Proposition 4.1 remains valid provided every selected neighbor passes the angular eligibility test in Algorithm 3. The expected prediction error bound in Theorem 4.2 assumes that Algorithm 3 includes every neighboring vertex satisfying the angular eligibility condition..

Prediction loss and regret on defensive parameter. Figure 11 shows that prediction loss and regret generally increase over the tested values of ϵ. This fits our theoretical result. Proposition 4.1 gives a regret upper bound linear in ϵ, while Theorem 4.2 gives a nondecreasing lower bound on expected prediction loss under its population-fitting condition.

## B PROOF

Proof of Proposition 2.2. Suppose that $s ( x ) = - p ( x )$ is the class utility vector. Then, the classifier selects a label by finding $i ( x ) \in \mathrm { a r g m i n } _ { 1 \leq i \leq n } p _ { i } ( x )$ . For every $w \in S$ , we have

$$
w ^ { \top } p ( x ) = \sum _ { i = 1 } ^ { n } w _ { i } p _ { i } ( x ) \geq \operatorname* { m i n } _ { i } p _ { i } ( x ) ,
$$

with equality at $e _ { i ( x ) }$ exactly when $i ( x ) \in \mathrm { a r g m i n } _ { 1 < i < n } p _ { i } ( x )$ . Matching the tie-breaking rules therefore identifies each returned class label $i ( x )$ with the returned vertex $e _ { i ( x ) }$ , and conversely.

Proof of Lemma 3.3. The key is to show that calibration is independent of the particular inverse feasibility loss. Indeed, we define the set of globally decision consistent matrices by

$$
\mathcal { F } = \{ P \in \mathbb R ^ { n \times d } \setminus \{ 0 \} : w ^ { \star } ( P ^ { D M } x ) \in W ^ { \star } ( P x ) \mathrm { ~ f o r ~ e v e r y ~ } x \notin \ker ( P ^ { D M } ) \} ,
$$

and the set of matrices with globally correct prediction directions by

$$
{ \mathcal { G } } = \{ P \in \mathbb { R } ^ { n \times d } \setminus \{ 0 \} : L _ { \mathrm { p r e d } } ( P , P ^ { D M } ; x ) = 0 { \mathrm { ~ f o r ~ e v e r y ~ } } x \not \in \ker ( P ^ { D M } ) \} .
$$

For any inverse feasibility loss $L ,$ we obtain that $L ( P , P ^ { D M } ; x ) = 0$ if and only if $w ^ { \star } ( P ^ { D M } x ) \in W ^ { \star } ( P x )$ for every nonzero matrix P and every context x. This implies that all inverse feasibility losses have the same global zero loss set ${ \mathcal F } .$ Then, we establish the relationship between $\mathcal { F }$ and G.

We claim that ${ \mathcal { G } } \subseteq { \mathcal { F } }$ . Indeed, we take any $P \in { \mathcal { G } }$ and x $\notin \ker ( P ^ { D M } )$ . Since $P ^ { D M } x \neq 0$ and $L _ { \mathrm { p r e d } } ( P , P ^ { D M } ; x ) = 0$ we have $P x = \gamma _ { x } P ^ { D M } x$ for some $\gamma _ { x } > 0$ . This implies $\dot { W } ^ { \star } ( P \dot { x } ) = \dot { W } ^ { \star } ( P ^ { D M } x )$ and $w ^ { \star } ( P ^ { D M } \bar { x } ) \in W ^ { \star } ( P x )$ . Since this holds at every nonkernel context, $P \in { \mathcal { F } }$ . This yields the desired claim.

It sufices to consider two cases. If ${ \mathcal { F } } = { \mathcal { G } }$ , the calibration holds for every inverse feasibility loss. Otherwise, there exists a matrix $P _ { \mathrm { b a d } } \in \mathcal { F } \backslash \mathcal { G } . \mathrm { B y }$ definition, we have $( \mathrm { i } ) \ L ( P _ { \mathrm { b a d } } , P ^ { D M } ; x ) = 0$ for every $x \notin \ker ( P ^ { D M } )$ and (ii) there exists a context x<sub>0</sub> $\notin$ ker $( P ^ { D M } )$ such that $L _ { \mathrm { p r e d } } ( P _ { \mathrm { b a d } } , P ^ { D M } ; x _ { 0 } ) > 0 $ , for every inverse feasibility loss $L$ . In this case, none of the inverse feasibility losses calibrates the prediction loss.

We now show that calibration fails for an n-dimensional hypercube. Indeed, for every $w \in S$ , we have $w = w _ { 0 } + \ell U z$ where $w _ { 0 } \in \mathbb { R } ^ { n } , \ell > 0 , z \in [ 0 , 1 ] ^ { n }$ and U is an orthogonal matrix. This implies that minimizing $c ^ { \top }$ w over S is equivalent to minimizing $( U ^ { \top } c ) ^ { \top } z$ over $[ 0 , 1 ] ^ { n }$ . This problem separates coordinate-wise: an optimal $z ^ { \star }$ satisfies

$$
z _ { i } ^ { \star } \in \left\{ \begin{array} { l l } { \{ 0 \} , } & { \mathrm { i f ~ } ( U ^ { \top } c ) _ { i } > 0 , } \\ { \{ 1 \} , } & { \mathrm { i f ~ } ( U ^ { \top } c ) _ { i } < 0 , \mathrm { ~ f o r ~ a l l ~ } i = 1 , 2 , \ldots , n . } \\ { { \left[ 0 , 1 \right] } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

We define $D = \operatorname { d i a g } ( 1 , \dots , 1 , 2 )$ and $P ^ { a } = U D U ^ { \top } P ^ { D M }$ . Since $U D U ^ { \top }$ is invertible and $P ^ { D M }$ has full row rank, $P ^ { a }$ has full row rank and is an admissible nonzero attacker matrix. For every context x, we have $U ^ { \top } P ^ { a } x = D U ^ { \top } P ^ { D M } x$ Since all diagonal entries of D are positive, these transformed cost vectors have identical coordinate-wise signs. The preceding characterization of the optimal solution set therefore gives $W ^ { \star } ( P ^ { a } x ) = W ^ { \star } ( P ^ { D M } x )$ for every $\boldsymbol { x } \in \mathbb { R } ^ { d }$ . In particular, every decision returned by the deployed model is optimal under the attacker cost. Thus, every inverse feasibility loss satisfies $L ( P ^ { a } , P ^ { D M } ; x ) = 0$ for every $x \in \mathbb { R } ^ { d }$

Since $P ^ { D M }$ has full row rank and $n \geq 2$ , there exists $x _ { 0 } \in \mathbb { R } ^ { d }$ such that $P ^ { D M } x _ { 0 } = U ( e _ { 1 } + e _ { n } )$ . By definition, we have $P ^ { a } x _ { 0 } = U ( e _ { 1 } + 2 e _ { n } )$ . Putting these pieces together yields

$$
\begin{array} { r } { L _ { \mathrm { p r e d } } ( P ^ { a } , P ^ { D M } ; x _ { 0 } ) = \angle ( e _ { 1 } + 2 e _ { n } , e _ { 1 } + e _ { n } ) = \operatorname { a r c c o s } \Big ( \frac { 3 } { \sqrt { 1 0 } } \Big ) > 0 . } \end{array}
$$

This implies that $x _ { 0 } \notin$ ker $\cdot ( P ^ { D M } )$ and $P ^ { a } \in { \mathcal { F } } \backslash { \mathcal { G } }$ . Therefore, we conclude that no inverse feasibility loss calibrates the prediction loss on an n-dimensional hypercube. □

For the remaining proofs, we let

$$
\begin{array} { r } { r _ { i } = - \frac { a _ { i } } { \| a _ { i } \| } , \quad C _ { k } = \mathrm { c o n e } ( w _ { k } ) , \quad I _ { k } = \{ i : a _ { i } ^ { \top } w _ { k } = b _ { i } \} , \quad \mathcal { C } _ { i } = \{ k : i \in I _ { k } \} , } \end{array}
$$

where $I _ { k }$ indexes the facets containing $w _ { k } , \mathcal { C } _ { i }$ indexes the vertices of facet $i ,$ and $C _ { k }$ is closed, full-dimensional and pointed. In our analysis, we will use three facts: ${ \mathrm { ( i ) ~ } } c \in \operatorname { i n t } ( C _ { k } ) \Leftrightarrow W ^ { \star } ( c ) = \{ w _ { k } \} , { \mathrm { ( i i ) ~ } } C _ { k } = \mathrm { p o l y c o n e } \{ r _ { i } : i \in I _ { k } \}$ and $( { \mathrm { i i i } } ) \cap _ { k \in { \mathcal { C } } _ { i } } C _ { k } = \{ \alpha r _ { i } : \alpha \geq 0 \}$ . The last fact follows since a cost vector makes every point of facet i optimal if and only if it is a nonnegative multiple of the inward normal $r _ { i }$

Proof of Theorem 3.5. Suppose that $P ^ { a } \neq 0$ satisfies $w ^ { \star } ( P ^ { D M } x ) \in W ^ { \star } ( P ^ { a } x )$ for every $x \notin \ker ( P ^ { D M } )$ . Then, we show that ker $( P ^ { D M } ) \subseteq \ker ( P ^ { a } )$ . Indeed, we $\mathrm { f i x } \ z \in \ker ( { \dot { P } } ^ { D M } )$ , choose $c _ { 0 } \in \mathrm { i n t } ( C _ { k } )$ for some vertex $w _ { k }$ , and take $x _ { 0 }$ satisfying $P ^ { D M } x _ { 0 } = c _ { 0 }$ . Since $P ^ { D M }$ has full row rank, such $x _ { 0 }$ exists. Then, we have $P ^ { D M } ( x _ { 0 } + t z ) = c _ { 0 } \neq 0$ for every $t \in \mathbb { R }$ and hence $W ^ { \star } ( c _ { 0 } ) = \{ w _ { k } \}$ . The consistency condition implies $P ^ { a } ( x _ { 0 } + t z ) \in C _ { k }$ for every $t \in \mathbb { R }$ Since $C _ { k }$ is a closed cone, $P ^ { a } z , - P ^ { a } z \in C _ { k }$ . Since $C _ { k }$ is pointed, $P ^ { a } z = 0$ . Thus, we have ker $( P ^ { D M } ) \subseteq \ker ( P ^ { a } )$

Since $P ^ { D M }$ has full row rank and ker $( P ^ { D M } ) \subseteq \ker ( P ^ { a } )$ , we can write $P ^ { a } = B P ^ { D M }$ for a linear map $B : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ It sufices to show that $B = \gamma I _ { n }$ for some $\gamma > 0$ . Indeed, for any $c \in \operatorname { i n t } ( C _ { k } )$ , we choose x with $P ^ { D M } x = c$ Consistency gives $w _ { k } \in W ^ { \star } ( B c )$ , so $B c \in C _ { k }$ . Since $C _ { k } = \overline { { \operatorname* { i n t } ( C _ { k } ) } }$ , continuity gives $B C _ { k } \subseteq C _ { k }$ for every k. This implies that, for each facet i, we have $\begin{array} { r } { B r _ { i } \in \bigcap _ { k \in \mathcal { C } _ { i } } C _ { k } = \{ \alpha r _ { i } : \alpha \geq 0 \} } \end{array}$ . Thus, we have $B a _ { i } = \lambda _ { i } a _ { i }$ for some $\lambda _ { i } \geq 0$ . Applying B to both sides of $\begin{array} { r } { a _ { k _ { n + 1 } } = \sum _ { j = 1 } ^ { n } \beta _ { j } a _ { k _ { j } } } \end{array}$ in Assumption 3.4 yields $\begin{array} { r } { \sum _ { j = 1 } ^ { n } \beta _ { j } \big ( \lambda _ { k _ { j } } - \lambda _ { k _ { n + 1 } } \big ) a _ { k _ { j } } = 0 } \end{array}$ Linear independence and $\beta _ { j } \neq 0$ give $\lambda _ { k _ { j } } = \lambda _ { k _ { n + 1 } }$ for every $j = 1 , \dots , n .$ Since $a _ { k _ { 1 } } , \ldots , a _ { k _ { n } }$ form a basis of R<sup>n</sup>, we have $B = \gamma I _ { n }$ , where $\gamma = \lambda _ { k _ { n + 1 } } \geq 0$ . As $P ^ { a } \neq 0$ , we have $\gamma > 0$ and $P ^ { a } = \gamma P ^ { D M }$ □

Proof of Proposition 3.6. The proof of Theorem 3.5 shows that global decision consistency alone gives a linear map $B : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ satisfying $P ^ { a } = B P ^ { D M } , B r _ { i } = \lambda _ { i } r _ { i }$ and $\lambda _ { i } \geq 0$ for every i. Since rank $( P ^ { a } ) = n$ , the relation $P ^ { a } = B P ^ { D M }$ implies that B is invertible. Since $r _ { i } \neq 0$ , we have $\lambda _ { i } > 0$ for every i.

Suppose that x¯ is the boundary context, so that $P ^ { a } \bar { x } = \alpha r _ { i }$ for some facet i and $\alpha > 0$ . Then, by using $B ^ { - \bar { 1 } } r _ { i } = \lambda _ { i } ^ { - 1 } r _ { i }$ , we have

$$
\begin{array} { r } { P ^ { D M } \bar { x } = B ^ { - 1 } P ^ { a } \bar { x } = \frac { \alpha } { \lambda _ { i } } r _ { i } = \frac { 1 } { \lambda _ { i } } P ^ { a } \bar { x } . } \end{array}
$$

This implies that the two costs are nonzero and positively aligned. Thus, we have $L _ { \mathrm { p r e d } } ( P ^ { a } , P ^ { D M } ; \bar { x } ) = 0 $ □

Proof of Proposition 3.7. Since $\| \cal P ^ { 0 } \| _ { F } = 1$ and every update is normalized, we have $\| \boldsymbol { P } ^ { t } \| _ { F } = 1$ for every t. By the zero-loss assumption and invariance of inverse feasibility under positive scaling, each $P ^ { s } , s \geq 1$ , is inverse feasible on $D ^ { s - 1 }$ . At any nonterminal iteration t, there exist $( x _ { t } , w _ { t } ) \in D ^ { t }$ and $v _ { t } \in V$ such that

$$
( P ^ { t } x _ { t } ) ^ { \top } ( w _ { t } - v _ { t } ) > \Delta .
$$

For every subsequent iterate $P ^ { t ^ { \prime } }$ , we have $D ^ { t } \subseteq D ^ { t ^ { \prime } - 1 }$ . This implies $( P ^ { t ^ { \prime } } x _ { t } ) ^ { \top } ( w _ { t } - v _ { t } ) \leq 0$ . Thus, we have

$$
\Delta < ( ( P ^ { t } - P ^ { t ^ { \prime } } ) x _ { t } ) ^ { \top } ( w _ { t } - v _ { t } ) \leq \| P ^ { t } - P ^ { t ^ { \prime } } \| _ { F } \| x _ { t } \| \| w _ { t } - v _ { t } \| \leq R L _ { e } \| P ^ { t } - P ^ { t ^ { \prime } } \| _ { F } .
$$

Thus, any two nonterminal iterates are separated by more than $\frac { \Delta } { R L _ { e } }$ in Frobenius norm.

For any finite collection of T nonterminal iterates, the open Frobenius balls of radius $\frac { \Delta } { 2 R L _ { e } }$ centered at these iterates are disjoint. Since every iterate has unit Frobenius norm, these balls lie in the ball centered at zero with radius $\begin{array} { r } { 1 + \frac { \Delta } { 2 R L _ { e } } } \end{array}$ . Then, we have

$$
\begin{array} { r } { T \left( \frac { \Delta } { 2 R L _ { e } } \right) ^ { n d } \leq \left( 1 + \frac { \Delta } { 2 R L _ { e } } \right) ^ { n d } , } \end{array}
$$

which implies

$$
\begin{array} { r } { T \leq \left( 1 + \frac { 2 R L _ { e } } { \Delta } \right) ^ { n d } . } \end{array}
$$

Each nonterminal iteration performs one inverse optimization update, and each sampling call is finite. Therefore, Algorithm 1 terminates after at most $( 1 + \textstyle \frac { 2 R L _ { e } } { \Delta } ) ^ { { \bar { n } } d }$ inverse optimization updates. □

Lemma B.1. Let $P ^ { a }$ have full row rank, and let $\bar { x } \in \mathbb { R } ^ { d }$ satisfy $P ^ { a } \bar { x } = \alpha r _ { i }$ for some facet i and $\alpha > 0$ . Suppose that, for some $\epsilon > 0$ , we have $w ^ { \star } ( P ^ { D M } x ) \in W ^ { \star } ( P ^ { a } x )$ for every $x \in \mathbb { R } ^ { d }$ such that $\mathbf { \bar { \psi } } P ^ { D M } x \neq 0 , \ P ^ { a } x \neq 0$ and $\angle ( P ^ { a } x , r _ { i } ) < \epsilon$ . Then, we have

$$
P ^ { D M } \bar { x } \in \{ \alpha r _ { i } : \alpha \geq 0 \} .
$$

In particular, if $P ^ { D M } \bar { x } \neq 0$ , we have $L _ { \mathrm { p r e d } } ( P ^ { a } , P ^ { D M } ; \bar { x } ) = 0 .$

Proof. If $P ^ { D M } \bar { x } = 0 .$ , the conclusion is immediate. Otherwise, we fix $k \in \mathcal { C } _ { i }$ . Since $P ^ { a }$ has full row rank, we choose $h _ { k } \in \mathbb { R } ^ { d }$ with $P ^ { a } h _ { k } \in \mathrm { i n t } ( C _ { k } )$ and define $x _ { t } = \bar { x } + t h _ { k }$ for any $t > 0$ . Then, we have $x _ { t } \to \bar { x }$ as $t \downarrow 0$ and

$$
P ^ { a } x _ { t } = \alpha r _ { i } + t P ^ { a } h _ { k } \in \mathrm { i n t } ( C _ { k } ) .
$$

For suficiently small $t > 0$ , we have $P ^ { D M } x _ { t } \neq 0$ and $\angle ( P ^ { a } x _ { t } , r _ { i } ) < \epsilon$ . Consistency guarantees $w ^ { \star } ( P ^ { D M } x _ { t } ) \in$ $W ^ { \star } ( P ^ { a } x _ { t } ) = \{ w _ { k } \}$ so we have $P ^ { D M } x _ { t } \in C _ { k }$ . Closedness guarantees $P ^ { D M } \bar { x } \in C _ { k }$ which holds for every $k \in \mathcal { C } _ { i }$ Thus, we have

$$
{ \cal P } ^ { D M } \bar { x } \in \bigcap _ { k \in \mathcal { C } _ { i } } { \cal C } _ { k } = \{ \alpha r _ { i } : \alpha \geq 0 \} .
$$

If $P ^ { D M } \bar { x } \neq 0$ , this implies that $P ^ { D M } \bar { x }$ is positively aligned with $P ^ { a } \bar { x } = \alpha r _ { i }$ and $L _ { \mathrm { p r e d } } ( P ^ { a } , P ^ { D M } ; \bar { x } ) = 0 $ □

Proof of Proposition 3.8. Suppose that no ϵ-near-boundary context belongs to $I ( P ^ { a } , P ^ { D M } )$ . Then, by using the definition of $\bar { I } ( P ^ { a } , P ^ { D M } )$ , every such context has zero suboptimality loss. Thus, we have $w ^ { \star } ( P ^ { D M } x ) \in W ^ { \star } ( P ^ { a } x )$ whenever $P ^ { a } x \neq 0$ and $\angle ( P ^ { a } x , r _ { i } ) < \epsilon$ for some facet i. By Lemma B.1, we have

$$
P ^ { D M } \bar { x } \in \{ \alpha r _ { i } : \alpha \geq 0 \} \mathrm { w h e n e v e r } P ^ { a } \bar { x } = r _ { i } .
$$

This implies that $P ^ { D M } \in \mathcal { P } _ { c } ( P ^ { a } )$ . By Assumption 3.4 and Proposition A.1, we have $P ^ { D M } = \gamma P ^ { a }$ for some $\gamma \in \mathbb { R }$ The displayed ray membership and $P ^ { D M } \neq 0 \ \mathrm { i m p l y } \ \gamma > 0$ . This implies that the prediction loss is zero at every context, yielding a contradiction. Therefore, an ϵ-near-boundary context in ${ \cal I } ( P ^ { a } , P ^ { D M } )$ must exist. □

Proof of Theorem 3.9. Theorem A.3 shows finite termination and $\begin{array} { r } { P _ { M ( \Delta ) , \Delta } ^ { a } \to \frac { { P } ^ { D M } } { \| P ^ { D M } \| _ { F } } } \end{array}$ . For every $x \notin \ker ( P ^ { D M } )$ 2 the limiting cost is nonzero. Therefore, the continuity of angular distance gives the desired result. □

Proof of Proposition $4 . 1 .$ . Fix a cache realization and a context $x ,$ and write $c = P ^ { D M } x , w _ { 0 } = w ^ { \star } ( c )$ and $w = \tilde { w } ( x )$ If $c = 0 ~ \mathrm { o r } ~ w = w _ { 0 }$ , the regret is zero. Otherwise, Algorithm 3 gives $\begin{array} { r } { \angle ( c , \mathrm { c o n e } ( w ) ) \le \epsilon . } \end{array}$ The intersection of cone(w) with the unit sphere is nonempty and compact. We choose a unit vector $u \in \mathrm { c o n e } ( w )$ satisfying

$$
\angle ( c , u ) = \angle ( c , \mathrm { c o n e } ( w ) ) \leq \epsilon .
$$

Since w is optimal under $u ,$ we have $u ^ { \top } ( w - w _ { 0 } ) \leq 0$ . Thus, we have

$$
\begin{array} { r } { c ^ { \top } ( w - w _ { 0 } ) \leq \left\| c \right\| \left( \frac { c } { \left\| c \right\| } - u \right) ^ { \top } ( w - w _ { 0 } ) \leq \left\| c \right\| \left\| w - w _ { 0 } \right\| \left\| \frac { c } { \left\| c \right\| } - u \right\| \leq 2 L _ { e } \left\| c \right\| \sin \left( \frac { \angle ( c , u ) } { 2 } \right) \leq L _ { e } \epsilon \left\| c \right\| . } \end{array}
$$

Taking expectation under $x \sim Q$ yields

$$
\begin{array} { r } { \mathbb { E } _ { { x } \sim { Q } } [ \mathrm { R e g r e t } ( \tilde { w } ( x ) ; x ) ] \le L _ { e } \epsilon \mathbb { E } _ { { x } \sim { Q } } [ \| P ^ { D M } x \| ] . } \end{array}
$$

This completes the proof.

Proof of Theorem $4 . 2 .$ Fix a cache realization and a context x with nonzero costs, and write $c = P ^ { D M } x , c _ { a } = P ^ { a } x .$ $w _ { 0 } = w ^ { \star } ( c )$ and $w _ { a } = w ^ { \star } ( c _ { a } )$ . The boundary of a vertex normal cone is the union of its facets shared with neighboring cones. Thus, at a cost with a unique optimizer, its margin equals its angular distance to the boundary of its normal cone. At a tied cost, this distance is zero for every optimal vertex.

Suppose that $\tilde { w } ( x ) \in W ^ { \star } ( c _ { a } )$ and $c \in { \mathcal { E } }$ . Then, we have $\tilde { w } ( x ) \ne w _ { 0 }$ . This implies that $\mathrm { c o n e } ( w _ { 0 } )$ and $\mathrm { c o n e } ( \tilde { w } ( x ) )$ have disjoint interiors. We consider a shortest spherical path from $\frac { c } { \| c \| }$ to $\frac { c _ { a } } { \| c _ { a } \| }$ . If c has a unique optimizer, the initial segment before leaving the interior of $\mathrm { c o n e } ( w _ { 0 } )$ has length at least $\psi ( x )$ . If $c _ { a }$ has a unique optimizer, the final segment after its last entry into the interior of cone $( \tilde { w } ( x ) )$ has length at least $\phi ( x )$ . At a tied endpoint, we take the corresponding segment to have length zero, equal to its margin. The two segments do not overlap, giving

$$
L _ { \mathrm { p r e d } } ( P ^ { a } , P ^ { D M } ; x ) = \angle ( c _ { a } , c ) \geq \psi ( x ) + \phi ( x ) .
$$

In particular, the loss is at least ϵ when $\phi ( x ) \geq \epsilon .$

Next, we consider the suficient condition for fitting. If $\phi ( x ) = 0$ , its strict inequality cannot hold since the optimality gap is nonnegative. Otherwise, $w _ { a }$ is the unique optimizer under $c _ { a }$ . For every $v \in V \setminus \{ w _ { a } \}$ , we have cone $( w _ { a } ) \subseteq \{ u \in \mathbb { R } ^ { n } : u ^ { \top } ( v - w _ { a } ) \geq 0 \}$ . A shortest spherical path from $\frac { c _ { a } } { \| c _ { a } \| }$ to the hyper-plane $\iota ^ { \top } ( v - w _ { a } ) = 0$ must reach the cone boundary no later than its endpoint. Thus, we have

$$
\begin{array} { r } { 0 < \phi ( x ) \leq \arcsin \left( \frac { c _ { a } ^ { \top } ( v - w _ { a } ) } { \| c _ { a } \| \| v - w _ { a } \| } \right) \leq \frac { \pi } { 2 } . } \end{array}
$$

This implies

$$
c _ { a } ^ { \top } ( v - w _ { a } ) \geq \| c _ { a } \| \| v - w _ { a } \| \sin ( \phi ( x ) ) \geq \| c _ { a } \| \ell _ { e } \sin ( \phi ( x ) ) .
$$

Therefore, we conclude that ${ c _ { a } ^ { \top } } ( \tilde { w } ( x ) - w _ { a } ) < \| c _ { a } \| \ell _ { e } \sin ( \phi ( x ) )$ forces $\tilde { w } ( x ) = w _ { a } \in W ^ { \star } ( c _ { a } )$

For the expected bound, the cache can equivalently be generated by assigning an independent uniformly chosen member to every possible candidate set before queries begin. For each fixed $x ,$ the set $\mathcal { W } _ { \epsilon } ( x )$ is independent of these assignments and contains $w ^ { \star } ( P ^ { D M } x )$ . Averaging over the cache randomness yields

$$
\begin{array} { r } { \mathbb { P } \{ \tilde { w } ( x ) \neq w ^ { \star } ( P ^ { D M } x ) \} = 1 - \frac { 1 } { | \mathcal { W } _ { \epsilon } ( x ) | } . } \end{array}
$$

Under the stated assumption, the pointwise bound is at least $\psi ( x )$ on this event and is nonnegative otherwise. Averaging over the defense and attacker randomness and an independent $x \sim Q$ therefore yields

$$
\begin{array} { r } { \mathbb { E } [ L _ { \mathrm { p r e d } } ( P ^ { a } , P ^ { D M } ; x ) ] \ge \mathbb { E } _ { x \sim Q } \left[ \psi ( x ) \left( 1 - \frac { 1 } { | \mathcal { W } _ { \epsilon } ( x ) | } \right) \right] = B _ { \epsilon } ( Q ) . } \end{array}
$$

Finally, $\mathcal { W } _ { \epsilon } ( x )$ is nondecreasing in ϵ under set inclusion, including at $\epsilon = 0$ , while $\psi ( x ) \geq 0$ is independent of $\epsilon .$ Therefore, we conclude that $B _ { \epsilon } ( Q )$ is nondecreasing in ϵ. □

Proof of Proposition A.1. Fix a full-row-rank matrix $P ^ { a }$ and $P \in \mathcal P _ { c } ( P ^ { a } )$ . For each facet $i ,$ we choose $x _ { i } \in \mathbb { R } ^ { d }$ with $P ^ { a } x _ { i } = r _ { i } . { \mathrm { ~ I f ~ } } z \in \ker ( P ^ { a } )$ , we have $P ^ { a } ( x _ { i } + z ) = r _ { i }$ . This implies $P x _ { i } , P ( x _ { i } + z ) \in \mathrm { s p a n } ( r _ { i } )$ ) and $P z \in \operatorname { s p a n } ( r _ { i } )$ Since the normals include two independent directions, we have $P z = 0$ . This implies ker $\cdot ( P ^ { a } ) \subseteq \ker ( P )$ and the full row rank of $P ^ { a }$ gives $P = B P ^ { a }$ for a linear map $B : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$

For each $i ,$ we have $B r _ { i } = P x _ { i } \in \mathrm { s p a n } ( r _ { i } )$ or equivalently $B a _ { i } = \lambda _ { i } a _ { i }$ for some $\lambda _ { i } \in \mathbb { R }$ . Applying B to the relation in Assumption 3.4 yields $\begin{array} { r } { \sum _ { j = 1 } ^ { n } \beta _ { j } \big ( \lambda _ { k _ { j } } - \lambda _ { k _ { n + 1 } } \big ) a _ { k _ { j } } = 0 } \end{array}$ . Linear independence and $\beta _ { j } \neq 0$ imply $\lambda _ { k _ { j } } = \lambda _ { k _ { n + 1 } }$ for every $j = 1 , \dotsc , n$ . Thus, we have $B = \gamma I _ { n }$ and $\dot { P _ { \mathrm { ~ \scriptsize ~ = ~ } \gamma P ^ { a } } }$ for some $\gamma \in \mathbb { R }$ . Conversely, every real multiple of P<sup>a</sup> belongs to the candidate space. Thus, we have ${ \mathcal { P } } _ { c } ( P ^ { a } ) = \operatorname { s p a n } \{ P ^ { a } \}$ and its dimension is 1. □

Proof of Corollary A.2. Suppose that $P ^ { a } \neq 0$ satisfies $w ^ { \star } ( P ^ { D M } x ) \in W ^ { \star } ( P ^ { a } x )$ for every x /∈ ker $( P ^ { D M } )$ . Then, the proof of Theorem 3.5 gives a linear map $B : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ satisfying $P ^ { a } = B \bar { P } ^ { D M }$ and $B r _ { i } \in \{ \alpha r _ { i } : \alpha \geq 0 \}$ for every i. Thus, we have $P ^ { a } \bar { x } \in \mathrm { s p a n } ( r _ { i } )$ whenever $P ^ { D M } \bar { x } = r _ { i }$ . This implies $P ^ { a } \in \mathcal { P } _ { c } ( P ^ { D M } )$ and $\dot { P } ^ { a } = \gamma P ^ { D \dot { M } }$ for some $\gamma \in \mathbb { R }$ . For any x¯ with $P ^ { D M } \bar { x } = r _ { i }$ , we have $\gamma r _ { i } = P ^ { a } \bar { x } = B r _ { i } \in \{ \alpha r _ { i } : \alpha \geq 0 \}$ . Since $P ^ { a } \neq 0$ , this implies $\gamma > 0$ . Thus, every inverse feasibility loss calibrates the prediction loss. □

Lemma B.2. For each $p \geq 1$ , we let $U _ { p } \in \mathbb { R } ^ { n \times d }$ be a full-row-rank terminal output of Algorithm 1 with parameters $M _ { p } \geq 1$ and $\Delta _ { p } > 0 _ { \mathrm { { \ell } } }$ , normalized so that $\| U _ { p } \| _ { F } = 1$ . Fix a nonempty set $J \subseteq \{ 1 , \dots , m \}$ . Suppose its final sampling call uses the construction in Algorithm 6 and retains every boundary and near-boundary query with $i \in J$ . Denote the boundary representatives and near-boundary queries in that call by $\bar { x } _ { i 0 j , p }$ and $\tilde { x } _ { i k j , p } ,$ , respectively, where $j = 1 , \ldots , d - n + 1$ and $k \in \mathcal { C } _ { i }$ . In particular, $U _ { p } \bar { x } _ { i 0 j , p } = r _ { i }$ . Assume that every boundary representative used to form these queries, including auxiliary representatives outside J, has norm at most a common constant $R > 0 . \ I f \ M _ { p } \to + \infty$ and $\Delta _ { p } ( M _ { p } + K - 1 ) < \eta _ { S }$ for all suficiently large $p ,$ we have

$$
\operatorname* { i n f } _ { \alpha > 0 } \big \| P ^ { D M } \bar { x } _ { i 0 j , p } - \alpha r _ { i } \big \| \to 0 \mathrm { ~ f o r ~ a l l ~ } i \in J , \ j = 1 , \dots , d - n + 1 ,
$$

where K and η<sub>S</sub> are defined in Appendix A.3.

Proof. Fix $i \in J , j \in \{ 1 , \dots , d - n + 1 \}$ , and $k \in \mathcal { C } _ { i }$ . Using the estimate in Appendix A.3 and the stated stopping condition, for suficiently large p, we have

$$
\begin{array} { r } { ( U _ { p } \tilde { x } _ { i k j , p } ) ^ { \top } ( v - w _ { k } ) \geq \frac { \eta _ { S } } { M _ { p } + K - 1 } > \Delta _ { p } \mathrm { ~ f o r ~ e v e r y ~ } v \in V \setminus \{ w _ { k } \} . } \end{array}
$$

Since this query is retained in the dataset on which $U _ { p }$ satisfies the stopping test, its returned vertex must be $w _ { k }$ Thus, we have $P ^ { D M } \tilde { x } _ { i k j , p } \in C _ { k }$ . For $M _ { p } > K - 1$ , we have

$$
\begin{array} { r } { \| \tilde { x } _ { i k j , p } - \bar { x } _ { i 0 j , p } \| \le \frac { 2 R ( K - 1 ) } { M _ { p } - ( K - 1 ) } \to 0 , } \end{array}
$$

which implies

$$
\begin{array} { r } { \| P ^ { D M } \tilde { x } _ { i k j , p } - P ^ { D M } \bar { x } _ { i 0 j , p } \| \leq \| P ^ { D M } \| _ { F } \| \tilde { x } _ { i k j , p } - \bar { x } _ { i 0 j , p } \|  0 . } \end{array}
$$

The sequence $\{ P ^ { D M } \bar { x } _ { i 0 j , p } \} _ { p \geq 1 }$ is bounded. Since each $C _ { k }$ is closed, every subsequential limit lies in $\cap _ { k \in \mathcal { C } _ { i } } C _ { k } =$ $\lbrace \alpha r _ { i } : \alpha \geq 0 \rbrace$ . If $\begin{array} { r } { \operatorname* { i n f } _ { \alpha \geq 0 } \| \dot { P } ^ { \overline { { D } } M } \bar { x } _ { i 0 j , p } - \alpha r _ { i } \| } \end{array}$ did not tend to zero, a bounded subsequence staying a positive distance from this closed ray would have a limit in the ray, yielding a contradiction. □

Lemma B.3. Let $J \subseteq \{ 1 , \dots , m \}$ be such that $\{ a _ { i } : i \in J \}$ contains $n + 1$ normals satisfying Assumption $\ 3 . 4 \cdot$ Let $\{ U _ { p } \} _ { p \ge 1 } \subseteq \mathbb { R } ^ { n \times d }$ consist of full-row-rank matrices with $\| U _ { p } \| _ { F } = 1$ . For each $i \in J$ and $p \geq 1$ , we choose $\bar { x } _ { i 0 j , p } \in \mathbb { R } ^ { d }$ satisfying $U _ { p } \bar { x } _ { i 0 j , p } = r _ { i }$ for $j = 1 , \ldots , d - n + 1$ , and set $X _ { i , p } = [ \bar { x } _ { i 0 1 , p } , \dotsc , \bar { x } _ { i 0 , d - n + 1 , p } ]$ . Suppose there are constants R, $\kappa > 0$ , independent of p, such that $\| \bar { x } _ { i 0 j , p } \| \le R$ and $\sigma _ { \operatorname* { m i n } } ( X _ { i , p } ) \geq \kappa _ { \mathbf { \mu } , \mathbf { \Sigma } }$ for every $i \in J$ and $j _ { ; }$ where $\sigma _ { \mathrm { m i n } }$ is the smallest of the $d - n + 1$ singular values. $\begin{array} { r } { \vert \vert \vert \mathrm { f } \operatorname* { i n f } _ { \alpha \ge 0 } \vert \vert P ^ { D M } \bar { x } _ { i 0 j , p } - \alpha r _ { i } \vert \vert  0 } \end{array}$ for all $i \in J$ and $j = 1 , \ldots , d - n + 1$ , we have $\begin{array} { r } { U _ { p } \to \frac { P ^ { D M } } { \left\| P ^ { D M } \right\| _ { F } } } \end{array}$ and $\angle ( U _ { p } x , P ^ { D M } x ) \to 0$ for every x /∈ ker(P<sup>DM</sup>).

Proof. Since $\| U _ { p } \| _ { F } = 1$ , it sufices to identify every subsequential limit. We take a subsequence with $U _ { p } \to U _ { \star }$ Boundedness of the finitely many representatives gives a further subsequence, denoted by the same indices, such that $\bar { x } _ { i 0 j , p }  \bar { x } _ { i 0 j } ,$ for all $i \in J$ and $j = 1 , \ldots , d - n + 1$

We define $X _ { i , \star } = [ \bar { x } _ { i 0 1 , \star } , \ldots , \bar { x } _ { i 0 , d - n + 1 , \star } ]$ . Then, $X _ { i , p } \to X _ { i , \star } ,$ , and continuity gives $\| U _ { \star } \| _ { F } = 1 , U _ { \star } \bar { x } _ { i 0 j , \star } = r _ { i }$ and $\sigma _ { \operatorname* { m i n } } ( X _ { i , \star } ) \geq \kappa$ . The image of $U _ { \star }$ contains the selected $r _ { i }$ , including n independent directions. Thus, $U _ { \star }$ has full row rank. For each $i \in J .$ , the columns of $X _ { i , { \ i } }$ lie in a $( d - n + 1 )$ -dimensional space $\{ x \in \mathbb { R } ^ { d } : U _ { \star } x \in \operatorname { s p a n } ( r _ { i } ) \}$ There are $d - n + 1$ such columns, and the conditioning bound makes them independent, so they form a basis of this space. The assumed convergence gives $P ^ { D M } \bar { x } _ { i 0 j , \star } \in \{ \alpha r _ { i } : \alpha \geq 0 \}$ }. By linearity, we have $\breve { P } ^ { D M } x \in \operatorname { s p a n } ( r _ { i } )$ whenever $U _ { \star } x \in \operatorname { s p a n } ( r _ { i } )$ , and in particular, whenever $U _ { \star } x = r _ { i }$

Since J contains $n + 1$ normals satisfying Assumption 3.4, the argument in the proof of Proposition A.1, restricted to $i \in J$ , gives $P ^ { D M } = \gamma U ,$ <sub>⋆</sub> for some $\gamma \in \mathbb { R }$ . Since $\gamma r _ { i } = P ^ { D M } \bar { x } _ { i 0 j , \star } \in \{ \alpha r _ { i } : \alpha \geq 0 \}$ and $P ^ { D M } \neq 0$ , we have $\gamma > 0$ . Normalization then gives $\gamma = \| P ^ { D M } \| _ { F }$ and $\begin{array} { r } { U _ { \star } = \frac { P ^ { D M } } { \| P ^ { D M } \| _ { F } } } \end{array}$ . Every subsequential limit is the same, proving convergence of the full sequence. For every $x \notin \ker ( P ^ { \mathrm { D M } } )$ , we have $\begin{array} { r } { U _ { p } x \to \frac { P ^ { D M } x } { \| P ^ { D M } \| _ { F } } \ne 0 } \end{array}$ . Continuity of angular distance therefore gives the desired convergence of the prediction loss. □

Proof of Theorem A.3. Under the stated assumptions, Proposition 3.7 guarantees that every run terminates after at most $\begin{array} { r } { \left( 1 + \frac { 2 R L _ { e } } { \Lambda } \right) ^ { n d } } \end{array}$ inverse optimization updates. We take any sequence $\Delta _ { p }  0$ in the specified family and set $M _ { p } = M ( \Delta _ { p } ^ { - } )$ and $U _ { p } = P _ { M _ { p } , \Delta _ { p } } ^ { a }$ . For each $p _ { : }$ , we let $\bar { x } _ { i 0 j , p }$ denote the boundary representatives in the final sampling call associated with $U _ { p } ,$ and set $X _ { i , p } = [ \bar { x } _ { i 0 1 , p } , \dotsc , \bar { x } _ { i 0 , d - n + 1 , p } ]$ . This call is included in the dataset used by the stopping test. Every boundary representative is queried and hence has norm at most R. Since $M _ { p } \to + \infty$ and the stopping condition holds for suficiently large $p ,$ Lemma B.2 applies with $J = \{ 1 , \dots , m \}$ and gives

$$
\operatorname* { i n f } _ { \alpha \geq 0 } \| P ^ { D M } \bar { x } _ { i 0 j , p } - \alpha r _ { i } \|  0 \mathrm { ~ f o r ~ a l l ~ } i = 1 , \dots , m , \ j = 1 , \dots , d - n + 1 .
$$

The bound $\sigma _ { \operatorname* { m i n } } ( X _ { i , p } ) \geq \kappa$ then allows us to apply Lemma B.3, yielding $\begin{array} { r } { P _ { M _ { p } , \Delta _ { p } } ^ { a } \to \frac { P ^ { D M } } { \parallel P ^ { D M } \parallel _ { F } } } \end{array}$ . Since the sequence $\Delta _ { p }  0$ was arbitrary, the matrix convergence holds for the full family. For every $x \not \in \ker ( \bar { P } ^ { D M } )$ , the same lemma gives $L _ { \mathrm { p r e d } } ( P _ { M ( \Delta ) , \Delta } ^ { a } , P ^ { D M } ; x )  0$ □

Proof of Lemma $A . 4 .$ Suppose that $P ^ { a } \neq 0$ satisfies the stated consistency condition on X \ ker $( P ^ { D M } )$ . Then, for each $i \in J ,$ we fix ${ \bar { x } } _ { i } \in \operatorname { i n t } ( \mathcal X )$ and $\alpha _ { i } > 0$ from the assumption, so that $\mathbf { \bar { \Pi } } _ { P } { \cal { D } } M _ { \bar { x } _ { i } } = \alpha _ { i } r _ { i }$ . We fix $i \in J$ and any $x \in \operatorname { i n t } ( \mathcal { X } )$ with $P ^ { D M } x = \alpha _ { i } r _ { i }$ . For each $k \in \mathcal { C } _ { i }$ , we choose $h _ { k } \in \mathbb { R } ^ { d }$ with $P ^ { D M } h _ { k } \in \mathrm { i n t } ( C _ { k } )$ , using the full row rank of $\dot { P } ^ { D M }$ . For suficiently small $t > 0$ , we have $x + t h _ { k } \in \mathcal { X }$ and $P ^ { D M } ( x + t h _ { k } ) = \alpha _ { i } r _ { i } + t P ^ { D \bar { M } } h _ { k } \in \mathrm { i n t } ( C _ { k } )$ Consistency gives $P ^ { a } ( x + t h _ { k } ) \in C _ { k }$ . Letting $t \downarrow 0$ and using closedness yields $P ^ { a } x \in \cap _ { k \in \mathcal { C } _ { i } } C _ { k } = \{ \alpha r _ { i } : \alpha \geq 0 \}$

We fix $z \in \ker ( P ^ { D M } )$ . For each $i \in J$ and suficiently small nonzero t, both $\bar { x } _ { i }$ and $\bar { x } _ { i } + t z$ lie in int(X) and have deployed cost $\alpha _ { i } r _ { i }$ . The preceding conclusion gives $P ^ { a } \bar { x } _ { i } , P ^ { a } ( \bar { x } _ { i } + t z ) \in \{ \alpha r _ { i } : \alpha \geq 0 \}$ and $P ^ { a } z \in \operatorname { s p a n } ( r _ { i } )$ . Since the selected normals include two independent directions, we have $P ^ { a } z = 0$ . This implies ker $( P ^ { D M } ) \subseteq \ker ( P ^ { a } )$ and $P ^ { a } = B P ^ { D M }$ for a linear map $B : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ . Evaluating at $\bar { x } _ { i }$ gives $\alpha _ { i } B r _ { i } = P ^ { a } \bar { x } _ { i } \in \{ \alpha r _ { i } : \alpha \geq 0 \}$ . Thus, we have $B r _ { i } = \lambda _ { i } r _ { i }$ for some $\lambda _ { i } \geq 0$ for each $i \in J$ . Applying the eigenvalue argument from the proof of Theorem 3.5 to the selected $n + 1$ normals gives $B = \gamma I _ { n }$ with $\gamma \geq 0$ . Since $P ^ { a } \neq 0$ , we have $\gamma > 0$ and $\mathsf { \bar { P } } ^ { a } = \gamma P ^ { D M }$ □

Proof of Lemma A.5. Under the stated assumptions, Proposition 3.7 guarantees that every run terminates after at most $\begin{array} { r } { \left( 1 + \frac { 2 R L _ { e } } { \Lambda } \right) ^ { n d } } \end{array}$ inverse optimization updates. We take any sequence $\Delta _ { p }  0$ in the specified family and set $\begin{array} { r } { M _ { p } = M ( \Delta _ { p } ^ { - } ) } \end{array}$ and $U _ { p } = P _ { M _ { p } , \Delta _ { p } } ^ { a }$ . For each p, we let $\bar { x } _ { i 0 j , p }$ denote the boundary representatives in the final sampling call associated with $U _ { p } { \mathrm { . } }$ , and set $X _ { i , p } = [ \bar { x } _ { i 0 1 , p } , \dotsc , \bar { x } _ { i 0 , d - n + 1 , p } ]$ for all $i \in J$ . Every query in each selected block $i \in J$ is retained in the dataset used by the stopping test. All representatives used to form these queries, including those outside $^ { J , }$ have norm at most R. Lemma B.2 therefore gives

$$
\operatorname* { i n f } _ { \alpha > 0 } \| P ^ { D M } \bar { x } _ { i 0 j , p } - \alpha r _ { i } \|  0 \mathrm { ~ f o r ~ a l l ~ } i \in J , \ j = 1 , \dots , d - n + 1 .
$$

The selected facets contain $n + 1$ normals satisfying Assumption 3.4, and $\sigma _ { \operatorname* { m i n } } ( X _ { i , p } ) \mathrm { ~ \geq ~ } \kappa$ for every $i \in J$ Lemma B.3 yields $\begin{array} { r } { U _ { p } \to \frac { P ^ { D M } } { \parallel P ^ { D M } \parallel _ { F } } } \end{array}$ . Since the sequence $\Delta _ { p }  0$ was arbitrary, the matrix convergence holds for the full family. For every $x \not \in \ker ( \overset { \cdot \cdot } { P } ^ { D M } )$ , the same lemma gives $L _ { \mathrm { p r e d } } ( P _ { M ( \Delta ) , \Delta } ^ { a } , P ^ { D M } ; x )  0$ □