# Performative Prediction with Selective Labels

Giovani Valdrighi<sup>∗</sup> Instituto de Computação Universidade Estadual de Campinas Campinas, Brazil

Isabel Valera Department of Computer Science Saarland University Saarbrücken, Germany

Marcos Medeiros Raimundo Instituto de Computação Universidade Estadual de Campinas Campinas, Brazil

## Abstract

Many social applications of machine learning exhibit performative effects: population behavior changes in response to deployed models. Performative prediction studies this interaction through a distribution map that relates each model to the population distribution it induces. One of the main results in this framework showed that repeated risk minimization (RRM), which updates models by retraining on the most recent data, can converge to a stable model that minimizes risk on its own induced distribution. However, existing analyses typically assume access to the complete distributions of features and labels after model deployment, ignoring the possibility of selective labels: observing labels only for the accepted subset of the population. In this work, we formalize performative prediction with selective labels and show that retraining only on observed data can misguide the retraining procedure and undermine the guarantees of convergence to a stable solution. We then propose a worst-case objective based on knowledge of a confidence interval on the probability of a positive label. Applying RRM to this objective permits us to remain within a bounded distance to the true stable point. Under a sensitivity assumption on the conditional label distribution, we further show how previously accepted data can tighten these confidence intervals over time. Experiments in a lending application with fairness regularization show that our robust optimization approach closely matches the performance of RRM with complete label access.

## 1 Introduction

In algorithmic decision-making, machine learning models support decisions and intervene on the outcomes they were trained to predict. For example, a high-risk prediction for a certain disease can prompt early treatment and help avoid this outcome. Additionally, highly consequential applications such as hiring, finance, and criminal justice define opportunities and incentives, thereby shaping the populations they interact with. The seminal work by Perdomo et al. [1] formalized these ideas in the framework of performative prediction, where each model θ induces a distribution $( X , Y ) \sim D ( \theta )$ In this sense, a model optimal under loss ℓ for a training distribution D is not necessarily optimal when deployed under the distribution $D ( \theta )$ . This shift can be expressed by the decoupled risk, which separates the model ϕ that induces the distribution and the model θ being evaluated:

$$
J ( \phi , \theta ) = \mathbb { E } _ { D ( \phi ) } [ \ell ( X , Y , \theta ) ]\tag{1}
$$

One of the main results from Perdomo et al. [1] was to show that if ℓ is strongly convex and smooth and D(θ) has bounded sensitivity w.r.t. θ, the iterative procedure of retraining the model in the most recent distribution would converge to an equilibrium where θ was optimal in the induced distribution $D ( \theta )$ , that is, $\theta ^ { P S } = \arg \operatorname* { m i n } _ { \theta } J ( \theta ^ { P S } , \mathbf { \bar { \theta } } )$ . This means $\theta ^ { P S }$ is robust to the distribution shift it induces. However, despite the practical relevance of performative prediction, many works still assume access to samples from the full distribution $D ( \theta )$ , ignoring that many decision-making applications have selective labels: labels are observed only for the accepted subset of the population. In a lending application, the decision-maker can access the economic profile and financial history from all applicants, while only observing the payment label for the subset that had their loan accepted; in a health diagnosis setting, patients’ profiles are accessible, while the final diagnosis is only known for the subset that underwent testing.

Performative prediction is a widely discussed topic, with multiple works examining conditions for achieving performative stability, such as stochastic learning [2–4], introducing dependence on previous states [5], or relaxing assumptions on the loss function [6]. In general, this literature considers variations of an iterative learning process called repeated risk minimization (RRM) in which, after selecting a model $\theta _ { t }$ , it is deployed in the world to obtain access to $D ( \theta _ { t } )$ , which is then used to define the next model $\theta _ { t + 1 }$ . However, many relevant applications suffer from selective labels: while features x are collected from all samples, the label y is observed only for a subset of samples accepted under the current θ. In that sense, we never observe the true distribution $D ( \theta )$ , but rather the conditional distribution $X , Y | O = 1$ , where $O = 1$ is the acceptance event.

Our contributions are:

• The Selective Performative Gap: We demonstrate that retraining with performativity fails to reach stability under selective labels. When observation is deterministic and threshold-based, models trained on the fully observed distribution $D ( \theta )$ versus the filtered distribution $\tilde { D } ( \theta )$ diverge arbitrarily. This fundamental gap motivates our new framework.

• Robust Repeated Risk Minimization: To overcome missing labels, we introduce a strongly convex upper-bound loss that operates independently of the true label y, extending partial identification techniques [7]. By leveraging limited knowledge of $P ( Y = 1 | X = \bar { x } , \bar { \theta } )$ , we bound the decoupled performative risk and optimize the model $\theta _ { t }$ through our novel Robust RRM procedure.

• Theoretical Guarantees: We prove that Robust RRM approximates the true performatively stable solution $\theta ^ { P S }$ within a bounded distance. This bound tightens with higher label accessibility, better distributional knowledge, and lower gradient sensitivity. Furthermore, we provide a principled method for iteratively tightening these bounds during retraining, based on prior observations.

• Novel Evaluation Framework: We propose a new experimental setting for performative prediction with selective labels, adapting the lending environment with fairness constraints introduced by Liu et al. [8] to validate our robust retraining dynamics.

Code for experiments is available at github.com/recod-ai/perf\_selective.

## 1.1 Related Work

For a comprehensive review of prior work in performative prediction, we refer to the surveys by Hardt and Mendler-Dünner [9] and Kehrenberg et al. [10].

Performative Prediction Motivated by ideas of performativity in economics [11], Perdomo et al. [1] introduced performative prediction in supervised learning and studied conditions for stable, optimal solutions. Since then, a larger literature has arisen in performative prediction, with works studying how to reach performative optimality with structural assumptions over D(θ) [12–14] or bandit-inspired algorithms [15, 16], considering constraints [17], fairness [18–20], or adapting it to reinforcement learning [21]. Reaching a stable solution has also been a relevant topic, with works studying the convergence to the stable solution of stochastic algorithms [2, 22, 3, 4], weakening the assumption of strong-convexity of the loss [6], introducing state dependency [5], reusing data from previous deployments to improve convergence rates [23], or reaching a fair stable point [20].

Selective Labels Selective labels is a special case of selection bias, a larger theme that studies settings where not all observations are available to the decision-maker. This problem has already been studied in prior work using static data distributions [24, 25, 7] and its interaction with fairness [26–29]. As Fawkes et al. [30] point out, most fairness benchmarks exhibit some form of selection bias. Yet, few works have considered the interaction between dynamic distributions and selective labels. Creager et al. [31] and Valdrighi et al. [32] both discuss fairness in dynamic environments with selective labels, with the former leveraging a doubly robust estimator and the latter presenting a reinforcement-learning algorithm. Yet neither solution can handle decisions made using thresholds.

A few works have already discussed particular cases of performative prediction with selective labels. In strategic classification [33], where individuals can alter their features to obtain the desired prediction without truly altering their label, Harris et al. [34] present online learning algorithms with bounded regret for classification using linear policies. Zhao et al. [35] extend these results by accounting for noise in label observations. Despite their similarity to this work, both focus only on the strategic classification scenario. Pombal et al. [36] discuss how selective labels can introduce unfairness in performative prediction settings and presents a fraud-detection simulation.

## 2 Preliminaries

We consider a population described by features $x \in \mathcal { X } \subseteq \mathbb { R } ^ { d }$ and a binary label $y \in \mathcal { Y } : = \{ 0 , 1 \}$ The decision-maker acts in the system with a model defined by parameters $\theta$ from a compact convex set $\Theta \subset \mathbb { R } ^ { d }$ , which incurs loss defined by $\ell : \mathcal { X } \times \mathcal { Y } \times \Theta \stackrel { \cdot \cdot } {  } \mathbb { R }$ . In a loan example, x could be socioeconomic indicators, y the loan payment, and ℓ is the binary cross-entropy. We will write the Euclidean norm as $\| \cdot \| _ { 2 } , \mathbf { 1 } \dot { \{ \cdot \} }$ as the indicator function and use $\theta$ and $\phi$ to refer to arbitrary models from Θ.

Performative Prediction After deploying a model θ, the population responds with a new distribution defined by the distribution map $\bar { D } : \mathsf { \hat { \Theta } } \to \Delta ( { \boldsymbol { \chi } } \times { \boldsymbol { \ y } } )$ . Thus, a training dataset D will not reflect the true performance of a model $\theta$ when deployed. We will write $p _ { \theta } ( x )$ as the probability density of $x$ with the distribution $D ( \theta )$ and define the label distribution $\alpha _ { \theta } \dot { ( } x \dot { ) } = P ( Y \bar { = } 1 | X = \dot { x } , \theta )$ . Many applications of social decision-making will present performativity. In a loan application, individuals might try to manipulate their features x to reach the desired approval, without necessarily altering the true label $y .$ In this example, both $p _ { \theta } ( x )$ and $\alpha _ { \theta } ( x )$ will be dependent on θ. Another application commonly discussed in long-term fairness studies [8, 37, 38] assumes that individuals will improve their condition (or have their condition altered by $\theta )$ , resulting in an update to $x ,$ which is then reflected in $y .$ In this scenario, $p _ { \theta } ( x )$ remains θ-dependent, while $\alpha \theta$ becomes invariant to $\theta .$ In the performative setting, the objective of the decision-maker is to minimize the performative risk:

$$
\operatorname* { m i n } _ { \theta \in \Theta } \mathbb { E } _ { D ( \theta ) } [ \ell ( X , Y , \theta ) ]\tag{2}
$$

However, this objective depends on $\theta$ both in the loss function and in the distribution, which we observe only after model deployment. We also define the decoupled performative risk $J ( \phi , \theta ) : =$ $\mathbb { E } _ { D ( \phi ) } [ \ell ( X , Y , \theta ) ]$ ], where a model $\theta$ is evaluated in the distribution induced by another model ϕ.

Stable Solutions In the performative prediction setting, two types of solutions are of interest. A performatively optimal solution is $\theta ^ { P O } \stackrel { \bullet } { = } \arg \operatorname* { m i n } _ { \theta \in \Theta } J ( \theta , \theta )$ , and a performatively stable solution is such that $\begin{array} { r } { \theta ^ { \dot { P } S } \stackrel { - } { = } \arg \operatorname* { m i n } _ { \theta \in \Theta } J ( \theta ^ { P S } , \theta ) } \end{array}$ . Performatively stable solutions can be interpreted as an equilibrium point, where $\theta ^ { P \bar { S } }$ minimizes the loss over the distribution it induces, and the decisionmaker has no incentive to select another model. Under the conditions we present below, retraining on the most recent data distribution converges to a performatively stable solution.

Definition 2.1 (RRM). Repeated Risk Minimization (RRM) is the procedure of starting from a model $\theta _ { 0 }$ , employing the iterative updates based on $\begin{array} { r } { \theta _ { t + 1 } = G ( \theta _ { t } ) : = \arg \operatorname* { m i n } _ { \theta \in \Theta } J ( \theta _ { t } , \theta ) } \end{array}$

Definition 2.2 (Sensitivity and regularity conditions [1]). A distribution map $D : \Theta \to \Delta ( \mathcal { X } \times \mathcal { Y } )$ is ϵ-sensitive if $\forall \theta , \phi \in \Theta$ , it holds that $\hat { \mathcal { W } } _ { 1 } ( D ( \theta ) , D ( \phi ) ) \leq \epsilon \| \theta - \phi \| _ { 2 }$ where $\mathcal { W } _ { 1 }$ is the Wasserstein-1 distance. Additionally, the loss $\ell : \mathcal { X } \times \dot { \mathcal { Y } } \times \dot { \Theta }  \dot { \mathbb { R } }$ is:

• γ-strongly convex if ∀θ, $\phi ~ \in ~ \Theta$ and $( x , y ) ~ \in ~ \mathcal { X } \times \mathcal { Y } , ~ \ell ( x , y , \theta ) ~ \geq ~ \ell ( x , y , \phi ) ~ + ~$ $\begin{array} { r } { \nabla _ { \theta } \ell ( x , y , \phi ) ^ { \top } ( \theta - \phi ) + \frac { \gamma } { 2 } \| \theta - \phi \| _ { 2 } ^ { 2 } } \end{array}$

$\beta .$ -smooth in $( x , y ) \mathrm { ~ i f ~ } \| \nabla _ { \theta } \ell ( x , y , \theta ) - \nabla _ { \theta } \ell ( x ^ { \prime } , y ^ { \prime } , \theta ) \| _ { 2 } \le \beta \| ( x , y ) - ( x ^ { \prime } , y ^ { \prime } ) \| _ { 2 } , \forall \theta \in  { \mathcal { O } } ^ { 3 } .$ Θ, $( x , y ) , ( x ^ { \prime } , y ^ { \prime } ) \in \mathcal { X } \times \mathcal { Y }$

Theorem 2.3 (Theo. 3.5 from [1]). If the loss ℓ is β-smooth in $( x , y )$ and γ-strongly convex, the distribution map is ϵ-sensitive, then RRM as in Def. 2.1 is a contraction satisfying:

$$
\lVert G ( \theta ) - G ( \phi ) \rVert _ { 2 } \leq \frac { \epsilon \beta } { \gamma } \lVert \theta - \phi \rVert _ { 2 }\tag{3}
$$

and $i f \epsilon \beta / \gamma < 1$ , the iterates $\theta _ { t } : = G ( \theta _ { t - 1 } )$ will converge to a performatively stable solution $\theta ^ { P S }$

However, as we argue in the following section, the decision-maker will commonly not have access to samples of the full distribution $D ( \theta )$

## 3 Selective Labels in Performative Prediction

```perl
for t = 1, 2, . . .
1. Decision-maker publishes a model $\theta _ { t }$
2. Environment (the population) realizes feature-label pair $( X , Y ) \sim D ( \theta _ { t } )$
3. Decision-maker observes X
4. Decision-maker computes score $f _ { \theta _ { t } } ( X )$ and defines $O : = { \bf 1 } \{ f _ { \theta _ { t } } ( X ) \geq \tau \}$
5. If O = 1, decision-maker observes Y
```  
Figure 1: Observation dynamics for performative prediction with selective labels.

In real-world applications such as loan requests, school and job admissions, healthcare, and criminal justice, decision-makers observe x but have access to labels y only for a subset of the population that was accepted, called selective labels [25]. For example, in judicial bail decisions, judges have access to criminal history x, while the label y representing recidivism is observed only for defendants released pretrial [25]. In health diagnosis, patient covariates, such as routinely collected clinical measurements, may be available before a definitive diagnostic test is ordered. A label defined by the final diagnosis is therefore available only for the subset of patients who underwent testing [28]. In both settings, deployed policies may also alter subsequent actions and outcomes, thereby inducing performative effects [39].

We extend the performative prediction framework to include selective labels as displayed in Fig. 1. In this extension, θ also determines label observations. More specifically, after deploying $\theta ,$ the population/environment observes it and reacts with the pair X, Y sampled from $D ( \theta )$ . The decisionmaker first observes X and, using the model θ, computes a score via ${ \bar { f } } _ { \theta } : \mathcal { X } \to$ R. In this work, we focus on the deterministic, threshold-based observation case: $O = { \bf 1 } \{ \bar { f } _ { \theta } ( X ) \geq \tau \}$ , where $\tau \in \mathbb { R }$ is a threshold. Finally, Y is revealed only upon acceptance, that is, when $O = 1$ . Prior online learning work by Kilbertus et al. [24] shows that such a configuration is challenging and can yield suboptimal performance even when the true distribution D is fixed.

To employ RRM, after deploying ϕ, the decision-maker will leverage the data to evaluate the loss of another model θ, that is, calculating $\mathbb { E } _ { D ( \phi ) } [ \ell ( X , Y , \theta ) ]$ ]. Yet, ℓ cannot be calculated for rejected samples. A naive solution for the selective labels problem would be to leverage only the data from the accepted population, that is, the distribution $\tilde { D } ( \phi )$ that results from first sampling $( X , Y ) \sim D ( \phi )$ and then keeping observations if $f _ { \phi } ( X ) \geq \tau$ . However, as our next example shows, this procedure can produce models arbitrarily far from the true performatively stable solution $\theta ^ { P S }$

Proposition 3.1. Define G<sup>˜</sup>(ϕ) := arg min $\mathsf { i } _ { \theta \in \Theta } \mathbb { E } _ { { \tilde { D } } ( \phi ) } [ \ell ( X , Y , \theta ) ]$ and let G be the RRM as Def. 2.1. The gap ofone iteration with a model θ, measured by the distance $\lVert G ( { \boldsymbol { \theta } } ) - \tilde { G } ( { \boldsymbol { \theta } } ) \rVert _ { 2 }$ can reach the maximum value ofdiam $\begin{array} { r } { \left( \Theta \right) : = \operatorname* { s u p } _ { \theta ^ { \prime } , \theta ^ { \prime \prime } \in \Theta } \| \theta ^ { \prime } - \theta ^ { \prime \prime } \| _ { 2 } } \end{array}$

Proof. We will prove this with a simple example where $\lVert G ( \theta ) - \tilde { G } ( \theta ) \rVert _ { 2 } = \mathrm { d i a m } ( \Theta )$ even when D is not dependent on θ. Consider a parameter space restricted to $\Theta = [ 0 , \dot { B } ]$ for a fixed $\dot { B ^ { \prime } } \in \mathbb { R } _ { > 0 } . \operatorname { L e t } x \in$ $\{ - 1 , 1 \}$ and the loss is the binary cross entropy $\ell ( x , y , \theta ) = - y \overset { \cdot } { \log } ( \sigma ( \theta x ) ) - ( 1 - y ) \log ( 1 - \sigma ( \theta x ) )$ The decision-maker uses a linear model $f _ { \boldsymbol { \theta } } ( x ) = \dot { \theta } x$ and a threshold $\tau = 0$ . Define a fixed distribution such that $P ( X = - 1 \land Y = 1 ) = 0 . 9$ and $P ( X = 1 \land Y = 1 ) = 0 . 1$ . The loss is strongly convex on Θ, and for any fixed $\phi , J ( \phi , \theta ) = 0 . 9 \log ( 1 + e ^ { \theta } ) + 0 . 1 \log ( 1 + e ^ { - \theta } )$ has derivative $\sigma ( \theta ) - 0 . 1 > 0$ on Θ, so it is strictly increasing in θ and its minimizer is 0. Consequently, $G ( \phi ) = \mathrm { { \dot { 0 } } }$ for every ϕ. However, after deploying any model $\phi \in ( 0 , B ]$ , only samples with $x = 1$ satisfy $f _ { \phi } ( x ) = \phi x \ge 0$ , so we observe only samples with $x = 1$ and $y = 1$ . On this selectively observed distribution, the loss is strictly decreasing and is minimized when $\theta = B$ . Therefore, $| G ( \phi ) - \tilde { G } ( \phi ) | = B = \mathrm { d i a m } ( \Theta )$ □

This proposition shows that retraining on $\tilde { D } ( \cdot )$ can end arbitrarily far from the performatively stable solution $\mathbf { \Sigma } _ { \theta ^ { P S } } ^ { \bullet }$ obtained from RRM, even when no performative effects exist. In this way, training on the selectively labeled data can be an inappropriate objective depending on the true distribution map $D ( \cdot )$ . Next, we study how to leverage the features x of rejected samples to ensure proximity to the true decoupled risk.

## 4 A Robust Optimization Approach

As our previous result showed, using only the accepted population to perform RRM can cause a departure from the stable point, so we present a robust optimization approach. Inspired by prior work on partial identification from Chen et al. [7], the proposed approach leverages a worst-case estimate of the decoupled risk. This lets us show that running RRM on this worst-case loss approximates the stable solution $\theta ^ { P S }$ within a bounded distance. We present all proofs in Appendix A. We first assume partial knowledge of the label distribution $\alpha \theta$

Assumption 4.1. There are functions $l : \mathcal { X } \times \Theta  [ 0 , 1 ] , u : \mathcal { X } \times \Theta  [ 0 , 1 ]$ such that for all $\theta \in \Theta , \bar { x } \in \mathcal { X } , l ( x , \theta ) \leq \alpha _ { \theta } ( x ) \leq u ( x , \theta )$

This assumption is always valid in the trivial case where $l ( x , \theta ) = 0 , u ( x , \theta ) = 1 , \forall x , \theta$ . Yet, as the following results show, improving these intervals is an important step toward tightening our robust approach. Later, we present a specific formulation that leverages additional knowledge about the label distribution. With intervals $[ l ( x , \theta ) , u ( x , \theta ) ]$ , we can define a worst-case loss as follows:

Definition 4.2 (Worst-Case Loss). With a loss $\ell : \mathcal { X } \times \mathcal { Y } \times \Theta  \mathbb { R }$ , define $\Delta _ { \ell } ( x , \theta ) : = \ell ( x , 1 , \theta ) -$ $\ell ( x , 0 , \theta )$ as the difference between the loss evaluated on $y = 1$ and $y = 0$ for sample x and model θ. The worst-case $\mathrm { l o s s } ^ { 2 }$ for sample x under the distribution $D ( \phi )$ is:

$$
\overline { { \ell } } _ { \phi } ( x , \theta ) : = \ell ( x , 0 , \theta ) + \log \Big ( e ^ { l ( x , \phi ) \Delta _ { \ell } ( x , \theta ) } + e ^ { u ( x , \phi ) \Delta _ { \ell } ( x , \theta ) } \Big )\tag{4}
$$

The worst-case loss upper-bounds the conditional expected loss and remains differentiable and strongly convex.

Claim 4.3. IfAS. 4.1 holds,for any distribution $D ( \phi )$ over $X , Y ,$ it holds that $\begin{array} { r } { \mathbb { E } _ { D ( \phi ) } [ \ell ( X , Y , \theta ) ] \le } \end{array}$ $\mathbb { E } _ { D ( \phi ) } [ \overline { { \ell } } _ { \phi } ( X , \theta ) ]$ . Additionally, if ℓ is γ-strongly convex and β-smooth, then $\overline { { \ell } } _ { \phi }$ is $\gamma { - } s t r o n g l y$ convex but not necessarily β-smooth.

Finally, we can leverage our worst-case loss $\ell _ { \phi }$ for samples for which we do not have access to true labels, while using the loss ℓ for accepted samples. By using the true loss measure on accepted samples, we avoid losing information from the observed labels.

Definition 4.4 (Worst-case Decoupled Risk). The worst-case decoupled risk for a loss ℓ, with the defined respective worst-case loss as in Def. 4.2, employs ℓ for samples with label-access and $\overline { { \ell } } _ { \phi }$ for samples without access, as:

$$
\overline { { J } } ( \phi , \theta ) : = \mathbb { E } _ { D ( \phi ) } [ O \ell ( X , Y , \theta ) + ( 1 - O ) \overline { { \ell } } _ { \phi } ( X , \theta ) ]\tag{5}
$$

Definition 4.5 (Robust RRM). Robust Repeated Risk Minimization is the procedure of starting from a model $\theta _ { 0 }$ , employing the iterative updates based on $\begin{array} { r } { \theta _ { t + 1 } = \overline { { G } } ( \theta _ { t } ) : = \arg \operatorname* { m i n } _ { \theta \in \Theta } \overline { { J } } ( \theta _ { t } , \theta ) } \end{array}$

Since $\overline { { \ell } } _ { \phi }$ is not necessarily β-smooth, we cannot directly apply Theo. 2.3 by replacing ℓ with $\overline { { \ell } } _ { \phi } .$ . In that case, convergence to a stable solution is not guaranteed, even to a solution different from $\theta ^ { P \dot { S } }$ . In the remainder of this section, we show that Robust RRM remains within a bounded distance of the stable point $\theta ^ { P S }$ obtained with access to all labels. Our next result studies this by comparing one-step updates under both risks.

Theorem 4.6 (One-step gap between full and robust updates). Fix a deployment model ϕ. Suppose ℓ is differentiable and γ-strongly convex on θ and AS. 4.1 holds. Let G be the RRM as in Def. 2.1 and G be the Robust RRM as in Def. 4.5. Then

$$
\| G ( \phi ) - \overline { { G } } ( \phi ) \| _ { 2 } \leq \frac { 1 } { \gamma } \operatorname* { s u p } _ { \theta \in \Theta } \mathbb { E } _ { D ( \phi ) } \left[ ( 1 - O ) | u ( X , \phi ) - l ( X , \phi ) | \| \nabla _ { \theta } \Delta _ { \ell } ( X , \theta ) \| _ { 2 } \right] .\tag{6}
$$

In particular, $i f \ell$ is the binary cross-entropy of a linear score $f _ { \boldsymbol { \theta } } ( x ) = \boldsymbol { \theta } ^ { \intercal }$ x and $\forall x \in { \mathcal { X } } , \| x \| _ { 2 } \leq R .$

$$
\| G ( \phi ) - \overline { { G } } ( \phi ) \| _ { 2 } \leq \frac { R } { \gamma } \mathbb { E } _ { D ( \phi ) } \left[ ( 1 - O ) | u ( X , \phi ) - l ( X , \phi ) | \right] .\tag{7}
$$

This theorem shows that one iteration performed both in $G$ and $\overline { { G } }$ will result in two models with a limited distance, which is related to the gap between gradients of the loss function on $y = 0$ and $y = 1$ and the width of the limits l and u over the induced distribution of $D ( \phi )$ . Since robust updates remain within a bounded distance of the true updates, and true updates converge to a stable solution, we can bound how close iteration with $\overline { { G } }$ stays to that stable solution.

Theorem 4.7 (Approximate stability of Robust RRM). Assume the conditions ofTheo. 2.3, such that thefull update G is an η-contraction with $\eta = \epsilon \beta / \gamma < 1$ , and let $\theta ^ { P S }$ denote its uniquefixed point. Suppose AS. 4.1 holds and let $\begin{array} { r } { \kappa : = \operatorname* { s u p } _ { \theta \in \Theta } \| \overline { { G } } ( \theta ) - G ( \theta ) \| _ { 2 } } \end{array}$ . For the robust iterates $\theta _ { t + 1 } = \overline { { G } } ( \theta _ { t } ) .$

$$
\lVert \theta _ { t } - \theta ^ { P S } \rVert _ { 2 } \leq \eta ^ { t } \lVert \theta _ { 0 } - \theta ^ { P S } \rVert _ { 2 } + \frac { 1 - \eta ^ { t } } { 1 - \eta } \kappa\tag{8}
$$

Consequently, lim $\begin{array} { r } { \operatorname* { s u p } _ { t \to \infty } \| \theta _ { t } - \theta ^ { P S } \| _ { 2 } \leq \frac { \kappa } { 1 - \eta } } \end{array}$

While this result does not prove the convergence of Robust RRM, it shows that it remains within a bounded distance of the stable solution with total access to labels. From Theo. 4.6, this distance κ relates to the strong-convexity constant γ of the loss ℓ, the width of $| u ( \cdot , \cdot ) - l ( \cdot , \cdot )$ | and the magnitude of the gradient $\lVert \nabla _ { \theta } \Delta _ { \ell } ( X , \theta ) \rVert _ { 2 }$ . In Appendix B, we also present an upper bound on the performative risk of Robust RRM with some additional conditions on ℓ. As ℓ is application-dependent and cannot be easily altered, we study how to reduce the width of intervals during learning.

## 4.1 Improving Label Distribution Limits

In this section, we present an approach to estimate limits l and u for data with unseen labels based on the knowledge of $\alpha \theta$ . We do this by introducing sensitivity constants that are specific to the label distribution as follows:

Definition 4.8. The label distribution $\alpha _ { \theta } ( x )$ is $( L _ { x } , L _ { \theta } )$ -sensitive if for all $x , x ^ { \prime } \in { \mathcal { X } }$ and $\theta , \phi \in \Theta$ $| \alpha _ { \theta } ( x ) - \alpha _ { \phi } ( x ^ { \prime } ) | \leq L _ { x } \| x - x ^ { \prime } \| _ { 2 } + L _ { \theta } \| \dot { \theta } - \phi \| _ { 2 }$

This definition controls how $\alpha _ { \theta } ( x )$ can change by varying x and $\theta .$ In this way, we can construct a limit for unseen $\alpha _ { \theta } ( x )$ based on the knowledge of observed samples x and models θ. In particular, if a label distribution $\alpha \theta$ is invariant to θ, we can conclude that it is $( L _ { x } , 0 )$ -sensitive. In the remainder of this section, we will focus on the case where $\alpha _ { \theta }$ is invariant to $\theta .$ , and we will simplify the notation to $\alpha ( x ) , l ( x ) , u ( x )$ . See Appendix C for the results when $L _ { \theta } \neq 0$

During the repeated risk minimization procedure, the decision-maker has access to data from previously accepted individuals, that is $X , Y | O = 1$ , for all previously deployed models $\theta _ { 1 } , \ldots , \theta _ { t }$ . At time $t ,$ this information can be used to assess the limit α for samples rejected by $\theta _ { t }$ . Let $\chi _ { 1 : t } ^ { \mathrm { o b s } }$ be the support of the features accepted by $\theta _ { 1 } , \ldots , \theta _ { t }$ . Then, for an unobserved sample $x _ { \mathrm { u n o b s } } .$ , the following limits are valid:

$$
l ( x _ { \mathrm { u n o b s } } ) : = \operatorname* { m a x } \Big \{ 0 , \operatorname* { s u p } _ { x _ { \mathrm { o b s } } \in \mathcal { X } _ { 1 : t } ^ { \mathrm { o b s } } } \alpha ( x _ { \mathrm { o b s } } ) - L _ { x } \| x _ { \mathrm { o b s } } - x _ { \mathrm { u n o b s } } \| _ { 2 } \Big \} \leq \alpha \big ( x _ { \mathrm { u n o b s } } \big )\tag{9}
$$

$$
u ( x _ { \mathrm { u n o b s } } ) : = \operatorname* { m i n } \Big \{ 1 , \operatorname* { i n f } _ { x _ { \mathrm { o b s } } \in \mathcal { X } _ { 1 : t } ^ { \mathrm { o b s } } } \alpha ( x _ { \mathrm { o b s } } ) + L _ { x } \| x _ { \mathrm { o b s } } - x _ { \mathrm { u n o b s } } \| _ { 2 } \Big \} \geq \alpha ( x _ { \mathrm { u n o b s } } )\tag{10}
$$

These two limits, while depending on quantities $\alpha ,$ could be estimated from the observed data, as for an accepted sample such that $O = 1$ , we have that $P ( Y = 1 | X = x _ { \mathrm { o b s } } ) = P ( Y = 1 | X =$ $x _ { \mathrm { o b s } } , O = 1 \rangle$ ). This definition of the bounds l and u gives a more specific expression for Theo. 4.6 that relates the sensitivity of α and the coverage of observation induced by previous models $\theta _ { 1 } , \ldots , \theta _ { t }$

Theorem 4.9. Assume the conditions ofTheo. 4.6, that $\alpha ( x )$ is $( L _ { x } , 0 )$ -sensitive, and define l and u as in Eqs. 9–10. Then, for any deployed model $\phi ,$

$$
\| G ( \phi ) - \overline { { G } } ( \phi ) \| _ { 2 } \leq \frac { 2 L _ { x } } { \gamma } \operatorname* { s u p } _ { \theta \in \Theta } \mathbb { E } _ { D ( \phi ) } \left[ ( 1 - O ) \operatorname* { i n f } _ { \substack { x _ { \mathrm { o b s } } \in \mathcal { X } _ { 1 : t } ^ { \mathrm { o b s } } } } \| X - x _ { \mathrm { o b s } } \| _ { 2 } \| \nabla _ { \theta } \Delta _ { \ell } ( X , \theta ) \| _ { 2 } \right] .\tag{11}
$$

In particular,for binary cross-entropy with linear score $f _ { \theta } ( x ) = \theta ^ { \top } x a n d \| x \| _ { 2 } \leq R ,$

$$
\| G ( \phi ) - \overline { { G } } ( \phi ) \| _ { 2 } \leq \frac { 2 L _ { x } R } { \gamma } \mathbb { E } _ { D ( \phi ) } \left[ ( 1 - O ) \operatorname* { i n f } _ { x _ { \mathrm { o b s } } \in \mathcal { X } _ { 1 : t } ^ { \mathrm { o b s } } } \| X - x _ { \mathrm { o b s } } \| _ { 2 } \right] .\tag{12}
$$

This result shows that the gap between $G$ and $\overline { G }$ depends on the sensitivity constant $L _ { x }$ and the coverage of the feature space $\mathcal { X }$ by previous models $\theta _ { 1 } , \ldots , \theta _ { t }$ . This shows that the gap is smaller when $\alpha ( x )$ changes little for similar features x and when the models $\theta _ { t }$ cover a large region of the feature space. Additionally, for a fixed deployed model ϕ, the bound can only tighten over iterations, as $\chi _ { 1 : t } ^ { \mathrm { o b s } }$ can only grow.

## 5 Experiments

We evaluate our framework in a lending application with fairness regularization. This setting was introduced by Liu et al. [8] and studied in long-term fairness work [32, 40], but we adapt it to the performative setting. In this section, we evaluate Robust RRM when α(x) is independent of θ and present experiments with $L _ { \theta } \neq 0$ in Appendix H. Our experiments were designed to answer three main questions regarding Robust RRM: 1) if the performative gap $J ( \theta _ { t + 1 } , \theta _ { t + 1 } \mathbf { \bar { ) } } - J ( \theta _ { t } , \theta _ { t + 1 } )$ decreases when t increases; 2) whether there are improvements in approximation to the stable solution $\theta ^ { P S }$ by using the proposed methodology to improve l and u during learning; and 3) if Robust RRM approximates RRM better than compared baselines. To isolate Robust RRM convergence, we evaluate questions 1-3 assuming knowledge of $L _ { x } .$ . Finally, we evaluate whether Robust RRM performance is robust to misspecifications of $L _ { x } ,$ , and in Appendix E we evaluate when $L _ { x }$ is estimated from data.

Baselines We compare Robust RRM to RRM with Oracle access to labels. This baseline serves as a reference for attainable performance with full label access and to obtain $\theta ^ { P S }$ . Additionally, we compare our framework to the naive procedure of optimizing only on the selected data, following $\tilde { G }$ (Accept-only). We also compare our algorithm with baselines from the literature: Affine Risk Minimization (Affine RM) [23] — which was proposed to improve convergence rates in performative prediction; an expectation-maximization algorithm for selective labels (DCEM) [28]; and a doubly robust estimator (DR Estimator) — two baselines that handle partial observation without addressing performativity. See Appendix F for implementation details of each agent.

Experimental Setup We implemented a simulated environment where, at each round, the decisionmaker obtained access to $n = 1 0 0 { , } 0 0 0$ observations of variables $\{ x _ { i } , y _ { i } \cdot o _ { i } , o _ { i } \}$ . We use a large sample to reduce variance when estimating empirical losses. From this dataset, the decision-maker learns parameters $\theta \in \mathbb { R } ^ { d }$ of a logistic regression model. The model computes the linear score $f _ { \boldsymbol { \theta } } ( \boldsymbol { x } ) \overset { = } \boldsymbol { \theta } ^ { \top } \boldsymbol { x }$ and is trained with the binary cross-entropy loss with $\sigma ( f _ { \boldsymbol { \theta } } ( \boldsymbol { x } ) )$ as the probability of the positive label. Applicants are accepted when $f _ { \theta } ( x ) \geq \tau$ with $\tau = 0 ,$ , equivalent to $\sigma ( f _ { \theta } ( x ) ) \geq 0 . 5$ The environment then receives θ to output the updated distribution $\{ \bar { x } _ { i } ^ { \prime } , y _ { i } ^ { \prime } \cdot o _ { i } ^ { \prime } , o _ { i } ^ { \prime } \}$ . The process repeats for 400 iterations, where at each step the decision-maker updates θ to minimize risk under the updated data distribution according to each agent’s objective. We approximate the risk minimizer using gradient descent for 5,000 steps or until the loss decreases by less than $1 0 ^ { - 7 }$ in one iteration.

## 5.1 Fair Lending

In this experiment, we validate our worst-case approach on the long-term fairness environment introduced by Liu et al. [8], which Valdrighi et al. [32] have already discussed with selective labels. This environment considers that model decisions will alter an individual’s credit score, and the decision-maker should satisfy fairness in this new data distribution.

We leverage a continuous adaptation of the setting. Individuals are described by $\boldsymbol { x } = ( c , z , 1 )$ , where $c \in \mathbb { R }$ is the credit score and z is a binary sensitive attribute indicating whether individuals self-identify as Black or white and the last is used for the intercept. After deploying model θ, the credit score will be updated by the rule $c _ { \theta } = c + \epsilon ( \alpha ( x ) - 0 . 5 ) \cdot \sigma ( f _ { \theta } ( x ) )$ . Whenever $\alpha ( x ) > 0 . 5 .$ , the credit score will increase proportionally to the predicted probability $\dot { \sigma } ( \dot { f } _ { \boldsymbol { \theta } } ( \boldsymbol { x } ) )$ ), and when $\alpha ( x ) < 0 . 5$ , it will decrease. Additionally, in this environment, the decision-maker should satisfy demographic parity, which enforces an equal acceptance rate between groups [41], that is, $D P ( \boldsymbol { \phi } , \boldsymbol { \theta } ) : = | \bar { \mathbb { E } } _ { D ( \boldsymbol { \phi } ) } [ \sigma ( f _ { \boldsymbol { \theta } } ( \boldsymbol { X } ) ) | Z =$ $1 ] - \mathbb { E } _ { D ( \phi ) } [ \sigma ( f _ { \theta } ( X ) ) | Z = 0 ] |$ . We leverage a continuous and differentiable proxy loss as follows:

![](images/72663d3a674bce5aaf2fae9e6f31f255f424f6aa41b660a1e8f808a6e34d4632.jpg)  
(a)

![](images/a923d5453edb26ed73949ce44dfef60ed29a19c1651fbe4b56e26bfd7d8060a2.jpg)

(b)  
![](images/fd8a4178cc706c535a02995fce1ea69a38314be74e4d6e154ba7ae5f1fdc0eec.jpg)  
Figure 2: Evaluation of Robust RRM. (a) Performative gap across different sensitivity levels of the distribution map (averages and 90% interval over 10 random seeds). (b) Comparison of executing Robust RRM with trivial limits versus learned limits (averages and standard deviations over 10 random seeds).

$$
J _ { \lambda } ^ { \mathrm { f a i r } } ( \phi , \theta ) : = J ( \phi , \theta ) + \lambda ( \mathbb { E } _ { D ( \phi ) } [ \theta ^ { \top } X | Z = 1 ] - \mathbb { E } _ { D ( \phi ) } [ \theta ^ { \top } X | Z = 0 ] ) ^ { 2 }\tag{13}
$$

where λ is a weight set to 0.3 in our experiment. This extra term depends only on predicted scores, not on $Y$ , and is unaffected by selective labels. We calculate the initial distribution of sensitive groups and credit scores using the FICO [42] dataset and fit a linear interpolation to simulate $\alpha ( \cdot )$ . For a detailed description, see Appendix F.2. We obtain the data distribution $D ( \cdot )$ by first updating all credit scores c using the described rule, then resampling y based on $\alpha ( x )$ . The label distribution satisfies that $L _ { x } = 0 . 1 5$

Performative Gap As we discuss in Claim 4.3, the proposed loss $\overline { { \ell } } _ { \phi }$ is not necessarily $\beta \mathrm { . }$ -smooth; therefore, it is not straightforward to prove that G is a contraction with a stable point. For this reason, we empirically evaluate whether the proposed approach yields a small performative gap, that ${ \mathrm { i s } } ,$ the increase in loss after deploying $\theta _ { t + 1 }$ in the distribution it induces, compared to the training distribution of $\theta _ { t }$ . In Fig. 2a, we present the evolution of the performative gap over training for three different sensitivity constants $\epsilon \in \{ 5 0 , 9 0 , 1 5 0 \}$ }. In all configurations, the performative gap reaches below $1 0 ^ { - 4 }$ , presenting the lowest values when ϵ is smaller. However, the results differ little between $\epsilon = 9 0$ and $\epsilon = 1 5 0$

Improvements of Limits To analyze the procedure proposed in Sec. 4.1, we compare two variants of Robust RRM: using the trivial limits $\bar { l ( x ) } = 0 , \bar { u ( x ) } = 1$ and updating $l , u$ based on $L _ { x }$ and historical accepted data $\chi _ { 1 : t } ^ { \mathrm { o b s } }$ . Here, we show the average values of $l ( X )$ and $u ( X )$ for the rejected population, that $\mathbf { i s } , O = 0 .$ . We obtain these results with $\epsilon = 5 0$ . As depicted in Fig. 2b, the average width of the interval $| u ( x ) - l ( x )$ | quickly decreases to values around $0 . 1$ , decreasing during learning to reach values of 0.08. At the end of 400 iterations, the average values of u and l are around 0.24 and 0.16, respectively, reflecting the lower probability of the positive label for the rejected population. To compute $\dot { \theta } ^ { P S }$ , we run the Oracle agent for 400 iterations and select the final $\theta _ { t }$ . This improvement in interval width is reflected in how close the solutions $\theta _ { t }$ are to $\theta ^ { P S }$ . As shown in Fig. 2b, using the trivial limits keeps the distance around 1.5, while it decreases during learning to 0.212 with tighter limits $l , u$

Comparative Results We compare Robust RRM against baselines. In this experiment, we use the methodology to improve l and u and set $\epsilon = 5 0$ . In Fig. 3, we display the risk $J ( \theta _ { t } , \theta _ { t } )$ calculated with the complete information of $Y$ and the demographic parity $D P ( \theta _ { t } , \theta _ { t } )$ . These quantities capture loss and unfairness after deployment, including performative effects. Additionally, we present the distance to the stable solution $\mathbf { \check { \theta } } ^ { P S }$ of RRM with Oracle access to labels. The Robust RRM agent approximated the Oracle’s performance and had the smallest distance to the stable solution. As predicted by Theo. $4 . 7 ,$ the solutions $\theta _ { t }$ obtained by Robust RRM had a bounded distance to the true stable solution $\theta ^ { P \acute { S } }$ , which decreased over iterations. Other agents had a larger distance to the stable solution, with Accept-only and DR Estimator having the largest distances. The DR Estimator agent diverged, presenting a very high performative risk and disparity. This may occur because the necessary assumptions for such an agent do not hold with thresholded decisions. Similar results are obtained with $\epsilon \in \{ 9 0 , 1 5 0 \}$ } and are presented in Appendix G.

![](images/caf027c250b74df44b21a0a4de23ede40e7ab32eeadc946532a8cd38daf6f89b.jpg)  
Figure 3: Loss, disparity, and distance to the stable solution for RRM with different objective configurations using $\epsilon = 5 0$ . The y-axis was cropped to improve readability and does not display the full results for the DR Estimator agent. Results are averages and standard deviations over 10 random seeds.

Table 1: Results of Robust RRM with misspecified $L _ { x }$ . Results are averages (and standard deviations) over 10 random seeds. The distance to the stable solution reaches a maximum of 0.282 with +50% misspecification.
<table><tr><td>Constant used (%misspecification) Performative risk (↓) Dist. to</td><td></td><td> $\overline { { { \theta ^ { P S } \left( \downarrow \right) } } }$ </td></tr><tr><td> $\overline { { L _ { x } = 0 . 1 5 } }$  correctly specified</td><td> $\overline { { 0 . 3 3 6 ( \pm 0 . 0 0 1 ) } }$ </td><td> $\overline { { 0 . 2 1 2 ( \pm 0 . 0 2 5 ) } }$ </td></tr><tr><td> $\overline { { L _ { x } + 0 . 0 3 7 5 \left( + 2 5 \% \right) } }$ </td><td> $\overline { { 0 . 3 3 6 ( \pm 0 . 0 0 1 ) } }$ </td><td> $\overline { { 0 . 2 4 1 ( \pm 0 . 0 2 0 ) } }$ </td></tr><tr><td> $L _ { x } - 0 . 0 3 7 5 \ : ( - 2 5 \% )$ </td><td> $0 . 3 3 6 \dot { ( } \pm 0 . 0 0 1 \dot { ) }$ </td><td> $0 . 1 8 7 \dot { ( } \pm 0 . 0 1 8 \dot { ) }$ </td></tr><tr><td> $L _ { x } + 0 . 0 7 5 \ : ( + 5 0 \% )$ </td><td> $0 . 3 3 6 ( \pm 0 . 0 0 1 )$ </td><td> $0 . 2 8 2 ( \pm 0 . 0 2 8 )$ </td></tr><tr><td> $L _ { x } - 0 . 0 7 5 ( - 5 0 \% )$ </td><td> $0 . 3 3 6 ( \pm 0 . 0 0 1 )$ </td><td> $0 . 1 5 9 ( \pm 0 . 0 2 4 )$ </td></tr></table>

Robustness to the Misspecification of $L _ { x }$ To evaluate robustness to misspecification of $L _ { x } ,$ , we ran an additional experiment in which we perturbed the $L _ { x }$ value used by Robust RRM by using $L _ { x } + c$ for $c \in \{ - 0 . 0 7 5 , - 0 . 0 3 7 5 , 0 . 0 3 7 5 , + 0 . 0 7 5 \}$ . Tab. 1 reports results after 400 retraining iterations. Interestingly, underestimating $L _ { x }$ produced a smaller empirical distance to the stable solution. A smaller value of $L _ { x }$ produces tighter intervals $| u ( x ) - l ( { \dot { x } } ) |$ and therefore a smaller bound on the gap between the RRM and Robust RRM updates (Theo. 4.9), which in turn reduces κ in Theo. 4.7. However, when $L _ { x }$ is underestimated, it is not guaranteed that $\alpha ( x ) \in [ l ( x ) , u ( x ) ]$ for all $x ,$ and the original theoretical guarantee may not hold globally. Empirical results suggest that the limits remain valid for most of the feature space.

## 6 Discussion

Intervention and Observation In our framework, observation is defined by the final binary decisions obtained from thresholding predicted scores, while the model parameters define the population response. As Mofakhami et al. [6] note, individuals typically have access only to final decisions, not model parameters, which motivates defining distribution shift as a consequence of predictions rather than parameters, consistent with our definition of observation. However, the discontinuity introduced by thresholding can break the distribution map’s ϵ-sensitivity, and this setting has not yet been widely studied. In our experiments in the lending environment, we introduce a distribution map that approximates the scenario where the distribution shift is only dependent on the final prediction, by defining it as a function of $f _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ . While simple, this environment serves as an example in which performativity is related to final decisions.

Performative Optimality We present theoretical results regarding the stability of robust iterates. However, as Miller et al. [14] discuss, the stable solution can be far from optimal. We opt to focus on achieving stable solutions, as optimality depends on multiple model deployments and exploration, which typically requires access to a simulator of the world dynamics. With access to such a simulator, partial label observation is not an issue. The social cost of deploying suboptimal models to gather data and achieve optimality is an interesting direction for future work.

Limitations Our guarantees are stated at the population level; deriving finite-sample bounds would require separately characterizing the estimation error of the decoupled risk and of the label distribution learned from selectively observed data. Moreover, the limits in Sec. 4.1 require the sensitivity constants $L _ { x }$ and $L _ { \theta }$ . Under deterministic threshold policies, these constants are not identifiable from accepted samples alone. Our experiments in Sec. 5 isolate the behavior of Robust RRM by assuming $L _ { x }$ is known, evaluate its robustness to misspecification of $L _ { x } ,$ , and, in Appendix E, use an estimate obtained from a small set of randomly accepted applicants. Finally, we focus on deterministic threshold policies; stochastic policies can increase label coverage and improve utility and fairness under selective labels [24], and extending our framework to such policies is a promising direction for future work.

## 7 Conclusion

In this paper, we studied the intersection of performative prediction and selective labels, two properties common in many real-world machine learning applications. We focus on the convergence of RRM under partial-label observation with a deterministic threshold, and show that this is sufficient to break convergence to the stable point, even in the absence of performative effects. This motivated a robust formulation of the decoupled risk based on partial knowledge of the label distribution. Our approach leverages the loss function’s linearity in the positive-label probability to obtain a strongly convex upper bound that does not depend on the true label. Our robust approach uses this upper bound when we have no access to the true label. When we use it for RRM, we show that it approximates the stable point up to a distance related to the acceptance rate, the convexity of the loss function, and partial knowledge of the label distribution. Our last result shows how to improve our knowledge of the label distribution by leveraging previously accepted samples and their feature sensitivity. We evaluated our approach in a fair lending environment, where it approximated the Oracle agent’s performance.

## Acknowledgments

This project was supported, in part, by the Brazilian Ministry of Science, Technology, and Innovation, with resources from Law nº 8,248, of October 23, 1991, within the scope of PPI-SOFTEX, coordinated by Softex, and published in Arquitetura Cognitiva (Phase 3), DOU 01245.003479/2024 -10, and by the São Paulo Research Foundation (FAPESP), Brazil, process number #2024/17292-9.

This work has been partially supported by the project “Society-Aware Machine Learning: The paradigm shift demanded by society to trust machine learning,” funded by the European Union and led by IV (ERC-2021-STG, SAML, 101040177). Views and opinions expressed are, however, those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

Gen AI Usage Statement The core ideas, proposed methodologies, and experiments were developed solely by the authors. AI tools such as ChatGPT and Grammarly were used only to revise parts of the text for grammatical correctness and improve presentation.

## References

[1] Juan Perdomo, Tijana Zrnic, Celestine Mendler-Dünner, and Moritz Hardt. Performative prediction. In International Conference on Machine Learning, pages 7599–7609. PMLR, 2020.

[2] Celestine Mendler-Dünner, Juan Perdomo, Tijana Zrnic, and Moritz Hardt. Stochastic optimization for performative prediction. Advances in Neural Information Processing Systems, 33: 4929–4939, 2020.

[3] Qiang Li and Hoi-To Wai. Stochastic optimization schemes for performative prediction with nonconvex loss. Advances in Neural Information Processing Systems, 37:8673–8697, 2024.

[4] Qiang Li, Michal Yemini, and Hoi To Wai. Clipped SGD algorithms for performative prediction: Tight bounds for stochastic bias and remedies. In International Conference on Machine Learning, 2025.

[5] Gavin Brown, Shlomi Hod, and Iden Kalemaj. Performative prediction in a stateful world. In International Conference on Artificial Intelligence and Statistics, pages 6045–6061. PMLR, 2022.

[6] Mehrnaz Mofakhami, Ioannis Mitliagkas, and Gauthier Gidel. Performative prediction with neural networks. In International Conference on Artificial Intelligence and Statistics, pages 11079–11093. PMLR, 2023.

[7] Jian Chen, Zhehao Li, and Xiaojie Mao. Learning with selectively labeled data from multiple decision-makers. In International Conference on Machine Learning, 2025.

[8] Lydia T Liu, Sarah Dean, Esther Rolf, Max Simchowitz, and Moritz Hardt. Delayed impact of fair machine learning. In International Conference on Machine Learning, pages 3150–3158. PMLR, 2018.

[9] Moritz Hardt and Celestine Mendler-Dünner. Performative prediction: Past and future. Statistical Science, 40(3):417–436, 2025.

[10] Thomas Kehrenberg, Francisco Javier Sanguino Bautiste, Jose Lozano, and Novi Quadrianto. Dissecting performative prediction: A comprehensive survey. ACM Computing Surveys, 58(13): 1–35, 2026.

[11] Donald A MacKenzie, Fabian Muniesa, and Lucia Siu. Do Economists Make Markets?: On the Performativity ofEconomics. Princeton University Press, 2007.

[12] Zachary Izzo, Lexing Ying, and James Zou. How to learn when data reacts to your model: Performative gradient descent. In International Conference on Machine Learning, pages 4641–4650. PMLR, 2021.

[13] Zachary Izzo, James Zou, and Lexing Ying. How to learn when data gradually reacts to your model. In International Conference on Artificial Intelligence and Statistics, pages 3998–4035. PMLR, 2022.

[14] John P Miller, Juan C Perdomo, and Tijana Zrnic. Outside the echo chamber: Optimizing the performative risk. In International Conference on Machine Learning, pages 7710–7720. PMLR, 2021.

[15] Haitong Liu, Qiang Li, and Hoi To Wai. Two-timescale derivative free optimization for performative prediction with Markovian data. In International Conference on Machine Learning, pages 31425–31450. PMLR, 2024.

[16] Meena Jagadeesan, Tijana Zrnic, and Celestine Mendler-Dünner. Regret minimization with performative feedback. In International Conference on Machine Learning, pages 9760–9785. PMLR, 2022.

[17] Wenjing Yan and Xuanyu Cao. Zero-regret performative prediction under inequality constraints. Advances in Neural Information Processing Systems, 36:1298–1308, 2023.

[18] Alan Mishler and Niccolò Dalmasso. Fair when trained, unfair when deployed: Observable fairness measures are unstable in performative prediction settings. arXiv preprint arXiv:2202.05049, 2022.

[19] Yaowei Hu and Lu Zhang. Achieving long-term fairness in sequential decision making. In AAAI Conference on Artificial Intelligence, volume 36, pages 9549–9557, 2022.

[20] Kun Jin, Tian Xie, Yang Liu, and Xueru Zhang. Addressing polarization and unfairness in performative prediction. In AAAI Conference on Artificial Intelligence, volume 40, pages 22408–22416, 2026.

[21] Debmalya Mandal, Stelios Triantafyllou, and Goran Radanovic. Performative reinforcement learning. In International Conference on Machine Learning, pages 23642–23680. PMLR, 2023.

[22] Qiang Li and Hoi-To Wai. State dependent performative prediction with stochastic approximation. In International Conference on Artificial Intelligence and Statistics, pages 3164–3186. PMLR, 2022.

[23] Pedram Khorsandi, Rushil Gupta, Mehrnaz Mofakhami, Simon Lacoste-Julien, and Gauthier Gidel. Tight lower bounds and improved convergence in performative prediction. In Advances in Neural Information Processing Systems, 2025.

[24] Niki Kilbertus, Manuel Gomez Rodriguez, Bernhard Schölkopf, Krikamol Muandet, and Isabel Valera. Fair decisions despite imperfect predictions. In International Conference on Artificial Intelligence and Statistics, pages 277–287. PMLR, 2020.

[25] Himabindu Lakkaraju, Jon Kleinberg, Jure Leskovec, Jens Ludwig, and Sendhil Mullainathan. The selective labels problem: Evaluating algorithmic predictions in the presence of unobservables. In ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 275–284, 2017.

[26] Miriam Rateike, Ayan Majumdar, Olga Mineeva, Krishna P Gummadi, and Isabel Valera. Don’t throw it away! the utility of unlabeled data in fair decision making. In ACM Conference on Fairness, Accountability, and Transparency, pages 1421–1433, 2022.

[27] Vijay Keswani, Anay Mehrotra, and L. Elisa Celis. Fair classification with partial feedback: an exploration-based data collection approach. In International Conference on Machine Learning, 2024.

[28] Trenton Chang and Jenna Wiens. From biased selective labels to pseudo-labels: an expectationmaximization framework for learning from biased decisions. In International Conference on Machine Learning, 2024.

[29] Dennis Frauen, Valentyn Melnychuk, and Stefan Feuerriegel. Fair off-policy learning from observational data. In International Conference on Machine Learning, pages 13943–13972, 2024.

[30] Jake Fawkes, Nic Fishman, Mel Andrews, and Zachary Chase Lipton. The fragility of fairness: Causal sensitivity analysis for fair machine learning. In Advances in Neural Information Processing Systems Datasets and Benchmarks Track, 2024.

[31] Elliot Creager, David Madras, Toniann Pitassi, and Richard Zemel. Causal modeling for fairness in dynamical systems. In International Conference on Machine Learning, pages 2185–2195. PMLR, 2020.

[32] Giovani Valdrighi, Isabel Valera, and Marcos M. Raimundo. Long-term fairness with selective labels. In International Conference on Machine Learning, 2026.

[33] Moritz Hardt, Nimrod Megiddo, Christos Papadimitriou, and Mary Wootters. Strategic classification. In ACM Conference on Innovations in Theoretical Computer Science, pages 111–122, 2016.

[34] Keegan Harris, Chara Podimata, and Steven Z Wu. Strategic apple tasting. Advances in Neural Information Processing Systems, 36:79918–79945, 2023.

[35] Tianrun Zhao, Xiaojie Mao, and Yong Liang. Online strategic classification with noise and partial feedback. In Advances in Neural Information Processing Systems, 2025.

[36] José Pombal, Pedro Saleiro, Mário AT Figueiredo, and Pedro Bizarro. Prisoners of their own devices: How models induce data bias in performative prediction. arXiv preprint arXiv:2206.13183, 2022.

[37] Miriam Rateike, Isabel Valera, and Patrick Forré. Designing long-term group fair policies in dynamical systems. In ACM Conference on Fairness, Accountability, and Transparency, pages 20–50, 2024.

[38] Sebastian Zezulka and Konstantin Genin. From the fair distribution of predictions to the fair distribution of social goods: Evaluating the impact of fair machine learning on long-term unemployment. In ACM Conference on Fairness, Accountability, and Transparency, pages 1984–2006, 2024.

[39] Philip Boeken, Onno Zoeter, and Joris Mooij. Evaluating and correcting performative effects of decision support systems via causal domain shift. In Causal Learning and Reasoning, pages 551–569. PMLR, 2024.

[40] Alexander D’Amour, Hansa Srinivasan, James Atwood, Pallavi Baljekar, David Sculley, and Yoni Halpern. Fairness is not static: deeper understanding of long term fairness via simulation studies. In ACM Conference on Fairness, Accountability, and Transparency, pages 525–534, 2020.

[41] Ninareh Mehrabi, Fred Morstatter, Nripsuta Saxena, Kristina Lerman, and Aram Galstyan. A survey on bias and fairness in machine learning. ACM Computing Surveys (CSUR), 54(6):1–35, 2021.

[42] US Federal Reserve. Report to the congress on credit scoring and its effects on the availability and affordability of credit, 2007.

[43] George Casella and Roger Berger. Statistical inference. CRC press, 2002.

[44] Celestine Mendler-Dünner, Frances Ding, and Yixin Wang. Anticipating performativity by predicting from predictions. Advances in Neural Information Processing Systems, 35:31171– 31185, 2022.

[45] Michael P. Kim and Juan C. Perdomo. Making decisions under outcome performativity. In Innovations in Theoretical Computer Science Conference, Leibniz International Proceedings in Informatics (LIPIcs), pages 79:1–79:15, 2023.

## A Proofs

## A.1 Proof of Claim 4.3

We prove the inequality by first proving that the loss:

$$
\tilde { \ell } _ { \phi } ( x , \theta ) : = \operatorname* { m a x } _ { \substack { c \in \{ l ( x , \phi ) , u ( x , \phi ) \} } } \ell ( x , 0 , \theta ) + c \Delta _ { \ell } ( x , \theta )\tag{14}
$$

satisfies $\mathbb { E } _ { D ( \phi ) } [ \ell ( X , Y , \theta ) ] \le \mathbb { E } _ { D ( \phi ) } [ \tilde { \ell } _ { \phi } ( X , \theta ) ]$ and then that $\mathbb { E } _ { D ( \phi ) } [ \tilde { \ell } _ { \phi } ( X , \theta ) ] \le \mathbb { E } _ { D ( \phi ) } [ \bar { \ell } _ { \phi } ( X , \theta ) ]$ We prove each inequality separately. First, we can open the conditional expectation of the loss:

$$
\mathbb { E } _ { D ( \phi ) } [ \ell ( X , Y , \theta ) | X = x ] = \alpha _ { \phi } ( x ) \ell ( x , 1 , \theta ) + ( 1 - \alpha _ { \phi } ( x ) ) \ell ( x , 0 , \theta )\tag{15}
$$

$$
= \ell ( x , 0 , \theta ) + \alpha _ { \phi } ( x ) \Delta _ { \ell } ( x , \theta )\tag{16}
$$

Since this expression is a linear function of $\alpha _ { \phi } ( x )$ , and if AS. 4.1 holds, then $\alpha _ { \phi } ( x ) ~ \in$ $[ l ( x , \phi ) , u ( x , \phi ) ]$ . Thus:

$$
\ell ( x , 0 , \theta ) + \alpha _ { \phi } ( x ) \Delta _ { \ell } ( x , \theta ) \leq \operatorname* { m a x } _ { c \in \{ l ( x , \phi ) , u ( x , \phi ) \} } \ell ( x , 0 , \theta ) + c \Delta _ { \ell } ( x , \theta )\tag{17}
$$

This shows that $\mathbb { E } _ { D ( \phi ) } [ \ell ( X , Y , \theta ) | X = x ] \le \widetilde { \ell } _ { \phi } ( x , \theta )$ , which by integrating over the marginal distribution of X permits us to conclude that:

$$
\mathbb { E } _ { D ( \phi ) } [ \ell ( X , Y , \theta ) ] \le \mathbb { E } _ { D ( \phi ) } [ \tilde { \ell } _ { \phi } ( X , \theta ) ]\tag{18}
$$

Let $A : = l ( x , \phi ) \Delta _ { \ell } ( x , \theta )$ and $B : = u ( x , \phi ) \Delta _ { \ell } ( x , \theta )$ . The second inequality can be proved with the following steps:

$$
\tilde { \ell } _ { \phi } ( x , \theta ) = \ell ( x , 0 , \theta ) + \operatorname* { m a x } \{ A , B \}\tag{19}
$$

$$
= \ell ( x , 0 , \theta ) + \log ( \operatorname* { m a x } \{ e ^ { A } , e ^ { B } \} )\tag{20}
$$

$$
< \ell ( x , 0 , \theta ) + \log ( e ^ { A } + e ^ { B } )\tag{21}
$$

$$
= \overline { { \ell } } _ { \phi } ( x , \theta )\tag{22}
$$

The third line follows from the inequality ma $\alpha \{ e ^ { A } , e ^ { B } \} < e ^ { A } + e ^ { B }$ . Finally, as this result is valid for any x, θ, we can integrate over the distribution of X to obtain:

$$
\mathbb { E } _ { D ( \phi ) } [ \tilde { \ell } _ { \phi } ( X , \theta ) ] \leq \mathbb { E } _ { D ( \phi ) } [ \overline { { \ell } } _ { \phi } ( X , \theta ) ]\tag{23}
$$

We now prove that $\overline { { \ell } } _ { \phi }$ is γ-strongly convex. Fix x and ϕ, and write $a : = l ( x , \phi ) , b : = u ( x , \phi )$ . For $c \in \{ a , b \}$ , define $g _ { c } ( \theta ) : = ( 1 - c ) \ell ( x , 0 , \theta ) + c \ell ( x , 1 , \theta )$ . Since $c \in [ 0 , 1 ]$ and $\ell ( x , y , \theta )$ is γ-strongly convex in θ for each $y ,$ each $g _ { c }$ is also γ-strongly convex. Hence

$$
h _ { c } ( \theta ) : = g _ { c } ( \theta ) - \frac { \gamma } { 2 } \| \theta \| _ { 2 } ^ { 2 }\tag{24}
$$

is convex. Moreover,

$$
\overline { { \ell } } _ { \phi } ( x , \theta ) = \log \Big ( e ^ { g _ { a } ( \theta ) } + e ^ { g _ { b } ( \theta ) } \Big )\tag{25}
$$

$$
= \frac { \gamma } { 2 } \lVert \theta \rVert _ { 2 } ^ { 2 } + \log \Big ( e ^ { h _ { a } ( \theta ) } + e ^ { h _ { b } ( \theta ) } \Big ) .\tag{26}
$$

The log-sum-exp of convex functions is convex, so $\overline { { \ell } } _ { \phi }$ is γ-strongly convex in θ.

The loss $\overline { { \ell } } _ { \phi }$ is not necessarily β-smooth in x because l and u may not be smooth. It suffices to give a counterexample. Let $\Theta = [ \bar { - 1 } , 1 ] , \mathcal { X } = [ - 1 , 1 ]$ , and

$$
\ell ( x , 0 , \theta ) = \frac { \gamma } { 2 } \theta ^ { 2 } , \qquad \ell ( x , 1 , \theta ) = \frac { \gamma } { 2 } \theta ^ { 2 } + \theta .
$$

Then $\Delta _ { \ell } ( x , \theta ) = \theta$ . Choose

$$
u ( x , \phi ) = 1 , \qquad l ( x , \phi ) = 1 \{ x \geq 0 \} .
$$

These functions satisfy AS. 4.1, for instance, with $\alpha _ { \phi } ( x ) = 1$ for all $x ,$ but

$$
\nabla _ { \theta } \bar { \ell } _ { \phi } ( x , 0 ) = \left\{ { 1 / 2 } , \begin{array} { l l } { x < 0 , } \\ { 1 , } & { x \geq 0 . } \end{array} \right.
$$

Taking $x _ { n } = - 1 / n$ and $x _ { n } ^ { \prime } = 0$ , we get

$$
\left| \nabla _ { \theta } \overline { { \ell } } _ { \phi } ( x _ { n } , 0 ) - \nabla _ { \theta } \overline { { \ell } } _ { \phi } ( x _ { n } ^ { \prime } , 0 ) \right| = \frac { 1 } { 2 } , \qquad \left| x _ { n } - x _ { n } ^ { \prime } \right| = \frac { 1 } { n } .
$$

Thus no finite $\beta$ can satisfy the smoothness inequality. Hence $\overline { { \ell } } _ { \phi }$ is γ-strongly convex, but not $\beta .$ -smooth in general.

## A.2 Proof of Theo. 4.6

Proof. As $f ( \cdot ) : = J ( \phi , \cdot )$ and $\overline { { f } } ( \cdot ) : = \overline { { J } } ( \phi , \cdot )$ are both γ-strongly convex functions then:

$$
\| G ( \phi ) - \overline { { G } } ( \phi ) \| _ { 2 } \leq \frac { 1 } { \gamma } \operatorname* { s u p } _ { \theta \in \Theta } \| \nabla _ { \theta } f ( \theta ) - \nabla _ { \theta } \overline { { f } } ( \theta ) \| _ { 2 }\tag{27}
$$

We have that the difference between gradients is:

$$
\nabla _ { \theta } f ( \theta ) - \nabla _ { \theta } \overline { { f } } ( \theta ) = \mathbb { E } _ { D ( \phi ) } \left[ ( 1 - O ) ( \nabla _ { \theta } \ell ( X , Y , \theta ) - \nabla _ { \theta } \overline { { \ell } } _ { \phi } ( X , \theta ) ) \right]\tag{28}
$$

Since $O = { \bf 1 } \{ f _ { \phi } ( X ) \geq \tau \}$ is a deterministic function of $X$ , we can condition on $X = x$ inside the expectation, and we have:

$$
\mathbb { E } _ { D ( \phi ) } [ \nabla _ { \theta } \ell ( X , Y , \theta ) | X = x ] = \nabla _ { \theta } \ell ( x , 0 , \theta ) + \alpha _ { \phi } ( x ) \nabla _ { \theta } \Delta _ { \ell } ( x , \theta )\tag{29}
$$

$$
\begin{array} { r } { \mathbb { E } _ { D ( \phi ) } [ \nabla _ { \theta } \overline { { \ell } } _ { \phi } ( X , \theta ) | X = x ] = \nabla _ { \theta } \ell ( x , 0 , \theta ) + \nabla _ { \theta } \log ( e ^ { l ( x , \phi ) \Delta _ { \ell } ( x , \theta ) } + e ^ { u ( x , \phi ) \Delta _ { \ell } ( x , \theta ) } ) } \end{array}\tag{30}
$$

$$
= \nabla _ { \boldsymbol { \theta } } \ell ( x , 0 , \boldsymbol { \theta } ) + w _ { \boldsymbol { \phi } } ( x , \boldsymbol { \theta } ) \nabla _ { \boldsymbol { \theta } } \Delta _ { \ell } ( x , \boldsymbol { \theta } )\tag{31}
$$

where

$$
w _ { \phi } ( x , \theta ) = \frac { e ^ { l ( x , \phi ) \Delta _ { \ell } ( x , \theta ) } l ( x , \phi ) + e ^ { u ( x , \phi ) \Delta _ { \ell } ( x , \theta ) } u ( x , \phi ) } { e ^ { l ( x , \phi ) \Delta _ { \ell } ( x , \theta ) } + e ^ { u ( x , \phi ) \Delta _ { \ell } ( x , \theta ) } }\tag{32}
$$

And then we have that:

$$
\nabla _ { \theta } f ( \theta ) - \nabla _ { \theta } \overline { { f } } ( \theta ) = \mathbb { E } _ { D ( \phi ) } \left[ ( 1 - O ) ( \alpha _ { \phi } ( X ) - w _ { \phi } ( X , \theta ) ) \nabla _ { \theta } \Delta _ { \ell } ( X , \theta ) \right]\tag{33}
$$

$$
\begin{array} { r } { \implies \lVert \nabla _ { \theta } f ( \theta ) - \nabla _ { \theta } \overline { { f } } ( \theta ) \rVert _ { 2 } \le \mathbb { E } _ { D ( \phi ) } \left[ ( 1 - O ) | \alpha _ { \phi } ( X ) - w _ { \phi } ( X , \theta ) | \lVert \nabla _ { \theta } \Delta _ { \ell } ( X , \theta ) \rVert _ { 2 } \right] } \end{array}\tag{34}
$$

As $w _ { \phi } ( x , \theta )$ is a weighted mean of $l ( x , \phi )$ and $u ( x , \phi )$ , then $w _ { \phi } ( x , \theta ) \in [ l ( x , \phi ) , u ( x , \phi ) ]$ , then, $| \alpha _ { \phi } ( \dot { x } ) - \dot { w _ { \phi } } ( x , \theta ) | \leq | u ( x , \phi ) - l ( x , \phi ) |$ for any $x , \phi .$ Finally, we can conclude that:

$$
\| G ( \phi ) - \overline { { G } } ( \phi ) \| _ { 2 } \leq \frac { 1 } { \gamma } \operatorname* { s u p } _ { \theta \in \Theta } \mathbb { E } _ { D ( \phi ) } [ ( 1 - O ) | u ( X , \phi ) - l ( X , \phi ) | \| \nabla _ { \theta } \Delta _ { \ell } ( X , \theta ) \| _ { 2 } ]\tag{35}
$$

We analyze $\nabla _ { \theta } \Delta _ { \ell } ( X , \theta )$ for common definitions of ℓ:

• Logistic regression: $\ell ( x , y , \theta ) : = - \left( y \log ( \sigma ( \theta ^ { \top } x ) ) + ( 1 - y ) \log ( 1 - \sigma ( \theta ^ { \top } x ) ) \right)$ where $\sigma : \mathbb { R }  ( 0 , 1 )$ is the logistic function. We then have $\Delta _ { \ell } ( x , \theta ) = - \log ( \sigma ( \theta ^ { \top } x ) / ( 1 -$ $\sigma ( \theta ^ { \top } x ) ) ) = - \theta ^ { \top }$ x and then $\| \nabla _ { \theta } \Delta _ { \ell } ( x , \theta ) \| _ { 2 } = \| x \| _ { 2 }$

• Mean-squared error: $\ell ( x , y , \theta ) : = ( y - \theta ^ { \top } x ) ^ { 2 } .$ . Then, we have that $\Delta _ { \ell } ( x , \theta ) = ( 1 -$ $\begin{array} { r } { \theta ^ { \top } x ) ^ { 2 } - ( 0 - \theta ^ { \top } x ) ^ { 2 } = 1 - 2 \theta ^ { \top } x \mathrm { ~ a n d ~ } \nabla _ { \theta } \Delta _ { \ell } ( x , \theta ) = - 2 ( 1 - \theta ^ { \top } x ) x - 2 ( \theta ^ { \top } x ) x = - 2 x } \end{array}$ and $\lVert \nabla _ { \theta } \Delta _ { \ell } ( x , \theta ) \rVert _ { 2 } = 2 \lVert x \rVert _ { 2 }$

Note that adding a regularization term $\| \theta \| _ { 2 } ^ { 2 }$ to ensure that ℓ is γ-strongly convex leaves $\Delta _ { \ell } ( x , \theta )$ unchanged, since the penalization cancels out. 口

## A.3 Proof of Theo. 4.7

Proof. For any $\theta \in \Theta$ , we can bound the distance between the worst-case update $\overline { { G } } ( \theta )$ and the stable solution $\theta ^ { P S }$ using the triangle inequality and $G ( \theta ^ { P S } ) = \theta ^ { P S }$

$$
\| \overline { { G } } ( \theta ) - \theta ^ { P S } \| _ { 2 } \le \| \overline { { G } } ( \theta ) - G ( \theta ) \| _ { 2 } + \| G ( \theta ) - G ( \theta ^ { P S } ) \| _ { 2 }\tag{36}
$$

$$
\leq \kappa + \eta \Vert \theta - \theta ^ { P S } \Vert _ { 2 }\tag{37}
$$

where $\eta : = \epsilon \beta / \gamma$ . Unrolling this recursion, the sequence defined as $\theta _ { t + 1 } = \overline { { G } } ( \theta _ { t } )$ satisfies:

$$
\| \theta _ { t } - \theta ^ { P S } \| _ { 2 } \leq \eta ^ { t } \| \theta _ { 0 } - \theta ^ { P S } \| _ { 2 } + \kappa \sum _ { s = 0 } ^ { t - 1 } \eta ^ { s } = \eta ^ { t } \| \theta _ { 0 } - \theta ^ { P S } \| _ { 2 } + \frac { 1 - \eta ^ { t } } { 1 - \eta } \kappa\tag{38}
$$

where the right side converges to $\kappa / ( 1 - \eta )$ when $t \to \infty ,$ , since $\eta < 1$

## A.4 Proof of Theo. 4.9

Proof. As α is $( L _ { x } , 0 )$ -sensitive, for every $x _ { \mathrm { o b s } } \in \mathcal { X } _ { 1 : t } ^ { \mathrm { o b s } }$ we have $\alpha ( x _ { \mathrm { o b s } } ) - L _ { x } \| x _ { \mathrm { u n o b s } } - x _ { \mathrm { o b s } } \| _ { 2 } \leq$ $\alpha ( x _ { \mathrm { u n o b s } } ) \leq \alpha ( x _ { \mathrm { o b s } } ) + L _ { x } \| x _ { \mathrm { u n o b s } } - x _ { \mathrm { o b s } } \| _ { 2 }$ . Taking the supremum and infimum over $x _ { \mathrm { o b s } }$ , and using $\alpha ( x _ { \mathrm { u n o b s } } ) \in [ 0 , 1 ]$ for the clipping, the limits in Eqs. 9–10 satisfy $0 \leq l ( x _ { \mathrm { u n o b s } } ) \leq \alpha ( x _ { \mathrm { u n o b s } } ) \leq$ $u ( x _ { \mathrm { u n o b s } } ) \leq 1$ , so AS. 4.1 holds. Moreover, since clipping can only reduce $u ( x _ { \mathrm { u n o b s } } ) - l ( x _ { \mathrm { u n o b s } } )$ for any $x _ { \mathrm { o b s } }$ we have that:

$$
\forall x _ { \mathrm { o b s } } \in \mathcal { X } _ { 1 : t } ^ { \mathrm { o b s } } , u ( x _ { \mathrm { u n o b s } } ) - l ( x _ { \mathrm { u n o b s } } ) \leq 2 L _ { x } \| x _ { \mathrm { u n o b s } } - x _ { \mathrm { o b s } } \| _ { 2 } \implies\tag{39}
$$

$$
u ( x _ { \mathrm { u n o b s } } ) - l ( x _ { \mathrm { u n o b s } } ) \leq 2 L _ { x } \operatorname* { i n f } _ { x _ { \mathrm { o b s } } \in \mathcal { X } _ { 1 : t } ^ { \mathrm { o b s } } } \| x _ { \mathrm { u n o b s } } - x _ { \mathrm { o b s } } \| _ { 2 }\tag{40}
$$

Substituting this bound on $| u ( \cdot ) - l ( \cdot ) |$ into Theo. 4.6 proves the general result and the special case for binary cross-entropy. □

## B Bounding the Performative Risk of Robust RRM

Theo. 4.7 bounds the distance between iterations of Robust RRM and the stable solution in parameter space; however, it does not bound the performative risk of Robust RRM. We can bound the performative risk by combining Theo. 4.7 with Lemma 2.1 of [16] that has extra assumptions on the loss function ℓ. Define as ${ \cal P } \bar { R ( \theta ) } : = { \cal J } ( \theta , \theta )$ the performative risk. Then, the following lemma holds:

Lemma B.1 (Adapted from Lemma 2.1 of [16]). Ifthe loss $\ell ( x , y , \theta )$ is $L _ { 1 } { - } L i p s c h i t z$ in $( x , y )$ and $L _ { 2 } { - } L i p s c h i t { : }$ z in θ, and the distribution map $D ( \cdot )$ is ϵ−sensitive, then the performative risk $P R ( \cdot )$ is $( \epsilon L _ { 1 } + L _ { 2 } ) { - } L i p s c h i t z$

Now, we can use the Lipschitzness of $P R$ to bound the iterates of Robust RRM.

Proposition B.2 (Performative Risk of Robust RRM). Let $\theta _ { t }$ be an iterate of Robust RRM as in Def. 4.5. With the same assumptions as in Lemma B.1 andfrom Theo. 4.7, then:

$$
\operatorname* { l i m } _ { t \to \infty } P R ( \theta _ { t } ) \leq P R ( \theta ^ { P S } ) + ( \epsilon L _ { 1 } + L _ { 2 } ) \frac { \kappa } { 1 - \eta }\tag{41}
$$

Proof. The proof is direct from the application of the Lipschitzness of $P R ( \cdot )$ and the bound on the distance $\lVert { \boldsymbol { \theta } } _ { t } ^ { \bullet } - { \boldsymbol { \theta } } ^ { P S } \rVert _ { 2 } \colon$

$$
| P R ( \theta _ { t } ) - P R ( \theta ^ { P S } ) | \leq ( \epsilon L _ { 1 } + L _ { 2 } ) \| \theta _ { t } - \theta ^ { P S } \| _ { 2 }\tag{42}
$$

$$
\leq ( \epsilon L _ { 1 } + L _ { 2 } ) \left( \eta ^ { t } \| \theta _ { 0 } - \theta ^ { P S } \| _ { 2 } + \frac { 1 - \eta ^ { t } } { 1 - \eta } \kappa \right)\tag{43}
$$

Moving $P R ( \theta ^ { P S } )$ to the right side and applying the limit $t \to \infty$ concludes the proof.

While the bound may be conservative when $L _ { 1 }$ or $L _ { 2 }$ is large, our empirical evaluations consistently show that Robust RRM matches the performative risk attained by Oracle RRM.

## C Improving Label Distribution Limits with Model Dependence

In this section, we extend the methodology from Sec. 4.1 when the label is dependent on the deployed model θ, that is, $L _ { \theta } \neq 0$ . We use similar steps to the ones presented in Sec. 4.1. First, let $\hat { \mathcal { X } } _ { s } ^ { \mathrm { o b s } }$ be the support of the features accepted at round s and define $\begin{array} { r } { H _ { 1 : t } : = \bigcup _ { s = 1 } ^ { t } ( \mathcal { X } _ { s } ^ { \mathrm { o b s } } \times \{ \theta _ { s } \} ) } \end{array}$ ) as the set of previously accepted features x paired with the respective deployed model. Then, let ϕ be the currently deployed model and $x _ { \mathrm { u n o b s } }$ an unobserved sample. We define:

$$
d ( x _ { \mathrm { u n o b s } } , x _ { \mathrm { o b s } } , \phi , \theta ^ { \prime } ) : = L _ { x } \| x _ { \mathrm { u n o b s } } - x _ { \mathrm { o b s } } \| _ { 2 } + L _ { \theta } \| \phi - \theta ^ { \prime } \| _ { 2 }\tag{44}
$$

$$
l ( x _ { \mathrm { u n o b s } } , \phi ) : = \operatorname* { m a x } \{ 0 , \operatorname* { s u p } _ { ( x _ { \mathrm { o b s } } , \theta ^ { \prime } ) \in H _ { 1 : t } } \alpha _ { \theta ^ { \prime } } ( x _ { \mathrm { o b s } } ) - d ( x _ { \mathrm { u n o b s } } , x _ { \mathrm { o b s } } , \phi , \theta ^ { \prime } ) \}\tag{45}
$$

$$
u ( x _ { \mathrm { u n o b s } } , \phi ) : = \operatorname* { m i n } \{ 1 , \operatorname* { i n f } _ { ( x _ { \mathrm { o b s } } , \theta ^ { \prime } ) \in H _ { 1 : t } } \alpha _ { \theta ^ { \prime } } ( x _ { \mathrm { o b s } } ) + d ( x _ { \mathrm { u n o b s } } , x _ { \mathrm { o b s } } , \phi , \theta ^ { \prime } ) \}\tag{46}
$$

Under these definitions, since $| \alpha _ { \phi } ( x _ { \mathrm { u n o b s } } ) ~ - ~ \alpha _ { \theta ^ { \prime } } ( x _ { \mathrm { o b s } } ) | ~ \leq ~ d ( x _ { \mathrm { u n o b s } } , x _ { \mathrm { o b s } } , \phi , \theta ^ { \prime } )$ for every $( x _ { \mathrm { o b s } } , \theta ^ { \prime } ) \in H _ { 1 : t }$ , it holds that $l ( x _ { \mathrm { u n o b s } } , \phi ) \le \alpha _ { \phi } ( x _ { \mathrm { u n o b s } } ) \le u ( x _ { \mathrm { u n o b s } } , \phi )$ for any unobserved sample and deployed model $\phi .$ . Notice that this formulation reduces to the definition of limits in Sec. 4.1 when $L _ { \theta } = 0$ . Next, to update Theo. 4.9, it is only necessary to update the bound of $| u ( \cdot , \cdot ) - l ( \cdot , \cdot ) |$ as follows:

$$
\begin{array} { r } { | u ( x _ { \mathrm { u n o b s } } , \phi ) - l ( x _ { \mathrm { u n o b s } } , \phi ) | \leq 2 L _ { x } \| x _ { \mathrm { u n o b s } } - x _ { \mathrm { o b s } } \| _ { 2 } + 2 L _ { \theta } \| \phi - \theta ^ { \prime } \| _ { 2 } , \quad \forall ( x _ { \mathrm { o b s } } , \theta ^ { \prime } ) \in H _ { 1 : t } } \end{array}\tag{47}
$$

Theorem C.1. Assume the conditions ofTheo. 4.6, that $\alpha _ { \theta } ( x )$ is $( L _ { x } , L _ { \theta } )$ -sensitive, and define $l ,$ u as in Eqs. 45–46 and the distance d as in Eq. 44. Then,for any deployed model $\phi ,$

$$
\| G ( \phi ) - \overline { { G } } ( \phi ) \| _ { 2 } \leq \frac { 2 } { \gamma } \operatorname* { s u p } _ { \theta \in \Theta } \mathbb { E } _ { D ( \phi ) } \left[ ( 1 - O ) \operatorname* { i n f } _ { ( x _ { \mathrm { o b s } } , \theta ^ { \prime } ) \in H _ { 1 : t } } d ( X , x _ { \mathrm { o b s } } , \phi , \theta ^ { \prime } ) \| \nabla _ { \theta } \Delta _ { \ell } ( X , \theta ) \| _ { 2 } \right] .\tag{48}
$$

In particular,for binary cross-entropy with linear score $f _ { \boldsymbol { \theta } } ( x ) = \boldsymbol { \theta } ^ { \top }$ x and $\| x \| _ { 2 } \leq R$

$$
\| G ( \phi ) - \overline { G } ( \phi ) \| _ { 2 } \leq \frac { 2 R } { \gamma } \mathbb { E } _ { D ( \phi ) } \left[ ( 1 - O ) \operatorname* { i n f } _ { ( x _ { \mathrm { o b s } } , \theta ^ { \prime } ) \in H _ { 1 : t } } d ( X , x _ { \mathrm { o b s } } , \phi , \theta ^ { \prime } ) \right] .\tag{49}
$$

In this result, we moved $L _ { x }$ inside the distance function $d .$ . Despite this change, the interpretation of the results is the same: the one-step gap is dependent on the strong convexity of the loss function, the rejection rate, the dependence of the gradient of ℓ on $Y$ , and the width of the limits $| u ( \cdot , \cdot ) - l ( \cdot , \cdot ) |$ that depends on the proximity of the current model ϕ and variable X to the support of the historical acceptance $H _ { 1 : t }$ .

This approach to define l and u requires estimating the label distribution α within the accepted data. This is valid, as conditioned on acceptance, the label distribution is identifiable. However, in practice, when $L _ { \theta } \neq 0$ , we must estimate one $\alpha \theta$ for each deployed θ, because label probabilities differ for each previously used model θ. Our experiments in Appendix H demonstrate the applicability of our approach in this more general setting.

## D Extension to Multi-class Problems

In this section, we briefly discuss how to extend our approach to multi-class problems and the challenges it introduces. One initial consideration when moving from binary classification to multiclass classification is updating the thresholded definition of the observation variable O, as in the multi-class problem, the model does not predict a unidimensional score but $K$ scores for $K$ classes. One approach is to separate the predicted labels into two categories: labels with observation $O = 1$ and unobserved labels. In a lending example, we could consider three risk classes (high, medium, low) and observe only the label (accepting) of applicants classified as medium or low risk.

The label distribution is defined for each class $\alpha _ { \theta } ^ { k } ( x ) = P ( Y = k | X = x , \theta )$ , as well as the limits $l ^ { k } ( x , \theta ) , u ^ { k } ( x , \theta )$ . A worst-case loss can be obtained using the same ideas of the linearity of ℓ with respect to $\alpha _ { \theta } ^ { k }$ :

$$
\overline { { \ell } } _ { \phi } ( x , \theta ) : = \sum _ { k = 1 } ^ { K } \log \Big ( e ^ { l ^ { k } ( x , \phi ) \ell ( x , k , \theta ) } + e ^ { u ^ { k } ( x , \phi ) \ell ( x , k , \theta ) } \Big )\tag{50}
$$

Taking the expectation over x upper-bounds the risk $\mathbb { E } [ \ell ( X , Y , \theta ) ]$ . This worst-case loss follows from the fact that the conditional expectation of the loss over x is a linear function of $\alpha _ { \theta } ^ { k } ( x )$ , which attains its maximum at the interval endpoints. However, more work is necessary to study the effectiveness of this loss as it combines multiple maximum operations. A tighter bound could be obtained by using the constraint that $\begin{array} { r } { \sum _ { k } \alpha _ { \theta } ^ { k } ( x ) { \dot { = } } 1 } \end{array}$ . However, simply adding such a constraint could break the strong convexity of the worst-case loss.

## E Estimating $L _ { x }$ and $L _ { \theta }$ from Data

The proposed approach for improving limits l and u depends on knowing the constants $L _ { x }$ and $L _ { \theta }$ which control how much the probability of the positive label can change when the sample x or model θ changes. In this section we discuss how such constants could be obtained. Under a deterministic acceptance rule defined by a threshold, both constants are generally not identifiable from accepted samples alone, as regions of the feature space will have zero probability of observation. Two label distributions may agree on the entire accepted region while disagreeing in the unobserved region. Estimating therefore requires domain knowledge, exploration, additional structure, or a combination of them.

Table 2: Estimation of constant $L _ { x }$ under the assumption of a logistic label distribution and randomized acceptance, independently of model decision.
<table><tr><td>Number of samples  $\widehat { L _ { x } }$ </td><td>(and std. over 10 random seeds)</td></tr><tr><td>200</td><td> $\overline { { 0 . 4 5 7 ( \pm 0 . 1 4 8 ) } }$ </td></tr><tr><td>1000</td><td> $0 . 3 3 1 ( \pm 0 . 1 2 9 )$ </td></tr><tr><td>2000</td><td> $0 . 2 8 7 ( \pm 0 . 0 6 2 )$ </td></tr><tr><td>10000</td><td> $0 . 2 0 7 ( \pm 0 . 0 3 4 )$ </td></tr></table>

In low-dimensional applications with interpretable features, experts may provide bounds on how rapidly label probabilities can change. In the lending experiment, x comprises the credit score and sensitive attribute, so $L _ { x }$ represents an upper bound on how quickly repayment probability changes across nearby applicant profiles. Importantly, the expert need not identify the smallest Lipschitz constant: any valid conservative upper bound is sufficient.

Without domain knowledge, we can estimate the constants with a finite set of samples collected by randomly accepting applicants. Given additional structure to $\alpha _ { \theta } ( \cdot )$ , we can obtain specific expressions for the estimator of the constants. We will give one example under the assumption that the label distribution is a logistic function, that is, $\bar { \alpha _ { \theta } ( x ) } : = \sigma ( w _ { x } ^ { \star \top } x + w _ { \theta } ^ { \star \top } \theta + b )$ for an unknown vector $w ^ { \star } : = ( w _ { x } ^ { \star } , w _ { \theta } ^ { \star } )$ where $w _ { x } ^ { \star }$ contains the influence of features on the label distribution and $\boldsymbol { w } _ { \boldsymbol { \theta } } ^ { \star }$ the influence of model parameters.

By analyzing the maximum value of the gradient of $\alpha _ { \theta } ( x )$ we have that $L _ { x } ~ \le ~ \| w _ { x } ^ { \star } \| _ { 2 } / 4$ and $\dot { L _ { \theta } } \leq \| \dot { w _ { \theta } ^ { \star } } \| _ { 2 } \overline { { / 4 } }$ . It is therefore sufficient to obtain a conservative upper bound on the coefficient norm rather than identify the exact model. Let w be the MLE of $w ^ { \star }$ and p the failure probability. Under correct logistic specification and MLE regularity conditions, the Delta method [43, Sec. 5.5.4] gives the following asymptotic one-sided $( 1 - p )$ -confidence upper bound:

$$
\widehat { L } _ { x } : = \frac { 1 } { 4 } \left( \| \widehat { w } _ { x } \| _ { 2 } + z _ { 1 - p } \sqrt { \frac { \widehat { w } _ { x } ^ { \top } \widehat { V } \widehat { w } _ { x } } { \| \widehat { w } _ { x } \| _ { 2 } ^ { 2 } } } \right) \qquad \widehat { V } = ( { \mathbf { X } ^ { \top } \widehat { W } \mathbf { X } } ) ^ { - 1 }\tag{51}
$$

where $z _ { 1 - p }$ is the $1 - p$ quantile of the normal, X is the design matrix, $\widehat { V }$ is restricted to the coordinates of $w _ { x }$ , and $\widehat { W }$ is a diagonal matrix with $\widehat { W } [ i , i ] : = s _ { i } ( 1 - s _ { i } )$ where $s _ { i }$ is the predicted probability for sample i. An analogous expression is valid for $\widehat { L } _ { \theta }$

To demonstrate this procedure, we present in Tab. 2 the results of estimating $L _ { x } ~ = ~ 0 . 1 5$ with the assumption that $\bar { \alpha } _ { \theta } ( x ) = \sigma ( w _ { x } ^ { \star \top } x + b )$ using a confidence of 95% and different numbers of samples. To complement this, we ran the misspecification experiment with the overestimate $\widehat { L } _ { x } = 0 . 2 8 7$ obtained by accepting 2,000 random samples independently of the threshold decision. The performative risk increased by only 0.001 compared with using the correctly specified $L _ { x }$ , and the distance to the stable solution increased from $0 . { \overset { - } { 2 } } 1 2 ( \pm 0 . 0 2 5 )$ to $0 . 3 6 3 ( \pm 0 . 0 \dot { 2 } 4 )$ , which is lower than results from compared agents. In the lending environment, the label distribution is not modeled by a logistic function, so the result should be interpreted as a data-derived heuristic rather than a calibrated constant $L _ { x }$ . However, the additional experiment tests whether Robust RRM remains effective when instantiated with a practical estimate.

## F Implementation Details

Computer Resources All experiments were executed on CPUs with 8GB of RAM, with no GPU use. Experiments ran on a cluster with automatic job assignment across AMD Ryzen Threadripper PRO 7975WX and Intel(R) Xeon(R) Silver 4410Y.

## F.1 Algorithms Implementation

In our experiments, we do not have access to the true values of $J ( \phi , \theta )$ (or other objectives), only empirical estimates. We retrain repeatedly using gradient descent on the empirical estimate of the objective. At each retraining round, we run 5,000 steps of gradient descent to find the model that approximately minimizes the objective, stopping early when the change in loss after one gradient update is below $1 0 ^ { - 7 }$ . We apply $L 2$ regularization to the model parameters θ with a weight of $1 0 ^ { - 2 }$ to ensure that the objective is strongly convex.

Robust RRM To calculate limits l and u we keep a buffer of observations $( \left\{ x _ { i } , y _ { i } , \theta _ { i } \right\} )$ of maximum size 500, 000. We used this dataset to estimate $\alpha _ { \theta } ( x )$ within the observed feature space with a logistic regression model. Solving the maximization/minimization in the definitions of l and u involves evaluating distances $\lVert x _ { \mathrm { o b s } } - x _ { \mathrm { u n o b s } } \rVert _ { 2 }$ among observed samples $x _ { \mathrm { o b s } }$ and unobserved samples $x _ { \mathrm { u n o b s } }$ To avoid computing every pair of $\| x _ { \mathrm { o b s } } - x _ { \mathrm { u n o b s } } \| _ { 2 }$ , we used a KD-Tree and searched for the 50 closest neighbors for each $x _ { \mathrm { u n o b s } }$ . Then, we evaluate the complete definitions of l and u among these 50 samples. During our preliminary evaluation, increasing the neighborhood did not significantly change the results.

Oracle RRM, Accept-only and Affine RM We implemented all three variations of repeated risk minimization by changing only the data distribution the agent could access. Oracle RRM had access to the true distribution $D ( \theta )$ and full label observations. Accept-only leverages the selected distribution $\tilde { D } ( \theta )$ using only samples accepted by the previous model. Affine RM [23] uses a larger historical dataset to minimize risk. We implemented this using the three previous iterations of selectively labeled data, that is, $\begin{array} { r } { \tilde { D } ^ { 3 } ( \theta _ { t } ) : = \frac { 1 } { 3 } \sum _ { i = 0 } ^ { 2 } \tilde { D } ( \theta _ { t - i } ) } \end{array}$

DR Estimator and DCEM These two agents leverage both the accept-only distribution $\tilde { D } ( \theta _ { t } )$ and the full observation of features $X \sim D ( \bar { \theta } _ { t } )$ (without access to non-observed labels) to construct an estimate yˆ of unseen labels y. Both agents are based on the causal structure among features, observations, and labels. The first step is to estimate the propensity score $\pi ( x ) : = P ( O = 1 \mid X = x )$ which reflects the probability of observation. A basic assumption of the DR Estimator is that $\pi ( x ) \geq \delta > 0$ for a small $\delta \in \mathbb { R }$ , that is, every feature x has a minimal probability of observation, which does not hold with thresholded decisions. To adapt to our setting, we leverage $\sigma ( f _ { \boldsymbol { \theta } _ { t } } ( \boldsymbol { x } ) )$ ) as the propensity score.

The DR Estimator agent estimates the label distribution within accepted samples using a model $\mu : \mathbb { R } ^ { d }  [ 0 , 1 ]$ to approximate $P ( Y = 1 \mid X = x , O = 1 )$ . Then, the pseudo label is defined as:

$$
y ^ { \mathrm { D R } } = \mu ( x ) + \frac { o } { \sigma ( f _ { \theta _ { t } } ( x ) ) } ( y - \mu ( x ) )\tag{52}
$$

which then replaces the true label y and is optimized using the logistic regression loss.

The DCEM [28] algorithm is based on expectation-maximization. It was designed to handle selective labels when the selection mechanism depends on a sensitive attribute, while the true label does not. The algorithm iteratively runs an expectation step (E-step) where the model at iteration $k , \theta ( k )$ is used to define the pseudo-label:

$$
y ^ { \mathrm { D C E M } } = y \cdot o + ( 1 - o ) \cdot \sigma ( f _ { \theta ( k ) } ( x ) )\tag{53}
$$

Then, the maximization step (M-step) updates the parameter to be the minimizer of:

$$
\theta ( k + 1 ) : = \arg \operatorname* { m i n } _ { \theta } \mathbb { E } _ { D } [ \mathcal { L } ( y ^ { \mathrm { { D C E M } } } , \sigma ( f _ { \theta } ( x ) ) ) + y ^ { \mathrm { { D C E M } } } \mathcal { L } ( y \cdot o , \sigma ( f _ { \theta } ( x ) ) \cdot \sigma ( f _ { \theta ( k ) } ( x ) ) ) ]\tag{54}
$$

where $\begin{array} { r } { \mathcal { L } ( u , v ) = - u \log v - ( 1 - u ) \log ( 1 - v ) } \end{array}$ is the binary cross-entropy loss. The first term trains the predictor on the imputed pseudo-labels, while the second term acts as a causal regularizer that enforces consistency with the partial observation $Y \cdot O$ . We run the M-step using gradient descent and stop after 20 iterations of the E-M cycle or when the change in one update is less than $1 0 ^ { - 7 }$

## F.2 Environment Implementation

We implement the lending environment with a population of $n = 1 0 0 , 0 0 0$ individuals sampled from an initial distribution $D _ { 0 }$ based on the FICO dataset and updated according to each deployed model θ.

![](images/a6d7510173e5803eaf4d4ee8c1c15d04e60fb15ffd62c6f96cb94bcd5567c53c.jpg)

![](images/6958931fe04b566f6f39c84c73b1b1b235677caf69b2966f680fc173b0ead220.jpg)  
Figure 4: Distribution of scores and positive label calculated from the FICO dataset. $z = 1$ represents individuals who self-identify as Black and $z = 0$ individuals who self-identify as white.

![](images/752e839cb67aff9f5edd9a62567a679e3a73f3574fe5e0080ca887d2247cb6f8.jpg)  
Figure 5: Loss, disparity, and distance to the stable solution presented by RRM with different objective configurations using $\epsilon = 9 0$ . Results are averages (and std.) over 10 random seeds.

In this environment, the feature vector was composed of the credit score, the sensitive attribute, and a constant intercept feature, $\boldsymbol { x } = ( c , z , 1 )$ . We first compute the dataset’s demographic distribution, split 12%/88% between Black/white individuals, and set $z = 1$ for Black individuals. Then, after sampling z, we sample a credit score $c \in \{ 0 , \ldots , 1 0 0 \}$ following the empirical conditional distribution of $\dot { C } | \check { Z }$ We calculate the empirical positive rate for each pair $( c , z )$ to estimate $\widehat { \alpha } ( x ) : = \widehat { P } ( Y = 1 \mid c , z )$ To handle continuous credit scores, we fit a linear interpolation to define the final label distribution. Fig. 4 depicts both $P ( c | z )$ and $\alpha ( x ) = P ( y | c , z )$

## G Additional Results

We report results under the same configuration as in Sec. 5, but with different levels of performativity, using both $\epsilon = 9 0 ( \mathrm { F i g . } 5 )$ and $\epsilon = 1 5 0 ( \mathrm { F i g . } 6 )$ . In both configurations, Robust RRM achieved a performative risk similar to RRM and a small distance $\lVert \theta _ { t } - \theta ^ { \check { P } S } \rVert _ { 2 }$ to the stable solution. In both scenarios, Affine RM achieved the same distance to the stable solution as Robust RRM; however, it had the second-highest performative risk with $\epsilon = 9 0$ and the highest with $\epsilon = 1 5 0$ . This result suggests that proximity to the stable solution does not guarantee similar performative risk; however, Robust RRM still matched the Oracle’s performance. In both scenarios, DCEM also matched the Oracle’s performance, despite being farther from the stable solution. Future studies should examine the relationship between distance to the stable solution and distance in performative risk, extending the initial results from Appendix B.

## H Experiments with Outcome Performativity

In this section, we empirically evaluate Robust RRM when $L _ { \theta } \neq 0$ using the limit-definition methodology in Appendix C. We adapt the lending environment based on ideas of outcome performativity

![](images/aa1c230f04e19da06eedaeb648caa992ea839b6ca3a35fde632b9b10c8be1ad3.jpg)  
Figure 6: Loss, disparity, and distance to the stable solution for RRM with different objective configurations using $\epsilon = 1 5 0$ . Results are averages (and std.) over 10 random seeds.

![](images/1a594966e569e2b98a1ac80dbcf126537286b533423941abf6386b88ee407665.jpg)  
Figure 7: Loss, disparity, and distance to the stable solution in the environment with outcome performativity with $\epsilon = 5 0$ . The y-axis was cropped to improve readability and does not display the full results for the DR Estimator agent. Results are averages (and std.) over 10 random seeds.

[44, 45] where the model θ (or the final predicted score $\hat { y } )$ has a direct effect on y. We use the following linear effect for $\boldsymbol { \rho } \in \mathbb { R } ^ { d }$

$$
\alpha _ { \theta } ( x ) = \alpha ( x ) + \theta ^ { \top } \rho\tag{55}
$$

where $\alpha ( x )$ is the probability of the positive label before the model effect and $\rho$ sets the strength of the effect. $\alpha _ { \theta } ( x )$ is clipped to [0, 1] to represent a valid probability. $\operatorname { I f } \alpha ( x )$ is $( L _ { x } , 0 ) – \mathrm { s e n s i t i v e }$ then, by the Cauchy–Schwarz inequality, $\alpha _ { \theta } ( x )$ is $( L _ { x } , \| \rho \| _ { 2 } )$ )-sensitive. In the experiment, we set $\rho = ( 5 , 0 , 0 )$ , following the coordinates of $\boldsymbol { x } = ( c , z , 1 )$ , so only the coefficient associated with credit score affects the label distribution. This illustrates a setting in which placing greater positive weight on credit score induces greater effort to repay the loan and improve future score. We also fix $\epsilon = 5 0$

We compare Robust RRM with all agents presented in Sec. 5. To compute label limits $u , l ,$ we use a modified version that only accounts for the coordinate of θ that induces performativity; that is, we replace $L _ { \theta } \| \theta - \theta ^ { \prime } \| _ { 2 } \ b \mathbf { y } \ \lvert \dot { \rho } _ { 1 } \rvert \ \lvert [ \theta ] _ { 1 } - [ \theta ^ { \prime } ] _ { 1 } \rvert$ in Eqs. $4 5 \substack { - 4 6 } ( [ \theta ] _ { 1 }$ is the first coordinate of θ).

Fig. 7 displays results for all agents. Robust RRM achieved a similar performative risk and disparity to RRM with Oracle access to labels, and it also had the smallest distance to the stable solution. Affine RM and DCEM achieved similar performative risk, but Affine RM had a smaller distance to the stable solution. Similarly to experiments without outcome performativity, using only accepted data or the DR Estimator performed the worst, with the DR Estimator agent diverging during repeated retraining. This result indicates that Robust RRM is also effective when the model affects both the feature and label distributions.