# Robust Reinforcement Learning for Dynamic Treatment Regimes under Model and Measurement Uncertainty

Chengze Xie<sup>∗</sup> Xuan Wang<sup>†</sup> Ying Ding<sup>‡</sup> Hairong Wang<sup>§</sup> Yan Li<sup>¶</sup>

## Abstract

We develop a robust reinforcement learning (RL) framework for learning adaptive treatment policies that generalize across patients and remain reliable under errors in the measured biomarker and shifts in the disease dynamics. By modeling the patient as a hidden context, we address generalization and robustness in a unified manner through robust contextual Markov decision processes. Applying our framework to adaptive therapy for metastatic prostate cancer in a simulator calibrated to clinical data, we obtain generalizable and robust policies that substantially outperform the existing adaptive therapy method. In particular, we attain robustness with almost no loss of performance when the biomarker is accurate and the disease evolves as anticipated, and find that the resulting policy Pareto-dominates the existing method across a diverse array of scenarios. Our findings suggest that robust RL holds great promise as a principled methodology for learning reliable adaptive treatment policies.

## 1 Introduction

Adapting treatment to the evolving condition of a patient has received considerable attention in medicine, with success reported in a broad range of applications [4, 7, 25, 26, 39, 50]. In oncology, adaptive therapy pursues this idea to delay the emergence of drug resistance [18]. Its rationale is that, under continuous treatment, drug-sensitive cells are eliminated and drug-resistant cells are left to grow without competition. By pausing treatment once the tumor has responded and resuming it once the tumor has regrown, clinicians can retain sensitive cells that continue to suppress the resistant ones [18, 45]. This approach was tested in a pilot clinical trial of the drug abiraterone for metastatic castration-resistant prostate cancer [52, 53]. In this clinical trial, clinicians paused abiraterone once prostate-specific antigen (PSA), a blood marker that tracks tumor burden, fell below half of its pretreatment value, and resumed it once PSA returned to the pretreatment value, a protocol we refer to as AT50. Compared with a contemporaneous cohort who received continuous treatment with abiraterone as the standard of care, the median time to progression more than doubled under AT50, while patients received no abiraterone during 46% of their time in the clinical trial [52].

Despite its remarkable success, AT50 is by design a simple heuristic in the sense that it applies the same pair of thresholds to every patient and responds only to the most recent PSA value. In addition, its benefit has so far been observed in only 17 patients, and, based on the models fitted to the same data, other schedules are expected to further improve outcomes [52]. Existing eforts toward more refined schedules include optimal control of competition models [9], deep reinforcement learning (RL) for controlling the emergence of drug resistance [12], and more recently RL policies trained within models calibrated to prostate cancer patients [16,33,34]. Nevertheless, these policies are evaluated within the same models used for their training, and their performance beyond these models remains largely unexamined. Ideally, one would like to learn, in a data-driven manner, a dynamic treatment policy that is more sophisticated than AT50, attains improved performance, and generalizes to the broad population of patients beyond those enrolled in the clinical trial. More importantly, such a policy should remain reliable under the shifts between training and deployment that are common in clinical practice [13].

To be more specific, we identify three properties that a treatment policy learned from data should possess. To begin with, the policy should generalize in the sense that the performance it attains on the patients seen during training carries over to new patients drawn from the same population. From the perspective of contextual and hidden-parameter Markov decision processes (MDPs) [11,19,28,44], the natural goal without interacting with a new patient is a policy that is near-optimal on average over the population [49], and generalization across populations of environments has been studied extensively in deep RL [24, 46]. With a simulation model, we can pursue this goal by training on as many simulated patients as needed, far beyond the 17 patients of the pilot clinical trial, provided that the simulated population is broad enough to contain the patients of interest. In addition, the policy should be robust to errors in its observation. In particular, PSA serves only as an indirect measure of the underlying tumor burden, and radiographic progression can occur without any rise in PSA [6]. Consequently, the quantity on which the policy acts may deviate from the quantity it aims to control, while RL policies are known to be sensitive to even small adversarial perturbations of their observations [21,51]. Most importantly, the policy should retain reliable performance under shifts in the patient dynamics. Indeed, the parameters of the model, including the strength of competition between sensitive and resistant cells on which the success of adaptive therapy hinges, are estimated from the sparse PSA records of a small cohort of patients, and are hence subject to considerable statistical error [5,52]. Such shifts are particularly consequential in medical domains, where real patients are scarce and the dynamics of a patient may further change during treatment [1].

To address these requirements, in this manuscript we propose to study adaptive therapy through the lens of robust contextual MDPs. A contextual MDP [19] describes a family of MDPs whose transition dynamics are indexed by a context, drawn from a population and hidden from the decision maker. Adaptive therapy fits naturally into this framework once we identify the context with the patient, namely the initia composition of the tumor together with its competition and growth parameters. Each patient then defines its own tumor dynamics, to be inferred by the clinician from the history of PSA measurements and treatment decisions, and AT50 is one particular policy acting on this history. By viewing the patient as a context, we treat generalization and robustness to shifts in the patient dynamics in a unified manner, since both concern the population from which the context is drawn. Specifically, generalization requires the policy to perform well on new contexts drawn from this population, which can be attained by training on suficiently many sampled contexts, while robustness requires the policy to remain reliable when the context deviates from this population and thereby shifts the transition kernel that it determines.

To attain such robustness, we draw on robust MDPs, in which one seeks the best worst-case performance over an ambiguity set of transition kernels, and which have been studied extensively for standard MDPs [22, 31, 38, 40, 41, 48, 51]. We carry this viewpoint over to the contextual MDP and formulate robustness as a Markov game between the controller and the context, in which the context may deviate adversarially from the population from which it is drawn [29, 32]. To account for errors in the observation, we further train the policy under random perturbations of the measured PSA. Within a model of abiraterone therapy fitted to clinical data [52], which we refer to as the nominal model, the resulting robust policies substantially outperform AT50. In addition, we attain robustness to shifts in the patient dynamics at almost no price in the nominal model, and the resulting policy nearly Pareto-dominates the nominally trained policy, i.e., the policy trained in the nominal model with exact observations, under a broad range of realistic errors in the observation and shifts in the patient dynamics. Our contributions can be summarized as follows.

First, we formulate adaptive therapy as a contextual MDP in which the patient is modeled as a hidden context, and find that a policy learned within this formulation substantially outperforms AT50 and generalizes to patients fitted to clinical data. Within the competition model of [52], the learned policy extends the mean progression-free time from 29.58 months under AT50 to 50.35 months, while treating in a comparable fraction of months. In addition, on the 32 patients whose initial tumor compositions are fitted to clinical data in [52], policies trained on a broad synthetic population constructed without reference to these patients perform comparably to policies trained on a population calibrated to them. Hence, generalization to the patients of interest can be attained by training on suficiently many simulated patients drawn from a broad population, without tailoring the population to these patients.

Second, we find that errors in PSA of realistic magnitude substantially reduce the benefit of policies that do not account for them. Specifically, under random errors of up to about 20% of the true PSA value, the nominally trained policy loses a substantial part of its benefit, and AT50 performs only marginally better than continuous treatment. By training under such errors, we obtain a policy that retains the benefit of adaptive therapy in the sense that its performance under such errors nearly matches that without them.

Finally, we formulate robustness to shifts in the patient dynamics as a Markov game between the controller and the context over an ambiguity set of transition kernels [29], and find that the resulting kernel-robust policy nearly Pareto-dominates the nominally trained policy and AT50 across all the conditions we consider. Specifically, the kernel-robust policy (1) performs nearly as well as the nominally trained policy in the nominal model, (2) substantially outperforms every other type of policy under shifts in the dynamics, and (3) substantially improves on the nominally trained policy and AT50 even under errors in the observation. Our findings suggest that robustness to shifts in the patient dynamics can be attained at almost no price in the nominal model, and that the resulting policy remains robust against a broad spectrum of errors in the observation and shifts in the patient dynamics. In view of this, we believe that robust RL holds great promise as a principled methodology for learning adaptive therapy policies.

The rest of this manuscript is organized as follows. In Section 2, we introduce contextual MDPs and their robust counterparts, and formulate adaptive therapy within this framework. We present the numerical experiments in Section 3, and conclude the manuscript in Section 4.

## 2 Problem Formulation

In this section, we discuss contextual MDPs and their robust counterparts, and formulate adaptive therapy for metastatic prostate cancer as a concrete instance. Specifically, in Section 2.1, we introduce contextual MDPs and establish finite-sample guarantees on the number of sampled contexts needed to learn a generalizable policy. The concrete instance for adaptive therapy is then introduced in Section 2.2, where we adopt the model of abiraterone therapy in [52], which is fitted to clinical data from a pilot clinical trial and a contemporaneous cohort. In Section 2.3, we further introduce the robust counterparts of the contextual MDP against errors in the observation and shifts in the patient dynamics.

## 2.1 Contextual Markov Decision Processes

We consider a finite-horizon contextual MDP [19, 44] with state space $x ,$ , action space ${ \mathcal { A } } ,$ observation space ${ \mathcal { V } } ,$ horizon $T ,$ and context space C. Each context $c \in { \mathcal { C } }$ specifies an initial distribution $\mu _ { c }$ over $\mathcal { X } , \mathrm { a }$ transition kernel $P _ { c } ,$ an observation map $O _ { c } : \mathcal { X }  \mathcal { Y }$ , and a reward function $r _ { c } : \mathcal { X } \times \mathcal { A }  \mathbb { R }$ , where ${ P _ { c } ( \cdot | x , a ) }$ denotes the distribution of the next state when action a is taken in state x. At the beginning of an episode, a context c is drawn from a population distribution D and remains fixed thereafter, after which the initial state is drawn as $x _ { 0 } \sim \mu _ { c }$ , and the state at every period t evolves as $x _ { t + 1 } \sim P _ { c } ( \cdot | x _ { t } , a _ { t } )$ . The decision maker observes neither c nor the state $x _ { t }$ , and receives at period t the observation $y _ { t } = O _ { c } ( x _ { t } )$ . Consequently, the action at period t can only be based on the history $h _ { t } = ( y _ { 0 } , a _ { 0 } , \ldots , a _ { t - 1 } , y _ { t } )$ of past observations and actions, i.e., a policy $\pi = ( \pi _ { t } ) _ { t \geq 0 }$ selects $a _ { t } \sim \pi _ { t } ( \cdot | h _ { t } )$ , and Π denotes a given class of such history-dependent policies. For each context $c ,$ let $\begin{array} { r } { J ( \pi ; c ) : = \mathbb { E } _ { c } ^ { \pi } \left[ \sum _ { t = 0 } ^ { T - 1 } r _ { c } ( x _ { t } , a _ { t } ) \right] } \end{array}$ denote the expected cumulative reward of $\pi ,$ where $\mathbb { E } _ { c } ^ { \pi }$ denotes the expectation over the trajectory generated by $\pi$ under $\mu _ { c }$ and $P _ { c }$ . We consider

$$
\operatorname* { m a x } _ { \pi \in \Pi } J ( \pi ; \mathcal { D } ) : = \mathbb { E } _ { c \sim \mathcal { D } } \left[ J ( \pi ; c ) \right] .\tag{2.1}
$$

In medical applications, the context collects the parameters that determine the dynamics of an individual patient, which vary across patients and are not observed by the decision maker. Since the context is hidden, the decision maker must infer it from the history, and an optimal policy for (2.1) is in general historydependent [23,28]. In (2.1), we average the performance over the population of patients, which is the natural goal when a policy is deployed on a new patient without further interaction [49]. In practice, the population distribution $\mathcal { D }$ is unknown, and is accessible only through sampled contexts, such as the parameters of patients fitted to clinical data or of virtual patients generated in a simulator. Given contexts $c _ { 1 } , \ldots , c _ { N }$ drawn independently from D, we consider the sample average approximation (SAA) [43] of (2.1),

$$
\operatorname* { m a x } _ { \pi \in \Pi } \widehat { J } _ { N } ( \pi ) : = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } J ( \pi ; c _ { i } ) .\tag{2.2}
$$

We proceed to show that an optimal solution of (2.2) generalizes to new contexts drawn from D once suficiently many contexts are sampled. To begin with, we establish such a guarantee for finite state, observation, and action spaces.

Theorem 2.1. Suppose X , Y, and A are finite, and $0 \leq r _ { c } ( x , a ) \leq \bar { r }$ for all $( c , x , a ) \in \mathcal { C } \times \mathcal { X } \times \mathcal { A }$ . Let $\begin{array} { r } { | \mathcal { H } | = \sum _ { t = 0 } ^ { T - 1 } | \mathcal { V } | ^ { t + 1 } | \mathcal { A } | ^ { t } } \end{array}$ denote the number of histories $h _ { t }$ with $t \leq T - 1$ . Then for any class Π of historydependent policies, any $\widehat { \pi } _ { N } \in \mathrm { a r g m a x } _ { \pi \in \Pi } \widehat { J } _ { N } ( \pi )$ , and any $\delta \in \mathsf { \Gamma } ( 0 , 1 )$ , with probability at least $1 - \delta _ { z }$ , we have

$$
J ( \widehat { \pi } _ { N } ; \mathcal { D } ) \ge \operatorname* { s u p } _ { \pi \in \Pi } J ( \pi ; \mathcal { D } ) - T \bar { r } \sqrt { \frac { 2 ( | A | - 1 ) | \mathcal { H } | \log \left( 1 + 8 T \sqrt { N } \right) + 2 \log ( 2 / \delta ) } { N } } - \frac { T \bar { r } } { \sqrt { N } } .
$$

In view of Theorem 2.1, the optimality gap of $\widehat { \pi } _ { N }$ on the population is $\mathcal { O } \big ( T \bar { r } \sqrt { | \boldsymbol { A } | | \mathcal { H } | \log ( T N ) / N } \big )$ with high probability. Consequently, a policy trained on suficiently many sampled contexts generalizes to new contexts drawn from D, provided that D is broad enough to contain the contexts of interest. In Section 3.6, we demonstrate such generalization for adaptive therapy by applying policies trained on a broad synthetic population to the patients fitted to clinical data in [52] without further training. The proof of Theorem 2.1 is deferred to Appendix A.

On the other hand, the number $| \mathcal { H } |$ of histories, and hence the number of sampled contexts required by Theorem 2.1, grows exponentially in the horizon T. We proceed to show that the exponential dependence on $T$ can be avoided when the expected cumulative reward $J ( \pi ; c )$ is Lipschitz continuous with respect to the context c.

Theorem 2.2. Suppose that $\mathcal { C } \subset \mathbb { R } ^ { d }$ is compact with diameter diam(C) under the Euclidean norm, $0 \leq$ $r _ { c } ( x , a ) \leq \bar { r } f o r$ all $( c , x , a ) \in \mathcal { C } \times \mathcal { X } \times \mathcal { A }$ , and there exists $L _ { T } < \infty$ such that

$$
| J ( \pi ; c ) - J ( \pi ; c ^ { \prime } ) | \leq L _ { T } \| c - c ^ { \prime } \| _ { 2 } \qquad f o r \ a l l \ \pi \in \Pi \ a n d \ c , c ^ { \prime } \in \mathcal C .\tag{2.3}
$$

Then there exists a constant $C _ { d }$ depending only on d such that for any $\widehat { \pi } _ { N } \in \mathrm { a r g m a x } _ { \pi \in \Pi } \widehat { J } _ { N } ( \pi )$ and any $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta$ , we have

$$
J ( \widehat { \pi } _ { N } ; \mathcal { D } ) \ \geq \ \operatorname* { s u p } _ { \pi \in \Pi } J ( \pi ; \mathcal { D } ) - 2 L _ { T } \mathrm { d i a m } ( \mathcal { C } ) \left( \frac { C _ { d } \log ( 2 N ) } { N ^ { \operatorname* { m i n } \{ 1 / 2 , 1 / d \} } } + \sqrt { \frac { \log ( 1 / \delta ) } { 2 N } } \right) .
$$

In view of Theorem 2.2, the exponential dependence on $T$ through |H| in Theorem 2.1 is replaced by $L _ { T }$ , which grows only polynomially in $T$ when the dynamics are Lipschitz continuous with respect to the context. To be more specific, we establish in the following proposition the Lipschitz continuity (2.3) for any class of policies, when the observation map does not depend on the context, and the initial distribution, the transition kernel, and the reward are Lipschitz continuous with respect to the context.

Proposition 2.1. Suppose $\mathcal { C } \subset \mathbb { R } ^ { d } , A$ is finite, the observation map $O _ { c } = O$ is independent of $c , \ 0 \ \leq$ $r _ { c } ( x , a ) \leq \bar { r }$ for all $( c , x , a ) \in \mathcal { C } \times \mathcal { X } \times \mathcal { A }$ , and there exists a constant $L \geq 0$ such that

$$
\begin{array} { r l r } & { } & { d _ { \mathrm { T V } } ( \mu _ { c } , \mu _ { c ^ { \prime } } ) \leq L \| c - c ^ { \prime } \| _ { 2 } , } \\ & { } & { d _ { \mathrm { T V } } \big ( P _ { c } ( \cdot \| x , a ) , P _ { c ^ { \prime } } ( \cdot \| x , a ) \big ) \leq L \| c - c ^ { \prime } \| _ { 2 } , } \\ & { } & { | r _ { c } ( x , a ) - r _ { c ^ { \prime } } ( x , a ) | \leq L \| c - c ^ { \prime } \| _ { 2 } } \end{array}
$$

for all $c , c ^ { \prime } \in \mathcal { C } , x \in \mathcal { X }$ , and $a \in A .$ where $d _ { \mathrm { T V } } ( P , Q ) : = \operatorname* { s u p } _ { B } | P ( B ) - Q ( B ) |$ denotes the total variation distance, and the supremum is taken over all measurable sets B. Then (2.3) holds for any class Π of history-dependent policies with

$$
L _ { T } = L T \left( 1 + \frac { \bar { r } ( T + 1 ) } { 2 } \right) .
$$

Given Proposition 2.1, we have $L _ { T } = \mathcal { O } \big ( L T ( 1 + \bar { r } T ) \big )$ , and hence Theorem 2.2 avoids the exponential dependence on T in Theorem 2.1. It is interesting to note that Theorem 2.2 does not require finite state, observation, and action spaces. On the other hand, the rate $\mathcal { O } \left( N ^ { - \operatorname* { m i n } \left\{ 1 / 2 , 1 / d \right\} } \log N \right)$ deteriorates with the dimension d once $d > 2$ . Nevertheless, such deterioration is mild for systems with a context of small dimension. For instance, the context of adaptive therapy in Section 2.2 is four-dimensional, which yields the rate $\mathcal { O } ( N ^ { - 1 / 4 } \log N )$ . The proofs of Theorem 2.2 and Proposition 2.1 are deferred to Appendix A.

## 2.2 Adaptive Therapy as a Contextual MDP

We adopt the model of abiraterone therapy in [52], which describes the competition between drug-sensitive and drug-resistant cancer cells. Let $S ( \tau )$ and $R ( \tau )$ denote the numbers of drug-sensitive and drug-resistant cells at time $\tau ,$ measured in days, and let $u ( \tau ) \in \{ 0 , 1 \}$ indicate whether abiraterone is administered. The dynamics are given by

$$
\frac { d S ( \tau ) } { d \tau } = r _ { S } S ( \tau ) \left( 1 - \frac { S ( \tau ) + R ( \tau ) } { \kappa \big ( 1 - \lambda u ( \tau ) \big ) } \right) , \qquad \frac { d R ( \tau ) } { d \tau } = r _ { R } R ( \tau ) \left( 1 - \frac { \alpha S ( \tau ) + R ( \tau ) } \kappa \right) ,\tag{2.4}
$$

where κ is the carrying capacity, i.e., the maximal number of cells supported by the available nutrients and space, and $r _ { S } = \beta \gamma _ { S }$ and $r _ { R } = \beta \gamma _ { R }$ are the growth rates of the sensitive and resistant cells per day, with base rates $\gamma _ { S } , \gamma _ { R }$ and a common factor $\beta$ scaling both rates.

In essence, (2.4) describes the diferent responses of the drug-sensitive and drug-resistant cells to abiraterone and the competition between them, which together underlie the rationale of adaptive therapy. Specifically, the sensitive cells depend on testosterone for survival and proliferation, while the resistant cells are independent of testosterone [52]. Since the synthesis of testosterone is inhibited by abiraterone, the sensitive cells utilize nutrients and space less eficiently under treatment, which is modeled in (2.4) by a reduction of their carrying capacity by the fraction $\lambda \in ( 0 , 1 )$ . In contrast, the resistant cells are not afected by abiraterone, and their growth is suppressed only through the competition from the sensitive cells, whose competitive efect on the resistant cells is measured by α. Consequently, once the sensitive cells are depleted under continuous treatment, the resistant cells are released from competition and eventually dominate the tumor. In adaptive therapy, treatment is instead paused before such depletion occurs, which allows the sensitive cells to recover during the treatment break and to continue suppressing the resistant ones.

Following [52], we set $( \lambda , \gamma _ { S } , \gamma _ { R } , \alpha , \beta ) = ( 0 . 9 5 , 0 . 0 1 5 6 / 6 , 0 . 0 0 9 1 / 6 , 6 , 8 )$ , which are estimated by fitting (2.4) to the longitudinal PSA records of 32 patients, namely 17 patients of the pilot clinical trial and 15 patients of a contemporaneous cohort who received continuous treatment with abiraterone as the standard of care. We refer to $( \alpha , \beta ) = ( 6 , 8 )$ , which are shared by all fitted patients, as the nominal values of the competition and growth parameters. We also take the initial cell numbers of each patient from the same fit (see Appendix B), and set $\kappa = 1 0 ^ { 4 }$ as in [52], which only determines the unit of the cell numbers, since rescaling S, R, and κ by a common factor leaves (2.4) unchanged. Since α and $\beta$ are estimated from the records of only 32 patients and are assumed to be shared by all of them, their values are subject to considerable statistical error, and may also difer across patients, as suggested in [52] by the poorer fits for three patients. Hence, in Sections 2.3 and 3, we allow both parameters to vary, and seek policies that remain reliable under shifts in them.

Since PSA is measured monthly in the pilot clinical trial [52], the clinician decides at the beginning of every month of 28 days whether to administer abiraterone until the next visit, i.e., chooses $a _ { t } \in \{ 0 , 1 \}$ at visit $t \in \{ 0 , 1 , \ldots , T - 1 \}$ and sets $u ( \tau ) = a _ { t } \mathrm { f o r } \tau \in [ 2 8 t , 2 8 ( t + 1 ) )$ . We take $T = 6 5 , \mathrm { i . e . }$ , a horizon of approximately five years, which exceeds the median overall survival in both the adaptive therapy cohort and the standardof-care cohort of [52]. At visit t, the state is given by the cell numbers $x _ { t } = ( S _ { t } , R _ { t } ) : = \big ( S ( 2 8 t ) , R ( 2 8 t ) \big )$ Since $u ( \tau )$ is constant within each month, the state at the next visit is determined by $x _ { t }$ and $a _ { t }$ through the transition

$$
x _ { t + 1 } = F _ { \alpha , \beta } ( x _ { t } , a _ { t } ) : = z ( 2 8 ) , \qquad \mathrm { w h e r e } \quad \frac { d z } { d \tau } = g _ { \alpha , \beta } \big ( z ( \tau ) , a _ { t } \big ) \mathrm { f o r } \tau \in [ 0 , 2 8 ] , \quad z ( 0 ) = x _ { t } ,\tag{2.5}
$$

and $g _ { \alpha , \beta } ( x , u )$ denotes the right-hand side of (2.4) evaluated at $( S ( \tau ) , R ( \tau ) ) = x$ and $u ( \tau ) = u ,$ which depends on α through the competition term and on $\beta$ through the growth rates. We compute $F _ { \alpha , \beta }$ by the classical fourth-order Runge-Kutta method with step size 1 on the grid $\tau \in \{ 0 , 1 , \ldots , 2 8 \}$ , and truncate S and R to [0, κ] after every step (see Appendix B).

We now formulate adaptive therapy as an instance of (2.1). For each patient, we take as context $c =$ $( S _ { 0 } , R _ { 0 } , \alpha , \beta )$ the initial numbers of sensitive and resistant cells together with the competition and growth parameters. Given the context, the initial distribution $\mu _ { c }$ is a point mass at $x _ { 0 } = ( S _ { 0 } , R _ { 0 } )$ , and the transition kernel ${ P _ { c } ( \cdot | x , a ) }$ is a point mass at $F _ { \alpha , \beta } ( x , a )$ defined in (2.5). Consequently, the context determines the course of the disease under any sequence of treatment decisions. The clinician, however, observes neither the context nor the cell numbers, and only measures PSA. Indeed, sensitive and resistant cells are assumed in [52] to produce equal amounts of PSA, in which case PSA is given by the tumor burden $B _ { t } = S _ { t } + R _ { t }$ at visit t multiplied by the amount of PSA produced per cell. Since the amount of PSA produced per cell is unknown, the pretreatment PSA does not reveal the initial cell numbers $( S _ { 0 } , R _ { 0 } )$ . We therefore take as the observation the logarithm of the ratio of PSA to its pretreatment value, which is given by $B _ { t } / B _ { 0 }$

$$
\begin{array} { r } { y _ { t } = \log \left( B _ { t } / B _ { 0 } \right) . } \end{array}\tag{2.6}
$$

Under the protocol of the pilot clinical trial [52], every patient receives abiraterone at enrollment, and treatment is reinitiated whenever PSA returns to or above its pretreatment value. We impose both requirements on every policy as a safety constraint, which we refer to as the forced-treatment rule. In terms of the observation (2.6), we require $a _ { t } = 1 { \mathrm { ~ a t ~ } } t = 0$ and at every visit with $y _ { t } \geq 0$ , and impose no restriction on $a _ { t }$ at any other visit. Accordingly, the policy class Π in (2.1) consists of the history-dependent policies satisfying this constraint.

We define progression as the first visit $t \geq 1$ at which the tumor burden reaches 1.2B , i.e., exceeds its pretreatment value by at least 20%, which is the threshold adopted in [5, 15, 45]. Since treatment can only be adjusted at visits, the tumor burden may reach $1 . 2 B _ { 0 }$ during a treatment break while the tumor is still partially sensitive to treatment. Such progression is referred to as premature progression in [15], where the threshold for pausing treatment is derived from the regrowth of the tumor between two consecutive visits to prevent it. Given that premature progression exposes the patient to a tumor burden that would be regarded as progression under treatment, we count progression irrespective of whether treatment is administered. In contrast, progression is declared only when the tumor grows under treatment in [5], which concerns intermittent androgen suppression instead of abiraterone and adopts the model of [45] (see Appendix D). We define the progression-free time $G ^ { \pi } ( c ) \in \{ 0 , 1 , \ldots , T \}$ as the number of months completed before progression. Correspondingly, we model progression as an absorbing state with zero reward, and set the reward at every visit t before progression to

$$
r _ { c } ( x _ { t } , a _ { t } ) = \mathbb { I } \big \{ B _ { t + 1 } < 1 . 2 B _ { 0 } \big \} , \qquad \mathrm { w h e r e } \quad ( S _ { t + 1 } , R _ { t + 1 } ) = F _ { \alpha , \beta } ( x _ { t } , a _ { t } ) .
$$

Hence the cumulative reward is given by $G ^ { \pi } ( c )$ , and $J ( \pi ; \mathcal { D } ) = \mathbb { E } _ { c \sim \mathcal { D } } \left[ \mathbb { E } ^ { \pi } \left[ G ^ { \pi } ( c ) \right] \right]$ . It should be noted that AT50, implemented as in Section 3.1, is a feasible solution of (2.1), whose two thresholds are fixed in advance rather than obtained by solving (2.1). By applying RL to (2.1), we instead search over the entire class Π for a policy that adapts the treatment to the whole history of PSA measurements and treatment decisions.

## 2.3 Robust Contextual MDPs

In clinical practice, a policy learned by solving the contextual MDP (2.1) is subject to uncertainty beyond the variation across patients, which we model through the population distribution D of contexts. In particular, the treatment decisions are made based on the measured PSA, which is afected by analytical and biological variation [35], and serves only as an indirect measure of the underlying tumor burden even when measured exactly [6]. The response of a patient to the treatment may also deviate from the nominal model in Section 2.2. Indeed, the key parameters of the nominal model are estimated from a small cohort of patients and are hence subject to considerable statistical error, while the patient population and the disease dynamics may shift between learning and deployment [13], or even during treatment. For instance, tumors may develop new mechanisms of resistance to abiraterone during treatment [1]. Since the learned policy tailors its decisions to the PSA trajectories generated by the nominal model, its performance may deteriorate substantially in the presence of such uncertainty. In view of this, we seek policies that are robust against both errors in the observation and shifts in the patient dynamics.

To this end, we model the errors in the observation as random, and account for shifts in the patient dynamics through the lens of distributionally robust optimization (DRO) [2, 3, 10, 17, 27, 37], in which one seeks the best worst-case performance over an ambiguity set of distributions. Following the game-theoretic view of distributionally robust MDPs [29, 32], we formulate robustness to such shifts as a Markov game between the controller and an adversary acting on the transition kernel.

## 2.3.1 Errors in the Observation

To account for errors in the measured PSA, we model the observation at visit t as

$$
y _ { t } = \log \left( B _ { t } / B _ { 0 } \right) + \delta _ { t } , \qquad | \delta _ { t } | \leq \varepsilon ,\tag{2.7}
$$

where $\delta _ { t }$ denotes the error at visit t, whose magnitude is bounded by $\varepsilon ,$ and the exact observation (2.6) corresponds to $\delta _ { t } = 0$ . Both the policy and the forced-treatment rule act on the perturbed observation, and hence an error may also delay the restart of treatment required by the forced-treatment rule, which we refer to as the forced restart. Since the measured PSA is afected by analytical and within-subject biological variation [35], we model the errors as random, and refer to them as random perturbations. Specifically, every $\delta _ { t }$ is drawn independently from the uniform distribution on $[ - \varepsilon , \varepsilon ]$ . We then learn policies that account for such errors by solving (2.1) under the observation (2.7).

## 2.3.2 Robustness to Shifts in the Patient Dynamics

To account for shifts in the patient dynamics, we allow the transition kernel of each patient to change in every month within an ambiguity set $\mathcal { P } _ { c }$ of transition kernels. Specifically, an adversary policy σ selects a kernel from $\mathcal { P } _ { c }$ at visit $t ,$ possibly at random, based on the observations up to visit $t ,$ and we define ${ J } ( \pi , \sigma ; \mathcal { D } )$ as the expected cumulative reward of $\pi$ under the kernels selected by $\sigma .$ We consider the robust counterpart of (2.1) given by the zero-sum game

$$
\operatorname* { m a x } _ { \pi \in \Pi } \operatorname* { m i n } _ { \sigma } J ( \pi , \sigma ; { \mathcal D } ) .\tag{2.8}
$$

Since the kernel is selected anew at every visit based on the realized history, (2.8) follows the dynamic formulation of robust MDPs [29, 30], and hence accounts for shifts in the dynamics that occur during treatment.

To construct $\mathcal { P } _ { c }$ in adaptive therapy, we consider shifts in the competition and growth parameters α and $\beta ,$ whose estimates are subject to statistical error and may difer across patients (Section 2.2). Following Remark 1 of [29], we draw K pairs of multipliers $m ^ { ( k ) } = ( m _ { \alpha } ^ { ( k ) } , m _ { \beta } ^ { ( k ) } )$ of $( \alpha , \beta ) , k = 1 , \ldots , K$ , once from a nominal distribution $Q _ { \mathrm { n o m } }$ concentrated around one (see Appendix B), and keep them fixed. For a patient with context $c ,$ the k-th generating kernel $F _ { c } ^ { ( k ) }$ is given by the transition (2.5) with parameters $( m _ { \alpha } ^ { ( k ) } \alpha , m _ { \beta } ^ { ( k ) } \beta )$ and $\mathcal { P } _ { c }$ consists of the mixtures of the $K$ generating kernels whose weights lie in

$$
\begin{array} { r } { \mathcal { W } _ { \rho } = \left\{ ( 1 - \rho ) \frac { 1 } { K } { \bf 1 } + \rho q : ~ q \in \Delta _ { K } \right\} , \qquad \rho \in [ 0 , 1 ] , } \end{array}\tag{2.9}
$$

where $\Delta _ { K }$ denotes the probability simplex. The radius $\rho$ can be interpreted as the expected fraction of months in which the patient dynamics are selected by the adversary. Indeed, since the expected cumulative reward is linear in the mixture weights at every visit, it sufices for the adversary to place the weight $\rho$ on a single generating kernel at every visit. Equivalently, at every visit the kernel is drawn uniformly from the $K$ generating kernels with probability $1 - \rho ,$ and is selected by the adversary with probability $\rho .$

Clearly, for $\rho \ = \ 0 ,$ , problem (2.8) reduces to (2.1) with the parameters redrawn uniformly from the generating kernels in every month, while for $\rho = 1$ the adversary selects the kernel at every visit. In Section 3.4, we refer to the radius used in training as the training radius, denoted by $\rho _ { \mathrm { t r } }$ , and evaluate every policy at a range of radii $\rho ,$ , which we refer to as evaluation radii.

## 3 Numerical Experiments

In this section, we demonstrate that the policies learned from the formulations in Section 2 possess the three properties identified in Section 1. After describing the experimental setup in Section 3.1, we show in Section 3.2 that a policy learned from the contextual MDP (2.1) substantially improves on AT50 without administering more drug. Such a policy, however, acts on PSA as if it were exact. In Section 3.3, we find that errors in PSA of realistic magnitude substantially reduce the benefit of this policy, while by training under random perturbations we retain the benefit of adaptive therapy under such errors. We then turn to shifts in the patient dynamics, and show in Section 3.4 that by solving the kernel-robust formulation (2.8), we attain robustness to such shifts at almost no price in the nominal model, and in Section 3.5 that the resulting policy nearly Pareto-dominates the policy learned from (2.1) and AT50 across all the sources of uncertainty we consider. Finally, in Section 3.6, we attain on the 32 fitted patients a performance comparable to that obtained with a population calibrated to them, by training on a broad population constructed without reference to these patients.

## 3.1 Experimental Setup

We train all policies on a calibrated population $\mathcal { D } _ { \mathrm { c a l } }$ of virtual patients, obtained by multiplying the initial cell numbers of the 32 fitted patients of [52] by independent lognormal factors, with the competition and growth parameters kept at their nominal values $( \alpha , \beta ) = ( 6 , 8 )$ . For evaluation, we consider two test populations, namely the virtual test population of 1,000 patients drawn from $\mathcal { D } _ { \mathrm { c a l } }$ independently of the training patients, and the 32 fitted patients, whose empirical distribution we denote by ${ \mathcal { D } } _ { \mathrm { f i t } }$ . All methods are evaluated on the same test patients with common random numbers. The construction of $\mathcal { D } _ { \mathrm { c a l } }$ and the evaluation protocol are given in Appendix B.

All learned policies are parameterized by a gated recurrent unit (GRU) network [8] acting on features of the history $h _ { t } .$ and trained by proximal policy optimization (PPO) [42] for 2,000 updates, each with 512 patients newly sampled from the training population. For each type of policy, we train three policies with diferent random seeds, and the remaining details are given in Appendix B.

We compare the learned policies with two clinical rules, namely continuous treatment, i.e., $a _ { t } = 1$ for all $t ,$ and AT50. In our model, we implement AT50 as the policy that pauses treatment once $y _ { t } < \log 0 . 5 $ resumes it once $y _ { t } \geq 0$ , and otherwise repeats its previous action. Both satisfy the forced-treatment rule.

Since we impose no explicit penalty in (2.1) on patients who fare worse than under the standard of care, we monitor such patients separately. To this end, we compare the progression-free time $G ^ { \pi } ( c )$ of each patient with the progression-free time $G ^ { \mathrm { o n } } ( c )$ of the same patient under continuous treatment. We say that π harms the patient if $G ^ { \pi } ( c ) < G ^ { \mathrm { o n } } ( c ) - 3 .$ , and that it benefits the patient if $G ^ { \pi } ( c ) > G ^ { \mathrm { o n } } ( c ) + 3$ , where the margin of three months corresponds to the interval of 12 weeks between two consecutive imaging assessments in the pilot clinical trial [52]. We refer to the fractions of patients harmed and benefited as the harm rate and the benefit rate, and report them together with the mean progression-free time G<sup>¯</sup>. For the learned policies, all values are averaged over the three seeds, with per-seed values in Appendix C.

## 3.2 Substantial Improvement over AT50

We begin with the contextual MDP (2.1) under the exact observation (2.6), and refer to the policy learned from (2.1) as nominal RL. Table 1 compares nominal RL with the two clinical rules on both test populations.

Table 1: Nominal RL and the clinical rules in the nominal model, in terms of the mean progression-free time G<sup>¯</sup> (months) and its gain over continuous treatment, the harm and benefit rates, the fraction of patients without progression after $T = 6 5$ months (PF at T), the fraction of treated months, and the mean number of treatment courses, i.e., maximal runs of consecutive treated months.
<table><tr><td>Method</td><td>G</td><td>Gain</td><td>Harm (%)</td><td>Benefit (%)</td><td>PF at T (%)</td><td>Treated (%)</td><td>Courses</td></tr><tr><td colspan="8">Virtual test population (1,000 patients)</td></tr><tr><td>Continuous treatment</td><td>15.81</td><td></td><td></td><td></td><td>2.7</td><td>100.0</td><td>1.0</td></tr><tr><td>AT50</td><td>29.58</td><td>+13.77</td><td>28.4</td><td>54.0</td><td>32.5</td><td>28.7</td><td>6.2</td></tr><tr><td>Nominal RL</td><td>50.35</td><td>+34.54</td><td>0.2</td><td>91.9</td><td>61.5</td><td>30.3</td><td>12.9</td></tr><tr><td colspan="8">Fitted patients (32 patients)</td></tr><tr><td>Continuous treatment 14.66</td><td></td><td></td><td></td><td></td><td>0.0</td><td>100.0</td><td>1.0</td></tr><tr><td>AT50</td><td> $2 5 . 3 4 \quad + 1 0 . 6 9 \quad$ </td><td></td><td>28.1</td><td>53.1</td><td>21.9</td><td>29.1</td><td>5.6</td></tr><tr><td>Nominal RL</td><td> $\begin{array} { r l } { 4 8 . 4 3 } & { { } + 3 3 . 7 7 } \end{array}$ </td><td></td><td>0.0</td><td>94.8</td><td>53.1</td><td>29.8</td><td>12.7</td></tr></table>

Nominal RL substantially outperforms AT50. Specifically, it attains a mean progression-free time of 50.35 months on the virtual test population, compared with 29.58 months for AT50 and 15.81 months for continuous treatment. Its gain over continuous treatment is 34.54 months, more than twice the gain of 13.77 months attained by AT50. In addition, nominal RL harms 0.2% of the patients and benefits 91.9%,

![](images/fe62a184976751c18af145e9ab44898527b81712c1b7a0a7dd21c3c15c1a24d5.jpg)  
Figure 1: Nominal RL, AT50, and continuous treatment on the virtual test population, split into tertiles of the initial resistant fraction $R _ { 0 } / B _ { 0 }$ . (a) Mean progression-free time. (b) Harm rate.

![](images/5100650430a7616dcc3f40e52932488d80e9739b4d5e6b084e6e88e145a3fe51.jpg)

$$
R _ { 0 } / B _ { 0 }
$$

Continuous treatment

compared with 28.4% and 54.0% for AT50. On the 32 fitted patients, nominal RL attains 48.43 months, compared with 25.34 months for AT50, and harms none of the fitted patients.

A few remarks are in order. First, the improvement is attained without administering more drug. Nominal RL treats in 30.3% of the months and AT50 in 28.7%, while nominal RL divides its treated months into 12.9 courses on average, compared with 6.2 for AT50. Hence nominal RL alternates between treatment and breaks roughly twice as often as AT50, with correspondingly shorter courses and breaks. Second, the improvement holds across patients with diferent initial resistance (Figure 1), a factor that correlates with the time to radiographic progression in both the adaptive therapy cohort and the standard-of-care cohort of [52]. When we split the virtual test population into tertiles of the initial resistant fraction $R _ { 0 } / B _ { 0 } .$ , nominal RL attains 60.81 months in the lowest tertile and 35.57 months in the highest, compared with 37.28 and 19.93 months for AT50 (Table 6). AT50 mainly harms patients with few resistant cells, with a harm rate of 39.9% in the lowest tertile compared with 10.8% in the highest, while the harm rate of nominal RL is at most 0.4% in every tertile. Third, the definition of progression in Section 2.2 is essential for an informative comparison. If we declare progression only when the tumor grows under treatment as in [5], the growth of the tumor during a break is never penalized, and the rule that withholds treatment whenever permitted attains 53.79 months, the same value as all three seeds of nominal RL trained under this definition, compared with 52.60 months for AT50 (Appendix D). Consequently, such a definition does not allow us to assess the benefit of adapting the treatment to PSA.

In summary, by learning from data we obtain a policy that substantially improves on AT50 for every level of initial resistance without administering more drug. In addition, since the virtual test patients are drawn independently of the training patients, nominal RL generalizes to new patients from the training population, in line with Theorem 2.1.

## 3.3 Retaining the Benefit of Adaptive Therapy under Errors in PSA

Nominal RL acts on PSA as if it were exact. In clinical practice, however, PSA is subject to measurement errors. We therefore evaluate nominal RL under the random perturbations of Section 2.3.1 with $\varepsilon = 0 . 2$ which is comparable to the reference change value of about 19% for total PSA, i.e., the smallest relative change between two consecutive measurements that can be distinguished from analytical and within-subject biological variation [35]. We also refer to the policy learned from (2.1) under the same perturbations as random-perturbation RL, and compare it with nominal RL and the two clinical rules in Figure 2.

We observe that errors in PSA with ε = 0.2 substantially reduce the benefit of the policies that do not account for them. Under random perturbations, nominal RL drops from 50.35 to 32.96 months, and its harm rate increases from 0.2% to 14.4%. Similarly, AT50 drops from 29.58 to 16.46 months, only marginally above the 15.81 months of continuous treatment, and its harm rate increases from 28.4% to 43.8%. In contrast, random-perturbation RL retains the benefit of adaptive therapy under such errors. Under random perturbations, it attains 44.49 months with a harm rate of 2.0%, close to the 45.94 months it attains without

![](images/a5a98b2b12734193df68416b7f098a241489bdf2c372174fbed6786e3212bb47.jpg)  
Figure 2: Policies on the virtual test population in the nominal model and under random perturbations of PSA with $\varepsilon = 0 . 2 .$ (a) Mean progression-free time. (b) Harm rate. See Table 4 of Appendix C for the values.

perturbation.

A few remarks are in order. First, under perturbations with $\varepsilon = 0 . 2 .$ , we can no longer ensure through the forced-treatment rule that treatment is restarted before progression. Indeed, since $\varepsilon >$ log 1.2, an error that lowers the measured PSA can postpone the forced restart until after progression. Since AT50 resumes treatment only through the condition $y _ { t } \geq 0$ , such an error can postpone its restart in every treatment break. Second, we pay a price in the nominal model for accounting for errors in $\mathrm { P S A }$ , with 45.94 months for random-perturbation RL compared with 50.35 months for nominal RL.

In summary, policies that do not account for errors in PSA lose a substantial part of their benefit under errors of realistic magnitude, and by training under random perturbations we retain the benefit of adaptive therapy under such errors.

## 3.4 Robustness to Shifts in the Patient Dynamics at Almost No Price

We next turn to shifts in the patient dynamics, which arise when the competition and growth parameters deviate from their estimated values. We model such shifts by the ambiguity set $\mathcal { P } _ { c }$ of Section 2.3.2 with $K = 3 2$ generating kernels drawn from $Q _ { \mathrm { n o m } } .$ , and refer to the policy learned from (2.8) with training radius $\rho _ { \mathrm { t r } } \in \{ 0 . 1 , 0 . 2 5 , 0 . 5 \}$ as kernel-robust RL.

We evaluate all policies at the evaluation radii $\rho \in \{ 0 , 0 . 1 , 0 . 2 5 , 0 . 5 , 1 \}$ . At $\rho = 0$ , the kernel is drawn uniformly from the generating kernels in each month, and for $\rho > 0 ;$ , each policy is evaluated against a best-response adversary, i.e., an adversary trained by PPO to minimize the reward of the policy while the policy is held fixed. In addition to the 32 generating kernels used in training, referred to as the training kernels, we evaluate the policies on 32 held-out kernels drawn independently from $Q _ { \mathrm { n o m } }$ . Figure 3 shows the mean progression-free time and the harm rate in the nominal model and as functions of the evaluation radius, with the complete values in Table 2.

Kernel-robust RL improves on nominal RL and AT50 at every evaluation radius on both sets of kernels. In the nominal model, it forgoes between 1.58 and 2.22 months compared with nominal RL, which is small compared with its improvement of between 7.33 and 7.71 months over nominal RL at $\rho = 0$ on the training kernels. On the training kernels, kernel-robust RL evaluated at its training radius $\rho ~ = ~ \rho _ { \mathrm { t r } }$ <sub>r</sub> improves on nominal RL by between 7.00 and 8.74 months, and in particular attains 46.57 months at $\rho = 0 . 2 5$ , compared with 39.12 months for nominal RL. At $\rho = 1$ , which exceeds every training radius, kernel-robust RL with $\rho _ { \mathrm { t r } } = 0 . 5$ attains 25.08 months, compared with 17.46, 11.65, and 12.72 months for nominal RL, AT50, and continuous treatment. In addition, the harm rate of kernel-robust RL is at most 1.7% at every radius on both sets of kernels, while on the training kernels, the harm rate of nominal RL grows from 8.9% at $\rho = 0$ to 29.1% at $\rho = 1$ , and that of AT50 from 44.4% to 65.1%.

A few remarks are in order. First, nominal RL loses much of its advantage once the parameters vary, even in the absence of an adversary. Since the multipliers $m _ { \alpha }$ of the generating kernels have mean larger than one, the resistant cells are on average suppressed more strongly at $\rho = 0$ than in the nominal model.

![](images/54b93bfee6727dabeed5932c9024fc227456ac52330a2f870fdb0165f569f0eb.jpg)  
Figure 3: Mean progression-free time (top) and harm rate (bottom) on the virtual test population. The left panels show the nominal model, and the middle and right panels show the performance as a function of the evaluation radius $\rho , \mathbf { \tau } ( \mathrm { a } )$ on the training kernels and (b) on the held-out kernels. Kernel-robust RL is shown for $\rho _ { \mathrm { t r } } = 0 . 2 5$

Indeed, at $\rho = 0 ,$ , continuous treatment improves from 15.81 to 16.11 months, and kernel-robust RL improves from between 48.13 and 48.77 months to between 53.07 and 53.45 months. Nominal RL, however, drops from 50.35 to 45.74 months. Hence a policy trained for a single pair $( \alpha , \beta )$ is tuned to the pair used in training, and we benefit from exposing the policy to varying dynamics during training. Second, the best training radius increases with the evaluation radius. The policy with $\rho _ { \mathrm { t r } } = 0 . 1$ is best at $\rho \in \{ 0 , 0 . 1 \}$ , while the policy with $\rho _ { \mathrm { t r } } = 0 . 5$ is best at $\rho \in \{ 0 . 5 , 1 \}$ on both sets of kernels. Hence the training radius can be chosen according to the magnitude of the shifts expected in deployment. Third, AT50 falls below continuous treatment at $\rho = 1$ on both sets of kernels. Indeed, at $\rho = 1$ the adversary can select the generating kernel with the weakest competition in every month, and adaptive therapy fails, or only delays failure, when the competitive efect of the sensitive cells on the resistant cells is small [52].

Table 2: Mean progression-free time (months) under shifts in the patient dynamics on the virtual test population for policies trained on the calibrated population, with harm rates (%) in parentheses.
<table><tr><td colspan="2"></td><td colspan="3">Kernel-robust RL</td><td colspan="2"></td></tr><tr><td>Condition</td><td>Nominal RL</td><td> $\rho _ { \mathrm { t r } } = 0 . 1$ </td><td> $\rho _ { \mathrm { t r } } = 0 . 2 5$ </td><td> $\rho _ { \mathrm { t r } } = 0 . 5$ </td><td>AT50</td><td>Continuous treatment</td></tr><tr><td>Nominal model</td><td>50.35 (0.2)</td><td>48.13 (0.1)</td><td>48.77 (0.0)</td><td>48.45 (0.0)</td><td>29.58 (28.4)</td><td>15.81</td></tr><tr><td>Training kernels</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\rho = 0$ </td><td>45.74 (8.9)</td><td>53.45 (0.9)</td><td>53.31 (1.0)</td><td>53.07 (0.9)</td><td>19.04 (44.4)</td><td>16.11</td></tr><tr><td> $\rho = 0 . 1$ </td><td>44.20 (8.9)</td><td>51.20 (0.9)</td><td>51.06 (1.2)</td><td>50.88 (1.0)</td><td>19.13 (43.8)</td><td>16.04</td></tr><tr><td> $\rho = 0 . 2 5$ </td><td>39.12 (12.2)</td><td>46.49 (0.9)</td><td>46.57 (1.3)</td><td>46.50 (1.1)</td><td>17.92 (45.9)</td><td>15.68</td></tr><tr><td> $\rho = 0 . 5$ </td><td>30.98 (18.2)</td><td>37.88 (0.8)</td><td>38.44 (1.2)</td><td>39.72 (0.8)</td><td>15.51 (51.1)</td><td>14.56</td></tr><tr><td> $\rho = 1$ </td><td>17.46 (29.1)</td><td>23.38 (0.1)</td><td>23.26 (0.6)</td><td>25.08 (0.4)</td><td>11.65 (65.1)</td><td>12.72</td></tr><tr><td>Held-out kernels</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\rho = 0$ </td><td>47.26 (8.6)</td><td>54.80 (1.0)</td><td>54.21 (1.7)</td><td>53.66 (1.7)</td><td>20.23 (43.6)</td><td>16.81</td></tr><tr><td> $\rho = 0 . 1$ </td><td>45.54 (9.4)</td><td>52.80 (0.9)</td><td>52.62 (1.2)</td><td>52.36 (1.1)</td><td>19.90 (44.4)</td><td>16.71</td></tr><tr><td> $\rho = 0 . 2 5$ </td><td>39.30 (13.6)</td><td>46.83 (0.6)</td><td>46.87 (0.8)</td><td>47.10 (0.7)</td><td>18.80 (46.1)</td><td>16.32</td></tr><tr><td> $\rho = 0 . 5$ </td><td>29.60 (19.9)</td><td>36.67 (0.4)</td><td>36.55 (0.5)</td><td>37.85 (0.4)</td><td>16.96 (49.8)</td><td>15.27</td></tr><tr><td> $\rho = 1$ </td><td>17.44 (20.4)</td><td>19.48 (0.1)</td><td>19.36 (0.1)</td><td>20.13 (0.1)</td><td>12.11 (63.1)</td><td>13.31</td></tr></table>

![](images/cf8d9309243dfac15ca202b632d018d7a8d1ef479d1fd9163af250eced5453ad.jpg)  
Figure 4: Cross-evaluation on the virtual test population. Each axis shows the mean progression-free time divided by the highest value attained in the corresponding condition by the policies in this figure. Kernel shift denotes the evaluation radius ρ on the training kernels, and random perturbation denotes errors in PSA with $\varepsilon = 0 . 2$ . Kernel-robust RL is trained with $\rho _ { \mathrm { t r } } = 0 . 2 5$ . See Table 5 of Appendix C for the values.

In summary, by training against shifts in the patient dynamics, we retain the benefit of adaptive therapy under such shifts, including those generated by held-out kernels, while sacrificing almost no performance when the dynamics evolve as anticipated.

## 3.5 Universal Benefits of Kernel-Robust RL

So far, we have evaluated random-perturbation RL and kernel-robust RL only under the source of uncertainty that each policy is trained against. In deployment, however, the source of uncertainty is not known in advance, and we would like a single policy that performs well under all of them. We therefore cross-evaluate the two policies, i.e., we evaluate random-perturbation RL under shifts in the dynamics and kernel-robust RL with training radius $\rho _ { \mathrm { t r } } = 0 . 2 5$ under random perturbations of PSA, and compare both policies with nominal RL and the two clinical rules in Figure 4.

We find that kernel-robust RL nearly Pareto-dominates nominal RL and AT50 across the five conditions of Figure 4, namely the nominal model, the evaluation radii $\rho \in \{ 0 , 0 . 2 5 , 1 \}$ on the training kernels, and random perturbations of PSA. Indeed, the polygon of kernel-robust RL encloses the polygon of AT50, and lies outside the polygon of nominal RL on every axis except the nominal model. In particular, kernel-robust RL forgoes 1.58 months compared with nominal RL in the nominal model, and attains the highest value under every shift in the dynamics, with 53.31, 46.57, and 23.26 months at $\rho \in \{ 0 , 0 . 2 5 , 1 \}$ , compared with 45.74, 39.12, and 17.46 months for nominal RL. Under random perturbations of PSA, which it never encounters during training, kernel-robust RL still improves on nominal RL, with 38.65 compared with 32.96 months. In addition, AT50 falls below kernel-robust RL, and indeed below all learned policies, in every condition.

On the other hand, by comparing kernel-robust RL with random-perturbation RL in Figure 4, we observe that each of the two policies attains the highest value under the source of uncertainty it is trained against. Specifically, random-perturbation RL attains 44.49 months under random perturbations of PSA, compared with 38.65 months for kernel-robust RL, while kernel-robust RL attains higher values than randomperturbation RL under every shift in the dynamics.

In summary, kernel-robust RL nearly Pareto-dominates nominal RL and AT50 across the sources of uncertainty, in the sense that it nearly matches nominal RL in the nominal model, outperforms every other policy under shifts in the dynamics, and substantially improves on nominal RL and AT50 under errors in PSA. In view of this, we believe that kernel-robust training is a natural default for learning adaptive therapy policies, to be complemented by training under random perturbations when errors in PSA are of concern.

![](images/d51a156f571b273318074548ca8da52131e1cba1e87a1d21b3f537cf476dc07f.jpg)  
Figure 5: Mean progression-free time on the 32 fitted patients in the nominal model for policies trained on the calibrated population and on the broad population.

## 3.6 Zero-Shot Generalization from a Broad Synthetic Population

By Theorem 2.1, a policy trained on suficiently many sampled patients generalizes to new patients drawn from the training population, provided that this population is broad enough to contain the patients of interest. So far, however, we have trained all policies on the calibrated population $\mathcal { D } _ { \mathrm { c a l } } .$ , which is constructed from the 32 fitted patients. We now train each type of policy on a broad population $\mathcal { D } _ { \mathrm { w i d e } }$ constructed without reference to the fitted patients, with $\alpha \in [ 2 , 8 ] , \beta \in [ 6 , 1 0 ]$ , and broad ranges of the initial tumor burden and the initial resistant fraction (see Appendix B). We then apply the resulting policies to the fitted patients without further training, which we refer to as zero-shot generalization, and evaluate them under shifts in the dynamics against best-response adversaries trained on the fitted patients. Figure 5 compares the two training populations, and Table 3 reports the policies trained on $\mathcal { D } _ { \mathrm { w i d e } }$ under errors in PSA and shifts in the dynamics.

Table 3: Mean progression-free time (months) on the 32 fitted patients for policies trained on $\mathcal { D } _ { \mathrm { c a l } }$ and on $\mathcal { D } _ { \mathrm { w i d e } }$ under the conditions of Table 5.
<table><tr><td rowspan="2">Method</td><td>Trained on  $\mathcal { D } _ { \mathrm { c a l } }$ </td><td colspan="5">Trained on  $\mathcal { D } _ { \mathrm { w i d e } }$ </td></tr><tr><td>Nominal model</td><td>Nominal model</td><td> $\rho = 0$ </td><td> $\rho = 0 . 2 5$ </td><td> $\rho = 1$ </td><td> $\varepsilon = 0 . 2$ </td></tr><tr><td>Nominal RL</td><td>48.43</td><td>48.59</td><td>46.14</td><td>33.99</td><td>13.64</td><td>26.85</td></tr><tr><td>Random-perturbation RL</td><td>45.17</td><td>43.54</td><td>46.64</td><td>38.37</td><td>22.04</td><td>42.08</td></tr><tr><td>Kernel-robust  $\mathrm { R L } , \rho _ { \mathrm { t r } } = 0 . 1$ </td><td>46.20</td><td>45.58</td><td>53.34</td><td>44.32</td><td>20.03</td><td>32.77</td></tr><tr><td>Kernel-robust RL,  $\rho _ { \mathrm { t r } } = 0 . 2 5$ </td><td>46.46</td><td>46.03</td><td>52.84</td><td>43.31</td><td>20.79</td><td>31.90</td></tr><tr><td>Kernel-robust RL,  $\rho _ { \mathrm { t r } } = 0 . 5$ </td><td>46.11</td><td>45.77</td><td>53.31</td><td>44.59</td><td>21.18</td><td>32.94</td></tr><tr><td>AT50</td><td>25.34</td><td>25.34</td><td>17.82</td><td>16.91</td><td>10.07</td><td>16.22</td></tr><tr><td>Continuous treatment</td><td>14.66</td><td>14.66</td><td>14.97</td><td>14.37</td><td>10.90</td><td>14.66</td></tr></table>

We observe almost no loss on the fitted patients from training on the broad population. Nominal RL trained on the broad population attains 48.59 months, compared with 48.43 months when trained on the calibrated population, and the two training populations difer by at most 1.63 months for every type of policy. In addition, we retain on the fitted patients the benefits of training under random perturbations and of kernel-robust training. Under random perturbations of PSA, random-perturbation RL attains 42.08 months, compared with 26.85 months for nominal RL. Under shifts in the dynamics on the training kernels, kernel-robust RL attains between 43.31 and 44.59 months at $\rho = 0 . 2 5$ and between 20.03 and 21.18 months at $\rho = 1$ , compared with 33.99 and 13.64 months for nominal RL.

Kernel-robust RL also harms substantially fewer fitted patients than nominal RL and AT50 under the strongest shifts in the dynamics. Specifically, at $\rho = 1$ on the training kernels, it harms between 8.4% and 16.7% of the fitted patients, compared with 47.3% and 79.3% for nominal RL and AT50, and on the held-out kernels it harms none of them, compared with 13.3% and 63.6% (Table 8).

In summary, by training on a broad synthetic population constructed without the fitted patients, we attain on these patients a performance comparable to that obtained with a population calibrated to them, in line with Theorem 2.1, and retain the benefits of both random-perturbation RL and kernel-robust RL.

## 4 Concluding Remarks

In this manuscript, we study adaptive therapy through the lens of robust contextual MDPs, in which the patient is modeled as a hidden context drawn from a population. To account for the uncertainty beyond the variation across patients, we further train the policy under random perturbations of the measured biomarker and against an adversary acting on the transition kernel. Within a model of abiraterone therapy fitted to clinical data, we find that (1) a policy learned within this formulation substantially outperforms AT50 and generalizes to the fitted patients from a broad population constructed without them, (2) errors in PSA of realistic magnitude substantially reduce the benefit of policies that do not account for them, while by training under such errors we retain the benefit of adaptive therapy, and (3) we attain robustness to shifts in the patient dynamics at almost no price in the nominal model, with the resulting policy nearly Paretodominating nominal RL and AT50 across all the errors and shifts we consider. In view of this, we believe that robust RL ofers a principled route toward adaptive treatment policies that remain reliable beyond the model in which they are learned.

Use of Large Language Models (LLMs). We acknowledge the assistance of LLMs in prototyping the numerical experiments and improving the presentation of this manuscript.

## References

[1] Emmanuel S. Antonarakis, Changxue Lu, Hao Wang, Brandon Luber, Mary Nakazawa, Jefrey C. Roeser, Yan Chen, Tabrez A. Mohammad, Yidong Chen, Helen L. Fedor, Tamara L. Lotan, Qizhi Zheng, Angelo M. De Marzo, John T. Isaacs, William B. Isaacs, Rosa Nadal, Channing J. Paller, Samuel R. Denmeade, Michael A. Carducci, Mario A. Eisenberger, and Jun Luo. AR-V7 and resistance to enzalutamide and abiraterone in prostate cancer. New England Journal of Medicine, 371(11):1028– 1038, 2014.

[2] A. Ben-Tal, D. den Hertog, A. M. B. De Waegenaere, B. Melenberg, and G. Rennen. Robust solutions of optimization problems afected by uncertain probabilities. Management Science, 59(2):341–357, 2013.

[3] Jose Blanchet and Karthyek Murthy. Quantifying distributional model risk via optimal transport. Mathematics of Operations Research, 44(2):565–600, 2019.

[4] S. A. Brown, B. P. Kovatchev, D. Raghinaru, J. W. Lum, B. A. Buckingham, Y. C. Kudva, L. M. Lafel, C. J. Levy, J. E. Pinsker, R. P. Wadwa, E. Dassau, F. J. Doyle, III, S. M. Anderson, M. M. Church, V. Dadlani, L. Ekhlaspour, G. P. Forlenza, E. Isganaitis, D. W. Lam, C. Kollman, R. W. Beck, and iDCL Trial Research Group. Six-month randomized, multicenter trial of closed-loop control in type 1 diabetes. New England Journal of Medicine, 381(18):1707–1717, 2019.

[5] Alexander P. Browning, Rebecca M. Crossley, Ryan J. Murphy, Helen Byrne, and Sara Hamis. Adaptive therapy under parametric, structural, and measurement uncertainty. arXiv:2608.18387, 2026.

[6] A. H. Bryce, J. J. Alumkal, A. Armstrong, C. S. Higano, P. Iversen, C. N. Sternberg, D. Rathkopf, Y. Loriot, J. de Bono, B. Tombal, S. Abhyankar, P. Lin, A. Krivoshik, D. Phung, and T. M. Beer. Radiographic progression with nonrising PSA in metastatic castration-resistant prostate cancer: post hoc analysis of PREVAIL. Prostate Cancer and Prostatic Diseases, 20(2):221–227, 2017.

[7] Bibhas Chakraborty and Susan A. Murphy. Dynamic treatment regimes. Annual Review of Statistics and Its Application, 1(1):447–464, 2014.

[8] Kyunghyun Cho, Bart van Merri¨enboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using RNN encoder–decoder for statistical machine translation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1724–1734, Doha, Qatar, 2014. Association for Computational Linguistics.

[9] Jessica J. Cunningham, Joel S. Brown, Robert A. Gatenby, and Kateˇrina Staˇnkov´a. Optimal control to develop therapeutic strategies for metastatic castrate resistant prostate cancer. Journal of Theoretical Biology, 459:67–78, 2018.

[10] Erick Delage and Yinyu Ye. Distributionally robust optimization under moment uncertainty with application to data-driven problems. Operations Research, 58(3):595–612, 2010.

[11] Finale Doshi-Velez and George Konidaris. Hidden parameter Markov decision processes: A semiparametric regression approach for discovering latent task parametrizations. In Proceedings of the Twenty-Fifth International Joint Conference on Artificial Intelligence (IJCAI-16), pages 1432–1440, 2016.

[12] Dalit Engelhardt. Dynamic control of stochastic evolution: A deep reinforcement learning approach to adaptively targeting emergent drug resistance. Journal of Machine Learning Research, 21(203):1–30, 2020.

[13] Samuel G. Finlayson, Adarsh Subbaswamy, Karandeep Singh, John Bowers, Annabel Kupke, Jonathan Zittrain, Isaac S. Kohane, and Suchi Saria. The clinician and dataset shift in artificial intelligence. New England Journal of Medicine, 385(3):283–286, 2021.

[14] Nicolas Fournier and Arnaud Guillin. On the rate of convergence in Wasserstein distance of the empirical measure. Probability Theory and Related Fields, 162(3–4):707–738, 2015.

[15] Kit Gallagher, Maximilian A. R. Strobl, Alexander R. A. Anderson, and Philip K. Maini. Deriving optimal treatment timing for adaptive therapy: Matching the model to the tumor dynamics. Bulletin of Mathematical Biology, 87(10):146, 2025.

[16] Kit Gallagher, Maximilian A. R. Strobl, Derek S. Park, Fabian C. Spoendlin, Robert A. Gatenby, Philip K. Maini, and Alexander R. A. Anderson. Mathematical model-driven deep learning enables personalized adaptive therapy. Cancer Research, 84(11):1929–1941, 2024.

[17] Rui Gao and Anton Kleywegt. Distributionally robust stochastic optimization with Wasserstein distance. Mathematics of Operations Research, 48(2):603–655, 2023.

[18] Robert A. Gatenby, Ariosto S. Silva, Robert J. Gillies, and B. Roy Frieden. Adaptive therapy. Cancer Research, 69(11):4894–4903, 2009.

[19] Assaf Hallak, Dotan Di Castro, and Shie Mannor. Contextual Markov decision processes. arXiv preprint arXiv:1502.02259, 2015.

[20] Wassily Hoefding. Probability inequalities for sums of bounded random variables. Journal of the American Statistical Association, 58(301):13–30, 1963.

[21] Sandy Huang, Nicolas Papernot, Ian Goodfellow, Yan Duan, and Pieter Abbeel. Adversarial attacks on neural network policies. arXiv preprint arXiv:1702.02284, 2017.

[22] Garud N. Iyengar. Robust dynamic programming. Mathematics of Operations Research, 30(2):257–280, 2005.

[23] Leslie Pack Kaelbling, Michael L. Littman, and Anthony R. Cassandra. Planning and acting in partially observable stochastic domains. Artificial Intelligence, 101(1–2):99–134, 1998.

[24] Robert Kirk, Amy Zhang, Edward Grefenstette, and Tim Rockt¨aschel. A survey of zero-shot generalisation in deep reinforcement learning. Journal of Artificial Intelligence Research, 76:201–264, 2023.

[25] Matthieu Komorowski, Leo A. Celi, Omar Badawi, Anthony C. Gordon, and A. Aldo Faisal. The artificial intelligence clinician learns optimal treatment strategies for sepsis in intensive care. Nature Medicine, 24(11):1716–1720, 2018.

[26] Michael R. Kosorok and Eric B. Laber. Precision medicine. Annual Review of Statistics and Its Application, 6(1):263–286, 2019.

[27] Daniel Kuhn, Soroosh Shafiee, and Wolfram Wiesemann. Distributionally robust optimization. Acta Numerica, 34:579–804, 2025.

[28] Jeongyeol Kwon, Yonathan Efroni, Constantine Caramanis, and Shie Mannor. RL for latent MDPs: Regret guarantees and a lower bound. In Advances in Neural Information Processing Systems, volume 34, pages 24523–24534. Curran Associates, Inc., 2021.

[29] Mengmeng Li, Yifan Hu, Daniel Kuhn, and Yan Li. Robust Markov decision processes on continuous state spaces. arXiv preprint arXiv:2605.28706, 2026.

[30] Yan Li. Computational hardness of static distributionally robust Markov decision processes. Operations Research Letters, 67:107465, 2026.

[31] Yan Li, Guanghui Lan, and Tuo Zhao. First-order policy optimization for robust Markov decision process. arXiv preprint arXiv:2209.10579, 2022.

[32] Yan Li and Alexander Shapiro. Rectangularity and duality of distributionally robust Markov decision processes. Mathematical Programming, 2025. Published online 3 November 2025.

[33] Yitao Lu, Qian Chu, Zhen Li, Mengdi Wang, Robert Gatenby, and Qingpeng Zhang. Deep reinforcement learning identifies personalized intermittent androgen deprivation therapy for prostate cancer. Briefings in Bioinformatics, 25(2):bbae071, 2024.

[34] Kexin Ma, Ningjing Wang, Zai Yang, Robert A. Cheke, and Biao Tang. Single-threshold–guided adaptive cancer therapy with partial-cycle treatment: A mechanistic and reinforcement learning analysis. PLOS Computational Biology, 22(6):e1014457, 2026.

[35] Fernando Marques-Garcia, Beatriz Boned, Elisabet Gonz´alez-Lao, Federica Braga, Anna Carobene, Abdurrahman Coskun, Jorge D´ıaz-Garz´on, Pilar Fern´andez-Calle, Maria Carmen Perich, Margarida Simon, Niels Jonker, Berna Aslan, William Alexander Bartlett, Sverre Sandberg, and Aasne K. Aarsand. Critical review and meta-analysis of biological variation estimates for tumor markers. Clinical Chemistry and Laboratory Medicine, 60(4):494–504, 2022.

[36] Colin McDiarmid. On the method of bounded diferences. In J. Siemons, editor, Surveys in Combinatorics, 1989, number 141 in London Mathematical Society Lecture Note Series, pages 148–188. Cambridge University Press, 1989.

[37] Peyman Mohajerin Esfahani and Daniel Kuhn. Data-driven distributionally robust optimization using the Wasserstein metric: performance guarantees and tractable reformulations. Mathematical Programming, 171(1–2):115–166, 2018.

[38] Jun Morimoto and Kenji Doya. Robust reinforcement learning. Neural Computation, 17(2):335–359, 2005.

[39] S. A. Murphy. Optimal dynamic treatment regimes. Journal of the Royal Statistical Society Series B: Statistical Methodology, 65(2):331–355, 2003.

[40] Arnab Nilim and Laurent El Ghaoui. Robust control of Markov decision processes with uncertain transition matrices. Operations Research, 53(5):780–798, 2005.

[41] Lerrel Pinto, James Davidson, Rahul Sukthankar, and Abhinav Gupta. Robust adversarial reinforcement learning. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 2817–2826. PMLR, 2017.

[42] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[43] Alexander Shapiro, Darinka Dentcheva, and Andrzej Ruszczy´nski. Lectures on Stochastic Programming: Modeling and Theory. Society for Industrial and Applied Mathematics, Philadelphia, PA, third edition, 2021.

[44] Lauren N. Steimle, David L. Kaufman, and Brian T. Denton. Multi-model Markov decision processes. IISE Transactions, 53(10):1124–1139, 2021.

[45] Maximilian A. R. Strobl, Jefrey West, Yannick Viossat, Mehdi Damaghi, Mark Robertson-Tessi, Joel S. Brown, Robert A. Gatenby, Philip K. Maini, and Alexander R. A. Anderson. Turnover modulates the need for a cost of resistance in adaptive therapy. Cancer Research, 81(4):1135–1147, 2021.

[46] Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, and Pieter Abbeel. Domain randomization for transferring deep neural networks from simulation to the real world. In 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 23–30. IEEE, 2017.

[47] C´edric Villani. Optimal Transport: Old and New, volume 338 of Grundlehren der mathematischen Wissenschaften. Springer, Berlin, Heidelberg, 2009.

[48] Wolfram Wiesemann, Daniel Kuhn, and Ber¸c Rustem. Robust Markov decision processes. Mathematics of Operations Research, 38(1):153–183, 2013.

[49] Haotian Ye, Xiaoyu Chen, Liwei Wang, and Simon Shaolei Du. On the power of pre-training for generalization in RL: Provable benefits and hardness. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 39770–39800, 2023.

[50] Chao Yu, Jiming Liu, Shamim Nemati, and Guosheng Yin. Reinforcement learning in healthcare: A survey. ACM Computing Surveys, 55(1):1–36, 2023.

[51] Huan Zhang, Hongge Chen, Chaowei Xiao, Bo Li, Mingyan Liu, Duane Boning, and Cho-Jui Hsieh. Robust deep reinforcement learning against adversarial perturbations on state observations. In Advances in Neural Information Processing Systems, volume 33, pages 21024–21037. Curran Associates, Inc., 2020.

[52] Jingsong Zhang, Jessica Cunningham, Joel Brown, and Robert Gatenby. Evolution-based mathematical models significantly prolong response to abiraterone in metastatic castrate-resistant prostate cancer and identify strategies to further improve outcomes. eLife, 11:e76284, 2022.

[53] Jingsong Zhang, Jessica J. Cunningham, Joel S. Brown, and Robert A. Gatenby. Integrating evolutionary dynamics into treatment of metastatic castrate-resistant prostate cancer. Nature Communications, 8(1):1816, 2017.

## A Supplementary Proofs in Section 2

To establish Theorem 2.1, we define the distance between two history-dependent policies π and $\pi ^ { \prime }$ as

$$
d _ { \Pi } ( \pi , \pi ^ { \prime } ) : = \operatorname* { m a x } _ { 0 \leq t \leq T - 1 } \operatorname* { m a x } _ { h _ { t } } \ \left\| \pi _ { t } ( \cdot \vert h _ { t } ) - \pi _ { t } ^ { \prime } ( \cdot \vert h _ { t } ) \right\| _ { 1 } ,
$$

where the inner maximum is taken over all histories $h _ { t }$ up to period t. We first show that the expected cumulative reward $J ( \pi ; c )$ is Lipschitz continuous in π with respect to $d _ { \Pi }$

Lemma A.1. Suppose $x , y ,$ and A are finite, and $0 \leq r _ { c } ( x , a ) \leq \bar { r }$ for all $( c , x , a ) \in \mathcal { C } \times \mathcal { X } \times \mathcal { A }$ . Then for any $c \in { \mathcal { C } }$ and any pair of history-dependent policies $\pi , \pi ^ { \prime }$ , we have

$$
\left| J ( \pi ; c ) - J ( \pi ^ { \prime } ; c ) \right| \leq { \frac { T ^ { 2 } \bar { r } } { 2 } } d _ { \Pi } ( \pi , \pi ^ { \prime } ) .
$$

Proof. Let us fix $c \in { \mathcal { C } } .$ , and denote by $z _ { t } = ( x _ { 0 } , y _ { 0 } , a _ { 0 } , \dotsc , a _ { t - 1 } , x _ { t } , y _ { t } )$ the full history up to period $t ,$ which consists of the history $h _ { t }$ together with the states $x _ { 0 } , \ldots , x _ { t }$ . To evaluate $\pi ^ { \prime }$ along the trajectories generated by $\pi ,$ , we set $V _ { T } \equiv 0$ and define recursively, for $t = T - 1 , \dots , 0$

$$
Q _ { t } ( z _ { t } , a ) : = r _ { c } ( x _ { t } , a ) + \mathbb { E } \left[ V _ { t + 1 } ( z _ { t + 1 } ) \mid z _ { t } , a _ { t } = a \right] , \qquad V _ { t } ( z _ { t } ) : = \sum _ { a \in \mathcal { A } } \pi _ { t } ^ { \prime } ( a \mid h _ { t } ) Q _ { t } ( z _ { t } , a ) ,
$$

where the conditional expectation is taken over $x _ { t + 1 } \sim P _ { c } ( \cdot | x _ { t } , a )$ with $y _ { t + 1 } = O _ { c } ( x _ { t + 1 } )$ , and is hence identical under $\pi$ and $\pi ^ { \prime } .$ . By induction, we have

$$
V _ { t } ( z _ { t } ) = \mathbb { E } _ { c } ^ { \pi ^ { \prime } } \left[ \sum _ { s = t } ^ { T - 1 } r _ { c } ( x _ { s } , a _ { s } ) \mid z _ { t } \right] , \qquad 0 \leq Q _ { t } ( z _ { t } , a ) \leq ( T - t ) \bar { r } .
$$

In addition, since the distribution of $z _ { 0 }$ does not depend on the policy, we have $J ( \pi ^ { \prime } ; c ) = \mathbb { E } _ { c } ^ { \pi } \left[ V _ { 0 } ( z _ { 0 } ) \right]$ Consequently,

$$
\begin{array} { r l } & { J ( \pi ; c ) - J ( \pi ^ { \prime } ; c ) \overset { ( a ) } { = } \mathbb { E } _ { c } ^ { \pi } \displaystyle \left[ \sum _ { t = 0 } ^ { T - 1 } \left( r _ { c } ( x _ { t } , a _ { t } ) + V _ { t + 1 } ( z _ { t + 1 } ) - V _ { t } ( z _ { t } ) \right) \right] } \\ & { \overset { ( b ) } { = } \mathbb { E } _ { c } ^ { \pi } \displaystyle \left[ \sum _ { t = 0 } ^ { T - 1 } \sum _ { a \in A } \left( \pi _ { t } ( a \mid h _ { t } ) - \pi _ { t } ^ { \prime } ( a \mid h _ { t } ) \right) \left( Q _ { t } ( z _ { t } , a ) - \frac { ( T - t ) \bar { r } } { 2 } \right) \right] , } \end{array}
$$

where (a) follows from $V _ { T } \equiv 0$ and $J ( \pi ^ { \prime } ; c ) = \mathbb { E } _ { c } ^ { \pi } \left[ V _ { 0 } ( z _ { 0 } ) \right]$ , and (b) follows from conditioning on $z _ { t }$ , the definitions of $Q _ { t }$ and $V _ { t }$ , and $\begin{array} { r } { \sum _ { a \in \mathcal { A } } \left( \pi _ { t } ( a \mid h _ { t } ) - \pi _ { t } ^ { \prime } ( a \mid h _ { t } ) \right) = 0 } \end{array}$ . Hence,

$$
\begin{array} { r l } & { \big | J ( \pi ; c ) - J ( \pi ^ { \prime } ; c ) \big | \overset { ( c ) } { \le } \mathbb { E } _ { c } ^ { \pi } \displaystyle \left[ \displaystyle \sum _ { t = 0 } ^ { T - 1 } \big \| \pi _ { t } ( \cdot \vert h _ { t } ) - \pi _ { t } ^ { \prime } ( \cdot \vert h _ { t } ) \big \| _ { 1 } \displaystyle \operatorname* { m a x } _ { a \in A } \Big | Q _ { t } ( z _ { t } , a ) - \frac { ( T - t ) \bar { r } } { 2 } \Big | \right] } \\ & { \qquad \overset { ( d ) } { \le } \displaystyle \sum _ { t = 0 } ^ { T - 1 } \frac { ( T - t ) \bar { r } } { 2 } d _ { \Pi } ( \pi , \pi ^ { \prime } ) = \displaystyle \frac { T ( T + 1 ) \bar { r } } { 4 } d _ { \Pi } ( \pi , \pi ^ { \prime } ) \le \displaystyle \frac { T ^ { 2 } \bar { r } } { 2 } d _ { \Pi } ( \pi , \pi ^ { \prime } ) , } \end{array}
$$

where (c) follows from H¨older’s inequality, and (d) follows from the definition of $d _ { \Pi }$ and $0 \leq Q _ { t } ( z _ { t } , a ) \leq$ $( T - t ) \bar { r }$ . The proof is then completed. □

We next control the number of policies needed to cover the class of all history-dependent policies.

Lemma A.2. Suppose $\mathcal { V }$ and A are finite. Then for any $\nu > 0$ , there exist $K \leq ( 1 + 4 / \nu ) ^ { ( | A | - 1 ) | \mathcal { H } | }$ history-dependent policies $\pi ^ { 1 } , \ldots , \pi ^ { K }$ such that

$$
\operatorname* { m i n } _ { 1 \leq k \leq K } d _ { \Pi } ( \pi , \pi ^ { k } ) \leq \nu \qquad \mathrm { f o r ~ a n y ~ h i s t o r y - d e p e n d e n t ~ p o l i c y ~ } \pi .
$$

Proof. Let us denote by $\Delta _ { \mathcal { A } }$ the probability simplex over ${ \mathcal { A } } ,$ and take a maximal subset $S \subset \Delta _ { { \cal A } }$ satisfying $\| q - q ^ { \prime } \| _ { 1 } > \nu$ for all distinct $q , q ^ { \prime } \in S$ . By the maximality of $S ,$ we have

$$
\operatorname* { m i n } _ { q \in S } \| p - q \| _ { 1 } \leq \nu \qquad { \mathrm { f o r ~ a n y ~ } } p \in \Delta _ { { \cal A } } .\tag{A.1}
$$

To control |S|, let us denote by $B ( q , r )$ the $\ell _ { 1 } { \mathrm { - b a l l } }$ of radius r around $q$ within the afine hull of $\Delta _ { \mathcal { A } }$ , which has dimension $| { \mathcal { A } } | - 1$ . Since $\| p - q \| _ { 1 } \leq 2$ for all $p , q \in \Delta _ { \mathcal { A } }$ , for any fixed $q _ { 0 } \in S$ , we have

$$
B ( q , \nu / 2 ) \cap B ( q ^ { \prime } , \nu / 2 ) = \emptyset \quad \mathrm { f o r ~ a l l ~ d i s t i n c t ~ } q , q ^ { \prime } \in S , \qquad \bigcup _ { q \in S } B ( q , \nu / 2 ) \subset B ( q _ { 0 } , 2 + \nu / 2 ) .
$$

By comparing the volumes in the above inclusion, we obtain

$$
\begin{array} { r } { | S | ( \nu / 2 ) ^ { | \mathcal { A } | - 1 } \le ( 2 + \nu / 2 ) ^ { | \mathcal { A } | - 1 } , \quad \mathrm { i . e . , } \quad | S | \le ( 1 + 4 / \nu ) ^ { | \mathcal { A } | - 1 } . } \end{array}
$$

Let $\{ \pi ^ { 1 } , \ldots , \pi ^ { K } \}$ be the set of policies satisfying $\pi _ { t } ( \cdot | h _ { t } ) \in S$ for any history $h _ { t } ,$ in which case $K = | S | ^ { | \mathcal { H } | } \le$ $( 1 + \dot { 4 } / \nu ) ^ { ( | A | - 1 ) | \mathcal { \dot { H } } | }$ . For any history-dependent policy π, by choosing $\bar { \pi ^ { k } }$ with $\pi _ { t } ^ { k } ( \cdot \vert h _ { t } ) \in \operatorname { a r g m i n } _ { q \in S } \| \pi _ { t } ( \cdot \vert h _ { t } ) -$ q∥<sub>1</sub> for any $h _ { t } .$ , we obtain $d _ { \Pi } ( \pi , \pi ^ { k } ) \leq \nu$ from (A.1). The proof is then completed. □

Proof of Theorem 2.1. Let us set $\eta = T \bar { r } / ( 4 \sqrt { N } )$ and $\nu = 2 \eta / ( T ^ { 2 } \bar { r } ) = 1 / ( 2 T \sqrt { N } )$ , and let $\pi ^ { 1 } , \ldots , \pi ^ { K }$ be the policies given by Lemma $\mathrm { A . 2 }$ , with $K \leq \dot { ( 1 + 8 T \sqrt { N } ) } ^ { ( | A | - 1 ) | \dot { \mathcal { H } } | }$ . Let us first fix k. Since $c _ { 1 } , \ldots , c _ { N }$ are drawn independently from D and $0 \leq r _ { c } \leq \bar { r }$ , the values $J ( \pi ^ { k } ; c _ { 1 } ) , \ldots , J ( \pi ^ { k } ; c _ { N } )$ are independent random variables taking values in $[ 0 , T \bar { r } ]$ with mean $J ( \pi ^ { k } ; \mathcal { D } )$ . By Hoefding’s inequality [20], we obtain

$$
\mathbb { P } \Big ( \big | \widehat { J } _ { N } ( \pi ^ { k } ) - J ( \pi ^ { k } ; \mathcal { D } ) \big | \geq s \Big ) \leq 2 \exp \left( - \frac { 2 N s ^ { 2 } } { T ^ { 2 } \bar { r } ^ { 2 } } \right)
$$

for any $s > 0$ . Let us take the union bound over $k = 1 , \ldots , K$ and choose

$$
s = s _ { N } : = T \bar { r } \sqrt { \frac { \log ( 2 K / \delta ) } { 2 N } } .
$$

Then with probability at least $1 - \delta .$ , we have

$$
\operatorname* { m a x } _ { 1 \leq k \leq K } \left| \widehat { J } _ { N } ( \pi ^ { k } ) - J ( \pi ^ { k } ; \mathcal { D } ) \right| \leq s _ { N } .\tag{A.2}
$$

Let us now fix $\pi \in \Pi$ and some k with $d _ { \Pi } ( \pi , \pi ^ { k } ) \leq \nu$ . Whenever (A.2) holds, we have

$$
\begin{array} { r l } & { \left| \widehat { J } _ { N } ( \pi ) - J ( \pi ; \mathcal D ) \right| \leq \left| \widehat { J } _ { N } ( \pi ) - \widehat { J } _ { N } ( \pi ^ { k } ) \right| + \left| \widehat { J } _ { N } ( \pi ^ { k } ) - J ( \pi ^ { k } ; \mathcal D ) \right| + \left| J ( \pi ^ { k } ; \mathcal D ) - J ( \pi ; \mathcal D ) \right| } \\ & { \qquad \overset { ( a ) } { \leq } s _ { N } + 2 \eta , } \end{array}\tag{A.3}
$$

where (a) follows from (A.2) and Lemma A.1 with $T ^ { 2 } \bar { r } \nu / 2 = \eta$ . Since (A.3) holds for any $\pi \in \Pi$ , we obtain

$$
J ( \widehat { \pi } _ { N } ; \mathcal { D } ) \overset { ( b ) } { \geq } \widehat { J } _ { N } ( \widehat { \pi } _ { N } ) - s _ { N } - 2 \eta \overset { ( c ) } { \geq } \widehat { J } _ { N } ( \pi ) - s _ { N } - 2 \eta \overset { ( d ) } { \geq } J ( \pi ; \mathcal { D } ) - 2 s _ { N } - 4 \eta ,
$$

where (b) and (d) follow from (A.3), and (c) follows from the definition of $\widehat { \pi } _ { N }$ . In addition, we have $4 \eta = T \bar { r } / \sqrt { N }$ and, since $K \leq ( 1 + 8 T { \sqrt { N } } ) ^ { ( | A | - 1 ) | \mathcal { H } | }$ ，

$$
2 s _ { N } = T \bar { r } \sqrt { \frac { 2 \log ( 2 K / \delta ) } { N } } \le T \bar { r } \sqrt { \frac { 2 ( | A | - 1 ) | \mathcal { H } | \log \left( 1 + 8 T \sqrt { N } \right) + 2 \log ( 2 / \delta ) } { N } } .
$$

The desired claim then follows by taking the supremum over $\pi \in \Pi .$

Proof of Theorem 2.2. Let $\begin{array} { r } { \widehat { \mathcal { D } } _ { N } = N ^ { - 1 } \sum _ { i = 1 } ^ { N } \delta _ { c _ { i } } } \end{array}$ denote the empirical distribution of $c _ { 1 } , \ldots , c _ { N }$ , which satisfies $\widehat J _ { N } ( \pi ) = \mathbb E _ { c \sim \widehat { D } _ { N } } \left[ J ( \pi ; c ) \right]$ . To compare $J ( \pi ; \mathcal { D } )$ with ${ \widehat { J } } _ { N } ( \pi )$ , we use the 1-Wasserstein distance [47] between distributions $\dot { P }$ and $Q$ on ${ \mathcal { C } } ,$ given by

$$
W _ { 1 } ( P , Q ) : = \operatorname* { i n f } _ { \gamma \in \Gamma ( P , Q ) } \int \| c - c ^ { \prime } \| _ { 2 } \gamma ( d c , d c ^ { \prime } ) ,
$$

where $\Gamma ( P , Q )$ denotes the set of couplings of $P$ and $Q .$ Let us first fix $\pi \in \Pi$ . For any $\gamma \in \Gamma ( \mathcal { D } , \widehat { \mathcal { D } } _ { N } )$ , we have

$$
\left| J ( \pi ; \mathcal { D } ) - \widehat { J } _ { N } ( \pi ) \right| = \left| \int \left( J ( \pi ; c ) - J ( \pi ; c ^ { \prime } ) \right) \gamma ( d c , d c ^ { \prime } ) \right| \leq L _ { T } \int \| c - c ^ { \prime } \| _ { 2 } \gamma ( d c , d c ^ { \prime } ) ,
$$

where the inequality follows from (2.3). Taking the infimum over $\gamma$ and then the supremum over $\pi \in \Pi$ , we obtain

$$
\operatorname* { s u p } _ { \pi \in \Pi } \big | J ( \pi ; \mathcal D ) - \widehat { J } _ { N } ( \pi ) \big | \leq L _ { T } W _ { 1 } ( \mathcal D , \widehat { \mathcal D } _ { N } ) .\tag{A.4}
$$

Consequently, for any $\pi \in \Pi$

$$
\begin{array} { r } { J \big ( \widehat \pi _ { N } ; \mathcal D \big ) \overset { ( a ) } \geq \widehat J _ { N } \big ( \widehat \pi _ { N } \big ) - L _ { T } W _ { 1 } \big ( \mathcal D , \widehat { \mathcal D } _ { N } \big ) \overset { ( b ) } \geq \widehat J _ { N } ( \pi ) - L _ { T } W _ { 1 } \big ( \mathcal D , \widehat { \mathcal D } _ { N } \big ) \overset { ( c ) } \geq J \big ( \pi ; \mathcal D \big ) - 2 L _ { T } W _ { 1 } \big ( \mathcal D , \widehat { \mathcal D } _ { N } \big ) , } \end{array}
$$

where (a) and (c) follow from (A.4), and (b) follows from the definition of $\widehat { \pi } _ { N }$ . Taking the supremum over $\pi \in \Pi$ , we obtain

$$
\operatorname* { s u p } _ { \pi \in \Pi } J ( \pi ; \mathcal { D } ) - J ( \widehat { \pi } _ { N } ; \mathcal { D } ) \leq 2 L _ { T } W _ { 1 } ( \mathcal { D } , \widehat { \mathcal { D } } _ { N } ) .\tag{A.5}
$$

It remains to control $W _ { 1 } ( \mathcal { D } , \widehat { \mathcal { D } } _ { N } )$ with high probability. Since ${ \cal W } _ { 1 } ( { \mathcal D } , \widehat { { \mathcal D } } _ { N } ) = 0$ whenever diam $( \mathcal { C } ) = 0$ and $W _ { 1 }$ is invariant under a common translation of both distributions, without loss of generality, let us assume diam $( \mathcal { C } ) > 0$ and $0 \in { \mathcal { C } }$ , in which case $\mathbb { E } _ { c \sim \mathcal { D } } \left\lceil \| c \| _ { 2 } ^ { 3 } \right\rceil \leq \dim ( \mathcal { C } ) ^ { 3 }$ . By Theorem 1 of [14] with transport cost $\| c - c ^ { \prime } \| _ { 2 }$ and moment of order 3, there exists a constant $C _ { d }$ depending only on d such that

$$
\mathbb { E } \left[ W _ { 1 } ( \mathcal { D } , \widehat { \mathcal { D } } _ { N } ) \right] \leq \frac { C _ { d } \mathrm { d i a m } ( \mathcal { C } ) \log ( 2 N ) } { N ^ { \operatorname* { m i n } \{ 1 / 2 , 1 / d \} } } ,
$$

where the logarithmic factor is needed only for $d = 2$ , and the additional term of order $N ^ { - 2 / 3 }$ in $[ 1 4 ]$ is absorbed into $C _ { d }$ . In addition, let $\widehat { \mathcal { D } } _ { N } ^ { ( i ) }$ denote the empirical distribution obtained by replacing $c _ { i }$ with some $c _ { i } ^ { \prime } \in \mathcal { C }$ . By coupling every unchanged sample with itself and $c _ { i }$ with $c _ { i } ^ { \prime } ,$ we have

$$
\big | W _ { 1 } ( \mathcal D , \widehat D _ { N } ) - W _ { 1 } ( \mathcal D , \widehat D _ { N } ^ { ( i ) } ) \big | \leq W _ { 1 } ( \widehat D _ { N } , \widehat D _ { N } ^ { ( i ) } ) \leq \frac { \| c _ { i } - c _ { i } ^ { \prime } \| _ { 2 } } { N } \leq \frac { \mathrm { d i a m } ( \mathcal C ) } { N } .
$$

Hence it follows from McDiarmid’s inequality [36] that

$$
\mathbb { P } \Big ( W _ { 1 } ( \mathcal { D } , \widehat { \mathcal { D } } _ { N } ) - \mathbb { E } \Big [ W _ { 1 } ( \mathcal { D } , \widehat { \mathcal { D } } _ { N } ) \Big ] \geq s \Big ) \leq \exp \left( - \frac { 2 N s ^ { 2 } } { \mathrm { d i a m } ( \mathcal { C } ) ^ { 2 } } \right)
$$

for any $s > 0$ . Let us choose $s = \dim ( { \mathcal { C } } ) \sqrt { \log ( 1 / \delta ) / ( 2 N ) }$ . Then with probability at least $1 - \delta$ , we have

$$
W _ { 1 } ( \mathcal { D } , \widehat { \mathcal { D } } _ { N } ) \leq \mathrm { d i a m } ( \mathcal { C } ) \left( \frac { C _ { d } \log ( 2 N ) } { N ^ { \operatorname* { m i n } \{ 1 / 2 , 1 / d \} } } + \sqrt { \frac { \log ( 1 / \delta ) } { 2 N } } \right) .\tag{A.6}
$$

The desired claim then follows by combining (A.5) and (A.6).

Proof of Proposition 2.1. Let us fix $\pi \in \Pi$ and $c , c ^ { \prime } \in { \mathcal { C } }$ , and denote by $z _ { t } = ( x _ { 0 } , y _ { 0 } , a _ { 0 } , \dotsc , a _ { t - 1 } , x _ { t } , y _ { t } )$ the full history up to period t as in the proof of Lemma A.1. Since $O _ { c } = O$ , we have $y _ { t } = O ( x _ { t } )$ under both c and $c ^ { \prime } .$ For $\widetilde { c } \in \{ c , c ^ { \prime } \}$ , we set $V _ { T } ^ { \widetilde { c } } \equiv 0$ and define recursively, for $t = T - 1 , \dots , 0$

$$
V _ { t } ^ { \tilde { c } } ( z _ { t } ) : = \sum _ { a \in \mathcal { A } } \pi _ { t } ( a \mid h _ { t } ) \Big ( r _ { \tilde { c } } ( x _ { t } , a ) + \int V _ { t + 1 } ^ { \tilde { c } } ( z _ { t } , a , x ^ { \prime } , O ( x ^ { \prime } ) ) P _ { \tilde { c } } ( d x ^ { \prime } \mid x _ { t } , a ) \Big ) .
$$

By induction, we have $0 \leq V _ { t } ^ { \widetilde { c } } ( z _ { t } ) \leq ( T - t ) \bar { r }$ and $\begin{array} { r } { J ( \pi ; \widetilde { c } ) = \int V _ { 0 } ^ { \widetilde { c } } d \mu _ { \widetilde { c } } } \end{array}$ . In addition, for any distributions $P$ and $Q$ and any measurable f taking values in $[ 0 , M ]$ , we have

$$
\bigg | \int f d P - \int f d Q \bigg | \leq M d _ { \mathrm { T V } } ( P , Q ) .\tag{A.7}
$$

Let us denote $\begin{array} { r } { \xi _ { t } = \operatorname* { s u p } _ { z _ { t } } \left| V _ { t } ^ { c } ( z _ { t } ) - V _ { t } ^ { c ^ { \prime } } ( z _ { t } ) \right| } \end{array}$ . Then for any $z _ { t } ,$

$$
\left| V _ { t } ^ { c } ( z _ { t } ) - V _ { t } ^ { c ^ { \prime } } ( z _ { t } ) \right| \leq \sum _ { a \in \mathcal { A } } \pi _ { t } ( a \mid h _ { t } ) \Big ( \big | r _ { c } ( x _ { t } , a ) - r _ { c ^ { \prime } } ( x _ { t } , a ) \big | + \Big | \int \left( V _ { t + 1 } ^ { c } - V _ { t + 1 } ^ { c ^ { \prime } } \right) ( z _ { t } , a , x ^ { \prime } , O ( x ^ { \prime } ) ) P _ { c } ( d x ^ { \prime } \mid x _ { t } , a ) \Big |
$$

$$
\begin{array} { r l r } & { } & { \displaystyle + \Big | \int V _ { t + 1 } ^ { c ^ { \prime } } ( z _ { t } , a , x ^ { \prime } , O ( x ^ { \prime } ) ) \left( P _ { c } ( d x ^ { \prime } \mid x _ { t } , a ) - P _ { c ^ { \prime } } ( d x ^ { \prime } \mid x _ { t } , a ) \right) \Big | \Big ) } \\ & { } & { \stackrel { ( a ) } { \leq } L \| c - c ^ { \prime } \| _ { 2 } + \xi _ { t + 1 } + ( T - t - 1 ) \bar { r } L \| c - c ^ { \prime } \| _ { 2 } , ~ } \end{array}
$$

where (a) follows from the conditions on $r _ { c }$ and $P _ { c }$ in Proposition 2.1, the definition of $\xi _ { t + 1 }$ , and (A.7) with $f = V _ { t + 1 } ^ { c ^ { \prime } }$ and $M = ( T - t - 1 ) \bar { r }$ . Taking the supremum over $z _ { t }$ and summing up the resulting relation over $t = 0 , \ldots , T - 1$ , we obtain from $\xi _ { T } = 0$ that

$$
\xi _ { 0 } \le \sum _ { t = 0 } ^ { T - 1 } \big ( 1 + ( T - t - 1 ) \bar { r } \big ) L \| c - c ^ { \prime } \| _ { 2 } = L T \left( 1 + \frac { \bar { r } ( T - 1 ) } { 2 } \right) \| c - c ^ { \prime } \| _ { 2 } .
$$

Consequently,

$$
\begin{array} { l } { \displaystyle | J ( \pi ; c ) - J ( \pi ; c ^ { \prime } ) | \leq \displaystyle | \int ( V _ { 0 } ^ { c } - V _ { 0 } ^ { c ^ { \prime } } ) d \mu _ { c } | + \displaystyle | \int V _ { 0 } ^ { c ^ { \prime } } d \mu _ { c } - \int V _ { 0 } ^ { c ^ { \prime } } d \mu _ { c ^ { \prime } } | \overset { ( b ) } { \leq } \xi _ { 0 } + T \bar { r } L \| c - c ^ { \prime } \| _ { 2 } } \\ { \displaystyle  \leq L T ( 1 + \frac { \bar { r } ( T + 1 ) } { 2 } ) \| c - c ^ { \prime } \| _ { 2 } , } \end{array}
$$

where (b) follows from the condition on $\mu _ { c }$ in Proposition 2.1 and (A.7) with $f = V _ { 0 } ^ { c ^ { \prime } }$ and $M = T \bar { r }$ . The proof is then completed. □

## B Implementation Details

## B.1 Patient Model

In the public code and data of [52], the PSA records and the fitted initial cell numbers are available for 32 patients, namely 17 patients of the pilot clinical trial and 15 patients of the contemporaneous cohort who received continuous treatment with abiraterone as the standard of care. Since the records of 25 patients begin before the first dose of abiraterone, we follow the public code to reconstruct the daily treatment indicator of each patient from the recorded treatment history, integrate (2.4) with $( \alpha , \beta ) = ( 6 , 8 )$ from the fitted initial cell numbers along the reconstructed indicator, and take the cell numbers on the day of the first dose as the fitted initial state $( S _ { 0 } , R _ { 0 } )$ . Throughout, we integrate (2.4) by the classical fourth-order Runge-Kutta method with step size 1 in $\tau ,$ and truncate S and R to [0, κ] after every step, as in the public code. For all 32 patients, we reproduce the trajectories computed by the public code up to a relative error below $1 0 ^ { - 1 3 }$

## B.2 Training Populations

For $\mathcal { D } _ { \mathrm { c a l } }$ , we select one of the 32 fitted initial states uniformly at random, multiply $S _ { 0 }$ and $R _ { 0 }$ by independent lognormal factors with log-standard deviation 0.2, and set $( \alpha , \beta ) = ( 6 , 8 )$ . For $\mathcal { D } _ { \mathrm { w i d e } }$ , we draw $\alpha \sim U [ 2 , 8 ]$ $\beta \sim U [ 6 , 1 0 ] , B _ { 0 } \sim U [ 0 . 1 \kappa , 0 . 8 \kappa ]$ , and the initial resistant fraction $R _ { 0 } / B _ { 0 }$ log-uniformly on $[ 0 . 0 0 1 , 0 . 5 ]$ . The range of α extends from the value assumed in the earlier model of [53] to above the fitted value 6, and the range of $\beta$ lies within 25% of the fitted value 8. We take the ranges of $B _ { 0 }$ and $R _ { 0 } / B _ { 0 }$ from the priors of [5], with the upper end of $B _ { 0 }$ lowered from κ to 0.8κ to keep the progression threshold $1 . 2 B _ { 0 }$ below the carrying capacity, and with $R _ { 0 } / B _ { 0 }$ drawn log-uniformly to adequately represent patients with few resistant cells. The virtual test population is drawn from $\mathcal { D } _ { \mathrm { c a l } }$ with a fixed seed.

## B.3 Generating Kernels

We take as $Q _ { \mathrm { n o m } }$ the distribution of $m = ( m _ { \alpha } , m _ { \beta } )$ with log $m _ { \alpha } \sim N ( 0 , 0 . 5 5 ^ { 2 } )$ and log $m _ { \beta } \sim N ( 0 , 0 . 1 2 5 ^ { 2 } )$ independent and truncated at two standard deviations. Under $Q _ { \mathrm { n o m } }$ , the competitive efect $m _ { \alpha } \alpha$ at $\alpha = 6$ ranges approximately over [2, 18], whose lower end corresponds to the value assumed in the earlier model of [53] and whose upper end remains within the range of competition coeficients reported among species in nature [52]. We draw the training kernels and the held-out kernels of Section 3.4 independently from $Q _ { \mathrm { n o m } }$ with diferent random seeds.

## B.4 Policy Network and Training

The input of the policy network at visit t consists of six features of $h _ { t }$ , namely $y _ { t }$ , the previous action, the elapsed time, the fraction of treated months, the time since the last change of action, and an indicator of forced treatment. We map these features through a linear layer with 64 units and tanh activation to a GRU with 64 hidden units, followed by linear heads for the action logits and the value. At visits with forced treatment, we exclude the action $a _ { t } = 0$ by masking its logit. The loss of PPO combines the clipped surrogate objective with clipping parameter 0.2, the value loss with coeficient 0.5, and the entropy bonus with coeficient 0.01, where the advantages are computed by generalized advantage estimation with parameter 0.95. In each update, we minimize the loss by the Adam optimizer with learning rate $3 \times 1 0 ^ { - 4 }$ over four epochs with four minibatches, clip the gradient norm at 0.5, and stop the update once the approximate Kullback-Leibler divergence between the new and old policies exceeds 0.03. In training, we divide the reward $r _ { c }$ of Section 2.2 by T. For every combination of policy type, training radius, and training population, we train three policies with the random seeds 11, 23, and 37, and report each policy after the final update, without selecting checkpoints on test data.

## B.5 Kernel Adversary

We parameterize the kernel adversary by a network of the same architecture as the policy network, and train it by PPO with the same hyperparameters, using the negative of the reward of the controller. In solving (2.8), we alternate between four adversary updates with the policy fixed and one policy update with the adversary fixed. The adversary selects the generating kernel of month t before observing $a _ { t } .$ , based on the observation $y _ { t }$ , the previous action $a _ { t - 1 }$ , the elapsed time $t / T$ , and the one-hot encoding of the kernel it proposed in the previous month. We use its output head to parameterize a categorical distribution over the $K = 3 2$ generating kernels. In accordance with (2.9), the patient evolves under the proposed kernel with probability $\rho$ and under a uniformly drawn generating kernel with probability $1 - \rho ,$ and the adversary does not observe which of the two occurs. We initialize the output head of the adversary at zero, and hence the adversary proposes the kernels uniformly at random at initialization.

## B.6 Evaluation

At evaluation, the learned policies select their most likely action at every visit. All methods are evaluated on the same test patients with the same random numbers, and on the fitted patients, each patient is evaluated under 50 independent replicates of the random perturbations. For every policy $\pi$ and evaluation radius $\rho > 0$ , we estimate the robust value min ${ } _ { \sigma } J ( \pi , \sigma ; \mathcal { D } )$ of $\pi$ by training a new best-response adversary against π, held fixed, for 2,000 PPO updates, on patients drawn from the calibrated population $\mathcal { D } _ { \mathrm { c a l } }$ for the virtual test population and on the fitted patients for the fitted test population, and average the performance of the policy against this adversary, which samples its actions, over 32 independent repetitions. It should be noted that the best-response adversary is computed numerically, and the value attained against it hence provides an upper estimate of the exact robust value. AT50 and continuous treatment are evaluated against their own best-response adversaries. Under shifts in the dynamics, we compute $G ^ { \mathrm { o n } } ( c )$ under the same adversary, held fixed, acting on the continuous-treatment trajectory, with common random numbers.

## C Complete Numerical Results

To complement Section 3, we report the complete values underlying its findings. The comparison across levels of initial resistance in Section 3.2 extends to both test populations, on which nominal RL improves on AT50 in every tertile of the initial resistant fraction, while the harm rate of AT50 is highest among patients with few resistant cells (Table 6). The per-seed values of the learned policies under the conditions of Table 5 agree closely for nominal RL, random-perturbation RL, and kernel-robust RL in the nominal model and at $\rho = 0 \ ( \mathrm { T a b l e } \ 7 )$ , and on the fitted patients, the three seeds of nominal RL attain 48.31, 48.66, and 48.31 months in the nominal model. The harm rates discussed in Section 3.6 for the policies trained on the broad population are given in Table 8, together with their mean progression-free times under shifts in the patient dynamics on the fitted patients.

Table 4: Mean progression-free time (months) on the virtual test population in the nominal model and under random perturbations of PSA with $\varepsilon = 0 . 2 .$ with harm rates (%) in parentheses.
<table><tr><td>Method</td><td>Nominal model</td><td> $\varepsilon = 0 . 2$ </td></tr><tr><td>Continuous treatment AT50</td><td>15.81 29.58 (28.4)</td><td>15.81 16.46 (43.8)</td></tr><tr><td>Nominal RL</td><td>50.35 (0.2)</td><td>32.96 (14.4)</td></tr><tr><td>Random-perturbation RL</td><td>45.94 (0.1)</td><td>44.49 (2.0)</td></tr></table>

Table 5: Cross-evaluation on the virtual test population (mean progression-free time in months). The columns $\rho \in \{ 0 , 0 . 2 5 , 1 \}$ denote evaluation radii on the training kernels, and the column $\varepsilon = 0 . 2$ denotes random perturbations of PSA.
<table><tr><td></td><td></td><td colspan="3">Shifts in the dynamics</td><td>Errors in PSA</td></tr><tr><td>Method</td><td>Nominal model</td><td> $\rho = 0$ </td><td> $\rho = 0 . 2 5$ </td><td> $\rho = 1$ </td><td> $\varepsilon = 0 . 2$ </td></tr><tr><td>Continuous treatment</td><td>15.81</td><td>16.11</td><td>15.68</td><td>12.72</td><td>15.81</td></tr><tr><td>AT50</td><td>29.58</td><td>19.04</td><td>17.92</td><td>11.65</td><td>16.46</td></tr><tr><td>Nominal RL</td><td>50.35</td><td>45.74</td><td>39.12</td><td>17.46</td><td>32.96</td></tr><tr><td>Random-perturbation RL</td><td>45.94</td><td>47.15</td><td>40.91</td><td>18.80</td><td>44.49</td></tr><tr><td>Kernel-robust RL,  $\rho _ { \mathrm { t r } } = 0 . 2 5$ </td><td>48.77</td><td>53.31</td><td>46.57</td><td>23.26</td><td>38.65</td></tr></table>

Table 6: Mean progression-free time (months) by tertile of the initial resistant fraction $R _ { 0 } / B _ { 0 }$ in the nominal model, with harm rates (%) in parentheses. Tertiles are computed separately for each test population.
<table><tr><td>Initial resistant fraction</td><td>Nominal RL</td><td>AT50</td><td>Continuous treatment</td></tr><tr><td>Virtual test population</td><td></td><td></td><td></td></tr><tr><td>Low</td><td>60.81 (0.4)</td><td>37.28 (39.9)</td><td>23.92</td></tr><tr><td>Middle</td><td>54.70 (0.1)</td><td>31.56 (34.5)</td><td>15.89</td></tr><tr><td>High</td><td>35.57 (0.0)</td><td>19.93 (10.8)</td><td>7.63</td></tr><tr><td>Fitted patients</td><td></td><td></td><td></td></tr><tr><td>Low</td><td>62.30 (0.0)</td><td>31.40 (50.0)</td><td>22.50</td></tr><tr><td>Middle</td><td>54.55 (0.0)</td><td>28.91 (27.3)</td><td>14.27</td></tr><tr><td>High</td><td>29.70 (0.0)</td><td>16.27 (9.1)</td><td>7.91</td></tr></table>

Table 7: Per-seed mean progression-free time (months) on the virtual test population for the conditions of Table 5.
<table><tr><td>Method</td><td>Seed</td><td>Nominal model</td><td> $\rho = 0$ </td><td> $\rho = 0 . 2 5$ </td><td> $\rho = 1$ </td><td> $\varepsilon = 0 . 2$ </td></tr><tr><td rowspan="3">Nominal RL</td><td>11</td><td>50.33</td><td>46.00</td><td>39.91</td><td>14.87</td><td>34.66</td></tr><tr><td>23</td><td>50.32</td><td>46.25</td><td>39.56</td><td>15.97</td><td>31.85</td></tr><tr><td>37</td><td>50.40</td><td>44.96</td><td>37.89</td><td>21.53</td><td>32.38</td></tr><tr><td rowspan="3">Random-perturbation RL</td><td>11</td><td>45.76</td><td>47.58</td><td>41.47</td><td>17.11</td><td>44.50</td></tr><tr><td>23</td><td>46.12</td><td>47.35</td><td>41.51</td><td>18.99</td><td>44.46</td></tr><tr><td>37</td><td>45.95</td><td>46.52</td><td>39.75</td><td>20.29</td><td>44.51</td></tr><tr><td rowspan="3">Kernel-robust RL,  $\rho _ { \mathrm { t r } } = 0 . 1$ </td><td>11</td><td>47.95</td><td>53.58</td><td>46.15</td><td>20.43</td><td>38.86</td></tr><tr><td>23</td><td>48.57</td><td>53.45</td><td>46.73</td><td>25.28</td><td>36.31</td></tr><tr><td>37</td><td>47.88</td><td>53.30</td><td>46.58</td><td>24.42</td><td>39.72</td></tr><tr><td rowspan="2">Kernel-robust RL,  $\rho _ { \mathrm { t r } } = 0 . 2 5$ </td><td>11</td><td>48.71</td><td>53.48</td><td>46.61</td><td>23.12</td><td>38.15</td></tr><tr><td>23</td><td>48.34</td><td>53.54</td><td>46.57</td><td>25.19</td><td>38.11</td></tr><tr><td>Method</td><td></td><td>Seed Nominal model</td><td> $\rho = 0$ </td><td> $\rho = 0 . 2 5$ </td><td> $\rho = 1$ </td><td> $\varepsilon = 0 . 2$ </td></tr><tr><td></td><td>37</td><td>49.25</td><td>52.90</td><td>46.54</td><td>21.48</td><td>39.67</td></tr><tr><td rowspan="3">Kernel-robust RL,  $\rho _ { \mathrm { t r } } = 0 . 5$ </td><td>11</td><td>48.66</td><td>53.02</td><td>46.11</td><td>24.66</td><td>38.27</td></tr><tr><td>23</td><td>48.13</td><td>53.19</td><td>46.93</td><td>25.77</td><td>36.10</td></tr><tr><td>37</td><td>48.56</td><td>52.98</td><td>46.48</td><td>24.79</td><td>36.38</td></tr></table>

Table 8: Shifts in the patient dynamics on the 32 fitted patients, policies trained on the broad population. Notation as in Table 2.
<table><tr><td rowspan="2">Condition Nominal RL</td><td rowspan="2"></td><td colspan="3">Kernel-robust RL</td><td rowspan="2"></td><td rowspan="2">Continuous treatment</td></tr><tr><td> $\rho _ { \mathrm { t r } } = 0 . 1$ </td><td> $\rho _ { \mathrm { t r } } = 0 . 2 5$ </td><td> $\rho _ { \mathrm { t r } } = 0 . 5$ </td></tr><tr><td>Nominal model</td><td>48.59 (0.0)</td><td>45.58 (0.0)</td><td>46.03 (0.0)</td><td>45.77 (0.0)</td><td>AT50 25.34 (28.1)</td><td>14.66</td></tr><tr><td>Training kernels</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\rho = 0$ </td><td>46.14 (7.1)</td><td>53.34 (0.7)</td><td>52.84 (0.9)</td><td>53.31 (0.9)</td><td>17.82 (45.4)</td><td>14.97</td></tr><tr><td> $\rho = 0 . 1$ </td><td>42.08 (10.3)</td><td>50.15 (0.9)</td><td>50.03 (1.1)</td><td>50.19 (1.4)</td><td>17.72 (46.1)</td><td>14.83</td></tr><tr><td> $\rho = 0 . 2 5$ </td><td>33.99 (15.3)</td><td>44.32 (1.3)</td><td>43.31 (2.9)</td><td>44.59 (2.0)</td><td>16.91 (48.3)</td><td>14.37</td></tr><tr><td> $\rho = 0 . 5$ </td><td>24.40 (24.2)</td><td>34.51 (5.8)</td><td>33.13 (8.4)</td><td>34.81 (6.2)</td><td>14.18 (53.9)</td><td>12.85</td></tr><tr><td> $\rho = 1$ </td><td>13.64 (47.3)</td><td>20.03 (16.7)</td><td>20.79 (12.3)</td><td>21.18 (8.4)</td><td>10.07 (79.3)</td><td>10.90</td></tr><tr><td>Held-out kernels</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\rho = 0$ </td><td>47.29 (8.5)</td><td>56.35 (0.9)</td><td>55.82 (1.1)</td><td>55.95 (1.3)</td><td>18.38 (47.9)</td><td>15.65</td></tr><tr><td> $\rho = 0 . 1$ </td><td>43.77 (10.7)</td><td>52.59 (1.0)</td><td>51.99 (1.3)</td><td>52.25 (1.5)</td><td>18.52 (46.9)</td><td>15.46</td></tr><tr><td> $\rho = 0 . 2 5$ </td><td>34.36 (16.7)</td><td>45.20 (1.3)</td><td>43.99 (2.4)</td><td>45.22 (1.5)</td><td>17.24 (50.5)</td><td>14.91</td></tr><tr><td> $\rho = 0 . 5$ </td><td>24.90 (25.8)</td><td>33.93 (4.1)</td><td>33.11 (4.7)</td><td>34.60 (3.2)</td><td>14.92 (56.3)</td><td>13.76</td></tr><tr><td> $\rho = 1$ </td><td>18.79 (13.3)</td><td>20.86 (0.0)</td><td>20.68 (0.0)</td><td>20.74 (0.0)</td><td>10.89 (63.6)</td><td>11.64</td></tr></table>

## D Alternative Definition of Progression

To justify the definition of progression in Section 2.2, we compare it with an alternative definition, under which progression is declared only when the tumor grows under treatment as in [5]. Specifically, we declare progression at the first visit t with $a _ { t - 1 } = 1 , B _ { t } \geq 1 . 2 B _ { 0 }$ , and $B _ { t } > B _ { t - 1 }$ , i.e., when the tumor exceeds the threshold during treatment and is still growing. Growth during a treatment break is then never penalized, and the forced restart at $y _ { t } \geq 0$ occurs below the threshold $1 . 2 B _ { 0 }$ . We report the results in the nominal model under this definition in Table 9, together with the forced-only rule, which withholds treatment whenever permitted, i.e., sets $a _ { t } = 1$ if and only if t = 0 or $y _ { t } \geq 0$ . All three seeds of nominal $\mathrm { R L } ,$ trained under this definition, attain 53.79 months on the virtual test population and 53.75 months on the fitted patients, which coincide with the values of the forced-only rule, while AT50 attains 52.60 and 53.06 months. Consequently, under the alternative definition, the benefit of adapting the treatment to PSA beyond the forced-treatment rule cannot be assessed.

Table 9: Mean progression-free time (months) under the alternative definition of progression in the nominal model. Forced only denotes the rule that treats only when forced.
<table><tr><td>Population</td><td>Continuous</td><td>AT50</td><td>Forced only</td><td>Nominal RL</td><td>Seeds 11 / 23 / 37</td></tr><tr><td>Virtual</td><td>15.81</td><td>52.60</td><td>53.79</td><td>53.79</td><td>53.79 / 53.79 / 53.79</td><td></td></tr><tr><td>Fitted patients</td><td>14.66</td><td>53.06</td><td>53.75</td><td>53.75</td><td>53.75 / 53.75 / 53.75</td><td></td></tr></table>