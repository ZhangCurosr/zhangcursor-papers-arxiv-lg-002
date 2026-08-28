(2) Smaller Pop Size for Larger Models

# Understanding Evolution Strategies for LLM Reasoning: Broader Reasoning Coverage than GRPO

Yunpeng Ba<sup>1,∗</sup>, Zhi Zheng<sup>2,∗</sup>, Yue Xie<sup>1</sup>, Jiaqing Li<sup>5</sup>, Xialiang Tong<sup>3</sup>, Tao Zhong<sup>3</sup>, Mingxuan Yuan<sup>3</sup>, Zhichao Lu<sup>4</sup>, Xuyang Wu<sup>1</sup>, Zhenkun Wang<sup>1</sup>

<sup>1</sup>Southern University of Science and Technology, <sup>2</sup>National University of Singapore, <sup>3</sup>Huawei Noah’s Ark Lab, <sup>4</sup>City University of Hong Kong, <sup>5</sup>Harbin Institute of Technology, Weihai

## Abstract

Evolution Strategies (ES) have recently emerged as a memory-eficient post-training paradigm for LLM reasoning. However, the optimization behavior of ES remains understudied, making it hard to define its advantage scope compared to mainstream post-training paradigms (e.g., Group Relative Policy Optimization (GRPO)). By systematically investigating ES dynamics and mechanisms, this paper first identifies a performance advantage of ES over GRPO, theoretically and empirically showing that ES can lead to broader reasoning coverage, thereby better exploiting the reasoning capabilities of pretrained LLMs. Theoretically, we show that verifier-projected Jensen–Shannon diversity across the ES population is helpful to higher Pass@K performances. Empirically, unlike GRPO, which exhibits entropy collapse, ES improves Pass@1 while attaining higher Pass@K than GRPO. We further develop a sequential GRPO–ES training strategy that combines GRPO’s strength in Pass@1 with ES’s gains in Pass@K. Second, we find that despite substantial whole-model parameter drift, the task-performance gains of ES are only contributed to a sparse subset of larger-magnitude updates. This functional sparsity suggests that large parameter movement need not imply widespread functional change, and held-out evaluations further show that it does not necessarily lead to catastrophic forgetting. Finally, we study how hyperparameter design afects the efectiveness of ES, demonstrating that ES requires a smaller population size in a larger LLM. These findings position ES as a distinct reasoning post-training paradigm rather than a less efective, memory-eficient alternative to GRPO.

GitHub : https://github.com/yunpengba7/understanding-es Correspondence: zhi.zheng@u.nus.edu, wangzhenkun90@gmail.com

## (a) RQ1: Post-Training Characteristics: Pass@K Advantage of ES

![](images/34762fe6be694cebe3c72a6ab4fbc675f38b346bf8c9265dad338bfa5640ca35.jpg)

(c) RQ3: Proper Settings (1) Z-Score Normalization  
![](images/6c94d48adc6a77e45ee8f5daa3ff313aaef206a9eb8001b62743822a596e9460.jpg)

![](images/fde8bab308ede4db9286113270c2ba224482b5c584ae5127a599cf0a2a69f2a3.jpg)

![](images/6c607cba9e8cb5b2a7648ea0665c689a97c1a2e631d3305856a6dba3299adf2d.jpg)

![](images/397db6b17646341579c5f3d469e22237ae41f320dc86ece9cf7a76e365f1b118.jpg)  
(b) RQ2: Large Parameter Drift ⇏ Catastrophic Forgetting

(3) One-Point Estimation  
![](images/da8fd3b29b39ea3dceed753746b8b3821eb95970d8cf586a27bd80325f4a77bb.jpg)

![](images/513c23db9c1a2de33f2058064ec31f93f2c6c0ee6e426c19323b94a6a42e798c.jpg)  
Figure 1 Overview of the three research questions and main findings. Panel (a) contrasts ES and GRPO post-training behavior; Panel (b) shows that keeping larger ES updates preserves target-task performance and reports held-out Maj@32 changes; Panel (c) summarizes normalization, population-size, and estimator choices.

## 1 Introduction

Recent work has shown that Evolution Strategies (ES) (Salimans et al., 2017) can fine-tune large language models (LLMs) to improve their reasoning capabilities (Qiu et al., 2026; Sarkar et al., 2026; Zheng et al., 2026a). ES optimizes a model by perturbing its parameters, evaluating the perturbed models through forward passes, and aggregating reward-weighted perturbations to estimate an update direction. By avoiding backpropagation, ES provides a memory-eficient and highly parallelizable approach to LLM post-training. Despite these eficiency advantages, the practical viability of ES remains unclear: its post-training behavior, susceptibility to catastrophic forgetting (Hoy et al., 2026; Abdi et al., 2026), and conditions for efective and scalable optimization are still poorly understood.

We compare ES with Group Relative Policy Optimization (GRPO) (Shao et al., 2024), a mainstream RL method that likewise optimizes verifier rewards from sampled responses. ES performs population-based search over parameter-perturbed policies, whereas GRPO samples multiple responses from a single policy and backpropagates a token-level objective based on their relative advantages. While GRPO efectively improves Pass@1, it can exhibit entropy collapse, leading to increasingly concentrated exploration (Cui et al., 2025; Petrenko et al., 2026; Jin et al., 2026). Consistently, prior analyses find that RL post-training can reduce reasoning-path diversity and even lower large-K Pass@K relative to the base model (Yue et al., 2025; Zhao et al., 2025; Wu et al., 2025). This suggests that GRPO may concentrate probability on a narrower set of successful reasoning modes, making low-probability correct solutions less accessible. We therefore investigate whether ES exhibits the same narrowing behavior or preserves broader reasoning coverage through population-based parameter-space exploration.

Beyond reasoning coverage, another important criterion for post-training is whether the model preserves its pretrained capabilities. Compared with supervised fine-tuning (SFT), RL-based post-training methods such as GRPO have been shown to better preserve pretrained capabilities and to be less prone to catastrophic forgetting (Jin et al., 2025; Yuan et al., 2025). This makes capability preservation an important reference point when evaluating whether ES can serve as a practical alternative to mainstream RL post-training.

For ES, however, whether such capability preservation holds remains unclear. Prior work observes substantial parameter drift alongside catastrophic forgetting under ES and attributes the latter to the former (Abdi et al., 2026). Yet whether parameter drift itself causes forgetting remains unverified, as existing evidence is limited to specific tasks and small training sets. Beyond capability preservation, deploying ES efectively also requires understanding how its hyperparameters and estimator designs afect optimization stability and scalability, particularly as model size increases. Together with the distinct exploration behavior discussed above, these open questions motivate our systematic study of ES. So, this paper summarizes our three research questions and main findings in Figure 1 and as follows:

1. RQ1: Does ES exhibit the same post-training characteristics as GRPO? We find that ES maintains broader reasoning coverage than GRPO. Across models post-trained on GSM8K and DeepScaleR, ES improves Pass@1 while achieving higher Pass@K than GRPO, without exhibiting the same entropy collapse. Theoretically, we show that verifier-projected Jensen–Shannon diversity across the ES population improves repeated-sampling success and can translate into higher Pass@K in the ES-updated policy. We further develop two sequential compositions, ES→GRPO and GRPO→ES, that combine GRPO’s strength in Pass@1 with ES’s gains in Pass@K.

2. RQ2: Does ES necessarily cause catastrophic forgetting? By examining the distribution of parameter changes, we find that the task-relevant efects of ES are concentrated in a small subset of larger-magnitude updates, while most parameter changes contribute little after perturbation cancellation. This functional sparsity suggests that substantial whole-model drift need not correspond to widespread functional change. Consistently, held-out capabilities remain largely preserved under appropriate training settings, indicating that large parameter movement alone does not imply catastrophic forgetting and that prior observations are better explained by training-set overfitting.

3. RQ3: What hyperparameter settings and estimators make ES effective and scalable? We systematically evaluate ES hyperparameters and estimator designs to identify stable and efective configurations. We find that z-score reward normalization is a key ingredient for efective ES training. Due to the discrete reward in reasoning, the two-point estimator commonly favored in zeroth-order SFT provides no advantage for ES. We further find that the population size required for efective optimization decreases as pretrained model scale increases.

## 2 Preliminaries

Let θ denote the model parameters and $\mathcal { D } _ { \mathrm { t r a i n } }$ the distribution of reasoning prompts. Given $x \sim \mathcal { D } _ { \mathrm { t r a i n } }$ , an LLM typically generates a chain-of-thought (CoT) trajectory c followed by a final answer a:

$$
c \sim \pi _ { \theta } ( \cdot \mid x ) , \qquad a \sim \pi _ { \theta } ( \cdot \mid x , c ) ,\tag{1}
$$

and we denote the complete response as $y = \left( c , a \right)$ . A verifier assigns a scalar reward $r ( x , y )$ , yielding the objective as follows:

$$
F ( \theta ) = \mathbb { E } _ { x \sim \mathcal { D } _ { \mathrm { t r a i n } } } \mathbb { E } _ { y \sim \pi _ { \theta } ( . | x ) } [ r ( x , y ) ] .\tag{2}
$$

Several post-training paradigms have been used to improve LLM reasoning capabilities, including $\mathrm { S F T }$ On-Policy Distillation (OPD), GRPO, and ES. These methods difer primarily in the supervision used to optimize the generated reasoning trajectories.

## 2.1 SFT & OPD Fine-tuning

SFT learns from expert CoT trajectories (Chu et al., 2025; Zheng and Lee, 2025), while OPD learns from token distributions of stronger teacher models (Song and Zheng, 2026). Both require supervision beyond scalar verifier rewards, so we exclude them from our main comparison.

## 2.2 GRPO Fine-tuning

GRPO instead learns directly from verifier rewards. We use GRPO as the primary gradient-based comparator to ES, as both optimize Eq. (2) but use fundamentally diferent update mechanisms.

For each reasoning prompt x, GRPO (Shao et al., 2024) samples a group of G responses $\{ y _ { i } = ( c _ { i } , a _ { i } ) \} _ { i = 1 } ^ { G }$ from the behavior policy $\pi _ { \theta _ { \mathrm { o l d } } }$ and obtains verifier rewards $r _ { i } = r ( x , y _ { i } )$ . Let $\bar { r } _ { G }$ and $s _ { G }$ denote the mean and population standard deviation of the group rewards. For $s _ { G } > 0$ , GRPO computes the response-level advantage

$$
\widehat { A } _ { i } = \frac { r _ { i } - \bar { r } _ { G } } { s _ { G } } .\tag{3}
$$

It then applies the same $\widehat { A } _ { i }$ to all tokens in $y _ { i }$ through a PPO-style clipped objective (Schulman et al., 2017). Let $T _ { i } = | y _ { i } |$ and

$$
\varrho _ { i , t } ( \theta ) = \frac { \pi _ { \theta } ( y _ { i , t } \mid x , y _ { i , < t } ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( y _ { i , t } \mid x , y _ { i , < t } ) } .
$$

The GRPO objective is

$$
\mathcal { L } _ { \mathrm { G R P O } } ( \theta ) = - \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { T _ { i } } \sum _ { t = 1 } ^ { T _ { i } } \operatorname* { m i n } \Bigl ( \varrho _ { i , t } \widehat { A } _ { i } , \mathrm { c l i p } ( \varrho _ { i , t } , 1 - \varepsilon _ { \mathrm { c l i p } } , 1 + \varepsilon _ { \mathrm { c l i p } } ) \widehat { A } _ { i } \Bigr ) .\tag{4}
$$

The group reward provides a prompt-specific baseline without a learned critic, while each response-level advantage supervises the entire CoT trajectory and final answer.

Entropy Collapse. A known issue of GRPO-style policy-gradient training is entropy collapse, where the policy distribution becomes increasingly concentrated during optimization. For a tabular softmax policy at state $s ,$ let $p _ { a } = \pi _ { \theta } ( a \mid s )$ and $A _ { a } = A ( s , a )$ , with $\mathbb { E } _ { a \sim \pi _ { \theta } } [ A _ { a } ] = 0$ . For policy entropy $\begin{array} { r } { H ( s ) = - \sum _ { a } p _ { a } \log p _ { a } } \end{array}$ prior work (Cui et al., 2025) gives the local relation

$$
\begin{array} { r } { \Delta H ( s ) \approx - \eta _ { \mathrm { P G } } \mathrm { C o v } ( \log p _ { a } , p _ { a } A _ { a } ) . } \end{array}\tag{5}
$$

When high-probability actions tend to receive positive advantages, the covariance becomes positive, yielding $\Delta H ( s ) < 0$ (Cui et al., 2025; Petrenko et al., 2026). Repeated updates can therefore reduce policy entropy and response diversity (Jin et al., 2026).

Correct Answers under Repeated Sampling. Entropy measures overall distributional concentration, but does not directly show whether repeated sampling can produce a correct response. We therefore use Pass@K, which measures whether at least one of K sampled responses is correct. Recent work shows that GRPO can reduce Pass@K relative to the Base Model, indicating a lower probability of finding a correct response through repeated sampling even when single-sample accuracy improves (Yue et al., 2025).

## 2.3 ES Optimization

ES optimizes F(θ) through parameter-space evaluations (Salimans et al., 2017). For perturbation scale $\sigma > 0$ it considers the Gaussian-smoothed objective as follows:

$$
F _ { \sigma } ( \theta ) = \mathbb { E } _ { \epsilon \sim \mathcal { N } ( 0 , I ) } [ F ( \theta + \sigma \epsilon ) ] .
$$

At each update, ES samples N perturbations $\{ \epsilon _ { i } \} _ { i = } ^ { N }$ and evaluates the corresponding perturbed models $\theta + \sigma \epsilon _ { i } ,$ obtaining rollout rewards $R _ { i }$ . The standard one-point estimator is as follows:

$$
\widehat { g } _ { \mathrm { E S } } ^ { \mathrm { r a w } } = \frac { 1 } { N \sigma } \sum _ { i = 1 } ^ { N } R _ { i } \epsilon _ { i } , \qquad \mathbb { E } [ \widehat { g } _ { \mathrm { E S } } ^ { \mathrm { r a w } } ] = \nabla F _ { \sigma } ( \theta ) .\tag{6}
$$

In practice, following Qiu et al. (2026), we standardize rewards within each population, analogous to the group normalization in Eq. (3). Let $z _ { i }$ denote the standardized reward. The ES search direction and center-model update are as follows:

$$
\widehat { d } _ { \mathrm { E S } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } z _ { i } \epsilon _ { i } , \qquad \theta ^ { + } = \theta + \alpha \widehat { d } _ { \mathrm { E S } } ,\tag{7}
$$

where $\alpha > 0$ is the update scale and absorbs the fixed $1 / \sigma$ factor.

GRPO versus ES. Table 1 highlights the key distinction: GRPO backpropagates token-level advantages through a single policy, whereas ES converts population-level reward diferences directly into a parameterspace update.

<table><tr><td>Aspect</td><td>ES</td><td>GRPO</td></tr><tr><td>Signal path</td><td> $R _ { i } \to z _ { i } \to \theta$ </td><td> $r _ { i } \xrightarrow [ ] { } \widehat { A } _ { i } \xrightarrow [ ] { } \theta$ </td></tr><tr><td>Update rule</td><td>Standardized perturbation average</td><td>Clipped policy-gradient surrogate</td></tr><tr><td>Execution</td><td>backprop</td><td>Perturbed-policy rollouts; no Policy rollouts and backprop</td></tr><tr><td>Backward state Not retained</td><td></td><td>Retained on GPU</td></tr></table>

This mechanism-level distinction motivates our comparison of posttraining behavior, while broader con-

Table 1. Update and eficiency comparison of ES and GRPO.

cerns about capability preservation and practical optimization motivate two additional questions. Accordingly, we investigate three questions about ES: (1) Does ES exhibit the same post-training characteristics as GRPO, particularly in terms of reasoning coverage? (2) Does large parameter movement under ES necessarily lead to catastrophic forgetting? (3) What hyperparameter settings and estimators make ES efective and scalable? The following sections investigate these questions in turn.

## 3 RQ1: Does ES Exhibit the Same Post-Training Characteristics as GRPO?

RQ1 examines whether ES can improve Pass@1 without degrading Pass@K at larger sampling budgets, unlike the entropy collapse and large-K Pass@K degradation that can occur under GRPO.

## 3.1 Theoretically, How ES Population Diversity Can Support Pass@K

Theoretically, we find that population diversity in ES can increase the probability that at least one of multiple sampled responses is correct, and that this advantage can transfer to the updated center policy under suitable conditions. The reason is that parameter perturbations induce heterogeneous policies with diferent success probabilities; sampling across these policies increases the chance of discovering a correct response, while reward weighting can preferentially emphasize more successful members. This subsection formalizes these efects and characterizes the conditions under which the resulting population-level advantage is preserved after the ES update.

ES Perturbations Induce Policy Diversity. We first quantify how ES parameter perturbations translate into diversity at the policy level. To characterize the diversity induced by parameter perturbations, let $s _ { \theta } ( y \mid x ) = \nabla _ { \theta } \log \pi _ { \theta } ( y \mid x )$ and $\pi _ { \theta } ^ { x } = \pi _ { \theta } ( \cdot \mid x )$ . For a local displacement $\delta ,$ , the prompt-conditioned Fisher information is

$$
\begin{array} { r l } & { \mathcal { T } _ { x } ( \theta ) = \mathbb { E } _ { Y \sim \pi _ { \theta } ( \cdot \vert x ) } \left[ s _ { \theta } ( Y \mid x ) s _ { \theta } ( Y \mid x ) ^ { \top } \right] , } \\ & { D _ { \mathrm { K L } } \left( \pi _ { \theta + \delta } ^ { x } \parallel \pi _ { \theta } ^ { x } \right) = \cfrac { 1 } { 2 } \delta ^ { \top } \mathcal { T } _ { x } ( \theta ) \delta + o ( \Vert \delta \Vert ^ { 2 } ) . } \end{array}\tag{8}
$$

For $\delta = \sigma \epsilon$ , the expected local policy displacement is proportional to $\begin{array} { r l } {  { { \frac { \sigma ^ { 2 } } { 2 } } \operatorname { t r } { \mathcal { T } } _ { x } ( \theta ) } } \end{array}$

Let D be the evaluation-prompt distribution, let $v ( x , y ) \in \{ 0 , 1 \}$ be a correctness verifier, and define the correct-response set $\mathcal { C } ( x ) = \{ y : v ( x , y ) = 1 \}$ . For any policy π, let $p _ { \pi } ( x ) = { \mathrm { P r } } _ { Y \sim \pi ( \cdot | x ) } [ Y \in { \mathcal { C } } ( x ) ]$ . Unless an expectation is shown, quantities below are conditioned on the realized population, fitness values, and resulting update. So, we provide the lemma as follows:

Lemma 1 (Perturbations induce policy diversity). For a fixed prompt x and population size $N _ { \ast }$ draw $\epsilon _ { i } \stackrel { \mathrm { i i d } } { \sim } { \mathcal { N } } ( 0 , I )$ , define $\pi _ { i } ( y \mid x ) = \pi _ { \theta + \sigma \epsilon _ { i } } ( y \mid x )$ and $\bar { \pi } _ { N } = N ^ { - 1 } \sum _ { i } \pi _ { i }$ , and let $\begin{array} { r } { \mathrm { J S } _ { N } ^ { \mathrm { p o l } } ( x ) = N ^ { - 1 } \sum _ { i } D _ { \mathrm { K L } } ( \pi _ { i } \| { \bar { \pi } } _ { N } ) } \end{array}$ The prompt-conditioned Fisher information ${ \mathcal { T } } _ { x } ( \theta )$ is defined in Equation $( 8 )$ . Assuming finite entropy and Fisher information, common local support, and suficient smoothness as detailed in Appendix $A . 1 ,$ as $\sigma \to 0$

$$
\mathbb { E } _ { \epsilon _ { 1 : N } } \bigg [ \mathrm { J S } _ { N } ^ { \mathrm { p o l } } ( x ) \bigg ] = \frac { \sigma ^ { 2 } } { 2 } \bigg ( 1 - \frac { 1 } { N } \bigg ) \operatorname { t r } \mathcal { T } _ { x } ( \theta ) + O ( \sigma ^ { 4 } ) .\tag{9}
$$

Multiple Policies in the ES Population Increase the Chance of Finding a Correct Answer. The benefit of sampling across diferent ES members is then established through comparison with matched sampling from a single policy.

Lemma 2 (Policy diversity improves correct-answer discovery). For these policies, define $p _ { i } ( x ) = p _ { \pi _ { i } } ( x )$ and $\begin{array} { r } { \bar { p } ( x ) = N ^ { - 1 } \sum _ { i } p _ { i } ( x ) } \end{array}$ . Using binary entropy $h ,$ define $\begin{array} { r } { \mathrm { J S } _ { N } ^ { \mathrm { s u c c } } ( x ) = h ( \bar { p ( x ) } ) - N ^ { - 1 } \sum _ { i } \bar { h ( p _ { i } ( x ) ) } } \end{array}$ . Data processing, followed by one independent sample per member, yields

$$
0 \leq \mathrm { J S } _ { N } ^ { \mathrm { s u c c } } ( x ) \leq \mathrm { J S } _ { N } ^ { \mathrm { p o l } } ( x ) ,\tag{10}
$$

$$
P _ { N } ^ { \mathrm { p o p } } ( x ) : = 1 - \prod _ { i = 1 } ^ { N } ( 1 - p _ { i } ( x ) ) \geq 1 - ( 1 - \bar { p } ( x ) ) ^ { N } = : P _ { N } ^ { \mathrm { s a m e } } ( x ) .\tag{11}
$$

Here $P _ { N } ^ { \mathrm { p o p } }$ and $P _ { N } ^ { \mathrm { s a m e } }$ denote one-response-per-member success and its matched single-policy baseline. Equality holds $\dot { i } \dot { f }$ and only $i f p _ { 1 } ( x ) = \cdot \cdot \cdot = p _ { N } ( x )$ , and the local gap is proportional to $\mathrm { J S } _ { N } ^ { \mathrm { s u c c } } ( x )$ as derived in Appendix A.2.

Reward Weighting Can Exploit Population Heterogeneity. The condition under which reward weighting improves the population-level success rate is then characterized.

Lemma 3 (Reward weighting improves success under positive correlation). Let $w _ { i } \geq 0 , \sum _ { i } w _ { i } = 1$ , be normalized weights obtained by a monotone transformation of realized fitness $R _ { i }$ . Define the reward-weighted policy mixture $\begin{array} { r } { \pi _ { w } = \sum _ { i } w _ { i } \pi _ { i } } \end{array}$ and its success probability $\begin{array} { r } { p _ { w } ( x ) = \sum _ { i } w _ { i } p _ { i } ( x ) } \end{array}$ . With $\operatorname { C o v } _ { i }$ taken under a uniform member draw, $\begin{array} { r } { \dot { p _ { w } } ( x ) - \bar { p } ( x ) = N \mathrm { C o v } _ { i } ( w _ { i } , p _ { i } ( x ) ) } \end{array}$

The analytical comparator $\pi _ { w }$ improves over the uniform population average exactly when the weights and member success are positively correlated. It is not the practical ES update or a comparison against $\pi _ { \theta }$

Suficient Conditions for Pass@K Improvement of the ES Center. Finally, suficient conditions are established under which the ES-updated model achieves higher Pass@K than the model before the update.

Proposition 1 (ES updates can improve Pass@K). Let $B _ { w } ( x )$ and $B _ { \theta ^ { + } } ( x )$ denote Bernoulli outcome distributions with success probabilities $p _ { w } ( x )$ and $p _ { \theta ^ { + } } ( x )$ , respectively. For a repeated-sampling budget K, let $J _ { K } ( \pi ) = \mathbb { E } _ { x \sim \mathcal { D } } [ 1 - ( \bar { 1 } - p _ { \pi } ( x ) ) ^ { K } ]$ denote expected Pass@K. Define the comparator margin $\begin{array} { r l } { m _ { K } } & { { } = } \end{array}$ $J _ { K } ( \pi _ { w } ) - J _ { K } ( \pi _ { \theta } )$ , and let $\varepsilon _ { \mathrm { s u c c } } \geq 0$ be the center-transfer error budget. Assume

$$
\begin{array} { r } { \mathbb E _ { \boldsymbol { x } \sim \mathcal { D } } { \cal D } _ { \mathrm { K L } } ( B _ { w } ( \boldsymbol { x } ) \| B _ { \theta ^ { + } } ( \boldsymbol { x } ) ) \le \varepsilon _ { \mathrm { s u c c } } , } \end{array}\tag{12}
$$

$$
m _ { K } > K \sqrt { \varepsilon _ { \mathrm { s u c c } } / 2 } .\tag{13}
$$

Then

$$
J _ { K } ( \pi _ { \theta ^ { + } } ) \geq J _ { K } ( \pi _ { w } ) - K \sqrt { \varepsilon _ { \mathrm { s u c c } } / 2 } > J _ { K } ( \pi _ { \theta } ) .\tag{14}
$$

Thus, when verifier-relevant heterogeneity improves population coverage, reward-aligned selection produces a suficient margin over the initial policy, and the center update preserves this margin; ES training can improve the model’s $\mathrm { P a s s } @ K$ . Proofs and implementation-related qualifications are provided in Appendix A.1–A.4.

![](images/65b2fe663fc64005799d3c35c9852cd3e5aeef292f59719fd17d93804492c199.jpg)  
(a) GRPO  
Qwen2.5-1.5B-Instruct

![](images/6183e80460b8d908270293b63ed1008297f2ef3154297fbfed0f84c7021b37ed.jpg)  
(b) ES  
Qwen2.5-1.5B-Instruct

![](images/726e58b3f749e667a0d3d363b1508abedaef20f27eac7f417218650a7112a630.jpg)  
(c) GSM8K → GPQA  
Llama-3.2-3B-Instruct

![](images/a0e139bcc81ee045d7f89140daac72fa6f9bccbe05c580fd32d860fd19797d03.jpg)  
(d) DeepScaleR → MATH-500  
DeepSeek-R1-Distill-Qwen-1.5B  
Figure 2 Comparison of ES and GRPO during training and testing. During GSM8K post-training, GPQA token-level entropy drops substantially under GRPO but remains largely stable under ES; GRPO finishes below the base model on Pass@16 and Pass@32, whereas ES finishes above it. On the representative task pairs, GRPO raises Pass@1 but lowers Pass@16 and Pass@32, whereas ES improves all three metrics.

## 3.2 Empirically, ES Achieves Higher Pass@K Than GRPO and Base LLM

Experimental Setup. In this part, we empirically evaluate whether ES can improve Pass@1 while preserving Pass@K under two post-training settings. The Easy Setting compares ES and GRPO on Qwen2.5- 1.5B-Instruct, Llama-3.2-3B-Instruct, and Qwen2.5-7B-Instruct after two epochs of GSM8K post-training (Cobbe et al., 2021). The Hard Setting compares them on DeepSeek-R1-Distill-Qwen-1.5B after one epoch of DeepScaleR post-training (Luo et al., 2025).

Training Dynamics. Figure 2(a–b) tracks held-out GPQA performance during GSM8K post-training of Qwen2.5-1.5B-Instruct. Under GRPO, token-level entropy declines substantially, while Pass@16 and Pass@32 finish below the base model. This pattern is consistent with prior reports linking entropy collapse under GRPO-style optimization to reduced reasoning diversity and weaker Pass@K scaling (Jang et al., 2026). In contrast, under ES, entropy changes modestly, and both metrics finish above the base model.

Test Results. The representative task pairs in Figure 2(c–d) expose the Pass@1–Pass@K diference directly. On GPQA after GSM8K post-training and MATH-500 after DeepScaleR post-training, GRPO improves Pass@1 but falls below the corresponding base model on Pass@16 and Pass@32, whereas ES improves all three metrics. The complete results in Tables 2 and 3 show that ES improves average Pass@1, Pass@16, and Pass@32 relative to the base model in both settings. Although GRPO achieves larger average Pass@1 gains, ES attains higher average Pass@16 and Pass@32 in both settings. Pass@K degradation is particularly common in the Easy Setting: GRPO falls below the corresponding base model on both Pass@16 and Pass@32 in 15 of 18 comparisons. These results indicate that ES yields gains at both K = 1 and larger K, whereas GRPO’s gains are concentrated at K = 1 and often accompanied by lower large-K performance.

Methodology: Sequentially Mixed Training of ES and GRPO ES leads in Pass@K over base and GRPO, while GRPO improve much Pass@1 than ES. So, to combine their respective advantages, we propose an intuitive method by sequentially mixing the training process of ES and GRPO. Under the same total update budget, we additionally split training equally between two stages and evaluate both sequential compositions: ES→GRPO applies ES first and GRPO second, whereas GRPO→ES reverses the order. Figure 2 presents a representative training trajectory and two representative training-to-test task pairs, while Tables 2 and 3 report the complete Pass@K results. The corresponding Maj@K results are reported in Appendix B. For each prompt, Pass@1 measures single-sample correctness, Pass@K records whether at least one of K independent responses is correct, and Maj@K reports majority-vote accuracy after answer normalization.

Pareto Trade-ofs from Sequential Compositions. Figure 3 treats Pass@1 and Pass@K as paired objectives in three representative model–task settings. In each panel, the reference Pareto front formed by Base, GRPO, and ES reflects the trade-of between GRPO’s higher Pass@1 and ES’s higher large-K coverage. Adding the endpoints from the two sequential compositions introduces additional non-dominated points. In particular, ES→GRPO attains the highest Pass@32 on the Hard-Setting math average while retaining most of GRPO’s Pass@1 gain, and GRPO→ES supplies an intermediate trade-of on GPQA with Llama-3.2-3B-Instruct.

Pass Metrics
<table><tr><td></td><td colspan="3">GSM8K</td><td colspan="3">CSQA</td><td colspan="3">HotpotQA</td><td colspan="3">Countdown</td><td colspan="3">GPQA</td><td colspan="3">MBPP</td><td colspan="3">Average</td></tr><tr><td>Method</td><td>@1</td><td>@16</td><td>@32</td><td>@1</td><td>@16</td><td>@32</td><td>@1</td><td>@16</td><td>@32</td><td>@1</td><td>@16</td><td>@32</td><td>@1</td><td>@16</td><td>@32</td><td>@1</td><td>@16</td><td>@32</td><td>@1</td><td>@16</td><td>@32</td></tr><tr><td colspan="10">Qwen2.5-1.5B-Instruct</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td>70.3</td><td>94.8</td><td>96.4</td><td>60.6</td><td>93.4</td><td>96.1</td><td>27.6</td><td>46.6</td><td>50.3</td><td>7.5</td><td>46.9</td><td>56.9</td><td>21.4</td><td>87.1</td><td>94.4</td><td>58.6</td><td>83.8</td><td>87.2</td><td>41.0</td><td>75.4</td><td></td><td>80.2</td></tr><tr><td>GRPO</td><td>78.1</td><td>93.9</td><td>95.3</td><td>63.7</td><td>93.2</td><td>95.5</td><td>25.0</td><td>44.3</td><td>48.1</td><td>8.5</td><td>49.5</td><td>58.8</td><td>22.2</td><td>87.3</td><td></td><td>96.0 60.0</td><td></td><td>82.4</td><td>86.0</td><td>42.9</td><td>75.1</td><td>79.9</td></tr><tr><td>ES</td><td>73.2</td><td>94.9</td><td>96.6</td><td>60.7</td><td>94.1</td><td>96.8</td><td>24.9</td><td>45.8</td><td>49.9</td><td>8.8</td><td>49.4</td><td></td><td>59.5 22.9</td><td></td><td>88.5</td><td>96.5</td><td>58.7</td><td>83.0</td><td>86.4</td><td>41.5</td><td>76.0</td><td>80.9</td></tr><tr><td>ES→GRPO</td><td>77.1</td><td>94.4</td><td>96.0</td><td>62.1</td><td>93.9</td><td>96.5</td><td>24.3</td><td>44.0</td><td>47.9</td><td>8.3</td><td>47.9</td><td>57.1</td><td>22.2</td><td>88.1</td><td></td><td>96.0</td><td>59.8</td><td>83.4</td><td>86.8</td><td>42.3</td><td>75.3</td><td>80.0</td></tr><tr><td>GRPO→ES</td><td>76.1</td><td>94.6</td><td>96.1</td><td>62.5</td><td>93.7</td><td>96.0</td><td>24.6</td><td>44.7</td><td>48.8</td><td>7.8</td><td>48.4</td><td>58.0</td><td>22.8</td><td>86.6</td><td></td><td>95.5</td><td>58.5</td><td>81.6</td><td>84.8</td><td>42.1</td><td>75.0</td><td>79.9</td></tr><tr><td colspan="10">Llama-3.2-3B-Instruct</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td>77.7</td><td>96.8</td><td>98.1</td><td>67.3</td><td>92.8</td><td>95.3</td><td>26.6</td><td>44.2</td><td>47.6</td><td>7.7</td><td></td><td>47.4</td><td>56.6</td><td>23.0</td><td>82.0</td><td>90.4</td><td>62.1</td><td>80.6</td><td>83.7</td><td>44.1</td><td>74.0</td><td>78.6</td></tr><tr><td>GRPO</td><td>84.7</td><td>95.6</td><td>96.4</td><td>72.1</td><td>90.5</td><td>92.8</td><td>27.5</td><td>40.4</td><td>43.0</td><td>9.6</td><td>52.9</td><td>61.6</td><td>26.9</td><td></td><td>80.3</td><td>88.4</td><td>62.1</td><td>76.7</td><td>79.8</td><td>47.1</td><td>72.7</td><td>77.0</td></tr><tr><td>ES</td><td>81.6</td><td>96.4</td><td>97.9</td><td>69.9</td><td>93.0</td><td>95.5</td><td>29.0</td><td>45.7</td><td>48.9</td><td>9.0</td><td>51.3</td><td>60.6</td><td>25.1</td><td></td><td>86.9</td><td>95.5</td><td>61.1</td><td>81.2</td><td>84.1</td><td>45.9</td><td>75.8</td><td>80.4</td></tr><tr><td>ES→GRPO</td><td>84.1</td><td>95.0</td><td>96.2</td><td>72.0</td><td>91.3</td><td>93.7</td><td>26.1</td><td>41.0</td><td>44.1</td><td>10.3</td><td>53.5</td><td></td><td>63.4</td><td>24.4</td><td>71.4</td><td>79.3</td><td>63.5</td><td>80.4</td><td>82.1</td><td>46.7</td><td>72.1</td><td>76.5</td></tr><tr><td>GRPO →ES</td><td>84.1</td><td>95.9</td><td>96.9</td><td>71.3</td><td>91.8</td><td>94.3</td><td>27.5</td><td>43.6</td><td>47.0</td><td>8.5</td><td>49.9</td><td>59.5</td><td>26.6</td><td></td><td>84.5</td><td>92.9</td><td>61.5</td><td>77.2</td><td>80.5</td><td>46.6</td><td>73.8</td><td>78.5</td></tr><tr><td colspan="10">Qwen2.5-7B-Instruct</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td>91.8</td><td>97.6</td><td>98.0</td><td>80.0</td><td>92.9</td><td>94.3</td><td>45.4</td><td>52.4</td><td>53.6</td><td>34.1</td><td>70.3</td><td>74.8</td><td></td><td>33.3</td><td>82.2</td><td>89.4</td><td>76.1</td><td>86.4</td><td>87.9</td><td>60.1</td><td>80.3</td><td>83.0</td></tr><tr><td>GRPO</td><td>92.3</td><td>97.0</td><td>97.4</td><td>80.8</td><td>91.7</td><td>92.7</td><td>46.2</td><td>52.3</td><td>53.3 53.4</td><td>36.8</td><td>68.4 71.9</td><td>72.9 76.7</td><td>34.8 34.4</td><td>79.3</td><td>85.4</td><td>74.8</td><td></td><td>85.5</td><td>87.6</td><td>61.0</td><td>79.0</td><td>81.5</td></tr><tr><td>ES</td><td>91.0 92.1</td><td>97.6 97.6</td><td>98.1 98.0</td><td>79.0 80.2</td><td>92.6 92.4</td><td>94.3 93.7</td><td>45.1 46.0</td><td>52.1 52.1</td><td>53.1</td><td>33.0 36.0</td><td>70.7</td><td>75.5</td><td>35.5</td><td></td><td>83.6 79.9</td><td>88.9 85.9</td><td>75.1 75.2</td><td>85.7 85.1</td><td>87.6 86.0</td><td>59.6 60.8</td><td>80.6 79.7</td><td>83.1 82.0</td></tr><tr><td>ES→GRPO GRPO→ES</td><td>92.3</td><td>97.4</td><td>98.0</td><td>80.2</td><td>91.9</td><td>93.2</td><td>45.9</td><td>52.4</td><td>53.6</td><td>35.0</td><td>69.4</td><td>73.8</td><td>34.2</td><td></td><td>80.8</td><td>87.4</td><td>75.1</td><td>86.0</td><td>87.9</td><td>60.4</td><td>79.6</td><td>82.3</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 2. Pass@K results (×100) in the Easy Setting after two epochs of GSM8K post-training. Underlining marks the best Base/GRPO/ES result in each task cell; bold and shading mark the best and second-best Average cells across all five methods.

Pass Metrics
<table><tr><td></td><td colspan="3">AIME24</td><td colspan="3">AIME25</td><td colspan="3">AMC23</td><td colspan="3">MATH500</td><td colspan="3">Average</td></tr><tr><td>Method</td><td>@1</td><td>@16</td><td>@32</td><td>@1</td><td>@16</td><td>@32</td><td>@1</td><td>@16</td><td>@32</td><td>@1</td><td>@16</td><td>@32</td><td>@1</td><td>@16</td><td>@32</td></tr><tr><td>DeepSeek-R1-Distill-Qwen-1.5B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td>23.4</td><td>59.9</td><td>66.7</td><td>22.6</td><td>43.8</td><td>50.0</td><td>64.1</td><td>93.6</td><td>95.0</td><td>80.9</td><td>96.7</td><td>98.0</td><td>47.7</td><td>73.5</td><td>77.4</td></tr><tr><td>GRPO</td><td>27.6</td><td>62.2</td><td>66.7</td><td>25.4</td><td>45.8</td><td>53.3</td><td>73.9</td><td>94.7</td><td>95.0</td><td>84.8</td><td>96.1</td><td>96.8</td><td>52.9</td><td>74.7</td><td>78.0</td></tr><tr><td>ES</td><td>22.8</td><td>60.3</td><td>70.0</td><td>23.2</td><td>46.7</td><td>50.0</td><td>70.5</td><td>96.0</td><td>97.5</td><td>83.2</td><td>97.2</td><td>98.2</td><td>49.9</td><td>75.0</td><td>78.9</td></tr><tr><td>ES→GRPO</td><td>29.0</td><td>70.6</td><td>76.7</td><td>23.8</td><td>43.0</td><td>50.0</td><td>72.3</td><td>92.5</td><td>92.5</td><td>84.4</td><td>97.1</td><td>97.8</td><td>52.3</td><td>75.8</td><td>79.2</td></tr><tr><td>GRPO→ES</td><td>25.8</td><td>61.8</td><td>63.3</td><td>24.5</td><td>51.0</td><td>56.7</td><td>71.6</td><td>94.9</td><td>95.0</td><td>84.5</td><td>97.0</td><td>97.6</td><td>51.6</td><td>76.2</td><td>78.2</td></tr></table>

Table 3. Pass@K results (×100) on mathematical benchmarks in the Hard Setting. Underlining marks the best Base/GRPO/ES result in each task cell; bold and shading mark the best and second-best Average cells across all five methods.

![](images/4dc37c8ad2701f40217524d1896cb3e2fa1add1f83220dcb0a2fe517cb679e6a.jpg)  
(a) GPQA  
Llama-3.2-3B-Instruct

![](images/2bef1ff875d5836683367e765c76dcb5af9bb1982bbb9cf44e57de284a66c378.jpg)  
(b) Math Average  
DeepSeek-R1-Distill-Qwen-1.5B

![](images/e71cd349e21a2d0e4bdb5933890ef12f35dd2dfb91580a00775a76c91d8e2540.jpg)  
(c) GSM8K  
Qwen2.5-1.5B-Instruct  
Figure 3 Representative Pass@1–Pass@K Pareto fronts across models and tasks. The math average is computed equally over AIME24, AIME25, AMC23, and MATH-500; higher values are better on both axes. Gray dashed lines connect the reference Pareto fronts formed by Base, GRPO, and ES. Solid dark lines trace the full Pareto fronts after adding the endpoints from the two sequential compositions. Faded markers are dominated in the corresponding pairwise comparison. The two sequential compositions add non-dominated trade-of points in all three settings.

## Takeaways.

• Sampling across ES perturbations can increase Pass@K. ES perturbations produce population members with diferent response distributions and success rates. Sampling once from each member is at least as likely to include a correct answer as drawing the same number of responses from a single policy with the same average success rate. Under the conditions in Section 3.1, the ES update can preserve this advantage in the updated model.

• ES improves Pass@1 and attains higher Pass@K than GRPO. Across the Easy and Hard Settings, ES improves average Pass@1, Pass@16, and Pass@32 over the Base Model and exceeds GRPO on average Pass@16 and Pass@32. Representative training dynamics further show that ES largely avoids the entropy decline and large-K degradation observed under GRPO.

• The two sequential compositions add new Pass@1–Pass@K trade-offs. Under the same total update budget, adding the endpoints from ES→GRPO and GRPO→ES expands the Pareto fronts in all three representative settings and provides additional non-dominated points.

## 4 RQ2: Does ES Necessarily Cause Catastrophic Forgetting?

RQ2 examines the scale and distribution of ES-induced parameter changes, identifies which updates account for performance gains, and evaluates whether large parameter drift necessarily leads to catastrophic forgetting.

## 4.1 ES Induces Greater Parameter Drift

ES evaluates full-parameter perturbations and can accumulate substantial whole-model parameter drift over successive updates. Prior works identify this large parameter movement as a characteristic diference between ES and gradient-based post-training (Hoy et al., 2026; Abdi et al., 2026). We first quantify this movement relative to matched GRPO training.

Following prior analyses of parameter drift of ES (Hoy et al., 2026; Abdi et al., 2026), we measure whole-mode movement using relative $L _ { 2 }$ distance,

$$
D _ { \mathrm { r e l } } ( \theta , \theta _ { 0 } ) = \frac { \| \theta - \theta _ { 0 } \| _ { 2 } } { \| \theta _ { 0 } \| _ { 2 } } .\tag{15}
$$

Here, $\theta _ { 0 }$ and θ denote the parameters before and after post-training, respectively. Table 4 shows that Full ES is 40.7–44.1 times farther from initialization than GRPO across the four models, confirming the substantially diferent parameter geometry reported in prior work. Hoy et al. (2026) explain this movement by decomposing ES dynamics into reward-aligned progress and difusion along locally flat or weakly reward-relevant directions. Along these directions, the expected optimization signal is small, but stochastic perturbation aggregation can accumulate as a high-dimensional random walk. Thus, the large total drift of ES may partly reflect accumulated movement in many directions with little efect on reward.

## 4.2 ES Effects Are Concentrated in a Small Subset of Larger-Magnitude Updates

However, relative whole-model distance aggregates changes across all parameter coordinates and does not reveal their magnitude distribution. We therefore quantify the proportions of parameter changes within and beyond a range of magnitude thresholds.

Let $\Delta \theta _ { i } = \theta _ { i } - \theta _ { 0 , i }$ <sub>i</sub> denote the change of parameter i from the Base Model. For a threshold $\tau ,$ we first quantify the fraction of nonzero updates whose magnitudes fall within (0, τ ]:

$$
s _ { \tau } = \frac { | \{ i : 0 < | \Delta \theta _ { i } | \leq \tau \} | } { | \{ i : | \Delta \theta _ { i } | > 0 \} | } ,\tag{16}
$$

The complementary fraction, $1 - s _ { \tau } ,$ , consists of updates with magnitudes greater than $\tau .$ . Table 4 shows that most changed parameter coordinates have relatively small update magnitudes, whereas larger-magnitude updates form a smaller subset. For example, at $\tau = 1 . 5 \times 1 0 ^ { - 3 }$ , which falls within the magnitude range of a single ES update, the interval (0, τ ] contains 77.6–93.0% of the nonzero updates across the four models, leaving 7.0–22.4% above the threshold.

Having established that larger-magnitude updates form a smaller subset, we next test whether removing the more numerous small-magnitude updates preserves task performance. For each $\tau ,$ we construct a magnitude-thresholded checkpoint by setting $\Delta \theta _ { i } = 0$ for coordinates satisfying $0 < | \Delta \theta _ { i } | \le \tau$ . Under this operation, $s _ { \tau }$ is the Update Sparsity, and Full ES corresponds to $s _ { \tau } = 0 .$ . Figure 4 shows that targettask Pass@1 remains broadly stable as progressively more small-magnitude updates are set to zero, with noticeable degradation emerging only at high update sparsity. Together, the update-magnitude statistics and thresholding experiments reveal magnitude sparsity with corresponding functional concentration: despite perturbing all parameters, ES produces larger-magnitude updates in a limited parameter subset, and retaining these updates preserves most of the performance gains. The retained coordinates can therefore be viewed as defining an approximately performance-preserving coordinate subspace identified after training.

<table><tr><td rowspan="2">Model</td><td rowspan="2">GRPO</td><td rowspan="2">Full ES</td><td colspan="3">Magnitude-Thresholded ES</td></tr><tr><td> $\tau = 1 . 0 \times 1 0 ^ { - 3 }$ </td><td> $\tau = 1 . 5 \times 1 0 ^ { - 3 }$ </td><td> $\tau = 2 . 0 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Qwen2.5-1.5B-Instruct</td><td>0.0475</td><td>1.933 (40.7×)</td><td>1.615 (79.11%)</td><td>1.242 (92.47%)</td><td>0.871 (97.57%)</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td>0.0954</td><td>4.185 (43.9×)</td><td>3.482 (78.08%)</td><td>2.589 (92.64%)</td><td>1.707 (97.89%)</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>0.0831</td><td>3.664 (44.1×)</td><td>3.032 (78.41%)</td><td>2.218 (93.02%)</td><td>1.427 (98.11%)</td></tr><tr><td>DeepSeek-R1-Distill-Qwen-1.5B</td><td>0.0548</td><td>2.338 (42.7×)</td><td>2.201 (62.18%)</td><td>2.017 (77.61%)</td><td>1.772 (87.29%)</td></tr></table>

Table 4. Relative whole-model $L _ { 2 }$ distance $( \times 1 0 ^ { - 2 } )$ and the distribution of nonzero ES updates across magnitude thresholds. Parentheses in the Full ES column report the distance relative to GRPO; parentheses in the magnitudethresholded columns report $s _ { \tau } ,$ the percentage of nonzero updates with magnitudes in (0, τ ]

![](images/9ab87a38deec8c28434b280a7343cf17877078dd28bc7d66c3530cf013508a74.jpg)  
Llama-3.2-3B-Instruct

![](images/53e1fa22f0cd488b26afd6e7080b46ace912cd42f2438bcb53525c7c6a32944a.jpg)  
(b) MATH-500  
DeepSeek-R1-Distill-Qwen-1.5B  
Figure 4 Target-task Pass@1 across update-sparsity levels in the Easy and Hard Settings. The dashed lines indicate previously recorded Base Model performance, which also forms the connected endpoint at 100% update sparsity.

Our Explanation for Why ES can Work on Full-Parameter LLM In our view, this post-hoc structure helps explain why full-parameter ES can remain efective in the high-dimensional parameter space of an LLM, because its performance gains do not depend equally on updates throughout the entire space. Frankle and Carbin (2018) mentioned that larger models contain more such efective structures, so ES can work on LLMs with billions of parameters.

We further compare the locations and scales of the largest ES and GRPO updates. The largest ES updates occur mainly in LayerNorm weights and attention projections. In Llama-3.2-3B-Instruct, the maximum magnitude is 0.01171875: five of the nine maximum coordinates are input-LayerNorm weights, while the remaining maxima include attention and MLP projections. In DeepSeek-R1-Distill-Qwen-1.5B, the maximum is 0.015625; 117 of its 144 maximum coordinates are LayerNorm weights and 24 are attention-projection biases. Normalization parameters further account for 72 and 80 of the 100 largest updates in Llama-3.2-3B-Instruct and DeepSeek-R1-Distill-Qwen-1.5B, respectively. By contrast, under GRPO, the largest Llama-3.2-3B-Instruct update is 0.00024414 (48× smaller) and its top 100 all occur in token embeddings; the DeepSeek-R1-Distill-Qwen-1.5B GRPO endpoint peaks at 0.00097656 (16× smaller), with its top 100 all in the language-model head. This contrast suggests that ES may adapt through normalization and attention parameters: LayerNorm rescales hidden states, while attention projections route information. GRPO instead concentrates its largest changes in token embeddings and the language-model head, adjacent to input representations and output logits, consistent with token-level gradient optimization.

Pass Metrics
<table><tr><td></td><td colspan="3">GPQA</td><td colspan="3">MBPP</td><td colspan="3">CSQA</td><td colspan="3">Countdown</td><td colspan="3">Average</td></tr><tr><td>Method</td><td>@1</td><td>@16</td><td>@32</td><td>@1</td><td>@16</td><td>@32</td><td>@1</td><td>@16</td><td>@32</td><td>@1</td><td>@16</td><td>@32</td><td>@1</td><td>@16</td><td>@32</td></tr><tr><td colspan="10">DeepSeek-R1-Distill-Qwen-1.5B</td><td colspan="3"></td><td colspan="3"></td></tr><tr><td>Base</td><td>24.2</td><td>80.8</td><td>91.4</td><td>62.9</td><td>85.6</td><td>87.6</td><td>46.8</td><td>89.2</td><td>93.7</td><td>37.5</td><td>79.9</td><td>85.7</td><td>42.9</td><td>83.9</td><td>89.6</td></tr><tr><td>GRPO</td><td>31.8</td><td>86.4</td><td>91.9</td><td>66.6</td><td>87.0</td><td>88.3</td><td>47.5</td><td>88.9</td><td>93.6</td><td>36.6</td><td>77.5</td><td>82.2</td><td>45.6</td><td>85.0</td><td>89.0</td></tr><tr><td>ES</td><td>28.0</td><td>84.6</td><td>92.4</td><td>63.5</td><td>85.9</td><td>87.2</td><td>46.1</td><td>88.9</td><td>93.4</td><td>38.5</td><td>79.4</td><td>84.4</td><td>44.0</td><td>84.7</td><td>89.3</td></tr><tr><td>ES→GRPO</td><td>30.6</td><td>84.5</td><td>91.4</td><td>64.6</td><td>86.7</td><td>87.9</td><td>46.8</td><td>88.8</td><td>93.3</td><td>38.8</td><td>80.6</td><td>86.8</td><td>45.2</td><td>85.1</td><td>89.9</td></tr><tr><td>GRPO→ES</td><td>30.6</td><td>86.7</td><td>93.4</td><td>64.2</td><td>85.2</td><td>87.2</td><td>47.3</td><td>88.7</td><td>93.2</td><td>38.1</td><td>79.9</td><td>85.3</td><td>45.1</td><td>85.1</td><td>89.8</td></tr></table>

Table 5. Pass@K results (×100) on four held-out benchmarks in the one-epoch Hard Setting. Underlining marks the best Base/GRPO/ES result in each task cell; bold and shading mark the best and second-best Average cells across all five methods.

## 4.3 Large Parameter Drift Does Not Necessarily Indicate Broad Forgetting

We next test whether the larger parameter movement of ES leads to held-out performance degradation. Tables 2 and 5 report held-out-task performance for the evaluated models in the Easy and Hard Settings, respectively. Table 10 in Appendix C additionally reports metric-wise performance changes for each Easy Setting model.

For all three Easy Setting models, the Pass@32 change averaged over the five held-out tasks is positive under ES but negative under GRPO (Table 10). In the Hard Setting, both methods improve average Pass@1 and Pass@16 on the held-out non-mathematical benchmarks, while ES achieves higher average Pass@32 than GRPO (Table 5). The corresponding majority-vote results show that ES exceeds both the Base Model and GRPO in average Maj@16 and Maj@32 (Table 9).

Taken together, these results suggest that ES generally maintains held-out performance despite large parameter drift. This finding difers from Abdi et al. (2026), who report catastrophic forgetting under ES. Their evidence was limited to a small training set and a single model, training task, and held-out benchmark, making broad capability forgetting dificult to distinguish from overfitting to the training set.

However, performance drops on some tasks are especially consequential in practically relevant continuallearning scenarios, where deployed models must repeatedly adapt to new tasks and knowledge while reliably preserving prior capabilities. Prior exploratory work suggests that the efects of ES on prior capabilities may depend on the task and training configuration (Hoy et al., 2026). This raises concern that ES may preserve capabilities inconsistently across tasks and repeated updates.

Takeaways.

• ES induces greater whole-model parameter drift than GRPO. ES-trained models move substantially farther from their initial parameters than their GRPO-trained counterparts.

• ES updates exhibit magnitude sparsity. Larger-magnitude updates occupy a small subset of parameters, while ignoring smaller updates largely preserves task performance. These larger updates occur mainly in normalization and attention parameters.

• Large parameter drift does not necessarily lead to catastrophic forgetting. ES generally preserves held-out performance and performs better than GRPO across the Easy and Hard Settings.

5 RQ3: What Parameter Settings and Estimators Make ES Effective and Scalable? RQ3 examines how reward normalization, perturbation scale, population size, and estimator choice afect efective and scalable ES training.

5.1 Reward Normalization, Perturbation Scale, and Population-Size Scaling
<table><tr><td rowspan="2">Update</td><td colspan="4">Qwen2.5-0.5B-Instruct</td><td colspan="4">Qwen2.5-1.5B-Instruct</td><td colspan="4">Qwen2.5-3B-Instruct</td></tr><tr><td>N = 8</td><td>N = 16</td><td>N = 32</td><td>N J = 64 (ref.)</td><td>N = 8</td><td>N = 16</td><td>N = 32</td><td>N = 64 (ref.)</td><td>N = 8</td><td>N = 16</td><td>N = 32</td><td>N = 64 (ref.)</td></tr><tr><td>100</td><td>0.4782</td><td>0.5021</td><td>0.5100</td><td>0.5085</td><td>0.8211</td><td>0.8250</td><td>0.8259</td><td>0.8244</td><td>0.8912</td><td>0.8945</td><td>0.8960</td><td>0.8957</td></tr><tr><td>200</td><td>0.4614</td><td>0.4995</td><td>0.5180</td><td>0.5201</td><td>0.8242</td><td>0.8314</td><td>0.8366</td><td>0.8352</td><td>0.8903</td><td>0.8970</td><td>0.8966</td><td>0.8992</td></tr><tr><td>300</td><td>0.4421</td><td>0.4935</td><td>0.5253</td><td>0.5287</td><td>0.8276</td><td>0.8387</td><td>0.8449</td><td>0.8438</td><td>0.8901</td><td>0.9001</td><td>0.9004</td><td>0.9031</td></tr></table>

Table 6. Smoothed GSM8K rewards across model and population sizes. Shading marks reduced-population results within 0.01 of the N = 64 reference; values use debiased exponential smoothing with weight 0.99.

Reward Normalization. We z-score population rewards before using them to weight perturbation directions, making each update depend on relative rather than absolute reward scales. In the matched single-point ES ablation, z-score normalization yields higher mean rewards than no normalization throughout the evaluated updates after initialization, indicating improved reward-guided optimization in this setting (Figure 1(c), item (1)).

Perturbation Scale. The perturbation scale σ sets both the radius of parameter-space exploration and the smoothing level of the efective objective $F _ { \sigma }$ . When σ is too small, ES explores a narrow neighborhood and may overfit the observed rewards, causing the search to become trapped around a local optimum. When σ is too large, perturbations span an overly broad region, leading to excessive exploration that can destabilize training and collapse performance. During training, σ should therefore be selected to limit early reward overfitting while maintaining stable reward-guided progress.

Population-Size Scaling. Section 4.2 shows that ES can preserve task performance with a small subset of larger-magnitude updates. Larger models may contain more such subsets with similar efects, making task-improving perturbations denser around pretrained weights, consistent with Frankle and Carbin (2018); Gan and Isola (2026). We therefore compare matched GSM8K runs of Qwen2.5-{0.5, 1.5, 3}B-Instruct with $N \in \{ 8 , 1 6 , 3 2 , 6 4 \}$ to test whether fewer directions sufice at larger scales. At update 300, Table 6 shows that only $N = 3 2$ is within 0.01 of N = 64 for 0.5B, while both N = 16 and $N = 3 2$ meet this threshold for 1.5B and 3B; correspondingly, the $N = 1 6 – \mathrm { t o } – N = 6 4$ gap falls from 0.0352 to 0.0051 and 0.0030. Together with the trajectories in Appendix F, these results indicate that smaller populations approach the N = 64 reward as model scale grows. Accordingly, the population required for stable ES training appears to decrease with model scale.

## 5.2 Why Two-Point ZO Intuition Does Not Directly Transfer to ES for Reasoning

Zeroth-order (ZO) optimization is a popular gradient-free approach that typically estimates gradients by subtracting objective values at two symmetric perturbations and is commonly evaluated on non-reasoning NLP tasks such as SST-2 (Malladi et al., 2023). ES can instead use one evaluation per direction, as in our one-point implementation, or the same antithetic two-point estimator; under matched objectives and perturbations, antithetic ES is algebraically identical to two-point ZO. The benefit of subtraction depends on evaluation: supervised tasks can reuse fixed data, whereas reasoning tasks regenerate autoregressive responses whose early-token divergence can weaken paired covariance. In our matched GSM8K run, two-point ES provides no training-reward or held-out advantage (Figure 1(c), item (3)), while Appendix E finds raw variance reduction for SST-2 but not reliably for regenerated GSM8K rewards. Thus, two-point ZO gains on non-reasoning tasks do not necessarily transfer to ES for reasoning; we favor one-point estimation when coupling is weak.

Takeaways.

• Z-score normalization yields higher training rewards. In the matched one-point ES ablation, z-score normalization consistently outperforms no normalization after initialization.

• Larger models can use smaller ES populations. As model scale increases, reduced-population runs more closely approach the N = 64 reference. At update 300, N = 16 is within 0.01 of N = 64 for the 1.5B and 3B models, whereas only N = 32 meets this criterion for the 0.5B model.

• Two-point estimation provides no advantage over one-point estimation. In the matched GSM8K experiment, two-point ES improves neither training reward nor held-out performance.

## 6 Conclusion and Future Work

Conclusion. This work examines ES’s reasoning ability gains, possible forgetting as a side efect, and conditions for efective training for LLMs. Theoretically and empirically, we demonstrate that ES improves Pass@1 while maintaining broader Pass@K coverage than GRPO, which can exhibit entropy collapse alongside a decline in Pass@K. Held-out evaluations further indicate that large parameter movement does not necessarily erase existing capabilities, in contrast to previous reports. Our design study identifies efective parameter and estimator choices and finds that larger models can train stably with smaller populations. Together, these results position ES as a distinct reasoning post-training paradigm rather than a memory-eficient GRPO alternative.

Future Work. Future work should better exploit ES’s incentivized-reasoning advantage by expanding access to correct paths assigned low probability by the base policy. Continual learning should be further explored to clarify how parameter drift afects previously acquired capabilities over longer training horizons spanning multiple tasks, and to investigate whether approaches such as parameter-eficient updates can suppress harmful drift without constraining beneficial exploration.

## References

Immanuel Abdi, Akshat Gupta, Micah Mok, Alex Lu, Nicholas Lee, and Gopala Anumanchipalli. Evolutionary strategies at scale lead to catastrophic forgetting. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 194–204, 2026. doi: 10.18653/v1/2026.acl-short.18. https://aclanthology.org/2026.acl-short.18/.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021. doi: 10.48550/arXiv.2108.07732. https://arxiv.org/abs/2108.07732.

Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V Le, Sergey Levine, and Yi Ma. Sft memorizes, rl generalizes: A comparative study of foundation model post-training. arXiv preprint arXiv:2501.17161, 2025.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021. doi: 10.48550/arXiv.2110.14168. https: //arxiv.org/abs/2110.14168.

Thomas M. Cover and Joy A. Thomas. Elements of Information Theory. Wiley-Interscience, 2 edition, 2006. doi: 10.1002/047174882X. https://doi.org/10.1002/047174882X.

Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, Zhiyuan Liu, Hao Peng, Lei Bai, Wanli Ouyang, Yu Cheng, Bowen Zhou, and Ning Ding. The entropy mechanism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617, 2025. doi: 10.48550/arXiv.2505.22617. https://arxiv.org/abs/2505.22617.

Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. arXiv preprint arXiv:1803.03635, 2018.

Yulu Gan and Phillip Isola. Neural thickets: Diverse task experts are dense around pretrained weights. In Proceedings of the 43rd International Conference on Machine Learning, 2026. https://openreview.net/forum?id=92oF5bU4cU.

Tanmay Gautam, Youngsuk Park, Hao Zhou, Parameswaran Raman, and Wooseok Ha. Variance-reduced zerothorder methods for fine-tuning language models. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 15180–15208. PMLR, 2024. https: //proceedings.mlr.press/v235/gautam24a.html.

Bingxiang He, Zekai Qu, Zeyuan Liu, Yinghao Chen, Yuxin Zuo, Cheng Qian, Kaiyan Zhang, Weize Chen, Chaojun Xiao, Ganqu Cui, Ning Ding, and Zhiyuan Liu. JustRL: Scaling a 1.5b LLM with a simple RL recipe. arXiv preprint arXiv:2512.16649, 2025. doi: 10.48550/arXiv.2512.16649. https://arxiv.org/abs/2512.16649.

William Hoy, Binxu Wang, and Xu Pan. Matching accuracy, diferent geometry: Evolution strategies vs GRPO in LLM post-training. arXiv preprint arXiv:2604.01499, 2026. doi: 10.48550/arXiv.2604.01499. https://arxiv.org/ abs/2604.01499.

HuggingFaceTB. Countdown-task-gold. Hugging Face dataset, 2025. https://huggingface.co/datasets/ HuggingFaceTB/Countdown-Task-GOLD.

Jaeeun Jang, Hansle Lee, and Sangmin Kim. A few bad apples spoil the bunch: Preventing global entropy collapse driven by a small set of tokens in LLM reasoning. In Findings of the Association for Computational Linguistics: ACL 2026, pages 13134–13154. Association for Computational Linguistics, 2026. doi: 10.18653/v1/2026.findings-acl.641. https://aclanthology.org/2026.findings-acl.641/.

Hangzhan Jin, Sitao Luan, Tianwei Ni, Sicheng Lyu, Guillaume Rabusseau, Reihaneh Rabbany, Doina Precup, and Mohammad Hamdaqa. Rl fine-tuning heals ood forgetting in sft. arXiv preprint arXiv:2509.12235, 2025.

Renren Jin, Pengzhi Gao, Yuqi Ren, Zhuowen Han, Tongxuan Zhang, Wuwei Huang, Wei Liu, Jian Luan, and Deyi Xiong. Revisiting entropy in reinforcement learning for large reasoning models. In Findings of the Association

for Computational Linguistics: ACL 2026, pages 25300–25322, 2026. doi: 10.18653/v1/2026.findings-acl.1266. https://aclanthology.org/2026.findings-acl.1266/.

Daria Korotyshova, Boris Shaposhnikov, Alexey Malakhov, Alexey Khokhulin, Nikita Surnachev, Kirill Ovcharenko, George Bredis, Alexey Gorbatovski, Viacheslav Sinii, and Daniil Gavrilov. ESSA: Evolutionary strategies for scalable alignment. arXiv preprint arXiv:2507.04453, 2025. https://arxiv.org/abs/2507.04453.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023. doi: 10.48550 arXiv.2305.20050. https://arxiv.org/abs/2305.20050.

Jianhua Lin. Divergence measures based on the shannon entropy. IEEE Transactions on Information Theory, 37(1): 145–151, 1991. doi: 10.1109/18.61115. https://doi.org/10.1109/18.61115.

Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, and Jefrey Luo. DeepScaleR: Surpassing o1-preview with a 1.5b model by scaling RL. Notion Blog, 2025. https://pretty-radio-b75.notion.site/ DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2.

Sadhika Malladi, Tianyu Gao, Eshaan Nichani, Alex Damian, Jason D. Lee, Danqi Chen, and Sanjeev Arora. Finetuning language models with just forward passes. In Advances in Neural Information Processing Systems, volume 36, pages 53038–53075, 2023. doi: 10.52202/075280-2308. https://proceedings.neurips.cc/paper\_files/paper/2023 hash/a627810151be4d13f907ac898f7e948-Abstract-Conference.html.

math-ai. AMC 2023. Hugging Face dataset, 2025. https://huggingface.co/datasets/math-ai/amc23.

Yurii E. Nesterov and Vladimir G. Spokoiny. Random gradient-free minimization of convex functions. Foundations of Computational Mathematics, 17(2):527–566, 2017. doi: 10.1007/s10208-015-9296-2. https://doi.org/10.1007/ s10208-015-9296-2.

Aleksei Petrenko, Ben Lipkin, Kevin Chen, Erik Wijmans, Marco Francis Cusumano-Towner, Raja Giryes, and Philipp Krähenbühl. Entropy-preserving reinforcement learning. In The Fourteenth International Conference on Learning Representations, 2026. https://iclr.cc/virtual/2026/poster/10010707.

Xin Qiu, Yulu Gan, Conor F. Hayes, Qiyao Liang, Yinggan Xu, Roberto Dailey, Elliot Meyerson, Babak Hodjat, and Risto Miikkulainen. Evolution strategies at scale: LLM fine-tuning beyond reinforcement learning. In Forty-third International Conference on Machine Learning, 2026. https://icml.cc/virtual/2026/poster/62279.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A graduate-level google-proof Q&A benchmark. arXiv preprint arXiv:2311.12022, 2023. doi: 10.48550/arXiv.2311.12022. https://arxiv.org/abs/2311.12022.

Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, and Ilya Sutskever. Evolution strategies as a scalable alternative to reinforcement learning. arXiv preprint arXiv:1703.03864, 2017. doi: 10.48550/arXiv.1703.03864. https://arxiv.org/abs/1703.03864.

Bidipta Sarkar, Mattie Fellows, Juan Agustin Duque, Alistair Letcher, Antonio León Villares, Anya Sims, Clarisse Wibault, Dmitry Samsonov, Dylan Cope, Jarek Liesen, Kang Li, Lukas Seier, Theo Wolf, Uljad Berdica, Valentin Mohl, Alexander David Goldie, Aaron Courville, Karin Sevegnani, Shimon Whiteson, and Jakob Nicolaus Foerster. Evolution strategies at the hyperscale. In Proceedings of the 43rd International Conference on Machine Learning, 2026. https://openreview.net/forum?id=bfVJ4GsHrO.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. doi: 10.48550/arXiv.1707.06347. https://arxiv.org/abs/1707. 06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. doi: 10.48550/arXiv.2402.03300. https://arxiv.org/abs/2402.03300.

Mingyang Song and Mao Zheng. A survey of on-policy distillation for large language models. arXiv preprint arXiv:2604.00626, 2026.

Zhishen Sun, Sizhe Dang, Guang Dai, and Haishan Ye. ESSAM: A novel competitive evolution strategies approach to reinforcement learning for memory eficient LLMs fine-tuning. arXiv preprint arXiv:2602.01003, 2026. https: //arxiv.org/abs/2602.01003.

Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. CommonsenseQA: A question answering challenge targeting commonsense knowledge. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4149–4158. Association for Computational Linguistics, 2019. doi: 10.18653/v1/N19-1421. https://aclanthology.org/N19-1421/.

Fang Wu, Weihao Xuan, Ximing Lu, Zaid Harchaoui, and Yejin Choi. The invisible leash: Why RLVR may not escape its origin. In The 2nd AI for MATH Workshop at the 42nd International Conference on Machine Learning, 2025. https://openreview.net/forum?id=KXtLWJAzgh.

Huimin Xu, Shuai Zhao, Xiaobao Wu, and Anh Tuan Luu. Understanding and preventing entropy collapse in RLVR with on-policy entropy flow optimization. In Findings of the Association for Computational Linguistics: ACL 2026, pages 17759–17771. Association for Computational Linguistics, 2026. doi: 10.18653/v1/2026.findings-acl.879. https://aclanthology.org/2026.findings-acl.879/.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2369–2380. Association for Computational Linguistics, 2018. doi: 10.18653/v1/D18-1259. https://aclanthology.org/D18-1259/.

Xiangchi Yuan, Xiang Chen, Tong Yu, Dachuan Shi, Can Jin, Wenke Lee, and Saayan Mitra. Mitigating forgetting between supervised and reinforcement learning yields stronger reasoners. arXiv preprint arXiv:2510.04454, 2025.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model? In Advances in Neural Information Processing Systems, volume 38, pages 57654–57689, 2025. https://proceedings.neurips.cc/paper\_files/paper/2025/ hash/537d5aa768c2d534016a4d06f87bc8fb-Abstract-Conference.html.

Yifan Zhang and Math-AI Team. American invitational mathematics examination (AIME) 2024. Hugging Face dataset, 2024. https://huggingface.co/datasets/math-ai/aime24.

Yifan Zhang and Math-AI Team. American invitational mathematics examination (AIME) 2025. Hugging Face dataset, 2025. https://huggingface.co/datasets/math-ai/aime25.

Yihua Zhang, Pingzhi Li, Junyuan Hong, Jiaxiang Li, Yimeng Zhang, Wenqing Zheng, Pin-Yu Chen, Jason D. Lee, Wotao Yin, Mingyi Hong, Zhangyang Wang, Sijia Liu, and Tianlong Chen. Revisiting zeroth-order optimization for memory-eficient LLM fine-tuning: A benchmark. In Proceedings of the 41st International Conference on Machine Learning, pages 59173–59190, 2024. https://proceedings.mlr.press/v235/zhang24ad.html.

Rosie Zhao, Alexandru Meterez, Sham M. Kakade, Cengiz Pehlevan, Samy Jelassi, and Eran Malach. Echo chamber: RL post-training amplifies behaviors learned in pretraining. In Proceedings of the 2nd Conference on Language Modeling, 2025. https://openreview.net/forum?id=dp4KWuSDzj.

Zhi Zheng and Wee Sun Lee. Reasoning-cv: Fine-tuning powerful reasoning llms for knowledge-assisted claim verification. arXiv preprint arXiv:2505.12348, 2025.

Zhi Zheng, Rongsheng Chen, Yunpeng Ba, Zhenkun Wang, Yee Whye Teh, and Wee Sun Lee. Agentic esopt: Fine-tuning long-horizon llm agents with minimal gpu requirements. arXiv preprint arXiv:2608.17310, 2026a.

Zhi Zheng, Yu Gu, Wei Liu, Yee Whye Teh, and Wee Sun Lee. SofT-GRPO: Surpassing discrete-token LLM reinforcement learning via gumbel-reparameterized soft-thinking policy optimization, 2026b. https://arxiv.org abs/2511.06411.

## A Proofs for the RQ1 Diversity–Coverage Analysis

This appendix provides the auxiliary identities, qualifications, and proofs behind Lemmas 1–3 and Proposition 1. The Jensen–Shannon (JS) definitions and entropy representation follow the standard formulation of Lin (1991). We use the usual information-theoretic forms of the data-processing and Pinsker inequalities (Cover and Thomas, 2006); the arithmetic–geometric mean, Taylor, and Jensen inequalities are invoked explicitly at the steps where they are used.

## A.1 Proof of Lemma 1

Finite response-distribution entropies imply this identity:

$$
H ( \bar { \pi } _ { N } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } H ( \pi _ { i } ) + \mathrm { J S } _ { N } ^ { \mathrm { p o l } } ( x ) .\tag{17}
$$

Thus, $\mathrm { J S } _ { N } ^ { \mathrm { p o l } }$ measures diversity across population members rather than sampling entropy within one policy. For a GRPO rollout group drawn from one behavior policy, all member policies are identical and the corresponding cross-parameter JS is zero, although within-policy sampling entropy can remain nonzero.

Take N and the parameter dimension to be fixed and finite, and let ${ \mathcal { \partial } } _ { x }$ be the countable response space for prompt x. Write $\begin{array} { r } { q _ { \Delta } ( y ) = { \pi } _ { \theta + \Delta } ( y \mid x ) , q _ { 0 } = q _ { \Delta = 0 } , s _ { \theta } = s _ { \theta } ( y \mid x ) , \bar { \Delta } = N ^ { - 1 } \sum _ { i } \Delta _ { i } , \bar { q } = N ^ { - 1 } \sum _ { i } q _ { \Delta _ { i } = 0 } f _ { i } , } \end{array}$ , and $\| \Delta _ { 1 : N } \| = \operatorname* { m a x } _ { i } \| \Delta _ { i } \|$ . For some $r > 0 .$ , assume that $q _ { \Delta }$ has common positive support and finite entropy on $\{ \| \Delta \| \leq r \}$ , is four times continuously diferentiable in $\Delta$ , and has finite Fisher information at $\Delta = 0$ . Define the per-response JS integrand $\begin{array} { r } { \ell _ { y } ( \Delta _ { 1 : N } ) = N ^ { - 1 } \sum _ { i } q _ { \Delta _ { i } } ( y ) \log [ q _ { \Delta _ { i } } ( y ) / \bar { q } ( y ) ] } \end{array}$ . For every multi-index $| \nu | \leq 4$ assume $\begin{array} { r } { \operatorname* { s u p } _ { \| \Delta _ { 1 : N } \| \le r } | \partial ^ { \nu } \ell _ { y } ( \Delta _ { 1 : N } ) | \le M _ { \nu } ( y ) } \end{array}$ for an envelope satisfying $\begin{array} { r } { \sum _ { y \in \mathcal { V } _ { x } } M _ { \nu } ( y ) < \infty } \end{array}$ . These conditions justify termwise diferentiation of the response sum and a uniform fourth-order Taylor remainder. For local ofsets $\Delta _ { 1 } , \ldots , \Delta _ { N }$ , Taylor expansion gives $q _ { \Delta } = q _ { 0 } [ 1 + s _ { \theta } ^ { \top } \Delta ] + O ( \| \Delta \| ^ { 2 } )$ and $\bar { q } = q _ { 0 } [ 1 + s _ { \theta } ^ { \top } \bar { \Delta } ] + O ( \| \Delta _ { 1 : N } \| ^ { 2 } )$ Substituting into the KL divergence and using $\mathbb { E } _ { q _ { 0 } } [ s _ { \theta } ] = 0$ and $\mathbb { E } _ { q _ { 0 } } [ s _ { \theta } s _ { \theta } ^ { \top } ] = \mathbb { Z } _ { x } ( \theta )$ yields the quadratic term below. Retaining the third- and fourth-order terms gives

$$
\begin{array} { r l } & { \mathrm { J S } _ { N } ^ { \mathrm { p o l } } ( x ) = \displaystyle \frac { 1 } { 2 N } \sum _ { i = 1 } ^ { N } ( \Delta _ { i } - \bar { \Delta } ) ^ { \top } \mathcal { T } _ { x } ( \theta ) ( \Delta _ { i } - \bar { \Delta } ) } \\ & { ~ + ~ \mathcal { T } _ { 3 } ( \Delta _ { 1 : N } ) + \mathcal { R } _ { 4 } ( \Delta _ { 1 : N } ) , } \end{array}\tag{18}
$$

where $\tau _ { 3 }$ is homogeneous of degree three and $\begin{array} { r } { | \mathcal { R } _ { 4 } ( \Delta _ { 1 : N } ) | \leq C \| \Delta _ { 1 : N } \| ^ { 4 } } \end{array}$ whenever $\| \Delta _ { 1 : N } \| \le r$ . For $\Delta _ { i } = \sigma \epsilon _ { i }$ the covariance becomes

$$
\mathbb { E } \left[ ( \Delta _ { i } - { \bar { \Delta } } ) ( \Delta _ { i } - { \bar { \Delta } } ) ^ { \top } \right] = \sigma ^ { 2 } \left( 1 - { \frac { 1 } { N } } \right) I .\tag{19}
$$

Therefore, the expected quadratic term in Equation (18) is

$$
\frac { \sigma ^ { 2 } } { 2 } \left( 1 - \frac { 1 } { N } \right) \mathrm { t r } \mathcal { T } _ { x } ( \theta ) .\tag{20}
$$

Let $E _ { \sigma } = \{ \operatorname* { m a x } _ { i } \lVert \sigma \epsilon _ { i } \rVert \ \leq \ r \}$ . It is sign invariant, so $\mathbb { E } [ \mathcal { T } _ { 3 } ( \sigma \epsilon _ { 1 : N } ) \mathbf { 1 } _ { E _ { \sigma } } ] = 0$ . The remainder on $E _ { \sigma }$ has expectation $O ( \sigma ^ { 4 } )$ . Moreover, $\mathrm { J S } _ { N } ^ { \mathrm { p o l } } \leq \log N$ and $\operatorname* { P r } ( E _ { \sigma } ^ { c } ) = O ( e ^ { - c / \sigma ^ { 2 } } )$ , so the JS tail and quadratic truncation are $o ( \sigma ^ { 4 } )$ , establishing the expansion in Equation (9).

## A.2 Proof of Lemma 2

The correctness verifier deterministically maps each response distribution $\pi _ { i } ( \cdot \mid x )$ to $B _ { i } ( x )$ = Bernoulli $\left( p _ { i } ( x ) \right)$ ). The data-processing inequality for generalized JS divergence therefore proves Equation (10). This projection is necessary because full policy JS may arise solely from variation among incorrect responses.

For a fixed prompt $x ,$ suppress the dependence on x and write $p _ { i } = p _ { i } ( x )$ and $\bar { p } = N ^ { - 1 } \sum _ { i } p _ { i }$ . By the arithmetic–geometric mean inequality,

$$
\prod _ { i = 1 } ^ { N } ( 1 - p _ { i } ) \leq \left( \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( 1 - p _ { i } ) \right) ^ { N } = ( 1 - \bar { p } ) ^ { N } .\tag{21}
$$

Subtracting both sides from one proves $P _ { N } ^ { \mathrm { p o p } } \geq P _ { N } ^ { \mathrm { s a m e } }$ . Equality in the arithmetic–geometric mean inequality holds exactly when $1 - p _ { 1 } = \cdot \cdot \cdot = 1 - p _ { N } .$ , equivalently $p _ { 1 } = \cdot \cdot \cdot = p _ { N }$

For the local relation, let $\begin{array} { r } { p _ { i } = \bar { p } + \xi _ { i } , \sum _ { i } \xi _ { i } = 0 } \end{array}$ , and $\beta = 1 - \bar { p } ;$ , where $\xi _ { i }$ is the centered member-success deviation and $\beta$ the average failure probability. Assume that $\bar { p }$ remains in a compact subset of (0, 1). At $\xi = 0$ , expand the product:

$$
\prod _ { i = 1 } ^ { N } ( \beta - \xi _ { i } ) = \beta ^ { N } - \frac { \beta ^ { N - 2 } } { 2 } \sum _ { i = 1 } ^ { N } \xi _ { i } ^ { 2 } + O ( \Vert \xi \Vert ^ { 3 } ) ,\tag{22}
$$

Subtracting from one gives the coverage diference:

$$
P _ { N } ^ { \mathrm { p o p } } - P _ { N } ^ { \mathrm { s a m e } } = \frac { \beta ^ { N - 2 } } { 2 } \sum _ { i = 1 } ^ { N } \xi _ { i } ^ { 2 } + O ( \Vert \xi \Vert ^ { 3 } ) .\tag{23}
$$

For $h ( p ) = - p \log p - ( 1 - p ) \log ( 1 - p )$ , the binary entropy satisfies $h ^ { \prime \prime } ( p ) = - 1 / [ p ( 1 - p ) \dot { }$ ]. A Taylor expansion of the success-JS definition in Lemma 2, with the linear term canceled by $\begin{array} { r } { \sum _ { i } \xi _ { i } = 0 . } \end{array}$ , yields

$$
\mathrm { J S } _ { N } ^ { \mathrm { s u c c } } = \frac { 1 } { 2 N \bar { p } ( 1 - \bar { p } ) } \sum _ { i = 1 } ^ { N } \xi _ { i } ^ { 2 } + O ( \| \xi \| ^ { 3 } ) .\tag{24}
$$

Eliminating $\textstyle \sum _ { i } \xi _ { i } ^ { 2 }$ between Equations (23) and (24) gives

$$
P _ { N } ^ { \mathrm { p o p } } ( x ) - P _ { N } ^ { \mathrm { s a m e } } ( x ) = N \bar { p } ( x ) ( 1 - \bar { p } ( x ) ) ^ { N - 1 } \mathrm { J S } _ { N } ^ { \mathrm { s u c c } } ( x ) + O ( \| \xi ( x ) \| ^ { 3 } ) .\tag{25}
$$

## A.3 Proof and Illustration of Lemma 3

Under a uniform draw of member index $i , \mathbb { E } _ { i } [ w _ { i } ] = 1 / N$ and $\mathbb { E } _ { i } [ p _ { i } ] = { \bar { p } } .$ . Hence

$$
N \mathrm { C o v } _ { i } ( w _ { i } , p _ { i } ) = \sum _ { i = 1 } ^ { N } w _ { i } p _ { i } - \bar { p } = p _ { w } - \bar { p } ,\tag{26}
$$

which proves the covariance identity in Lemma 3.

For intuition only, define the prompt-adaptive oracle gate

$$
\begin{array} { c } { { \displaystyle \widetilde w _ { i } ( x ) = \frac { \exp ( \lambda p _ { i } ( x ) ) } { \sum _ { j = 1 } ^ { N } \exp ( \lambda p _ { j } ( x ) ) } , \hfill } } \\ { { \displaystyle \widetilde { p } _ { w } ( x ) = \sum _ { i = 1 } ^ { N } \widetilde w _ { i } ( x ) p _ { i } ( x ) , \hfill } } \end{array}\tag{27}
$$

where $\lambda > 0$ . For fixed $N \geq 2$ , let $p _ { i } = \bar { p } + \xi _ { i }$ and assume that $\bar { p } ( x )$ is bounded away from zero and one. This parameterization gives the following normalized weights:

$$
\widetilde { w } _ { i } = \frac { \exp ( \lambda \xi _ { i } ) } { \sum _ { j = 1 } ^ { N } \exp ( \lambda \xi _ { j } ) } .\tag{28}
$$

Since $\textstyle \sum _ { i } \xi _ { i } = 0$ , the ratio expands as

$$
\widetilde { w } _ { i } = \frac { 1 } { N } + \frac { \lambda } { N } \xi _ { i } + O ( \Vert \xi \Vert ^ { 2 } ) .\tag{29}
$$

This expansion gives the selected-policy shift:

$$
\widetilde { p } _ { w } - \bar { p } = \sum _ { i = 1 } ^ { N } \widetilde { w } _ { i } \xi _ { i } = \frac { \lambda } { N } \sum _ { i = 1 } ^ { N } \xi _ { i } ^ { 2 } + O ( \Vert \xi \Vert ^ { 3 } ) .\tag{30}
$$

Substituting Equation (24) into Equation (30) gives

$$
\widetilde { p } _ { w } ( x ) - \bar { p } ( x ) = 2 \lambda \bar { p } ( x ) ( 1 - \bar { p } ( x ) ) \mathrm { J S } _ { N } ^ { \mathrm { s u c c } } ( x ) + O ( \| \xi ( x ) \| ^ { 3 } ) .\tag{31}
$$

This calculation illustrates how success/failure predictive JS becomes exploitable when fitness is calibrated to success.

## A.4 Proof of Proposition 1

Data processing yields the relaxed full-policy KL bound:

$$
D _ { \mathrm { K L } } ( B _ { w } \| B _ { \theta ^ { + } } ) \leq D _ { \mathrm { K L } } ( \pi _ { w } \| \pi _ { \theta ^ { + } } ) .\tag{32}
$$

It allows the center policy to use diferent responses while preserving total verifier success probability. This empirically testable condition connects the analytical comparator to the actual ES center.

Define the prompt-wise transfer divergence as

$$
D _ { x } = D _ { \mathrm { K L } } \left( B _ { w } ( x ) \Vert B _ { \theta ^ { + } } ( x ) \right) .\tag{33}
$$

For Bernoulli distributions, their total variation distance is the absolute diference between their success probabilities. Pinsker’s inequality bounds this diference:

$$
| p _ { w } ( x ) - p _ { \theta ^ { + } } ( x ) | \leq \sqrt { \frac { D _ { x } } { 2 } } .\tag{34}
$$

For the K-sample success map $f _ { K } ( p ) = 1 - ( 1 - p ) ^ { K }$ , the derivative is bounded as follows:

$$
0 \leq f _ { K } ^ { \prime } ( p ) = K ( 1 - p ) ^ { K - 1 } \leq K \qquad { \mathrm { f o r ~ } } p \in [ 0 , 1 ] ,\tag{35}
$$

Thus, $f _ { K }$ is K-Lipschitz. From Equation (34), Lipschitz continuity implies

$$
f _ { K } ( p _ { \theta ^ { + } } ( x ) ) \geq f _ { K } ( p _ { w } ( x ) ) - K \sqrt { \frac { D _ { x } } { 2 } } .\tag{36}
$$

Taking expectation over x and applying Jensen’s inequality to the concave square-root function gives

$$
J _ { K } ( \pi _ { \theta ^ { + } } ) \geq J _ { K } ( \pi _ { w } ) - K \mathbb { E } _ { x \sim \mathcal { D } } \sqrt { \frac { D _ { x } } { 2 } }\tag{37}
$$

$$
J _ { K } ( \pi _ { \theta ^ { + } } ) \geq J _ { K } ( \pi _ { w } ) - K \sqrt { \frac { \mathbb { E } _ { x \sim \mathcal { D } } [ D _ { x } ] } { 2 } }\tag{38}
$$

$$
J _ { K } ( \pi _ { \theta ^ { + } } ) \geq J _ { K } ( \pi _ { w } ) - K \sqrt { \frac { \varepsilon _ { \mathrm { s u c c } } } { 2 } } ,\tag{39}
$$

where the final inequality uses Equation (12). Combining this bound with the margin condition in Equation (13) proves the chained conclusion in Equation (14).

## B Complete Majority-Vote Results

This appendix reports the Maj@16 and Maj@32 results corresponding to the Pass@K results in Tables 2, 3, and 5. Maj@K applies deterministic plurality voting after task-specific answer normalization, as defined in Appendix G.6.
<table><tr><td></td><td colspan="2">GSM8K</td><td colspan="2">CSQA</td><td colspan="2">HotpotQA</td><td colspan="2">Countdown</td><td colspan="2">GPQA</td><td colspan="2">MBPP</td><td colspan="2">Average</td></tr><tr><td>Method</td><td>M@16</td><td>M@32</td><td>M@16</td><td>M@32</td><td>M@16</td><td>M@32</td><td>M@16</td><td>M@32</td><td>M@16</td><td>M@32</td><td>M@16</td><td>M@32</td><td>M@16</td><td>M@32</td></tr><tr><td colspan="9">Qwen2.5-1.5B-Instruct</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td>82.6</td><td>84.3</td><td>67.8</td><td>68.8</td><td>31.1</td><td>31.1</td><td>21.7</td><td>27.0</td><td>20.7</td><td>20.2</td><td>61.9</td><td>61.5</td><td>47.6</td><td>48.8</td></tr><tr><td>GRPO</td><td>84.2</td><td>84.8</td><td>70.0</td><td>71.0</td><td>28.8</td><td>29.0</td><td>27.8</td><td>31.5</td><td>22.7</td><td>22.7</td><td>60.3</td><td>61.9</td><td>49.0</td><td>50.2</td></tr><tr><td>ES</td><td>82.3</td><td>83.9</td><td>67.2</td><td>68.9</td><td>29.8</td><td>29.8</td><td>27.6</td><td>33.1</td><td>25.3</td><td>24.2</td><td>58.4</td><td>62.3</td><td>48.4</td><td>50.4</td></tr><tr><td>ES→GRPO</td><td>83.9</td><td>84.2</td><td>69.2</td><td>69.6</td><td>28.8</td><td>29.1</td><td>26.2</td><td>31.6</td><td>20.7</td><td>18.7</td><td>61.5</td><td>62.7</td><td>48.4</td><td>49.3</td></tr><tr><td>GRPO→ES</td><td>83.1</td><td>83.6</td><td>68.7</td><td>70.0</td><td>29.2</td><td>29.4</td><td>20.7</td><td>24.6</td><td>23.7</td><td>23.7</td><td>58.8</td><td>61.9</td><td>47.4</td><td>48.9</td></tr><tr><td colspan="9">Llama-3.2-3B-Instruct</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td>86.9</td><td>87.8</td><td>74.1</td><td>74.6</td><td>28.1</td><td>28.5</td><td>39.2</td><td>45.7</td><td>33.8</td><td>33.8</td><td>62.3</td><td>64.6</td><td>54.1</td><td>55.8</td></tr><tr><td>GRPO</td><td>88.9</td><td>89.5</td><td>74.3</td><td>74.6</td><td>27.4</td><td>27.7</td><td>44.6</td><td>50.8</td><td>35.4</td><td>33.8</td><td>62.3</td><td>63.0</td><td>55.5</td><td>56.6</td></tr><tr><td>ES</td><td>88.6</td><td>88.9</td><td>74.9</td><td>75.5</td><td>29.5</td><td>30.0</td><td>40.4</td><td>48.9</td><td>30.3</td><td>30.8</td><td>62.3</td><td>61.9</td><td>54.4</td><td>56.0</td></tr><tr><td>ES→GRPO</td><td>88.4</td><td>88.2</td><td>75.9</td><td>75.8</td><td>26.6</td><td>26.8</td><td>41.9</td><td>48.7</td><td>29.8</td><td>29.3</td><td>63.8</td><td>62.3</td><td>54.4</td><td>55.2</td></tr><tr><td>GRPO→ES</td><td>89.1</td><td>89.8</td><td>74.9</td><td>75.6</td><td>28.5</td><td>28.6</td><td>40.0</td><td>47.5</td><td>30.8</td><td>31.3</td><td>63.0</td><td>62.7</td><td>54.4</td><td>55.9</td></tr><tr><td colspan="9">Qwen2.5-7B-Instruct</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td>94.5</td><td>94.5</td><td>81.4</td><td>81.7</td><td>45.9</td><td>46.0</td><td>56.6</td><td>58.4</td><td>37.4</td><td>38.4</td><td>77.0</td><td>77.0</td><td>65.5</td><td>66.0</td></tr><tr><td>GRPO</td><td>93.8</td><td>93.6</td><td>81.9</td><td>81.4</td><td>46.8</td><td>46.9</td><td>56.6</td><td>57.4</td><td>34.9</td><td>37.9</td><td>75.9</td><td>75.9</td><td>65.0</td><td>65.5</td></tr><tr><td>ES</td><td>94.2</td><td>94.2</td><td>80.8</td><td>81.2</td><td>45.7</td><td>45.7</td><td>57.9</td><td>60.1</td><td>40.4</td><td>41.9</td><td>75.1</td><td>75.9</td><td>65.7</td><td>66.5</td></tr><tr><td>ES→GRPO</td><td>94.3</td><td>94.6</td><td>81.4</td><td>81.4</td><td>46.5</td><td>46.5</td><td>57.3</td><td>59.2</td><td>37.4</td><td>37.9</td><td>75.1</td><td>74.7</td><td>65.3</td><td>65.7</td></tr><tr><td>GRPO→ES</td><td>93.7</td><td>94.0</td><td>82.2</td><td>81.7</td><td>46.2</td><td>46.3</td><td>56.1</td><td>57.2</td><td>40.4</td><td>39.4</td><td>75.1</td><td>75.9</td><td>65.6</td><td>65.8</td></tr></table>

Table 7. Maj@K results (×100) in the Easy Setting after two epochs of GSM8K post-training, corresponding to the Pass@K results in Table 2. Underlining marks the best Base/GRPO/ES result in each task cell; bold and shading mark the best and second-best Average cells across all five methods.

<table><tr><td></td><td colspan="2">AIME24</td><td colspan="2">AIME25</td><td colspan="2">AMC23</td><td colspan="2">MATH500</td><td colspan="2">Average</td></tr><tr><td>Method</td><td>M@16</td><td>M@32</td><td>M@16</td><td>M@32</td><td>M@16</td><td>M@32</td><td>M@16</td><td>M@32</td><td>M@16</td><td>M@32</td></tr><tr><td colspan="9">DeepSeek-R1-Distill-Qwen-1.5B</td><td></td></tr><tr><td>Base</td><td>36.7</td><td>40.0</td><td>33.3</td><td>33.3</td><td>82.5</td><td>82.5</td><td>90.0</td><td>91.0</td><td>60.6</td><td>61.7</td></tr><tr><td>GRPO</td><td>43.3</td><td>46.7</td><td>30.0</td><td>33.3</td><td>85.0</td><td>87.5</td><td>91.2</td><td>91.8</td><td>62.4</td><td>64.8</td></tr><tr><td>ES</td><td>36.7</td><td>43.3</td><td>36.7</td><td>33.3</td><td>90.0</td><td>92.5</td><td>91.8</td><td>92.2</td><td>63.8</td><td>65.3</td></tr><tr><td>ES→GRPO</td><td>46.7</td><td>53.3</td><td>33.3</td><td>33.3</td><td>85.0</td><td>92.5</td><td>91.6</td><td>92.6</td><td>64.2</td><td>67.9</td></tr><tr><td>GRPO→ES</td><td>46.7</td><td>46.7</td><td>40.0</td><td>36.7</td><td>90.0</td><td>85.0</td><td>91.4</td><td>92.2</td><td>67.0</td><td>65.1</td></tr></table>

Table 8. Maj@K results (×100) on mathematical benchmarks in the Hard Setting, corresponding to the Pass@K results in Table 3. Underlining marks the best Base/GRPO/ES result in each task cell; bold and shading mark the best and second-best Average cells across all five methods.

<table><tr><td></td><td colspan="2">GPQA</td><td colspan="2">MBPP</td><td colspan="2">CSQA</td><td colspan="2">Countdown</td><td colspan="2">Average</td></tr><tr><td>Method</td><td>M@16</td><td>M@32</td><td>M@16</td><td>M@32</td><td>M@16</td><td>M@32</td><td>M@16</td><td>M@32</td><td>M@16</td><td>M@32</td></tr><tr><td colspan="9">DeepSeek-R1-Distill-Qwen-1.5B</td><td></td></tr><tr><td>Base</td><td>33.3</td><td>33.3</td><td>69.7</td><td>70.4</td><td>54.1</td><td>54.1</td><td>71.1</td><td>75.1</td><td>57.0</td><td>58.2</td></tr><tr><td>GRPO</td><td>35.9</td><td>35.4</td><td>69.7</td><td>70.8</td><td>53.7</td><td>52.7</td><td>68.5</td><td>72.8</td><td>56.9</td><td>57.9</td></tr><tr><td>ES</td><td>39.4</td><td>39.4</td><td>70.8</td><td>69.7</td><td>52.7</td><td>53.6</td><td>71.7</td><td>76.2</td><td>58.6</td><td>59.7</td></tr><tr><td>ES→GRPO</td><td>36.4</td><td>37.4</td><td>71.6</td><td>72.0</td><td>52.4</td><td>53.0</td><td>70.9</td><td>73.6</td><td>57.8</td><td>59.0</td></tr><tr><td>GRPO→ES</td><td>33.8</td><td>34.9</td><td>66.5</td><td>69.7</td><td>53.3</td><td>54.1</td><td>72.2</td><td>76.9</td><td>56.5</td><td>58.9</td></tr></table>

Table 9. Maj@K results (×100) on four held-out benchmarks in the one-epoch Hard Setting, corresponding to the Pass@K results in Table 5. Underlining marks the best Base/GRPO/ES result in each task cell; bold and shading mark the best and second-best Average cells across all five methods.

## C Detailed Held-Out Performance Changes

Table 10 reports the metric-wise Easy Setting results underlying the held-out summary in Figure 1(b). For each model and metric, we compute post-training minus initial performance on each held-out benchmark and then average these changes equally over the five non-GSM8K tasks.

<table><tr><td></td><td colspan="3">Pass changes</td><td colspan="2">Majority-vote changes</td></tr><tr><td>Method</td><td>∆@1</td><td>∆@16</td><td>∆@32</td><td>∆M@16</td><td>∆M@32</td></tr><tr><td colspan="6">Qwen2.5-1.5B-Instruct</td></tr><tr><td>GRPO</td><td>+0.7</td><td>-0.2</td><td>-0.1</td><td>+1.3</td><td>+1.5</td></tr><tr><td>ES</td><td>+0.0</td><td>+0.6</td><td>+0.8</td><td>+1.0</td><td>+2.0</td></tr><tr><td colspan="6">Llama-3.2-3B-Instruct</td></tr><tr><td>GRPO</td><td>+2.3</td><td>-1.3</td><td>-1.6</td><td>+1.3</td><td>+0.6</td></tr><tr><td>ES</td><td>+1.4</td><td>+2.2</td><td>+2.2</td><td>-0.0</td><td>-0.0</td></tr><tr><td colspan="6">Qwen2.5-7B-Instruct</td></tr><tr><td>GRPO</td><td>+0.9</td><td>-1.4</td><td>-1.6</td><td>-0.5</td><td>-0.4</td></tr><tr><td>ES</td><td>-0.5</td><td>+0.3</td><td>+0.2</td><td>+0.3</td><td>+0.7</td></tr></table>

Table 10. Held-out performance changes for matched Easy Setting runs, averaged equally over five non-GSM8K tasks and reported in percentage points. M@16 and M@32 denote Maj@16 and Maj@32. Positive values denote gains.

Averaged across the three models, all five ES metric changes are positive, whereas GRPO decreases Pass@16 and Pass@32.

## D Complete Results for Magnitude-Thresholded ES

This appendix reports the complete target-task evaluation results for magnitude-thresholded ES across all four models. For a threshold τ , coordinates satisfying $0 < | \Delta \theta _ { i } | \leq \tau$ are set to zero, equivalently resetting the corresponding parameters to their Base Model values, and update sparsity is computed over nonzero ES changes according to Equation (16). Coordinates with zero change are excluded from both the sparsity denominator and the ablated set. Pass@1 is the empirical single-sample accuracy estimated from 32 retained responses per problem. The $\Delta$ Full ES column reports the signed absolute diference from the corresponding unablated ES endpoint on the same percentage scale.

Table 11. Complete GSM8K Pass@1 results for magnitude-thresholded ES in the Easy Setting. Base Model values are included as absolute references.
<table><tr><td>Threshold τ</td><td>Update sparsity (%)</td><td>Pass@1 (%)</td><td>∆ Full ES (%)</td></tr><tr><td colspan="4">Qwen2.5-1.5B-Instruct</td></tr><tr><td>Full ES</td><td>0.00</td><td>73.351</td><td>+0.000</td></tr><tr><td> $2 . 5 \times 1 0 ^ { - 4 }$ </td><td>27.82</td><td>73.330</td><td>-0.021</td></tr><tr><td> $5 . 0 \times 1 0 ^ { - 4 }$ </td><td>50.77</td><td>73.102</td><td>-0.249</td></tr><tr><td> $1 . 0 \times 1 0 ^ { - 3 }$ </td><td>79.11</td><td>73.000</td><td>-0.351</td></tr><tr><td> $1 . 5 \times 1 0 ^ { - 3 }$ </td><td>92.47</td><td>72.091</td><td>-1.260</td></tr><tr><td> $2 . 0 \times 1 0 ^ { - 3 }$ </td><td>97.57</td><td>70.681</td><td>-2.670</td></tr><tr><td>Base Model</td><td>100.00</td><td>70.271</td><td></td></tr><tr><td colspan="4">Llama-3.2-3B-Instruct</td></tr><tr><td>Full ES</td><td>0.00</td><td>81.691</td><td>+0.000</td></tr><tr><td> $2 . 5 \times 1 0 ^ { - 4 }$ </td><td>25.27</td><td>81.658</td><td>-0.033</td></tr><tr><td> $5 . 0 \times 1 0 ^ { - 4 }$ </td><td>47.74</td><td>81.440</td><td>-0.251</td></tr><tr><td> $1 . 0 \times 1 0 ^ { - 3 }$ </td><td>78.08</td><td>81.560</td><td>-0.130</td></tr><tr><td> $1 . 5 \times 1 0 ^ { - 3 }$ </td><td>92.64</td><td>78.433</td><td>-3.258</td></tr><tr><td> $2 . 0 \times 1 0 ^ { - 3 }$ </td><td>97.89</td><td>77.883</td><td>-3.807</td></tr><tr><td>Base Model</td><td>100.00</td><td>77.670</td><td></td></tr><tr><td colspan="4">Qwen2.5-7B-Instruct</td></tr><tr><td>Full ES</td><td>0.00</td><td>91.016</td><td>+0.000</td></tr><tr><td> $2 . 5 \times 1 0 ^ { - 4 }$ </td><td>25.43</td><td>91.087</td><td>+0.071</td></tr><tr><td> $5 . 0 \times 1 0 ^ { - 4 }$ </td><td>47.51</td><td>91.028</td><td>+0.012</td></tr><tr><td> $1 . 0 \times 1 0 ^ { - 3 }$ </td><td>78.41</td><td>91.504</td><td>+0.488</td></tr><tr><td> $1 . 5 \times 1 0 ^ { - 3 }$ </td><td>93.02</td><td>91.736</td><td>+0.720</td></tr><tr><td> $2 . 0 \times 1 0 ^ { - 3 }$ </td><td>98.11</td><td>91.824</td><td>+0.808</td></tr><tr><td>Base Model</td><td>100.00</td><td>91.774</td><td></td></tr></table>

Table 12. Complete MATH-500 Pass@1 results for magnitude-thresholded ES with DeepSeek-R1-Distill-Qwen-1.5B in the Hard Setting. The Base Model value is included as an absolute reference.
<table><tr><td>Threshold τ</td><td>Update sparsity (%)</td><td>Pass@1 (%)</td><td>∆ Full ES (%)</td></tr><tr><td>Full ES</td><td>0.00</td><td>82.844</td><td>+0.000</td></tr><tr><td> $2 . 5 \times 1 0 ^ { - 4 }$ </td><td>17.80</td><td>83.112</td><td>+0.269</td></tr><tr><td> $5 . 0 \times 1 0 ^ { - 4 }$ </td><td>38.32</td><td>82.919</td><td>+0.075</td></tr><tr><td> $1 . 0 \times 1 0 ^ { - 3 }$ </td><td>62.18</td><td>83.075</td><td>+0.231</td></tr><tr><td> $1 . 5 \times 1 0 ^ { - 3 }$ </td><td>77.61</td><td>83.013</td><td>+0.169</td></tr><tr><td> $2 . 0 \times 1 0 ^ { - 3 }$ </td><td>87.29</td><td>82.394</td><td>-0.450</td></tr><tr><td> $2 . 5 \times 1 0 ^ { - 3 }$ </td><td>93.16</td><td>82.213</td><td>-0.631</td></tr><tr><td> $3 . 0 \times 1 0 ^ { - 3 }$ </td><td>96.64</td><td>81.388</td><td>-1.456</td></tr><tr><td>Base Model</td><td>100.00</td><td>80.856</td><td></td></tr></table>

At comparable sparsity near 78%, all four endpoints remain close to Full ES: the Pass@1 changes are −0.351, −0.130, +0.488, and +0.169 percentage points for Qwen2.5-1.5B-Instruct, Llama-3.2-3B-Instruct, Qwen2.5-7B-Instruct, and DeepSeek-R1-Distill-Qwen-1.5B, respectively. Their behavior diverges as the threshold expands farther into the update distribution. Qwen2.5-1.5B-Instruct and Llama-3.2-3B-Instruct degrade at approximately 92% sparsity, DeepSeek degrades more gradually, and Qwen2.5-7B improves across the tested range.

## E ES–ZO Estimator Equivalence and Its Finite-Sample Boundary

This appendix separates an ideal estimator identity from a finite-sample property of stochastic reasoning evaluations. The identity shows that ES and ZO can estimate the same smoothed objective; the diagnostic tests whether paired evaluations retain enough covariance to reduce variance. The Gaussian smoothing and score-function construction below are standard tools in gradient-free optimization and ES (Nesterov and Spokoiny, 2017; Salimans et al., 2017).

## E.1 A Unified Smoothed-Objective View

For a measurable scalar objective F, we use a maximization convention throughout. For reward maximization, we take $F ( \theta ) = R ( \theta )$ ; for loss minimization, we take $F ( \theta ) = - { \mathcal { L } } ( \theta )$ , so that all estimators below are written in ascent form. Assume that some neighborhood $U$ of θ satisfies s $\begin{array} { r } { \log _ { \theta ^ { \prime } \in U } \mathbb { E } _ { \epsilon } [ | F ( \theta ^ { \prime } + \sigma \epsilon ) | ( 1 + \| \epsilon \| ) ] < \infty } \end{array}$ , and define its Gaussian smoothing as

$$
F _ { \sigma } ( \theta ) = \operatorname { \mathbb { E } } _ { \epsilon \sim \mathcal { N } ( 0 , I ) } \left[ F ( \theta + \sigma \epsilon ) \right] .\tag{40}
$$

The Gaussian score-function identity gives

$$
\nabla F _ { \sigma } ( \theta ) = \frac { 1 } { \sigma } \mathbb { E } _ { \epsilon } \left[ F ( \theta + \sigma \epsilon ) \epsilon \right] .\tag{41}
$$

The resulting one-point estimator over N directions is

$$
{ \widehat { g } } ^ { 1 p } = { \frac { 1 } { N \sigma } } \sum _ { i = 1 } ^ { N } F ( \theta + \sigma \epsilon _ { i } ) \epsilon _ { i } , \qquad { \mathbb { E } } \bigl [ { \widehat { g } } ^ { 1 p } \bigr ] = \nabla F _ { \sigma } ( \theta ) .\tag{42}
$$

The multi-direction two-point estimator is

$$
\widehat { g } ^ { 2 p } = \frac { 1 } { 2 N \sigma } \sum _ { i = 1 } ^ { N } \left[ F ( \theta + \sigma \epsilon _ { i } ) - F ( \theta - \sigma \epsilon _ { i } ) \right] \epsilon _ { i } .\tag{43}
$$

Because ϵ and −ϵ have the same distribution,

$$
\mathbb { E } _ { \epsilon } \left[ F ( \theta - \sigma \epsilon ) \epsilon \right] = - \mathbb { E } _ { \epsilon } \left[ F ( \theta + \sigma \epsilon ) \epsilon \right] .\tag{44}
$$

Substitution into Equation (43) yields

$$
\mathbb { E } \big [ \widehat { g } ^ { 2 p } \big ] = \nabla F _ { \sigma } ( \theta ) = \mathbb { E } \big [ \widehat { g } ^ { 1 p } \big ] .\tag{45}
$$

If ES uses the same positive–negative perturbation pairs, its antithetic estimator is exactly Equation (43). Thus, antithetic ES and two-point ZO are the same estimator when they share the objective, perturbation distribution, scale, and raw paired-diference rule. This identity does not imply equal finite-budget variance because one pair consumes two function evaluations, whereas one one-point sample consumes one. Thus, N counts perturbation directions in both estimators, while their perturbed-evaluation counts are $N _ { \mathrm { e v a l } } = N$ and $N _ { \mathrm { e v a l } } = 2 N$ , respectively.

## E.2 Covariance Required by Paired Subtraction

Suppose an objective evaluation is stochastic because it includes prompt sampling, autoregressive generation, or verifier noise. Let $\xi$ collect this evaluation randomness and let $\widehat { F } ( \vartheta ; \xi )$ be an unbiased scalar evaluation of the objective, such that

$$
\mathbb { E } _ { \xi } \Big [ \widehat { F } ( \vartheta ; \xi ) \Big ] = F ( \vartheta )\tag{46}
$$

for every evaluated parameter point ϑ. For fixed $( \theta , \epsilon , \sigma )$ , the repeated evaluations are

$$
\begin{array} { c } { { X _ { + } = \widehat { F } ( \theta + \sigma \epsilon ; \xi _ { + } ) , } } \\ { { { } } } \\ { { X _ { 0 } = \widehat { F } ( \theta ; \xi _ { 0 } ) , } } \\ { { { } } } \\ { { X _ { - } = \widehat { F } ( \theta - \sigma \epsilon ; \xi _ { - } ) . } } \end{array}\tag{47}
$$

(a) Objective comparison at $\sigma = 1 0 ^ { - 3 }$
<table><tr><td>Objective</td><td>Evaluation coupling</td><td>Corr  $\left( X _ { + } , X _ { 0 } \right)$ </td><td> $\kappa _ { \mathrm { b a s e } }$ </td><td> $\mathrm { C o r r } ( X _ { + } , X _ { - } )$ </td><td> $\kappa _ { \mathrm { p a i r } }$ </td></tr><tr><td>GSM8K regenerated-rollout reward</td><td>Common seed</td><td>0.2777</td><td>1.4445</td><td>0.1304</td><td>1.9870</td></tr><tr><td>GSM8K regenerated-rollout reward</td><td>Independent seed</td><td>0.0561</td><td>1.3577</td><td>0.0224</td><td>2.7847</td></tr><tr><td>SST-2 supervised CE</td><td>Fixed batch</td><td>0.9776</td><td>0.0447</td><td>0.9196</td><td>0.1557</td></tr><tr><td>Mean token log-probability</td><td>Fixed rollout</td><td>0.9939</td><td>0.0125</td><td>0.9857</td><td>0.0285</td></tr><tr><td>Policy sequence loss</td><td>Fixed rollout</td><td>0.9939</td><td>0.0167</td><td>0.9903</td><td>0.0194</td></tr><tr><td>GRPO surrogate</td><td>Fixed rollout</td><td>0.9998</td><td>0.0006</td><td>0.9997</td><td>0.0007</td></tr></table>

(b) Perturbation-scale sweep for GSM8K regenerated-rollout reward
<table><tr><td>σ</td><td>Seed coupling</td><td> $\mathrm { C o r r } ( X _ { + } , X _ { 0 } )$ </td><td> $\kappa _ { \mathrm { b a s e } }$ </td><td> $\mathrm { C o r r } ( X _ { + } , X _ { - } )$ </td><td> $\kappa _ { \mathrm { p a i r } }$ </td></tr><tr><td> $1 0 ^ { - 4 }$ </td><td>Common</td><td>0.8797</td><td>0.2288</td><td>0.5537</td><td>0.8194</td></tr><tr><td> $1 0 ^ { - 4 }$ </td><td>Independent</td><td>-0.0245</td><td>1.4080</td><td>0.0450</td><td>2.3822</td></tr><tr><td> $3 \times 1 0 ^ { - 4 }$ </td><td>Common</td><td>0.5607</td><td>1.0424</td><td>0.3026</td><td>1.9434</td></tr><tr><td> $3 \times 1 0 ^ { - 4 }$ </td><td>Independent</td><td>-0.0303</td><td>1.6687</td><td>0.0532</td><td>2.4379</td></tr><tr><td> $1 0 ^ { - 3 }$ </td><td>Common</td><td>0.2777</td><td>1.4445</td><td>0.1304</td><td>1.9870</td></tr><tr><td> $1 0 ^ { - 3 }$ </td><td>Independent</td><td>0.0561</td><td>1.3577</td><td>0.0224</td><td>2.7847</td></tr></table>

Table 13. Local subtraction diagnostic at the Qwen2.5-1.5B-Instruct base checkpoint. $\kappa _ { \mathrm { b a s e } }$ and $\kappa _ { \mathrm { p a i r } }$ are the raw scalar variance ratios in Equation (51); lower is better and values below 1 indicate variance reduction. Panel (a) compares objectives at $\sigma = 1 0 ^ { - 3 }$ , and Panel (b) sweeps the scale for GSM8K regenerated-rollout reward.

The random variables $( \xi _ { + } , \xi _ { 0 } , \xi _ { - } )$ may be coupled through common random numbers or sampled independently. Their raw diference variances satisfy

$$
\operatorname { V a r } ( X _ { + } - X _ { 0 } ) = \operatorname { V a r } ( X _ { + } ) + \operatorname { V a r } ( X _ { 0 } ) - 2 \operatorname { C o v } ( X _ { + } , X _ { 0 } ) ,\tag{48}
$$

$$
\operatorname { V a r } ( X _ { + } - X _ { - } ) = \operatorname { V a r } ( X _ { + } ) + \operatorname { V a r } ( X _ { - } ) - 2 \operatorname { C o v } ( X _ { + } , X _ { - } ) .\tag{49}
$$

In particular, $\mathrm { V a r } ( X _ { + } - X _ { - } ) < \mathrm { V a r } ( X _ { + } )$ if and only if

$$
\operatorname { C o v } ( X _ { + } , X _ { - } ) > { \frac { 1 } { 2 } } \operatorname { V a r } ( X _ { - } ) .\tag{50}
$$

We report the correlations and the raw variance ratios

$$
\kappa _ { \mathrm { b a s e } } = \frac { \mathrm { V a r } ( X _ { + } - X _ { 0 } ) } { \mathrm { V a r } ( X _ { + } ) } , \qquad \kappa _ { \mathrm { p a i r } } = \frac { \mathrm { V a r } ( X _ { + } - X _ { - } ) } { \mathrm { V a r } ( X _ { + } ) } .\tag{51}
$$

A ratio below one means that subtraction reduces the variance of the scalar objective values in this diagnostic.

## E.3 Local Perturbation Diagnostic

Protocol. We conduct a local diagnostic around the Qwen2.5-1.5B-Instruct base checkpoint. For each sampled full-parameter direction, we evaluate Equation (47) at several perturbation scales. The reported statistics pool scalar observations across prompts, generation seeds, and directions.

Objectives. We compare three evaluation protocols. First, GSM8K regenerated-rollout reward regenerates a response independently at each of the positive, center, and negative parameter points and assigns a terminal correctness reward. Second, SST-2 supervised CE evaluates a diferentiable cross-entropy objective on the same fixed batch at all three parameter points. Third, GSM8K fixed rollout generates responses once at the base checkpoint, freezes them, and recomputes mean token log-probability, policy sequence loss, and a GRPO surrogate at the perturbed points. Fixed rollouts isolate rescoring from autoregressive generation noise.

Randomness Coupling. For regenerated rollouts, common uses the same generation seed at the three parameter points, whereas independent uses diferent seeds. Common seeds implement common random numbers, but they do not guarantee identical autoregressive trajectories after a parameter perturbation changes an early token distribution.

## E.4 Diagnostic Results

Table 13(a) compares the three objective classes at $\sigma = 1 0 ^ { - 3 }$ . The repeated-direction regenerated-rollout run uses 16 prompts, 16 generation seeds per prompt, and 4 perturbation directions. The fixed-rollout run uses 16 prompts, 8 seeds per prompt, and 4 directions. The supervised control contains 128 pooled scalar observations.

GSM8K regenerated-rollout rewards are weakly correlated across the positive and negative perturbations at this scale, and paired subtraction increases their raw scalar variance. The same perturbation implementation yields strong correlations and ratios below one for supervised CE and all fixed-rollout objectives. This contrast localizes the observed instability to the combination of regenerating autoregressive responses and evaluating terminal rewards rather than to paired parameter perturbations alone. The very small fixed-rollout GRPO ratio further shows that a GRPO surrogate is not intrinsically incompatible with paired ZO evaluation when the responses are held fixed.

Table 13(b) shows that coupling regenerated-rollout reward evaluations with common random numbers is scale dependent. At $\sigma = 1 0 ^ { - 4 }$ , common seeds produce $\kappa _ { \mathrm { p a i r } } = 0 . 8 1 9 4$ , whereas independent seeds remain above one. At the two larger scales, paired subtraction does not reduce raw variance under either seed mode.

## E.5 Scope of the Evidence

The diagnostic supports a conclusion: near the tested base checkpoint, regenerated-rollout correctness rewards provide substantially weaker positive–negative coupling than fixed-batch supervised or fixed-rollout objectives, so raw paired subtraction does not reliably cancel evaluation variance except at small scales.

Figure 1(c), item (3), complements the local diagnostic with a three-epoch comparison under matched perturbed evaluations and reward normalization.

## F Population-Size Scaling Trajectories

Figure 5 reports the complete per-update mean-reward trajectories summarized by Table 6. The colored curves use debiased exponential smoothing with weight 0.99. Distinct colors and line styles identify $N \in \{ 8 , 1 6 , 3 2 , 6 4 \}$ The same-color shaded region spans one local standard deviation of the raw-minus-smoothed residuals. This band visualizes within-trajectory fluctuation.

For Qwen2.5-0.5B-Instruct, stable late-stage improvement requires a larger population: $N = 8$ and $N = 1 6$ decline over later updates, whereas $N = 3 2$ and N = 64 continue improving. For Qwen2.5-1.5B-Instruct and Qwen2.5-3B-Instruct, the $N = 1 6 , N = 3 2$ , and $N = 6 4$ curves remain substantially closer throughout training.

N = 8 N = 16 N = 32 N = 64  
![](images/f4baee5b228a576b86579e67f63baa54a48a2631fcd46d8f18cc060184d5e7e0.jpg)

(a) Qwen2.5-0.5B-Instruct  
![](images/c73585e688bfca5d0a457794fe292faa584ebf2b4527ebdd2167a53743617942.jpg)  
(b) Qwen2.5-1.5B-Instruct

![](images/cc1f99da1daf87b03523de000a83795581a0962b5ff24ab5545b61cd23699cd0.jpg)  
(c) Qwen2.5-3B-Instruct  
Figure 5 Full GSM8K population-size trajectories across three model scales. Lines show debiased exponential smoothing with weight 0.99. Same-color shading shows the local standard deviation of raw-minus-smoothed residuals; it describes within-trajectory fluctuation. Model-specific vertical ranges expose within-model diferences.

## G Experimental Details

This appendix records the settings used to produce the reported results. We separate training hyperparameters, evaluation settings, prompt templates, and dataset provenance. The locally trained ES/GRPO runs use full-parameter updates. The training and testing hardware used NVIDIA A100-SXM4-80GB GPUs.

## G.1 Implementation-Level Training Procedure

Full-parameter one-point ES. The following procedure describes the implementation used for all locally trained ES models. It is the practical one-point, reward-standardized update studied in the paper. The Easy and Hard settings use the same update procedure and difer only in the model, dataset, epoch budget, and generation settings listed in Table 14.

## Implementation Procedure: Full-parameter one-point ES training

1. Initialize the center model $\theta _ { 0 } ,$ shufle the training split, and construct one keep-tail mini-batch schedule per epoch.

2. At update t, draw N independent 32-bit perturbation seeds. A seed and the stable name of each parameter tensor deterministically reconstruct an independent standard-Gaussian tensor, yielding a full-model direction ϵ<sub>i</sub>.

3. Distribute the N directions over eight single-GPU vLLM engines. For direction $i ,$ add $\sigma \epsilon _ { i }$ to every model

parameter in place, generate one response for every prompt in the shared mini-batch, compute the task-verifier rewards, average them into $R _ { i } ,$ and then subtract the same seeded perturbation to restore the center.

4. Standardize the population rewards with

$$
z _ { i } = \frac { R _ { i } - \bar { R } } { \sqrt { N ^ { - 1 } \sum _ { j = 1 } ^ { N } ( R _ { j } - \bar { R } ) ^ { 2 } } + \varepsilon _ { \mathrm { n u m } } } .
$$

5. Reconstruct the same directions and update every engine locally by

$$
\theta _ { t + 1 } = \theta _ { t } + \frac { \alpha } { N } \sum _ { i = 1 } ^ { N } z _ { i } \epsilon _ { i } .
$$

Here α is the center-update scale in Table 14.

The response-level ES training reward is computed by the task-specific verifier. For the reported mathematical runs, a verifier-correct response receives reward 1; a response that follows the required final-answer format but is incorrect receives the configured format reward 0.1; and an answer that fails the required format receives 0.

## G.2 Training Hyperparameters

Table 14 summarizes the training configurations used for the Easy and Hard ES/GRPO comparisons.

<table><tr><td>Parameter</td><td>Easy ES</td><td>Easy GRPO</td><td>Hard ES</td><td>Hard GRPO</td></tr><tr><td>Model</td><td>Qwen2.5-1.5B-Instruct, Llama-3.2-3B-Instruct, Qwen2.5-7B-Instruct</td><td>Same as Easy ES</td><td>DeepSeek-R1-Distill- Qwen-1.5B</td><td>Same as Hard ES</td></tr><tr><td>Training data</td><td>GSM8K</td><td>GSM8K</td><td>DeepScaleR</td><td>DeepScaleR</td></tr><tr><td>Epochs</td><td>2</td><td>2</td><td>1</td><td>1</td></tr><tr><td>Global train batch</td><td>64</td><td>64</td><td>64</td><td>64</td></tr><tr><td>Population / rollout group</td><td>32 directions</td><td>8 responses</td><td>32 directions</td><td>8 responses</td></tr><tr><td>Learning scale</td><td> $\sigma = 1 . 5 { \times } 1 0 ^ { - 3 } ,$   $\alpha = 2 . 5 { \times } 1 0 ^ { - 4 }$ </td><td> $1 . 0 \times 1 0 ^ { - 6 }$  learning rate</td><td> $\begin{array} { c } { \sigma = 1 . 5 { \times } 1 0 ^ { - 3 } , } \\ { \alpha = 2 . 5 { \times } 1 0 ^ { - 4 } } \end{array}$ </td><td> $1 . 0 \times 1 0 ^ { - 6 }$  learning rate</td></tr><tr><td>Train-time sampling</td><td> $\tau = 0$ </td><td>τ = 1.0</td><td> $\tau = 0 . 6$ </td><td>τ = 1.0</td></tr><tr><td>Response limit</td><td>2,048</td><td>2,048</td><td>8,192</td><td>8,192</td></tr><tr><td>Reward normalization</td><td>Population z-score</td><td>Group-relative advantages</td><td>Population z-score</td><td>Group-relative standardized advantages</td></tr><tr><td>GRPO update details</td><td></td><td>PPO minibatch 32; microbatch 2;</td><td></td><td>clip 0.2; KL coefficient 0.001</td></tr></table>

Table $1 \% .$ Training settings used for the reported comparisons. Hard GRPO is the released discrete-token GRPO checkpoint. Its training parameters are transcribed from Appendix D.1 of the SofT-GRPO paper (Zheng et al., 2026b).

## G.3 Evaluation Configuration

Table 15 summarizes the ofline evaluation configurations used for the Easy and Hard comparisons. Both protocols retain 32 responses per problem at temperature 0.6 and report $K \in \{ 1 6 , 3 2 \}$ . The Easy Setting limits each response to 2,048 new tokens, whereas the Hard Setting allows 8,192 to accommodate longer reasoning traces. Both protocols run on a single GPU. The Easy protocol additionally includes a one-sample greedy evaluation at temperature 0; the main Pass@K and majority-vote tables use the sampled protocol shown here.

<table><tr><td>Parameter</td><td>Easy</td><td>Hard</td></tr><tr><td>Samples per problem</td><td>32</td><td>32</td></tr><tr><td>Temperature</td><td>0.6</td><td>0.6</td></tr><tr><td>Maximum new tokens</td><td>2,048</td><td>8,192</td></tr><tr><td>Reported K</td><td>16, 32</td><td>16, 32</td></tr><tr><td>Execution</td><td>Single GPU</td><td>Single GPU</td></tr></table>

Table 15. Ofline sampled-evaluation settings.

## G.4 Prompt Templates

The following messages are rendered with each model’s oficial chat template.

<table><tr><td>Prompt for Mathematical Reasoning</td></tr><tr><td>system Please reason step by step, and put your final answer within \boxed{}. user {Question}</td></tr><tr><td>Prompt for Multiple-Choice Reasoning system Please reason step by step to solve the following multiple-choice question. Put your final choice letter inside \boxed{}, e.g., \boxed{C}.</td></tr><tr><td>user {Question and labeled choices} Prompt for Open Question Answering</td></tr><tr><td>system Answer the question using the provided context. Keep the final answer short, and put it inside &lt;answer&gt; &lt;/answer&gt; tags. user {Context and question}</td></tr><tr><td>Prompt for MBPP Code Generation system</td></tr><tr><td>You are a Python programming assistant. Write clean, correct Python code that passes the given tests. The function name and signature must match exactly as specified in the test cases. The final code block must not include the tests; do not include the tests in your answer. Do not include explanations or text</td></tr><tr><td>outside the code block. Output exactly one Python markdown code block containing only the final solution code. user</td></tr></table>

Countdown uses the role/content messages distributed with the dataset and renders them without rewriting the task statement. Answer extraction follows the same task-family contract as scoring: boxed answers for mathematics and multiple choice, <answer> tags for open QA and Countdown, and one Python code block for MBPP.

## G.5 Datasets and Setting Rationale

The Easy Setting uses GSM8K because it is a widely used, readily verifiable mathematical-reasoning task and supports a controlled comparison across three instruction-tuned model scales (Cobbe et al., 2021). Its held-out suite uses CommonsenseQA (Talmor et al., 2019), HotpotQA (Yang et al., 2018), Countdown-Task-GOLD (HuggingFaceTB, 2025), GPQA (Rein et al., 2023), and MBPP (Austin et al., 2021) to test cross-domain retention.

The Hard Setting is a harder post-training protocol, not a claim that every included item is uniformly harder than every GSM8K item. It combines the DeepSeek-R1-Distill-Qwen-1.5B backbone with DeepScaleR mathematical RL training data (Luo et al., 2025). This choice aligns the training regime with DeepScaleR and the model/evaluation regime with representative mathematical GRPO work. In particular, JustRL uses the same DeepSeek 1.5B backbone and reports AIME 2024, AIME 2025, AMC 2023, and MATH-500 among its standard mathematical benchmarks (He et al., 2025). The Hard mathematical suite uses the math-ai AIME and AMC releases (Zhang and Math-AI Team, 2024, 2025; math-ai, 2025) and MATH-500 (Lightman et al., 2023).

<table><tr><td>Setting</td><td>Role</td><td>Dataset / split</td><td>Purpose</td></tr><tr><td rowspan="3">Easy</td><td>Post-training</td><td>GSM8K train</td><td>Standard grade-school mathematical word problems with rule-verifiable numeric answers.</td></tr><tr><td>In-domain test</td><td>GSM8K test</td><td>Measures transfer to unseen problems from the post-training task family.</td></tr><tr><td>Held-out test</td><td>CommonsenseQA validation; HotpotQA distractor validation; Countdown test; GPQA Diamond; MBPP sanitized test</td><td>Covers commonsense choice, multi-hop QA, arithmetic planning, scientific reasoning, and code generation.</td></tr><tr><td rowspan="3">Hard</td><td>Post-training</td><td>DeepScaleR Preview</td><td>Mathematical RL training data used in the DeepScaleR line of work. Competition and advanced mathematical reasoning benchmarks</td></tr><tr><td>Mathematical test Held-out test</td><td>AIME 2024; AIME 2025; AMC 2023; MATH-500 GPQA Diamond; MBPP</td><td>commonly used for 1.5B reasoning-RL evaluation. Tests whether mathematical post-training transfers to or</td></tr><tr><td></td><td>sanitized test; CommonsenseQA validation; Countdown test</td><td>degrades non-training task families.</td></tr></table>

Table 16. Datasets and splits used in the reported comparisons. No held-out dataset contributes post-training rewards.
<table><tr><td>Dataset</td><td>Distributed source used for provenance</td><td>License declared by the source</td></tr><tr><td>GSM8K</td><td>openai/gsm8k</td><td>MIT</td></tr><tr><td>CommonsenseQA</td><td>tau/commonsense_qa</td><td>MIT</td></tr><tr><td>HotpotQA Countdown</td><td>Official HotpotQA dataset release HuggingFaceTB/Countdown-Task-GOLD; the</td><td>CC BY-SA 4.0 No license specified in either dataset card</td></tr><tr><td></td><td>fixed test construction corresponds to Jiayi-Pan/Countdown-Tasks-3to4</td><td></td></tr><tr><td>GPQA MBPP</td><td>Idavidrein/gpqa</td><td>CC BY 4.0 CC BY 4.0</td></tr><tr><td>DeepScaleR Preview</td><td>google-research-datasets/mbpp</td><td>MIT</td></tr><tr><td>AIME 2024 / AIME</td><td>agentica-org/DeepScaleR-Preview-Dataset math-ai/aime24; math-ai/aime25</td><td>Apache-2.0</td></tr><tr><td>2025</td><td></td><td></td></tr><tr><td>AMC 2023</td><td>math-ai/amc23</td><td>No license specified in the dataset card</td></tr><tr><td>MATH-500</td><td>HuggingFaceH4/MATH-500</td><td>No license specified in the dataset card</td></tr></table>

Table 17. Licenses declared by the dataset distributions used in this work.

## G.6 Evaluation Metrics

For a problem with M = 32 retained responses and c verifier-correct responses, Pass@1 is the average per-response correctness $c / M$ . For $K \leq M$ , Pass@K uses the standard without-replacement estimator

$$
{ \widehat { \mathrm { P a s s @ } K } } = 1 - { \frac { \binom { M - c } { K } } { \binom { M } { K } } } ,\tag{52}
$$

so Pass@32 is the fraction of problems with at least one correct response. Maj@K normalizes task-specific answers, takes a deterministic plurality vote over the first K retained responses, treats failed extraction as abstention, and scores the selected answer against the gold answer. Reported task scores are arithmetic means over problems, with each problem weighted equally; multi-task averages likewise assign equal weight to each benchmark.

## H Related Work

## H.1 Evolution Strategies for LLM Reasoning Post-Training

Evolution strategies (ES) are population-based black-box optimizers that estimate parameter updates from forward-evaluated perturbations and naturally distribute candidate evaluations across workers (Salimans et al., 2017). Recent work has adapted this paradigm to LLM reasoning post-training. Qiu et al. demonstrate that full-parameter ES can operate directly in billion-dimensional parameter spaces and optimize LLMs with outcome-level rewards (Qiu et al., 2026). ESSA instead restricts evolutionary search to low-rank adapters and further reduces the search space by optimizing their singular values, enabling quantized forward-only alignment on mathematical reasoning and instruction-following tasks (Korotyshova et al., 2025). EGGROLL structures perturbations as low-rank matrices to improve the arithmetic intensity of population evaluation and applies the resulting system to reasoning post-training at large population sizes (Sarkar et al., 2026). ESSAM combines full-parameter ES with a sharpness-aware objective and evaluates this design on GSM8K, emphasizing generalization and memory-eficient reasoning fine-tuning (Sun et al., 2026). These methods establish several viable implementations of ES for LLM post-training, spanning direct full-parameter search, parameter-eficient search, and structured perturbations.

A related line of work asks whether the geometry of ES updates predicts capability retention. Abdi et al. report that ES updates have larger norms and afect a larger fraction of parameter elements than GRPO updates (Abdi et al., 2026). They associate these update properties with held-out degradation. However, their forgetting result is obtained from a restricted experimental configuration comprising a small training set, one model, one training task, and one held-out benchmark, making training-set overfitting dificult to distinguish from broad capability forgetting. Hoy et al. provide complementary evidence that larger parameter movement does not necessarily lead to consistent behavioral degradation: even when ES and GRPO attain similar target-task accuracy, ES travels much farther in weight space and induces broader of-task KL shifts, while the resulting accuracy changes vary across held-out benchmarks and ES iteration budgets (Hoy et al., 2026). Taken together, these studies show that large parameter displacement can accompany capability degradation, but neither weight-space distance nor policy KL has been established as a general predictor, much less a cause, of forgetting.

## H.2 Reinforcement Learning with Verifiable Rewards

Reinforcement learning with verifiable rewards (RLVR) improves reasoning models using automatically checked outcome rewards from domains such as mathematics and code. Group Relative Policy Optimization (GRPO) removes the learned critic by normalizing rewards within a group of responses and optimizing a clipped token-level policy objective (Shao et al., 2024). Although this recipe can improve single-sample accuracy, recent work identifies policy-entropy collapse as a recurring limitation of RLVR. The entropy-mechanism analysis relates entropy change under policy-gradient updates to a probability–update covariance and observes sustained entropy decrease during reasoning training (Cui et al., 2025). Subsequent studies attribute collapse to factors including positive-advantage tokens, imbalanced token-level entropy flows, clipping and of-policy reuse, and premature confidence at a small set of structurally important decision points (Xu et al., 2026; Petrenko et al., 2026; Jin et al., 2026; Jang et al., 2026).

A second line of work evaluates coverage behaviorally through large-budget sampling and asks whether RLVR expands the set of correct reasoning paths accessible from the base model. Large-budget sampling studies find that RLVR can improve Pass@1 while underperforming the base model at larger K, indicating more eficient exploitation of high-probability correct paths but weaker coverage of less likely solutions (Yue et al., 2025). Controlled pretraining experiments further show that RL post-training tends to amplify behaviors already represented in the pretraining distribution (Zhao et al., 2025). Support-based analyses similarly find that RLVR may shrink access to some correct solutions even as it raises single-attempt accuracy (Wu et al., 2025). These findings motivate evaluating Pass@1 together with Pass@K, rather than interpreting an increase in single-sample accuracy as an unconditional expansion of reasoning capability.

## H.3 Zeroth-Order Optimization for LLMs

Zeroth-order (ZO) optimization replaces backpropagated derivatives with a gradient-shaped estimate constructed solely from scalar objective values. Given a Gaussian direction $u \sim \mathcal { N } ( 0 , I )$ and perturbation radius $\mu ,$ two forward evaluations first estimate the directional derivative as $\delta _ { u } f ( \theta ) = [ f ( \theta + \mu u ) - f ( \theta - \mu u ) ] / ( 2 \mu )$ then map this scalar back to parameter space as the pseudo-gradient ${ \widehat { g } } _ { \mathrm { Z O } } ( \theta ; u ) = \delta _ { u } f ( \theta ) u$ . Averaging these pseudo-gradients across directions estimates the gradient of a Gaussian-smoothed objective, enabling parameter updates without diferentiating through the model (Nesterov and Spokoiny, 2017). MeZO adapts simultaneous perturbation-based ZO optimization to operate in place, reducing memory to approximately the inference footprint and supporting both full-parameter and parameter-eficient language-model fine tuning (Malladi et al., 2023). Variance-reduced extensions such as MeZO-SVRG improve the stability and convergence of forward-only language-model fine-tuning by periodically constructing lower-variance update estimates (Gautam et al., 2024). ZO-Bench broadens the comparison across model families, task types, and fine-tuning schemes, and studies blockwise descent, hybrid optimization, forward-gradient estimators, and gradient sparsity (Zhang et al., 2024).

ES and ZO are not disjoint estimator families. With the same objective, perturbation distribution, scale, and raw positive–negative diference, antithetic ES is algebraically identical to a two-point ZO estimator. The practical distinction in LLM reasoning arises from the evaluation protocol: much of the ZO fine-tuning literature studies fixed downstream objectives, whereas reasoning post-training repeatedly generates stochastic autoregressive responses and scores them with terminal verifiers. As analyzed in Appendix E, resampling responses at the perturbed parameter points can weaken the covariance required for paired subtraction to reduce variance. This boundary makes one- versus two-point estimation an empirical design choice under rollout rewards, rather than one whose behavior can be inferred unchanged from fixed supervised objectives.

## I Pass@K Profiles of Individual Post-Training Methods and Sequential Compositions

Figure 6 first compares the Pass@K profiles of ES and GRPO when applied individually, and then examines whether the two sequential compositions can exploit their complementary strengths. In panels (a–b), GRPO leads at Pass@1, whereas ES overtakes it as K grows while remaining above the Base Model across all reported K values. In panels (c–d), ES→GRPO gives the strongest AIME24 profile, while GRPO→ES leads on AIME25 for $K \geq 2$ . Thus, the two sequential compositions can better exploit the complementary strengths of ES and GRPO, although the efective training order is task-dependent.

DeepSeek-R1-Distill-Qwen-1.5B  
![](images/9bc5e6720e2ac3fbc9d845b7a955231064f188515f3205832cf2d9ac8c306ea7.jpg)

![](images/acab8ca6a2431fa2e041f11b1ced906ecd0edef0d8bfbc6e641df909cd456e2d.jpg)

![](images/ac77f43ff9e8f4eaf2fb32dc144f3e05943456df1411007b4ebdc6484c77b93e.jpg)

![](images/bd0d58f98ec447389baf17045d1ae29a183cbbce03ef8166c594c97e0a18af6c.jpg)  
DeepSeek-R1-Distill-Qwen-1.5B  
Figure 6 Pass@K profiles of individual post-training methods and sequential compositions. For individual post-training, GRPO leads at $K = 1$ , whereas ES overtakes it at larger K while remaining above the Base Model throughout. Among the two sequential compositions, ES→GRPO gives the strongest AIME24 profile, while GRPO→ES leads on AIME25 for $K \geq 2 ,$ , showing complementary but order-dependent benefits.