# Risk-Averse Inference-Time Diffusion Model Alignment with Finite-Sample Guarantees

Yuchen Liang<sup>∗</sup> Indiana University Indianapolis

Xian Yu<sup>∗,†</sup> The Ohio State University

## Abstract

Inference-time alignment steers diffusion models toward downstream objectives without costly finetuning, but existing methods typically optimize expected cost (or reward) without explicitly controlling tail risk. In this work, we propose a risk-averse framework for inference-time alignment of diffusion models that balances expected cost and Conditional Value-at-Risk (CVaR) under trajectory-level Kullback-Leibler regularization. We characterize the optimal path distribution and reverse transitions as functions of the CVaR threshold, reducing the alignment problem to a one-dimensional optimization problem over the threshold. We show that this one-dimensional optimization problem can be nonconvex in general, establish sufficient conditions for convexity, and introduce quadratic regularization to enable efficient derivative-free ternary search. Under suitable assumptions, we provide finite-sample guarantees for threshold optimization and value estimation, and combine them into a terminal total-variation bound relative to the regularized target under exact resampling. Experiments on synthetic data and image generation using Stable Diffusion demonstrate improved tail performance over risk-neutral baselines while maintaining competitive average cost.

## 1 Introduction

Pretrained diffusion models [1, 2] have shown remarkable success in generating high-quality images and videos [3, 4, 5], but their outputs may not meet the specific needs of a downstream application. One approach is to fine-tune the pretrained model using task-specific rewards [6], but repeated fine-tuning can be expensive as needs change and may be infeasible when model weights are unavailable. An alternative approach is inference-time alignment, which steers the sampling process toward desired outputs without modifying the pretrained model [7, 8]. A classical formulation of inference-time alignment is to optimize the expected cost (resp. reward) at the output, while ensuring that the steered sampling path is close to the pretrained one [6, 7]. However, in practice, good average performance does not preclude the possibility of undesirable tail outcomes. For example, in molecular design where the expected chemical stability is optimized [9], there could still be a non-negligible probability of generating highly unstable candidates. As another example, diffusion-based robot motion planning requires explicit attention to collision avoidance [10, 11], since strong average task performance alone does not rule out occasional dangerous trajectories.

To this end, there is an urgent need for risk-averse inference-time alignment that explicitly accounts for risk by augmenting the expectation-based objective with a coherent risk measure. Among coherent risk measures, Conditional Value-at-Risk (CVaR) provides a principled way to capture tail behavior, as it measures the average cost beyond a prescribed threshold (i.e., Value-at-Risk) of the output distribution. According to [12, 13], CVaR can be expressed as an optimization problem over this one-dimensional threshold. Recent works have incorporated CVaR into generative model fine-tuning through sequential updates [14] or threshold-based reward transformation [9]. However, these works focus on risk-averse fine-tuning, which can be computationally costly and impractical for large pretrained models.

This paper aims to develop a risk-averse inference-time alignment framework that combines expected cost and CVaR under trajectory-level Kullback-Leibler (KL) regularization while keeping the pretrained model frozen. Establishing the reliability of such a framework requires a non-asymptotic analysis of its sampling accuracy. While existing risk-averse fine-tuning work studies sensitivity to threshold errors [9], inference-time alignment must additionally account for estimation errors in the value functions used to guide sampling. A central theoretical challenge is therefore to characterize how errors in CVaR threshold optimization and value function estimation jointly affect the final output distribution, and how many samples are needed to achieve a prescribed accuracy. This leads to the following question:

How can we align pretrained diffusion models at inference time to control tail risk, and what finite-sample guarantees can we establish?

## 1.1 Our Contributions

Our main contributions can be summarized as follows.

CVaR-based inference-time alignment framework. We characterize the optimal risk-averse sampling policy as a function of the CVaR threshold, reducing the risk-averse alignment problem to a one-dimensional threshold optimization problem. This characterization enables existing riskneutral inference-time samplers, including SVDD and SMC, to incorporate tail-risk control through risk-adjusted value functions while keeping the pretrained diffusion backbone fixed.

Efficient ternary search algorithm for CVaR threshold optimization. We show that the threshold objective can be nonconvex in general and provide sufficient conditions for convexity. For the potentially nonconvex case, we introduce quadratic regularization to obtain a strongly convex objective and employ a ternary search algorithm with an inexact objective evaluation oracle, establishing finite-iteration complexity.

End-to-end sampling guarantees. We further establish finite-sample guarantees for threshold optimization and value estimation under suitable assumptions. Under exact tilted-transition sampling, we combine these results into an end-to-end total-variation bound, quantifying the joint effects of threshold and value-estimation errors on sampling accuracy.

Synthetic and image-generation experiments. We evaluate our risk-averse framework through extensive experiments on heavy-tailed synthetic data and Stable Diffusion image generation [4] with compressibility and aesthetic rewards. Notably, CVaR-SVDD improves empirical tail performance, consistent with our risk-sensitive objective. We further conduct ablation studies across different prompts, sampling procedures, and threshold-optimization algorithms.

## 1.2 Related Work

We review the most relevant works below and discuss additional related works in Appendix A.

CVaR-based generative alignment. Recent works have incorporated CVaR into generative model fine-tuning. Flow Density Control [14] optimizes distributional objectives, including CVaR, through a sequence of fine-tuning problems. Tail-aware Flow Fine-Tuning [9] characterizes an optimal terminal distribution through scalar threshold optimization and uses the resulting transformed reward for fine-tuning. Differently, our framework incorporates CVaR at inference time while keeping the pretrained model fixed. Related inference-time work, RS-Diffuser [15], guides denoising using gradients of CVaR computed from a distributional critic conditioned on noisy candidate plans. In contrast, our formulation optimizes mean-CVaR over the terminal distribution induced by the sampling policy, under trajectory-level KL regularization.

Theoretical guarantees for alignment. Several recent works have established theoretical guarantees for inference-time alignment. Learnable Chernoff Baselines [16] provide total-variation guarantees for approximate rejection sampling from exponentially tilted transitions. DOIT [17] analyzes sampling accuracy under approximation of the dynamic correction induced by Doob’s h-transform. [18] studies how approximate value functions, quantified through Bellman errors, affect inference-time sampling, including finite-particle SMC complexity. For CVaR-based fine-tuning, [9] also studies sensitivity of the target distribution to threshold errors. Our analysis complements these results by establishing sample-complexity bounds for CVaR threshold optimization and value estimation, and quantifying how these two sources of error jointly affect the final output distribution in total variation.

## 2 Background

In this section, we introduce the necessary background.

## 2.1 (Continuous-Space) Diffusion Models

Diffusion models [1, 2] generate samples through a reverse process that progressively removes noise added by a forward process. Given initial data $\boldsymbol { x } _ { 0 } \in \mathbb { R } ^ { d } ,$ , the forward process gradually adds Gaussian noise:

$$
x _ { t } = \sqrt { 1 - \beta _ { t } } x _ { t - 1 } + \sqrt { \beta _ { t } } w _ { t } , \qquad t = 1 , \ldots , T ,
$$

where $w _ { t } \stackrel { i . i . d . } { \sim } \mathcal { N } ( 0 , I _ { d } )$ and $\beta _ { t } \in ( 0 , 1 )$ is the noise schedule. Define $\alpha _ { t } : = 1 - \beta _ { t }$ and $\textstyle { \bar { \alpha } } _ { t } : = \prod _ { i = 1 } ^ { t } \alpha _ { i }$ Accumulating the forward steps gives

$$
q _ { t \vert 0 } ( x _ { t } \mid x _ { 0 } ) = \mathcal { N } \big ( \sqrt { \bar { \alpha } _ { t } } x _ { 0 } , ( 1 - \bar { \alpha } _ { t } ) I _ { d } \big ) .
$$

Thus, when $\hat { \alpha } _ { T }$ is sufficiently small, the terminal distribution is approximately standard Gaussian.

The reverse sampling process starts from $x _ { T } \sim q _ { T } : = \mathcal { N } ( 0 , I _ { d } )$ and applies learned denoising transitions for $t = T , \dots , 1$ . Tweedie’s formula yields (cf. [19])

$$
q _ { t - 1 | t } ( x _ { t - 1 } | x _ { t } ) = \mathcal { N } \left( m _ { t } ( x _ { t } ) , \sigma _ { t } ^ { 2 } I _ { d } \right) ,
$$

where $\begin{array} { r } { m _ { t } ( x _ { t } ) : = \frac { 1 } { \sqrt { \alpha _ { t } } } \left( x _ { t } + ( 1 - \alpha _ { t } ) s _ { t } ( x _ { t } ) \right) } \end{array}$ , s<sub>t</sub> estimates the score, i.e., ∇ log $q _ { t }$ at time $t ,$ and $\sigma _ { t } ^ { 2 }$ is a prescribed reverse variance. To unify notation, we also define $x _ { T + 1 }$ as a (dummy) deterministic state, with $q _ { T | T + 1 } ( x _ { T } | x _ { T + 1 } ) = q _ { T } ( x _ { T } )$

Throughout this paper, $q$ denotes the joint distribution induced by the pretrained sampling path:

$$
\boldsymbol { q } ( \boldsymbol { x } _ { 0 : T } ) = \boldsymbol { q } _ { T } ( \boldsymbol { x } _ { T } ) \prod _ { t = 1 } ^ { T } \boldsymbol { q } _ { t - 1 | t } ( \boldsymbol { x } _ { t - 1 } | \boldsymbol { x } _ { t } ) ,
$$

with $q _ { t }$ denoting the marginal distribution at time t. In particular, $q _ { 0 }$ is the pretrained model’s output distribution, which serves as the reference distribution for inference-time sampling.

## 2.2 (Risk-Neutral) Inference-Time Alignment

In practice, one often seeks to align a pretrained diffusion model with a reward function to generate samples with higher rewards. To this end, inference-time alignment aims to generate samples with

higher rewards without changing the weights of the pretrained model. Writing the terminal cost as $c ( \bar { x } _ { 0 } ) : = - r ( x _ { 0 } )$ , a typical KL-regularized objective is [7, 20]:

$$
\operatorname* { m i n } _ { \pi } \mathbb { E } _ { \pi } [ c ( x _ { 0 } ) ] + \alpha { \mathrm { K L } } ( \pi _ { 1 : T + 1 } | | \mu _ { 1 : T + 1 } ) ,\tag{1}
$$

where $\alpha > 0$ controls the regularization strength. Here $\pi : = \pi _ { 1 : T + 1 }$ denotes the joint path distribution induced by the initial distribution $\pi _ { T + 1 }$ and reverse transitions $\pi _ { t } , t = 1 , . . . , T$ . The reference distribution $\mu _ { 1 : T + 1 } = q$ is induced by the pretrained transitions $\mu _ { t } : = q _ { t - 1 | t }$ with initialization $\mu _ { T + 1 } = q _ { T } = \mathcal { N } ( 0 , I _ { d } )$

For $t = 1 , \dots , T + 1$ , let $V _ { t } ^ { * } ( x _ { t } )$ denote the optimal regularized cost-to-go (or soft value function) from state ${ \boldsymbol { x } } _ { t } ,$ defined as follows:

$$
V _ { t } ^ { * } ( x _ { t } ) : = \operatorname* { m i n } _ { \pi _ { 1 : t } } \mathbb { E } _ { \pi } \left[ c ( x _ { 0 } ) + \alpha \mathrm { K L } ( \pi _ { 1 : t } | | \mu _ { 1 : t } ) \mid x _ { t } \right] .\tag{2}
$$

Here $\pi _ { 1 : t } ( \cdot \mid x _ { t } )$ and $\mu _ { 1 : t } ( \cdot \mid x _ { t } )$ denote the laws of the remaining trajectory $x _ { 0 : t - 1 }$ starting from $x _ { t }$ . In other words, $V _ { t } ^ { * } ( x _ { t } )$ accounts for both the expected terminal cost and the accumulated KL penalties over the remaining reverse steps. Note that the terminal condition is $V _ { 0 } ^ { * } ( x _ { 0 } ) = c ( x _ { 0 } )$ . From $[ 7 ] ,$ , the optimal soft value function at time t has the following recursion

$$
V _ { t } ^ { * } ( x _ { t } ) = - \alpha \log \mathbb { E } _ { \mu _ { t } } \left[ e ^ { - V _ { t - 1 } ^ { * } ( x _ { t - 1 } ) / \alpha } \Big | x _ { t } \right] ,\tag{3}
$$

and the corresponding optimal reverse transitions satisfy

$$
\begin{array} { r l } & { \pi _ { t } ^ { * } ( x _ { t - 1 } | x _ { t } ) = \mu _ { t } ( x _ { t - 1 } | x _ { t } ) e ^ { - ( V _ { t - 1 } ^ { * } ( x _ { t - 1 } ) - V _ { t } ^ { * } ( x _ { t } ) ) / \alpha } } \\ & { \phantom { e } \propto \mu _ { t } ( x _ { t - 1 } | x _ { t } ) e ^ { - V _ { t - 1 } ^ { * } ( x _ { t - 1 } ) / \alpha } . } \end{array}
$$

Concatenation over t further yields

$$
\begin{array} { r l r } & { } & { V _ { t } ^ { * } ( x _ { t } ) = - \alpha \log \mathbb { E } _ { q _ { 0 \mid t } } \left[ e ^ { - V _ { 0 } ^ { * } ( x _ { 0 } ) / \alpha } \bigg | x _ { t } \right] } \\ & { } & { = - \alpha \log \mathbb { E } _ { q _ { 0 \mid t } } \left[ e ^ { - c ( x _ { 0 } ) / \alpha } \bigg | x _ { t } \right] . } \end{array}
$$

Denote the initial value as $V _ { T + 1 } ^ { * } ,$ which is independent of the sampling path. Let $p _ { t } ^ { * }$ denote the marginal distribution at time t under the optimal policy. The optimal output distribution is the exponentially tilted distribution $[ 7 , 2 0 ] \colon p _ { 0 } ^ { * } ( x _ { 0 } ) \propto q _ { 0 } ( x _ { 0 } ) e ^ { - c ( x _ { 0 } ) / \alpha }$ . For the rest of this paper, we omit the conditional dependence of $V _ { t } ^ { * }$ on $x _ { t }$ when it is obvious from the context.

In practice, the optimal value function $V _ { t } ^ { * }$ is typically intractable to obtain and is replaced by an estimate $\hat { V } _ { t } .$ . Although sampling from the pretrained reverse kernels is straightforward, incorporating the exponential tilt requires additional sampling procedures. Common approaches use “samplingresampling” techniques, including SVDD [7] and SMC [8]. The corresponding pseudocode is provided in Appendix B.

Given the value estimates $\hat { V } _ { t } ,$ , define the approximate policy with exact tilting as

$$
\hat { \pi } _ { t } ( x _ { t - 1 } | x _ { t } ) \propto \mu _ { t } ( x _ { t - 1 } | x _ { t } ) e ^ { - \hat { V } _ { t - 1 } ( x _ { t - 1 } ) / \alpha } ,
$$

with the normalizing constant $\hat { Z } _ { t } ( x _ { t } ) : = \mathbb { E } _ { \mu _ { t } } \left[ e ^ { - \hat { V } _ { t - 1 } ( x _ { t - 1 } ) / \alpha } \Big | x _ { t } \right]$ . Unlike the optimal value functions, the estimates need not satisfy the soft Bellman recursion (3). Consequently, $\hat { Z } _ { t } ( x _ { t } )$ generally does not equal $e ^ { - \hat { V } _ { t } ( x _ { t } ) / \alpha }$

## 2.3 Conditional Value-at-Risk (CVaR)

The risk-neutral objective in (1) optimizes the expected cost without explicitly controlling high-cost outcomes. To account for such tail risk, one popular method is to use CVaR [13], which quantifies the expectation of the upper tail distribution. Given a pre-specified risk level $\beta \in ( 0 , 1 )$ (close to 1), the CVaR of the cost distribution $c ( x _ { 0 } )$ , x<sub>0</sub> $\sim \pi$ is defined as follows:

$$
\operatorname { C V a R } _ { \beta , \pi } [ c ( x _ { 0 } ) ] : = \operatorname* { m i n } _ { \eta \in \mathbb { R } } \left\{ \eta + \frac { 1 } { 1 - \beta } \mathbb { E } _ { \pi } \left[ ( c ( x _ { 0 } ) - \eta ) ^ { + } \right] \right\} ,\tag{4}
$$

where $( a ) ^ { + } : = \operatorname* { m a x } \{ 0 , a \}$ . According to [13], the optimal threshold for (4) is given by the Value-at-Risk (VaR), i.e., the β-quantile of the distribution $c ( x _ { 0 } )$

$$
\begin{array} { r } { \eta _ { \pi } ^ { * } = \mathrm { V a R } _ { \beta , \pi } [ c ( x _ { 0 } ) ] : = \operatorname* { i n f } \left\{ v : \mathbb { P } _ { \pi } \left\{ c ( x _ { 0 } ) \leq v \right\} \geq \beta \right\} , } \end{array}
$$

which depends on π. For notational simplicity, we suppress this dependence and write $\eta ^ { * }$ instead.

## 3 Risk-averse Inference-time Alignment

To balance average performance and tail risk, we consider the following convex combination

$$
\rho _ { \beta , \pi } [ c ( x _ { 0 } ) ] : = ( 1 - \lambda ) \operatorname { \mathbb { E } } _ { \pi } [ c ( x _ { 0 } ) ] + \lambda { \mathrm { C V a R } } _ { \beta , \pi } [ c ( x _ { 0 } ) ] ,\tag{5}
$$

where $\lambda \in [ 0 , 1 ]$ controls the degree of risk aversion. The choices $\lambda = 0$ and $\lambda = 1$ recover the expected cost and CVaR, respectively. Replacing the expectation with ${ \rho } _ { \beta , \pi } ,$ a risk-averse counterpart of (1) is

$$
\operatorname* { m i n } _ { \pi _ { 1 : T + 1 } } \rho _ { \beta , \pi } [ c ( x _ { 0 } ) ] + \alpha \mathrm { K L } ( \pi _ { 1 : T + 1 } | | \mu _ { 1 : T + 1 } ) .
$$

Using the threshold representation of CVaR in (4), it is equivalent to optimize jointly over π and η:

$$
\operatorname* { m i n } _ { \pi , \eta } \mathbb { E } _ { \pi } \left[ c _ { \lambda , \eta } ( x _ { 0 } ) \right] + \alpha \mathrm { K L } ( \pi _ { 1 : T + 1 } | | \mu _ { 1 : T + 1 } ) ,\tag{6}
$$

where the risk-adjusted terminal cost is defined as

$$
c _ { \lambda , \eta } ( x _ { 0 } ) : = ( 1 - \lambda ) c ( x _ { 0 } ) + \lambda \left( \eta + \frac { ( c ( x _ { 0 } ) - \eta ) ^ { + } } { 1 - \beta } \right) .\tag{7}
$$

For each fixed $\eta ,$ let $V _ { t , \eta } ^ { * }$ denote the optimal regularized cost-to-go obtained by replacing c with $c _ { \lambda , \eta }$ in the risk-neutral problem (2). The following theorem characterizes the optimal path distribution $\pi ^ { * }$ as a function of $\eta ,$ reducing (6) to a one-dimensional optimization problem over η. Theorem 1 (Reduction to η optimization). For every $\eta \in \mathbb { R }$ , assume that $\bar { \mathbb { E } } _ { q _ { 0 } } \big [ ( 1 + | c ( x _ { 0 } ) | ) e ^ { - c _ { \lambda , \eta } ( x _ { 0 } ) / \alpha } \big ] <$ ∞. Then, the optimal reverse transitions in (6) satisfy

$$
\pi _ { t , \eta } ^ { * } ( x _ { t - 1 } \mid x _ { t } ) \propto \mu _ { t } ( x _ { t - 1 } \mid x _ { t } ) e ^ { - V _ { t - 1 , \eta } ^ { * } ( x _ { t - 1 } ) / \alpha }
$$

for each $t = T + 1 , \ldots , 1$ , where

$$
\begin{array} { r } { V _ { t , \eta } ^ { * } ( x _ { t } ) = - \alpha \log \mathbb { E } _ { q _ { 0 \mid t } } \left[ e ^ { - c _ { \lambda , \eta } ( x _ { 0 } ) / \alpha } \Big | x _ { t } \right] , } \end{array}\tag{8}
$$

and $V _ { 0 , \eta } ^ { * } ( x _ { 0 } ) : = c _ { \lambda , \eta } ( x _ { 0 } )$ . Thus, (6) reduces to the one-dimensional optimization problem over η:

$$
\begin{array} { r l } & { \underset { \eta \in \mathbb { R } } { \operatorname* { m i n } } G _ { \lambda } ( \eta ) : = V _ { T + 1 , \eta } ^ { * } = - \alpha \log \mathbb { E } _ { q _ { 0 } } \left[ e ^ { - c _ { \lambda , \eta } ( x _ { 0 } ) / \alpha } \right] } \\ & { = \lambda \eta - \alpha \log \mathbb { E } _ { q _ { 0 } } \left[ e ^ { - \left( ( 1 - \lambda ) c ( x _ { 0 } ) + \frac { \lambda } { 1 - \beta } ( c ( x _ { 0 } ) - \eta ) ^ { + } \right) / \alpha } \right] . } \end{array}\tag{9}
$$

The proof of Theorem 1 is provided in Appendix C. It uses the same line of proof as [6, Theorem 1], with the terminal cost replaced by $c _ { \lambda , \eta } .$ . For each fixed $\eta ,$ the risk-averse problem therefore retains the structure of risk-neutral alignment: the pretrained reverse kernels remain unchanged, while the risk preference is incorporated through the modified value functions $V _ { t , \eta } ^ { * } .$ As a result, existing inference-time samplers, such as SVDD and SMC, can still be adapted by using the corresponding risk-adjusted value estimates, as long as the optimal $\eta ^ { * }$ is given. For completeness, Algorithm 1 presents the CVaR extension of the SVDD sampler. The SMC sampler can be adapted analogously.

Algorithm 1: CVaR-SVDD   
Input: Pretrained reverse kernels $\{ \mu _ { t } \} _ { t = 1 } ^ { T + 1 }$ , threshold $\eta ,$ risk-adjusted cost $c _ { \lambda , \eta } ,$ value estimates   
$\{ \bar { \widehat { V } } _ { t , \eta } \} _ { t = 1 } ^ { T }$ , number of samples $N ,$ number of candidates $M _ { ☉ }$ , temperature α > 0   
Set $\widehat { V } _ { 0 , \eta } = c _ { \lambda , \eta }$   
Run SVDD (Algorithm 3) with the given $\{ \mu _ { t } \} _ { t = 1 } ^ { T + 1 } , N , M ,$ and $\alpha ,$ using $\widehat { V } _ { t } = \widehat { V } _ { t , \eta }$ for $t = 0 , \ldots , T$   
Output: Samples $\{ x _ { 0 } ^ { ( i ) } \} _ { i = 1 } ^ { N }$ returned by SVDD

## 4 Threshold Optimization

To employ Algorithm 1, it remains to determine the optimal $\eta ^ { * }$ . This section analyzes the convexity of the threshold optimization problem (9) and proposes a ternary search algorithm to solve it efficiently.

## 4.1 Convexity of the Threshold Objective

The following proposition shows that the objective function $G _ { \lambda } ( \eta )$ in (9) is nonconvex in general and provides a sufficient and necessary condition for establishing its convexity.

Proposition 1 (Non-convexity of $G _ { \lambda } )$ . Suppose that the distribution of $\scriptstyle ( c ( x _ { 0 } )$ under $q _ { 0 }$ admits a continuous density h. Define

$$
\begin{array} { r } { A _ { \lambda } ( \eta ) : = \mathbb { E } _ { q _ { 0 } } \left[ e ^ { - c _ { \lambda , \eta } ( x _ { 0 } ) / \alpha } \mathbb { 1 } \left\{ c ( x _ { 0 } ) \leq \eta \right\} \right] , } \\ { B _ { \lambda } ( \eta ) : = \mathbb { E } _ { q _ { 0 } } \left[ e ^ { - c _ { \lambda , \eta } ( x _ { 0 } ) / \alpha } \mathbb { 1 } \left\{ c ( x _ { 0 } ) > \eta \right\} \right] , } \end{array}
$$

and assume that $Z _ { \lambda } ( \eta ) : = A _ { \lambda } ( \eta ) + B _ { \lambda } ( \eta )$ is finite for every $\eta \in \mathbb { R }$ . Then $G _ { \lambda }$ in (9) is convex if and only if

$$
e ^ { - \eta / \alpha } h ( \eta ) Z _ { \lambda } ( \eta ) \ge \frac { \lambda } { \alpha ( 1 - \beta ) } A _ { \lambda } ( \eta ) B _ { \lambda } ( \eta )\tag{10}
$$

for every $\eta \in \mathbb R$ . Moreover, we have $\begin{array} { r } { G _ { \lambda } ^ { \prime } ( \eta ) = \lambda - \frac { \lambda } { 1 - \beta } \cdot \frac { B _ { \lambda } ( \eta ) } { Z _ { \lambda } ( \eta ) } } \end{array}$ and $\begin{array} { r } { G _ { \lambda } ^ { \prime \prime } ( \eta ) \ge - \frac { \lambda ^ { 2 } } { 4 \alpha ( 1 - \beta ) ^ { 2 } } } \end{array}$ for all $\eta \in \mathbb { R }$

The proof of Proposition 1 is provided in Appendix D. Note that for each fixed policy π, the objective in (6) is convex in η. However, minimizing over π does not generally preserve this convexity. We provide a simple sufficient condition where the resulting $G _ { \lambda } ( \eta )$ is non-convex in $\eta$ towards the end of Appendix D.

While (10) characterizes the sufficient and necessary condition for convexity, verifying it requires knowledge of the cost density h and the corresponding tilted expectations at every threshold η. One way to establish the convexity of $G _ { \lambda }$ is to impose additional structural conditions, avoiding the need to directly verify (10). The following proposition provides such an example.

Proposition 2 (Convexity of $G _ { \lambda } )$ . Suppose that the pretrained terminal density $q _ { 0 }$ is log-concave and that $c ( x _ { 0 } )$ is convex and bounded on the support of $q _ { 0 }$ . Then, the objectivefunction ${ \dot { G } } _ { \lambda } ( \eta )$ in (9) is convex in η.

The proof is in Appendix E. Note that the stated assumption is one sufficient condition for convexity, and other simple sufficient conditions may also be possible.

## 4.2 Quadratic Regularization

Although $G _ { \lambda }$ is nonconvex in general, its negative curvature is uniformly bounded by $\frac { \lambda ^ { 2 } } { 4 \alpha ( 1 - \beta ) ^ { 2 } }$ , as shown in Proposition 1. In other words, $G _ { \lambda } \mathrm { i s } \frac { \lambda ^ { 2 } } { 4 \alpha ( 1 - \beta ) ^ { 2 } }$ -weakly convex. This motivates a quadratically regularized threshold objective, defined as follows:

$$
\operatorname* { m i n } _ { \eta \in \mathbb { R } } G _ { \lambda , \kappa } ( \eta ) : = G _ { \lambda } ( \eta ) + \frac { \kappa } { 2 } ( \eta - \eta _ { c } ) ^ { 2 } ,\tag{11}
$$

where $\eta _ { c }$ is an arbitrary center and $\kappa \geq 0$ controls the regularization strength. For the most general case, Proposition 1 implies that $G _ { \lambda , \kappa }$ is strongly convex with parameter $\begin{array} { r } { \kappa - \frac { \lambda ^ { 2 } } { 4 \alpha ( 1 - \beta ) ^ { 2 } } } \end{array}$ whenever $\begin{array} { r } { \kappa > \frac { \lambda ^ { 2 } } { 4 \alpha ( 1 - \beta ) ^ { 2 } } . \operatorname { I f } G _ { \lambda } } \end{array}$ is already convex (as shown in Proposition 2), any $\kappa > 0$ instead yields κ-strong convexity. Furthermore, as we show in Corollary 1 in Appendix ${ \mathrm { F } } ,$ for any $\kappa \geq 0$ and $\eta _ { c } \in \mathbb { R }$ , the quadratically regularized version of (6) can also be recast as the minimization of $G _ { \lambda , \kappa } ( \eta )$

Let $\eta ^ { * }$ denote any minimizer of $G _ { \lambda }$ (assuming it exists), and let $\eta _ { \kappa } ^ { * }$ denote the unique minimizer of (11) under the corresponding strong-convexity condition. Optimality of $\eta _ { \kappa } ^ { * }$ yields

$$
\begin{array} { l } { 0 \leq G _ { \lambda } ( \eta _ { \kappa } ^ { * } ) - G _ { \lambda } ( \eta ^ { * } ) } \\ { \leq \displaystyle \frac { \kappa } { 2 } \left[ ( \eta ^ { * } - \eta _ { c } ) ^ { 2 } - ( \eta _ { \kappa } ^ { * } - \eta _ { c } ) ^ { 2 } \right] \leq \frac { \kappa } { 2 } ( \eta ^ { * } - \eta _ { c } ) ^ { 2 } . } \end{array}
$$

Hence, the regularization bias in objective value vanishes as $\kappa  0$ . Thus, in the remainder of this paper, we focus on solving the regularized objective in (11).

## 4.3 Ternary Search with Inexact Objective Evaluation Oracle

We next consider how to compute $\eta _ { \kappa } ^ { * }$ . According to Proposition 1, the first-order optimality condition of (11) gives $\begin{array} { r } { G _ { \lambda , \kappa } ^ { \prime } ( \eta ) = \lambda - \frac { \lambda } { 1 - \beta } \cdot \frac { B _ { \lambda } ( \eta ) } { Z _ { \lambda } ( \eta ) } + \kappa ( \eta - \eta _ { c } ) = 0 , } \end{array}$ , which generally does not yield a closed-form solution. We therefore turn to a derivative-free one-dimensional search that relies only on inexact estimates of the objective. To specify a finite search interval, we impose the following regularity condition.

Assumption 1. The cost function c satisfies $c ( x _ { 0 } ) \in [ l , u ]$ on the support of $q _ { 0 } ,$ where $- \infty < l < u <$ ∞.

Such a bounded-cost assumption also appears in sampling-error analyses, such as [16]. Under Assumption 1, if $\eta _ { c } \in [ l , u ]$ , then $G _ { \lambda , \kappa } ^ { \prime } ( \eta )$ is negative for $\eta < l$ and positive for $\eta > u . \ S 0 , \eta _ { \kappa } ^ { * } \in [ l , u ]$ and the search can be restricted to $[ l , u ]$ without loss of optimality.

Under Assumption 1 and the strong-convexity condition on $G _ { \lambda , \kappa } ,$ we employ ternary search [21] with an inexact objective oracle $\widehat { G } _ { \lambda , \kappa } .$ . The procedure is summarized in Algorithm 2. Specifically, the search starts from the full range $[ l , u ]$ . At each iteration, the algorithm partitions the current interval into thirds by placing two interior points, $z _ { l }$ at one-third of the way and $z _ { u }$ at two-thirds, and queries the inexact oracle to evaluate the objective at these two points. Now, if $G _ { \lambda , \kappa } ( z _ { l } ) \leq G _ { \lambda , \kappa } ( z _ { u } )$ , then for a convex function the minimizer cannot lie to the right of $z _ { u }$ . As a result, the algorithm discards the right third and keeps $[ y _ { l } , z _ { u } ]$ . With the inexact oracle $\widehat { G } _ { \lambda , \kappa } ,$ the resulting objective error can still be controlled with a uniform bound on the oracle error (see Lemma 1). The algorithm then compares $z _ { l }$ with the incumbent and makes $z _ { l }$ as the new incumbent if it is as least as good. Otherwise, the algorithm discards the left third and keeps $[ z _ { l } , y _ { u } ]$ . Each iteration shrinks the interval by a factor of

Algorithm 2: CVaR-Optimizer   
Input: Interval $[ l , u ] \subset \mathbb { R } ,$ target error $\epsilon ,$ Lipschitz bound $L > 0$   
Input: Inexact objective evaluation oracle $\widehat { G } _ { \lambda , \kappa } ( \cdot ) : \mathbb { R } \to \mathbb { R }$   
Let $\eta ^ { ( 0 ) } = l , y _ { l } ^ { ( 0 ) } = l , y _ { u } ^ { ( 0 ) } = u$   
for $i = 1 , . . . , \left\lceil \log _ { 3 / 2 } ( \frac { L ( u - l ) } { \epsilon } ) \right\rceil$ do   
Let $\begin{array} { r } { z _ { l } ^ { ( i ) } = \frac { 2 y _ { l } ^ { ( i - 1 ) } + y _ { u } ^ { ( i - 1 ) } } { 3 } , z _ { u } ^ { ( i ) } = \frac { y _ { l } ^ { ( i - 1 ) } + 2 y _ { u } ^ { ( i - 1 ) } } { 3 } } \end{array}$   
if $\widehat { G } _ { \lambda , \kappa } ( z _ { l } ^ { ( i ) } ) \leq \bar { G } _ { \lambda , \kappa } ( z _ { u } ^ { ( i ) } )$ then   
Let $( y _ { l } ^ { ( i ) } , y _ { u } ^ { ( i ) } ) = ( y _ { l } ^ { ( i - 1 ) } , z _ { u } ^ { ( i ) } ) .$   
If $\widehat { G } _ { \lambda , \kappa } ( z _ { l } ^ { ( i ) } ) \leq \widehat { G } _ { \lambda , \kappa } ( \eta ^ { ( i - 1 ) } )$ update $\eta ^ { ( i ) } = z _ { l } ^ { ( i ) } ;$ ; else update $\eta ^ { ( i ) } = \eta ^ { ( i - 1 ) }$   
else if $\widehat { G } _ { \lambda , \kappa } ( z _ { l } ^ { ( i ) } ) > \widehat { G } _ { \lambda , \kappa } ( z _ { u } ^ { ( i ) } )$ then   
Let $( y _ { l } ^ { ( i ) } , \dot { y _ { u } ^ { ( i ) } } ) = ( z _ { l } ^ { ( i ) } , y _ { u } ^ { ( i - 1 ) } )$   
If $\widehat { G } _ { \lambda , \kappa } ( z _ { u } ^ { ( i ) } ) \leq \widehat { G } _ { \lambda , \kappa } ( \eta ^ { ( i - 1 ) } )$ update $\eta ^ { ( i ) } = z _ { u } ^ { ( i ) } ;$ ; else update $\eta ^ { ( i ) } = \eta ^ { ( i - 1 ) }$   
end   
end   
Output: $\hat { \eta } : = \eta ^ { ( \mathrm { l a s t } ) }$

$2 / 3$ and after $\left\lceil \log _ { 3 / 2 } ( \frac { L ( u - l ) } { \epsilon } ) \right\rceil$ iterations, its length is at most $\epsilon / L$ . We will establish the iteration and sample complexity guarantees of this derivative-free ternary search algorithm in Section 5.1. Based on the numerical results in Section $6 ,$ Algorithm 2 is much more efficient than grid search and the gradient descent algorithm.

## 5 Theoretical Guarantees

In this section, we establish theoretical guarantees for the sampler in Algorithm 1. We first analyze the error of the threshold optimization in Algorithm 2 (Section 5.1), then the value estimation error (Section 5.2), and finally combine them into an end-to-end total variation bound on the output distribution (Section 5.3).

## 5.1 Threshold Optimization Guarantees

We first quantify the accuracy of the threshold returned by Algorithm 2. Since the regularized objective (11) involves an expectation over the pretrained distribution $q _ { 0 } ,$ , we evaluate it using a Monte Carlo estimator based on i.i.d. samples from $q _ { 0 }$ . Specifically, given a single batch of i.i.d. samples $\{ x _ { 0 } ^ { i } \} _ { i = 1 } ^ { N } \sim q _ { 0 }$ , define an inexact objective evaluation oracle $\widehat { G } _ { \lambda , \kappa , N } ( \cdot ) : \mathbb { R }  \mathbb { R }$ as

$$
\widehat { G } _ { \lambda , \kappa , N } ( \eta ) : = - \alpha \log \Big ( \frac { 1 } { N } \sum _ { i = 1 } ^ { N } e ^ { - c _ { \lambda , \eta } ( x _ { 0 } ^ { i } ) / \alpha } \Big ) + \frac { \kappa } { 2 } ( \eta - \eta _ { c } ) ^ { 2 } .
$$

The following theorem states a finite-sample guarantee that Algorithm 2 produces a near-optimal solution to (11) when using the Monte Carlo oracle $\widehat { G } _ { \lambda , \kappa , N } ( \eta )$

Theorem 2 (Finite-sample guarantees for threshold optimization). Suppose Assumption 1 holds. Let $\eta _ { c } \in [ l , u ]$ be arbitrary, and suppose that $G _ { \lambda , \kappa }$ is $m _ { \kappa }$ -strongly convex on [l, u] for some $m _ { \kappa } > 0 .$ . Define

$$
L : = \lambda \operatorname* { m a x } \left\{ 1 , \frac { \beta } { 1 - \beta } \right\} , \quad L _ { \kappa } : = L + \kappa ( u - l ) .\tag{12}
$$

For any $\delta \in ( 0 , 1 )$ and sufficiently small $\varepsilon _ { \eta } > 0$ , define

$$
K : = \Big \lceil \log _ { 3 / 2 } \Big ( \frac { 8 L _ { \kappa } ( u - l ) } { m _ { \kappa } \varepsilon _ { \eta } ^ { 2 } } \Big ) \Big \rceil , ~ J : = \Big \lceil \frac { 3 2 L ( u - l ) } { m _ { \kappa } \varepsilon _ { \eta } ^ { 2 } } \Big \rceil + 1 .
$$

Then, with probability at least $1 - \delta ,$ the output ηˆ of Algorithm 2 (with Lipschitz bound $L _ { \kappa }$ and target error $m _ { \kappa } \varepsilon _ { \eta } ^ { 2 } / 8 )$ satisfies $| \hat { \eta } - \eta _ { \kappa } ^ { * } | \leq \varepsilon _ { \eta }$ as long as

$$
N \ge \frac { 4 C _ { G } } { m _ { \kappa } ^ { 2 } \varepsilon _ { \eta } ^ { 4 } } \log \frac { 2 J } { \delta }
$$

where $C _ { G } > 0$ is a constant depending on $\alpha , \beta , \lambda , l , u .$ . Furthermore, the algorithm calls the Monte Carlo oracle $\widehat { G } _ { \lambda , \kappa , N }$ at most $1 + 2 K$ times.

The proof is in Appendix G. We first use Bernstein’s inequality to obtain a pointwise bound on the error of the inexact objective oracle, and then discretize the η-space to turn this pointwise bound into a uniform one using a union bound, which yields the final sample complexity. The oracle round (iteration) complexity is $\begin{array} { r } { O ( \log \frac { L _ { \kappa } ( u - l ) } { m _ { \kappa } \varepsilon _ { \eta } ^ { 2 } } ) } \end{array}$ , and the sample complexity is $\begin{array} { r } { O \big ( \frac { \hat { 1 } } { m _ { \kappa } ^ { 2 } \varepsilon _ { \eta } ^ { 4 } } \log \frac { L ( u - l ) } { m _ { \kappa } \varepsilon _ { \eta } ^ { 2 } \delta } \big ) } \end{array}$ . Note that these samples are drawn once and reused for all queried points in Algorithm 2.

## 5.2 Value Estimation Guarantees

Having established the complexity of the threshold optimization problem, we next proceed to the sample complexity of the value estimates used by the SVDD sampler. Different from Section 5.1 where $G _ { \lambda , \kappa } ( \bar { \eta } )$ is estimated at a fixed terminal state by averaging i.i.d. samples from $q _ { 0 } ,$ , the value functions $V _ { t , \eta } ^ { * }$ will be estimated by regressing on paired samples $\left\{ ( x _ { t } ^ { i } , x _ { 0 } ^ { i } ) \right\} _ { i = 1 } ^ { N } .$

For each $t = 1 , \ldots , T _ { \scriptscriptstyle { \mathscr { T } } }$ , suppose that $\{ ( x _ { t } ^ { i } , x _ { 0 } ^ { i } ) \} _ { i = 1 } ^ { N \mathrm { ~ \stackrel { \scriptstyle ~ i . i . d . } { ~ \sim ~ } } q _ { t } ( x _ { t } ) q _ { 0 | t } ( x _ { 0 } \mid x _ { t } ) }$ . Recall $c _ { \lambda , \eta }$ from (7) and $V _ { t , \eta } ^ { * }$ from (8). Define the (normalized) target as $Y _ { \eta } ( x _ { 0 } ) : = e ^ { - \frac { c _ { \lambda , \eta } ( x _ { 0 } ) - a _ { \eta } } { \alpha } }$ where $a _ { \eta } : = ( 1 - \lambda ) l + \lambda \eta$ is a lower bound on $c _ { \lambda , \eta } ( \boldsymbol { x } _ { 0 } )$ . Let

$$
\xi : = \frac { 2 ( u - l ) } { \alpha } \left( 1 - \lambda + \frac { \lambda } { 1 - \beta } \right) .\tag{13}
$$

Then $Y _ { \eta }$ takes values in $[ e ^ { - \xi / 2 } , 1 ]$ . Note that $V _ { t , \eta } ^ { * } = a _ { \eta } - \alpha \log \mathbb { E } _ { q _ { 0 | t } } Y _ { \eta } ( x _ { 0 } )$

Let $\mathcal { F } _ { t }$ be the class of linear regressors with F fixed features bounded by one, coefficients in $[ - 1 , 1 ] .$ and outputs clipped to $[ e ^ { - \xi / 2 } , 1 ]$ . For convenience of analysis, suppose that the ground-truth regression function $\mathbb { E } [ Y _ { \eta } ( X _ { 0 } ) \mid X _ { t } ] \in \mathcal { F } _ { t }$ for every $t , \eta ,$ and we have access to an approximate empirical risk minimizer (ERM) $\widehat { f } _ { t , \eta } \in \mathcal { F } _ { t }$ , from which we construct a value function estimate $\widehat { V } _ { t , \eta , N }$ for $V _ { t , \eta } ^ { * }$

The following theorem provides the sample complexity to establish an error bound between $\widehat { V } _ { t , \eta , N }$ and $V _ { t , \eta } ^ { * }$ that holds uniformly over $\eta \in \left[ l , u \right]$ and $t = \mathrm { \bar { 1 } } , \dots , T$ (and thus any η returned by Algorithm 2).

Theorem 3 (Value estimation from paired samples). Suppose that Assumption 1 holds. Define the loss $\begin{array} { r } { \widehat { \mathcal { L } } _ { t , \eta } ( g ) : = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( g ( x _ { t } ^ { i } ) - Y _ { \eta } ( x _ { 0 } ^ { i } ) \right) ^ { 2 } } \end{array}$ . For any $\delta \in ( 0 , 1 )$ and sufficiently small $\varepsilon _ { v } > 0$ , suppose that we have access to an approximate ERM $\widehat { f } _ { t , \eta } \in \mathcal { F } _ { t }$ such that,for all $t = 1 , \dots , T , \eta \in [ l , u ]$

$$
\widehat { \mathcal { L } } _ { t , \eta } ( \widehat { f } _ { t , \eta } ) \leq \operatorname* { m i n } _ { g \in \mathcal { F } _ { t } } \widehat { \mathcal { L } } _ { t , \eta } ( g ) + \frac { \varepsilon _ { v } ^ { 2 } } { 8 ( 1 + \alpha ^ { 2 } ) } e ^ { - \xi }
$$

Now construct value function estimates with the approximate ERM $\widehat { f } _ { t , \eta }$ as follows:

$$
\widehat { V } _ { t , \eta , N } : = a _ { \eta } - \alpha \log \widehat { f } _ { t , \eta } , \forall t = 1 , \ldots , T .\tag{14}
$$

Then, with probability at least $1 - \delta ,$ we have su $\begin{array} { r } { \mathsf { p } _ { \eta \in [ l , u ] } \operatorname* { m a x } _ { t \in \{ 1 , \ldots , T \} } \left\| \widehat { V } _ { t , \eta , N } - V _ { t , \eta } ^ { * } \right\| _ { L _ { 2 } ( q _ { t } ) } \leq \varepsilon _ { v } , } \end{array}$ as long as

$$
N \geq \frac { C } { \varepsilon _ { v } ^ { 2 } } \log \frac { T } { \delta \varepsilon _ { v } } ,
$$

where $C > 0$ is a constant depending on $\alpha , \beta , \lambda , l , u , F$

The proof is in Appendix H. Similarly, we first use Bernstein’s inequality to obtain a pointwise bound on the difference between the population excess loss and its empirical counterpart for a fixed predictor $g \in \mathcal { F } _ { t }$ and a fixed threshold η. Then, we discretize both the function class $\mathcal { F } _ { t }$ and the η-space and take a union bound to turn this pointwise bound into a uniform one. Together with the realizability assumption and the near-optimality of the ERM, this yields the final sample complexity. Note that Algorithm 1, Algorithm 2 and their theoretical guarantees can also be applied to risk-neutral inference-time alignment by setting $\lambda = 0$

## 5.3 End-to-End Sampling Guarantees

We now combine the threshold optimization and value estimation guarantees to provide an end-toend guarantee for the sampling process. Specifically, for any $\eta ,$ let $p _ { \eta } ^ { \ast }$ and $\hat { p } _ { \eta }$ denote the terminal marginal distributions induced by the exact and estimated value functions, respectively. We provide a high-probability bound (over the training samples) for the total variation distance between the sampler’s output distribution $\hat { p } _ { \hat { \eta } }$ (with clipped value estimates) and the regularized target $p _ { \eta _ { \kappa } ^ { * } } ^ { * }$

Theorem 4. Suppose that the assumptions of Theorems 2 and 3 hold. Recall from (13) that $\begin{array} { l l } { \displaystyle { \lvert \xi \ } } & { \displaystyle { : = } } \end{array}$ $\begin{array} { r } { \frac { 2 ( u - l ) } { \alpha } \left( 1 - \lambda + \frac { \lambda } { 1 - \beta } \right) } \end{array}$ . Then, under exact tilted-transition sampling, using Algorithm 2 with the hyperparameters chosen as in Theorem $^ { 2 , }$ and using the value estimator according to (14), we have, with probability at least $1 - \delta$

$$
\mathrm { T V } ( p _ { \eta _ { \kappa } ^ { * } } ^ { * } ( x _ { 0 } ) , \hat { p } _ { \hat { \eta } } ( x _ { 0 } ) ) \leq \frac { e ^ { \xi / 2 } } { 2 \alpha } \sqrt { T } \varepsilon _ { v } + \frac { \lambda \varepsilon _ { \eta } } { 4 \alpha ( 1 - \beta ) } .
$$

The proof is in Appendix I. The two terms in Theorem 4 quantify distinct sources of error. The first arises from approximating the value functions used to tilt the reverse transitions (Theorem 3). These errors accumulate across the denoising steps, yielding the $\sqrt { T } \varepsilon _ { v }$ dependence in TV distance. The second arises from using the estimated threshold ηˆ instead of $\eta _ { \kappa } ^ { * }$ (Theorem 2). It controls the change in the terminal tilted distribution and is independent of $T$ . Thus, for fixed problem parameters and T, both contributions vanish as the training sample size increases.

## 6 Numerical Experiments

We numerically validate our risk-averse inference-time alignment framework, including the sampling (Algorithm 1) and threshold optimization (Algorithm 2) procedures, on synthetic data in Section 6.1 and on real-world image data in Section 6.2, respectively.

## 6.1 Synthetic Data

We first evaluate whether CVaR risk-control reduces high-cost outcomes on a two-dimensional mixture of rotated product-Cauchy distributions (see (39)). This heavy-tailed example is evaluated using a bounded quadratic terminal cost (see (40)). We compare our risk-averse CVaR-SVDD sampler against two baselines: (i) the frozen base diffusion sampler, without any inference-time guidance, and (ii) vanilla risk-neutral SVDD, with $\lambda = 0 .$ . For CVaR-SVDD, we use the pure tail-risk objective with $\lambda = 1$ , with the same hyperparameters as in vanilla SVDD whenever applicable.

Table 1: Sampling performance on synthetic Cauchy data. Entries are mean ± standard error across ten seeds. All metrics are costs, so smaller values are better. “0 observed” indicates no sample exceeded $\tau _ { 8 0 }$
<table><tr><td>Method</td><td> $\mathbf { M e a n C o s t \downarrow }$ </td><td>Empirical CVaR↓</td><td> $\hat { \mathbb { P } } ( C > \tau _ { 8 0 } )$   $( \% ) \downarrow$ </td></tr><tr><td>Base</td><td> $2 . 3 7 3 \pm 0 . 1 0 9$ </td><td> $8 . 3 7 9 \pm 0 . 5 2 6$ </td><td> $1 9 . 3 7 \pm 0 . 6 2$ </td></tr><tr><td>SVDD [7]</td><td> $0 . 7 0 6 \pm 0 . 0 0 3$ </td><td> $1 . 3 4 3 \pm 0 . 0 1 1$ </td><td> $1 . 9 5 \pm 0 . 2 0$ </td></tr><tr><td>CVaR-SVDD</td><td> $\mathbf { 0 . 5 8 7 \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 1 . 0 1 7 \pm 0 . 0 0 6 }$ </td><td>0 observed</td></tr></table>

We set the CVaR confidence level to $\beta = 0 . 8 ,$ the temperature to $\alpha = 1 .$ , and the regularization strength to $\kappa = 0 . 0 2 .$ . For CVaR-SVDD, we optimize the threshold $\eta$ using Algorithm 2. For value function estimation, we compare linear regressors with neural networks in Figure 7. Preliminary results show that neural-network value estimators produce fewer high-cost outcomes, so we adopt neural networks to estimate value functions in the remaining experiments. Additional experimenta setups are provided in Appendix K.1.1.

Results. Table 1 shows that CVaR-SVDD improves empirical tail performance. Compared with vanilla SVDD, it reduces empirical CVaR from 1.343 to 1.017 and lowers the fraction of samples with costs exceeding $\tau _ { 8 0 } ,$ which is the empirical 80-th percentile estimated from 50000 samples from the base model. The survival curves in Figure 1 further show lower tail probabilities across a broad range of cost thresholds.

Interestingly, Table 1 shows that CVaR-SVDD also reduces the mean cost. This observation is consistent with CVaR penalization shifting probability mass away from high-cost regions, as illustrated in the top panel of Figure 3. However, improvement in the mean depends on the underlying distribution and cost function and is not guaranteed by the CVaR objective. We also observe a similar improvement in the average reward in the real-image experiments in Section 6.2.

![](images/27cd0625e628f323a1f24b43f018517a990daa019ecfa506a5a50f93e4bf30cb.jpg)  
(a) Empirical cost survival functions.

![](images/0213b07b7c8a2b6772c078f916bbd75354def81f1d6e9cb2953765cd803eff38.jpg)  
(b) Paired tail-probability differences (CVaR-SVDD minus vanilla SVDD).  
Figure 1: Synthetic mixed-Cauchy results. Vertical lines mark the 80-th, 90-th, and 95-th cost percentiles estimated from 50000 base-model samples. Shading indicates pointwise 95% confidence intervals.

Ablations. Additional ablation results are provided in Appendix K.1.2. (1) Inference-time sampler: The reduction in high-cost outcomes persists when SVDD is replaced by SMC (see Figure 4). (2) Temperature α: At every tested temperature $\alpha \in \{ 0 . 5 , 1 , 5 , 1 0 \}$ , CVaR-SVDD yields a lower empirical tail probability at $\tau _ { 8 0 }$ than vanilla SVDD (see Figure 5). It also achieves a more favorable trade-off between tail performance and base log-likelihood. (3) Risk parameter λ: Increasing $\lambda \in$ {0, 0.25, 0.5, 0.75, 1} progressively reduces empirical CVaR and tail probabilities, illustrating the gradual transition from risk-neutral to pure-CVaR risk control (see Table 4). (4) Regularization strength κ: We vary κ from 0 to 2 and observe nearly identical tail performance when $\overline { { \kappa \in [ 0 , 0 . 2 ] } }$ (see Figure 6). As a result, we fix $\kappa = 0 . 0 2$ in our default setting.

Threshold optimization. Figure 2 compares Algorithm 2 with uniform-grid search and projected gradient descent. When the number of oracle evaluations increases, Algorithm 2 achieves substantially smaller threshold errors than both baselines, demonstrating its effectiveness (see Figure 8 for more details).

![](images/85734b4ccb657ef4d629e68b67977f12480858492351961424153166ab060480.jpg)  
Figure 2: Absolute threshold error $| { \widehat \eta } - \eta _ { \kappa } ^ { \star } |$ versus the number of oracle evaluations. Gradient descent results are averaged over five randomly sampled learning rates.

## 6.2 Image Data

We evaluate risk-averse inference-time sampling for text-to-image generation using a frozen Stable Diffusion v1.5 model [4]. We consider two terminal rewards: compressibility and aesthetic score [20]. In both cases, we set the terminal cost to $c ( x _ { 0 } ) = - r ( x _ { 0 } )$ . We again compare the base diffusion sampler, SVDD-MC [7], and our CVaR-SVDD with a learned hinge-type value function. See Appendix K.2.1 for experimental details.

Results. Table 2 shows that CVaR-SVDD improves both the mean and lower-tail compressibility reward for cat. The reward distributions and CDFs in Figure 9 further illustrate the reduction in low-reward outcomes. For aesthetics with cat, CVaR-SVDD increases the lower-tail reward while slightly improving the mean. The overall aesthetic distributions remain close, with the modest lower-tail gains most apparent in the enlarged CDF in Figure 10.

Ablations. Additional ablation results are reported in Table 6 and Appendix K.2.2. (1) Prompt: The compressibility gains persist for the dog and rabbit prompts. (2) Inference-time sampler: CVaR-SMC also yields a slightly higher observed lower-tail mean. Appendix K.2.3 further shows randomly selected generated images from the worst-20% tail.

Table 2: Image generation for the cat prompt. Entries are mean ± standard error across five seeds. Both metrics are rewards, so larger values are better.
<table><tr><td>Method Mean Reward ↑ Avg Tail Reward ↑</td></tr><tr><td>Compressibility Base  $- 9 5 . 9 8 6 \pm 0 . 2 2 4$   $- 1 2 3 . 3 7 7 \pm 1 . 0 4 2$ </td></tr><tr><td>SVDD [7]  $- 7 3 . 0 3 4 \pm 0 . 1 2 1$   $- 8 9 . 6 7 1 \pm 0 . 2 9 2$  CVaR-SVDD  $\mathbf { - 6 7 . 1 3 6 \pm 0 . 1 3 1 }$   $\mathbf { - 7 8 . 4 0 5 \pm 0 . 1 3 5 }$ </td></tr><tr><td>Aesthetic score</td></tr><tr><td>Base  $5 . 8 8 4 5 \pm 0 . 0 0 7 6$   $5 . 4 6 0 0 \pm 0 . 0 1 0 5$ </td></tr><tr><td>SVDD [7]  $5 . 9 1 6 5 \pm 0 . 0 0 3 6$   $5 . 5 1 5 4 \pm 0 . 0 0 3 7$  CVaR-SVDD  $\mathbf { 5 . 9 2 8 5 \pm 0 . 0 0 4 1 }$ </td></tr></table>

## 7 Conclusions

In this work, we developed a risk-averse inference-time diffusion alignment framework by combining CVaR and expected cost. By characterizing the optimal sampling policies, we reduced the alignment problem to scalar threshold optimization. Our finite-sample analysis quantifies how errors in threshold optimization and value estimation affect the final output distribution. Synthetic and image-generation experiments further demonstrate improved empirical tail control. Future work includes extending the guarantees to unbounded costs and explicitly accounting for the finite numbers of proposals or particles used by practical samplers.

## References

[1] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33, pages 6840–6851, 2020.

[2] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In The Ninth International Conference on Learning Representations, 2021.

[3] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.

[4] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, June 2022.

[5] Andrew Melnik, Michal Ljubljanac, Cong Lu, Qi Yan, Weiming Ren, and Helge Ritter. Video diffusion models: A survey. arXiv preprint arXiv:2405.03150, 2024.

[6] Masatoshi Uehara, Yulai Zhao, Ehsan Hajiramezanali, Gabriele Scalia, Gökcen Eraslan, Avantika Lal, Sergey Levine, and Tommaso Biancalani. Bridging model-based optimization and generative modeling via conservative fine-tuning of diffusion models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

[7] Xiner Li, Yulai Zhao, Chenyu Wang, Gabriele Scalia, Gökcen Eraslan, Surag Nair, Tommaso Biancalani, Shuiwang Ji, Aviv Regev, Sergey Levine, and Masatoshi Uehara. Derivative-free guidance in continuous and discrete diffusion models with soft value-based decoding. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[8] Raghav Singhal, Zachary Horvitz, Ryan Teehan, Mengye Ren, Zhou Yu, Kathleen McKeown, and Rajesh Ranganath. A general framework for inference-time scaling and steering of diffusion models. In Forty-second International Conference on Machine Learning, 2025.

[9] Zifan Wang, Riccardo De Santi, Xiaoyu Mo, Michael M. Zavlanos, Andreas Krause, and Karl Henrik Johansson. Efficient tail-aware generative optimization via flow model fine-tuning. In Forty-third International Conference on Machine Learning, 2026.

[10] Wei Xiao, Tsun-Hsuan Wang, Chuang Gan, Ramin Hasani, Mathias Lechner, and Daniela Rus. Safediffuser: Safe planning with diffusion probabilistic models. In The Thirteenth International Conference on Learning Representations, 2025.

[11] Hao Ma, Sabrina Bodmer, Andrea Carron, Melanie Zeilinger, and Michael Muehlebach. Constraint-aware diffusion guidance for robotics: Real-time obstacle avoidance for autonomous racing. In 9th Annual Conference on Robot Learning, 2025.

[12] R Tyrrell Rockafellar, Stanislav Uryasev, et al. Optimization of conditional value-at-risk. Journal of Risk, 2(3):21–42, 2000.

[13] R Tyrrell Rockafellar and Stanislav Uryasev. Conditional value-at-risk for general loss distributions. Journal of Banking & Finance, 26(7):1443–1471, 2002.

[14] Riccardo De Santi, Marin Vlastelica, Ya-Ping Hsieh, Zebang Shen, Niao He, and Andreas Krause. Flow density control: Generative optimization beyond entropy-regularized fine-tuning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[15] Shiqiang Gong. Rs-diffuser: Risk-sensitive diffusion planning with distributional value guidance. arXiv preprint arXiv:2606.27766, 2026.

[16] Sunil Madhow, Yuchen Liang, Ness Shroff, Yingbin Liang, and Yu-Xiang Wang. Learnable chernoff baselines for provable inference-time alignment. In The Fortieth Annual Conference on Neural Information Processing Systems, 2026.

[17] Qijie Zhu, Zeqi Ye, Han Liu, Zhaoran Wang, and Minshuo Chen. Training-free adaptation of diffusion models via doob’s \$h\$-transform. In Forty-third International Conference on Machine Learning, 2026.

[18] Youheng Zhu and Yiping Lu. On the power of (approximate) reward models for inference-time scaling. arXiv preprint arXiv:2602.01381, 2026.

[19] Bradley Efron. Tweedie’s formula and selection bias. Journal of the American Statistical Association, 106(496):1602–1614, 2011.

[20] Chenyu Wang, Masatoshi Uehara, Yichun He, Amy Wang, Avantika Lal, Tommi Jaakkola, Sergey Levine, Aviv Regev, Hanchen, and Tommaso Biancalani. Fine-tuning discrete diffusion models via reward optimization with applications to DNA and protein design. In The Thirteenth International Conference on Learning Representations, 2025.

[21] Michael B Cohen, Yin Tat Lee, Gary Miller, Jakub Pachocki, and Aaron Sidford. Geometric median in nearly linear time. In Proceedings of the forty-eighth annual ACM symposium on Theory of Computing, pages 9–21, 2016.

[22] Masatoshi Uehara, Yulai Zhao, Tommaso Biancalani, and Sergey Levine. Understanding reinforcement learning-based fine-tuning of diffusion models: A tutorial and review. arXiv preprint arXiv:2407.13734, 2024.

[23] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2024.

[24] Kevin Clark, Paul Vicol, Kevin Swersky, and David J Fleet. Directly fine-tuning diffusion models on differentiable rewards. arXiv preprint arXiv:2309.17400, 2024.

[25] Masatoshi Uehara, Yulai Zhao, Chenyu Wang, Xiner Li, Aviv Regev, Sergey Levine, and Tommaso Biancalani. Inference-time alignment in diffusion models with reward-guided generation: Tutorial and review. arXiv preprint arXiv:2501.09685, 2025.

[26] Hyungjin Chung, Jeongsol Kim, Michael Thompson Mccann, Marc Louis Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. In The Eleventh International Conference on Learning Representations, 2023.

[27] Yuchen Liang, Peizhong Ju, Yingbin Liang, and Ness Shroff. Theory on score-mismatched diffusion models and zero-shot conditional samplers. In The Thirteenth International Conference on Learning Representations, 2025.

[28] Jiaming Song, Arash Vahdat, Morteza Mardani, and Jan Kautz. Pseudoinverse-guided diffusion models for inverse problems. In International Conference on Learning Representations, 2023.

[29] Luhuan Wu, Brian L. Trippe, Christian A Naesseth, John Patrick Cunningham, and David Blei. Practical and asymptotically exact conditional sampling in diffusion models. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.

[30] Bingliang Zhang, Wenda Chu, Julius Berner, Chenlin Meng, Anima Anandkumar, and Yang Song. Improving diffusion inverse problem solving with decoupled noise annealing. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[31] Jacob K Christopher, Stephen Baek, and Ferdinando Fioretto. Constrained synthesis with projected diffusion models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

[32] Sai Shankar Narasimhan, Shubhankar Agarwal, Litu Rout, Sanjay Shakkottai, and Sandeep P. Chinchali. Constrained posterior sampling: Time series generation with hard constraints. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[33] András Prékopa. On logarithmic concave measures and functions. Acta Sci. Math., 34:335, 1973.

[34] Pierre Del Moral, Arnaud Doucet, and Ajay Jasra. Sequential Monte Carlo samplers. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 2006.

[35] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models. arXiv preprint arXiv:2305.16381, 2023.

[36] Christoph Schuhmann. LAION-Aesthetics Predictor V2. https://github.com/ christophschuhmann/improved-aesthetic-predictor, 2022. GitHub repository.

[37] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021.

## Appendix

A Additional Related Work 17   
B Vanilla Inference-time Alignment Algorithms 17   
C Proof of Theorem 1 17   
D Proof of Proposition 1 and An Example for Non-Convexity 19   
E Proof of Proposition 2 21   
F Corollary 1 and its proof 21   
G Proof of Theorem 2 22   
H Proof of Theorem 3 24   
I Proof of Theorem 4 26   
J Proofs of Auxiliary Lemmas 30   
J.1 Proof of Lemma 2 . 30   
J.2 Proof of Lemma 3 30   
J.3 Proof of Lemma 4 31   
J.4 Proof of Lemma 6 . 31   
K Experimental Result Details 32   
K.1 Additional details for synthetic experiments 32   
K.2 Additional details for image experiments 40

## A Additional Related Work

We have provided the most relevant works in Section 1.2. As follows, we provide all the additional related work.

Reward-based fine-tuning. One way to align diffusion models for downstream task is rewardbased fine-tuning, which adapts pretrained diffusion models by updating their model weights to improve downstream objectives (see [22] for a survey). For example, Denoising Diffusion Policy Optimization (DDPO) [23] formulates denoising as a sequential decision-making problem and applies policy-gradient methods to optimize downstream rewards, and Direct Reward Fine-Tuning (DRaFT) [24] backpropagates differentiable rewards through the sampling procedure, with variants that truncate backpropagation to reduce computation. These approaches incorporate alignment into the generative model’s parameters.

Inference-time alignment. Different from fine-tuning, inference-time alignment methods steer the diffusion generation process toward optimizing downstream objectives while keeping the pretrained generative model frozen (see [25] for a survey). For example, Soft Value-Based Decoding (SVDD) [7] uses soft value estimates to select intermediate candidates based on their expected future desirability, and Feynman-Kac steering [8] provides a particle-based framework that guides generation through intermediate weighting and resampling. Their standard formulations target reward-tilted distributions without explicitly optimizing the CVaR of the output distribution. In contrast, our framework incorporates the mean-CVaR interpolated objective into this sampling perspective through a threshold-dependent terminal cost, enabling CVaR extensions of existing inference-time samplers.

Inference-time conditional and constrained sampling. Diffusion alignment is closely related to conditional and constrained sampling, for which numerous inference-time methods have been developed. These include methods for zero-shot conditional sampling [26, 27, 28, 29, 30]. A related line incorporates explicit feasibility constraints into the sampling process [31, 32]. Indeed, conditional sampling can be formulated as reward-guided alignment by using the conditional log-likelihood as the reward [25]. This connection suggests potential applications of our mean-CVaR framework to controlling the upper tail of negative log-likelihoods or constraint-violation costs.

## B Vanilla Inference-time Alignment Algorithms

In this section, we review two risk-neutral inference-time alignment algorithms: the SVDD (Algorithm 3) and the SMC (Algorithm 4) sampling algorithm. Both of them can be used within our CVaR framework.

## C Proof of Theorem 1

Recall the tail-controlled objective in (6):

$$
\operatorname* { m i n } _ { \pi , \eta } \left\{ \mathbb { E } _ { \pi } \left[ c _ { \lambda , \eta } ( x _ { 0 } ) \right] + \alpha \mathrm { K L } ( \pi _ { 1 : T + 1 } | | \mu _ { 1 : T + 1 } ) \right\} .
$$

For a fixed $\eta ,$ the integrability condition guarantees all expectations below are finite. Fix $t \in$ $\{ 1 , \ldots , T + 1 \}$ . Notice that the inner problem has the same entropy-regularized objective form as (1). Recall the variational definition of the risk-adjusted value function:

$$
V _ { t , \eta } ^ { * } ( x _ { t } ) : = \operatorname* { m i n } _ { \pi _ { 1 : t } } \mathbb { E } _ { \pi } \left[ c _ { \lambda , \eta } ( x _ { 0 } ) + \alpha \sum _ { k = 1 } ^ { t } \mathrm { K L } ( \pi _ { k } | | \mu _ { k } ) \bigg | x _ { t } \right] .\tag{15}
$$

Algorithm 3: SVDD Sampling [7]   
Input: Pretrained reverse kernels $\{ \mu _ { t } ( x _ { t - 1 } \mid x _ { t } ) \} _ { t = 1 } ^ { T + 1 }$ , value estimates $\{ \widehat { V } _ { t } \} _ { t = 0 } ^ { T }$ (with $\widehat V _ { 0 } ( \cdot ) : = c ( \cdot ) ) ,$   
number of samples $N ,$ number of candidates $M _ { ☉ }$ , temperature $\alpha > 0$   
Obtain N initial dummy variables $x _ { T + 1 } ^ { ( 1 ) } , \dots , x _ { T + 1 } ^ { ( N ) }$   
for $t = T + 1 , T , \dots , 1$ do   
for $i = 1 , \ldots , N$ do   
Sample M candidates independently: $z _ { t , m } ^ { ( i ) } \sim \mu _ { t } ( \cdot \mid x _ { t } ^ { ( i ) } )$ $m = 1 , \ldots , M .$   
Calculate the normalized weights:   
w¯<sub>t,m</sub> (i) = exp −Vb<sub>t−1</sub>(z<sup>(i)</sup><sub>t,m</sub>)/α $m = 1 , \ldots , M .$   
$\begin{array} { r } { \overline { { \sum _ { j = 1 } ^ { M } \exp \left( - \widehat { V } _ { t - 1 } ( z _ { t , j } ^ { ( i ) } ) / \alpha \right) } } } \end{array}$   
Sample $a _ { t } ^ { ( i ) } \sim$ Categorical $( \bar { w } _ { t , 1 } ^ { ( i ) } , \dots , \bar { w } _ { t , M } ^ { ( i ) } )$   
Set $x _ { t - 1 } ^ { ( i ) } = z _ { t , a _ { t } ^ { ( i ) } } ^ { ( i ) }$   
end   
end   
Output: $\{ x _ { 0 } ^ { ( i ) } \} _ { i = 1 } ^ { N }$

Algorithm 4: SMC Sampling [8]   
Input: Pretrained reverse kernels $\{ \mu _ { t } ( x _ { t - 1 } \mid x _ { t } ) \} _ { t = 1 } ^ { T + 1 }$ , value estimates $\{ \widehat { V } _ { t } \} _ { t = 0 } ^ { T + 1 }$ (with $\widehat V _ { 0 } ( \cdot ) : = c ( \cdot ) ) .$   
number of particles $N ,$ temperature $\alpha > 0 ,$ ESS threshold $\gamma \in ( 0 , 1 ]$   
Obtain N initial dummy variables $x _ { T + 1 } ^ { ( 1 ) } , \dots , x _ { T + 1 } ^ { ( N ) } ;$   
Let $\ell ^ { ( i ) } = 0$ and $\phi ^ { ( i ) } = - \widehat { V } _ { T + 1 } / \alpha$ for all $i = 1 , \ldots , N ;$   
for $t = T + 1 , T , \dots , 1$ do   
Sample $\widetilde { \boldsymbol { x } } _ { t - 1 } ^ { ( i ) } \sim \mu _ { t } ( \cdot \mid \boldsymbol { x } _ { t } ^ { ( i ) } ) , \forall i = 1 , \dots , N$   
for $i = 1 , . . . , N$ do   
Define $\begin{array} { r } { \widetilde { \phi } ^ { ( i ) } : = \frac { - \widehat { V } _ { t - 1 } ( \widetilde { x } _ { t - 1 } ^ { ( i ) } ) } { \alpha } } \end{array}$   
Update the accumulated log-weight: $\ell ^ { ( i ) } = \ell ^ { ( i ) } + \widetilde { \phi } ^ { ( i ) } - \phi ^ { ( i ) }$   
end   
Calculate weight $\begin{array} { r } { \bar { w } ^ { ( i ) } = \frac { \exp \left( \ell ^ { ( i ) } \right) } { \sum _ { j = 1 } ^ { N } \exp \left( \ell ^ { ( j ) } \right) } } \end{array}$ for each $i = 1 , \ldots , N$ and $\begin{array} { r } { \mathrm { E S S } = \frac { 1 } { \sum _ { i = 1 } ^ { N } \left( \bar { w } ^ { ( i ) } \right) ^ { 2 } } } \end{array}$   
if $t > 1$ and $\mathrm { E S S } < \gamma N$ then   
Sample ancestor indices $a _ { 1 } , \dots , a _ { N }$ independently from Categorical $( \bar { w } ^ { ( 1 ) } , \dots , \bar { w } ^ { ( N ) } )$   
Perform resampling: $\mathrm { i . e . , }$ for each $i = 1 , \ldots , N ,$ update $x _ { t - 1 } ^ { ( i ) } = \widetilde { x } _ { t - 1 } ^ { ( a _ { i } ) } , \ \phi ^ { ( i ) } = \widetilde { \phi } ^ { ( a _ { i } ) }$   
Reset $\ell ^ { ( i ) } = 0$ for all $i = 1 , \ldots , N$   
else   
Keep $x _ { t - 1 } ^ { ( i ) } = \widetilde { x } _ { t - 1 } ^ { ( i ) }$ and $\phi ^ { ( i ) } = \widetilde { \phi } ^ { ( i ) }$ for each $i = 1 , \ldots , N .$   
end   
end   
Calculate $\bar { w } ^ { ( i ) }$ and perform resampling again.   
Output: $\{ x _ { 0 } ^ { ( i ) } \} _ { i = 1 } ^ { N }$

We can also define the corresponding Q-function:

$$
Q _ { t , \eta } ^ { * } ( x _ { t } , x _ { t - 1 } ) : = \operatorname* { m i n } _ { \pi _ { 1 : t - 1 } } \mathbb { E } _ { \pi } \left[ c _ { \lambda , \eta } ( x _ { 0 } ) + \alpha \sum _ { k = 1 } ^ { t - 1 } { \mathrm { K L } } ( \pi _ { k } | | \mu _ { k } ) \bigg | x _ { t } , x _ { t - 1 } \right] .
$$

Once $x _ { t - 1 }$ is fixed, the remaining optimization depends only on $x _ { t - 1 }$ by the Markov property. Hence,

$$
Q _ { t , \eta } ^ { \ast } ( x _ { t } , x _ { t - 1 } ) = V _ { t - 1 , \eta } ^ { \ast } ( x _ { t - 1 } ) .
$$

With this setup, the optimization in (15) becomes

$$
\operatorname* { m i n } _ { \pi _ { t } } \mathbb { E } _ { \pi _ { t } } \left[ V _ { t - 1 , \eta } ^ { * } ( x _ { t - 1 } ) \Big \vert x _ { t } \right] + \alpha \mathrm { K L } ( \pi _ { t } | | \mu _ { t } ) ,\tag{16}
$$

and thus, taking the derivative and setting it to zero, we get the optimal transition at time $t { \mathrm { ~ a s ~ } }$

$$
\pi _ { t , \eta } ^ { * } ( x _ { t - 1 } \mid x _ { t } ) \propto \mu _ { t } ( x _ { t - 1 } \mid x _ { t } ) e ^ { - Q _ { t , \eta } ^ { * } ( x _ { t } , x _ { t - 1 } ) / \alpha } = \mu _ { t } ( x _ { t - 1 } \mid x _ { t } ) e ^ { - V _ { t - 1 , \eta } ^ { * } ( x _ { t - 1 } ) / \alpha } .
$$

Plugging $\pi _ { t , \eta } ^ { * }$ back into (16), we have the following recursion:

$$
\begin{array} { r } { V _ { t , \eta } ^ { * } ( x _ { t } ) = - \alpha \log \mathbb { E } _ { \mu _ { t } } \left[ e ^ { - V _ { t - 1 , \eta } ^ { * } ( x _ { t - 1 } ) / \alpha } \Big | x _ { t } \right] \iff e ^ { - V _ { t , \eta } ^ { * } ( x _ { t } ) / \alpha } = \mathbb { E } _ { \mu _ { t } } \left[ e ^ { - V _ { t - 1 , \eta } ^ { * } ( x _ { t - 1 } ) / \alpha } \Big | x _ { t } \right] , } \end{array}
$$

with $V _ { 0 , \eta } ^ { * } ( x _ { 0 } ) = c _ { \lambda , \eta } ( x _ { 0 } )$ . Consequently,

$$
V _ { t , \eta } ^ { * } ( x _ { t } ) = - \alpha \log \mathbb { E } _ { q _ { 0 \mid t } } \left[ e ^ { - c _ { \lambda , \eta } ( x _ { 0 } ) / \alpha } \Big | x _ { t } \right] .
$$

The proof is now complete.

## D Proof of Proposition 1 and An Example for Non-Convexity

Proof of Proposition 1. When $\lambda = 0 , G _ { 0 }$ is constant in η and thus convex. Note that $A _ { 0 }$ and $B _ { 0 }$ are still well defined, and the criterion always holds when $\lambda = 0$ . As follows, we focus on the case where $\lambda > 0$

Recall from (7) that

$$
c _ { \lambda , \eta } ( x _ { 0 } ) = ( 1 - \lambda ) c ( x _ { 0 } ) + \lambda \left( \eta + \frac { 1 } { 1 - \beta } ( c ( x _ { 0 } ) - \eta ) ^ { + } \right) .
$$

Also recall the definitions:

$$
A _ { \lambda } ( \eta ) : = \mathbb { E } _ { q _ { 0 } } \left[ e ^ { - c _ { \lambda , \eta } ( x _ { 0 } ) / \alpha } \mathbb { 1 } \left\{ c ( x _ { 0 } ) \leq \eta \right\} \right] , \quad B _ { \lambda } ( \eta ) : = \mathbb { E } _ { q _ { 0 } } \left[ e ^ { - c _ { \lambda , \eta } ( x _ { 0 } ) / \alpha } \mathbb { 1 } \left\{ c ( x _ { 0 } ) > \eta \right\} \right] .
$$

Let

$$
Z _ { \lambda } ( \eta ) = \mathbb { E } _ { q _ { 0 } } \left[ e ^ { - c _ { \lambda , \eta } ( \boldsymbol { x } _ { 0 } ) / \alpha } \right] , \qquad G _ { \lambda } ( \eta ) = - \alpha \log Z _ { \lambda } ( \eta ) .
$$

Writing $C : = c ( x _ { 0 } )$ and splitting the expectation at $C = \eta ,$ we have

$$
\begin{array} { r l r } {  { Z _ { \lambda } ( \eta ) = A _ { \lambda } ( \eta ) + B _ { \lambda } ( \eta ) } } \\ & { } & { = \int _ { - \infty } ^ { \eta } \exp ( - \frac { ( 1 - \lambda ) C + \lambda \eta } { \alpha } ) h ( C ) d C + \int _ { \eta } ^ { \infty } \exp ( - \frac { ( 1 - \lambda ) C + \lambda \eta + \frac { \lambda } { 1 - \beta } ( C - \eta ) } { \alpha } ) h ( C ) d C . } \end{array}\tag{17}
$$

Since $c ( x _ { 0 } )$ has continuous density $h ,$ differentiating the first term in (17) yields

$$
\frac { d } { d \eta } A _ { \lambda } ( \eta )
$$

$$
\begin{array} { r l r } {  { = \frac { d } { d \eta } \mathbb { E } _ { q _ { \flat } } [ \exp ( - \frac { ( 1 - \lambda ) e ( x _ { 0 } ) + \lambda \eta } { \alpha } ) \mathbb { 1 } \{ c ( x _ { 0 } ) \le \eta \} ] } } \\ & { = \mathbb { E } _ { \phi _ { 0 } } [ ( \exp ( - \frac { ( 1 - \lambda ) e ( x _ { 0 } ) + \lambda \eta } { \alpha } ) ) ^ { \prime } \mathbb { 1 } \{ c ( x _ { 0 } ) \le \eta \} ] + \mathbb { E } _ { q _ { \flat } } [ \exp ( - \frac { ( 1 - \lambda ) e ( x _ { 0 } ) + \lambda \eta } { \alpha } ) ( \mathbb { 1 } \{ c ( x _ { 0 } ) \le \eta \} ) ^ { \prime } ] } \\ & { = - \frac { \lambda } { \alpha } \mathbb { E } _ { q _ { \flat } } [ \exp ( - \frac { ( 1 - \lambda ) e ( x _ { 0 } ) + \lambda \eta } { \alpha } ) \mathbb { 1 } \{ c ( x _ { 0 } ) \le \eta \} ] + e ^ { - \eta / \alpha } h ( \eta ) } \\ & { = - \frac { \lambda } { \alpha } A _ { \lambda } ( \eta ) + e ^ { - \eta / \alpha } h ( \eta ) , } & { ( 1 8 ) } \end{array}
$$

where the derivative of the indicator function follows again by the sifting property of the delta function.

Similarly, differentiating the second term in (17) gives

$$
\begin{array} { r l } & { \frac { d } { d \eta } B _ { \alpha } ( u ) } \\ & { = \frac { \lambda } { d \eta } \frac { 1 } { \mathcal { Z } _ { 0 } } \Bigg [ \exp \Bigg ( - \frac { ( 1 - \lambda ) \epsilon ( z _ { 0 } ) + \lambda \eta + \frac { \lambda } { 1 - \mathcal { S } } ( \epsilon ( z _ { 0 } ) - \eta ) } { \alpha } \Bigg ) \textbf { 1 } \{ c ( x _ { 0 } ) > \eta \Bigg \} \Bigg ] } \\ & { - \frac { \mathbb { Z } _ { 0 } } { \mathcal { Z } _ { 0 } } \Bigg [ \Bigg ( \exp \Bigg ( - \frac { ( 1 - \lambda ) \epsilon ( z _ { 0 } ) + \lambda \eta + \frac { \lambda } { \epsilon } \lambda _ { 0 } ^ { - } \epsilon ( z _ { 0 } ) - \eta ) } { \alpha } \Bigg ) \Bigg ) ^ { \epsilon } \textbf { 1 } \{ c ( z _ { 0 } ) > \eta \Bigg \} \Bigg ] } \\ & { \quad + \frac { \lambda } { \mathcal { Z } _ { 0 } } \Bigg [ \exp \Bigg ( - \frac { ( 1 - \lambda ) \epsilon ( z _ { 0 } ) + \lambda \eta + \frac { \lambda } { \epsilon } \lambda _ { 0 } ^ { - } \epsilon ( z _ { 0 } ) - \eta ) } { \alpha } \Bigg ) \textbf { 1 } \{ c ( x _ { 0 } ) > \eta \} ^ { \epsilon } \Bigg ] } \\ & { = \frac { \lambda \beta } { \alpha ( 1 - \beta ) } \mathbb { J } _ { 0 } \Bigg [ \exp \Bigg ( - \frac { ( 1 - \lambda ) \epsilon ( z _ { 0 } ) + \lambda \eta + \frac { \lambda } { 1 - \mathcal { S } } ( \epsilon ( z _ { 0 } ) - \eta ) } { \alpha } \Bigg ) \textbf { 1 } \{ c ( x _ { 0 } ) > \eta \Bigg \} - \epsilon ^ { - \eta / \alpha } h ( z _ { 0 } ) \Bigg ] } \\ & { = \frac { \lambda \beta } { \alpha ( 1 - \beta ) } \beta _ { \mathcal { S } } ( \exp \Bigg ( - \frac { ( 1 - \lambda ) \epsilon ( z _ { 0 } ) } { \epsilon } ) - \frac { \lambda } { \alpha } } \\ &  = \frac { \lambda \beta }  \alpha (  \end{array}\tag{19}
$$

Adding (18) and (19), the two boundary terms cancel, and hence

$$
Z _ { \lambda } ^ { \prime } ( \eta ) = - \frac { \lambda } { \alpha } A _ { \lambda } ( \eta ) + \frac { \lambda \beta } { \alpha ( 1 - \beta ) } B _ { \lambda } ( \eta ) .\tag{20}
$$

Differentiating (20) and using (18) and (19) yields

$$
\begin{array} { l } { { \displaystyle Z _ { \lambda } ^ { \prime \prime } ( \eta ) } } \\ { { \displaystyle = - \frac { \lambda } { \alpha } \left( - \frac { \lambda } { \alpha } A _ { \lambda } ( \eta ) + e ^ { - \eta / \alpha } h ( \eta ) \right) + \frac { \lambda \beta } { \alpha ( 1 - \beta ) } \left( \frac { \lambda \beta } { \alpha ( 1 - \beta ) } B _ { \lambda } ( \eta ) - e ^ { - \eta / \alpha } h ( \eta ) \right) } } \\ { { \displaystyle = \frac { \lambda ^ { 2 } } { \alpha ^ { 2 } } A _ { \lambda } ( \eta ) + \frac { \lambda ^ { 2 } \beta ^ { 2 } } { \alpha ^ { 2 } ( 1 - \beta ) ^ { 2 } } B _ { \lambda } ( \eta ) - \frac { \lambda } { \alpha ( 1 - \beta ) } e ^ { - \eta / \alpha } h ( \eta ) . } } \end{array}\tag{21}
$$

Now, since $G _ { \lambda } ( \eta ) = - \alpha \log Z _ { \lambda } ( \eta )$ , we have

$$
G _ { \lambda } ^ { \prime } ( \eta ) = - \alpha \frac { Z _ { \lambda } ^ { \prime } ( \eta ) } { Z _ { \lambda } ( \eta ) } = \lambda - \frac { \lambda } { 1 - \beta } \cdot \frac { B _ { \lambda } ( \eta ) } { Z _ { \lambda } ( \eta ) }
$$

and

$$
G _ { \lambda } ^ { \prime \prime } ( \eta ) = - \alpha \frac { Z _ { \lambda } ^ { \prime \prime } ( \eta ) Z _ { \lambda } ( \eta ) - ( Z _ { \lambda } ^ { \prime } ( \eta ) ) ^ { 2 } } { Z _ { \lambda } ( \eta ) ^ { 2 } } .\tag{22}
$$

Substituting (20) and (21) into (22) and simplifying yields

$$
G _ { \lambda } ^ { \prime \prime } ( \eta ) = \frac { \lambda } { ( 1 - \beta ) Z _ { \lambda } ( \eta ) ^ { 2 } } \left( e ^ { - \eta / \alpha } h ( \eta ) Z _ { \lambda } ( \eta ) - \frac { \lambda } { \alpha ( 1 - \beta ) } A _ { \lambda } ( \eta ) B _ { \lambda } ( \eta ) \right) .\tag{23}
$$

Note that here the coefficient $\frac { \lambda } { ( 1 - \beta ) Z _ { \lambda } ( \eta ) ^ { 2 } }$ is strictly positive. Hence, $G _ { \lambda } ( \eta )$ is convex if and only if

$$
e ^ { - \eta / \alpha } h ( \eta ) Z _ { \lambda } ( \eta ) \geq \frac { \lambda } { \alpha ( 1 - \beta ) } A _ { \lambda } ( \eta ) B _ { \lambda } ( \eta ) , \quad \forall \eta \in \mathbb { R } ,
$$

which is exactly (10). Before we end the proof, we also note that, since the first term in (23) is nonnegative,

$$
G _ { \lambda } ^ { \prime \prime } ( \eta ) \geq - \frac { \lambda ^ { 2 } } { \alpha ( 1 - \beta ) ^ { 2 } } \frac { A _ { \lambda } ( \eta ) B _ { \lambda } ( \eta ) } { Z _ { \lambda } ( \eta ) ^ { 2 } } \geq - \frac { \lambda ^ { 2 } } { 4 \alpha ( 1 - \beta ) ^ { 2 } } ,\tag{24}
$$

since $0 \leq ( A - B ) ^ { 2 } = ( A + B ) ^ { 2 } - 4 A B$ for $A , B \geq 0$ . The proof is now complete.

Next, we provide a simple sufficient condition for non-convexity. Suppose that, for some $\bar { \eta } \in \mathbb { R } ,$

$$
h ( \bar { \eta } ) = 0 , \qquad 0 < \mathbb { P } _ { q _ { 0 } } \left\{ c ( x _ { 0 } ) \le \bar { \eta } \right\} < 1 .\tag{25}
$$

Intuitively, (25) means that there exists an interior splitting point of the distribution. This holds, for example, when the density h is bimodal with a zero-probability belt in the midst. Note that this is only a sufficient condition rather than necessary. Then, at $\eta = \bar { \eta } ,$ the l.h.s. of (10) becomes $0 ,$ whereas the r.h.s. remains positive. Thus, $G _ { \lambda }$ is not convex.

## E Proof of Proposition 2

Define

$$
g ( x _ { 0 } , \eta ) : = q _ { 0 } ( x _ { 0 } ) e ^ { - c _ { \lambda , \eta } ( x _ { 0 } ) / \alpha } , \quad Z _ { \lambda } ( \eta ) : = \mathbb { E } _ { x _ { 0 } \sim q _ { 0 } } \left[ e ^ { - c _ { \lambda , \eta } ( x _ { 0 } ) / \alpha } \right] = \int _ { \mathcal { X } } g ( x _ { 0 } , \eta ) \mathrm { d } x _ { 0 } .
$$

From the definition of $c _ { \lambda , \eta }$ in $( 7 )$ , the logarithm of $g ( x _ { 0 } , \eta )$ is

$$
\log g ( x _ { 0 } , \eta ) = \log q _ { 0 } ( x _ { 0 } ) - \frac { 1 } { \alpha } c \lambda , \eta ( x _ { 0 } ) = \log q _ { 0 } ( x _ { 0 } ) - \frac { 1 - \lambda } { \alpha } c ( x _ { 0 } ) - \frac { \lambda } { \alpha } \eta - \frac { \lambda } { \alpha ( 1 - \beta ) } ( c ( x _ { 0 } ) - \eta ) ^ { + } .
$$

Because $c ( x _ { 0 } )$ and $( c ( x _ { 0 } ) - \eta ) ^ { + } = \operatorname* { m a x } \{ c ( x _ { 0 } ) - \eta , 0 \}$ are jointly convex in $( x _ { 0 } , \eta )$ and log $q _ { 0 } ( x _ { 0 } )$ is concave in $( x _ { 0 } , \eta ) , g ( x _ { 0 } , \eta )$ is jointly log-concave in $( x _ { 0 } , \eta )$ . By Prékopa’s Theorem, marginalization preserves log-concavity (see, e.g., Theorem 6 in [33]). Therefore, $\begin{array} { r } { \dot { Z } _ { \lambda } ( \eta ) = \int _ { x _ { 0 } } g ( x _ { 0 } , \breve { \eta } ) \mathrm { d } x _ { 0 } } \end{array}$ is logconcave in η. As a result, the objective function $G _ { \lambda } ( \eta ) = - c$ α log $Z _ { \lambda } ( \eta )$ is convex in $\eta .$ . The proof is now complete.

## F Corollary 1 and its proof

Corollary 1 (Regularized version’s reduction to η optimization). Under the conditions of Theorem $1 , f i x$ $\kappa \geq 0$ and $\eta _ { c } \in \mathbb { R }$ . The quadratically regularized version of (6) is:

$$
\operatorname* { m i n } _ { \pi , \eta } \left\{ \mathbb { E } _ { \pi } \left[ c _ { \lambda , \eta } ( x _ { 0 } ) \right] + \alpha \mathrm { K L } ( \pi _ { 1 : T + 1 } | | \mu _ { 1 : T + 1 } ) + \frac { \kappa } { 2 } ( \eta - \eta _ { c } ) ^ { 2 } \right\} ,
$$

which can be reduced to

$$
\operatorname* { m i n } _ { \eta \in \mathbb { R } } G _ { \lambda , \kappa } ( \eta ) = G _ { \lambda } ( \eta ) + \frac { \kappa } { 2 } ( \eta - \eta _ { c } ) ^ { 2 } ,
$$

as defined in (11). Meanwhile, for any minimizer $\eta _ { \kappa } ^ { * } ,$ , the corresponding optimal reverse transitions satisfy

$$
\pi _ { t , \eta _ { \kappa } ^ { * } } ^ { * } ( x _ { t - 1 } \mid x _ { t } ) \propto \mu _ { t } ( x _ { t - 1 } \mid x _ { t } ) e ^ { - V _ { t - 1 , \eta _ { \kappa } ^ { * } } ^ { * } ( x _ { t - 1 } ) / \alpha } ,
$$

for $t = T + 1 , \ldots , 1$ , where $V _ { t , \eta } ^ { * }$ is given by (8).

ProofofCorollary 1. For fixed $\eta ,$ the regularization term $\mathinner { \frac { \kappa } { 2 } \mathopen { \left( \eta - \eta _ { c } \right) } } ^ { 2 }$ is independent of π. It therefore leaves the minimizing policies in (15) unchanged. By Theorem 1 and (9), the optimal value of the regularized inner problem is

$$
V _ { T + 1 , \eta } ^ { \ast } + \frac { \kappa } { 2 } ( \eta - \eta _ { c } ) ^ { 2 } = G _ { \lambda } ( \eta ) + \frac { \kappa } { 2 } ( \eta - \eta _ { c } ) ^ { 2 } = G _ { \lambda , \kappa } ( \eta ) .
$$

Minimizing over η proves the reduction. Substituting any minimizer $\eta _ { \kappa } ^ { * }$ into the optimal transitions in Theorem 1 gives an optimal policy for the regularized problem. □

## G Proof of Theorem 2

To prove Theorem 2, we first provide a helpful lemma for the ternary search algorithm below, adapted from Lemma 33 of [21].

Lemma 1 (Lemma 33, [21]). Let $g : \mathbb { R }  \mathbb { R }$ be an L-Lipschitz convexfunction defined on the $[ l , u ]$ interval and let $\hat { g } : \mathbb { R } \to \mathbb { R }$ be an oracle such that $| \hat { g } ( y ) - g ( y ) | \leq \tilde { \epsilon }$ for all $y .$ With at most $1 + 2 \lceil \log _ { 3 / 2 } ( \frac { L ( u - l ) } { \tilde { \epsilon } } ) \rceil$ calls to ${ \hat { g } } ,$ the algorithm CVaR-Optimizer with Lipschitz bound L and target error ϵ˜ outputs a point x such that

$$
g ( x ) - \operatorname* { m i n } _ { y \in [ l , u ] } g ( y ) \leq 4 \tilde { \epsilon } .
$$

To invoke Algorithm 2 and apply Lemma 1, we provide the following lemma to characterize the Lipschitz constants for the regularized objectives.

Lemma 2 (Lipschitz continuity of the objectives). Recall the definitions of L and $L _ { \kappa }$ in (12):

$$
L : = \lambda \operatorname* { m a x } \left\{ 1 , \frac { \beta } { 1 - \beta } \right\} , \quad L _ { \kappa } : = L + \kappa ( u - l ) .
$$

Under the assumptions of Theorem 2, define the Monte Carlo estimate of the objective in (9) as

$$
\widehat { G } _ { \lambda , N } ( \eta ) : = - \alpha \log \left( \frac { 1 } { N } \sum _ { i = 1 } ^ { N } e ^ { - c _ { \lambda , \eta } ( x _ { 0 } ^ { i } ) / \alpha } \right) .
$$

Then, we have

1. Both $G _ { \lambda }$ and $\widehat { G } _ { \lambda , N }$ are L-Lipschitz;

2. Both $G _ { \lambda , \kappa }$ and $\widehat { G } _ { \lambda , \kappa , N }$ are $L _ { \kappa } – L$ ipschitz on $[ l , u ] ;$

3. The difference $\widehat { G } _ { \lambda , \kappa , N } - G _ { \lambda , \kappa }$ is 2L-Lipschitz on $[ l , u ]$

Proof. See Appendix J.1.

While Lemma 1 establishes that the distance between the objective function value at the output solution and the optimal objective value can be upper bounded, for the regularized objective function (11), we can further bound the distance between the output ηˆ and the optimal $\eta _ { \kappa } ^ { * }$ using the strongly convex property of $G _ { \lambda , \kappa } ( \eta )$

Lemma 3. Let $\delta \in ( 0 , 1 )$ denote the confidence level, and define $K : = \lceil \log _ { 3 / 2 } ( \frac { 4 L _ { \kappa } ( u - l ) } { \epsilon } ) \rceil$ . Suppose that $G _ { \lambda , \kappa } ( \eta )$ is $m _ { \kappa }$ -strongly convex on $[ l , u ]$ , and suppose we have an inexact objective evaluation oracle $\hat { G } _ { \lambda , \kappa } ( \eta )$ such that, with probability at least $\begin{array} { r } { 1 - \delta , \operatorname* { s u p } _ { \eta \in [ l , u ] } | G _ { \lambda , \kappa } ( \eta ) - \hat { G } _ { \lambda , \kappa } ( \eta ) | \leq \epsilon / 4 } \end{array}$ . Then, with probability at least $1 - \delta$ , the output ηˆ of Algorithm 2 (with Lipschitz bound $L _ { \kappa }$ and target error $\tilde { \epsilon } = \epsilon / 4 )$ is in the neighborhood of the optimal solution $\eta _ { \kappa } ^ { * }$ of the regularized problem (11) $( i . e . , | \hat { \eta } - \eta _ { \kappa } ^ { * } | \leq \sqrt { 2 \epsilon / m _ { \kappa } } )$ by calling the inexact oracle at most $1 + 2 K$ times.

Proof. See Appendix J.2.

For the regularized objective $G _ { \lambda , \kappa } ( \eta )$ , suppose that we have a Monte Carlo estimator $\hat { G } _ { \lambda , \kappa , N } ( \eta ) ;$ $\begin{array} { r } { \hat { G } _ { \lambda , \kappa , N } ( \eta ) = - \alpha \log ( \frac { 1 } { N } \sum _ { i = 1 } ^ { N } e ^ { - c _ { \lambda , \eta } ( x _ { 0 } ^ { i } ) / \alpha } ) + \frac { \kappa } { 2 } ( \eta - \eta _ { c } ) ^ { 2 } } \end{array}$ , where $\{ x _ { 0 } ^ { i } \} _ { i = 1 } ^ { N }$ is a single batch of i.i.d. samples from $q _ { 0 } ,$ used for every η. We next derive the sample complexity of this Monte Carlo estimator, uniformly over η.

Lemma 4. Suppose Assumption 1 holds, and let

$$
\begin{array} { r } { \underline { { y } } : = \exp \Big ( { - \frac { 1 - \lambda } { \alpha } \big ( \mathbb { E } [ c ( X ) ] - l \big ) - \frac { \lambda } { \alpha ( 1 - \beta ) } \mathbb { E } \big [ ( c ( X ) - l ) ^ { + } \big ] } \Big ) , } \end{array}
$$

$$
L : = \lambda \operatorname * { m a x } \Bigl \{ 1 , \frac { \beta } { 1 - \beta } \Bigr \} , \qquad J : = \Bigl \lceil \frac { 1 6 L ( u - l ) } { \epsilon } \Bigr \rceil + 1 .
$$

Suppose $\epsilon \leq 8 \alpha$ . Then $\begin{array} { r } { \operatorname* { s u p } _ { \eta \in [ l , u ] } | \hat { G } _ { \lambda , \kappa , N } ( \eta ) - G _ { \lambda , \kappa } ( \eta ) | \leq \epsilon / 4 } \end{array}$ with probability at least $1 - \delta$ as long as

$$
N \ \geq \ { \frac { C _ { G } } { \epsilon ^ { 2 } } } \log { \frac { 2 J } { \delta } } , \qquad w h e r e \quad C _ { G } : = { \frac { 2 0 4 8 \alpha ^ { 2 } } { 3 \underline { { y } } } } .
$$

Proof. See Appendix J.3.

Theorem 2. Suppose Assumption 1 holds. Let $\eta _ { c } \in [ l , u ]$ be arbitrary, and suppose that $G _ { \lambda , \kappa }$ is $m _ { \kappa }$ -strongly convex on [l, u] for some $m _ { \kappa } > 0$ . Define

$$
L : = \lambda \operatorname* { m a x } \left\{ 1 , \frac { \beta } { 1 - \beta } \right\} , \quad L _ { \kappa } : = L + \kappa ( u - l ) .
$$

For any $\delta \in ( 0 , 1 )$ and sufficiently small $\varepsilon _ { \eta } > 0 ,$ define

$$
K : = \Big \lceil \log _ { 3 / 2 } \Big ( \frac { 8 L _ { \kappa } ( u - l ) } { m _ { \kappa } \varepsilon _ { \eta } ^ { 2 } } \Big ) \Big \rceil , \quad J : = \Big \lceil \frac { 3 2 L ( u - l ) } { m _ { \kappa } \varepsilon _ { \eta } ^ { 2 } } \Big \rceil + 1 .
$$

Then, with probability at least $1 - \delta ,$ the output ηˆ of Algorithm 2 (with Lipschitz bound $L _ { \kappa }$ and target error $m _ { \kappa } \varepsilon _ { \eta } ^ { 2 } / 8 )$ satisfies $| \hat { \eta } - \eta _ { \kappa } ^ { * } | \leq \varepsilon _ { \eta }$ as long as

$$
N \ge \frac { 4 C _ { G } } { m _ { \kappa } ^ { 2 } \varepsilon _ { \eta } ^ { 4 } } \log \frac { 2 J } { \delta }
$$

where $C _ { G } > 0$ is a constant depending on $\alpha , \beta , \lambda , l , u .$ . Furthermore, the algorithm calls the Monte Carlo oracle $\widehat { G } _ { \lambda , \kappa , N }$ at most $1 + 2 K$ times.

Proof. Apply Lemmas 3 and 4 with $\begin{array} { r } { \epsilon : = \frac { m _ { \kappa } \varepsilon _ { \eta } ^ { 2 } } { 2 } } \end{array}$ . Then $\epsilon \leq$ 8α and $\begin{array} { r } { \epsilon / 4 = \frac { m _ { \kappa } \varepsilon _ { \eta } ^ { 2 } } { 8 } } \end{array}$ . Moreover, the quantities K of Lemma 3 and J of Lemma 4 coincide with the K and J defined above, and $\begin{array} { r } { \frac { C _ { G } } { \epsilon ^ { 2 } } = \frac { 4 \bar { C _ { G } } } { m _ { \kappa } ^ { 2 } \varepsilon _ { \eta } ^ { 4 } } } \end{array}$

By Lemma 4, with probability at least $\begin{array} { r } { 1 - \delta , \operatorname* { s u p } _ { \eta \in [ l , u ] } | { \hat { G } } _ { \lambda , \kappa , N } ( \eta ) - G _ { \lambda , \kappa } ( \eta ) | \leq \epsilon / 4 } \end{array}$ . This is exactly the assumption of Lemma 3, with oracle $\hat { G } _ { \lambda , \kappa } = \hat { G } _ { \lambda , \kappa , N }$ . Therefore, with probability at least $1 - \delta ,$ the output satisfies

$$
\lvert \hat { \eta } - \eta _ { \kappa } ^ { * } \rvert \leq \sqrt { \frac { 2 \epsilon } { m _ { \kappa } } } = \varepsilon _ { \eta } ,
$$

using at most $1 + 2 K$ oracle calls.

## H Proof of Theorem 3

First, for every $\eta \in [ l , u ]$ , we have

$$
0 \leq c _ { \lambda , \eta } ( x _ { 0 } ) - a _ { \eta } = ( 1 - \lambda ) ( c ( x _ { 0 } ) - l ) + \frac { \lambda } { 1 - \beta } ( c ( x _ { 0 } ) - \eta ) ^ { + } \leq ( u - l ) ( 1 - \lambda + \frac { \lambda } { 1 - \beta } ) = \frac { \alpha \xi } { 2 } ,\tag{26}
$$

where $\xi$ is defined in (13). As a result, the targets $Y _ { \eta } ,$ their conditional means $\mathbb { E } _ { q _ { 0 | t } } Y _ { \eta } ,$ and all functions in $\mathcal { F } _ { t }$ take values in $[ e ^ { - \frac { \xi } { 2 } } , 1 ]$ . By the mean value theorem, for any $a , b \in [ e ^ { - \frac { \xi } { 2 } } , 1 ]$ , we have $\begin{array} { r } { | \log a - \log b | = \frac { 1 } { \zeta } | a - b | } \end{array}$ for some $\zeta \in [ a , b ]$ . Since $\zeta \geq e ^ { - \frac { \xi } { 2 } } .$ , this gives | log $a - \log b | \leq e ^ { \frac { \xi } { 2 } } | a - b |$ . Thus, using the logarithmic Lipschitz bound above, we obtain

$$
\begin{array} { r } { | \widehat { V } _ { t , \eta , N } ( x _ { t } ) - V _ { t , \eta } ^ { * } ( x _ { t } ) | ^ { 2 } \leq \left| a _ { \eta } - \alpha \log \widehat { f } _ { t , \eta } ( x _ { t } ) - V _ { t , \eta } ^ { \star } ( x _ { t } ) \right| ^ { 2 } \leq \alpha ^ { 2 } e ^ { \xi } | \widehat { f } _ { t , \eta } ( x _ { t } ) - \mathbb { E } _ { q _ { 0 1 } } Y _ { \eta } ( x _ { t } ) | ^ { 2 } , \quad \forall x _ { t } . } \end{array}
$$

Integrating both sides w.r.t. $q _ { t } ( x _ { t } )$ yields

$$
\begin{array} { r } { \left\| \widehat { V } _ { t , \eta , N } - V _ { t , \eta } ^ { * } \right\| _ { L _ { 2 } ( q _ { t } ) } ^ { 2 } \leq \alpha ^ { 2 } e ^ { \xi } \left\| \widehat { f } _ { t , \eta } - \mathbb { E } _ { q _ { 0 | t } } Y _ { \eta } \right\| _ { L _ { 2 } ( q _ { t } ) } ^ { 2 } . } \end{array}\tag{27}
$$

Starting from here, we omit the dependence of $\mathbb { E } _ { q _ { 0 \mid t } } Y _ { \eta }$ on $x _ { t }$ .

We next control the regression error. For any fixed $g \in { \mathcal { F } } _ { t } ,$ , we have

$$
\begin{array} { r l } & { ( g ( x _ { t } ) - Y _ { \eta } ( x _ { 0 } ) ) ^ { 2 } = ( ( g ( x _ { t } ) - \mathbb { E } _ { q _ { 0 1 } } Y _ { \eta } ) + ( \mathbb { E } _ { q _ { 0 1 } } Y _ { \eta } - Y _ { \eta } ( x _ { 0 } ) ) ) ^ { 2 } } \\ & { \qquad = ( g ( x _ { t } ) - \mathbb { E } _ { q _ { 0 1 } } Y _ { \eta } ) ^ { 2 } + \underbrace { 2 ( g ( x _ { t } ) - \mathbb { E } _ { q _ { 0 1 } } Y _ { \eta } ) ( \mathbb { E } _ { q _ { 0 1 } } Y _ { \eta } - Y _ { \eta } ( x _ { 0 } ) ) } _ { \normalfont \in \mathcal { J } ( \mathcal { X } _ { t } ) } + ( \mathbb { E } _ { q _ { 0 1 } } Y _ { \eta } - Y _ { \eta } ( x _ { 0 } ) ) ^ { 2 } . } \end{array}
$$

Now, take an expectation by conditioning on $x _ { t }$ on both sides. By tower property, $\mathbb { E } [ h ] ~ =$ $\mathbb { E } _ { x _ { t } \sim q _ { t } } [ \mathbb { E } [ h | x _ { t } ] ]$ . For the cross term, we have $\begin{array} { r } { \bar { \mathbb { E } } [ ( g ( x _ { t } ) - \mathbb { E } _ { q _ { 0 \mid t } } Y _ { \eta } ) ( \mathbb { E } _ { q _ { 0 \mid t } } Y _ { \eta } - Y _ { \eta } ( x _ { 0 } ) ) ] { \mathrm { ~ \hat { = } ~ \hat { \mathbb { E } } _ { x _ { t } \sim \bar { q } _ { t } } [ ( { g } ( x _ { t } ) - \hat { \eta } ( x _ { t } ) ) ] ~ } } , } \end{array}$ $\mathbb { E } _ { q _ { 0 | t } } \dot { Y } _ { \eta } ) ( \mathbb { E } _ { q _ { 0 | t } } \ddot { Y } _ { \eta } - \mathbb { E } [ Y _ { \eta } ( x _ { 0 } ) | x _ { t } ] ) ] = 0$ because $\mathbb { E } [ Y _ { \eta } ( x _ { 0 } ) \mid x _ { t } ] = \mathbb { E } _ { q _ { 0 \mid t } } Y _ { \eta }$ . As a result, we have

$$
\begin{array} { r } { \mathbb { E } _ { q _ { 0 } , t } \left[ ( g ( x _ { t } ) - Y _ { \eta } ( x _ { 0 } ) ) ^ { 2 } - ( \mathbb { E } _ { q _ { 0 } \mid t } Y _ { \eta } - Y _ { \eta } ( x _ { 0 } ) ) ^ { 2 } \right] = \left. g - \mathbb { E } _ { q _ { 0 } \mid t } Y _ { \eta } \right. _ { L _ { 2 } ( q _ { t } ) } ^ { 2 } . } \end{array}
$$

Moreover, the excess loss factors as

$$
( g - Y _ { \eta } ) ^ { 2 } - ( \mathbb { E } _ { q _ { 0 | t } } Y _ { \eta } - Y _ { \eta } ) ^ { 2 } = ( g - \mathbb { E } _ { q _ { 0 | t } } Y _ { \eta } ) ( g + \mathbb { E } _ { q _ { 0 | t } } Y _ { \eta } - 2 Y _ { \eta } ) .
$$

Since all predictions and targets lie in $[ 0 , 1 ] .$ , its absolute value is at most one, and $| g + \mathbb { E } _ { q _ { 0 | t } } Y _ { \eta } - 2 Y _ { \eta } | \le 2$ As a result, its second moment satisfies

$$
\begin{array} { r } { \mathbb { E } _ { q _ { 0 , t } } \left[ ( g - \mathbb { E } _ { q _ { 0 | t } } Y _ { \eta } ) ^ { 2 } ( g + \mathbb { E } _ { q _ { 0 | t } } Y _ { \eta } - 2 Y _ { \eta } ) ^ { 2 } \right] \leq 4 \left\| g - \mathbb { E } _ { q _ { 0 | t } } Y _ { \eta } \right\| _ { L _ { 2 } ( q _ { t } ) } ^ { 2 } . } \end{array}
$$

Thus its variance is controlled by its mean, allowing us to apply the Bernstein’s inequality.

Recall Bernstein’s inequality: for i.i.d. $Z _ { i }$ with $| Z _ { i } | \le 1 \ ( \implies | Z _ { i } - \mathbb { E } Z _ { i } | \le 2 )$

$$
\mathbb { P } \left( \mathbb { E } [ Z ] - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } Z _ { i } > \epsilon \right) \le \exp \left( - \frac { N \epsilon ^ { 2 } } { 2 \mathrm { V a r } ( Z ) + \frac { 4 } { 3 } \epsilon } \right) .
$$

Setting the right-hand side to $\delta ^ { \prime } ,$ we get, with probability at least $1 - \delta ^ { \prime } .$

$$
\mathbb { E } [ Z ] - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } Z _ { i } \le \sqrt { \frac { 2 \mathrm { V a r } ( Z ) \log ( 1 / \delta ^ { \prime } ) } { N } } + \frac { 4 \log ( 1 / \delta ^ { \prime } ) } { 3 N } .
$$

Fix $g \in \mathcal { F } _ { t }$ and $\eta \in [ l , u ]$ , and let $\mu _ { \eta } : = \mathbb { E } _ { q _ { 0 | t } } Y _ { \eta }$ . For each sample $i ,$ define $Z _ { i } = ( g ( x _ { t } ^ { i } ) - Y _ { \eta } ( x _ { 0 } ^ { i } ) ) ^ { 2 } -$ $( \mu _ { \eta } ( x _ { t } ^ { i } ) - Y _ { \eta } ( x _ { 0 } ^ { i } ) ) ^ { 2 }$ . Its population and empirical averages are

$$
D : = \mathbb { E } [ Z ] = \Vert g - \mu _ { \eta } \Vert _ { L _ { 2 } ( q _ { t } ) } ^ { 2 } , \widehat { D } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } Z _ { i } = \widehat { \mathcal { L } } _ { t , \eta } ( g ) - \widehat { \mathcal { L } } _ { t , \eta } ( \mu _ { \eta } ) .
$$

Then we have, with probability at least $1 - \delta ^ { \prime }$

$$
D \leq \widehat { D } + \sqrt { \frac { 8 D \log ( 1 / \delta ^ { \prime } ) } { N } } + \frac { 4 \log ( 1 / \delta ^ { \prime } ) } { 3 N } .\tag{28}
$$

Next, we make this bound uniform using a grid with the union bound. Recall $\begin{array} { r } { \xi : = \frac { 2 ( u - l ) } { \alpha } ( 1 - \lambda + \frac { \lambda } { 1 - \beta } ) } \end{array}$ Let $m : = e ^ { - \xi / 2 } ,$ , and $\begin{array} { r } { \tau : = \frac { \varepsilon _ { v } ^ { 2 } } { 4 0 ( 1 + \alpha ^ { 2 } ) } e ^ { - \xi } } \end{array}$

Grid. Write $g _ { \theta } : = \mathrm { c l i p } _ { [ m , 1 ] } ( \theta ^ { \top } \phi )$ . Since clipping is 1-Lipschitz and $| \phi _ { j } | \leq 1 , | | g _ { \theta } - g _ { \theta ^ { \prime } } | | _ { \infty } \leq | | \theta - \theta ^ { \prime } | | _ { 1 }$ Now take a grid with spacing $2 \tau / F$ in each coordinate. Every $\theta _ { j }$ is within $\tau / F$ of a grid value $\theta _ { j } ^ { \prime }$ for $j = 1 , \dots , F$ . Summing over the $F$ coordinates gives $\| \theta - \theta ^ { \prime } \| _ { 1 } \leq \tau$ and thus $\| g _ { \theta } - g _ { \theta ^ { \prime } } \| _ { \infty } \leq \bar { \tau } .$ Therefore, a $\mathrm { g r i d } \Theta _ { \tau } \subset [ - 1 , 1 ] ^ { F }$ with spacing $2 \tau / F$ in each coordinate yields $\mathcal { G } _ { \tau } : = \{ g _ { \theta } : \theta \in \Theta _ { \tau } \} \subseteq \mathcal { F } _ { t } ,$ which is a τ-cover of $\mathcal { F } _ { t }$ in sup norm. Each coordinate has at most $\dot { 2 / } ( 2 \tau / F ) + 1 = F / \tau + 1 \le 2 F / \tau$ points. $\begin{array} { r } { \mathrm { S o } , | \mathcal { G } _ { \tau } | = | \Theta _ { \tau } | \le ( \frac { 2 \hat { F } } { \tau } ) ^ { F } = ( \frac { 8 0 F ( 1 + \alpha ^ { 2 } ) e ^ { \xi } } { \varepsilon _ { \ast } ^ { 2 } } ) ^ { F } } \end{array}$ . Since $\begin{array} { r } { c _ { \lambda , \eta } - a _ { \eta } = ( 1 - \lambda ) ( c - l ) + \frac { \lambda } { 1 - \beta } ( c - \eta ) ^ { + } } \end{array}$ and $s \mapsto e ^ { - s / \alpha }$ is α<sup>−1</sup>-Lipschitz on [0, ∞), $\begin{array} { r } { \| Y _ { \eta } - Y _ { \eta ^ { \prime } } \| _ { \infty } \le \frac { \lambda } { \alpha ( 1 - \beta ) } | \eta - \eta ^ { \prime } | } \end{array}$ . By Jensen’s inequality, the same bound holds for $\| \mu _ { \eta } - \mu _ { \eta ^ { \prime } } \| _ { \infty }$ . Hence there is a grid $H _ { \tau } \subset [ l , u ]$ with spacing $2 \tau \alpha ( 1 - \beta ) / \lambda$ such that every $\eta \in [ l , u ]$ has some $\eta ^ { \prime } \in H _ { \tau }$ with $\| Y _ { \eta } - Y _ { \eta ^ { \prime } } \| _ { \infty } \leq \tau , \ \| \mu _ { \eta } - \mu _ { \eta ^ { \prime } } \| _ { \infty } \leq \tau . \ \mathrm { W h e n } \lambda = 0 .$ $c _ { \lambda , \eta } - a _ { \eta } = c - l$ does not depend on $\eta ,$ , so $Y _ { \eta }$ and $\mu _ { \eta }$ are the same for all $\eta ,$ and we can take any single point, e.g., $H _ { \tau } = \{ l \}$

The number of points in this grid $H _ { \tau }$ is then at most

$$
\frac { u - l } { 2 \tau \alpha ( 1 - \beta ) / \lambda } + 1 = \frac { 1 0 \lambda \xi ( 1 + \alpha ^ { 2 } ) e ^ { \xi } } { ( 1 - \beta + \lambda \beta ) \varepsilon _ { v } ^ { 2 } } + 1 \le \frac { \frac { 1 0 \lambda \xi } { 1 - \beta + \lambda \beta } ( 1 + \alpha ^ { 2 } ) e ^ { \xi } + 1 } { \varepsilon _ { v } ^ { 2 } } .
$$

Now take $\begin{array} { r } { C _ { 0 } : = \operatorname* { m a x } \{ 8 0 F ( 1 + \alpha ^ { 2 } ) e ^ { \xi } , \frac { 1 0 \lambda \xi } { 1 - \beta + \lambda \beta } ( 1 + \alpha ^ { 2 } ) e ^ { \xi } + 1 \} } \end{array}$ . Then we have $| \mathcal { G } _ { \tau } | \leq ( C _ { 0 } / \varepsilon _ { v } ^ { 2 } ) ^ { F }$ and $| H _ { \tau } | \le C _ { 0 } / \varepsilon _ { v } ^ { 2 }$ . Both grids depend only on $( \alpha , \beta , \lambda , l , u , F , \varepsilon _ { v } )$ and not on the data.

Uniform bound on the grid. Let $K : = T | \mathcal { G } _ { \tau } | | H _ { \tau } | \leq T ( C _ { 0 } / \varepsilon _ { v } ^ { 2 } ) ^ { F + 1 }$ and $L : = \log ( K / \delta )$ . Apply (28) with $\delta ^ { \prime } = \delta / K$ to every $( t , g , \eta ) \in [ T ] \times \mathcal { G } _ { \tau } \times H _ { \tau } ,$ , and use the union bound P(some grid bound $\mathrm { f a i l s } ) \leq$ $K \delta ^ { \prime } = \delta$ . Let E be the event that all K bounds hold simultaneously. Then $\mathbb { P } ( \mathcal { E } ) \geq 1 - \delta$ . In other words, with probability at least $1 - \delta ,$ simultaneously for all grid triples, we have

$$
D \leq \widehat { D } + \sqrt { \frac { 8 D L } { N } } + \frac { 4 L } { 3 N } \leq \widehat { D } + \frac { D } { 2 } + \frac { 4 L } { N } + \frac { 4 L } { 3 N }
$$

where the second inequality uses the arithmetic mean-geometric mean (AM-GM) inequality $\sqrt { a b } \le$ $( a + b ) / 2$ . This is equivalent to

$$
\begin{array} { r l } & { \displaystyle \| g - \mu _ { \eta } \| _ { L _ { 2 } ( q _ { t } ) } ^ { 2 } \leq 2 \big ( \widehat { \mathcal L } _ { t , \eta } ( g ) - \widehat { \mathcal L } _ { t , \eta } ( \mu _ { \eta } ) \big ) + \frac { 3 2 } { 3 } \cdot \frac { L } { N } } \\ & { \qquad \leq 2 \big ( \widehat { \mathcal L } _ { t , \eta } ( g ) - \widehat { \mathcal L } _ { t , \eta } ( \mu _ { \eta } ) \big ) + \frac { C } { N } \log \frac { T } { \delta \varepsilon _ { v } } } \end{array}\tag{29}
$$

where the second inequality is because for sufficiently small $\varepsilon _ { v } ,$ we have $\begin{array} { r } { L \leq \log \frac { T } { \delta } + ( F + 1 ) } \end{array}$ log $\begin{array} { r } { \frac { C _ { 0 } } { \varepsilon _ { v } ^ { 2 } } \leq } \end{array}$ $[ 1 + ( F + 1 ) ( \log C _ { 0 } + 2 ) ]$ ] log $\frac { T } { \delta \varepsilon _ { v } }$ and all the constants are absorbed to $\begin{array} { r } { C : = \frac { 3 2 } { 3 } [ 1 + ( F + 1 ) ( \log C _ { 0 } + 2 ) ] } \end{array}$ Perturbation. For a $, a ^ { \prime } , b , b ^ { \prime } \in [ 0 , 1 ] , | ( a - b ) ^ { 2 } - ( a ^ { \prime } - b ^ { \prime } ) ^ { 2 } | \leq 2 ( | a - a ^ { \prime } | + | b - b ^ { \prime } | )$ , since $( a - b ) ^ { 2 } - ( a ^ { \prime } - b ^ { \prime } ) ^ { 2 } =$ $( ( a - b ) - ( a ^ { \prime } - b ^ { \prime } ) ) ( ( a - b ) + ( a ^ { \prime } - \dot { b } ^ { \prime } ) )$ and the second factor is at most 2 in absolute value. Applying this pointwise, both the empirical losses and D change by at most $2 ( \| g - g ^ { \prime } \| _ { \infty } + \| h - h ^ { \prime } \| _ { \infty } )$ when the prediction g and the target or conditional mean $h$ are perturbed in sup norm. For example, for $D ,$ , we have

$$
\begin{array} { r } { \lvert \lVert g - h \rvert \rvert _ { L _ { 2 } ( q _ { t } ) } ^ { 2 } - \lvert \lvert g ^ { \prime } - h ^ { \prime } \rvert \rvert _ { L _ { 2 } ( q _ { t } ) } ^ { 2 } \rvert \leq 2 ( \lVert g - g ^ { \prime } \rVert _ { \infty } + \lVert h - h ^ { \prime } \rVert _ { \infty } ) . } \end{array}\tag{30}
$$

For the empirical loss, we have

$$
\begin{array} { r } { \Big | \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( \underbrace { g ( x _ { t } ^ { i } ) } _ { a } - \underbrace { Y ( x _ { 0 } ^ { i } ) } _ { b } ) ^ { 2 } - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( \underbrace { g ^ { \prime } ( x _ { t } ^ { i } ) } _ { a ^ { \prime } } - \underbrace { Y ^ { \prime } ( x _ { 0 } ^ { i } ) } _ { b ^ { \prime } } ) ^ { 2 } \Big | \leq 2 \big ( \| g - g ^ { \prime } \| _ { \infty } + \| Y - Y ^ { \prime } \| _ { \infty } \big ) . } \end{array}\tag{31}
$$

Transfer to $\widehat { f } _ { t , \eta }$ and arbitrary η. Let $\begin{array} { r } { \rho ( \varepsilon _ { v } ) : = \frac { \varepsilon _ { v } ^ { 2 } } { 8 ( 1 + \alpha ^ { 2 } ) } e ^ { - \xi } } \end{array}$ denote the optimality gap of $\widehat { f } _ { t , \eta }$ . On $\mathcal { E } ,$ fix t and η. Pick $\eta ^ { \prime } \in H _ { \tau }$ and $g ^ { \prime } \in \mathcal G _ { \tau }$ such that $\| Y _ { \eta } - Y _ { \eta ^ { \prime } } \| _ { \infty } \leq \tau , \ \| \mu _ { \eta } - \mu _ { \eta ^ { \prime } } \| _ { \infty } \leq \tau , \ \| \widehat { f } _ { t , \eta } - g ^ { \prime } \| _ { \infty } \leq \tau .$ . By the perturbation bound, the near-optimality of $\widehat { f } _ { t , \eta } ,$ and realizability $\mu _ { \eta } \in \mathcal { F } _ { t } ,$

$$
\widehat { \mathscr { L } } _ { t , \eta ^ { \prime } } ( g ^ { \prime } ) \overset { ( i ) } { \leq } \widehat { \mathscr { L } } _ { t , \eta } ( \widehat f _ { t , \eta } ) + 4 \tau \overset { ( i i ) } { \leq } \widehat { \mathscr { L } } _ { t , \eta } ( \mu _ { \eta } ) + \rho ( \varepsilon _ { v } ) + 4 \tau \overset { ( i i i ) } { \leq } \widehat { \mathscr { L } } _ { t , \eta ^ { \prime } } ( \mu _ { \eta ^ { \prime } } ) + \rho ( \varepsilon _ { v } ) + 8 \tau ,\tag{32}
$$

where (i) is by applying the perturbation bound (31) with $a = g ^ { \prime } ( x _ { t } ^ { i } ) , a ^ { \prime } = \hat { f } _ { t , \eta } ( x _ { t } ^ { i } ) , b = Y _ { \eta ^ { \prime } } ( x _ { 0 } ^ { i } ) , b ^ { \prime } =$ $Y _ { \eta } ( x _ { 0 } ^ { i } )$ , (ii) is because $\widehat { \mathscr { L } } _ { t , \eta } ( \widehat { f } _ { t , \eta } ) \leq \operatorname* { m i n } _ { g \in \mathscr { F } _ { t } } \widehat { \mathscr { L } } _ { t , \eta } ( g ) + \rho ( \varepsilon _ { v } ) \leq \widehat { \mathscr { L } } _ { t , \eta } ( \mu _ { \eta } ) + \rho ( \varepsilon _ { v } )$ , and (iii) is by applying the perturbation bound in (31) with $a = \mu _ { \eta } ( x _ { t } ^ { i } ) , a ^ { \prime } = \mu _ { \eta ^ { \prime } } ( x _ { t } ^ { i } ) , b = Y _ { \eta } ( x _ { 0 } ^ { i } ) , b ^ { \prime } = Y _ { \eta ^ { \prime } } ( x _ { 0 } ^ { i } )$ . Combining this with the perturbation bound for $D$ and with (29) at $( t , g ^ { \prime } , \eta ^ { \prime } )$ gives

$$
\| \widehat { f } _ { t , \eta } - \mu _ { \eta } \| _ { L _ { 2 } ( q _ { t } ) } ^ { 2 } \overset { ( i v ) } { \leq } \| g ^ { \prime } - \mu _ { \eta ^ { \prime } } \| _ { L _ { 2 } ( q _ { t } ) } ^ { 2 } + 4 \tau \overset { ( v ) } { \leq } 2 0 \tau + 2 \rho ( \varepsilon _ { v } ) + \frac { C } { N } \log \frac { T } { \delta \varepsilon _ { v } } ,
$$

where (iv) is by applying the perturbation bound (30) and $( v )$ is by Eqs. (29) and (32). Multiplying this by $\alpha ^ { 2 } e ^ { \xi }$ and using (27), since $\alpha ^ { 2 } e ^ { \xi } \cdot 2 0 \tau \le \varepsilon _ { v } ^ { 2 } / 2$ and $\alpha ^ { 2 } e ^ { \xi } \cdot 2 \rho ( \bar { \varepsilon _ { v } } ) \leq \bar { \varepsilon _ { v } } / 4$ , we have

$$
\lVert \widehat { V } _ { t , \eta , N } - V _ { t , \eta } ^ { * } \rVert _ { L _ { 2 } ( q _ { t } ) } ^ { 2 } \leq \frac { 3 \varepsilon _ { v } ^ { 2 } } { 4 } + \alpha ^ { 2 } e ^ { \xi } \frac { C } { N } \log \frac { T } { \delta \varepsilon _ { v } } \leq \varepsilon _ { v } ^ { 2 }
$$

whenever $\begin{array} { r } { N \geq \frac { C } { \varepsilon _ { v } ^ { 2 } } \log \frac { T } { \delta \varepsilon _ { v } } } \end{array}$ , with $4 \alpha ^ { 2 } e ^ { \xi }$ absorbed into C. Since t and η were arbitrary and everything holds on $\mathcal { E } ,$ , the bound $\begin{array} { r } { \operatorname* { s u p } _ { \eta \in [ l , u ] } \operatorname* { m a x } _ { t \in [ T ] } \| \widehat { V } _ { t , \eta , N } - V _ { t , \eta } ^ { * } \| _ { L _ { 2 } ( q _ { t } ) } \leq \varepsilon _ { v } } \end{array}$ holds with probability at least $1 - \delta$

## I Proof of Theorem 4

Throughout the proof, we assume that tilted-transition sampling is exact (i.e., the number of candidates $M  \infty$ in SVDD) and that the training events in Theorems 2 and 3 succeed with probability at least $1 - \delta / 2$ , respectively. Specifically, define the events

$$
\begin{array} { r l } & { \mathcal { E } _ { \eta } : = \left\{ \vert \widehat { \eta } - \eta _ { \kappa } ^ { * } \vert \leq \varepsilon _ { \eta } \right\} , } \\ & { \mathcal { E } _ { v } : = \left\{ \underset { \eta \in [ l , u ] } { \operatorname* { s u p } } \ \underset { t \in [ T ] } { \operatorname* { m a x } } \Vert \widehat { V } _ { t , \eta } - V _ { t , \eta } ^ { * } \Vert _ { L ^ { 2 } ( q _ { t } ) } \leq \varepsilon _ { v } \right\} . } \end{array}
$$

By Theorem 2 and Theorem $^ { 3 , }$ under the corresponding choice of hyperparameters (particularly with failure probability $\delta / 2 )$ , these events satisfy

$$
\mathbb { P } ( \mathcal { E } _ { \eta } \cap \mathcal { E } _ { v } ) \geq 1 - \delta .
$$

We write $\eta$ for $\hat { \eta }$ for brevity. In the rest of the proof, we would fix a realization of the estimation samples in $\mathcal { E } _ { \eta } \cap \mathcal { E } _ { v }$ . Thus, from Theorem 2,

$$
| \eta - \eta _ { \kappa } ^ { \star } | \leq \varepsilon _ { \eta } .\tag{33}
$$

Also, from Theorem $^ { 3 , }$

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } _ { x _ { t } \sim q _ { t } } \left\| \widehat { V } _ { t , \eta } ( x _ { t } ) - V _ { t , \eta } ^ { * } ( x _ { t } ) \right\| ^ { 2 } \leq \varepsilon _ { v } ^ { 2 } .\tag{34}
$$

Total error decomposition. To begin, note that from Theorem 1, the marginal distribution induced by $\pi _ { t , \eta } ^ { * }$ also satisfies (cf. [20, Theorem 1])

$$
p _ { \eta } ^ { \star } ( x _ { 0 } ) \propto \exp \left( - \frac { c _ { \lambda , \eta } ( x _ { 0 } ) } { \alpha } \right) q _ { 0 } ( x _ { 0 } ) .
$$

Note that the target distribution of interest is $p _ { \eta _ { \kappa } ^ { \star } } ^ { \star } ( x _ { 0 } )$ (that solves the objective in (11)). The actual sampling distribution is $\hat { p } _ { \eta } ( x _ { 0 } )$ , where η is close to $\eta _ { \kappa } ^ { \star }$ and $\hat { p }$ is different from $p ^ { \star }$ due to inaccurate value estimate. We can first decompose the sampling error as

$$
\mathrm { T V } ( p _ { \eta _ { \kappa } ^ { \star } } ^ { \star } ( x _ { 0 } ) , \hat { p } _ { \eta } ( x _ { 0 } ) ) \leq \underbrace { \mathrm { T V } ( p _ { \eta _ { \kappa } ^ { \star } } ^ { \star } ( x _ { 0 } ) , p _ { \eta } ^ { \star } ( x _ { 0 } ) ) } _ { \mathrm { e r r o r ~ c a u s e d b y ~ i n a c c u r a t e l y ~ s o l v e d } \eta } + \underbrace { \mathrm { T V } ( p _ { \eta } ^ { \star } ( x _ { 0 } ) , \hat { p } _ { \eta } ( x _ { 0 } ) ) } _ { \mathrm { e r r o r ~ c a u s e d b y ~ i n a c c u r a t e ~ v a l u e ~ e s t i m a t e s } }\tag{35}
$$

While as follows we assume perfect resampling (i.e., when $M \to \infty )$ , the corresponding resampling error can be incorporated in the TV triangle inequality above with additional error terms.

On the error caused by inaccurate value estimates. We first tackle the second term in (35). The structure follows similarly from [16, Theorem 3.1]. Note that the step-wise KL error is, by dataprocessing inequality and the KL chain-rule,

$$
\mathrm { K L } ( p _ { \eta } ^ { \star } ( x _ { 0 } ) | | \hat { p } _ { \eta } ( x _ { 0 } ) ) \leq \sum _ { t = 0 } ^ { T } \mathbb { E } _ { p _ { t + 1 , \eta } ^ { \star } } \left[ \mathrm { K L } ( p _ { t | t + 1 , \eta } ^ { \star } | | \hat { p } _ { t | t + 1 , \eta } | x _ { t + 1 } ) \right] = \sum _ { t = 0 } ^ { T } \mathbb { E } _ { p _ { t + 1 , \eta } ^ { \star } } \left[ \mathrm { K L } ( \pi _ { t + 1 , \eta } ^ { \star } | | \hat { \pi } _ { t + 1 , \eta } | x _ { t + 1 } ) \right] ,\tag{36}
$$

where we note that $x _ { T + 1 }$ is a dummy initialization state. The following lemma, originally in [16, Lemma C.1], is pasted below for convenience.

Lemma 5 (Lemma C.1 in [16]). Let $p _ { a } ( x ) \propto e ^ { a ( x ) } q ( x )$ and $p _ { b } ( x ) \propto e ^ { b ( x ) } q ( x )$ , with $| a ( x ) | \leq U$ and $| b ( x ) | \leq U$ . Then,

$$
\mathrm { K L } ( p _ { a } | | p _ { b } ) \leq \frac { e ^ { 2 U } \| a ( x ) - b ( x ) \| _ { L ^ { 2 } ( q ) } ^ { 2 } } { 2 } .
$$

From Theorem 1, we have

$$
\pi _ { t + 1 , \eta } ^ { * } ( x _ { t } \mid x _ { t + 1 } ) \propto \mu _ { t + 1 } ( x _ { t } \mid x _ { t + 1 } ) e ^ { - V _ { t , \eta } ^ { * } ( x _ { t } ) / \alpha } .
$$

The following lemma characterizes the limiting transition kernel of SVDD as $M \to \infty ,$ , with the value estimates held fixed. Related asymptotic consistency results for SMC estimators are established in [34].

Lemma 6. Fix η and bounded value estimates $\{ \hat { V } _ { t , \eta } \} _ { t = 0 } ^ { T }$ . Let $\hat { \pi } _ { t + 1 , \eta } ^ { ( M ) }$ denote the SVDD transition law with M candidates, averaged over candidate generation and resampling. Then,for each t, $\eta , x _ { t } ,$ , and $x _ { t + 1 }$ , we have

$$
\hat { \pi } _ { t + 1 , \eta } ^ { ( M ) } ( x _ { t } \mid x _ { t + 1 } ) \to \hat { \pi } _ { t + 1 , \eta } ( x _ { t } \mid x _ { t + 1 } ) , \quad a s \ M \to \infty ,
$$

where

$$
\hat { \pi } _ { t + 1 , \eta } ( x _ { t } \mid x _ { t + 1 } ) \propto \mu _ { t + 1 } ( x _ { t } \mid x _ { t + 1 } ) e ^ { - \hat { V } _ { t , \eta } ( x _ { t } ) / \alpha } .
$$

Proof. See Appendix J.4.

Note that, since both $\hat { \pi } _ { t + 1 , \eta } ^ { ( M ) }$ and $\hat { \pi } _ { t + 1 , \eta }$ are probability densities and the limit integrates to one, Lemma 6 further implies a convergence in total variation by Scheffé’s lemma. Later, because of the tilting form, Lemma 6 would enable us to invoke Lemma $\dot { 5 }$ for KL $\cdot ( \pi _ { t + 1 , \eta } ^ { \star } ( \cdot \mid x _ { t + 1 } ) \vert \vert \hat { \pi } _ { t + 1 , \eta } ( \cdot \mid x _ { t + 1 } ) )$

Before that step, we first need to identify the correct upper bound U to invoke the lemma. From (26) in Theorem 3, we have

$$
a _ { 7 } \leq c _ { \lambda , \eta } ( x _ { 0 } ) \leq a _ { \eta } + \frac { \alpha \xi } { 2 } ,
$$

where we recall that $a _ { \eta } = ( 1 - \lambda ) l + \lambda \eta$ and $\begin{array} { r } { \xi = \frac { 2 ( u - l ) } { \alpha } \left( 1 - \lambda + \frac { \lambda } { 1 - \beta } \right) } \end{array}$ . Note that the function $f ( z ) = - \alpha \log \mathbb { E } \left[ e ^ { - z / \alpha } \right]$ preserves the bounds in z. Thus, from the inductive relationship in (3), it follows that

$$
a _ { \eta } \leq V _ { t , \eta } ^ { * } ( x _ { t } ) \leq a _ { \eta } + \frac { \alpha \xi } { 2 } , \qquad \forall t \geq 0 , \forall x _ { t } \in \mathbb { R } ^ { d } .
$$

Now, for the estimated value function $\hat { V } _ { t , \eta } ( x _ { t } )$ , by the clipping range of $\widehat { f } _ { t , \eta }$ and the definition of the value estimator, we again have

$$
a _ { \eta } \le \hat { V } _ { t , \eta } ( x _ { t } ) \le a _ { \eta } + \frac { \alpha \xi } { 2 } , \qquad \forall t = 0 , \ldots , T ,
$$

where at $t = 0$ we use $\hat { V } _ { 0 , \eta } = V _ { 0 , \eta } ^ { * } = c _ { \lambda , \eta } .$ . Since adding a constant to a potential does not affect the normalized tilted distribution, we center this common interval at $a _ { \eta } + \alpha \xi / 4$ . The normalized potentials in Lemma 5 are then bounded in absolute value by $\frac { \xi } { 4 }$

Thus, by Lemma 5, given $x _ { t + 1 }$

$$
\mathrm { K L } ( \pi _ { t + 1 , \eta } ^ { \star } ( \cdot \mid x _ { t + 1 } ) | | \hat { \pi } _ { t + 1 , \eta } ( \cdot \mid x _ { t + 1 } ) ) \leq \frac { \exp { ( \xi / 2 ) } } { 2 \alpha ^ { 2 } } \left\| \hat { V } _ { t , \eta } - V _ { t , \eta } ^ { * } \right\| _ { L ^ { 2 } ( \mu _ { t + 1 } ( \cdot \mid x _ { t + 1 } ) ) } ^ { 2 } .
$$

From (36) and using Pinsker’s inequality, we get

$$
\begin{array} { r l } & { \begin{array} { r l } & { \displaystyle \mathrm { T V } ( p _ { \eta } ^ { * } ( x _ { 0 } ) , \hat { p } _ { \eta } ( x _ { 0 } ) ) \leq \sqrt { \frac { 1 } { 2 } \sum _ { t = 0 } ^ { T } \mathbb { E } _ { p _ { t + 1 , \eta } ^ { * } } \left[ \mathrm { K L } ( \pi _ { t + 1 , \eta } ^ { * } \left. \hat { { \boldsymbol { x } } } _ { t + 1 , \eta } \right. x _ { t + 1 } ) \right] } } \\ & { \displaystyle \stackrel { ( i ) } { \leq } \sqrt { \frac { T } { 2 } \epsilon ^ { \xi / 2 } \sum _ { t = 0 } ^ { T } \mathbb { E } _ { q _ { t + 1 } } \left[ \mathrm { K L } ( \pi _ { t + 1 , \eta } ^ { \star } \left. \hat { { \boldsymbol { x } } } _ { t + 1 , \eta } \right. x _ { t + 1 } ) \right] } } \end{array} } \\ & { \qquad \leq \sqrt { \frac { 1 } { 2 } \frac { \epsilon ^ { \xi } } { 2 \alpha ^ { 2 } } \sum _ { t = 1 } ^ { T } \mathbb { E } _ { q _ { t } } \left. \hat { { \boldsymbol { V } } } _ { t , \eta } - { \boldsymbol { V } } _ { t , \eta } ^ { * } \right. ^ { 2 } } } \\ & { \qquad \mathrm { ~ } \stackrel { ( i i ) } { \leq } \frac { \epsilon ^ { \xi / 2 } } { 2 \alpha ^ { 2 } } \sqrt { T } \varepsilon _ { \nu } , } \end{array}\tag{37}
$$

where (ii) follows from (34). To justify (i), we note that under the optimal policy in Theorem 1, the corresponding marginal sampling distribution is also a tilted form of the pretrained marginal (cf. [6, Theorem 1]):

$$
p _ { t + 1 , \eta } ^ { \star } \propto q _ { t + 1 } e ^ { - V _ { t + 1 , \eta } ^ { \star } / \alpha } \implies p _ { t + 1 , \eta } ^ { \star } ( x _ { t + 1 } ) = \frac { q _ { t + 1 } ( x _ { t + 1 } ) \mathbb { E } _ { q _ { 0 | t + 1 } } \left[ e ^ { - c _ { \lambda , \eta } ( x _ { 0 } ) / \alpha } \right] } { \mathbb { E } _ { q _ { 0 } } \left[ e ^ { - c _ { \lambda , \eta } ( x _ { 0 } ) / \alpha } \right] } .
$$

Since $0 \leq c _ { \lambda , \eta } ( x _ { 0 } ) - a _ { \eta } \leq \alpha \xi / 2$ , we have

$$
\frac { p _ { t + 1 , \eta } ^ { \star } ( x _ { t + 1 } ) } { q _ { t + 1 } ( x _ { t + 1 } ) } = \frac { \mathbb { E } _ { q _ { 0 | t + 1 } } \left[ e ^ { - ( c _ { \lambda , \eta } ( x _ { 0 } ) - a _ { \eta } ) / \alpha } \right] } { \mathbb { E } _ { q _ { 0 } } \left[ e ^ { - ( c _ { \lambda , \eta } ( x _ { 0 } ) - a _ { \eta } ) / \alpha } \right] } \leq e ^ { \xi / 2 } .
$$

On the error caused by inaccurately solved $\eta .$ We now turn to the first term in (35). Without loss of generality assume $\eta > \eta _ { \kappa } ^ { * }$ . Define

$$
\Delta _ { \lambda , \eta } ( x _ { 0 } ) : = \frac { \lambda } { \alpha ( 1 - \beta ) } \left( ( c ( x _ { 0 } ) - \eta _ { \kappa } ^ { * } ) ^ { + } - ( c ( x _ { 0 } ) - \eta ) ^ { + } \right) \in \left[ 0 , \frac { \lambda ( \eta - \eta _ { \kappa } ^ { * } ) } { \alpha ( 1 - \beta ) } \right] ,
$$

since $c ( x _ { 0 } ) \in [ l , u ]$ . Also, by the definition of $c _ { \lambda , \eta }$ in (7), we have

$$
p _ { \eta } ^ { * } ( x _ { 0 } ) \propto \exp \left( - \frac { c _ { \lambda , \eta } ( x _ { 0 } ) } { \alpha } \right) q _ { 0 } ( x _ { 0 } ) \propto \exp \left( \Delta _ { \lambda , \eta } ( x _ { 0 } ) \right) p _ { \eta _ { \kappa } ^ { * } } ^ { * } ( x _ { 0 } ) .
$$

Write $\partial _ { s } ( \cdot )$ for $\frac { \partial } { \partial s } ( \cdot )$ . Thus, we have

$$
\begin{array} { r l } {  { \operatorname { T V } [ \hat { U } _ { \mathrm { E } _ { \perp } } ^ { \mathrm { o r } } , \hat { P } _ { \mathrm { p } } ^ { \mathrm { i n } } ] - \frac { 1 } { 2 } \int _ { \mathcal { M } _ { \mathrm { E } _ { \perp } } } [ e ^ { \mathrm { i } \alpha _ { \mathrm { p } } t } - \hat { P } _ { \mathrm { p } \mathrm { t } } ^ { \mathrm { i n } } ( \alpha _ { \mathrm { p } } ) ] \mathrm { d } x _ { \mathrm { E } _ { \perp } } } } \\ & { - \frac { 1 } { 2 } \int _ { \mathcal { M } _ { \mathrm { E } _ { \perp } } } [ \frac { \Theta ^ { \mathrm { i n } } ( x _ { \mathrm { E } } ) - e ^ { \mathrm { i } \alpha _ { \mathrm { p } } t } } { [ 1 - e ^ { \mathrm { i } \alpha _ { \mathrm { E } _ { \mathrm { E } _ { \perp } } } ( x _ { \mathrm { E } } ) } ] - 1 } ] \hat { P } _ { \mathrm { e } _ { \perp } } ^ { \mathrm { i n } } \langle x _ { \mathrm { E } } ^ { \mathrm { o r } } , \hat { \Phi } _ { \mathrm { E } } \rangle \mathrm { d } X _ { \mathrm { E } _ { \perp } } } \\ & { - \frac { 1 } { 2 } \int _ { \mathcal { M } _ { \mathrm { E } _ { \perp } } } [ \frac { e ^ { - \mathrm { i } \alpha _ { \mathrm { E } _ { \mathrm { E } _ { \perp } } } t } } { [ 1 - e ^ { \mathrm { i } \alpha _ { \mathrm { E } _ { \mathrm { E } _ { \mathrm { E } _ { \perp } } } } t } ] - 1 } e ^ { \mathrm { i } \alpha _ { \mathrm { E } _ { \mathrm { E } _ { \perp } } } t } ] \hat { P } _ { \mathrm { e } _ { \perp } } ^ { \mathrm { i n } } ( x _ { \mathrm { E } } ^ { \mathrm { o r } } ) \mathrm { d } x _ { \mathrm { E } _ { \perp } } } \\ &  < \frac { 1 } { 2 } \int _  \mathcal { M } _  \ \end{array}
$$

where (i) follows because

$$
\partial _ { s } \frac { e ^ { s \Delta _ { \lambda , \eta } ( x _ { 0 } ) } } { \mathbb { E } _ { p _ { r _ { \kappa } ^ { * } } ^ { * } } \left[ e ^ { s \Delta _ { \lambda , \eta } ( x _ { 0 } ) } \right] } = \frac { e ^ { s \Delta _ { \lambda , \eta } ( x _ { 0 } ) } } { \mathbb { E } _ { p _ { r _ { \kappa } ^ { * } } ^ { * } } \left[ e ^ { s \Delta _ { \lambda , \eta } ( x _ { 0 } ) } \right] } \left( \Delta _ { \lambda , \eta } ( x _ { 0 } ) - \mathbb { E } _ { p _ { s } ^ { * } } \Delta _ { \lambda , \eta } ( x _ { 0 } ) \right) ,
$$

where $p _ { s } ^ { * } ( x _ { 0 } ) \propto \exp \left( s \Delta _ { \lambda , \eta } ( x _ { 0 } ) \right) p _ { \eta _ { \kappa } ^ { * } } ^ { * } ( x _ { 0 } )$ , and (ii) follows from Jensen’s inequality. Further, since $\Delta _ { \lambda , \eta } ( \boldsymbol { x } _ { 0 } )$ is bounded, we have

$$
\sqrt { \operatorname { V a r } _ { p _ { s } ^ { * } } ( \Delta _ { \lambda , \eta } ) } \leq \frac { 1 } { 2 } \cdot \frac { \lambda ( \eta - \eta _ { \kappa } ^ { * } ) } { \alpha ( 1 - \beta ) } ,
$$

and thus, under (33), we have

$$
\mathrm { T V } ( p _ { \eta _ { \kappa } ^ { * } } ^ { \star } , p _ { \eta } ^ { * } ) \leq \frac { 1 } { 4 } \frac { \lambda ( \eta - \eta _ { \kappa } ^ { * } ) } { \alpha ( 1 - \beta ) } \leq \frac { 1 } { 4 } \cdot \frac { \lambda \varepsilon _ { \eta } } { \alpha ( 1 - \beta ) } .\tag{38}
$$

Plugging (37) and (38) into (35), we finally get

$$
\mathrm { T V } ( p _ { \eta _ { \kappa } ^ { \star } } ^ { \star } ( x _ { 0 } ) , \hat { p } _ { \eta } ( x _ { 0 } ) ) \leq \frac { e ^ { \xi / 2 } } { 2 \alpha } \sqrt { T } \varepsilon _ { v } + \frac { \lambda \varepsilon _ { \eta } } { 4 \alpha ( 1 - \beta ) } ,
$$

where $\xi$ is defined in (13). The proof is now complete.

## J Proofs of Auxiliary Lemmas

We provide the proofs of all auxiliary lemmas in this appendix.

## J.1 Proof of Lemma 2

For the first statement, recall that from (7), we have

$$
c _ { \lambda , \eta } ( x _ { 0 } ) = ( 1 - \lambda ) c ( x _ { 0 } ) + \lambda \left( \eta + \frac { ( c ( x _ { 0 } ) - \eta ) ^ { + } } { 1 - \beta } \right) .
$$

Thus, the function $\eta \mapsto c _ { \lambda , \eta } ( x _ { 0 } )$ is continuous and piecewise affine, with slope $\lambda - \lambda / ( 1 - \beta )$ on the interval $c ( x _ { 0 } ) > \eta$ and slope λ on the interval $c ( x _ { 0 } ) \leq \eta$ . Thus,

$$
| c _ { \lambda , \eta } ( x _ { 0 } ) - c _ { \lambda , \eta ^ { \prime } } ( x _ { 0 } ) | \le L | \eta - \eta ^ { \prime } | ,
$$

with $L = \lambda$ max $\left\{ 1 , { \frac { \beta } { 1 - \beta } } \right\}$ . Organizing the terms, we get, for each $x _ { 0 } ,$

$$
\begin{array} { r l } & { \quad - L | \eta - \eta ^ { \prime } | \leq c _ { \lambda , \eta ^ { \prime } } ( x _ { 0 } ) - c _ { \lambda , \eta } ( x _ { 0 } ) \leq L | \eta - \eta ^ { \prime } | } \\ & { \Longrightarrow e ^ { - L | \eta - \eta ^ { \prime } | / \alpha } \leq e ^ { c _ { \lambda , \eta ^ { \prime } } ( x _ { 0 } ) / \alpha - c _ { \lambda , \eta } ( x _ { 0 } ) / \alpha } \leq e ^ { L | \eta - \eta ^ { \prime } | / \alpha } } \\ & { \Longrightarrow e ^ { - L | \eta - \eta ^ { \prime } | / \alpha } e ^ { - c _ { \lambda , \eta ^ { \prime } } ( x _ { 0 } ) / \alpha } \leq e ^ { - c _ { \lambda , \eta } ( x _ { 0 } ) / \alpha } \leq e ^ { L | \eta - \eta ^ { \prime } | / \alpha } e ^ { - c _ { \lambda , \eta ^ { \prime } } ( x _ { 0 } ) / \alpha } } \end{array}
$$

Since this holds for each $x _ { 0 } ,$ it also holds for both the true and the empirical average at $x _ { 0 }$ . To see this, consider the case with empirical average:

$$
\begin{array} { r l } & { \displaystyle e ^ { - L | \eta - \eta ^ { \prime } | / \alpha } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } e ^ { - c _ { \lambda , \eta ^ { \prime } } ( x _ { 0 } ^ { i } ) / \alpha } \leq \frac { 1 } { N } \sum _ { i = 1 } ^ { N } e ^ { - c _ { \lambda , \eta } ( x _ { 0 } ^ { i } ) / \alpha } \leq e ^ { L | \eta - \eta ^ { \prime } | / \alpha } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } e ^ { - c _ { \lambda , \eta ^ { \prime } } ( x _ { 0 } ^ { i } ) / \alpha } } \\ & { \Longrightarrow e ^ { - L | \eta - \eta ^ { \prime } | / \alpha } \leq \frac { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } e ^ { - c _ { \lambda , \eta } ( x _ { 0 } ^ { i } ) / \alpha } } { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } e ^ { - c _ { \lambda , \eta ^ { \prime } } ( x _ { 0 } ^ { i } ) / \alpha } } \leq e ^ { L | \eta - \eta ^ { \prime } | / \alpha } . } \end{array}
$$

Taking logarithms and using the definition of $\widehat { G } _ { \lambda , N }$ yields

$$
| \widehat { G } _ { \lambda , N } ( \eta ) - \widehat { G } _ { \lambda , N } ( \eta ^ { \prime } ) | \leq L | \eta - \eta ^ { \prime } | .
$$

Thus, $\widehat { G } _ { \lambda , N }$ is L-Lipschitz (for every realization of the samples). For the true average, we simply replace all $\textstyle { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } { \bar { ( } } \cdot { \big ) }$ with $\mathbb { E } _ { q _ { 0 } } [ \cdot ]$ . This proves the first statement.

For the second statement, note that for $\eta , \eta ^ { \prime } , \eta _ { c } \in [ l , u ]$ , we have

$$
\frac { \kappa } 2 | ( \eta - \eta _ { c } ) ^ { 2 } - ( \eta ^ { \prime } - \eta _ { c } ) ^ { 2 } | = \frac { \kappa } 2 | \eta - \eta ^ { \prime } | | \eta + \eta ^ { \prime } - 2 \eta _ { c } | \leq \kappa ( u - l ) | \eta - \eta ^ { \prime } | .
$$

Thus, adding this bound to the vanilla objective yields the stated regularized Lipschitz constant (for both the true and the empirical average). This proves the second statement.

Finally, for the last statement, note that the regularization terms cancel:

$$
\widehat { G } _ { \lambda , \kappa , N } - G _ { \lambda , \kappa } = \widehat { G } _ { \lambda , N } - G _ { \lambda } .
$$

This explains why the Lipschitz constant 2L (rather than $2 L _ { \kappa } )$ suffices. The proof is now complete.

## J.2 Proof of Lemma 3

Let $\eta _ { 1 } , \dots , \eta _ { 1 + 2 K } \in [ l , u ]$ be the queried points. These may depend on the oracle values, and hence on the samples. Let $\begin{array} { r } { \mathcal { A } : = \{ \operatorname* { s u p } _ { \eta \in [ l , u ] } | \hat { G } _ { \lambda , \kappa } ( \eta ) - G _ { \lambda , \kappa } ( \eta ) | \leq \epsilon / 4 \} } \end{array}$ . By assumption, $\mathbb { P } ( \mathcal { A } ) \geq 1 - \delta$ On $\scriptstyle A ,$ we have max<sub>1≤i≤1+2K</sub> $| \hat { G } _ { \lambda , \kappa } ( \eta _ { i } ) - G _ { \lambda , \kappa } ( \eta _ { i } ) | \leq \epsilon / 4$ , regardless of how the queried points are chosen. Applying Lemma 1 with $g = G _ { \lambda , \kappa } ,$ , Lipschitz constant $L _ { \kappa } ,$ , and $\tilde { \epsilon } : = \epsilon / 4 ,$ , we obtain that on $\mathcal { A } ,$ Algorithm 2 outputs an ηˆ such that $\begin{array} { r } { G _ { \lambda , \kappa } ( \hat { \eta } ) - { \bar { G } _ { \lambda , \kappa } } ( \eta _ { \kappa } ^ { * } ) = G _ { \lambda , \kappa } ( \hat { \eta } ) - \operatorname* { m i n } _ { \eta \in [ l , u ] } G _ { \lambda , \kappa } ( \eta ) \leq \epsilon } \end{array}$ by calling the inexact oracle $\hat { G } _ { \lambda , \kappa }$ at most $1 + 2 \lceil \log _ { 3 / 2 } ( L _ { \kappa } ( u - l ) / \tilde { \epsilon } ) \rceil = 1 + 2 K$ times. Furthermore, because $G _ { \lambda , \kappa } ( \eta )$ is $m _ { \kappa }$ -strongly convex and $\eta _ { \kappa } ^ { * }$ minimizes it over $[ l , u ]$ , we have $\begin{array} { r } { \vert \hat { \eta } - \eta _ { \kappa } ^ { * } \vert \leq \sqrt { \frac { 2 } { m _ { \kappa } } ( G _ { \lambda , \kappa } ( \hat { \eta } ) - G _ { \lambda , \kappa } ( \eta _ { \kappa } ^ { * } ) ) } \leq } \end{array}$ $\sqrt { \frac { 2 \epsilon } { m _ { \kappa } } } .$

## J.3 Proof of Lemma 4

Step 1: Pointwise bound. Fix $\eta \in [ l , u ]$ . Let $a _ { \eta } : = ( 1 - \lambda ) l + \lambda \eta , Y _ { \eta } ( x ) : = e ^ { - ( c _ { \lambda , \eta } ( x ) - a _ { \eta } ) / \alpha } , \mu _ { \eta } : =$ $\mathbb { E } [ Y _ { \eta } ( X ) \ \vert \ X _ { T + 1 } = x _ { T + 1 } ]$ , and $\begin{array} { r } { \bar { Y } _ { \eta , N } : = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } Y _ { \eta } ( x _ { 0 } ^ { i } ) } \end{array}$ . The regularizer $\mathinner { \frac { \kappa } { 2 } \mathopen { \left( \eta - \eta _ { c } \right) } } ^ { 2 }$ is deterministic and cancels, and so do the factors $e ^ { - a _ { \eta } / \alpha }$ . Hence $| \hat { G } _ { \lambda , \kappa , N } ( \eta ) - G _ { \lambda , \kappa } ( \eta ) | = \alpha | \log ( \bar { Y } _ { \eta , N } / \mu _ { \eta } )$ |. Since $\begin{array} { r } { c _ { \lambda , \eta } - a _ { \eta } = ( 1 - \lambda ) ( c - l ) + \frac { \lambda } { 1 - \beta } ( c - \eta ) ^ { + } \ge 0 , } \end{array}$ , we have ${ { Y } _ { \eta } } \mathrm { ~ \in ~ } \left[ 0 , 1 \right]$ . By Jensen’s inequality and $( c - \eta ) ^ { + } \leq ( c - l ) ^ { + } , \mu _ { \eta } \geq e ^ { - \mathbb { E } [ c _ { \lambda , \eta } - a _ { \eta } ] / \alpha } \geq \underline { { y } }$ . Let $\nu : = \epsilon / ( 8 \alpha ) \leq 1$ and $r : = 1 - e ^ { - \nu } . \mathrm { I f } | \bar { Y } _ { \eta , N } - \mu _ { \eta } | \leq r \mu _ { \eta } ,$ then $\mu _ { \eta } e ^ { - \nu } \leq \bar { Y } _ { \eta , N } \leq \mu _ { \eta } ( 2 - e ^ { - \nu } ) \leq \mu _ { \eta } e ^ { \nu } , \mathrm { s o } \left| \hat { G } _ { \lambda , \kappa , N } ( \eta ) - G _ { \lambda , \kappa } ( \eta ) \right| \leq \epsilon / 8 .$ . Since $| Y _ { \eta } ( x _ { 0 } ^ { i } ) - \mu _ { \eta } | \le 1$ and $\begin{array} { r } { \mathrm { V a r } ( Y _ { \eta } ( x _ { 0 } ^ { i } ) ) \le \dot { \mathbb { E } } [ Y _ { \eta } ( x _ { 0 } ^ { i } ) ^ { 2 } ] \le \mathbb { E } [ Y _ { \eta } ( x _ { 0 } ^ { i } ) ] \stackrel { . } { = } \mu _ { \eta } , } \end{array}$ Bernstein’s inequality gives

$$
\mathbb { P } \big ( | \bar { Y } _ { \eta , N } - \mu _ { \eta } | > r \mu _ { \eta } \big ) \le 2 \exp \Big ( - \frac { N r ^ { 2 } \mu _ { \eta } ^ { 2 } } { 2 \mu _ { \eta } + \frac 2 3 r \mu _ { \eta } } \Big ) \le 2 \exp \Big ( - \frac { N r ^ { 2 } y } { 2 ( 1 + r / 3 ) } \Big ) \le 2 \exp \Big ( - \frac { 3 N y \epsilon ^ { 2 } } { 2 0 4 8 \alpha ^ { 2 } } \Big ) ,
$$

where the last step uses $r \leq 1$ and $\begin{array} { r } { r \ge \frac { \nu } { 1 + \nu } \ge \frac { \nu } { 2 } } \end{array}$ (since $e ^ { \nu } \geq 1 + \nu$ and $e ^ { - \nu } \leq 1 / ( 1 + \nu ) )$ .

Step 2: Lipschitz continuity in η. By Lemma 2, the error $\Delta ( \eta ) : = \hat { G } _ { \lambda , \kappa , N } ( \eta ) - G _ { \lambda , \kappa } ( \eta )$ is 2L-Lipschitz in η.

Step 3: Uniform bound. Let $H \subset [ l , u ]$ be a grid with spacing at most $h : = \epsilon / ( 1 6 L )$ and $| H | \leq J : =$ $\begin{array} { r } { \left\lceil \frac { 1 \dot { 6 } L ( u - l ) } { \epsilon } \right\rceil ^ { . } + 1 . ( \mathrm { I f } \lambda = 0 } \end{array}$ , then $L = 0 , \Delta$ does not depend on $\eta ,$ and one point suffices.) By Step 1 and a union bound over $H ,$ with probability at least $\begin{array} { r } { 1 - 2 J \exp ( - \frac { 3 N y \epsilon ^ { 2 } } { 2 0 4 8 \alpha ^ { 2 } } ) \geq 1 - \delta , } \end{array}$ , we have $| \Delta ( \eta ^ { \prime } ) | \le \epsilon / 8$ for all $\eta ^ { \prime } \in H$ . On this event, for any $\eta \in [ l , u ]$ , choose $\eta ^ { \prime } \in H \ \widetilde { \mathrm { w i t h } } \ | \eta - \eta ^ { \prime } | \leq h .$ . By Step 2,

$$
| \Delta ( \eta ) | \leq | \Delta ( \eta ^ { \prime } ) | + 2 L | \eta - \eta ^ { \prime } | \leq \frac { \epsilon } { 8 } + \frac { \epsilon } { 8 } = \frac { \epsilon } { 4 } .
$$

The probability condition holds whenever $\begin{array} { r } { N \ge \frac { C _ { G } } { \epsilon ^ { 2 } } \log \frac { 2 J } { \delta } } \end{array}$

## J.4 Proof of Lemma 6

Fix $\eta , t \in \{ 1 , \dots , T + 1 \}$ , and $x _ { t }$ . Let us first recall the SVDD algorithm in Algorithm 3. We first draw M candidates from the pretrained model:

$$
Z = ( Z _ { 1 } , \ldots , Z _ { M } ) , \qquad Z _ { i } \stackrel { \mathrm { i . i . d . } } { \sim } \mu _ { t } ( \cdot \mid x _ { t } ) .
$$

Then, we define the unnormalized resampling weight

$$
w _ { t - 1 } ( z ) : = e ^ { - \hat { V } _ { t - 1 , \eta } ( z ) / \alpha } > 0 ,
$$

where η is omitted for brevity. Since the value estimates are bounded, there exist constants $0 < a \le$ $b < \infty$ such that $a \leq w _ { t - 1 } ( z ) \leq b$ for all z. (While our result certainly extends beyond bounded $c ( x _ { 0 } )$ it suffices to assume it here, as in the rest of the paper.) Finally, conditioned on Z, SVDD selects an index $I \in \{ 1 , \ldots , M \}$ according to

$$
\mathbb { P } ( I = i \mid Z , x _ { t } ) = { \frac { w _ { t - 1 } ( Z _ { i } ) } { \sum _ { k = 1 } ^ { M } w _ { t - 1 } ( Z _ { k } ) } } ,
$$

and returns $X = Z _ { I }$ as the next state. Also define $\delta ( z _ { i } - x )$ as the delta function at $z _ { i }$

With this setup, for each $x ,$ we have

$$
\begin{array} { r l } { \hat { \pi } _ { \boldsymbol { \mathrm { t } } , \boldsymbol { \eta } } ^ { ( \boldsymbol { \mathrm { A } } ) } ( x \mid x ) = \int \operatorname { P } ( x \mid I , Z ) \mathrm { d } \mathbb { P } ( I \mid Z ) \mathrm { d } \mathbb { P } ( Z \mid x _ { \boldsymbol { \mathrm { t } } } ) } & { } \\ & { = \int _ { \mathcal { Z } } ( \displaystyle \prod _ { i = 1 } ^ { M } \mu ( z _ { k } \mid x _ { \boldsymbol { \mathrm { t } } } ) ) \displaystyle \sum _ { i = 1 } ^ { M } ( \displaystyle \frac { w _ { \boldsymbol { \mathrm { t } } } \cdot 1 ( z _ { k } ) } { \sum _ { k } w _ { i - 1 } ( z _ { k } ) } \delta ( z _ { k } - x ) ) \mathrm { d } z } \\ & { = \displaystyle \sum _ { i = 1 } ^ { M } \int _ { \mathcal { Z } } ( \displaystyle \prod _ { i = 1 } ^ { M } \mu ( z _ { k } \mid x _ { \boldsymbol { \mathrm { t } } } ) ) ( \displaystyle \frac { w _ { \boldsymbol { \mathrm { t } } } \cdot 1 ( z _ { i } ) } { \sum _ { k } w _ { i - 1 } ( z _ { k } ) } \delta ( z _ { i } - x ) ) \mathrm { d } z } \\ & { = \displaystyle \sum _ { i = 1 } ^ { M } \mu _ { \boldsymbol { \mathrm { t } } } ( x \mid x _ { \boldsymbol { \mathrm { t } } } ) w _ { i - 1 } ( x ) \int _ { \mathcal { Z } _ { i - 1 } } ( \displaystyle \prod _ { k \neq i } \mu _ { \boldsymbol { \mathrm { t } } } ( z _ { k } \mid x _ { \boldsymbol { \mathrm { t } } } ) ) \displaystyle \frac { 1 } { w _ { \boldsymbol { \mathrm { t } } - 1 } ( x ) + \sum _ { k \neq i } w _ { i - 1 } ( z _ { k } ) } \mathrm { d } z _ { - i } } \\ &  = M \mu ( x \mid x _ { \boldsymbol { \mathrm { t } } } ) w _ { t - 1 } ( x ) \mathbb { E } _ { \boldsymbol { \mathrm { t } } } ( \displaystyle \prod _ { i = 1 } ^ { M } ( \displaystyle \frac { 1 }  w _  i -  \end{array}
$$

where the last line follows from symmetry over i. Thus, by the strong law of large numbers, as $M \to \infty ,$

$$
\frac { 1 } { M } \left( w _ { t - 1 } ( x ) + \sum _ { k = 1 } ^ { M - 1 } w _ { t - 1 } ( z _ { k } ) \right) \xrightarrow { \mathrm { a . s . } } \mathbb { E } _ { \mu _ { t } } \left[ w _ { t - 1 } ( Z _ { 1 } ) \right] .
$$

Moreover, for every M,

$$
0 < \frac { M } { w _ { t - 1 } ( x ) + \sum _ { k = 1 } ^ { M - 1 } w _ { t - 1 } ( Z _ { k } ) } \leq \frac { 1 } { a } .
$$

Thus, by the dominated convergence theorem, we may pass the limit through the expectation in the preceding expression for $\widehat { \pi } _ { t , \eta } ^ { ( M ) }$ , obtaining

$$
{ \widehat { \pi } } _ { t , \eta } ^ { ( M ) } ( x \mid x _ { t } ) \longrightarrow \frac { \mu _ { t } ( x \mid x _ { t } ) w _ { t - 1 } ( x ) } { \displaystyle \int \mu _ { t } ( z \mid x _ { t } ) w _ { t - 1 } ( z ) \mathrm { d } z } = { \widehat { \pi } } _ { t , \eta } ( x \mid x _ { t } ) , \quad \mathrm { a s } \ M  \infty .
$$

The proof is now complete.

## K Experimental Result Details

In this section, we include all the experimental details omitted in the main text.

## K.1 Additional details for synthetic experiments

We now provide the implementation details and additional results for the synthetic experiments in Section 6.1.

## K.1.1 Experimental Setup and Complete Results

We first provide the details for the synthetic experiment, as follows.

Base distribution We use a two-dimensional base distribution, which is a two-component mixture of rotated product-Cauchy distributions:

$$
X _ { 0 } = \mu _ { K } + R \mathrm { d i a g } ( \gamma ) Z , \qquad Z _ { j } \stackrel { \mathrm { i . i . d . } } { \sim } \mathrm { C a u c h y } ( 0 , 1 ) ,\tag{39}
$$

where $K \in \{ 1 , 2 \} , \mathbb { P } \{ K = 1 \} = 0 . 4 = 1 - \mathbb { P } \{ K = 2 \}$ and

$$
\mu _ { 1 } = \left( { \begin{array} { l } { 0 . 5 4 3 } \\ { - 0 . 9 5 8 } \end{array} } \right) , \quad \mu _ { 2 } = \left( { 0 . 2 6 7 } \right) , \quad R = \left( { \begin{array} { l l } { - 0 . 8 6 8 } & { 0 . 4 9 6 } \\ { 0 . 4 9 6 } & { 0 . 8 6 8 } \end{array} } \right) , \quad \gamma = \left( { 0 . 2 9 6 } \right) .
$$

Both mixture components use the same principal-axis rotation and principal Cauchy scales.

Cost function The terminal objective is the hard-capped quadratic cost

$$
c ( x ) = \operatorname* { m i n } \left\{ \frac { \| x - x ^ { \star } \| _ { 2 } ^ { 2 } } { 2 s ^ { 2 } } , B \right\} , \qquad x ^ { \star } = ( 3 , 3 ) , \quad s = 3 , \quad B = 5 0 .\tag{40}
$$

Note that the cost is continuous, nonnegative, and bounded in $[ 0 , B ]$ . For every threshold $\tau < B ,$ its level set is a circle centered at $x ^ { \star }$ with radius

$$
r ( \tau ) = s \sqrt { 2 \tau } .
$$

We also define $\tau _ { q }$ (for $q \in \{ 8 0 , 9 0 , 9 5 \} )$ as the threshold for the $q { \mathrm { - t h } }$ empirical quantile, which is estimated with an independent set of 50000 samples from the frozen base sampler.

Sampling Parameters We use a frozen DDPM-style sampler with $T = 6 0$ reverse steps. We set the temperature to $\alpha = 1$ and the CVaR confidence level to $\bar { \beta } = 0 . 8$ . The vanilla method uses $\lambda = 0$ whereas the risk-averse method uses $\lambda = 1$ . Both guided methods use $M = 1 0$ candidate reverse transitions at each of the $T = 6 0$ reverse steps. For each method, we generate 1000 terminal samples under each of 10 inference seeds.

Value Estimations For a terminal cost $c ( x _ { 0 } )$ , recall the definition of weighted cost in $( 7 )$ . Following SVDD-MC [7], we use conditional-mean regression as a practical approximation to the soft value. Specifically, we train two separate reverse-look-ahead models using rollouts from the frozen base sampler:

$$
m _ { \phi } ( x _ { t } , t ) \approx \mathbb { E } _ { \mu } \left[ c ( x _ { 0 } ) \mid x _ { t } \right] , \quad h _ { \psi } ( x _ { t } , t , \eta ) \approx \mathbb { E } _ { \mu } \left[ ( c ( x _ { 0 } ) - \eta ) ^ { + } \mid x _ { t } \right] .
$$

Specifically, we generate 15000 training trajectories. From each trajectory, we retain several states and pair them with the terminal cost of this trajectory to perform training. At inference time, given $\eta ,$ these models are combined through the value surrogate:

$$
\widehat { V } _ { \lambda , \beta , \eta } ( x _ { t } , t ) = \left[ ( 1 - \lambda ) m _ { \phi } ( x _ { t } , t ) + \lambda \left( \eta + \frac { h _ { \psi } ( x _ { t } , t , \eta ) } { 1 - \beta } \right) \right] .\tag{41}
$$

Then we used $\widehat { V } _ { \lambda , \beta , \eta }$ for resampling, with candidate probabilities proportional to $\exp ( - \widehat { V } _ { \lambda , \beta , \eta } / \alpha )$ We compare (41) with alternative value estimators in Figure $^ { 7 , }$ including estimators based on the exponential targets used in Theorem 3. The conditional-mean construction in (41) is a practical approximation to soft-value guidance and generally differs from the exact construction even when the conditional expectations are evaluated exactly.

Choosing $\eta$ The threshold $\eta$ used by CVaR-SVDD is selected by minimizing the fixed-sample empirical regularized objective in (11), with $\mathbb { E } _ { \mu }$ replaced by 20000 base-sample average drawn from the frozen base sampler. We set $\kappa = 0 . 0 2$ and $\eta _ { c } = 1$ .752599 to be the empirical VaR calculated from the pretrained samples. We use 20000 base samples and report the minimizer from Algorithm 2. A separate empirical global minimizer is adopted, which returns a numerically nearly identical value of $\eta ,$ indicating that the reported result is insensitive to the numerical solver used for the one-dimensional outer problem. We have provided an ablation study on the optimizer below.

![](images/5ad44a190e7c7b356570a5bd05f2cf6ba6de27e63b56e35108926463716b67c0.jpg)

![](images/85767328adb971dc7bdcd49eef897ff55be7be8211551a930a3f30a3fce6b363.jpg)

![](images/3a835b465789e1730bb73ee900e4f667ae1dbaa963b63151418893e2e1ba8ba8.jpg)  
Figure 3: One realization of the terminal samples from the frozen base sampler, vanilla risk-neutra SVDD $( \lambda = 0 )$ , and CVaR-SVDD $( \lambda = 1 )$ on the synthetic mixed-Cauchy example, using one representative random seed. The star denotes the target $x ^ { \star } = ( 3 , 3 )$ , the point colors denote the bounded quadratic cost, and the dashed circles denote the common cost thresholds $\tau _ { 8 0 } , \tau _ { 9 0 } ,$ , and $\tau _ { 9 5 }$

Table 3: Complete sampling performance of synthetic Cauchy data. Here $\tau _ { 9 0 }$ and τ<sub>95</sub> represent empirical VaR under the base model, calculated from 50000 independent samples. Lower values are better throughout.
<table><tr><td>Method</td><td>Mean cost</td><td>Median cost</td><td>Empirical CVaR</td><td> $\hat { \mathbb { P } } ( C > \tau _ { 8 0 } )$  (%)</td><td> $\hat { \mathbb { P } } ( C > \tau _ { 9 0 } )$  (%)</td><td> $\hat { \mathbb { P } } ( C > \tau _ { 9 5 } )$  (%)</td></tr><tr><td>Base</td><td>2.373 [2.126, 2.621]</td><td>1.004 [0.9917, 1.016]</td><td>8.379 [7.190, 9.569]</td><td>19.37 [17.97, 20.77]</td><td>9.760 [8.791, 10.73]</td><td>4.800 [4.208, 5.392]</td></tr><tr><td>SVDD [7]</td><td>0.7057 [0.7000, 0.7114]</td><td>0.6703 [0.6601, 0.6806]</td><td>1.343 [1.319, 1.367] [1.507, 2.393]</td><td>1.950</td><td>0.1400 [0.07968, 0.2003]</td><td>0 observed</td></tr><tr><td>CVaR-SVDD</td><td>0.5865 [0.5814, 0.5917]</td><td>0.6008 [0.5937, 0.6078]</td><td>1.017 [1.003, 1.031]</td><td>0 observed</td><td>0 observed</td><td>0 observed</td></tr></table>

Complete Result Table 3 supplements the main-text comparison with median costs and exceedance probabilities at $\tau _ { 9 0 }$ and $\tau _ { 9 5 }$ . In addition to the mean-cost and CVaR improvements reported in Section 6.1, CVaR-SVDD lowers median cost from 0.6703 to 0.6008 relative to vanilla SVDD. $\mathrm { A t \thinspace } \tau _ { 9 0 } ,$ vanilla SVDD has an observed exceedance rate of 0.14%, whereas CVaR-SVDD has no observed exceedances. Neither guided method produces an exceedance at $\tau _ { 9 5 }$ . The terminal samples and survival curves in Figure 3 provide complementary views of this shift toward lower costs. Note that zero empirical exceedance rates indicate that no such events occurred within the available samples. Since the samples are random, they do not establish zero probabilities. Overall, this experiment demonstrates that CVaR-SVDD can improve the full risk profile of the generated samples while using the same inference-time sampling budget as vanilla SVDD.

We also note that the improvement in the overall cost is indeed an added benefit, which depends heavily on the structures of both the underlying target distribution and the cost function. In particular, CVaR guidance does not distinguish among samples whose costs lie below the optimized threshold $\eta .$ For example, when the distribution of $c ( x _ { 0 } )$ has substantial probability mass concentrated right below the corresponding VaR level, this cut-off effect can leave many generated samples in the moderate-cost region near the threshold. In this case, CVaR-SVDD can achieve better control of the upper tail, while possibly yielding a higher overall average cost than vanilla SVDD. Therefore, better average cost should be viewed as an additional distribution-dependent benefit rather than an intrinsic consequence of the CVaR objective.

![](images/10c41e5eee6936ae111250c4be71750873043672be67b2cceb6d19e6517d1b43.jpg)

![](images/412865fb8ccddb0981cbb9e6962edca1cf4d0b2d2c6637c4e9bf913f29e0e14e.jpg)  
Figure 4: Cost survival curves for the synthetic mixed-Cauchy data using the SMC sampler in Algorithm 4. Left: empirical survival probabilities $\widehat { \mathbb { P } } \{ c ( x _ { 0 } ) > \tau \}$ for vanilla SMC $( \lambda = 0 )$ and CVaR-SMC (λ = 1). $R i g h t \colon$ paired difference $\widehat { \mathbb { P } } _ { \mathrm { C V a R - S M C } } \{ c ( x _ { 0 } ) > \tau \} - \widehat { \mathbb { P } } _ { \mathrm { V a n i l l a - S M C } } \{ c ( x _ { 0 } ) > \tau \}$ . Negative values indicate an improvement from CVaR guidance. The dashed vertical lines indicate the common thresholds $\tau _ { 8 0 } , \tau _ { 9 0 } ,$ , and $\tau _ { 9 5 }$

## K.1.2 Synthetic Ablations

We now provide some ablation studies using the synthetic data.

Ablation on the sampler: SMC. We further test whether the benefit of our risk-averse objective depends on the particular inference-time sampler. In addition to SVDD, we consider a Sequential Monte Carlo (SMC) sampler, which differs from SVDD in its resampling mechanism. While SVDD draws multiple candidate transitions for each current sample and performs resampling locally within the corresponding candidate set, SMC propagates each particle once through the pretrained reverse transition and performs resampling globally over the entire particle population. We have provided a reference pseudo-code in Algorithm 4.

For the experiment, we use the same frozen diffusion model, learned mean- and hinge-value models, optimized $\eta ,$ CVaR confidence level $\beta = 0 . 8 ,$ and temperature $\alpha = 1$ . We compare vanilla SMC with $\lambda = 0$ against CVaR-SMC with λ = 1. The number of SMC particles is set equal to the number of generated samples $( \mathrm { i } . \mathrm { e } . , N = 1 0 0 0 0 )$ , and the ESS resampling threshold is fixed at $\gamma = 0 . 5$ . The result is averaged over 5 random seeds.

The results in Figure 4 are qualitatively consistent with those obtained using SVDD in Figure 3. CVaR-SMC produces a uniformly lower survival curve than vanilla SMC over essentially the full range considered. In particular, the paired CVaR-minus-vanilla difference remains negative around $\tau _ { 8 0 }$ and approaches zero only after both methods have already driven the corresponding tail probabilities to very small values. Since the behavior closely parallels the SVDD results, we omit a repetition of the detailed tail analysis here.

This experiment provides evidence that the improvement from CVaR guidance is not specific to the local candidate-resampling mechanism of SVDD. The same risk-adjusted value function can instead be used as a potential in a global SMC resampling procedure and yields a similar reduction in high-cost outcomes. Therefore, our framework is compatible with different inference-time sampling mechanisms: the CVaR construction determines the risk-sensitive change of measure, whereas SVDD and SMC provide different numerical procedures for approximately realizing this tilted distribution.

![](images/ed12f30bf9523fba4e5288c23ec0da8df911b376525619274d58c1ae2e086fe2.jpg)

![](images/0fc22f295f11c93e5a62a1c6567ae9f719b13c1feda033f81b6b7ac14e938a34.jpg)  
Figure 5: Ablation on the temperature α for vanilla SVDD (λ = 0) and CVaR-SVDD (λ = 1). Left: Empirical bad-sample probability $\widehat { \mathbb { P } } \{ c ( x _ { 0 } ) > \tau _ { 8 0 } \}$ as a function of α. Both axes are logarithmic. Note that the CVaR point at the smallest temperature has zero observed exceedances and therefore cannot be displayed on the logarithmic vertical axis. Right: Base log-likelihood as a function of tail probability. Each point is annotated by its value of α. Higher values on the vertical axis indicate that the generated samples lie, on average, in higher-density regions of the pretrained base distribution.

Ablation on the temperature α. We next study the effect of varying the SVDD temperature α. Recall the optimal path-wise distribution in Theorem 1. Specifically, a small value of α produces a sharper and more nearly deterministic preference for candidates with high estimated utility, whereas a large value of α flattens the resampling probabilities and moves the sampler closer to the unguided transition.

In Figure 5, we compare vanilla SVDD, corresponding to $\lambda = 0$ , with pure CVaR-SVDD (λ = 1) over $\alpha \in \{ 0 . 5 , 1 , 5 , 1 0 \}$ . For both methods, we fix the CVaR confidence level at $\beta = 0 . 8$ and keep the same sampling parameters otherwise.

The left panel of Figure 5 shows that the bad-sample probability decreases with smaller α for both methods, with CVaR-SVDD achieving a better tail probability at the same temperature. In the right panel of Figure 5, we investigate sample fidelity at different temperature values. In particular, we plot the sample-averaged log-likelihood $\begin{array} { r } { \widehat { \mathcal { L } } _ { q _ { 0 } } : = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log q _ { 0 } ( x _ { 0 } ^ { i } ) } \end{array}$ as a function of the tail probability. Note that a larger value of $\widehat { \mathcal { L } } _ { q _ { 0 } }$ means that the generated samples lie, on average, in regions assigned higher density by the pretrained base model. Compared to vanilla SVDD, CVaR-SVDD exhibits a favorable empirical frontier.

Ablation on the risk weight λ. We investigate how the interpolation weight λ affects tail performance. We vary $\lambda \in \{ 0 , 0 . \bar { 2 } 5 , 0 . 5 , 0 . 7 5 , 1 \}$ while fixing the other hyperparameters. For each nonzero λ, we optimize η using the same fixed set of base-model costs. We then generate 10000 samples for each of five inference seeds and use the same evaluation thresholds across all values of λ.

As shown in Table 4, increasing λ progressively reduces both empirical CVaR and the exceedance probability at τ<sub>80</sub>. The intermediate values of λ produce intermediate tail performance, illustrating how the mean-CVaR interpolation provides a tunable degree of tail control.

Ablation on the regularization strength κ. We investigate how quadratic threshold regularization affects mixed-Cauchy CVaR-SVDD. We sweep $\kappa \in \{ 0 , 0 . 0 0 0 2 , 0 . 0 0 \bar { 2 } , 0 . 0 2 , 0 . 0 6 , 0 . 2 , 2 \}$ , holding λ = 1, $\beta = 0 . 8 , \alpha = 1 , T = 6 0$ , and M = 10 fixed. Note that we use a separately trained hinge-value network shared across all regularization settings, which differs from that in the main experiment. We report averages over 5 seeds and 95% Student-t confidence intervals. Empirical CVaR is calculated directly from each run’s highest-cost 20%, independently of the optimized threshold.

Table 4: Effect of the risk weight λ on synthetic sampling performance. Empirical CVaR is reported as the mean [95% confidence interval] across five inference seeds. Exceedance probabilities are seed averages. The threshold η has no effect when λ = 0 and it is thus omitted.
<table><tr><td>λ</td><td>η</td><td>Empirical CVaR ↓</td><td> $\widehat { \mathbb P } \{ c ( x _ { 0 } ) > \tau _ { 8 0 } \} \downarrow$ </td></tr><tr><td>0</td><td></td><td>1.346 [1.327, 1.365]</td><td>0.01796</td></tr><tr><td>0.25</td><td>1.075862</td><td>1.192 [1.181, 1.203]</td><td>0.00454</td></tr><tr><td>0.50</td><td>1.022672</td><td>1.116 [1.106, 1.125]</td><td>0.00088</td></tr><tr><td>0.75</td><td>0.982972</td><td>1.065 [1.055, 1.076]</td><td>0.00022</td></tr><tr><td>1</td><td>0.955166</td><td>1.031 [1.023, 1.039]</td><td>0.00004</td></tr></table>

![](images/84adabc6e167862f1736bd6ab3b04199e0aebc7f36b1cb301adfacec880d1640.jpg)  
Figure 6: Effect of the regularization strength κ on the empirical exceedance probability ${ \widehat { \mathbb { P } } } \{ c ( x _ { 0 } ) >$ $\tau _ { 8 0 } \}$ for CVaR-SVDD. Results are averaged over five random seeds, with shaded regions indicating 95% Student-t confidence intervals. The dotted horizontal line marks the mean at $\kappa = 0$

In Figure 6 and Table 5, it is observed that small regularization coefficients preserve the observed tail performance. In particular, the default $\kappa = 0 . 0 2$ increases mean cost and empirical CVaR by only 0.18% and 0.19%, respectively, relative to the unregularized case. Larger penalties $( \mathrm { e } . \mathrm { g } . , \kappa = 2 )$ however, produce more pronounced changes. This is expected, as stronger regularization pulls the optimized threshold toward the larger reference value $\eta _ { c } ,$ raising the threshold at which the hinge penalty becomes active and consequently weakening tail control.

Ablation on value estimation. We compare four value estimators obtained by combining conditional-mean/hinge or normalized-exponential regression targets with neural networks or clipped linear regressors. We retain the same setup. Results are averaged over five inference seeds, with 10000 samples per method per seed.

As shown in Figure 7, the conditional-mean/hinge neural estimator provides the strongest empirical tail control among the compared methods. This result supports the practical surrogate used in our experiments.

Table 5: Effect of the regularization strength κ on CVaR-SVDD. Performance metrics are reported as means [95% confidence intervals] across five random seeds.
<table><tr><td>κ</td><td> $\widehat { \eta } _ { \kappa }$ </td><td>Mean cost</td><td>Empirical CVaR</td></tr><tr><td>0</td><td>0.950100</td><td>0.6250 [0.6189, 0.6311]</td><td>1.036 [1.028, 1.045]</td></tr><tr><td>0.0002</td><td>0.950100</td><td>0.6250 [0.6189, 0.6311]</td><td>1.036 [1.028, 1.045]</td></tr><tr><td>0.002</td><td>0.950492</td><td>0.6251 [0.6190, 0.6312]</td><td>1.037 [1.028, 1.045]</td></tr><tr><td>0.02</td><td>0.955166</td><td>0.6261 [0.6200, 0.6323]</td><td>1.038 [1.030, 1.046]</td></tr><tr><td>0.06</td><td>0.967847</td><td>0.6289 [0.6227, 0.6351]</td><td>1.042 [1.034, 1.050]</td></tr><tr><td>0.2</td><td>1.022609</td><td>0.6405 [0.6345, 0.6465]</td><td>1.061 [1.054, 1.069]</td></tr><tr><td>2</td><td>1.425546</td><td>0.7198 [0.7128, 0.7268]</td><td>1.199 [1.188, 1.210]</td></tr></table>

![](images/2eb717165cb941479bf26ceb6d072b4daa2c3dabadd1d1c495f68d8dc2522ce6.jpg)  
Figure 7: Value-estimator ablation on the synthetic example. We compare conditional-mean/hinge and normalized-exponential targets using neural networks (NN) and clipped linear regressors with fixed bounded features (Linear). Curves show empirical cost survival probabilities averaged over five inference seeds. Dashed vertical lines mark the base-model cost thresholds τ<sub>80</sub>, τ<sub>90</sub>, and τ<sub>95</sub>. Lower curves indicate fewer high-cost outcomes.

Ablation on the optimization of $\eta .$ We also investigate the numerical behavior of the onedimensional outer optimization over the CVaR threshold $\eta .$ . The objective used in this experiment is the sample-averaged version of the objective in (11). Specifically, we use the same fixed set of $N _ { \eta } = 2 0 0 0 \bar { 0 }$ base-model costs in every objective evaluation, with $\kappa = 0 . 0 2 , \lambda = 1 , \beta = 0 . 8 ,$ and $\alpha = 1$

In Figure 8, we compare the CVaR-Optimizer in Algorithm 2 with two baselines: uniform-grid search and projected gradient descent. All three methods optimize the same fixed-sample empirical objective $\widehat { G } _ { \lambda , \kappa } ( \eta )$ and use the same search interval $[ \ell , u ] = [ \operatorname* { m i n } _ { i } c ( x _ { 0 } ^ { i } ) , \operatorname* { m a x } _ { i } c ( x _ { 0 } ^ { i } ) ]$ ]. We use a global minimizer $\eta _ { \kappa } ^ { \star }$ of $\widehat { G } _ { \lambda , \kappa }$ over [ℓ, u] as the reference solution.

Different from the ternary search approach in Algorithm 2, which adaptively contracts the search interval according to objective evaluations at two interior points, the uniform-grid baseline evaluates $\widehat { G } _ { \lambda , \kappa } ( \eta )$ at equally spaced points over the entire interval $[ \ell , u ]$ and returns the point attaining the smallest objective value. For each point in Figure 8, the number of grid points is chosen to match the number of oracle evaluations used by ternary search, so that the two methods have the same evaluation budget. For gradient descent, we use the analytic derivative of the empirical objective. In particular, for $\lambda = 1$ , away from empirical cost atoms,

![](images/c964dadee689698eafd5fbca05ddd5e86c6da2db537d231816084acde65bf8ba.jpg)

![](images/b749c9a81e8c20467d58341713e21421551a5f5a3999916e4f519401e923b71b.jpg)  
Figure 8: Full ablation on the numerical optimization of the regularized CVaR threshold $\eta .$ The empirical global minimizer $\eta _ { \kappa } ^ { \star }$ is used as reference. $L e f t { \mathrm { : } }$ absolute eta error $| \widehat { \eta } - \eta _ { \kappa } ^ { \star } |$ versus the number of oracle evaluations. $R i g h t \colon$ empirical objective gap $\widehat { G } _ { \lambda , \kappa } ( \widehat { \eta } ) - \widehat { G } _ { \lambda , \kappa } ( \eta _ { \kappa } ^ { \star } )$ versus the same oracle-evaluation budget. The uniform-grid baseline uses the same number of grid evaluations as the corresponding ternary-search budget of Algorithm 2. For gradient descent, one joint evaluation of the objective and its analytic derivative is counted as one oracle evaluation, and the reported curve is averaged over five randomly sampled learning rates.

$$
\widehat { G } _ { 1 , \kappa } ^ { \prime } ( \eta ) = 1 - \frac { 1 } { 1 - \beta } \sum _ { i = 1 } ^ { N _ { \eta } } w _ { i } ( \eta ) \mathbf { 1 } \{ c ( x _ { 0 } ^ { i } ) > \eta \} + \kappa ( \eta - \eta _ { c } ) ,
$$

where

$$
w _ { i } ( \eta ) = \frac { \exp { \left( - \frac { ( c ( x _ { 0 } ^ { i } ) - \eta ) ^ { + } } { \alpha ( 1 - \beta ) } \right) } } { \sum _ { j = 1 } ^ { N _ { \eta } } \exp { \left( - \frac { ( c ( x _ { 0 } ^ { j } ) - \eta ) _ { + } } { \alpha ( 1 - \beta ) } \right) } } .
$$

We apply projected gradient descent so that every iterate remains inside [ℓ, u]. The initialization is set to be the base β-quantile. To reduce sensitivity to the learning-rate choice, we randomly sample five learning rates log-uniformly from [0.05, 0.5] and report the average performance across these five runs. We use a diminishing step size proportional to $1 / \sqrt { k }$ at iteration k. Since the objective and its derivative can be computed from the same tilted weights, one joint objective-gradient computation is counted as one oracle evaluation. Thus, all methods in Figure 8 are compared as a function of the same oracle evaluation budget.

As shown in the left panel of Figure 8, the minimizer by Algorithm 2 rapidly approaches the exact empirical solution. Its error initially remains nearly constant while the search interval is broad and the incumbent is located far from $\eta _ { \kappa } ^ { \star }$ . Once the retained interval enters the neighborhood of the empirical minimizer, the error decreases in a sequence of sharp steps. The plateaus in the curve are expected because we report the best incumbent found so far: a ternary-search iteration can substantially shrink the retained interval without changing the incumbent whenever neither newly evaluated point improves the current best objective value. (a) Comparison with gradient descent: By contrast, gradient descent improves much more smoothly. It quickly reduces the eta error initially and then continues to improve gradually as the oracle budget increases. However, its convergence is substantially slower than that of Algorithm 2 near the optimum. At the largest displayed budget, the averaged gradient-descent error remains on the order of $1 0 ^ { - 2 } ,$ , whereas that of Algorithm 2 reaches an error on the order of $1 0 ^ { - 5 }$ . This difference is particularly notable because gradient descent has access to first-order derivative information, while Algorithm 2 uses only objective evaluations. (b) Comparison with uniform-grid: Also, the uniform-grid baseline is substantially less efficient over most of the evaluation range. Since it allocates evaluations uniformly over the entire initial interval, many evaluations are spent far from the minimizer. Its error also varies non-monotonically with the number of grid points because successive uniform grids are not nested: increasing the grid size changes the locations of the interior points, and hence a larger grid does not necessarily contain the best point from the previous grid. Although the uniform grid eventually reaches an eta error comparable to gradient descent at one favorable budget, its error increases again when the grid locations shift.

The right panel of Figure 8 exhibits the same behavior more clearly in terms of objective suboptimality. Algorithm 2 reduces $\widehat { G } _ { \lambda } ( \widehat { \eta } ) - \widehat { G } _ { \lambda } ( \eta _ { \kappa } ^ { \star } )$ by many orders of magnitude and eventually reaches the numerical precision floor. Gradient descent decreases the objective gap steadily, but remains several orders of magnitude above ternary search at the largest evaluation budget. Uniform-grid search again converges more slowly and exhibits the same non-monotone behavior caused by the changing grid locations.

Overall, the experiment shows that Algorithm 2 provides substantially more accurate optimization of η under a fixed evaluation budget than both uniform-grid search and gradient descent on this mixed-Cauchy example. In particular, Algorithm 2 achieves much smaller error in both the recovered threshold η and the empirical regularized objective value, despite being derivative-free. These results empirically support the convergence behavior of the η-optimization procedure.

## K.2 Additional details for image experiments

We now provide the implementation details and additional results for the image experiments in Section 6.2.

## K.2.1 Experimental Setup

We first provide the setup for the image experiment, as follows. The value networks are trained on an NVIDIA H100 GPU, while inference-time sampling is performed on an NVIDIA RTX PRO 6000 Blackwell Server Edition GPU (G4) through Google Colab.

Base diffusion model We use the Stable Diffusion v1.5 [4], with checkpoint from Huggingface (runwayml/stable-diffusion-v1-5), to generate RGB images of resolution $5 1 2 \times 5 1 2$ from latents of shape $4 \times 6 4 \times 6 4$ . Sampling uses $T = 5 0$ reverse steps, classifier-free guidance scale 7.5, and DDIM stochasticity η<sub>DDIM</sub> = 1 (which is essentially DDPM [1]). The diffusion U-Net, VAE decoder, and text encoder are frozen and run in evaluation mode.

Reward functions Both reward functions are employed in [7, 23, 35]. Specifically, For compressibility, the reward is defined as the negative file size after JPEG compression, and the cost is defined as the negative reward. For aesthetics, we use LAION Aesthetic Predictor V2 [36] as the reward (and the negative cost) function, which predicts human aesthetic ratings of generated images, with higher scores indicating greater aesthetic appeal.

Value estimates Let $x _ { t }$ denote the current state at time t and p the chosen text prompt. Vanilla SVDD-MC uses a learned terminal-reward estimate, written in cost form as $\widehat { m } ( x _ { t } , t , p )$ , and our learned CVaR estimator approximates the corresponding positive part:

$$
\widehat { m } ( x _ { t } , t , p ) \approx \mathbb { E } _ { \mu } [ c ( x _ { 0 } ) \mid x _ { t } , t , p ] , \quad \widehat { h } ( x _ { t } , t , p , \eta ) \approx \mathbb { E } _ { \mu } [ ( c ( x _ { 0 } ) - \eta ) ^ { + } \mid x _ { t } , t , p ] .\tag{42}
$$

As in the synthetic experiment, we follow the conditional-mean approximation of SVDD-MC [7].

Training prompts and labels We use the animal prompts from $[ 7 , 2 3 ] $

$$
\{ { \mathrm { c a t } } , { \mathrm { d o g } } , { \mathrm { h o r s e } } , { \mathrm { m o n k e y } } , { \mathrm { r a b b i t } } , { \mathrm { b u t t e r f 1 y } } , { \mathrm { p a n d a } } \} .
$$

Prompts are sampled uniformly with replacement within training minibatches. Frozen base-model trajectories supply the terminal labels. At a recorded state, the mean-reward model is trained against $r ( x _ { 0 } )$ , while the CVaR model is trained against $( c ( x _ { 0 } ) - \eta ) ^ { + }$ . One η is sampled per trajectory, uniformly from a prompt-specific interval, and is shared by the recorded states on that trajectory. Squared-error losses are averaged over samples and recorded states.

Choosing the inference threshold For each prompt and reward, we optimize for η on a separate set of $N _ { \eta } = 2 5 6$ base-model samples, which are generated to approximate $\widehat { G } _ { 1 } ( \eta )$ (with $\lambda = 1$ and $\kappa = 0 )$

$$
\widehat { G } _ { 1 , N _ { \eta } } ( \eta ) = \eta - \alpha \log \left( \frac { 1 } { N _ { \eta } } \sum _ { i = 1 } ^ { N _ { \eta } } \mathrm { e x p } \left\{ - \frac { ( c ( x _ { 0 } ^ { i } ) - \eta ) ^ { + } } { \alpha ( 1 - \beta ) } \right\} \right) .\tag{43}
$$

Note that the empirical quantile cutoff discussed below is also derived from these generated samples. To obtain these base-model samples, we follow [7] and use deterministic DDIM sampling $( \eta _ { \mathrm { D D I M } } = 0 )$ , motivated by its reported sample-quality advantage over DDPM at reduced sampling budgets [37]. We also verified that transferring η<sub>DDIM</sub> = 0 to $\eta _ { \mathrm { D D I M } } = 1$ incurs small objective gaps of only 1.558, 1.318, and 8.110 for $\mathtt { C a t } ,$ dog, and rabbi $\scriptstyle \ t ,$ respectively.

After obtaining (43), we compute an empirical global minimizer by evaluating all observed cost knots together with a uniform grid of 2001 points on the empirical cost range. For this unregularized objective, we can indeed show that a global minimum is attained at an empirical cost knot, since the objective is concave between consecutive knots. With $N _ { \eta } = 2 5 6$ , this scalar search is inexpensive. We also verified that the ternary-search procedure produces thresholds very close to the empirical global minimizers. Specifically, ternary search achieves an absolute difference in the empirical threshold of below $1 . 6 \times 1 0 ^ { - 6 }$ after 40 iterations for both dog and rabbit and below $1 . 2 \times 1 0 ^ { - 5 }$ after 80 iterations for cat. Thus, the procedures produce nearly identical thresholds in all the scenarios considered.

Sampling Hyperparameters We compare three inference-time guided methods: the vanilla SVDD-MC sampler [7], the CVaR-SVDD(-MC) sampler. We further introduce a plug-in CVaR alternative that directly applies the hinge-type transformation to the vanilla value estimate. They share the same pretrained sampling step, but they differ in how the resampling weights are chosen. Specifically, let these weights be proportional to $\exp ( \ell ( x _ { t } ) )$ ), where

$$
\ell _ { \mathrm { v a n i l n } } ( x _ { t } ) = - \widehat { m } ( x _ { t } , t , p ) / \alpha , \quad \ell _ { \mathrm { C V a R } } ( x _ { t } ) = - \frac { \widehat { h } ( x _ { t } , t , p , \eta ) } { \alpha ( 1 - \beta ) } , \quad \ell _ { \mathrm { p l u g i n } } ( x _ { t } ) = - \frac { ( \widehat { m } ( x _ { t } , t , p ) - \eta ) ^ { + } } { \alpha ( 1 - \beta ) } .\tag{44}
$$

Note that the two CVaR methods use the pure tail-risk objective, corresponding to $\lambda = 1 ,$ and share the calculated η. At each of the reverse sampling step before the last one, SVDD generates $M = 2 0$ candidate transitions per current state and selects one with probability proportional to the exponential of its log weight in (44). All compressibility experiments use $\alpha = 5 ,$ , and all aesthetic experiments use $\alpha = 0 . 5$ . For the two CVaR-based methods, we further set $\beta = 0 . 8$

We also conduct ablation studies with the SMC sampler [8] to assess the effectiveness of our framework beyond SVDD. Each seed generates 16 independently initialized populations of $P = 3 2$ particles. Following Algorithm $^ { 4 , }$ we initialize the particle weights using the value potential at the initial noise state and update them by consecutive potential differences during reverse sampling. At the terminal step, we evaluate the exact transformed cost $c _ { \lambda , \eta } ( \boldsymbol { x } _ { 0 } )$ and include its potential $- c _ { \lambda , \eta } ( x _ { 0 } ) / \alpha$ before the final multinomial resampling. See Algorithm 4 for details.

Table 6: Complete image results in reward (i.e., negative cost). Each entry is the mean ± standard error of seed-level metrics. All values are rewards and larger is better. The “lower-tail mean reward” evaluates the average rewards at or below that run’s empirical 20th-percentile cutoff (including ties). The best result for each metric within each block is boldfaced.
<table><tr><td>Reward</td><td>Prompt</td><td>Method</td><td>Mean reward ↑</td><td>Lower-tail mean reward ↑</td></tr><tr><td>Compressibility</td><td>cat</td><td>Base</td><td> $- 9 5 . 9 8 6 \pm 0 . 2 2 4$ </td><td> $- 1 2 3 . 3 7 7 \pm 1 . 0 4 2$ </td></tr><tr><td></td><td></td><td>Vanilla SVDD</td><td> $- 7 3 . 0 3 4 \pm 0 . 1 2 1$ </td><td> $- 8 9 . 6 7 1 \pm 0 . 2 9 2$ </td></tr><tr><td></td><td></td><td>CVaR-SVDD</td><td> $\mathbf { - 6 7 . 1 3 6 \pm 0 . 1 3 1 }$ </td><td> $\mathbf { - 7 8 . 4 0 5 \pm 0 . 1 3 5 }$ </td></tr><tr><td></td><td></td><td>Plug-in CVaR-SVDD</td><td> $- 7 1 . 4 0 1 \pm 0 . 1 3 1$ </td><td> $- 8 3 . 4 3 4 \pm 0 . 2 7 9$ </td></tr><tr><td>Compressibility</td><td>dog</td><td>Base</td><td> $- 1 0 0 . 7 2 1 \pm 0 . 7 2 2$ </td><td> $- 1 3 2 . 2 8 6 \pm 1 . 3 6 6$ </td></tr><tr><td></td><td></td><td>Vanilla SVDD</td><td> $- 7 1 . 3 0 8 \pm 0 . 1 1 3$ </td><td> $- 9 0 . 6 1 2 \pm 0 . 2 0 9$ </td></tr><tr><td></td><td></td><td>CVaR-SVDD</td><td> $\mathbf { - 5 9 . 7 8 4 \pm 0 . 1 6 3 }$ </td><td> $\mathbf { - 7 1 . 2 0 2 \pm 0 . 3 8 2 }$ </td></tr><tr><td></td><td></td><td>Plug-in CVaR-SVDD</td><td> $- 6 4 . 7 5 3 \pm 0 . 0 4 4$ </td><td> $- 7 3 . 8 5 2 \pm 0 . 2 1 5$ </td></tr><tr><td>Compressibility</td><td>rabbit</td><td>Base</td><td> $- 1 1 2 . 4 9 8 \pm 0 . 5 5 2$ </td><td> $- 1 4 4 . 1 6 4 \pm 0 . 4 3 6$ </td></tr><tr><td></td><td></td><td>Vanilla SVDD</td><td> $- 8 2 . 1 4 5 \pm 0 . 2 7 1$ </td><td> $- 1 0 2 . 6 0 9 \pm 0 . 0 9 7$ </td></tr><tr><td></td><td></td><td>CVaR-SVDD</td><td> $\mathbf { - 6 8 . 1 3 3 \pm 0 . 2 8 5 }$ </td><td> $\mathbf { - 8 2 . 8 7 9 \pm 0 . 2 7 1 }$ </td></tr><tr><td></td><td></td><td>Plug-in CVaR-SVDD</td><td> $- 7 2 . 4 1 7 \pm 0 . 5 1 1$ </td><td> $- 8 3 . 9 5 1 \pm 0 . 4 3 0$ </td></tr><tr><td>Compressibility</td><td>cat</td><td>Base</td><td> $- 9 5 . 9 8 6 \pm 0 . 2 1 4$ </td><td> $- 1 2 3 . 3 6 7 \pm 1 . 0 2 5$ </td></tr><tr><td></td><td></td><td>Vanilla SMC</td><td> $\mathbf { - 6 0 . 0 0 4 \pm 0 . 7 3 7 }$ </td><td> $- 7 4 . 5 7 3 \pm 0 . 9 0 5$ </td></tr><tr><td></td><td></td><td>CVaR-SMC</td><td> $- 6 4 . 5 3 3 \pm 0 . 6 0 3$ </td><td> $\mathbf { - 7 3 . 5 9 1 \pm 0 . 1 8 4 }$ </td></tr><tr><td></td><td></td><td>Plug-in CVaR-SMC</td><td> $- 6 4 . 6 5 5 \pm 0 . 4 8 3$ </td><td> $- 7 3 . 6 0 8 \pm 0 . 1 8 3$ </td></tr><tr><td>Aesthetics</td><td>cat</td><td>Base</td><td> $5 . 8 8 4 5 \pm 0 . 0 0 7 6$ </td><td> $5 . 4 6 0 0 \pm 0 . 0 1 0 5$ </td></tr><tr><td></td><td></td><td>Vanilla SVDD</td><td> $5 . 9 1 6 5 \pm 0 . 0 0 3 6$ </td><td> $5 . 5 1 5 4 \pm 0 . 0 0 3 7$ </td></tr><tr><td></td><td></td><td>CVaR-SVDD</td><td> $5 . 9 2 8 5 \pm 0 . 0 0 4 1$ </td><td> $5 . 5 3 4 1 \pm 0 . 0 0 4 3$ </td></tr><tr><td></td><td></td><td>Plug-in CVaR-SVDD</td><td> $\mathbf { 5 . 9 2 9 3 \pm 0 . 0 0 6 1 }$ </td><td> $\mathbf { 5 . 5 3 4 5 \pm 0 . 0 0 7 3 }$ </td></tr><tr><td>Aesthetics</td><td>dog</td><td>Base</td><td> $5 . 5 6 1 1 \pm 0 . 0 1 2 1$ </td><td> $5 . 1 0 5 5 \pm 0 . 0 0 6 3$ </td></tr><tr><td></td><td></td><td>Vanilla SVDD</td><td> $5 . 5 7 0 9 \pm 0 . 0 0 5 5$ </td><td> $5 . 0 9 7 8 \pm 0 . 0 1 0 6$ </td></tr><tr><td></td><td></td><td>CVaR-SVDD</td><td> $5 . 5 8 4 7 \pm 0 . 0 0 1 3$ </td><td> $\mathbf { 5 . 1 3 1 1 \pm 0 . 0 0 7 8 }$ </td></tr><tr><td></td><td></td><td>Plug-in CVaR-SVDD</td><td> $\mathbf { 5 . 5 9 1 3 \pm 0 . 0 0 3 8 }$ </td><td> $5 . 1 2 8 9 \pm 0 . 0 0 7 2$ </td></tr><tr><td>Aesthetics</td><td>rabbit</td><td>Base</td><td> $5 . 7 9 5 9 \pm 0 . 0 0 3 1$ </td><td> $5 . 3 6 6 1 \pm 0 . 0 0 3 3$ </td></tr><tr><td></td><td></td><td>Vanilla SVDD</td><td> $5 . 8 1 7 7 \pm 0 . 0 0 2 3$ </td><td> $5 . 4 2 4 1 \pm 0 . 0 0 4 6$ </td></tr><tr><td></td><td></td><td>CVaR-SVDD</td><td> $\mathbf { 5 . 8 4 1 2 \pm 0 . 0 0 8 0 }$ </td><td> $\mathbf { 5 . 4 3 5 3 \pm 0 . 0 0 9 7 }$ </td></tr><tr><td></td><td></td><td>Plug-in CVaR-SVDD</td><td> $5 . 8 2 6 3 \pm 0 . 0 1 1 3$ </td><td> $5 . 4 1 5 3 \pm 0 . 0 0 6 1$ </td></tr></table>

Each run generates 512 images per method per inference seed in batches of 32. The compressibility cat, aesthetic cat, and compressibility cat SMC results use five seeds per method. The prompt ablations on dog and rabbit use three seeds instead. Initial latents are shared across methods within each seed. Value networks and the threshold-optimization set are fixed across the inference seeds.

## K.2.2 Complete results and ablations

Table 6 reports the mean and tail-average rewards for all evaluated scenarios, averaged across inference seeds with standard errors rather than the 95% confidence intervals used in the preceding tables.

Ablation on the plug-in CVaR sampler. Plug-in CVaR reuses the vanilla value predictor without additional training (see (44)). Across the evaluated compressibility settings using SVDD, plug-in CVaR improves both mean and tail-average rewards over the vanilla sampler, though extra-learning the conditional positive-part cost yields further gains. Figure 9 further illustrates the shift toward higher compressibility rewards. For aesthetics, the differences between learned and plug-in CVaR are smaller and depend on the prompt and metric. Overall, while its tail improvements are less consistent, the plug-in CVaR offers a useful baseline, which demonstrates that the risk-adjusted resampling rule alone can improve tail performance in various settings.

Ablation on the reward: aesthetics. CVaR-SVDD improves the observed mean and lower-tail aesthetic rewards relative to vanilla SVDD across the evaluated prompts. For cat, learned and plug-in CVaR yield nearly identical results, with differences small relative to the reported standard errors. The CDF and violin plot in Figure 10 show that the aesthetic distributions remain close.

Ablation on the prompt. The improvements over vanilla SVDD persist for dog and rabbit under both rewards. Learned CVaR also outperforms plug-in CVaR on both compressibility metrics, whereas their aesthetic rankings depend on the prompt and metric. For aesthetic rabbit, plug-in CVaR reduces the cutoff exceedance fraction but worsens the tail-average reward, illustrating that these metrics capture different aspects of the lower tail. We also show the corresponding reward distributions in Figures 11 and 12. For compressibility, both learned and plug-in CVaR shift the reward distribution toward larger values relative to vanilla SVDD, with learned CVaR providing greater improvements. For aesthetics, the two methods perform similarly for cat and dog, while learned CVaR achieves larger gains for rabbit, where plug-in CVaR fails to improve the tail-average reward over vanilla SVDD.

Ablation on the sampler: SMC. For compressibility with cat, CVaR-SMC exhibits an empirical mean-tail trade-off relative to vanilla SMC: its observed lower-tail mean reward increases, while its overall mean reward decreases. Here learned and plug-in CVaR-SMC yield similar results. Such a trade-off is compatible with the pure CVaR objective (λ = 1), which does not reward further cost reductions below the threshold η. See Figure 13 for the reward distributions.

During the ablation study, we also considered an SMC variant that omits the terminal-cost correction before final output, as well as the initial importance weighting. With this variant, vanilla SMC achieves mean and lower-tail rewards of −75.477 and −89.215, respectively, while the corresponding rewards for CVaR-SMC are −67.399 and −76.288. Both methods attain lower rewards than the corrected implementation reported in Table 6, with a larger difference for vanilla SMC. This comparison highlights the practical importance of the SMC weight corrections. Also, intuitively, such a larger gain for vanilla SMC is consistent with the different terminal objectives: the vanilla tilting factor $\exp ( - c ( x _ { 0 } ) / \alpha )$ favors every reduction in cost, whereas the pure-CVaR tilting factor $\exp ( - ( c ( x _ { 0 } ) - \eta ) ^ { + } / ( \alpha ( 1 - \beta ) ) )$ ) becomes constant for all $c ( x _ { 0 } ) \leq \eta .$ . In other words, the vanilla objective continues to favor further cost reductions below the threshold, while the pure-CVaR objective does not distinguish among these outcomes.

Remark on the wall-clock sampling time. Within each evaluated scenario, vanilla SVDD-MC, learned CVaR-SVDD, and plug-in CVaR-SVDD have similar wall-clock sampling times. For example, for aesthetics with rabbit, generating 512 images in batches of 32 takes approximately 767, 770, and 769 seconds, respectively. These times cover guided image generation and exclude value-network training and the separate threshold-optimization stage. Notably, the CVaR transformation introduces little additional sampling overhead relative to vanilla SVDD-MC in the evaluated settings.

![](images/0257cc937c234e46cf6469d89d9395625bd936f0f8fcbcbc08b294a6f55d0eda.jpg)  
(a) Reward distributions.

![](images/5784cee408f8dfa46a8ba7597189bf2d6b950a81364aa08af36ed04d89428feb.jpg)  
(b) Full reward CDF.

![](images/5e857d2fd4740c312e3072d47a3aae10a3d15653b69825a9412c4d4e645bd5c3.jpg)  
(c) CDF near the reference cutoff.

Figure 9: Compressibility reward distributions for the cat prompt. Vertical lines in the CDF panels mark the 20th-percentile reward of samples for threshold optimization, held fixed across methods. Panel (c) contains a magnified view of (b) near the reference cutoff.  
![](images/ab939cf80539bdbab8f0d5cd1738a363ecc243a709aa9e6f104fa3b9c3ba0624.jpg)  
(a) Reward distributions.

![](images/5325e5d44f100555ee6ef23194b44ba1ea43415b11e1dda3b50e465f7d6631a9.jpg)  
(b) Full reward CDF.

![](images/2f9f8b49eec81b943dc9ce283f7211eed92092ac392bd76b4e653edf63b64c1b.jpg)  
(c) CDF near the reference cutoff.  
Figure 10: Aesthetic reward distributions for the cat prompt. Vertical lines in the CDF panels mark the 20th-percentile reward of samples for threshold optimization, held fixed across methods. Panel (c) contains a magnified view of (b) near the reference cutoff.

## K.2.3 Qualitative tail examples

For each method, we pool its evaluation rewards across the displayed seeds, compute its own 20thpercentile reward, and select four images uniformly without replacement from observations at or below that threshold, including ties. Figures 14 to 17 display the saved examples. For compressibility, reduced image detail and smoother regions can contribute to smaller JPEG sizes. Such changes explain the reward but should not be equated with an improvement in perceptual quality or image diversity. The aesthetic examples illustrate the contents of the low-scoring tail, while the quantitative oracle measurements provide the comparison.

![](images/458841e0c9ad8d91f9bea5b5ed49d5fbd82ce81010e8475a7de300fe7def0373.jpg)  
(a) dog: reward distributions.

![](images/a0787111c4431bedc79d3a9761db98365b250277c29ba52af5732528c3c82187.jpg)  
(b) rabbit: reward distributions.

![](images/12d4ee489fa7f92e5f48f104a3988593ded79dbb5e80308726c3c012d1459aae.jpg)  
(c) dog: full reward CDF.

![](images/e23243c9999fb1cb283435ec8d805705f2f9741f9a213388c928d8be58c089e6.jpg)  
(d) rabbit: full reward CDF.  
Figure 11: Prompt ablation for compressibility. Vertical lines in the CDF panels mark the 20thpercentile reward of samples for threshold optimization, held fixed across methods.

![](images/6a72a291df8fdfadc17f612e6e927f42b8ce44ebf4766e1a7f51ad0275b09101.jpg)  
(a) dog: full reward CDF.

![](images/3a7c73fa685e8ac09f302a2928cb41953c25e0759d8d7b293c56e82cb33aa790.jpg)  
(b) rabbit: full reward CDF.

![](images/fae9fbb512bdae973a7b60b6ed965d8cef2cb2d2a8f103677f35dfba0e8ddc9b.jpg)  
(c) dog: reference-cutoff enlargement.

![](images/e8ba7fabd1b49e4438b06c6435c5ea0bd332edbec4e1c3f8843f197ff61ea15e.jpg)  
(d) rabbit: reference-cutoff enlargement.  
Figure 12: Prompt ablation for aesthetics. Vertical lines in the CDF panels mark the 20th-percentile reward of samples for threshold optimization, held fixed across methods. Full CDFs include the base sampler; enlarged panels show the three guided methods.

![](images/3ec8c334f03003868a7f1bee887c9b7a52b788093724b77403ca85fd996385b8.jpg)  
(a) Reward distributions.

![](images/341e1cf1a3f72ee16bf985a3ad84f0238e4a066477a568d84a60e3ffe3123f12.jpg)  
(b) Full reward CDF.  
Figure 13: SMC ablation for compressibility with cat. Vertical lines in the CDF panels mark the 20th-percentile reward of samples for threshold optimization, held fixed across methods.

![](images/586e75db438a42dfb664c59bc4f1269e7770c194946d367901cff9c8ab67cf4c.jpg)  
Figure 14: Random examples from each method’s own worst-20% compressibility-reward tail for cat. Rows are base, vanilla SVDD, learned CVaR-SVDD, and plug-in CVaR-SVDD. Each row uses its own pooled reward quantile, shown at left. Columns are not paired comparisons.

![](images/cac6f27b18d61160e3074ac915868e687b9c761fa64b7ed229b7a248e26297ba.jpg)  
Figure 15: Random examples from each method’s own worst-20% aesthetic-reward tail for cat. Rows are base, vanilla SVDD, learned CVaR-SVDD, and plug-in CVaR-SVDD. Each row uses its own pooled reward quantile, shown at left. Columns are not paired comparisons.

![](images/da6c46423141a3a211016819b89440bd50fbb6c814424c3a6f12984dc05f4493.jpg)  
Figure 16: Compressibility tail examples for the SMC ablation with cat. Rows are base, vanilla SMC, learned CVaR-SMC, and plug-in CVaR-SMC. Each row uses its own pooled reward quantile, shown at left. Columns are not paired comparisons.

![](images/b80b264efac7eb7d9625e927a0262bb75fda1a27184b8ae4b985849195a56aea.jpg)

![](images/8335dff066f93810eb0c8cb698ff99e60c978b43fe5de3a2e859254fb2872e60.jpg)  
(c) Aesthetics: dog.  
(d) Aesthetics: rabbit.  
Figure 17: Tail examples for dog and rabbit under compressibility and aesthetic rewards. In each panel, rows correspond to base, vanilla SVDD, learned CVaR-SVDD, and plug-in CVaR-SVDD, with four random images from each method’s own pooled lower tail. Columns are not paired comparisons.