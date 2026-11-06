# Optimal Competitive Ratios for Chasing Convex Functions with Quadratic Growth

Linzhe He∗ helz@lamda.nju.edu.cn Sifan Yang∗ yangsf@lamda.nju.edu.cn State Key Laboratory of Novel Software Technology, Nanjing University   
School of Artificial Intelligence, Nanjing University

Wei Jiang

12025220@njust.edu.cn

School of Computer Science and Engineering, Nanjing University of Science and Technology State Key Laboratory of Novel Software Technology, Nanjing University

Lijun Zhang

zhanglj@lamda.nju.edu.cn

State Key Laboratory of Novel Software Technology, Nanjing University School of Artificial Intelligence, Nanjing University

## Abstract

Chasing convex function has attracted much attention in recent years, which can be formulated as a game between a learner and an adversary. In each round, the adversary reveals a convex function, and the learner selects a decision and incurs a hitting cost and a movement cost for changing its decision. The performance of the learner is measured by the competitive ratio, which compares its cumulative cost with that of an optimal of fline decision sequence. For Euclidean movement, existing upper and lower bounds for κ-well-conditioned functions exhibit a polynomial gap. For squared Euclidean movement, dimension-free competitive guarantees have been established under quadratic growth, but the optimal competitive ratio remains unclear.

In this paper, we establish optimal dimension-free competitive ratios for convex function chasing under two types of movement costs. For Euclidean movement, we propose Clipped Proximal Descent (CPD), which combines a balanced proximal step with a clipped correction toward the current minimizer. CPD achieves an $\bar { O } ( \kappa ^ { 1 / 3 } )$ competitive ratio for smooth convex functions satisfying quadratic growth, improving the previous $O ( { \sqrt { \kappa } } )$ upper bound and matching the existing $\Omega ( \kappa ^ { 1 / 3 } )$ lower bound. For squared Euclidean movement, we establish the exact optimal dimension-free competitive ratio for convex functions satisfying quadratic growth. Our result is based on a new tight competitive analysis of Regularized Online Balanced Descent (R-OBD) and a matching lower bound against randomized algorithms.

Keywords: Convex function chasing, quadratic growth, competitive ratio, online optimization, optimal guarantee

## 1 Introduction

In this paper, we study convex function chasing, which is a fundamental problem in online optimization with applications in dynamic resource provisioning and online control (Argue et al., 2020a). Convex function chasing can be formulated as a game between a learner and an adversary. In each round $t ,$ the adversary reveals a convex function $f _ { t } ( \cdot )$ , and the learner selects a decision $\mathbf x _ { t } \in \mathcal X \subseteq \mathbb R ^ { d }$ . The learner then incurs a cost of $f _ { t } ( \mathbf { x } _ { t } ) + m ( \mathbf { x } _ { t } , \mathbf { x } _ { t - 1 } )$ , where $f _ { t } ( \mathbf { x } _ { t } )$ is the hitting cost and m $\mathbf { \Gamma } _ { | } ( \mathbf { x } _ { t } , \mathbf { x } _ { t - 1 } )$ is the movement cost of changing the decision from $\mathbf { x } _ { t - 1 }$ to $\mathbf { x } _ { t }$ . To measure the performance of online algorithms, we adopt the competitive ratio, defined as

$$
\mathrm { C R } = \frac { \sum _ { t = 1 } ^ { T } \left( f _ { t } ( \mathbf { x } _ { t } ) + m ( \mathbf { x } _ { t } , \mathbf { x } _ { t - 1 } ) \right) } { \operatorname* { m i n } _ { \mathbf { u } _ { 1 } , \dots , \mathbf { u } _ { T } \in \mathcal { X } } \sum _ { t = 1 } ^ { T } \left( f _ { t } ( \mathbf { u } _ { t } ) + m ( \mathbf { u } _ { t } , \mathbf { u } _ { t - 1 } ) \right) } ,
$$

where $\mathbf { u } _ { 0 } = \mathbf { x } _ { 0 }$ . The competitive ratio measures the worst-case ratio between the learner’s cumulative cost and the minimum total cost achievable by an ofline decision sequence.

For general convex functions with Euclidean movement, i.e., $m ( \mathbf { x } , \mathbf { y } ) = \| \mathbf { x } - \mathbf { y } \|$ , Friedman and Linial (1993) establish an $\Omega ( { \sqrt { d } } )$ lower bound, while subsequent studies (Argue et al., 2020a; Sellke, 2020) achieve an O(min $\{ d , { \sqrt { d \log T } } \} )$ competitive ratio. To obtain dimension-free guarantees, Argue et al. (2020b) leverage additional structural properties of the hitting costs. They demonstrate that neither strong convexity nor smoothness alone sufices and consider κ-well-conditioned functions, which are α-strongly convex and κα- smooth. For this class of functions, they propose constrained Online Balanced Descent, which achieves an $O ( { \sqrt { \kappa } } )$ competitive ratio, whereas any online algorithm has a worst-case ratio of at least $\Omega ( \kappa ^ { 1 / 3 } )$ . While they establish a better bound, there still exists a polynomial gap between the upper and lower bounds in their dependence on κ.

On the other hand, for squared Euclidean movement, i.e., $\begin{array} { r } { m ( \mathbf { x } , \mathbf { y } ) = \frac { 1 } { 2 } \| \mathbf { x } - \mathbf { y } \| ^ { 2 } } \end{array}$ , strong convexity alone sufices to achieve dimension-free competitive ratios. Goel et al. (2019) develop Regularized Online Balanced Descent (R-OBD), achieving the optimal competitive ratio of $\textstyle { \frac { 1 } { 2 } } ( 1 + { \sqrt { 1 + 4 / \alpha } } )$ for α-strongly convex functions. To extend these guarantees beyond strongly convex functions, Zhang et al. (2021) consider hitting costs satisfying quadratic growth, a standard relaxation of strong convexity (Necoara et al., 2019; Drusvyatskiy and Lewis, 2018). They provide a competitive ratio of $1 + 4 / \lambda$ under λ-quadratic growth and $1 + 2 / \sqrt { \lambda }$ with the additional assumption of convexity.

In this paper, we establish optimal dimension-free competitive guarantees for convex function chasing with quadratic growth under both movement costs. For Euclidean movement, we propose Clipped Proximal Descent (CPD), which combines a balanced proximal step with a clipped correction toward the current minimizer. When each hitting cost $f _ { t } ( \cdot )$ is $L _ { t } { \mathrm { - s m o o t h } }$ and satisfies $\lambda _ { t } { \mathrm { - q u a d r a t i c } }$ growth with $L _ { t } \leq \kappa \lambda _ { t }$ , CPD achieves a better $O ( \kappa ^ { 1 / 3 } )$ competitive ratio. This improves the previous $O ( { \sqrt { \kappa } } )$ upper bound and matches the existing $\Omega ( \kappa ^ { 1 / 3 } )$ lower bound, establishing the optimal $\Theta ( \kappa ^ { 1 / 3 } )$ dependence on κ under a weaker assumption than strong convexity. For squared Euclidean movement, we determine the exact optimal dimension-free competitive ratio for convex functions satisfying λ-quadratic growth. We establish a tight competitive analysis of R-OBD and a matching lower bound against randomized algorithms, and further quantify the efect of relaxing strong convexity on the optimal competitive ratio.

## 1.1 Main results

We establish dimension-free competitive guarantees under Assumptions 1 and $2 ,$ where dimension-free means that the guarantees hold for all dimensions d and feasible sets .

Both results rely on the quadratic growth condition (Zhang et al., 2021). Specifically, a hitting cost $f _ { t } ( \cdot )$ with minimizer $\mathbf { v } _ { t }$ satisfies $\lambda _ { t } .$ -quadratic growth defined as

$$
f _ { t } ( \mathbf { x } ) - f _ { t } ( \mathbf { v } _ { t } ) \geq \frac { \lambda _ { t } } { 2 } \| \mathbf { x } - \mathbf { v } _ { t } \| ^ { 2 }\tag{1}
$$

for all $\mathbf { x }$ in the feasbile domain. For Euclidean movement, $\mathbf { v } _ { t }$ is a global minimizer and the condition holds on $\mathbb { R } ^ { d }$ , whereas for squared Euclidean movement, $\mathbf { v } _ { t }$ minimizes $f _ { t } ( \cdot )$ over $\mathcal { X }$ and the condition holds on $\mathcal { X }$

Euclidean movement. For Euclidean movement, let $\mathcal { F } _ { \kappa }$ denote the function class satisfying Assumptions 1 to 3 and 5, i.e., each hitting cost $f _ { t } ( \cdot )$ is $L _ { t } .$ -smooth and satisfies λ<sub>t</sub>-quadratic growth on $\mathbb { R } ^ { d }$ , with $L _ { t } \le \kappa \lambda _ { t }$ . The parameters may vary across rounds. We propose Clipped Proximal Descent (CPD), summarized in Algorithm 1, which combines a balanced proximal step with a clipped correction toward the current minimizer.

Theorem 1 For any $\kappa \geq 1$ , CPD achieves a competitive ratio of $\frac { 3 2 } { 7 } ( 1 + \kappa ^ { 1 / 3 } )$ on $\mathcal { F } _ { \kappa }$

Theorem 1 improves the $O ( { \sqrt { \kappa } } )$ upper bound of Argue et al. (2020a) and matches their $\Omega ( \kappa ^ { 1 / 3 } )$ lower bound (Argue et al., 2020a). The dependence on κ is therefore optimal.

Squared Euclidean movement. For $\begin{array} { r } { m ( \mathbf { x } , \mathbf { y } ) = \frac { 1 } { 2 } \| \mathbf { x } - \mathbf { y } \| ^ { 2 } } \end{array}$ , let $\mathcal { Q } _ { \lambda }$ denote the class of instances where each hitting cost $f _ { t } ( \cdot )$ attains its minimum at $\mathbf { v } _ { t } \in \mathcal { X }$ and satisfies λ-quadratic growth on $\mathcal { X } .$ . Unlike in the Euclidean movement setting, we do not require smoothness of the hitting costs here. We employ Regularized Online Balanced Descent (R-OBD) (Goel et al., 2019) and develop a novel tight competitive analysis. We further construct an oblivious distribution of hitting costs to establish a matching lower bound against randomized online algorithms.

Let $C ^ { * } ( \lambda )$ denote the optimal competitive ratio on $\mathcal { Q } _ { \lambda }$ . To characterize ${ \mathrm { i t } } ,$ we define

$$
\Lambda ( q , \delta ) = \frac { ( 1 + q \delta ^ { 2 } ) ^ { 2 } } { q ( 1 - \delta ^ { 2 } ) ( q + \delta ) } , \qquad \phi ( q ) = \operatorname* { m i n } _ { 0 \leq \delta < 1 } \Lambda ( q , \delta ) ,\tag{2}
$$

where $q > 0$ . The function $\phi ( q )$ is continuous and strictly decreasing. Let $q ^ { * } ( \lambda )$ denote the unique positive solution to $\phi ( q ) = \lambda$ . We establish the following matching upper and lower bounds.

Theorem 2 R-OBD achieves a competitive ratio of $1 + q ^ { * } ( \lambda )$ on $\mathcal { Q } _ { \lambda }$ . Moreover, any randomized online algorithm has a competitive ratio of at least $1 + q ^ { * } ( \lambda )$ on this class.

Combining the upper and lower bounds in Theorem 2, we obtain the exact optimal competitive ratio $C ^ { * } ( \lambda ) = 1 + q ^ { * } ( \lambda )$ We further derive an explicit upper bound and characterize its asymptotic behavior.

Corollary 3 The optimal competitive ratio satisfies

$$
C ^ { * } ( \lambda ) \leq 1 + \operatorname* { m i n } \left\{ \frac { 1 } { \sqrt { \lambda } } , \frac { 3 \sqrt { 3 } } { 2 \lambda } \right\} .
$$

Moreover,

$$
\operatorname * { l i m } _ { \lambda \to 0 ^ { + } } \sqrt \lambda C ^ { * } ( \lambda ) = 1 , \qquad \operatorname * { l i m } _ { \lambda \to \infty } \lambda \bigl ( C ^ { * } ( \lambda ) - 1 \bigr ) = { \frac { 3 \sqrt 3 } { 2 } } .
$$

Corollary 3 improves the previous upper bounds of $1 + 2 / \sqrt { \lambda }$ and $1 + 4 / \lambda$ established by Zhang et al. (2021). Moreover, compared with the optimal competitive ratio for strongly convex functions (Goel et al., 2019), our result quantifies the efect of relaxing strong convexity to quadratic growth.

## 1.2 Key techniques

Euclidean movement. The analysis of Online Balanced Descent (OBD) in previous work (Argue et al., 2020a) bounds the cost of each round through a decrease in the potential ${ \left\| { { \bf { x } } _ { t } } - { { \bf { u } } _ { t } } \right\| }$ , where $\mathbf { u } _ { t }$ is the comparator. A balanced step incurs hitting and movement costs proportional to its length, but the decrease in the potential depends on the direction of movement. In particular, when the step is nearly orthogonal to the direction toward the comparator, the potential may decrease much less than the movement cost, limiting the progress guaranteed by the balanced step alone. To overcome this challenge, we exploit the quadratic growth condition, which bounds the distance between the comparator and the current minimizer in terms of the comparator’s excess hitting cost. Consequently, a comparator with small excess hitting cost lies close to the minimizer, and moving toward the minimizer can provide additional progress even when the balanced step is poorly aligned with the comparator. Based on this property, we propose CPD, which supplements the balanced proximal step with a clipped correction toward the current minimizer. Our analysis combines the progress of these two steps and employs quadratic growth and smoothness to control the additional movement and the error introduced by the correction, establishing an $O ( \kappa ^ { 1 / 3 } )$ competitive ratio.

Squared Euclidean movement. For squared Euclidean movement, we develop a novel competitive analysis of R-OBD (Goel et al., 2019) under quadratic growth. The main challenge is that quadratic growth only relates the comparator’s excess hitting cost to its distance from the minimizer, without providing the curvature inequality between the online decision and the comparator available under strong convexity. Consequently, existing analyses relying on strong convexity cannot be directly applied. To overcome this dificulty, we apply the first-order optimality condition of R-OBD to both the comparator $\mathbf { u } _ { t }$ and the current minimizer $\mathbf { v } _ { t } ,$ and combine the resulting inequalities with a quadratic potential. To verify its optimality, we further construct a matching lower bound against randomized algorithms based on the equality conditions of the upper-bound analysis.

Organization. Section 2 reviews related work, and Section 3 states the assumptions and analytical tools. Section 4 proves Theorem 1; Section 5 provide the upper and lower bounds of squared Euclidean movement. Section 6 concludes the paper. Supporting proofs are presented in Section A.

## 2 Related work

Chasing convex bodies and functions. Friedman and Linial (1993) introduce body chasing, establish an $\Omega ( { \sqrt { d } } )$ lower bound, and give a constant-competitive algorithm for $d = 2$ . Bansal et al. (2015); Antoniadis and Schewior (2018) prove the optimal competitive ratio for function chasing for $d = 1$ . For nested sets $( S _ { t } \subseteq S _ { t - 1 } )$ case, Bansal et al. (2018) establish a ratio depending only on d, and Argue et al. (2019) improve it to $O ( d \log d )$ . For arbitrary sets case, Bubeck et al. (2019) obtain the first $2 ^ { O ( d ) }$ ratio and reduce function chasing in $\mathbb { R } ^ { d }$ to body chasing in $\mathbb { R } ^ { d + 1 }$ Subsequently, Argue et al. (2020b) obtain an $O ( \operatorname* { m i n } \{ d , \sqrt { d \log T } \} )$ ratio for body chasing, and Sellke (2020) establish the same guarantee for both problems. These results make general chasing possible in every fixed dimension, but their guarantees depend on dimension d. This motivates the search for stronger guarantees when the cost functions have additional structure.

Dimension-free guarantees. For Euclidean distance movement, Chen et al. (2018) achieve a constant competitive ratio for locally polyhedral hitting costs. For strongly convex and smooth hitting costs with condition number at most κ, Argue et al. (2020a) obtain an $O ( { \sqrt { \kappa } } )$ ratio, and a lower bound $\Omega ( \kappa ^ { 1 / 3 } )$ for all online algorithms. For squared Euclidean movement $\begin{array} { r } { m ( \mathbf { x } , \mathbf { y } ) = \frac { 1 } { 2 } \| \mathbf { x } - \mathbf { y } \| _ { 2 } ^ { 2 } } \end{array}$ and λ-strongly convex hitting costs, Goel and Wierman (2019) establish a $3 + O ( 1 / \lambda )$ ratio. Later, Goel et al. (2019) attain the optimal ratio of $( 1 + \sqrt { 1 + 4 / \lambda } ) / 2$ for the λ-strongly convex hitting costs.

Quadratic growth condition. Quadratic growth is a weaker assumption than strong convexity and has been widely used to establish linear convergence in static convex optimization (Necoara et al., 2019; Drusvyatskiy and Lewis, 2018). For squared Euclidean movement and quasi-convex hitting costs satisfying λ-quadratic growth, Goel et al. (2019) prove that G-OBD is $O ( \lambda ^ { - 1 / 2 } )$ )-competitive as λ approaches 0. Lin et al. (2020) establish the ratio of max $\{ 1 + 6 / \lambda , 4 \}$ under λ-quadratic growth. Zhang et al. (2021) improve this bound to $1 + 4 / \lambda$ , and obtain $\mathrm { ~ a ~ } 1 + 2 / \sqrt { \lambda }$ ratio for convex hitting costs.

## 3 Preliminaries

Notation. and $\langle \cdot , \cdot \rangle$ denote the Euclidean norm and the standard inner product, respectively. For a nonempty closed convex set $S \subseteq \mathbb { R } ^ { d }$ , let Π<sub>S</sub> denote Euclidean projection onto S. We introduce the following standard assumptions (Argue et al., 2020b).

Assumption 1 The decision set $\mathcal { X } \subseteq \mathbb { R } ^ { d }$ is nonempty, closed, and convex.

Assumption 2 For every round t, the hitting cost $f _ { t } ( \cdot ) : \mathbb { R } ^ { d } \mapsto [ 0 , \infty )$ is a finite-valued convex function.

Euclidean movement. In this paper, we consider two types of movement costs: Euclidean movement $m ( \mathbf { x } , \mathbf { y } ) \ = \ \| \mathbf { x } - \mathbf { y } \|$ in Section 4 and squared Euclidean movement $\begin{array} { r } { m ( \mathbf { x } , \mathbf { y } ) = \frac { 1 } { 2 } \| \mathbf { x } - \mathbf { y } \| ^ { 2 } } \end{array}$ in Section 5. Two types of movements are commonly used in online optimization and control (Argue et al., 2020a; Goel et al., 2019; Zhang et al., 2021).

Quadratic growth and strong convexity. Quadratic growth is a standard relaxation of strong convexity, because strong convexity with parameter λ implies λ-quadratic growth at a minimizer, while the converse fails (Necoara et al., 2019; Drusvyatskiy and Lewis, 2018). For the Euclidean movement, each hitting cost $f _ { t } ( \cdot )$ satisfies the following assumptions:

Assumption 3 For every t, the function $f _ { t } ( \cdot )$ attains its global minimum at $\mathbf { v } _ { t } \in \mathbb { R } ^ { d }$ and satisfies λ<sub>t</sub>-quadratic growth defined in (1) on $\mathbb { R } ^ { d }$ for some $\lambda _ { t } > 0$

For the squared Euclidean movement, each hitting cost $f _ { t } ( \cdot )$ satisfies the following assumptions.

Assumption 4 For every t, the function $f _ { t } ( \cdot )$ attains its minimum over $\mathcal { X }$ at $\mathbf { v } _ { t } \in \mathcal { X }$ and satisfies λ-quadratic growth defined in (1) on at this minimizer, in the sense defined in Section 1.1.

Assumption 3 is used for Section 4 (standard Euclidean movement), while Assumption 4 is used for Section 5 (squared Euclidean movement). In both cases, quadratic growth is stated relative to a fixed minimizer and therefore implies uniqueness of that minimizer.

Smoothness and a gradient bound. Apart from Assumption 3, the function $f _ { t } ( \cdot )$ also satisfies the following smoothness assumption, which is standard in chasing convex functions (Argue et al., 2020a).

Assumption 5 (L<sub>t</sub>-Smoothness) For every t, the function $f _ { t } ( \cdot )$ is diferentiable on $\mathbb { R } ^ { d }$ and there exists $L _ { t } > 0$ such that

$$
\begin{array} { r } { \| \nabla f _ { t } ( \mathbf { x } ) - \nabla f _ { t } ( \mathbf { y } ) \| \leq L _ { t } \| \mathbf { x } - \mathbf { y } \| , \ f o r \ a l l \ \mathbf { x } , \mathbf { y } \in \mathbb { R } ^ { d } . } \end{array}
$$

The parameters satisfy $L _ { t } \ \leq \ \kappa \lambda _ { t }$ for all t and may vary across rounds, where λ is the quadratic growth parameter in Assumption 3 and κ is a fixed condition number. The condition of L<sub>t</sub>-smoothness and $\lambda _ { t }$ quadratic gworth implies that $\kappa \geq 1$

The following gradient bound is a standard consequence of the descent inequality (Bubeck, 2015, Lemma 3.4).

Lemma 4 Let $f ( \cdot ) : \mathbb { R } ^ { d } \mapsto \mathbb { R }$ be L-smooth for some constant $L > 0$ , and suppose that $f ( \cdot )$ attains its global minimum value $f ^ { \star }$ . Then, for every $\mathbf { x } \in \mathbb { R } ^ { d }$

$$
\begin{array} { r } { \| \nabla f ( { \mathbf x } ) \| ^ { 2 } \leq 2 L ( f ( { \mathbf x } ) - f ^ { \star } ) . } \end{array}
$$

Potential method. Fix a comparator sequence $\{ \mathbf { u } \} _ { t = 0 } ^ { T }$ with ${ \bf u } _ { 0 } = { \bf x } _ { 0 }$ . Following the standard potential argument (Argue et al., 2020a), for either movement cost it sufices to construct a nonnegative potential $\Phi _ { t }$ with $\Phi _ { 0 } = 0$ such that

$$
f _ { t } ( \mathbf { x } _ { t } ) + m ( \mathbf { x } _ { t } , \mathbf { x } _ { t - 1 } ) + \Phi _ { t } - \Phi _ { t - 1 } \leq C \big ( f _ { t } ( \mathbf { u } _ { t } ) + m ( \mathbf { u } _ { t } , \mathbf { u } _ { t - 1 } ) \big ) ,
$$

for every round and every comparator sequence. Therefore, we can prove the C-competitiveness by summing over $t = 1 , \dots , T$

$$
\sum _ { t = 0 } ^ { T } f _ { t } ( \mathbf { x } _ { t } ) + m ( \mathbf { x } _ { t } , \mathbf { x } _ { t - 1 } ) \leq \sum _ { t = 0 } ^ { T } f _ { t } ( \mathbf { x } _ { t } ) + m ( \mathbf { x } _ { t } , \mathbf { x } _ { t - 1 } ) + \Phi _ { T } - \Phi _ { 0 } \leq C \left( \sum _ { t = 0 } ^ { T } f _ { t } ( \mathbf { u } _ { t } ) + m ( \mathbf { u } _ { t } , \mathbf { u } _ { t - 1 } ) \right) .
$$

Randomization. For either movement cost, the one-round cost is jointly convex in the current and previous decisions. Consequently, randomization can not improve the optimal competitive ratio against oblivious inputs, as observed in Bansal et al. (2015); Argue et al. (2020a).

## 4 Euclidean movement

This section presents our proposed method and proves Theorem 1 for instances in $\mathcal { F } _ { \kappa }$ . The algorithm combines a balanced proximal step with a clipped correction and is summarized in Algorithm 1.

Algorithm 1: Clipped proximal descent   
Input: feasible set $x ,$ initial decision $\mathbf { x } _ { 0 } \in { \mathcal { X } } .$ , and $\kappa \geq 1$ , and parameters $r , \beta$ from   
(3).   
1 for $t = 1 , \dots , T$ do   
2 Observe $f _ { t } ( \cdot )$ and compute $\mathbf { v } _ { t }$   
3 $\begin{array} { r } { \mathbf { v } _ { t } ^ { \mathcal { X } } = \Pi _ { \arg \operatorname* { m i n } _ { \mathbf { z } \in \mathcal { X } } g _ { t } ( \mathbf { z } ) } \big \{ \mathbf { x } _ { t - 1 } \big \} } \end{array}$   
4 if $g _ { t } ( \mathbf { v } _ { t } ^ { \mathcal { X } } ) \geq \bar { \beta } \| \mathbf { v } _ { t } ^ { \mathcal { X } } - \mathbf { x } _ { t - 1 } \|$ then   
5 $\mathbf { x } _ { t } = \mathbf { v } _ { t } ^ { \mathcal { X } }$   
6 else   
7 Choose the smallest $\mu _ { t } > 0$ satisfying $g _ { t } ( \mathbf { y } _ { t } ( \mu _ { t } ) ) = \beta \| \mathbf { y } _ { t } ( \mu _ { t } ) - \mathbf { x } _ { t - 1 } \|$   
$\mathbf { y } _ { t } = \mathbf { y } _ { t } ( \mu _ { t } ) , s _ { t } = \| \mathbf { y } _ { t } - \mathbf { x } _ { t - 1 } \| , \gamma _ { t } = { \frac { r ^ { 2 } s _ { t } } { s _ { t } + \beta \mu _ { t } } } , \eta _ { t } = \operatorname* { m i n } \left\{ \gamma _ { t } , { \frac { r s _ { t } } { \| \mathbf { y } _ { t } - \mathbf { v } _ { t } \| } } \right\} ,$   
$\mathbf { x } _ { t } = \Pi _ { \mathcal { X } \cap \left\{ \mathbf { z } : g _ { t } ( \mathbf { z } ) \leq g _ { t } ( \mathbf { y } _ { t } ) \right\} } \left\{ ( 1 - \eta _ { t } ) \mathbf { y } _ { t } + \eta _ { t } \mathbf { v } _ { t } \right\}$   
8 end   
9 Play $\mathbf { x } _ { t }$ and incur $f _ { t } ( \mathbf { x } _ { t } ) + \| \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } \|$   
10 end

## 4.1 Clipped Proximal Descent

At each round t, the algorithm starts from $\mathbf { x } _ { t - 1 } \in \mathcal { X }$ and observes the hitting cost $f _ { t } ( \cdot )$ It computes the global minimizer $\mathbf { v } _ { t }$ from Assumption 3 and defines $g _ { t } ( \mathbf { z } ) = f _ { t } ( \mathbf { z } ) - f _ { t } ( \mathbf { v } _ { t } )$ This standard normalization Argue et al. (2020a) preserves the growth and smoothness parameters, so $g _ { t } ( \mathbf { v } _ { t } ) = 0$ and $L _ { t } \leq \kappa \lambda _ { t }$ . By Assumptions 3 and 5 and Theorem 4, for every $\mathbf { z } \in \mathbb { R } ^ { d }$ we have

$$
g _ { t } ( { \mathbf { z } } ) \geq \frac { \lambda _ { t } } { 2 } \| { \mathbf { z } } - { \mathbf { v } } _ { t } \| ^ { 2 } , \qquad \| \nabla g _ { t } ( { \mathbf { z } } ) \| ^ { 2 } \leq 2 L _ { t } g _ { t } ( { \mathbf { z } } ) .
$$

Then we show that a competitive guarantee for these shifted costs also holds for the hitting costs $f _ { t } ( \cdot )$

For the update, we set

$$
r = { \frac { 1 } { 2 \kappa ^ { 1 / 3 } } } , \qquad \beta = 1 + 3 r , \qquad C = { \frac { 3 2 } { 7 } } \bigl ( 1 + \kappa ^ { 1 / 3 } \bigr ) = { \frac { 8 ( 2 + 4 r ) } { 7 r } } ,\tag{3}
$$

where $\begin{array} { r } { 0 < r \le { \frac { 1 } { 2 } } } \end{array}$ and $\begin{array} { r } { \kappa r ^ { 3 } = \frac { 1 } { 8 } } \end{array}$

We next find the constrained minimizer $\mathbf { v } _ { t } ^ { \mathcal { X } }$ closest to $\mathbf { x } _ { t - 1 }$ . If $g _ { t } ( \mathbf { v } _ { t } ^ { \mathcal { X } } ) \geq \beta \| \mathbf { v } _ { t } ^ { \mathcal { X } } - \mathbf { x } _ { t - 1 } \|$ 2 it sets $\mathbf { x } _ { t } = \mathbf { v } _ { t } ^ { \mathcal { X } }$ . Every feasible comparator pays a hitting cost of at least $g _ { t } ( \mathbf { v } _ { t } ^ { \mathcal { X } } )$ , and the condition bounds the movement by $g _ { t } ( \mathbf { v } _ { t } ^ { \mathcal { X } } ) / \beta$

Otherwise, the algorithm enters the balanced case. For $\mu > 0$ , consider the proximal point

$$
\mathbf { y } _ { t } ( \mu ) = \underset { \mathbf { z } \in \mathcal { X } } { \arg \operatorname* { m i n } } \left\{ g _ { t } ( \mathbf { z } ) + \frac { \Vert \mathbf { z } - \mathbf { x } _ { t - 1 } \Vert ^ { 2 } } { 2 \mu } \right\} .
$$

The quadratic term penalizes distance from the previous decision and $\mu$ controls its weight relative to the hitting cost. Choose the smallest $\mu _ { t } > 0$ such that $g _ { t } ( \mathbf { y } _ { t } ( \mu _ { t } ) ) = \beta \| \mathbf { y } _ { t } ( \mu _ { t } ) -$ $\mathbf { x } _ { t - 1 } \Vert$ , and set $\mathbf { y } _ { t } = \mathbf { y } _ { t } ( \mu _ { t } )$ and $s _ { t } = \| \mathbf y _ { t } - \mathbf x _ { t - 1 } \|$ . Thus $g _ { t } ( \mathbf { y } _ { t } ) = \beta s _ { t }$ : the hitting cost at the proximal point equals $\beta$ times the Euclidean movement needed to reach it.

![](images/804e3a9980beb6267efcc934e14d425b10b4d40150953a9e306a0928a4fb7f65.jpg)  
Figure 1: A balanced round of Algorithm 1 with $\mathcal { X } = \mathbb { R } ^ { d }$ . The proximal step moves $\mathbf { x } _ { t - 1 }$ to $\mathbf { y } _ { t }$ , and the clipped correction moves $\mathbf { y } _ { t }$ toward $\mathbf { v } _ { t }$

Next, take a clipped step from $\mathbf { y } _ { t }$ toward the global minimizer $\mathbf { v } _ { t } ,$ , using the fraction $\eta _ { t }$ defined by

$$
\gamma _ { t } = \frac { r ^ { 2 } s _ { t } } { s _ { t } + \beta \mu _ { t } } , \qquad \eta _ { t } = \operatorname* { m i n } \left\{ \gamma _ { t } , \frac { r s _ { t } } { \| \mathbf { y } _ { t } - \mathbf { v } _ { t } \| } \right\} .
$$

The resulting point is $( 1 - \eta _ { t } ) \mathbf { y } _ { t } + \eta _ { t } \mathbf { v } _ { t }$ . The cap $\eta _ { t } \leq \gamma _ { t }$ controls the localization error in the analysis, while $\eta _ { t } \| \mathbf { y } _ { t } - \mathbf { v } _ { t } \| \leq r s _ { t }$ limits the additional movement to a fraction r of the proximal step.

Finally, project this point onto $\mathcal { X } \cap \{ \mathbf { z } : g _ { t } ( \mathbf { z } ) \leq g _ { t } ( \mathbf { y } _ { t } ) \}$ to obtain $\mathbf { x } _ { t }$ . The projection ensures feasibility and $g _ { t } ( \mathbf { x } _ { t } ) \leq g _ { t } ( \mathbf { y } _ { t } )$ . If $\mathcal { X } = \mathbb { R } ^ { d }$ , Assumption 2 gives

$$
\begin{array} { r } { g _ { t } \big ( ( 1 - \eta _ { t } ) \mathbf { y } _ { t } + \eta _ { t } \mathbf { v } _ { t } \big ) \leq ( 1 - \eta _ { t } ) g _ { t } ( \mathbf { y } _ { t } ) + \eta _ { t } g _ { t } ( \mathbf { v } _ { t } ) \leq g _ { t } ( \mathbf { y } _ { t } ) , } \end{array}
$$

so the projection leaves the corrected point unchanged, as illustrated in Figure 1. The complete update is summarized in Algorithm 1. It uses exact minimization, proximal, projection oracles and a one-dimensional search for $\mu _ { t }$ Its inputs are $\kappa , \mathbf { x } _ { 0 } , { \mathcal { X } }$ , and $r , \beta$ can be computed using κ.

## 4.2 Theoretical analysis

Firstly we demonstrate that the algorithm is well defined and the parameters are bounded using the following lemma.

Lemma 5 In the case of line 7 in Algorithm 1, the smallest positive balancing parameter $\mu _ { t }$ exists, $s _ { t } > 0$ , and $g _ { t } ( \mathbf { y } _ { t } ) = \beta s _ { t } > 0$ . Moreover, $\mathbf { y } _ { t } \neq \mathbf { v } _ { t }$ and $0 < \eta _ { t } \leq \gamma _ { t } < r ^ { 2 } \leq \frac { 1 } { 4 }$

The proof of Theorem 5 is deferred to Section A.1. After the existence of $\mu _ { t }$ is established, one-dimensional bisection can be used to find it eficiently in its implementation. The remaining parameters can be computed in closed form, and $\mathbf { x } _ { t }$ can be computed using a projection oracle. Therefore, every update of Algorithm 1 is well defined.

We next show that the competitive ratio of Algorithm 1 is $C$ in (3), which proves Theorem 1.

Proof Algorithm 1 evaluates $g _ { t } ( \mathbf { z } ) = f _ { t } ( \mathbf { z } ) - f _ { t } ( \mathbf { v } _ { t } )$ instead of $f _ { t } ( \mathbf { z } )$ , which is a standard normalization (Argue et al., 2020a). We first show that this normalization preserves the competitive ratio. For any comparator sequence $\{ { \mathbf { u } } _ { t } \} _ { t = 0 } ^ { T }$ with $\mathbf { u } _ { 0 } = \mathbf { x } _ { 0 }$ , if a online algorithm achieves a C-competitive guarantee on the shifted costs $g _ { t }$ , namely

$$
\begin{array} { r l } {  { \sum _ { t = 1 } ^ { T } \bigl ( g _ { t } ( \mathbf { x } _ { t } ) + \| \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } \| \bigr ) \le C \sum _ { t = 1 } ^ { T } \bigl ( g _ { t } ( \mathbf { u } _ { t } ) + \| \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } \| \bigr ) , } } \end{array}
$$

then it also achieves a C-competitive guarantee on the original costs $f _ { t } ( \cdot )$ , since

$$
\begin{array} { l } { \displaystyle \sum _ { t = 1 } ^ { T } \left( f _ { t } ( \mathbf { x } _ { t } ) + \| \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } \| \right) = \displaystyle \sum _ { t = 1 } ^ { T } \left( g _ { t } ( \mathbf { x } _ { t } ) + \| \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } \| \right) + \displaystyle \sum _ { t = 1 } ^ { T } f _ { t } ( \mathbf { v } _ { t } ) } \\ { \displaystyle \leq C \sum _ { t = 1 } ^ { T } \left( g _ { t } ( \mathbf { u } _ { t } ) + \| \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } \| \right) + \displaystyle \sum _ { t = 1 } ^ { T } f _ { t } ( \mathbf { v } _ { t } ) \leq C \sum _ { t = 1 } ^ { T } \left( f _ { t } ( \mathbf { u } _ { t } ) + \| \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } \| \right) } \end{array}
$$

where the last inequality uses $C \geq 1$ and $f _ { t } ( \mathbf { v } _ { t } ) \geq 0$ from Assumption 2. Therefore, it sufices to analyse the competitive ratio for the shifted costs $g _ { t } ( \cdot )$ . We first establish the amortized bound

$$
\begin{array} { r } { g _ { t } ( \mathbf { x } _ { t } ) + \left. \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } \right. + C \left( \left. \mathbf { x } _ { t } - \mathbf { u } _ { t } \right. - \left. \mathbf { x } _ { t - 1 } - \mathbf { u } _ { t } \right. \right) \leq C g _ { t } ( \mathbf { u } _ { t } ) . } \end{array}\tag{4}
$$

Suppose first that the condition in line 4 holds. Then $\mathbf { x } _ { t } = \mathbf { v } _ { t } ^ { \mathcal { X } }$ and

$$
\| \mathbf { v } _ { t , \mathcal { X } } - \mathbf { x } _ { t - 1 } \| \leq \frac { g _ { t } ( \mathbf { v } _ { t } ^ { \mathcal { X } } ) } { \beta } .
$$

The triangle inequality gives

$$
\begin{array} { r l } & { \quad g _ { t } ( { \mathbf x } _ { t } ) + \| { \mathbf x } _ { t } - { \mathbf x } _ { t - 1 } \| + C \left( \| { \mathbf x } _ { t } - { \mathbf u } _ { t } \| - \| { \mathbf x } _ { t - 1 } - { \mathbf u } _ { t } \| \right) } \\ & { \leq g _ { t } ( { \mathbf v } _ { t } ^ { \mathcal { X } } ) + ( 1 + C ) \| { \mathbf v } _ { t } ^ { \mathcal { X } } - { \mathbf x } _ { t - 1 } \| \leq \displaystyle \left( 1 + \frac { 1 + C } { \beta } \right) g _ { t } ( { \mathbf v } _ { t } ^ { \mathcal { X } } ) \leq C g _ { t } ( { \mathbf v } _ { t } ^ { \mathcal { X } } ) . } \end{array}
$$

The last inequality is by $\begin{array} { r } { C = \frac { 3 2 } { 7 } ( 1 + \kappa ^ { 1 / 3 } ) } \end{array}$ , and

$$
C - 1 - ( 1 + C ) \beta = \frac { 3 \left( \frac { 6 8 } { 2 1 } \kappa ^ { 1 / 3 } + \frac { 2 5 } { 7 } \right) } { 2 \kappa ^ { 1 / 3 } + 3 } \ge 0 .
$$

Since $\mathbf { v } _ { t } ^ { \mathcal { X } }$ minimizes $g _ { t }$ over $\mathcal { X } _ { : }$ , the preceding bound is at most $C g _ { t } ( \mathbf { v } _ { t } ^ { \mathcal { X } } ) \leq C g _ { t } ( \mathbf { u } _ { t } )$ . This proves (4) in the first case.

Suppose that the condition in line 4 of Algorithm 1 is false. The projection in line $7$ ensures $g _ { t } ( \mathbf { x } _ { t } ) \leq g _ { t } ( \mathbf { y } _ { t } )$ . Its projection set contains $\mathbf { y } _ { t }$ , so the expansion gives

$$
\begin{array} { r } { \| \mathbf { x } _ { t } - \mathbf { y } _ { t } \| \leq \| ( 1 - \eta _ { t } ) \mathbf { y } _ { t } + \eta _ { t } \mathbf { v } _ { t } - \mathbf { y } _ { t } \| = \eta _ { t } \| \mathbf { y } _ { t } - \mathbf { v } _ { t } \| \leq r \| \mathbf { y } _ { t } - \mathbf { x } _ { t - 1 } \| . } \end{array}
$$

Combining this estimate with the triangle inequality and $g _ { t } ( \mathbf { y } _ { t } ) = \beta \| \mathbf { y } _ { t } - \mathbf { x } _ { t - 1 } \|$ , we obtain

$$
\begin{array} { r l r } {  { g _ { t } ( \mathbf { x } _ { t } ) + \| \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } \| \leq g _ { t } ( \mathbf { y } _ { t } ) + \| \mathbf { y } _ { t } - \mathbf { x } _ { t - 1 } \| + \| \mathbf { x } _ { t } - \mathbf { y } _ { t } \| } } \\ & { } & { \leq ( 2 + 4 r ) \| \mathbf { y } _ { t } - \mathbf { x } _ { t - 1 } \| = C \frac { 7 } { 1 6 \kappa ^ { 1 / 3 } } \| \mathbf { y } _ { t } - \mathbf { x } _ { t - 1 } \| . } \end{array}
$$

We next bound this term use the following lemma:

Lemma 6 Suppose that the condition in Line 4 of Algorithm 1 is false. Then every $\mathbf { u } _ { t } \in \mathcal { X }$ satisfies

$$
\| \mathbf { x } _ { t - 1 } - \mathbf { u } _ { t } \| - \| \mathbf { x } _ { t } - \mathbf { u } _ { t } \| + g _ { t } ( \mathbf { u } _ { t } ) \geq \frac { 7 } { 1 6 \kappa ^ { 1 / 3 } } \| \mathbf { y } _ { t } - \mathbf { x } _ { t - 1 } \| .
$$

Combining this lemma with the preceding cost estimate yields

$$
g _ { t } ( \mathbf { x } _ { t } ) + \left\| \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } \right\| \leq C \left( \left\| \mathbf { x } _ { t - 1 } - \mathbf { u } _ { t } \right\| - \left\| \mathbf { x } _ { t } - \mathbf { u } _ { t } \right\| + g _ { t } ( \mathbf { u } _ { t } ) \right) ,
$$

which proves (4) in the second case.

To sum the amortized bounds, apply the triangle inequality

$$
\left\| \mathbf { x } _ { t - 1 } - \mathbf { u } _ { t } \right\| \leq \left\| \mathbf { x } _ { t - 1 } - \mathbf { u } _ { t - 1 } \right\| + \left\| \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } \right\| .
$$

Substituting into (4) gives

$$
\begin{array} { r } { g _ { t } ( \mathbf { x } _ { t } ) + \left\| \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } \right\| + C \left( \left\| \mathbf { x } _ { t } - \mathbf { u } _ { t } \right\| - \left\| \mathbf { x } _ { t - 1 } - \mathbf { u } _ { t - 1 } \right\| \right) \leq C \left( g _ { t } ( \mathbf { u } _ { t } ) + \left\| \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } \right\| \right) . } \end{array}
$$

Summing over $t = 1 , \dots , T$ and using $\mathbf { u } _ { 0 } = \mathbf { x } _ { 0 }$ , we obtain

$$
\begin{array} { r l } {  { \sum _ { t = 1 } ^ { T } ( g _ { t } ( \mathbf { x } _ { t } ) + \| \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } \| ) \le C \sum _ { t = 1 } ^ { T } ( g _ { t } ( \mathbf { u } _ { t } ) + \| \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } \| ) . } } \end{array}
$$

This concludes the proof of the C-competitive guarantee for Algorithm 1.

The strongly convex construction in (Argue et al., 2020a, Proposition 13) gives the matching $\Omega ( \kappa ^ { 1 / 3 } )$ lower bound. Its independent random signs also establish the expected lower bound against randomized algorithms, completing the minimax claim.

## 5 Squared movement

This section establishes the upper and lower bounds in Theorem 2 by the R-OBD (Goel et al., 2019) algorithm. Throughout, we consider instances in $\mathcal { Q } _ { \lambda }$ , with minimizers $\mathbf { v } _ { t }$ specified in Assumption 4.

## 5.1 Regularized online balanced descent

Given the quadratic growth parameter λ and the feasible set $\mathcal { X } .$ , choose $q > 0$ satisfying

$$
\operatorname* { m i n } _ { 0 \leq d < 1 } \Lambda ( q , d ) = \lambda ,
$$

where Λ is defined in (2). Choose $\delta = \arg \operatorname* { m i n } _ { 0 \leq d < 1 } \Lambda ( q , d )$ and set the weights $a , b > 0$ according to (5). At each round t, R-OBD updates

$$
{ \bf x } _ { t } = \underset { { \bf x } \in \mathcal { X } } { \arg \operatorname* { m i n } } \left. f _ { t } ( { \bf x } ) + \frac { a } { 2 } \| { \bf x } - { \bf x } _ { t - 1 } \| ^ { 2 } + \frac { b } { 2 } \| { \bf x } - { \bf v } _ { t } \| ^ { 2 } \right. ,
$$

where $\mathbf { v } _ { t }$ minimizes $f _ { t } ( \cdot )$ over $\mathcal { X }$ and $\mathbf { x } _ { t - 1 }$ is the previous decision.

By completing the square norms, R-OBD can be viewed as the equivalent proximal update

$$
\mathbf { x } _ { t } = \underset { \mathbf { x } \in \mathcal { X } } { \arg \operatorname* { m i n } } \{ f _ { t } ( \mathbf { x } ) + ( a + b ) / 2 \| \mathbf { x } - \mathbf { c } _ { t } \| ^ { 2 } \} , \quad \mathbf { c } _ { t } = ( a \mathbf { x } _ { t - 1 } + b \mathbf { v } _ { t } ) / ( a + b ) .
$$

The center $\mathbf { c } _ { t }$ is a convex combination of $\mathbf { x } _ { t - 1 }$ and $\mathbf { v } _ { t }$ . Our analysis requires only convexity and quadratic growth of $f _ { t } ( \cdot )$ . In particular, the bound $f _ { t } ( \mathbf { v } _ { t } ) - f _ { t } ( \mathbf { u } ) \leq - \lambda / 2 \| \mathbf { u } - \mathbf { v } _ { t } \| ^ { 2 }$ is used to establish the competitive guarantee below.

Lemma 7 Let $q > 0$ and $0 \leq \delta < 1$ , and set the parameters of R-OBD as

$$
h = 1 + q \delta ^ { 2 } , \qquad a = \frac { h } { 1 - \delta } , \qquad b = \frac { a ^ { 2 } } { q ( 1 + \delta ) } .\tag{5}
$$

$I f \lambda \ge \Lambda ( q , \delta )$ , then R-OBD with weights a and b is $( 1 + q )$ -competitive on $\mathcal { Q } _ { \lambda }$

Proof Let $c = C a / ( q ( 1 + \delta ) )$ . Fix any comparator sequence $\{ { \mathbf { u } } _ { t } \} _ { t = 0 } ^ { T }$ with $\mathbf { u } _ { 0 } = \mathbf { x } _ { 0 }$ , and define

$$
\Phi _ { t } = \frac { c } { 2 } \| \mathbf { x } _ { t } - \mathbf { u } _ { t } \| ^ { 2 } .
$$

We first establish the amortized bound

$$
f _ { t } ( \mathbf { x } _ { t } ) + \frac { 1 } { 2 } \| \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } \| ^ { 2 } + \Phi _ { t } - \Phi _ { t - 1 } \leq C \left( f _ { t } ( \mathbf { u } _ { t } ) + \frac { 1 } { 2 } \| \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } \| ^ { 2 } \right) .\tag{6}
$$

In round $t ,$ optimality of $\mathbf { x } _ { t }$ imply that, for every $\mathbf { y } \in \mathcal { X }$

$$
f _ { t } ( \mathbf { x } _ { t } ) + \frac { a } { 2 } \| \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } \| ^ { 2 } + \frac { b } { 2 } \| \mathbf { x } _ { t } - \mathbf { v } _ { t } \| ^ { 2 } \leq f _ { t } ( \mathbf { y } _ { \varepsilon } ) + \frac { a } { 2 } \| \mathbf { y } _ { \varepsilon } - \mathbf { x } _ { t - 1 } \| ^ { 2 } + \frac { b } { 2 } \| \mathbf { y } _ { \varepsilon } - \mathbf { v } _ { t } \| ^ { 2 } ,
$$

where ${ \bf y } _ { \varepsilon } = { \bf x } _ { t } + \varepsilon ( { \bf y } - { \bf x } _ { t } )$ . Rearranging and expanding the quadratic terms gives

$$
f _ { t } ( \mathbf { x } _ { t } ) - f _ { t } ( \mathbf { y } ) \leq \varepsilon a \left. \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } , \mathbf { y } - \mathbf { x } _ { t } \right. + \varepsilon ^ { 2 } { \frac { a } { 2 } } \| \mathbf { y } - \mathbf { x } _ { t } \| ^ { 2 } + \varepsilon b \left. \mathbf { x } _ { t } - \mathbf { v } _ { t } , \mathbf { y } - \mathbf { x } _ { t } \right. + \varepsilon ^ { 2 } { \frac { b } { 2 } } \| \mathbf { y } - \mathbf { x } _ { t } \| ^ { 2 } .
$$

And by the convexity of $f _ { t } ( \cdot )$ , we have

$$
\varepsilon \left( f _ { t } ( \mathbf { x } _ { t } ) - f _ { t } ( \mathbf { y } ) \right) \leq f _ { t } ( \mathbf { x } _ { t } ) - f _ { t } ( \mathbf { y } ) \leq \varepsilon \left. a ( \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } ) + b ( \mathbf { x } _ { t } - \mathbf { v } _ { t } ) , \mathbf { y } - \mathbf { x } _ { t } \right. + \varepsilon ^ { 2 } { \frac { a + b } { 2 } } \| \mathbf { y } - \mathbf { x } _ { t } \| ^ { 2 } .
$$

This inequality holds for every $\varepsilon \in ( 0 , 1 )$ , and dividing by $\varepsilon$ and letting $\varepsilon \to 0$ gives

$$
f _ { t } ( \mathbf { x } _ { t } ) - f _ { t } ( \mathbf { y } ) \leq \left. a ( \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } ) + b ( \mathbf { x } _ { t } - \mathbf { v } _ { t } ) , \mathbf { y } - \mathbf { x } _ { t } \right. .
$$

Apply the above inequality at $\mathbf { y } = \mathbf { u } _ { t }$ and $\mathbf { y } = \mathbf { v } _ { t }$ , with weights $1 - \delta$ and δ respectively. Then by summing the two inequalities, we obtain

$$
f _ { t } ( \mathbf { x } _ { t } ) \leq ( 1 - \delta ) f _ { t } ( \mathbf { u } _ { t } ) + \delta f _ { t } ( \mathbf { v } _ { t } ) - \left. a ( \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } ) + b ( \mathbf { x } _ { t } - \mathbf { v } _ { t } ) , \mathbf { x } _ { t } - \mathbf { v } _ { t } - ( 1 - \delta ) ( \mathbf { u } _ { t } - \mathbf { v } _ { t } ) \right. .
$$

By Assumption 4,

$$
f _ { t } ( \mathbf { v } _ { t } ) - f _ { t } ( \mathbf { u } _ { t } ) \leq - \frac { \lambda } { 2 } \| \mathbf { u } _ { t } - \mathbf { v } _ { t } \| ^ { 2 } , \quad - q f _ { t } ( \mathbf { u } _ { t } ) \leq - q \frac { \lambda } { 2 } \| \mathbf { u } _ { t } - \mathbf { v } _ { t } \| ^ { 2 } - q f _ { t } ( \mathbf { v } _ { t } ) .
$$

Terms $( 1 - \delta ) f _ { t } ( \mathbf { u } _ { t } ) + \delta f _ { t } ( \mathbf { v } _ { t } ) = f _ { t } ( \mathbf { u } _ { t } ) + \delta ( f _ { t } ( \mathbf { v } _ { t } ) - f _ { t } ( \mathbf { u } _ { t } ) )$ are bounded by the first above inequality, and adding the second above inequality and the preceding inequalities yields

$$
\begin{array} { r l r } {  { f _ { t } ( \mathbf { x } _ { t } ) - C f _ { t } ( \mathbf { u } _ { t } ) \le \mathbf { \delta } - ( q + \delta ) \frac { \lambda } { 2 } \| \mathbf { u } _ { t } - \mathbf { v } _ { t } \| ^ { 2 } - q f _ { t } ( \mathbf { v } _ { t } ) } } \\ & { } & { \qquad -  a ( \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } ) + b ( \mathbf { x } _ { t } - \mathbf { v } _ { t } ) , \mathbf { x } _ { t } - \mathbf { v } _ { t } - ( 1 - \delta ) ( \mathbf { u } _ { t } - \mathbf { v } _ { t } )  } \end{array} .\tag{7}
$$

Here $C = 1 + q$ is the competitive ratio.

We next bound the potential diference. Expanding the square norm gives

$$
\begin{array} { l } { \displaystyle \Phi _ { t - 1 } + \frac { C } { 2 } \| \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } \| ^ { 2 } = \frac { C } { 2 } \| \mathbf { x } _ { t - 1 } - \mathbf { u } _ { t - 1 } \| ^ { 2 } + \displaystyle \frac { C } { 2 } \| \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } \| ^ { 2 } } \\ { \displaystyle \quad \quad = \frac { C } { 2 } \| \mathbf { x } _ { t - 1 } - \mathbf { u } _ { t } + \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } \| ^ { 2 } + \displaystyle \frac { C } { 2 } \| \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } \| ^ { 2 } } \\ { \displaystyle \quad = \frac { C } { 2 } \| \mathbf { x } _ { t - 1 } - \mathbf { u } _ { t } \| ^ { 2 } + \displaystyle \frac { C + C } { 2 } \| \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } \| ^ { 2 } + c \langle \mathbf { x } _ { t - 1 } - \mathbf { u } _ { t } , \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } \rangle } \\ { \displaystyle \quad \quad = \frac { C + C } { 2 } \| \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } + \displaystyle \frac { c } { C + c } ( \mathbf { x } _ { t - 1 } - \mathbf { u } _ { t } ) \| ^ { 2 } + \displaystyle \frac { c C } { 2 ( c + C ) } \| \mathbf { x } _ { t - 1 } - \mathbf { u } _ { t } \| ^ { 2 } . } \end{array}
$$

Using $c C / ( C + c ) = h$ gives

$$
\Phi _ { t - 1 } + \frac { C } { 2 } \| \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } \| ^ { 2 } \geq \frac { h } { 2 } \| \mathbf { x } _ { t - 1 } - \mathbf { u } _ { t } \| ^ { 2 } .
$$

Consequently,

$$
\Phi _ { t } - \Phi _ { t - 1 } - \frac { C } { 2 } \| \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } \| ^ { 2 } \leq \frac { c } { 2 } \| \mathbf { x } _ { t } - \mathbf { u } _ { t } \| ^ { 2 } - \frac { h } { 2 } \| \mathbf { x } _ { t - 1 } - \mathbf { u } _ { t } \| ^ { 2 } .
$$

To combine this bound with (7), we have

$$
\begin{array} { l } { \displaystyle f _ { t } ( { \mathbf x } _ { t } ) - C ( f _ { t } ( { \mathbf u } _ { t } ) + \frac { 1 } { 2 } \| { \mathbf u } _ { t } - { \mathbf u } _ { t - 1 } \| ^ { 2 } ) + \Phi _ { t } - \Phi _ { t - 1 } + \frac { 1 } { 2 } \| { \mathbf x } _ { t } - { \mathbf x } _ { t - 1 } \| ^ { 2 } } \\ { \displaystyle \le - ( q + \delta ) \frac { \lambda } { 2 } \| { \mathbf u } _ { t } - { \mathbf v } _ { t } \| ^ { 2 } - q f _ { t } ( { \mathbf v } _ { t } ) - \langle a ( { \mathbf x } _ { t } - { \mathbf x } _ { t - 1 } ) + b ( { \mathbf x } _ { t } - { \mathbf v } _ { t } ) , { \mathbf x } _ { t } - { \mathbf v } _ { t } - ( 1 - \delta ) ( { \mathbf u } _ { t } - { \mathbf v } _ { t } ) \rangle } \\ { + \frac { c } { 2 } \| { \mathbf x } _ { t } - { \mathbf u } _ { t } \| ^ { 2 } - \frac { h } { 2 } \| { \mathbf x } _ { t - 1 } - { \mathbf u } _ { t } \| ^ { 2 } + \frac { 1 } { 2 } \| { \mathbf x } _ { t } - { \mathbf x } _ { t - 1 } \| ^ { 2 } . } \end{array}
$$

Expand the squared norms terms by

$$
\begin{array} { r } { \mathbf { x } _ { t } - \mathbf { u } _ { t } = ( \mathbf { x } _ { t } - \mathbf { v } _ { t } ) - ( \mathbf { u } _ { t } - \mathbf { v } _ { t } ) , \quad \mathbf { x } _ { t - 1 } - \mathbf { u } _ { t } = ( \mathbf { x } _ { t } - \mathbf { v } _ { t } ) - ( \mathbf { u } _ { t } - \mathbf { v } _ { t } ) - ( \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } ) . } \end{array}
$$

And inner product terms involving $\mathbf { u } _ { t } - \mathbf { v } _ { t }$ are cancelled by $( 1 - \delta ) a = h$ and $( 1 - \delta ) b = c - h$ Thus,

$$
\begin{array} { l } { { \displaystyle f _ { t } ( { \bf x } _ { t } ) + \frac { 1 } { 2 } \| { \bf x } _ { t } - { \bf x } _ { t - 1 } \| ^ { 2 } - C ( f _ { t } ( { \bf u } _ { t } ) + \frac { 1 } { 2 } \| { \bf u } _ { t } - { \bf u } _ { t - 1 } \| ^ { 2 } ) + \Phi _ { t } - \Phi _ { t - 1 } } } \\ { { \le \ - \frac { ( 1 + \delta ) b } { 2 } \| { \bf x } _ { t } - { \bf v } _ { t } \| ^ { 2 } - \delta a \left. { \bf x } _ { t } - { \bf v } _ { t } , { \bf x } _ { t } - { \bf x } _ { t - 1 } \right. - \frac { q \delta ^ { 2 } } { 2 } \| { \bf x } _ { t } - { \bf x } _ { t - 1 } \| ^ { 2 } - q f _ { t } ( { \bf v } _ { t } ) } } \\ { { \displaystyle \quad + \frac { ( 1 - \delta ) b - ( q + \delta ) \lambda } { 2 } \| { \bf u } _ { t } - { \bf v } _ { t } \| ^ { 2 } } . } \end{array}
$$

Completing the first three terms in the right-hand side by substituting $( 1 + \delta ) b = a ^ { 2 } / q$ and $\| A + B \| ^ { 2 } = \| A \| + 2 \left. A , B \right. + \| B \| ^ { 2 }$ , and substituting $( 1 - \delta ) b = ( q + \delta ) \Lambda ( q , \delta )$ give

$$
\begin{array} { r l } & { \quad f _ { t } ( { \mathbf x } _ { t } ) + \displaystyle \frac { 1 } { 2 } \| { \mathbf x } _ { t } - { \mathbf x } _ { t - 1 } \| ^ { 2 } - C ( f _ { t } ( { \mathbf u } _ { t } ) + \displaystyle \frac { 1 } { 2 } \| { \mathbf u } _ { t } - { \mathbf u } _ { t - 1 } \| ^ { 2 } ) + \Phi _ { t } - \Phi _ { t - 1 } } \\ & { \le \displaystyle - \frac { 1 } { 2 q } \| a ( { \mathbf x } _ { t } - { \mathbf v } _ { t } ) + q \delta ( { \mathbf x } _ { t } - { \mathbf x } _ { t - 1 } ) \| ^ { 2 } - q f _ { t } ( { \mathbf v } _ { t } ) - \displaystyle \frac { q + \delta } { 2 } ( \lambda - \Lambda ( q , \delta ) ) \| { \mathbf u } _ { t } - { \mathbf v } _ { t } \| ^ { 2 } \le 0 . } \end{array}
$$

Here the last inequality uses Assumption $2 f _ { t } ( \cdot ) \geq 0$ and the pre-request $\lambda \ge \Lambda ( q , \delta )$ , which proves (6).

Summing (6) over $t = 1 , \dots , T$ , and using $\Phi _ { 0 } = 0$ and $\Phi _ { T } \geq 0$ , we obtain

$$
\begin{array} { l } { \displaystyle \sum _ { t = 1 } ^ { T } \left( f _ { t } ( \mathbf { x } _ { t } ) + \frac { 1 } { 2 } \| \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } \| ^ { 2 } \right) \leq \displaystyle \sum _ { t = 1 } ^ { T } \left( f _ { t } ( \mathbf { x } _ { t } ) + \frac { 1 } { 2 } \| \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } \| ^ { 2 } \right) + \Phi _ { T } - 0 } \\ { \leq C \displaystyle \sum _ { t = 1 } ^ { T } \left( f _ { t } ( \mathbf { u } _ { t } ) + \frac { 1 } { 2 } \| \mathbf { u } _ { t } - \mathbf { u } _ { t - 1 } \| ^ { 2 } \right) . } \end{array}
$$

This concludes the proof of the C-competitive guarantee.

The best ratio provided by Theorem 7 is the smallest $1 + q$ for which some δ satisfies $\Lambda ( q , \delta ) \leq \lambda$ . The following lemma characterizes this.

Lemma 8 For every $q > 0$ , the function $\Lambda ( q , \cdot )$ has a unique minimizer $\delta ^ { * } ( q )$ on $[ 0 , 1 )$ , and $\delta ^ { * } ( q )$ is the unique root in $( 0 , 1 / \sqrt { 3 } )$ of

$$
P _ { q } ( \delta ) = 2 \delta ( 2 - \delta ^ { 2 } ) q ^ { 2 } + \delta ( 2 + 3 \delta - \delta ^ { 3 } ) q + 3 \delta ^ { 2 } - 1 .
$$

Moreover, $P _ { q } < 0$ on $[ 0 , \delta ^ { * } ( q ) )$ and $P _ { q } > 0$ on $( \delta ^ { * } ( q ) , 1 )$ . The function ϕ defined in (2) equals $\Lambda ( q , \delta ^ { * } ( q ) )$ , is continuous and strictly decreasing on $( 0 , \infty )$ , and satisfies

$$
\frac { 1 } { q ( q + 1 ) } \leq \phi ( q ) \leq \frac { 1 } { q ^ { 2 } } .
$$

By the last inequalities, the inverse $\phi ^ { - 1 } ( \lambda )$ exists for every $\lambda > 0$ and is the unique q satisfying $\begin{array} { r } { \operatorname* { m i n } _ { 0 \leq d < 1 } \Lambda ( q , d ) = \lambda } \end{array}$

## 5.2 Lower bound

This section proves the lower bound of Theorem 2. The construction realizes the equality conditions of Theorem 7. Every online algorithm pays at least the cost of a reference path in expectation, and an ofline comparator pays a fraction $1 / ( 1 + q ^ { * } ( \lambda ) )$ ) of this cost per round. Throughout the section, $\begin{array} { r } { m ( \mathbf { x } , \mathbf { y } ) = \frac { 1 } { 2 } \| \mathbf { x } - \mathbf { y } \| ^ { 2 } , \mathcal { X } = \mathbb { R } ^ { d } } \end{array}$ , and $\mathbf { x } _ { 0 } = \mathbf { 0 }$

Lemma 9 (Lower bound for squared Euclidean movement) Fix $\lambda > 0$ and an infinitedimensional real Hilbert space $\mathcal { H } _ { \infty }$ . Let $\mathcal { D } = \{ \mathbb { R } ^ { d } : d \in \mathbb { N } , d \geq 1 \} \cup \{ \mathcal { H } _ { \infty } \}$ and let be the class of all randomized online algorithms defined on the spaces in . For each $\mathcal { H } \in \mathcal { D } _ { : }$ , let $\mathcal { F } _ { \lambda } ( \mathcal { H } )$ be the class of nonnegative convex functions on  satisfying the λ-quadratic growth condition. Then the dimension-uniform minimax competitive ratio satisfies

$$
\operatorname* { i n f } _ { A \in \mathcal { A } } \operatorname* { s u p } _ { T \in \mathbb { N } , \ \mathcal { H } \in \mathcal { D } \atop ( f _ { 1 } , \ldots , f _ { T } ) \in \mathcal { F } _ { \lambda } ( \mathcal { H } ) ^ { T } } \frac { \mathbb { E } _ { \xi } [ \mathrm { A L G } _ { T } ] } { \mathrm { O P T } _ { T } } \geq 1 + q _ { * } ( \lambda ) .
$$

Here the expectation is taken over the algorithm’s internal randomness $\xi .$

Combining Theorem 7 and Theorem 9 proves Theorem 2. The proof is deferred to Section A.7. In other words, for every $\lambda > 0$ and every $R < 1 + q _ { * } ( \lambda )$ , any randomized online algorithm defined across all finite-dimensional Euclidean spaces and Hilbert space, a finite horizon $T ,$ and a sequence of cost functions, we have $\mathbb { E } _ { \xi } [ \mathrm { A L G } _ { T } ] > R \mathrm { O P T } _ { T }$ , where $\xi$ denotes the algorithm’s internal randomness. Consequently, no such algorithm can achieve a competitive ratio strictly smaller than $1 + q _ { * } ( \lambda )$ uniformly over all finite dimensions.

Here, other algorithms may depend on the dimension and the announced horizon. However, we use dimension-uniformity to refers to a competitive guarantee whose ratio is independent of the dimension $d ,$ and the corresponding algorithm works for any finite dimension d and any horizon $T _ { i }$ , but and its parameters are independent of the dimension $d .$ The same lower bound also holds on any fixed infinite-dimensional real Hilbert space. Theorem 9 establishes the lower bound for the dimension-uniform minimax competitive ratio of any randomized online algorithm on the class of convex functions with λ-quadratic growth and squared Euclidean movement cost.

## 6 Conclusion

In this paper, we investigate convex function chasing with quadratic growth under two movement costs. For Euclidean movement, we propose Clipped Proximal Descent (CPD), which achieves an $O ( \kappa ^ { 1 / 3 } )$ competitive ratio for smooth convex functions satisfying quadratic growth. This matches the existing lower bound (Argue et al., 2020a) and resolves the gap for κ-well-conditioned functions under a weaker assumption than strong convexity. For squared Euclidean movement, we establish the exact optimal dimension-free competitive ratio for convex functions with quadratic growth through a tight analysis of R-OBD and a matching lower bound against randomized algorithms.

## Disclosure of generative AI use

We use GPT-6 Astra to assist with exploring proof strategies and checking mathematical derivations. The authors also use GPT-6 Astra for identifying relevant literature and polishing the paper. The tool materially afected Section 4 and Section 5. The authors verified the correctness and originality of all content, including references.

## References

Antonios Antoniadis and Kevin Schewior. A tight lower bound for online convex optimization with switching costs. In Approximation and Online Algorithms (WAOA 2017), volume 10787, pages 164–175, 2018.

C. J. Argue, S´ebastien Bubeck, Michael B. Cohen, Anupam Gupta, and Yin Tat Lee. A nearly-linear bound for chasing nested convex bodies. In Proceedings of the 30th Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pages 117–122, 2019.

C. J. Argue, Anupam Gupta, and Guru Guruganesh. Dimension-free bounds for chasing convex functions. In Proceedings of the 33rd Conference on Learning Theory (COLT), volume 125 of Proceedings of Machine Learning Research, pages 219–241, 2020a.

C. J. Argue, Anupam Gupta, Guru Guruganesh, and Ziye Tang. Chasing convex bodies with linear competitive ratio. In Proceedings of the 31st Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pages 1519–1524, 2020b.

Nikhil Bansal, Anupam Gupta, Ravishankar Krishnaswamy, Kirk Pruhs, Kevin Schewior, and Cliford Stein. A 2-competitive algorithm for online convex optimization with switching costs. In Approximation, Randomization, and Combinatorial Optimization. Algorithms and Techniques (APPROX/RANDOM), 2015.

Nikhil Bansal, Martin B¨ohm, Marek Eli´aˇs, Grigorios Koumoutsos, and Seeun William Umboh. Nested convex bodies are chaseable. In Proceedings of the 29th Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), 2018.

S´ebastien Bubeck. Convex optimization: Algorithms and complexity. Foundations and Trends in Machine Learning, 8(3–4):231–357, 2015.

S´ebastien Bubeck, Yin Tat Lee, Yuanzhi Li, and Mark Sellke. Competitively chasing convex bodies. In Proceedings of the 51st Annual ACM SIGACT Symposium on Theory of Computing (STOC), pages 861–868, 2019.

Niangjun Chen, Gautam Goel, and Adam Wierman. Smoothed online convex optimization in high dimensions via online balanced descent. In Proceedings of the 31st Conference on Learning Theory (COLT), volume 75 of Proceedings of Machine Learning Research, pages 1574–1594, 2018.

Dmitriy Drusvyatskiy and Adrian S. Lewis. Error bounds, quadratic growth, and linear convergence of proximal methods. Mathematics of Operations Research, 43(3):919–948, 2018.

Joel Friedman and Nathan Linial. On convex body chasing. Discrete & Computational Geometry, 9(3):293–321, 1993.

Gautam Goel and Adam Wierman. An online algorithm for smoothed regression and LQR control. In Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics (AISTATS), volume 89 of Proceedings of Machine Learning Research, pages 2504–2513, 2019.

Gautam Goel, Yiheng Lin, Haoyuan Sun, and Adam Wierman. Beyond online balanced descent: An optimal algorithm for smoothed online optimization. In Advances in Neural Information Processing Systems 32 (NeurIPS), 2019.

Yiheng Lin, Gautam Goel, and Adam Wierman. Online optimization with predictions and non-convex losses. Proceedings of the ACM on Measurement and Analysis of Computing Systems, 4(1):18:1–18:32, 2020.

Ion Necoara, Yurii Nesterov, and Fran¸cois Glineur. Linear convergence of first order methods for non-strongly convex optimization. Mathematical Programming, 175:69–107, 2019.

Mark Sellke. Chasing convex bodies optimally. In Proceedings of the 31st Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pages 1509–1518, 2020.

Lijun Zhang, Wei Jiang, Shiyin Lu, and Tianbao Yang. Revisiting smoothed online learning. In Advances in Neural Information Processing Systems 34 (NeurIPS), 2021.

## Appendix A. Proofs of supporting results

## A.1 Proof of Theorem 5

Proof Fix a round t. Assumption 3 gives $g _ { t } ( \mathbf { z } ) \geq \lambda _ { t } \Vert \mathbf { z } - \mathbf { v } _ { t } \Vert ^ { 2 } / 2  \infty { \mathrm { ~ a s ~ } } \Vert \mathbf { z } \Vert  \infty$ Thus $g _ { t }$ is coercive. By Assumption 2, it is also continuous and convex. Together with Assumption 1, these properties imply that its minimizers over  form a nonempty compact convex set. The projection defining $\mathbf { v } _ { t } ^ { \mathcal { X } }$ is therefore unique. For $\mu > 0$ , recall that

$$
\mathbf { y } _ { t } ( \mu ) = \underset { \mathbf { z } \in \mathcal { X } } { \arg \operatorname* { m i n } } \left. g _ { t } ( \mathbf { z } ) + \frac { \Vert \mathbf { z } - \mathbf { x } _ { t - 1 } \Vert ^ { 2 } } { 2 \mu } \right. .\tag{8}
$$

The objective in (8) is coercive and $( 1 / \mu )$ -strongly convex, so $\mathbf { y } _ { t } ( \mu )$ is also unique. Comparison with $\mathbf { v } _ { t } ^ { \mathcal { X } }$ gives

$$
g _ { t } ( \mathbf { y } _ { t } ( \mu ) ) + \frac { \| \mathbf { y } _ { t } ( \mu ) - \mathbf { x } _ { t - 1 } \| ^ { 2 } } { 2 \mu } \leq g _ { t } ( \mathbf { v } _ { t } ^ { \mathcal { X } } ) + \frac { \| \mathbf { v } _ { t } ^ { \mathcal { X } } - \mathbf { x } _ { t - 1 } \| ^ { 2 } } { 2 \mu } .
$$

Using $g _ { t } ( \mathbf { y } _ { t } ( \mu ) ) \geq g _ { t } ( \mathbf { v } _ { t } ^ { \mathcal { X } } )$ and dropping nonnegative terms yields

$$
\begin{array} { r } { \| \mathbf { y } _ { t } ( \mu ) - \mathbf { x } _ { t - 1 } \| \leq \| \mathbf { v } _ { t } ^ { \mathcal { X } } - \mathbf { x } _ { t - 1 } \| , } \\ { 0 \leq g _ { t } ( \mathbf { y } _ { t } ( \mu ) ) - g _ { t } ( \mathbf { v } _ { t } ^ { \mathcal { X } } ) \leq \frac { \| \mathbf { v } _ { t } ^ { \mathcal { X } } - \mathbf { x } _ { t - 1 } \| ^ { 2 } } { 2 \mu } . } \end{array}\tag{9}
$$

To show continuity of the proximal path, take $\mu , \nu > 0$ . The two optimality conditions are

$$
\begin{array} { r l } & { \bigg \langle \nabla g _ { t } ( \mathbf y _ { t } ( \boldsymbol \mu ) ) + \frac { \mathbf y _ { t } ( \boldsymbol \mu ) - \mathbf x _ { t - 1 } } { \boldsymbol \mu } , \mathbf y _ { t } ( \boldsymbol \nu ) - \mathbf y _ { t } ( \boldsymbol \mu ) \bigg \rangle \geq 0 , } \\ & { \bigg \langle \nabla g _ { t } ( \mathbf y _ { t } ( \boldsymbol \nu ) ) + \frac { \mathbf y _ { t } ( \boldsymbol \nu ) - \mathbf x _ { t - 1 } } { \boldsymbol \nu } , \mathbf y _ { t } ( \boldsymbol \mu ) - \mathbf y _ { t } ( \boldsymbol \nu ) \bigg \rangle \geq 0 . } \end{array}
$$

Assumption 2 gives the two inequalities

$$
\begin{array} { r l } & { g _ { t } ( \mathbf { y } _ { t } ( \boldsymbol { \mu } ) ) - g _ { t } ( \mathbf { y } _ { t } ( \boldsymbol { \nu } ) ) \geq \langle \nabla g _ { t } ( \mathbf { y } _ { t } ( \boldsymbol { \nu } ) ) , \mathbf { y } _ { t } ( \boldsymbol { \mu } ) - \mathbf { y } _ { t } ( \boldsymbol { \nu } ) \rangle , } \\ & { g _ { t } ( \mathbf { y } _ { t } ( \boldsymbol { \nu } ) ) - g _ { t } ( \mathbf { y } _ { t } ( \boldsymbol { \mu } ) ) \geq \langle \nabla g _ { t } ( \mathbf { y } _ { t } ( \boldsymbol { \mu } ) ) , \mathbf { y } _ { t } ( \boldsymbol { \nu } ) - \mathbf { y } _ { t } ( \boldsymbol { \mu } ) \rangle . } \end{array}
$$

Adding them yields

$$
\langle \nabla g _ { t } ( { \mathbf { y } } _ { t } ( \mu ) ) - \nabla g _ { t } ( { \mathbf { y } } _ { t } ( \nu ) ) , { \mathbf { y } } _ { t } ( \mu ) - { \mathbf { y } } _ { t } ( \nu ) \rangle \geq 0 .
$$

Adding the optimality conditions and using this inequality, we obtain

$$
\begin{array} { r l r } & { } & { \displaystyle \frac { \| { \bf y } _ { t } ( \mu ) - { \bf y } _ { t } ( \nu ) \| ^ { 2 } } { \mu } \leq \left( \frac { 1 } { \nu } - \frac { 1 } { \mu } \right) \langle { \bf y } _ { t } ( \nu ) - { \bf x } _ { t - 1 } , { \bf y } _ { t } ( \mu ) - { \bf y } _ { t } ( \nu ) \rangle } \\ & { } & { \displaystyle \leq \frac { | \mu - \nu | } { \mu \nu } \| { \bf v } _ { t } ^ { \mathcal { X } } - { \bf x } _ { t - 1 } \| \| { \bf y } _ { t } ( \mu ) - { \bf y } _ { t } ( \nu ) \| , } \end{array}
$$

where the last step uses Cauchy–Schwarz and (9). Dividing when the distance is nonzero (the zero case is immediate) gives

$$
\| \mathbf { y } _ { t } ( \mu ) - \mathbf { y } _ { t } ( \nu ) \| \leq \frac { | \mu - \nu | } { \nu } \| \mathbf { v } _ { t } ^ { \mathcal { X } } - \mathbf { x } _ { t - 1 } \| \longrightarrow 0 \quad \mathrm { a s ~ } \mu \to \nu .
$$

Comparing the proximal objective with its value at $\mathbf { x } _ { t - 1 }$ gives

$$
\begin{array}{c} g _ { t } ( \mathbf { y } _ { t } ( \mu ) ) + { \frac { \| \mathbf { y } _ { t } ( \mu ) - \mathbf { x } _ { t - 1 } \| ^ { 2 } } { 2 \mu } } \leq g _ { t } ( \mathbf { x } _ { t - 1 } ) ,  \\ { \| \mathbf { y } _ { t } ( \mu ) - \mathbf { x } _ { t - 1 } \| ^ { 2 } \leq 2 \mu g _ { t } ( \mathbf { x } _ { t - 1 } ) \longrightarrow 0 \quad ( \mu \downarrow 0 ) . } \end{array}
$$

As $\mu \to \infty , ( 9 )$ makes the proximal points bounded. Any convergent subsequence ${ \bf y } _ { t } ( \mu _ { j } ) \to$ $\bar { \mathbf { y } } _ { t }$ with $\mu _ { j } \to \infty$ satisfies

$$
\bar { \mathbf { y } } _ { t } \in \mathcal { X } , \qquad g _ { t } ( \bar { \mathbf { y } } _ { t } ) = g _ { t } ( \mathbf { v } _ { t } ^ { \mathcal { X } } ) , \qquad \| \bar { \mathbf { y } } _ { t } - \mathbf { x } _ { t - 1 } \| \leq \| \mathbf { v } _ { t } ^ { \mathcal { X } } - \mathbf { x } _ { t - 1 } \| .
$$

The closest constrained minimizer is unique, so every such limit is $\mathbf { v } _ { t } ^ { \mathcal { X } }$ . We have therefore shown that $\mathbf { y } _ { t } ( \mu ) \to \mathbf { x } _ { t - 1 }$ as $\mu \downarrow 0$ and $\mathbf { y } _ { t } ( \mu ) \to \mathbf { v } _ { t } ^ { \mathcal { X } }$ as $\mu \to \infty$

In the balanced case, $g _ { t } ( \mathbf { v } _ { t } ^ { \mathcal { X } } ) < \beta \| \mathbf { v } _ { t } ^ { \mathcal { X } } - \mathbf { x } _ { t - 1 } \|$ . Thus $\mathbf { x } _ { t - 1 }$ is not a constrained minimizer: otherwise $\mathbf { v } _ { t } ^ { \mathcal { X } } = \mathbf { x } _ { t - 1 }$ would contradict this inequality. In particular, $g _ { t } ( \mathbf { x } _ { t - 1 } ) > g _ { t } ( \mathbf { v } _ { t } ^ { \mathcal { X } } ) \geq 0$ The limits just proved imply

$$
\begin{array} { r l } & { \underset { \mu \downarrow 0 } { \operatorname* { l i m } } \big ( g _ { t } ( \mathbf { y } _ { t } ( \mu ) ) - \beta \| \mathbf { y } _ { t } ( \mu ) - \mathbf { x } _ { t - 1 } \| \big ) = g _ { t } ( \mathbf { x } _ { t - 1 } ) > 0 , } \\ & { \underset { \mu  \infty } { \operatorname* { l i m } } \big ( g _ { t } ( \mathbf { y } _ { t } ( \mu ) ) - \beta \| \mathbf { y } _ { t } ( \mu ) - \mathbf { x } _ { t - 1 } \| \big ) = g _ { t } ( \mathbf { v } _ { t } ^ { \mathcal { X } } ) - \beta \| \mathbf { v } _ { t } ^ { \mathcal { X } } - \mathbf { x } _ { t - 1 } \| < 0 . } \end{array}
$$

Continuity and the intermediate value theorem give a positive root. The expression is positive for all suficiently small $\mu > 0$ , so the root set is bounded away from zero. Its infimum is a root by continuity, proving the existence of the smallest root $\mu _ { t }$

At this root, ${ \bf y } _ { t } = { \bf x } _ { t - 1 }$ would imply $g _ { t } ( \mathbf x _ { t - 1 } ) = 0$ , contradicting the strict positivity above. Therefore $s _ { t } > 0$ and $g _ { t } ( \mathbf { y } _ { t } ) = \beta s _ { t } > 0 = g _ { t } ( \mathbf { v } _ { t } )$ , so $\left\| \mathbf { y } _ { t } - \mathbf { v } _ { t } \right\| > 0$ . The definitions give $0 < \gamma _ { t } = r ^ { 2 } s _ { t } / ( s _ { t } + \beta \mu _ { t } ) < r ^ { 2 } \le 1 / 4$ and $0 < \eta _ { t } \le \gamma _ { t }$ . Finally, Assumptions 1 and 2 imply that the projection set is closed and convex. It contains $\mathbf { y } _ { t }$ , so the final projection exists and is unique.

## A.2 Proof of Theorem 6

Proof Fix t and write $g = g _ { t } , p = x _ { t - 1 } , x = x _ { t } , u = u _ { t } , v = v _ { t } , L = L _ { t } $ , and $\lambda = \lambda _ { t }$ Suppose that the condition in line 4 is false, and retain the notation of the theorem proof. For every $u \in \mathcal X$ , by the proximal optimality condition and convexity of $^ { g , }$ we have

$$
\langle \nabla g ( y ) + { \frac { 1 } { \mu } } ( y - p ) , u - y \rangle \geq 0 , \qquad g ( u ) - g ( y ) \geq \langle \nabla g ( y ) , u - y \rangle \geq { \frac { 1 } { \mu } } \langle p - y , u - y \rangle .\tag{10}
$$

Therefore,

$$
\begin{array} { l } { \displaystyle \| p - u \| ^ { 2 } = \| y - u \| ^ { 2 } + \| y - p \| ^ { 2 } - 2 \langle p - y , u - y \rangle } \\ { \geq \| y - u \| ^ { 2 } + \| y - p \| ^ { 2 } + 2 \mu \left( g ( y ) - g ( u ) \right) . } \end{array}\tag{11}
$$

When $g ( u ) \leq g ( y )$ , the set $S = \mathcal { X } \cap \{ z : g ( z ) \leq g ( y ) \}$ contains y and u. We have the following lemma:

Lemma 10 Let $p , y , u , v \in \mathbb { R } ^ { d } , p \neq y , y \neq v , \mu , \lambda > 0 , \kappa \geq 1$ , and $0 < \gamma < 1$ . Suppose that

$$
0 \leq g ( u ) \leq g ( y ) , \qquad g ( u ) \geq { \frac { \lambda } { 2 } } \| u - v \| ^ { 2 } ,
$$

$$
\| p - u \| ^ { 2 } \geq \| y - u \| ^ { 2 } + \| y - p \| ^ { 2 } + 2 \mu \left( g ( y ) - g ( u ) \right) .
$$

Let S be a closed convex set containing y and $u ,$ and set

$$
\eta = \operatorname* { m i n } \left\{ \gamma , \frac { \left\| y - p \right\| } { 2 \kappa ^ { 1 / 3 } \left\| y - v \right\| } \right\} , \qquad x = \Pi _ { S } \left\{ ( 1 - \eta ) y + \eta v \right\} .
$$

Then

$$
\begin{array} { r l } & { \quad \| p - u \| - \| x - u \| + g ( u ) } \\ & { \geq \operatorname* { m i n } \left\{ \frac { \| y - p \| } { 2 \kappa ^ { 1 / 3 } } , \sqrt { \gamma ( 2 - \gamma ) \left( \| y - p \| ^ { 2 } + 2 \mu ( g ( y ) - g ( u ) ) \right) } \right\} + g ( u ) - \gamma \sqrt { \frac { 2 g ( u ) } { \lambda } } . } \end{array}\tag{12}
$$

And we have

$$
\begin{array} { r l } & { \quad \| p - u \| - \| x - u \| + g ( u ) } \\ & { \geq \operatorname* { m i n } \left\{ \frac { \| y - p \| } { 2 \kappa ^ { 1 / 3 } } , \sqrt { \gamma ( 2 - \gamma ) \left( \| y - p \| ^ { 2 } + 2 \mu ( g ( y ) - g ( u ) ) \right) } \right\} + g ( u ) - \gamma \sqrt { \frac { 2 g ( u ) } { \lambda } } . } \end{array}\tag{13}
$$

$$
\begin{array} { r l } & { \mathrm { ~ f ~ f ~ } g ( u ) \leq g ( y ) / 2 , \mathrm { t h e n ~ } 0 < \gamma < 1 \mathrm { ~ a n d ~ } \mu = \| y - p \| / G \operatorname { i m p l y } } \\ & { \qquad \gamma ( 2 - \gamma ) \left( \| y - p \| ^ { 2 } + 2 \mu ( g ( y ) - g ( u ) ) \right) } \\ & { \geq \gamma \left( \| y - p \| ^ { 2 } + \mu g ( y ) \right) } \\ & { = \frac { G } { 4 \kappa ^ { 2 / 3 } \left( G + 1 + \frac { 3 } { 2 \kappa ^ { 1 / 3 } } \right) } \left( 1 + \frac { 1 + \frac { 3 } { 2 \kappa ^ { 1 / 3 } } } { G } \right) \| y - p \| ^ { 2 } } \\ & { \qquad = \frac { \| y - p \| ^ { 2 } } { 4 \kappa ^ { 2 / 3 } } . } \end{array}
$$

Substituting this into (13) gives

$$
\begin{array} { r l } & { \quad \| p - u \| - \| x - u \| + g ( u ) } \\ & { \ge \frac { \| y - p \| } { 2 \kappa ^ { 1 / 3 } } + g ( u ) - \gamma \sqrt { \frac { 2 g ( u ) } { \lambda } } } \\ & { = \frac { \| y - p \| } { 2 \kappa ^ { 1 / 3 } } + \Big ( \sqrt { g ( u ) } - \frac { \gamma } { \sqrt { 2 \lambda } } \Big ) ^ { 2 } - \frac { \gamma ^ { 2 } } { 2 \lambda } } \\ & { \ge \frac { \| y - p \| } { 2 \kappa ^ { 1 / 3 } } - \frac { \| y - p \| } { 1 6 \kappa ^ { 1 / 3 } } } \\ & { = \frac { 7 } { 1 6 \kappa ^ { 1 / 3 } } \| y - p \| . } \end{array}
$$

If $g ( y ) / 2 \le g ( u ) \le g ( y )$ , Taking $u = p$ in the optimality condition (10) and using the Cauchy–Schwarz inequality yields

$$
\begin{array} { r l } {  { G \| y - p \| = \frac { \| y - p \| ^ { 2 } } { \mu } } } \\ & { \leq \langle \nabla g ( y ) , p - y \rangle } \\ & { \leq \| \nabla g ( y ) \| \| y - p \| . } \end{array}
$$

Since $y \ne p ,$ , we obtain $G \leq \| \nabla g ( y ) \|$ . Moreover, global nonnegativity and smoothness imply the one-step improvement bound of the smoothed function:

$$
0 \leq g \left( y - \frac { \nabla g ( y ) } { L } \right) \leq g ( y ) - \frac { \| \nabla g ( y ) \| ^ { 2 } } { 2 L } .
$$

Consequently,

$$
\begin{array} { r l } & { \frac { \hat { \eta } ^ { 2 } } { \lambda } = \frac { G ^ { 2 } } { 1 6 \lambda \sqrt { 4 \pi } ^ { 4 / 3 } } \left( G + 1 + \frac { 3 } { 2 \pi ^ { 4 / 5 } } \right) ^ { 2 } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ &  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \end{array}
$$

Here we used $G ^ { 2 } \leq \| \nabla g ( y ) \| ^ { 2 } \leq 2 L g ( y )$ $L / \lambda \leq \kappa$ , and $g ( y ) = ( 1 + 3 / ( 2 \kappa ^ { 1 / 3 } ) ) \| y - p \|$ <sup>Taking</sup> <sup>this</sup> <sup>into</sup> <sup>(13),</sup> <sup>and</sup> <sup>eliminating</sup> <sup>the</sup> <sup>min</sup>{<sup>.</sup> <sup>.</sup> <sup>.</sup> } <sup>term</sup> <sup>give</sup>

$$
\begin{array} { r l } & { \quad \| \boldsymbol { p } - \boldsymbol { u } \| - \| \boldsymbol { x } - \boldsymbol { u } \| + g ( \boldsymbol { w } ) } \\ & { \le g ( \boldsymbol { u } ) - \gamma \sqrt { \frac { g _ { 0 } ( \boldsymbol { u } ) } { \lambda } } } \\ & { = \frac { g ( \boldsymbol { u } ) } { 2 } + \bigg ( \sqrt { \frac { g ( \boldsymbol { u } ) } { 2 } } - \frac { \gamma } { \sqrt { \lambda } } \bigg ) ^ { 2 } - \frac { \gamma ^ { 2 } } { \lambda } } \\ & { \le \frac { g ( \boldsymbol { v } ) } { 4 } - \frac { \| g - \boldsymbol { v } \| } { 8 \kappa ^ { 1 / 3 } } } \\ & { = \bigg ( \frac { 1 } { 4 } + \frac { 1 } { 4 \kappa ^ { 1 / 3 } } \bigg ) \| \boldsymbol { y } - \boldsymbol { p } \| } \\ & { \ge \frac { \| \boldsymbol { y } - \boldsymbol { p } \| } { 2 \kappa ^ { 1 / 3 } } } \\ & { \le \frac { \| g \| } { \| g \kappa ^ { 1 / 3 } } } \\ & { \le \frac { \| g \| } { \| g \kappa ^ { 1 / 3 } } \| \boldsymbol { v } - \boldsymbol { p } \| . } \end{array}
$$

The penultimate inequality uses $\kappa \geq 1$

If $g ( u ) \geq g ( y )$ , then the triangle inequality and $\lVert x - y \rVert \le \eta \lVert y - v \rVert \le \lVert y - p \rVert / ( 2 \kappa ^ { 1 / 3 } )$ give

$$
\begin{array} { l } { \displaystyle | | p - u | | - | | x - u | | + g ( u ) \ge g ( u ) - | | x - p | | } \\ { \ge g ( y ) - | | y - p | | - | | x - y | | } \\ { \ge \left( 1 + \frac { 3 } { 2 \kappa ^ { 1 / 3 } } - 1 - \frac { 1 } { 2 \kappa ^ { 1 / 3 } } \right) \| y - p | } \\ { = \frac { \| y - p \| } { \kappa ^ { 1 / 3 } } } \\ { \ge \frac { 7 } { 1 6 \kappa ^ { 1 / 3 } } \| y - p \| . } \end{array}
$$

The desired bound therefore holds in all three cases.

## A.3 Proof of Theorem 10

Proof Since $u \in S$ , nonexpansiveness of projection and the triangle inequality give

$$
\begin{array} { r l } & { \| x - u \| \leq \| ( 1 - \eta ) y + \eta v - u \| } \\ & { \qquad = \| ( 1 - \eta ) ( y - u ) + \eta ( v - u ) \| } \\ & { \qquad \leq ( 1 - \eta ) \| y - u \| + \eta \| u - v \| . } \end{array}
$$

Combining this with the assumed lower bound (11) yields

$$
\begin{array} { r l } & { \quad \left\| p - u \right\| - \left\| x - u \right\| } \\ & { \geq \sqrt { \| y - u \| ^ { 2 } + \| y - p \| ^ { 2 } + 2 \mu ( g ( y ) - g ( u ) ) } } \\ & { \qquad - \left( 1 - \eta \right) \| y - u \| - \eta \| u - v \| } \\ & { \geq \operatorname* { m i n } \left\{ \frac { \| y - p \| } { 2 \kappa ^ { 1 / 3 } } , \sqrt { \gamma ( 2 - \gamma ) \left( \| y - p \| ^ { 2 } + 2 \mu ( g ( y ) - g ( u ) ) \right) } \right\} - \gamma \| u - v \| } \\ & { \geq \operatorname* { m i n } \left\{ \frac { \| y - p \| } { 2 \kappa ^ { 1 / 3 } } , \sqrt { \gamma ( 2 - \gamma ) \left( \| y - p \| ^ { 2 } + 2 \mu ( g ( y ) - g ( u ) ) \right) } \right\} - \gamma \sqrt { \frac { 2 g ( u ) } { \lambda } } . } \end{array}
$$

The second inequality follows from the following lemma:

Lemma 11 Let $y , u , v \in \mathbb { R } ^ { d } , A , R > 0$ , and $0 < \gamma < 1$ . Set

$$
\eta = \operatorname* { m i n } \{ \gamma , R / \lVert y - v \rVert \} , \qquad w i t h \ \eta = \gamma \ i f \ y = v .
$$

In particular,

$$
\| u - v \| \geq | \| y - v \| - \| y - u \| | , \qquad 0 < \eta \leq \gamma , \qquad \eta \| y - v \| = R \quad i f \eta < \gamma .
$$

Then

$$
\begin{array} { r l } & { \sqrt { \| y - u \| ^ { 2 } + A ^ { 2 } } - ( 1 - \eta ) \| y - u \| + ( \gamma - \eta ) \| u - v \| } \\ & { \geq \operatorname* { m i n } \{ R , A \sqrt { \gamma ( 2 - \gamma ) } \} . } \end{array}\tag{14}
$$

And

$$
A ^ { 2 } = \| y - p \| ^ { 2 } + 2 \mu \left( g ( y ) - g ( u ) \right) > 0 , \qquad R = \frac { \| y - p \| } { 2 \kappa ^ { 1 / 3 } } .
$$

The last inequality uses $g ( u ) \geq \lambda \Vert u - v \Vert ^ { 2 } / 2$ . Adding $g ( u )$ proves the claim.

## A.4 Proof of Theorem 11

Proof For $0 \leq a \leq 1$ , the Cauchy–Schwarz inequality gives

$$
\begin{array} { c } { { a \| y - u \| + A \sqrt { 1 - a ^ { 2 } } \leq \sqrt { \| y - u \| ^ { 2 } + A ^ { 2 } } \sqrt { a ^ { 2 } + 1 - a ^ { 2 } } } } \\ { { = \sqrt { \| y - u \| ^ { 2 } + A ^ { 2 } } . } } \end{array}
$$

Hence

$$
\sqrt { \| y - u \| ^ { 2 } + A ^ { 2 } } - a \| y - u \| \geq A \sqrt { 1 - a ^ { 2 } } .\tag{15}
$$

$\mathrm { I f } \ \eta = \gamma$ , taking $a = 1 - \gamma$ proves (14). It remains to consider $\eta < \gamma$ , for which $\eta \lVert y - v \rVert = R$

First suppose that $R \leq A { \sqrt { \gamma ( 2 - \gamma ) } } . { \mathrm { ~ I f ~ } } \| y - u \| \geq \| y - v \|$ , then

$$
\begin{array} { r l } & { \quad \sqrt { \| y - u \| ^ { 2 } + A ^ { 2 } } - ( 1 - \eta ) \| y - u \| + ( \gamma - \eta ) \| u - v \| } \\ & { \geq \sqrt { \| y - u \| ^ { 2 } + A ^ { 2 } } - ( 1 - \eta ) \| y - u \| + ( \gamma - \eta ) \left( \| y - u \| - \| y - v \| \right) } \\ & { = \sqrt { \| y - u \| ^ { 2 } + A ^ { 2 } } - \| y - u \| + \gamma \left( \| y - u \| - \| y - v \| \right) + R } \\ & { \geq R . } \end{array}
$$

Here we used the reverse triangle inequality and $\eta \lVert y - v \rVert = R . \mathrm { ~ H ~ } \lVert y - u \rVert \leq \lVert y - v \rVert$ and $\eta \leq \gamma / 2$ , then

$$
\begin{array} { r l } & { \quad \sqrt { \| y - u \| ^ { 2 } + A ^ { 2 } } - ( 1 - \eta ) \| y - u \| + ( \gamma - \eta ) \| u - v \| } \\ & { \geq \sqrt { \| y - u \| ^ { 2 } + A ^ { 2 } } - ( 1 - \eta ) \| y - u \| + ( \gamma - \eta ) \left( \| y - v \| - \| y - u \| \right) } \\ & { = \sqrt { \| y - u \| ^ { 2 } + A ^ { 2 } } - \| y - u \| + \left( \gamma - 2 \eta \right) ( \| y - v \| - \| y - u \| ) + R } \\ & { \geq R . } \end{array}
$$

The first inequality again uses the reverse triangle inequality, and the last uses $\gamma - 2 \eta \ge 0$ If $\| y - u \| \leq \| y - v \|$ and $\gamma / 2 < \eta < \gamma$ , then

$$
\begin{array} { r l } & { \eta ^ { 2 } ( 2 - 2 \eta + \gamma ) - \gamma ( 2 - \gamma ) ( 2 \eta - \gamma ) } \\ & { = ( \gamma - \eta ) \left( 2 \eta ^ { 2 } + ( 2 - \gamma ) ( \gamma - \eta ) \right) } \\ & { \geq 0 . } \end{array}
$$

Since $0 < 2 \eta - \gamma < 1$ and $R \leq A \sqrt { \gamma ( 2 - \gamma ) }$ , this implies

$$
\begin{array} { l } { { \displaystyle { \cal A } \sqrt { ( 2 \eta - \gamma ) ( 2 - 2 \eta + \gamma ) } \geq \frac { { \cal A } \sqrt { \gamma ( 2 - \gamma ) } ( 2 \eta - \gamma ) } { \eta } } } \\ { { \displaystyle \quad \geq \frac { R ( 2 \eta - \gamma ) } { \eta } . } } \end{array}
$$

Therefore,

$$
\begin{array} { r l } & { \quad \sqrt { \| y - u \| ^ { 2 } + A ^ { 2 } } - ( 1 - \eta ) \| y - u \| + ( \gamma - \eta ) \| u - v \| } \\ & { \geq \sqrt { \| y - u \| ^ { 2 } + A ^ { 2 } } - ( 1 - 2 \eta + \gamma ) \| y - u \| + ( \gamma - \eta ) \| y - v \| } \\ & { \geq A \sqrt { ( 2 \eta - \gamma ) ( 2 - 2 \eta + \gamma ) } + ( \gamma - \eta ) \| y - v \| } \\ & { \geq \frac { R ( 2 \eta - \gamma ) } { \eta } + \frac { R ( \gamma - \eta ) } { \eta } } \\ & { = R . } \end{array}
$$

The first inequality uses the reverse triangle inequality, the second uses (15) with $a =$ $1 - 2 \eta + \gamma$ , and the third uses the preceding bound and $\eta \| y - v \| = R$

Finally, suppose that $R \geq A \sqrt { \gamma ( 2 - \gamma ) }$ and $y \ne v$ . The preceding argument with radius $A \sqrt { \gamma ( 2 - \gamma ) }$ gives the lower bound $A \sqrt { \gamma ( 2 - \gamma ) }$ at

$$
\eta = \operatorname* { m i n } \left\{ \gamma , \frac { A \sqrt { \gamma ( 2 - \gamma ) } } { \Vert y - v \Vert } \right\} .
$$

The case $\eta = \gamma$ gives the same lower bound at the other endpoint. Since the left side of (14) is afine in $\eta ,$ , the bound holds whenever

$$
\operatorname* { m i n } \left\{ \gamma , { \frac { A { \sqrt { \gamma ( 2 - \gamma ) } } } { \| y - v \| } } \right\} \leq \eta \leq \gamma .
$$

The actual choice $\eta =$ min $\{ \gamma , R / \Vert y - v \Vert \}$ satisfies these inequalities, which proves the claim.

## A.5 Proof of Theorem 8

Proof Fix $q > 0$ . The function $\Lambda ( q , \cdot )$ is positive and continuous on $[ 0 , 1 )$ and diverges as $\delta \uparrow 1$ , and it therefore attains its minimum. Logarithmic diferentiation yields

$$
\frac { \partial } { \partial \delta } \log \Lambda ( q , \delta ) = \frac { 4 q \delta } { 1 + q \delta ^ { 2 } } + \frac { 2 \delta } { 1 - \delta ^ { 2 } } - \frac { 1 } { q + \delta } ,
$$

and multiplication by the positive quantity $( 1 + q \delta ^ { 2 } ) ( 1 - \delta ^ { 2 } ) ( q + \delta )$ yields $P _ { q } ( \delta )$ . We have $P _ { q } ( 0 ) = - 1$ and $P _ { q } ( 1 / \sqrt { 3 } ) > 0$ . On $[ 0 , 1 / \sqrt { 3 } ]$ , the derivative

$$
P _ { q } ^ { \prime } ( \delta ) = 2 ( 2 - 3 \delta ^ { 2 } ) q ^ { 2 } + ( 2 + 6 \delta - 4 \delta ^ { 3 } ) q + 6 \delta
$$

is positive. On $[ 1 / \sqrt { 3 } , 1 )$ , the first two terms of $P _ { q }$ are positive and $3 \delta ^ { 2 } - 1 \geq 0$ . Hence $P _ { q }$ has a unique root $\delta ^ { * } ( q ) \in ( 0 , 1 / \sqrt { 3 } )$ , it is negative on $[ 0 , \delta ^ { * } ( q ) )$ and positive on $( \delta ^ { * } ( q ) , 1 )$ and $\Lambda ( q , \cdot )$ decreases on $[ 0 , \delta ^ { * } ( q ) ]$ and increases on $[ \delta ^ { * } ( q ) , 1 )$ . The point $\delta ^ { * } ( q )$ is consequently the unique minimizer.

For every $\delta \in [ 0 , 1 )$ ,

$$
\frac { \partial } { \partial q } \log \Lambda ( q , \delta ) = - \frac { q ( 2 - \delta ^ { 3 } ) + \delta } { q ( q + \delta ) ( 1 + q \delta ^ { 2 } ) } < 0 .
$$

For $q _ { 1 } ~ < ~ q _ { 2 }$ , this inequality yields $\phi ( q _ { 2 } ) ~ \leq ~ \Lambda ( q _ { 2 } , \delta ^ { * } ( q _ { 1 } ) ) ~ < ~ \Lambda ( q _ { 1 } , \delta ^ { * } ( q _ { 1 } ) ) ~ = ~ \phi ( q _ { 1 } ) ~$ , and $\phi$ is strictly decreasing. The function ϕ is the minimum of the jointly continuous function Λ over δ in the compact interval $[ 0 , 1 / \sqrt { 3 } ]$ , and it is therefore continuous. Finally, $\phi ( q ) \leq \Lambda ( q , 0 ) = q ^ { - 2 }$ , and the bounds $( 1 + q \delta ^ { 2 } ) ^ { 2 } \geq 1 , 1 - \delta ^ { 2 } \leq 1$ , and $q + \delta \leq q + 1$ yield $\phi ( q ) \geq 1 / ( q ( q + 1 ) )$

## A.6 Proof of Theorem 3

Proof We first prove the upper bound. Since $\phi$ is decreasing and $\phi ( q ^ { * } ( \lambda ) ) = \lambda$ , every pair $( q , \delta )$ satisfies

$$
\phi ( q ) \leq \Lambda ( q , \delta ) \leq \lambda \implies q ^ { * } ( \lambda ) \leq q .
$$

Taking $q = \lambda ^ { - 1 / 2 }$ and $\delta = 0$ gives

$$
\Lambda ( \lambda ^ { - 1 / 2 } , 0 ) = \lambda ,
$$

and hence $q ^ { * } ( \lambda ) \leq \lambda ^ { - 1 / 2 }$ . Let $K = 3 \sqrt { 3 } / 2$ . For $\lambda \geq K ^ { 2 }$ , take $q = K / \lambda$ and $\delta = 1 / \sqrt { 3 }$ Direct computation gives

$$
\frac { K } { q } - \Lambda \left( q , \frac { 1 } { \sqrt { 3 } } \right) = \frac { 9 \sqrt { 3 } - 6 - q } { 6 ( q + 1 / \sqrt { 3 } ) } \geq 0 ,
$$

where the inequality uses $q \leq 1 / K < 9 \sqrt { 3 } - 6$ . Thus $\Lambda ( K / \lambda , 1 / \sqrt { 3 } ) \le \lambda$ , so $q ^ { * } ( \lambda ) \leq K / \lambda$ Therefore

$$
C ^ { * } ( \lambda ) = 1 + q ^ { * } ( \lambda ) \leq 1 + \operatorname* { m i n } \left\{ \lambda ^ { - 1 / 2 } , \frac { K } { \lambda } \right\} .
$$

The min branches when $\lambda \le K ^ { 2 }$

We next establish the two limits. As $\lambda  0 ^ { + }$ , we have $q ^ { * } ( \lambda )  \infty$ . The bounds in Theorem 8 give

$$
\frac { q } { q + 1 } \leq q ^ { 2 } \phi ( q ) \leq 1 .
$$

Taking $q = q ^ { * } ( \lambda )$ and using the equality $\phi ( q ^ { * } ( \lambda ) ) = \lambda$ , we obtain

$$
1 = \operatorname* { l i m } _ { q ^ { * } ( \lambda ) \to \infty } \frac { q ^ { * } ( \lambda ) } { q ^ { * } ( \lambda ) + 1 } \leq \operatorname* { l i m } _ { \lambda \to 0 ^ { + } } q ^ { * } ( \lambda ) ^ { 2 } \lambda \leq 1 .
$$

Therefore,

$$
\operatorname* { l i m } _ { \lambda \to 0 ^ { + } } \sqrt { \lambda } C ^ { * } ( \lambda ) = \operatorname* { l i m } _ { \lambda \to 0 ^ { + } } \sqrt { \lambda } + \sqrt { \lambda q ^ { * } ( \lambda ) ^ { 2 } } = 1 .
$$

$\operatorname { A s } \lambda \to \infty$ , we have $q ^ { * } ( \lambda )  0$ . We first show that lim $\begin{array} { r } { { 1 } _ { q \to 0 } \delta ^ { * } ( q ) = 1 / \sqrt { 3 } } \end{array}$ . The polynomial $P _ { q }$ converges uniformly on [0, 1] to $3 \delta ^ { 2 } - 1$ . Since $P _ { q } ( \delta ^ { * } ( q ) ) = 0$ , this convergence implies

$$
\left| 3 \delta ^ { * } ( q ) ^ { 2 } - 1 \right| \leq \operatorname* { s u p } _ { \delta \in [ 0 , 1 ] } \left| P _ { q } ( \delta ) - ( 3 \delta ^ { 2 } - 1 ) \right| = 0 .
$$

Together with $\delta ^ { * } ( q ) \in [ 0 , 1 ]$ , this proves the claimed limit. Consequently,

$$
\operatorname * { l i m } _ { q \to 0 } q \phi ( q ) = \frac { ( 1 + q \delta ^ { * } ( q ) ^ { 2 } ) ^ { 2 } } { ( 1 - \delta ^ { * } ( q ) ^ { 2 } ) ( q + \delta ^ { * } ( q ) ) } = \frac { 3 \sqrt { 3 } } { 2 } .
$$

Substituting $q = q ^ { * } ( \lambda )$ gives

$$
\operatorname * { l i m } _ { \lambda \to \infty } \lambda \bigl ( C ^ { * } ( \lambda ) - 1 \bigr ) = \operatorname * { l i m } _ { q ^ { * } ( \lambda ) \to 0 } q ^ { * } ( \lambda ) \phi ( q ^ { * } ( \lambda ) ) = \frac { 3 \sqrt { 3 } } { 2 } .
$$

This completes the proof.

## A.7 Proof of Theorem 9

Proof Put $q = q _ { * } ( \lambda )$ and $\delta = \delta _ { * } ( q )$ . Let $C = 1 + q$ be the targeted competitive ratio, $a = h / ( 1 - \delta ) , b = a ^ { 2 } / ( q ( 1 + \delta ) )$ be the parameters of the proposed algorithm, and $h = 1 + q \delta ^ { 2 }$ be a frequent notation in the proof. They depend only on $\lambda .$ By the previous section proving the upper bound,

$$
\lambda = \frac { h ^ { 2 } } { q ( 1 - \delta ^ { 2 } ) ( q + \delta ) } , P _ { q } ( \delta ) = 0 , 0 < \delta < \frac { 1 } { \sqrt { 3 } } ,
$$

where δ is the unique root of $P _ { q } ( \delta ) = 2 \delta ( 2 - \delta ^ { 2 } ) q ^ { 2 } + \delta ( 2 + 3 \delta - \delta ^ { 3 } ) q + 3 \delta ^ { 2 } - 1 = 0 \mathrm { i n } ( 0 , 1 / \sqrt { 3 } )$ Fix $0 \leq R < C$ . We will construct finite integers $L \geq 1$ and $T _ { 0 } \geq 2$ at the end, using only $\lambda , R .$ . For now, take arbitrary integers $L \geq 1$ and $T \geq 2$ , set $d = T + L + 1$ , and fix any online algorithm $\mathcal { A }$ in this dimension and horizon. All requests and feasible decisions in the constructed instance belong to $\mathbb { R } ^ { d }$ . Let cost $\begin{array} { r } { ( \mathbf { x } _ { t } ) = \sum _ { t = 1 } ^ { T } f _ { t } ( \mathbf { x } _ { t } ) + 1 / 2 \| \mathbf { x } _ { t } - \mathbf { x } _ { t - 1 } \| ^ { 2 } } \end{array}$ denote the total cost of a path $\mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { T }$ under the constructed instance.

We first construct a reference path $\{ X _ { t } \} _ { t = 0 } ^ { T }$ such that every online path $\{ Y _ { t } \} _ { t = 0 } ^ { T }$ generated by satisfies

$$
\mathbb { E } [ \mathrm { A L G } _ { T } ] = \mathbb { E } [ \mathrm { c o s t } ( Y _ { t } ) ] \geq \mathrm { c o s t } ( X _ { t } ) .\tag{16}
$$

The reference cost will be deterministic even though $X _ { t }$ is random. Let $\left\{ { { \mathbf { e } } _ { t } } \right\} _ { t = 0 } ^ { T + L }$ be the orthonormal coordinate vectors of $\mathbb { R } ^ { d }$ , and $\{ \sigma _ { t } \} _ { t = 0 } ^ { T + L }$ be independent uniform signs in $\{ - 1 , 1 \}$ Sample all these signs before play, and define

$$
\mathbf { z } _ { t } = \left\{ \begin{array} { l l } { \mathbf { e } _ { 0 } , } & { \mathrm { ~ i f ~ } t = 0 } \\ { \mu \mathbf { z } _ { t - 1 } + \sqrt { 1 - \mu ^ { 2 } } \sigma _ { t } \mathbf { e } _ { t } , } & { \mathrm { ~ o t h e r w i s e } } \end{array} \right. .\tag{17}
$$

Here we define

$$
r = \frac { q \delta } { a } , \quad G = \frac { b } { \delta } , \quad \mu = 1 - G r = - \frac { C \delta ^ { 2 } } { 1 - \delta ^ { 2 } } .\tag{18}
$$

To verify $- 1 < \mu < 0$ , evaluate $P _ { q }$ at $1 / \sqrt { q + 2 }$ . Grouping the even and odd powers gives

$$
\begin{array} { l c l } { { P _ { q } \left( \displaystyle \frac { 1 } { \sqrt { q + 2 } } \right) = \displaystyle \frac { 2 q ^ { 2 } } { \sqrt { q + 2 } } \left( 2 - \displaystyle \frac { 1 } { q + 2 } \right) + \displaystyle \frac { 2 q } { \sqrt { q + 2 } } + \displaystyle \frac { 3 ( q + 1 ) } { q + 2 } - \displaystyle \frac { q } { ( q + 2 ) ^ { 2 } } - 1 } } \\ { { \ } } \\ { { \displaystyle = \displaystyle \frac { 2 q ^ { 2 } } { \sqrt { q + 2 } } \left( 2 - \displaystyle \frac { 1 } { q + 2 } \right) + \displaystyle \frac { 2 q } { \sqrt { q + 2 } } + \displaystyle \frac { 3 ( q + 1 ) ( q + 2 ) - q - ( q + 2 ) ^ { 2 } } { ( q + 2 ) ^ { 2 } } } } \\ { { \ } } \\ { { \displaystyle = \displaystyle \frac { 2 q ^ { 2 } } { \sqrt { q + 2 } } \left( 2 - \displaystyle \frac { 1 } { q + 2 } \right) + \displaystyle \frac { 2 q } { \sqrt { q + 2 } } + \displaystyle \frac { 2 ( q + 1 ) ^ { 2 } } { ( q + 2 ) ^ { 2 } } } } \\ { { \ } } \\ { { \displaystyle > 0 . } } \end{array}
$$

Previous section has shown that $P _ { q }$ is negative before its unique root and positive afterward on (0, 1). Hence

$$
\delta < \frac { 1 } { \sqrt { q + 2 } } \Longrightarrow C \delta ^ { 2 } < 1 - \delta ^ { 2 } .
$$

Together with the expression for $\mu$ in (18), this proves $\mu \in ( - 1 , 0 )$ . Since $\mathbf { z } _ { t - 1 }$ lies in the span of $\mathbf { e } _ { 0 } , \ldots , \mathbf { e } _ { t - 1 }$ , it is orthogonal to $\mathbf { e } _ { t }$ . Induction gives

$$
\begin{array}{c} \left\{ \| \mathbf { z } _ { 0 } \| = 0 \right. \qquad & { \implies \| \mathbf { z } _ { t } \| ^ { 2 } = 1 , { \mathrm { ~ f o r ~ a l l ~ } } t \geq 0 . \qquad } \end{array}
$$

Thus

$$
\langle \mathbf { z } _ { s } , \mathbf { z } _ { t } \rangle = \mu \langle \mathbf { z } _ { s } , \mathbf { z } _ { t - 1 } \rangle = \cdot \cdot \cdot = \mu ^ { t - s } , { \mathrm { ~ f o r ~ a l l ~ } } 0 \leq s < t .\tag{19}
$$

Independence and the zero mean of $\sigma _ { t + 1 }$ also give

$$
\mathbb { E } \left[ \mathbf { z } _ { t + 1 } \mid \sigma _ { 1 } , \ldots , \sigma _ { t } \right] = \mu \mathbf { z } _ { t } .\tag{20}
$$

Define the reference path by

$$
X _ { t } = \left\{ \begin{array} { l l } { 0 , } & { \mathrm { ~ i f ~ } t = 0 } \\ { X _ { t - 1 } - \frac { \mathbf { z } _ { t } } { r } , } & { \mathrm { ~ o t h e r w i s e } } \end{array} \right. .\tag{21}
$$

For the nonterminal rounds, set

$$
f _ { t } ( \mathbf { x } ) = \left\{ \begin{array} { l l } { \operatorname* { m a x } \left\{ \frac { \lambda } { 2 } \| \mathbf { x } - \mathbf { v } _ { t } \| ^ { 2 } , G \langle \mathbf { z } _ { t } , \mathbf { x } - \mathbf { v } _ { t } \rangle \right\} , } & { \mathrm { ~ i f ~ } t < T } \\ { \frac { \lambda } { 2 } \| \mathbf { x } - X _ { T } + \frac { \mathbf { z } _ { T } } { \lambda r } \| ^ { 2 } , } & { \mathrm { ~ o t h e r w i s e } } \end{array} , \right.
$$

where ${ \mathbf v } _ { t } = X _ { t } - { \mathbf z } _ { t }$ . In case of $t < T$ , these functions are convex and nonnegative, have unique minimizers $\mathbf { v } _ { t }$ with value zero, and satisfy the required quadratic-growth inequality. $\operatorname { A t } X _ { t }$ , the two max branches in $t < T$ case have values $\lambda / 2$ and $G$ respectively. To identify the active branch, compute

$$
{ \begin{array} { r l } & { { \frac { G } { \lambda } } = { \frac { a ^ { 2 } } { q \delta ( 1 + \delta ) } } { \frac { q ( 1 - \delta ^ { 2 } ) ( q + \delta ) } { h ^ { 2 } } } = { \frac { a ^ { 2 } ( 1 - \delta ) ( q + \delta ) } { \delta h ^ { 2 } } } } \\ & { \quad = { \frac { q + \delta } { \delta ( 1 - \delta ) } } = 1 + { \frac { q + \delta ^ { 2 } } { \delta ( 1 - \delta ) } } > 1 . } \end{array} }
$$

The third equality uses $a = h / ( 1 - \delta )$ . By the above inequality, the afine branch is strictly active there, so

$$
f _ { t } ( X _ { t } ) = G , \quad \nabla f _ { t } ( X _ { t } ) = G \mathbf { z } _ { t } , \mathrm { ~ f o r ~ a l l ~ } 1 \leq t < T .\tag{22}
$$

In the case of $t = T , f _ { t } ( \mathbf { x } )$ is a nonnegative λ-QG cost with minimizer $X _ { T } - \mathbf { z } _ { T } / ( \lambda r )$ , and

$$
f _ { T } ( X _ { T } ) = \frac { 1 } { 2 \lambda r ^ { 2 } } , \quad \nabla f _ { T } ( X _ { T } ) = \frac { { \bf z } _ { T } } { r } .\tag{23}
$$

The input depends only on the signs sampled before play, not on the learner’s decisions.

We now prove (16). For any path $Y _ { 0 } = 0 , Y _ { 1 } , \ldots , Y _ { T }$ , set $\Delta _ { t } = Y _ { t } - X _ { t }$ , so $\Delta _ { 0 } = 0$ and

$$
Y _ { t } - Y _ { t - 1 } = - \frac { \mathbf { z } _ { t } } { r } + \Delta _ { t } - \Delta _ { t - 1 } .
$$

The movement-cost diference is exactly

$$
\frac { 1 } { 2 } \| Y _ { t } - Y _ { t - 1 } \| ^ { 2 } - \frac { 1 } { 2 } \| X _ { t } - X _ { t - 1 } \| ^ { 2 } = - \frac { 1 } { r } \langle \mathbf { z } _ { t } , \Delta _ { t } - \Delta _ { t - 1 } \rangle + \frac { 1 } { 2 } \| \Delta _ { t } - \Delta _ { t - 1 } \| ^ { 2 } .
$$

Convexity of $f _ { t } ( \cdot )$ , (22) and (23) give

$$
\begin{array} { r l } & { f _ { t } ( Y _ { t } ) - f _ { t } ( X _ { t } ) \geq \langle \nabla f _ { t } ( X _ { t } ) , Y _ { t } - X _ { t } \rangle = G \langle \mathbf { z } _ { t } , \Delta _ { t } \rangle , \mathrm { ~ f o r ~ a l l ~ } t < T , } \\ & { f _ { T } ( Y _ { T } ) - f _ { T } ( X _ { T } ) \geq \langle \nabla f _ { T } ( X _ { T } ) , Y _ { T } - X _ { T } \rangle = \displaystyle \frac { 1 } { r } \langle \mathbf { z } _ { T } , \Delta _ { T } \rangle . } \end{array}
$$

Therefore,

$$
\begin{array} { l } { { \displaystyle \mathrm { c o s t } ( Y _ { t } ) - \mathrm { c o s t } ( X _ { t } ) } } \\ { { \displaystyle \geq G \sum _ { t = 1 } ^ { T - 1 } \langle \mathbf { z } _ { t } , \Delta _ { t } \rangle + \frac { 1 } { r } \langle \mathbf { z } _ { T } , \Delta _ { T } \rangle - \frac { 1 } { r } \sum _ { t = 1 } ^ { T } \langle \mathbf { z } _ { t } , \Delta _ { t } - \Delta _ { t - 1 } \rangle + \frac { 1 } { 2 } \displaystyle \sum _ { t = 1 } ^ { T } \lVert \Delta _ { t } - \Delta _ { t - 1 } \rVert ^ { 2 } } } \\ { { \displaystyle = G \sum _ { t = 1 } ^ { T - 1 } \langle \mathbf { z } _ { t } , \Delta _ { t } \rangle + \frac { 1 } { r } \langle \mathbf { z } _ { T } , \Delta _ { T } \rangle - \frac { 1 } { r } \sum _ { t = 1 } ^ { T } \langle \mathbf { z } _ { t } , \Delta _ { t } \rangle + \frac { 1 } { r } \sum _ { t = 1 } ^ { T } \langle \mathbf { z } _ { t } , \Delta _ { t - 1 } \rangle + \frac { 1 } { 2 } \displaystyle \sum _ { t = 1 } ^ { T } \lVert \Delta _ { t } - \Delta _ { t - 1 } \rVert ^ { 2 } } } \\ { { \displaystyle = \sum _ { t = 1 } ^ { T - 1 } \langle ( G - \frac { 1 } { r } ) \mathbf { z } _ { t } + \frac { 1 } { r } \mathbf { z } _ { t + 1 } , \Delta _ { t } \rangle + \frac { 1 } { 2 } \displaystyle \sum _ { t = 1 } ^ { T } \lVert \Delta _ { t } - \Delta _ { t - 1 } \rVert ^ { 2 } } . } \end{array}\tag{24}
$$

The last step shifts the index in the last inner-product sum, uses $\Delta _ { 0 } = 0$ , and cancels the two terms involving $\Delta _ { T }$

Let $\mathcal { F } _ { t }$ be the sigma-field generated by $\sigma _ { 1 } , \ldots , \sigma _ { t }$ and $\xi .$ For $t < T$ , the requests revealed through round t depend only on $\sigma _ { 1 } , \ldots , \sigma _ { t }$ . Hence $\Delta _ { t }$ is $\mathcal { F } _ { t }$ -measurable for an online path. Using (20) and $\mu = 1 - G r$ , we have

$$
\begin{array} { r l } & { \mathbb E \left[ \langle ( G - \frac { 1 } { r } ) \mathbf z _ { t } + \frac { 1 } { r } \mathbf z _ { t + 1 } , \Delta _ { t } \rangle \right] = \mathbb E \left[ \langle \mathbb E [ ( G - \frac { 1 } { r } ) \mathbf z _ { t } + \frac { 1 } { r } \mathbf z _ { t + 1 } \mid \mathcal F _ { t } ] , \Delta _ { t } \rangle \right] } \\ & { \quad \quad \quad \quad = \mathbb E \left[ \langle ( G - \frac { 1 } { r } + \frac { \mu } { r } ) \mathbf z _ { t } , \Delta _ { t } \rangle \right] } \\ & { \quad \quad \quad \quad = \mathbb E \left[ \langle ( G - \frac { 1 } { r } + \frac { 1 - G r } { r } ) \mathbf z _ { t } , \Delta _ { t } \rangle \right] } \\ & { \quad \quad \quad = 0 . } \end{array}
$$

Therefore,

$$
\mathbb { E } [ \mathrm { c o s t } ( Y _ { t } ) ] - \mathbb { E } [ \mathrm { c o s t } ( X _ { t } ) ] = \mathbb { E } \left[ \frac { 1 } { 2 } \sum _ { t = 1 } ^ { T } \lVert \Delta _ { t } - \Delta _ { t - 1 } \rVert ^ { 2 } \right] \geq 0 .
$$

Taking expectations in (24) yields

$$
\begin{array} { r l } { \mathbb { E } _ { \sigma , \xi } [ \mathrm { A L G } _ { T } ] = \mathbb { E } _ { \sigma , \xi } \left[ \mathrm { c o s t } ( Y _ { t } ) \right] } & { } \\ & { \geq \mathbb { E } _ { \sigma , \xi } [ \mathrm { c o s t } ( X _ { t } ) ] } \\ & { = ( T - 1 ) \left( G + \frac { 1 } { 2 r ^ { 2 } } \right) + \frac { 1 } { 2 \lambda r ^ { 2 } } + \frac { 1 } { 2 r ^ { 2 } } } \\ & { = ( T - 1 ) \left( G + \frac { 1 } { 2 r ^ { 2 } } \right) + \frac { 1 + 1 / \lambda } { 2 r ^ { 2 } } . } \end{array}\tag{25}
$$

Here the second equality is by (22) and (23). This argument still applies if the learner knows $T$ and the input distribution.

We next construct a feasible ofline path $\{ U _ { t } \} _ { t = 0 } ^ { T }$ with small total cost. Since the ofline optimum minimizes over all feasible paths, this will give $\mathrm { O P T } _ { T } \leq \cot ( U _ { t } )$ for each fixed input. Introduce the weight

$$
\beta = \frac { q ( 1 - \delta ^ { 2 } ) } { C } .
$$

It is positive, and

$$
1 - \beta = \frac { C - q ( 1 - \delta ^ { 2 } ) } { C } = \frac { 1 + q - q + q \delta ^ { 2 } } { C } = \frac { h } { C } > 0 .
$$

Hence $0 < \beta < 1$ . Use the next L directions to define

$$
U _ { t } = \left\{ { \begin{array} { l l } { 0 , } & { { \mathrm { ~ i f ~ } } t = 0 } \\ { X _ { t } - E _ { t } ^ { ( L ) } , } & { { \mathrm { ~ f o r ~ a l l ~ } } 1 \leq t < T , } \\ { X _ { T } , } & { { \mathrm { ~ i f ~ } } t = T } \end{array} } \right.\tag{26}
$$

where

$$
E _ { t } ^ { ( L ) } = \frac { 1 } { r } \sum _ { j = 1 } ^ { L } \beta ^ { j } \mathbf { z } _ { t + j } .
$$

Each $U _ { t }$ belongs to the same $\mathbb { R } ^ { d }$ as the requests, and $U _ { 0 } = \mathbf { x } _ { 0 }$ . Therefore,

$$
\mathrm { O P T } _ { T } \leq \mathrm { c o s t } ( U _ { t } ) .
$$

To compute the cost of $\{ U _ { t } \} _ { t = 0 } ^ { T }$ , we first evaluate its untruncated counterpart

$$
{ \widehat { U } } _ { t } = X _ { t } - E _ { t } , { \mathrm { ~ w h e r e ~ } } E _ { t } = { \frac { 1 } { r } } \sum _ { j = 1 } ^ { \infty } \beta ^ { j } \mathbf { z } _ { t + j } , .
$$

To show the validation of this infinite series $\{ \mathbf { z } _ { t } \} _ { t = 0 } ^ { \infty } ,$ we embed $\mathbb { R } ^ { d }$ into the Hilbert space $\mathbb { R } ^ { \infty }$ satisfying square-summable sequences $\begin{array} { r } { \| \mathbf { x } \| ^ { 2 } = \sum _ { t = 0 } ^ { \infty } x _ { t } ^ { 2 } < \infty } \end{array}$ , and extend coordinate basis to $\{ \mathbf { e } _ { t } \} _ { t = 0 } ^ { \infty }$ and additional independent uniform signs $\{ \sigma _ { t } \} _ { t = 0 } ^ { \infty }$ . Use the same recursion (17) and (21) to extend $\mathbf { z } _ { t }$ and $X _ { t }$ , and $f _ { t } ( \mathbf { x } ) = \mathrm { m a x } \{ \lambda / 2 \| \mathbf { x } - \mathbf { v } _ { t } \| ^ { 2 } , G \langle \mathbf { z } _ { t } , \mathbf { x } - \mathbf { v } _ { t } \rangle \}$ to extend $f _ { t } ( \cdot )$ . This extension does not change the finite-dimensional instance or the path $\{ U \} _ { t = 0 } ^ { T } .$ Since $0 < \beta < 1$ and $\left\| \mathbf { z } _ { t } \right\| = 1$ , this series converges absolutely.

We consider evaluate the cost of $\{ \widehat { U } _ { t } \} _ { t = 0 } ^ { T }$ . To compute $f _ { t } ( \widehat { U } _ { t } ) = \operatorname* { m a x } \{ G \langle \mathbf { z } _ { t } , \widehat { U } _ { t } - \mathbf { v } _ { t } \rangle , \lambda / 2 \| \widehat { U } _ { t } -$ $\mathbf { v } _ { t } \| ^ { 2 } \}$ b, we first compute the inner product $\langle \mathbf { z } _ { t } , \widehat { U } _ { t } - \mathbf { v } _ { t } \rangle$ . Since $\mathbf { v } _ { t } = X _ { t } - \mathbf { z } _ { t } , { \hat { U } } _ { t } - \mathbf { v } _ { t } = \mathbf { z } _ { t } - E _ { t }$ The afine branch equals

$$
G \langle \mathbf { z } _ { t } , \widehat { U } _ { t } - \mathbf { v } _ { t } \rangle = G \langle \mathbf { z } _ { t } , \mathbf { z } _ { t } - E _ { t } \rangle = G ( 1 - \langle \mathbf { z } _ { t } , E _ { t } \rangle ) .
$$

For the inner product of $\mathbf { z } _ { t }$ and $E _ { t }$ , (19) gives

$$
\begin{array} { l } { \displaystyle \langle \mathbf { z } _ { t } , E _ { t } \rangle = \frac { 1 } { r } \sum _ { j = 1 } ^ { \infty } \beta ^ { j } \mu ^ { j } = \frac { \beta \mu } { r ( 1 - \beta \mu ) } } \\ { \displaystyle = - \frac { q \delta ^ { 2 } } { r h } = - \frac { q \delta ^ { 2 } a } { q \delta h } = - \frac { a \delta } { h } } \\ { \displaystyle = - \frac { \delta } { 1 - \delta } . } \end{array}
$$

Here the third equality is by

$$
\beta \mu = \frac { q ( 1 - \delta ^ { 2 } ) } { C } \left( - \frac { C \delta ^ { 2 } } { 1 - \delta ^ { 2 } } \right) = - q \delta ^ { 2 } , \quad 1 - \beta \mu = h .
$$

Therefore,

$$
G \langle \mathbf { z } _ { t } , \widehat { U } _ { t } - \mathbf { v } _ { t } \rangle = \frac { G } { 1 - \delta } .
$$

Then we compute $\lambda / 2 \| \widehat { U } _ { t } - \mathbf { v } _ { t } \| ^ { 2 }$

$$
\| \widehat { U } _ { t } - \mathbf { v } _ { t } \| ^ { 2 } = \| \mathbf { z } _ { t } - E _ { t } \| ^ { 2 } = 1 - 2 \langle \mathbf { z } _ { t } , E _ { t } \rangle + \| E _ { t } \| ^ { 2 } = 1 + \frac { 2 \delta } { 1 - \delta } + \| E _ { t } \| ^ { 2 } .
$$

For the squared norm term $\| E _ { t } \| ^ { 2 }$ , absolute convergence permits grouping the diagonal terms and the two equal of-diagonal sums:

$$
\begin{array} { l } { \displaystyle \| E _ { t } \| ^ { 2 } - \frac { 1 } { r ^ { 2 } } \sum _ { i = 1 } ^ { \infty } \sum _ { j = 1 } ^ { \infty } \beta ^ { i \cdot 1 } j ^ { | i \cdot | i \cdot j | } } \\ { = \frac { 1 } { r ^ { 2 } } \left( \frac { \displaystyle \sum _ { i = 1 } ^ { \infty } \beta ^ { i \cdot 1 } + 2 \displaystyle \sum _ { i = 1 } ^ { \infty } \sum _ { k = 1 } ^ { \infty } \beta ^ { 2 i \cdot 1 } \mu ^ { i } } { \displaystyle \sum _ { i = 1 } ^ { \infty } \beta ^ { i \cdot 1 } \mu ^ { i } } \right) } \\ { = \frac { \displaystyle \beta ^ { 2 } } { \displaystyle r ^ { 2 } ( 1 - \beta ^ { 2 } ) } \left( 1 + 2 \sum _ { \ell = 1 } ^ { \infty } ( \beta \mu ) ^ { \ell } \right) } \\ { = \frac { \displaystyle \beta ^ { 2 } } { \displaystyle r ^ { 2 } ( 1 - \beta ^ { 2 } ) } \left( 1 + \frac { 2 \beta \mu } { 1 - \beta \mu } \right) } \\ { = \frac { \displaystyle \beta ^ { 2 } ( 1 + \beta \mu ) } { \displaystyle r ^ { 2 } ( 1 - \beta ^ { 2 } ) ( 1 - \beta \mu ) } - \left( \frac { \beta } { r } \right) ^ { 2 } \frac { 1 } { 1 - \beta ^ { 2 } } \frac { 1 + \beta \mu } { 1 - \beta \mu } . } \end{array}
$$

We next substitute the parameters $\beta = ( q ( 1 - \delta ^ { 2 } ) ) / C , r = ( q \delta ) / a , a = h / ( 1 - \delta )$ into the above equation. The two factors needed for the substitution are

$$
\begin{array} { r l } & { \quad \quad \quad \frac { \beta } { r } = \frac { q ( 1 - \delta ^ { 2 } ) } { C } \frac { a } { q \delta } = \frac { a ( 1 - \delta ^ { 2 } ) } { C \delta } = \frac { h ( 1 + \delta ) } { C \delta } , } \\ & { 1 - \beta ^ { 2 } = \frac { C ^ { 2 } - q ^ { 2 } ( 1 - \delta ^ { 2 } ) ^ { 2 } } { C ^ { 2 } } = \frac { ( C - q ( 1 - \delta ^ { 2 } ) ) ( C + q ( 1 - \delta ^ { 2 } ) ) } { C ^ { 2 } } = \frac { h ( 1 + 2 q - q \delta ^ { 2 } ) } { C ^ { 2 } } . } \end{array}
$$

Together with $\beta \mu = - q \delta ^ { 2 }$ and $1 - \beta \mu = h$ , these identities give

$$
\| E _ { t } \| ^ { 2 } = \frac { h ^ { 2 } ( 1 + \delta ) ^ { 2 } } { C ^ { 2 } \delta ^ { 2 } } \frac { ( 1 - q \delta ^ { 2 } ) C ^ { 2 } } { h ^ { 2 } ( 1 + 2 q - q \delta ^ { 2 } ) } = \frac { ( 1 + \delta ) ^ { 2 } ( 1 - q \delta ^ { 2 } ) } { \delta ^ { 2 } ( 1 + 2 q - q \delta ^ { 2 } ) } .\tag{27}
$$

In particular, $\| E _ { t } \| ^ { 2 }$ is deterministic and independent of t. To further evaluate $( ( 1 + \delta ) ^ { 2 } ( 1$ $q \delta ^ { 2 } ) ) / ( \delta ^ { 2 } ( 1 + 2 q - q \delta ^ { 2 } ) )$ , we use the equation $P _ { q } ( \delta ) = 2 \delta ( 2 - \delta ^ { 2 } ) q ^ { 2 } + \delta ( 2 + 3 \delta - \delta ^ { 3 } ) q + 3 \delta ^ { 2 } - 1 = 0$ and expand

$$
( 1 + \delta ) ^ { 2 } ( 1 - q \delta ^ { 2 } ) \cdot ( 1 - \delta ) ^ { 2 } = 1 - 2 \delta ^ { 2 } + \delta ^ { 4 } - q ( \delta ^ { 2 } - 2 \delta ^ { 4 } + \delta ^ { 6 } ) , ,
$$

and

$$
\delta ( 1 + 2 q - q \delta ^ { 2 } ) \cdot ( 2 q + \delta + \delta ^ { 3 } ) = ( 4 \delta - 2 \delta ^ { 3 } ) q ^ { 2 } + ( 2 \delta + 2 \delta ^ { 2 } + \delta ^ { 4 } - \delta ^ { 6 } ) q + \delta ^ { 2 } + \delta ^ { 4 } .
$$

Subtracting these expansions yields

$$
\begin{array} { r l } & { \quad ( 1 + \delta ) ^ { 2 } ( 1 - q \delta ^ { 2 } ) \cdot ( 1 - \delta ) ^ { 2 } - \delta ( 1 + 2 q - q \delta ^ { 2 } ) \cdot ( 2 q + \delta + \delta ^ { 3 } ) } \\ & { = 1 - 3 \delta ^ { 2 } - ( 2 \delta + 3 \delta ^ { 2 } - \delta ^ { 4 } ) q - ( 4 \delta - 2 \delta ^ { 3 } ) q ^ { 2 } } \\ & { = \ - \ P _ { q } ( \delta ) = 0 . } \end{array}
$$

Dividing by the positive denominator $\delta ^ { 2 } ( 1 - \delta ) ^ { 2 } ( 1 + 2 q - q \delta ^ { 2 } )$ gives

$$
\frac { ( 1 + \delta ) ^ { 2 } ( 1 - q \delta ^ { 2 } ) } { \delta ^ { 2 } ( 1 + 2 q - q \delta ^ { 2 } ) } = \frac { 2 q + \delta + \delta ^ { 3 } } { \delta ( 1 - \delta ) ^ { 2 } } .
$$

Therefore, $\lVert E _ { t } \rVert ^ { 2 } = ( 2 q + \delta + \delta ^ { 3 } ) / ( \delta ( 1 - \delta ) ^ { 2 } )$ , and

$$
\begin{array} { l } { \displaystyle | | \widehat { U } _ { t } - \mathbf { v } _ { t } | | ^ { 2 } = 1 - 2 \langle \mathbf { z } _ { t } , E _ { t } \rangle + | | E _ { t } | | ^ { 2 } } \\ { \displaystyle = 1 + \frac { 2 \delta } { 1 - \delta } + \frac { 2 q + \delta + \delta ^ { 3 } } { \delta ( 1 - \delta ) ^ { 2 } } } \\ { \displaystyle = \frac { \delta ( 1 - \delta ) ^ { 2 } + 2 \delta ^ { 2 } ( 1 - \delta ) + 2 q + \delta + \delta ^ { 3 } } { \delta ( 1 - \delta ) ^ { 2 } } } \\ { \displaystyle = \frac { 2 ( q + \delta ) } { \delta ( 1 - \delta ) ^ { 2 } } . } \end{array}
$$

By $\lambda = \Lambda _ { 0 } ( q , \delta ) = h ^ { 2 } / ( q ( 1 - \delta ^ { 2 } ) ( q + \delta ) )$ the quadratic branch at $\widehat { U } _ { t }$ equals

$$
\begin{array} { l } { \displaystyle \frac { \lambda } { 2 } \| \widehat { U } _ { t } - \mathbf { v } _ { t } \| ^ { 2 } = \lambda \cdot \displaystyle \frac { ( q + \delta ) } { \delta ( 1 - \delta ) ^ { 2 } } = \displaystyle \frac { h ^ { 2 } } { q ( 1 - \delta ^ { 2 } ) ( q + \delta ) } \cdot \displaystyle \frac { q + \delta } { \delta ( 1 - \delta ) ^ { 2 } } } \\ { \displaystyle = \displaystyle \frac { a ^ { 2 } } { q \delta ( 1 - \delta ^ { 2 } ) } = \displaystyle \frac { b } { \delta ( 1 - \delta ) } } \\ { \displaystyle = \displaystyle \frac { G } { 1 - \delta } . } \end{array}
$$

Combining the two max branches gives

$$
f _ { t } ( \widehat { U } _ { t } ) = \frac { G } { 1 - \delta } , \mathrm { ~ f o r ~ a l l ~ } 1 \leq t < T .
$$

We next evaluate the moving cost $\lVert \widehat { U } _ { t } - \widehat { U } _ { t - 1 } \rVert ^ { 2 }$

$$
\begin{array} { l } { { \displaystyle { \widehat { U } _ { t } - \widehat { U } _ { t - 1 } = - \frac { \mathbf { z } _ { t } } { r } - E _ { t } + E _ { t - 1 } } } } \\ { { \displaystyle ~ = - \frac { \mathbf { z } _ { t } } { r } - \frac { 1 } { r } \sum _ { j = 1 } ^ { \infty } \beta ^ { j } \mathbf { z } _ { t + j } + \frac { 1 } { r } \sum _ { j = 1 } ^ { \infty } \beta ^ { j } \mathbf { z } _ { t - 1 + j } } } \\ { { \displaystyle ~ = - \frac { 1 } { r \beta } \sum _ { j = 1 } ^ { \infty } \beta ^ { j } \mathbf { z } _ { t - 1 + j } + \frac { 1 } { r } \sum _ { j = 1 } ^ { \infty } \beta ^ { j } \mathbf { z } _ { t - 1 + j } } } \\ { { \displaystyle ~ = ( 1 - \frac { 1 } { \beta } ) E _ { t - 1 } . } } \end{array}
$$

Since $\| E _ { t - 1 } \| ^ { 2 } = \| E _ { t } \| ^ { 2 }$ , its nonterminal one-round cost is

$$
f _ { t } ( \widehat { U } _ { t } ) + \frac { 1 } { 2 } \| \widehat { U } _ { t } - \widehat { U } _ { t - 1 } \| ^ { 2 } = \frac { G } { 1 - \delta } + \frac { ( 1 - \beta ) ^ { 2 } } { 2 \beta ^ { 2 } } \| E _ { t } \| ^ { 2 } .\tag{28}
$$

We simplify the two terms using the expressions for $\| E _ { t } \| ^ { 2 }$ derived above. $\lVert E _ { t } \rVert ^ { 2 } = ( 2 q +$ $\delta + \delta ^ { 3 } ) / ( \delta ( 1 - \delta ) ^ { 2 } )$ and (27) implies

$$
\begin{array} { l } { \displaystyle \frac { 1 } { \delta ^ { 2 } } = q + \frac { 1 + 2 q - q \delta ^ { 2 } } { ( 1 + \delta ) ^ { 2 } } \| E _ { t } \| ^ { 2 } } \\ { \displaystyle = \frac { 1 + q } { ( 1 + \delta ) ^ { 2 } } \| E _ { t } \| ^ { 2 } + q \left( 1 + \frac { 1 - \delta } { 1 + \delta } \| E _ { t } \| ^ { 2 } \right) } \\ { \displaystyle = \frac { 1 + q } { ( 1 + \delta ) ^ { 2 } } \| E _ { t } \| ^ { 2 } + q \left( 1 + \frac { 2 q + \delta + \delta ^ { 3 } } { \delta ( 1 - \delta ^ { 2 } ) } \right) } \\ { \displaystyle = \frac { 1 + q } { ( 1 + \delta ) ^ { 2 } } \| E _ { t } \| ^ { 2 } + \frac { 2 q ( q + \delta ) } { \delta ( 1 - \delta ^ { 2 } ) } . } \end{array}
$$

Thus,

$$
\frac { \| E _ { t } \| ^ { 2 } } { ( 1 + \delta ) ^ { 2 } } = \frac { 1 } { 1 + q } \left( \frac { 1 } { \delta ^ { 2 } } - \frac { 2 q ( q + \delta ) } { \delta ( 1 - \delta ^ { 2 } ) } \right) .
$$

Since

$$
\frac { \lambda } { 2 } \| \mathbf { z } _ { t } - E _ { t } \| ^ { 2 } = \frac { G } { 1 - \delta } , \qquad \| \mathbf { z } _ { t } - E _ { t } \| ^ { 2 } = 1 + \frac { 2 \delta } { 1 - \delta } + \| E _ { t } \| ^ { 2 } ,
$$

by $\lVert E _ { t } \rVert ^ { 2 } = ( 2 q + \delta + \delta ^ { 3 } ) / ( \delta ( 1 - \delta ) ^ { 2 } )$ we obtain

$$
\begin{array} { l } { G = \displaystyle \frac { \lambda } { 2 } \left( 1 + \delta + ( 1 - \delta ) \| E _ { t } \| ^ { 2 } \right) = \displaystyle \frac { \lambda } { 2 } \left( 1 + \delta + \frac { 2 q + \delta + \delta ^ { 3 } } { \delta ( 1 - \delta ) } \right) } \\ { = \displaystyle \frac { \lambda ( q + \delta ) } { \delta ( 1 - \delta ) } = \displaystyle \frac { b } { \delta } = \frac { a ^ { 2 } } { q \delta ( 1 + \delta ) } , } \end{array}\tag{29}
$$

where we used $( q + \delta ) \lambda = ( 1 - \delta ) b$ and $b = a ^ { 2 } / [ q ( 1 + \delta ) ]$ . Substituting this expression and $( 1 - \beta ) / \beta = a / [ q ( 1 + \delta ) ]$ into (28) gives

$$
\begin{array} { r l } { f _ { i } ( \hat { U } _ { i } ) + \displaystyle \frac { 1 } { 2 } | \hat { U } _ { i } - \hat { U } _ { i - 1 } | ^ { 2 } = \displaystyle \frac { G } { 1 - \delta } + \frac { a ^ { 2 } } { 2 q ^ { 2 } } \frac { | E _ { i } | ^ { 2 } } { ( 1 + \delta ) ^ { 2 } } } & { } \\ { } & { = \displaystyle \frac { G } { 1 - \delta } + \frac { a ^ { 2 } } { 2 q ^ { 2 } ( 1 + q ) } \left( \frac { 1 } { \delta ^ { 2 } } - \frac { 2 q ( q + \delta ) } { \delta ( 1 - \delta ^ { 2 } ) } \right) } \\ { } & { = \displaystyle \frac { G } { 1 - \delta } + \frac { a ^ { 2 } } { 2 q ^ { 2 } ( 1 + q ) \delta ^ { 2 } } - \frac { q + \delta } { ( 1 + q ) ( 1 - \delta ) } \frac { a ^ { 2 } } { q \delta ( 1 + \delta ) } } \\ { } & { = \displaystyle \frac { G } { 1 - \delta } + \frac { a ^ { 2 } } { 2 q ^ { 2 } ( 1 + q ) \delta ^ { 2 } } - \frac { ( q + \delta ) G } { ( 1 + q ) ( 1 - \delta ) } } \\ { } & { = \displaystyle \frac { ( 1 + q ) - ( q + \delta ) } { ( 1 + q ) ( 1 - \delta ) } G + \frac { a ^ { 2 } } { 2 q ^ { 2 } ( 1 + q ) \delta ^ { 2 } } } \\ { } & { = \displaystyle \frac { 1 } { 1 + q } \left( G + \frac { a ^ { 2 } } { 2 q ^ { 2 } \delta ^ { 2 } } \right) , } \end{array}
$$

where the fourth equality uses (29). Finally, recalling $1 / r = a / ( q \delta )$ and $C = 1 + q$ , we conclude that

$$
f _ { t } ( \widehat { U } _ { t } ) + \frac { 1 } { 2 } \| \widehat { U } _ { t } - \widehat { U } _ { t - 1 } \| ^ { 2 } = \frac { G + 1 / ( 2 r ^ { 2 } ) } { C } , \mathrm { ~ f o r ~ a l l ~ } 1 \leq t < T .\tag{30}
$$

We now bound the truncation error and the cost of the two corrected endpoints in (26). At a nonterminal round, $U _ { t } - \mathbf { v } _ { t } = \mathbf { z } _ { t } - E _ { t } ^ { ( L ) }$ and $\widehat { U } _ { t } - { \mathbf v } _ { t } = { \mathbf z } _ { t } - E _ { t }$ , the finite and infinite sums satisfy

$$
\begin{array} { r } { \| E _ { t } \| , \| E _ { t } ^ { ( L ) } \| \le \displaystyle \frac { 1 } { r } \sum _ { j = 1 } ^ { \infty } \beta ^ { j } = \frac { \beta } { r ( 1 - \beta ) } , } \\ { \| U _ { t } - \widehat { U } _ { t } \| \le \| E _ { t } - E _ { t } ^ { ( L ) } \| \le \displaystyle \frac { 1 } { r } \sum _ { j = L + 1 } ^ { \infty } \beta ^ { j } = \frac { \beta ^ { L + 1 } } { r ( 1 - \beta ) } . } \end{array}
$$

At a nonterminal round, $U _ { t } - \mathbf { v } _ { t } = \mathbf { z } _ { t } - E _ { t } ^ { ( L ) }$ and ${ \widehat { U } } _ { t } - \mathbf { v } _ { t } = \mathbf { z } _ { t } - E _ { t } .$ , so both points lie within distance $1 + \beta / ( r ( 1 - \beta ) )$ of $\mathbf { v } _ { t }$ . Define

$$
M _ { f } = \operatorname* { m a x } \left\{ \lambda \left( 1 + \frac { \beta } { r ( 1 - \beta ) } \right) , G \right\} .
$$

The diference between the quadratic-branch values is at most

$$
\begin{array} { l } { \displaystyle \frac { \lambda } { 2 } \left| \| U _ { t } - \mathbf { v } _ { t } \| ^ { 2 } - \| \widehat { U } _ { t } - \mathbf { v } _ { t } \| ^ { 2 } \right| \leq \displaystyle \frac { \lambda } { 2 } \left( \| U _ { t } - \mathbf { v } _ { t } \| + \| \widehat { U } _ { t } - \mathbf { v } _ { t } \| \right) \| U _ { t } - \widehat { U } _ { t } \| } \\ { \displaystyle \leq \frac { \lambda } { 2 } \left( 2 \| \mathbf { z } _ { t } \| + \| E _ { t } \| + \| E _ { t } ^ { ( L ) } \| \right) \| U _ { t } - \widehat { U } _ { t } \| } \\ { \displaystyle \leq \lambda \left( 1 + \frac { \beta } { r ( 1 - \beta ) } \right) \frac { \beta ^ { L + 1 } } { r ( 1 - \beta ) } . } \end{array}
$$

The diference between the afine-branch values is at most

$$
G \left| \langle \mathbf { z } _ { t } , U _ { t } - \widehat { U } _ { t } \rangle \right| \leq G \| \mathbf { z } _ { t } \| \| U _ { t } - \widehat { U } _ { t } \| \leq \frac { G \beta ^ { L + 1 } } { r ( 1 - \beta ) } .
$$

The absolute diference of two maxima is bounded by the maximum of the absolute diferences of their corresponding branches. Consequently,

$$
| f _ { t } ( U _ { t } ) - f _ { t } ( \widehat { U } _ { t } ) | \leq \frac { M _ { f } \beta ^ { L + 1 } } { r ( 1 - \beta ) } .\tag{31}
$$

For $2 \leq t < T$ , the two movement vectors satisfy

$$
\begin{array} { r l r } & { } & { \| U _ { t } - U _ { t - 1 } \| = \| - \mathbf { z } _ { t } / r - E _ { t } ^ { ( L ) } + E _ { t - 1 } ^ { ( L ) } \| } \\ & { } & { \leq \frac { 1 } { r } + \frac { 2 \beta } { r ( 1 - \beta ) } = \frac { 1 + \beta } { r ( 1 - \beta ) } , } \\ & { } & { \| \widehat { U } _ { t } - \widehat { U } _ { t - 1 } \| = \| - \mathbf { z } _ { t } / r - E _ { t } + E _ { t - 1 } \| } \\ & { } & { \leq \frac { 1 } { r } + \frac { 2 \beta } { r ( 1 - \beta ) } = \frac { 1 + \beta } { r ( 1 - \beta ) } , } \\ & { } & { \| ( U _ { t } - U _ { t - 1 } ) - ( \widehat { U } _ { t } - \widehat { U } _ { t - 1 } ) \| \leq \| E _ { t } - E _ { t } ^ { ( L ) } \| + \| E _ { t - 1 } - E _ { t - 1 } ^ { ( L ) } \| } \\ & { } & { \leq \frac { 2 \beta ^ { L + 1 } } { r ( 1 - \beta ) } . } \end{array}
$$

Applying the diference-of-squares identity and the Cauchy–Schwarz inequality to these movement vectors gives

$$
\begin{array} { r l } & { \frac { 1 } { 2 } \left| \| U _ { t } - U _ { t - 1 } \| ^ { 2 } - \| \widehat { U } _ { t } - \widehat { U } _ { t - 1 } \| ^ { 2 } \right| } \\ & { = \left( \| U _ { t } - U _ { t - 1 } \| + \| \widehat { U } _ { t } - \widehat { U } _ { t - 1 } \| \right) \frac { 1 } { 2 } \left| \| U _ { t } - U _ { t - 1 } \| - \| \widehat { U } _ { t } - \widehat { U } _ { t - 1 } \| \right| } \\ & { \leq \frac { 1 } { 2 } \left( \frac { 2 ( 1 + \beta ) } { r ( 1 - \beta ) } \right) \left( \frac { 2 \beta ^ { L + 1 } } { r ( 1 - \beta ) } \right) } \\ & { = \frac { 2 ( 1 + \beta ) \beta ^ { L + 1 } } { r ^ { 2 } ( 1 - \beta ) ^ { 2 } } . } \end{array}
$$

Together with (30) and (31), this proves

$$
\begin{array} { r l } & { \quad f _ { t } ( U _ { t } ) + \displaystyle \frac { 1 } { 2 } \| U _ { t } - U _ { t - 1 } \| ^ { 2 } } \\ & { \le f _ { t } ( \widehat { U } _ { t } ) + \displaystyle \frac { 1 } { 2 } \| \widehat { U } _ { t } - \widehat { U } _ { t - 1 } \| ^ { 2 } + | f _ { t } ( U _ { t } ) - f _ { t } ( \widehat { U } _ { t } ) | + \displaystyle \frac { 1 } { 2 } \left| \| U _ { t } - U _ { t - 1 } \| ^ { 2 } - \| \widehat { U } _ { t } - \widehat { U } _ { t - 1 } \| ^ { 2 } \right| } \\ & { \le \displaystyle \frac { G + 1 / ( 2 r ^ { 2 } ) } { C } + \displaystyle \frac { \beta ^ { L + 1 } } { r ( 1 - \beta ) } \left( M _ { f } + \displaystyle \frac { 2 ( 1 + \beta ) } { r ( 1 - \beta ) } \right) , } \end{array}
$$

for all $2 \leq t < T$ . In the first round,

$$
\left\| U _ { 1 } - \mathbf { v } _ { 1 } \right\| \leq 1 + { \frac { \beta } { r ( 1 - \beta ) } } , \quad \left\| U _ { 1 } - U _ { 0 } \right\| \leq { \frac { 1 } { r } } + { \frac { \beta } { r ( 1 - \beta ) } } = { \frac { 1 } { r ( 1 - \beta ) } } .
$$

Define the resulting hitting-cost bound by

$$
H _ { 0 } = \operatorname* { m a x } \left\{ \frac { \lambda } { 2 } \left( 1 + \frac { \beta } { r ( 1 - \beta ) } \right) ^ { 2 } , G \left( 1 + \frac { \beta } { r ( 1 - \beta ) } \right) \right\} .
$$

Then

$$
f _ { 1 } ( U _ { 1 } ) + \frac { 1 } { 2 } \| U _ { 1 } - U _ { 0 } \| ^ { 2 } \leq H _ { 0 } + \frac { 1 } { 2 r ^ { 2 } ( 1 - \beta ) ^ { 2 } } .
$$

In the last round, $U _ { T } = X _ { T }$ and $U _ { T } - U _ { T - 1 } = - \mathbf { z } _ { T } / r + E _ { T - 1 } ^ { ( L ) }$ , so

$$
f _ { T } ( U _ { T } ) + \frac { 1 } { 2 } \| U _ { T } - U _ { T - 1 } \| ^ { 2 } \leq \frac { 1 } { 2 \lambda r ^ { 2 } } + \frac { 1 } { 2 r ^ { 2 } ( 1 - \beta ) ^ { 2 } } .
$$

Summing the interior and endpoint bounds yields

$$
\begin{array} { r l } & { \displaystyle \mathrm { O P T } _ { T } \leq \mathrm { c o s t } ( U ) = \displaystyle \sum _ { t = 1 } ^ { T } f _ { t } ( U _ { t } ) + \frac { 1 } { 2 } \| U _ { t } - U _ { t - 1 } \| ^ { 2 } } \\ & { \qquad \leq ( T - 2 ) \left( \frac { G + 1 / ( 2 r ^ { 2 } ) } { C } + \frac { \beta ^ { L + 1 } } { r \left( 1 - \beta \right) } \left( M _ { f } + \frac { 2 ( 1 + \beta ) } { r ( 1 - \beta ) } \right) \right) } \\ & { \qquad + H _ { 0 } + \frac { 1 } { 2 \lambda r ^ { 2 } } + \frac { 1 } { r ^ { 2 } ( 1 - \beta ) ^ { 2 } } . } \end{array}
$$

Here $M _ { f } , H _ { 0 }$ depend only on $\lambda ,$ and both the requests and the comparator are restricted to the space of original dimension.

The reference and auxiliary costs are now explicit. Choose a finite $L \geq 1$ such that

$$
R \frac { \beta ^ { L + 1 } } { r ( 1 - \beta ) } \left( M _ { f } + \frac { 2 ( 1 + \beta ) } { r ( 1 - \beta ) } \right) < \frac { 1 } { 4 } ( 1 - \frac { R } { C } ) ( G + \frac { 1 } { 2 r ^ { 2 } } ) .
$$

Such an L exists because $0 < \beta < 1$ . Next choose a finite $T _ { 0 } \geq 2$ such that

$$
\frac { A + R \left( H _ { 0 } + 1 / ( 2 \lambda r ^ { 2 } ) + 1 / ( r ^ { 2 } ( 1 - \beta ) ^ { 2 } ) \right) } { T _ { 0 } } < \frac { 1 } { 4 } ( 1 - \frac { R } { C } ) ( G + \frac { 1 } { 2 r ^ { 2 } } ) .
$$

Both choices depend only on $\lambda , R ,$ and are independent of the algorithm.

For every $T \geq T _ { 0 }$ , the online and ofline bounds imply

$$
\begin{array} { r l } & { \mathbb { E } _ { \sigma , \xi } ( \mathrm { A L G } _ { T } ) - R \mathbb { E } _ { \sigma } ( { \mathrm { O P T } } _ { T } ) } \\ & { \leq ( T - 1 ) ( G + \frac { 1 } { 2 r ^ { 2 } } ) } \\ & { \quad - R T \left( \frac { G + \frac { 1 } { 2 r ^ { 2 } } } { C } + \frac { \beta ^ { L + 1 } } { r ( 1 - \beta ) } \left( M _ { f } + \frac { 2 ( 1 + \beta ) } { r ( 1 - \beta ) } \right) \right) - R \left( H _ { 0 } + \frac { 1 } { 2 \lambda r ^ { 2 } } + \frac { 1 } { r ^ { 2 } ( 1 - \beta ) ^ { 2 } } \right) } \\ & { = T ( 1 - \frac { R } { C } ) ( G + \frac { 1 } { 2 r ^ { 2 } } ) } \\ & { \quad - R T \frac { \beta ^ { L + 1 } } { r ( 1 - \beta ) } \left( M _ { f } + \frac { 2 ( 1 + \beta ) } { r ( 1 - \beta ) } \right) - ( G + \frac { 1 } { 2 r ^ { 2 } } ) - R \left( H _ { 0 } + \frac { 1 } { 2 \lambda r ^ { 2 } } + \frac { 1 } { r ^ { 2 } ( 1 - \beta ) ^ { 2 } } \right) } \\ & { > \frac { 1 } { 2 } T ( 1 - \frac { R } { C } ) ( G + \frac { 1 } { 2 r ^ { 2 } } ) > 0 . } \end{array}
$$

The first inequality drops the positive terminal term in (25), and replaces $T - 2$ by $T$ in the nonnegative ofline bound. The second inequality follows from the choices of L and $T _ { 0 }$ above.

Therefore, for every $T \geq T _ { 0 }$

$$
\frac { \mathbb { E } _ { \xi } [ \mathrm { A L G } _ { T } ] } { \mathrm { O P T } _ { T } } \geq C = 1 + q _ { * } ( \lambda ) .
$$

This means

$$
\operatorname* { i n f } _ { A \in \mathcal { A } } \operatorname* { s u p } _ { \stackrel { d \in \mathcal { D } } { ( f _ { 1 } , \ldots , f _ { T } ) \in \mathcal { F } } } \frac { \mathbb { E } [ \mathrm { A L G } _ { T } ] } { \mathrm { O P T } _ { T } } \geq 1 + q _ { * } ( \lambda ) .
$$