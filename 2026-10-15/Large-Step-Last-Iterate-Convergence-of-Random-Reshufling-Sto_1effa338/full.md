# Large-Step Last-Iterate Convergence of Random-Reshufling Stochastic Gradient Descent

Zijun Chen   
Department of Computer Science and Engineering   
Hong Kong University of Science and Technology zchendg@connect.ust.hk Zihan Zhang<sup>∗</sup>   
Department of Computer Science and Engineering   
Hong Kong University of Science and Technology zsubfunc@outlook.com

## Abstract

Random reshufling stochastic gradient descent (RR-SGD) is widely used for finite-sum optimization in machine learning. For convex L-smooth components, recent work establishes convergence guarantees for an averaged iterate with component stepsizes as large as order 1/L (Liu, 2026). However, the final iterate is a natural output in practice, and existing last-iterate guarantees in this setting require stepsizes that scale inversely with the number of components n. We establish an expected objective-gap bound for the last iterate of standard RR-SGD on unconstrained problems with a constant component stepsize satisfying $0 < \eta \lesssim 1 / ( L ( 1 + \log ( n K ) ) )$ , where K is the number of epochs. By retaining the sampling bias at random comparison points, we remove the explicit $1 / n$ factor from the stepsize restriction. Combined with existing small-step bounds, our result matches the corresponding averaged-iterate convergence rate up to logarithmic factors.

## 1 Introduction

Finite-sum optimization is a basic problem in empirical risk minimization, where a learning objective averages losses over a fixed dataset. We consider

$$
\operatorname* { m i n } _ { \mathbf { x } \in \mathbb { R } ^ { d } } f ( \mathbf { x } ) , \qquad f ( x ) = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } f _ { i } ( \mathbf { x } ) ,\tag{1}
$$

with convex, L-smooth component functions and an attained minimum. When n is large, evaluating a single component gradient can be much cheaper than computing the full gradient. Stochastic gradient descent (SGD) exploits this structure by sampling a component independently at each update. Random reshufling (RR) instead visits every component once per epoch, drawing a fresh independent uniform permutation for each pass through the data. This sampling rule is common in training implementations and has motivated extensive work on optimization without replacement (Bottou, 2009; Shamir, 2016)

After a prescribed number of epochs, the final parameter is a natural output of the algorithm. A convergence bound for an average of earlier iterates does not, however, directly describe this output. For RR, the distinction is complicated by the dependence between the current iterate and the components remaining in an epoch: their gradients are generally biased estimates of the full gradient. Understanding the final iterate therefore requires accounting for the sampling rule throughout the optimization trajectory.

Recent work has established last-iterate guarantees for smooth convex shufling methods using component-gradient information at a minimizer (Liu and Zhou, 2024; Cai and Diakonikolas, 2025). In their plain-RR specializations, the stepsize multiplying each component gradient is restricted to order $1 / ( n L )$ , with an additional logarithmic factor. By contrast, Liu (2026) allows component stepsizes of order $1 / L$ under the same convex smoothness assumptions, but controls an averaged output. Its bound combines the usual SGD noise scale with a sharper term available from reshufling. The averaged-output result shows that the stepsize upper bound can depend only on the smoothness parameter $L ,$ independently of the number of components n. This naturally raises the question: Under the same convexity and smoothness assumptions, can we remove the explicit $1 / n$ factor from the stepsize restriction for last-iterate convergence?

## 1.1 Our Contribution

We answer this question in the afirmative, up to logarithmic factors. For a run of K epochs, we control the last iterate endpoint ${ \bf x } _ { K + 1 } ^ { 1 } { } ^ { 1 }$ . For smooth convex optimization (i.e., each $f _ { i }$ is convex and L-smooth), Theorem 1 states that RR-SGD with a constant stepsize η satisfies that

$$
\mathbb { E } [ f ( \mathbf { x } _ { K + 1 } ^ { 1 } ) - f ( \mathbf { x } _ { \star } ) ] \lesssim \frac { D ^ { 2 } } { \eta n K } + \operatorname* { m i n } \{ \eta \sigma _ { \star } ^ { 2 } , \eta ^ { 2 } n L \sigma _ { \star } ^ { 2 } \} .
$$

provided that the stepsize satisfies $\eta \leq c / ( L ( 1 + \log ( n K ) ) )$ for a universal constant $c > 0$ . Here, $\lesssim$ only hides the universal constants and logarithmic factors in $n K$ $D = \| { \bf x } _ { 1 } ^ { 1 } - { \bf x } _ { \star } \| { \bf \zeta } ( \| \cdot \| { \bf \zeta } \|$ is the $\ell _ { 2 } { \mathrm { - n o r m } } )$ is the distance from the initial point to a minimizer $\mathbf { x } _ { \star }$ , and $\sigma _ { * } ^ { 2 }$ is the gradient variance at the minimizer, which is invariant even when $f$ has multiple minimizers.

The first term, $D ^ { 2 } / ( \eta n K )$ , depends on the initial distance from the optimum and decreases with the number of updates for a fixed stepsize. The second term, min $\{ \eta \sigma _ { \star } ^ { 2 } , \eta ^ { 2 } n L \sigma _ { \star } ^ { 2 } \}$ , accounts for the uncertainty induced by shufling. The two expressions in this minimum describe the same stepsize regimes as the averaged-output bound of Liu (2026): $\eta ^ { 2 } n L \sigma _ { \star } ^ { 2 }$ is smaller when ηn $L \leq 1$ , whereas $\eta \sigma _ { \star } ^ { 2 }$ is smaller when $\eta n L \ge 1$ . For $D > 0$ , specializing to the epoch endpoint after $T = n K$ updates and choosing a constant stepsize using the horizon and problem parameters recovers the polynomial parameter dependence of the corresponding tuned averaged guarantee, up to logarithmic factors (Corollary 1). When $\sigma _ { \star } = 0$ , taking $\eta = c / [ L ( 1 + \log T ) ]$ yields an endpoint gap of ${ \widetilde O } ( L D ^ { 2 } / T )$ .

Technical contribution. The large-step analysis of Liu (2026) controls the sampling bias (see definition at eq. (3)) by comparing iterates with a fixed minimizer and summing over complete epochs. Extending this approach to the last iterate requires analyzing sufixes that may begin within an epoch. The starting iterate of each sufix serves as a random comparison point and may be correlated with the components remaining in that epoch. This dependence introduces an additional bias that must be controlled jointly with the sampling bias along the trajectory. To address this issue, our analysis combines two key ingredients:

• Signed sampling bias cancellation. We track the sampling bias contributions both along the trajectory and at the random comparison point used in each sufix. The contribution at the comparison point equals the negative of the sampling bias at the sufix’s starting position throughout the remainder of that epoch and vanishes in subsequent epochs. Retaining these signs through the sufix-to-last conversion reveals a partial cancellation between the two contributions. The resulting weights are nonnegative and satisfy a within-epoch summation bound, allowing us to apply a one-sided sampling-bias estimate and control repeated contributions from earlier updates without an additional factor of $n .$

Table 1: Comparison of expected objective-gap bounds for RR-SGD after $K \geq 2$ epochs with a constant component stepsize η. All results are stated for unconstrained finite-sum optimization with convex, L-smooth components and an attained minimum. Here $D = \| { \bf x } _ { 1 } ^ { 1 } - { \bf x } _ { * } \| , \ \sigma _ { * } ^ { 2 } =$ $\begin{array} { r } { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \| \nabla f _ { i } ( \mathbf { x } _ { * } ) \| ^ { 2 } , \ell _ { K } = 1 + \log K } \end{array}$ , and $\Lambda = 1 + \log ( n K )$ . We use a common smoothness bound $L ;$ the original results may retain finer smoothness parameters. $O ( \cdot )$ hides universal constants, while ${ \widetilde { O } } ( \cdot )$ additionally hides logarithmic factors. The average-iterate and last-iterate bounds control $\mathbb { E } [ f ( \bar { \mathbf { x } } ) - f ( \mathbf { x } _ { * } ) ]$ and $\mathbb { E } [ f ( \mathbf { x } _ { K + 1 } ^ { 1 } ) - f ( \mathbf { x } _ { * } ) ]$ , respectively, where $\bar { \bf x }$ denotes the averaged output used in the corresponding reference.
<table><tr><td>Stepsize</td><td>Reference</td><td>Rate</td><td>Output</td></tr><tr><td> $\eta \lesssim 1 / ( n L )$ </td><td>Cai et al. (2024)</td><td> $\begin{array} { r l r } {  { O \Big ( \frac { D ^ { 2 } } { \eta n K } + \eta ^ { 2 } n L \sigma _ { * } ^ { 2 } \Big ) } } \end{array}$ </td><td>Average iterate</td></tr><tr><td> $\eta \lesssim 1 / ( n L \sqrt { \ell _ { K } } )$ </td><td>Liu and Zhou (2024)</td><td> $\begin{array} { r } { \tilde { O } \left( \frac { D ^ { 2 } } { \eta n K } + \eta ^ { 2 } n L \sigma _ { * } ^ { 2 } \right) } \end{array}$ </td><td>Last iterate</td></tr><tr><td> $\eta \lesssim 1 / ( n L \sqrt { \ell _ { K } } )$ </td><td>Cai and Diakonikolas (2025)</td><td> $\begin{array} { r } { \tilde { O } \left( \frac { D ^ { 2 } } { \eta n K } + \eta ^ { 2 } n L \sigma _ { * } ^ { 2 } \right) } \end{array}$ </td><td>Last iterate</td></tr><tr><td> $\eta \lesssim 1 / L$ </td><td>Liu (2026)</td><td> $\begin{array} { r } { O \Big ( \frac { D ^ { 2 } } { \eta n K } + \sigma _ { * } ^ { 2 } \operatorname* { m i n } \{ \eta , \eta ^ { 2 } n L \} \Big ) } \end{array}$ </td><td>Average iterate</td></tr><tr><td> $\eta \lesssim 1 / ( L \Lambda )$ </td><td>This work (Theorem 1)¹</td><td> $\begin{array} { r } { \widetilde { \cal O } \left( \frac { D ^ { 2 } } { \eta n K } + \sigma _ { * } ^ { 2 } \operatorname* { m i n } \{ \eta , \eta ^ { 2 } n L \} \right) } \end{array}$ </td><td>Last iterate</td></tr></table>

<sup>1</sup> Theorem 1 additionally controls every prescribed component-time iterate, with nK replaced by $\tau ( k , i )$ in the initial-distance term.

• A gradient-moment recursion for last-iterate analysis. Building on the permutationswap tools of Liu (2026), we derive a self-bounding estimate that relates the sampled-gradient second moment to the current objective gap, the gradient variance at a minimizer, and earlier gradient moments within the same epoch. The history term carries a coeficient of order $\eta ^ { 2 } L ^ { 2 } / n$ , allowing its accumulated contribution under the sufix-to-last weights to be absorbed with only logarithmic overhead. Together with the signed sampling bias cancellation, this estimate yields a harmonic recursion in the objective gaps that preserves the inverse-time decay of the initial-error term and closes under $\eta L ( 1 + \log ( n K ) ) \lesssim 1$ , without an explicit inverse-n restriction on the stepsize.

## 2 Related Work

Table 1 compares the closest smooth convex RR guarantees, including their admissible component stepsizes and output choices.

Convergence of shufling methods. Early incremental analyses established convergence under cyclic and related sampling rules (Nedic and Bertsekas, 2001), while Recht and Ré (2012) connected without-replacement iterations to matrix-product inequalities. Under strong convexity, Gürbüzbalaban et al. (2021) explained asymptotic advantages of RR, and Haochen and Sra (2019); Rajput et al. (2020) obtained increasingly sharp finite-epoch guarantees. Related analyses characterize cyclic incremental methods (Gurbuzbalaban et al., 2019) and the steady-state behavior of constantstep reshufling (Ying et al., 2018). Ahn et al. (2020) weakened component assumptions under a Polyak–Łojasiewicz condition and removed large-epoch restrictions in a strongly convex result using varying steps.

Smooth convex objectives. For general smooth convex components, Nagaraj et al. (2019) obtained an averaged-output guarantee with component steps as large as order $1 / L$ , assuming bounded gradients on a bounded domain. Mishchenko et al. (2020); Nguyen et al. (2021) obtained averaged guarantees based on component gradients at a minimizer, with component steps of order $1 / ( n L )$ . Cai et al. (2024) refined the dependence on smoothness and data geometry through a primal-dual analysis, again for an averaged output. The large-step bound of Liu (2026) combines the SGD and reshufling noise regimes without a global variance assumption.

Last-iterate convergence. Liu and Zhou (2024) proved objective-gap guarantees for the last iterate of proximal shufling methods, including plain RR when the regularizer is zero. Cai and Diakonikolas (2025) established related endpoint guarantees for incremental gradient and proximal methods. Their smooth convex RR bounds use a stepsize ceiling of order $1 / ( n L { \sqrt { \log K } } )$ for $K \geq 2$ , with an explicit $1 / n$ factor. Liu and Zhou (2025) studied outputs at arbitrary component times for convex, Lipschitz components, a diferent assumption from smoothness with potentially unbounded gradients. Classical last-iterate SGD analyses (Shamir and Zhang, 2013; Harvey et al., 2019; Jain et al., 2019) supply useful techniques, but independent unbiased sampling avoids the within-epoch dependence present in RR. Attia et al. (2025) established last-iterate guarantees for without-replacement SGD with convex, L-smooth components under interpolation $( \sigma _ { * } ^ { 2 } = 0 )$ . Their without-replacement analysis is restricted to iterates within a single pass without replacement and does not cover nonzero noise at the optimum. In contrast, we establish last-iterate guarantees for multi-epoch random reshufling beyond interpolation, allowing $\sigma _ { * } ^ { 2 } > 0$ and capturing both the SGD-scale and reshufling-specific noise terms.

Lower bounds and epoch dependence. Safran and Shamir (2020, 2021) showed that the advantage of reshufling depends on the function class, conditioning, and number of epochs. For convex objectives, Cha et al. (2023) established a lower bound at the reshufling scale for suficiently many epochs, a restricted small-step range, and a global gradient-deviation parameter. That result does not settle optimality throughout the larger stepsize range considered here, or the necessity of our logarithmic factors. The small-epoch lower bounds of Kim et al. (2025) concern deterministic cyclic IGD and do not directly apply to fresh RR. Related comparisons between reshufling and reuse of a single permutation also depend on the problem class (Yun et al., 2021).

## 3 Preliminary

Notation. Let N denote the set of natural numbers. Given $n \in \mathbb { N }$ , we write $[ n ] : = \{ 1 , \dots , n \} , \langle \cdot , \cdot \rangle$ is the standard Euclidean inner product, and $\| \cdot \| : = \sqrt { \langle \cdot , \cdot \rangle }$ is the ℓ -norm. Given a real-valued diferentiable function $h : \mathbb { R } ^ { d }  \mathbb { R } , \nabla h ( \mathbf { x } )$ denotes the gradient at $\mathbf { x } \in \mathbb { R } ^ { d }$ . A permutation π of $[ n ]$ is a bijective mapping from [n] to itself. Let $S _ { n }$ denote the set of all permutations of [n]. We represent each permutation as $\pi = ( \pi ^ { 1 } , \ldots , \pi ^ { n } )$ , where $\pi ^ { i }$ denotes the index assigned to position i.

Objective. We consider the finite-sum optimization problem

$$
\operatorname* { m i n } _ { \mathbf { x } \in \mathbb { R } ^ { d } } f ( \mathbf { x } ) , \quad f ( \mathbf { x } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } f _ { i } ( \mathbf { x } ) , \quad \mathbf { x } \in \mathbb { R } ^ { d }
$$

and $n \in \mathbb { N }$ and for each $f _ { i } : \mathbb { R } ^ { d }  \mathbb { R }$ is convex diferentiable. Assume f attains its minimum. Fix $\mathbf { x } _ { * } \in$ arg $\mathrm { m i n } _ { \mathbf { x } \in \mathbb { R } ^ { d } } f ( \mathbf { x } )$ with a deterministic initialization ${ \bf x } _ { 1 } ^ { 1 }$

Our analysis relies on the following key Assumptions

Assumption 1 (Minimizer). There exists a minimizer $\mathbf { x } _ { * } \in \mathbb { R } ^ { d }$ such that $f ( \mathbf { x } _ { * } ) = \operatorname* { i n f } _ { \mathbf { x } \in \mathbb { R } ^ { d } } f ( \mathbf { x } )$

Assumption 2 (Convexity). For all $i \in [ n ] , f _ { i }$ is convex.

Assumption 3 (Smoothness). Each $f _ { i }$ is $L _ { i } { - } s m o o t h , \ i . e . , \ \forall \mathbf { x } , \mathbf { y } \in \mathbb { R } ^ { d } ;$

$$
\left\| \nabla f _ { i } ( \mathbf { x } ) - \nabla f _ { i } ( \mathbf { y } ) \right\| \leq L _ { i } \left\| \mathbf { x } - \mathbf { y } \right\| .
$$

Denote $L : = { \mathrm { m a x } } _ { i \in [ n ] } L _ { i }$ . We assume $L > 0$

All of the three assumptions are standard in the literature on stochastic optimization. We introduce two notations to simplify the expressions in the subsequent sections

$$
D : = \left\| \mathbf { x } _ { 1 } ^ { 1 } - \mathbf { x } _ { * } \right\| , \qquad \sigma _ { * } ^ { 2 } : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \| \nabla f _ { i } ( \mathbf { x } _ { * } ) \| ^ { 2 }
$$

We note that D is the distance between the initial point and the optimal solution. $\sigma _ { * }$ is commonly used to measure the noise level of the stochastic gradients at the optimum $\mathbf { x } _ { * }$ , which is invariant even when $f$ has multiple minimizers. In particular, $\sigma _ { * } ^ { 2 } = 0$ corresponds to the case in which all $f _ { i }$ have a same optimal solution.

## 4 RR-SGD and Main Result

Algorithm 1 Random-Reshufling Stochastic-Gradient-Descent (RR-SGD)   
1: input: initial point $\mathbf { x } _ { 1 } ^ { 1 } \in \mathbb { R } ^ { d }$ , stepsize $\{ \eta _ { k } > 0 \}$   
2: for $k = 1 , \ldots , K$ do   
3: Independently draw a permutation $\pi _ { k } \sim S _ { n }$   
4: for $i = 1 , \ldots , n$ do   
5: $\mathbf { x } _ { k } ^ { i + 1 } = \mathbf { x } _ { k } ^ { i } - \eta _ { k } \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } )$   
6: end for   
7: $\mathbf { x } _ { k + 1 } ^ { 1 } = \mathbf { x } _ { k } ^ { n + 1 }$   
8: end for

Algorithm 1 describes random reshufling stochastic gradient descent (RR-SGD). At the start of epoch k, RR-SGD draws a fresh uniform permutation $\pi _ { k } \in S _ { n }$ , independently of earlier epochs. It then visits the components in this order: at position $i ,$ it evaluates $\nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } )$ and takes a gradient step with stepsize $\eta _ { k }$ . Each component is therefore used exactly once per epoch. The endpoint of epoch k becomes the starting point of the next epoch, $\mathbf { x } _ { k + 1 } ^ { 1 } = \mathbf { x } _ { k } ^ { n + 1 }$ . We henceforth consider the constant stepsize $\eta _ { k } \equiv \eta$ for all $k \in [ K ]$ . The output of the algorithm is the final iterate ${ \bf x } _ { K + 1 } ^ { 1 }$ after K epochs.

## 4.1 Last-Iterate Convergence Guarantee

We now state our last-iterate guarantee. The stepsize is chosen before the run and remains constant across all component updates. Theorem 1 controls the iterate at any prescribed component time, including the endpoint ${ \bf x } _ { K + 1 } ^ { 1 }$ returned by Algorithm 1. Write

$$
\tau ( k , i ) = ( k - 1 ) n + i , \qquad i \in [ n ]
$$

so that $\mathbf { x } _ { k } ^ { i }$ is the iterate after $\tau ( k , i ) - 1$ updates.

Theorem 1. Let $K \geq 1$ and $\Lambda = 1 + \log ( n K )$ . Under Assumptions $\begin{array} { r } { 1 , \ 2 , } \end{array}$ and 3, run RR-SGD (Algorithm 1) with $\eta _ { k } \equiv \eta \ f o r \ k \in [ K ]$ , where $0 < \eta \leq 1 / ( 1 0 2 4 L \Lambda )$ . For every prescribed pair $( k , i )$ with $i \in [ n ]$ and $2 \leq \tau ( k , i ) \leq n K + 1$ , the iterate $\mathbf { x } _ { k } ^ { i }$ satisfies

$$
\begin{array} { r l } & { \mathbb { E } \left[ f ( \mathbf { x } _ { k } ^ { i } ) - f ( \mathbf { x } _ { * } ) \right] } \\ & { \quad \leq O \left( \frac { D ^ { 2 } } { \eta \tau ( k , i ) } + \Lambda ^ { 3 / 2 } \operatorname* { m i n } \left\{ \eta \sigma _ { * } ^ { 2 } , \eta ^ { 2 } n L \sigma _ { * } ^ { 2 } \right\} \right) . } \end{array}
$$

Admissible stepsize. Theorem 1 allows a constant component stepsize of order $1 / ( L \log ( n K ) )$ , without an explicit inverse dependence on the epoch length n. To the best of our knowledge, this is the first last-iterate guarantee for standard RR-SGD over any number of epochs that allows nonzero noise $\sigma _ { * } ^ { 2 } > 0$ and removes the inverse-n factor from the component-stepsize restriction. The existing small-step last-iterate guarantees discussed above impose a component stepsize ceiling of order $1 / ( n L )$ Liu and Zhou (2024). This distinction matters for the initial-error term $D ^ { 2 } / ( \eta \tau ( k , i ) )$ : a larger admissible stepsize allows this term to decrease more rapidly when the noise is suficiently small. The guarantee applies at every prescribed component time; the epoch endpoint is the case $( k , i ) = ( K + 1 , 1 )$

Two noise regimes. The minimum in the bound captures two complementary regimes. Since $\eta ^ { 2 } n L \sigma _ { \star } ^ { 2 } = ( \eta n L ) \eta \sigma _ { \star } ^ { 2 }$ , the reshufling term $\eta ^ { 2 } n L \sigma _ { \star } ^ { 2 }$ is smaller when ηn $L \leq 1$ , whereas the SGD-scale term $\eta \sigma _ { \star } ^ { 2 }$ is smaller when $\eta n L \ge 1$ . Thus, the theorem retains the sharper reshufling dependence for small stepsizes while extending last-iterate control to a substantially larger stepsize range. Our new analysis establishes the large-step estimate; the two-regime guarantee follows by combining it with an existing small-step endpoint bound and a within-epoch comparison (Lemma 8).

Corollary 1. Under the assumptions of Theorem 1, let $D > 0$ and $\Lambda = 1 + \log ( n K )$ . For $\sigma _ { * } > 0$ define

$$
\begin{array} { c } { { \eta _ { \mathrm { l i n } } : = \frac { D } { \Lambda ^ { 3 / 4 } \sigma _ { \ast } \sqrt { n K } } , } } \\ { { \eta _ { \mathrm { q u a d } } : = \left( \frac { D ^ { 2 } } { \Lambda ^ { 3 / 2 } n ^ { 2 } L K \sigma _ { \ast } ^ { 2 } } \right) ^ { 1 / 3 } , } } \end{array}
$$

and choose the constant stepsize

$$
\eta = \operatorname* { m i n } \left\{ \frac { 1 } { 1 0 2 4 L \Lambda } , \operatorname* { m a x } \{ \eta _ { \mathrm { l i n } } , \eta _ { \mathrm { q u a d } } \} \right\} .
$$

When $\sigma _ { * } = 0$ , set $\eta = 1 / ( 1 0 2 4 L \Lambda )$ . In either case, RR-SGD satisfies

$$
\begin{array} { r l } & { \mathbb { E } \left[ f ( \mathbf { x } _ { K + 1 } ^ { 1 } ) - f ( \mathbf { x } _ { * } ) \right] } \\ & { \leq \tilde { O } \left( \frac { L D ^ { 2 } } { n K } + \operatorname* { m i n } \left\{ \frac { \sigma _ { * } D } { \sqrt { n K } } , \left( \frac { L \sigma _ { * } ^ { 2 } D ^ { 4 } } { n K ^ { 2 } } \right) ^ { 1 / 3 } \right\} \right) } \end{array}\tag{2}
$$

The candidates $\eta _ { \mathrm { l i n } }$ and $\eta _ { \mathrm { q u a d } }$ balance the initial-distance term against the two expressions in the noise minimum in Theorem 1, respectively. Taking their maximum gives the smaller of the two balanced bounds; truncating the stepsize at $1 / ( 1 0 2 4 L \Lambda )$ adds at most a term of order $\Lambda L D ^ { 2 } / ( n K )$ Substitution into Theorem 1 therefore gives Corollary 1. Up to logarithmic factors, this recovers the polynomial parameter dependence of the corresponding averaged-output guarantee of Liu (2026). In particular, when $\sigma _ { \star } = 0$ , choosing the largest admissible stepsize gives ${ \tilde { O } } ( L D ^ { 2 } / T )$ after $T = n K$ component-gradient evaluations. The general guarantee also permits $\sigma _ { \star } > 0$ and requires no global bound on gradients or their variance.

## 5 Theoretical Proof

We prove Theorem 1 in four steps. First, we use permutation-swap stability to derive self-bounding estimates for the sampling bias and the sampled-gradient second moment in Section 5.1. Second, we retain the bias at random comparison points throughout a sufix-to-last conversion, obtaining weights that control repeated within-epoch contributions in Section 5.2. Third, we combine these estimates and close a harmonic-weight recursion to establish the large-step last-iterate guarantee in Section 5.3. Finally, we combine this guarantee with an existing small-step bound and a within-epoch comparison to obtain the two noise regimes in Theorem 1 in Section 5.4.

We define the auxiliary quantities

$$
\begin{array} { r l } & { \Delta _ { k } ^ { i } : = \mathbb { E } \left[ f ( \mathbf { x } _ { k } ^ { i } ) - f ( \mathbf { x } _ { * } ) \right] } \\ & { \beta _ { k } ^ { i } : = \mathbb { E } \left[ f ( \mathbf { x } _ { k } ^ { i } ) - f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) \right] } \end{array}\tag{3}
$$

Here, $\beta _ { k } ^ { i }$ is the sampling bias. The challenge is not just that $\beta _ { k } ^ { i }$ may be nonzero: sufix comparisons also introduce bias at an earlier random iterate. For a sufix starting at $\mathbf { x } _ { p } ^ { q } .$ , this comparator bias equals $- \beta _ { p } ^ { q }$ in epoch p and vanishes thereafter (Lemma 3). We combine the two signed contributions before bounding them. Their partial cancellation yields nonnegative bias weights with controlled within-epoch sums, allowing the one-sided estimate in eq. $( 7 )$ to be used in the last-iterate bound. Together with the sampled-gradient self-bound, this produces a recursion that closes under $\eta L ( 1 + \log ( n K ) ) \lesssim 1$ without requiring $\eta n L \lesssim 1$

## 5.1 Permutation Stability and Self-Bounding Estimates

The current iterate in RR-SGD depends on the components already visited in its epoch. Consequently, sampled function values and gradient moments cannot be directly replaced by their full averages. Following the arguments from Liu (2026), we control this dependence by comparing the original trajectory with a trajectory generated by swapping two positions of the permutation.

Given a permutation $\pi \in S _ { n }$ and two indices $i , j \in [ n ]$ satisfying the $j \leq i ,$ Let $\pi ( i , j ) : = \pi \circ ( i j )$ denote the permutation obtained by swapping the i-th and j-th entries of π.

In words, $\pi ( i , j )$ is the permutation generated by exchanging the elements. Equipped with the above notation, we can define the virtual iterate $\mathbf { x } _ { k } ^ { i } ( i , j )$ as the i-th iterate of the k-th epoch with the permutation $\pi _ { k } ( i , j )$ , i.e., for all $l \in [ i ]$

$$
\begin{array} { r } { \mathbf { x } _ { k } ^ { l + 1 } ( i , j ) = \mathbf { x } _ { k } ^ { l } ( i , j ) - \eta _ { k } \nabla f _ { \pi _ { k } ^ { l } ( i , j ) } ( \mathbf { x } _ { k } ^ { l } ( i , j ) ) } \end{array}\tag{4}
$$

where $\mathbf { x } _ { k } ^ { 1 } ( i , j ) : = \mathbf { x } _ { k } ^ { 1 }$

The two trajectories coincide before the first swapped position. After that position and before the second swapped update, they apply the same sequence of component-gradient maps. Nonexpansiveness of these maps therefore controls the discrepancy between the two trajectories.

Following the above observation and based on Lemma 3 in Liu (2026), we have the lemmas below to bound the drift term.

Lemma 1. Under Assumptions 2 and ${ \mathcal { B } } ,$ suppose $\eta \le 2 / L , \forall k \in [ K ]$ , then for any $k \in [ K ] , i \in [ n ]$ and $j \in [ i - 1 ]$ , RR-SGD (Algorithm 1) satisfies

$$
\left\| \mathbf { x } _ { k } ^ { i } ( i , j ) - \mathbf { x } _ { k } ^ { i } \right\| \leq \eta \left\| \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { j } ) - \nabla f _ { \pi _ { k } ^ { j } } ( \mathbf { x } _ { k } ^ { j } ) \right\|\tag{5}
$$

and

$$
\mathbb { E } \left[ \left\| \mathbf { x } _ { k } ^ { i } ( i , j ) - \mathbf { x } _ { k } ^ { i } \right\| ^ { 2 } \right] \leq 4 \eta ^ { 2 } \mathbb { E } \left[ \left\| \nabla f _ { \pi _ { k } ^ { j } } ( \mathbf { x } _ { k } ^ { j } ) \right\| ^ { 2 } \right]\tag{6}
$$

Equation (5) is the swap-stability bound of Lemma 3 in Liu (2026), and eq. (6) gives its secondmoment consequence. Crucially, the perturbation is controlled by the earlier update j, without an additional factor dependence on the separation $i - j$ . We retain this per-update structure rather than collapse it into an epoch-wide bound, allowing the sufix weights to control the history terms later.

Lemma 2 (Self-bounding estimates). When $0 \leq \eta \leq 1 / ( 2 L )$ , for any $k \in [ K ]$ and $i \in [ n ]$

$$
\begin{array} { r l } & { \mathbb { E } \left[ f ( \mathbf { x } _ { k } ^ { i } ) - f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) \right] \leq \displaystyle \frac { \eta } { 2 } \mathbb { E } \left[ \left\| \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) \right\| ^ { 2 } \right] } \\ & { \quad \quad \quad + \displaystyle \frac { 3 \eta } { n } \sum _ { j < i } \mathbb { E } \left[ \left\| \nabla f _ { \pi _ { k } ^ { j } } ( \mathbf { x } _ { k } ^ { j } ) \right\| ^ { 2 } \right] . } \end{array}\tag{7}
$$

When $\eta \leq 2 / L$ , then for any $k \in [ K ]$ and $i \in [ n ]$

$$
\begin{array} { r } { \mathbb { E } \left[ \left\| \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) \right\| ^ { 2 } \right] \leq 8 L \mathbb { E } \left[ f ( \mathbf { x } _ { k } ^ { i } ) - f ( \mathbf { x } _ { * } ) \right] } \\ { + 4 \sigma _ { * } ^ { 2 } + \frac { 8 \eta ^ { 2 } L ^ { 2 } } { n } \sum _ { j < i } \mathbb { E } \left[ \left\| \nabla f _ { \pi _ { k } ^ { j } } ( \mathbf { x } _ { k } ^ { j } ) \right\| ^ { 2 } \right] } \end{array}\tag{8}
$$

Equation (7) adapts the sampling bias estimate of Lemma 5 in Liu (2026). The additional ingredient is eq. (8): applying the permutation identity to squared gradient norms relates $Q _ { k } ^ { i }$ to the full component average, while retaining the history term $\begin{array} { r } { ( 8 \eta ^ { 2 } L ^ { 2 } / n ) \sum _ { j < i } Q _ { k } ^ { j } } \end{array}$ . The full average is controlled by $\Delta _ { k } ^ { i }$ and $\sigma _ { * } ^ { 2 }$ , without a global gradient bound. Crucially, the $\ddot { \eta } ^ { 2 } L ^ { 2 } / n$ scaling makes the history term compatible with last-iterate weighting: combining Lemma 5 bounds its total contribution by $1 6 \eta ^ { 2 } L ^ { 2 } \mathcal { H } _ { N } W _ { N }$ as in eq. (15). This term is absorbable under the theorem’s stepsize condition, closing the moment estimate without an inverse-n stepsize restriction.

## 5.2 Sufix-to-Last Conversion

At epoch index k and inner iteration index $i ,$ the corresponding global iteration index is $\tau ( k , i )$ . For a fixed $N \geq 2$ , let $( k _ { N } , i _ { N } )$ be the unique pair satisfying

$$
N = \tau ( k _ { N } , i _ { N } ) , \quad 1 \leq i _ { N } \leq n .
$$

We write $\Delta _ { N } : = \Delta _ { k _ { N } } ^ { i _ { N } }$ for the objective gap at global position N. Define the index set

$$
\begin{array} { r } { \textstyle \mathcal { T } _ { N } ( p , q ) : = \left\{ ( k , i ) : \tau ( p , q ) \leq \tau ( k , i ) \leq N \right\} , } \end{array}
$$

which contains all iteration pairs from $( p , q )$ through global iteration N, inclusive. Let $\mathcal { T } _ { N } : = \mathcal { T } _ { N } ( 1 , 1 )$ denote the set of iteration pairs corresponding to the first N steps. For each $( k , i ) \in \mathcal { T } _ { N }$ , let

$$
e _ { N , k } ^ { i } : = \operatorname* { m i n } { \{ n k , N \} }
$$

denote the global index of the last position of epoch k included in the horizon N. To record the preceding positions within epoch $k ,$ define

$$
\mathcal { P } _ { k } ^ { i } : = \{ ( k , j ) : 1 \leq j < i \} , \qquad k \geq 1 , \quad i \in [ n ] .
$$

In particular, $\mathcal { P } _ { k } ^ { 1 } = \emptyset$

Fix a sufix starting at $( p , q )$ and use $\mathbf { x } _ { p } ^ { q }$ as its comparison point. Within epoch $p ,$ all unrevealed positions have the same conditional marginal distribution. In later epochs, the fresh permutation is independent of $\mathbf { x } _ { p } ^ { q }$ . These observations give an exact expression for the comparator bias and the resulting sufix inequalities.

Lemma 3. For every $( p , q ) \in \mathcal { I } _ { N }$ , and every $( k , i ) \in \mathcal { I } _ { N } ( p , q )$ , we have

(9)

$$
\begin{array} { l } { { \mathbb { E } \left[ f _ { \tau _ { k } ^ { i } } ( \mathbf { x } _ { p } ^ { a } ) - f ( \mathbf { x } _ { p } ^ { a } ) \right] = - \beta _ { p } ^ { a } \mathbf { 1 } _ { \left\{ k = p \right\} } ; } } \\ { { \displaystyle \sum _ { ( k , i ) \in \mathbb { Z } \backslash \left( p , q \right) \atop ( k , i ) \in \mathbb { Z } \backslash \left( p , q \right) } ( \Delta _ { k } ^ { i } - \Delta _ { p } ^ { a } ) \leq - ( e _ { N , p } ^ { a } - \tau ( p , q ) + 1 ) \beta _ { p } ^ { a } } } \\ { { \displaystyle + \frac { \eta } { 2 } \sum _ { ( k , i ) \in \mathbb { Z } \backslash \left( p , q \right) \atop ( k , i ) \in \mathbb { Z } \backslash \left( p , q \right) } \mathbb { E } \left[ \left\| \nabla f _ { \tau _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) \right\| ^ { 2 } \right] + \sum _ { ( k , i ) \in \mathbb { Z } \backslash \left( p , q \right) } \beta _ { k } ^ { i } ; } } \\ { { \displaystyle \sum _ { ( k , i ) \in \mathbb { Z } \backslash \left( p , k \right) } \Delta _ { k } ^ { i } \leq \frac { D ^ { 2 } } { 2 \eta } + \sum _ { ( k , i ) \in \mathbb { Z } _ { N } } \beta _ { k } ^ { i } } } \\ { { \displaystyle + \frac { \eta } { 2 } \sum _ { ( k , i ) \in \mathbb { Z } \backslash \left( \mathbb { Z } \backslash \left( p , \tau _ { k } ^ { i } \right) \right) } \mathbb { E } \left[ \left\| \nabla f _ { \tau _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) \right\| ^ { 2 } \right] } } \end{array}\tag{10}
$$

(11)

We now apply the sufix-to-last identity to these inequalities. Here, the comparator bias refers to $\mathbb { E } [ f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { p } ^ { q } ) - f ( \mathbf { x } _ { p } ^ { q } ) ]$ , the discrepancy between the sampled component loss and the full objective at the random comparison point $\mathbf { x } _ { p } ^ { q }$ . By $\operatorname { e q . } \ ( 9 )$ , this quantity equals $- \beta _ { p } ^ { q } \mathbf { 1 } _ { \{ k = p \} }$ . We retain its sign rather than bound it separately. Combining its contribution with the trajectory sampling bias contribution yields the weights below.

Lemma 4. We have

$$
\begin{array} { r l } & { \displaystyle \Delta _ { N } \leq \frac { D ^ { 2 } } { 2 \eta N } + \frac { \eta } { 2 } \sum _ { ( k , i ) \in \mathcal { T } _ { N } } w _ { N , k } ^ { i } \mathbb { E } \left[ \left\| \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) \right\| ^ { 2 } \right] } \\ & { \quad \quad \quad + \displaystyle \sum _ { ( k , i ) \in \mathcal { T } _ { N } } v _ { N , k } ^ { i } \beta _ { k } ^ { i } } \end{array}\tag{12}
$$

where $v _ { N , k } ^ { i } , \mathrm { \Delta } w _ { N , k } ^ { i }$ are defined as

$$
\begin{array} { r l } & { w _ { N , k } ^ { i } : = \cfrac { 1 } { \operatorname* { m a x } \left\{ N - \tau ( k , i ) , 1 \right\} } } \\ & { v _ { N , k } ^ { i } : = \left\{ \cfrac { N - e _ { N , k } ^ { i } } { ( N - \tau ( k , i ) ) ( N - \tau ( k , i ) + 1 ) } , \ \tau ( k , i ) < N , \right. } \\ & { \left. \tau ( k , i ) = N . \right. } \end{array}
$$

The weights play diferent roles: $w _ { N , k } ^ { i }$ weights the sampled-gradient moments, whereas $v _ { N , k } ^ { i }$ is the residual coeficient of the signed sampling bias. The cancellation is exact at every nonterminal position in the final truncated epoch, where $v _ { N , k } ^ { i } = 0$ . Moreover, the residual weights are nonnegative and satisfy the terminal-inclusive block-sum bound established in Appendix B.5. These properties allow us to apply eq. (7) and absorb its within-epoch history terms into $W _ { N }$ , yielding Proposition 1.

Proposition 1. If $\eta L \leq 1 / 2$ , then for any $N \geq 2$

$$
\Delta _ { N } \leq \frac { D ^ { 2 } } { 2 \eta N } + 4 \eta \underbrace { \sum _ { ( k , i ) \in \mathbb { Z } _ { N } } w _ { N , k } ^ { i } \mathbb { E } \left[ \left\| \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) \right\| ^ { 2 } \right] } _ { : = W _ { N } }\tag{13}
$$

Proposition 1 reduces the last-iterate analysis to controlling a weighted sum of sampled-gradient second moments and the detailed proof is provided in the Appendix B.5. In eq. (13), the sampling bias no longer appears explicitly.

## 5.3 Closing the Recursion: The Large-Step Guarantee

We now control the weighted gradient moments in Proposition 1. Multiplying the sampled-gradient self-bound eq. (8) by $w _ { N , k } ^ { i } .$ , summing over the horizon, and applying the block-sum estimate eq. (15) gives

$$
\begin{array} { r l } & { W _ { N } \leq 8 L \underset { ( k , i ) \in \mathbb { Z } _ { N } } { \sum } w _ { N , k } ^ { i } \Delta _ { k } ^ { i } + 4 \sigma _ { * } ^ { 2 } \underset { ( k , i ) \in \mathbb { Z } _ { N } } { \sum } w _ { N , k } ^ { i } } \\ & { + 8 \eta ^ { 2 } L ^ { 2 } \underset { ( p , q ) \in \mathbb { Z } _ { N } } { \sum } \mathbb { E } \left[ \left\| \nabla f _ { \pi _ { p } ^ { q } } ( \mathbf { x } _ { p } ^ { q } ) \right\| ^ { 2 } \right] \left( \frac { 1 } { n } \underset { ( k , i ) \in \mathbb { Z } _ { N } } { \sum } w _ { N , k } ^ { i } \right) } \\ & { \qquad ( p , q ) \in \mathcal { P } _ { k } ^ { i } } \end{array}\tag{14}
$$

The weight of $\mathbb { E } [ \| \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) \| ^ { 2 } ]$ in eq. (14) can be controlled by the harmonic weights:

Lemma 5. For every $( p , q ) \in \mathcal { I } _ { N }$ , let $\begin{array} { r } { \mathcal { H } _ { N } : = \sum _ { l = 1 } ^ { N } \frac { 1 } { l } } \end{array}$ be the N-th harmonic number, then we have

$$
\frac { 1 } { n } \sum _ { ( k , i ) \in \mathbb { Z } _ { N } , ( p , q ) \in \mathcal { P } _ { k } ^ { i } } w _ { N , k } ^ { i } \leq 2 \mathcal { H } _ { N } w _ { N , p } ^ { q } .\tag{15}
$$

Plug eq. (15) into eq. (14), we have

$$
\begin{array} { r l } & { W _ { N } \leq 8 L \displaystyle \sum _ { ( k , i ) \in \mathbb { Z } _ { N } } w _ { N , k } ^ { i } \Delta _ { k } ^ { i } + 4 \sigma _ { * } ^ { 2 } ( \mathcal { H } _ { N - 1 } + 1 ) } \\ & { ~ + 1 6 \eta ^ { 2 } L ^ { 2 } \mathcal { H } _ { N } W _ { N } } \end{array}\tag{16}
$$

Under $\eta L \mathcal { H } _ { N } \leq 1 / 5 1 2$ , the last term can be absorbed into the left-hand side, yielding

$$
W _ { N } \leq 1 6 L \sum _ { ( k , i ) \in \mathcal { Z } _ { N } } w _ { N , k } ^ { i } \Delta _ { k } ^ { i } + 8 \sigma _ { * } ^ { 2 } ( \mathcal { H } _ { N - 1 } + 1 )\tag{17}
$$

Substituting eq. (17) into eq. (13), we obtain

$$
\Delta _ { N } \leq \frac { D ^ { 2 } } { 2 \eta N } + 6 4 \eta L \sum _ { ( k , i ) \in \mathbb { Z } _ { N } } w _ { N , k } ^ { i } \Delta _ { k } ^ { i } + 3 2 \eta \sigma _ { * } ^ { 2 } ( \mathcal { H } _ { N - 1 } + 1 )\tag{18}
$$

Thus, the weighted second moments $W _ { N }$ of the sampled gradients are controlled by the weighted error sequence and noise term. The remaining step is to close the recursion in the objective gaps. Since the terminal weight equals one,

$$
\sum _ { ( k , i ) \in \mathcal { I } _ { N } } w _ { N , k } ^ { i } \Delta _ { k } ^ { i } = \Delta _ { N } + \sum _ { ( k , i ) \in \mathcal { I } _ { N } , \tau ( k , i ) < N } \frac { \Delta _ { k } ^ { i } } { N - \tau ( k , i ) } .
$$

The identity $\smash { \sum _ { t = 1 } ^ { N - 1 } [ t ( N - t ) ] ^ { - 1 } = 2 H _ { N - 1 } / N }$ shows that sufix weighting preserves the $1 / N$ decay of the initial-error term up to a harmonic factor. After isolating the terminal gap in eq. (18), the induction closes under $\eta L H _ { N } \lesssim 1$ rather than ηn $L \lesssim 1$ . We then have the lemma below.

Lemma 6 (All-time last-iterate bound). For a fixed $N \geq 2$ , let $N = \tau ( k _ { N } , i _ { N } )$ and $\Delta _ { N } = \Delta _ { k _ { N } } ^ { i _ { N } }$ . If η $L \mathcal { H } _ { N }  \leq 1 / 5 1 2$ , then

$$
\Delta _ { N } \leq \frac { D ^ { 2 } } { \eta N } + 1 2 8 \eta \sigma _ { * } ^ { 2 } \mathcal { H } _ { N } .
$$

Reindex the horizon N by the epoch and inner-iteration indices $( k , i )$ , and recall $\tau ( k , i )$ denotes the corresponding global position. Since $\Lambda = 1 + \log ( n K )$ and $\mathcal { H } _ { n K + 1 } \leq 2 \Lambda$ , the stepsize condition $\eta \leq 1 / ( 1 0 2 4 L \Lambda )$ gives

$$
\mathbb { E } \left[ f ( \mathbf { x } _ { k } ^ { i } ) - f ( \mathbf { x } _ { * } ) \right] \lesssim \frac { D ^ { 2 } } { \eta \tau ( k , i ) } + \Lambda \eta \sigma _ { * } ^ { 2 } ,\tag{19}
$$

for every fixed $( k , i )$ with $i \in [ n ]$ and $2 \leq \tau ( k , i ) \leq n K + 1$ . At global position $n K + 1$ , we use the representation $( k , i ) = ( K + 1 , 1 )$ . The trajectory may be extended by a fresh auxiliary permutation for this next epoch; the comparison uses its first update only in the analysis and requires no additional gradient evaluation by the algorithm.

eq. (19) controls every component-time iterate with an SGD-scale noise term throughout the admissible stepsize range. For smaller stepsizes, however, a reshufling-specific quadratic noise term can be sharper. We combine the large-step estimate with the following existing small-step endpoint bound in Lemma $7$ and a within-epoch comparison in Lemma 8 to recover that dependence in Section 5.4.

## 5.4 Putting Together

The large-step estimate provides the noise term $\eta \sigma _ { \star } ^ { 2 }$ up to logarithmic factors. To obtain our results in both small- and large-step regimes, we combine the following last-iterate guarantee in previous work (Liu and Zhou, 2024).

Lemma 7 (Theorem C.1 in Liu and Zhou $\left( 2 0 2 4 \right) )$ . Under Assumptions $\begin{array} { r } { 1 , \ 2 , } \end{array}$ and 3, for any $m \geq 2$ $\displaystyle { \mathrm { : } f 0 < \eta \le 1 / \bigl ( 4 n L \sqrt { 2 ( 1 + \log m ) } \bigr ) }$ , the endpoint $\mathbf { x } _ { m + 1 } ^ { 1 }$ after m epochs of RR-SGD with constant stepsize η satisfies

$$
\mathbb { E } \left[ f ( \mathbf { x } _ { m + 1 } ^ { 1 } ) - f ( \mathbf { x } _ { * } ) \right] \lesssim \frac { D ^ { 2 } } { \eta n m } + \eta ^ { 2 } n L \sigma _ { * } ^ { 2 } \log m .
$$

The following comparison extends this endpoint guarantee to the iterates within the next epoch.

Lemma 8 (Within-epoch comparison). Under Assumptions 1, 2, and 3, suppose each epoch uses a fresh independent uniform permutation and $0 < \eta n L \leq 1 / 2$ . Then, for every epoch k and every $1 \leq i \leq n + 1$

$$
\Delta _ { k } ^ { i } \le 6 \Delta _ { k } ^ { 1 } + 8 \eta ^ { 2 } n L \sigma _ { * } ^ { 2 } .
$$

The proof is given in Appendix B.8. It controls the squared displacement from the epoch’s starting point using the second moment of a uniformly sampled prefix, then applies the smoothness of $f .$

Fix (k, i) with $i \in [ n ]$ and $2 \leq \tau ( k , i ) \leq n K + 1$ . We set threshold

$$
\eta _ { 0 } : = \frac { 1 } { 4 n L \sqrt { 2 ( 1 + \log K ) } } .
$$

Small stepsizes (Lemma $7 ) \colon \eta \le \eta _ { 0 }$ . For $k \geq 3 .$ , apply Lemma 7 with $m = k - 1 \leq K$ to the epoch’s starting point $\mathbf { x } _ { k } ^ { 1 }$ , then apply Lemma 8. Since $\tau ( k , i ) \leq 2 n ( k - 1 )$ and $1 + \log ( k - 1 ) \leq \Lambda$ this gives

$$
\Delta _ { k } ^ { i } \lesssim \frac { D ^ { 2 } } { \eta \tau ( k , i ) } + \Lambda \eta ^ { 2 } n L \sigma _ { * } ^ { 2 } .
$$

For the first two epochs where $k \leq 2$ , applying Lemma $8$ once or twice gives the same bound using $\Delta _ { 1 } ^ { 1 } \leq L D ^ { 2 } / 2 , N \leq 2 n$ , and $\eta n L \le 1 / 2$ . Thus the bound covers every admissible $( k , i )$ . Since ηn $L < 1$ and $\Lambda \geq 1$ , it implies the claimed $\Lambda ^ { 3 / 2 }$ min $\{ \eta \sigma _ { \ast } ^ { 2 } , \eta ^ { 2 } n L \sigma _ { \ast } ^ { 2 } \}$ term.

Intermediate stepsizes (Lemma ${ \bf 6 } ) \colon \eta > \eta _ { 0 }$ and ηn $L < 1$ . In this regime,

$$
\frac { 1 } { \eta n L } < 4 \sqrt { 2 ( 1 + \log K ) } \leq 4 \sqrt { 2 \Lambda } .
$$

consequently, the noise term in the all-time large-step estimate eq. (19) satisfies

$$
\Lambda \eta \sigma _ { \star } ^ { 2 } = \frac { \Lambda } { \eta n L } \eta ^ { 2 } n L \sigma _ { \star } ^ { 2 } \lesssim \Lambda ^ { 3 / 2 } \eta ^ { 2 } n L \sigma _ { \star } ^ { 2 } .
$$

Thus, the large-step estimate also yields the desired reshufling noise scale in the intermediate regime, with an additional logarithmic factor.

Larger stepsizes (Lemma 6): ηn $L \geq 1$ . Here,

$$
\operatorname* { m i n } \{ \eta \sigma _ { \star } ^ { 2 } , \eta ^ { 2 } n L \sigma _ { \star } ^ { 2 } \} = \eta \sigma _ { \star } ^ { 2 } ,
$$

so eq. (19) gives the required bound directly.

Combining the three regimes proves Theorem 1 for every fixed component-time iterate in the stated horizon. Taking $( k , i ) = ( K + 1 , 1 )$ recovers the returned endpoint. When $\sigma _ { * } ^ { 2 } = 0$ , the largest admissible stepsize gives $\tilde { O } ( L D ^ { 2 } / \tau ( k , i ) )$ , and hence ${ \cal \tilde { O } } ( L D ^ { 2 } / ( n K ) )$ at the endpoint.

## 6 Discussion

For convex, smooth components, we establish expected last-iterate guarantees for RR-SGD with constant component stepsizes up to order $1 / ( L \log ( n K ) )$ ), removing the explicit $1 / n$ factor from the stepsize restriction. The result applies at every prescribed component time, and the tuned endpoint rate matches the averaged-output rate up to logarithmic factors. Our analysis retains the signed bias at random comparison points and exploits its partial cancellation with the trajectory bias. The resulting weights, together with a sampled-gradient second-moment bound, allow the harmonic recursion to close without an extra factor of n.

An immediate question for future work is whether the logarithmic losses can be removed, permitting component stepsizes of order $1 / L$ . Non-constant stepsizes may also improve last-iterate convergence rates by using larger early steps to reduce the initial error and smaller later steps to control the noise contribution. Finally, retaining the individual smoothness constants $L _ { i }$ in the drif estimates may clarify which error terms can depend on an average of these constants rather than their maximum.

## References

Ahn, K., Yun, C., and Sra, S. (2020). Sgd with shufling: optimal rates without component convexity and large epoch requirements. Advances in Neural Information Processing Systems, 33:17526–17535.

Attia, A., Schliserman, M., Sherman, U., and Koren, T. (2025). Fast last-iterate convergence of sgd in the smooth interpolation regime.

Bottou, L. (2009). Curiously fast convergence of some stochastic gradient descent algorithms. Unpublished open problem ofered to the attendance of the SLDS 2009 conference.

Cai, X. and Diakonikolas, J. (2025). Last iterate convergence of incremental methods as a model of forgetting. In The Thirteenth International Conference on Learning Representations.

Cai, X., Lin, C. Y., and Diakonikolas, J. (2024). Tighter convergence bounds for shufled sgd via primal-dual perspective. Advances in Neural Information Processing Systems, 37:72475–72524.

Cha, J., Lee, J., and Yun, C. (2023). Tighter lower bounds for shufling sgd: Random permutations and beyond. In International Conference on Machine Learning, pages 3855–3912. PMLR.

Gurbuzbalaban, M., Ozdaglar, A., and Parrilo, P. A. (2019). Convergence rate of incremental gradient and incremental newton methods. SIAM Journal on Optimization, 29(4):2542–2565.

Gürbüzbalaban, M., Ozdaglar, A., and Parrilo, P. A. (2021). Why random reshufling beats stochastic gradient descent. Mathematical Programming, 186(1):49–84.

Haochen, J. and Sra, S. (2019). Random shufling beats sgd after finite epochs. In International Conference on Machine Learning, pages 2624–2633. PMLR.

Harvey, N. J., Liaw, C., Plan, Y., and Randhawa, S. (2019). Tight analyses for non-smooth stochastic gradient descent. In Conference on Learning Theory, pages 1579–1613. PMLR.

Jain, P., Nagaraj, D., and Netrapalli, P. (2019). Making the last iterate of sgd information theoretically optimal. In Conference on Learning Theory, pages 1752–1755. PMLR.

Kim, Y., Cha, J., and Yun, C. (2025). Incremental gradient descent with small epoch counts is surprisingly slow on ill-conditioned problems. arXiv preprint arXiv:2506.04126.

Liu, Z. (2026). Random reshufling dominates stochastic gradient descent. arXiv preprint arXiv:2606.32005.

Liu, Z. and Zhou, Z. (2024). On the last-iterate convergence of shufling gradient methods. arXiv preprint arXiv:2403.07723.

Liu, Z. and Zhou, Z. (2025). Improved last-iterate convergence of shufling gradient methods for nonsmooth convex optimization. arXiv preprint arXiv:2505.23056.

Mishchenko, K., Khaled, A., and Richtárik, P. (2020). Random reshufling: Simple analysis with vast improvements. Advances in neural information processing systems, 33:17309–17320.

Nagaraj, D., Jain, P., and Netrapalli, P. (2019). Sgd without replacement: Sharper rates for general smooth convex functions. In International Conference on Machine Learning, pages 4703–4711. PMLR.

Nedic, A. and Bertsekas, D. P. (2001). Incremental subgradient methods for nondiferentiable optimization. SIAM Journal on Optimization, 12(1):109–138.

Nguyen, L. M., Tran-Dinh, Q., Phan, D. T., Nguyen, P. H., and Van Dijk, M. (2021). A unified convergence analysis for shufling-type gradient methods. Journal of Machine Learning Research, 22(207):1–44.

Rajput, S., Gupta, A., and Papailiopoulos, D. (2020). Closing the convergence gap of sgd without replacement. In International Conference on Machine Learning, pages 7964–7973. PMLR.

Recht, B. and Ré, C. (2012). Toward a noncommutative arithmetic-geometric mean inequality: Conjectures, case-studies, and consequences. In Conference on Learning Theory, pages 11.1–11.24. JMLR Workshop and Conference Proceedings.

Safran, I. and Shamir, O. (2020). How good is sgd with random shufling? In Conference on Learning Theory, pages 3250–3284. PMLR.

Safran, I. and Shamir, O. (2021). Random shufling beats sgd only after many epochs on illconditioned problems. Advances in Neural Information Processing Systems, 34:15151–15161.

Shamir, O. (2016). Without-replacement sampling for stochastic gradient methods. Advances in neural information processing systems, 29.

Shamir, O. and Zhang, T. (2013). Stochastic gradient descent for non-smooth optimization: Convergence results and optimal averaging schemes. In International conference on machine learning, pages 71–79. PMLR.

Ying, B., Yuan, K., Vlaski, S., and Sayed, A. H. (2018). Stochastic learning under random reshufling with constant step-sizes. IEEE Transactions on Signal Processing, 67(2):474–489.

Yun, C., Sra, S., and Jadbabaie, A. (2021). Open problem: Can single-shufle sgd be better than reshufling sgd and gd? In Conference on Learning Theory, pages 4653–4658. PMLR.

## Appendices

## A Notation and Technical Lemmas

## A.1 Notation Tables

For readability, we recall and summarize the notation used in the paper and define the auxiliary quantities used in the proof.

$f _ { i } { \mathrm { : } }$ the i-th component function of the finite-sum optimization problem, $f _ { i }$ is convex and diferentiable.

$f \colon$ the optimization objective, $\begin{array} { r } { f ( \mathbf { x } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } f _ { i } ( \mathbf { x } ) } \end{array}$

$D ;$ the distance between the initial point and the minimizer, $D : = \| \mathbf { x } _ { 1 } ^ { 1 } - \mathbf { x } _ { * } \|$

• $\mathbf { x } _ { k } ^ { i } \mathrm { . }$ : the iteration point at position i of epoch k.

$\eta \colon$ the stepsize.

• $\sigma _ { * } ^ { 2 } \colon$ : the variance of the gradients at the minimizer, $\begin{array} { r } { \sigma _ { * } ^ { 2 } : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \| \nabla f _ { i } ( \mathbf { x } _ { * } ) \| ^ { 2 } } \end{array}$

$k , i \colon$ the epoch index and the within-epoch position, respectively.

$\tau ( k , i )$ : the global iteration index, $\tau ( k , i ) : = ( k - 1 ) n + i .$

$\pi _ { k } ^ { i } \colon$ the component index sampled at position i of epoch k.

$\Delta _ { k } ^ { i } \mathrm { : }$ the expected suboptimality at position i of epoch k,

$$
\Delta _ { k } ^ { i } : = \mathbb { E } \left[ f ( \mathbf { x } _ { k } ^ { i } ) - f ( \mathbf { x } _ { * } ) \right]
$$

• $Q _ { k } ^ { i }$ : the expected squared norm of the gradient at position i of epoch $k ,$

$$
Q _ { k } ^ { i } : = \mathbb { E } \left[ \left\| \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) \right\| ^ { 2 } \right]
$$

• $\beta _ { k } ^ { i }$ : the expected bias of the component function at position i of epoch $k ,$

$$
\beta _ { k } ^ { i } : = \mathbb { E } \left[ f ( \mathbf { x } _ { k } ^ { i } ) - f _ { \pi _ { k } ( i ) } ( \mathbf { x } _ { k } ^ { i } ) \right]
$$

• $M _ { k } ^ { i } \mathrm { . }$ : the expected average squared norm of the gradients at position i of epoch $k$

$$
M _ { k } ^ { i } : = \mathbb { E } \left[ \frac { 1 } { n } \sum _ { r = 1 } ^ { n } \left\| \nabla f _ { r } ( \mathbf { x } _ { k } ^ { i } ) \right\| ^ { 2 } \right]
$$

• $W _ { N } ;$ : the weighted sum of the expected squared norm of the gradients up to iteration $N _ { ; }$

$$
W _ { N } : = \sum _ { ( k , i ) \in \mathcal { T } _ { N } } w _ { N , k } ^ { i } Q _ { k } ^ { i }
$$

• $V _ { N } \mathbf { : }$ : the weighted sum of the expected suboptimality up to iteration $N _ { ; }$

$$
V _ { N } : = \sum _ { ( k , i ) \in \mathbb { Z } _ { N } } w _ { N , k } ^ { i } \Delta _ { k } ^ { i }
$$

$\mathcal { H } _ { N } \mathrm { : }$ : the N-th harmonic number, $\begin{array} { r } { \mathcal { H } _ { N } : = \sum _ { l = 1 } ^ { N } \frac { 1 } { l } } \end{array}$

For $j < i ,$ the swapped permutation $\pi ( i , j )$ is defined componentwise by

$$
\pi ^ { r } ( i , j ) = \left\{ { \pi ^ { i } , \quad r = j , } \atop { \pi ^ { j } , \quad r \not \in \{ i , j \} , } \right. \qquad r \in [ n ] .
$$

For $j = i , \pi ( i , i ) = \pi$

## A.2 Technical Lemmas

The proofs of our main Theorem and Lemmas in the main text involve several technical lemmas, which are summarized below.

Lemma 9 (Co-coercivity, Lemma 1 in Liu (2026)). Let $h : \mathbb { R } ^ { d }  \mathbb { R }$ be a diferentiable convex function that is also L-smooth, then we have, for any $\mathbf { x } , \mathbf { y } \in \mathbb { R } ^ { d }$

$$
\| \nabla h ( \mathbf { x } ) - \nabla h ( \mathbf { y } ) \| ^ { 2 } \leq 2 L \left( h ( \mathbf { x } ) - h ( \mathbf { y } ) - \langle \nabla h ( \mathbf { y } ) , \mathbf { x } - \mathbf { y } \rangle \right)
$$

Lemma 10 (Lemma 2 in Liu (2026)). Given an arbitrary finite-sum function $\begin{array} { r } { \ell ( \mathbf { x } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \ell _ { i } ( \mathbf { x } ) } \end{array}$ suppose RR-SGD is employed, then for all $k \in [ K ]$ and $i \in [ n ]$ , we have

$$
\mathbb { E } \left[ \ell ( { \mathbf x } _ { k } ^ { i } ) - \ell _ { \pi _ { k } ^ { i } } ( { \mathbf x } _ { k } ^ { i } ) \right] = \frac { 1 } { n } \sum _ { j < i } \mathbb { E } \left[ \ell _ { \pi _ { k } ^ { i } } ( { \mathbf x } _ { k } ^ { i } ( i , j ) ) - \ell _ { \pi _ { k } ^ { i } } ( { \mathbf x } _ { k } ^ { i } ) \right]
$$

where $\mathbf { x } _ { k } ^ { i } ( i , j )$ is defined in eq. (4).

Lemma 11 (Non-expansiveness and the best-point self-boundary). For any convex, diferentiable, and L-smooth function $f : \mathbb { R } ^ { d }  \mathbb { R }$ , any $\mathbf { x } , \mathbf { y } \in \mathbb { R } ^ { d }$ , and $0 \leq \eta \leq 2 / L$

$$
\left\| \left( \mathbf { x } - \eta \nabla f ( \mathbf { x } ) \right) - \left( \mathbf { y } - \eta \nabla f ( \mathbf { y } ) \right) \right\| \leq \left\| \mathbf { x } - \mathbf { y } \right\|\tag{20}
$$

Moreover, under Assumptions $\begin{array} { r } { 1 , \ 2 , } \end{array}$ and ${ \mathcal { B } } ,$ the finite-sum objective $\begin{array} { r } { f = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } f _ { i } } \end{array}$ satisfies, for every $\mathbf { x } \in \mathbb { R } ^ { d }$

$$
\frac { 1 } { n } \sum _ { i = 1 } ^ { n } \| \nabla f _ { i } ( { \bf x } ) \| ^ { 2 } \le 4 L \left( f ( { \bf x } ) - f ( { \bf x } _ { * } ) \right) + 2 \sigma _ { * } ^ { 2 }\tag{21}
$$

Proof. First, for a fixed ${ \mathbf { y } } ,$ define

$$
h _ { \mathbf { y } } ( \mathbf { x } ) : = f ( \mathbf { x } ) - f ( \mathbf { y } ) - \langle \nabla f ( \mathbf { y } ) , \mathbf { x } - \mathbf { y } \rangle
$$

By convexity, $h _ { \mathbf { y } } ( \mathbf { x } ) \geq 0$ , and $h _ { \mathbf { y } }$ is L-smooth. we have

$$
0 \leq h _ { \mathbf { y } } \left( \mathbf { x } - \frac { 1 } { L } \nabla h _ { \mathbf { y } } ( \mathbf { x } ) \right) \leq h _ { \mathbf { y } } ( \mathbf { x } ) - \frac { 1 } { 2 L } \left. \nabla h _ { \mathbf { y } } ( \mathbf { x } ) \right. ^ { 2 }
$$

Thus,

$$
\| \nabla f ( \mathbf { x } ) - \nabla f ( \mathbf { y } ) \| ^ { 2 } \leq 2 L \left( f ( \mathbf { x } ) - f ( \mathbf { y } ) - \langle \nabla f ( \mathbf { y } ) , \mathbf { x } - \mathbf { y } \rangle \right)
$$

switch x and y, we have

$$
\| \nabla f ( \mathbf { y } ) - \nabla f ( \mathbf { x } ) \| ^ { 2 } \leq 2 L \left( f ( \mathbf { y } ) - f ( \mathbf { x } ) - \langle \nabla f ( \mathbf { x } ) , \mathbf { y } - \mathbf { x } \rangle \right)
$$

Summing the two inequalities, we have

$$
{ \frac { 1 } { L } } \left\| \nabla f ( \mathbf { x } ) - \nabla f ( \mathbf { y } ) \right\| ^ { 2 } \leq \left. \nabla f ( \mathbf { x } ) - \nabla f ( \mathbf { y } ) , \mathbf { x } - \mathbf { y } \right.
$$

Finally,

$$
\begin{array} { r l } { \displaystyle \| ( \mathbf x - \eta \nabla f ( \mathbf x ) ) - ( \mathbf y - \eta \nabla f ( \mathbf y ) ) \| ^ { 2 } = \| \mathbf x - \mathbf y \| ^ { 2 } - 2 \eta \langle \nabla f ( \mathbf x ) - \nabla f ( \mathbf y ) , \mathbf x - \mathbf y \rangle } & { } \\ { + \eta ^ { 2 } \left\| \nabla f ( \mathbf x ) - \nabla f ( \mathbf y ) \right\| ^ { 2 } } & { } \\ { \le \left\| \mathbf x - \mathbf y \right\| ^ { 2 } - \eta \left( \displaystyle \frac { 2 } { L } - \eta \right) \left\| \nabla f ( \mathbf x ) - \nabla f ( \mathbf y ) \right\| ^ { 2 } } & { } \\ { \le \left\| \mathbf x - \mathbf y \right\| ^ { 2 } } & { } \end{array}
$$

where the last inequality holds when $0 \leq \eta \leq 2 / L$ , then we complete the proof of eq. (20).

Proof of eq. (21). Let

$$
h _ { i } ( { \bf x } ) = f _ { i } ( { \bf x } ) - f _ { i } ( { \bf x _ { * } } ) - \left. \nabla f _ { i } ( { \bf x _ { * } } ) , { \bf x } - { \bf x _ { * } } \right.
$$

By the squared triangle inequality and co-coercivity of each component, we have

$$
\begin{array} { r l r } {  { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \| \nabla f _ { i } ( { \mathbf x } ) \| ^ { 2 } \le \displaystyle \frac { 2 } { n } \sum _ { i = 1 } ^ { n } \| \nabla f _ { i } ( { \mathbf x } ) - \nabla f _ { i } ( { \mathbf x } _ { * } ) \| ^ { 2 } + \frac { 2 } { n } \sum _ { i = 1 } ^ { n } \| \nabla f _ { i } ( { \mathbf x } _ { * } ) \| ^ { 2 } } } \\ & { } & { \le \displaystyle \frac { 4 L } { n } \sum _ { i = 1 } ^ { n } ( f _ { i } ( { \mathbf x } ) - f _ { i } ( { \mathbf x } _ { * } ) - \langle \nabla f _ { i } ( { \mathbf x } _ { * } ) , { \mathbf x } - { \mathbf x } _ { * } \rangle ) + 2 \sigma _ { * } ^ { 2 } } \\ & { } & { = 4 L ( f ( { \mathbf x } ) - f ( { \mathbf x } _ { * } ) ) + 2 \sigma _ { * } ^ { 2 } . } \end{array}
$$

The last equality uses $\begin{array} { r } { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \nabla f _ { i } ( \mathbf { x } _ { * } ) = \nabla f ( \mathbf { x } _ { * } ) = \mathbf { 0 } } \end{array}$ , since x<sub>∗</sub> minimizes $f .$ This proves eq. (21).

## B Missing Proofs

## B.1 Proof of Lemma 1

The proof of eq. (5) is exactly the same as the proof of Lemma 3 in Liu (2026). We provide the proof of eq. (6) here. The original and swapped trajectories use the same component indices before position $j ,$ , so

$$
\mathbf { x } _ { k } ^ { j } ( i , j ) = \mathbf { x } _ { k } ^ { j }
$$

The trajectories may difer after the j-th update. By eq. (4), their diference is

$$
\mathbf { x } _ { k } ^ { j + 1 } ( i , j ) - \mathbf { x } _ { k } ^ { j + 1 } = - \eta \left[ \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { j } ) - \nabla f _ { \pi _ { k } ^ { j } } ( \mathbf { x } _ { k } ^ { j } ) \right]
$$

For $j + 1 \le l \le i - 1$ , the two trajectories apply the same component-gradient map. Its non expansiveness gives

$$
\left\| \mathbf { x } _ { k } ^ { l + 1 } ( i , j ) - \mathbf { x } _ { k } ^ { l + 1 } \right\| \leq \left\| \mathbf { x } _ { k } ^ { l } ( i , j ) - \mathbf { x } _ { k } ^ { l } \right\| .
$$

Iterating this inequality up to position $i ,$ before the second swapped update, yields

$$
\begin{array} { r l } { \left\| \mathbf { x } _ { k } ^ { i } ( i , j ) - \mathbf { x } _ { k } ^ { i } \right\| \leq \left\| \mathbf { x } _ { k } ^ { j + 1 } ( i , j ) - \mathbf { x } _ { k } ^ { j + 1 } \right\| } & { } \\ { = \eta \left\| \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { j } ) - \nabla f _ { \pi _ { k } ^ { j } } ( \mathbf { x } _ { k } ^ { j } ) \right\| } & { } \end{array}
$$

When $i = j + 1$ , the same bound follows directly from the diference after the j-th update, with no intervening updates. Suppose at k-th epoch and j-th inner iteration, the filtration $\mathcal { F } _ { k } ^ { j }$ :

$$
\mathcal { F } _ { k } ^ { j } : = \sigma \left( \pi _ { 1 } , \ldots , \pi _ { k - 1 } , \pi _ { k } ^ { 1 } , \ldots , \pi _ { k } ^ { j - 1 } \right) .
$$

For a fixed $\mathcal { F } _ { k } ^ { j } ,$ , the iteration point $\mathbf { x } _ { k } ^ { j }$ is fixed, and $\pi _ { k } ^ { j } , \pi _ { k } ^ { i } \in [ n ] \backslash \{ \pi _ { k } ^ { 1 } , . . . , \pi _ { k } ^ { j - 1 } \}$ admits the uniform distribution. Thus, we have

$$
\mathbb { E } \left[ \left. \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { j } ) \right. ^ { 2 } | \mathcal { F } _ { k } ^ { j } \right] = \frac { 1 } { n - j + 1 } \sum _ { \substack { s \in [ n ] \backslash \{ \pi _ { k } ^ { 1 } , \ldots , \pi _ { k } ^ { j - 1 } \} } } \left. \nabla f _ { s } ( \mathbf { x } _ { k } ^ { j } ) \right. ^ { 2 } = \mathbb { E } \left[ \left. \nabla f _ { \pi _ { k } ^ { j } } ( \mathbf { x } _ { k } ^ { j } ) \right. ^ { 2 } | \mathcal { F } _ { k } ^ { j } \right]
$$

Finally, taking the total expectation, we have

$$
\begin{array} { r } { \mathbb { E } \left[ \left\| \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { j } ) \right\| ^ { 2 } \right] = \mathbb { E } \left[ \left\| \nabla f _ { \pi _ { k } ^ { j } } ( \mathbf { x } _ { k } ^ { j } ) \right\| ^ { 2 } \right] = Q _ { k } ^ { j } } \end{array}
$$

Here we only relies on the uniform distribution of the permutation $\pi _ { k }$ , without requiring the independence of $\pi _ { k } ^ { i }$ and $\pi _ { k } ^ { j }$ . Thus, we have

$$
\begin{array} { r l } { \mathbb { E } \left[ \left. \mathbf { x } _ { k } ^ { i } ( i , j ) - \mathbf { x } _ { k } ^ { i } \right. ^ { 2 } \right] \leq \eta ^ { 2 } \mathbb { E } \left[ \left. \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { j } ) - \nabla f _ { \pi _ { k } ^ { j } } ( \mathbf { x } _ { k } ^ { j } ) \right. ^ { 2 } \right] } & { } \\ { \leq 2 \eta ^ { 2 } \mathbb { E } \left[ \left. \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { j } ) \right. ^ { 2 } + \left. \nabla f _ { \pi _ { k } ^ { j } } ( \mathbf { x } _ { k } ^ { j } ) \right. ^ { 2 } \right] } & { } \\ { = 4 \eta ^ { 2 } Q _ { k } ^ { j } } \end{array}
$$

which completes the proof of eq. (6).

## B.2 Proof of Lemma 2

Proof of eq. (7). If $\eta = 0$ , all iterates equal the deterministic initialization ${ \bf x } _ { 1 } ^ { 1 }$ . Since each $\pi _ { k } ^ { i }$ is marginally uniform on [n], we have $\beta _ { k } ^ { i } = 0$ , and eq. (7) holds immediately. For the remainder of this argument, assume $\eta > 0$ . Denote $\delta : = \mathbf { x } _ { k } ^ { i } ( i , j ) - \mathbf { x } _ { k } ^ { i }$ . By smoothness and Young’s inequality, we have

$$
\begin{array} { l } { f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } + \delta ) - f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) \leq \langle \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) , \delta \rangle + \displaystyle \frac { L } { 2 } \left\| \delta \right\| ^ { 2 } } \\ { \displaystyle \leq \displaystyle \frac { \eta } { 2 } \left\| \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) \right\| ^ { 2 } + \displaystyle \frac { 1 } { 2 \eta } \left\| \delta \right\| ^ { 2 } + \displaystyle \frac { L } { 2 } \left\| \delta \right\| ^ { 2 } } \\ { \displaystyle = \displaystyle \frac { \eta } { 2 } \left\| \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) \right\| ^ { 2 } + \left( \displaystyle \frac { 1 } { 2 \eta } + \displaystyle \frac { L } { 2 } \right) \left\| \delta \right\| ^ { 2 } } \end{array}
$$

Taking expection on the both sides, combine eq. (6) in Lemma 1, we have

$$
\mathbb { E } \left[ f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ( i , j ) ) - f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) \right] \leq \frac { \eta } { 2 } Q _ { k } ^ { i } + \left( \frac { 1 } { 2 \eta } + \frac { L } { 2 } \right) \mathbb { E } \left[ \left. \delta \right. ^ { 2 } \right] \leq \frac { \eta } { 2 } Q _ { k } ^ { i } + ( 2 \eta + 2 \eta ^ { 2 } L ) \mathbb { E } \left[ \left. \nabla f _ { \pi _ { k } ^ { j } } ( \mathbf { x } _ { k } ^ { j } ) \right. ^ { 2 } \right]
$$

Apply Lemma 10 with $l _ { i } = f _ { i }$

$$
\begin{array} { r l } { \displaystyle \mathbb { E } \left[ f ( \mathbf { x } _ { k } ^ { i } ) - f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) \right] = \frac { 1 } { n } \sum _ { j < i } \mathbb { E } \left[ f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ( i , j ) ) - f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) \right] } & { } \\ { \displaystyle } & { \leq \frac { 1 } { n } \sum _ { j < i } \left[ \frac { \eta } { 2 } Q _ { k } ^ { i } + ( 2 \eta + 2 \eta ^ { 2 } L ) \mathbb { E } \left[ \left\| \nabla f _ { \pi _ { k } ^ { j } } ( \mathbf { x } _ { k } ^ { j } ) \right\| ^ { 2 } \right] \right] } \\ & { \displaystyle \leq \frac { \eta } { 2 } Q _ { k } ^ { i } + \frac { 3 \eta } { n } \sum _ { j < i } \mathbb { E } \left[ \left\| \nabla f _ { \pi _ { k } ^ { j } } ( \mathbf { x } _ { k } ^ { j } ) \right\| ^ { 2 } \right] } \end{array}
$$

when $\eta L \leq 1 / 2$ , then the coeficient of E $\left[ \left\| \nabla f _ { \pi _ { k } ^ { j } } ( \mathbf { x } _ { k } ^ { j } ) \right\| ^ { 2 } \right]$ satisfies $2 \eta + 2 \eta ^ { 2 } L = 2 \eta ( 1 + \eta L ) \leq 3 \eta$ when $i \leq n$ . Then we complete the proof of eq. (7)

Proof of eq. (8). Apply Lemma 10, with

$$
\ell _ { j } ( { \mathbf { x } } ) = \| \nabla f _ { j } ( { \mathbf { x } } ) \| ^ { 2 } , \qquad \ell ( { \mathbf { x } } ) : = { \frac { 1 } { n } } \sum _ { j = 1 } ^ { n } \ell _ { j } ( { \mathbf { x } } )
$$

then

$$
\begin{array} { r l } {  { \mathbb { E } [ \frac { 1 } { n } \sum _ { r = 1 } ^ { n } \| \nabla f _ { r } ( \mathbf { x } _ { k } ^ { i } ) \| ^ { 2 } ] - \mathbb { E } [ \| \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) \| ^ { 2 } ] = \frac { 1 } { n } \sum _ { j \leq i } ( \mathbb { E } [ \| \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ( i , j ) ) \| ^ { 2 } - \| \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) \| ^ { 2 } ] ) } \quad } & { } \\ & { = \frac { 1 } { n } \sum _ { j \leq i } \mathbb { E } [ \| \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ( i , j ) ) \| ^ { 2 } ] - \frac { i - 1 } { n } \mathbb { E } [ \| \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) \| ^ { 2 } ] } \end{array}
$$

$$
\begin{array} { l } { { \displaystyle M _ { k } ^ { i } - Q _ { k } ^ { i } = \frac { 1 } { n } \sum _ { j < i } \left( \mathbb { E } \left[ \left\| \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ( i , j ) ) \right\| ^ { 2 } - Q _ { k } ^ { i } \right] \right) } } \\ { { \displaystyle \quad \quad = \frac { 1 } { n } \sum _ { j < i } \mathbb { E } \left[ \left\| \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ( i , j ) ) \right\| ^ { 2 } \right] - \frac { i - 1 } { n } \mathbb { E } \left[ \left\| \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) \right\| ^ { 2 } \right] } } \end{array}
$$

thus

$$
\sum _ { r = 1 } ^ { n } \mathbb { E } \left[ \left. \nabla f _ { r } ( \mathbf { x } _ { k } ^ { i } ) \right. ^ { 2 } \right] = \sum _ { j < i } \mathbb { E } \left[ \left. \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ( i , j ) ) \right. ^ { 2 } \right] + ( n - ( i - 1 ) ) Q _ { k } ^ { i }\tag{22}
$$

At the same time,

$$
\begin{array} { r l } & { \left\| \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) \right\| ^ { 2 } \leq 2 \left\| \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ( i , j ) ) \right\| ^ { 2 } + 2 \left\| \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) - \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ( i , j ) ) \right\| ^ { 2 } } \\ & { \qquad \leq 2 \left\| \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ( i , j ) ) \right\| ^ { 2 } + 2 L ^ { 2 } \left\| \mathbf { x } _ { k } ^ { i } - \mathbf { x } _ { k } ^ { i } ( i , j ) \right\| ^ { 2 } } \end{array}
$$

Taking the expectation on both sides, we have

$$
\begin{array} { r } { \mathbb { E } \left[ \left. \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) \right. ^ { 2 } \right] \leq 2 \mathbb { E } \left[ \left. \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ( i , j ) ) \right. ^ { 2 } \right] + 2 L ^ { 2 } \mathbb { E } \left[ \left. \mathbf { x } _ { k } ^ { i } ( i , j ) - \mathbf { x } _ { k } ^ { i } \right. ^ { 2 } \right] } \end{array}
$$

Taking summation on the both sides for $j < i ,$ , we have

$$
( i - 1 ) \mathbb { E } \left[ \left\| \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) \right\| ^ { 2 } \right] \leq 2 \sum _ { j < i } \mathbb { E } \left[ \left\| \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ( i , j ) ) \right\| ^ { 2 } \right] + 2 L ^ { 2 } \sum _ { j < i } \mathbb { E } \left[ \left\| \mathbf { x } _ { k } ^ { i } ( i , j ) - \mathbf { x } _ { k } ^ { i } \right\| ^ { 2 } \right]
$$

Combining eq. (22), we further have

$$
\begin{array} { r l } {  { \operatorname { n E } [ \| \nabla f _ { \pi _ { k } ^ { * } } ( \mathbf { x } _ { k } ^ { * } ) \| ^ { 2 } ] \leq 2 \sum _ { j \in S } ^ { 2 } \mathbb { E } [ \| \nabla f _ { \pi _ { k } ^ { * } } ( \mathbf { x } _ { k } ^ { * } ( i , j , j ) ) \| ^ { 2 } ] + 2 L ^ { 2 } \sum _ { j \in S } ^ { 2 } \mathbb { E } [ \| \mathbf { x } _ { k } ^ { * } ( i , j ) - \mathbf { x } _ { k } ^ { * } \| ^ { 2 } ] }  }  \\ & { \qquad +  ( n - ( i - 1 ) ) \mathbb { E } [ \| \nabla f _ { \pi _ { k } ^ { * } } ( \mathbf { x } _ { k } ^ { * } ) \| ^ { 2 } ] } \\ & { = 2 ( \sum _ { j = 1 } ^ { \infty } \mathbb { E } [ \| \nabla f _ { \pi ^ { * } } ( \mathbf { x } _ { k } ^ { * } ) \| ^ { 2 } ] - ( n - ( i - 1 ) ) \mathbb { E } [ \| \nabla f _ { \pi _ { k } ^ { * } } ( \mathbf { x } _ { k } ^ { * } ) \| ^ { 2 } ] ) } \\ & { \qquad + 2 L ^ { 2 } \sum _ { j \in S } ^ { 2 } \mathbb { E } [ \| \mathbf { x } _ { k } ^ { * } ( i , j ) - \mathbf { x } _ { k } ^ { * } \| ^ { 2 } ] + ( n - ( i - 1 ) ) \mathbb { E } [ \| \nabla f _ { \pi _ { k } ^ { * } } ( \mathbf { x } _ { k } ^ { * } ) \| ^ { 2 } ] } \\ &  = 2 \sum _ { j = 1 } ^ { N } \mathbb { E } [ \| \nabla f _ { \pi ^ { * } } ( \mathbf { x } _ { k } ^ { * } ) \| ^ { 2 } ] - ( n - ( i - 1 ) ) \mathbb { E } [ \| \nabla f _ { \pi _ { k } ^ { * } } ( \mathbf  \end{array}
$$

Since $i - 1 \leq n - 1$ and $Q _ { k } ^ { i } \ge 0$ , then

$$
\begin{array} { r l r } {  { \mathbb { E } [ \| \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) \| ^ { 2 } ] \leq \frac { 2 } { n } \sum _ { r = 1 } ^ { n } \mathbb { E } [ \| \nabla f _ { r } ( \mathbf { x } _ { k } ^ { i } ) \| ^ { 2 } ] + \frac { 2 L ^ { 2 } } { n } \sum _ { j < i } \mathbb { E } [ \| \mathbf { x } _ { k } ^ { i } ( i , j ) - \mathbf { x } _ { k } ^ { i } \| ^ { 2 } ] } } \\ & { } & { \leq \frac { 2 } { n } \sum _ { r = 1 } ^ { n } \mathbb { E } [ \| \nabla f _ { r } ( \mathbf { x } _ { k } ^ { i } ) \| ^ { 2 } ] + \frac { 8 \eta ^ { 2 } L ^ { 2 } } { n } \sum _ { j < i } \mathbb { E } [ \| \nabla f _ { \pi _ { k } ^ { j } } ( \mathbf { x } _ { k } ^ { j } ) \| ^ { 2 } ] } \\ & { } & { \leq 8 L \mathbb { E } [ f ( \mathbf { x } _ { k } ^ { i } ) - f ( \mathbf { x } _ { * } ) ] + 4 \sigma _ { * } ^ { 2 } + \frac { 8 \eta ^ { 2 } L ^ { 2 } } { n } \sum _ { j < i } \mathbb { E } [ \| \nabla f _ { \pi _ { k } ^ { j } } ( \mathbf { x } _ { k } ^ { j } ) \| ^ { 2 } ] } \end{array}
$$

Where the last inequality holds by Lemma 11, we complete the proof of eq. (8).

## B.3 Proof of Lemma 3

Proof of eq. (9). Fix $( p , q ) \in \mathcal { I } _ { N }$ and define the history before this position and the set of remaining component indices by

$$
\mathcal F _ { p } ^ { q } : = \sigma ( \pi _ { 1 } , \ldots , \pi _ { p - 1 } , \pi _ { p } ^ { 1 } , \ldots , \pi _ { p } ^ { q - 1 } ) , \qquad \mathcal R _ { p } ^ { q } : = [ n ] \setminus \left\{ \pi _ { p } ^ { 1 } , \ldots , \pi _ { p } ^ { q - 1 } \right\} .
$$

The initialization is deterministic, and $\mathbf { x } _ { p } ^ { q }$ depends only on the indices revealed before $( p , q )$ , so $\mathbf { x } _ { p } ^ { q }$ is ${ \mathcal { F } } _ { p } ^ { q } .$ -measurable. Also, $| \mathcal { R } _ { v } ^ { q } | = n - q + 1$

First consider $( k , i ) \in \mathcal { I } _ { N } ( p , q )$ with $k = p ,$ , so that $i \geq q$ . Conditional on $\mathcal { F } _ { p } ^ { q }$ , the unrevealed sufix of $\pi _ { p }$ is a uniform permutation of $\mathcal { R } _ { p } ^ { q }$ . In particular, both $\pi _ { p } ^ { i }$ and $\pi _ { p } ^ { q }$ are conditionally uniform on this set. Hence

$$
\mathbb { E } \left[ f _ { \pi _ { p } ^ { i } } ( \mathbf { x } _ { p } ^ { q } ) \mid \mathcal { F } _ { p } ^ { q } \right] = \frac { 1 } { n - q + 1 } \sum _ { r \in \mathcal { R } _ { p } ^ { q } } f _ { r } ( \mathbf { x } _ { p } ^ { q } ) = \mathbb { E } \left[ f _ { \pi _ { p } ^ { q } } ( \mathbf { x } _ { p } ^ { q } ) \mid \mathcal { F } _ { p } ^ { q } \right] .
$$

This equality uses only the identical conditional marginal distributions of the two indices. Taking total expectations and using the definition of $\beta _ { p } ^ { q }$ gives

$$
\mathbb { E } \left[ f _ { \pi _ { p } ^ { i } } ( \mathbf { x } _ { p } ^ { q } ) - f ( \mathbf { x } _ { p } ^ { q } ) \right] = \mathbb { E } \left[ f _ { \pi _ { p } ^ { q } } ( \mathbf { x } _ { p } ^ { q } ) - f ( \mathbf { x } _ { p } ^ { q } ) \right] = - \beta _ { p } ^ { q } .
$$

Next consider $k > p$ . The fresh permutation $\pi _ { k }$ is independent of $\mathcal { F } _ { p } ^ { q }$ , and each of its positions is uniform on $[ n ]$ . Therefore

$$
\mathbb { E } \left[ f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { p } ^ { q } ) \mid \mathcal { F } _ { p } ^ { q } \right] = \frac { 1 } { n } \sum _ { r = 1 } ^ { n } f _ { r } ( \mathbf { x } _ { p } ^ { q } ) = f ( \mathbf { x } _ { p } ^ { q } ) ,
$$

so $\mathbb { E } [ f _ { \pi _ { \iota } ^ { i } } ( \mathbf { x } _ { p } ^ { q } ) - f ( \mathbf { x } _ { p } ^ { q } ) ] = 0$ . Combining the two cases proves eq. (9): the comparison-point bias equals $- \beta _ { p } ^ { q }$ at every remaining position of epoch $p$ and vanishes in every later epoch.

Proof of eq. (10). We first establish an inequality that holds for each realization of the permutations. For any comparison point $\mathbf { x } \in \mathbb { R } ^ { d }$ , convexity gives

$$
f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) - f _ { \pi _ { k } ^ { i } } ( \mathbf { x } ) \leq \langle \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) , \mathbf { x } _ { k } ^ { i } - \mathbf { x } \rangle .
$$

Expanding the squared distance after the RR-SGD update yields

$$
\left\| { \bf x } _ { k } ^ { i + 1 } - { \bf x } \right\| ^ { 2 } = \left\| { \bf x } _ { k } ^ { i } - { \bf x } \right\| ^ { 2 } - 2 \eta \langle \nabla f _ { \pi _ { k } ^ { i } } ( { \bf x } _ { k } ^ { i } ) , { \bf x } _ { k } ^ { i } - { \bf x } \rangle + \eta ^ { 2 } \left\| \nabla f _ { \pi _ { k } ^ { i } } ( { \bf x } _ { k } ^ { i } ) \right\| ^ { 2 } .
$$

Combining these two relations and dividing by 2η gives

$$
f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) - f _ { \pi _ { k } ^ { i } } ( \mathbf { x } ) \leq \frac { \left\| \mathbf { x } _ { k } ^ { i } - \mathbf { x } \right\| ^ { 2 } - \left\| \mathbf { x } _ { k } ^ { i + 1 } - \mathbf { x } \right\| ^ { 2 } } { 2 \eta } + \frac { \eta } { 2 } \left\| \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) \right\| ^ { 2 } .
$$

Recall that $N = \tau ( k _ { N } , i _ { N } )$ . Summing over the consecutive positions in $\mathcal { T } _ { N } ( p , q )$ telescopes the distance terms, including across epoch boundaries because $\mathbf { x } _ { k } ^ { n + 1 } = \mathbf { x } _ { k + 1 } ^ { 1 }$ . Thus

$$
\begin{array} { l } { \displaystyle \sum _ { ( k , i ) \in \mathbb { Z } _ { N } ( p , q ) } \left[ f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) - f _ { \pi _ { k } ^ { i } } ( \mathbf { x } ) \right] } \\ { \displaystyle \quad \leq \frac { \left\| \mathbf { x } _ { p } ^ { q } - \mathbf { x } \right\| ^ { 2 } - \left\| \mathbf { x } _ { k N } ^ { i \setminus 1 } - \mathbf { x } \right\| ^ { 2 } } { 2 \eta } + \frac { \eta } { 2 } \displaystyle \sum _ { ( k , i ) \in \mathbb { Z } _ { N } ( p , q ) } \left\| \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) \right\| ^ { 2 } } \\ { \displaystyle \leq \frac { \left\| \mathbf { x } _ { p } ^ { q } - \mathbf { x } \right\| ^ { 2 } } { 2 \eta } + \frac { \eta } { 2 } \displaystyle \sum _ { ( k , i ) \in \mathbb { Z } _ { N } ( p , q ) } \left\| \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) \right\| ^ { 2 } . } \end{array}\tag{23}
$$

The last step discards a nonpositive terminal-distance term. If needed, the successor $\mathbf { x } _ { k _ { N } } ^ { i _ { N } + 1 }$ is defined by one additional update for the analysis.

Since $\mathrm { e q . }$ (23) holds for every realized trajectory and every comparison point, we may substitute the random point $\mathbf { x } = \mathbf { x } _ { p } ^ { q }$ before taking expectations. No independence between this point and the subsequent sampled gradients is needed. With this choice, the initial-distance term is zero. The definitions of $\Delta _ { k } ^ { i }$ and $\beta _ { k } ^ { i }$ , together with eq. (9), give

$$
\begin{array} { r l } & { \mathbb { E } \left[ f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) \right] = f ( \mathbf { x } _ { * } ) + \Delta _ { k } ^ { i } - \beta _ { k } ^ { i } , } \\ & { \mathbb { E } \left[ f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { p } ^ { q } ) \right] = f ( \mathbf { x } _ { * } ) + \Delta _ { p } ^ { q } - \beta _ { p } ^ { q } \mathbf { 1 } _ { \{ k = p \} } . } \end{array}
$$

Consequently, each expected summand on the left-hand side of eq. (23) is

$$
\mathbb { E } \left[ f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) - f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { p } ^ { q } ) \right] = \Delta _ { k } ^ { i } - \Delta _ { p } ^ { q } - \beta _ { k } ^ { i } + \beta _ { p } ^ { q } \mathbf { 1 } _ { \{ k = p \} } .
$$

The positions belonging to epoch $p$ run from global index $\tau ( p , q )$ through $e _ { N , p } ^ { q } = \operatorname* { m i n } { \{ n p , N \} }$ inclusive. Their number is therefore

$$
\sum _ { ( k , i ) \in \mathcal { T } _ { N } ( p , q ) } { \mathbf { 1 } } _ { \left\{ k = p \right\} } = e _ { N , p } ^ { q } - \tau ( p , q ) + 1 .
$$

Taking expectations in eq. (23) now yields

$$
\sum _ { ( k , i ) \in \mathbb { Z } _ { N } ( p , q ) } ( \Delta _ { k } ^ { i } - \Delta _ { p } ^ { q } ) - \sum _ { ( k , i ) \in \mathbb { Z } _ { N } ( p , q ) } \beta _ { k } ^ { i } + ( e _ { N , p } ^ { q } - \tau ( p , q ) + 1 ) \beta _ { p } ^ { q } \leq \frac { \eta } { 2 } \sum _ { ( k , i ) \in \mathbb { Z } _ { N } ( p , q ) } \mathbb { E } \left[ \left\| \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) \right\| ^ { 2 } \right] .
$$

Rearranging proves eq. (10), with the sign of the comparison-point bias retained.

Proof of eq. (11). In eq. (23), take $\left( p , q \right) = \left( 1 , 1 \right)$ and $\mathbf { x } = \mathbf { x } _ { * }$ . Since $\mathbf { x } _ { * }$ is fixed and $\pi _ { k } ^ { i }$ is marginally uniform on $[ n ]$

$$
\mathbb { E } \left[ f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { * } ) \right] = \frac { 1 } { n } \sum _ { r = 1 } ^ { n } f _ { r } ( \mathbf { x } _ { * } ) = f ( \mathbf { x } _ { * } ) .
$$

It follows that

$$
\mathbb { E } \left[ f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) - f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { * } ) \right] = \Delta _ { k } ^ { i } - \beta _ { k } ^ { i } .
$$

Taking expectations in eq. (23) and using $\left\| \mathbf { x } _ { 1 } ^ { 1 } - \mathbf { x } _ { * } \right\| ^ { 2 } = D ^ { 2 }$ , we obtain

$$
\sum _ { ( k , i ) \in \mathcal { T } _ { N } } \left( \Delta _ { k } ^ { i } - \beta _ { k } ^ { i } \right) \leq \frac { D ^ { 2 } } { 2 \eta } + \frac { \eta } { 2 } \sum _ { ( k , i ) \in \mathcal { T } _ { N } } \mathbb { E } \left[ \left. \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) \right. ^ { 2 } \right] .
$$

Moving the sampling-bias sum to the right-hand side proves eq. (11).

## B.4 Proof of Lemma 4

For simplicity, we use the notation where

$$
Q _ { k } ^ { i } = \mathbb { E } \left[ \left. \nabla f _ { \pi _ { k } ^ { i } } ( \mathbf { x } _ { k } ^ { i } ) \right. ^ { 2 } \right]
$$

For the real-valued sequence $\{ \Delta _ { k } ^ { i } : ( k , i ) \in \mathcal { T } _ { N } \}$ , define the averaged sufix

$$
A _ { p } ^ { q } : = \frac { 1 } { N - \tau ( p , q ) + 1 } \sum _ { ( k , i ) \in \mathcal { T } _ { N } ( p , q ) } \Delta _ { k } ^ { i } .
$$

we have

$$
A _ { 1 } ^ { 1 } = \frac { 1 } { N } \sum _ { ( k , i ) \in \mathcal { T } _ { N } } \Delta _ { k } ^ { i } , \qquad A _ { k _ { N } } ^ { i _ { N } } = \Delta _ { k _ { N } } ^ { i _ { N } } = \Delta _ { N }
$$

Since $N = \tau ( k _ { N } , i _ { N } )$ , we have ${ \mathcal { T } _ { N } ( k _ { N } , i _ { N } ) = \{ ( k _ { N } , i _ { N } ) \} }$ }. For $\tau ( p , q ) < N$ , let $( p ^ { \prime } , q ^ { \prime } ) \in \mathcal { I } _ { N }$ be the next position, so that $\tau ( p ^ { \prime } , q ^ { \prime } ) = \tau ( p , q ) + 1$ . Then

$$
\begin{array} { l } { { A _ { p ^ { \prime } } ^ { q ^ { \prime } } - A _ { p } ^ { q } = \displaystyle \frac { 1 } { N - \tau ( p ^ { \prime } , q ^ { \prime } ) + 1 } \sum _ { ( k , i ) \in \mathbb { Z } \backslash N ( p ^ { \prime } , q ^ { \prime } ) } \Delta _ { k } ^ { i } - \displaystyle \frac { 1 } { N - \tau ( p , q ) + 1 } \sum _ { ( k , i ) \in \mathbb { Z } _ { N } ( p , q ) } \Delta _ { k } ^ { i } } } \\ { { \ \quad \ = \displaystyle \frac { \sum _ { ( k , i ) \in \mathbb { Z } _ { N } ( p , q ) } \Delta _ { k } ^ { i } - \Delta _ { p } ^ { q } } { N - \tau ( p , q ) } - \displaystyle \frac { \sum _ { ( k , i ) \in \mathbb { Z } _ { N } ( p , q ) } \Delta _ { k } ^ { i } } { N - \tau ( p , q ) + 1 } } } \\ { { \ \ \quad = \displaystyle \frac { \sum _ { ( k , i ) \in \mathbb { Z } _ { N } ( p , q ) } \Delta _ { k } ^ { i } - \big ( N - \tau ( p , q ) + 1 \big ) \Delta _ { p } ^ { q } } { \big ( N - \tau ( p , q ) \big ) \big ( N - \tau ( p , q ) + 1 \big ) } } } \\ { { \ \quad \quad = \displaystyle \frac { \sum _ { ( k , i ) \in \mathbb { Z } _ { N } ( p , q ) } \big ( \Delta _ { k } ^ { i } - \Delta _ { p } ^ { q } \big ) } { \big ( N - \tau ( p , q ) \big ) \big ( N - \tau ( p , q ) + 1 \big ) } . } } \end{array}
$$

Summing over $A _ { p ^ { \prime } } ^ { q ^ { \prime } } - A _ { p } ^ { q }$ for all $\tau ( p , q ) < N$ , since $( p ^ { \prime } , q ^ { \prime } )$ is the next position after $( p , q )$ , it gives

$$
\sum _ { ( p , q ) \in \mathcal { T } _ { N } , \tau ( p , q ) < N } ( A _ { p ^ { \prime } } ^ { q ^ { \prime } } - A _ { p } ^ { q } ) = A _ { k _ { N } } ^ { i _ { N } } - A _ { 1 } ^ { 1 } = \Delta _ { N } - \frac { 1 } { N } \sum _ { ( k , i ) \in \mathcal { T } _ { N } } \Delta _ { k } ^ { i }
$$

this further implies

$$
\begin{array} { l } { { \Delta _ { N } = \displaystyle \frac { 1 } { N } \sum _ { ( k , i ) \in { \mathcal { T } } _ { N } } \Delta _ { k } ^ { i } + \sum _ { ( p , q ) \in { \mathcal { T } } _ { N } , \tau ( p , q ) < N } ( A _ { p ^ { \prime } } ^ { q ^ { \prime } } - A _ { p } ^ { q } ) } } \\ { { = \displaystyle \frac { 1 } { N } \sum _ { ( k , i ) \in { \mathcal { T } } _ { N } } \Delta _ { k } ^ { i } + \sum _ { ( p , q ) \in { \mathcal { T } } _ { N } } \frac { \sum _ { ( k , i ) \in { \mathcal { T } } _ { N } ( p , q ) } ( \Delta _ { k } ^ { i } - \Delta _ { p } ^ { q } ) } { ( N - \tau ( p , q ) ) ( N - \tau ( p , q ) + 1 ) } . } } \\ { { \displaystyle \qquad \tau ( p , q ) \ll N } } \end{array}
$$

Define the weight coeficient

$$
c _ { N , p } ^ { q } : = \frac { 1 } { ( N - \tau ( p , q ) ) ( N - \tau ( p , q ) + 1 ) } , \qquad \tau ( p , q ) < N .
$$

Apply eqs. (10) and (11), then

$$
\begin{array} { r l } & { \Delta _ { N } \leq \displaystyle \frac { D ^ { 2 } } { 2 \eta N } + \displaystyle \frac { \eta } { 2 N } \sum _ { ( k , i ) \in { \cal Z } _ { N } } Q _ { k } ^ { i } + \displaystyle \frac { \eta } { 2 } \sum _ { ( p , q ) \in { \cal Z } _ { N } } c _ { N , p } ^ { q } \displaystyle \sum _ { ( k , i ) \in { \cal Z } _ { N } ( p , q ) } Q _ { k } ^ { i } } \\ & { \quad \quad \quad \quad + \displaystyle \frac { 1 } { N } \sum _ { ( k , i ) \in { \cal Z } _ { N } } \beta _ { k } ^ { i } + \displaystyle \sum _ { ( p , q ) \in { \cal Z } _ { N } } c _ { N , p } ^ { q } \displaystyle \sum _ { ( k , i ) \in { \cal Z } _ { N } ( p , q ) } \beta _ { k } ^ { i } - \displaystyle \sum _ { ( k , i ) \in { \cal Z } _ { N } } c _ { N , k } ^ { i } \big ( e _ { N , k } ^ { i } - \tau ( k , i ) + 1 \big ) \beta _ { k } ^ { i } . } \\ & { \quad \quad \quad \quad \quad \quad \tau ( p , q ) < N } \end{array}
$$

Term contains β

We collect the coeficients of $Q _ { k } ^ { i }$ and $\beta _ { k } ^ { i }$ respectively.

The coeficient of $Q _ { k } ^ { i }$ . Exchange the order of summation, since for $( p , q )$ whenever $\tau ( p , q ) < N$ and $( k , i ) \in \mathcal { I } _ { N } ( p , q )$ , it is equivalent to $\tau ( p , q ) \leq$ min $\{ \tau ( k , i ) , N - 1 \}$ . Therefore

$$
\sum _ { ( p , q ) \in \mathbb { Z } _ { N } } c _ { N , p } ^ { q } \sum _ { \substack { ( k , i ) \in \mathbb { Z } _ { N } ( p , q ) } } Q _ { k } ^ { i } = \sum _ { ( k , i ) \in \mathbb { Z } _ { N } } \left( \sum _ { \substack { ( p , q ) \in \mathbb { Z } _ { N } } } c _ { N , p } ^ { q } \right) Q _ { k } ^ { i } .\tag{24}
$$

Thus, the gradient terms satisfy

$$
\begin{array} { r l } & { \frac { \eta } { 2 N } \displaystyle \sum _ { ( k , j ) \in \mathbb { Z } _ { N } } Q _ { k } ^ { i } + \frac { \eta } { 2 } \displaystyle \sum _ { ( \eta , q ) \in \mathbb { Z } _ { N } } c _ { W , p } ^ { q } \sum _ { ( k , j ) \in \mathbb { Z } _ { N } ( p , q ) } Q _ { k } ^ { i } } \\ & { \quad \quad = \frac { \eta } { 2 } \displaystyle \sum _ { ( k , j ) \in \mathbb { Z } _ { N } } \left( \frac { 1 } { N } + \frac { \sum _ { i = 1 } ^ { n } c _ { W , p } ^ { i } } { \tau ( p , q ) \in \mathbb { Z } _ { N } \left[ ( - k , j ) , N - 1 \right] } \right) Q _ { k } ^ { i } } \\ & { \quad \quad = \frac { \eta } { 2 } \displaystyle \sum _ { ( k , j ) \in \mathbb { Z } _ { N } } \left( \frac { 1 } { N } + \frac { 1 } { N - \operatorname* { m i n } \left\{ \tau ( k , i ) , N - 1 \right\} } - \frac { 1 } { N } \right) Q _ { k } ^ { i } } \\ & { \quad \quad = \frac { \eta } { 2 } \displaystyle \sum _ { ( k , j ) \in \mathbb { Z } _ { N } } w _ { N , k } ^ { i } Q _ { k } ^ { i } . } \end{array}
$$

with the definition of $\begin{array} { r } { w _ { N , k } ^ { i } = \frac { 1 } { \operatorname* { m a x } \{ N - \tau ( k , i ) , 1 \} } . } \end{array}$

The coeficient of $\beta .$ . Recall the term that contains $\beta$ is.

$$
\underbrace { \frac { 1 } { N } \sum _ { ( k , i ) \in { \mathcal { I } } _ { N } } \beta _ { k } ^ { i } + \sum _ { ( p , q ) \in { \mathcal { T } } _ { N } } c _ { N , p } ^ { q } \sum _ { ( k , i ) \in { \mathcal { I } } _ { N } ( p , q ) } \beta _ { k } ^ { i } - \sum _ { ( k , i ) \in { \mathcal { I } } _ { N } } c _ { N , k } ^ { i } ( e _ { N , k } ^ { i } - \tau ( k , i ) + 1 ) \beta _ { k } ^ { i } } _ { \mathrm { T e r m } _ { 1 } } .
$$

For the Term<sub>1</sub>, similar to eq. (24), the coeficient of $\beta _ { k } ^ { i }$ is

$$
\begin{array} { r l } { \displaystyle \frac { 1 } { N } \sum _ { ( k , i ) \in \mathcal { I } _ { N } } \beta _ { k } ^ { i } + \sum _ { ( p , q ) \in \mathcal { I } _ { N } } c _ { N , p } ^ { q } \sum _ { ( k , i ) \in \mathcal { I } _ { N } ( p , q ) } \beta _ { k } ^ { i } = \sum _ { ( k , i ) \in \mathcal { I } _ { N } } \left( \frac { 1 } { N } + \sum _ { ( p , q ) \in \mathcal { I } _ { N } \atop \tau ( p , q ) \leq \operatorname* { m i n } \{ \tau ( k , i ) , N - 1 \} } c _ { N , p } ^ { q } \right) \beta _ { k } ^ { i } } & { \mathrm { ~ o t ~ } \gamma \leq \tau _ { N - 1 } \atop \beta _ { k } ^ { i } } \\ & { = \displaystyle \sum _ { ( k , i ) \in \mathcal { I } _ { N } } \frac { \beta _ { k } ^ { i } } { N - \operatorname* { m i n } \big \{ \tau ( k , i ) , N - 1 \big \} } . } \end{array}
$$

For a fixed (k, i), when $\tau ( k , i ) < N$ , subtracting the Term<sub>2</sub>, the coeficient of $\beta _ { k } ^ { i }$ is

$$
\begin{array} { r l } { \displaystyle \frac { 1 } { N - \tau ( k , i ) } - c _ { N , k } ^ { i } ( e _ { N , k } ^ { i } - \tau ( k , i ) + 1 ) = \frac { 1 } { N - \tau ( k , i ) } - \frac { e _ { N , k } ^ { i } - \tau ( k , i ) + 1 } { ( N - \tau ( k , i ) ) ( N - \tau ( k , i ) + 1 ) } } & { } \\ { \displaystyle = \frac { N - e _ { N , k } ^ { i } } { ( N - \tau ( k , i ) ) ( N - \tau ( k , i ) + 1 ) } } & { } \\ { \displaystyle = v _ { N , k } ^ { i } } \end{array}
$$

When $\tau ( k , i ) = N$ , only coeficient in Term<sub>1</sub> remains, since the coeficient in Term<sub>2</sub> is zero. Thus, the coeficient of $\beta _ { k } ^ { i }$ is $1 = v _ { N , k } ^ { i }$ . Therefore

$$
\Delta _ { N } \leq \frac { D ^ { 2 } } { 2 \eta N } + \frac { \eta } { 2 } \sum _ { ( k , i ) \in \mathbb { Z } _ { N } } w _ { N , k } ^ { i } Q _ { k } ^ { i } + \sum _ { ( k , i ) \in \mathbb { Z } _ { N } } v _ { N , k } ^ { i } \beta _ { k } ^ { i } ,
$$

which proves eq. (12).

At the same time, we also show two inequalities eqs. (25) and (26) for $v _ { N , k } ^ { i }$ and $w _ { N , k } ^ { i }$ , which will be used in the proof of Proposition proposition 1.

Lemma 12. For every $( k , i ) , ( p , q ) \in \mathcal { T } _ { N }$ , the weights $v _ { N , k } ^ { i }$ and $w _ { N , k } ^ { i }$ defined in Lemma $\it 4$ satisfy

$$
0 \leq v _ { N , k } ^ { i } \leq w _ { N , k } ^ { i }\tag{25}
$$

$$
\frac { 1 } { n } \sum _ { \begin{array} { c } { ( k , i ) \in \mathbb { Z } _ { N } } \\ { \tau ( k , i ) < N , \ ( p , q ) \in \mathcal { P } _ { k } ^ { i } } \end{array} } { v _ { N , k } ^ { i } } \leq w _ { N , p } ^ { q }\tag{26}
$$

Proof. In proving eq. (25), the definition of $e _ { N , k } ^ { i } = \operatorname* { m i n } \{ n k , N \} \ { \mathrm { g i v e s } } \ \tau ( k , i ) \leq e _ { N , k } ^ { i } \leq N$ . When $\tau ( k , i ) < N$ ，

$$
\begin{array} { r l r } {  { 0 \le v _ { N , k } ^ { i } = \frac { N - e _ { N , k } ^ { i } } { ( N - \tau ( k , i ) ) ( N - \tau ( k , i ) + 1 ) } } } \\ & { } & { \le \frac { 1 } { N - \tau ( k , i ) + 1 } \le \frac { 1 } { N - \tau ( k , i ) } = w _ { N , k } ^ { i } . } \end{array}
$$

When $\tau ( k , i ) = N , v _ { N , k } ^ { i } = w _ { N , k } ^ { i } = 1$ . This proves eq. (25).

For eq. (26), fix $( p , q ) \in \mathcal { I } _ { N }$ . Recall the definition of $\mathcal { P } _ { k } ^ { i } = \{ ( k , j ) : 1 \leq j < i \}$ , the condition $( p , q ) \in \mathcal { P } _ { k } ^ { i }$ is equivalent to $k = p$ and $q < i$ . Hence

$$
\begin{array} { r } { \left\{ ( k , i ) \in \mathbb { Z } _ { N } : ( p , q ) \in \mathcal { P } _ { k } ^ { i } \right\} = \left\{ ( p , j ) : q < j \leq n , \tau ( p , j ) \leq N \right\} = \left\{ ( p , j ) : \tau ( p , q ) < \tau ( p , j ) \leq e _ { N , p } ^ { q } \right\} . } \end{array}
$$

where the last equality follows from $e _ { N , p } ^ { q } = \operatorname* { m i n } \{ n p , N \}$ . For these positions, $e _ { N , k } ^ { i } = e _ { N , p } ^ { q } .$

${ \mathrm { I f ~ } } \tau ( p , q ) = N$ , then there is not pair $( k , i )$ satisfying $\tau ( p , q ) < \tau ( k , i ) \leq N$ , thus

$$
\frac { 1 } { n } \sum _ { \begin{array} { c } { ( k , i ) \in \mathcal { T } _ { N } } \\ { \tau ( k , i ) < N , \ ( p , q ) \in \mathcal { P } _ { k } ^ { i } } \end{array} } v _ { N , k } ^ { i } = 0 \leq w _ { N , p } ^ { q } .
$$

When $\tau ( p , q ) < N .$

Case 1: $e _ { N , p } ^ { q } < N$ . All relevant positions satisfy $\tau ( k , i ) < N$ , and $e _ { N , p } ^ { q } = n p$ . Thus

$$
\begin{array} { r l } { \displaystyle \sum _ { ( k , i ) \in X _ { N } } } & { v _ { N , k } ^ { i } = \displaystyle \sum _ { j = q + 1 } ^ { n } \frac { N - e _ { N , p } ^ { i } } { ( N - \tau ( p , j ) ) ( N - \tau ( p , j ) + 1 ) } } \\ { \displaystyle \tau ( k , i ) \le N , ( p , q ) \in P _ { k } ^ { i } } & { \displaystyle = ( N - e _ { N , p } ^ { q } ) \sum _ { j = q + 1 } ^ { n } \left( \frac { 1 } { N - \tau ( p , j ) } - \frac { 1 } { N - \tau ( p , j ) + 1 } \right) } \\ & { \displaystyle = ( N - e _ { N , p } ^ { q } ) \left( \frac { 1 } { N - e _ { N , p } ^ { q } } - \frac { 1 } { N - \tau ( p , q ) } \right) } \\ & { \displaystyle = \frac { e _ { N , p } ^ { q } - \tau ( p , q ) } { N - \tau ( p , q ) } . } \end{array}
$$

When $e _ { N , p } ^ { q } = \tau ( p , q )$ , both sides are zero. Since the positions lie in the same epoch of length $n ,$

$$
0 \leq e _ { N , p } ^ { q } - \tau ( p , q ) \leq n - 1 \leq n .
$$

Consequently,

$$
\frac { 1 } { n } \sum _ { \substack { ( k , i ) \in \mathcal { T } _ { N } } \atop \tau ( k , i ) < N , \ ( p , q ) \in \mathcal { P } _ { k } ^ { i } } v _ { N , k } ^ { i } = \frac { e _ { N , p } ^ { q } - \tau ( p , q ) } { n ( N - \tau ( p , q ) ) } \leq \frac { 1 } { N - \tau ( p , q ) } = w _ { N , p } ^ { q } .
$$

Case 2: $e _ { N , p } ^ { q } = N$ . For every relevant position with $\tau ( p , q ) < \tau ( k , i ) < N$ , we have $e _ { N , k } ^ { i } = N$ , so

$$
v _ { N , k } ^ { i } = \frac { N - e _ { N , k } ^ { i } } { ( N - \tau ( k , i ) ) ( N - \tau ( k , i ) + 1 ) } = 0 .
$$

The terminal position is excluded by the condition $\tau ( k , i ) < N$ in eq. (26). Therefore

$$
\frac { 1 } { n } \sum _ { \begin{array} { c } { ( k , i ) \in \mathcal { T } _ { N } } \\ { \tau ( k , i ) < N , \ ( p , q ) \in \mathcal { P } _ { k } ^ { i } } \end{array} } v _ { N , k } ^ { i } = 0 \leq w _ { N , p } ^ { q } .
$$

Combining the two cases prove eq. (26).

## B.5 Proof of Proposition 1

By eqs. (7) and (8), for every $( k , i ) \in \mathcal { T } _ { N }$

$$
\beta _ { k } ^ { i } \le \frac { \eta } { 2 } Q _ { k } ^ { i } + \frac { 3 \eta } { n } \sum _ { ( p , q ) \in \mathcal { P } _ { k } ^ { i } } Q _ { p } ^ { q } ,\tag{27}
$$

$$
Q _ { k } ^ { i } \leq 8 L \Delta _ { k } ^ { i } + 4 \sigma _ { * } ^ { 2 } + \frac { 8 \eta ^ { 2 } L ^ { 2 } } { n } \sum _ { ( p , q ) \in \mathcal { P } _ { k } ^ { i } } Q _ { p } ^ { q } .\tag{28}
$$

For the change in summation order below, we also need the block sum including the terminal position. For a fixed $( p , q ) \in \mathcal { Z } _ { N } . \mathrm { ~ H ~ } e _ { N , p } ^ { q } < N$ , all relevant positions precede N, so eq. (26) applies. If $\tau ( p , q ) = N$ , the sum is empty. If $e _ { N , p } ^ { q } = N$ and $\tau ( p , q ) < N$ , only the terminal position contributes, and

$$
\frac { 1 } { n } \sum _ { \stackrel { ( k , i ) \in { \mathcal { T } } _ { N } } { ( p , q ) \in { \mathcal { P } } _ { k } ^ { i } } } v _ { N , k } ^ { i } = \frac { 1 } { n } \leq \frac { 1 } { N - \tau ( p , q ) } = w _ { N , p } ^ { q } ,
$$

because $1 \leq N - \tau ( p , q ) \leq n - 1$ . Hence, in every case,

$$
\frac { 1 } { n } \sum _ { \stackrel { ( k , i ) \in \mathcal { T } _ { N } } { ( p , q ) \in \mathcal { P } _ { k } ^ { i } } } v _ { N , k } ^ { i } \leq w _ { N , p } ^ { q } .
$$

Since $v _ { N , k } ^ { i } \ge 0$ , multiplying eq. (27) by $v _ { N , k } ^ { i }$ and summing gives

$$
\begin{array} { r l } & { \displaystyle \sum _ { ( k , i ) \in \mathbb { Z } _ { N } } \displaystyle v _ { N , k } ^ { i } \beta _ { k } ^ { i } \leq \frac { \eta } { 2 } \sum _ { ( k , i ) \in \mathbb { Z } _ { N } } v _ { N , k } ^ { i } Q _ { k } ^ { i } + \frac { 3 \eta } { n } \sum _ { ( k , i ) \in \mathbb { Z } _ { N } } v _ { N , k } ^ { i } \sum _ { ( p , q ) \in \mathcal { P } _ { k } ^ { i } } Q _ { p } ^ { q } } \\ & { \quad \quad \quad \quad \quad \quad = \frac { \eta } { 2 } \sum _ { ( k , i ) \in \mathbb { Z } _ { N } } v _ { N , k } ^ { i } Q _ { k } ^ { i } + 3 \eta \displaystyle \sum _ { ( p , q ) \in \mathbb { Z } _ { N } } Q _ { p } ^ { q } \left( \frac { 1 } { n } \sum _ { ( k , i ) \in \mathbb { Z } _ { N } } v _ { N , k } ^ { i } \right) } \\ & { \quad \quad \quad \quad \quad \leq \frac { \eta } { 2 } \sum _ { ( k , i ) \in \mathbb { Z } _ { N } } w _ { N , k } ^ { i } Q _ { k } ^ { i } + 3 \eta \displaystyle \sum _ { ( p , q ) \in \mathbb { Z } _ { N } } w _ { N , p } ^ { q } Q _ { p } ^ { q } . } \end{array}
$$

The last inequality uses eq. (25) and the block-sum bound above. Combining this with eq. (12) proves eq. (13).

## B.6 Proof of Lemma 5

Fix $( p , q ) \in \mathcal { Z } _ { N } . \mathrm { ~ I f ~ } \tau ( p , q ) = N$ , the sum in eq. (15) is empty. Otherwise, let

$$
a : = N - \tau ( p , q ) \geq 1 , \qquad m : = e _ { N , p } ^ { q } - \tau ( p , q ) \leq \operatorname* { m i n } \left\{ n - 1 , a \right\} .
$$

We first bound the sum including the terminal position. Since $( p , q ) \in \mathcal { P } _ { k } ^ { i }$ means $k = p$ and $q < i ,$ the definition of $w _ { N , k } ^ { i }$ gives

$$
\frac { 1 } { n } \sum _ { ( k , i ) \in \mathcal { I } _ { N } \atop ( p , q ) \in \mathcal { P } _ { k } ^ { i } } w _ { N , k } ^ { i } = \frac { 1 } { n } \left( \sum _ { j = 1 } ^ { \operatorname* { m i n } \{ m , a - 1 \} } \frac { 1 } { a - j } + \mathbf { 1 } _ { \{ m = a \} } \right) .
$$

If $a \leq n .$ , then $a / n \le 1$ , and

$$
\frac { a } { n } \left( \sum _ { j = 1 } ^ { \operatorname* { m i n } \{ m , a - 1 \} } \frac { 1 } { a - j } + { \bf 1 } _ { \{ m = a \} } \right) \leq \mathcal { H } _ { a - 1 } + 1 \leq 2 \mathcal { H } _ { N } .
$$

If $a > n$ , then $m \leq n - 1 < a ,$ so the terminal term is absent. For $1 \leq j \leq n - 1$ , we have $a / ( a - j ) \leq n / ( n - j )$ . Consequently,

$$
\frac { a } { n } \sum _ { j = 1 } ^ { m } \frac { 1 } { a - j } \leq \sum _ { j = 1 } ^ { n - 1 } \frac { 1 } { n - j } = \mathcal { H } _ { n - 1 } \leq \mathcal { H } _ { N } .
$$

Since $w _ { N , p } ^ { q } = 1 / a$ , these two cases yield

$$
\frac { 1 } { n } \sum _ { \begin{array} { c } { ( k , i ) \in \mathcal { T } _ { N } } \\ { ( p , q ) \in \mathcal { P } _ { k } ^ { i } } \end{array} } w _ { N , k } ^ { i } \leq 2 \mathcal { H } _ { N } w _ { N , p } ^ { q } .
$$

This proves eq. (15), including the contribution at the terminal position.

## B.7 Proof of Lemma 6

We argue by induction over the global positions. $\mathrm { A t } \ ( k , i ) = ( 1 , 1 )$ , smoothness and $\nabla f ( \mathbf { x } _ { * } ) = 0$ give

$$
\Delta _ { 1 } ^ { 1 } = \mathbb { E } \left[ f ( \mathbf { x } _ { 1 } ^ { 1 } ) - f ( \mathbf { x } _ { * } ) \right] \leq \frac { L } { 2 } D ^ { 2 } \leq \frac { D ^ { 2 } } { \eta } .
$$

For a fixed $N = \tau ( k _ { N } , i _ { N } ) \ge 2$ , suppose the conclusion holds at all earlier positions:

$$
\Delta _ { k } ^ { i } \leq \frac { D ^ { 2 } } { \eta \tau ( k , i ) } + 1 2 8 \eta \sigma _ { * } ^ { 2 } \mathcal { H } _ { \tau ( k , i ) } , \qquad ( k , i ) \in \mathcal { T } _ { N } , \quad \tau ( k , i ) < N .
$$

Since $w _ { N , k _ { N } } ^ { i _ { N } } = 1$ and $\Delta _ { N } = \Delta _ { k _ { N } } ^ { i _ { N } }$ 2

$$
V _ { N } = \Delta _ { N } + \sum _ { ( k , i ) \in \mathbb { Z } _ { N } \atop \tau ( k , i ) < N } \frac { \Delta _ { k } ^ { i } } { N - \tau ( k , i ) } .
$$

At the same time, we have the harmonic type bound:

$$
\begin{array} { r l r } {  { \sum _ { ( k , i ) \in \mathbb { Z } _ { N } } \frac { 1 } { \tau ( k , i ) ( N - \tau ( k , i ) ) } = \frac { 2 \mathcal { H } _ { N - 1 } } { N } , } } \\ & { \tau ( k , i ) < N } \\ & { } & { \displaystyle \sum _ { t = 1 } ^ { N - 1 } \frac { \mathcal { H } _ { t } } { N - t } \leq \mathcal { H } _ { N } \sum _ { t = 1 } ^ { N - 1 } \frac { 1 } { N - t } \leq \mathcal { H } _ { N } \mathcal { H } _ { N - 1 } , } \end{array}
$$

respectively. By the induction hypothesis,

$$
\begin{array} { r l } & { V _ { N } \leq \Delta _ { N } + \displaystyle \frac { D ^ { 2 } } { \eta } \displaystyle \sum _ { ( k , i ) \in \bar { \cal Z } _ { N } } \displaystyle \frac { 1 } { \tau ( k , i ) ( N - \tau ( k , i ) ) } + 1 2 8 \eta \sigma _ { * ( k , i ) \in \bar { \cal Z } _ { N } } ^ { 2 } \displaystyle \sum _ { ( k , i ) < \bar { \cal Z } _ { N } } \displaystyle \frac { \mathcal { H } _ { \tau ( k , i ) } } { N - \tau ( k , i ) } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \tau ( k , i ) < N } \\ & { \quad \quad \quad \quad \quad \leq \Delta _ { N } + \displaystyle \frac { 2 D ^ { 2 } \mathcal { H } _ { N - 1 } } { \eta N } + 1 2 8 \eta \sigma _ { * ( k , i ) \in \bar { \cal Z } _ { N } } ^ { 2 } \displaystyle \sum _ { ( k , i ) \in \bar { \cal Z } _ { N } } \displaystyle \frac { \mathcal { H } _ { N } } { N - \tau ( k , i ) } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \tau ( k , i ) < N } \\ & { \quad \quad \quad \quad \quad \quad \quad \leq \Delta _ { N } + \displaystyle \frac { 2 D ^ { 2 } \mathcal { H } _ { N - 1 } } { \eta N } + 1 2 8 \eta \sigma _ { * } ^ { 2 } \mathcal { H } _ { N } \mathcal { H } _ { N - 1 } . } \end{array}\tag{29}
$$

Substituting eq. (29) into eq. (18) gives

$$
\begin{array} { r l } {  { \Delta _ { N } \leq \frac { D ^ { 2 } } { 2 \eta N } + 6 4 \eta L V _ { N } + 3 2 \eta \sigma _ { * } ^ { 2 } ( \mathcal { H } _ { N - 1 } + 1 ) } } \\ & { \leq \frac { D ^ { 2 } } { 2 \eta N } + 6 4 \eta L \Delta _ { N } + 1 2 8 \eta L \frac { D ^ { 2 } \mathcal { H } _ { N - 1 } } { \eta N } + 8 1 9 2 \eta L \eta \sigma _ { * } ^ { 2 } \mathcal { H } _ { N } \mathcal { H } _ { N - 1 } + 3 2 \eta \sigma _ { * } ^ { 2 } ( \mathcal { H } _ { N - 1 } + 1 ) . } \end{array}
$$

Since $\eta L \mathcal { H } _ { N } \leq 1 / 5 1 2$ , we obtain

$$
\frac { 1 } { 2 } + 1 2 8 \eta L \mathcal { H } _ { N - 1 } \le \frac { 3 } { 4 } , \qquad 8 1 9 2 \eta L \mathcal { H } _ { N } \mathcal { H } _ { N - 1 } + 3 2 ( \mathcal { H } _ { N - 1 } + 1 ) \le 8 0 \mathcal { H } _ { N } .
$$

Moving $6 4 \eta L \Delta _ { N }$ to the left hand side

$$
\begin{array} { r l } & { ( 1 - 6 4 \eta L ) \Delta _ { N } \leq \displaystyle \frac { D ^ { 2 } } { \eta N } \left( \frac { 1 } { 2 } + 1 2 8 \eta L \mathcal { H } _ { N - 1 } \right) + \left( 8 1 9 2 \eta L \mathcal { H } _ { N } \mathcal { H } _ { N - 1 } + 3 2 ( \mathcal { H } _ { N - 1 } + 1 ) \right) \eta \sigma _ { * } ^ { 2 } } \\ & { \qquad \leq \displaystyle \frac { 3 D ^ { 2 } } { 4 \eta N } + 8 0 \eta \sigma _ { * } ^ { 2 } \mathcal { H } _ { N } . } \end{array}\tag{30}
$$

Also, $1 - 6 4 \eta L \geq 7 / 8$ . Therefore

$$
\Delta _ { N } \leq \frac { 6 } { 7 } \frac { D ^ { 2 } } { \eta N } + \frac { 6 4 0 } { 7 } \eta \sigma _ { * } ^ { 2 } \mathcal { H } _ { N } \leq \frac { D ^ { 2 } } { \eta N } + 1 2 8 \eta \sigma _ { * } ^ { 2 } \mathcal { H } _ { N } ,
$$

which completes the induction.

## B.8 Proof of Lemma 8

When $n = 1$ , we have $\sigma _ { * } ^ { 2 } = 0$ , and RR-SGD degenerates to gradient descent. Since $\eta L \leq 1 / 2$ , its objective does not increase in a single update, which proves the claim for $i = 1 , 2$ . We therefore assume $n \geq 2$

Fix an epoch $k$ and condition on its starting point $\mathbf { y } = \mathbf { x } _ { k } ^ { 1 }$ The fresh permutation $\pi _ { k }$ is independent of ${ \bf y } ;$ write $\mathbb { E } _ { k } [ \cdot ] = \mathbb { E } [ \cdot \mid \mathbf { x } _ { k } ^ { 1 } ]$ and $G _ { \mathbf { y } } = f ( \mathbf { y } ) - f ( \mathbf { x } _ { * } )$ . Define

$$
a _ { j } : = \nabla f _ { j } ( \mathbf { x } _ { * } ) , \qquad b _ { j } ( \mathbf { y } ) : = \nabla f _ { j } ( \mathbf { y } ) - \nabla f _ { j } ( \mathbf { x } _ { * } ) .
$$

Since $\nabla f ( \mathbf { x } _ { * } ) = 0$ , we have $\textstyle \sum _ { j = 1 } ^ { n } a _ { j } = 0$ and $n ^ { - 1 } \sum _ { j = 1 } ^ { n } \| a _ { j } \| ^ { 2 } = \sigma _ { * } ^ { 2 }$ . Co-coercivity, averaged over the components, also gives

$$
{ \frac { 1 } { n } } \sum _ { j = 1 } ^ { n } \| b _ { j } ( \mathbf { y } ) \| ^ { 2 } \leq { \frac { 2 L } { n } } \sum _ { j = 1 } ^ { n } { \big ( } f _ { j } ( \mathbf { y } ) - f _ { j } ( \mathbf { x } _ { * } ) - \langle a _ { j } , \mathbf { y } - \mathbf { x } _ { * } \rangle { \big ) } = 2 L G _ { \mathbf { y } } .
$$

For $0 \leq s \leq n$ , the variance formula for sampling without replacement yields

$$
\mathbb { E } _ { k } \left\| \sum _ { r = 1 } ^ { s } a _ { \pi _ { k } ^ { r } } \right\| ^ { 2 } = \frac { s ( n - s ) } { n - 1 } \sigma _ { * } ^ { 2 } \leq n \sigma _ { * } ^ { 2 } .
$$

Indeed, for distinct positions $r , u .$ the expected inner product is $\mathbb { E } _ { k } \langle a _ { \pi _ { k } ^ { r } } , a _ { \pi _ { k } ^ { u } } \rangle = - \sigma _ { * } ^ { 2 } / ( n - 1 )$ . By Cauchy–Schwarz and the uniform marginal distribution at each position,

$$
\mathbb { E } _ { k } \left\| \sum _ { r = 1 } ^ { s } b _ { \pi _ { k } ^ { r } } ( \mathbf { y } ) \right\| ^ { 2 } \leq s \sum _ { r = 1 } ^ { s } \mathbb { E } _ { k } \| b _ { \pi _ { k } ^ { r } } ( \mathbf { y } ) \| ^ { 2 } \leq 2 s ^ { 2 } L G _ { \mathbf { y } } .
$$

Consequently, the sum of gradients evaluated at the epoch’s starting point satisfies

$$
\mathbb { E } _ { k } \left\| \sum _ { r = 1 } ^ { s } \nabla f _ { \pi _ { k } ^ { r } } ( \mathbf { y } ) \right\| ^ { 2 } \leq 4 n ^ { 2 } L G _ { \mathbf { y } } + 2 n \sigma _ { * } ^ { 2 } = : A _ { \mathbf { y } } ^ { 2 } .
$$

Let $d _ { s } = ( \mathbb { E } _ { k } \| { \bf x } _ { k } ^ { s + 1 } - { \bf y } \| ^ { 2 } ) ^ { 1 / 2 }$ , so that $d _ { 0 } = 0$ . The update rule gives

$$
{ \mathbf x } _ { k } ^ { s + 1 } - { \mathbf y } = - \eta \sum _ { r = 1 } ^ { s } \nabla f _ { \pi _ { k } ^ { r } } ( { \mathbf y } ) - \eta \sum _ { r = 1 } ^ { s } \left( \nabla f _ { \pi _ { k } ^ { r } } ( { \mathbf x } _ { k } ^ { r } ) - \nabla f _ { \pi _ { k } ^ { r } } ( { \mathbf y } ) \right) .
$$

Taking conditional $L ^ { 2 }$ norms and applying smoothness yields

$$
d _ { s } \leq \eta A _ { \mathbf { y } } + \eta L \sum _ { r = 0 } ^ { s - 1 } d _ { r } .
$$

An induction therefore gives, for $1 \leq s \leq n ,$

$$
d _ { s } \leq \eta A _ { \mathbf { y } } ( 1 + \eta L ) ^ { s - 1 } \leq \eta A _ { \mathbf { y } } e ^ { \eta n L } .
$$

Since $\eta n L \leq 1 / 2$ , this implies

$$
\begin{array} { r } { \mathbb { E } _ { k } \| { \bf x } _ { k } ^ { s + 1 } - { \bf y } \| ^ { 2 } \leq e \eta ^ { 2 } \left( 4 n ^ { 2 } L G _ { { \bf y } } + 2 n \sigma _ { * } ^ { 2 } \right) . } \end{array}
$$

Finally, smoothness of $f$ and $\| \nabla f ( \mathbf { y } ) \| ^ { 2 } \leq 2 L G _ { \mathbf { y } }$ imply, for any $\mathbf { x } ,$

$$
f ( \mathbf { x } ) - f ( \mathbf { x } _ { * } ) \leq 2 G _ { \mathbf { y } } + L \| \mathbf { x } - \mathbf { y } \| ^ { 2 } .
$$

Combining the preceding bounds, for $i = s + 1$ we obtain

$$
\begin{array} { r } { \mathbb { E } _ { k } [ f ( \mathbf { x } _ { k } ^ { i } ) - f ( \mathbf { x } _ { * } ) ] \leq \left( 2 + 4 e ( \eta n L ) ^ { 2 } \right) G _ { \mathbf { y } } + 2 e \eta ^ { 2 } n L \sigma _ { * } ^ { 2 } \leq 6 G _ { \mathbf { y } } + 8 \eta ^ { 2 } n L \sigma _ { * } ^ { 2 } . } \end{array}
$$

The case $i = 1$ is immediate. Taking total expectations proves the claim for every $1 \leq i \leq n + 1$

## B.9 Proof of Theorem 1

Fix a pair $( k , i )$ chosen before the run, with $1 \leq i \leq n$ and $2 \leq N = \tau ( k , i ) \leq n K + 1$ , and let $\Lambda = 1 + \log ( n K )$ . We retain explicit constants throughout the proof. The assumed stepsize satisfies

$$
\eta L \mathcal { H } _ { N } \leq \eta L \mathcal { H } _ { n K + 1 } \leq 2 \eta L \Lambda \leq \frac { 1 } { 5 1 2 } .
$$

Thus Lemma 6 gives, at every such fixed position,

$$
\Delta _ { k } ^ { i } \le \frac { D ^ { 2 } } { \eta N } + 2 5 6 \Lambda \eta \sigma _ { * } ^ { 2 } .\tag{31}
$$

We combine this bound with the small-step endpoint result using

$$
\eta _ { 0 } : = \frac { 1 } { 4 n L \sqrt { 2 ( 1 + \log K ) } } .
$$

Small stepsizes: $\eta \leq \eta _ { 0 }$ . Here

$$
\eta n L \leq 1 / ( 4 \sqrt { 2 } ) < 1 / 2 ,
$$

so Lemma 8 applies. We first handle positions in the first two epochs. For $k = 1$ , smoothness at the minimizer gives $\Delta _ { 1 } ^ { 1 } \le L D ^ { 2 } / 2$ , and hence

$$
\begin{array} { r } { \Delta _ { 1 } ^ { i } \leq 6 \Delta _ { 1 } ^ { 1 } + 8 \eta ^ { 2 } n L \sigma _ { * } ^ { 2 } } \\ { \leq 3 L D ^ { 2 } + 8 \eta ^ { 2 } n L \sigma _ { * } ^ { 2 } } \\ { \leq \displaystyle \frac { 3 D ^ { 2 } } { 2 \eta N } + 8 \eta ^ { 2 } n L \sigma _ { * } ^ { 2 } . } \end{array}
$$

The last inequality uses $N = i \leq n$ and $\eta L N \leq \eta L n \leq 1 / 2$ . For $k = 2 .$ , use $\mathbf { x } _ { 2 } ^ { 1 } = \mathbf { x } _ { 1 } ^ { n + 1 }$ and apply the same lemma twice:

$$
\begin{array} { l } { \displaystyle \Delta _ { 2 } ^ { i } \leq 6 \Delta _ { 2 } ^ { 1 } + 8 \eta ^ { 2 } n L \sigma _ { * } ^ { 2 } } \\ { \displaystyle \quad \leq 3 6 \Delta _ { 1 } ^ { 1 } + 5 6 \eta ^ { 2 } n L \sigma _ { * } ^ { 2 } } \\ { \displaystyle \quad \leq 1 8 L D ^ { 2 } + 5 6 \eta ^ { 2 } n L \sigma _ { * } ^ { 2 } } \\ { \displaystyle \quad \leq \frac { 1 8 D ^ { 2 } } { \eta N } + 5 6 \eta ^ { 2 } n L \sigma _ { * } ^ { 2 } . } \end{array}
$$

Here $N = n + i \leq 2 n$ , so $\eta L N \leq 2 \eta n L \leq 1$

For $k \geq 3 ,$ , set $m = k - 1$ . Then $2 \leq m \leq K$ , and

$$
\eta \leq \eta _ { 0 } \leq \frac { 1 } { 4 n L \sqrt { 2 ( 1 + \log m ) } } .
$$

The random-permutation case in the proof of Theorem C.1 of Liu and Zhou (2024) gives an explicit version of Lemma 7. With no regularizer and $L = \operatorname* { m a x } _ { r } L _ { r }$ , its bound after m completed epochs is

$$
\Delta _ { k } ^ { 1 } \leq \frac { D ^ { 2 } } { \eta n m } + \frac { 1 6 } { 3 } \eta ^ { 2 } n L \sigma _ { * } ^ { 2 } ( 1 + \log m ) .
$$

Applying Lemma 8 therefore gives

$$
\begin{array} { r l } & { \Delta _ { k } ^ { i } \leq 6 \Delta _ { k } ^ { 1 } + 8 \eta ^ { 2 } n L \sigma _ { * } ^ { 2 } } \\ & { \quad \leq \displaystyle \frac { 6 D ^ { 2 } } { \eta n m } + \big [ 3 2 ( 1 + \log m ) + 8 \big ] \eta ^ { 2 } n L \sigma _ { * } ^ { 2 } } \\ & { \quad \leq \displaystyle \frac { 1 2 D ^ { 2 } } { \eta N } + 4 0 \Lambda \eta ^ { 2 } n L \sigma _ { * } ^ { 2 } . } \end{array}
$$

The last step uses $N = n m + i \leq 2 n m , 1 + \log m \leq \Lambda$ , and $\Lambda \geq 1$ . Combining the three cases $k = 1$ 2 $k = 2$ , and $k \geq 3$ , we obtain

$$
\Delta _ { k } ^ { i } \leq \frac { 1 8 D ^ { 2 } } { \eta N } + 5 6 \Lambda \eta ^ { 2 } n L \sigma _ { * } ^ { 2 } \leq \frac { 1 8 D ^ { 2 } } { \eta N } + 5 6 \Lambda ^ { 3 / 2 } \eta ^ { 2 } n L \sigma _ { * } ^ { 2 } .\tag{32}
$$

Since $\eta n L < 1$ , the noise term in the minimum of Theorem 1 is $\eta ^ { 2 } n L \sigma _ { * } ^ { 2 }$ in this regime.

Intermediate stepsizes: $\eta > \eta _ { 0 }$ and $\eta n L < 1$ . In this case,

$$
\frac { 1 } { \eta n L } < 4 \sqrt { 2 ( 1 + \log K ) } \leq 4 \sqrt { 2 \Lambda } .
$$

Consequently,

$$
2 5 6 \Lambda \eta \sigma _ { \ast } ^ { 2 } = \frac { 2 5 6 \Lambda } { \eta n L } \eta ^ { 2 } n L \sigma _ { \ast } ^ { 2 } \leq 1 0 2 4 \sqrt { 2 } \Lambda ^ { 3 / 2 } \eta ^ { 2 } n L \sigma _ { \ast } ^ { 2 } .
$$

Substitution into eq. (31) yields

$$
\Delta _ { k } ^ { i } \leq { \frac { D ^ { 2 } } { \eta N } } + 1 0 2 4 \sqrt { 2 } \Lambda ^ { 3 / 2 } \eta ^ { 2 } n L \sigma _ { * } ^ { 2 } \leq { \frac { D ^ { 2 } } { \eta N } } + 2 0 4 8 \Lambda ^ { 3 / 2 } \eta ^ { 2 } n L \sigma _ { * } ^ { 2 } .\tag{33}
$$

Again, the minimum equals $\eta ^ { 2 } n L \sigma _ { * } ^ { 2 }$ because ηn $L < 1$

Larger stepsizes: ηn $L \geq 1$ . Now the minimum equals $\eta \sigma _ { * } ^ { 2 }$ . By eq. (31) and $\Lambda \le \Lambda ^ { 3 / 2 }$ 2

$$
\Delta _ { k } ^ { i } \leq \frac { D ^ { 2 } } { \eta N } + 2 5 6 \Lambda \eta \sigma _ { * } ^ { 2 } \leq \frac { D ^ { 2 } } { \eta N } + 2 5 6 \Lambda ^ { 3 / 2 } \eta \sigma _ { * } ^ { 2 } .\tag{34}
$$

Combining eqs. (32) to (34), and using $1 8 \geq 1$ and $2 0 4 8 \geq \operatorname* { m a x } \{ 5 6 , 2 5 6 \}$ , proves the explicit bound

$$
\Delta _ { k } ^ { i } \leq \frac { 1 8 D ^ { 2 } } { \eta \tau ( k , i ) } + 2 0 4 8 \Lambda ^ { 3 / 2 } \operatorname* { m i n } \left. \eta \sigma _ { * } ^ { 2 } , \eta ^ { 2 } n L \sigma _ { * } ^ { 2 } \right.
$$

for every pair fixed before the run in the stated range, including $( k , i ) = ( K + 1 , 1 )$ . This establishes Theorem 1.

## B.10 Proof of Corollary 1

Let $T = n K , \Lambda = 1 + \log T$ , and $M = 1 / ( 1 0 2 4 L \Lambda )$ . Applying Theorem 1 at $( K + 1 , 1 )$ and using $\tau ( K + 1 , 1 ) = T + 1 \geq T$ gives

$$
\mathbb { E } [ f ( \mathbf { x } _ { K + 1 } ^ { 1 } ) - f ( \mathbf { x } _ { * } ) ] \lesssim \frac { D ^ { 2 } } { \eta T } + \Lambda ^ { 3 / 2 } \sigma _ { * } ^ { 2 } \operatorname* { m i n } \{ \eta , n L \eta ^ { 2 } \} .
$$

If $\sigma _ { * } = 0$ , choosing $\eta = M$ gives the claim immediately. For $\sigma _ { * } > 0$ , the two candidates in Corollary 1 are obtained by balancing the initial-distance term in Theorem 1 against each of its two noise expressions:

$$
\frac { D ^ { 2 } } { T \eta _ { \mathrm { l i n } } } = \Lambda ^ { 3 / 2 } \sigma _ { * } ^ { 2 } \eta _ { \mathrm { l i n } } , \qquad \frac { D ^ { 2 } } { T \eta _ { \mathrm { q u a d } } } = \Lambda ^ { 3 / 2 } \sigma _ { * } ^ { 2 } n L \eta _ { \mathrm { q u a d } } ^ { 2 } .
$$

Taking $u = \mathrm { m a x } \{ \eta _ { \mathrm { l i n } } , \eta _ { \mathrm { q u a d } } \}$ selects the smaller of the two balanced initial-distance terms. Since u equals one of these candidates, the corresponding equality above implies

$$
\Lambda ^ { 3 / 2 } \sigma _ { * } ^ { 2 } \operatorname* { m i n } \{ u , n L u ^ { 2 } \} \leq \frac { D ^ { 2 } } { T u } .
$$

The prescribed choice $\eta = \operatorname* { m i n } \{ M , u \}$ enforces the stepsize restriction in Theorem 1. The noise term is nondecreasing in the stepsize, and $1 / \eta \le 1 / M + 1 / u$ . Hence

$$
\begin{array} { r l r } {  { \mathbb { E } [ f ( { \mathbf x } _ { K + 1 } ^ { 1 } ) - f ( { \mathbf x } _ { * } ) ] \lesssim \frac { D ^ { 2 } } { M T } + \frac { 2 D ^ { 2 } } { T u } } } \\ & { } & { \lesssim \frac { \Lambda L D ^ { 2 } } { n K } + \operatorname* { m i n } \{ \frac { \Lambda ^ { 3 / 4 } \sigma _ { * } D } { \sqrt { n K } } , \Lambda ^ { 1 / 2 } ( \frac { L \sigma _ { * } ^ { 2 } D ^ { 4 } } { n K ^ { 2 } } ) ^ { 1 / 3 } \} . } \end{array}
$$

Hiding the logarithmic factors yields Corollary 1.