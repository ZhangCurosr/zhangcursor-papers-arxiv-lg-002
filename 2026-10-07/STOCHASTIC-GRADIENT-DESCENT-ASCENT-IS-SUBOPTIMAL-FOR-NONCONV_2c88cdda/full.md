# STOCHASTIC GRADIENT DESCENT ASCENT IS SUBOPTIMAL FOR NONCONVEX-PŁ MIN-MAX GAMES

Junsoo Ha

junsoo.ha.contact@gmail.com

## ABSTRACT

How far can stochastic gradient descent ascent (SGDA) go by tuning its timescale ratio and step sizes in nonconvex min-max games? We answer this question for nonconvex-PŁ (NC-PŁ) games by establishing the first tight complexity of twotimescale SGDA with a fixed timescale ratio and non-increasing step sizes. For ℓ-smooth games with an inner $\mu { \mathrm { - P E } }$ inequality, we prove a complexity lower bound $\Omega ( \kappa ^ { 2 } \ell \varepsilon ^ { - 2 } + \kappa ^ { 4 } \ell \sigma ^ { 2 } \varepsilon ^ { - 4 } )$ , where $\kappa = \ell / \mu$ is the condition number, $\sigma ^ { 2 }$ is the gradient variance, and ε measures the outer gradient norm. This matches existing SGDA upper bounds and establishes a complexity separation from Smoothed-AGDA (Yang et al., 2022). In addition, we show that SGDA can fail to find a stationary point when its timescale ratio is as small as $o ( \kappa ^ { 2 } )$ ). Our negative results highlight the fundamental limitation of SGDA in NC-PŁ games, and justify the development of alternative methods.

## 1 INTRODUCTION

Nonconvex min-max games have become a cornerstone of modern machine learning, underpinning generative adversarial networks (GANs) (Goodfellow et al., 2014), distributionally robust learning (Sinha et al., 2018), and language model alignment (Liu et al., 2026; Xu et al., 2025). Without additional structure, however, even finding a stationary point of a min-max game can be computationally intractable (Daskalakis et al., 2021; Bernasconi et al., 2026). For instance, games often exhibit limit cycles (Hsieh et al., 2021) and chaotic behavior (Sato et al., 2002). Such pathological examples motivate the study of structured nonconvex min-max games (Lin et al., 2020; 2025; Ha, 2026). In particular, nonconvex-PŁ (NC-PŁ) games assume a Polyak–Łojasiewicz (PŁ) inequality (Polyak, 1963) on the inner maximization problem and have been actively studied as one of the weakest forms of structure (Nouiehed et al., 2019; Yang et al., 2022; Cho and Yun, 2023; Ha, 2026; Cai et al., 2026).

Even in NC-PŁ games, finding a Nash equilibrium (Nash, 1951) can be still intractable; hence, we seek a stationary point of the outer function $\Phi ( x ) : = \operatorname* { m a x } _ { y } f ( x , y )$ . Two-timescale SGDA is a direct analog of SGD for such min-max games: it uses a faster ascent step to track the outer gradient $\nabla \Phi ( x )$ and performs a slower inexact descent on Φ. For ℓ-smooth nonconvex-µ-PŁ games with condition number $\kappa = \ell / \mu$ and gradient variance $\sigma ^ { 2 }$ , SGDA admits a complexity upper bound $O ( \kappa ^ { 2 } \ell \varepsilon ^ { - 2 } + \kappa ^ { 4 } \ell \sigma ^ { 2 } \varepsilon ^ { - 4 } )$ for finding an ε-stationary point (Yang et al., 2022; Ha, 2026). Meanwhile, recent studies have developed more sophisticated techniques to attain sharper bounds (Yang et al., 2022; Huang et al., 2025; Cai et al., 2026). This raises the following natural question:

Can SGDA obtain better complexity by tuning its timescale ratio and step sizes?

Contributions. In this work, we answer this question negatively. Our contributions are twofold.

(i) Suboptimality of SGDA. We prove matching lower bounds for SGDA with a fixed timescale ratio and nonincreasing step sizes, and establish its suboptimality in NC-PŁ games.

(ii) Optimal timescale ratio. Additionally, we prove that SGDA with a small timescale ratio $o \bar { ( \kappa ^ { 2 } ) }$ can fail to find a stationary point, and that existing $\Theta ( \kappa ^ { 2 } )$ timescale ratio is optimal.

Sections 2–4 cover related work, the setup, and main results; Sections 5–6 prove the lower bounds;   
Section 7 discusses implications and future directions.

## 2 RELATED WORK

Nonconvex games are computationally intractable in general (Daskalakis et al., 2021), and even finding an approximate stationary point can be hard as well (Bernasconi et al., 2026). Such hardness results motivate structured games, including nonconvex-PŁ (NC-PŁ) (Nouiehed et al., 2019; Yang et al., 2020; 2022; Cho and Yun, 2023) and nonconvex-concave (NC-C) games (Thekumparampil et al., 2019; Lin et al., 2020; 2025; Ha, 2026), which assume a PŁ inequality and concavity on inner maximization problems, respectively. Notably, these structures make finding stationary points computationally tractable and admit inexact-descent interpretations (Nouiehed et al., 2019; Lin et al., 2020; 2025; Ha, 2026). In particular, Nouiehed et al. (2019) showed that taking multiple ascent steps per descent step suffices to find an approximate stationary point in NC-PŁ games, and interpreted these updates as inexact descent on ${ \bf { \bar { \Phi } } } ( x ) = \operatorname* { m a x } _ { y } f ( x , { \bf \bar { y } } )$ . Lin et al. (2020; 2025) extended the same interpretation to two-timescale SGDA for nonconvex-strongly-concave (NC-SC) and NC-C games. On the other hand, weakly-convex-weakly-concave games (Liu et al., 2021; Grimmer et al., 2023; Hajizadeh et al., 2024) and nonmonotone variational inequalities (Mertikopoulos et al., 2019; Diakonikolas et al., 2021; Lee and Kim, 2021; Pethick et al., 2022; Böhm, 2023) have been studied as well, focusing on extragradient and optimistic gradient (OGDA), rather than SGDA.

NC-PŁ games. Yang et al. (2020) gave the first convergence analysis of two-timescale alternating gradient descent ascent (GDA) for NC-PŁ games. Subsequently, Yang et al. (2022) extended the analysis to stochastic settings and introduced Smoothed-AGDA, attaining $O ( \kappa \ell \varepsilon ^ { - 2 } + \kappa ^ { 2 } \ell \sigma ^ { 2 } \varepsilon ^ { - 4 } )$ under the comparison conditions in Appendix E. Meanwhile, Huang et al. (2025) used momentum to attain an improved leading stochastic complexity of $\widetilde { \cal O } ( ( \kappa ^ { 3 } \ell \Delta \sigma + \kappa ^ { 4 } \sigma ^ { 3 } ) \varepsilon ^ { - 3 } )$ under Lipschitz sample gradient assumptions; Cai et al. (2026) obtained the same dependence on ε using Hessianvector products to correct momentum bias. For finite-sum NC-PŁ games, Chen et al. (2022) and Cho and Yun (2023) obtained better complexities through variance reduction and sampling without replacement, respectively. Finally, recent studies also established high-probability guarantees for Smoothed-AGDA (Laguel et al., 2024) and SGDA (Ha, 2026).

Two-timescale SGDA exposes a timescale ratio hyperparameter $\rho$ that controls the size of ascent steps relative to descent steps. Fiez and Ratliff (2021) showed that a sufficiently large timescale ratio ρ yields local linear convergence of simultaneous GDA to differential Stackelberg equilibria. For NC-SC games, Li et al. (2022) proved that $\rho = \Theta ( \kappa )$ suffices for local convergence, whereas GDA with $\rho = o ( \kappa )$ can fail to converge locally. On the other hand, existing global convergence guarantees for GDA and SGDA in NC-PŁ games rely on a timescale ratio $\rho = \hat { \Theta } ( \kappa ^ { 2 } )$ (Fiez et al., 2021; Yang et al., 2022; Ha, 2026). Cho and Yun (2023) showed that an excessively large timescale ratio can worsen convergence rates of GDA in strongly-convex-strongly-concave (SC-SC) games, and raised the question whether $\rho = \Theta ( \kappa ^ { 2 } )$ is necessary for global convergence in NC-PŁ games.

Lower bounds for structured nonconvex min-max games have mainly addressed the minimax complexity of first-order methods. For instance, in NC-SC games, Zhang et al. (2021) proved a deterministic $\Omega ( \sqrt { \kappa } \ell \Delta \varepsilon ^ { - 2 } )$ lower bound, and Li et al. (2021) proved a stochastic $\Omega ( \sqrt { \kappa } \bar { \ell } \Delta \varepsilon ^ { - 2 } +$ $\kappa ^ { 1 / 3 } \ell \Delta \sigma ^ { 2 } \varepsilon ^ { - 4 } )$ bound. More recently, Pan and Li (2026) proved a sharper deterministic NC-PŁ lower bound $\Omega ( \kappa \ell \Delta \dot { \varepsilon } ^ { - 2 } )$ ), establishing the minimax optimality of Smoothed-AGDA’s leading complexity. In this work, we derive SGDA-specific lower bounds that match existing upper bounds (Yang et al., 2022; Ha, 2026), and establish provable suboptimality of SGDA in NC-PŁ games.

## 3 PROBLEM SETUP

We study the following min-max optimization problem:

$$
\operatorname* { m i n } _ { \boldsymbol { x } \in \mathbb { R } ^ { d _ { \boldsymbol { x } } } } \operatorname* { m a x } _ { \boldsymbol { y } \in \mathbb { R } ^ { d _ { \boldsymbol { y } } } } f ( \boldsymbol { x } , \boldsymbol { y } ) ,\tag{3.1}
$$

where $f$ satisfies a smoothness assumption and $f ( x , \cdot )$ satisfies a PŁ inequality (Polyak, 1963).

Assumption 3.1 (Smoothness). The game $f : \mathbb { R } ^ { d _ { x } } \times \mathbb { R } ^ { d _ { y } }  \mathbb { R }$ is differentiable and ℓ-smooth:

$$
\begin{array} { r } { \| \nabla _ { x } f ( x _ { 1 } , y _ { 1 } ) - \nabla _ { x } f ( x _ { 2 } , y _ { 2 } ) \| \leq \ell \big ( \| x _ { 1 } - x _ { 2 } \| + \| y _ { 1 } - y _ { 2 } \| \big ) , \quad \forall x _ { 1 } , x _ { 2 } \in \mathbb { R } ^ { d _ { x } } , \ y _ { 1 } , y _ { 2 } \in \mathbb { R } ^ { d _ { y } } , } \end{array}
$$

$$
\begin{array} { r } { \| \nabla _ { y } f ( x _ { 1 } , y _ { 1 } ) - \nabla _ { y } f ( x _ { 2 } , y _ { 2 } ) \| \leq \ell \big ( \| x _ { 1 } - x _ { 2 } \| + \| y _ { 1 } - y _ { 2 } \| \big ) , \quad \forall x _ { 1 } , x _ { 2 } \in \mathbb { R } ^ { d _ { x } } , \ y _ { 1 } , y _ { 2 } \in \mathbb { R } ^ { d _ { y } } . } \end{array}
$$

We denote the outer objective of a game by $\Phi ( x ) : = \operatorname* { m a x } _ { \boldsymbol { u } \in \mathbb { R } ^ { d _ { \boldsymbol { y } } } } f ( x , \boldsymbol { y } )$ , and let $\Phi ^ { * } : = \operatorname* { i n f } _ { x \in \mathbb { R } ^ { d _ { x } } } \Phi ( x )$ We define the condition number $\kappa : = \ell / \mu \geq 1$ , where $\mu > 0$ is the PŁ parameter.

Assumption 3.2 (NC-PŁ). The outer objective satisfies $\Phi ( x ) < \infty$ for every $x \in \mathbb { R } ^ { d _ { x } }$ , and $Y ^ { \star } ( x ) : =$ arg $\operatorname* { m a x } _ { y \in \mathbb { R } ^ { d _ { y } } } f ( x , y )$ is nonempty for every $x \in \mathbb { R } ^ { d _ { x } }$ . There exists $\mu > 0$ such that the following inequality holds for each $x \in \mathbb { R } ^ { d _ { x } }$

$$
\frac { 1 } { 2 } \left\| \nabla _ { y } f ( x , y ) \right\| ^ { 2 } \geq \mu \big ( \Phi ( x ) - f ( x , y ) \big ) , \qquad \forall y \in \mathbb { R } ^ { d _ { y } } .\tag{3.2}
$$

Assumption 3.3 (Initialization). There exists $\Delta > 0$ and a universal constant $c _ { \mathrm { i n i t } } \geq 1$ such that

$$
0 \le \Phi ( x _ { 0 } ) - \Phi ^ { * } \le \Delta , \qquad 0 \le \Phi ( x _ { 0 } ) - f ( x _ { 0 } , y _ { 0 } ) \le c _ { \mathrm { { i n i t } } } \Delta .\tag{3.3}
$$

Throughout the paper, we make the following standard stochastic gradient estimator assumptions. Assumption 3.4 (Unbiasedness). The gradient estimators $G _ { x } , G _ { y }$ are measurable and use fresh independent samples ξ from a fixed distribution. For every $( x , y )$

$$
\mathbb { E } G _ { x } ( x , y ; \xi ) = \nabla _ { x } f ( x , y ) , \qquad \mathbb { E } G _ { y } ( x , y ; \xi ) = \nabla _ { y } f ( x , y ) .\tag{3.4}
$$

Assumption 3.5 (Bounded variance). For a fixed $\sigma \geq 0$ and every $( x , y )$

$$
\begin{array} { r } { \mathbb { E } \left\| G _ { x } ( x , y ; \xi ) - \nabla _ { x } f ( x , y ) \right\| ^ { 2 } \leq \sigma ^ { 2 } , \qquad \mathbb { E } \left\| G _ { y } ( x , y ; \xi ) - \nabla _ { y } f ( x , y ) \right\| ^ { 2 } \leq \sigma ^ { 2 } . } \end{array}\tag{3.5}
$$

Two-timescale SGDA. SGDA with a fixed timescale ratio $\rho \geq 1$ and step sizes $\eta _ { t } \geq 0$ follows:

$$
x _ { t + 1 } = x _ { t } - \frac { \eta _ { t } } { \rho } G _ { x } ( x _ { t } , y _ { t } ; \xi _ { t } ^ { x } ) , \qquad y _ { t + 1 } = y _ { t } + \eta _ { t } G _ { y } ( x _ { t } ^ { \prime } , y _ { t } ; \xi _ { t } ^ { y } ) ,\tag{3.6}
$$

where we use $x _ { t } ^ { \prime } = x _ { t + 1 }$ for $\mathrm { S G D A _ { A l t } } .$ , and $\boldsymbol { x } _ { t } ^ { \prime } = \boldsymbol { x } _ { t }$ for $\operatorname { S G D A } _ { \operatorname { S i m } }$

Minimax complexity. Let $\mathcal { C } ( \ell , \kappa , \Delta , \sigma )$ collect all NC-PŁ game instances that satisfy Assumptions 3.1–3.5 with $d _ { x } , d _ { y } < \infty$ . For each SGDA update order $\textsf { o } \in \ \{ { \mathrm { A l t } } , { \mathrm { S i m } } \}$ , we define the minimax convergence rate of SGDA with a fixed timescale ratio ρ and nonincreasing step sizes $\eta _ { t }$ as

$$
\mathcal { R } _ { T , \circ } ( \rho ) : = \operatorname* { i n f } _ { \substack { \eta _ { 0 } \ge \cdots \ge \eta _ { T - 2 } \ge 0 } } \operatorname* { s u p } _ { I \in \mathcal { C } ( \ell , \kappa , \Delta , \sigma ) } \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left. \nabla \Phi ( x _ { t } ) \right. ^ { 2 } .\tag{3.7}
$$

We write the version optimized over the timescale ratio as $\mathcal { R } _ { T , 0 } ^ { \star } : = \operatorname* { i n f } _ { \rho \geq 1 } \mathcal { R } _ { T , \circ } ( \rho )$ . We emphasize that we allow parameter-dependent tuning of timescale ratio $\rho ( \ell , \kappa , \Delta , \sigma , T )$ and step sizes $\eta _ { t } ( \ell , \kappa , \Delta , \sigma , T )$ . Finally, we define the corresponding iteration complexities as

$$
\begin{array} { r } { N _ { \varepsilon , \mathrm { o } } ( \rho ) : = \operatorname* { i n f } \big \{ T \in {  { \mathbb { N } } } _ { \geq 1 } : \mathscr { R } _ { T , \mathrm { o } } ( \rho ) \leq \varepsilon ^ { 2 } \big \} , \qquad N _ { \varepsilon , \mathrm { o } } ^ { \star } : = \operatorname* { i n f } \big \{ T \in {  { \mathbb { N } } } _ { \geq 1 } : \mathscr { R } _ { T , \mathrm { o } } ^ { \star } \leq \varepsilon ^ { 2 } \big \} . } \end{array}\tag{3.8}
$$

## 4 MAIN RESULTS

Our main result establishes a tight lower bound for two-timescale SGDA with a fixed timescale ratio $\rho \geq 1$ and non-increasing step size schedules η .

Theorem 4.1 (A lower bound for two-timescale SGDA in NC-PŁ games). There are universal constants $c > 0$ and $\kappa _ { 0 } \geq 2 ,$ , depending on $c _ { \mathrm { i n i t } }$ , such that,for every $\kappa \geq \kappa _ { 0 } , \ell > 0 , \Delta > 0 , \sigma \geq 0$ and integer $T \geq 1$

$$
\mathcal { R } _ { T , \mathrm { S i m } } ^ { \star } , \mathcal { R } _ { T , \mathrm { A l t } } ^ { \star } \geq c \operatorname* { m i n } \left\{ \ell \Delta , \operatorname* { m a x } \left\{ \frac { \kappa ^ { 2 } \ell \Delta } { T } , \sqrt { \frac { \kappa ^ { 4 } \ell \Delta \sigma ^ { 2 } } { T } } \right\} \right\} .\tag{4.1}
$$

Intuitively, the first term $\ell \Delta$ inside the minimum accounts for short horizons. Hence, in the usual small-error regime $\varepsilon ^ { 2 } \le c \ell \Delta$ , Theorem 4.1 and the matching upper bounds for $\mathbf { S G D A _ { A l t } }$ from Yang et al. (2022) and for $\operatorname { S G D A } _ { \operatorname { S i m } }$ in Theorem A.1 yield a tight complexity:

$$
N _ { \varepsilon , \mathrm { S i m } } ^ { \star } , N _ { \varepsilon , \mathrm { A l t } } ^ { \star } \asymp \frac { \kappa ^ { 2 } \ell \Delta } { \varepsilon ^ { 2 } } + \frac { \kappa ^ { 4 } \ell \Delta \sigma ^ { 2 } } { \varepsilon ^ { 4 } } .\tag{4.2}
$$

Table 1: Upper and lower bounds in NC-PŁ games. Our lower bounds match SGDA’s upper bounds after optimizing fixed timescale ratios and predetermined nonincreasing step sizes. With exact gradients, Theorem $6 . { \overset { \vartriangle } { 3 } }$ rules out uniform convergence for $\rho = o ( \kappa ^ { 2 } ) ; \Omega ( 1 )$ denotes nonvanishing stationarity error. MSGDA and AdaMSGDA use $0 < \varepsilon \leq \sigma .$ . † AdaMSGDA measures the average expected squared norm of $A _ { t } ^ { - 1 } \nabla \Phi ( x _ { t } )$ , where $A _ { t }$ is its primal preconditioner. $^ \ddag$ HCMM-2 shows only the variance-dependent cost for E $\| \nabla \Phi \| \leq \varepsilon$ , with Hessian variance at most $\sigma ^ { 2 } .$ . Appendix F gives the complete bounds and assumptions.
<table><tr><td>Algorithm</td><td>Citation</td><td>Feedback</td><td>Complexity / obstruction</td></tr><tr><td> $\mathrm { \ S G D A _ { A l t } }$ </td><td>Yang et al. (2022)</td><td>Bounded  $\sigma ^ { 2 }$ </td><td> $O ( \kappa ^ { 2 } \ell \varepsilon ^ { - 2 } + \kappa ^ { 4 } \ell \sigma ^ { 2 } \varepsilon ^ { - 4 } )$ </td></tr><tr><td> $\operatorname { S G D A } _ { \operatorname { S i m } }$ </td><td>Theorem A.1</td><td>Bounded  $\sigma ^ { 2 }$ </td><td> $O \dot { ( } \kappa ^ { 2 } \ell \varepsilon ^ { - 2 } + \kappa ^ { 4 } \ell \sigma ^ { 2 } \varepsilon ^ { - 4 } \big )$ </td></tr><tr><td>Smoothed-AGDA</td><td>Yang et al. (2022)</td><td> $\mathrm { B o u n d e d } \sigma ^ { 2 }$ </td><td> $\widetilde O ( \kappa \ell \varepsilon ^ { - 2 } + \kappa ^ { 2 } \ell \sigma ^ { 2 } \varepsilon ^ { - 4 } )$ </td></tr><tr><td>SPIDER-GDA</td><td>Chen et al. (2022)</td><td>Finite-sum</td><td> $O \dot { ( } n + \sqrt { n } \kappa ^ { 2 } \ell \varepsilon ^ { - 2 } )$ </td></tr><tr><td>SGDA-RR</td><td>Cho and Yun (2023)</td><td>Finite-sum</td><td> $O \dot { ( } n \kappa ^ { 2 } \ell \dot { \varepsilon } ^ { - 2 } + \sqrt { n } \kappa ^ { 3 } \ell \sigma \varepsilon ^ { - 3 } )$ </td></tr><tr><td>MSGDA</td><td>Huang et al. (2025)</td><td>Smooth samples</td><td> $\widetilde O ( ( \kappa ^ { 3 } \ell \sigma + \kappa ^ { 4 } \sigma ^ { 3 } ) \varepsilon ^ { - 3 } )$ </td></tr><tr><td>AdaMSGDA†</td><td>Huang et al. (2025)</td><td>Smooth samples</td><td> $\widetilde O ( ( \kappa ^ { 3 } \ell \sigma + \kappa ^ { 4 } \sigma ^ { 3 } ) \varepsilon ^ { - 3 } )$ </td></tr><tr><td>HCMM-2‡</td><td>Cai et al. (2026)</td><td>Hessian products</td><td> $\dot { O _ { \ell } ( \kappa ^ { 3 } \sigma ^ { 3 } \varepsilon ^ { - 3 } ) }$ </td></tr><tr><td> $\mathbf { S G D A } _ { \mathbf { A l t } } , \mathbf { S G D A } _ { \mathbf { S i m } }$ </td><td>Theorem 4.1</td><td>Bounded  $\sigma ^ { 2 }$ </td><td> $\Theta ( \kappa ^ { 2 } \ell \varepsilon ^ { - 2 } + \kappa ^ { 4 } \ell \sigma ^ { 2 } \varepsilon ^ { - 4 } )$ </td></tr><tr><td>Linear-span (NC-SC)</td><td>Zhang et al. (2021)</td><td>Deterministic</td><td> $\Omega ( \sqrt { \kappa } \ell \varepsilon ^ { - 2 } )$ </td></tr><tr><td>Linear-span (NC-SC)</td><td>Zhang et al. (2021)</td><td>Finite-sum</td><td> $\Omega ( n + \sqrt { n \kappa } \ell \varepsilon ^ { - 2 } )$ </td></tr><tr><td>Zero-respecting (NC-SC)</td><td>Li et al. (2021)</td><td>Bounded  $\sigma ^ { 2 }$ </td><td> $\Omega \dot { ( } \sqrt { \kappa } \ell \dot { \varepsilon } ^ { - 2 } + \kappa ^ { 1 / 3 } \ell \sigma ^ { 2 } \varepsilon ^ { - 4 } )$ </td></tr><tr><td>First-order (NC-PŁ)</td><td>Pan and Li (2026)</td><td>Deterministic</td><td> $\Omega ( \kappa { \ell } \varepsilon ^ { - 2 } )$ </td></tr><tr><td> $\operatorname { S G D A } _ { \operatorname { S i m } }$ </td><td>Li et al. (2022)</td><td>Deterministic</td><td>Local failure if  $\rho = o ( \kappa )$ </td></tr><tr><td> $\mathbf { S G D A } _ { \mathbf { A l t } } , \mathbf { S G D A } _ { \mathbf { S i m } }$ </td><td>Theorem 6.3</td><td>Deterministic</td><td> $\Omega ( 1 ) \ { \mathrm { i f } } \ \rho = o ( \kappa ^ { 2 } )$ </td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td> $\mathbf { S G D A } _ { \mathbf { A l t } } , \mathbf { S G D A } _ { \mathbf { S i m } }$ </td><td>Theorem 5.1</td><td>Bounded  $\sigma ^ { 2 }$ </td><td> $\Omega ( \rho \ell \varepsilon ^ { - 2 } + \kappa ^ { 4 } \ell \sigma ^ { 2 } \varepsilon ^ { - 4 } )$ </td></tr></table>

Suboptimality of SGDA. Theorem 4.1 establishes a provable suboptimality of SGDA in NC-PŁ games: it yields a tight sample complexity $\Theta ( \kappa ^ { 2 } \ell \Delta \dot { \varepsilon } ^ { - 2 } + \kappa ^ { 4 } \ell \Delta \sigma ^ { \dot { 2 } } \varepsilon ^ { - 4 } )$ for SGDA, which is suboptimal compared with the complexity upper bound $O ( \kappa \ell \Delta \varepsilon ^ { - 2 } + \kappa ^ { 2 } \ell \Delta \sigma ^ { 2 } \varepsilon ^ { - 4 } )$ of Smoothed-AGDA (Yang et al., 2022).<sup>1</sup> Hence, our result justifies the recent development of alternative methods such as Smoothed-AGDA (Yang et al., 2022) and momentum (Huang et al., 2025; Cai et al., 2026).

Optimal timescale ratio and step-size schedule. Notably, every existing analysis of two-timescale SGDA in NC-PŁ games relies on the choice of timescale ratio $\dot { \rho } = \Theta ( \kappa ^ { \overline { { 2 } } } )$ (Fiez et al., 2021; Yang et al., 2022; Cho and Yun, 2023; Ha, 2026). However, the question whether it is actually optimal remained open. Together with the matching upper bounds (Yang et al., 2022; Ha, 2026), Theorem 4.1 shows that the timescale ratio $\rho = \Theta ( \kappa ^ { 2 } )$ and the constant step size $\eta _ { t } \equiv c _ { \eta }$ min $\{ 1 / \ell , \sqrt { \Delta / ( \ell \sigma ^ { 2 } T ) } \}$ are optimal for two-timescale SGDA with a fixed timescale ratio and non-increasing step sizes.

In the following sections, we study two distinctive regimes of timescale ratios, namely the high-ratio regime $\rho = \bar { \Omega ( \kappa ^ { 2 } ) }$ and low-ratio regime $\rho = O ( \bar { \kappa ^ { 2 } } )$ , and explain the intuition behind the hard instances that establish Theorem 4.1. We defer the proof of Theorem 4.1 to Appendix A.1.

## 5 THE HIGH-RATIO LOWER BOUND

In this section, we establish the lower bound for large timescale ratios $\rho = \Omega ( \kappa ^ { 2 } )$ . In this high-ratio regime, smaller descent steps slow down the outer progress, whereas larger ascent steps increase noise-induced errors that accumulate on the outer variable. These complementary obstructions prevent SGDA from obtaining sharper convergence rates.

Theorem 5.1 (High-ratio lower bound). There are universal constants $c , c _ { \mathrm { h i g h } } > 0$ and $\kappa _ { \mathrm { h i g h } } \geq 2$ such that, for every choice ofproblem parameters $\ell , \Delta > 0 , \sigma \geq 0 \quad$ , and $\kappa \geq \kappa _ { \mathrm { h i g h } }$ , time horizon $T \in \mathbb { N } _ { \geq 1 }$ , and timescale ratio $\rho \geq c _ { \mathrm { h i g h } } \kappa ^ { 2 }$ , both $S G D A _ { S i m }$ and $S G D A _ { A l t }$ satisfy

$$
\mathcal { R } _ { T , \mathrm { S i m } } ( \rho ) , \mathcal { R } _ { T , \mathrm { A l t } } ( \rho ) \geq c \operatorname* { m i n } \left\{ \ell \Delta , \operatorname* { m a x } \left\{ \frac { \rho \ell \Delta } { T } , \sqrt { \frac { \kappa ^ { 4 } \ell \Delta \sigma ^ { 2 } } { T } } \right\} \right\} .\tag{5.1}
$$

Together with the matching upper bounds from Yang et al. (2022) and Theorem A.1, Theorem 5.1 shows that the smallest possible timescale ratio $\rho = \Theta ( \kappa ^ { 2 } )$ is optimal in the high-ratio regime, as larger ratios only increase the lower bound. This suggests that the standard choice $\rho = \Theta ( \kappa ^ { \widetilde { 2 } } )$ (Yang et al., 2022) might be optimal among all fixed ratios. We establish this optimality in Section 6 by ruling out a uniform convergence guarantee for timescale ratios of smaller order $\rho = o ( \kappa ^ { 2 } )$

In the following subsections, we construct two hard NC-PŁ game instances that yield Theorem 5.1.

## 5.1 THE INSTANCE FOR UNSTABLE STEP SIZES

Consider a simple linear quadratic game $f _ { \mathrm { u n s t a b l e } } : \mathbb { R } \times \mathbb { R } \to \mathbb { R }$ and its outer objective:

$$
f _ { \mathrm { u n s t a b l e } } ( x ; y ) = \frac { \ell + \mu } { 4 } ( x y - y ^ { 2 } ) , \qquad \Phi _ { \mathrm { u n s t a b l e } } ( x ) = \frac { \ell + \mu } { 1 6 } x ^ { 2 } .\tag{5.2}
$$

This game is ℓ-smooth and µ-strongly concave in y, and therefore, is an NC-PŁ game. For this game, an inner ascent step becomes unstable for $\eta _ { t } \geq 8 / \ell$ . The following result shows that such m initial unstable steps can push x away from its minimizer.

Lemma 5.2 (A lower bound for $f _ { \mathrm { u n s t a b l e } } )$ . There is a universal $c > 0$ such that, for $\ell , \Delta > 0 , \kappa \geq 2$ $\sigma \geq 0 , \rho \geq 1$ , integers $T \geq 1$ and $0 \leq m < T$ , and predetermined nonnegative non-increasing step sizes with $\eta _ { t } \geq 8 / \ell f o r t < m ,$ , each of $S G D A _ { S i m }$ and $S G D A _ { A l t }$ admits an initialization $( x _ { 0 } , y _ { 0 } )$ making f<sub>unstable</sub> with $\mu = \ell /$ κ an exact-gradient instance in $\mathcal { C } ( \ell , \kappa , \Delta , \sigma )$ with

$$
{ \frac { 1 } { T } } \sum _ { t = 0 } ^ { T - 1 } | \Phi _ { \mathrm { u n s t a b l e } } ^ { \prime } ( x _ { t } ) | ^ { 2 } \geq c { \frac { m + 1 } { T } } \ell \Delta .\tag{5.3}
$$

For $m + 1 \ge T / 2$ , Lemma 5.2 immediately proves Theorem 5.1 with the constant lower bound $c \ell \Delta$ We construct a distinct hard instance $f _ { \mathrm { s t a b l e } }$ for the other case m $, + 1 < T / 2$ in the next subsection.

## 5.2 THE INSTANCE FOR STABLE STEP SIZES

For the remaining case $m + 1 < T / 2$ , we design an instance $f _ { \mathrm { s t a b l e } }$ with outer coordinates $\boldsymbol { x } = \left( u , v \right)$

$$
f _ { \mathrm { s t a b l e } } ( \boldsymbol { u } , \boldsymbol { v } ; \boldsymbol { y } ) = g _ { \mathrm { c o r r i d o r } } ( \boldsymbol { u } ) + g _ { \sigma } ( \boldsymbol { v } , \boldsymbol { y } ) , \qquad \Phi _ { \mathrm { s t a b l e } } ( \boldsymbol { u } , \boldsymbol { v } ) = g _ { \mathrm { c o r r i d o r } } ( \boldsymbol { u } ) + \Phi _ { \sigma } ( \boldsymbol { v } ) ,\tag{5.4}
$$

where we use a scalar coordinate $u \in \mathbb { R }$ for the deterministic function $g _ { \mathrm { c o r r i d o r } } ,$ and define the outer objective $\Phi _ { \sigma } ( v ) : = \operatorname* { m a x } _ { y } g _ { \sigma } ( v , y )$ for the noisy function $g _ { \sigma } ( v , y )$ . Intuitively, the corridor function $g _ { \mathrm { c o r r i d o r } }$ limits progress from small descent steps, and the noisy function $g _ { \sigma }$ transfers the inner stochastic perturbations to the outer variable. The stable ascent steps sum to $\begin{array} { r } { \dot { S } = \sum _ { t = m } ^ { T - 2 } \eta _ { t } } \end{array}$

## 5.2.1 THE CORRIDOR FUNCTION g<sub>corridor</sub>

The following lower bound shows that the corridor function $g _ { \mathrm { c o r r i d o r } }$ keeps the remaining primal gradients constant and nonzero.

Lemma 5.3 (A lower bound for $g _ { \mathrm { c o r r i d o r } } )$ . For $\ell , \Delta > 0 , \rho \geq 1$ , integers $T \geq 1 , 0 \leq m < T$ and nonnegative steps with $\textstyle S = \sum _ { t = m } ^ { T - 2 } \eta _ { t }$ , there is an ℓ-smooth function g<sub>corridor</sub> : $\mathbb { R } \to \mathbb { R }$ with initial gap $g _ { \mathrm { c o r r i d o r } } ( 0 ) - \operatorname* { i n f } _ { u } g _ { \mathrm { c o r r i d o r } } ( u ) = \Delta$ . Gradient descent from $u _ { 0 } = 0$ with steps $\eta _ { t } / \rho$ sa $t i s f i e s g _ { \mathrm { c o r r i d o r } } ^ { \prime } ( u _ { t } ) < 0 \mathrm { { j } }$ for all $t < T$ and $g _ { \mathrm { c o r r i d o r } } ^ { \prime } ( u _ { t } ) = - G f o r m \le t < T$ , where $G > 0$ and $G ^ { 2 } \breve { \asymp } \operatorname* { m i n } \{ \ell \Delta , \rho \Delta / S \}$ . Consequentially,

$$
\frac { 1 } { T } \sum _ { t < T } | g _ { \mathrm { c o r r i d o r } } ^ { \prime } ( u _ { t } ) | ^ { 2 } \gtrsim \frac { T - m } { T } \operatorname* { m i n } \left\{ \ell \Delta , \frac { \rho \Delta } { S } \right\} .\tag{5.5}
$$

![](images/ac516ad08548a73dc2683cc2e2b37ca77c27cf74ad617737e738351d0846cfc2.jpg)  
(a) The corridor function

![](images/d935e0ded09912f135250b441789464db2ecf4e9ca6f1b39ede5cc2b272ea6e1.jpg)  
(b) Inner suboptimality

![](images/5fffa6de0cd16055d67024a2d359e8a2101e22d93ea50c4bd0e3b46f7d816a1d.jpg)  
(c) Mean squared outer gradient  
Figure 1: The competing bounds in $f _ { \mathrm { s t a b l e } }$ (schematic). Let m count the initial large steps and $S =$ $\scriptstyle \sum _ { t = m } ^ { \overline { { T } } - 2 } \eta _ { t }$ . (a) The corridor keeps gradient magnitude $G$ at $u _ { m } , \ldots , u _ { T - 1 }$ . (b) Gradient noise creates inner gap $\Phi _ { \sigma } ( v _ { t } ) - g _ { \sigma } ( v _ { t } , y _ { t } )$ . (c) Pink bands show contributions to $\begin{array} { r } { t ^ { - 1 } \sum _ { i < t } \| \nabla \Phi _ { \sigma } ( v _ { i } ) \| ^ { 2 } ; } \end{array}$ ; the initial increases are omitted. The dashed level is the bound’s scale $\kappa ^ { 4 } \ell \sigma ^ { 2 } S / ( \rho \dot { T } )$ for $S \geq C \rho / ( \ell \kappa )$

We illustrate the corridor function $g _ { \mathrm { c o r r i d o r } }$ in Figure 1 (a). Intuitively, it consists of an entry curve and a constant-slope segment, followed by an exit curve. The entry curve brings the first m updates to the corridor, and the remaining iterates travel its length $G S / \rho$ at slope $- G$ , using gap $G ^ { 2 } S / \bar { \rho }$ . The exit curve brings the slope to zero; with curvature of magnitude $\ell ,$ the entry and exit curves together use gap of order $G ^ { 2 } / \ell$ . Hence $G ^ { 2 } ( S / \rho + 1 / \ell ) \asymp \Delta$ gives the stated lower bound (Appendix C.2).

While Lemma 5.3 provides a lower bound for small cumulative steps $S ,$ it weakens as $S$ grows. We complement this by constructing the function $g _ { \sigma }$ which induces a lower bound that grows with S.

## 5.2.2 THE LASTING EFFECT OF INNER PERTURBATION

To construct a noisy function $g _ { \sigma }$ whose lower bound grows with the cumulative steps $S ,$ , we study an inner perturbation that causes a long-lasting outer error. To make our exposition clear, we represent the perturbation by a deterministic shift $\delta > 0$ in a simple linear-quadratic game $g _ { \delta } : \mathbb { R } \times \mathbb { R } ^ { \hat { 2 } } \to \mathbb { R } .$

$$
g _ { \delta } ( z ; a , b ) = \ell z a - \frac { \ell } { 2 \kappa } \big [ a ^ { 2 } + ( a + b + \delta ) ^ { 2 } \big ]\tag{5.6}
$$

with an outer variable z and inner variables a, $b .$ Here, the inner maximizer $\left( \kappa z , - \kappa z - \delta \right)$ gives outer objective $\Phi _ { g _ { \delta } } ( z ) = \ell \kappa z ^ { 2 } / 2 $ ; hence, the inner perturbation δ does not affect the outer objective. From the origin, however, an ascent step follows the negative derivative $\partial _ { a } g _ { \delta } = - \ell \delta / \kappa$ and makes the inner variable a negative. A descent step then follows the derivative $- \partial _ { z } g _ { \delta } = - \ell a > 0$ and moves the outer variable z away from zero, which is the minimizer of the outer objective $\Phi _ { g _ { \delta } }$ . Consequentially, the inner perturbation δ induces an outer error through GDA dynamics.

In the high-ratio regime $\rho = \Omega ( \kappa ^ { 2 } )$ , such an outer error takes longer to remove than to create. To see this clearly, let us measure the time by cumulative steps $\textstyle \sum _ { i < t } \eta _ { i }$ . The outer variable z moves slowly, so we may approximate its effect by fixing $z = 0$ . Then, the inner gradients follow $\partial _ { a } g _ { \delta } = - ( \dot { \ell } / \kappa ) ( 2 a + b \dot { + } \bar { \delta } )$ and $\partial _ { b } g _ { \delta } = - ( \ell / \kappa ) \overset { \cdot } ( a + \bar { b + \delta } )$ . In particular, the gradients start at $- \ell \delta / \kappa$ and remain negative while $| a | + | b |$ is small relative to δ. Therefore, a reaches a negative constant fraction of δ on the cumulative timescale $\kappa / \ell ;$ ; then it returns to zero as b approaches −δ. Substituting this response into $z _ { i + 1 } - z _ { i } = - \ell \eta _ { i } a _ { i } / \rho$ predicts the scale

$$
\underbrace { \frac { \ell \delta } { \rho } } _ { \mathrm { o u t e r ~ s p e e d } } \times \underbrace { \frac { \kappa } { \ell } } _ { \mathrm { f o r m a t i o n } \mathrm { t i m e } } = \underbrace { \frac { \kappa \delta } { \rho } } _ { \mathrm { e r r o r ~ s i z e } } .
$$

After the inner variables adapt enough to the perturbation δ, the inner maximizer gives the approximation $a \approx \kappa z$ , so we obtain an outer contraction $z _ { i + 1 } \approx ( 1 - \ell \kappa \eta _ { i } / \rho ) z _ { i }$ . This suggests that removing a fixed fraction of the error induced by an inner perturbation takes cumulative steps of order $\rho / ( \ell \kappa )$ a factor $\rho / \kappa ^ { 2 }$ longer than its formation time. The following lemma captures this intuition and bounds the error on the outer variable $z _ { t }$ , and specifies the interval over which this error persists.

Lemma 5.4 (Persistent error under a deterministic shift). There are universal constants $c , C , C _ { 0 } > 0$ such that,for $\ell , \delta > 0 , \kappa \geq 1 , \rho \geq C _ { 0 } \kappa ^ { 2 }$ , and anyfinite step-size sequence with $0 \leq \ell \eta _ { t } \leq c \kappa$ , both $S G D A _ { S i m }$ and $S G D A _ { A l t }$ with exact gradients on the linear quadratic game g<sub>δ</sub> given $b y \left( 5 . 6 \right)$ , initialized at $( 0 , 0 , 0 )$ , satisfy $| z _ { t } | \leq C \kappa \delta / \rho$ and $| a _ { t } | + | b _ { t } | \leq C \delta$ whenever $\begin{array} { r } { \breve { \sum } _ { i < t } \eta _ { i } \le c \rho / ( \dot { \ell } \kappa ) } \end{array}$ , and

$$
z _ { t } \geq c \frac { \kappa \delta } { \rho } \quad i f \quad C \frac { \kappa } { \ell } \leq \sum _ { i < t } \eta _ { i } \leq c \frac { \rho } { \ell \kappa } .\tag{5.7}
$$

## 5.2.3 THE NOISY FUNCTION $g _ { \sigma }$

We now construct the noisy function $g _ { \sigma }$ with its coordinate-wise components $g _ { \sigma , j } ,$ each of which implements $g _ { \delta }$ through a stochastic noise oracle. Specifically, we assign each update $j \in J : =$ $\{ 0 , \bar { , } . . . , T { - } \bar { 2 } \}$ a separate component $g _ { \sigma , j } : \mathbb { R } \times \mathbb { R } ^ { 3 } $ R with $y = ( a , b , s )$ , initialized at the origin in coordinates $v [ j ] , y [ j ] = ( a [ j ] , \dot { b } [ j ] , s [ j ] )$ . We denote its outer objective by $\Phi _ { \sigma , j }$ , choose the constants $K \asymp \kappa$ and $L \asymp \ell$ with fixed universal factors, and set $g _ { \sigma , j } \equiv 0 \mathrm { i f } \eta _ { j } \sigma = 0$

$$
g _ { \sigma } ( v , y ) = \sum _ { j \in J } { g _ { \sigma , j } ( v [ j ] ; y [ j ] ) } ,\tag{5.8}
$$

$$
g _ { \sigma , j } ( v ; a , b , s ) : = L v a - \frac { L } { 2 K } \left[ a ^ { 2 } + \left( a + b + \chi _ { j } ( a + b ) \delta _ { j } ( s ) \right) ^ { 2 } \right] ,\tag{5.9}
$$

$$
\delta _ { j } ( s ) : = \sqrt K \eta _ { j } \sigma \sin ^ { 2 } \biggl ( \frac { \pi s } { 2 \eta _ { j } \sigma } \biggr ) .\tag{5.10}
$$

The smooth cutoff $\chi _ { j } : \mathbb { R } \to [ 0 , 1 ]$ equals one along the trajectory segment used in Lemma $5 . 4 .$ preserving $g _ { \delta }$ with parameters $\dot { L } , \dot { K }$ and inner perturbation $\delta _ { j } ( s )$ . The cutoff function $\chi _ { j }$ and the choice of constants $K _ { i }$ , L ensure ℓ-smoothness and the inner $( \check { \ell } / \kappa ) { - } \mathrm { P } \mathrm { \mathbf { \Psi } }$ inequality (Appendix C.3.3).

Our stochastic oracle induces a gradient error $\pm \sigma$ in $s [ j ]$ , and SGDA turns that error into $s [ j ] =$ $\pm \eta _ { i } \sigma$ . Specifically, let $e ( y )$ be the unit vector in the $s [ j ]$ -coordinate for the smallest $j \in J$ with $s [ j ] = 0$ , or zero if none exists. With fresh independent uniform signs $\xi \in \{ - 1 , 1 \}$ , we introduce a stochastic oracle

$$
{ \cal G } _ { x } = \nabla _ { x } f _ { \mathrm { s t a b l e } } , \qquad { \cal G } _ { y } = \nabla _ { y } g _ { \sigma } + \sigma \xi e ( y ) .\tag{5.11}
$$

Since the derivative of $\delta _ { j }$ vanishes at $s = \pm \eta _ { j } \sigma , \operatorname { e a c h } s [ j ]$ stays fixed once activated, while the remaining s-coordinates stay at the origin; only $s [ j ]$ receives noise. The oracle has variance $\sigma ^ { 2 } \left\| e ( y ) \right\| ^ { 2 } \leq \sigma ^ { 2 }$ and the estimators in (5.11) are pointwise unbiased, so they satisfy Assumptions $3 . 4 \mathrm { - } 3 . 5$

For $j \geq m$ with enough remaining cumulative steps to cover $\asymp \rho / ( \ell \kappa )$ , Lemma 5.4 and curvature $\Phi _ { \sigma , j } ^ { \prime \prime } \asymp \ell \kappa$ give an outer gradient $\Phi _ { \sigma , i } ^ { \prime } ( v _ { t } [ j ] ) \gtrsim \ell \kappa ^ { 5 / 2 } \eta _ { j } \sigma / \rho .$ . Subsequent steps are at most $\eta _ { j }$ due to the non-increasing step size schedule, hence, the following lower bound holds:

$$
\sum _ { t < T } \mathbb { E } | \Phi _ { \sigma , j } ^ { \prime } ( v _ { t } [ j ] ) | ^ { 2 } \gtrsim \underbrace { \left( \frac { \ell \kappa ^ { 5 / 2 } \eta _ { j } \sigma } { \rho } \right) ^ { 2 } } _ { \mathrm { s q u a r e d ~ o u t e r ~ g r a d i e n t } } \times \underbrace { \frac { \rho } { \ell \kappa \eta _ { j } } } _ { \mathrm { n u m b e r ~ o f ~ i t e r a t e s } } = \frac { \ell \kappa ^ { 4 } \sigma ^ { 2 } \eta _ { j } } { \rho } .\tag{5.12}
$$

The next lemma captures this derivation, and lower bounds the outer gradient accumulation from a single inner perturbation.

Lemma 5.5 (Contribution of a single inner-gradient perturbation). There are universal constants $c , C , c _ { \mathrm { s t e p } } , c _ { \mathrm { h i g h } } > 0$ and $\kappa _ { \mathrm { h i g h } } \geq 2$ such that, for $\ell , \sigma > 0 , \kappa \geq \kappa _ { \mathrm { h i g h } } ,$ $\rho \geq$ max $\{ 1 , c _ { \mathrm { h i g h } } \kappa ^ { 2 } \}$ , an integer ${ \dot { T } } \geq 2 ,$ any predetermined step sizes $\eta _ { 0 } \geq \cdot \cdot \cdot \geq \eta _ { T - 2 } \geq 0$ and index $\bar { 0 } \leq j \leq \bar { T } - 2$ with $0 < \ell \eta _ { j } \le c _ { \mathrm { s t e p } } \kappa$ admit an ℓ-smooth $N C \cdot ( \ell / \kappa ) \ – P \cal { L }$ component $g _ { \sigma , j } : \mathbb { R } \times \mathbb { R } ^ { 3 } \to \mathbb { R } .$ Its outer objective $\begin{array} { r } { \Phi _ { \sigma , j } ^ { - } ( v ) : = \operatorname* { m a x } _ { y \in \mathbb { R } ^ { 3 } } g _ { \sigma , j } ( v ; y ) } \end{array}$ is quadratic with curvature $\tilde { \Phi } _ { \sigma , j } ^ { \prime \prime } \asymp$ ℓκ and initial gaps $\begin{array} { r } { \Phi _ { \sigma , j } ( 0 ) - \operatorname* { m i n } _ { v } \Phi _ { \sigma , j } ( v ) = \Phi _ { \sigma , j } ( 0 ) - g _ { \sigma , j } ( 0 ; 0 ) = 0 } \end{array}$ . Startingfrom the origin with one equiprobable $\pm \sigma$ error in the third inner gradient at update j and exact gradients otherwise, both $S G D A _ { S i m }$ and $S G D A _ { A l t }$ generate outer iterates $v _ { t } [ j ]$ satisfying

$$
\sum _ { t < T } \mathbb { E } | \Phi _ { \sigma , j } ^ { \prime } ( v _ { t } [ j ] ) | ^ { 2 } \geq c \frac { \ell \kappa ^ { 4 } \sigma ^ { 2 } \eta _ { j } } { \rho } \quad i f \quad \sum _ { t = j + 1 } ^ { T - 2 } \eta _ { t } \geq C \frac { \rho } { \ell \kappa } .\tag{5.13}
$$

## 5.2.4 COMBINING THE CORRIDOR FUNCTION g<sub>corridor</sub> AND NOISY FUNCTION $g _ { \sigma }$

With enough remaining cumulative steps, $S \geq C \rho / ( \ell \kappa )$ , we can sum Lemma 5.5, omitting late perturbations with insufficient subsequent cumulative steps; their total step size is $O ( \rho / ( \ell \kappa ) )$ . The corridor covers the other case of small remaining cumulative steps. The following lemma combines these bounds and captures a tradeoff in S (Appendix C.4).

Lemma 5.6 (Stable-step lower bound and admissibility). There are universal $c , c _ { \mathrm { h i g h } } > 0$ and $\kappa _ { \mathrm { h i g h } } \geq 2$ such that,for $\ell , \Delta > 0 , \sigma \geq 0 , \kappa \geq \kappa _ { \mathrm { h i g h } } , \rho \geq \operatorname* { m a x } \{ 1 , c _ { \mathrm { h i g h } } \kappa ^ { 2 } \}$ , an integer $T \geq 1 ,$ , and predetermined steps $\eta _ { 0 } \geq \cdot \cdot \cdot \geq \eta _ { T - 2 } \geq 0 ,$ , the following holds. Let m count the entries $\eta _ { t } \ge 8 / \ell$ and assume $m + 1 < T / 2$ . For each of $S G D A _ { S i m }$ and $S G D A _ { A l t } ,$ , the game $f _ { \mathrm { s t a b l e } }$ given by $_ { ( 5 . 4 ) }$ and afixed oracle can be chosen as an instance in $\mathcal { C } ( \ell , \kappa , \Delta , \sigma )$ with initial outer gap $\Delta$ and inner gap zero at the origin. Writing $\textstyle S = \sum _ { t = m } ^ { T - 2 }$ η<sub>t</sub> and $\rho \Delta / 0 : = + \infty$ , its iterates satisfy

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left. \nabla \Phi _ { \mathrm { s t a b l e } } ( u _ { t } , v _ { t } ) \right. ^ { 2 } \geq c \operatorname* { m i n } \left\{ \ell \Delta , \operatorname* { m a x } \left\{ \frac { \rho \Delta } { S } , \frac { \kappa ^ { 4 } \ell \sigma ^ { 2 } S } { \rho T } \right\} \right\} .\tag{5.14}
$$

Figure 1 illustrates the tradeoff suggested by Lemma $5 . 6 \colon$ increasing S weakens the corridor bound $\rho \Delta / S$ but strengthens the stochastic bound $\kappa ^ { 4 } \ell \sigma ^ { 2 } S / ( \rho T )$ . Applying the AM–GM inequality and $\dot { S } < 8 T / \ell$ to Lemma 5.6 proves Theorem 5.1 as follows:

$$
\operatorname* { m a x } \biggl \{ \frac { \rho \Delta } { S } , \frac { \ell \kappa ^ { 4 } \sigma ^ { 2 } S } { \rho T } \biggr \} \geq \operatorname* { m a x } \biggl \{ \frac { \rho \ell \Delta } { 8 T } , \sqrt { \frac { \ell \kappa ^ { 4 } \Delta \sigma ^ { 2 } } { T } } \biggr \} .\tag{5.15}
$$

## 6 LOW-RATIO LOWER BOUND

In the low ratio regime $\rho = O ( \kappa ^ { 2 } )$ , inner variables may fail to track their maximizer due to relatively small inner step sizes. In particular, Mourrat (2026) recently constructed a PŁ-PŁ game that exhibits a periodic orbit of a continuous-time GDA flow, suggesting that a PŁ inequality can fail to prevent limit cycles of GDA. Below, we realize a similar construction for discrete-time GDA in NC-PŁ games.

## 6.1 INNER PŁ DOES NOT PREVENT A LIMIT CYCLE

We start with a simple bilinear game that creates a periodic orbit for GDA:

$$
g ( u , a ) = \sqrt { 2 } u ^ { \mathsf { T } } a , \qquad u , a \in \mathbb { R } ^ { 2 } .\tag{6.1}
$$

For timescale ratio $\rho = 1 , { \mathrm { G D A } }$ flow oscillates in this game (Hsieh et al., 2021), but the infinite inner maximum for $u \ne ($ 0 violates NC-PŁ. Adding $- \left\| a \right\| ^ { 2 }$ gives the outer objective $\left\| u \right\| ^ { 2 } / 2$ yet damps the flow: $\ddot { u } + 2 \dot { u } + \dot { 2 } u = 0$ . Hence, we introduce $b \in \mathbb { R } ^ { 2 }$ and a residual $\dot { R ^ { \prime } } : \mathbb { R } ^ { 2 } \times \ddot { \mathbb { R } } ^ { 2 } \stackrel { } { \to } \dot { \mathbb { R } } ^ { 2 }$ and define

$$
\begin{array} { r } { g _ { \mathrm { c y c l e } } ( u , a , b ) = \sqrt { 2 } \boldsymbol { u } ^ { \intercal } \boldsymbol { a } - \left. \boldsymbol { a } \right. ^ { 2 } - \frac { 1 } { 2 } \left. \boldsymbol { R } ( \boldsymbol { a } , b ) \right. ^ { 2 } . } \end{array}\tag{6.2}
$$

Designing the residual $R ( a , b )$ We want the residual $R ( a , b )$ to restore a periodic motion while preserving an inner PŁ inequality and the quadratic outer objective $\left\| u \right\| ^ { 2 } / 2$ . If every a admits b that satisfies $\bar { R } ( a , b ) = 0$ , maximization over $( a , b )$ can cancel the residual and leave the outer objective unchanged. On the other hand, the inner PŁ inequality follows when the map $( a , b ) \mapsto ( { \sqrt { 2 } } a , R ( a , b ) )$ has a global smooth inverse with a uniformly bounded derivative (c.f. Lemma B.1). To specify the periodic motion we want $g _ { \mathrm { c y c l e } }$ to induce, let J denote counterclockwise rotation by $\pi / 2$ , and define

$$
u _ { \star } ( t ) = ( \cos t , \sin t ) ^ { \top } , \qquad a _ { \star } ( t ) = - J u _ { \star } ( t ) / \sqrt { 2 } , \qquad b _ { \star } ( t ) = u _ { \star } ( t ) / \sqrt { 2 } .\tag{6.3}
$$

Our goal is to design the residual $R ( a , b )$ for GDA to follow this path. Outer descent already has the required velocity along this path, since $\dot { u } = J u = - \sqrt { 2 } a$ . The inner variables must have velocities $\dot { a } = J a$ and $\dot { b } = - a$ . Hence, to obtain the desired inner motion, we can impose the restriction

$$
\begin{array} { r } { \nabla _ { a } \Big ( \frac { 1 } { 2 } \left\| R ( a , b ) \right\| ^ { 2 } \Big ) = - 2 a + J a , \qquad \nabla _ { b } \Big ( \frac { 1 } { 2 } \left\| R ( a , b ) \right\| ^ { 2 } \Big ) = a , } \end{array}\tag{6.4}
$$

on the circle $b = J a , \| a \| = 1 / { \sqrt { 2 } }$ traced by the motion $( a _ { \star } , b _ { \star } )$ . With these identities, the GDA equations agree with the target path. We choose further derivatives so that nearby trajectories approach the orbit, and use cutoffs for global smoothness without changing these dynamics (c.f. Appendix D.1).

Lemma 6.1 (An attracting periodic orbit of GDA flow in $g _ { \mathrm { c y c l e } } )$ . There are universal $C , \mu , h _ { \star } > 0$ and a smooth residual R such that the base game $g _ { \mathrm { c y c l e } } : \mathbb { R } ^ { 2 } \times \mathbb { R } ^ { 4 } \to \mathbb { R }$ in (6.2) is jointly C-smooth and affine in u, has outer objective $\begin{array} { r } { \Phi _ { g _ { \mathrm { c y c l e } } } ( u ) = \frac { 1 } { 2 } \left. u \right. ^ { 2 } } \end{array}$ , and satisfies the inner $\mu { - } P \mathbf { \mathcal { L } }$ inequality. The path (6.3), with $y _ { \star } = ( a _ { \star } , b _ { \star } )$ , traces a locally stable 2π-periodic orbit Γ of GDA flow. There is a compact neighborhood U ofΓ such that both $S G D A _ { S i m }$ and $S G D A _ { A l t } ,$ , started in U with exact gradients and equal primal and dual steps $h _ { t } \in [ 0 , h _ { \star } ]$ , satisfy $( u _ { t } , y _ { t } ) \in \mathcal { U }$ and $\begin{array} { r } { \left\| \nabla \Phi _ { g _ { \mathrm { c y c l e } } } ( u _ { t } ) \right\| = \| u _ { t } \| \geq \frac { 1 } { 2 } } \end{array}$ for all $t \geq 0$ , where $y _ { t } = ( a _ { t } , b _ { t } )$

## 6.2 FROM THE CYCLE TO THE LOW-RATIO LOWER BOUND

Lemma 6.1 shows that continuous-time GDA can be trapped in a limit cycle in $g _ { \mathrm { c y c l e } }$ . However, its discrete-time guarantee remains local and requires step sizes at most $h _ { \star }$ , which can be extremely small. The next lemma fills this gap by modifying the game, and shows that GDA can reach the orbit from a suitable initialization after any finite sequence of larger steps and remain nearby under subsequent small steps.

Lemma 6.2 (GDA can reach the orbit in a modified game $\tilde { g } _ { \mathrm { c y c l e } } )$ . There are universal $C , \mu > 0$ with the following property. For any finite sequence of steps $h _ { 0 } , \ldots , h _ { m - 1 } > h _ { \star }$ , including $m = 0 ;$ , we can modify the game $g _ { \mathrm { c y c l e } }$ in Lemma 6.1 to obtain $\tilde { g } _ { \mathrm { c y c l e } } : \mathbb { R } ^ { 2 } \times \mathbb { R } ^ { 5 } \to \mathbb { R }$ . This game is jointly C-smooth and affine in u, satisfies the inner $\mu { - } P \mathbf { \mathcal { L } }$ inequality, and has outer objective $\Phi _ { \tilde { g } _ { \mathrm { c y c l e } } } ( u ) = \left\| u \right\| ^ { 2 } / 2$ $\boldsymbol { W i t h } \ y = ( a , b , r )$ and $e _ { 1 } = ( 1 , 0 ) ^ { \mathsf { T } }$ , there is an initialization $( e _ { 1 } , y _ { 0 } )$ with inner gap at most C. From this initialization, both $G D A _ { S i m }$ and $G D A _ { A l t }$ , using exact gradients and equal primal and dual steps, keep $u _ { t } = e _ { 1 } f o r 0 \le t \le m$ and reach $y _ { m } = ( y _ { \star } ( 0 ) , 0 )$ . For every continuation with $h _ { t } \in [ 0 , h _ { \star } ]$ each algorithmfollows its g<sub>cycle</sub> dynamics in $( u , a , b )$ , while r remains zero.

We now rescale $\tilde { g } _ { \mathrm { c y c l e } }$ to satisfy ℓ-smoothness and the inner $( \ell / \kappa ) { \ - } \mathrm { P f }$ inequality. We choose $f _ { \mathrm { c y c l e } } ( x , y ) = A \tilde { g } _ { \mathrm { c y c l e } } ( x / s _ { x } , y / s _ { y } )$ with positive scales satisfying $s _ { y } ^ { 2 } = \rho s _ { x } ^ { 2 }$ , which converts both updates at ratio $\rho$ to equal-step dynamics. The scales in Appendix $\mathrm { A } . 3$ make $f _ { \mathrm { c y c l e } }$ ℓ-smooth with inner PŁ constant of order $\ell / \sqrt { \rho } ,$ , so $\rho \le c _ { \mathrm { l o w } } \kappa ^ { 2 }$ ensures inner PŁ at least $\ell / \kappa$ . They also meet the initial-gap budgets and give $\| \nabla \Phi _ { \mathrm { c y c l e } } ( x _ { t } ) \| ^ { 2 } \gtrsim \ell \Delta$ at every iterate, proving the following theorem.

Theorem ${ \bf 6 . 3 }$ (Necessity of an order- $\cdot \kappa ^ { 2 }$ fixed ratio). There are universal $c , c _ { \mathrm { l o w } } > 0$ and $\kappa _ { \mathrm { l o w } } \geq 2$ such that,for every $\kappa \geq \kappa _ { \mathrm { l o w } } , \ell , \Delta > 0 , \sigma \geq 0 ,$ , integer $T \geq 1 , 1 \leq \rho \leq c _ { \mathrm { l o w } } \kappa ^ { 2 }$ , and predetermined nonnegative non-increasing step sizes, each $o f S G D A _ { S i m }$ and $S G D A _ { A l t }$ admits a deterministic instance $f _ { \mathrm { c y c l e } } i n \mathcal { C } ( \ell , \kappa , \Delta , \sigma )$ with $d _ { x } = 2 , d _ { y } = 5 ,$ , a strongly convex quadratic outer objective, and

$$
\operatorname* { m i n } _ { 0 \leq t < T } \| \nabla \Phi ( x _ { t } ) \| ^ { 2 } \geq c \ell \Delta .\tag{6.5}
$$

Crucially, the theorem answers the question raised by Cho and Yun (2023) affirmatively: timescale ratio $\rho = \Omega ( \kappa ^ { 2 } )$ is necessary for SGDA to converge in NC-PŁ games. In particular, it shows that smaller timescale ratios $\rho = { \dot { o } } ( \kappa ^ { 2 } )$ cannot provide convergence guarantees in general, even with deterministic gradients and non-increasing step sizes. Together with Theorem 5.1, it yields our tight main lower bound in Theorem 4.1.

## 7 CONCLUSION

In this work, we established the first tight lower bounds of two-timescale SGDA with a fixed timescale ratio and nonincreasing step sizes in NC-PŁ games. The lower bounds explain the fundamental limitation of two-timescale SGDA: small ratios can keep outer gradients bounded away from zero, while large ratios leave a tradeoff between outer progress and errors induced by inner noise. Our negative result justifies the recent line of work developing alternative methods, such as Smoothed-AGDA (Yang et al., 2022) and methods based on momentum (Huang et al., 2025; Cai et al., 2026).

On the other hand, our analysis only applies to SGDA with a fixed timescale ratio and nonincreasing step sizes; therefore, it does not rule out nonmonotone or signed step size schedules (Altschuler and Parrilo, 2025; Shugart and Altschuler, 2025) that might improve SGDA’s complexity. An additional interesting direction to explore would be to extend the ideas behind our lower bounds to nonconvex-concave problems. We leave these two questions as intriguing future work.

## AI USE STATEMENT

Generative AI tools were used to brainstorm and refine theoretical constructions, formulate and check proof strategies, survey relevant literature, draft and edit mathematical exposition, and prepare and check the LAT<sub>E</sub>X source. The authors are responsible for the final content, including all AI-assisted text, claims, proofs, and artifacts.

## REPRODUCIBILITY STATEMENT

All assumptions and algorithms are stated in Section 3. Appendix A contains the proofs of the main theorem, the high- and low-ratio lower bounds, and the matching $\mathbf { S G D A _ { S i m } }$ upper bound. Appendices C and D prove the construction lemmas. Together, these proofs give the construction formulas, parameter choices, oracle definitions, initialization checks, and arguments for $\mathbf { S G D A } _ { \mathrm { S i m } }$ and $\mathbf { S G D A _ { A l t } }$

## REFERENCES

Jason M. Altschuler and Pablo A. Parrilo. Acceleration by stepsize hedging: Silver Stepsize Schedule for smooth convex optimization. Mathematical Programming, 213:1105–1118, 2025. doi: 10.1007/ s10107-024-02164-2. URL https://doi.org/10.1007/s10107-024-02164-2.

Martino Bernasconi, Matteo Castiglioni, Andrea Celli, and Alexandros Hollender. The complexity of min-max optimization for quadratic polynomials. arXiv preprint arXiv:2606.17000, 2026. URL https://arxiv.org/abs/2606.17000.

Axel Böhm. Solving nonconvex-nonconcave min-max problems exhibiting weak Minty solutions. Transactions on Machine Learning Research, 2023. URL https://openreview.net/ forum?id=Gp0pHyUyrb.

Haoyuan Cai, Sulaiman A. Alghunaim, and Ali H. Sayed. Accelerated stochastic min-max optimization based on bias-corrected momentum. arXiv preprint arXiv:2406.13041v3, 2026. URL https://arxiv.org/abs/2406.13041v3.

Lesi Chen, Boyuan Yao, and Luo Luo. Faster stochastic algorithms for minimax optimization under Polyak–Łojasiewicz condition. In Advances in Neural Information Processing Systems, volume 35, 2022. URL https://arxiv.org/abs/2307.15868.

Hanseul Cho and Chulhee Yun. SGDA with shuffling: Faster convergence for nonconvex-PŁ minimax optimization. In International Conference on Learning Representations, 2023. URL https://iclr.cc/virtual/2023/poster/11501.

Constantinos Daskalakis, Stratis Skoulakis, and Manolis Zampetakis. The complexity of constrained min-max optimization. In Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory of Computing, 2021. doi: 10.1145/3406325.3451125. URL https://arxiv.org/abs/2009. 09623.

Jelena Diakonikolas, Constantinos Daskalakis, and Michael I. Jordan. Efficient methods for structured nonconvex-nonconcave min-max optimization. In Proceedings ofthe 24th International Conference on Artificial Intelligence and Statistics, volume 130 of Proceedings of Machine Learning Research, pages 2746–2754. PMLR, 2021. URL https://proceedings.mlr.press/ v130/diakonikolas21a.html.

Tanner Fiez and Lillian J. Ratliff. Local convergence analysis of gradient descent ascent with finite timescale separation. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=AWOSz\_mMAPx.

Tanner Fiez, Lillian J. Ratliff, Eric Mazumdar, Evan Faulkner, and Adhyyan Narang. Global convergence to local minmax equilibrium in classes of nonconvex zero-sum games. In Advances in Neural Information Processing Systems, volume 34, 2021.

Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems, volume 27, 2014. URL https://arxiv.org/abs/1406. 2661.

Benjamin Grimmer, Haihao Lu, Pratik Worah, and Vahab Mirrokni. The landscape of the proximal point method for nonconvex-nonconcave minimax optimization. Mathematical Programming, 201(1–2):373–407, 2023. doi: 10.1007/s10107-022-01910-8. URL https://doi.org/10. 1007/s10107-022-01910-8.

Junsoo Ha. High probability convergence guarantees of stochastic gradient descent ascent in structured nonconvex min-max games. In Proceedings of Thirty Ninth Conference on Learning Theory, volume 336 of Proceedings of Machine Learning Research, pages 3005–3075, 2026. URL https://proceedings.mlr.press/v336/ha26a.html.

Saeed Hajizadeh, Haihao Lu, and Benjamin Grimmer. On the linear convergence of extragradient methods for nonconvex-nonconcave minimax problems. INFORMS Journal on Optimization, 6 (1):19–31, 2024. doi: 10.1287/ijoo.2022.0004. URL https://doi.org/10.1287/ijoo. 2022.0004.

Ya-Ping Hsieh, Panayotis Mertikopoulos, and Volkan Cevher. The limits of min-max optimization algorithms: Convergence to spurious non-critical sets. In Proceedings ofthe 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pages 4337–4348, 2021. URL https://proceedings.mlr.press/v139/hsieh21a. html.

Feihu Huang, Chunyu Xuan, Xinrui Wang, Siqi Zhang, and Songcan Chen. Enhanced adaptive gradient algorithms for nonconvex-PŁ minimax optimization. In Proceedings ofthe 28th Interna tional Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings of Machine Learning Research, pages 3439–3447, 2025. URL https://proceedings.mlr.press/ v258/huang25d.html.

Yassine Laguel, Yasa Syed, Necdet Serhat Aybat, and Mert Gürbüzbalaban. High-probability complexity bounds for stochastic non-convex minimax optimization. In Advances in Neural Information Processing Systems, volume 37, 2024. URL https://arxiv.org/abs/2405. 14130.

Sucheol Lee and Donghwan Kim. Fast extra gradient methods for smooth structured nonconvexnonconcave minimax problems. In Advances in Neural Information Processing Systems, volume 34, pages 22588–22600, 2021. URL https://proceedings.neurips.cc/paper/2021/ hash/be767243ca8f574c740fb4c26cc6dceb-Abstract.html.

Haochuan Li, Yi Tian, Jingzhao Zhang, and Ali Jadbabaie. Complexity lower bounds for nonconvexstrongly-concave min-max optimization. In Advances in Neural Information Processing Systems, volume 34, 2021. URL https://proceedings.neurips.cc/paper\_files/paper 2021/hash/0e105949d99a32ca1751703e94ece601-Abstract.html.

Haochuan Li, Farzan Farnia, Subhro Das, and Ali Jadbabaie. On convergence of gradient descent ascent: A tight local analysis. In Proceedings ofthe 39th International Conference on Machine Learning, volume 162 of Proceedings ofMachine Learning Research, pages 12717–12740, 2022. URL https://proceedings.mlr.press/v162/li22e.html.

Tianyi Lin, Chi Jin, and Michael I. Jordan. On gradient descent ascent for nonconvex-concave minimax problems. In Proceedings ofthe 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 6083–6093, 2020. URL https://proceedings.mlr.press/v119/lin20a.html.

Tianyi Lin, Chi Jin, and Michael I. Jordan. Two-timescale gradient descent ascent algorithms for nonconvex minimax optimization. Journal ofMachine Learning Research, 26(11):1–45, 2025. URL https://jmlr.org/papers/v26/22-0863.html.

Mingrui Liu, Hassan Rafique, Qihang Lin, and Tianbao Yang. First-order convergence theory for weakly-convex-weakly-concave min-max problems. Journal of Machine Learning Research, 22 (169):1–34, 2021. URL https://jmlr.org/papers/v22/20-533.html.

Qihao Liu, Luoxin Ye, Wufei Ma, Yu-Cheng Chou, and Alan Yuille. Generative adversarial reasoner: Enhancing LLM reasoning with adversarial reinforcement learning. In International Conference on Learning Representations, 2026. URL https://arxiv.org/abs/2512.16917.

Panayotis Mertikopoulos, Bruno Lecouat, Houssam Zenati, Chuan-Sheng Foo, Vijay Chandrasekhar, and Georgios Piliouras. Optimistic mirror descent in saddle-point problems: Going the extra (gradient) mile. In International Conference on Learning Representations, 2019. URL https: //openreview.net/forum?id=Bkg8jjC9KQ.

Jean-Christophe Mourrat. PŁ conditions do not guarantee convergence of gradient descent-ascent dynamics. arXiv preprint arXiv:2602.16517, 2026. URL https://arxiv.org/abs/2602. 16517.

John Nash. Non-cooperative games. Annals ofMathematics, 54(2):286–295, 1951. doi: 10.2307/ 1969529.

Maher Nouiehed, Maziar Sanjabi, Tianjian Huang, Jason D. Lee, and Meisam Razaviyayn. Solving a class of non-convex min-max games using iterative first order methods. In Advances in Neural Information Processing Systems, volume 32, 2019.

Siyu Pan and Jiajin Li. Lower bounds for nonconvex-PŁ minimax optimization. arXiv preprint arXiv:2608.26799, 2026. URL https://arxiv.org/abs/2608.26799.

Thomas Pethick, Puya Latafat, Panagiotis Patrinos, Olivier Fercoq, and Volkan Cevher. Escaping limit cycles: Global convergence for constrained nonconvex-nonconcave minimax problems. In International Conference on Learning Representations, 2022. URL https://openreview. net/forum?id=2\_vhkAMARk.

Boris T. Polyak. Gradient methods for minimizing functionals. USSR Computational Mathematics and Mathematical Physics, 3(4):864–878, 1963. doi: 10.1016/0041-5553(63)90382-3.

Yuzuru Sato, Eizo Akiyama, and J. Doyne Farmer. Chaos in learning a simple two-person game. Proceedings of the National Academy of Sciences, 99(7):4748–4751, 2002. doi: 10.1073/pnas. 032086299. URL https://pmc.ncbi.nlm.nih.gov/articles/PMC123719/.

Henry Shugart and Jason M. Altschuler. Negative stepsizes make Gradient-Descent-Ascent converge. arXiv preprint arXiv:2505.01423, 2025. doi: 10.48550/arXiv.2505.01423. URL https:// arxiv.org/abs/2505.01423.

Aman Sinha, Hongseok Namkoong, and John Duchi. Certifying some distributional robustness with principled adversarial training. In International Conference on Learning Representations, 2018. URL https://arxiv.org/abs/1710.10571v4.

Kiran K. Thekumparampil, Prateek Jain, Praneeth Netrapalli, and Sewoong Oh. Efficient algorithms for smooth minimax optimization. In Advances in Neural Information Processing Systems, volume 32, 2019. URL https://arxiv.org/abs/1907.01543.

Zaiyan Xu, Sushil Vemuri, Kishan Panaganti, Dileep Kalathil, Rahul Jain, and Deepak Ramachandran. Robust LLM alignment via distributionally robust direct preference optimization. In Advances in Neural Information Processing Systems, volume 38, 2025. URL https://arxiv.org/abs/ 2502.01930.

Junchi Yang, Negar Kiyavash, and Niao He. Global convergence and variance reduction for a class of nonconvex-nonconcave minimax problems. In Advances in Neural Information Processing Systems, volume 33, 2020. URL https://papers.neurips.cc/paper\_files/paper/2020/ hash/0cc6928e741d75e7a92396317522069e-Abstract.html.

Junchi Yang, Antonio Orvieto, Aurélien Lucchi, and Niao He. Faster single-loop algorithms for minimax optimization without strong concavity. In Proceedings ofthe 25th International Conference on Artificial Intelligence and Statistics, volume 151 of Proceedings of Machine Learning Research, pages 5485–5517, 2022. URL https://proceedings.mlr.press/v151/yang22b. html.

Siqi Zhang, Junchi Yang, Cristóbal Guzmán, Negar Kiyavash, and Niao He. The complexity of nonconvex-strongly-concave minimax optimization. In Proceedings of the Thirty-Seventh Conference on Uncertainty in Artificial Intelligence, volume 161 of Proceedings of Machine Learning Research, pages 482–492, 2021. URL https://proceedings.mlr.press/ v161/zhang21c.html.

## APPENDIX CONTENTS

A Proof of main result 15   
A.1 Proof of Theorem 4.1 15   
A.2 Proof of Theorem 5.1 15   
A.3 Proof of Theorem 6.3 16   
A.4 Upper bound for SGDA 17   
B Preliminary lemmas 19   
B.1 NC-PŁ games with a predetermined outer function 19   
B.2 NC-PŁ geometry 19   
C Proofs for the high-ratio lower bound 20   
C.1 Proof of the lower bound for f<sub>unstable</sub> 20   
C.2 Construction of g<sub>corridor</sub> 21   
C.3 Construction of $g _ { \sigma }$ 22   
C.4 Proof of the lower bound for f<sub>stable</sub> 28   
D Proofs for the low-ratio lower bound 29   
D.1 Construction of g<sub>cycle</sub> 29   
D.2 Construction of g˜<sub>cycle</sub> 32   
E Comparison with Smoothed-AGDA 34   
E.1 Initialization budget of the hard instances 37   
F Complexity calculations for Table 1 37   
F.1 SGDA and Smoothed-AGDA 38   
F.2 Finite-sum upper bounds 38   
F.3 MSGDA and AdaMSGDA 39   
F.4 HCMM-2 40   
F.5 Lower bounds and the nonconvergence entries 41

## A PROOF OF MAIN RESULT

We first combine the two ratio regimes to prove Theorem 4.1. We then prove Theorems 5.1 and 6.3 using the construction lemmas from Sections 5 and 6, and establish the matching upper bound for $\operatorname { S G D A } _ { \operatorname { S i m } }$

## A.1 PROOF OF THEOREM 4.1

Theorem 4.1 (A lower bound for two-timescale SGDA in NC-PŁ games). There are universal constants $c > 0$ and $\kappa _ { 0 } \geq 2 ,$ , depending on $c _ { \mathrm { i n i t } }$ , such that,for every $\kappa \geq \kappa _ { 0 } , \ell > 0 , \Delta > 0 , \sigma \geq 0$ and integer $T \geq 1$

$$
\mathcal { R } _ { T , \mathrm { S i m } } ^ { \star } , \mathcal { R } _ { T , \mathrm { A l t } } ^ { \star } \geq c \operatorname* { m i n } \left\{ \ell \Delta , \operatorname* { m a x } \left\{ \frac { \kappa ^ { 2 } \ell \Delta } { T } , \sqrt { \frac { \kappa ^ { 4 } \ell \Delta \sigma ^ { 2 } } { T } } \right\} \right\} .\tag{4.1}
$$

Proof. We first choose the construction constants so that the two ratio ranges overlap. The fixed cycle and its extension in Lemma 6.2 determine a universal inner-PŁ constant $\mu .$ . Let $C _ { \mathrm { l o w } }$ be the smoothness-normalization constant chosen in Appendix $\operatorname { A . 3 ; }$ the low-ratio construction then applies up $\textnormal { \em o c } _ { \mathrm { l o w } } = ( \mu / C _ { \mathrm { l o w } } ) ^ { 2 }$ . These constants are fixed before we calibrate the noise component. Let $\bar { C } _ { \mathrm { r e s p } }$ be the ratio constant $C _ { 0 }$ in Lemma $5 . 4$ , enlarged as in Appendix C.3.3, let $c _ { 0 } , C _ { 0 }$ be the constants from Lemma C.1, and choose

$$
0 < \gamma \leq \operatorname* { m i n } \left\{ \sqrt { \frac { c _ { \mathrm { l o w } } } { C _ { \mathrm { r e s p } } } } , \frac { c _ { 0 } } { 2 C _ { 0 } } \right\} , \qquad \theta : = \frac { \gamma } { c _ { 0 } } .
$$

These choices calibrate only the high-ratio block: they give (C.23) and $c _ { \mathrm { h i g h } } = \gamma ^ { 2 } C _ { \mathrm { r e s p } } \leq c _ { \mathrm { l o w } }$ Thus its ratio range begins no later than the endpoint of the low-ratio range. The low-ratio constants do not depend on γ or θ.

Choose $\kappa _ { 0 }$ above both theorem thresholds. In the high-ratio bound, $\rho \ell \Delta / T \geq c _ { \mathrm { h i g h } } \kappa ^ { 2 } \ell \Delta / T ;$ the stochastic term already matches (4.1). The low-ratio bound supplies the cap cℓ∆. Since the ranges cover every $\rho \geq 1$ , taking the ratio infimum proves (4.1).

For $\varepsilon ^ { 2 } \le c \ell \Delta$ with sufficiently small universal $c > 0 .$ , inversion gives the lower complexity bound in (4.2). The matching constant-step upper bounds cited in Section 4 complete the proof. □

## A.2 PROOF OF THEOREM 5.1

Theorem 5.1 (High-ratio lower bound). There are universal constants $c , c _ { \mathrm { h i g h } } > 0$ and $\kappa _ { \mathrm { h i g h } } \geq 2$ such that, for every choice ofproblem parameters $\ell , \Delta > 0 , \sigma \geq 0 \quad$ , and $\kappa \geq \kappa _ { \mathrm { h i g h } }$ , time horizon $T \in \mathbb { N } _ { > 1 }$ , and timescale ratio $\rho \geq c _ { \mathrm { h i g h } } \kappa ^ { 2 }$ , both $S G D A _ { S i m }$ and $S G D A _ { A l t }$ satisfy

$$
\mathcal { R } _ { T , \mathrm { S i m } } ( \rho ) , \mathcal { R } _ { T , \mathrm { A l t } } ( \rho ) \geq c \operatorname* { m i n } \left\{ \ell \Delta , \operatorname* { m a x } \left\{ \frac { \rho \ell \Delta } { T } , \sqrt { \frac { \kappa ^ { 4 } \ell \Delta \sigma ^ { 2 } } { T } } \right\} \right\} .\tag{5.1}
$$

Proof. For each of $\operatorname { S G D A } _ { \operatorname { S i m } }$ and $\mathrm { S G D A _ { A l t } }$ with predetermined non-increasing step sizes, let m count the initial excessively large steps $\ell \eta _ { t } \geq 8$ and set $\begin{array} { r } { S = \sum _ { t = m } ^ { T - 2 } \eta _ { t } . \mathrm { I f } m + 1 \stackrel { \mathrm { \tiny ~ \cdot ~ } } { \ge } T / 2 , } \end{array}$ Lemma 5.2 gives an average squared outer gradient at least $c \ell \Delta$ on $f _ { \mathrm { u n s t a b l e } } ,$ proving the capped bound. Otherwise, Lemma 5.6 applies to $f _ { \mathrm { s t a b l e } }$ and also gives the cap when $S = 0$ . For $S > 0$ , the remaining steps satisfy $S \le 8 \bar { T } / \ell$ , while the maximum of two nonnegative numbers is at least their geometric mean. Hence

$$
\operatorname* { m a x } \biggl \{ \frac { \rho \Delta } { S } , \frac { \kappa ^ { 4 } \ell \sigma ^ { 2 } S } { \rho T } \biggr \} \geq \operatorname* { m a x } \biggl \{ \frac { \rho \ell \Delta } { 8 T } , \sqrt { \frac { \kappa ^ { 4 } \ell \Delta \sigma ^ { 2 } } { T } } \biggr \} .\tag{A.1}
$$

Substituting into (5.14) and adjusting the universal constant proves (5.1). In each case, the instance is chosen after fixing the schedule. Taking the instance supremum and then the schedule infimum proves the theorem. □

## A.3 PROOF OF THEOREM 6.3

Theorem 6.3 (Necessity of an order- $\cdot \kappa ^ { 2 }$ fixed ratio). There are universal $c , c _ { \mathrm { l o w } } > 0$ and $\kappa _ { \mathrm { l o w } } \geq 2$ such that,for every $\kappa \geq \kappa _ { \mathrm { l o w } } , \ell , \Delta > 0 , \sigma \geq 0 ,$ , integer $T \geq 1 , 1 \leq \rho \leq c _ { \mathrm { l o w } } \kappa ^ { 2 }$ , and predetermined nonnegative non-increasing step sizes, each of $S G D A _ { S i m }$ and $S G D A _ { A l t }$ admits a deterministic instance $f _ { \mathrm { c y c l e } } i n \mathcal { C } ( \ell , \kappa , \Delta , \sigma )$ with $d _ { x } = 2 , d _ { y } = 5 ,$ , a strongly convex quadratic outer objective, and

$$
\operatorname* { m i n } _ { 0 \leq t < T } \left\| \nabla \Phi ( x _ { t } ) \right\| ^ { 2 } \geq c \ell \Delta .\tag{6.5}
$$

Proof. We rescale the fixed-dimensional game while retaining its strongly convex quadratic outer objective, then verify smoothness, inner $\mathrm { P E } ,$ , and the initial-gap budgets.

Let $C \geq 1$ bound the smoothness constant and initial inner gap in Lemma $6 . 2 \AA$ . Its construction also gives $\lVert \nabla _ { u } \tilde { g } _ { \mathrm { c y c l e } } ( e _ { 1 } , y _ { 0 } ) \rVert \leq 1$ , as verified in Appendix D.2. Choose a universal $C _ { \mathrm { l o w } } \geq 2 C$ , and fix $0 < \zeta \leq \operatorname* { m i n } \{ 1 / 2 , c _ { \mathrm { i n i t } } / ( 3 C ) \}$ . This choice also covers the regularized initial gap in Appendix E.1. For a ratio $\rho ,$ set

$$
A : = \frac { 2 \zeta \Delta } { \sqrt { \rho } } , \qquad s _ { x } ^ { 2 } : = \frac { 2 C _ { \mathrm { l o w } } \zeta \Delta } { \ell \rho } , \qquad s _ { y } ^ { 2 } : = \frac { 2 C _ { \mathrm { l o w } } \zeta \Delta } { \ell } .\tag{A.2}
$$

For a fixed horizon and predetermined non-increasing step sizes $( \eta _ { t } )$ , put $h _ { t } = \ell \eta _ { t } / ( C _ { \mathrm { l o w } } \sqrt { \rho } )$ . Let m count the initial steps with $h _ { t } > h _ { \star } . \ \mathrm { A p p l y }$ Lemma 6.2 to those steps, writing $\tilde { g } _ { \mathrm { c y c l e } }$ for its game. For both $\mathbf { S G D A _ { S i m } }$ and $\mathbf { S G D A _ { A l t } }$ , the lemma keeps $u _ { t } ~ = ~ e _ { 1 }$ through time m and then identifies the continuation with the small-step dynamics of $g _ { \mathrm { c y c l e } }$ . Hence Lemma 6.1 keeps all later states in ${ \mathcal { U } } \times \{ 0 \} , { \mathrm { s o } } \\| u _ { t } \| \geq 1 / 2$ at every iterate. This includes $m = 0 , m = T - 1 , T = 1$ , and zero later steps.

Define the game $f _ { \mathrm { c y c l e } } : \mathbb { R } ^ { 2 } \times \mathbb { R } ^ { 5 } \to \mathbb { R }$ by

$$
f _ { \mathrm { c y c l e } } ( x , y ) : = A \tilde { g } _ { \mathrm { c y c l e } } \left( \frac { x } { s _ { x } } , \frac { y } { s _ { y } } \right) ,\tag{A.3}
$$

initialized at $x _ { 0 } = s _ { x } e _ { 1 }$ and $y _ { 0 } ^ { \mathrm { c y c l e } } = s _ { y } y _ { 0 }$ . Its outer objective and Hessian are

$$
\Phi _ { \mathrm { c y c l e } } ( x ) = \frac { A } { 2 s _ { x } ^ { 2 } } \left. x \right. ^ { 2 } = \frac { \ell \sqrt { \rho } } { 2 C _ { \mathrm { l o w } } } \left. x \right. ^ { 2 } , \qquad \nabla ^ { 2 } \Phi _ { \mathrm { c y c l e } } = \frac { \ell \sqrt { \rho } } { C _ { \mathrm { l o w } } } I _ { 2 } \times 0 .\tag{A.4}
$$

Thus the full outer objective, not merely one component, is strongly convex and quadratic.

After substituting $u = x / s _ { x }$ and $v = y / s _ { y }$ into the updates, both gradient coefficients equal

$$
\frac { \eta _ { t } A } { \rho s _ { x } ^ { 2 } } = \frac { \eta _ { t } A } { s _ { y } ^ { 2 } } = \frac { \ell \eta _ { t } } { C _ { \mathrm { l o w } } \sqrt { \rho } } = h _ { t } .\tag{A.5}
$$

Since $f _ { \mathrm { c y c l e } }$ is affine in $x ,$ its Hessian with respect to x vanishes: $\nabla _ { x x } ^ { 2 } f _ { \mathrm { c y c l e } } = 0$ . For every $( x , y )$ differentiating the partial gradients $\nabla _ { x } f _ { \mathrm { c y c l e } }$ and $\nabla _ { y } f _ { \mathrm { c y c l e } }$ with respect to y gives

$$
\left. \nabla _ { x y } ^ { 2 } f _ { \mathrm { c y c l e } } ( x , y ) \right. _ { \mathrm { o p } } \leq C \frac { A } { s _ { x } s _ { y } } = C \frac { \ell } { C _ { \mathrm { l o w } } } ,
$$

$$
\left. \nabla _ { y y } ^ { 2 } f _ { \mathrm { c y c l e } } ( x , y ) \right. _ { \mathrm { o p } } \leq C \frac { A } { s _ { y } ^ { 2 } } = C \frac { \ell } { C _ { \mathrm { l o w } } \sqrt { \rho } } .
$$

The operator norm of the full Hessian is at most the sum of these two bounds, which is at most $2 C \ell / \dot { C } _ { \mathrm { l o w } } \leq \ell$ since $\rho \geq 1$ . Thus the full gradient is ℓ-Lipschitz. If $\mu$ is the inner-PŁ constant of $\tilde { g } _ { \mathrm { c y c l e } } ,$ the scaled constant is $\mu A / s _ { y } ^ { 2 } = \mu \ell / ( \bar { C } _ { \mathrm { l o w } } \sqrt { \rho } )$ , which is at least $\ell / \kappa$ when

$$
\rho \leq c _ { \mathrm { l o w } } \kappa ^ { 2 } , \qquad c _ { \mathrm { l o w } } : = \left( \frac { \mu } { C _ { \mathrm { l o w } } } \right) ^ { 2 } .\tag{A.6}
$$

By Lemma $6 . 2 , \tilde { g } _ { \mathrm { c y c l e } } ( x / s _ { x } , \cdot )$ has a maximizer for every fixed $x .$ Multiplying that maximizer by $s _ { y }$ gives a maximizer of $f _ { \mathrm { c y c l e } } ( x , \cdot )$ ), since $A > 0$

The initial outer gap $\Phi _ { \mathrm { c y c l e } } ( x _ { 0 } ) - \mathrm { i n f } _ { x } \Phi _ { \mathrm { c y c l e } } ( x )$ and inner gap $\Phi _ { \mathrm { c y c l e } } ( x _ { 0 } ) - f _ { \mathrm { c y c l e } } ( x _ { 0 } , y _ { 0 } ^ { \mathrm { c y c l e } } )$ satisfy the budgets in Assumption 3.3:

$$
\Phi _ { \mathrm { c y c l e } } ( x _ { 0 } ) - \operatorname* { i n f } _ { x } \Phi _ { \mathrm { c y c l e } } ( x ) = \frac { A } { 2 } = \frac { \zeta \Delta } { \sqrt { \rho } } \leq \Delta ,
$$

$$
\Phi _ { \mathrm { c y c l e } } ( x _ { 0 } ) - f _ { \mathrm { c y c l e } } ( x _ { 0 } , y _ { 0 } ^ { \mathrm { c y c l e } } ) = A \left[ \frac { 1 } { 2 } - \tilde { g } _ { \mathrm { c y c l e } } ( e _ { 1 } , y _ { 0 } ) \right] \leq C A \leq 2 C \zeta \Delta \leq c _ { \mathrm { i n i t } } \Delta .\tag{A.7}
$$

The last inequalities use $\rho \geq 1$ and the chosen $\zeta .$ . Finally, every iterate satisfies

$$
\left. \nabla \Phi _ { \mathrm { c y c l e } } ( x _ { t } ) \right. ^ { 2 } = \frac { A ^ { 2 } } { s _ { x } ^ { 2 } } \left. u _ { t } \right. ^ { 2 } = \frac { 2 \zeta \ell \Delta } { C _ { \mathrm { l o w } } } \left. u _ { t } \right. ^ { 2 } \geq \frac { \zeta } { 2 C _ { \mathrm { l o w } } } \ell \Delta .\tag{A.8}
$$

The instance therefore belongs to $\mathcal { C } ( \ell , \kappa , \Delta , \sigma )$ for every $\sigma \geq 0$ , using exact gradients. Its dimension is $( d _ { x } , d _ { y } ) = ( 2 , 5 )$ , independent of the horizon and schedule. Equation $( \mathrm { A } . 8 )$ proves the every-iterate bound (6.5). Averaging this inequality, taking the instance supremum, and then taking the schedule infimum in (3.7) gives $\mathbf { \bar { \mathcal { R } } } _ { T , \circ } ( \rho ) \geq c \ell \dot { \Delta }$ for both update orders.

□

## A.4 UPPER BOUND FOR SGDA<sub>SIM</sub>

We prove the matching upper bound for $\mathtt { S G D A } _ { \mathrm { S i m } }$ under the same assumptions and average squaredgradient criterion as the lower bound. The deterministic one-sided-PŁ counterpart is already established by Chen et al. (2022, Appendix B.2); the argument below includes unbiased stochastic gradients with bounded variance and our convention of averaging over $x _ { 0 } , \ldots , x _ { T - 1 }$ The main issue is that $\nabla _ { x } f ( x _ { t } , y _ { t } )$ differs from $\nabla \Phi ( x _ { t } )$ . Inner PŁ controls this discrepancy through the inner gradient by Lemma B.2; the potential below uses a faster inner step to absorb it.

Theorem A.1 (Upper bound for $\mathbf { S G D A } _ { \mathrm { S i m } } )$ . There are constants $c _ { \mathrm { s t e p } } , C > 0 ;$ , depending at most on $\it { c _ { \mathrm { { i n i t } } } } ,$ , such that the following holds for every $\kappa \geq 2 , \ell , \Delta > 0 , \sigma \geq 0 ,$ , and integer $T \geq 1$ . On any instance in $\mathcal { C } ( \ell , \kappa , \bar { \Delta } , \sigma )$ , run $S G D A _ { S i m }$ with

$$
\rho = 1 6 \kappa ^ { 2 } , \qquad \eta _ { x } : = c _ { \mathrm { s t e p } } \operatorname* { m i n } \left\{ \frac { 1 } { \kappa ^ { 2 } \ell } , \sqrt { \frac { \Delta } { \kappa ^ { 4 } \ell \sigma ^ { 2 } T } } \right\} , \qquad \eta _ { t } \equiv \eta _ { y } : = 1 6 \kappa ^ { 2 } \eta _ { x } ,\tag{A.9}
$$

where the second term $i s + \infty$ when $\sigma = 0 .$ . Then

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left. \nabla \Phi ( x _ { t } ) \right. ^ { 2 } \leq C \left( \frac { \kappa ^ { 2 } \ell \Delta } { T } + \sqrt { \frac { \kappa ^ { 4 } \ell \Delta \sigma ^ { 2 } } { T } } \right) .\tag{A.10}
$$

Consequently, whenever $\varepsilon ^ { 2 } \le c \ell \Delta$ for a sufficiently small universal $c > 0 ,$

$$
N _ { \varepsilon , \mathrm { S i m } } ( 1 6 \kappa ^ { 2 } ) , N _ { \varepsilon , \mathrm { S i m } } ^ { \star } \leq C \left( \frac { \kappa ^ { 2 } \ell \Delta } { \varepsilon ^ { 2 } } + \frac { \kappa ^ { 4 } \ell \Delta \sigma ^ { 2 } } { \varepsilon ^ { 4 } } \right) .\tag{A.11}
$$

Proof of Theorem A.1. We track the outer objective together with the inner suboptimality through

$$
\mathcal { V } _ { t } : = \Phi ( x _ { t } ) + \frac { 1 } { 8 } [ \Phi ( x _ { t } ) - f ( x _ { t } , y _ { t } ) ] .\tag{A.12}
$$

This potential is bounded below by $\Phi ^ { * }$ . Its decrease combines $9 / 8$ times the outer-objective decrease with $\bar { 1 } / 8$ times the game-value increase. The faster dual step makes the latter absorb the discrepancy between the primal and outer gradients.

Let $\mathcal { F } _ { t }$ be the history before the two oracle calls at iteration t and $\mathbb { E } _ { t } [ \cdot ] : = \mathbb { E } [ \cdot \mid \mathcal { F } _ { t } ]$ . The constant primal and dual steps are $\eta _ { x }$ and $\eta _ { y } = 1 6 \kappa ^ { 2 } \eta _ { x }$ from (A.9). Choose the universal constant $c _ { \mathrm { s t e p } } \leq$ $1 / 1 2 8$ . Then

$$
\ell _ { \Phi } \eta _ { x } \le \frac 1 4 , \qquad \ell \eta _ { x } \le \frac 1 8 , \qquad \ell \eta _ { y } \le \frac 1 8 .\tag{A.13}
$$

By ℓ<sub>Φ</sub>-smoothness of Φ and the unbiased primal oracle,

$$
\begin{array} { r l } & { \mathbb { E } _ { t } [ \Phi ( x _ { t } ) - \Phi ( x _ { t + 1 } ) ] \geq \eta _ { x } \left. \nabla \Phi ( x _ { t } ) , \nabla _ { x } f ( x _ { t } , y _ { t } ) \right. - \displaystyle \frac { \ell _ { \Phi } \eta _ { x } ^ { 2 } } { 2 } ( \| \nabla _ { x } f ( x _ { t } , y _ { t } ) \| ^ { 2 } + \sigma ^ { 2 } ) } \\ & { \quad \quad \quad = \displaystyle \frac { \eta _ { x } } { 2 } \left\| \nabla \Phi ( x _ { t } ) \right\| ^ { 2 } + \displaystyle \frac { \eta _ { x } } { 2 } ( 1 - \ell _ { \Phi } \eta _ { x } ) \left\| \nabla _ { x } f ( x _ { t } , y _ { t } ) \right\| ^ { 2 } } \\ & { \quad \quad \quad \quad - \displaystyle \frac { \eta _ { x } } { 2 } \left\| \nabla _ { x } f ( x _ { t } , y _ { t } ) - \nabla \Phi ( x _ { t } ) \right\| ^ { 2 } - \displaystyle \frac { \ell _ { \Phi } \eta _ { x } ^ { 2 } } { 2 } \sigma ^ { 2 } . } \end{array}\tag{A.14}
$$

Assumption 3.1 implies that the full gradient is 2ℓ-Lipschitz in the Euclidean product norm. Since the primal and dual gradients in $\mathbf { S G D A } _ { \mathrm { S i m } }$ are evaluated at $( x _ { t } , y _ { t } )$ ), this gives

$$
\begin{array} { r l } { \mathbb { E } _ { t } \big [ f ( x _ { t + 1 } , y _ { t + 1 } ) - f ( x _ { t } , y _ { t } ) \big ] \geq - \left( \eta _ { x } + \ell \eta _ { x } ^ { 2 } \right) \big \Vert \nabla _ { x } f ( x _ { t } , y _ { t } ) \big \Vert ^ { 2 } + \left( \eta _ { y } - \ell \eta _ { y } ^ { 2 } \right) \big \Vert \nabla _ { y } f ( x _ { t } , y _ { t } ) \big \Vert ^ { 2 } } & { } \\ { - \ell ( \eta _ { x } ^ { 2 } + \eta _ { y } ^ { 2 } ) \sigma ^ { 2 } . } & { \qquad ( \mathrm { A } . 1 \big ) } \end{array}\tag{5}
$$

Combining $\frac { 9 } { 8 }$ times (A.14) with $\frac { 1 } { 8 }$ times (A.15), using the gradient-bias bound (B.6), and then applying (A.13), yields

$$
\mathbb { E } _ { t } [ \mathcal { V } _ { t } - \mathcal { V } _ { t + 1 } ] \geq \frac { 9 \eta _ { x } } { 1 6 } \left. \nabla \Phi ( x _ { t } ) \right. ^ { 2 } - 3 3 \kappa ^ { 4 } \ell \eta _ { x } ^ { 2 } \sigma ^ { 2 } .\tag{A.16}
$$

Indeed, (A.13) makes the remaining squared-gradient coefficients nonnegative:

$$
\begin{array} { r } { \displaystyle \frac { 9 \eta _ { x } } { 1 6 } ( 1 - \ell _ { \Phi } \eta _ { x } ) - \displaystyle \frac { \eta _ { x } } { 8 } - \displaystyle \frac { \ell \eta _ { x } ^ { 2 } } { 8 } \geq 0 , } \\ { \displaystyle \frac { \eta _ { y } } { 8 } ( 1 - \ell \eta _ { y } ) - \displaystyle \frac { 9 \eta _ { x } \kappa ^ { 2 } } { 1 6 } \geq 0 . } \end{array}
$$

The dual-noise contribution is $\ell \eta _ { u } ^ { 2 } / 8 = 3 2 \kappa ^ { 4 } \ell \eta _ { x } ^ { 2 }$ , which determines the fourth power of κ in the variance bound. The full noise coefficient satisfies

$$
\frac { 9 \ell _ { \Phi } \eta _ { x } ^ { 2 } } { 1 6 } + \frac { \ell } { 8 } ( \eta _ { x } ^ { 2 } + \eta _ { y } ^ { 2 } ) \leq 3 3 \kappa ^ { 4 } \ell \eta _ { x } ^ { 2 } \qquad ( \kappa \geq 2 ) .
$$

We sum $\left( \mathrm { A . 1 6 } \right)$ over the $T - 1$ updates, retaining the terminal potential to control the last reported iterate. Smoothness and lower boundedness of Φ give $\left. \nabla \Phi ( x ) \right. ^ { 2 } \leq 2 \ell _ { \Phi } [ \Phi ( x ) - \Phi ^ { * } ]$ . Thus (A.13) implies

$$
\frac { 9 \eta _ { x } } { 1 6 } \mathbb { E } \left\| \nabla \Phi ( x _ { T - 1 } ) \right\| ^ { 2 } \leq \frac { 9 } { 3 2 } \mathbb { E } [ \mathcal { V } _ { T - 1 } - \Phi ^ { * } ] .
$$

Adding this bound to the sum and using both initial-gap bounds gives

$$
\begin{array} { r l r } {  { \frac { 9 \eta _ { x } } { 1 6 } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E }  \nabla \Phi ( x _ { t } )  ^ { 2 } \leq \mathcal { V } _ { 0 } - \Phi ^ { * } - \frac { 2 3 } { 3 2 } \mathbb { E } [ \mathcal { V } _ { T - 1 } - \Phi ^ { * } ] } } \\ & { } & { \quad + 3 3 \kappa ^ { 4 } \ell \eta _ { x } ^ { 2 } \sigma ^ { 2 } ( T - 1 ) } \\ & { } & { \quad \leq ( 1 + \frac { C _ { \mathrm { i n i t } } } { 8 } ) \Delta + 3 3 \kappa ^ { 4 } \ell \eta _ { x } ^ { 2 } \sigma ^ { 2 } T . } \end{array}\tag{A.17}
$$

With the empty-sum convention, the same argument applies when $T = 1$ . Dividing by $\eta _ { x } T$ and absorbing the numerical constants gives

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| \nabla \Phi ( x _ { t } ) \right\| ^ { 2 } \leq C \left[ \frac { \Delta } { \eta _ { x } T } + \kappa ^ { 4 } \ell \eta _ { x } \sigma ^ { 2 } \right] .\tag{A.18}
$$

Substitute (A.9). If the step-size bound $1 / ( \kappa ^ { 2 } \ell )$ is active, then $\sigma ^ { 2 } T \le \ell \Delta$ and both terms in (A.18) are bounded by $C \kappa ^ { 2 } \ell \Delta / T$ . Otherwise the two terms are both bounded by $C \sqrt { \kappa ^ { 4 } \ell \Delta \sigma ^ { 2 } / T }$ . This proves (A.10).

Finally, choosing

$$
T \geq C \left( \frac { \kappa ^ { 2 } \ell \Delta } { \varepsilon ^ { 2 } } + \frac { \kappa ^ { 4 } \ell \Delta \sigma ^ { 2 } } { \varepsilon ^ { 4 } } \right)
$$

makes the right-hand side of (A.10) at most $\varepsilon ^ { 2 }$ . The constant schedule is nonincreasing, so the minimax definitions in (3.7) give (A.11). □

## B PRELIMINARY LEMMAS

We first record the completion-of-squares construction used to preserve the outer function and inner PŁ. We then prove the consequences of inner PŁ used in the matching upper bound.

## B.1 NC-PŁ GAMES WITH A PREDETERMINED OUTER FUNCTION

We use an onto inner map to determine the outer maximum exactly. A lower bound on its transposed derivative then converts the residual norm into an inner-gradient bound. This lemma proves inner PŁ, not global smoothness; the latter is checked for each constructed map.

Lemma B.1 (Outer function and inner PŁ for a completion-of-squares construction). Let $b : \mathbb { R } ^ { d _ { x } } $ $\mathbb { R } ^ { m }$ and $T : \mathbb { R } ^ { d _ { y } }  \mathbb { R } ^ { m }$ be continuously differentiable, with T onto (surjective). Define

$$
g ( x , y ) : = \left. b ( x ) , T ( y ) \right. - \frac { 1 } { 2 } \left. T ( y ) \right. ^ { 2 } .\tag{B.1}
$$

Then

$$
\operatorname* { m a x } _ { y } g ( x , y ) = \frac { 1 } { 2 } \left\| b ( x ) \right\| ^ { 2 } , \qquad \operatorname* { m a x } _ { y } g ( x , y ) - g ( x , y ) = \frac { 1 } { 2 } \left\| b ( x ) - T ( y ) \right\| ^ { 2 } .\tag{B.2}
$$

If, for some $\nu > 0$

$$
\begin{array} { r } { \left\| D T ( \boldsymbol { y } ) ^ { \mathsf { T } } \boldsymbol { z } \right\| \ge \nu \left\| \boldsymbol { z } \right\| \qquad f o r a l l y , \boldsymbol { z } , } \end{array}\tag{B.3}
$$

then

$$
\frac { 1 } { 2 } \left\| \nabla _ { y } g ( x , y ) \right\| ^ { 2 } \geq \nu ^ { 2 } \left[ \operatorname* { m a x } _ { v } g ( x , v ) - g ( x , y ) \right] .\tag{B.4}
$$

Proof. Since $T$ is onto, maximizing (B.1) over y is equivalent to maximizing $\left. b ( x ) , z \right. - \left\| z \right\| ^ { 2 } / 2$ over $z \in \mathbb { R } ^ { m }$ . Completing the square gives (B.2). Moreover,

$$
\nabla _ { y } g ( x , y ) = D T ( y ) ^ { \mathsf { T } } [ b ( x ) - T ( y ) ] ,
$$

so (B.3) and (B.2) imply (B.4).

## B.2 NC-PŁ GEOMETRY

The following are standard consequences of smoothness and inner $\mathrm { P E } ;$ see Nouiehed et al. (2019, Lemma A.5), Chen et al. (2022, Lemmas $\mathsf { A } . 7 \mathrm { - A } . 8 )$ , and Ha (2026, Lemmas A.2–A.3). We prove them under Assumptions 3.1 and 3.2, without concavity or uniqueness of the maximizer.

Lemma B.2 (Inner error bound, gradient bias, and outer smoothness). Under Assumptions 3.1 and 3.2, Φ is differentiable with $\nabla \bar { \Phi } ( x ) = \nabla _ { x } f ( x , y ^ { \star } )$ for every $y ^ { \star } \in Y ^ { \star } ( x )$ . For every $( x , y )$

$$
\mathrm { d i s t } ( y , Y ^ { \star } ( x ) ) \leq \frac { \kappa } { \ell } \left\| \nabla _ { y } f ( x , y ) \right\| ,\tag{B.5}
$$

$$
\begin{array} { r } { \| \nabla _ { x } f ( x , y ) - \nabla \Phi ( x ) \| \le \kappa \left\| \nabla _ { y } f ( x , y ) \right\| . } \end{array}\tag{B.6}
$$

Moreover, Φ has $\ell _ { \Phi }$ -Lipschitz gradient with

$$
\ell _ { \Phi } \leq \ell ( 1 + \kappa ) .\tag{B.7}
$$

Proof. We first bound the inner gap after changing x while keeping a maximizer fixed; the resulting quadratic remainder proves differentiability of Φ. An inner gradient-flow argument then bounds the distance to a maximizer, giving the gradient-bias and outer-smoothness estimates.

Fix x $: , y ^ { \star } \in Y ^ { \star } ( x )$ , and $\boldsymbol { h } \in \mathbb { R } ^ { d _ { x } }$ . Since the inner domain is unconstrained, $\nabla _ { y } f ( x , y ^ { \star } ) = 0$ . Inner PŁ at $( x + h , y ^ { \star } )$ and Assumption 3.1 give

$$
0 \leq \Phi ( x + h ) - f ( x + h , y ^ { \star } ) \leq \frac { \| \nabla _ { y } f ( x + h , y ^ { \star } ) \| ^ { 2 } } { 2 \mu } \leq \frac { \ell \kappa } { 2 } \left\| h \right\| ^ { 2 } .
$$

Combining this with the Taylor bound for $f ( \cdot , y ^ { \star } )$ yields

$$
\left| \Phi ( x + h ) - \Phi ( x ) - \langle \nabla _ { x } f ( x , y ^ { \star } ) , h \rangle \right| \leq \frac { \ell ( 1 + \kappa ) } { 2 } \left\| h \right\| ^ { 2 } .
$$

Thus Φ is differentiable and $\nabla \Phi ( x ) = \nabla _ { x } f ( x , y ^ { \star } )$ for every maximizer, including when the maximizer is not unique. If $d _ { y } = 0$ , this identity and all the claimed bounds follow directly from Assumption 3.1; henceforth assume $d _ { y } \geq 1$

Fix x and define the inner objective gap $g _ { x } ( y ) : = \Phi ( x ) - f ( x , y ) \geq 0$ . It is ℓ-smooth, attains its minimum zero, and satisfies $\lVert \nabla g _ { x } ( y ) \rVert ^ { 2 } \geq 2 ( \ell / \kappa ) g _ { x } ( y )$ . Since $\nabla g _ { x }$ is globally Lipschitz, the gradient flow $\dot { y } ( s ) = - \nabla g _ { x } ( y ( s ) )$ , initialized at $y ( 0 ) = y ,$ exists for all $s \geq 0$ . If it reaches $g _ { x } = 0$ in finite time, the desired distance bound follows immediately by integrating up to that time. Otherwise $g _ { x } ( y ( s ) ) > 0$ for every finite $s ,$ and

$$
- \frac { d } { d s } \sqrt { g _ { x } ( y ( s ) ) } = \frac { \left. \nabla g _ { x } ( y ( s ) ) \right. ^ { 2 } } { 2 \sqrt { g _ { x } ( y ( s ) ) } } \geq \sqrt { \frac { \ell } { 2 \kappa } } \left. \nabla g _ { x } ( y ( s ) ) \right. .
$$

Hence the total distance traveled by the gradient-flow solution is finite:

$$
\int _ { 0 } ^ { \infty } \| \dot { y } ( s ) \| ~ d s \leq \sqrt { \frac { 2 \kappa g _ { x } ( y ) } { \ell } } .
$$

The finite path length makes $y ( s )$ a Cauchy curve in $\mathbb { R } ^ { d _ { y } }$ , so it converges to a limit $y ^ { \star }$ . The limit must satisfy $g _ { x } ( y ^ { \star } ) = 0 \colon$ otherwise PŁ would keep $\| \nabla g _ { x } \|$ bounded away from zero and the integral of its speed would be infinite. Thus $y ^ { \star } \in Y ^ { \star } ( x )$ , and therefore

$$
\mathrm { d i s t } ( y , Y ^ { \star } ( x ) ) \leq \sqrt { \frac { 2 \kappa g _ { x } ( y ) } { \ell } } \leq \frac { \kappa } { \ell } \left\| \nabla g _ { x } ( y ) \right\| ,
$$

where the second inequality is the PŁ inequality. This proves (B.5).

Choose the maximizer supplied by (B.5). The gradient identity just proved and Assumption 3.1 give

$$
\| \nabla _ { x } f ( x , y ) - \nabla \Phi ( x ) \| \leq \ell \left\| y - y ^ { \star } \right\| \leq \kappa \left\| \nabla _ { y } f ( x , y ) \right\| ,
$$

which is (B.6).

Finally, fix $x , x ^ { \prime }$ and $y ^ { \star } \in Y ^ { \star } ( x )$ . Since $\nabla _ { y } f ( x , y ^ { \star } ) = 0$ , Assumption 3.1 gives $\| \nabla _ { y } f ( x ^ { \prime } , y ^ { \star } ) \| \leq$ $\ell \| x ^ { \prime } - x \|$ . Applying (B.5) to the inner problem at $x ^ { \prime }$ yields a maximizer ${ \bar { y } } \in Y ^ { \star } ( x ^ { \prime } )$ such that $\| \ b { \bar { y } } - \ b { y } ^ { \star } \| \leq \kappa \| \ b { x } ^ { \prime } - \ b { x } \|$ . Thus

$$
\begin{array} { r l } & { \| \nabla \Phi ( { x } ^ { \prime } ) - \nabla \Phi ( { x } ) \| = \| \nabla _ { x } f ( { x } ^ { \prime } , \bar { y } ) - \nabla _ { x } f ( { x } , { y } ^ { \star } ) \| } \\ & { \qquad \le \ell \bigl ( \| { x } ^ { \prime } - { x } \| + \| \bar { y } - { y } ^ { \star } \| \bigr ) \le \ell ( 1 + \kappa ) \| { x } ^ { \prime } - { x } \| . } \end{array}
$$

This proves (B.7).

## C PROOFS FOR THE HIGH-RATIO LOWER BOUND

We prove the construction lemmas used in Section 5. The lower bound for $f _ { \mathrm { u n s t a b l e } }$ accounts for initially large dual steps. On the other hand, the corridor function $g _ { \mathrm { c o r r i d o r } }$ controls the remaining primal movement, and the noisy function $g _ { \sigma }$ supplies the stochastic term in the lower bound. Finally, we combine the corridor function with the noisy function to obtain a single NC-PŁ game instance and prove its lower bound.

## C.1 PROOF OF THE LOWER BOUND FOR f<sub>unstable</sub>

Lemma 5.2 (A lower bound for $f _ { \mathrm { u n s t a b l e } } )$ . There is a universal $c > 0$ such that,for $\ell , \Delta > 0 , \kappa \geq 2$ $\sigma \geq 0 , \rho \geq 1$ , integers $T \geq 1$ and $0 \leq m < T$ , and predetermined nonnegative non-increasing step sizes with $\eta _ { t } \geq 8 / \ell f o r t < m _ { \cdot }$ , each ofSGDA and $S G D A _ { A l t }$ admits an initialization $( x _ { 0 } , y _ { 0 } )$ making $f _ { \mathrm { u n s t a b l e } }$ with $\mu = \ell /$ κ an exact-gradient instance in $\mathcal { C } ( \ell , \kappa , \Delta , \sigma )$ with

$$
{ \frac { 1 } { T } } \sum _ { t = 0 } ^ { T - 1 } | \Phi _ { \mathrm { u n s t a b l e } } ^ { \prime } ( x _ { t } ) | ^ { 2 } \geq c { \frac { m + 1 } { T } } \ell \Delta .\tag{5.3}
$$

Proof. We use the same initialization for both update rules. We show that two consecutive updates with dual step sizes at least $8 / \ell$ take any state $( x , y )$ satisfying $0 \leq x \leq y$ to a state $( \dot { x } ^ { \prime \prime } , y ^ { \prime \prime } )$ satisfying $0 \overset { \cdot } { \leq } x \leq x ^ { \prime \prime } \leq y ^ { \prime \prime }$ . For $m \geq 1$ , the first update gives $x _ { 1 } = x _ { 0 }$ and $y _ { 1 } \geq 2 x _ { 0 }$ . Repeating the two-update argument then gives $x _ { t } \geq x _ { 0 }$ for $t = \bar { 1 } , 3 , 5 , . . .$ . with $t \leq m$

Set $\mu = \ell / \kappa , x _ { 0 } = 4 \sqrt { \Delta / ( \ell + \mu ) }$ , and $y _ { 0 } = 0$ . The Hessian is $\begin{array} { r } { ( \ell + \mu ) { \binom { 0 } { 1 / 4 } } \qquad } \end{array}$ , and the inner maximum is attained at $y = x / 2$ . Hence

$$
\begin{array} { r l r } & { } & { \displaystyle \left\| \nabla ^ { 2 } f _ { \mathrm { u n s t a b l e } } \right\| _ { \mathrm { o p } } = \frac { 1 + \sqrt { 2 } } { 4 } ( \ell + \mu ) \le \frac { 3 ( 1 + \sqrt { 2 } ) } { 8 } \ell < \ell , } \\ & { } & { \displaystyle \Phi _ { \mathrm { u n s t a b l e } } ( x ) - f _ { \mathrm { u n s t a b l e } } ( x , y ) = \frac { \ell + \mu } { 4 } ( y - x / 2 ) ^ { 2 } , } \\ & { } & { \displaystyle \frac { 1 } { 2 } | \nabla _ { y } f _ { \mathrm { u n s t a b l e } } ( x , y ) | ^ { 2 } = \frac { \ell + \mu } { 2 } \big ( \Phi _ { \mathrm { u n s t a b l e } } ( x ) - f _ { \mathrm { u n s t a b l e } } ( x , y ) \big ) . } \end{array}
$$

Since $( \ell + \mu ) / 2 \geq \mu$ , the game satisfies inner $\mu { \mathrm { - } } \mathrm { P f }$ . Its outer objective is differentiable and lower bounded. Both initial gaps equal ∆, so the initialization is admissible because $c _ { \mathrm { i n i t } } \geq 1$

Write $s _ { t } = ( \ell + \mu ) \eta _ { t } / 2 \geq 4$ for $t < m$ . A step of normalized size a sends $( x , y )$ to

$$
x ^ { \prime } = x - \textstyle { \frac { a } { 2 \rho } } y , \qquad y ^ { \prime } = \textstyle { \left\{ \begin{array} { l l } { ( 1 - a ) y + \frac { a } { 2 } x , } & { \mathrm { S i m } , } \\ { ( 1 - a ) y + \frac { a } { 2 } x ^ { \prime } , } & { \mathrm { A l t } . } \end{array} \right. }\tag{C.1}
$$

If $m \geq 1$ , the first step gives $x _ { 1 } = x _ { 0 }$ and $y _ { 1 } = s _ { 0 } x _ { 0 } / 2 \geq 2 x _ { 0 }$ . Consider two subsequent steps $a , b \geq 4$ starting from $0 \leq x \leq y .$ , with states $( x ^ { \prime } , y ^ { \prime } )$ and $( x ^ { \prime \prime } , y ^ { \prime \prime } )$ . Since $x ^ { \prime } \leq x \leq y .$ , both update rules satisfy

$$
\begin{array} { c } { y ^ { \prime } \leq ( 1 - a / 2 ) y \leq 0 , } \\ { x ^ { \prime \prime } - x = - \displaystyle \frac { a y + b y ^ { \prime } } { 2 \rho } \geq \frac { a b / 2 - a - b } { 2 \rho } y \geq 0 . } \end{array}
$$

The last coefficient is nonnegative because $( a - 2 ) ( b - 2 ) \geq 4$ . For $\mathrm { S G D A _ { A l t } } , y ^ { \prime \prime } = ( b / 2 ) x ^ { \prime \prime } + ( 1 -$ $b ) y ^ { \prime } \geq 2 x ^ { \prime \prime }$ , since $x ^ { \prime \prime } \geq 0$ and $y ^ { \prime } \leq 0$ . For $\mathbf { S G D A } _ { \mathrm { S i m } }$

$$
\begin{array} { c } { { x ^ { \prime } - y ^ { \prime } = ( 1 - a / 2 ) x + ( a - 1 - a / ( 2 \rho ) ) y \geq \displaystyle \frac { a } { 2 } ( 1 - \rho ^ { - 1 } ) x \geq 0 , } } \\ { { \displaystyle y ^ { \prime \prime } - x ^ { \prime \prime } = ( b / 2 - 1 ) x ^ { \prime } + ( 1 - b + b / ( 2 \rho ) ) y ^ { \prime } \geq - \displaystyle \frac { b } { 2 } ( 1 - \rho ^ { - 1 } ) y ^ { \prime } \geq 0 . } } \end{array}
$$

Here the first line uses $y \geq x$ and $a - 1 - a / ( 2 \rho ) \geq 0 ;$ the second uses $x ^ { \prime } \geq y ^ { \prime }$ and $b / 2 - 1 \geq 0$ Thus either pair of steps gives $0 \leq x \leq x ^ { \prime \prime } \leq y ^ { \prime \prime }$

Induction from $( x _ { 1 } , y _ { 1 } )$ yields $x _ { t } \geq x _ { 0 }$ at every odd $t \leq m$ . Including $t = 0$ , we obtain

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } | \Phi _ { \mathrm { u n s t a b l e } } ^ { \prime } ( x _ { t } ) | ^ { 2 } \geq \frac { ( \ell + \mu ) ^ { 2 } } { 6 4 T } \sum _ { t = 0 } ^ { m } x _ { t } ^ { 2 } \geq \frac { ( \ell + \mu ) ^ { 2 } x _ { 0 } ^ { 2 } } { 6 4 T } \left( 1 + \left\lceil \frac { m } { 2 } \right\rceil \right) \geq \frac { m + 1 } { 8 T } \ell \Delta .\tag{C.2}
$$

The same bound holds directly for $m = 0$ . Exact gradients are admissible for every $\sigma \geq 0$ , completing the proof. □

## C.2 CONSTRUCTION OF g<sub>corridor</sub>

Lemma 5.3 (A lower bound for $g _ { \mathrm { c o r r i d o r } } ) .$ . For $\ell , \Delta > 0 , \rho \geq 1$ , integers $T \geq 1 , 0 \leq m < T$ and nonnegative steps with $\begin{array} { r } { S = \sum _ { t = m } ^ { T - 2 } \eta _ { t } , } \end{array}$ , there is an ℓ-smooth function $g _ { \mathrm { c o r r i d o r } } : \mathbb { R } $ R with initial gap $g _ { \mathrm { c o r r i d o r } } ( 0 ) - \mathrm { i n f } _ { u } g _ { \mathrm { c o r r i d o r } } \ddot { ( u ) } = \Delta$ . Gradient descent from $u _ { 0 } = 0$ with steps $\eta _ { t } / \rho$ satisfies $g _ { \mathrm { c o r r i d o r } } ^ { \prime } ( u _ { t } ) < 0$ for all $t < T$ and $g _ { \mathrm { c o r r i d o r } } ^ { \prime } ( u _ { t } ) = - G f o r m \le t < T$ , where $G > 0$ and $G ^ { 2 } \breve { \asymp } \operatorname* { m i n } \{ \ell \Delta , \rho \Delta / S \}$ . Consequentially,

$$
\frac { 1 } { T } \sum _ { t < T } | g _ { \mathrm { c o r r i d o r } } ^ { \prime } ( u _ { t } ) | ^ { 2 } \gtrsim \frac { T - m } { T } \operatorname* { m i n } \left\{ \ell \Delta , \frac { \rho \Delta } { S } \right\} .\tag{5.5}
$$

Proof. We prescribe the derivative before choosing its magnitude. An initial linear segment makes the first m updates reach a constant-derivative interval. Its length is the distance traveled by the remaining steps; a final linear segment joins it to a minimum. Integrating the derivative then calibrates the exact initial gap.

Fix $m \in \{ 0 , \ldots , T - 1 \}$ and let $\textstyle S = \sum _ { t = m } ^ { T - 2 } \eta _ { t }$ . When $S = 0$ , the minimum in the statement is $\ell \Delta$ Define

$$
P _ { 0 } : = 1 , \qquad P _ { t } : = \prod _ { i = 0 } ^ { t - 1 } \left( 1 + { \frac { \ell \eta _ { i } } { \rho } } \right) \quad ( 1 \le t \le m ) ,\tag{C.3}
$$

$$
G ^ { 2 } : = \frac { \Delta } { S / \rho + \ell ^ { - 1 } ( 1 - 1 / ( 2 P _ { m } ^ { 2 } ) ) } , \qquad G _ { 0 } : = \frac { G } { P _ { m } } , \qquad b : = \frac { G - G _ { 0 } } { \ell } + \frac { G S } { \rho } .\tag{C.4}
$$

For $G \ > \ 0$ , write $[ z ] _ { [ 0 , G ] } : = \mathrm {  ~ \ m i n } \{ G , \mathrm { m a x } \{ 0 , z \} \}$ . Add a constant to $g _ { \mathrm { c o r r i d o r } }$ so that $\operatorname* { i n f } _ { u } g _ { \mathrm { c o r r i d o r } } ( u ) = 0$ , and define

$$
g _ { \mathrm { c o r r i d o r } } ^ { \prime } ( u ) : = - [ G _ { 0 } + \ell u ] _ { [ 0 , G ] } + [ \ell ( u - b ) ] _ { [ 0 , G ] } .\tag{C.5}
$$

The two intervals on which the clipping terms vary are disjoint, so $g _ { \mathrm { c o r r i d o r } } ^ { \prime }$ is continuous and ℓ-Lipschitz.

Starting from $u _ { 0 } = 0 ,$ , induction gives

$$
u _ { t } = \frac { P _ { t } - 1 } { \ell } G _ { 0 } , \qquad g _ { \mathrm { c o r r i d o r } } ^ { \prime } ( u _ { t } ) = - P _ { t } G _ { 0 } , \qquad 0 \le t \le m ,\tag{C.6}
$$

$$
u _ { t } = \frac { G - G _ { 0 } } { \ell } + G \sum _ { j = m } ^ { t - 1 } \frac { \eta _ { j } } { \rho } , \qquad g _ { \mathrm { c o r r i d o r } } ^ { \prime } ( u _ { t } ) = - G , \qquad m \leq t \leq T - 1 .\tag{C.7}
$$

Direct integration gives

$$
g _ { \mathrm { c o r r i d o r } } ( 0 ) = \frac { G ^ { 2 } - G _ { 0 } ^ { 2 } } { 2 \ell } + \frac { G ^ { 2 } S } { \rho } + \frac { G ^ { 2 } } { 2 \ell } = G ^ { 2 } \left( \frac { S } { \rho } + \frac { 1 } { \ell } - \frac { 1 } { 2 \ell P _ { m } ^ { 2 } } \right) = \Delta ,\tag{C.8}
$$

and

$$
G ^ { 2 } \geq \frac { \Delta } { S / \rho + 1 / \ell } \geq \frac { 1 } { 2 } \operatorname* { m i n } \left\{ \ell \Delta , \frac { \rho \Delta } { S } \right\} .\tag{C.9}
$$

Since $P _ { m } \ge 1$ , the denominator in (C.4) is at least $S / \rho + 1 / ( 2 \ell )$ , so also $G ^ { 2 } \le 2 \operatorname* { m i n } \{ \ell \Delta , \rho \Delta / S \}$ This proves the two-sided comparison in the statement.

The game is the single-player objective $f ( u ) = g _ { \mathrm { c o r r i d o r } } ( u )$ , equivalently an instance with no inner variable. Hence the inner-PŁ condition is vacuous. Equations (C.7) and (C.9), summed over the last $T - m$ iterates, prove (5.5). Since $G _ { 0 } , G > 0$ and $P _ { t } > 0$ , the derivative is negative at every iterate, so each positive update strictly increases the scalar iterate. The initial gap equals $\Delta$ by (C.8). Exact gradients are admissible for every $\sigma .$ □

## C.3 CONSTRUCTION OF $g _ { \sigma }$

We construct $g _ { \sigma }$ by giving each inner-gradient error its own component. The shifted quadratic game $g _ { \delta }$ in Lemma 5.4 shows why this is useful: a shift in its inner gradients moves the outer variable away from its minimizer, and that displacement persists over a range of cumulative steps. We then build a smooth inner-PŁ component $g _ { \sigma , j }$ in which either sign of the error at update $j$ activates the same shift. The displacement estimate lets us count how many subsequent iterates have a large outer gradient. Summing the resulting squared-gradient bounds over components with enough remaining cumulative steps gives the lower bound on the average squared outer-gradient norm in Lemma C.2. The fixed oracle in (5.11) generates all of these errors.

## C.3.1 PERSISTENT ERROR UNDER A DETERMINISTIC SHIFT

Lemma 5.4 (Persistent error under a deterministic shift). There are universal constants $c , C , C _ { 0 } > 0$ such that, $f o r \ell , \delta > 0 , \kappa \geq 1 , \rho \geq C _ { 0 } \kappa ^ { 2 }$ , and any finite step-size sequence with $0 \leq \ell \eta _ { t } \leq c \kappa ,$ , both $S G D A _ { S i m }$ and $S G D A _ { A l t }$ with exact gradients on the linear quadratic game $g _ { \delta }$ given $b y \left( 5 . 6 \right) _ { \mathrm { : } }$ , initialized at $( 0 , 0 , 0 )$ , satisfy $| z _ { t } | \le C \kappa \delta / \rho$ and $| a _ { t } | + | b _ { t } | \leq C \delta$ whenever $\begin{array} { r } { \breve { \sum } _ { i < t } \eta _ { i } \le c \rho / ( \dot { \ell } \kappa ) } \end{array}$ , and

$$
z _ { t } \geq c \frac { \kappa \delta } { \rho } \quad i f \quad C \frac { \kappa } { \ell } \leq \sum _ { i < t } \eta _ { i } \leq c \frac { \rho } { \ell \kappa } .\tag{5.7}
$$

Proof. Recall that $\begin{array} { r } { g _ { \delta } ( z ; a , b ) = \ell z a - \frac { \ell } { 2 \kappa } [ a ^ { 2 } + ( a + b + \delta ) ^ { 2 } ] } \end{array}$ and $( z _ { 0 } , a _ { 0 } , b _ { 0 } ) = ( 0 , 0 , 0 )$ . We bound the inner coordinates in terms of the largest outer iterate, then use an exact identity to close the outer bound. Once the initial inner transient has decayed, the same identity gives a positive lower bound on the outer displacement. Normalize the coordinates and dual steps by setting

$$
X _ { n } = \frac { \kappa z _ { n } } { \delta } , \qquad Y _ { n } = \frac { a _ { n } } { \delta } , \qquad Z _ { n } = 1 + \frac { b _ { n } } { \delta } , \qquad h _ { n } = \frac { \ell \eta _ { n } } { \kappa } , \qquad \alpha = \frac { \kappa ^ { 2 } } { \rho } .
$$

The SGDA updates for $g _ { \delta }$ become

$$
\begin{array} { r l } & { X _ { n + 1 } = X _ { n } - \alpha h _ { n } Y _ { n } , } \\ & { Y _ { n + 1 } = Y _ { n } + h _ { n } ( X _ { n } ^ { \prime } - 2 Y _ { n } - Z _ { n } ) , \qquad ( X _ { 0 } , Y _ { 0 } , Z _ { 0 } ) = ( 0 , 0 , 1 ) , } \\ & { Z _ { n + 1 } = Z _ { n } - h _ { n } ( Y _ { n } + Z _ { n } ) , } \end{array}\tag{C.10}
$$

where $X _ { n } ^ { \prime } \ = \ X _ { n }$ for $\mathtt { S G D A } _ { \mathrm { S i m } }$ and $X _ { n } ^ { \prime } \ = \ X _ { n + 1 }$ for $\mathrm { S G D A _ { A l t } }$ . Write $\begin{array} { r } { \tau _ { n } ~ = ~ \sum _ { i < n } h _ { i } } \end{array}$ for the cumulative normalized step. We choose universal positive constants $\alpha _ { \star } , \bar { h } , \tau _ { - } , \tau _ { + }$ below. For $\alpha \leq \alpha ,$ and $h _ { n } \leq { \bar { h } } .$ , we will bound $\vert X _ { n } \vert$ | by a universal multiple of α when $\tau _ { n } \leq \tau _ { + } / \alpha$ , and prove $X _ { n } \geq \alpha / 4$ when $\tau _ { - } \le \tau _ { n } \le \tau _ { + } / \alpha$

Subtracting the updates for $Y _ { n }$ and $Z _ { n }$ in (C.10) yields $Y _ { n + 1 } - Z _ { n + 1 } - ( Y _ { n } - Z _ { n } ) = h _ { n } ( X _ { n } ^ { \prime } - Y _ { n } )$ Summing this equality over $0 \leq i < n$ , using $Y _ { 0 } - Z _ { 0 } = \dot { - } 1$ , and then summing the update for $X _ { n }$ gives

$$
X _ { n } = \alpha \left[ 1 + Y _ { n } - Z _ { n } - \sum _ { i < n } h _ { i } X _ { i } ^ { \prime } \right] .\tag{C.11}
$$

The constant 1 records the initial offset $Y _ { 0 } - Z _ { 0 } = - 1$ . To obtain a positive outer displacement, we must make the inner-coordinate difference small while keeping the accumulated outer term small.

Let $q _ { n } = ( Y _ { n } , Z _ { n } ) ^ { \mathsf { T } } \in \mathbb { R } ^ { 2 }$ collect the normalized inner coordinates in (C.10). Its update is $q _ { n + 1 } =$ $( I + h _ { n } B ) q _ { n } + h _ { n } X _ { n } ^ { \prime } ( 1 , 0 ) ^ { \mathsf { T } }$ , where $B = \left( \begin{array} { l } { { - 2 \ : - 1 } } \\ { { - 1 \ : - 1 } } \end{array} \right)$ . The eigenvalues of $- B$ are $( 3 \pm { \sqrt { 5 } } ) / 2$ Write $\lambda _ { 0 } = { ( 3 - \sqrt { 5 } ) } / { 2 }$ and choose $\bar { h } \le 2 / ( 3 + \sqrt { 5 } )$ . Then $I + h B$ is positive semidefinite and $\| I + h B \| _ { \mathrm { o p } } = 1 - \lambda _ { 0 } h$ for $0 \leq h \leq \bar { h }$

Fix an endpoint $N ,$ and let $M _ { N } = \operatorname* { m a x } _ { 0 \leq i \leq N } \left| X _ { i } \right|$ be the largest magnitude of the normalized outer coordinate through iterate $N _ { \cdot }$ . Since $| X _ { n } ^ { \overline { { \prime } } } | \overset { - } { \leq } \dot { M } _ { N }$ for $n < N$ , the formula for $q _ { n + 1 }$ and the norm bound on $\boldsymbol { I } + \boldsymbol { h } _ { n } \boldsymbol { B }$ give

$$
\left. q _ { n + 1 } \right. \leq \left( 1 - \lambda _ { 0 } h _ { n } \right) \left. q _ { n } \right. + h _ { n } M _ { N } ,\tag{C.12}
$$

$$
\| q _ { n } \| \leq \prod _ { i < n } ( 1 - \lambda _ { 0 } h _ { i } ) + \frac { M _ { N } } { \lambda _ { 0 } } \left[ 1 - \prod _ { i < n } ( 1 - \lambda _ { 0 } h _ { i } ) \right] \leq e ^ { - \lambda _ { 0 } \tau _ { n } } + \frac { M _ { N } } { \lambda _ { 0 } } .\tag{C.13}
$$

Iterating the first inequality from $\boldsymbol { q } _ { 0 } = ( 0 , 1 ) ^ { \intercal }$ gives the product bound in the second line; $1 - a \leq e ^ { - a }$   
gives its exponential bound.

For every $n \leq N , | Y _ { n } - Z _ { n } | \leq { \sqrt { 2 } } \| q _ { n } \|$ and $\begin{array} { r } { \sum _ { i < n } h _ { i } | X _ { i } ^ { \prime } | \le \tau _ { N } M _ { N } } \end{array}$ . Substituting these bounds and (C.13) into $( \mathrm { C } . 1 1 )$ , then taking the maximum over $0 \leq n \leq N$ , gives

$$
M _ { N } \leq \alpha ( 1 + \sqrt { 2 } ) + \alpha \left( \frac { \sqrt { 2 } } { \lambda _ { 0 } } + \tau _ { N } \right) M _ { N } .\tag{C.14}
$$

Put $C _ { X } = 2 ( 1 + { \sqrt { 2 } } )$ . If $\alpha _ { \star } \sqrt { 2 } / \lambda _ { 0 } \leq 1 / 4$ and $\tau _ { + } \leq 1 / 4$ , then $\tau _ { N } \leq \tau _ { + } / \alpha$ makes the coefficient multiplying $M _ { N }$ in $( \mathrm { C } . 1 4 )$ at most $1 / 2$ . Thus $M _ { N } \leq C _ { X } \alpha ,$ , and (C.13) gives $\| q _ { n } \| \leq e ^ { - \lambda _ { 0 } \tau _ { n } } +$ $C _ { X } \alpha / \lambda _ { 0 } { \mathrm { ~ f o r ~ } } 0 \leq n \leq N$

For a lower bound on $X _ { N }$ , use $Y _ { N } - Z _ { N } \ge - \sqrt { 2 } \| q _ { N } \|$ in (C.11), together with (C.13) and $M _ { N } \leq$ $C _ { X } \alpha$ , to obtain

$$
\frac { X _ { N } } { \alpha } \geq 1 - \sqrt { 2 } e ^ { - \lambda _ { 0 } \tau _ { N } } - \frac { \sqrt { 2 } } { \lambda _ { 0 } } M _ { N } - \tau _ { N } M _ { N } \geq 1 - \sqrt { 2 } e ^ { - \lambda _ { 0 } \tau _ { N } } - \frac { \sqrt { 2 } C _ { X } } { \lambda _ { 0 } } \alpha - C _ { X } \alpha \tau _ { N } .\tag{C.15}
$$

Choose $\tau _ { - } = \lambda _ { 0 } ^ { - 1 } \log ( 4 \sqrt { 2 } )$ , take $\tau _ { + } \leq$ min $\left\{ 1 / 4 , 1 / ( 4 C _ { X } ) \right\}$ , and reduce $\alpha _ { \star }$ so that

$$
\alpha _ { \star } \leq \operatorname* { m i n } \left\{ \frac { \lambda _ { 0 } } { 4 \sqrt { 2 } C _ { X } } , 1 , \frac { \tau _ { + } } { 2 ( \tau _ { - } + 2 \bar { h } ) } \right\} .
$$

For $\tau _ { - } \le \tau _ { N } \le \tau _ { + } / \alpha$ , the three subtracted terms $\sqrt { 2 } e ^ { - \lambda _ { 0 } \tau _ { N } } , ( \sqrt { 2 } C _ { X } / \lambda _ { 0 } ) \alpha$ , and $C _ { X } \alpha \tau _ { N }$ in (C.15) are each at most $1 / 4 .$ , so $X _ { N } \geq \alpha / 4$ . The choices also satisfy the conditions used in $( \mathrm { C } . 1 4 )$ and give $\tau _ { - } < \tau _ { + } / \alpha$ . Since N was arbitrary, $| X _ { n } | \leq C _ { X } \alpha$ holds when $\tau _ { n } \leq \tau _ { + } / \alpha _ { : }$ , and $X _ { n } \geq \alpha / 4$ when $\tau _ { - } \le \tau _ { n } \le \tau _ { + } / \alpha$ . The argument permits $h _ { n } = 0$ and both definitions of ${ \dot { X } } _ { n } ^ { \prime }$ in (C.10).

To express the bounds in the original coordinates $\left( z _ { n } , a _ { n } , b _ { n } \right)$ , choose $c > 0$ no larger than $\bar { h } , \tau _ { + }$ and $1 / 4 .$ , and take $C \ge \tau _ { - }$ <sub>−</sub> large enough to bound $| X _ { n } | / \alpha$ and $| Y _ { n } | + | Z _ { n } - 1 |$ . Set $C _ { 0 } = \alpha _ { \star } ^ { - 1 }$ These constants are fixed before choosing the cutoff in Appendix C.3.2 and the component scales in Appendix C.3.3.

If $\rho \geq C _ { 0 } \kappa ^ { 2 }$ and $\ell \eta _ { n } \leq c \kappa .$ , then $\alpha \leq \alpha ,$ and $h _ { n } \leq \bar { h }$ . Substituting $z _ { n } = \delta X _ { n } / \kappa , a _ { n } = \delta Y _ { n }$ , and $b _ { n } = \delta ( Z _ { n } - 1 )$ into the bounds on $M _ { N }$ and $\| q _ { n } \|$ gives

$$
| z _ { n } | \leq C \frac { \kappa \delta } { \rho } , \qquad | a _ { n } | + | b _ { n } | \leq C \delta \quad \mathrm { w h e n } \quad \sum _ { i < n } \eta _ { i } \leq c \frac { \rho } { \ell \kappa } .
$$

If $\begin{array} { r } { C \kappa / \ell \leq \sum _ { i < n } \eta _ { i } \leq c \rho / ( \ell \kappa ) } \end{array}$ , then $\tau _ { - } \ \leq \ \tau _ { n } \ \leq \ \tau _ { + } / \alpha$ , and (C.15) gives $z _ { n } ~ \ge ~ c \kappa \delta / \rho$ . No monotonicity of the steps was used. □

## C.3.2 A SMOOTH NC-PŁ GAME WITH THE SAME MECHANISM

The shifted quadratic starts with a nonzero inner gap at the origin. We introduce a third inner coordinate s so that a gradient error can activate the shift from an inner maximizer. We also cut off the shift when $| a + b |$ is large to bound the Hessian globally while preserving the quadratic updates along the trajectory segment in Lemma 5.4. Fix $R _ { 0 }$ larger than the inner-coordinate bound in that lemma, and choose a smooth cutoff $\chi : \mathbb { R } \to [ 0 , 1 ]$ ] equal to one on $[ - R _ { 0 } , R _ { 0 } ] ,$ , zero outside $[ - 2 R _ { 0 } , 2 R _ { 0 } ]$ with $| \chi ^ { \prime } | \leq 1 / 4$ and bounded first two derivatives. The constant $R _ { 0 }$ is fixed before all smoothness and PŁ constants below.

For a provisional condition parameter $K \geq 1$ , define the residual $r _ { K } : \mathbb { R } ^ { 3 } \to \mathbb { R }$ and game $g _ { K } :$ $\mathbb { R } \times \mathbb { R } ^ { 3 } \to \mathbb { R } \mathbb { \ : } \mathbb { \ : }$ by

$$
r _ { K } ( a , b , s ) = a + b + \sqrt { K } \chi \bigg ( \frac { a + b } { \sqrt { K } } \bigg ) \sin ^ { 2 } \bigg ( \frac { \pi s } { 2 } \bigg ) , \qquad g _ { K } ( u ; a , b , s ) = u a - \frac { a ^ { 2 } + r _ { K } ( a , b , s ) ^ { 2 } } { 2 K } .\tag{C.16}
$$

The profile vanishes at $s = 0$ and has zero derivative there. ${ \mathrm { A t ~ } } s = \pm 1$ , it supplies the same shift and again has zero derivative. Thus either sign of a unit perturbation activates the shift, and the activation coordinate then stays fixed. Without the cutoff, the second derivative of $g _ { K }$ in s would grow linearly with $a + b .$

Lemma C.1 (Smoothness and inner PŁ of the single-perturbation game). There are universal $c , C > 0$ such that,for every $K \geq 1 , g _ { K }$ has a C-Lipschitz gradient,

$$
\operatorname* { m a x } _ { a , b , s } g _ { K } ( u ; a , b , s ) = \frac { K } { 2 } u ^ { 2 } , \qquad \frac 1 2 \left. \nabla _ { a , b , s } g _ { K } \right. ^ { 2 } \geq \frac { c } { K } \left[ \frac K 2 u ^ { 2 } - g _ { K } ( u ; a , b , s ) \right] .\tag{C.17}
$$

The origin is stationary.

Proof. Completing the square gives

$$
g _ { K } ( u ; a , b , s ) = \frac { K } { 2 } u ^ { 2 } - \frac { ( a - K u ) ^ { 2 } + r _ { K } ( a , b , s ) ^ { 2 } } { 2 K } .\tag{C.18}
$$

Both residuals vanish at $( a , b , s ) = ( K u , - K u , 0 )$ , proving the outer-objective formula and attainment. For the PŁ inequality, write $r = r _ { K } ( a , b , s ) , e = a - K u ,$ and

$$
p = \partial _ { a } r _ { K } = \partial _ { b } r _ { K } = 1 + \chi ^ { \prime } \bigg ( \frac { a + b } { \sqrt { K } } \bigg ) \sin ^ { 2 } \bigg ( \frac { \pi s } { 2 } \bigg ) \in [ 3 / 4 , 5 / 4 ] .
$$

The restriction on $\chi ^ { \prime }$ keeps p bounded away from zero, so the gradients in a and b control both residuals e and r without using the gradient in s. Indeed, $\partial _ { a } g _ { K } = - ( e + p r ) / K$ and $\partial _ { b } g _ { K } = - p r / K$ Since $e = ( e + p r ) - p r$ and $r ^ { 2 } \leq 1 6 ( p r ) ^ { 2 } / 9$

$$
e ^ { 2 } + r ^ { 2 } \le 2 ( e + p r ) ^ { 2 } + \left( 2 + \frac { 1 6 } { 9 } \right) ( p r ) ^ { 2 } \le 4 \bigl [ ( e + p r ) ^ { 2 } + ( p r ) ^ { 2 } \bigr ] ,\tag{C.19}
$$

$$
\frac { 1 } { 2 } \left\| \nabla _ { a , b , s } g _ { K } \right\| ^ { 2 } \geq \frac { e ^ { 2 } + r ^ { 2 } } { 8 K ^ { 2 } } = \frac { 1 } { 4 K } \left[ \frac { K } { 2 } u ^ { 2 } - g _ { K } ( u ; a , b , s ) \right] .\tag{C.20}
$$

This proves the inner-PŁ inequality.

For smoothness, $| \partial _ { a } r _ { K } | , | \partial _ { b } r _ { K } | \leq C$ and $| \partial _ { s } r _ { K } | \le C \sqrt { K }$ . Wherever a second derivative is nonzero, the cutoff implies $| a + b | \le 2 R _ { 0 } \sqrt { K }$ and $| r _ { K } | \le C \sqrt { K }$ . Direct differentiation gives second-derivative bounds of order $\dot { K } ^ { - 1 / 2 }$ in $( a , b )$ , order one for mixed derivatives with s, and order $K ^ { 1 / 2 }$ for $\partial _ { s s } r _ { K }$ Hence, for $y = ( a , b , s )$

$$
\left\| \nabla _ { y } ^ { 2 } ( r _ { K } ^ { 2 } / ( 2 K ) ) \right\| \leq \frac { \left\| \nabla _ { y } r _ { K } \right\| ^ { 2 } + | r _ { K } | \left\| \nabla _ { y } ^ { 2 } r _ { K } \right\| } { K } \leq C .\tag{C.21}
$$

Outside that region the residual is affine, so the same bound holds globally. The remaining terms u $\textstyle { a - a ^ { 2 } / ( 2 K ) }$ also have bounded Hessian for $K \geq 1$ . All first derivatives vanish at the origin.

At $s = \pm 1$ and $| a + b | \le R _ { 0 } \sqrt { K }$ , the residual satisfies

$$
r _ { K } ( a , b , s ) = a + b + \sqrt { K } , \qquad \partial _ { s } r _ { K } ( a , b , s ) = 0 .\tag{C.22}
$$

Thus s stays fixed and the other coordinates follow the shifted quadratic of Lemma 5.4 while the cutoff equals one. At $u \ : = \ : a \ : = \ : b \ : = \ : 0$ , a unit perturbation of s creates inner gap $1 / 2 .$ , since $g _ { K } ( 0 ; 0 , \bar { 0 } , \pm 1 ) = - 1 / 2 .$ . A unit perturbation of b, with $u = a = s = 0$ , instead creates inner gap $1 / ( 2 K )$ . The third coordinate therefore increases the gap by a factor K for the same perturbation magnitude. On the region where the cutoff equals one, the quadratic curvature in $( a , b )$ remains of order $1 / K$

## C.3.3 CHANGING VARIABLES AND THE CONTRIBUTION OF A SINGLE PERTURBATION

Write $c _ { r } , C _ { r } , C _ { \mathrm { r e s p } }$ for $c , C , C _ { 0 }$ in Lemma 5.4, and enlarge $C _ { \mathrm { r e s p } }$ so that $c _ { r } C _ { \mathrm { r e s p } } \geq 2 ( C _ { r } + 2 c _ { r } )$ Let $c _ { 0 } , C _ { 0 }$ be the constants in Lemma C.1. Choose universal $\dot { \gamma } , \theta > 0$ , fixed after the low-ratio construction, such that

$$
\theta C _ { 0 } \leq 1 , \qquad { \frac { \theta c _ { 0 } } { \gamma } } \geq 1 .\tag{C.23}
$$

For condition number κ, set

$$
K : = \gamma \kappa , \qquad L : = \theta \ell .\tag{C.24}
$$

In (5.8), set $g _ { \sigma , j } \equiv 0$ whenever $\eta _ { j } \sigma = 0 ;$ these components have zero gradients and gaps. Increase the universal lower bound on κ so that $K \geq 1$ . For every block $j \in \{ \bar { 0 } , \dots , T - 2 \}$ with $\eta _ { j } \sigma > 0$ write $y [ j ] = ( a [ j ] , b [ j ] , s [ j ] )$ and define

$$
g _ { \sigma , j } ( v [ j ] ; a [ j ] , b [ j ] , s [ j ] ) : = L ( \eta _ { j } \sigma ) ^ { 2 } g _ { K } \left( \frac { v [ j ] } { \eta _ { j } \sigma } ; \frac { a [ j ] } { \eta _ { j } \sigma } , \frac { b [ j ] } { \eta _ { j } \sigma } , \frac { s [ j ] } { \eta _ { j } \sigma } \right) .\tag{C.25}
$$

The squared amplitude compensates exactly for the spatial rescaling. Writing $w \in \mathbb { R } ^ { 4 }$ for the component coordinates, we have

$$
\nabla _ { w } ^ { 2 } \left[ ( \eta _ { j } \sigma ) ^ { 2 } g _ { K } \left( \frac { w } { \eta _ { j } \sigma } \right) \right] = \nabla ^ { 2 } g _ { K } \left( \frac { w } { \eta _ { j } \sigma } \right) .\tag{C.26}
$$

Thus changing the positive amplitude $\eta _ { j } \sigma$ does not change the Hessian bound. Define the shift as a function of the perturbed inner coordinate

$$
\delta _ { j } ( s ) = \sqrt { K } \eta _ { j } \sigma \sin ^ { 2 } \biggl ( \frac { \pi s } { 2 \eta _ { j } \sigma } \biggr ) .\tag{C.27}
$$

The profile and its derivative vanish at zero, while $\delta _ { j } ( \pm \eta _ { j } \sigma ) = \sqrt { K } \eta _ { j } \sigma$ and $\delta _ { j } ^ { \prime } ( \pm \eta _ { j } \sigma ) = 0$ . Thus either sign of the gradient error activates the same shift and leaves its derivative zero. Writing $\chi _ { j } ( q ) : =$ $\chi ( q / ( \sqrt { K } \eta _ { j } \sigma ) )$ gives exactly the component formula (5.9). Its displacement $\chi _ { j } ( a [ j ] + b [ j ] ) \delta _ { j } ( s [ j ] )$ depends on the inner coordinates, and the cutoff keeps second derivatives bounded globally. Once $s [ \bar { j } ]$ is fixed at either activated value, the updates of $\bar { ( } v [ j ] , a [ j ] , b [ j ] )$ agree with those of $g _ { \delta }$ in (5.6), with parameters $( L , K , \sqrt { K } \eta _ { j } \sigma )$ , wherever $| a [ j ] + b [ j ] | \leq R _ { 0 } \sqrt { K } \eta _ { j } \sigma$

Its full-gradient Lipschitz constant is at most $L C _ { 0 } \ \leq \ \ell ,$ and its inner-PŁ parameter is at least $L c _ { 0 } / K \ge \ell / \kappa$ . Also $\Phi _ { \sigma , j } ( v [ j ] ) = L K ( v [ j ] ) ^ { 2 } / 2$ , and the origin is both an outer minimizer and an inner maximizer. These properties hold for every positive amplitude, including the initial large steps; the step-size restriction in Lemma 5.5 is needed only for its trajectory bound.

Lemma 5.5 (Contribution of a single inner-gradient perturbation). There are universal constants $c , C , c _ { \mathrm { s t e p } } , c _ { \mathrm { h i g h } } > 0$ and $\kappa _ { \mathrm { h i g h } } \geq 2$ such that, for $\ell , \overset { \cdot } { \sigma } > 0 , \kappa \geq \kappa _ { \mathrm { h i g h } } , \rho \geq \operatorname* { m a x } \{ 1 , c _ { \mathrm { h i g h } } \kappa ^ { 2 } \}$ , an integer ${ \bar { T } } \geq { \bar { 2 } } ,$ any predetermined step sizes $\eta _ { 0 } \geq \cdot \cdot \cdot \geq \eta _ { T - 2 } \geq \bar { 0 }$ and index $0 \leq j \leq T - 2$ with $0 < \ell \eta _ { j } \le c _ { \mathrm { s t e p } } \kappa$ admit an ℓ-smooth $N C - ( \ell / \kappa ) – P E$ component $g _ { \sigma , j } : \mathbb { R } \times \mathbb { R } ^ { 3 } \to \bar { \mathbb { R } }$ . Its outer objective $\begin{array} { r } { \Phi _ { \sigma , j } ( v ) : = \operatorname* { m a x } _ { y \in \mathbb { R } ^ { 3 } } g _ { \sigma , j } ( v ; y ) } \end{array}$ is quadratic with curvature $\bar { \Phi } _ { \sigma , j } ^ { \prime \prime } \asymp$ ℓκ and initial gaps $\begin{array} { r } { \Phi _ { \sigma , j } ( 0 ) - \operatorname* { m i n } _ { v } \Phi _ { \sigma , j } ( v ) = \Phi _ { \sigma , j } ( 0 ) - g _ { \sigma , j } ( 0 ; 0 ) = 0 } \end{array}$ . Starting from the origin with one equiprobable ±σ error in the third inner gradient at update j and exact gradients otherwise, both $S G D A _ { S i m }$ and $S G D A _ { A l t }$ generate outer iterates $v _ { t } [ j ]$ satisfying

$$
\sum _ { t < T } \mathbb { E } | \Phi _ { \sigma , j } ^ { \prime } ( v _ { t } [ j ] ) | ^ { 2 } \geq c \frac { \ell \kappa ^ { 4 } \sigma ^ { 2 } \eta _ { j } } { \rho } \quad i f \quad \sum _ { t = j + 1 } ^ { T - 2 } \eta _ { t } \geq C \frac { \rho } { \ell \kappa } .\tag{5.13}
$$

Proof. Once the error activates a shift, the component agrees with the shifted quadratic as long as its inner coordinates stay inside the cutoff. We verify this agreement over the displacement window, then use the non-increasing steps to count the iterates in that window.

Let $\xi$ take values $\pm 1$ with equal probability. All coordinates remain zero until update $j ,$ which sends $s [ j ]$ to $\eta _ { j } \sigma \xi$ and leaves the other coordinates zero. At either sign, $\delta _ { j } ( s [ j ] ) = \sqrt { K } \eta _ { j } \sigma$ and $\delta _ { j } ^ { \prime } ( s [ j ] ) \dot { = } 0$ . Thus $s [ j ]$ stays fixed and both signs activate the same shift.

Choose

$$
c _ { \mathrm { h i g h } } = \gamma ^ { 2 } C _ { \mathrm { r e s p } } , \qquad c _ { \mathrm { s t e p } } = \frac { c _ { r } \gamma } { \theta } ,
$$

and take the constant in the subsequent-sum condition at least $c _ { r } / ( \theta \gamma )$ . Assume that this condition holds. For $j + 1 \le t \le T - 1$ , write

$$
\tau _ { t } = \frac { L } { K } \sum _ { r = j + 1 } ^ { t - 1 } \eta _ { r } .
$$

By the hypotheses, $\rho \geq C _ { \mathrm { r e s p } } K ^ { 2 }$ , each increment of $\tau _ { t }$ is at most $L \eta _ { j } / K \le c _ { r }$ , and $\tau _ { T - 1 } \geq c _ { r } \rho / K ^ { 2 }$ Consider the exact-gradient trajectory $\left( v _ { t } , a _ { t } , b _ { t } \right)$ of the uncut shifted quadratic in (5.6), with parameters $( L , K , \sqrt { K } \eta _ { j } \sigma )$ , starting at the origin at time $j + 1$ . Lemma 5.4 gives

$$
v _ { t } \geq c _ { r } \frac { K ^ { 3 / 2 } \eta _ { j } \sigma } { \rho }
$$

$$
\mathrm { w h e n } C _ { r } \leq \tau _ { t } \leq \frac { c _ { r } \rho } { K ^ { 2 } } ,
$$

$$
| a _ { t } | + | b _ { t } | \le C _ { r } \sqrt { K } \eta _ { j } \sigma
$$

$$
\mathrm { w h e n } \tau _ { t } \leq \frac { c _ { r } \rho } { K ^ { 2 } } .
$$

Let i and k be the first indices with $\tau _ { i } \geq C _ { r }$ and $\tau _ { k } \geq c _ { r } \rho / K ^ { 2 }$ . They exist because $\tau _ { T - 1 } \geq c _ { r } \rho / K ^ { 2 }$ moreover, $c _ { r } \rho / K ^ { 2 } \geq 2 ( C _ { r } + 2 c _ { r } )$ and each increment is at most $c _ { r } , \mathrm { s o } i < k \leq T - 1$

For $j + 1 \leq t < k$ , the quadratic trajectory satisfies $| a _ { t } + b _ { t } | \leq C _ { r } \sqrt { K } \eta _ { j } \sigma < R _ { 0 } \sqrt { K } \eta _ { j } \sigma$ . Starting from the common state at time $j + 1$ , induction therefore identifies it with the actual component trajectory through iterate $k - 1$ . Every gradient query needed for this induction lies where the cutoff is identically one. For $\mathrm { S G D A _ { A l t } }$ , the inner-gradient query after the primal update has the same $a + b ,$ so the agreement also holds at that query. Thus $v _ { t } [ j ] = { \dot { v } } _ { t }$ on the interval we use below.

Since $\eta _ { t } \leq \eta _ { j }$ for $t > j$ and the first crossing of $C _ { r }$ overshoots by at most $c _ { r }$

$$
( k - i ) \frac { L \eta _ { j } } { K } \ge \tau _ { k } - \tau _ { i } \ge \frac { c _ { r } \rho } { K ^ { 2 } } - C _ { r } - c _ { r } \ge \frac { c _ { r } \rho } { 2 K ^ { 2 } } .\tag{C.28}
$$

For every $i \ \leq \ t \ < \ k$ , the displacement bound and $\Phi _ { \sigma , j } ^ { \prime } ( v ) ~ = ~ L K v$ give $\Phi _ { \sigma , j } ^ { \prime } ( v _ { t } [ j ] ) \ \geq$ $c _ { r } L K ^ { 5 / 2 } \eta _ { j } \sigma / \rho .$ These are $k - i$ reported iterates, so

$$
\sum _ { t < T } \mathbb { E } | \Phi _ { \sigma , j } ^ { \prime } ( v _ { t } [ j ] ) | ^ { 2 } \geq ( k - i ) \left( \frac { c _ { r } L K ^ { 5 / 2 } \eta _ { j } \sigma } { \rho } \right) ^ { 2 } \geq \frac { c _ { r } ^ { 3 } } { 2 } \frac { L K ^ { 4 } \sigma ^ { 2 } \eta _ { j } } { \rho } \geq c \frac { \ell \kappa ^ { 4 } \sigma ^ { 2 } \eta _ { j } } { \rho } .\tag{C.29}
$$

All constants are independent of the horizon and step-size sequence, and the same displacement estimate holds for either noise sign. In the aggregate game, the fixed oracle constructed in Lemma C.2 generates each component’s prescribed perturbation at its designated update. □

We now realize the prescribed perturbations through one fixed oracle and sum their contributions. Only components with enough subsequent cumulative steps contribute to the estimate; the omitted components account for a bounded total step size, leaving a constant fraction of $S$ when $S$ is sufficiently large.

Lemma C.2 (Lower bound from the noise components). There are universal $c , C > 0$ with the following property. Fix $\ell , \sigma > 0 , \kappa \geq \kappa _ { \mathrm { h i g h } } , \rho \geq \operatorname* { m a x } \{ 1 , c _ { \mathrm { h i g h } } \kappa ^ { 2 } \}$ , an integer $T \geq 2 , a$ predetermined schedule $\eta _ { 0 } \geq \cdot \cdot \cdot \geq \eta _ { T - 2 } \geq 0$ , and $m \in \{ 0 , \ldots , T - \overset { \cdot } { 1 } \}$ with $\ell \eta _ { t } < 8 f o r$ m $\leq t \leq T - 2$ Set $\begin{array} { r } { S = \sum _ { t = m } ^ { T - 2 } \eta _ { t } } \end{array}$ . The components in $( 5 . 8 )$ can be chosen ℓ-smooth and $N C _ { - } ( \ell / \kappa ) { - } P E$ , with $\Phi _ { \sigma , j } ( 0 ) = \overline { { { \operatorname* { m i n } } } } \Phi _ { \sigma , j } = g _ { \sigma , j } ( 0 , \mathrm { \bar { 0 } } ) = 0 .$ . Both $S G D A _ { S i m }$ and $S G D A _ { A l t }$ on $g _ { \sigma }$ , initialized at the origin with exact outer gradients and the fixed inner oracle $G _ { y } ( v , y ; \xi ) = \nabla _ { y } g _ { \sigma } ( v , y ) + \sigma \xi e ( y )$ from Section 5.2.3, satisfy

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left. \nabla \Phi _ { \sigma } ( v _ { t } ) \right. ^ { 2 } \geq c \frac { \kappa ^ { 4 } \ell \sigma ^ { 2 } S } { \rho T } \quad i f S \geq \frac { C \rho } { \kappa \ell } .\tag{C.30}
$$

Proof. Let $J = \{ 0 , \ldots , T - 2 \}$ . Set $g _ { \sigma , j } \equiv 0$ for $\eta _ { j } = 0 .$ , and choose every positive-step block from (C.25). The preceding smoothness, inner- $\mathbf { \nabla \cdot P E }$ , and initialization estimates apply to all these blocks, with no restriction on the initial step sizes.

The selector $e ( y )$ from Section 5.2.3 chooses the $s [ j ]$ -coordinate of the smallest $j \in J$ with $s [ j ] = 0$ or is zero if no such index exists. Its selection regions are finite intersections of coordinate hyperplanes and their complements, so it is measurable and has norm at most one. Fresh independent uniform signs make the inner oracle pointwise unbiased, with variance $\sigma ^ { 2 } \left\| e ( y ) \right\| ^ { 2 } \leq \sigma ^ { 2 }$

We next verify that this oracle activates the intended component at each positive update. The induction invariant is that, at the start of a positive update $j ,$ , all blocks with index at least j remain at the origin, while every earlier block has $s [ i ] = \pm \eta _ { i } \sigma \bar { \neq } 0$ . Stationarity at the origin keeps the unactivated blocks fixed during the primal update. Thus $e ( y )$ selects the third inner coordinate of block $j$ for either update order, and update j sends $s [ j ]$ to $\pm \eta _ { j } \sigma$ while leaving its other coordinates at zero.

At these peaks, $\partial _ { s } r _ { K } ( a , b , \pm 1 ) = 0$ for every $a , b ,$ including outside the cutoff region. All subsequent exact $s [ j ] { \cdot } \underline { { \operatorname { g } } } 1$ adients therefore vanish, so $s [ j ]$ stays fixed even if a large step changes the cutoff factor. The remaining unactivated blocks stay at the origin, completing the induction. Since the schedule is non-increasing, zero steps are trailing. The selector then repeatedly chooses the first zero-step block, but the zero step sizes leave every iterate unchanged. Consequently, each nonzero block receives one equiprobable ±σ error at its assigned update and exact gradients otherwise, with independent signs across blocks.

Since the coordinates are disjoint,

$$
\left\| \nabla \Phi _ { \sigma } ( v _ { t } ) \right\| ^ { 2 } = \sum _ { j \in J } \left\| \nabla \Phi _ { \sigma , j } ( v _ { t } [ j ] ) \right\| ^ { 2 } .
$$

Let $C _ { \mathrm { o n e } }$ be the remaining-step constant in Lemma 5.5, and enlarge $\kappa _ { \mathrm { h i g h } }$ so that $8 \leq c _ { \mathrm { s t e p } } \kappa .$ Every positive step with index $t \geq m$ then satisfies its step-size condition. Define the tail indices $J _ { \mathrm { t a i l } } : = \{ m , \dots , T - 2 \}$ and $J _ { \mathrm { g o o d } } : = \{ j \in J _ { \mathrm { t a i l } } : \sum _ { t = i + 1 } ^ { T - 2 } \eta _ { t } \geq C _ { \mathrm { o n e } } \rho / ( \kappa \ell ) \}$ . Non-increasingness ensures $\eta _ { j } > 0$ for every $j \in J _ { \mathrm { g o o d } } . \ \mathrm { I f } \ J _ { \mathrm { t a i l } } \ \backslash \ J _ { \mathrm { g o o d } }$ is nonempty, its first index $j _ { 0 }$ satisfies

$$
\sum _ { j \in J _ { \mathrm { t a i l } } \setminus J _ { \mathrm { g o o d } } } \eta _ { j } \le \sum _ { t = j _ { 0 } } ^ { T - 2 } \eta _ { t } < \frac { 8 } { \ell } + \frac { C _ { \mathrm { o n e } } \rho } { \kappa \ell } \le \frac { C _ { 1 } \rho } { \kappa \ell } , \qquad C _ { 1 } : = C _ { \mathrm { o n e } } + \frac { 8 } { c _ { \mathrm { h i g h } } } ,
$$

where the last inequality uses $\rho \geq c _ { \mathrm { h i g h } } \kappa ^ { 2 }$ and $\kappa \geq 1$ . If the excluded set is empty, its total step size is zero. Taking $C : = 2 C _ { 1 }$ , the condition $S \geq C \rho / ( \kappa \ell )$ implies $\sum _ { j \in J _ { \mathrm { g o o d } } } \eta _ { j } \geq \dot { S } / 2$ . Summing (5.13) over these indices gives

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| \nabla \Phi _ { \sigma } ( v _ { t } ) \right\| ^ { 2 } \geq c \frac { \kappa ^ { 4 } \ell \sigma ^ { 2 } } { \rho T } \sum _ { j \in J _ { \mathrm { g o o d } } } \eta _ { j } \geq \frac { c } { 2 } \frac { \kappa ^ { 4 } \ell \sigma ^ { 2 } S } { \rho T } .
$$

Reducing the universal constant c proves the claim.

## C.4 PROOF OF THE LOWER BOUND FOR f<sub>stable</sub>

The aggregate construction already supplies a fixed oracle using only the inner coordinates. We add the deterministic corridor and combine its lower bound with the contributions of the noise components.

Lemma 5.6 (Stable-step lower bound and admissibility). There are universal $c , c _ { \mathrm { h i g h } } > 0$ and $\kappa _ { \mathrm { h i g h } } \geq 2$ such that, ${ { \mathrm { \it ~ f o r ~ } } \ell , \Delta > 0 , \sigma \geq 0 , \kappa \geq \kappa _ { \mathrm { h i g h } } , \rho \geq \operatorname* { m a x } \{ 1 , c _ { \mathrm { h i g h } } \kappa ^ { 2 } \} }$ , an integer $T \geq 1$ , and predetermined steps $\eta _ { 0 } \geq \cdot \cdot \cdot \geq \eta _ { T - 2 } \geq 0 ,$ , the following holds. Let m count the entries $\eta _ { t } \ge 8 / \ell$ and assume $m + 1 < T / 2 .$ . For each of $S G D A _ { S i m }$ and $S G D A _ { A l t } ,$ , the game f<sub>stable</sub> given by (5.4) and afixed oracle can be chosen as an instance in $\mathcal { C } ( \ell , \kappa , \Delta , \sigma )$ with initial outer gap $\Delta$ and inner gap zero at the origin. Writing $\begin{array} { r } { S = \sum _ { t = m } ^ { T - 2 } \eta _ { t } } \end{array}$ and $\rho \Delta / 0 : = + \infty ,$ , its iterates satisfy

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left. \nabla \Phi _ { \mathrm { s t a b l e } } ( u _ { t } , v _ { t } ) \right. ^ { 2 } \geq c \operatorname* { m i n } \left\{ \ell \Delta , \operatorname* { m a x } \left\{ \frac { \rho \Delta } { S } , \frac { \kappa ^ { 4 } \ell \sigma ^ { 2 } S } { \rho T } \right\} \right\} .\tag{5.14}
$$

Proof. For the chosen algorithm, $\operatorname { S G D A } _ { \operatorname { S i m } }$ or $\mathbf { S G D A _ { A l t } }$ , use the predetermined non-increasing step sizes in the statement. Let $J = \{ 0 , \ldots , T - 2 \}$ , and choose the corridor from Lemma 5.3. For $\sigma > 0$ , choose the aggregate noise game $g _ { \sigma }$ and its fixed inner oracle from Lemma C.2; for $\sigma = 0$ set $g _ { \sigma , j } \equiv 0$ for every $j \in J$ and use exact gradients. Use these games in $( 5 . 4 ) .$ , take exact outer gradients, and initialize every coordinate at zero. In either case, the dimensions are $d _ { x } = 1 + | J | = T$ and $d _ { y } = 3 | J | = 3 ( T - 1 )$

We first verify the game assumptions. Each component has full-gradient Lipschitz constant at most $\ell .$ Since the components use disjoint coordinates, summing the squared component inequalities gives

$$
\big \| \nabla f _ { \mathrm { s t a b l e } } ( u , v ; y ) - \nabla f _ { \mathrm { s t a b l e } } ( \widetilde { u } , \widetilde { v } ; \widetilde { y } ) \big \| ^ { 2 } \leq \ell ^ { 2 } \big ( | u - \widetilde { u } | ^ { 2 } + \| v - \widetilde { v } \| ^ { 2 } + \| y - \widetilde { y } \| ^ { 2 } \big ) .
$$

Thus the smoothness assumption holds without a dimension factor. Each inner maximum is finite and attained, so their finite product attains the maximum of $f _ { \mathrm { s t a b l e } }$ . Its outer objective is

$$
\Phi _ { \mathrm { s t a b l e } } ( u , v ) = g _ { \mathrm { c o r r i d o r } } ( u ) + \Phi _ { \sigma } ( v ) = g _ { \mathrm { c o r r i d o r } } ( u ) + \frac { L K } { 2 } \sum _ { \stackrel { j \in J } { \eta _ { j } \sigma > 0 } } ( v [ j ] ) ^ { 2 } .
$$

It is differentiable and lower bounded. Summing the component inner-PŁ inequalities gives

$$
\begin{array} { r l r } {  { \frac { 1 } { 2 } \| \nabla _ { y } f _ { \mathrm { s t a b l e } } ( u , v ; y ) \| ^ { 2 } = \frac { 1 } { 2 } \sum _ { j \in J } \| \nabla _ { y [ j ] } g _ { \sigma , j } ( v [ j ] ; y [ j ] ) \| ^ { 2 } } } \\ & { } & { \geq \frac { \ell } { \kappa } \sum _ { j \in J } \bigl ( \Phi _ { \sigma , j } ( v [ j ] ) - g _ { \sigma , j } ( v [ j ] ; y [ j ] ) \bigr ) } \\ & { } & { = \frac { \ell } { \kappa } \big ( \Phi _ { \mathrm { s t a b l e } } ( u , v ) - f _ { \mathrm { s t a b l e } } ( u , v ; y ) \big ) . } \end{array}
$$

The corridor supplies outer gap $\Delta ,$ each perturbation component has zero outer gap at the origin, and every component is initially maximized in its inner variables. Hence the initial outer and inner gaps are $\Delta$ and zero. These conclusions also hold for the corridor alone, with a vacuous inner-PŁ inequality.

Adding the corridor does not change any noise-component trajectory, because its coordinates are disjoint and the inner oracle depends only on $( v , y )$ . The oracle verification in the proof of Lemma C.2 therefore also applies to $f _ { \mathrm { s t a b l e } }$ . Together with the game assumptions above, this proves that the origin-initialized instance belongs to $\mathcal { C } ( \ell , \kappa , \Delta , \sigma )$

It remains to prove the lower bound. Let C be the threshold constant in Lemma C.2, and increase $\kappa _ { \mathrm { h i g h } }$ so that $\kappa _ { \mathrm { h i g h } } \geq C$ . If $S ~ < ~ C \rho / ( \kappa \ell )$ , including $S ~ = ~ 0$ , the corridor gives $c \ell \Delta .$ , since $( T - m ) / T > 1 / \breve { 2 }$ and $S / \rho < 1 / \ell .$ . This proves the capped bound.

Otherwise $S > 0$ , and Lemma C.2 applies when $\sigma > 0 ;$ when $\sigma = 0 ,$ , the noise contribution is zero. Since the squared outer gradients of disjoint components add, their bounds give

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left. \nabla \Phi _ { \mathrm { s t a b l e } } ( u _ { t } , v _ { t } ) \right. ^ { 2 } \geq c \left[ \operatorname* { m i n } \Bigl \{ \ell \Delta , \frac { \rho \Delta } { S } \Bigr \} + \frac { \kappa ^ { 4 } \ell \sigma ^ { 2 } S } { \rho T } \right] .\tag{C.31}
$$

The bracket dominates the capped maximum in (5.14), proving the lemma.

## D PROOFS FOR THE LOW-RATIO LOWER BOUND

We prove the two properties used by Theorem 6.3: small steps preserve a region of nonzero outer gradient, and the initially larger steps can be made to end inside that region. The first requires a smooth game with a stable periodic solution; the second modifies that specific game away from the solution. The final rescaling and initialization checks appear in Appendix A.3.

## D.1 CONSTRUCTION OF $_ { g _ { \mathrm { c y c l e } } }$

Lemma 6.1 (An attracting periodic orbit of GDA flow in $g _ { \mathrm { c y c l e } } )$ . There are universal $C , \mu , h _ { \star } > 0$ and a smooth residual R such that the base game $g _ { \mathrm { c y c l e } } : \mathbb { R } ^ { 2 } \times \mathbb { R } ^ { 4 } $ R in (6.2) is jointly C-smooth and affine in u, has outer objective $\begin{array} { r } { \Phi _ { g _ { \mathrm { c y c l e } } } ( u ) = \frac { 1 } { 2 } \left. u \right. ^ { 2 } } \end{array}$ , and satisfies the inner $\mu { - } P \mathbf { \mathcal { L } }$ inequality. The path (6.3), with $y _ { \star } = ( a _ { \star } , b _ { \star } )$ , traces a locally stable 2π-periodic orbit Γ of GDA flow. There is a compact neighborhood U of Γ such that both $S G D A _ { S i m }$ and $S G D A _ { A l t }$ , started in U with exact gradients and equal primal and dual steps $h _ { t } \in [ 0 , h _ { \star } ]$ , satisfy $( u _ { t } , y _ { t } ) \in \mathcal { U }$ and $\begin{array} { r } { \left\| \nabla \Phi _ { g _ { \mathrm { c y c l e } } } ( u _ { t } ) \right\| = \| u _ { t } \| \geq \frac { 1 } { 2 } } \end{array}$ for all $t \geq 0$ , where $y _ { t } = ( a _ { t } , b _ { t } )$

Proof. We first verify global smoothness and inner $\mathrm { P E } ,$ , then check that the prescribed curve solves the GDA equations. A rotating frame separates phase shifts along the orbit from departures that decay toward it. We use this decay to construct a Lyapunov function whose sublevel set is invariant under both discrete updates.

Global smoothness and inner PŁ. We construct $g _ { \mathrm { c y c l e } } ( u , a , b ) = \sqrt { 2 } u ^ { \mathsf { T } } a - \left\| a \right\| ^ { 2 } - \left\| R ( a , b ) \right\| ^ { 2 } / 2$ on $\mathbb { R } ^ { 2 } \times \mathbb { R } ^ { 2 } \times \mathbb { R } ^ { 2 }$ . Its affine dependence on u will be essential to the ratio scaling. The residual has the form $R ( a , b ) = b + \chi _ { b } ( b ) \psi ( a )$ : the values and first derivatives of $\psi$ produce the periodic solution, its second derivatives control departures from that solution, and the cutoffs enforce global smoothness.

Let $\begin{array} { r } { J = \left( \begin{array} { c c } { 0 } & { - 1 } \\ { 1 } & { 0 } \end{array} \right) } \end{array}$ be rotation by a right angle. The target path is $\boldsymbol { u } _ { \star } ( t ) = ( \cos t , \sin t ) ^ { \sf T }$ , with $a _ { \star } = - J u _ { \star } / \sqrt { 2 }$ and $\begin{array} { r } { b _ { \star } = u _ { \star } / \sqrt { 2 } } \end{array}$ . Since $\dot { u } = - \sqrt { 2 } a$ and, where the cutoffs equal one, $\dot { b } =$ $- ( b + \psi ( a ) )$ , the first and third equations require $\psi ( a _ { \star } ) = a _ { \star } - J a ,$ <sub>⋆</sub>. The remaining equation requires $\hat { D \psi } ( a _ { \star } ) ^ { \top } a _ { \star } = - 2 a _ { \star } + \bar { J a } _ { \star }$ . We satisfy these value and derivative conditions first, then check contraction through the remaining derivatives.

Choose a smooth cutoff $\chi _ { a } : [ 0 , \infty ) \to [ 0 , 1 ]$ that equals one on a neighborhood of $\textstyle { \frac { 1 } { 2 } }$ and vanishes outside a slightly larger compact interval. Define the polynomials

$$
c ( q ) : = 1 - 3 ( q - \textstyle { \frac { 1 } { 2 } } ) + \frac { 7 } { 2 } ( q - \textstyle { \frac { 1 } { 2 } } ) ^ { 2 } , \qquad d ( q ) : = - 1 + ( q - \textstyle { \frac { 1 } { 2 } } ) ,\tag{D.1}
$$

and set

$$
\psi ( a ) : = \chi _ { a } ( \left\| a \right\| ^ { 2 } ) \left[ c ( \left\| a \right\| ^ { 2 } ) a + d ( \left\| a \right\| ^ { 2 } ) J a \right] .\tag{D.2}
$$

The orbit conditions require $c ( 1 / 2 ) = 1 , d ( 1 / 2 ) = - 1$ , and $c ^ { \prime } ( 1 / 2 ) = - 3$ . The remaining choices $c ^ { \prime \prime } ( 1 / 2 ) = 7$ and $d ^ { \prime } ( 1 / 2 ) = 1$ make the transverse linearization stable, as verified below. The map ψ is smooth and compactly supported. Choose a smooth radial cutoff $\chi _ { b } : \mathbb { R } ^ { 2 }  [ 0 , 1 ]$ that equals one on a neighborhood of the circle $\| b \| = 1 / \sqrt { 2 }$ , vanishes outside a sufficiently large ball, and satisfies

$$
\left\| \psi \right\| _ { \infty } \| \nabla \chi _ { b } \| _ { \infty } \leq \frac { 1 } { 2 } .\tag{D.3}
$$

Define

$$
R ( a , b ) : = b + \chi _ { b } ( b ) \psi ( a ) ,\tag{D.4}
$$

and

$$
g _ { \mathrm { c y c l e } } ( u , a , b ) : = \sqrt 2 u ^ { \mathsf { T } } a - \left\| a \right\| ^ { 2 } - \frac 1 2 \left\| R ( a , b ) \right\| ^ { 2 } .\tag{D.5}
$$

For every fixed a and target $w \in \mathbb { R } ^ { 2 }$ , the map $b \mapsto w - \chi _ { b } ( b ) \psi ( a )$ is a contraction with factor at most $1 / 2$ by (D.3). Its unique fixed point solves $R ( a , b ) = w$ . Thus $\displaystyle { \dot { b } } \mapsto R ( a , b )$ is bijective. Moreover,

$$
\begin{array} { r } { D _ { b } R = I + \psi ( a ) \nabla { \chi } _ { b } ( b ) ^ { \top } , \qquad \| D _ { b } R - I \| \leq \frac { 1 } { 2 } , \qquad \big \| ( D _ { b } R ) ^ { - 1 } \big \| \leq 2 . } \end{array}
$$

The inverse function theorem gives a smooth inverse. Thus $T ( a , b ) : = ( \sqrt { 2 } a , R ( a , b ) )$ is a smooth bijection. Its triangular derivative has diagonal blocks ${ \sqrt { 2 } } I _ { 2 }$ and $D _ { b } R ;$ boundedness of $D _ { a } R$ and $( \check { D } _ { b } R ) ^ { - 1 } \mathrm { ~ g i v e s ~ } \big \| ( \check { D } T ) ^ { - 1 } \big \| \leq C .$ , and hence $\left\| D T ^ { \mathsf { T } } q \right\| \geq C ^ { - { \overline { { 1 } } } } \left\| q \right\|$ for every $( a , b )$ and $q \in \mathbb { R } ^ { 4 }$

All derivatives of $\psi$ are bounded. Whenever a second derivative of R is nonzero, both a and b lie in fixed compact sets; elsewhere R is affine. Hence DR and $R D ^ { 2 } R$ are globally bounded, and so is $\nabla ^ { 2 } \left\| \boldsymbol { R } \right\| ^ { 2 } / 2$ . The full gradient of $g _ { \mathrm { c y c l e } }$ is therefore universally Lipschitz.

For each $^ { a , }$ maximizing over b cancels the residual. Maximizing over a then gives $\Phi _ { g _ { \mathrm { c y c l e } } } ( u ) =$ $\left\| u \right\| ^ { 2 } / 2$ and

$$
\Phi _ { g _ { \mathrm { c y c l e } } } ( u ) - g _ { \mathrm { c y c l e } } ( u , a , b ) = \frac { 1 } { 2 } \left. \left. u - \sqrt { 2 } a \right. \right. ^ { 2 } + \frac { 1 } { 2 } \left. \left. R ( a , b ) \right. \right. ^ { 2 } .\tag{D.6}
$$

The surjectivity and transposed-derivative bound for T verify the hypotheses of Lemma B.1 with outer vector $( u , 0 )$ , giving inner $\mu { \mathrm { - P E } }$ for a universal $\mu > 0$

The periodic orbit and its attraction. Along $( 6 . 3 ) , b _ { \star } = J a _ { \star } , \| a _ { \star } \| ^ { 2 } = \frac { 1 } { 2 }$ , and both cutoffs are constant. Hence

$$
\psi ( a _ { \star } ) = a _ { \star } - J a _ { \star } , \qquad R ( a _ { \star } , b _ { \star } ) = a _ { \star } .
$$

On the neighborhood where both cutoff functions equal one, writing $q = \left. a \right. ^ { 2 }$

$$
D \psi ( a ) ^ { \mathsf { T } } a = [ c ( q ) + 2 q c ^ { \prime } ( q ) ] a - d ( q ) J a .\tag{D.7}
$$

The values $c ( { \textstyle { \frac { 1 } { 2 } } } ) = 1 , c ^ { \prime } ( { \textstyle { \frac { 1 } { 2 } } } ) = - 3 .$ , and $d ( \textstyle { \frac { 1 } { 2 } } ) = - 1$ give

$$
\begin{array} { r } { \dot { u } _ { \star } = - \sqrt { 2 } a _ { \star } = J u _ { \star } , \quad \dot { a } _ { \star } = \sqrt { 2 } u _ { \star } - 2 a _ { \star } - D \psi ( a _ { \star } ) ^ { \top } a _ { \star } = J a _ { \star } , \quad \dot { b } _ { \star } = - R ( a _ { \star } , b _ { \star } ) = J b _ { \star } . } \end{array}
$$

Thus (6.3) solves the GDA differential equation and is 2π-periodic.

Along this solution, $- \nabla _ { u } g _ { \mathrm { c y c l e } } = J u _ { \star } , \nabla \Phi _ { g _ { \mathrm { c y c l e } } } = u _ { \star }$ , and $\left\| \nabla _ { a , b } g _ { \mathrm { c y c l e } } \right\| ^ { 2 } = \left\| \dot { a } _ { \star } \right\| ^ { 2 } + \left\| \dot { b } _ { \star } \right\| ^ { 2 } = 1$ By (D.6), the inner gap equals $5 / 4$ . Inner ascent decreases this gap at rate one, while the motion of u increases it at the same rate: $\left\| \nabla _ { u } g _ { \mathrm { c y c l e } } \right\| ^ { 2 } = 1$ and $u _ { \star } ^ { \mathsf { T } } \dot { u } _ { \star } = 0$ . These contributions cancel, so the gap remains positive despite global inner PŁ.

The definitions imply $\psi ( e ^ { t J } a ) = e ^ { t J } \psi ( a )$ and $R ( e ^ { t J } a , e ^ { t J } b ) = e ^ { t J } R ( a , b )$ Thus the derivative calculation at any point of the periodic solution is the rotation of the calculation at $t = 0$ . Direct differentiation at $t = 0$ , using $c ^ { \prime \prime } ( { \textstyle { \frac { 1 } { 2 } } } ) = 7$ 2 and $\begin{array} { r } { d ^ { \prime } ( \frac { 1 } { 2 } ) = 1 } \end{array}$ , gives

$$
A _ { \star } : = D \psi ( a _ { \star } ( 0 ) ) = \left( { \begin{array} { l l } { 1 } & { 0 } \\ { - 1 } & { - 2 } \end{array} } \right) , \qquad S _ { \star } : = D _ { a } [ D \psi ( a ) ^ { \top } R ( a , b ) ] _ { ( a _ { \star } ( 0 ) , b _ { \star } ( 0 ) ) } = \left( { \begin{array} { l l } { - 1 } & { 3 } \\ { 3 } & { 2 } \end{array} } \right) .\tag{D.8}
$$

Write $z = ( u , a , b ) , z _ { \star } ( t ) = ( u _ { \star } ( t ) , a _ { \star } ( t ) , b _ { \star } ( t ) )$ , and let $F = ( - \nabla _ { u } g _ { \mathrm { c y c l e } } , \nabla _ { a , b } g _ { \mathrm { c y c l e } } )$ be the GDA vector field. Set $\mathcal { I } =$ diag $ { \mathbf { \chi } } _ { : ( J , J , J ) }$ and $\mathcal { Q } _ { t } = \mathrm { d i a g } ( e ^ { t J } , e ^ { t J } , e ^ { t J } )$ . Rotational equivariance gives $F ( \mathcal { Q } _ { t } z ) = \mathcal { Q } _ { t } F ( z ) , \thinspace \mathrm { s o } \ : \widetilde { z } ( t ) = \mathcal { Q } _ { t } ^ { \top } z ( t )$ satisfies $\dot { \tilde { z } } = F ( \widetilde { z } ) - \mathcal { T } \widetilde { z } .$ The periodic solution becomes the fixed point $z _ { \star } ( 0 )$ , whose Jacobian is

$$
M _ { \star } = D F ( z _ { \star } ( 0 ) ) - \mathcal { I } = \left( \begin{array} { c c c } { - J } & { - \sqrt { 2 } I _ { 2 } } & { 0 } \\ { \sqrt { 2 } I _ { 2 } } & { - 2 I _ { 2 } - S _ { \star } - J } & { - A _ { \star } ^ { \top } } \\ { 0 } & { - A _ { \star } } & { - I _ { 2 } - J } \end{array} \right) .\tag{D.9}
$$

A direct determinant calculation gives

$$
\operatorname * { d e t } ( \lambda I - M _ { \star } ) = \lambda ( \lambda ^ { 5 } + 7 \lambda ^ { 4 } + 7 \lambda ^ { 3 } + 2 1 \lambda ^ { 2 } + 1 0 \lambda + 4 ) .\tag{D.10}
$$

For the quintic factor, the first column of the Routh array is $1 , 7 , 4 , 9 / 2 , 3 7 0 / 6 3 , 4$ . It has no sign change, so the Routh–Hurwitz criterion excludes roots in the open right half-plane. The array has neither a zero first-column entry nor an all-zero row, excluding roots on the imaginary axis as well. Thus all five roots have strictly negative real parts. The remaining zero eigenvalue is simple and represents phase shifts $z _ { \star } ( t ) \mapsto z _ { \star } ( t + s )$ ; its eigenvector is $\dot { z } _ { \star } ( 0 ) \stackrel { = } { = } \mathcal { I } z _ { \star } ( 0 )$

To control departures from the orbit, choose a basis $S \in \mathbb { R } ^ { 6 \times 5 }$ of the stable invariant subspace of $M _ { \star }$ and let A be its restriction in this basis.

For sufficiently small $\varepsilon > 0 .$ , the map $\mathcal { X } ( \vartheta , \xi ) = \mathcal { Q } _ { \vartheta } ( z _ { \star } ( 0 ) + S \xi )$ , with $\vartheta \in \mathbb { R } / ( 2 \pi \mathbb { Z } )$ and $\xi \in \mathbb { R } ^ { 5 }$ $\| \xi \| < \varepsilon ,$ gives smooth coordinates near Γ: the columns of S complement its tangent direction, and the orbit is compact and embedded. Using $M _ { \star } S = S A _ { - }$ and Taylor expansion of $F \mathrm { a t } z _ { \star } ( 0 )$ differentiation of these coordinates along GDA gives

$$
( \dot { \vartheta } - 1 ) \mathcal { I } ( z _ { \star } ( 0 ) + S \xi ) + S ( \dot { \xi } - A _ { - } \xi ) = O ( \| \xi \| ^ { 2 } ) .
$$

The coefficient matrix $[ \mathcal { T } ( z _ { \star } ( 0 ) + S \xi ) , S ]$ remains uniformly invertible for small $\xi .$ Hence $\dot { \vartheta } =$ $1 + O ( \left\| \xi \right\| ^ { 2 } )$ and $\dot { \xi } = A _ { - } \xi + O ( \left\| \xi \right\| ^ { 2 } )$ , uniformly in the phase ϑ. Since A is Hurwitz, choose $P \succ 0$ satisfying $A _ { - } ^ { \top } P + \breve { P A } _ { - } = - \breve { I } ,$ , and define $\dot { V } ( \mathcal { X } ( \boldsymbol { \vartheta } , \boldsymbol { \dot { \xi } } ) ) = \boldsymbol { \xi } ^ { \intercal } P \boldsymbol { \xi }$ . On a smaller neighborhood,

$$
V ( z ) \asymp \mathrm { d i s t } ( z , \Gamma ) ^ { 2 } , \qquad \dot { V } = - \left\| \xi \right\| ^ { 2 } + O ( \left\| \xi \right\| ^ { 3 } ) \leq - c V
$$

for a universal $c > 0$ . This proves local exponential orbital stability.

An invariant neighborhood for discrete GDA. We use $\dot { V } \leq - c V$ to obtain $V ( z ^ { + } ) \leq ( 1 -$ $c h ) V ( z ) + C h ^ { 2 }$ for both update rules, then choose $h _ { \star }$ so that a fixed sublevel set of $\dot { V }$ is invariant. Choose a fixed small $\delta > { 0 }$ so that $\mathcal { U } : = \{ V \le \delta \}$ is compactly contained in this neighborhood and in $\{ \| u \| \geq 1 / 2 \}$ . Simultaneous GDA gives $z _ { \mathrm { S i m } } ^ { + \bar { } } = z + h F ( z )$ . In the alternating update, the change in u is $- h { \sqrt { 2 } } a$ , so its inner-gradient query changes only the a-gradient, $ { \mathrm { b y } } - 2 h a$ . Thus $z _ { \mathrm { A l t } } ^ { + } - z _ { \mathrm { S i m } } ^ { + } = ( 0 , - 2 h ^ { 2 } a , 0 )$ . For $z \in \mathcal { U }$ and sufficiently small $h ,$ , both updates and the line segments joining them to z remain in a fixed compact neighborhood where $V$ is smooth. Taylor expansion therefore gives, uniformly for either update,

$$
V ( z ^ { + } ) \leq V ( z ) + h \nabla V ( z ) ^ { \mathsf { T } } F ( z ) + C h ^ { 2 } \leq ( 1 - c h ) V ( z ) + C h ^ { 2 } \leq ( 1 - c h ) \delta + C h ^ { 2 } \leq \delta ,
$$

where we choose $h _ { \star } > 0$ small enough that $c h _ { \star } \leq 1$ and $C h _ { \star } \leq c \delta$ . Thus U is invariant for every $h \in [ 0 , h _ { \star } ]$ , including $h = 0$ . Induction and $\nabla \Phi _ { g _ { \mathrm { c v c l e } } } ( u ) = u$ give the discrete conclusion for every such step sequence. All constants depend only on the fixed game.

For the next lemma, fix a bounded open neighborhood $\mathcal { W } \subset \mathbb { R } ^ { 4 }$ of the inner projection of U. We will preserve the game for every u whenever $( a , b ) \in \mathcal { W }$ . This also protects the alternating inner-gradient query after the primal update, since that update changes only u. □

## D.2 CONSTRUCTION OF $\tilde { g } _ { \mathrm { c y c l e } }$

To construct $\tilde { g } _ { \mathrm { c y c l e } }$ , we extend the base game to accommodate finitely many initial steps $h _ { t } > h _ { \star }$ . At the initial query states $y _ { t }$ to be chosen, for $0 \leq t <$ m and $e _ { 1 } = ( 1 , 0 ) ^ { \mathsf { T } }$ , the required gradients are

$$
\nabla _ { u } \tilde { g } _ { \mathrm { c y c l e } } ( e _ { 1 } , y _ { t } ) = 0 , \qquad \nabla _ { y } \tilde { g } _ { \mathrm { c y c l e } } ( e _ { 1 } , y _ { t } ) = \frac { y _ { t + 1 } - y _ { t } } { h _ { t } } .\tag{D.11}
$$

They keep the outer variable fixed and send each inner state to the next, ending on the cycle. A local change of inner coordinates prescribes these gradients through its derivative. We preserve a to retain both the primal gradient ${ \sqrt { 2 } } a$ and the bilinear term ${ \sqrt { 2 } } u ^ { \mathsf { T } } a$ . A nonlinear change of these coordinates would introduce inner-Hessian terms proportional to $u ,$ obstructing global joint smoothness. The following lemma realizes the prescribed derivative inside a ball; disjoint balls will keep its bounds independent of the number of steps.

Lemma D.1 (Local derivative modification that leaves the first two coordinates unchanged). Fix $M < \infty$ and $\delta > 0$ . There is $C = C ( M , \delta )$ such that,for every $z \in \mathbb { R } ^ { 5 }$ and every

$$
N = \binom { I _ { 2 } } { P } \quad 0 \biggr ) , \qquad \operatorname* { d e t } N > 0 , \qquad \| N \| + \| N ^ { - 1 } \| \le M ,\tag{D.12}
$$

there is a smooth bijection $G : \mathbb { R } ^ { 5 }  \mathbb { R } ^ { 5 }$ with a smooth inverse. For every w, thefirst two coordinates of $G ( w )$ equal those ofw. Moreover, $G ( z ) = z , D G ( z ) = N$ , G equals the identity outside $B ( z , \delta )$ G maps this ball onto itself, and

$$
\left\| D G \right\| _ { \infty } + \left\| D G ^ { - 1 } \right\| _ { \infty } + \left\| D ^ { 2 } G \right\| _ { \infty } \leq C .\tag{D.13}
$$

Proof. We join the identity matrix to N through uniformly invertible matrices, then realize this matrix path as the derivative of a compactly supported flow. Since det $N > 0$ and the first diagonal block is $I _ { 2 } ,$ the matrix $Q$ is invertible and has positive determinant. Write $Q = U S$ , where $U ^ { \mathsf { T } } U = I .$ det $U = 1$ , and $S$ is symmetric positive definite. Choose a skew-symmetric matrix A with $e ^ { A } = U$ and $\| A \| \leq \pi$ , and define

$$
Q _ { \tau } = e ^ { \tau A } ( ( 1 - \tau ) I + \tau S ) , \qquad N _ { \tau } = \left( { I _ { 2 } } \frac { \quad 0 } { \tau P } \right) .
$$

The matrices $N _ { \tau }$ , their inverses, and their derivatives are uniformly bounded in terms of $M .$ . Put $A _ { \tau } = \dot { N } _ { \tau } N _ { \tau } ^ { - 1 }$ . Choose a smooth radial cutoff $\chi$ that equals one on $B ( z , \delta / 3 )$ and vanishes outside $B ( z , 2 \delta / 3 )$ , and set

$$
X _ { \tau } ( w ) = \chi ( w ) A _ { \tau } ( w - z ) .
$$

This time-dependent differential equation has zero first two components and uniformly bounded first two derivatives. Let $G _ { \tau } ( w )$ denote its solution at time τ starting from w at time zero. Then $G _ { \tau }$ fixes z, preserves the first two coordinates, and satisfies

$$
\frac { d } { d \tau } D G _ { \tau } ( z ) = A _ { \tau } D G _ { \tau } ( z ) , \qquad D G _ { 0 } ( z ) = I ,
$$

so $D G _ { \tau } ( z ) = N _ { \tau }$ . Differentiating this differential equation once and twice with respect to the initial point, and then applying Gronwall’s inequality, gives (D.13) for $G = G _ { 1 }$ and its inverse. Since the vector field vanishes in a neighborhood of the boundary, the ball is mapped onto itself. □

Lemma 6.2 (GDA can reach the orbit in a modified game $\tilde { g } _ { \mathrm { c y c l e } } ) .$ . There are universal $C , \mu > 0$ with the following property. For any finite sequence of steps $h _ { 0 } , \ldots , h _ { m - 1 } > h _ { \star }$ , including $m = 0 ;$ , we can modify the game $g _ { \mathrm { c y c l e } }$ in Lemma 6.1 to obtain $\bar { \tilde { g } } _ { \mathrm { c y c l e } } : \mathbb { R } ^ { 2 } \times \mathbb { R } ^ { 5 }  \mathbb { R } .$ . This game is jointly C-smooth and affine in u, satisfies the inner $\mu { - } P \mathbf { \mathcal { L } }$ inequality, and has outer objective $\Phi _ { \tilde { g } _ { \mathrm { c y c l e } } } ( u ) = \left\| u \right\| ^ { 2 } / 2$ With $y = ( a , b , r )$ and $e _ { 1 } = ( 1 , 0 ) ^ { \mathsf { T } }$ , there is an initialization $( e _ { 1 } , y _ { 0 } )$ with inner gap at most $C .$ . From this initialization, both $G D A _ { S i m }$ and $G D A _ { A l t }$ , using exact gradients and equal primal and dual steps, keep $u _ { t } = e _ { 1 } f o r 0 \leq t \leq$ m and reach $y _ { m } = ( y _ { \star } ( 0 ) , 0 )$ . For every continuation with $h _ { t } \in [ 0 , h _ { \star } ]$ each algorithmfollows its $g _ { \mathrm { c y c l e } }$ dynamics in $( u , a , b )$ , while r remains zero.

Proof. We choose separated inner states whose required update vectors are uniformly bounded. At each state, a local coordinate modification realizes the desired gradient while preserving the first two inner coordinates. The residual representation then verifies inner PŁ, and the disjoint modification regions give uniform smoothness.

The case $m = 0$ uses $\tilde { g } _ { \mathrm { c y c l e } } ( u , v , r ) = g _ { \mathrm { c y c l e } } ( u , v )$ and $y _ { 0 } = ( y _ { \star } ( 0 ) , 0 )$ . All initialization quantities are bounded on the fixed periodic solution, and the initial primal gradient is $\sqrt { 2 } a _ { \star } ( 0 ) = - J e _ { 1 }$ , of norm one. Suppose $m \geq 1$ and write $y = ( a , b , r ) \in \mathbb { R } ^ { 2 } \times \dot { \mathbb { R } } ^ { 2 } \times \mathbb { R }$ . The extra coordinate r separates the query states even when their first four coordinates agree. Its value is not itself penalized, so this separation need not increase the initial gap. Define

$$
\begin{array} { r l } & { r _ { t } = - \displaystyle \sum _ { j = t } ^ { m - 1 } h _ { j } \quad ( 0 \leq t < m ) , \quad r _ { m } = 0 , } \\ & { y _ { t } = ( 0 , - B e _ { 1 } , r _ { t } ) \quad ( t < m ) , \quad y _ { m } = ( a _ { \star } ( 0 ) , b _ { \star } ( 0 ) , 0 ) . } \end{array}\tag{D.14}
$$

Here B is a fixed universal constant large enough that a neighborhood of $( a , b ) = ( 0 , - B e _ { 1 } )$ is outside the support of the nonlinear part of R and separated from W. We protect the lift $\mathcal { W } \times \mathbb { R }$ in the inner space $( a , b , r )$ . Let $\bar { H } ( a , b , \bar { r } ) = ( ( a , R ( a , \bar { b ) } ) , r )$ . At all $y _ { t }$ with $t < m$ , it is the identity in a fixed neighborhood.

Put $v _ { t } = ( y _ { t + 1 } - y _ { t } ) / h _ { t } = ( v _ { t } ^ { a } , v _ { t } ^ { b } , 1 )$ for $0 \leq t < m$ . Its first four coordinates are zero except at $t = m - 1 ;$ ; their magnitude at the last step is bounded by a universal constant since $h _ { m - 1 } > h _ { \ast }$ and the first four endpoint coordinates are fixed. Let $\pi _ { b } : \mathbb { R } ^ { 5 } \to \mathbb { R } ^ { 2 }$ project onto the two coordinates corresponding to b. We seek a smooth bijection $\widetilde { H } : \mathbb { R } ^ { 5 } \to \mathbb { R } ^ { 5 }$ that preserves $^ { a , }$ fixes the initial query states $y _ { 0 } , \ldots , y _ { m - 1 }$ , and agrees with H<sup>¯</sup> on $\mathcal { W } \times \mathbb { R }$ . Define the game in terms of this map by

$$
\widetilde { g } _ { \mathrm { c y c l e } } ( \boldsymbol { u } , \boldsymbol { y } ) = \left. ( \boldsymbol { u } , 0 ) , \widetilde { T } ( \boldsymbol { y } ) \right. - \frac { 1 } { 2 } \left. \widetilde { T } ( \boldsymbol { y } ) \right. ^ { 2 } , \qquad \widetilde { T } ( \boldsymbol { a } , \boldsymbol { b } , \boldsymbol { r } ) = ( \sqrt { 2 } \boldsymbol { a } , \pi _ { \boldsymbol { b } } \widetilde { H } ( \boldsymbol { a } , \boldsymbol { b } , \boldsymbol { r } ) ) \in \mathbb { R } ^ { 4 } .\tag{D.15}
$$

At each initial query state $y _ { t } , t < m$ , we have $\widetilde { T } ( y _ { t } ) = ( 0 , - B e _ { 1 } ) , \mathrm { s o } \ \nabla _ { u } \widetilde { g } _ { \mathrm { c y c l e } } ( e _ { 1 } , y _ { t } ) = 0$ . The required inner gradient is therefore equivalent to

$$
D \widetilde { H } ( y _ { t } ) ^ { \mathsf { T } } ( \sqrt { 2 } e _ { 1 } , B e _ { 1 } , 0 ) = v _ { t } .\tag{D.16}
$$

This identity determines the row corresponding to $b _ { 1 }$ . We choose the remaining two rows to make the derivative uniformly invertible with positive determinant:

$$
D \widetilde { H } ( y _ { t } ) = \left( \begin{array} { c c c } { I _ { 2 } } & { 0 } & { 0 } \\ { ( v _ { t } ^ { a } - \sqrt { 2 } e _ { 1 } ) ^ { \mathsf { T } } / B } & { ( v _ { t } ^ { b } ) ^ { \mathsf { T } } / B } & { 1 / B } \\ { 0 } & { e _ { 2 } ^ { \mathsf { T } } } & { 0 } \\ { 0 } & { - B e _ { 1 } ^ { \mathsf { T } } } & { 0 } \end{array} \right) .\tag{D.17}
$$

The lower-right $3 \times 3$ block has determinant one, so the full derivative also has determinant one. It and its inverse are uniformly bounded, since $B$ is fixed and $v _ { t }$ is uniformly bounded. Distinct $y _ { t }$ with $t < m$ are separated by at least $h _ { \star }$ , because their scalar coordinates differ by sums of steps exceeding $h _ { \star } .$ . Choose a universal ball radius smaller than one quarter of this separation and small enough that $\bar { H }$ is the identity on every ball and all balls avoid $\mathcal { W } \times \mathbb { R }$

Apply Lemma D.1 in these disjoint balls. Each local map fixes its center, has the prescribed derivative, preserves the first two coordinates, and maps its ball onto itself. Let G be their composition and set ${ \tilde { H } } = G \circ { \bar { H } }$ . Since H<sup>¯</sup> is the identity near every center, (D.17) holds. At most one factor of $G$ is nonidentity along the image of any point, so the derivative, inverse-derivative, and second-derivative bounds do not grow with $m .$ Since $\bar { H }$ is bijective and equals the identity on each modification ball, no point outside those balls is mapped into them. Thus $\tilde { H } = \bar { H }$ on $\mathcal { W } \times \mathbb { R }$

Since $\widetilde { H }$ is onto and preserves its first two coordinates, $\widetilde { T }$ is onto. For $q = ( q _ { a } , q _ { b } ) \in \mathbb { R } ^ { 4 }$

$$
\begin{array} { r } { D \widetilde { T } ( \boldsymbol { y } ) ^ { \mathsf { T } } q = D \widetilde { H } ( \boldsymbol { y } ) ^ { \mathsf { T } } ( \sqrt { 2 } q _ { a } , q _ { b } , 0 ) , \qquad \Big \| D \widetilde { T } ( \boldsymbol { y } ) ^ { \mathsf { T } } q \Big \| \ge c \| q \| , } \end{array}
$$

where the last inequality uses the uniform inverse-derivative bound for ${ \cal \widetilde { H } } .$ Lemma B.1 now gives ma $\mathfrak { x } _ { y } \tilde { g } _ { \mathrm { c y c l e } } ( u , y ) = \left. u \right. ^ { 2 } / 2$ and global inner PŁ with a universal constant.

To verify joint smoothness, write $K : = \pi _ { b } \widetilde { H }$ for the last two residual coordinates. The derivative bounds above control $D K$ and $D ^ { 2 } K$ uniformly. Wherever $D ^ { 2 } K$ is nonzero, K is bounded as well: outside the modification balls this follows from the fixed compact support in $( a , b )$ of the original residual’s nonlinear part, and inside a ball it follows because the ball maps onto itself. In particular, its b-coordinates stay near $- B e _ { 1 }$ , regardless of the scalar center $r _ { t }$ . Thus

$$
\left\| \nabla ^ { 2 } \left( { \frac { 1 } { 2 } } \left\| K \right\| ^ { 2 } \right) \right\| \leq \left\| D K \right\| ^ { 2 } + \left\| K \right\| \left\| D ^ { 2 } K \right\| \leq C .
$$

Expanding (D.15) gives $\widetilde { g } _ { \mathrm { c y c l e } } ( u , a , b , r ) = \sqrt { 2 } u ^ { \mathsf { T } } a - \left\| a \right\| ^ { 2 } - \left\| K ( a , b , r ) \right\| ^ { 2 } / 2 .$ . Since a is unchanged, the outer-dependent term remains bilinear, so the full Hessian is uniformly bounded. On $\mathcal { W } \times \mathbb { R } .$ , we have $K ( a , \dot { b , } r ) = R ( a , b )$ ; hence $\tilde { g } _ { \mathrm { c y c l e } }$ agrees with $g _ { \mathrm { c y c l e } }$ for every u and is independent of r there.

Finally, (D.16) gives the target gradients in (D.11). At each initial query, the zero primal gradient leaves $u = e _ { 1 }$ unchanged, so the simultaneous and alternating methods use the same inner-gradient query and both move from $y _ { t }$ to $y _ { t + 1 }$ . The initial inner gap is $( 1 + B ^ { 2 } ) / 2$ and the initial primal gradient is zero.

At time $m ,$ both methods reach $( e _ { 1 } , y _ { \star } ( 0 ) , 0 )$ . Agreement with the base game on $\mathcal { W } \times \mathbb { R }$ covers both update orders, including the alternating inner-gradient query after the primal update. Independence of r keeps $\boldsymbol r _ { t } = 0 ,$ , while Lemma 6.1 keeps $( u _ { t } , a _ { t } , b _ { t } ) \in \mathcal { U }$ under every small-step continuation. For either $m = 0 \mathrm { o r } m \geq 1$ , the construction gives $\| \nabla _ { u } \tilde { g } _ { \mathrm { c y c l e } } ( e _ { 1 } , y _ { 0 } ) \| \le \mathrm { \dot { 1 } }$ . All constants depend only on the fixed small-step construction, not on the number or values of the prescribed steps. □

## E COMPARISON WITH SMOOTHED-AGDA

We compare the expected squared outer gradient at a uniformly selected SGDA iterate with that at a point computed from a Smoothed-AGDA iterate pair. Algorithm 1 states the updates of Yang et al. (2022, Algorithm 2), with an auxiliary center $z _ { t }$ initialized at $x _ { 0 }$ and regularization coefficient 2ℓ.

Algorithm 1 Stochastic Smoothed-AGDA   
Require: Initial pair $( x _ { 0 } , y _ { 0 } )$ , integer $N \geq 1$ , and step $0 < \alpha \leq 1 / ( 3 \ell )$   
1: Set z<sub>0</sub> = x<sub>0</sub> and $\beta = { \alpha \ell } / ( { 7 6 8 0 0 \kappa } )$   
2: for $t = 0 , \ldots , N - 1$ do   
3: Draw fresh independent samples $\xi _ { t } ^ { x } , \xi _ { t } ^ { y } .$   
4: $x _ { t + 1 } = x _ { t } - \alpha \big [ \dot { G } _ { x } ( x _ { t } , y _ { t } ; \xi _ { t } ^ { \hat { x } } ) + \dot { 2 } \ell ( \dot { x _ { t } } - z _ { t } ) \big ] .$   
5: $y _ { t + 1 } = y _ { t } + ( \hat { \alpha / } 4 8 ) G _ { y } ( x _ { t + 1 } , y _ { t } ; \xi _ { t } ^ { y } )$   
6: $z _ { t + 1 } = z _ { t } + \beta ( x _ { t + 1 } - z _ { t } ) .$   
7: end for   
8: Draw I uniformly from $\{ 0 , \ldots , N - 1 \}$ , independently of the oracle samples.   
9: return $( \widetilde { x } , \widetilde { y } ) = ( x _ { I } , y _ { I } )$

Smoothed-AGDA controls the squared game gradients, but directly bounding the squared outer gradient introduces an extra factor of $\kappa$ through the inner-gradient error. The exact minimizer of the regularized outer objective avoids this loss, and stochastic updates approximate it. We apply the dual-first updates (E.7) to $f ( x , y ) + \ell \left\| x - \widetilde { x } \right\| ^ { 2 }$ , starting from $( \widetilde x , \widetilde y )$ . The guarantee below combines the mean-square estimates of Yang et al. (2022, Theorem 4.1, Proposition 2.1(b), and Corollary 4.1) with strong convexity in x.

Proposition E.1 (Outer stationarity for Smoothed-AGDA). Suppose Assumptions 3.1–3.5 hold with $\kappa \geq 2 ,$ and assume in addition tha

$$
\Phi ( x _ { 0 } ) - \operatorname* { i n f } _ { x } \{ f ( x , y _ { 0 } ) + \ell \left. x - x _ { 0 } \right. ^ { 2 } \} \leq c _ { \mathrm { i n i t } } \Delta .\tag{E.1}
$$

For every $0 < \varepsilon ^ { 2 } \le \ell \Delta$ , Smoothed-AGDA (Algorithm 1)followed by the dual-first conversion in (E.7) returns a point xb with <sup>E</sup> $\left\| \nabla \Phi ( \widehat { x } ) \right\| ^ { 2 } \leq \varepsilon ^ { 2 }$ , using at most

$$
C \left[ \frac { \kappa \ell \Delta } { \varepsilon ^ { 2 } } + \frac { \kappa ^ { 2 } \ell \Delta \sigma ^ { 2 } } { \varepsilon ^ { 4 } } + \left( \kappa + \frac { \kappa ^ { 4 } \sigma ^ { 2 } } { \varepsilon ^ { 2 } } \right) \log ( 2 \kappa ) \right]\tag{E.2}
$$

stochastic partial-gradient calls, where $C$ depends at most on $c _ { \mathrm { i n i t } } .$ . In particular, $i f \ \varepsilon ^ { 2 } \ \leq$ $c \ell \Delta / ( \kappa ^ { 2 } \log ( 2 \kappa ) )$ for a sufficiently small universal $c > 0 ,$ , this cost is

$$
O \left( \frac { \kappa \ell \Delta } { \varepsilon ^ { 2 } } + \frac { \kappa ^ { 2 } \ell \Delta \sigma ^ { 2 } } { \varepsilon ^ { 4 } } \right) .\tag{E.3}
$$

Proof. All step sizes and iteration counts below depend only on the accuracy and problem parameters, not on the realized pair.

The iterate pair from Smoothed-AGDA. Initialize the auxiliary center of Yang et al. (2022, Algorithm 2) at $z _ { 0 } = x _ { 0 }$ . Their initial quantity $b _ { 0 }$ is twice the left side of (E.1), so $b _ { 0 } \le 2 c _ { \mathrm { i n i t } } \Delta$ Before tuning the primal step $\alpha .$ , the proof of their Theorem 4.1, Eqs. (40)–(41), bounds the average squared game gradients by

$$
\frac { 1 } { N } \sum _ { t = 0 } ^ { N - 1 } \mathbb { E } \Big [ \| \nabla _ { x } f ( x _ { t } , y _ { t } ) \| ^ { 2 } + \kappa \| \nabla _ { y } f ( x _ { t } , y _ { t } ) \| ^ { 2 } \Big ] \leq C \frac { \kappa \Delta } { N \alpha } + C \kappa \ell \alpha \sigma ^ { 2 } , \qquad 0 < \alpha \leq \frac { 1 } { 3 \ell } .\tag{E.4}
$$

Here the dual step is $\alpha / 4 8 .$ , the regularization coefficient is $2 \ell ,$ and the center relaxation is $\alpha \ell / ( 7 6 8 0 0 \kappa )$ , as in their proof. Their descent estimates use only the conditional means and variances of the two gradient calls; no smoothness of individual sampled gradients is required. They therefore apply to the oracle formulation in Assumptions 3.4 and 3.5.

Choose $\alpha = \operatorname* { m i n } \{ 1 / ( 3 \ell ) , \sqrt { { \Delta } / ( { N \ell \sigma ^ { 2 } } ) } / { 2 } \}$ , omitting the second term when $\sigma = 0$ . With $N \geq$ $C ( \kappa \ell \Delta / \varepsilon ^ { 2 } + \kappa ^ { 2 } \ell \Delta \dot { \sigma } ^ { 2 } / \varepsilon ^ { 4 } )$ , a uniformly selected pair (x, e ye) satisfies $\mathbb { E } e \ \le \ \varepsilon ^ { 2 } / 1 6$ , where $e : =$ $\| \nabla _ { x } f ( \widetilde { x } , \widetilde { y } ) \| ^ { 2 } + \kappa \left\| \nabla _ { y } f ( \widetilde { x } , \widetilde { y } ) \right\| ^ { 2 }$ . Using the upper budget $\Delta$ in this choice, rather than the actual initial outer gap, also covers initializations with zero outer gap.

The exact proximal point. The gradient-bias bound in Lemma B.2 would only give $\left\| \nabla \Phi ( \widetilde { \boldsymbol { x } } ) \right\| ^ { 2 } \leq$ 2κe. Instead, condition on the selected pair and define the auxiliary game ${ \widehat { f } } ( x , y ) : = f ( x , y ) +$ $\ell \| x - \widetilde x \| ^ { 2 }$ . It is ℓ-strongly convex in $x ,$ retains inner PŁ with parameter $\ell / \kappa .$ , and is 3ℓ-smooth in the partial-gradient convention of Assumption 3.1. Since $f ( \cdot , y ) + \ell \left\| \cdot \right\| ^ { 2 } / 2$ is convex for every $y ,$ its supremum has the same property. Thus $\Phi ( x ) + \ell \left\| x - { \widetilde { x } } \right\| ^ { 2 }$ is ℓ-strongly convex and has a unique minimizer $p : = \mathrm { a r g }$ min $_ { x } \{ \Phi ( x ) + \ell \| x - \widetilde { x } \| ^ { 2 } \}$ . Inner PŁ and strong convexity in x give

$$
\begin{array} { c } { \displaystyle \Phi ( \widetilde x ) - \displaystyle \operatorname* { m i n } _ { x } \widehat f ( x , \widetilde y ) = \Phi ( \widetilde x ) - f ( \widetilde x , \widetilde y ) + \widehat f ( \widetilde x , \widetilde y ) - \displaystyle \operatorname* { m i n } _ { x } \widehat f ( x , \widetilde y ) \leq \frac { e } { 2 \ell } , } \\ { \displaystyle \qquad \frac \ell 2 \left\| p - \widetilde x \right\| ^ { 2 } \leq \Phi ( \widetilde x ) - \Phi ( p ) - \ell \left\| p - \widetilde x \right\| ^ { 2 } \leq \frac { e } { 2 \ell } . } \end{array}\tag{E.5}
$$

The second line uses strong convexity of the auxiliary outer objective, then $\Phi ( p ) + \ell \left\| p - { \widetilde { x } } \right\| ^ { 2 } \geq$ mi $_ { 1 _ { x } } \widehat { f } ( x , \widetilde { y } )$ . Optimality of $p$ now yields the pointwise bound

$$
\nabla \Phi ( p ) = 2 \ell ( \widetilde { x } - p ) , \qquad \| \nabla \Phi ( p ) \| ^ { 2 } \leq 4 e .\tag{E.6}
$$

This is the squared-gradient estimate underlying Yang et al. (2022, Proposition 2.1(b), Eq. (7)); no conversion from a first-moment norm bound is used.

A finite stochastic conversion. Starting at $( x ^ { 0 } , y ^ { 0 } ) = ( \widetilde { x } , \widetilde { y } )$ , apply the dual-first alternating method used in that proposition to ${ \widehat { f } } ,$ with dual step τ and primal step 162τ . Explicitly, use fresh independent samples to update

$$
\begin{array} { r l r l } & { y ^ { k + 1 } = y ^ { k } + \tau G _ { y } ( x ^ { k } , y ^ { k } ; \xi _ { k } ^ { y } ) , } & \\ & { x ^ { k + 1 } = x ^ { k } - 1 6 2 \tau \big [ G _ { x } ( x ^ { k } , y ^ { k + 1 } ; \xi _ { k } ^ { x } ) + 2 \ell ( x ^ { k } - \widetilde { x } ) \big ] , } & & { 0 < \tau \leq \frac { 1 } { 4 8 6 \ell } . } \end{array}\tag{E.7}
$$

These are an additional postprocessing procedure, not the primal-first recursion defining $\mathrm { S G D A _ { A l t } }$ in Section 3.

We use the potential contraction of Yang et al. (2022, Theorem A.1 and the proof of Proposition 2.1(b)), then obtain the required distance bound directly from strong convexity. Write $\Psi ( y ) : = \mathrm { m i n } _ { x } \widehat { f } ( x , y )$ and $\Psi ^ { * } : = \operatorname* { m a x } _ { y } \Psi ( y )$ . The auxiliary game has a saddle with outer coordinate $p \mathrm { : }$ any inner maximizer at $p$ also minimizes $\widehat { f } ( \cdot , y )$ there, by Lemma B.2 and strong convexity. The potential combines the dual gap with the error in minimizing ${ \widehat { f } } ( \cdot , y ^ { k } )$

$$
P _ { k } : = \mathbb { E } _ { \mathrm { c o n v } } \bigg [ \Psi ^ { * } - \Psi ( y ^ { k } ) + \frac { 1 } { 1 0 } \big ( \widehat f ( x ^ { k } , y ^ { k } ) - \Psi ( y ^ { k } ) \big ) \bigg ] ,
$$

where $\mathbb { E } _ { \mathrm { c o n v } }$ averages the fresh conversion samples conditional on $( \widetilde x , \widetilde y )$ . The initialization estimate and their contraction give

$$
P _ { 0 } \leq \frac { 1 1 } { 1 0 } \left[ \Phi ( \widetilde x ) - \operatorname* { m i n } _ { x } \widehat f ( x , \widetilde y ) \right] \leq \frac { 1 1 e } { 2 0 \ell } , \qquad P _ { k } \leq \left( 1 - \frac { \ell \tau } { 2 \kappa } \right) ^ { k } P _ { 0 } + C \kappa \tau \sigma ^ { 2 } .\tag{E.8}
$$

To bound distance to $p ,$ let $x _ { y } : = \arg \operatorname* { m i n } _ { x } \widehat { f } ( x , y )$ . Strong convexity at this minimizer and the saddle inequality $\widehat { f } ( p , y ) \leq \Psi ^ { * }$ give

$$
\left\| x - p \right\| ^ { 2 } \leq 2 \left\| x - x _ { y } \right\| ^ { 2 } + 2 \left\| x _ { y } - p \right\| ^ { 2 } \leq { \frac { 4 } { \ell } } { \Big [ } { \widehat { f } } ( x , y ) - \Psi ( y ) + \Psi ^ { * } - \Psi ( y ) { \Big ] } .\tag{E.9}
$$

Hence $\mathbb { E } _ { \mathrm { c o n v } } \left\| x ^ { k } - p \right\| ^ { 2 } \leq 4 0 P _ { k } / \ell .$ . The step choices satisfy their conditions with smoothness 3ℓ and PŁ parameters ℓ and $\ell / \kappa$ . In particular, their variance term is $O ( \kappa \tau \sigma ^ { 2 } )$ . Substitution and expectation over the selected pair give

$$
\mathbb { E } \left\| x ^ { k } - p \right\| ^ { 2 } \leq \frac { C } { \ell ^ { 2 } } \left( 1 - \frac { \ell \tau } { 2 \kappa } \right) ^ { k } \mathbb { E } e + \frac { C \kappa \tau \sigma ^ { 2 } } { \ell } .\tag{E.10}
$$

By Lemma B.2, the outer gradient is $2 \kappa \ell { - } \mathbf { I }$ ipschitz. To keep the conversion’s contribution to the squared outer gradient below $\varepsilon ^ { 2 } / 2$ , it therefore suffices to make the squared distance at most $\varepsilon ^ { 2 } / ( 1 \dot { 6 } \kappa ^ { 2 } \ell ^ { 2 } )$ . Combining (E.6) and (E.10) gives

$$
\begin{array} { r l r } {  { \mathbb { E } \| \nabla \Phi ( { \boldsymbol x } ^ { k } ) \| ^ { 2 } \leq 8 \mathbb { E } e + 8 \kappa ^ { 2 } \ell ^ { 2 } \mathbb { E } \| { \boldsymbol x } ^ { k } - { \boldsymbol p } \| ^ { 2 } } } \\ & { } & { \leq 8 \mathbb { E } e + C \kappa ^ { 2 } ( 1 - \displaystyle \frac { \ell \tau } { 2 \kappa } ) ^ { k } \mathbb { E } e + C \kappa ^ { 3 } \ell \tau \sigma ^ { 2 } . } \end{array}\tag{E.11}
$$

The last term determines the conversion step: for $\sigma > 0$ , choose $\tau = \operatorname* { m i n } \{ 1 / ( 4 8 6 \ell ) , c \varepsilon ^ { 2 } / ( \kappa ^ { 3 } \ell \sigma ^ { 2 } ) \}$ with a sufficiently small universal $c > 0 { ; }$ ; use $\tau = 1 / ( 4 8 6 \ell )$ when $\sigma = 0$

After

$$
k = \left\lceil \frac { C \kappa } { \ell \tau } \log ( 2 \kappa ) \right\rceil = O \left( \left[ \kappa + \frac { \kappa ^ { 4 } \sigma ^ { 2 } } { \varepsilon ^ { 2 } } \right] \log ( 2 \kappa ) \right)
$$

updates, (E.10) meets this distance target. The factor $\kappa ^ { 2 }$ multiplying the contracting term in (E.11) accounts for the logarithm. This choice works in expectation over the random starting pair, without requiring its realized game gradients to satisfy a deterministic accuracy bound. With ${ \widehat { x } } = x ^ { k }$ , the first line of (E.11) concludes

$$
\mathbb { E } \left\| \nabla \Phi ( \widehat { x } ) \right\| ^ { 2 } \leq 8 \mathbb { E } e + \frac { \varepsilon ^ { 2 } } { 2 } \leq \varepsilon ^ { 2 } .\tag{E.12}
$$

There are two partial-gradient calls per update in both stages. Adding their costs proves (E.2). Under the stated small-error condition, the term $\kappa \log ( 2 \kappa )$ is absorbed by $\kappa \ell \Delta / \varepsilon ^ { 2 }$ , and $\stackrel { \bullet } { \kappa } { } ^ { 4 } \sigma ^ { 2 } \varepsilon ^ { - 2 } \log ( 2 \kappa )$ is absorbed by $\kappa ^ { 2 } \ell \Delta \sigma ^ { 2 } \varepsilon ^ { - 4 }$ , giving (E.3). □

## E.1 INITIALIZATION BUDGET OF THE HARD INSTANCES

The additional budget (E.1) consists of the initial inner gap and the decrease obtainable by minimizing $f ( x , y _ { 0 } ) + \ell \left\| x - x _ { 0 } \right\| ^ { 2 }$ over x while keeping $y _ { 0 }$ fixed. We verify that their sum is at most $c _ { \mathrm { i n i t } } \Delta$ for every hard instance in Theorem 4.1, using the same declared budget $\Delta$ and its original initialization. The ordinary outer and inner gap checks remain in the construction proofs.

The unstable quadratic. For $f _ { \mathrm { u n s t a b l e } }$ in (5.2), Appendix C.1 chooses $x _ { 0 } = 4 \sqrt { \Delta / ( \ell + \mu ) }$ and $y _ { 0 } = 0$ . Thus $f _ { \mathrm { u n s t a b l e } } ( x , y _ { 0 } ) = 0$ for every $x ,$ and

$$
\Phi _ { \mathrm { u n s t a b l e } } ( x _ { 0 } ) - \operatorname* { i n f } _ { x } \{ f _ { \mathrm { u n s t a b l e } } ( x , y _ { 0 } ) + \ell ( x - x _ { 0 } ) ^ { 2 } \} = \Phi _ { \mathrm { u n s t a b l e } } ( x _ { 0 } ) = \Delta \le c _ { \mathrm { i n i t } } \Delta .
$$

This verification includes the one-iterate case $T = 1$ , for which the high-ratio proof uses this quadratic.

The corridor and noise components. The game $f _ { \mathrm { s t a b l e } }$ in (5.4) is initialized at $( u _ { 0 } , v _ { 0 } ; y _ { 0 } ) =$ $( 0 , 0 ; 0 )$ . Equation (C.25) and the zero-component convention give $g _ { \sigma , j } ( v [ j ] ; 0 ) = 0$ for every component, so $f _ { \mathrm { s t a b l e } } ( u , v ; 0 ) = g _ { \mathrm { c o r r i d o r } } ( u )$ . Also $\Phi _ { \mathrm { s t a b l e } } ( 0 , 0 ) = g _ { \mathrm { c o r r i d o r } } ( 0 )$ , and (C.8) gives g<sub>corridor</sub>(0) − inf<sub>u</sub> g<sub>corridor</sub>(u) = ∆. Therefore,

$$
\begin{array} { r l r } {  { \Phi _ { \mathrm { s t a b l e } } ( 0 , 0 ) - \operatorname* { i n f } _ { u , v } \{ f _ { \mathrm { s t a b l e } } ( u , v ; 0 ) + \ell ( u ^ { 2 } +  v  ^ { 2 } ) \} } } \\ & { } & { = g _ { \mathrm { c o r r i d o r } } ( 0 ) - \operatorname* { i n f } _ { u } \{ g _ { \mathrm { c o r r i d o r } } ( u ) + \ell u ^ { 2 } \} \le \Delta \le c _ { \mathrm { i n i t } } \Delta . } \end{array}
$$

This also verifies the corridor alone, including $\sigma = 0$ and schedules with no positive noise-component steps, when all noise components vanish. Each noise component alone has regularized initial gap zero.

The low-ratio game. Retain the constants $C , C _ { \mathrm { l o w } } , \zeta$ and scales $A , s _ { x } , s _ { y }$ from Appendix $\mathrm { A . } 3 ,$ with $f _ { \mathrm { c y c l e } }$ defined by (A.3) and initialized at $x _ { 0 } = s _ { x } e _ { 1 } , y _ { 0 } ^ { \mathrm { c y c l e } } = s _ { y } y _ { 0 }$ . The construction in Appendix D.2 gives $\lVert \nabla _ { u } \tilde { g } _ { \mathrm { c y c l e } } ( e _ { 1 } , y _ { 0 } ) \rVert \leq 1$ , so $\left\| \nabla _ { x } f _ { \mathrm { c y c l e } } ( x _ { 0 } , y _ { 0 } ^ { \mathrm { c y c l e } } ) \right\| \leq A / s _ { x }$ . Affinity in $x ,$ the gap bound (A.7), and the scales in (A.2) give

$$
\begin{array} { r l } & { \Phi _ { \mathrm { c y c l e } } ( x _ { 0 } ) - \underset { x } { \operatorname* { i n f } } \left\{ f _ { \mathrm { c y c l e } } ( x , y _ { 0 } ^ { \mathrm { c y c l e } } ) + \ell \left. x - x _ { 0 } \right. ^ { 2 } \right\} } \\ & { \qquad = \Phi _ { \mathrm { c y c l e } } ( x _ { 0 } ) - f _ { \mathrm { c y c l e } } ( x _ { 0 } , y _ { 0 } ^ { \mathrm { c y c l e } } ) + \frac { \left. \nabla _ { x } f _ { \mathrm { c y c l e } } ( x _ { 0 } , y _ { 0 } ^ { \mathrm { c y c l e } } ) \right. ^ { 2 } } { 4 \ell } } \\ & { \qquad \le \left( 2 C + \frac { 1 } { 2 C _ { \mathrm { l o w } } } \right) \zeta \Delta \le 3 C \zeta \Delta \le c _ { \mathrm { i n i t } } \Delta . } \end{array}
$$

The last inequalities use $C \geq 1 , C _ { \mathrm { l o w } } \geq 2 C .$ , and the choice $\zeta \leq c _ { \mathrm { i n i t } } / ( 3 C )$ already made in Appendix $\mathbf { A . } \dot { 3 }$ . The check is uniform over the horizon and schedule, including $T = 1$ . Both this game and the unstable quadratic use exact gradients, so their checks apply for every $\sigma \geq 0$

Scope of the separation. All hard instances satisfy (E.1), so the SGDA lower bound persists on the subclass for which Proposition E.1 applies. Under its small-error condition, the complexity comparison uses the same declared gap budget $\Delta .$ . SGDA returns a uniformly selected iterate, whereas xb is computed by additional iterations.

## F COMPLEXITY CALCULATIONS FOR TABLE 1

We derive the entries of Table 1, retaining initialization and additional source parameters until the final specialization. For the upper bounds, assume that the initial outer and inner gaps are $O ( \Delta )$ ; the table fixes ∆. MSGDA and AdaMSGDA require every sampled gradient to be ℓ-Lipschitz, while SPIDER-GDA and SGDA-RR require ℓ-smooth finite-sum components. SGDA and Smoothed-AGDA require only population smoothness and bounded oracle variance. Expected-norm criteria use tolerance ε, and expected-squared-norm criteria use $\varepsilon ^ { 2 }$ . The complete complexities below specify the regimes for the MSGDA and AdaMSGDA specializations and distinguish HCMM-2’s noise contribution from its full cost.

## F.1 SGDA AND SMOOTHED-AGDA

Simultaneous and alternating SGDA. Under the setup of this paper, Yang et al. (2022, Theorem 3.1) for $\mathrm { S G D A _ { A l t } }$ and Theorem A.1 for $\operatorname { S G D A } _ { \operatorname { S i m } }$ give

$$
\begin{array} { r l } { \displaystyle \frac { 1 } { T } \sum _ { t < T } \mathbb { E } \left\| \nabla \Phi ( x _ { t } ) \right\| ^ { 2 } \leq C \left[ \frac { \kappa ^ { 2 } \ell \Delta } { T } + \kappa ^ { 2 } \sigma \sqrt { \displaystyle \frac { \ell \Delta } { T } } \right] , } & { } \\ { \displaystyle T = O \left( 1 + \frac { \kappa ^ { 2 } \ell \Delta } { \varepsilon ^ { 2 } } + \frac { \kappa ^ { 4 } \ell \Delta \sigma ^ { 2 } } { \varepsilon ^ { 4 } } \right) . } & { } \end{array}\tag{F.1}
$$

The second line makes each contribution in the first at most $\varepsilon ^ { 2 } / 2$ . Each iteration uses two fresh partial-gradient calls, so iteration and oracle complexities differ by a constant factor. Fixing $\Delta$ and taking the usual small-error regime $\varepsilon ^ { 2 } \le c \ell \Delta$ gives the two upper-bound entries. The matching lower bound in Theorem 4.1 uses the same squared outer-gradient criterion and optimizes over fixed timescale ratios and predetermined nonincreasing step sizes.

Smoothed-AGDA. Proposition E.1, derived from Yang et al. (2022, Theorem 4.1, Proposition 2.1(b), and Corollary 4.1), includes the conversion to a point x satisfying $\mathbb { E } \left\| \nabla \Phi ( \widehat { \boldsymbol { x } } ) \right\| ^ { 2 } \leq \varepsilon ^ { 2 }$ Under the additional regularized initialization condition (E.1), its total oracle cost is

$$
O \left( \frac { \kappa \ell \Delta } { \varepsilon ^ { 2 } } + \frac { \kappa ^ { 2 } \ell \Delta \sigma ^ { 2 } } { \varepsilon ^ { 4 } } + \left( \kappa + \frac { \kappa ^ { 4 } \sigma ^ { 2 } } { \varepsilon ^ { 2 } } \right) \log ( 2 \kappa ) \right) .\tag{F.2}
$$

$\mathrm { I f } \varepsilon ^ { 2 } \le c \ell \Delta / ( \kappa ^ { 2 } \log ( 2 \kappa ) )$ , then $\kappa ^ { 4 } \sigma ^ { 2 } \varepsilon ^ { - 2 } \log ( 2 \kappa ) \lesssim \kappa ^ { 2 } \ell \Delta \sigma ^ { 2 } \varepsilon ^ { - 4 }$ and $\kappa \log ( 2 \kappa ) \lesssim \kappa \ell \Delta \varepsilon ^ { - 2 }$ . These absorptions give the table’ $\mathsf { s } \widetilde O ( \kappa \ell \varepsilon ^ { - 2 } + \kappa ^ { 2 } \ell \sigma ^ { 2 } \varepsilon ^ { - 4 } )$ entry with $\Delta$ fixed. The returned point is computed by the conversion in Appendix $\mathrm { E , }$ rather than selected directly from the original trajectory.

## F.2 FINITE-SUM UPPER BOUNDS

SPIDER-GDA. For $f = n ^ { - 1 } \sum _ { i = 1 } ^ { n } f _ { i }$ with ℓ-smooth components and attained inner $\mu { \mathrm { - } } \mathrm { P } \mathbf { \dot { E } }$ maximization, Chen et al. (2022, Theorem 6.1 and Lemma $\mathbf { D } . \bar { 2 } )$ use $\kappa = \ell / \mu$ and initial potential $V _ { 0 } = \Phi ( x _ { 0 } ) - \operatorname* { i n f } \Phi + ( \Phi ( x _ { 0 } ) - f ( x _ { 0 } , y _ { 0 } ) ) / 2 4$ . With primal step $\tau _ { x } = \dot { \Theta } ( ( \kappa ^ { 2 } \ell ) ^ { - 1 } )$ , their proof gives a uniformly selected iterate xb with

$$
\begin{array} { c } { \displaystyle \mathbb { E } \left\| \nabla \Phi ( \widehat { x } ) \right\| ^ { 2 } \leq \frac { 2 V _ { 0 } } { K \tau _ { x } } \leq \frac { C \kappa ^ { 2 } \ell V _ { 0 } } { K } , } \\ { \displaystyle \mathrm { c o m p o n e n t - g r a d i e n t c a l l s } = O ( n + K \sqrt { n } ) = O \left( n + \frac { \sqrt { n } \kappa ^ { 2 } \ell V _ { 0 } } { \varepsilon ^ { 2 } } \right) . } \end{array}\tag{F.3}
$$

Here we choose $K = \operatorname* { m a x } \{ 1 , \lceil C \kappa ^ { 2 } \ell V _ { 0 } / \varepsilon ^ { 2 } \rceil \}$ updates, and the source chooses both the refresh period and recursive-estimator batch size of order ${ \sqrt { n } } . \mathrm { A }$ full gradient costs n component calls, which also absorbs the rounding cost. The squared bound gives the source’s stated expected-norm criterion by Jensen’s inequality; $V _ { 0 } = O ( \Delta )$ gives the table entry. No uniform component-variance bound is needed.

SGDA-RR. Cho and Yun (2023, Theorem 1) allow a partial-gradient component variance bound of the form $\begin{array} { r } { n ^ { - 1 } \sum _ { i } \| \nabla _ { j } f _ { i } ( z ) - \nabla _ { j } f ( z ) \| ^ { 2 } \leq A \left\| \nabla _ { j } f ( z ) \right\| ^ { 2 } + B , \operatorname { f o r } j \in \{ x , y \} } \end{array}$ . The table specializes to $A = 0$ and $B = \sigma ^ { 2 }$ . With ratio $\rho \ge 1 4 \kappa ^ { 2 }$ , K epochs, and $V _ { 0 } = 4 ( \Phi ( x _ { 0 } ) - \mathrm { i n f } \Phi ) + \Phi ( x _ { 0 } ) -$ $f ( x _ { 0 } , y _ { 0 } )$ , their bound gives

$$
\begin{array} { r l r } {  { \displaystyle \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathbb { E } \| \nabla \Phi ( { \boldsymbol x } _ { 0 } ^ { k } ) \| ^ { 2 } \leq C [ \frac { \rho \ell V _ { 0 } } { K } + \rho ( \frac { \ell ^ { 2 } \sigma ^ { 2 } V _ { 0 } ^ { 2 } } { n K ^ { 2 } } ) ^ { 1 / 3 } ] , } } \\ & { } & { \quad n K = O ( n + \frac { n \rho \ell V _ { 0 } } { \varepsilon ^ { 2 } } + \frac { \sqrt { n } \rho ^ { 3 / 2 } \ell V _ { 0 } \sigma } { \varepsilon ^ { 3 } } ) . } \end{array}\tag{F.4}
$$

The second line makes both error terms $O ( \varepsilon ^ { 2 } )$ and counts n component calls per epoch, up to the constant factor for the two partial gradients; its first term covers rounding to a positive integer number of epochs. Substituting $\rho \stackrel { \cdot } { = } \Theta ( \kappa ^ { \stackrel { \cdot } { 2 } } )$ and $V _ { 0 } = O ( \Delta )$ , then using $\varepsilon ^ { 2 } \le { \overline { { c } } } \ell \Delta$ to absorb this rounding term, gives the table’s two terms. The output is sampled from epoch starts. Component reuse and sampling without replacement differ from fresh-sample SGDA.

## F.3 MSGDA AND ADAMSGDA

MSGDA. We specialize Huang et al. (2025, Theorem 1) to zero nonsmooth regularizer and unconstrained variables. Assume that each sampled partial gradient is unbiased with variance at most $\sigma ^ { 2 }$ , and that every sampled joint gradient is ℓ-Lipschitz. The source’s full-gradient variance bound is then $2 \sigma ^ { 2 }$ , while its $L _ { f } \mathbf { \bar { \Psi } } = \bar { \ell } \mathbf { a n d } \bar { \kappa } = \ell / \mu$ . We use the valid outer-smoothness bound $L = \ell ( 1 + \kappa )$ from Lemma B.2 in place of the smaller constant printed in the source. Each update draws one fresh sample and evaluates that same sampled function at two iterates, using a constant number of gradient calls.

The source’s proof controls the primal estimator error, the inner gap, and the primal update direction. With $w _ { t }$ denoting its primal gradient estimator, the inner distance estimate in the proof of Lemma B.2 gives

$$
\begin{array} { r } { \left\| \nabla \Phi ( x _ { t } ) \right\| ^ { 2 } \leq 3 \left\| w _ { t } - \nabla _ { x } f ( x _ { t } , y _ { t } ) \right\| ^ { 2 } + 6 \kappa \ell [ \Phi ( x _ { t } ) - f ( x _ { t } , y _ { t } ) ] + 3 \left\| w _ { t } \right\| ^ { 2 } . } \end{array}\tag{F.5}
$$

Indeed, smoothness bounds the squared difference between the outer gradient and the game gradient by 2κℓ times the inner gap. With no regularizer, the source’s primal gradient mapping equals $w _ { t }$ Thus its averaged squared estimates transfer directly to the squared outer-gradient criterion.

Write $D _ { x } = \Phi ( x _ { 1 } ) - \operatorname* { i n f } \Phi , D _ { y } = \Phi ( x _ { 1 } ) - f ( x _ { 1 } , y _ { 1 } )$ , and $D = D _ { x } + D _ { y }$ . The accumulated estimate below bounds the initial gaps and initial estimator variance by a term proportional to $( \kappa \ell D + \kappa ^ { 2 } \sigma ^ { 2 } ) / k .$ and the accumulated estimator variance by a term proportional to κ $^ { \mathrm { t } } k ^ { 2 } \sigma ^ { 2 }$ up to logarithms. Balancing them for $\sigma > 0$ gives the following choice of the source’s parameters:

$$
\begin{array} { l l l } { { k = \displaystyle \left( \frac { \kappa \ell D + \kappa ^ { 2 } \sigma ^ { 2 } } { \kappa ^ { 4 } \sigma ^ { 2 } } \right) ^ { 1 / 3 } , ~ } } & { { ~ } } & { { ~ c _ { 1 } = \displaystyle \frac { 2 } { 3 k ^ { 3 } } + 9 \kappa ^ { 2 } , ~ } } & { { ~ c _ { 2 } = \displaystyle \frac { 2 } { 3 k ^ { 3 } } + \frac { 9 } { 4 } , } } \\ { { m = ( c _ { 1 } k ) ^ { 3 } , ~ } } & { { ~ } } & { { ~ \lambda = \displaystyle \frac { \kappa } { 8 \ell } , ~ } } & { { ~ \gamma = \displaystyle \frac { 1 } { 2 5 6 \kappa \ell } . } } \end{array}\tag{F.6}
$$

These choices satisfy the source proof’s restrictions for every $\ell > 0$ and $\kappa \ge 1 \colon k \ge \kappa ^ { - 2 / 3 }$ $9 \kappa ^ { 2 } \le c _ { 1 } \le ( 2 9 / 3 ) \bar { \kappa } ^ { 2 }$ , and $c _ { 2 } \leq c _ { 1 }$ give all lower bounds on m. In particular, the restriction involving the corrected outer-smoothness constant holds because

$$
\gamma = { \frac { 1 } { 2 5 6 \kappa \ell } } \leq { \frac { 1 } { 1 2 8 \ell ( 1 + \kappa ) } } = { \frac { \lambda \mu } { 1 6 L } } .
$$

The remaining primal and dual step restrictions are unchanged and are met by the displayed choices. The schedule is $\eta _ { t } = k / ( m + t ) ^ { 1 / \bar { 3 } }$ , with momentum weights $c _ { 1 } \eta _ { t } ^ { 2 }$ and $c _ { 2 } \eta _ { t } ^ { 2 }$

The source’s coefficient H and the squared-gradient estimate obtained from its proof equations (59) and (61) satisfy

$$
\begin{array} { c } { \displaystyle H = \frac { D _ { x } } { \gamma k } + \frac { 9 \kappa ^ { 2 } D _ { y } } { k \lambda } + \frac { 4 \sigma ^ { 2 } m ^ { 1 / 3 } } { k ^ { 2 } } + 4 k ^ { 2 } ( c _ { 1 } ^ { 2 } + c _ { 2 } ^ { 2 } ) \sigma ^ { 2 } \log ( m + T ) } \\ { \displaystyle \qquad \leq C \left[ \frac { \kappa \ell D + \kappa ^ { 2 } \sigma ^ { 2 } } { k } + \kappa ^ { 4 } k ^ { 2 } \sigma ^ { 2 } \log ( m + T ) \right] , } \\ { \displaystyle \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \left\| \nabla \Phi ( x _ { t } ) \right\| ^ { 2 } \leq \frac { 1 2 H ( m + T ) ^ { 1 / 3 } } { T } \leq 1 2 H \left( \frac { m ^ { 1 / 3 } } { T } + \frac { 1 } { T ^ { 2 / 3 } } \right) . } \end{array}\tag{F.7}
$$

The last line combines the source’s averaged squared estimates with (F.5). Making its two contributions at most $\varepsilon ^ { 2 } / 2$ requires $T = { \cal O } ( 1 + \bar { H } m ^ { 1 / \bar { 3 } } \varepsilon ^ { - 2 } + H ^ { 3 / 2 } \varepsilon ^ { - 3 } )$ . Substituting the balanced choice of k gives

$$
\begin{array} { c } { { H m ^ { 1 / 3 } = \widetilde { \cal O } ( \kappa ^ { 3 } \ell D + \kappa ^ { 4 } \sigma ^ { 2 } ) , } } \\ { { T = \widetilde { \cal O } \left( 1 + \displaystyle \frac { \kappa ^ { 3 } \ell D + \kappa ^ { 4 } \sigma ^ { 2 } } { \varepsilon ^ { 2 } } + \displaystyle \frac { \kappa ^ { 3 } \ell D \sigma + \kappa ^ { 4 } \sigma ^ { 3 } } { \varepsilon ^ { 3 } } \right) . } } \end{array}
$$

$$
H ^ { 3 / 2 } = \widetilde { \cal O } ( \kappa ^ { 3 } \ell D \sigma + \kappa ^ { 4 } \sigma ^ { 3 } ) ,\tag{F.8}
$$

This is a feasible specialization derived from the cited theorem, rather than the source’s accuracy-only statement. For $0 < \varepsilon \le \sigma$ and fixed D, the $\varepsilon ^ { - 3 }$ terms dominate the $\varepsilon ^ { - 2 }$ terms, yielding the table’s $\widetilde O ( ( \kappa ^ { 3 } \ell \sigma + \kappa ^ { 4 } \sigma ^ { 3 } ) \varepsilon ^ { - 3 } )$ entry. $\mathbf { A } \mathbf { t } \ \sigma = 0$ , instead take $k = \operatorname* { m a x } \{ \kappa ^ { - 2 / 3 } , ( \ell D / ( \kappa ^ { 3 } \varepsilon ^ { 2 } ) ) ^ { 1 / 3 } \}$ with the remaining choices in (F.6); the same estimate gives $O ( 1 + \kappa ^ { 3 } \ell D \dot { \varepsilon } ^ { - 2 } )$

AdaMSGDA. Theorem 2 of Huang et al. (2025) additionally assumes $A _ { t } \succeq a I$ and $b _ { - } I \preceq B _ { t } \preceq$ $b _ { + } I$ , where $a > 0$ and $0 < b _ { - } \le b _ { + }$ are fixed spectral constants. To apply the source’s dual-update estimate, enlarge the upper bound to $\bar { b } _ { + } = \operatorname* { m a x } \{ b _ { + } , b _ { - } ^ { - 1 } \}$ . Then $B _ { t } \preceq \bar { b } _ { + } I$ and $b _ { - } \bar { b } _ { + } \geq 1$ . For its dual gradient estimator $v _ { t } .$ , the update $\widetilde { y } _ { t + 1 } = y _ { t } + \lambda B _ { t } ^ { - 1 } v _ { t }$ satisfies

$$
v _ { t } ^ { \top } B _ { t } ^ { - 1 } v _ { t } \geq b _ { - } \left. B _ { t } ^ { - 1 } v _ { t } \right. ^ { 2 } \geq \frac { \left. \widetilde { y } _ { t + 1 } - y _ { t } \right. ^ { 2 } } { \lambda ^ { 2 } \bar { b } _ { + } } .
$$

This supplies the coefficient used in equation (66) of the source proof with $\bar { b } _ { + }$ as the spectral upper bound.

Use the same $k , c _ { 2 }$ as above, but take $c _ { 1 } = 2 / ( 3 k ^ { 3 } ) + 9 ( \bar { b } _ { + } / b _ { - } ) \kappa ^ { 2 }$ and $m = ( c _ { 1 } k ) ^ { 3 }$ . For explicit admissible steps, set $d = 1 / ( 8 \bar { b } _ { + } )$ and $g \ = \ \operatorname* { m i n } \{ d / ( 3 2 \bar { b } _ { + } ) , \dot { b } _ { - } / ( 1 6 \bar { b } _ { + } ) , a / \dot { 8 } , a ^ { 2 } d / ( 9 \bar { b } _ { + } ) \}$ , and choose $\lambda = d \kappa / \ell$ and $\gamma = g / ( \kappa \ell )$ . The outer-smoothness restriction again holds with $L = \ell ( 1 + \kappa )$ since $g \leq d / ( 3 2 \bar { b } _ { + } )$ gives

$$
\gamma \leq \frac { d } { 3 2 \bar { b } _ { + } \kappa \ell } \leq \frac { d } { 1 6 \bar { b } _ { + } \ell ( 1 + \kappa ) } = \frac { \lambda \mu } { 1 6 \bar { b } _ { + } L } .
$$

The other terms defining $g$ satisfy the remaining primal step restrictions, while $m ^ { 1 / 3 } / k = c _ { 1 }$ satisfies the schedule-dependent primal and dual restrictions. Its coefficient G obeys

$$
\begin{array} { l } { { \displaystyle { G = \frac { D _ { x } } { \gamma k a } + \frac { 9 \bar { b } _ { + } \kappa ^ { 2 } D _ { y } } { k \lambda a ^ { 2 } } + \frac { 4 \sigma ^ { 2 } m ^ { 1 / 3 } } { k ^ { 2 } a ^ { 2 } } + \frac { 4 k ^ { 2 } ( c _ { 1 } ^ { 2 } + c _ { 2 } ^ { 2 } ) \sigma ^ { 2 } } { a ^ { 2 } } \log ( m + T ) } } } \\ { { \displaystyle { \quad \le C _ { a , b _ { - } , b _ { + } } \left[ \frac { \kappa \ell D + \kappa ^ { 2 } \sigma ^ { 2 } } { k } + \kappa ^ { 4 } k ^ { 2 } \sigma ^ { 2 } \log ( m + T ) \right] . } } } \end{array}\tag{F.9}
$$

Equations (96)–(97) of the source proof give $\begin{array} { r } { T ^ { - 1 } \sum _ { t = 1 } ^ { T } \mathbb { E } \left\| A _ { t } ^ { - 1 } \nabla \Phi ( x _ { t } ) \right\| ^ { 2 } \leq 1 2 G ( m + T ) ^ { 1 / 3 } / T } \end{array}$ Thus (F.8) and its table specialization hold for this adaptive gradient mapping, with constants depending on $a , b _ { - } , b _ { + } . \textrm { A }$ Euclidean outer-gradient guarantee additionally requires $A _ { t } \preceq a _ { + } I \colon$ then $\| \nabla \Phi ( x _ { t } ) \| \le a _ { + } \| A _ { t } ^ { - 1 } \nabla \Phi ( x _ { t } ) \|$ , so replace the mapping tolerance by $\varepsilon / a _ { + }$ . The source’s lower bound on A alone does not supply this conversion. Both sampled-gradient methods therefore require stronger oracle information than the bounded-variance fresh-sample model of this paper.

## F.4 HCMM-2

Cai et al. (2026, Theorem 2) control the average expected outer-gradient norm. Their assumptions require attained inner PŁ maximization, population-gradient smoothness ℓ, population-Hessian Lipschitzness $M ,$ and stochastic gradient and Hessian oracles based on fresh independent samples across iterations, with conditional unbiasedness and variances at most $\sigma ^ { 2 }$ and $\sigma _ { h } ^ { 2 } .$ , respectively. The parameters M and $\sigma _ { h }$ are not determined by $\ell , \kappa , \sigma$ . Initialization uses one stochastic gradient; each subsequent iteration uses a constant number of gradient and Hessian-vector operations.

Write $D _ { x } = \Phi ( x _ { 0 } ) - \operatorname* { i n f } \Phi , D _ { y } = \Phi ( x _ { 0 } ) - f ( x _ { 0 } , y _ { 0 } )$ , and $D = D _ { x } + D _ { y }$ . For momentum weight $b \in ( 0 , 1 ]$ and normalized step lengths $a _ { y } = a > 0$ and $a _ { x } = a / ( 6 \kappa )$ , the source’s conditions (27) hold. Its Appendix E, equation (134), gives

$$
\begin{array} { r } { \displaystyle \frac { 1 } { T } \sum _ { t < T } \mathbb { E } \left\| \nabla \Phi ( x _ { t } ) \right\| \leq C \Bigg [ \frac { \kappa D } { a T } + \frac { \kappa \sigma } { b T } + \frac { \kappa M a ^ { 2 } } { b } + \frac { \kappa \sigma _ { h } a } { \sqrt { b } } } \\ { \displaystyle + \kappa \sigma \sqrt { b } + \kappa ^ { 2 } \ell a + \frac { \sqrt { \kappa \ell D _ { y } } } { T } \Bigg ] . } \end{array}\tag{F.10}
$$

The last term uses the PŁ quadratic-growth bound dist $\begin{array} { r } { ( y _ { 0 } , \mathrm { a r g } \operatorname* { m a x } _ { y } f ( x _ { 0 } , y ) ) \le \sqrt { 2 \kappa D _ { y } / \ell } } \end{array}$ . We use the larger outer-gap coefficient obtained directly from the source’s descent inequality (46); the coefficient printed in (134) differs by a constant factor, which does not change these orders.

Choose $a = b = T ^ { - 2 / 3 }$ . Substitution into (F.10), followed by making each contribution at most $\varepsilon / 3$ yields

$$
\begin{array} { c } { \displaystyle \frac { 1 } { T } \sum _ { t < T } \mathbb { E } \left\| \nabla \Phi ( x _ { t } ) \right\| \leq C \left[ \frac { \kappa ( D + \sigma + \sigma _ { h } ) } { T ^ { 1 / 3 } } + \frac { \kappa M + \kappa ^ { 2 } \ell } { T ^ { 2 / 3 } } + \frac { \sqrt { \kappa \ell D _ { y } } } { T } \right] , } \\ { \displaystyle T = O \left( 1 + \frac { \kappa ^ { 3 } ( D ^ { 3 } + \sigma ^ { 3 } + \sigma _ { h } ^ { 3 } ) } { \varepsilon ^ { 3 } } + \frac { ( \kappa M + \kappa ^ { 2 } \ell ) ^ { 3 / 2 } } { \varepsilon ^ { 3 / 2 } } + \frac { \sqrt { \kappa \ell D _ { y } } } { \varepsilon } \right) . } \end{array}\tag{F.11}
$$

For a uniformly selected iterate, this is a sufficient complexity for expected norm at most ε, not expected squared norm at most $\varepsilon ^ { 2 }$ . Under $\sigma _ { h } \le \sigma$ , its noise contribution at order $\varepsilon ^ { - 3 } \operatorname { i s } O ( \kappa ^ { 3 } \sigma ^ { 3 } \varepsilon ^ { - 3 } )$ giving the table’s $O _ { \ell }$ entry.

An additional sufficient regime in which this entry bounds the complete cost is $\sigma > 0 , D = O ( \sigma )$ $\sigma _ { h } \ \leq \ \sigma$ , and $0 < \varepsilon \le \bar { c } \operatorname* { m i n } \{ \sigma , \sigma ^ { 2 } / ( \ell + M / \kappa ) \}$ The second accuracy restriction absorbs the Hessian-regularity term, and the two restrictions together with $D _ { y } \leq D$ absorb the terminal term. Hence (F.11) becomes $O ( \kappa ^ { 3 } \sigma ^ { 3 } \varepsilon ^ { - 3 } )$ throughout this regime. For general initialization, the full bound also contains $\kappa ^ { 3 } D ^ { 3 } \varepsilon ^ { - 3 }$ ; fixing the initial gaps does not make this term smaller in $\varepsilon .$ The table then displays a noise contribution, while (F.11) retains the initialization and Hessian-regularity terms.

## F.5 LOWER BOUNDS AND THE NONCONVERGENCE ENTRIES

NC-SC oracle lower bounds. For deterministic linear-span methods, Zhang et al. (2021, Theorem 3.1) prove $\Omega ( \sqrt { \kappa } \ell \Delta \varepsilon ^ { - 2 } )$ full-gradient calls to reach outer-gradient norm at most ε. Their finite-sum Theorem 3.2 gives $\Omega ( n + \sqrt { n \kappa } \ell \Delta \varepsilon ^ { - 2 } )$ expected component calls to reach an ε-stationary query for randomized linear-span methods. Its criterion is expected hitting time, as defined in their equations $( 4 ) \ - - ( 6 )$ , rather than the expected gradient norm of a specified output. This result requires $\kappa \geq 2 n$ and averaged component smoothness $\begin{array} { r } { n ^ { - 1 } \sum _ { i } \| \nabla f _ { i } ( z ) ^ { - } \nabla f _ { i } ( z ^ { \prime } ) \| ^ { \bar { 2 } } \le \ell ^ { 2 } \left\| z - z ^ { \prime } \right\| ^ { 2 } } \end{array}$ , which is weaker than smoothness of every component. Both rates apply in the stated small-error regime $\varepsilon ^ { 2 } \le c \ell \Delta$ ; fixing $\Delta$ gives the two table entries.

The first-order lower bound of Li et al. (2021, Theorem 2) is proved for zero-respecting algorithms on nonempty closed convex domains X, Y, with $0 \in \mathcal { X }$ and $\begin{array} { r } { \Phi ( 0 ) - \operatorname* { m i n } _ { x \in \mathcal { X } } \Phi ( x ) \leq \Delta } \end{array}$ . Its projected mapping is $G _ { \Phi } ( x ) : = L _ { \Phi } [ x - P _ { \mathcal { X } } ( x - \nabla \Phi ( x ) / L _ { \Phi } ) ]$ , where $P _ { \mathcal { X } }$ is Euclidean projection and $L _ { \Phi } \le ( \kappa + 1 ) \ell$ is the source’s value-function smoothness constant. The target is $\mathbb { E } \left. G _ { \Phi } ( \widehat { x } ) \right. \leq \varepsilon$ and the lower bound is $\Omega ( \sqrt { \kappa } \ell \Delta \varepsilon ^ { - 2 } + \kappa ^ { 1 / 3 } \ell \Delta \sigma ^ { 2 } \varepsilon ^ { - 4 } )$ calls to its bounded-variance stochastic first-order oracle. Fixing ∆ gives the table entry; its criterion and algorithm class differ from the unconstrained SGDA recursion studied here.

NC-PŁ oracle lower bound. Pan and Li (2026, Theorem 3.2) prove $\Omega ( \kappa \ell \Delta \varepsilon ^ { - 2 } )$ exact first-order calls for arbitrary deterministic first-order methods, with only an initial outer-gap budget. The target is ordinary outer-gradient stationarity, with κ above a universal constant and $\bar { \varepsilon } ^ { 2 } \le \breve { c } \ell \Delta$ . Fixing ∆ gives the NC-PŁ first-order row.

Ratio obstructions and our matching complexity. Li et al. (2022, Theorem 4.1) construct an NC-SC quadratic on which the joint iterates of simultaneous GDA fail to converge locally for every positive constant step when $\rho \le \kappa$ . This implies the displayed local obstruction for $\rho = o ( \kappa )$ ; it is not an averaged outer-gradient complexity bound.

For our update family, Theorems 4.1, 5.1 and 6.3 and the matching upper bounds give, respectively,

$$
\begin{array} { r l r } & { N _ { \varepsilon , \mathrm { S i m } } ^ { \star } , \ N _ { \varepsilon , \mathrm { A l t } } ^ { \star } = \Theta \left( \displaystyle \frac { \kappa ^ { 2 } \ell \Delta } { \varepsilon ^ { 2 } } + \frac { \kappa ^ { 4 } \ell \Delta \sigma ^ { 2 } } { \varepsilon ^ { 4 } } \right) , } & \\ & { N _ { \varepsilon , \mathrm { S i m } } ( \rho ) , \ N _ { \varepsilon , \mathrm { A l t } } ( \rho ) \geq c \left( \displaystyle \frac { \rho \ell \Delta } { \varepsilon ^ { 2 } } + \frac { \kappa ^ { 4 } \ell \Delta \sigma ^ { 2 } } { \varepsilon ^ { 4 } } \right) \quad \mathrm { i f } \rho \geq c _ { \mathrm { h i g h } } \kappa ^ { 2 } . } & \end{array}\tag{F.12}
$$

These complexity statements use $\varepsilon ^ { 2 } \le c \ell \Delta$ , with κ above the respective theorem thresholds. Fixing $\Delta$ gives the optimized and high-ratio entries. For $1 \le \rho \le \bar { c _ { \mathrm { l o w } } } \kappa ^ { 2 }$ , Theorem 6.3 instead gives mi $\begin{array} { r } { \mathbf { \Phi } _ { 1 : < T } \left\| \nabla \Phi ( x _ { t } ) \right\| ^ { 2 } \geq c \ell \Delta } \end{array}$ on a hard deterministic instance for every prescribed horizon and nonincreasing schedule. Thus the $\Omega ( 1 )$ entry records a nonvanishing stationarity error, rather than an iteration complexity; the hard instance may depend on that horizon and schedule.