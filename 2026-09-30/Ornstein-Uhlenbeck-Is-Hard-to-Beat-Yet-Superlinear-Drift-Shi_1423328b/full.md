# Ornstein–Uhlenbeck Is Hard to Beat, Yet Superlinear Drift Ships Lower Transport Costs

Attila Lovas

L´or´ant Nagy

HUN-REN Alfr´ed R´enyi Institute of Mathematics Budapest, Hungary

## Abstract

Breˇsar and Mijatovi´c [1] show that Ornstein–Uhlenbeck difusion is hard to beat in forward convergence under assumptions that exclude superlinear drift. We instead test superlinear Langevin difusions for score-based image generation, computing their conditional scores numerically from a Fokker–Planck equation. In our experiments, the superlinear models beat the Ornstein–Uhlenbeck baseline on empirical Wasserstein distance across nearly the entire tested grid and show less variation across difusion horizons. The “hard to beat” verdict of [1] thus fails to be universal.

## 1 Introduction

Let $\mu _ { \star }$ be a data distribution on $\mathbb { R } ^ { d }$ . Difusion-based generative modelling introduces a forward stochastic process that progressively destroys the structure of $\mu _ { \star } .$ , and then constructs a reverse process that transports a simple terminal law back toward the data distribution. This principle appears in the difusion model of Sohl-Dickstein et al. [10], the denoising difusion model of Ho et al. [7], and the continuous-time score-based formulation of Song et al. [11]. The reverse-time difusion formula itself is classical [2, 6], and score matching provide the learning principle.

Common continuous-time constructions use afine Gaussian corruptions whose conditional transition laws and conditional scores are available analytically. The present experiment replaces the linear mean-reverting drift by a coordinatewise superlinear Langevin drift. The transition score is then computed numerically from the associated Fokker–Planck equation and used as the regression target for the score network.

The construction does not require an analytical expression for either the corrupted-data marginal $p _ { t }$ or the conditional transition density $p _ { t | 0 }$ . When the forward difusion has a suficiently regular transition density, its conditional score can instead be approximated by solving the associated Fokker–Planck equation. This permits forward processes beyond those with closed-form Gaussian transitions. In the present experiment we restrict attention to coordinatewise Langevin difusions, for which the conditional score can be computed from a one-dimensional equation and reused across coordinates.

## 2 Forward difusion

Fix $\alpha \geq 1$ and $c _ { \alpha } > 0$ . Define the scalar potential

$$
U _ { \alpha } ( y ) = \frac { c _ { \alpha } } { \alpha + 1 } | y | ^ { \alpha + 1 } ,
$$

and the potential on $\mathbb { R } ^ { d }$ by

$$
\mathcal { U } _ { \alpha } ( { \boldsymbol { x } } ) = \sum _ { i = 1 } ^ { d } U _ { \alpha } ( { \boldsymbol { x } } ^ { i } ) .
$$

The forward process is the gradient Langevin difusion

$$
d X _ { t } = B _ { \alpha } ( X _ { t } ) d t + d W _ { t } , \qquad X _ { 0 } \sim \mu _ { \star } , \quad t \in [ 0 , T ] .\tag{1}
$$

where

$$
B _ { \alpha } ( x ) = - \nabla { \mathcal U } _ { \alpha } ( x ) .
$$

Coordinatewise we use the notation,

$$
B _ { \alpha } ( x ) = \bigl ( b _ { \alpha } ( x ^ { 1 } ) , \dots , b _ { \alpha } ( x ^ { d } ) \bigr ) ,
$$

where by definition

$$
b _ { \alpha } ( y ) = - U _ { \alpha } ^ { \prime } ( y ) = - c _ { \alpha } | y | ^ { \alpha } \operatorname { s g n } ( y ) .
$$

Define

$$
Z _ { \alpha } = \int _ { \mathbb { R } ^ { d } } \exp [ - 2 \mathcal { U } _ { \alpha } ( x ) ] \ d x , \qquad V _ { \alpha } ( x ) = 2 \mathcal { U } _ { \alpha } ( x ) + \log Z _ { \alpha } .
$$

The invariant probability measure of (1) is

$$
\Pi _ { \alpha } ( d x ) = e ^ { - V _ { \alpha } ( x ) } d x = Z _ { \alpha } ^ { - 1 } \exp [ - 2 \mathcal { U } _ { \alpha } ( x ) ] \ d x .\tag{2}
$$

The coordinates evolve independently conditional on the initial image. For $y _ { 0 } \in \mathbb { R }$ , let $q ^ { y _ { 0 } } ( t , y )$ denote the transition density of the scalar difusion

$$
d Y _ { t } = b _ { \alpha } ( Y _ { t } ) d t + d B _ { t } , \qquad Y _ { 0 } = y _ { 0 } .
$$

Then the conditional transition density of (1) factorizes as

$$
p _ { t | 0 } ( x \mid x _ { 0 } ) = \prod _ { i = 1 } ^ { d } q ^ { x _ { 0 } ^ { i } } ( t , x ^ { i } ) .\tag{3}
$$

## 3 Time reversal for the superlinear family

We now justify the reverse-time dynamics of the continuous-time process (1) using the findings in [4]. Assume that

$$
H ( \mu _ { \star } \mid \Pi _ { \alpha } ) < \infty .\tag{4}
$$

Let Y follow the same Langevin dynamics as X, with an invariant initial law:

$$
d Y _ { t } = B _ { \alpha } ( Y _ { t } ) d t + d W _ { t } , \qquad Y _ { 0 } \sim \Pi _ { \alpha } .
$$

Denote the path laws of X and Y by P and R, respectively. In the notation of [4], the difusion matrix in the present setup is

$$
a = I _ { d } .
$$

Since $V _ { \alpha } ( x ) = 2 { \mathcal U } _ { \alpha } ( x ) +$ log $Z _ { \alpha }$ , we have $\nabla V _ { \alpha } ( x ) = 2 \nabla \mathcal { U } _ { \alpha } ( x )$ , and therefore $\begin{array} { r } { B _ { \alpha } ( x ) = - \frac { 1 } { 2 } a \nabla V _ { \alpha } ( x ) } \end{array}$ Thus R is the reversible difusion associated with the invariant measure

$$
\Pi _ { \alpha } ( d x ) = e ^ { - V _ { \alpha } ( x ) } d x .
$$

Moreover, $V _ { \alpha } \in C ^ { 1 } (  { \mathbb { R } } ^ { d } ) , a = I _ { d }$ is constant, and the growth condition of [4] is satisfied. Indeed,

$$
\begin{array} { l } { x \cdot B _ { \alpha } ( x ) + \mathrm { t r } ( a ) = - c _ { \alpha } \displaystyle \sum _ { i = 1 } ^ { d } x ^ { i } | x ^ { i } | ^ { \alpha } \mathrm { s g n } ( x ^ { i } ) + d } \\ { = - c _ { \alpha } \displaystyle \sum _ { i = 1 } ^ { d } | x ^ { i } | ^ { \alpha + 1 } + d } \\ { \leq d . } \end{array}
$$

The processes X and Y have the same conditional path law given their initial point and difer only in their initial distributions. Consequently,

$$
H ( P \mid R ) = H ( \mu _ { \star } \mid \Pi _ { \alpha } ) < \infty .
$$

Thus the reference difusion satisfies the required nondegeneracy, reversibility, and growth conditions moreover, $P$ is Markov and, under assumption (4), has finite relative entropy with respect to R. The time-reversal theorem of [4] therefore applies, and hence the stationary process $Y$ is reversible.

Remark. The finite-entropy assumption (4) can be enforced by an arbitrarily small Gaussian regularization of the data distribution. Indeed, if $\mu _ { \star }$ is supported on a bounded subset of $\mathbb { R } ^ { d }$ , as is the case for normalized image data, then for any $\varepsilon > 0$ the regularized law

$$
\mu _ { \star } ^ { \varepsilon } = \mu _ { \star } * \mathcal { N } ( 0 , \varepsilon ^ { 2 } I _ { d } )
$$

has a smooth strictly positive density and satisfies

$$
H ( \mu _ { \star } ^ { \varepsilon } \mid \Pi _ { \alpha } ) < \infty .
$$

Thus the assumption (4) may be ensured by adding an arbitrarily small amount of independent Gaussian noise to the data.

## 4 Reverse difusion and the score

Let $p _ { t }$ denote the density of $X _ { t } .$ , and define reverse time by

$$
\tau = T - t , \qquad \bar { X } _ { \tau } = X _ { T - \tau } .
$$

Under the finite-entropy assumption (4), the argument in the preceding section gives

$$
\begin{array} { r } { d \bar { X } _ { \tau } = \left[ - B _ { \alpha } ( \bar { X } _ { \tau } ) + \nabla \log p _ { T - \tau } ( \bar { X } _ { \tau } ) \right] d \tau + d \bar { W } _ { \tau } , \quad \quad \bar { X } _ { 0 } \sim \mathrm { L a w } ( X _ { T } ) . } \end{array}\tag{5}
$$

As usual, the unknown quantity in this equation is the marginal score

$$
\nabla \log p _ { t } ( x ) .
$$

Although the conditional forward dynamics decouple coordinatewise, the marginal law at time t generally does not, because the initial distribution $\mu _ { \star }$ contains dependencies between image coordinates. Indeed,

$$
p _ { t } ( x ) = \int _ { \mathbb { R } ^ { d } } p _ { t | 0 } ( x \mid x _ { 0 } ) \mu _ { \star } ( d x _ { 0 } ) .
$$

The marginal score at x depends on the posterior distribution of the original image $X _ { 0 }$ given $X _ { t } = x$ A neural network $s _ { \theta } ( x , t )$ is therefore trained to approximate the full d-dimensional marginal score. The regression target is the conditional score. The denoising score-matching objective is

$$
\begin{array} { r } { \mathcal { L } ( \boldsymbol { \theta } ) = \mathbb { E } \left[ \left. \boldsymbol { s } _ { \boldsymbol { \theta } } ( X _ { t } , t ) - \nabla _ { \boldsymbol { x } } \log p _ { t | 0 } ( X _ { t } \mid X _ { 0 } ) \right. _ { 2 } ^ { 2 } \right] . } \end{array}\tag{6}
$$

Under the usual regularity assumptions,

$$
\mathbb { E } \left[ \nabla _ { x } \log p _ { t | 0 } ( X _ { t } \mid X _ { 0 } ) \mid X _ { t } = x \right] = \nabla _ { x } \log p _ { t } ( x ) ,\tag{7}
$$

so the population minimizer of (6) is the marginal score required in (5) [12, 11].

## 5 Score construction and numerical implementation

The time-reversal formula specifies the reverse drift, but its marginal score ∇ log $p _ { t }$ is not available in closed form. To train the network, we instead construct the conditional score in (6). For fixed y<sub>0</sub>, the scalar transition density $q ^ { y _ { 0 } } ( t , y )$ defined in Section 2 satisfies the Fokker–Planck equation

$$
\partial _ { t } q = - \partial _ { y } \bigl ( b _ { \alpha } ( y ) q \bigr ) + \frac { 1 } { 2 } \partial _ { y } ^ { 2 } q , \qquad q ( 0 , \cdot ) = \delta _ { y _ { 0 } } .\tag{8}
$$

Consequently, the conditional score has components

$$
\begin{array} { r } { s ( y _ { 0 } , t , y ) = \partial _ { y } \log q ^ { y _ { 0 } } ( t , y ) , \qquad \left[ \nabla _ { x } \log p _ { t | 0 } ( x \mid x _ { 0 } ) \right] ^ { i } = s ( x _ { 0 } ^ { i } , t , x ^ { i } ) . } \end{array}\tag{9}
$$

For the nonlinear choices of α, we approximate $q ^ { y _ { 0 } }$ numerically rather than use a closed-form transition density [9, 3].

We solve (8) on a finite grid for a range of starting values y<sub>0</sub>. A narrow Gaussian approximates the initial point mass. We then take finite diferences of the log-density and store the resulting conditional scores in a table indexed by $( y _ { 0 } , t , y )$ . During training, scores are obtained from this table by trilinear interpolation.

For the experiments, the data consist of $2 8 \times 2 8$ grayscale images normalized to $[ - 1 , 1 ] ^ { d }$ , where $d = 7 8 4$ . We set $c _ { \alpha } = 0 . 5 , c _ { 0 } = 0$ , and test $\alpha \in \{ 1 , 2 , 3 \} , T \in \{ 3 , \ldots , 1 0 \}$ , and $N \in \{ 1 0 0 , 1 5 0 , 2 0 0 \}$ The score table uses 200 starting-value points in $[ - 1 . 5 , 1 . 5 ]$ , 700 time points in $[ 0 , T ]$ , and 200 state points in $[ - 5 . 5 , 5 . 5 ]$ . The initial Gaussian has standard deviation 0.03, and densities are floored at $1 0 ^ { - 1 2 }$ before taking logarithms. For $\alpha = 1$ , the forward process is $\begin{array} { r } { d X _ { t } = - \frac { 1 } { 2 } X _ { t } d t + d W _ { t } } \end{array}$ and its invariant law is $\mathcal { N } ( 0 , I _ { d } )$

Training samples of the forward process are generated by Euler–Maruyama with step size $h =$ $T / N$

$$
\widehat { X } _ { k + 1 } = \widehat { X } _ { k } + h B _ { \alpha } ( \widehat { X } _ { k } ) + \sqrt { h } \xi _ { k } , \qquad \xi _ { k } \sim \mathcal { N } ( 0 , I _ { d } ) .\tag{10}
$$

For each image, one nonzero time is selected uniformly from its N-step trajectory. A U-Net is trained to predict the tabulated conditional score at that time. It has 64 base channels, channe multipliers (1, 2), two residual blocks per level, dropout 0.1, and a time embedding of dimension 128. We use AdamW for 15 epochs with batch size 64, learning rate $1 0 ^ { - 5 }$ , and zero weight decay.

Reverse sampling also uses Euler–Maruyama with N steps. Its starting law is approximated by the invariant law $\Pi _ { \alpha } ;$ samples from that law are approximated by evolving standard normal samples under the forward Langevin equation to time $T _ { \mathrm { e q } } = 3 0$ . This equilibration uses N steps of size $3 0 / N$ . For superlinear drifts, explicit Euler can produce rare, very large excursions [8]; numerical explosions are monitored, and generated images with more than ten percent non-finite coordinates are discarded at evaluation time.

The code used for the numerical experiments is available at https: $/ / { \tt g i }$ thub.com/lorant-nagy/ genai.

## 6 Evaluation and results

We evaluate generated images using the empirical Wasserstein–1 distance with Euclidean ground cost. Real and generated $2 8 \times 2 8$ images are mapped to $[ 0 , 1 ] ^ { d }$ and flattened, a fixed set of 3000 rea images is used. The distance is computed by solving the discrete optimal-transport problem [5].

Across the tested grid, a superlinear choice $( \alpha = 2 \ \mathrm { o r } \ \alpha = 3 )$ gives a lower value than the linear baseline in 23 of the 24 fixed $( T , N )$ comparisons; the exception is $N = 1 0 0 , T = 8$ . The best value in each table is also attained by a superlinear model, and the row summaries show less variation across T. We conjecture that stronger mean reversion contributes to both the improved Wasserstein–1 scores and their reduced sensitivity to the choice of T.

Our findings qualify the “hard to beat” characterization of the Ornstein–Uhlenbeck process in [1]: the results therein concern forward convergence to stationarity under at-most-linear drift, while our superlinear models achieve lower empirical generation error in the experiments reported here.

<table><tr><td></td><td colspan="8"> $T$ </td></tr><tr><td> $\alpha$ </td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td></tr><tr><td>3</td><td>7.4895</td><td>7.1817</td><td>7.2128</td><td>7.3525</td><td>7.5041</td><td>7.6127</td><td>7.6784</td><td>7.7590</td></tr><tr><td>2</td><td>8.0436</td><td>7.1735</td><td>7.1995</td><td>7.2805</td><td>7.4185</td><td>7.5190</td><td>7.7112</td><td>7.6554</td></tr><tr><td>1</td><td>10.7980</td><td>8.1018</td><td>9.5526</td><td>7.4124</td><td>7.4386</td><td>7.4421</td><td>7.7372</td><td>7.7217</td></tr></table>

Table 1: $\mathcal { W } _ { 1 }$ for $N = 1 0 0$ . Lower is better; darker shading indicates a smaller value.

$$
T
$$

<table><tr><td>α</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td></tr><tr><td>3</td><td>7.6303</td><td>7.2553</td><td>7.2215</td><td>7.4292</td><td>7.4934</td><td>7.4947</td><td>7.5667</td><td>7.9462</td></tr><tr><td>2</td><td>8.3078</td><td>7.2908</td><td>7.4071</td><td>7.4038</td><td>7.4565</td><td>7.5470</td><td>7.5285</td><td>8.1154</td></tr><tr><td>1</td><td>11.1001</td><td>11.1918</td><td>9.2831</td><td>7.5673</td><td>10.7467</td><td>7.9927</td><td>8.3327</td><td>9.8957</td></tr></table>

Table 2: $\mathcal { W } _ { 1 }$ for $N = 1 5 0$ . Lower is better; darker shading indicates a smaller value.

$$
T
$$

<table><tr><td>α</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td></tr><tr><td>3</td><td>8.3180</td><td>7.2164</td><td>7.3147</td><td>7.3457</td><td>7.3593</td><td>7.5742</td><td>7.8070</td><td>7.6255</td></tr><tr><td>2</td><td>9.4270</td><td>7.4931</td><td>7.5542</td><td>7.5268</td><td>7.3764</td><td>7.5384</td><td>8.1468</td><td>8.1259</td></tr><tr><td>1</td><td>11.1501</td><td>8.5657</td><td>9.3656</td><td>11.5609</td><td>10.7299</td><td>10.1254</td><td>9.6737</td><td>8.2259</td></tr></table>

Table 3: $\mathcal { W } _ { 1 }$ for $N = 2 0 0$ . Lower is better; darker shading indicates a smaller value.
<table><tr><td>N</td><td> $\alpha = 1$ </td><td> $\alpha = 2$ </td><td> $\alpha = 3$ </td></tr><tr><td>100</td><td> $7 . 9 2 \pm 0 . 7 6$ </td><td> $7 . 4 2 \pm 0 . 2 2$ </td><td> $7 . 4 7 \pm 0 . 2 3$ </td></tr><tr><td>150</td><td> $9 . 2 9 \pm 1 . 3 9$ </td><td> $7 . 5 4 \pm 0 . 2 7$ </td><td> $7 . 4 9 \pm 0 . 2 4$ </td></tr><tr><td>200</td><td> $9 . 7 5 \pm 1 . 1 7$ </td><td> $7 . 6 8 \pm 0 . 3 2$ </td><td> $7 . 4 6 \pm 0 . 2 1$ </td></tr></table>

Table 4: Mean and sample standard deviation over $T \in \{ 4 , \ldots , 1 0 \}$

## References

[1] M. Breˇsar and A. Mijatovi´c. Nonasymptotic bounds for forward processes in denoising diffusions: Ornstein–Uhlenbeck is hard to beat. Annals of Applied Probability, 35(6):4439–4463, 2025.

[2] B. D. O. Anderson. Reverse-time difusion equation models. Stochastic Processes and their Applications, 12(3):313–326, 1982.

[3] V. I. Bogachev, N. V. Krylov, M. R¨ockner, and S. V. Shaposhnikov. Fokker–Planck– Kolmogorov Equations. American Mathematical Society, 2015.

[4] P. Cattiaux, G. Conforti, I. Gentil, and C. L´eonard. Time reversal of difusion processes under a finite entropy condition. Annales de l’Institut Henri Poincar´e, Probabilit´es et Statistiques, 59(4):1844–1881, 2023.

[5] R. Flamary, N. Courty, A. Gramfort, et al. POT: Python Optimal Transport. Journal of Machine Learning Research, 22(78):1–8, 2021.

[6] U. G. Haussmann and E. Pardoux. Time reversal of difusions. The Annals of Probability, 14(4):1188–1205, 1986.

[7] J. Ho, A. Jain, and P. Abbeel. Denoising difusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020.

[8] M. Hutzenthaler, A. Jentzen, and P. E. Kloeden. Strong and weak divergence in finite time of Euler’s method for stochastic diferential equations with non-globally Lipschitz continuous coeficients. Proceedings of the Royal Society A, 467(2130):1563–1576, 2011.

[9] H. Risken. The Fokker–Planck Equation: Methods of Solution and Applications. 2nd edition, Springer, 1989.

[10] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. Proceedings of the 32nd International Conference on Machine Learning, 2256–2265, 2015.

[11] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. Score-based generative modeling through stochastic diferential equations. International Conference on Learning Representations, 2021.

[12] P. Vincent. A connection between score matching and denoising autoencoders. Neural Computation, 23(7):1661–1674, 2011.