# Patch Attention Wisely: Understanding Patch Length in Attention-Based Forecasting

Junyu Chen<sup>\*</sup> Department of Economics, Econometrics and Finance University of Groningen

Likai Chen Department of Statistics and Data Science Washington University in St. Louis

Weining Wang School of Economics University of Bristol

## Abstract

Attention-based time series forecasters commonly group consecutive observations into patches. Patch length is known to have a substantial impact on forecasting performance, yet its selection remains largely empirical. We develop a theoretical framework that explains why patch length matters for forecasting accuracy and provides a principled basis for its selection. For linear state-space models, we derive an exact factorization of the forecasting coeficient matrix that separates forecast dynamics from filtering dynamics. We show that dominance of a single forecast component at long horizons does not generally imply a rank-one representation. This explains why longer horizons alone need not reduce representational complexity. We combine the factorization with estimation bounds to develop a patch-selection criterion that balances representation and estimation errors under temporal dependence. Our analysis establishes conditions under which balanced patching is justified and clarifies how predictive structure, head count, training sample size, and long-run dependence enter the selection criterion.

## 1 INTRODUCTION

Patching has become a basic design choice in attention-based time-series forecasting. A history of length L can be presented as L point tokens or as $N = L / p$ non-overlapping patches of length p. Increasing p shortens the token sequence and can make long contexts computationally feasible. Within one attention head, a shared linear value map applies the same p within-patch weights to every patch, and the query–key scores only rescale this pattern from patch to patch. Patch length is therefore not only a computational choice: it also determines which forecasts a fixed number of heads can express and how many coefficients must be learned from data. With limited data or long histories, an unsuitable patch length can leave too many coeficients to estimate or too little flexibility to fit the target, even when it reduces attention cost.

Prior work has established that patch tokens can be both accurate and eficient. PatchTST (Nie et al., 2023) showed that patching preserves local temporal structure, reduces the number of tokens, and permits longer look-back windows, and patch tokenization is now used in large pretrained forecast ers such as TimesFM (Das et al., 2024). It has also shown that a single temporal resolution need not sufice: MTST (Zhang et al., 2024) uses parallel patch scales, Pathformer (Chen et al., 2024) combines scales through adaptive pathways, and TimeSqueeze (Ankireddy et al., 2026) moves patch boundaries according to local signal complexity. Complementary studies compare point-wise, patch-wise, and variatewise tokens (Chen et al., 2025), and a controlled anal ysis finds that adaptive patching should be compared against carefully tuned uniform patches (Zucchi et al., 2026). These works explain how to build efective tokenizers and when they perform well empirically.

Most existing work on patching is algorithmic and empirical: it proposes tokenization schemes and compares their accuracy and computational cost, but it provides no statistical theory for choosing a fixed patch length for a given forecasting model. Prediction errors alone cannot isolate the efect of patch length, because changing p also changes the number of tokens, how parameters are shared across time, the number of coeficients to estimate, the optimization problem, and the computational cost. Multiscale and adaptive architectures face the same ambiguity. Moreover, nonoverlapping patching does not discard any observation but only regroups the observations into patches, so any loss of forecasting power must come from the model applied after this regrouping. A statistical theory of patch selection must therefore first characterize how patch length changes that model and then determine when one patch length should be preferred.

We provide such a characterization for a multi-head linear attention model. Each head applies one set of within-patch weights, and its query–key scores determine how strongly these weights are used in each patch. Summed over heads, these contributions form an $N \times p$ coeficient matrix that contains the weight on every observation in the history, and with m heads this matrix has rank at most m. When the attention weights do not depend on the input and every patch position has its own learnable key, the model can represent exactly the coeficient matrices of rank at most m. Choosing p therefore reshapes the same length-L linear forecast into a diferent matrix, which can change both the rank needed to represent it and the dificulty of estimating it. This reduction to reduced-rank regression isolates the efect of patch length without claiming to describe every component of a full Transformer.

A central dificulty is that a long horizon does not make the forecast simple to represent. In a partially observed dynamical system, a long-horizon forecast may be dominated by one slowly evolving state compo nent, and it is tempting to conclude that one weight ing pattern then sufices. We show that this is not true, because the forecaster must first recover that slow component from observations contaminated by faster latent dynamics and measurement noise. Using a linear state-space model, we factor the optimal forecast coeficients into horizon-dependent weights and horizon-independent filtering patterns. The factorization shows that the component dominating the distant future can still require several filtering patterns to recover. It also identifies patch aliasing, in which a patch length makes two diferent filtering patterns produce identical weights across patches. Consequently, the best low-rank approximation can keep a positive relative error as the horizon grows.

We relate the infinite-past coeficients to the best forecast from a finite window and bound the error of rank constrained least squares fitted to overlapping windows of one dependent series. The bound separates approximation error from an estimation term governed by the dimension of the rank-constrained coeficient class. When the rank constraint is active, this dimension is $m ( L / p + p - m )$ , which is smallest when the patch length and the number of patches are equal. This suggests patch lengths near $\sqrt { L }$ when approximation errors are comparable, although with enough data a less balanced reshape can be preferable if it represents the target more accurately.

In summary, we contribute (i) an exact identity linking head count and patch length to the coeficient class of the model, (ii) a state-space factorization and a long-horizon floor on the relative approximation error, which separate the components that drive the forecast from the filtering patterns needed to recover them, (iii) finite-window and dependent-data bounds that expose the approximation–estimation tradeof in patch selection, and (iv) a mixture setting in which inputdependent attention weights can improve the forecast because the observed history reveals which system generated it. Simulations show that intermediate patch lengths near $\sqrt { L }$ help most when training data are scarce, and that their advantage narrows as the sample grows.

Related work. Our question difers from showing that attention can implement a Kalman filter or learn an autoregressive rule in context (Goel and Bartlett, 2024; Sander et al., 2024; Lu and Yang, 2025): we study how a specified forecasting model distributes coeficients over a patched history. The non-overlapping coeficient reshape is related to Pagematrix methods, which use low-rank structure for time-series estimation and prediction (Damen et al., 1982; Agarwal et al., 2018), and contrasts with the overlapping Hankel matrices common in realization theory (Glover, 1984). Spectral learning of dynamical systems provides complementary low-dimensional descriptions of long-memory predictors (Hazan et al., 2017; Rashidinejad et al., 2020). We combine these structural ideas with reduced-rank regression (Izenman, 1975; Velu and Reinsel, 1987) and dependent time-series estimation (Basu and Michailidis, 2015). The new link is that the rank constraint is induced directly by attention heads and changes shape with the patch length.

## 2 PATCH ATTENTION AS REDUCED-RANK REGRESSION

In this section, we connect attention to reduced-rank regression and separate approximation error from estimation error.

Let $( X _ { t } )$ be a centered stationary scalar process with finite second moments. We forecast $Y _ { t } = X _ { t + H } , H \geq$ 1, from the window $\boldsymbol { x } _ { t , L } = ( X _ { t } , \ldots , X _ { t - L + 1 } ) ^ { \top }$ . Let $p$ denote the patch length, assume that $p$ divides $L ,$ and write $N = L / p$ for the number of non-overlapping patch tokens. Define

$$
Z _ { t , p } [ j , s ] = X _ { t - ( j p + s ) } , \quad 0 \leq j < N , \quad 0 \leq s < p .\tag{1}
$$

Thus $Z _ { t , p } \in \mathbb { R } ^ { N \times p }$ , with rows ordered from recent to

older patches and observations within each row ordered in reverse chronological order. Patching rearranges the window without discarding observations.

To describe the attention mechanism, let $\mathbf { T } \in \mathbb { R } ^ { N ; }$ ×d contain token embeddings incorporating patch content and position. Head a forms the arrays

$$
\mathbf { Q } _ { a } = \mathbf { T } \widetilde { W } _ { Q , a } , \quad \mathbf { K } _ { a } = \mathbf { T } \widetilde { W } _ { K , a } , \quad \mathbf { V } _ { a } = \mathbf { T } \widetilde { W } _ { V , a } ,\tag{2}
$$

where $\mathbf { Q } _ { a } , \mathbf { K } _ { a } \in \mathbb { R } ^ { N \times d _ { k } }$ and $\mathbf { V } _ { a } \in \mathbb { R } ^ { N \times d _ { v } }$ . Standard dot-product attention normalizes query–key similarities through a softmax. To obtain a tractable model, we replace this normalization by unnormalized scores with a fixed factor $1 / N$ . The resulting all-token attention operation is

$$
\mathbf { O } _ { a } = \frac { 1 } { N } ( \mathbf { Q } _ { a } \mathbf { K } _ { a } ^ { \top } ) \mathbf { V } _ { a } = \frac { 1 } { N } \mathbf { Q } _ { a } ( \mathbf { K } _ { a } ^ { \top } \mathbf { V } _ { a } ) .\tag{3}
$$

The scores may be signed and need not sum to one, and the factor $1 / N$ depends only on the token count.

For forecasting, we specialize this operation to one forecast query per head, and we compute each value by applying the same linear map to the raw patch. Write $x _ { j } ~ \in ~ \mathbb { R } ^ { p }$ for the transpose of row j of an input Z. A forecast-token representation $c _ { H } ( Z ) \in \mathbb { R } ^ { d }$ constructed from the observed window and horizon, supplies the query $q _ { a } ( Z ) = \widetilde { W } _ { Q , a } ^ { \top } c _ { H } ( Z )$ . Let $k _ { a , j } ( Z )$ be the column vector corresponding to row $j$ of ${ \bf K } _ { a } .$ We take $v _ { a , j } ( Z ) \ : = \ : W _ { V , a } x _ { j }$ , where the same matrix $W _ { V , a } \in \mathbb { R } ^ { d _ { v } \times p }$ is used for every patch. A linear patch embedding that is common to all patches can be absorbed into $W _ { V , a }$ . The analyzed forecast consists solely of these attention outputs combined through bias-free linear output weights $w _ { a } ~ \in ~ \mathbb { R } ^ { d _ { v } }$ With $\alpha _ { a , j } ( Z ) =$ $N ^ { - 1 } k _ { a , j } ( Z ) ^ { \top } q _ { a } ( Z ) ^ { \top }$ , it is

$$
\widehat { f } ( Z ) = \sum _ { a = 1 } ^ { m } \sum _ { j = 0 } ^ { N - 1 } \alpha _ { a , j } ( Z ) w _ { a } ^ { \top } W _ { V , a } x _ { j } .\tag{4}
$$

Because every patch passes through the same value and output maps, the coeficients on the raw observations have the following structure.

Proposition 1 (Coeficient matrix). Let b<sub>a</sub> = W<sup>⊤</sup> V,a w<sub>a</sub> ∈ R<sup>p</sup> and α<sub>a</sub>(Z) $( \alpha _ { a , 0 } ( Z ) , \ldots , \alpha _ { a , N - 1 } ( Z ) ) ^ { \top }$ . Then

$$
\widehat { f } ( Z ) = \langle B ( Z ) , Z \rangle _ { \mathrm { F } } , \qquad B ( Z ) = \sum _ { a = 1 } ^ { m } \alpha _ { a } ( Z ) b _ { a } ^ { \top } ,\tag{5}
$$

and rank $B ( Z ) \leq \operatorname* { m i n } \{ m , N , p \}$ for every input $Z .$

Each head supplies one within-patch weighting pattern $b _ { a } .$ , whose contribution across patches is determined by $\alpha _ { a } ( Z )$ . Unlike the $N \times N$ query–key score matrix,

$B ( Z )$ acts directly on the observations. Its rank bound holds even for input-dependent attention weights, although $B ( Z )$ then varies with Z and the forecast is generally nonlinear.

For tractability, we restrict the model to fixed attention weights: each head uses a constant forecast query and keys that depend only on the patch position, so the weights do not depend on the input and $B ( Z ) = B$ . This restriction suits linear state-space forecasting, whose target is the best linear forecast from the window, and allows representation and estimation errors to be separated explicitly. It is an additional assumption, not a consequence of Proposition 1.

If every patch position has its own learnable key and the value and output maps are unrestricted, every $N \times$ p matrix of rank at most m can be represented with m heads (Appendix B.2). Thus the predictor class induced by Equation (4) with fixed attention weights is exactly

$$
\mathcal { M } _ { m } ( p ) = \{ B \in \mathbb { R } ^ { N \times p } : \operatorname { r a n k } ( B ) \leq m \} .\tag{6}
$$

Fitting this model by least squares is therefore reduced-rank regression on the reshaped window.

Let $b ^ { \circ } \in \mathbb { R } ^ { L }$ denote the coeficient vector of the best linear forecast of $Y _ { t } ~ = ~ X _ { t + H }$ from $\begin{array} { r } { x _ { t , L } , } \end{array}$ and let $R _ { 0 }$ denote its mean squared prediction error. Assume $\Gamma _ { L } \ = \ \mathrm { C o v } ( x _ { t , L } ) \ \succ \ 0$ . For each divisor p of $L ,$ let $B _ { p } ^ { \circ } \in \mathbb { R } ^ { N \times p }$ be the corresponding reshape:

$$
B _ { p } ^ { \circ } [ j , s ] = b _ { j p + s } ^ { \circ } , \qquad 0 \leq j < N , \quad 0 \leq s < p .\tag{7}
$$

Lemma 2 (Reshaping invariance). With columnstacking vectorization, there is a permutation matrix $\Pi _ { p }$ such that vec $( Z _ { t , p } ) = \Pi _ { p } x _ { t , L }$ . Consequently,

$$
\Gamma _ { L , p } : = \mathrm { C o v } \{ \mathrm { v e c } ( Z _ { t , p } ) \} = \Pi _ { p } \Gamma _ { L } \Pi _ { p } ^ { \intercal }\tag{8}
$$

has the same eigenvalues as $\Gamma _ { L }$ for every p. For every $\boldsymbol { B } \in \mathbb { R } ^ { N \times p }$ R

$$
\mathbb { E } ( Y _ { t } - \langle B , Z _ { t , p } \rangle _ { \mathrm { F } } ) ^ { 2 } = R _ { 0 } + \| B - B _ { p } ^ { \circ } \| _ { \Gamma } ^ { 2 } ,\tag{9}
$$

where $\| D \| _ { \Gamma } ^ { 2 } = \operatorname { v e c } ( D ) ^ { \top } \Gamma _ { L , p } \operatorname { v e c } ( D )$ . In particular, the minimum risk over unrestricted linear forecasts is $R _ { 0 }$ for every patch length.

Patching does not change the best linear forecast, but it changes the shape of the coeficient matrix that m heads must approximate. Define the target rank and covariance-weighted approximation error by

$$
r _ { p } = \mathrm { r a n k } B _ { p } ^ { \circ } , \quad A _ { m } ( p ) = \operatorname* { m i n } _ { B \in \mathcal { M } _ { m } ( p ) } \| B - B _ { p } ^ { \circ } \| _ { \Gamma } ^ { 2 } .\tag{10}
$$

The approximation error vanishes if and only if $m \geq$ $r _ { p } .$ . Otherwise, its size depends on both the reshaped

coeficients and the covariance of the observations. Patch selection balances this error against the dificulty of estimating a matrix in $\mathcal { M } _ { m } ( p )$ (Section 4).

## 3 FORECAST HORIZON AND APPROXIMATION LOSS UNDER A STATE-SPACE MODEL

Section 2 characterizes forecasting loss through the target coeficients and their approximation by attention heads, but leaves the role of horizon implicit. To characterize this loss further, we use a state-space model as a tractable benchmark. Forecasting then involves estimating the current latent state from noisy observations and propagating it to the forecast horizon. This separation makes the horizon dependence of the coeficients explicit, allowing us to study their rank and approximation loss under patching. In particular, recovering a single dominant state component at long horizons can still require several temporal filtering patterns, so its patch representation need not have rank one.

## 3.1 The State-Space Model and Its Forecast Coeficients

Let

$$
S _ { t + 1 } = A S _ { t } + \eta _ { t + 1 } , \qquad X _ { t } = C S _ { t } + \xi _ { t } ,\tag{11}
$$

where $S _ { t } \in \mathbb { R } ^ { r } , A \in \mathbb { R } ^ { r \times r }$ , and $C \in \mathbb { R } ^ { 1 \times r }$ . The centered noises are mutually independent and indepen dent across time, with finite second moments, process covariance $Q \succeq 0$ , and measurement variance $R > 0$ Assume $\rho _ { A } < 1$ , where $\rho _ { M }$ denotes the spectral radius of a matrix M, and stationary initialization.

To expose the filtering structure, first consider predic tion from the entire observed past. Let $K \in \mathbb { R } ^ { r \times 1 }$ be the steady-state Kalman predictor gain and set $F = A - K C$ The best linear predictor $m _ { t }$ of $S _ { t }$ based on $( X _ { s } : s < t )$ satisfies

$$
m _ { t + 1 } = F m _ { t } + K X _ { t } , \qquad \rho _ { F } < 1 .\tag{12}
$$

Consequently, the best linear forecast from the infinite past is

$$
\begin{array} { c } { { \displaystyle \widehat { X } _ { t + H | t } = \sum _ { \ell \geq 0 } \beta _ { H } ( \ell ) X _ { t - \ell } , } } \\ { { \beta _ { H } ( \ell ) = C A ^ { H - 1 } F ^ { \ell } K . } } \end{array}\tag{13}
$$

We now split the coeficients in (13) into filtering modes. A mode is a dynamical component associated with an eigenvalue and its eigendirection. Filtering modes are the modes of F, whose powers describe how past observations reveal the current state, and forecast modes are the modes of A, which carries that estimate to the forecast horizon. This separates the efect of the horizon, which changes their weights, from the efect of patch length, which changes how they are arranged. These two roles need not have the same efective dimension.

Keep the latest L coeficients and arrange them like the input:

$$
B _ { H , p } [ j , s ] = \beta _ { H } ( j p + s ) .\tag{14}
$$

The matrix $B _ { H , p }$ keeps the first L coeficients of the infinite-past forecast, whereas $B _ { p } ^ { \circ }$ is the best forecast that uses only the window. We also write $B _ { H , L , p } ^ { \circ } =$ $B _ { p } ^ { \circ }$ when displaying the horizon and window length explicitly. Theorem 22 in Appendix D bounds this diference, so the spectral results for $B _ { H , p }$ below also describe $B _ { p } ^ { \circ }$

## 3.2 From Filtering Modes to Patch Patterns

We next express the coeficients in (13) through filtering modes and connect these modes to patch patterns. Assume that A and F have distinct eigenvalues and are diagonalizable over C. Write $\begin{array} { r } { A = \sum _ { a = 1 } ^ { \bar { r } } \lambda _ { a } P _ { a } } \end{array}$ and $\begin{array} { r } { F = \sum _ { k = 1 } ^ { r } \varphi _ { k } Q _ { k } } \end{array}$ , where $P _ { a }$ and $Q _ { k }$ are the corresponding spectral projectors. Define

$$
g _ { a k } = C P _ { a } Q _ { k } K , \qquad d _ { k } ( H ) = \sum _ { a = 1 } ^ { r } g _ { a k } \lambda _ { a } ^ { H - 1 } ,\tag{15}
$$

and the geometric patterns

$$
\begin{array} { c } { { u _ { N , p } ( z ) = ( 1 , z ^ { p } , \ldots , z ^ { ( N - 1 ) p } ) ^ { \top } , } } \\ { { v _ { p } ( z ) = ( 1 , z , \ldots , z ^ { p - 1 } ) ^ { \top } . } } \end{array}\tag{16}
$$

Theorem 3 (Filtering-mode factorization). For every $H \geq 1$ ，

$$
B _ { H , p } = \sum _ { k = 1 } ^ { r } d _ { k } ( H ) u _ { N , p } ( \varphi _ { k } ) v _ { p } ( \varphi _ { k } ) ^ { \top } ,\tag{17}
$$

so rank $( B _ { H , p } ) \ \leq \ \operatorname* { m i n } \{ r , N , p \}$ . The filtering factors are independent of the forecast horizon.

Each filtering mode contributes one across-patch pattern and one within-patch pattern, since $\varphi ^ { j p + s } =$ $( \varphi ^ { p } ) ^ { j } \varphi ^ { s }$ The horizon changes their weights $d _ { k } ( H )$ rather than their shapes. Complex terms occur in conjugate pairs and sum to a real matrix. Appendix C provides the proof.

## 3.3 A Relative Filtering Floor (H → ∞)

We now show that a low-rank patch representation can keep a non-vanishing relative approximation error, even when one forecast mode dominates as $H  \infty$ To see this, suppose one forecast eigenvalue dominates: $\lambda _ { 1 } \neq 0$ is simple and $\begin{array} { r } { \left| \lambda _ { 1 } \right| > \operatorname* { m a x } _ { a \geq 2 } \left| \lambda _ { a } \right| } \end{array}$ , with $r \geq 2$ Set

$$
\begin{array} { r } { q _ { A } = \underset { a \geq 2 } { \operatorname* { m a x } } \left| \lambda _ { a } / \lambda _ { 1 } \right| < 1 , } \\ { B _ { \infty , p } = \displaystyle \sum _ { k = 1 } ^ { r } g _ { 1 k } u _ { N , p } ( \varphi _ { k } ) v _ { p } ( \varphi _ { k } ) ^ { \top } , } \end{array}\tag{18}
$$

and assume $\begin{array} { r l r l r l r l } { B _ { \infty , p } } & { { } \neq } & { 0 . } & { { } \quad } & { \mathrm { L e t } \quad C _ { p } } & { } & { { } = } \end{array}$ $\begin{array} { r } { \sum _ { a = 2 } ^ { r } \sum _ { k = 1 } ^ { r } | g _ { a k } | \| u _ { N , p } ^ { \blacktriangle } ( \varphi _ { k } ) \| _ { 2 } \| v _ { p } ( \varphi _ { k } ) \| _ { 2 } . } \end{array}$ Rescaling by $\lambda _ { 1 } ^ { - ( H - 1 ) }$ removes the decay of the dominant forecast mode and reveals the limiting coeficient matrix $B _ { \infty , p } .$ For a rank budget h, the following theorem shows that the relative approximation error in operator norm converges to that of this matrix. Thus a longer horizon need not remove the relative error caused by using too few heads.

Theorem 4 (Long-horizon filtering floor). For every $H \geq 1$ ,

$$
\| \lambda _ { 1 } ^ { - ( H - 1 ) } B _ { H , p } - B _ { \infty , p } \| _ { \mathrm { o p } } \leq C _ { p } q _ { A } ^ { H - 1 } .\tag{19}
$$

Let $R _ { H , p , h } ^ { \mathrm { o p } } ~ = ~ \sigma _ { h + 1 } ( B _ { H , p } ) / \sigma _ { 1 } ( B _ { H , p } )$ and $\kappa _ { \infty , h } ( p ) \ =$ $\sigma _ { h + 1 } ( B _ { \infty , p } ) / \sigma _ { 1 } ( B _ { \infty , p } ) . \quad { \cal I } f \ C _ { p } q _ { A } ^ { H - 1 } \ \le \ \sigma _ { 1 } ( B _ { \infty , p } ) / 2 ,$ then

$$
| R _ { H , p , h } ^ { \mathrm { o p } } - \kappa _ { \infty , h } ( p ) | \leq \frac { 4 C _ { p } q _ { A } ^ { H - 1 } } { \sigma _ { 1 } ( B _ { \infty , p } ) } .\tag{20}
$$

The limit is positive when $h < \mathrm { r a n k } ( B _ { \infty , p } )$

The limiting rank depends on how many filtering modes contribute to the dominant forecast component. If the patch matrix has enough rows and columns, and these modes remain distinct both across patches and within each patch, their patterns are linearly independent. The following corollary identifies the limiting rank with the number of active filtering modes.

Corollary 5 (Active filtering modes and patch aliasing). Let $S = \{ k : g _ { 1 k } \neq 0 \}$ $I f \mid S \mid \leq$ min $\{ N , p \}$ and the nodes $\{ \varphi _ { k } : k \in S \}$ and $\{ \varphi _ { k } ^ { p } : k \in \mathcal { S } \}$ are each pairwise distinct, then rank $( B _ { \infty , p } ) = | S |$

The active modes depend on the system, while patching determines whether their across-patch patterns remain distinct. For example, φ and −φ produce identical across-patch patterns when $p$ is even. We call this coincidence patch aliasing.

The floor is relative. Writing $\begin{array} { r l } { \mathcal { E } _ { h } ^ { \Gamma } ( B ) } & { { } = } \end{array}$ min<sub>rank</sub> $\ b { M } \leq \ b { h } \ \| \ b { B } - \ b { M } \| _ { \Gamma } ^ { 2 }$ , for fixed $L , p , h$

$$
\mathcal { E } _ { h } ^ { \Gamma } ( B _ { H , p } ) = | \lambda _ { 1 } | ^ { 2 ( H - 1 ) } \{ \mathcal { E } _ { h } ^ { \Gamma } ( B _ { \infty , p } ) + O ( q _ { A } ^ { H - 1 } ) \} .\tag{21}
$$

Absolute approximation error therefore vanishes with the horizon, even though its share of the squared coeficient size remains positive when $h < \mathrm { r a n k } ( B _ { \infty , p } )$

## 4 CHOOSING PATCH LENGTH

Choosing a patch length from finite data also requires accounting for estimation error. We connect the population results to the finite-window target, establish an estimation bound, and derive its implications for patch choice. Throughout, L, H, and $m \geq 1$ are fixed.

Truncating an infinite-past forecast need not yield the optimal coeficients for a finite observation window. Under the diagonalizability assumption on F in Section 3, the finite-window argument underlying Theorem 22 in Appendix D gives, for fixed H and any fixed $\bar { \rho } _ { F } \in \left( \rho _ { F } , 1 \right)$ ,

$$
\| B _ { p } ^ { \circ } - B _ { H , p } \| _ { \Gamma } = O ( \bar { \rho } _ { F } ^ { L } ) ,\tag{22}
$$

uniformly over admissible patch lengths. Since $\rho _ { F } < 1$ the diference becomes geometrically negligible as L increases. In particular, when $m \geq r ,$ , Theorem 3 gives $A _ { m } ( p ) = { \cal O } ( \bar { \rho } _ { F } ^ { 2 L } )$ , without requiring exact low rank of $B _ { p } ^ { \circ } .$

## 4.1 Estimation from Overlapping Windows

To account for estimation, let $\widehat { B } _ { p , m , H }$ be a global leastsquares estimator over $\mathcal { M } _ { m } ( p )$ , fitted to the same n consecutive windows for each p in the finite nonempty set $\mathcal { P } _ { L }$ of candidate divisors of $L .$ . Use the same windows for every candidate. The empirical counterpart of the population approximation problem is

$$
\widehat B _ { p , m , H } \in \mathop { \mathrm { \arg m i n } } _ { B \in \mathcal { M } _ { m } ( p ) } \frac { 1 } { n } \sum _ { t = 1 } ^ { n } \{ X _ { t + H } - \langle B , Z _ { t , p } \rangle _ { \mathrm { F } } \} ^ { 2 } .\tag{23}
$$

For an independent stationary test window, the conditional prediction risk of this fitted coeficient is $R _ { 0 } + \| \widehat { B } _ { p , m , H } - B _ { p } ^ { \circ } \| _ { \Gamma } ^ { 2 }$ , by the projection identity in Section 2. It therefore sufices to control this predictionnorm error.

The estimation bound depends on the dimension of the coeficient class $\mathcal { M } _ { m } ( p )$ , denoted by

$$
\begin{array} { r } { d _ { m } ( p ) = \left\{ \begin{array} { l l } { m ( L / p + p - m ) , } & { m < \operatorname* { m i n } \{ L / p , p \} , } \\ { L , } & { m \geq \operatorname* { m i n } \{ L / p , p \} . } \end{array} \right. } \end{array}\tag{24}
$$

Theorem 6 (Finite-sample prediction bound). Maintain the stationary state-space assumptions of Section 3, and additionally assume Gaussian process and measurement noises. There exist positive constants $C _ { 0 } , C _ { 1 } , C _ { 2 }$ , depending only on the state-space model, H, and $L ,$ such that, for every $0 < \delta < 1$ , if

$$
n \geq C _ { 0 } L \operatorname* { m a x } _ { p \in \mathcal { P } _ { L } } \left\{ d _ { m } ( p ) + m \log \frac { 1 2 | \mathcal { P } _ { L } | } { \delta } \right\} ,\tag{25}
$$

then, with probability at least $1 - \delta ,$ , simultaneously for every $p \in \mathcal { P } _ { L }$ , any global minimizer in (23) satisfies

$$
\begin{array} { r l r } {  { \frac { \| \widehat { B } _ { p , m , H } - B _ { p } ^ { \circ } \| _ { \Gamma } ^ { 2 } } { n } } } \end{array}\tag{26}
$$

The constants are common to all patch lengths and independent of $n , \ m , \ \mathcal { P } _ { L }$ , and δ. The bound applies whether or not the rank constraint is active.

In (26), n counts the observed training windows. Their overlap and serial dependence are accounted for in the constants and the sample-size requirement. The Gaussian assumption in Theorem 6 can be relaxed under suitable independence and tail conditions on the driving noise. Sub-Gaussian and sub-exponential extensions require adapting the concentration bounds in Appendix $\mathrm { E } ,$ while retaining the deterministic oracle argument. The constants, rates, and sample-size requirements may change. The result is a predictionerror upper bound, not an exact bias–variance decomposition, and does not require the constrained estimator to be unbiased.

## 4.2 Patch Selection and the Approximation–Estimation Tradeof

For fixed $L , H , m ,$ and confidence level, the logarith mic term in (26) is common to all candidates. Comparing the stated prediction bounds therefore reduces to minimizing

$$
C _ { 1 } A _ { m } ( p ) + \frac { C _ { 2 } } { n } d _ { m } ( p ) , \qquad p \in \mathcal { P } _ { L } ,\tag{27}
$$

where $C _ { 1 }$ and $C _ { 2 }$ are the constants in Theorem $6 .$ Since the approximation errors and these constants may be unknown, (27) is an oracle comparison of prediction bounds, rather than a feasible selection rule or an exact expression for prediction risk.

When the rank constraint is active, $m < \operatorname* { m i n } \{ L / p , p \}$ minimizing $d _ { m } ( p )$ for fixed L and m is equivalent to minimizing $p + L / p ,$ which favors a balanced choice of patch length and patch count, with $p = L / p = \sqrt { L }$ when admissible. Balanced patches therefore minimize the estimation penalty in (27), but need not minimize the approximation error $A _ { m } ( p )$

Theorem 7 (Balanced patch lengths). Maintain the conditions of Theorem $\delta ,$ including (25), and suppose $m < \operatorname* { m i n } \{ L / q , q \}$ for every $q \in \mathcal { P } _ { L }$ . Let $p \in \mathcal { P } _ { L }$ minimize $d _ { m } ( p )$ . If

$$
C _ { 1 } \{ A _ { m } ( p ) - A _ { m } ( q ) \} < \frac { C _ { 2 } } { n } \{ d _ { m } ( q ) - d _ { m } ( p ) \}\tag{28}
$$

for every $q \in \mathcal { P } _ { L }$ with $d _ { m } ( q ) > d _ { m } ( p )$ , then every minimizer of (27) minimizes $p + L / p$ over $\mathcal { P } _ { L }$ . In particular, $i f \sqrt { L } \in \mathcal { P } _ { L }$ , it uniquely minimizes (27).

Condition (28) allows a less balanced patch to approximate the target better, provided that this improvement is smaller than its additional estimation penalty. If $A _ { m } ( p )$ is constant across candidates, as when rank $( B _ { p } ^ { \circ } ) \ \leq \ m$ for every candidate, the criterion and $d _ { m } ( \bar { p } )$ have exactly the same minimizers. If several divisors attain the minimum dimension, diferences in approximation error determine their ordering. The margin shrinks as n grows: for two candidates with $d _ { m } ( p ) < d _ { m } ( q )$ but $A _ { m } ( p ) > A _ { m } ( q )$ , the criterion at $p$ is no larger than that at q precisely when

$$
n \leq \frac { C _ { 2 } } { C _ { 1 } } \frac { d _ { m } ( q ) - d _ { m } ( p ) } { A _ { m } ( p ) - A _ { m } ( q ) } .\tag{29}
$$

Equality gives a tie, and the comparison applies only to sample sizes satisfying (25). The smaller class is favored below this threshold and the more accurate representation above it. The square-root rule is therefore not universal: its advantage comes from estimation.

Extreme and intermediate patch lengths. The two extreme patch lengths, point tokens $( p = 1 )$ and a single token containing the whole window $( p = L )$ lie at one end of the tradeof in (27). Their coeficient matrix is a single column or row, so its rank is at most one and one head represents every linear forecast: $A _ { m } ( p ) \ = \ 0$ for any target. The price is that the rank constraint is inactive. Nothing is shared across patches, the fitted class is the unrestricted set of linear forecasts on the window, and $d _ { m } ( p ) = L ,$ the largest possible dimension. Rank one is automatic here and is not a form of compression (Lemma 29). $\mathrm { A n }$ intermediate patch length with $m < \operatorname* { m i n } \{ L / p , p \}$ makes the opposite trade. All patches must reuse the same m within-patch weighting patterns, which lowers the dimension to $d _ { m } ( p ) = m ( L / p + p - m ) < L$ This sharing can introduce approximation error, since $A _ { m } ( p ) \ > \ 0$ unless rank $( B _ { p } ^ { \circ } ) ~ \leq ~ m$ Taking $q$ to be an extreme patch length in (29) shows that the criterion is no larger at the intermediate patch exactly when $C _ { 1 } A _ { m } ( p ) \leq C _ { 2 } \{ L - d _ { m } ( p ) \} / n$ , that is, when the approximation error it adds is at most the estimation penalty it saves. This favors intermediate patches when data are limited or the reshaped target is close to rank $m .$ . Because the saving shrinks with $n ,$ an extreme patch length is eventually preferred whenever $A _ { m } ( p ) \ > \ 0$ Computational cost, which the bound omits, also falls as the token count $L / p$ decreases $( \mathrm { A p - }$ pendix F.6).

## 5 NONLINEAR DECODING AND SCOPE

Although our guarantees concern a structured attention model, they isolate mechanisms relevant to more general patch-based forecasters. Patch length changes how temporal information is grouped and how rep resentations are shared, afecting both predictive ap proximation and learning complexity. The distinction between forecasting and filtering also explains why a longer horizon need not simplify extracting predictive information from the past. These insights motivate tuning patch length jointly with model capacity and using balanced patches as a reference when representation errors are comparable. More expressive architectures introduce additional efects, but the underlying representation–estimation tradeof remains a useful organizing principle.

Proposition 8 (Nonlinear decoding after shared patch encoding). Under the notation ofSection 2, suppose $( x _ { t , L } , X _ { t + H } )$ is jointly Gaussian. For a shared encoder $\boldsymbol { E } \in \mathbb { R } ^ { p \times m }$ and any measurable decoder g with square-integrable output,

$$
\operatorname* { i n f } _ { E , g } \mathbb { E } \big [ \{ X _ { t + H } - g ( Z _ { t , p } \boldsymbol { E } ) \} ^ { 2 } \big ] = R _ { 0 } + A _ { m } ( p ) .\tag{30}
$$

Here m is the encoder width, and the decoder receives only the encoded patches $Z _ { t , p } E$

Thus the population approximation limit also applies to nonlinear attention and multilayer decoders operating only on the encoded patches. This result concerns encoder width. It does not extend the head-count characterization or the estimation bound to arbitrary nonlinear models. The proof is in Appendix B.9.

## 6 SIMULATIONS

## 6.1 Design

We generate stationary paths from a three-state model of the form (11), whose parameters are given in $\mathrm { A p \mathrm { - } }$ pendix I.1. The forecasting target is $X _ { t + H }$ , using the latest L observations. We use window lengths $L \in \{ 6 4 , 1 4 4 \}$ , horizons $H \in \{ 1 , 2 , 4 , 8 , 1 2 \}$ , and $n \in$ {5, 25, 200} independent training series.

The fitted model is one-head attention with fixed attention weights: a constant query, a learnable key for each patch position, and the same linear value map for every patch. It can represent every rank-one coeficient matrix, so the experiment isolates how patch length afects learning. Appendix I gives the design details, the population construction, diagnostic definitions, and reporting protocol.

## 6.2 Forecast Horizon and the Interior Patch Range

Figure 1 reports excess MSE: the held-out mean of $( \widehat { f } ( x ) - f _ { H , L } ^ { \circ } ( x ) ) ^ { 2 }$ , where $f _ { H , L } ^ { \circ }$ is the analytic finitewindow conditional mean. At the population level, total forecast risk is the sum of this excess risk and the Bayes residual variance. Measuring the distance to the conditional mean removes the unpredictable noise in the target, which would otherwise hide diferences between fitted models.

Table 1: One-step test MSE in units of $1 0 ^ { - 3 }$ for the balanced patch and both extremes. Entries are means ± standard errors over 50 data repetitions, and n counts independent training series.
<table><tr><td> $L$ </td><td>n</td><td> $p = 1$ </td><td> $p = { \sqrt { L } }$ </td><td> $p = L$ </td></tr><tr><td>64</td><td>5</td><td> $2 3 8 . 6 \pm 2 . 2$ </td><td> $1 8 4 . 4 \pm 1 . 3$ </td><td> $2 3 8 . 9 \pm 2 . 1$ </td></tr><tr><td>64</td><td>25</td><td> $1 8 2 . 8 \pm { 0 . 8 }$ </td><td> $1 7 2 . 0 \pm 0 . 6$ </td><td> $1 8 2 . 8 \pm { 0 . 8 }$ </td></tr><tr><td>64</td><td>200</td><td> $1 7 1 . 6 \pm 0 . 6$ </td><td> $1 6 9 . 8 \pm 0 . 6$ </td><td> $1 7 2 . 0 \pm 0 . 6$ </td></tr><tr><td>144</td><td>5</td><td> $3 2 0 . 7 \pm 3 . 0$ </td><td> $1 9 4 . 1 \pm 1 . 7$ </td><td> $3 2 8 . 4 \pm 3 . 7$ </td></tr><tr><td>144</td><td>25</td><td> $2 0 0 . 5 \pm 1 . 0$ </td><td> $1 7 3 . 5 \pm 0 . 7$ </td><td> $2 0 1 . 5 \pm 1 . 0$ </td></tr><tr><td>144</td><td>200</td><td> $1 7 4 . 7 \pm 0 . 6$ </td><td> $1 7 0 . 2 \pm 0 . 6$ </td><td> $1 7 5 . 4 \pm 0 . 6$ </td></tr></table>

Across the displayed horizons, the curves have a broad interior minimum near ${ \sqrt { L } } ,$ , with larger errors at both extremes. The preferred range moves toward larger patches when the window grows, consistent with the dimension balance in (24). However, neighboring patches remain competitive: at the longest horizon, a larger neighboring patch can match or slightly improve on the balanced reference, without a statistically resolved diference. The evidence therefore favors a range of patch lengths rather than a unique optimum.

Fitted excess error also increases across these horizons. A more distant target does not automatically make the one-head forecasting problem easier, even as its predictable signal becomes smaller. Appendix J.4 reports the population diagnostic for the persistent relative filtering error predicted by Theorem 4. The fitted curves additionally reflect finite data and the training procedure.

## 6.3 Training Size and Interpretation

Figure 2 and Table 1 show that interior patches have their largest advantage when training data are scarce. As more independent series become available, prediction error falls throughout the grid and the absolute gap between interior and extreme patch lengths narrows. This pattern appears at both window lengths. The preferred interior patch also varies across the displayed horizons, so the balanced reference need not minimize every fitted curve.

The results are consistent with the estimation term in (27) and with the extreme-versus-intermediate comparison in Section 4. Here the approximation error of interior patches is negligible (Appendix J.3), so the

![](images/ae76ba3207b3267aae7b4aff2be2a1b4671932d1a3271ba20a617336587909d4.jpg)

Figure 1: Excess MSE across forecast horizons for fixed one-head attention, with $n = 2 0 0$ independent training series. Curves show means over 50 data repetitions. Shaded bands are pointwise 95% bootstrap intervals. Dashed lines mark $p = { \sqrt { L } }$ . Patch candidates are displayed at equally spaced categorical positions, and panel scales difer.  
![](images/87bf32d5e416a2fb8b585f98508e070f63e8db41c72d8feb5f81ed7e5790fad5.jpg)  
Figure 2: Total test MSE at H = 1 for fixed one-head attention as the number of independent training series grows. Means and pointwise 95% bootstrap intervals use 50 repetitions. Other plotting conventions match Figure 1.

## disadvantage of the extremes reflects estimation.

The experiment nevertheless does not identify class dimension as the sole explanation: network parameter count is minimized at the same patch length in this architecture, and regularization and optimization can afect the fitted curves. Taken together, the horizon and sample-size comparisons support testing an interior range around ${ \sqrt { L } } ,$ with validation used to choose among nearby candidates. Two window lengths alone cannot establish a square-root scaling law, and $\mathrm { A p \mathrm { - } }$ pendix J.7 examines nine.

## 7 CONCLUSION

This paper makes three contributions. First, we show that multi-head patch attention with shared linear values forecasts through a coeficient matrix of rank at most the head count, even with input-dependent weights; for Gaussian data, no decoder on a widthm shared linear patch encoding can do better than the best rank-m matrix. Second, for linear statespace models, an exact factorization shows that a dominant long-horizon component need not be rank-one representable. Third, with dependent-data estimation bounds, we derive a patch-selection criterion that balances representation and estimation errors and identifies when balanced patches are justified. Future work includes learned content-dependent attention.

## References

Agarwal, A., Amjad, M. J., Shah, D., and Shen, D. (2018). Model agnostic time series analysis via matrix estimation. Proceedings of the ACM on Measurement and Analysis of Computing Systems, 2(3):40:1–40:39.

Ankireddy, S. K., Seleznev, N., Nguyen, N. H., Wu, Y., Kumar, S., Huang, F., and Bruss, C. B. (2026). TimeSqueeze: Dynamic patching for eficient time series forecasting.

Basu, S. and Michailidis, G. (2015). Regularized estimation in sparse high-dimensional time series models. The Annals of Statistics, 43(4):1535–1567.

Cand\`es, E. J. and Plan, Y. (2011). Tight oracle inequalities for low-rank matrix recovery from a minimal number of noisy random measurements. IEEE Transactions on Information Theory, 57(4):2342– 2359.

Chen, P., Zhang, Y., Cheng, Y., Shu, Y., Wang, Y., Wen, Q., Yang, B., and Guo, C. (2024). Pathformer: Multi-scale transformers with adaptive pathways for time series forecasting. In International Conference on Learning Representations.

Chen, Y., C´espedes, N., and Barnaghi, P. (2025). A closer look at transformers for time series forecasting: Understanding why they work and where they struggle. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 7763–7780.

Damen, A. A. H., van den Hof, P. M. J., and Hajdasinski, A. K. (1982). Approximate realization based upon an alternative to the Hankel matrix: The Page matrix. Systems & Control Letters, 2(4):202–208.

Das, A., Kong, W., Sen, R., and Zhou, Y. (2024). A decoder-only foundation model for time-series forecasting. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 10148–10167.

Glover, K. (1984). All optimal Hankel-norm approximations of linear multivariable systems and their $L _ { \infty }$ -error bounds. International Journal of Control, 39(6):1115–1193.

Goel, G. and Bartlett, P. (2024). Can a transformer represent a Kalman filter? In Proceedings of the 6th Annual Learning for Dynamics & Control Conference, volume 242 of Proceedings of Machine Learning Research, pages 1502–1512.

Hazan, E., Singh, K., and Zhang, C. (2017). Learning linear dynamical systems via spectral filtering.

In Advances in Neural Information Processing Systems, volume 30.

Izenman, A. J. (1975). Reduced-rank regression for the multivariate linear model. Journal of Multivariate Analysis, 5(2):248–264.

Krahmer, F., Mendelson, S., and Rauhut, H. (2014). Suprema of chaos processes and the restricted isometry property. Communications on Pure and Applied Mathematics, 67(11):1877–1904.

Lu, J. and Yang, S. (2025). Linear transformers as VAR models: Aligning autoregressive attention mechanisms with autoregressive forecasting. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 40848–40867.

Nie, Y., Nguyen, N. H., Sinthong, P., and Kalagnanam, J. (2023). A time series is worth 64 words: Long-term forecasting with transformers. In International Conference on Learning Representations.

Rashidinejad, P., Jiao, J., and Russell, S. J. (2020). SLIP: Learning to predict in unknown dynamical systems with long-term memory. In Advances in Neural Information Processing Systems, volume 33.

Rudelson, M. and Vershynin, R. (2013). Hanson– Wright inequality and sub-Gaussian concentration. Electronic Communications in Probability, 18(82):1–9.

Sander, M. E., Giryes, R., Suzuki, T., Blondel, M., and Peyr´e, G. (2024). How do transformers perform in-context autoregressive learning? In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 43235–43254.

Velu, R. P. and Reinsel, G. C. (1987). Reduced rank regression with autoregressive errors. Journal of Econometrics, 35(2–3):317–335.

Zhang, Y., Ma, L., Pal, S., Zhang, Y., and Coates, M. (2024). Multi-resolution time-series transformer for long-term forecasting. In Proceedings of the 27th International Conference on Artificial Intelligence and Statistics, volume 238 of Proceedings of Machine Learning Research, pages 4222–4230.

Zucchi, F., Xie, Y., Zhang, C., Luo, K., Lampert, T., and Li, Z. (2026). Adaptive patching is harder than it looks for time-series forecasting.

# Patch Attention Wisely: Understanding Patch Length in Attention-Based Forecasting Supplementary Materials

## A GUIDE AND NOTATION

The appendix follows the argument of the main paper: attention representation, state-space approximation, finite-window prediction, estimation from dependent data, and patch selection. Each result is followed by its proof. The remaining sections treat context adaptation, practical implications, and the simulations.

• Appendix B proves the coeficient-matrix identity and gives explicit fixed positional attention constructions for Section 2.

• Appendix C proves the filtering-mode factorization, the long-horizon floor, and the shared-horizon extension from Section 3.

• Appendix D relates the truncated infinite-past coeficients to the optimal finite-window forecast.

• Appendix E derives the dependent-data rate and proves Theorem 6.

• Appendix F states a conditional selection criterion, proves the patch-selection results, and collects the dimension and computational-cost calculations.

• Appendix G studies adaptation across generating systems.

• Appendix H discusses when context dependence can help and how patch lengths should be compared in practice.

• Appendix I gives the simulation diagnostics and evaluation protocol.

• Appendix J reports additional experiments: a second observation design, the population filtering floor, head counts for rank-one and rank-three targets, and nine window lengths.

## A.1 Notation and types of results

We keep the notation from the main paper. The context contains L observations, each patch contains p observations, and $N = L / p$ is the number of patches. The forecast horizon is H, and h denotes a rank or head budget. The main paper uses m for the fixed head budget. The matrix $B _ { H , p }$ is obtained by truncating the infinite-past forecast to the latest L observations and reshaping its coeficients into N rows and p columns. The matrix $B _ { H , L , p } ^ { \circ }$ instead denotes the best coeficient matrix when the forecaster is allowed to use only those L observations. These matrices are close for a long stable context, but they need not be identical.

For a coeficient error D, the prediction norm $\| D \| _ { \Gamma _ { L , p } } ^ { 2 }$ is the mean squared change in the forecast caused by D. It is the relevant error measure because it accounts for correlation among lagged observations. When ordinary Frobenius or operator norms are used instead, the text says so explicitly. All additional symbols are introduced in the section where they are needed.

The supplement distinguishes three kinds of statements. Representation results describe which fixed forecast coeficients an attention model can express. Population results describe the best predictor when the datagenerating system is known. Estimation results describe what can be learned from a finite, dependent trajectory. Keeping these levels separate is important: a low-rank coeficient matrix does not by itself guarantee that a trained attention model will find it.

## B ATTENTION REPRESENTATION

This section establishes the representation claim used in Section 2. The coeficient-matrix identity gives the rank upper bound. The fixed positional construction gives the converse. The subsequent extensions treat shared horizons, signed softmax pooling, approximation, and input-dependent attention weights.

## B.1 Coeficient-matrix identity

Let $Z \in \mathbb { R } ^ { N \times p }$ contain the raw patches as rows. For head a, write its score for patch $j$ as

$$
\alpha _ { H , a , j } ( Z ) = N ^ { - 1 } k _ { j , a } ( Z ) ^ { \top } q _ { H , a } ( Z ) .\tag{31}
$$

Unlike softmax weights, these scores may be negative and need not sum to one. The factor $1 / N$ is only a scaling convention. When models with diferent token counts are compared, the convention should be kept fixed. The shared linear value path reduces to a vector $b _ { a } \in \mathbb { R } ^ { p }$ that tells the head how to weight observations within every patch. Thus head a contributes $\alpha _ { H , a } ( Z ) b _ { a } ^ { \top }$ , an outer product. Define the resulting coeficient matrix and forecast by

$$
B _ { H } ( Z ) = \sum _ { a } \alpha _ { H , a } ( Z ) b _ { a } ^ { \top } , \qquad \widehat { f } _ { H } ( Z ) = \langle B _ { H } ( Z ) , Z \rangle _ { \mathrm { \tiny { F } } } .\tag{32}
$$

Proof of Proposition 1. With $b _ { a } = W _ { V , a } ^ { \top } w _ { a } ,$ expand the forecast:

$$
\widehat { f } _ { H } ( Z ) = \sum _ { a , j } \alpha _ { H , a , j } ( Z ) b _ { a } ^ { \top } x _ { j } = \left. \sum _ { a } \alpha _ { H , a } ( Z ) b _ { a } ^ { \top } , Z \right. _ { \mathrm { \tiny ~ F } } .\tag{33}
$$

Each summand is an $N \times p$ outer product, so its sum has rank at most min $\{ m , N , p \}$ . Vector-valued values do not change this conclusion for this one scalar linear output, because $W _ { V , a }$ and $w _ { a }$ are fixed and shared across positions. Nonlinear values, query-dependent output maps, and other paths need separate treatment. No converse head lower bound for adaptive functions follows from this pointwise identity. □

## B.2 Exact fixed positional pooling

Let a context of length $L = N p$ be divided into nonoverlapping patches $x _ { j } \in \mathbb { R } ^ { p }$ , and stack $\boldsymbol { x } _ { j } ^ { \top }$ as the rows of $ { \boldsymbol { X } } \in \mathbb { R } ^ { N \times p }$ . For $\boldsymbol { B } \in \mathbb { R } ^ { N \times p }$ , define

$$
f _ { B } ( \boldsymbol { X } ) = \langle \boldsymbol { B } , \boldsymbol { X } \rangle _ { \mathrm { F } } = \sum _ { j = 1 } ^ { N } B _ { j , : } x _ { j } .\tag{34}
$$

Use N data tokens and one fixed forecast-query token,

$$
z _ { j } = { \binom { x _ { j } } { e _ { j } } } , \qquad z _ { \star } = { \binom { 0 _ { p } } { 0 _ { N } } } , \qquad D = p + N + 1 ,\tag{35}
$$

where $e _ { j }$ is the jth absolute positional basis vector. A permutation-symmetric hard mask admits all N data positions and excludes the forecast token itself. This is equivalent to assigning the forecast token $\mathrm { a - } \infty$ softmax logit. A finite mask penalty changes the softmax normalizer and is outside the exact result.

For one scalar unnormalized pooling head, put

$$
o _ { a } ( X ) = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } ( W _ { Q } ^ { ( a ) } z _ { \star } ) ( W _ { K } ^ { ( a ) } z _ { j } ) ( W _ { V } ^ { ( a ) } z _ { j } ) .\tag{36}
$$

For m heads, stack $o = \left( o _ { 1 } , \ldots , o _ { m } \right) ^ { \top }$ . A standard multi-head output projection and scalar output layer have the form

$$
y _ { \star } = W _ { O } o , \qquad { \widehat { y } } = r ^ { \top } y _ { \star } , \qquad W _ { O } \in \mathbb { R } ^ { D \times m } , \quad r \in \mathbb { R } ^ { D } .\tag{37}
$$

Any desired head-mixing vector $c \in \mathbb { R } ^ { m }$ is implemented by $W _ { O } = e _ { 1 } c ^ { \top }$ and $r = e _ { 1 }$ , giving $\widehat { \boldsymbol { y } } = \boldsymbol { c } ^ { \intercal } \boldsymbol { o }$

Theorem 9 (Exact fixed positional linear pooling). Suppose rank $( B ) \leq h$ and choose

$$
B = \sum _ { a = 1 } ^ { h } u _ { a } v _ { a } ^ { \top } , \qquad u _ { a } \in \mathbb { R } ^ { N } , \quad v _ { a } \in \mathbb { R } ^ { p } .\tag{38}
$$

There is a one-layer fixed positional pooling construction with $N + 1$ tokens, width $D = p + N + 1$ , h heads, scalar query/key/value dimensions, a hard mask, standard $W _ { O }$ , and a scalar output layer that equals $f _ { B } ( { \cal { X } } )$ for every X. Head a uses

$$
\begin{array} { r } { W _ { Q } ^ { ( a ) } z _ { \star } = 1 , \qquad W _ { K } ^ { ( a ) } z _ { j } = N u _ { a , j } , \qquad W _ { V } ^ { ( a ) } z _ { j } = v _ { a } ^ { \top } x _ { j } , } \end{array}\tag{39}
$$

and (37) uses $c = \mathbf { 1 } _ { h }$

Proof. Head a returns $\begin{array} { r } { o _ { a } ( X ) = N ^ { - 1 } \sum _ { j } ( N u _ { a , j } ) v _ { a } ^ { \top } x _ { j } = \langle u _ { a } v _ { a } ^ { \top } , X \rangle _ { \mathrm { F } } } \end{array}$ . The positional key factor N exactly cancels the fixed token-count factor. Summing the heads and using (38) gives $f _ { B } ( { \cal { X } } )$ □

Remark 10 (Positional parameters). The explicit width uses N one-hot positional coordinates. A B-specific learned positional table may instead store $( u _ { 1 , j } , \ldots , u _ { h , j } ) \in \mathbb { R } ^ { h }$ , reducing width to $p + h + 1$ while retaining Nh positional parameters. The smaller width does not eliminate the order information needed to store a general $N \times h$ factor.

Proposition 11 (Restricted-architecture head lower bound). Consider one hard-masked layer with a fixed forecast query, scalar values that are fixed linear functionals of $x _ { j }$ , content-independent scalar keys that depend only on position, standard $W _ { O }$ plus a scalar linear output layer, and no residual or feed-forward bypass. Every such m-head unnormalized pooling layer represents a coeficient matrix of rank at most m. Thus $m \geq \mathrm { r a n k } ( B )$ is necessary within this full restricted architecture.

Proof. After absorbing the scalar query and output coeficient, head a contributes some rank-one matrix $u _ { a } v _ { a } ^ { \top }$ The sum of m such matrices has rank at most m. This fixed-operator lower bound does not extend to arbitrary content-dependent functions, nonlinear values, depth, or another input-to-output path. Vector-valued linear values alone do not remove the rank bound under the shared scalar output weights of Proposition 1. □

## B.3 Shared pooling across finitely many horizons

Corollary 12 (Finite shared multi-horizon pooling). Let $B ^ { ( t ) } \in \mathbb { R } ^ { N \times p } ~ f o r ~ t = 1 , . . . , T$ satisfy

$$
B ^ { ( t ) } = \sum _ { a = 1 } ^ { h } d _ { t , a } u _ { a } v _ { a } ^ { \top } .\tag{40}
$$

Use N data tokens and T one-hot horizon-query tokens, all of width $D = p + N + T$ . One hard-masked layer with h scalar heads and shared key/value projections outputs $f _ { B ^ { ( t ) } } ( \boldsymbol { X } )$ at query t by setting

$$
W _ { Q } ^ { ( a ) } z _ { \star } ^ { ( t ) } = d _ { t , a } , \qquad W _ { K } ^ { ( a ) } z _ { j } = N u _ { a , j } , \qquad W _ { V } ^ { ( a ) } z _ { j } = v _ { a } ^ { \top } x _ { j } .\tag{41}
$$

The construction uses $N + T$ tokens and explicitly stores a $T \times h$ table $( d _ { t , a } )$

Proof. The mask admits exactly the N data tokens. At horizon query t, head a outputs $\begin{array} { r } { N ^ { - 1 } \sum _ { j } d _ { t , a } ( N u _ { a , j } ) v _ { a } ^ { \top } x _ { j } } \end{array}$ Summing over heads gives $\langle B ^ { ( t ) } , X \rangle _ { \mathrm { F } }$ . The factor $1 / N$ counts the admitted data tokens, not the $N + T$ total tokens. □

Remark 13 (No unseen-horizon extrapolation). The corollary is transductive over the displayed T horizons. It provides no coeficient for an unseen horizon. Such an extension requires a specified feature map $\phi ( H )$ and a structural rule, for example $d _ { H , a } = \theta _ { a } ^ { \top } \phi ( H )$

## B.4 Signed softmax realization

Theorem 14 (Exact signed fixed positional softmax pooling). Under (38), fix $\epsilon > 0$ . Let $u _ { a } ^ { + } = ( u _ { a } ) _ { + } , u _ { a } ^ { - } =$ $( - u _ { a } ) _ { + }$ <sub>+</sub>, and

$$
s _ { a } ^ { \pm } = \| u _ { a } ^ { \pm } \| _ { 1 } , \qquad c _ { a } ^ { \pm } = s _ { a } ^ { \pm } + N \epsilon , \qquad \ell _ { a , j } ^ { \pm } = \log ( u _ { a , j } ^ { \pm } + \epsilon ) .\tag{42}
$$

A hard-masked fixed positional softmax-pooling layer with $N { + 1 }$ tokens, width $p { + } N { + } 1$ , 2h scalar heads, standard $W _ { O }$ , and a scalar output layer represents $f _ { B }$ exactly. The $( a , \pm )$ head uses query 1, positional logit $\ell _ { a , j } ^ { \pm }$ , and value $v _ { a } ^ { \top } x _ { j }$ . The head-mixing vector in (37) is $( c _ { 1 } ^ { + } , - c _ { 1 } ^ { - } , \ldots , c _ { h } ^ { + } , - c _ { h } ^ { - } )$ .

Proof. The $( a , \pm )$ attention probability is

$$
\pi _ { a , j } ^ { \pm } = \frac { u _ { a , j } ^ { \pm } + \epsilon } { s _ { a } ^ { \pm } + N \epsilon } .\tag{43}
$$

The signed output for pair a is therefore

$$
c _ { a } ^ { + } \sum _ { j } \pi _ { a , j } ^ { + } v _ { a } ^ { \top } x _ { j } - c _ { a } ^ { - } \sum _ { j } \pi _ { a , j } ^ { - } v _ { a } ^ { \top } x _ { j } = \sum _ { j } u _ { a , j } v _ { a } ^ { \top } x _ { j } ,\tag{44}
$$

because the two ϵ ofsets cancel. Summing over a proves the claim.

Remark 15 (Architecture and conditioning restrictions). The theorem fixes one query, a permutation-symmetric hard mask, absolute positional keys, scalar patch-linear values, tied values within each sign pair, one layer, no bypass, and signed head mixing through $W _ { O }$ and a scalar output layer. Moreover, its numerical conditioning depends on the chosen factorization and ϵ: rescaling $u _ { a } \mapsto \alpha _ { a } u _ { a }$ and $v _ { a } \mapsto v _ { a } / \alpha _ { a }$ leaves B fixed but changes logits, output weights, and perturbation sensitivity.

## B.5 Approximation and perturbation bounds

Let $B _ { h }$ be a best rank-h Frobenius approximation to $B .$

Theorem 16 (Approximation and covariance-weighted prediction error). Either exact construction applied to $B _ { h }$ gives a fixed positional pooling output $f _ { B _ { h } } ( \boldsymbol { X } )$ such that

$$
\begin{array} { r l } & { | f _ { B } ( \boldsymbol { X } ) - f _ { B _ { h } } ( \boldsymbol { X } ) | \le \| B - B _ { h } \| _ { \mathrm { F } } \| \boldsymbol { X } \| _ { \mathrm { F } } , } \\ & { \qquad \| B - B _ { h } \| _ { \mathrm { F } } ^ { 2 } = \displaystyle \sum _ { k > h } \sigma _ { k } ( B ) ^ { 2 } . } \end{array}\tag{45}
$$

$I f \Gamma = \mathbb { E } [ \mathrm { v e c } ( X ) \mathrm { v e c } ( X ) ^ { \top } ]$ , then

$$
{ \mathbb E } \{ f _ { B } ( X ) - f _ { B _ { h } } ( X ) \} ^ { 2 } \leq \lambda _ { \operatorname* { m a x } } ( \Gamma ) \sum _ { k > h } \sigma _ { k } ( B ) ^ { 2 } .\tag{46}
$$

Proof. The first statement is Cauchy–Schwarz and Eckart–Young–Mirsky. Vectorizing the Frobenius inner product gives the covariance-weighted statement. □

Lemma 17 (Softmax perturbation). For $r , s \in \mathbb { R } ^ { N }$ , ∥softmax(r) − softmax $( s ) \| _ { 1 } \leq 2 \| r - s \| _ { \infty }$ . Consequently, if every signed logit in Theorem $1 \mathit { 4 }$ is approximated within δ in $\ell _ { \infty } ,$ , the induced coeficient matrix $\widetilde { B } _ { h }$ obeys

$$
\| \widetilde { B } _ { h } - B _ { h } \| _ { \mathrm { F } } \leq 2 \delta \sum _ { a = 1 } ^ { h } ( c _ { a } ^ { + } + c _ { a } ^ { - } ) \| v _ { a } \| _ { 2 } .\tag{47}
$$

Proof. Along the segment between r and s, the softmax Jacobian bounds the $\ell _ { \infty } { - } \mathrm { t o } { - } \ell _ { 1 }$ derivative norm by 2. Each head coeficient perturbation is an outer product of the changed probability vector and $v _ { a }$ . The triangle inequality over the 2h heads proves (47). □

## B.6 The role of positional information

Proposition 18 (Order information under permutation-symmetric pooling). Fix a forecast query and a mask that treats all admitted data positions symmetrically. If every token is formed by the same map $z _ { j } = g ( x _ { j } )$ with no absolute, relative, causal-distance, convolutional, or other order-dependent information, the pooling output is permutation invariant. A linear f is permutation invariant for every X only if all rows ofB are equal. Therefore a general coeficient matrix needs some symmetry-breaking order information under these assumptions.

Proof. The pooling operation depends only on the multiset of key–value pairs. Invariance of $f _ { B }$ under arbitrary row permutations forces every pair of rows of B to agree. The proposition does not privilege absolute one-hot positions and does not apply to asymmetric masks or content maps that already encode order. □

Relation to existing attention theory. Goel and Bartlett study causal softmax Transformers that approximate Kalman filtering (Goel and Bartlett, 2024), Sander et al. study autoregressive learning in context (Sander et al., 2024), and Lu and Yang connect linear attention to dynamic VAR computations (Lu and Yang, 2025). Here the object is a fixed finite-window coeficient matrix obtained by a nonoverlapping reshape. Page-matrix and spectral-LDS methods provide related representations of time-series predictors (Damen et al., 1982; Rashidinejad et al., 2020). The constructions above specify how the positional tokens, mask, query/key/value maps, and scalar output layer realize that matrix. They do not provide recursive filtering or optimization guarantees.

## B.7 Input-dependent attention weights and the full Jacobian

All learned parameters are held fixed when taking derivatives or population expectations. The following results distinguish the pointwise coeficient matrix from the derivative of the complete forecast.

Proposition 19 (Attention weights and the full input Jacobian). At diferentiability points of $B _ { H }$ , the Frobenius gradient is

$$
J _ { H } ( Z ) = \nabla _ { Z } \widehat { f } _ { H } ( Z ) = B _ { H } ( Z ) + \sum _ { a , j } ( b _ { a } ^ { \top } x _ { j } ) \nabla _ { Z } \alpha _ { H , a , j } ( Z ) .\tag{48}
$$

Only the first term is the derivative obtained by freezing the attention weights at the current input. The second need not vanish or preserve the rank bound.

Proof. Apply the product rule to each term $\alpha _ { H , a , j } ( Z ) b _ { a } ^ { \top } x _ { j }$ and sum.

Why the Jacobian rank can be larger. To see the distinction algebraically, use one-based indices, let one head have $b = ( 1 , 0 ) ^ { \top }$ , take scalar keys $k _ { j } = x _ { j , 2 }$ , and fix the scalar query. Then

$$
\alpha _ { j } = N ^ { - 1 } x _ { j , 2 } , \qquad f ( Z ) = N ^ { - 1 } \sum _ { j = 1 } ^ { N } x _ { j , 1 } x _ { j , 2 } .\tag{49}
$$

The coeficient matrix has only its first column nonzero and therefore has rank at most one. By contrast, the first two columns of the input Jacobian are $N ^ { - 1 } Z [ : , 2 ]$ and $N ^ { - 1 } Z [ : , 1 ]$ . They are generically linearly independent, so the Jacobian can have rank two. Thus content-dependent keys can increase sensitivity rank even with a fixed query and a rank-one coeficient matrix.

Three diferent matrices. The unmasked token mixing matrix $N ^ { - 1 } \mathbf { Q } \mathbf { K } ^ { \top }$ has shape $N \times N$ and rank at most min $( N , d _ { k } )$ . It is diferent from the $N \times p$ coeficient matrix $B _ { H } ( Z )$ and the $N \times p$ input gradient $J _ { H } ( Z )$ ). A triangular causal mask can invalidate the token-matrix rank bound. None of these statements bounds the number of trainable parameters in an arbitrary context encoder.

## B.8 Comparison of adaptive attention weights with a fixed benchmark

Proposition 20 (Comparison with a fixed spectral benchmark). Let $B _ { 0 }$ be deterministic, let $B _ { 0 , h }$ be its rank-h truncated SVD, and let $\Gamma = \mathbb { E } [ \mathrm { v e c } ( Z ) \mathrm { v e c } ( Z ) ^ { \top } ]$ . For the payload-linear adaptive model, when the right side is

finite,

$$
\begin{array} { r l } & { \displaystyle \mathbb { E } | \widehat { f } _ { H } ( Z ) - \langle B _ { 0 } , Z \rangle _ { \mathrm { F } } | ^ { 2 } \leq 2 \lambda _ { \operatorname* { m a x } } ( \Gamma ) \sum _ { k > h } \sigma _ { k } ( B _ { 0 } ) ^ { 2 } } \\ & { \qquad + 2 \mathbb { E } \{ \| B _ { H } ( Z ) - B _ { 0 , h } \| _ { \mathrm { F } } ^ { 2 } \| Z \| _ { \mathrm { F } } ^ { 2 } \} . } \end{array}\tag{50}
$$

Proof. Decompose the prediction error into $\langle B _ { H } ( Z ) - B _ { 0 , h } , Z \rangle _ { \mathrm { F } } + \langle B _ { 0 , h } - B _ { 0 } , Z \rangle _ { \mathrm { F } }$ . Use $( u + v ) ^ { 2 } \leq 2 u ^ { 2 } + 2 v ^ { 2 }$ 2 pointwise Cauchy–Schwarz for the first term, and the deterministic covariance bound for the second. □

The second term contains representation and fitting error. The singular tail alone cannot control it. Since $B _ { H } ( Z )$ depends on $Z ,$ , replacing this term by an unconditional covariance eigenvalue times its expected squared matrix norm is unjustified without an additional condition. Taking $B _ { 0 } = B _ { H , L , p } ^ { \circ }$ compares to the finite-window oracle. Taking $B _ { 0 } = B _ { H , p }$ compares to the truncated infinite-past kernel and still requires the finite-window bridge. This proposition supplies no sample-size rate for learned attention.

## B.9 Nonlinear decoding after shared patch encoding

Proof of Proposition $\delta .$ For fixed $E ,$ Gaussian conditioning implies that the optimal decoder is linear: $\mathbb { E } [ X _ { t + H } \ ]$ $Z _ { t , p } E ] = \langle D , Z _ { t , p } E \rangle _ { \mathrm { F } }$ for some $D \in \mathbb { R } ^ { N \times m }$ . Since

$$
\langle D , Z _ { t , p } E \rangle _ { \mathrm { F } } = \langle D E ^ { \top } , Z _ { t , p } \rangle _ { \mathrm { F } } ,
$$

and the products $D E ^ { \top }$ range over all matrices of rank at most $m ,$ the projection identity (9) yields

$$
\operatorname* { i n f } _ { E , g } \mathbb { E } \{ X _ { t + H } - g ( Z _ { t , p } E ) \} ^ { 2 } = R _ { 0 } + \operatorname* { i n f } _ { \operatorname { r a n k } ( B ) \leq m } \| B - B _ { p } ^ { \circ } \| _ { \Gamma } ^ { 2 } .
$$

The last infimum is $A _ { m } ( p )$

## C STATE-SPACE FILTERING AND APPROXIMATION

The state-space calculation separates two jobs. The transition matrix A propagates the current state toward the forecast horizon, while the filtering matrix F describes how past observations reveal that state. The proofs below show that patching does not create new filtering modes: it only splits each lag pattern into an across-patch part and a within-patch part.

## C.1 Filtering-mode factorization

Proof of Theorem 3. The proof has one idea: each exponential lag pattern splits into an across-patch factor and a within-patch factor because $z ^ { j p + s } = ( z ^ { p } ) ^ { j } z ^ { s }$ . The forecast horizon changes the coeficient of that pattern, not its shape.

By spectral decomposition,

$$
A ^ { H - 1 } = \sum _ { a = 1 } ^ { r } \lambda _ { a } ^ { H - 1 } P _ { a } , \qquad F ^ { \ell } = \sum _ { k = 1 } ^ { r } \varphi _ { k } ^ { \ell } Q _ { k } .\tag{51}
$$

Substituting into (13) gives

$$
\beta _ { H } ( \ell ) = \sum _ { a = 1 } ^ { r } \sum _ { k = 1 } ^ { r } ( C P _ { a } Q _ { k } K ) \lambda _ { a } ^ { H - 1 } \varphi _ { k } ^ { \ell } = \sum _ { k = 1 } ^ { r } d _ { k } ( H ) \varphi _ { k } ^ { \ell } .\tag{52}
$$

With $\ell = j p + s ,$

$$
\beta _ { H } ( j p + s ) = \sum _ { k = 1 } ^ { r } d _ { k } ( H ) ( \varphi _ { k } ^ { p } ) ^ { j } \varphi _ { k } ^ { s } ,\tag{53}
$$

which is exactly (17). For nonreal modes this identity is over $\mathbb { C } ,$ with conjugate terms summing to the real matrix $B _ { H , p }$ . Each summand is rank one over $\mathbb { C } ,$ which proves the stated rank bound. Because $B _ { H , p }$ is real, its real and complex matrix ranks agree. □

## C.2 Long-horizon filtering floor

Proof of Theorem 4. Since A is real, nonreal eigenvalues occur in conjugate pairs with equal modulus. The uniquely dominant $\lambda _ { 1 }$ is therefore real. Consequently $B _ { \infty , p }$ and the rescaled matrices below are real, even when individual filtering factors are complex.

Theorem 3 gives

$$
\lambda _ { 1 } ^ { - ( H - 1 ) } B _ { H , p } - B _ { \infty , p } = \sum _ { a = 2 } ^ { r } \sum _ { k = 1 } ^ { r } g _ { a k } \left( \frac { \lambda _ { a } } { \lambda _ { 1 } } \right) ^ { H - 1 } u _ { N , p } ( \varphi _ { k } ) v _ { p } ( \varphi _ { k } ) ^ { \top } .\tag{54}
$$

The triangle inequality and $\| u v ^ { \top } \| _ { \mathrm { o p } } = \| u \| _ { 2 } \| v \| _ { 2 }$ prove (19). Weyl’s inequality implies

$$
\left| \frac { \sigma _ { j } ( B _ { H , p } ) } { | \lambda _ { 1 } | ^ { H - 1 } } - \sigma _ { j } ( B _ { \infty , p } ) \right| \le C _ { p } q _ { A } ^ { H - 1 } .\tag{55}
$$

Let $e _ { H } = C _ { p } q _ { A } ^ { H - 1 } , s _ { j } = \sigma _ { j } ( B _ { \infty , p } )$ , and $\widetilde s _ { j } = \sigma _ { j } ( \lambda _ { 1 } ^ { - ( H - 1 ) } B _ { H , p } )$ . If $e _ { H } \leq s _ { 1 } / 2$ , then $\widetilde { s } _ { 1 } \geq s _ { 1 } / 2$ and

$$
\left| \frac { \widetilde s _ { h + 1 } } { \widetilde s _ { 1 } } - \frac { s _ { h + 1 } } { s _ { 1 } } \right| \le \frac { e _ { H } } { \widetilde s _ { 1 } } + \frac { s _ { h + 1 } e _ { H } } { s _ { 1 } \widetilde s _ { 1 } } \le \frac { 4 e _ { H } } { s _ { 1 } } .\tag{56}
$$

By Eckart–Young, $R _ { H , p , h } ^ { \mathrm { o p } }$ is the relative best rank-h operator-norm error, and (20) follows. If $h < \mathrm { r a n k } ( B _ { \infty , p } ) ;$ then $s _ { h + 1 } > 0$ 口

## C.3 Active modes and patch aliasing

Proof of Corollary 5. Let $S = \{ k : g _ { 1 k } \neq 0 \}$ be the filtering modes coupled to the dominant forecast component. Restrict the two modal factor matrices to the active set S, and write

$$
B _ { \infty , p } = U _ { S } \mathrm { d i a g } \{ g _ { 1 k } : k \in S \} V _ { S } ^ { \top } .\tag{57}
$$

The columns of $V _ { S }$ form a Vandermonde matrix with nodes $\varphi _ { k } .$ , and the columns of $U _ { S }$ form a Vandermonde matrix with nodes $\varphi _ { k } ^ { p } .$ . The distinct-node assumptions and $| S | \le \operatorname* { m i n } \{ N , p \}$ make both matrices full column rank. The diagonal middle factor is nonsingular, so Sylvester’s rank inequality, applied twice, gives rank $( B _ { \infty , p } ) = | S |$ The limiting rank-h operator error is $\sigma _ { h + 1 } ( B _ { \infty , p } ) / \sigma _ { 1 } ( B _ { \infty , p } ) > 0$ whenever $h < | S |$ □

## C.4 Absolute approximation risk

Write $\widetilde { B } _ { H , p } = \lambda _ { 1 } ^ { - ( H - 1 ) } B _ { H , p }$ and assume $\Gamma _ { L , p } \preceq M _ { \Gamma } I$ . The operator-norm bound from Theorem 4 then implies

$$
\begin{array} { r } { \| \widetilde { B } _ { H , p } - B _ { \infty , p } \| _ { \Gamma _ { L , p } } \le \delta _ { H , p } , \qquad \delta _ { H , p } = \sqrt { M _ { \Gamma } \operatorname* { m i n } \{ N , p \} } C _ { p } q _ { A } ^ { H - 1 } . } \end{array}\tag{58}
$$

The rest of the argument converts this matrix bound into a bound on the best rank-h approximation error.

Let $\mathcal { R } _ { h }$ be the closed cone of matrices of rank at most h, and let $\begin{array} { r } { d _ { \Gamma } ( B , \mathcal { R } _ { h } ) = \operatorname* { i n f } _ { M \in \mathcal { R } _ { h } } \| B - M \| _ { \Gamma _ { L , p } } } \end{array}$ . Distance to any nonempty set is 1-Lipschitz in the ambient seminorm, hence

$$
\begin{array} { r } { \left| d _ { \Gamma } ( \widetilde { B } _ { H , p } , \mathcal { R } _ { h } ) - d _ { \Gamma } ( B _ { \infty , p } , \mathcal { R } _ { h } ) \right| \leq \| \widetilde { B } _ { H , p } - B _ { \infty , p } \| _ { \Gamma _ { L , p } } . } \end{array}\tag{59}
$$

The covariance upper bound, the inequality $\| D \| _ { \mathrm { F } } \le \sqrt { \operatorname* { m i n } \{ N , p \} } \| D \| _ { \mathrm { o p } } ,$ and Theorem 4 bound the right-hand side by $\delta _ { H , p }$ . This proves the root-risk bound. Since $| a ^ { \dot { 2 } } - b ^ { 2 } | ^ { \cdot } = | a - b | ( a + b )$ and $a \leq b + | a - b |$ , the squared-risk bound follows. Finally, $\mathcal { R } _ { h }$ is a cone, so for every nonzero scalar c, $, d _ { \Gamma } ( c B , \mathcal { R } _ { h } ) = | c | d _ { \Gamma } ( B , \mathcal { R } _ { h } )$ . Taking $c = \lambda _ { 1 } ^ { H - 1 }$ proves the displayed absolute expansion.

## C.5 Two-mode closed form

For $r \ = \ 2$ , write $B _ { H , p } ~ = ~ U _ { p } D _ { H } V _ { p } ^ { \top }$ , where $U _ { p } ~ = ~ [ u _ { N , p } ( \varphi _ { 1 } ) , u _ { N , p } ( \varphi _ { 2 } ) ] , ~ V _ { p } ~ = ~ [ v _ { p } ( \varphi _ { 1 } ) , v _ { p } ( \varphi _ { 2 } ) ]$ , and $D _ { H } \ =$ diag $; ( d _ { 1 } ( H ) , d _ { 2 } ( H ) )$ . Let $G _ { U } = U _ { p } ^ { * } \hat { U } _ { p }$ and $G _ { V } \stackrel { \cdot } { = } V _ { p } ^ { * } V _ { p }$ . The nonzero squared singular values are the eigenvalues of

$$
G _ { U } ^ { 1 / 2 } D _ { H } \overline { { G _ { V } } } D _ { H } ^ { * } G _ { U } ^ { 1 / 2 } .\tag{60}
$$

Define

$$
T _ { H } = \mathrm { t r } ( G _ { U } D _ { H } \overline { { { G _ { V } } } } D _ { H } ^ { * } ) , \qquad \Delta _ { H } = \operatorname * { d e t } ( G _ { U } ) \operatorname * { d e t } ( G _ { V } ) | d _ { 1 } ( H ) d _ { 2 } ( H ) | ^ { 2 } .\tag{61}
$$

Then

$$
\sigma _ { 1 , 2 } ^ { 2 } ( B _ { H , p } ) = \frac { T _ { H } \pm \sqrt { T _ { H } ^ { 2 } - 4 \Delta _ { H } } } { 2 } .\tag{62}
$$

Here ${ \overline { { G _ { V } } } } = V _ { p } ^ { \top } { \overline { { V _ { p } } } }$ . For real modes it equals $G _ { V }$ , recovering the simpler real formula. The Gram entries are finite geometric sums. Replacing $D _ { H }$ by $D _ { \infty } = \mathrm { d i a g } ( g _ { 1 1 } , g _ { 1 2 } )$ gives the exact two-mode floor. For rank-one approximation with $N , p \geq 2 ,$ , the two-mode floor is positive exactly when the limiting matrix has rank two. Under two active modes this is equivalent to $g _ { 1 1 } g _ { 1 2 } \neq 0 , \varphi _ { 1 } \neq \varphi _ { 2 }$ , and $\varphi _ { 1 } ^ { p } \neq \varphi _ { 2 } ^ { p }$

## C.6 Filtering patterns shared across horizons

For a positive integer $T$ of consecutive horizons, define

$$
\begin{array} { r } { \mathcal { B } _ { T , p } [ q , j , s ] = C A ^ { q - 1 } F ^ { j p + s } K , \quad 1 \leq q \leq T . } \end{array}\tag{63}
$$

The factorization gives

$$
\mathcal { B } _ { T , p } [ q , j , s ] = \sum _ { k = 1 } ^ { r } d _ { k } ( q ) u _ { N , p } ( \varphi _ { k } ) _ { j } v _ { p } ( \varphi _ { k } ) _ { s } .\tag{64}
$$

The same filtering patterns can therefore be reused across horizons, with diferent weights $d _ { k } ( q )$ . Formally, the tensor’s canonical polyadic rank, the minimum number of separable three-factor terms, is at most r over C. The real bound requires real factors. Reusing patterns suggests a joint model, but does not prove that learning horizons together reduces error. Any weights used to combine horizon-specific risks should be fixed before testing.

Derivation of (64). Using the spectral decompositions of A and $F _ { ; }$

$$
C A ^ { q - 1 } F ^ { j p + s } K = \sum _ { a = 1 } ^ { r } \sum _ { k = 1 } ^ { r } g _ { a k } \lambda _ { a } ^ { q - 1 } ( \varphi _ { k } ^ { p } ) ^ { j } \varphi _ { k } ^ { s } = \sum _ { k = 1 } ^ { r } d _ { k } ( q ) u _ { N , p } ( \varphi _ { k } ) _ { j } v _ { p } ( \varphi _ { k } ) _ { s } .\tag{65}
$$

For each $k ,$ the dependence on $q , j ,$ and s separates into three factors. The tensor is therefore a sum of at most r rank-one $\mathrm { C P }$ components over C. If all factors are real, this is also a real CP-rank bound. With nonreal conjugate modes, no unqualified equality between real and complex CP rank is asserted. □

## D FINITE-WINDOW PREDICTION AND APPROXIMATION

This section supplies the population bridge needed for patch selection. It first records how the covariance geometry changes low-rank approximation, then bounds the efect of replacing the infinite past by a finite observation window. The final result expresses that bound in the notation of Section 4.

## D.1 Covariance-weighted approximation

The coeficient matrix is judged in prediction norm, not in ordinary Frobenius norm. The next proposition explains when the familiar singular-value picture remains exact and, more generally, how covariance eigenvalue bounds translate between the two norms.

Proposition 21 (Approximation risk under structured designs). If $\Gamma _ { L , p } = \Sigma _ { p } \otimes \Sigma _ { N }$ with positive-definite factors, then $\mathcal { E } _ { h } ^ { \Gamma } ( B )$ equals the squared singular-value tail of $\Sigma _ { N } ^ { 1 / 2 } B \Sigma _ { p } ^ { 1 / 2 }$ . In particular, under isotropy it equals $\textstyle \sum _ { j > h } \sigma _ { j } ^ { 2 } ( B )$ . More generally, $i f m _ { \Gamma } I \preceq \Gamma _ { L , p } \preceq M _ { \Gamma } I$ , then

$$
m _ { \Gamma } \sum _ { j > h } \sigma _ { j } ^ { 2 } ( B ) \leq \mathcal { E } _ { h } ^ { \Gamma } ( B ) \leq M _ { \Gamma } \sum _ { j > h } \sigma _ { j } ^ { 2 } ( B ) .\tag{66}
$$

Proof. Under column-major vectorization and a Kronecker covariance,

$$
\begin{array} { r } { \mathrm { v e c } ( E ) ^ { \top } ( \Sigma _ { p } \otimes \Sigma _ { N } ) \mathrm { v e c } ( E ) = \| \Sigma _ { N } ^ { 1 / 2 } E \Sigma _ { p } ^ { 1 / 2 } \| _ { \mathrm { F } } ^ { 2 } . } \end{array}\tag{67}
$$

Because the two covariance factors are positive definite, the map $M \mapsto \Sigma _ { N } ^ { 1 / 2 } M \Sigma _ { p } ^ { 1 / 2 }$ is bijective and preserves rank. The weighted rank-h problem is therefore exactly the ordinary Frobenius rank-h approximation of $\Sigma _ { N } ^ { 1 / 2 } B \Sigma _ { p } ^ { 1 / 2 }$ proving the singular-value-tail formula by Eckart–Young.

For any error matrix $E ,$

$$
\begin{array} { r } { m _ { \Gamma } \| E \| _ { \mathrm { F } } ^ { 2 } \leq \mathrm { v e c } ( E ) ^ { \top } \Gamma _ { L , p } \mathrm { v e c } ( E ) \leq M _ { \Gamma } \| E \| _ { \mathrm { F } } ^ { 2 } . } \end{array}\tag{68}
$$

Taking the infimum over $E = B - M$ with $\operatorname { r a n k } ( M ) \leq h$ proves the lower bound in (66). Evaluating the upper quadratic form at the ordinary truncated-SVD optimizer gives the upper bound. Under isotropy both inequalities are equalities. □

## D.2 The finite-window bridge

Truncating an infinite-past forecast leaves its retained weights unchanged. A predictor restricted to the window can do better by refitting those weights. Define $b _ { H , L } ^ { \circ } = \arg$ min $\mathbb { E } ( X _ { t + H } - b ^ { \top } x _ { t , L } ) ^ { 2 }$ and let $B _ { H , L , p } ^ { \circ }$ be its patch reshape. The superscript ◦ distinguishes this finite-window target from $B _ { H , p } .$

Let $\mathsf { M } _ { X }$ be the essential upper bound of the observation spectral density, normalized so that $\operatorname { C o v } ( X _ { t + k } , X _ { t } ) =$ $\begin{array} { r } { ( 2 \pi ) ^ { - 1 } \int _ { - \pi } ^ { \pi } e ^ { \mathrm { i } k \omega } f _ { X } ( \omega ) d \omega } \end{array}$ . Stability and $R > 0$ imply positive lower and finite upper spectral bounds, hence a unique finite-window target. Choose $K _ { H } < \infty$ and $\rho _ { F } < \bar { \rho } _ { F } < 1$ with $| \beta _ { H } ( \ell ) | \le K _ { H } \bar { \rho } _ { F } ^ { \ell }$ , and set $D _ { H , L } =$ $\mathsf { M } _ { X } K _ { H } ^ { 2 } \bar { \rho } _ { F } ^ { 2 L } / ( 1 - \bar { \rho } _ { F } ^ { 2 } )$

Theorem 22 (Finite-window approximation bound). Uniformly over divisors p of L,

$$
\| B _ { H , L , p } ^ { \circ } - B _ { H , p } \| _ { \Gamma _ { L , p } } ^ { 2 } \leq D _ { H , L } ,\tag{69}
$$

$$
\left| \sqrt { \mathscr { E } _ { h } ^ { \Gamma } ( B _ { H , L , p } ^ { \circ } ) } - \sqrt { \mathscr { E } _ { h } ^ { \Gamma } ( B _ { H , p } ) } \right| \leq \sqrt { D _ { H , L } } .\tag{70}
$$

$I f \Gamma _ { L , p } \succeq m _ { \Gamma } I _ { \vdots }$ , the squared Frobenius diference is at most $D _ { H , L } / m _ { \Gamma }$

Proof. The omitted old observations contribute a tail to the infinite-past forecast. Refitting the retained window can recover only the part of that tail predictable from the retained observations. Orthogonal projection cannot increase its mean squared size. This is why the same tail bound controls the change in coeficients measured by prediction error.

Let $b _ { H , L } = ( \beta _ { H } ( 0 ) , \dots , \beta _ { H } ( L - 1 ) ) ^ { \intercal }$ , ordered consistently with $\boldsymbol { x } _ { t , L } = ( X _ { t } , . . . , X _ { t - L + 1 } ) ^ { \top }$ , and write the full-past forecast as

$$
\Pi _ { t , H } = b _ { H , L } ^ { \top } x _ { t , L } + T _ { t , H , L } , \qquad T _ { t , H , L } = \sum _ { \ell \geq L } \beta _ { H } ( \ell ) X _ { t - \ell } .\tag{71}
$$

Let $\mathsf { P } _ { L }$ be orthogonal projection in $L _ { 2 }$ onto the span of $x _ { t , L }$ . Since the full-past forecast error is orthogonal to the observed past,

$$
( b _ { H , L } ^ { \circ } ) ^ { \top } x _ { t , L } = \mathsf { P } _ { L } \Pi _ { t , H } .\tag{72}
$$

Projection contraction and the spectral-density upper bound give

$$
{  { \mathbb E } } \{ \Pi _ { t , H } - \mathsf { P } _ { L } \Pi _ { t , H } \} ^ { 2 } \le {  { \mathbb E } } T _ { t , H , L } ^ { 2 } \le \mathsf { M } _ { X } \sum _ { \ell \ge L } \beta _ { H } ( \ell ) ^ { 2 } \le D _ { H , L } .\tag{73}
$$

The middle inequality follows directly from the stated Fourier convention and Parseval’s identity: for every square-summable scalar filter a, Var $\begin{array} { r } { ( \sum _ { \ell } a _ { \ell } X _ { t - \ell } ) \le \mathsf { M } _ { X } \sum _ { \ell } | a _ { \ell } | ^ { 2 } } \end{array}$ . Moreover,

$$
\begin{array} { r } { \big ( b _ { H , L } ^ { \circ } - b _ { H , L } \big ) ^ { \top } x _ { t , L } = \mathsf { P } _ { L } T _ { t , H , L } , } \end{array}\tag{74}
$$

and projection contraction gives

$$
\begin{array} { r } { \mathbb { E } ( \mathsf { P } _ { L } T _ { t , H , L } ) ^ { 2 } \leq \mathbb { E } T _ { t , H , L } ^ { 2 } \leq D _ { H , L } . } \end{array}\tag{75}
$$

This proves (69) after the fixed permutation from lag order to patch order. Positive definiteness of $\Gamma _ { L , p }$ makes the finite-window coeficient unique. The square root of approximation risk is distance to the closed rank-at-most-h set in the $\Gamma _ { L , p }$ norm, hence is 1-Lipschitz. This proves (70). The spectral lower bound gives the Frobenius conclusion. □

Because the bound decreases geometrically with L, the spectrum of the truncated coeficient matrix describes the finite-window approximation problem increasingly well. Exact low rank of one target, however, does not imply exact low rank of the other.

## D.3 Transfer to the patch-selection approximation error

Theorem 23 (Finite-window approximation). Under the state-space and stable-filter assumptions of Section 3, for every divisor p of L,

$$
\begin{array} { r } { \left| \sqrt { A _ { m } ( p ) } - \sqrt { \mathcal { E } _ { m } ^ { \Gamma } ( B _ { H , p } ) } \right| \leq \| B _ { p } ^ { \circ } - B _ { H , p } \| _ { \Gamma } , } \end{array}\tag{76}
$$

$$
0 \leq A _ { m } ( p ) \leq \| B _ { p } ^ { \circ } - B _ { H , p } \| _ { \Gamma } ^ { 2 } \quad i f m \geq r .\tag{77}
$$

The squared discrepancy $\| B _ { p } ^ { \circ } - B _ { H , p } \| _ { \Gamma } ^ { 2 }$ admits a geometrically decreasing bound in $L ,$ uniformly over patch lengths, as quantified in Theorem 22.

Proof. For any $B \in \mathcal { M } _ { m } ( p )$ , the triangle inequality gives

$$
\begin{array} { r } { \sqrt { A _ { m } ( p ) } \leq \| B _ { p } ^ { \circ } - B \| _ { \Gamma } \leq \| B _ { p } ^ { \circ } - B _ { H , p } \| _ { \Gamma } + \| B _ { H , p } - B \| _ { \Gamma } . } \end{array}\tag{78}
$$

Taking the infimum over $B ,$ , and then interchanging $B _ { p } ^ { \circ }$ and $B _ { H , p }$ in the same argument, proves (76).

Theorem 22 gives the stated geometric bound on $\| B _ { p } ^ { \circ } - B _ { H , p } \| _ { \Gamma } ^ { 2 }$ , uniformly in $p .$ Finally, Theorem 3 gives rank $( B _ { H , p } ) \leq r$ . If $m \ge r , B _ { H , p }$ is feasible in the minimization defining $A _ { m } ( p )$ , which yields (77). □

## E ESTIMATION FROM ONE DEPENDENT TRAJECTORY

The estimation argument has three stages. A deterministic oracle inequality reduces prediction error to curvature, score, and comparator bounds. Gaussian concentration then establishes these bounds simultaneously over the candidate patches, ranks, and horizons. Combining the two gives the uniform rate and the finite-sample theorem in the main paper. Universal constants are denoted by $c , C$ and may change from line to line. Readers interested in the final bounds can go directly to Corollary 27 and the proof of Theorem 6 below.

## E.1 Deterministic oracle inequality

The proof compares the fitted coeficients with the best allowed population coeficients. It bounds how much the training loss can favor an incorrect coeficient diference because of noise. The three assumed events are proved for the dependent Gaussian data in the subsections below.

Fix a candidate patch length $p ,$ rank bound $h ,$ and horizon $H .$ . Let ${ \mathcal { T } } _ { n } = \{ 1 , \ldots , n \}$ and define

$$
Q _ { p } ( D ) = \mathbb { E } \langle D , Z _ { t , p } \rangle _ { \mathrm { F } } ^ { 2 } , \qquad { \widehat { Q } } _ { p } ( D ) = { \frac { 1 } { n } } \sum _ { t \in { \cal T } _ { n } } \langle D , Z _ { t , p } \rangle _ { \mathrm { F } } ^ { 2 } .\tag{79}
$$

Write $\widehat { B } _ { p , h } = \widehat { B } _ { p , h , H }$ for a global rank-constrained empirical least-squares minimizer and $u _ { t , H , L } \ = \ X _ { t + H } -$ $\langle B _ { H , L , p } ^ { \circ } , \bar { Z } _ { t , p } \rangle _ { \mathrm { F } }$ for its population residual. The regression score matrix is $\begin{array} { r } { G _ { p } = n ^ { - 1 } \sum _ { t \in \mathbb { Z } _ { n } } u _ { t , H , L } Z _ { t , p } } \end{array}$ . Let $B _ { h } ^ { \circ }$ be a population rank-h optimizer and $E = B _ { h } ^ { \circ } - B _ { H , L , p } ^ { \circ }$ . Assume constants $\kappa _ { p } , M _ { p } > 0$ such that, for every rankat-most-2h coeficient diference $D , \widehat { Q } _ { p } ( D ) \geq \kappa _ { p } \| D \| _ { \mathrm { F } } ^ { 2 }$ and $Q _ { p } ( D ) \leq M _ { p } \| D \| _ { \mathrm { F } } ^ { 2 }$ . Also assume $\widehat { Q } _ { p } ( E ) \leq 2 Q _ { p } ( E )$ These are the restricted-curvature, population-upper-bound, and oracle-residual conditions used below.

Theorem 24 (Oracle inequality on dependent-data events). On the restricted-curvature, population-upperbound, and oracle-residual events stated above and verified in Appendix $E _ { i }$

$$
\begin{array} { c } { { Q _ { p } ( \widehat B _ { p , h } - B _ { H , L , p } ^ { \circ } ) \leq \left( 2 + \displaystyle \frac { 3 2 M _ { p } } { \kappa _ { p } } \right) { \mathscr E } _ { h } ^ { \Gamma } ( B _ { H , L , p } ^ { \circ } ) } } \\ { { + \displaystyle \frac { 6 4 M _ { p } h } { \kappa _ { p } ^ { 2 } } \| G _ { p } \| _ { \mathrm { o p } } ^ { 2 } . } } \end{array}\tag{80}
$$

Proof. Let $B _ { h } ^ { \circ }$ be a population rank-h optimizer, set $E = B _ { h } ^ { \circ } - B _ { H , L , p } ^ { \circ }$ and $\Delta = \widehat { B } _ { p , h } - B _ { h } ^ { \circ }$ . Then $\mathrm { r a n k } ( \Delta ) \le 2 h$ The empirical basic inequality is

$$
\widehat { Q } _ { p } ( E + \Delta ) \leq \widehat { Q } _ { p } ( E ) + 2 \langle G _ { p } , \Delta \rangle _ { \mathrm { F } } .\tag{81}
$$

Using $\begin{array} { r } { \| a + b \| ^ { 2 } \geq \frac { 1 } { 2 } \| b \| ^ { 2 } - \| a \| ^ { 2 } } \end{array}$ in the empirical prediction norm,

$$
\widehat { Q } _ { p } ( \Delta ) \leq 4 \widehat { Q } _ { p } ( E ) + 4 | \langle G _ { p } , \Delta \rangle _ { \mathrm { F } } | .\tag{82}
$$

Nuclear/operator duality and ran $\lfloor \Delta ) \leq 2 h$ imply

$$
| \langle G _ { p } , \Delta \rangle _ { \mathrm { F } } | \leq \| G _ { p } \| _ { \mathrm { o p } } \| \Delta \| _ { * } \leq \sqrt { 2 h } \| G _ { p } \| _ { \mathrm { o p } } \| \Delta \| _ { \mathrm { F } } .\tag{83}
$$

Restricted curvature, the assumed oracle-residual concentration $\widehat { Q } _ { p } ( E ) \leq 2 Q _ { p } ( E )$ , and Young’s inequality give

$$
\| \Delta \| _ { \mathrm { F } } ^ { 2 } \le \frac { 1 6 } { \kappa _ { p } } Q _ { p } ( E ) + \frac { 3 2 h } { \kappa _ { p } ^ { 2 } } \| G _ { p } \| _ { \mathrm { o p } } ^ { 2 } .\tag{84}
$$

Finally,

$$
Q _ { p } ( E + \Delta ) \le 2 Q _ { p } ( E ) + 2 Q _ { p } ( \Delta ) \le \left( 2 + \frac { 3 2 M _ { p } } { \kappa _ { p } } \right) Q _ { p } ( E ) + \frac { 6 4 M _ { p } h } { \kappa _ { p } ^ { 2 } } \| G _ { p } \| _ { \mathrm { o p } } ^ { 2 } ,\tag{85}
$$

and $Q _ { p } ( E ) = \mathcal { E } _ { h } ^ { \Gamma } ( B _ { H , L , p } ^ { \circ } )$

For completeness, $\Gamma _ { L , p } \succ 0$ makes the population objective coercive on the closed rank-at-most-h variety, so $B _ { h } ^ { \circ }$ exists. On the stated restricted-curvature event, $\widehat { Q } _ { p } ( B ) \geq \kappa _ { p } \| B \| _ { \mathrm { F } } ^ { 2 }$ for every rank-at-most-h matrix B. Hence the empirical loss is also coercive on that closed variety and a global empirical minimizer exists. The result concerns such a global minimizer and does not assert that a particular nonconvex algorithm finds it. □

## E.2 Gaussian model and spectral bounds

Assume the scalar stationary process is generated by

$$
S _ { t + 1 } = A S _ { t } + \eta _ { t + 1 } , \qquad X _ { t } = C S _ { t } + \xi _ { t } ,\tag{86}
$$

where $\rho _ { A } < 1 , \eta _ { t } \overset { \mathrm { i i d } } { \sim } N ( 0 , Q ) , \xi _ { t } \overset { \mathrm { i i d } } { \sim } N ( 0 , R ) , R > 0$ , and the two noise sequences are mutually independent. With the convention

$$
\gamma _ { X } ( k ) = { \frac { 1 } { 2 \pi } } \int _ { - \pi } ^ { \pi } e ^ { \mathrm { i } k \omega } f _ { X } ( \omega ) d \omega ,\tag{87}
$$

the spectral density is

$$
f _ { X } ( \omega ) = C ( I - A e ^ { - \mathrm { i } \omega } ) ^ { - 1 } Q ( I - A ^ { \top } e ^ { \mathrm { i } \omega } ) ^ { - 1 } C ^ { \top } + R .\tag{88}
$$

Define

$$
\mathsf { m } _ { X } = \mathsf { e s s i n f } \ f _ { X } ( \omega ) , \qquad \mathsf { M } _ { X } = \mathsf { e s s s u p } \ f _ { X } ( \omega ) , \qquad \chi _ { X } = \mathsf { M } _ { X } / \mathsf { m } _ { X } .\tag{89}
$$

These quantities are finite and strictly positive. Indeed, with

$$
\mathcal { R } _ { A } = \operatorname* { s u p } _ { | z | = 1 } \| ( I - A z ) ^ { - 1 } \| _ { \mathrm { o p } } ,\tag{90}
$$

stability and $R > 0$ give the explicit bounds

$$
R \leq \mathsf { m } _ { X } \leq \mathsf { M } _ { X } \leq R + \| C \| _ { 2 } ^ { 2 } \| Q \| _ { \mathrm { o p } } \mathcal { R } _ { A } ^ { 2 } .\tag{91}
$$

Thus $\chi _ { X }$ , unlike $\rho _ { A }$ alone, records both persistence and nonnormal resolvent amplification.

## E.3 Lag filters and candidate classes

Fix a context length L. For every patch $p \ | \ L .$ , put $N _ { p } = L / p ,$ and let $Z _ { t , p } \in \mathbb { R } ^ { N _ { p } \times p }$ be the fixed permutation/reshape of $\boldsymbol { x } _ { t , L } = ( X _ { t } , \ldots , X _ { t - L + 1 } ) ^ { \top }$ . For any matrix D, let $\log _ { p } ( D ) \in \mathbb { R } ^ { L }$ be its coeficient vector in lag order and define

$$
a _ { p , D } ( \omega ) = \sum _ { \ell = 0 } ^ { L - 1 } \log _ { p } ( D ) _ { \ell } e ^ { - \mathrm { i } \ell \omega } .\tag{92}
$$

Because patching is a permutation,

$$
\| \log _ { p } ( D ) \| _ { 2 } = \| D \| _ { \mathrm { F } } , \qquad \operatorname* { s u p } _ { \omega } | a _ { p , D } ( \omega ) | ^ { 2 } \leq L \| D \| _ { \mathrm { F } } ^ { 2 } .\tag{93}
$$

Consequently,

$$
\mathsf { m } _ { X } \| D \| _ { \mathrm { F } } ^ { 2 } \leq Q _ { p } ( D ) : = \mathbb { E } \langle D , Z _ { t , p } \rangle _ { \mathrm { F } } ^ { 2 } \leq \mathsf { M } _ { X } \| D \| _ { \mathrm { F } } ^ { 2 } .\tag{94}
$$

For the consecutive training set $T _ { n } = \{ 1 , \ldots , n \}$ , define

$$
\widehat { Q } _ { p } ( D ) = \frac { 1 } { n } \sum _ { t = 1 } ^ { n } \langle D , Z _ { t , p } \rangle _ { \mathrm { \tiny { F } } } ^ { 2 } .\tag{95}
$$

Let $\mathcal { P } _ { L }$ be a nonempty finite set of admissible divisors, let $\mathcal { H } \subset \{ 1 , 2 , \dotsc \}$ be a nonempty finite set of horizons, and, for every $p ,$ let $\mathcal { R } _ { p }$ be a nonempty finite set of positive candidate ranks. Assume $0 < \delta < 1$ . Put

$$
\ L \mathfrak { R } = | \mathcal { H } | \sum _ { p \in \mathcal { P } _ { L } } | \mathcal { R } _ { p } | , \qquad \ell _ { \delta } = \log \frac { 1 2 \operatorname* { m a x } \{ 1 , \mathfrak { R } \} } { \delta } .\tag{96}
$$

There are $\textstyle \sum _ { p } | { \mathcal { R } } _ { p } |$ curvature classes, $2 | \mathcal { P } _ { L } | | \mathcal { H } |$ signed score-polarization classes, and K fixed-oracle classes. Nonemptiness implies that their total is at most 4K. This is the union-class count used with $\ell _ { \delta }$ . For $( p , h )$ , define

$$
r _ { p , h } = \operatorname * { m i n } \{ 2 h , N _ { p } , p \} , \qquad s _ { p , h } = r _ { p , h } ( N _ { p } + p + 1 ) ,\tag{97}
$$

$$
k _ { p , h } = s _ { p , h } + \ell _ { \delta } , \qquad q _ { p } = N _ { p } + p + 1 + \ell _ { \delta } .\tag{98}
$$

## E.4 Forecast residuals and spectral scales

For each H, let $b _ { H , L } ^ { \circ }$ be the population least-squares coeficient for predicting $X _ { t + H }$ from $x _ { t , L }$ . Its residual

$$
u _ { t , H , L } = X _ { t + H } - ( b _ { H , L } ^ { \circ } ) ^ { \top } x _ { t , L }\tag{99}
$$

is jointly Gaussian with the lag vector and is independent of that lag vector at the same time because population orthogonality and joint Gaussianity imply independence. Define the residual transfer polynomial, variance, and spectral maximum by

$$
q _ { H , L } ( \omega ) = e ^ { \mathrm { i } H \omega } - \sum _ { \ell = 0 } ^ { L - 1 } b _ { H , L } ^ { \circ } ( \ell ) e ^ { - \mathrm { i } \ell \omega } ,\tag{100}
$$

$$
\sigma _ { u , H , L } ^ { 2 } = \frac { 1 } { 2 \pi } \int _ { - \pi } ^ { \pi } f _ { X } ( \omega ) | q _ { H , L } ( \omega ) | ^ { 2 } d \omega ,\tag{101}
$$

$$
\mathsf { M } _ { u , H , L } = \underset { \omega } { \mathrm { e s s } } \underset { G } { \mathrm { s u p } } f _ { X } ( \omega ) | q _ { H , L } ( \omega ) | ^ { 2 } ,\tag{102}
$$

$$
\mathsf { S } _ { H , L } = \sigma _ { u , H , L } ^ { 2 } + \mathsf { M } _ { X } ,
$$

$$
\Lambda _ { H , L } = \mathsf { M } _ { u , H , L } + L \mathsf { M } _ { X } .\tag{103}
$$

All of these are finite deterministic functionals of the SSM and the finite-window projection.

The one-sided suficient restricted-curvature scale used below is

$$
n _ { \mathrm { c u r v } } ^ { \mathrm { s u f f } } ( L ) = \frac { n } { \chi _ { X } ^ { 2 } L } .\tag{104}
$$

It is only a conservative suficient scale for the curvature proof. The factor L is the exact Fourier-envelope penalty in (93), not an independence-blocking assumption.

## E.5 Gaussian concentration tools

For a metric space $( \tau , d )$ , define $\begin{array} { r } { \gamma _ { 2 } ( { \mathcal T } , d ) = \operatorname* { i n f } _ { ( { \mathcal T } _ { j } ) } \operatorname* { s u p } _ { t \in { \mathcal T } } \sum _ { j \geq 0 } 2 ^ { j / 2 } d ( t , { \mathcal T } _ { j } ) } \end{array}$ , where $d ( t , { \mathcal { T } } _ { j } ) = \operatorname* { i n f } _ { s \in { \mathcal { T } } _ { j } } d ( t , s )$ $| \mathcal { T } _ { 0 } | = 1$ , and $| \mathcal { T } _ { j } | \leq 2 ^ { 2 ^ { j } }$ . This is the generic-chaining functional used in the following Gaussian chaos bound. It is the Gaussian specialization obtained from Theorem 3.5 and Proposition 2.6 of Krahmer et al. (2014).

Lemma 25 (Gaussian chaos supremum). Let $g \sim N ( 0 , I _ { m } )$ and let A be a bounded set of deterministic real matrices with a common number of columns. Define

$$
d _ { F } = \operatorname* { s u p } _ { A \in \mathcal { A } } \| A \| _ { \mathrm { F } } , \quad d _ { \mathrm { o p } } = \operatorname* { s u p } _ { A \in \mathcal { A } } \| A \| _ { \mathrm { o p } } , \quad \gamma = \gamma _ { 2 } ( \mathcal { A } , \| \cdot \| _ { \mathrm { o p } } ) .\tag{105}
$$

There are universal constants $C , c > 0$ such that, for every $v \geq 1$ , with probability at least $1 - 2 e ^ { - c v ^ { 2 } }$

$$
\begin{array} { r l r } {  { \operatorname* { s u p } _ { A \in \mathcal { A } } \big \| | A g | \| _ { 2 } ^ { 2 } - \mathbb { E } \| A g \| _ { 2 } ^ { 2 } \big | \leq C \{ \gamma ^ { 2 } + d _ { F } \gamma + d _ { F } d _ { \mathrm { o p } } } }  \\ & { } & { \quad + v d _ { \mathrm { o p } } ( \gamma + d _ { F } ) + v ^ { 2 } d _ { \mathrm { o p } } ^ { 2 } \} . } \end{array}\tag{106}
$$

The assumptions of this imported result are met below because the entire finite observation vector is Gaussian and can be written as a deterministic covariance square root times a standard Gaussian vector. All indexing matrices are deterministic population objects. The proof also uses the fixed-matrix Hanson–Wright inequality, Theorem 1.1 of Rudelson and Vershynin (2013).

Let $\mathcal { N } ( \mathcal { T } , d , \epsilon )$ denote the smallest number of radius-ϵ balls in metric d covering T . The rank-class entropy bound is

$$
\log \mathcal { N } ( \{ D : \mathrm { r a n k } ( D ) \leq r , \ \| D \| _ { \mathrm { F } } = 1 \} , \| \cdot \| _ { \mathrm { F } } , \epsilon ) \leq r ( N _ { p } + p + 1 ) \log ( 9 / \epsilon ) ,\tag{107}
$$

for $0 < \epsilon < 1$ . This is Lemma 3.1 of Cand\`es and Plan (2011) and also follows directly by covering the two Stiefel factors and singular values in an SVD. Dudley’s entropy integral applied to (107) is finite and contributes the square root of the exponent in (97).

## E.6 Simultaneous dependent-data events

Use the consecutive training targets ${ \mathcal { T } } _ { n } = \{ 1 , \ldots , n \}$ . All variables needed by every horizon and lag lie in the single Gaussian vector

$$
\mathbf { X } = ( X _ { 1 - L + 1 } , \ldots , X _ { n + \operatorname* { m a x } \mathcal { H } } ) ^ { \top } .\tag{108}
$$

Thus the following result accounts for the full overlap among the training examples, whose context-to-target spans have lengths $L + H$

Theorem 26 (Uniform dependent-data events). There is a universal constant $C > 0$ such that, with probability at least $1 - \delta _ { i }$ , the following statements hold simultaneously for every $p \in \mathcal { P } _ { L } , h \in \mathcal { R } _ { p }$ , and $H \in { \mathcal { H } }$

First, for every $\boldsymbol { D } \in \mathbb { R } ^ { N _ { p } \times p }$ with rank $( D ) \leq r _ { p , h }$ ，

$$
| \widehat { Q } _ { p } ( D ) - Q _ { p } ( D ) | \leq C \mathsf { M } _ { X } \left\{ \sqrt { \frac { L k _ { p , h } } { n } } + \frac { L k _ { p , h } } { n } \right\} \| D \| _ { \mathrm { F } } ^ { 2 } .\tag{109}
$$

Second, for

$$
G _ { p , H } = \frac { 1 } { n } \sum _ { t = 1 } ^ { n } u _ { t , H , L } Z _ { t , p } ,\tag{110}
$$

$$
\| G _ { p , H } \| _ { \mathrm { o p } } \leq C \left\{ \sqrt { \frac { \mathsf { S } _ { H , L } \Lambda _ { H , L } q _ { p } } { n } } + \frac { \Lambda _ { H , L } q _ { p } } { n } \right\} .\tag{111}
$$

Third, let $B _ { p , H } ^ { \circ } = B _ { H , L , p } ^ { \circ }$ be the patch reshape of $b _ { H , L } ^ { \circ }$ , let

$$
B _ { p , h , H } ^ { \circ } \in \arg \operatorname* { m i n } _ { \operatorname { r a n k } ( B ) \leq h } Q _ { p } ( B - B _ { p , H } ^ { \circ } ) , \qquad E _ { p , h , H } = B _ { p , h , H } ^ { \circ } - B _ { p , H } ^ { \circ } .\tag{112}
$$

Then

$$
| \widehat { Q } _ { p } ( E _ { p , h , H } ) - Q _ { p } ( E _ { p , h , H } ) | \leq C Q _ { p } ( E _ { p , h , H } ) \left\{ \sqrt { \frac { \chi _ { X } L \ell _ { \delta } } { n } } + \frac { \chi _ { X } L \ell _ { \delta } } { n } \right\} .\tag{113}
$$

If $E _ { p , h , H } = 0$ , the left side is identically zero.

Proof. Step 1: restricted curvature. Represent ${ \bf X } = \Sigma _ { \bf X } ^ { 1 / 2 } g$ with $g \sim N ( 0 , I )$ . For a lag coeficient $d = \log _ { p } ( D )$ let $T _ { d } \mathbf { X } = ( ( d ^ { \top } x _ { t , L } ) ) _ { t = 1 } ^ { n }$ and define $A _ { D } = n ^ { - 1 / 2 } T _ { d } \Sigma _ { \mathbf { X } } ^ { 1 / 2 }$ . Stationarity and the spectral integral give

$$
\begin{array} { r } { \| A _ { D } \| _ { \mathrm { F } } ^ { 2 } = Q _ { p } ( D ) \leq \mathsf { M } _ { X } \| D \| _ { \mathrm { F } } ^ { 2 } , } \\ { \| A _ { D } - A _ { D ^ { \prime } } \| _ { \mathrm { o p } } \leq \sqrt { \frac { \mathsf { M } _ { X } L } { n } } \| D - D ^ { \prime } \| _ { \mathrm { F } } . } \end{array}\tag{114}
$$

The second line follows because the covariance matrix of a length-n sample from a process with spectral density $f _ { X } | a _ { p , D - D ^ { \prime } } | ^ { 2 }$ has operator norm at most its spectral supremum, followed by (93).

Apply Lemma 25 to $\mathcal { A } _ { p , h } = \{ A _ { D } : \mathrm { r a n k } ( D ) \leq r _ { p , h } , \| D \| _ { \mathrm { F } } = 1 \}$ . Equations (107)–(114) and Dudley’s integral give

$$
\begin{array} { c } { d _ { F } \leq \sqrt { { \mathsf { M } } _ { X } } , \qquad d _ { \mathrm { o p } } \leq \sqrt { { \mathsf { M } } _ { X } L / n } , } \\ { \gamma _ { \mathrm { 2 } } ( \mathcal { A } _ { p , h } , \parallel \cdot \parallel _ { \mathrm { o p } } ) \leq C \sqrt { \frac { { \mathsf { M } } _ { X } L } { n } } \sqrt { s _ { p , h } } . } \end{array}\tag{115}
$$

With $v ^ { 2 }$ proportional to $\ell _ { \delta } .$ , the six terms in (106) are bounded, respectively, by universal multiples of

$$
\begin{array} { r l } & { \frac { \mathsf { M } _ { X } L s _ { p , h } } { n } , \qquad \mathsf { M } _ { X } \sqrt { \frac { L s _ { p , h } } { n } } , \qquad \mathsf { M } _ { X } \sqrt { \frac { L } { n } } , } \\ & { \frac { \mathsf { M } _ { X } L \sqrt { s _ { p , h } \ell _ { \delta } } } { n } , \qquad \mathsf { M } _ { X } \sqrt { \frac { L \ell _ { \delta } } { n } } , \qquad \frac { \mathsf { M } _ { X } L \ell _ { \delta } } { n } . } \end{array}\tag{116}
$$

Using $\sqrt { s \ell } \leq ( s + \ell ) / 2$ shows term by term that their sum is bounded by the right side of (109) for $\| D \| _ { \mathrm { F } } = 1$ Both quadratic forms are homogeneous of degree two, so rescaling gives the claim for every nonzero D. It is trivial for $D = 0$ . The stated union-class count then proves the simultaneous curvature event.

Step 2: score concentration. For the score, let $U _ { H } \mathbf { X } = ( u _ { t , H , L } ) _ { t = 1 } ^ { n }$ and put $R _ { H } = n ^ { - 1 / 2 } U _ { H } \Sigma _ { \mathbf { X } } ^ { 1 / 2 }$ . The residual spectral density and contemporaneous projection orthogonality give

$$
\begin{array} { l } { \displaystyle \| R _ { H } \| _ { \mathrm { F } } ^ { 2 } = \sigma _ { u , H , L } ^ { 2 } , \qquad \| R _ { H } \| _ { \mathrm { o p } } ^ { 2 } \le \mathsf { M } _ { u , H , L } / n , } \\ { \displaystyle \mathrm { t r } ( R _ { H } A _ { D } ^ { \top } ) = \mathbb { E } \langle R _ { H } g , A _ { D } g \rangle } \\ { \displaystyle \qquad = \frac { 1 } { n } \displaystyle \sum _ { t = 1 } ^ { n } \mathbb { E } \{ u _ { t , H , L } \langle D , Z _ { t , p } \rangle _ { \mathrm { F } } \} } \\ { \displaystyle \qquad = 0 \qquad \mathrm { f o r ~ e v e r y ~ } D . } \end{array}\tag{117}
$$

The last equality is the trace orthogonality needed for centered polarization. It follows because $b _ { H , L } ^ { \circ }$ is the population projection onto the full L-lag span. Operator/nuclear duality gives

$$
\| G _ { p , H } \| _ { \mathrm { o p } } = \operatorname* { s u p } _ { \mathrm { r a n k } ( D ) = 1 , \| D \| _ { \mathrm { F } } = 1 } | \langle R _ { H } g , A _ { D } g \rangle | .\tag{118}
$$

For every such D, polarization yields

$$
\langle R _ { H } g , A _ { D } g \rangle = \frac { 1 } { 4 } \{ \| ( R _ { H } + A _ { D } ) g \| _ { 2 } ^ { 2 } - \| ( R _ { H } - A _ { D } ) g \| _ { 2 } ^ { 2 } \} .\tag{119}
$$

Apply Lemma $2 5$ to the two matrix classes $\{ R _ { H } \pm A _ { D } : \mathrm { r a n k } ( D ) = 1 , \| D \| _ { \mathrm { F } } = 1 \}$ . Their parameters satisfy

$$
d _ { F } ^ { 2 } \leq { \sf S } _ { H , L } , \qquad d _ { \mathrm { o p } } ^ { 2 } \leq 2 \Lambda _ { H , L } / n ,
$$

$$
\gamma _ { 2 } \leq C \sqrt { \frac { \mathsf { M } _ { X } L } { n } } \sqrt { N _ { p } + p + 1 } .\tag{120}
$$

Write $d _ { p } ^ { 0 } = N _ { p } + p + 1$ , so $q _ { p } = d _ { p } ^ { 0 } + \ell _ { \delta }$ . Term by term, the six quantities in (106) satisfy

$$
\begin{array} { r l } & { \gamma _ { 2 } ^ { 2 } \leq C \Lambda _ { H , L } q _ { p } / n , } \\ & { d _ { F } \gamma _ { 2 } \leq C \sqrt { 5 _ { H , L } \Lambda _ { H , L } q _ { p } / n } , } \\ & { d _ { F } d _ { \mathrm { o p } } \leq C \sqrt { 5 _ { H , L } \Lambda _ { H , L } q _ { p } / n } , } \\ & { v d _ { \mathrm { o p } } \gamma _ { 2 } \leq C \Lambda _ { H , L } \sqrt { d _ { p } ^ { 0 } \ell _ { \delta } } / n \leq C \Lambda _ { H , L } q _ { p } / n , } \\ & { v d _ { \mathrm { o p } } d _ { F } \leq C \sqrt { 5 _ { H , L } \Lambda _ { H , L } q _ { p } / n } , } \\ & { v ^ { 2 } d _ { \mathrm { o p } } ^ { 2 } \leq C \Lambda _ { H , L } q _ { p } / n . } \end{array}\tag{121}
$$

Here we used $\mathsf { M } _ { X } L \le \Lambda _ { H , L } , \sqrt { d _ { p } ^ { 0 } \ell _ { \delta } } \le q _ { p } / 2$ , and $q _ { p } \geq 1$ . This proves (111). The two signs and the stated union-class count give simultaneity over $( p , H )$

Step 3: the fixed population comparator. Finally, for a fixed nonzero $E = E _ { p , h , H }$ , the same convolution representation has

$$
\| A _ { E } \| _ { \mathrm { F } } ^ { 2 } = Q _ { p } ( E ) , \qquad { \frac { \| A _ { E } \| _ { \mathrm { o p } } ^ { 2 } } { Q _ { p } ( E ) } } \leq { \frac { \chi _ { X } L } { n } } .\tag{122}
$$

The fixed-matrix Hanson–Wright inequality applied to $\| A _ { E } g \| _ { 2 } ^ { 2 }$ , followed by a union bound over the finite triples, proves (113). □

## E.7 Uniform rank-constrained regression rate

For every triple, let a global empirical rank-constrained estimator satisfy

$$
\widehat { B } _ { p , h , H } \in \arg \operatorname* { m i n } _ { \mathrm { r a n k } ( B ) \leq h } \frac { 1 } { n } \sum _ { t = 1 } ^ { n } \{ X _ { t + H } - \langle B , Z _ { t , p } \rangle _ { \mathrm { F } } \} ^ { 2 } ,\tag{123}
$$

and define the approximation error

$$
\begin{array} { r } { A _ { p , h , H } = Q _ { p } ( B _ { p , h , H } ^ { \circ } - B _ { p , H } ^ { \circ } ) . } \end{array}\tag{124}
$$

Corollary 27 (Uniform one-trajectory oracle rate). Suppose

$$
n \geq C \chi _ { X } ^ { 2 } L \operatorname* { m a x } _ { p \in \mathcal { P } _ { L } , h \in \mathcal { R } _ { p } } k _ { p , h } .\tag{125}
$$

Then, after increasing the universal constant if necessary, with probability at least $1 - \delta ,$ simultaneously for all $( p , h , H )$

$$
\begin{array} { r l } & { Q _ { p } ( \widehat { B } _ { p , h , H } - { B } _ { p , H } ^ { \circ } ) \leq ( 2 + 6 4 \chi _ { X } ) { A } _ { p , h , H } } \\ & { \qquad + C \frac { \mathsf { M } _ { X } h } { \mathsf { m } _ { X } ^ { 2 } } \left\{ \frac { \mathsf { S } _ { H , L } \Lambda _ { H , L } q _ { p } } { n } + \frac { \Lambda _ { H , L } ^ { 2 } q _ { p } ^ { 2 } } { n ^ { 2 } } \right\} . } \end{array}\tag{126}
$$

If additionally

$$
n \geq C \operatorname* { m a x } _ { p , H } \frac { \Lambda _ { H , L } } { \mathsf { S } _ { H , L } } q _ { p } ,\tag{127}
$$

then the last display simplifies to

$$
Q _ { p } ( \widehat { B } _ { p , h , H } - B _ { p , H } ^ { \circ } ) \leq ( 2 + 6 4 \chi _ { X } ) \mathcal { A } _ { p , h , H } + C \frac { \mathsf { M } _ { X } h \mathsf { S } _ { H , L } \Lambda _ { H , L } } { \mathsf { m } _ { X } ^ { 2 } } \frac { q _ { p } } { n } .\tag{128}
$$

Proof. Condition (125) makes the right side of (109) at most $\mathsf { m } _ { X } \| D \| _ { \mathrm { F } } ^ { 2 } / 2$ . Hence the restricted-curvature event holds with $\kappa _ { p } = \mathsf { m } _ { X } / 2$ . The same condition implies the fixed-oracle event $\widehat { Q } _ { p } ( E _ { p , h , H } ) \leq 2 Q _ { p } ( E _ { p , h , H } )$ . Theorem 24, with $M _ { p } = \mathsf { M } _ { X }$ , gives

$$
Q _ { p } ( \widehat { B } _ { p , h , H } - B _ { p , H } ^ { \circ } ) \leq ( 2 + 6 4 \chi _ { X } ) \varLambda _ { p , h , H } + \frac { 2 5 6 \mathsf { M } _ { X } h } { \mathsf { m } _ { X } ^ { 2 } } \| G _ { p , H } \| _ { \mathrm { o p } } ^ { 2 } .\tag{129}
$$

Substitute (111) and use $( a + b ) ^ { 2 } \leq 2 a ^ { 2 } + 2 b ^ { 2 }$ to obtain (126). Condition (127) absorbs the quadratic score term into the linear term. □

Interpretation. The sample-size condition can be written as the one-sided implication

$$
n _ { \mathrm { c u r v } } ^ { \mathrm { s u f f } } ( L ) \geq C \operatorname* { m a x } _ { p , h } k _ { p , h } \quad \Longrightarrow \quad \mathrm { t h e ~ s t a t e d ~ r e s t r i c t e d ~ c u r v a t u r e ~ e v e n t . }\tag{130}
$$

This is only a suficient curvature scale. It is not an information-equivalent sample size and no reverse implication is claimed. The score bound separately contains the factor $q _ { p } = N _ { p } + p + 1 + \ell _ { \delta }$ . The overlap cost is not asserted to be minimax sharp. It arises from the worst-case Fourier concentration of an arbitrary low-rank reshaped lag filter. A sharper theorem would replace the envelope L by a localized Fourier-width functional of the rank class.

## E.8 Proof of the main finite-sample bound

Proof of Theorem 6. For each candidate p, apply Corollary 27 with rank $h = \operatorname* { m i n } \{ m , L / p , p \}$ and horizon H. This rank gives exactly the same coeficient class as $\mathcal { M } _ { m } ( p )$ , including when the original rank constraint is inactive. Thus its approximation term is $A _ { m } ( p )$ , and its population quadratic form is the squared prediction norm used here. There is one candidate rank per patch length, so its confidence term is $\log ( 1 2 \vert \mathcal { P } _ { L } \vert / \delta )$

In this proof, the dimension formula can be written as $d _ { m } ( p ) = h ( L / p + p - h )$ . Since $1 \leq h \leq \operatorname* { m i n } \{ L / p , p \}$ ， both $h ^ { 2 } \leq d _ { m } ( p )$ and $h \leq d _ { m } ( p )$ . Consequently,

$$
\begin{array} { c } { h ( L / p + p + 1 ) = d _ { m } ( p ) + h ^ { 2 } + h \leq 3 d _ { m } ( p ) , } \\ { \operatorname* { m i n } ( 2 h , L / p , p ) ( L / p + p + 1 ) \leq 6 d _ { m } ( p ) . } \end{array}\tag{131}
$$

Also $L / p + p + 1 \leq h ( L / p + p + 1 ) \leq 3 d _ { m } ( p )$ . These inequalities, together with $L , m \ge 1$ , show that (25) implies both (125) and (127) when $C _ { 0 }$ is suficiently large. The spectral factors entering these conditions depend only on the model, H, and $L \colon$ reshaping preserves the population forecast residual and the eigenvalues of the context covariance.

The estimation term in (128) is a fixed spectral prefactor times

$$
\frac { h } { n } \left\{ \frac { L } { p } + p + 1 + \log \frac { 1 2 | \mathcal { P } _ { L } | } { \delta } \right\} .\tag{132}
$$

Since $h \leq m$ , this expression is bounded by

$$
\frac { 3 } { n } \left\{ d _ { m } ( p ) + m \log \frac { 1 2 | \mathcal { P } _ { L } | } { \delta } \right\} .\tag{133}
$$

Absorb the factor three and the spectral prefactor into $C _ { 2 }$ , and take $C _ { 1 }$ to be the approximation coeficient in that corollary. This proves (26) simultaneously over all candidates with probability at least 1 − δ. The constants are independent of $p , \ n , \ m , \ \mathcal { P } _ { L }$ , and δ. The underlying concentration result accounts for overlap and serial dependence among training windows.

For an independent stationary test window, the projection identity (9), conditional on the fitted coeficient, adds the common residual risk $R _ { 0 }$ to this squared prediction-norm error. This proves the stated interpretation as a bound on prediction risk. □

## E.9 Squared and root risk under exact coeficient rank

We combine the finite-window bridge with the uniform estimation bound. Here $B _ { p , H } ^ { \circ } = B _ { H , L , p } ^ { \circ }$ and $\mathsf { M } _ { X }$ is the observation spectral upper bound.

Corollary 28 (Squared and root risk under exact coeficient rank). Assume the finite-window bridge conditions of Theorem 22, the conditions of Corollary 27, and the linear-score regime (127). Fix a candidate $( p , h , H )$ for which the reshaped infinite-past kernel satisfies rank $\left( B _ { H , p } \right) \leq h$ . Then, on the simultaneous event of Corollary 27,

$$
\begin{array} { r l } & { { \cal Q } _ { p } ( \widehat { B } _ { p , h , H } - { B } _ { p , H } ^ { \circ } ) \leq ( 2 + 6 4 \chi _ { X } ) { D } _ { H , L } } \\ & { \quad \quad \quad \quad \quad + { C } \frac { \mathsf { M } _ { X } h \mathsf { S } _ { H , L } \Lambda _ { H , L } } { \mathsf { m } _ { X } ^ { 2 } } \frac { q _ { p } } { n } . } \end{array}\tag{134}
$$

Consequently,

$$
\begin{array} { r l } & { \sqrt { Q _ { p } ( \widehat { B } _ { p , h , H } - B _ { p , H } ^ { \circ } ) } \leq \sqrt { 2 + 6 4 \chi _ { X } } \sqrt { D _ { H , L } } } \\ & { \qquad + C \left\{ \frac { \mathsf { M } _ { X } h \mathsf { S } _ { H , L } \Lambda _ { H , L } } { \mathsf { m } _ { X } ^ { 2 } } \frac { q _ { p } } { n } \right\} ^ { 1 / 2 } . } \end{array}\tag{135}
$$

In particular, the finite-window contribution to root risk is ${ \cal O } ( \bar { \rho } _ { F } ^ { L } )$ , while the displayed estimation contribution is on the square root of the prediction-risk scale.

Proof. Because $B _ { H , p }$ itself is an admissible rank-h matrix, $\mathcal { E } _ { h } ^ { \Gamma } ( B _ { H , p } ) = 0$ . The root-risk bridge (70) therefore gives

$$
\sqrt { \mathcal { A } _ { p , h , H } } = \sqrt { \mathcal { E } _ { h } ^ { \Gamma } ( B _ { H , L , p } ^ { \circ } ) } \leq \sqrt { D _ { H , L } } ,\tag{136}
$$

and hence $\begin{array} { r } { A _ { p , h , H } \leq D _ { H , L } . } \end{array}$ . Substitution into (128) proves (134). Apply ${ \sqrt { a + b } } \leq { \sqrt { a } } + { \sqrt { b } }$ to obtain (135). Finally, $D _ { H , L } = \mathsf { M } _ { X } K _ { H } ^ { 2 } \bar { \rho } _ { F } ^ { 2 L } / ( 1 - \bar { \rho } _ { F } ^ { 2 } )$ by Theorem 22. □

## E.10 Scope and possible refinements

The logical chain is

$$
\begin{array} { r l } & { \mathrm { s t a b l e ~ G a u s s i a n ~ S S M } \Longrightarrow ( \mathfrak { m } _ { X } , \mathsf { M } _ { X } , \mathsf { M } _ { u , H , L } ) , } \\ & { \qquad \mathrm { ( m } _ { X } , \mathsf { M } _ { X } , \mathsf { M } _ { u , H , L } ) \Longrightarrow \left\{ \begin{array} { l l } { \mathrm { l o w - r a n k ~ c u r v a t u r e } , } \\ { \mathrm { s c o r e ~ c o n c e n t r a t i o n } , } \\ { \qquad \mathrm { f i x e d - o r a c l e ~ c o n c e n t r a t i o n } , } \\ { \qquad \mathrm { t h r e e ~ e v e n t s \Longrightarrow d e t e r m i n i s t i c ~ o r a c l e ~ i n e q u a l i t y \Longrightarrow ~ u n i f o r m ~ o r a c l e ~ r a t e . } } \end{array} \right. } \end{array}\tag{137}
$$

The concentration tools in Appendix E establish the stated uniform prediction bound for global rank-constrained least-squares minimizers. The analysis does not provide a localized improvement of the worst-case L envelope, data-driven estimates of the spectral constants, an optimization guarantee for finding the global minimizer, or an oracle theorem for selecting $( p , h )$ from the same trajectory.

A possible localized refinement. A sharper patch-dependent bound would require upper and lower bounds for

$$
\gamma _ { 2 } ( \{ D : { \mathrm { r a n k } } ( D ) \leq r , \ \| D \| _ { \mathrm { F } } = 1 \} , ( D , D ^ { \prime } ) \mapsto \| a _ { p , D - D ^ { \prime } } \| _ { \infty } )\tag{138}
$$

on a localized secant or tangent class around the population optimizer. Establishing that this width is smaller than the global bound $C \sqrt { L r ( N _ { p } + p + 1 ) }$ under verifiable spectral conditions on the coeficient matrix remains outside the present analysis. Neither Theorem 26 nor Corollary 27 depends on such an improvement.

## F PATCH SELECTION AND COMPUTATIONAL TRADEOFFS

The results below combine the approximation and estimation terms from the preceding sections. We first state a conditional selection criterion that allows a declared dependence adjustment and calculate the dimension of the admissible coeficient class, including the extreme cases. We then prove the selection statements from Section 4, give a margin condition for the conditional criterion, and account for computational cost.

## F.1 When balanced patches are a useful reference

To state a heuristic that allows a user-specified dependence adjustment, we now write h for a generic rank budget and $n _ { \mathrm { e f f } } ( p )$ for a declared efective-sample proxy. For fixed rank bound h, the largest-dimensional part of the coeficient class has dimension

$$
d _ { p } = \widetilde { q } _ { p } ( N + p - \widetilde { q } _ { p } ) , \qquad \widetilde { q } _ { p } = \mathrm { m i n } ( h , N , p ) .\tag{139}
$$

In the unsaturated case this is $h ( L / p + p - h )$ , minimized over continuous $p$ at $\sqrt { L }$ . At either extreme, $p = 1$ or $p = L$ , every coeficient vector has a rank-one reshape, but the class still has $L$ free coeficients. Low rank alone therefore does not imply low complexity.

A simple fixed-regression comparison is

$$
\begin{array} { c } { { \displaystyle \Psi ( p ) = A _ { H , L , h } ( p ) + c _ { H , L } \frac { d _ { p } } { n _ { \mathrm { e f f } } ( p ) } , } } \\ { { \displaystyle A _ { H , L , h } ( p ) = \mathcal { E } _ { h } ^ { \Gamma } ( B _ { H , L , p } ^ { \circ } ) , } } \end{array}\tag{140}
$$

where $c _ { H , L } > 0$ is a penalty scale and $n _ { \mathrm { e f f } } ( p ) > 0$ is a declared dependence proxy. The estimation theorem does not supply a universal choice of these quantities. When approximation error and the proxy are constant across patches, divisors near $\sqrt { L }$ minimize this criterion. Appendix F.5 gives a margin condition allowing small approximation diferences. When representation error changes substantially with $p ,$ it can outweigh the dimension advantage.

Nor is $d _ { p }$ the parameter count of an attention network. The simulation control has separate query, positional-key, and value parameters, and two parameterizations of the same coeficient class can have diferent initialization, regularization, and optimization behavior. This distinction is not resolved by the fixed-regression bound and must be assessed empirically for the implemented architecture.

## F.2 Dimension and extreme patch lengths

The manifold of real $N \times p$ matrices of exact rank q has dimension $q ( N + p - q )$ : write $B = U V ^ { \top }$ and subtract the $q ^ { 2 } \cdot$ -dimensional change-of-basis indeterminacy. The top attainable rank is $\widetilde { q } _ { p } = \operatorname* { m i n } \{ h , N , p \}$ , proving (139). In the interior,

$$
d _ { p } = h ( L / p + p - h ) .\tag{141}
$$

Thus minimizing dimension is equivalent to minimizing $L / p + p$ , whose continuous minimizer is $\sqrt { L }$ by differentiation or the arithmetic–geometric mean inequality. For saturated candidates with min $( N , p ) \leq h$ , the dimension is $L .$ For interior candidates it is $h ( N + p - h ) < L$ Thus a divisor minimizing $N + p$ is always dimension-minimizing. If every candidate is saturated, all dimensions equal $L ,$ so no unique preference follows.

Lemma 29 (Rank one at extreme patch lengths is not compression). For a scalar series with context length $L$ and any $h \geq 1$ , both extreme reshapes $p = 1$ and $p = L$ have matrix rank at most one. Their rank-one approximation error is therefore zero, but the corresponding rank class has dimension L at both extremes. Under dense token mixing, the analytic cost proxy is $L ^ { 2 }$ at $p = 1$ and 1 at $p = L$

Proof. At $p = 1$ , the coeficient matrix is $L \times 1$ . At $p = L ,$ it is $1 \times L$ . Each has rank at most one. With attainable rank $q = 1$ , the rank-class dimension is $q ( N + p - q ) = N + p - 1 = L$ in either case. The dense nonoverlapping token-mixing proxy is $N ^ { 2 } = ( L / p ) ^ { 2 }$ , which gives the two stated values. □

## F.3 A suficient condition from the forecast kernel

The filtering analysis provides a structural suficient condition for the theorem. By Theorem $3 , m \geq r$ makes the truncated infinite-past kernel $B _ { H , p }$ representable at every patch length. This does not imply that the optimal finite-window coeficient $B _ { p } ^ { \circ }$ has rank at most m. The relevant question is instead whether the discrepancy between these two coeficients is small enough to preserve the estimation advantage of balance.

Corollary 30 (A suficient condition from the forecast kernel). Maintain the conditions of Theorems $\delta , \ \mathcal { B } ,$ and 22. Suppose $r \leq m < \operatorname* { m i n } \{ L / q , q \}$ for every $q \in \mathcal { P } _ { L }$ , and let $p \in \mathcal { P } _ { L }$ minimize $d _ { m } ( p ) . ~ H$

$$
C _ { 1 } \| B _ { p } ^ { \circ } - B _ { H , p } \| _ { \Gamma } ^ { 2 } < \frac { C _ { 2 } } { n } \{ d _ { m } ( q ) - d _ { m } ( p ) \}\tag{142}
$$

for every $q \in \mathcal { P } _ { L }$ with $d _ { m } ( q ) > d _ { m } ( p )$ , then every minimizer of (27) minimizes $p + L / p$ over $\mathcal { P } _ { L }$

The corollary follows because the representable kernel is an admissible approximation to the finite-window target:

$$
0 \leq A _ { m } ( p ) \leq \| B _ { p } ^ { \circ } - B _ { H , p } \| _ { \Gamma } ^ { 2 } .\tag{143}
$$

Its condition therefore requires the cost of using this approximation to be smaller than the estimation saving, even relative to a higher-dimensional candidate with no approximation error. Theorem 22 bounds the discrepancy geometrically in L. The displayed condition specifies how small it must be for the comparison at hand. Exact low rank of the finite-window target is unnecessary.

Theorem 7 does not require $m \geq r .$ . With fewer heads, however, the preceding suficient argument is unavailable, and patch choice must account for the approximation errors described by the filtering spectrum and (70). A single dominant forecast component does not by itself remove the filtering patterns needed to recover that component from past observations. Nor does having fewer heads imply that patch length is immaterial: its efect depends on how well the reshaped target can be approximated.

## F.4 Proofs of the main patch-selection results

Proof of Theorem 7. On the stated candidate set, minimizing $d _ { m } ( p )$ is equivalent to minimizing $p + L / p$ . To prove the theorem, the diference between the criterion at q and at a dimension-minimizing p is

$$
C _ { 1 } \{ A _ { m } ( q ) - A _ { m } ( p ) \} + \frac { C _ { 2 } } { n } \{ d _ { m } ( q ) - d _ { m } ( p ) \} .\tag{144}
$$

Under (28), every candidate of higher dimension has a strictly larger criterion than $p .$ Moreover, $p + L / p \geq 2 { \sqrt { L } } ,$ with equality only at $p = { \sqrt { L } }$ . Hence $\sqrt { L }$ , when admissible, is the unique dimension minimizer. Rearranging the inequality between the criteria at $p$ and q gives (29), with equality corresponding to a tie. □

Proof of Corollary 30. By Theorem 3, $B _ { H , p } \in \mathcal { M } _ { m } ( p )$ when $m \geq r$ . Feasibility and nonnegativity of approximation error therefore give $A _ { m } ( p ) - A _ { m } ( q ) \leq \lVert \boldsymbol { B } _ { p } ^ { \circ } - \boldsymbol { B } _ { H , p } \rVert _ { \Gamma } ^ { 2 }$ . The stated condition therefore implies (28).

## F.5 A margin condition for balanced patches

Fix H, L, h and abbreviate $A ( p ) = A _ { H , L , h } ( p )$ . Let $p _ { \mathrm { b a l } }$ minimize $d _ { p }$ over the candidate set. Assume $c _ { H , L } > 0$ and a patch-independent proxy $n _ { \mathrm { e f f } } > 0$ . If every candidate has the same dimension, the margin below is unnecessary. Otherwise the minimum positive dimension gap in the statement is well-defined.

Lemma 31 (Patch-selection margin). Let $p _ { \mathrm { b a l } }$ minimize $d _ { p }$ over a finite candidate set $\mathcal { P } _ { L }$ with at least two distinct dimensions. $I f$

$$
\operatorname* { s u p } _ { p } A ( p ) - \operatorname* { i n f } _ { p } A ( p ) < c _ { H , L } \frac { \operatorname* { m i n } _ { d _ { p } > d _ { p _ { \mathrm { b a l } } } } ( d _ { p } - d _ { p _ { \mathrm { b a l } } } ) } { n _ { \mathrm { e f f } } } ,\tag{145}
$$

then every minimizer of $\Psi$ over $\mathcal { P } _ { L }$ belongs to the dimension-minimizing set.

Proof. Let $p _ { 0 }$ be any dimension minimizer. For every p with $d _ { p } > d _ { p _ { 0 } }$ ，

$$
\Psi ( p ) - \Psi ( p _ { 0 } ) \geq - \left\{ \operatorname* { s u p } _ { q \in \mathcal { P } _ { L } } A ( q ) - \operatorname* { i n f } _ { q \in \mathcal { P } _ { L } } A ( q ) \right\} + \frac { c _ { H , L } } { n _ { \mathrm { e f f } } } ( d _ { p } - d _ { p _ { 0 } } ) > 0 .\tag{146}
$$

Thus no candidate outside the dimension-minimizing set can minimize $\Psi$

## F.6 Computational-cost tradeofs

Let $a > 0$ collect the statistical penalty scale and let $c _ { \mathrm { d e n s e } } \geq 0$ price dense token mixing. Suppressing approximation error gives

$$
\Phi ( p ) = a ( L / p + p ) + c _ { \mathrm { d e n s e } } L ^ { 2 } / p ^ { 2 } .\tag{147}
$$

Its positive continuous minimizer solves

$$
p ^ { 3 } - L p - 2 ( c _ { \mathrm { d e n s e } } / a ) L ^ { 2 } = 0 .\tag{148}
$$

For fixed $m , d _ { k } , d _ { v }$ , the dot-product attention contraction instead costs $O ( m N d _ { k } d _ { v } )$ . For $c _ { \mathrm { l i n } } \geq 0$ , a corresponding linear-mixing proxy is

$$
\begin{array} { r } { \Phi _ { \mathrm { l i n } } ( p ) = a ( L / p + p ) + c _ { \mathrm { l i n } } L / p . } \end{array}\tag{149}
$$

Its continuous minimizer is $\sqrt { L ( 1 + c _ { \mathrm { l i n } } / a ) }$ . These formulas show why the computational part of patch selection must match the architecture. Evaluate the admissible patch lengths directly, since not every integer divides L. The formulas omit projection and encoder costs. They do not predict total running time or neural-network test error. Softmax attention with a single query also has mixing cost linear in N. The rank-based penalty is a fixed-regression heuristic, not a count of learned attention capacity. For learned attention, use the validation procedure in Appendix H and report the full prediction-error versus cost comparison, not only the selected patch.

Derivation of the cost minimizers. For the compute proxy (147), diferentiation yields

$$
\Phi ^ { \prime } ( p ) = a \left( 1 - \frac { L } { p ^ { 2 } } \right) - 2 c _ { \mathrm { d e n s e } } \frac { L ^ { 2 } } { p ^ { 3 } } .\tag{150}
$$

Multiplying by $p ^ { 3 } / a$ gives (148). When $c _ { \mathrm { d e n s e } } > 0$ , the cubic is negative at $p = { \sqrt { L } }$ and has a unique positive zero above it. If $p ^ { 3 }$ dominates $L p$ , balancing with $2 ( c _ { \mathrm { d e n s e } } / a ) L ^ { 2 }$ gives the $L ^ { 2 / 3 }$ regime.

For the linear-mixing proxy (149), the derivative vanishes at $p ^ { 2 } = L ( 1 + c _ { \mathrm { l i n } } / a )$ and the second derivative is $2 ( a + c _ { \mathrm { l i n } } ) L / p ^ { 3 } > 0$ . Restricting to $1 \le p \le L$ clips the continuous optimum. A divisor constraint requires evaluating the discrete candidates. This calculation includes neither a learned query/key complexity term nor an approximation curve. □

## F.7 Interpretation and validation

The theory does not say that a longer patch is always a stronger token, or that $p = { \sqrt { L } }$ is universally optimal. A patch token is the vector $( X _ { t - j p } , \dots , X _ { t - j p - p + 1 } )$ . Its length p determines how much local temporal structure is packaged together, while $N = L / p$ determines how many tokens must interact. Table 2 summarizes the competing forces. Here $n _ { \mathrm { e f f } } ( p )$ denotes a declared dependence-adjustment proxy used only in the conditional selector. It is not asserted to equal the one-sided suficient curvature scale $n / ( \chi _ { X } ^ { 2 } L )$ proved in Appendix E.

Table 2: Approximation, estimation, and computational considerations in patch selection.
<table><tr><td>Force</td><td>Quantity represented in the theory</td><td>Typical pressure on p</td></tr><tr><td>Predictive approxi- mation</td><td> $A _ { H , L , h } ( p )$  or its multi-horizon aggregate</td><td>Depends on the coefficient reshape under a rank constraint. Patching alone is lossless. No monotone direction is proved.</td></tr><tr><td>Statistical complex- ity</td><td> $d _ { p } / n _ { \mathrm { e f f } } ( p )$  , with  $d _ { p } = h ( L / p + p - h )$  when  $h \leq \operatorname* { m i n } ( N , p )$ </td><td>Toward a balanced factorization near  $\sqrt { L }$  only when approximation and the depen- dence proxy vary little.</td></tr><tr><td>Serial dependence</td><td>Patch-dependent score concentration and restricted curvature</td><td>Smaller  $n _ { \mathrm { e f f } }$  magnifies complexity control. The direction is not universally larger be- cause  $n _ { \mathrm { e f f } } ( p )$  can change with tokenization.</td></tr><tr><td>Token mixing</td><td> $( L / p ) ^ { 2 }$  for dense all-token mixing, and  $L / p$  for fixed-width linear attention</td><td>Larger patches reduce mixing. The  $L ^ { 2 / 3 }$  scale belongs only to the dense quadratic proxy.</td></tr><tr><td>Multiple horizons</td><td>Weighted-average or worst-case approxi- mation over  $H \in { \mathcal { H } }$ </td><td>One shared patch only if horizon-specific curves share a near-minimizer. Otherwise multiresolution or horizon-specific tokens.</td></tr></table>

At $p = 1$ , each scalar observation is a token and $N = L$ . The matrix $B _ { H , 1 }$ is necessarily rank one because it has one column, so a zero rank-one approximation error carries no compression content. These facts identify two costs of pointwise tokenization: it leaves all L coeficients free and maximizes dense token mixing. The separate weak-token hypothesis concerns the local structure that learned attention scores must recover. It is an architectural consideration, not a theorem that scalar tokens are intrinsically uninformative.

For implementation, use the candidate divisors of $L ,$ estimate forecast loss with chronological validation and guard gaps, and prespecify how computational cost enters the comparison. Report the entire patch-risk curve as well as the selected value. The curve is scientifically more informative than a single optimizer because it reveals whether the choice is driven by approximation, statistical uncertainty, or compute. For multiple horizons, prespecify the horizon weights or worst-case criterion rather than choosing them after observing test performance.

## G CONTEXT ADAPTATION ACROSS SYSTEMS

For one known Gaussian system, the optimal finite-window forecast is already linear, so input-dependent attention weights cannot lower the population Bayes risk. One setting in which adaptation can be useful is when the observed context reveals which forecasting rule should be used. A mixture of systems provides a simple example: the history updates the posterior probability of each system, and those posterior probabilities mix the corresponding linear forecasts.

Let $Y = X _ { t + H }$ be the forecast target. For any square-integrable forecast $g ( Z )$ , conditional expectation gives

$$
\begin{array} { r } { \mathbb { E } ( Y - g ( Z ) ) ^ { 2 } = \mathbb { E } ( Y - f ^ { * } ( Z ) ) ^ { 2 } + \mathbb { E } ( g ( Z ) - f ^ { * } ( Z ) ) ^ { 2 } . } \end{array}\tag{151}
$$

Here $f ^ { * } ( Z ) = \mathbb { E } ( Y \mid Z )$ in the model under consideration. Thus adaptation has a strict population advantage over the best fixed linear predictor exactly when that linear predictor difers from $f ^ { * }$ with positive probability. Merely varying the weights does not establish this advantage.

## G.1 Mixture model and posterior weights

Let M be the number of generating systems and draw a latent index $U \in \{ 1 , \ldots , M \}$ once per independent episode, with prior probabilities $\omega _ { u } = \mathrm { P r } ( U = u ) > 0$ summing to one. Conditional on $U = u$ , generate a stationary episode from a specified stable mean-zero Gaussian system. There is no switching within an episode. Let $\ell _ { t } = ( X _ { t } , \ldots , X _ { t - L + 1 } ) ^ { \intercal }$ have conditional covariance $G _ { u } \ \succ \ 0 ,$ and let $B _ { u } ^ { \circ }$ be the optimal finite-window coeficient matrix in system u. These coeficients need not have the same rank as the truncated infinite-past kernels. Writing $\mathcal { N } ( \ell ; 0 , G )$ for the centered Gaussian density with covariance $G ,$ , the posterior is

$$
\pi _ { u } ( Z ) = \frac { \omega _ { u } \mathcal { N } ( \ell _ { t } ; 0 , G _ { u } ) } { \sum _ { s } \omega _ { s } \mathcal { N } ( \ell _ { t } ; 0 , G _ { s } ) } .\tag{152}
$$

This probability uses observed inputs only, not the future target. Identical past covariances give constant posteriors, so context adaptation is not guaranteed for arbitrary choices of the systems.

## G.2 Oracle forecast and rank truncation

Theorem 32 (Finite-mixture oracle and spectral truncation). Under the preceding assumptions, the Bayes forecast is $\begin{array} { r } { f ^ { * } ( Z ) = \sum _ { u } \pi _ { u } ( Z ) \langle B _ { u } ^ { \circ } , Z \rangle _ { \mathrm { F } } } \end{array}$ . Choose deterministic best rank-h Frobenius approximations $B _ { u , h } ^ { \circ }$ and define $\begin{array} { r } { f _ { h } ^ { \operatorname* { m i x } } ( Z ) = \sum _ { u } \pi _ { u } ( Z ) \langle B _ { u , h } ^ { \circ } , Z \rangle _ { \mathrm { F } } } \end{array}$ . Writing $\Gamma _ { u } = \mathbb { E } [ \mathrm { v e c } ( Z ) \mathrm { v e c } ( Z ) ^ { \top } \mid U = u ]$ ,

$$
\begin{array} { r l r } {  { \mathbb { E } ( Y - f _ { h } ^ { \operatorname* { m i x } } ( Z ) ) ^ { 2 } - \mathbb { E } ( Y - f ^ { * } ( Z ) ) ^ { 2 } = \mathbb { E } ( f _ { h } ^ { \operatorname* { m i x } } ( Z ) - f ^ { * } ( Z ) ) ^ { 2 } } } \\ & { } & { \leq \displaystyle \sum _ { u } \omega _ { u } \lambda _ { \operatorname* { m a x } } ( \Gamma _ { u } ) \sum _ { k > h } \sigma _ { k } ( B _ { u } ^ { \circ } ) ^ { 2 } . } \end{array}\tag{153}
$$

Proof. Conditional Gaussian regression gives $\mathbb { E } ( Y \mid Z , U = u ) = \langle B _ { u } ^ { \circ } , Z \rangle _ { \mathrm { F } }$ The tower property gives $f ^ { * }$ Orthogonality of $Y - f ^ { * } ( Z )$ to any square-integrable function of Z gives the excess-risk identity. Set $\Delta _ { u } =$ $B _ { u , h } ^ { \circ } - B _ { u } ^ { \circ }$ . Conditional Jensen yields

$$
\left( \sum _ { u } \pi _ { u } ( Z ) \langle \Delta _ { u } , Z \rangle _ { \mathrm { F } } \right) ^ { 2 } \leq \sum _ { u } \pi _ { u } ( Z ) \langle \Delta _ { u } , Z \rangle _ { \mathrm { F } } ^ { 2 } .\tag{154}
$$

For any integrable $g ( Z )$ $\mathbb { E } [ \pi _ { u } ( Z ) g ( Z ) ] = \omega _ { u } \mathbb { E } [ g ( Z ) \mid U = u ]$ . Apply this identity, then bound the conditional quadratic form by $\lambda _ { \operatorname* { m a x } } ( \Gamma _ { u } ) \Vert \Delta _ { u } \Vert _ { \mathrm { F } } ^ { 2 }$ and use truncated SVD. □

## G.3 Implications for adaptive attention

This is an oracle mixture approximation result, not a claim that one dot-product attention layer exactly implements Gaussian posterior inference. The $\pi _ { u } ( Z )$ here are posterior probabilities over systems. They are not the signed patch coeficients $\alpha _ { H , a , j } ( Z )$ . The efective mixture coeficient $\begin{array} { r } { \sum _ { u } \pi _ { u } ( Z ) B _ { u , h } ^ { \circ } } \end{array}$ has rank at most $M h ,$ which can exceed $h .$ . Shared right-factor subspaces can reduce this upper bound. For a learned ${ \widehat { f } } ,$ its squared error to $f ^ { * }$ is at most twice the right side of (153) plus $2 \mathbb { E } | \widehat { f } ( Z ) - \widehat { f } _ { h } ^ { \operatorname* { m i x } } ( Z ) | ^ { 2 }$ . Controlling the latter requires an approximation and learning analysis of the actual architecture. The single-system, one-trajectory rate in Appendix E is not a theorem for the non-Gaussian marginal mixture or for neural training.

## H DISCUSSION AND IMPLICATIONS

This section collects practical implications of the results: when context-dependent attention weights can help, and how patch lengths and head counts should be compared in practice.

## H.1 When context dependence can help

For a known Gaussian system, the optimal finite-window forecast is linear. Context dependence can become useful when each episode is generated by an unknown system and the observed window contains information about its identity. For a fixed patch length, let $U \in \{ 1 , \ldots , M \}$ index the generating system, let $B _ { u } ^ { \circ }$ be its optimal finite-window coeficient matrix, and write $\pi _ { u } ( Z ) = \operatorname* { P r } ( U = u \mid Z )$ . If each system is centered and Gaussian, the Bayes forecast under squared loss is

$$
f ^ { * } ( Z ) = \sum _ { u = 1 } ^ { M } \pi _ { u } ( Z ) \langle B _ { u } ^ { \circ } , Z \rangle _ { \mathrm { { F } } } .\tag{155}
$$

The posterior weights generally vary with $Z ,$ so this forecast can be nonlinear even though the forecast within each system is linear. Theorem 32 in Appendix G bounds the approximation error when these posterior weights are available. Learning the weights and representing the resulting forecast require a separate analysis of the attention architecture. Fixed-system experiments alone cannot establish this potential benefit of adaptation across systems.

## H.2 Computational cost and validation

Computational cost introduces a separate consideration. At fixed widths and head count, dense all-token mixing costs $O ( N ^ { 2 } )$ , whereas associative unnormalized attention and single-query attention have mixing costs of order $N ,$ where $N = L / p$ . Larger patches reduce these mixing costs, but the total cost also includes patch projections and any context encoder. A computational penalty that varies with p changes the objective in (27) and may change its minimizer. Appendix F.6 gives illustrative mixing-cost calculations. Runtime and memory should also be measured for the implemented model.

In practice, compare patch lengths and head counts using common training and validation splits, comparable tuning budgets, and a fixed linear or positional-pooling control. For a single trajectory, use chronological splits with a gap that prevents training and validation windows and targets from overlapping. For independent series or episodes, hold out entire series or episodes. Select patch length, head count, checkpoints, and other tuning parameters using validation data. Prespecify how prediction error is traded against computational cost and how results are aggregated across horizons, and reserve the test set for final evaluation.

Pointwise tokenization $( p = 1 )$ can produce weak tokens for learned attention: queries and keys formed from individual observations do not directly encode local temporal patterns. Empirical ablations demonstrate benefits from replacing such tokens with patches (Nie et al., 2023). This is compatible with $A _ { m } ( 1 ) = 0$ , which concerns the representational capacity of the fixed linear class, rather than the learning of informative attention scores. Weak-token efects thus provide an additional architectural motivation for patching that is not captured by the present fixed-weight estimation bound.

## I SIMULATION DIAGNOSTICS AND REPRODUCIBILITY

This section separates calculations that depend only on the known state-space model from errors introduced by finite samples and model fitting. All population quantities use the same lag ordering as the main paper. Random seeds, candidate grids, optimizer settings, stopping rules, the number of windows per series, and the computing environment are listed in Appendix J.1.

## I.1 Design of the main simulations

The simulations in Section 6 use the three-state model (11) with

$$
A = \mathrm { d i a g } ( 0 . 9 5 , 0 . 8 8 , 0 . 8 0 ) , \qquad C = ( 1 , 0 . 7 , 0 . 5 ) .\tag{156}
$$

The latent components have diferent persistence levels, and each observation combines all three states with measurement noise. Process and measurement noises are independent Gaussian draws. Starting from process covariance $I _ { 3 }$ and measurement variance 0.3, we rescale both so that the stationary observation variance is one (Appendix I.2), and we initialize each independent path from its stationary distribution.

We consider window lengths $L \in \{ 6 4 , 1 4 4 \}$ , horizons $H \in \{ 1 , 2 , 4 , 8 , 1 2 \}$ , and training sets of $n \in \{ 5 , 2 5 , 2 0 0 \}$ independent series. Unlike the theoretical $n ,$ which counts consecutive windows from one trajectory, this n counts independent series. Each series supplies overlapping input–target windows, and entire series are kept separate across the training, validation, and test sets. For each window length, the patch grid contains the balanced reference $p = { \sqrt { L } } ,$ nearby divisors, and the extremes $p = 1$ and $p = L$ . Appendix J.1 lists the training protocol.

## I.2 Population construction

Begin with the transition and observation matrices in (156), process covariance $I _ { 3 } , $ , and measurement variance 0.3. If P solves

$$
P = A P A ^ { \top } + I _ { 3 } ,\tag{157}
$$

then the unscaled stationary observation variance is $v = C P C ^ { \top } + 0 . 3$ . Dividing both noise covariances by v gives

$$
Q = I _ { 3 } / v , \qquad R = 0 . 3 / v .\tag{158}
$$

Thus $\mathrm { V a r } ( X _ { t } ) = 1$ . This common rescaling of the process and measurement noise does not change the steadystate Kalman gain. The predictor gain $K$ and $F = A - K C$ are obtained from the stabilizing discrete Riccat solution under the predictor convention in Equation (12).

For an independent check that does not rely on the Kalman recursion, let $\gamma _ { X } ( k ) = \operatorname { C o v } ( X _ { t + k } , X _ { t } )$ . The exact finite-window coeficient $b _ { H , L } ^ { \circ }$ solves

$$
\Gamma _ { L } b _ { H , L } ^ { \circ } = g _ { H , L } , \qquad [ \Gamma _ { L } ] _ { i j } = \gamma _ { X } ( i - j ) , \quad [ g _ { H , L } ] _ { j } = \gamma _ { X } ( H + j ) ,\tag{159}
$$

for $0 \leq i , j < L$ . Reshaping this vector gives $B _ { H , L , p } ^ { \circ }$ for every candidate $p \ | \ L$ . The population rank-h approximation error is then

$$
A _ { H , L , h } ( p ) = \operatorname* { m i n } _ { \operatorname { r a n k } ( B ) \leq h } \| B - B _ { H , L , p } ^ { \circ } \| _ { \Gamma _ { L , p } } ^ { 2 } .\tag{160}
$$

These calculations distinguish approximation error from the error of a trained network.

## I.3 Population filtering-floor diagnostic

For each $p$ and horizon $H ,$ form the truncated infinite-past matrix $B _ { H , p }$ from Equation (14). For a one-head model, report both the absolute prediction-norm error $\mathcal { E } _ { 1 } ^ { \Gamma } ( B _ { H , p } )$ and the relative spectral diagnostic

$$
\mathcal { R } _ { H , p } = \frac { \sigma _ { 2 } ( B _ { H , p } ) } { \sigma _ { 1 } ( B _ { H , p } ) } ,\tag{161}
$$

whenever $B _ { H , p } \neq 0$ . The first quantity can shrink with H as the predictable signal decays, whereas Theorem 4 allows the second to converge to a positive limit. Plotting both prevents a relative filtering floor from being mistaken for a nonvanishing absolute forecast error. The finite-window analogue replaces $B _ { H , p }$ by $B _ { H , L , p } ^ { \circ }$ computed from (159).

## I.4 Fitted-model protocol and robustness sweeps

Each repetition begins with newly simulated stationary series. Entire series, rather than overlapping windows, are assigned to training, validation, or test sets. All patch candidates within a repetition use the same simulated series, splits, initialization schedule, tuning budget, and checkpoint rule. Any standardization beyond the population scaling above is computed from the training split and then held fixed. Validation data select the checkpoint and any optimization hyperparameters. Test data are used once for the reported comparison.

The horizon sweep holds the number of training series fixed and evaluates every $H \in \{ 1 , 2 , 4 , 8 , 1 2 \}$ . The trainingsize sweep holds the data-generating model, window length, horizon, and per-series window construction fixed while varying the number of independent training series. For each setting, report all admissible patch candidates, not only $p = 1 , p = { \sqrt { L } }$ , and $p = L$ . Excess MSE is evaluated against the analytic conditional mean obtained from (159). Total MSE is evaluated against the realized future observation. Pointwise bootstrap intervals resample complete data repetitions, not individual overlapping windows, and table standard errors are computed across those repetitions.

Because n in these simulations counts independent series while n in the one-trajectory bound counts consecutive windows, the experiments test the qualitative approximation–estimation tradeof rather than calibrating the constants in Theorem 6. The exact settings are listed in Appendix J.1.

## J ADDITIONAL EXPERIMENTS

This section reports further simulations with the fixed attention model of Section 6. Appendix J.1 lists the training protocol shared by all runs. Appendices J.2–J.4 complement the main simulations. Appendices J.5–J.7 vary the head count, the rank of the target, and the window length. Unless stated otherwise, population excess risk is $\| \widehat { B } _ { p , m , H } - B _ { H , L , p } ^ { \circ } \| _ { \Gamma _ { L , p } } ^ { 2 }$ , computed exactly from the true model for each trained coeficient matrix, and intervals are pointwise 95% bootstrap intervals that resample complete data repetitions.

## J.1 Training protocol and computing environment

All runs train the fixed model of Section 6: an input-independent query, free positional keys, shared linear values and output weights without bias, and key and value width 8 per head. Training uses AdamW with batch size 64 and weight decay $1 0 ^ { - 4 }$ , at most 4,000 updates, validation every 100 updates, and early stopping with patience 10. The learning rate is chosen from {0.001, 0.003, 0.015} and the checkpoint by validation MSE. Every patch length receives the same search and update budget, and test data are never used for selection.

For the simulations of Section 6 and Appendices J.2–J.3, each data repetition draws stationary series of length 576 with 32 forecast windows per series, holds out 50 validation and 100 test series, and uses nested training sets of $n \in \{ 5 , 2 5 , 2 0 0 \}$ series. There are 50 data repetitions (seeds 10, 20, . . . , 500) and one optimization seed. The patch grids are {1, 2, 4, 8, 16, 32, 64} for $L = 6 4$ and {1, 2, 6, 12, 24, 72, 144} for $L = 1 4 4$ . These runs used PyTorch 2.5.1 in single precision on a CPU (Apple silicon, six threads).

The studies in Appendices J.5–J.7 use the same model, optimizer, and selection rules with 100 training, 50 validation, and 100 test series of length 6,400 per repetition, 32 forecast origins per series shared across all settings, 20 data repetitions (seeds 3010, 3020, . . . , 3200), and every divisor of $L$ as a candidate patch length.

## J.2 A second observation design

Section 6 uses the dense observation vector $C = ( 1 , 0 . 7 , 0 . 5 )$ , which mixes all three latent states. We repeat the complete experiment with the near-aligned vector $C = ( 1 , 0 . 0 5 , 0 . 0 2 )$ , keeping $A = \mathrm { d i a g } ( 0 . 9 5 , 0 . 8 8 , 0 . 8 0 )$ and rescaling the noises so that $\mathrm { V a r } ( X _ { t } ) = 1$ . The observation is then dominated by the most persistent state, and the one-step Bayes MSE is 0.144, against 0.170 for the dense design.

Figure 3 and Table 3 reproduce the pattern of Section 6. With $n = 2 0 0$ , the patch length with the lowest population excess risk is $\sqrt { L }$ or the next larger divisor at every horizon and both window lengths, and both extremes have 2.1 to 4.8 times the excess risk at $\sqrt { L }$ . The interior advantage is largest when data are scarce: at $L = 1 4 4$ and $n = 5$ , the one-step test MSE is $1 6 5 . 6 \times 1 0 ^ { - 3 }$ at $p = { \sqrt { L } }$ against $2 6 5 . 0 \times 1 0 ^ { - 3 }$ and $2 7 0 . 1 \times 1 0 ^ { - 3 }$ at the extremes, and the gap narrows as n grows.

Excess MSE over the analytic optimum  
![](images/79af5aa1677f8f246af6e811072ef0349f6e2a851b0b1ca222107dbc78f14440.jpg)

Test MSE under different training sizes  
![](images/f5eaf8f11109f38f7494310be4c30d2a676ff59d3f2ba922ecfd3d9e9036956b.jpg)  
Figure 3: Near-aligned observation design, fixed one-head attention, 50 data repetitions. Top: excess MSE across forecast horizons with $n = 2 0 0$ training series. Bottom: total test MSE at $H = 1$ as the number of training series grows. Conventions match Figures 1 and 2.

Table 3: Near-aligned observation design: one-step test MSE in units of $1 0 ^ { - 3 }$ for the balanced patch and both extremes (means ± standard errors over 50 data repetitions). The one-step Bayes MSE is 144 in these units. Compare Table 1.
<table><tr><td>L</td><td>n</td><td> $p = 1$ </td><td> $p = { \sqrt { L } }$ </td><td> $p = L$ </td></tr><tr><td>64</td><td>5</td><td> $1 9 5 . 6 \pm 1 . 4$ </td><td> $1 5 5 . 1 \pm 0 . 8$ </td><td> $1 9 6 . 4 \pm 1 . 5$ </td></tr><tr><td>64</td><td>25</td><td> $1 5 5 . 3 \pm 0 . 5$ </td><td> $1 4 5 . 9 \pm 0 . 4$ </td><td> $1 5 5 . 5 \pm 0 . 6$ </td></tr><tr><td>64</td><td>200</td><td> $1 4 6 . 0 \pm 0 . 4$ </td><td> $1 4 4 . 3 \pm 0 . 4$ </td><td> $1 4 6 . 1 \pm 0 . 5$ </td></tr><tr><td>144</td><td>5</td><td> $2 6 5 . 0 \pm 3 . 0$ </td><td> $1 6 5 . 6 \pm 1 . 2$ </td><td> $2 7 0 . 1 \pm 3 . 3$ </td></tr><tr><td>144</td><td>25</td><td> $1 6 9 . 5 \pm 0 . 7$ </td><td> $1 4 7 . 4 \pm 0 . 5$ </td><td> $1 7 0 . 9 \pm 0 . 7$ </td></tr><tr><td>144</td><td>200</td><td> $1 4 8 . 3 \pm 0 . 5$ </td><td> $1 4 4 . 6 \pm 0 . 4$ </td><td> $1 4 8 . 8 \pm 0 . 5$ </td></tr></table>

## J.3 Representation versus estimation in the main simulations

The population quantities of Appendix I separate the two error sources directly. Table 4 compares an upper bound on the one-head approximation error, obtained from the truncated singular value decomposition of $B _ { H , L , p } ^ { \circ }$ and evaluated in the prediction norm, with the population excess risk of the trained model at $p = { \sqrt { L } } .$ Even with $n = 2 0 0$ training series the approximation bound is at most 7% of the trained excess risk for the dense design, and it is negligible for the near-aligned design. At the extremes the approximation error is exactly zero. The diferences between patch lengths in Figures 1, 2 and 3 are therefore driven by estimation rather than representation, as the dimension term in (27) predicts.

Figure 4 shows the scarce-data case for the dense design. With five training series, the extremes have 1.4 to 6.3 times the excess risk at $\sqrt { L }$ . The lowest risk is at $p = { \sqrt { L } }$ for short horizons and at the next larger divisor for longer horizons, again consistent with a broad interior range rather than a unique optimum.

Table 4: One-head approximation error versus trained excess risk. “Approx. bound” is an upper bound on $\mathcal { E } _ { 1 } ^ { \Gamma } ( B _ { H , L , p } ^ { \circ } )$ at $p = { \sqrt { L } }$ , and “max over $p ^ { \dag }$ its largest value over the patch grid. “Trained” is the mean population excess risk of the fitted model at $p = { \sqrt { L } }$ (50 repetitions).
<table><tr><td>Design</td><td>L</td><td>H</td><td>Signal</td><td>Approx. bound</td><td>Max over p</td><td>Trained, n = 5</td><td>Trained,  $n = 2 0 0$ </td></tr><tr><td>Dense</td><td>64</td><td>1</td><td>0.830</td><td> $2 . 3 \cdot 1 0 ^ { - 5 }$ </td><td> $4 . 6 \cdot 1 0 ^ { - 5 }$ </td><td> $1 . 5 0 \cdot 1 0 ^ { - 2 }$ </td><td> $9 . 2 \cdot 1 0 ^ { - 4 }$ </td></tr><tr><td>Dense</td><td>64</td><td>12</td><td>0.206</td><td> $2 . 6 \cdot 1 0 ^ { - 4 }$ </td><td> $4 . 7 \cdot 1 0 ^ { - 4 }$ </td><td> $5 . 8 0 \cdot 1 0 ^ { - 2 }$ </td><td> $3 . 8 \cdot 1 0 ^ { - 3 }$ </td></tr><tr><td>Dense</td><td>144</td><td>1</td><td>0.830</td><td> $9 . 3 \cdot 1 0 ^ { - 6 }$ </td><td> $3 . 5 \cdot 1 0 ^ { - 5 }$ </td><td> $2 . 5 2 \cdot 1 0 ^ { - 2 }$ </td><td> $1 . 3 \cdot 1 0 ^ { - 3 }$ </td></tr><tr><td>Dense</td><td>144</td><td>12</td><td>0.206</td><td> $1 . 1 \cdot 1 0 ^ { - 4 }$ </td><td> $3 . 8 \cdot 1 0 ^ { - 4 }$ </td><td> $1 . 0 8 \cdot 1 0 ^ { - 1 }$ </td><td> $5 . 9 \cdot 1 0 ^ { - 3 }$ </td></tr><tr><td>Near-aligned</td><td>64</td><td>1</td><td>0.856</td><td> $8 . 2 \cdot 1 0 ^ { - 1 0 }$ </td><td> $1 . 6 \cdot 1 0 ^ { - 9 }$ </td><td> $1 . 1 2 \cdot 1 0 ^ { - 2 }$ </td><td> $6 . 9 \cdot 1 0 ^ { - 4 }$ </td></tr><tr><td>Near-aligned</td><td>64</td><td>12</td><td>0.277</td><td> $1 . 0 \cdot 1 0 ^ { - 8 }$ </td><td> $2 . 0 \cdot 1 0 ^ { - 8 }$ </td><td> $5 . 4 0 \cdot 1 0 ^ { - 2 }$ </td><td> $3 . 3 \cdot 1 0 ^ { - 3 }$ </td></tr><tr><td>Near-aligned</td><td>144</td><td>1</td><td>0.856</td><td> $3 . 0 \cdot 1 0 ^ { - 1 0 }$ </td><td> $1 . 3 \cdot 1 0 ^ { - 9 }$ </td><td> $2 . 1 5 \cdot 1 0 ^ { - 2 }$ </td><td> $1 . 1 \cdot 1 0 ^ { - 3 }$ </td></tr><tr><td>Near-aligned</td><td>144</td><td>12</td><td>0.277</td><td> $3 . 9 \cdot 1 0 ^ { - 9 }$ </td><td> $1 . 6 \cdot 1 0 ^ { - 8 }$ </td><td> $9 . 1 6 \cdot 1 0 ^ { - 2 }$ </td><td> $4 . 9 \cdot 1 0 ^ { - 3 }$ </td></tr></table>

Excess MSE over the analytic optimum  
![](images/c85a7cd00a697a65a18f91e9a2bc4a22b3210ec7fa6a236ed09b8b02bf7f7972.jpg)  
Figure 4: Dense observation design with $n = 5$ training series: excess MSE across forecast horizons for fixed one-head attention (50 repetitions). Conventions match Figure 1.

## J.4 The long-horizon filtering floor

Figure 5 reports the population diagnostic (161) for both observation designs at $L = 1 4 4$ , patch lengths $p \in$ {2, 3, 6, 12}, a rank-one budget, and horizons up to $H = 1 9 2$ . Nothing is trained. For every patch length the ratio $\sigma _ { 2 } ( B _ { H , p } ) / \sigma _ { 1 } ( B _ { H , p } )$ increases with the horizon and converges to the positive limit $\kappa _ { \infty , 1 } ( p )$ of Theorem 4, essentially reached by $H = 9 6$ . The operator-norm bound (19) holds in all 72 (design, patch, horizon) cases.

The floor is modest in these systems, at most 0.038 for the dense design and $2 . 0 \cdot 1 0 ^ { - 4 }$ for the near-aligned design, while the predictable signal decays from 0.83 at $H = 1$ to about $2 \cdot 1 0 ^ { - 9 } \mathrm { ~ a t ~ } H = 1 9 2$ . This illustrates the distinction in (21): a positive relative floor accompanies a vanishing absolute approximation error.

![](images/5df57667c49f620f3cf4d39c3e764cd2fb609ed8b45a56edd8350b4889de242d.jpg)

![](images/791b2caa28eaa8472ed37fd6fe55d3e6d93910ab40e8857afb7c93a0bea28867.jpg)  
Figure 5: Population ratio $\sigma _ { 2 } / \sigma _ { 1 }$ of the truncated infinite-past coeficient matrix $B _ { H , p }$ against the horizon, $L = 1 4 4 .$ , logarithmic axes. Left: near-aligned design. Right: dense design. Dashed lines: the analytic longhorizon limits $\kappa _ { \infty , 1 } ( p )$ of Theorem 4.

## J.5 Head count when the target is nearly rank one

We train m $\in \ \{ 1 , 2 , 3 , 4 \}$ heads on the dense design of Section 6 with $L \ \in \ \{ 3 6 , 6 4 , 1 4 4 , 2 5 6 , 5 7 6 , 1 0 2 4 \}$ and $H \in \{ 1 , 2 , 4 , 8 , 1 2 , 1 6 \}$ (36 settings), using the protocol of Appendix J.1. Here the finite-window target is close to rank one: at $p = { \sqrt { L } } .$ , an upper bound on the one-head approximation error is at most 12% of the trained one-head excess risk in every setting.

Additional heads enlarge the coeficient class, and its dimension $d _ { m } ( p )$ in (24), without a meaningful representation gain, so the estimation term dominates. With the patch chosen by validation, adding heads raises the mean excess risk relative to one head in almost every setting (34, 35, and 36 of the 36 settings for two, three, and four heads). The paired 95% interval for this change lies entirely above zero in most settings (28, 32, and 30 of 36) and never lies entirely below zero (Figure 6). The patch with the lowest mean risk stays close to $\sqrt { L }$ for every head count: $p = { \sqrt { L } }$ is within 5% of the best mean risk in 115 of the 144 (head count, L, H) settings.

## J.6 Head count when the target has rank three

To create a target that one head cannot represent, we use the delayed autoregression

$$
X _ { t + 1 } = c _ { 1 } X _ { t - 4 } + c _ { 2 } X _ { t - 2 9 } + c _ { 3 } X _ { t - 5 8 } + \varepsilon _ { t + 1 } ,\tag{162}
$$

written as a state-space model with the last 59 observations as the state and no measurement noise. The latter is a boundary case of the assumption $R > 0$ , chosen to test the head-count restriction rather than the filtering results. The innovation variance is set so that $\mathrm { V a r } ( X _ { t } ) = 1$ . For $H \leq 4$ and $L \geq 6 4$ , the finite-window optimal forecast places weights $c _ { 1 } , c _ { 2 } , c _ { 3 }$ on the lags $5 - H , 3 0 - H$ , and $5 9 - H$ $\mathrm { A t } \ p = \sqrt { L }$ these three entries fall in distinct rows and columns of $B _ { H , L , p } ^ { \circ } .$ , which therefore has rank three with singular values $\left( c _ { 1 } , c _ { 2 } , c _ { 3 } \right)$ . Since $\begin{array} { r } { \mathcal { E } _ { m } ^ { \Gamma } ( B ) \geq \lambda _ { \mathrm { m i n } } ( \Gamma _ { L , p } ) \sum _ { k > m } \sigma _ { k } ( B ) ^ { 2 } } \end{array}$ , one and two heads face a positive approximation floor that three heads remove. We consider equal weights (0.3, 0.3, 0.3) and unequal weights (0.45, 0.30, 0.15), with $m \in \{ 1 , 2 , 3 , 4 \}$ $L \in \{ 6 4 , 1 4 4 , 2 5 6 , 5 7 6 \}$ , and horizons up to $H = 1 6 ~ \mathrm { ( e q u a l ) }$ or H = 12 (unequal).

Table 5 and Figure 7 show the approximation–estimation tradeof in the head count. $\mathrm { U p }$ to the target rank, each additional head lowers the excess risk. At $L = 6 4 , H = 1$ , and $p = 8$ , the mean excess risk with equal weights is 0.134, 0.068, 0.010, and 0.011 for one to four heads. A fourth head, which adds no representation, never lowers the mean risk and raises it significantly in 10 of 12 settings. With unequal weights the third singular value is small, and the third head helps in 9 of 12 settings but is significantly worse than two heads in one $( L = 5 7 6 $ $H = 4 )$

![](images/f34b05b2043d427b776cdcd4d23c38c662d540f1e8541ecfb1a725f873f58067.jpg)  
Figure 6: Nearly rank-one target (dense design): population excess risk against the number of heads, with the patch selected by validation in each repetition. Panels show horizons, colors show window lengths, and shading shows pointwise 95% bootstrap intervals over 20 repetitions.

Table 5: Rank-three targets: efect of adding one head at $p = { \sqrt { L } } ,$ over 12 settings (4 windows, $H \in \{ 1 , 2 , 4 \} )$ and 20 repetitions. “Lower mean” counts settings where the larger model has lower mean population excess risk. The interval columns count paired 95% intervals entirely favoring the larger or the smaller model.
<table><tr><td>Weights</td><td>Heads</td><td>Lower mean</td><td>Favors more heads</td><td>Favors fewer heads</td></tr><tr><td rowspan="3">(0.30, 0.30, 0.30)</td><td>1 to 2</td><td>12/12</td><td>12/12</td><td>0/12</td></tr><tr><td>2 to 3</td><td>12/12</td><td>12/12</td><td>0/12</td></tr><tr><td>3 to 4</td><td>0/12</td><td>0/12</td><td>10/12</td></tr><tr><td rowspan="3">(0.45, 0.30, 0.15)</td><td>1 to 2</td><td>12/12</td><td>12/12</td><td>0/12</td></tr><tr><td>2 to 3</td><td>9/12</td><td>9/12</td><td>1/12</td></tr><tr><td>3 to 4</td><td>0/12</td><td>0/12</td><td>10/12</td></tr></table>

![](images/24f4ee80d6648e992556e503dcf4f5f0625b6851fdb404edb552d524db45838f.jpg)  
Figure 7: Rank-three target with equal weights, $H = 1 , p = { \sqrt { L } } { \mathrm { : } }$ population excess risk of the trained model (blue) and the rank lower bound $\lambda _ { \operatorname* { m i n } } ( \Gamma _ { L , p } ) 0 . 0 9 ( 3 - m ) .$ <sub>+</sub> (orange) against the number of heads $m ,$ for four window lengths.

## J.7 Patch length across window lengths

Section 6 uses two window lengths. To examine how the preferred patch length grows with $L ,$ we train one head at nine window lengths L ∈ {36, 64, 144, 256, 576, 784, 1024, 1296, 1600}, all perfect squares, and six horizons $H \in$ $\{ 1 , 2 , 4 , 8 , 1 2 , 1 6 \}$ , for four designs. These are the dense design of Section 6 (baseline), $A = \mathrm { d i a g } ( 0 . 3 0 , - 0 . 6 0 , 0 . 9 8 )$ with the same $C$ (changed A), $C = ( 0 . 1 , 2 . 0 , 0 . 1 )$ with the same $A$ (changed $C )$ , and both changes together. For each design, window, and horizon, $\widehat { p } ^ { \star }$ minimizes the mean population excess risk over all divisors of $L _ { : }$ and we fit $p = C \sqrt { L }$ through the origin, $\begin{array} { r } { \widehat { C } = \sum _ { L } \sqrt { L } \widehat { p } ^ { \star } ( L ) / \sum _ { L } L } \end{array}$ , with intervals from 2,000 paired bootstrap resamples that reselect $\widehat { p } ^ { \star }$ in each draw.

Across the 216 (design, L, H) conditions, ${ \widehat { p } } ^ { \star } = { \sqrt { L } }$ in $9 8 , p = { \sqrt { L } }$ is within $5 \%$ of the best mean risk in 137, and an extreme patch length is best only once. For the baseline, $\widehat { p } ^ { \star }$ tracks $\sqrt { L }$ closely (Figure 8): $\widehat { C }$ rises from 1.04 at H = 1 to 1.58 at $H = 1 6$ , with centered $R ^ { 2 } \ge 0 . 8 4$ (Table 6). The changed-A and changed-C designs behave similarly, except for a poor fit for changed $C$ at $H = 1 6$ . With both changes, the square-root relation fails at $H \geq 8 :$ the preferred patches are much shorter $( \widehat { C } \approx 0 . 1 3 )$ and the fit is worse than a constant. The square-root scale is therefore a useful reference whose constant depends on the system and the horizon, not a universal law, in line with the conditional statements of Section 4.

Table 6: Fitted constants $\widehat { C }$ in $p = C \sqrt { L }$ over nine window lengths, with centered $R ^ { 2 }$ in parentheses (one head, 100 training series, 20 repetitions). Negative $R ^ { 2 }$ means that the fit is worse than a constant.
<table><tr><td>H</td><td>Baseline</td><td>Changed A</td><td>Changed C</td><td>Changed A and  $C$ </td></tr><tr><td>1</td><td>1.04 (0.94)</td><td>0.96 (0.83)</td><td>1.06 (0.94)</td><td>1.33 (0.64)</td></tr><tr><td>2</td><td>1.08 (0.93)</td><td>1.00 (0.99)</td><td>1.15 (0.94)</td><td>1.06 (0.87)</td></tr><tr><td>4</td><td>1.13 (0.90)</td><td>1.14 (0.94)</td><td>0.98 (0.96)</td><td>1.36 (0.75)</td></tr><tr><td>8</td><td>1.43 (0.89)</td><td>1.41 (0.76)</td><td>1.32 (0.85)</td><td>0.13 (−0.74)</td></tr><tr><td>12</td><td>1.33 (0.84)</td><td>1.66 (0.94)</td><td>1.55 (0.78)</td><td>0.13 (−0.73)</td></tr><tr><td>16</td><td>1.58 (0.85)</td><td>1.60 (0.92)</td><td>1.41 (0.34)</td><td>0.14 (−0.66)</td></tr></table>

![](images/5bfdf65d0ece280521c911ed399149412bc63c19f20910be382a7d63267b1e28.jpg)  
Figure 8: Baseline (dense) design: the patch length $\widehat { p } ^ { \star }$ with the lowest mean population excess risk (blue) against $\sqrt { L }$ for nine window lengths, with the through-origin fit $p = \widehat { C } \widehat { \sqrt { L } }$ and its 95% bootstrap band (orange). Panels: horizons, with centered $R ^ { 2 }$