# Stability of Measure-to-Measure Transformers on Sub-Gaussian Data

Frank Cole UCLA fcole99@g.ucla.edu

Nicholas H. Nelsen UT Austin nnelsen@oden.utexas.edu

Takashi Furuya Doshisha University, RIKEN AIP tfuruya@mail.doshisha.ac.jp

## Abstract

Transformers have exhibited impressive empirical success across various domains, but their theoretical foundations remain less developed. This work constitutes a mathematical study of the measure-to-measure operators defined by transformers. We show that transformers map sub-Gaussian inputs to sub-Gaussian outputs; this ensures that taking arbitrarylength compositions of the softmax operator is well-defined. We then show that transformers are H¨older continuous with respect to the 1-Wasserstein distance on appropriate spaces of sub-Gaussian inputs. This allows us to establish estimates on the error propagation along a transformer between a sub-Gaussian measure and its empirical approximation. We also study a mean-field analog of the crossattention mechanism, which is an operator from a pair of probability measures to a single probability measure. We show that crossattention exhibits diferent H¨older regularity and sample-complexity in its two input arguments. Last, we apply our results to deduce approximation guarantees for measureto-measure transformers. Together, these results provide a firm stability and finite-sample theory for transformers on sub-Gaussian data.

## 1 INTRODUCTION

Transformers (Vaswani et al., 2017) are the backbone of the modern generative artificial intelligence (GenAI) revolution and the architecture behind large language models like ChatGPT (Achiam et al., 2023). They also play a burgeoning role in scientific discovery through the development of scientific foundation models (Bodnar et al., 2025; Herde et al., 2024; Horawalavithana et al., 2022; Subramanian et al., 2023; Takeda et al., 2023; Wiesner et al., 2025). While the scale and applications of GenAI continue to grow, our theoretical understanding of these models is far less mature. This paper falls within a recent research area which studies transformers from a mathematical perspective, with the hope of eventually narrowing the gap between theory and practice. By developing a rigorous mathematical theory of GenAI architectures, one hopes to reveal the structure behind both their successes and failures. One can also leverage these theoretical insights to design more reliable and robust algorithms for GenAI and machine learning more generally.

The analysis in this paper is based on the mean-field or measure-theoretic perspective on encoder-only transformers (Castin et al., 2025; Furuya et al., 2025b; Geshkovski et al., 2025; Rigollet, 2026; Sander et al., 2022; Vuckovic et al., 2020): the output of a transformer is a function of the empirical probability distribution of its input tokens. There are several motivations for studying transformers from this perspective. First, by lifting the analysis of transformers from a discrete to a continuum setting, we unlock powerful and welldeveloped mathematical tools for understanding the behavior of transformers and machine learning on spaces of probability measures more broadly. Second, highdimensional data often admit natural interpretations as collections of probability measures. Measure-valued data arises, for instance, in physical sciences, where swarms of particles interact according to a time-varying probability distribution, or in neural surrogates for probabilistic models, where the goal is to train a network that directly maps problem conditions (e.g., prior distributions and observations) to solutions. Last, AI methods for such problems have achieved substantial empirical success, but lack the theoretical grounding necessary to engender reliability and trust.

While recent work in this area studies transformers as in-context or contextual mappings (Furuya et al., 2025b)—i.e., vector-to-vector mappings which depend on a probability measure or context—our work focuses on the measure-to-measure operators defined by transformers (Furuya et al., 2025a, 2026b; Geshkovski et al., 2026; Lavenant and Savar´e, 2026; Vandergrift et al., 2026), which we refer to as mean-field transformer operators. The goal of this paper is to answer fundamental questions about the mathematical properties of these operators, particularly for input probability measures which have unbounded support. Prior theoretical results focused primarily on bounded data (Castin et al., 2024; Vuckovic et al., 2021), for which the answers to these questions difer substantially. To this end, sub-Gaussian distributions are useful for modeling distributions of tokens which occasionally take very large values, but otherwise remain bounded. Additionally, many probability distributions arising naturally in machine learning and scientific computing, such as posterior distributions for Bayesian inverse problems with Gaussian priors (Stuart, 2010) and data distributions along the forward/backward processes of a score-based generative model (Song et al., 2021), often exhibit fast tail decay, but necessarily have unbounded support. At the same time, when working with distributions of unbounded support, some assumption of tail decay is necessary to ensure that the softmax operation is well-defined. Our work finds that sub-Gaussianity is a suficient tail condition for the stability of mean-field transformer operators, while being broad enough not to preclude the examples of interest mentioned.

## 1.1 Main Contributions

Our main contributions are stated as follows.

(C1) Well-definedness. We show that if $\mu$ is sub-Gaussian and T is an arbitrarily deep meanfield transformer, then ${ \mathsf { T } } ( \mu , \cdot ) _ { \# } \mu$ remains sub-Gaussian, ensuring that the softmax normalizing constant remains finite across layers. This result thus identifies sub-Gaussianity as a suficient condition for the image of measures with unbounded support under mean-field transformers to be welldefined; see Section 3.1 for precise results.

(C2) Regularity. We show that the mapping $\mu \mapsto$ ${ \mathsf { T } } ( \mu , \cdot ) _ { \# } \mu$ is H¨older continuous on appropriate subsets of sub-Gaussian measures with respect to the 1-Wasserstein metric; see Section 3.2 for precise results.

(C3) Sample complexity. We estimate the expected distance between a sub-Gaussian measure $\mu$ and its empirical counterpart $\mu _ { N }$ under a mean-field transformer in the 1-Wasserstein distance. Our estimate is of the form

$$
{ \mathbb E } \Big [ { \mathbb W } _ { 1 } \big ( { \mathbb T } ( \mu , \cdot ) _ { \# } \mu , { \mathbb T } ( \mu _ { N } , \cdot ) _ { \# } \mu _ { N } \big ) \Big ] = \widetilde O \big ( N ^ { - \gamma / d } \big ) ,
$$

where $\gamma$ is the H¨older exponent of the mapping $\mu \mapsto { \mathsf { T } } ( \mu , \cdot ) _ { \# } \mu ;$ see Section 3.3 for precise results. As an auxiliary result which is possibly of independent interest, we prove concentration inequalities for the sub-Gaussian parameters of $\mu _ { N }$ (Proposition B.7).

(C4) Cross-attention. We also study mean-field transformers with the cross-attention mechanism, which leads to a mapping of the form $( \mu , \eta ) \mapsto { \sf T } ( \mu , \cdot ) _ { \# } \eta .$ . We show that mean-field cross-attention is jointly H¨older continuous, and that the diferent variables exhibit diferent regularity. We also prove a sample complexity result for this mapping under the assumption that $\mu$ is compactly supported and η is sub-Gaussian; see Section 3.4 for precise results.

(C5) Measure-to-measure approximation. By pairing our sample complexity estimates with recent approximation-theoretic results (Furuya et al., 2026b), we prove that mean-field transformer operators universally approximate continuous measure-to-measure operators on certain $\mathsf { W } _ { 1 }$ -compact subsets of ${ \mathcal { P } } ( \mathbb { R } ^ { d } )$ , even when the transformer only has access to samples from the input measure. Section 3.5 has the precise results.

Our work thus provides initial understanding of the behavior of measure-to-measure transformers on unbounded data. The results in this paper may be of use in other applications, such as to study the generalization properties of transformer-based machine learning models on the space of probability measures.

## 1.2 Related Work

The mean-field perspective on transformers has received significant recent research attention.

Mean-field dynamics of transformers. The works of Geshkovski et al. (2023, 2025, 2026) pioneered the interpretation of infinitely-deep transformers as interacting particle systems, where residual connections are viewed as steps of an Euler method. To better understand these dynamics as the number of tokens increases, several works have studied the corresponding meanfield dynamics of transformers (Rigollet, 2026). Recent works in mean-field dynamics of transformers study the emergence of clusters (Bruno et al., 2025; Chen et al., 2025; Karagodin et al., 2024; Li et al., 2026), softmax temperature scaling (Bruno et al., 2026a; Chen et al., 2026b), multi-scale analysis (Bruno et al., 2026b), generalizations to multi-head attention (Pendharkar, 2026), and the efects of non-compactly supported initial data (Castin et al., 2025), among other topics.

Regularity of transformers. Several works have established the Lipschitz regularity of mean-field selfattention (Castin et al., 2024; Kim et al., 2021; Vuckovic et al., 2021); these results show that (dot product) attention is not Lipschitz unless measures are restricted to have common compact support, or the negative squared distance is used in the place of the dot product to compute attention scores. The work of Mroueh (2023) studies the Lipschitz regularity of in-context learning in the mean-field setting, with transformers as a representative example. For diferent perspectives on the regularity of attention, see work of Sander et al. (2022), which connects attention with Sinkhorn normalization to heat difusion, Kan et al. (2026), which interprets regularized transformers as solutions to optimal control problems, and Kan et al. (2025), which studies the Lipschitz stability of transformers induced by layer normalization. Closely related to our work is that of Hargenrader et al. (2026), who study the integrability of post-norm transformers (beyond softmax attention) when applied to heavy-tailed input measures; in particular, such measures have unbounded support.

Sample complexity of transformers. Bohbot et al. (2026); Boursier and Boyer (2026); Chen et al. (2026a) study the sample complexity of mean-field transformers. Their contributions share a similar flavor with our results in Section 3.3: in particular, Bohbot et al. (2026); Boursier and Boyer (2026) prove concentration inequalities for attention with sub-Gaussian inputs. The main diference is that their results study the sample complexity of the in-context map associated with transformers, i.e., the mapping $x \mapsto { \mathsf { T } } ( \mu , x )$ , whereas our results are concerned with the sample complexity of the measure-to-measure operators $\mu \mapsto \mathsf T ( \mu , \cdot ) _ { \# } \mu$ defined by transformers T. Thus, the results of Bohbot et al. (2026); Boursier and Boyer (2026); Chen et al. (2026a) are not directly comparable with our results.

Learning on spaces of probability measures. Mean-field transformers fall within the wider area of machine learning on spaces of probability measures (Le´on and Niles-Weed, 2026; Panaretos and Zemel, 2020; Yao et al., 2026; Zaheer et al., 2017). Applications include regression on point cloud data (Cai et al., 2020; Vandergrift et al., 2026), training distribution design (Guerra et al., 2025), amortized solvers for mean-field games and optimal transport (Cole et al., 2026; Huang and Lai, 2025), amortized conditioning and generative modeling (Fishman et al., 2026; Generale et al., 2026; Gloeckler et al., 2024; Tsimpos et al., 2026; Whittle et al., 2026), and inverse problems (Nelsen and Yang, 2026) such as non-Gaussian Bayesian data assimilation (Al-Jarrah et al., 2025; Bach et al., 2026, 2025; Wang et al., 2026).

## 1.3 Notation

For a natural number $N \in \mathbb { N } .$ , we denote by $[ N ] : =$ $\{ 1 , \ldots , N \}$ . For a vector $x \in \mathbb { R } ^ { d }$ , the number ∥x∥ denotes the Euclidean norm; for a matrix $A \in \mathbb { R } ^ { d \times k }$ the number $\| A \|$ denotes the operator norm of $A$ with respect to the Euclidean norms on $\mathbb { R } ^ { d }$ and $\mathbb { R } ^ { k }$ . For $R > 0$ , we let $B _ { R } ( \mathbb { R } ^ { d } ) : = \{ x \in \mathbb { R } ^ { d } \colon \| x \| \leq R \}$ denote the closed ball of radius $R$ in $\mathbb { R } ^ { d } ;$ when the dimension $d$ is clear from context, we simply write $B _ { R } ( \mathbb { R } ^ { d } ) = B _ { R }$ We define $\log _ { + } ( R ) : = \operatorname* { m a x } ( 0 , \log ( R ) )$ , where log is the natural logarithm. Given a real-valued function $f ,$ we denote by $\operatorname { L i p } ( f )$ the (minimal) Lipschitz constant of $f ,$ with the convention that $\operatorname { L i p } ( f ) = \infty { \mathrm { ~ i f ~ } } f$ is not Lipschitz continuous. Given a metric space $x ,$ let $\mathcal { P } ( \mathcal { X } )$ be the set of Borel probability measures on $\mathcal { X }$ . Given $\mu$ and ν in $\mathcal { P } ( \mathcal { X } )$ with finite first moment, we denote their 1-Wasserstein distance $\mathsf { W } _ { 1 } ( \mu , \nu )$ by

$$
\begin{array} { l } { \displaystyle \mathsf { W } _ { 1 } ( \mu , \nu ) : = \operatorname* { i n f } \bigg \{ \int _ { \mathcal { X } \times \mathcal { X } } \| x - y \| \gamma ( d x , d y ) \colon \gamma \in \Pi ( \mu , \nu ) \bigg \} } \\ { \displaystyle = \operatorname* { s u p } \bigg \{ \int _ { \mathcal { X } } f d \mu - \int _ { \mathcal { X } } f d \nu \colon \mathrm { L i p } ( f ) \le 1 \bigg \} , } \end{array}
$$

where $\Pi ( \mu , \nu )$ denotes the set of couplings of $\mu$ and $\nu .$ Given $\mu \in { \mathcal { P } } ( { \mathcal { X } } )$ on a set $\mathcal { X }$ and a function $T$ on $x ,$ we denote by $T _ { \# } \mu$ the pushforward measure. Given a measurable space X and a measurable subset $\mathcal A \subseteq \mathcal X$ we denote by $\mathbf { 1 } _ { \mathcal { A } } \colon \mathcal { X } \to \{ 0 , 1 \}$ the indicator function of ${ \mathcal { A } } .$ Given positive functions $f$ and $^ { g , }$ we use the notation $f ( x ) = O ( g ( x ) )$ or $f ( x ) \lesssim g ( x )$ if there exists a positive constant $C$ such that $f ( x ) \leq C g ( x )$ for all x in a set of interest that should be clear from the context. We also write $f ( x ) = \Omega ( g ( x ) )$ if $g ( x ) = O ( f ( x ) )$ ). We often use $C$ to denote a positive constant whose value changes from line to line.

## 2 BACKGROUND

This section provides the mathematical and probabilistic preliminaries on measure-theoretic transformers.

## 2.1 Mean-Field Transformers

We begin with a review of the traditional self-attention mechanism. Let $X = ( x _ { 1 } , \ldots , x _ { N } ) \subset \mathbb { R } ^ { d }$ denote an input sequence. A single-head self-attention module takes X as an input and returns an output sequence $\operatorname { A t t } _ { \theta } ( X ) \subset \mathbb { R } ^ { k }$ defined by

$$
\mathrm { A t t } _ { \theta } ( X ) _ { i } = \frac { \sum _ { j = 1 } ^ { N } \exp \bigl ( \langle x _ { i } , A x _ { j } \rangle \bigr ) V x _ { j } } { \sum _ { k = 1 } ^ { N } \exp \bigl ( \langle x _ { i } , A x _ { k } \rangle \bigr ) } \mathrm { f o r } i \in [ N ] .
$$

Here, $V \in \mathbb { R } ^ { k \times d }$ and $A \in \mathbb { R } ^ { d \times d }$ denote the learnable parameters, and $\theta = ( A , V )$ . In short, an attention head updates tokens by computing a weighted average of $\{ V x _ { 1 } , \ldots , V x _ { N } \}$ , where the weights are determined by the attention scores $\{ \exp ( \langle x _ { i } , A x _ { j } \rangle ) \} _ { ( i , j ) \in [ N ] ^ { 2 } }$ . By defining the in-context map

$$
{ \mathrm { A t t } } _ { \theta } ( X , x ) = { \frac { \sum _ { j = 1 } ^ { N } \exp \bigl ( \langle x , A x _ { j } \rangle \bigr ) V x _ { j } } { \sum _ { k = 1 } ^ { N } \exp \bigl ( \langle x , A x _ { k } \rangle \bigr ) } }\tag{2.1}
$$

associated with the attention layer, we can write the i-th token of the output sequence as ${ \mathrm { A t t } } _ { \theta } ( X ) _ { i } \ =$ $\operatorname { A t t } _ { \theta } ( X , x _ { i } )$ . The function in (2.1) is called an in-context map because it depends on the input sequence (or context) as well as x. Notice that the in-context map depends on the sequence X only through the empirical measure $\begin{array} { r } { \mu _ { N } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \delta _ { x _ { i } } } \end{array}$ . This motivates us to define a mean-field self-attention head as a mapping ${ \mathrm { A t t } } _ { \theta } \colon K \times \mathbb { R } ^ { d } \to \mathbb { R } ^ { k }$ with $K \subset \mathcal { P } ( \mathbb { R } ^ { d } )$ given by

$$
{ \mathrm { A t t } } _ { \theta } ( \mu , x ) = { \frac { \int \exp \bigl ( \langle x , A y \rangle \bigr ) V y \mu ( d y ) } { \int \exp \bigl ( \langle x , A z \rangle \bigr ) \mu ( d z ) } } .\tag{2.2}
$$

This formulation is valid for all probability measures for which the above integrals converge and recovers the traditional attention mechanism when µ is an empirical measure. Hereafter, for $K \subset \mathcal { P } ( \mathbb { R } ^ { d } )$ , we refer to any mapping $K \times \mathbb { R } ^ { d } \to \mathbb { R } ^ { k }$ as an in-context map. A multihead mean-field self-attention layer is an in-context map $\Gamma _ { \theta } \colon K \times  { \mathbb { R } ^ { d } } \to  { \mathbb { R } ^ { d } }$ of the form

$$
\Gamma _ { \theta } ( \mu , x ) = x + \sum _ { h = 1 } ^ { H } W _ { h } \mathrm { A t t } _ { \theta _ { h } } ( \mu , x ) ,\tag{2.3}
$$

where $W _ { 1 } , \dots , W _ { H } \in \mathbb { R } ^ { d \times k }$ , each $\mathrm { A t t } _ { \theta _ { h } }$ is given by $( 2 . 2 )$ and the learnable parameters are $\boldsymbol { \theta } \stackrel { \cdot \cdot } { = } \{ ( \theta _ { h } , W _ { h } ) ) \} _ { h = 1 } ^ { H } .$ Equation (2.3) is a direct generalization of the standard multi-head self-attention module for sequences to the measure-theoretic setting. Throughout the rest of this work, θ will be fixed or clear from context; we may simply abbreviate $\operatorname { A t t } _ { \theta }$ to Att and $\Gamma _ { \theta }$ to Γ.

Given two in-context mappings $G _ { 1 } \colon K _ { 1 } \times \mathbb { R } ^ { d _ { 1 } } \to \mathbb { R } ^ { d _ { 2 } }$ and $G _ { 2 } \colon K _ { 2 } \times \mathbb { R } ^ { d _ { 2 } } \to \mathbb { R } ^ { d _ { 3 } }$ with $K _ { 1 } \subset \mathcal { P } ( \mathbb { R } ^ { d _ { 1 } } )$ such that $G _ { 1 } ( \mu , \cdot ) _ { \# } \mu \in K _ { 2 }$ for all $\mu \in K _ { 1 }$ , there is a natural operation to produce a third in-context mapping $( G _ { 2 } \diamond$ $\bar { G } _ { 1 } ) \colon K _ { 1 } \times \mathbb { R } ^ { \bar { d } _ { 1 } } \to \mathbb { R } ^ { d _ { 3 } }$ defined by

$$
( G _ { 2 } \diamond G _ { 1 } ) ( \mu , x ) = G _ { 2 } \bigl ( G _ { 1 } ( \mu , \cdot ) _ { \# } \mu , G _ { 1 } ( \mu , x ) \bigr ) .
$$

The operation $^ \circ \circ$ is associative. Thus, it allows us to define compositions of in-context maps. A meanfield transformer is an in-context mapping defined by composing multi-head self-attention layers and feedforward layers. Recall that a (shallow) feedforward neural network is a function $\phi \colon \bar { \mathbb { R } ^ { d } }  \mathbb { R } ^ { k }$ of the form

$$
\phi ( x ) = W _ { 2 } \mathrm { R e L U } ( W _ { 1 } x ) ,\tag{2.4}
$$

where $W _ { 1 } \in \mathbb { R } ^ { m \times d } , W _ { 2 } \in \mathbb { R } ^ { k \times m }$ , and the activation function ReL $\mathrm { { I U } : \mathbb { R }  [ 0 , \infty ) }$ defined by ${ \mathrm { R e L U } } ( x ) =$ max $( 0 , x )$ is applied component-wise to the vector $W _ { 1 } x .$

A mean-field transformer is then defined as an incontext mapping of the form

$$
{ \sf T } = \left( \phi _ { L } \diamond \Gamma _ { \theta _ { L } } \diamond \cdot \cdot \cdot \diamond \phi _ { 1 } \diamond \Gamma _ { \theta _ { 1 } } \right) \quad \mathrm { f o r } \quad L \in \mathbb { N } ,\tag{2.5}
$$

where each $\Gamma _ { \theta _ { i } }$ has the form (2.3) and each $\phi _ { i }$ is a ReLU network of the form (2.4), which we identify with the (constant in $\mu )$ in-context map $( \mu , x ) \mapsto \phi _ { i } ( x )$ Hereafter, we use $\cdot \mathsf { T } ^ { \star }$ to denote a general mean-field transformer of the form (2.5), whereas $\cdot \Gamma ^ { \prime }$ refers to a single multi-head self-attention layer. Furuya et al. (2025b) proves that the set of mean-field transformers is dense in the set of continuous in-context mappings from $\mathcal { P } ( \Omega ) \times \Omega$ to $\mathbb { R } ^ { k }$ , where $\Omega \subset \mathbb { R } ^ { d }$ is compact.

Remark 2.1. In (2.5), we could also allow the ReLU networks $\phi _ { 1 } , \ldots , \phi _ { L }$ to be arbitrarily deep. However, since multi-head attention layers can represent the identity map, there is no generality lost in restricting $\phi _ { 1 } , \ldots , \phi _ { L }$ to be shallow.

Remark 2.2. Our sub-Gaussian analysis omits normalization layers (Ba et al., 2016). The main obstruction presented by layer norm (LN) is that it breaks the continuity of transformers. Developing analogs of our results which are compatible with LN is an interesting problem for future work. Nonetheless, although pre-LN transformers only operate on compactly supported measures (Kan et al., 2025), our work is relevant for understanding post-LN transformers that do accommodate unbounded support in the initial input layer (Hargenrader et al., 2026). While often harder to train, post-LN models can exhibit improved performance in some settings (Liu et al., 2020).

A mean-field transformer T : $K \times \mathbb { R } ^ { d } \to \mathbb { R } ^ { k }$ naturally defines an operator from K to ${ \mathcal { P } } ( \mathbb { R } ^ { k } )$ by

$$
\mu \mapsto \mathsf { T } ( \mu , \cdot ) _ { \# } \mu .
$$

We refer to this operator as a mean-field transformer operator. Notice that if ${ \sf T } _ { 1 }$ and ${ \sf T } _ { 2 }$ are mean-field transformers (or more general in-context mappings) for which $( \mathsf { T } _ { 2 } \circ \mathsf { T } _ { 1 } )$ is well-defined, then

$$
( \mathsf T _ { 2 } \diamond \mathsf T _ { 1 } ) ( \mu , \cdot ) _ { \# } \mu = \mathsf T _ { 2 } ( \mathsf T _ { 1 } ( \mu , \cdot ) _ { \# } \mu , \cdot ) _ { \# } ( \mathsf T _ { 1 } ( \mu , \cdot ) _ { \# } \mu ) .
$$

In other words, given two mean-field transformers, the in-context composition rule corresponds to the standard composition rule for their corresponding measure-tomeasure operators.

## 2.2 Sub-Gaussian Probability Distributions

Sub-Gaussian probability distributions are central objects in high-dimensional probability and machine learning which generalize the tail decay properties of the Gaussian distribution. Given $\alpha \geq 1$ and $\beta > 0$ , we define the class of (α, β)-sub-Gaussian measures on $\mathbb { R } ^ { d }$ , denoted by $\mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ , to consist of all measures $\mu \in \mathcal P ( \mathbb { R } ^ { d } )$ with square-exponential tail decay:

$$
\operatorname* { s u p } _ { R > 0 } \mu \big ( \{ x \in \mathbb { R } ^ { d } \colon \| x \| > R \} \big ) e ^ { \frac { R ^ { 2 } } { \beta } } \leq \alpha .
$$

The sets $\{ \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) \} _ { \alpha \geq 1 , \beta > 0 }$ are monotonic, $\mathrm { i . e . , }$ if $\alpha _ { 1 } ~ \leq ~ \alpha _ { 2 }$ and $\beta _ { 1 } ~ \leq ~ \beta _ { 2 } .$ , then $\mathrm { S G } _ { \alpha _ { 1 } , \beta _ { 1 } } ( \mathbb { R } ^ { d } ) \subseteq$ $\mathrm { S G } _ { \alpha _ { 2 } , \beta _ { 2 } } ( \mathbb { R } ^ { d } )$ . We then define

$$
\operatorname { S G } ( \mathbb { R } ^ { d } ) : = \bigcup _ { \alpha \geq 1 , \beta > 0 } \operatorname { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) .
$$

This definition coincides with the classical definition of sub-Gaussianity, i.e., $\mu \in \operatorname { S G } ( \mathbb { R } ^ { d } )$ if and only if $X \sim \mu$ has finite sub-Gaussian norm (Vershynin, 2018, Chapter 2). If $\mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ , then we refer to α and $\beta$ as the sub-Gaussian parameters of $\mu .$

## 3 MAIN RESULTS

This section develops contributions (C1)–(C5).

Remark 3.1. In this paper, most of our upper bounds on the $\mathsf { W } _ { 1 }$ metric are based on coupling arguments. Thus, we expect that our results also hold for $\mathsf { W } _ { p }$ with $p > 1$ , up to constant factors which depend on $p .$ We leave this extension to future work.

## 3.1 Transformers Preserve Sub-Gaussianity

As alluded to previously, defining mean-field transformers on probability measures with unbounded support requires care. Indeed, if $\operatorname { A t t } _ { \theta }$ is a self-attention layer with parameters A and $V$ , then $\operatorname { A t t } _ { \theta } ( \mu , x )$ is well-defined if and only if the integrals

$$
\int e ^ { \langle x , A y \rangle } V y \mu ( d y ) \quad { \mathrm { a n d } } \quad \int e ^ { \langle x , A y \rangle } \mu ( d y )
$$

converge. If these integrals converge for every $x \in \mathbb { R } ^ { d }$ then $\mu$ necessarily has sub-exponential tails. The matter is further complicated by considering compositions of attention layers. Indeed, it could be the case that $\mu$ is suficiently light-tailed to make $\operatorname { A t t } _ { \theta } ( \mu , \cdot )$ well-defined, but $\operatorname { A t t } _ { \theta } ( \mu , \cdot ) _ { \# } \mu$ is heavy-tailed. To circumvent these concerns, we would like to identify a set $K \subset \mathcal { P } ( \mathbb { R } ^ { d } )$ on which transformers are well-defined and stable, in the sense that $\Psi ( K ) \subset K$ for any mean-field transformer operator Ψ. Below, we establish $\operatorname { S G } ( \mathbb { R } ^ { d } )$ as having this desired property. The key insight is that if $\mu \in \mathrm { S G } ( \mathbb { R } ^ { d } )$ 2 then $\operatorname { A t t } _ { \theta } ( \mu , \cdot )$ grows at most linearly.

Proposition 3.2. Fix $\alpha \geq 1$ and $\beta > 0$ . Let Γ be a mean-field self-attention operator given by

$$
\Gamma ( \mu , x ) = x + \sum _ { h = 1 } ^ { H } W _ { h } \frac { \int e ^ { \langle x , A _ { h } y \rangle } V _ { h } y \mu ( d y ) } { \int e ^ { \langle x , A _ { h } y \rangle } \mu ( d y ) } .\tag{3.1}
$$

Then there exist constants $K _ { 1 } , K _ { 2 } > 0$ , depending only on $\alpha , \beta , H$ , and the attention weights, such that

$$
\operatorname* { s u p } _ { \mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) } \| \Gamma ( \mu , x ) \| \leq K _ { 1 } + K _ { 2 } \| x \| \quad f o r \ a l l \quad x \in \mathbb { R } ^ { d } .
$$

Castin et al. (2025) established that when $\mu$ is a Gaussian measure, the mapping $x \mapsto \operatorname { A t t } _ { \theta } ( \mu , x )$ is linear. As a consequence, $\mu \mapsto \operatorname { A t t } _ { \theta } ( \mu , \cdot ) _ { \# } \mu$ sends Gaussian measures to Gaussian measures. In contrast, our result asserts that when $\mu$ is sub-Gaussian, the self-attention map $\operatorname { A t t } _ { \theta } ( \mu , \cdot )$ grows afine-linearly; thus, its pushforward preserves sub-Gaussianity.

A consequence of Proposition 3.2 is that mean-field transformers map $\operatorname { S G } ( \bar { \mathbb { R } } ^ { d } )$ to $\operatorname { S G } ( \mathbb { R } ^ { k } )$

Theorem 3.3. Let T be a mean-field transformer. Then $\mathsf { T } ( \mu , \cdot ) _ { \# } \mu \in \mathrm { S G } ( \mathbb { R } ^ { k } )$ for every $\mu \in \mathrm { S G } ( \mathbb { R } ^ { d } )$

Proof. Let $\mu \in \mathrm { S G } ( \mathbb { R } ^ { d } )$ . By Proposition 3.2 and the fact that ReLU feedforward networks are Lipschitz continuous, there exist constants $K _ { 1 } , K _ { 2 } > 0$ (depending on the parameters of T and the sub-Gaussian parameters of $\mu )$ such that $\| \mathsf { T } ( \mu , x ) \| \leq K _ { 1 } + K _ { 2 } \| x \|$ for all $\boldsymbol { x } \in \mathbb { R } ^ { d }$ . The conclusion follows by Lemma B.2. □

Theorem 3.3 identifies sub-Gaussianity as a natural tail assumption for mean-field transformers. Importantly, it applies not only to individual attention layers, but also to arbitrarily deep transformers.

## 3.2 H¨older Regularity of Transformers on Sub-Gaussian Domains

Understanding the regularity of neural architectures is important, for instance, to ensure their robustness under corruptions of the data, to characterize the functions which they can eficiently approximate, and to prove generalization bounds. In this subsection, we study the regularity of measure-to-measure operators of the form $\mu \mapsto \mathsf T ( \mu , \cdot ) _ { \# } \mu$ , where T is a mean-field transformer. With the results of the previous subsection in mind, we restrict the domain of these operators to be $\mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ for a fixed (but arbitrary) pair $( \alpha , \beta ) \in [ 1 , \infty ) \times ( 0 , \infty )$ . Previous works have studied the regularity of such mappings on the domain $\mathcal { P } ( B _ { R } )$ ， where it was shown that mean-field self-attention operators are Lipschitz continuous (Castin et al., 2024). In particular, these results established lower bounds on the Lipschitz constant of a mean-field self-attention map $\operatorname { A t t } _ { \theta }$ of the form

$$
\mathrm { L i p } _ { \mathcal { P } ( B _ { R } ) } ( \mathrm { A t t } _ { \theta } ) = \Omega \Big ( e ^ { \Omega ( R ^ { 2 } ) } \Big ) .
$$

These lower bounds suggest that mean-field transformer operators might not be Lipschitz continuous on $\mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ , since $\mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ contains measures with unbounded support. Instead, we prove that meanfield transformer operators are H¨older continuous on $\mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ . We first state a result for a multi-head selfattention layer, where we have a precise (but possibly sub-optimal) estimate of the H¨older exponent.

Theorem 3.4. Let Γ: $\mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) \times \mathbb { R } ^ { d } \to \mathbb { R } ^ { d }$ be a mean-field self-attention operator as in (3.1). $D e \mathrm { - }$ fine $c _ { A } : = \textstyle \operatorname* { m a x } _ { h \in [ H ] } \| A _ { h } \| , c _ { V } : = \operatorname* { m a x } _ { h \in [ H ] } \| V _ { h } \|$ , and $c _ { W } : = \operatorname* { m a x } _ { h \in [ H ] } \| W _ { h } \|$ . Further, define

$$
\begin{array} { c } { k _ { 1 } : = 5 c _ { A } , k _ { 2 } : = \operatorname* { m i n } \biggr ( \frac { 1 } { \beta } , \frac { 1 } { 2 ( \beta ^ { 3 } \alpha ^ { 2 } c _ { A } ^ { 2 } + 1 ) } \biggr ) , } \\ { \gamma : = \frac { k _ { 2 } } { k _ { 1 } + k _ { 2 } } . } \end{array}
$$

Then there exists a constant C, depending on $\alpha , \beta , H , c _ { A } , c _ { V }$ , and $c _ { W }$ , such that, for all $\epsilon > 0$

$$
\mathsf { W } _ { 1 } \big ( \Gamma ( \mu , \cdot ) _ { \# } \mu , \Gamma ( \mu _ { 0 } , \cdot ) _ { \# } \mu _ { 0 } \big ) \le C \log _ { + } ( \epsilon ^ { - 1 } ) \epsilon ^ { \gamma }
$$

holds for all $\mu$ and $\mu _ { 0 }$ in $\mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ with $\mathsf { W } _ { 1 } ( \mu , \mu _ { 0 } ) \le \epsilon$ In particular, by enlarging the constant and decreasing the H¨older exponent too, it follows that there exists $C ( \alpha , \beta , H , c _ { A } , c _ { V } ) > 0$ and $\gamma ^ { \prime } \in ( 0 , \gamma )$ such that

$$
\mathsf { W } _ { 1 } \big ( \Gamma ( \mu , \cdot ) _ { \# } \mu , \Gamma ( \mu _ { 0 } , \cdot ) _ { \# } \mu _ { 0 } \big ) \le C \mathsf { W } _ { 1 } ( \mu , \mu _ { 0 } ) ^ { \gamma ^ { \prime } }
$$

for all $\mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ and all $\mu _ { 0 } \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$

Theorem 3.4 is proven by truncating the supports of $\mu$ and $\mu _ { 0 } .$ controlling the $\mathsf { W } _ { 1 }$ error between truncated versions of $\mu$ and $\mu _ { 0 }$ , and trading of the truncation error with the Lipschitz constant. The details can be found in Appendix A.2. The dependence of the H¨older exponent on $\| A \|$ is unavoidable from our proof strategy because the Lipschitz constant of $\mu \mapsto \Gamma ( \mu , \cdot ) _ { \# } \mu$ on $\mathcal { P } ( B _ { R } )$ is $\Omega ( e ^ { \Omega ( c _ { A } R ^ { 2 } ) } )$ , while for $\mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ , the Wasserstein distance between $\mu$ and its truncation to $\mathcal { P } ( B _ { R } )$ is $O ( e ^ { - R ^ { 2 } / \beta } )$ However, we do not attempt to optimize the exponent $\gamma$ in our proof. The final estimate crudely removes the log factor by enlarging the H¨older constant and reducing the H¨older exponent, thus establishing a bona fide H¨older continuity estimate of the mean-field multi-head self-attention layer.

We extend the H¨older regularity to general mean-field transformers, though the generality of the statement requires sacrificing control on the H¨older exponent.

Theorem 3.5. Fix $ { d ^ { \mathrm { ~ ~ ~ } } } \in  { \mathbb { N } }$ with $d \_ \mathrm { ~ { ~ \scriptsize ~ 3 ~ } ~ }$ . Let $\mathsf { T } \colon \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) \times \mathbb { R } ^ { d } \to \mathbb { R } ^ { k }$ be a mean-field transformer. Then there exist constants $C > 0$ and $\gamma \in ( 0 , 1 )$ (depending on T, α, and $\beta )$ such that

$$
\mathsf { W } _ { 1 } \big ( \mathsf { T } ( \mu , \cdot ) _ { \# } \mu , \mathsf { T } ( \mu _ { 0 } , \cdot ) _ { \# } \mu _ { 0 } \big ) \le C \epsilon ^ { \gamma }
$$

holds for all $\mu$ and µ<sub>0</sub> in $\mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ with $\mathsf { W } _ { 1 } ( \mu , \mu _ { 0 } ) \le \epsilon .$

The proof can be found in Appendix A.2. Theorem 3.5 complements the results of Section 3.1 by characterizing the smoothness of mean-field transformers on sub-Gaussian data. A natural application of these regularity results is to generalization error estimates for learning—e.g., via empirical risk minimization—on the space of probability measures with mean-field transformers. We leave this direction to future work.

Remark 3.6. The assumption that $d \geq 3$ in Theorem 3.7 is only made to simplify the presentation and avoid a more complex case-based result. In more detail, the proof of Theorem $3 . 7$ uses a Wasserstein law of large numbers of the form $\mathbb { E } \mathsf { W } _ { 1 } ( \mu , \mu _ { N } ) \lesssim N ^ { - 1 / d }$ , for which the cases $d = 1$ and $d = 2$ are exceptional. Our bound directly generalizes to these cases, but we choose to present the result in its simplest form.

## 3.3 Sample Complexity of Transformers

While the mean-field framework is a convenient setting to mathematically study transformers, probability measures must be quantized in practice. A natural means of quantization is independent and identically distributed (iid) sampling. As an application of our regularity results, we estimate the sample complexity of mean-field transformer operators in the iid setting. Given $\mu \in \mathcal P ( \mathbb { R } ^ { d } )$ , we use $\mu _ { N }$ to denote the (random) empirical measure associated with $\mu .$ That is, if $x _ { n } \sim \mu$ for $n \in [ N ]$ are iid, then $\begin{array} { r } { \mu _ { N } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \delta _ { x _ { n } } } \end{array}$ Theorem 3.7. Fix $d \in \mathbb { N }$ with $d \ge 3 , \alpha \ge 1$ , and $\beta > 0$ . Let T be a mean-field transformer. Let $\gamma \in ( 0 , 1 )$ be a H¨older exponent of T on $\mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ . Then there exist constants $C > 0$ and $C ^ { \prime } \geq 1$ , depending on ${ \mathsf { T } } , \alpha ,$ $\beta , \gamma$ , and $d ,$ such that

$$
\begin{array} { r l } {  { \operatorname* { s u p } _ { \mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) } \mathbb { E } \mathsf { W } _ { 1 } \big ( \mathsf { T } ( \mu , \cdot ) _ { \# } \mu , \mathsf { T } ( \mu _ { N } , \cdot ) _ { \# } \mu _ { N } \big ) } \quad } & { } \\ & { \leq C \log ( N ) ^ { \gamma / 2 } \Big ( 1 + \log \big ( 1 + N ^ { 1 / d } \big ) \Big ) ^ { \gamma / 2 } N ^ { - \gamma / d } } \end{array}
$$

for all $N \geq C ^ { \prime }$

Theorem 3.7 does not follow simply by applying the H¨older regularity of T. The issue is that the empirical measure $\mu _ { N }$ may not belong to $\mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ . To deal with this, we first prove that when N is suficiently large, there exist constants $\widehat { \alpha } \geq 1$ and ${ \widehat { \beta } } > 0$ , which depend on α and $\beta$ but are independent of $N$ , such that $\mu _ { N } \in \mathrm { S G } _ { \widehat { \alpha } , \widehat { \beta } } ( \mathbb { R } ^ { d } )$ with high probability; see Proposition B.7 for a precise statement. We denote the event on which this inclusion holds by $\mathcal { A } _ { N }$ . On $\mathcal { A } _ { N }$ , we use the H¨older regularity of T and a uniform Wasserstein law of large numbers to control the averaged diference between $\mathsf { T } ( { \boldsymbol \mu } , \cdot \mathsf { ) } _ { \# } { \boldsymbol \mu }$ and $\mathsf { T } ( \mu _ { N } , \cdot ) _ { \# } \mu _ { N }$ . We conclude by proving that the residual error term incurred by the tail probability of the bad set $\mathcal { A } _ { N } ^ { c }$ is of higher order. The details can be found in Appendix A.2.

To complement Theorem $3 . 7 ,$ we provide an analogous result under the assumption that $\mu$ has compact support. This assumption leads to an improved convergence rate with respect to the sample size $N$ and highlights a diference between the behavior of transformers on bounded and unbounded data. Although the next result is likely not novel, we were unable to locate a precise reference for it. Thus, we decided to present it in case it is of use in other contexts.

Proposition 3.8. Fix $d \in \mathbb { N }$ with $d \geq 3$ . Let $R > 0$ and $N \in \mathbb { N }$ , and let T: $\mathcal { P } ( \mathbb { R } ^ { d } ) \times \mathbb { R } ^ { d } \to \mathbb { R } ^ { k }$ be a meanfield transformer. Then there is a constant $C > 0$ depending on d, $R ,$ and T, such that

$$
\begin{array} { r l } & { \underset { \mu \in \mathcal P ( B _ { R } ( \mathbb { R } ^ { d } ) ) } { \operatorname* { s u p } } \mathbb { E } \mathsf { W } _ { 1 } \big ( \mathsf T ( \mu , \cdot ) _ { \# } \mu , \mathsf T ( \mu _ { N } , \cdot ) _ { \# } \mu _ { N } \big ) } \\ & { \qquad \leq C \bigg ( 1 + \sqrt { \log ( 1 + N ^ { 1 / d } ) } \bigg ) N ^ { - 1 / d } . } \end{array}
$$

The proof of Proposition 3.8, which is presented in Appendix A.3, is substantially simpler than the proof of Theorem 3.7 for two reasons: first, the mapping $\mu \mapsto \mathsf T ( \mu , \cdot ) _ { \# } \mu$ is globally Lipschitz on $\mathcal { P } ( B _ { R } )$ , and second, the empirical measure $\mu _ { N }$ belongs to $\mathcal { P } ( B _ { R } )$ almost surely.

## 3.4 Transformers With Mean-Field Cross-Attention

Given two input sequences $X = ( x _ { 1 } , \ldots , x _ { N } ) \subset \mathbb { R } ^ { d }$ and $Z = ( z _ { 1 } , \ldots , z _ { M } ) \subset \mathbb { R } ^ { d ^ { \prime } }$ , a (single-head) crossattention layer maps X and $Z$ to a third sequence $Y = ( y _ { 1 } , \dots , \bar { y } _ { M } ) \subset \mathbb { R } ^ { d ^ { \prime } }$ , where

$$
y _ { i } = z _ { i } + \frac { \sum _ { n = 1 } ^ { N } e ^ { \langle z _ { i } , A x _ { n } \rangle } V x _ { n } } { \sum _ { j = 1 } ^ { N } e ^ { \langle z _ { i } , A x _ { j } \rangle } }
$$

for $i \in [ M ]$ and V and A belong to $\mathbb { R } ^ { d ^ { \prime } \times d }$ . Taking the limit as the number of tokens $N$ and M tend to infinity, we consider a mean-field analog of transformers with the cross-attention architecture by considering operators

$$
( \mu , \eta ) \mapsto \Gamma ( \mu , \cdot ) _ { \# } \eta ,
$$

where Γ is the in-context mapping associated with a mean-field attention layer of the form (2.3). To complement the results of the previous subsections, we derive regularity and sample complexity estimates for mean-field transformers with cross-attention. We begin by establishing a joint H¨older continuity bound when the mapping above is defined by a multi-head self-attention layer.

Theorem 3.9. Let Γ: $\mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) \times \mathbb { R } ^ { d ^ { \prime } }  \mathbb { R } ^ { d ^ { \prime } }$ be a mean-field attention operator given by

$$
\Gamma ( \mu , x ) : = x + \sum _ { h = 1 } ^ { H } W _ { h } \frac { \int e ^ { \langle x , A _ { h } y \rangle } V _ { h } y ~ \mu ( d y ) } { \int e ^ { \langle x , A _ { h } z \rangle } ~ \mu ( d z ) } .\tag{3.2}
$$

Let $c _ { A } , \ : c _ { V }$ , and $c _ { W }$ be as in Theorem $\ 3 . 4 \cdot$ . Define

$$
\begin{array} { c } { k _ { 1 } : = \displaystyle \frac { 1 } { 2 \beta ( \beta ^ { 3 } \alpha ^ { 2 } c _ { A } ^ { 2 } + 1 ) } , k _ { 2 } : = 2 \sqrt { \displaystyle \frac { 2 c _ { A } ^ { 2 } } { \beta ^ { 3 } \alpha ^ { 2 } c _ { A } ^ { 2 } + 1 } } , } \\ { \gamma : = \displaystyle \frac { k _ { 2 } } { k _ { 1 } + k _ { 2 } } . } \end{array}
$$

Then there exists a constant $C = C ( \alpha , \beta , H , c _ { A } , c _ { V } ) > 0$ such that for all $\mu$ and $\mu _ { 0 }$ in $\mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ and all η and $\eta _ { 0 }$ in $\mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d ^ { \prime } } )$ , it holds that

$$
\begin{array} { r l } & { \mathsf { W } _ { 1 } \big ( \Gamma ( \mu , \cdot ) _ { \# } \eta , \Gamma ( \mu _ { 0 } , \cdot ) _ { \# } \eta _ { 0 } \big ) \le C \times } \\ & { \Big ( 1 \vee \log _ { + } \big ( \mathsf { W } _ { 1 } ( \mu , \mu _ { 0 } ) ^ { - 1 } \big ) ^ { \frac 3 2 } \Big ) \Big ( \mathsf { W } _ { 1 } ( \mu , \mu _ { 0 } ) ^ { \gamma } + \mathsf { W } _ { 1 } ( \eta , \eta _ { 0 } ) \Big ) . } \end{array}
$$

Theorem 3.9 establishes joint H¨older regularity of the mean-field cross-attention mapping, up to logarithmic factors. Note that the estimate is not symmetric in $\mu$ and $\eta$ because the mapping $( \mu , \eta ) \mapsto \Gamma ( \mu , \cdot ) _ { \# } \eta$ is not symmetric. Rather, the estimate shows that the measure-to-measure operator associated with crossattention is smoother as a function of $\eta$ than as a function of $\mu ;$ in particular, cross-attention is nearly Lipschitz continuous with respect to the second argument, up to a logarithmic factor. This asymmetry is due to the diference in scale between the local Lipschitz constant of the mapping $( \mu , x ) \mapsto \Gamma ( \mu , x )$ with respect to each variable. Theorem $3 . 9$ can also be generalized to repeated compositions of cross- and self-attention layers and feedforward neural networks, but the propagation of log factors across layers is more complicated than for mean-field transformers with self-attention. Moreover, this deep setting is also more challenging because the implied deep transformer in-context map can depend on both $\mu$ and $\eta ,$ instead of just $\mu .$ For these reasons, we choose not to generalize Theorem 3.9 beyond a single attention layer.

Paralleling Theorem 3.7, we also establish a sample complexity bound for transformers with cross-attention. To simplify the presentation, we again restrict to a single attention layer, but the result can straightforwardly be generalized to deep transformers at the expense of a looser scaling law. In the result below, we consider two assumptions on the pair of measures $( \mu , \eta )$ : when $\mu \in \mathcal P ( B _ { R } )$ and $\eta \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ for some $R > 0 , \alpha \geq 1$ and $\beta > 0$ , we obtain nonparametric sample complexity rates analogous to those in Theorem $3 . 7 ;$ on the other hand, when both $\mu$ and $\eta$ belong to $\mathcal { P } ( B _ { R } )$ , we find that the sample complexity with respect to $\mu$ achieves a dimension-free rate of $\check { N } ^ { - 1 / 2 }$ , up to log factors. We leave extending the cross-attention sample complexity bounds in the case where both $\mu$ and γ have unbounded support to future work.

Theorem 3.10. Fix $d \in \mathbb { N }$ with $d \geq 3$ . Fix $\alpha \geq 1$ ， $\beta > 0$ , and $R > 0$ . Let Γ : $\mathcal { P } ( \mathbb { R } ^ { d } ) \times \mathbb { R } ^ { d ^ { \prime } } \to \mathbb { R } ^ { d ^ { \prime } }$ be a mean-field multi-head attention layer as in $_ { ( 3 . 2 ) }$ . Let $c _ { A } , \ : c _ { V }$ , and c be as defined in Theorem $\ 3 . 4 \cdot$ Then there exist constants $C _ { 1 } > 0 , C _ { 2 } > 0 \quad$ , and $C _ { 3 } \geq 1$ which depend on $d , d ^ { \prime } , \alpha , \beta , { \cal R } , { \cal H } , c _ { \cal A }$ , and $c _ { V } .$ , such that for all $M \in \mathbb { N }$ and all $N \geq C _ { 3 }$ , it holds that

$$
\begin{array} { r l } & { \underset { \mu \in \mathcal { P } ( B _ { R } ) , \eta \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) } { \operatorname* { s u p } } ~ \mathbb { E } \mathbb { W } _ { 1 } \big ( \Gamma ( \mu , \cdot ) _ { \# } \eta , \Gamma ( \mu _ { N } , \cdot ) _ { \# } \eta _ { M } \big ) } \\ & { ~ \leq C _ { 1 } \bigg ( \mathsf { p o l y l o g } ( N ) N ^ { - \frac { 1 } { d ( 4 _ { \alpha } \beta + 1 ) } } + \mathsf { p o l y l o g } ( M ) M ^ { - 1 / d } \bigg ) } \\ & { a n d } \\ & { \underset { \mu \in \mathcal { P } ( B _ { R } ) , \eta \in \mathcal { P } ( B _ { R } ) } { \operatorname* { s u p } } ~ \mathbb { E } \mathbb { W } _ { 1 } \big ( \Gamma ( \mu , \cdot ) _ { \# } \eta , \Gamma ( \mu _ { N } , \cdot ) _ { \# } \eta _ { M } \big ) } \\ & { ~ \leq C _ { 2 } \bigg ( N ^ { - 1 / 2 } + \bigg ( 1 + \sqrt { \log \big ( 1 + M ^ { 1 / d } \big ) } \bigg ) M ^ { - 1 / d } \bigg ) . } \end{array}
$$

As is the case in Theorem 3.9, Theorem 3.10 suggests that the sample complexity of cross-attention difers with respect to each variable. These scaling laws may provide practical suggestions for how long the relative contexts of each component should be. This is particularly relevant for generative modeling settings, where η may denote a reference measure from which new samples can be readily generated.

## 3.5 Consequences for Measure-to-Measure Approximation With Transformers

We conclude the section with an application of our results to the approximation of measure-to-measure operators by transformers. Recently, Furuya et al. (2026b) established suficient conditions on a $\mathsf { W } _ { p }$ -compact subset $K \subset \mathcal { P } ( \mathbb { R } ^ { d } )$ under which the pushforward operators induced by transformers can approximate continuous measure-to-measure operators to arbitrary accuracy, uniformly over K. However, this result does not have immediate consequences for transformers evaluated at empirical measures because the hypotheses of the result prohibit K from containing atomic measures. Below, we state a generalization of the result to transformers evaluated at empirical measures, where the sampling error is instead controlled using Theorem 3.7.

Theorem 3.11. Fix $\alpha \geq 1$ and $\beta > 0$ . Let $\mathcal { P } _ { 1 } ( \mathbb { R } ^ { d } )$ denote the subset of probability measures in ${ \mathcal { P } } ( \mathbb { R } ^ { d } )$ with finite first moment. Let $K \subset \operatorname { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ , and suppose there exists a non-atomic measure $\tau \in \mathcal { P } ( \mathbb { R } ^ { d } )$ such that $\mu$ is absolutely continuous with respect to τ for every $\mu \in K$ . Let $F \colon ( K , \mathsf { W } _ { 1 } ) \to ( \mathcal { P } _ { 1 } ( \mathbb { R } ^ { d } ) , \mathsf { W } _ { 1 } )$ be continuous. Then for any $\epsilon > 0$ , there exists a transformer ${ { \sf T } _ { \epsilon } }$ of the form (2.5) and constants $C _ { \epsilon } > 0$ and $\gamma _ { \epsilon } \in ( 0 , 1 )$ (depending on d, $\alpha , \beta$ , and $T _ { \epsilon } )$ such that

$$
\operatorname* { s u p } _ { \mu \in K } \mathbb { E } \mathsf { W } _ { 1 } \big ( \mathsf { T } _ { \epsilon } ( \mu _ { N } , \cdot ) _ { \# } \mu _ { N } , F ( \mu ) \big ) < \frac { \epsilon } { 2 } +
$$

$$
C _ { \epsilon } \log ( N ) ^ { \gamma _ { \epsilon } / 2 } \Bigl ( 1 + \log \bigl ( 1 + N ^ { 1 / d } \bigr ) \Bigr ) ^ { \gamma _ { \epsilon } / 2 } N ^ { - \gamma _ { \epsilon } / d }
$$

for N large enough. In particular, for each $\epsilon > 0 _ { : }$ , there exists $N _ { \epsilon } \in \mathbb { N }$ (depending on d, $\alpha , \beta ,$ , and $\epsilon )$ such that

$$
\operatorname* { s u p } _ { \mu \in K } \mathbb { E } \mathsf { W } _ { 1 } \big ( \mathsf { T } _ { \epsilon } ( \mu _ { N } , \cdot ) _ { \# } \mu _ { N } , F ( \mu ) \big ) < \epsilon
$$

for all $N \geq N _ { \epsilon }$

Theorem 3.11 states that mean-field transformer operators are universal approximators for continuous measure-to-measure operators, even when the input measure is only accessible $b y$ samples, provided that the number of samples is suficiently large and the compact set $K$ upon which the uniform approximation occurs is uniformly subgaussian and absolutely continuous with respect to a common non-atomic reference. Moreover, for a fixed approximating transformer ${ { \mathsf { T } } _ { \epsilon } } ,$ the sampling error converges to zero at a polynomial rate, up to log factors. The assumption that K comprises measures which are dominated by a common non-atomic reference measure ensures that continuous operators on K can be universally approximated by pushforwards, even when $\mu \mapsto F ( \mu )$ does not have an exact pushforward representation; see Assumption 3.1 in the work of Furuya et al. (2026b) for a more general suficient condition. Indeed, without any atomloss assumptions on the measures belonging to $K ,$ it is easy to construct examples of continuous operators on $K$ which cannot be approximated by pushforwards.

## 4 CONCLUSION

This work establishes mathematical properties of the measure-to-measure operators defined by transformers. The main contributions are a stability result stating that transformers map sub-Gaussian measures to sub-Gaussian measures, a H¨older continuity bound in the 1-Wasserstein distance, and an estimate on the error propagation between a true probability measure and its empirical counterpart along a transformer. Analogs of these results are provided for transformers which use the cross-attention mechanism. The results herein motivate several directions for future work. First, our sample complexity results sufer from a curse of dimensionality which is intrinsic to high-dimensional nonparametric statistical problems; it is unclear whether the parametric complexity of transformers can be leveraged to improve these rates, or whether faster rates occur for probability measures with low-dimensional structure. Second, motivated by language models, we would like to prove analogs of our sample complexity results for empirical measures of non-iid data, $\mathrm { e . g . }$ , for tokens generated autoregressively. Last, our results on regularity and sample complexity motivate the development of a generalization error analysis of transformers for supervised learning problems on spaces of probability measures. We leave these questions to future work.

## AI Use Statement

In this work, we used generative AI tools for formulating mathematical claims and providing critical ingredients to prove mathematical claims. We did not use generative AI tools for generating synthetic data sets, developing theoretical models or conceptual frameworks, assisting in the writing of proofs, proposing or refining hypotheses, designing or providing feedback on research methodology or experiments, implementing methods, assisting with translation, cleaning and reformatting datasets, supporting qualitative and thematic data analysis, or interpreting results. We have reviewed all AI-assisted work. In more detail, the proof ingredients for Proposition 3.2 and Lemma A.1 came about between the authors and ChatGPT 5.6. We take responsibility for the final content of this work, including text, claims, or artifacts produced with the aid of generative AI.

## Acknowledgments

The research of N.H.N. is supported by a Klarman Fellowship through Cornell University’s College of Arts & Sciences and by startup funds at The University of Texas at Austin. T.F. acknowledges funding from JSPS KAKENHI (grants JP24K16949 and 25H01453), JST CREST (JPMJCR24Q5), and JST ASPIRE (JP-MJAP2329).

## References

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. (2023). GPT-4 technical report. preprint arXiv:2303.08774.

Al-Jarrah, M., Hosseini, B., and Taghvaei, A. (2025). Fast filtering of non-Gaussian models using amortized optimal transport maps. IEEE Control Systems Letters.

Ba, J. L., Kiros, J. R., and Hinton, G. E. (2016). Layer normalization. preprint arXiv:1607.06450.

Bach, E., Baptista, R., Br¨ocker, J., Chen, B., and Stuart, A. M. (2026). Learning probabilistic filters with strictly proper scoring rules. preprint arXiv:2606.26497.

Bach, E., Baptista, R., Calvello, E., Chen, B., and Stuart, A. M. (2025). Learning enhanced ensemble filters. Journal of Computational Physics.

Bodnar, C., Bruinsma, W. P., Lucic, A., Stanley, M., Allen, A., Brandstetter, J., Garvan, P., Riechert, M., Weyn, J. A., Dong, H., et al. (2025). A foundation model for the Earth system. Nature, 641(8065):1180– 1187.

Bohbot, L., Letrouit, C., Peyr´e, G., and Vialard, F.-X. (2026). Token sample complexity of attention. In International Conference on Machine Learning.

Boursier, E. and Boyer, C. (2026). Softmax as linear attention in the large-prompt regime: A measurebased perspective. In International Conference on Machine Learning.

Bruno, G., Chen, S., Lin, Z., Polyanskiy, Y., and Rigollet, P. (2026a). Scaling limits of long-context transformers. preprint arXiv:2605.08505.

Bruno, G., Pasqualotto, F., and Agazzi, A. (2025). Emergence of meta-stable clustering in mean-field transformer models. In International Conference on Learning Representations, volume 2025, pages 7496–7526.

Bruno, G., Pasqualotto, F., and Agazzi, A. (2026b). A multiscale analysis of mean-field transformers in the moderate interaction regime. In Advances in Neural Information Processing Systems, volume 38, pages 133305–133341.

Cai, R., Yang, G., Averbuch-Elor, H., Hao, Z., Belongie, S., Snavely, N., and Hariharan, B. (2020). Learning gradient fields for shape generation. In European Conference on Computer Vision, pages 364–381. Springer.

Castin, V., Ablin, P., Carrillo, J. A., and Peyr´e, G. (2025). A unified perspective on the dynamics of deep transformers. Foundations of Computational Mathematics.

Castin, V., Ablin, P., and Peyr´e, G. (2024). How smooth is attention? In International Conference on Machine Learning, pages 5817–5840. PMLR.

Chen, S., Lin, Z., Liu, K., and Rigollet, P. (2026a). Propagation of chaos in contextual flow maps. preprint arXiv:2605.16747.

Chen, S., Lin, Z., Polyanskiy, Y., and Rigollet, P. (2025). Quantitative clustering in mean-field transformer models. preprint arXiv:2504.14697.

Chen, S., Lin, Z., Polyanskiy, Y., and Rigollet, P. (2026b). Critical attention scaling in long-context transformers. In International Conference on Learning Representations, volume 2026, pages 86454– 86478.

Chewi, S., Niles-Weed, J., and Rigollet, P. (2025). Statistical optimal transport. Springer.

Cole, F., Wang, D., Chen, Y., Lu, Y., and Lai, R. (2026). In-context operator learning on the space of probability measures. preprint arXiv:2601.09979.

Dudley, R. M. (2018). Real analysis and probability. Chapman and Hall/CRC.

Fishman, N., Gowri, G., Fischer, P. L., Zitnik, M., Abudayyeh, O., and Gootenberg, J. (2026). Distributionconditioned transport. preprint arXiv:2603.04736.

Furuya, T., de Hoop, M. V., and Lassas, M. (2025a). Transformers through the lens of support-preserving maps between measures. preprint arXiv:2509.25611.

Furuya, T., de Hoop, M. V., and Peyr´e, G. (2025b). Transformers are universal in-context learners. In International Conference on Learning Representations.

Furuya, T., Murari, D., and Sch¨onlieb, C.-B. (2026a). Approximation theory for Lipschitz continuous transformers. In International Conference on Machine Learning.

Furuya, T., Nelsen, N. H., and Cole, F. (2026b). Universal approximation of measure-to-measure operators by pushforwards. preprint arXiv:2609.35483.

Generale, A. P., Robertson, A. E., and Kalidindi, S. (2026). Modeling stochastic conditional dynamics from sparse observations via kernel-stabilized flow matching. Transactions on Machine Learning Research.

Geshkovski, B., Letrouit, C., Polyanskiy, Y., and Rigollet, P. (2023). The emergence of clusters in selfattention dynamics. In Advances in Neural Information Processing Systems, volume 36, pages 57026– 57037.

Geshkovski, B., Letrouit, C., Polyanskiy, Y., and Rigollet, P. (2025). A mathematical perspective on transformers. Bulletin of the American Mathematical Society, 62(3):427–479.

Geshkovski, B., Rigollet, P., and Ruiz-Balet, D. (2026). Measure-to-measure interpolation using transformers. Foundations of Computational Mathematics, pages 1–50.

Gloeckler, M., Deistler, M., Weilbach, C. D., Wood, F., and Macke, J. H. (2024). All-in-one simulation-based inference. In International Conference on Machine Learning, pages 15735–15766.

Guerra, N., Nelsen, N. H., and Yang, Y. (2025). Learning where to learn: Training data distribution optimization for scientific machine learning. preprint arXiv:2505.21626.

Hargenrader, K., Calvello, E., and Chen, B. (2026). Attention kernels for learning maps between heavytailed measures. preprint arXiv:2610.00564.

Herde, M., Raoni´c, B., Rohner, T., K¨appeli, R., Molinaro, R., De Bezenac, E., and Mishra, S. (2024). Poseidon: Eficient foundation models for PDEs. In Advances in Neural Information Processing Systems, volume 37, pages 72525–72624.

Horawalavithana, S., Ayton, E., Sharma, S., Howland, S., Subramanian, M., Vasquez, S., Cosbey, R., Glenski, M., and Volkova, S. (2022). Foundation models of scientific knowledge for chemistry: Opportunities, challenges and lessons learned. In Proceedings of BigScience Episode# 5–Workshop on Challenges & Perspectives in Creating Large Language Models, pages 160–172.

Huang, H. and Lai, R. (2025). Unsupervised solution operator learning for mean-field games. Journal of Computational Physics, 537.

Kan, K., Li, X., Zhang, B., Sahai, T., Osher, S., and Katsoulakis, M. (2026). Optimal control for transformer architectures: Enhancing generalization, robustness and eficiency. In Advances in Neural Information Processing Systems, volume 38, pages 64183–64228.

Kan, K., Li, X., Zhang, B. J., Sahai, T., Osher, S., Kumar, K., and Katsoulakis, M. A. (2025). Stability of transformers under layer normalization. preprint arXiv:2510.09904.

Karagodin, N., Polyanskiy, Y., and Rigollet, P. (2024). Clustering in causal attention masking. In Advances in Neural Information Processing Systems, volume 37, pages 115652–115681.

Kim, H., Papamakarios, G., and Mnih, A. (2021). The Lipschitz constant of self-attention. In International Conference on Machine Learning, pages 5562–5571. PMLR.

Lavenant, H. and Savar´e, G. (2026). Continuous transformations of probability measures and their transport representations. preprint arXiv:2604.16653.

Le´on, U. M. and Niles-Weed, J. (2026). Wasserstein least squares: A canonical regression method for probability distributions. preprint arXiv:2605.30266.

Li, S., Maranzatto, T. J., Peszek, J., Teolis, T., Akkoc, S., Riedl, K., Ulukus, S., and Trillos, N. G. (2026). On the diverse dynamical behaviors arising in deep linear transformers. preprint arXiv:2607.18584.

Liu, L., Liu, X., Gao, J., Chen, W., and Han, J. (2020). Understanding the dificulty of training transformers. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 5747–5763.

Mroueh, Y. (2023). Towards a statistical theory of learning to learn in-context with transformers. In NeurIPS 2023 Workshop Optimal Transport and Machine Learning.

Nelsen, N. H. and Yang, Y. (2026). Operator learning meets inverse problems: A probabilistic perspective. In Hauptmann, A., Jin, B., and Sch¨onlieb, C.-B., editors, Handbook of Numerical Analysis, volume 27. Elsevier.

Panaretos, V. M. and Zemel, Y. (2020). An invitation to statistics in Wasserstein space. Springer Nature.

Pendharkar, A. (2026). Gradient flow structure and quantitative dynamics of multi-head self-attention. preprint arXiv:2605.04279.

Rigollet, P. (2026). The mean-field dynamics of transformers. In International Congress of Mathematicians 2026, pages 389–404. SIAM.

Sander, M. E., Ablin, P., Blondel, M., and Peyr´e, G. (2022). Sinkformers: Transformers with doubly stochastic attention. In International Conference on Artificial Intelligence and Statistics, pages 3515– 3530.

Shalev-Shwartz, S. and Ben-David, S. (2014). Understanding machine learning: From theory to algorithms. Cambridge University Press.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. (2021). Score-based generative modeling through stochastic diferential equations. In International Conference on Learning Representations.

Stuart, A. M. (2010). Inverse problems: A Bayesian perspective. Acta Numerica, 19:451–559.

Subramanian, S., Harrington, P., Keutzer, K., Bhimji, W., Morozov, D., Mahoney, M. W., and Gholami, A. (2023). Towards foundation models for scientific machine learning: Characterizing scaling and transfer behavior. In Advances in Neural Information Processing Systems, volume 36, pages 71242–71262.

Takeda, S., Kishimoto, A., Hamada, L., Nakano, D., and Smith, J. R. (2023). Foundation model for material science. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 15376– 15383.

Tsimpos, P., Calvello, E., Belhadji, A., and Nelsen, N. H. (2026). One operator for many densities: Amortized approximation of conditioning by neural operators. preprint arXiv:2605.06873.

Vandergrift, M., White, M., Polyanskiy, Y., Rigollet, P., and Atanackovic, L. (2026). Measure-to-measure regression with transformers. In The 2026 Workshop on Generative and Agentic AI for Biology.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. (2017). Attention is all you need. In Advances in Neural Information Processing Systems, volume 30.

Vershynin, R. (2018). High-dimensional probability: An introduction with applications in data science, volume 47. Cambridge University Press.

Villani, C. (2009). Optimal Transport: Old and New, volume 338. Springer.

Vuckovic, J., Baratin, A., and Combes, R. T. d. (2020). A mathematical theory of attention. preprint arXiv:2007.02876.

Vuckovic, J., Baratin, A., and Combes, R. T. d. (2021). On the regularity of attention. preprint arXiv:2102.05628.

Wang, X., Zhang, Z., Lyngaas, I., Yoon, H.-J., Choi, J.-Y., Liang, S., Wang, J., Chipilski, H. G., Aji, A. M., Bao, F., et al. (2026). Global attention with linear complexity for exascale generative data assimilation in earth system prediction. preprint arXiv:2604.16590.

Whittle, G., Ziomek, J., Rawling, J., and Osborne, M. A. (2026). Distribution transformers: Fast approximate Bayesian inference with on-the-fly prior adaptation. In International Conference on Machine Learning.

Wiesner, F., Gray, Z. J., Wessling, M., and Baek, S. (2025). Towards a physics foundation model. preprint arXiv:2509.13805.

Yao, S., Levin, E., and D´ıaz, M. (2026). Anydimensional invariant universality. preprint arXiv:2605.23156.

Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R. R., and Smola, A. J. (2017). Deep sets. In Advances in Neural Information Processing Systems, volume 30.

## CHECKLIST

1. For all models and algorithms presented, check if you include:

(a) A clear description of the mathematical setting, assumptions, algorithm, and/or model. Yes.

(b) An analysis of the properties and complexity (time, space, sample size) of any algorithm. Not Applicable

(c) (Optional) Anonymized source code, with specification of all dependencies, including external libraries. Not Applicable

2. For any theoretical claim, check if you include:

(a) Statements of the full set of assumptions of all theoretical results. Yes

(b) Complete proofs of all theoretical results. Yes

(c) Clear explanations of any assumptions. Yes

3. For all figures and tables that present empirical results, check if you include:

(a) The code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL). Not Applicable

(b) All the training details (e.g., data splits, hyperparameters, how they were chosen). Not Applicable

(c) A clear definition of the specific measure or statistics and error bars (e.g., with respect to the random seed after running experiments multiple times). Not Applicable

(d) A description of the computing infrastructure used. (e.g., type of GPUs, internal cluster, or cloud provider). Not Applicable

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets, check if you include:

(a) Citations of the creator If your work uses existing assets. Not Applicable

(b) The license information of the assets, if applicable. Not Applicable

(c) New assets either in the supplemental material or as a URL, if applicable. Not Applicable

(d) Information about consent from data providers/curators. Not Applicable

(e) Discussion of sensible content if applicable, e.g., personally identifiable information or offensive content. Not Applicable

5. If you used crowdsourcing or conducted research with human subjects, check if you include:

(a) The full text of instructions given to participants and screenshots. Not Applicable

(b) Descriptions of potential participant risks, with links to Institutional Review Board (IRB) approvals if applicable. Not Applicable

(c) The estimated hourly wage paid to participants and the total amount spent on participant compensation. Not Applicable

# Supplementary Materials for: Stability of Measure-to-Measure Transformers on Sub-Gaussian Data

## A PROOFS

This appendix collects the proofs of results presented in the main text.

## A.1 Proofs for Section 3.1: Transformers Preserve Sub-Gaussianity

Before proving Proposition 3.2, we need to prove a technical lemma on the growth of gradients of convex functions. Lemma A.1. Let $f \colon  { \mathbb { R } ^ { d } } \to  { \mathbb { R } }$ be a diferentiable convex function such that there exist constants $A , B > 0$ for which $| f ( x ) | \leq A + B \| x \| ^ { 2 }$ for all $\| { \boldsymbol x } \| \geq 1$ . Then

$$
\| \nabla f ( x ) \| \leq ( 2 A + B ) + 5 B \| x \|
$$

for all $\| \boldsymbol { x } \| > 1$

Proof. Fix x with $\| { \boldsymbol x } \| > 1$ . Let $h > 0$ be a positive number to be chosen precisely later, and let $u \in \mathbb { R } ^ { d }$ be an arbitrary unit vector. We apply the first order condition for convex functions to obtain

$$
f ( x + h u ) \geq f ( x ) + h \nabla f ( x ) \cdot u ,
$$

or equivalently,

$$
\nabla f ( x ) \cdot u \leq { \frac { f ( x + h u ) - f ( x ) } { h } } .
$$

If $\| { \boldsymbol { x } } + h { \boldsymbol { u } } \| > 1$ , then it holds that

$$
\begin{array} { r l r } {  { \frac { f ( x + h u ) - f ( x ) } { h } \leq \frac { 2 A + B ( \| x + h u \| ^ { 2 } + \| x \| ^ { 2 } ) } { h } } } \\ & { } & { \qquad = \frac { 2 A + 2 B \| x \| ^ { 2 } } { h } + 2 B x \cdot u + B h } \\ & { } & { \qquad \leq \frac { 2 A + 2 B \| x \| ^ { 2 } } { h } + 2 B \| x \| + B h . } \end{array}
$$

If $\| x + h u \| < 1$ , we can write $x + h u = \lambda z _ { 1 } + ( 1 - \lambda ) z$ <sub>2</sub> for some $\lambda \in ( 0 , 1 )$ and $z _ { 1 } , z _ { 2 } \in \mathbb { S } ^ { d - 1 }$ . Convexity of f implies that

$$
| f ( x + h u ) | \leq \operatorname* { m a x } ( f ( z _ { 1 } ) , f ( z _ { 2 } ) ) \leq A + B ,
$$

which further implies that

$$
\begin{array} { r l r } {  { \frac { f ( \boldsymbol { x } + h \boldsymbol { u } ) - f ( \boldsymbol { x } ) } { h } \le \frac { ( 2 A + B ) + B \| \boldsymbol { x } \| ^ { 2 } } { h } } } \\ & { } & { \le \frac { ( 2 A + B ) + 2 B \| \boldsymbol { x } \| ^ { 2 } } { h } + 2 B \| \boldsymbol { x } \| + B h . } \end{array}
$$

Therefore,

$$
{ \frac { f ( x + h u ) - f ( x ) } { h } } \leq { \frac { ( 2 A + B ) + 2 B \| x \| ^ { 2 } } { h } } + 2 B \| x \| + B h
$$

for all $\| x \| > 1 , h > 0$ , and $u \in \mathbb { S } ^ { d - 1 }$ . Choosing $h = \| x \|$ , and using the fact that $\| \boldsymbol { x } \| > 1$ , we have that

$$
\nabla f ( x ) \cdot u \leq ( 2 A + B ) + 5 B \| x \| .
$$

Next, we apply the first-order bound to $f ( x - h u )$ :

$$
f ( x - h u ) \geq f ( x ) - h \nabla f ( x ) \cdot u ,
$$

which implies

$$
- \nabla f ( x ) \cdot u \leq { \frac { f ( x - h u ) - f ( x ) } { h } } .
$$

A parallel argument shows that

$$
{ \frac { f ( x - h u ) - f ( x ) } { h } } \leq { \frac { ( 2 A + B ) + B \| x \| ^ { 2 } } { h } } + 2 B \| x \| + B h
$$

for all $\| x \| > 1 , h > 0$ , and $u \in \mathbb { S } ^ { d - 1 }$ . Choosing $h = \| x \|$ yields

$$
- \nabla f ( x ) \cdot u \leq ( 2 A + B ) + 5 B \| x \| .
$$

This proves that

$$
| \nabla f ( x ) \cdot u | \leq ( 2 A + B ) + 5 B \| x \| ,
$$

for all $\| \boldsymbol { x } \| > 1$ , and since u is arbitrary, we have

$$
\| \nabla f ( x ) \| \leq ( 2 A + B ) + 5 B \| x \| .
$$

Proof of Proposition 3.2. Clearly it sufices to verify the result in the case of a single head and without the skip connection: given weight matrices $V \in \mathbb { R } ^ { k \times d }$ and $A \in \mathbb { R } ^ { d \times d }$ , we aim to prove that there exist constants $K _ { 1 } , K _ { 2 } > 0$ such that

$$
\mathrm { A t t } _ { \theta } ( \mu , x ) : = \frac { \int e ^ { \langle x , A y \rangle } V y \mu ( d y ) } { \int e ^ { \langle x , A z \rangle } \mu ( d z ) } \le K _ { 1 } + K _ { 2 } \| x \|
$$

for all $x \in \mathbb { R } ^ { d }$ . To this end, define the cumulant generating function

$$
Z _ { \mu } ( t ) : = \log \int e ^ { \langle t , y \rangle } \mu ( d y ) ,
$$

and note that $Z _ { \mu }$ is convex. We can express the map $\operatorname { A t t } _ { \theta } ( \mu , x )$ in terms of the cumulant generating function:

$$
\operatorname { A t t } _ { \theta } ( \mu , x ) = V \nabla Z _ { \mu } ( A x ) .
$$

Thus, to show the desired claim, it sufices to prove that $\nabla Z _ { \mu } ( t )$ grows at most afine-linearly, with constants depending only on the sub-Gaussian parameters of $\mu .$ We divide this into two cases.

Case 1: $\| t \| \leq 1$ . We claim that $\nabla Z _ { \mu } ( t )$ is bounded on the unit ball by a constant depending only on the sub-Gaussian parameters of $\mu .$ Observe that

$$
\nabla Z _ { \mu } ( t ) = \frac { \int y e ^ { \langle y , t \rangle } \mu ( d y ) } { \int e ^ { \langle y , t \rangle } \mu ( d y ) } = \frac { Z _ { \mu } ^ { ( 1 ) } ( t ) } { Z _ { \mu } ^ { ( 2 ) } ( t ) } .
$$

For the numerator, we have

$$
Z _ { \mu } ^ { ( 1 ) } ( t ) \leq { \biggl ( } \int \| y \| ^ { 2 } \mu ( d y ) { \biggr ) } ^ { 1 / 2 } { \biggl ( } \int e ^ { 2 \| y \| } \mu ( d y ) { \biggr ) } ^ { 1 / 2 }
$$

for all $\| t \| \leq 1 . \mathrm { B y }$ Lemma B.1, this expression admits an upper bound depending only on the sub-Gaussian parameters α and $\beta .$ . For the denominator, we have by Jensen’s inequality and the Cauchy–Schwarz inequality,

$$
Z _ { \mu } ^ { ( 2 ) } ( t ) \geq \exp \biggl ( - \int \| y \| \mu ( d y ) \biggr ) ,
$$

and by Lemma B.1 the integrand in the exponential can be upper bounded uniformly in $\mu .$ This proves that $\begin{array} { r } { \operatorname* { s u p } _ { \| t \| \leq 1 } \| \nabla _ { \mu } Z ( t ) \| } \end{array}$ is uniformly bounded over $\mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$

Case 2: $\| t \| \geq 1$ . Since $\mu$ is sub-Gaussian, it follows (see the book of Vershynin (2018) for equivalent characterizations of sub-Gaussian random variables) that there exist constants $M _ { 1 } , M _ { 2 } > 0$ depending only on α and $\beta$ such that

$$
Z _ { \mu } ( t ) \le M _ { 1 } + M _ { 2 } \| t \| ^ { 2 } ,
$$

for all t with $\| t \| > 1$ . Since $Z _ { \mu } ( t )$ is convex, it holds by Lemma A.1 that

$$
\| \nabla Z _ { \mu } ( t ) \| \le ( 2 M _ { 1 } + M _ { 2 } ) + 5 M _ { 2 } \| t \|
$$

for all t with $\| t \| \geq 1$ and $\mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ . Combining the two cases, we have shown that $\operatorname { A t t } _ { \theta } ( \mu , x )$ grows afine linearly with constants depending only on the sub-Gaussian parameters of $\mu$ and the attention matrices. □

## A.2 Proofs for Section 3.2: H¨older Regularity of Transformers on Sub-Gaussian Domains

Fix $\alpha \geq 1$ and $\beta > 0$ , and recall that our goal is to bound the Wasserstein distance

$$
\mathsf { W } _ { 1 } \big ( \Gamma ( \boldsymbol { \mu } , \cdot ) _ { \# } \boldsymbol { \mu } , \Gamma ( \boldsymbol { \mu } _ { 0 } , \cdot ) _ { \# } \boldsymbol { \mu } _ { 0 } \big )
$$

in terms of $\mathsf { W } _ { 1 } ( \mu , \mu _ { 0 } )$ , where Γ is a mean-field multi-head attention layer. Our proof is based on a truncation argument. For $R > 0$ , we let $\Pi _ { R } \colon  { \mathbb { R } ^ { d } } \to  { \mathbb { R } ^ { d } }$ denote the projection map onto the ball of radius R. An important step in the proof of Theorem 3.4 is the following truncation estimate, which controls the diference between $\Gamma ( \boldsymbol { \mu } , \boldsymbol { x } )$ and $\Gamma ( ( \Pi _ { R } ) _ { \# } \mu , x )$ . We first prove the estimate for a single-head attention layer (with the head matrix included in the definition for emphasis), from which the multi-head case follows as a corollary.

Proposition A.2. Define

$$
\mathrm { A t t } _ { \theta } ( \mu , x ) = { W } \frac { \int e ^ { \langle A x , y \rangle } V y \mu ( d y ) } { \int e ^ { \langle A x , z \rangle } \mu ( d z ) } .
$$

Then for any $R > 0$ and $M > 0$ , it holds that

$$
\begin{array} { r l } & { \underset { \mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) } { \operatorname* { s u p } } \underset { | | x | \leq M } { \operatorname* { s u p } } \left. \mathrm { A t t } _ { \theta } ( \mu , x ) - \mathrm { A t t } _ { \theta } \big ( ( \Pi _ { R } ) _ { \# } \mu , x \big ) \right. } \\ & { \leq \| W \| \| V \| e ^ { \| A \| M C _ { \alpha , \beta } } ( 2 + R ) \alpha \| A \| M e ^ { \frac { \beta ^ { 2 } \alpha ^ { 2 } \| A \| ^ { 2 } M ^ { 2 } } { 2 } } \bigg ( \frac { \| A \| M \beta ^ { 2 } } { R } + 2 \beta \bigg ) e ^ { - \frac { R ^ { 2 } } { 2 \beta } } , } \end{array}
$$

where we have defined

$$
C _ { \alpha , \beta } : = \operatorname* { s u p } _ { \nu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) } \int \| x \| \nu ( d x ) < \infty .
$$

Proof. Throughout the proof we will assume tha $M > \| A \| ^ { - 1 }$ . Clearly this assumption is without loss of generality because

$$
\begin{array} { r l } & { \underset { \mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) } { \operatorname* { s u p } } \underset { \| x \| \leq M } { \operatorname* { s u p } } \big \| \mathrm { A t t } _ { \theta } ( \mu , x ) - \mathrm { A t t } _ { \theta } \big ( ( \Pi _ { R } ) _ { \# } \mu , x \big ) \big \| } \\ & { \quad \quad \quad \leq \underset { \mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) } { \operatorname* { s u p } } \underset { \| x \| \leq \operatorname* { m a x } ( M , \| A \| ^ { - 1 } ) } { \operatorname* { s u p } } \big \| \mathrm { A t t } _ { \theta } ( \mu , x ) - \mathrm { A t t } _ { \theta } \big ( ( \Pi _ { R } ) _ { \# } \mu , x \big ) \big \| . } \end{array}
$$

Fix $\mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ and $x \in B _ { M }$ . Write $\mu _ { R } : = ( \Pi _ { R } ) _ { \# } \mu$ . Then

$$
\mathrm { A t t } _ { \theta } ( \mu _ { R } , x ) = \frac { \mathsf { N } _ { \mu , R } ( x ) } { \mathsf { D } _ { \mu , R } ( x ) } : = \frac { \int _ { \mathbb { R } ^ { d } } W V \Pi _ { R } ( y ) e ^ { \langle \Pi _ { R } ( y ) , A x \rangle } \mu ( d y ) } { \int _ { \mathbb { R } ^ { d } } e ^ { \langle \Pi _ { R } ( z ) , A x \rangle } \mu ( d z ) } .
$$

We similarly write $\mathrm { A t t } _ { \theta } ( \mu , x ) = \mathsf { N } _ { \mu } ( x ) / \mathsf { D } _ { \mu } ( x )$ . We estimate

$$
\| \mathrm { A t t } _ { \theta } ( \mu , x ) - \mathrm { A t t } _ { \theta } ( \mu _ { R } , x ) \| \leq \underbrace { \frac { 1 } { \mathrm { D } _ { \mu } ( x ) } \| \mathsf { N } _ { \mu } ( x ) - \mathsf { N } _ { \mu , R } ( x ) \| } _ { \mathrm { ( I ) } } + \underbrace { \| \mathsf { N } _ { \mu , R } ( x ) \| \bigg | \frac { 1 } { \mathrm { D } _ { \mu } ( x ) } - \frac { 1 } { \mathrm { D } _ { \mu , R } ( x ) } \bigg | } _ { \mathrm { ( I I ) } } .
$$

We begin with term (II). Since

$$
\left| \left. \int _ { \mathbb { R } ^ { d } } z \mu ( d z ) , A x \right. \right| \leq \| A \| M \operatorname* { s u p } _ { \nu \in { \mathrm { S G } } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) } \int _ { \mathbb { R } ^ { d } } \| x \| \nu ( d x ) = : \| A \| M C _ { \alpha , \beta } ,
$$

Jensen’s inequality gives

$$
\mathsf { D } _ { \mu } ( x ) \geq \exp \left( \left. \int _ { \mathbb { R } ^ { d } } z \mu ( d z ) , A x \right. \right) \geq \exp ( - \| A \| M C _ { \alpha , \beta } ) .
$$

Moreover, since $\mu _ { R } \in \mathcal P ( B _ { R } )$ , Lemma B.13 gives $\| \mathrm { A t t } _ { \theta } ( \mu _ { R } , x ) \| \leq \| W \| \| V \| R .$ . Thus,

$$
\begin{array} { l } { \displaystyle \mathrm { ( I I ) } = \Big \| \frac { \mathsf { N } _ { \mu , R } ( x ) } { \mathsf { D } _ { \mu , R } ( x ) } \Big \| \frac { 1 } { \mathsf { D } _ { \mu } ( x ) } \big | \mathsf { D } _ { \mu } ( x ) - \mathsf { D } _ { \mu , R } ( x ) \big | } \\ { \displaystyle \qquad \leq \| W \| \| V \| R e ^ { \| A \| M C _ { \alpha , \beta } } \big | \mathsf { D } _ { \mu } ( x ) - \mathsf { D } _ { \mu , R } ( x ) \big | . } \end{array}
$$

By the mean value theorem, $| e ^ { s } - e ^ { t } | \leq e ^ { \operatorname* { m a x } ( s , t ) } | s - t |$ . Then

$$
\begin{array} { r l } {  { \big | \mathrm { D } _ { \mu } ( x ) - \mathrm { D } _ { \mu , R } ( x ) \big | \leq \int _ { \mathbb { R } ^ { d } } \big | e ^ { \langle y , A x \rangle } - e ^ { \langle \Pi _ { R } ( y ) , A x \rangle } \big | \mu ( d y ) } \ ~ } & { } \\ & { \leq \int _ { \mathbb { R } ^ { d } } e ^ { \operatorname* { m a x } ( \langle y , A x \rangle , \langle \Pi _ { R } ( y ) , A x \rangle ) } \big | \langle y - \Pi _ { R } ( y ) , A x \rangle \big | \mu ( d y ) } \\ & { \leq \| A \| M \int _ { \mathbb { R } ^ { d } } e ^ { \| A \| M \operatorname* { m a x } ( \| y \| , R ) } \| y - \Pi _ { R } ( y ) \| \mu ( d y ) } \\ & { \leq \| A \| M \int _ { \| y \| > R } e ^ { \| A \| M \| y \| } \big ( \| y \| - R \big ) \mu ( d y ) . } \end{array}
$$

We can bound the last line using the exponential uniform integrability implied by Lemma B.12.

Turning to term (I), it sufices to bound the diference in numerators because we have already bounded $\mathsf { D } _ { \mu } ( x )$ from below uniformly in $\mu$ and x. Let $f ( y ) : = y e ^ { \langle y , \lambda \rangle }$ . Then $D f ( y ) = e ^ { \langle y , \lambda \rangle } ( I + y \otimes \lambda )$ . Hence, the mean value theorem implies that

$$
\begin{array} { r } { \| f ( y ) - f ( z ) \| \leq e ^ { \operatorname* { m a x } ( \langle y , \lambda \rangle , \langle z , \lambda \rangle ) } \big ( 1 + \operatorname* { m a x } ( \| y \| , \| z \| ) \| \lambda \| \big ) \| y - z \| . } \end{array}
$$

Applying this result gives

$$
\begin{array} { r l r } {  { \frac { \| { \mathsf { N } } _ { \mu } ( x ) - { \mathsf { N } } _ { \mu , R } ( x ) \| } { \| W \| \| V \| } \leq \int _ { \mathbb { R } ^ { d } } \| y e ^ { \langle y , A x \rangle } - { \Pi } _ { R } ( y ) e ^ { \langle \Pi _ { R } ( y ) , A x \rangle } \| \mu ( d y ) } } \\ & { } & { \leq \int _ { \| y \| > R } e ^ { \| A \| M \| y \| } ( 1 + \| A \| M \| y \| ) ( \| y \| - R ) \mu ( d y ) , } \end{array}
$$

which is further bounded by Lemma B.12. Applying Lemma B.12 to bound the sub-Gaussian integrals and putting together the pieces completes the proof. □

## Proposition A.3. Define

$$
\Gamma ( \mu , x ) = x + \sum _ { h = 1 } ^ { H } W _ { h } \frac { \int e ^ { \langle x , A _ { h } y \rangle } V _ { h } y \mu ( d y ) } { \int e ^ { \langle x , A _ { h } z \rangle } \mu ( d z ) } ,
$$

where $A _ { 1 } , \dots , A _ { H } \in \mathbb { R } ^ { d \times d } , \ V _ { 1 } , \dots , V _ { H } \in \mathbb { R } ^ { k \times d }$ , and $W _ { 1 } , \dots , W _ { H } \in \mathbb { R } ^ { d \times k }$ . Set $c _ { A } = \operatorname* { m a x } _ { h \in [ H ] } \| A _ { h } \| , \ W _ { h } =$ max $\mathsf { \tilde { a } } _ { h \in [ H ] } \parallel W _ { h } \parallel$ , and $c _ { V } = \operatorname* { m a x } _ { h \in [ H ] } \left\| V _ { h } \right\|$ . Then for any $R > 0$ and $M > 0$ , it holds that

$$
\begin{array} { r l } & { \underset { \mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) } { \operatorname* { s u p } } \ \underset { \| x \| \leq M } { \operatorname* { s u p } } \left\| \Gamma ( \mu , x ) - \Gamma \big ( ( \Pi _ { R } ) _ { \# } \mu , x \big ) \right\| } \\ & { \quad \quad \leq H c _ { W } c _ { V } e ^ { c _ { A } M C _ { \alpha , \beta } } ( 2 + R ) \alpha c _ { A } M e ^ { \frac { \beta ^ { 2 } \alpha ^ { 2 } c _ { A } ^ { 2 } M ^ { 2 } } { 2 } } \bigg ( \frac { c _ { A } M \beta ^ { 2 } } { R } + 2 \beta \bigg ) e ^ { - \frac { R ^ { 2 } } { 2 \beta } } , } \end{array}
$$

where we have defined

$$
C _ { \alpha , \beta } : = \operatorname* { s u p } _ { \nu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) } \int \| x \| \nu ( d x ) < \infty .
$$

Proof. This follows directly from Proposition A.2.

Proof of Theorem 3.4. Throughout the proof, we use the notation $\mu ^ { r } = ( \Pi _ { r } ) _ { \# } \mu$ and $K = \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ . We begin with the error decomposition

$$
\begin{array} { r l } & { \mathsf { W } _ { 1 } ( \Gamma ( \mu ) _ { \# } \mu , \Gamma ( \mu _ { 0 } ) _ { \# } \mu _ { 0 } ) \le \mathsf { W } _ { 1 } ( \Gamma ( \mu ) _ { \# } \mu , \Gamma ( \mu ^ { r } ) _ { \# } \mu ^ { r } ) } \\ & { \qquad + \mathsf { W } _ { 1 } ( \Gamma ( \mu ^ { r } ) _ { \# } \mu ^ { r } , \Gamma ( \mu _ { 0 } ^ { r } ) _ { \# } \mu _ { 0 } ^ { r } ) } \\ & { \qquad + \mathsf { W } _ { 1 } ( \Gamma ( \mu _ { 0 } ^ { r } ) _ { \# } \mu _ { 0 } ^ { r } , \Gamma ( \mu _ { 0 } ) _ { \# } \mu _ { 0 } ) } \\ & { \qquad \le \underbrace { \mathsf { W } _ { 1 } ( \Gamma ( \mu ^ { r } ) _ { \# } \mu ^ { r } , \Gamma ( \mu _ { 0 } ^ { r } ) _ { \# } \mu _ { 0 } ^ { r } ) } _ { ( \mathrm { I } ) } + \underbrace { 2 \mathsf { s u p } \mathsf { W } _ { 1 } ( \Gamma ( \mu ) _ { \# } \mu , \Gamma ( \mu ^ { r } ) _ { \# } \mu ^ { r } ) } _ { ( \mathrm { I I I } ) } } \end{array}
$$

for any $r > 0$ . For term (I), since $\mu ^ { r }$ and $\mu _ { 0 } ^ { r }$ are supported in $B _ { r }$ , Lemma B.15 guarantees that

$$
\begin{array} { r } { ( I ) = \mathsf { W } _ { 1 } ( \Gamma ( \mu ^ { r } ) _ { \# } \mu ^ { r } , \Gamma ( \mu _ { 0 } ^ { r } ) _ { \# } \mu _ { 0 } ^ { r } ) \le C _ { c _ { A } , c _ { V } , c _ { W } } e ^ { 5 c _ { A } r ^ { 2 } } \mathsf { W } _ { 1 } ( \mu ^ { r } , \mu _ { 0 } ^ { r } ) \le C _ { c _ { A } , c _ { V } , c _ { W } } e ^ { 5 c _ { A } r ^ { 2 } } \mathsf { W } _ { 1 } ( \mu , \mu _ { 0 } ) , } \end{array}
$$

where the second inequality follows from the fact that Π<sub>r</sub> is 1-Lipschitz for any $r > 0$ . We next bound term (II). Since $\| \Pi _ { r } ( x ) \| \leq \operatorname* { m i n } ( r , \| x \| ) \leq \| x \|$ , we have $\mu ^ { r } \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ by (3.2). Then

$$
\begin{array} { r l } {  { \mathbb { W } _ { 1 } ( \Gamma ( \mu ) _ { \# } \mu , \Gamma ( \mu ^ { r } ) _ { \# } \mu ^ { r } ) \leq \mathbb { W } _ { 1 } ( \Gamma ( \mu ) _ { \# } \mu , \Gamma ( \mu ^ { r } ) _ { \# } \mu ) + \mathbb { W } _ { 1 } ( \Gamma ( \mu ^ { r } ) _ { \# } \mu , \Gamma ( \mu ^ { r } ) _ { \# } \mu ^ { r } ) } } \\ & { \leq \| \Gamma ( \mu ) - \Gamma ( \mu ^ { r } ) \| _ { L _ { \mu } ^ { 1 } } + \mathrm { L i p } ( \Gamma ( \mu ^ { r } ) ) \mathbb { W } _ { 1 } ( \mu , \mu ^ { r } ) } \\ & { \leq \underset { \| x \| \leq R } { \operatorname* { s u p } } \| \Gamma ( \mu , x ) - \Gamma ( \mu ^ { r } , x ) \| + \int _ { \| x \| > R } \Big ( \| \Gamma ( \mu , x ) \| + \| \Gamma ( \mu ^ { r } , x ) \| \Big ) \mu ( d x ) } \\ & { \qquad + \mathrm { L i p } ( \Gamma ( \mu ^ { r } ) ) \mathbb { W } _ { 1 } ( \mu , \mu ^ { r } ) } \\ & { \leq \underset { \| x \| \leq R } { \operatorname* { s u p } } \| \Gamma ( \mu , x ) - \Gamma ( \mu ^ { r } , x ) \| + 2 C _ { \alpha , \beta , V , Q } \int _ { \| x \| > R } \big ( 1 + \| x \| \big ) \mu ( d x )  } \\ & { \qquad +  ( 1 + H c _ { A } c _ { V } c _ { W } r ^ { 2 } ) \mathbb { W } _ { 1 } ( \mu , \mu ^ { r } )  } \end{array}
$$

for any $R > 0$ , where we used Lemma B.13 to bound the Lipschitz constant of $\Gamma ( \boldsymbol { \mu } ^ { r } )$ in the final line. Using Proposition A.3 to bound the supremum above, Lemma B.11 to bound the integral $\begin{array} { r } { \int _ { \| x \| > R } ( 1 + \| x \| ) \mu ( d x ) } \end{array}$ , and Lemma B.10 to bound the truncation error $\mathsf { W } _ { 1 } ( \mu , \mu ^ { r } )$ , we have the following bound for term (II):

$$
\begin{array} { r l r } {  { ( I I ) \leq H c _ { V } c _ { W } e ^ { c _ { A } R C _ { \alpha , \beta } } ( 2 + R ) \alpha c _ { A } R e ^ { \frac { \beta ^ { 2 } \alpha ^ { 2 } c _ { A } ^ { 2 } R ^ { 2 } } { 2 } } \bigg ( \frac { c _ { A } R \beta ^ { 2 } } { r } + 2 \beta \bigg ) e ^ { - \frac { r ^ { 2 } } { 2 \beta } } } } \\ & { } & \\ & { } & { + 2 C _ { \alpha , \beta , c _ { V } , c _ { A } , c _ { W } } \alpha \bigg ( \frac { \beta } { 2 R } + 1 + R \bigg ) e ^ { - R ^ { 2 } / \beta } + \frac { \alpha \beta } { 2 r } e ^ { - r ^ { 2 } / \beta } } \\ & { } & \\ & { } & { \leq H e ^ { \frac { C _ { \alpha , \beta } ^ { 2 } } { 2 \alpha ^ { 2 } \beta ^ { 2 } } } c _ { V } c _ { W } ( 2 + R ) \alpha c _ { A } R e ^ { \beta ^ { 2 } \alpha ^ { 2 } c _ { A } ^ { 2 } R ^ { 2 } } \bigg ( \frac { c _ { A } R \beta ^ { 2 } } { r } + 2 \beta \bigg ) e ^ { - \frac { r ^ { 2 } } { 2 \beta } } } \\ & { } & \\ & { } & { + 2 C _ { \alpha , \beta , c _ { V } , c _ { A } , c _ { W } } \alpha \bigg ( \frac { \beta } { 2 R } + 1 + R \bigg ) e ^ { - R ^ { 2 } / \beta } + \frac { \alpha \beta } { 2 r } e ^ { - r ^ { 2 } / \beta } , } \end{array}
$$

where the last inequality follows from Young’s inequality. To simplify the rest of this proof, we express this inequality as

$$
\begin{array} { r l } & { ( I I ) \le C _ { H , \alpha , \beta , c _ { { A } } , c _ { V } , c _ { W } } \Big ( R ( 1 + R ) ( R r ^ { - 1 } + 1 ) e ^ { \beta ^ { 2 } \alpha ^ { 2 } c _ { { A } } ^ { 2 } R ^ { 2 } - \frac { r ^ { 2 } } { 2 \beta } } + ( 1 + r + r ^ { - 1 } ) \big ( e ^ { - r ^ { 2 } / \beta } + e ^ { - R ^ { 2 } / \beta } \big ) \Big ) , } \end{array}
$$

where the constant $C _ { H , \alpha , \beta , c _ { A } , c _ { V } , c _ { W } }$ may change from line to line throughout the rest of the proof. To conclude the proof, we must choose r and R as functions of $\mathsf { W } _ { 1 } ( \mu , \mu _ { 0 } )$ . We first set

$$
R = \sqrt { \frac { 1 } { 2 ( \beta ^ { 3 } \alpha ^ { 2 } c _ { A } ^ { 2 } + 1 ) } } r ,
$$

so that the upper bound on term (II) becomes

$$
\begin{array} { r } { ( I I ) \le C _ { H , \alpha , \beta , c _ { A } , c _ { V } , c _ { W } } \bigl ( 1 + r ^ { 2 } + r ^ { - 1 } \bigr ) e ^ { - \operatorname* { m i n } \bigl ( \frac { 1 } { 2 ( \beta ^ { 3 } \alpha ^ { 2 } c _ { A } ^ { 2 } + 1 ) } , \frac { 1 } { \beta } \bigr ) r ^ { 2 } } . } \end{array}
$$

To balance terms (I) and (II), we set

$$
r = \sqrt { \frac { 1 } { k _ { 1 } + k _ { 2 } } \log _ { + } ( \mathsf { W } _ { 1 } ( \mu , \mu _ { 0 } ) ^ { - 1 } ) } ,
$$

where

$$
k _ { 1 } : = 5 c _ { A } \quad \mathrm { a n d } \quad k _ { 2 } : = \operatorname* { m i n } \biggl ( \frac { 1 } { \beta } , \frac { 1 } { 2 ( \beta ^ { 3 } \alpha ^ { 2 } c _ { A } ^ { 2 } + 1 ) } \biggr ) .\tag{A.1}
$$

This leads to the estimate

$$
\begin{array} { r l } & { \mathsf { W } _ { 1 } ( \Gamma ( \mu , \cdot ) _ { \# } \mu , \Gamma ( \mu _ { 0 } , \cdot ) _ { \# } \mu _ { 0 } ) } \\ & { \qquad \le C _ { H , \alpha , \beta , c _ { A } , c _ { V } , c _ { W } } \Big ( 1 + \operatorname* { m a x } \big ( \log ( \mathsf { W } _ { 1 } ( \mu , \mu _ { 0 } ) ) , \log _ { + } ( \mathsf { W } _ { 1 } ( \mu , \mu _ { 0 } ) ^ { - 1 } ) \big ) \Big ) \mathsf { W } _ { 1 } ( \mu , \mu _ { 0 } ) ^ { \frac { k _ { 2 } } { k _ { 1 } + k _ { 2 } } } , } \end{array}
$$

where $k _ { 1 }$ and $k _ { 2 }$ are defined in Equation (A.1). To conclude, we remove the log-factor at the expense of enlarging the constant factor and the H¨older estimate. This gives the final estimate

$$
\mathsf { W } _ { 1 } ( \Gamma ( \mu , \cdot ) _ { \# } \mu , \Gamma ( \mu _ { 0 } , \cdot ) _ { \# } \mu _ { 0 } ) \leq C _ { H , \alpha , \beta , c _ { A } , c _ { V } , c _ { W } } \mathsf { W } _ { 1 } ( \mu , \mu _ { 0 } ) ^ { \gamma }
$$

for $\begin{array} { r } { \gamma \in ( 0 , \frac { k _ { 2 } } { k _ { 1 } + k _ { 2 } } ) } \end{array}$ . We conclude the proof.

Proof of Theorem 3.5. We will prove the statement by induction on the number of layers L. For the base case, assume that T is an in-context mapping given by ${ \sf T } ( \mu , x ) = \phi ( { \Gamma } ( \mu , x ) )$ , where Γ is a multi-head self-attention layer and ϕ is a ReLU feedforward neural network. The mapping $\mu \mapsto \Gamma ( \mu , \cdot ) _ { \# } \mu$ is H¨older continuous on $\mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ and the mapping $\mu \mapsto \phi _ { \# } \mu$ is globally Lipschitz (since ϕ is a Lipschitz mapping). Therefore, the mapping $\mu \mapsto { \mathsf { T } } ( \mu ) _ { \# } \mu$ is H¨older continuous on $\mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ ; this establishes the base case. For the induction step, let T be an in-context composition of L multi-head self-attention layers and ReLU feedforward layers. Then we can write

$$
\mathsf { T } = ( \mathsf { T } _ { L } \circ \mathsf { T } _ { L - 1 } ) ,
$$

where $\mathsf { T } _ { L - 1 }$ is an in-context composition of $L - 1$ multi-head self-attention layers and ReLU feedforward layers, and $\mathsf { T } _ { L } = \phi _ { L } \diamondsuit \Gamma _ { L }$ is a multi-head self attention layer composed with a ReLU feedforward layer. By induction, there exist constants $C > 0$ and $\gamma \in ( 0 , 1 )$ such that $G _ { L - 1 }$ satisfies the estimate

$$
\mathsf { W } _ { 1 } ( \mathsf { T } _ { L - 1 } ( \mu , \cdot ) _ { \# } \mu , \mathsf { T } _ { L - 1 } ( \mu _ { 0 } , \cdot ) _ { \# } \mu _ { 0 } ) \le C \mathsf { W } _ { 1 } ( \mu , \mu _ { 0 } ) ^ { \gamma }
$$

for all $\mu$ and $\mu _ { 0 }$ in $\mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ . In addition, by Theorem 3.3, there exist constants $\alpha ^ { \prime } \geq 1$ and $\beta ^ { \prime } > 0 .$ , depending only on $\alpha , \beta ,$ and $\mathsf { T } _ { L - 1 }$ , such that $\mathsf { T } _ { L - 1 } ( \mu , \cdot ) _ { \# } \mu \in \mathrm { S G } _ { \alpha ^ { \prime } , \beta ^ { \prime } } ( \mathbb { R } ^ { k } )$ for all $\mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ . Since the base case argument guarantees that $\mathsf { T } _ { L }$ is H¨older continuous on $\mathrm { S G } _ { \alpha ^ { \prime } , \beta ^ { \prime } } ( \mathbb { R } ^ { k } )$ (say, with constant $C ^ { \prime } > 0$ and exponent $\gamma ^ { \prime } \in ( 0 , 1 ) )$ , it follows that

$$
\begin{array} { r l } & { \mathsf { W } _ { 1 } ( { \mathsf { T } } ( \mu , \cdot ) _ { \# } \mu , \mathsf { T } ( \mu _ { 0 } , \cdot ) _ { \# } \mu _ { 0 } ) = \mathsf { W } _ { 1 } ( ( \mathsf { T } _ { L } \diamond \mathsf { T } _ { L - 1 } ) ( \mu , \cdot ) _ { \# } \mu , ( \mathsf { T } _ { L } \diamond \mathsf { T } _ { L - 1 } ) ( \mu _ { 0 } , \cdot ) _ { \# } \mu _ { 0 } ) } \\ & { \qquad \leq C ^ { \prime } \mathsf { W } _ { 1 } ( \mathsf { T } _ { L - 1 } ( \mu , \cdot ) _ { \# } \mu , \mathsf { T } _ { L - 1 } ( \mu _ { 0 } , \cdot ) _ { \# } \mu _ { 0 } ) ^ { \gamma } } \\ & { \qquad \leq C C ^ { \prime } \mathsf { W } _ { 1 } ( \mu , \mu _ { 0 } ) ^ { \gamma \gamma ^ { \prime } } . } \end{array}
$$

We conclude the proof.

## A.3 Proofs for Section 3.3: Sample Complexity of Transformers

Proof of Theorem 3.7. Throughout this proof, C denotes a positive constant—depending only on $\alpha , \beta ,$ and the parameters of T—whose precise value may difer from line to line. Fix $\mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ , let $x _ { 1 } , \ldots , x _ { N } \sim \mu$ be iid,

and let $\begin{array} { r } { \mu _ { N } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \delta _ { x _ { i } } } \end{array}$ . Let $\mathcal { A } _ { N }$ denote the event on which $\mu _ { N } \in \mathrm { S G } _ { 3 + \alpha / 7 , 7 \beta }$ . By Proposition B.7, there exists a $N _ { \alpha , \beta } \in$ N such that

$$
\begin{array} { r } { \mathbb { P } ( \mathcal { A } _ { N } ) \geq 1 - 2 \alpha N ^ { - 2 } \quad \mathrm { f o r ~ a l l } \quad N \geq N _ { \alpha , \beta } . } \end{array}
$$

Notice that

$$
\begin{array} { r } { \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) \bigcup \mathrm { S G } _ { 3 + \alpha / 7 , 7 \beta } ( \mathbb { R } ^ { d } ) \subset \mathrm { S G } _ { \operatorname* { m a x } ( \alpha , 3 + \alpha / 7 ) , 7 \beta } ( \mathbb { R } ^ { d } ) . } \end{array}
$$

By Theorem 3.5, the mapping $\mu \mapsto { \mathsf { T } } ( \mu , \cdot ) _ { \# } \mu$ is H¨older continuous on $\mathrm { S G } _ { \operatorname* { m a x } ( \alpha , 3 + \alpha / 7 ) , 7 \beta } ( \mathbb { R } ^ { d } )$ with H¨older exponent $\gamma = \gamma ( \alpha , \beta , { \sf T } ) \in ( 0 , 1 )$ and constant $\bar { C } = C ( \alpha , \beta , { \sf T } ) > 0$ depending only on $\alpha , \beta ,$ and the parameters of T. We then proceed to estimate

$$
\begin{array} { r l } & { \mathbb { E } \mathbb { W } _ { 1 } ( { \mathsf { T } } ( \mu , \cdot ) _ { \# } \mu , { \mathsf { T } } ( \mu _ { N } , \cdot ) _ { \# } \mu _ { N } ) = \mathbb { E } \big [ \mathbb { W } _ { 1 } ( { \mathsf { T } } ( \mu , \cdot ) _ { \# } \mu , { \mathsf { T } } ( \mu _ { N } , \cdot ) _ { \# } \mu _ { N } ) ( \mathbf { 1 } _ { A _ { N } } + \mathbf { 1 } _ { A _ { N } ^ { c } } ) \big ] } \\ & { \qquad \leq C _ { G } \mathbb { E } [ \mathbb { W } _ { 1 } ( \mu , \mu _ { N } ) ^ { \gamma } ] + \mathbb { E } [ \mathbb { W } _ { 1 } ^ { 2 } ( { \mathsf { T } } ( \mu , \cdot ) _ { \# } \mu , { \mathsf { T } } ( \mu _ { N } , \cdot ) _ { \# } \mu _ { N } ) ] ^ { 1 / 2 } \cdot \sqrt { \mathbb { P } ( { \mathcal { A } } _ { N } ^ { c } ) } } \\ & { \qquad \leq C _ { G } \mathbb { E } [ \mathbb { W } _ { 1 } ( \mu , \mu _ { N } ) ] ^ { \gamma } + \mathbb { E } [ \mathbb { W } _ { 1 } ^ { 2 } ( { \mathsf { T } } ( \mu , \cdot ) _ { \# } \mu , { \mathsf { T } } ( \mu _ { N } , \cdot ) _ { \# } \mu _ { N } ) ] ^ { 1 / 2 } \cdot \frac { \sqrt { 2 \alpha } } { N } , } \end{array}\tag{A.2}
$$

where the second line follows from the Cauchy–Schwarz inequality and the final line follows from Jensen’s inequality. To estimate the second term, we have

$$
\begin{array} { r l } & { \mathbb { E } [ \mathbb { W } _ { 1 } ^ { 2 } ( \mathbb { T } ( \mu , \cdot ) _ { \# } \mu , \mathbb { T } ( \mu _ { N } , \cdot ) _ { \# } \mu _ { N } ) ] \leq 2 \mathbb { E } [ \mathbb { W } _ { 1 } ^ { 2 } ( \mathbb { T } ( \mu , \cdot ) _ { \# } \mu , \mathbb { T } ( \mu , \cdot ) _ { \# } \mu _ { N } ) ] + 2 \mathbb { E } [ \mathbb { W } _ { 1 } ^ { 2 } ( \mathbb { T } ( \mu , \cdot ) _ { \# } \mu _ { N } , \mathbb { T } ( \mu _ { N } , \cdot ) _ { \# } \mu _ { N } ) ] } \\ & { \quad \quad = : I + I I . } \end{array}
$$

For term $I , \mathrm { i f } \gamma _ { N }$ denotes any coupling of $\mu$ and $\mu _ { N }$ , we have

$$
\begin{array} { r l } { \mathbb { E } [ \mathcal { W } _ { 1 } ^ { 2 } ( \Gamma ( \mu , \cdot ) , \cdot \varphi , \Psi ( \mu , \cdot ) \varphi , \nu ) ] \leq \mathbb { E } \mathbb { W } _ { 2 } ^ { 2 } ( \Gamma ( \mu , \cdot ) , \cdot \varphi , \cdot \Gamma ( \mu , \cdot ) \varphi , \nu ) ] } \\ & { \leq \mathbb { E } \Bigg [ \Bigg [ \int \| \Gamma ( \mu , x ) - \Gamma ( \mu , x ^ { \prime } ) \| ^ { 2 } \gamma \Re ( d x , d x ^ { \prime } ) \Bigg ] } \\ & { \leq 2 \mathbb { E } \Bigg [ \int \left( \| \Gamma ( \mu , x ) \| ^ { 2 } + \Gamma ( \mu , x ^ { \prime } ) \| ^ { 2 } \right) \gamma \Re ( d x , d x ^ { \prime } ) \Bigg ] } \\ & { \leq 2 C \mathbb { E } \Bigg [ \int ( 1 + \| x \| ^ { 2 } + \| x ^ { \prime } \| ^ { 2 } ) \gamma \Re ( d x , d x ^ { \prime } ) \Bigg ] } \\ & { \leq 2 C \Bigg ( 1 + \int \| x \| ^ { 2 } \mu ( d x ) + \mathbb { E } \Bigg [ \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \| x _ { i } \| ^ { 2 } \Bigg ] \Bigg ) } \\ & { = 2 C \Bigg ( 1 + 2 \int \| x \| ^ { 2 } \mu ( d x ) \Bigg ) } \\ & { = O _ { \alpha , \delta , \Gamma } ( 1 ) , } \end{array}
$$

where the third line applies Lemma B.2 to deduce that $\| \mathsf { T } ( \mu , x ) \| ^ { 2 } + \| \mathsf { T } ( \mu , x ^ { \prime } ) \| ^ { 2 } \lesssim 1 + \| x \| ^ { 2 } + \| x ^ { \prime } \| ^ { 2 }$ , and the last line applies Lemma B.11. This shows that term I can be bounded by a constant independent of N and depending only on α and $\beta$ and the parameters of T. For term II, we have

$$
\begin{array} { r l } { \mathbb { E } [ \mathsf { W } _ { 1 } ^ { 2 } ( \mathsf { T } ( \mu , \cdot ) _ { \# } \mu _ { N } , \mathsf { T } ( \mu _ { N } , \cdot ) _ { \# } \mu _ { N } ) ] \leq \mathbb { E } [ \mathsf { W } _ { 2 } ^ { 2 } ( \mathsf { T } ( \mu , \cdot ) _ { \# } \mu _ { N } , \mathsf { T } ( \mu _ { N } , \cdot ) _ { \# } \mu _ { N } ) ] } & { } \\ & { \leq \mathbb { E } \Bigg [ \frac { 1 } { N } \displaystyle \sum _ { i = 1 } ^ { N } \| \mathsf { T } ( \mu , x _ { i } ) - \mathsf { T } ( \mu _ { N } , x _ { i } ) \| ^ { 2 } \Bigg ] } \\ & { \leq 2 \mathbb { E } \Bigg [ \frac { 1 } { N } \displaystyle \sum _ { i = 1 } ^ { N } \big ( \| \mathsf { T } ( \mu , x _ { i } ) \| ^ { 2 } + \| \mathsf { T } ( \mu _ { N } , x _ { i } ) \| ^ { 2 } \big ) \Bigg ] } \\ & { \leq 2 \mathbb { E } \Bigg [ \frac { 1 } { N } \displaystyle \sum _ { i = 1 } ^ { N } C ( 1 + \| x _ { i } \| ^ { 2 } ) + \frac { 1 } { N } \displaystyle \sum _ { i = 1 } ^ { N } \| \mathsf { T } ( \mu _ { N } , x _ { i } ) \| ^ { 2 } \Bigg ] . } \end{array}
$$

Lemma B.16 proves that

$$
\operatorname* { m a x } _ { i \in N } \| \mathsf { T } ( \mu _ { N } , x _ { i } ) \| \leq C \operatorname* { m a x } _ { n \in [ N ] } \| x _ { n } \| ,
$$

where the constant C depends only on the parameters of T. This proves that term II satisfies the bound

$$
\begin{array} { c } { I I \leq C \mathbb { E } \bigg [ 1 + \underset { i \in [ N ] } { \operatorname* { m a x } } \| x _ { i } \| ^ { 2 } \bigg ] } \\ { \leq C \big ( 1 + \log ( N ) \big ) , } \end{array}
$$

where we applied Lemma B.5 in the second line. Combining the estimates for terms I and $I I ,$ we conclude that

$$
\begin{array} { r } { \mathbb { E } [ \mathsf { W } _ { 1 } ^ { 2 } ( \mathsf { T } ( \mu , \cdot ) _ { \# } \mu , \mathsf { T } ( \mu _ { N } , \cdot ) _ { \# } \mu _ { N } ) ] ^ { 1 / 2 } \le C \sqrt { 1 + \log ( N ) } . } \end{array}
$$

Plugging this bound into (A.2), we obtain

$$
\mathbb { E } [ \mathsf { W } _ { 1 } ( \mathsf { T } ( \mu , \cdot ) _ { \# } \mu , \mathsf { T } ( \mu _ { N } , \cdot ) _ { \# } \mu _ { N } ) ] \leq C \Bigg ( \mathbb { E } [ \mathsf { W } _ { 1 } ( \mu , \mu _ { N } ) ] ^ { \gamma } + \frac { \sqrt { 1 + \log ( N ) } } { N } \Bigg ) .
$$

To conclude, Proposition B.18 guarantees that

$$
\begin{array} { r } { \mathbb { E } [ \mathsf { W } _ { 1 } ( \mu , \mu _ { N } ) ] \le C \sqrt { \log ( N ) \big ( 1 + \log ( 1 + N ^ { 1 / d } ) \big ) } N ^ { - 1 / d } , } \end{array}
$$

where the implicit constant depends only on $d , \alpha ,$ and $\beta .$ In turn, this proves that

$$
\begin{array} { r l } & { \mathbb { E } [ \mathsf { W } _ { 1 } ( \mathsf { T } ( \mu , \cdot ) _ { \# } \mu , \mathsf { T } ( \mu _ { N } , \cdot ) _ { \# } \mu _ { N } ) ] \leq C \Bigg ( \log ( N ) ^ { \gamma / 2 } \Big ( 1 + \log ( 1 + N ^ { 1 / d } ) \Big ) ^ { \gamma / 2 } N ^ { - \gamma / d } + \frac { \sqrt { 1 + \log ( N ) } } { N } \Bigg ) } \\ & { \qquad \leq C \log ( N ) ^ { \gamma / 2 } \Big ( 1 + \log ( 1 + N ^ { 1 / d } ) \Big ) ^ { \gamma / 2 } N ^ { - \gamma / d } , } \end{array}
$$

as was to be shown.

Proof of Proposition 3.8. Fix $\mu \in \mathcal P ( B _ { R } )$ . Lemma B.15, along with the fact that ReLU feedforward networks are globally Lipschitz continuous, guarantees that the mapping $\mu \mapsto \mathsf T ( \mu , \cdot ) _ { \# } \mu$ is Lipschitz continuous on $\mathcal { P } ( B _ { R } )$ with Lipschitz constant $C _ { R , \mathsf { T } }$ depending on R and the parameters of T. Therefore,

$$
\mathbb { E } \mathsf { W } _ { 1 } ( \mathsf { T } ( \mu , \cdot ) _ { \# } \mu , \mathsf { T } ( \mu _ { N } , \cdot ) _ { \# } \mu _ { N } ) \le C _ { R , \mathsf { T } } \mathbb { E } \mathsf { W } _ { 1 } ( \mu , \mu _ { N } ) .
$$

Then, by Proposition B.17, there is a constant $C _ { d }$ depending on the dimension d such that

$$
\begin{array} { r } { \mathbb { E } \mathsf { W } _ { 1 } ( \mu , \mu _ { N } ) \leq C _ { d } R \bigg ( 1 + \sqrt { \log ( 1 + N ^ { 1 / d } ) } \bigg ) N ^ { - 1 / d } . } \end{array}
$$

Combining these two inequalities $\mathrm { g i }$ ves the desired result.

## A.4 Proofs for Section 3.4: Transformers With Mean-Field Cross-Attention

Proof of Theorem 3.9. We seek to bound

$$
\mathsf { W } _ { 1 } ( \Gamma ( \boldsymbol { \mu } , \cdot ) _ { \# } \eta , \Gamma ( \boldsymbol { \mu } _ { 0 } , \cdot ) _ { \# } \eta _ { 0 } ) ,
$$

where $\mu , \mu _ { 0 } \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ and $\eta , \eta _ { 0 } \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d ^ { \prime } } )$ for $\alpha \geq 1$ and $\beta > 0$ . We decompose this as

$$
\begin{array} { r l } & { \mathsf { W } _ { 1 } ( \Gamma ( \mu , \cdot ) _ { \# } \eta , \Gamma ( \mu _ { 0 } , \cdot ) _ { \# } \eta _ { 0 } ) \le \mathsf { W } _ { 1 } ( \Gamma ( \mu , \cdot ) _ { \# } \eta , \Gamma ( \mu _ { 0 } , \cdot ) _ { \# } \eta ) + \mathsf { W } _ { 1 } ( \Gamma ( \mu _ { 0 } , \cdot ) _ { \# } \eta , \Gamma ( \mu _ { 0 } , \cdot ) _ { \# } \eta _ { 0 } ) } \\ & { \qquad = : I + I I . } \end{array}
$$

For term I, we introduce auxiliary parameters $R > 0$ and $r > 0 .$ , and further decompose it as

$$
\begin{array} { r l } & { I \leq \displaystyle \int \| \Gamma ( \mu , z ) - \Gamma ( \mu _ { 0 } , z ) \| \eta ( d z ) } \\ & { \quad \leq \displaystyle \int \left( \| \Gamma ( \mu , z ) - \Gamma ( \mu ^ { r } , z ) \| + \| \Gamma ( \mu ^ { r } , z ) - \Gamma ( \mu _ { 0 } ^ { r } , z ) \| + \| \Gamma ( \mu _ { 0 } ^ { r } , z ) - \Gamma ( \mu _ { 0 } , z ) \| \right) \eta ( d z ) } \\ & { \quad \leq 2 \underset { \quad \nu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) , \| z \| \leq R } { \operatorname* { s u p } } \| \Gamma ( \nu , z ) - \Gamma ( \nu ^ { r } , z ) \| + \underset { \| z \| \leq R } { \operatorname* { s u p } } \| \Gamma ( \mu ^ { r } , z ) - \Gamma ( \mu _ { 0 } ^ { r } , z ) \| } \\ & { \qquad + 3 \underset { \quad \nu , \upsilon _ { 0 } \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) , \rho \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) } { \operatorname* { s u p } } \int _ { \| z \| > R } \| \Gamma ( \nu , z ) - \Gamma ( \nu _ { 0 } , z ) \| \rho ( d z ) } \\ & { \quad = : I _ { I } + I _ { I I I } . } \end{array}
$$

By Proposition A.3, term $I _ { I }$ is bounded by

$$
\begin{array} { l } { { I _ { I } \le 2 H c _ { V } c _ { W } e ^ { c _ { A } R C _ { \alpha , \beta } } ( 2 + r ) \alpha c _ { A } R e ^ { \frac { \beta ^ { 2 } \alpha ^ { 2 } c _ { A } ^ { 2 } R ^ { 2 } } { 2 } } \bigg ( \frac { c _ { A } R \beta ^ { 2 } } { r } + 2 \beta \bigg ) e ^ { - \frac { r ^ { 2 } } { 2 \beta } } } } \\ { { \phantom { I } \le C _ { H , c _ { A } , c _ { V } , c _ { W } , \alpha , \beta } ( 1 + r ) R ( 1 + R r ^ { - 1 } ) e ^ { \beta ^ { 2 } \alpha ^ { 2 } c _ { A } ^ { 2 } R ^ { 2 } - \frac { r ^ { 2 } } { 2 \beta } } , } } \end{array}
$$

where the last line is an application of Young’s inequality. For term $I _ { I I }$ , we apply Corollary B.14, which bounds the Lipschitz constant of the mapping $\nu \mapsto \Gamma ( \nu , z )$ for $\nu \in \mathcal { P } ( B _ { r } )$ and $z \in B _ { R } ,$ , to obtain

$$
\begin{array} { r } { I _ { I I } \leq H e ^ { 4 c _ { A } R r } \operatorname* { m a x } ( c _ { V } c _ { W } ( 1 + c _ { A } r R ) , c _ { A } c _ { V } c _ { W } r R ) \mathsf { W } _ { 1 } ( \mu ^ { r } , \mu _ { 0 } ^ { r } ) } \\ { \leq H e ^ { 4 c _ { A } R r } \operatorname* { m a x } ( c _ { V } c _ { W } ( 1 + c _ { A } r R ) , c _ { A } c _ { V } c _ { W } r R ) \mathsf { W } _ { 1 } ( \mu , \mu _ { 0 } ) , } \end{array}
$$

where the last line follows from the fact that the truncation map is 1-Lipschitz for each $r > 0$ , and hence the mapping $\nu \mapsto \nu ^ { r }$ is 1-Lipschitz with respect to $\mathsf { W } _ { 1 }$ . For term $I _ { I I I }$ , we use the linear growth of $\Gamma ( \nu , z )$ with respect to z and the tail decay of $\rho ;$ this yields

$$
\begin{array} { r l } & { I _ { I I I } \le 3 \underset { \nu , \nu _ { 0 } , \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) , \rho \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d ^ { \prime } } ) } { \operatorname* { s u p } } \int _ { \| z \| > R } ( \| \Gamma ( \nu , z ) + \Gamma ( \nu _ { 0 } , z ) \| \rho ( d z ) } \\ & { \qquad \le 3 C _ { \alpha , \beta } \underset { \rho \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d ^ { \prime } } ) } { \operatorname* { s u p } } \int _ { \| z \| > R } ( 1 + \| z \| ) \rho ( d z ) } \\ & { \qquad \le 3 C _ { \alpha , \beta } \alpha \bigg ( 1 + \frac { \beta } { 2 R } + R \bigg ) e ^ { - R ^ { 2 } / \beta } , } \end{array}
$$

where the second line follows from Proposition 3.2 and the last line follows from Lemma B.11. Summing the estimates of terms $I _ { I } , I _ { I I }$ , and $I _ { I I I }$ furnishes a bound on term I.

Turning to term II, we let Π denote the optimal coupling between η and $\eta _ { 0 }$ . Then we have

$$
\begin{array} { r l } & { I I \leq \displaystyle \int \| \Gamma ( \mu _ { 0 } , z ) - \Gamma ( \mu _ { 0 } , z ^ { \prime } ) \| \Pi ( d z , d z ^ { \prime } ) } \\ & { \quad \leq \displaystyle \int \| \Gamma ( \mu _ { 0 } , z ) - \Gamma ( \mu _ { 0 } ^ { r } , z ) \| \eta ( d z ) + \displaystyle \int \| \Gamma ( \mu _ { 0 } , z ^ { \prime } ) - \Gamma ( \mu _ { 0 } ^ { r } , z ^ { \prime } ) \| \eta _ { 0 } ( d z ) + \displaystyle \int \| \Gamma ( \mu _ { 0 } ^ { r } , z ) - \Gamma ( \mu _ { 0 } ^ { r } , z ^ { \prime } ) \| \Pi ( d z , d z ^ { \prime } ) } \\ & { \quad \leq \displaystyle 2 _ { \nu \in \mathbb { S } _ { G , \nu } \times \mathbb { R } ^ { d } , \| z \| \leq R } \| \Gamma ( \nu , z ) - \Gamma ( \nu ^ { r } , z ) \| + \displaystyle 2 _ { \nu \in \mathbb { S } G _ { \alpha , \nu } ( \mathbb { R } ^ { d } ) , \rho \in \mathbb { S } G _ { \alpha , \nu } ( \mathbb { R } ^ { d } ) } \displaystyle \int _ { \| z \| > R } \| \Gamma ( \nu , z ) - \Gamma ( \nu ^ { r } , z ) \| \rho ( d z ) } \\ & { \qquad + \displaystyle \int \| \Gamma ( \mu _ { 0 } ^ { r } , z ) - \Gamma ( \mu _ { 0 } ^ { r } , z ^ { \prime } ) \| \Pi ( d z , d z ^ { \prime } ) } \\ & { \quad = : I I _ { I } + I I _ { I I } + I I _ { I I I } . } \end{array}
$$

Terms $I I _ { I }$ and $I I _ { I I }$ can be bounded using the estimates of terms $I _ { I }$ and $I _ { I I I }$ :

$$
\begin{array} { r l } & { I I _ { I } \le 2 H c _ { V } e ^ { c _ { A } R C _ { \alpha , \beta } } ( 2 + r ) \alpha c _ { A } R e ^ { \frac { \beta ^ { 2 } \alpha ^ { 2 } c _ { A } ^ { 2 } R ^ { 2 } } { 2 } } \bigg ( \frac { c _ { A } R \beta ^ { 2 } } { r } + 2 \beta \bigg ) e ^ { - \frac { r ^ { 2 } } { 2 \beta } } } \\ & { \qquad \le C _ { H , c _ { A } , c _ { V } , c _ { W } , \alpha , \beta } ( 1 + r ) R ( 1 + R r ^ { - 1 } ) e ^ { \beta ^ { 2 } \alpha ^ { 2 } c _ { A } ^ { 2 } R ^ { 2 } - \frac { r ^ { 2 } } { 2 \beta } } } \end{array}
$$

and

$$
I I _ { I I } \leq 2 C _ { \alpha , \beta } \alpha \bigg ( 1 + \frac { \beta } { 2 R } + R \bigg ) e ^ { - R ^ { 2 } / \beta } .
$$

To bound term $I I _ { I I I }$ , we can apply Corollary B.14, which guarantees that the mapping $z \mapsto \Gamma ( \nu , z ) { \mathrm { i s } } ( 1 + H c _ { A } c _ { V } r ^ { 2 } ) .$ Lipschitz for any $\nu \in \mathcal P ( B _ { r } )$ . This implies that

$$
\begin{array} { l } { { \displaystyle I I _ { I I I } \leq ( 1 + H c _ { A } c _ { V } r ^ { 2 } ) \int \| z - z ^ { \prime } \| \Pi ( d z , d z ^ { \prime } ) } } \\ { ~ } \\ { { \displaystyle = ( 1 + H c _ { A } c _ { V } r ^ { 2 } ) \mathsf { W } _ { 1 } ( \eta , \eta _ { 0 } ) . } } \end{array}
$$

To finish the proof, let us assume that $r \geq 1$ and $R \geq 1$ ; hence, the estimates $1 / r \le 1$ and $1 / R \le 1$ hold. Combining estimates for terms I and $I I ,$ and applying Young’s inequality as in the proof of Theorem 3.4, we have

$$
\begin{array} { r } { \mathbb { W } _ { 1 } ( \Gamma ( \mu , \cdot ) _ { \# } \eta , \Gamma ( \mu _ { 0 } , \cdot ) _ { \# } \eta _ { 0 } ) \le C \Big ( \underbrace { r R ^ { 2 } e ^ { \beta ^ { 2 } \alpha ^ { 2 } c _ { 4 } ^ { 2 } R ^ { 2 } - r ^ { 2 } / ( 2 \beta ) } } _ { I _ { I } + I I _ { I } } + \underbrace { R e ^ { - R ^ { 2 } / \beta } } _ { I _ { I I I } + I I _ { I I } } + \underbrace { r R e ^ { i c _ { A } r R } \mathbb { W } _ { 1 } ( \mu , \mu _ { 0 } ) } _ { I _ { I I } } + \underbrace { r ^ { 2 } \mathbb { W } _ { 1 } ( \eta , \eta _ { 0 } ) } _ { I _ { I I I } } \Big ) , } \end{array}
$$

where C denotes a constant which depends only on $\alpha , \beta , H , c _ { A } , c _ { V }$ , and $c _ { W }$ , and which may change from line to line. To finish the proof, we first choose R as a suitable function of r, and then choose r as a suitable function of $\mathsf { W } _ { 1 } ( \mu , \mu _ { 0 } )$ . By choosing

$$
R = \sqrt { \frac { 1 } { 2 ( \beta ^ { 3 } \alpha ^ { 2 } c _ { A } ^ { 2 } + 1 ) } } r ,
$$

the upper bound on $\mathsf { W } _ { 1 } ( \Gamma ( \boldsymbol { \mu } , \cdot ) _ { \# } \eta , \Gamma ( \boldsymbol { \mu } _ { 0 } , \cdot ) _ { \# } \eta _ { 0 } )$ simplifies to

$$
\begin{array} { r } { \mathsf { W } _ { 1 } ( \Gamma ( \mu , \cdot ) _ { \# } \eta , \Gamma ( \mu _ { 0 } , \cdot ) _ { \# } \eta _ { 0 } ) \le C \left( r ^ { 3 } e ^ { - \frac { r ^ { 2 } } { 2 \beta ( \beta ^ { 3 } \alpha ^ { 2 } c _ { A } ^ { 2 } + 1 ) } } + r ^ { 2 } e ^ { 2 \sqrt { \frac { 2 c _ { A } ^ { 2 } } { \beta ^ { 3 } \alpha ^ { 2 } c _ { A } ^ { 2 } + 1 } } r ^ { 2 } } \mathsf { W } _ { 1 } ( \mu , \mu _ { 0 } ) + r ^ { 2 } \mathsf { W } _ { 1 } ( \eta , \eta _ { 0 } ) \right) } \end{array}
$$

up to a change in the value of C from line to line. We write

$$
k _ { 1 } : = \frac { 1 } { 2 \beta ( \beta ^ { 3 } \alpha ^ { 2 } c _ { A } ^ { 2 } + 1 ) } \quad \mathrm { a n d } \quad k _ { 2 } : = 2 \sqrt { \frac { 2 c _ { A } ^ { 2 } } { \beta ^ { 3 } \alpha ^ { 2 } c _ { A } ^ { 2 } + 1 } }\tag{A.3}
$$

for the exponents above. We then set

$$
r = \sqrt { \frac { 1 } { k _ { 1 } + k _ { 2 } } \log ( \mathsf { W } _ { 1 } ( \mu , \mu _ { 0 } ) ^ { - 1 } ) } .
$$

Under the assumption that $r \geq 1$ and $R \geq 1$ , this leads to a final estimate of the form

$$
\mathsf { W } _ { 1 } ( \Gamma ( \mu , \cdot ) _ { \# } \mu , \Gamma ( \mu _ { 0 } , \cdot ) _ { \# } \mu _ { 0 } ) \le C \log ( \mathsf { W } _ { 1 } ( \mu , \mu _ { 0 } ) ^ { - 1 } ) ^ { 3 / 2 } ( \mathsf { W } _ { 1 } ( \mu , \mu _ { 0 } ) ^ { \gamma } + \mathsf { W } _ { 1 } ( \eta , \eta _ { 0 } ) ) ,
$$

where $\begin{array} { r } { \gamma : = \ \frac { k _ { 2 } } { k _ { 1 } + k _ { 2 } } } \end{array}$ , with $k _ { 1 }$ and $k _ { 2 }$ defined in (A.3). Notice that there exists a suficiently small $M \ =$ $M ( \alpha , \beta , H , c _ { A } , c _ { V } ) ^ { - } > 0$ such that if $\mathsf { W } _ { 1 } ( \mu , \mu _ { 0 } ) \le M$ , then the estimates $r \geq 1$ and $R \geq 1$ indeed hold. On the other hand, if $\mathsf { W } _ { 1 } ( \mu , \mu _ { 0 } ) > M$ , then Proposition 3.2 implies that

$$
\begin{array} { r l } { \mathbb { W } _ { 1 } \left( \Gamma ( \mu , \cdot ) _ { \# } \eta , \Gamma ( \mu _ { 0 } , \cdot ) _ { \# } \eta _ { 0 } \right) \le \mathbb { W } _ { 1 } \left( \Gamma ( \mu , \cdot ) _ { \# } \eta , \delta _ { 0 } \right) + \mathbb { W } _ { 1 } \left( \delta _ { 0 } , \Gamma ( \mu _ { 0 } , \cdot ) _ { \# } \eta _ { 0 } \right) } & { } \\ & { \le \displaystyle \int \| \Gamma ( \mu , x ) \| \eta ( d x ) + \displaystyle \int \| \Gamma ( \mu _ { 0 } , x ) \| \eta _ { 0 } ( d x ) } \\ & { \le 2 C ( \alpha , \beta , H , c _ { A } , c _ { V } ) \underset { \nu \in \mathbb S \mathbb G _ { \alpha , \delta } ( \mathbb { R } ^ { d ^ { \prime } } ) } { \operatorname* { s u p } } \int \left( 1 + \| x \| \right) \nu ( d x ) = : C ^ { \prime } ( \alpha , \beta , H , c _ { A } , c _ { V } ) . } \end{array}
$$

It therefore holds that

$$
\begin{array} { r l } & { \mathsf { W } _ { 1 } \big ( G ( \mu , \cdot ) _ { \# } \eta , G ( \mu _ { 0 } , \cdot ) _ { \# } \eta _ { 0 } \big ) \le C ^ { \prime } M ^ { - \gamma } \mathsf { W } _ { 1 } ( \mu , \mu _ { 0 } ) ^ { \gamma } } \\ & { \qquad \le C ^ { \prime } M ^ { - \gamma } \big ( \mathsf { W } _ { 1 } ( \mu , \mu _ { 0 } ) ^ { \gamma } + \mathsf { W } _ { 1 } ( \eta , \eta _ { 0 } ) \big ) } \\ & { \qquad \le C ^ { \prime } M ^ { - \gamma } \operatorname* { m a x } \big ( 1 , \log _ { + } ( \mathsf { W } _ { 1 } ( \mu , \mu _ { 0 } ) ^ { - 1 } ) ^ { 3 / 2 } \big ) \big ( \mathsf { W } _ { 1 } ( \mu , \mu _ { 0 } ) ^ { \gamma } + \mathsf { W } _ { 1 } ( \eta , \eta _ { 0 } ) \big ) . } \end{array}
$$

Enlarging constants completes the proof.

Proof of Theorem 3.10. Case 1: $\mu \in \mathcal P ( B _ { R } )$ and $\eta \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ : Fix $\mu \in \mathcal P ( B _ { R } ( \mathbb { R } ^ { d } ) )$ and $\eta \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ . By the triangle inequality, we have

$$
\begin{array} { r l } & { \mathbb { E } W _ { 1 } ( \Gamma ( \mu , \cdot ) _ { \# } \eta , \Gamma ( \mu _ { N } , \cdot ) _ { \# } \eta _ { M } ) \leq \mathbb { E } W _ { 1 } ( \Gamma ( \mu , \cdot ) _ { \# } \eta , \Gamma ( \mu _ { N } , \cdot ) _ { \# } \eta ) + \mathbb { E } W _ { 1 } ( \Gamma ( \mu _ { N } , \cdot ) _ { \# } \eta , \Gamma ( \mu _ { N } , \cdot ) _ { \# } \eta _ { M } ) } \\ & { \qquad = : I + I I . } \end{array}
$$

To bound term I, we introduce an auxiliary parameter $r > 0$ and use the estimate

$$
\begin{array} { l } { I \leq \mathbb { E } \displaystyle \int \| \Gamma ( \mu , x ) - \Gamma ( \mu _ { N } , x ) \| \eta ( d x ) } \\ { \leq \mathbb { E } \operatorname* { s u p } _ { \| x \| \leq r } \| \Gamma ( \mu , x ) - \Gamma ( \mu _ { N } , x ) \| + \mathbb { E } \displaystyle \int _ { \| x \| > r } \| \Gamma ( \mu , x ) - \Gamma ( \mu _ { N } , x ) \| \eta ( d x ) } \\ { = : I _ { I } + I _ { I I } . } \end{array}
$$

To bound term $I _ { I }$ , we apply Corollary B.14, which controls the Lipschitz constant of the mapping $\mu \mapsto \Gamma ( \mu , x )$ for $x \in B _ { r }$ and $\mu \in \mathcal P ( B _ { R } )$

$$
\begin{array} { r l } & { I _ { I } \leq H e ^ { 4 c _ { A } r ^ { 2 } } \operatorname* { m a x } ( c _ { V } c _ { W } ( 1 + c _ { A } r R ) , c _ { A } c _ { V } c _ { W } r R ) \mathbb { E } \mathbb { W } _ { 1 } ( \mu , \mu _ { N } ) } \\ & { \quad \leq H e ^ { 4 c _ { A } r ^ { 2 } } \operatorname* { m a x } ( c _ { V } c _ { W } ( 1 + c _ { A } r R ) , c _ { A } c _ { V } c _ { W } r R ) C _ { d } R \bigg ( 1 + \sqrt { \log ( 1 + N ^ { 1 / d } ) } \bigg ) N ^ { - 1 / d } , } \end{array}
$$

where the last line follows from Wasserstein law of large numbers stated in Proposition B.17.

To bound term $I _ { I I }$ , note that since both $\mu$ and $\mu _ { N }$ belong to $\mathcal { P } ( B _ { R } )$ , it holds by Lemma B.4 that there exists an $\alpha _ { R } \geq 1$ and $\beta _ { R } > 0$ such that $\mu$ and $\mu _ { N }$ belong to $\mathrm { S G } _ { \alpha _ { R } , \beta _ { R } } ( \mathbb { R } ^ { d } )$ . Using Proposition 3.2, it follows that there exists a constant C depending on $\alpha _ { R }$ and $\beta _ { R }$ such that

$$
\begin{array} { r l } & { I _ { I I } \leq \mathbb E \displaystyle \int _ { \| x \| > r } \left( \| \Gamma ( \mu , x ) \| + \| \Gamma ( \mu _ { N } , x ) \| \right) \eta ( d x ) } \\ & { \quad \quad \leq C \displaystyle \int _ { \| x \| > r } ( 1 + \| x \| ) \eta ( d x ) } \\ & { \quad \quad \lesssim e ^ { - r ^ { 2 } / \beta } , } \end{array}
$$

where the last line follows from Lemma B.11.

For term II, Corollary B.14 guarantees that the Lipschitz constant of the map $\Gamma ( \nu , \cdot )$ is bounded by $( 1 + H c _ { A } c _ { V } R ^ { 2 } )$ for any $\nu \in \mathcal P ( B _ { R } )$ . It follows from Proposition B.18 that

$$
\begin{array} { r l } & { I I \leq \bigl ( 1 + H c _ { A } c _ { V } c _ { W } R ^ { 2 } \bigr ) \mathbb { E } \mathsf { W } _ { 1 } ( \eta , \eta _ { M } ) } \\ & { \qquad \leq \bigl ( 1 + H c _ { A } c _ { V } c _ { W } R ^ { 2 } \bigr ) C _ { d , \alpha , \beta } \sqrt { \log ( M ) \bigl ( 1 + \log ( 1 + M ^ { 1 / d } ) \bigr ) } M ^ { - 1 / d } } \end{array}
$$

for all $M \in \mathbb { N }$ . To finish the proof, we set the value of r as

$$
r = \sqrt { \frac { \beta } { d ( 4 c _ { A } \beta + 1 ) } \log ( N ) } ,
$$

which leads to the estimates

$$
e ^ { 4 c _ { A } r ^ { 2 } } N ^ { - 1 / d } = e ^ { - r ^ { 2 } / \beta } = N ^ { - \frac { 1 } { d ( 4 c _ { A } \beta + 1 ) } } .
$$

This concludes the proof of the first case.

Case 2: $\mu \in \mathcal P ( B _ { R } )$ and $\eta \in \mathcal { P } ( B _ { R } )$ : Our proof is similar to the proof of results from work of Bohbot et al. (2026, Section 4.2) and involves controlling the covering numbers of certain function classes associated with multi-head attention layers. Fixing $\mu \in \mathcal P ( B _ { R } )$ and $\gamma \in \mathcal { P } ( B _ { R } )$ , it holds that

$$
\begin{array} { r l } & { \mathbb { E } W _ { 1 } ( \Gamma ( \mu , \cdot ) _ { \# } \eta , \Gamma ( \mu _ { N } , \cdot ) _ { \# } \eta _ { M } ) \leq \mathbb { E } W _ { 1 } ( \Gamma ( \mu , \cdot ) _ { \# } \eta , \Gamma ( \mu _ { N } , \cdot ) _ { \# } \eta ) + \mathbb { E } W _ { 1 } ( \Gamma ( \mu _ { N } , \cdot ) _ { \# } \eta , \Gamma ( \mu _ { N } , \cdot ) _ { \# } \eta _ { M } ) } \\ & { \qquad = : I + I I . } \end{array}
$$

For term I, we have the bound

$$
\begin{array} { r l } & { I \leq \underset { x \in B _ { R } } { \mathbb { E } } \ \lVert \Gamma ( \mu , x ) - \Gamma ( \mu _ { N } , x ) \rVert } \\ & { \leq H c _ { W } \cdot \underset { \lVert A \rVert \leq c _ { A } , \lVert V \rVert \leq c _ { V } } { \operatorname* { m a x } } \ \underset { x \in B _ { R } } { \mathbb { E } } \bigg \lVert \frac { \int e ^ { \langle x , A y \rangle } V y \mu ( d y ) } { \int e ^ { \langle x , A y \rangle } \mu ( d y ) } - \frac { \int e ^ { \langle x , A y \rangle } V y \mu _ { N } ( d y ) } { \int e ^ { \langle x , A y \rangle } \mu _ { N } ( d y ) } \bigg \rVert , } \end{array}
$$

where we used the definition of Γ as a multi-head attention layer. We further decompose term I:

$$
\begin{array} { r l } & { \mathbb { E } \underset { x \in B _ { R } } { \operatorname* { s u p } } \left\| \int \frac { e ^ { \langle x , A y \rangle } V y \mu ( d y ) } { \int e ^ { \langle x , A y \rangle } \mu ( d y ) } - \frac { \int e ^ { \langle x , A y \rangle } V y \mu _ { N } ( d y ) } { \int e ^ { \langle x , A y \rangle } \mu _ { N } ( d y ) } \right\| } \\ & { \leq \mathbb { E } \underset { x \in B _ { R } } { \operatorname* { s u p } } \left( \left\| \frac { \int e ^ { \langle x , A y \rangle } V y ( \mu - \mu _ { N } ) ( d y ) } { \int e ^ { \langle x , A y \rangle } \mu ( d y ) } \right\| + \left\| \int e ^ { \langle x , A y \rangle } V y \mu _ { N } ( d y ) \left( \frac { 1 } { \int e ^ { \langle x , A y \rangle } \mu ( d y ) } - \frac { 1 } { \int e ^ { \langle x , A y \rangle } \mu _ { N } ( d y ) } \right) \right\| \right) } \\ & { \leq e ^ { c _ { 4 } R ^ { 2 } } \underset { x \in B _ { R } } { \operatorname* { s u p } } \left\| \int e ^ { \langle x , A y \rangle } V y ( \mu - \mu _ { N } ) ( d y ) \right\| + c _ { V } R e ^ { 3 c _ { 4 } R ^ { 2 } } \underset { x \in B _ { R } } { \operatorname* { s u p } } \left\| \int e ^ { \langle x , A y \rangle } ( \mu - \mu _ { N } ) ( d y ) \right\| } \\ & { = : I _ { I } + I _ { H } . } \end{array}
$$

We bound terms $I _ { I }$ and $I _ { I I }$ using covering number estimates of appropriate function spaces. In more detail, let us define the function spaces

$$
{ \mathcal { F } } _ { 1 , i } ( R ) : = \{ y \mapsto e ^ { \langle x , A y \rangle } ( V y ) _ { i } \colon x \in B _ { R } \}
$$

and

$$
{ \mathcal { F } } _ { 2 } ( R ) : = \{ y \mapsto e ^ { \langle x , A y \rangle } : x \in B _ { R } \} .
$$

Then by Dudley’s chaining estimate, there exists a universal constant $C > 0$ such that

$$
I _ { I } \leq \frac { C e ^ { c _ { A } R ^ { 2 } } \sqrt { d } } { \sqrt { N } } \operatorname* { m a x } _ { i \in [ d ] } \int _ { 0 } ^ { c _ { V } R e ^ { c _ { A } R ^ { 2 } } } \sqrt { \log \Bigl ( \mathcal { N } ( \mathcal { F } _ { 1 , i } ( R ) ; \epsilon , \| \cdot \| _ { L ^ { \infty } ( B _ { R } ) } ) \Bigr ) } d \epsilon
$$

and

$$
I _ { I I } \leq \frac { C c _ { V } R e ^ { 3 c _ { A } R ^ { 2 } } } { \sqrt { N } } \int _ { 0 } ^ { e ^ { c _ { A } R ^ { 2 } } } \sqrt { { \log } \Big ( \mathcal { N } ( \mathcal { F } _ { 2 } ( R ) ; \epsilon , \| \cdot \| _ { L ^ { \infty } ( B _ { R } ) } ) \Big ) } d \epsilon .
$$

Lemma B.20 bounds the covering numbers of the functions classes $\mathcal { F } _ { 1 , i } ( R )$ and $\mathcal { F } _ { 2 } ( R )$ . Plugging these covering number bounds into the estimates on $I _ { I }$ and $I _ { I I }$ , we deduce that

$$
\begin{array} { l } { { I _ { I } \leq \displaystyle \frac { C e ^ { c _ { A } R ^ { 2 } } d } { \sqrt { N } } \int _ { 0 } ^ { c _ { V } R e ^ { c _ { A } R ^ { 2 } } } \sqrt { \log ( 1 + 2 c _ { A } c _ { V } R ^ { 3 } e ^ { c _ { A } R ^ { 2 } } \epsilon ^ { - 1 } ) } d \epsilon } } \\ { { = \displaystyle \frac { 2 C c _ { A } c _ { V } R ^ { 3 } e ^ { 2 c _ { A } R ^ { 2 } } d } { \sqrt { N } } \int _ { 0 } ^ { \frac { 1 } { 2 } c _ { A } ^ { - 1 } R ^ { - 2 } } \sqrt { \log ( 1 + x ^ { - 1 } ) } d x } } \\ { { \leq \displaystyle \frac { C \sqrt { 8 } c _ { A } ^ { 1 / 2 } c _ { V } R ^ { 2 } e ^ { 2 c _ { A } R ^ { 2 } } d } { \sqrt { N } } , } } \end{array}
$$

where we used Lemma B.21 in the final line to bound the logarithmic integral. Similarly,

$$
\begin{array} { r l } {  { I _ { I I } \leq \frac { C c _ { V } R e ^ { 3 c _ { A } R ^ { 2 } } d ^ { 1 / 2 } } { \sqrt { N } } \int _ { 0 } ^ { e ^ { c _ { A } R ^ { 2 } } } \sqrt { \log ( 1 + 2 c _ { A } R ^ { 2 } e ^ { c _ { A } R ^ { 2 } } \epsilon ^ { - 1 } ) } d \epsilon } } \\ & { = \frac { 2 C c _ { V } c _ { A } R ^ { 3 } e ^ { 4 c _ { A } R ^ { 2 } } d ^ { 1 / 2 } } { \sqrt { N } } \int _ { 0 } ^ { \frac { 1 } { 2 } R ^ { - 2 } } \sqrt { \log ( 1 + x ^ { - 1 } ) } d x } \\ & { \leq \frac { C \sqrt { 8 } c _ { V } e ^ { 4 c _ { A } R ^ { 2 } } d ^ { 1 / 2 } } { \sqrt { N } } . } \end{array}
$$

Combining the estimates for $I _ { I }$ and $I _ { I I }$ , we obtain

$$
I \leq 4 C H c _ { V } c _ { W } \operatorname* { m a x } \biggl ( c _ { A } ^ { 1 / 2 } R ^ { 2 } e ^ { 2 c _ { A } R ^ { 2 } } d , e ^ { 4 c _ { A } R ^ { 2 } } d ^ { 1 / 2 } \biggr ) N ^ { - 1 / 2 } .
$$

To bound term II, Corollary B.14 ensures that the Lipschitz constant of the mapping $x \mapsto \Gamma ( \mu _ { N } , x )$ is at most $1 + H c _ { A } c _ { V } c _ { W } R ^ { 2 }$ with probability 1. Therefore,

$$
\begin{array} { r l } & { I I \leq ( 1 + H c _ { A } c _ { V } c _ { W } R ^ { 2 } ) \mathbb { E } \mathbb { W } _ { 1 } ( \eta , \eta _ { M } ) } \\ & { \quad \leq ( 1 + H c _ { A } c _ { V } c _ { W } R ^ { 2 } ) C _ { d } R \bigg ( 1 + \sqrt { \log ( 1 + M ^ { 1 / d } ) } \bigg ) M ^ { - 1 / d } } \end{array}
$$

where the last line follows from an application of Proposition B.17.

## A.5 Proof of Theorem 3.11

Proof. Given $\epsilon > 0$ , Theorem 4.4 due to Furuya et al. (2026b) ensures the existence of a transformer ${ { \sf T } _ { \epsilon } }$ such that

$$
\operatorname* { s u p } _ { \mu \in K } \mathsf { W } _ { 1 } ( \mathsf { T } _ { \epsilon } ( \mu , \cdot ) _ { \# } \mu , F ( \mu ) ) \leq \epsilon / 2 .
$$

Then, by the triangle inequality, we have

$$
\begin{array} { r l } { \underset { \mu \in \mathcal K } { \operatorname* { s u p } } \mathbb { E } W _ { 1 } ( \mathsf T _ { \epsilon } ( \mu _ { N } , \cdot ) _ { \# } \mu _ { N } , F ( \mu ) ) \leq \underset { \mu \in \mathcal K } { \operatorname* { s u p } } \mathbb { E } W _ { 1 } ( \mathsf T _ { \epsilon } ( \mu _ { N } , \cdot ) _ { \# } \mu _ { N } , \mathsf T _ { \epsilon } ( \mu , \cdot ) _ { \# } \mu ) + \underset { \mu \in \mathcal K } { \operatorname* { s u p } } W _ { 1 } ( \mathsf T _ { \epsilon } ( \mu , \cdot ) _ { \# } \mu , F ( \mu ) ) } & { } \\ { } & { \leq \underset { \mu \in \mathcal K } { \operatorname* { s u p } } \mathbb { E } W _ { 1 } ( \mathsf T _ { \epsilon } ( \mu _ { N } , \cdot ) _ { \# } \mu _ { N } , \mathsf T _ { \epsilon } ( \mu , \cdot ) _ { \# } \mu ) + \epsilon / 2 } \\ { } & { \leq C _ { \epsilon } \log ( N ) ^ { \gamma _ { \epsilon } / 2 } \Big ( 1 + \log ( 1 + N ^ { 1 / d } ) \Big ) ^ { \gamma _ { \epsilon } / 2 } N ^ { - \gamma _ { \epsilon } / d } + \epsilon / 2 , } \end{array}
$$

where the last inequality follows from an application of Theorem 3.7. We conclude the proof.

## B AUXILIARY LEMMAS

This appendix contains auxiliary lemmas that support the proofs of the main results.

## B.1 Properties of Sub-Gaussian Measures

We record some useful properties of sub-Gaussian measures. We begin with the following result which characterizes the compactness and uniform integrability properties of the sets $\{ \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) \} _ { \alpha \geq 1 , \beta > 0 } .$

Lemma B.1. For any $\alpha \geq 1$ and $\beta > 0$ , the set $\mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ is $\mathsf { W _ { 1 } } - c o m p a c t .$ Moreover, for any $p > 0$ , it holds that

$$
\operatorname* { s u p } _ { \mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) } \int _ { \mathbb { R } ^ { d } } \| x \| ^ { p } \mu ( d x ) < \infty \quad a n d \quad \operatorname* { s u p } _ { \mu \in \mathrm { S G } _ { \alpha , \beta } } \int _ { \mathbb { R } ^ { d } } e ^ { p \| x \| } \mu ( d x ) < \infty .
$$

Proof. We first prove the second assertion. For any $\mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ , we have by Fubini’s theorem that

$$
\begin{array} { r l r } {  { \int \| x \| ^ { p } \mu ( d x ) = \int _ { 0 } ^ { \infty } \mu ( \| x \| ^ { p } > t ) d t } } \\ & { } & { = \int _ { 0 } ^ { \infty } \mu ( \| x \| > t ^ { 1 / p } ) d t } \\ & { } & { \leq \alpha \int _ { 0 } ^ { \infty } \exp ( - \frac { t ^ { 2 / p } } { \beta } ) d t . } \end{array}
$$

This upper bounds $\textstyle \int \| x \| ^ { p } \mu ( d x )$ by a convergent integral which only depends on $\mu$ through its sub-Gaussian parameters. The proof of the exponential moment bounds is essentially the same.

For the first assertion, we use the characterization due to Villani (2009, Section 7.2) that a sequence of measures $\{ \mu _ { n } \} _ { n = 1 } ^ { \infty }$ converges to a measure $\mu$ in $\mathsf { W } _ { 1 }$ if and only if $\mu _ { n }  \mu$ and $\begin{array} { r } { \int \| x \| \mu _ { n } ( d x )  \int \| x \| \mu ( d x ) } \end{array}$ . Let $\{ \mu _ { n } \} _ { n \geq 1 }$ be a sequence in $\mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ . Then clearly $\{ \mu _ { n } \} _ { n \geq 1 }$ is tight, so by Prokhorov’s theorem (after passing to a subsequence), $\{ \mu _ { n } \} _ { n \ge 1 }$ has a weak limit $\mu \in \mathcal P ( \mathbb { R } ^ { d } )$ . To prove that $\mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ , let $B _ { R } \subset \mathbb { R } ^ { d }$ denote the closed ball of radius $R$ centered at zero. Then, since $\mu _ { n }  \mu _ { \mathrm { : } }$ , we have by the Portmanteau Theorem that

$$
1 - \alpha \exp ( - \frac { R ^ { 2 } } { \beta } ) \leq \operatorname* { l i m } _ { n  \infty } \int _ { B _ { R } } \mu _ { n } ( d x ) \leq \int _ { B _ { R } } \mu ( d x ) ,
$$

and therefore

$$
\int _ { \| x \| > R } \mu ( d x ) \leq \alpha \exp \biggl ( - \frac { R ^ { 2 } } { \beta } \biggr ) ,
$$

which proves that $\mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ . Last, to prove that the first moments of $\{ \mu _ { n } \} _ { n \geq 1 }$ converge to the first moment of $\mu _ { : }$ define the functions $f _ { R } ( x ) = \mathrm { m i n } ( \| x \| , R )$ . Then we have

$$
\begin{array} { l } { \displaystyle \left. \int \| x \| \mu _ { n } ( d x ) - \int \| x \| \mu ( d x ) \right. } \\ { \displaystyle \leq \left. \int f _ { R } ( x ) \mu _ { n } ( d x ) - \int f _ { R } ( x ) \mu ( d x ) \right. + \left. \int _ { \| x \| > R } ( \| x \| - R ) \mu _ { n } ( d x ) - \int _ { \| x \| > R } ( \| x \| - R ) \mu ( d x ) \right. } \\ { \displaystyle = : I + I I . } \end{array}
$$

Since both the sequence elements $\{ \mu _ { n } \} _ { n \geq 1 }$ and the weak limit $\mu$ belong to $\mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ , the second term can be bounded by Fubini’s theorem:

$$
\begin{array} { r l } & { I I \le 2 \underset { \nu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) } { \operatorname* { s u p } } \int _ { \| x \| > R } ( \| x \| - R ) \nu ( d x ) } \\ & { \quad \le 2 \underset { \nu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) } { \operatorname* { s u p } } \int _ { R } ^ { \infty } \nu ( \| x \| > t ) d t } \\ & { \quad \le 2 \alpha \int _ { R } ^ { \infty } \exp \left( - \frac { t ^ { 2 } } { \beta } \right) d t } \\ & { \quad = O ( \exp \bigl ( - \Omega ( R ^ { 2 } ) \bigr ) . } \end{array}
$$

Given $\epsilon > 0$ , we may choose $R _ { \epsilon } > 0$ large enough that the above expression is no larger than $\epsilon / 2 ,$ , and hence

$$
\left| \int \left\| x \right\| \mu _ { n } ( d x ) - \int \left\| x \right\| \mu ( d x ) \right| \leq \left| \int f _ { R _ { \epsilon } } ( x ) \mu _ { n } ( d x ) - \int f _ { R _ { \epsilon } } ( x ) \mu ( d x ) \right| + \epsilon / 2 .
$$

Since $f _ { R _ { \epsilon } }$ is bounded and continuous and $\mu _ { n }  \mu$ , we can choose $n _ { \epsilon } \in \mathbb { N }$ depending on $R _ { \epsilon }$ such that

$$
\left| \int f _ { R _ { \epsilon } } ( x ) \mu _ { n } ( d x ) - \int f _ { R _ { \epsilon } } ( x ) \mu ( d x ) \right| < \epsilon / 2 \quad \mathrm { f o r ~ a l l } \quad n \geq n _ { \epsilon } .
$$

This ensures that the first moment of $\mu _ { n }$ is within ϵ of the first moment of $\mu$ for all $n \geq n _ { \epsilon }$

Another important property, which plays a crucial role in the results of Section 3.1, is that the set $\operatorname { S G } ( \mathbb { R } ^ { d } )$ is closed under certain pushforwards.

Lemma B.2. Fix $\alpha \geq 1$ and $\beta > 0 . \ I f \mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ and $T \colon  { \mathbb { R } } ^ { d } \to  { \mathbb { R } } ^ { k }$ satisfies $\| T ( x ) \| \leq a + b \| x \|$ for positive constants a and b, then $T _ { \# } \mu \in \mathrm { S G } _ { \widetilde { \alpha } , \widetilde { \beta } } ( \mathbb { R } ^ { k } )$ , where $\widetilde { \alpha } = \alpha e ^ { a ^ { 2 } / ( b ^ { 2 } \beta ) }$ and ${ \widetilde { \beta } } = 2 b ^ { 2 } \beta$

Proof. By our assumptions, we have $\{ \| T ( x ) \| > R \} \subseteq \{ | x | > ( R - a ) / b \}$ . Then

$$
\begin{array} { r l } { \displaystyle \int _ { | \tau | > R } ( T _ { + } \mu ) ( d x ) = \int _ { | \tau | > R } \mu ( d x ) } \\ { \displaystyle } & { \le \int _ { | \tau | > ( R - a ) | \beta } \mu ( d x ) } \\ { \displaystyle } & { \le \alpha \exp \left( - \frac { ( R - a ) ^ { 2 } } { b ^ { 2 } \beta } \right) } \\ & { = \alpha \exp \left( \frac { - \alpha ^ { 2 } } { b ^ { 2 } \beta } \right) \exp \left( \frac { - R ^ { 2 } + 2 \alpha R } { b ^ { 2 } \beta } \right) } \\ { \displaystyle } & { \le \alpha \exp \left( \frac { - \alpha ^ { 2 } } { b ^ { 2 } \beta } \right) \exp \left( \frac { - ( 1 / 2 ) R ^ { 2 } + 2 \alpha ^ { 2 } } { b ^ { 2 } \beta } \right) } \\ & { = \tilde { C } _ { 1 } \exp \left( - \frac { R ^ { 2 } } { \tilde { C } _ { 2 } } \right) , } \end{array}
$$

where the final inequality is an application of Young’s inequality.

The following result characterizes sub-Gaussian sets in terms of Gaussian moment bounds.

Lemma B.3. Let $\alpha \geq 1$ and $\beta > 0$ . Let $\mu \in \mathcal P ( \mathbb { R } ^ { d } )$

1. If $\begin{array} { r } { { \bf \ddot { \rho } } _ { \mathbb { R } ^ { d } } \exp \left( \frac { \| x \| ^ { 2 } } { \beta } \right) \mu ( d x ) \leq \alpha _ { : } } \end{array}$ , then $\mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$

2. Conversely, if $\mu \in \operatorname { S G } _ { \alpha , \beta }$ , then

$$
\int _ { { \mathbb R } ^ { d } } \exp \left( \frac { \| x \| ^ { 2 } } { M } \right) \mu ( d x ) \leq 1 + \frac { \alpha \beta } { M - \beta } \quad f o r \ a l l \quad M > \beta .
$$

Proof. The first assertion follows from the standard technique of applying Markov’s inequality to the exponentia moment:

$$
\begin{array} { r l } & { \mu ( \| x \| > R ) = \mu \biggl ( \exp \biggl ( \displaystyle \frac { \| x \| ^ { 2 } } { \beta } \biggr ) > \exp \biggl ( \frac { R ^ { 2 } } { \beta } \biggr ) \biggr ) } \\ & { \qquad \leq \biggl ( \displaystyle \int \exp \biggl ( \frac { \| x \| ^ { 2 } } { \beta } \biggr ) \mu ( d x ) \biggr ) \exp \biggl ( - \frac { R ^ { 2 } } { \beta } \biggr ) } \\ & { \qquad \leq \alpha \exp \biggl ( - \frac { R ^ { 2 } } { \beta } \biggr ) . } \end{array}
$$

For the second assertion, by the layer cake formula with derivatives, we have

$$
\begin{array} { l } { \displaystyle \int \exp \left( \frac { \| x \| ^ { 2 } } { M } \right) \mu ( d x ) = 1 + \frac { 2 } { M } \int _ { 0 } ^ { \infty } t \exp \left( \frac { t ^ { 2 } } { M } \right) \mu ( \| x \| > t ) d t } \\ { \displaystyle \qquad \leq 1 + \frac { 2 \alpha } { M } \int _ { 0 } ^ { \infty } t \exp \left( - \left( \beta ^ { - 1 } - M ^ { - 1 } \right) t ^ { 2 } \right) d t } \\ { \displaystyle \qquad = 1 + \frac { \alpha } { M ( \beta ^ { - 1 } - M ^ { - 1 } ) } \int _ { 0 } ^ { \infty } e ^ { - t } d t } \\ { \displaystyle \qquad = 1 + \frac { \alpha } { M ( \beta ^ { - 1 } - M ^ { - 1 } ) } } \\ { \displaystyle \qquad = 1 + \frac { \alpha \beta } { M - \beta } . } \end{array}
$$

The proof is complete.

The following result shows that $\mathcal { P } ( B _ { R } ( \mathbb { R } ^ { d } ) )$ is contained in $\mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ for appropriate choices of α and $\beta .$

Lemma B.4. Fix $R > 0$ . For any $\beta > 0$ , it holds that

$$
\mathcal { P } \big ( B _ { R } ( \mathbb { R } ^ { d } ) \big ) \subset \mathrm { S G } _ { e ^ { R ^ { 2 } / \beta } , \beta } ( \mathbb { R } ^ { d } ) .
$$

Proof. Since

$$
\operatorname* { s u p } _ { \mu \in \mathcal { P } ( B _ { R } ) } \int \exp \left( \frac { \| x \| ^ { 2 } } { \beta } \right) \mu ( d x ) \leq e ^ { R ^ { 2 } / \beta } ,
$$

the result follows from Lemma B.3.

The following standard lemma controls the expected maximum of sub-Gaussians.

Lemma B.5. Fix $\alpha \geq 1$ and $\beta > 0$ . There exists a constant $C _ { \alpha , \beta } > 0$ such that

$$
\operatorname* { s u p } _ { \mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) } \mathbb { E } _ { ( x _ { 1 } , \ldots , x _ { N } ) \sim \mu ^ { \otimes N } } \left[ \operatorname* { m a x } _ { n \in [ N ] } \| x _ { n } \| ^ { 2 } \right] \leq C _ { \alpha , \beta } \big ( 1 + \log ( N ) \big ) .
$$

Proof. By Lemma B.3, if $\mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ , then

$$
\int \exp \left( \frac { \| x \| ^ { 2 } } { 2 \beta } \right) \mu ( d x ) \leq 1 + \alpha .
$$

For $\mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ , we then estimate

$$
\begin{array} { r } { \mathbb { E } \left[ \underset { i \in [ N ] } { \operatorname* { m a x } } \| x _ { i } \| ^ { 2 } \right] \leq 2 \beta \log \mathbb { E } \left[ \underset { i \in [ N ] } { \operatorname* { m a x } } e ^ { \| x _ { i } \| ^ { 2 } / ( 2 \beta ) } \right] } \\ { \leq 2 \beta \log \mathbb { E } \left[ \underset { i = 1 } { \overset { N } { \sum } } e ^ { \| x _ { i } \| ^ { 2 } / ( 2 \beta ) } \right] } \\ { \leq 2 \beta ( \log ( N ) + \log ( 1 + \alpha ) ) } \\ { \leq C _ { \alpha , \beta } ( 1 + \log ( N ) ) . } \end{array}
$$

The proof is complete.

The following technical lemma controls the diference in Gaussian moments between $\mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ and its normalized restriction to a ball of radius R.

Lemma B.6. Let $\mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ and $R > { \sqrt { \beta \log ( \alpha ) } }$ . Let $\mu _ { R }$ denote the conditional distribution of $\mu$ given $\| x \| \leq R$ . Then, for any $k \geq 2 ,$ , it holds that

$$
\left| \int e ^ { \| x \| ^ { 2 } / ( k \beta ) } \mu ( d x ) - \int e ^ { \| x \| ^ { 2 } / ( k \beta ) } \mu _ { R } ( d x ) \right| \leq \alpha \frac { \left( 1 + \frac { 2 } { k ^ { 2 } \beta ^ { 2 } } \right) e ^ { - R ^ { 2 } / ( k \beta ) } + \left( 1 + \frac { \alpha } { k - 1 } \right) e ^ { - R ^ { 2 } / \beta } } { 1 - \alpha e ^ { - R ^ { 2 } / \beta } } .
$$

Proof. We have

$$
\begin{array} { r l } & { \left| \int e ^ { \| x \| ^ { 2 } / ( k \beta ) } \mu ( d x ) - \int e ^ { \| x \| ^ { 2 } / ( k \beta ) } \mu _ { R } ( d x ) \right| = \left| \int e ^ { \| x \| ^ { 2 } / ( k \beta ) } \mu ( d x ) - \frac { \int _ { \| x \| \leq R } e ^ { \| x \| ^ { 2 } / ( k \beta ) } \mu ( d x ) } { \int _ { \| x \| \leq R } \mu ( d x ) } \right| } \\ & { \leq \left| \frac { \int _ { \| x \| > R } e ^ { \| x \| ^ { 2 } / ( k \beta ) } \mu ( d x ) } { \int _ { \| x \| \leq R } \mu ( d x ) } \right| + \left| 1 - \frac { 1 } { \int _ { \| x \| \leq R } \mu ( d x ) } \right| \cdot \int e ^ { \| x \| ^ { 2 } / ( k \beta ) } \mu ( d x ) } \\ & { = : I + I I . } \end{array}
$$

For term I, we upper bound the numerator and lower bound the denominator. For the numerator, we use the layer cake formula with derivative:

$$
\begin{array} { r l } { \displaystyle \int _ { \| x \| > R } e ^ { \| x \| ^ { 2 } / ( k \beta ) } \mu ( d x ) = e ^ { R ^ { 2 } / ( k \beta ) } \mu ( \| x \| > R ) + \displaystyle \frac { 2 } { k \beta } \int _ { R } ^ { \infty } t e ^ { t ^ { 2 } / ( k \beta ) } \mu ( \| x \| > t ) d t } & { } \\ { \displaystyle } & { \le \alpha e ^ { - \frac { ( k - 1 ) R ^ { 2 } } { k \beta } } + \frac { 2 \alpha } { k \beta } \int _ { R } ^ { \infty } t e ^ { - t ^ { 2 } / ( k \beta ) } d t } \\ { \displaystyle } & { = \alpha e ^ { - \frac { ( k - 1 ) R ^ { 2 } } { k \beta } } + \frac { \alpha } { k \beta } \int _ { R ^ { 2 } } ^ { \infty } e ^ { - t / ( k \beta ) } d t } \\ { \displaystyle } & { = \alpha e ^ { - \frac { ( k - 1 ) R ^ { 2 } } { k \beta } } + \frac { 2 \alpha } { k ^ { 2 } \beta ^ { 2 } } e ^ { - R ^ { 2 } / ( k \beta ) } } \\ { \displaystyle } & { = \alpha \bigg ( 1 + \frac { 2 } { k ^ { 2 } \beta ^ { 2 } } \bigg ) e ^ { - R ^ { 2 } / ( k \beta ) } , } \end{array}
$$

where the last line uses the fact that $k \geq 2$ , and hence $e ^ { - \frac { ( k - 1 ) R ^ { 2 } } { k \beta } } < e ^ { - \frac { R ^ { 2 } } { k \beta } }$ . For the denominator, we have the bound

$$
\int _ { \| x \| \leq R } \mu ( d x ) \geq 1 - \alpha e ^ { - R ^ { 2 } / \beta } ,
$$

simply due to the fact that $\mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ . This establishes the bound

$$
I \leq \frac { \alpha \Big ( 1 + \frac { 2 } { k ^ { 2 } \beta ^ { 2 } } \Big ) e ^ { - R ^ { 2 } / ( k \beta ) } } { 1 - \alpha e ^ { - R ^ { 2 } / \beta } } .
$$

For term $I I ,$ we use the lower bound on $\textstyle \int _ { \| x \| \leq R } \mu ( d x )$ to deduce that

$$
\left| 1 - \frac { 1 } { \int _ { \| x \| \le R } \mu ( d x ) } \right| \le \left| 1 - \frac { 1 } { 1 - \alpha e ^ { - R ^ { 2 } / \beta } } \right| = \frac { \alpha e ^ { - R ^ { 2 } / \beta } } { 1 - \alpha e ^ { - R ^ { 2 } / \beta } } .
$$

Next, since $\mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ , Lemma B.3 furnishes the exponential moment bound

$$
\int e ^ { \| x \| ^ { 2 } / ( k \beta ) } \mu ( d x ) \leq 1 + { \frac { \alpha } { k - 1 } } .
$$

Putting things together, we get

$$
I I \leq \frac { \alpha \Big ( 1 + \frac { \alpha } { k - 1 } \Big ) e ^ { - R ^ { 2 } / \beta } } { 1 - \alpha e ^ { - R ^ { 2 } / \beta } } ,
$$

and combining the bounds for terms I and II yields the estimate as stated in the lemma.

The following provides high-probability bounds on the sub-Gaussian parameters of an empirical measure $\mu _ { N }$ when $\mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ , which may be a result of independent interest.

Proposition B.7. Let $\alpha \geq 1$ and $\beta > 0$ , and let $\mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ . Fix $\rho > 1$ and $k > 2 \rho . \ I f x _ { 1 } , . . . , x _ { N }$ are iid samples from $\mu$ and $\begin{array} { r } { \mu _ { N } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \delta _ { x _ { i } } } \end{array}$ , then there exists $N _ { \alpha , \beta , \rho , k } \in$ N such that

$$
\begin{array} { r } { \mathbb { P } _ { x _ { 1 } , \ldots , x _ { N } } \Big ( \mu _ { N } \in \mathrm { S G } _ { ( 3 + \alpha / ( k - 1 ) ) , k \beta } ( \mathbb { R } ^ { d } ) \Big ) \geq 1 - 2 \alpha N ^ { 1 - \rho } \quad f o r \ a l l \quad N \geq N _ { \alpha , \beta , \rho , k } . } \end{array}
$$

Proof. For $R > 0$ , let $\mathcal { E } _ { N , R }$ denote the event that $\operatorname* { m a x } _ { i \in [ N ] } \| x _ { i } \| \leq R .$ . Since $\mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ , it follows from an application of the union bound that

$$
\mathbb { P } _ { x _ { 1 } , \ldots , x _ { N } } ( \mathcal { E } _ { N , R } ) \ge 1 - \alpha N \exp \biggl ( - \frac { R ^ { 2 } } { \beta } \biggr ) .
$$

In particular, choosing $R = R _ { N } = \sqrt { \beta \rho \log ( N ) }$ , we have

$$
\begin{array} { r } { \mathbb { P } _ { x _ { 1 } , \ldots , x _ { N } } ( { \mathcal E } _ { N , R } ) \ge 1 - \alpha N ^ { 1 - \rho } . } \end{array}
$$

To bound the desired probability, we use Lemma B.3, which states that if $\nu \in \mathcal { P } ( \mathbb { R } ^ { d } )$ and $\begin{array} { r } { \int e ^ { \frac { \| x \| ^ { 2 } } { k \beta } } \nu ( d x ) \leq 2 + \frac { \alpha } { k - 1 } } \end{array}$ then $\nu \in \mathrm { S G } _ { ( 2 + \alpha / ( k - 1 ) ) , k \beta } ( \mathbb R ^ { d } )$ . Applying this to the empirical measure $\mu _ { N }$ , it implies that

$$
\mathbb { P } _ { x _ { 1 } , \ldots , x _ { N } } \left( \mu _ { N } \in \mathrm { S G } _ { \left( 2 + \alpha / ( k - 1 ) \right) , k \beta } \right) \geq \mathbb { P } _ { x _ { 1 } , \ldots , x _ { N } } \left( \frac { 1 } { N } \sum _ { i = 1 } ^ { N } e ^ { \frac { \| x _ { i } \| ^ { 2 } } { k \beta } } \leq 2 + \frac { \alpha } { k - 1 } \right) .
$$

To bound the above probability, we use the fact that

$$
\begin{array} { r l r } {  { \mathbb { P } _ { x _ { 1 } , \ldots , x _ { N } } ( \frac { 1 } { N } \sum _ { i = 1 } ^ { N } e ^ { \frac { \| x _ { i } \| ^ { 2 } } { k \beta } } \leq 2 + \frac { \alpha } { k - 1 } ) \geq \mathbb { P } _ { x _ { 1 } , \ldots , x _ { N } } ( \{ \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } e ^ { \frac { \| z _ { i } \| ^ { 2 } } { k \beta } } \leq 2 + \frac { \alpha } { k - 1 } \} \bigcap \mathcal { E } _ { N , R } ) } } \\ & { } & { = \mathbb { P } _ { x _ { 1 } , \ldots , x _ { N } } ( \mathcal { E } _ { N , R } ) \mathbb { P } _ { x _ { 1 } , \ldots , x _ { N } } ( \frac { 1 } { N } \sum _ { i = 1 } ^ { N } e ^ { \frac { \| z _ { i } \| ^ { 2 } } { k \beta } } \leq 2 + \frac { \alpha } { k - 1 }  \mathcal { E } _ { N , R } ) } \\ & { } & { \geq \big ( 1 - \alpha N ^ { 1 - \rho } \big ) \mathbb { P } _ { x _ { 1 } , \ldots , x _ { N } } ( \frac { 1 } { N } \sum _ { i = 1 } ^ { N } e ^ { \frac { \| z _ { i } \| ^ { 2 } } { k \beta } } \leq 2 + \frac { \alpha } { k - 1 } \bigg \vert \mathcal { E } _ { N , R } ) . } \end{array}
$$

It therefore sufices to bound the conditional probability above. Let $\mu _ { R }$ denote the normalized restriction of $\mu$ to the ball of radius $R ,$ and note that the joint law of $x _ { 1 } , \ldots , x _ { N }$ conditioned on $\mathcal { E } _ { N , R }$ coincides with the joint law of $N$ iid samples from $\mu _ { R } .$ . Therefore, as a consequence of Hoefding’s inequality, we have

$$
\begin{array} { r l } { \mathbb { P } _ { x _ { 1 } , \ldots , x _ { N } } ( | \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } e ^ { \frac { \| x _ { i } \| ^ { 2 } } { k ^ { 3 } } } - \int e ^ { \frac { \| x _ { i } \| ^ { 2 } } { k ^ { 3 } } } \mu _ { R } ( d x ) | > t | \mathcal { E } _ { N , R } ) = \mathbb { P } _ { x _ { 1 } , \ldots , x _ { N } \sim \mu _ { R } } ( | \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } e ^ { \frac { \| x _ { i } \| ^ { 2 } } { k ^ { 3 } } } - \int e ^ { \frac { \| x _ { i } \| ^ { 2 } } { k ^ { 3 } } } \mu _ { R } ( d x ) | > t ) } & { } \\ { \leq 2 \exp ( - \frac { 2 N l ^ { 2 } } { \exp ( \frac { 2 R ^ { 2 } } { k ^ { 3 } } ) } ) } & { } \\ { = 2 \exp ( - 2 N ^ { 1 - 2 \rho / k } t ^ { 2 } ) , } \end{array}
$$

where we used the definition of $R = R _ { N }$ . Note that $\begin{array} { r } { 1 - \frac { 2 \rho } { k } > 0 } \end{array}$ by assumption. Since $\mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ , Lemma B.3 guarantees that

$$
\int e ^ { \frac { \| x \| ^ { 2 } } { k \beta } } \mu ( d x ) \leq 1 + \frac { \alpha } { k - 1 } .
$$

In addition, Lemma B.6 proves, provided $R > { \sqrt { \beta \log ( \alpha ) } }$ , that

$$
\int e ^ { \frac { \| x \| ^ { 2 } } { k \beta } } \mu _ { R } ( d x ) \leq \int e ^ { \frac { \| x \| ^ { 2 } } { k \beta } } \mu ( d x ) + \alpha \frac { \left( 1 + \frac { 2 } { k ^ { 2 } \beta ^ { 2 } } \right) e ^ { - R ^ { 2 } / ( k \beta ) } + \left( 1 + \frac { \alpha } { k - 1 } \right) e ^ { - R ^ { 2 } / \beta } } { 1 - \alpha e ^ { - R ^ { 2 } / \beta } } .
$$

Recalling the definition of $R = R _ { N }$ , the second term on the right-hand side of the preceding display converges to zero as $N \to \infty$ . Therefore, there exists a natural number $N _ { \alpha , \beta , \rho , k }$ such that for all $N \geq N _ { \alpha , \beta , \rho , k }$ , we have $R > { \sqrt { \beta \log ( \alpha ) } }$ and

$$
\int e ^ { \frac { \| x \| ^ { 2 } } { k \beta } } \mu _ { R } ( d x ) \leq 2 + \frac { \alpha } { k - 1 } \quad \mathrm { f o r ~ a l l } \quad N \geq N _ { \alpha , \beta } .
$$

Thus, setting t = 1 in Hoefding’s inequality, we have whenever $N \geq N _ { \alpha , \beta , \rho , k }$ that

$$
\begin{array} { r l r } {  { \mathbb { P } _ { x _ { 1 } , \ldots , x _ { N } } ( \frac { 1 } { N } \sum _ { i = 1 } ^ { N } e ^ { \frac { \| x _ { i } \| ^ { 2 } } { k ^ { 5 } } } > 3 + \frac { \alpha } { k - 1 } \bigg | \mathcal { E } _ { N , R } ) \le \mathbb { P } _ { x _ { 1 } , \ldots , x _ { N } } ( \frac { 1 } { N } \sum _ { i = 1 } ^ { N } e ^ { \frac { \| x _ { i } \| ^ { 2 } } { k ^ { 5 } } } > 1 + \int e ^ { \frac { \| x _ { i } \| ^ { 2 } } { k ^ { 5 } } } \mu _ { R } ( d x ) \bigg | \mathcal { E } _ { N , R } ) } } \\ & { } & { \le \mathbb { P } _ { x _ { 1 } , \ldots , x _ { N } \sim \mu _ { R } } ( | \frac { 1 } { N } \sum _ { i = 1 } ^ { N } e ^ { \frac { \| x _ { i } \| ^ { 2 } } { k \beta } } - \int e ^ { \frac { \| x _ { i } \| ^ { 2 } } { k \beta } } \mu _ { R } ( d x ) | > 1 ) } \\ & { } & { \le 2 \exp ( - 2 N ^ { 1 - 2 \rho / k } ) . } \end{array}
$$

Putting all the pieces together, we have

$$
\begin{array} { r } { \mathbb { P } _ { x _ { 1 } , \ldots , x _ { N } } \bigl ( \mu _ { N } \in \mathrm { S G } _ { ( 3 + \frac { \alpha } { k - 1 } ) , k \beta } \bigr ) \geq \bigl ( 1 - \alpha N ^ { 1 - \rho } \bigr ) \biggl ( 1 - 2 \exp \biggl ( - 2 N ^ { 1 - 2 \rho / k } \biggr ) \biggr ) . } \end{array}
$$

To conclude, we choose $N _ { \alpha , \beta , \rho , k }$ large enough that

$$
\big ( 1 - \alpha N ^ { 1 - \rho } \big ) \Big ( 1 - 2 \exp \Big ( - 2 N ^ { 1 - 2 \rho / k } \Big ) \Big ) \geq 1 - 2 \alpha N ^ { 1 - \rho } \quad \mathrm { f o r ~ a l l } \quad N \geq N _ { \alpha , \beta , \rho , k } .
$$

The proof is complete.

## B.2 Gaussian and Sub-Gaussian Tail Bounds

We herein record relevant several tail estimates for Gaussian and sub-Gaussian distributions. These estimates are used several times in establishing the H¨older continuity of transformers. We begin with some elementary tail bounds for the one-dimensional Gaussian.

Lemma B.8. Let $R > 0$ . Then

$$
\begin{array} { l } { { \displaystyle { \int _ { R } ^ { \infty } e ^ { - x ^ { 2 } } d x \le \frac { 1 } { 2 R } e ^ { - R ^ { 2 } } } , } } \\ { { \displaystyle { \int _ { R } ^ { \infty } x e ^ { - x ^ { 2 } } d x = \frac { 1 } { 2 } e ^ { - R ^ { 2 } } } , \quad a n d } } \\ { { \displaystyle { \int _ { R } ^ { \infty } x ^ { 2 } e ^ { - x ^ { 2 } } d x \le \left( \frac { R } { 2 } + \frac { 1 } { 4 R } \right) e ^ { - R ^ { 2 } } } . } } \end{array}
$$

Proof. The second line follows from direct integration. The first line follows from the second line:

$$
\int _ { R } ^ { \infty } e ^ { - x ^ { 2 } } d x \leq \int _ { R } ^ { \infty } { \frac { x } { R } } e ^ { - x ^ { 2 } } d x = { \frac { 1 } { 2 R } } e ^ { - R ^ { 2 } } .
$$

The third line follows from the first line and integration by parts:

$$
\begin{array} { r l r } {  { \int _ { R } ^ { \infty } x ^ { 2 } e ^ { - x ^ { 2 } } d x = \frac { 1 } { 2 } R e ^ { - R ^ { 2 } } + \frac { 1 } { 2 } \int _ { R } ^ { \infty } e ^ { - x ^ { 2 } } d x } } \\ & { } & { \leq \bigg ( \frac { R } { 2 } + \frac { 1 } { 4 R } \bigg ) e ^ { - R ^ { 2 } } . } \end{array}
$$

The proof is complete.

Lemma B.9. Let $a , b > 0$ and $\textstyle R > { \frac { a } { 2 b } }$ . Then

$$
\begin{array} { r l } & { \int _ { R } ^ { \infty } e ^ { a x - b x ^ { 2 } } d x \le \frac { 1 } { 2 b R - a } e ^ { - b R ^ { 2 } + a R } , } \\ & { \int _ { R } ^ { \infty } x e ^ { a x - b x ^ { 2 } } d x \le \left( \frac { a } { 2 b ^ { 2 } R - a b } + b ^ { - 1 } \right) e ^ { - b R ^ { 2 } + a R } , \quad a n d } \\ & { \int _ { R } ^ { \infty } x ^ { 2 } e ^ { a x - b x ^ { 2 } } d x \le \left( \frac { a ^ { 2 } } { 4 b ^ { 3 / 2 } \left( b ^ { 1 / 2 } R - a / \left( 2 b ^ { 1 / 2 } \right) \right) } + \frac { a } { 2 b ^ { 2 } } + \frac { b ^ { 1 / 2 } R - a / \left( 2 b ^ { 1 / 2 } \right) } { 2 b ^ { 3 / 2 } } + \frac { 1 } { 4 b ^ { 3 / 2 } \left( b ^ { 1 / 2 } R - a / \left( 2 b ^ { 1 / 2 } \right) \right) } \right) e ^ { - b R ^ { 2 } + a R } . } \end{array}
$$

Proof. For the first line, we have

$$
\begin{array} { r l } {  { \int _ { R } ^ { \infty } e ^ { a x - b x ^ { 2 } } d x = e ^ { \frac { a ^ { 2 } } { 4 b } } \int _ { R } ^ { \infty } e ^ { - ( b ^ { 1 / 2 } x - \frac { a } { 2 b ^ { 1 / 2 } } ) ^ { 2 } } d x } } \\ & { = b ^ { - 1 / 2 } e ^ { \frac { a ^ { 2 } } { 4 b } } \int _ { b ^ { 1 / 2 } R - \frac { a } { 2 b ^ { 1 / 2 } } } ^ { \infty } e ^ { - x ^ { 2 } } d x } \\ & { \le \frac { b ^ { - 1 / 2 } } { 2 ( b ^ { 1 / 2 } R - a / ( 2 b ^ { 1 / 2 } ) ) } e ^ { \frac { a ^ { 2 } } { 4 b } - ( b ^ { 1 / 2 } R - a / ( 2 b ^ { 1 / 2 } ) ) ^ { 2 } } } \\ & { = \frac { 1 } { 2 b R - a } e ^ { - b R ^ { 2 } + a R } , } \end{array}
$$

where we used Lemma B.8. For the second line, using a similar argument, we have

$$
\begin{array} { r l } {  { \int _ { R } ^ { \infty } x e ^ { a x - b x ^ { 2 } } d x = b ^ { - 1 } e ^ { \frac { a ^ { 2 } } { 4 b } } \int _ { b ^ { 1 / 2 } R - \frac { a } { 2 b ^ { 1 / 2 } } } ^ { \infty } \biggl ( x + \frac { a } { 2 \sqrt { b } } \biggr ) e ^ { - x ^ { 2 } } d x } } \\ & { \leq \biggl ( \frac { a } { 2 b ^ { 2 } R - a b } + b ^ { - 1 } \biggr ) e ^ { - b R ^ { 2 } + a R } , } \end{array}
$$

where we again used Lemma B.8. Finally, for the third line, we use the same change of variables, along with the inequalities in the first to second lines of the statement of the result:

$$
\begin{array} { r l } & { \displaystyle \int _ { R } ^ { \infty } x ^ { 2 } e ^ { a x - b x ^ { 2 } } d x = b ^ { - 3 / 2 } e ^ { \frac { a ^ { 2 } } { 4 b } } \int _ { b ^ { 1 / 2 } R - \frac { a } { 2 b ^ { 1 / 2 } } } ^ { \infty } \left( \frac { a ^ { 2 } } { 4 b } + \frac { a } { b ^ { 1 / 2 } } x + x ^ { 2 } \right) e ^ { - x ^ { 2 } } } \\ & { \quad \quad \quad \quad \leq \left( \frac { a ^ { 2 } } { 4 b ^ { 5 / 2 } \left( b ^ { 1 / 2 } R - a / ( 2 b ^ { 1 / 2 } ) \right) } + \frac { a } { 2 b ^ { 2 } } + \frac { b ^ { 1 / 2 } R - a / \left( 2 b ^ { 1 / 2 } \right) } { 2 b ^ { 3 / 2 } } + \frac { 1 } { 4 b ^ { 3 / 2 } \left( b ^ { 1 / 2 } R - a / \left( 2 b ^ { 1 / 2 } \right) \right) } \right) e ^ { - b R ^ { 2 } + a R } , } \end{array}
$$

where we used Lemma B.8 yet again.

We apply these one-dimensional Gaussian tail bounds to prove tail bounds for certain moments of sub-Gaussian measures. Our first result shows that truncation is uniformly accurate.

Lemma B.10. Let $\Pi _ { R } \colon \mathbb { R } ^ { d } \to B _ { R } \subset \mathbb { R } ^ { d }$ be the truncation map. Then

$$
\operatorname* { s u p } _ { \mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) } \mathsf { W } _ { 1 } \big ( \mu , ( \Pi _ { R } ) _ { \# } \mu \big ) \le \frac { \alpha \beta } { 2 R } e ^ { - R ^ { 2 } / \beta } .
$$

Proof. For any $\mu \in \operatorname { S G } _ { \alpha , \beta }$ , the quantity $\mathsf { W } _ { 1 } ( \mu , ( \Pi _ { R } ) _ { \# } \mu )$ is bounded above by

$$
\int _ { \mathbb { R } ^ { d } } \lVert x - \Pi _ { R } ( x ) \rVert \mu ( d x ) = \int _ { \lVert x \rVert > R } ( \lVert x \rVert - R ) \mu ( d x ) \le \int _ { R } ^ { \infty } \mu ( \lVert x \rVert > t ) d t \le \alpha \int _ { R } ^ { \infty } e ^ { - t ^ { 2 } / \beta } d t .
$$

Application of Lemma B.9 with $r = R , a = 0$ , and $b = 1 / \beta$ completes the proof.

We need a similar bound for quantitative sub-Gaussian uniform integrability of first moments.

Lemma B.11. For any $R > 0$ , it holds that

$$
\operatorname* { s u p } _ { \mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) } \int _ { \| x \| > R } \left( 1 + \| x \| \right) \mu ( d x ) \leq \alpha \biggl ( \frac { \beta } { 2 R } + 1 + R \biggr ) e ^ { - R ^ { 2 } / \beta } .
$$

Proof. By the proof of Lemma B.10, we have

$$
\begin{array} { r l } {  { \int _ { \| x \| > R } ( 1 + \| x \| ) \mu ( d x ) = \int _ { \| x \| > R } ( 1 + R ) \mu ( d x ) + \int _ { \| x \| > R } ( \| x \| - R ) \mu ( d x ) } } \\ & { \leq \alpha ( 1 + R ) e ^ { - R ^ { 2 } / \beta } + \frac { \alpha \beta } { 2 R } e ^ { - R ^ { 2 } / \beta } , } \end{array}
$$

which is the assertion.

Finally, we prove tail bounds for modulated exponential moments of sub-Gaussian measures.

Lemma B.12. For all $\alpha \ge 1 , \beta > 0 , \mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) , \lambda > 0$ , and $R > 0$ , it holds that

$$
\int _ { \| y \| > R } \lambda e ^ { \lambda \| y \| } \left( \| y \| - R \right) \mu ( d y ) \leq \alpha \lambda e ^ { \beta \lambda ^ { 2 } / 2 } \bigg ( \frac \beta R + \lambda \frac { \beta ^ { 2 } } { R ^ { 2 } } \bigg ) e ^ { - \frac { R ^ { 2 } } { 2 \beta } } \quad a n d \quad
$$

$$
\int _ { \| y \| > R } \lambda e ^ { \lambda \| y \| } \| y \| \big ( \| y \| - R \big ) \mu ( d y ) \leq \alpha \lambda e ^ { \beta \lambda ^ { 2 } / 2 } \bigg ( \lambda \frac { \beta ^ { 2 } } { R } + 2 \beta \bigg ) e ^ { - \frac { R ^ { 2 } } { 2 \beta } } .
$$

Proof. Fix $\mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ throughout the proof. Write $I _ { 1 }$ and $I _ { 2 }$ for the first and second integrals, respectively. The layer cake formula states that if $f : \mathbb { R } \to \mathbb { R }$ is nonnegative and diferentiable, then

$$
\int _ { \| y \| > R } f ( \| y \| ) \mu ( d y ) = f ( R ) \mu ( \| y \| > R ) + \int _ { R } ^ { \infty } f ^ { \prime } ( t ) \mu ( \| y \| > t ) d t .
$$

Applying this to $f ( t ) : = \lambda e ^ { \lambda t } ( t - R )$ , we get $f ^ { \prime } ( t ) = \lambda ( 1 + \lambda ( t - R ) ) e ^ { \lambda t } , f ( R ) = 0$ , and

$$
{ I _ { 1 } } \le \alpha \int _ { R } ^ { \infty } f ^ { \prime } ( t ) e ^ { - { t ^ { 2 } } / { \beta } } d t = \alpha \lambda \int _ { R } ^ { \infty } \bigl ( 1 + \lambda ( t - R ) \bigr ) e ^ { \lambda t - { t ^ { 2 } } / { \beta } } d t .
$$

Young’s inequality shows that

$$
\lambda t = ( \sqrt { \beta } \lambda ) ( t / \sqrt { \beta } ) \leq \frac { \beta \lambda ^ { 2 } } { 2 } + \frac { t ^ { 2 } } { 2 \beta } .
$$

Subtracting $t ^ { 2 } / \beta$ and exponentiating the preceding display yields

$$
I _ { 1 } \leq \alpha \lambda e ^ { \frac { \beta \lambda ^ { 2 } } { 2 } } \int _ { R } ^ { \infty } \bigl ( 1 + \lambda ( t - R ) \bigr ) e ^ { - \frac { t ^ { 2 } } { 2 \beta } } d t .
$$

Now since $t ^ { 2 } = ( R + ( t - R ) ) ^ { 2 } \geq R ^ { 2 } + 2 R ( t - R )$ , we have

$$
\int _ { R } ^ { \infty } ( t - R ) e ^ { - { \frac { t ^ { 2 } } { 2 \beta } } } d t \leq e ^ { \frac { - R ^ { 2 } } { 2 \beta } } \int _ { R } ^ { \infty } ( t - R ) e ^ { - R ( t - R ) / \beta } d t = { \frac { \beta ^ { 2 } } { R ^ { 2 } } } e ^ { \frac { - R ^ { 2 } } { 2 \beta } } \int _ { 0 } ^ { \infty } u e ^ { - u } d u = { \frac { \beta ^ { 2 } } { R ^ { 2 } } } e ^ { \frac { - R ^ { 2 } } { 2 \beta } } .
$$

The first equality is due to the variable substitution $u = R ( t - R ) / \beta$ and the second is due to integration by parts. The preceding display and Lemma B.9 imply the asserted upper bound on $I _ { 1 }$

For $I _ { 2 } ,$ let $f ( t ) : = \lambda e ^ { \lambda t } t ( t - R )$ so that $f ^ { \prime } ( t ) = \lambda ( \lambda t ( t - R ) + ( 2 t - R ) ) e ^ { \lambda t }$ and $f ( R ) = 0$ . The layer cake formula and Young’s inequality yield

$$
I _ { 2 } \leq \alpha \lambda e ^ { \frac { \beta \lambda ^ { 2 } } { 2 } } \int _ { R } ^ { \infty } \bigl ( \lambda t ( t - R ) + ( 2 t - R ) \bigr ) e ^ { - \frac { t ^ { 2 } } { 2 \beta } } d t .
$$

Integration by parts shows that

$$
\int _ { R } ^ { \infty } t ( t - R ) e ^ { - { \frac { t ^ { 2 } } { 2 \beta } } } d t = \beta \int _ { R } ^ { \infty } e ^ { - { \frac { t ^ { 2 } } { 2 \beta } } } d t .
$$

This, bounding $( 2 t - R ) \leq 2 t$ , and Lemma B.9 imply the asserted upper bound on $I _ { 2 } .$

## B.3 Properties of Transformers

We record some prior results on the regularity of self-attention mappings. The result below controls the Lipschitz constants of in-context mappings defined by self-attention layers. These results are proven by Furuya et al. (2026a) when $A = V$ , but we record the full proofs for posterity and to make the constants explicit.

Lemma B.13. Let $G ( \mu , x )$ be defined by

$$
G ( \mu , x ) = W \frac { \int e ^ { \langle x , A y \rangle } V y \mu ( d y ) } { \int e ^ { \langle x , A y \rangle } \mu ( d y ) } ,
$$

where $A \in \mathbb { R } ^ { d \times d } , V \in \mathbb { R } ^ { k \times d }$ , and $W \in \mathbb { R } ^ { d \times k }$ . Then for any $R > 0$ and $R ^ { \prime } > 0$ , it holds that

$$
\operatorname* { s u p } _ { \mu \in \mathcal { P } ( B _ { R } ) , x \in \mathbb { R } ^ { d } } \| G ( \mu , x ) \| \leq \| W \| \| V \| R ,
$$

$$
\operatorname* { s u p } _ { \mu \in \mathcal { P } ( B _ { R } ) } \mathrm { L i p } \big ( G ( \mu , \cdot ) \big ) \leq \| A \| \| V \| \| W \| R ^ { 2 } , \quad a n d
$$

$$
\operatorname* { s u p } _ { x \in B _ { R ^ { \prime } } } \mathrm { L i p } \Big ( G ( \cdot , x ) \big | _ { \mathcal { P } ( B _ { R } ) } \Big ) \leq e ^ { 4 \| A \| R R ^ { \prime } } \operatorname* { m a x } \big ( \| V \| \| W \| ( 1 + \| A \| R R ^ { \prime } ) , \| A \| \| V \| \| W \| R R ^ { \prime } \big ) .
$$

Proof. For the first estimate, fix $\mu \in \mathcal P ( B _ { R } )$ and $x \in \mathbb { R } ^ { d }$ . We can express $G ( \mu , x ) = \mathbb { E } _ { y \sim \mu _ { x } } [ V y ]$ , where $\mu _ { x }$ is defined by

$$
\frac { d \mu _ { x } } { d \mu } ( y ) \propto e ^ { \langle x , A y \rangle } .
$$

In particular, $\mu _ { x }$ is supported in $B _ { R }$ (since µ is supported in $B _ { R } )$ , which yields

$$
\begin{array} { r } { \| G ( \mu , x ) \| = \| \mathbb { E } _ { y \sim \mu _ { x } } [ W V y ] \| \leq \| W \| \| V \| \mathbb { E } _ { y \sim \mu _ { x } } [ \| y \| ] \leq \| W \| \| V \| R . } \end{array}
$$

For the second estimate, viewing $G ( \mu , x )$ again as a conditional expectation, a direct computation yields

$$
\nabla _ { \boldsymbol { x } } G ( \mu , \boldsymbol { x } ) = W V \operatorname { C o v } ( \mu _ { x } ) \boldsymbol { A } ^ { \intercal } .
$$

Since $\mu _ { x }$ is supported in $B _ { R }$ , we have $\| \Sigma _ { x } \| \leq R ^ { 2 }$ . Therefore,

$$
\operatorname* { s u p } _ { x \in \mathbb { R } ^ { d } } \| \nabla _ { x } G ( \mu , x ) \| \leq \| W \| \| V \| \| \operatorname { C o v } ( \mu _ { x } ) \| \| A ^ { \top } \| \leq \| A \| \| W \| \| V \| R ^ { 2 }
$$

holds uniformly in $\mu \in \mathcal P ( B _ { R } )$

For the third estimate, it holds that

$$
\begin{array} { r l } & { \| G ( \mu , x ) - G ( \nu , x ) \| \leq \| W \| \frac { \left\| \int e ^ { \langle x , A y \rangle } V y d ( \mu - \nu ) \right\| } { \left| \int e ^ { \langle x , A y \rangle } d \mu \right| } + \| W \| \left\| \int e ^ { \langle x , A y \rangle } V y d \nu \right\| \left| \frac { 1 } { \left| \int e ^ { \langle x , A y \rangle } d \mu \right| } - \frac { 1 } { \left| \int e ^ { \langle x , A y \rangle } d \nu \right| } \right| } \\ & { \leq \| W \| e ^ { \| A \| R R ^ { \prime } } \left\| \int e ^ { \langle x , A y \rangle } V y d ( \mu - \nu ) \right\| + \| W \| \| V \| R e ^ { \| A \| R R ^ { \prime } } \left| \frac { 1 } { \left| \int e ^ { \langle x , A y \rangle } d \mu \right| } - \frac { 1 } { \left| \int e ^ { \langle x , A y \rangle } d \nu \right| } \right| , } \end{array}
$$

where we used the facts that $x \in B _ { R ^ { \prime } }$ and both µ and ν belong to $\mathcal { P } ( B _ { R } )$ . For the first term in the last line of the preceding display, a direct computation of the Jacobian of the mapping $y \mapsto e ^ { \langle x , A y \rangle } V y$ shows that

$$
\mathrm { L i p } \Big ( \big ( y \mapsto e ^ { \langle x , A y \rangle } V y \big ) \big | _ { B _ { R } } \Big ) \leq e ^ { \| A \| R R ^ { \prime } } \| V \| \big ( 1 + \| A \| R R ^ { \prime } \big ) ,
$$

and therefore, by the dual formulation of the $\mathsf { W } _ { 1 }$ metric,

$$
\left\| \int e ^ { \langle x , A y \rangle } V y d ( \mu - \nu ) \right\| \le e ^ { \| A \| R R ^ { \prime } } \| V \| ( 1 + \| A \| R R ^ { \prime } ) \mathsf { W } _ { 1 } ( \mu , \nu ) .
$$

For the second term, using the fact that the Jacobian of $y \mapsto e ^ { \langle x , A y \rangle }$ is bounded by $\| A \| R ^ { \prime } e ^ { \| A \| R R ^ { \prime } }$ for $x \in B _ { R ^ { \prime } }$ we have the bounds

$$
\begin{array} { r l r } & { } & { \left| \displaystyle \frac { 1 } { \left| \int e ^ { \langle x , A y \rangle } d \mu \right| } - \frac { 1 } { \left| \int e ^ { \langle x , A y \rangle } d \nu \right| } \right| \le e ^ { 2 \| A \| R R ^ { \prime } } \left| \int e ^ { \langle x , A y \rangle } d ( \mu - \nu ) \right| } \\ & { } & { \le \| A \| R ^ { \prime } e ^ { 3 \| A \| R R ^ { \prime } } \mathbb { W } _ { 1 } ( \mu , \nu ) . } \end{array}
$$

Putting the pieces together, we obtain

$$
\begin{array} { r l } & { \| G ( \mu , x ) - G ( \nu , x ) \| \leq \Big ( e ^ { 2 \| A \| R R ^ { \prime } } \| W \| \| V \| ( 1 + \| A \| R R ^ { \prime } ) + \| A \| \| W \| \| V \| R R ^ { \prime } e ^ { 4 \| A \| R R ^ { \prime } } \Big ) \mathsf { W } _ { 1 } ( \mu , \nu ) } \\ & { \qquad \leq e ^ { 4 \| A \| R R ^ { \prime } } \operatorname* { m a x } \big ( \| W \| \| V \| ( 1 + \| A \| R R ^ { \prime } ) , \| A \| \| W \| \| V \| R R ^ { \prime } \big ) \mathsf { W } _ { 1 } ( \mu , \nu ) , } \end{array}
$$

as was to be shown.

We now present a straightforward generalization of the previous result to multi-head self-attention layers with residual connections.

Corollary B.14. Define

$$
( \mu , x ) \mapsto \Gamma ( \mu , x ) : = x + \sum _ { h = 1 } ^ { H } W _ { h } \frac { \int e ^ { \langle x , A _ { h } y \rangle } V _ { h } y \mu ( d y ) } { \int e ^ { \langle x , A _ { h } y \rangle } \mu ( d y ) }\tag{B.2}
$$

where $A _ { 1 } , \dots , A _ { H } \in \mathbb { R } ^ { d \times d } , \ V _ { 1 , \dots , V _ { H } } \in \mathbb { R } ^ { k \times d }$ , and $W _ { 1 } , \ldots , W _ { h } \in \mathbb { R } ^ { d \times k }$ . Define $c _ { A } : = \operatorname* { m a x } _ { h \in [ H ] } \| A _ { h } \| , \ c _ { V } =$ $\operatorname* { m a x } _ { h \in [ H ] } \| V _ { h } \|$ , and $c _ { W } = \operatorname* { m a x } _ { h \in [ H ] } \| W _ { h } \|$ . Then for any $R > 0$ and $R ^ { \prime } > 0$ , it holds that

$$
\operatorname* { s u p } _ { \mu \in \mathcal { P } ( B _ { R } ) } \mathrm { L i p } \big ( \Gamma ( \mu , \cdot ) \big ) \leq 1 + H c _ { A } c _ { V } c _ { W } R ^ { 2 } \quad a n d
$$

$$
\operatorname* { s u p } _ { x \in B _ { R ^ { \prime } } } \mathrm { L i p } \Big ( \Gamma \big ( \cdot , x \big ) \big | _ { \mathcal { P } ( B _ { R } ) } \Big ) \leq H e ^ { 4 c _ { A } R R ^ { \prime } } \operatorname* { m a x } \big ( c _ { V } c _ { W } \big ( 1 + c _ { A } R R ^ { \prime } \big ) , c _ { A } c _ { V } c _ { W } R R ^ { \prime } \big ) .
$$

Via simple coupling arguments, these estimates on the Lipschitz constants of the in-context mappings defined by multi-head self-attention layers translate directly to estimates on the Lipschitz constant of the corresponding multi-head self-attention layers themselves when viewed as measure-to-measure operators.

Lemma B.15. Instate the hypotheses of Corollary $B . 1 \llangle$ . Let Γ be as in (B.2). There exists a constant $C = C ( c _ { A } , c _ { V } , c _ { W } , H ) > 0$ such that for any $R > 0$ , it holds that

$$
\operatorname* { s u p } _ { \substack { \mu , \nu \in \mathcal { P } ( B _ { R } ) : \mathbb { W } _ { 1 } ( \mu , \nu ) \leq \epsilon } } \mathbb { W } _ { 1 } \big ( \Gamma ( \mu , \cdot ) _ { \# } \mu , \Gamma ( \nu , \cdot ) _ { \# } \nu \big ) \leq C e ^ { 5 c _ { A } R ^ { 2 } } \epsilon .
$$

Proof. By the triangle inequality and standard Wasserstein stability estimates, it holds that

$$
\begin{array} { r l } & { \mathsf { W } _ { 1 } ( \Gamma ( \mu , \cdot ) _ { \# } \mu , \Gamma ( \nu , \cdot ) _ { \# } \nu ) \le \mathsf { W } _ { 1 } ( \Gamma ( \mu , \cdot ) _ { \# } \mu , \Gamma ( \nu , \cdot ) _ { \# } \mu ) + \mathsf { W } _ { 1 } ( \Gamma ( \nu , \cdot ) _ { \# } \mu , \Gamma ( \nu , \cdot ) _ { \# } \nu ) } \\ & { \qquad \le \bigg ( \underset { x \in B _ { R } } { \operatorname* { s u p } } \mathrm { L i p } \Big ( \Gamma ( \cdot , x ) \big | _ { \mathcal { P } ( B _ { R } ) } \Big ) + \mathrm { L i p } \big ( \Gamma ( \nu , \cdot ) \big ) \bigg ) \mathsf { W } _ { 1 } ( \mu , \nu ) . } \end{array}
$$

Applying the Lipschitz constant estimates from Corollary B.14, the preceding display shows that

$$
\begin{array} { r l } & { \mathrm { L i p } \big ( \mu \mapsto \Gamma ( \mu , \cdot ) _ { \# } \mu \big ) \leq \biggl ( \underset { x \in B _ { R } } { \operatorname* { s u p } } \mathrm { L i p } \Bigl ( \Gamma ( \cdot , x ) \big | _ { \mathcal { P } ( B _ { R } ) } \Bigr ) + \mathrm { L i p } \bigl ( \Gamma ( \nu , \cdot ) \bigr ) \biggr ) } \\ & { \qquad \leq \Bigl ( 1 + H c _ { A } c _ { V } c _ { W } R ^ { 2 } + H e ^ { 4 c _ { A } R ^ { 2 } } \operatorname* { m a x } \bigl ( c _ { V } c _ { W } ( 1 + c _ { A } R ^ { 2 } ) , c _ { A } c _ { V } c _ { W } R \bigr ) \Bigr ) . } \end{array}
$$

To conclude the proof, we choose $C = C ( c _ { A } , c _ { V } , H ) > 0$ suficiently large so that

$$
\Bigl ( 1 + H c _ { A } c _ { V } c _ { W } R ^ { 2 } + H e ^ { 4 c _ { A } R ^ { 2 } } \operatorname * { m a x } ( c _ { V } c _ { W } ( 1 + c _ { A } R ^ { 2 } ) , c _ { A } c _ { V } c _ { W } R ) \Bigr ) \le C e ^ { 5 c _ { A } R ^ { 2 } }
$$

for all $R > 0$

For technical reasons, we will need estimates on the quantity

$$
\operatorname* { m a x } _ { i \in [ N ] } \| \mathsf T ( \mu _ { N } , x _ { i } ) \| ,
$$

where $\begin{array} { r } { \mu _ { N } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \delta _ { x _ { i } } } \end{array}$ is a discrete measure and $\top$ is a mean-field transformer. Such estimates are furnished in the following lemma.

Lemma B.16. Let $\mathsf { T } \colon \mathcal { P } ( \mathbb { R } ^ { d } ) \times \mathbb { R } ^ { d } \to \mathbb { R } ^ { k }$ be a mean-field transformer. Then there exists a constant $C _ { \mathsf { T } }$ such that $f o r$ every $N \in \mathbb N$ and every measure $\nu _ { N }$ supported on a finite set of points $\{ x _ { 1 } , \ldots , x _ { N } \}$ , it holds that

$$
\operatorname* { m a x } _ { i \in [ N ] } \| \mathsf { T } ( \nu _ { N } , x _ { i } ) \| \leq C \mathsf { \mathsf { T } } \bigg ( \operatorname* { m a x } _ { n \in [ N ] } \| x _ { n } \| \bigg ) .
$$

Proof. We will prove the result by induction on the number L of layers of T, where we view a single layer of the transformer as an application of a mean-field multi-head self-attention operator followed by a context-free MLP layer with ReLU activation

$$
( \mu , x ) \mapsto \phi { \big ( } \Gamma ( \mu , x ) { \big ) } ,\tag{B.3}
$$

and where we recall that the mean-field multi-headed self-attention operator with parameter matrices $A _ { 1 } , \dotsc , A _ { H } \in$ $\mathbb { R } ^ { d \times d } , V _ { 1 } , \ldots , V _ { H } \in \mathbb { R } ^ { k \times d }$ , and $W _ { 1 } , \dots , W _ { H } \in \mathbb { R } ^ { d \times k }$ is defined by

$$
\Gamma ( \mu , x ) = x + \sum _ { h = 1 } ^ { H } W _ { h } \frac { \int e ^ { \langle x , A _ { h } y \rangle } V _ { h } y \mu ( d y ) } { \int e ^ { \langle x , A _ { h } y \rangle } \mu ( d y ) } .
$$

For the base case, we assume that $\mathsf { T } ( \mu , x )$ is given by (B.3) for some choice of $A _ { 1 } , \dotsc , A _ { H } , V _ { 1 } , \dotsc , V _ { H }$ , and $\phi .$ First, since $\phi$ is a ReLU MLP layer, it is globally Lipschitz. Hence

$$
\begin{array} { l } { \underset { i \in [ N ] } { \operatorname* { s u p } } \left. \phi ( \Gamma ( \nu _ { N } , x _ { i } ) ) \right. \leq ( \left. \phi ( 0 ) \right. + \mathrm { L i p } ( \phi ) ) \underset { i \in [ N ] } { \operatorname* { s u p } } \left. \Gamma ( \nu _ { N } , x _ { i } ) \right. } \\ { \leq ( \left. \phi ( 0 ) \right. + \mathrm { L i p } ( \phi ) ) \left( \underset { i \in [ N ] } { \operatorname* { m a x } } \left. x _ { i } \right. + \underset { h \in [ H ] } { \operatorname* { m a x } } \left. W _ { h } \right. \cdot \underset { i \in [ N ] } { \operatorname* { m a x } } \left. \frac { H } { \int e ^ { \langle x _ { i } , A _ { h } y \rangle } V _ { h } y \mu _ { N } ( d y ) } \right. \right) . } \end{array}
$$

Next, we note that for each $h \in [ H ]$ , the quantity

$$
\frac { \int e ^ { \langle x _ { i } , A _ { h } y \rangle } V _ { h } y \mu _ { N } ( d y ) } { \int e ^ { \langle x _ { i } , A _ { h } y \rangle } \mu _ { N } ( d y ) }
$$

belongs to the convex hull of $V \boldsymbol { x } _ { 1 } , \dots , V \boldsymbol { x } _ { N }$ . Therefore,

$$
\operatorname* { m a x } _ { i \in [ N ] } \left\| \frac { \int e ^ { \langle x _ { i } , A _ { h } y \rangle } V _ { h } y \mu _ { N } ( d y ) } { \int e ^ { \langle x _ { i } , A _ { h } y \rangle } \mu _ { N } ( d y ) } \right\| \leq \| V _ { h } \| \operatorname* { m a x } _ { i \in [ N ] } \| x _ { i } \| .
$$

This proves the base case. Next, suppose that the claim has been proven for all transformers defined by L compositions of (B.3). Let $\mathsf { T } \colon \mathcal { P } ( \mathbb { R } ^ { d } ) \times \mathbb { R } ^ { d } \to \mathbb { R } ^ { k }$ be a composition of $( L + 1 )$ applications of (B.3). Write

$$
\mathsf { T } ( \mu , x ) = ( \phi _ { L + 1 } \circ \Gamma _ { L + 1 } \circ G _ { L } ) ( \mu , x ) = \phi _ { L + 1 } ( \Gamma _ { L + 1 } ( \mathsf { T } _ { L } ( \mu , \cdot ) _ { \# } \mu , \mathsf { T } _ { L } ( \mu , x ) ) ) ,
$$

where $\mathsf { T } _ { L } \colon \mathcal { P } ( \mathbb { R } ^ { d } ) \times \mathbb { R } ^ { d } \to \mathbb { R } ^ { d _ { L } }$ is an L-layer transformer, and $\phi _ { L + 1 } \diamond \Gamma _ { L + 1 } \colon \mathcal { P } ( \mathbb { R } ^ { d _ { L } } ) \times \mathbb { R } ^ { d _ { L } } \to \mathbb { R } ^ { k }$ is a single transformer layer. The inductive hypothesis furnishes a constant $\widetilde { C } _ { \mathrm { T } } > 0$ such that

$$
\operatorname* { s u p } _ { x \in \mathbb { R } ^ { d } } \| \mathsf { T } _ { L } ( \nu _ { N } , x ) \| \leq \widetilde { C } _ { \mathsf { T } } \operatorname* { m a x } _ { i \in [ N ] } \| x _ { i } \| .
$$

In addition, we can write $\begin{array} { r } { \mathsf { T } _ { L } ( \nu _ { N } ) _ { \# } \nu _ { N } = \sum _ { i = 1 } ^ { N } c _ { i } \delta _ { z _ { i } } } \end{array}$ , where $z _ { i } = \mathsf T _ { L } ( \nu _ { N } , x _ { i } )$ . Thus, a parallel argument to the one used to prove the base case yields

$$
\begin{array} { r l } { \displaystyle \underset { i \in [ N ] } { \operatorname* { m a x } } \| \Gamma ( \nu _ { N } , x ) \| = \displaystyle \operatorname* { s u p } _ { x \in \mathbb { R } ^ { d } } \| ( \phi _ { L + 1 } \circ \Gamma _ { L + 1 } ) ( \mathsf { T } _ { L } ( \nu _ { N } ) _ { \# } \nu _ { N } , \mathsf { T } _ { L } ( \nu _ { N } , x _ { i } ) ) \| } & { } \\ { \displaystyle } & { \leq \displaystyle \operatorname* { m a x } _ { i \in [ N ] } \| ( \phi _ { L + 1 } \circ \Gamma _ { L + 1 } ) ( \mathsf { T } _ { L } ( \nu _ { N } ) _ { \# } \nu _ { N } , z _ { i } ) \| } \\ { \displaystyle } & { \lesssim \displaystyle \operatorname* { m a x } _ { i \in [ N ] } \| z _ { i } \| } \\ { \displaystyle } & { \lesssim \displaystyle \operatorname* { m a x } _ { i \in [ N ] } \| x _ { i } \| , } \end{array}
$$

where the inequality in the third line follows from the proof of the base case, and the inequality in the final line follows from the induction hypothesis. We conclude the proof. □

Lemma B.16 guarantees that when a transformer is applied to an empirical measure $\nu _ { N }$ , the in-context map is uniformly bounded by the radius of the support of $\nu _ { N }$ , up to multiplicative constants which naturally may depend exponentially on the number of layers

## B.4 Wasserstein Law of Large Numbers

One variant of the Wasserstein law of large numbers asserts that

$$
\operatorname* { l i m } _ { N  \infty } \mathbb { E } \mathsf { W } _ { 1 } ( \mu , \mu _ { N } ) = 0 ,
$$

where $\mu \in \mathcal P ( \mathbb { R } ^ { d } )$ has finite first moment and $\mu _ { N }$ is its associated empirical measure. For every $\mathsf { W } _ { 1 }$ -compact set $K \subset \mathcal { P } _ { 1 } ( \mathbb { R } ^ { d } )$ , one can actually prove that

$$
\operatorname* { l i m } _ { N \to \infty } \operatorname* { s u p } _ { \mu \in { \cal K } } \mathbb { E } \mathsf { W } _ { 1 } ( \mu , \mu _ { N } ) = 0 .
$$

Moreover, for more structured compact sets $K ,$ the rate of convergence in the previous display can be quantified. The following is a well-known quantitative Wasserstein law of large numbers for measures supported in a ball of radius R (which is a $\mathsf { W } _ { 1 }$ -compact set). The proof, based on a chaining argument, is included for posterity. Proposition B.17. Let $d \in \mathbb { N }$ with $d \geq 3$ , let $\Omega \subset \mathbb { R } ^ { d }$ be a compact set, and let $R : = \operatorname* { s u p } _ { x \in \Omega } \| x \|$ . Then

$$
\operatorname* { s u p } _ { \mu \in \mathcal { P } ( \Omega ) } \mathbb { E } \mathbb { W } _ { 1 } ( \mu , \mu _ { N } ) \leq C _ { d } R \bigg ( 1 + \sqrt { \log ( 1 + N ^ { 1 / d } ) } \bigg ) N ^ { - 1 / d } \quad f o r \ a l l \quad N \in \mathbb { N } ,
$$

where $C _ { d }$ is a constant depending only on d and $\begin{array} { r } { \mu _ { N } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \delta _ { X _ { n } } ~ f o r ~ X _ { n } \sim \mu } \end{array}$ iid.

Proof. We will prove the result for $\Omega = B _ { R }$ , but the proof can easily be adapted to the case of general Ω. Let $\operatorname { L i p } _ { 1 } ( R ) = \{ f \colon B _ { R } \to \mathbb { R } \mid f$ is 1-Lipschitz, $f ( 0 ) = 0 \}$ . Then

$$
\begin{array} { r l } { \mathbb { E } \mathsf { W } _ { 1 } ( \mu , \mu _ { N } ) = \mathbb { E } \Bigg [ \underset { f \in \mathrm { L i p } _ { 1 } ( R ) } { \operatorname* { s u p } } \frac { 1 } { N } \underset { i = 1 } { \overset { N } { \sum } } ( f ( X _ { i } ) - \mathbb { E } _ { \mu } [ f ] ) \Bigg ] } & { } \\ { \le 2 \mathbb { E } \mathbb { E } _ { \{ \sigma _ { 1 } , \ldots , \sigma _ { N } \} \sim ( \mathrm { U n i f } \{ 1 , - 1 \} ) ^ { \otimes N } } \Bigg [ \underset { f \in \mathrm { L i p } _ { 1 } ( R ) } { \operatorname* { s u p } } \frac { 1 } { N } \underset { i = 1 } { \overset { N } { \sum } } \sigma _ { i } f ( X _ { i } ) \Bigg ] } & { } \\ { \le 2 \underset { \tau > 0 } { \operatorname* { i n f } } \Bigg ( \tau + \frac { 1 2 } { \sqrt { N } } \int _ { \tau } ^ { 2 R } \sqrt { \log \mathcal { N } ( \mathrm { L i p } _ { 1 } ( R ) , \lVert \cdot \rVert _ { \infty } , \epsilon ) } d \epsilon \Bigg ) , } \end{array}
$$

where we applied the symmetrization identity for Rademacher complexities (Shalev-Shwartz and Ben-David, 2014) and Dudley’s chaining bound (Dudley, 2018). Note that the diameter bound in the integral follows from the fact that $f \in \operatorname { L i p } _ { 1 } ( R )$ maps $B _ { R }$ into $B _ { R }$ . Next, we need to estimate the covering number of the class of 1-Lipschitz functions. Dudley (2018) proves that

$$
\log \mathcal { N } ( \mathrm { L i p } _ { 1 } ( R ) , \| \cdot \| _ { \infty } , \epsilon ) \lesssim \biggl ( 1 + \frac { R } { \epsilon } \biggr ) ^ { d } \log ( 1 + R / \epsilon ) ,
$$

where the implicit constant is dimension-dependent. We can then upper bound the integral as

$$
\begin{array} { r l } {  { \int _ { \tau } ^ { 2 R } \sqrt { \log \mathcal { N } ( \mathrm { L i p } _ { 1 } ( R ) , \| \cdot \| _ { \infty } , \epsilon ) } d \epsilon \lesssim \int _ { \tau } ^ { R } \sqrt { ( 1 + \frac { R } { \epsilon } ) ^ { d } \log ( 1 + R / \epsilon ) } d \epsilon } } \\ & { \leq \sqrt { \log ( 1 + R / \tau ) } \int _ { \tau } ^ { R } ( 1 + \frac { R } { \epsilon } ) ^ { d / 2 } d \epsilon } \\ & { \lesssim \sqrt { \log ( 1 + R / \tau ) } ( ( R - \tau ) + R ^ { d / 2 } \int _ { \tau } ^ { R } \epsilon ^ { - d / 2 } d \epsilon ) } \\ & { \leq \sqrt { \log ( 1 + R / \tau ) } ( ( R - \tau ) + \frac { 2 } { d - 2 } R ^ { d / 2 } \tau ^ { ( 2 - d ) / 2 } ) } \\ & { \leq \sqrt { \log ( 1 + R / \tau ) } R ^ { d / 2 } \tau ^ { ( 2 - d ) / 2 } . } \end{array}
$$

Thus, taking $\tau = R N ^ { - 1 / d }$ , we get the bound

$$
\mathbb { E } \mathsf { W } _ { 1 } ( \mu , \mu _ { N } ) \lesssim R \bigg ( 1 + \sqrt { \log ( 1 + N ^ { 1 / d } ) } \bigg ) N ^ { - 1 / d } ,
$$

where the implicit constants are independent of R and depend on $\mu$ only through the dimension d. This proves that

$$
\operatorname* { s u p } _ { \mu \in \mathcal { P } ( B _ { R } ) } \mathbb { E } \mathbb { W } _ { 1 } ( \mu , \mu _ { N } ) \lesssim R \biggr ( 1 + \sqrt { \log ( 1 + N ^ { 1 / d } ) } \biggr ) N ^ { - 1 / d } .
$$

The proof is complete.

Below, we extend the Wasserstein law of large numbers from $\mathcal { P } ( B _ { R } )$ to $\mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ using a truncation argument. Proposition B.18. Fix $d \in \mathbb { N }$ with $d \geq 3$ . There exists a constant $C _ { d , \alpha , \beta }$ depending only on $d , \alpha .$ and $\beta$ such that

$$
\operatorname* { s u p } _ { \substack { \mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) } } \mathbb { E } \mathbb { W } _ { 1 } ( \mu , \mu _ { N } ) \leq C _ { d , \alpha , \beta } \sqrt { \log ( N ) \big ( 1 + \log ( 1 + N ^ { 1 / d } ) \big ) } N ^ { - 1 / d } \quad f o r \ a l l \quad N \in \mathbb { N } .
$$

Proof. For $R > 0$ and $\mu \in \mathcal P ( \mathbb { R } ^ { d } )$ , we let $\Pi _ { R } \colon \mathbb { R } ^ { d } \to B _ { R } \subset \mathbb { R } ^ { d }$ denote the truncation operator to $B _ { R }$ and $\mu _ { R } \ : = \ : ( \Pi _ { R } ) _ { \# } \mu$ . Given $\mu \in \mathsf { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } )$ and iid samples $X _ { 1 } , \dots , X _ { N } \ \sim \ \mu$ we define the random measures $\begin{array} { r } { \mu _ { N } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \delta _ { X _ { i } } } \end{array}$ and $\begin{array} { r } { \mu _ { N , R } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \delta _ { \Pi _ { R } ( X _ { i } ) } } \end{array}$ . Then by the triangle inequality, we have

$$
\mathbb { E } \mathbb { W } _ { 1 } ( \mu , \mu _ { N } ) \leq \mathbb { W } _ { 1 } ( \mu , \mu _ { R } ) + \mathbb { E } \mathbb { W } _ { 1 } ( \mu _ { R } , \mu _ { N , R } ) + \mathbb { E } \mathbb { W } _ { 1 } ( \mu _ { N , R } , \mu _ { N } ) .
$$

For the first term, we have

$$
\begin{array} { l } { \mathsf { W } _ { 1 } ( \mu , \mu _ { R } ) \leq \displaystyle \int \| x - \Pi _ { R } ( x ) \| \mu ( d x ) } \\ { \quad \leq \displaystyle \int _ { \| x \| > R } \| x \| \mu ( d x ) } \\ { \quad \leq \left( \displaystyle \operatorname* { s u p } _ { \nu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) } \int \| x \| ^ { 2 } \nu ( d x ) \right) ^ { 1 / 2 } \sqrt { \alpha } \exp \biggl ( - \frac { R ^ { 2 } } { 2 \beta } \biggr ) . } \end{array}
$$

By Lemma B.1, the supremum above is finite. For the second term, Proposition B.17 guarantees that

$$
\mathbb { E } \mathbb { W } _ { 1 } ( \mu _ { R } , \mu _ { N , R } ) \le C _ { d } R \bigg ( 1 + \sqrt { \log ( 1 + N ^ { 1 / d } ) } \bigg ) N ^ { - 1 / d }
$$

for some dimension-dependent constant $C _ { d } ,$ uniformly in $\mu .$ For the third term, we have

$$
\begin{array} { l } { \displaystyle \mathbb { E } \mathsf { W } _ { 1 } ( \mu _ { N , R } , \mu _ { N } ) \leq \mathbb { E } \Bigg [ \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \| \Pi _ { R } ( X _ { i } ) - X _ { i } \| \Bigg ] } \\ { \displaystyle \qquad = \int \| \Pi _ { R } ( x ) - x \| \mu ( d x ) } \\ { \displaystyle \qquad \leq \left( \operatorname* { s u p } _ { \nu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) } \int \| x \| ^ { 2 } \nu ( d x ) \right) ^ { 1 / 2 } \sqrt { \alpha } \exp \left( - \frac { R ^ { 2 } } { 2 \beta } \right) } \end{array}
$$

yet again. We therefore have that

$$
\begin{array} { r l } & { \underset { \mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) } { \operatorname* { s u p } } \mathbb { E } \mathbb { W } _ { 1 } ( \mu , \mu _ { N } ) \leq 2 \left( \underset { { \nu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) } } { \operatorname* { s u p } } \int \| x \| ^ { 2 } \mu ( d x ) \right) ^ { 1 / 2 } \sqrt { \alpha } \exp \left( - \frac { R ^ { 2 } } { 2 \beta } \right) } \\ & { \quad \quad \quad \quad \quad \quad \quad + C _ { d } R \bigg ( 1 + \sqrt { \log ( 1 + N ^ { 1 / d } ) } \bigg ) N ^ { - 1 / d } . } \end{array}
$$

By choosing $R = \sqrt { ( 2 \beta ) / d \log ( N ) }$ , we have

$$
\begin{array} { r l } & { \underset { \mu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) } { \operatorname* { s u p } } \mathbb { E } \mathbb { W } _ { 1 } ( \mu , \mu _ { N } ) \leq 2 \Bigg ( \underset { \nu \in \mathrm { S G } _ { \alpha , \beta } ( \mathbb { R } ^ { d } ) } { \operatorname* { s u p } } \int \| x \| ^ { 2 } \mu ( d x ) \Bigg ) ^ { 1 / 2 } \sqrt { \alpha } N ^ { - 1 / d } } \\ & { \qquad + C _ { d } \sqrt { ( 2 \beta ) / d \log ( N ) } \Bigg ( 1 + \sqrt { \log ( 1 + N ^ { 1 / d } ) } \Bigg ) N ^ { - 1 / d } . } \\ & { \qquad \leq C _ { d , \alpha , \beta } \sqrt { \log ( N ) } \Bigg ( 1 + \sqrt { \log ( 1 + N ^ { 1 / d } ) } \Bigg ) N ^ { - 1 / d } } \\ & { \qquad \leq C _ { d , \alpha , \beta } \sqrt { \log ( N ) \big ( 1 + \log ( 1 + N ^ { 1 / d } ) \big ) } N ^ { - 1 / d } . } \end{array}
$$

This concludes the proof.

## B.5 Covering Number Estimates

The proof of Theorem 3.10 uses estimates on the covering numbers of various function classes associated with multi-head attention layers. We first quote the standard upper bound on the metric entropy of Euclidean balls, which can be found, $\mathrm { e . g . } _ { \mathrm { } } . . \mathrm { } $ in work of Chewi et al. (2025, Chapter 3).

Lemma B.19. For any $\epsilon > 0$ and $R > 0 .$ , it holds that

$$
\begin{array} { r } { \log \Bigl ( \mathcal { N } ( B _ { R } ; \epsilon , \| \cdot \| ) \Bigr ) \le d \log ( 1 + 2 R \epsilon ^ { - 1 } ) . } \end{array}
$$

The following result contains the main upper bound used to control the entropy integrals associated with multi-head attention.

Lemma B.20. Define the function classes

$$
{ \mathcal { F } } _ { 1 , i } ( R ) : = \{ y \mapsto e ^ { \langle x , A y \rangle } ( V y ) _ { i } \colon x \in B _ { R } \}
$$

and

$$
{ \mathcal { F } } _ { 2 } ( R ) : = \{ y \mapsto e ^ { \langle x , A y \rangle } : x \in B _ { R } \} .
$$

Then, for any $\epsilon > 0$ , it holds that

$$
\begin{array} { r l } { \underset { i \in [ d ] } { \operatorname* { m a x } } \log \mathcal { N } ( \mathcal { F } _ { 1 , i } ( R ) ; \epsilon , \| \cdot \| _ { L ^ { \infty } ( B _ { R } ) } ) \leq d \log \Bigl ( 1 + 2 c _ { A } c _ { V } R ^ { 3 } e ^ { c _ { A } R ^ { 2 } } \epsilon ^ { - 1 } \Bigr ) } & { \ a n d } \\ { \quad } & { \qquad \log \mathcal { N } ( \mathcal { F } _ { 2 } ( R ) ; \epsilon , \| \cdot \| _ { L ^ { \infty } ( B _ { R } ) } ) \leq d \log \Bigl ( 1 + 2 c _ { A } R ^ { 2 } e ^ { c _ { A } R ^ { 2 } } \epsilon ^ { - 1 } \Bigr ) . } \end{array}
$$

Proof. Fix $i \in [ d ]$ . Observe that, for any $x \in B _ { R } , x ^ { \prime } \in B _ { R }$ , and $y \in B _ { R } .$ , it holds that

$$
\left| e ^ { \langle x , A y \rangle } ( V y ) _ { i } - e ^ { \langle x ^ { \prime } , A y \rangle } ( V y ) _ { i } \right| \leq c _ { V } R { \Big | } e ^ { \langle x , A y \rangle } - e ^ { \langle x ^ { \prime } , A y \rangle } { \Big | } .
$$

The mapping $x \mapsto e ^ { \langle x , A y \rangle }$ has gradient given by $e ^ { \langle x , A y \rangle } A y$ , and hence is $c _ { A } R e ^ { c _ { A } R ^ { 2 } }$ Lipschitz on $B _ { R }$ when $y \in B _ { R }$ Along with the previous estimate, this implies that

$$
\left| e ^ { \langle x , A y \rangle } ( V y ) _ { i } - e ^ { \langle x ^ { \prime } , A y \rangle } ( V y ) _ { i } \right| \leq c _ { A } c _ { V } R ^ { 2 } e ^ { c _ { A } R ^ { 2 } } \| x - x ^ { \prime } \| \quad \mathrm { f o r ~ a l l } \quad ( x , x ^ { \prime } , y ) \in ( B _ { R } ) ^ { 3 } \quad \mathrm { a n d } \quad i \in [ d ] .
$$

It follows that a $( c _ { A } ^ { - 1 } c _ { V } ^ { - 1 } R ^ { - 2 } e ^ { - c _ { A } R ^ { 2 } } )$ ϵ-covering of $B _ { R }$ in the Euclidean norm induces an ϵ covering of $\mathcal { F } _ { 1 , i } ( R )$ in the $L ^ { \infty } ( B _ { R } )$ norm. Therefore,

$$
\begin{array} { r l } & { \underset { i \in [ d ] } { \operatorname* { m a x } } \log \bigl ( \mathcal { N } ( \mathcal { F } _ { 1 , i } ( R ) ; \epsilon , \| \cdot \| _ { L ^ { \infty } ( B _ { R } ) } ) \bigr ) \leq \log \Bigl ( \mathcal { N } \bigl ( B _ { R } ; ( c _ { A } ^ { - 1 } c _ { V } ^ { - 1 } R ^ { - 2 } e ^ { - c _ { A } R ^ { 2 } } ) \epsilon , \| \cdot \| \bigr ) \Bigr ) } \\ & { \qquad \leq d \log \Bigl ( 1 + 2 c _ { A } c _ { V } R ^ { 3 } e ^ { c _ { A } R ^ { 2 } } \epsilon ^ { - 1 } \Bigr ) , } \end{array}
$$

where the final estimate follows from Lemma B.19. This proves the first inequality in the statement of the lemma. The second inequality follows by an essentially analogous argument. Using the fact that $x \mapsto e ^ { \langle x , A y \rangle }$ is $c _ { A } R e ^ { c _ { A } R ^ { 2 } }$ -Lipschitz on $B _ { R }$ uniformly in $y \in B _ { R }$ , it holds that

$$
\begin{array} { r l r } {  { \log \mathcal { N } ( \mathcal { F } _ { 2 } ( R ) ; \epsilon , \| \cdot \| _ { L ^ { \infty } ( B _ { R } ) } ) \le \log ( B _ { R } ; c _ { A } ^ { - 1 } R ^ { - 1 } e ^ { - c _ { A } R ^ { 2 } } \epsilon , \| \cdot \| ) } } \\ & { } & { \le d \log ( 1 + 2 c _ { A } R ^ { 2 } e ^ { c _ { A } R ^ { 2 } } \epsilon ^ { - 1 } ) , \quad } \end{array}
$$

where the last line follows from Lemma B.19.

The following elementary estimate is useful in upper bounding entropy integrals.

Lemma B.21. For any $C > 0 .$ it holds that

$$
\int _ { 0 } ^ { C } { \sqrt { \log ( 1 + x ^ { - 1 } ) } } d x \leq 2 { \sqrt { C } } .
$$

Proof. Since $1 + x \leq e ^ { x }$ for $x > 0$ , it follows that

$$
\int _ { 0 } ^ { C } { \sqrt { \log ( 1 + x ^ { - 1 } ) } } d x \leq \int _ { 0 } ^ { C } x ^ { - 1 / 2 } d x = 2 { \sqrt { C } } .
$$

The proof is complete.