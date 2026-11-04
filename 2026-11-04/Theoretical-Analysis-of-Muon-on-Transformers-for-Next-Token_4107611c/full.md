# Theoretical Analysis of Muon on Transformers for Next-Token Prediction

Shengwei Wu <sup>\*</sup> <sup>1</sup> Yixuan Yang <sup>\*</sup> <sup>1</sup> Gao Huang <sup>\*</sup> <sup>2</sup> Song Li <sup>1</sup>

## Abstract

Muon has recently shown strong empirical performance in training Transformer-based large language models (LLMs), yet its optimization dynamics remain poorly understood from a theoretical perspective. In this paper, we analyze the training dynamics of Muon for next-token prediction (NTP) in Transformers. The training procedure consists of two phases: the preprocessing phase learns the output-value component, while the main phase learns the key-query component. For the preprocessing phase, we characterize the optimization trajectory exactly and show that the normalized spectral margin is optimal after each update. For the main phase, we establish nonasymptotic convergence rates for the training loss and the normalized margin gap under softmax attention. These results suggest that Muon’s update rule can help models efficiently learn which tokens matter for prediction, offering a possible explanation for its strong performance in large language model training. The learned weight structure also yields a generalization guarantee for unseen contexts. Furthermore, we validate our theoretical insights with experiments on next-token prediction tasks.

## 1. Introduction

The Transformer (Vaswani et al., 2017) has become a popular architecture in modern machine learning. Large language models (LLMs) built on this architecture have demonstrated strong capabilities in language understanding (Devlin et al., 2019), text generation and in-context learning (Brown et al., 2020), and reasoning tasks (Wei et al., 2022). Training these models requires substantial data and computational resources, and the choice of optimizer can affect both training efficiency and model performance. Understanding the behavior of optimization algorithms can therefore help reduce training costs and guide the development of more effective training methods for Transformers.

Adaptive first order methods such as Adam and AdamW are standard choices for Transformer training (Kingma & Ba, 2015; Loshchilov & Hutter, 2019; Vaswani et al., 2017; Brown et al., 2020). These methods treat parameters as vectors and adapt the update scale of each coordinate separately, without explicitly using the matrix structure of the weights. Matrix structure are particularly relevant to Transformers, where key and query projections interact through inner products to determine attention scores, while value and output transformations jointly map the attended information to prediction logits. More recently, Muon was proposed as an optimizer for hidden layer weight matrices, constructing update directions by approximately orthogonalizing matrix momentum (Jordan et al., 2024). It has shown strong empirical performance in image classification and language model training, including large language model pretraining (Jordan et al., 2024; Liu et al., 2025).

For a loss function L(W) with a matrix parameter W, Muon updates the momentum and parameter matrices as follows:

$$
\begin{array} { r l } & { M _ { t } = \beta M _ { t - 1 } + ( 1 - \beta ) \nabla \mathcal { L } ( W _ { t } ) , } \\ & { W _ { t + 1 } = W _ { t } - \eta _ { t } { U _ { t } V _ { t } } ^ { \top } . } \end{array}
$$

Here, $\beta \in [ 0 , 1 )$ is the momentum parameter, and $\eta _ { t } > 0$ is the step size. The matrices $U _ { t }$ and $V _ { t }$ are the left and right singular vector matrices in the compact singular value decomposition $M _ { t } ~ = ~ U _ { t } \Sigma _ { t } V _ { t } ^ { \top }$ , where $\Sigma _ { t }$ contains the nonzero singular values on its diagonal. Muon thus constructs its update direction by preserving the singular vectors of the momentum matrix and setting its nonzero singular values to one, yielding the compact polar factor $U _ { t } V _ { t } ^ { \top }$ . In practice, this factor is approximated using Newton–Schulz iterations, avoiding an explicit singular value decomposition (Bernstein & Newhouse, 2024; Jordan et al., 2024).

In this paper, we study the training dynamics of Muon for next-token prediction in a one-layer Transformer model. Training this model requires learning how to map tokens to output labels and select relevant information from the context. Our study focuses on the following question:

How does Muon optimize Transformers for next-token prediction?

We investigate this question through a two phase training procedure. During the preprocessing phase (Phase I), Muon trains the linear mapping from token representations to output scores using contexts containing a single token. During the main phase (Phase II), this mapping remains unchanged while Muon trains the matrix that determines the attention scores of tokens in the context.

Contributions. We characterize Muon’s training dynamics under this two phase procedure.

• We characterize the entire Muon trajectory in the preprocessing phase. Every normalized iterate attains the optimal spectral margin, and the cross entropy loss decays at an explicit rate. We further prove that the training loss in the main phase converges to the limit determined by the preprocessing phase. Together, these results clarify the roles of the two phases: preprocessing determines how confidently each token predicts its target label, while the main phase learns to direct attention toward the correct token in the context.

• In the main phase, we prove that the normalized spectral margin converges to its optimal value. We also derive explicit nonasymptotic bounds for the training loss and the normalized margin gap, quantifying how Muon’s updates help attention distinguish the correct token from other tokens in the same context. With step sizes proportional to $( t + 1 ) ^ { - 1 / 2 }$ , our bound on the normalized margin gap is $O ( ( 1 + \log t ) / \sqrt { t } )$ . This is the fastest asymptotic order among the bounds obtained for the power law schedules considered, providing theoretical support for square root step size decay.

• We establish a finite iteration lower bound on the probability of the correct next token in unseen contexts. This offers a theoretical perspective on the generalization ability of the Transformer architecture, showing how attention can reuse learned token relations to support prediction in new contexts.

## 2. Preliminaries

Notations. For a matrix A, let $A ^ { \top }$ denote its transpose and $A [ i , j ]$ denote its $( i , j ) \ – \ t h$ entry. For a positive integer m, write $[ m ] : = \{ 1 , \dots , m \}$ . For a vector u, the notation $\lVert u \rVert _ { 2 }$ denotes the Euclidean norm. The Frobenius inner product is defined by $\langle A , B \rangle : = \operatorname { T r } ( A ^ { \top } B )$ . We use $\| A \| _ { F } , \| A \| _ { \mathrm { o p } } .$ , and $\| A \|$ <sub>∗</sub> to denote the Frobenius, spectral, and nuclear norms, respectively. The spectral and nuclear norms are dual: $\| A \| _ { * } = \operatorname* { s u p } _ { \| B \| _ { \mathrm { o p } } \leq 1 } \left. A , B \right.$ . For a vector $a \in \mathbb { R } ^ { m }$ , the softmax function is defined coordinatewise by [softmax $\begin{array} { r } { \mathbf { \Sigma } ( a ) ] _ { i } = \exp ( a _ { i } ) / \left( \sum _ { j = 1 } ^ { m } \exp ( a _ { j } ) \right) } \end{array}$ for $i \in [ m ]$ The vector $e _ { i } \in \mathbb { R } ^ { K }$ denotes the i-th standard basis vector. For nonnegative sequences $a _ { t }$ and $b _ { t } .$ , with $b _ { t } > 0$ for all sufficiently large t, we write $a _ { t } = O ( b _ { t } )$ if $a _ { t } \leq C b _ { t }$ and $a _ { t } ~ = ~ \Omega ( b _ { t } )$ if $a _ { t } ~ \geq ~ c b _ { t }$ for all sufficiently large t, where $C , c > 0$ are constants independent of t. We write $a _ { t } = o ( b _ { t } )$ if $a _ { t } / b _ { t } \to 0$ as $t \to \infty$

Next-token prediction. Let $\nu \subset \mathbb { R } ^ { d }$ be a finite vocabulary of K orthonormal token embeddings, where $2 \leq K \leq d .$ In high dimensional spaces, suitably normalized random vectors are often nearly pairwise orthogonal (Vershynin, 2019). We adopt the exact orthonormal model to remove correlations among token representations and focus on the effect of Muon’s matrix updates.

Let $I : \mathcal { V } \to [ K ]$ be a bijection assigning each token an index. The training dataset consists of N samples,

$$
\begin{array} { r } { \mathcal { D } : = \Big ( ( X ^ { ( n ) } , x _ { L _ { n } + 1 } ^ { ( n ) } ) \Big ) _ { n = 1 } ^ { N } , } \end{array}
$$

where $X ^ { ( n ) } = [ x _ { 1 } ^ { ( n ) } , \dots , x _ { { L } _ { n } } ^ { ( n ) } ]$ is a nonempty context of tokens in V and $x _ { L _ { n } + 1 } ^ { ( n ) } \in \mathcal { V }$ is the next token. For each sample, $q _ { n } : = x _ { L _ { n } } ^ { ( n ) }$ is the query token and $y _ { n } : = I ( x _ { L _ { n } + 1 } ^ { ( n ) } )$ is the target label. We assume that for every $n \in [ N ]$ and $1 \leq j < L _ { n } , \mathcal { D }$ also contains a sample that uses the first $j$ tokens of $X ^ { ( n ) }$ to predict $x _ { j + 1 } ^ { ( n ) }$

For the preprocessing phase, let $\mathcal { D } _ { 0 }$ denote the set of all training pairs whose contexts contain a single token. We assume that every token in V appears as a context in $\mathcal { D } _ { 0 }$ The corresponding target labels are specified by a bijection $h : \mathcal { V } \to [ K ]$ , so that $h ( x ) = I ( x ^ { \prime } )$ for every $( [ x ] , x ^ { \prime } ) \in \mathcal { D } _ { 0 }$

Architecture. We consider a one-layer Transformer model with softmax attention and a linear prediction layer (Huang et al., 2024; Li et al., 2024; Ildiz et al., 2024; Tarzanagh et al., 2023). Let $X = [ x _ { 1 } , \dots , x _ { L } ] \in \mathbb { R } ^ { d \times L }$ denote an input sequence and $q : = x _ { L }$ its query token. Let $W ^ { \mathrm { k } } , W ^ { \mathrm { q } } , W ^ { \mathrm { \bar { v } } } \in \bar { \mathbb R } ^ { d \times d }$ be the key, query, and value matrices, respectively, and let $W ^ { \mathrm { o } } \in \bar { \mathbb { R } ^ { K \times d } }$ be the output matrix. The model produces the following probability vector over the vocabulary:

$$
f ( X ) : = \mathrm { s o f t m a x } \left( W ^ { \circ } W ^ { \vee } X \mathrm { s o f t m a x } \left( ( W ^ { \mathrm { k } } X ) ^ { \top } W ^ { \mathrm { q } } q \right) \right) .
$$

The inner softmax assigns attention weights to the contextual tokens, and the outer softmax converts prediction logits into vocabulary probabilities.

To simplify the theoretical analysis, we directly train the combined prediction and attention matrices, following similar simplifications in related theoretical studies (Tarzanagh et al., 2023; Li et al., 2024; Ildiz et al., 2024; Huang et al.,

2024; 2025b;a; Julistiono et al., 2026):

$$
\begin{array} { r } { W _ { 1 } : = W ^ { \mathrm { o } } W ^ { \mathrm { v } } \in \mathbb R ^ { K \times d } , \qquad W _ { 2 } : = ( W ^ { \mathrm { k } } ) ^ { \top } W ^ { \mathrm { q } } \in \mathbb R ^ { d \times d } . } \end{array}
$$

We apply Muon directly to $W _ { 1 }$ and $W _ { 2 }$ , which determine the prediction logits and attention scores, respectively. Writing $\theta : = \left( W _ { 1 } , W _ { 2 } \right)$ , the prediction probabilities are

$$
f _ { \theta } ( X ) : = \mathrm { s o f t m a x } ( W _ { 1 } X \alpha _ { \theta } ( X ) ) ,
$$

where $\alpha _ { \theta } ( X ) : = \operatorname { s o f t m a x } ( X ^ { \top } W _ { 2 } q )$ is the vector of attention weights. The vector $X \alpha _ { \theta } ( X )$ is a weighted average of the contextual token embeddings, and $W _ { 1 }$ maps this representation to prediction logits.

The vector $f _ { \theta } ( \boldsymbol { X } ) ~ \in ~ \mathbb { R } ^ { K }$ gives the probabilities of the vocabulary tokens through the index map I. In particular, the probability assigned to the next token $x _ { L + 1 }$ is $\mathbb { P } _ { \theta } \{ x _ { L + 1 } \mid X \} : = [ f _ { \theta } ( X ) ] _ { I ( x _ { L + 1 } ) } .$

Loss. We train the model by minimizing the empirical cross entropy loss

$$
\mathcal { L } ( \theta ) : = - \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \log \left( e _ { y _ { n } } ^ { \top } f _ { \theta } ( X ^ { ( n ) } ) \right) .
$$

Algorithm. We train the model with Muon in two phases, as summarized in Algorithm 1. Both $W _ { 1 }$ and $W _ { 2 }$ are initialized at zero.

During the preprocessing phase (Phase I), we update only $W _ { 1 }$ on $\mathcal { D } _ { 0 }$ to learn the correspondence between tokens and their target labels. Since each context in $\mathcal { D } _ { 0 }$ contains a single token, its attention weight is one, and the prediction depends only on $W _ { 1 }$ . The loss in the preprocessing phase is obtained by replacing the full dataset $\mathcal { D }$ with $\mathcal { D } _ { 0 }$ in the definition of L:

$$
\mathcal { L } _ { 0 } ( W _ { 1 } ) : = - \frac { 1 } { K } \sum _ { x \in \mathcal { V } } \log \left( e _ { h ( x ) } ^ { \top } \mathrm { s o f t m a x } ( W _ { 1 } x ) \right) .
$$

After T preprocessing iterations, we train attention on the full dataset D by updating only $W _ { 2 }$ in the main phase (Phase II). The prediction matrix $W _ { 1 , T }$ remains unchanged, and the corresponding training loss is $\mathcal { L } _ { T } ( W _ { 2 } ) : = \mathcal { L } ( W _ { 1 , T } , W _ { 2 } )$

## 3. Main Results

This section is organized as follows. We first introduce the assumptions for the two phase analysis. We then present the exact characterization of the preprocessing phase and the nonasymptotic bounds for the training loss and normalized margin in the main phase. Finally, we study prediction on unseen contexts.

We impose three assumptions on the training contexts.

Algorithm 1 Two Phase Muon Training   
1: PHASE I: Preprocessing   
2: Input $\mathcal { D } _ { 0 } ; T ; \bar { \{ \eta _ { 1 , s } \} } _ { s = 0 } ^ { T - 1 } ; \beta _ { 1 } \in [ 0 , 1 ) .$   
3: Initialization $W _ { 1 , 0 } = 0$ and $M _ { 1 , - 1 } = 0 .$   
4: for $s = 0 , 1 , \ldots , T - 1$ do   
5: $M _ { 1 , s } = \beta _ { 1 } M _ { 1 , s - 1 } + ( 1 - \beta _ { 1 } ) \nabla \mathcal { L } _ { 0 } ( W _ { 1 , s } ) .$   
6: $\begin{array} { r } { ( U _ { 1 , s } , \Sigma _ { 1 , s } , V _ { 1 , s } ) = \mathrm { S V D } ( M _ { 1 , s } ) . } \end{array}$   
7: $\begin{array} { r } { W _ { 1 , s + 1 } = W _ { 1 , s } - \eta _ { 1 , s } U _ { 1 , s } ( V _ { 1 , s } ) ^ { \top } . } \end{array}$   
8: end for   
9: Output $W _ { 1 , T } .$   
10: PHASE II: Main Training   
11: Input D; $W _ { 1 , T } ; t ; \{ \eta _ { 2 , r } \} _ { r = 0 } ^ { t - 1 } ; \beta _ { 2 } \in [ 0 , 1 ) .$   
12: Initialization $W _ { 2 , 0 } = 0$ and $M _ { 2 , - 1 } = 0 .$   
13: for $r = 0 , 1 , \ldots , t - 1$ do   
14: $M _ { 2 , r } = \beta _ { 2 } M _ { 2 , r - 1 } + ( 1 - \beta _ { 2 } ) \nabla \mathcal { L } _ { T } ( W _ { 2 , r } ) .$   
15: $( U _ { 2 , r } , \Sigma _ { 2 , r } , V _ { 2 , r } ) = \mathrm { S V D } ( M _ { 2 , r } ) .$   
16: $W _ { 2 , r + 1 } = W _ { 2 , r } - \eta _ { 2 , r } U _ { 2 , r } ( V _ { 2 , r } ) ^ { \top }$   
17: end for   
18: Output $( W _ { 1 , T } , W _ { 2 , t } ) .$

Assumption 1. For every n $\mathbf { \mu } \in [ N ]$ , the context $X ^ { ( n ) }$ contains the token $x _ { n } ^ { \star } : = h ^ { - 1 } ( y _ { n } )$

Assumption 1 requires each context to contain a token whose label under h matches the target label. We refer to $x _ { n } ^ { \star }$ as the correct token in the context.

To state the remaining two assumptions, let $c _ { n , x }$ denote the number of occurrences of token $x \in \nu$ in $X ^ { ( n ) }$ . We write $\mathcal { X } _ { n } ^ { - } : = \{ x \in \mathcal { V } : c _ { n , x } > 0 , x \neq x _ { n } ^ { \star } \}$ for the set of tokens appearing in the context other than the correct token.

Assumption 2. For every $n \in [ N ] ,$ , the correct token occurs at least as often as any other token in the same context:

$$
c _ { n , x _ { n } ^ { \star } } \geq c _ { n , x } \qquad f o r e \nu e r y x \in \mathcal { X } _ { n } ^ { - } .
$$

Assumption 2 ensures that the correct token receives at least as much total attention weight as any other token at zero initialization. Indeed, each position receives weight $1 / L _ { n } ,$ so the total attention weight over all occurrences of token x is $c _ { n , x } / L _ { n }$

Assumption 3. There exists a matrix $W \in \mathbb { R } ^ { d \times d }$ that separates the correct token from every other token in each training context:

$$
( x _ { n } ^ { \star } - x ) ^ { \top } W q _ { n } \geq 1 \qquad ( n \in [ N ] , x \in \mathcal { X } _ { n } ^ { - } ) .
$$

Assumption 3 requires a common matrix that assigns the correct token a strictly larger attention score than every other token in each training context. The separating matrix W need not minimize the spectral norm. Similar separability assumptions are used in attention maximum-margin analyses (Tarzanagh et al., 2023; Vasudeva et al., 2025; Julistiono et al., 2026).

## 3.1. Preprocessing Phase: Exact Training Dynamics

We first analyze the preprocessing phase (Phase I), which updates $W _ { 1 }$ using examples whose contexts contain only one token. We then turn to Phase II, where $W _ { 1 }$ remains fixed at $W _ { 1 , T }$ and only $W _ { 2 }$ is updated.

In this phase, $W _ { 1 }$ learns to distinguish each token’s target label from the other labels. To characterize the label separation achieved by Muon’s iterates, we introduce a spectral hard margin problem associated with the preprocessing dataset $\mathcal { D } _ { 0 }$ . Let $R _ { 1 } ^ { \star }$ denote its optimal value:

$$
\begin{array} { r l } { \underset { W \in \mathbb { R } ^ { K \times d } } { \mathrm { m i n } } } & { \| W \| _ { \mathrm { o p } } } \\ { \mathrm { s u b j e c t ~ t o } } & { ( e _ { h ( x ) } - e _ { v } ) ^ { \top } W x \geq 1 , } \\ & { \forall x \in \mathcal { V } , \quad v \in [ K ] \setminus \{ h ( x ) \} . } \end{array}\tag{1}
$$

Let $\gamma _ { 1 } : = 1 / R _ { 1 } ^ { \star }$ denote the optimal normalized spectral margin. For any $W \in \mathbb { R } ^ { K \times \bar { d } }$ , define the unnormalized margin by

$$
\rho _ { 1 } ( W ) : = \operatorname* { m i n } _ { x \in \mathcal { V } } \operatorname* { m i n } _ { v \in [ K ] \backslash \{ h ( x ) \} } ( e _ { h ( x ) } - e _ { v } ) ^ { \top } W x .
$$

The following theorem characterizes every preprocessing iterate exactly.

Theorem 1. Consider the preprocessing phase (Phase I) iterates generated by Algorithm 1. Fix $\beta _ { 1 } ~ \in ~ [ 0 , 1 )$ and $\eta _ { 1 , 0 } > 0 ,$ , and take $\eta _ { 1 , s } = \eta _ { 1 , 0 } ( s + 1 ) ^ { - 1 / 2 } f o r s \ge 0 .$ . For each integer $T \geq 1$ , define $\begin{array} { r } { \Delta _ { T } : = \sum _ { s = 0 } ^ { T - 1 } \eta _ { 1 , s } } \end{array}$ . Then, for every $x \in \nu$ and $v \in [ K ] \setminus \{ h ( x ) \}$

$$
( e _ { h ( x ) } - e _ { v } ) ^ { \top } W _ { 1 , T } x = \| W _ { 1 , T } \| _ { \mathrm { o p } } = \Delta _ { T } .\tag{2}
$$

The loss satisfies

$$
\begin{array} { r l r } {  { \mathcal { L } _ { 0 } ( W _ { 1 , T } ) = \log \big ( 1 + ( K - 1 ) e ^ { - \Delta _ { T } } \big ) } } \\ & { } & { \leq ( K - 1 ) e ^ { - \eta _ { 1 , 0 } \sqrt { T } } . } \end{array}\tag{3}
$$

Moreover, $R _ { 1 } ^ { \star } = 1$ , and every normalized iterate attains the optimal spectral margin:

$$
\rho _ { 1 } \left( { \frac { W _ { 1 , T } } { \lVert W _ { 1 , T } \rVert _ { \mathrm { o p } } } } \right) = \gamma _ { 1 } = 1 .\tag{4}
$$

Theorem 1 shows that Muon attains the optimal normalized spectral margin after the first update and maintains it throughout the preprocessing phase. Subsequent updates enlarge the common logit gap $\Delta _ { T }$ , increasing the probability assigned to each token’s target label and reducing the cross entropy loss.

Remark 1. The uniform weights 1/K in ${ \mathcal D } _ { 0 } ,$ , orthonormal token embeddings, and zero initialization keep the gradient and momentum proportional to afixed matrix, giving Muon a constant polar update direction. The iterates therefore depend on the step sizes only through $\Delta _ { T }$ , and the exact identities hold for any positive step size sequence. The square root schedule gives the loss decay bound (3).

## 3.2. Main Phase: Spectral Maximum Margin Convergence

The preprocessing phase produces a prediction matrix $W _ { 1 , T }$ that associates each token with its target label. With this matrix unchanged, the main phase (Phase II) trains $W _ { 2 }$ to assign the correct token a larger attention score than every other token in the same context. To quantify this separation relative to the spectral norm of $W _ { 2 }$ , we introduce a spectral hard margin problem based on the attention score gaps.

For an attention matrix $W \in \mathbb { R } ^ { d \times d }$ , the score gap between the correct token $x _ { n } ^ { \star }$ and another token $x \in \mathcal { X } _ { n } ^ { - }$ is $( x _ { n } ^ { \star } -$ $x ) ^ { \top } W q _ { n }$ . We consider the nontrivial case in which at least one training context contains a token other than its correct token. Let $R _ { 2 } ^ { \star }$ denote the minimum spectral norm needed to make every such gap at least one:

$$
\begin{array} { r l } { \underset { W \in \mathbb R ^ { d \times d } } { \mathrm { m i n } } } & { \| W \| _ { \mathrm { o p } } } \\ { \mathrm { s u b j e c t ~ t o } } & { ( x _ { n } ^ { \star } - x ) ^ { \top } W q _ { n } \geq 1 , } \\ & { \forall n \in [ N ] , \quad x \in \mathcal { X } _ { n } ^ { - } . } \end{array}\tag{5}
$$

Assumption 3 ensures that this problem is feasible.

The unnormalized training margin is the smallest attention score gap across the training contexts:

$$
\rho _ { 2 } ( W ) : = \operatorname* { m i n } _ { \substack { n \in [ N ] , x \in \mathcal { X } _ { n } ^ { - } } } ( x _ { n } ^ { \star } - x ) ^ { \top } W q _ { n } .
$$

Let $\gamma _ { 2 } ~ : = ~ 1 / R _ { 2 } ^ { \star }$ denote the spectral maximum margin value, which is the largest value of $\rho _ { 2 } ( W )$ attainable under $\| W \| _ { \mathrm { o p } } \leq 1$

Our main result gives nonasymptotic bounds for the training loss and the normalized spectral margin gap in the main phase. We write $\ell _ { T } ^ { \star } : = \mathcal { L } _ { 0 } ( W _ { 1 , T } )$ for the loss after T preprocessing iterations, whose exact value is given in (3).

Theorem 2. Suppose that Assumptions 1, 2, and 3 hold. Fix the preprocessing parameters as in Theorem 1, and let $\beta _ { 2 } \in [ 0 , 1 )$ . There exists $\bar { T } \geq 0$ such that, for every $T > \bar { T } $ there exists $\bar { \eta } _ { T } > 0$ with the following property. For any $0 < \eta _ { 2 , 0 } \leq \bar { \eta } _ { T }$ , let $\{ W _ { 2 , t } \} _ { t \ge 0 }$ be the main phase iterates generated by Algorithm 1 after T preprocessing iterations, with step sizes $\eta _ { 2 , t } = \eta _ { 2 , 0 } ( t + 1 ) ^ { - 1 / 2 }$ . Then there exist constants $C _ { 1 } , C _ { 2 } , C _ { 3 } , C _ { 4 } > 0 $ , independent of t, such that thefollowing bounds hold.

(i) Training loss. For every $t \geq 1$

$$
0 < \mathcal { L } _ { T } ( W _ { 2 , t } ) - \ell _ { T } ^ { \star } \leq C _ { 1 } t ^ { C _ { 2 } } e ^ { - C _ { 3 } \sqrt { t } } .\tag{6}
$$

(ii) Normalized spectral margin. There exists $t _ { T } \geq 1$ such that, for every $t \geq t _ { T } , W _ { 2 , t } \neq 0$ and

$$
0 \leq \gamma _ { 2 } - \rho _ { 2 } \left( \frac { W _ { 2 , t } } { \| W _ { 2 , t } \| _ { \mathrm { o p } } } \right) \leq C _ { 4 } \frac { 1 + \log t } { \sqrt { t } } .\tag{7}
$$

The constants $C _ { 1 } , C _ { 2 } , C _ { 3 } , C _ { 4 }$ and $t _ { T }$ depend only on $K ,$ $\Delta _ { T } , \beta _ { 2 } , \eta _ { 2 , 0 }$ , and the finite training data.

Together, Theorems 1 and 2 characterize how Muon’s updates affect token prediction and attention training. In the preprocessing phase, Muon attains the optimal normalized spectral margin after the first update, and subsequent updates increase the probability assigned to each token’s target label. The resulting prediction matrix encodes the correspondence between tokens and target labels and determines the limiting prediction confidence in the main phase. With this matrix unchanged, the main phase increases attention score gaps and concentrates attention on the correct token in each training context. The excess loss $E _ { T } ( W ) : = { \mathcal { L } } _ { T } ( W ) - \ell _ { T } ^ { \star }$ therefore vanishes, and the training loss converges to the value $\ell _ { T } ^ { \star }$ determined by the preprocessing phase.

The normalized margin result connects Muon’s polar updates to attention score separation relative to input sensitivity. The spectral norm measures the largest amplification of changes in the query representation by the attention parameter matrix:

$$
\operatorname* { s u p } _ { \| \boldsymbol { \delta q } \| _ { 2 } \le 1 } \| \boldsymbol { W } \boldsymbol { \delta q } \| _ { 2 } = \| \boldsymbol { W } \| _ { \mathrm { o p } } \le \| \boldsymbol { W } \| _ { \mathrm { F } } .
$$

With the contextual token representations fixed, this amplification directly controls how changes in the query affect attention scores. Theorem 2 shows that the minimum score gap between the correct token and other tokens across the training contexts, normalized by this amplification, approaches its largest possible value.

Remark 2. Our analysis also covers main phase step sizes $\eta _ { 2 , t } = \eta _ { 2 , 0 } ( t + 1 ) ^ { - \alpha }$ with $0 < \alpha \leq 1$ ,for eachfixed α and sufficiently small $\eta _ { 2 , 0 } .$ . Among these schedules, $\alpha = 1 / 2$ gives the fastest asymptotic order of the upper bound on the normalized margin gap, namely $O ( ( 1 + \log t ) / \sqrt { t } )$ . These rate bounds support using square root step size decay in the main phase. Performance at afixed iteration also depends on the initial step size and bound constants.

Prediction on Unseen Contexts. We next show how the attention score orderings learned during training can support prediction on unseen contexts. For a fixed query $q ,$ the score difference $( u - v ) ^ { \top } W _ { 2 } q$ does not depend on the other tokens in the context. Thus, once the model assigns u a strictly larger attention score than v, the same ordering holds in any new context containing both tokens with query $q .$

We write $u  _ { q }$ v if some training context with query q has u as its correct token and also contains $v \neq u .$ . Once learned, these orderings yield score inequalities that can be combined along finite chains with a common query, even across training contexts.

For an unseen sample $( X , x _ { L + 1 } )$ , let $X = [ x _ { 1 } , \dots , x _ { L } ]$ $q = x _ { L }$ , and $x _ { X } ^ { \star } : = h ^ { - 1 } ( I ( x _ { L + 1 } ) )$ ). We consider samples for which $x _ { X } ^ { \star }$ appears in X and

$$
x _ { X } ^ { \star } \to _ { q } \cdot \cdot \cdot \to _ { q } x _ { \ell } ( \ell \in [ L ] , x _ { \ell } \neq x _ { X } ^ { \star } ) .\tag{8}
$$

Each chain consists of finitely many steps $u \to _ { q }$ v arising from training contexts with query $q .$ Its intermediate tokens need not appear in $X$

Theorem 3. Let the iterates be as in Theorem 2. Suppose that an unseen sample $( X , x _ { L + 1 } )$ satisfies (8) and that its context contains the correct token $x _ { X } ^ { \star }$ . Then, for all sufficiently large t,

$$
\begin{array} { r l } {  { \mathbb { P } _ { ( W _ { 1 , T } , W _ { 2 , t } ) } \{ x _ { L + 1 } ~ | ~ X \} } } \\ & { \geq e ^ { - \ell _ { T } ^ { \star } } - \frac { \Delta _ { T } ( L - 1 ) } { 2 } C _ { 1 } t ^ { C _ { 2 } } e ^ { - C _ { 3 } \sqrt { t } } , } \end{array}\tag{9}
$$

where $C _ { 1 } , C _ { 2 } , C _ { 3 }$ are the constants in (6).

The first term $e ^ { - \ell _ { T } ^ { \star } }$ is the maximum probability assigned to the correct next token with the prediction matrix $W _ { 1 , T }$ unchanged. It is obtained when all attention is assigned to $x _ { X } ^ { \star }$ , whose target label then has a logit gap of $\Delta _ { T }$ over all other labels. The second term bounds the reduction in this probability caused by attention on other tokens and vanishes as main phase training proceeds. For fixed T, the prediction probability therefore converges to $e ^ { - \ell _ { T } ^ { \star } }$ as $t \to \infty$ . Taking $T \to \infty$ afterward increases the preprocessing logit gap $\Delta _ { T }$ without bound, so this limiting probability approaches one. This result shows how attention can combine score orderings learned across training contexts to predict on new combinations of contextual tokens with the same query.

Extensions to Schatten $S _ { p }$ Updates and Adam. The appendix extends the two phase analysis to Schatten $S _ { p }$ steepest descent and Adam under the same data assumptions. For each fixed $1 ~ \leq ~ p ~ < ~ \infty$ , the Schatten extension uses normalized momentum steepest descent in the same $S _ { p }$ norm in both phases. Both phases permit any exact maximizing direction when the momentum is nonzero, and use the zero direction at zero momentum. The Schatten $S _ { p }$ norm is the $\ell _ { p }$ norm of the singular values. With sufficiently long preprocessing and a small enough Phase II initial step size, $\rho _ { 2 } ( W _ { 2 , t } ) / \Vert W _ { 2 , t } \Vert _ { S _ { p } }$ converges to the corresponding maximum margin with gap $O ( ( 1 + \log t ) / \sqrt { t } )$ under square root step sizes in both phases.

The Adam extension uses bias correction and a fixed $\varepsilon \ > \ 0$ added outside the square root in the denominator. Both phases use Adam with step sizes $\begin{array} { r l } { \eta _ { j , s } } & { { } = } \end{array}$ $\eta _ { j , 0 } ( s + 1 ) ^ { - \alpha _ { j } }$ , where $0 ~ < ~ \alpha _ { j } ~ \le ~ 1 ~ \mathrm { f o r } ~ j ~ \in ~ \{ 1 , 2 \}$ The prediction matrix $W _ { 1 , T }$ remains fixed during the main phase. With sufficiently small initial step sizes in both phases and sufficiently long preprocessing, the normalized margin $\rho _ { 2 } ( W _ { 2 , t } ) / \| W _ { 2 , t } \| _ { F }$ converges to the

![](images/d4ee4d05733f20e6c0f3b51e4e4b0af072087c41a6aab9a1519bc6edc582a063.jpg)  
(a)

![](images/b3052865dfe44e9e56bcd48fcd3f1d5f2d964adb117dd8645307a3c17d9eefce.jpg)  
(b)

![](images/0c8a9c4ffceac81f4692e4b40665e2f0e6f4873c524fd0e309938d341dd73388.jpg)  
(c)

![](images/ec339ad67909b4e03bb131d71da3832fbc46b7105b2f73a63675a15d49253e71.jpg)  
(d)  
Figure 1. Normalized margin behavior of Muon and Adam on the synthetic task. Subfigures (a) and (b) show the relative spectral and Frobenius margin gaps $G _ { \infty } ( t )$ and $G _ { 2 } ( t )$ , respectively. Subfigures (c) and (d) show the Frobenius cosine similarities to the unique Frobenius hard margin solution $\widehat { V } _ { 2 }$ and a selected spectral hard margin solution $\widehat { V } _ { \infty }$ , respectively. Each curve shows the mean over ten independent runs.

Frobenius maximum margin value. The margin gap is $O ( 1 / \log t )$ for $0 ~ < ~ \alpha _ { 2 } ~ < ~ 1$ and $O ( 1 / \log \log t )$ for $\alpha _ { 2 } = 1$ . The normalized attention matrix $W _ { 2 , t } / \Vert W _ { 2 , t } \Vert _ { F }$ also converges to the unique Frobenius hard margin direction.

These extensions relate the update rule to the norm used for margin normalization: Muon leads to a spectral maximum margin limit, whereas Adam leads to a Frobenius maximum margin limit.

## 4. Numerical Experiments

We compare the normalized margins and parameter directions of Muon and Adam on synthetic tasks and examine how Muon’s step sizes affect its margin gap.

Both experiments use $K = 6 4$ orthonormal token embeddings in $\mathbb { R } ^ { 1 0 2 4 }$ and $N = 1 0 2 4$ training contexts. We train the prediction matrix $W _ { 1 , T }$ with Muon in Phase I and hold it fixed throughout Phase II. All step sizes specified below are used to update the attention matrix $W _ { 2 }$ during Phase II.

For the optimizer comparison in Figure 1, the maximum context length is 26. Within each run, Muon and Adam use the same dataset and prediction matrix and start from $W _ { 2 , 0 } = 0$ . This allows us to compare their margin behavior under the same training objective and initialization.

Let $\gamma _ { 2 }$ and $\gamma _ { 2 , F }$ denote the spectral and Frobenius maximum margin values, respectively. For nonzero iterates, we define the relative spectral and Frobenius margin gaps by

$$
G _ { \infty } ( t ) : = 1 - \frac { \rho _ { 2 } ( W _ { 2 , t } ) } { \gamma _ { 2 } \| W _ { 2 , t } \| _ { \mathrm { o p } } } ,
$$

$$
G _ { 2 } ( t ) : = 1 - \frac { \rho _ { 2 } ( W _ { 2 , t } ) } { \gamma _ { 2 , F } \| W _ { 2 , t } \| _ { F } } .
$$

Smaller gaps indicate that the normalized margin is closer to its maximum value under the corresponding norm.

Each curve in Figure 1 averages ten independent runs of 20,000 main phase iterations. Muon uses momentum parameter $\beta _ { 2 } = 0 . 9$ and step size $( t + 1 ) ^ { - 1 / 2 }$ . Adam uses step size $0 . 0 1 ( t + 1 ) ^ { - 1 / 2 }$ , first and second moment parameters 0.9 and 0.999, respectively, $\varepsilon = 1 . 0 \times 1 0 ^ { - 6 }$

For nonzero iterates, we also compare parameter directions using the Frobenius cosine similarity

$$
\cos _ { F } ( W _ { 2 , t } , \widehat { V } _ { p } ) : = \frac { \left. W _ { 2 , t } , \widehat { V } _ { p } \right. } { \| W _ { 2 , t } \| _ { F } \| \widehat { V } _ { p } \| _ { F } } , \qquad p \in \{ 2 , \infty \} ,
$$

where $\widehat { V } _ { 2 }$ is the unique solution to the Frobenius hard margin problem and $\widehat { V } _ { \infty }$ is a selected solution to the spectral hard margin problem.

$\mathrm { A t } ~ t = 2 0 { , } 0 0 0$ , Muon has the smaller spectral margin gap, while Adam has the smaller Frobenius margin gap. Muon also has the larger cosine similarity to $\widehat { V } _ { \infty }$ , and Adam to $\widehat { V } _ { 2 }$ These observations illustrate how the update rule affects the normalized margin and parameter direction under the same training objective.

For the step size comparison, we run each setting once for 8,000 main phase iterations. All settings use the same dataset and preprocessing output. We use $\eta _ { 2 , t } = \eta _ { 2 , 0 } ( t +$ $1 ) ^ { - \alpha }$ with $\eta _ { 2 , 0 } \in \{ 0 . 5 , 1 , 2 \}$ and $\alpha \in \{ 1 / 4 , 1 / 2 , 3 / 4 , 1 \}$

For nonzero iterates, let

$$
D ( t ) : = \left| \rho _ { 2 } \left( \frac { W _ { 2 , t } } { \| W _ { 2 , t } \| _ { \mathrm { o p } } } \right) - \gamma _ { 2 } \right|
$$

denote the absolute gap in the normalized spectral margin. Since the normalized spectral margin is at most $\gamma _ { 2 }$ , we have

![](images/09bc32c00b0adb4dffac150610e7f17e2b34f9e572378821d7a4b4e66c611ebe.jpg)  
(a)  
(b)  
(c)  
Figure 2. Dependence of Muon’s relative spectral margin gap $G _ { \infty } ( t )$ on the initial step size and decay exponent. Subfigures $( \mathbf { a } ) , ( \mathbf { b } )$ , and (c) use $\eta _ { 2 , 0 } = 0 . 5 ,$ 1, and 2, respectively, and compare $\alpha \in \{ 1 / 4 , 1 / 2 , 3 / 4 , 1 \}$ under the step size schedule $\eta _ { 2 , t } = \eta _ { 2 , 0 } ( t + 1 ) ^ { - \alpha }$ . The exponent giving the smallest observed gap varies with both $\eta _ { 2 , 0 }$ and the iteration count.

Table 1. Normalized spectral margin gap $D ( t )$ and the terms $1 / S _ { t }$ and $Q _ { t } / S _ { t }$ for $\eta _ { 2 , 0 } = 0 . 5$ at iteration $t = 8 , \dot { 0 } 0 0$ . The last column indicates the larger of the two terms for each decay exponent.
<table><tr><td>α</td><td> $D ( t )$ </td><td> $1 / S _ { t }$ </td><td> $Q _ { t } / S _ { t }$ </td><td>Larger term</td></tr><tr><td> $1 / 4$ </td><td>0.086223</td><td>0.001774</td><td>0.078711</td><td> $Q _ { t } / S _ { t }$ </td></tr><tr><td> $1 / 2$ </td><td>0.075970</td><td>0.011272</td><td>0.026953</td><td> ${ Q _ { t } } / { S _ { t } }$ </td></tr><tr><td> $3 / 4$ </td><td>0.114558</td><td>0.058158</td><td>0.037658</td><td> $1 / S _ { t }$ </td></tr><tr><td>1</td><td>0.122928</td><td>0.209107</td><td>0.085985</td><td> $1 / S _ { t }$ </td></tr></table>

$D ( t ) = \gamma _ { 2 } G _ { \infty } ( t )$ . Figure 2 plots $G _ { \infty } ( t )$ , while Table 1 lists $D ( t )$

For $t \geq 1$ , let $S _ { t }$ and $Q _ { t }$ denote the cumulative step size and the sum of squared step sizes, respectively:

$$
S _ { t } : = \sum _ { r = 0 } ^ { t - 1 } \eta _ { 2 , r } , \qquad Q _ { t } : = \sum _ { r = 0 } ^ { t - 1 } \eta _ { 2 , r } ^ { 2 } .
$$

The normalized margin bound (B.24) contains the two terms $1 / S _ { t }$ and $Q _ { t } / S _ { t }$ . For $0 \textless \alpha \textless 1$ , we have $S _ { t } =$ $\Theta ( \eta _ { 2 , 0 } t ^ { 1 - \alpha } )$ , while $S _ { t } = \Theta ( \eta _ { 2 , 0 } \log t )$ for $\alpha = 1$ . The sum $Q _ { t }$ grows polynomially for $\alpha < 1 / 2 .$ , logarithmically for $\alpha = 1 / 2$ , and remains bounded for $\alpha > 1 / 2$ . Consequently, $Q _ { t } / S _ { t }$ dominates $1 / S _ { t }$ asymptotically for $\alpha \leq 1 / 2$ , and the two terms have the same order for $\alpha > 1 / 2$

Figure 2 shows that the exponent giving the smallest observed gap varies with both the initial step size and the iteration count. $\mathrm { A t } \ t = 8 , 0 0 0 , \alpha = 1 / 2$ gives the smallest gap for all three initial step sizes. Table 1 lists $D ( t ) , 1 / S _ { t }$ , and $Q _ { t } / S _ { t }$ for $\eta _ { 2 , 0 } = 0 . 5$ at this iteration. Here $Q _ { t } / S _ { t }$ is larger for $\alpha \in \{ 1 / 4 , 1 / 2 \}$ , while $1 / S _ { t }$ is larger for $\alpha \in \{ 3 / 4 , 1 \}$ Among the four exponents, $\alpha = 1 / 2$ also gives the fastest asymptotic decay of the margin upper bound.

## 5. Related Work

Transformers and NTP. Prior works interpret Transformer forward computation as implementing gradient descent, preconditioned gradient descent, and mirror descent (Akyurek et al.¨ , 2023; Von Oswald et al., 2023; Ahn et al., 2023; D’Angelo & Flammarion, 2026). Other analyses study retrieval and composition in next token prediction, maximum margin token selection, and the directional convergence of self attention (Li et al., 2024; Tarzanagh et al., 2023; Ataee Tarzanagh et al., 2023; Vasudeva et al., 2025). Related work examines implicit bias in linear NTP and one layer softmax attention, sign gradient descent on two layer Transformers, regular language recognition, Markov structure, representation geometry, and next token prediction capacity (Thrampoulidis, 2024; Sheen et al., 2024; Li et al., 2025; Huang et al., 2025a; Ildiz et al., 2024; Zhao et al., 2024; Madden et al., 2025). Softmax attention has also been studied under mirror descent, preconditioned optimization, and multihead dynamics (Julistiono et al., 2026; Goel et al., 2026; Chen et al., 2024), while prior work has analyzed next-token prediction (Huang et al., 2024).

Muon. Muon approximates the compact polar factor of momentum matrices using Newton-Schulz iterations and has been scaled to large language model training through adjusted update scales and weight decay (Jordan et al., 2024; Bernstein & Newhouse, 2024; Liu et al., 2025). Recent theory studies layerwise loss decrease under spectral updates, simplified Muon in matrix factorization and linear Transformers, associative memory and Transformer feedforward modules, spectral descent in imbalanced models (Davis & Drusvyatskiy, 2025; Ma et al., 2026; Wang et al., 2026; Vasudeva et al., 2026). Other work provides convergence guarantees, spectral norm constrained interpretations, and broader non Euclidean formulations (Shen et al., 2025;

Yang et al., 2026; Chen et al., 2026; Crawshaw et al., 2026). These results do not characterize Muon’s implicit bias under nonlinear softmax attention.

Maximum Margin Theory. Classical work establishes directional convergence (Soudry et al., 2018; Nacson et al., 2019) and develops primal-dual analyses of maximum margin solutions (Ji & Telgarsky, 2021). Later studies derive faster margin convergence rates (Ji et al., 2021; Wang et al., 2023) and generalized margin results under broader optimization geometries (Sun et al., 2022; 2023). Extensions to homogeneous networks yield KKT characterizations of normalized limit points (Lyu & Li, 2020; Hui et al., 2025), while analyses of separable multiclass linear classification establish directional convergence under suitable assumptions (Ravi et al., 2024). For matrix parameters, recent work studies spectral maximum margin convergence for spectral descent and exact SVD Muon, norm dependent KKT characterizations for Adam and Muon, and optimizer dependent simplicity bias in deep linear networks (Fan et al., 2025; Gronich & Vardi, 2026; Dragutinovic & Ranganath´ , 2026).

In comparison, we analyze Muon’s training dynamics in a Transformer with nonlinear softmax attention for nexttoken prediction. Our analysis accounts for the interaction of nonlinear attention with momentum and polar updates. We characterize the Phase I trajectory exactly, establish nonasymptotic bounds for the training loss and normalized margin gap in Phase II, and derive finite iteration lower bounds on the probability of the correct next token.

## 6. Conclusion and Future Work

Our analysis characterizes Muon’s two phase training in a Transformer with nonlinear softmax attention. After preprocessing, the output value matrix is held fixed while Phase II drives the excess cross entropy to zero and the normalized key query margin to the spectral maximum margin value. For contexts covered by our guarantees, Phase I determines the limiting probability of the correct next token.

Among the power law schedules considered, $\alpha = 1 / 2$ gives the fastest asymptotic order in the proved normalized margin bound. Experiments also illustrate how optimizer choice and step size schedules affect normalized margins.

Future work includes joint training of the two matrices, extensions to deeper and multihead Transformers, and analysis with finite Newton–Schulz iterations, stochastic gradients, and weaker NTP data assumptions.

## Acknowledgements

This work was supported by NSFC under grant number 12671672.

## References

Ahn, K., Cheng, X., Daneshmand, H., and Sra, S. Transformers learn to implement preconditioned gradient descent for in-context learning. In Advances in Neural Information Processing Systems, volume 36, pp. 45614– 45650. Curran Associates, Inc., 2023.

Akyurek, E., Schuurmans, D., Andreas, J., Ma, T., and¨ Zhou, D. What learning algorithm is in-context learning? investigations with linear models. In International Conference on Learning Representations, 2023. URL https: //openreview.net/forum?id=0g0X4H8yN4I.

Ataee Tarzanagh, D., Li, Y., Zhang, X., and Oymak, S. Max-margin token selection in attention mechanism. In Advances in Neural Information Processing Systems, volume 36, pp. 48314–48362. Curran Associates, Inc., 2023.

Bernstein, J. and Newhouse, L. Old optimizer, new norm: An anthology. arXiv preprint arXiv:2409.20325, 2024.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pp. 1877–1901. Curran Associates, Inc., 2020.

Chen, L., Li, J., and Liu, Q. Muon optimizes under spectral norm constraints. Transactions on Machine Learning Research, 2026. URL https://openreview.net /forum?id=Blz4hjxLwU.

Chen, S., Sheen, H., Wang, T., and Yang, Z. Training dynamics of multi-head softmax attention for in-context learning: Emergence, convergence, and optimality (extended abstract). In Proceedings of the Thirty-Seventh Conference on Learning Theory, volume 247 of Proceedings ofMachine Learning Research, pp. 4573–4573. PMLR, 2024. URL https://proceedings.mlr.pres s/v247/siyu24a.html.

Crawshaw, M., Modi, C., Liu, M., and Gower, R. M. An exploration of non-euclidean gradient descent: Muon and its many variants. In International Conference on Machine Learning, 2026. URL https://openrevi ew.net/forum?id=O5xoqSmNzc.

D’Angelo, F. and Flammarion, N. Transformers learn latent mixture models in-context via mirror descent. In International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id= SHidElLSVt.

Davis, D. and Drusvyatskiy, D. When do spectral gradient updates help in deep learning? arXiv preprint arXiv:2512.04299, 2025.

Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference ofthe North American chapter ofthe associationfor computational linguistics: human language technologies, volume 1 (long and short papers), pp. 4171–4186, 2019.

Dragutinovic, S. and Ranganath, R. To use or not to use´ muon: How simplicity bias in optimizers matters. In ICLR Workshop on Scientific Methodsfor Understanding Deep Learning, 2026. URL https://openreview .net/forum?id=GsZtgQf3IM.

Fan, C., Schmidt, M., and Thrampoulidis, C. Implicit bias of spectral descent and muon on multiclass separable data. In Advances in Neural Information Processing Systems, volume 38, pp. 39622–39669. Curran Associates, Inc., 2025. URL https://proceedings.neurips. cc/paper\_files/paper/2025/hash/38643 2c7534eec9a1cd7cbeea90d7e9f-Abstrac t-Conference.html.

Goel, G., Soltanolkotabi, M., and Bartlett, P. Training dynamics of softmax self-attention: Fast global convergence via preconditioning. In ICML Workshop on Highdimensional Learning Dynamics, 2026. URL https: //openreview.net/forum?id=oL1GngGtP0.

Gronich, E. and Vardi, G. The implicit bias of adam and muon on smooth homogeneous neural networks. In International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id=DpIc 1cpNKG.

Huang, R., Liang, Y., and Yang, J. Non-asymptotic convergence of training transformers for next-token prediction. In Advances in Neural Information Processing Systems, volume 37, pp. 80634–80673. Curran Associates, Inc., 2024.

Huang, R., Liang, Y., and Yang, J. How transformers learn regular language recognition: A theoretical study on training dynamics and implicit bias. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 25516–25549. PMLR, 2025a. URL https://proceedings.mlr.press/v267/h uang25s.html.

Huang, Y., Wen, Z., Chi, Y., and Liang, Y. A theoretical analysis of self-supervised learning for vision transformers. In International Conference on Learning Representations, 2025b. URL https://proceedings.iclr.cc/ paper\_files/paper/2025/file/0440de96 23f97491b6de9b53afcfe6f4-Paper-Confe rence.pdf.

Hui, X., Ma, X., Yang, Y., and Li, S. The implicit regularization of gradient flow on separable datasets in relu networks. Neurocomputing, 656:131367, 2025. ISSN 0925- 2312. doi: https://doi.org/10.1016/j.neucom.2025.1313 67. URL https://www.sciencedirect.com/ science/article/pii/S0925231225020399.

Ildiz, M. E., Huang, Y., Li, Y., Rawat, A. S., and Oymak, S. From self-attention to markov models: Unveiling the dynamics of generative transformers. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pp. 20955–20982. PMLR, 2024. URL https://proc eedings.mlr.press/v235/ildiz24a.html.

Ji, Z. and Telgarsky, M. Characterizing the implicit bias via a primal-dual analysis. In Algorithmic Learning Theory, pp. 772–804. PMLR, 2021.

Ji, Z., Srebro, N., and Telgarsky, M. Fast margin maximization via dual acceleration. In International Conference on Machine Learning, pp. 4860–4869. PMLR, 2021.

Jordan, K., Jin, Y., Boza, V., You, J., Cesista, F., Newhouse, L., and Bernstein, J. Muon: An optimizer for hidden layers in neural networks. Blog post, 2024. URL https: //kellerjordan.github.io/posts/muon/. Accessed: 2026-07-27.

Julistiono, A. K., Ataee Tarzanagh, D., and Azizan, N. Optimizing attention with mirror descent: Generalized maxmargin token selection. Journal of Machine Learning Research, 27(36):1–61, 2026.

Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.

Li, B., Huang, W., Han, A., Zhou, Z., Suzuki, T., Zhu, J., and Chen, J. On the optimization and generalization of two-layer transformers with sign gradient descent. In International Conference on Learning Representations, 2025. URL https://openreview.net/forum ?id=97rOQDPmk2.

Li, Y., Huang, Y., Ildiz, M. E., Rawat, A. S., and Oymak, S. Mechanics of next token prediction with self-attention. In International Conference on Artificial Intelligence and Statistics, pp. 685–693. PMLR, 2024.

Liu, J., Su, J., Yao, X., Jiang, Z., Lai, G., Du, Y., Qin, Y., Xu, W., Lu, E., Yan, J., et al. Muon is scalable for llm training. arXiv preprint arXiv:2502.16982, 2025.

Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview .net/forum?id=Bkg6RiCqY7.

Lyu, K. and Li, J. Gradient descent maximizes the margin of homogeneous neural networks. In International Conference on Learning Representations, 2020.

Ma, J., Huang, Y., Chi, Y., and Chen, Y. Preconditioning benefits of spectral orthogonalization in muon. arXiv preprint arXiv:2601.13474, 2026.

Madden, L., Fox, C., and Thrampoulidis, C. Next-token prediction capacity: general upper bounds and a lower bound for transformers. IEEE Transactions on Information Theory, 71(9):7134–7148, 2025. doi: 10.1109/TIT. 2025.3584013. URL https://doi.org/10.110 9/TIT.2025.3584013.

Nacson, M. S., Lee, J., Gunasekar, S., Savarese, P. H. P., Srebro, N., and Soudry, D. Convergence of gradient descent on separable data. In The 22nd International Conference on Artificial Intelligence and Statistics, pp. 3420–3428. PMLR, 2019.

Ravi, H., Scott, C., Soudry, D., and Wang, Y. The implicit bias of gradient descent on separable multiclass data. In Advances in Neural Information Processing Systems, volume 37, pp. 81324–81359. Curran Associates, Inc., 2024.

Sheen, H., Chen, S., Wang, T., and Zhou, H. H. Implicit regularization of gradient flow on one-layer softmax attention. In ICLR Workshop on Bridging the Gap Between Practice and Theory in Deep Learning, 2024. URL https://iclr.cc/virtual/2024/23412.

Shen, W., Huang, R., Huang, M., Shen, C., and Zhang, J. On the convergence analysis of muon. arXiv preprint arXiv:2505.23737, 2025.

Soudry, D., Hoffer, E., Nacson, M. S., Gunasekar, S., and Srebro, N. The implicit bias of gradient descent on separable data. Journal ofMachine Learning Research, 19 (70):1–57, 2018.

Sun, H., Ahn, K., Thrampoulidis, C., and Azizan, N. Mirror descent maximizes generalized margin and can be implemented efficiently. In Advances in Neural Information Processing Systems, volume 35, pp. 31089–31101, 2022.

Sun, H., Gatmiry, K., Ahn, K., and Azizan, N. A unified approach to controlling implicit regularization via mirror descent. Journal ofMachine Learning Research, 24(393): 1–58, 2023.

Tarzanagh, D. A., Li, Y., Thrampoulidis, C., and Oymak, S. Transformers as support vector machines. In Mathematics of Modern Machine Learning Workshop at NeurIPS, 2023. URL https://openreview.net/forum ?id=gLwzzmh79K.

Thrampoulidis, C. Implicit optimization bias of next-token prediction in linear models. In Advances in Neural Information Processing Systems, volume 37, pp. 22624–22656. Curran Associates, Inc., 2024.

Vasudeva, B., Deora, P., and Thrampoulidis, C. Implicit bias and fast convergence rates for self-attention. Transactions on Machine Learning Research, 2025. URL https: //openreview.net/forum?id=pKilnjQsb0.

Vasudeva, B., Deora, P., Zhao, Y., Sharan, V., and Thrampoulidis, C. How muon’s spectral design benefits generalization: A study on imbalanced data. In International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=YzjS 4jcfmS.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.

Vershynin, R. High-dimensional probability. Cambridge Series in Statistical and Probabilistic Mathematics, 47, 2019.

Von Oswald, J., Niklasson, E., Randazzo, E., Sacramento, J., Mordvintsev, A., Zhmoginov, A., and Vladymyrov, M. Transformers learn in-context by gradient descent. In International Conference on Machine Learning, pp. 35151–35174. PMLR, 2023.

Wang, G., Hu, Z., Muthukumar, V., and Abernethy, J. D. Faster margin maximization rates for generic optimization methods. In Advances in Neural Information Processing Systems, volume 36, pp. 62488–62518, 2023.

Wang, S., Zhang, F., Li, J., Du, C., Du, C., Pang, T., Yang, Z., Hong, M., and Tan, V. Y. Muon outperforms adam in tail-end associative memory learning. In International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=twbM FL0DMp.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Yang, Y., He, Y., and Li, S. Convergence of spectral descent for non-smooth optimization. arXiv preprint arXiv:2605.26977, 2026.

Zhang, C., Zou, D., and Cao, Y. The implicit bias of adam on separable data. Advances in neural information processing systems, 37:23988–24021, 2024.

Zhao, Y., Behnia, T., Vakilian, V., and Thrampoulidis, C. Implicit geometry of next-token prediction: From language sparsity patterns to model representations. In First Conference on Language Modeling, 2024. URL https: //openreview.net/forum?id=qyilOnIRHI.

## Appendix Contents

A Additional experiments 12   
B Proofs of the main results 14   
B.1 Preprocessing: proof of Theorem 1 . 14   
B.2 Main phase: proof of Theorem 2 16   
B.3 Prediction: proof of Theorem 3 . 35   
C Convergence of Schatten Norm Updates 36   
D Convergence of Adam 50

## A. Additional experiments

We supplement the main experiments by varying the embedding dimension, vocabulary nonorthogonality, and Muon’s momentum parameter. The first two experiments compare the relative margin gaps of Muon and Adam under changes to the token embeddings. The third examines how momentum affects the decay of Muon’s spectral margin gap. The Phase II update rules and the definitions of $G _ { \infty } ( t )$ and $G _ { 2 } ( t )$ are the same as in the main experiments. The optimizer settings are given below.

Ambient embedding dimension. We vary the ambient embedding dimension $d \in \{ 6 4 , 1 2 8 , 2 5 6 , 5 1 2 , 1 0 2 4 , 2 0 4 8 \}$ while keeping the vocabulary size, the token indices in each context, and the labels fixed. Muon and adam use the same optimizer settings at every dimension.

Figure A.1 shows the relative spectral and Frobenius margin gaps at $t = 2 0 { , } 0 0 0$ . At every dimension, Muon has a smaller spectral margin gap than Adam, while Adam has a smaller Frobenius margin gap. Both gaps for Muon remain unchanged across dimensions, consistent with the invariance of its updates under isometric embeddings of the vocabulary span when the vocabulary is orthonormal and the initialization is zero. Both gaps for Adam decrease as d increases over the tested values. These results show that Muon’s smaller spectral margin gap is not specific to the embedding dimension.

![](images/c63237075c25ac548ea168d5527054aa025926d0ebcc3973d58da2acb2d93ee9.jpg)  
(a)

![](images/9b0b1d6724f3785e09bfbdb485ac935b5cea9661142cfe35d28316787d41cbe2.jpg)  
(b)  
Figure A.1. Effect of the ambient embedding dimension. Subfigures (a) and (b) show the relative spectral and Frobenius margin gaps at $t = 2 0 { , } 0 0 0$ , respectively. At every dimension, Muon has the smaller spectral margin gap, while Adam has the smaller Frobenius margin gap. Both gaps remain unchanged for Muon and decrease for Adam as d increases.

Nonorthogonal vocabularies. Our theoretical analysis assumes an orthonormal vocabulary. We examine sensitivity to this assumption by perturbing the token embeddings while keeping the token indices, context structures, query indices, and labels fixed. Let $x _ { 1 } , \ldots , x _ { K }$ be the original orthonormal token embeddings. We generate a random subspace $\mathcal { P } \subset \operatorname { s p a n } ( \mathcal { V } ) ^ { \perp }$ of dimension 16 and sample a unit vector $\xi _ { j } \in \mathcal { P }$ for each $j \in [ K ]$ . We keep this subspace and these vectors fixed across all perturbation strengths. For each $\sigma \in \{ 0 , 0 . 1 , 0 . 2 , 0 . 5 \}$ , we define

![](images/aa56d9be5536f61f5572996bd67c02facbb88d5da26de7545236a82ae6ed7ab1.jpg)

![](images/ce0afdf6fd1bae8ec31855647f02f76fd3eed155110157723dd0fe4ad26a3d36.jpg)  
(a)  
(b)  
Figure A.2. Effect of vocabulary nonorthogonality. Subfigures (a) and (b) show the relative spectral and Frobenius margin gaps at $t = 2 0 { , } 0 0 0$ , respectively, for different perturbation strengths σ. At every perturbation strength, Muon has the smaller spectral margin gap, while Adam has the smaller Frobenius margin gap.

$$
\begin{array} { r } { \widetilde x _ { j } ^ { ( \sigma ) } : = \sqrt { 1 - \sigma ^ { 2 } } x _ { j } + \sigma \xi _ { j } . } \end{array}
$$

Since $\xi _ { j }$ is orthogonal to every original token embedding, we have

$$
\begin{array} { r l } & { \left\| \widetilde { x } _ { j } ^ { ( \sigma ) } \right\| _ { 2 } ^ { 2 } = ( 1 - \sigma ^ { 2 } ) + \sigma ^ { 2 } = 1 , } \\ & { \left. \widetilde { x } _ { i } ^ { ( \sigma ) } , \widetilde { x } _ { j } ^ { ( \sigma ) } \right. = \sigma ^ { 2 } \langle \xi _ { i } , \xi _ { j } \rangle \qquad ( i \neq j ) . } \end{array}
$$

Thus the embeddings remain unit vectors, and the magnitudes of the inner products between distinct tokens scale as $\sigma ^ { 2 }$

For each perturbed vocabulary, we construct a fixed prediction matrix $W _ { 1 , T } ^ { ( \sigma ) }$ satisfying

$$
W _ { 1 , T } ^ { ( \sigma ) } \widetilde { x } _ { j } ^ { ( \sigma ) } = \Delta _ { T } \left( e _ { h ( x _ { j } ) } - K ^ { - 1 } \mathbf { 1 } \right) , \qquad j \in [ K ] .
$$

Each token therefore has the same prediction logits as after preprocessing on the original vocabulary. Muon and Adam use this same prediction matrix throughout Phase II, with their respective optimizer settings unchanged across perturbation strengths. For each vocabulary, we recompute the spectral and Frobenius maximum margin values used in $G _ { \infty } ( t )$ and $G _ { 2 } ( t )$

Figure A.2 shows the relative spectral and Frobenius margin gaps at $t = 2 0 { , } 0 0 0$ . At every perturbation strength, Muon has a smaller spectral margin gap than Adam, while Adam has a smaller Frobenius margin gap. As σ increases from 0 to 0.5, Muon’s spectral margin gap increases from approximately 0.051 to 0.087. Adam’s Frobenius margin gap increases from approximately 0.070 to 0.451 over the same range.

Momentum. We next vary Muon’s momentum parameter over $\beta _ { 2 } \in \{ 0 , 0 . 5 , 0 . 9 , 0 . 9 9 \}$ while keeping the remaining optimizer settings fixed. Figure A.3(a) compares the relative spectral margin gap over the first 20,000 Phase II iterations. The values $\beta _ { 2 } \in \{ 0 , 0 . 5 , 0 . 9 \}$ give similar final gaps, although their trajectories differ. For $\beta _ { 2 } = 0 . 9 9$ , the gap decreases more slowly and remains approximately 0.951 at $t = 2 0 { , } 0 0 0$

Figure A.3(b) shows the gap at $t = 1 0 0 , 0 0 0$ for $\beta _ { 2 } = 0 . 9 9$ and at $t = 2 0 { , } 0 0 0$ for the other tested momentum values. At $t = 1 0 0 , 0 0 0$ , the gap for $\beta _ { 2 } = 0 . 9 9$ is approximately $0 . 1 0 4 ^ { }$ , still larger than the gaps of approximately 0.041, 0.038, and 0.051 reached by the other three values at $t = 2 0 { , } 0 0 0$ . Thus $\beta _ { 2 } = 0 . 9 9$ gives slower convergence over the tested iteration range.

Under the data, preprocessing, and step size conditions of Theorem 2, the normalized spectral margin converges to $\gamma _ { 2 }$ for every fixed $\beta _ { 2 } \in [ 0 , 1 )$ . The convergence constants and the upper bound on the initial step size may depend on $\beta _ { 2 } .$ , so convergence over a fixed number of iterations can differ across momentum values.

$$
\begin{array} { r }  \overbrace  \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm {  ~ \Lambda ~ } \mathrm   ~  \end{array}
$$

![](images/7fb8df3bd3718631d3c624a07e68fe48c6e9e83985c94a620efe299daa0193ae.jpg)

![](images/a73baf93094b7ab106f248cdf46b36b9a1f7d5b5f32f7005da7b5ac58638290d.jpg)  
Figure A.3. Effect of Muon’s momentum parameter. Subfigure (a) shows the relative spectral margin gap over the first 20,000 Phase II iterations for different values of $\beta _ { 2 } ,$ , together with the Adam trajectory. Subfigure (b) shows the gap at $t = 1 0 0 , 0 0 0$ for $\beta _ { 2 } = 0 . 9 9$ and at $t = 2 0 { , } 0 0 0$ for the other momentum values. The horizontal line in (b) shows Adam’s gap at $\bar { t } = 2 0 { , } 0 0 0$ . The spectral margin gap decreases more slowly for $\beta _ { 2 } = 0 . 9 9$ over the tested iteration range.

## B. Proofs of the main results

We first prove Theorem 1 for the preprocessing phase. For the main phase, we derive loss and margin bounds for a class of matrix norms under power law step sizes. We then specialize these bounds to the spectral norm to prove Theorem 2. Finally, we use the training margin bound to prove Theorem 3.

## B.1. Preprocessing: proof of Theorem 1

We first characterize the preprocessing iterates generated by Algorithm 1 for an arbitrary positive step size sequence.   
Theorem 1 then follows by choosing the square root step size schedule.

In the preprocessing phase, each context consists of a single token. Write $\mathcal { V } = \{ x _ { 1 } , \ldots , x _ { K } \}$ , and let $y _ { i } : = h ( x _ { i } )$ denote the target label for $[ x _ { i } ]$ . Define

$$
\begin{array} { r l r l } & { X _ { \mathcal { V } } : = [ x _ { 1 } , \dots , x _ { K } ] \in \mathbb { R } ^ { d \times K } , \quad } & & { P : = [ e _ { y _ { 1 } } , \dots , e _ { y _ { K } } ] \in \mathbb { R } ^ { K \times K } , } \\ & { C : = I _ { K } - \displaystyle \frac { 1 } { K } \mathbf { 1 } \mathbf { 1 } ^ { \top } , \quad } & & { B : = P C X _ { \mathcal { V } } ^ { \top } . } \end{array}
$$

Here $C$ centers the output logits.

Since h is bijective, $P$ is a permutation matrix. Together with the orthonormality of the vocabulary embeddings and the definition of C, this gives

$$
X _ { \mathcal { V } } ^ { \top } X _ { \mathcal { V } } = I _ { K } , \qquad P { \bf 1 } = { \bf 1 } , \qquad P C = C P .
$$

The next lemma expresses $W _ { 1 , s }$ as B multiplied by the cumulative step size.

Lemma B.1. Consider the preprocessing iterates generated by Algorithm 1, using the compact polarfactor and an arbitrary positive step size sequence. Let

$$
\Delta _ { s } : = \sum _ { j = 0 } ^ { s - 1 } \eta _ { 1 , j } , \qquad \Delta _ { 0 } : = 0 .
$$

For every integer $s \geq 0 ,$ , thefollowing statements hold.

(i) The iterates and their logits satisfy

$$
\begin{array} { r } { W _ { 1 , s } = C W _ { 1 , s } = \Delta _ { s } B , } \end{array}\tag{B.1}
$$

$$
W _ { 1 , s } x _ { i } = \Delta _ { s } \left( e _ { y _ { i } } - \frac 1 K { \bf 1 } \right) , \qquad i \in [ K ] .\tag{B.2}
$$

(ii) The loss, margin, and operator norm satisfy

$$
\begin{array} { r l } & { \mathcal { L } _ { 0 } ( W _ { 1 , s } ) = \log \left( 1 + ( K - 1 ) e ^ { - \Delta _ { s } } \right) , } \\ & { \rho _ { 1 } ( W _ { 1 , s } ) = \| W _ { 1 , s } \| _ { \mathrm { o p } } = \Delta _ { s } . } \end{array}
$$

Moreover, the spectral hard margin problem in (1) has optimal value $R _ { 1 } ^ { \star } = 1$ , and hence $\gamma _ { 1 } = 1$

Proof. Orthonormality, $P C = C P ,$ , and $C ^ { 2 } = C$ give

$$
B B ^ { \top } = P C X _ { \nu } ^ { \top } X _ { \nu } C P ^ { \top } = P C P ^ { \top } = C , \qquad C B = B .
$$

Since $C$ is the orthogonal projection onto $\mathbf { 1 ^ { \perp } }$ , the identity $B B ^ { \top } = C$ implies that B has $K - 1$ nonzero singular values, all equal to one. Hence $\| B \| _ { \mathrm { o p } } = 1 , \| B \| _ { * } = K - 1$ , and the compact polar factor of B is B. Moreover,

$$
B x _ { i } = P C e _ { i } = e _ { y _ { i } } - \frac { 1 } { K } { \bf 1 } , \qquad i \in [ K ] .\tag{B.3}
$$

(i) We prove by induction that $W _ { 1 , s } = \Delta _ { s } B$ and $M _ { 1 , s - 1 } = - \mu _ { s - 1 } B$ for some $\mu _ { s - 1 } \geq 0$ . Let $\mu _ { - 1 } = 0$ . Both identities hold at $s = 0$ by zero initialization.

Suppose they hold at iteration s, and let $Z _ { s } : = e ^ { \Delta _ { s } } + K - 1$ . By the induction hypothesis, (B.3), and shift invariance of softmax, the probability of $y _ { i } \mathrm { i s } e ^ { \Delta _ { s } } / Z _ { s }$ and that of each other label is $1 / Z _ { s }$ . Using the uniform preprocessing weights and the softmax cross entropy gradient, we obtain

$$
\mathrm { s o f t m a x } ( W _ { 1 , s } x _ { i } ) - e _ { y _ { i } } = - \frac { K } { Z _ { s } } \left( e _ { y _ { i } } - \frac { 1 } { K } { \bf 1 } \right) ,
$$

$$
\nabla \mathcal { L } _ { 0 } ( W _ { 1 , s } ) = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \left( \mathrm { s o f t m a x } ( W _ { 1 , s } x _ { i } ) - e _ { y _ { i } } \right) x _ { i } ^ { \top } = - Z _ { s } ^ { - 1 } B .
$$

The momentum update therefore gives

$$
M _ { 1 , s } = - \mu _ { s } B , \qquad \mu _ { s } : = \beta _ { 1 } \mu _ { s - 1 } + ( 1 - \beta _ { 1 } ) Z _ { s } ^ { - 1 } .
$$

Since $\mu _ { s - 1 } \geq 0 , \beta _ { 1 } \in [ 0 , 1 )$ , and $Z _ { s } > 0$ , we have $\mu _ { s } > 0$ . Thus the compact polar factor of $M _ { 1 , s } \mathrm { i } s - B _ { : }$ , and the parameter update becomes

$$
W _ { 1 , s + 1 } = W _ { 1 , s } + \eta _ { 1 , s } B = ( \Delta _ { s } + \eta _ { 1 , s } ) B = \Delta _ { s + 1 } B .
$$

Both induction identities therefore hold at iteration $s + 1$ . Combining $W _ { 1 , s } = \Delta _ { s } B$ with $C B = B$ and (B.3) proves (B.1) and (B.2).

(ii) The probability assigned to the correct label is the same for every preprocessing sample, so

$$
\mathcal { L } _ { 0 } ( W _ { 1 , s } ) = - \log \frac { e ^ { \Delta _ { s } } } { e ^ { \Delta _ { s } } + K - 1 } = \log \left( 1 + ( K - 1 ) e ^ { - \Delta _ { s } } \right) .
$$

Equation (B.2) gives

$$
( e _ { y _ { i } } - e _ { v } ) ^ { \top } W _ { 1 , s } x _ { i } = \Delta _ { s } , \qquad i \in [ K ] , \quad v \in [ K ] \setminus \{ y _ { i } \} .
$$

Hence $\rho _ { 1 } ( W _ { 1 , s } ) = \Delta _ { s }$ . Since $\Delta _ { s } \geq 0$ and $\| B \| _ { \mathrm { o p } } = 1$ , we also have $\| W _ { 1 , s } \| _ { \mathrm { o p } } = \Delta _ { s }$

It remains to determine the optimal value of (1). For $i \in \left[ K \right]$ and $v \in [ K ] \setminus \{ y _ { i } \}$ , let $C _ { x _ { i } , v } : = ( e _ { y _ { i } } - e _ { v } ) x _ { i } ^ { \top }$ . Equation (B.3) gives $\langle B , C _ { x _ { i } , v } \rangle = 1$ , so $B$ is feasible. Since $\| B \| _ { \mathrm { o p } } = 1$ , this yields $R _ { 1 } ^ { \star } \leq 1$ . To obtain the reverse inequality, note that

$$
\sum _ { i = 1 } ^ { K } \sum _ { v \in [ K ] \setminus \{ y _ { i } \} } C _ { x _ { i } , v } = \sum _ { i = 1 } ^ { K } ( K e _ { y _ { i } } - \bf 1 ) x _ { i } ^ { \top } = K B .
$$

For any feasible matrix $U ,$ summing its constraints and using duality of the operator and nuclear norms gives

$$
K ( K - 1 ) \leq \left. U , \sum _ { i = 1 } ^ { K } \sum _ { \substack { v \in [ K ] \setminus \{ y _ { i } \} } } C _ { x _ { i } , v } \right. = K \left. U , B \right. \leq K \| U \| _ { \mathrm { o p } } \| B \| _ { * } = K ( K - 1 ) \| U \| _ { \mathrm { o p } } .
$$

Since $K \geq 2 .$ , every feasible matrix has operator norm at least one. Therefore $R _ { 1 } ^ { \star } = 1 \mathrm { a n d } \gamma _ { 1 } = 1$

ProofofTheorem 1. Lemma B.1 gives (2), the exact loss identity in (3), and $R _ { 1 } ^ { \star } = \gamma _ { 1 } = 1$ . For $\eta _ { 1 , s } = \eta _ { 1 , 0 } ( s + 1 ) ^ { - 1 / 2 }$ and every integer $T \geq 1$ , we have

$$
\Delta _ { T } = \eta _ { 1 , 0 } \sum _ { k = 1 } ^ { T } k ^ { - 1 / 2 } \geq \eta _ { 1 , 0 } \sqrt { T } ,
$$

where the inequality follows from $k ^ { - 1 / 2 } \geq T ^ { - 1 / 2 } \mathrm { f o r } 1 \leq k \leq T$ . Combining the exact loss identity with log $( 1 + u ) \leq u$ for $u \geq 0$ yields

$$
\begin{array} { r } { \mathcal L _ { 0 } ( W _ { 1 , T } ) \le ( K - 1 ) e ^ { - \Delta _ { T } } \le ( K - 1 ) e ^ { - \eta _ { 1 , 0 } \sqrt { T } } . } \end{array}
$$

Finally, $\rho _ { 1 } ( W _ { 1 , T } ) = \| W _ { 1 , T } \| _ { \mathrm { o p } } = \Delta _ { T } > 0$ , so positive homogeneity of $\rho _ { 1 }$ gives

$$
\rho _ { 1 } \left( \frac { W _ { 1 , T } } { \vert \vert W _ { 1 , T } \vert \vert _ { \mathrm { o p } } } \right) = \frac { \rho _ { 1 } ( W _ { 1 , T } ) } { \vert \vert W _ { 1 , T } \vert \vert _ { \mathrm { o p } } } = 1 = \gamma _ { 1 } .
$$

This proves (4).

## B.2. Main phase: proof of Theorem 2

Proof overview. With $W _ { 1 , T }$ fixed, we express the loss and gradient in terms of the total attention weight of each token. Sufficient preprocessing ensures that the correct token’s total attention weight is uniformly bounded away from zero on the initial loss sublevel set. For a sufficiently small initial step size, gradient and momentum estimates yield a descent recurrence that keeps all iterates in this set and makes the excess losses summable when weighted by the step sizes. This summability controls the accumulated error from the sharper gradient lower bound, whose leading coefficient is $\gamma _ { 2 }$ , and yields the excess loss decay estimate. Finally, we use the relation between excess loss and attention score gaps to bound the training margin from below; combining this with an upper bound on $\| W _ { 2 , t } \| _ { \mathrm { o p } }$ gives the normalized spectral margin rate.

## ATTENTION LOSS AND SEPARABILITY

Output identities and token orderings. Fix an integer $T \geq 1$ and keep $W _ { 1 , T }$ fixed throughout Phase II. Lemma B.1 gives

$$
W _ { 1 , T } = C W _ { 1 , T } = \Delta _ { T } B , \qquad \Delta _ { T } > 0 .
$$

Recall that the preprocessing loss satisfies

$$
\ell _ { T } ^ { \star } = \mathcal { L } _ { 0 } ( W _ { 1 , T } ) = \log \left( 1 + ( K - 1 ) e ^ { - \Delta _ { T } } \right) .
$$

To record the required attention score inequalities, let

$$
\begin{array} { r l r } & { \mathcal { T } _ { 2 } : = \{ ( n , x ) : n \in [ N ] , x \in \mathcal { X } _ { n } ^ { - } \} , } \\ & { \mathcal { C } _ { q } : = \{ ( x _ { n } ^ { \star } , x ) : ( n , x ) \in \mathbb { Z } _ { 2 } , q _ { n } = q \} , } & { q \in \mathcal { V } . } \end{array}
$$

Thus $( u , v ) \in \mathcal { C } _ { q }$ exactly when $u  _ { q }$ v in the notation of the main text. The next lemma characterizes separability in terms of token orderings, which will also be used to establish a gradient lower bound.

Lemma B.2. Suppose that the vocabulary is orthonormal and $\mathcal { T } _ { 2 } \neq \mathcal { O }$ . Assumption 3 holds if and only if, for every $q \in \nu$ there is a permutation $u _ { q , 1 } , \dots , u _ { q , K } o f \mathcal { V }$ such that

$$
( u _ { q , i } , u _ { q , j } ) \in \mathcal { C } _ { q } \quad \implies \quad i < j .
$$

Under these equivalent conditions, the minimum in (5) is attained, and

$$
0 < R _ { 2 } ^ { \star } < \infty , \qquad \gamma _ { 2 } > 0 .
$$

Proof. Suppose first that Assumption 3 holds, and let W be a corresponding separating matrix. For each $q \in \mathcal { V } .$ , list the tokens in nonincreasing order of $u ^ { \top } W q$ , breaking ties arbitrarily. For every $( u , v ) \in \mathcal { C } _ { q } ,$ , separability gives $( u - v ) ^ { \top } W q \geq 1$ so u precedes v.

Conversely, suppose that such a permutation exists for every $q \in \mathcal V$ , and let

$$
v _ { q } : = \sum _ { i = 1 } ^ { K } ( K - i ) u _ { q , i } , \qquad W ^ { \star } : = \sum _ { q \in \mathcal { V } } v _ { q } q ^ { \top } .
$$

Orthonormality $\mathrm { g i v e s } .$ , for every $q \in \mathcal V$ and $i \in [ K ]$

$$
W ^ { \star } q = v _ { q } , \qquad u _ { q , i } ^ { \top } v _ { q } = K - i .
$$

For every $( u _ { q , i } , u _ { q , j } ) \in \mathcal { C } _ { q } ,$ , the ordering condition gives $i < j$ , so

$$
\begin{array} { r } { ( u _ { q , i } - u _ { q , j } ) ^ { \top } W ^ { \star } q = j - i \geq 1 . } \end{array}
$$

Thus $W ^ { \star }$ is feasible for (5).

For each $( n , x ) \in \mathbb { Z } _ { 2 }$ , let $A _ { n , x } : = ( x _ { n } ^ { \star } - x ) q _ { n } ^ { \top }$ . The rank one norm identity and orthonormality give

$$
\lVert A _ { n , x } \rVert _ { * } = \lVert x _ { n } ^ { \star } - x \rVert _ { 2 } \lVert q _ { n } \rVert _ { 2 } = \sqrt { 2 } .
$$

Since $\mathcal { T } _ { 2 } \neq \emptyset , \operatorname { f i x } \left( n , x \right) \in \mathcal { T } _ { 2 }$ . For every feasible $W ,$ , duality of the operator and nuclear norms yields

$$
1 \leq \langle W , A _ { n , x } \rangle \leq \| W \| _ { \mathrm { o p } } \| A _ { n , x } \| _ { * } = \sqrt { 2 } \| W \| _ { \mathrm { o p } } .
$$

Intersecting the closed feasible set with $\{ W : \| W \| _ { \mathrm { o p } } \leq \| W ^ { \star } \| _ { \mathrm { o p } } \}$ gives a nonempty compact set. Since every feasible matrix outside this set has larger norm than $W ^ { \star }$ , minimizing the continuous operator norm on this set also solves (5). Consequently,

$$
\frac { 1 } { \sqrt { 2 } } \le R _ { 2 } ^ { \star } \le \| \boldsymbol { W } ^ { \star } \| _ { \mathrm { o p } } < \infty , \qquad \gamma _ { 2 } = \frac { 1 } { R _ { 2 } ^ { \star } } > 0 .
$$

Attention and loss identities. Fix the prediction matrix $W _ { 1 , T }$ satisfying $C W _ { 1 , T } = \Delta _ { T } B$ with $\Delta _ { T } > 0$ . For $W \in \mathbb { R } ^ { d \times d }$ the total attention weight over all occurrences of token $x \in \nu$ in sample $n \in [ N ]$ is

$$
a _ { n , x } ( W ) : = \frac { c _ { n , x } e ^ { x ^ { \top } W q _ { n } } } { \sum _ { z \in \mathcal { V } } c _ { n , z } e ^ { z ^ { \top } W q _ { n } } } .\tag{B.4}
$$

The denominator is positive because the context is nonempty. These weights are nonnegative, sum to one, and vanish whenever $c _ { n , x } = 0$ . The next lemma expresses the prediction probabilities and losses in terms of these weights and identifies the infimum of $\mathcal { L } _ { T }$

Lemma B.3. Suppose that Assumption 1 holds. Then thefollowing statements hold.

(i) For every $W \in \mathbb { R } ^ { d \times d } , n \in [ N ]$ , and $x \in \nu$ , the probability assigned to label $h ( x )$ is

$$
p _ { n , x } ( W ) : = e _ { h ( x ) } ^ { \top } f _ { ( W _ { 1 } , _ { T } , W ) } ( X ^ { ( n ) } ) = \frac { e ^ { \Delta _ { T } a _ { n , x } ( W ) } } { \sum _ { z \in \mathcal { V } } e ^ { \Delta _ { T } a _ { n , z } ( W ) } } .
$$

The corresponding sample loss satisfies

$$
\ell _ { T , n } ( W ) : = - \log p _ { n , x _ { n } ^ { \star } } ( W ) = \log \left( 1 + \sum _ { x \in \mathcal { V } \backslash \{ x _ { n } ^ { \star } \} } e ^ { - \Delta _ { T } ( a _ { n , x _ { n } ^ { \star } } ( W ) - a _ { n , x } ( W ) ) } \right) .
$$

Moreover, for every $x \in \mathcal { X } _ { n } ^ { - }$

$$
\frac { a _ { n , x } ( W ) } { a _ { n , x _ { n } ^ { \star } } ( W ) } = \frac { c _ { n , x } } { c _ { n , x _ { n } ^ { \star } } } \exp \left( - \left. W , A _ { n , x } \right. \right) .\tag{B.5}
$$

(ii) For every $W \in \mathbb { R } ^ { d \times d }$ and $n \in [ N ]$

$$
\ell _ { T , n } ( W ) \geq \ell _ { T } ^ { \star } , \qquad \mathcal { L } _ { T } ( W ) \geq \ell _ { T } ^ { \star } .
$$

IfAssumption 3 also holds, then

$$
\operatorname* { i n f } _ { W \in \mathbb { R } ^ { d \times d } } { \mathcal { L } } _ { T } ( W ) = \ell _ { T } ^ { \star } .
$$

Proof. (i) The attention output for sample n is $\textstyle \sum _ { x \in \nu } a _ { n , x } ( W ) x$ . Using $C W _ { 1 , T } = \Delta _ { T } B , B x = e _ { h ( x ) } - K ^ { - 1 } { \bf 1 }$ , and $\textstyle \sum _ { x \in \nu } a _ { n , x } ( W ) = 1$ , its centered prediction logits are

$$
C W _ { 1 , T } \sum _ { x \in \mathcal { V } } a _ { n , x } ( W ) x = \Delta _ { T } \sum _ { x \in \mathcal { V } } a _ { n , x } ( W ) e _ { h ( x ) } - \frac { \Delta _ { T } } { K } \mathbf { 1 } .
$$

Centering and the common term do not affect softmax, so bijectivity of h gives the stated expression for $p _ { n , x } ( W )$ . The sample loss formula follows from $h ( x _ { \underline { { n } } } ^ { \star } ) = y _ { n }$ . Assumption 1 ensures $c _ { n , x _ { n } ^ { \star } } > 0$ and hence $a _ { n , x _ { n } ^ { \star } } ( W ) > 0$ . Taking the ratio in (B.4) and using $A _ { n , x } = ( x _ { n } ^ { \star } - x ) q _ { n } ^ { \top }$ proves (B.5).

(ii) Since the attention weights lie in [0, 1], every difference $a _ { n , x _ { n } ^ { \star } } ( W ) - a _ { n , x } ( W )$ is at most one. Because $\Delta _ { T } > 0$ , each of the $K - 1$ exponential terms in the sample loss formula is at least $e ^ { - \Delta _ { T } }$ . Therefore,

$$
\ell _ { T , n } ( W ) \geq \log \left( 1 + ( K - 1 ) e ^ { - \Delta _ { T } } \right) = \ell _ { T } ^ { \star } .
$$

Averaging over the samples gives $\mathcal { L } _ { T } ( W ) \geq \ell _ { T } ^ { \star }$

Under Assumption 3, choose a matrix $W ^ { \star }$ satisfying $\langle W ^ { \star } , A _ { n , x } \rangle \geq 1$ for every $( n , x ) \in \mathbb { Z } _ { 2 }$ . For $r \geq 0$ , these inequalities, (B.5), and normalization give

$$
a _ { n , x _ { n } ^ { \star } } ( r W ^ { \star } ) = \left( 1 + \sum _ { x \in \mathcal { X } _ { n } ^ { - } } \frac { a _ { n , x } ( r W ^ { \star } ) } { a _ { n , x _ { n } ^ { \star } } ( r W ^ { \star } ) } \right) ^ { - 1 } \geq \left( 1 + e ^ { - r } \sum _ { x \in \mathcal { X } _ { n } ^ { - } } \frac { c _ { n , x } } { c _ { n , x _ { n } ^ { \star } } } \right) ^ { - 1 } .
$$

The counts are fixed and $c _ { n , x _ { n } ^ { \star } } > 0 .$ , so the lower bound above gives $a _ { n , x _ { n } ^ { \star } } ( r W ^ { \star } ) \to 1 { \mathrm { ~ a s ~ } } r \to \infty$ . Nonnegativity and normalization imply that every other attention weight tends to zero. Continuity of the sample loss formula and the finite training set therefore give

$$
\mathcal { L } _ { T } ( r W ^ { \star } ) = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \ell _ { T , n } ( r W ^ { \star } ) \longrightarrow \ell _ { T } ^ { \star } .
$$

Combining this limit with the lower bound gives ${ \mathrm { i n f } } _ { W \in \mathbb { R } ^ { d \times d } } { \mathcal { L } } _ { T } ( W ) = \ell _ { T } ^ { \star }$

Define the sample excess loss by

$$
E _ { T , n } ( W ) : = \ell _ { T , n } ( W ) - \ell _ { T } ^ { \star } .
$$

The empirical excess loss introduced in the main text satisfies

$$
E _ { T } ( W ) = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } E _ { T , n } ( W ) = \mathcal { L } _ { T } ( W ) - \ell _ { T } ^ { \star } .
$$

The next lemma compares the attention weights and bounds the loss at the start of Phase II.

Lemma B.4. Suppose that Assumptions 1 and 2 hold. At $W = 0 ;$

$$
a _ { n , x _ { n } ^ { \star } } ( 0 ) \geq a _ { n , x } ( 0 ) , \qquad n \in [ N ] , \quad x \in \mathcal { V } \setminus \{ x _ { n } ^ { \star } \} , \quad a n d \quad \mathcal { L } _ { T } ( 0 ) \leq \log K .
$$

Proof. At $W = 0 ,$ , each position receives attention weight $1 / L _ { n } , \ L _ { 0 } a _ { n , x } ( 0 ) = c _ { n , x } / L _ { n }$ for every $x \in \nu$ . Assumption 2 gives $c _ { n , x _ { n } ^ { \star } } \ \geq \ c _ { n , x }$ for every $x \in \mathcal { X } _ { n } ^ { - }$ , and the same inequality holds when $c _ { n , x } = 0$ . Thus, for every $n \in [ N ]$ and $x \in \mathcal { V } \setminus \{ x _ { n } ^ { \star } \}$

$$
a _ { n , x _ { n } ^ { \star } } ( 0 ) - a _ { n , x } ( 0 ) = \frac { c _ { n , x _ { n } ^ { \star } } - c _ { n , x } } { L _ { n } } \geq 0 .
$$

These nonnegative differences and $\Delta _ { T } > 0$ imply that each of the $K - 1$ exponential terms in the sample loss formula of Lemma B.3 is at most one. Therefore,

$$
\ell _ { T , n } ( 0 ) \leq \log K .
$$

Averaging over the samples yields $\mathcal { L } _ { T } ( 0 ) \leq \log K$

With sufficient preprocessing, the correct token’s total attention weight is uniformly bounded away from zero on the loss sublevel set $\mathcal { L } _ { T } ( W ) \leq \log K$

Lemma B.5. Suppose that Assumption 1 holds. For every $W \in \mathbb { R } ^ { d \times d }$ and $n \in [ N ]$

$$
\ell _ { T , n } ( W ) \geq \frac { \Delta _ { T } \left( 1 - K a _ { n , x _ { n } ^ { \star } } ( W ) \right) } { K - 1 } .\tag{B.6}
$$

Let $\delta _ { 0 } : = 1 / ( 2 K ) . I f \Delta _ { T } \geq 2 N ( K - 1 )$ log K, then, for every $W \in \mathbb { R } ^ { d \times d } ,$

$$
{ \mathcal { L } } _ { T } ( W ) \leq \log K \quad \implies \quad a _ { n , x _ { n } ^ { \star } } ( W ) \geq \delta _ { 0 } \qquad ( n \in [ N ] ) .\tag{B.7}
$$

Proof. Fix $W \in \mathbb { R } ^ { d \times d }$ and $n \in [ N ]$ , and let $u : = a _ { n , x _ { n } ^ { \star } } ( W )$ . The other $K - 1$ attention weights sum to $1 - u .$ , so Jensen’s inequality gives

$$
\sum _ { x \in \mathcal { V } \backslash \{ x _ { n } ^ { \star } \} } e ^ { \Delta _ { T } a _ { n , x } ( W ) } \geq ( K - 1 ) \exp \left( \frac { \Delta _ { T } ( 1 - u ) } { K - 1 } \right) .
$$

Using the sample loss formula and discarding the positive term ${ } _ { e ^ { \sum { } x ^ { u } } }$ inside the logarithm, we obtain

$$
\begin{array} { l l } { \displaystyle \ell _ { T , n } ( W ) = \log \left( e ^ { \Delta \tau \tau } + \sum _ { x \in \mathcal { V } \backslash \{ x _ { n } ^ { * } \} } e ^ { \Delta \tau a _ { n , c } ( W ) } \right) - \Delta _ { T } u } \\ { \displaystyle \geq \log \left( \sum _ { x \in \mathcal { V } \backslash \{ x _ { n } ^ { * } \} } e ^ { \Delta \tau a _ { n , c } ( W ) } \right) - \Delta _ { T } u } \\ { \displaystyle \geq \log ( K - 1 ) + \frac { \Delta _ { T } ( 1 - u ) } { K - 1 } - \Delta _ { T } u } \\ { \displaystyle \geq \frac { \Delta _ { T } ( 1 - K u ) } { K - 1 } , } \end{array}
$$

where the last inequality uses $K \geq 2$ and hence $\log ( K - 1 ) \geq 0$ . This proves (B.6).

Now suppose that $\mathcal { L } _ { T } ( W ) \leq \log K$ and $\Delta _ { T } \geq 2 N ( K - 1 )$ log K. Since the empirical loss is the uniform average of nonnegative sample losses,

$$
\ell _ { T , n } ( W ) \leq N \mathcal { L } _ { T } ( W ) \leq N \log K .
$$

Combining this with (B.6) yields

$$
u \geq \frac { 1 } { K } \left( 1 - \frac { N ( K - 1 ) \log K } { \Delta _ { T } } \right) \geq \frac { 1 } { 2 K } = \delta _ { 0 } .
$$

The following corollary translates the logit gap condition in Lemma B.5 into an explicit sufficient condition on $T ,$ Corollary B.1. Suppose that Assumptions 1 and 2 hold. Let the preprocessing iterates be as in Theorem 1, and let

$$
\bar { T } : = \left( \frac { 2 N ( K - 1 ) \log K } { \eta _ { 1 , 0 } } \right) ^ { 2 } .
$$

For every integer $T > \bar { T } ,$

$$
\Delta _ { T } \geq 2 N ( K - 1 ) \log K , \qquad \mathscr { L } _ { T } ( 0 ) \leq \log K ,
$$

and (B.7) holds.

Proof. For $T > \bar { T }$ , the square root step size and the definition of $\bar { T }$ give

$$
\Delta _ { T } = \eta _ { 1 , 0 } \sum _ { k = 1 } ^ { T } k ^ { - 1 / 2 } \geq \eta _ { 1 , 0 } \sqrt { T } \geq 2 N ( K - 1 ) \log K .
$$

Lemma B.4 gives $\mathcal { L } _ { T } ( 0 ) \leq \log K$ . Applying Lemma B.5 then proves (B.7).

Excess loss and gradients. We express the sample excess loss as a function of the attention weights to derive loss bounds and gradient identities.

For each sample $n \in [ N ]$ , define the set of admissible attention vectors by

$$
\mathcal { A } _ { n } : = \left\{ a \in \mathbb { R } ^ { \mathcal { V } } : a _ { x } \geq 0 \left( x \in \mathcal { V } \right) , \sum _ { x \in \mathcal { V } } a _ { x } = 1 , a _ { x } = 0 \mathrm { ~ w h e n e v e r ~ } c _ { n , x } = 0 \right\} .
$$

The attention vector $a _ { n } ( W ) : = \left( a _ { n , x } ( W ) \right) _ { x \in \mathcal { V } }$ belongs to $\mathcal { A } _ { n }$ . For each $x \in \mathcal { V } .$ , let $e _ { x } \in \mathbb { R } ^ { \nu }$ denote the standard basis vector indexed by x. Define the total attention weight assigned to the other tokens in the context and its empirical average by

$$
s _ { n } ( W ) : = 1 - a _ { n , x _ { n } ^ { \star } } ( W ) = \sum _ { x \in \mathcal { X } _ { n } ^ { - } } a _ { n , x } ( W ) , \qquad \bar { s } ( W ) : = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } s _ { n } ( W ) .
$$

For a fixed sample $n \in [ N ]$ , let $c : = x _ { n } ^ { \star }$ . For $a \in \mathbb { R } ^ { \nu }$ and $x \in \mathcal { V } ,$ , define

$$
p _ { n , x } ( a ) : = \frac { e ^ { \Delta _ { T } a _ { x } } } { \sum _ { z \in \mathcal { V } } e ^ { \Delta _ { T } a _ { z } } } , \qquad F _ { T , n } ( a ) : = \log \left( \sum _ { z \in \mathcal { V } } e ^ { \Delta _ { T } a _ { z } } \right) - \Delta _ { T } a _ { c } - \ell _ { T } ^ { \star } .
$$

These formulas define smooth functions on $\mathbb { R } ^ { \nu }$ , which we will evaluate on $\mathcal { A } _ { n }$ . The preceding attention and loss identities give

$$
p _ { n , x } ( W ) = p _ { n , x } ( a _ { n } ( W ) ) , \qquad E _ { T , n } ( W ) = F _ { T , n } ( a _ { n } ( W ) ) .
$$

Assumption 1 ensures $e _ { c } \in A _ { n } .$ , and the definition of $\ell _ { T } ^ { \star }$ gives $F _ { T , n } ( e _ { c } ) = 0$ . For the following estimates, let

$$
Z _ { T } : = e ^ { \Delta _ { T } } + K - 1 , \qquad \kappa _ { T } : = \frac { K \Delta _ { T } } { Z _ { T } } > 0 .
$$

For $x \neq c ,$ let $D _ { n , x } ( a )$ denote the difference between the partial derivatives with respect to $a _ { x }$ and $a _ { c } .$ . For $a \in A _ { n }$ , let $H _ { n } ( a )$ be the sum of these differences weighted by the attention weights:

$$
\begin{array} { r l } & { D _ { n , x } ( a ) : = \langle \nabla _ { a } F _ { T , n } ( a ) , e _ { x } - e _ { c } \rangle = \Delta _ { T } \left( 1 + p _ { n , x } ( a ) - p _ { n , c } ( a ) \right) , \qquad x \neq c , } \\ & { H _ { n } ( a ) : = \displaystyle \sum _ { x \in \mathcal { V } \backslash \{ c \} } a _ { x } D _ { n , x } ( a ) = \langle \nabla _ { a } F _ { T , n } ( a ) , a - e _ { c } \rangle . } \end{array}\tag{B.8}
$$

The inner product expression for $H _ { n } ( a )$ follows from $\textstyle \sum _ { x \in \nu } a _ { x } = 1$ for $a \in A _ { n }$

The next lemma bounds the sample excess loss and $H _ { n } ( a )$ in terms of the total attention weight assigned to the other tokens.   
These bounds will be used in the gradient estimates.

Lemma B.6. For every $n \in [ N ]$ and $a \in A _ { n }$ , let $s _ { n } ( a ) : = 1 - a _ { x _ { n } ^ { \star } }$ . Then

$$
\kappa _ { T } s _ { n } ( a ) \leq F _ { T , n } ( a ) \leq H _ { n } ( a ) \leq 2 \Delta _ { T } s _ { n } ( a ) .\tag{B.9}
$$

Consequently, for every $W \in \mathbb { R } ^ { d \times d }$

$$
\kappa _ { T } \bar { s } ( W ) \leq E _ { T } ( W ) \leq 2 \Delta _ { T } \bar { s } ( W ) .
$$

Proof. Fix $n \in [ N ]$ and $a \in A _ { n }$ , and let $v : = a - e _ { c }$ . The function $F _ { T , n }$ is convex, and its partial derivatives satisfy

$$
\partial _ { x } F _ { T , n } ( a ) = \Delta _ { T } \left( p _ { n , x } ( a ) - \mathbf { 1 } _ { \left\{ x = c \right\} } \right) .
$$

Since $a \in A _ { n }$

$$
v _ { c } = - s _ { n } ( a ) , \qquad \sum _ { x \in \mathcal { V } \backslash \{ c \} } v _ { x } = s _ { n } ( a ) .
$$

At $e _ { c } ,$ the partial derivatives are

$$
\partial _ { c } F _ { T , n } ( e _ { c } ) = - \frac { ( K - 1 ) \Delta _ { T } } { Z _ { T } } , \qquad \partial _ { x } F _ { T , n } ( e _ { c } ) = \frac { \Delta _ { T } } { Z _ { T } } \quad ( x \neq c ) .
$$

Applying convexity at $e _ { c }$ and using $F _ { T , n } ( e _ { c } ) = 0$ , we obtain

$$
F _ { T , n } ( a ) \geq \langle \nabla _ { a } F _ { T , n } ( e _ { c } ) , v \rangle = - \frac { ( K - 1 ) \Delta _ { T } } { Z _ { T } } v _ { c } + \frac { \Delta _ { T } } { Z _ { T } } \sum _ { x \in \mathcal { V } \backslash \left\{ c \right\} } v _ { x } = \frac { K \Delta _ { T } } { Z _ { T } } s _ { n } ( a ) = \kappa _ { T } s _ { n } ( a ) .
$$

Since all softmax probabilities lie in (0, 1), the derivative differences in (B.8) satisfy

$$
0 < D _ { n , x } ( a ) \leq 2 \Delta _ { T } , \qquad x \neq c .
$$

Applying convexity at a gives

$$
0 = F _ { T , n } ( e _ { c } ) \geq F _ { T , n } ( a ) + \langle \nabla _ { a } F _ { T , n } ( a ) , e _ { c } - a \rangle = F _ { T , n } ( a ) - H _ { n } ( a ) .
$$

Summing the upper bounds on $D _ { n , x } ( a )$ with weights $a _ { x }$ gives $H _ { n } ( a ) \leq 2 \Delta _ { T } s _ { n } ( a )$ . This proves (B.9).

Finally, $F _ { T , n } ( a _ { n } ( W ) ) = E _ { T , n } ( W )$ and $s _ { n } ( a _ { n } ( W ) ) = s _ { n } ( W )$ . Evaluating the sample bounds at $a _ { n } ( W )$ and averaging over the N samples proves the empirical bounds. □

The next lemma converts derivatives with respect to the attention weights into a gradient formula for $W$

Lemma B.7. Fix $n \in [ N ]$ , and let $f _ { n }$ be differentiable on an open neighborhood of $A _ { n }$ in $\mathbb { R } ^ { \nu }$ . For $a \in A _ { n }$ and $x \in \mathcal V ,$ , let

$$
d _ { n , x } ( a ) : = \partial _ { x } f _ { n } ( a ) - \sum _ { z \in \mathcal { V } } a _ { z } \partial _ { z } f _ { n } ( a ) .
$$

Let $\varphi _ { n } ( W ) : = f _ { n } ( a _ { n } ( W ) )$ . Then, for every $W \in \mathbb { R } ^ { d \times d }$

$$
- \nabla _ { W } \varphi _ { n } ( W ) = \sum _ { x \in \mathcal { X } _ { n } ^ { - } } a _ { n , x } ( W ) d _ { n , x } ( a _ { n } ( W ) ) A _ { n , x } .
$$

Moreover, for every $a \in A _ { n }$

$$
\begin{array} { r l } { ~ } & { \displaystyle \sum _ { x \in \mathcal { X } _ { n } ^ { - } } a _ { x } d _ { n , x } ( a ) = - a _ { c } d _ { n , c } ( a ) = a _ { c } \nabla _ { a } f _ { n } ( a ) ^ { \top } ( a - e _ { c } ) . } \end{array}
$$

The coefficients $d _ { n , x } ( a )$ need not be nonnegative.

Proof. Let $s _ { n , x } : = x ^ { \top } W q _ { r }$ denote the attention score of token x, and let $a : = a _ { n } ( W )$ . Differentiating the attention formula (B.4) gives

$$
\frac { \partial a _ { n , z } } { \partial s _ { n , x } } = a _ { z } \left( \mathbf { 1 } _ { \left\{ z = x \right\} } - a _ { x } \right) , \qquad x , z \in \mathcal { V } .
$$

The chain rule therefore yields

$$
\frac { \partial \varphi _ { n } } { \partial s _ { n , x } } = \sum _ { z \in \mathcal { V } } \partial _ { z } f _ { n } ( a ) \frac { \partial a _ { n , z } } { \partial s _ { n , x } } = a _ { x } \left( \partial _ { x } f _ { n } ( a ) - \sum _ { z \in \mathcal { V } } a _ { z } \partial _ { z } f _ { n } ( a ) \right) = a _ { x } d _ { n , x } ( a ) .
$$

For every $a \in A _ { n }$ , the definition of $d _ { n , x }$ and $\textstyle \sum _ { x \in \nu } a _ { x } = 1$ imply

$$
\sum _ { x \in \mathcal { V } } a _ { x } d _ { n , x } ( a ) = 0 .
$$

Applying this identity at $a = a _ { n } ( W )$ and using $\nabla _ { W } s _ { n , x } = x q _ { n } ^ { \top }$ , we obtain

$$
\nabla _ { W } \varphi _ { n } ( W ) = \sum _ { x \in \mathcal { V } } a _ { x } d _ { n , x } ( a ) x q _ { n } ^ { \top } = \sum _ { x \in \mathcal { V } } a _ { x } d _ { n , x } ( a ) ( x - c ) q _ { n } ^ { \top } = - \sum _ { x \in \mathcal { X } _ { n } ^ { - } } a _ { x } d _ { n , x } ( a ) A _ { n , x } .
$$

In the last equality, the correct token contributes zero, and tokens absent from the context have zero attention weight. For any $a \in A _ { n }$ , the identity $\begin{array} { r } { \sum _ { x \in \mathcal { V } } a _ { x } d _ { n , x } ( a ) = 0 } \end{array}$ and the zero coordinates of absent tokens give

$$
\sum _ { x \in \mathcal { X } _ { n } ^ { - } } a _ { x } d _ { n , x } ( a ) = - a _ { c } d _ { n , c } ( a ) .
$$

By the definition of $d _ { n , c } ,$

$$
\nabla _ { a } f _ { n } ( a ) ^ { \top } ( a - e _ { c } ) = \sum _ { z \in \mathcal { V } } a _ { z } \partial _ { z } f _ { n } ( a ) - \partial _ { c } f _ { n } ( a ) = - d _ { n , c } ( a ) .
$$

Multiplying this identity by $a _ { c }$ proves the remaining equality.

For the cross entropy loss, we use $D _ { n , x }$ and $H _ { n }$ for gradient lower bounds and introduce centered residuals to contro changes in the gradient. For $a \in A _ { n }$ and $x \in \nu$ , let

$$
r _ { n , x } ( a ) : = p _ { n , x } ( a ) - \mathbf { 1 } _ { \{ x = c \} } , \qquad { \bar { r } } _ { n } ( a ) : = \sum _ { z \in \mathcal { V } } a _ { z } r _ { n , z } ( a ) , \qquad b _ { n , x } ( a ) : = r _ { n , x } ( a ) - { \bar { r } } _ { n } ( a ) .
$$

For $f _ { n } = F _ { T , n }$ , the definition of $d _ { n , x }$ and $\partial _ { x } F _ { T , n } ( a ) = \Delta _ { T } r _ { n , x } ( a )$ give

$$
d _ { n , x } ( a ) = \Delta _ { T } b _ { n , x } ( a ) .\tag{B.10}
$$

We also write $r _ { n , x } ( W ) : = r _ { n , x } ( a _ { n } ( W ) )$ , and use the same convention for $\bar { r } _ { n }$ and $b _ { n , x }$

The following lemma gives the score derivatives and the empirical loss gradient.

Lemma B.8. Fix $W \in \mathbb { R } ^ { d \times d }$ and $n \in [ N ]$ , and let $a : = a _ { n } ( W )$ . Let $g _ { n , x } : = \partial \ell _ { T , n } / \partial s _ { n , x }$ denote the loss derivative with respect to the attention score $s _ { n , x } .$ . Then

$$
g _ { n , c } = - a _ { c } H _ { n } ( a ) , \qquad g _ { n , x } = \Delta _ { T } a _ { x } b _ { n , x } ( a ) , \qquad x \neq c .\tag{B.11}
$$

Consequently,

$$
- \nabla _ { W } \mathcal { L } _ { T } ( W ) = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \sum _ { x \in \mathcal { X } _ { n } ^ { - } } a _ { n , x } ( W ) \left( D _ { n , x } ( a _ { n } ( W ) ) - H _ { n } ( a _ { n } ( W ) ) \right) A _ { n , x } .\tag{B.12}
$$

Proof. The sample loss satisfies $\ell _ { T , n } ( W ) = F _ { T , n } ( a _ { n } ( W ) ) + \ell _ { T } ^ { \star }$ , where $\ell _ { T } ^ { \star }$ is constant. Taking $f _ { n } = F _ { T , n }$ in Lemma B.7 and using (B.10), we obtain

$$
g _ { n , x } = \Delta _ { T } a _ { x } b _ { n , x } ( a ) , \qquad x \in \mathcal { V } .
$$

The definition of $H _ { n }$ and $\partial _ { x } F _ { T , n } ( a ) = \Delta _ { T } r _ { n , x } ( a )$ give

$$
H _ { n } ( a ) = \Delta _ { T } \left( \sum _ { z \in \mathcal { V } } a _ { z } r _ { n , z } ( a ) - r _ { n , c } ( a ) \right) = \Delta _ { T } \left( \bar { r } _ { n } ( a ) - r _ { n , c } ( a ) \right) .
$$

The definitions of $b _ { n , x }$ and $D _ { n , x }$ then imply

$$
\Delta _ { T } b _ { n , c } ( a ) = - H _ { n } ( a ) , \qquad \Delta _ { T } b _ { n , x } ( a ) = D _ { n , x } ( a ) - H _ { n } ( a ) \quad ( x \neq c ) .
$$

Substituting the first identity into the expression for $g _ { n , c }$ proves (B.11).

Finally, substituting $d _ { n , x } ( a ) = D _ { n , x } ( a ) - H _ { n } ( a )$ for $x \neq$ c into the matrix gradient identity in Lemma B.7 gives

$$
- \nabla _ { W } \ell _ { T , n } ( W ) = \sum _ { x \in \mathcal { X } _ { n } ^ { - } } a _ { n , x } ( W ) \left( D _ { n , x } ( a _ { n } ( W ) ) - H _ { n } ( a _ { n } ( W ) ) \right) A _ { n , x } .
$$

Averaging over the N samples proves (B.12).

## GRADIENT AND MOMENTUM ESTIMATES

The following estimates bound the loss gradient and control the error arising from momentum. The attention bounds, convex loss estimates, and scalar momentum bound below will also be used in the Schatten and Adam analyses.

Stability of attention weights. To compare gradients at different iterates, we first control changes in the attention weights under perturbations of W.

Lemma B.9. For any $W , W ^ { \prime } \in \mathbb { R } ^ { d \times d } , l e t r : = \| W ^ { \prime } - W \| _ { \mathrm { o p } } .$ Then, for every n $\in [ N ]$ and every $x \in \nu$ with $c _ { n , x } > 0 _ { ; }$

$$
e ^ { - 2 r } \leq \frac { a _ { n , x } ( W ^ { \prime } ) } { a _ { n , x } ( W ) } \leq e ^ { 2 r } .\tag{B.13}
$$

${ \cal { H } } { \cal { X } } _ { n } ^ { - } \neq \emptyset ,$ then

$$
e ^ { - 2 r } \leq \frac { s _ { n } ( W ^ { \prime } ) } { s _ { n } ( W ) } \leq e ^ { 2 r } .\tag{B.14}
$$

Moreover, for every $n \in [ N ] _ { : }$

$$
\| a _ { n } ( W ^ { \prime } ) - a _ { n } ( W ) \| _ { 1 } \leq 2 \left( e ^ { 2 r } - 1 \right) s _ { n } ( W ) .\tag{B.15}
$$

Proof. Fix $n \in [ N ]$ . The vocabulary embeddings and the query have unit Euclidean norm, so

$$
| x ^ { \top } ( W ^ { \prime } - W ) q _ { n } | \leq \| x \| _ { 2 } \| W ^ { \prime } - W \| _ { \mathrm { o p } } \| q _ { n } \| _ { 2 } = r , \qquad x \in \mathcal { V } .
$$

Let $\begin{array} { r } { Z _ { n } ( W ) : = \sum _ { z \in \mathcal { V } } c _ { n , z } e ^ { z ^ { \top } W q _ { n } } } \end{array}$ . Since the counts are fixed and nonnegative, the score bound gives

$$
e ^ { - r } Z _ { n } ( W ) \leq Z _ { n } ( W ^ { \prime } ) \leq e ^ { r } Z _ { n } ( W ) .
$$

For every $x \in \nu$ with $c _ { n , x } > 0$

$$
{ \frac { a _ { n , x } ( W ^ { \prime } ) } { a _ { n , x } ( W ) } } = e ^ { x ^ { \top } ( W ^ { \prime } - W ) q _ { n } } { \frac { Z _ { n } ( W ) } { Z _ { n } ( W ^ { \prime } ) } } .
$$

Both factors on the right lie in $[ e ^ { - r } , e ^ { r } ]$ , proving (B.13).

Summing these bounds over $x \in \mathcal { X } _ { n } ^ { - }$ gives

$$
e ^ { - 2 r } s _ { n } ( W ) \leq s _ { n } ( W ^ { \prime } ) \leq e ^ { 2 r } s _ { n } ( W ) .
$$

If ${ \mathcal { X } } _ { n } ^ { - } \neq \emptyset$ , then $s _ { n } ( W ) > 0$ , so dividing by $s _ { n } ( W )$ proves (B.14).

By (B.13) and $1 - e ^ { - 2 r } \leq e ^ { 2 r } - 1$ , we have

$$
| a _ { n , x } ( W ^ { \prime } ) - a _ { n , x } ( W ) | \leq \bigl ( e ^ { 2 r } - 1 \bigr ) a _ { n , x } ( W ) , \qquad x \in \mathcal { X } _ { n } ^ { - } .
$$

The correct token has weight $1 - s _ { n } ( W )$ , and tokens absent from the context have zero weight. The triangle inequality therefore gives

$$
\begin{array} { l } { \displaystyle \| a _ { n } ( W ^ { \prime } ) - a _ { n } ( W ) \| _ { 1 } = | s _ { n } ( W ^ { \prime } ) - s _ { n } ( W ) | + \displaystyle \sum _ { x \in \mathcal { X } _ { n } ^ { - } } | a _ { n , x } ( W ^ { \prime } ) - a _ { n , x } ( W ) | } \\ { \displaystyle \qquad \leq 2 \displaystyle \sum _ { x \in \mathcal { X } _ { n } ^ { - } } | a _ { n , x } ( W ^ { \prime } ) - a _ { n , x } ( W ) | } \\ { \displaystyle \qquad \leq 2 \left( e ^ { 2 r } - 1 \right) s _ { n } ( W ) . } \end{array}
$$

This proves (B.15), including the case ${ \mathcal { X } } _ { n } ^ { - } = \emptyset$ , when both sides are zero.

Matrix norms and maximum margins. Muon and Schatten updates subtract the step size times a maximizer of the inner product with the momentum matrix over the corresponding norm unit ball. We use a common matrix norm to state the attention, gradient, and momentum estimates for both methods.

Let N be a norm on $\mathbb { R } ^ { d \times d }$ with dual norm $\mathsf { N } _ { * }$ , satisfying

$$
\| H \| _ { \mathrm { o p } } \leq \mathsf { N } ( H ) \quad ( H \in \mathbb { R } ^ { d \times d } ) , \qquad \mathsf { N } _ { * } ( A _ { n , x } ) = \sqrt { 2 } \quad ( ( n , x ) \in \mathbb { Z } _ { 2 } ) .\tag{B.16}
$$

The first condition allows us to control attention changes using $\mathsf { N } ( W ^ { \prime } - W )$ ; the second condition requires each $A _ { n , x }$ in the gradient expansion to have dual norm ${ \sqrt { 2 } }$

Under Assumption 3 and $\mathcal { T } _ { 2 } \neq \mathcal { O }$ , let $R _ { 2 , \mathsf { N } } ^ { \star }$ denote the optimal value of

$$
\begin{array} { r l } { \underset { W \in \mathbb R ^ { d \times d } } { \operatorname* { m i n } } } & { \mathsf { N } ( W ) } \\ { \mathrm { s u b j e c t ~ t o } } & { \langle W , A _ { n , x } \rangle \geq 1 , \qquad ( n , x ) \in \mathbb Z _ { 2 } . } \end{array}\tag{B.17}
$$

Lemma B.2 supplies a feasible matrix $W ^ { \star }$ . Norm duality and (B.16) imply $\mathsf { N } ( W ) \geq 1 / \sqrt { 2 }$ for every feasible $W .$ , so

$$
\frac { 1 } { \sqrt { 2 } } \leq R _ { 2 , \mathsf { N } } ^ { \star } \leq \mathsf { N } ( W ^ { \star } ) < \infty .
$$

Norm sublevel sets are compact in finite dimensions, so the same argument as in the proof of Lemma B.2 shows that the minimum is attained.

Let $\gamma _ { 2 , \mathsf { N } } : = 1 / R _ { 2 , \mathsf { N } } ^ { \star }$ . Let $W _ { \mathsf { N } } ^ { \star }$ be a minimizer, and let $V _ { \mathsf { N } } ^ { \star } : = W _ { \mathsf { N } } ^ { \star } / R _ { 2 , \mathsf { N } } ^ { \star } .$ Then $\mathsf { N } ( V _ { \mathsf { N } } ^ { \star } ) ~ = ~ 1$ , and feasibility gives $\rho _ { 2 } ( V _ { \mathsf { N } } ^ { \star } ) \geq \gamma _ { 2 , \mathsf { N } }$ . Conversely, if $\mathsf { N } ( V ) \leq 1$ and $\rho _ { 2 } ( V ) > 0$ , then $V / \rho _ { 2 } ( V )$ is feasible, giving

$$
R _ { 2 , \mathsf { N } } ^ { \star } \leq \frac { \mathsf { N } ( V ) } { \rho _ { 2 } ( V ) } \leq \frac { 1 } { \rho _ { 2 } ( V ) } .
$$

Hence $\rho _ { 2 } ( V ) \leq \gamma _ { 2 , \mathsf { N } }$ . This upper bound also holds when $\rho _ { 2 } ( V ) \leq 0 .$ , since $\gamma _ { 2 , \mathsf { N } } > 0$ . Consequently,

$$
{ \sf N } ( V _ { \sf N } ^ { \star } ) = 1 , \qquad \rho _ { 2 } ( V _ { \sf N } ^ { \star } ) = \gamma _ { 2 , \sf N } = \operatorname* { m a x } _ { { \sf N } ( V ) \leq 1 } \rho _ { 2 } ( V ) .\tag{B.18}
$$

For Muon, let ${ \mathsf { N } } = { \parallel { \cdot } } { \parallel _ { \mathrm { o p } } }$ and $\mathsf { N } _ { * } = \parallel \cdot \parallel _ { * }$ , so $R _ { 2 , \mathsf { N } } ^ { \star } = R _ { 2 } ^ { \star }$ and $\gamma _ { 2 , \mathsf { N } } = \gamma _ { 2 }$ . For the Schatten $S _ { p }$ norm with $p \in [ 1 , \infty ]$ , let

$$
\mathsf { N } = \parallel \cdot \parallel _ { S _ { p } } , \qquad \mathsf { N } _ { * } = \parallel \cdot \parallel _ { S _ { p ^ { \prime } } } , \qquad \frac { 1 } { p } + \frac { 1 } { p ^ { \prime } } = 1 ,
$$

with the convention $1 / \infty = 0$ . The first condition in (B.16) follows because the largest singular value is at most the Schatten $S _ { p }$ norm. Each $A _ { n , x }$ has rank one and its only nonzero singular value is ${ \sqrt { 2 } } ,$ so $\| A _ { n , x } \| _ { S _ { p ^ { \prime } } } = \sqrt { 2 }$ . Thus both conditions hold for every $p \in [ 1 , \infty ]$ , including the endpoints.

Gradient lower bounds for convex losses. We state the following gradient lower bounds for convex functions of the attention weights so that they can also be used in the Schatten and Adam extensions. In the Muon analysis, the first bound is used on the initial loss sublevel set to establish loss decrease and summability of the excess losses weighted by the step sizes. Combining this summability with the second bound, whose leading coefficient is $\gamma _ { 2 , \mathsf { N } } ,$ gives the sharper excess loss estimate needed for convergence of the normalized margin.

In the next lemma and its proof, $D _ { n , x }$ and $H _ { n }$ are defined using $F _ { n }$

Lemma B.10. Suppose that Assumption 3 holds, $\mathcal { T } _ { 2 } \neq \mathcal { O }$ , and N satisfies (B.16). For each $n \in [ N ]$ , let $c : = x _ { n } ^ { \star }$ and let $F _ { n }$ be differentiable on an open neighborhood of $A _ { n }$ in $\mathbb { R } ^ { \nu }$ and convex on $A _ { n }$ , with $F _ { n } ( e _ { c } ) = 0 .$ . Let

$$
E _ { F } ( W ) : = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } F _ { n } ( a _ { n } ( W ) ) , \qquad D _ { n , x } ( a ) : = \langle \nabla _ { a } F _ { n } ( a ) , e _ { x } - e _ { c } \rangle ,
$$

and suppose that $D _ { n , x } ( a ) > 0$ for every $n \in [ N ] , a \in { \mathcal { A } } _ { n }$ , and $x \in \mathcal { X } _ { n } ^ { - }$ . There is a constant $c _ { \mathsf { N } } \in ( 0 , \delta _ { 0 } / 2 ]$ , depending only on the training data and N, such that thefollowing estimates hold.

(i) For every $W \in \mathbb { R } ^ { d \times d }$ satisfying $a _ { n , c } ( W ) \geq \delta _ { 0 } .$ for all $n \in [ N ]$

$$
\mathsf { N } _ { * } ( \nabla _ { W } E _ { F } ( W ) ) \geq c _ { \mathsf { N } } E _ { F } ( W ) .
$$

(ii) If there are $L , \kappa > 0$ such that

$$
D _ { n , x } ( a ) \leq L \quad ( x \in \mathcal { X } _ { n } ^ { - } ) , \qquad F _ { n } ( a ) \geq \kappa s _ { n } ( a ) , \qquad n \in [ N ] , \quad a \in \mathcal { A } _ { n } ,
$$

then, for every $W \in \mathbb { R } ^ { d \times d } ,$

$$
{ \sf N } _ { * } ( \nabla _ { W } E _ { F } ( W ) ) \geq \gamma _ { 2 , \sf N } E _ { F } ( W ) - \frac { \sqrt { 2 } N L } { \kappa ^ { 2 } } E _ { F } ( W ) ^ { 2 } .
$$

Proof. To prove (i), we construct an auxiliary matrix using the token ordering for each query. Geometrically increasing weights make the correct token’s positive contribution dominate the possible negative contributions under the assumed attention lower bound.

For each $q \in \mathcal V$ with $\mathcal { C } _ { q } \neq \emptyset$ , the ordering in Lemma B.2 gives indices $r _ { q } : \mathcal { V } \to \{ 0 , \dots , K - 1 \}$ satisfying

$$
r _ { q } ( x _ { n } ^ { \star } ) \geq r _ { q } ( x ) + 1 , \qquad ( n , x ) \in \mathcal { T } _ { 2 } , \quad q _ { n } = q .
$$

Let

$$
\Lambda : = 4 K = \frac { 2 } { \delta _ { 0 } } , \qquad \widetilde { W } : = \sum _ { \stackrel { q \in \mathcal { V } } { C _ { q } \neq \mathcal { O } } } \left( \sum _ { x \in \mathcal { V } } \Lambda ^ { r _ { q } ( x ) } x \right) q ^ { \top } .
$$

Orthonormality gives

$$
\| \widetilde W \| _ { F } ^ { 2 } = \sum _ { \stackrel { q \in \mathcal { V } } { c _ { q } \neq \mathcal { O } } } \sum _ { x \in \mathcal { V } } \Lambda ^ { 2 r _ { q } ( x ) } \leq K ^ { 2 } \Lambda ^ { 2 ( K - 1 ) } .
$$

Since $\mathcal { T } _ { 2 } \neq \mathcal { O }$ , there is a query q with $\mathcal { C } _ { q } \neq \emptyset$ . For this query and any $x \in \nu$

$$
1 \leq \Lambda ^ { r _ { q } ( x ) } = | x ^ { \top } \widetilde { W } q | \leq \| \widetilde { W } \| _ { \mathrm { o p } } \leq \mathsf { N } ( \widetilde { W } ) .
$$

We may therefore take

$$
c _ { \mathsf { N } } : = \frac { \delta _ { 0 } } { 2 \mathsf { N } ( \widetilde { W } ) } \in ( 0 , \delta _ { 0 } / 2 ] .
$$

Fix W satisfying the attention lower bound in (i). For a sample n with ${ \mathcal { X } } _ { n } ^ { - } \neq \emptyset$ , let $a : = a _ { n } ( W ) , w _ { x } : = \Lambda ^ { r _ { q _ { n } } ( x ) }$ , and

$$
H _ { n } : = \sum _ { x \in \mathcal { X } _ { n } ^ { - } } a _ { x } D _ { n , x } ( a ) .
$$

Positivity of $D _ { n , x } ( a )$ gives $H _ { n } \geq 0$ , while convexity and $F _ { n } ( e _ { c } ) = 0$ give

$$
F _ { n } ( a ) \leq \left. \nabla _ { a } F _ { n } ( a ) , a - e _ { c } \right. = H _ { n } .
$$

The derivatives in Lemma B.7 satisfy $d _ { n , c } ( a ) = - H _ { n } { \mathrm { ~ a n d ~ } } d _ { n , x } ( a ) = D _ { n , x } ( a ) - H _ { n } { \mathrm { ~ f o r ~ } } x \in \chi _ { n } ^ { - }$ . Using that lemma and orthonormality, we obtain

$$
\begin{array} { r l } { \langle - \nabla _ { W } F _ { n } ( a _ { n } ( W ) ) , \widetilde { W } \rangle = a _ { c } w _ { c } H _ { n } + } & { \displaystyle \sum _ { x \in { \mathcal X } _ { n } } a _ { x } \left( H _ { n } - D _ { n , x } ( a ) \right) w _ { x } } \\ & { \ge a _ { c } w _ { c } H _ { n } - \displaystyle \frac { w _ { c } } { \Lambda } \displaystyle \sum _ { x \in { \mathcal X } _ { n } ^ { - } } a _ { x } D _ { n , x } ( a ) } \\ & { = w _ { c } \left( a _ { c } - \Lambda ^ { - 1 } \right) H _ { n } } \\ & { \ge \displaystyle \frac { \delta _ { 0 } } { 2 } H _ { n } } \\ & { \ge \displaystyle \frac { \delta _ { 0 } } { 2 } F _ { n } ( a ) . } \end{array}
$$

Here we used $0 < w _ { x } \le w _ { c } / \Lambda$ for $x \in \mathcal { X } _ { n } ^ { - } , w _ { c } \geq 1 , a _ { c } \geq \delta _ { 0 }$ , and $\Lambda ^ { - 1 } = \delta _ { 0 } / 2$ . If ${ \mathcal { X } } _ { n } ^ { - } = { \mathcal { O } }$ , then $a _ { n } ( W ) = e _ { c }$ for every $W$ , so both $F _ { n } ( a _ { n } ( W ) )$ and $\nabla _ { W } F _ { n } ( a _ { n } ( W ) )$ vanish. Averaging over the samples and applying norm duality therefore give

$$
\frac { \delta _ { 0 } } { 2 } E _ { F } ( W ) \leq \langle - \nabla _ { W } E _ { F } ( W ) , \widetilde { W } \rangle \leq \mathsf { N } _ { * } ( \nabla _ { W } E _ { F } ( W ) ) \mathsf { N } ( \widetilde { W } ) .
$$

The definition of c<sub>N</sub> proves (i).

Next, assume the additional conditions in (ii), and fix an arbitrary $W \in \mathbb { R } ^ { d \times d }$ . Let $V _ { \mathsf { N } } ^ { \star }$ be the matrix in (B.18), and let $m _ { n , x } : = \langle V _ { \mathsf { N } } ^ { \star } , A _ { n , x } \rangle$ . Equation (B.18) and norm duality give

$$
\gamma _ { 2 , \mathsf { N } } \leq m _ { n , x } \leq \sqrt { 2 } , \qquad ( n , x ) \in { \mathcal { T } } _ { 2 } .
$$

For each sample, let $a : = a _ { n } ( W )$ and $\begin{array} { r } { H _ { n } : = \sum _ { x \in \mathcal { X } _ { n } ^ { - } } a _ { x } D _ { n , x } ( a ) } \end{array}$ . The assumptions and convexity imply

$$
0 \leq F _ { n } ( a ) \leq H _ { n } \leq L s _ { n } ( W ) .
$$

Lemma B.7 yields

$$
\begin{array} { l } { \displaystyle \langle - \nabla _ { W } F _ { n } ( a _ { n } ( W ) ) , V _ { \mathsf { N } } ^ { \star } \rangle = \displaystyle \sum _ { x \in \mathcal { X } _ { n } ^ { - } } a _ { x } D _ { n , x } ( a ) m _ { n , x } - H _ { n } \displaystyle \sum _ { x \in \mathcal { X } _ { n } ^ { - } } a _ { x } m _ { n , x } } \\ { \ge \gamma _ { 2 , \mathsf { N } } H _ { n } - \sqrt { 2 } s _ { n } ( W ) H _ { n } } \\ { \ge \gamma _ { 2 , \mathsf { N } } F _ { n } ( a ) - \sqrt { 2 } L s _ { n } ( W ) ^ { 2 } . } \end{array}
$$

The last inequality uses $H _ { n } \geq F _ { n } ( a )$ in the positive term and $H _ { n } \leq L s _ { n } ( W )$ in the negative term. Moreover, $F _ { n } ( a _ { n } ( W ) ) \geq$ $\kappa s _ { n } ( W ) \geq 0$ , so

$$
\frac { 1 } { N } \sum _ { n = 1 } ^ { N } s _ { n } ( W ) ^ { 2 } \leq \frac { 1 } { N \kappa ^ { 2 } } \sum _ { n = 1 } ^ { N } F _ { n } ( a _ { n } ( W ) ) ^ { 2 } \leq \frac { 1 } { N \kappa ^ { 2 } } \left( \sum _ { n = 1 } ^ { N } F _ { n } ( a _ { n } ( W ) ) \right) ^ { 2 } = \frac { N } { \kappa ^ { 2 } } E _ { F } ( W ) ^ { 2 } .
$$

Averaging the sample estimates and using $\mathsf { N } ( V _ { \mathsf { N } } ^ { \star } ) = 1$ give

$$
\begin{array} { r l } {  { \mathsf { N } _ { * } ( \nabla _ { W } E _ { F } ( W ) ) \geq \langle - \nabla _ { W } E _ { F } ( W ) , V _ { \mathsf { N } } ^ { \star } \rangle } } \\ & { \geq \gamma _ { 2 , \mathsf { N } } E _ { F } ( W ) - \frac { \sqrt { 2 } L } { N } \sum _ { n = 1 } ^ { N } s _ { n } ( W ) ^ { 2 } } \\ & { \geq \gamma _ { 2 , \mathsf { N } } E _ { F } ( W ) - \frac { \sqrt { 2 } N L } { \kappa ^ { 2 } } E _ { F } ( W ) ^ { 2 } . } \end{array}
$$

This proves (ii).

Under the additional conditions in (ii), $E _ { F } ( W ) \geq 0$ . Combining (i) and (ii), for every W satisfying $a _ { n , c } ( W ) \geq \delta _ { 0 }$ for all $n \in [ N ]$ , we obtain

$$
\mathsf { N } _ { \ast } ( \nabla _ { W } E _ { F } ( W ) ) \geq E _ { F } ( W ) \operatorname* { m a x } \left\{ c _ { \mathsf { N } } , \gamma _ { 2 , \mathsf { N } } - \frac { \sqrt { 2 } N L } { \kappa ^ { 2 } } E _ { F } ( W ) \right\} .\tag{B.19}
$$

For $F _ { n } ~ = ~ F _ { T , n }$ , Lemma B.6 and (B.8) verify the hypotheses of Lemma B.10 with $L \ = \ 2 \Delta _ { T }$ and $\begin{array} { r } { \kappa \ = \ \kappa _ { T } } \end{array}$ . Let $C _ { T } : = 2 \sqrt { 2 } N \Delta _ { T } / \kappa _ { T } ^ { 2 }$ . Since $E _ { F } = E _ { T }$ and $\nabla _ { W } E _ { T } = \nabla _ { W } \mathcal { L } _ { T }$ , the combined bound (B.19) gives

$$
\mathsf { N } _ { * } ( \nabla _ { W } \mathcal { L } _ { T } ( W ) ) \ge E _ { T } ( W ) \operatorname* { m a x } \{ c _ { \mathsf { N } } , \gamma _ { 2 , \mathsf { N } } - C _ { T } E _ { T } ( W ) \}\tag{B.20}
$$

for every $W \in \mathbb { R } ^ { d \times d }$ satisfying $a _ { n , x _ { n } ^ { \star } } ( W ) \geq \delta _ { 0 }$ for all $n \in [ N ]$

We next bound the change in the gradient under perturbations of W. Keeping the factor s¯(W) allows us to control the momentum error and the Taylor remainder by the excess loss using Lemma B.6.

Lemma B.11. Let N satisfy (B.16), and let

$$
L _ { 1 , T } : = 2 \sqrt { 2 } \Delta _ { T } ( 2 + \Delta _ { T } ) .
$$

For every $W , W ^ { \prime } \in \mathbb { R } ^ { d \times d }$ , let $d _ { W } : = \mathsf { N } ( W ^ { \prime } - W )$ . Then

$$
\mathsf { N } _ { * } ( \nabla _ { W } \mathcal { L } _ { T } ( W ^ { \prime } ) - \nabla _ { W } \mathcal { L } _ { T } ( W ) ) \le L _ { 1 , T } ( e ^ { 2 d _ { W } } - 1 ) \bar { s } ( W ) .\tag{B.21}
$$

The constant $L _ { 1 , T }$ depends only on $\Delta _ { T }$ and is independent of the initial step size in Phase II.

Proof. Fix $n \in [ N ]$ , and let $a : = a _ { n } ( W )$ and $a ^ { \prime } : = a _ { n } ( W ^ { \prime } )$ . For this sample, we omit the index n in $p , r , { \bar { r } } ,$ and $b ,$ and use primes to indicate evaluation at $a ^ { \prime }$ . The residuals lie in $[ - 1 , 1 ]$ , and r¯ is their average with attention weights. Hence $| b _ { x } | \leq 2$ and $| b _ { x } ^ { \prime } | \leq 2$ for every $x \in \nu$

Since $\| W ^ { \prime } - W \| _ { \mathrm { o p } } \leq d _ { W }$ , Lemma B.9 gives

$$
\| a ^ { \prime } - a \| _ { 1 } \leq 2 ( e ^ { 2 d _ { W } } - 1 ) s _ { n } ( W ) .
$$

The Jacobian of $p ( a )$ is $\Delta _ { T } ( \mathrm { d i a g } ( p ) - p p ^ { \top } )$ . For each $z \in \mathcal { V } ,$ , the sum of the absolute values in column z satisfies

$$
\sum _ { x \in \mathcal { V } } \left| \frac { \partial p _ { x } } { \partial a _ { z } } \right| = 2 \Delta _ { T } p _ { z } ( 1 - p _ { z } ) \leq \frac { \Delta _ { T } } { 2 } ,
$$

where we used $p _ { z } ( 1 - p _ { z } ) \leq 1 / 4$ . Integrating along the segment from a to $a ^ { \prime }$ gives

$$
\| p ^ { \prime } - p \| _ { 1 } \leq \frac { \Delta _ { T } } { 2 } \| a ^ { \prime } - a \| _ { 1 } .
$$

Using $r ^ { \prime } - r = p ^ { \prime } - p , \| r ^ { \prime } \| _ { \infty } \leq 1$ , and $\| a \| _ { 1 } = 1$ , we obtain

$$
| \bar { r } ^ { \prime } - \bar { r } | = | ( a ^ { \prime } - a ) ^ { \top } r ^ { \prime } + a ^ { \top } ( r ^ { \prime } - r ) | \leq \| a ^ { \prime } - a \| _ { 1 } + \| p ^ { \prime } - p \| _ { \infty } .
$$

Consequently, for every $x \in \nu$

$$
\begin{array} { r l } & { | b _ { x } ^ { \prime } - b _ { x } | \leq | p _ { x } ^ { \prime } - p _ { x } | + | \bar { r } ^ { \prime } - \bar { r } | } \\ & { \qquad \leq \| a ^ { \prime } - a \| _ { 1 } + 2 \| p ^ { \prime } - p \| _ { 1 } } \\ & { \qquad \leq ( 1 + \Delta _ { T } ) \| a ^ { \prime } - a \| _ { 1 } . } \end{array}
$$

For $x \in \mathcal { X } _ { n } ^ { - }$ , the score derivative in (B.11) is $g _ { n , x } ( W ) = \Delta _ { T } a _ { n , x } ( W ) b _ { n , x } ( W )$ . Its change can be written as

$$
g _ { n , x } ( W ^ { \prime } ) - g _ { n , x } ( W ) = \Delta _ { T } \left( ( a _ { x } ^ { \prime } - a _ { x } ) b _ { x } ^ { \prime } + a _ { x } ( b _ { x } ^ { \prime } - b _ { x } ) \right) .
$$

The attention ratio bound in Lemma B.9 gives

$$
| a _ { x } ^ { \prime } - a _ { x } | \leq ( e ^ { 2 d _ { W } } - 1 ) a _ { x } , \qquad x \in \mathcal { X } _ { n } ^ { - } .
$$

Combining these estimates with $| b _ { x } ^ { \prime } | \leq 2$ and $\textstyle \sum _ { x \in { \mathcal { X } } _ { n } ^ { - } } a _ { x } = s _ { n } ( W )$ , we obtain

$$
\begin{array} { l } { \displaystyle \sum _ { x \in \mathcal { X } _ { n } ^ { - } } | g _ { n , x } ( W ^ { \prime } ) - g _ { n , x } ( W ) | \leq 2 \Delta _ { T } ( e ^ { 2 d _ { W } } - 1 ) s _ { n } ( W ) + \Delta _ { T } ( 1 + \Delta _ { T } ) s _ { n } ( W ) \| a ^ { \prime } - a \| _ { 1 } } \\ { \displaystyle \qquad \leq 2 \Delta _ { T } ( e ^ { 2 d _ { W } } - 1 ) s _ { n } ( W ) + 2 \Delta _ { T } ( 1 + \Delta _ { T } ) ( e ^ { 2 d _ { W } } - 1 ) s _ { n } ( W ) ^ { 2 } } \\ { \displaystyle \qquad \leq 2 \Delta _ { T } ( 2 + \Delta _ { T } ) ( e ^ { 2 d _ { W } } - 1 ) s _ { n } ( W ) . } \end{array}
$$

The last inequality uses $0 \leq s _ { n } ( W ) \leq 1$

Finally, the gradient representation in Lemma B.8, the dual norm triangle inequality, and $\mathsf { N } _ { * } ( A _ { n , x } ) = \sqrt { 2 }$ give

$$
\begin{array} { r l } {  { \boldsymbol { \Lambda } _ { * } ( \nabla _ { W } \mathcal { L } _ { T } ( W ^ { \prime } ) - \nabla _ { W } \mathcal { L } _ { T } ( W ) ) \le \frac { \sqrt { 2 } } { N } \displaystyle \sum _ { n = 1 } ^ { N } \displaystyle \sum _ { x \in \mathcal { X } _ { n } ^ { - } } \vert g _ { n , x } ( W ^ { \prime } ) - g _ { n , x } ( W ) \vert } \quad } & { } \\ & { \le \frac { 2 \sqrt { 2 } \Delta _ { T } ( 2 + \Delta _ { T } ) } { N } ( e ^ { 2 d _ { W } } - 1 ) \displaystyle \sum _ { n = 1 } ^ { N } s _ { n } ( W ) } \\ & { = L _ { 1 , T } ( e ^ { 2 d _ { W } } - 1 ) \bar { s } ( W ) . } \end{array}
$$

Momentum and gradient alignment. To analyze Muon and the Schatten updates together, we write $W _ { t } , M _ { t } , \eta _ { t }$ , and $\beta$ for the attention matrix, momentum matrix, step size, and momentum parameter. For Phase II of Algorithm 1, these are $W _ { 2 , t } , M _ { 2 , t } , \eta _ { 2 , t }$ , and $\beta _ { 2 }$ , respectively. Let N satisfy (B.16), and consider the updates

$$
\begin{array} { r l } & { M _ { t } = \beta M _ { t - 1 } + ( 1 - \beta ) \nabla _ { W } \mathcal { L } _ { T } ( W _ { t } ) , } \\ & { D _ { t } \in \underset { \mathbb { N } ( D ) \leq 1 } { \mathrm { a r g m a x } } \langle M _ { t } , D \rangle , } \\ & { W _ { t + 1 } = W _ { t } - \eta _ { t } D _ { t } , } \end{array}\tag{B.22}
$$

where $0 \leq \beta < 1 , M _ { - 1 } = 0$ , and $D _ { t } = 0$ when $M _ { t } = 0 .$ . For ${ \mathsf { N } } = { \parallel { \cdot } } { \parallel _ { \mathrm { o p } } }$ and $M _ { t } \neq 0$ , Muon chooses $D _ { t } = U _ { t } V _ { t } ^ { \top }$ , where $M _ { t } = U _ { t } \Sigma _ { t } V _ { t } ^ { \top }$ is the compact singular value decomposition.

Let

$$
\widehat { M } _ { t } : = \frac { M _ { t } } { 1 - \beta ^ { t + 1 } } , \qquad \Omega _ { t } : = \widehat { M } _ { t } - \nabla _ { W } \mathcal { L } _ { T } ( W _ { t } ) , \qquad E _ { t } : = E _ { T } ( W _ { t } ) .
$$

Expanding the momentum recurrence from $M _ { - 1 } = 0$ gives

$$
\widehat { M } _ { t } = \frac { 1 - \beta } { 1 - \beta ^ { t + 1 } } \sum _ { r = 0 } ^ { t } \beta ^ { r } \nabla _ { W } \mathcal { L } _ { T } ( W _ { t - r } ) .
$$

The coefficients are nonnegative and sum to one. When $\beta = 0$ , this average reduces to the current gradient, so $\Omega _ { t } = 0$ Since $1 - \beta ^ { t + 1 } > 0$ , the matrices $M _ { t }$ and $\widehat { M _ { t } }$ have the same maximizing directions in (B.22).

In this notation, the cumulative sums defined in the main text are

$$
S _ { t } : = \sum _ { r = 0 } ^ { t - 1 } \eta _ { r } , \qquad Q _ { t } : = \sum _ { r = 0 } ^ { t - 1 } \eta _ { r } ^ { 2 } , \qquad S _ { 0 } = Q _ { 0 } = 0 .
$$

The next lemma bounds the exponential terms that arise when Lemma B.11 is applied to past iterates. Lemma C.1 of Zhang et al. (2024) gives a related estimate for power law step sizes with exponent in $( 0 , 1 ]$ . We give a proof for all $\alpha > 0$ and $t \geq 0 ,$ , with a constant independent of the initial step size.

Lemma B.12. Let $\eta _ { t } : = \eta _ { 0 } ( t + 1 ) ^ { - \alpha }$ , where α $> 0$ and $0 < \eta _ { 0 } \leq 1$ , and let $c > 0$ and $\beta \in [ 0 , 1 )$ . There is a finite constant $C ( \alpha , c , \beta )$ , depending only on $\alpha , c , \beta ,$ , such that,for every integer $t \geq 0 ,$

$$
\frac { 1 - \beta } { 1 - \beta ^ { t + 1 } } \sum _ { r = 0 } ^ { t } \beta ^ { r } \left[ \exp \left( c \sum _ { j = 1 } ^ { r } \eta _ { t - j } \right) - 1 \right] \leq C ( \alpha , c , \beta ) \eta _ { t } .
$$

An empty sum is zero. When $\beta = 0$ , the sum contains only the term with $r = 0 ,$ , and we take $C ( \alpha , c , 0 ) : = 0$

Proof. $\operatorname { I f } t = 0 \mathrm { o r } \beta = 0$ , the left hand side is zero. Suppose that $0 < \beta < 1$ and $t \geq 1$ . Let

$$
P _ { t } : = \sum _ { k = 1 } ^ { t } k ^ { - \alpha } , \qquad U _ { t , r } : = \sum _ { k = t - r + 1 } ^ { t } k ^ { - \alpha } , \qquad U _ { t , 0 } : = 0 .
$$

Then $\begin{array} { r } { \sum _ { j = 1 } ^ { r } \eta _ { t - j } = \eta _ { 0 } U _ { t , r } } \end{array}$ . For $1 \leq r \leq t ,$ , we have

$$
U _ { t , r } \leq \frac { r P _ { t } } { t } , \qquad U _ { t , r } \leq \frac { r ( r + 1 ) ^ { \alpha } } { ( t + 1 ) ^ { \alpha } } .
$$

The first bound follows because the average of the last r terms of the decreasing sequence $( k ^ { - \alpha } ) _ { k = 1 } ^ { t }$ is at most the average of all t terms. The second follows from $U _ { t , r } \leq r ( t - r + 1 ) ^ { - \alpha }$ and $( t + 1 ) / ( t - r + 1 ) \leq r + 1$

To control the exponential factor, note that $P _ { t } / t \to 0$ because $k ^ { - \alpha } \to 0$ . Choose an integer $\tau \geq 1$ , depending only on $\alpha , c , \beta .$ such that $c P _ { t } / t \leq - ( \log \beta ) / 2$ for all $t \geq \tau$ . For $t \geq \tau$ , the first bound on $U _ { t , r }$ gives $\beta ^ { r } e ^ { c U _ { t , r } } \le \beta ^ { r / 2 }$ . For $1 \leq t < \tau$ , we use $U _ { t , r } \leq P _ { \tau }$ and $\beta ^ { r } \le \beta ^ { r / 2 }$ . Thus

$$
\begin{array} { r } { \beta ^ { r } e ^ { c U _ { t , r } } \leq e ^ { c P _ { \tau } } \beta ^ { r / 2 } , \qquad t \geq 1 , \quad 0 \leq r \leq t . } \end{array}
$$

Let

$$
C ( \alpha , c , \beta ) : = c e ^ { c P _ { \tau } } \sum _ { r = 1 } ^ { \infty } r ( r + 1 ) ^ { \alpha } \beta ^ { r / 2 } .
$$

The series converges because $0 < \beta < 1$ , so this constant is finite and depends only on $\alpha , c , \beta$

Convexity of the exponential and $e ^ { v } - 1 \leq v e ^ { v }$ for $v \geq 0$ give

$$
e ^ { c \eta _ { 0 } U } - 1 \le \eta _ { 0 } ( e ^ { c U } - 1 ) \le c \eta _ { 0 } U e ^ { c U } , \qquad U \ge 0 , \quad 0 < \eta _ { 0 } \le 1 .
$$

Using $( 1 - \beta ) / ( 1 - \beta ^ { t + 1 } ) \le 1$ and the preceding bounds, we obtain

$$
\begin{array} { c } { \displaystyle \frac { 1 - \beta } { 1 - \beta ^ { t + 1 } } \sum _ { r = 0 } ^ { t } \beta ^ { r } \left[ \exp \left( c \sum _ { j = 1 } ^ { r } \eta _ { t - j } \right) - 1 \right] \leq c \eta _ { 0 } \displaystyle \sum _ { r = 1 } ^ { t } \beta ^ { r } U _ { t , r } e ^ { c U _ { t , r } } } \\ { \leq \frac { c \eta _ { 0 } e ^ { c P _ { \tau } } } { ( t + 1 ) ^ { \alpha } } \displaystyle \sum _ { r = 1 } ^ { \infty } r ( r + 1 ) ^ { \alpha } \beta ^ { r / 2 } } \\ { = c ( \alpha , c , \beta ) \eta _ { t } . } \end{array}
$$

We now combine Lemmas B.11 and B.12 to bound the momentum error by a multiple of $\eta _ { t } \bar { s } ( W _ { t } )$

Lemma B.13. Let N satisfy (B.16), and consider the updates in (B.22) with $\eta _ { t } = \eta _ { 0 } ( t { + } 1 ) ^ { - \alpha }$ , where $\alpha > 0 a n d 0 < \eta _ { 0 } \leq 1$ Let

$$
C _ { M , T , \alpha } : = L _ { 1 , T } C ( \alpha , 2 , \beta ) .
$$

Then, for every integer $t \geq 0 ,$

$$
\mathsf { N } _ { \ast } ( \Omega _ { t } ) \leq C _ { M , T , \alpha } \eta _ { t } \bar { s } ( W _ { t } ) .
$$

The constant $C _ { M , T , \alpha }$ depends only on $\Delta _ { T } , \alpha , \beta$ and is independent of η<sub>0</sub> and t.

Proof. If t = 0 or $\beta = 0$ , then $\Omega _ { t } = 0 .$ , so the bound holds. Suppose that $t \geq 1$ and $0 < \beta < 1$ . Let

$$
w _ { t , r } : = \frac { ( 1 - \beta ) \beta ^ { r } } { 1 - \beta ^ { t + 1 } } , \qquad 0 \leq r \leq t .
$$

These weights are nonnegative and satisfy $\scriptstyle \sum _ { r = 0 } ^ { t } w _ { t , r } = 1$ . Using the expansion of $\widehat { M _ { t } }$ and $\scriptstyle \sum _ { r = 0 } ^ { t } w _ { t , r } = 1$ , we obtain

$$
\Omega _ { t } = \sum _ { r = 0 } ^ { t } w _ { t , r } \left( \nabla _ { W } \mathcal { L } _ { T } ( W _ { t - r } ) - \nabla _ { W } \mathcal { L } _ { T } ( W _ { t } ) \right) .
$$

For $0 \leq r \leq t$ , summing the updates over $s = t - r , \ldots , t - 1$ and using $\mathsf { N } ( D _ { s } ) \leq 1$ gives

$$
\mathsf { N } ( W _ { t } - W _ { t - r } ) \leq \sum _ { s = t - r } ^ { t - 1 } \eta _ { s } \mathsf { N } ( D _ { s } ) \leq \sum _ { j = 1 } ^ { r } \eta _ { t - j } .
$$

We apply Lemma B.11 with $W = W _ { t }$ and $W ^ { \prime } = W _ { t - r }$ so that $\bar { s } ( W _ { t } )$ appears as a factor in the bound:

$$
\mathsf { N } _ { * } ( \nabla _ { W } \mathcal { L } _ { T } ( W _ { t - r } ) - \nabla _ { W } \mathcal { L } _ { T } ( W _ { t } ) ) \le L _ { 1 , T } \left[ \exp \left( 2 \sum _ { j = 1 } ^ { r } \eta _ { t - j } \right) - 1 \right] \bar { s } ( W _ { t } ) .
$$

The dual norm triangle inequality and Lemma B.12 with $c = 2$ now give

$$
\begin{array} { l } { \displaystyle \mathsf { N } _ { * } ( \Omega _ { t } ) \leq \sum _ { r = 0 } ^ { t } w _ { t , r } \mathsf { N } _ { * } \big ( \nabla _ { W } \mathcal { L } _ { T } ( W _ { t - r } ) - \nabla _ { W } \mathcal { L } _ { T } ( W _ { t } ) \big ) } \\ { \displaystyle \qquad \leq L _ { 1 , T } \bar { s } ( W _ { t } ) \sum _ { r = 0 } ^ { t } w _ { t , r } \left[ \exp \left( 2 \sum _ { j = 1 } ^ { r } \eta _ { t - j } \right) - 1 \right] } \\ { \displaystyle \qquad \leq L _ { 1 , T } C ( \alpha , 2 , \beta ) \eta _ { t } \bar { s } ( W _ { t } ) } \\ { \displaystyle \qquad = C _ { M , T , \alpha } \eta _ { t } \bar { s } ( W _ { t } ) . } \end{array}
$$

## DESCENT AND MARGIN CONVERGENCE

Loss decrease on the initial loss sublevel set. We first use the momentum error bound to bound the inner product of the current gradient with the update direction from below. We then combine this bound with the gradient lower bounds and the Taylor remainder estimate to prove loss decrease for a sufficiently small initial step size.

Lemma B.14. Under the assumptions of Lemma B.13, every maximizing direction $D _ { t }$ in (B.22) satisfies

$$
\langle \nabla _ { W } \mathcal { L } _ { T } ( W _ { t } ) , D _ { t } \rangle \ge \mathsf { N } _ { * } ( \nabla _ { W } \mathcal { L } _ { T } ( W _ { t } ) ) - 2 C _ { M , T , \alpha } \eta _ { t } \bar { s } ( W _ { t } )
$$

for every integer $t \geq 0 .$

Proof. Since $D _ { t }$ maximizes the inner product with $\widehat { M _ { t } }$ over the unit ball of N, we have $\langle \widehat { M } _ { t } , D _ { t } \rangle = \mathsf { N } _ { * } ( \widehat { M } _ { t } )$ and $\mathsf { N } ( D _ { t } ) \leq 1$ Using $\nabla _ { W } \mathcal { L } _ { T } ( W _ { t } ) = \widehat { M } _ { t } - \Omega _ { t }$ , norm duality, and the reverse triangle inequality, we obtain

$$
\begin{array} { r l } & { \langle \nabla _ { W } \mathcal { L } _ { T } ( W _ { t } ) , D _ { t } \rangle = \mathsf { N } _ { * } ( \widehat { M } _ { t } ) - \langle \Omega _ { t } , D _ { t } \rangle } \\ & { \qquad \ge \mathsf { N } _ { * } ( \widehat { M } _ { t } ) - \mathsf { N } _ { * } ( \Omega _ { t } ) } \\ & { \qquad \ge \mathsf { N } _ { * } ( \nabla _ { W } \mathcal { L } _ { T } ( W _ { t } ) ) - 2 \mathsf { N } _ { * } ( \Omega _ { t } ) . } \end{array}
$$

Applying Lemma B.13 to the last term proves the result.

The following estimate bounds the remainder in the first order expansion of the loss by a multiple of $\eta ^ { 2 } \bar { s } ( W )$ .

Lemma B.15. Let N satisfy (B.16), and let

$$
C _ { Q , T } : = e ^ { 2 } L _ { 1 , T } .
$$

For every W, $D \in \mathbb { R } ^ { d \times d }$ with $\mathsf { N } ( D ) \leq 1$ and every $\eta \in [ 0 , 1 ]$ ],

$$
\mathcal { L } _ { T } ( W - \eta D ) \leq \mathcal { L } _ { T } ( W ) - \eta \langle \nabla _ { W } \mathcal { L } _ { T } ( W ) , D \rangle + C _ { Q , T } \eta ^ { 2 } \bar { s } ( W ) .
$$

Proof. The fundamental theorem of calculus gives

$$
\mathcal { L } _ { T } ( W - \eta D ) - \mathcal { L } _ { T } ( W ) + \eta \langle \nabla _ { W } \mathcal { L } _ { T } ( W ) , D \rangle = - \int _ { 0 } ^ { \eta } \langle \nabla _ { W } \mathcal { L } _ { T } ( W - \tau D ) - \nabla _ { W } \mathcal { L } _ { T } ( W ) , D \rangle d \tau .
$$

Since $\mathsf { N } ( \tau D ) \le \tau$ and $\mathsf { N } ( D ) \leq 1$ , norm duality and Lemma B.11 bound the absolute value of the remainder by

$$
L _ { 1 , T } \bar { s } ( W ) \int _ { 0 } ^ { \eta } ( e ^ { 2 \tau } - 1 ) d \tau \le e ^ { 2 } L _ { 1 , T } \eta ^ { 2 } \bar { s } ( W ) = C _ { Q , T } \eta ^ { 2 } \bar { s } ( W ) .
$$

Here we used $e ^ { 2 \tau } - 1 \leq 2 e ^ { 2 } \tau$ for $0 \leq \tau \leq \eta \leq 1$

By Lemma B.6, $\bar { s } ( W _ { t } ) \leq E _ { t } / \kappa _ { T }$ . In the loss increment, this bounds both the momentum contribution and the Taylor remainder by multiples of $\eta _ { t } ^ { 2 } E _ { t }$ . For a fixed exponent $\alpha > 0$ , let

$$
C _ { D , T , \alpha } : = \frac { 2 C _ { M , T , \alpha } + C _ { Q , T } } { \kappa _ { T } } .
$$

To ensure that these errors are dominated by the gradient lower bound, let

$$
\bar { \eta } _ { T , \alpha , \mathsf { N } } : = \operatorname* { m i n } \left\{ 1 , \frac { c _ { \mathsf { N } } } { 2 C _ { D , T , \alpha } } \right\} > 0 .
$$

For fixed $\Delta _ { T } , \alpha , \beta , \mathsf { N }$ , and training data, these constants are independent of $\eta _ { 0 }$ and t. For spectral Muon with $\alpha = 1 / 2$ and $\beta = \beta _ { 2 }$ , we take $\bar { \eta } _ { T } : = \bar { \eta } _ { T , 1 / 2 , \| \cdot \| _ { \mathrm { o p } } }$ in Theorem 2.

Lemma B.16. Suppose that Assumptions 1 and 3 hold, $\begin{array} { r } { \mathcal { I } _ { 2 } \ne \emptyset , } \end{array}$ , and N satisfies (B.16). Fix a prediction matrix $W _ { 1 , T }$ satisfying $C W _ { 1 , T } = \Delta _ { T } B$ with $\Delta _ { T } \geq 2 N ( K - 1 )$ log $K _ { i }$ , and suppose that $\mathcal { L } _ { T } ( 0 ) \leq \log K$ . Use (B.22) with $W _ { 0 } = 0 $ $M _ { - 1 } = 0 ,$ , and

$$
\eta _ { t } = \eta _ { 0 } ( t + 1 ) ^ { - \alpha } , \qquad \alpha > 0 , \qquad 0 < \eta _ { 0 } \leq \bar { \eta } _ { T , \alpha , \tt N } .
$$

Then, for every integer $t \geq 0 ,$

$$
\begin{array} { r } { \mathcal { L } _ { T } ( W _ { t } ) \leq \mathcal { L } _ { T } ( 0 ) , \qquad a _ { n , x _ { n } ^ { \star } } ( W _ { t } ) \geq \delta _ { 0 } \quad ( n \in [ N ] ) , } \end{array}
$$

and the excess loss satisfies

$$
\begin{array} { r } { E _ { t + 1 } \leq \left[ 1 - \eta _ { t } \operatorname* { m a x } \{ c _ { \mathsf { N } } , \gamma _ { 2 , \mathsf { N } } - C _ { T } E _ { t } \} + C _ { D , T , \alpha } \eta _ { t } ^ { 2 } \right] E _ { t } . } \end{array}\tag{B.23}
$$

Proof. Let $S _ { 0 } : = \{ W : \mathcal { L } _ { T } ( W ) \leq \mathcal { L } _ { T } ( 0 ) \}$ }. Since $\mathcal { L } _ { T } ( 0 ) \leq \log K$ , Lemma B.5 gives $a _ { n , x _ { n } ^ { \star } } ( W ) \geq \delta _ { 0 }$ for every $W \in S _ { 0 }$ and $n \in [ N ]$ . We prove by induction that $W _ { t } \in S _ { 0 }$ for every $t \geq 0$ , starting from $W _ { 0 } = 0$

Suppose that $W _ { t } \in S _ { 0 }$ . Since $\eta _ { t } \leq 1$ , Lemmas B.14 and B.15, together with $E _ { T } = \mathcal { L } _ { T } - \ell _ { T } ^ { \star } , \mathrm { g i v e }$

$$
\begin{array} { r l } & { E _ { t + 1 } - E _ { t } \leq - \eta _ { t } \langle \nabla _ { W } \mathcal { L } _ { T } ( W _ { t } ) , D _ { t } \rangle + C _ { Q , T } \eta _ { t } ^ { 2 } \bar { s } ( W _ { t } ) } \\ & { \qquad \leq - \eta _ { t } \mathsf { N } _ { * } ( \nabla _ { W } \mathcal { L } _ { T } ( W _ { t } ) ) + ( 2 C _ { M , T , \alpha } + C _ { Q , T } ) \eta _ { t } ^ { 2 } \bar { s } ( W _ { t } ) . } \end{array}
$$

Using $\kappa _ { T } \bar { s } ( W _ { t } ) \leq E _ { t }$ , the gradient lower bound (B.20), and the definition of $C _ { D , T , \alpha }$ gives (B.23). The step size bound also gives

$$
\operatorname* { m a x } \{ c _ { \mathsf { N } } , \gamma _ { 2 , \mathsf { N } } - C _ { T } E _ { t } \} - C _ { D , T , \alpha } \eta _ { t } \ge c _ { \mathsf { N } } - C _ { D , T , \alpha } \eta _ { t } \ge \frac { c _ { \mathsf { N } } } { 2 } .
$$

Since $E _ { t } \geq 0$ by Lemma B.6, the recurrence yields

$$
E _ { t } - E _ { t + 1 } \geq \frac { c _ { \mathsf { N } } } { 2 } \eta _ { t } E _ { t } \geq 0 .
$$

Thus $\mathcal { L } _ { T } ( W _ { t + 1 } ) \leq \mathcal { L } _ { T } ( W _ { t } ) \leq \mathcal { L } _ { T } ( 0 )$ , which closes the induction. The attention lower bound and (B.23) therefore hold at every iteration. □

Loss decay and margin growth. The next lemma converts upper bounds on the excess loss into lower bounds on the training margin. We will combine it with (B.23) in the subsequent rate estimates. Let $L _ { \operatorname* { m a x } } : = \operatorname* { m a x } _ { n \in [ N ] } L _ { n }$ denote the largest training context length

Lemma B.17. Suppose that Assumption 1 holds and $\begin{array} { r } { \mathcal { I } _ { 2 } \ne \emptyset . } \end{array}$ . Let

$$
\nu _ { T } : = \frac { \kappa _ { T } \delta _ { 0 } } { N L _ { \mathrm { m a x } } } > 0 .
$$

For every $W \in \mathbb { R } ^ { d \times d }$ satisfying $a _ { n , x _ { n } ^ { \star } } ( W ) \geq \delta _ { 0 }$ for all $n \in [ N ]$

$$
E _ { T } ( W ) \geq \nu _ { T } e ^ { - \rho _ { 2 } ( W ) } .
$$

Proof. Since $\mathcal { T } _ { 2 }$ is finite and nonempty, choose $( n , x ) \in \mathbb { Z } _ { 2 }$ such that $\langle W , A _ { n , x } \rangle = \rho _ { 2 } ( W )$ , and let $c : = x _ { n } ^ { \star }$ . We have $c _ { n , x } \geq 1$ and $0 < c _ { n , c } \leq L _ { n } \leq L _ { \operatorname* { m a x } }$ . The attention ratio identity (B.5) and the assumed bound $a _ { n , c } ( W ) \geq \delta _ { 0 }$ therefore give

$$
a _ { n , x } ( W ) = a _ { n , c } ( W ) \frac { c _ { n , x } } { c _ { n , c } } e ^ { - \rho _ { 2 } ( W ) } \geq \frac { \delta _ { 0 } } { L _ { \operatorname* { m a x } } } e ^ { - \rho _ { 2 } ( W ) } .
$$

Since all $s _ { m } ( W )$ are nonnegative and $s _ { n } ( W ) \geq a _ { n , x } ( W )$ , Lemma B.6 yields

$$
E _ { T } ( W ) \ge \kappa _ { T } \bar { s } ( W ) \ge \frac { \kappa _ { T } } { N } s _ { n } ( W ) \ge \frac { \kappa _ { T } } { N } a _ { n , x } ( W ) \ge \nu _ { T } e ^ { - \rho _ { 2 } ( W ) } .
$$

We now combine the descent recurrence with Lemma B.17 to derive bounds on the excess loss and the normalized margin. Lemma B.18. Under the hypotheses ofLemma B.16, $E _ { t } > 0$ for every integer $t \geq 0 .$ . There are constants $C _ { L } , C _ { \rho } > 0 _ { : }$ depending only on $\Delta _ { T } , \alpha , \beta ,$ N and thefinite training data, and independent $o f \eta _ { 0 } , t$ and the maximizing directions, such that thefollowing bounds hold.

(i) For every integer $t \geq 0 ,$

$$
\log E _ { t } \leq - \gamma _ { 2 , \mathsf { N } } S _ { t } + C _ { L } Q _ { t } + C _ { L } .
$$

(ii) For every integer $t \geq 1 s a t i s f y i n g \gamma _ { 2 , \mathsf { N } } S _ { t } - C _ { L } Q _ { t } - C _ { \rho } > 0 ,$ , we have $W _ { t } \neq 0$ and

$$
0 \leq \gamma _ { 2 , \mathsf { N } } - \rho _ { 2 } \left( \frac { W _ { t } } { \mathsf { N } ( W _ { t } ) } \right) \leq \frac { C _ { L } Q _ { t } + C _ { \rho } } { S _ { t } } .\tag{B.24}
$$

$I f 0 < \alpha \leq 1$ , then the iterates are eventually nonzero and $\rho _ { 2 } ( W _ { t } / { \mathsf { N } } ( W _ { t } ) ) \to \gamma _ { 2 , \mathsf { N } } a s t \to \infty .$

Proof. Summing the updates from $W _ { 0 } = 0$ and using $\mathsf { N } ( D _ { s } ) \leq 1$ give

$$
W _ { t } = - \sum _ { s = 0 } ^ { t - 1 } \eta _ { s } D _ { s } , \qquad \mathsf { N } ( W _ { t } ) \le S _ { t } .
$$

Thus $\rho _ { 2 } ( W _ { t } )$ is finite at every finite iteration. Lemmas B.16 and B.17 imply

$$
E _ { t } \ge \nu _ { T } e ^ { - \rho _ { 2 } ( W _ { t } ) } > 0 .
$$

The recurrence (B.23) and the step size bound give, for every integer $m \geq 0$

$$
{ \frac { c _ { \mathsf { N } } } { 2 } } \sum _ { s = 0 } ^ { m } \eta _ { s } E _ { s } \leq \sum _ { s = 0 } ^ { m } ( E _ { s } - E _ { s + 1 } ) = E _ { 0 } - E _ { m + 1 } \leq E _ { 0 } .
$$

Letting $m  \infty$ yields

$$
\sum _ { s = 0 } ^ { \infty } \eta _ { s } E _ { s } \leq \frac { 2 E _ { 0 } } { c _ { \mathsf { N } } } .
$$

This summability bound controls the accumulated $C _ { T } \eta _ { s } E _ { s }$ correction in the logarithmic loss estimate. Since $E _ { s } , E _ { s + 1 } > 0$ the factor multiplying $E _ { s }$ in (B.23) is positive. Taking logarithms, using $\log ( 1 + v ) \leq v$ for $v > - 1$ and max $\{ c _ { \mathsf { N } } , \gamma _ { 2 , \mathsf { N } } -$ $C _ { T } E _ { s } \} \ge \gamma _ { 2 , \mathsf { N } } - C _ { T } E _ { s }$ , and summing give

$$
\begin{array} { r l } & { \log E _ { t } \leq \log E _ { 0 } - \displaystyle \sum _ { s = 0 } ^ { t - 1 } \eta _ { s } \left[ \operatorname* { m a x } \{ c _ { \mathsf { N } } , \gamma _ { 2 , \mathsf { N } } - C _ { T } E _ { s } \} - C _ { D , T , \alpha } \eta _ { s } \right] } \\ & { \qquad \leq \log E _ { 0 } - \gamma _ { 2 , \mathsf { N } } S _ { t } + C _ { T } \displaystyle \sum _ { s = 0 } ^ { t - 1 } \eta _ { s } E _ { s } + C _ { D , T , \alpha } Q _ { t } } \\ & { \qquad \leq - \gamma _ { 2 , \mathsf { N } } S _ { t } + C _ { D , T , \alpha } Q _ { t } + \log E _ { 0 } + \displaystyle \frac { 2 C _ { T } E _ { 0 } } { c _ { \mathsf { N } } } . } \end{array}
$$

Let

$$
C _ { L } : = \operatorname* { m a x } \left\{ 1 , C _ { D , T , \alpha } , \big \vert \log E _ { 0 } \big \vert + \frac { 2 C _ { T } E _ { 0 } } { c _ { \mathsf { N } } } \right\} , \qquad C _ { \rho } : = C _ { L } + \big \vert \log \nu _ { T } \big \vert .
$$

Since $C W _ { 1 , T } = \Delta _ { T } B$ and $W _ { 0 } = 0 ,$ the initial excess loss $E _ { 0 }$ depends only on $\Delta _ { T }$ and the training data. Thus these constants have the stated dependence, and (i) follows from the definition of $C _ { L }$

Lemma B.17 and (i) give

$$
\begin{array} { r l } & { \rho _ { 2 } ( W _ { t } ) \geq \log \nu _ { T } - \log E _ { t } } \\ & { \qquad \quad \geq \gamma _ { 2 , \mathsf { N } } S _ { t } - C _ { L } Q _ { t } - C _ { L } + \log \nu _ { T } } \\ & { \qquad \quad \geq \gamma _ { 2 , \mathsf { N } } S _ { t } - C _ { L } Q _ { t } - C _ { \rho } . } \end{array}
$$

When this lower bound is positive, $\rho _ { 2 } ( W _ { t } ) > 0$ and hence $W _ { t } \neq 0$ . Positive homogeneity and $\mathsf { N } ( W _ { t } ) \leq S _ { t }$ then imply

$$
\rho _ { 2 } \left( \frac { W _ { t } } { \mathsf { N } ( W _ { t } ) } \right) = \frac { \rho _ { 2 } ( W _ { t } ) } { \mathsf { N } ( W _ { t } ) } \ge \frac { \rho _ { 2 } ( W _ { t } ) } { S _ { t } } \ge \gamma _ { 2 , \mathsf { N } } - \frac { C _ { L } Q _ { t } + C _ { \rho } } { S _ { t } } .
$$

The maximum margin identity (B.18) gives $\rho _ { 2 } ( W _ { t } / \mathsf { N } ( W _ { t } ) ) \leq \gamma _ { 2 , \mathsf { N } } .$ , proving (ii).

Finally, suppose that $0 < \alpha \leq 1$ . Then $S _ { t } \to \infty$ and $\eta _ { s }  0$ . For every $\varepsilon > 0$ , choose m such that $\eta _ { s } \leq \varepsilon$ for all $s \geq m$ For $t \geq m$

$$
Q _ { t } = Q _ { m } + \sum _ { s = m } ^ { t - 1 } \eta _ { s } ^ { 2 } \le Q _ { m } + \varepsilon S _ { t } .
$$

Dividing by $S _ { t }$ , letting $t \to \infty$ , and then letting $\varepsilon \downarrow 0$ give $Q _ { t } / S _ { t }  0$ . Consequently, $\gamma _ { 2 , \mathsf { N } } S _ { t } - C _ { L } Q _ { t } - C _ { \rho } > 0$ for all sufficiently large t, and (ii) proves the stated limit. □

Power law step sizes. We now express the bounds in Lemma B.18 in terms of the iteration number.

Corollary B.2. Suppose that the hypotheses ofLemma B.16 hold and that $0 < \alpha \leq 1$ . Use the constants $C _ { L } , C _ { \rho }$ constructed in the proofofLemma B.18. Then thefollowing bounds hold.

(i) There are an integer $t _ { \alpha , \mathsf { N } } \geq 1$ and a constant $C _ { \alpha , \Nu } > 0$ such that, for every integer $t \geq t _ { \alpha , \ N }$ , we have $W _ { t } \neq 0$ and

$$
0 \leq \gamma _ { 2 , \mathsf { N } } - \rho _ { 2 } \left( \frac { W _ { t } } { \mathsf { N } ( W _ { t } ) } \right) \leq C _ { \alpha , \mathsf { N } } \left\{ \begin{array} { l l } { t ^ { - \alpha } , } & { 0 < \alpha < 1 / 2 , } \\ { ( 1 + \log t ) / \sqrt { t } , } & { \alpha = 1 / 2 , } \\ { t ^ { - ( 1 - \alpha ) } , } & { 1 / 2 < \alpha < 1 , } \\ { 1 / \log ( t + 1 ) , } & { \alpha = 1 . } \end{array} \right.\tag{B.25}
$$

(ii) $H \alpha = 1 / 2$ , there are constants $C _ { 1 } , C _ { 2 } , C _ { 3 } , C _ { 4 } > 0$ such that, for every integer $t \geq 1$

$$
\operatorname* { m a x } \{ E _ { t } , e ^ { - \rho _ { 2 } ( W _ { t } ) } \} \leq C _ { 1 } t ^ { C _ { 2 } } e ^ { - C _ { 3 } \sqrt { t } } .\tag{B.26}
$$

Whenever $\gamma _ { 2 , \mathsf { N } } S _ { t } - C _ { L } Q _ { t } - C _ { \rho } > 0 ,$ , we also have $W _ { t } \neq 0$ and

$$
0 \leq \gamma _ { 2 , \mathsf { N } } - \rho _ { 2 } \left( \frac { W _ { t } } { \mathsf { N } ( W _ { t } ) } \right) \leq C _ { 4 } \frac { 1 + \log t } { \sqrt { t } } .\tag{B.27}
$$

The constants and the iteration threshold may depend on $\Delta _ { T } , \alpha , \beta ,$ N, the finite training data, and $\eta _ { 0 } ,$ , but not on t or the maximizing directions.

Proof. For every integer $t \geq 1$ , termwise and integral comparison give

$$
S _ { t } \geq \eta _ { 0 } \left\{ \begin{array} { l l } { t ^ { 1 - \alpha } , } & { 0 < \alpha < 1 , } \\ { \log ( t + 1 ) , } & { \alpha = 1 , } \end{array} \right.
$$

$$
Q _ { t } \leq \eta _ { 0 } ^ { 2 } \left\{ \begin{array} { l l } { 1 + \displaystyle \frac { t ^ { 1 - 2 \alpha } - 1 } { 1 - 2 \alpha } , } & { 0 < \alpha < 1 / 2 , } \\ { 1 + \log t , } & { \alpha = 1 / 2 , } \\ { \displaystyle \frac { 2 \alpha } { 2 \alpha - 1 } , } & { 1 / 2 < \alpha \leq 1 . } \end{array} \right.
$$

For $0 < \alpha < 1$ , the lower bound on $S _ { t }$ follows from $k ^ { - \alpha } \geq t ^ { - \alpha }$ for $1 \leq k \leq t$ . The positivity condition in (B.24) holds for all sufficiently large t by Lemma B.18.

Substituting the bounds on $S _ { t } , Q _ { t }$ into (B.24) gives the four rates in (i). These sum bounds and the constants $C _ { L } , C _ { \rho }$ are independent of the maximizing directions, so $C _ { \alpha , \ N }$ and $t _ { \alpha , \ N }$ can also be chosen independently of them.

For (ii), let $\alpha = 1 / 2$ . For $t \geq 1$ , integral comparison and $\sqrt { t + 1 } + 1 \leq ( 1 + \sqrt { 2 } ) \sqrt { t }$ give

$$
S _ { t } \geq 2 \eta _ { 0 } ( \sqrt { t + 1 } - 1 ) = \frac { 2 \eta _ { 0 } t } { \sqrt { t + 1 } + 1 } \geq \frac { 2 \eta _ { 0 } } { 1 + \sqrt { 2 } } \sqrt { t } .
$$

For any $\kappa \geq C _ { L }$ , let

$$
\begin{array} { r l } & { C _ { 1 } : = \exp ( \displaystyle K + \displaystyle | \log \nu _ { T } | + C _ { 2 } + C _ { 3 } ) , } \\ & { C _ { 2 } : = \displaystyle K \eta _ { 0 } ^ { 2 } , \qquad C _ { 3 } : = 2 \gamma _ { 2 , \mathrm { N } } \eta _ { 0 } , } \\ & { C _ { 4 } : = \displaystyle \frac { 1 + \sqrt { 2 } } { 2 \eta _ { 0 } } \left( C _ { 2 } + K + | \log \nu _ { T } | \right) . } \end{array}\tag{B.28}
$$

Lemma B.16 verifies the attention condition needed for Lemma B.17. Together with (i) of Lemma B.18, this gives

$$
\operatorname* { m a x } \{ \log E _ { t } , - \rho _ { 2 } ( W _ { t } ) \} \leq - \gamma _ { 2 , \mathsf { N } } S _ { t } + C _ { L } Q _ { t } + C _ { L } + | \log \nu _ { T } | .
$$

Using $\mathcal { K } \geq C _ { L } , Q _ { t } \leq \eta _ { 0 } ^ { 2 } ( 1 + \log t )$ , and $S _ { t } \geq 2 \eta _ { 0 } ( \sqrt { t } - 1 )$ , we obtain

$$
\begin{array} { c } { { \operatorname* { m a x } \{ \log E _ { t } , - \rho _ { 2 } ( W _ { t } ) \} \leq - C _ { 3 } \sqrt { t } + C _ { 2 } \log t + K + | \log \nu _ { T } | + C _ { 2 } + C _ { 3 } } } \\ { { { } } } \\ { { = \log C _ { 1 } + C _ { 2 } \log t - C _ { 3 } \sqrt { t } . } } \end{array}
$$

Exponentiating proves (B.26).

Finally, since $C _ { \rho } = C _ { L } + \vert \log \nu _ { T } \vert$ , the bounds on $S _ { t } , Q _ { t }$ give

$$
\frac { C _ { L } Q _ { t } + C _ { \rho } } { S _ { t } } \leq \frac { 1 + \sqrt { 2 } } { 2 \eta _ { 0 } } \frac { C _ { 2 } ( 1 + \log { t } ) + K + | \log { \nu _ { T } } | } { \sqrt { t } } \leq C _ { 4 } \frac { 1 + \log { t } } { \sqrt { t } } .
$$

The last inequality uses $1 + \log t \geq 1$ . Whenever the positivity condition holds, (B.24) therefore proves (B.27). □

ProofofTheorem 2. Let $\mathsf { N } = \parallel \cdot \parallel _ { \mathrm { o p } } , \alpha = 1 / 2 , \beta = \beta _ { 2 }$ , and $\eta _ { 0 } = \eta _ { 2 , 0 }$ , so that $\gamma _ { 2 , \mathsf { N } } = \gamma _ { 2 }$ . Lemma B.1 gives ${ \cal C } W _ { 1 , T } =$ $\Delta _ { T } B$ . For $T > \bar { T }$ , Corollary B.1 ensures that $\Delta _ { T } \geq 2 N ( K - 1 )$ log K and $\mathcal { L } _ { T } ( 0 ) \leq \log K$ . The choice $\bar { \eta } _ { T } = \bar { \eta } _ { T , 1 / 2 , \parallel \cdot \parallel _ { \mathrm { o p } } }$ ensures the step size condition in Lemma B.16.

For nonzero momentum, Muon’s compact polar factor maximizes the momentum inner product over the spectral norm unit ball; when the momentum is zero, the prescribed zero direction is also a maximizer. Thus Lemma B.16 and Corollary B.2 apply.

Let $C _ { L } , C _ { \rho }$ be the constants constructed in the proof of Lemma B.18. That lemma gives $E _ { T } ( W _ { 2 , t } ) > 0$ at every finite iteration. Take $\displaystyle { \mathcal { K } } = C _ { L }$ in (B.28) to obtain $C _ { 1 } , \ldots , C _ { 4 }$ . Then (B.26) proves (6) and also gives

$$
e ^ { - \rho _ { 2 } ( W _ { 2 , t } ) } \leq C _ { 1 } t ^ { C _ { 2 } } e ^ { - C _ { 3 } \sqrt { t } } , \qquad t \geq 1 .\tag{B.29}
$$

We use the same $C _ { 1 } , C _ { 2 } , C _ { 3 }$ in the proof of Theorem 3.

By Lemma B.18, there is an integer $t _ { T } \geq 1$ such that $\gamma _ { 2 } S _ { t } - C _ { L } Q _ { t } - C _ { \rho } > 0$ for every $t \geq t _ { T }$ . Hence $W _ { 2 , t } \neq 0$ for such $t ,$ and (B.27) proves (7). The stated dependence of $C _ { 1 } , \ldots , C _ { 4 }$ follows from Lemma B.18 and their definitions in (B.28). The threshold $t _ { T }$ has the stated dependence because its defining condition involves only $C _ { L } , C _ { \rho } , S _ { t } , Q _ { t }$

Finally, $C _ { 3 } > 0$ and log $t / \sqrt { t } \to 0$ imply $E _ { T } ( W _ { 2 , t } )  0$ , and therefore ${ \mathcal L } _ { T } ( W _ { 2 , t } ) \to \ell _ { T } ^ { \star }$ . The normalized margin bound also gives $\rho _ { 2 } ( W _ { 2 , t } / \| W _ { 2 , t } \| _ { \mathrm { o p } } )  \gamma _ { 2 }$ □

## B.3. Prediction: proof of Theorem 3

We sum the training score gaps along the assumed chains to bound the attention assigned to other tokens, and then derive the prediction probability bound.

Proof. Let $W _ { t } : = W _ { 2 , t }$ and $\zeta _ { t } : = C _ { 1 } t ^ { C _ { 2 } } e ^ { - C _ { 3 } \sqrt { t } }$ . For all sufficiently large t, Theorem 2 gives $\rho _ { 2 } ( W _ { t } ) > 0$ , and (B.29) gives $e ^ { - \rho _ { 2 } ( W _ { t } ) } \leq \zeta _ { t }$ . Fix an unseen sample $( X , x _ { L + 1 } )$ satisfying the hypotheses of Theorem 3, with $X = [ x _ { 1 } , \dots , x _ { L } ]$ and $q = x _ { L }$

For each position $\ell \in [ L ]$ with $x _ { \ell } \neq x _ { X } ^ { \star }$ , choose a chain $u _ { 0 } , \ldots , u _ { r }$ with $r \geq 1 , u _ { 0 } = x _ { X } ^ { \star } , u _ { r } = x _ { \ell } .$ and $( u _ { j - 1 } , u _ { j } ) \in \mathcal { C } _ { q }$ for $j \in [ r ]$ ]. Each pair belongs to the training constraints with query q, so $( u _ { j - 1 } - u _ { j } ) ^ { \top } W _ { t } q \geq \rho _ { 2 } ( W _ { t } )$ . Summing along the chain and using $\rho _ { 2 } ( W _ { t } ) > 0$ give

$$
( x _ { X } ^ { \star } - x _ { \ell } ) ^ { \top } W _ { t } q = \sum _ { j = 1 } ^ { r } ( u _ { j - 1 } - u _ { j } ) ^ { \top } W _ { t } q \geq r \rho _ { 2 } ( W _ { t } ) \geq \rho _ { 2 } ( W _ { t } ) .
$$

Consequently, $e ^ { - ( x _ { X } ^ { \star } - x _ { \ell } ) ^ { \top } W _ { t } q } \leq \zeta _ { t }$ for every position containing another token.

Let $\alpha _ { t } : = \operatorname { s o f t m a x } ( X ^ { \top } W _ { t } q )$ , and let $\begin{array} { r } { a : = \sum _ { \ell : x _ { \ell } = x _ { \mathrm { v } } ^ { \star } } [ \alpha _ { t } ] _ { \ell } } \end{array}$ be the total attention weight assigned to the correct token. The correct token occurs at least once, and at most $L - 1$ positions contain other tokens. Thus

$$
\frac { 1 - a } { a } = \frac { \sum _ { \ell : x _ { \ell } \neq x _ { X } ^ { \star } } e ^ { - ( x _ { X } ^ { \star } - x _ { \ell } ) ^ { \top } W _ { t } q } } { | \{ \ell \in [ L ] : x _ { \ell } = x _ { X } ^ { \star } \} | } \leq ( L - 1 ) \zeta _ { t } .
$$

Since $a \leq 1$ , this implies $1 - a \leq ( L - 1 ) \zeta _ { t }$

Let $y : = I ( x _ { L + 1 } ) = h ( x _ { X } ^ { \star } )$ , and let $\widetilde { \boldsymbol { a } } \in \mathbb { R } ^ { K }$ collect the attention weights by output label:

$$
\widetilde { a } _ { v } : = \sum _ { \ell \in [ L ] : h ( x _ { \ell } ) = v } [ \alpha _ { t } ] _ { \ell } , \qquad v \in [ K ] .
$$

The coordinates of ea are nonnegative and sum to one. Since h is bijective, $\widetilde { a } _ { y } = a$ . Lemma B.1 gives $C W _ { 1 , T } = \Delta _ { T } B$ Using $B x = e _ { h ( x ) } - K ^ { - 1 } \mathbf { 1 }$ for $x \in \nu$ , we obtain

$$
C W _ { 1 , T } X \alpha _ { t } = \Delta _ { T } \left( \widetilde { a } - K ^ { - 1 } { \bf 1 } \right) .
$$

Adding the same constant to all logits leaves the softmax probabilities unchanged. Hence the prediction probability is $g ( \widetilde { a } )$ where $g ( u ) : = [ \mathrm { s o f t m a x } ( \Delta _ { T } u ) ] _ { y }$

For $u \in \mathbb { R } ^ { K }$ , let $p : = \operatorname { s o f t m a x } ( \Delta _ { T } u )$ . The softmax derivative and $p _ { v } \le 1 - p _ { y }$ for $v \neq y$ give

$$
\| \nabla g ( u ) \| _ { \infty } = \Delta _ { T } p _ { y } ( 1 - p _ { y } ) \leq \frac { \Delta _ { T } } { 4 } .
$$

Moreover, (3) gives

$$
g ( e _ { y } ) = \left[ 1 + ( K - 1 ) e ^ { - \Delta _ { T } } \right] ^ { - 1 } = e ^ { - \ell _ { T } ^ { \star } } .
$$

Since $\| \widetilde { \boldsymbol { a } } - \boldsymbol { e } _ { y } \| _ { 1 } = 2 ( 1 - \boldsymbol { a } )$ , integrating the gradient bound along the segment from $e _ { y }$ to $\widetilde { a }$ yields

$$
\begin{array} { r l } { \mathbb { P } _ { ( W _ { 1 } , T , W _ { 2 , t } ) } \{ x _ { L + 1 } ~ | ~ X \} = g ( \widetilde { a } ) } \\ & { \ge g ( e _ { y } ) - \frac { \Delta _ { T } } { 4 } \| \widetilde { a } - e _ { y } \| _ { 1 } } \\ & { = e ^ { - \ell _ { T } ^ { \star } } - \frac { \Delta _ { T } } { 2 } ( 1 - a ) } \\ & { \ge e ^ { - \ell _ { T } ^ { \star } } - \frac { \Delta _ { T } ( L - 1 ) } { 2 } \zeta _ { t } . } \end{array}
$$

Substituting $\zeta _ { t } = C _ { 1 } t ^ { C _ { 2 } } e ^ { - C _ { 3 } \sqrt { t } }$ gives (9).

## C. Convergence of Schatten Norm Updates

Fix $1 \leq p < \infty .$ , and let $p ^ { \prime }$ be its Holder conjugate, with¨ $p ^ { \prime } = \infty$ when $p = 1$ . Both phases use the Schatten $S _ { p }$ norm. For a momentum matrix H, let

$$
{ \mathcal { D } } _ { p } ( H ) : = \operatorname * { a r g m a x } _ { \| D \| _ { { \boldsymbol v } _ { p } } \leq 1 } \left. H , D \right. .
$$

Schatten norm duality gives $\langle H , D \rangle = \| H \| _ { S _ { p ^ { \prime } } }$ for every $D \in { \mathcal { D } } _ { p } ( H )$ . Algorithm C.1 allows any exact maximizing direction when the momentum is nonzero and uses the zero direction otherwise.

## Preprocessing phase

We use the notation of Appendix B.1. Recall that

$$
B B ^ { \top } = C , \qquad C B = B , \qquad \sum _ { i = 1 } ^ { K } \sum _ { v \neq h ( x _ { i } ) } C _ { x _ { i } , v } = K B .
$$

The $K - 1$ nonzero singular values of B are all one. Let $\gamma _ { 1 , p } : = ( K - 1 ) ^ { - 1 / p } .$

The next lemma identifies the maximum preprocessing margin and bounds the distance to $\gamma _ { 1 , p } B$ in terms of the margin gap.   
We will use it to prove convergence of the normalized preprocessing iterates.

Lemma C.1. Suppose that $K \geq 2 ,$ , h is bijective, and the vocabulary embeddings are orthonormal. Then

$$
\operatorname* { m a x } _ { \| V \| _ { \boldsymbol { s } _ { p } } \leq 1 } \rho _ { 1 } ( V ) = \gamma _ { 1 , p } .\tag{C.1}
$$

There is a constant $C _ { p , K } > 0 ,$ , depending only on $p , K ,$ , such that, for every $\varepsilon \geq 0$ and every $V \in \mathbb { R } ^ { K \times d }$ satisfying

$$
C V = V , \qquad \| V \| _ { S _ { p } } \leq 1 , \qquad \rho _ { 1 } ( V ) \geq \gamma _ { 1 , p } - \varepsilon ,
$$

Algorithm C.1 Two Phase Schatten $S _ { p }$ Training   
1: PHASE I: Preprocessing   
2: Input $p ; \mathcal { D } _ { 0 } ; \bar { T } ; \{ \eta _ { 1 , s } \} _ { s = 0 } ^ { T - 1 } ; \beta _ { 1 } \in [ 0 , 1 )$   
3: Initialization $W _ { 1 , 0 } = 0 ; M _ { 1 , - 1 } = 0 .$   
4: for $s = 0 , \ldots , T - 1$ do   
5: $M _ { 1 , s } = \beta _ { 1 } M _ { 1 , s - 1 } + ( 1 - \beta _ { 1 } ) \nabla \mathcal { L } _ { 0 } ( W _ { 1 , s } ) .$   
6: if ${ { M } _ { 1 , s } } = 0$ then   
7: $D _ { 1 , s } = 0 .$   
8: else   
9: $D _ { 1 , s } \in \mathcal { D } _ { p } ( M _ { 1 , s } ) .$   
10: end if   
11: $W _ { 1 , s + 1 } = W _ { 1 , s } - \eta _ { 1 , s } D _ { 1 , s } .$   
12: end for   
13: PHASE II: Main Training   
14: Input D; $W _ { 1 , T } ; t ; \{ \eta _ { 2 , r } \} _ { r = 0 } ^ { t - 1 } ; \beta _ { 2 } \in [ 0 , 1 )$   
15: Loss $\mathcal { L } _ { T } ( W ) : = \mathcal { L } ( W _ { 1 , T } , W ) .$   
16: Initialization $W _ { 2 , 0 } = 0 ; M _ { 2 , - 1 } = 0 .$   
17: for $r = 0 , \ldots , t - 1$ do   
18: $M _ { 2 , r } = \beta _ { 2 } M _ { 2 , r - 1 } + ( 1 - \beta _ { 2 } ) \nabla \mathcal { L } _ { T } ( W _ { 2 , r } ) .$   
19: if $M _ { 2 , r } = 0$ then   
20: $D _ { 2 , r } = 0 .$   
21: else   
22: $D _ { 2 , r } \in \mathcal { D } _ { p } ( M _ { 2 , r } )$   
23: end if   
24: $W _ { 2 , r + 1 } = W _ { 2 , r } - \eta _ { 2 , r } D _ { 2 , r } ,$   
25: end for   
26: Output $( W _ { 1 , T } , W _ { 2 , t } ) .$

we have

$$
\| V - \gamma _ { 1 , p } B \| _ { S _ { p } } \leq C _ { p , K } { \sqrt { \varepsilon } } .
$$

Proof. Let $r : = K - 1 , \gamma : = \gamma _ { 1 , p } ,$ and $y _ { i } : = h ( x _ { i } )$ . Since B has $r$ unit singular values, $\| \gamma B \| _ { S _ { p } } = \gamma r ^ { 1 / p } = 1$ . Also, $B x _ { i } = e _ { y _ { i } } - K ^ { - 1 } { \bf 1 }$ , so every preprocessing margin of $\gamma B$ equals γ.

For any V with $\| V \| _ { S _ { p } } \leq 1$ , let $g _ { i , v } : = ( e _ { y _ { i } } - e _ { v } ) ^ { \top } V x _ { i }$ for $i \in [ K ]$ and $v \in [ K ] \setminus \{ y _ { i } \}$ . Averaging these $K r$ margins and using the identity $\begin{array} { r } { \sum _ { i = 1 } ^ { K } \sum _ { v \neq y _ { i } } C _ { x _ { i } , v } = K B } \end{array}$ and Schatten norm duality give

$$
\rho _ { 1 } ( V ) \leq \frac { 1 } { K r } \sum _ { i = 1 } ^ { K } \sum _ { v \neq y _ { i } } g _ { i , v } = \frac { \langle V , B \rangle } { r } \leq \frac { \| V \| _ { S _ { p } } \| B \| _ { S _ { p ^ { \prime } } } } { r } \leq \gamma .
$$

This proves (C.1).

To prove the stability bound, let V satisfy its assumptions. We first bound $\| V X _ { \mathcal { V } } - \gamma P C \| _ { F }$ using the margin assumptions. The margin lower bound and the averaged estimate give

$$
g _ { i , v } - \gamma \geq - \varepsilon , \qquad \sum _ { i = 1 } ^ { K } \sum _ { v \neq y _ { i } } ( g _ { i , v } - \gamma ) \leq 0 .
$$

For any one term, the other $K r - 1$ terms have sum at least $- ( K r - 1 ) \varepsilon$ . Consequently,

$$
| g _ { i , v } - \gamma | \leq K r \varepsilon , \qquad i \in [ K ] , \quad v \neq y _ { i } .
$$

Let $A : = V X _ { \nu }$ . Since $C A = A$ , each column of A has coordinate sum zero. Its entries therefore satisfy

$$
A _ { y _ { i } , i } = \frac { 1 } { K } \sum _ { v \neq y _ { i } } g _ { i , v } , \qquad A _ { v , i } = A _ { y _ { i } , i } - g _ { i , v } \quad ( v \neq y _ { i } ) .
$$

The corresponding entries of $\gamma P C$ are $\gamma r / K$ at row $y _ { i }$ and $- \gamma / K$ elsewhere. Thus every entry of $A - \gamma P C$ has absolute value at most $2 K r \varepsilon ,$ giving

$$
\| A - \gamma P C \| _ { F } \leq L _ { K } \varepsilon , \qquad L _ { K } : = 2 K ^ { 2 } r .
$$

We next use the Schatten norm constraint to control the component outside the vocabulary span. Let $H : = V - A X _ { \nu } ^ { \top }$ Orthonormality and $C V = V$ give

$$
H X _ { \mathcal V } = 0 , \qquad C H = H , \qquad V V ^ { \top } = A A ^ { \top } + H H ^ { \top } .
$$

Let $\varepsilon _ { 0 } : = \operatorname* { m i n } \{ 1 , \gamma / ( 2 L _ { K } ) \}$ . First suppose that $0 \leq \varepsilon \leq \varepsilon _ { 0 }$ , and let $a : = \gamma - L _ { K } \varepsilon \geq \gamma / 2$ . Since P preserves $\mathbf { 1 } ^ { \perp }$ and $\| A - \gamma P C \| _ { \mathrm { o p } } \leq L _ { K } \varepsilon , { \mathrm { e v e r y ~ } } z \in { \bf 1 } ^ { \perp }$ satisfies

$$
\begin{array} { r } { \| A ^ { \top } z \| _ { 2 } \geq \gamma \| z \| _ { 2 } - \| ( A - \gamma P C ) ^ { \top } z \| _ { 2 } \geq a \| z \| _ { 2 } . } \end{array}
$$

Together with $A ^ { \top } \mathbf { 1 } = 0$ , this gives $A A ^ { \top } \succeq a ^ { 2 } C$

Let $\lambda _ { 1 } , \ldots , \lambda _ { \ i }$ <sub>r</sub> be the eigenvalues of $V V ^ { \top }$ on $\mathbf { 1 ^ { \perp } }$ . Since $V V ^ { \top } \succeq A A ^ { \top } \succeq a ^ { 2 } C$ , we have $\lambda _ { i } \geq a ^ { 2 }$ . The Schatten norm constraint also gives

$$
\lambda _ { i } \leq 1 , \qquad \sum _ { i = 1 } ^ { r } \lambda _ { i } ^ { p / 2 } \leq 1 .
$$

Let $m _ { p , K } > 0$ , depending only on $p , K$ , be a lower bound for the derivative of $u ^ { p / 2 }$ on $[ ( \gamma / 2 ) ^ { 2 } , 1 ]$ . The mean value theorem gives

$$
\lambda _ { i } ^ { p / 2 } - a ^ { p } \geq m _ { p , K } ( \lambda _ { i } - a ^ { 2 } ) , \qquad i \in [ r ] .
$$

Taking traces in $\begin{array} { r } { V V ^ { \top } = A A ^ { \top } + H H ^ { \top } } \end{array}$ and using $A A ^ { \top } \succeq a ^ { 2 } C$ yield

$$
\sum _ { i = 1 } ^ { r } ( \lambda _ { i } - a ^ { 2 } ) = \| A \| _ { F } ^ { 2 } + \| H \| _ { F } ^ { 2 } - r a ^ { 2 } \geq \| H \| _ { F } ^ { 2 } .
$$

Summing the bounds on $\lambda _ { i } ^ { p / 2 } - a ^ { p }$ and using $r \gamma ^ { p } = 1$ , we obtain

$$
\begin{array} { l } { m _ { p , K } \| H \| _ { F } ^ { 2 } \leq m _ { p , K } \displaystyle \sum _ { i = 1 } ^ { r } ( \lambda _ { i } - a ^ { 2 } ) } \\ { \leq \displaystyle \sum _ { i = 1 } ^ { r } ( \lambda _ { i } ^ { p / 2 } - a ^ { p } ) } \\ { \leq 1 - r a ^ { p } } \\ { = r ( \gamma ^ { p } - a ^ { p } ) } \\ { \leq r p \gamma ^ { p - 1 } L _ { K } \varepsilon . } \end{array}
$$

The last inequality follows from the mean value theorem, since $p \geq 1 , a \leq \gamma$ , and $\gamma - a = L _ { K } \varepsilon$

Since $B = P C X _ { \nu } ^ { \top }$ and $H X _ { \nu } = 0$ , the matrices $( A - \gamma P C ) X _ { \nu } ^ { \top }$ and H are orthogonal in the Frobenius inner product. Using $\varepsilon \le \varepsilon _ { 0 } \le 1$ , we therefore have

$$
\begin{array} { r l } & { \| V - \gamma B \| _ { F } ^ { 2 } = \| A - \gamma P C \| _ { F } ^ { 2 } + \| H \| _ { F } ^ { 2 } } \\ & { \qquad \leq L _ { K } ^ { 2 } \varepsilon ^ { 2 } + \frac { r p \gamma ^ { p - 1 } L _ { K } } { m _ { p , K } } \varepsilon } \\ & { \qquad \leq \left( L _ { K } ^ { 2 } + \frac { r p \gamma ^ { p - 1 } L _ { K } } { m _ { p , K } } \right) \varepsilon . } \end{array}
$$

Since $C ( V - \gamma B ) = V - \gamma B$ , this matrix has rank at most $^ r \cdot$ Comparison of its singular value norms yields

$$
\begin{array} { r } { \| V - \gamma B \| _ { S _ { p } } \leq r ^ { \operatorname* { m a x } \{ 1 / p - 1 / 2 , 0 \} } \| V - \gamma B \| _ { F } \leq C _ { p , K } \sqrt { \varepsilon } , } \end{array}
$$

where $C _ { p , K }$ depends only on $p , K$

For $\varepsilon > \varepsilon _ { 0 }$ , the triangle inequality and the norm constraint give

$$
\| V - \gamma B \| _ { S _ { p } } \leq \| V \| _ { S _ { p } } + \| \gamma B \| _ { S _ { p } } \leq 2 \leq \frac { 2 } { \sqrt { \varepsilon _ { 0 } } } \sqrt { \varepsilon } .
$$

Enlarge $C _ { p , K }$ , if necessary, so that $C _ { p , K } \geq 2 / \sqrt { \varepsilon _ { 0 } }$ . This preserves the bound for $\varepsilon \le \varepsilon _ { 0 }$ and gives the required estimate for all $\varepsilon \geq 0$ . The enlarged constant depends only on $p , K$ , since $\varepsilon _ { 0 }$ does. □

For $y _ { i } : = h ( x _ { i } )$ , let $p _ { i } ( W ) : =$ softmax $W x _ { i } )$ and $\ell _ { i } ( W ) : = - \log [ p _ { i } ( W ) ] _ { y _ { i } }$ . For the preprocessing iterates, write $L _ { s } : = \mathcal { L } _ { 0 } ( W _ { 1 , s } )$ and $q _ { s } : = q ( W _ { 1 , s } )$ , where

$$
q ( W ) : = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \left( 1 - [ p _ { i } ( W ) ] _ { y _ { i } } \right) .
$$

The next lemma bounds the loss decrease in terms of $q _ { s }$

Lemma C.2. Under the assumptions of Lemma C.1, consider Phase I of Algorithm C.1 with

$$
\eta _ { 1 , s } = \eta _ { 1 , 0 } ( s + 1 ) ^ { - \alpha _ { 1 } } , \qquad \eta _ { 1 , 0 } > 0 , \qquad 0 < \alpha _ { 1 } \le 1 .
$$

There is a constant $A _ { \mathrm { D } } > 0 ,$ , depending only on $K , p , \alpha _ { 1 } , \beta _ { 1 } , \eta _ { 1 , 0 } ,$ such that, whenever $\eta _ { 1 , s } \leq 1$

$$
L _ { s + 1 } - L _ { s } \leq - \eta _ { 1 , s } ( \gamma _ { 1 , p } - A _ { \mathrm { D } } \eta _ { 1 , s } ) q _ { s } .\tag{C.2}
$$

Proof. Let $\gamma : = \gamma _ { 1 , p } , \eta _ { s } : = \eta _ { 1 , s }$ , and $G _ { s } : = \nabla \mathcal { L } _ { 0 } ( W _ { 1 , s } )$ . Recall that $C _ { x _ { i } , v } = ( e _ { y _ { i } } - e _ { v } ) x _ { i } ^ { \top }$ for $v \neq y _ { i }$ . The gradient expansion gives

$$
- \boldsymbol { \nabla } \mathcal { L } _ { 0 } ( W ) = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \sum _ { v \neq y _ { i } } [ p _ { i } ( W ) ] _ { v } C _ { x _ { i } , v } .
$$

Since $\langle B , C _ { x _ { i } , v } \rangle = 1$ and $\| \gamma B \| _ { S _ { p } } = 1$ , norm duality yields

$$
\begin{array} { r } { \| \nabla \mathcal { L } _ { 0 } ( W ) \| _ { S _ { p ^ { \prime } } } \geq - \langle \nabla \mathcal { L } _ { 0 } ( W ) , \gamma B \rangle = \gamma q ( W ) . } \end{array}
$$

For $d _ { W } : = \| W ^ { \prime } - W \| _ { S _ { p } }$ , each output logit changes by at most $d _ { W }$ . The corresponding softmax probability ratios lie in $[ e ^ { - 2 d _ { W } } , e ^ { 2 d _ { W } } ]$ , so

$$
| [ p _ { i } ( W ^ { \prime } ) ] _ { v } - [ p _ { i } ( W ) ] _ { v } | \leq ( e ^ { 2 d _ { W } } - 1 ) [ p _ { i } ( W ) ] _ { v } , \qquad i , v \in [ K ] .
$$

Using the gradient expansion and $\| C _ { x _ { i } , v } \| _ { S _ { p ^ { \prime } } } = \sqrt { 2 } ,$ , we obtain

$$
\| \nabla \mathcal { L } _ { 0 } ( W ^ { \prime } ) - \nabla \mathcal { L } _ { 0 } ( W ) \| _ { S _ { p ^ { \prime } } } \leq \frac { \sqrt { 2 } } { K } \sum _ { i = 1 } ^ { K } \sum _ { v \neq y _ { i } } | [ p _ { i } ( W ^ { \prime } ) ] _ { v } - [ p _ { i } ( W ) ] _ { v } | \leq \sqrt { 2 } ( e ^ { 2 d _ { W } } - 1 ) q ( W ) .
$$

Let $\widehat { M } _ { 1 , s } : = M _ { 1 , s } / ( 1 - \beta _ { 1 } ^ { s + 1 } )$ . For $0 < \beta _ { 1 } < 1$ , let

$$
w _ { s , r } : = \frac { ( 1 - \beta _ { 1 } ) \beta _ { 1 } ^ { r } } { 1 - \beta _ { 1 } ^ { s + 1 } } , \qquad 0 \le r \le s .
$$

Expanding the momentum recursion and using $\textstyle \sum _ { r = 0 } ^ { s } w _ { s , r } = 1$ give

$$
\widehat { M } _ { 1 , s } - G _ { s } = \sum _ { r = 0 } ^ { s } w _ { s , r } ( G _ { s - r } - G _ { s } ) .
$$

Since $\| D _ { 1 , j } \| _ { S _ { p } } \leq 1$ , summing the parameter updates gives

$$
\| W _ { 1 , s } - W _ { 1 , s - r } \| _ { S _ { p } } \leq \sum _ { j = 1 } ^ { r } \eta _ { s - j } .
$$

The gradient variation bound and Lemma B.12 therefore imply

$$
\begin{array} { r l r } {  { \| \widehat { M } _ { 1 , s } - G _ { s } \| _ { S _ { p ^ { \prime } } } \leq \sum _ { r = 0 } ^ { s } w _ { s , r } \| G _ { s - r } - G _ { s } \| _ { S _ { p ^ { \prime } } } } } \\ & { } & { \leq \sqrt { 2 } q _ { s } \sum _ { r = 0 } ^ { s } w _ { s , r } [ \exp ( 2 \sum _ { j = 1 } ^ { r } \eta _ { s - j } ) - 1 ] } \\ & { } & { \leq A _ { \mathrm { M } } \eta _ { s } q _ { s } , } \end{array}
$$

where

$$
A _ { \mathrm { M } } : = \frac { \sqrt { 2 } C ( \alpha _ { 1 } , 2 \eta _ { 1 , 0 } , \beta _ { 1 } ) } { \eta _ { 1 , 0 } } .
$$

For the last inequality, we applied Lemma B.12 to the sequence $( j + 1 ) ^ { - \alpha _ { 1 } }$ with exponential parameter $2 \eta _ { 1 , 0 }$ . Its upper bound is $C ( \alpha _ { 1 } , 2 \eta _ { 1 , 0 } , \beta _ { 1 } ) ( s + 1 ) ^ { - \alpha _ { 1 } } = C ( \alpha _ { 1 } , 2 \eta _ { 1 , 0 } , \beta _ { 1 } ) \eta _ { s } / \eta _ { 1 , 0 }$ . When $\beta _ { 1 } = 0 .$ , we have $\widehat { M } _ { 1 , s } = G _ { s }$ and $A _ { \mathrm { M } } = 0$ , so the same error bound holds.

The maximizing property of $D _ { 1 , s }$ and the duality argument in the proof of Lemma B.14 give

$$
\langle G _ { s } , D _ { 1 , s } \rangle \geq \| G _ { s } \| _ { S _ { p ^ { \prime } } } - 2 \| \widehat { M } _ { 1 , s } - G _ { s } \| _ { S _ { p ^ { \prime } } } \geq ( \gamma - 2 A _ { \mathrm M } \eta _ { s } ) q _ { s } .
$$

Let $A _ { \mathrm { D } } : = 2 A _ { \mathrm { M } } + \sqrt { 2 } e ^ { 2 }$ . For $\eta _ { s } \leq 1$ , integration along the update segment yields

$$
\begin{array} { r l } & { L _ { s + 1 } - L _ { s } \le - \eta _ { s } \left. { G _ { s } , D _ { 1 , s } } \right. + \sqrt 2 q _ { s } \displaystyle \int _ { 0 } ^ { \eta _ { s } } \left( e ^ { 2 \tau } - 1 \right) d \tau } \\ & { \qquad \le - \eta _ { s } ( \gamma - 2 A _ { \mathrm { M } } \eta _ { s } ) q _ { s } + \sqrt 2 e ^ { 2 } \eta _ { s } ^ { 2 } q _ { s } } \\ & { \qquad = - \eta _ { s } ( \gamma - A _ { \mathrm { D } } \eta _ { s } ) q _ { s } , } \end{array}
$$

where the second inequality uses $e ^ { 2 \tau } - 1 \leq 2 e ^ { 2 } \tau$ for $0 \leq \tau \leq 1$

We next use the descent estimate to obtain loss and margin bounds, then apply Lemma C.1 to the normalized iterates. Lemma C.3. Under the hypotheses of Lemma C.2, let $\begin{array} { r } { S _ { 1 , s } : = \sum _ { j = 0 } ^ { s - 1 } \eta _ { 1 , j } } \end{array}$ . In the decomposition

$$
C W _ { 1 , s } = \delta _ { s } B + R _ { s } , \qquad \delta _ { s } : = { \frac { \langle C W _ { 1 , s } , B \rangle } { K - 1 } } , \qquad \langle R _ { s } , B \rangle = 0 ,
$$

we have $\delta _ { s } > 0$ eventually and

$$
\frac { \delta _ { s } } { S _ { 1 , s } } \longrightarrow \gamma _ { 1 , p } , \qquad \frac { \| R _ { s } \| _ { S _ { p } } } { \delta _ { s } } \longrightarrow 0 .\tag{C.3}
$$

Proof. Let $\gamma : = \gamma _ { 1 , p } , \eta _ { s } : = \eta _ { 1 , s } ,$ and $\begin{array} { r } { Q _ { 1 , s } : = \sum _ { j = 0 } ^ { s - 1 } \eta _ { j } ^ { 2 } } \end{array}$ . Since $\eta _ { s }  0$ , choose an integer $s _ { 0 } \geq 0 ,$ , depending only on the fixed parameters, such that

$$
\eta _ { s } \leq \mathrm { \operatorname * { m i n } } \{ 1 , \gamma / ( 2 A _ { \mathrm { D } } ) \} , \qquad s \geq s _ { 0 } .
$$

By (C.2),

$$
L _ { s + 1 } - L _ { s } \leq - \frac { \gamma } { 2 } \eta _ { s } q _ { s } \leq 0 , \qquad s \geq s _ { 0 } .
$$

Let B<sub>0</sub> := log $K + 2 S _ { 1 , s _ { 0 } }$ . Since $W _ { 1 , 0 } = 0$ and $\| D _ { 1 , j } \| _ { S _ { p } } \leq 1$ , we have $\| W _ { 1 , s } \| _ { S _ { p } } \leq S _ { 1 , }$ <sub>s</sub> and hence $\ell _ { i } ( W _ { 1 , s } ) \leq B _ { 0 }$ for $s \leq s _ { 0 }$ . For $s \geq s _ { 0 }$ , monotonicity gives $L _ { s } \leq L _ { s _ { 0 } } \leq B _ { 0 }$ and $\ell _ { i } ( W _ { 1 , s } ) \leq K L _ { s }$ . Thus

$$
L _ { s } \le B _ { 0 } , \qquad 0 \le \ell _ { i } ( W _ { 1 , s } ) \le K B _ { 0 } , \qquad i \in [ K ] , \quad s \ge 0 .
$$

On $[ 0 , K B _ { 0 } ]$ , there is a constant $c > 0$ , depending only on the fixed parameters, such that $1 - e ^ { - u } \ge c u$ . Since $1 - [ p _ { i } ( W _ { 1 , s } ) ] _ { y _ { i } } = 1 - e ^ { - \ell _ { i } ( W _ { 1 , s } ) }$ , averaging gives $q _ { s } \geq c L _ { s }$ . Summing the descent estimate yields

$$
\frac { c \gamma } { 2 } \sum _ { j = s _ { 0 } } ^ { s - 1 } \eta _ { j } L _ { j } \leq L _ { s _ { 0 } } - L _ { s } \leq L _ { s _ { 0 } } .
$$

For every integer $s \geq s _ { 0 }$ , splitting the sum at $s _ { 0 }$ gives

$$
\begin{array} { r l } { \displaystyle \sum _ { j = 0 } ^ { s - 1 } \eta _ { j } L _ { j } = \sum _ { j = 0 } ^ { s _ { 0 } - 1 } \eta _ { j } L _ { j } + \sum _ { j = s _ { 0 } } ^ { s - 1 } \eta _ { j } L _ { j } } & { } \\ { \displaystyle \le B _ { 0 } \sum _ { j = 0 } ^ { s _ { 0 } - 1 } \eta _ { j } + \frac { 2 L _ { s _ { 0 } } } { c \gamma } } & { } \\ { \displaystyle \le B _ { 0 } S _ { 1 , s _ { 0 } } + \frac { 2 B _ { 0 } } { c \gamma } } & { } \\ { \displaystyle = B _ { 0 } \left( S _ { 1 , s _ { 0 } } + \frac { 2 } { c \gamma } \right) . } \end{array}
$$

Here we used $L _ { j } \ \leq \ B _ { 0 }$ for the first sum and the preceding descent estimate for the second. Since all summands are nonnegative, letting $s $ ∞ yields

$$
\sum _ { j = 0 } ^ { \infty } \eta _ { j } L _ { j } \le B _ { 0 } \left( S _ { 1 , s _ { 0 } } + \frac { 2 } { c \gamma } \right) < \infty .
$$

To obtain the logarithmic loss estimate with coefficient $\gamma ,$ we use $1 - e ^ { - u } \ge u - u ^ { 2 } / 2$ for u $\geq 0 \colon$

$$
q _ { s } \geq L _ { s } - \frac { 1 } { 2 K } \sum _ { i = 1 } ^ { K } \ell _ { i } ( W _ { 1 , s } ) ^ { 2 } \geq L _ { s } - \frac { K } { 2 } L _ { s } ^ { 2 } .
$$

Every finite iterate has finite logits, so $L _ { s } > 0$ . Dividing (C.2) by $L _ { s }$ therefore gives a positive upper bound on $L _ { s + 1 } / L _ { s }$ For $s \geq s _ { 0 }$ , taking logarithms and using $\log ( 1 + v ) \leq v$ give

$$
\begin{array} { r l r } {  { \log L _ { s } \leq \log L _ { s _ { 0 } } - \sum _ { j = s _ { 0 } } ^ { s - 1 } \eta _ { j } ( \gamma - A _ { \mathrm { D } } \eta _ { j } ) \frac { q _ { j } } { L _ { j } } } } \\ & { } & \\ & { } & { \leq \log L _ { s _ { 0 } } - \gamma ( S _ { 1 , s } - S _ { 1 , s _ { 0 } } ) + A _ { \mathrm { D } } ( Q _ { 1 , s } - Q _ { 1 , s _ { 0 } } ) + \frac { \gamma K } { 2 } \sum _ { j = s _ { 0 } } ^ { s - 1 } \eta _ { j } L _ { j } . } \end{array}
$$

For the second inequality, we use $q _ { j } / L _ { j } \geq 1 - K L _ { j } / 2$ and $\gamma - A _ { \mathrm { D } } \eta _ { j } > 0$ to obtain

$$
\begin{array} { r l } { \displaystyle \left( \gamma - A _ { \mathrm { D } } \eta _ { j } \right) \frac { q _ { j } } { L _ { j } } \geq \left( \gamma - A _ { \mathrm { D } } \eta _ { j } \right) \left( 1 - \frac K 2 L _ { j } \right) } & { } \\ { = \gamma - A _ { \mathrm { D } } \eta _ { j } - \frac { \gamma K } 2 L _ { j } + \frac { A _ { \mathrm { D } } K } 2 \eta _ { j } L _ { j } } & { } \\ { \geq \gamma - A _ { \mathrm { D } } \eta _ { j } - \frac { \gamma K } 2 L _ { j } . } \end{array}
$$

The weighted loss sum is uniformly bounded, and log $L _ { s _ { 0 } } \leq \log B _ { 0 }$ . Hence all terms other than $- \gamma S _ { 1 , s } + A _ { \mathrm { D } } Q _ { 1 , s }$ are bounded above by a constant depending only on the fixed parameters.

For each $i \in [ K ]$ , the sample loss satisfies

$$
e ^ { \ell _ { i } ( W ) } - 1 = \sum _ { v \neq y _ { i } } e ^ { - ( e _ { y _ { i } } - e _ { v } ) ^ { \top } W x _ { i } } .
$$

By the definition of $\rho _ { 1 }$ and the bound $\ell _ { i } ( W _ { 1 , s } ) \leq K L _ { s }$ , we obtain

$$
\begin{array} { r l } {  { e ^ { - \rho _ { 1 } ( W _ { 1 , s } ) } = \operatorname* { m a x } _ { i \in [ K ] , v \neq y _ { i } } e ^ { - ( e _ { y _ { i } } - e _ { v } ) ^ { \top } W _ { 1 , s } x _ { i } } } } \\ & { \leq \operatorname* { m a x } _ { i \in [ K ] } ( e ^ { \ell _ { i } ( W _ { 1 , s } ) } - 1 ) } \\ & { \leq e ^ { K L _ { s } } - 1 } \\ & { \leq K e ^ { K B _ { 0 } } L _ { s } . } \end{array}
$$

The last inequality follows from the mean value theorem and $L _ { s } \leq B _ { 0 }$

Taking logarithms in the preceding margin bound gives

$$
\rho _ { 1 } ( W _ { 1 , s } ) \geq - \log L _ { s } - \log K - K B _ { 0 } .
$$

Combining this with the logarithmic loss estimate, we may take $C _ { 1 } : = A _ { \mathrm { D } }$ and choose $C _ { 0 } > 0$ such that, for every $s \geq 0$

$$
\begin{array} { r l } & { 0 < L _ { s } \leq \exp ( C _ { 0 } - \gamma S _ { 1 , s } + C _ { 1 } Q _ { 1 , s } ) , } \\ & { \rho _ { 1 } ( W _ { 1 , s } ) \geq \gamma S _ { 1 , s } - C _ { 1 } Q _ { 1 , s } - C _ { 0 } . } \end{array}
$$

For $0 \leq s < s _ { 0 }$ , we have $L _ { s } \leq B _ { 0 }$ and $\rho _ { 1 } ( W _ { 1 , s } ) \geq - 2 S _ { 1 , s _ { 0 } } .$ . Increasing $C _ { 0 }$ , if necessary, makes both estimates valid on this interval as well. The constants depend only on $K , p , \alpha _ { 1 } , \beta _ { 1 } , \eta _ { 1 , 0 }$

For $s \geq 1$ , let

$$
V _ { s } : = \frac { C W _ { 1 , s } } { S _ { 1 , s } } , \qquad e _ { s } : = C _ { 0 } + C _ { 1 } Q _ { 1 , s } .
$$

Multiplication by $C$ does not increase the Schatten norm and leaves all preprocessing margins unchanged. Together with $C ^ { 2 } = C$ and the preceding bounds, this gives

$$
C V _ { s } = V _ { s } , \qquad \| V _ { s } \| _ { S _ { p } } \leq 1 , \qquad \rho _ { 1 } ( V _ { s } ) \geq \gamma - \frac { e _ { s } } { S _ { 1 , s } } .
$$

Since $0 < \alpha _ { 1 } \leq 1$ , we have $S _ { 1 , s }  \infty$ . The argument at the end of Lemma B.18 also gives $Q _ { 1 , s } / S _ { 1 , s } \to 0$ . Thus $e _ { s } / S _ { 1 , s }  0$ , and Lemma C.1 yields

$$
\| V _ { s } - \gamma B \| _ { S _ { p } } \le C _ { p , K } \sqrt { \frac { e _ { s } } { S _ { 1 , s } } } \longrightarrow 0 .
$$

Finally, the definition of $\delta _ { s }$ and Schatten norm duality give

$$
\left| \frac { \delta _ { s } } { S _ { 1 , s } } - \gamma \right| = \frac { | \left. V _ { s } - \gamma B , B \right. | } { K - 1 } \le \gamma \| V _ { s } - \gamma B \| _ { S _ { p } } ,
$$

where $\| B \| _ { S _ { v ^ { \prime } } } / ( K - 1 ) = \gamma$ . Since $R _ { s } / S _ { 1 , s } = V _ { s } - ( \delta _ { s } / S _ { 1 , s } ) B$ and $\gamma \| B \| _ { S _ { p } } = 1$ , we also have

$$
\frac { \| R _ { s } \| _ { S _ { p } } } { S _ { 1 , s } } \leq \| V _ { s } - \gamma B \| _ { S _ { p } } + \left| \frac { \delta _ { s } } { S _ { 1 , s } } - \gamma \right| \| B \| _ { S _ { p } } \leq 2 \| V _ { s } - \gamma B \| _ { S _ { p } } .
$$

For all sufficiently large s, the first bound gives $\delta _ { s } / S _ { 1 , s } \geq \gamma / 2 > 0$ . Therefore,

$$
\frac { \| R _ { s } \| _ { S _ { p } } } { \delta _ { s } } \leq \frac { 4 } { \gamma } \| V _ { s } - \gamma B \| _ { S _ { p } } \longrightarrow 0 .
$$

This proves (C.3). The bounds above are uniform over the maximizing directions and independent of $d .$

## Main phase

Suppose that the data assumptions of Lemma C.1 hold. For the full training data, suppose that Assumptions 1, 2, and 3 hold and $\mathcal { T } _ { 2 } \neq \mathcal { O }$

Prediction matrix and loss. Let $U : = W _ { 1 , T }$ be fixed during Phase II, and write

$$
C U = \delta B + R , \qquad \delta : = \frac { \langle C U , B \rangle } { K - 1 } , \qquad \langle R , B \rangle = 0 , \qquad r _ { U } : = \| R \| _ { \mathrm { o p } } .\tag{C.4}
$$

By Lemma C.3, sufficiently long preprocessing ensures

$$
\delta \geq 2 ( K - 1 ) \left[ N \log K + \sqrt { 2 } ( N + 1 ) r _ { U } \right] .\tag{C.5}
$$

In particular, $\delta > 2 r _ { U }$

For each sample $n \in [ N ]$ , let $c : = x _ { n } ^ { \star }$ . For $a \in A _ { n } .$ , let

$$
m ( a ) : = \sum _ { x \in \mathcal { V } } a _ { x } x , \qquad p _ { n } ( a ) : = \mathrm { s o f t m a x } ( U m ( a ) ) .
$$

The sample loss and its excess over the loss at $e _ { c }$ are

$$
f _ { U , n } ( a ) : = - \log [ p _ { n } ( a ) ] _ { \mathcal { Y } _ { n } } , \qquad F _ { U , n } ( a ) : = f _ { U , n } ( a ) - f _ { U , n } ( e _ { c } ) .
$$

Let $\mathcal { L } _ { U } ( W ) : = \mathcal { L } ( U , W )$ and

$$
\ell _ { U } ^ { \star } : = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } f _ { U , n } ( e _ { x _ { n } ^ { \star } } ) , \qquad E _ { U } ( W ) : = \mathcal { L } _ { U } ( W ) - \ell _ { U } ^ { \star } .
$$

For $x \in \mathcal { X } _ { n } ^ { - }$ , let

$$
D _ { n , x } ^ { U } ( a ) : = \langle \nabla _ { a } f _ { U , n } ( a ) , e _ { x } - e _ { c } \rangle , \qquad \kappa _ { - } : = \operatorname* { m i n } _ { ( n , x ) \in \mathcal { Z } _ { 2 } } D _ { n , x } ^ { U } ( e _ { x _ { n } ^ { \star } } ) .
$$

Also let $Q _ { U } : = \| C U \| _ { \mathrm { o p } }$

The next two lemmas establish the loss and attention bounds needed for the common gradient estimates.

Lemma C.4. Suppose that U satisfies (C.5). Then $D _ { n , x } ^ { U } ( a ) > 0$ for every $n \in [ N ] , a \in { \mathcal { A } } _ { n } $ , and $x \in \mathcal { X } _ { n } ^ { - }$ . Moreover, $\kappa _ { - } > 0$ and

$$
0 < \kappa _ { - } \bar { s } ( W ) \leq E _ { U } ( W ) \leq 2 Q _ { U } \bar { s } ( W ) , \qquad W \in \mathbb { R } ^ { d \times d } .\tag{C.6}
$$

The training loss satisfies inf<sub>W</sub> $\mathcal { L } _ { U } ( W ) = \ell _ { U } ^ { \star }$

Proof. Fix $n \in [ N ] , a \in { \mathcal { A } } _ { n }$ , and $x \in \mathcal { X } _ { n } ^ { - }$ . Let $y : = y _ { n } , k : = h ( x )$ , and $q : = 1 - [ p _ { n } ( a ) ] _ { y }$ . The cross entropy derivative and $B ( x - c ) = e _ { k } - e _ { y }$ give

$$
D _ { n , x } ^ { U } ( a ) = ( p _ { n } ( a ) - e _ { y } ) ^ { \top } C U ( x - c ) = \delta \left( [ p _ { n } ( a ) ] _ { k } + q \right) + ( p _ { n } ( a ) - e _ { y } ) ^ { \top } R ( x - c ) .
$$

Since $\| p _ { n } ( a ) - e _ { y } \| _ { 2 } \leq \sqrt { 2 } q$ and $\| x - c \| _ { 2 } = { \sqrt { 2 } } .$

$$
| ( p _ { n } ( a ) - e _ { y } ) ^ { \top } R ( x - c ) | \leq \| p _ { n } ( a ) - e _ { y } \| _ { 2 ^ { T } U } \| x - c \| _ { 2 } \leq 2 r _ { U } q .
$$

The logits are finite and $K \geq 2 , \thinspace \thinspace \mathrm { s o ~ } q > 0$ . Using $\delta > 2 r _ { U }$ and $[ p _ { n } ( a ) ] _ { k } \geq 0$ , we obtain

$$
D _ { n , x } ^ { U } ( a ) \geq ( \delta - 2 r _ { U } ) q > 0 .
$$

Since $\mathcal { T } _ { 2 }$ is finite and nonempty, the minimum defining κ<sub>−</sub> is positive.

The function $f _ { U , n }$ is convex, so

$$
F _ { U , n } ( a ) \geq \langle \nabla _ { a } f _ { U , n } ( e _ { c } ) , a - e _ { c } \rangle = \sum _ { x \in \mathcal { X } _ { n } ^ { - } } a _ { x } D _ { n , x } ^ { U } ( e _ { c } ) \geq \kappa _ { - } s _ { n } ( a ) .
$$

Since $\| p _ { n } ( a ) - e _ { y } \| _ { 2 } \leq \sqrt { 2 } q \leq \sqrt { 2 }$ and $\| x - c \| _ { 2 } = { \sqrt { 2 } }$ , the derivative formula gives

$$
D _ { n , x } ^ { U } ( a ) \leq \| p _ { n } ( a ) - e _ { y } \| _ { 2 } Q _ { U } \| x - c \| _ { 2 } \leq 2 Q _ { U } .
$$

Integrating along the segment from $e _ { c }$ to a, we obtain

$$
F _ { U , n } ( a ) = \int _ { 0 } ^ { 1 } \sum _ { x \in \mathcal { X } _ { n } ^ { - } } a _ { x } D _ { n , x } ^ { U } ( e _ { c } + \tau ( a - e _ { c } ) ) d \tau \leq 2 Q _ { U } s _ { n } ( a ) .
$$

Evaluating these bounds at $a _ { n } ( W )$ and averaging over the samples give the empirical bounds. For finite $W$ , every token present in a context has positive attention weight. Since $\mathcal { T } _ { 2 } \neq \mathcal { O }$ , we have $\bar { s } ( W ) > 0$ , proving the strict lower bound in (C.6).

By Assumption 3, choose $W ^ { \star }$ with $\langle W ^ { \star } , A _ { n , x } \rangle \geq 1$ for every $( n , x ) \in \mathbb { Z } _ { 2 }$ . The attention ratio identity (B.5) gives $a _ { n } ( \lambda W ^ { \star } ) \to e _ { c }$ $\mathcal { L } _ { U } ( W ) = \ell _ { U } ^ { \star }$ $\lambda \to \infty$ . By continuity and finiteness of the training set, $\mathcal { L } _ { U } ( \lambda W ^ { \star } )  \ell _ { U } ^ { \star }$ . Together with $\mathcal { L } _ { U } ( W ) \geq \ell _ { U } ^ { \star }$ <sub>U</sub>,

Lemma C.5. Suppose that U satisfies (C.5), and let $L _ { 0 } : = \mathcal { L } _ { U } ( 0 )$ . Then

$$
L _ { 0 } \leq \log K + \sqrt 2 r _ { U } .\tag{C.7}
$$

For every $W \in \mathbb { R } ^ { d \times d }$

$$
{ \mathcal { L } } _ { U } ( W ) \leq L _ { 0 } \quad \implies \quad a _ { n , c } ( W ) \geq \delta _ { 0 } , \qquad n \in [ N ] .
$$

On this sublevel set, with $\nu _ { U } : = \kappa _ { - } \delta _ { 0 } / ( N L _ { \operatorname* { m a x } } ) > 0 ,$

$$
E _ { U } ( W ) \geq \nu _ { U } e ^ { - \rho _ { 2 } ( W ) } .\tag{C.8}
$$

Proof. Let $f _ { \delta , n }$ be the sample loss with prediction matrix δB. Adding the same constant to all logits leaves the cross entropy loss unchanged, so ${ f } _ { U , n } ( a )$ can be computed from $C U m ( a ) = \delta B m ( a ) + R m ( a )$

For $z \in \mathbb { R } ^ { K }$ , the cross entropy gradient softmax $( z ) - e _ { y _ { r } }$ has Euclidean norm at most $\sqrt { 2 }$ . Thus cross entropy is 2-Lipschitz in its logits. Since $\| m ( a ) \| _ { 2 } \leq 1$ , we obtain

$$
| f _ { U , n } ( a ) - f _ { \delta , n } ( a ) | \leq \sqrt { 2 } \| R m ( a ) \| _ { 2 } \leq \sqrt { 2 } r _ { U } .
$$

$\mathbf { A } \mathbf { t } W = 0$ , we have $a _ { n , x } ( 0 ) = c _ { n , x } / L _ { n }$ . Assumption 2 therefore makes the correct output logit under $\delta B$ maximal, giving $f _ { \delta , n } ( a _ { n } ( 0 ) ) \leq \log K$ . Averaging the loss perturbation bound proves (C.7).

Now fix W with $\mathcal { L } _ { U } ( W ) \leq L _ { 0 }$ and $n \in [ N ]$ , and let $u : = a _ { n , c } ( W )$ . The Jensen estimate in the proof of Lemma B.5, applied with logit gap δ, gives

$$
f _ { \delta , n } ( a _ { n } ( W ) ) \geq { \frac { \delta ( 1 - K u ) } { K - 1 } } .
$$

Combining this with the loss perturbation bound and nonnegativity of the sample losses yields

$$
\frac { \delta ( 1 - K u ) } { K - 1 } - \sqrt { 2 } r _ { U } \le f _ { U , n } ( a _ { n } ( W ) ) \le N \mathcal { L } _ { U } ( W ) \le N ( \log K + \sqrt { 2 } r _ { U } ) .
$$

Rearranging gives

$$
u \geq \frac { 1 } { K } \left( 1 - \frac { ( K - 1 ) [ N \log K + \sqrt { 2 } ( N + 1 ) r _ { U } ] } { \delta } \right) \geq \frac { 1 } { 2 K } = \delta _ { 0 } ,
$$

where the last inequality follows from (C.5).

Finally, choose $( n , x ) \in \mathbb { Z } _ { 2 }$ attaining $\rho _ { 2 } ( W )$ . Since $c _ { n , x } \geq 1$ and $c _ { n , c } \leq L _ { \operatorname* { m a x } } .$ the attention ratio identity (B.5) gives

$$
a _ { n , x } ( W ) = a _ { n , c } ( W ) \frac { c _ { n , x } } { c _ { n , c } } e ^ { - \rho _ { 2 } ( W ) } \geq \frac { \delta _ { 0 } } { L _ { \operatorname* { m a x } } } e ^ { - \rho _ { 2 } ( W ) } .
$$

Using (C.6) and $\bar { s } ( W ) \geq a _ { n , x } ( W ) / N$ , we obtain

$$
E _ { U } ( W ) \ge \kappa _ { - } \bar { s } ( W ) \ge \frac { \kappa _ { - } } { N } a _ { n , x } ( W ) \ge \nu _ { U } e ^ { - \rho _ { 2 } ( W ) } .
$$

This proves (C.8).

Gradient and momentum estimates. We first apply Lemma B.10 to $F _ { U , n }$ . Let $\begin{array} { r } { C _ { U } : = \frac { 2 \sqrt { 2 } N Q _ { U } } { \kappa _ { - } ^ { 2 } } } \end{array}$

Lemma C.6. Suppose that U satisfies (C.5) and N satisfies (B.16). For every $W \in \mathbb { R } ^ { d \times d }$

$$
\mathsf { N } _ { * } ( \nabla _ { W } \mathcal { L } _ { U } ( W ) ) \ge \gamma _ { 2 , \mathsf { N } } E _ { U } ( W ) - C _ { U } E _ { U } ( W ) ^ { 2 } ,\tag{C.9}
$$

$$
\begin{array} { r } { \mathsf { N } _ { * } ( \nabla _ { W } \mathcal { L } _ { U } ( W ) ) \leq 2 \sqrt { 2 } Q _ { U } \bar { s } ( W ) . } \end{array}\tag{C.10}
$$

If ${ \mathcal L } _ { U } ( W ) \leq L _ { 0 } ,$ , then

$$
\mathsf { N } _ { \ast } ( \nabla _ { W } \mathcal { L } _ { U } ( W ) ) \geq E _ { U } ( W ) \operatorname* { m a x } \{ c _ { \mathsf { N } } , \gamma _ { 2 , \mathsf { N } } - C _ { U } E _ { U } ( W ) \} ,\tag{C.11}
$$

where $c _ { \mathsf { N } }$ is the constant in Lemma B.10.

Proof. Fix $n \in [ N ]$ and $a \in A _ { n }$ . The term $f _ { U , n } ( e _ { c } )$ is constant in $^ { a , }$ and $p _ { n } ( a ) - e _ { y _ { r } }$ is orthogonal to 1. The cross entropy chain rule therefore gives

$$
\partial _ { x } F _ { U , n } ( a ) = ( p _ { n } ( a ) - e _ { y _ { n } } ) ^ { \top } U x = ( p _ { n } ( a ) - e _ { y _ { n } } ) ^ { \top } C U x .
$$

Using the definition in Lemma B.7, we obtain

$$
d _ { n , x } ^ { U } ( a ) : = \partial _ { x } F _ { U , n } ( a ) - \sum _ { z \in \mathcal { V } } a _ { z } \partial _ { z } F _ { U , n } ( a ) = ( p _ { n } ( a ) - e _ { y _ { n } } ) ^ { \top } C U ( x - m ( a ) ) .
$$

Applying the matrix gradient identity in that lemma and averaging over the samples yield

$$
- \nabla _ { W } \mathcal { L } _ { U } ( W ) = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \sum _ { x \in \mathcal { X } _ { n } ^ { - } } a _ { n , x } ( W ) d _ { n , x } ^ { U } ( a _ { n } ( W ) ) A _ { n , x } .\tag{C.12}
$$

Orthonormality of the vocabulary and $a \in A _ { n }$ give

$$
\| x - m ( a ) \| _ { 2 } ^ { 2 } = 1 - 2 a _ { x } + \sum _ { z \in \mathcal { V } } a _ { z } ^ { 2 } \leq 2 .
$$

Together with $\| p _ { n } ( a ) - e _ { y _ { n } } \| _ { 2 } \leq \sqrt { 2 } .$ , this implies

$$
| d _ { n , x } ^ { U } ( a ) | \leq \| p _ { n } ( a ) - e _ { y _ { n } } \| _ { 2 } Q _ { U } \| x - m ( a ) \| _ { 2 } \leq 2 Q _ { U } .
$$

Using $\mathsf { N } _ { * } ( A _ { n , x } ) = \sqrt { 2 }$ and the dual norm triangle inequality, we obtain

$$
\begin{array} { r l } & { \mathsf { N } _ { * } ( \nabla _ { W } \mathcal { L } _ { U } ( W ) ) \le \displaystyle \frac { \sqrt { 2 } } { N } \sum _ { n = 1 } ^ { N } \sum _ { x \in \mathcal { X } _ { n } ^ { - } } a _ { n , x } ( W ) | d _ { n , x } ^ { U } ( a _ { n } ( W ) ) | } \\ & { \qquad \le 2 \sqrt { 2 } Q _ { U } \bar { s } ( W ) . } \end{array}
$$

This proves (C.10).

For the lower bounds, each $F _ { U , n }$ extends to a smooth convex function on $\mathbb { R } ^ { \nu }$ and satisfies $F _ { U , n } ( e _ { c } ) = 0$ . By Lemma C.4,

$$
0 < D _ { n , x } ^ { U } ( a ) \leq 2 Q _ { U } , \qquad F _ { U , n } ( a ) \geq \kappa _ { - } s _ { n } ( a ) .
$$

Thus the hypotheses of Lemma B.10(ii) hold with $F _ { n } = F _ { U , n } , L = 2 Q _ { U }$ , and $\kappa = \kappa _ { - }$ . Under this choice, $E _ { F } = E _ { U }$ and $\nabla _ { W } E _ { F } = \nabla _ { W } \mathcal { L } _ { U }$ , so that lemma gives

$$
\begin{array} { r } { \Lambda _ { * } ( \nabla _ { W } \mathcal { L } _ { U } ( W ) ) \ge \gamma _ { 2 , \mathsf { N } } E _ { U } ( W ) - \frac { 2 \sqrt { 2 } N Q _ { U } } { \kappa _ { - } ^ { 2 } } E _ { U } ( W ) ^ { 2 } = \gamma _ { 2 , \mathsf { N } } E _ { U } ( W ) - C _ { U } E _ { U } ( W ) ^ { 2 } . } \end{array}
$$

This proves (C.9).

If $\mathcal { L } _ { U } ( W ) \leq L _ { 0 }$ , Lemma C.5 gives $a _ { n , c } ( W ) \geq \delta _ { 0 }$ for all $n \in [ N ]$ . Part (i) of Lemma B.10 then supplies the lower bound $c _ { \mathsf { N } } E _ { U } ( W )$ . Combining the two lower bounds and using $E _ { U } ( W ) \geq 0$ , we obtain

$$
\begin{array} { r l } & { \mathsf { N } _ { \ast } ( \nabla _ { W } \mathcal { L } _ { U } ( W ) ) \geq \operatorname* { m a x } \{ c _ { \mathsf { N } } E _ { U } ( W ) , \gamma _ { 2 , \mathsf { N } } E _ { U } ( W ) - C _ { U } E _ { U } ( W ) ^ { 2 } \} } \\ & { \qquad = E _ { U } ( W ) \operatorname* { m a x } \{ c _ { \mathsf { N } } , \gamma _ { 2 , \mathsf { N } } - C _ { U } E _ { U } ( W ) \} . } \end{array}
$$

This proves (C.11).

We next bound the change in the loss gradient by a multiple of $\bar { s } ( W )$ . Together with (C.6), this estimate will give bounds on the momentum error and the Taylor remainder in terms of the excess loss.

Lemma C.7. Let U befixed, and let N satisfy (B.16). There is a constant $L _ { 1 , U } > 0$ , depending only on $Q _ { U }$ , such that, for all W, $W ^ { \prime } \in \mathbb { R } ^ { d \times d }$

$$
\begin{array} { r } { \mathsf { N } _ { * } ( \nabla _ { W } \mathcal { L } _ { U } ( W ^ { \prime } ) - \nabla _ { W } \mathcal { L } _ { U } ( W ) ) \le L _ { 1 , U } ( e ^ { 2 \mathsf { N } ( W ^ { \prime } - W ) } - 1 ) \bar { s } ( W ) . } \end{array}
$$

Proof. Fix $n \in [ N ]$ , and let $a : = a _ { n } ( W ) , a ^ { \prime } : = a _ { n } ( W ^ { \prime } )$ , and $u : = e ^ { 2 \mathsf { N } ( W ^ { \prime } - W ) } - 1$ . Since $\| W ^ { \prime } - W \| _ { \mathrm { o p } } \leq \mathsf { N } ( W ^ { \prime } - W )$ Lemma B.9 gives

$$
| a _ { x } ^ { \prime } - a _ { x } | \leq u a _ { x } , \qquad \| a ^ { \prime } - a \| _ { 1 } \leq 2 u s _ { n } ( W ) .
$$

Also,

$$
\| m ( a ^ { \prime } ) - m ( a ) \| _ { 2 } \leq \sum _ { x \in \mathcal { V } } | a _ { x } ^ { \prime } - a _ { x } | = \| a ^ { \prime } - a \| _ { 1 } .
$$

To bound the probability change, let $a _ { \tau } : = ( 1 - \tau ) a + \tau a ^ { \prime }$ for $\tau \in [ 0 , 1 ]$ . The softmax Jacobian along this segment is

$$
J _ { \tau } : = \mathrm { d i a g } ( p _ { n } ( a _ { \tau } ) ) - p _ { n } ( a _ { \tau } ) p _ { n } ( a _ { \tau } ) ^ { \top } .
$$

It satisfies $J _ { \tau } \mathbf { 1 } = 0$ and $0 \preceq J _ { \tau } \preceq I _ { K } , \operatorname { s o } \| J _ { \tau } \| _ { \mathrm { o p } } \leq 1$ . Integration along the segment therefore gives

$$
\begin{array} { r l r } {  { \| p _ { n } ( a ^ { \prime } ) - p _ { n } ( a ) \| _ { 2 } = \| \int _ { 0 } ^ { 1 } J _ { \tau } C U ( m ( a ^ { \prime } ) - m ( a ) ) d \tau \| _ { 2 } } } \\ & { } & { \leq Q _ { U } \| m ( a ^ { \prime } ) - m ( a ) \| _ { 2 } } \\ & { } & { \leq Q _ { U } \| a ^ { \prime } - a \| _ { 1 } . } \end{array}
$$

For this sample, let $d _ { x } : = d _ { n , x } ^ { U } ( a )$ and $d _ { x } ^ { \prime } : = d _ { n , x } ^ { U } ( a ^ { \prime } )$ . The formula for $d _ { n , x } ^ { U }$ gives

$$
\begin{array} { r } { d _ { x } ^ { \prime } - d _ { x } = \left( p _ { n } ( a ^ { \prime } ) - p _ { n } ( a ) \right) ^ { \top } C U ( x - m ( a ^ { \prime } ) ) - ( p _ { n } ( a ) - e _ { y _ { n } } ) ^ { \top } C U ( m ( a ^ { \prime } ) - m ( a ) ) . } \end{array}
$$

Using $\| x - m ( a ^ { \prime } ) \| _ { 2 } \leq 2$ and $\| p _ { n } ( a ) - e _ { y _ { n } } \| _ { 2 } \leq \sqrt { 2 } .$ , we obtain

$$
\begin{array} { r l } & { | d _ { x } ^ { \prime } - d _ { x } | \leq 2 Q _ { U } \| p _ { n } ( a ^ { \prime } ) - p _ { n } ( a ) \| _ { 2 } + \sqrt { 2 } Q _ { U } \| m ( a ^ { \prime } ) - m ( a ) \| _ { 2 } } \\ & { \qquad \leq ( 2 Q _ { U } ^ { 2 } + \sqrt { 2 } Q _ { U } ) \| a ^ { \prime } - a \| _ { 1 } . } \end{array}
$$

We now use the identity

$$
a _ { x } ^ { \prime } d _ { x } ^ { \prime } - a _ { x } d _ { x } = ( a _ { x } ^ { \prime } - a _ { x } ) d _ { x } ^ { \prime } + a _ { x } ( d _ { x } ^ { \prime } - d _ { x } ) .
$$

The bound $| d _ { x } ^ { \prime } | \leq 2 Q _ { U }$ and the attention estimates above yield

$$
\begin{array} { r l r } {  { \sum _ { x \in \mathcal { X } _ { n } ^ { - } } | a _ { x } ^ { \prime } d _ { x } ^ { \prime } - a _ { x } d _ { x } | \le 2 Q _ { U } u s _ { n } ( W ) + ( 2 Q _ { U } ^ { 2 } + \sqrt { 2 } Q _ { U } ) s _ { n } ( W ) \| a ^ { \prime } - a \| _ { 1 } } } \\ & { } & \\ & { } & { \le 2 Q _ { U } u s _ { n } ( W ) + 2 ( 2 Q _ { U } ^ { 2 } + \sqrt { 2 } Q _ { U } ) u s _ { n } ( W ) ^ { 2 } } \\ & { } & \\ & { } & { \le 2 Q _ { U } ( 1 + 2 Q _ { U } + \sqrt { 2 } ) u s _ { n } ( W ) , } \end{array}
$$

where the last inequality uses $s _ { n } ( W ) \leq 1$

Let

$$
L _ { 1 , U } : = \operatorname* { m a x } \{ 1 , 2 \sqrt { 2 } Q _ { U } ( 1 + 2 Q _ { U } + \sqrt { 2 } ) \} .
$$

Applying the dual norm triangle inequality to (C.12) and using $\mathsf { N } _ { * } ( A _ { n , x } ) = \sqrt { 2 }$ , we obtain

$$
\mathsf { N } _ { * } ( \nabla _ { W } \mathcal { L } _ { U } ( W ^ { \prime } ) - \nabla _ { W } \mathcal { L } _ { U } ( W ) ) \le \frac { L _ { 1 , U } u } { N } \sum _ { n = 1 } ^ { N } s _ { n } ( W ) = L _ { 1 , U } u \bar { s } ( W ) .
$$

Substituting the definition of u proves the result.

Loss decrease and margin convergence. For ${ \mathsf { N } } = \| \cdot \| _ { S _ { p } }$ , let $\gamma _ { 2 , p } : = \gamma _ { 2 , \mathsf { N } }$ and $c _ { p } : = c _ { \mathsf { N } }$ . Equation (B.18) gives $\gamma _ { 2 , p } > 0$ For $\alpha > 0$ , let

$$
C _ { D , U , \alpha } : = \frac { L _ { 1 , U } [ 2 C ( \alpha , 2 , \beta _ { 2 } ) + e ^ { 2 } ] } { \kappa _ { - } } , \qquad \bar { \eta } _ { U , p , \alpha } : = \operatorname * { m i n } \left\{ 1 , \frac { c _ { p } } { 2 C _ { D , U , \alpha } } \right\} .
$$

The next lemma proves loss decrease under this step size bound and gives a recurrence for the excess loss. We will use this recurrence to derive the loss and margin estimates below.

Lemma C.8. Suppose that U satisfies (C.5). Consider Phase II of Algorithm C.1 with any exact maximizing directions and

$$
\eta _ { 2 , t } = \eta _ { 2 , 0 } ( t + 1 ) ^ { - \alpha } , \qquad \alpha > 0 , \qquad 0 < \eta _ { 2 , 0 } \leq \bar { \eta } _ { U , p , \alpha } .
$$

Let $\begin{array} { r } { E _ { t } : = E _ { U } ( W _ { 2 , t } ) , S _ { t } : = \sum _ { s = 0 } ^ { t - 1 } \eta _ { 2 , s } , } \end{array}$ and $\begin{array} { r } { Q _ { t } : = \sum _ { s = 0 } ^ { t - 1 } \eta _ { 2 , s } ^ { 2 } } \end{array}$ . Then, for every $t \geq 0 ,$

$$
\begin{array} { r l } & { \mathcal { L } _ { U } ( W _ { 2 , t } ) \leq L _ { 0 } , \qquad a _ { n , c } ( W _ { 2 , t } ) \geq \delta _ { 0 } \quad ( n \in [ N ] ) , } \\ & { 0 < E _ { t + 1 } \leq \left[ 1 - \eta _ { 2 , t } \operatorname* { m a x } \{ c _ { p } , \gamma _ { 2 , p } - C _ { U } E _ { t } \} + C _ { D , U , \alpha } \eta _ { 2 , t } ^ { 2 } \right] E _ { t } . } \end{array}\tag{C.13}
$$

Proof. For this proof, let $W _ { t } : = W _ { 2 , t } , \eta _ { t } : = \eta _ { 2 , t }$ , and $G _ { t } : = \nabla _ { W } \mathcal { L } _ { U } ( W _ { t } )$ . Let

$$
\widehat { M } _ { t } : = \frac { M _ { 2 , t } } { 1 - \beta _ { 2 } ^ { t + 1 } } , \qquad \Omega _ { t } : = \widehat { M } _ { t } - G _ { t } .
$$

Since $\| D _ { 2 , s } \| _ { S _ { p } } \leq 1$ , the updates give

$$
\Vert W _ { t } - W _ { t - r } \Vert _ { S _ { p } } \leq \sum _ { j = 1 } ^ { r } \eta _ { t - j } , \qquad 0 \leq r \leq t .
$$

For $0 < \beta _ { 2 } < 1$ , let

$$
w _ { t , r } : = \frac { ( 1 - \beta _ { 2 } ) \beta _ { 2 } ^ { r } } { 1 - \beta _ { 2 } ^ { t + 1 } } , \qquad 0 \leq r \leq t .
$$

Expanding the momentum recursion gives $\begin{array} { r } { \Omega _ { t } = \sum _ { r = 0 } ^ { t } w _ { t , r } ( G _ { t - r } - G _ { t } ) } \end{array}$ . Lemmas C.7 and B.12 then yield

$$
\begin{array} { l } { \displaystyle \| \Omega _ { t } \| _ { S _ { p ^ { \prime } } } \leq L _ { 1 , U } \bar { s } ( W _ { t } ) \displaystyle \sum _ { r = 0 } ^ { t } w _ { t , r } \left[ \exp \left( 2 \displaystyle \sum _ { j = 1 } ^ { r } \eta _ { t - j } \right) - 1 \right] } \\ { \leq L _ { 1 , U } C ( \alpha , 2 , \beta _ { 2 } ) \eta _ { t } \bar { s } ( W _ { t } ) . } \end{array}
$$

When $\beta _ { 2 } = 0$ , we have $\Omega _ { t } = 0 .$ , so the same bound holds.

The direction $D _ { 2 , t }$ maximizes the inner product with $\widehat { M } _ { t }$ over the Schatten $S _ { p }$ unit ball. As in the proof of Lemma B.14, norm duality gives

$$
\langle G _ { t } , D _ { 2 , t } \rangle = \| \widehat { M } _ { t } \| _ { S _ { p ^ { \prime } } } - \langle \Omega _ { t } , D _ { 2 , t } \rangle \geq \| G _ { t } \| _ { S _ { p ^ { \prime } } } - 2 \| \Omega _ { t } \| _ { S _ { p ^ { \prime } } } .
$$

Since $\eta _ { t } \leq 1$ , integration along the update segment and the momentum error bound imply

$$
\begin{array} { r l } & { E _ { t + 1 } - E _ { t } \leq - \eta _ { t } \langle G _ { t } , D _ { 2 , t } \rangle + L _ { 1 , U } \bar { s } ( W _ { t } ) \int _ { 0 } ^ { \eta _ { t } } ( e ^ { 2 \tau } - 1 ) d \tau } \\ & { \qquad \leq - \eta _ { t } \langle G _ { t } , D _ { 2 , t } \rangle + e ^ { 2 } L _ { 1 , U } \eta _ { t } ^ { 2 } \bar { s } ( W _ { t } ) } \\ & { \qquad \leq - \eta _ { t } \| G _ { t } \| _ { S _ { p ^ { \prime } } } + L _ { 1 , U } [ 2 C ( \alpha , 2 , \beta _ { 2 } ) + e ^ { 2 } ] \eta _ { t } ^ { 2 } \bar { s } ( W _ { t } ) } \\ & { \qquad = - \eta _ { t } \| G _ { t } \| _ { S _ { p ^ { \prime } } } + \kappa _ { - } C _ { D , U , \alpha } \eta _ { t } ^ { 2 } \bar { s } ( W _ { t } ) . } \end{array}
$$

We now prove $\mathcal { L } _ { U } ( W _ { t } ) \leq L _ { 0 }$ by induction. The claim holds at $t = 0$ because $W _ { 0 } = 0$ . Suppose that $\mathcal { L } _ { U } ( W _ { t } ) \leq L _ { 0 }$ Lemma C.5 gives $a _ { n , c } ( W _ { t } ) \geq \delta _ { 0 }$ for all $n \in [ N ]$ , so (C.11) applies. Using this gradient lower bound and $\kappa _ { - } \bar { s } ( W _ { t } ) \leq E _ { t }$ we obtain

$$
E _ { t + 1 } - E _ { t } \leq - \eta _ { t } E _ { t } \operatorname* { m a x } \{ c _ { p } , \gamma _ { 2 , p } - C _ { U } E _ { t } \} + C _ { D , U , \alpha } \eta _ { t } ^ { 2 } E _ { t } .
$$

This is the upper bound in (C.13). The step size condition gives

$$
\operatorname* { m a x } \{ c _ { p } , \gamma _ { 2 , p } - C _ { U } E _ { t } \} - C _ { D , U , \alpha } \eta _ { t } \geq c _ { p } - C _ { D , U , \alpha } \bar { \eta } _ { U , p , \alpha } \geq \frac { c _ { p } } { 2 } .
$$

Consequently,

$$
E _ { t } - E _ { t + 1 } \geq \frac { c _ { p } } { 2 } \eta _ { t } E _ { t } \geq 0 .
$$

Since $E _ { t } = \mathcal { L } _ { U } ( W _ { t } ) - \ell _ { I 7 } ^ { \star }$ , this implies $\mathcal { L } _ { U } ( W _ { t + 1 } ) \leq \mathcal { L } _ { U } ( W _ { t } ) \leq L _ { 0 }$ and closes the induction. The attention lower bound therefore holds at every iteration. Every finite iterate is a finite matrix, so (C.6) also gives $E _ { t } > 0$

Finally, summing the loss decrease yields

$$
{ \frac { c _ { p } } { 2 } } \sum _ { s = 0 } ^ { t - 1 } \eta _ { 2 , s } E _ { s } \leq \sum _ { s = 0 } ^ { t - 1 } ( E _ { s } - E _ { s + 1 } ) = E _ { 0 } - E _ { t } \leq E _ { 0 } .
$$

The next lemma uses the bound on $\textstyle \sum _ { s = 0 } ^ { t - 1 } \eta _ { 2 , s } E _ { s }$ to obtain a loss estimate with leading term $- \gamma _ { 2 , p } S _ { t }$ . Equation (C.8) and an upper bound on the parameter norm then give the margin estimates.

Lemma C.9. Under the hypotheses of Lemma C.8, there are $C _ { L , U } , C _ { \rho , U } > 0 ,$ , independent of $\eta _ { 2 , 0 }$ and the Phase II maximizing directions, such that, for every integer $t \geq 0$

$$
\begin{array} { r l } & { \log E _ { t } \leq - \gamma _ { 2 , p } S _ { t } + C _ { L , U } Q _ { t } + C _ { L , U } , } \\ & { \rho _ { 2 } ( W _ { 2 , t } ) \geq \gamma _ { 2 , p } S _ { t } - C _ { L , U } Q _ { t } - C _ { \rho , U } . } \end{array}
$$

For every integer $t \geq 1$ satisfying $\gamma _ { 2 , p } S _ { t } - C _ { L , U } Q _ { t } - C _ { \rho , U } > 0 ,$ , we have $W _ { 2 , t } \neq 0$ and

$$
0 \leq \gamma _ { 2 , p } - \rho _ { 2 } \left( \frac { W _ { 2 , t } } { \| W _ { 2 , t } \| _ { S _ { p } } } \right) \leq \frac { C _ { L , U } Q _ { t } + C _ { \rho , U } } { S _ { t } } .\tag{C.14}
$$

Proof. Since $E _ { t } , E _ { t + 1 } > 0$ , the factor multiplying $E _ { t }$ in (C.13) is positive. Taking logarithms and using $\log ( 1 + z ) \leq z$ give

$$
\begin{array} { r l } & { \log E _ { t + 1 } - \log E _ { t } \leq - \eta _ { 2 , t } \operatorname* { m a x } \{ c _ { p } , \gamma _ { 2 , p } - C _ { U } E _ { t } \} + C _ { D , U , \alpha } \eta _ { 2 , t } ^ { 2 } } \\ & { \qquad \leq - \gamma _ { 2 , p } \eta _ { 2 , t } + C _ { U } \eta _ { 2 , t } E _ { t } + C _ { D , U , \alpha } \eta _ { 2 , t } ^ { 2 } . } \end{array}
$$

Summing over $s = 0 , \ldots , t - 1$ , we obtain

$$
\begin{array} { l } { { \displaystyle \log E _ { t } \leq \log E _ { 0 } - \gamma _ { 2 , p } S _ { t } + C _ { U } \sum _ { s = 0 } ^ { t - 1 } \eta _ { 2 , s } E _ { s } + C _ { D , U , \alpha } Q _ { t } } } \\ { { \displaystyle \qquad \leq - \gamma _ { 2 , p } S _ { t } + C _ { D , U , \alpha } Q _ { t } + \log E _ { 0 } + \frac { 2 C _ { U } E _ { 0 } } { c _ { p } } } . } \end{array}
$$

The last inequality uses ${ \textstyle \sum _ { s = 0 } ^ { t - 1 } \eta _ { 2 , s } E _ { s } \le 2 E _ { 0 } / c _ { p } }$ from Lemma C.8. Let

$$
C _ { L , U } : = \operatorname* { m a x } \left\{ 1 , C _ { D , U , \alpha } , | \log E _ { 0 } | + \frac { 2 C _ { U } E _ { 0 } } { c _ { p } } \right\} .
$$

This choice proves the loss estimate.

Let $C _ { \rho , U } : = C _ { L , U } + \vert \log \nu _ { U } \vert . \mathrm { { B y } \left( C . 8 \right) }$

$$
\begin{array} { r l } & { \rho _ { 2 } ( W _ { 2 , t } ) \geq \log \nu _ { U } - \log E _ { t } } \\ & { \qquad \geq \gamma _ { 2 , p } S _ { t } - C _ { L , U } Q _ { t } - C _ { L , U } + \log \nu _ { U } } \\ & { \qquad \geq \gamma _ { 2 , p } S _ { t } - C _ { L , U } Q _ { t } - C _ { \rho , U } . } \end{array}
$$

Since $E _ { 0 } = E _ { U } ( 0 )$ , the quantities defining $\mathit { C } _ { \mathit { L } , \mathit { U } }$ and $C _ { \rho , U }$ are independent of $\eta _ { 2 , 0 }$ and the Phase II maximizing directions. Summing the updates from $W _ { 2 , 0 } = 0$ gives

$$
\| W _ { 2 , t } \| _ { S _ { p } } \leq \sum _ { s = 0 } ^ { t - 1 } \eta _ { 2 , s } \| D _ { 2 , s } \| _ { S _ { p } } \leq S _ { t } .
$$

I $\mathrm { ~ f ~ } \gamma _ { 2 , p } S _ { t } - C _ { L , U } Q _ { t } - C _ { \rho , U } > 0$ , then $\rho _ { 2 } ( W _ { 2 , t } ) > 0$ and hence $W _ { 2 , t } \neq 0$ . Positive homogeneity and the norm upper bound therefore yield

$$
\rho _ { 2 } \left( \frac { W _ { 2 , t } } { \| W _ { 2 , t } \| _ { S _ { p } } } \right) = \frac { \rho _ { 2 } ( W _ { 2 , t } ) } { \| W _ { 2 , t } \| _ { S _ { p } } } \geq \frac { \rho _ { 2 } ( W _ { 2 , t } ) } { S _ { t } } \geq \gamma _ { 2 , p } - \frac { C _ { L , U } Q _ { t } + C _ { \rho , U } } { S _ { t } } .
$$

On the other hand, the maximum margin identity (B.18) gives

$$
\rho _ { 2 } \left( \frac { W _ { 2 , t } } { \| W _ { 2 , t } \| _ { S _ { p } } } \right) \leq \operatorname* { m a x } _ { \| V \| _ { S _ { p } } \leq 1 } \rho _ { 2 } ( V ) = \gamma _ { 2 , p } .
$$

Combining these bounds proves (C.14).

We now combine the preprocessing and main phase estimates.

Theorem C.1. Fix $1 \leq p <$ ∞ and allow any exact maximizing directions in both phases of Algorithm C.1. Suppose that $K \geq 2 ,$ h is bijective, the vocabulary embeddings are orthonormal, Assumptions 1, 2, and 3 hold, and $\mathcal { T } _ { 2 } \neq \mathcal { O }$ . Let $\beta _ { 1 } , \beta _ { 2 } \in [ 0 , 1 )$ and use

$$
\eta _ { 1 , s } = \eta _ { 1 , 0 } ( s + 1 ) ^ { - 1 / 2 } , \qquad \eta _ { 2 , t } = \eta _ { 2 , 0 } ( t + 1 ) ^ { - 1 / 2 } ,
$$

with $\eta _ { 1 , 0 } > 0$ . There is an integer $T _ { p } \geq 1$ , independent ofthe Phase I maximizing directions and the ambient dimension, such that every $T \ \geq \ T _ { p }$ produces $U : = W _ { 1 , T }$ satisfying (C.5). $I f 0 < \eta _ { 2 , 0 } \leq \bar { \eta } _ { U , p , 1 / 2 } ,$ , then there are constants $C _ { 1 , U } , C _ { 2 , U } , C _ { 3 , U } , C _ { 4 , U } > 0 ,$ , independent ofthe Phase II maximizing directions, such that thefollowing bounds hold.

(i) Training loss. For every integer $t \geq 1$

$$
0 < E _ { U } ( W _ { 2 , t } ) \leq C _ { 1 , U } t ^ { C _ { 2 , U } } e ^ { - C _ { 3 , U } \sqrt { t } } .\tag{C.15}
$$

In particular, $\mathcal { L } _ { U } ( W _ { 2 , t } ) \to \ell _ { U } ^ { \star }$

(ii) Normalized Schatten margin. There is an integer $t _ { U } \geq 1$ such that, for every $t \geq t _ { U }$ , we have $W _ { 2 , t } \neq 0$ and

$$
0 \leq \gamma _ { 2 , p } - \rho _ { 2 } \left( \frac { W _ { 2 , t } } { \| W _ { 2 , t } \| _ { S _ { p } } } \right) \leq C _ { 4 , U } \frac { 1 + \log t } { \sqrt { t } } .\tag{C.16}
$$

Proof. Lemma C.3 gives $\delta _ { T } / S _ { 1 , T }  \gamma _ { 1 , p } > 0$ . Since $S _ { 1 , T } \geq \eta _ { 1 , 0 } \sqrt { T }$ , for all sufficiently large T we have

$$
\delta _ { T } \geq \frac { \gamma _ { 1 , p } } { 2 } S _ { 1 , T } \geq \frac { \gamma _ { 1 , p } \eta _ { 1 , 0 } } { 2 } \sqrt { T } \longrightarrow \infty .
$$

The same lemma also gives

$$
\frac { \| R _ { T } \| _ { \mathrm { o p } } } { \delta _ { T } } \leq \frac { \| R _ { T } \| _ { S _ { p } } } { \delta _ { T } } \longrightarrow 0 .
$$

Consequently,

$$
{ \frac { 2 ( K - 1 ) [ N \log K + \sqrt { 2 } ( N + 1 ) \| R _ { T } \| _ { \mathrm { o p } } ] } { \delta _ { T } } } \longrightarrow 0 .
$$

We can therefore choose an integer $T _ { p } \geq 1$ such that $\delta _ { T } > 0$ and this ratio is at most one for every $T \geq T _ { p }$ . This is precisely (C.5). The preprocessing bounds are uniform over the Phase I maximizing directions and independent of the ambient dimension, so $T _ { p }$ has the stated independence.

Fix such a T and let $U : = W _ { 1 , T }$ . Lemmas C.8 and C.9 apply with $\alpha = 1 / 2$ . For $t \geq 1$ , integral comparison gives

$$
S _ { t } \geq 2 \eta _ { 2 , 0 } ( \sqrt { t + 1 } - 1 ) , \qquad Q _ { t } \leq \eta _ { 2 , 0 } ^ { 2 } ( 1 + \log t ) .
$$

Substituting into the logarithmic loss estimate yields

$$
\begin{array} { r l } & { \log E _ { t } \leq - \gamma _ { 2 , p } S _ { t } + C _ { L , U } Q _ { t } + C _ { L , U } } \\ & { \qquad \leq - 2 \gamma _ { 2 , p } \eta _ { 2 , 0 } ( \sqrt { t + 1 } - 1 ) + C _ { L , U } \eta _ { 2 , 0 } ^ { 2 } ( 1 + \log t ) + C _ { L , U } } \\ & { \qquad \leq - 2 \gamma _ { 2 , p } \eta _ { 2 , 0 } \sqrt { t } + C _ { L , U } \eta _ { 2 , 0 } ^ { 2 } \log t + C _ { L , U } ( 1 + \eta _ { 2 , 0 } ^ { 2 } ) + 2 \gamma _ { 2 , p } \eta _ { 2 , 0 } . } \end{array}
$$

Let

$$
C _ { 1 , U } : = \exp \left( C _ { L , U } ( 1 + \eta _ { 2 , 0 } ^ { 2 } ) + 2 \gamma _ { 2 , p } \eta _ { 2 , 0 } \right) , \qquad C _ { 2 , U } : = C _ { L , U } \eta _ { 2 , 0 } ^ { 2 } , \qquad C _ { 3 , U } : = 2 \gamma _ { 2 , p } \eta _ { 2 , 0 } .
$$

Exponentiating the preceding inequality gives the upper bound in (C.15). Strict positivity follows from Lemma C.8.

Since $S _ { t } \to \infty$ and $Q _ { t } / S _ { t }  0$ , choose an integer $t _ { U } \geq 1$ such that

$$
\gamma _ { 2 , p } S _ { t } - C _ { L , U } Q _ { t } - C _ { \rho , U } > 0 , \qquad t \ge t _ { U } .
$$

Lemma C.9 then gives $W _ { 2 , t } \neq 0$ and (C.14) for every $t \geq t _ { U }$ . We also have $S _ { t } \geq \eta _ { 2 , 0 } \sqrt { t } .$ , since $k ^ { - 1 / 2 } \geq t ^ { - 1 / 2 }$ for $1 \leq k \leq t .$ Let

$$
C _ { 4 , U } : = \frac { C _ { L , U } \eta _ { 2 , 0 } ^ { 2 } + C _ { \rho , U } } { \eta _ { 2 , 0 } } .
$$

For $t \geq t _ { U }$ , we obtain

$$
\begin{array} { l } { \displaystyle \gamma _ { 2 , p } - \rho _ { 2 } \left( \frac { W _ { 2 , t } } { \| W _ { 2 , t } \| _ { S _ { p } } } \right) \leq \frac { C _ { L , U } Q _ { t } + C _ { \rho , U } } { S _ { t } } } \\ { \displaystyle \qquad \leq \frac { C _ { L , U } \eta _ { 2 , 0 } ^ { 2 } ( 1 + \log t ) + C _ { \rho , U } } { \eta _ { 2 , 0 } \sqrt { t } } } \\ { \displaystyle \qquad \leq C _ { 4 , U } \frac { 1 + \log t } { \sqrt { t } } . } \end{array}
$$

This proves (C.16). The constructions give the stated dependence of the constants.

Finally, $C _ { 3 , U } > 0$ implies $E _ { U } ( W _ { 2 , t } )  0 .$ , and hence $\mathcal { L } _ { U } ( W _ { 2 , t } ) \to \ell _ { U } ^ { \star }$ . Since $( 1 + \log t ) / \sqrt { t } \to 0$ , the normalized margin also converges to $\gamma _ { 2 , p } .$ □

We next give the normalized margin rates for general power law step sizes in Phase II.

Corollary C.1. Under the data and preprocessing conditions ofTheorem $C . I , f x T \geq T _ { p }$ and let $U : = W _ { 1 , T }$ . For Phase II, use

$$
\eta _ { 2 , t } = \eta _ { 2 , 0 } ( t + 1 ) ^ { - \alpha } , \qquad 0 < \alpha \leq 1 , \qquad 0 < \eta _ { 2 , 0 } \leq \bar { \eta } _ { U , p , \alpha } .
$$

For all sufficiently large t, we have $W _ { 2 , t } \neq 0 ,$ and the normalized margin satisfies thefour rates in (B.25), with $W _ { t } = W _ { 2 , t } ,$ $\mathsf { N } = \parallel \cdot \parallel _ { S _ { p } } , a n d \gamma _ { 2 , \mathsf { N } } = \gamma _ { 2 , p } .$

Proof. Lemmas C.8 and C.9 apply with the chosen exponent α. Since $S _ { t } \to \infty$ and $Q _ { t } / S _ { t }  0$ , the margin lower bound is eventually positive. Applying the scalar sum estimates used in the proof of Corollary B.2 to (C.14) gives the four stated rates. □

## D. Convergence of Adam

We analyze Adam with a fixed constant $\varepsilon > 0$ . Both phases use this constant outside the square root in the denominator. For phase $j \in \{ 1 , 2 \}$ , the parameters $\beta _ { j , 1 } , \beta _ { j , 2 } \in [ 0 , 1 )$ control the first and second moments, respectively. All squares, divisions, and square roots in the updates are computed separately for each matrix entry. Phase II holds the prediction matrix $W _ { 1 , T }$ fixed.

Algorithm D.1 Two Phase Adam Training   
1: PHASE I: Preprocessing   
2: Input $\mathcal { D } _ { 0 } ; T ; \{ \eta _ { 1 , s } \} _ { s = 0 } ^ { T - 1 } ; \beta _ { 1 , 1 } , \beta _ { 1 , 2 } \in [ 0 , 1 ) ; \varepsilon > 0 .$   
3: Initialization $W _ { 1 , 0 } = m _ { 1 , - 1 } = v _ { 1 , - 1 } = 0 .$   
4: for $s = 0 , \ldots , T - 1$ do   
5: $g _ { 1 , s } = \nabla { \mathcal { L } } _ { 0 } ( W _ { 1 , s } ) .$   
6: $m _ { 1 , s } = \beta _ { 1 , 1 } m _ { 1 , s - 1 } + ( 1 - \beta _ { 1 , 1 } ) g _ { 1 , s } .$   
7: $v _ { 1 , s } = \beta _ { 1 , 2 } v _ { 1 , s - 1 } + ( 1 - \beta _ { 1 , 2 } ) g _ { 1 , s } ^ { \odot 2 } .$   
8: $\widehat { m } _ { 1 , s } = m _ { 1 , s } / ( 1 - \beta _ { 1 , 1 } ^ { s + 1 } ) , \qquad \widehat { v } _ { 1 , s } = v _ { 1 , s } / ( 1 - \beta _ { 1 , 2 } ^ { s + 1 } ) .$   
9: $W _ { 1 , s + 1 } = W _ { 1 , s } - \eta _ { 1 , s } \widehat { m } _ { 1 , s } / ( \sqrt { \widehat { v } _ { 1 , s } } + \varepsilon ) .$   
10: end for   
11: PHASE II: Main Training   
12: Input $\mathcal { D } ; W _ { 1 , T } ; t ; \{ \eta _ { 2 , r } \} _ { r = 0 } ^ { t - 1 } ; \beta _ { 2 , 1 } , \beta _ { 2 , 2 } \in [ 0 , 1 )$   
13: Loss $\mathcal { L } _ { T } ( W ) : = \mathcal { L } ( W _ { 1 , T } , \dot { W } ) .$   
14: Initialization $W _ { 2 , 0 } = m _ { 2 , - 1 } = v _ { 2 , - 1 } = 0 .$   
15: for $r = 0 , \ldots , t - 1$ do   
16: $g _ { 2 , r } = \nabla \mathcal { L } _ { T } ( W _ { 2 , r } ) .$   
17: $m _ { 2 , r } = \beta _ { 2 , 1 } m _ { 2 , r - 1 } + ( 1 - \beta _ { 2 , 1 } ) g _ { 2 , r } .$   
18: $v _ { 2 , r } = \beta _ { 2 , 2 } v _ { 2 , r - 1 } + ( 1 - \beta _ { 2 , 2 } ) g _ { 2 , r } ^ { \odot 2 } .$   
19: $\widehat { m } _ { 2 , r } = m _ { 2 , r } / ( 1 - \beta _ { 2 , 1 } ^ { r + 1 } ) , \qquad \widehat { v } _ { 2 , r } = v _ { 2 , r } / ( 1 - \beta _ { 2 , 2 } ^ { r + 1 } ) .$   
20: $W _ { 2 , r + 1 } = W _ { 2 , r } - \eta _ { 2 , r } \widehat { m } _ { 2 , r } / ( \sqrt { \widehat { v } _ { 2 , r } } + \varepsilon ) .$   
21: end for   
22: Output $( W _ { 1 , T } , W _ { 2 , t } ) .$

## Preprocessing phase

Let $R _ { 1 , F } ^ { \star }$ be the optimal value of

$$
\begin{array} { r l } { \underset { W \in \mathbb { R } ^ { K \times d } } { \operatorname* { m i n } } } & { \| W \| _ { F } } \\ { \mathrm { s u b j e c t ~ t o } } & { ( e _ { h ( x ) } - e _ { v } ) ^ { \top } W x \ge 1 , } \\ & { x \in \mathcal { V } , \quad v \in [ K ] \setminus \{ h ( x ) \} . } \end{array}\tag{D.1}
$$

Use $\eta _ { 1 , s } = \eta _ { 1 , 0 } ( s + 1 ) ^ { - \alpha _ { 1 } }$ with $0 < \alpha _ { 1 } \le 1$ , and let $L _ { s } : = \mathcal { L } _ { 0 } ( W _ { 1 , s } )$ and $\begin{array} { r } { S _ { 1 , s } : = \sum _ { r = 0 } ^ { s - 1 } \eta _ { 1 , r } } \end{array}$ . For a matrix H, let $\| H \| _ { \operatorname* { m a x } } : = \operatorname* { m a x } _ { u , v } | H _ { u v } |$ . The next lemma compares the two moments with the current gradient. These estimates give loss decrease for a sufficiently small initial step size.

Lemma D.1. Suppose that $K \ \geq \ 2 ,$ , h is bijective, the vocabulary embeddings are orthonormal, and the K single token preprocessing samples are equally weighted. There are $C _ { 1 , m } , C _ { 1 , v } \ \geq \ 0$ and $\bar { \eta } _ { 1 , \mathrm { A } } \in ( 0 , 1 ] ,$ , depending only on $K , \alpha _ { 1 } , \beta _ { 1 , 1 } , \beta _ { 1 , 2 } , \varepsilon ,$ , such that, whenever $0 < \eta _ { 1 , 0 } \leq \bar { \eta } _ { 1 , \mathrm { A } }$ , thefollowing bounds holdfor every integer $s \geq 0$

$$
\begin{array} { r } { \| \widehat { m } _ { 1 , s } - \nabla \mathcal { L } _ { 0 } ( W _ { 1 , s } ) \| _ { F } \leq C _ { 1 , m } \eta _ { 1 , s } \| \nabla \mathcal { L } _ { 0 } ( W _ { 1 , s } ) \| _ { F } , } \end{array}
$$

$$
\begin{array} { r } { \| \sqrt { \widehat { v } _ { 1 , s } } \| _ { \operatorname* { m a x } } \leq C _ { 1 , v } \| \nabla \mathcal { L } _ { 0 } ( W _ { 1 , s } ) \| _ { F } . } \end{array}\tag{D.2}
$$

Moreover, there is $c _ { D } > 0$ such that

$$
L _ { s } \leq L _ { 0 } = \log K ,\tag{D.3}
$$

$$
L _ { s + 1 } \leq L _ { s } - c _ { D } \eta _ { 1 , s } \| \nabla \mathcal { L } _ { 0 } ( W _ { 1 , s } ) \| _ { F } ^ { 2 } ,\tag{D.4}
$$

$$
\frac { c _ { K } } { \sqrt { K } } L _ { s } \leq \| \nabla \mathcal { L } _ { 0 } ( W _ { 1 , s } ) \| _ { F } \leq \sqrt { 2 } L _ { s } ,\tag{D.5}
$$

where $c _ { K } : = ( 1 - e ^ { - K \log K } ) / ( K \log K ) . L e t c _ { L } : = c _ { D } c _ { K } ^ { 2 } / K .$

Proof. For this proof, let $G _ { s } : = \nabla \mathcal { L } _ { 0 } ( W _ { 1 , s } )$ and $y _ { i } : = h ( x _ { i } )$ , and use $p _ { i } ( W ) = \mathrm { s o f t m a x } ( W x _ { i } )$ . Let

$$
r _ { i } ( W ) : = p _ { i } ( W ) - e _ { y _ { i } } , \qquad q _ { i } ( W ) : = 1 - [ p _ { i } ( W ) ] _ { y _ { i } } .
$$

The probabilities of the other labels sum to $q _ { i } ( W )$ , so

$$
\| r _ { i } ( W ) \| _ { 2 } ^ { 2 } = q _ { i } ( W ) ^ { 2 } + \sum _ { v \ne y _ { i } } [ p _ { i } ( W ) ] _ { v } ^ { 2 } \le 2 q _ { i } ( W ) ^ { 2 } \le 2 .
$$

Orthonormality of the vocabulary gives

$$
\nabla \mathcal { L } _ { 0 } ( W ) = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } r _ { i } ( W ) x _ { i } ^ { \top } ,
$$

$$
\lVert \nabla \mathcal { L } _ { 0 } ( W ) \rVert _ { F } ^ { 2 } = \frac { 1 } { K ^ { 2 } } \sum _ { i = 1 } ^ { K } \lVert r _ { i } ( W ) \rVert _ { 2 } ^ { 2 } \leq \frac { 2 } { K } .
$$

Let $d _ { W } : = \| W ^ { \prime } - W \| _ { \mathrm { o p } }$ . Each output logit changes by at most $d _ { W } ,$ , so the softmax probability ratios lie in $[ e ^ { - 2 d _ { W } } , e ^ { 2 d _ { W } } ]$ For $v \neq y _ { i }$ , this gives

$$
| [ r _ { i } ( W ^ { \prime } ) ] _ { v } - [ r _ { i } ( W ) ] _ { v } | \leq ( e ^ { 2 d _ { W } } - 1 ) [ p _ { i } ( W ) ] _ { v } .
$$

For the correct label, we use $\begin{array} { r } { [ r _ { i } ( W ) ] _ { y _ { i } } = - \sum _ { v \ne y _ { i } } [ p _ { i } ( W ) ] _ { v } } \end{array}$ to obtain

$$
| [ r _ { i } ( W ^ { \prime } ) ] _ { y _ { i } } - [ r _ { i } ( W ) ] _ { y _ { i } } | \leq \sum _ { v \neq y _ { i } } | [ p _ { i } ( W ^ { \prime } ) ] _ { v } - [ p _ { i } ( W ) ] _ { v } | \leq ( e ^ { 2 d _ { W } } - 1 ) q _ { i } ( W ) .
$$

Squaring and summing these component bounds, followed by the gradient decomposition, yield

$$
\| \nabla \mathcal { L } _ { 0 } ( W ^ { \prime } ) - \nabla \mathcal { L } _ { 0 } ( W ) \| _ { F } ^ { 2 } = \frac { 1 } { K ^ { 2 } } \sum _ { i = 1 } ^ { K } \| r _ { i } ( W ^ { \prime } ) - r _ { i } ( W ) \| _ { 2 } ^ { 2 } \le ( e ^ { 2 d w } - 1 ) ^ { 2 } \| \nabla \mathcal { L } _ { 0 } ( W ) \| _ { F } ^ { 2 } .
$$

Taking square roots gives the relative gradient difference bound. The triangle inequality also gives

$$
\begin{array} { r } { \| \nabla \mathcal { L } _ { 0 } ( W ^ { \prime } ) \| _ { F } \leq \| \nabla \mathcal { L } _ { 0 } ( W ) \| _ { F } + \| \nabla \mathcal { L } _ { 0 } ( W ^ { \prime } ) - \nabla \mathcal { L } _ { 0 } ( W ) \| _ { F } \leq e ^ { 2 d _ { W } } \| \nabla \mathcal { L } _ { 0 } ( W ) \| _ { F } . } \end{array}
$$

We first establish the moment estimates for $0 < \eta _ { 1 , 0 } \leq 1$ . For $q \in \{ 1 , 2 \}$ , let

$$
w _ { s , r } ^ { ( q ) } : = \frac { ( 1 - \beta _ { 1 , q } ) \beta _ { 1 , q } ^ { r } } { 1 - \beta _ { 1 , q } ^ { s + 1 } } , \qquad 0 \leq r \leq s .
$$

These weights are nonnegative and sum to one. Expanding the moment recursions gives

$$
\widehat { m } _ { 1 , s } = \sum _ { r = 0 } ^ { s } w _ { s , r } ^ { ( 1 ) } G _ { s - r } , \qquad \widehat { v } _ { 1 , s } = \sum _ { r = 0 } ^ { s } w _ { s , r } ^ { ( 2 ) } G _ { s - r } ^ { \odot 2 } .
$$

When a moment parameter is zero, the corresponding sum contains only the term with $r = 0 .$ . Let

$$
D _ { 1 } : = \frac { \sqrt { 2 / K } } { \varepsilon } , \qquad \Pi _ { s , r } : = \sum _ { j = 1 } ^ { r } \eta _ { 1 , s - j } .
$$

The denominator lower bound and the global gradient bound imply

$$
\lVert W _ { 1 , s + 1 } - W _ { 1 , s } \rVert _ { F } \leq \frac { \eta _ { 1 , s } } { \varepsilon } \sum _ { r = 0 } ^ { s } w _ { s , r } ^ { ( 1 ) } \lVert G _ { s - r } \rVert _ { F } \leq D _ { 1 } \eta _ { 1 , s } .
$$

Summing the updates therefore gives

$$
\| W _ { 1 , s } - W _ { 1 , s - r } \| _ { \mathrm { o p } } \leq D _ { 1 } \Pi _ { s , r } .
$$

The relative gradient bounds now yield

$$
\Vert \widehat { m } _ { 1 , s } - G _ { s } \Vert _ { F } \leq \Vert G _ { s } \Vert _ { F } \sum _ { r = 0 } ^ { s } w _ { s , r } ^ { ( 1 ) } ( e ^ { 2 D _ { 1 } \Pi _ { s , r } } - 1 ) ,
$$

$$
\| \sqrt { \hat { v } _ { 1 , s } } \| _ { \operatorname* { m a x } } ^ { 2 } \leq \sum _ { r = 0 } ^ { s } w _ { s , r } ^ { ( 2 ) } \| G _ { s - r } \| _ { F } ^ { 2 } \leq \| G _ { s } \| _ { F } ^ { 2 } \sum _ { r = 0 } ^ { s } w _ { s , r } ^ { ( 2 ) } e ^ { 4 D _ { 1 } \Pi _ { s , r } } .
$$

Let

$$
C _ { 1 , m } : = C ( \alpha _ { 1 } , 2 D _ { 1 } , \beta _ { 1 , 1 } ) , \qquad C _ { 1 , v } : = \sqrt { 1 + C ( \alpha _ { 1 } , 4 D _ { 1 } , \beta _ { 1 , 2 } ) } .
$$

Lemma B.12 gives the first moment bound in (D.2). For the second moment, the same lemma and $\eta _ { 1 , s } \leq 1$ give

$$
\sum _ { r = 0 } ^ { s } w _ { s , r } ^ { ( 2 ) } e ^ { 4 D _ { 1 } \Pi _ { s , r } } = 1 + \sum _ { r = 0 } ^ { s } w _ { s , r } ^ { ( 2 ) } ( e ^ { 4 D _ { 1 } \Pi _ { s , r } } - 1 ) \le 1 + C ( \alpha _ { 1 } , 4 D _ { 1 } , \beta _ { 1 , 2 } ) \eta _ { 1 , s } \le C _ { 1 , v } ^ { 2 } .
$$

This proves (D.2) with constants independent of $\eta _ { 1 , 0 } \in ( 0 , 1 ]$

We next prove loss decrease. Let $P _ { s }$ be the linear operator

$$
P _ { s } Z : = ( \sqrt { \widehat { v } _ { 1 , s } } + \varepsilon ) ^ { - 1 } \odot Z ,
$$

and let $p _ { - } : = ( \varepsilon + \sqrt { 2 / K } ) ^ { - 1 }$ . Each entry of $\widehat { v } _ { 1 , s }$ is an average of squared gradient entries and is therefore at most $2 / K$ Hence

$$
p _ { - } \mathrm { I d } \preceq P _ { s } \preceq \varepsilon ^ { - 1 } \mathrm { I d } .
$$

For any $H \in \mathbb { R } ^ { K \times d }$ , the softmax Jacobian formula gives

$$
\begin{array} { r l r } {  { 0 \leq \nabla ^ { 2 } \mathcal { L } _ { 0 } ( W ) [ H , H ] = \frac { 1 } { K } \displaystyle \sum _ { i = 1 } ^ { K } ( H x _ { i } ) ^ { \top } ( \mathrm { d i a g } ( p _ { i } ( W ) ) - p _ { i } ( W ) p _ { i } ( W ) ^ { \top } ) ( H x _ { i } ) } } \\ & { } & { \leq \frac { 1 } { K } \displaystyle \sum _ { i = 1 } ^ { K } \| H x _ { i } \| _ { 2 } ^ { 2 } } \\ & { } & { \leq \frac { 1 } { K } \| H \| _ { F } ^ { 2 } . } \end{array}
$$

Thus the Hessian is bounded above by $K ^ { - 1 } \mathrm { { I d } }$ in the Frobenius inner product.

Choose $\bar { \eta } _ { 1 , \mathrm { A } } \in ( 0 , 1 ]$ sufficiently small that

$$
\frac { C _ { 1 , m } \bar { \eta } _ { 1 , \mathrm { A } } } { \varepsilon } \leq \frac { p _ { - } } { 2 } , \qquad \frac { ( 1 + C _ { 1 , m } ) ^ { 2 } \bar { \eta } _ { 1 , \mathrm { A } } } { 2 K \varepsilon ^ { 2 } } \leq \frac { p _ { - } } { 4 } .
$$

We also require

$$
\bar { \eta } _ { 1 , \mathrm { A } } \leq K \varepsilon ,\tag{D.6}
$$

which will be used in the comparison with gradient descent below. All quantities in these conditions are fixed independently of $\eta _ { 1 , 0 }$ , so such a positive threshold exists.

For $0 < \eta _ { 1 , 0 } \leq \bar { \eta } _ { 1 , \mathrm { A } }$ , the operator bounds and the first moment estimate imply

$$
\langle G _ { s } , P _ { s } \widehat { m } _ { 1 , s } \rangle = \langle G _ { s } , P _ { s } G _ { s } \rangle + \langle G _ { s } , P _ { s } ( \widehat { m } _ { 1 , s } - G _ { s } ) \rangle \geq \left( p _ { - } - \frac { C _ { 1 , m } \eta _ { 1 , s } } { \varepsilon } \right) \| G _ { s } \| _ { F } ^ { 2 } \geq \frac { p _ { - } } { 2 } \| G _ { s } \| _ { F } ^ { 2 } .
$$

Also,

$$
\| P _ { s } \widehat { m } _ { 1 , s } \| _ { F } \leq \frac { \| G _ { s } \| _ { F } + \| \widehat { m } _ { 1 , s } - G _ { s } \| _ { F } } { \varepsilon } \leq \frac { 1 + C _ { 1 , m } } { \varepsilon } \| G _ { s } \| _ { F } .
$$

Taylor’s theorem therefore gives

$$
\begin{array} { r l } & { L _ { s + 1 } \leq L _ { s } - \eta _ { 1 , s } \left. { G } _ { s } , P _ { s } \widehat { m } _ { 1 , s } \right. + \displaystyle \frac { \eta _ { 1 , s } ^ { 2 } } { 2 K } \| P _ { s } \widehat { m } _ { 1 , s } \| _ { F } ^ { 2 } } \\ & { \qquad \leq L _ { s } - \eta _ { 1 , s } \left( \displaystyle \frac { p _ { - } } { 2 } - \displaystyle \frac { ( 1 + C _ { 1 , m } ) ^ { 2 } } { 2 K \varepsilon ^ { 2 } } \eta _ { 1 , s } \right) \| G _ { s } \| _ { F } ^ { 2 } } \\ & { \qquad \leq L _ { s } - \displaystyle \frac { p _ { - } } { 4 } \eta _ { 1 , s } \| G _ { s } \| _ { F } ^ { 2 } . } \end{array}
$$

Taking $c _ { D } : = p _ { - } / 4$ proves (D.4). Induction from $W _ { 1 , 0 } = 0$ then gives $L _ { s } \leq L _ { 0 } = \log K$ , proving (D.3).

It remains to compare the gradient and the loss. At $W _ { 1 , s } ,$ let $\ell _ { i } : = - \log [ p _ { i } ( W _ { 1 , s } ) ] _ { y _ { i } }$ and $q _ { i } : = 1 - e ^ { - \ell _ { i } }$ . The loss bound gives $0 < \ell _ { i } \le K \log K$ . Concavity of $1 - e ^ { - u }$ on [0, K log K] and $1 - e ^ { - u } \leq$ u imply

$$
\begin{array} { r } { c _ { K } \ell _ { i } \leq q _ { i } \leq \ell _ { i } . } \end{array}
$$

The residual formula above also gives $q _ { i } \le \| r _ { i } ( W _ { 1 , s } ) \| _ { 2 } \le \sqrt { 2 } q _ { i }$ <sub>i</sub>. Consequently,

$$
\| G _ { s } \| _ { F } \ge \frac { c _ { K } } { K } \left( \sum _ { i = 1 } ^ { K } \ell _ { i } ^ { 2 } \right) ^ { 1 / 2 } \ge \frac { c _ { K } } { \sqrt { K } } L _ { s } ,
$$

$$
\| G _ { s } \| _ { F } \leq \frac { \sqrt { 2 } } { K } \left( \sum _ { i = 1 } ^ { K } \ell _ { i } ^ { 2 } \right) ^ { 1 / 2 } \leq \sqrt { 2 } L _ { s } .
$$

This proves (D.5). The construction gives the stated dependence of all constants.

The next lemma gives the growth and limiting direction of the preprocessing iterates. We prove it by comparing Adam with gradient descent and showing that the sum of the update errors is finite.

Lemma D.2. Suppose that the hypotheses of Lemma D.1 hold. Then B is the unique minimizer of (D.1), and $R _ { 1 , F } ^ { \star } =$ $\sqrt { K - 1 }$ . For every $s \geq 0$

$$
\begin{array} { r } { 0 < L _ { s } \leq ( L _ { 0 } ^ { - 1 } + c _ { L } S _ { 1 , s } ) ^ { - 1 } . } \end{array}\tag{D.7}
$$

$$
A s \ s \to \infty ,
$$

$$
W _ { 1 , s } = ( \log S _ { 1 , s } ) B + O _ { F } ( 1 ) , \qquad C W _ { 1 , s } = ( \log S _ { 1 , s } ) B + O _ { F } ( 1 ) .\tag{D.8}
$$

Here $O _ { F } ( 1 )$ ) denotes a matrix term whose Frobenius norm remains bounded as $s  \infty ,$

In particular, $W _ { 1 , s }$ and $C W _ { 1 , s }$ are nonzero for all sufficiently large s. Their Frobenius normalizations converge to $B / \sqrt { K - 1 }$ with error $O ( ( \log S _ { 1 , s } ) ^ { - 1 } )$ .

Proof. For every feasible W, summing the margin constraints gives

$$
K \left. W , B \right. = \sum _ { i = 1 } ^ { K } \sum _ { v \neq y _ { i } } ( e _ { y _ { i } } - e _ { v } ) ^ { \top } W x _ { i } \geq K ( K - 1 ) .
$$

Since $\| B \| _ { F } = \sqrt { K - 1 }$ , the Cauchy-Schwarz inequality yields

$$
K - 1 \le \langle W , B \rangle \le \| W \| _ { F } \sqrt { K - 1 } .
$$

Thus $\| W \| _ { F } \ge \sqrt { K - 1 }$ . The matrix B is feasible and attains this bound. For any minimizer $W ,$ , we therefore have

$$
\| W - B \| _ { F } ^ { 2 } = \| W \| _ { F } ^ { 2 } + ( K - 1 ) - 2 \left. W , B \right. \le 0 .
$$

Hence $W = B ,$ proving uniqueness.

Equations (D.4) and (D.5) imply

$$
L _ { s + 1 } \leq L _ { s } - c _ { L } \eta _ { 1 , s } L _ { s } ^ { 2 } .
$$

Since $0 < L _ { s + 1 } \leq L _ { s }$

$$
L _ { s + 1 } ^ { - 1 } - L _ { s } ^ { - 1 } = \frac { L _ { s } - L _ { s + 1 } } { L _ { s } L _ { s + 1 } } \geq c _ { L } \eta _ { 1 , s } \frac { L _ { s } } { L _ { s + 1 } } \geq c _ { L } \eta _ { 1 , s } .
$$

Summing gives

$$
L _ { s } ^ { - 1 } \geq L _ { 0 } ^ { - 1 } + c _ { L } \sum _ { j = 0 } ^ { s - 1 } \eta _ { 1 , j } = L _ { 0 } ^ { - 1 } + c _ { L } S _ { 1 , s } ,
$$

which proves (D.7). Summing the loss decrease also yields

$$
c _ { L } \sum _ { j = 0 } ^ { s - 1 } \eta _ { 1 , j } L _ { j } ^ { 2 } \le \sum _ { j = 0 } ^ { s - 1 } ( L _ { j } - L _ { j + 1 } ) = L _ { 0 } - L _ { s } \le L _ { 0 } .
$$

Letting $s  \infty ,$ , we obtain

$$
\sum _ { s = 0 } ^ { \infty } \eta _ { 1 , s } L _ { s } ^ { 2 } \le \frac { L _ { 0 } } { c _ { L } } < \infty .\tag{D.9}
$$

To compare Adam with gradient descent, we use the operator $P _ { s }$ from the preceding proof. Let $G _ { s } : = \nabla \mathcal { L } _ { 0 } ( W _ { 1 , s } )$ and

$$
\xi _ { 1 , s } : = P _ { s } \widehat { m } _ { 1 , s } - \varepsilon ^ { - 1 } G _ { s } .
$$

We separate the first moment error from the change in the denominator:

$$
\xi _ { 1 , s } = P _ { s } ( \hat { m } _ { 1 , s } - G _ { s } ) + ( P _ { s } - \varepsilon ^ { - 1 } \mathrm { I d } ) G _ { s } .
$$

For $z \geq 0 ,$

$$
0 \leq \varepsilon ^ { - 1 } - ( \varepsilon + z ) ^ { - 1 } = \frac { z } { \varepsilon ( \varepsilon + z ) } \leq \frac { z } { \varepsilon ^ { 2 } } .
$$

Using (D.2) and (D.5), we obtain

$$
\begin{array} { r l } & { \| \xi _ { 1 , s } \| _ { F } \leq \varepsilon ^ { - 1 } \| \widehat { m } _ { 1 , s } - G _ { s } \| _ { F } + \varepsilon ^ { - 2 } \| \sqrt { \widehat { v } _ { 1 , s } } \| _ { \operatorname* { m a x } } \| G _ { s } \| _ { F } } \\ & { \qquad \leq \frac { C _ { 1 , m } } { \varepsilon } \eta _ { 1 , s } \| G _ { s } \| _ { F } + \frac { C _ { 1 , v } } { \varepsilon ^ { 2 } } \| G _ { s } \| _ { F } ^ { 2 } } \\ & { \qquad \leq \frac { \sqrt { 2 } C _ { 1 , m } } { \varepsilon } \eta _ { 1 , s } L _ { s } + \frac { 2 C _ { 1 , v } } { \varepsilon ^ { 2 } } L _ { s } ^ { 2 } . } \end{array}
$$

For $s \geq 1$ , we have $S _ { 1 , s } \geq \eta _ { 1 , 0 } s ^ { 1 - \alpha _ { 1 } }$ , so

$$
\eta _ { 1 , s } ^ { 2 } L _ { s } \leq \frac { \eta _ { 1 , 0 } ^ { 2 } ( s + 1 ) ^ { - 2 \alpha _ { 1 } } } { c _ { L } S _ { 1 , s } } \leq \frac { \eta _ { 1 , 0 } } { c _ { L } } s ^ { - 1 - \alpha _ { 1 } } .
$$

Since $\alpha _ { 1 } > 0$ , it follows that $\textstyle \sum _ { s = 0 } ^ { \infty } \eta _ { 1 , s } ^ { 2 } L _ { s } < \infty$ . Together with (D.9), this gives

$$
\begin{array} { r l } & { W _ { 1 , s + 1 } = W _ { 1 , s } - ( \eta _ { 1 , s } / \varepsilon ) G _ { s } - \eta _ { 1 , s } \xi _ { 1 , s } , } \\ & { \displaystyle \sum _ { s = 0 } ^ { \infty } \eta _ { 1 , s } \| \xi _ { 1 , s } \| _ { F } \leq \frac { \sqrt { 2 } C _ { 1 , m } } { \varepsilon } \displaystyle \sum _ { s = 0 } ^ { \infty } \eta _ { 1 , s } ^ { 2 } L _ { s } + \frac { 2 C _ { 1 , v } } { \varepsilon ^ { 2 } } \displaystyle \sum _ { s = 0 } ^ { \infty } \eta _ { 1 , s } L _ { s } ^ { 2 } < \infty . } \end{array}\tag{D.10}
$$

Let $\Phi _ { s } ( W ) : = W - ( \eta _ { 1 , s } / \varepsilon ) \nabla { \mathcal { L } } _ { 0 } ( W )$ , and let $Z _ { 0 } : = 0$ and $Z _ { s + 1 } : = \Phi _ { s } ( Z _ { s } )$ . The Hessian bound from the previous proof and (D.6) give

$$
0 \preceq D \Phi _ { s } ( W ) = \operatorname { I d } - \frac { \eta _ { 1 , s } } { \varepsilon } \nabla ^ { 2 } \mathcal { L } _ { 0 } ( W ) \preceq \operatorname { I d } .
$$

For any $W , Z \in \mathbb { R } ^ { K \times d }$ , integration along the segment from Z to W therefore yields

$$
\| \Phi _ { s } ( W ) - \Phi _ { s } ( Z ) \| _ { F } \le \int _ { 0 } ^ { 1 } \| D \Phi _ { s } ( Z + \tau ( W - Z ) ) [ W - Z ] \| _ { F } d \tau \le \| W - Z \| _ { F } .
$$

Using $W _ { 1 , s + 1 } = \Phi _ { s } ( W _ { 1 , s } ) - \eta _ { 1 , s } \xi _ { 1 , s } ,$ , we obtain

$$
\begin{array} { r } { \| W _ { 1 , s + 1 } - Z _ { s + 1 } \| _ { F } \leq \| W _ { 1 , s } - Z _ { s } \| _ { F } + \eta _ { 1 , s } \| \xi _ { 1 , s } \| _ { F } . } \end{array}
$$

Since $W _ { 1 , 0 } = Z _ { 0 } = 0$ , repeated application of this inequality gives

$$
\Vert W _ { 1 , s } - Z _ { s } \Vert _ { F } \leq \sum _ { j = 0 } ^ { s - 1 } \eta _ { 1 , j } \Vert \xi _ { 1 , j } \Vert _ { F } \leq \sum _ { j = 0 } ^ { \infty } \eta _ { 1 , j } \Vert \xi _ { 1 , j } \Vert _ { F } < \infty .\tag{D.11}
$$

The gradient computation in the proof of Lemma B.1 gives $\nabla { \mathcal { L } } _ { 0 } ( b B ) = - B / ( e ^ { b } + K - 1 )$ ). Induction therefore shows that $Z _ { s } = b _ { s } B$ , where $b _ { 0 } = 0$ and

$$
b _ { s + 1 } - b _ { s } = \frac { \eta _ { 1 , s } } { \varepsilon ( e ^ { b _ { s } } + K - 1 ) } .
$$

The increments are nonnegative, so $b _ { s } \geq 0 .$ . Let $u _ { s } : = b _ { s + 1 } - b _ { s }$ . Since $b _ { s } \geq 0$ and $\eta _ { 1 , s } \leq \eta _ { 1 , 0 } \leq \bar { \eta } _ { 1 , \mathrm { A } } \leq K \varepsilon$ , we have

$$
0 \leq u _ { s } = \frac { \eta _ { 1 , s } } { \varepsilon ( e ^ { b _ { s } } + K - 1 ) } \leq \frac { \eta _ { 1 , s } } { K \varepsilon } \leq 1 .
$$

Using $u \leq e ^ { u } - 1 \leq$ eu on [0, 1] and $1 / K \le e ^ { b _ { s } } / ( e ^ { b _ { s } } + K - 1 ) \le 1$ , we obtain

$$
\frac { \eta _ { 1 , s } } { K \varepsilon } \leq e ^ { b _ { s } } u _ { s } \leq e ^ { b _ { s + 1 } } - e ^ { b _ { s } } \leq e e ^ { b _ { s } } u _ { s } \leq \frac { e \eta _ { 1 , s } } { \varepsilon } .
$$

Summing and using $b _ { 0 } = 0$ give

$$
1 + \frac { S _ { 1 , s } } { K \varepsilon } \leq e ^ { b _ { s } } \leq 1 + \frac { e S _ { 1 , s } } { \varepsilon } .
$$

For $s \geq 1$ , dividing the previous bounds by $S _ { 1 , s }$ and taking logarithms give

$$
\log \left( \frac { 1 } { S _ { 1 , s } } + \frac { 1 } { K \varepsilon } \right) \leq b _ { s } - \log S _ { 1 , s } \leq \log \left( \frac { 1 } { S _ { 1 , s } } + \frac { e } { \varepsilon } \right) .
$$

Since $S _ { 1 , s }  \infty$ , both bounds converge to finite constants. Thus $b _ { s } - \log S _ { 1 , s } = O ( 1 )$

Using $Z _ { s } = b _ { s } B , ( \mathrm { D } . 1 1 )$ , and the triangle inequality, we obtain

$$
\begin{array} { r } { \| W _ { 1 , s } - ( \log S _ { 1 , s } ) B \| _ { F } \leq \| W _ { 1 , s } - Z _ { s } \| _ { F } + | b _ { s } - \log S _ { 1 , s } | \| B \| _ { F } = O ( 1 ) . } \end{array}
$$

This proves the first identity in (D.8).

Since $C B = B$ and multiplication by C does not increase the Frobenius norm,

$$
\begin{array} { r } { \| C W _ { 1 , s } - ( \log S _ { 1 , s } ) B \| _ { F } = \| C ( W _ { 1 , s } - ( \log S _ { 1 , s } ) B ) \| _ { F } \leq \| W _ { 1 , s } - ( \log S _ { 1 , s } ) B \| _ { F } = O ( 1 ) . } \end{array}
$$

This proves the second identity in (D.8).

For either matrix in (D.8), write $A _ { s } = a _ { s } B + H _ { s }$ , where $a _ { s } : = \log S _ { 1 , s }  \infty$ and $\| H _ { s } \| _ { F } = O ( 1 )$ . For all sufficiently large s, we have $a _ { s } > 0$ and

$$
\| A _ { s } \| _ { F } \geq a _ { s } \sqrt { K - 1 } - \| H _ { s } \| _ { F } > 0 .
$$

The triangle inequality and $\left| \Vert A _ { s } \Vert _ { F } - a _ { s } \sqrt { K - 1 } \right| \leq \Vert H _ { s } \Vert _ { F }$ then give

$$
\begin{array} { r l r } {  { \| \frac { A _ { s } } { \| A _ { s } \| _ { \cal F } } - \frac { B } { \sqrt { K - 1 } } \| _ { \cal F } \le \frac { \| H _ { s } \| _ { \cal F } + \big | \| A _ { s } \| _ { \cal F } - a _ { s } \sqrt { K - 1 } } { a _ { s } \sqrt { K - 1 } } } } \\ & { } & { \le \frac { 2 \| H _ { s } \| _ { \cal F } } { a _ { s } \sqrt { K - 1 } } . } \end{array}
$$

This proves the stated convergence and rate.

Consequently, for $0 < \alpha _ { 1 } < 1$ , we have $L _ { s } = O ( s ^ { - ( 1 - \alpha _ { 1 } ) } )$ , and the error after Frobenius normalization is $O ( ( \log ( s + 1 ) ) ^ { - 1 } )$ . For $\alpha _ { 1 } = 1$ , we have $L _ { s } = O ( ( \log ( s + 1 ) ) ^ { - 1 } )$ , and the error after Frobenius normalization is $O ( ( \log \log s ) ^ { - 1 } )$

## Main phase

Under the data assumptions of Theorem C.1, fix a prediction matrix $U$ satisfying (C.5). We use the notation ${ \mathcal { L } } _ { U } ( W ) =$ ${ \mathcal { L } } ( U , W )$ and $E _ { U } ( W ) = \mathcal { L } _ { U } ( W ) - \ell _ { U } ^ { \star }$ . Let $R _ { 2 , F } ^ { \star }$ be the optimal value of (B.17) for $\mathsf { N } = \parallel \cdot \parallel _ { F }$ , and let $W _ { F } ^ { \star }$ be a minimizer. The minimizer is unique because the feasible set is convex and the squared Frobenius norm is strictly convex. Let $\gamma _ { 2 , F } : = 1 / R _ { 2 , F } ^ { \star }$

Use $\eta _ { 2 , t } = \eta _ { 2 , 0 } ( t + 1 ) ^ { - \alpha _ { 2 } }$ with $0 < \alpha _ { 2 } \leq 1$ , and let

$$
E _ { t } : = E _ { U } ( W _ { 2 , t } ) , \qquad L _ { 0 } : = \mathcal { L } _ { U } ( 0 ) , \qquad S _ { t } : = \sum _ { s = 0 } ^ { t - 1 } \eta _ { 2 , s } .
$$

The next lemma bounds the two moments and proves that the training loss never exceeds its initial value. It also gives an excess loss bound and shows that the total attention weight on the correct token tends to one for every sample.

Lemma D.3. There are constants $C _ { m } , C _ { v } \geq 0 \mathrm { { ; } }$ , depending only on $U ,$ , thefinite training data, $\alpha _ { 2 } , \beta _ { 2 , 1 } , \beta _ { 2 , 2 } , \varepsilon ,$ , such that, for every $0 < \eta _ { 2 , 0 } \leq 1$ and every integer $t \geq 0$

$$
\| \widehat { m } _ { 2 , t } - \nabla { \mathcal { L } } _ { U } ( W _ { 2 , t } ) \| _ { F } \leq C _ { m } \eta _ { 2 , t } \bar { s } ( W _ { 2 , t } ) , \qquad \| \sqrt { \bar { v } _ { 2 , t } } \| _ { \operatorname* { m a x } } \leq C _ { v } \bar { s } ( W _ { 2 , t } ) .\tag{D.12}
$$

Here $C _ { m } = 0 \ i f \beta _ { 2 , 1 } = 0 .$ . Moreover, there are $\bar { \eta } _ { U } \in ( 0 , 1 ]$ and $c _ { D } , c _ { R } > 0$ such that, whenever $0 < \eta _ { 2 , 0 } \leq \bar { \eta } _ { U }$ , we have $\mathcal { L } _ { U } ( W _ { 2 , t } ) \leq L _ { 0 } , a _ { n , c } ( W _ { 2 , t } ) \geq \delta _ { 0 }$ for all $n \in [ N ]$ , and

$$
\begin{array} { r l } & { E _ { t + 1 } \leq E _ { t } - c _ { D } \eta _ { 2 , t } \| \nabla \mathcal { L } _ { U } ( W _ { 2 , t } ) \| _ { F } ^ { 2 } , } \\ & { 0 < E _ { t } \leq ( E _ { 0 } ^ { - 1 } + c _ { R } S _ { t } ) ^ { - 1 } } \end{array}\tag{D.13}
$$

for every integer $t \geq 0 .$ . Furthermore, $a _ { n , c } ( W _ { 2 , t } ) \to 1$ for every $n \in [ N ]$

Proof. For this proof, let $W _ { t } : = W _ { 2 , t } , \eta _ { t } : = \eta _ { 2 , t }$ , and $G _ { t } : = \nabla \mathcal { L } _ { U } ( W _ { t } )$ . Let $C _ { 0 } : = 2 \sqrt { 2 } Q _ { U }$ and $D _ { 0 } : = C _ { 0 } / \varepsilon$ . Equation (C.10) gives $\| G _ { t } \| _ { F } \le C _ { 0 } \bar { s } ( W _ { t } ) \le C _ { 0 }$

For $q \in \{ 1 , 2 \}$ , let

$$
w _ { t , r } ^ { ( q ) } : = \frac { ( 1 - \beta _ { 2 , q } ) \beta _ { 2 , q } ^ { r } } { 1 - \beta _ { 2 , q } ^ { t + 1 } } , \qquad 0 \leq r \leq t .
$$

These weights are nonnegative and sum to one, and the moment recursions give

$$
\widehat { m } _ { 2 , t } = \sum _ { r = 0 } ^ { t } w _ { t , r } ^ { ( 1 ) } G _ { t - r } , \qquad \widehat { v } _ { 2 , t } = \sum _ { r = 0 } ^ { t } w _ { t , r } ^ { ( 2 ) } G _ { t - r } ^ { \odot 2 } .
$$

If $\beta _ { 2 , q } = 0$ , the corresponding sum contains only the term with $r = 0$ . Since every denominator is at least ε,

$$
\| W _ { t + 1 } - W _ { t } \| _ { F } \leq \frac { \eta _ { t } } { \varepsilon } \| \widehat { m } _ { 2 , t } \| _ { F } \leq D _ { 0 } \eta _ { t } .
$$

Thus, with $\textstyle \Pi _ { t , r } : = \sum _ { j = 1 } ^ { r } \eta _ { t - j }$

$$
\| W _ { t } - W _ { t - r } \| _ { F } \leq D _ { 0 } \Pi _ { t , r } .
$$

Lemma C.7 gives

$$
\| G _ { t - r } - G _ { t } \| _ { F } \leq L _ { 1 , U } ( e ^ { 2 D _ { 0 } \Pi _ { t , r } } - 1 ) \bar { s } ( W _ { t } ) .
$$

The gradient upper bound and Lemma B.9 also give

$$
\| G _ { t - r } \| _ { F } \leq C _ { 0 } \bar { s } ( W _ { t - r } ) \leq C _ { 0 } e ^ { 2 D _ { 0 } \Pi _ { t , r } } \bar { s } ( W _ { t } ) .
$$

Let

$$
C _ { m } : = L _ { 1 , U } C ( \alpha _ { 2 } , 2 D _ { 0 } , \beta _ { 2 , 1 } ) , \qquad C _ { v } : = C _ { 0 } \sqrt { 1 + C ( \alpha _ { 2 } , 4 D _ { 0 } , \beta _ { 2 , 2 } ) } .
$$

Expanding the first moment error and applying Lemma B.12, we obtain

$$
\Vert \widehat { m } _ { 2 , t } - G _ { t } \Vert _ { F } \leq L _ { 1 , U } \bar { s } ( W _ { t } ) \sum _ { r = 0 } ^ { t } w _ { t , r } ^ { ( 1 ) } ( e ^ { 2 D _ { 0 } \Pi _ { t , r } } - 1 ) \leq C _ { m } \eta _ { t } \bar { s } ( W _ { t } ) .
$$

For the second moment, the same lemma gives

$$
\begin{array} { l } { \displaystyle | | \sqrt { \widehat { v } _ { 2 , t } } | | _ { \operatorname* { m a x } } ^ { 2 } \leq \sum _ { r = 0 } ^ { t } w _ { t , r } ^ { ( 2 ) } \| G _ { t - r } \| _ { F } ^ { 2 } } \\ { \leq C _ { 0 } ^ { 2 } \bar { s } ( W _ { t } ) ^ { 2 } \left[ 1 + \displaystyle \sum _ { r = 0 } ^ { t } w _ { t , r } ^ { ( 2 ) } ( e ^ { 4 D _ { 0 } \Pi _ { t , r } } - 1 ) \right] } \\ { \leq C _ { 0 } ^ { 2 } \left[ 1 + C ( \alpha _ { 2 } , 4 D _ { 0 } , \beta _ { 2 , 2 } ) \eta _ { t } \right] \bar { s } ( W _ { t } ) ^ { 2 } } \\ { \leq C _ { v } ^ { 2 } \bar { s } ( W _ { t } ) ^ { 2 } . } \end{array}
$$

The last inequality uses $\eta _ { t } \leq 1$ . This proves (D.12) without assuming a loss upper bound.

We next obtain a global Hessian bound from Lemma C.7 with ${ \mathsf { N } } = \parallel \cdot \parallel _ { F }$ . For any $W , Z \in \mathbb { R } ^ { d \times d }$ , smoothness of the loss gives

$$
\begin{array} { r l } {  { \| \nabla ^ { 2 } \mathcal { L } _ { U } ( W ) [ Z ] \| _ { F } = \operatorname* { l i m } _ { h \to 0 } \frac { \| \nabla \mathcal { L } _ { U } ( W + h Z ) - \nabla \mathcal { L } _ { U } ( W ) \| _ { F } } { | h | } } } \\ & { \leq L _ { 1 , U } \bar { s } ( W ) \operatorname* { l i m } _ { h \to 0 } \frac { e ^ { 2 | h | \| Z \| _ { F } } - 1 } { | h | } } \\ & { \leq 2 L _ { 1 , U } \| Z \| _ { F } . } \end{array}
$$

Here we used $\bar { s } ( W ) \leq 1$ in the last inequality. Consequently,

$$
\operatorname* { s u p } _ { W } \Vert \nabla ^ { 2 } \mathcal { L } _ { U } ( W ) \Vert _ { F \to F } \leq H _ { U } , \qquad H _ { U } : = 2 L _ { 1 , U } .\tag{D.14}
$$

Let $c _ { F } : = c _ { \parallel \cdot \parallel _ { F } }$ be the constant from Lemma B.10, and let

$$
P _ { t } Z : = ( \sqrt { \widehat { v } _ { 2 , t } } + \varepsilon ) ^ { - 1 } \odot Z , \qquad p _ { - } : = ( \varepsilon + C _ { v } ) ^ { - 1 } , \qquad A : = \frac { C _ { m } } { c _ { F } \kappa _ { - } } .
$$

The second moment bound and $\bar { s } ( W _ { t } ) \leq 1$ imply

$$
p _ { - } \mathrm { I d } \preceq P _ { t } \preceq \varepsilon ^ { - 1 } \mathrm { I d } .
$$

Choose $\bar { \eta } _ { U } \in ( 0 , 1 ]$ sufficiently small that

$$
\frac { A \bar { \eta } _ { U } } { \varepsilon } \leq \frac { p _ { - } } { 2 } , \qquad \frac { H _ { U } ( 1 + A ) ^ { 2 } \bar { \eta } _ { U } } { 2 \varepsilon ^ { 2 } } \leq \frac { p _ { - } } { 4 } .
$$

All quantities in these conditions are independent of the choice of $\eta _ { 2 , 0 }$

We prove $\mathcal { L } _ { U } ( W _ { t } ) \leq L _ { 0 }$ by induction. The claim holds at $t = 0$ because $W _ { 0 } = 0 .$ . Suppose that $\mathcal { L } _ { U } ( W _ { t } ) \leq L _ { 0 }$ . Lemma C.5 gives $a _ { n , c } ( W _ { t } ) \geq \delta _ { 0 }$ for all $n \in [ N ]$ . Equations (C.11) and (C.6) therefore give

$$
\| G _ { t } \| _ { F } \ge c _ { F } E _ { t } \ge c _ { F } \kappa _ { - } \bar { s } ( W _ { t } ) .
$$

The first moment estimate now implies

$$
\begin{array} { r } { \| \widehat { m } _ { 2 , t } - G _ { t } \| _ { F } \leq C _ { m } \eta _ { t } \bar { s } ( W _ { t } ) \leq A \eta _ { t } \| G _ { t } \| _ { F } . } \end{array}
$$

Using the operator bounds on $P _ { t }$ , we obtain

$$
\begin{array} { r l } { \langle G _ { t } , P _ { t } \widehat { m } _ { 2 , t } \rangle = \langle G _ { t } , P _ { t } G _ { t } \rangle + \langle G _ { t } , P _ { t } ( \widehat { m } _ { 2 , t } - G _ { t } ) \rangle } & { } \\ { \displaystyle } & { \geq \left( p _ { - } - \frac { A \eta _ { t } } { \varepsilon } \right) \| G _ { t } \| _ { F } ^ { 2 } } \\ { \displaystyle } & { \geq \frac { p _ { - } } { 2 } \| G _ { t } \| _ { F } ^ { 2 } . } \end{array}
$$

Also,

$$
\Vert P _ { t } \widehat { m } _ { 2 , t } \Vert _ { F } \leq \frac { \Vert G _ { t } \Vert _ { F } + \Vert \widehat { m } _ { 2 , t } - G _ { t } \Vert _ { F } } { \varepsilon } \leq \frac { 1 + A } { \varepsilon } \Vert G _ { t } \Vert _ { F } .
$$

Taylor’s theorem and (D.14) yield

$$
\begin{array} { r l } & { E _ { t + 1 } \leq E _ { t } - \eta _ { t } \left. { G } _ { t } , P _ { t } \widehat m _ { 2 , t } \right. + \displaystyle \frac { H _ { U } \eta _ { t } ^ { 2 } } { 2 } \| P _ { t } \widehat m _ { 2 , t } \| _ { F } ^ { 2 } } \\ & { \qquad \leq E _ { t } - \eta _ { t } \left( \displaystyle \frac { p _ { - } } { 2 } - \displaystyle \frac { H _ { U } ( 1 + A ) ^ { 2 } } { 2 \varepsilon ^ { 2 } } \eta _ { t } \right) \| G _ { t } \| _ { F } ^ { 2 } } \\ & { \qquad \leq E _ { t } - \displaystyle \frac { p _ { - } } { 4 } \eta _ { t } \| G _ { t } \| _ { F } ^ { 2 } . } \end{array}
$$

Let $c _ { D } : = p _ { - } / 4$ . Since $E _ { t } = \mathcal { L } _ { U } ( W _ { t } ) - \ell _ { U } ^ { \star }$ , this gives $\mathcal { L } _ { U } ( W _ { t + 1 } ) \leq \mathcal { L } _ { U } ( W _ { t } ) \leq L _ { 0 }$ and closes the induction. The loss and attention bounds therefore hold at every iteration.

Let $c _ { R } : = c _ { D } c _ { F } ^ { 2 }$ . Using $\| G _ { t } \| _ { F } \geq c _ { F } E _ { t }$ in the descent estimate gives

$$
E _ { t + 1 } \leq E _ { t } - c _ { R } \eta _ { t } E _ { t } ^ { 2 } .
$$

Equation (C.6) gives $E _ { t } > 0$ at every finite iteration. Thus

$$
E _ { t + 1 } ^ { - 1 } - E _ { t } ^ { - 1 } = \frac { E _ { t } - E _ { t + 1 } } { E _ { t } E _ { t + 1 } } \geq c _ { R } \eta _ { t } \frac { E _ { t } } { E _ { t + 1 } } \geq c _ { R } \eta _ { t } .
$$

Summing yields $E _ { t } ^ { - 1 } \geq E _ { 0 } ^ { - 1 } + c _ { R } S _ { t }$ , proving (D.13). The same descent estimate gives

$$
c _ { R } \sum _ { s = 0 } ^ { t - 1 } \eta _ { 2 , s } E _ { s } ^ { 2 } \leq \sum _ { s = 0 } ^ { t - 1 } ( E _ { s } - E _ { s + 1 } ) = E _ { 0 } - E _ { t } \leq E _ { 0 } .
$$

Letting $t \to \infty ,$ , we obtain

$$
\sum _ { t = 0 } ^ { \infty } \eta _ { 2 , t } E _ { t } ^ { 2 } \le \frac { E _ { 0 } } { c _ { R } } < \infty .\tag{D.15}
$$

Since $0 < \alpha _ { 2 } \leq 1$ , we have $S _ { t } \to \infty$ and hence $E _ { t } \to 0$ . For every $n \in [ N ]$ , (C.6) implies

$$
0 \leq 1 - a _ { n , c } ( W _ { t } ) = s _ { n } ( W _ { t } ) \leq N \bar { s } ( W _ { t } ) \leq \frac { N } { \kappa _ { - } } E _ { t } \longrightarrow 0 .
$$

This proves the attention convergence.

We next bound the Frobenius norm of the iterates in terms of the logarithm of the inverse excess loss. Combining this estimate with the loss lower bound in (C.8) gives convergence of the normalized margin and the parameter direction.

Lemma D.4. Consider the iterates in Lemma $D . 3$ with $0 < \eta _ { 2 , 0 } \leq \bar { \eta } _ { U }$ , and let $J _ { t } : = \log ( E _ { t } ^ { - 1 } )$ . There is $C _ { W } > 0$ such that, for every integer $t \geq 0 ,$

$$
\| W _ { 2 , t } \| _ { F } \leq \gamma _ { 2 , F } ^ { - 1 } J _ { t } + C _ { W } .\tag{D.16}
$$

Moreover, there are constants $C _ { \mathrm { m a r } } , C _ { \mathrm { d i r } } > 0$ and an integer $t _ { \mathrm { m a r } } \geq 1$ such that $W _ { 2 , t } \neq 0$ and

$$
0 \leq \gamma _ { 2 , F } - \frac { \rho _ { 2 } ( W _ { 2 , t } ) } { \| W _ { 2 , t } \| _ { F } } \leq \frac { C _ { \operatorname* { m a r } } } { 1 + \log ( 1 + S _ { t } ) } ,\tag{D.17}
$$

$$
\left\| \frac { W _ { 2 , t } } { \| W _ { 2 , t } \| _ { F } } - \frac { W _ { F } ^ { \star } } { R _ { 2 , F } ^ { \star } } \right\| _ { F } \leq \frac { C _ { \mathrm { d i r } } } { \sqrt { 1 + \log ( 1 + S _ { t } ) } }\tag{D.18}
$$

for every integer $t \geq t _ { \mathrm { m a r } }$

Proof. For this proof, let $W _ { t } : = W _ { 2 , t } , \eta _ { t } : = \eta _ { 2 , t }$ , and $G _ { t } : = \nabla \mathcal { L } _ { U } ( W _ { t } )$ . Equations (C.10) and (C.6) give

$$
\| G _ { t } \| _ { F } \leq 2 \sqrt { 2 } Q _ { U } \bar { s } ( W _ { t } ) \leq \frac { 2 \sqrt { 2 } Q _ { U } } { \kappa _ { - } } E _ { t } .
$$

As in the previous proof, let $P _ { t } Z : = ( \sqrt { \widehat { v } _ { 2 , t } } + \varepsilon ) ^ { - 1 } \odot Z$ . We write the Adam update as

$$
W _ { t + 1 } - W _ { t } = - \frac { \eta _ { t } } { \varepsilon } G _ { t } + r _ { t } ,
$$

where

$$
r _ { t } = - \eta _ { t } P _ { t } ( \widehat { m } _ { 2 , t } - G _ { t } ) - \eta _ { t } ( P _ { t } - \varepsilon ^ { - 1 } \mathrm { I d } ) G _ { t } .
$$

For $z \geq 0 ,$ , we have

$$
0 \leq \varepsilon ^ { - 1 } - ( \varepsilon + z ) ^ { - 1 } = \frac { z } { \varepsilon ( \varepsilon + z ) } \leq \frac { z } { \varepsilon ^ { 2 } } .
$$

Using this inequality, (D.12), and $\bar { s } ( W _ { t } ) \leq E _ { t } / \kappa _ { - }$ , we obtain

$$
\| r _ { t } \| _ { F } \leq \frac { \eta _ { t } } { \varepsilon } \| \widehat { m } _ { 2 , t } - G _ { t } \| _ { F } + \frac { \eta _ { t } } { \varepsilon ^ { 2 } } \| \sqrt { \widehat { v } _ { 2 , t } } \| _ { \operatorname* { m a x } } \| G _ { t } \| _ { F } \leq C ( \eta _ { t } ^ { 2 } E _ { t } + \eta _ { t } E _ { t } ^ { 2 } ) .
$$

For $t \geq 1$ , the step sizes satisfy $S _ { t } \geq \eta _ { 2 , 0 } t ^ { 1 - \alpha _ { 2 } }$ . Hence (D.13) gives

$$
\eta _ { t } ^ { 2 } E _ { t } \leq \frac { \eta _ { 2 , 0 } ^ { 2 } ( t + 1 ) ^ { - 2 \alpha _ { 2 } } } { c _ { R } S _ { t } } \leq \frac { \eta _ { 2 , 0 } } { c _ { R } } t ^ { - 1 - \alpha _ { 2 } } .
$$

Since $\alpha _ { 2 } > 0$ , we have $\textstyle \sum _ { t = 0 } ^ { \infty } \eta _ { t } ^ { 2 } E _ { t } < \infty$ . Together with (D.15), this proves

$$
\sum _ { t = 0 } ^ { \infty } \| r _ { t } \| _ { F } \leq C \sum _ { t = 0 } ^ { \infty } ( \eta _ { t } ^ { 2 } E _ { t } + \eta _ { t } E _ { t } ^ { 2 } ) < \infty .
$$

We now relate the accumulated gradient terms to $J _ { t }$ . Let $\Delta _ { t } : = W _ { t + 1 } - W _ { t }$ . The moment estimates and the gradient upper bound imply

$$
\| \Delta _ { t } \| _ { F } \leq \frac { \eta _ { t } } { \varepsilon } \left( \| G _ { t } \| _ { F } + \| \widehat { m } _ { 2 , t } - G _ { t } \| _ { F } \right) \leq C \eta _ { t } E _ { t } .
$$

Taylor’s theorem and (D.14) give

$$
E _ { t + 1 } = E _ { t } - \frac { \eta _ { t } } { \varepsilon } \| G _ { t } \| _ { F } ^ { 2 } + q _ { t } ,
$$

where

$$
| q _ { t } | \leq \| G _ { t } \| _ { F } \| r _ { t } \| _ { F } + \frac { H _ { U } } { 2 } \| \Delta _ { t } \| _ { F } ^ { 2 } \leq C E _ { t } \| r _ { t } \| _ { F } + C \eta _ { t } ^ { 2 } E _ { t } ^ { 2 } .
$$

It follows that

$$
\sum _ { t = 0 } ^ { \infty } \frac { | q _ { t } | } { E _ { t } } \le C \sum _ { t = 0 } ^ { \infty } \| r _ { t } \| _ { F } + C \sum _ { t = 0 } ^ { \infty } \eta _ { t } ^ { 2 } E _ { t } < \infty .
$$

Since $E _ { t } , E _ { t + 1 } > 0$ , the inequality $- \log z \geq 1 - z$ gives

$$
J _ { t + 1 } - J _ { t } = - \log ( E _ { t + 1 } / E _ { t } ) \geq { \frac { E _ { t } - E _ { t + 1 } } { E _ { t } } } = { \frac { \eta _ { t } } { \varepsilon } } { \frac { \| G _ { t } \| _ { F } ^ { 2 } } { E _ { t } } } - { \frac { q _ { t } } { E _ { t } } } .
$$

Summing over $s = 0 , \ldots , t - 1$ , we obtain

$$
\sum _ { s = 0 } ^ { t - 1 } \frac { \eta _ { s } } { \varepsilon } \frac { \| G _ { s } \| _ { F } ^ { 2 } } { E _ { s } } \leq J _ { t } - J _ { 0 } + \sum _ { s = 0 } ^ { t - 1 } \frac { | q _ { s } | } { E _ { s } } \leq J _ { t } + C .
$$

For the Frobenius norm, (C.9) gives

$$
\| G _ { t } \| _ { F } \geq \gamma _ { 2 , F } E _ { t } - C _ { U } E _ { t } ^ { 2 } .
$$

Dividing by $E _ { t } ~ > ~ 0$ and rearranging yield $\gamma _ { 2 , F } \leq \Vert G _ { t } \Vert _ { F } / E _ { t } + C _ { U } E _ { t }$ . Multiplying by $\eta _ { t } \| G _ { t } \| _ { F } / \varepsilon$ , and then using $\| G _ { t } \| _ { F } \leq C E _ { t }$ , we obtain

$$
\gamma _ { 2 , F } \frac { \eta _ { t } } { \varepsilon } \| G _ { t } \| _ { F } \leq \frac { \eta _ { t } } { \varepsilon } \frac { \| G _ { t } \| _ { F } ^ { 2 } } { E _ { t } } + \frac { C _ { U } \eta _ { t } } { \varepsilon } E _ { t } \| G _ { t } \| _ { F } \leq \frac { \eta _ { t } } { \varepsilon } \frac { \| G _ { t } \| _ { F } ^ { 2 } } { E _ { t } } + C \eta _ { t } E _ { t } ^ { 2 } .
$$

Since $W _ { 0 } = 0$ , summing the updates gives

$$
\begin{array} { r l } {  { \gamma _ { 2 , F } \| W _ { t } \| _ { F } \leq \gamma _ { 2 , F } \sum _ { s = 0 } ^ { t - 1 } \frac { \eta _ { s } } { \varepsilon } \| G _ { s } \| _ { F } + \gamma _ { 2 , F } \sum _ { s = 0 } ^ { t - 1 } \| r _ { s } \| _ { F } } } \\ & { \leq \sum _ { s = 0 } ^ { t - 1 } \frac { \eta _ { s } } { \varepsilon } \frac { \| G _ { s } \| _ { F } ^ { 2 } } { E _ { s } } + C \sum _ { s = 0 } ^ { t - 1 } \eta _ { s } E _ { s } ^ { 2 } + \gamma _ { 2 , F } \sum _ { s = 0 } ^ { t - 1 } \| r _ { s } \| _ { F } } \\ & { \leq J _ { t } + C . } \end{array}
$$

This proves (D.16) for a sufficiently large $C _ { W } > 0 .$

We next combine the norm bound with the margin lower bound. Equation (C.8) gives

$$
\rho _ { 2 } ( W _ { t } ) \geq J _ { t } + \log \nu _ { U } \geq J _ { t } - | \log \nu _ { U } | .
$$

Since $E _ { t } \to 0 ,$ , we have $J _ { t } \to \infty$ . Thus, for all sufficiently large t, the margin is positive and $W _ { t } \neq 0$ . For such $t ,$ let $V _ { t } : = W _ { t } / \| W _ { t } \| _ { F }$ and $m _ { t } : = \rho _ { 2 } ( V _ { t } )$ . The maximum margin identity (B.18) gives $m _ { t } \leq \gamma _ { 2 , F }$ . Using positive homogeneity and (D.16), we obtain

$$
m _ { t } = \frac { \rho _ { 2 } ( W _ { t } ) } { \| W _ { t } \| _ { F } } \geq \frac { J _ { t } - | \log \nu _ { U } | } { \gamma _ { 2 , F } ^ { - 1 } J _ { t } + C _ { W } } .
$$

Consequently,

$$
0 \leq \gamma _ { 2 , F } - m _ { t } \leq \frac { \gamma _ { 2 , F } C _ { W } + | \log \nu _ { U } | } { \gamma _ { 2 , F } ^ { - 1 } J _ { t } + C _ { W } } .
$$

Moreover, (D.13) gives

$$
J _ { t } \geq \log ( E _ { 0 } ^ { - 1 } + c _ { R } S _ { t } ) \geq \log ( 1 + S _ { t } ) + \log \operatorname* { m i n } \{ E _ { 0 } ^ { - 1 } , c _ { R } \} .
$$

As $S _ { t } \to \infty$ , this implies $J _ { t } \geq \frac { 1 } { 2 } [ 1 + \log ( 1 + S _ { t } ) ]$ for all sufficiently large t. Substituting into the previous margin estimate proves (D.17).

Finally, we use the Frobenius hard margin problem to control the direction error. Since $m _ { t } > 0$ , the matrix $V _ { t } / m _ { t }$ is feasible.   
The feasible set is convex, and $W _ { F } ^ { \star }$ also minimizes $\textstyle { \frac { 1 } { 2 } } \| W \| _ { F } ^ { 2 }$ over this set.

For $0 \leq \lambda \leq 1$ , the matrix $W _ { F } ^ { \star } + \lambda ( V _ { t } / m _ { t } - W _ { F } ^ { \star } )$ is feasible because it is a convex combination of two feasible matrices. Since $W _ { F } ^ { \star }$ minimizes the squared Frobenius norm over the feasible set, for $0 < \lambda \leq 1$ we have

$$
0 \leq \frac { 1 } { 2 } \| \boldsymbol { W } _ { F } ^ { \star } + \lambda ( \boldsymbol { V } _ { t } / m _ { t } - \boldsymbol { W } _ { F } ^ { \star } ) \| _ { F } ^ { 2 } - \frac { 1 } { 2 } \| \boldsymbol { W } _ { F } ^ { \star } \| _ { F } ^ { 2 } = \lambda \langle \boldsymbol { W } _ { F } ^ { \star } , \boldsymbol { V } _ { t } / m _ { t } - \boldsymbol { W } _ { F } ^ { \star } \rangle + \frac { \lambda ^ { 2 } } { 2 } \| \boldsymbol { V } _ { t } / m _ { t } - \boldsymbol { W } _ { F } ^ { \star } \| _ { F } ^ { 2 } .
$$

Dividing by $\lambda > 0$ and letting $\lambda \downarrow 0$ give

$$
0 \leq \langle W _ { F } ^ { \star } , V _ { t } / m _ { t } - W _ { F } ^ { \star } \rangle = \frac { \langle W _ { F } ^ { \star } , V _ { t } \rangle } { m _ { t } } - \| W _ { F } ^ { \star } \| _ { F } ^ { 2 } .
$$

Since $m _ { t } > 0$ and $\lVert W _ { F } ^ { \star } \rVert _ { F } = R _ { 2 , F } ^ { \star } = \gamma _ { 2 , F } ^ { - 1 }$ , it follows that

$$
\langle W _ { F } ^ { \star } , V _ { t } \rangle \geq m _ { t } \| W _ { F } ^ { \star } \| _ { F } ^ { 2 } = \frac { m _ { t } } { \gamma _ { 2 , F } ^ { 2 } } .
$$

Both $V _ { t }$ and $\gamma _ { 2 , F } W _ { F } ^ { \star }$ have unit Frobenius norm. Expanding the squared distance and using the inner product bound, we obtain

$$
\begin{array} { r l } & { \| V _ { t } - \gamma _ { 2 , F } W _ { F } ^ { \star } \| _ { F } ^ { 2 } = \| V _ { t } \| _ { F } ^ { 2 } + \gamma _ { 2 , F } ^ { 2 } \| W _ { F } ^ { \star } \| _ { F } ^ { 2 } - 2 \gamma _ { 2 , F } \langle W _ { F } ^ { \star } , V _ { t } \rangle } \\ & { \qquad = 2 - 2 \gamma _ { 2 , F } \langle W _ { F } ^ { \star } , V _ { t } \rangle } \\ & { \qquad \leq \frac { 2 \left( \gamma _ { 2 , F } - m _ { t } \right) } { \gamma _ { 2 , F } } . } \end{array}
$$

Taking square roots and applying (D.17) prove (D.18), since $\gamma _ { 2 , F } W _ { F } ^ { \star } = W _ { F } ^ { \star } / R _ { 2 , F } ^ { \star }$ . Both rates hold for every $t \geq t _ { \mathrm { m a r } }$ after choosing $t _ { \mathrm { m a r } }$ sufficiently large. □

We now combine the preprocessing and main phase estimates for Adam.

Theorem D.1. Suppose that $K \geq 2 ,$ , h is bijective, the vocabulary embeddings are orthonormal, and the K single token preprocessing samples are equally weighted. Assume that Assumptions 1, 2, and 3 hold and $\mathcal { T } _ { 2 } \neq \mathcal { O }$ . Use Algorithm D.1 withfixed $\varepsilon > 0 , \beta _ { j , 1 } , \beta _ { j , 2 } \in [ 0 , 1 )$ , and

$$
\eta _ { j , s } = \eta _ { j , 0 } ( s + 1 ) ^ { - \alpha _ { j } } , \qquad 0 < \alpha _ { j } \leq 1 , \qquad j \in \{ 1 , 2 \} .
$$

There is $\bar { \eta } _ { 1 , \mathrm { A } } \in ( 0 , 1 ]$ such that,for every $0 < \eta _ { 1 , 0 } \leq \bar { \eta } _ { 1 , \mathrm { A } } ,$ , there is an integer $T _ { \mathrm { A } } \geq 1$ for which every $T \geq T _ { \mathrm { A } }$ produces $U : = W _ { 1 , T }$ satisfying (C.5). For each such U, there are $\bar { \eta } _ { U } \in ( 0 , 1 ]$ and $c _ { R } > 0 _ { ; }$ , independent of $\eta _ { 2 , 0 } ,$ such that every $0 < \eta _ { 2 , 0 } \leq \bar { \eta } _ { U }$ gives thefollowing conclusionsfor some constants $C _ { 1 , \mathrm { A } } , C _ { 2 , \mathrm { A } } > 0$

(i) Training loss. For every integer $t \geq 0 _ { i }$

$$
0 < E _ { U } ( W _ { 2 , t } ) \leq \left( E _ { U } ( 0 ) ^ { - 1 } + c _ { R } S _ { t } \right) ^ { - 1 } .\tag{D.19}
$$

In particular, $\mathcal { L } ( U , W _ { 2 , t } ) \to \ell _ { U } ^ { \star }$

(ii) Normalized Frobenius margin. There is an integer $t _ { U } \geq 1$ such that, for every $t \geq t _ { U }$ , we have $W _ { 2 , t } \neq 0$ and

$$
0 \le \gamma _ { 2 , F } - \rho _ { 2 } \left( \frac { W _ { 2 , t } } { \| W _ { 2 , t } \| _ { F } } \right) \le \frac { C _ { 1 , \mathrm { A } } } { 1 + \log ( 1 + S _ { t } ) } .\tag{D.20}
$$

The normalized margin converges to $\gamma _ { 2 , F }$ , with gap $O ( ( \log ( t + 1 ) ) ^ { - 1 } ) f o r 0 < \alpha _ { 2 } < 1$ and $O ( ( \log \log t ) ^ { - 1 } )$ for $\alpha _ { 2 } = 1$ (iii) Normalized direction. For every $t \geq t _ { U }$

$$
\left\| \frac { W _ { 2 , t } } { \| W _ { 2 , t } \| _ { F } } - \frac { W _ { F } ^ { \star } } { R _ { 2 , F } ^ { \star } } \right\| _ { F } \leq \frac { C _ { 2 , \mathrm { A } } } { \sqrt { 1 + \log ( 1 + S _ { t } ) } } .\tag{D.21}
$$

Consequently, $W _ { 2 , t } / \Vert W _ { 2 , t } \Vert _ { F } \to W _ { F } ^ { \star } / R _ { 2 , F } ^ { \star } .$

Proof. Choose $\bar { \eta } _ { 1 , \mathrm { A } }$ as in Lemma D.1, and fix $0 < \eta _ { 1 , 0 } \leq \bar { \eta } _ { 1 , \mathrm { A } }$ . Since $0 < \alpha _ { 1 } \leq 1$ , we have $S _ { 1 , T }  \infty$ . Lemma D.2 gives

$$
C W _ { 1 , T } = a _ { T } B + H _ { T } , \qquad a _ { T } : = \log S _ { 1 , T } \to \infty , \qquad \| H _ { T } \| _ { F } = O ( 1 ) .
$$

For $U = W _ { 1 , T }$ , the definition of δ in (C.4) and $\| B \| _ { F } ^ { 2 } = K - 1$ imply

$$
\delta = a _ { T } + \frac { \langle H _ { T } , B \rangle } { K - 1 } , \qquad | \delta - a _ { T } | \leq \frac { \| H _ { T } \| _ { F } } { \sqrt { K - 1 } } = { \cal O } ( 1 ) .
$$

Also, $R = H _ { T } - ( \delta - a _ { T } ) B _ { \mathrm { { + } } }$ , so

$$
r _ { U } \le \| R \| _ { F } \le \| H _ { T } \| _ { F } + | \delta - a _ { T } | \| B \| _ { F } \le 2 \| H _ { T } \| _ { F } = O ( 1 ) .
$$

Thus $\delta \to \infty$ while $r _ { U }$ remains bounded, and hence

$$
{ \frac { 2 ( K - 1 ) [ N \log K + \sqrt { 2 } ( N + 1 ) r _ { U } ] } { \delta } } \longrightarrow 0 .
$$

We can therefore choose an integer $T _ { \mathrm { A } } \geq 1$ such that $\delta > 0$ and this ratio is at most one for every $T \geq T _ { \mathrm { A } }$ . This proves (C.5).

Fix such a T and let $U : = W _ { 1 , T }$ . Lemma D.3 supplies $\bar { \eta } _ { U }$ and $c _ { R }$ , independently of $\eta _ { 2 , 0 }$ . For $0 < \eta _ { 2 , 0 } \leq \bar { \eta } _ { U }$ , its excess loss estimate gives (D.19). Since $S _ { t } \to \infty$ , we have $E _ { U } ( W _ { 2 , t } ) \to 0$ and therefore $\mathcal { L } ( U , W _ { 2 , t } ) \to \ell _ { U } ^ { \star }$

Lemma D.4 gives an integer $t _ { U } \geq 1$ such that $W _ { 2 , t } \neq 0$ and both (D.20) and (D.21) hold for every $t \geq t _ { U }$ . The right sides tend to zero because $S _ { t } \to \infty$ , proving the margin and direction limits. Finally, $S _ { t } = \Theta ( t ^ { 1 - \alpha _ { 2 } } )$ for $0 < \alpha _ { 2 } < 1$ and $S _ { t } = \Theta ( \log ( t + 1 ) )$ ) for $\alpha _ { 2 } = 1$ . Substitution into (D.20) gives the stated rates in terms of t. □