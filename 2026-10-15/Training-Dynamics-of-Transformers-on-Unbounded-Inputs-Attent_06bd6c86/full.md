# Training Dynamics of Transformers on Unbounded Inputs: Attention Learning Can Lag Output Layer Learning

Yaomengxi Han, Debarghya Ghoshdastidar Technical University of Munich

## Abstract

The success of transformers is attributed to several factors, including their capacity to express formal languages over variable-length inputs, and ability of the attention mechanism to focus on relevant input features. However, there is limited understanding of the training dynamics of shallow transformers, specifically how to decouple the learning of the attention and the output (feedfor ward) layer. A natural extension of the the ory of one-step gradient feature learning in feedforward networks suggests that one could first learn the attention weights with a single gradient step, and then train the output layer. We show that this fails for a shallow transformer trained to recognize a star-free language from unbounded-length input. We prove that the transformer can be trained to recognize the language with a three-phase gradient descent: the output layer is updated first, followed by an attention update, and the output layer is fine-tuned in the third phase. In contrast, the language cannot be learned by training only the attention or output layer, or by a single attention-layer up date followed by finetuning the output layer. We empirically show that “attention learning lagging output layer learning” is beneficial in other language recognition, regression, and in-context learning tasks as well.

## 1 INTRODUCTION

How do neural architectures learn features and attentions? What is the impact of the learning dynamics on the generalization abilities of the trained model? These are major open questions in deep learning theory. Significant progress has been made to address these questions in feedforward neural networks, both from a broad optimization perspective—for instance, implicit biases of gradient descent (Lyu and Li, 2020)—as well as from the viewpoint of learning specific tasks, like learning parity, sparse polynomial or gaussian mixture classification (Daniely and Malach, 2020; Abbe et al., 2023; Frei et al., 2023). A notable theoretical insight has emerged in this context, sometimes referred to as one-step gradient feature learning (Ba et al., 2022; Shi et al., 2023)—for learning tasks using a shallow feedforward network, it often sufices to perform one step of gradient descent on the input layer to learn the features and then train the output layer till convergence. Decoupling the training of diferent layers helps to study simplicity bias, lottery tickets, etc., in feed forward networks (Abbe et al., 2023; Shi et al., 2023).

Much less is understood about how attention learning happens in transformers. There are studies on implicit biases and training dynamics (see Relation to prior work), but it is not clear what the ideal way is to theoretically decouple the training of attention and output layers in a shallow transformer, comprising a softmax attention layer followed by a feedforward output layer. For example, a natural extension of the gradient feature learning paradigm in a shallow transformer is to perform a single gradient descent update of the attention weights, followed by training of the output layer. It is not clear if a “one-step attention learning” paradigm is suitable for training shallow transformers or a diferent multi-phase decoupled training is needed, for example, in context of the prototypical task of learning formal languages (Strobl et al., 2024).

Contributions. We study how attention learning happens in a shallow transformer trained to recognize formal language, where the training consists of words of varying length (potentially unbounded). Despite active research on the theory of attention learning, little is known about the training dynamics of transformers when trained on variable-length inputs. We aim to address if a one-step attention learning paradigm—akin to gradient feature learning in feedforward networks (Shi et al., 2023)—is an appropriate simplification of how shallow transformers learn to recognize a formal language. We have the following key contributions.

Multi-phase gradient descent dynamics in learning the occurrence language. We consider a minimalist example of learning to recognize the occurrence language $A _ { s } = \Sigma ^ { * } s \Sigma ^ { * }$ on the finite alphabet $\Sigma = \{ 1 , \ldots , d \}$ , i.e., the language of all words containing s, where $s \in \Sigma$ is not known apriori (Figure 1a). The occurrence language can be expressed by a shallow transformer with d-variate encoding of each bit, a single attention head, and a linear output layer (Proposition 1). Empirically, for this task, gradient descent (GD) first learns the output layer before the attention weights (Figure 1b). Theorem 2 formally shows that it is possible to learn the optimal transformer using multi-phase GD: (Phase 1) one-step GD update of output layer; (Phase 2) one-step GD update of attention weights; (Phase 3) training output layer till convergence. This result indicates that a one-step attention learning works in transformers, but when the attention learning happens after output layer is partly trained (see Figure 1d-e).

Impossibility results for simpler learning approaches. It is natural to question if the single-head transformer can also be trained using a simpler gradient-based rule. Propositions 2 and 3 show that, if either only the output or the attention layer is trained with the other held fixed, then a shallow transformer cannot learn uniformly over the set of d languages $\{ A _ { s } : s \in \Sigma \}$ . The same also holds for transformers with $O ( { \sqrt { d } } )$ heads (Proposition 4), but arbitrary fixed attention, showing that lazy training fails for this task, unless with a large number of heads. Furthermore, Theorem 1 shows that a two-phase approach of one step of GD update of attention, followed by training the output, cannot learn the language. Thus a na¨ıve extension of gradient feature learning principle does not work in this case.

Finite sample analysis and empirical study on other languages. While the aforementioned results consider GD on population loss, Corollary 1 extends the learnability result to finite sample setting, showing that the three-phase training also works with O(d ln d) samples for any fixed tolerance. We also numerically study other regression, language recognition, or in-context learning tasks with shallow transformers, and show that in most cases, the three-phase approach (with one-step GD update of output layer before one-step attention learning) outperforms other decoupled training of attention and output layers. Hence, we hypothesize that this approach is potentially a reasonable theoretical simplification of transformer training dynamics.

Relation to prior work. The expressivity and training dynamics of transformers is an active area of research in deep learning theory. While there is a rich literature on the expressivity of transformers in terms of formal languages, circuit complexity or space/time complexity classes (see surveys in Strobl et al. (2024); Lin and Barcel´o (2025); Barcel´o et al. (2025)), the current research on training dynamics of transformers has predominantly focused on finite-dimensional learn ing problems. There are results on the lazy training regime of shallow transformers (Hron et al., 2020; Arda et al., 2026) or implicit biases towards maximum margin (Tarzanagh et al., 2023), which hold in general data settings. However, fine-grained characterization of the training dynamics requires studying specific data distributions or learning tasks, which we also assume.

(a)  
![](images/360f7a3a6e1b7169393b9da202b329ee2e25411d1001ff3cdc94b5a47f71450c.jpg)

(c)
<table><tr><td>Training Plan</td><td>Phase</td><td>Train Loss (mean ± std)</td><td>Test Acc. (mean ± std)</td></tr><tr><td>attention(A) only</td><td>1</td><td> $1 . 6 4 \pm 0 . 0 8$ </td><td> $0 . 5 0 \pm 0 . 0 1$ </td></tr><tr><td>output(O) only</td><td>1</td><td> $0 . 7 4 \pm 0 . 0 5$ </td><td> $0 . 8 6 \pm 0 . 0 1$ </td></tr><tr><td>A(1x)→0</td><td>2</td><td> $0 . 7 2 \pm 0 . 2 6$ </td><td> $0 . 7 6 \pm 0 . 1 3$ </td></tr><tr><td>O(1x)→A(1x)→0</td><td>3</td><td> $( 2 \pm 1 ) \times 1 0 ^ { - 6 }$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td></tr></table>

(b)  
![](images/bde376feb09c04cf793a948e058d3073812dadaa7cdc75cd2f86a987c0ef211b.jpg)

(d)  
(e)  
![](images/b5477c929efcdaf3b1f46c7f4facb2ec4c8babd3c172e78cb6cac31b4c78fbcb.jpg)  
Figure 1: (a) Automaton of the occurrence language $A _ { s } = \Sigma ^ { * } s \Sigma ^ { * }$ with alphabet $\Sigma = \{ 1 , \ldots , d \}$ and unknown $s \in \Sigma$ . The language can be expressed by a transformer with one attention head and linear output layer (1), which encodes word $w = ( w _ { 1 } , \dots , w _ { m } )$ as $( \mathbf { e } _ { w _ { 1 } } , \hdots , \mathbf { e } _ { w _ { m } } )$ , where $\mathbf { e } _ { i } \in \mathbb { R } ^ { d }$ is i-th standard basis vector. The optimal shallow transformer has weights of the output layer $\beta ~ = ~ 2 \mathbf { e } _ { s } ~ \in ~ \mathbb { R } ^ { d }$ , and attention score for symbol $i \in \Sigma$ is $\gamma _ { i } ~ = ~ 1$ if $\textit { i } = \textit { s }$ , and 0 otherwise. (b) A shallow transformer is trained to recognize A using gradient descent jointly on both layers. The output gap $( \beta _ { s } - \operatorname* { m a x } _ { i \neq s } \beta _ { i } ) / 2$ and attention gap $( \gamma _ { s } - \operatorname* { m a x } _ { i \neq s } \gamma _ { i } )$ , plotted across training epochs, show that output gap increases before attention gap appears, indicating that attention learning lags output layer learning. (c) Several training plans are considered, which decouple gradient descent on attention layer (A) and output layer (O), sometimes using a single gradient step (1x). “One-step attention learning”, where one attention update is followed by training output layer till loss converges, $\mathrm { A } ( 1 \mathrm { x } ) {  } \mathrm { O }$ , does not correctly learn the language. We later prove that a 3-phase training, $\mathrm { O } ( \mathrm { 1 x } ) {  } \mathrm { A } ( \mathrm { 1 x } ) {  } \mathrm { O }$ , correctly learns to recognize $A _ { s } .$ . (d)–(e) show how our 3-phase plan reaches the ideal gaps in both layers, but training attention/output only, or attention before output do not.

Wen et al. (2025); Han and Ghoshdastidar (2026) study the training dynamics of transformers when trained on sparse parity function—a prototypical task for feedforward networks. Marion et al. (2025); Dohmatob (2025) study the joint dynamics of attention and output layer of a shallow transformer for single-location regression, while Wang et al. (2024) derive the joint training dynamics for sparse token averaging. The settings in these works correspond to finite/fixed-dimensional inputs, thereby ignoring the ability of transformers to handle varying input length. Furthermore, the attention-only or joint training dynamics do not reveal the importance of learning diferent layers. Bietti et al. (2023); Chen et al. (2024) study the dynamics of training induction heads in transformers for next token prediction on bigrams and in-context n-gram Markov chains. Bietti et al. (2023) empirically observe, under joint training, the induction head is learned more slowly than the global bigram model, similar to Figure 1b, and show that the induction head can be learned by sequential steps: first on output matrix, then on two attention key matrices. Chen et al. (2024) also consider a multi-phase training of diferent layers, involving a parameter-ineficient polynomial regression. Both works, however, restrict the problems to fixed context windows, efectively to fixed-length inputs. Moreover, these works do not explore if attention or output needs to be learned first.

Several empirical works study how transformers learn formal languages (Liu et al., 2023a,b; Butoi et al., 2025), but theoretical characterization of the training dynamics of transformers in learning formal languages is rather recent. Yang et al. (2024) study the gradient flow dynamics of learning word co-occurrence for fixed-length input, showing that the first phase mainly trains the output layer, before the joint training of output and attention happens. Huang et al. (2025) study the joint training dynamics of even-pairs and paritycheck languages (the latter with chain of thought), restricting to bounded-length words, and observe a similar slow learning of attention in early phase. These works implicitly suggest that a phenomenon equivalent to one-step gradient feature learning in feedforward networks may not hold for transformers, where attention learning typically happens late. Our work takes a step towards resolving this matter by proving that one-step attention learning mechanism is indeed possible, but needs to follow a step of output layer update, which aligns with the intuition from known joint training dynamics. Such a simplifying characterization of the training dynamics is a key contribution of our work, along with the impossibility of learning via simpler attention-only or attention-first training strategies. Prior works also assume fixed- or bounded length inputs, which our analysis does not require.

## 2 PROBLEM FORMULATION

This section describes the occurrence language recognition task, the training data distribution, the transformer model, and its ability to express the language.

Occurrence language and Training data. Let $\Sigma = \{ 1 , 2 , \ldots , d \}$ be a finite alphabet of size $d \ge 2$ For a symbol $s \in \Sigma .$ , the occurrence language, defined as $A _ { s } : = \Sigma ^ { * } s \Sigma ^ { * }$ , is the set of strings that contain s. We encode membership by the label function $\chi _ { s } ( w ) : = 2 \cdot { \bf 1 }$ s occurs in $w \} - 1 \in \{ \pm 1 \}$ . Consider any non-empty word $w = ( w _ { 1 } , \ldots , w _ { m } ) \in \Sigma ^ { * }$ drawn from a distribution, say ρ, such that the length is $m \sim \mathrm { G e o m e t r i c } ( p )$ , i.e., $\mathbb { P } ( m = k ) = p ( 1 - p ) ^ { k - 1 }$ for $k \geq 1$ , and given m, the symbols $w _ { 1 } , \ldots , w _ { m } \sim _ { i . i . d . }$ Uniform(Σ). Equivalently, after the first symbol, the sequence terminates at each step independently with probability p. Setting $\begin{array} { r l } { p } & { { } = \quad \frac { 1 } { d - 1 } } \end{array}$ exactly balances the proportion of positive/negative samples since, for any $s \in \Sigma ,$ , it holds that $\mathbb { P } _ { w \sim \rho } ( \chi _ { s } ( w ) ~ = ~ - 1 ) ~ =$ $\begin{array} { r } { \sum _ { m = 1 } ^ { \infty } \operatorname { \mathbb { P } } \left( | w | = m \right) \cdot \left( 1 - 1 / d \right) ^ { m } \stackrel { \cdot } { = } \frac { 1 } { 2 } . } \end{array}$ We use $D _ { s }$ to denote the joint distribution over $( w , y )$ where w $\sim \rho$ and $y = \chi _ { s } ( w ) \in \{ \pm 1 \}$ . Throughout this paper, we fix a target symbol $s \in \Sigma$ , unknown to the learner, and write $D \equiv D _ { s } , \chi \equiv \chi _ { s }$ when s is unambiguous from context. The transformer, described next, is trained by gradient descent on the population loss over D.

Shallow transformer architecture. To recognize the language $A _ { s } ,$ we consider a transformer with a single attention layer with one head, and a linear output layer. Every $i \in \Sigma$ is embedded as i-th standard basis vector $\mathbf { e } _ { i } \in \mathbb { R } ^ { d } .$ , and a word $w = ( w _ { 1 } , \ldots , w _ { m } ) \in \Sigma ^ { * }$ encodes into $\left( x _ { 1 } , \ldots , x _ { m } \right)$ , where $\begin{array} { r } { x _ { t } \ = \ \mathbf { e } _ { w _ { t } } . } \end{array}$ Positional encoding is not used since membership $\chi _ { s } ( w )$ is permutation-invariant. We prepend a fixed classification token x<sub>CLS</sub>, which attends to the m input positions but not to itself. The prediction is computed from the attention output of this token alone.

Since the model prediction depends only on the output of the classification token, the query and key matrices enter the model only through the vector $\alpha : =$ $W _ { K } ^ { \top } W _ { Q } x _ { \mathrm { C L S } } \in \mathbb { R } ^ { d }$ , whose i-th entry is the attention logit that the CLS token assigns to symbol i. Likewise, the value matrix and the linear output enter only through their product $\beta \in \mathbb { R } ^ { d }$ . We treat $( \alpha , \beta )$ as the trainable parameters, and call them the attention and output weights, respectively. The model computes

$$
\begin{array} { r l r } {  { f _ { \alpha , \beta } ( w ) = \beta ^ { \top } ( \sum _ { t = 1 } ^ { m } \frac { \exp ( \alpha ^ { \top } x _ { t } ) } { \sum _ { r = 1 } ^ { m } \exp ( \alpha ^ { \top } x _ { r } ) } \cdot x _ { t } ) - 1 } } \\ & { } & { = \displaystyle \sum _ { i = 1 } ^ { d } \beta _ { i } \cdot \frac { C _ { i } ( w ) \exp ( \alpha _ { i } ) } { \sum _ { j = 1 } ^ { d } C _ { j } ( w ) \exp ( \alpha _ { j } ) } - 1 , } \end{array}\tag{1}
$$

where the second line groups the attention scores from positions that carry the same symbol by using $\begin{array} { r } { C _ { i } ( w ) : = \sum _ { i = 1 } ^ { m } \mathbf { 1 } \{ w _ { j } = i \} } \end{array}$ to denote the number of occurrences of symbol i in w; a fixed bias 1 is used. We use $\begin{array} { r } { \gamma _ { i } ( w ) : = \frac { C _ { i } ( w ) \exp ( \alpha _ { i } ) } { \sum _ { i = 1 } ^ { d } C _ { j } ( w ) \exp ( \alpha _ { j } ) } } \end{array}$ to denote the total attention scores placed on symbol $i \in [ d ]$

Loss and expressivity. The transformer is trained to minimize the population (squared) risk over $D$

$$
\mathcal { L } ( \alpha , \beta ) = \mathbb { E } _ { ( w , y ) \sim D } \left[ \frac { 1 } { 2 } \left( f _ { \alpha , \beta } ( w ) - y \right) ^ { 2 } \right] .
$$

In Corollary 1, some results are extended to a finite sample case, where the empirical risk is minimized.

The following expressivity result, proved in $\mathrm { A p \mathrm { - } }$ pendix A.2, shows that there is a choice of $\alpha , \beta$ for which the loss can be made arbitrarily small.

Proposition 1 (Shallow transformer expresses occurrence language). For any specified $\varepsilon \in ( 0 , 1 )$ , a shallow transformer with $\beta = 2 { \bf e } _ { s }$ and ${ \pmb { \alpha } } = \ln ( 4 d / \varepsilon ^ { 2 } ) { \bf e } _ { s }$ achieves $\mathcal { L } ( \alpha , \beta ) \leq \varepsilon$ . The result also holds for any α such that $\alpha _ { s } - \operatorname* { m a x } _ { j \neq s } \alpha _ { j } \geq \ln ( 4 d / \varepsilon ^ { 2 } )$

Training. Our main results on training dynamics assume that transformer is trained to minimise the population risk  with GD. A gradient step on the parameters is written as $\pmb { \alpha } ^ { \prime } = \pmb { \alpha } - \eta \nabla _ { \pmb { \alpha } } \mathcal { L } ( \pmb { \alpha } , \beta )$ ， $\beta ^ { \prime } = \beta -$ $\eta ^ { \prime } \nabla _ { \beta } \mathcal { L } ( \alpha , \beta )$ , with parameter-wise step sizes $\eta , \eta ^ { \prime } \geq 0 .$ We subsequently use training plans that decouple GD updates of $\alpha , \beta$ , that is, each step updates only one of the two parameters, while the other is frozen $( \mathrm { i . e . }$ the learning rate for that parameter is 0). Our results in Sections 3 and 4 difer in the initialization of $( \alpha , \beta )$ and in the order of training the layers, as explained in the respective results.

## 3 FAILURE OF ONE- AND TWO-PHASE TRAINING

We first show that training a single layer is insuficient to learn the occurrence language, that is, if either layer is fixed independently of the target, then training the other cannot compensate.

Proposition 2 (Training attention only cannot learn the occurrence language). There exists an absolute constant $c _ { 1 } > 0$ , such that for every $d \geq 3$ and $\tilde { \boldsymbol { \beta } } \in \mathbb { R } ^ { d }$

$$
\operatorname* { m a x } _ { s \in [ d ] } \operatorname* { i n f } _ { \alpha \in \mathbb { R } ^ { d } } \mathcal { L } _ { s } ( \alpha , \tilde { \beta } ) \geq c _ { 1 } ,
$$

that $i s ,$ for any fixed output, there is a target $s \in \Sigma$ where no attention weight achieves arbitrarily low risk.

Since the bound holds for the infimum over $\pmb { \alpha } \in \mathbb { R } ^ { d }$ it applies to any procedure for training the attention

weights, not only to gradient descent. A similar result holds with the roles of the two layers exchanged.

Proposition 3 (Training output only cannot learn the occurrence language). There exists an absolute constant $c _ { 2 } > 0$ , such that for every $l \geq 3$ and $\tilde { \pmb { \alpha } } \in \mathbb { R } ^ { d }$

$$
\operatorname* { m a x } _ { s \in [ d ] } \operatorname* { i n f } _ { \beta \in \mathbb { R } ^ { d } } \mathcal { L } _ { s } ( \tilde { \alpha } , \beta ) \geq c _ { 2 } ,
$$

that is, for any fixed attention, there is a symbol $s , f o r$ which no output can achieve an arbitrarily small risk.

Proposition 3 concerns training the output over only one attention head. A natural objection is that fixed attention may sufice with enough heads. In response to this, consider the multi-head extension of (1):

$$
f _ { \mathbf { } A , \mathcal { B } } ( w ) = \sum _ { \begin{array} { l } { h \in [ H ] } \\ { i \in [ d ] } \end{array} } \beta _ { i } ^ { ( h ) } \frac { C _ { i } ( w ) \exp { \left( \alpha _ { i } ^ { ( h ) } \right) } } { \sum _ { j = 1 } ^ { d } C _ { j } ( w ) \exp { \left( \alpha _ { j } ^ { ( h ) } \right) } } - 1 ,\tag{2}
$$

where H is the number of heads. One can write the attention weights across heads as $\pmb { \mathcal { A } } = ( \pmb { \alpha } ^ { ( 1 ) } , \dots , \pmb { \alpha } ^ { ( H ) } ) \in$ $\mathbb { R } ^ { d \times H }$ and the output weights $\boldsymbol { B } = ( \beta ^ { ( 1 ) } , \dots , \beta ^ { ( H ) } ) \in$ $\mathbb { R } ^ { d \times H }$ . With $H = d$ heads, fixed attention does suffice: if each head $h \in [ H ]$ are initialized to attend to symbol $h , \ \mathrm { i . e . , } \ \alpha ^ { ( h ) }$ satisfies the configuration in Proposition 1, then for any target s, when the out put weights converge to $\beta ^ { ( \mathrm { \Delta } s ) } = 2 \mathbf { e } _ { s }$ and $\boldsymbol { \beta } ^ { ( h ) } = \mathbf { 0 } _ { d }$ for h $\neq s ,$ the transformer achieve arbitrarily small risk. Further, in the limit $H  \infty$ with random attention weights, training the output layer becomes kernel regression with a deterministic kernel (Hron et al., 2020). The next result shows that substantially fewer than d heads, specifically $O ( { \sqrt { d } } )$ are not enough.

Proposition 4 (Risk lower bound for output-only training with multiple heads). There is an absolute constant ${ \mathfrak { z } } \in ( 0 , 1 )$ , such that when the number ofheads $H \leq c { \sqrt { d } }$ , for any $d \geq 3$ and fixed attention ${ \tilde { \cal A } } ,$

$$
\operatorname* { m a x } _ { s \in [ d ] } \operatorname* { m i n } _ { \| B \| _ { F } \leq d ^ { 1 / 4 } } \mathcal { L } _ { s } ( \tilde { \mathcal { A } } , B ) \geq c .
$$

The norm constraint $\| \boldsymbol { B } \| _ { F } \le d ^ { 1 / 4 }$ is a requirement of our proof technique, and we do not know whether it can be removed. It is, however, not restrictive, since every solution we are aware of has a norm that is constant or decays with H. Consider the training plan in Theorem 2 for two cases: if all heads but one are frozen with zero output weights, the single-head solution can be achieved, whose output weights have constant norm $\| B \| _ { F } = 2$ . If all H heads are trained from identical zero initialization, they remain identical by symmetry, and the parameter $\overset { \cdot } { \alpha } { } ^ { ( h ) }$ will converge to satisfy the configuration in Proposition 1, while ${ \bf \vec { \boldsymbol { \beta } } } ^ { ( h ) }$ will converge to $\overline { { \frac { 2 } { H } } } { \bf e } _ { s }$ for each head $h \in [ H ]$ . In this case, the output norm is $\| B \| _ { F } = 2 / \sqrt { H }$

Two-phase training with one-step attention learning. Propositions 2–4 separate the occurrence language from settings in which training a single layer sufices, whether learning the output layer over a fixed representation (Hron et al., 2020; Arda et al., 2026) or learning the features under a fixed output layer (Frei et al., 2023; Han and Ghoshdastidar, 2026; Tarzanagh et al., 2023)—both approaches lead to the risk bounded away from zero for some target. We next consider a two-phase plan: a single gradient step on the attention weights, followed by training the output layer. This scheme is an adaptation of the one-step gradient feature learning in feed-forward networks, where it provably learns certain target classes (Ba et al., 2022; Shi et al., 2023). As we show next, this 2-phase GD training also cannot learn the occurrence language if one starts from uniform attention $\alpha = \mathbf { 0 } _ { d }$

Theorem 1 (Attention-first two-phase GD fails to learn occurrence). There is absolute constant $c _ { 3 } > 0$ such that the following holds for every $d \geq 9 , \tilde { \boldsymbol { \beta } } \in \mathbb { R } ^ { d }$ and $\eta > 0$ . Suppose the transformer is initialized as $\alpha = \mathbf { 0 } _ { d }$ (uniform attention) and $\beta = \tilde { \beta }$ , and one gradient step is used to update the attention weights as

$$
\alpha _ { s } ^ { \prime } : = - \eta \nabla _ { \alpha } \mathcal { L } _ { s } ( \mathbf { 0 } _ { d } , \tilde { \beta } ) ,
$$

where $s \in \Sigma$ is unknown but influences the updated attention $\pmb { \alpha } _ { s } ^ { \prime }$ through the population risk gradient. Then

$$
\operatorname* { m a x } _ { s \in [ d ] } \operatorname* { i n f } _ { \beta \in \mathbb { R } ^ { d } } \mathcal { L } _ { s } ( \alpha _ { s } ^ { \prime } , \beta ) \geq c _ { 3 } ,
$$

that is, there is a target $s \in \Sigma$ for which the optimal output layer on top of the updated attention $\pmb { \alpha } _ { s } ^ { \prime }$ cannot achieve an arbitrarily small risk.

The proofs of all results in this section are deferred to Appendix A.3-A.6. Together, these impossibility results show that neither layer can be trained alone to learn the occurrence language, and that taking the attention step first does not help. In Section 4, we show that a three-phase training plan (training the output, then the attention, then the output again) does help the transformer to learn the language.

## 4 LEARNING WITH THREE-PHASE TRAINING

The GD dynamics in Figure 1b show that output weights separate before the attention weights move. This is in line observations of Bietti et al. (2023); Huang et al. (2025) about joint gradient dynamics in other settings. Motivated by this, we study the following three-phase training plan. We initialize $\alpha = \beta =$ $\mathbf { 0 } _ { d } .$ which is agnostic to s, and train as follows:

Phase 1: One GD step on the output with step size η<sub>1</sub> and frozen attention: $\beta ^ { \prime } = - \eta _ { 1 } \nabla _ { \beta } \mathcal { L } _ { s } ( \mathbf { 0 } _ { d } , \mathbf { 0 } _ { d } )$

![](images/0151bf1efe49616b2f753b16e04bb32feb4aba82ee77db25d30991dcadca3a1d.jpg)  
Figure 2: Joint GD (left), O-A-O training (middle), and A-O training (right). Attention gap, output gap, and test accuracy for $d = 5 0$ are shown with $5 d ^ { 2 }$ training and $d ^ { 2 }$ test samples. Joint GD has no phases; for comparison, we start its Phase 2 when the attention starts to move $( \mathrm { g a p } = 0 . 1$ , at step 31) and end its Phase 2 when the attention almost saturates $( \mathrm { g a p } = 0 . 9$ , at step 127). In the 3-phase plan, the test accuracy reaches 1 by the end of Phase 2. Phase 3 only reduces training loss by fine-tuning output layer.

Phase 2: One GD step on the attention weights with step size $\eta _ { 2 }$ and frozen output: ${ \pmb \alpha } ^ { \prime } = - \eta _ { 2 } \nabla _ { { \pmb \alpha } } \mathcal { L } _ { s } ( { \bf 0 } _ { d } , \beta ^ { \prime } )$

Phase 3: T steps of GD on the output weights with step size $\eta _ { 3 } .$ , starting from $\beta ^ { \prime }$ with the attention fixed at $\alpha ^ { \prime } .$ . Formally, $\beta ( 0 ) = \beta ^ { \prime }$ and $\beta ( t + 1 ) = \beta ( t ) -$ $\eta _ { 3 } \nabla _ { \beta } \mathcal { L } _ { s } ( \pmb { \alpha } ^ { \prime } , \beta ( t ) )$ for $t = 0 , \ldots , T - 1$ . We denote the resulting output weights by $\beta ^ { \prime \prime } : = \beta ( T )$

The following theorem shows that this training plan learns the occurrence language for every target symbol up to any specified arbitrarily small risk bound.

Theorem 2 (Three-phase training learns the occurrence language). For every $d \geq 3 ,$ every target symbol $s \in [ d ]$ , and every $\varepsilon \in ( 0 , 1 )$ , when the step sizes and steps satisfy: $\begin{array} { r } { 0 < \eta _ { 1 } < \frac { d - 1 } { 2 } , \eta _ { 2 } > \frac { 2 0 ( d - 1 ) ^ { 2 } } { \eta _ { 1 } } . } \end{array}$ ln $\begin{array} { r } { \frac { 8 d } { \varepsilon } , ~ \eta _ { 3 } = } \end{array}$ 1, $T \geq$ 4d ln $\frac { 1 6 d } { \varepsilon }$ , then the parameters $( \alpha ^ { \prime } , \beta ^ { \prime \prime } )$ obtained by the above 3-phase training plan satisfy:

$$
\begin{array} { r } { \mathcal { L } _ { s } ( \pmb { \alpha } ^ { \prime } , \beta ^ { \prime \prime } ) \leq \varepsilon . } \end{array}
$$

We present the complete proof in Appendix A.7, and discuss three implications of the result below.

The role of each phase. Figure 2 compares joint GD, in which both parameters are updated at every step, with the three-phase plan and the attention-first plan. Phase 1 cannot be omitted, otherwise the plan is the attention-first schedule of Theorem 1, in which the attention step receives no guidance from the output layer and may not move towards the target (Figure 2c). Phase 1 also mirrors the early stage of joint GD, where the output separates while the attention weights barely move (Figure 2a). Phase 2 then saturates the attention on the target with a single large step, and Phase 3 trains the output weights towards $2 \mathbf { e } _ { s }$

The three-phase training plan difers from joint GD in two ways. First, in joint GD, the output weight on the target symbol can overshoot and then decrease while the attention saturates; freezing the output in Phase 2 removes this efect. Second, in joint GD, attention saturation and output weight convergence occur simultaneously. Our analysis, based on decoupling the two layers, shows that these aspects of joint GD are not necessary for learning. However, our analysis cannot explain this specific behavior of joint GD, which requires analysis of joint training dynamics.

Initialization Although zero initialization $( \alpha = \beta =$ $\mathbf { 0 } _ { d } )$ is agnostic to s, it is not standard in practice. We adopt this to simplify the theoretical analysis, as also done in Tarzanagh et al. (2023). Empirically, the same behavior occurs under random initialization (Table 1).

Finite-Sample GD analysis. The population gradient is not accessible in practice. We extend Theorem 2 to finite-sample GD to show that $O ( d \ln d + ( \ln d ) / \varepsilon )$ samples sufice to reach a population risk of at most ε.

Corollary 1 (Three-phase training with finite samples). Consider the same initialization $\alpha = \beta = \mathbf { 0 } _ { d } ,$ and the three-phase plan where the gradients of Phases 1 and 2 are computed on the empirical risk of $N _ { 1 }$ i.i.d. samples from $D ,$ and the gradients of Phase 3 on the empirical risk of $N _ { 3 }$ fresh i.i.d. samples. For every $c \ > \ 0 .$ , there exists a constant $C \ > \ 0$ such that the following holds for every $d \geq 3 ,$ , every target symbol $s \in [ d ]$ , and every $\varepsilon \in \mathsf { \Gamma } ( 0 , 1 )$ $I f \mathrm { ~ 0 ~ ~ < ~ }$ $\begin{array} { r } { \eta _ { 1 } \leq \sqrt { d } , \eta _ { 2 } \geq \frac { C d ^ { 2 } } { \eta _ { 1 } } \ln \frac { d } { \varepsilon } , \eta _ { 3 } = 1 , T = \left\lceil \frac { 7 2 } { \varepsilon } \right\rceil } \end{array}$ , and $N _ { 1 } \geq C d \ln d ,$ $\begin{array} { r } { N _ { 3 } \geq \frac { C \ln d } { \varepsilon } } \end{array}$ , then with probability at least $1 - 3 d ^ { - c }$ over the two draws, the resulting parameters $( \hat { \alpha } ^ { \prime } , \hat { \boldsymbol { \beta } } ^ { \prime \prime } )$ achieve $\mathcal { L } _ { s } \left( \hat { \alpha } ^ { \prime } , \hat { \beta } ^ { \prime \prime } \right) \leq \varepsilon$

The proof is deferred to Appendix A.8. Note that Phase 3 uses a batch of samples independent of those used in Phases 1 and 2. Redrawing samples between training stages is a standard approach in layer-wise training: Ba et al. (2022) finetune the second layer with a new set of samples after one GD step on the first layer, Nichani et al. (2023) also adopt similar sample splitting technique in their layer-wise training plan.

In contrast to the one- or two-phase training plans, the three-phase plan learns the occurrence language for every target symbol with polynomially many samples.

## 5 EXPERIMENTS

We empirically examine if our three-phase plan learns other language, regression, and in-context learning tasks. We present the setup and discuss the results.

Tasks. We consider six tasks: (i) occurrence; (ii) OR, where a string is positive if it contains at least one of two unknown target symbols; (iii) AND, where a string is positive if it contains both of two unknown target symbols; (iv) even pairs, the language over the binary alphabet $\{ a , b \}$ of strings where ab and ba occur even number of times (Huang et al., 2025); (v) in-context bigram learning (ICL bigram), a simplified version of Bietti et al. (2023) where the first attention layer is fixed, so that each position locks onto its preceding token; and (vi) single-location regression (SLR) (Marion et al., 2025). For occurrence and OR, we use the single-head model in (1), and for AND its multihead extension (2) with trainable bias and four heads, since one head cannot express it. For others, we use one-layer, single-head models with similar parameters to our attention and output (see Appendix B.1).

Training plans. We compare joint GD, as a baseline, with five other training plans: (i) attention only (A), corresponds to Proposition 2; (ii) output only (O), corresponds to Proposition 3; (iii) $\mathrm { A } ( 1 \mathrm { x } ) {  } \mathrm { O } _ { \mathrm { ; } }$ , matches the plan in Theorem 1; $\mathrm { ( i v ) \colon \ O ( 1 x ) { \to } A ( 1 x ) { \to } O }$ , following the phases in Theorem 2; and (v) $\mathrm { O } ( \mathrm { 1 x } ) {  } \mathrm { A } ( \mathrm { 1 x } )$ one output step followed by one attention step with a larger step size. The last plan isolates the role of the final output phase. All plans are run from a random initialization, with $\mathrm { O } ( \mathrm { 1 x } ) {  } \mathrm { A } ( \mathrm { 1 x } ) {  } \mathrm { O }$ additionally from the fixed initialization in Section 4.

We use $d = 5 0 , N = 5 d ^ { 2 }$ training samples, and $d ^ { 2 }$ samples for testing, with the following two exceptions: for even pairs, we train and test on N = 2046 strings of length at most 10 (Huang et al., 2025); for SLR, we fix input length $L = 1 0$ (Marion et al., 2025). All tasks are trained with Mean Squared Error (MSE), except even pairs and ICL bigram, which use the logistic and cross-entropy loss. Other setups can be found in Appendix B.1. We report the train loss and the test accuracy (test loss for SLR) as mean over 50 seeds.

Results. Table 1 shows the performance of various training plans and Figure 3 visualizes corresponding learning progress. We make the following observations.

Which training plans learn each task? Joint GD learns all six tasks, with accuracy at least 0.99 on the five classification tasks and test loss 0.01 on SLR. The three-phase plan matches it on five of them, from both the random and the zero initialization: it reaches accuracy 1.00 on occurrence, OR, even pairs, and ICL bigram, and test loss 0.01 on SLR. No one- or two-phase plan achieves this across the five tasks.

The $\mathrm { O } ( \mathrm { 1 x } ) {  } \mathrm { A } ( \mathrm { 1 x } )$ plan shows the contribution of the final phase. On occurrence and ICL bigram, it reaches perfect accuracy, but the training loss remains high. On these tasks, two phases sufice to classify, and the third is needed to reduce the loss. On the other tasks, two phases are not enough, even for classification.

Table 1: Train loss and test accuracy (loss for SLR), with mean over 50 seeds. Standard deviations are reported in Appendix B.4. Values below $1 0 ^ { - 3 }$ are reported with their upper bounds; those that match or surpass the joint GD baseline are bold. For AND, all attention-training phases have multiple steps instead of one.
<table><tr><td rowspan="2" colspan="2">Training plan</td><td colspan="2">Occurrence</td><td colspan="2">OR</td><td colspan="2">AND</td><td colspan="2">Even-Pair</td><td colspan="2">ICL bigram</td><td colspan="2">SLR</td></tr><tr><td>Train Loss</td><td>Test Acc.</td><td>Train Loss</td><td>Test Acc.</td><td>Loss</td><td>Train Test Acc.</td><td>Train Loss</td><td>Test Acc.</td><td>Train Loss</td><td>Test Acc.</td><td>Train Loss</td><td>Test Loss</td></tr><tr><td rowspan="5">Coupled with rand init.</td><td>Joint GD</td><td>10-3</td><td>1.00</td><td>10-3</td><td>1.00</td><td>0.04</td><td>0.99</td><td>0.01</td><td>1.00</td><td> $1 0 ^ { - 3 }$ </td><td>1.00</td><td>0.01</td><td>0.01</td></tr><tr><td>Attention (A) only</td><td>1.64</td><td>0.50</td><td>1.67</td><td>0.50</td><td>0.24</td><td>0.94</td><td>0.69</td><td>0.51</td><td>3.90</td><td>0.02</td><td>0.50</td><td>0.50</td></tr><tr><td>Output (O) only</td><td>0.74</td><td>0.86</td><td>0.68</td><td>0.85</td><td>0.79</td><td>0.75</td><td>0.63</td><td>0.50</td><td>2.70</td><td>0.16</td><td>0.48</td><td>0.48</td></tr><tr><td>A(1x)→0</td><td>0.72</td><td>0.76</td><td>0.69</td><td>0.75</td><td>0.30</td><td>0.91</td><td>0.62</td><td>0.50</td><td>2.58</td><td>0.26</td><td>0.41</td><td>0.43</td></tr><tr><td>O(1x)→A(1x)</td><td>0.47 10-5</td><td>1.00 1.00</td><td>0.64 10-5</td><td>0.78 1.00</td><td>0.52 0.27</td><td>0.87 0.93</td><td>0.52 0.08</td><td>0.50 1.00</td><td>3.20 10-3</td><td>1.00 1.00</td><td>0.28 0.01</td><td>0.28 0.01</td></tr><tr><td>Sec 4: 3-phase, zero init.</td><td>O(1x)→A(1x)→O  $\mathrm { O } ( \mathrm { 1 x } ) {  } \mathrm { A } ( \mathrm { 1 x } ) {  } \mathrm { O }$ </td><td> $\mathbf { 1 0 ^ { - 8 } }$ </td><td>1.00</td><td>10⁻7</td><td>1.00</td><td>0.90</td><td>0.68</td><td>0.08</td><td>1.00</td><td> $\mathbf { 1 0 ^ { - 3 } }$ </td><td>1.00</td><td>0.01</td><td>0.01</td></tr></table>

AND is the exception. From the random initialization, the three-phase plan only reaches an accuracy of 0.93, below joint GD (0.99), and attention-only training matches it (0.94). The three-phase plan with zero initialization performs worse than with random initialization, since all heads remain identical throughout training and cannot express the language. We return to further discussions of AND in Section 6.

Learning progress of joint GD and the phased plans. The final train loss and test accuracy / loss show that the three-phase plan can learn other formal languages, but they do not show how the output and attention weights evolve during the learning process. We therefore ask two further questions. First, does joint GD exhibit the same order of learning on these tasks as on the occurrence language, with the output weights separating before the attention weights move? Second, how closely do the trajectories of different phased plans follow that of joint GD?

Figure 3 addresses both questions. First, on AND, ICL bigram, and SLR, joint GD follows the same order as on occurrence and OR: the attention gap stays near zero while the output gap grows, and only begins to rise when the output separates significantly. even pairs is slightly diferent: under joint GD, the output only leads at a very short period, and the attention grows as fast as the output during the remaining training time, and the attention saturates before the output does. However, this still satisfies the principle that attention lags output learning at the start of learning.

For the second question, the three-phase plan yields the same final gaps as joint GD on even pairs, ICL bigram, and SLR. On ICL bigram and SLR, the trajectories follow a similar path as joint GD. On even pairs, the paths difer: the first output step of the three-phase plan already brings the gap close to its ideal value, whereas under joint GD the output gap grows slowly throughout. On AND, joint GD overshoots the ideal output gap while the attention saturates, as on the occurrence language, whereas the output gap of the three-phase plan remains below the ideal value, consistent with its lower accuracy in Table 1.

The other plans show how the order matters. In A(1x) O, the attention step leaves the attention gap near zero on ICL bigram and SLR, and the subsequent output training cannot compensate. On AND, in contrast, the attention gap grows even when the attention is trained first or alone, possibly because the anti-symmetric initialization ensures that some heads favor a target symbol from the start. Finally, on even pairs, ICL bigram and SLR, output-only training aligns the output with the ideal direction, yet Table 1 shows that it does not learn the task: correct output alone is insuficient without correct attention.

## 6 DISCUSSIONS

In this work, we study the (decoupled) training dynamics of a shallow transformer that recognizes a starfree language—the occurrence language—from inputs of unbounded length. We prove that one-step attention learning succeeds when it lags output layer learning. The natural two-phase adaptation of gradient feature learning—a single attention update followed by training the output layer—provably fails for some target symbol, as does training either layer alone (Propositions 2–4, Theorem 1). However, a three-phase training plan—involving one-step attention learning, sandwiched between two phases of output layer training— sufices to learn the occurrence language (Theorem 2, Corollary 1). The underlying mechanism is simple: the gradient of the attention is driven by the separation of the output, so the attention receives no signal about the target symbol until the output layer is informative.

![](images/922e1af2b1864d80669e4a68d8e07ff6b74ce2c77e6bcb594bf41e364d698b84.jpg)  
Figure 3: Attention and output gaps on four tasks. Mean standard deviation over 50 runs; dot ted lines mark the ideal gaps (see Appendix B.2). Phases are defined as in Figure 2; single-layer plans end after Phase 1 and $\mathrm { A } {  } \mathrm { O }$ after Phase 2. For AND, we plot the head with the largest gap for each plan (all heads: Appendix B.3). For even pairs and $\ \mathrm { { S L R } , }$ the output gap is $\cos ( u , w ^ { * } )$ and $\cos ( v , v ^ { * } )$ , with $w ^ { * }$ the max-margin and $v ^ { * }$ the ground-truth direction. Since $w ^ { * }$ depends on the attention weights, the output gap on even pairs can change in Phase 2 although the output is frozen. $\mathrm { O } {  } \mathrm { A }$ and the zero initialization are omitted, as they coincide with the first two phases of $\mathrm { O \to A \to O }$ and with the random initialization, respectively. The trajectories on OR resemble those on occurrence in Figure 2 and are deferred to Appendix B.5.

We expect this mechanism to be relevant beyond the specific problem studied in this paper. Our experiments already show that, in many cases, attention learning lags output learning, and the accuracy rises only once the attention begins to saturate. This may help explain the “abrupt learning” observed in transformers (Singh et al., 2024; Gopalani and Hu, 2025). Our impossibility results also suggest that parametereficient methods where only the attention or the output layer is finetuned can be fundamentally limited, and that the common theoretical assumption of a faster-learning output layer (Tian et al., 2023; Chen et al., 2024) may reflect a necessary order of learning. A broad open problem is to establish these connections for deeper models and richer tasks. Our multi-phased training paradigm could complement existing works on joint training dynamics of shallow Transformers (Yang et al., 2024; Huang et al., 2025) to investigate this question. We also note specific questions to build upon the presented results.

Finite sample complexity and realistic transformer. The focus of the present study was on the interplay of (decoupled) attention ans output layer training. To derive this, we made two key simplifications— focused on population training risk and parameterized the transformer by $( \alpha , \beta )$ . Parameterization in terms of key, query, value matrices leads to an overparametrized system in this case and, as shown in Tarzanagh et al. (2023), leads to diferent dynamics and implicit biases of GD. While the finite sample analysis in Corollary 1 shows that $O ( d \ln d )$ samples sufice, we do not know if such a sample complexity is necessary for the occurrence language. Furthermore, we assume a batch-GD-style split of training data across phases, and the complexity could difer considerably depending on use of full batch GD or stochastic GD (SGD)—a topic that is currently investigated for parity learning (Abbe et al., 2023).

Attention-first training with multiple heads. Table 1 shows that the AND task is the one case where the three-phase plan does not match joint GD, and where training the attention alone does equally well. A possible explanation is from the multi-head model we use for this task. With four heads and a random output layer, the output weights for some heads may already separate a target symbol at initialization, so that their attention receives a useful gradient without a preceding output step. This mirrors prior work on feedforward networks (Ba et al., 2022; Shi et al., 2023), where wider networks can make feature-first learning suficient. It suggests that the lower bounds for singlelayer and attention-first training weaken as the number of heads increases, and that the training order matters most for narrow and shallow models. How many heads are needed for attention-only or attention-first training to succeed is an open question.

Learnability and the expressivity hierarchy for transformers. The expressive power of transformers over formal languages is well studied: depending on the attention model, depth, and position embed dings, they recognize the star-free languages and beyond (Strobl et al., 2024; Barcel´o et al., 2024). However, expressivity does not guarantee learnability, and our work is only a first step towards a theory of transformer learnability. We establish lower bounds and a convergence guarantee for a single language, and show empirically that the three-phase plan extends to a few other star-free languages. These languages also form a very limited subset of the star-free class: occurrence, OR, and AND are piecewise testable, among the simplest star-free languages, and even pairs compares only the first and last symbols. Characterizing which formal languages are learnable or not learnable by transformers with GD remains wide open. A multi-phase training leads to simple decoupled dynamics and could be potentially useful to answer this question.

## AI use statement

In this work, we used generative AI tools to assist in the writing of proofs and provide feedback on research methodology or experiments. We have not used generative AI tools to generate synthetic datasets, help develop theoretical models or conceptual frameworks, formulate mathematical claims, provide critical ingredients for proving mathematical claims, propose or refine hypotheses, implement methods, support data analysis, or interpret results. Cleaning and reformating datasets, and assisting with translation, are not applicable to this work.

Additionally, we used generative AI tools to improve scientific figures, suggest experimental parameters, edit software code, and identify relevant literature.

We have reviewed all AI-assisted work. LLMgenerated code was verified and refactored by human authors; we use LLM to restructure the proof and check line by line afterward. We take responsibility for the final content of this work, including text, claims, or artifacts produced with the aid of generative AI.

## Acknowledgment

This work has been supported by the German Research Foundation (DFG) through the Research Training Group GRK 2428 ConVeY.

## REFERENCES

Emmanuel Abbe, Enric Boix Adser\`a, and Theodor Misiakiewicz. Sgd learning on neural networks: leap complexity and saddle-to-saddle dynamics. In Gergely Neu and Lorenzo Rosasco, editors, The Thirty Sixth Annual Conference on Learning Theory, COLT 2023, 12-15 July 2023, Bangalore, India, volume 195 of Proceedings of Machine Learning Research, pages 2552–2623. PMLR, 2023.

Enes Arda, Semih Cayci, and Atilla Eryilmaz. Finite-time analysis of gradient descent for shallow transformers. CoRR, abs/2601.16514, 2026. doi: 10.48550/ARXIV.2601.16514. URL https://doi.org/10.48550/arXiv.2601.16514.

Jimmy Ba, Murat A. Erdogdu, Taiji Suzuki, Zhichao Wang, Denny Wu, and Greg Yang. Highdimensional asymptotics of feature learning: How one gradient step improves the representation. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022.

Pablo Barcel´o, Alexander Kozachinskiy, Anthony Widjaja Lin, and Vladimir V. Podolskii. Logical languages accepted by transformer encoders with hard attention. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024.

Pablo Barcel´o, David Chiang, George Cybenko, Lena Strobl, and Andy Yang. Theory of neural language models (dagstuhl seminar 25282). Dagstuhl Reports, 15(7):22–52, 2025.

Alberto Bietti, Vivien Cabannes, Diane Bouchacourt, Herv´e J´egou, and L´eon Bottou. Birth of a transformer: A memory viewpoint. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023.

Alexandra Butoi, Ghazal Khalighinejad, Anej Svete, Josef Valvoda, Ryan Cotterell, and Brian DuSell. Training neural networks as recognizers of formal languages. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https://openreview.net/forum?id=aWLQTbfFgV.

Siyu Chen, Heejune Sheen, Tianhao Wang, and Zhuoran Yang. Unveiling induction heads: Provable training dynamics and feature learning in transformers. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024.

Amit Daniely and Eran Malach. Learning parities with neural networks. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.

Elvis Dohmatob. Understanding softmax attention layers: Exact mean-field analysis on a toy problem. In Danielle Belgrave, Cheng Zhang, Laura N. Montoya, Hsuan-Tien Lin, Razvan Pascanu, Piotr Koniusz, Marzyeh Ghassemi, Nancy Chen, Iv´an Vladimir Meza Ru´ız, and Arturo Loaiza-Bonilla, editors, Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025,

San Diego, CA, USA, December 2-7, 2025 / Mexico City, Mexico, November 30 - December 5, 2025, 2025.

Spencer Frei, Niladri S. Chatterji, and Peter L. Bartlett. Random feature amplification: Feature learning and generalization in neural networks. J. Mach. Learn. Res., 24:303:1–303:49, 2023.

Pulkit Gopalani and Wei Hu. What happens during the loss plateau? understanding abrupt learning in transformers. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025, San Diego, CA, USA, December 2-7, 2025 / Mexico City, Mexico, November 30 - December 5, 2025, 2025.

Yaomengxi Han and Debarghya Ghoshdastidar. Transformers provably learn sparse xor with polylogarithmic parameters, 2026. URL https://arxiv.org/abs/2502.07553.

Jiri Hron, Yasaman Bahri, Jascha Sohl-Dickstein, and Roman Novak. Infinite attention: NNGP and NTK for deep attention networks. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pages 4376–4386. PMLR, 2020.

Ruiquan Huang, Yingbin Liang, and Jing Yang. How transformers learn regular language recognition: A theoretical study on training dynamics and implicit bias. In Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025, volume 267 of Proceedings of Machine Learning Research. PMLR / Open-Review.net, 2025.

Anthony W. Lin and Pablo Barcel´o. The role of logic and automata in understanding transformers. In Pierre Ganty and Alessio Mansutti, editors, Reachability Problems - 19th International Conference, RP 2025, Madrid, Spain, October 1-3, 2025, Proceedings, volume 16230 of Lecture Notes in Computer Science, pages 17–30. Springer, 2025.

Bingbin Liu, Jordan T. Ash, Surbhi Goel, Akshay Krishnamurthy, and Cyril Zhang. Transformers learn shortcuts to automata. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. Open-Review.net, 2023a.

Bingbin Liu, Jordan T. Ash, Surbhi Goel, Akshay Krishnamurthy, and Cyril Zhang. Exposing attention glitches with flip-flop language modeling. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems

36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023b.

Kaifeng Lyu and Jian Li. Gradient descent maximizes the margin of homogeneous neural networks. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26- 30, 2020. OpenReview.net, 2020.

Pierre Marion, Rapha¨el Berthier, G´erard Biau, and Claire Boyer. Attention layers provably solve single-location regression. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https://openreview.net/forum?id=DVlPp7Jd7P.

Eshaan Nichani, Alex Damian, and Jason D. Lee. Provable guarantees for nonlinear feature learning in three-layer neural networks. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023.

Zhenmei Shi, Junyi Wei, and Yingyu Liang. Provable guarantees for neural networks via gradient feature learning. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023.

Aaditya K. Singh, Ted Moskovitz, Felix Hill, Stephanie C. Y. Chan, and Andrew M. Saxe. What needs to go right for an induction head? A mechanistic study of in-context learning circuits and their formation. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, volume 235 of Proceedings of Machine Learning Research, pages 45637–45662. PMLR / Open-Review.net, 2024.

Lena Strobl, William Merrill, Gail Weiss, David Chiang, and Dana Angluin. What formal languages can transformers express? A survey. Trans. Assoc. Comput. Linguistics, 12:543–561, 2024. doi: 10.1162/TACL A 00663.

Davoud Ataee Tarzanagh, Yingcong Li, Christos Thrampoulidis, and Samet Oymak. Transformers as support vector machines. In NeurIPS 2023 Workshop on Mathematics of Modern Machine Learning, 2023.

Yuandong Tian, Yiping Wang, Beidi Chen, and Simon S. Du. Scan and snap: Understanding training dynamics and token composition in 1-layer transformer. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023.

Joel A. Tropp. An introduction to matrix concentration inequalities. CoRR, abs/1501.01571, 2015. doi: 10.48550/ARXIV.1501.01571. URL https://doi.org/10.48550/arXiv.1501.01571.

Zixuan Wang, Stanley Wei, Daniel Hsu, and Jason D. Lee. Transformers provably learn sparse token selection while fully-connected nets cannot. In Ruslan Salakhutdinov, Zico Kolter, Katherine A. Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, volume 235 of Proceedings of Machine Learning Research, pages 51854–51912. PMLR / OpenReview.net, 2024.

Kaiyue Wen, Huaqing Zhang, Hongzhou Lin, and Jingzhao Zhang. From sparse dependence to sparse attention: Unveiling how chain-of-thought enhances transformer sample eficiency. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https://openreview.net/forum?id=AmEgWDhmTr.

Hongru Yang, Bhavya Kailkhura, Zhangyang Wang, and Yingbin Liang. Training dynamics of transformers to recognize word co-occurrence via gradient flow analysis. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024.

# A PROOFS FOR THE MAIN RESULTS

## A.1 Notations and Preliminaries

We introduce some notations, and provide a few calculations that is useful for the remainder of the proofs. The general per-sample gradients of $\ell ( w )$ with respect to each parameter for each index $i \in [ d ]$ is listed as follows:

$$
\begin{array} { l } { \displaystyle \frac { \partial \ell ( w ) } { \partial \beta _ { i } } = ( \hat { y } ( w ) - y ( w ) ) \cdot \gamma _ { i } } \\ { \displaystyle \frac { \partial \ell ( w ) } { \partial \alpha _ { i } } = ( \hat { y } ( w ) - y ( w ) ) \cdot \left( \beta _ { i } \gamma _ { i } ( 1 - \gamma _ { i } ) - \sum _ { j \neq i } \beta _ { j } \gamma _ { i } \gamma _ { j } \right) } \end{array}
$$

We write the expectations of first and the second conditional moments of $\gamma _ { s }$ at initialisation $\alpha = { \bf 0 } _ { d } \mathrm { : }$

$$
{ \begin{array} { l } { \mathbb { E } [ \gamma _ { s } \mid y = 1 ] = \displaystyle \sum _ { m = 1 } ^ { \infty } \mathbb { P } \left( | w | = m \mid y = 1 \right) \cdot \sum _ { n = 1 } ^ { m } \mathbb { P } \left( C _ { s } ( w ) = n \mid | w | = m , y = 1 \right) \cdot { \frac { n } { m } } } \\ { \displaystyle = \sum _ { m = 1 } ^ { \infty } { \frac { \mathbb { P } \left( | w | = m \right) \cdot \mathbb { P } \left( y = 1 \mid | w | = m \right) } { \mathbb { P } ( y = 1 ) } } \cdot \sum _ { n = 1 } ^ { m } { \frac { \mathbb { P } \left( | w | = m \right) \cdot \mathbb { P } \left( C _ { s } ( w ) = n \mid | w | = m \right) } { \mathbb { P } \left( | w | = m \right) \cdot \mathbb { P } \left( y = 1 \mid | w | = m \right) } } \cdot { \frac { n } { m } } } \\ { \displaystyle = 2 \sum _ { m = 1 } ^ { \infty } \mathbb { P } ( | w | = m ) \cdot \mathbb { E } _ { n \sim \mathrm { b i n } ( m , { \frac { 1 } { d } } ) } \left[ { \frac { n } { m } } \right] = { \frac { 2 } { d } } } \end{array} }\tag{3}
$$

$$
\begin{array} { r l } {  { \mathbb { E } [ \gamma _ { s } ^ { 2 } \mid y = 1 ] = 2 \sum _ { m = 1 } ^ { \infty } ( 1 - p ) ^ { m - 1 } p \cdot \mathbb { E } _ { n \sim \mathrm { b i n } ( m , \frac { 1 } { d } ) } [ ( \frac { n } { m } ) ^ { 2 } ] } } \\ & { = 2 \sum _ { m = 1 } ^ { \infty } ( 1 - p ) ^ { m - 1 } p \cdot \frac { m - 1 + d } { m d ^ { 2 } } = 2 \sum _ { m = 1 } ^ { \infty } ( 1 - p ) ^ { m - 1 } p \cdot ( \frac { 1 } { d ^ { 2 } } - \frac { 1 } { m d ^ { 2 } } + \frac { 1 } { m d } ) } \\ & { = 2 [ \frac { 1 } { d ^ { 2 } } + ( \frac { 1 } { d } - \frac { 1 } { d ^ { 2 } } ) \cdot \sum _ { m = 1 } ^ { \infty } ( 1 - p ) ^ { m - 1 } p \cdot \frac { 1 } { m } ] } \\ & { = 2 [ \frac { 1 } { d ^ { 2 } } + \frac { ( d - 1 ) \cdot p } { d ^ { 2 } \cdot ( 1 - p ) } \cdot \ln { ( p ) } ] = 2 [ \frac { 1 } { d ^ { 2 } } + \frac { ( d - 1 ) \cdot \ln ( d - 1 ) } { d ^ { 2 } ( d - 2 ) } ] } \end{array}\tag{4}
$$

(5)

For the background positions $j \neq s ,$ , we have:

$$
\mathbb { E } [ \gamma _ { j } \mid y = 1 ] = { \frac { 1 } { d - 1 } } \left( 1 - { \frac { 2 } { d } } \right) , \mathbb { E } [ \gamma _ { j } \mid y = - 1 ] = { \frac { 1 } { d - 1 } }\tag{6}
$$

$$
\begin{array} { r l } { \zeta _ { 2 } ^ { \pm } \| \psi = - 1 \| - \displaystyle \sum _ { i = 1 } ^ { N } z _ { i } \| w - z _ { i } \| \psi - x _ { i } \| ^ { 2 } \displaystyle \sum _ { i = 1 } ^ { N } \| E ( \zeta _ { i } | \psi _ { i } ) - y _ { i } \| ^ { 2 } = 1 , \quad | \psi - x _ { i } | ^ { 2 } = 1 , \quad | \psi | _ { i , i } ^ { 2 } = 1 , \quad | \psi | _ { i , i } ^ { 2 } = 1 \rangle ^ { 2 } \bigg [ \displaystyle \binom { ( i ) ^ { 2 } } { 2 } \bigg ] } & { } \\ { - \displaystyle \sum _ { i = 1 } ^ { N } z _ { i } \| w - z _ { i } | \psi - x _ { i } | ^ { 2 } \geq 1 , \quad | \psi | _ { i , i } ^ { 2 } = 1 , \quad | \psi | _ { i , i } ^ { 2 } = 1 , \quad | \psi | _ { i , i } ^ { 2 } = 1 , \quad | \psi | _ { i , i } ^ { 2 } = 1 \rangle ^ { 2 } \bigg ] } & { } \\ { \leq \displaystyle \sum _ { i = 1 } ^ { N } z _ { i } \| w - z _ { i } | \psi - x _ { i } | ^ { 2 } = 1 , \quad | \psi | _ { i , i } ^ { 2 } = 1 , \quad | \psi | _ { i , i } ^ { 2 } = 1 , \quad | \psi | _ { i , i } ^ { 2 } = 1 , \quad | \psi | _ { i , i } ^ { 2 } = 1 \rangle ^ { 2 } \bigg ] } & { } \\ { - \displaystyle \sum _ { i = 1 } ^ { N } \| z _ { i } \| - y _ { i } | ^ { 2 } = 1 , \quad | \psi | _ { i , i } ^ { 2 } = 1 , \quad | \psi | _ { i , i } ^ { 2 } = 1 , \quad | \psi | _ { i , i } ^ { 2 } = 1 , \quad | \psi | _ { i , i } ^ { 2 } = 1 \rangle ^ { 2 } \bigg ] } & { } \\  \leq \displaystyle \sum _ { i = 1 } ^ { N } \frac { 1 } { N } \displaystyle \sum _ { i = 1 } ^ { N } \| \psi - \end{array}\tag{7}
$$

$$
\begin{array} { r l } { \sum _ { 1 , 2 } ^ { n } | z | ^ { 2 } } & { z = - 1 \} - { \frac { \displaystyle \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { n } | ( x ) - x | ^ { 2 } } { \displaystyle \sum _ { n = 1 } ^ { n } \sum _ { j = 1 } ^ { n } ( ( x - 1 ) ^ { n } - 1 ) ^ { n } \sum _ { k = 1 } ^ { n } ( ( x - 1 ) ( x - 1 ) \cdot { \frac { x } { \omega _ { i } } } ) ^ { 2 } } } \cdot { \frac { \displaystyle \binom { n } { 2 } } { \displaystyle \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { n } ( x - 1 ) ^ { n } + 1 } } } \\ & { \qquad - { \frac { \displaystyle \sum _ { i = 1 } ^ { n } \gamma _ { i } ( 1 ) } { \displaystyle \sum _ { n = 1 } ^ { n } ( x - 1 ) ^ { n } ( x - 1 ) ^ { n } - 1 \cdot { \frac { x } { \omega _ { i } \cdot \omega _ { i } } } ) \cdot ( ( { \frac { n } { 2 } } ) ^ { 2 } } ) ^ { 2 } } } \\ &  = { \frac { \displaystyle \sum _ { i = 1 } ^ { n } { \cdot } { \frac { \displaystyle ( 1 } { \omega _ { i } } } ) ^ { n - 1 } \cdot { \frac { \gamma _ { i } ( \frac { n + 1 } { 2 } ) ^ { n } } { \displaystyle \sum _ { i = 1 } ^ { n } ( x - 1 ) ^ { n } } } \cdot [ { \frac { \omega ( n - 1 ) \cdot { \frac { 1 } { \omega _ { i } } } + n \cdot { \frac { 1 } { \omega _ { i } } } } { \displaystyle \sum _ { k = 1 } ^ { n } ( x - 1 ) ^ { 2 } + 1 } } + m \cdot { \frac { 1 } { \omega _ { i } } } ] } } \\ &  \leq  \frac  \displaystyle \sum _ { i = 1 } ^ { n } \cdot { \frac { \gamma _ { i } ( 1 ) - 1 } { \omega _ { i } } } \cdot (  \frac  \end{array}\tag{8}
$$

## A.2 Proof of Proposition 1

Proof. Since $\beta _ { s } = 2$ and $\beta _ { j } = 0$ for every $j \neq s ,$ we have $f _ { \alpha , \beta } ( w ) = 2 \gamma _ { s } ( w ) - 1$ . One can verify that $f _ { \alpha , \beta } ( w ) = - 1$ for every w $\notin A _ { s } ,$ and hence, the negative examples do not contribute to the loss. For the positive examples $w \in A _ { s } .$ , we separately consider words of length shorter or longer than $\begin{array} { r } { M = { \frac { 2 d } { \varepsilon } } } \end{array}$ . Since length w is a geometric random variable, $\begin{array} { r } { \mathbb { P } _ { ( w , y ) \sim D } ( | w | > M ) = ( 1 - \frac { 1 } { d - 1 } ) ^ { M } < e ^ { - 2 / \varepsilon } } \end{array}$ . Since the squared loss remains bounded by 2 for the choice of $f _ { \alpha , \beta } ( w ) = 2 \gamma _ { s } ( w ) - 1$ , the contribution of long words to the risk $\mathcal { L } ( \alpha , \beta )$ is bounded by $2 e ^ { - 2 / \varepsilon } < \varepsilon / 2$ For short words, $| w | \le M = 2 d / \varepsilon$ with $C _ { s } ( w ) ~ \geq ~ 1$ , the attention is bounded from below as $\gamma _ { s } ( w ) \geq$ $\begin{array} { r } { \frac { 4 d \cdot C _ { s } ( w ) } { 4 d \cdot C _ { s } ( w ) + \varepsilon ^ { 2 } \left( m - C _ { s } ( w ) \right) } \geq \frac { 4 d } { 4 d + 2 d \varepsilon } = \frac { 2 } { 2 + \varepsilon } } \end{array}$ , and from above since $\gamma _ { s } ( w ) \leq 1$ due to softmax. Hence, the loss for any short word $w \in A _ { s }$ is bounded as $\begin{array} { r } { \frac { 1 } { 2 } \left( f _ { \alpha , \beta } ( w ) - 1 \right) ^ { 2 } \leq \frac { 2 \varepsilon ^ { 2 } } { ( 2 + \varepsilon ) ^ { 2 } } < \frac { \varepsilon } { 2 } } \end{array}$ . Hence, short words also contribute to at most $\varepsilon / 2$ risk, and overall $\mathcal { L } ( \alpha , \beta ) \leq \varepsilon$ □

## A.3 Proof of Proposition 2

Proof. Let the target index be $s = \arg \operatorname* { m i n } _ { i \in [ d ] } \tilde { \beta } _ { i }$ and discuss the loss based on its value. When $\tilde { \beta } _ { s } \geq 1$ , it follows that $\begin{array} { r } { \hat { y } ( w ) = \sum _ { i = 1 } ^ { d } \tilde { \beta } _ { i } \gamma _ { i } ( w ) - 1 \ge \operatorname* { m i n } _ { i \in [ d ] } \tilde { \beta } _ { i } - 1 \ge 0 } \end{array}$ for any input w and any attention weights α. In this case, the expected risk is lower bounded by $\begin{array} { r } { \dot { \mathbb P } ( y = - 1 ) \times \frac { 1 } { 2 } \times ( 0 + 1 ) ^ { 2 } = \frac { 1 } { 4 } } \end{array}$

When $\tilde { \beta } _ { s } < 1$ , we lower bound the risk incurred on negative samples z of length m together with their positive counterparts. For a given z of length $m _ { \colon }$ , consider all positive samples $z ^ { \prime }$ obtained by inserting $n \geq 1$ copies of s into z. write $\begin{array} { r } { R ( d ) : = \frac { d ^ { 2 } - d } { d ^ { 2 } - 2 d + 2 } > 1 } \end{array}$ , we first compute the ratio of the two probability masses:

$$
\begin{array} { r l r } {  { \frac { \sum _ { z ^ { \prime } } \rho ( z ^ { \prime } ) } { \rho ( z ) } = \frac { \sum _ { n \geq 1 } { \binom { m + n } { n } ( 1 - p ) ^ { m + n - 1 } p ( \frac { 1 } { d } ) ^ { m + n } } } { ( 1 - p ) ^ { m - 1 } p ( \frac { 1 } { d } ) ^ { m } } } } \\ & { } & \\ & { } & { = \sum _ { n = 0 } ^ { \infty } { \binom { m + n } { n } ( 1 - p ) ^ { n } ( \frac { 1 } { d } ) ^ { n } - 1 } } \\ & { } & { = { R ( d ) ^ { m + 1 } - 1 \geq ( m + 1 ) ( R ( d ) - 1 ) } } \end{array}
$$

Therefore, for each negative sample z whose length satisfies $\textstyle m \geq { \frac { d + 1 } { 2 } }$ , it holds that $\begin{array} { r } { \frac { \sum _ { z ^ { \prime } } \rho ( z ^ { \prime } ) } { \rho ( z ) } \ge \frac { 1 } { 2 } } \end{array}$

For each negative sample $z ,$ write the output as $\begin{array} { r } { \hat { y } ( z ) = G ( z ) - 1 = \sum _ { i \neq s } \tilde { \beta } _ { i } \gamma _ { i } ( z ) - 1 } \end{array}$ , then for each of the positive counterpart $z ^ { \prime } ,$ , we have $\hat { y } ( z ^ { \prime } ) = \tilde { \beta } _ { s } \gamma _ { s } ( z ^ { \prime } ) + G ( z ) \cdot ( 1 - \gamma _ { s } ( z ^ { \prime } ) ) - 1$ . Now we distinguish two cases:

$G ( z ) \geq 1 { : }$ : then $\hat { y } ( z ) \ge 0$ , and the error on the negative sample is $\begin{array} { r } { \frac 1 2 ( \hat { y } ( z ) + 1 ) ^ { 2 } = \frac 1 2 G ( z ) ^ { 2 } \geq \frac 1 2 } \end{array}$

• If $G ( z ) < 1 \colon$ since $\widetilde { \beta } _ { s } < 1$ , the convex combination $\tilde { \beta } _ { s } \gamma _ { s } ( z ^ { \prime } ) + G ( z ) \cdot ( 1 - \gamma _ { s } ( z ^ { \prime } ) ) < 1$ for every $\gamma _ { s } ( z ^ { \prime } ) \in [ 0 , 1 ]$ ， so $\hat { y } \big ( z ^ { \prime } \big ) < 0$ , giving $\begin{array} { r } { \frac { 1 } { 2 } ( \hat { y } ( z ^ { \prime } ) - 1 ) ^ { 2 } > \frac { 1 } { 2 } } \end{array}$ on every positive counterpart, regardless of n or α.

In either case, the probability mass of samples with an error greater than $\begin{array} { l } { { \frac { 1 } { 2 } } } \end{array}$ is at least min $\begin{array} { r } { \{ \rho ( z ) , \sum _ { z ^ { \prime } } \rho ( z ^ { \prime } ) \} } \end{array}$ Summing over the window where the negative samples has length $\begin{array} { r } { m \geq \lceil \frac { d + 1 } { 2 } \rceil } \end{array}$ , we get:

$$
\mathcal { L } ( \alpha , \tilde { \beta } ) \geq \sum _ { | z | \geq \lceil \frac { d + 1 } { 2 } \rceil } \frac { 1 } { 2 } \cdot \operatorname* { m i n } \{ \rho ( z ) , \sum _ { z ^ { \prime } } \rho ( z ^ { \prime } ) \} \geq \sum _ { | z | \geq \lceil \frac { d + 1 } { 2 } \rceil } \frac { 1 } { 2 } \cdot \frac { 1 } { 2 } \cdot \rho ( z ) = \frac { 1 } { 8 } \left( 1 - \frac { 2 } { d } \right) ^ { \lceil \frac { d + 1 } { 2 } \rceil - 1 } \geq \frac { 1 } { 3 2 } ,
$$

where the last inequality follows from the monotonicity of the bound on the even and odd sequences of $d ,$ minimized at $d = 4$ and $d = 3$ respectively. □

## A.4 Proof of Proposition 3

Proof. We consider the target symbol as $s = \arg \operatorname* { m i n } _ { i \in [ d ] } \tilde { \alpha } _ { i }$ . First, consider any $\beta _ { s } \geq d + 1$ , we can lower bound the expected risk by summing up the error on all samples made from only the set symbols:

$$
\mathcal { L } _ { s } ( \tilde { \alpha } , \beta ) \geq \frac 1 2 ( \beta _ { s } - 2 ) ^ { 2 } \cdot \sum _ { m = 1 } ^ { \infty } ( 1 - p ) ^ { m - 1 } p \left( \frac 1 d \right) ^ { m } \geq \frac { d ^ { 2 } } { 8 \left[ ( d - 1 ) ^ { 2 } + 1 \right] } \geq \frac 1 8 .
$$

Otherwise, when $\beta _ { s } ~ < ~ d + 1$ , we consider the prediction for every negative sample z whose length satisfies $m = | z | > d$ and its corresponding positive counterparts where we insert one s into z. We first compute the ratio between the two masses, which gives:

$$
\frac { \sum _ { z ^ { \prime } } \rho ( z ^ { \prime } ) } { \rho ( z ) } = \frac { \binom { m + 1 } { 1 } ( 1 - p ) ^ { m } p \left( \frac { 1 } { d } \right) ^ { m + 1 } } { ( 1 - p ) ^ { m - 1 } p \left( \frac { 1 } { d } \right) ^ { m } } = \frac { ( m + 1 ) ( 1 - p ) } { d } \geq \frac { 1 } { 2 } .\tag{9}
$$

For each negative sample z, write the output as $\begin{array} { r } { \hat { y } ( z ) = G ( z ) - 1 = \sum _ { i \neq s } \beta _ { i } \gamma _ { i } ( z ) - 1 } \end{array}$ . And for each counterpart $z ^ { \prime } , \hat { y } ( z ^ { \prime } ) = G ( z ) \cdot ( 1 - \gamma _ { s } ( z ^ { \prime } ) ) + \beta _ { s } \cdot \gamma _ { s } ( z ^ { \prime } ) - 1$ . Now we distinguish two cases:

$\begin{array} { r } { G ( z ) > \frac { 1 } { 2 } ; } \end{array}$ then $\hat { y } ( z ) > - \frac { 1 } { 2 }$ , and the error on the negative sample is at least $\frac { 1 } { 8 }$

$\begin{array} { r } { G ( z ) \leq \frac { 1 } { 2 } } \end{array}$ : since $\exp ( \alpha _ { s } ) < \exp ( \alpha _ { i } )$ for any $i \neq s$ , we have $\begin{array} { r } { \gamma _ { s } ( z ^ { \prime } ) \leq \frac { 1 } { 1 + d } } \end{array}$ , and $\hat { y } ( z ^ { \prime } ) \leq \frac { 1 } { 2 }$ , therefore the error on each $z ^ { \prime }$ is at least $\frac { 1 } { 8 }$

In either case, the probability mass of samples with an error greater than $\frac { 1 } { 8 }$ is at least min $\{ \rho ( z ) , \ \sum _ { z ^ { \prime } } \rho ( z ^ { \prime } ) \}$ Summing over the window where the negative samples has length $m \geq d + 1$ , we have:

$$
\mathcal { L } ( \bar { \alpha } , \beta ) \geq \sum _ { | z | \geq d + 1 } \frac { 1 } { 8 } \cdot \operatorname* { m i n } \{ \rho ( z ) , \sum _ { z ^ { \prime } } \rho ( z ^ { \prime } ) \} \geq \sum _ { | z | \geq d + 1 } \frac { 1 } { 1 6 } \cdot \rho ( z ) = \frac { 1 } { 1 6 } \cdot \frac { 1 } { 2 } \left( 1 - \frac { 2 } { d } \right) ^ { d } = \frac { 1 } { 3 2 } \left( 1 - \frac { 2 } { d } \right) ^ { d } \geq \frac { 1 } { 8 6 4 } .
$$

## A.5 Proof of Proposition 4

We begin by selecting the target index that receives a low attention score. We first prove that such an index exists:

Lemma 1 (Existence of Index with Low Attention). For any fixed attention weights $\pmb { \alpha } ^ { ( 1 ) } , \ldots , \pmb { \alpha } ^ { ( H ) } \in \mathbb { R } ^ { d }$ , there exists $s \in [ d ]$ , such that $\sum _ { h = 1 } ^ { H } \mathbb { E } \left[ \left( \gamma _ { s } ^ { ( h ) } ( w ) \right) ^ { 2 } \right] \leq \frac { H } { d }$

Proof. By the definition of softmax, for each string w and each head $h , \gamma _ { i } ^ { ( h ) } ( w ) \in [ 0 , 1 ]$ for every $i \in [ d ]$ , hence:

$$
\sum _ { i = 1 } ^ { d } \left( \gamma _ { i } ^ { ( h ) } ( w ) \right) ^ { 2 } \leq \left( \sum _ { i = 1 } ^ { d } \gamma _ { i } ^ { ( h ) } ( w ) \right) ^ { 2 } = 1 .
$$

Taking expectation over w and summing over all heads gives $\begin{array} { r } { \sum _ { h = 1 } ^ { H } \sum _ { i = 1 } ^ { d } \mathbb { E } \left[ \left( \gamma _ { i } ^ { ( h ) } ( w ) \right) ^ { 2 } \right] \leq H } \end{array}$ . Therefore, there exists an index $s \in [ d ]$ such that $\begin{array} { r } { \sum _ { h = 1 } ^ { H } \mathbb { E } \left[ \left( \gamma _ { s } ^ { ( h ) } ( w ) \right) ^ { 2 } \right] \leq \frac { H } { d } } \end{array}$ □

We choose s that satisfies Lemma 1 as the target index. Consider any negative string z whose length m satisfies $\lceil \frac { d } { 2 } \rceil \leq m \leq d - 1$ , it follows that $\begin{array} { r } { \rho ( z ) = ( 1 - \bar { p } ) ^ { m - 1 } p \left( \frac { 1 } { d } \right) ^ { m } } \end{array}$ . Let $\mathcal { P } ( z )$ be the set of positive strings which after

removing all s will result in $z ,$ we have $\begin{array} { r } { \rho ( \mathcal P ( z ) ) = \sum _ { n = 1 } ^ { \infty } \binom { m + n } { n } ( 1 - p ) ^ { m + n - 1 } p \cdot \left( \frac 1 d \right) ^ { m + n } } \end{array}$ . Further, we can consider the ratio between the two masses, which simplifies into:

$$
\frac { \rho ( \mathcal { P } ( z ) ) } { \rho ( z ) } = \sum _ { n = 1 } ^ { \infty } \binom { m + n } { n } ( 1 - p ) ^ { n } \left( \frac { 1 } { d } \right) ^ { n } = \left( \frac { d - 2 } { d ^ { 2 } - 2 d + 2 } + 1 \right) ^ { m + 1 } - 1 \geq \frac { 1 } { 2 } .
$$

For any $z ^ { \prime } \in \mathcal { P } ( z )$ , we can write the output as $f ( \boldsymbol { z } ^ { \prime } ) = \sum _ { h = 1 } ^ { H } \left[ \left( 1 - \gamma _ { s } ^ { ( h ) } ( \boldsymbol { z } ^ { \prime } ) \right) g ^ { ( h ) } ( \boldsymbol { z } ) + \beta _ { s } ^ { ( h ) } \cdot \gamma _ { s } ^ { ( h ) } ( \boldsymbol { z } ^ { \prime } ) \right] - 1$ , where $\begin{array} { r } { g ^ { ( h ) } ( z ) = \sum _ { i \neq s } \beta _ { i } ^ { ( h ) } \gamma _ { i } ^ { ( h ) } ( z ) } \end{array}$ . Since $| g ^ { ( h ) } ( z ) | \leq \| \beta ^ { ( h ) } \| _ { 2 } \| \gamma ^ { ( h ) } ( z ) \| _ { 2 } \leq \| \beta ^ { ( h ) } \| _ { 2 }$ and $\left| \beta _ { s } ^ { ( h ) } \right| \le \left\| \beta ^ { ( h ) } \right\| _ { 2 }$ , we have:

$$
| f ( z ^ { \prime } ) - f ( z ) | = \left| \sum _ { h = 1 } ^ { H } \gamma _ { s } ^ { ( h ) } ( z ^ { \prime } ) \left( \beta _ { s } ^ { ( h ) } - g ^ { ( h ) } ( z ) \right) \right| \leq 2 \sum _ { h = 1 } ^ { H } \gamma _ { s } ^ { ( h ) } ( z ^ { \prime } ) \| \beta ^ { ( h ) } \| _ { 2 } \leq 2 \| \mathcal { B } \| _ { F } \left( \sum _ { h = 1 } ^ { H } \left( \gamma _ { s } ^ { ( h ) } ( z ^ { \prime } ) \right) ^ { 2 } \right) ^ { 1 / 2 } ,
$$

where the last step is from Cauchy-Schwarz inequality over the H heads.

For each negative string z and each $z ^ { \prime } \in \mathcal { P } ( z )$ , we define a coupling weight $\begin{array} { r } { \nu ( z , z ^ { \prime } ) : = \operatorname* { m i n } \left( \rho ( z ) , \rho ( \mathcal { P } ( z ) ) \right) \cdot \frac { \rho ( z ^ { \prime } ) } { \rho ( \mathcal { P } ( z ) ) } } \end{array}$ First, it directly follows that $\nu ( z , z ^ { \prime } ) \leq \rho ( z ^ { \prime } )$ . Second, $\begin{array} { r } { \sum _ { z ^ { \prime } \in \mathcal { P } ( z ) } \nu \big ( z , z ^ { \prime } \big ) < \rho ( z ) } \end{array}$ . Since $\begin{array} { r } { \frac { \rho ( \mathcal { P } ( z ) ) } { \rho ( z ) } \ge \frac { 1 } { 2 } } \end{array}$ in the length window we considered for $z ,$ it holds that min $( \rho ( z ) , \rho ( \mathcal { P } ( z ) ) ) \ge { \frac { 1 } { 2 } } \rho ( z )$ , hence the total coupling mass is lower bounded by:

$$
\begin{array} { l } { { \displaystyle \sum _ { \tau = 1 / 2 \rceil \leq 1 \leq i = 1 } \displaystyle \sum _ { \boldsymbol { x } ^ { \prime } \in P ( z ) } \nu ( \boldsymbol { z } , \boldsymbol { z } ^ { \prime } ) = \sum _ { z } \operatorname* { m i n } ( \rho ( z ) , \rho ( P ( z ) ) ) \geq \frac { 1 } { 2 } \sum _ { \tau | \boldsymbol { \mathcal { A } } / 2 ) | \leq i \leq i - 1 } \rho ( z ) = \frac { 1 } { 2 } \sum _ { m = \lceil a / 2 \rceil } ^ { d } ( 1 - p ) ^ { m - 1 } p \left( \frac { d - 1 } { d } \right) ^ { m } } } \\ { { = \displaystyle \frac { 1 } { 4 } \left[ \left( 1 - \frac { 2 } { \bar { d } } \right) ^ { [ d / 2 ] - 1 } - \left( 1 - \frac { 2 } { \bar { d } } \right) ^ { d - 1 } \right] : = M } } \end{array}
$$

We write $\begin{array} { r } { x = 1 - \frac { 2 } { d } . } \end{array}$ For any $0 \textless t \textless 1$ , we have exp $\begin{array} { r } { \left( - \frac { t } { 1 - t } \right) \le 1 - t \le \exp ( - t ) } \end{array}$ . Plug in $t = 2 / d ,$ , we get $\begin{array} { r } { x \ge \exp \left( - \frac { 2 } { d - 2 } \right) } \end{array}$ Since $\begin{array} { r } { \lceil d / 2 \rceil - 1 \leq \frac { d } { 2 } } \end{array}$ , we get $\ 2 \cdot \frac { \lceil d / 2 \rceil - 1 } { d - 2 } \ \leq \ \frac { d } { d - 2 } \ \leq \ 3$ for all $d \ \geq \ 3$ , so $x ^ { \lceil d / 2 \rceil - 1 } \geq$ exp $\left( - 2 \frac { \lceil d / 2 \rceil - 1 } { d - 2 } \right) \ge \exp ( - 3 )$ . Using the other side of the inequality, since $x \leq \exp \left( - \textstyle { \frac { 2 } { d } } \right)$ and $\begin{array} { r } { \lfloor d / 2 \rfloor \geq \frac { d - 1 } { 2 } } \end{array}$ , we have $\begin{array} { r } { x ^ { \lfloor d / 2 \rfloor } \leq \exp \left( - 2 \frac { \lfloor d / 2 \rfloor } { d } \right) \leq \exp \left( - \frac { d - 1 } { d } \right) \leq \exp \left( - \frac { 2 } { 3 } \right) } \end{array}$ , which gives us $1 - x ^ { \lfloor d / 2 \rfloor } \geq 1 - \exp \left( - \frac { 2 } { 3 } \right)$ . Combining both sides, we have $\begin{array} { r } { M \geq \frac { 1 } { 4 } \exp ( - 3 ) \left( 1 - \exp \left( - \frac { 2 } { 3 } \right) \right) > 0 } \end{array}$ , which is an absolute constant independent of d.

Define $\begin{array} { r } { \mathbb { E } _ { \mathrm { p a i r } } \left[ \cdot \right] : = \frac { 1 } { M } \sum _ { z , z ^ { \prime } } \nu ( z , z ^ { \prime } ) \times ( \cdot ) } \end{array}$ . Since $\nu ( z , z ^ { \prime } ) \leq \rho ( z ^ { \prime } )$ , we have:

$$
\mathbb { E } _ { \mathrm { p a i r } } \left[ \sum _ { h = 1 } ^ { H } \left( \gamma _ { s } ^ { ( h ) } ( z ^ { \prime } ) \right) ^ { 2 } \right] \leq \frac { 1 } { M } \sum _ { h = 1 } ^ { H } \mathbb { E } _ { \rho } \left[ \left( \gamma _ { s } ^ { ( h ) } ( w ) \right) ^ { 2 } \right] \leq \frac { H } { d M } ,
$$

using Lemma 1 in the last step. Finally, by Jensen’s inequality, we have:

$$
\mathbb { E } _ { \mathrm { p a i r } } | f ( z ^ { \prime } ) - f ( z ) | \leq 2 \| \boldsymbol { \mathcal { B } } \| _ { F } \sqrt { \frac { H } { d M } } .\tag{10}
$$

For every pair $( z , z ^ { \prime } ) , y ( z ) = - 1$ and $y ( z ^ { \prime } ) = 1$ . Denote $e _ { z } : = f ( z ) + 1$ and $e _ { z ^ { \prime } } : = f ( z ^ { \prime } ) - 1$ , we have $e _ { z } ^ { 2 } + e _ { z ^ { \prime } } ^ { 2 } \geq$ ${ \begin{array} { r l } { { \frac { 1 } { 2 } } \left( 2 - | f ( z ^ { \prime } ) - f ( z ) | \right) _ { + } ^ { 2 } } \end{array} }$ . Since $u \mapsto ( 2 - u ) _ { + } ^ { 2 }$ is convex, combining Jensen’s inequality and Eq. 10, we have:

$$
\mathbb { E } _ { \mathrm { p a i r } } \left[ e _ { z } ^ { 2 } + e _ { z ^ { \prime } } ^ { 2 } \right] \geq \frac { 1 } { 2 } \left( 2 - 2 \| \boldsymbol { \mathcal { B } } \| _ { F } \sqrt { \frac { H } { d M } } \right) _ { + } ^ { 2 } .
$$

Because $\sum _ { z ^ { \prime } \in \mathcal { P } ( z ) } \nu ( z , z ^ { \prime } ) \leq \rho ( z )$ and $\nu ( z , z ^ { \prime } ) \leq \rho ( z ^ { \prime } )$ as established above, the weighted sum $\begin{array} { r } { \sum _ { z , z ^ { \prime } } \nu ( z , z ^ { \prime } ) ( e _ { z } ^ { 2 } + e _ { z ^ { \prime } } ^ { 2 } ) } \end{array}$ is bounded above by the corresponding disjoint contribution to $\mathbb { E } _ { \rho } \left[ ( f ( w ) - y ( w ) ) ^ { 2 } \right]$ , hence we can bound the

risk by:

$$
\mathcal { L } _ { D } ( f ) = \frac { 1 } { 2 } \cdot \mathbb { E } _ { \rho } \left[ \left( f ( w ) - y ( w ) \right) ^ { 2 } \right] \geq \frac { 1 } { 2 } M \cdot \mathbb { E } _ { \mathrm { p a i r } } \left[ e _ { z } ^ { 2 } + e _ { z ^ { \prime } } ^ { 2 } \right] \geq \frac { M } { 4 } \left( 2 - 2 \| B \| _ { F } \sqrt { \frac { H } { d M } } \right) _ { + } ^ { 2 } .
$$

Equivalently, when the number of heads $\begin{array} { r } { H \leq \frac { d M } { 4 \| B \| _ { F } ^ { 2 } } } \end{array}$ , the loss is lower bounded by $\frac { M } { 4 }$ . Since we know the output norm is bounded by $\| B \| _ { F } ^ { 2 } \leq { \sqrt { d } }$ , we have that when $\begin{array} { r } { H \le \frac { M } { 4 } \sqrt { d } . } \end{array}$ , the loss is lower bounded by $\frac { M } { 4 }$

## A.6 Proof of Theorem 1

Proof. First, we prove that for whatever $\pmb { \alpha } ^ { \prime }$ we obtained after the first step, if $\begin{array} { r } { \beta _ { s } > \frac { d } { 4 } } \end{array}$ then the population risk is already lower bounded by a constant. Here $\beta \in \mathbb { R } ^ { d }$ is an arbitrary vector. Consider all positive samples with only s, then:

$$
{ \mathcal { L } } ( \alpha ^ { \prime } , \beta ) \geq \sum _ { m = 1 } ^ { \infty } { \frac { 1 } { 2 } } \left( { \frac { d } { 4 } } - 2 \right) ^ { 2 } \cdot ( 1 - p ) ^ { m - 1 } p \left( { \frac { 1 } { d } } \right) ^ { m } = { \frac { ( d - 8 ) ^ { 2 } } { 3 2 \left[ ( d - 1 ) ^ { 2 } + 1 \right] } } \geq { \frac { 1 } { 2 0 8 0 } } .
$$

Hence in the rest of the analysis, we will only consider the case where $\beta _ { s } \leq \frac { d } { 4 }$ . Now we prove that after the first gradient step, we can find an index s such that after the update, $\alpha _ { s } ^ { \prime } \leq 0$ . Consider the following sum of the gradients:

$$
\begin{array} { r l } {  { \sum _ { i = 1 } ^ { d } \frac { \partial \mathcal { L } _ { i } } { \partial \alpha _ { i } } = \sum _ { i = 1 } ^ { d } \mathbb { E } [ ( \hat { y } ( w ) - \chi _ { i } ( w ) ) \cdot \gamma _ { i } ( \beta _ { i } - \sum _ { j = 1 } ^ { d } \beta _ { j } \gamma _ { j } ) ] } \quad } & { } \\ & { = \sum _ { i = 1 } ^ { d } \mathbb { E } [ ( \sum _ { j = 1 } ^ { d } \beta _ { i } \gamma _ { j } - 1 - 2 \cdot 1 \{ C _ { i } ( w ) > 0 \} + 1 ) \cdot \Re ( \beta _ { i } - \sum _ { j = 1 } ^ { d } \beta _ { j } \gamma _ { j } ) ] } \\ & { = \mathbb { E } [ \sum _ { i = 1 } ^ { d } \beta _ { i } \gamma _ { i } \cdot ( \sum _ { j = 1 } ^ { d } \beta _ { j } \gamma _ { j } ) - 2 \sum _ { i = 1 } ^ { d } \beta _ { i } \gamma _ { i } - \sum _ { j = 1 } ^ { d } \tau ( \sum _ { j = 1 } ^ { d } \beta _ { j } \gamma _ { j } ) ^ { 2 } + 2 \sum _ { i = 1 } ^ { d } \eta ( \sum _ { j = 1 } ^ { d } \beta _ { j } \gamma _ { j } ) ^ { 2 } ] } \\ & { = \mathbb { E } [ ( \sum _ { i = 1 } ^ { d } \beta _ { i } \gamma _ { i } ) ^ { 2 } - 2 \sum _ { i = 1 } ^ { d } \beta _ { i } \gamma _ { i } - ( \sum _ { j = 1 } ^ { d } \beta _ { j } \gamma _ { j } ) ^ { 2 } + 2 \sum _ { j = 1 } ^ { d } \beta _ { j } \gamma _ { j } ] = 0 } \end{array}
$$

So there exists $s \in [ d ]$ such that $\frac { \partial \mathcal { L } } { \partial \alpha _ { s } } \geq 0$ and $\alpha _ { s } ^ { \prime } \leq 0$ . Consider the sum of gradients of $\mathcal { L } _ { s }$ across all indices:

$$
\sum _ { i = 1 } ^ { d } \frac { \partial \mathcal { L } _ { s } } { \partial \alpha _ { i } } = \sum _ { i = 1 } ^ { d } \mathbb { E } \left[ ( \hat { y } - y ) \cdot \gamma _ { i } \cdot \left( \beta _ { i } - \sum _ { j = 1 } ^ { d } \beta _ { j } \gamma _ { j } \right) \right] = \mathbb { E } \left[ ( \hat { y } - y ) \cdot \sum _ { i = 1 } ^ { d } \gamma _ { i } \cdot \left( \beta _ { i } - \sum _ { j = 1 } ^ { d } \beta _ { j } \gamma _ { j } \right) \right] = 0 ,
$$

hence it follows $\textstyle \sum _ { i = 1 } ^ { d } \alpha _ { i } ^ { \prime } = 0$ regardless of the learning rate. Therefore, $\sum _ { i = 1 } ^ { d } \exp ( \alpha _ { i } ^ { \prime } ) \geq d \cdot { \Bigg ( } \prod _ { i = 1 } ^ { d } \exp ( \alpha _ { i } ^ { \prime } ) { \Bigg ) } ^ { \frac { 1 } { d } } = d . \mathrm { ~ A t ~ }$ the same time, $\alpha _ { s } ^ { \prime } \leq 0$ , which results in $\exp ( \alpha _ { s } ^ { \prime } ) \le 1$ . We will shorthand $a _ { i } : = \exp ( \alpha _ { i } ^ { \prime } )$ in the remaining analysis. Now we consider a negative string z with length m, where $m \geq d ,$ and the set of its positive counterparts $Z$ where each $z ^ { \prime } \in Z$ is obtained by inserting exactly one symbol s into z. From (9) we have $\begin{array} { r } { \frac { \rho ( Z ) } { \rho ( z ) } \geq \frac { 1 } { 2 } } \end{array}$

For each counter part $z ^ { \prime } ,$ write $\gamma _ { s } ( z ^ { \prime } ) = \frac { a _ { s } } { \sum _ { i \neq s } C _ { i } ( z ) a _ { i } + a _ { s } }$ . Consider the two following cases:

• max $a _ { i } \geq \frac { d } { 2 } .$ In this case, the probability that the maximizer i appears in string z is $\begin{array} { r } { 1 - \left( 1 - \frac { 1 } { d - 1 } \right) ^ { m } \ge } \end{array}$ <sup>i</sup≯<sup>=s</sup> $\begin{array} { r } { 1 - \left( 1 - \frac { 1 } { d - 1 } \right) ^ { \frac { d } { 2 } } \geq 1 - \exp \left( - \frac { 1 } { 2 } \right) \geq \frac { 1 } { 8 } } \end{array}$ , under which case we have $\gamma _ { s } ( z ^ { \prime } ) \leq \frac { 1 } { \frac { d } { 2 } + 1 } = \frac { 2 } { d + 2 } < \frac { 4 } { d } .$

• max $a _ { i } < \frac { d } { 2 } .$ . In this case, consider the r.v. $\xi \sim \operatorname { U n i f } ( \{ a _ { i } : i \neq s \} )$ , and we rewrite $\sum _ { i \neq s } C _ { i } ( z ) a _ { i } = \sum _ { t = 1 } ^ { m } \xi _ { t } : = S .$ <sup>i</sup≯<sup>=s</sup>

We further derive the first and the second moments of $\xi \colon$

$$
\mathbb { E } [ \xi ] = \frac { 1 } { d - 1 } \sum _ { i \neq s } a _ { i } \ge \frac { 1 } { d - 1 } \cdot ( d - 1 ) = 1 , \quad \mathbb { E } [ \xi ^ { 2 } ] \le \operatorname* { m a x } _ { i \neq s } a _ { i } \mathbb { E } [ \xi ] \le \frac { d } { 2 } \cdot \mathbb { E } [ \xi ] .
$$

Hence, the first and the second moments of S can be written as:

$$
\mathbb { E } [ S ] = m \mathbb { E } [ \xi ] , \quad \mathbb { E } [ S ^ { 2 } ] = m \mathbb { E } [ \xi ^ { 2 } ] + m ( m - 1 ) \mathbb { E } [ \xi ] ^ { 2 } \leq m \cdot { \frac { d } { 2 } } \cdot \mathbb { E } [ \xi ] + m ( m - 1 ) \mathbb { E } [ \xi ] ^ { 2 } .
$$

Using the Paley-Zygmund inequality, we have:

$$
\mathbb { P } \left( S \ge \frac { 1 } { 2 } \mathbb { E } [ S ] \right) \ge \frac { \mathbb { E } [ S ] ^ { 2 } } { 4 \mathbb { E } [ S ^ { 2 } ] } \ge \frac { m ^ { 2 } \mathbb { E } [ \xi ] ^ { 2 } } { 4 m _ { 2 } ^ { d } \mathbb { E } [ \xi ] + 4 m ( m - 1 ) \mathbb { E } [ \xi ] ^ { 2 } } = \frac { m \mathbb { E } [ \xi ] } { 2 d + 4 ( m - 1 ) \mathbb { E } [ \xi ] } \ge \frac { m } { 2 d + 4 ( m - 1 ) } \ge \frac { 1 } { 2 \cdot \frac { d } { m } + 4 } \ge \frac { 1 } { 8 } .
$$

Since $\mathbb { E } [ S ] \ge m$ , we have $\mathbb { P } \left( S \geq \frac { m } { 2 } \right) \geq \frac { 1 } { 8 }$ . Under which case we have $\gamma _ { s } ( z ^ { \prime } ) \leq \frac { a _ { s } } { a _ { s } + \frac { m } { 2 } } \leq \frac { 4 } { d + 4 } < \frac { 4 } { d } .$

For each $m \geq d ,$ let $\begin{array} { r } { G _ { m } : = \big \{ z \in ( \Sigma \setminus \{ s \} ) ^ { m } : \gamma _ { s } ( z ^ { \prime } ) < \frac { 4 } { d } } \end{array}$ for every $z ^ { \prime } \in Z ( z ) \big \}$

Conditional on $| z | = m$ and $\chi _ { s } ( z ) = - 1$ , the symbols of z are i.i.d. uniform on $\Sigma \ \backslash \ \{ s \}$ . The two cases before show that $\mathbb { P } ( z ~ \in ~ G _ { m } ~ | ~ | z | ~ = ~ m , y ~ = ~ - 1 ) ~ \ge ~ \frac { 1 } { 8 }$ . Fix any final output vector $\beta$ with $\beta _ { s } \ \leq \ d / 4$ , and write $\begin{array} { r } { G ( z ) : = \sum _ { i \neq s } \beta _ { i } \gamma _ { i } ( z ) } \end{array}$ . For every $z \in G _ { m }$ , either $G ( z ) \geq 1 / 2$ , in which case $\begin{array} { r } { \ell ( z ) = \frac { 1 } { 2 } G ( z ) ^ { 2 } \geq \frac { 1 } { 8 } , } \end{array}$ or $G ( z ) < 1 / 2$ In the latter case, for every $z ^ { \prime } \in Z ( z )$ , we have $\begin{array} { r } { f _ { \alpha ^ { \prime } , \beta } ( z ^ { \prime } ) - 1 = G ( z ) ( 1 - \gamma _ { s } ( z ^ { \prime } ) ) + \beta _ { s } \gamma _ { s } ( z ^ { \prime } ) - 2 < \frac { 1 } { 2 } + 1 - 2 = - \frac { 1 } { 2 } } \end{array}$ 2 so $\ell { \left( z ^ { \prime } \right) } \geq 1 / 8$

Consequently, it follows that $\begin{array} { r } { \rho ( z ) \ell ( z ) + \sum _ { z ^ { \prime } \in Z ( z ) } \rho ( z ^ { \prime } ) \ell ( z ^ { \prime } ) \geq \frac { 1 } { 8 } \operatorname* { m i n } \{ \rho ( z ) , \rho ( Z ( z ) ) \} \geq \frac { 1 } { 1 6 } \rho ( z ) } \end{array}$ . The sets $\{ z \} \cup Z ( z )$ are disjoint across negative strings z. Hence, with $\begin{array} { r } { G : = \bigcup _ { m \geq d } G _ { m } , \ \mathcal { L } _ { s } ( \alpha ^ { \prime } , \beta ) \ \geq \ \frac { 1 } { 1 6 } \rho ( G ) \ \geq \ \frac { 1 } { 1 2 8 } \mathbb { P } _ { \rho } ( y ( w ) \ = } \end{array}$ $\begin{array} { r } { - 1 , \ | w | \ge d ) = \frac { 1 } { 2 5 6 } \left( 1 - \frac { 2 } { d } \right) ^ { d - 1 } \ge \frac { 1 } { 6 9 1 2 } . } \end{array}$

For the last inequality, when $\begin{array} { r } { d \ge 9 , ( d - 1 ) \ln \left( 1 - \frac { 2 } { d } \right) \ge - \frac { 2 ( d - 1 ) } { d - 2 } \ge - \frac { 1 6 } { 7 } > - 3 . } \end{array}$ , and $e ^ { - 3 } > 1 / 2 7$ . Combining this bound with the earlier case $\beta _ { s } > d / 4$ , which gives $\bar { \mathcal { L } _ { s } } ( \alpha ^ { \prime } , \beta ) \bar { \geq } \bar { 1 } / 2 0 8 0$ , proves the theorem with $c _ { 3 } = 1 / 6 9 1 2$ .

## A.7 Proof of Theorem 2

Proof. Throughout this proof, $\beta ^ { \prime }$ denotes the output weights after Phase 1, $\mathbf { { \boldsymbol { \alpha } } ^ { \prime } }$ the attention weights after Phase 2, and $\beta ( t )$ the output weights after $t \geq 0$ steps of Phase 3, so that $\beta ( 0 ) = \beta ^ { \prime }$ and $\beta ^ { \prime \prime } = \beta ( T )$ . We first prove the following lemma.

Lemma 2 (Induction on Symmetry). For any target index $s \in [ d ]$ , if at the current step we have $\beta _ { i } = \beta _ { j }$ and $\alpha _ { i } = \alpha _ { j }$ for every pair of background indices $i , j \neq s ,$ then after one step of population GD, the updated parameters satisfy $\alpha _ { i } ^ { \prime } = \alpha _ { j } ^ { \prime }$ , and $\beta _ { i } ^ { \prime } = \beta _ { j } ^ { \prime }$

Proof. For any fixed target index $s ,$ let $\tau _ { i , j } ~ : = ~ ( i ~ j ) ~ \in ~ \mathrm { S y m } ( [ d ] )$ denote the transposition swapping $i , j \ \ne \ s$ while keeping other indices fixed. ${ \mathrm { F o r ~ a n y ~ s t r i n g ~ } } w \ = \ ( x _ { 1 } , x _ { 2 } \cdot \cdot \cdot x _ { m } ) \in \ \Sigma ^ { m }$ , we write $\tau _ { i , j } ( w ) =$ $\tau _ { i , j } ( x _ { 1 } ) \tau _ { i , j } ( x _ { 2 } ) \cdot \cdot \cdot \tau _ { i , j } ( x _ { m } )$ . It follows that $\rho ( w ) = \rho ( \tau _ { i , j } ( w ) )$ for every $w \in \Sigma ^ { + }$ . Moreover, since $i , j \ \ne s _ { ; }$ swapping $v _ { i }$ and $v _ { j }$ will not change the acceptance of the string, i.e., $y ( w ) = y ( \tau _ { i , j } ( w ) )$

Next, define the corresponding action on parameters: for $( \alpha , \beta ) \in \mathbb { R } ^ { d } \times \mathbb { R } ^ { d }$ , let $\tau _ { i , j } ( \alpha , \beta )$ denote the parameter vector obtained by swapping $\alpha _ { i }$ with $\alpha _ { j }$ and $\beta _ { i }$ with $\beta _ { j }$ , leaving all other coordinates fixed. It follows that $f _ { \alpha , \beta } ( w ) = f _ { \tau _ { i , j } ( \alpha , \beta ) } ( \tau _ { i , j } ( w ) )$ for every $w \in \Sigma ^ { + }$

Combining these three facts, the population loss is invariant under this joint relabeling:

$$
\begin{array} { r l } & { \mathcal { L } ( { \alpha } , \beta ) = \mathbb { E } _ { w \sim \rho } \left[ \left( f _ { \alpha , \beta } ( w ) - y ( w ) \right) ^ { 2 } \right] = \mathbb { E } _ { w \sim \rho } \left[ \left( f _ { \tau _ { i , j } ( \alpha , \beta ) } ( \tau _ { i , j } ( w ) ) - y ( \tau _ { i , j } ( w ) ) \right) ^ { 2 } \right] } \\ & { \quad \quad \quad = \mathbb { E } _ { w ^ { \prime } \sim \rho } \left[ \left( f _ { \tau _ { i , j } ( \alpha , \beta ) } ( w ^ { \prime } ) - y ( w ^ { \prime } ) \right) ^ { 2 } \right] = \mathcal { L } \left( \tau _ { i , j } ( \alpha , \beta ) \right) . } \end{array}
$$

Therefore, for every pair of weights $( \alpha , \beta )$ , it holds that $\begin{array} { r } { \frac { \partial \mathcal { L } } { \partial \alpha _ { i } } \left( \tau _ { i , j } ( \alpha , \beta ) \right) ~ = ~ \frac { \partial \mathcal { L } } { \partial \alpha _ { i } } ( \alpha , \beta ) , ~ \frac { \partial \mathcal { L } } { \partial \beta _ { i } } \left( \tau _ { i , j } ( \alpha , \beta ) \right) ~ = ~ } \end{array}$ $\frac { \partial \mathcal { L } } { \partial \beta _ { j } } ( \pmb { \alpha } , \beta )$ . Since the current parameters satisfy $\alpha _ { i } = \alpha _ { j }$ and $\beta _ { i } = \beta _ { j }$ , then $\tau _ { i , j } ( \alpha , \beta ) = ( \alpha , \beta )$ , and the identities above rewrites as $\begin{array} { r } { \frac { \partial \mathcal { L } } { \partial \alpha _ { i } } ( \alpha , \beta ) = \frac { \partial \mathcal { L } } { \partial \alpha _ { j } } ( \alpha , \beta ) , \frac { \partial \mathcal { L } } { \partial \beta _ { i } } ( \alpha , \beta ) = \frac { \partial \mathcal { L } } { \partial \beta _ { j } } ( \alpha , \beta ) } \end{array}$ . Plugging these equalities into the GD update rule we have $\begin{array} { r } { \alpha _ { i } ^ { \prime } = \alpha _ { i } - \eta \frac { \partial \mathcal { L } } { \partial \alpha _ { i } } = \alpha _ { j } - \eta \frac { \partial \mathcal { L } } { \partial \alpha _ { j } } = \alpha _ { j } ^ { \prime } } \end{array}$ . Same equality holds for $\beta$ as well. □

With this lemma, and the symmetrical initialisation for $_ { \pmb { \alpha } }$ and $\beta ,$ the parameter can be reduced to 4 scalars, $\alpha _ { s }$ and $\beta _ { s }$ for the target, and $\alpha _ { j } , \beta _ { j }$ as the common weight for the background indices. We also shorthand $\gamma _ { i } ( w )$ as the aggregated softmax-attention score for symbol $v _ { i } ,$ note that sample-wise there is no symmetry for $\gamma _ { i }$ when $i \neq s ,$ but on expectation level we have $\mathbb { E } [ \gamma _ { j } ]$ which is shared by all background indices $j \neq s$ . We also shorthand $\Delta \beta = \beta _ { s } - \beta _ { j }$ and $\begin{array} { r } { \bar { \beta } = \frac { 1 } { d - 1 } \sum _ { i \neq s } \beta _ { i } } \end{array}$

First Phase In the first step, we calculate the gradient for $\beta _ { s }$ and $\beta _ { j }$ in closed forms:

$$
\begin{array} { r l } & { \frac { \partial \mathcal { L } } { \partial \beta _ { s } } = \mathbb { E } \left[ ( \hat { y } ( w ) - y ( w ) ) \gamma _ { s } \right] \overset { ( a ) } { = } \mathbb { E } \left[ ( - 1 - y ( w ) ) \gamma _ { s } \right] = \mathbb { P } ( y = 1 ) \cdot - 2 \cdot \mathbb { E } \left[ \gamma _ { s } \mid y = 1 \right] = - \frac { 2 } { d } \cdot } \\ & { \frac { \partial \mathcal { L } } { \partial \beta _ { j } } = \mathbb { E } \left[ ( \hat { y } ( w ) - y ( w ) ) \gamma _ { j } \right] = \mathbb { E } \left[ ( - 1 - y ( w ) ) \gamma _ { j } \right] = \mathbb { P } ( y = 1 ) \cdot - 2 \cdot \mathbb { E } \left[ \gamma _ { j } \mid y = 1 \right] = \frac { 1 } { d - 1 } \cdot \left( \frac { 2 } { d } - 1 \right) } \end{array}
$$

Hence, the updated parameter is $\begin{array} { r } { \beta _ { s } ^ { \prime } = \frac { 2 \eta _ { 1 } } { d } , \beta _ { j } ^ { \prime } = \frac { \eta _ { 1 } } { d - 1 } \cdot \left( 1 - \frac { 2 } { d } \right) } \end{array}$ for every $j \neq s .$ . The separation of target and the background output weights grows from $\Delta \beta = 0$ to $\begin{array} { r } { \dot { \Delta \beta ^ { \prime } } = \frac { \eta _ { 1 } ^ { \alpha } } { d - 1 } } \end{array}$ . We choose the learning rate of the first phase to be $\begin{array} { r } { 0 < \eta _ { 1 } < \frac { d - 1 } { 2 } } \end{array}$ and therefore the separation will be controlled within a constant range $\begin{array} { r } { 0 < \Delta \beta ^ { \prime } < \frac { 1 } { 2 } } \end{array}$

Second Phase In the second phase, we calculate the gradient for $\alpha _ { s }$ and $\alpha _ { j }$ in closed forms:

$$
\begin{array} { r l } & { \displaystyle \frac { \partial \mathcal { L } } { \partial \alpha _ { s } } = \Delta \beta ^ { \prime } \cdot \mathbb { E } \left[ \left( \hat { y } ( w ) - y ( w ) \right) \cdot \gamma _ { s } ( 1 - \gamma _ { s } ) \right] } \\ & { \displaystyle \frac { \partial \mathcal { L } } { \partial \alpha _ { j } } = - \frac { 1 } { d - 1 } \cdot \frac { \partial \mathcal { L } } { \partial \alpha _ { s } } } \end{array}
$$

Consider $\begin{array} { r } { 0 < \eta _ { 1 } < \frac { d - 1 } { 2 } } \end{array}$ , we have $\beta _ { i } ^ { \prime } < 1$ for every $i \in [ d ]$ , therefore for positive words $\begin{array} { r } { \hat { y } ( w ) = \sum _ { i = 1 } ^ { d } \gamma _ { i } \beta _ { i } ^ { \prime } - 1 < 0 } \end{array}$ and $f - y < - 1$ , hence we get an upper bound for the target gradient (which holds for negative words too since $\gamma _ { s } = 0 )$

$$
\frac { \partial \mathcal { L } } { \partial \alpha _ { s } } \leq - \Delta \beta ^ { \prime } \mathbb { E } [ \gamma _ { s } ( 1 - \gamma _ { s } ) ]
$$

At uniform attention, the moments in (3) and (4) give $\begin{array} { r } { d \mathbb { E } { [ \gamma _ { s } ( 1 - \gamma _ { s } ) ] } = \frac { d - 1 } { d } \left( 1 - \frac { \ln ( d - 1 ) } { d - 2 } \right) \ge \frac { 2 } { 3 } ( 1 - \ln 2 ) > \frac { 1 } { 2 0 } } \end{array}$ for every $d \geq 3$ . Hence we have $\begin{array} { r } { \frac { \partial \mathcal { L } } { \partial \alpha _ { s } } \leq - \frac { \eta _ { 1 } } { 2 0 d ( d - 1 ) } } \end{array}$ and $\begin{array} { r } { \frac { \partial \mathcal { L } } { \partial \alpha _ { j } } > \frac { \eta _ { 1 } } { 2 0 d ( d - 1 ) ^ { 2 } } } \end{array}$ . Combining both bounds, the attention gap after the second step satisfies

$$
\Delta \alpha ^ { \prime } : = \alpha _ { s } ^ { \prime } - \alpha _ { j } ^ { \prime } = - \frac { d } { d - 1 } \eta _ { 2 } \frac { \partial \mathcal { L } } { \partial \alpha _ { s } } \geq \frac { \eta _ { 1 } \eta _ { 2 } } { 2 0 ( d - 1 ) ^ { 2 } } \geq \ln \frac { 8 d } { \varepsilon } .
$$

For a positive string w with $| w | = m$ and $C _ { s } ( w ) = n \geq 1$ ，

$$
1 - \gamma _ { s } ^ { \prime } ( w ) = \frac { m - n } { n \exp ( \Delta \alpha ^ { \prime } ) + m - n } \ \leq \ ( m - 1 ) \exp ( - \Delta \alpha ^ { \prime } ) .
$$

Since $\begin{array} { r } { \mathbb { E } \left[ m - 1 \ \middle | \ y = 1 \right] \leq 2 \mathbb { E } \left[ m - 1 \right] = 2 ( d - 2 ) } \end{array}$ , we obtain

$$
\mathbb { E } \left[ 1 - \gamma _ { s } ^ { \prime } \vert y = 1 \right] \le \delta : = 2 d \exp ( - \Delta \alpha ^ { \prime } ) \le \frac { \varepsilon } { 4 } .
$$

Third Phase During the third step, the attention is fixed at $\alpha ^ { \prime } .$ , so $\mathcal { L } ( \alpha ^ { \prime } , \beta )$ is a convex quadratic function $\mathrm { w . r . t }$ $\beta$ with Hessian $H = \mathbb { E } [ \gamma ^ { \prime } \gamma ^ { \prime \top } ]$ . Let $\mu _ { k } ^ { \prime } : = \mathbb { E } \left[ ( \gamma _ { s } ^ { \prime } ) ^ { k } \mid y = 1 \right]$ for $k = 1 , 2 , 3$ , so that $1 - \delta \leq \mu _ { 1 } ^ { \prime }$ and $( \mu _ { 1 } ^ { \prime } ) ^ { 2 } \leq \mu _ { 2 } ^ { \prime } \leq 1$ Let $S : = \operatorname { s p a n } \{ e _ { s } , { \bf 1 } _ { - s } \}$ and $u : = { { \bf 1 } } _ { - s } / \sqrt { d - 1 }$

On $S ,$ the risk is minimized at $\beta ^ { \star }$ with:

$$
\beta _ { s } ^ { \star } = \frac { 2 \mu _ { 1 } ^ { \prime } + 2 \mu _ { 2 } ^ { \prime } - 2 ( \mu _ { 1 } ^ { \prime } ) ^ { 2 } } { 2 \mu _ { 2 } ^ { \prime } - ( \mu _ { 1 } ^ { \prime } ) ^ { 2 } } , \quad \beta _ { j } ^ { \star } = \frac { 2 \mu _ { 2 } ^ { \prime } - 2 ( \mu _ { 1 } ^ { \prime } ) ^ { 2 } } { 2 \mu _ { 2 } ^ { \prime } - ( \mu _ { 1 } ^ { \prime } ) ^ { 2 } } .
$$

And the expected risk at this point will be:

$$
\begin{array} { l } { \displaystyle { \mathcal { L } ( { \boldsymbol { \alpha } } ^ { \prime } , { \boldsymbol { \beta } } ^ { \star } ) = \frac { 1 } { 2 } \cdot \mathbb { E } \left[ \frac { 1 } { 2 } \left( \beta _ { s } ^ { \star } \gamma _ { s } ^ { \prime } + \sum _ { j \neq s } \beta _ { j } ^ { \star } \gamma _ { j } ^ { \prime } - 1 - 1 \right) ^ { 2 } \mid y = 1 \right] + \frac { 1 } { 2 } \cdot \mathbb { E } \left[ \frac { 1 } { 2 } \left( \sum _ { j \neq s } \beta _ { j } ^ { \star } \gamma _ { j } ^ { \prime } - 1 + 1 \right) ^ { 2 } \mid y = - 1 \right] } } \\ { \displaystyle { = \frac { 1 } { 4 } \cdot \left( \mathbb { E } \left[ \left( \Delta \beta ^ { \star } \gamma _ { s } ^ { \prime } + \beta _ { j } ^ { \star } - 2 \right) ^ { 2 } \mid y = 1 \right] + \left( \beta _ { j } ^ { \star } \right) ^ { 2 } \right) } } \\ { \displaystyle { = \frac { 1 } { 4 } \left( \Delta \beta ^ { \star } \right) ^ { 2 } \mu _ { 2 } ^ { \prime } + \frac { 1 } { 2 } \Delta \beta ^ { \star } ( \beta _ { j } ^ { \star } - 2 ) \mu _ { 1 } ^ { \prime } + \frac { 1 } { 4 } ( \beta _ { j } ^ { \star } - 2 ) ^ { 2 } + \frac { 1 } { 4 } ( \beta _ { j } ^ { \star } ) ^ { 2 } = \frac { \mu _ { 2 } ^ { \prime } - \left( \mu _ { 1 } ^ { \prime } \right) ^ { 2 } } { 2 \mu _ { 2 } ^ { \prime } - \left( \mu _ { 1 } ^ { \prime } \right) ^ { 2 } } < 2 \delta } } \end{array}
$$

Moreover, we have $\beta _ { j } ^ { \star } = 2 \mathcal { L } ( \alpha ^ { \prime } , \beta ^ { \star } ) \in [ 0 , 4 \delta ]$ and $\begin{array} { r } { 0 \le \beta _ { s } ^ { \star } \le 4 \delta + \frac { 2 } { \mu _ { 1 } ^ { \prime } } < 4 . } \end{array}$

At any step in the third phase, the symmetry is preserved for the $\alpha ^ { \prime } , \beta$ , and the $\mathbb { E } [ \gamma ^ { \prime } ]$ among all indices $j \neq s ,$ as a result of Lemma 2. Since $\beta ^ { \star }$ minimizes the risk over S, its gradient is orthogonal to S and hence zero. Therefore, for every $\beta \in \mathbb { R } ^ { d }$

$$
\begin{array} { r } { \mathcal L ( \boldsymbol { \alpha } ^ { \prime } , \boldsymbol { \beta } ) - \mathcal L ( \boldsymbol { \alpha } ^ { \prime } , \boldsymbol { \beta } ^ { \star } ) = \frac { 1 } { 2 } ( \boldsymbol { \beta } - \boldsymbol { \beta } ^ { \star } ) ^ { \top } H ( \boldsymbol { \beta } - \boldsymbol { \beta } ^ { \star } ) , } \end{array}
$$

and the GD iterates with $\eta _ { 3 } = 1$ satisfy $\beta ( t + 1 ) - \beta ^ { \star } = ( I - H ) ( \beta ( t ) - \beta ^ { \star } )$ . By Lemma 2, all iterates lie in $S ,$ so only the restriction H<sup>˜</sup> of H to S matters. Take an orthonormal basis $\left( e _ { s } , \frac { { \bf 1 } _ { - s } } { \sqrt { d - 1 } } , \ \xi _ { 1 } , \ldots , \xi _ { d - 2 } \right)$ of $\mathbb { R } ^ { d }$ , where $\xi _ { 1 } , \ldots , \xi _ { d - 2 }$ span the subspace $\{ e _ { s } , \mathbf { 1 } _ { - s } \} ^ { \perp }$ . We can then block-diagonalize H in this basis as $H = \tilde { H } \oplus a I _ { d - 2 }$ with $a = \Theta \left( { \textstyle { \frac { \ln d } { d ^ { 2 } } } } \right)$ , where $\bar { H } \in \mathbb { R } ^ { \bar { 2 } \times 2 }$ acts on span $\{ e _ { s } , \mathbf { 1 } _ { - s } \}$ . Each component of H<sup>˜</sup> has closed forms:

$$
\begin{array} { r l r } & { } & { \tilde { H } _ { 1 1 } = \mathbb { E } [ ( \gamma _ { s } ^ { \prime } ) ^ { 2 } ] = \frac { 1 } { 2 } \mu _ { 2 } ^ { \prime } \in \left[ \frac { 1 } { 2 } ( 1 - \delta ) ^ { 2 } , \frac 1 2 \right] , } \\ & { } & { \tilde { H } _ { 2 2 } = \frac { \mathbb { E } [ ( 1 - \gamma _ { s } ^ { \prime } ) ^ { 2 } ] } { d - 1 } \in \left[ \frac { 1 } { 2 ( d - 1 ) } , \frac { 1 + \delta } { 2 ( d - 1 ) } \right] , } \\ & { } & { \tilde { H } _ { 1 2 } = \tilde { H } _ { 2 1 } = \frac { \mathbb { E } [ \gamma _ { s } ^ { \prime } ( 1 - \gamma _ { s } ^ { \prime } ) ] } { \sqrt { d - 1 } } \in \left[ 0 , \frac { \delta } { 2 \sqrt { d - 1 } } \right] , \quad } \end{array}
$$

where we used that $\mathbb { E } [ 1 - \gamma _ { s } ^ { \prime } \ | \ y = - 1 ] = 1$ and $\mathbb { E } \left[ 1 - \gamma _ { s } ^ { \prime } \vert y = 1 \right] \leq \delta$ . Hence $\begin{array} { r } { \mathrm { t r } ( \tilde { H } ) \leq \frac { 1 } { 2 } + \frac { 1 + \delta } { 4 } \leq 1 } \end{array}$ for $d \geq 3$ , and

$$
\operatorname* { d e t } ( \tilde { H } ) \geq \frac { ( 1 - \delta ) ^ { 2 } - \delta ^ { 2 } } { 4 ( d - 1 ) } = \frac { 1 - 2 \delta } { 4 ( d - 1 ) } , \qquad \lambda _ { \operatorname* { m i n } } ( \tilde { H } ) \geq \frac { \operatorname* { d e t } ( \tilde { H } ) } { \operatorname { t r } ( \tilde { H } ) } \geq \frac { 1 } { 8 d } ,
$$

using $\delta \leq { \frac { 1 } { 4 } }$ . Since also $\lambda _ { \operatorname* { m a x } } ( \tilde { H } ) \leq 1$ , each step contracts the error by a factor of at most $1 - { \textstyle { \frac { 1 } { 8 d } } }$

At the beginning of Phase 3, output starts from $\beta ^ { \prime }$ , with $\beta _ { s } ^ { \prime } \in ( 0 , 1 )$ and $\beta _ { j } ^ { \prime } \in ( 0 , \frac { 1 } { 2 } )$ , so $\| \beta ^ { \prime } - \beta ^ { \star } \| ^ { 2 } \leq \| \beta ^ { \star } \| ^ { 2 } +$ $\| \beta ^ { \prime } \| ^ { 2 } \leq 1 6 + ( d - 1 ) \leq 1 6 d .$ . Therefore $\mathcal { L } ( { \alpha ^ { \prime } } , \beta ( T ) ) - \mathcal { L } ( { \alpha ^ { \prime } } , \beta ^ { \star } ) \leq \frac { 1 } { 2 } \| \beta ( T ) - \beta ^ { \star } \| ^ { 2 } \leq 8 d \left( 1 - \frac { 1 } { 8 d } \right) ^ { 2 T } \leq$ 8d exp $\left( - \frac { T } { 4 d } \right) \ \leq \ \frac { \varepsilon } { 2 }$ when plugging in $\begin{array} { r } { T \geq 4 d \ln { \frac { 1 6 d } { \varepsilon } } } \end{array}$ . After $T$ steps we have $\beta ^ { \prime \prime } : = \beta ( T )$ and hence $\begin{array} { r } { { \cal { L } } ( \alpha ^ { \prime } , \beta ^ { \prime \prime } ) \le } \end{array}$ $2 \delta + { \frac { \varepsilon } { 2 } } \leq \varepsilon .$ □

## A.8 Proof of Corollary 1

Proof. We use $( w ^ { ( k ) } , y ^ { ( k ) } ) _ { k = 1 } ^ { N _ { 1 } }$ denotes the first batch, and for any function $g ( w )$ of a sample we write $\hat { g } : =$ $\begin{array} { r } { \frac { 1 } { N _ { 1 } } \sum _ { k = 1 } ^ { N _ { 1 } } g ( w ^ { ( k ) } , y ^ { ( k ) } ) } \end{array}$ for its average over this batch. In Phases 1 and 2, $\begin{array} { r } { \pmb { \alpha } = \mathbf { 0 } _ { d } , \mathrm { s o } \ \gamma _ { i } ( w ) = \frac { C _ { i } ( w ) } { | w | } } \end{array}$

With respect to the first draw of samples, we consider the following variables. For any sample $( w , y )$ and $i \in [ d ]$ we define $Z _ { i } : = { \bf 1 } \{ y = 1 \} \gamma _ { i } ( w ) , V : = \gamma _ { s } ( w ) ( 1 - \gamma _ { s } ( w ) )$ . Note that none of these quantities depend on the trained parameters but their randomness come from the raw strings that are drawn.

Lemma 3 (Concentration on the first batch). For every $c > 0 _ { ; }$ , there exists a constant $C > 0$ such that, when $N _ { 1 } \geq C d \ln d$ , with probability at least $1 - d ^ { - c }$ , the following event A holds :

1. $\begin{array} { r } { \left| \hat { Z } _ { i } - \mathbb { E } [ Z _ { i } ] \right| \leq \frac { 1 } { 8 0 0 d } } \end{array}$ for every $i \in [ d ] ,$

2. ${ \hat { V } } \geq { \frac { 1 } { 2 } } \mathbb { E } [ V ]$ ;

3. $\textstyle { \hat { \gamma } } _ { j } \leq { \frac { 2 } { d } }$ for every $j \neq s$

Proof. All of $Z _ { i } , V , \gamma _ { j }$ take values in [0, 1]. We can bound their second moments by:

$$
\mathbb { E } [ Z _ { i } ^ { 2 } ] = \frac { 1 } { 2 } \cdot \mathbb { E } [ \gamma _ { i } ^ { 2 } \mid y = 1 ] = \frac { 1 } { ( d - 1 ) ^ { 2 } } + \frac { \ln ( d - 1 ) } { ( d - 1 ) ( d - 2 ) } \leq \frac { 1 + \ln ( d - 1 ) } { ( d - 1 ) ^ { 2 } } ;
$$

$$
\mathbb { E } [ V ^ { 2 } ] \leq \mathbb { E } [ \gamma _ { s } ^ { 2 } \mid y = 1 ] \leq \frac { 2 + 2 \ln d } { d ^ { 2 } } ;
$$

$$
\mathbb { E } [ \gamma _ { j } ^ { 2 } ] = \frac { 1 } { d ^ { 2 } } + \frac { ( d - 1 ) \ln ( d - 1 ) } { d ^ { 2 } ( d - 2 ) } \le \frac { 1 + 2 \ln d } { d ^ { 2 } } .
$$

Hence, we can use $\textstyle \sigma ^ { 2 } = { \frac { 2 + 2 \ln d } { ( d - 1 ) ^ { 2 } } } \leq { \frac { 1 2 \ln d } { d ^ { 2 } } }$ as a common upper bound for the three terms above. And we also have $\begin{array} { r } { \mathbb { E } [ V ] = \mathbb { E } [ \gamma _ { s } ] - \mathbb { E } [ \gamma _ { s } ^ { 2 } ] = \frac { 1 } { d } - \frac { 1 } { d ^ { 2 } } - \frac { ( d - 1 ) \ln ( d - 1 ) } { d ^ { 2 } ( d - 2 ) } \ge \frac { 1 } { 5 d } } \end{array}$ for all $d \geq 3$ . By Bernstein’s inequality, for each statement in the event $A ,$ the failure probability of the each term can be bounded by:

$$
\mathbb { P } ( | \hat { Z } _ { i } - \mathbb { E } [ Z _ { i } ] | > \frac { 1 } { 8 0 0 d } ) < 2 \exp \left( - \frac { N _ { 1 } \left( \frac { 1 } { 8 0 0 d } \right) ^ { 2 } } { 2 \cdot \frac { 1 2 \ln d } { d ^ { 2 } } + \frac { 1 } { 1 2 0 0 d } } \right) < 2 \exp \left( - \frac { N _ { 1 } } { C _ { 0 } \ln d + C _ { 1 } d } \right) ,
$$

$$
\begin{array} { r } { \mathbb { P } ( \mathbb { E } [ V ] - \hat { V } > \frac { 1 } { 2 } \mathbb { E } [ V ] ) < \exp \left( - \frac { N _ { 1 } \left( \frac { 1 } { 1 0 d } \right) ^ { 2 } } { 2 \cdot \frac { 1 2 \ln d } { d ^ { 2 } } + \frac { 1 } { 1 5 d } } \right) < \exp \left( - \frac { N _ { 1 } } { C _ { 0 } ^ { \prime } \ln d + C _ { 1 } ^ { \prime } d } \right) } \end{array}
$$

$$
\mathbb { P } ( \hat { \gamma } _ { j } - \mathbb { E } [ \gamma _ { j } ] > \frac { 1 } { d } ) < \exp \left( - \frac { N _ { 1 } \left( \frac { 1 } { d } \right) ^ { 2 } } { 2 \cdot \frac { 1 2 \ln d } { d ^ { 2 } } + \frac { 2 } { 3 d } } \right) < \exp \left( - \frac { N _ { 1 } } { C _ { 0 } ^ { \prime \prime \prime } \ln d + C _ { 1 } ^ { \prime \prime \prime } d } \right)
$$

Taking the union bound of all the $2 d + 1$ statements, choosing $N _ { 1 } \geq C d \ln d$ with C large enough (depending on c) makes the combined failure at most $\frac { d ^ { - c } } { 2 d + 1 }$ □

First Phase. When $\alpha = \beta = \mathbf { 0 } _ { d }$ , the output is $f ( w ) = - 1$ on every string, so $\frac { \partial \hat { \mathcal { L } } } { \partial \beta _ { i } } = - 2 \hat { Z } _ { i }$ and the updated output weight becomes $\hat { \beta } _ { i } ^ { \prime } = 2 \eta _ { 1 } \hat { Z } _ { i } \ge 0$ . By the moment of the attention scores specified in (6), (7), (8), and $\mathbb { P } ( y = - 1 ) = \textstyle { \frac { 1 } { 2 } }$ , we have $\begin{array} { r } { \mathbb { E } [ Z _ { s } ] = \frac { 1 } { d } , \ \mathbb { E } [ \dot { Z } _ { j } ] = \frac { 1 } { d } - \frac { 1 } { 2 ( d - 1 ) } = \frac { d - 2 } { 2 d ( d - 1 ) } } \end{array}$ for every $j \neq s$ . With Lemma 3 and $0 < \eta _ { 1 } \leq \sqrt { d } .$ we have:

$$
\hat { \beta } _ { s } ^ { \prime } \leq \eta _ { 1 } \left( \frac { 2 } { d } + \frac { 1 } { 4 0 0 d } \right) \leq \frac { 2 . 0 1 } { \sqrt { d } } \leq \frac { 5 } { 4 }\tag{11}
$$

$$
\hat { \beta } _ { s } ^ { \prime } - \operatorname* { m a x } _ { i \neq s } \hat { \beta } _ { i } ^ { \prime } \geq \eta _ { 1 } \left( \frac { 1 } { d - 1 } - \frac { 1 } { 2 0 0 d } \right) \geq \frac { 1 9 9 } { 2 0 0 } \cdot \frac { \eta _ { 1 } } { d } ,\tag{12}
$$

$$
\left| { \hat { \beta } } _ { j } ^ { \prime } - { \hat { \beta } } _ { j ^ { \prime } } ^ { \prime } \right| \leq { \frac { \eta _ { 1 } } { 2 0 0 d } } , { \mathrm { ~ f o r ~ } } j , j ^ { \prime } \neq s ; 0 \leq { \hat { \beta } } _ { j } ^ { \prime } \leq { \frac { 1 . 0 1 \eta _ { 1 } } { d } } .\tag{13}
$$

Second Phase. For a sample $( w , y )$ and $i \in [ d ]$ , we define $\begin{array} { r } { T _ { i } : = ( y - f ( w ) ) \gamma _ { i } \sum _ { j \neq i } \left( \hat { \beta } _ { i } ^ { \prime } - \hat { \beta } _ { j } ^ { \prime } \right) \gamma _ { j } } \end{array}$ , Then the empirical gradient w.r.t. $\alpha _ { i }$ can be written as $\begin{array} { r } { \frac { \partial \hat { \mathcal { L } } } { \partial \alpha _ { i } } ( \mathbf { 0 } _ { d } , \hat { \boldsymbol { \beta } } ^ { \prime } ) = - \hat { T } _ { i } } \end{array}$ , hence for every background symbol $j \neq s \colon$

$$
\hat { \alpha } _ { s } ^ { \prime } - \hat { \alpha } _ { j } ^ { \prime } = \eta _ { 2 } \left( \hat { T } _ { s } - \hat { T } _ { j } \right)\tag{14}
$$

Since the gap and the output weights $\hat { \boldsymbol { \beta } } ^ { \prime }$ depend on the same batch of samples, we will not apply a concentration inequality directly to $T _ { i }$ . Instead, we bound $T _ { i }$ sample by sample, in terms of V and $\gamma _ { j }$

We first consider a lowe bound for $T _ { s }$ . For negative samples, we have $y = - 1$ , then $\gamma _ { s } = 0$ and $T _ { s } = 0 = V$ . If $y = 1$ , then $\begin{array} { r } { y - f ( w ) = 2 - \sum _ { i = 1 } ^ { d } \hat { \beta } _ { i } ^ { \prime } \gamma _ { i } \geq 2 - \hat { \beta } _ { s } ^ { \prime } \geq \frac { 3 } { 4 } } \end{array}$ by (11), and $\begin{array} { r } { \sum _ { i \neq s } ( \hat { \beta } _ { s } ^ { \prime } - \hat { \beta } _ { i } ^ { \prime } ) \gamma _ { i } \geq \frac { 1 9 9 } { 2 0 0 } \frac { \eta _ { 1 } } { d } ( 1 - \gamma _ { s } ) } \end{array}$ by (12). Hence $\begin{array} { r } { T _ { s } \geq \frac { 3 } { 4 } \cdot \frac { 1 9 9 } { 2 0 0 } \cdot \frac { \eta _ { 1 } } { d } V } \end{array}$ for every sample, regardless of positive or negative, and by Lemma 3, when A happens:

$$
\hat { T } _ { s } \ge \frac { 3 } { 4 } \cdot \frac { 1 9 9 } { 2 0 0 } \cdot \frac { \eta _ { 1 } } { d } \cdot \frac { 1 } { 1 0 d } \ge \frac { 7 \eta _ { 1 } } { 1 0 0 d ^ { 2 } } .
$$

Then we bound $T _ { j }$ from above. For positive samples, $y = 1$ , then $0 < y - f ( w ) \leq 2$ , and

$$
\sum _ { i \neq j } \left( \hat { \beta } _ { j } ^ { \prime } - \hat { \beta } _ { i } ^ { \prime } \right) \gamma _ { i } = \left( \hat { \beta } _ { j } ^ { \prime } - \hat { \beta } _ { s } ^ { \prime } \right) \gamma _ { s } + \sum _ { i \neq j , s } \left( \hat { \beta } _ { j } ^ { \prime } - \hat { \beta } _ { i } ^ { \prime } \right) \gamma _ { i } \leq \frac { \eta _ { 1 } } { 2 0 0 d } ,
$$

since the first term is negative and the second is bounded by (13). For negative samples $y = - 1$ , then $\gamma _ { s } = 0 .$ $\begin{array} { r } { | y - f ( w ) | = \sum _ { i \neq s } \hat { \beta } _ { i } ^ { \prime } \gamma _ { i } \le 2 } \end{array}$ , and $\begin{array} { r } { \left| \sum _ { i \neq j } ( \hat { \beta } _ { j } ^ { \prime } - \hat { \beta } _ { i } ^ { \prime } ) \gamma _ { i } \right| \leq \frac { \eta _ { 1 } } { 2 0 0 d } } \end{array}$ . For both positive and negative samples, we have $T _ { j } \leq 2 \gamma _ { j } \cdot \frac { \eta _ { 1 } } { 2 0 0 d }$ , and by Lemma 3, we have $\begin{array} { r } { \hat { T } _ { j } \le \frac { \eta _ { 1 } } { 1 0 0 d } \cdot \frac { 2 } { d } = \frac { 2 \eta _ { 1 } } { 1 0 0 d ^ { 2 } } } \end{array}$

Combining both bounds with (14), on the event $A ,$ the empirical attention gap is bounded by:

$$
\hat { \Delta } : = \hat { \alpha } _ { s } ^ { \prime } - \operatorname* { m a x } _ { j \neq s } \hat { \alpha } _ { j } ^ { \prime } \ \geq \ \frac { \eta _ { 1 } \eta _ { 2 } } { 2 0 d ^ { 2 } } .\tag{15}
$$

Third phase. Conditioned on event A happens with the first batch, we fix $( \hat { \alpha } ^ { \prime } , \hat { \beta } ^ { \prime } )$ . All probabilities below are over the fresh batch $( w ^ { ( k ) } , y ^ { ( k ) } ) _ { k = 1 } ^ { N _ { 3 } } ,$ , which is independent of $( \hat { \alpha } ^ { \prime } , \hat { \beta } ^ { \prime } )$ . From here on, γ(w) denotes the attention weights under $\hat { \alpha } ^ { \prime } , \gamma ^ { ( k ) } : = \gamma ( w ^ { ( k ) } )$ , and we denote the expected as well as the empirical hessians of $\beta$ as $H : = \mathbb { E } \left\lceil \gamma \gamma ^ { \top } \right\rceil$ and $\begin{array} { r } { \hat { H } : = \frac { 1 } { N _ { 3 } } \sum _ { k = 1 } ^ { N _ { 3 } } \gamma ^ { ( k ) } \left( \gamma ^ { ( k ) } \right) ^ { \top } } \end{array}$ . We write $\mathcal { L } ( \beta ) : = \mathcal { L } _ { s } ( \hat { \alpha } ^ { \prime } , \beta )$ and $\hat { \mathcal { L } } ( \beta )$ for the empirical risk on the fresh batch. In contrast to the third phase of population GD, we do not analyze the convergence of $\beta$ to the risk minimizer. Instead, we compare the iterates with the fixed reference point $z : = 2 e _ { s }$ .

We first prove that the reference point has negligible risk. Let $r ( w ) : = f _ { \hat { \alpha } ^ { \prime } , z } ( w ) - y ( w ) = 2 \gamma _ { s } ( w ) - 1 - y ( w )$ . For negative samples, $r ( w ) = 0 - 1 + 1 = 0$ . For positive samples, when $| w | = m$ and $C _ { s } ( w ) = n \geq 1$ , we have

$$
| r ( w ) | = 2 - 2 \gamma _ { s } ( w ) = \frac { 2 \cdot \sum _ { j \neq s } C _ { j } ( w ) \exp ( \hat { \alpha } _ { j } ^ { \prime } ) } { n \exp ( \hat { \alpha } _ { s } ^ { \prime } ) + \sum _ { j \neq s } C _ { j } ( w ) \exp ( \hat { \alpha } _ { j } ^ { \prime } ) } \le 2 ( m - 1 ) \exp ( - \hat { \Delta } ) ,
$$

Note that this bound does not depend on how the attention is distributed among the non-target symbols. Define the random variable $\kappa : = r ( w ) ^ { 2 }$ and $\begin{array} { r } { \hat { \kappa } : = \frac { 1 } { N _ { 3 } } \sum _ { k = 1 } ^ { N _ { 3 } } r ( w ^ { ( k ) } ) ^ { 2 } } \end{array}$ . Since E $\left\lceil ( m - 1 ) ^ { 2 } \right\rceil ^ { - } = ( d - 2 ) ( 2 \breve { d } - 3 \bar { ) } \leq 2 d ^ { 2 }$ , we can bound the expectation of κ by:

$$
\mathbb { E } [ \kappa ] \leq 8 d ^ { 2 } \exp ( - 2 \hat { \Delta } ) ,\tag{16}
$$

and by Markov’s inequality, we have $\hat { \kappa } \leq d ^ { c } \mathbb { E } [ \kappa ]$ with probability at least $1 - d ^ { - c }$

Now we compare population and empirical risk through z. For every $\beta \in \mathbb { R } ^ { d }$ and every string w, we have $f _ { \hat { \alpha } ^ { \prime } , \beta } ( w ) - y ( w ) = ( \beta - z ) ^ { \top } \gamma ( w ) + r ( w )$ , and the loss can be bounded from above and below by:

$$
\begin{array} { r } { \mathcal L ( \beta ) \leq ( \beta - z ) ^ { \top } H ( \beta - z ) + \mathbb E [ \kappa ] , } \end{array}\tag{17}
$$

$$
( \beta - z ) ^ { \top } \hat { H } ( \beta - z ) \leq 4 \hat { \mathcal { L } } ( \beta ) + 2 \hat { \kappa } ;\tag{18}
$$

from the inequalities $( a + b ) ^ { 2 } \leq 2 a ^ { 2 } + 2 b ^ { 2 }$ and $( a + b ) ^ { 2 } \geq { \textstyle { \frac { 1 } { 2 } } } a ^ { 2 } - b ^ { 2 }$ . In particular, $\begin{array} { r } { \hat { \mathcal { L } } ( z ) = \frac { 1 } { 2 } \hat { \kappa } } \end{array}$

The empirical risk $\hat { \mathcal { L } }$ is a convex quadratic with Hessian ${ \hat { H } } \preceq I ,$ , so it is 1-smooth. Let $\beta ( 0 ) = { \hat { \boldsymbol { \beta } } } ^ { \prime }$ and let $\beta ( t )$ denote the GD iterates with $\eta _ { 3 } = 1$ . By the descent lemma and convexity, for every $t \geq 0$

$$
\| \beta ( t + 1 ) - z \| ^ { 2 } \leq \| \beta ( t ) - z \| ^ { 2 } - 2 \left( \hat { \mathcal { L } } ( \beta ( t + 1 ) ) - \hat { \mathcal { L } } ( z ) \right) ,
$$

and $\hat { \mathcal { L } } ( \beta ( t ) )$ is non-increasing in t. Summing over $t = 0 , \ldots , T - 1 $ yields

$$
\hat { \mathcal { L } } ( \beta ( T ) ) \leq \hat { \mathcal { L } } ( z ) + \frac { \| \beta ( 0 ) - z \| ^ { 2 } } { 2 T } , \ \| \beta ( T ) - z \| ^ { 2 } \leq \| \beta ( 0 ) - z \| ^ { 2 } + 2 T \hat { \mathcal { L } } ( z ) .\tag{19}
$$

By (11) and (13), $0 \leq \hat { \beta } _ { s } ^ { \prime } < 2$ and $\begin{array} { r } { \sum _ { j \neq s } ( \hat { \beta } _ { j } ^ { \prime } ) ^ { 2 } \leq ( d - 1 ) \frac { 1 . 0 3 \eta _ { 1 } ^ { 2 } } { d ^ { 2 } } \leq 2 } \end{array}$ , hence $\| \beta ( 0 ) - z \| ^ { 2 } \leq 6$ . Together with $\begin{array} { r } { \hat { \mathcal { L } } ( z ) = \frac { 1 } { 2 } \hat { \kappa } } \end{array}$ , (19) gives

$$
\hat { \mathcal { L } } ( \beta ( T ) ) \leq \frac { \hat { \kappa } } { 2 } + \frac { 3 } { T } , \ \lVert \beta ( T ) - z \rVert ^ { 2 } \leq 6 + T \hat { \kappa } .\tag{20}
$$

We next consider the concentration on the hessian with the following lemma:

Lemma 4. Let $\begin{array} { r } { \lambda : = \frac { 1 6 ( c + 1 ) \ln d } { N _ { 3 } } } \end{array}$ and assume $\lambda \leq 1$ . Then, with probability at least $1 - d ^ { - c }$

$$
H \preceq 2 { \hat { H } } + \lambda I .
$$

Proof. Let $H _ { \lambda } : = H + \lambda I$ and define the independent positive semidefinite matrices

$$
Y _ { k } : = H _ { \lambda } ^ { - 1 / 2 } \gamma ^ { ( k ) } \left( \gamma ^ { ( k ) } \right) ^ { \top } H _ { \lambda } ^ { - 1 / 2 } + \lambda H _ { \lambda } ^ { - 1 } , \ k \in [ N _ { 3 } ] .
$$

Then $\mathbb { E } [ Y _ { k } ] = I ,$ so $\begin{array} { r } { \lambda _ { \operatorname* { m i n } } \left( \sum _ { k } \mathbb { E } [ Y _ { k } ] \right) = N _ { 3 } } \end{array}$ , and $\begin{array} { r } { \lambda _ { \operatorname* { m a x } } ( Y _ { k } ) \leq \frac { \| \gamma ^ { ( k ) } \| _ { 2 } ^ { 2 } } { \lambda } + 1 \leq \frac { 2 } { \lambda } } \end{array}$ . By the matrix Chernof inequality (Tropp, 2015), we have:

$$
\mathbb { P } \left( \lambda _ { \operatorname* { m i n } } \left( \sum _ { k = 1 } ^ { N _ { 3 } } Y _ { k } \right) \leq \frac { N _ { 3 } } { 2 } \right) \leq d \exp \left( - \frac { N _ { 3 } } { 8 } \cdot \frac { \lambda } { 2 } \right) = d ^ { - c } .
$$

On the complementary event, $H _ { \lambda } ^ { - 1 / 2 } ( \hat { H } + \lambda I ) H _ { \lambda } ^ { - 1 / 2 } \succeq \frac { 1 } { 2 } I$ , that is, $\begin{array} { r } { \hat { H } + \lambda I \succeq \frac 1 2 ( H + \lambda I ) } \end{array}$ , proving the claim.

On the events of Step 1 and Lemma 4, which hold simultaneously with probability at least $1 - 2 d ^ { - c }$ , we combine (17), Lemma 4, (18) and (20):

$$
\begin{array} { r l } & { \mathcal { L } ( \beta ( T ) ) \leq ( \beta ( T ) - z ) ^ { \top } H ( \beta ( T ) - z ) + \mathbb { E } [ \kappa ] \leq 2 ( \beta ( T ) - z ) ^ { \top } \hat { H } ( \beta ( T ) - z ) + \lambda \| \beta ( T ) - z \| ^ { 2 } + \mathbb { E } [ \kappa ] } \\ & { \qquad \leq 8 \hat { \mathcal { L } } ( \beta ( T ) ) + 4 \hat { \kappa } + \lambda \| \beta ( T ) - z \| ^ { 2 } + \mathbb { E } [ \kappa ] \leq \displaystyle \frac { 2 4 } { T } + 6 \lambda + ( 8 + \lambda T ) \hat { \kappa } + \mathbb { E } [ \kappa ] \leq \displaystyle \frac { 2 4 } { T } + 6 \lambda + ( 9 + T ) d ^ { c } \mathbb { E } [ \kappa ] . } \end{array}
$$

We bound the three terms by $\varepsilon / 3$ each. The first term, we can set the training step $T = \lceil 7 2 / \varepsilon \rceil ;$ for the second term, we set $N _ { 3 } \geq \frac { 2 8 \bar { 8 } ( c + 1 ) \ln d } { \varepsilon }$ , then $6 \lambda \leq \frac { \varepsilon } { 3 }$ . For the third term, since $\begin{array} { r } { 9 + T \le { \frac { 8 2 } { \varepsilon } } } \end{array}$ , by (16) we know $\begin{array} { r } { ( 9 + T ) d ^ { c } \mathbb { E } [ \kappa ] \leq \frac { 6 5 6 d ^ { c + 2 } } { \varepsilon } \exp ( - 2 \hat { \Delta } ) \leq \frac { \varepsilon } { 3 } \mathrm { ~ i f ~ } \hat { \Delta } \geq \ln \frac { 4 5 d ^ { ( c + 2 ) / 2 } } { \varepsilon } } \end{array}$ . By (15), this will hold if $\begin{array} { r } { \eta _ { 2 } \geq \frac { 2 0 d ^ { 2 } } { \eta _ { 1 } } \ln \frac { 4 5 d ^ { ( c + 2 ) / 2 } } { \varepsilon } } \end{array}$ which is implied by $\begin{array} { r } { \eta _ { 2 } \geq \frac { C d ^ { 2 } } { \eta _ { 1 } } } \end{array}$ ln $\frac { d } { \varepsilon }$ for a suitable constant C. Hence $\mathcal { L } _ { s } ( \hat { \alpha } ^ { \prime } , \hat { \beta } ^ { \prime \prime } ) \leq \varepsilon$ with $\hat { \boldsymbol { \beta } } ^ { \prime \prime } : = \beta ( \boldsymbol { T } )$ . A union bound over the event A and the two events above gives a total probability at least $1 - 3 d ^ { - c }$ □

## B ADDITIONAL EXPERIMENTS

## B.1 Experiment Setup

To produce the experiment results, we adopt the following initialization, losses, step sizes, and step counts:

Models. For even pairs and SLR, we use the one-layer, single-head models of Huang et al. (2025) and Marion et al. (2025), with parameters $( W \in \mathbb { R } ^ { D \times D } , u \in \mathbb { R } ^ { D } )$ and $( k \in \bar { \mathbb { R } ^ { D } } , v \in \mathbb { R } ^ { D } )$ , respectively, where D is the embedding dimension. In both cases the first parameter plays the role of our attention weights and the second that of our output weights. For ICL bigram, we use a single attention layer with one-hot token embeddings and parameters $\boldsymbol { A } , \boldsymbol { B } \in \mathbb { R } ^ { \bar { d } \times d }$ . Given the query token $q ,$ position t receives the attention logit $A _ { q , x _ { t - 1 } }$ , which depends on its preceding token $x _ { t - 1 }$ through the fixed first layer. The model outputs the logits $\textstyle \sum _ { t }$ softmax<sub>t</sub> $\left( A _ { q , x _ { t - 1 } } \right) B _ { : , x _ { t } } \in \mathbb { R } ^ { d }$ over the next token and is trained with the cross-entropy loss. Here, A and B play the roles of our attention and output weights, respectively.

Initialization. Table 2 lists the initializations. AND difers in three respects. First, we draw ${ \pmb { \alpha } } ^ { ( 1 ) } , { \pmb { \alpha } } ^ { ( 2 ) } , { \pmb { \beta } } ^ { ( 1 ) } , { \pmb { \beta } } ^ { ( 2 ) }$ with i.i.d. $\mathcal { N } ( 0 , 1 )$ entries and set ${ \pmb \alpha } ^ { ( 3 ) } = - { \pmb \alpha } ^ { ( 1 ) } , { \pmb \alpha } ^ { ( 4 ) } = - { \pmb \alpha } ^ { ( 2 ) } , \beta ^ { ( 3 ) } = { \bf \hat { \alpha } } ^ { , } { \pmb \beta } ^ { ( 1 ) } , \beta ^ { ( 4 ) } = - \beta ^ { ( 2 ) }$ , which prevents head collapse. Second, in the plans $\mathrm { A \mathrm { \to } O , O \mathrm { \to } A }$ , and $\mathrm { O \to A \to O }$ , the attention phase consists of 1250 GD steps instead of one, since a single step did not sufice for the heads to attend to diferent targets.

Table 2: Initializations. The fixed initialization is used only for $_ \mathrm { O \to A \to O ; }$ Gaussian entries are i.i.d.
<table><tr><td>Task</td><td>Fixed</td><td>Random</td></tr><tr><td>Occurrence, OR</td><td> $\alpha = \beta = \mathbf { 0 } _ { d }$   $\pmb { \alpha } ^ { ( h ) } = \pmb { \beta } ^ { ( h ) } = \mathbf { 0 } _ { d } \mathrm { ~ f o r ~ } h \in [ 4 ]$ </td><td> $\alpha , \beta \sim \mathcal { N } ( 0 , 0 . 1 ^ { 2 } )$ </td></tr><tr><td>AND</td><td></td><td> $_ \mathrm { s e e \ t e x t }$ </td></tr><tr><td>Even pairs</td><td> $W = \mathbf { 0 } _ { D \times D } , \ u = \mathbf { 0 } _ { D }$ </td><td> $W , u \sim \mathcal { N } ( 0 , 0 . 1 ^ { 2 } )$ </td></tr><tr><td> $\operatorname { I C L - b i g r a m }$ </td><td> $A = B = \mathbf { 0 } _ { d }$ </td><td> $A , B \sim \mathcal { N } ( 0 , 0 . 1 ^ { 2 } )$ </td></tr><tr><td>SLR</td><td> $k = v = \mathbf 0 _ { d }$ </td><td> $k , v \sim \mathcal N ( 0 , 0 . 1 ^ { 2 } / d )$ </td></tr></table>

Step sizes and counts. Every plan is built from the same three ingredients: a base step size η for multi-step phases and for joint GD, and the step sizes $\eta _ { \mathrm { O } }$ and $\eta _ { \mathrm { A } }$ of the single output and attention steps. Table 3 gives the schedule of each plan and Table 4 gives the values per task.

For squared-loss experiments, we minimize and report MSE without the factor $1 / 2$ used in the theoretica analysis. An experimental learning rate η therefore corresponds to 2η under the theoretical loss normalization.

Table 3: Schedule of each training plan. ${ } ^ { 6 4 } \mathrm { A } \colon \eta _ { \mathrm { A } } \times 1 { } ^ { 5 }$ means one GD step on the attention weights with step size $\eta _ { \mathrm { A } } ; \mathrm { ~ } ^ {  } \mathrm { O } ; \eta \times 2 5 0 0 ^ { \cdots }$ means 2500 GD steps on the output weights with step size η.  
Table 4: Step sizes per task. For AND, the attention phase uses multiple steps with the base step size η (see text).
<table><tr><td>Plan</td><td>Phase 1</td><td>Phase 2</td><td>Phase 3</td></tr><tr><td>A only</td><td> $\mathrm { A } \colon \eta \times 2 5 0 2$ </td><td></td><td></td></tr><tr><td> $\mathrm { O \ o n l y }$ </td><td> $\mathrm { O } \colon \dot { \eta } \times 2 5 0 2$ </td><td></td><td></td></tr><tr><td> $\mathrm { A } ( 1 \mathrm { x } ) {  } \mathrm { O }$ </td><td> $\mathrm { A \colon \eta _ { A } \times 1 }$ </td><td> $\mathrm { O } \colon \eta \times 2 5 0 0$ </td><td></td></tr><tr><td> $\operatorname { O } ( 1 \mathrm { { x } ) \to \mathbf { A } ( 1 \mathrm { { x } ) } }$ </td><td> $\mathrm { O } \colon \eta _ { \mathrm { O } } \times 1$ </td><td> $\mathrm { A \colon \eta _ { A } \times 1 }$ </td><td></td></tr><tr><td> $\mathrm { O } ( \mathrm { 1 x } ) {  } \mathrm { A } ( \mathrm { 1 x } ) {  } \mathrm { O }$ </td><td> $\mathrm { O } \colon \eta _ { \mathrm { O } } \times 1$ </td><td> $\mathrm { A \colon \eta _ { A } \times 1 }$ </td><td> $\mathrm { O } \colon \eta \times 2 5 0 0$ </td></tr><tr><td>Joint GD</td><td>A and O jointly:</td><td> $\eta \times 2 5 0 2$ </td><td></td></tr></table>

<table><tr><td>Task</td><td>η</td><td>ηo</td><td>ηA</td></tr><tr><td>Occurrence</td><td>1</td><td>50/3</td><td>500</td></tr><tr><td>OR</td><td>1</td><td>12.5</td><td>500</td></tr><tr><td>AND</td><td>0.25</td><td>8</td><td></td></tr><tr><td>Even pairs</td><td>1</td><td>100</td><td>1000</td></tr><tr><td>SLR</td><td>0.03</td><td>3</td><td>100</td></tr><tr><td>ICL bigram</td><td>50</td><td>625</td><td>25,000</td></tr></table>

For AND, every single attention step is replaced by a multi-step phase with step size η: $\mathrm { A } {  } \mathrm { O }$ uses 1250 attention steps followed by 1250 output steps, $\mathrm { O } ( \mathrm { 1 x } ) {  } \mathrm { A }$ uses one output step followed by 1250 attention steps, and $\mathrm { O } ( \mathrm { 1 x } ) {  } \mathrm { A {  } O }$ uses 1250 attention steps in the second phase and 1250 output steps in the third.

## B.2 Attention and Output Gaps

OR. With target set $T \ ( | T | = 2 )$ , the attention gap is $\begin{array} { r } { \mathbb { E } _ { + } \big [ \sum _ { t \in T } \gamma _ { t } ( w ) - \operatorname* { m a x } _ { i \notin T } \gamma _ { i } ( w ) \big ] } \end{array}$ . The output gap is $\begin{array} { r } { \big ( \frac { 1 } { | T | } \sum _ { t \in T } \beta _ { t } - \operatorname* { m a x } _ { j \notin T } \beta _ { j } \big ) / 2 . } \end{array}$

AND. For each head $h ,$ , let $t _ { h } = \arg \operatorname* { m a x } _ { t \in T } \alpha _ { t } ^ { ( h ) }$ be the target that head favours. Its attention gap is $\mathbb { E } _ { + } \big [ \gamma _ { t _ { h } } ^ { ( h ) } ( w ) -$ $\mathrm { m a x } _ { i \neq t _ { h } } \gamma _ { i } ^ { ( h ) } ( w ) ]$ . Heads are ranked by their final gap, separately in every run.

even pairs. Let $\begin{array} { r } { z { ( X ) } = \sum _ { l } \phi _ { l } ( X ) x _ { l } } \end{array}$ be the attention output, where ϕ are the attention weights under W. The attention measure is the linear separability of $\{ ( z ( X ) , y ) \}$ over all strings. It is the accuracy of the max-margin linear classifier $w ^ { * }$ through the origin, computed with a squared-hinge SVM with $C = 1 0 ^ { 3 }$ . The output gap is $\cos ( u , w ^ { * } )$ . Since $w ^ { * }$ is recomputed from the current attention, this gap can change while u is frozen.

ICL bigram. For a sequence with query q, let $t ^ { * }$ be the position after the query key, i.e., the position with $x _ { t ^ { * } - 1 } = q$ . The attention gap is the average over the training sequences of softmax $t ^ { * } \ - \ \operatorname* { m a x } _ { t \neq t ^ { * } }$ softmax<sub>t</sub>, with weights softm $\mathrm { a x } _ { t } ( A _ { q , x _ { t - 1 } } )$ . The output gap is $\cos ( \tilde { B } , \tilde { I } )$ , using the Frobenius inner product. Here $\tilde { B }$ and $\begin{array} { r } { \tilde { I } = I - \frac { 1 } { d } \mathbf { 1 } \mathbf { 1 } ^ { \top } } \end{array}$ are B and the identity with each column centered over the classes.

SLR. With J the relevant location of a sample, the attention gap is the average over the training samples of softmax $( X k ) _ { J } - \operatorname* { m a x } _ { l \neq J }$ softmax(Xk)<sub>l</sub>. The output gap is cos $( v , v ^ { * } )$ with the ground-truth direction $v ^ { * }$

Compute resources. All experiments were run on an Apple MacBook Air with an M2 chip (8 CPU cores) and 8 GB of unified memory, running macOS 14.4. No GPU was used; all training is full-batch GD on the CPU.

![](images/39263c2f672ab9eb11d84b8f667e1ddb777e240eb3bb7a9f41df584851371504.jpg)  
Figure 4: The learning progress of the AND task for each head, with average trajectory and std (shade). The heads are ranked by the gap they reach at the end, per head and per run.

## B.3 Learning progress for AND task across all heads.

Figure 4 shows the evolution of the per-head attention gap for the AND task. Under all training plans, at most two of the four heads develop a positive gap, with the gaps of Heads 3 and 4 staying near zero or becoming negative. The plans difer mostly in the second head. Under joint GD, both Head 1 and Head 2 reach a gap close to the ideal value, so the two heads attend to the two target symbols. Under the three-phase plan, Head 1 saturates closely in the same way, but the gap of Head 2 reaches only about 0.4 on average, with a large variance across runs: in many runs, the second target symbol receives little more attention than the non-target symbols. A O and attention-only training show the same pattern, with one head close to saturation and a weaker second head. This explains why the phased plans do not match joint GD on AND in Table 1: recognizing the language requires attending to both target symbols, and only joint GD reliably achieves this.

## B.4 Complete results with standard deviation.

Table 5: Train loss and test accuracy (loss for SLR) as mean over 50 seeds, with the standard deviation in parentheses. $\mathrm { V a l u e s } \le 1 0 ^ { - 3 }$ are shown in upper bounds; those that match or surpass joint GD are bold. The red entries are deterministic runs because the training and test sets are the entire instance space.
<table><tr><td rowspan="2" colspan="2">Training plan</td><td colspan="2">Occurrence</td><td colspan="2">OR</td><td colspan="2">AND</td><td colspan="2">Even-Pair</td><td colspan="2">ICL bigram</td><td colspan="2">SLR</td></tr><tr><td>Train Loss</td><td>Test Acc.</td><td>Train Loss</td><td>Test Acc.</td><td>Train Loss</td><td>Test Acc.</td><td>Train Loss</td><td>Test Acc.</td><td>Train Loss</td><td>Test Acc.</td><td>Train Loss</td><td>Test Loss</td></tr><tr><td>Coupled with rand init.</td><td>Joint GD</td><td>10-3 (10−⁶)</td><td>1.00 (0.00)</td><td>10-3 (10−⁵)</td><td>1.00 (0.00)</td><td>0.04 (0.06)</td><td>0.99 (10−3)</td><td>0.01 (10−4)</td><td>1.00 (0.00)</td><td>10-3 (10−7)</td><td>1.00 (0.00)</td><td>0.01 (10−³)</td><td>0.01 (10−3)</td></tr><tr><td rowspan="9">Decoupled 1/2/3-phase</td><td rowspan="3">Attention (A) only Output (O) only</td><td>1.64</td><td>0.50</td><td>1.67</td><td>0.50</td><td>0.24</td><td>0.94</td><td>0.69</td><td>0.51</td><td>3.90</td><td>0.02</td><td>0.50</td><td>0.50</td></tr><tr><td>(0.08)</td><td>(0.01)</td><td>(0.06)</td><td>(0.01)</td><td>(0.18)</td><td>(0.07)</td><td>(10−3)</td><td>(0.02)</td><td>(0.00)</td><td>(0.01)</td><td>(0.02)</td><td>(0.02)</td></tr><tr><td>0.74</td><td>0.86</td><td>0.68</td><td>0.85</td><td>0.79</td><td>0.75</td><td>0.63</td><td>0.50</td><td>2.70</td><td>0.16</td><td>0.48</td><td>0.48</td></tr><tr><td rowspan="3">A(1x)→0</td><td>(0.05)</td><td>(0.01)</td><td>(0.04)</td><td>(0.02)</td><td>(0.03)</td><td>(0.02)</td><td>(0.00)</td><td>(0.00)</td><td>(0.04)</td><td>(0.01)</td><td>(0.01)</td><td>(0.01)</td></tr><tr><td>0.72</td><td>0.76</td><td>0.69</td><td>0.75</td><td>0.30</td><td>0.91</td><td>0.62</td><td>0.50</td><td>2.58</td><td>0.26</td><td>0.41</td><td>0.43</td></tr><tr><td>(0.26)</td><td>(0.13)</td><td>(0.21)</td><td>(0.11)</td><td>(0.19)</td><td>(0.09)</td><td>(0.02)</td><td>(0.11)</td><td>(0.86)</td><td>(0.22)</td><td>(0.14)</td><td>(0.14)</td></tr><tr><td rowspan="3">O(1x)→A(1x)</td><td>0.47</td><td>1.00</td><td>0.64</td><td>0.78</td><td>0.52</td><td>0.87</td><td>0.52</td><td>0.50</td><td>3.20</td><td>1.00</td><td>0.28</td><td>0.28</td></tr><tr><td></td><td>(0.07) (0.00)</td><td>(0.07)</td><td>(0.19)</td><td>(0.19)</td><td>(0.07)</td><td>(10−3)</td><td>(10−3)</td><td>(0.02)</td><td>(0.00)</td><td>(0.03)</td><td>(0.03)</td></tr><tr><td>O(1x)→A(1x)→O 10-5</td><td>1.00</td><td>10-5</td><td>1.00</td><td>0.27</td><td>0.93</td><td>0.08</td><td>1.00</td><td>10-3</td><td>1.00</td><td>0.01</td><td>0.01</td></tr><tr><td rowspan="2">Sec. 4: 3-phase, zero init. O(1x)→A(1x)→O</td><td rowspan="2"></td><td>(10-6)</td><td>(0.00)</td><td>(10−5)</td><td>(0.00)</td><td>(0.13)</td><td>(0.06) 0.68</td><td>(10−³)</td><td>(0.00)</td><td>(10−8) 10−3</td><td>(0.00)</td><td>(10−3)</td><td>(10−3)</td></tr><tr><td>10-8 (10−9)</td><td>1.00 (0.00)</td><td>10-7 (10−6)</td><td>1.00 (0.00)</td><td>0.90 (0.02)</td><td>(0.02)</td><td>0.08 (0.00)</td><td>1.00 (0.00)</td><td>(10−⁸)</td><td>1.00 (0.00)</td><td>0.01 (10−3)</td><td>0.01 (10−3)</td></tr></table>

## B.5 Learning progress on occurrence and OR

Figure 5 shows the trajectories on occurrence and OR for all training plans. Both tasks behave as in Figure 2: under joint GD, the output gap grows first, overshoots while the attention saturates, and then settles at its ideal value, and the three-phase plan reaches the same final gaps. When the attention is trained first or alone, its gap moves in the wrong direction, and the output weights then grow far beyond their ideal value.

![](images/1c12aecebaf2c0f4f74764188eb04281f42ab6cea08539a0c1292720973d2b44.jpg)  
Figure 5: Attention and output gaps on occurrence and OR. Notation and phases are as in Figure 3.