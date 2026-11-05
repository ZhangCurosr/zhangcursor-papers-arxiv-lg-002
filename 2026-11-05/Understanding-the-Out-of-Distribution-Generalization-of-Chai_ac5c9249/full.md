# Understanding the Out-of-Distribution Generalization of Chain-of-Thought Reasoning in LLMs

Nuo-Jing Liang<sup>1</sup> Xiao-Tong Yuan<sup>1∗</sup> Pan Zhou<sup>2</sup>

<sup>1</sup>School of Intelligence Science and Technology, Nanjing University <sup>2</sup>Singapore Management University nuojingliang@smail.nju.edu.cn, xtyuan@nju.edu.cn, panzhou@smu.edu.sg

## Abstract

Chain-of-Thought (CoT) reasoning substantially improves Large Language Models (LLMs), yet its out-of-distribution (OOD) generalization mechanism remains theoretically underexplored. We study CoT as compositional OOD generalization: the target task lacks complete reasoning trajectories or direct input-output pairs, while training provides only in-distribution (ID) atomic step data. We show that CoT generalizes by repeatedly applying a shared single-step predictor to compose reusable ID transitions. Theoretically, let T denote the number of reasoning steps in the target task, n the number of training samples per sub-task, and k the number of output tokens per step; then the OOD task risk is controlled by the sum of the ID sub-task risks, scaled by a coefficient of order O(k) that is independent of T, although the summed risk itself may still grow with T. A Rademacher-complexity analysis further shows that the estimation term is of order $\mathcal { O } ( \sqrt { T / n } )$ , rather than linear in T, due to predictor sharing under a uniform capacity control on the shared class. For autoregressive Next-Token Prediction, the final-task risk is bounded by the best-in-class step-level risk scaled by the same coefficient of order O(k), plus an estimation term of order $\widetilde { \mathcal { O } } ( k \sqrt { T / n } )$ . A parity-coupled construction further shows that, for the sub-task empirical risk minimizer, the expected final-task risk is at least the best in-class risk plus $\Omega ( { \sqrt { T / n } } )$ when $T \leq n$ . Experiments on Atom-Task and GSM8K support the theory: CoT enables short-to-long and compositional generalization, while explicit state tracking is crucial.

Our code is available at https://github.com/maxliang114514/CoT-OOD.

## 1 Introduction

Chain-of-Thought (CoT) reasoning has become a central mechanism for improving the reasoning ability of Large Language Models (LLMs). By prompting or training models to generate intermediate steps, CoT substantially improves mathematical and logical reasoning [1–8]. Recent reasoningoriented systems further treat step generation as an explicit training signal [9–11], shifting LLM reasoning from direct pattern matching toward structured stepwise computation [12]. Despite this empirical success, a fundamental theoretical question remains open: why can CoT-trained models generalize to complex reasoning tasks that are not directly observed during training?

Existing theoretical studies provide important insights into the expressivity and computational role of CoT. Some works show that CoT can increase the effective depth of Transformers or enable serial computation [13–15], while others analyze task decomposition, in-context learning, or localized reasoning settings [16–18]. However, most existing analyses are either qualitative, restricted to narrow or two-step settings, or do not provide explicit generalization bounds for the autoregressive Next-Token Prediction (NTP) regime commonly used by modern large-scale LLMs. As a result, the out-of-distribution (OOD) generalization mechanism of multi-step CoT remains theoretically underexplored.

![](images/12f9c129c4f6701d5a0b1ab31c45a58e53bdaf8f8df3cee9a23303315d035ca0.jpg)  
Figure 1: CoT as compositional OOD generalization. Left: without CoT, the model must learn a direct mapping from input $z _ { \mathrm { 0 } }$ to final answer $\Delta z _ { T }$ , whose target distribution may be unavailable during training. With $\mathrm { C o T } ,$ the same predictor is repeatedly applied to in-distribution step distributions, mapping each state $z _ { i - 1 }$ to update $\Delta z _ { i }$ . Right: a two-hop translation example solves an unseen English-to-German query by composing learned English-to-French and French-to-German steps.

This paper studies the mechanism of CoT through compositional OOD generalization. We consider a target task requiring a complex multi-step mapping from an input state $z _ { 0 }$ to a final answer $\Delta z _ { T }$ while neither complete reasoning trajectories $( z _ { 0 } , \Delta z _ { 1 } , . . . , \Delta z _ { T - 1 } , \Delta z _ { T } )$ nor direct input-output pairs $( z _ { 0 } , \Delta z _ { T } )$ are available during training. Thus, the target task’s composition is OOD relative to the observed atomic transitions. We refer to this setting as compositional OOD; Section 3.2 separately treats mismatched training and target step distributions (Corollary 4). Instead, the model has access to in-distribution (ID) atomic step data: samples $( z _ { i - 1 } , \Delta z _ { i } )$ , where $z _ { i - 1 }$ is the reasoning state before step i and $\Delta z _ { i }$ is the next incremental output. Here, $T$ denotes the total number of generation steps: the first $T - 1$ are intermediate reasoning updates, and $\Delta z _ { T }$ is the final answer.

Under this view, CoT does not require the model to learn the entire OOD mapping at once from scratch. Instead, it repeatedly applies a shared single-step predictor. At each step, the model reads the current state $z _ { i - 1 } .$ , generates $\Delta z _ { i }$ containing k new tokens, and updates the state by appending these tokens while keeping a fixed context length m. CoT therefore converts a complex target mapping into a composition of reusable ID atomic transitions (Fig. 1). This motivates our central question:

When a model learns only atomic step transitions, can repeated CoT composition generalize to unseen multi-step OOD tasks, and how does the resulting error scale with the reasoning depth $T ?$

Contributions We answer this question by developing a generalization framework for CoT-based task decomposition. First, we prove that the risk of the complex OOD task can be upper-bounded by the sum of risks over ID sub-tasks, scaled by a coefficient of order ${ \mathcal { O } } ( k )$ that is independent of T. The coefficient results from deterministic argmax decoding together with the bounded sequence loss and the local error floor $( \log 2 ) / k$ that a hard-decoding (argmax) error induces. CoT thus avoids fitting the unseen composite mapping directly and composes local transitions at test time.

Second, we derive a Rademacher-complexity bound for the multi-step setting. A naive analysis that treats each reasoning step independently would yield a linear dependence on $T .$ In contrast, because CoT applies the same predictor across all steps, we can analyze the combined step-level sample set and obtain a sublinear generalization-gap term scaling as $\mathcal { O } ( \sqrt { T / n } )$ , where n is the number of training samples per atomic step distribution.

Third, we specialize the analysis to the NTP setting of autoregressive LLMs. We directly analyze the final k target tokens of each atomic sequence and take the $N = n T$ complete sequences as the sampling units. The resulting final-task risk bound keeps the best in-class step-level risk explicit and adds an estimation term of order $\widetilde { \mathcal { O } } ( k \sqrt { T / n } )$ . Appendix B.10 shows that the $\sqrt { T / n }$ dependence cannot be improved for the sub-task empirical risk minimizer: a construction with $k = 1$ , in which the per-step errors are coupled through the XOR accumulated over the chain, makes this minimizer incur an excess of $\Omega ( { \sqrt { T / n } } )$ when $T \leq n ,$ , and carries that excess through to the final task.

We further validate our theory through controlled experiments. We introduce Atom-Task, a symbolic compositional benchmark involving multi-step sequence operations, to isolate the effect of CoT on OOD compositional generalization. Experiments show that CoT-trained models generalize from shorter or constrained training chains to longer unseen compositions, whereas models trained without intermediate reasoning fail to capture the underlying compositional structure. Ablation results further show that explicitly tracking the current reasoning state is crucial, matching our theoretical requirement that each state $z _ { i - 1 }$ contains sufficient information for the next atomic transition. Additional experiments on GSM8K, a benchmark of grade-school math word problems, support the same conclusion in real-world mathematical reasoning.

## 2 Preliminaries

Problem Setting We consider an autoregressive predictor with a context window size of $m .$ . Let V be a vocabulary (a set of tokens), with $n _ { v } = | \boldsymbol { \nu } |$ denoting its size, and let $\mathcal { V } ^ { m }$ denote the set of all token sequences of length m formed by tokens from $\nu .$ . The predictor $\psi : \mathcal { V } ^ { m } \to \mathbb { R } ^ { n _ { v } \times k }$ $( m \gg k )$ maps an input sequence to probability distributions over k output positions, and belongs to a hypothesis class Ψ. Losses are computed on these soft predictions; for autoregressive generation, discrete tokens are obtained via argmax, denoted $\mathrm { h a r d } ( \psi ( \cdot ) ) \in \mathcal { V } ^ { k }$

Under the CoT paradigm, the model does not predict the final answer directly via $\psi ( z _ { 0 } )$ , but generates a chain of $T$ reasoning steps. We write $\widehat { \cdot }$ for model-generated quantities and reserve the unhatted symbol for the ground truth. At step $i ,$ the incremental output is $\widehat { \Delta z } _ { i } = \mathrm { h a r d } \big ( \psi ( \widehat { z } _ { i - 1 } ) \big )$ ), and the state updates by appending these k tokens while discarding the first k (sliding window):

${ \widehat { z } } _ { i } = { \widehat { z } } _ { i - 1 } [ k + 1 : m ] \oplus { \widehat { \Delta z } } _ { i }$ (⊕ denotes sequence concatenation), $\widehat { \Delta z } _ { T } = \mathrm { h a r d } \big ( \psi ( \widehat { z } _ { T - 1 } ) \big )$

thus maintaining the sequence length m for all intermediate states, with $\widehat { z } _ { 0 } = z _ { 0 } ;$ the true state follows the same recursion with $\Delta z _ { i }$ in place of $\widehat { \Delta z _ { i } } , \mathrm { i . e . , } z _ { i } = z _ { i - 1 } [ k + 1 : m ] \oplus \Delta z _ { i }$

For a given complex task, any instance $( z _ { 0 } , \Delta z _ { T } )$ with $z _ { 0 } \in \mathcal { V } ^ { m }$ and $\Delta z _ { T } \in \mathcal { V } ^ { k }$ satisfies $( z _ { 0 } , \Delta z _ { T } ) \sim$ D. There exists a ground-truth mapping $\psi _ { * }$ (not necessarily within Ψ) such that hard $\left( \psi _ { * } ( z _ { i - 1 } ) \right) =$ $\Delta z _ { i }$ for all $i ,$ implying ψ<sub>∗</sub> generates the CoT sequence $( \Delta z _ { 1 } , \dots , \Delta z _ { T } )$ from $z _ { 0 }$ to yield the target output $\Delta z _ { T }$

Define the state transition $\bar { \psi } : \mathcal { V } ^ { m } \to \mathcal { V } ^ { m }$ as ${ \bar { \psi } } ( z ) : = z [ k + 1 : m ] \oplus$ hard(ψ(z)), representing the input state for the next step (note $\psi$ uses hard predictions for transitions). To distinguish from $\psi _ { : }$ , let $F _ { \psi } : \mathcal { V } ^ { m }  \mathbb { R } ^ { n _ { v } \times k }$ denote the composite CoT mapping $F _ { \psi } ( z _ { 0 } ) : = \psi ( \bar { \psi } ^ { ( T - 1 ) } ( \bar { z _ { 0 } } ) )$ , which encapsulates the T-step reasoning process (note $F _ { \psi }$ applies $\psi$ to the final state, outputting soft predictions for loss computation). Our learning goal is to minimize the population risk:

$$
R _ { \mathcal { D } } ( F _ { \psi } ) = \int \mathfrak { L } ( F _ { \psi } ( z _ { 0 } ) , \Delta z _ { T } ) d P ( z _ { 0 } , \Delta z _ { T } ) = \int \mathfrak { L } ( \psi ( \bar { \psi } ^ { ( T - 1 ) } ( z _ { 0 } ) ) , \Delta z _ { T } ) d P ( z _ { 0 } , \Delta z _ { T } ) ,
$$

where P is the probability measure associated with D and $\mathfrak { L } : \mathbb { R } ^ { n _ { v } \times k } \times \mathbb { R } ^ { n _ { v } \times k } \to \mathbb { R }$ is the sequencelevel loss function that compares the model’s soft predictions (in ${ \mathbb R } ^ { n _ { v } \times k } )$ with the ground-truth output $\Delta z _ { T }$ (converted from discrete tokens to one-hot vectors in $\mathbb R ^ { \dot { n } _ { v } \times k } )$ ; the formal definition is provided in Section 3. This aims to identify a $\psi \in \Psi$ approximating $\psi _ { * }$ , thereby generating the CoT sequence from $z _ { \mathrm { 0 } }$ to yield $\Delta z _ { T }$ .

Notations $\bar { \psi } ^ { ( T ) }$ denotes the T-fold composition $\bar { \psi } \circ \cdots \circ \bar { \psi } ( T$ times in total). The final prediction is extracted via $\psi \big ( \bar { \psi } ^ { ( T - 1 ) } \big ( z _ { 0 } \big ) \big )$ . For a sequence $z , z [ a : b ]$ denotes the subsequence from index a to b (inclusive), and $( z ) _ { j }$ denotes its $j \cdot$ -th component (or token). The operator ⊕ denotes the concatenation of two sequences. Other key notations are summarized in Table 4 in Appendix B.

## 3 Theoretical Analysis: Task Decomposition for OOD Tasks

We analyze the generalization error of CoT reasoning by decomposing the risk of the complex OOD task into per-step risks (Section 3.1) and then bounding the generalization gap (Section 3.2).

Assumption 1 (Bounded and Lipschitz Loss Function). For a single token represented as a probability vector in $\mathbb { R } ^ { n _ { v } }$ , the token-level loss function $\ell ( \cdot , \cdot )$ is defined as the $\epsilon _ { \ell } .$ -clipped cross-entropy loss with a fixed clipping threshold $\epsilon _ { \ell } \in ( 0 , 1 / 4 )$ , i.e., $\ell ( p , y ) : = -$ log max $\{ p _ { y } , ~ \epsilon _ { \ell } \}$ , where p denotes the predicted probability vector and y the ground-truth token. It is upper-bounded by $B _ { \ell }$ , i.e., $\ell ( p , y ) \leq B _ { \ell }$ , and is G -Lipschitz continuous with respect to the $\ell _ { 2 }$ -norm of the predicted probability vector, $\mathrm { i . e . , } | \ell ( \hat { y } _ { 1 } , y ) - \ell ( \hat { y _ { 2 } } , y ) | \leq G _ { \ell } \| \hat { y } _ { 1 } - \hat { y } _ { 2 } \| _ { 2 }$ for all probability vectors $\hat { y } _ { 1 } , \hat { y } _ { 2 } \in \Delta _ { n _ { v } - 1 } : = \{ p \in$ $[ 0 , 1 ] ^ { n _ { v } } : \textstyle \sum _ { a = 1 } ^ { n _ { v } } p _ { a } = 1 \}$ and ground-truth token y. Without loss of generality, assume $G _ { \ell } \geq 1$

Remark 1. Assumption 1 is standard in learning theory [19–22]. Both requirements follow directly from clipping, which lower-bounds the probability assigned to the true token by $\epsilon _ { \ell } ;$ both conditions are therefore satisfied by construction.

Definition 1 (Sequence Loss). For a predicted probability distribution $\hat { y } \in \mathbb R ^ { n _ { v } \times k }$ and a ground-truth token sequence $\bar { \Delta } z$ (represented as one-hot vectors in $\dot { \mathbb R } ^ { n _ { v } \times k } )$ , the sequence loss is defined as the average of the token-wise loss ℓ over these k generated tokens, where $\hat { y } _ { j }$ denotes the probability vector at output position j (the j-th column of yˆ) [23]:

$$
\mathfrak { L } ( \hat { y } , \Delta z ) : = \frac { 1 } { k } \sum _ { j = 1 } ^ { k } \ell ( \hat { y } _ { j } , ( \Delta z ) _ { j } )
$$

## 3.1 Generalization Error Decomposition via CoT

For $\begin{array} { r l r } { ( z _ { 0 } , \Delta z _ { T } ) } & { { } \sim } & { \mathcal { D } } \end{array}$ , let $\mathcal { D } ^ { \prime }$ denote the joint distribution over the reasoning sequence $( z _ { 0 } , \Delta z _ { 1 } , . . . , \Delta z _ { T } )$ , and let $\mathcal { D } _ { i }$ denote the marginal law of $( z _ { i - 1 } , \Delta z _ { i } )$ under $\mathcal { D } ^ { \prime }$ (a state on the ground-truth trajectory is called an oracle state). Lemma 7 in Appendix B.2 shows that the step-level loss of an erroneous hard-decoded block is at least $( \log 2 ) / \bar { k } ,$ , a floor that is independent of the reasoning depth T.

Theorem 1 (Generalization Error Decomposition). Under Assumption 1, we have:

$$
R _ { \mathcal { D } } ( F _ { \psi } ) \leq \frac { B _ { \ell } k } { \log 2 } \sum _ { i = 1 } ^ { T } \int \mathfrak { L } ( \psi ( z _ { i - 1 } ) , \Delta z _ { i } ) d \mathcal { D } _ { i } ( z _ { i - 1 } , \Delta z _ { i } ) ,
$$

where $( \log 2 ) / k$ is the local errorfloor ofLemma 7.

Remark 2 (Implications of Theorem 1). Theorem 1 reveals the fundamental principle underlying CoT’s OOD generalization: the generalization risk of a complex OOD task is dominated by the sum ofper-step in-distribution risks, scaled by an ${ \mathcal { O } } ( k )$ coefficient that is independent of $T ,$ . This is qualitatively different from direct OOD learning, where one would attempt to bound the risk of the composite mapping $F _ { \psi }$ directly—without decomposition, not only is the mapping $F _ { \psi }$ inherently more complex, but more critically, we lack the direct samples $( z _ { 0 } , \Delta z _ { T } ) \sim \mathcal { D }$ , so the composite risk cannot be estimated from target data alone.

Theorem 1 bounds the target risk by the summed step-level risks, and the additive dependence on those risks is unavoidable in general. A step-level error can propagate along the chain through the state update rather than being averaged out, so the per-step risks accumulate in the final answer.

Remark 3 (Tightness of Theorem 1). Appendix B.9 constructs, for $k = 1$ , a binary sliding-window task with a shared single-step predictor $\psi$ and a bounded sequence loss, for which

$$
R _ { \mathcal { D } } ( F _ { \psi } ) = \Omega \left( \sum _ { i = 1 } ^ { T } R _ { \mathcal { D } _ { i } } ( \psi ) \right) .
$$

Consequently, the additive form of Theorem 1 is tight up to its depth-independent coefficient: when every step-level error reaches the final answer, the linear growth in $T$ of the summed risk is attained, and no argument that charges only per-step risks can remove it.

## 3.2 Bounds via Rademacher Complexity

Let $S _ { i } = \{ ( z _ { i - 1 } ^ { ( j ) } , \Delta z _ { i } ^ { ( j ) } ) \} _ { j = 1 } ^ { n }$ be n independent atomic pairs drawn from $\mathcal { D } _ { i } .$ , with $\textstyle S _ { 0 } = \bigcup _ { i = 1 } ^ { T } S _ { i }$ All nT atomic examples are mutually independent, with the n examples in $S _ { i }$ drawn from $\mathcal { D } _ { i } ;$ each comes from some source realizing the same step-level skill distribution and need not lie on a common trajectory. Define $f _ { \psi } ( z , \Delta z ) : = \bar { \mathfrak { L } } ( \psi ( z ) , \Delta z )$ and $\mathcal { F } : = \{ f _ { \psi } : \psi \in \Psi \}$ . To bound the generalization gap we use Rademacher complexity, writing $R _ { D _ { i } } ( \psi )$ and $\widehat { R } _ { S _ { i } } ( \psi )$ for the associated population and empirical risks (Appendix B.4). Applying Rademacher bounds to the $T$ steps separately makes the resulting gap grow linearly in $\stackrel { \bullet } { T }$ . Since the same learner $\psi$ processes samples from all $T$ distributions, analyzing the combined set $S _ { 0 }$ under a uniform capacity control makes both the deviation term and the complexity term (from $\textstyle \sum _ { i = 1 } ^ { T } { \widehat { \mathfrak { R } } } _ { S _ { i } } ( { \mathcal { F } } )$ to $T \mathfrak { R } _ { S _ { 0 } } ( \mathcal { F } ) )$ ) improve from $\mathcal { O } ( T )$ to $\mathcal { O } ( \sqrt { T } )$ —analogous to multi-task learning, where sharing a hypothesis across related tasks improves generalization [24]. Extending the classical symmetrization lemma [19] to multiple distributions (Lemma 8 in Appendix B.4) gives:

Theorem 2 (Rademacher Complexity Bound). For any $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta ,$ uniformly over $\psi \in \Psi$ , we have:

$$
\left| \sum _ { i = 1 } ^ { T } ( R _ { \mathcal { D } _ { i } } ( \psi ) - \widehat { R } _ { S _ { i } } ( \psi ) ) \right| \leq 2 T \widehat { \mathfrak { N } } _ { S _ { 0 } } ( \mathcal { F } ) + 3 B _ { \ell } \sqrt { \frac { 2 T \log ( 4 / \delta ) } { n } } ,
$$

where $\widehat { \mathfrak { R } } _ { S _ { 0 } } ( \cdot )$ denotes the empirical Rademacher complexity on $S _ { 0 } , \mathcal { F } = \mathfrak { L } \circ \Psi$ is the induced bounded loss class, and $B _ { \ell }$ is the bound ofthe sequence loss L according to Assumption 1.

Remark 4. Sharing a single learner ψ across all T reasoning steps, under a uniform capacity control on the shared class, is what permits treating the $T$ step-level datasets as one combined sample $S _ { 0 }$ of size $n T \colon$ the same predictor is fitted to all steps, so the estimation dependence improves from $T$ to $\sqrt { T }$

The following corollary records the resulting guarantee, first for a fixed predictor and then, after the ERM comparison, for the empirical risk minimizer; both proofs are in Appendix B.5.

Corollary 3 (Generalization Bound for Complex OOD Tasks). Under Assumption 1, with probability at least $1 - \delta ,$ , for every $\psi \in \Psi$

$$
R _ { \mathcal { D } } ( F _ { \psi } ) \leq \frac { B _ { \ell } k } { \log 2 } \biggl [ \sum _ { i = 1 } ^ { T } \widehat { R } _ { S _ { i } } ( \psi ) + 2 T \widehat { \mathfrak { N } } _ { S _ { 0 } } ( \mathcal { F } ) + 3 B _ { \ell } \sqrt { \frac { 2 T \log ( 4 / \delta ) } { n } } \biggr ] ,
$$

andfor the sub-task ERM $\boldsymbol { { \widehat { \psi } } } \in$ arg min<sub>ψ∈Ψ</sub> $\textstyle \sum _ { i = 1 } ^ { T } { \widehat { R } } _ { S _ { i } } ( \psi )$ , with $\begin{array} { r } { \psi ^ { \star } \in \arg \operatorname* { m i n } _ { \psi \in \Psi } \sum _ { i = 1 } ^ { T } R _ { { D } _ { i } } ( \psi ) } \end{array}$

$$
R _ { \mathcal { D } } ( F _ { \widehat { \psi } } ) \leq \frac { B _ { \ell } k } { \log 2 } \Big [ \sum _ { i = 1 } ^ { T } R _ { \mathcal { D } _ { i } } ( \psi ^ { \star } ) + 4 T \widehat { \mathfrak { R } } _ { S _ { 0 } } ( \mathcal { F } ) + 6 B _ { \ell } \sqrt { \frac { 2 T \log ( 4 / \delta ) } { n } } \Big ] .
$$

Remark 5 (Tightness of Corollary 3). The estimation term of Theorem 2 is order-tight, and its rate is inherited by Corollary 3: Appendix B.10 constructs a binary compositional task and a two-element class for which the sub-task ERM $\widehat { \psi }$ of Corollary 3 satisfies

$$
\mathbb { E } _ { S } \left[ \sum _ { i = 1 } ^ { T } R _ { \mathcal { D } _ { i } } ( \widehat { \psi } ) - \operatorname* { m i n } _ { \psi \in \Psi } \Big \{ \sum _ { i = 1 } ^ { T } R _ { \mathcal { D } _ { i } } ( \psi ) \Big \} \right] = \Omega \left( \operatorname* { m i n } \left\{ 1 , \sqrt { \frac { T } { n } } \right\} \right) ,
$$

i.e., its atomic excess, the gap between its summed step-level risk and the best in-class value of that sum, is of that order; and in that instance the atomic excess coincides with the excess on the final task, so the same bound holds for $\mathbb { E } _ { S } [ R _ { \mathcal { D } } ( F _ { \widehat { \psi } } ) - \operatorname* { m i n } _ { \psi \in \Psi } R _ { \mathcal { D } } ( F _ { \psi } ) ]$ . In particular it is $\Omega ( { \sqrt { T / n } } )$ when $T \leq n$ . Two features of the construction are responsible: the per-step separation is amplified by a factor of $T$ at the final token, and the best-in-class per-step risk is $\Theta ( 1 )$ , which makes the excess the identifiable quantity.

The preceding analysis addresses compositional OOD generalization: the target task is out of distribution, whereas the step-level distributions $\mathcal { D } _ { i }$ are not. When the step-level data are themselves drawn from shifted distributions $Q _ { i } ,$ , the decomposition carries over with one additional per-step term.

Corollary 4 (Atomic Distribution Shift). Let $Q _ { i }$ be the training distribution at step i, let $\mathcal { D } _ { i }$ be the corresponding target oracle-state marginal, and let $S _ { i } \sim Q _ { i } ^ { n }$ . Define

$$
\operatorname { d i s c } _ { \mathcal { F } } ( \mathcal { D } _ { i } , Q _ { i } ) : = \operatorname* { s u p } _ { \psi \in \Psi } \left| R _ { \mathcal { D } _ { i } } ( \psi ) - R _ { Q _ { i } } ( \psi ) \right| .
$$

Then, with probability at least $1 - \delta ,$ , for every $\psi \in \Psi$

$$
R _ { \mathcal { D } } ( F _ { \psi } ) \leq \frac { B _ { \ell } k } { \log 2 } \left[ \sum _ { i = 1 } ^ { T } \widehat { R } _ { S _ { i } } ( \psi ) + 2 T \widehat { \mathfrak { N } } _ { S _ { 0 } } ( \mathcal { F } ) + 3 B _ { \ell } \sqrt { \frac { 2 T \log ( 4 / \delta ) } { n } } + \sum _ { i = 1 } ^ { T } \mathrm { d i s c } _ { \mathcal { F } } ( \mathcal { D } _ { i } , Q _ { i } ) \right] ,
$$

andfor the sub-task ERM $\boldsymbol { { \widehat { \psi } } } \in$ arg min $\textstyle \psi \in \Psi \sum _ { i = 1 } ^ { T } \widehat { R } _ { S _ { i } } ( \psi )$ , with $\psi ^ { \star } \in$ arg min<sub>ψ∈Ψ</sub> $\textstyle \sum _ { i = 1 } ^ { T } R _ { D _ { i } } ( \psi )$

$$
R _ { \mathcal { D } } ( F _ { \widehat { \psi } } ) \leq \frac { B _ { \ell } k } { \log 2 } \left[ \sum _ { i = 1 } ^ { T } R _ { \mathcal { D } _ { i } } ( \psi ^ { \star } ) + 4 T \widehat { \mathfrak { N } } _ { S _ { 0 } } ( \mathcal { F } ) + 6 B _ { \ell } \sqrt { \frac { 2 T \log ( 4 / \delta ) } { n } } + 2 \sum _ { i = 1 } ^ { T } \mathrm { d i s c } _ { \mathcal { F } } ( \mathcal { D } _ { i } , Q _ { i } ) \right] .
$$

The proof is given in Appendix B.6. The corollary thus controls shifts between training and target step distributions; the drift of the model’s own generated trajectory (its rollout) is handled only through the hard-decoding reduction and is not itself a distributional guarantee.

## 4 Specific Analysis: Results in the Next-Token Prediction Setting

In the preceding analysis the OOD generalization error of complex compositional tasks is upperbounded by a sum of per-step in-distribution risks, with the gap governed by the Rademacher complexity of the shared hypothesis class. We now specialize this decomposition to the Next-Token Prediction (NTP) pretraining paradigm [23]—the standard training regime for LLMs—and derive a target-token bound with explicit depth, sample-size, and decoder-capacity dependence. Relative to the general analysis of Section 3.2 the additional ingredients are (i) a reduction from the final-task risk of free-running generation, in which the model conditions on its own outputs, to target-token risks under teacher forcing, in which every prefix is taken from the ground truth (Lemma 10, Appendix B.7) and (ii) an explicit evaluation of the complexity term for the decoder (Appendix A.2); the ERM comparison is the one already used in Corollary 3.

## 4.1 Problem Formulation

The model is a composition $\operatorname { L M } = g \circ h$ (representation learner h and token predictor $g )$ . Here, the abstract single-step mapping ψ is instantiated by the block predictor $\psi _ { g , h }$ induced by k successive next-token predictions via $g \circ h$ , and $F _ { \psi _ { g , h } }$ denotes the resulting T-step final predictor. For step $i = 1 , \dots , T$ , let ${ \cal { S } } _ { i } = \{ X _ { i j } \} _ { i = 1 } ^ { n }$ , where each $X _ { i j } = ( t _ { i j , 1 } , \dots , t _ { i j , m } )$ is a complete sequence. We denote the teacher-forced context from $X _ { i j }$ before token t by $X _ { i j , < t }$ and set $N \overset { \cdot } { = } n T$ . Here and in Appendix $\mathbf { A } ,$ , m denotes the complete-record length: an $( m - k )$ -token state followed by k target tokens. Thus the general setup of Section 2 is instantiated with state length $m - k$ . Before each block, both oracle and generated histories retain their most recent $m - k$ tokens; within the block, the corresponding true or generated prefixes are appended autoregressively. The oracle record is therefore ${ \bf \dot { X } } = z _ { i - 1 } \oplus \Delta z _ { i } .$ , and $\mathcal { D } _ { i }$ also denotes its induced distribution. This length convention leaves the general risk bounds unchanged.

For a complete sequence $X = ( t _ { 1 } , \ldots , t _ { m } )$ and its prefix $X _ { < t } = ( t _ { 1 } , \dots , t _ { t - 1 } )$ , define the targettoken loss, with g applied to the hidden state at the last position of its argument:

$$
q _ { \psi _ { g , h } } ( X ) : = \frac { 1 } { k } \sum _ { t = m - k + 1 } ^ { m } \ell ( g ( h ( X _ { < t } ) ) , t _ { t } ) ,\tag{1}
$$

and let $\mathcal { Q } _ { k } = \{ q _ { \psi _ { q , h } } : g \in \mathcal { G } , h \in \mathcal { H } \}$ , where G, H are the classes of the token predictor and of the representation learner. For ${ \cal { S } } _ { i } = \{ X _ { i j } \} _ { j = 1 } ^ { n }$ , define

$$
R _ { \mathcal { D } _ { i } } ^ { k } ( \psi _ { g , h } ) : = \mathbb { E } _ { X \sim \mathcal { D } _ { i } } [ q _ { \psi _ { g , h } } ( X ) ] , \qquad \widehat { R } _ { S _ { i } } ^ { k } ( \psi _ { g , h } ) : = \frac 1 n \sum _ { j = 1 } ^ { n } q _ { \psi _ { g , h } } ( X _ { i j } ) .
$$

The $N = n T$ complete sequences are the independent sampling units; tokens and prefixes within each sequence may be dependent. We abbreviate $R _ { i } ^ { k } : = R _ { D _ { i } } ^ { k }$ and $\widehat { R } _ { i } ^ { k } : = \widehat { R } _ { S _ { i } } ^ { k }$

Remark 6 (Teacher-Forced NTP and Free-Running Generation). The target-token loss in Eq. (1) is teacher-forced: it uses the true within-step prefixes, whereas $F _ { \psi _ { g , h } }$ runs freely and conditions on its own generated tokens. These block losses need not coincide. Lemma 10 in Appendix B.7 extends the first-error argument of Theorem 1 to individual tokens and gives $\begin{array} { r } { R _ { \mathcal { D } } ( F _ { \psi _ { g , h } } ) \le \frac { B _ { \ell } k } { \log 2 } \sum _ { i = 1 } ^ { T } R _ { \mathcal { D } _ { i } } ^ { k } ( \psi _ { g , h } ) } \end{array}$ with the same coefficient $B _ { \ell } k / \log 2 = \mathcal { O } ( k )$

In this setting we write $( \widehat { g } , \widehat { h } )$ for the minimizer of $\textstyle \sum _ { i } { \widehat { R } } _ { S _ { i } } ^ { k } ( \psi _ { g , h } )$ and $( g ^ { \star } , h ^ { \star } )$ for the minimizer of $\textstyle \sum _ { i } R _ { D _ { i } } ^ { k } ( \psi _ { g , h } )$ , with $\widehat { \psi } : = \psi _ { \widehat { g } , \widehat { h } }$ and $\psi ^ { \star } : = \psi _ { g ^ { \star } , h ^ { \star } } ;$ ; the resulting target-token excess is the NTP analogue of the atomic excess of Section 3.2.

## 4.2 Main Theoretical Results

Let $\widehat { \mathfrak { R } } _ { S _ { 0 } } ( \mathcal { Q } _ { k } )$ denote the empirical Rademacher complexity of the target-token class on the combined sample $S _ { 0 }$ of $N = n T$ records, i.e., the quantity of Appendix B.1 with $\mathcal { F }$ replaced by $\mathcal { Q } _ { k }$ and one

independent sign per record. Applying Theorem 2 to the target-token class yields the following deviation bound, uniform over the class.

Theorem 5 (Target-Token Risk Bound). Under Assumption 1, with probability at least $1 - \delta ,$ uniformly over $g \in { \mathcal { G } }$ and $h \in \mathcal H$

$$
\left| \sum _ { i = 1 } ^ { T } \left( R _ { \mathcal { D } _ { i } } ^ { k } ( \psi _ { g , h } ) - \widehat { R } _ { S _ { i } } ^ { k } ( \psi _ { g , h } ) \right) \right| \leq 2 T \widehat { \mathfrak { N } } _ { S _ { 0 } } ( \mathcal { Q } _ { k } ) + 3 B _ { \ell } \sqrt { \frac { 2 T \log ( 4 / \delta ) } { n } } .\tag{2}
$$

The next corollary is the target-token counterpart of the ERM form of Corollary 3: for the empirical risk minimizer the empirical risks in the bound are replaced by the best-in-class population risks, at the price of paying the deviation twice.

Corollary 6 (ERM Final-Task Risk). Let $( \widehat { g } , \widehat { h } )$ and $( g ^ { \star } , h ^ { \star } )$ be defined as above. With probability at least $1 - \delta$

$$
R _ { \mathcal { D } } ( F _ { \psi _ { \hat { g } , \hat { h } } } ) \leq \frac { B _ { \ell } k } { \log 2 } \left[ \sum _ { i = 1 } ^ { T } R _ { \mathcal { D } _ { i } } ^ { k } ( \psi _ { g ^ { \star } , h ^ { \star } } ) + 4 T \widehat { \mathfrak { N } } _ { S _ { 0 } } ( Q _ { k } ) + 6 B _ { \ell } \sqrt { \frac { 2 T \log ( 4 / \delta ) } { n } } \right] .\tag{3}
$$

Under Assumption 2, the target-token complexity obeys the following bound. Deriving it $( \mathsf { A p - }$ pendix $\mathbf { A } . 2 )$ requires two ingredients beyond the general analysis of Section 3.2: the layerwise matrix-covering argument of [23], which we instantiate for our decoder block and then restrict to the final k output rows, and a sequence-level loss-covering step of our own that charges one Rademacher sign per complete sequence rather than per generated token; the latter is what produces the $1 / \sqrt { N }$ normalization with ${ \dot { N } } = n T$

$$
\mathop { \widehat { \mathfrak { R } } } _ { S _ { 0 } } ( \mathcal { Q } _ { k } ) \lesssim G _ { \ell } \sqrt { \frac { \Theta d H \tau _ { m , N } } { N } } + G _ { \ell } \sqrt { \frac { d n _ { v } } { N } } , \qquad N = n T ,\tag{4}
$$

where Θ counts trainable parameters, $L , H$ , d are the layer count, head count and model dimension, and $\tau _ { m , N } = \log ( 1 + \rho _ { L } s _ { L } ( m , N ) )$ is the decoder-capacity factor (Appendix A.2); below we write τ for $\tau _ { m , N }$

The resulting final-task bound is, for constants $C _ { 1 } , C _ { 2 } > 0$ depending only on the fixed norm and normalization bounds of Assumption 2,

$$
R _ { \mathcal { D } } ( F _ { \psi _ { \bar { g } , \bar { h } } } ) \leq \frac { B _ { \ell } k } { \log 2 } \big [ \sum _ { i = 1 } ^ { T } R _ { i } ^ { k } ( \psi ^ { \star } ) + C _ { 1 } G _ { \ell } \sqrt { \frac { T \Theta d H \tau } { n } } + C _ { 2 } G _ { \ell } \sqrt { \frac { T d n _ { v } } { n } } + 6 B _ { \ell } \sqrt { \frac { 2 T \log ( 4 / \delta ) } { n } } \big ] .\tag{5}
$$

Since $B _ { \ell } k / \log 2 = \mathcal { O } ( k )$ , for fixed model dimensions, norm bounds and clipping threshold the sample-dependent atomic excess contributes $\widetilde { \mathcal { O } } ( k \sqrt { T / n } )$ to the bound, where $\widetilde { \mathcal { O } }$ suppresses the logarithmic factors of τ and of the confidence term.

## 4.3 Discussion of the Final Bound

Depth and sample size. The target-token analysis treats the $N = n T$ complete sequences as the independent observations while allowing arbitrary dependence among tokens within each sequence. Pooling these observations through one shared predictor under a uniform capacity control on the class yields an atomic excess of order ${ \sqrt { T / n } } = { \sqrt { T } } \cdot n ^ { - 1 / 2 } { \overline { { - } } } \mathbf { a } ~ { \sqrt { T } }$ accumulation across the $T$ composed steps, sublinear in place of the naive linear $T ,$ , times the $n ^ { - 1 / 2 }$ statistical cost of a single atomic step—rather than the $\mathbf { \bar { \cal T } } / \sqrt { n }$ dependence obtained by analyzing the $T$ steps separately. We state this with the per-step sample size n held fixed, so that the depth dependence is ${ \sqrt { T } } ;$ if the total budget $N = n T$ is held fixed instead, then n shrinks and the per-step sampling factor alone grows like $\sqrt { T / N }$ , giving $T / \sqrt { N }$ —linear behavior that comes from the shrinking per-step budget, not from any loss of the pooling gain. Equation (3) retains the best-in-class step-level risk in the complete final-task risk bound; after the depth-independent factor $B _ { \ell } k / \log 2 = \dot { \mathcal { O } } ( k )$ , the sample-dependent atomic excess contributes $\widetilde { \mathcal { O } } ( k \sqrt { T / n } )$ to the bound.

Context length and decoder capacity. We analyze the final-k-token loss directly, which permits dependence among tokens within a sequence; concentration is taken over the $N = { \overline { { n } } } T$ independent complete sequences. In the parameterized bound of Eq. (5), context length enters through τ evaluated at ${ \bar { N } } = n T$ and also through Θ if the trainable positional-embedding parameter count grows with m.

Theoretical context Unlike prior work that restricts compositional analysis to 2-step cases without explicit parameter-dependent bounds [25], focuses on sub-task identification sample complexity rather than OOD error decomposition [26], proves that CoT reduces learning difficulty without generalization bounds [16], or addresses only single-distribution pretraining [23], our framework unites CoT error decomposition with a shared-hypothesis structure and directly controls targettoken risk with reasoning-depth and decoder-capacity dependence. Maurer et al. [24] provide a shared-representation perspective for pooled estimation, whereas Li et al. [23] provide decoder covering-number tools for full-sequence NTP. We use the latter only to control the restricted final-k output class; the target-token risk bound itself is derived directly at the sequence level.

## 5 Experiments

We evaluate compositional generalization on Atom-Task, a task of multi-step operations on sequences; GSM8K experiments, extended setup details, and noise-robustness results are in Appendix C (GSM8K results in Section C.2).

Experimental Setup We introduce Atom-Task to isolate compositional reasoning, requiring sequential execution of atomic operations on symbolic sequences over a six-symbol alphabet $( \{ \alpha , \beta , \gamma , \delta , \varepsilon , \omega \} )$ ). We define five operations: Reverse, Substitute, Duplicate, Deduplicate, and Rotate.

Models and Datasets For the Atom-Task, we use Qwen3-4B. We design four training datasets: (1) Single-Step: 5k uniformly distributed one-step samples; (2) Three-Step: 1.66k three-step samples; (3) Mixed 3+5: 1,333 samples (833 three-step and 500 five-step; the latter use ≤ 2 distinct operations, bridging length and combinatorial generalization); (4) Mixed 1-, 3-, and 5-Step: 4,000 one-step, 600 three-step, and 400 five-step samples (for each operation op, a dedicated 5k-sample dataset excludes op from all multi-step chains but keeps it in single-step data). For dataset-size accounting, an L-step trajectory contributes L step-level samples; this convention is used only to control the training-data budget. All models are evaluated on held-out 3-, 5-, 7-, and 9-step test sets (500 samples each). Full implementation details are given in Appendix C.

Table 1: Accuracy (%) by training data and test steps. All entries are means over 3 seeds (42, 2026, 12345); standard deviations are shown as subscripts.
<table><tr><td colspan="5">(a) CoT Setting</td><td colspan="5">(b) No-CoT Setting</td></tr><tr><td>Data</td><td>3-step</td><td>5-step</td><td>7-step</td><td>9-step</td><td>|Data</td><td>3-step</td><td>5-step</td><td>7-step</td><td>9-step</td></tr><tr><td>1-step</td><td>91.80±2.24</td><td> $6 3 . 1 3 { \scriptstyle \pm 4 . 1 5 }$ </td><td>36.27±7.87</td><td> $2 2 . 2 7 { \scriptstyle \pm 3 . 6 7 }$ </td><td>1-step</td><td> $1 0 . 9 3 { \scriptstyle \pm 0 . 8 4 }$ </td><td> $7 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $2 . 7 3 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $0 . 6 7 { \scriptstyle \pm 0 . 3 4 }$ </td></tr><tr><td>3-step</td><td> $9 9 . 3 3 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $9 5 . 7 3 { \scriptstyle \pm 0 . 6 8 }$ </td><td>88.87±1.54</td><td> $7 9 . 2 7 { \scriptstyle \pm 3 . 0 3 }$ </td><td>3-step</td><td>75.87±0.84 30.20±0.65 11.00±0.85</td><td></td><td></td><td> $5 . 1 3 { \scriptstyle \pm 0 . 0 9 }$ </td></tr><tr><td> $\mathbf { M i x } _ { 3 + 5 }$ </td><td> $1 0 0 . 0 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td>97.87±0.68</td><td> $9 4 . 0 0 { \scriptstyle \pm 0 . 5 9 }$ </td><td> $9 0 . 4 0 { \scriptstyle \pm 1 . 5 7 }$ </td><td> $\mathbf { M } \mathrm { i } \mathbf { x } _ { 3 + 5 }$ </td><td>51.00±0.75</td><td> $2 6 . 6 0 { \scriptstyle \pm 1 . 4 0 }$ </td><td> $1 2 . 2 0 { \scriptstyle \pm 0 . 7 5 }$ </td><td> $5 . 7 3 { \scriptstyle \pm 0 . 0 9 }$ </td></tr></table>

## 5.1 Main Results and Analysis

Short-to-Long Generalization & CoT vs. No-CoT. Table 1 compares CoT and No-CoT settings. While CoT with single-step data shows some compositionality, its degradation at longer test steps indicates the benefit of short-chain training data. Crucially, models trained on short chains (3-step/3+5 mixed) perform well on longer tasks; since the 5-step chains limit operation types, this implies models generalize stepwise decomposition patterns. In contrast, No-CoT degradation indicates a failure to capture compositional structures without intermediate steps, causing overfitting to spurious correlations.

Generalization to Unseen Combinations. Table 2(a) shows performance when specific operations are excluded from multi-step samples. Even with 9-step test sequences, all experimental groups maintain high accuracy, indicating that models can effectively compose and generalize operations encountered only in single-step data to multi-step tasks, thereby achieving compositional generalization. Table 2(b) again confirms the necessity of CoT: without it, performance remains consistently poor regardless of the excluded operation.

Table 2: Atom-Task: Accuracy (%) when an operation is excluded from multi-step samples. Entries are means over 3 seeds (42, 2026, 12345); standard deviations are shown as subscripts. Rev: Reverse, Dedup: Deduplicate, Sub: Substitute, Dup: Duplicate, Rot: Rotate.
<table><tr><td colspan="5">(a) CoT Setting</td><td colspan="5">(b) No-CoT Setting</td></tr><tr><td>Op</td><td>3-step</td><td>5-step</td><td>7-step</td><td>9-step</td><td>|Op</td><td>3-step</td><td>5-step</td><td>7-step</td><td>9-step</td></tr><tr><td>Rev</td><td> $9 9 . 5 3 { \scriptstyle \pm 0 . 3 4 }$ </td><td> $9 7 . 1 3 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $9 2 . 6 7 { \scriptstyle \pm 1 . 6 0 }$ </td><td> $8 7 . 1 3 { \scriptstyle \pm 0 . 8 4 }$ </td><td>Rev</td><td> $5 3 . 7 3 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $3 0 . 3 3 { \scriptstyle \pm 0 . 7 4 }$ </td><td> $1 6 . 0 0 { \scriptstyle \pm 1 . 1 8 }$ </td><td> $5 . 5 3 { \scriptstyle \pm 0 . 4 7 }$ </td></tr><tr><td>Dedup</td><td> $9 7 . 8 7 { \scriptstyle \pm 0 . 4 7 }$ </td><td> $9 3 . 2 7 { \scriptstyle \pm 1 . 1 1 }$ </td><td> $8 6 . 2 0 { \scriptstyle \pm 1 . 8 2 }$ </td><td> $7 6 . 7 3 { \scriptstyle \pm 0 . 6 6 }$ </td><td>Dedup</td><td> $4 0 . 4 0 { \scriptstyle \pm 0 . 6 5 }$ </td><td> $1 6 . 9 3 { \scriptstyle \pm 0 . 6 6 }$ </td><td> $6 . 8 0 { \scriptstyle \pm 0 . 4 3 }$ </td><td> $1 . 5 3 { \scriptstyle \pm 0 . 0 9 }$ </td></tr><tr><td>Sub</td><td> $9 6 . 8 0 { \scriptstyle \pm 0 . 4 3 }$ </td><td> $9 0 . 7 3 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $8 4 . 3 3 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $7 0 . 6 7 { \scriptstyle \pm 0 . 8 1 }$ </td><td>Sub</td><td> $4 2 . 8 7 { \scriptstyle \pm 0 . 8 1 }$ </td><td> $2 1 . 4 0 { \scriptstyle \pm 0 . 6 5 }$ </td><td> $1 1 . 2 0 { \scriptstyle \pm 0 . 3 3 }$ </td><td> $5 . 0 7 { \scriptstyle \pm 0 . 5 0 }$ </td></tr><tr><td>Dup</td><td> $9 3 . 6 0 { \scriptstyle \pm 0 . 5 9 }$ </td><td> $8 4 . 8 0 { \scriptstyle \pm 0 . 9 9 }$ </td><td> $7 6 . 7 3 { \scriptstyle \pm 1 . 7 5 }$ </td><td> $6 6 . 7 3 { \scriptstyle \pm 1 . 2 5 }$ </td><td>Dup</td><td> $3 8 . 8 7 { \scriptstyle \pm 1 . 2 3 }$ </td><td> $1 7 . 2 0 { \scriptstyle \pm 0 . 7 5 }$ </td><td> $1 1 . 0 7 { \scriptstyle \pm 1 . 0 9 }$ </td><td> $3 . 4 0 { \scriptstyle \pm 0 . 5 7 }$ </td></tr><tr><td>Rot</td><td> $9 6 . 5 3 { \scriptstyle \pm 0 . 6 6 }$ </td><td> $9 1 . 9 3 _ { \pm 0 . 6 6 }$ </td><td> $8 6 . 9 3 { \scriptstyle \pm 0 . 3 4 }$ </td><td> $8 0 . 1 3 { \scriptstyle \pm 0 . 4 1 }$ </td><td>Rot</td><td> $5 5 . 4 7 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $3 5 . 6 0 { \scriptstyle \pm 1 . 8 4 }$ </td><td> $2 2 . 6 7 { \scriptstyle \pm 1 . 0 9 }$ </td><td> $9 . 7 3 { \scriptstyle \pm 0 . 9 3 }$ </td></tr></table>

![](images/83852af3f6dca0f1190f81cc56dc73bea44c99a808ce3783c5f5bd175d435033.jpg)  
Figure 2: Comparison of complete vs. incomplete CoT state recordings.

Effect of State Tracking in CoT. We ablate the CoT format by removing explicit state tracking (remaining tasks and input context) from the intermediate outputs (shown in Figure 2), such that the intermediate state $z _ { i - 1 }$ can no longer carry complete information. Table 3 shows this ablation causes a consistent performance degradation (see the drop in average accuracy on the 7-step test set). This demonstrates that when $z _ { i - 1 }$ fails to preserve the full state, multi-step decomposition is impaired. This validates our theoretical requirement that $z _ { i - 1 }$ must carry complete state information.

Table 3: Format ablation: Accuracy (%) with vs. without explicit state tracking. Abl.: means over 3 seeds (42, 2026, 12345); Orig.: means over 3 seeds from Table 2.
<table><tr><td>Step</td><td>Reverse</td><td>Rotate</td><td>Duplicate</td><td>Deduplicate</td><td>Substitute </td><td>Avg. (Abl.)</td><td> $\mathbf { A v g . \nabla ( O r i g . ) }$ </td></tr><tr><td> $3 { \mathrm { - s t e p } }$ </td><td> $9 8 . 4 7 { \scriptstyle \pm 1 . 0 5 }$ </td><td> $9 5 . 5 3 { \scriptstyle \pm 2 . 0 8 }$ </td><td> $9 2 . 3 3 { \scriptstyle \pm 4 . 9 7 }$ </td><td> $9 6 . 9 3 { \scriptstyle \pm 1 . 4 3 }$ </td><td> $9 4 . 5 3 { \scriptstyle \pm 4 . 2 0 }$ </td><td> $9 5 . 5 6 { \scriptstyle \pm 1 . 4 1 }$ </td><td> $9 6 . 8 7 { \scriptstyle \pm 0 . 2 3 }$ </td></tr><tr><td> $5 \mathrm { - s t e p }$ </td><td> $9 5 . 8 0 { \scriptstyle \pm 2 . 6 9 }$ </td><td> $8 8 . 1 3 { \scriptstyle \pm 5 . 1 2 }$ </td><td> $8 3 . 2 0 { \scriptstyle \pm 6 . 9 5 }$ </td><td> $9 1 . 7 3 { \scriptstyle \pm 5 . 4 9 }$ </td><td> $8 7 . 4 0 { \scriptstyle \pm 7 . 0 8 }$ </td><td> $8 9 . 2 5 { \scriptstyle \pm 2 . 5 5 }$ </td><td> $9 1 . 5 7 { \scriptstyle \pm 0 . 3 4 }$ </td></tr><tr><td> $7 { \cdot } \mathrm { s t e p }$ </td><td> $7 9 . 4 0 { \scriptstyle \pm 1 0 . 1 6 }$ </td><td> $6 7 . 8 0 { \scriptstyle \pm 2 . 6 1 }$ </td><td> $6 7 . 4 7 { \scriptstyle \pm 6 . 1 3 }$ </td><td> $6 2 . 4 7 { \scriptstyle \pm 4 . 4 3 }$ </td><td> $6 0 . 6 7 { \scriptstyle \pm 2 0 . 0 2 }$ </td><td> $6 7 . 5 6 { \scriptstyle \pm 4 . 7 7 }$ </td><td> $8 5 . 3 7 { \scriptstyle \pm 0 . 6 1 }$ </td></tr></table>

Scaling Trends of Compositional Generalization We study performance as a function of reasoning depth, training sample size, and the number of inference-time examples, as summarized in Figure 3. For depth scaling (Figure 3a), error increases with T, with diminishing increments for single-step training. Theorem 2 gives sublinear-in-T control of the estimation term at fixed n. For sample scaling (Figure 3b), increasing the sample size consistently reduces the test error rate, particularly for longer reasoning chains, which is qualitatively consistent with a smaller estimation error at larger sample sizes. Figure 3c varies the number of in-context examples at inference time rather than the theoretical sliding-window length m. Detailed data for these scaling experiments are provided in Appendix C.1.3.

## 6 Further Related Work

CoT Prompting and Training Chain-of-Thought (CoT) enhances complex problem-solving by generating intermediate reasoning steps [1], improving performance on math benchmarks [27]. This led to extensions such as Tree of Thoughts [28], Graph of Thoughts [29], and Program of Thoughts [30], alongside robustness improvements via self-consistency [31], self-refinement [32], and retrieval-augmented generation [33]. Decomposition methods like Least-to-Most prompting [34] and Automatic CoT [35] further automate this process. However, most work focuses on inferencestage prompts, with limited discussion of $\mathrm { C o T } \mathrm { \bar { s } }$ intrinsic training mechanisms [18]. Training LLMs with CoT via supervised fine-tuning or reinforcement learning, on the other hand, enhances capability [3, 5, 36] and improves compositional generalization [25].

![](images/bee5e009f03384c728ad5dae24502348251719309fe4171215ff61e6c279c6a2.jpg)

![](images/7c22c5c3d7dacfc0cceaa9bec5f691f3ce726d8b8dd1e0e0a6ff6cbccf34e0ae.jpg)  
Inverse Sqrt Sample Size (1/ n)

![](images/4657cfda00534d2e4ae1ff3920ecc263560819366aad69743fc37793f7ce7595.jpg)  
Figure 3: Each entry is a mean over 3 seeds (42, 2026, 12345). Atom-Task results when varying (a) test depth, (b) training sample size, and (c) inference-time examples; in (b) the horizontal axis is $1 / { \sqrt { n } } ,$ where n is the sample size in units of the full budget (10%, 20%, 50%, 100%).

CoT Theoretical Analysis Theoretically, CoT increases Transformers’ effective depth for serial problems [14, 15], and linear next-token predictors on CoT data can approximate Turing-computable functions [13]. Further analyses cover sample/time complexity [37], task decomposition [16], incontext learning bounds [17], and expressive power on arithmetic or gradient descent [38], with structural analysis showing reasoning internalized as combinatorial circuits [39, 40]. However, these studies are often qualitative or limited to localized tasks [11, 18]. More specifically, Joshi et al. [37] and Hanneke et al. [41] analyze the PAC (probably approximately correct) error of the final symbol under complete-trajectory supervision and realizability, whereas our setting supervises individual atomic steps through a target-token objective and provides explicit depth-dependent bounds. The analysis of [25] considers out-of-distribution generalization but is restricted to two-step tasks and lacks LLM-specific analysis. Maurer et al. [24] provide a shared-representation perspective for pooled estimation, whereas Li et al. [23] provide the decoder covering-number tool used for our restricted final-k output class.

## 7 Conclusion and Future Work

In this work we asked where the out-of-distribution capability of CoT comes from, and located it in the reuse of one transition model along the chain. The analysis separates a multiplicative price for inference errors, of order ${ \mathcal { O } } ( k )$ , set by the block length rather than by the number of steps, from a statistical cost that grows only as $\mathcal { O } ( \sqrt { T / n } )$ , since a single predictor serves all T sub-tasks and can be assessed on a combined sample. In the token-by-token prediction regime this gives an end-to-end bound that is $B _ { \ell } k /$ log 2 times the best-in-class summed step-level risk, plus an estimation term of order ${ \widetilde { \mathcal { O } } } ( k { \sqrt { T / n } } )$ ; a matched construction for the sub-task ERM shows that the $\sqrt { T / n }$ dependence is not an artefact of the analysis. The decomposition further indicates how learning reusable transitions can support compositional transfer—consistent with our experiments, where explicit state tracking proves essential. Taken together, the analysis sharpens our picture of how compositional OOD generalization arises under CoT, and that picture is not tied to autoregressive token-by-token output: it may also inform future analyses of agents that maintain and update an external memory.

Limitations & Future Work Our theory studies explicit CoT under supervised NTP-style training, where intermediate states and step-level transitions are observable; it excludes adaptive paradigms such as reinforcement fine-tuning and test-time search [11, 42–45], where trajectories may be selected, refined, or rewarded. Extending our decomposition framework to these settings is a natural future direction, and does not affect the mechanism identified here.

## Acknowledgments and Disclosure of Funding

This work was supported by the Special Fund for Key Program of Science and Technology of Jiangsu Province (No. BG2024042); the Gusu Leading Talents Program for Innovation and Entrepreneurship (No. ZXL2025323); the Fundamental and Interdisciplinary Disciplines Breakthrough Plan of the Ministry of Education of China (No. JYB2025XDXM118); and the “111 Center” (No. B26023). This work was also partially supported by the Lee Kong Chian Fellowship awarded to Pan Zhou by Singapore Management University. We thank the NeurIPS reviewers for their valuable feedback and suggestions.

## References

[1] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

[2] Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mammoth: Building math generalist models through hybrid instruction tuning. In The Twelfth International Conference on Learning Representations, 2024.

[3] Longhui Yu, Weisen Jiang, Han Shi, Jincheng YU, Zhengying Liu, Yu Zhang, James Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. In The Twelfth International Conference on Learning Representations, 2024.

[4] Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 9426–9439, 2024.

[5] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

[6] Fangxu Yu. Flow of reasoning: Efficient training of llm policy with divergent thinking. arXiv (Cornell University), 2024.

[7] Namgyu Ho, Laura Schmid, and Se-Young Yun. Large language models are reasoning teachers. In Proceedings of the 61st annual meeting of the association for computational linguistics (volume 1: long papers), pages 14852–14882, 2023.

[8] Seungone Kim, Se Joo, Doyoung Kim, Joel Jang, Seonghyeon Ye, Jamin Shin, and Minjoon Seo. The cot collection: Improving zero-shot and few-shot learning of language models via chain-of-thought fine-tuning. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 12685–12708, 2023.

[9] Team OpenAI. Learning to reason with llms. OpenAI Blog, 2024.

[10] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

[11] Kanishk Gandhi, Ayush K Chakravarthy, Anikait Singh, Nathan Lile, and Noah Goodman. Cognitive behaviors that enable self-improving reasoners, or, four habits of highly effective stars. In Second Conference on Language Modeling, 2025.

[12] Akshara Prabhakar, Thomas L Griffiths, and R Thomas McCoy. Deciphering the factors influencing the efficacy of chain-of-thought: Probability, memorization, and noisy reasoning. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2024, pages 3710–3724, 2024.

[13] Guhao Feng, Bohang Zhang, Yuntian Gu, Haotian Ye, Di He, and Liwei Wang. Towards revealing the mystery behind chain of thought: a theoretical perspective. Advances in Neural Information Processing Systems, 36:70757–70798, 2023.

[14] William Merrill and Ashish Sabharwal. The expressive power of transformers with chain of thought. In The Twelfth International Conference on Learning Representations, 2024.

[15] Zhiyuan Li, Hong Liu, Denny Zhou, and Tengyu Ma. Chain of thought empowers transformers to solve inherently serial problems. In The Twelfth International Conference on Learning Representations, 2024.

[16] Chenxiao Yang, Zhiyuan Li, and David Wipf. Chain-of-thought provably enables learning the (otherwise) unlearnable. In The Thirteenth International Conference on Learning Representations, 2025.

[17] Alireza Amiri Bavandpour, Xinting Huang, Mark Rofin, and Michael Hahn. Lower bounds for chain-of-thought reasoning in hard-attention transformers. In International Conference on Machine Learning, pages 3274–3306. PMLR, 2025.

[18] Ben Prystawski, Michael Li, and Noah Goodman. Why think step by step? reasoning emerges from the locality of experience. Advances in Neural Information Processing Systems, 36:70926– 70947, 2023.

[19] Shai Shalev-Shwartz and Shai Ben-David. Understanding machine learning: From theory to algorithms. Cambridge university press, 2014.

[20] Liyuan Liu, Biqin Song, Zhibin Pan, Chuanwu Yang, Chi Xiao, and Weifu Li. Gradient learning under tilted empirical risk minimization. Entropy, 24(7):956, 2022.

[21] Yuyang Deng, Junyuan Hong, Jiayu Zhou, and Mehrdad Mahdavi. On the generalization ability of unsupervised pretraining. In International Conference on Artificial Intelligence and Statistics, pages 4519–4527. PMLR, 2024.

[22] Liyuan Liu, Hong Chen, Chi Xiao, and Weifu Li. The consistency analysis of gradient learning under independent covariate shift. Neurocomputing, 635:129883, 2025.

[23] Zhihao Li, Xue Jiang, Liyuan Liu, Xuelin Zhang, Hong Chen, and Feng Zheng. On the generalization ability of next-token-prediction pretraining. In Forty-second International Conference on Machine Learning, 2025.

[24] Andreas Maurer, Massimiliano Pontil, and Bernardino Romera-Paredes. The benefit of multitask representation learning. Journal ofMachine Learning Research, 17(81):1–32, 2016.

[25] Xinhao Yao, Ruifeng Ren, Yun Liao, Lizhong Ding, and Yong Liu. Compositional generalization from learned skills via cot training: A theoretical and structural analysis for reasoning. In The Fourteenth International Conference on Learning Representations, 2026.

[26] Amirhesam Abedsoltan, Huaqing Zhang, Kaiyue Wen, Hongzhou Lin, Jingzhao Zhang, and Mikhail Belkin. Task generalization with autoregressive compositional structure: Can learning from d tasks generalize to d<sup>T</sup> tasks? arXiv preprint arXiv:2502.08991, 2025.

[27] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

[28] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems, 36:11809–11822, 2023.

[29] Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, et al. Graph of thoughts: Solving elaborate problems with large language models. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pages 17682–17690, 2024.

[30] Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. Transactions on Machine Learning Research, 2023.

[31] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2023.

[32] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. Advances in neural information processing systems, 36:46534–46594, 2023.

[33] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33:9459–9474, 2020.

[34] Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc V Le, et al. Least-to-most prompting enables complex reasoning in large language models. In The Eleventh International Conference on Learning Representations, 2023.

[35] Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. Automatic chain of thought prompting in large language models. In The Eleventh International Conference on Learning Representations, 2023.

[36] Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. Star: Bootstrapping reasoning with reasoning. Advances in Neural Information Processing Systems, 35:15476–15488, 2022.

[37] Nirmit Joshi, Gal Vardi, Adam Block, Surbhi Goel, Zhiyuan Li, Theodor Misiakiewicz, and Nathan Srebro. A theory of learning with autoregressive chain of thought. In The Thirty Eighth Annual Conference on Learning Theory, pages 3161–3212. PMLR, 2025.

[38] Jianhao Huang, Zixuan Wang, and Jason D Lee. Transformers learn to implement multi-step gradient descent with chain of thought. In The Thirteenth International Conference on Learning Representations, 2025.

[39] Boshi Wang, Xiang Yue, Yu Su, and Huan Sun. Grokking of implicit reasoning in transformers: A mechanistic journey to the edge of generalization. Advances in Neural Information Processing Systems, 37:95238–95265, 2024.

[40] Eden Biran, Daniela Gottesman, Sohee Yang, Mor Geva, and Amir Globerson. Hopping too late: Exploring the limitations of large language models on multi-hop queries. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 14113–14130, 2024.

[41] Steve Hanneke, Idan Mehalel, and Shay Moran. Sample complexity of autoregressive reasoning: Chain-of-thought vs. end-to-end. arXiv preprint arXiv:2604.12013, 2026.

[42] Gokul Swamy, Sanjiban Choudhury, Wen Sun, Zhiwei Steven Wu, and J Andrew Bagnell. All roads lead to likelihood: The value of reinforcement learning in fine-tuning. arXiv preprint arXiv:2503.01067, 2025.

[43] Amrith Setlur, Nived Rajaraman, Sergey Levine, and Aviral Kumar. Scaling test-time compute without verification or rl is suboptimal. In International Conference on Machine Learning, pages 54058–54094. PMLR, 2025.

[44] Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[45] Darsh J Shah, Peter Rushton, Somanshu Singla, Mohit Parmar, Kurt Smith, Yash Vanjani, Ashish Vaswani, Adarsh Chaluvaraju, Andrew Hojel, Andrew Ma, et al. Rethinking reflection in pre-training. arXiv preprint arXiv:2504.04022, 2025.

## A Supplementary Details on NTP Setting

This section supplements Section 4 by defining the decoder architecture and the sampling units used in the target-token analysis. We use the convention of Section 4.1: m denotes the complete-record length (and the maximum sequence length used by the decoder), the block-boundary state has length $m - k$ , and V denotes the vocabulary.

## A.1 Decoder-only Model Architecture

Contemporary Large Language Models (LLMs) predominantly utilize a decoder-only Transformer architecture. As described in Section 4.1, the composite model $\mathrm { L M } = g \circ h$ consists of a representation learner h and a token predictor g. We formalize the representation learner h as an L-layer Transformer decoder, following the block description of [23, Eq. (4)], mapping an input token sequence $Z \in \mathcal { V } ^ { m }$ to a hidden representation $h ( Z ) \in \mathbf { \bar { \mathbb { R } } } ^ { m \times d }$ . The decoder applies to any prefix $Z \in { \bar { \mathcal { V } } } ^ { s }$ with $s \leq m$ and the prediction head reads its last hidden state.

The process begins with an Embedding Layer, which converts discrete tokens into continuous vectors $Z _ { 0 } \in \mathbb { R } ^ { m \times d }$ via:

$$
Z _ { 0 } = \mathrm { E m b e d d i n g } ( Z ) : = \mathrm { O n e H o t } ( Z ) W _ { e } + W _ { p } ,
$$

where $\mathrm { O n e H o t } ( Z ) \in \{ 0 , 1 \} ^ { m \times n _ { v } }$ is the one-hot encoding of the m tokens, $W _ { e } \in \mathbb R ^ { n _ { v } \times d }$ is the token embedding matrix, and $W _ { p } \in \mathbb { R } ^ { m \times d }$ is the position embedding matrix. The core of the architecture consists of L stacked Transformer-decoder blocks. For layer $l \in \{ 1 , \ldots , L \}$ , the output $Z _ { l }$ is computed from $Z _ { l - 1 }$ via masked self-attention and feed-forward networks (FFN). Specifically, let H be the number of attention heads and $d _ { k } = d / H$ . The masked self-attention head $\dot { h } \in [ H ]$ is defined as:

$$
\mathrm { A t t n } _ { h } ( Z _ { l - 1 } ) = \mathrm { s o f t m a x } \left( \frac { Q _ { h } K _ { h } ^ { \top } } { \sqrt { d _ { k } } } + M \right) V _ { h } ,
$$

$$
Q _ { h } = Z _ { l - 1 } W _ { h } ^ { Q } , \quad K _ { h } = Z _ { l - 1 } W _ { h } ^ { K } , \quad V _ { h } = Z _ { l - 1 } W _ { h } ^ { V } .
$$

Here, $M \in \mathbb { R } ^ { m \times m }$ is a causal mask matrix where $M _ { i j } = - { \infty } \operatorname { i f } j > i$ and 0 otherwise. This mask ensures that the hidden state at position t attends only to tokens at positions $\leq t ,$ , so that the token at position t is predicted from the state at position $t - 1$ , thereby implementing the autoregressive property essential for Next-Token Prediction (NTP). The final output of the layer is given by:

$$
Y _ { l } : = \mathrm { N o r m } _ { 1 } \left( \sum _ { h = 1 } ^ { H } \mathrm { A t t m } _ { h } ( Z _ { l - 1 } ) W _ { h } ^ { O } + Z _ { l - 1 } \right) , \quad Z _ { l } = \mathrm { N o r m } _ { 2 } \left( \mathrm { F F N } ( Y _ { l } ) + Y _ { l } \right) ,
$$

where Norm , Norm denote Layer Normalization and $\mathrm { F F N } ( Y ) = \sigma ( Y W _ { F 1 } ) W _ { F 2 }$ is a two-layer feed-forward network with hidden width $d _ { f } = 4 d$ . The representation learner thus serializes the input into a deep representation $h ( Z ) = Z _ { L }$

The token predictor g applies a linear projection to the hidden state at the last position, $g ( h ( Z ) ) =$ softmax $( ( h ( Z ) ) _ { s , : } W _ { P } )$ with $s = | Z |$ and $W _ { P } \in \mathbb { R } ^ { d \times n _ { \imath } }$ , mapping it to a probability distribution over the vocabulary V.

## A.2 Explicit Parameterization and Derivation of the Dimension-Dependent Bound

To quantify the Rademacher term in Corollary 6, we use the matrix-covering tools recorded in [23, Lemma C.3]. The statistical argument below is carried out directly for sequence-level losses.

Assumption 2 (Capacity and Regularity Constraints). For the L-layer, H-head decoder-only Transformer defined in Appendix A.1 (following [23, Eq. (4) and Assumption 4.14]), we assume:

• The Layer Normalization operators $\Pi _ { \mathrm { n o r m } }$ (the operators Norm and Norm of $\mathsf { A p - }$ pendix A.1) are G<sub>π</sub>-Lipschitz continuous with respect to the 2-norm.

• For all layers $l \in [ L ]$ and attention heads $h \in [ H ]$ , the scaled attention scores satisfy

$$
\| Q _ { l } ^ { h } ( K _ { l } ^ { h } ) ^ { \top } / \sqrt { d _ { k } } \| _ { \ell _ { \infty } } \leq C _ { l } .
$$

• Each attention and feed-forward parameter matrix in layer l has Frobenius norm at most $B _ { l }$

The embedding matrices and output head also have bounded Frobenius norms. Let $B _ { * } \geq 1$ be a common upper bound for these norms and the $B _ { l }$ . For the capacity estimate, normalization is a fixed rowwise operator whose output rows have norm at most $C _ { \pi } { \sqrt { d } } ,$ with fixed $C _ { \pi }$ ; the feed-forward activation is 1-Lipschitz and vanishes at zero, as does ReLU. These are the usual bounded-row normalization and feed-forward conditions. The constants $B _ { * } , G _ { \pi } , C _ { \pi }$ are fixed independently of the sample size and model dimensions. Let Θ count all trainable scalar parameters, including token and positional embeddings whenever they are trained, and the output head.

The norm and Lipschitz upper bounds may be enlarged so that $B _ { l } , G _ { \pi } \ge 1$ . Let $N = n T$ and $\begin{array} { r } { \rho _ { L } = \sum _ { l = 1 } ^ { L } B _ { l } ^ { 2 } } \end{array}$ , and set

$$
s _ { L } ( m , N ) = \prod _ { l = 1 } ^ { L } G _ { \pi } ^ { 2 } ( B _ { l } ^ { 2 } + 1 ) \left( e ^ { C _ { l } } B _ { l } ^ { 2 } H \sqrt { N } m d + 1 \right) ,
$$

Here $s _ { L }$ collects the per-layer amplification factors that a parameter perturbation of size η accumulates over the L blocks, and $\tau _ { m , N }$ is the logarithm of the resulting covering-number bound; the individual factors are derived in the propagation estimate below.

For fixed norm, attention-score, and normalization bounds, $\tau _ { m , N } = \mathcal { O } ( L \log ( 2 + N H m d ) +$ $\log ( 1 + L ) )$ . Below, constants in $\lesssim$ may depend on the fixed bounds $B _ { * } , G _ { \pi } , C _ { \pi }$ , but not on $G _ { \ell } , N , m , k , L , d , H , n _ { v }$

Derivation of $E q .$ . (4). We first record a coarse output cover, including every trainable matrix. A Frobenius ball of radius $B _ { * }$ in $\mathbb { R } ^ { a \times b }$ has an η-net, i.e., a set of points whose η-neighbourhood covers the ball, of cardinality at most $( 1 + 2 B _ { * } / \eta ) ^ { a b }$ , by vectorization and [23, Lemma C.3]. Taking the product of these nets gives at most $( 1 + 2 B _ { * } / \eta ) ^ { \Theta }$ parameter representatives. Rather than composing the layerwise coverings of [23, Lemma 4.15 and Proposition 4.17], we propagate the approximation errors through the actual block of Appendix ${ \bf A } . 1 .$ , which keeps our normalization placement unchanged.

For completeness, write $\mathrm { T F } _ { l }$ for block l. A normalized row has norm at most $C _ { \pi } { \sqrt { d } } .$ , and the bounded feed-forward weights and residual therefore give a block-output row norm at most a constant times ${ \sqrt { d } }$ . Embedding rows satisfy the same type of bound. The causal softmax has Euclidean Lipschitz constant at most 1, and its attention matrix has operator norm at most $\sqrt { m }$ . For two such admissible inputs, if every matrix in the two blocks differs by at most η, multiplication of the query, key, value, and output matrices, followed by the residual and feed-forward operations, gives

$$
\| \mathrm { T F } _ { l } ( Z ) - \mathrm { T F } _ { l } ^ { \prime } ( Z ^ { \prime } ) \| _ { F } \leq \Gamma \big ( \| Z - Z ^ { \prime } \| _ { F } + \sqrt { m } \eta \big ) , \qquad \Gamma = c H m d ^ { 2 } \geq 1 ,
$$

where c depends only on $B _ { * } , G _ { \pi } , C _ { \pi }$ . Indeed, the query-key input perturbation is bounded by a constant times $\sqrt { m d } \| Z - Z ^ { \prime } \| _ { F }$ , its parameter perturbation by a constant times mdη, and the remaining projections contribute at most a constant times ${ \sqrt { m d } } ;$ the displayed Γ bounds all these factors. The embedding approximation error is at most $2 { \sqrt { m } } \eta .$ . Induction over the $L$ blocks and inclusion of the output head thus give, for the concatenated probability outputs on $N$ records,

$$
\| A - A ^ { \prime } \| _ { F } \leq \sqrt { N m } \Lambda \eta , \qquad \Lambda = ( L + 3 ) \Gamma ^ { L + 1 } .
$$

This argument covers trainable embeddings as part of the parameter vector; fixed embeddings require no net. Covering representatives can be taken in the norm balls, and recentering in the original function class, if necessary, only changes numerical constants.

Let $\Pi _ { k }$ select the final k prediction rows, aligned with the target labels as in Eq. (1). Every such row is computed from its teacher-forced prefix. Since

$$
\Vert \Pi _ { k } A - \Pi _ { k } B \Vert _ { F } \leq \Vert A - B \Vert _ { F } ,
$$

a full-output cover also covers the selected rows. For one record, write $q ( A )$ for its target-token loss with the labels fixed. The loss Lipschitz condition and Cauchy–Schwarz give

$$
\begin{array} { r l r } {  { | q ( A ) - q ( B ) | \le \frac { G _ { \ell } } { k } \sum _ { r = 1 } ^ { k } \| A _ { m - k + r } - B _ { m - k + r } \| _ { 2 } } } \\ & { } & { \le \frac { G _ { \ell } } { \sqrt { k } } \| \Pi _ { k } A - \Pi _ { k } B \| _ { F } . ~ } \end{array}
$$

Hence the concatenated N-dimensional loss vectors are approximated within $G _ { \ell } \sqrt { N m / k } \Lambda \eta$ (aggregating the N records in $\ell _ { 2 } )$ . Choosing $\eta = \epsilon \sqrt { k } / ( G _ { \ell } \sqrt { N m } \Lambda )$ , and setting $\epsilon = G _ { \ell } \sqrt { N } u ,$ yields the covering-number bound

$$
\log \mathcal { N } ( \mathcal { Q } _ { k } | _ { S _ { 0 } } , G _ { \ell } \sqrt { N } u , \| \cdot \| _ { 2 } ) \leq \Theta \log \left( 1 + \frac { 2 B _ { * } \Lambda \sqrt { m / k } } { u } \right) ,
$$

where $\mathcal { N } ( \cdot , \cdot , \| \cdot \| _ { 2 } )$ denotes the covering number, i.e., the smallest cardinality of such a net in the $\ell _ { 2 }$ norm, and $\mathcal { Q } _ { k } | _ { S _ { 0 } }$ is the restriction of $\mathcal { Q } _ { k }$ to the sample $S _ { 0 }$ . Center the loss vectors at any fixed member of ${ \mathcal { Q } } _ { k } | _ { S _ { 0 } }$ . This leaves their empirical Rademacher complexity unchanged. The probability simplex has diameter $\sqrt { 2 }$ , so each centered loss vector has norm at most $G _ { \ell } \sqrt { 2 N }$ . Dudley’s entropy integral with the sequence-level normalization $1 / N$ gives

$$
\begin{array} { r l } & { \widehat { \mathfrak { R } } _ { S _ { 0 } } ( \mathcal { Q } _ { k } ) \lesssim G _ { \ell } \sqrt { \frac { \Theta } { N } } \displaystyle \int _ { 0 } ^ { \sqrt { 2 } } \sqrt { \log \left( 1 + \frac { 2 B _ { * } \Lambda \sqrt { m / k } } { u } \right) } d u } \\ & { \qquad \lesssim G _ { \ell } \sqrt { \frac { \Theta } { N } } \displaystyle \sqrt { 1 + \log \left( 1 + 2 B _ { * } \Lambda \sqrt { m / k } \right) } . } \end{array}
$$

The last step uses $\begin{array} { r } { \int _ { 0 } ^ { \sqrt { 2 } } \sqrt { \log ( 1 + a / u ) } d u \ \lesssim \ \sqrt { 1 + \log ( 1 + a ) } } \end{array}$ for $a \ge ~ 0$ . Since $k \geq 1$ , the logarithmic factor is at most a constant times $1 + \log ( L + 3 ) + ( L + 1 ) \log ( 2 + H m d )$ . On the other hand, $B _ { l } , G _ { \pi } \ge 1$ and $C _ { l } \geq 0$ give $\rho _ { L } \geq L$ and $s _ { L } \geq [ 2 ( 1 + \tilde { H m } d ) ] ^ { L }$ , hence

$$
\tau _ { m , N } \geq \log \left( 1 + L \left[ 2 ( 1 + H m d ) \right] ^ { L } \right) ,
$$

which dominates this factor. We have therefore obtained

$$
\widehat { \mathfrak { R } } _ { S _ { 0 } } ( \mathcal { Q } _ { k } ) \lesssim G _ { \ell } \sqrt { \frac { \Theta \tau _ { m , N } } { N } } .
$$

Because $d H \geq 1$ , this also proves the coarser form used in the main text,

$$
\widehat { \mathfrak { R } } _ { S _ { 0 } } ( \mathcal { Q } _ { k } ) \lesssim G _ { \ell } \sqrt { \frac { \Theta d H \tau _ { m , N } } { N } } + G _ { \ell } \sqrt { \frac { d n _ { v } } { N } } .
$$

Here Θ already counts the head and any trainable embeddings, so the first term alone is valid and the second is redundant; the displayed bound is a looser consequence of the preceding estimate. The proof uses one Rademacher sign per sequence and introduces one factor $\bar { G _ { \ell } } ;$ it never replaces these signs by independent signs for individual tokens.

Finally, $N = n T$ gives

$$
T \widehat { \mathfrak { R } } _ { S _ { 0 } } ( { \mathcal { Q } } _ { k } ) \lesssim G _ { \ell } \sqrt { \frac { \Theta d H \tau _ { m , n T } T } { n } } + G _ { \ell } \sqrt { \frac { T d n _ { v } } { n } } .
$$

Substituting this in Eq. (3) gives Eq. (5). The coefficients 4T and $6 B _ { \ell }$ in the ERM bound are unchanged, and the reduction factor $B _ { \ell } k /$ log 2 is applied once. Context length enters through $\tau _ { m , n T }$ and also through Θ when the number of trainable positional-embedding parameters grows with m. □

## A.3 Sampling Units

For each step i, the dataset $S _ { i }$ contains n independently sampled complete sequences from the corresponding step-level skill distribution $\mathcal { D } _ { i }$ . These sequences need not be extracted from a shared trajectory; each is drawn independently from some real-world instance of the same skill. Dependence among tokens and prefixes within a complete sequence is unrestricted. This independent-record view is the idealization underlying the bounds. The Atom-Task experiments use the one-step and short-chain datasets specified in Section 5, including three- and five-step chains; the L-step accounting serves only as a budget convention, not an independence assumption for transitions within a chain.

Table 4: Summary of Main Notations
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $\nu$ </td><td>Vocabulary of size  $n _ { v } .$ </td></tr><tr><td>hard(·)</td><td>Operator converting probability distributions to discrete tokens via argmax.</td></tr><tr><td> $\mathcal { D } , \mathcal { D } _ { i }$ </td><td>Target problem distribution, and data distribution for the ¿-th reasoning step.</td></tr><tr><td> $\Psi , \psi , \psi _ { * }$ </td><td>Hypothesis class for single-step functions, learned single-step function, and ground-truth mapping (not necessarily within  $\Psi )$ </td></tr><tr><td> $\bar { \psi }$ </td><td>State transition mapping,  $\bar { \psi } ( z ) : = z [ k + 1 : m ] \oplus \mathrm { h a r d } ( \psi ( z ) )$  , representing the input state for the next step.</td></tr><tr><td> $\mathcal { G } , \mathcal { H } , g , h$   $\psi _ { g , h } , F _ { \psi _ { g , h } }$ </td><td>Hypothesis classes and functions for the token predictor and representation learner. Block predictor of k successive next-token predictions by  $g \circ h ,$  and the resulting final</td></tr><tr><td> $\ell , \mathfrak { L }$ </td><td>T-step predictor. Token-level loss (€e-clipped cross-entropy of Assumption 1) and sequence-level loss</td></tr><tr><td> $\epsilon _ { \ell } , G _ { \ell } , B _ { \ell }$ </td><td>(average over the k generated tokens). Clipping threshold of  $\ell ( \epsilon _ { \ell } \in ( 0 , 1 / 4 ) )$  , Lipschitz constant of l w.r.t. the 2-norm, and</td></tr><tr><td> $\mathcal { F }$ </td><td>bound of l (hence of the sequence-level loss £). Sequence-loss class  ${ \mathcal { F } } = { \dot { \mathfrak { L } _ { \circ } } } \Psi = \{ f _ { \psi } : \psi \in \Psi \}$  with  $f _ { \psi } ( z , \Delta z ) = \mathfrak { L } ( \psi ( z ) , \Delta z )$ </td></tr><tr><td> $T , m , k , n$ </td><td>Total number of reasoning steps, state length in Section 2 (complete-record length in</td></tr><tr><td> $R _ { \mathcal { D } } ( F _ { \psi } )$ </td><td>Section 4), number of generated tokens per step, number of samples per step.</td></tr><tr><td> $\mathcal { D } ^ { \prime } , P$ </td><td>Final-task population risk,  $\begin{array} { r } { R _ { \mathcal { D } } ( F _ { \psi } ) = \hat { \int } \hat { \mathfrak { L } } ( F _ { \psi } ( z _ { 0 } ) , \Delta z _ { T } ) d P ( \acute { z } _ { 0 } , \dot { \Delta z } _ { T } ) } \end{array}$  D is the law of input-answer pairs  $( z _ { 0 } , \Delta z _ { T } )$  with measure  $P ,$  and  $\mathcal { D } ^ { \prime }$  the joint law of the</td></tr><tr><td> $R _ { D _ { i } } ^ { k } , \widehat { R } _ { S _ { i } } ^ { k }$ </td><td>reasoning trajectory  $( z _ { 0 } , \Delta z _ { 1 } , \dots , \Delta z _ { T } )$  Target-token population and empirical risks of the block predictor at the i-th reasoning</td></tr><tr><td> $N , q _ { \psi _ { g , h } } , \mathcal { Q } _ { k }$ </td><td>step. Number of complete sequence samples, target-token loss, and its function class.</td></tr><tr><td> $Q _ { i } , \mathrm { d i s c } \mathcal { F }$ </td><td>Atomic training distribution and train-target discrepancy.</td></tr><tr><td> $\mathcal { E } _ { k } ( \widehat \psi )$ </td><td> $\widehat { \psi } : = \psi _ { \widehat { g } , \widehat { h } }$  Target-token atomic excess risk of the learned block predictor, where</td></tr><tr><td> $\Theta , L , d , H , n _ { v }$ </td><td>Number of model parameters, number of layers, model dimension, number of attention</td></tr><tr><td> $\Pi _ { k } , \tau _ { m , N }$ </td><td>heads, vocabulary size.</td></tr></table>

## B Technical appendices and supplementary material

## B.1 Technical Preliminaries

We provide the formal definitions for the theoretical frameworks used in our analysis.

PAC Learning We conduct analysis within the Probably Approximately Correct (PAC) framework, deriving explicit upper bounds for the generalization error that hold with probability at least $1 - \delta .$

Rademacher Complexity Rademacher complexity measures the richness of a function class ${ \mathcal F } .$ For a fixed sample $S = \{ x _ { 1 } , \ldots , x _ { N } \}$ , the empirical Rademacher complexity and expected Rademacher complexity are defined as

$$
\widehat { \mathfrak { R } } _ { S } ( \mathcal { F } ) = \mathbb { E } _ { \sigma } \left[ \operatorname* { s u p } _ { f \in \mathcal { F } } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sigma _ { i } f ( x _ { i } ) \right] , \quad \mathfrak { R } _ { N } ( \mathcal { F } ) = \mathbb { E } _ { S } [ \widehat { \mathfrak { R } } _ { S } ( \mathcal { F } ) ] ,
$$

where $\sigma _ { i }$ are independent Rademacher random variables taking values $\pm 1$ with equal probability.

Total Variation Distance For two probability distributions $P$ and $Q$ over the same measurable space X, the total variation distance is defined as

$$
\| P - Q \| _ { \mathrm { T V } } = \operatorname* { s u p } _ { A \subseteq \mathcal { X } } | P ( A ) - Q ( A ) | .
$$

Since $| \mathfrak { L } | \le B _ { \ell }$ , this distance bounds the discrepancy of Corollary 4: dis $\begin{array} { r } { \mathsf { c } _ { \mathcal { F } } ( D _ { i } , Q _ { i } ) \le 2 B _ { \ell } \| \mathcal { D } _ { i } - } \end{array}$ $Q _ { i } \| _ { \mathrm { T V } }$

## B.2 Local Error Floor of a Hard-Decoding Error

To derive the generalization error decomposition, we first record the loss incurred by a hard-decoding error at a single step.

Lemma 7 (Local Error Floor of a Hard-Decoding Error). For the autoregressive predictor $\psi ,$ every state s and every ground-truth block $\Delta s$ satisfy

$$
\mathfrak { L } ( \psi ( s ) , \Delta s ) \ge \frac { \log 2 } { k } \mathbb { 1 } \{ \mathrm { h a r d } ( \psi ( s ) ) \ne \Delta s \} ,
$$

where $( \log 2 ) / k$ is a lower bound on the sequence-level loss L whenever the hard decoding differs from the ground truth.

Proof. Fix a state s and a ground-truth block $\Delta s .$ , and consider a token whose ground-truth index is p and whose argmax index is $q \neq p .$ . Then $\hat { y } _ { q } \ge \hat { y } _ { p }$ , and since probabilities are non-negative and sum to at most 1:

$$
1 \geq \hat { y } _ { p } + \hat { y } _ { q } \geq 2 \hat { y } _ { p } \implies \hat { y } _ { p } \leq 1 / 2 .
$$

Since max $\{ \hat { y } _ { p } , \epsilon _ { \ell } \} \le 1 / 2$ whenever $\hat { y } _ { p } \le 1 / 2$ and $\epsilon _ { \ell } < 1 / 4$ , the token-level clipped cross-entropy of every mispredicted token satisfies $\dot { \ell } = \dot { \mathrm { ~ - ~ } } \log \operatorname* { m a x } \{ \hat { y } _ { p } , \epsilon _ { \ell } \} \geq \log 2$ . A block is hard-decodingerroneous only if at least one of its k tokens is mispredicted, so averaging over the k tokens yields $\begin{array} { r } { \mathfrak { L } ( \psi ( s ) , \Delta s ) \ge \frac { \log 2 } { k } > 0 } \end{array}$ , which is the claim. □

Remark 7. The bound uses deterministic hard predictions: if the hard predictions at a step match the ground truth then no error is recorded, and otherwise the deviation is charged through the lower bound $( \log 2 ) / k$ on the local loss. This is the fact used by the risk decomposition of Theorem 1.

## B.3 Proof of Theorem 1 (Generalization Error Decomposition)

Proof. We aim to bound the generalization error $R _ { \mathcal { D } } ( F _ { \psi } )$ . Let $\mathbf { z } = ( z _ { 0 } , \Delta z _ { 1 } , \hdots , \Delta z _ { T } ) \sim \mathcal { D } ^ { \prime }$ and let $p ( \mathbf { z } )$ denote its density. Recall that the oracle state is written $z _ { i } ,$ whereas the free-running (FR) state generated by the model is written with a hat, $\widehat { z } _ { 0 } = z _ { 0 }$ and $\widehat { z } _ { i } = \bar { \psi } ( \widehat { z } _ { i - 1 } )$ , so that

$$
R _ { \mathcal { D } } ( F _ { \psi } ) = \int \mathfrak { L } \big ( \psi ( \widehat { z } _ { T - 1 } ) , \Delta z _ { T } \big ) d p ( \mathbf { z } ) .
$$

We first record a fact that will be used repeatedly: the FR and oracle trajectories coincide exactly up to thefirst hard-decoding error. Formally, let

$$
u _ { i } : = \mathbb { 1 } \big \{ \mathrm { h a r d } ( \psi ( \widehat { z } _ { i - 1 } ) ) \neq \Delta z _ { i } \big \} , \qquad I : = \operatorname* { i n f } \{ i \geq 1 : u _ { i } = 1 \} ,
$$

with $I = \infty$ if no error occurs. Then on the event $\{ u _ { 1 } = \cdot \cdot \cdot = u _ { i - 1 } = 0 \}$ the two states agree, $\widehat { z } _ { i - 1 } = z _ { i - 1 }$ . Indeed, $\widehat { z } _ { 0 } = z _ { 0 }$ , and if ${ \widehat { z } } _ { j - 1 } = z _ { j - 1 }$ while $u _ { j } = 0$ , then

$$
\widehat { \Delta { z } } _ { j } = \mathrm { h a r d } ( \psi ( \widehat { z } _ { j - 1 } ) ) = \mathrm { h a r d } ( \psi ( z _ { j - 1 } ) ) = \Delta z _ { j } ,
$$

so that

$$
\widehat { z } _ { j } = \widehat { z } _ { j - 1 } [ k + 1 : m ] \oplus \widehat { \Delta z } _ { j } = z _ { j - 1 } [ k + 1 : m ] \oplus \Delta z _ { j } = z _ { j } .
$$

The claim follows by induction. We stress that this is a consequence of hard decoding alone—before the first error the two trajectories are identical rather than merely close—and that no Lipschitz or stability property of ψ is used.

Next, we split the final-step risk at the first error. On $\lbrace I > T - 1 \rbrace$ the observation above gives $\widehat { z } _ { T - 1 } = z _ { T - 1 }$ , so the integrand is exactly the step-T oracle loss; on $\left\{ I \leq T - 1 \right\}$ it is bounded by $B _ { \ell } .$ . Pointwise,

$$
\mathfrak { L } \big ( \psi ( \widehat { z } _ { T - 1 } ) , \Delta z _ { T } \big ) \leq \mathfrak { L } \big ( \psi ( z _ { T - 1 } ) , \Delta z _ { T } \big ) + B _ { \ell } \mathbb { 1 } \{ I \leq T - 1 \} ,
$$

and integrating against $p ( \mathbf { z } )$ yields

$$
R _ { \mathcal { D } } ( F _ { \psi } ) \leq R _ { \mathcal { D } _ { T } } ( \psi ) + B _ { \ell } \operatorname* { P r } ( I \leq T - 1 ) = R _ { \mathcal { D } _ { T } } ( \psi ) + B _ { \ell } \sum _ { i = 1 } ^ { T - 1 } \operatorname* { P r } ( I = i ) ,
$$

where the last step uses that the events $\{ I = i \}$ are disjoint.

It remains to charge each first-error probability to the corresponding atomic risk. Fix $i \leq T - 1$ . Since $\{ I = i \} = \{ I \geq \bar { i } \} \cap \{ u _ { i } = 1 \}$ , and since on $\{ I \geq i \}$ the observation above gives $\widehat { z } _ { i - 1 } = z _ { i - 1 }$ , the event $\{ I = i \}$ coincides with

$$
\{ I \geq i \} \cap \{ { \mathrm { h a r d } } ( \psi ( z _ { i - 1 } ) ) \neq \Delta z _ { i } \} .
$$

Dropping the event $\{ I \geq i \}$ , which can only increase the probability, and using that $( z _ { i - 1 } , \Delta z _ { i } )$ has law $\mathcal { D } _ { i }$ , we obtain

$$
\begin{array} { r } { \operatorname* { P r } ( I = i ) \le \operatorname* { P r } _ { i } \big ( \mathrm { h a r d } ( \psi ( z _ { i - 1 } ) ) \ne \Delta z _ { i } \big ) . } \end{array}
$$

Finally, applying the error floor of Lemma $^ { 7 }$ pointwise and taking the expectation over $\mathcal { D } _ { i }$ gives

$$
\operatorname* { P r } _ { \mathcal { D } _ { i } } \left( \operatorname { h a r d } ( \psi ( z _ { i - 1 } ) ) \neq \Delta z _ { i } \right) \leq \frac { k } { \log 2 } \mathbb { E } _ { \mathcal { D } _ { i } } \left[ \mathfrak { L } ( \psi ( z _ { i - 1 } ) , \Delta z _ { i } ) \right] = \frac { k R _ { \mathcal { D } _ { i } } ( \psi ) } { \log 2 } .
$$

Summing over $i = 1 , \dots , T - 1$ thus gives

$$
\operatorname* { P r } ( I \leq T - 1 ) \leq \frac { k } { \log 2 } \sum _ { i = 1 } ^ { T - 1 } R _ { \mathcal { D } _ { i } } ( \psi ) .
$$

Finally, we combine the two estimates. Since $\ell ( p , y ) \leq \log ( 1 / \epsilon _ { \ell } )$ for all $p , y .$ , we may take $B _ { \ell } =$ $\log ( 1 \bar { / } \epsilon _ { \ell } )$ ; then $\epsilon _ { \ell } < 1 / 4$ and $k \geq 1$ imply

$$
{ \frac { B _ { \ell } k } { \log 2 } } \geq 2 > 1 ,
$$

the single term $R _ { { D } _ { T } } ( \psi )$ is dominated by $\frac { B \ell k } { \log 2 } R _ { { \cal D } _ { T } } ( \psi )$ , and hence

$$
R _ { \mathcal { D } } ( F _ { \psi } ) \leq \frac { B _ { \ell } k } { \log 2 } \sum _ { i = 1 } ^ { T } R _ { \mathcal { D } _ { i } } ( \psi ) .
$$

Because each intermediate state $z _ { i - 1 }$ is deterministically constructed from the prefix $( z _ { 0 } , \Delta z _ { 1 } , . . . , \Delta z _ { i - 1 } )$ , the joint law of $( z _ { i - 1 } , \Delta z _ { i } )$ induced by $\mathcal { D } ^ { j }$ is exactly the sub-task distribution $\mathcal { D } _ { i } \mathbf { ; }$ ; marginalizing out the remaining variables, the bound becomes

$$
R _ { \mathcal { D } } ( F _ { \psi } ) \leq \frac { B _ { \ell } k } { \log 2 } \sum _ { i = 1 } ^ { T } \int \mathfrak { L } ( \psi ( z _ { i - 1 } ) , \Delta z _ { i } ) d \mathcal { D } _ { i } ( z _ { i - 1 } , \Delta z _ { i } ) ,
$$

which is the desired decomposition. The coefficient $\frac { B _ { \ell } k } { \log 2 }$ is thus depth-independent, with $( \log 2 ) / k$ the local error floor of Lemma 7. □

## B.4 Proof of Theorem 2 (Rademacher Complexity Bound)

We first present the multi-distribution symmetrization lemma used in Section 3.2.

Lemma 8. Define the population risk $\begin{array} { r } { R _ { \mathcal { D } _ { i } } ( \psi ) : = \int \mathfrak { L } ( \psi ( z _ { i - 1 } ) , \Delta z _ { i } ) d \mathcal { D } _ { i } ( z _ { i - 1 } , \Delta z _ { i } ) } \end{array}$ and empirical risk $\begin{array} { r } { \widehat { R } _ { S _ { i } } ( \psi ) : = \frac { 1 } { n } \sum _ { ( z _ { i - 1 } , \Delta z _ { i } ) \in S _ { i } } \mathfrak { L } ( \psi ( z _ { i - 1 } ) , \Delta z _ { i } ) } \end{array}$ . We have:

$$
\begin{array} { r } { \mathbb { E } _ { \{ S _ { j } \sim \mathcal { D } _ { j } ^ { n } \} _ { j = 1 } ^ { T } } \left[ \operatorname* { s u p } _ { \psi \in \Psi } \left( \sum _ { i = 1 } ^ { T } ( R _ { \mathcal { D } _ { i } } ( \psi ) - \widehat { R } _ { S _ { i } } ( \psi ) ) \right) \right] \leq 2 T \mathbb { E } _ { \{ S _ { j } \sim \mathcal { D } _ { j } ^ { n } \} _ { j = 1 } ^ { T } } [ \widehat { \mathfrak { N } } _ { S _ { 0 } } ( \mathcal { F } ) ] , } \end{array}
$$

where $\textstyle { \mathcal { F } } = \mathfrak { L } \circ \Psi , S _ { 0 } = \bigcup _ { i = 1 } ^ { T } S _ { i } ,$ , and $\mathfrak { R } _ { S _ { 0 } } ( \mathcal { F } )$ is the empirical Rademacher complexity of $\ B = F$ on $S _ { 0 }$ We now prove Lemma 8.

Proof. Let $S _ { 1 } ^ { \prime } , S _ { 2 } ^ { \prime } , \ldots , S _ { T } ^ { \prime }$ be another set of i.i.d. samples, where $S _ { i } ^ { \prime } \sim \mathcal { D } _ { i } ^ { n }$ and

$$
S _ { i } ^ { \prime } = \{ ( x _ { i 1 } ^ { \prime } , y _ { i 1 } ^ { \prime } ) , ( x _ { i 2 } ^ { \prime } , y _ { i 2 } ^ { \prime } ) , \ldots , ( x _ { i n } ^ { \prime } , y _ { i n } ^ { \prime } ) \} \quad \mathrm { f o r } i = 1 , 2 , \ldots , T .
$$

Clearly, for all $\psi \in \Psi$ , the population risk is the expectation of the empirical risk over the ghost sample:

$$
R _ { \mathcal { D } _ { i } } ( \psi ) = \mathbb { E } _ { \{ S _ { j } ^ { \prime } \sim \mathcal { D } _ { j } ^ { n } \} _ { j = 1 } ^ { T } } [ \hat { R } _ { S _ { i } ^ { \prime } } ( \psi ) ] .
$$

Therefore, for every $\psi \in \Psi$ , we have

$$
R _ { \mathcal { D } _ { i } } ( \psi ) - \widehat { R } _ { S _ { i } } ( \psi ) = \mathbb { E } _ { \{ S _ { j } ^ { \prime } \sim \mathcal { D } _ { j } ^ { n } \} _ { j = 1 } ^ { T } } [ \widehat { R } _ { S _ { i } ^ { \prime } } ( \psi ) - \widehat { R } _ { S _ { i } } ( \psi ) ] .
$$

Taking the supremum over ψ $\in \Psi$ on both sides, and using the fact that the supremum of an expectation is upper bounded by the expectation of the supremum (Jensen’s inequality), we obtain

$$
\begin{array} { r l } & { \displaystyle \operatorname* { s u p } _ { \psi \in \Psi } \left( \displaystyle \sum _ { i = 1 } ^ { T } ( R _ { \mathcal D _ { i } } ( \psi ) - \widehat { R } _ { S _ { i } } ( \psi ) ) \right) } \\ & { \displaystyle = \operatorname* { s u p } _ { \psi \in \Psi } \mathbb { E } _ { \{ S _ { j } ^ { \prime } \sim \mathcal D _ { j } ^ { n } \} _ { j = 1 } ^ { T } } \left[ \displaystyle \sum _ { i = 1 } ^ { T } ( \hat { R } _ { S _ { i } ^ { \prime } } ( \psi ) - \widehat { R } _ { S _ { i } } ( \psi ) ) \right] } \\ & { \displaystyle \leq \mathbb { E } _ { \{ S _ { j } ^ { \prime } \sim \mathcal D _ { j } ^ { n } \} _ { j = 1 } ^ { T } } \left[ \displaystyle \operatorname* { s u p } _ { \psi \in \Psi } \left( \displaystyle \sum _ { i = 1 } ^ { T } ( \hat { R } _ { S _ { i } ^ { \prime } } ( \psi ) - \hat { R } _ { S _ { i } } ( \psi ) ) \right) \right] . } \end{array}
$$

Taking expectation over $\{ S _ { j } \sim \mathcal { D } _ { j } ^ { n } \} _ { j = 1 } ^ { T }$ on both sides yields

$$
\begin{array} { r l } & { \mathbb { E } _ { \{ S _ { j } \sim \mathcal { D } _ { j } ^ { n } \} _ { j = 1 } ^ { T } } \left[ \underset { \psi \in \Psi } { \operatorname* { s u p } } \left( \displaystyle \sum _ { i = 1 } ^ { T } ( R _ { \mathcal { D } _ { i } } ( \psi ) - \widehat { R } _ { S _ { i } } ( \psi ) ) \right) \right] } \\ & { \le \mathbb { E } _ { \{ S _ { j } \sim \mathcal { D } _ { j } ^ { n } \} _ { j = 1 } ^ { T } , \{ S _ { j } ^ { \prime } \sim \mathcal { D } _ { j } ^ { n } \} _ { j = 1 } ^ { T } } \left[ \underset { \psi \in \Psi } { \operatorname* { s u p } } \left( \displaystyle \sum _ { i = 1 } ^ { T } ( \widehat { R } _ { S _ { i } ^ { \prime } } ( \psi ) - \widehat { R } _ { S _ { i } } ( \psi ) ) \right) \right] } \\ & { = \frac 1 n \mathbb { E } _ { \{ S _ { j } \sim \mathcal { D } _ { j } ^ { n } \} _ { j = 1 } ^ { T } , \{ S _ { j } ^ { \prime } \sim \mathcal { D } _ { j } ^ { n } \} _ { j = 1 } ^ { T } } \left[ \underset { \psi \in \Psi } { \operatorname* { s u p } } \left( \displaystyle \sum _ { i = 1 } ^ { T } \sum _ { k = 1 } ^ { n } ( \mathfrak { L } ( \psi ( x _ { i k } ^ { \prime } ) , y _ { i k } ^ { \prime } ) - \mathfrak { L } ( \psi ( x _ { i k } ) , y _ { i k } ) ) \right) \right] . } \end{array}
$$

Next, we note that for each i and $k , ( x _ { i k } , y _ { i k } )$ and $( x _ { i k } ^ { \prime } , y _ { i k } ^ { \prime } )$ are i.i.d. variables. Therefore, we can replace them without affecting the expectation. For any specific pair $( i , k )$ , let the summation over the remaining indices be denoted by $\sum { } _ { ( i ^ { \prime } , k ^ { \prime } ) \neq ( i , k ) }$ . We have:

$$
\begin{array} { r l } & { \mathbb { E } _ { \{ S _ { j } \sim \mathcal D _ { j } ^ { n } \} _ { j = 1 } ^ { T } , \{ S _ { j } ^ { \prime } \sim \mathcal D _ { j } ^ { n } \} _ { j = 1 } ^ { T } } \Bigg [ \operatorname* { s u p } _ { \psi \in \Psi } \Bigg ( ( \mathbb { E } ( \psi ( x _ { i k } ^ { \prime } ) , y _ { i k } ^ { \prime } ) - \mathcal { L } ( \psi ( x _ { i k } ) , y _ { i k } ) ) } \\ & { \qquad + \sum _ { i \in \{ i , k ^ { \prime } \} \in \{ i , k \} } ( \mathcal { L } ( \psi ( x _ { i ^ { \prime } k ^ { \prime } } ^ { \prime } ) , y _ { i ^ { \prime } k ^ { \prime } } ^ { \prime \prime } ) - \mathcal { L } ( \psi ( x _ { i ^ { \prime } k ^ { \prime } } ) , y _ { i ^ { \prime } k ^ { \prime } } ) ) \Bigg ) \Bigg ] } \\ & { = \mathbb { E } _ { \{ S _ { j } \sim \mathcal D _ { j } ^ { n } \} _ { j = 1 } ^ { T } , \ \{ S _ { j } ^ { \prime } \sim \mathcal D _ { j } ^ { n } \} _ { j = 1 } ^ { T } } \Bigg [ \operatorname* { s u p } _ { \psi \in \Psi } \Bigg ( ( - \mathcal { L } ( \psi ( x _ { i k } ^ { \prime } ) , y _ { i k } ^ { \prime } ) + \mathcal { L } ( \psi ( x _ { i k } ) , y _ { i k } ) ) } \\ &  \qquad + \sum _ { i \in \{ i , k ^ { \prime } \} \neq i , k \} ( \mathcal { L } ( \psi ( x _ { i ^ { \prime } k ^ { \prime } k } ^ { \prime } ) , y _ { i ^ { \prime } k ^ { \prime } } ^ { \prime \prime } ) - \mathcal { L } ( \psi ( x _ { i ^ { \prime } k ^ { \prime } } ) , y _ { i ^ { \prime } k ^ { \prime } } ) ) \Bigg ) \Bigg ] . } \end{array}
$$

Let $\sigma _ { i k }$ be a random variable such that $\mathbb { P } [ \sigma _ { i k } = 1 ] = \mathbb { P } [ \sigma _ { i k } = - 1 ] = \frac { 1 } { 2 }$ . Combining the two equalities above, we obtain:

$$
\begin{array} { r l } & { \mathbb { E } _ { \{ S _ { j } \sim \mathcal { D } _ { j } ^ { n } \} _ { j = 1 } ^ { \mathcal { T } } } \{ S _ { j } ^ { \prime } \sim D _ { j } ^ { n } \} _ { j = 1 } ^ { \mathcal { T } } \Bigg [ \underset { \psi \in \Psi } { \operatorname* { s u p } } \Bigg ( \big ( \mathcal { E } \big ( \psi \big ( x _ { i k } ^ { \prime } \big ) , y _ { i k } ^ { \prime } \big ) - \mathcal { E } \big ( \psi ( x _ { i k } ) , y _ { i k } \big ) \big ) } \\ & { \qquad + \underset { \big ( i ^ { \prime } , k ^ { \prime } \big ) \not \in \binom { i , k } { i , k ^ { \prime } } } { \sum } \big ( \mathfrak { E } \big ( \psi \big ( x _ { i ^ { \prime } k ^ { \prime } } ^ { \prime } \big ) , y _ { i ^ { \prime } k ^ { \prime } } ^ { \prime } \big ) - \mathfrak { E } \big ( \psi \big ( x _ { i ^ { \prime } k ^ { \prime } k ^ { \prime } } \big ) , y _ { i ^ { \prime } k ^ { \prime } } \big ) \big ) \Bigg ) \Bigg ] } \\ & { = \mathbb { E } _ { \{ S _ { j } \sim \mathcal { D } _ { j } ^ { n } \} _ { j = 1 } ^ { \mathcal { T } } } \{ S _ { j } ^ { \prime } \sim D _ { j } ^ { n } \} _ { j = 1 } ^ { \mathcal { T } } \Bigg [ \mathbb { E } _ { \sigma _ { k _ { k } } } \underset { \psi \in \Psi } { \operatorname* { s u p } } \Bigg ( \sigma _ { i k } \big ( \mathcal { E } \big ( \psi \big ( x _ { i k } ^ { \prime } \big ) , y _ { i k } ^ { \prime } \big ) - \mathcal { E } \big ( \psi \big ( x _ { i k } \big ) , y _ { i k } \big ) \big ) } \\ &  \qquad + \underset { \big ( i ^ { \prime } , k ^ { \prime } \big ) \not \in \binom { i , k } { i , k ^ { \prime } } } { \sum } \big ( \mathcal { E } \big ( \psi \big ( x _  i ^ { \prime } k ^  \ \end{array}
$$

The inner expectation over $\sigma _ { i k }$ splits the right-hand side into the two half-terms carrying signs + and −; the swap symmetry of the i.i.d. pair $( \hat { x _ { i k } } , y _ { i k } )  ( x _ { i k } ^ { \prime } , y _ { i k } ^ { \prime } )$ used in the preceding display shows

that these two half-terms are equal, so the identity holds after averaging over $\sigma _ { i k }$ . Repeating this process for all $i = 1 , \dots , T$ and $k = 1 , \dots , n$ , we have

$$
\begin{array} { r l } & { \mathbb { E } _ { \{ S _ { j } \sim \mathcal { D } _ { j } ^ { n } \} _ { j = 1 } ^ { T } , \{ S _ { j } ^ { \prime } \sim \mathcal { D } _ { j } ^ { n } \} _ { j = 1 } ^ { T } } \Bigg [ \underset { \psi \in \Psi } { \operatorname* { s u p } } \left( \displaystyle \sum _ { i = 1 } ^ { T } \sum _ { k = 1 } ^ { n } ( \mathfrak { L } ( \psi ( x _ { i k } ^ { \prime } ) , y _ { i k } ^ { \prime } ) - \mathfrak { L } ( \psi ( x _ { i k } ) , y _ { i k } ) ) \right) \Bigg ] } \\ & { = \mathbb { E } _ { \{ S _ { j } \sim \mathcal { D } _ { j } ^ { n } \} _ { j = 1 } ^ { T } , \ \{ S _ { j } ^ { \prime } \sim \mathcal { D } _ { j } ^ { n } \} _ { j = 1 } ^ { T } , } \Bigg [ \underset { \psi \in \Psi } { \operatorname* { s u p } } \left( \displaystyle \sum _ { i = 1 } ^ { T } \sum _ { k = 1 } ^ { n } \sigma _ { i k } ( \mathfrak { L } ( \psi ( x _ { i k } ^ { \prime } ) , y _ { i k } ^ { \prime } ) - \mathfrak { L } ( \psi ( x _ { i k } ) , y _ { i k } ) ) \right) \Bigg ] . } \end{array}
$$

Finally, applying the triangle inequality to the supremum:

$$
\begin{array} { r l } & { \displaystyle \operatorname* { s u p } _ { \psi \in \Psi } \displaystyle \sum _ { i = 1 } ^ { T } \sum _ { k = 1 } ^ { n } \sigma _ { i k } \big ( \mathfrak { L } \big ( \psi \big ( x _ { i k } ^ { \prime } \big ) , y _ { i k } ^ { \prime } \big ) - \mathfrak { L } \big ( \psi \big ( x _ { i k } \big ) , y _ { i k } \big ) \big ) } \\ & { \leq \displaystyle \operatorname* { s u p } _ { \psi \in \Psi } \displaystyle \sum _ { i = 1 } ^ { T } \sum _ { k = 1 } ^ { n } \sigma _ { i k } \mathfrak { L } \big ( \psi \big ( x _ { i k } ^ { \prime } \big ) , y _ { i k } ^ { \prime } \big ) } \\ & { \quad + \displaystyle \operatorname* { s u p } _ { \psi \in \Psi } \displaystyle \sum _ { i = 1 } ^ { T } \sum _ { k = 1 } ^ { n } - \sigma _ { i k } \mathfrak { L } \big ( \psi \big ( x _ { i k } \big ) , y _ { i k } \big ) . } \end{array}
$$

The joint distribution of the Rademacher signs is invariant under sign reversal. Taking expectations in the preceding inequality and using this symmetry for the second term, we get

$$
\begin{array} { r l } & { \mathbb { E } _ { \{ S _ { j } \sim D _ { j } ^ { n } \} _ { j = 1 } ^ { T } \{ S _ { j } ^ { \prime } \sim D _ { j } ^ { n } \} _ { j = 1 } ^ { T } , \{ \{ \{ \sigma _ { i k } \} _ { i = 1 } ^ { T } \} _ { k = 1 } ^ { T } \} _ { k = 1 } ^ { n } } \left[ \underset { \psi \in \Psi } { \operatorname* { s u p } } \displaystyle \sum _ { i = 1 } ^ { T } \sum _ { k = 1 } ^ { n } \sigma _ { i k } ( \mathfrak { L } ( \psi ( x _ { i k } ^ { \prime } ) , y _ { i k } ^ { \prime } ) - \mathfrak { L } ( \psi ( x _ { i k } ) , y _ { i k } ) ) \right] } \\ &  \le \mathbb { E } _ { \{ S _ { j } \sim D _ { j } ^ { n } \} _ { j = 1 } ^ { T } \{ S _ { j } ^ { \prime } \sim D _ { j } ^ { n } \} _ { j = 1 } ^ { T } , \{ \{ \sigma _ { i k } \} _ { i = 1 } ^ { T } \} _ { k = 1 } ^ { n } \Big [ \underset { \psi \in \Psi } { \operatorname* { s u p } } \displaystyle \sum _ { i = 1 } ^ { T } \sum _ { k = 1 } ^ { n } \sigma _ { i k } \mathfrak { L } ( \psi ( x _ { i k } ^ { \prime } ) , y _ { i k } ^ { \prime } ) \Big ] } \\ &  \quad + \mathbb { E } _ { \{ S _ { j } \sim D _ { j } ^ { n } \} _ { j = 1 } ^ { T } , \{ S _ { j } ^ { \prime } \sim D _ { j } ^ { n } \} _ { j = 1 } ^ { T } , \{ \{ \sigma _ { i k } \} _ { i = 1 } ^ { T } \} _ { k = 1 } ^ { n } \Big [ \underset { \psi \in \Psi } { \operatorname* { s u p } } \displaystyle \sum _ { i = 1 } ^ { T } \sum _ { k = 1 } ^ { n } \sigma _ { i k } \mathfrak { L } ( \psi ( x _ { i k } ) , y _ { i k } ) \Big ] } \\ &  = 2 \cdot \mathbb { E } _  \{ S _ { j } \sim D _ { j } ^ \end{array}
$$

Substituting this result into the previous equation involving the $1 / n$ factor, we obtain the final bound:

$$
\begin{array} { r l } & { \mathbb { E } _ { \{ S _ { j } \sim \mathcal { D } _ { j } ^ { n } \} _ { j = 1 } ^ { T } } \left[ \displaystyle \operatorname* { s u p } _ { \psi \in \Psi } \left( \displaystyle \sum _ { i = 1 } ^ { T } ( R _ { \mathcal { D } _ { i } } ( \psi ) - \hat { R } _ { S _ { i } } ( \psi ) ) \right) \right] } \\ & { \quad \le \displaystyle \frac { 1 } { n } \cdot 2 n T \mathbb { E } _ { \{ S _ { j } \sim \mathcal { D } _ { j } ^ { n } \} _ { j = 1 } ^ { T } } [ \widehat \Re _ { S _ { 0 } } ( \mathcal { F } ) ] } \\ & { \quad = 2 T \mathbb { E } _ { \{ S _ { j } \sim \mathcal { D } _ { j } ^ { n } \} _ { j = 1 } ^ { T } } [ \widehat \Re _ { S _ { 0 } } ( \mathcal { F } ) ] . } \end{array}
$$

Lemma 9 (McDiarmid’s Inequality[19]). Let V be a domain, and let $f : V ^ { n } \to \mathbb { R }$ be afunction ofn variables satisfyingfor some $c > 0 ,$ , for all $i \in [ n ]$ and all $x _ { 1 } , \ldots , x _ { n } , x _ { i } ^ { \prime } \in V { \mathrm { : } }$

$$
| f ( x _ { 1 } , \dots , x _ { i } , \dots , x _ { n } ) - f ( x _ { 1 } , \dots , x _ { i } ^ { \prime } , \dots , x _ { n } ) | \leq c .
$$

Let $X _ { 1 } , X _ { 2 } , \ldots , X _ { n }$ be independent random variables taking values in V. Then with probability at least $1 - \delta \colon$

$$
{ \Big | } f ( X _ { 1 } , \ldots , X _ { n } ) - \operatorname { \mathbb { E } } { \big [ } f ( X _ { 1 } , \ldots , X _ { n } ) { \big ] } { \Big | } \leq c { \sqrt { { \frac { n } { 2 } } \ln \left( { \frac { 2 } { \delta } } \right) } } .
$$

Now we give the proof of Theorem 2.

Theorem 2 (Rademacher Complexity Bound). 1. Under the preceding conditions (where the step-level loss is bounded by $B _ { \ell } )$ , with probability at least $1 - \delta \colon$

$$
\sum _ { i = 1 } ^ { T } ( R _ { \mathcal { D } _ { i } } ( \psi ) - \widehat { R } _ { S _ { i } } ( \psi ) ) \leq 2 T \mathbb { E } _ { \{ S _ { j } \sim \mathcal { D } _ { j } ^ { n } \} _ { j = 1 } ^ { T } } [ \widehat { \mathfrak { R } } _ { S _ { 0 } } ( \mathcal { F } ) ] + B _ { \ell } \sqrt { \frac { 2 T \log \frac { 2 } { \delta } } { n } } .
$$

2. Under the preceding conditions (where the step-level loss is bounded by $B _ { \ell } )$ , with probability at least $1 - \delta \colon$

$$
\left| \sum _ { i = 1 } ^ { T } ( R _ { \mathcal { D } _ { i } } ( \psi ) - \widehat { R } _ { S _ { i } } ( \psi ) ) \right| \leq 2 T \widehat { \mathfrak { R } } _ { S _ { 0 } } ( \mathcal { F } ) + 3 B _ { \ell } \sqrt { \frac { 2 T \log \frac { 4 } { \delta } } { n } } .
$$

Proof. We first prove the first inequality. Since $\begin{array} { r l } { \sum _ { i = 1 } ^ { T } ( R _ { \mathcal { D } _ { i } } ( \psi ) - \widehat { R } _ { S _ { i } } ( \psi ) ) } & { { } } \end{array}$ ) satisfies the conditions of Lemma 9 with Tn variables and constant $2 B _ { \ell } / n$ , applying Lemma 9 and then Lemma 8 yields that with probability at least $1 - \delta \colon$

$$
\begin{array} { r l } & { \quad \displaystyle \sum _ { i = 1 } ^ { T } ( R _ { D _ { i } } ( \psi ) - \widehat { R } _ { S _ { i } } ( \psi ) ) } \\ & { \le \operatorname* { s u p } \left( \displaystyle \sum _ { i = 1 } ^ { T } ( R _ { D _ { i } } ( \psi ) - \widehat { R } _ { S _ { i } } ( \psi ) ) \right) } \\ & { \le \mathbb { E } _ { \{ S _ { i } > D _ { j } ^ { n } \} _ { j = 1 } ^ { T } } \left[ \displaystyle \operatorname* { s u p } _ { \psi \in \Psi } \left( \displaystyle \sum _ { i = 1 } ^ { T } ( R _ { D _ { i } } ( \psi ) - \widehat { R } _ { S _ { i } } ( \psi ) ) \right) \right] + B _ { \ell } \sqrt { \displaystyle \frac { 2 T \log \frac { 2 } { \delta } } { n } } } \\ & { \le 2 T \mathbb { E } _ { \{ S _ { i } > D _ { j } ^ { n } \} _ { j = 1 } ^ { T } } [ \widehat { \mathfrak { N } } _ { S _ { 0 } } ( \psi ) ] + B _ { \ell } \sqrt { \displaystyle \frac { 2 T \log \frac { 2 } { \delta } } { n } } . } \end{array}
$$

For the second inequality, based on the first one, since $\mathfrak { R } _ { S _ { 0 } } ( \mathcal { F } )$ also satisfies the conditions of Lemma 9 (with Tn variables, constant $2 B _ { \ell } / ( T n ) $ ), with probability at least $1 - 2 \delta$ we have:

$$
\begin{array} { r l } & { \quad \displaystyle \sum _ { i = 1 } ^ { T } ( R _ { \mathcal D _ { i } } ( \psi ) - \widehat R _ { S _ { i } } ( \psi ) ) } \\ & { \le 2 T \mathbb { E } _ { \{ S _ { j } > D _ { j } ^ { n } \} _ { j = 1 } ^ { T } } [ \widehat \Omega _ { S _ { 0 } } ( F ) ] + B _ { \ell } \sqrt { \frac { 2 T \log \frac 2 \delta } { n } } } \\ & { \le 2 T \widehat \mathfrak R _ { S _ { 0 } } ( F ) + 2 T \left( \frac { 2 B _ { \ell } } { n T } \sqrt { \frac { n T \log \frac 2 \delta } { 2 } } \right) + B _ { \ell } \sqrt { \frac { 2 T \log \frac 2 \delta } { n } } } \\ & { - 2 T \widehat \mathfrak R _ { S _ { 0 } } ( F ) + 3 B _ { \ell } \sqrt { \frac { 2 T \log \frac 2 \delta } { n } } . } \end{array}
$$

The reverse deviation follows by the same sign-symmetry argument. Using $\delta / 2$ in place of $\delta$ converts the $1 - 2 \delta$ event and the log $( 2 / \bar { \delta } )$ factors above into the $1 - \delta$ event and the $\log ( 4 / \delta )$ factors of part 2, without changing the coefficients. □

## B.5 Proof of Corollary 3

Proof. Let $U : = 2 T \widehat { \Re } _ { S _ { 0 } } ( \mathcal { F } ) + 3 B _ { \ell } \sqrt { 2 T \log ( 4 / \delta ) / n } ,$ , so that Theorem 2 bounds $\begin{array} { r l } { \mathrm { ~ } } & { { } \vert \sum _ { i } ( R _ { D _ { i } } ( \psi ) - } \end{array}$ $\widehat { R } _ { S _ { i } } ( \psi ) ) |$ by U uniformly over $\psi _ { : }$ , on an event of probability at least $1 - \delta .$

First display. Theorem 1 gives $\begin{array} { r } { R _ { \mathcal { D } } ( F _ { \psi } ) \le \frac { B _ { \ell } k } { \log 2 } \sum _ { i } R _ { \mathcal { D } _ { i } } ( \psi ) } \end{array}$ ; replacing the population risks by their empirical counterparts costs at most $U ,$ and multiplying by $B _ { \ell } k / \log 2$ gives the first display.

Second display. Let ψb be the sub-task ERM and $\psi ^ { \star }$ a population minimizer. Empirical optimality gives $\begin{array} { r } { \sum _ { i } \widehat { R } _ { S _ { i } } ( \widehat { \psi } ) \leq \sum _ { i } \widehat { R } _ { S _ { i } } ( \psi ^ { \star } ) } \end{array}$ , so the atomic excess of $\widehat { \psi }$ obeys

$$
\begin{array} { r l } & { \displaystyle \sum _ { i } R _ { \mathcal D _ { i } } ( \widehat \psi ) - \sum _ { i } R _ { \mathcal D _ { i } } ( \psi ^ { \star } ) = \underbrace { \sum _ { i } \big ( R _ { \mathcal D _ { i } } ( \widehat \psi ) - \widehat R _ { S _ { i } } ( \widehat \psi ) \big ) } _ { \le U } + \underbrace { \sum _ { i } \big ( \widehat R _ { S _ { i } } ( \widehat \psi ) - \widehat R _ { S _ { i } } ( \psi ^ { \star } ) \big ) } _ { \le 0 } } \\ & { \quad \ + \underbrace { \sum _ { i } \big ( \widehat R _ { S _ { i } } ( \psi ^ { \star } ) - R _ { \mathcal D _ { i } } ( \psi ^ { \star } ) \big ) } _ { \le U } \le 2 U . } \end{array}
$$

Substituting $\begin{array} { r } { \sum _ { i } R _ { { \mathcal { D } } _ { i } } ( \widehat \psi ) \leq \sum _ { i } R _ { { \mathcal { D } } _ { i } } ( \psi ^ { \star } ) + 2 U } \end{array}$ into the bound of Theorem 1 gives the second display. □

## B.6 Proof of Corollary 4

Proof. By the oracle-prefix reduction of Theorem 1 the final-task risk is bounded by $\begin{array} { r } { \frac { B \ell k } { \log 2 } \sum _ { i = 1 } ^ { T } R _ { D _ { i } } ( \psi ) } \end{array}$ ; transferring each term via $R _ { \mathcal { D } _ { i } } ( \psi ) \leq R _ { Q _ { i } } ( \psi ) + \mathrm { d i s c } _ { \mathcal { F } } ( \mathcal { D } _ { i } , Q _ { i } )$ and applying Theorem 2 to the $Q _ { i }$ proves the first display. The second follows from the same comparison with the deviation paid once for each of $\widehat { \psi }$ and $\psi ^ { \star }$ , and the transfer from $Q _ { i }$ to $\mathcal { D } _ { i }$ likewise paid twice. □

## B.7 Teacher-Forced NTP Reduction

Lemma 10 (Teacher-Forced NTP Reduction). Under Assumption 1, consider the NTP setting of Section $4 . l ,$ , with deterministic argmax decoding and the same context-window rulefor teacher-forced and free-running prefixes. Then

$$
R _ { \mathcal { D } } ( F _ { \psi _ { g , h } } ) \leq \frac { B _ { \ell } k } { \log 2 } \sum _ { i = 1 } ^ { T } R _ { \mathcal { D } _ { i } } ^ { k } ( \psi _ { g , h } ) .
$$

The risks on the right are the teacher-forced target-token risks defined in $E q . \ ( 1 )$ , whereas the risk on the left is evaluated on the free-running final prediction.

Proof. We follow the first-error argument of Theorem 1, now locating the first erroneous token within the first erroneous block. Fix $g , h ,$ , and let $\psi : = \psi _ { g , h }$ be the block predictor given by k successive free-running next-token predictions via $g \circ h$ . As in Appendix B.3, let $\mathbf { z } = ( z _ { 0 } , \mathsf { \bar { \Delta } } z _ { 1 } , \mathsf { \bar { \Delta } } . \mathsf { \bar { \Delta } } , \Delta z _ { T } ) \sim \mathcal { D } ^ { \prime }$ with joint law $p ( \mathbf { z } )$ , and write

$$
\widehat { z } _ { 0 } = z _ { 0 } , \qquad \widehat { \Delta z } _ { i } = \mathrm { h a r d } ( \psi ( \widehat { z } _ { i - 1 } ) ) , \qquad \widehat { z } _ { i } = \bar { \psi } ( \widehat { z } _ { i - 1 } ) .
$$

For $r = 1 , \ldots , k .$ , let $X _ { i , < r }$ be the oracle context before token $( \Delta z _ { i } ) ,$ <sub>r</sub>, formed from $z _ { i - 1 }$ and the true within-step prefix $\left( \Delta z _ { i } \right) _ { < r }$ . Let $\widehat { X } _ { i , < r }$ be its free-running counterpart, formed from $\widehat { z } _ { i - 1 }$ and $\widehat { \left( \Delta z _ { i } \right) } _ { < r }$ . Both contexts use the same context-window rule, including truncation. For each oracle block, abbreviate its teacher-forced target-token loss by

$$
q _ { i } : = \frac { 1 } { k } \sum _ { r = 1 } ^ { k } \ell ( g ( h ( X _ { i , < r } ) ) , ( \Delta z _ { i } ) _ { r } ) .
$$

This is Eq. (1) evaluated on the oracle record for step $i ;$ consequently, $\begin{array} { r } { \int q _ { i } d p ( { \mathbf { z } } ) = R _ { \mathcal { D } _ { i } } ^ { k } ( \psi _ { g , h } ) } \end{array}$ Define, as in the proof of Theorem 1,

$$
u _ { i } : = \mathbb { 1 } \big \{ \widehat { \Delta z } _ { i } \neq \Delta z _ { i } \big \} , \qquad I : = \operatorname* { i n f } \{ i \in \{ 1 , \dots , T \} : u _ { i } = 1 \} ,
$$

with $I = \infty$ if no error occurs. The same induction gives $\widehat { z } _ { i - 1 } = z _ { i - 1 }$ on $\{ I \geq i \}$ . On $\{ I = i \}$ , let r be the first position in block i with $( \widehat { \Delta z } _ { i } ) _ { r } \neq ( \Delta z _ { i } ) _ { i }$ <sub>r</sub>. Then

$$
\widehat { z } _ { i - 1 } = z _ { i - 1 } , \qquad ( \widehat { \Delta z } _ { i } ) _ { < r } = ( \Delta z _ { i } ) _ { < r } , \qquad \widehat { X } _ { i , < r } = X _ { i , < r } .
$$

Thus the erroneous hard prediction at this position is the argmax of the same probability vector $g ( h ( X _ { i , < r } ) )$ that appears in the teacher-forced loss. The probability assigned to the true token is therefore at most 1/2. By the clipped cross-entropy calculation in Lemma $\bar { 7 }$

$$
\ell ( g ( h ( X _ { i , < r } ) ) , ( \Delta z _ { i } ) _ { r } ) \geq \log 2 , \qquad q _ { i } \geq \frac { \log 2 } { k } \quad \mathrm { o n } ~ \{ I = i \} .
$$

This argument includes $i = T ,$ , so it also accounts for errors within the final output block.

Next, split the final-step loss according to whether any hard error occurs. On $\{ I > T \}$ , all generated tokens agree with their oracle counterparts, including every prefix inside the final block, and hence

$$
\mathfrak { L } \big ( \psi ( \widehat { z } _ { T - 1 } ) , \Delta z _ { T } \big ) = q _ { T } .
$$

On $\{ I = i \}$ , boundedness gives $\mathfrak { L } ( \psi ( \widehat { z } _ { T - 1 } ) , \Delta z _ { T } ) \le B _ { \ell }$ . Since $B _ { \ell } k / \log 2 \geq 1$ and the events $\{ I \stackrel { \setminus } { > } T \} , \{ \stackrel { } { I } = 1 \} , . . . , \{ I \stackrel {  } { = } T \}$ form a partition, pointwise we obtain

$$
\begin{array} { r l } {  { \mathfrak { L } \big ( \psi ( \widehat { z } _ { T - 1 } ) , \Delta z _ { T } \big ) \leq \mathbb { 1 } \{ I > T \} q _ { T } + B _ { \ell } \sum _ { i = 1 } ^ { T } \mathbb { 1 } \{ I = i \} } \ }  \\ & { \leq \frac { B _ { \ell } k } { \log 2 } [ \mathbb { 1 } \{ I > T \} q _ { T } + \displaystyle \sum _ { i = 1 } ^ { T } \mathbb { 1 } \{ I = i \} q _ { i } ] } \\ & { \leq \frac { B _ { \ell } k } { \log 2 } \displaystyle \sum _ { i = 1 } ^ { T } q _ { i } . } \end{array}
$$

Finally, integrating against $p ( \mathbf { z } )$ and using the oracle step marginals $\mathcal { D } _ { i }$ yields

$$
R _ { \mathcal { D } } ( F _ { \psi _ { g , h } } ) \leq \frac { B _ { \ell } k } { \log 2 } \sum _ { i = 1 } ^ { T } \int q _ { i } d p ( { \mathbf { z } } ) = \frac { B _ { \ell } k } { \log 2 } \sum _ { i = 1 } ^ { T } R _ { \mathcal { D } _ { i } } ^ { k } ( \psi _ { g , h } ) ,
$$

which proves the claim. The argument uses equality of the two prefixes only before the first hard error; it does not identify the teacher-forced and free-running block losses after that error. □

## B.8 Proof of Theorem 5

Proof. Apply Theorem 2 to the bounded sequence-level class $\mathcal { Q } _ { k }$ and the samples $S _ { i } \sim \mathcal { D } _ { i } ^ { n }$ . Uniformly over $g \in { \mathcal { G } }$ and $h \in \mathcal H$

$$
\left| \sum _ { i = 1 } ^ { T } \left( R _ { \mathcal { D } _ { i } } ^ { k } ( \psi _ { g , h } ) - \widehat { R } _ { S _ { i } } ^ { k } ( \psi _ { g , h } ) \right) \right| \leq 2 T \widehat { \mathfrak { N } } _ { S _ { 0 } } ( \mathcal { Q } _ { k } ) + 3 B _ { \ell } \sqrt { \frac { 2 T \log ( 4 / \delta ) } { n } } ,
$$

which is Eq. (2).

For the ERM comparison, the three-term decomposition of the proof of Corollary $3 ( \mathrm { A }$ ppendix B.5) applies verbatim with $\mathcal { F }$ replaced by $\mathcal { Q } _ { k }$ and $R _ { D _ { i } }$ by $R _ { \mathcal { D } _ { i } } ^ { k }$ : writing $\begin{array} { r } { \mathcal { E } _ { k } ( \widehat { \psi } ) : = \sum _ { i } R _ { D _ { i } } ^ { k } ( \widehat { \psi } ) - } \end{array}$ $\begin{array} { r } { \sum _ { i } R _ { D _ { i } } ^ { k } ( \psi _ { g ^ { \star } , h ^ { \star } } ) } \end{array}$ for the target-token atomic excess and $\begin{array} { r } { U _ { k } : = 2 T \widehat { \mathfrak { R } } _ { S _ { 0 } } ( \mathcal { Q } _ { k } ) + 3 B _ { \ell } \sqrt { 2 T \log ( 4 / \delta ) / n } } \end{array}$ empirical optimality and the two directions of the same uniform event give $\mathcal { E } _ { k } ( \widehat { \psi } ) \leq U _ { k } + 0 + U _ { k }$ Lemma 10 then gives

$$
R _ { \mathcal { D } } ( F _ { \psi _ { \hat { g } , \hat { h } } } ) \leq \frac { B _ { \ell } k } { \log 2 } \sum _ { i = 1 } ^ { T } R _ { \mathcal { D } _ { i } } ^ { k } ( \widehat { \psi } ) = \frac { B _ { \ell } k } { \log 2 } \left[ \sum _ { i = 1 } ^ { T } R _ { \mathcal { D } _ { i } } ^ { k } ( \psi _ { g ^ { \star } , h ^ { \star } } ) + \mathcal { E } _ { k } ( \widehat { \psi } ) \right] ,
$$

which is Eq. (3) after substituting $U _ { k } + 0 + U _ { k } = 4 T \widehat { \Re } _ { S _ { 0 } } ( \mathcal { Q } _ { k } ) + 6 B _ { \ell } \sqrt { 2 T \log ( 4 / \delta ) / n }$ . The projection and loss-cover argument in Appendix A.2 gives Eq. (4). □

## B.9 Tightness of the Additive Reduction

Proposition 11 (Tightness of the Additive Reduction). For every $T \geq 1$ and every $p \in [ 0 , 1 / T ]$ there is a task with the following complete description. Let $\gamma = \{ 0 , 1 \} , k = 1 , m = T + 1$ , and draw $u _ { 1 } , \ldots , u _ { T }$ i.i.d. Bernoulli(p). The oracle trajectory is $z _ { 0 } = ( u _ { 1 } , \dots , u _ { T } , 0 )$ with $\Delta z _ { i } = 0 f o r$ every $i ,$ so that the oracle state before step i is $z _ { i - 1 } = ( u _ { i } , \dots , u _ { T } , 0 , \dots , 0 )$ , whose first token is $u _ { i }$ and whose last token is 0, and the step marginal $\mathcal { D } _ { i }$ is the law of $( z _ { i - 1 } , 0 )$ induced by $( u _ { i } , \dots , u _ { T } )$ . The shared single-step predictor is the deterministic rule hard $( \psi ( z ) ) = \operatorname* { m a x } \{ ( z ) _ { 1 } , ( z ) _ { m } \}$ , with $\psi ( z )$ putting probability 1 on that token, so that the clipped cross-entropy ofAssumption 1 charges 0 when the output is correct and $B _ { \ell } = \log ( 1 / \epsilon _ { \ell } )$ otherwise. Writing $e _ { 0 } = 0$ and $e _ { i } = e _ { i - 1 } \vee u _ { i }$ for the error flag (∨ is logical OR, and a local error puts the state in an absorbing error mode), this task satisfies

$$
R _ { \mathcal { D } } ( F _ { \psi } ) = B _ { \ell } \left[ 1 - ( 1 - p ) ^ { T } \right] = \Omega \left( \sum _ { i = 1 } ^ { T } R _ { \mathcal { D } _ { i } } ( \psi ) \right) , \qquad \sum _ { i = 1 } ^ { T } R _ { \mathcal { D } _ { i } } ( \psi ) = T p B _ { \ell } .
$$

Proof. Verify the construction of the statement. On an oracle state the last token is 0, so the hard output equals $u _ { i }$ and the error event is $u _ { i } = 1 $ ; hence $R _ { D _ { i } } ( \psi ) = p B _ { \ell }$ . Once an error occurs, its output 1 becomes the last token of every later state and the rule keeps emitting 1, so the final output is wrong exactly when $\textstyle \bigvee _ { i = 1 } ^ { T } u _ { i } = 1$ , which gives

$$
R _ { \mathcal { D } } ( F _ { \psi } ) = B _ { \ell } \operatorname* { P r } \left( \bigvee _ { i = 1 } ^ { T } u _ { i } = 1 \right) = B _ { \ell } \left[ 1 - ( 1 - p ) ^ { T } \right] .
$$

The union bound gives $1 - ( 1 - p ) ^ { T } \leq T p$ , and for $\begin{array} { r } { T p \leq 1 , 1 - ( 1 - p ) ^ { T } \geq T p - \binom { T } { 2 } p ^ { 2 } \geq \frac { 1 } { 2 } T p . } \end{array}$ Consequently

$$
\frac { 1 } { 2 } T p B _ { \ell } \leq R _ { \mathcal { D } } ( F _ { \psi } ) \leq T p B _ { \ell } = \sum _ { i = 1 } ^ { T } R _ { \mathcal { D } _ { i } } ( \psi ) ,
$$

hence $\begin{array} { r } { R _ { \mathcal { D } } ( F _ { \psi } ) = \Omega \big ( \sum _ { i = 1 } ^ { T } R _ { \mathcal { D } _ { i } } ( \psi ) \big ) } \end{array}$ , which proves the order-tightness of the additive reduction. □

What the construction shows. The witness combines two features: the local error events are independent across steps, and the state carries an absorbing error flag, so that the final output is wrong exactly when at least one local error has occurred. That event has probability $1 - ( 1 - p ) ^ { \mathbf { \hat { T } } } = \Theta ( T p )$ for $T p \ \leq \ 1$ , of the same order as the sum of the $T$ step risks, while the per-step risk itself is $R _ { \mathcal { D } _ { i } } ( \psi ) = p B _ { \ell }$ . The additive form of Theorem 1 is therefore tight up to its depth-independent coefficient, and the linear-in-T growth of the bound—here the genuine linear growth of the target risk—cannot be removed by any refinement of the argument.

## B.10 Tightness of the Atomic Excess

Proposition 12 (Tightness of the Atomic Excess). Fix $T = 4 r + 1$ with $r \in \mathbb { N } , r \geq 1$ , a sign $\theta \in \mathsf { \bar { \{ - 1 , + 1 \} } }$ , and $\bar { \gamma } \in \left( 0 , 1 / ( 2 T ) \right]$ ; the sign θ only selects which ofthe two hypotheses isfavoured, and the constants below do not depend on it. Let ${ \dot { \mathcal { V } } } = \{ 0 , 1 \} , k = { \dot { 1 } } , m = T { \ddot { + } } 1$ , and let $P _ { \theta }$ be the law on $\{ 0 , 1 \} ^ { T }$ that assigns $w e i g h t \ \frac { 1 } { \gamma T } + \theta \gamma$ to each vector $u ^ { ( j ) } \in \{ 0 , 1 \} ^ { T } , j \in [ T ]$ , whose support is the cyclic block $\{ j , j + 1 , \ldots , \bar { j } + \bar { 2 } r \} o f { 2 r + 1 }$ consecutive coordinates (indices modulo $\bar { T ) }$ , and $w e i g h t \ \frac { 1 } { 2 T } - \theta \gamma \ t o \ v ^ { ( j ) }$ <sup>)</sup>, the same block with its leading coordinate j flipped, so that $v ^ { ( j ) }$ has ones exactly on $\{ j + 1 , \ldots , j + 2 r \}$ ; these 2T weights sum to one. For $\begin{array} { r } { X \sim P _ { \theta } , P _ { \theta } ( X _ { i } = 1 ) = \frac { 1 } { 2 } + \theta \gamma } \end{array}$ and $\begin{array} { r } { P _ { \theta } \big ( \bigoplus _ { i = 1 } ^ { T } X _ { i } = 1 \big ) = \frac { 1 } { 2 } + \theta T \gamma } \end{array}$ , where L denotes bitwise XOR in this construction (not sequence concatenation). The oracle trajectory is $z _ { 0 } = ( X _ { 1 } , \dots , X _ { T } , 0 )$ with $\Delta z _ { i } = p _ { i }$ and $p _ { i } = p _ { i - 1 } \oplus X _ { i }$ $p _ { 0 } = 0 ,$ , so that the state before step i exposes $X _ { i }$ as its first token and $p _ { i - 1 }$ as its last token, and $\mathcal { D } _ { i } ^ { \theta }$ is the law of the i-th oracle state and update. The two-element class is $\Psi = \{ \psi _ { 0 } , \psi _ { 1 } \}$ , with $\psi _ { s } ( z )$ putting probability 3/4 on token $( z ) _ { m } \oplus s$ and $1 / 4$ on the other token, so that a correct output costs $\bar { \ell _ { \mathrm { o k } } } = \log ( \bar { 4 } / 3 )$ and a wrong one $\ell _ { \mathrm { e r r } } = \log 4$ (clipping is inactive since $\epsilon _ { \ell } < 1 / 4 )$ With n i.i.d. draws from each $\mathcal { D } _ { i } ^ { \theta } { } _ { : }$ , the sub-task ERM $\begin{array} { r } { \widehat { \psi } = \arg \operatorname* { m i n } _ { \psi \in \Psi } \sum _ { i = 1 } ^ { T } \widehat { R } _ { S _ { i } } ( \psi ) } \end{array}$ satisfies, for $\gamma = c _ { 0 }$ min $\{ 1 / \sqrt { n T } , 1 / T \}$ with a universal constant $c _ { 0 } \leq 1 / 2$

$$
\mathbb { E } _ { S } \left[ \sum _ { i = 1 } ^ { T } R _ { \mathcal { D } _ { i } } ^ { \theta } ( \widehat { \psi } ) - \operatorname* { m i n } _ { \psi \in \Psi } \Big \{ \sum _ { i = 1 } ^ { T } R _ { \mathcal { D } _ { i } } ^ { \theta } ( \psi ) \Big \} \right] = \Omega \left( \operatorname* { m i n } \left\{ 1 , \sqrt { \frac { T } { n } } \right\} \right) .
$$

In this construction the atomic excess coincides with thefinal-task excess, so the same bound holds $\begin{array} { r } { f o r \mathbb { E } _ { S } [ R _ { \mathcal { D } } ^ { \theta } ( F _ { \widehat { \psi } } ) - \operatorname* { m i n } _ { \psi \in \Psi } R _ { \mathcal { D } } ^ { \theta } ( F _ { \psi } ) ] } \end{array}$ , and it is $\Omega ( { \sqrt { T / n } } )$ when $T \leq n$

Proof. Verify the construction of the statement. The weights of $P _ { \theta }$ are nonnegative and sum to one; each $u ^ { ( j ) }$ has odd parity and each $v ^ { ( j ) }$ even parity, and each coordinate lies in exactly $2 r + 1$ of the $u ^ { ( j ) }$ and $2 r$ of the $v ^ { ( j ) }$ , so that for every coordinate i

$$
P _ { \theta } ( X _ { i } = 1 ) = { \frac { 1 } { 2 } } + \theta \gamma , \qquad P _ { \theta } \left( \bigoplus _ { i = 1 } ^ { T } X _ { i } = 1 \right) = { \frac { 1 } { 2 } } + \theta T \gamma .\tag{6}
$$

The oracle performs serial XOR, so the final target is $p _ { T } = \bigoplus _ { i = 1 } ^ { T } X _ { i }$ . At atomic step $i , \psi _ { \varepsilon }$ <sub>s</sub> incurs loss $\ell _ { \mathrm { o k } }$ when $X _ { i } = s$ and $\ell _ { \mathrm { e r r } }$ otherwise; hence

$$
R _ { \mathcal { D } _ { i } } ^ { \theta } ( \psi _ { s } ) = \ell _ { \mathrm { o k } } + \Delta _ { \ell } P _ { \theta } ( X _ { i } \neq s ) , \qquad \Delta _ { \ell } = \ell _ { \mathrm { e r r } } - \ell _ { \mathrm { o k } } = \log 3 .
$$

Writing $s _ { \theta } = ( 1 + \theta ) / 2 ,$ , Eq. (6) gives

$$
\sum _ { i = 1 } ^ { T } \left[ R _ { \mathcal { D } _ { i } } ^ { \theta } ( \psi _ { 1 - s _ { \theta } } ) - R _ { \mathcal { D } _ { i } } ^ { \theta } ( \psi _ { s _ { \theta } } ) \right] = 2 \Delta _ { \ell } T \gamma .\tag{7}
$$

The learned rollout uses the same s at every step. Since T is odd, its final hard output is $s ,$ so the final loss is $\ell _ { \mathrm { o k } }$ when $p _ { T } = s$ and $\ell _ { \mathrm { e r r } }$ otherwise, and

$$
R _ { \mathcal { D } } ^ { \theta } ( F _ { \psi _ { 1 - s _ { \theta } } } ) - R _ { \mathcal { D } } ^ { \theta } ( F _ { \psi _ { s _ { \theta } } } ) = 2 \Delta _ { \ell } T \gamma .\tag{8}
$$

Equations (7) and (8) show that the atomic excess equals the final excess exactly.

For training, step i uses only the i-th coordinate of its own draws, written $X _ { i 1 } , \ldots , X _ { i n } .$ , so the nT coordinates $X _ { i j }$ used are independent. The sub-task ERM therefore chooses

$$
\widehat { s } = \mathbb { 1 } \left\{ \sum _ { i = 1 } ^ { T } \sum _ { j = 1 } ^ { n } X _ { i j } \geq \frac { n T } { 2 } \right\} ,
$$

a majority test on nT independent Bernoulli variables with mean $1 / 2 + \theta \gamma$ . The choice of $\gamma$ guarantees $0 < \gamma \leq 1 / ( 2 \bar { T } )$ , and standard binomial anti-concentration gives a universal $c _ { 1 } > 0$ such that $P _ { \theta } ( \widehat { s } \neq s _ { \theta } ) \geq c _ { 1 }$ . Whenever the wrong hypothesis is selected, Eq. (7) gives atomic excess $2 \Delta _ { \ell } T \gamma$ . Therefore

$$
\mathbb { E } _ { S } \left[ \sum _ { i = 1 } ^ { T } R _ { \mathcal { D } _ { i } } ^ { \theta } ( \psi _ { \hat { s } } ) - \operatorname* { m i n } _ { \psi \in \Psi } \Big \{ \sum _ { i = 1 } ^ { T } R _ { \mathcal { D } _ { i } } ^ { \theta } ( \psi ) \Big \} \right] \geq 2 c _ { 1 } \Delta _ { \ell } T \gamma = \Omega \left( \operatorname* { m i n } \left\{ 1 , \sqrt { \frac { T } { n } } \right\} \right) .
$$

By Eq. (8) the atomic excess equals the final-task excess here, so the same bound holds for $\mathbb { E } _ { S } [ R _ { \mathcal { D } } ^ { \theta } ( F _ { \psi _ { \widehat { s } } } ) - \operatorname* { m i n } _ { \psi \in \Psi } R _ { \mathcal { D } } ^ { \theta } ( F _ { \psi } ) ]$ ; it is $\Omega ( { \sqrt { T / n } } )$ ) when $T \leq n$ □

What the lower bound certifies. The quantity bounded below is exactly the sample-dependent atomic excess of the upper bound—the sub-task ERM’s excess over the best in-class value—and it coincides with the final-task excess. Since the per-step best-in-class loss is $\begin{array} { r } { \Theta ( 1 ) ( \ell _ { \mathrm { o k } } + \frac { 1 } { 2 } \Delta _ { \ell } \approx 0 . 8 3 7 ) } \end{array}$ the excess is identifiable in this instance. Hence the $\Omega ( { \sqrt { T / n } } )$ rate is not an artifact of estimation: it concerns the estimation component only and says nothing about the approximation term.

## C Full Experiments

In this section, we provide the extended experimental details and additional results that complement the main text.

## C.1 Atom-Task Experiments

## C.1.1 Experimental Setup

Atom-Task Benchmark. To validate compositional reasoning capability, we introduce Atom-Task. This task requires the model to sequentially execute a series of atomic operations on a symbolic sequence multiple times, necessitating multi-step reasoning capabilities.

Symbol System and Atomic Operations. The input consists of sequences formed by 6 Greek letters $\{ \dot { \alpha } , \beta , \gamma , \dot { \delta } , \varepsilon , \omega \}$ . We define five basic operations (mapped to IDs t1–t5):

• Reverse (t1): Reverses the order of the sequence.

• Substitute (t2): Swaps symbols in pairs (α ↔ $\cdot \ \beta , \gamma  \delta , \varepsilon  \omega )$

• Duplicate (t3): Duplicates each symbol consecutively.

• Deduplicate (t4): Removes consecutive duplicates.

• Rotate (t5): Moves the first element to the end.

Dataset Design and Distribution. A core challenge in compositional generalization is the distribution shift between training and testing. We design specific training datasets to investigate these effects.

Design Principles. Our training sets adhere to two core principles:

• Dominance of atomic primitives: In Mixed 1-, 3-, and 5-Step, single-step samples constitute 80% of the training trajectories, supporting the acquisition of basic operations.

• Controlled exposure to composition: Three- and five-step training chains provide structural patterns for compositional decomposition; in Mixed 3+5, the five-step chains use ≤ 2 distinct operations, as specified in Section 5.

Training Datasets. To evaluate OOD generalization, we design four training datasets:

1. Single-Step: 5k uniformly distributed one-step samples.

2. Three-Step: 1.66k three-step samples.

3. Mixed 3+5: 1,333 samples (833 three-step and 500 five-step; the five-step chains use at most two distinct operations, bridging length and combinatorial generalization).

4. Mixed 1-, 3-, and 5-Step: For each operation op, a dedicated 5k-sample dataset excludes op from all multi-step chains but keeps it in single-step data. Specifically, it contains 4,000 one-step, 600 three-step, and 400 five-step samples (5,000 training trajectories, or 7,800 atomic transitions in total; see Table 5).

Balanced Supervision: While the sample sizes vary across settings, the number of atomic transitions is held roughly constant: the single-step, three-step and mixed 3+5 datasets each contain about 5,000 transitions (5,000, 1,666 × 3, and ${ \bar { 8 } } 3 3 \times 3 + { \bar { 5 } } 0 0 \times 5 )$ , while the operation-exclusion dataset contains 7,800 and is compared across excluded operations under that same budget, which controls for confounding variables.

Test Sets. We evaluate on four test sets with increasing chain lengths: 3-step, 5-step, 7-step, and 9- step. Each set contains 500 unseen samples, requiring the model to perform compositional reasoning not explicitly memorized from training.

Table 5: Sample distribution in Ablation Datasets. Each dataset excludes a specific operation from multi-step chains. The 5,000 trajectories comprise 7,800 atomic transitions.
<table><tr><td>Data</td><td>Total</td><td>Single-step</td><td>3-step</td><td>5-step</td></tr><tr><td>Ablation_no_*</td><td>5,000</td><td>4,000</td><td>600</td><td>400</td></tr></table>

## C.1.2 Implementation Details

For the Atom-Task, we use Qwen3-4B as the base model. Fine-tuning is performed using low-rank adaptation (LoRA; rank 16, alpha 32) for 3 epochs with a learning rate of $5 \times 1 0 ^ { - 5 }$ . The maximum generation length (max\_new\_tokens) is set to 2048 to accommodate longer reasoning chains. All experiments are conducted on a single NVIDIA RTX 4090 GPU.

Parameter Efficiency: We employ LoRA with the following configuration:

• Rank r = 16 for a balance between expressiveness and parameter efficiency.

• LoRA modules are applied to all linear layers within the attention and MLP blocks.

We employ a Chain-of-Thought training format, where the model is trained to explicitly predict the intermediate output after each atomic operation. For the $\mathrm { \bf { \tilde { \Delta } N o - C o T } \bf { \vec { \Sigma } } }$ baseline, the model is trained to predict only the final output without intermediate states.

Reproducibility: For the main results (Table 1), both CoT and No-CoT entries are means across 3 random seeds (42, 2026, 12345), and the same 3 seeds are used for the operation-exclusion study, the format ablation, the noise tests, and the scaling points. For the format ablation (Table 3), the ablated CoT entries are means over 3 seeds and the original entries average the three-seed means of Table 2. Standard deviations are reported as subscripts and the seed setting is stated in each caption; differences of a few points between entries sharing a seed set should be read as indicative rather than as significance statements.

## C.1.3 Details on Scaling Laws

To complement Figure 3, we provide the detailed accuracy numbers for training-sample scaling and inference-example scaling in Tables 6 and 7, respectively. All scaling experiments are conducted using the “Three-Step” training dataset and evaluated on held-out test sets, each reported point being a mean over 3 seeds (42, 2026, 12345).

Table 6: Accuracy (%) for Sample Size Scaling (n). Each entry is a mean over 3 seeds (42, 2026, 12345).
<table><tr><td>Prop.</td><td>3-step</td><td>5-step</td><td>7-step</td></tr><tr><td>10%</td><td> $9 1 . 9 3 { \scriptstyle \pm 1 . 3 2 }$ </td><td> $7 8 . 0 0 { \scriptstyle \pm 1 . 1 8 }$ </td><td> $5 8 . 7 3 { \scriptstyle \pm 2 . 8 1 }$ </td></tr><tr><td>20%</td><td> $9 5 . 3 3 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $8 4 . 2 7 { \scriptstyle \pm 1 . 2 7 }$ </td><td> $6 5 . 8 7 { \scriptstyle \pm 2 . 3 3 }$ </td></tr><tr><td>50%</td><td> $9 8 . 1 3 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $9 1 . 4 0 { \scriptstyle \pm 1 . 3 0 }$ </td><td> $8 0 . 4 0 { \scriptstyle \pm 3 . 5 6 }$ </td></tr><tr><td>100%</td><td> $9 9 . 3 3 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $9 5 . 7 3 { \scriptstyle \pm 0 . 6 8 }$ </td><td> $8 8 . 8 7 { \scriptstyle \pm 1 . 5 4 }$ </td></tr></table>

Table 7: Accuracy (%) versus Number of Inference-Time Examples. Each entry is a mean over 3 seeds (42, 2026, 12345).
<table><tr><td>Shots</td><td>3-step</td><td>5-step</td><td>7-step</td></tr><tr><td>0-shot</td><td> $9 9 . 2 7 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $9 3 . 4 0 { \scriptstyle \pm 2 . 8 3 }$ </td><td> $8 2 . 5 3 { \scriptstyle \pm 1 . 7 6 }$ </td></tr><tr><td>1-shot</td><td> $9 9 . 3 3 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $9 5 . 7 3 { \scriptstyle \pm 0 . 6 8 }$ </td><td> $8 8 . 8 7 { \scriptstyle \pm 1 . 5 4 }$ </td></tr><tr><td>2-shot</td><td> $9 9 . 3 3 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $9 6 . 6 0 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $9 0 . 9 3 { \scriptstyle \pm 1 . 3 2 }$ </td></tr><tr><td>3-shot</td><td> $9 9 . 0 7 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $9 5 . 7 3 { \scriptstyle \pm 0 . 6 6 }$ </td><td> $8 9 . 9 3 { \scriptstyle \pm 0 . 8 1 }$ </td></tr></table>

## C.1.4 Extended Results and Analysis

Robustness to Noise in Reasoning Steps. To evaluate the robustness of CoT training to data quality, we conduct a noise interference experiment on the Qwen3-4B model. We test: Multi-step (Pure 3-step), Multi-step (Mixed 3+5), and the w/o Substitute ablation setting. (Note: These earlier noise robustness experiments used max\_new\_tokens=1024, but the conclusions remain consistent).

Noise Injection Strategy. We define the basic unit of noise injection as a single inference step. We select a proportion $p \in \{ 0 . 0 5 , 0 . 1 , 0 . 2 \}$ of the total reasoning steps in the training set for interference. A standard reasoning step contains four discrete elements: the operation, the input state, the output state, and the remaining tasks. For each selected step, we randomly choose two of these four elements and perform either Tampering (replacing content with random symbols) or Deletion (emptying the content).

Table 8: Each entry is a mean over 3 seeds (42, 2026, 12345). Robustness to training noise (Qwen3- 4B). Accuracy (%) under varying noise ratios. Each cell shows the result for the 3-step / 5-step / 7-step test sets.
<table><tr><td>Noise Ratio</td><td>Multi-step (Pure 3-step)</td><td>Multi-step (Mix3+5)</td><td>w/o Substitute</td></tr><tr><td> $p = 0 . 0 5$ </td><td> $9 9 . 7 3 _ { \pm 0 . 0 9 } / 9 7 . 8 0 _ { \pm 0 . 8 6 } / 9 0 . 4 7 _ { \pm 2 . 4 6 }$ </td><td> $9 9 . 3 3 _ { \pm 0 . 0 9 } / 9 8 . 5 3 _ { \pm 0 . 3 4 } / 9 4 . 8 7 _ { \pm 0 . 8 1 }$ </td><td> $9 6 . 2 0 { \scriptstyle \pm 0 . 6 5 } / 9 2 . 3 3 { \scriptstyle \pm 1 . 0 6 } / 8 5 . 4 7 { \scriptstyle \pm 2 . 8 0 }$ </td></tr><tr><td> $p = 0 . 1 0$ </td><td> $9 9 . 4 7 { \scriptstyle \pm 0 . 0 9 } / 9 7 . 2 0 { \scriptstyle \pm 0 . 5 9 } / 9 1 . 6 7 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $9 9 . 2 7 \pm 0 . 3 4 / 9 8 . 8 7 \pm 0 . 2 5 / 9 4 . 8 0 \pm 0 . 4 9$ </td><td> $9 5 . 8 0 { \scriptstyle \pm 0 . 3 3 / 9 2 . 3 3 \pm 1 . 2 7 / 8 6 . 7 3 \pm 1 . 0 4 }$ </td></tr><tr><td> $p = 0 . 2 0$ </td><td>99.  $4 0 _ { \pm 0 . 0 0 } / 9 6 . 6 0 _ { \pm 0 . 7 5 } / 9 0 . 7 3 _ { \pm 1 . 8 9 }$ </td><td> $9 9 . 1 3 _ { \pm 0 . 0 9 } / 9 8 . 9 3 _ { \pm 0 . 6 2 } / 9 4 . 8 0 _ { \pm 0 . 2 8 }$ </td><td> $9 6 . 9 3 _ { \pm 0 . 7 7 } / 9 2 . 4 7 _ { \pm 0 . 5 0 } / 8 7 . 1 3 _ { \pm 1 . 9 7 }$ </td></tr></table>

Table 8 presents accuracy for 3-step / 5-step / 7-step test sets respectively. The results indicate strong robustness to noise in reasoning steps. Across all training configurations, increasing the noise ratio does not lead to significant performance degradation; accuracy remains largely consistent with the clean baselines reported in previous tables. In some cases, minor variations are observed but without a clear negative trend, suggesting that the learned compositional reasoning patterns are resilient to localized inconsistencies in the training data.

Table 9: GSM8K baseline and comparative results. Each entry is a mean over 3 seeds (42, 2026, 12345).
<table><tr><td>Model</td><td>Experiment</td><td>Acc(%)</td></tr><tr><td rowspan="3">Llama2-7B</td><td>Untrained</td><td>2.50</td></tr><tr><td>CoT Training</td><td> $3 9 . 1 4 { \scriptstyle \pm 0 . 9 9 }$ </td></tr><tr><td>No-CoT Training</td><td> $8 . 1 6 { \scriptstyle \pm 2 . 5 0 }$ </td></tr><tr><td rowspan="3">Qwen2.5-3B</td><td>Untrained</td><td>64.44</td></tr><tr><td>CoT Training</td><td> $7 9 . 2 3 { \scriptstyle \pm 0 . 2 5 }$ </td></tr><tr><td>No-CoT Training</td><td> $1 9 . 4 1 { \scriptstyle \pm 1 . 1 7 }$ </td></tr></table>

Table 10: GSM8K accuracy under distribution shifts. Each entry is a mean over 3 seeds (42, 2026, 12345).
<table><tr><td>Model</td><td>Shift Type</td><td>Acc(%)</td></tr><tr><td>Llama2-7B</td><td>Scale Shift Complexity Shift Topic Shift</td><td> $3 8 . 7 4 { \scriptstyle \pm 0 . 7 0 }$   $3 6 . 5 7 { \scriptstyle \pm 0 . 1 7 }$   $3 7 . 5 1 { \scriptstyle \pm 2 . 1 4 }$ </td></tr><tr><td>Qwen2.5-3B</td><td>Topic Shift Complexity Shift Answer Scale Shift</td><td> $7 8 . 6 0 { \scriptstyle \pm 0 . 5 5 }$   $7 9 . 5 3 { \scriptstyle \pm 0 . 4 5 }$   $7 9 . 1 0 { \scriptstyle \pm 0 . 2 5 }$ </td></tr></table>

## C.2 GSM8K Experiments

To validate our findings on real-world, complex reasoning scenarios, we conduct experiments on the GSM8K benchmark using Llama2-7B and Qwen2.5-3B fine-tuned on MetaMathQA, a math instruction-tuning dataset (395K samples in total). This protocol evaluates a controlled withinfamily distribution shift from MetaMathQA to GSM8K rather than source-disjoint transfer. Due to computational constraints, we use 30,000 training samples for each experiment. For standard experiments (baseline and noise), we randomly sample these 30,000 instances from the full dataset; for distribution shift experiments, we select 30,000 samples according to specific shift criteria.

## C.2.1 Experimental Setup

Models and Training. We employ Llama2-7B and Qwen2.5-3B as base models. Models are finetuned using LoRA (rank=16, alpha=16) for 1 epoch with a learning rate of $2 \times 1 0 ^ { - 5 }$ and a maximum sequence length of 512. Training data is drawn from the MetaMathQA dataset. For baseline and noise experiments, we randomly sample 30,000 instances; for distribution shift experiments, we construct 30,000-sample subsets by filtering according to three shift indicators (Topic, Complexity, Answer Scale). All experiments are conducted on a single NVIDIA RTX 4090 GPU, and each experiment is run with three random seeds (42, 2026, 12345); the training data is identical across seeds, so reported standard deviations reflect model-side randomness.

Evaluation. All models are evaluated on the standard GSM8K test set. We report accuracy as the primary metric.

## C.2.2 CoT vs. No-CoT & OOD Generalization

Baseline Comparison. As shown in Table 9, CoT training significantly boosts accuracy over untrained baselines, while No-CoT training yields only marginal improvements or even degradation.

Distribution Shift Experiments. In contrast to the random sampling used for baseline and noise studies, here we deliberately create distribution gaps by filtering 30,000 samples from MetaMathQA based on the following criteria:

• Topic Shift (ood\_domain): shift $\begin{array} { r } { \mathrm { \ t o p i c = \frac { | K _ { \mathrm { s a m p l e } } \setminus K _ { \mathrm { G S M 8 K } } | } { | K _ { \mathrm { s a m p l e } } \cup K _ { \mathrm { G S M 8 K } } | } } } \end{array}$ , where $K _ { \mathrm { s a m p l e } }$ is the keyword set of the sample and $K _ { \mathrm { G S M 8 K } }$ the top-20 keyword set of GSM8K; this asymmetric keyword-novelty score measures the fraction of sample keywords absent from GSM8K’s set, normalized by the union size.

• Complexity Shift (ood\_complexity): shi $\begin{array} { r } { \mathrm { f t } _ { \mathrm { c o m p l e x i t y } } = \frac { | \mathrm { s t e p s } _ { \mathrm { s a m p l e } } - \mu _ { \mathrm { G S M 8 K } } | } { \sigma _ { \mathrm { G S M 8 K } } } } \end{array}$ (absolute Z-score of the sample’s step count relative to GSM8K’s mean and standard deviation).

• Answer Scale Shift (ood\_magnitude): $\mathrm { s h i f t } _ { \mathrm { m a g n i t u d e } } = 1 - P _ { \mathrm { G S M 8 K } } ( \mathrm { m a g n i t u d e } _ { \mathrm { s a m p l e } } )$ (one minus the GSM8K probability of the sample’s answer magnitude bin).

The results in Table 10 show that performance remains robust across all shifts, with only mild fluctuations. This stability suggests that the compositional patterns learned via CoT can generalize to practical domains like mathematical reasoning.

Robustness to Reasoning Noise. We further test resilience to imperfect supervision by training on 30,000 randomly sampled CoT instances from MetaMathQA, injected with noise via a controlled strategy: (1) Sample-level probability: Each sample has a probability $p$ of being selected; (2) Singlestep injection: For selected samples, noise is injected into a single randomly chosen reasoning step; (3) Noise type: ‘delete’ (removing the step content) or ‘wrong\_number’ (modifying the numerical result), applied with equal probability (50%). This assesses the model’s ability to handle corrupted intermediate reasoning.  
Table 11: Each entry is a mean over 3 seeds (42, 2026, 12345). GSM8K accuracy under training with different proportions (p) of reasoning noise. All models trained on 30,000 samples.
<table><tr><td>Model</td><td> $p = 0 \ ( \mathbf { B a s e l i n e } )$ </td><td> $p = 0 . 0 5$ </td><td> $p = 0 . 1 0$ </td><td> $p = 0 . 2 0$ </td><td> $p = 0 . 5 0$ </td></tr><tr><td>Llama2-7B</td><td> $3 9 . 1 4 { \scriptstyle \pm 0 . 9 9 }$ </td><td> $3 9 . 2 0 { \scriptstyle \pm 0 . 5 6 }$ </td><td> $3 9 . 7 3 { \scriptstyle \pm 1 . 0 2 }$ </td><td> $3 7 . 6 3 { \scriptstyle \pm 0 . 8 8 }$ </td><td> $3 7 . 6 6 { \scriptstyle \pm 0 . 8 9 }$ </td></tr><tr><td>Qwen2.5-3B</td><td> $7 9 . 2 3 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $7 9 . 2 7 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $7 8 . 5 7 { \scriptstyle \pm 0 . 3 7 }$ </td><td> $7 7 . 2 0 { \scriptstyle \pm 0 . 6 0 }$ </td><td> $7 5 . 6 9 { \scriptstyle \pm 0 . 5 6 }$ </td></tr></table>

Both models exhibit strong resistance to noise (Table 11). Even with up to 50% of samples corrupted, accuracy fluctuates within a narrow band and never collapses. This suggests that the model is relatively insensitive to localized noise in intermediate steps and can still absorb the correct logical patterns.

## C.2.3 Summary of Findings

The GSM8K experiments reveal three main patterns: (1) CoT training substantially improves accuracy over untrained and No-CoT baselines, highlighting the benefit of intermediate step supervision; (2) models are robust to significant reasoning noise, maintaining stable performance even when training data contains deleted or incorrect steps, indicating that supervision remains effective even when individual steps are corrupted; (3) models generalize effectively across distribution shifts in topic, complexity, and answer scale, with only mild accuracy fluctuations, indicating that the reasoning patterns acquired via CoT transfer robustly to real-world mathematical domains.