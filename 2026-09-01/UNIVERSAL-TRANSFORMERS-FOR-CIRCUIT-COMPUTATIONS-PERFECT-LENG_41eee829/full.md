# UNIVERSAL TRANSFORMERS FOR CIRCUIT COMPUTATIONS: PERFECT LENGTH GENERALIZATION IN TINY TRANSFORMERS

Takuya Ito<sup>∗</sup> IBM Research

Ruchir Puri IBM Research

Murray Campbell IBM Research

Parikshit Ram IBM Research

September 1, 2026

## ABSTRACT

Learning generalizable algorithmic computations remains a challenge for neural networks, as reflected in persistent failures on compositional and length generalization benchmarks. We present a provably correct, transformer parameterization (with only 280 learnable parameters for Boolean algebra tasks) capable of learning and evaluating problems of any depth or length. We assume inputs are fully parenthesized, well-formed expressions. Our approach conceptualizes algorithmic tasks as circuit models embedded in transformers, enabling depth-1 circuit reduction in a single forward pass. To achieve depth generalization, we introduce a positional encoding that tracks each gate’s depth within the circuit, enabling the model to identify evaluable subexpressions at each iteration via masked hard attention, with O(n) per-iteration complexity via linear attention. Combined with an autonomous halting criterion, the model terminates after d iterations for problems of depth d, yielding O(n · d) total complexity. We show that training on shallow problem instances (depth 1 and depth 2) effectively recovers interpretable parameters that snap into place, resulting in exact length generalization. Though we establish that our construction provably evaluates Boolean expressions – a universal symbolic computation – of arbitrary length perfectly, in other experiments we also demonstrate that our transformer variant can learn and generalize perfectly (100% accuracy) on other common length generalization benchmarks, including modular arithmetic and ListOps.

Keywords universal transformer · length generalization · circuits · mechanistic interpretability

## 1 Introduction

Understanding how transformers can execute exact symbolic computations remains a challenge in artificial intelligence. Standard transformer architectures have struggled with symbolic generalization [Lake and Baroni, 2018], particularly when generalizing to longer input sequences [Hupkes et al., 2020, Dziri et al., 2023, Zhou et al., 2024]. Although recent works have reported substantial progress on algorithmic benchmarks – such as arithmetic reasoning and formal language recognition – and in some cases have identified interpretable internal representations [He et al., 2024, Cho et al., 2025, Fan et al., 2025, Adler et al., 2025], these advances often rely on task-specific heuristics and lack guarantees of exact algorithmic correctness outside the training distribution. Thus, strong benchmark performance does not ensure that a transformer learns a generalizable algorithm.

We first address this by studying transformers on Boolean formulas – a canonical family of compositional computations that can be represented as circuits [Arora and Barak, 2009]. Circuits provide explicit algorithmic structure – gates, edges, and intermediate values – and naturally scale to arbitrary sizes (Fig. 1). Many string-based tasks (e.g., modular arithmetic and context-free grammars) can be mapped to circuits, yielding a unified notion of algorithmic generalization via a graph-based structure [Ito et al., 2025]. Here, we parameterize by hand a fixed, small transformer (280 parameters) that provably evaluates arbitrary non-uniform Boolean expressions via a circuit reduction procedure. When randomly initialized, the model can learn to generalize perfectly from depth-1 and depth-2 samples. This behavior is enabled by a novel positional encoding (PE) mechanism that infers circuit depth from a parenthesized string and identifies evaluable subexpressions at each step. The model is also computationally efficient: attention scales linearly in memory, and the number of iterations scales approximately logarithmically in input length, suggesting that exact algorithmic computation in transformers can be realized without sacrificing efficiency or parameter compactness.

![](images/f69b40b1b1df36186f903bc65ad94450714bedde0e08ba686c1f8eebb9074536.jpg)  
Figure 1: Example A) Boolean and B) modular arithmetic expressions, encoded as circuits. Circuits provide a structured algorithm to evaluate an expression by following the edges from the input gates to the output gate. We illustrate circuit evaluation through circuit reduction, whereby each operator gate is iteratively updated by the subexpression’s computed value until the output is obtained.

To demonstrate generality beyond Boolean formulas, we show that the same architecture learns and perfectly generalizes to other common compositional tasks, such as modular arithmetic and ListOps [Nangia and Bowman, 2018] from shallow training data. Together, these results provide an example of a transformer that provably implements an exact circuit algorithm while remaining interpretable, efficient, and learnable. We present our findings first by introducing the exact model construction that can evaluate Boolean formulas without any training, followed by empirical learnability experiments for all tasks in randomly initialized models.

## Contributions.

1. A provably-correct, white-box transformer construction for non-uniform (arbitrary depth) Boolean expressions, enabling perfect evaluation via a circuit reduction procedure.

2. Empirical results (and code) demonstrating that this architecture can learn and perfectly generalize common compositional tasks, including Boolean algebra, modular arithmetic, and ListOps.

## 2 Preliminaries

We begin by introducing notation and problem preliminaries. For clarity, we focus on Boolean expressions, which provide the simplest setting for encoding universal symbolic computation and requires only an embedding dimension of 7 to one-hot encode its entire vocabulary. While we focus on this language for exposition, the general model construction applies to any domain that can be represented as a circuit, with only superficial changes to the model (e.g., changes to the requisite embedding dimension according to the size of the task vocabulary).

We use a universal Boolean language: $\Sigma _ { \mathrm { b o o l } } = \left( \left( , \right) , 1 , 0 , \Lambda , \vee , \sim \right)$ . Operands are 1 (TRUE) and 0 (FALSE). Operators are unary ∼ (NOT) and binary ∧ (AND), ∨ (OR). See Fig. 1A for examples of expressions, and Appendix B.1 for further details. Token Embedding. Let $D = | \Sigma _ { \mathrm { b o o l } } |$ . Each token is encoded as a one-hot vector in $\{ 0 , \dot { 1 } \} ^ { D }$ . Note that $\Sigma _ { \mathrm { b o o l } }$ is an ordered vocabulary. Thus, the embedding space for each token of $\Sigma _ { \mathrm { b o o l } }$ is mapped to the embedding space $\{ 0 , 1 \} ^ { D }$ where the i-th coordinate corresponds to the i-th token in Σ. An input expression of length n is represented by

$$
X ^ { ( 0 ) } = ( x _ { 1 } ^ { ( 0 ) } , \ldots , x _ { n } ^ { ( 0 ) } ) \in \{ 0 , 1 \} ^ { n \times D }\tag{1}
$$

We write $X ^ { ( t ) }$ for the sequence state after t model iterations.

Terminology. Let x denote the semantic value (or output) of a Boolean expression x $( \mathrm { i . e . }$ , eval $\mathbf { \tau } ( \mathbf { x } ) )$ , such that $[ [ x ] ] \in \{ 1 , 0 \}$ . Let $d ( x )$ denote the depth of x represented as a circuit, and active depth $\delta ( X )$ as the maximum depth among unreduced subexpressions in X. For fully parenthesized expressions, depth is defined recursively: a (0 or 1) has depth 0, while a compound expression has depth one greater than the deepest of its constituent subexpressions. For fully parenthesized expressions, a reducible subexpression is an innermost parenthesized span containing only an operator and its operands. A sequence state $X ^ { ( t ) } \in \{ \hat { 0 } , 1 \} ^ { n \times D }$ is a valid expression if each innermost subexpression corresponds either to an unevaluated reducible subexpression or its semantic value, and its global evaluation (fina semantic value $\mathbb { [ } X ^ { ( t ) } ] \mathbb { I }$ obtained by recursively evaluating all subexpressions) is well-defined. Two valid expressions X and Y are semantically equivalent, written ${ \dot { X } } \equiv Y$ , if they have the same global evaluation.

Model Overview. The model evaluates expressions by iterative local circuit reduction. At each iteration it reduces the expression’s depth by 1 through the following steps:

1. Computes a reduction mask identifying the deepest reducible subexpressions.

2. Decomposes that mask into parallel subexpressions that can be evaluated independently.

3. Routes token information within a subexpression with attention.

4. Applies a local semantic map to compute the semantic value of each subexpression via MLP.

5. Evaluates masked subexpressions while preserving non-reduced tokens with a gated residual.

## 3 Model Construction and Formalization

The following formal argument proceeds step by step by crafting an exact parameterization. For each model component, we formally construct each component and summarize its key contribution in a corresponding lemma. Throughout, we also provide figures that illustrate, with concrete examples, how these components collectively implement circuit depth reduction (Fig. 2-3). We combine these ingredients into theorem statements for depth-1 circuit reduction and arbitrary-depth correctness. By construction, the model does not require training and achieves perfect performance, correctly evaluating Boolean expressions of arbitrary depth. (In the subsequent section, we explore learnability under random initialization given the base architecture.) For our formal claims, we make the following assumptions: 1) Inputs are fully parenthesized (balanced parentheses) valid expressions. 2) Tokens within a valid subexpression are order-invariant (commutative up to parentheses), e.g., $( 1 \vee 0 ) \equiv ( 0 \vee 1 )$ . We visualize all the chosen model parameter in Appendix Fig. A.1, and provide the code for this solution in the supplementary zip file.

## 3.1 Positional encoding: Identifying a reduction mask and parallel subexpressions

The key insight that enables a single-layer transformer to implement depth-1 circuit reduction is that representing a circuit from a string fundamentally requires encoding the circuit’s depth (i.e., by counting parentheses). In what follows, we describe a novel PE mechanism that, given a valid expression $X ^ { ( t ) }$ , is able to identify the deepest reducible subexpression(s).

Reduction mask. We construct a PE mechanism with a gating matrix parameter $G _ { \mathrm { P E } } \in \{ - 1 , 0 , 1 \} ^ { D \times D }$ that produces a reduction mask $P ^ { ( t ) } \in \{ 0 , 1 \} ^ { n }$ identifying the tokens belonging to the deepest unreduced subexpressions, i.e., those at depth $\delta ( X ^ { ( t ) } )$ . Formally, $P _ { i } ^ { ( t ) } = 1$ if and only if the i-th token of $X ^ { ( t ) }$ belongs to a maximally nested (deepest) parenthesized subexpression $( \mathrm { i . e . }$ , it is reducible). Identifying such subexpressions reduces to computing parenthesis nesting depth. Let $" ( "$ and ")" correspond to the first and second embedding dimensions, respectively. We define

$$
G _ { \mathrm { P E } } = \mathrm { d i a g } ( 1 , - 1 , 0 , 0 , \ldots , 0 ) ,\tag{2}
$$

so that when an expression $X ^ { ( t ) }$ is gated (multiplied) by $G _ { \mathrm { P E } } .$ , open parentheses contribute +1, closed parentheses contribute −1, and all other tokens are ignored.

To count the number of open/closed parentheses, we compute the cumulative sum of parentheses in $X ^ { ( t ) }$ , both forward and backwards along the sequence. Let $L \in \{ 0 , 1 \} ^ { n \times n }$ be a lower-triangular matrix of ones. The model computes the cumulative sum of parenthesis counts by gating $X ^ { ( t ) }$ with $G _ { \mathrm { P E } }$ , and then computes the cumulative sum forwards and backwards with $L$ to maintain symmetry:

$$
{ \cal A } _ { 1 } ^ { ( t ) } = ( L X ^ { ( t ) } G _ { \mathrm { P E } } ) \mathbf { 1 } , \qquad { \cal A } _ { 2 } ^ { ( t ) } = - ( L ^ { \top } X ^ { ( t ) } G _ { \mathrm { P E } } ) \mathbf { 1 } ,\tag{3}
$$

where $A _ { 1 } ^ { ( t ) }$ counts the net number of open parentheses from position 1 to $n ,$ and ${ A } _ { 2 } ^ { ( t ) }$ counts the net number of close parentheses from position n to 1. These are then stacked into

$$
\begin{array} { r } { \boldsymbol { A } ^ { ( t ) } = \operatorname { s t a c k } ( \boldsymbol { A } _ { 1 } ^ { ( t ) } , \boldsymbol { A } _ { 2 } ^ { ( t ) } ) \in \mathbb { Z } ^ { n \times 2 } , } \end{array}\tag{4}
$$

and a hardmax operation is applied to identify sequence positions with maximal depth:

$$
P ^ { ( t ) } = \mathbb { I } [ \operatorname* { m a x } _ { j \in \{ 1 , 2 \} } A _ { i , j } ^ { ( t ) } = \operatorname* { m a x } _ { i ^ { \prime } , j ^ { \prime } } A _ { i ^ { \prime } , j ^ { \prime } } ^ { ( t ) } ] \in \{ 0 , 1 \} ^ { n } ,\tag{5}
$$

where $\mathbb { I } [ \cdot ]$ is the indicator function. This selects positions i where the maximum depth value across both forward and backward counts equals the global maximum depth, identifying the deepest nested positions $( \mathrm { e . g } ^ { } )$ ., see Fig. 2A for intuition). Thus, the resulting mask $P ^ { ( t ) }$ selects exactly the tokens belonging to the deepest reducible subexpressions. (Note that in learnable experiments reported in Section 4, $G _ { \mathrm { P E } }$ is set to be learnable.)

Lemma 1 (Reduction mask). Define parameter $G _ { P E }$ as in Eq. 2. Let $X ^ { ( t ) }$ be a valid expression. The PE mechanism equipped with $G _ { P E }$ and described in Eqs. 3-5 produces the binary reduction mask $P ^ { ( t ) }$ , whose support consists exactly ofthe tokens in the deepest parenthesized spans of $X ^ { ( t ) }$

Proof. The claim follows directly from the construction in Eqs. 2-5. A visual illustration with an example can be found in Fig. 2A. □

A  
![](images/db35ef2d744e699d85d1b0a25aea3c74690653bffb27bf1c6509364eed49af45.jpg)

![](images/1ba8780b163e9f3f7b69e2b438c6bc3dcf656229e8a399371088ce9e28153d31.jpg)  
Figure 2: A visual demonstration of the PE and attention mechanism constructions (Lemmas 1-3. A) PE gating identifies reducible subexpressions by first computing the reduction mask $P ^ { ( t ) }$ , and then decomposing the independent reducible subexpressions $C _ { 1 } ^ { ( t ) }$ and $C _ { 2 } ^ { ( t ) }$ . Beginning with the specific example input $^ { \ast \ast } ( ( 1 \vee 0 ) \wedge ( \sim 0 ) ) ^ { \ast } ( X ^ { ( t ) } )$ , we characterize and label each of the PE operations and variable assignments. B) Type-constrained attention mechanism is implemented on each subexpression cluster (green and pink masks) generated from the PE. $W _ { Q }$ and $W _ { K }$ represent the actual $\{ 0 , 1 \} ^ { D \times 4 }$ matrix construction our solution (Lemma 3) uses. By implementing attention on each reducible subexpression independently, we constrain attention routing to independent subexpressions.

Parallel subexpression decomposition. Given the binary reduction mask $P ^ { ( t ) }$ from Lemma 1 generated from parameter $G _ { \mathrm { P E } }$ , multiple disjoint subexpressions may be reducible in parallel for expressions x with $\bar { d } ( x ) > 1$ . Thus, the model must decompose these subexpressions into separate clusters that can be independently evaluated.

We now construct a PE submechanism that partitions $P ^ { ( t ) }$ into disjoint contiguous sets $\boldsymbol { \mathcal { C } } ^ { ( t ) } = \{ C _ { 1 } ^ { ( t ) } , \ldots , C _ { k _ { t } } ^ { ( t ) } \}$ such that each cluster $C _ { c } ^ { ( t ) }$ corresponds to exactly one locally reducible subexpression, and distinct clusters can be evaluated independently in parallel (Fig. 2A). Let $\mathbf { 0 } _ { 1 } \in \{ 0 , 1 \} ^ { 1 }$ denote a leading zero. Define the padded mask $\widetilde P ^ { ( t ) } = [ \mathbf { 0 } _ { 1 } ; \overset { \textstyle \mathsf { \sim } } { P ^ { ( t ) } } ] \in \{ \overset { \textstyle } { 0 } , 1 \} ^ { \overset { \textstyle \mathsf { \sim } } { n } + 1 }$ . The discrete difference operator identifies transitions from unmasked to masked positions:

$$
D _ { i } ^ { ( t ) } = \widetilde { P } _ { i } ^ { ( t ) } - \widetilde { P } _ { i - 1 } ^ { ( t ) } , \quad i = 2 , \ldots , n + 1 .\tag{6}
$$

Each positive transition $( D _ { i } ^ { ( t ) } > 0 )$ marks the start of a new subexpression. Cumulative summation of cluster starts assigns a unique integer ID to each contiguous masked segment, then unmasked positions are zeroed by multiplying with the original reduction mask $P ^ { ( t ) }$

$$
\mathrm { i d } ^ { ( t ) } = P ^ { ( t ) } \odot L \mathbb { I } ( D ^ { ( t ) } > 0 )\tag{7}
$$

This assigns each position a cluster ID: $\mathrm { i d } _ { i } ^ { ( t ) } = 0$ for unmasked positions, and $\mathrm { i d } _ { i } ^ { ( t ) } = c$ (where $c \in \{ 1 , \ldots , k _ { t } \} )$ ) for positions in the c-th subexpression cluster. The total number of clusters $k _ { t }$ equals the number of $0  1$ transitions in $P ^ { ( t ) }$ . Each cluster corresponds to a contiguous interval of positions. For cluster ID $c \in \{ 1 , \ldots , k _ { t } \}$ , define the position set

$$
C _ { c } ^ { ( t ) } = \{ i \in \{ 1 , \ldots , n \} : \mathrm { i d } _ { i } ^ { ( t ) } = c \} ,\tag{8}
$$

which forms a contiguous interval $C _ { c } ^ { ( t ) } = \{ i _ { c } , i _ { c } + 1 , \ldots , j _ { c } \}$ for $1 \leq i _ { c } \leq j _ { c } \leq n$ . The collection of all clusters is thus

$$
\mathcal { C } ^ { ( t ) } = \{ C _ { 1 } ^ { ( t ) } , C _ { 2 } ^ { ( t ) } , \ldots , C _ { k _ { t } } ^ { ( t ) } \} .\tag{9}
$$

Because the support of $P ^ { ( t ) }$ consists of disjoint contiguous reducible spans, each cluster ${ C } _ { c } ^ { ( t ) }$ coincides with one span (subexpression) such that each subexpression can be evaluated independently.

Lemma 2 (Parallel subexpression decomposition). Let $P ^ { ( t ) }$ be the binary reduction mask specified in Lemma 1, whose support is the disjoint union of reducible subexpressions, such as the example provided in Fig. 2A). Then the clustering vector $\mathrm { i d } ^ { ( t ) }$ in $E q .$ 7 partitions the support of $P ^ { ( t ) }$ into disjoint contiguous clusters $\mathcal { C } ^ { ( t ) } = \{ C _ { 1 } ^ { ( t ) } , \ldots , C _ { k _ { t } } ^ { ( t ) } \} ( E q .$ . 9) such that each cluster ${ C } _ { c } ^ { ( t ) }$ corresponds to exactly one locally reducible subexpression, and distinct clusters can be evaluated in parallel.

Proof. The claim follows directly from the construction in Eqs. 6-9. A visual illustration with an example can be found in the upper half of Fig. 2A. □

## 3.2 Attention Mechanism: Type-constrained token routing

By Lemma 2, we constructed disjoint contiguous clusters $\mathcal { C } ^ { ( t ) }$ of $X ^ { ( t ) }$ that can be evaluated in parallel. Next, we apply attention separately to each subexpression ${ C } _ { c } ^ { ( t ) }$ . This ensures that attention is limited to a single, reducible span (subexpression). Moreover, reducible subexpressions encode a local, depth-1 circuit structure. In Boolean algebra, thi local structure follows a grammar that obeys strict type constraints: operands map to operators $( { \mathrm { e . g . , i n } } \ ^ { \ast } ( 1 \vee 0 ) ^ { \cdots }$ , 1 $ \lor$ and $0  \lor ;$ Fig. 1A). We leverage these constraints to construct token type-specific mappings within the attention mechanism, enabling faithful representation of Boolean grammars.

Local and type-based attention. Without loss of generality, choose $C _ { c } ^ { ( t ) } \in \mathcal { C } ^ { ( t ) }$ , and let

$$
X _ { c } ^ { ( t ) } = \mathbb { I } [ i \in C _ { c } ^ { ( t ) } ] \odot X ^ { ( t ) }\tag{10}
$$

denote the token embeddings for subexpression ${ C } _ { c } ^ { ( t ) }$ . Let queries, keys, and values be given by

$$
Q _ { c } ^ { ( t ) } = X _ { c } ^ { ( t ) } W _ { Q } , \quad K _ { c } ^ { ( t ) } = X _ { c } ^ { ( t ) } W _ { K } , \quad V _ { c } ^ { ( t ) } = X _ { c } ^ { ( t ) } W _ { V } ,\tag{11}
$$

where $W _ { Q } , W _ { K } \in \{ 0 , 1 \} ^ { D \times d _ { h } }$ are binary routing matrices that permit only binarized admissible interactions between token types (operands to operators, and operator self-interactions), and 0 everywhere else; see Fig. 2B (or Fig. A.1) for the exact matrix constructions for $W _ { Q }$ and $W _ { K }$ . Specifically for Boolean expressions, $D = 7$ and $d _ { h } = 4$ , where $d _ { h }$ encodes one dimension for all operands (0 and 1), and a dimension each for each operator separately $( \wedge , \vee$ and ∼). $W _ { V } = I _ { D }$ is fixed as the identity. Notably, the product $Q _ { c } ^ { ( t ) } \cdot ( K _ { c } ^ { ( t ) } ) ^ { \top }$ produces an attention weight matrix that routes the operand and operator embeddings within $C _ { c } ^ { ( t ) }$ to the operator’s token position (Fig 2B).

Bag-of-tokens aggregation. Within $C _ { c } ^ { ( t ) } , V _ { c } ^ { ( t ) } = X _ { c } ^ { ( t ) }$ , since $W _ { V } = I _ { D }$ . Thus, the attention output aggregates token embeddings via

$$
H _ { c } ^ { ( t ) } = Q _ { c } ^ { ( t ) } ( K _ { c } ^ { ( t ) } ) ^ { \top } X _ { c } ^ { ( t ) }\tag{12}
$$

Critically, contributions are summed over tokens locally in $C _ { c } ^ { ( t ) }$ (and therefore do not interact across subexpressions) and depend only on their routed types. By construction of ${ Q } _ { c } ^ { ( t ) }$ and $K _ { c } ^ { ( t ) }$ (Fig. 2B), the resulting vector is a “bag-of-words embedding that simply sums the admissible token embeddings (operands with operators) at the operator’s token position using only the admissible tokens in ${ C } _ { c } ^ { ( t ) }$ , and leaves all other positions in $C _ { c } ^ { ( t ) }$ as 0 vectors. Note that the attention mechanism does not use a softmax. Thus, the operation in Eq. 12 reduces to linear attention, with $Q _ { c } ^ { ( t ) } \big ( ( K _ { c } ^ { ( t ) } ) ^ { \top } X _ { c } ^ { ( t ) } \big )$ which allows us to avoid the quadratic cost of standard softmax attention [Katharopoulos et al., 2020].

Lemma 3 (Type-constrained attention mechanism). Using $W _ { Q }$ and $W _ { K }$ as defined in Eq. 11 and exactly constructed in $F i g . ~ 2 B ) ,$ , the attention output at the operator position computes a permutation-invariant summed aggregation of tokens in $C _ { c } ^ { ( t ) }$ , yielding a bag-of-tokens (counts) vector ofthe admissible token types in the reducible subexpression at the token embedding location ofthe operator.

Proof. The claim follows directly from the construction in Eqs. 10-12. A visual illustration with an example can be found in Fig. 2B. □

## 3.3 Feedforward Layer: Local semantic evaluation

The output of the preceding Lemma 3 is a bag-of-words vector of a locally reducible subexpression. For reducible Boolean expressions, the semantic value can be fully determined by the operator type together with the pair of operand values, independent of order. This mapping can be realized as a finite look-up table of depth-1 Boolean tree. Since the set of these valid reducible subexpressions is finite, f is a function on a finite domain. Any function on a finite domain is realizable by a single-hidden-layer MLP [Hornik et al., 1989]. In practice, in our constructed model solution, we train a single-hidden-layer MLP to simulate a finite truth table for depth-1 Boolean expressions and append this module to our transformer, although a fixed dictionary look-up can also be straightforwardly implemented (see Code and Appendix Fig. A.1A).

Lemma 4 (Feedforward lookup table). There exists a function $f : \mathbb { Z } ^ { V } \to \{ 0 , 1 \} ^ { D }$ such that, when applied to the bag-of-word token representation ofany valid reducible subexpression $C _ { c } ^ { ( t ) }$ , it outputs the correct one-hot embedding ofthe resulting subexpression, and a 0 vector otherwise.

Proof. The proof follows by a standard finite-domain argument and is deferred to Appendix A.2.

## 3.4 Gated residual update

After the feedforward network (Lemma 4), reducible subexpressions produce a single token embedding that correctly evaluates that subexpression at the token position of that subexpression’s operator, and 0 vectors everywhere else (see Fig. 3). However, to evaluate the full circuit beyond depth-1 expressions, we require a mechanism to preserve unreduced tokens/expressions from $X ^ { ( t ) } \mathrm { t o } X ^ { ( t + 1 ) }$ . Concretely, once a subexpression is reduced, its interior tokens (the operands and the consumed parentheses) become the 0 vector, and only the operator position carries the computed value forward. The gated residual then leaves these zeroed positions untouched at subsequent iterations.

Lemma 5 (Gated residual connection). Let $P ^ { ( t ) } \in \{ 0 , 1 \} ^ { n }$ be the reduction mask specified in Lemma 1 and $M ^ { ( t ) } =$ $\mathbf { 1 } - P ^ { ( t ) }$ its complement. Define the gated residual update

$$
\boldsymbol { X } ^ { ( t + 1 ) } = \boldsymbol { M } ^ { ( t ) } \odot \boldsymbol { X } ^ { ( t ) } + \boldsymbol { P } ^ { ( t ) } \odot \boldsymbol { Y } ^ { ( t ) } .\tag{13}
$$

Then every position outside the reduction mask is preserved, and only positions inside the reduction mask are updated.

Proof. Deferred to Appendix A.2

## 3.5 Depth-1 circuit reduction and global circuit evaluation

The preceding lemmas construct specific model components that collectively imply that a single model iteration implements a depth-1 circuit reduction step (Fig. 3).

Theorem 1 (Depth-1 circuit reduction). Let F be one transformer iteration (PE, attention, MLP, and gated residual; Lemmas 1–5). For any valid $X ^ { ( t ) }$ with active depth $\delta ( X ^ { ( t ) } ) > 0 ,$ , we construct F such that $X ^ { ( t + \bar { 1 } ) } = F ( X ^ { ( t ) } )$ simultaneously evaluates every reducible subexpression, replaces each by its correct value, and preserves all inactive positions. Then,

$$
X ^ { ( t + 1 ) } \equiv X ^ { ( t ) } \qquad a n d \qquad \delta ( X ^ { ( t + 1 ) } ) = \delta ( X ^ { ( t ) } ) - 1 .
$$

Proof. The proof follows by combining Lemmas 1-5, and is deferred to Appendix A.2.

Theorem 2 (Global circuit evaluation). For any valid Boolean expression x of depth $d ( x )$ , with $X ^ { ( t + 1 ) } = F ( X ^ { ( t ) } )$ and $X ^ { ( 0 ) }$ as its embedding,

$$
X ^ { ( d ( x ) ) } \equiv [ [ x ] ] \quad a n d \quad \delta ( X ^ { ( d ( x ) } ) = 0
$$

Proof. The proof follows by applying Theorem 1 inductively, and is deferred to Appendix A.2.

Importantly, this construction halts autonomously, as $\delta ( X ^ { ( t ) } )$ is computed within the PE mechanism (by Eq. 4, when $\textstyle \sum _ { i , j } A _ { i j } ^ { ( t ) } = 0$ and there remain no parentheses in the sequence). This transformer guarantees exact circuit evaluation, and we visualize the parameters for this model in Fig. A.1.

![](images/1fcf5540cacbe22920249a1d444c64aac045d6ae6059b27b54180348d57c8095.jpg)  
Theorem 2: Global Correctness  
Figure 3: An illustrative example of Theorem 1 and 2. Note that each token position is a D-dimensional embedding that encodes a one-hot or counts vector. Empty gray token boxes in the sequence denote the 0 vector. We visualize evaluation on the depth 2 expression $^ { \circ \circ } ( ( 1 \lor 0 ) \land ( \sim 0 ) ) ^ { \prime }$

Computational complexity Each transformer iteration has near-linear attention cost $O ( n )$ , with the dominant typed routing step scaling as $O ( k n )$ for k reducible subexpressions, but effectively $O ( n )$ for structured inputs since $k \ll n$ The number of model iterations is $d ( x )$ , yielding fewer iterations than standard token-by-token evaluation (e.g., in RNNs), which typically requires Θ(n) steps [Deletang et al., 2022] (Appendix A.1 for further details). Thus, the tota model complexity is ${ \dot { O ( } } n \cdot { \bar { d } } ( x ) { \dot { ) } }$ . Because each additional level of nesting introduces at least one pair of parentheses and one operator, $d ( x ) \leq n / 3 .$ . This implies the cost is always sub-quadratic; the worst case $O ( n ^ { 2 } )$ arises only for maximally unbalanced circuits with $d ( x ) = \Theta ( n )$ . Empirically, input length grows exponentially with depth (d ≈ log n), giving $d ( x ) = O ( \log n )$ and an effective cost of O(n log n) (see Fig. 4C).

## 4 Numerical Experiments

In this section, we demonstrate that this architecture learns compositional tasks and achieves perfect length generaliza tion, extrapolating to arbitrary depths after training only on shallow instances.

Tasks. We evaluated three families of algorithmic tasks: modular arithmetic, Boolean logic, and ListOps [Nangia and Bowman, 2018]. In modular arithmetic, the input is a parenthesized expression over digits and the operators + and \*, and the label is the value of the full expression modulo 10. For example, (2 + 3) maps to 5, while $\left( \left( 2 \ + \ 3 \right) \right.$ \* 4) maps to 0 modulo 10. In Boolean evaluation, the input is a formula over 1, 0, ∧, ∨, and ∼ and the label is its truth value. For instance, (1∧0) evaluates to 0, while (∼(1∨0)) also evaluates to 0. In ListOps, the input is a nested expression over digits with operators such as MAX, MIN, SM (sum modulo), and MED, and the label is the resulting digit. For example, ( MAX 2 7 4 ) evaluates to 7, and ( SM 3 4 8 ) evaluates to 5 modulo 10, and ( MAX ( MIN 8 3 $5 ~ ) ~ ( ~ \tilde { \textsf { S M 4 7 6 } } ) )$ evaluates to 7. Note for ListOps, we limited the number of fan-in to 3 as our focus was on depth generalization. See Appendix B.1 for further details.

Training. We train all learnable components of the universal transformer end-to-end from random initialization, including parameters: PE parameters $G _ { \mathrm { P E } } .$ , attention mechanism parameters, and the MLP. The total number of parameters per task is visualized in Fig. 4B; the number of parameters per model is dictated by the size of the vocabulary required for the task and the MLP size, which scales with the finite lookup table). The $G _ { \mathrm { P E } }$ is parameterized by continuous logits that are ternarized $^ { \mathrm { t o } - 1 , 0 , 1 }$ with a straight through estimator. The attention mechanism are likewise initialized from random logits, and binarized with a straight-through estimators in $W _ { Q }$ and $W _ { K }$ , while $W _ { V } = I _ { D }$ is fixed (see Fig. A.1). The feedforward module is an MLP with a single task-dependent expansive hidden layer (2x embedding dimension for Boolean and arithmetic evaluation and 8x for ListOps) and activation functions (quadratic activation function for Boolean and arithmetic, and ReLU for ListOps), and the output is generated from a gated normalization step to produce a one-hot vector that corresponds to a token in the vocabulary. Training employed within-epoch curriculum learning: each epoch first trained exclusively on depth-1 expressions, then jointly on depths 1 and 2. Models were optimized on a Binary Cross Entropy objective function using Adam (learning rate 0.01) with a two-phase schedule (10% linear warmup from 30% to 100% of base rate, then cosine annealing to 5%) to address vanishing gradients in shallow expressions. Architecturally, models were the same across tasks with the exception of the embedding dimension and MLP size. We trained 10 random seeds per model. Additional details can be found in Appendix B.

Table 1: We achieved 100% generalization accuracy on all 3 tasks by training models from random initialization on depth 1 and 2 problems, and evaluating 100 problems at each depth from 3-30.
<table><tr><td>Task</td><td>Avg Accuracy</td><td>Avg Tokens (SD)</td><td>Max Tokens</td><td>Depths (OOD)</td></tr><tr><td>Arithmetic</td><td>1.000</td><td>1937.89 (2725.32)</td><td>18085</td><td>3-30</td></tr><tr><td>Boolean</td><td>1.000</td><td>710.85 (914.74)</td><td>7441</td><td>3-30</td></tr><tr><td>ListOps</td><td>1.000</td><td>3852.161(5819.27)</td><td>41389</td><td>3-30</td></tr></table>

![](images/da3c95faf565cd80e02dba3c198a653f91f3f4db11fe501f54e44a7ef12d30f1.jpg)

![](images/5f70164240269107bd40b2f7a28249daf29c161fc52b8626b9d7ee5f0005c68d.jpg)

B  
![](images/1d5310f560b19205e9eb15e87521cd5e32acde107e12962e1c8e88396a35446e.jpg)

![](images/cc20cf07c93b5d947fb87ab2fd2a7ec7c459a009aadcf2a621f193d5af9b2778.jpg)

C  
![](images/b38537dfdcbf3df43e2eb67447afc46e0038beea361ac6aedaedd9825e993c88.jpg)  
Figure 4: A) Averaged loss trajectories for arithmetic, Boolean, and ListOps tasks (including only seeds that converged). (Shaded regions indicate 95% CI.) B) Parameter counts across each of the models, which are dictated by 1) the task vocabulary, and 2) the size of the MLP’s hidden dimension. C) The number of input tokens required to express tasks of a given depth. As depth linearly increases, the number of input tokens required to encode a problem rapidly increase (log-linear scale).

Results. Randomly initialized models converged for 10/10 seeds on the modular arithmetic task, and 9/10 seeds for Boolean and ListOps evaluations (Fig. 4A). During training, optimization proceeded until the model snapped into an exact solution with zero training loss. We evaluated trained models on each task using samples with deeper depth than seen during training (Table 1). We limited evaluation to problems with up to depth 30 due to memory limitations (depth 30 ListOps problems had upwards of 41k tokens). The model halts autonomously when there remain no parentheses left in the sequence $X ^ { ( t ) }$ (i.e., by Eq. 4, when $\begin{array} { r } { \sum _ { i , j } A _ { i j } ^ { ( t ) } = 0 ) } \end{array}$ , with the number of iterations matching exactly the depth of the problem. Critically, our models achieved 100% generalization accuracy on all tasks, irrespective of the problem depth or the number of tokens. (We tested on 100 problems per depth sampled from a uniform distribution.) Notably, prior works studying length generalization typically limit evaluations of up to 1k tokens, an order of magnitude less than the problems evaluated here [Li et al., 2024, Cho et al., 2025, Zhou et al., 2024, Kazemnejad et al., 2023]. Due to the compactness of the models, we are able to provide complete visualizations of the learned representations in Appendix Fig. A.1B,C), directly comparing the randomly initialized model, the learned model, and the constructed model defined Section 3. We note that while learned models exactly recover the prescribed PE gating parameters defined in the construction, the attention parameters $( W _ { Q } , W _ { K } )$ often differed from our formulation due to solution degeneracy. Nevertheless, all learned models exhibited perfect generalization for arbitrary depth algorithmic problems.

## 4.1 Comparison to standard architectures

To test whether the parenthesized structure alone makes these tasks easy, we train a standard looped (weight-shared) transformer [Fan et al., 2025] with full softmax attention and learnable $\dot { W _ { Q } } , W _ { K } , W _ { V }$ under an identical protocol (fully parenthesized inputs, depth-1 and depth-2 curriculum, same iteration count, and matched hyperparameters). (We supply the ground-truth depth at test time, as these models do not autonomously halt.) We evaluate three positional encodings: learned PEs, RoPE [Su et al., 2022], and FIRE [Li et al., 2024]. We report accuracy for seeds that reach $\geq 0 . 9 5$ on both training depths. We found that these models still collapse towards chance immediately beyond the training depths (Table 2), while our construction remains exact. We further compare to CRvNN [Chowdhury and Caragea, 2021], a continuous recursive network previously shown to length generalize well on similar tasks. However, on ListOps, using our same shallow training protocol and despite roughly 500× more parameters, it degrades from 0.67 at depth 3 to 0.17 at depth 10 despite fitting the training depths (Appendix C).

Table 2: Comparison to a standard looped transformer. We report held-out accuracy at the last training depth $( d = 2 )$ and two out-of-distribution depths $( d = 3$ , 10); chance level in parentheses. Accuracy includes seeds that reach ≥ 0.95 on both training depths. On arithmetic and ListOps, only 1/5 seeds fit this criterion per positional encoding within the current training budget. On Boolean, 5/5 (learned, RoPE) and 3/5 (FIRE). Every trained model still collapses toward chance out of distribution, whereas our construction remains exact. See Appendix C for further details.
<table><tr><td></td><td colspan="3">Arithmetic (0.10)</td><td colspan="3">Boolean (0.50)</td><td colspan="3">ListOps (0.10)</td></tr><tr><td>Model</td><td>depth-2</td><td>depth-3</td><td>depth-10</td><td>depth-2</td><td>depth-3</td><td>depth-10</td><td>depth-2</td><td>depth-3</td><td>depth-10</td></tr><tr><td>Ours</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>Looped, learned</td><td>1.00</td><td>0.10</td><td>0.05</td><td>1.00</td><td>0.55</td><td>0.48</td><td>0.98</td><td>0.37</td><td>0.07</td></tr><tr><td>Looped, RoPE</td><td>0.73</td><td>0.18</td><td>0.00</td><td>0.98</td><td>0.50</td><td>0.12</td><td>1.00</td><td>0.43</td><td>0.08</td></tr><tr><td>Looped, FIRE</td><td>1.00</td><td>0.23</td><td>0.03</td><td>1.00</td><td>0.61</td><td>0.44</td><td>0.97</td><td>0.38</td><td>0.12</td></tr></table>

## 5 Discussion and Conclusion

Significance. We give a tiny transformer construction with hand chosen parameters that evaluates arbitrary-depth circuits via reduction. Empirically, we show that when we randomly initialize our parameterization, models achieve perfect length generalization. Transformers often fail at symbolic generalization despite strong empirical performance. Recent architectural variants improve length generalization but remain imperfect [Cho et al., 2025, Zhou et al., 2024, Fan et al., 2025, Kazemnejad et al., 2023, Hou et al., 2024], suggesting a lack of faithful algorithmic implementation. This empirical gap contrasts with the theoretical results showing that these models are universal in principle [Pérez et al., 2021, Merrill and Sabharwal, 2024]. Yet, such results are largely non-constructive and provide limited insight into how exact algorithmic computation may be mechanistically realized or learned in practice.

We instead give a simple and interpretable construction, repurposing existing architectural primitives used in the literature to implement a structured algorithmic process – a variant of contextual PE [Golovneva et al., 2024] that delimits positions based on token occurrences, linear attention mechanisms [Katharopoulos et al., 2020], gated residuals as a variation on hyperconnections [Zhu et al., 2025], and straight-through estimators throughout to simulate discrete algorithmic representations [Bengio et al., 2013, van den Oord et al., 2017]. Experiments further show this behavior is learnable: dense random models acquire ternarized PE gating $( G _ { \mathrm { P E } } )$ , binary attention, and lookup-table MLPs, enabling exact in-context circuit evaluation.

Related work. Compositional and length generalization: Generalization to greater compositional complexity remains challenging [Ito et al., 2025, Hupkes et al., 2020, Lake and Baroni, 2018, Zhou et al., 2024, Saxton et al., 2019, Kim and Linzen, 2020, Dziri et al., 2023]. Prior work improves extrapolation via ad hoc architectural modifications, especially PE in transformers [Csordás et al., 2021, Kazemnejad et al., 2023, Ruoss et al., 2023, Shen et al., 2024, Li et al., 2024, McLeish et al., 2024, Cho et al., 2025, Jelassi et al., 2023], scratchpads [Cho et al., 2025, Nye et al., 2021], looped/universal transformers [Fan et al., 2025, Giannou et al., 2023], and program compilation approaches [Weiss et al., 2021, Shaw et al., 2024]. While these improve empirical extrapolation, they lack guarantees of faithful computation at arbitrary depth or are not differentiable. Complementary theoretical analyses characterize transformer expressivity limits [Strobl et al., 2024, Merrill and Sabharwal, 2025, Huang et al., 2025, Amiri et al., 2025, Yang et al., 2024, Deletang et al., 2022], but rarely provide actual constructions for exact computation. Recursive and structured neural architectures: A complementary line of work composes representations along latent tree structures, including recursive neural networks and their continuous relaxations such as CRvNN [Chowdhury and Caragea, 2021] (which we compare to directly, see Section 4.1) and beam-tree recursion [Ray Chowdhury and Caragea, 2023], together with the Neural Data Router [Csordás et al., 2022]; Chowdhury and Caragea [2024] lay out the design space between transformers and recursive nets. These models learn composition order through continuous probabilities, achieving strong but imperfect length generalization. Our construction instead occupies a maximally constrained point in this space – deterministic composition order, hard gating, and adaptive halting with a correctness proof – trading flexibility for exactness. Graph neural networks and transformers: Graph neural networks (GNNs) naturally model circuits via message passing over nodes and edges [Scarselli et al., 2009], and have been studied for algorithmic reasoning [Selsam et al., 2019, Xu et al., 2019, Loukas, 2020]. However, standard GNNs require depth proportional to graph diameter and suffer from oversmoothing and oversquashing [Alon and Yahav, 2021, Gravina et al., 2025]. Graph transformers alleviate this with global attention mechanisms [Dwivedi and Bresson, 2021, Rampášek et al., 2023, Ying et al., 2021], but require $O ( n ^ { 2 } )$ cost and often rely on explicit predfined knowledge of the graph structure, rather than learning and computing it on-the-fly as we do here. In contrast, our approach uses learned PE gating and structured attention routing to simulate circuit reduction directly. Mechanistic interpretability: Mechanistic interpretability aims to uncover circuits underlying model behavior [Sharkey et al., 2025, Olah et al., 2020, Olsson et al., 2022, Elhage et al., 2021, Adler et al., 2025]. While instructive, these analyses are typically performed post hoc (after learning) and lack guarantees of exact computation.

Limitations. Our results rely on several assumptions that limit the scope of the current construction while highlighting directions for future work. First, we assume inputs are fully parenthesized, well-formed expressions, enabling the identification of reducible subexpressions via locally-derived masks. Although this setting captures a broad class of algorithmic tasks, it excludes problems where structure must be inferred implicitly. Fully parenthesized expressions form a bracketed, context-free (Dyck) language whose parse is explicit in the surface string. The model therefore counts brackets rather than inferring the structure. Removing the brackets makes the structure latent – a structure-induction problem that changes the premise of our theorems – and natural language, in particular, is generally regarded as beyond context-free [Shieber]. Nevertheless, the core components of our construction – token-dependent positional gating, subexpression masking, and type-constrained attention routing – provide key ingredients for extending the approach to settings where circuit structure must be learned without parentheses or brackets. Second, the model evaluates expressions through iterative local reductions and, therefore, does not directly support algorithms with inherently global sequential dependencies such as long addition with carry [Dziri et al., 2023, Cho et al., 2025]. Addressing this limitation may require alternative routing mechanisms or an explicit representation of the underlying carry circuit structure, which can still be expressed combinatorially (e.g., Fig. 1 of Dziri et al. [2023]). Third, our construction also focuses on tasks with a finite fan-in circuit (≤ 3). Although higher fan-in operations can be reduced to binary compositions, doing so may increase depth and obscure the local semantic structure leveraged by our model. Designing architectures that support general fan-in while preserving interpretability and generalization remains an important direction for future work. Finally, our exactness guarantees rely on the discrete machinery of the construction: hardmax attention masking, discrete token states maintained by straight-through estimators, deterministic halting when no parentheses remain, and exact one-hot arithmetic. Nevertheless, the learned models remarkably realize these operations with finite-precision relaxations, where straight-through estimators recover the discrete solution.

Conclusion. We present a single-layer, universal transformer that implements circuit computations via circuit reduction, achieving perfect length generalization on Boolean algebra, modular arithmetic, and ListOps evaluation. We demonstrate that the construction is learnable, requiring only 924 (arithmetic), 280 (Boolean), and 4416 (ListOps) parameters. Importantly, this ability is enabled through a novel PE gating mechanism, which identifies reducible subexpressions by tracking problem depth. This mechanism induces hard attention masking (i.e., scoping) over the input sequence, enforcing structured information flow within the transformer. It will be important for future work to explore how such hard masking mechanisms can be learned to adapt dynamically, enabling transformers to discover latent structural boundaries (i.e., structure induction) in more complex, unstructured domains.

## References

M. Adler, D. Alistarh, and N. Shavit. Towards Combinatorial Interpretability of Neural Computation, May 2025. URL http://arxiv.org/abs/2504.08842. arXiv:2504.08842 [cs].

U. Alon and E. Yahav. On the Bottleneck of Graph Neural Networks and its Practical Implications, Mar. 2021. URL http://arxiv.org/abs/2006.05205. arXiv:2006.05205 [cs].

A. Amiri, X. Huang, M. Rofin, and M. Hahn. Lower Bounds for Chain-of-Thought Reasoning in Hard-Attention Transformers, Feb. 2025. URL http://arxiv.org/abs/2502.02393. arXiv:2502.02393 [cs].

S. Arora and B. Barak. Computational complexity: a modern approach. Cambridge University Press, 2009. URL https://books.google.com/books?hl=en&lr=&id=nGvI7cOuOOQC&oi=fnd&pg=PA9&dq= computational+complexity&ots=Dbc7zGdDry&sig=ZA5nA3Cq8n5hZSkjaxA-cCxvhMc.

Y. Bengio, N. Léonard, and A. Courville. Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation, Aug. 2013. URL http://arxiv.org/abs/1308.3432. arXiv:1308.3432 [cs].

H. Cho, J. Cha, S. Bhojanapalli, and C. Yun. Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count, Apr. 2025. URL http://arxiv.org/abs/2410.15787. arXiv:2410.15787 [cs].

J. R. Chowdhury and C. Caragea. Modeling hierarchical structures with continuous recursive neural networks. In International Conference on Machine Learning, pages 1975–1988. PMLR, 2021.

J. R. Chowdhury and C. Caragea. On the design space between transformers and recursive neural nets. arXiv preprint arXiv:2409.01531, 2024.

R. Csordás, K. Irie, and J. Schmidhuber. The Devil is in the Detail: Simple Tricks Improve Systematic Generalization of Transformers. In M.-F. Moens, X. Huang, L. Specia, and S. W.-t. Yih, editors, Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 619–634, Online and Punta Cana, Dominican Republic, Nov. 2021. Association for Computational Linguistics. doi:10.18653/v1/2021.emnlp-main.49. URL https://aclanthology.org/2021.emnlp-main.49.

R. Csordás, K. Irie, and J. Schmidhuber. The Neural Data Router: Adaptive Control Flow in Transformers Improves Systematic Generalization, May 2022. URL http://arxiv.org/abs/2110.07732. arXiv:2110.07732 [cs].

G. Deletang, A. Ruoss, J. Grau-Moya, T. Genewein, L. K. Wenliang, E. Catt, C. Cundy, M. Hutter, S. Legg, J. Veness, and P. A. Ortega. Neural Networks and the Chomsky Hierarchy. International Conference on Learning Representations, Sept. 2022. URL https://openreview.net/forum?id=WbxHAzkeQcn.

V. P. Dwivedi and X. Bresson. A Generalization of Transformer Networks to Graphs, Jan. 2021. URL http: //arxiv.org/abs/2012.09699. arXiv:2012.09699 [cs].

N. Dziri, X. Lu, M. Sclar, X. L. Li, L. Jiang, B. Y. Lin, S. Welleck, P. West, C. Bhagavatula, R. Le Bras, J. Hwang, S. Sanyal, X. Ren, A. Ettinger, Z. Harchaoui, and Y. Choi. Faith and Fate: Limits of Transformers on Compositionality. Advances in Neural Information Processing Systems, 36: 70293–70332, Dec. 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/hash/ deb3c28192f979302c157cb653c15e90-Abstract-Conference.html.

N. Elhage, N. Nanda, C. Olsson, T. Henighan, N. Joseph, B. Mann, A. Askell, Y. Bai, A. Chen, and T. Conerly. A mathematical framework for transformer circuits. Transformer Circuits Thread, 1(1):12, 2021.

Y. Fan, Y. Du, K. Ramchandran, and K. Lee. Looped Transformers for Length Generalization, May 2025. URL http://arxiv.org/abs/2409.15647. arXiv:2409.15647 [cs].

A. Giannou, S. Rajput, J.-y. Sohn, K. Lee, J. D. Lee, and D. Papailiopoulos. Looped Transformers as Programmable Computers, Jan. 2023. URL http://arxiv.org/abs/2301.13196. arXiv:2301.13196 [cs].

O. Golovneva, T. Wang, J. Weston, and S. Sukhbaatar. Contextual Position Encoding: Learning to Count What’s Important, May 2024. URL http://arxiv.org/abs/2405.18719. arXiv:2405.18719 [cs].

A. Gravina, M. Eliasof, C. Gallicchio, D. Bacciu, and C.-B. Schönlieb. On Oversquashing in Graph Neural Networks Through the Lens of Dynamical Systems, Feb. 2025. URL http://arxiv.org/abs/2405.01009. arXiv:2405.01009 [cs].

T. He, D. Doshi, A. Das, and A. Gromov. Learning to grok: Emergence of in-context learning and skill composition in modular arithmetic tasks. Nov. 2024. URL https://openreview.net/forum?id=aVh9KRZdRk.

K. Hornik, M. Stinchcombe, and H. White. Multilayer feedforward networks are universal approximators. Neural networks, 2(5):359–366, 1989. URL https://www.sciencedirect.com/science/article/pii/ 0893608089900208.

K. Hou, D. Brandfonbrener, S. Kakade, S. Jelassi, and E. Malach. Universal Length Generalization with Turing Programs, July 2024. URL http://arxiv.org/abs/2407.03310. arXiv:2407.03310 [cs].

X. Huang, A. Yang, S. Bhattamishra, Y. Sarrof, A. Krebs, H. Zhou, P. Nakkiran, and M. Hahn. A Formal Framework for Understanding Length Generalization in Transformers, May 2025. URL http://arxiv.org/abs/2410.02140. arXiv:2410.02140 [cs].

D. Hupkes, V. Dankers, M. Mul, and E. Bruni. Compositionality Decomposed: How do Neural Networks Generalise? Journal ofArtificial Intelligence Research, 67:757–795, Apr. 2020. ISSN 1076-9757. doi:10.1613/jair.1.11674. URL https://www.jair.org/index.php/jair/article/view/11674.

T. Ito, M. Campbell, L. Horesh, T. Klinger, and P. Ram. Quantifying artificial intelligence through algorithmic generalization. Nature Machine Intelligence, 7(8):1195–1205, Aug. 2025. ISSN 2522-5839. doi:10.1038/s42256- 025-01092-w. URL https://www.nature.com/articles/s42256-025-01092-w.

S. Jelassi, S. d’Ascoli, C. Domingo-Enrich, Y. Wu, Y. Li, and F. Charton. Length Generalization in Arithmetic Transformers, June 2023. URL http://arxiv.org/abs/2306.15400. arXiv:2306.15400 [cs].

A. Katharopoulos, A. Vyas, N. Pappas, and F. Fleuret. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention, Aug. 2020. URL http://arxiv.org/abs/2006.16236. arXiv:2006.16236 [cs].

A. Kazemnejad, I. Padhi, K. Natesan Ramamurthy, P. Das, and S. Reddy. The Impact of Positional Encoding on Length Generalization in Transformers. Advances in Neural Information Processing Systems,

36:24892–24928, Dec. 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/hash/ 4e85362c02172c0c6567ce593122d31c-Abstract-Conference.html.

N. Kim and T. Linzen. COGS: A Compositional Generalization Challenge Based on Semantic Interpretation. In B. Webber, T. Cohn, Y. He, and Y. Liu, editors, Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 9087–9105, Online, Nov. 2020. Association for Computational Linguistics. doi:10.18653/v1/2020.emnlp-main.731. URL https://aclanthology.org/2020.emnlp-main.731.

B. Lake and M. Baroni. Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks. In International Conference on Machine Learning, pages 2873–2882. PMLR, July 2018. URL http://proceedings.mlr.press/v80/lake18a.html.

S. Li, C. You, G. Guruganesh, J. Ainslie, S. Ontanon, M. Zaheer, S. Sanghai, Y. Yang, S. Kumar, and S. Bhojanapalli. Functional Interpolation for Relative Positions Improves Long Context Transformers, Mar. 2024. URL http: //arxiv.org/abs/2310.04418. arXiv:2310.04418 [cs].

A. Loukas. What graph neural networks cannot learn: depth vs width, Jan. 2020. URL http://arxiv.org/abs/ 1907.03199. arXiv:1907.03199 [cs].

S. McLeish, A. Bansal, A. Stein, N. Jain, J. Kirchenbauer, B. R. Bartoldson, B. Kailkhura, A. Bhatele, J. Geiping, A. Schwarzschild, and T. Goldstein. Transformers Can Do Arithmetic with the Right Embeddings, May 2024. URL http://arxiv.org/abs/2405.17399. arXiv:2405.17399 [cs].

W. Merrill and A. Sabharwal. The Expressive Power of Transformers with Chain of Thought. International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=NjNGlPh8Wh.

W. Merrill and A. Sabharwal. A Little Depth Goes a Long Way: The Expressive Power of Log-Depth Transformers, Mar. 2025. URL http://arxiv.org/abs/2503.03961. arXiv:2503.03961 [cs].

N. Nangia and S. R. Bowman. ListOps: A Diagnostic Dataset for Latent Tree Learning, Apr. 2018. URL http: //arxiv.org/abs/1804.06028. arXiv:1804.06028 [cs].

M. Nye, A. J. Andreassen, G. Gur-Ari, H. Michalewski, J. Austin, D. Bieber, D. Dohan, A. Lewkowycz, M. Bosma, and D. Luan. Show your work: Scratchpads for intermediate computation with language models. 2021. URL https://openreview.net/forum?id=iedYJm92o0a&ref=morioh.com&utm\_source=morioh.com.

C. Olah, N. Cammarata, L. Schubert, G. Goh, M. Petrov, and S. Carter. Zoom in: An introduction to circuits. Distill, 5 (3):e00024–001, 2020. URL https://distill.pub/2020/circuits/zoom-in/?ref=cold-takes.

C. Olsson, N. Elhage, N. Nanda, N. Joseph, N. DasSarma, T. Henighan, B. Mann, A. Askell, Y. Bai, A. Chen, T. Conerly, D. Drain, D. Ganguli, Z. Hatfield-Dodds, D. Hernandez, S. Johnston, A. Jones, J. Kernion, L. Lovitt, K. Ndousse, D. Amodei, T. Brown, J. Clark, J. Kaplan, S. McCandlish, and C. Olah. In-context Learning and Induction Heads, Sept. 2022. URL http://arxiv.org/abs/2209.11895. arXiv:2209.11895 [cs].

J. Pérez, P. Barceló, and J. Marinkovic. Attention is Turing-Complete. Journal of Machine Learning Research, 22(75): 1–35, 2021. ISSN 1533-7928. URL http://jmlr.org/papers/v22/20-302.html.

L. Rampášek, M. Galkin, V. P. Dwivedi, A. T. Luu, G. Wolf, and D. Beaini. Recipe for a General, Powerful, Scalable Graph Transformer, Jan. 2023. URL http://arxiv.org/abs/2205.12454. arXiv:2205.12454 [cs].

J. Ray Chowdhury and C. Caragea. Efficient Beam Tree Recursion. Advances in Neural Information Processing Systems, 36:29126–29148, Dec. 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/ hash/5cf93940e37f7a7877cd57b6dba6b7ab-Abstract-Conference.html.

A. Ruoss, G. Delétang, T. Genewein, J. Grau-Moya, R. Csordás, M. Bennani, S. Legg, and J. Veness. Randomized Positional Encodings Boost Length Generalization of Transformers. In A. Rogers, J. Boyd-Graber, and N. Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 1889–1903, Toronto, Canada, July 2023. Association for Computational Linguistics. doi:10.18653/v1/2023.acl-short.161. URL https://aclanthology.org/2023.acl-short.161.

D. Saxton, E. Grefenstette, F. Hill, and P. Kohli. Analysing Mathematical Reasoning Abilities of Neural Models. Interna tional Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=H1gR5iR5FX.

F. Scarselli, M. Gori, A. C. Tsoi, M. Hagenbuchner, and G. Monfardini. The Graph Neural Network Model. IEEE Transactions on Neural Networks, 20(1):61–80, Jan. 2009. ISSN 1941-0093. doi:10.1109/TNN.2008.2005605. URL https://ieeexplore.ieee.org/abstract/document/4700287.

D. Selsam, M. Lamm, B. Bünz, P. Liang, L. d. Moura, and D. L. Dill. Learning a SAT Solver from Single-Bit Supervision, Mar. 2019. URL http://arxiv.org/abs/1802.03685. arXiv:1802.03685 [cs].

L. Sharkey, B. Chughtai, J. Batson, J. Lindsey, J. Wu, L. Bushnaq, N. Goldowsky-Dill, S. Heimersheim, A. Ortega, J. Bloom, S. Biderman, A. Garriga-Alonso, A. Conmy, N. Nanda, J. Rumbelow, M. Wattenberg, N. Schoots, J. Miller, E. J. Michaud, S. Casper, M. Tegmark, W. Saunders, D. Bau, E. Todd, A. Geiger, M. Geva, J. Hoogland, D. Murfet, and T. McGrath. Open Problems in Mechanistic Interpretability, Jan. 2025. URL http://arxiv.org/abs/2501. 16496. arXiv:2501.16496 [cs].

P. Shaw, J. Cohan, J. Eisenstein, K. Lee, J. Berant, and K. Toutanova. ALTA: Compiler-Based Analysis of Transformers. Transactions on Machine Learning Research, Nov. 2024. ISSN 2835-8856. URL https://openreview.net/ forum?id=h751wl9xiR.

R. Shen, S. Bubeck, R. Eldan, Y. T. Lee, Y. Li, and Y. Zhang. Positional Description Matters for Transformers Arithmetic. International Conference on Learning Representations, 2024. URL https://openreview.net/ forum?id=ZMuPAOY8Oz.

S. M. Shieber. Evidence against the context-freeness of natural language. In The Formal complexity of natural language, pages 320–334. Springer.

L. Strobl, D. Angluin, D. Chiang, J. Rawski, and A. Sabharwal. Transformers as Transducers. Transactions of the Associationfor Computational Linguistics, 13:200–219, Feb. 2024. ISSN 2307-387X. doi:10.1162/tacl\_a\_00736. URL https://doi.org/10.1162/tacl\_a\_00736.

J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, and Y. Liu. RoFormer: Enhanced Transformer with Rotary Position Embedding, Aug. 2022. URL http://arxiv.org/abs/2104.09864. arXiv:2104.09864 [cs].

A. van den Oord, O. Vinyals, and k. kavukcuoglu. Neural Discrete Representation Learning. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips. cc/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html.

G. Weiss, Y. Goldberg, and E. Yahav. Thinking Like Transformers. In Proceedings ofthe 38th International Conference on Machine Learning, pages 11080–11090. PMLR, July 2021. URL https://proceedings.mlr.press/v139/ weiss21a.html.

K. Xu, W. Hu, J. Leskovec, and S. Jegelka. How Powerful are Graph Neural Networks?, Feb. 2019. URL http: //arxiv.org/abs/1810.00826. arXiv:1810.00826 [cs].

A. Yang, D. Chiang, and D. Angluin. Masked Hard-Attention Transformers Recognize Exactly the Star-Free Languages, Oct. 2024. URL http://arxiv.org/abs/2310.13897. arXiv:2310.13897 [cs].

C. Ying, T. Cai, S. Luo, S. Zheng, G. Ke, D. He, Y. Shen, and T.-Y. Liu. Do Transformers Really Perform Badly for Graph Representation? In Advances in Neural Information Processing Systems, volume 34, pages 28877–28888. Curran Associates, Inc., 2021. URL https://proceedings.neurips.cc/paper/2021/hash/ f1c1592588411002af340cbaedd6fc33-Abstract.html.

H. Zhou, A. Bradley, E. Littwin, N. Razin, O. Saremi, J. M. Susskind, S. Bengio, and P. Nakkiran. What Algorithms can Transformers Learn? A Study in Length Generalization. International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=AssIuHnmHX.

D. Zhu, H. Huang, Z. Huang, Y. Zeng, Y. Mao, B. Wu, Q. Min, and X. Zhou. Hyper-Connections, Mar. 2025. URL http://arxiv.org/abs/2409.19606. arXiv:2409.19606 [cs].

## A Model construction details

## A.1 Computational complexity of model construction

Theorem 1 characterizes the cost of a single universal-transformer iteration. For an input of length n, reduction mask construction requires parenthesis counting and thresholding over the sequence, which is linear in n (Lemma 1). Subexpression decomposition is likewise linear in n, since it consists of a discrete difference and cumulative sum over the binary mask (Lemma 2). The dominant cost is the typed routing step, whose linear-attention implementation scales as $O ( k n )$ , where k is the number of parallel reducible subexpressions. For the parenthesized expressions considered here – where $k \ll n -$ the effective cost is near-linear in n. Thus, the dominant cost per iteration is $O ( n )$ , aside from storing the fixed routing parameters (e.g., attention head dimension $d _ { h } )$

By Theorem 2, the number of transformer iterations required to evaluate an expression x requires exactly $d ( x )$ iterations, where $d ( x )$ is the depth of the original expression. Because each additional level of nesting introduces at least one pair of parentheses and one operator, $d ( x ) \leq n / 3$ . This implies that the total cost $O ( n \cdot d ( x ) )$ is always sub-quadratic; the worst case $O ( n ^ { 2 } )$ arises only for maximally unbalanced expressions with $d ( x ) = \Theta ( n )$ . For balanced expressions $d ( x ) = O ( \log n )$ , giving an effective total cost of $O ( n \log n )$ . In this sense, the construction has lower sequential depth

A - Construction

![](images/d1b4a2d5a739b2747940fd9b0fbc8d7c1531fe506504e29e2e50ad96c934ebd0.jpg)

B - Random initialization (seed 7520)  
![](images/1cec82089eeae03c5699fcaaa4de33656f0afae07684335c7a610cd4472f9322.jpg)

C - Learned solution (seed 7520)  
![](images/f1d6da3909fd3c3619bbd0daba4e45e86710bed42419345bf69ea3c577ff7ba9.jpg)  
Figure A.1: Visualization of all Boolean model parameters for the constructed (formal) solution, a randomly initialized model (before training), and the learned model that exhibits perfect depth generalization. Note that for the randomly initialized and learned model, we visualize both the raw logits, as well as the parameter after applying the threshold for binarization/ternarization. MLP bias terms are not visualized here. A) Our construction of the model that implements circuit reduction, as specified in Theorem 1-2. Here we visualize a learned MLP that is trained on a Boolean look-up table (the 8 depth-1 Boolean trees). (MLP refers to the input-to-hidden weights, and MLP refers to the hidden-tooutput mapping.) B) A randomly initialized model before training for a single seed (7520). Note that logits are typically initialized near or below the straight-through estimator thresholds. C) The learned model after training (seed 7520) Note that while the model does not learn the exact same parameterization as our constructed solution, this model still exhibits perfect generalization.

than standard sequential token-by-token evaluation procedures, such as stack-based parsing or left-to-right symbolic evaluation, which typically require Θ(n) sequential steps on an input of length n (e.g., see Deletang et al. [2022]).

## A.2 Additional proofs for model construction

Lemma 4 (Feedforward lookup table). There exists a function $f : \mathbb { Z } ^ { V } \to \{ 0 , 1 \} ^ { D }$ such that, when applied to the bag-of-word token representation ofany valid reducible subexpression $C _ { c } ^ { ( t ) }$ , it outputs the correct one-hot embedding ofthe resulting subexpression, and a 0 vector otherwise.

Table 3: Model Performance Summary on all 3 tasks fo every depth. Note that depth 1 and 2 are in the training distribution.
<table><tr><td rowspan="2"></td><td colspan="2">Arithmetic</td><td colspan="2">Boolean</td><td colspan="2">ListOps</td></tr><tr><td>Depth Acc.</td><td>Tokens</td><td>Acc.</td><td>Tokens</td><td> $\operatorname { A c c } .$ </td><td>Tokens</td></tr><tr><td>1</td><td>1.000</td><td>5.00</td><td>1.000</td><td>4.63</td><td>1.000</td><td>5.00</td></tr><tr><td>2</td><td>1.000</td><td>9.64</td><td>1.000</td><td>9.00</td><td>1.000</td><td>11.28</td></tr><tr><td>3</td><td>1.000</td><td>16.84</td><td>1.000</td><td>14.29</td><td>1.000</td><td>19.56</td></tr><tr><td>4</td><td>1.000</td><td>25.96</td><td>1.000</td><td>21.44</td><td>1.000</td><td>31.12</td></tr><tr><td>5</td><td>1.000</td><td>36.92</td><td>1.000</td><td>28.90</td><td>1.000</td><td>50.52</td></tr><tr><td>6</td><td>1.000</td><td>54.76</td><td>1.000</td><td>37.25</td><td>1.000</td><td>70.52</td></tr><tr><td>7</td><td>1.000</td><td>74.32</td><td>1.000</td><td>50.54</td><td>1.000</td><td>102.36</td></tr><tr><td>8</td><td>1.000</td><td>97.64</td><td>1.000</td><td>66.31</td><td>1.000</td><td>146.80</td></tr><tr><td>9</td><td>1.000</td><td>127.56</td><td>1.000</td><td>82.78</td><td>1.000</td><td>178.92</td></tr><tr><td>10</td><td>1.000</td><td>173.12</td><td>1.000</td><td>110.18</td><td>1.000</td><td>259.76</td></tr><tr><td>11</td><td>1.000</td><td>218.88</td><td>1.000</td><td>122.81</td><td>1.000</td><td>329.64</td></tr><tr><td>12</td><td>1.000</td><td>288.32</td><td>1.000</td><td>168.45</td><td>1.000</td><td>418.76</td></tr><tr><td>13</td><td>1.000</td><td>316.32</td><td>1.000</td><td>191.55</td><td>1.000</td><td>559.48</td></tr><tr><td>14</td><td>1.000</td><td>459.72</td><td>1.000</td><td>224.85</td><td>1.000</td><td>727.52</td></tr><tr><td>15</td><td>1.000</td><td>570.84</td><td>1.000</td><td>295.72</td><td>1.000</td><td>961.68</td></tr><tr><td>16</td><td>1.000</td><td>665.28</td><td>1.000</td><td>311.39</td><td>1.000</td><td>1086.52</td></tr><tr><td>17</td><td>1.000</td><td>896.12</td><td>1.000</td><td>421.03</td><td>1.000</td><td>1493.24</td></tr><tr><td>18</td><td>1.000</td><td>1025.92</td><td>1.000</td><td>458.13</td><td>1.000</td><td>1722.08</td></tr><tr><td>19</td><td>1.000</td><td>1254.52</td><td>1.000</td><td>535.26</td><td>1.000</td><td>2239.52</td></tr><tr><td>20</td><td>1.000</td><td>1572.56</td><td>1.000</td><td>659.82</td><td>1.000</td><td>2746.20</td></tr><tr><td>21</td><td>1.000</td><td>1892.24</td><td>1.000</td><td>749.34</td><td>1.000</td><td>3584.56</td></tr><tr><td>22</td><td>1.000</td><td>2130.96</td><td>1.000</td><td>919.54</td><td>1.000</td><td>4100.28</td></tr><tr><td>23</td><td>1.000</td><td>2654.28</td><td>1.000</td><td>1057.65</td><td>1.000</td><td>5199.72</td></tr><tr><td>24</td><td>1.000</td><td>3022.92</td><td>1.000</td><td>1217.01</td><td>1.000</td><td>6126.32</td></tr><tr><td>25</td><td>1.000</td><td>4068.36</td><td>1.000</td><td>1496.06</td><td>1.000</td><td>7593.32</td></tr><tr><td>26</td><td>1.000</td><td>4560.12</td><td>1.000</td><td>1553.46</td><td>1.000</td><td>8747.92</td></tr><tr><td>27</td><td>1.000</td><td>5484.48</td><td>1.000</td><td>1859.65</td><td>1.000</td><td>11437.36</td></tr><tr><td>28</td><td>1.000</td><td>6442.20</td><td>1.000</td><td>1954.27</td><td>1.000</td><td>14215.60</td></tr><tr><td>29</td><td>1.000</td><td>7620.96</td><td>1.000</td><td>2548.09</td><td>1.000</td><td>15335.56</td></tr><tr><td>30</td><td>1.000</td><td>8509.00</td><td>1.000</td><td>2748.17</td><td>1.000</td><td>18375.68</td></tr></table>

Proof. The set of valid reducible subexpressions over fan-in-2 Boolean trees is finite; enumeration yields exactly 8 distinct cases $( \land , \lor , \sim$ operators applied to 0, 1 operands. We list these explicitly as input-output pairs $( \mathbf { b } _ { i } , \mathbf { y } _ { i } ) _ { i = 1 } ^ { 8 } .$ where $\mathbf { b } _ { i } \in \mathbb { Z } ^ { D }$ is the bag-of-words representation of the i-th subexpression and $\mathbf { y } _ { i } \in \{ 0 , 1 \} ^ { D }$ is its evaluated output embedding. Any input b not matching a valid reducible expression maps to $\mathbf { 0 } ^ { D }$

Since the domain $\{ \mathbf { b } _ { 1 } , . . . , \mathbf { b } _ { 8 } \}$ is finite, $f$ is simply a lookup table on $^ 8$ entries. By the universal approximation theorem Hornik et al. [1989], any function on a finite domain is exactly realizable by a single-hidden-layer MLP with sufficiently many units and a suitable activation function. Hence, such an f exists. □

Lemma 5 (Gated residual connection). Let $P ^ { ( t ) } \in \{ 0 , 1 \} ^ { n }$ be the reduction mask specified in Lemma 1 and $M ^ { ( t ) } =$ $\mathbf { 1 } - P ^ { ( t ) }$ its complement. Define the gated residual update

$$
\boldsymbol { X } ^ { ( t + 1 ) } = \boldsymbol { M } ^ { ( t ) } \odot \boldsymbol { X } ^ { ( t ) } + \boldsymbol { P } ^ { ( t ) } \odot \boldsymbol { Y } ^ { ( t ) } .\tag{13}
$$

Then every position outside the reduction mask is preserved, and only positions inside the reduction mask are updated.

Proof. By Lemma $1 , P ^ { ( t ) }$ denotes the mask of reducible subexpressions in $X ^ { ( t ) }$ , and let $Y ^ { ( t ) } \in \{ 0 , 1 \} ^ { n \times V }$ denote the sequence output vector after the feedforward layer (Lemma 4). For each position i,

$$
X _ { i } ^ { ( t + 1 ) } = M _ { i } ^ { ( t ) } X _ { i } ^ { ( t ) } + P _ { i } ^ { ( t ) } Y _ { i } ^ { ( t ) } .
$$

Since $M _ { i } ^ { ( t ) } = 1 - P _ { i } ^ { ( t ) }$ and $P _ { i } ^ { ( t ) } \in \{ 0 , 1 \}$ , exactly one of $M _ { i } ^ { ( t ) }$ or $P _ { i } ^ { ( t ) }$ equals 1. Thus, if $P _ { i } ^ { ( t ) } = 0 ;$ , then $X _ { i } ^ { ( t + 1 ) } = X _ { i } ^ { ( t ) }$ and if $P _ { i } ^ { ( t ) } = 1$ , then $X _ { i } ^ { ( t + 1 ) } = Y _ { i } ^ { ( t ) }$ . Thus, positions outside the reduction mask are preserved and those inside are updated. □

Theorem 1 (Depth-1 circuit reduction). Let F be one transformer iteration $( P E ,$ , attention, MLP, and gated residual; Lemmas 1–5). For any valid $X ^ { ( t ) }$ with active depth $\delta ( X ^ { ( t ) } ) > 0 ,$ , we construct F such that $X ^ { ( t + \bar { 1 } ) } = F ( X ^ { ( t ) } )$ simultaneously evaluates every reducible subexpression, replaces each by its correct value, and preserves all inactive positions. Then,

$$
X ^ { ( t + 1 ) } \equiv X ^ { ( t ) } \qquad a n d \qquad \delta ( X ^ { ( t + 1 ) } ) = \delta ( X ^ { ( t ) } ) - 1 .
$$

Proof. We show that one iteration evaluates all deepest reducible subexpressions while preserving all other tokens. By Lemma 1, the reduction mask identifies exactly the deepest reducible subexpressions. By Lemma 2, parallel subexpression decomposition isolates independent subexpressions. By Lemma 3, the attention mechanism takes each independent subexpression, and sums together local operand and operator embeddings at the operator position creating a bag-of-words vector embedding. Tokens across independent subexpressions do not interfere with subexpressions, because they are disjoint subsequences (Lemma 2). By Lemma 4, the bag-of-words embedding vector at the operator’s token position is evaluated to its semantic value; all other tokens within that subexpression are set to the 0 vector. By Lemma 5 the residual update preserves all remaining non-reduced positions. Since the MLP reduces all reducible subexpressions (Lemma 4) and the gated residual connection preserves all other tokens (Lemma 5), this implies $X ^ { ( t + 1 ) } \equiv X ^ { ( t ) }$ . Since only the deepest subexpressions are reduced, all nodes at maximal depth are eliminated and no deeper nodes remain (parentheses of the deepest subexpressions are replaced by 0 vectors by Lemma 4). This yields $\delta ( \bar { X ^ { ( t + 1 ) } } ) = \delta ( X ^ { ( t ) } ) \bar { - } 1$ . Hence one iteration performs a parallel local reduction step, preserving the global evaluation and decreasing active depth by one. □

Theorem 2 (Global circuit evaluation). For any valid Boolean expression x of depth $d ( x )$ , with $X ^ { ( t + 1 ) } = F ( X ^ { ( t ) } )$ and $X ^ { ( 0 ) }$ as its embedding,

$$
X ^ { ( d ( x ) ) } \equiv [ [ x ] ] \quad a n d \quad \delta ( X ^ { ( d ( x ) } ) = 0
$$

Proof. We apply Theorem 1 inductively on t. $\mathrm { A t } \ t = 0 , X ^ { ( 0 ) }$ corresponds to x, so $[ X ^ { ( 0 ) } ] \equiv [ [ X ^ { ( 0 ) } ] ]$ trivially with $\delta ( X ^ { ( 0 ) } ) = d ( x )$

Assume for $t < d ( x )$ that

$$
X ^ { ( t ) } \equiv X ^ { ( 0 ) } \quad { \mathrm { a n d } } \quad \delta ( X ^ { ( t ) } ) = d ( x ) - t .
$$

Since $\delta ( X ^ { ( t ) } ) > 0 .$ , we can apply Theorem 1, yielding

$$
X ^ { ( t + 1 ) } \equiv X ^ { ( t ) } \quad \mathrm { a n d } \quad \delta ( X ^ { ( t + 1 ) } ) = \delta ( X ^ { ( t ) } ) - 1
$$

By induction

$$
X ^ { ( t + 1 ) } \equiv X ^ { ( 0 ) } \quad { \mathrm { a n d } } \quad \delta ( X ^ { ( t + 1 ) } ) = d ( x ) - ( t + 1 )
$$

Thus, $\forall t \leq d ( x )$

$$
X ^ { ( t ) } \equiv X ^ { ( 0 ) } \quad \mathrm { a n d } \quad \delta ( X ^ { ( t ) } ) = d ( x ) - t
$$

When $t = d ( x )$ , we have $\delta ( X ^ { ( d ( x ) ) } ) = 0$ , so no reducible subexpression remains, and $X ^ { ( d ( x ) ) }$ is a Boolean atom $( \mathrm { i . e . }$ $\{ 0 , 1 \} )$ . Since semantic equivalence is preserved throughout, we have

$$
X ^ { ( d ( x ) ) } \equiv [ [ x ] ]
$$

## B Experimental Details

## B.1 Task details

Arithmetic. Each arithmetic example is a fully parenthesized expression built recursively from digits $\{ 0 , \ldots , 9 \}$ and the binary operators + and \*. The output is the numerical value of the expression modulo 10. Depth-0 instances are single digits. Depth-1 instances contain a single operation, $\mathrm { e . g . , ( 7 ~ * ~ 8 ) ~ \bar { \mapsto } ~ 6 } .$ . Larger-depth instances are formed by nesting subexpressions, $\mathrm { e . g . , ( ( 1 ~ + ~ 2 ) ~ * ~ ( \bar { 3 } ~ + ~ \bar { 4 } ) ) \mapsto 1 }$ . To illustrate the complexity of deeper problems, a depth-3 example is $( { \bar { ( } } ( 1 ~ + ~ 2 ) ~ * ~ ( 3 ~ + ~ 4 ) ) ~ + ~ ( 5 ~ * ~ ( 6 ~ + ~ 7 ) ) ) \mapsto 6$ . On average, the number of tokens exponentially increases as the problem’s depth increases.

Boolean. Each Boolean example is a recursively defined expression over leaves TRUE and FALSE, unary NOT, and binary AND/OR. The output is the truth value of the full formula. A depth-3 formula (((NOT FALSE) AND TRUE) OR (FALSE AND (NOT TRUE))) 7→ TRUE.

ListOps. Each ListOps example is a nested expression whose internal nodes are operators and whose leaves are digits 0 through 9 (see Nangia and Bowman [2018]; MIT License). The operators are MAX (maximum), MIN (minimum), SM (sum modulo 10), and MED (median). For example, ( MIN 8 3 5 ) 7→ 3, ( MAX $\begin{array} { c c } { { 1 } } & { { 9 } } & { { 4 } } \end{array} ) \mapsto \ 9$ , and ( MED 8 $2 \ 5 \ ) \mapsto 5 . \ \mathrm { A }$ nested example is ( MAX $( \texttt { M I N 8 3 5 } ) ( \texttt { S M 4 7 6 } ) )$ , which evaluates to 7, since the inner expressions yield 3 and 7, respectively. Note for our experiments, we limited the fan-in (or number of arguments per operator) to max 3. This meant randomly sampled operators could have either 2 or 3 arguments. By default, operator MED chose the floor operand for fan-in 2 (or for sequences with repeating operands).

Note that Lemma 2 assumes that independently evaluable subexpressions are non-adjacent, which holds for in-fix grammars such as Boolean formulas and modular arithmetic considered here. However, for prefix grammars, such as ListOps, subexpressions are adjacent (i.e., there is no separation between the closing parenthesis of one expression and the opening parenthesis of the next). In principle, this solution can be addressed by tokenizing whitespace – for example, even assigning a 0 vector to whitespace tokens. However, this approach doubles the total number of tokens. Thus, in practice, we instead tokenize closing parentheses using two tokens: a one-hot encoded vector followed by an adjacent 0 vector. This increases the number of input tokens in proportion to the number of closed parentheses.

## B.2 Model Training

Overview. All models were trained end-to-end from random initialization using iterative computation, where the number of iterations was specified by problem depth. Training therefore directly optimizes the full multi-step reduction procedure rather than a surrogate objective on intermediate states.

Domains. We conducted experiments across three symbolic evaluation tasks:

• Boolean logic: Expressions over the vocabulary $\Sigma _ { \mathrm { b o o l } } = \{ ( , ) , 1 , 0 , \wedge , \vee , \sim \}$ with operators $\{ \land , \lor , \sim \}$

• Modular arithmetic: Expressions over $\Sigma _ { \mathrm { a r i t h } } = \{ ( , ) , 0 , 1 , \dots , 9 , + , * \}$ evaluated modulo 10.

• ListOps: Hierarchical list operations following the benchmark of Nangia and Bowman [2018], which tests compositional reasoning over nested list structures.

Input representation and initialization. Tokens are represented as fixed one-hot vectors. In the experiments reported, the model dimension equals the vocabulary size for the corresponding task domain. The model operates directly on the one-hot representations. However, the model can similarly operate on dense input embeddings via an orthogonal embedding map. All learnable parameters are initialized randomly with small magnitude so that training begins near the discrete decision boundaries used by the straight-through estimators.

Positional encoding gate. The PE mechanism is parameterized by a learnable vector of continuous logits, one per embedding dimension. During training, these logits are converted to ternary values in $\{ - 1 , 0 , 1 \}$ using fixed thresholds of −0.5 and 0.5. In the forward pass, dimensions with logits below −0.5 are mapped to −1, while dimensions above the upper threshold are mapped $\mathbf { t o } + 1$ , and the remaining dimensions are mapped to 0. The resulting diagonal gate matrix is then used to count open and closed parentheses to identify the most deeply nested subexpressions. Importantly, when there are no remaining parentheses the model halts. Although the forward PE is discrete, gradients are passed to the underlying logits as if the quantization step were the identity, enabling standard gradient-based optimization.

Attention mechanism. The query and key matrices $W _ { Q }$ and $W _ { K }$ are learned as binary matrices using straight-through estimators. Each matrix is parameterized by a real-valued logit matrix. During training, the logits are passed through a sigmoid and thresholded at 0.5 to obtain a binary routing pattern, while gradients flow through the continuous sigmoid. This setup encourages sparse routing patterns that approximate the constructive circuit design. We fix the value matrix $W _ { V }$ to the identity, so attention only redistributes existing token information rather than learning a separate value transformation.

Feedforward layer. After attention, the resulting local count-like representation is processed by a standard two-layer MLP. This is the only dense component of the model. For Boolean and arithmetic tasks, we use a hidden dimension equal to 2D and a quadratic activation. For ListOps, we use a wider hidden layer of size 8D together with a ReLU activation. (Note here that the MLP embedding dimensions were determined by simple experimentation, to identify when the model would converge.) The role of the MLP is to map routed local representations to the semantic value of a subexpression, i.e., a finite-lookup table for each task domain. After the MLP, the model applies a gated normalization step to generate a one-hot vector that corresponds to the look-up table. In the forward pass, activations are thresholded to {0, 1}; in the backward pass, gradients are propagated through the corresponding continuous activations (another straight-through estimator). This encourages the internal token states to remain discrete symbolic representations across iterations, which is key for a single transformer iteration to implement an exact, depth-1 circuit reduction.

Straight-through vs. Gumbel-Softmax. We also experimented with a Gumbel-Softmax relaxation in place of the straight-through estimators for the discrete components. Training was considerably less stable; models did not reliably snap to the exact solution, in contrast to the straight-through estimators used throughout. A dedicated study of discrete-gradient estimators for this class of constructions would be an interesting direction for future work.

Gated residual update. We include a gated residual update, whereby non-reducible portions of the input string (i.e., tokens outside the innermost expressions) are residually added to the output of the feedforward layer. Success of this update is determined by the success of learning the correct PE gating matrix G (see 5).

Training Data and Optimization. For each domain, we generated fully enumerated training sets containing all possible valid expressions at depths 1 and 2. (However, for ListOps, because enumerating all possible depth 2 circuits was computationally inefficient due to the large combinatorial space (ListOps problems had up to 3 arguments), we opted to uniformly randomly sample new sets of depth 2 circuits after each epoch. The size of the depth-2 dataset we sampled for each epoch was 2x the size of the enumerated depth-1 task set, which was 4400 samples for all depthproblems, and thus we randomly sampled 8800 depth-2 ListOps problems.) The enumeration of depth-1 circuits ensured complete coverage of shallow compositional patterns while maintaining tractable dataset sizes. Training sets were balanced across output classes (e.g., TRUE/FALSE for Boolean, 0-9 for arithmetic) to prevent class imbalance during optimization. Models were optimized with Adam using a Binary Cross Entropy loss. Rather than calculating the loss from a single classification token, the loss was calculated across all tokens in the sequence. In particular, the target sequence was generated following the circuit reduction procedure we described in Theorem 1 – the operator token at the highest (shallowest) token position encoded the final target class, while the embeddings of all other tokens in the sequence were trained to be the 0 vector. While this required us to keep track of which token position encoded the correct target class in the training data, at test time, the max pooling of the output’s embedding dimensions of size n × D across all n tokens exactly encoded the correct one-hot token embedding. This was because non-output token positions were optimized to be the 0 vector. Across experiments, we used a learning rate of 0.01, batch size of 500, no dropout, and gradient clipping with a maximum norm of 1.0. Note that for depth 1 circuits, the number of total samples was typically fewer than 500. In those cases, batch size would reduce to the total number of enumerated depth 1 samples. We trained models for a fixed number of epochs, which were dictated by the number of total samples per circuit depth: 50 epochs for modular arithmetic, 1000 for boolean expressions, and 300 for ListOps.

Curriculum learning. We employed a progressive curriculum learning strategy. Within each epoch, the model first trained exclusively on depth-1 expressions, then on both depth-1 and depth-2 expressions together. This curriculum approach helped the model first learn the fundamental depth-1 reduction operations before tackling more complex nested structures (which were aimed at learning the PE gating parameters).

Learning rate schedule. To address vanishing gradient issues in shallow expressions (depths 1-2), we employed a two-phase learning rate schedule:

1. Warmup phase (10% of total steps): Linear warmup from 30% to 100% of the base learning rate.

2. Cosine annealing phase (remaining 90% of steps): Cosine decay from the peak learning rate to 5% of the base rate $( \eta _ { \mathrm { m i n } } = 0 . 0 0 0 5 )$ .

Computational environment. All experiments could be efficiently implemented on a CPU (Macbook). This ensured numerical stability and reproducibility. (Anecdotally, we observed that CPU training produced more consistent convergence behavior across random seeds, likely due to deterministic floating-point operations.) Each seed took no longer than 2 hours to train per seed (Boolean models converged within 2 hours for all 10 seeds). However, once learned, we evaluated models (particularly with large context inputs) on an H100 GPU, achieving perfect performance.

Random seeds. We trained 10 independent model instances per domain using the following random seeds: 8739, 7520, 1936, 7439, 6210, 5650, 5395, 3140, 1243, 5353. These seeds controlled initialization of model parameters, data shuffling, and any stochastic operations during training. Seed 1936 failed to converge for Boolean evaluation, and seed 7439 failed to converge for ListOps.

## B.3 Evaluation Protocol

Generalization testing. After training on depths 1-2, models were evaluated on held-out test sets at depths 3-30.   
These deeper expressions were never seen during training, allowing us to measure depth/length generalization.

## B.4 Code & Implementation Details

The code required to run experiments is provided in the supplementary zip file, along with a README and several python notebooks. The trained model required to produce visualizations in Fig. A.1 is also included, along with the corresponding notebook viz\_boolean\_params.ipynb.

For reproducibility, the exact command used to train models was:

python run\_transformer\_enumerated.py \   
--model\_type {boolean,arithmetic,listops} \   
--seeds 8739 7520 1936 7439 6210 5650 5395 3140 1243 5353 \   
--min\_train\_depth 1 \   
--max\_train\_depth 2 \   
--n\_epochs {1000,50,300} \   
--learning\_rate 0.01 \   
--wdecay 0.0 \   
--device cpu \   
--learnable\_pe \   
--learnable\_attn \   
--learnable\_mlp \   
--optimizer adamw \   
--train\_batch\_size 500 \   
--balanced \   
--loss bce

where model\_type was set to boolean, arithmetic, or listops for the respective experiments.

## C Comparison to standard architectures: details

We expand the comparison summarized in Section 4.1 and Table 2 in this section. The looped-transformer baseline is a standard weight-shared transformer with full softmax attention and learnable $W _ { Q } , W _ { K } , { \bf \bar { W } } _ { V }$ , trained on the same fully parenthesized inputs, the same depth-1 and depth-2 curriculum, and the same per-task epoch budget as our own models (50 epochs for arithmetic, 1000 for Boolean, 300 for ListOps) [Fan et al., 2025]. Because the baselines do not autonomously halt, we supply the ground-truth depth at test time and run exactly that many iterations. We evaluated five seeds per positional encoding (8739, 7520, 1936, 7439, 6210) and selected the best checkpoint by training accuracy.

We report accuracy conditioned on the seeds that reach ≥ 0.95 on both training depths. For arithmetic and ListOps, only one seed per positional encoding meets this criterion; for Boolean, all five seeds fit under learned and RoPE encodings, and three of five under FIRE. Visual inspection of the training trajectories showed that the non-fitting seeds are not fully converged after the fixed number of epochs. We therefore do not claim that the baseline architecture cannot fit the training depths; rather, among seeds that do fit within a matched budget, every one collapses toward chance out of distribution (Table 4).

Continuous recursive networks (CRvNN). We additionally compared to CRvNN [Chowdhury and Caragea, 2021], a continuous relaxation of recursive neural networks, on the ListOps task under the same fan-in and depth 1 and 2 training protocol. We used the original architecture unmodified (521,611 parameters, roughly 500× ours) and trained five seeds, selecting the best checkpoint by validation accuracy. CRvNN fitted the training depths but degraded out of distribution (Table 5), in contrast to the exact generalization of our construction. This mirrors the looped-transformer result: expressive, higher-capacity models fitted the shallow training depths yet failed to generalize exactly, whereas our constrained construction remains exact at every depth.

Table 4: Full generalization accuracy of the looped-transformer baseline across depths, per task and positional encoding, including only fitted seeds. Chance level is 0.10 for arithmetic and ListOps, and 0.50 for Boolean. Our construction achieves 1.0 at every depth on all three tasks (Table 1).
<table><tr><td>Positional encoding</td><td>d=1</td><td> $d { = } 2$ </td><td> $d { = } 3$ </td><td> $d { = } 5$ </td><td> $d { = } 1 0$ </td></tr><tr><td colspan="6">Arithmetic (chance 0.10)</td></tr><tr><td>Learned</td><td>1.00</td><td>1.00</td><td>0.10</td><td>0.05</td><td>0.05</td></tr><tr><td>RoPE</td><td>1.00</td><td>0.73</td><td>0.18</td><td>0.18</td><td>0.00</td></tr><tr><td>FIRE</td><td>1.00</td><td>1.00</td><td>0.23</td><td>0.13</td><td>0.03</td></tr><tr><td colspan="6">Boolean (chance 0.50)</td></tr><tr><td>Learned</td><td>1.00</td><td>1.00</td><td>0.55</td><td>0.50</td><td>0.48</td></tr><tr><td>RoPE</td><td>1.00</td><td>0.98</td><td>0.50</td><td>0.27</td><td>0.12</td></tr><tr><td>FIRE</td><td>1.00</td><td>1.00</td><td>0.61</td><td>0.48</td><td>0.44</td></tr><tr><td colspan="6">ListOps (chance 0.10)</td></tr><tr><td>Learned</td><td>1.00</td><td>0.98</td><td>0.37</td><td>0.12</td><td>0.07</td></tr><tr><td>RoPE</td><td>1.00</td><td>1.00</td><td>0.43</td><td>0.15</td><td>0.08</td></tr><tr><td>FIRE</td><td>1.00</td><td>0.97</td><td>0.38</td><td>0.20</td><td>0.12</td></tr></table>

<table><tr><td>Model</td><td> $d = 1$ </td><td> $d { = } 2$ </td><td> $d { = } 3$ </td><td> $d { = } 4$ </td><td> $d { = } 5$ </td><td> $d { = } 8$ </td><td> $d { = } 1 0$ </td></tr><tr><td>Ours</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td></tr><tr><td>CRvNN</td><td>1.000 (.000)</td><td>0.961 (.060)</td><td>0.674 (.186)</td><td>0.431 (.110)</td><td>0.302 (.065)</td><td>0.177 (.071)</td><td>0.169 (.056)</td></tr></table>

Table 5: CRvNN [Chowdhury and Caragea, 2021] versus our construction on ListOps (10-way classification, chance 0.10), reported as mean (standard deviation) over five seeds. CRvNN has 521,611 parameters. Our construction attained 1.000 at every depth.