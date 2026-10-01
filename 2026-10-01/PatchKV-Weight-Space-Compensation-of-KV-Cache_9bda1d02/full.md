# PatchKV: Weight-Space Compensation of KV Cache

Chanryeol Lee Chanhyuk Lee Yeonwoo Choi Donggyun Kim† Seunghoon Hong† KAIST

{1cy9442, chan3684, lotus\_68, kdgyun425, seunghoon.hong}@kaist.ac.kr

## Abstract

Long-context inference with Large Language Models (LLMs) is bottlenecked by the linearly growing memory of the key-value (KV) cache. Existing compression methods reduce the cache through token eviction or approximation, but degrade sharply at aggressive compression budgets. We propose PatchKV, a training-free framework that compensates KV cache compression methods by carrying part of the context in the model's weights. PatchKV pairs an off-the-shelf compressed KV cache with a context-specific weight patch, which is computed once at contextloading time and served for downstream queries for the context. The weight patch is derived in closed form via ridge regression, by aligning the block-wise activations of context-derived reference query tokens under the full cache and the compressed cache. Once merged into the model, the patch leaves the forward graph and per-query inference cost unchanged in the single-context, multi-query setting. Across long-context QA (SCBench with up to 170K tokens, SQuAD, NIAH) and math (GSM8K) benchmarks on three model architectures, PatchKV consistently improves cache compression methods, suggesting an alternative direction to compensate them at aggressive budgets.

## 1 Introduction

Many practical applications of Large Language Models (LLMs) follow a single-context, multi-query pattern [1], where a long context is read once and then queried repeatedly. This includes scenarios such as a code assistant grounding many edits in the same repository, a document analyst asking multiple questions about the same report, and a customer-support agent handling each conversation against a fixed knowledge base. In these settings, the cost of processing the context can be amortized across subsequent queries, while the latency and memory cost of each query directly affect serving efficiency.

To avoid recomputing context-token activations for every query, modern LLMs store their key and value states as a KV cache during generation, whose size grows linearly with context length [2, 3]. At long context lengths, this cache can dominate the memory footprint and memory-bandwidth cost of serving additional queries, making KV cache reduction a central tool for efficient long-context inference [4, 5]. A straightforward way to reduce this bottleneck is local context windowing, which retains only the most recent context tokens and discards distant ones. However, simply discarding distant context tokens can degrade downstream performance, since subsequent queries may require information contained in the discarded portion of the context [6, 7].

To reduce KV cache memory without simply truncating the context to a local window, recent work compresses the KV cache more selectively. Token-eviction methods discard cache entries judged less important by attention scores or other heuristics [3, 6, 8, 9], and cache-approximation methods construct a smaller surrogate cache that approximates the full one [10, 11, 12]. These methods differ in how they select or synthesize cache entries, but they primarily rely on the compressed cache to store context information after compression. This design is effective across many settings, but its accuracy can degrade as the cache budget becomes more aggressive [13, 14]. In particular, a small fixed cache may fail to retain some facts, long-range dependencies, or query-relevant details needed by future queries, especially when the relevant content is difficult to anticipate at compression time. This motivates a complementary mechanism that can recover missing context information through a channel other than the compressed cache itself.

Meanwhile, recent theoretical studies suggest that the model weights can be used as such a secondary channel for storing context information [15, 16, 17]. The results show that, under simplified settings, the behavior of attending to a fixed context can be replaced by context-dependent weight updates of the model. This perspective suggests that context information can be expressed through two coupled mechanisms: the KV cache and the model weights. However, existing work has largely focused on theoretical structure or toy settings, often with fixed query tokens and the retained context restricted to a subset of the original tokens. This leads us to explore more practical usage of the context-to-weight conversion for KV cache compression: can a weight correction purely driven by the context recover part of the information lost by compression?

We answer this question with PatchKV, a weight-based approach for compensating KV cache compression in a single-context, multi-query setting. Given a context, PatchKV first obtains an off-the-shelf compressed KV cache. It then computes a context-specific weight patch by comparing two intermediate activations on context-derived reference sequences: a full-cache teacher and a compressed-cache student. For each transformer block, we solve a closed-form ridge regression that aligns the student's block activations with those of the teacher, and we apply the resulting weight patch to the MLP output projection. Once the patch is merged into the weights for the active context, the generation-time forward graph is unchanged, thus all additional computation is paid once at context-loading time and amortized across later queries.

We empirically evaluate PatchKV with three LLM backbones and two KV cache compression methods on a benchmark suite spanning long-context QA (SCBench [1], SQuAD [18], NIAH [19]) and math reasoning (GSM8K [20]). Across token-eviction and cache-approximation compression methods, PatchKV consistently improves performance at aggressive cache budgets $( e . g . , < 2 0 \% )$ and recovers a substantial fraction of the accuracy gap between compressed-cache and full-cache inference. The results show that PatchKV can partially restore missing information in context through the weight patch, suggesting a complementary direction to memory-efficient generation in long-context settings.

Our contributions are as follows.

• Weight-space compensation of KV cache compression. We formulate context-specific weight patching as a complementary mechanism to KV cache compression, allowing context information lost by aggressive compression to be partially recovered through the model weights.

• A closed-form patch construction. We derive a gradient-free block-wise ridge regression that patches MLP output projections by matching compressed-cache student activations to full-cache teacher activations, without access to downstream queries.

• Compressor-agnostic validation. We instantiate the framework on top of both token-eviction and cache-approximation methods and evaluate its compensating behavior through extensive empirical studies in long-context QA and math benchmarks.

## 2 Method

## 2.1 Problem Formulation

We consider tasks where a single long context is prefilled once and reused across many downstream queries, such as multi-turn document QA, repeated retrieval over a database-like context, or iterative analysis of a long document. In this regime, the one-time work at context loading can be amortized over many future queries. In contrast, per-query cost such as latency, memory footprint, and memory bandwidth remain directly exposed at serving time. Since each query must attend to the long context to produce context-dependent responses, preprocessing the context so that per-query access is efficient becomes a major practical concern in the single-context, multi-query setting.

A common strategy for efficient inference is to manage a KV cache [21] of the prefilled context. Given a context token sequence $c = ( c _ { 1 } , \ldots , c _ { n } )$ of length $n ,$ a transformer-based LLM first runs a prefill pass to produce a full KV cache $\mathcal { C } = \{ ( K _ { \ell } , V _ { \ell } ) \} _ { \ell = 1 } ^ { L }$ , where $K _ { \ell }$ and $V _ { \ell }$ are the key and value activations of all context tokens at attention layer l. The cache is retained in running memory so that subsequent queries can attend to the context during autoregressive generation without recomputing the prefill. However, as the memory footprint of $\mathcal { C }$ scales linearly with n, long-context serving often requires reducing the cache before answering future queries to avoid memory overflow.

![](images/410a8c78a57a1b0fff9822e9c9c74a12d7d49d72bc81f7592b35a74f06685ca8.jpg)  
Figure 1: Overview of PatchKV. (a) Standard KV cache compression stores context information only in a compressed cache. (b) PatchKV pairs the compressed cache with a context-specific patch to the MLP output projections, compensating for part of the full-cache behavior lost during compression.

Background: KV Cache Compression. Recent KV cache compression approaches reduce the cache memory at context-prefill time either by evicting cache entries judged less important [8, 9] or by constructing a smaller cache that approximates the full one [12] (see Fig. 1(a)). Let $f _ { \theta } ( q ; \mathcal { C } )$ 1 denote the output distribution of a model with parameters θ when a query sequence $q = ( q _ { 1 } , \dots , q _ { m } )$ attends to cache ${ \mathcal { C } } ,$ and let $\mathcal L ( \cdot , \cdot )$ be a discrepancy between output distributions. An idealized cache-compression objective can be written as

$$
\operatorname* { m i n } _ { \mathcal { C } _ { b } : | \mathcal { C } _ { b } | \leq b } \ \underset { q \sim p _ { \mathrm { d o w n } } ( q | c ) } { \mathbb { E } } \Big [ \mathcal { L } \big ( f _ { \theta } ( q ; \mathcal { C } ) , f _ { \theta } ( q ; \mathcal { C } _ { b } ) \big ) \Big ] ,\tag{1}
$$

where $\mathcal { C } _ { b }$ is the compressed cache, b is the cache budget, and $p _ { \mathrm { d o w n } } ( q | c )$ is the downstream query distribution conditioned on context c. Since $p _ { \mathrm { d o w n } } ( q | c )$ is unknown at compression time, a common practice is to approximate it using proxy queries derived from the context [8, 9, 10, 12]. Once constructed, $\mathcal { C } _ { b }$ is reused across all downstream queries, reducing the per-query serving cost by a factor of $b / n$

Cache compression methods preserve context information solely through ${ \mathcal { C } } _ { b } ,$ where b must be carefully chosen to balance accuracy against efficiency. At aggressive budgets $b ,$ this cache-only bottleneck can drop details needed by future queries, and recovering them by increasing b comes at a direct per-query cost. We therefore propose to augment the compressed cache with a second channel: a patch to the model weights.

Our Approach: Compensating Compression Error via Weight Patching. To complement the compressed cache $\mathcal { C } _ { b }$ without increasing the budget b, we introduce a context-specific weight patch $\Delta \theta _ { c }$ that is constructed in the context-prefill stage and then merged into the base weights θ to serve downstream queries. Ideally, the patch is designed to narrow the remaining gap between full-cache inference and compressed-cache inference on the downstream queries:

$$
\operatorname* { m i n } _ { \Delta \theta _ { c } } \underset { q \sim p _ { \mathrm { d o w n } } ( q | c ) } { \mathbb { E } } \Big [ \mathcal { L } \big ( f _ { \theta } ( q ; \mathcal { C } ) , f _ { \theta + \Delta \theta _ { c } } ( q ; \mathcal { C } _ { b } ) \big ) \Big ] .\tag{2}
$$

As in KV cache compression, one practical challenge is that $p _ { \mathrm { d o w n } } ( q | c )$ is unavailable at the patch construction time. Therefore, we follow their practices to approximate it with a finite reference set $\mathcal { Q } _ { c }$ derived from the context c:

$$
\operatorname* { m i n } _ { \Delta \theta _ { c } } \underset { q \sim \mathcal { Q } _ { c } } { \mathbb { E } } \Big [ \mathcal { L } \big ( f _ { \theta } ( q ; \mathcal { C } ) , f _ { \theta + \Delta \theta _ { c } } ( q ; \mathcal { C } _ { b } ) \big ) \Big ] .\tag{3}
$$

Sec. 2.4 describes specific reference queries used in our experiments, while Sec. 2.2 and 2.3 describe a specific instantiation of this objective and practical techniques for long-context problems. The overall algorithm is provided in App. A.

The key benefit of the weight-space compensation for KV cache compression is that it shifts the cost of recovery from per-query to a one-time prefill stage. Once $\Delta \theta _ { c }$ is merged into the base weights θ, the patched model uses the same forward graph as the base model and introduces no additional per-token computation during generation. The patched model can then fully inherit the memory and latency improvements of the compressed KV cache at more aggressive budgets b, without paying the per-query cost that a larger b would otherwise incur. Moreover, our formulation is agnostic to the choice of $\mathcal { C } _ { b } .$ Any off-the-shelf KV cache compression method that operates at the prefill stage can be plugged in, making the weight-space patch complementary to existing token-space compression.

## 2.2 Closed-Form Weight Patch Derivation

To obtain the weight patch minimizing the approximated objective Eq. (3) without performing heavy training on the model parameters, we borrow ideas from recent studies in context-to-weight conversion [15, 16, 17]. Prior work suggests a mechanism to convert the effect of context in transformer architectures into weight updates on the post-attention MLP layers. We adapt this idea to our long-context, multi-query setting as a complement to KV cache compression.

MLP activation matching. Motivated by the prior work [15, 16, 17], we instantiate Eq. (3) as a block-wise activation matching objective on reference queries $q \in \mathcal { Q } _ { c }$ , constructing a separate patch for each MLP block. For clarity, we present the derivation for a single block and drop the block index throughout. At each block, we write the block output as the sum of the MLP output Wh and a residual term z. Here $W \in \mathbb { R } ^ { d \times d _ { \mathrm { f f } } }$ is the MLP output projection (the down-projection, in SwiGLU variants), $\boldsymbol { h } \in \mathbb { R } ^ { d _ { \mathrm { f f } } }$ is the post-activation intermediate, and $\dot { z } \in \mathbb { R } ^ { d }$ is the attention output, where $d$ and $d _ { \mathrm { f f } }$ denote the hidden and MLP intermediate dimensions. KV cache compression perturbs the hidden states entering each block and therefore changes both terms. For each reference token i in some $q \in \mathcal { Q } _ { c }$ , we obtain teacher activations $( h _ { i } , z _ { i } )$ and student activations $( h _ { i } ^ { \prime } , z _ { i } ^ { \prime } )$ from teacher-forced forward passes over q with caches C and $\mathcal { C } _ { b }$ respectively.

We seek $\Delta W$ such that the patched compressed-cache block output approximates the corresponding full-cache block output for reference tokens:

$$
W h _ { i } + z _ { i } = ( W + \Delta W ) h _ { i } ^ { \prime } + z _ { i } ^ { \prime } .\tag{4}
$$

Equivalently, the patch uses the compressed-cache MLP features $h _ { i } ^ { \prime }$ as a basis for correcting the full-cache versus compressed-cache block-output discrepancy. We derive patches sequentially across blocks: when solving for $\Delta W$ at block l, the upstream blocks $1 , \ldots , \ell - 1$ are already patched, sO $h _ { i } ^ { \prime }$ and $z _ { i } ^ { \prime }$ reflect activations from a model that has itself been partially corrected. This greedy formulation is consistent with how compression errors propagate at inference time, where each block receives the residual stream produced by the corrected blocks beneath it.

Closed-form solution with ridge regression. Rearranging Eq. (4) gives the per-token condition:

$$
\Delta W h _ { i } ^ { \prime } = t _ { i } , \quad \mathrm { w h e r e } \quad t _ { i } = W ( h _ { i } - h _ { i } ^ { \prime } ) + ( z _ { i } - z _ { i } ^ { \prime } ) .\tag{5}
$$

For a single token, this condition is underdetermined and admits infinitely many token-specific solutions [15, 16], none of which necessarily generalizes beyond that token. To obtain a single context-specific but query-agnostic patch shared across all reference tokens, we stack $\{ h _ { i } ^ { \prime } \}$ and $\{ \bar { t } _ { i } \}$ as rows of $H ^ { \prime } \in \mathbb { R } ^ { \bar { N } _ { \mathrm { r e f } } \times \bar { d } _ { \mathrm { f f } } }$ and $T \in \mathbb { R } ^ { N _ { \mathrm { r e f } } \times d }$ , where $N _ { \mathrm { r e f } }$ is the total number of tokens across all sequences in $\mathcal { Q } _ { c }$ . We then solve the ridge-regularized least-squares problem as in Mazzawi et al. [17]:

$$
\operatorname* { m i n } _ { \Delta W } \big \| \Delta W H ^ { \prime \top } - T ^ { \top } \big \| _ { F } ^ { 2 } + \lambda \| \Delta W \| _ { F } ^ { 2 } ,\tag{6}
$$

where the ridge coefficient $\lambda > 0$ ensures the system is well-conditioned and discourages overly large patches that overfit the reference query set. Because the scale of $H ^ { \prime \top } H ^ { \prime }$ varies substantially across layers, we introduce an adaptive ridge scaling as

$$
\lambda = \lambda _ { 0 } \cdot { \frac { \| H ^ { \prime \top } H ^ { \prime } \| _ { F } ^ { 2 } } { \operatorname { t r } ( H ^ { \prime \top } H ^ { \prime } ) } } ,\tag{7}
$$

where $\lambda _ { 0 }$ is a single hyperparameter shared across layers. This normalization makes λ comparable across layers and architectures, requiring no per-layer tuning. The ridge objective (Eq. (6)) admits the closed-form solution

$$
\Delta W = T ^ { \top } H ^ { \prime } \big ( H ^ { \prime \top } H ^ { \prime } + \lambda I \big ) ^ { - 1 } .\tag{8}
$$

After solving $\Delta W$ for each block, we merge them into the base model to serve downstream queries.

## 2.3 Efficient Computation for Long Reference Sequences

Naively applying Eq. (8) requires materializing all per-token activations $\{ ( h _ { i } ^ { \prime } , t _ { i } ) \}$ across the full reference query in a single forward pass and stacking them into matrices $H ^ { \prime }$ and T, incurring memory cost that grows linearly with the total number of query tokens $N _ { \mathrm { r e f } }$ . This becomes prohibitive in our long-context setting, where reference queries can be as long as the context itself $( e . g .$ , repeat-context introduced in Sec. 2.4). We address this in two stages.

Chunked query encoding. First, we process the reference sequence in fixed-size chunks at the forward-pass level. Each chunk is run through the model independently, so peak activation memory is determined by the chunk size rather than the full sequence length. This introduces an approximation, since cross-chunk attention is dropped. However, our reference queries are designed so that the relevant dependencies are local, e.g., reconstructing specific local segments of the context (Sec. 2.4), which keeps this approximation mild. We evaluate its effect in Sec. 3.4.

Efficient regression with sufficient statistics. Second, even with chunked forward passes, naively stacking the resulting activations $\{ ( h _ { i } ^ { \prime } , t _ { i } ) \}$ across all chunks would still incur memory linear in the total number of tokens $N _ { \mathrm { r e f } }$ . The key observation is that, at any given block, the closed-form solution depends on reference tokens only through two sufficient statistics,

$$
S _ { H } : = H ^ { \prime \top } H ^ { \prime } \in \mathbb { R } ^ { d _ { \mathrm { H } } \times d _ { \mathrm { f f } } } , \qquad S _ { T } : = T ^ { \top } H ^ { \prime } \in \mathbb { R } ^ { d \times d _ { \mathrm { f f } } } ,\tag{9}
$$

both of which are sums of per-token outer products. We therefore accumulate $( S _ { H } , S _ { T } )$ as running sums starting from zero matrices. For each chunk $j ,$ we compute its outer-product contributions $S _ { H } ^ { ( j ) } = H _ { i } ^ { \prime \top } H _ { i } ^ { \prime }$ and $S _ { T } ^ { ( j ) } = T _ { i } ^ { \top } H _ { i } ^ { \prime }$ where $H _ { i } ^ { \prime }$ and $T _ { j }$ denote the stacked activations of $\{ h _ { i } ^ { \prime } \}$ and $\{ t _ { i } \}$ within chunk $j ,$ add them to the růnning totals, and discard the activations of chunk $j$ before moving on. This makes peak activation memory independent of $N _ { \mathrm { r e f } }$ and depends only on the chunk size $L _ { \mathrm { c h u n k } }$ and on the size of the running statistics.

Once all chunks have been processed, we solve a single ridge regression $\Delta W = S _ { T } ( S _ { H } + \lambda I ) ^ { - 1 }$ per block, identical in form to Eq. (8) but using statistics accumulated incrementally rather than materialized all at once.

## 2.4 Reference Query Construction

As discussed in Sec. 2.1, we do not have access to ground-truth downstream query distribution $p _ { \mathrm { d o w n } } ( q | c )$ and thus need an approximation. Following prior KV cache compression work, we approximate $p _ { \mathrm { d o w n } } ( q | c )$ with a finite reference set $\mathcal { Q } _ { c } = \{ q ^ { ( 1 ) } , \ldots , q ^ { ( r ) } \}$ . The reference set is derived purely from the context itself without any downstream labels. We consider three referenceconstruction strategies.

• Repeat-context. Following Kim et al. [9], we construct teacher-forced reconstruction sequences from the context itself. For short contexts, we use a repeat prompt $P$ such as Repeat the previous context. followed by the context tokens, forming $\mathcal { Q } _ { c } = \{ [ P ; c ] \}$ . For long contexts, we split c into N chunks $\{ c ^ { ( j ) } \} _ { j = 1 } ^ { N }$ with chunk-specific prompts $P ^ { ( j ) } , e . g .$ , Repeat the previous context after <chunk anchor>, giving $\mathcal { Q } _ { c } = \{ [ P ^ { ( j ) } ; c ^ { ( j ) } ] \} _ { j = 1 } ^ { N }$ . All tokens are teacher-forced rather than generated. This strategy has no sampling overhead and encourages reference activations to cover fine-grained facts across the context.

• Self-study. Following Eyuboglu et al. [10], we prompt the full-cache base model with a small set of fixed, context-agnostic instructions $\bar { I } ^ { ( 1 ) } , \ldots , I ^ { ( M ) }$ , e.g., Aggregate all key facts mentioned in the context. The model generates an answer $a ^ { ( j ) }$ for each instruction using the full cache, and we use the resulting pairs as teacher-forced reference sequences $\mathcal { Q } _ { c } = \{ [ \bar { I ^ { ( 1 ) } } ; a ^ { ( 1 ) } ] , \dots , [ I ^ { ( M ) } ; a ^ { ( M ) } ] \}$ . Since the answers $a ^ { ( j ) }$ are generally short, each sequence $[ I ^ { ( j ) } ; a ^ { ( j ) } ]$ is treated as a single chunk. Compared with repeat-context, self-study may better cover higher-level summaries and reasoning patterns, but it adds generation cost at context loading and risks larger discrepancy from downstream queries.

• Joint. We pool the reference queries from repeat-context and self-study into a single set $\mathcal { Q } _ { c }$ This provides a conservative approximation to $p _ { \mathrm { d o w n } } ( q | c )$ : self-study aggressively explores plausible downstream queries, while repeat-context completely covers the context.

![](images/d5b10faee375cac1574be4073afda41e5fd8856a290751419fcfb763003ab2aa.jpg)  
Figure 2: Benchmark results using Qwen2.5-7B-1M with PatchKV applied on top of (a) an eviction based method (KVzip) and (b) an approximation based method (Attention Matching). In (b), † marks shortened (tiny) task variants for SCBench retrieval tasks.

## 3 Experiments

Our experiments focus on evaluating whether PatchKV can compensate for the information lost during KV compression. We first evaluate the effect of PatchKV over KV cache compression methods, and then analyze the behavior of the weight patch through ablations and controlled experiments.

## 3.1 Setup

KV cache compression methods. We consider two off-the-shelf KV cache compression methods to which PatchKV is applied: KVzip [9] and Attention Matching [12], representing token-eviction and cache-approximation methods, respectively. KVzip first prefills the context and scores each context KV pair via a teacher-forced reconstruction pass with the repeat-context prompt described in Sec. 2.4. KV pairs that receive low maximum cross-attention during reconstruction are then evicted according to the target cache budget. Attention Matching, in contrast, constructs a compacted cache by solving a regression problem that matches full-cache attention outputs using compacted value activations and additional bias terms while evicting the key activations, using both the repeat-context and self-study prompts described in Sec. 2.4. In both cases, PatchKV is applied after compression, and the resulting patched model is then used to serve downstream queries.

Benchmarks and models. We evaluate on benchmarks spanning three modes of long-context utilization: exact retrieval, contextual reasoning, and robustness to redundant context. Specifically, we use SQuAD [18], GSM8K [20], needle-in-a-haystack (NIAH) [19], and seven benchmarks from SCBench [1], which together span context lengths up to 170K tokens. SCBench additionally includes Retrieve .Multihop and ICL .ManyShot, which we report separately because they exhibit qualitatively different behavior under compression; see App. B.2. We follow the official metric of each benchmark: exact-match accuracy for retrieval and QA tasks, answer accuracy for GSM8K, Pass @1 for code tasks, and ROUGE [22] for summarization.

Table 1: Perplexity (↓) on SQuAD across compression ratios. ref. stands for reference query.
<table><tr><td rowspan="2">Ratio</td><td colspan="2">Full KV</td><td colspan="2">None</td><td colspan="2">repeat-context</td><td colspan="2">self-study</td><td colspan="2">joint</td></tr><tr><td>ref.</td><td>test</td><td>ref.</td><td>test</td><td>ref.</td><td>test</td><td>ref.</td><td>test</td><td>ref.</td><td>test</td></tr><tr><td>0.20</td><td>1.00</td><td>8.79</td><td>1.00</td><td>18.16</td><td>1.00</td><td>12.24</td><td>1.00</td><td>17.01</td><td>1.00</td><td>17.07</td></tr><tr><td>0.10</td><td>1.00</td><td>8.79</td><td>1.10</td><td>17.16</td><td>1.00</td><td>14.03</td><td>1.03</td><td>16.51</td><td>1.00</td><td>16.60</td></tr><tr><td>0.05</td><td>1.00</td><td>8.79</td><td>1.81</td><td>123.36</td><td>1.00</td><td>21.08</td><td>1.10</td><td>18.90</td><td>1.00</td><td>18.84</td></tr><tr><td>0.02</td><td>1.00</td><td>8.79</td><td>6.18</td><td>180.63</td><td>1.00</td><td>32.82</td><td>1.39</td><td>28.34</td><td>1.00</td><td>30.74</td></tr></table>

Results are reported across KV cache budget ratios $b / n$ in the aggressive regime, i.e. at most 20%. For the Attention Matching results in Fig. 2(b), we evaluate shortened variants of the SCBench retrieval tasks, as both Attention Matching and Attention Matching with PatchKV achieve zero accuracy on retrieval tasks under the evaluated cache budgets. We report the original task results in App. B.2. To validate PatchKV across different model architectures, we use three base LLMs: Qwen2.5-7B-1M [23], Qwen3-4B [24], and LLaMA3.1-8B [25]. Implementation details are provided in App. B.

## 3.2 Main Results

Fig. 2 reports the effect of PatchKV applied on top of KVzip and Attention Matching, with three variants corresponding to the reference query strategies introduced in Sec. 2.4. The most pronounced gains occur in the aggressive cache regime $( b / n$ between 0.02 and 0.1), where the cache-only baselines drop sharply while PatchKV substantially narrows the gap to full-cache inference. In this regime, PatchKV often matches or exceeds the cache-only baseline obtained with a noticeably larger budget. This suggests that part of the missing context information can be recovered through the weight patch rather than through additional cache capacity.

The effect is most striking at the extreme end of the budget range: on information-dense tasks such as SQuAD and GSM8K, PatchKV retains nontrivial accuracy even when no KV cache remains. On SQuAD, in particular, PatchKV recovers up to 73% of full-KV accuracy with no cache, whereas the cache-only baseline collapses to near zero. Fig. 3 illustrates this behavior on a representative SQuAD example: KVzip evicts most of the context surrounding the answer token “Leprechaun," causing the cache-only model to produce the incorrect answer “Fighting Irish," whereas applying PatchKV on top of the same compressed cache recovers the correct answer “Notre Dame Leprechaun." Additional qualitative examples are provided in App. C.

These improvements hold across both KVzip (token eviction) and Attention Matching (cache approximation), suggesting that the weight-patching mechanism is largely orthogonal to the choice of compression strategy. Interestingly, the three reference query strategies exhibit clearly different profiles: self-study is most effective on contextual QA and redundancy-heavy tasks, while repeat-context and the joint variant are stronger on retrieval-heavy tasks. This pattern is consistent with the design of each strategy where self-study emphasizes higher-level summaries and reasoning patterns, whereas repeat-context provides fine-grained coverage of every context segment. Overall, PatchKV delivers consistent gains across context lengths, compression methods, and task types, with the largest benefits realized in the regime where compression methods strongly degrade.

## 3.3 Analysis

Generalization ability of reference queries. We examine how well a weight patch constructed from the reference query set $\mathcal { Q } _ { c }$ generalizes to downstream queries $q \sim p _ { \mathrm { d o w n } } ( q | c )$ . On SQuAD, we measure teacher-forced perplexity under both patched and non-patched models on two sets of tokens: (i) the reference queries and (ii) the gold answer tokens conditioned on each downstream query. The results are reported in Tab. 1. The patched model nearly perfectly fits the reference queries, matching the perplexity obtained with the full KV cache. This indicates that the block-wise activation matching used to derive the ridge solution (Eq. (8)) is an effective proxy for the distribution-level reference objective (Eq. (3)) introduced in Sec. 2.1. On downstream test queries, the patched model does not fully close the gap to full-KV perplexity, particularly in the low-cache regime. Nevertheless, it consistently narrows the perplexity gap between the full and compressed KV caches across all budgets, indicating that the patch generalizes beyond the reference set rather than merely memorizing it. We observe that patches built from self-study based reference queries generalize better than repeat-context in majority.

<table><tr><td>Question: What type of mascot do the Notre Dame sport teams have? [SQuAD]</td></tr><tr><td>Pruned Context: [...] The official colors of Notre-Dame are Navy Blue and Gold Rush which are worn in competition by its athletic teams.In addition,-the color green is often worn because of the Fighting Irish nickname.-The Notre DameLepreehaunis the mascot of the athletic teams. Created by Theodore W. Drake in-1964, thelepreehaunwas first used on-the football pocket schedule and later on-the football program covers. Thelepreehaunwas featured on the cover of Time in November 1964 and gained national exposure.</td></tr><tr><td>Full KV: The Notre Dame sport teams have the Notre Dame Leprechaun as their mascot. CORRECT</td></tr><tr><td>Pruned KV (w/o patch): The Notre Dame sport teams have a Fighting Irish mascot. INCORRECT</td></tr><tr><td>PatchKV (Ours): The Notre Dame sport teams have a mascot called the Notre Dame Leprechaun. CORRECT</td></tr></table>

Figure 3: Qualitative example of PatchKV over KVzip on SQuAD. Evicted tokens shown in red.  
![](images/989b4f8d8f8bb1a644bf0f782f8236dc17082a5db0874ab0aeedf03ed61dac24.jpg)  
Figure 4: Multi-task results on RepoQA+KV and Summary+NIAH using Qwen2.5-7B-1M

Weight patch complements the compressed cache. Next, we test whether the patch captures information lost by compression rather than information redundant with the compressed KV cache. On SQuAD, we first evict 80% of KV pairs via KVzip and randomly split the remaining 20% into two disjoint subsets, $\mathcal { C } _ { A }$ and $\mathcal { C } _ { B }$ . We then construct two weight patches $\Delta W _ { A }$ and $\Delta W _ { B }$ using only $\mathcal { C } _ { A }$ and $\mathcal { C } _ { B } .$ , respectively, and measure performance under three configurations: (i) the compressed cache alone $( \mathcal { C } _ { A } \ \mathrm { o r } \mathcal { C } _ { B } )$ , (ii) each compressed cache with its corresponding weight patch $( \mathcal { C } _ { A } \bar { + } \Delta W _ { A } , \mathcal { C } _ { B } + \Delta W _ { B } ) .$ , and (iii) each compressed cache with the patch derived from the disjoint subset $( { \mathcal { C } } _ { A } + \Delta W _ { B } , { \mathcal { C } } _ { B } + \Delta W _ { A } )$

The results are reported in Tab. 2. Since $\mathcal { C } _ { A }$ and $\mathcal { C } _ { B }$ are randomly drawn from the same context, the two caches have similar capacity, as reflected by their comparable no-patch performance (top row). Each patch performs best when paired with its source cache (diagonal entries) and degrades when paired with the disjoint subset's patch (off-diagonal entries), while still outperforming the cache-only baseline. This intermediate behavior is consistent with the design of the weight-patching objective (Eq. (3)): both patches are trained to fill the gap between full-cache and compressed-cache behavior, so they share the bulk of that gap (the evicted 80% of the context) but differ in the smaller portion that corresponds to the other subset. The shared component transfers under swapping and explains the gain over the cache-only baseline, whereas the subset-specific component is recovered only when the patch is paired with its source cache.

Table 2: Cross-evaluation of patches built from disjoint cache subsets. Rows: weight patch applied at inference. Columns: compressed cache used at inference.
<table><tr><td rowspan="2">Patch</td><td colspan="2">Eval cache</td></tr><tr><td> $\mathcal { C } _ { A }$ </td><td> $\mathcal { C } _ { B }$ </td></tr><tr><td>None</td><td>54.95</td><td>57.74</td></tr><tr><td> $\Delta W _ { A }$ </td><td>85.39</td><td>66.60</td></tr><tr><td> $\Delta W _ { B }$ </td><td>65.63</td><td>88.09</td></tr></table>

Multi-task KV caches. To evaluate PatchKV beyond the single-task setting of Fig. 2, we construct mixed contexts concatenated from multiple benchmarks and assess whether a single patch can serve different downstream tasks simultaneously. In Fig. 4, we report performance on two mixed contexts: RepoQA+KV, combining two retrieval tasks, and Summary+NIAH, combining long-document summarization with text retrieval. For each mixture, we build a single compressed cache and a corresponding weight patch, then evaluate on the queries associated with each constituent task separately. PatchKV consistently improves over the compressed baseline on both tasks, with the gain persisting across compression ratios. This indicates that a single patch built jointly over a heterogeneous context is not dominated by one task at the expense of another. We attribute this to two aspects of our design: the reference query set represents the query distribution regardless of task structure, and joint regression in Eq. (6) aggregates compression-induced errors from all regions into a single weight patch. We also observe that reference query strategies transfer across task mixtures: self-study performs better on summary contexts, while repeat-context gains an edge when retrieval is present, consistent with our findings in Sec. 3.2.

![](images/42311b54011991fc38aa8e6d3f59dba5fd281b8b7b91c8f11b8150275f052926.jpg)  
Figure 5: Averaged score across datasets, normalized by the full-KV score, for different models.

## 3.4 Ablation Study

Effect of chunk size. We study how the chunk size $L _ { \mathrm { c h u n k } }$ used in patch construction (Sec. 2.3) trades off patch quality against construction cost. Fixing the cache budget at $b / n = 0 . 1$ , we vary $L _ { \mathrm { c h u n k } }$ from 1K to 10K tokens and report average relative accuracy and patch construction overhead over the benchmark in Tab. 3. Accuracy stays flat from 1K to 5K and slightly drops at 8K and beyond, consistent with the assumption in Sec. 2.3 that cross-chunk attention can be safely dropped only when relevant dependencies remain within a single chunk. We attribute the mild accuracy drop at larger chunk sizes to the difficulty of long-span reference reconstruction, which yields more noised teacher activations and degrades the ridge target. Time decreases sharply from 1K to 5K and saturates beyond that, while peak memory grows monotonically with $L _ { \mathrm { c h u n k } }$ , as expected from the chunked-computation design. We therefore use 5K as the default chunk size with the best balance.

Table 3: Effect of chunk size.
<table><tr><td>Chunk</td><td>1K</td><td>2K</td><td>5K</td><td>8K</td><td>10K</td></tr><tr><td>Rel. Acc.</td><td>.696</td><td>.695</td><td>.696</td><td>.672</td><td>.679</td></tr><tr><td>Time (s)</td><td>114</td><td>98</td><td>85</td><td>86</td><td>81</td></tr><tr><td>Mem. (GB)</td><td>30.8</td><td>31.6</td><td>33.0</td><td>34.1</td><td>35.9</td></tr></table>

Table 4: Ridge scaling results.

Effect of ridge scaling. In Eq. (6), H' and T vary across layers and contexts, so the ridge term should be scaled relative to the layerwise regression matrix. We compare different layer-adaptive schemes that normalize λ by spectral quantities of $\dot { S } _ { H } = \dot { H } ^ { \prime \top } H ^ { \prime } ;$ trace mean $\operatorname { t r } ( S _ { H } ) / d _ { f f } .$ , and our variant $| | S _ { H } | | _ { F } ^ { 2 } / \mathrm { t r } ( S _ { H } )$ , corresponding to the mean, and weighted mean of eigenvalues respectively. Tab. 4 shows that our default $\vert \vert { S _ { H } } \vert \vert _ { F } ^ { 2 } / \mathrm { t r } ( { S _ { H } } )$ consistently performs better between 3 tasks of each group: NIAH, En.QA, and Math.Find, patched by self-study reference query. For each scaling strategy, we report the best performance obtained by selecting best $\lambda _ { 0 }$ . Our normalization provides an intermediate scale between the trace mean and spectral norm, emphasizing dominant directions without collapsing onto the largest eigenvalue. We empirically find that it transfers across contexts, models, and cache ratios without any retuning.

<table><tr><td>Ratio</td><td>Trace</td><td>Weighted</td></tr><tr><td>0.2</td><td>0.92</td><td>0.93</td></tr><tr><td>0.1</td><td>0.88</td><td>0.91</td></tr><tr><td>0.05</td><td>0.72</td><td>0.72</td></tr><tr><td>0.02</td><td>0.45</td><td>0.46</td></tr><tr><td>0.01</td><td>0.32</td><td>0.36</td></tr></table>

Model scale and architectures. We also test PatchKV across model families and scales, including Qwen2.5-7B-1M [23], Qwen3-4B [24], and LLaMA3.1-8B [25]. Fig. 5 reports the average relative accuracy across datasets, where each score is normalized by the performance with full KV cache. Applied on top of KVzip, PatchKV consistently improves over the cache-only baseline across all models, with the largest gains in the aggressive cache regime $( b / n$ between 0.02 and 0.1) where the baseline degrades sharply. The relative ranking of reference query strategies is also stable across architectures: the joint and self-study variants are strongest at aggressive budgets, while repeatcontext becomes competitive as more KV pairs are retained. Together, these results suggest that PatchKV is largely architecture-agnostic, compensating for compression-induced information loss across models.

## 4 Related Work

KV cache compression. Token-eviction methods retain a subset of the cache by scoring tokens with attention statistics or reconstruction signals [8, 3, 26], and are query-aware, requiring the downstream query at selection time. KVzip [9] extends this to the query-agnostic setting by scoring tokens through context reconstruction, producing a single compressed cache reusable across queries Attention Matching [12] instead approximates the full cache by constructing a compact cache that reproduces full-cache attention outputs. PatchKV pairs a compressed cache with a context-specific weight patch that compensates for compression-induced error, making it compatible with different compression methods.

Compensating for compression error. Concurrent work has also explored compensating for information discarded during KV cache compression. GRKV [27] redistributes information from evicted tokens into retained KV states, while MomentKV [28] maintains compact statistics of evicted states to approximate their residual attention contribution. While these methods compensate for discarded information through KV cache or attention-side representations, PatchKV represents the residual correction through context-specific MLP weight updates.

Converting context into weights. Dherin et al. [15] show that the effect of a context segment on a transformer block can be represented exactly by a low-rank MLP weight update, and subsequent works extend this to more general architectures [16] and sequential optimization over reference completions [17]. Doc-to-LoRA [29] learns a low-rank adapter that internalizes a document through context distillation at inference time. These works largely treat the weight update as a stand-alone replacement for the context, often under simplified assumptions such as fixed query tokens or access to only a subset of the original context. PatchKV repurposes context-to-weight conversion for a more practical use case: rather than replacing the context, the patch complements an off-the-shelf compressed KV cache by closing its residual compression error in closed form, making the mechanism directly applicable to existing long-context serving pipelines.

## 5 Conclusion

We introduce PatchKV, a framework that complements KV cache compression with context-derived weight patches. Given an off-the-shelf compressed cache, PatchKV constructs block-wise weight patches by aligning activations of reference queries with full cache and compressed cache, yielding a closed-form ridge-regression solution. Experiments show that our method offers consistent improvements over compression methods at low-cache regime, across diverse benchmarks and models. These results suggest that PatchKV provides a general complement to cache-only compression, enabling more aggressive KV cache compression without increasing per-query inference cost.

## References

[1] YUCHENG LI, Huiqiang Jiang, Qianhui Wu, Xufang Luo, Surin Ahn, Chengruidong Zhang, Amir H. Abdi, Dongsheng Li, Jianfeng Gao, Yuqing Yang, and Lili Qiu. SCBench: A KV cache-centric analysis of long-context methods. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=gkUyYcY1W9.

[2] Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean. Efficiently scaling transformer inference. Proceedings of machine learning and systems, 5:606–624, 2023.

[3] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, et al. H2o: Heavy-hitter oracle for efficient generative inference of large language models. Advances in Neural Information Processing Systems, 36:34661–34710, 2023.

[4] Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. Kivi: A tuning-free asymmetric 2bit quantization for kv cache. In International Conference on Machine Learning, pages 32332–32344. PMLR, 2024.

[5] Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W Mahoney, Yakun S Shao, Kurt Keutzer, and Amir Gholami. Kvquant: Towards 10 million context length llm inference with kv cache quantization. Advances in Neural Information Processing Systems, 37:1270–1303, 2024.

[6] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In The Twelfth International Conference on Learning Representations, 2024.

[7] Chaojun Xiao, Pengle Zhang, Xu Han, Guangxuan Xiao, Yankai Lin, Zhengyan Zhang, Zhiyuan Liu, and Maosong Sun. Infllm: Training-free long-context extrapolation for llms with an efficient context memory. Advances in neural information processing systems, 37:119638–119661, 2024.

[8] Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. Snapkv: Llm knows what you are looking for before generation. Advances in Neural Information Processing Systems, 37:22947–22970, 2024.

[9] Jang-Hyun Kim, Jinuk Kim, Sangwoo Kwon, Jae W. Lee, Sangdoo Yun, and Hyun Oh Song. KVzip: Query-agnostic KV cache compression with context reconstruction. In The Thirtyninth Annual Conference on Neural Information Processing Systems, 2025. URL https: //openreview.net/forum?id=JFygzwx8SJ.

[10] Sabri Eyuboglu, Ryan Saul Ehrlich, Simran Arora, Neel Guha, Dylan Zinsley, Emily Ruoyu Liu, Atri Rudra, James Y Zou, Azalia Mirhoseini, and Christopher Re. Cartridges: Lightweight and general-purpose long context representations via self-study. In ES-FoMo III: 3rd Workshop on Efficient Systems for Foundation Models, 2025.

[11] Hanshi Sun, Li-Wen Chang, Wenlei Bao, Size Zheng, Ningxin Zheng, Xin Liu, Harry Dong, Yuejie Chi, and Beidi Chen. Shadowkv: Kv cache in shadows for high-throughput long-context 1lm inference. In International Conference on Machine Learning, pages 57355–57373. PMLR, 2025.

[12] Adam Zweiger, Xinghong Fu, Han Guo, and Yoon Kim. Fast KV compaction via attention matching. In Forty-third International Conference on Machine Learning, 2026. URL https: //openreview.net/forum?id=t01SKTj8pP.

[13] Hanlin Tang, Yang Lin, Jing Lin, Qingsen Han, Danning Ke, Shikuan Hong, Yiwu Yao, and Gongyi Wang. Razorattention: Efficient kv cache compression through retrieval heads. In The Thirteenth International Conference on Learning Representations, 2025.

[14] Akshat Sharma, Hangliang Ding, Jianping Li, Neel Dani, and Minjia Zhang. Minikv: Pushing the limits of 2-bit kv cache via compression and system co-design for efficient long context inference. In Findings of the Association for Computational Linguistics: ACL 2025, pages 18506–18523, 2025.

[15] Benoit Dherin, Michael Munn, Hanna Mazzawi, Michael Wunder, and Javier Gonzalvo. Learning without training: The implicit dynamics of in-context learning. arXiv preprint arXiv:2507.16003, 2025.

[16] Adrian Goldwaser, Michael Munn, Javier Gonzalvo, and Benoit Dherin. Equivalence of context and parameter updates in modern transformer blocks. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id=aZ5Di8Ii4X.

[17] Hanna Mazzawi, Benoit Dherin, Michael Munn, Michael Wunder, and Javier Gonzalvo. Transmuting prompts into weights. arXiv preprint arXiv:2510.08734, 2025.

[18] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. SQuAD: 100,000+ questions for machine comprehension of text. In Jian Su, Kevin Duh, and Xavier Carreras, editors, Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2383–2392, Austin, Texas, November 2016. Association for Computational Linguistics. doi: 10.18653/v1/D16-1264. URL https://aclanthology.org/D16-1264/.

[19] Greg Kamradt. Needle in a haystack-pressure testing llms, 2023.

[20] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

[21] LI Haoyang, Yiming Li, Anxin Tian, Tianhao Tang, Zhanchao Xu, Xuejia Chen, HU Nicole, Wei Dong, Li Qing, and Lei Chen. A survey on large language model acceleration based on kv cache management. Transactions on Machine Learning Research, 2025.

[22] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81, 2004.

[23] Qwen: An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115.

[24] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[25] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

[26] Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu, Yucheng Li, Tianyu Liu, Keming Lu, Wayne Xiong, Yue Dong, Junjie Hu, and Wen Xiao. PyramidKV: Dynamic KV cache compression based on pyramidal information funneling. In Second Conference on Language Modeling, 2025. URL https://openreview.net/forum?id=ayi7qezU87.

[27] Junjie Peng, You Wu, Haoyi Wu, Jialong Han, Xiaohua Xie, Kewei Tu, and Jianhuang Lai. Grkv: Global regression for training-free kv cache compression in long-context llms. arXiv preprint arXiv:2605.31105, 2026.

[28] Yu Li, Binxu Li, and Tian Lan. MomentKV: Closing the directional gap in KV cache eviction for long-context inference. In Third Conference on Language Modeling, 2026. URL https: //openreview.net/forum?id=Mvdvz3jHiK.

[29] Rujikorn Charakorn, Edoardo Cetin, Shinnosuke Uesaka, and Robert Tjarko Lange. Doc-toloRA: Learning to instantly internalize contexts. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id=iW1oBB072S.

[30] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 conference on empirical methods in natural language processing, pages 2369–2380, 2018.

[31] Amir Zandieh, Majid Daliri, Majid Hadian, and Vahab Mirrokni. Turboquant: Online vector quantization with near-optimal distortion rate. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=t03ASKZlok.

[32] Alan Julian Izenman. Reduced-rank regression for the multivariate linear model. Journal of multivariate analysis, 5(2):248–264, 1975.

## A Algorithms

Algorithm 1 PatchKV weight patch construction   
Require: Context $c ,$ base model $f _ { \theta }$ with L blocks, KV cache compressor, cache budget b, ridge   
coefficient $\lambda _ { 0 } ,$ chunk size Lchunk   
1: $\Delta \theta _ { c } \gets \emptyset$   
2: Cache preparation   
3: Prefill full cache $C \gets f _ { \theta } ( c )$   
4: Compress $C _ { b } \gets \mathrm { C o m p r e s s } ( C , b )$   
5: Build reference set $Q _ { c }$ (repeat-context / self-study / joint)   
6: Partition $Q _ { c }$ into chunks $\{ q _ { j } \}$   
7: Initialize teacher and student states $x _ { j , 1 } ^ { F } , x _ { j , 1 } ^ { C }$ from each reference chunk $q _ { j }$   
8: Block-wise patch construction   
9: for $\ell = 1$ to $\bar { \boldsymbol { \mathbf { \rho } } } _ { L }$ do   
10: Initialize $S _ { H }  0 \in \mathbb { R } ^ { d _ { \mathrm { f f } } \times d _ { \mathrm { f f } } } , S _ { T }  0 \in \mathbb { R } ^ { d \times d _ { \mathrm { f f } } }$   
11: for each reference chunk $q _ { j }$ do   
12: Teacher forward: forward block l from $x _ { j , \ell } ^ { F }$ with full cache $C ;$ collect $( h _ { i } , z _ { i } )$ and obtain   
$x _ { j , \ell + 1 } ^ { F }$   
13: Student forward: forward block l from $x _ { j , \ell } ^ { C }$ with compressed cache $C _ { b } ;$ collect $( h _ { i } ^ { \prime } , z _ { i } ^ { \prime } )$   
14: Compute targets   
$t _ { i } \gets W _ { \ell } ( h _ { i } - h _ { i } ^ { \prime } ) + ( z _ { i } - z _ { i } ^ { \prime } )$   
15: Stack $H _ { j } ^ { \prime } \gets [ h _ { i } ^ { \prime } ] _ { i } , T _ { j } \gets [ t _ { i } ] _ { i }$   
16: Accumulate   
$S _ { H } \mathrel { + } = H _ { j } ^ { \prime \top } H _ { j } ^ { \prime } , \qquad S _ { T } \mathrel { + } = T _ { j } ^ { \top } H _ { j } ^ { \prime }$   
17: Discard per-token activations $( h _ { i } , z _ { i } , h _ { i } ^ { \prime } , z _ { i } ^ { \prime } )$   
18: end for   
19: Patch computation:   
$\lambda  \lambda _ { 0 } \frac { \| S _ { H } \| _ { F } ^ { 2 } } { \mathrm { t r } ( S _ { H } ) }$   
20:   
$\Delta W _ { \ell } \gets S _ { T } ( S _ { H } + \lambda I ) ^ { - 1 }$   
21: Merge $W _ { \ell } \gets W _ { \ell } + \Delta W _ { \ell }$   
22: $\Delta \theta _ { c } \dot { \left. \right. } \Delta \theta _ { c } \cup \left\{ \Delta W _ { \ell } \right\}$   
23: for each reference chunk $q _ { j }$ do   
24: Student-state propagation: forward the patched block l from $x _ { j , \ell } ^ { C }$ with $C _ { b }$ and store the   
output as $x _ { j , \ell + 1 } ^ { C }$   
25: end for   
26: end for   
27: return Patched weights $\theta + \Delta \theta _ { c }$ and compressed cache $C _ { b }$

## B Further Details in Experimental Setup

## B.1 Implementation details.

KV cache compression. As described in Sec. 3.1, we initialized the compressed KV cache with existing methods, KVzip [9] and Attention Matching [12]. For KVzip, we follow their default setting with prefill chunk size 2000 and global head scoring.

In Attention Matching, due to large memory and time cost in extremely long contexts such as SCBench [1], we select the key as HIGHESTATTENTIONKEYS that select KV cache key $C _ { k }$ by scoring. For other hyperparameters, we follow the best reported setting of Attention Matching using HIGHESTATTENTIONKEYS.

![](images/4bb43d04629181325b4dd460a2bf9940b9f4549e4f1deb7381da6887c9ae3085.jpg)  
Figure 6: FP32 vs. TF32 patch construction.

![](images/08ac4408796da78b4f688253aeb23fc0971e27123733223d3a05e7164e083730.jpg)  
Figure 7: Additional benchmark results across KV cache budgets using Qwen2.5-7B-1M and KVzip.

Numerical Precision. Patch construction involves accumulating sufficient statistics and solving the ridge regression in Eq. (8), which can be sensitive to numerical errors under low-precision computation. We therefore perform patch construction at higher precision than the BF16 precision used for the base model inference. Specifically, we keep patch construction in FP32 while enabling TF32 for the matrix multiplications involved in computing the sufficient statistics. As shown in Fig. 6, TF32 achieves downstream performance comparable to FP32 across KV cache budgets, while reducing patch construction time from 147.91 s to 85.35 s. We therefore use TF32 matrix multiplication for patch construction throughout our experiments.

## B.2 Dataset Selection and Task Variants

Dataset selection. SCBench includes nine tasks, of which seven are used in our main evaluation. We report Retrieve. MultiHop and ICL.ManyShot separately since compression itself can improve performance on these tasks in these evaluation settings. This makes it difficult to distinguish the effect of PatchKV from the benefit of compression itself. For Retrieve . MultiHop, which contains a large amount of distracting context, the evaluation results show that removing more context can improve performance by removing distractors. For ICL.ManyShot, we observe a more unusual pattern in which the baseline achieves its highest accuracy when no context KV pairs are retained, even exceeding the full-KV baseline. This suggests that the many-shot context may not always be beneficial in this setting. One possible explanation is that the pretrained model can solve much of the task directly, while the provided context introduce additional distracting signals. We report the results of Retrieve .MultiHop and ICL.ManyShot in Fig. 7.

Nevertheless, multi-hop reasoning remains an important setting for evaluating whether PatchKV remains effective when answering requires information from multiple parts of the context. As shown in Fig. 7, PatchKV with repeat-context or joint reference queries further improves over KVzip across several cache budgets in Retrieve. MultiHop. To further verify that this result is not specific to the highly redundant synthetic structure of Retrieve .MultiHop, we additionally evaluate PatchKV on HotpotQA [30], a natural multi-hop question answering benchmark. PatchKV improves over

Table 5: Score (%) on the original SCBench retrieval tasks using Qwen2.5-7B-1M. Both methods achieve zero accuracy on Retr.KV and Retr.Prefix-Suffix under the evaluated cache budgets.
<table><tr><td>Task</td><td>Method</td><td>0.20</td><td>0.10</td><td>0.05</td><td>0.02</td><td>0.01</td></tr><tr><td>Retr.KV</td><td>Attention Matching + PatchKV</td><td>0.0 0.0</td><td>0.0 0.0</td><td>0.0 0.0</td><td>0.0 0.0</td><td>0.0 0.0</td></tr><tr><td>Retr.Prefix-Suffix</td><td>Attention Matching + PatchKV</td><td>0.0 0.0</td><td>0.0 0.0</td><td>0.0 0.0</td><td>0.0 0.0</td><td>0.0 0.0</td></tr><tr><td>Code.RepoQA</td><td>Attention Matching + PatchKV</td><td>15.23 33.64</td><td>3.18 8.64</td><td>1.59 3.41</td><td>0.23 0.91</td><td>0.23 0.68</td></tr></table>

Table 6: Stagewise wall-clock time and peak GPU memory for PatchKV construction.
<table><tr><td>Stage</td><td>Time (s)</td><td>Peak Memory (GB)</td></tr><tr><td>(i) Cache compression (KVzip)</td><td>67.82</td><td>24.89</td></tr><tr><td>(ii) Teacher forward</td><td>41.41</td><td>30.92</td></tr><tr><td>(iii) Student forward</td><td>16.38</td><td>30.99</td></tr><tr><td>(iv) Patch computation</td><td>11.40</td><td>32.98</td></tr><tr><td>(v) Student-state propagation</td><td>16.16</td><td>31.11</td></tr><tr><td>Total</td><td>153.17</td><td>32.98</td></tr></table>

KVzip across the evaluated nonzero cache budgets, providing complementary evidence that PatchKV remains effective on multi-hop reasoning tasks.

Task variants. For the Attention Matching experiments in Fig. 2, we use shortened variants of the SCBench retrieval tasks. On Retr.KV and Retr.Prefix-Suffix, Attention Matching collapses to zero accuracy under the evaluated budget, and applying PatchKV does not recover nonzero performance. The original tasks therefore provide little resolution for evaluating the additional compensating effect of PatchKV over Attention Matching. We instead evaluate the corresponding shortened variants, where the cache-only baseline retains nonzero performance under aggressive compression and the effect of weight-space compensation can be measured. For completeness, we report the results on the original tasks in Tab. 5.

## C Additional Experiment Results.

Runtime Analysis. We present a stagewise breakdown of the additional computational cost introduced by PatchKV. Patch construction consists of a one-time cache-compression stage, followed by four stages repeated for each transformer block: (ii) teacher forward, (iii) student forward, (iv) patch computation, and (v) student-state propagation. Wall-clock time and peak GPU memory for each stage are reported in Tab. $^ { 6 , }$ measured on a context of 169,035 tokens with repeat-context reference query and cache budget 0.2. For steps (ii)-(v), wall-clock time is aggregated for all transformer blocks.

Reference context budget. In Fig. 8, we analyze the effect of limiting the amount of context used for patch construction on SCBench RepoQA, where the context length reaches up to 68K tokens. By controlling how many chunks of the context are fed as reference queries, we show that PatchKV achieves notable gains even with a small reference budget: using only 5K tokens (a single chunk) already recovers 15% of the gap between the compressed and full-KV baseline. Performance continues to improve as more context is included, but with diminishing returns, suggesting that a modest reference budget is sufficient in practice and that PatchKV's construction cost can be reduced significantly without sacrificing much accuracy.

Compatibility with quantized KV caches. A complementary line of work reduces KV cache memory by quantizing it to lower precision [5, 31]. Since PatchKV operates as a residual update to the model weights and is agnostic to how the cache itself is stored, it can be combined with quantization without any modification. To verify this, we apply PatchKV on top of INT4 and INT8 quantized KV caches and report the results in Fig. 10. PatchKV improves accuracy over the baseline at both INT8 and INT4 precision, with gains comparable to those observed in the unquantized setting. This confirms that PatchKV serves as an orthogonal solution to KV cache quantization for efficient LLM inference.

![](images/181a24f5389fe1000fa2d554635ecf8b290a61477d5f5b20805eb772900e4098.jpg)  
Figure 8: Performance across reference context lengths.

![](images/c15525272a98ee640c54b0a495b1c0ccb30e82a842c05df9c0e160aec9680270.jpg)  
Figure 9: Performance when restricting the rank of weight patch of PatchKV.

Low-rank weight patches. In some deployment scenarios, the weight patch may need to be stored alongside the base model, introducing additional memory overhead. We therefore consider a low-rank variant of PatchKV that restricts the patch to rank r (we set $r = 3 2 )$ . Concretely, we apply a reducedrank regression [32] to the closed-form ridge solution, where we project $\Delta W _ { \mathrm { r i d g e } } = \dot { S _ { T } } ( \dot { S } _ { H } + \lambda I ) ^ { - 1 }$ onto its top-r output subspace via $\Delta W _ { r } = U _ { r } U _ { r } ^ { \top } \Delta W _ { \mathrm { r i d g e } }$ . Here $U _ { r }$ contains the top-r eigenvectors of $\Delta W _ { \mathrm { r i d g e } } S _ { H } \Delta W _ { \mathrm { r i d g e } } ^ { \top } .$ This keeps the closed-form structure intact and reduces per-context patch storage from $d \cdot d _ { \mathrm { f f } }$ to $( d + d _ { \mathrm { f f } } ) \cdot r$ . In Fig. 9, we compare the performance of full-rank PatchKV with the variant with weight patch with rank-32 on Qwen2.5-7B-1M, constructed by repeat-context reference query. As shown, the low-rank patch matches full-rank performance within a small margin while reducing per-context patch storage by 99%, demonstrating its practical viability.

Online Extension. Our main experiments construct a context-specific patch once and reuse it across subsequent queries. We additionally examine whether weight-space compensation can be repeatedly applied as the KV cache evolves during generation. Following the online compaction setting of Attention Matching (AM), we evaluate an online variant of PatchKV on AIME 2025 using Qwen3-4B, repeat-context reference queries. During decoding, we compact the KV cache whenever it reaches $P \doteq 2 0 4 8$ entries. At each event, we protect the most recent 20 entries and apply AM. PatchKV is then applied on top of the compacted cache before decoding resumes.

We evaluate generation limits of 4K and 8K tokens while varying the KV cache size after compaction. As shown in Tab. 7, PatchKV provides larger gains under more aggressive compaction. With an 8K and 4K generation limit, PatchKV show improvements compared to AM in more aggressive budgets, whereas PatchKV provides little or no improvement at larger post-compaction cache sizes. These results provide initial evidence that weight-space compensation can be repeatedly applied as generated KV states are compressed during long reasoning. A broader evaluation of repeated updates and their long-horizon stability is left for future work.

Table 7: Online evaluation on AIME 2025 using Qwen3-4B. A compaction event is triggered whenever the physical KV cache reaches 2048 entries, with the most recent 20 entries protected during compaction. 2048 → m denotes the post-compaction KV cache size.
<table><tr><td>KV cache size</td><td>Method</td><td>4K Acc.</td><td>8K Acc.</td></tr><tr><td>Full KV</td><td>Full</td><td>9/30</td><td>14/30</td></tr><tr><td rowspan="2">2048 → 1034</td><td>AM</td><td>9/30</td><td>13/30</td></tr><tr><td>AM + PatchKV</td><td>9/30</td><td>12/30</td></tr><tr><td rowspan="2">2048 → 527</td><td>AM</td><td>9/30</td><td>11/30</td></tr><tr><td>AM + PatchKV</td><td>8/30</td><td>13/30</td></tr><tr><td rowspan="2"> $2 0 4 8  2 2 2$ </td><td>AM</td><td>2/30</td><td>4/30</td></tr><tr><td>AM + PatchKV</td><td>8/30</td><td>8/30</td></tr><tr><td rowspan="2"> $2 0 4 8  1 2 1$ </td><td>AM</td><td>2/30</td><td>2/30</td></tr><tr><td>AM + PatchKV</td><td>6/30</td><td>5/30</td></tr></table>

![](images/475f3c33a11feec4be0812d77fe9d8c00b2eef220a9d8e30def4ed5a65275792.jpg)

![](images/d7c6d85e5510e73ea89e4923741a6254df4f7bcfeea4d865e8922df09185cde6.jpg)

![](images/e2847be3f8fe94d97bbf9be3c7d2ed17df20609ea129859dbdd23e6d3c625b7e.jpg)

![](images/a0e8dfa83a412447e989580d0746b263ca64961aec20dd40a0dc24afdd69e15d.jpg)  
Figure 10: Performance of PatchKV on INT4/INT8 quantized KV cache.

More qualitative examples. From Fig. 11 to Fig. 14, we show qualitative examples of query and answer with and without PatchKV on evicted context.

Full experiment results. In Fig.15 and Fig.16, we report the full result of benchmarks evaluated in Qwen3-4B and LLaMA3.1-8B. Both models show similar property with Qwen2.5-7B-1M, consistently improving across all benchmarks and variants.

## D Limitations

PatchKV relies on context-derived reference queries as proxies for unknown future queries, so the reference-query mismatch may limit recovery under larger distribution shifts. Patch construction also incurs a one-time preprocessing cost and access to the full KV cache as a teacher, making the method most suitable when this cost can be amortized across repeated queries or sufficiently long decoding. Our primary evaluation targets the single-context, multi-query setting, while concurrent serving of many context-specific patches requires additional serving mechanisms. The online results in Tab. 7 provide initial evidence beyond this static-context setting, but broader evaluation remains future work.

![](images/ee6886b83abedfb16ae25d978ab2fd77811bfc4025ab2b24c28d678cbb7003cf.jpg)  
Figure 11: Qualitative example of PatchKV over KVzip on SQuAD. Evicted tokens shown in red.

![](images/ddd2b22d30c39df060575328f3a10c5fdfd46c034a77fc6304bd5400b84bfba6.jpg)  
Figure 12: Qualitative example of PatchKV over KVzip on SQuAD. Evicted tokens shown in red.

![](images/d77edf1f0aeb32a4dc7ff7c9b92393c21e5f799d68b63ffa8af0bc929d6ccb8b.jpg)  
Figure 13: Qualitative example of PatchKV over KVzip on GSM8K. Evicted tokens shown in red.

![](images/01fa1671446832f7e09356f83cc8ce359c33d11d1d97a36b577ee583c070abe0.jpg)  
Figure 14: Qualitative example of PatchKV over KVzip on GSM8K. Evicted tokens shown in red.

![](images/4d8d89ca030386c68b34a05a292b34663ccb1ea13fa7089cf02fded6542d99f7.jpg)  
Figure 15: Benchmark results using LLaMA3.1-8B with PatchKV applied on top of an eviction based method (KVzip).

![](images/261d1a698b9b2306fc65d02e3e8169ac6410a416bb6dddc20d14a86c97e6f65e.jpg)  
Figure 16: Benchmark results using Qwen3-4B with PatchKV applied on top of an eviction based method (KVzip).