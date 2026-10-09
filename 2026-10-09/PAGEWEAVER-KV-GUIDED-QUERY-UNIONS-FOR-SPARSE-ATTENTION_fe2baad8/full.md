# PAGEWEAVER: KV-GUIDED QUERY UNIONS FOR SPARSE ATTENTION

Zhiyuan Li <sup>1</sup> <sup>\*</sup> Zihan Li <sup>1</sup> <sup>\*</sup> Zefang Yuan <sup>1</sup> Lei Wang <sup>1</sup> Hao Wang <sup>1</sup>

## ABSTRACT

Dynamic sparse attention limits the KV pages selected by each query, but a small support does not necessarily yield efficient GPU work. Query unions share page loads and populate Tensor Core tiles; their cost depends on which queries are grouped together. We present PAGEWEAVER, an execution design that uses selected-page affinity to assemble query groups while preserving each query’s original support and complete output ownership. A bounded GPU search produces query IDs, and an ID-aware two-CTA kernel consumes them without materializing reordered Q tensors or cross-page partial outputs. A direct KV-page union implementation provides a complementary design study of nonlocal reuse and reduction cost. With FP8 KV throughout, the H200 Union8 implementation achieves a 1.70× geometric-mean complete-call speedup over the measured FlashInfer path on six captures. Online regrouping further lowers latency by 3.26–7.66% on five selected 64K-context captures. Whole-model prefill throughput is 7.88–14.36% above the tested native path; the incremental regrouping benefit is smaller, with observed median gains of 0.47–0.73% at 32K/64K and regressions at 8K. A B300 comparison identifies cases where preparation cost and a stronger native kernel remove the advantage. These results separate execution-group reuse from the complete cost of exploiting it online.

## 1 INTRODUCTION

Sparse attention selects a limited number of KV pages for each query, reducing the number of interactions relative to dense attention. Yet equal numbers of selected edges can produce very different execution costs. GPU Tensor Cores operate on matrix tiles, while dynamic page selection produces an irregular relation between queries and pages. Filling those tiles introduces a second scheduling problem: which queries should share a page traversal?

Query union is an established answer to part of this problem. Neighboring queries traverse the union of their selected pages, and membership masks retain each query’s own support (LiteTopK Contributors, 2026). Grouping query heads supplies enough rows for a matrix tile and amortizes KV loads. Increasing the group size, however, can also enlarge the union and introduce more masked work. The choice of group size and the choice of group members are distinct optimization decisions.

Reuse must respect execution ownership. When adjacent queries share few pages, a page-oriented view can reveal reuse among distant queries. MSA’s KV-outer design exploits such reuse by gathering queries for a page and merging partial attention outputs (Lai et al., 2026). We extend this view to pairs of pages sharing queries. Direct page union can reduce partial results, but also requires incidence inversion, work lists, and a final reduction. Our measurements show that this organization can win on controlled interleaved supports while losing on the examined model captures. Shared structure alone does not determine a profitable traversal.

![](images/9f4dceb7e85da71e21a52cba9bc98081170f2f79c020f0933f0eac8f024684af.jpg)  
Figure 1. PageWeaver overview. Fixed TopK pages guide bounded query regrouping. Original IDs carry support, causal positions, and output addresses through a two-CTA Union8 kernel. The illustrated page sets are a subset; measured groups use up to eight queries and Top16.

PAGEWEAVER instead uses the page relation to change the members of a query-owned group (Figure 1). Within a bounded window, a GPU kernel greedily collects queries whose selected sets add few new pages. An ID-aware Union8 implementation then executes every query’s complete attention and writes directly to its original output row. The selector is unchanged; membership and causal masks follow original query IDs. This design trades an online grouping cost for a smaller union without introducing crosspage output reduction.

Contributions and scope. We make three contributions. (1) We connect query-group size, page-set affinity, and output ownership in a common execution model, with direct KV-page union as a measured design alternative. (2) We implement bounded GPU regrouping and an ID-aware FP8 sparse-attention path on Hopper, retaining a complete query output within one CTA. (3) We separate base-kernel gains, incremental grouping gains, and whole-model prefill effects, including numerical comparisons and an auxiliary Blackwell study. The main evidence is an H200 MiniMax-M3 case study; selected operator captures and limited service repetitions constrain the generality of its conclusions.

## 2 RELATED WORKS

Selecting sparse support. MSA uses a block indexer and sparse main attention (Lai et al., 2026). BLASST prunes blocks using online softmax statistics (Yuan et al., 2026); LServe combines sparse attention with long-context serving (Yang et al., 2025). These approaches choose or exploit a reduced set of interactions. Our input is an already selected support: we reorganize its execution without adding or removing any query’s selected edges. BLASST’s pruning rule and its quality calibration are therefore separate from the grouping problem studied here.

Grouping and sparse execution. LiteDSA unions adjacent queries’ indices and applies per-query membership masks (Yin et al., 2026; LiteTopK Contributors, 2026). MSA’s KV-outer implementation chunks popular pages into bounded query work units and combines partial results (Lai et al., 2026). Both query grouping and page-oriented execution are existing techniques. Our direct page-pair path evaluates another point in this design space, while the final path changes query membership within a bounded search window and retains query-owned accumulation.

Reordering and kernel infrastructure. Sparse row reordering also seeks denser Tensor Core work. DTC-SpMM clusters rows by column-set similarity and cache reuse, with reordering amortized as preprocessing (Fan et al., 2024). Fused3S fuses sparse attention within row-owned blocks, schedules row windows by work count, and remaps operand layouts (Li & Chandramowlishwaran, 2025). Complete row ownership and similarity-based grouping are established techniques. Our focus is their bounded, per-call realization for dynamic page-selected attention: grouping is charged on every call, and original causal coordinates must survive the permutation. FlashAttention-3, FlashInfer, and FlexAttention provide complementary attention execution techniques (Shah et al., 2024; Ye et al., 2025; Dong et al., 2025). Our comparisons identify each backend’s runtime and arithmetic rather than treating all FP8-KV kernels as interchangeable.

## 3 METHODOLOGY

## 3.1 Fixed support, variable execution groups

Consider one request and one KV head; its associated query heads share a selected-page set. Let $S _ { i }$ be the valid, deduplicated logical pages selected by query i, P the page size, c<sub>i</sub> its original causal position, and $L _ { i }$ the valid context length. The supported tokens are

$$
A _ { i } = \{ t : \lfloor t / P \rfloor \in S _ { i } , 0 \leq t < L _ { i } , t \leq c _ { i } \} .\tag{1}
$$

For each query head, with $z _ { i t } = q _ { i } ^ { \top } k _ { t } / \sqrt { d } ,$

$$
o _ { i } = \frac { \sum _ { t \in A _ { i } } \exp ( z _ { i t } ) v _ { t } } { \sum _ { t \in A _ { i } } \exp ( z _ { i t } ) } .\tag{2}
$$

Head indices are suppressed. Invalid IDs contribute no edges, duplicate IDs do not duplicate probability mass, and empty support produces zero output. Logical page IDs are resolved through the request’s physical KV map.

Define the incidence matrix $M _ { i p } = { \bf 1 } [ p \in S _ { i } ]$ . For a group $G$ of $g$ queries, $\textstyle U _ { G } = \bigcup _ { i \in G } S _ { i }$ is its page union. Across a partition ${ \mathcal { G } } ,$ logical page reuse is

$$
\rho _ { Q } ( \mathcal { G } ) = \frac { \sum _ { i } \left| S _ { i } \right| } { \sum _ { G \in \mathcal { G } } \left| U _ { G } \right| } .\tag{3}
$$

An execution rectangle has up to $g | U _ { G } |$ query–page slots but only $\textstyle \sum _ { i \in G } | S _ { i } |$ selected edges. Masks preserve support; masked Tensor Core slots are not generally free. Figure 2 holds the selected edges fixed and illustrates the effect of changing the groups.

## 3.2 Group size and output ownership

Larger groups can improve reuse and tile utilization, but increase union expansion, staging, or synchronization. We compare implemented Union4 and Union8 configurations under matched arithmetic. H200 Union8 uses an eight-query transfer group and two four-query compute subgroups. Its logical transfer proxy is $\sum _ { G } | U _ { G } |$ ; a closer compute proxy is

![](images/80af4c550360d2330a60a11c2f924cfd0a223cb41936233f8390a4921e216d71.jpg)  
Identical 16 selected edges; g = 2 illustrates the mapping. Measured groups use g = 8.  
Figure 2. The same support can induce different amounts of work. Adjacent groups mix two page communities; the page-side relation reveals nonadjacent queries sharing support. Regrouping reduces logical page visits from 16 to 8 and masked query–page slots from 16 to 0 in this illustrative example. All 16 selected edges are retained. The diagram uses two-query groups and two selected pages; the measured implementation uses groups of eight and Top16.

$$
C _ { 4 } = \displaystyle \sum _ { G \in \mathcal { G } } \big ( | U _ { G _ { 0 } } | + | U _ { G _ { 1 } } | \big ) ,\tag{4}
$$

where $G _ { 0 } , G _ { 1 }$ are the two CTA subgroups. A subgroup can skip a page that none of its queries selects. These proxies describe different work: minimizing the full union need not minimize both subgroup rounds or balance the two CTAs. We use measured complete-call time to select the implemented group size.

The page-oriented alternative. Let $R _ { p } = \{ i : p \in S _ { i } \}$ For a pair of pages $( a , b ) , R _ { a } \cap R _ { b }$ identifies common queries, possibly far apart in sequence position. Accumulating both pages before writing a partial can save one partial per common query. For disjoint pairs, the potential reduction is $\sum _ { ( a , b ) } \left| R _ { a } \cap R _ { b } \right|$ |. Exclusive queries still require single-page work, and every selected edge must be covered exactly once.

This organization partitions a query’s output across page groups. If group r produces maximum $m _ { r } ,$ denominator $\ell _ { r } ,$ , and normalized partial $o _ { r }$ , let $\mathcal { R } = \{ r : \ell _ { r } > 0 \}$ . For nonempty R, the stable merge is

$$
o = \frac { \sum _ { r \in \mathcal { R } } \ell _ { r } e ^ { m _ { r } - m _ { o _ { r } } } } { \sum _ { r \in \mathcal { R } } \ell _ { r } e ^ { m _ { r } - m } } , \qquad m = \operatorname* { m a x } _ { r \in \mathcal { R } } m _ { r } .\tag{5}
$$

An empty R produces zero output. Preparation, partial writes, and this reduction belong to the page path’s cost. Our final design uses the same incidence information to find query neighbors while retaining complete query outputs in the query-oriented kernel.

## 3.3 Bounded online grouping

Page affinity between two queries is conceptually $( M M ^ { \top } ) _ { i j } = | S _ { i } \cap S _ { j } |$ . We do not construct this dense matrix. Within a window of W queries, Algorithm 1 grows each group by minimizing the marginal union increase:

$$
i ^ { * } = \arg _ { i } \operatorname* { m i n } _ { \substack { i \mathrm { u n a s s i g n e d } } } | S _ { i } \setminus U | .\tag{6}
$$

Ties use original query order. The execution size is fixed at $g = 8$ , independently of the search window W. Windows remain within request boundaries and within the current prefill chunk. The method rebuilds IDs from current Top-K sets on every call; it does not reuse page-pair decisions or previous requests’ routes.

Preserving semantics. Execution slot r reads $q _ { \pi ( r ) }$ , uses membership $M _ { \pi ( r ) , p }$ and causal position $c _ { \pi ( r ) }$ , and writes $o _ { \pi ( r ) }$ The permutation is bijective within each request. Thus each original query sees exactly the same supported token set $A _ { i }$ in Equation 2; no probability mass or softmax state is shared between queries. This is a supportpreservation argument in exact arithmetic. Quantization and reduction order remain separate numerical considerations.

Algorithm 1 Windowed page-affinity grouping   
Require: Selected sets $S _ { i }$ , request boundaries, W, $g = 8$   
Ensure: Request-local query permutation π   
1: for each request-local window B do   
2: if context exceeds the bitmap capacity then   
3: Append original query IDs of B to π   
4: else   
5: $R \gets B$   
6: while $R \neq \emptyset$ do   
7: i ← min R; $G  [ i ] ; U  S _ { i }$   
8: $R \gets R \backslash \{ i \}$   
9: while $| G | < g$ and R ̸= ∅ do   
10: j ← arg mi $\mathrm { n } _ { j \in { \cal R } } ( | S _ { j } \setminus U | , , j )$   
11: Append j to $\bar { G } ; U  U \cup S _ { j }$   
12: $R  R \backslash \{ j \}$   
13: end while   
14: Append G to π   
15: end while   
16: end if   
17: end for

## 3.4 When grouping pays for itself

Let $T _ { 0 }$ be the complete original-order Union8 call, $T _ { \mathrm { f i x e d } }$ the corresponding ID-aware call with a precomputed permutation, and $T _ { \mathrm { o n l i n e } }$ the call including GPU grouping. Define

$$
\begin{array} { r } { S _ { \mathrm { e x e c } } = T _ { 0 } - T _ { \mathrm { f i x e d } } , \quad } \\ { C _ { \mathrm { o n l i n e } } = T _ { \mathrm { o n l i n e } } - T _ { \mathrm { f i x e d } } , } \\ { S _ { \mathrm { n e t } } = S _ { \mathrm { e x e c } } - C _ { \mathrm { o n l i n e } } . } \end{array}\tag{7}
$$

The design is useful when $S _ { \mathrm { n e t } } > 0$ . The difference $C _ { \mathrm { o n l i n e } }$ is an operational measure, not necessarily identical to a separately timed grouping kernel. Smaller unions may save traversal and arithmetic while nonadjacent IDs add address indirection or reduce locality. Bounded windows constrain both search cost and displacement. All required quantization, packing, metadata, attention, and output processing remain inside the measured call.

## 4 KERNEL DESIGN

## 4.1 GPU grouping and ID-aware preparation

One grouping CTA owns a window. W64 and W128 launch 64 and 128 threads; each query thread represents its Top16 pages with sixteen 32-bit bitmap words. Population counts of bitmap differences evaluate Equation 6. Warp reductions and a shared-memory winner reduction select the next query with deterministic tie breaking. The union resets after eight selections. Only an int32 query-ID array is produced; Q is not physically reordered.

The current capacity is 512 logical pages, or 65,536 tokens at the H200 page size of 128. Larger contexts use an identity permutation. Tail windows and partial groups preserve bijection. With N queries and bitmap capacity B pages, total grouping arithmetic scales as $O ( N W \lceil B / 3 2 \rceil )$ ; independent windows run in parallel.

The union-preparation kernel gathers original Top-K rows through these IDs, deduplicates group pages, and emits membership masks. Membership bits address execution slots; causal positions, request maps, and final stores address original queries. The existing FP8 KV packing path is retained and included in timing. No input-dependent CPU decision or host synchronization is added.

## 4.2 Two-CTA Union8 execution

The attention kernel launches clusters of two 128-thread CTAs (Figure 3). Four queries times sixteen local Q heads provide 64 rows per CTA, mapping to an M64 Tensor Core tile. The cluster’s eight-query union determines the page traversal. Each CTA applies its own four-query membership and causal masks, and retains FP32 running softmax statistics through the complete traversal.

Two KV staging slots separate readiness from reuse. The producer publishes staged data to the cluster; a slot becomes reusable only after the scheduled consumers finish. A CTA with no selected query on a page may skip its arithmetic but must still respect this lifetime protocol. Query grouping therefore changes the shared traversal and the number of useful subgroup rounds, rather than simply doubling independent CTA work.

The H200 path accepts BF16 Q and FP8 E4M3 KV. It quantizes Q per head-row for FP8 QK, keeps the softmax statistics in FP32, and scales probabilities before E4M3 conversion for FP8 PV. The compensating scale enters normalization; output is BF16. These arithmetic choices are fixed between original-order and regrouped Union8. Each CTA finishes its queries locally and stores directly to their original rows, eliminating a separate reorder-output pass and page-partial combine.

## 4.3 Alternative execution and runtime boundaries

The direct KV-page implementation inverts incidence, proposes disjoint page pairs, and forms common/exclusive query lists using full membership. Its CUDA producer accumulates paired pages into one local softmax state and writes a normalized partial plus log-sum-exp. A combine kernel finishes each query. This path evaluates a different execution organization; it changes work lists, producer mapping, and workspace together. Its selected padded partial workspace is approximately 1 GiB for 16K queries.

The SGLang bridge applies W128 regrouping to each current prefill chunk in all 57 sparse layers. Original request offsets remain authoritative in packed batches. The measured service has no learned or layer-selective scheduler.

2 Both CTAs released slot 0: load page 2.  
![](images/80f7fb8d0cf10510169969a9d9ed0c4251cb04cefa657f9941c20db72a183350.jpg)  
CTA 0 / thread 0 multicasts each 32 KiB KV page to corresponding local slots in both CTAs.  
Empty subgroups still honor buffer lifetime; stage widths illustrate dependencies, not measured time.

Figure 3. Hopper execution and buffer ownership. Each CTA has four warps, four queries, and two local 32 KiB KV slots. Memory and compute lanes share an illustrative left-to-right schedule. Event 1 marks KV readiness; event 2 permits slot 0 to be refilled only after both CTAs release page 0. The same warps execute QK, softmax, and PV, retaining each query’s state and original output ID. Empty subgroups still honor buffer lifetime. Widths are schematic, not measured durations.

The B300 port maps the same IDs into a CuTe Union8 path with direct FP8 Q and physical page size 64. It includes the copy into a stable descriptor workspace required by that callable. These runtime and arithmetic differences are accounted for through within-platform comparisons.

## 5 EXPERIMENTS

## 5.1 Setup and evaluation questions

We ask four questions: how much does the base Union8 execution improve over compatible kernels; how much additional work and time does online regrouping save; how much of the gain reaches whole-model prefill; and under what conditions does it disappear? H200 is the primary platform, with MiniMax-M3, FP8 E4M3 KV, Top16, head dimension 128, sixteen Q heads per local KV head, and page size 128. We use the fixed SGLang CUDA12.9 image and the pinned native SM90 implementation (SGLang Contributors, 2026). FlashInfer is a compatible completeoperator baseline. The measured native interface is adapted to the fixed runtime without changing its device algorithm.

Measurement boundaries. Every complete operator call includes live preparation, quantization, packing, attention, and combine when required. CUDA Graph replays exclude compilation and reusable allocation; B300 also has eager event measurements. Backend order rotates on one device, with profiling collected separately. The service rotates three arms in one resident model after draining outstanding requests: native, original Union8, and online W128 plus Union8. It fixes TP8/DP2 with attention-TP4, effective 16K prefill chunks, disabled radix cache and CUDA Graphs, and one output token. There are 18 unmeasured warmup and 54 measured blocks, totaling 675 completed requests. Input IDs and token counts match across each cell’s arms. Table 1 distinguishes selected captures from independent workloads.

Table 1. Evaluation units and their scope. Replays measure timing variability; they do not create additional independent documents.
<table><tr><td>Study</td><td>Workloads and repetitions</td></tr><tr><td>Base execution</td><td>Six captures, one request; L3/L32/L59, 0/16K prefixes. Seven rounds × 20 replays.</td></tr><tr><td>Online grouping</td><td>Five captures of one constructed 64K input: L39/r0–3, L43/r1; plus one 16K L51 control.</td></tr><tr><td>H200 service</td><td>Eleven rounds × 50. Five frozen prompts per 8K/32K/64K bucket; concurrency 1/4; three measured</td></tr><tr><td>B300 auxiliary</td><td>blocks per arm/cell. Eight TP4/rank0 captures: L3/L20/L32/L59 at two chunks. Seven rotated graph/eager rounds.</td></tr></table>

## 5.2 Base execution: the value of a larger union group

Figure 4 separates total implementation performance from group-size effects. Across six captures, Union8 achieves a 1.701× geometric-mean speedup over FlashInfer and 2.662× over native. Although all receive FP8 KV bytes, backend families differ in Q and probability arithmetic. The matched-arithmetic Union4-to-Union8 comparison gives a 1.075× geometric-mean speedup, with bitwise-identical outputs on these inputs. Eight queries provide a useful Hopper execution unit; the result does not establish a universal optimum over group sizes.

## 5.3 From nonlocal page reuse to query regrouping

An eight-source development survey contains 2,736 correlated layer/rank/chunk observations across 57 sparse layers and four attention ranks. Per-source median reuse ranges from 3.85 to 4.54, with lower-overlap cases in particular later chunks. This motivates support-dependent grouping, but does not establish a universal layer policy or the frequency of performance wins.

Controlled interleaved supports make the page-side opportunity explicit: adjacent queries select different page communities, while distant queries share them. Direct KV-page union lowers complete-call time relative to FlashInfer by 15.6% at 32K context (1.340 versus 1.588 ms) and 10.1% at 64K (2.852 versus 3.172 ms). Figure 5 contrasts these constructed inputs with two model captures, where Union8 is faster. In the deliberately low-overlap L43/r1 64K capture, page union also takes 1.827 ms versus 1.368 ms for Union8. Thus nonlocal reuse exists, but work-list and partial-output execution can absorb its value. This motivates using page relations for group membership while preserving queryowned accumulation.

## 5.4 Online regrouping: saved work, paid overhead, net gain

The five 64K captures have 15,272 current queries following a 49,152-token prefix, for 64,424 context tokens. Online W128 lowers complete-call latency by 3.26–7.66% relative to original Union8; the 16K high-overlap control improves by 4.11%. W64 and W128 grouping cost approximately 19.3–19.5 and 34.6–34.9 µs when timed alone. These costs are included in the online comparison. W64 wins over W128 on L39/r1, so a larger search window is not universally better.

Figure 6 exposes the cost balance rather than reporting only final speedups. On L43/r1, W128 reduces logical union visits from 102,429 to 89,426, or 12.70%. Using medians from the same rotated experiment, original Union8 takes 1.36633 ms, execution with fixed grouped IDs takes 1.27037 ms, and online grouping plus execution takes 1.30792 ms. In Equation 7, execution saves $9 5 . 9 6 \mu \mathrm { s } ,$ , the online increment costs $3 7 . 5 5 \ : \mu \mathrm { s } .$ , and the net saving is 58.41 µs. The grouping-only measurement is $3 4 . 6 5 \ \mu \mathrm { s } ;$ it should not be substituted for the operational difference. Logical visits describe structural work, not measured HBM traffic.

Affinity and indirection matter. A separate L43/r1 diagnostic increases latency from 1.371 to 1.832 ms with random query IDs. Fixed W512 affinity IDs execute in 1.234 ms; materializing reordered Q/TopK and restoring output order takes 1.315 ms. This approximately $8 2 \mu \mathrm { s }$ difference supports direct ID-based access. Global hash ordering followed by local grouping takes 1.298 ms, showing that similarity must be considered together with locality. W2048 reaches 1.185 ms execution time but requires 1.50 seconds of CPU search. These offline results measure execution potential, not deployable online speedup (Appendix A).

## 5.5 Whole-model prefill: total and incremental benefit

Figure 7 reports input-token throughput, including one generated token per request. The final path’s median throughput is 7.88–14.36% above the tested native service. Most of this advantage is already present in original Union8. Regrouping changes the median by +0.47–0.73% at 32K/64K, and by $- 0 . 5 6 \% / - 1 . 3 1 \%$ at 8K with concurrency 1/4. These are observed median differences, not established population-level improvements.

Pairing the three 64K/C1 rounds yields +0.630%, +0.666%, and +0.641%. Other cells show larger variation. For 64K/C4, one original-Union8 block drops to 26.51K token/s while the other two reach 34.82K and 34.83K; the paired ratios therefore include a +32.36% outlier alongside approximately +0.43% and +0.47%. We retain it rather than filtering after observing the result.

Actual token ranges are 4,984–8,270, 30,563–32,842, and 63,997–65,666. Two of five unique prompts in the 64K bucket cross the bitmap capacity; affected chunks use identity IDs. All arms receive the same inputs, all requests complete without reported errors, and reported cached tokens are zero. No regrouping policy is selected per layer. The service comparison does not include FlashInfer; consequently it measures gain over the pinned native integration, not over every available serving backend. A complete service-stage trace and more independent repetitions are needed to attribute and validate the small increment.

## 5.6 Numerical checks and quality scope

The implementation checks request-local bijection, membership preservation, causal and partial pages, invalid/duplicate IDs, empty support, packed requests, and fresh-input graph replay. GPU grouping IDs match the CPU greedy reference. Compute Sanitizer reports no memory errors. All six online

![](images/93424661a7c6965b979b4457d69b8744c14a9bf41fc4138500f2c02a5ecea617.jpg)

![](images/83e4b3d7ca5f189f25a61252dfbec70c57e77411759685433d77a34e2e4b3505.jpg)

Figure 4. Base-kernel and group-size gains on H200. Complete-call timings use the same six captures and FP8 KV bytes. Backend families can differ in Q/P arithmetic; Union4 and Union8 use matched arithmetic. Repeated-round ranges show measurement variability. L denotes layer and p the prefix preceding the current 16K-query chunk.  
![](images/0d5d62e43fed276d924edd694a7cfa69f27ce9356e55e308056f561b05c50167.jpg)

(b) Captured model support  
![](images/9844996d1fa2e9a16f65019b93efb5df3fd2939a711064c592b8f3a76bbc35dd.jpg)  
Figure 5. Direct page execution exposes an opportunity and its cost. Controlled interleaved supports and model captures are distinct workload groups. Every timing includes required preparation and output processing. The constructed cases demonstrate a possible advantage of KV-page union; they do not estimate its prevalence in a model. Ranges cover repeated timing rounds.

H200 captures are bitwise identical to original Union8; this observation is limited to the tested inputs.

For output O and reference R, we report $\mathrm { N R M S } = \Vert O -$ $R \| _ { F } / \operatorname* { m a x } ( \| R \| _ { F } , \epsilon )$ . Table 2 compares the base backends against the same Triton reference. Union8’s differences are close to FlashInfer’s on this set, while native differs more. These are numerical comparisons to a reference implementation, not downstream accuracy scores. On the separate online captures, a flattened FP32 reference sampled at 33 query rows per input yields NRMS of approximately 2.66–4.51%;

regrouping matches its parent output, including those sampled differences.

Preserving the selected edges does not remove the parent’s FP8 approximation. The latest one-token service workload validates execution and throughput, not long-context generation quality. A paired task-quality evaluation of the final policy remains necessary before a quality-qualified deployment claim.

![](images/1c93681dafc5a7a6a107d831a4e577be02a769ebc307d76a7f23567dbec361c3.jpg)

![](images/c030a27a331bb5589cac840e03934bdb3fbf6ffbdc6941b345c03c61a4d74c6d.jpg)

Figure 6. Explaining the incremental regrouping gain. The complete online call is compared with original-order Union8 and execution using fixed grouped IDs on the same capture. Execution savings and the additional online cost determine the net saving (Equation 7). Structural statistics come from separate diagnostics of the same captures; page visits are logical counts, not HBM bytes. Timing repetitions do not expand the document sample size.  
(a) H200 service  
![](images/850547e748e55b0aea143cd978f1d97837e275cb82a3341c0cff8c02d844ce57.jpg)

(b) All rounds  
![](images/823922205b06a19b45dcd0afc693637e37e03fccee08306c1b23e007f7bd9dfe.jpg)

(c) Zoom near zero  
![](images/0bce840e7deb504fca057380bea34ab74e19af7902645a65a0def2436ac760d4.jpg)  
Figure 7. Whole-model H200 prefill and repeated regrouping effects. Three arms use identical prompts within each length/concurrency cell. Throughput medians and ranges retain every block; the incremental view pairs online and original Union8 within each round. Large positive ratios arise when an original-Union8 block slows down, and remain visible. Three technical repeats do not establish statistica significance across documents.

## 5.7 B300 and the limits of regrouping

The auxiliary B300 experiment uses CUDA13/SM103, direct FP8 Q, physical page size 64, and eight TP4/rank0 captures. W128 gives a 1.068–1.156× eager speedup over original Union8 on L20/L32, but native remains faster there. For L20’s second chunk, original Union8, W128, and native take 1.497, 1.295, and 0.592 ms. On L3’s first chunk, their times are 0.475, 0.500, and 0.547 ms: original Union8 already wins, and regrouping adds cost. Figure 8 shows that neither online window beats the faster existing backend on any of the eight tested inputs. No B300 serving experiment follows from this result.

![](images/d81d2659538db317c004d64f5bbd5f17f8135cb9c9a5b8d768269947936bfa47.jpg)  
Figure 8. B300 auxiliary comparison. Each input is normalized to its faster existing native/original-Union8 backend. Online grouping can improve Union8 while still losing to native. These are complete eager operator timings, not service throughput or a controlled comparison of H200 and B300 hardware.

Table 2. Numerical differences on six base-kernel captures. All rows use the same Triton reference and full output. Ranges are across inputs, not confidence intervals.
<table><tr><td>Backend</td><td>NRMS (%)</td></tr><tr><td>Union8 FlashInfer</td><td>2.246–3.967 2.288-4.077</td></tr><tr><td></td><td></td></tr><tr><td>Native SM90</td><td>6.258–9.120</td></tr></table>

The B port preserves support and has at most $1 . 5 5 \times 1 0 ^ { - 5 }$ full-output NRMS relative to its parent. Its lack of an overall win bounds the current implementation, not page-affinity grouping in general. Native algorithms, input sets, Q entry formats, page sizes, and descriptor costs differ between the two platforms; hardware alone cannot explain the contrast.

Limitations and next tests. The operator evidence emphasizes selected inputs; independent natural-text validation should precede a general scheduling rule. The fixed 64K bitmap limits active regrouping on longer contexts. The present greedy objective minimizes the eight-query union, not the subgroup compute proxy in Equation 4. A future selector could consider both, but no such cost-aware layer policy is evaluated here. More service repetitions, explicit stage attribution, a strongest-compatible service baseline, and final generation quality are the next evidence requirements.

## 6 CONCLUSION

PAGEWEAVER uses the relation between selected queries and KV pages to improve execution groups while retaining each query’s full attention output. A bounded online search and direct query-ID access connect this relation to a two-CTA FP8 Union8 kernel. The H200 study separates substantial base-kernel gains from a smaller additional regrouping benefit: the latter reduces complete-operator latency on selected inputs, but yields only modest and workloaddependent service changes. Direct KV-page execution and the B300 comparison identify why reuse must be evaluated together with preparation and output costs. The central design principle is to optimize the cost of executing a fixed sparse relation, including the cost of constructing its execution groups.

## REFERENCES

Dong, J., Feng, B., Guessous, D., Liang, Y., and He, H. Flex attention: a programming model for generating optimized attention kernels. In Proceedings of Machine Learning and Systems, 2025. URL https://proceedings. mlsys.org/paper\_files/paper/2025/fil e/61a9278dfef5f871b5e472389f8d6fa1-P aper-Conference.pdf.

Fan, R., Wang, W., and Chu, X. DTC-SpMM: Bridging the gap in accelerating general sparse matrix multiplication with tensor cores. In Proceedings ofthe 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3, 2024. doi: 10.1145/3620666.3651378. URL

https://home.cse.ust.hk/\~weiwa/paper s/dtc-spmm-asplos24.pdf.

Lai, X., Xu, W., Yang, Y., Chen, Q., Xu, Y., Zeng, L., Li, X., Sun, H., Zhu, H., Zhang, V., Hu, J., Li, J., Gao, R., Li, Z., Zhu, S., Zhou, J., and Zhao, P. MiniMax sparse attention. arXiv:2606.13392, 2026. URL https: //arxiv.org/abs/2606.13392.

Li, Z. and Chandramowlishwaran, A. Fused3S: Fast sparse attention on tensor cores. In Proceedings ofthe 39th ACM International Conference on Supercomputing, 2025. doi: 10.1145/3721145.3730430. URL https://arxiv. org/abs/2505.08098.

LiteTopK Contributors. LiteTopK and LiteDSA: Sparse attention kernel implementation, 2026. URL https: //github.com/Heisenberg-Yin/LiteTopK. Generic FP8 LiteDSA query grouping and membership masks; accessed September 28, 2026.

SGLang Contributors. MiniMax-M3: Add native SM90 Q8KV8 sparse prefill step 3. GitHub PR 37236, 2026. URL https://github.com/sgl-project/s glang/pull/37236. Benchmark source pinned to head 62bae6102875a4ecb16d764df4f7622fe7fd0347 on September 18, 2026.

Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., and Dao, T. FlashAttention-3: Fast and accurate attention with asynchrony and low-precision. arXiv:2407.08608, 2024. URL https://arxiv.org/abs/2407.0 8608.

Yang, S., Guo, J., Tang, H., Hu, Q., Xiao, G., Tang, J., Lin, Y., Liu, Z., Lu, Y., and Han, S. LServe: Efficient long-sequence LLM serving with unified sparse attention. arXiv:2502.14866, 2025. URL https://arxiv.or g/abs/2502.14866.

Ye, Z., Chen, L., Lai, R., Lin, W., Zhang, Y., Wang, S., Chen, T., Kasikci, B., Grover, V., Krishnamurthy, A., and Ceze, L. FlashInfer: Efficient and customizable attention engine for LLM inference serving. arXiv:2501.01005, 2025. URL https://arxiv.org/abs/2501.01005.

Yin, Z., Gao, J., Yin, P., Li, J., and Cong, G. Lite-TopK: Exploiting the curse of dimensionality for a fused indexer-TopK kernel in long-context sparse attention. arXiv:2607.11976, 2026. URL https://arxiv.or g/abs/2607.11976.

Yuan, J., Shinn, C., Xu, K., Cui, J., Klimiashvili, G., Xiao, G., Zheng, P., Li, B., Zhou, Y., Ye, Z., You, W., Zheng, T., Brown, D., Wang, P., Hoehnerbach, M., Cai, R., Demouth, J., Owens, J. D., Hu, X., Han, S., Liu, T., and Mao, H. BLASST: Dynamic blocked attention sparsity via

softmax thresholding. arXiv:2512.12087, 2026. URL ht tps://arxiv.org/abs/2512.12087. MLSys 2026.

![](images/6ac4c4e693efd1cbaf1f7f4b834d7006665db1ff070612427b6360fe6d73d775.jpg)

![](images/119a778767af4ed5f56f3f0f75c6ae3e025aa6ad58d252e32043bf13dcfda5ce.jpg)  
Figure 9. Offline execution potential; grouping search excluded. Fixed-ID and materialized variants use the same L43/r1 capture. Random IDs degrade locality and grouping; affinity reduces execution time. Materialization adds tensor movement. W2048’s approximately 1.50-second CPU search is outside its execution bar and makes it unsuitable as an online implementation.

Figure 9 examines how affinity and data movement affect execution with fixed permutations on L43/r1. The CPU grouping search is excluded from every offline bar. Larger windows expose more potential reuse but can be prohibitively expensive to search. The W2048 CPU search takes approximately 1.50 seconds, compared with 1.185 ms for the resulting attention execution. It is an explanatory diagnostic rather than an online candidate.

## B REPRODUCIBILITY AND INTERPRETATION

The artifact contains the figure builders, source hashes, selected-kernel snapshots, per-round timing JSON, service block summaries, and numerical checks. Raw model weights and large captured GPU tensors are stored separately from the repository. Figures are regenerated from archived measurements; no new GPU run was performed to construct this manuscript revision.

H200 runtime. The image is the fixed SGLang perf-m3-h200-d5658f11-cu129 build with digest a1344259...85631; the artifact records the full digest. Native SM90 source is pinned to PR 37236 head 62bae610...0347. Source-hash verification and perlayer backend-hit logs check that service arms use their intended paths. Original and regrouped H200 Union8 both use two KV staging slots; the historical directory name union8\_lb3 refers to launch bounds, not three stages.

Timing and sampling. H200 base-kernel results use seven rotated rounds of twenty graph replays; online grouping uses eleven rounds of fifty. Service cells use three measured blocks per arm, each containing 5C requests for concurrency C, drawn from five unique prompts. Median throughput is computed per arm and cell. Paired round ratios answer a different question and are shown separately. Neither replays nor repeated prompts create independent documents. Profiling is separate from performance timing.

Direct KV-page diagnosis. An earlier N256 Triton producer used 255 registers per thread, with 12.4% achieved occupancy and 96.5% L2 hit rate in one NCU run. These measurements motivated the later CUDA producer, but they are not a profile of that producer. The direct-page path changes several execution choices simultaneously; its results cannot isolate output ownership as a single causal variable. The main-text controlled workloads demonstrate a possible reuse pattern, while captured workloads retain their own provenance and comparison scope.