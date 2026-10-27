# One Megakernel is Not Enough: Predicting Where to Cut for Streaming Inference

Aditya Chatterjee   
NVIDIA   
Bangalore, India   
adichatterje@nvidia.com   
Anand Joseph X M   
NVIDIA   
Hyderabad, India   
anajoseph@nvidia.com   
Sanjay Singh Chauhan   
NVIDIA   
Pune, India   
schauhan@nvidia.com

Abstract—A megakernel is a large monolithic kernel that fuses multiple kernels/operators into a single persistent kernel. We show that the best configuration is a composition of multiple megakernels with no overlapping conventional graph component, or a fully conventional graph. We present a framework that selects megakernel regions in a streaming inference model from conventional-graph profiles and standalone kernel benchmarks, without running any candidate design end to end. Across NH-TTS, Qwen3-TTS and Llama-1B, the best composition ranges from zero to four megakernels at different batch sizes. We introduce four model categories (megakernel-stable/ receding/ advancing/ averse) that capture how a model’s best megakernel composition changes with batch size. Using NH-TTS model on H100 NVL GPU, we demonstrate that preserving three specialized megakernels can improve both latency and throughput despite introducing additional kernel boundaries.

Index Terms—ML inference, Megakernel, Kernel fusion, GPU performance modeling

## I. INTRODUCTION

Time to First Audio (TTFA)/Time to First Token (TTFT), Inter-Token Latency (ITL)/Inter-Chunk Latency (ICL) and Throughput (RTFx)/Realtime Factor (RTF) are the key metrics for real-time serving of streaming inference models. So long as the RTFx is greater than 1, the overall latency of the response is dictated by the TTFA / TTFT. Minimizing TTFA/TTFT is critical in applications that support bidirectional streaming (input and output). Being able to serve requests at the maximum batch size while keeping these metrics within acceptable limits is critical for real-time serving.

A megakernel fuses multiple operators into one persistent kernel whose thread blocks stay resident on the GPU and hand-over work to one another without returning to the host. Large Language Models (LLMs) operate in two phases per request - prefill (run once) and decode (run autoregressively token by token). Existing megakernel designs usually fuse the decode phase and leave prefill to conventional execution or out of scope. In [1] up to 1.7× lower per-token latency than vLLM [2] and SGLang [3] is reported for small-batch LLM decoding. Multi-modal models like text-to-speech (TTS) operate in multiple phases. A TTS model runs prefill once per request, an autoregressive producer once per step, a local transformer N times per step and an audio codec once per chunk of frames. These phases run at different rates and prefer different thread-block shapes. These phase differences make fusion boundaries a performance choice. Each cut decides which phases share one launch configuration, how long thread blocks stay held and how the phases share the GPU.

We evaluate an upcoming autoregressive TTS model with a Nemotron-H backbone [4] denoted as NH-TTS in this paper. Figure 1 shows the six representative NH-TTS layouts at batch size 32. For NH-TTS at batch 32, Figure 2 shows that the three-megakernel design publishes the first audio 4.05× sooner than the conventional TensorRT [5] pipeline and 21.4% sooner than a single megakernel for the whole model. Across 13 combinations of model and batch size on three frameworks, the best tested design uses anywhere from zero to four megakernels. Qwen3-TTS [6] needs four up to batch 32 and none from batch 48, where the same cuts make it slower. Guessing is costly: for NH-TTS and Qwen3-TTS, the next-best layout is 3 to 29% slower than the best.

![](images/dd060801776fcb79ffc33bdfb309b9098835d6af84cc009044f0165506e63ebd.jpg)  
Fig. 1. Six representative NH-TTS layouts from the 81-candidate design space. The three-megakernel design with the lowest measured first-publication latency at batch 32 on H100 NVL is highlighted.

![](images/256bc19bba20d4b3fb38ee27ad2f9a6bd73dc82bce6ec9da07f40071f1d609eb.jpg)

![](images/c3cde9f8d5fd4e065c92a6042a55cba3ed5c6e0ab6a1b364ab133772a5449888.jpg)  
Fig. 2. TTFA and RTFx for NH-TTS at batch size 32 with the selected threemegakernel design

Exhaustively implementing and timing compositions is costly: our NH-TTS search contains 81 candidates. We address first-publication-aware selection among supported implementations by combining compatible service measurements with persistent resource lifetimes and streaming publication dependencies.

We present a boundary planner that ranks supported compositions using conventional-graph profiles, operator benchmarks and source-declared implementation contracts. These contracts specify geometry, storage, compiled-entry membership and synchronization. The planner combines their costs with streaming dependencies to select boundaries and sharing policies without timing candidate compositions end to end.

We evaluate the planner on H100 NVL GPU with three models on three frameworks: NH-TTS (TensorRT), Qwen3- TTS (vLLM [2]) and Llama (PyTorch [7]). We show that the planner selects the measured best design in all 13 model-batch cases.

## II. BACKGROUND

In codec-based autoregressive TTS, prefill initializes the request context. The acoustic-token generator (producer) emits frames of discrete acoustic codes and the neural codec decoder (consumer) converts buffered frames into PCM waveform samples. Producer and consumer denote scheduling roles relative to the acoustic-code buffer. A frame denotes one codec time step; a chunk groups frames for decoding and publication.

## A. Streaming Inference and First Publication

A model’s streaming schedule states when output may be published. A TTS model may wait a few steps before speech starts, decode its first chunk once enough frames exist and publish that chunk only after one more producer step shows that the stream continues. First publication occurs when the first audio chunk or token becomes available. The planner minimizes a deterministic first-publication score.

## B. Conventional Graph Execution

Inference frameworks such as TensorRT, vLLM and Py-Torch run a model as a sequence of kernels, typically one per operator or small fused group. Each kernel launches a grid of thread blocks called cooperative thread arrays (CTAs) which the hardware spreads over the GPU’s streaming multiprocessors (SMs). H100 NVL GPU has 132 SMs. A kernel boundary is a barrier: the next kernel starts only after the previous one completes. CUDA graphs record a sequence of launches and replay it with one host call which removes host launch overhead. At small batch size, the kernels are short so host launches, barriers and host synchronization take a large share.

## C. Megakernels

A megakernel is a persistent kernel [1]. It is launched once with no more thread blocks than the GPU can hold at the same time and those blocks stay resident and run many operators in turn from an instruction stream. At a scheduler level, this adds multiple advantages like memory prefetch and tile execution across original kernel boundaries leading to better GPU utilization [8].

Fusion also adds costs. All operators inside a megakernel share one launch configuration lacking specialization and higher register pressure leading to register spilling [9].

## D. Megakernel Designs and the Boundary Selection Problem

When a model is split into several regions, the regions can share the GPU serially, each taking the whole GPU in turn; side by side on a dynamic or static split of thread blocks. A design makes three choices: a partition of the model’s phases into contiguous regions, a backend for each region (conventional or megakernel) and a sharing policy.

The boundary-selection problem is to choose a supported composition using conventional-execution profiles and operator benchmarks. The planner assigns each candidate a deterministic first-publication score and selects the candidate with the lowest score. This score is a ranking proxy, rather than a direct prediction of the latency distribution or its 95th percentile. We evaluate selection quality against the measured latency statistic defined for each model in Section VI, within that model’s implemented validation pool.

## III. COMPOSING MEGAKERNELS: A NEMOTRON-H BASED TTS CASE STUDY

Like a typical autoregressive TTS model, NH-TTS comprises prefill, acoustic-token generation and codec decoding. Each generation step executes the Nemotron-H backbone once and a local transformer N times, once per codebook, to produce one acoustic-code frame. The codec decoder converts chunks of frames into PCM samples. Under the evaluated streaming policy, first publication waits for both the decoded chunk and one further generation step.

The phases run at different rates, use different tensor shapes and thread-block sizes. Each boundary between them is a natural place to cut and each cut has consequences for the whole pipeline.

## A. Multiple Combinations

We compare the conventional baseline of TensorRT with CUDA graphs against multiple native megakernel and conventional graph design combinations. Each design runs at batch size 32 on H100 NVL GPU. Three megakernels (Figure 3) with overlap publish the first audio 4.05× sooner than conventional execution and 21.4% sooner than one megakernel design at batch size 32. The ordering holds at other batch sizes (Bn denotes batch size n): at B64, 47.4 ms against 61.0 ms for one megakernel and 221.5 ms for conventional; at B1, 27.5 ms against 32.7 and 73.0 ms respectively.

## IV. AUTOMATED FIRST-ORDER BOUNDARY DISCOVERY FROM FIRST PRINCIPLES

This section presents our method for identifying which kernel boundaries are worth preserving. It is first-order because it uses only measurements of the conventional graph and of standalone operators, never of a candidate design, and it works from first principles because it composes those measurements through explicit models of tiles, launches and shared GPU capacity.

![](images/73019b72a5051da8cec0f416f93c3014fe7698d950de19eddbcc416e97f77146.jpg)  
Fig. 3. Best megakernel composition for NH-TTS at batch size 32

The planner selects fusion boundaries and execution policies using conventional-graph profiles, standalone operator or step benchmarks, and source-declared implementation contracts. It first constructs a region graph (rGraph) and enumerates supported compositions. It then estimates implementationspecific service costs by constructing an intermediate tile graph (tGraph) and then, constructs an event graph (evGraph) for each composition. Simulating this graph yields a firstpublication score used to rank candidates, without measuring candidate compositions end to end.

## A. Region graph (rGraph) and candidates

We take a model with its runtime implementation as our input. We run the conventional graph implementation of the model using inference frameworks like TensorRT and capture the following information:

• Execution frequency: which model region runs once, every producer step or every audio chunk.

• Workload characteristics: Arithmetic intensity, supported tile geometry, tensor sizes, memory traffic.

• Dependencies: data and state updates which must finish before the current operation can proceed.

• Publication: what computation must complete before the application can receive audio.

This results in Region Graph (rGraph) with three group types:

• Admission group: runs once per request

• Producer group: run once per generated step

• Consumer group: run once per output chunk

The region graph has one node per group with edges for data, state and publication dependencies. A region is a contiguous set of kernels executed by one backend:

• conventional execution with one kernel per operator

• or a megakernel that runs the region in one persistent launch.

A cover partitions a fixed ordering of the groups into contiguous regions and assigns each region a conventional or megakernel backend. Before filtering by implementation support, n groups yield 2 · 3<sup>n−1</sup> covers: the first group chooses either backend, and each subsequent group joins the preceding region or starts a new region with either backend. A candidate combines a cover with one supported execution policy.

An execution policy specifies how the regions of a cover share the GPU:

• Serial policy: regions run one at a time with the whole GPU.

• Fixed split: producer and consumer regions are coresident for the whole run with CTA budgets X and Y.

• Dynamic split: producer uses the whole GPU until the consumer’s input is ready, then continues on X CTAs beside the consumer’s Y CTAs.

The policy is declared from launch configurations that the implementations support and a policy is attached only to the cover where it applies. Overlap requires the producer groups to form one megakernel region and the consumer to be a megakernel. A dynamic split also requires them to be separate launches. A candidate is a cover together with one applicable policy.

A model’s streaming schedule determines how much producer work precedes the first output. Three of its parameters matter:

• a delay of d producer steps whose output is not yet emitted;

• a first chunk of $c _ { 1 }$ steps;

• a lookahead of l steps that shows whether the stream continues.

For NH-TTS, this exposes four groups:

• P: Prefill (admission)

• B: Nemotron-H backbone (producer)

• L: Local transformer (producer)

• C: Codec (consumer)

For NH-TTS, the four ordered groups yield 54 serial covers. Fixed splitting applies to six covers: three with producer and consumer work inside one megakernel and three with separate producer and consumer launches. Dynamic splitting applies only to the latter three. With three supported CTA-budget pairs, these policies add $6 \times 3 + 3 \times 3 = 2 7$ candidates, giving 81 in total.

## B. Tile Graph (tGraph) and Service Table

For each group, we build a tile graph (tGraph) [10]:

• Each supported output tile becomes a task carrying its compute, weight and activation loads, output write, and staging storage.

• Edges record which tiles supply each tile’s inputs. The original kernel barriers are kept as a separate edge set.

• Tile dependencies come from source-declared contracts, profiling kernels, grids and resource footprints. The planner checks these contracts against the observed profiles and calibrates tile service times to reproduce standalone execution. Operators without a tile decomposition remain opaque tasks occupying the modeled worker pool. Their reduced-budget costs use the CTA-wave approximation.

We schedule a tile graph with an event-driven list scheduler over a fixed pool of W workers where W is the CTA budget. A tile starts when its dependencies are complete and a worker with enough staging space is free. Memory transfers pass through a single FIFO memory queue and time advances to the next completion.

Two megakernel mechanisms act on this schedule:

• Removing the barrier edges lets ready tiles of the next kernel fill idle workers.

• Weight prefetch lets a computing worker stage the weights of an upcoming tile.

The result is a service table: the estimated time of one execution for every group for each backend and CTA budget.

Service costs depend on the implementation contract of the candidate region. We write them as $S [ g , \beta , W ; \kappa ]$ , indexed by operation group g, backend β, CTA budget W, tensor shapes, and precision, under an implementation context κ. The context captures thread counts, block geometry, call and storage policies, compiled-entry membership, and synchronization requirements. Each record stores a measured or tile-simulated duration together with its provenance. Timing evidence is reused only under its declared transfer conditions: a record is reused only when its key matches exactly, and any candidate without compatible evidence is excluded. Consequently, the same group, backend, and CTA budget can incur different costs when the enclosing implementation changes.

Each candidate uses a source-supported implementation configuration. The cost model selects compatible operator measurements for that configuration; it does not infer arbitrary fused-entry compiler behavior from conventional timings alone. Thus, enclosing-entry context changes the service cost before composition scoring.

## C. Event Graph (evGraph)

Each candidate is lowered to an event graph extending to first publication. Admission groups occur once, producer groups execute through step $h = d + c _ { 1 } + l ,$ and the first consumer chunk becomes eligible after step $r = d + c _ { 1 }$ . Conventional regions retain their constituent operator launches. Each megakernel launch holds its CTA allocation until all tasks assigned to that launch finish, including periods when those tasks wait for dependencies. Dynamic splitting ends the full-GPU producer launch at step r and continues with separate producer and consumer launches using budgets X and Y. Dispatch nodes represent the associated launches.

Node durations come from the service table. Edges encode execution dependencies and chunk readiness. Publication waits for both the first consumer chunk and producer step h; consequently, $T _ { \mathrm { p u b } } = \mathrm { m a x } ( F _ { \mathrm { c h u n k 1 } } , F _ { \mathrm { p r o d u c e r h } } )$ , where F denotes simulated completion time.

## D. Megakernel Impact Simulation

The tile scheduler produces implementation-specific service costs. Algorithm 1 uses these costs to simulate the event graph

under launch-allocation and shared-service constraints. Table I identifies the mechanisms modeled at each level.

TABLE I  
FEATURE MODELING AND ALGORITHMIC CHANGES.
<table><tr><td rowspan=1 colspan=1>Feature</td><td rowspan=1 colspan=1>Modelein</td><td rowspan=1 colspan=1>Algorithmic change</td></tr><tr><td rowspan=1 colspan=1>Cross-kernelreadiness</td><td rowspan=1 colspan=1>Tilegraph</td><td rowspan=1 colspan=1>Edges: drop the barrier edges betweenconsecutive kernels; keep the tile-to-tiletensor edges</td></tr><tr><td rowspan=1 colspan=1>Weight prefetch</td><td rowspan=1 colspan=1>Tilegraph</td><td rowspan=1 colspan=1>Scheduling rule: one staging slot perworker. A staged tile&#x27;s weight load isqueued while its predecessors still run;its compute still waits for them</td></tr><tr><td rowspan=1 colspan=1>On-chipforwarding</td><td rowspan=1 colspan=1>Tilegraph</td><td rowspan=1 colspan=1>Node weights: remove the forwarded ten-sor&#x27;s store and reload from the memoryqueue</td></tr><tr><td rowspan=1 colspan=1>Producer-consumeroverlap</td><td rowspan=1 colspan=1>Eventgraph</td><td rowspan=1 colspan=1>Edges, capacity: the first chunk dependson step r instead of the preceding task,and runs in its own CTA allocation</td></tr><tr><td rowspan=1 colspan=1>Fewer launches,internalsynchronization</td><td rowspan=1 colspan=1>Eventgraph</td><td rowspan=1 colspan=1>Nodes: a dispatch node before eachlaunch, weighted by the per-launch cost;synchronization adds to tile weights</td></tr><tr><td rowspan=1 colspan=1>Smaller CTAbudget</td><td rowspan=1 colspan=1>Servicetable</td><td rowspan=1 colspan=1>Tile scheduling uses W workers. For anopaque operator with a grid of g CTAs,compute time is approximated by scalingwith $[ g / W ] / [ g / \dot { 1 } 3 2 ]$ , assuming equal-cost CTA waves.</td></tr><tr><td rowspan=1 colspan=1>CTAs heldwhile waiting</td><td rowspan=1 colspan=1>Eventgraph</td><td rowspan=1 colspan=1>Edges, capacity: a launch&#x27;s release nodedepends on all its tasks; a launch isadmitted only when its CTAs are free</td></tr><tr><td rowspan=1 colspan=1>Registerpressure insidethe fused entry</td><td rowspan=1 colspan=1>Servicetable</td><td rowspan=1 colspan=1>Node weights: each task&#x27;s standalonetime at the region&#x27;s register limit, whichis the register file divided by the residentthreads per SM, at most 255</td></tr><tr><td rowspan=1 colspan=1>Shared threadcount &amp; tileshape</td><td rowspan=1 colspan=1>Servicetable</td><td rowspan=1 colspan=1>Node weights: pick one geometry perregion, the one minimizing its summedtask weights, and weight every task at it</td></tr><tr><td rowspan=1 colspan=1>Bandwidthcontention</td><td rowspan=1 colspan=1>Eventgraph</td><td rowspan=1 colspan=1>Rates: each running task progresses atmin(1, capacity ÷ total demand)</td></tr></table>

## E. Boundary Selection

The planner exhaustively scores the supported candidate set and returns all minimum-score candidates. Candidates require compatible service-cost evidence; deadlocked schedules are infeasible. Ties are retained rather than resolved using validation measurements.

Algorithm 2 describes the common scoring procedure; the model adapters use different representations of service costs. At NH-TTS batch 32, the planner selects $P / ( B + L ) / C$ : separate megakernels for prefill, the fused Nemotron-H backbone/local transformer and the codec with a dynamic producer/consumer split of 88/44 CTAs.

An illustrative calculation shows the tradeoff. Suppose chunk readiness occurs at 20 ms. Remaining producer and codec work take 4/6 ms serially and 6/8 ms under an 88/44 split. With serial / overlap transition costs of 0.1 / 0.2 ms and no bandwidth contention, publication scores are $2 0 + 4 + 6$ + 0.1 = 30.1ms and $2 0 + \operatorname* { m a x } ( 6 , 8 ) + 0 . 2 = 2 8 . 2 \mathrm { m s }$ . Overlap therefore wins despite slower individual components.

Algorithm 1 SIMULATE(G, T, L, C)   
Require:   
${ \overline { { G } } } = ( V , E ) \colon$ event graph   
$T [ v ] { \mathrm { : } }$ service time of node v when it runs alone.   
launch[v]: launch contract of node $v ,$ where for a launch L:   
cta[L]: CTAs that L holds;   
max tasks $[ L ] \colon$ tasks of L that can run at same time;   
release $[ L ] \colon$ control node that follows all tasks of L; its com  
pletion frees cta[L];   
C: shared memory bandwidth, where   
$B _ { \mathrm { m e m } } \colon$ memory bandwidth of the device;   
$N _ { \mathrm { { C T A } } } { \mathrm { : } }$ number of CTAs in the system;   
$Q [ v ] { \mathrm { : } }$ memory demand of node v.   
Ensure: finish[v]: completion time of each node $v ,$ or DEADLOCK.   
State:   
time: current simulated time; $T _ { \mathrm { r e m } } [ v ] \colon$ time node v still needs;   
rate[v]: relative speed of $v \ ( 1$ when standalone);   
READY: queue of unstarted nodes whose predecessors are   
all in DONE;   
RUNNING: nodes started but not finished;   
DONE: finished nodes;   
ADMITTED: launches currently holding CTAs;   
free cta: CTAs not held by an admitted launch.   
1: time $ 0 ; \quad T _ { \mathrm { r e m } } [ v ]  T [ v ]$ for every $v \in V$   
2: RUNNING $ \infty ;$ DONE ← ∅; ADMITTED $ \infty ;$   
free cta ← N<sub>CTA</sub>   
3: RE ${ \mathrm { A D Y } }  \{ v \in V :$ v has no predecessors }   
4: while DONE ̸= V do   
5: for all $v \in \mathrm { R E A D Y }$ do   
6: $L \gets \mathrm { l a u n c h } [ v ]$   
7: if $L \neq$ NONE and $L \not \in$ ADMITTED and $f r e e \_ c t a \geq$   
$\mathrm { c t a } [ L ]$ then   
8: ADMITTED ← ADMITTED ∪ {L}   
9: $f r e e \_ c t a \gets f r e e \_ c t a - \mathrm { c t a } [ L ]$   
10: end if   
11: if L = NONE or $( L \ \in \ \mathrm { \ A D M I T T E D }$ and $\mathrm { ~ \textit ~ { ~  ~ } ~ }$   
RUNNING : launch $[ u ] = L \} | <$ max tasks $[ L ] )$ then   
12: move v from READY to RUNNING   
13: end if   
14: end for   
15: if RUNNING = ∅ then   
16: return DEADLOCK   
17: end if   
18: $\begin{array} { r } { Q _ { \mathrm { t o t } }  \sum _ { v \in \mathrm { R U N N I N G } } Q [ v ] } \end{array}$   
19: for all $v \in$ RUNNING do   
20: if $Q [ v ] = 0$ then   
21: rate[v] ← 1   
22: else   
23: $\mathrm { r a t e } [ v ] \gets \operatorname* { m i n } ( 1 , B _ { \mathrm { m e m } } / Q _ { \mathrm { t o t } } )$   
24: end if   
25: end for   
26: ∆ ← min<sub>v∈RUNNING</sub> T<sub>rem</sub>[v]/rate[v]   
27: time $\gets t i m e + \Delta$   
28: for all $v \in \mathrm { R U N N I N G }$ do   
29: $T _ { \mathrm { r e m } } [ v ] \gets T _ { \mathrm { r e m } } [ v ] - \mathrm { r a t e } [ v ] \cdot \Delta$   
30: end for   
31: for all $v \in \mathrm { R U N N I N G }$ with $T _ { \mathrm { r e m } } [ v ] = 0$ do   
32: finish[v] ← time   
33: move v from RUNNING to DONE   
34: if v = release[L] for some launch L then   
35: ADMITTED ← ADMITTED \ {L}   
36: $f r e e \_ c t a \gets f r e e \_ c t a + \mathrm { c t a } [ L ]$   
37: end if   
38: for all successors w of v in G whose predecessors are   
all in DONE do   
39: append w to READY   
40: end for   
41: end for   
42: end while   
43: return finish

Algorithm 2 SELECTCOMPOSITION(R, b, E)   
Require:   
R: region graph; one node per group; edges for data, state, and   
publication dependencies.   
b: batch size.   
E: profiles (kernel durations, HBM traffic, launch configuration),   
service records (measured or tile-simulated durations), and imple  
mentation contracts (backend, launch configuration, CTA budget,   
tile geometry).   
Ensure: BEST: set of feasible candidates with minimum first   
publication score.   
State:   
CANDIDATES: supported candidates $\pi = ( \mathrm { c o v e r , p o l i c y } )$   
for R at batch size $b ;$   
Cover: split of the ordered groups into contiguous regions,   
each run by one backend;   
Policy: how regions share the GPU (serial, fixed split of   
$\mathrm { C T A s } ,$ or dynamic split);   
κ[π]: implementation context of π (thread counts, tile geome  
try, call and storage policies);   
$S [ \pi ] \colon$ service cost of every group of π for its backend, CTA   
budget, and context $\kappa [ \pi ] ;$   
$\check { T _ { \mathrm { p u b } } } [ \pi ] ;$ : first-publication time of π (when the first output can   
be returned).   
1: CANDIDATES ← every cover of $R$ paired with each applicable   
execution policy and supported contracts in E at batch size b   
2: for all $\pi \in { \mathrm { C A N D I D A } } { \mathrm { \dot { T } E S } }$ do   
3: $\kappa [ \pi ] $ implementation context of π, from the contracts in   
$E$   
4: S[π] ← service costs from the records in $E$ whose context   
matches $\kappa [ \pi ]$   
5: if some group of π has no matching record then   
6: $T _ { \mathrm { p u b } } \bar { [ } \pi ] \stackrel { . } {  } \infty ;$ continue   
7: end if   
8: $( G , t , L , C ) \gets$ event graph of π: its tasks, launches, dispatch   
and release nodes, with times t from $S [ \pi ] ,$ launch contracts   
L from its regions and policy, bandwidth demands $C ,$ and a   
publication node that waits for the first consumer chunk and   
producer step h   
9: finish $\gets \boldsymbol { \mathrm S } 1$ IMULATE(G, t, L, C)   
10: if finish = DEADLOCK then   
11: $T _ { \mathrm { p u b } } [ \pi ]  \infty$   
12: else   
13: $T _ { \mathrm { p u b } } [ \pi ] $ finish[publication]   
14: end if   
15: end for   
16: $T _ { \mathrm { b e s t } } $ min<sub>π∈CANDIDATES</sub> $T _ { \mathrm { p u b } } [ \pi ]$   
17: return $\begin{array} { r c l c r } { { \mathrm { B E S T } } } & { {  } } & { { \{ \pi \in { \begin{array} { c } { { \dot { \mathrm { C A N D I D A T E S } } } } \\ { { \mathrm { ~ } } } \end{array} } : } } & { { T _ { \mathrm { p u b } } [ \pi ] } } & { { = } } \end{array}$   
$T _ { \mathrm { b e s t } }$ and $T _ { \mathrm { b e s t } } < \infty \dot  \}$

## V. WHAT THE BOUNDARY MODEL REVEALS ABOUTCOMPOSITION

The tile and event graphs predict and explain multiple observations on performance of Megakernels (and kernels in general) in a model execution environment.

## A. Observation 1: A faster kernel can make the pipeline slower

In event graph, every occurrence is bound to a launch contract, which holds a CTA allocation from admission until its last task finishes. Co-running launches interact only through the joint simulation (Algorithm 1): they compete for CTAs and progress at the shared-service rate. The planner uses no measured co-run timings.

A slowdown with a faster kernel can happen due to changed contract that is more CTAs held or more memory traffic impacting the peer co-located kernels.

Based on experiments testing the observation, with 88 CTAs instead of 80, the producer megakernel in NH-TTS is 2.2% faster alone yet beside the conventional codec, the pair takes 30.3% longer because the codec loses eight CTAs; beside the codec megakernel, the pair gets 2.0% faster.

## B. Observation 2: Colocation cost depends on the peer implementation

A persistent launch retains its CTA allocation until completion. In Table II’s NH-TTS experiment, the conventional codec slows by 7.57× beside the producer megakernel, whereas the codec megakernel slows by 1.23×. The tested allocation therefore exhibits substantially different interference depending on the peer implementation.

TABLE II  
MEGAKERNEL VS TENSORRT LATENCIES
<table><tr><td rowspan=2 colspan=2></td><td rowspan=1 colspan=2>Stage 1ª</td></tr><tr><td rowspan=1 colspan=1>Megakernel</td><td rowspan=1 colspan=1>TensorRT</td></tr><tr><td rowspan=2 colspan=1>Stage 0ª</td><td rowspan=1 colspan=1>Megakernel</td><td rowspan=1 colspan=1>0.991ms</td><td rowspan=1 colspan=1>1.290ms</td></tr><tr><td rowspan=1 colspan=1>TensorRT</td><td rowspan=1 colspan=1>1.216ms</td><td rowspan=1 colspan=1>9.760ms</td></tr></table>

<sup>a</sup>Refer Figure 3

## C. Observation 3: Serial cuts can be neutral under fixed contracts

Excluding dispatch overhead and holding implementation and resource contracts fixed, inserting a serial boundary leaves the modeled device schedule unchanged. Under these assumptions, all eight all-megakernel partitions of NH-TTS predict 37.067 ms TTFA p95 on H100 NVL.

On H100 NVL GPU, the monolith and the two-megakernel design differ only by the prefill cut, with the same implementation and CTA split, and they measure within 0.5% of each other at every batch size, as predicted. The monolith is slower than the serial three-megakernel design for a different reason: it runs every phase with 512-thread blocks, while the threemegakernel designs run Stage 0 with 256-thread blocks. The specialization, not the boundary, accounts for the gap.

## D. Observation 4: A megakernel’s fixed cost decides when conventional execution wins

Megakernel has fixed costs such as launches, host protocol and internal synchronization and cache residency across operations.

A megakernel’s fixed cost does not shrink with the work it saves. Its lead comes from the overhead it removes and at large batches, conventional execution spreads that overhead over more work. Once the lead falls below the fixed costs, conventional execution publishes first.

On H100 NVL, Qwen3-TTS’s four-region megakernel lead falls from 15.9 ms at B16 to 0.65 ms at B64 while its fixed costs stay at 0.7 to 1.1 ms so conventional graph wins at B48 and B64.

With NH-TTS, the lead with Megakernels keeps on increasing with batch size so conventional execution never wins at any tested batch size.

## VI. EVALUATION ACROSS MODELS AND FRAMEWORKS

Using only execution of the conventional graph baseline and kernel benchmarks, our method picks the fastest megakernel composition.

## A. Setup

We evaluate the models and batch sizes shown in Table III. Within each model, designs share weights, operator precision, input fixtures, generation limits, deterministic decoding and publication rules. The artifact records checkpoint hashes, software versions, text/context lengths and $( d , c _ { 1 } , l )$ . Before timing, each design is checked against an unfused reference on separate correctness fixtures: generated token or code indices and publication prerequisites must match exactly, and floating-point outputs must have relative L2 error at most 10- 2. Failed designs are reported and excluded. Qualified designs are measured in balanced randomized order within matched GPU blocks.

TABLE III  
THE THREE MODELS IN OUR EVALUATION.
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Conventional baseline</td><td rowspan=1 colspan=1>First publication</td><td rowspan=1 colspan=1>Batch sizes</td></tr><tr><td rowspan=1 colspan=1>NH-TTS</td><td rowspan=1 colspan=1>TensorRT-LLM withCUDA graphs, plusTensorRT engines</td><td rowspan=1 colspan=1>Time to First Au-dio (TTFA) at en-gine publication</td><td rowspan=1 colspan=1>1,32, 64</td></tr><tr><td rowspan=1 colspan=1>Qwen3-TTS</td><td rowspan=1 colspan=1>PyTorch with vLLMFlashAttention-3,torch.compile andCUDA graphs</td><td rowspan=1 colspan=1>Time to First Au-dio (TTFA) onthe host</td><td rowspan=1 colspan=1>1, 16, 32,48,64</td></tr><tr><td rowspan=1 colspan=1>aLlama3.2 1B</td><td rowspan=1 colspan=1>PyTorch SDPA(FlashAttention) in oneCUDA graph</td><td rowspan=1 colspan=1>Time to First To-ken (TTFT)</td><td rowspan=1 colspan=1>1, 2, 16,32, 48</td></tr></table>

<sup>a</sup>496-token prompt

The pools contain the designs we could build and qualify, not every candidate: the planner ranks all 81 NH-TTS candidates, but only these are measured. Each design runs in its own process with exclusive use of one H100 NVL GPU. Each NVIDIA H100 NVL GPU has 94 GB of HBM3 memory with 3.9 TB/s peak bandwidth [11]; we use CUDA 13.2 and NVIDIA driver 595.84.

NH-TTS reports the mean of per-batch request p95 TTFA over 36 batches in the main campaign. Qwen3-TTS computes p95 over three cohort-level first-PCM publication times per run and averages 9 run-p95 values across 3 processes at B1/B48/B64, or twelve across four processes at B16/B32. Llama computes p95 over 30 repetitions per process and aggregates across three processes using the arithmetic mean. We discard two warmup cohorts per process for NH-TTS, Qwen3- TTS and Llama, before measurement. Ratio confidence intervals use 20,000 bootstrap resamples of four matched

GPU blocks for NH-TTS and Qwen3-TTS B16/B32, or nine matched process/run units for Qwen3-TTS B1/B48/B64, with paired resampling across designs. Collecting the planner’s inputs takes minutes: kernel benchmarks took 4.6 GPU-minutes for NH-TTS and 15.9 for Qwen3-TTS across four batch sizes.

We fixed the scoring rules and candidate-generation policy using a separate development set. Evaluation batch sizes and input fixtures were excluded from development. At each evaluation batch size, conventional and standalone component profiling supplied the permitted inputs. Candidate pools, service records, predicted scores and tie rules were frozen before candidate end-to-end timing. Validation results were not used to revise the model. Previously inspected cases were reported separately as development evidence.

## B. Selection accuracy on held-out cases

The frozen planner uniquely selected the measured fastest design in 13/13 held-out model-batch cases. This establishes selection accuracy within the declared validation pools across the tested workloads on H100 NVL. It does not establish optimality over unmeasured compositions or transfer to other GPU architectures.

The planner ranks designs; it does not calibrate absolute time. Conventional designs, scored from a measured run, land within 5% of validation. NH-TTS’s one- and two-megakernel designs share one persistent implementation and land within 5%; its three-megakernel designs share another and are overpredicted by 6 to 22% (42.52 ms predicted for the selected B32 design, 35.99 ms measured). Llama-1B designs are overpredicted by up to 11%, while Qwen3-TTS’s four-megakernel design is under-predicted by 1.1 to 1.5 ms at B48 and B64. Despite these absolute errors, the predicted winner matches the measured winner in each validation pool. The closest decisions hold within their 95% intervals: two megakernels are 8.1% slower than four at Qwen3-TTS B1 (7.8 to 8.3%), and four megakernels are 4.2% (3.1 to 4.9%) and 3.2% (3.0 to 3.4%) slower than conventional at B48 and B64.

Within NH-TTS’s three-megakernel layout, the method also ranks the resource policies correctly: giving the producer and codec 80 and 52 thread blocks is 1.0% slower than 88 and 44 at batch size 32 and running the three megakernels one after another is 4.2% slower than overlapping them at B64.

With identical service costs and candidate pools, the full planner matches 13/13 measured winners; additive scoring and removal of overlap each match 11/13, with maximum regret of 4.50%. Both misses occur at NH-TTS B32/B64, where the simpler selectors choose serial execution. The full planner instead accounts for chunk readiness, CTA allocations and concurrent completion. These results localize the benefit of event simulation to overlap-sensitive decisions; implementation-specific component costs explain selections where execution is serial.

For Llama-1B, the two-megakernel design runs prefill and decode as separate megakernels. It beats one combined megakernel even on first-token latency because the combined entry spills registers to local memory, which makes each prefill layer about 15% slower in the planner’s step benchmarks (316 vs 276 µs at B1).

![](images/d698a99a8db9b7d409fa48343dc53f1c45bba2f7442ddaee7dc463cd9b111203.jpg)  
Fig. 4. Predicted selections within the implemented validation pools. Predicted and measured margins compare the winner with the nearest alternative in the same pool.

Additive scoring sums event service and dispatch costs. The no-overlap ablation serializes otherwise concurrent work in a fixed topological order while preserving component costs and CTA budgets. Regret is selected measured latency divided by the measured minimum, minus one. Tied predictions do not count as unique-winner matches and receive their worstmember regret.

## VII. HOW FAR SHOULD KERNEL FUSION EXTEND?

Fusion can stop anywhere between two extremes: conventional graph execution and monolith megakernel.

Making a region persistent adds a cost F that is fixed per step. Against this, the region saves S(b) relative to the conventional kernels it replaces at batch size b. Launch and host dispatch that a CUDA graph has not already removed are independent of b. Intermediate traffic that no longer reaches memory grows with b. The service difference between one compiled schedule and individually tuned kernels can grow, shrink or change sign as tiles fill and the number of waves changes. A region pays when $S ( b ) > F .$ . As F is fixed while S(b) can rise, fall or stay flat, the best extent of fusion moves with batch size and in different directions for different models.

This results in two observed categories of models, and two are predicted. For the two unobserved categories:

• Megakernel-advancing: We expect it where conventional cost is dominated by work-proportional traffic or by serialized compute, memory and communication that a persistent schedule can overlap.

• Megakernel-averse: We expect it for regions already compute-bound such as large-model prefill component.

A region that gains little from a megakernel in isolation may nevertheless benefit when colocated, as the Nemotron experiment illustrates.

The answer to “how far” therefore ranges from multiple megakernels to conventional execution with thousands of kernels per phase. It depends on the model architecture, the batch size, and the objective. The boundaries worth preserving are those whose crossing costs less than the specialization and publication control they protect.

TABLE IV  
FOUR MODEL CATEGORIES BY WHERE MEGAKERNELS PAY
<table><tr><td rowspan=2 colspan=2>Batch</td><td rowspan=1 colspan=2>Large</td></tr><tr><td rowspan=1 colspan=1>Pays</td><td rowspan=1 colspan=1>Doesn&#x27;t</td></tr><tr><td rowspan=2 colspan=1>Small</td><td rowspan=1 colspan=1>Pays</td><td rowspan=1 colspan=1>Megakernel-stable (NH-TTS; Llama-1B for first-token latency)</td><td rowspan=1 colspan=1>Megakernel-receding(Qwen3-TTS; Llama-1Bfor inter-token latency)</td></tr><tr><td rowspan=1 colspan=1>Doesn&#x27;t</td><td rowspan=1 colspan=1>Megakernel-advancinga</td><td rowspan=1 colspan=1>Megakernel-aversea</td></tr></table>

<sup>a</sup>Predicted, not observed.

A model is megakernel-stable when its stable kernels:

• Account for most of the time on the path to the objective. For first-token latency that path is only prefill, which is why Llama-1B is stable for first-token latency but receding for decode.

• Form contiguous regions whose savings exceed their fixed costs at every batch size. Stable kernels interleaved one by one with others can’t pay for the cuts around them.

• Do not lose their gains to co-located conventional parts, as in Observation 2.

## VIII. RELATED WORK

Persistent kernels are a long-standing GPU programming style [12]. Hand-built megakernels for Llama-1B decoding and tensor-parallel Llama-70B [9], [8], compilers such as MPK [1] and Event Tensor [13] and systems such as Ada-MK [14], TileRT [15] and FlashMoE [16] build fast megakernels for the regions they are given; we decide which regions those should be.

Rammer [17] selects persistent-program boundaries, while VersaPipe [18] tunes pipeline stage groups, execution models and resource assignments. Our focus is first-publicationaware composition using conventional and operator-level evidence. GPU performance forecasting models [19] and interference-aware sharing systems [20] provide complementary approaches; we combine service estimates with persistent resource lifetimes and streaming publication dependencies.

## IX. CONCLUSION

Fusing a streaming model into megakernels is a question of where to cut, not whether to fuse. Each cut decides which phases share a launch configuration, how long thread blocks stay held, how the phases share the GPU and which fixed costs are paid and the best answer ranges from zero to four megakernels across models and batch sizes. We showed that this answer can be predicted without timing candidate composition end to end, from the conventional baseline and standalone benchmarks of the operators involved. It selected the measured best design in all 13 cases across three models and three frameworks. The design it selects for NH-TTS achieves a 4.05× speedup in engine-level mean batch-p95 TTFA at batch 32 and the four rules derived from its model explain when fusion helps and when it does not.

These results open two directions. First, are megakernelstable architectures universally superior to other models with comparable accuracy in terms of compute performance? Second, can a software-level scheduling prevent the interference between a co-located conventional graph and megakernel in GPU?

## REFERENCES

[1] X. Cheng et al., “MPK: A compiler and runtime for mega-kernelizing tensor programs,” in Proc. USENIX Symposium on Operating Systems Design and Implementation (OSDI), 2026.

[2] W. Kwon et al., “Efficient memory management for large language model serving with PagedAttention,” in Proc. ACM Symposium on Operating Systems Principles (SOSP), 2023.

[3] L. Zheng et al., “SGLang: Efficient execution of structured language model programs,” in Proc. Advances in Neural Information Processing Systems (NeurIPS), 2024.

[4] NVIDIA, “Nemotron-H: A family of accurate and efficient hybrid Mamba-Transformer models,” NVIDIA ADLR Research Blog, 2025, accessed: Oct. 19, 2026. [Online]. Available: https://research.nvidia. com/labs/adlr/nemotronh/

[5] ——, “TensorRT,” 2026. [Online]. Available: https://developer.nvidia. com/tensorrt

[6] H. Hu et al., “Qwen3-TTS technical report,” 2026, arXiv:2601.15621.

[7] J. Ansel et al., “PyTorch 2: Faster machine learning through dynamic Python bytecode transformation and graph compilation,” in Proc. ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS), 2024.

[8] B. Spector et al., “Look ma, no bubbles! designing a low-latency megakernel for Llama-1B,” Hazy Research blog, May 2025.

[9] ——, “We bought the whole GPU, so we’re damn well going to use the whole GPU,” Hazy Research blog, Sep. 2025.

[10] Y. Shi et al., “Welder: Scheduling deep learning memory access via tile-graph,” in Proc. USENIX Symposium on Operating Systems Design and Implementation (OSDI), 2023.

[11] NVIDIA, “NVIDIA H100 NVL GPU,” Product Brief PB-11773- 001 v01, Mar. 2024. [Online]. Available: https://www.nvidia.com/ content/dam/en-zz/Solutions/Data-Center/h100/PB-11773-001 v01.pdf

[12] K. Gupta, J. A. Stuart, and J. D. Owens, “A study of persistent threads style GPU programming for GPGPU workloads,” in Proc. Innovative Parallel Computing (InPar), 2012.

[13] H. Jin et al., “Event tensor: A unified abstraction for compiling dynamic megakernel,” in Proc. Conference on Machine Learning and Systems (MLSys), 2026.

[14] W. Dong et al., “Ada-MK: Adaptive MegaKernel optimization via automated DAG-based search for LLM inference,” 2026, arXiv:2605.11581.

[15] “TileRT: Tile-based runtime for ultra-low-latency LLM inference,” GitHub repository, 2026. [Online]. Available: https://github.com/tile-ai/ TileRT

[16] O. J. Aimuyo, B. Oh, and R. Singh, “FlashMoE: Fast distributed MoE in a single kernel,” in Proc. Advances in Neural Information Processing Systems (NeurIPS), 2025.

[17] L. Ma et al., “Rammer: Enabling holistic deep learning compiler optimizations with rTasks,” in Proc. USENIX Symposium on Operating Systems Design and Implementation (OSDI), 2020.

[18] Z. Zheng et al., “VersaPipe: A versatile programming framework for pipelined computing on GPU,” in Proc. IEEE/ACM International Symposium on Microarchitecture (MICRO), 2017.

[19] S. Lee, A. Phanishayee, and D. Mahajan, “Forecasting GPU performance for deep learning training and inference,” in Proc. ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS), 2025.

[20] F. Strati, X. Ma, and A. Klimovic, “Orion: Interference-aware, finegrained GPU sharing for ML applications,” in Proc. European Conference on Computer Systems (EuroSys), 2024.