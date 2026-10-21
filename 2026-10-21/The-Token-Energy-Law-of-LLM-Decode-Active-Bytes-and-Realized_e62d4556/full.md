# The Token-Energy Law of LLM Decode: Active Bytes and Realized Time Account for Joules per Token

CHENXU NIU, NVIDIA Corporation, USA

BATUHAN SENCER, Texas Tech University, USA

WEI ZHANG, Texas Advanced Computing Center, USA

YONG CHEN, Texas Tech University, USA

Energy-eficient deployment of large language models requires understanding how model architecture, hard ware, and serving settings shape energy per generated token. We introduce the Token-Energy Law, a compact, interpretable model of GPU-board energy per output token for dense, full-history decoding. It relates token energy to two core dimensions: active bytes and realized decode time. Active bytes describe logical byte demand per output, separating dense weights shared within a decode iteration from private key–value (KV) histories read and updated for each request. Realized decode time is the elapsed decode duration divided by the number of output tokens produced. We combine the two byte coordinates and time in a common afine form, with coeficients calibrated for each model and hardware/runtime configuration. This formulation explains how batching amortizes shared-weight work, longer attended histories increase private KV demand, and throughput determines the time-associated cost per output. Our experiments cover three models under BF16 decoding on NVIDIA Hopper and Blackwell GPUs, including two-GPU tensor parallelism on Hopper. Using fixed coeficients and observed decode time, held-out condition-mean energy estimates difer from measurements by at most 3.73%. Controlled request-departure tests further show that previously identified model-specific coeficients remain efective as batch membership changes, without recalibration for departure patterns. The central contribution is a reusable work–time relationship whose structure remains consistent across configurations while its coeficients describe local energy responses. It provides researchers and practi tioners with a quantitative basis for interpreting inference energy and comparing serving trade-ofs. Combined with independently validated performance models, the law can also support prospective energy estimation and energy-eficient deployment planning.

Additional Key Words and Phrases: large language model inference, energy measurement, scaling law, decode inference, KV cache, data movement

## 1 Introduction

Large language models (LLMs) are increasingly deployed in applications ranging from information retrieval to code generation and conversational systems. The scale of this activity is growing rapidly. Google reports processing over 3.2 quadrillion tokens per month across its products in May 2026 [32], compared with approximately 480 trillion a year earlier. At suficient serving volume, cumulative inference energy can rival the energy invested in training; Luccioni et al. place this crossover at hundreds of millions of requests for the BLOOMz models they study [25]. Understanding the energy cost of inference is therefore an increasingly important problem in the design and operation of LLM services. This raises a fundamental systems question: Can a common law explain how energy per generated token varies across models, hardware platforms, and deployment configurations?

Research on inference energy spans benchmarking, workload-based estimation, and production measurement. TokenPowerBench [28] and ML.ENERGY [5] characterize energy across models and serving configurations. Fernandez et al. examine the efects of workload geometry and execution configuration and develop measurement-based energy estimates for ofline workloads [12]. Google’s production measurements account for the broader system costs of delivering AI services [10]. These studies provide an empirical foundation for understanding inference energy across models and serving configurations. Building on this foundation, we seek a common analytical law that relates energy per generated token to the underlying execution characteristics, providing a systematic explanation of how model and deployment choices afect energy consumption.

The dificulty of such an efort is that inference energy consumption depends on several interacting dimensions - (1) model architecture and numerical representation determine the weights and KV state used during generation; (2) hardware and kernels determine how that work is executed; (3) optimization strategies applied to deployment - such as batching, tensor parallelism, and pipeline parallelism - may afect work sharing, resource occupancy, communication, and execution time. Context length changes the history processed by attention. Nominal configuration parameters, however, do not always reflect the work performed during execution: quantization changes stored bytes without changing parameter count [23]; iteration-level scheduling allows the active batch size to difer from the configured maximum [45]; and allocated KV-cache capacity does not directly indicate the amount of history processed by attention [19]. A useful energy model must identify execution quantities that capture these efects while retaining a consistent interpretation across configurations.

Recent analytical approaches already make progress toward this goal. SweetSpot [33] and From Tokens to Watt-hours [40] relate inference energy to workload complexity and memory access; two frameworks named EnergyLens [30, 39] develop closed-form and operator-level models for deployment choices. Request-level analyses also show why average token energy depends on the work included in the measurement window [41]. We address a specific question within this broader efort: can the sharing structure of decode supply a small set of interpretable work coordinates that, together with execution time, explain energy per useful output token across models and systems? Our approach connects a trace-level accounting identity, an independently tested relationship to physical memory trafic, and a compact energy response.

Our starting point is exact decoding with dense weights and full attention history. Within an iteration, all useful outputs share the dense weight operators, while each output attends to its own KV history. To capture this sharing structure, we define active bytes as the logical byte demand of shared dense-weight and private KV-cache accesses per useful output token. Let $M _ { w }$ denote the shared dense-weight footprint in bytes, $B _ { \mathrm { e f f } }$ the realized mean number of useful outputs per iteration, $K _ { \mathrm { K V } }$ the combined key and value bytes per cached token (summed over layers), and $\bar { L } _ { \mathrm { t o k e n } }$ the mean attended history length, weighted by useful tokens. The logical read demand per useful output from the weight and KV components is then $A _ { \mathrm { t o k e n } } = M _ { \mathrm { w } } / B _ { \mathrm { e f f } } + K _ { \mathrm { K V } } \bar { L } _ { \mathrm { t o k e n } }$ . The first term is shared work amortized by batch; the second is private work growing with context. Batch and context therefore afect per-token work in fundamentally diferent ways. Accounting for the write of one new KV entry per output gives the read–write demand $A _ { \mathrm { t o k e n } } ^ { \mathrm { r w } } = M _ { \mathrm { w } } / B _ { \mathrm { e f f } } + K _ { \mathrm { K V } } ( \bar { L } _ { \mathrm { t o k e n } } + 1 )$ These coordinates describe the trace’s dense-weight and KV demand, not all resident memory or all operator accesses. Indexed embedding and positional lookups lie outside this principal account and require separate access accounting (Section 4); physical HBM measurements characterize how the principal coordinates are realized by an implementation.

Energy depends on both active bytes and realized decode time. We express the weight and KV components in decimal GB as $q _ { \mathrm { w } } = M _ { \mathrm { w } } / ( 1 0 ^ { 9 } B _ { \mathrm { e f f } } )$ and $q _ { \mathrm { K V } } = K _ { \mathrm { K V } } \big ( \bar { L } _ { \mathrm { t o k e n } } + 1 \big ) \big / 1 0 ^ { 9 }$ , where the latter includes one new KV-record write per output. For decode duration � and $T$ useful outputs, the realized time per token is $\tau = D / T$ . The Token-Energy Law is

$$
\begin{array} { r } { \left| e _ { \mathrm { g r o s s } , \sigma } = c _ { \sigma } + \alpha _ { \sigma } q _ { \mathrm { w } } + \beta _ { \sigma } q _ { \mathrm { K V } } + P _ { \tau , \sigma } \tau + u _ { \sigma } . \right| } \end{array}\tag{1}
$$

Here $e _ { \mathrm { g r o s s } , \sigma }$ is GPU-board energy in joules per useful output token. The coeficients $\alpha _ { \sigma } , \beta _ { \sigma } ,$ and $P _ { \tau , \sigma }$ describe efective energy responses to weight demand, KV demand, and time; $c _ { \sigma }$ is an approximately constant per-token overhead within execution setting �, and $u _ { \sigma }$ is the residual.

Generality lies in this reusable work–time structure, rather than universal energy-per-byte constants. We identify coeficients for each specified model and hardware–runtime setting � and test the empirical relationship on held-out workload configurations without refitting. The model provides an interpretable basis for comparing deployment choices under a common energy boundary. Our evaluation uses observed decode duration; prospective deployment estimates can combine the law with a separately validated duration model.

## Contributions.

Trace-based active-byte accounting. We derive active-byte coordinates from iterationlevel weight sharing and token-weighted KV history. Their accounting accommodates changing request membership, and independent HBM measurements assess their correspondence with physical reads.

A unified work–time energy model. We formulate a locally calibrated energy response over shared-weight demand, private-KV demand, and realized decode time. Frozen-coeficient evaluations test whether this response remains predictive across workload changes, including heterogeneous request departures.

Validation across models and execution settings. Across three models on NVIDIA Hopper and Blackwell GPUs, our evaluation covers single-node tensor parallelism, same-SKU device transfer, and controlled request departures. All evaluated condition means from 374 held-out energy executions are within 3.73% of measurement. Separately, 177 HBM profiles support the physical relevance of the byte coordinates, while matched GB300 ablations assess the contributions of realized time and separate weight/KV terms.

## 2 Background and Related Work

Understanding inference energy requires connecting execution behavior to measured energy. We review the relevant serving mechanisms, measurement principles, and prior work that motivate our model.

## 2.1 LLM Inference as a Serving Workload

Prefill and autoregressive decode. During prefill, the model processes the prompt and constructs the key and value (KV) cache. During conventional exact decode, each active request generates one new token per iteration, using cached KV state to attend to its history. Prefill therefore exposes parallelism across input positions, whereas decode relies more heavily on concurrency across requests. Sarathi-Serve [1] exploits this distinction by co-scheduling chunked prefills with ongoing decodes, whereas DistServe [49] and Splitwise [31] instead separate prefill and decode execution

Batching and dynamic membership. Batching combines requests into a single execution that shares model weights across requests. It does not make their attention histories identical: each request still has its own sequence and attention work. Orca schedules at iteration granularity, allowing completed requests to leave and new requests to enter [45]. Consequently, a configured concurrency limit need not equal the number of useful outputs actually produced in each iteration. Similarly, context length is an evolving property of an active request, not just a maximum supported by the model. These distinctions matter when attributing work and energy to generated tokens rather than to a batch reservation.

Model representation and memory management. Parameter count alone does not determine the bytes accessed during inference. Weight precision changes storage and kernel behavior [23, 43]; multi-query and grouped-query attention reduce the number of KV heads [2, 37]. KV quantization shrinks the per-token record, while history eviction alters the set of attended tokens [24, 47]. PagedAttention [19] reduces allocation waste and enables KV sharing, and SGLang [48] exploits reusable prefixes. Attention kernels can further reduce or reorganize transfers without changing the intended operator [7, 8, 44]. Thus, allocated memory, logically attended state, and physical memory trafic are distinct quantities.

Hardware and parallel deployment. Accelerators difer in compute capability, memory bandwidth, on-chip storage, and execution eficiency. Tensor parallelism partitions work within layers and introduces collective communication; pipeline parallelism assigns layer groups to stages and introduces inter-stage dependencies and scheduling bubbles. Their costs depend on workload and mapping, rather than scaling directly with GPU count. LLMCompass and GenZ model such hardware and execution interactions [3, 46]. An energy comparison must therefore specify both the deployment and the resources included in its measurement. For multi-GPU inference, energy comparisons should account for all participating GPUs.

## 2.2 From Power Measurements to Token Energy

Power is an instantaneous rate; energy accumulates over an execution interval. For measured power $P _ { \mathcal { E } } ( t )$ at boundary $\varepsilon ,$ an interval of duration � and � useful outputs has energy and average output-token energy:

$$
E _ { \mathcal { E } } = \int _ { 0 } ^ { D } P _ { \mathcal { E } } ( t ) d t , \qquad e _ { \mathcal { E } } = E _ { \mathcal { E } } / T .\tag{2}
$$

A power trace, total request energy, and joules per output token answer diferent questions. Lower average power can coincide with higher energy if execution takes longer. Longer generation can increase request energy while reducing its average cost per token by amortizing request-level work, as Vellaisamy et al. demonstrate [41].

The boundary also changes the interpretation. GPU-board energy, the sum over all GPUs in a replica, server energy, and datacenter energy include diferent components. TokenPowerBench [28] separates inference phases and system-component contributions. Google’s production study accounts for accelerators, host resources, provisioned idle capacity, and datacenter overhead [10]. Lifecycle and carbon models introduce further quantities, including embodied emissions and the carbon intensity of electricity [11, 13]. These are complementary accounting levels, not interchangeable energy measurements.

Token normalization additionally requires an explicit execution phase. Including prefill in the numerator charges prompt processing to the generated outputs; measuring decode alone isolates the recurring generation process. Speculative decoding performs draft and verification work in addition to accepted output generation [22], while ofloaded inference accesses resources beyond the GPU [38]. A common unit therefore requires consistent treatment of useful output, extra work, and participating resources. Our target is GPU-board energy of the declared decode execution, summed across participating GPUs when needed.

## 2.3 What Measurement Studies Have Established

Cross-configuration characterization. From Words to Watts benchmarks LLaMA models ofdiferent sizes on V100 and A100 GPUs, using Alpaca and GSM8K workloads and model sharding across multiple GPUs and nodes [36]. Its measurements relate runtime and energy to model size, batch size, output length, and hardware allocation. The study establishes that inference eficiency is a property of the complete configuration, not simply the parameter count of the model. Its central contribution is experimental characterization: it supplies measured operating trends rather than a shared-weight/private-KV token-energy equation derived from the execution trace.

Table 1. Comparison of inference-energy measurement and modeling approaches.
<table><tr><td>Perspective and examples</td><td>Main question</td><td>Connection to this paper</td></tr><tr><td>Configuration measurement [5, 28, 36]</td><td>Effects of model, workload, and serving choices on energy.</td><td>Execution variables that explain configuration-dependent trends.</td></tr><tr><td>Analytical work-energy models [4, 33, 40]</td><td>Energy costs of computation, data movement, and time.</td><td>Work-time foundation for shared-weight and KV accounting.</td></tr><tr><td>Deployment and operator predictors [30, 39]</td><td>Energy prediction from configurations or operators.</td><td>Complements modeling based on realized trace quantities.</td></tr><tr><td>Request-window amortization [41]</td><td>Fixed and incremental energy within a request.</td><td>Explicit phase boundaries and useful-output normalization.</td></tr><tr><td>This work (Token-Energy Law)</td><td>Decode energy from shared weights, private KV history, and realized time.</td><td>Trace-derived work coordinates, independent HBM measurements, and frozen-coefficient validation under static and changing request membership.</td></tr></table>

Phase- and component-aware benchmarking. Niu et al. [29] compare inference engines by separating setup and token-generation energy and measuring GPU, CPU, and DRAM contributions. TokenPowerBench [28] extends this measurement perspective by examining prefill and decode, model and precision choices, input/output lengths, batch size, and tensor- and pipeline-parallel configurations. It distinguishes GPU and other system-component contributions and connects observed behavior to utilization, memory access, and parallel execution. These decompositions motivate explanatory variables that relate deployment choices to energy while retaining phase and component boundaries.

Serving eficiency and energy diagnosis. ML.ENERGY [5] combines service-aware accounting with latency-constrained configuration selection, and relates energy eficiency to KV-cache memory, batch size, and low compute intensity of decoding. Subsequent work further diagnoses inference energy consumption [6]. Fernandez et al. [12] examine how model, workload, and execution choices afect inference energy and develop a measurement-based estimation method for binned ofline workloads. They show that FLOP-based estimates assuming ideal hardware utilization can substantially underestimate energy consumption in the workloads they study. These studies thus include estimation and optimization as well as measurement. Luccioni et al. [25] compare energy across tasks and models; multi-request workflow studies extend the performance–energy question beyond isolated requests [16]. Configuration sweeps expose coupled efects: batch size can change weight reuse, utilization, and duration, while context can change KV access, arithmetic work, and kernel choice. Explaining these trends requires identifying the work shared across outputs and the private work performed for each output.

## 2.4 Analytical Models and the Level of Explanation

Work, data movement, and time. Roofline relates performance to operational intensity and hardware limits [42]; Energy Roofline combines computation and data-movement costs with energy from constant power accumulated over execution time [4]. Hardware energy analyses and I/O lower bounds further connect representation, access location, and data movement to execution cost [9, 14, 15, 35]. These foundations motivate work-based coordinates without equating logical demand with HBM trafic or assuming a universal energy cost per byte.

LLM-specific analytical and fitted predictors. SweetSpot [33] derives eficiency relationships over input and output lengths from transformer compute and memory complexity. From Tokens to Watthours separates computation, parameter access, KV writes, and attention reads using hardware coeficients and calibrated reuse factors [40]. Palladino et al.’s EnergyLens [30] uses symbolic regression to model energy across sequence lengths, batch sizes, and parallel configurations, with parameters fitted for diferent models and platforms.

Operator, hardware, and request models. Song et al.’s EnergyLens [39] combines operator and communication models, including execution overlap, for multi-GPU inference. GPUWattch, AccelWattch, and EnergAIzer model finer-grained GPU power [18, 20, 21], while Vellaisamy et al. distinguish fixed request-window energy from incremental generation-step energy [41]. GenZ models distributed inference performance and estimates operator energy from execution time, static power, and utilization-dependent compute, memory, and network power [3]. Table 1 summarizes these complementary perspectives. Our question is whether iteration-level sharing and outputweighted history provide suficient work coordinates for a compact energy response that remains predictive when request membership changes. We assess these coordinates independently through HBM measurements and test previously identified energy coeficients on controlled departure traces.

## 2.5 Motivation

These studies motivate a decode-energy model organized around active bytes and realized time. The work coordinates should account for weights shared within an iteration and KV history processed by each output, including changes in request membership. Independent trafic measurements and held-out energy evaluations then test whether these quantities provide a useful basis for explaining energy across workloads.

## 3 The Law at a Glance

The Token-Energy Law combines active bytes and realized time to describe energy per useful output.

Principal dense-weight and KV work $A _ { \mathrm { t o k e n } } ^ { \mathrm { r w } } = \frac { M _ { \mathrm { w } } } { B _ { \mathrm { e f f } } } + K _ { \mathrm { K V } } ( \bar { L } _ { \mathrm { t o k e n } } + 1 )$

<table><tr><td>Shared dense weights</td><td>Private KV</td><td>Realized time</td></tr><tr><td> $q _ { \mathrm { w } } = \frac { M _ { \mathrm { w } } } { 1 0 ^ { 9 } B _ { \mathrm { e f f } } }$ </td><td> $q _ { \mathrm { K V } } = \frac { K _ { \mathrm { K V } } ( \bar { L } _ { \mathrm { t o k e n } } + 1 ) } { 1 0 ^ { 9 } }$ </td><td> $\tau = { \frac { D } { T } }$ </td></tr><tr><td></td><td></td><td></td></tr><tr><td>decimal GB / output token</td><td>decimal GB / output token</td><td>seconds / output token</td></tr></table>

The Token-Energy Law

$$
\widehat { e } _ { \mathrm { g r o s s } , \sigma } = c _ { \sigma } + \alpha _ { \sigma } q _ { \mathrm { w } } + \beta _ { \sigma } q _ { \mathrm { K V } } + P _ { \tau , \sigma } \tau
$$

Same form; stratum-specific coeficients. � fixes the model, representation, hardware, runtime, topology, power policy, and energy boundary. The trace supplies the coordinates; local calibration supplies their energy response.

Interpreting the coeficients. The active-byte coeficients $\alpha _ { \sigma }$ and $\beta _ { \sigma }$ express efective J/decimal-GB responses for shared-weight and private-KV demand. Their values can difer because these accesses have diferent reuse and access patterns. $P _ { \tau , \sigma }$ , in watts, describes time-associated energy conditional on the byte coordinates, while $c _ { \sigma }$ represents an approximately workload-invariant per-token term within �. These are fitted responses, not separately metered energy components; $u _ { \sigma }$ is the remaining discrepancy.

Beyond $E = \bar { P } D$ . The identity $E = \bar { P } D$ describes each run using its own average power. The law instead uses one locally calibrated coeficient set to estimate token energy across workload configurations from their active bytes and realized time. On the evaluated grid, retaining the weight and KV coordinates improves held-out agreement over a time-only model (Table 5).

Testable consequences. Within a fixed stratum, positive coeficients imply three conditional relationships. At fixed context and $\tau ,$ the shared-weight contribution decreases as $1 / B _ { \mathrm { e f f } }$ . At fixed batch and $\tau ,$ the KV contribution grows linearly with attended history. At fixed active-byte coordinates, predicted energy increases linearly with �.

## 4 Formulating the Token-Energy Law

We formulate the Token-Energy Law using active bytes and realized decode time per useful output. We define the measurement boundary and useful-output count, then derive the shared-weight and private-KV coordinates and explain their roles alongside time in the energy response.

## 4.1 Defining the Token and the Energy Boundary

Our unit is one output token produced by dependency-ordered exact decode. The metered interval starts after an unmetered bootstrap makes the requests decode-ready and ends after the last counted output and device synchronization. Model loading and prefill are outside this interval. Let $T > 0$ denote the number of output tokens generated within the metered decode interval and � its duration in seconds. Each output token is counted once across the serving replica. Non-useful work consumes measured energy but does not increase the output count.

The energy boundary must match the serving replica whose work is counted. For a single-GPU replica, we use GPU-board energy. For a replica distributed across multiple GPUs, we sum board energies over the same synchronized interval and count each useful output once, not once per rank. With $\mathcal { G }$ denoting the replica’s boards and $E _ { g }$ the energy of board $g ,$ , the energy outcome and time coordinate are

$$
e _ { \mathrm { g r o s s } } = \frac { \sum _ { g \in { \cal G } } E _ { g } } { T } \quad [ \mathrm { J / t o k e n } ] , \qquad \tau = \frac { D } { T } \quad [ \mathrm { s / t o k e n } ] .\tag{3}
$$

We report gross GPU-board energy without subtracting an idle baseline. The measurement covers the participating GPU boards, not the entire server or facility.

Figure 1 illustrates the shared measurement interval and replica-wide output count for two participating GPUs.

Realized batch and attended history. We characterize decode work using actual batch membership and attended history. For a trace with � decode iterations, let $A _ { s }$ denote the queries producing useful outputs in iteration �, and $L _ { i , s }$ the number of historical records attended by query � in that iteration. With $\begin{array} { r } { T = \sum _ { s } \left| A _ { s } \right| } \end{array}$ , we define

$$
B _ { \mathrm { e f f } } = \frac { T } { S } , \qquad \bar { L } _ { \mathrm { t o k e n } } = \frac { 1 } { T } \sum _ { s = 1 } ^ { S } \sum _ { i \in A _ { s } } L _ { i , s } .\tag{4}
$$

![](images/86d43c85a6735df6bf51d459d426655fd1891d7bb558baaea30ae2d383ef1239.jpg)  
Fig. 1. A common decode measurement interval for a two-GPU replica. Shaded areas under schematic power traces represent $E _ { 0 }$ and $E _ { 1 }$ over duration �; � counts useful outputs once for the replica.

Here $B _ { \mathrm { e f f } }$ is the mean useful-output count per iteration, and $\bar { L } _ { \mathrm { t o k e n } }$ is the mean attended history per useful output. Both describe executed work rather than configured concurrency or reserved KV capacity.

For � synchronized requests, each attending � historical records at the first metered step and producing � outputs, we obtain $T = B O , B _ { \mathrm { e f f } } = B$ , and $\bar { L } _ { \mathrm { t o k e n } } = I + ( O - 1 ) / 2$ , since each history grows by one record per iteration. When requests finish at diferent iterations, Equation 4 accounts for their changing membership and histories. Section 5.4 evaluates these coordinates in controlled request-departure experiments.

## 4.2 Weight Work is Shared Across a Batch

The weight component of active bytes follows from the sharing structure of dense decode. Within an iteration, the useful outputs share the same dense-weight operations. Let $M _ { \mathrm { w } }$ denote the distinct weight bytes required by these operations, assumed constant across the counted iterations. Over � iterations producing � useful outputs, the logical weight demand per output, expressed in decimal GB, is

$$
q _ { \mathrm { w } } = \frac { S M _ { \mathrm { w } } } { 1 0 ^ { 9 } T } = \frac { M _ { \mathrm { w } } } { 1 0 ^ { 9 } B _ { \mathrm { e f f } } } .\tag{5}
$$

Doubling realized batch therefore halves the weight component of active bytes per output. It does not necessarily halve token energy, because private KV work and execution time need not decrease proportionally. The count describes logical demand, not the number of times a kernel physically loads each byte from HBM.

$M _ { \mathrm { w } }$ includes dense weights and associated parameter state consumed by the declared operations, rather than all resident model storage. Full input-embedding tables accessed through row lookups and positional lookup bufers are excluded; their accessed entries constitute auxiliary work. A tied embedding/output matrix is counted once when consumed by the dense output projection. For rank-resolved trafic accounting, aliased required storage is counted once within each rank, while distinct shards and physical replicas are summed across ranks. The TP=2 energy evaluation uses full-model reference coordinates rather than an audited rank-local storage sum (Section 5.5.2).

Figure 2(a,b) illustrates the distinction: increasing batch shares the same weight footprint across more outputs, while each output retains its own KV history. Panel (c) plots the principal logical read–write demand $M _ { \mathrm { w } } / B + K _ { \mathrm { K V } } ( L + 1 )$ for fixed history length �. Here �<sub>KV</sub> is the size of a complete KV record, and the 1 accounts for one new-record write per useful output. As batch grows, weight

Solid: total; dashed: KV; green: weights. Circles: equal weight/KV contributions; K = 1024.

demand falls and the combined demand approaches the KV term. Longer histories reach equal weight and KV demand at a smaller batch, given by $B ^ { \star } = { M _ { \mathrm { w } } } / [ { K _ { \mathrm { K V } } ( L + 1 ) } ]$ . This is a balance of logical bytes, not an energy crossover. The next subsection derives the KV coordinate for a complete decode trace.

![](images/e6d30e564527c9f6e762b4630dbfa41cabdffc9a80cafa6f4f20d1e35bae2d2f.jpg)

![](images/355e04d5316a6501484e601a9884958dfa0295b4047b12840e55065e7b986cb5.jpg)  
Fig. 2. Batch amortization of active bytes. (a,b) Weight demand is shared across outputs; KV histories remain private. (c) Analytical weight-plus-KV demand for Qwen2.5-7B (BF16). Circles mark equal weight and KV byte contributions.

## 4.3 KV Work Grows with Private History

Unlike shared dense weights, each request processes its own private, materialized KV history. Let $K _ { \mathrm { K V } }$ denote the constant byte size of one complete KV record across the declared serving replica. An output attending � historical records incurs $K _ { \mathrm { K V } } L$ logical read bytes and is aligned with one new-record write. Figure 3 illustrates this accounting over three consecutive decode steps.

Using the output-weighted mean history $\bar { L } _ { \mathrm { t o k e n } }$ defined in Equation 4, the KV active-byte coordinate is

$$
q _ { \mathrm { K V } } = \frac { K _ { \mathrm { K V } } ( \bar { L } _ { \mathrm { t o k e n } } + 1 ) } { 1 0 ^ { 9 } } .\tag{6}
$$

The 1 counts one input-token KV write per useful output, not an additional historical read. At fixed history and record size, batching increases private KV work and useful outputs proportionally, leaving the per-output coordinate unchanged.

For a packed, homogeneous, unreplicated layout, $K _ { \mathrm { K V } } = 2 N _ { \ell } N _ { \mathrm { k v } } d _ { h } s _ { \mathrm { k v } }$ , where the factors count keys and values, layers, KV heads per layer, head dimension, and bytes per element. Multi-query and grouped-query attention reduce the KV-head count [2, 37]. Nonuniform or replicated layouts require summing the record footprints included in the declared accounting boundary.

Combining weight and KV active bytes. For constant $M _ { \mathrm { w } }$ and $K _ { \mathrm { K V } }$ , let $D _ { \mathrm { r w } }$ denote the total principal logical weight and KV read–write demand over the trace. Combining their contributions gives

$$
\frac { D _ { \mathrm { r w } } } { T } = A _ { \mathrm { t o k e n } } ^ { \mathrm { r w } } = \frac { M _ { \mathrm { w } } } { B _ { \mathrm { e f f } } } + K _ { \mathrm { K V } } ( \bar { L } _ { \mathrm { t o k e n } } + 1 ) .\tag{7}
$$

![](images/7099bb44a3acca3ab393dd085314c0cab7600dc7d7e6da22132ae40ee9682fad.jpg)

![](images/decbc335ad939ff732fcafe99cb682bf8bd067b831979a4b3727dd22f50506cc.jpg)  
Fig. 3. KV active bytes across three decode steps: growing history reads and one input-token KV write per useful output.

This identity captures two distinct scaling efects: batching amortizes shared weights, whereas attended history determines private KV demand. These active bytes are computed from model and layout information and the realized decode trace, not inferred from measured HBM trafic. We next combine these work coordinates with realized time to model token energy.

## 4.4 Modeling Token Energy with Active Bytes and Decode Time

Equal active-byte demand can accompany diferent execution times. Following the work–time perspective of energy roofline modeling [4], we use $\tau = D / T$ , the inverse aggregate useful-output throughput measured over the same interval. Within a fixed stratum $\sigma ,$ Figure 4 illustrates the conditional response $\Delta \widehat { e } = P _ { \tau , \sigma } \Delta \tau$ at fixed active bytes and output count. For $P _ { \tau , \sigma } > 0$ , longer bduration raises predicted token energy. This efective coeficient has units of watts and is fitted jointly with the byte responses across workload cells; it is neither run-specific average power nor independently measured idle power.

![](images/6e342814bdfdf7070d2f862309fee00dfde3138c2c5f623ec4cea5352adff9e9.jpg)  
Fig. 4. Conditional energy response to decode time at fixed active bytes, output count, and execution stratum $( P _ { \tau , \sigma } > 0 )$

We retain separate weight and KV coeficients because their reuse and kernel access patterns difer, allowing distinct energy responses per logical byte. For total dense-weight demand $S M _ { \mathrm { w } }$ , total KV demand $D _ { \mathrm { K V } } = T K _ { \mathrm { K V } } ( \bar { L } _ { \mathrm { t o k e n } } + 1 )$ , and decode duration �, we propose the first-order additive response

$$
E _ { \sigma } = c _ { \sigma } T + \alpha _ { \sigma } \frac { S M _ { \mathrm { w } } } { 1 0 ^ { 9 } } + \beta _ { \sigma } \frac { D _ { \mathrm { K V } } } { 1 0 ^ { 9 } } + P _ { \tau , \sigma } D + U _ { \sigma } .\tag{8}
$$

Dividing by the useful-output count � yields the Token-Energy Law:

$$
e _ { \mathrm { g r o s s } , \sigma } = c _ { \sigma } + \alpha _ { \sigma } q _ { \mathrm { w } } + \beta _ { \sigma } q _ { \mathrm { K V } } + P _ { \tau , \sigma } \tau + u _ { \sigma } ,\tag{9}
$$

where $u _ { \sigma } = U _ { \sigma } / T _ { \sigma }$

The coeficients $\alpha _ { \sigma }$ and $\beta _ { \sigma }$ , in J/decimal-GB, describe the efective responses to weight and KV demand. The intercept $c _ { \sigma }$ captures an approximately constant per-output ofset, and $u _ { \sigma }$ is the remaining discrepancy. These are efective responses rather than separately metered physical components; auxiliary computation and runtime activity can influence both the coeficients and the residual. Within a fixed execution stratum, the byte footprints are constant, so the response is algebraically equivalent to a regression on inverse efective batch, token-weighted history, and time per output. The constant KV-append contribution cannot be identified separately from the intercept through this regression. The model uses one coeficient set within each declared stratum, with held-out workloads testing its adequacy across diferent batches, histories, and durations. A deployment change may afect several coordinates together, so their joint response matters. Our evaluation uses observed decode time; prediction before execution additionally requires an independently validated duration estimate.

Section 5 tests the active-byte coordinates against HBM measurements and evaluates the combined work–time response on held-out workloads and against simpler alternatives.

## 5 Evaluation and Validation

We evaluate the Token-Energy Law using three dense language models on NVIDIA GH200 and GB300 GPUs. HBM trafic profiling examines how logical weight and KV demand correspond to physical memory trafic. Energy measurements assess the accuracy of predictions based on active bytes and decode time. Comparisons with simpler models establish the value of the separate weight, KV, and time terms, while experiments across devices, tensor-parallel configurations, and request-departure patterns test the law’s applicability.

## 5.1 Experimental Setup

5.1.1 Platforms and Workloads. We study these three dense language models: Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct, and Mistral-7B-Instruct-v0.3 [17, 27, 34]. All experiments use BF16 weights and KV state, FlashAttention, and full-history attention. The GH200 platform is a dual-GPU node with 144-GB HBM3e per device, running ARM64 NVIDIA vLLM 25.09 with a fixed 700-W power limit per GPU. Single-GPU experiments use one device, while tensor-parallel experiments use both. The GB300 experiments use an ARM64 vLLM runtime validated for that platform, a fixed 1,300-W power limit per GPU, and tensor-parallel degree one. The evaluated vLLM V0 runtime uses hybrid CUDA-graph/eager execution with a default sequence-capture cap of 8192 tokens. Workload cells vary realized batch size and token-weighted attended history. Each static-batch cell is evaluated in five separate runs. After an unmetered bootstrap, each static decode episode generates 1,024 metered outputs per request on GH200 and 2,048 on GB300, with all requests remaining active throughout the episode. Each formal energy run accumulates at least 30 seconds of metered decode, summing energy, duration, and outputs across separately bracketed episodes and excluding intervening preparation. Separate GH200 trace experiments vary request output budgets and departure patterns, as described in Section 5.4. Iteration traces determine realized batch size and token-weighted attended history, while archived model inventories provide the weight and KV-layout information used to compute active bytes. Prefix caching, speculative decoding, chunked prefill, ofload, and preemption are disabled.

5.1.2 Energy and Trafic Measurement. Active-byte coordinates are computed from archived BF16 model inventories and execution traces, independently of HBM measurements. The weight coordinate includes recurring dense operators and the output projection, but excludes the studied models’ untied input-embedding tables and rotary-position caches; recorded storage aliases are counted once. This inventory-based correction does not constitute a new kernel-access or rank-local audit. Corresponding coeficient rescaling preserves the separate-coordinate models’ predictions without changing the measurements. The corrected total-bytes baseline requires a post-hoc refit using the original training runs, as detailed in the ablation results.

The primary outcome is gross GPU-board energy per useful decode output. We integrate scope-0 instantaneous NVML power over synchronized decode boundaries, using the same interval for duration and useful-output count. Samples must bracket both endpoints, with no gap above 50 ms. On GH200, integrated scope-1 module power must also agree with the scope-matched cumulative energy counter within 2%; GB300 uses the qualified scope-0 integral without this independent counter check. For two-GPU tensor parallelism, board energies are summed and normalized by the shared output count. Runs must satisfy the specified execution configuration, output counts, and scheduler constraints. Technical failures are retained but excluded from scientific repetitions; acceptance is independent of measured energy. Independent Nsight Compute executions measure HBM reads and writes for one post-bootstrap decode iteration. The dataset contains 177 profiles: 60 for Qwen7B, 45 for Qwen14B, and 45 for Mistral on GH200, plus 27 for Qwen7B on GB300. Application-range replay uses one metric range, one replay pass, zero warmup passes, and no profiler-controlled cache flushing or clock adjustment. The two-coordinate read model is fitted without an intercept, and its errors describe agreement on the profiled grid. Profiling is separate from all energy and latency measurements.

5.1.3 Identification and Validation. We estimate energy coeficients from run-level observations using ordinary least squares with HC3 standard errors [26]. Fitting and held-out evaluation use disjoint batch–history cells. Separately interleaved executions are the statistical units. The primary GH200 Qwen7B response is fitted to 75 development observations, comprising all runs from the initial fitting, calibration, and evaluation stages. Its coeficients were frozen before collecting 45 new executions at nine previously unobserved batch–history cells. For the other six locally identified responses, residual-calibration observations do not enter the coeficient fits.

Table 2 lists the workload grids using $B = B _ { \mathrm { e f f } }$ and $H = \bar { L } _ { \mathrm { t o k e n } } / 1 0 2 4$ . Each listed batch is paired with every listed history, with five runs per cell. Q7, Q14, and M7 denote Qwen2.5-7B, Qwen2.5-14B, and Mistral-7B; the four model–platform instances grouped in the second row are fitted separately. For each evaluation cell, we average measured and predicted energy per token separately and compute the absolute relative error as $| \bar { e } _ { j } - \overline { { \widehat { e } } } _ { j } | / \bar { e } _ { j }$ . Prespecified accuracy criteria require median band maximum cell-mean errors of at most 5% and 10%, respectively. Error bars show pointwise 95% Student-� confidence intervals for measured means; standard-deviation shading is identified separately in figure captions.

The same-SKU transfer test applies the GH200 GPU0 coeficients unchanged to 45 GPU1 executions on the same nine-cell evaluation grid. Static-batch validation therefore comprises 270 executions across 54 cohort-specific cells. Trace validation adds 40 executions each for Qwen7B and Mistral, followed by 24 additional Qwen repetitions reported separately. The resulting 374 validation executions exclude fitting, residual calibration, qualification, and HBM profiling.

Use of generative AI. Generative AI tools, including OpenAI Codex, assisted with experimental planning, code development and debugging, data analysis and visualization, and manuscript drafting and revision. The authors take full responsibility for the methods, results, and conclusions presented in this paper.

Table 2. Workload grids for identification and held-out evaluation.
<table><tr><td>Configuration</td><td>Fitting / development</td><td>Residual calibration</td><td>Held-out evaluation</td></tr><tr><td>GH200, Q7</td><td> $B \in \{ 4 , 8 , 1 6 , 2 4 , 3 2 \}$   $H \in \{ 4 , 8 , 1 6 \}$  75 runs</td><td>No independent set</td><td> $B \in \{ 1 2 , 2 0 , 2 8 \}$   $H \in \{ 6 , 1 0 , 1 4 \}$  45 runs</td></tr><tr><td>GH200 / GB300 Q14 and M7</td><td> $B \in \{ 4 , 1 6 \}$   $H \in \{ 4 , 1 0 , 1 6 \}$   $3 0 \mathrm { \ r u n s \ e a c h }$ </td><td> $B = 8$   $H \in \{ 4 , 1 0 , 1 6 \}$   $1 5 { \mathrm { ~ r u n s ~ e a c h } }$ </td><td> $B \in \{ 6 , 1 2 \}$   $H \in \{ 6 , 1 2 , 1 4 \}$   $3 0 \mathrm { \ r u n s \ e a c h }$ </td></tr><tr><td>GH200, Q7 TP=2</td><td> $B \in \{ 4 , 3 2 \}$   $H \in \{ 4 , 1 0 , 1 6 \}$  30 runs</td><td> $B = 1 6$   $H \in \{ 4 , 1 0 , 1 6 \}$   $1 5 \mathrm { \ r u n s }$ </td><td> $B \in \{ 8 , 2 4 \}$   $H \in \{ 6 , 1 2 , 1 4 \}$  30 runs</td></tr><tr><td>GB300, Q7</td><td> $B \in \{ 4 , 3 2 \}$   $H \in \{ 4 , 8 , 1 6 \}$  30 runs</td><td> $B = 1 6$   $H \in \{ 4 , 8 , 1 6 \}$  15 runs</td><td> $B \in \{ 1 2 , 2 8 \}$   $H \in \{ 6 , 1 0 , 1 4 \}$  30 runs</td></tr></table>

## 5.2 Active Bytes and Measured HBM Trafic

![](images/9579097dc3e627fbeeada59fffddd4ed2a86e884463a7f5c7004a0f90fe4a7f8.jpg)  
Qwen2.5-7B / GH200 | 60 profiles; 5 per cell | Whiskers: pointwise 95% mean CIs  
Fig. 5. Qwen2.5-7B HBM reads on GH200. Top: logical weight/KV components, their unscaled sum, and measured means. Botom: 60 fit residuals, $1 0 ^ { 4 } ( Q - \widehat { Q } ) / \widehat { Q } ;$ diamonds mark five-run means. Whiskers show pointwise 95% Student-� confidence intervals.

The active-byte coordinates describe the logical byte demand of dense shared weights and private KV histories. Caching and kernel implementation afect how this demand translates into physical

memory trafic. We therefore test whether the read coordinates capture measured HBM trafic using 177 profiles covering three models on GH200 and Qwen7B on GB300. Batch and history vary independently across the profiled grids. We fit the no-intercept model

$$
\frac { Q _ { \mathrm { H B M , r e a d } } ^ { \mathrm { o b s } } } { T } = \rho _ { \mathrm { w } } \frac { M _ { \mathrm { w } } } { B _ { \mathrm { e f f } } } + \rho _ { \mathrm { K V } } K _ { \mathrm { K V } } \bar { L } _ { \mathrm { t o k e n } } + \varepsilon _ { \mathrm { H B M } } ,\tag{10}
$$

where the dimensionless coeficients relate logical read demand to aggregate HBM reads.

Table 3. HBM-read fits on GH200 for $B \in \{ 4 , 8 , 1 6 \}$ and $\bar { L } ~ \in ~ \{ 4 0 9 6 , 8 1 9 2 , 1 6 3 8 4 \}$ (45 profiles per model). Dense-sum gap is the mean absolute relative deviation of measured reads from logical weight-plus-KV reads, normalized by logical reads.
<table><tr><td rowspan="2">Model</td><td rowspan="2"> $\rho _ { \mathrm { w } }$ </td><td rowspan="2">ρKV</td><td rowspan="2">MAPE</td><td rowspan="2">Max. APE</td><td rowspan="2">Dense-sum gap</td></tr><tr><td></td></tr><tr><td>Qwen2.5-7B</td><td>1.001189</td><td>1.002124</td><td>0.0460%</td><td>0.1757%</td><td>0.1714%</td></tr><tr><td>Qwen2.5-14B</td><td>1.001661</td><td>1.000586</td><td>0.0324%</td><td>0.0822%</td><td>0.1494%</td></tr><tr><td>Mistral-7B</td><td>1.001828</td><td>1.000494</td><td>0.0304%</td><td>0.0787%</td><td>0.1490%</td></tr></table>

For the full 60-profile Qwen7B grid on GH200, $\rho _ { \mathrm { w } } = 1 . 0 0 1 2 1 3$ and $\rho _ { \mathrm { K V } } = 1 . 0 0 2 0 4 2$ , with a mean absolute percentage error of 0.0498% relative to measured reads. Figure 5 shows the underlying work decomposition. Weight demand per useful output decreases with batch, while KV demand remains constant at fixed history and increases with history length. Measured HBM reads closely follow the unscaled sum of these logical demands; the lower panels show residuals relative to the fitted two-coordinate model.

Table 3 extends the comparison to three models on their shared nine-cell GH200 grid. For Qwen14B and Mistral, extending history from 4096 to 16384 at fixed batch increases measured reads by 0.99793–1.00045 times the additional logical KV demand, supporting the history-proportional KV term. The weight coeficients refer to the audited dense shared-weight footprint defined in the experimental setup.

On GB300, the same form gives $\rho _ { \mathrm { w } } = 1 . 0 0 0 7 8 4$ and $\rho _ { \mathrm { K V } } = 1 . 0 0 4 9 5 3$ (Table 4). The matched crossplatform grid uses $B \in \{ 4 , 1 6 , 3 2 \}$ and $\bar { L } _ { \mathrm { t o k e n } } ~ \in$ 4096, 8192, 16384 . Across this grid, GH200 and GB300 cell-mean reads difer by at most 0.2384%. These results support the use of shared-weight and private-KV active bytes to describe HBM reads across the tested hardware and runtime configurations. These fits characterize aggregate reads on the profiled grids. Aggregate writes include trafic beyond logical KV appends and are outside this read-model validation.

Table 4. Qwen2.5-7B HBM-read fits on matched grids.
<table><tr><td>Metric</td><td>GH200</td><td>GB300</td></tr><tr><td>Profiles</td><td>45</td><td>27</td></tr><tr><td> $\rho _ { \mathrm { w } }$ </td><td>1.001003</td><td>1.000784</td></tr><tr><td> $\rho _ { \mathrm { K V } }$ </td><td>1.002258</td><td>1.004953</td></tr><tr><td>MAPE</td><td>0.0444%</td><td>0.0730%</td></tr><tr><td>Max. APE</td><td>0.1172%</td><td>0.2440%</td></tr></table>

## 5.3 Validating the Work–Time Energy Model

The Token-Energy Law predicts held-out decode energy using active-byte coordinates and observed decode time per useful output. For Qwen2.5-7B on GH200 GPU0, nine evaluation cells cross $\bar { L } _ { \mathrm { t o k e n } } \in$ 6144, 10240, 14336 with $B _ { \mathrm { e f f } } \in \{ 1 2 , 2 0 , 2 8 \}$ , with five independent executions per cell. With coeficients fixed before evaluation, the median and maximum absolute relative errors of cell means are 0.409% and 2.22%, respectively. Figure 6(a) compares the predicted energy contributions with measurements; panel (c) summarizes the cell-mean residuals.

![](images/77a4044af10e658a7cf2fe2e9bc8893e73f5040de9c6eb6a525bc900f7335c9b.jpg)  
Fig. 6. Qwen2.5-7B energy prediction on GH200 GPU0 and unchanged-coeficient transfer to GPU1. Stacks show model terms; diamonds and whiskers show measured means and 95% confidence intervals. Residuals $1 0 0 ( e - { \widehat { e } } ) / e \mathrm { ~ }$

The combined weight, KV, and observed-time terms track measured energy across the heldout workloads (Figure 6). Stacked segments show fitted model contributions, not independently metered energy components. To assess the contribution of the modeling terms, Table 5 compares four specifications for each GB300 model using the same 30 fitting runs and six held-out cells. These specifications retain separate weight and KV coordinates with time, separate byte coordinates alone, time alone, or a single total-byte coordinate with time. The total-bytes-plus-time specification constrains weight and KV demand to have the same energy coeficient. The complete work–time

Table 5. Maximum cell-mean absolute percentage error on GB300 holdouts (six cells and 30 runs per model).
<table><tr><td>Model</td><td>Bytes + time Bytes only</td><td></td><td></td><td>Time only Total bytes + time</td></tr><tr><td>Qwen2.5-7B</td><td>2.137%</td><td>35.458%</td><td>29.395%</td><td>5.721%</td></tr><tr><td>Qwen2.5-14B</td><td>0.971%</td><td>28.228%</td><td>10.972%</td><td>4.381%</td></tr><tr><td>Mistral-7B</td><td>0.887%</td><td>33.564%</td><td>14.117%</td><td>5.589%</td></tr></table>

model gives the lowest maximum cell-mean error for all three models, ranging from 0.887% to 2.137%. Constraining weight and KV bytes to share one coeficient raises these errors to 4.381–5.721%, although this specification also meets the prespecified median and maximum error thresholds. Separate weight and KV coeficients therefore improve accuracy on the tested grids. Omitting either realized time or the byte coordinates produces maximum errors above 10% for all three models. The corrected total-bytes-plus-time comparison is a descriptive post-hoc reanalysis fitted only to the original training runs; held-out outcomes were not used to estimate its coeficients.

## 5.4 Validation under Changing Request Membership

We apply the previously identified Qwen2.5-7B and Mistral-7B coeficients to GH200 decode traces in which requests finish at diferent iterations. Each model has eight conditions combining $B _ { \mathrm { e f f } } \in \{ 1 0 , 2 0 \}$ with a static control and three draining patterns. The draining patterns use uniform initial histories or positive or negative associations between initial history length and request lifetime. Drains begin with 16 or 32 requests that depart in four equal groups, with no new arrivals. All conditions match the token-weighted history $\bar { \bar { L } } _ { \mathrm { t o k e n } } = 9 9 8 3 . \bar { 5 }$ . Five repetitions per condition give 40 runs per model; three additional Qwen repetitions per condition bring the total to 104 executions.

![](images/ec5acf7547c9e5762f0f7bd875df2dbb4cb977d7253dd512dd214279c3ae6b1e.jpg)  
Stacks show frozen model terms; dots and diamonds show measured runs and means.

Fig. 7. Frozen-coeficient predictions under changing request membership on GH200. Stacks show mode terms; dots and diamonds denote measured runs and condition means.

In Figure 7, matching $B _ { \mathrm { e f f } }$ and $\bar { L } _ { \mathrm { t o k e n } }$ within each model fixes the weight and KV contributions across departure patterns; diferences in predicted energy enter only through observed �. Increasing $B _ { \mathrm { e f f } }$ from 10 to 20 halves the weight term while leaving the matched-history KV term unchanged.

We examine the coordinate definitions through post-hoc input substitutions on the original 40- run cohort for each model, excluding the 24 additional Qwen runs. Let $B _ { 0 }$ denote initial cohort size and ${ \bar { L } } _ { \mathrm { r e q } }$ the equally weighted average of each request’s mean attended history over its metered outputs. We replace either $B _ { \mathrm { e f f } }$ with $B _ { 0 }$ or $\bar { L } _ { \mathrm { t o k e n } }$ with ${ \bar { L } } _ { \mathrm { r e q } } ,$ , keeping the coeficients, observed duration, and all other inputs fixed.

Table 6. Maximum condition-mean APE (%) after coordinate substitution.
<table><tr><td>Coordinates</td><td>Q7</td><td>M7</td></tr><tr><td>Realized</td><td>0.404</td><td>0.870</td></tr><tr><td> $B _ { 0 }$  replaces  $B _ { \mathrm { e f f } }$ </td><td>9.824</td><td>8.258</td></tr><tr><td> ${ \bar { L } } _ { \mathrm { r e q } }$  replaces  $\bar { L } _ { \mathrm { t o k e n } }$ </td><td>3.578</td><td>6.395</td></tr></table>

Both substitutions increase the maximum

condition-mean error despite retaining observed duration (Table 6). This diagnostic supports accounting for realized weight sharing and output-weighted KV history when transferring the fixed coeficients to changing request membership.

Every condition-mean error remains below 1% within its evaluation cohort, including Mistral at $B _ { \mathrm { e f f } } = 2 0$ , beyond its identification maximum of 16. Additional Qwen repetitions retain agreement across three blocks separated by at least two hours, supporting unchanged-coeficient transfer to these controlled departure patterns.

## 5.5 Device Transfer and Tensor-Parallel Execution

5.5.1 Transfer between physical devices. The Qwen2.5-7B coeficients identified on GH200 GPU0 predict energy on GPU1 of the same node without refitting. The model, runtime, board power limit, and workload grid remain fixed. Across 45 executions at nine target-device cells, the maximum cell-mean error is 3.7234%, rounded to 3.72% (Figure 6). Predicted cell means are consistently below the measured GPU1 means, while all cells satisfy the evaluation criterion. This result supports coeficient transfer between the two tested devices of the same SKU under the matched execution configuration.

5.5.2 Two-GPU tensor-parallel decode. The Token-Energy Law also describes Qwen2.5-7B decoding across two NVLink-connected GH200 GPUs using coeficients identified for this tensor-parallel configuration. Energy is summed across both boards and normalized by their common usefuloutput count. We evaluate the fixed coeficients on six held-out conditions crossing $\bar { L } _ { \mathrm { t o k e n } } ~ \in$ 6144, 12288, 14336 with $B _ { \mathrm { e f f } } \in \{ 8 , 2 4 \}$ , with five executions per condition.

![](images/4366997527746f46a78d3884db5fd26091cefa5338ce425f88dbb813b6bd3337.jpg)  
Fig. 8. Qwen2.5-7B decoding on two GH200 boards. (a) Measured per-board energy and frozen total-energy predictions. (b) Realized decode time per useful output. (c) Signed errors of condition means. Dots show five runs per condition; whiskers are pointwise 95% Student-� confidence intervals for measured means.

Figure 8 compares per-board measurements with frozen summed-board predictions, alongside observed time and signed errors. At each history, batch 24 has lower energy and shorter time per output than batch 8. Across six condition means, median and maximum absolute relative errors are 0.245% and 0.559%, respectively.

These results support the same active-byte and time formulation for aggregate two-board energy after local calibration. The inputs use full-model reference byte footprints, and the fitted $P _ { \tau } = 3 1 4 . 1 0 1$ W characterizes the deployment’s time-associated energy response. Rank-local trafic and NVLink energy are not separately measured.

## 5.6 Cross-Model and Cross-Platform Validation

The same active-byte and time formulation describes held-out decode energy for Qwen2.5-7B Qwen2.5-14B, and Mistral-7B on GH200 and GB300. Qwen2.5-14B extends the evaluation to a larger model within the Qwen family, while Mistral-7B changes the architecture and KV geometry at a comparable parameter scale. Each model and execution configuration has its own coeficient vector, which remains fixed across the evaluated batches and histories.

Identified energy responses. The fitted instances make the common structure explicit. Let Q7, Q14, and M7 denote Qwen2.5-7B, Qwen2.5-14B, and Mistral-7B. Using the model-specific active-byte coordinates $q _ { \mathrm { w } }$ and �<sub>KV</sub> in decimal GB per useful output, and realized time � in seconds per useful output, the single-GPU responses are

$$
\widehat { e } _ { \mathrm { Q 7 } } ^ { \mathrm { G H 2 0 0 } } \simeq 0 . 0 1 2 3 + 0 . 1 1 9 4 q _ { \mathrm { w } } + 0 . 1 8 0 1 q _ { \mathrm { K V } } + 1 3 9 . 5 6 \tau ,
$$

$$
\widehat { e } _ { \mathrm { Q 1 4 } } ^ { \mathrm { G H 2 0 0 } } \simeq 0 . 0 1 7 9 + 0 . 1 1 9 0 q _ { \mathrm { w } } + 0 . 1 8 3 7 q _ { \mathrm { K V } } + 1 3 9 . 0 7 \tau ,\tag{11}
$$

$$
\widehat { e } _ { \mathrm { M 7 } } ^ { \mathrm { G H 2 0 0 } } \simeq 0 . 0 0 9 5 + 0 . 1 1 6 6 q _ { \mathrm { w } } + 0 . 1 8 2 5 q _ { \mathrm { K V } } + 1 3 8 . 4 3 \tau ,
$$

and

$$
\widehat { e } _ { \mathrm { Q 7 } } ^ { \mathrm { G B 3 0 0 } } \simeq 0 . 0 0 9 9 + 0 . 1 1 7 3 q _ { \mathrm { w } } + 0 . 1 9 1 5 q _ { \mathrm { K V } } + 1 9 8 . 5 7 \tau ,
$$

$$
\widehat { e } _ { \mathrm { Q 1 4 } } ^ { \mathrm { G B 3 0 0 } } \simeq 0 . 0 1 7 3 + 0 . 1 1 6 3 q _ { \mathrm { w } } + 0 . 1 9 2 7 q _ { \mathrm { K V } } + 1 9 9 . 3 3 \tau ,\tag{12}
$$

$$
\widehat { e } _ { \mathrm { M 7 } } ^ { \mathrm { G B 3 0 0 } } \simeq 0 . 0 1 0 8 + 0 . 1 1 3 4 q _ { \mathrm { w } } + 0 . 1 8 8 8 q _ { \mathrm { K V } } + 2 0 0 . 9 5 \tau .
$$

All responses give joules per useful output token. Coeficients are rounded for presentation; predictions use the full-precision values. For the two-GPU GH200 configuration evaluated in Section 5.5.2, the separately identified response is

$$
\widehat { e } _ { \mathrm { Q 7 } } ^ { \mathrm { G H 2 0 0 , T P 2 } } \simeq 0 . 0 1 4 4 + 0 . 1 2 0 4 q _ { \mathrm { w } } + 0 . 1 8 0 6 q _ { \mathrm { K V } } + 3 1 4 . 1 0 \tau .\tag{13}
$$

Here the outcome sums both boards’ energy, and the byte coordinates use full-model reference footprints. The same-SKU device-transfer experiment instead retains the GH200 Q7 response unchanged.

Table 7 reports HC3 standard errors for the seven fitted responses. Across these fits, standardized design-matrix condition numbers range from 3.51 to 4.55, with a maximum variance-inflation factor of 5.69. The standard errors quantify coeficient uncertainty conditional on the observed design.

Table 7. HC3 standard errors of the fited coeficients.
<table><tr><td>Platform</td><td>Model</td><td> $\operatorname { S E } ( c )$  J/token</td><td> $\operatorname { S E } ( \alpha )$  J/decimal-GB</td><td> ${ \mathrm { S E } } ( \beta )$  J/decimal-GB</td><td> $\mathrm { S E } ( P _ { \tau } )$  W</td></tr><tr><td>GH200</td><td>Q7</td><td>0.000861</td><td>0.000861</td><td>0.001947</td><td>0.820</td></tr><tr><td>GH200</td><td>Q14</td><td>0.003848</td><td>0.000939</td><td>0.001830</td><td>0.986</td></tr><tr><td>GH200</td><td>M7</td><td>0.002479</td><td>0.000876</td><td>0.001865</td><td>0.564</td></tr><tr><td>GB300</td><td>Q7</td><td>0.001024</td><td>0.000431</td><td>0.002242</td><td>0.570</td></tr><tr><td>GB300</td><td>Q14</td><td>0.002096</td><td>0.000359</td><td>0.001181</td><td>0.369</td></tr><tr><td>GB300</td><td>M7</td><td>0.002688</td><td>0.001186</td><td>0.002215</td><td>0.959</td></tr><tr><td>GH200 TP=2</td><td>Q7</td><td>0.001174</td><td>0.000434</td><td>0.002019</td><td>0.267</td></tr></table>

The coeficients characterize each model and hardware–runtime configuration; their timeassociated energy contribution depends on both $P _ { \tau }$ and the observed �.

Table 8. Static-workload energy validation. Errors compare predicted and measured cell means; each cell contains five executions. The device-transfer test retains the source-device coeficients.
<table><tr><td>Platform</td><td>Model</td><td>Coefficients</td><td>Cells</td><td>Runs</td><td>Median</td><td>Maximum</td></tr><tr><td>GH200</td><td>Qwen2.5-7B</td><td>Local fit</td><td>9</td><td>45</td><td>0.409%</td><td>2.22%</td></tr><tr><td>GH200</td><td>Qwen2.5-14B</td><td>Local fit</td><td>6</td><td>30</td><td>0.130%</td><td>1.92%</td></tr><tr><td>GH200</td><td>Mistral-7B</td><td>Local fit</td><td>6</td><td>30</td><td>0.390%</td><td>1.43%</td></tr><tr><td>GH200 TP=2</td><td>Qwen2.5-7B</td><td>Local fit</td><td>6</td><td>30</td><td>0.245%</td><td>0.559%</td></tr><tr><td>GB300</td><td>Qwen2.5-7B</td><td>Local fit</td><td>6</td><td>30</td><td>0.637%</td><td>2.137%</td></tr><tr><td>GB300</td><td>Qwen2.5-14B</td><td>Local fit</td><td>6</td><td>30</td><td>0.172%</td><td>0.971%</td></tr><tr><td>GB300</td><td>Mistral-7B</td><td>Local fit</td><td>6</td><td>30</td><td>0.565%</td><td>0.887%</td></tr><tr><td>GH200 GPU1</td><td>Qwen2.5-7B</td><td>Zero-refit transfer</td><td>9</td><td>45</td><td>2.62%</td><td>3.72%</td></tr><tr><td colspan="3">Static-batch validation subtotal</td><td>54</td><td>270</td><td>一</td><td>一</td></tr></table>

Table 8 summarizes the eight static-batch evaluations, comprising seven locally calibrated instances and one unchanged-coeficient device transfer. Across 270 held-out executions at 54 cohort-specific conditions, all condition-mean predictions are within 3.73% of measurement.

![](images/2521abcdca2ea38669e34d059dfb1f7fd474cd5b63702558ff822dbcc9da39cc.jpg)  
Fig. 9. Held-out energy across models and platforms. Lines connect predictions using observed decode time; markers show measured means. Shading denotes 1 sample standard deviation across five runs per condition.

Held-out energy across models and platforms. Figure 9 summarizes 195 held-out executions at 39 conditions for six single-GPU model–platform pairs. Each pair retains one coeficient vector, with predictions evaluated using each run’s active bytes and observed decode time. The figure shows two consistent workload efects across the tested models and platforms. At fixed model, platform, and history, the largest tested batch has 33.2–46.8% lower measured mean energy per useful output than the smallest tested batch. In the law, batching amortizes shared-weight active bytes, while its accompanying execution-time changes enter through �. At fixed batch, longer histories increase private-KV active bytes, and measured mean energy rises in every plotted series.

For GH200 Qwen2.5-7B at batch 12, mean energy rises from 0.323 to 0.609 to 0.638 J/token across histories 6144, 10240, and 14336. Archived step timings increase sharply when the maximum active sequence length exceeds 8192 tokens, consistent with eager fallback. These energy changes reflect both KV demand and runtime-dependent duration, which enters the predictions through observed �.

Across these six locally fitted single-GPU instances, the maximum condition-mean absolute relative error is 2.22%. These results support the shared work–time form across the tested models and platforms without batch-specific or history-specific refitting.

## 6 Discussion and Limitations

Our validation covers GPU-board energy for full-history dense BF16 decode under the evaluated runtime policy, excluding host-CPU energy. The maximum active sequence length exceeds the capture cap throughout the metered departure tests. Transfer to diferent graph policies, multi-node execution, and MoE requires further validation.

Multi-node and pipeline-parallel execution. Communication, synchronization, and pipeline bubbles change execution duration and device utilization. Although $\tau = D / T$ includes these delays, it does not resolve computation, communication, and waiting across individual GPUs. Equal activebyte demand and duration can therefore accompany diferent aggregate energy, and $P _ { \tau }$ should not be interpreted as measured idle power. Our two-GPU NVLink experiments include communication efects within board energy, but do not validate multi-node or pipeline-parallel serving. Extensions require synchronized, boundary-matched measurements and may need stage- or communicationspecific coordinates without double-counting existing terms. Network equipment outside the GPU-board boundary requires separate accounting.

Mixture-of-experts models. The coordinate $M _ { \mathrm { w } } / B _ { \mathrm { e f f } }$ assumes a fixed dense-weight footprint shared across outputs within each iteration. In routed MoE models, expert sharing depends on token assignments, so identical batch sizes and per-token activated parameter counts can imply diferent batch-level weight demands. Activated bytes cannot simply replace $M _ { \mathrm { w } }$ and be divided by the full batch. Routing, placement, and caching afect physical trafic, while expert-parallel dispatch and load imbalance also afect duration. Extending the model requires routing-aware accounting of shared dense and expert work, followed by trafic and energy validation; recalibrating the dense coeficients alone is insuficient.

## 7 Conclusion

The Token-Energy Law provides a compact, interpretable account of GPU-board energy per useful output token for dense, full-history decoding. It combines two central factors: execution-derived active bytes and realized decode time. Active bytes distinguish dense-weight work shared across outputs from private KV work determined by attended history. Together with time per useful output, these coordinates yield a common energy response whose locally identified coeficients characterize the model and hardware–runtime configuration. Experiments across three dense models on Hopper and Blackwell support the same functional form with locally identified coeficients. Across the evaluated settings, held-out condition-mean energy estimates remain within 3.73% of measurement.

The law gives researchers and system designers a quantitative basis for relating serving choices to token-level energy costs. Combined with independently validated performance models, it can support energy budgeting and comparisons among deployment options. Its explicit accounting of shared weights, private histories, and execution time also provides a testable starting point for extending energy models to other workloads and serving systems.

## References

[1] Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav Gulavani, Alexey Tumanov, and Ramachandran Ramjee. 2024. Taming Throughput-Latency Tradeof in LLM Inference with Sarathi-Serve. In 18th USENIX Symposium on Operating Systems Design and Implementation. USENIX Association, 117–134. https: //www.usenix.org/conference/osdi24/presentation/agrawal

[2] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit Sanghai. 2023. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 4895–4901. doi:10.18653/v1/2023.emnlp-main.298

[3] Abhimanyu Bambhaniya, Ritik Raj, Geonhwa Jeong, Souvik Kundu, Sudarshan Srinivasan, Suvinay Subramanian, Midhilesh Elavazhagan, Madhu Kumar, and Tushar Krishna. 2025. Demystifying AI Platform Design for Distributed Inference of Next-Generation LLM Models. arXiv:2406.01698 doi:10.48550/arXiv.2406.01698 Version 3, revised 15 May 2025.

[4] Jee Whan Choi, Daniel Bedard, Robert Fowler, and Richard Vuduc. 2013. A Roofline Model of Energy. In 2013 IEEE 27th International Symposium on Parallel and Distributed Processing. IEEE, Los Alamitos, CA, USA, 661–672. doi:10.1109/IPDPS.2013.77

[5] Jae-Won Chung, JefJ. Ma, Ruofan Wu, Jiachen Liu, Oh Jun Kweon, Yuxuan Xia, Zhiyu Wu, and Mosharaf Chowdhury. 2025. The ML.ENERGY Benchmark: Toward Automated Inference Energy Measurement and Optimization. In Advances in Neural Information Processing Systems, Vol. 38. Curran Associates, Inc., Red Hook, NY, USA, 29 pages. doi:10.52202 085713-3652

[6] Jae-Won Chung, Ruofan Wu, Jef J. Ma, and Mosharaf Chowdhury. 2026. Where Do the Joules Go? Diagnosing Inference Energy Consumption. arXiv:2601.22076 doi:10.48550/arXiv.2601.22076

[7] Tri Dao. 2024. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. In The Twelfth International Conference on Learning Representations. 35549–35562. https://proceedings.iclr.cc/paper\_files/paper/2024 hash/98ed250b203d1ac6b24bbcf263e3d4a7-Abstract-Conference.htm

[8] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. FlashAttention: Fast and Memory-Eficient Exact Attention with IO-Awareness. In Advances in Neural Information Processing Systems, Vol. 35. Curran Associates, Inc., Red Hook, NY, USA, 16344–16359. doi:10.52202/068431-1189

[9] Venmugil Elango, Naser Sedaghati, Fabrice Rastello, Louis-Noël Pouchet, J. Ramanujam, Radu Teodorescu, and P. Sadayappan. 2015. On Using the Roofline Model with Lower Bounds on Data Movement. ACM Transactions on Architecture and Code Optimization 11, 4, Article 67 (2015), 23 pages. doi:10.1145/2693656

[10] Cooper Elsworth, Keguo Huang, David Patterson, Ian Schneider, Robert Sedivy, Savannah Goodman, Ben Townsend, Parthasarathy Ranganathan, Jef Dean, Amin Vahdat, Ben Gomes, and James Manyika. 2025. Measuring the Environmental Impact of Delivering AI at Google Scale. arXiv:2508.15734 doi:10.48550/arXiv.2508.15734

[11] Ahmad Faiz, Sotaro Kaneda, Ruhan Wang, Rita Osi, Prateek Sharma, Fan Chen, and Lei Jiang. 2024. LLMCarbon: Modeling the End-to-End Carbon Footprint of Large Language Models. In The Twelfth International Conference on Learning Representations. https://proceedings.iclr.cc/paper\_files/paper/2024/hash/6b4ac044095525631df38e20919b45d2- Abstract-Conference.html

[12] Jared Fernandez, Clara Na, Vashisth Tiwari, Yonatan Bisk, Sasha Luccioni, and Emma Strubell. 2025. Energy Considerations of Large Language Model Inference and Eficiency Optimizations. In Proceedings ofthe 63rd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, Vienna, Austria, 32556–32569. doi:10.18653/v1/2025.acl-long.1563

[13] Zhenxiao Fu, Fan Chen, Shan Zhou, Haitong Li, and Lei Jiang. 2025. LLMCO2: Advancing Accurate Carbon Footprint Prediction for LLM Inferences. ACM SIGEnergy Energy Informatics Review 5, 2 (2025), 63–68. doi:10.1145/3757892. 3757901

[14] Jia-Wei Hong and H. T. Kung. 1981. I/O Complexity: The Red-Blue Pebble Game. In Proceedings ofthe Thirteenth Annual ACM Symposium on Theory of Computing. ACM, 326–333. doi:10.1145/800076.802486

[15] Mark Horowitz. 2014. Computing’s Energy Problem (and What We Can Do about It). In 2014 IEEE International Solid-State Circuits Conference Digest ofTechnical Papers. IEEE, 10–14. doi:10.1109/ISSCC.2014.6757323

[16] Md. Monzurul Amin Ifath and Israat Haque. 2026. Characterizing Performance–Energy Trade-ofs of Large Language Models in Multi-Request Workflows. Proceedings of the ACM on Measurement and Analysis of Computing Systems 10, 1 (2026), 26 pages. doi:10.1145/3788089

[17] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7B. arXiv:2310.06825 doi:10.48550/arXiv.2310.06825

[18] Vijay Kandiah, Scott Peverelle, Mahmoud Khairy, Junrui Pan, Amogh Manjunath, Timothy G. Rogers, Tor M. Aamodt, and Nikos Hardavellas. 2021. AccelWattch: A Power Modeling Framework for Modern GPUs. In Proceedings ofthe 54th Annual IEEE/ACM International Symposium on Microarchitecture. ACM, 738–753. doi:10.1145/3466752.3480063

[19] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Eficient Memory Management for Large Language Model Serving with PagedAttention. In Proceedings ofthe 29th Symposium on Operating Systems Principles. ACM, New York, NY, USA, 611–626. doi:10.1145 3600006.3613165

[20] Kyungmi Lee, Zhiye Song, Eun Kyung Lee, Xin Zhang, Tamar Eilam, and Anantha P. Chandrakasan. 2026. EnergAIzer: Fast and Accurate GPU Power Estimation Framework for AI Workloads. In 2026 IEEE International Symposium on Performance Analysis ofSystems and Software (ISPASS). IEEE, 435–447. doi:10.1109/ISPASS69572.2026.00049

[21] Jingwen Leng, Tayler Hetherington, Ahmed ElTantawy, Syed Gilani, Nam Sung Kim, Tor M. Aamodt, and Vijay Janapa Reddi. 2013. GPUWattch: Enabling Energy Optimizations in GPGPUs. In Proceedings ofthe 40th Annual International Symposium on Computer Architecture. ACM, 487–498. doi:10.1145/2485922.2485964

[22] Yaniv Leviathan, Matan Kalman, and Yossi Matias. 2023. Fast Inference from Transformers via Speculative Decoding. In Proceedings ofthe 40th International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 202). PMLR, 19274–19286. https://proceedings.mlr.press/v202/leviathan23a.html

[23] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. 2024. AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration. In Proceedings ofMachine Learning and Systems, Vol. 6. 87–100. https://proceedings.mlsys.org paper\_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html

[24] Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. 2024. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. In Proceedings ofthe 41st International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 235). PMLR, 32332–32344. https: //proceedings.mlr.press/v235/liu24bz.html

[25] Sasha Luccioni, Yacine Jernite, and Emma Strubell. 2024. Power Hungry Processing: Watts Driving the Cost of AI Deployment?. In The 2024 ACM Conference on Fairness, Accountability, and Transparency. ACM, 85–99. doi:10.1145 3630106.3658542

[26] James G. MacKinnon and Halbert White. 1985. Some Heteroskedasticity-Consistent Covariance Matrix Estimators wit Improved Finite Sample Properties. Journal of Econometrics 29, 3 (1985), 305–325. doi:10.1016/0304-4076(85)90158-7

[27] Mistral AI. 2024. Model Card for Mistral-7B-Instruct-v0.3. Hugging Face model card. https://huggingface.co/mistralai/ Mistral-7B-Instruct-v0.3

[28] Chenxu Niu, Wei Zhang, Jie Li, Yongjian Zhao, Tongyang Wang, Xi Wang, and Yong Chen. 2026. TokenPowerBench: Benchmarking the Power Consumption ofLLM Inference. In Proceedings ofthe AAAIConference on Artificial Intelligence, Vol. 40. AAAI Press, Palo Alto, CA, USA, 32582–32590. doi:10.1609/aaai.v40i38.40535

[29] Chenxu Niu, Wei Zhang, Yongjian Zhao, and Yong Chen. 2025. Energy Eficient or Exhaustive? Benchmarking Power Consumption of LLM Inference Engines. ACM SIGEnergy Energy Informatics Review 5, 2 (2025), 56–62. doi:10.1145/3757892.3757900

[30] Vittorio Palladino, Gianluca Palermo, Michael E. Papka, and Zhiling Lan. 2026. EnergyLens: Interpretable Closed-Form Energy Models for Multimodal LLM Inference Serving. arXiv:2605.10556 doi:10.48550/arXiv.2605.10556

[31] Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki, and Ricardo Bianchini. 2024. Splitwise: Eficient Generative LLM Inference Using Phase Splitting. In 2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture. IEEE, 118–132. doi:10.1109/ISCA59077.2024.00019

[32] Sundar Pichai. 2026. I/O 2026: Welcome to the Agentic Gemini Era. Google Blog. https://blog.google/intl/enafrica/products/explore-get-answers/sundar-pichai-io-2026/ Published 19 May 2026.

[33] Hiari Pizzini Cavagna, Andrea Proia, Giacomo Madella, Giovanni Battista Esposito, Francesco Antici, Daniele Cesarini, Zeynep Kiziltan, and Andrea Bartolini. 2026. SweetSpot: An Analytical Model for Predicting Energy Eficiency of LLM Inference. In Proceedings ofthe 17th ACM/SPEC International Conference on Performance Engineering. ACM, New York, NY, USA, 83–95. doi:10.1145/3777884.3797011

[34] Qwen Team. 2025. Qwen2.5 Technical Report. arXiv:2412.15115 doi:10.48550/arXiv.2412.15115 Version 2, revised 3 January 2025.

[35] Barna Saha and Christopher Ye. 2024. I/O Complexity of Attention, or How Optimal is FlashAttention?. In Proceedings ofthe 41st International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 235). PMLR, Vienna, Austria, 43024–43042. https://proceedings.mlr.press/v235/saha24a.htm

[36] Siddharth Samsi, Dan Zhao, Joseph McDonald, Baolin Li, Adam Michaleas, Michael Jones, William Bergeron, Jeremy Kepner, Devesh Tiwari, and Vijay Gadepally. 2023. From Words to Watts: Benchmarking the Energy Costs of Large Language Model Inference. In 2023 IEEE High Performance Extreme Computing Conference (HPEC). IEEE, 1–9. doi:10.1109/HPEC58863.2023.10363447

[37] Noam Shazeer. 2019. Fast Transformer Decoding: One Write-Head is All You Need. arXiv:1911.02150 doi:10.48550 arXiv.1911.02150

[38] Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Beidi Chen, Percy Liang, Christopher Ré, Ion Stoica, and Ce Zhang. 2023. FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU. In Proceedings of the 40th International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 202). PMLR, 31094–31116. https://proceedings.mlr.press/v202/sheng23a.html

[39] Zhiye Song, Kyungmi Lee, Eun Kyung Lee, Xin Zhang, Tamar Eilam, and Anantha P. Chandrakasan. 2026. EnergyLens: Predictive Energy-Aware Exploration for Multi-GPU LLM Inference Optimization. arXiv:2605.14249 doi:10.48550/ arXiv.2605.14249

[40] Tina Vartziotis, Rodopi Kosteli, Elli Danae Vartziotis, George Dasoulas, Michael Keckeisen, Konstantinos Skianis, Sotirios Kotsopoulos, and Francesca Dominici. 2026. From Tokens to Watt-hours: Analytical Energy Estimation for LLM Inference on Modern GPUs. arXiv:2607.26571 doi:10.48550/arXiv.2607.26571

[41] Prabhu Vellaisamy, Vanessa Lam, Shawn Blanton, and John Paul Shen. 2026. Characterization of Request and Token Energy Costs for LLM Inference Workloads on GPU Platforms. arXiv:2608.28044 doi:10.48550/arXiv.2608.28044

[42] Samuel Williams, Andrew Waterman, and David Patterson. 2009. Roofline: An Insightful Visual Performance Mode for Multicore Architectures. Commun. ACM 52, 4 (2009), 65–76. doi:10.1145/1498765.1498785

[43] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. 2023. SmoothQuant: Accurate and Eficient Post-Training Quantization for Large Language Models. In Proceedings ofthe 40th International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 202). PMLR, 38087–38099. https://proceedings.mlr. press/v202/xiao23c.html

[44] Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, Stephanie Wang, Tianqi Chen, Baris Kasikci, Vinod Grover, Arvind Krishnamurthy, and Luis Ceze. 2025. FlashInfer: Eficient and Customizable Attention Engine for LLM Inference Serving. In Proceedings ofMachine Learning and Systems, Vol. 7. MLSys. https://proceedings.mlsys.org paper\_files/paper/2025/hash/dbf02b21d77409a2db30e56866a8ab3a-Abstract-Conference.html

[45] Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-Gon Chun. 2022. Orca: A Distributed Serving System for Transformer-Based Generative Models. In 16th USENIX Symposium on Operating Systems Design and Implementation. USENIX Association, 521–538. https://www.usenix.org/conference/osdi22/presentation/yu

[46] Hengrui Zhang, August Ning, Rohan Baskar Prabhakar, and David Wentzlaf. 2024. LLMCompass: Enabling Eficient Hardware Design for Large Language Model Inference. In 2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture. IEEE, 1080–1096. doi:10.1109/ISCA59077.2024.00082

[47] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang Wang, and Beidi Chen. 2023. H<sub>2</sub>O: Heavy-Hitter Oracle for Eficient Generative Inference of Large Language Models. In Advances in Neural Information Processing Systems, Vol. 36. Curran Associates, Inc., 34661–34710. doi:10.52202/075280-1506

[48] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jef Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, and Ying Sheng. 2024. SGLang: Eficient Execution of Structured Language Model Programs. In Advances in Neural Information Processing Systems, Vol. 37. Curran Associates, Inc., 62557–62583. doi:10.52202/079017-2000

[49] Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao Zhang. 2024. DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving. In 18th USENIXSymposium on Operating Systems Design and Implementation. USENIX Association, 193–210. https://www.usenix.org/conference osdi24/presentation/zhong-yinmin